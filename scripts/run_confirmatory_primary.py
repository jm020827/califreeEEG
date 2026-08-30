#!/usr/bin/env python
from __future__ import annotations

import argparse
import copy
import datetime as dt
import hashlib
import json
import os
from pathlib import Path

import torch
from _bootstrap import add_src_to_path

add_src_to_path()

from cfeg.execution_manifest import (
    confirmatory_training_grid_sha256,
    sha256_json,
    validate_confirmatory_training_artifacts,
    validate_primary_execution_manifest,
)
from cfeg.governance import current_source_revision_contract, validate_frozen_analysis_plan
from cfeg.prediction import load_verified_prediction_bundle, run_prediction
from cfeg.train_loop import run_training
from cfeg.utils.checkpoint import save_json
from cfeg.utils.config import load_config


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Execute the manifest-bound six-run independent-lockbox primary grid."
    )
    parser.add_argument("command", choices=("status", "train", "predict"))
    parser.add_argument(
        "--manifest", default="outputs/confirmatory-primary/execution_manifest.json"
    )
    parser.add_argument("--plan", default="configs/analysis/wearable_primary.yaml")
    args = parser.parse_args()

    plan = load_config(args.plan, strict_env=False)
    manifest_path = Path(args.manifest)
    manifest = _load_manifest(manifest_path, plan, plan_path=Path(args.plan))
    if args.command == "status":
        _print_status(manifest, manifest_path.parent)
        return

    # This gate runs before a dataset or checkpoint is opened.
    validate_frozen_analysis_plan(plan)
    if manifest.get("execution_allowed") is not True or manifest.get("plan_status") != "frozen":
        raise ValueError("The execution manifest was not generated from the frozen plan.")
    if manifest.get("source_contract") != current_source_revision_contract():
        raise ValueError("Current source differs from the frozen execution manifest.")

    if args.command == "train":
        for job in manifest["jobs"]:
            completion = Path(job["output_dir"]) / "training_completion.json"
            if completion.is_file():
                continue
            run_training(copy.deepcopy(job["config"]), resume_exact=True)
        _validate_training_complete(manifest)
        _print_status(manifest, manifest_path.parent)
        return

    _validate_training_complete(manifest)
    _register_lockbox_reveal(manifest_path.parent, manifest)
    processed_dir = _resolve_processed_dir(manifest["jobs"][0]["config"])
    _predict_lockbox_grid_atomically(
        manifest_path=manifest_path,
        manifest=manifest,
        processed_dir=processed_dir,
    )
    _print_status(manifest, manifest_path.parent)


def _load_manifest(path: Path, plan: dict, *, plan_path: Path) -> dict:
    manifest = json.loads(path.read_text(encoding="utf-8"))
    recorded = manifest.pop("manifest_content_sha256", None)
    if recorded != sha256_json(manifest):
        raise ValueError("Execution manifest content changed after generation.")
    validate_primary_execution_manifest(
        manifest,
        plan=plan,
        analysis_plan_sha256=_sha256_file(plan_path),
        manifest_path=path,
    )
    manifest["manifest_content_sha256"] = recorded
    return manifest


def _validate_training_complete(manifest: dict) -> None:
    incomplete = []
    pair_records: list[dict] = []
    for job in manifest["jobs"]:
        root = Path(job["output_dir"])
        required = [
            root / "training_completion.json",
            root / "final.pt",
            root / "metrics_train.csv",
            root / "split.csv",
        ]
        if any(not path.is_file() for path in required) or (root / "metrics_val.csv").exists():
            incomplete.append(job["job_id"])
            continue
        checkpoint = torch.load(
            root / "final.pt", map_location="cpu", weights_only=False
        )
        validate_confirmatory_training_artifacts(job, manifest, checkpoint)
        pair_records.append({"job": job, "checkpoint": checkpoint})
    if incomplete:
        raise ValueError(f"Confirmatory training grid is incomplete: {incomplete}.")
    _validate_pair_wide_training_fairness(pair_records, manifest)


def _validate_pair_wide_training_fairness(records: list[dict], manifest: dict) -> None:
    """Reject an unfair A0/A2 training grid before any lockbox reveal is registered."""

    expected_roles = {"A0_eeg_only", "A2_structured_condition_prompt"}
    expected_keys = {
        (int(job["seed"]), int(job["fold_index"]), str(job["role"]))
        for job in manifest["jobs"]
    }
    observed_keys = {
        (
            int(record["job"]["seed"]),
            int(record["job"]["fold_index"]),
            str(record["job"]["role"]),
        )
        for record in records
    }
    if len(records) != len(manifest["jobs"]) or observed_keys != expected_keys:
        raise ValueError("Confirmatory fairness preflight lacks the exact training grid.")

    groups: dict[tuple[int, int], dict[str, dict]] = {}
    for record in records:
        job = record["job"]
        groups.setdefault((int(job["seed"]), int(job["fold_index"])), {})[
            str(job["role"])
        ] = record
    pair_equal_runtime_fields = (
        "parameter_schema_sha256",
        "initial_trainable_state_sha256",
        "split_assignment_sha256",
        "vocabulary_sha256",
        "asset_provenance_sha256",
        "execution_phase",
        "analysis_plan_sha256",
        "cohort_roles_sha256",
        "cohort_sha256",
        "outer_test_access_during_training",
        "source_commit_sha",
        "source_dirty",
        "source_tree_sha256",
        "environment_sha256",
        "development_control",
        "development_control_sha256",
        "loader_settings_sha256",
    )
    initial_states_by_seed: dict[int, str] = {}
    for key, roles in groups.items():
        if set(roles) != expected_roles:
            raise ValueError(f"Confirmatory seed/fold {key} lacks one A0/A2 role.")
        baseline = roles["A0_eeg_only"]
        candidate = roles["A2_structured_condition_prompt"]
        baseline_cfg = baseline["checkpoint"].get("config") or {}
        candidate_cfg = candidate["checkpoint"].get("config") or {}
        baseline_runtime = baseline_cfg.get("runtime_contract") or {}
        candidate_runtime = candidate_cfg.get("runtime_contract") or {}
        mismatched = [
            field
            for field in pair_equal_runtime_fields
            if baseline_runtime.get(field) != candidate_runtime.get(field)
        ]
        if mismatched:
            raise ValueError(
                f"Confirmatory A0/A2 training fairness differs before reveal: {mismatched}."
            )
        baseline_mode = (
            baseline_cfg.get("model", {})
            .get("condition_encoder", {})
            .get("external_metadata_mode")
        )
        candidate_mode = (
            candidate_cfg.get("model", {})
            .get("condition_encoder", {})
            .get("external_metadata_mode")
        )
        if baseline_mode != "null" or candidate_mode != "observed":
            raise ValueError("Confirmatory A0/A2 treatment access is not null versus observed.")
        if (
            baseline_cfg.get("protocol", {}).get("primary_fairness_hash")
            != manifest["primary_fairness_hash"]
            or candidate_cfg.get("protocol", {}).get("primary_fairness_hash")
            != manifest["primary_fairness_hash"]
            or baseline_runtime.get("execution_phase") != "confirmatory_training"
            or bool(baseline_runtime.get("outer_test_access_during_training"))
            or bool(baseline_runtime.get("source_dirty"))
        ):
            raise ValueError("Confirmatory A0/A2 pre-reveal governance contract is invalid.")
        baseline_root = Path(baseline["job"]["output_dir"])
        candidate_root = Path(candidate["job"]["output_dir"])
        if _sha256_file(baseline_root / "split.csv") != _sha256_file(
            candidate_root / "split.csv"
        ):
            raise ValueError("Confirmatory A0/A2 split manifests differ before reveal.")
        if _sha256_file(baseline_root / "final.pt") == _sha256_file(
            candidate_root / "final.pt"
        ):
            raise ValueError("Confirmatory A0/A2 unexpectedly use an identical checkpoint.")
        seed = int(key[0])
        initial_state = str(baseline_runtime.get("initial_trainable_state_sha256") or "")
        prior = initial_states_by_seed.setdefault(seed, initial_state)
        if prior != initial_state:
            raise ValueError(f"Confirmatory seed {seed} has inconsistent initial model state.")
    if len(set(initial_states_by_seed.values())) != len(initial_states_by_seed):
        raise ValueError("Distinct confirmatory seeds produced identical initial model states.")


def _register_lockbox_reveal(root: Path, manifest: dict) -> None:
    path = root / "lockbox_reveal_receipt.json"
    expected = {
        "schema": "cfeg.confirmatory-lockbox-reveal.v1",
        "execution_contract_sha256": manifest["execution_contract_sha256"],
        "execution_manifest_content_sha256": manifest["manifest_content_sha256"],
        "training_grid_sha256": confirmatory_training_grid_sha256(manifest),
        "scope": "all-six-seed-role-lockbox-predictions",
        "confirmatory_reveal_count": 1,
    }
    if path.exists():
        observed = json.loads(path.read_text(encoding="utf-8"))
        if any(observed.get(key) != value for key, value in expected.items()):
            raise ValueError("A different confirmatory lockbox reveal is already registered.")
        return
    save_json(
        path,
        {
            **expected,
            "registered_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        },
    )


def _predict_lockbox_grid_atomically(
    *,
    manifest_path: Path,
    manifest: dict,
    processed_dir: str,
) -> None:
    final_paths = [Path(job["prediction_csv"]) for job in manifest["jobs"]]
    final_parents = {path.parent.resolve() for path in final_paths}
    if len(final_parents) != 1:
        raise ValueError("Confirmatory predictions must share one atomic publication directory.")
    final_dir = next(iter(final_parents))
    complete = [
        path.is_file()
        and path.with_name(f"{path.stem}_provenance.json").is_file()
        for path in final_paths
    ]
    if all(complete):
        _validate_prediction_directory(final_dir, manifest["jobs"])
        for job in manifest["jobs"]:
            _validate_prediction_bundle(
                Path(job["prediction_csv"]),
                expected_recorded_path=Path(job["prediction_csv"]),
                job=job,
                manifest=manifest,
                receipt_path=manifest_path.parent / "lockbox_reveal_receipt.json",
            )
        return
    if any(complete) or final_dir.exists():
        raise ValueError(
            "Confirmatory prediction publication is partial; do not expose or overwrite it."
        )

    staging = final_dir.with_name(
        f".{final_dir.name}.{manifest['execution_contract_sha256']}.staging"
    )
    staging.mkdir(parents=True, exist_ok=True, mode=0o700)
    staging.chmod(0o700)
    authorization = {
        "execution_manifest": str(manifest_path),
        "lockbox_reveal_receipt": str(
            manifest_path.parent / "lockbox_reveal_receipt.json"
        ),
    }
    for job in manifest["jobs"]:
        final_path = Path(job["prediction_csv"])
        staged_path = staging / final_path.name
        staged_provenance = staged_path.with_name(
            f"{staged_path.stem}_provenance.json"
        )
        if staged_path.is_file() != staged_provenance.is_file():
            raise ValueError("Confirmatory staging contains a partial prediction bundle.")
        if not staged_path.is_file():
            run_prediction(
                job["checkpoint_path"],
                processed_dir,
                staged_path,
                split="test",
                scenario="clean",
                confirmatory_authorization=authorization,
            )
        artifact = json.loads(staged_provenance.read_text(encoding="utf-8"))
        recorded = Path(str(artifact.get("prediction_csv", ""))).expanduser().resolve()
        allowed_recorded = {staged_path.resolve(), final_path.resolve()}
        if recorded not in allowed_recorded:
            raise ValueError("Confirmatory staging sidecar records an unexpected CSV path.")
        _validate_prediction_bundle(
            staged_path,
            expected_recorded_path=recorded,
            job=job,
            manifest=manifest,
            receipt_path=manifest_path.parent / "lockbox_reveal_receipt.json",
        )
        if recorded != final_path.resolve():
            artifact["prediction_csv"] = str(final_path.resolve())
            save_json(staged_provenance, artifact)
        _validate_prediction_bundle(
            staged_path,
            expected_recorded_path=final_path,
            job=job,
            manifest=manifest,
            receipt_path=manifest_path.parent / "lockbox_reveal_receipt.json",
        )

    _validate_prediction_directory(staging, manifest["jobs"])
    final_dir.parent.mkdir(parents=True, exist_ok=True)
    os.replace(staging, final_dir)
    _validate_prediction_directory(final_dir, manifest["jobs"])
    for job in manifest["jobs"]:
        _validate_prediction_bundle(
            Path(job["prediction_csv"]),
            expected_recorded_path=Path(job["prediction_csv"]),
            job=job,
            manifest=manifest,
            receipt_path=manifest_path.parent / "lockbox_reveal_receipt.json",
        )


def _validate_prediction_directory(directory: Path, jobs: list[dict]) -> None:
    expected = {
        name
        for job in jobs
        for name in (
            Path(job["prediction_csv"]).name,
            f"{Path(job['prediction_csv']).stem}_provenance.json",
        )
    }
    observed = {path.name for path in directory.iterdir() if path.is_file()}
    if observed != expected:
        raise ValueError(
            "Confirmatory prediction directory is not the exact manifest grid: "
            f"missing={sorted(expected - observed)}, unexpected={sorted(observed - expected)}."
        )


def _validate_prediction_bundle(
    actual_path: Path,
    *,
    expected_recorded_path: Path,
    job: dict,
    manifest: dict,
    receipt_path: Path,
) -> None:
    bundle = load_verified_prediction_bundle(
        actual_path,
        expected_recorded_csv_path=expected_recorded_path,
    )
    artifact = bundle.artifact
    contract = artifact.get("primary_contract") or {}
    expected_contract = {
        "execution_job_id": job["job_id"],
        "execution_contract_sha256": manifest["execution_contract_sha256"],
        "execution_manifest_content_sha256": manifest["manifest_content_sha256"],
        "lockbox_reveal_receipt_sha256": _sha256_file(receipt_path),
        "planned_config_sha256": job["resolved_config_sha256"],
        "training_completion_sha256": _sha256_file(
            Path(job["output_dir"]) / "training_completion.json"
        ),
        "primary_ablation_role": job["role"],
        "primary_fairness_hash": manifest["primary_fairness_hash"],
        "optimization_seed": job["seed"],
        "fold_index": job["fold_index"],
    }
    mismatched = {
        key: {"expected": value, "observed": contract.get(key)}
        for key, value in expected_contract.items()
        if str(contract.get(key)) != str(value)
    }
    checkpoint_path = Path(str(artifact.get("checkpoint_path", ""))).expanduser().resolve()
    if checkpoint_path != Path(job["checkpoint_path"]).expanduser().resolve():
        mismatched["checkpoint_path"] = {
            "expected": str(Path(job["checkpoint_path"]).expanduser().resolve()),
            "observed": str(checkpoint_path),
        }
    if artifact.get("scenario") != "clean" or set(bundle.frame["scenario"].astype(str)) != {
        "clean"
    }:
        mismatched["scenario"] = {"expected": "clean", "observed": artifact.get("scenario")}
    if artifact.get("selection_split") != "test" or set(
        bundle.frame["selection_split"].astype(str)
    ) != {"test"}:
        mismatched["selection_split"] = {
            "expected": "test",
            "observed": artifact.get("selection_split"),
        }
    if mismatched:
        raise ValueError(
            f"Confirmatory prediction bundle differs from job {job['job_id']}: {mismatched}."
        )


def _resolve_processed_dir(cfg: dict) -> str:
    value = os.path.expandvars(str(cfg["data"]["processed_dirs"][0]))
    if "${env:" in value or not Path(value).is_dir():
        raise ValueError("EEG_DATA_ROOT must resolve the audited wearable_v3 asset.")
    return value


def _sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _print_status(manifest: dict, root: Path) -> None:
    trained = sum(
        (Path(job["output_dir"]) / "training_completion.json").is_file()
        for job in manifest["jobs"]
    )
    predicted = sum(Path(job["prediction_csv"]).is_file() for job in manifest["jobs"])
    print(
        json.dumps(
            {
                "plan_status": manifest["plan_status"],
                "execution_allowed": manifest["execution_allowed"],
                "expected_jobs": len(manifest["jobs"]),
                "trained": trained,
                "predicted": predicted,
                "lockbox_revealed": (root / "lockbox_reveal_receipt.json").is_file(),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
