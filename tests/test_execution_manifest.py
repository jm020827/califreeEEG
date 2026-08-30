from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import pytest

from cfeg.execution_manifest import (
    build_primary_execution_manifest,
    confirmatory_training_grid_sha256,
    file_sha256,
    normalized_primary_fairness_config,
    resolved_config_sha256,
    sha256_json,
    validate_canonical_execution_paths,
    validate_confirmatory_training_artifacts,
    validate_primary_execution_manifest,
    validate_primary_pair_and_hash,
)
from cfeg.utils.config import load_config

REPO = Path(__file__).resolve().parents[1]


def _inputs():
    base = load_config(REPO / "configs/train/wearable_loso.yaml", strict_env=False)
    suite = load_config(REPO / "configs/train/ablation.yaml", strict_env=False)
    plan = load_config(REPO / "configs/analysis/wearable_primary.yaml", strict_env=False)
    return base, suite["variants"], plan


def _bind_execution_paths(plan: dict, tmp_path: Path, *, suffix: str = "") -> tuple[Path, Path]:
    manifest_path = tmp_path / f"execution_manifest{suffix}.json"
    output_root = tmp_path / f"runs{suffix}"
    plan["execution_manifest_path"] = str(manifest_path)
    plan["execution_output_root"] = str(output_root)
    return manifest_path, output_root


def test_fairness_hash_ignores_seed_fold_and_storage_but_not_learning_rate() -> None:
    base, variants, _ = _inputs()
    reference = validate_primary_pair_and_hash(base, variants)
    moved = copy.deepcopy(base)
    moved["seed"] = 44
    moved["data"]["fold_index"] = 4
    moved["run_name"] = "another-run"
    moved["output_dir"] = "/tmp/another-output"
    moved["tracking"]["wandb"]["tags"] = ["fold-4", "seed-44"]

    assert validate_primary_pair_and_hash(moved, variants) == reference

    changed = copy.deepcopy(base)
    changed["train"]["lr"] *= 2
    assert validate_primary_pair_and_hash(changed, variants) != reference


def test_generic_ablation_can_use_nonconfirmatory_backbone_contract() -> None:
    suite = load_config(REPO / "configs/train/ablation.yaml", strict_env=False)
    base = load_config(REPO / suite["base_config"], strict_env=False)

    digest = validate_primary_pair_and_hash(
        base,
        suite["variants"],
        enforce_frozen_architecture=False,
    )

    assert isinstance(digest, str) and len(digest) == 64


def test_primary_pair_rejects_legacy_prompt_architecture() -> None:
    base, variants, _ = _inputs()
    broken = copy.deepcopy(variants)
    for role in ("A0_eeg_only", "A2_structured_condition_prompt"):
        broken[role]["model.conditioning.architecture"] = "prompt_adapter_v1"

    with pytest.raises(ValueError, match="physical_hybrid_v1"):
        validate_primary_pair_and_hash(base, broken)


@pytest.mark.parametrize(
    ("path", "value"),
    [
        (("model", "conditioning", "hidden_dim"), 32),
        (("model", "conditioning", "common_query_film", "max_shift"), 0.5),
        (("model", "conditioning", "external_global_film", "max_scale_delta"), 0.5),
        (("model", "conditioning", "channel_quality", "max_gain_delta"), 0.5),
        (("model", "latent", "enabled"), True),
    ],
)
def test_primary_pair_rejects_joint_drift_from_frozen_physical_contract(
    path: tuple[str, ...], value
) -> None:
    base, variants, _ = _inputs()
    override_key = ".".join(path)
    for role in ("A0_eeg_only", "A2_structured_condition_prompt"):
        variants[role][override_key] = value

    with pytest.raises(ValueError, match="physical_hybrid_v1"):
        validate_primary_pair_and_hash(base, variants)


def test_resolved_hash_retains_coordinates_hidden_by_fairness_normalization() -> None:
    base, _, _ = _inputs()
    moved = copy.deepcopy(base)
    moved["seed"] = 43
    moved["data"]["fold_index"] = 1

    assert normalized_primary_fairness_config(base) == normalized_primary_fairness_config(moved)
    assert resolved_config_sha256(base) != resolved_config_sha256(moved)


def test_primary_manifest_is_exact_six_run_lockbox_cartesian_product(tmp_path) -> None:
    base, variants, plan = _inputs()
    manifest_path, output_root = _bind_execution_paths(plan, tmp_path)
    manifest = build_primary_execution_manifest(
        base=base,
        variants=variants,
        plan=plan,
        output_root=output_root,
        manifest_path=manifest_path,
        plan_sha256=hashlib.sha256(b"plan").hexdigest(),
        source_contract={"source_tree_sha256": "a" * 64},
    )

    assert len(manifest["jobs"]) == 6
    assert manifest["expected_job_count"] == 6
    assert manifest["execution_allowed"] is False
    assert len({job["output_dir"] for job in manifest["jobs"]}) == 6
    assert {job["normalized_fairness_sha256"] for job in manifest["jobs"]} == {
        manifest["primary_fairness_hash"]
    }
    assert {job["config"]["protocol"]["execution_manifest"] for job in manifest["jobs"]} == {
        str(manifest_path)
    }


def test_frozen_plan_must_bind_the_computed_primary_fairness_hash(tmp_path) -> None:
    base, variants, plan = _inputs()
    manifest_path, output_root = _bind_execution_paths(plan, tmp_path)
    plan["status"] = "frozen"
    plan["primary_fairness_hash"] = "c" * 64

    with pytest.raises(ValueError, match="bind the execution fairness hash"):
        build_primary_execution_manifest(
            base=base,
            variants=variants,
            plan=plan,
            output_root=output_root,
            manifest_path=manifest_path,
            plan_sha256=hashlib.sha256(b"plan").hexdigest(),
            source_contract={"source_tree_sha256": "a" * 64},
        )


def test_primary_manifest_rejects_missing_or_mutated_job(tmp_path) -> None:
    base, variants, plan = _inputs()
    manifest_path, output_root = _bind_execution_paths(plan, tmp_path)
    manifest = build_primary_execution_manifest(
        base=base,
        variants=variants,
        plan=plan,
        output_root=output_root,
        manifest_path=manifest_path,
        plan_sha256="b" * 64,
        source_contract={},
    )
    manifest["jobs"].pop()
    with pytest.raises(ValueError, match="Cartesian product"):
        validate_primary_execution_manifest(manifest, plan=plan)

    manifest_path, output_root = _bind_execution_paths(plan, tmp_path, suffix="-2")
    manifest = build_primary_execution_manifest(
        base=base,
        variants=variants,
        plan=plan,
        output_root=output_root,
        manifest_path=manifest_path,
        plan_sha256="c" * 64,
        source_contract={},
    )
    manifest["jobs"][0]["config"]["train"]["lr"] = 1.0
    with pytest.raises(ValueError, match="Resolved config digest mismatch"):
        validate_primary_execution_manifest(manifest, plan=plan)


def test_primary_manifest_rejects_top_level_contract_tampering(tmp_path) -> None:
    base, variants, plan = _inputs()
    manifest_path, output_root = _bind_execution_paths(plan, tmp_path)
    plan_sha = "d" * 64
    manifest = build_primary_execution_manifest(
        base=base,
        variants=variants,
        plan=plan,
        output_root=output_root,
        manifest_path=manifest_path,
        plan_sha256=plan_sha,
        source_contract={"source_tree_sha256": "a" * 64},
    )
    manifest["execution_contract_sha256"] = "0" * 64

    with pytest.raises(ValueError, match="contract digest"):
        validate_primary_execution_manifest(manifest, plan=plan, analysis_plan_sha256=plan_sha)

    manifest_path, output_root = _bind_execution_paths(plan, tmp_path, suffix="-2")
    manifest = build_primary_execution_manifest(
        base=base,
        variants=variants,
        plan=plan,
        output_root=output_root,
        manifest_path=manifest_path,
        plan_sha256=plan_sha,
        source_contract={"source_tree_sha256": "a" * 64},
    )
    with pytest.raises(ValueError, match="analysis-plan digest"):
        validate_primary_execution_manifest(manifest, plan=plan, analysis_plan_sha256="e" * 64)


def test_primary_manifest_rejects_noncanonical_locations_and_job_escape(tmp_path) -> None:
    base, variants, plan = _inputs()
    manifest_path, output_root = _bind_execution_paths(plan, tmp_path)
    with pytest.raises(ValueError, match="output root differs"):
        build_primary_execution_manifest(
            base=base,
            variants=variants,
            plan=plan,
            output_root=tmp_path / "alternate-runs",
            manifest_path=manifest_path,
            plan_sha256="f" * 64,
            source_contract={},
        )
    with pytest.raises(ValueError, match="manifest path differs"):
        validate_canonical_execution_paths(
            plan,
            manifest_path=tmp_path / "copied.json",
            output_root=output_root,
        )

    manifest = build_primary_execution_manifest(
        base=base,
        variants=variants,
        plan=plan,
        output_root=output_root,
        manifest_path=manifest_path,
        plan_sha256="f" * 64,
        source_contract={},
    )
    manifest["jobs"][0]["prediction_csv"] = str(tmp_path / "escaped.csv")
    with pytest.raises(ValueError, match="artifact paths"):
        validate_primary_execution_manifest(manifest, plan=plan)


def test_primary_manifest_rejects_coordinate_and_internal_manifest_tampering(tmp_path) -> None:
    base, variants, plan = _inputs()
    manifest_path, output_root = _bind_execution_paths(plan, tmp_path)
    manifest = build_primary_execution_manifest(
        base=base,
        variants=variants,
        plan=plan,
        output_root=output_root,
        manifest_path=manifest_path,
        plan_sha256="1" * 64,
        source_contract={},
    )
    manifest["jobs"][0]["pair_id"] = "seed999-fold0"
    with pytest.raises(ValueError, match="pair/job identity"):
        validate_primary_execution_manifest(manifest, plan=plan)

    manifest = build_primary_execution_manifest(
        base=base,
        variants=variants,
        plan=plan,
        output_root=output_root,
        manifest_path=manifest_path,
        plan_sha256="1" * 64,
        source_contract={},
    )
    config = manifest["jobs"][0]["config"]
    config["protocol"]["execution_manifest"] = str(tmp_path / "other.json")
    manifest["jobs"][0]["resolved_config_sha256"] = resolved_config_sha256(config)
    with pytest.raises(ValueError, match="role/fairness binding"):
        validate_primary_execution_manifest(manifest, plan=plan)


def test_primary_manifest_recomputes_role_treatment_from_job_config(tmp_path) -> None:
    base, variants, plan = _inputs()
    manifest_path, output_root = _bind_execution_paths(plan, tmp_path)
    manifest = build_primary_execution_manifest(
        base=base,
        variants=variants,
        plan=plan,
        output_root=output_root,
        manifest_path=manifest_path,
        plan_sha256="9" * 64,
        source_contract={},
    )
    job = next(job for job in manifest["jobs"] if job["role"] == "A0_eeg_only")
    job["config"]["model"]["condition_encoder"]["external_metadata_mode"] = "observed"
    job["resolved_config_sha256"] = resolved_config_sha256(job["config"])

    with pytest.raises(ValueError, match="treatment mode mismatch"):
        validate_primary_execution_manifest(manifest, plan=plan)


def test_training_grid_digest_binds_every_completion_artifact(tmp_path) -> None:
    jobs = []
    for index in range(6):
        root = tmp_path / f"job-{index}"
        root.mkdir()
        for name in (
            "training_completion.json",
            "final.pt",
            "last.pt",
            "metrics_train.csv",
            "split.csv",
            "development_control.json",
        ):
            (root / name).write_text(f"{index}:{name}\n", encoding="utf-8")
        jobs.append({"job_id": f"job-{index}", "output_dir": str(root)})
    manifest = {"expected_job_count": 6, "jobs": jobs}
    before = confirmatory_training_grid_sha256(manifest)

    (tmp_path / "job-4" / "metrics_train.csv").write_text("tampered\n", encoding="utf-8")

    assert confirmatory_training_grid_sha256(manifest) != before


def test_confirmatory_completion_requires_fixed_epoch_runtime_and_resume_receipt(
    tmp_path,
) -> None:
    root = tmp_path / "job"
    root.mkdir()
    (root / "final.pt").write_bytes(b"checkpoint")
    (root / "last.pt").write_bytes(b"last checkpoint")
    (root / "metrics_train.csv").write_text("epoch,loss\n10,0.1\n", encoding="utf-8")
    (root / "split.csv").write_text("sample_id,split\na,train\n", encoding="utf-8")
    (root / "development_control.json").write_text(
        '{"name":"none"}\n', encoding="utf-8"
    )
    cfg = {
        "train": {"epochs": 10},
        "protocol": {
            "execution_job_id": "seed42-fold0-A0_eeg_only",
            "execution_contract_sha256": "e" * 64,
        },
        "runtime_contract": {"execution_phase": "confirmatory_training"},
    }
    planned = copy.deepcopy(cfg)
    planned.pop("runtime_contract")
    job = {
        "job_id": "seed42-fold0-A0_eeg_only",
        "output_dir": str(root),
        "resolved_config_sha256": resolved_config_sha256(planned),
        "config": planned,
    }
    manifest = {"execution_contract_sha256": "e" * 64}
    receipt = {
        "schema": "cfeg.training-completion.v1",
        "status": "completed",
        "completed_epoch": 10,
        "stop_reason": "max_epochs",
        "resolved_config_sha256": job["resolved_config_sha256"],
        "runtime_contract_sha256": sha256_json(cfg["runtime_contract"]),
        "resume_generation": 9,
        "resume_state_sha256": "a" * 64,
        "checkpoint_sha256": {
            "final.pt": file_sha256(root / "final.pt"),
            "last.pt": file_sha256(root / "last.pt"),
        },
        "artifact_sha256": {
            "metrics_train.csv": file_sha256(root / "metrics_train.csv"),
            "split.csv": file_sha256(root / "split.csv"),
            "development_control.json": file_sha256(root / "development_control.json"),
        },
    }
    (root / "training_completion.json").write_text(json.dumps(receipt), encoding="utf-8")
    checkpoint = {
        "config": cfg,
        "checkpoint_role": "confirmatory_fixed_epoch",
        "selection_split": None,
        "selection_metric": "fixed_epoch",
        "epoch": 10,
    }

    validate_confirmatory_training_artifacts(job, manifest, checkpoint)

    receipt["completed_epoch"] = 9
    (root / "training_completion.json").write_text(json.dumps(receipt), encoding="utf-8")
    with pytest.raises(ValueError, match="completed_epoch"):
        validate_confirmatory_training_artifacts(job, manifest, checkpoint)


def test_canonical_manifest_publish_is_exclusive_and_only_replaces_unused_dev_draft(
    tmp_path,
    monkeypatch,
) -> None:
    monkeypatch.syspath_prepend(str(REPO / "scripts"))
    from build_primary_run_manifest import _publish_manifest

    path = tmp_path / "confirmatory" / "execution_manifest.json"
    plan = {
        "execution_manifest_path": str(path),
        "execution_output_root": str(tmp_path / "confirmatory" / "runs"),
    }

    def draft(marker: str) -> dict:
        value = {
            "schema": "cfeg.primary-execution-manifest.v1",
            "plan_status": "dev_not_frozen",
            "execution_allowed": False,
            "expected_job_count": 6,
            "jobs": [{"job_id": f"job-{index}"} for index in range(6)],
            "marker": marker,
        }
        value["manifest_content_sha256"] = sha256_json(value)
        return value

    _publish_manifest(
        path,
        draft("first"),
        plan=plan,
        replace_unexecuted_draft=False,
    )
    with pytest.raises(FileExistsError, match="already exists"):
        _publish_manifest(
            path,
            draft("second"),
            plan=plan,
            replace_unexecuted_draft=False,
        )

    _publish_manifest(
        path,
        draft("second"),
        plan=plan,
        replace_unexecuted_draft=True,
    )
    assert json.loads(path.read_text(encoding="utf-8"))["marker"] == "second"

    (path.parent / "lockbox_reveal_receipt.json").write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="artifacts appeared"):
        _publish_manifest(
            path,
            draft("third"),
            plan=plan,
            replace_unexecuted_draft=True,
        )
