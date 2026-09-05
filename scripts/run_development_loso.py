#!/usr/bin/env python
from __future__ import annotations

import argparse
import copy
import datetime as dt
import json
import os
from pathlib import Path

import pandas as pd
import torch
from _bootstrap import add_src_to_path

add_src_to_path()

from cfeg.analysis.development_loso import aggregate_complete_development_loso
from cfeg.execution_manifest import (
    PRIMARY_ROLES,
    resolved_config_sha256,
    sha256_json,
    validate_primary_pair_and_hash,
)
from cfeg.governance import (
    current_source_revision_contract,
    reject_retired_physical_candidate_action,
    validate_physical_candidate_retirement,
)
from cfeg.prediction import run_prediction
from cfeg.train_loop import _resolve_augmentation_channel_sets, run_training
from cfeg.utils.checkpoint import save_json
from cfeg.utils.config import load_config, merge_overrides


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the outcome-gated S1-S3 fixed-epoch A0/A2 LOSO grid."
    )
    parser.add_argument("command", choices=("prepare", "status", "train", "predict", "aggregate"))
    parser.add_argument(
        "--config", default="configs/train/wearable_development_loso.yaml"
    )
    parser.add_argument("--ablation-config", default="configs/train/ablation.yaml")
    parser.add_argument("--root", default="outputs/development-loso/fixed-epoch-v1")
    args = parser.parse_args()
    validate_physical_candidate_retirement()
    if args.command != "status":
        reject_retired_physical_candidate_action()
    root = Path(args.root)
    manifest_path = root / "grid_manifest.json"

    if args.command == "prepare":
        manifest = _build_grid_manifest(args.config, args.ablation_config, root)
        if root.exists() and any(path.name != "grid_manifest.json" for path in root.iterdir()):
            raise ValueError("Cannot replace a development manifest beside existing run artifacts.")
        save_json(manifest_path, manifest)
        _print_status(manifest, root)
        return

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    _validate_grid_manifest(manifest)
    if args.command == "status":
        _print_status(manifest, root)
        return
    if manifest["design_status"] != "frozen":
        raise ValueError(
            "Development learning rule is not frozen. Complete source-only learning/profile "
            "checks, then explicitly freeze the design before any S1-S3 outcome reveal."
        )
    if current_source_revision_contract() != manifest["source_contract"]:
        raise ValueError("Current source differs from the prepared development grid manifest.")

    if args.command == "train":
        for job in manifest["jobs"]:
            completion = Path(job["output_dir"]) / "training_completion.json"
            if completion.is_file():
                continue
            run_training(copy.deepcopy(job["config"]), resume_exact=True)
        _validate_training_complete(manifest)
        _print_status(manifest, root)
        return

    _validate_training_complete(manifest)
    if args.command == "predict":
        _register_reveal(root, manifest)
        processed_dir = _resolve_processed_dir(manifest["jobs"][0]["config"])
        for job in manifest["jobs"]:
            output = Path(job["prediction_csv"])
            if output.is_file() and output.with_name(
                f"{output.stem}_provenance.json"
            ).is_file():
                continue
            run_prediction(
                Path(job["output_dir"]) / "final.pt",
                processed_dir,
                output,
                split="val",
                scenario="development_fixed_epoch_clean",
            )
        _print_status(manifest, root)
        return

    _validate_predictions_complete(manifest)
    full_manifest = pd.read_parquet(
        Path(_resolve_processed_dir(manifest["jobs"][0]["config"])) / "manifest.parquet"
    )
    subjects, summary = aggregate_complete_development_loso(
        manifest["jobs"],
        full_manifest,
        expected_subjects=["sub001", "sub002", "sub003"],
        expected_n_samples=720,
    )
    subjects_path = root / "revealed_subjects.csv"
    summary_path = root / "revealed_summary.json"
    if subjects_path.exists() or summary_path.exists():
        raise ValueError("Development aggregate already exists; result-driven overwrite is blocked.")
    subjects.to_csv(subjects_path, index=False)
    summary.update(
        {
            "grid_manifest_sha256": sha256_json(manifest),
            "reveal_ledger": str((root / "reveal_ledger.json").resolve()),
        }
    )
    save_json(summary_path, summary)
    print(json.dumps({"status": summary["status"], "summary": str(summary_path)}))


def _build_grid_manifest(config_path: str, ablation_path: str, root: Path) -> dict:
    base = load_config(config_path, strict_env=False)
    design = base.get("development_grid") or {}
    if design.get("schema") != "cfeg.development-loso-design.v1":
        raise ValueError("Unknown development LOSO design schema.")
    if design.get("grid_id") == "wearable-v3-s1-s3-a0-a2-fixed-epoch-v1":
        raise ValueError(
            "The first S1-S3 prompt grid is completed and immutable. Use its existing "
            "manifest for status/history; preparing it from the mutable shared ablation "
            "suite could mislabel a new architecture under the historical grid ID."
        )
    variants = load_config(ablation_path, strict_env=False)["variants"]
    fairness = validate_primary_pair_and_hash(base, variants)
    jobs = []
    for fold in design["folds"]:
        for role in PRIMARY_ROLES:
            cfg = merge_overrides(
                copy.deepcopy(base),
                [f"{key}={value!r}" for key, value in variants[role].items()],
            )
            cfg["seed"] = 42
            cfg["data"]["fold_index"] = int(fold)
            cfg["run_name"] = f"development-loso-fold{fold}-{role}"
            cfg["output_dir"] = str(root / "runs" / f"fold{fold}" / role)
            cfg["protocol"]["primary_ablation_role"] = role
            cfg["protocol"]["primary_fairness_hash"] = fairness
            _resolve_augmentation_channel_sets(cfg)
            jobs.append(
                {
                    "job_id": f"fold{fold}-{role}",
                    "seed": 42,
                    "fold_index": int(fold),
                    "role": role,
                    "output_dir": cfg["output_dir"],
                    "prediction_csv": str(
                        root / "predictions" / f"fold{fold}-{role}.csv"
                    ),
                    "resolved_config_sha256": resolved_config_sha256(cfg),
                    "config": cfg,
                }
            )
    manifest = {
        "schema": "cfeg.development-loso-grid.v1",
        "grid_id": design["grid_id"],
        "design_status": design["status"],
        "interpretation": design["interpretation"],
        "recommended_max_outcome_reveals": int(
            design["recommended_max_outcome_reveals"]
        ),
        "absolute_max_outcome_reveals": int(design["absolute_max_outcome_reveals"]),
        "primary_fairness_hash": fairness,
        "source_contract": current_source_revision_contract(),
        "jobs": jobs,
    }
    manifest["grid_content_sha256"] = sha256_json(manifest)
    _validate_grid_manifest(manifest)
    return manifest


def _validate_grid_manifest(manifest: dict) -> None:
    recorded = manifest.get("grid_content_sha256")
    payload = {key: value for key, value in manifest.items() if key != "grid_content_sha256"}
    if recorded != sha256_json(payload):
        raise ValueError("Development grid manifest changed after preparation.")
    keys = {
        (int(job["seed"]), int(job["fold_index"]), str(job["role"]))
        for job in manifest.get("jobs", [])
    }
    expected = {(42, fold, role) for fold in range(3) for role in PRIMARY_ROLES}
    if len(manifest.get("jobs", [])) != 6 or keys != expected:
        raise ValueError("Development grid is not the exact six-job Cartesian product.")
    for job in manifest["jobs"]:
        if resolved_config_sha256(job["config"]) != job["resolved_config_sha256"]:
            raise ValueError(f"Development job config changed: {job['job_id']}.")


def _validate_training_complete(manifest: dict) -> None:
    incomplete = []
    for job in manifest["jobs"]:
        root = Path(job["output_dir"])
        required = [root / "training_completion.json", root / "final.pt", root / "metrics_train.csv"]
        if any(not path.is_file() for path in required) or (root / "metrics_val.csv").exists():
            incomplete.append(job["job_id"])
            continue
        checkpoint = torch.load(
            root / "final.pt", map_location="cpu", weights_only=False
        )
        cfg = checkpoint.get("config") or {}
        if (
            checkpoint.get("checkpoint_role") != "development_fixed_epoch"
            or checkpoint.get("selection_split") is not None
            or checkpoint.get("selection_metric") != "fixed_epoch"
            or cfg.get("protocol", {}).get("primary_ablation_role") != job["role"]
            or int(cfg.get("data", {}).get("fold_index", -1)) != int(job["fold_index"])
        ):
            raise ValueError(f"Development final checkpoint contract failed: {job['job_id']}.")
    if incomplete:
        raise ValueError(f"Development training grid is incomplete: {incomplete}.")


def _validate_predictions_complete(manifest: dict) -> None:
    missing = []
    for job in manifest["jobs"]:
        path = Path(job["prediction_csv"])
        if not path.is_file() or not path.with_name(f"{path.stem}_provenance.json").is_file():
            missing.append(job["job_id"])
    if missing:
        raise ValueError(f"Development prediction grid is incomplete: {missing}.")


def _register_reveal(root: Path, manifest: dict) -> None:
    path = root / "reveal_ledger.json"
    ledger = (
        json.loads(path.read_text(encoding="utf-8"))
        if path.exists()
        else {"schema": "cfeg.development-reveal-ledger.v1", "entries": []}
    )
    grid_hash = manifest["grid_content_sha256"]
    if any(entry.get("grid_content_sha256") == grid_hash for entry in ledger["entries"]):
        return
    if len(ledger["entries"]) >= int(manifest["absolute_max_outcome_reveals"]):
        raise ValueError("Absolute S1-S3 outcome-reveal budget is exhausted.")
    ledger["entries"].append(
        {
            "reveal_index": len(ledger["entries"]) + 1,
            "grid_content_sha256": grid_hash,
            "grid_id": manifest["grid_id"],
            "registered_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
            "scope": "all-six-fixed-epoch-validation-predictions",
            "confirmatory_claim_allowed": False,
        }
    )
    save_json(path, ledger)


def _resolve_processed_dir(cfg: dict) -> str:
    value = os.path.expandvars(str(cfg["data"]["processed_dirs"][0]))
    if "${env:" in value or not Path(value).is_dir():
        raise ValueError("EEG_DATA_ROOT must resolve the audited wearable_v3 processed asset.")
    return value


def _print_status(manifest: dict, root: Path) -> None:
    trained = sum(
        (Path(job["output_dir"]) / "training_completion.json").is_file()
        for job in manifest["jobs"]
    )
    predicted = sum(Path(job["prediction_csv"]).is_file() for job in manifest["jobs"])
    print(
        json.dumps(
            {
                "design_status": manifest["design_status"],
                "trained": trained,
                "expected_training_jobs": 6,
                "predicted": predicted,
                "outcomes_revealed": (root / "reveal_ledger.json").is_file(),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
