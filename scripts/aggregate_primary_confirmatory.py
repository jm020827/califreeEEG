#!/usr/bin/env python
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import yaml
from _bootstrap import add_src_to_path

add_src_to_path()

from cfeg.analysis.primary_aggregate import aggregate_primary_confirmatory
from cfeg.data.schema import load_manifest
from cfeg.execution_manifest import (
    confirmatory_training_grid_sha256,
    sha256_json,
    validate_primary_execution_manifest,
)
from cfeg.governance import validate_frozen_analysis_plan
from cfeg.prediction import load_verified_prediction_bundle
from cfeg.train_loop import _environment_contract, _sha256_json, _source_revision_contract


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Aggregate the frozen six-run A0/A2 independent-lockbox grid."
    )
    parser.add_argument("--execution-manifest", required=True)
    parser.add_argument("--processed-dir", required=True)
    parser.add_argument("--plan", default="configs/analysis/wearable_primary.yaml")
    parser.add_argument("--out-prefix", required=True)
    args = parser.parse_args()

    plan_path = Path(args.plan).expanduser().resolve()
    plan = yaml.safe_load(plan_path.read_text(encoding="utf-8")) or {}
    _validate_plan(plan)
    execution_manifest = _load_execution_manifest(
        args.execution_manifest, plan, analysis_plan_sha256=_file_sha256(plan_path)
    )
    jobs = {
        (int(job["seed"]), int(job["fold_index"]), str(job["role"])): job
        for job in execution_manifest["jobs"]
    }
    bundles = []
    for seed in plan["optimization_seeds"]:
        for fold in plan["fold_indices"]:
            baseline = jobs[(int(seed), int(fold), "A0_eeg_only")]["prediction_csv"]
            candidate = jobs[
                (int(seed), int(fold), "A2_structured_condition_prompt")
            ]["prediction_csv"]
            bundles.append(
                (
                    load_verified_prediction_bundle(baseline),
                    load_verified_prediction_bundle(candidate),
                )
            )
    outputs = aggregate_primary_confirmatory(
        bundles,
        load_manifest(args.processed_dir),
        expected_seeds=list(plan["optimization_seeds"]),
        expected_n_folds=len(plan["fold_indices"]),
        expected_n_samples=int(plan["expected_n_samples"]),
        expected_n_subjects=int(plan["expected_n_subjects"]),
        expected_n_labels=int(plan["expected_n_labels"]),
        expected_dataset_id=str(plan["dataset_id"]),
        expected_split_seed=int(plan["split_seed"]),
        condition_column=str(plan["condition_column"]),
        expected_conditions=list(plan["expected_conditions"]),
        inference_seed=int(plan["inference_seed"]),
        n_resamples=int(plan["n_resamples"]),
        inference_alpha=float(plan["inference"]["alpha"]),
        inference_alternative=str(plan["inference"]["alternative"]),
        success_threshold=plan["success_threshold"],
        sesoi_balanced_accuracy=plan["sesoi_balanced_accuracy"],
        primary_excluded_subject_ids=list(plan["primary_excluded_subject_ids"]),
        primary_included_subject_ids=list(plan["primary_included_subject_ids"]),
        execution_manifest=execution_manifest,
    )
    runs, seed_subjects, subjects, condition_subjects, summary = outputs
    plan_sha256 = _file_sha256(plan_path)
    roles_path = Path(str(plan["cohort_roles_config"]))
    if not roles_path.is_absolute():
        roles_path = (Path(__file__).resolve().parents[1] / roles_path).resolve()
    roles_sha256 = _file_sha256(roles_path)
    expected_cohort_sha256 = _expected_cohort_sha256(
        load_manifest(args.processed_dir), plan, roles_sha256
    )
    current_source = _source_revision_contract()
    expected_runtime = {
        "analysis_plan_sha256": plan_sha256,
        "cohort_roles_sha256": roles_sha256,
        "cohort_sha256": expected_cohort_sha256,
        "source_commit_sha": current_source["source_commit_sha"],
        "source_dirty": current_source["source_dirty"],
        "source_tree_sha256": current_source["source_tree_sha256"],
    }
    mismatched_runtime = {
        key: {"expected": value, "observed": summary.get(key)}
        for key, value in expected_runtime.items()
        if summary.get(key) != value
    }
    if mismatched_runtime:
        raise ValueError(
            "Prediction bundles do not match the current frozen execution contract: "
            f"{mismatched_runtime}."
        )
    prefix = Path(args.out_prefix).expanduser()
    prefix.parent.mkdir(parents=True, exist_ok=True)
    paths = {
        "runs": prefix.with_name(f"{prefix.name}_runs.csv"),
        "subject_seed": prefix.with_name(f"{prefix.name}_subject_seed.csv"),
        "subjects": prefix.with_name(f"{prefix.name}_subjects.csv"),
        "condition_subjects": prefix.with_name(f"{prefix.name}_condition_subjects.csv"),
    }
    for table, path in zip((runs, seed_subjects, subjects, condition_subjects), paths.values()):
        table.to_csv(path, index=False)
    analysis_environment = _environment_contract()
    summary.update(
        {
            "analysis_plan": str(plan_path),
            "analysis_plan_sha256": plan_sha256,
            "analysis_plan_status": plan["status"],
            "confirmatory_claim_allowed": plan["status"] == "frozen",
            "analysis_source_revision": _source_revision_contract(),
            "analysis_environment": analysis_environment,
            "analysis_environment_sha256": _sha256_json(analysis_environment),
            "output_sha256": {name: _file_sha256(path) for name, path in paths.items()},
            "execution_manifest": str(Path(args.execution_manifest).expanduser().resolve()),
            "execution_manifest_sha256": _file_sha256(args.execution_manifest),
        }
    )
    summary_path = prefix.with_name(f"{prefix.name}_summary.json")
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(summary_path)


def _validate_plan(plan: dict) -> None:
    validate_frozen_analysis_plan(plan)


def _load_execution_manifest(
    path: str | Path,
    plan: dict,
    *,
    analysis_plan_sha256: str,
) -> dict:
    manifest_path = Path(path).expanduser().resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    recorded = manifest.pop("manifest_content_sha256", None)
    if sha256_json(manifest) != recorded:
        raise ValueError("Execution manifest content changed after generation.")
    validate_primary_execution_manifest(
        manifest,
        plan=plan,
        analysis_plan_sha256=analysis_plan_sha256,
        manifest_path=manifest_path,
    )
    if manifest.get("execution_allowed") is not True or manifest.get("plan_status") != "frozen":
        raise ValueError("Execution manifest is not authorized for confirmatory aggregation.")
    current_source = _source_revision_contract()
    if manifest.get("source_contract") != current_source:
        raise ValueError("Execution manifest source contract differs from current source.")
    manifest["manifest_content_sha256"] = recorded
    receipt_path = manifest_path.parent / "lockbox_reveal_receipt.json"
    if not receipt_path.is_file():
        raise ValueError("Confirmatory lockbox reveal receipt is missing.")
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    expected_receipt = {
        "schema": "cfeg.confirmatory-lockbox-reveal.v1",
        "execution_contract_sha256": manifest["execution_contract_sha256"],
        "execution_manifest_content_sha256": recorded,
        "training_grid_sha256": confirmatory_training_grid_sha256(manifest),
        "scope": "all-six-seed-role-lockbox-predictions",
        "confirmatory_reveal_count": 1,
    }
    if any(receipt.get(key) != value for key, value in expected_receipt.items()):
        raise ValueError("Confirmatory lockbox reveal receipt is invalid or stale.")
    manifest["lockbox_reveal_receipt_sha256"] = _file_sha256(receipt_path)
    return manifest


def _expected_cohort_sha256(manifest, plan: dict, roles_sha256: str) -> str:
    excluded = {str(value) for value in plan["primary_excluded_subject_ids"]}
    subjects = manifest["subject_id"].astype(str)
    selected = manifest.loc[~subjects.isin(excluded)]
    payload = {
        "decision_id": plan["cohort_decision_id"],
        "role": "confirmatory_primary",
        "role_manifest_sha256": roles_sha256,
        "selected_subject_ids": sorted(selected["subject_id"].astype(str).unique()),
        "excluded_subject_ids": sorted(excluded),
        "sample_ids": sorted(selected["sample_id"].astype(str).tolist()),
    }
    return _sha256_json(payload)


def _file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


if __name__ == "__main__":
    main()
