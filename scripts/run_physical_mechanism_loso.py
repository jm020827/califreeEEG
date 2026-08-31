#!/usr/bin/env python
from __future__ import annotations

import argparse
import copy
import datetime as dt
import fcntl
import hashlib
import json
import math
import os
import shutil
import subprocess
from contextlib import contextmanager
from pathlib import Path

import pandas as pd
import torch
from _bootstrap import add_src_to_path

add_src_to_path()

from run_ablation import (
    _validate_development_control_family,
    _validate_physical_mechanism_family,
)

from cfeg.analysis.physical_mechanism import (
    PHYSICAL_MECHANISM_ROLES,
    aggregate_complete_physical_mechanism,
    evaluate_predeclared_physical_gates,
)
from cfeg.data.metadata_controls import build_development_control_plan
from cfeg.eval_loop import run_evaluation
from cfeg.execution_manifest import (
    resolved_config_sha256,
    sha256_json,
    validate_primary_pair_and_hash,
)
from cfeg.governance import (
    WEARABLE_V3_PHYSICAL_FREEZE_TAG,
    current_source_revision_contract,
)
from cfeg.prediction import load_verified_prediction_bundle, run_prediction
from cfeg.train_loop import _resolve_augmentation_channel_sets, run_training
from cfeg.utils.checkpoint import save_json
from cfeg.utils.config import load_config, merge_overrides

REPO = Path(__file__).resolve().parents[1]
CANONICAL_BASE_CONFIG = REPO / "configs/train/wearable_physical_development_loso.yaml"
CANONICAL_MECHANISM_CONFIG = REPO / "configs/train/wearable_physical_mechanism.yaml"
CANONICAL_INTERVENTION_CONFIG = REPO / "configs/eval/wearable_physical_mechanism.yaml"
CANONICAL_OUTPUT_ROOT = REPO / "outputs/development-loso/physical-mechanism-v2"
CANONICAL_DECISION_RECEIPT = (
    REPO / "configs/governance/wearable_physical_reveal2_decision.json"
)
PHYSICAL_FREEZE_TAG = WEARABLE_V3_PHYSICAL_FREEZE_TAG

_FREEZE_MARGIN_KEYS = (
    "clean_a2_mean_minimum_delta",
    "clean_a2_subject_harm_margin",
    "shortcut_equivalence_margin",
    "pairing_mechanism_margin",
    "counterfactual_mechanism_margin",
    "wrong_metadata_safety_harm_margin",
    "minimum_bundle_changed_fraction",
    "minimum_condition_flip_fraction",
    "confirmatory_control_policy",
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the outcome-gated S1-S3 physical mechanism LOSO grid."
    )
    parser.add_argument("command", choices=("prepare", "status", "train", "reveal"))
    parser.add_argument("--config", default="configs/train/wearable_physical_development_loso.yaml")
    parser.add_argument(
        "--mechanism-config", default="configs/train/wearable_physical_mechanism.yaml"
    )
    parser.add_argument("--root", default="outputs/development-loso/physical-mechanism-v2")
    args = parser.parse_args()

    _reject_symlink_components(CANONICAL_OUTPUT_ROOT)
    root = _repository_path(args.root)
    manifest_path = root / "grid_manifest.json"
    if args.command == "prepare":
        manifest = build_grid_manifest(args.config, args.mechanism_config, root)
        if root.exists() and any(path.name != "grid_manifest.json" for path in root.iterdir()):
            raise ValueError("Cannot replace a physical grid manifest beside run artifacts.")
        save_json(manifest_path, manifest)
        print_status(manifest, root)
        return

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    validate_grid_manifest(manifest, manifest_path=manifest_path)
    if current_source_revision_contract() != manifest["source_contract"]:
        raise ValueError("Current source differs from the prepared physical grid manifest.")
    if args.command == "status":
        print_status(manifest, root)
        return
    if manifest["design_status"] != "frozen":
        raise ValueError(
            "Physical mechanism design is not frozen. Owner approval, margins, and the "
            "second-reveal decision must be recorded before outcome training."
        )
    if args.command == "train":
        _preflight_global_reveal(
            manifest,
            root=root,
            allow_receipt_recovery=False,
        )
        for job in manifest["jobs"]:
            completion = Path(job["output_dir"]) / "training_completion.json"
            if completion.is_file():
                continue
            run_training(copy.deepcopy(job["config"]), resume_exact=True)
        validate_training_complete(manifest)
        print_status(manifest, root)
        return

    validate_training_complete(manifest)
    if args.command == "reveal":
        with _exclusive_reveal_publication(root):
            _stage_aggregate_and_publish(manifest, manifest_path=manifest_path, root=root)
        print_status(manifest, root)
        return


def _stage_aggregate_and_publish(
    manifest: dict, *, manifest_path: Path, root: Path
) -> None:
    """Build one complete private outcome bundle, then publish it atomically."""

    processed_dir = resolve_processed_dir(manifest["jobs"][0]["config"])
    authorization = _private_staging_authorization(manifest, manifest_path=manifest_path)
    final_dir = root / "reveal-bundle"
    staging_dir = root / ".reveal-staging"
    if root.is_symlink() or staging_dir.is_symlink() or final_dir.is_symlink():
        raise ValueError("Physical reveal root and bundle paths cannot be symlinks.")
    precommitted_digest = _preflight_global_reveal(manifest, root=root)
    if final_dir.exists():
        if staging_dir.exists():
            raise ValueError(
                "Physical reveal has both private staging and a public bundle."
            )
        bundle_digest = _validate_bundle_manifest(final_dir, manifest)
        if precommitted_digest is None:
            raise ValueError(
                "Public physical bundle exists without a prior reveal-budget precommit."
            )
        if precommitted_digest != bundle_digest:
            raise ValueError(
                "Public physical bundle differs from the precommitted reveal digest."
            )
        _finalize_reveal_publication(manifest, root=root, bundle_digest=bundle_digest)
        _validate_complete_publication(manifest, root=root)
        return

    staging_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
    staging_dir.chmod(0o700)
    for job in manifest["jobs"]:
        _prepare_private_prediction(
            job,
            processed_dir=processed_dir,
            authorization=authorization,
        )
    for intervention in manifest["interventions"]:
        _prepare_private_intervention(
            intervention,
            authorization=authorization,
        )
    _validate_prediction_directory(manifest, use_staging=True)
    _validate_intervention_directory(manifest, use_staging=True, authorization=authorization)

    full_manifest = pd.read_parquet(Path(processed_dir) / "manifest.parquet")
    staging_jobs = []
    for job in manifest["jobs"]:
        staged = copy.deepcopy(job)
        staged["prediction_csv"] = job["prediction_staging_csv"]
        staging_jobs.append(staged)
    subjects, summary = aggregate_complete_physical_mechanism(
        staging_jobs,
        full_manifest,
        expected_subjects=["sub001", "sub002", "sub003"],
        expected_n_samples=720,
    )
    intervention_rows, intervention_summary = _aggregate_intervention_results(
        manifest, use_staging=True
    )
    training_controls = _training_control_rows(manifest)
    gate_evaluation = evaluate_predeclared_physical_gates(
        subjects,
        intervention_rows,
        training_controls,
        manifest["development_control_values"],
    )
    summary.update(
        {
            "grid_manifest_sha256": manifest["grid_content_sha256"],
            "decision_receipt_sha256": manifest["decision_receipt_sha256"],
            "global_reveal_ledger": manifest["global_reveal_ledger"],
            "reveal_index": manifest["outcome_reveal_index"],
            "a2_intervention_summary": intervention_summary,
            "predeclared_gate_evaluation": gate_evaluation,
        }
    )
    _write_staged_analysis(
        staging_dir,
        subjects=subjects,
        interventions=intervention_rows,
        summary=summary,
    )
    bundle_digest = _write_bundle_manifest(staging_dir, manifest)
    if precommitted_digest is not None and precommitted_digest != bundle_digest:
        raise ValueError(
            "Recovered private physical bundle differs from the precommitted reveal digest."
        )
    _publish_prepared_bundle(
        manifest,
        root=root,
        staging_dir=staging_dir,
        final_dir=final_dir,
        bundle_digest=bundle_digest,
    )


def _publish_prepared_bundle(
    manifest: dict,
    *,
    root: Path,
    staging_dir: Path,
    final_dir: Path,
    bundle_digest: str,
) -> None:
    """Durably precommit one validated private tree, then publish and finalize it."""

    _validate_reveal_bundle_layout(manifest, use_staging=True)
    _validate_bundle_manifest(staging_dir, manifest, expected_digest=bundle_digest)
    _fsync_bundle_tree(staging_dir)
    register_global_reveal(
        manifest,
        root,
        bundle_path=staging_dir,
        bundle_content_sha256=bundle_digest,
    )
    os.replace(staging_dir, final_dir)
    _fsync_directory(root)
    _finalize_reveal_publication(manifest, root=root, bundle_digest=bundle_digest)
    _validate_complete_publication(manifest, root=root)


def _prepare_private_prediction(
    job: dict,
    *,
    processed_dir: str,
    authorization: dict,
) -> None:
    """Resume or deterministically rebuild one incomplete private prediction pair."""

    output = Path(job["prediction_staging_csv"])
    provenance = output.with_name(f"{output.stem}_provenance.json")
    _cleanup_known_atomic_json_temps(provenance)
    expected = {output.resolve(), provenance.resolve()}
    observed = {
        path.resolve()
        for path in output.parent.glob(f"{output.stem}*")
        if path.is_file()
    }
    undeclared = observed - expected
    if undeclared:
        raise ValueError(
            f"Physical private prediction prefix has undeclared artifacts: "
            f"{sorted(str(path) for path in undeclared)}."
        )
    if observed == expected:
        load_verified_prediction_bundle(
            output,
            expected_recorded_csv_path=Path(job["prediction_csv"]),
        )
        return
    if observed:
        for path in sorted(observed):
            path.unlink()
    run_prediction(
        Path(job["output_dir"]) / "final.pt",
        processed_dir,
        output,
        split="val",
        scenario="physical_mechanism_fixed_epoch_clean",
        development_authorization=authorization,
        recorded_output_path=Path(job["prediction_csv"]),
    )


def _private_intervention_artifacts(intervention: dict) -> set[Path]:
    output = Path(intervention["staging_output_csv"])
    _cleanup_known_atomic_json_temps(
        output.with_name(f"{output.stem}_provenance.json")
    )
    perturbations = list(intervention["config"].get("perturbations") or [])
    scenarios = {"clean", *(str(spec["name"]) for spec in perturbations)}
    donor_scenarios = {
        str(spec["name"])
        for spec in perturbations
        if spec.get("type") in {"metadata_shuffle", "metadata_counterfactual_wet_dry"}
    }
    paths = {
        output,
        output.with_name(f"{output.stem}_provenance.json"),
        output.with_name(f"{output.stem}_confusion.csv"),
        *(output.with_name(f"{output.stem}_{name}_predictions.csv") for name in scenarios),
        *(output.with_name(f"{output.stem}_{name}_donors.csv") for name in donor_scenarios),
    }
    return {path.resolve() for path in paths}


def _prepare_private_intervention(
    intervention: dict,
    *,
    authorization: dict,
) -> None:
    """Resume or deterministically rebuild one incomplete private intervention set."""

    output = Path(intervention["staging_output_csv"])
    expected = _private_intervention_artifacts(intervention)
    observed = {
        path.resolve()
        for path in output.parent.glob(f"{output.stem}*")
        if path.is_file()
    }
    undeclared = observed - expected
    if undeclared:
        raise ValueError(
            f"Physical private intervention prefix has undeclared artifacts: "
            f"{sorted(str(path) for path in undeclared)}."
        )
    if observed == expected:
        return
    if observed:
        for path in sorted(observed):
            path.unlink()
    evaluation = copy.deepcopy(intervention["config"])
    evaluation["development_authorization"] = authorization
    run_evaluation(
        evaluation,
        intervention["checkpoint_path"],
        access_route="physical_mechanism_intervention",
    )


def build_grid_manifest(config_path: str, mechanism_path: str, root: Path) -> dict:
    _reject_symlink_components(CANONICAL_OUTPUT_ROOT)
    base_path = _repository_path(config_path)
    mechanism_config_path = _repository_path(mechanism_path)
    if base_path != CANONICAL_BASE_CONFIG.resolve():
        raise ValueError(f"Physical grid requires canonical base config: {CANONICAL_BASE_CONFIG}.")
    if mechanism_config_path != CANONICAL_MECHANISM_CONFIG.resolve():
        raise ValueError(
            f"Physical grid requires canonical mechanism config: {CANONICAL_MECHANISM_CONFIG}."
        )
    base = load_config(base_path, strict_env=False)
    design = base.get("development_grid") or {}
    if design.get("schema") != "cfeg.development-loso-design.v2":
        raise ValueError("Unknown physical development LOSO design schema.")
    decision = _validate_design_freeze_authorization(base)
    canonical_root = _repository_path(str(design.get("canonical_output_root", "")))
    if canonical_root != CANONICAL_OUTPUT_ROOT.resolve() or root.resolve() != canonical_root:
        raise ValueError(
            f"Physical mechanism root must be canonical: {canonical_root}; got {root}."
        )
    suite = load_config(mechanism_config_path, strict_env=False)
    variants = suite.get("variants") or {}
    roles = tuple(design.get("roles") or [])
    if roles != PHYSICAL_MECHANISM_ROLES:
        raise ValueError("Physical design must declare the exact ordered six-role matrix.")
    fairness = validate_primary_pair_and_hash(base, variants)
    if fairness is None:
        raise ValueError("Physical mechanism suite has no complete A0/A2 pair.")
    control_family = _validate_development_control_family(base, suite)
    mechanism_family = _validate_physical_mechanism_family(base, suite)
    if not control_family or not mechanism_family:
        raise ValueError("Physical mechanism suite must bind control and mechanism families.")

    historical_ledger = _repository_path(str(design["historical_reveal_ledger"]))
    historical = _load_reveal_ledger(historical_ledger)
    historical_indices = [int(entry["reveal_index"]) for entry in historical["entries"]]
    outcome_reveal_index = int(design["outcome_reveal_index_if_authorized"])
    if sorted(historical_indices) != list(range(1, outcome_reveal_index)):
        raise ValueError(
            "Historical S1-S3 reveal ledger does not establish the expected prior reveal count."
        )

    jobs = _build_grid_jobs(
        base,
        variants,
        design=design,
        roles=roles,
        root=root,
        fairness=fairness,
        control_family=control_family,
        mechanism_family=mechanism_family,
    )
    intervention_base = load_config(CANONICAL_INTERVENTION_CONFIG, strict_env=False)
    interventions = _build_intervention_jobs(intervention_base, jobs, root=root)

    source_contract = current_source_revision_contract()
    _validate_clean_source_freeze(source_contract, decision)
    manifest = {
        "schema": "cfeg.physical-mechanism-grid.v1",
        "base_config": str(CANONICAL_BASE_CONFIG.relative_to(REPO)),
        "base_config_sha256": file_sha256(CANONICAL_BASE_CONFIG),
        "mechanism_config": str(CANONICAL_MECHANISM_CONFIG.relative_to(REPO)),
        "mechanism_config_sha256": file_sha256(CANONICAL_MECHANISM_CONFIG),
        "intervention_config": str(CANONICAL_INTERVENTION_CONFIG.relative_to(REPO)),
        "intervention_config_sha256": file_sha256(CANONICAL_INTERVENTION_CONFIG),
        "decision_receipt": str(CANONICAL_DECISION_RECEIPT.relative_to(REPO)),
        "decision_receipt_sha256": file_sha256(CANONICAL_DECISION_RECEIPT),
        "development_control_values": copy.deepcopy(decision["control_values"]),
        "grid_id": design["grid_id"],
        "design_status": design["status"],
        "freeze_authorization": copy.deepcopy(design["freeze_authorization"]),
        "interpretation": design["interpretation"],
        "canonical_output_root": str(root),
        "historical_reveal_ledger": str(historical_ledger),
        "historical_reveal_ledger_sha256": file_sha256(historical_ledger),
        "global_reveal_ledger": str(_repository_path(str(design["global_reveal_ledger"]))),
        "outcome_reveal_index": outcome_reveal_index,
        "recommended_max_outcome_reveals": int(design["recommended_max_outcome_reveals"]),
        "absolute_max_outcome_reveals": int(design["absolute_max_outcome_reveals"]),
        "primary_fairness_hash": fairness,
        "development_control_family_sha256": control_family,
        "physical_mechanism_family_sha256": mechanism_family,
        "source_freeze_tag": PHYSICAL_FREEZE_TAG,
        "source_contract": source_contract,
        "expected_job_count": len(jobs),
        "jobs": jobs,
        "expected_intervention_count": len(interventions),
        "interventions": interventions,
    }
    manifest["grid_content_sha256"] = sha256_json(manifest)
    validate_grid_manifest(manifest, manifest_path=root / "grid_manifest.json")
    return manifest


def validate_grid_manifest(manifest: dict, *, manifest_path: Path) -> None:
    if manifest.get("schema") != "cfeg.physical-mechanism-grid.v1":
        raise ValueError("Unknown physical mechanism grid schema.")
    recorded = manifest.get("grid_content_sha256")
    payload = {key: value for key, value in manifest.items() if key != "grid_content_sha256"}
    if recorded != sha256_json(payload):
        raise ValueError("Physical mechanism grid manifest changed after preparation.")
    root = Path(manifest["canonical_output_root"]).resolve()
    if root != CANONICAL_OUTPUT_ROOT.resolve():
        raise ValueError("Physical mechanism manifest does not use the canonical output root.")
    if manifest_path.resolve() != (root / "grid_manifest.json").resolve():
        raise ValueError("Physical mechanism manifest is outside its canonical root.")
    jobs = list(manifest.get("jobs") or [])
    expected = {(42, fold, role) for fold in range(3) for role in PHYSICAL_MECHANISM_ROLES}
    observed = {(int(job["seed"]), int(job["fold_index"]), str(job["role"])) for job in jobs}
    if len(jobs) != 18 or observed != expected or manifest.get("expected_job_count") != 18:
        raise ValueError("Physical mechanism grid is not the exact 18-job Cartesian product.")
    if manifest.get("outcome_reveal_index") != 2:
        raise ValueError("Physical mechanism grid must be the second S1-S3 outcome reveal.")
    historical = Path(manifest["historical_reveal_ledger"])
    if not historical.is_file() or file_sha256(historical) != manifest.get(
        "historical_reveal_ledger_sha256"
    ):
        raise ValueError("Historical first-reveal ledger is missing or changed.")
    bound_files = {
        "base_config": CANONICAL_BASE_CONFIG,
        "mechanism_config": CANONICAL_MECHANISM_CONFIG,
        "intervention_config": CANONICAL_INTERVENTION_CONFIG,
        "decision_receipt": CANONICAL_DECISION_RECEIPT,
    }
    for field, expected_path in bound_files.items():
        if (
            _repository_path(str(manifest.get(field, ""))) != expected_path.resolve()
            or manifest.get(f"{field}_sha256") != file_sha256(expected_path)
        ):
            raise ValueError(f"Physical grid {field} binding is missing or stale.")
    base = load_config(CANONICAL_BASE_CONFIG, strict_env=False)
    suite = load_config(CANONICAL_MECHANISM_CONFIG, strict_env=False)
    design = base.get("development_grid") or {}
    decision = _validate_design_freeze_authorization(base)
    if (
        manifest.get("grid_id") != design.get("grid_id")
        or manifest.get("design_status") != design.get("status")
        or manifest.get("freeze_authorization") != design.get("freeze_authorization")
        or manifest.get("decision_receipt_sha256")
        != file_sha256(CANONICAL_DECISION_RECEIPT)
        or manifest.get("development_control_values") != decision.get("control_values")
        or manifest.get("source_freeze_tag") != PHYSICAL_FREEZE_TAG
        or manifest.get("interpretation") != design.get("interpretation")
        or Path(str(manifest.get("global_reveal_ledger"))).resolve()
        != _repository_path(str(design.get("global_reveal_ledger")))
    ):
        raise ValueError("Physical grid design differs from its canonical source config.")
    _validate_clean_source_freeze(manifest.get("source_contract") or {}, decision)
    variants = suite.get("variants") or {}
    fairness = validate_primary_pair_and_hash(base, variants)
    control_family = _validate_development_control_family(base, suite)
    mechanism_family = _validate_physical_mechanism_family(base, suite)
    expected_jobs = _build_grid_jobs(
        base,
        variants,
        design=design,
        roles=PHYSICAL_MECHANISM_ROLES,
        root=root,
        fairness=str(fairness),
        control_family=str(control_family),
        mechanism_family=str(mechanism_family),
    )
    expected_interventions = _build_intervention_jobs(
        load_config(CANONICAL_INTERVENTION_CONFIG, strict_env=False),
        expected_jobs,
        root=root,
    )
    if (
        manifest.get("primary_fairness_hash") != fairness
        or manifest.get("development_control_family_sha256") != control_family
        or manifest.get("physical_mechanism_family_sha256") != mechanism_family
        or jobs != expected_jobs
        or manifest.get("expected_intervention_count") != 3
        or manifest.get("interventions") != expected_interventions
    ):
        raise ValueError("Physical mechanism jobs differ from the canonical six-role design.")
    if len({job["job_id"] for job in jobs}) != len(jobs):
        raise ValueError("Physical mechanism job IDs must be unique.")
    for job in jobs:
        if resolved_config_sha256(job["config"]) != job["resolved_config_sha256"]:
            raise ValueError(f"Physical job config changed: {job['job_id']}.")
        cfg = job["config"]
        protocol = cfg.get("protocol", {})
        expected_output = root / "runs" / f"fold{job['fold_index']}" / job["role"]
        expected_prediction = (
            root / "reveal-bundle" / "predictions" / f"{job['job_id']}.csv"
        )
        expected_staging = (
            root / ".reveal-staging" / "predictions" / f"{job['job_id']}.csv"
        )
        if (
            Path(job["output_dir"]).resolve() != expected_output.resolve()
            or Path(job["prediction_csv"]).resolve() != expected_prediction.resolve()
            or Path(job["prediction_staging_csv"]).resolve() != expected_staging.resolve()
            or cfg.get("output_dir") != job["output_dir"]
            or protocol.get("development_grid_job_id") != job["job_id"]
            or protocol.get("primary_ablation_role") != job["role"]
            or protocol.get("physical_mechanism_family_sha256")
            != manifest["physical_mechanism_family_sha256"]
        ):
            raise ValueError(f"Physical job binding mismatch: {job['job_id']}.")


def _build_grid_jobs(
    base: dict,
    variants: dict,
    *,
    design: dict,
    roles: tuple[str, ...],
    root: Path,
    fairness: str,
    control_family: str,
    mechanism_family: str,
) -> list[dict]:
    jobs: list[dict] = []
    for seed in design["seeds"]:
        for fold in design["folds"]:
            for role in roles:
                cfg = merge_overrides(
                    copy.deepcopy(base),
                    [f"{key}={value!r}" for key, value in variants[role].items()],
                )
                cfg["seed"] = int(seed)
                cfg["data"]["fold_index"] = int(fold)
                cfg["run_name"] = f"physical-mechanism-fold{fold}-{role}"
                cfg["output_dir"] = str(root / "runs" / f"fold{fold}" / role)
                protocol = cfg.setdefault("protocol", {})
                protocol["primary_ablation_role"] = role
                protocol["primary_fairness_hash"] = fairness
                protocol["development_control_family_sha256"] = control_family
                protocol["physical_mechanism_family_sha256"] = mechanism_family
                job_id = f"seed{seed}-fold{fold}-{role}"
                protocol["development_grid_job_id"] = job_id
                protocol["development_grid_manifest"] = str(root / "grid_manifest.json")
                _resolve_augmentation_channel_sets(cfg)
                jobs.append(
                    {
                        "job_id": job_id,
                        "seed": int(seed),
                        "fold_index": int(fold),
                        "role": role,
                        "output_dir": cfg["output_dir"],
                        "prediction_csv": str(
                            root / "reveal-bundle" / "predictions" / f"{job_id}.csv"
                        ),
                        "prediction_staging_csv": str(
                            root / ".reveal-staging" / "predictions" / f"{job_id}.csv"
                        ),
                        "resolved_config_sha256": resolved_config_sha256(cfg),
                        "config": cfg,
                    }
                )
    return jobs


def _build_intervention_jobs(eval_base: dict, jobs: list[dict], *, root: Path) -> list[dict]:
    interventions: list[dict] = []
    for job in jobs:
        if job["role"] != "A2_structured_condition_prompt":
            continue
        fold = int(job["fold_index"])
        config = copy.deepcopy(eval_base)
        config["seed"] = int(job["seed"])
        config["data"]["processed_dirs"] = list(job["config"]["data"]["processed_dirs"])
        config["data"]["split"] = "val"
        config["output_csv"] = str(
            root / ".reveal-staging" / "interventions" / f"fold{fold}-A2.csv"
        )
        config["save_predictions"] = True
        interventions.append(
            {
                "job_id": job["job_id"],
                "fold_index": fold,
                "checkpoint_path": str(Path(job["output_dir"]) / "final.pt"),
                "output_csv": str(
                    root / "reveal-bundle" / "interventions" / f"fold{fold}-A2.csv"
                ),
                "staging_output_csv": config["output_csv"],
                "resolved_config_sha256": resolved_config_sha256(config),
                "config": config,
            }
        )
    if len(interventions) != 3:
        raise ValueError("Physical intervention matrix requires one Full A2 job per fold.")
    return interventions


def _validate_design_freeze_authorization(base: dict) -> dict:
    design = base.get("development_grid") or {}
    authorization = design.get("freeze_authorization") or {}
    status = design.get("status")
    if base.get("protocol", {}).get("development_outcome_gate_required") is not True:
        raise ValueError(
            "Physical development must keep protocol.development_outcome_gate_required=true."
        )
    if status == "draft_pre_freeze":
        expected = {
            "status": "pending_owner_approval",
            "decision_id": None,
            "approved_by": None,
            "approved_at_utc": None,
            "authorization_source": None,
            "decision_receipt": None,
            "decision_receipt_sha256": None,
            "outcome_reveal_2_approved": False,
        }
        if authorization != expected:
            raise ValueError("Draft physical grid has an invalid owner-authorization state.")
        return {}
    if status != "frozen":
        raise ValueError("Physical grid status must be draft_pre_freeze or frozen.")
    required_text = (
        "decision_id",
        "approved_by",
        "approved_at_utc",
        "authorization_source",
        "decision_receipt",
        "decision_receipt_sha256",
    )
    if (
        authorization.get("status") != "approved"
        or authorization.get("outcome_reveal_2_approved") is not True
        or any(not str(authorization.get(key) or "").strip() for key in required_text)
    ):
        raise ValueError(
            "Frozen physical grid requires an explicit owner decision, identity, timestamp, "
            "and reveal #2 approval."
        )
    try:
        approved_at = dt.datetime.fromisoformat(str(authorization["approved_at_utc"]))
    except ValueError as exc:
        raise ValueError("Physical grid approval timestamp must be ISO-8601.") from exc
    if approved_at.tzinfo is None or approved_at.utcoffset() != dt.timedelta(0):
        raise ValueError("Physical grid approval timestamp must be explicitly UTC.")
    decision_path = _repository_path(str(authorization["decision_receipt"]))
    if decision_path != CANONICAL_DECISION_RECEIPT.resolve() or not decision_path.is_file():
        raise ValueError("Physical grid decision receipt path is missing or noncanonical.")
    if authorization["decision_receipt_sha256"] != file_sha256(decision_path):
        raise ValueError("Physical grid decision receipt hash is stale.")
    decision = json.loads(decision_path.read_text(encoding="utf-8"))
    scope = decision.get("scope") or {}
    stopping = decision.get("stopping_rules") or {}
    correction = decision.get("technical_correction") or {}
    paths = decision.get("canonical_paths") or {}
    bound = decision.get("bound_artifacts") or {}
    if (
        decision.get("schema") != "cfeg.physical-development-freeze-decision.v1"
        or decision.get("status") != "approved"
        or decision.get("decision_id") != authorization["decision_id"]
        or decision.get("supersedes_decision_id") != "DEC-20260831-003"
        or decision.get("approved_by") != authorization["approved_by"]
        or decision.get("approval_recorded_at_utc") != authorization["approved_at_utc"]
        or decision.get("authority_basis") != authorization["authorization_source"]
        or scope.get("grid_id") != design.get("grid_id")
        or int(scope.get("outcome_reveal_index", -1))
        != int(design.get("outcome_reveal_index_if_authorized", -2))
        or int(scope.get("seed", -1)) != 42
        or scope.get("folds") != [0, 1, 2]
        or tuple(scope.get("roles") or []) != PHYSICAL_MECHANISM_ROLES
        or int(scope.get("training_job_count", -1)) != 18
        or int(scope.get("intervention_bundle_count", -1)) != 3
        or int(scope.get("fixed_epochs", -1)) != int(base.get("train", {}).get("epochs", -2))
        or stopping.get("no_partial_reveal") is not True
        or stopping.get("any_gate_failure")
        != "block_confirmatory_and_forbid_s1_s3_retuning"
        or stopping.get("all_gates_pass")
        != "require_separate_owner_review_before_any_confirmatory_freeze"
        or stopping.get("further_s1_s3_outcome_reveals")
        != "forbidden_by_default_after_reveal_2"
        or correction.get("prior_attempt_stopped_before_reveal_budget_precommit")
        is not True
        or correction.get("prior_attempt_global_ledger_entry_created") is not False
        or correction.get("prior_attempt_public_bundle_created") is not False
        or correction.get("performance_outcomes_reviewed") is not False
        or correction.get("prior_attempt_source_commit")
        != "ed5e7dce385fc456b33350079a98baa4aac88ec0"
        or correction.get("diagnostic_access_limited_to")
        != "donor_mapping_and_control_provenance_fields_only"
        or correction.get("quarantine_path")
        != (
            "outputs/development-loso/quarantine/"
            "physical-mechanism-v2-aborted-precommit-ed5e7dc-20260901"
        )
        or correction.get("quarantine_mode") != "0700"
        or correction.get("restart_rule")
        != (
            "retrain_the_exact_18_job_grid_from_scratch_and_reuse_reveal_index_2_"
            "without_threshold_or_model_changes"
        )
        or (decision.get("source_freeze") or {}).get("annotated_tag")
        != PHYSICAL_FREEZE_TAG
        or paths.get("base_config")
        != str(CANONICAL_BASE_CONFIG.relative_to(REPO))
        or paths.get("mechanism_config")
        != str(CANONICAL_MECHANISM_CONFIG.relative_to(REPO))
        or paths.get("intervention_config")
        != str(CANONICAL_INTERVENTION_CONFIG.relative_to(REPO))
        or paths.get("historical_reveal_ledger")
        != str(Path(design["historical_reveal_ledger"]))
        or paths.get("global_reveal_ledger")
        != str(Path(design["global_reveal_ledger"]))
        or paths.get("output_root") != str(Path(design["canonical_output_root"]))
        or bound.get("mechanism_config_sha256")
        != file_sha256(CANONICAL_MECHANISM_CONFIG)
        or bound.get("intervention_config_sha256")
        != file_sha256(CANONICAL_INTERVENTION_CONFIG)
        or bound.get("historical_reveal_ledger_sha256")
        != file_sha256(_repository_path(str(design["historical_reveal_ledger"])))
    ):
        raise ValueError("Physical grid decision receipt does not match the frozen design.")
    control = decision.get("control_values") or {}
    missing = [key for key in _FREEZE_MARGIN_KEYS if control.get(key) is None]
    if missing:
        raise ValueError(
            "Frozen physical grid requires all mechanism/safety decisions before training: "
            f"{missing}."
        )
    for key in (
        "clean_a2_mean_minimum_delta",
        "clean_a2_subject_harm_margin",
        "shortcut_equivalence_margin",
        "pairing_mechanism_margin",
        "counterfactual_mechanism_margin",
        "wrong_metadata_safety_harm_margin",
    ):
        if not 0.0 <= float(control[key]) <= 1.0:
            raise ValueError(f"{key} must be within [0, 1].")
    for key in ("minimum_bundle_changed_fraction", "minimum_condition_flip_fraction"):
        if not 0.0 < float(control[key]) <= 1.0:
            raise ValueError(f"{key} must be within (0, 1].")
    if control["confirmatory_control_policy"] != "development_gate_only":
        raise ValueError(
            "Only confirmatory_control_policy=development_gate_only is currently implemented."
        )
    return decision


def _validate_clean_source_freeze(source_contract: dict, decision: dict) -> None:
    """Require the recorded clean commit to be the annotated physical freeze tag."""

    expected_tag = str((decision.get("source_freeze") or {}).get("annotated_tag") or "")
    commit = str(source_contract.get("source_commit_sha") or "")
    if source_contract.get("source_dirty") is not False or not commit or not expected_tag:
        raise ValueError("Physical reveal preparation requires a clean tagged source tree.")
    try:
        tag_type = subprocess.run(
            ["git", "cat-file", "-t", f"refs/tags/{expected_tag}"],
            cwd=REPO,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        tagged_commit = subprocess.run(
            ["git", "rev-parse", f"refs/tags/{expected_tag}^{{commit}}"],
            cwd=REPO,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except subprocess.CalledProcessError as exc:
        raise ValueError("Physical reveal source freeze tag is missing or invalid.") from exc
    if tag_type != "tag" or tagged_commit != commit:
        raise ValueError("Physical reveal requires an annotated tag on the recorded commit.")


def validate_training_complete(manifest: dict) -> None:
    expected_controls = {
        "A0_eeg_only": "none",
        "A2_structured_condition_prompt": "none",
        "M1_global_only": "none",
        "M2_channel_only": "none",
        "M3_full_shuffle_train": "within_class_shuffle",
        "M4_metadata_only": "metadata_only",
    }
    incomplete: list[str] = []
    for job in manifest["jobs"]:
        root = Path(job["output_dir"])
        completion_path = root / "training_completion.json"
        final_path = root / "final.pt"
        last_path = root / "last.pt"
        metrics_path = root / "metrics_train.csv"
        split_path = root / "split.csv"
        control_path = root / "development_control.json"
        donor_path = root / "development_control_donors.csv"
        required = [
            completion_path,
            final_path,
            last_path,
            metrics_path,
            split_path,
            control_path,
        ]
        if any(not path.is_file() for path in required) or (root / "metrics_val.csv").exists():
            incomplete.append(job["job_id"])
            continue
        completion = json.loads(completion_path.read_text(encoding="utf-8"))
        checkpoint = torch.load(final_path, map_location="cpu", weights_only=False)
        cfg = checkpoint.get("config") or {}
        recorded_control = json.loads(control_path.read_text(encoding="utf-8"))
        runtime_control = (cfg.get("runtime_contract") or {}).get("development_control")
        planned_cfg = copy.deepcopy(cfg)
        runtime_contract = planned_cfg.pop("runtime_contract", {})
        protocol = cfg.get("protocol", {})
        expected_completion = {
            "schema": "cfeg.training-completion.v1",
            "status": "completed",
            "completed_epoch": int(job["config"]["train"]["epochs"]),
            "stop_reason": "max_epochs",
            "resolved_config_sha256": job["resolved_config_sha256"],
            "runtime_contract_sha256": sha256_json(runtime_contract),
        }
        completion_mismatch = {
            key: {"expected": value, "observed": completion.get(key)}
            for key, value in expected_completion.items()
            if completion.get(key) != value
        }
        expected_artifact_hashes = {
            "final.pt": file_sha256(final_path),
            "last.pt": file_sha256(last_path),
            "metrics_train.csv": file_sha256(metrics_path),
            "split.csv": file_sha256(split_path),
            "development_control.json": file_sha256(control_path),
        }
        if donor_path.is_file():
            expected_artifact_hashes["development_control_donors.csv"] = file_sha256(
                donor_path
            )
        expected_checkpoint_hashes = {
            name: expected_artifact_hashes[name] for name in ("final.pt", "last.pt")
        }
        if completion.get("checkpoint_sha256") != expected_checkpoint_hashes:
            completion_mismatch["checkpoint_sha256"] = {
                "expected": expected_checkpoint_hashes,
                "observed": completion.get("checkpoint_sha256"),
            }
        artifact_names = [
            "metrics_train.csv",
            "split.csv",
            "development_control.json",
        ]
        if donor_path.is_file():
            artifact_names.append("development_control_donors.csv")
        expected_completion_artifacts = {
            name: expected_artifact_hashes[name] for name in artifact_names
        }
        if completion.get("artifact_sha256") != expected_completion_artifacts:
            completion_mismatch["artifact_sha256"] = {
                "expected": expected_completion_artifacts,
                "observed": completion.get("artifact_sha256"),
            }
        resume_generation = completion.get("resume_generation")
        resume_state_sha256 = str(completion.get("resume_state_sha256") or "")
        if (
            not isinstance(resume_generation, int)
            or isinstance(resume_generation, bool)
            or resume_generation < 0
            or len(resume_state_sha256) != 64
            or any(character not in "0123456789abcdef" for character in resume_state_sha256)
        ):
            completion_mismatch["exact_resume"] = {
                "expected": "non-negative generation and lowercase SHA-256 state",
                "observed": {
                    "generation": resume_generation,
                    "state_sha256": completion.get("resume_state_sha256"),
                },
            }
        source_contract = {
            key: runtime_contract.get(key)
            for key in ("source_commit_sha", "source_dirty", "source_tree_sha256")
        }
        expected_control = expected_controls[job["role"]]
        control_invalid = (
            recorded_control != runtime_control
            or (cfg.get("runtime_contract") or {}).get("development_control_sha256")
            != sha256_json(recorded_control)
            or recorded_control.get("name") != expected_control
            or int(recorded_control.get("natural_missingness_pattern_count", -1)) != 1
        )
        if expected_control == "within_class_shuffle":
            if not donor_path.is_file():
                control_invalid = True
            else:
                donors = pd.read_csv(donor_path)
                changed_fraction = float(
                    recorded_control.get(
                        "effective_external_metadata_changed_fraction", -1.0
                    )
                )
                control_invalid = control_invalid or (
                    recorded_control.get("donor_scope")
                    != "within_dataset_block_derangement_label_aligned"
                    or recorded_control.get("label_alignment")
                    != "exact_label_and_window_within_donor_block"
                    or int(recorded_control.get("mapping_row_count", -1)) != 480
                    or int(recorded_control.get("fixed_point_count", -1)) != 0
                    or float(recorded_control.get("pair_coverage", -1.0)) != 1.0
                    or not math.isfinite(changed_fraction)
                    or not 0.0 <= changed_fraction <= 1.0
                    or recorded_control.get("donor_mapping_sha256")
                    != file_sha256(donor_path)
                    or len(donors) != 480
                    or donors["sample_id"].astype(str).duplicated().any()
                    or not donors["target_label"].astype(int).equals(
                        donors["donor_label"].astype(int)
                    )
                )
        elif donor_path.exists():
            control_invalid = True
        if (
            completion_mismatch
            or control_invalid
            or resolved_config_sha256(planned_cfg) != job["resolved_config_sha256"]
            or checkpoint.get("checkpoint_role") != "development_fixed_epoch"
            or checkpoint.get("selection_split") is not None
            or checkpoint.get("selection_metric") != "fixed_epoch"
            or int(checkpoint.get("epoch", -1)) != int(job["config"]["train"]["epochs"])
            or protocol.get("primary_ablation_role") != job["role"]
            or protocol.get("development_grid_job_id") != job["job_id"]
            or protocol.get("development_grid_manifest")
            != job["config"]["protocol"]["development_grid_manifest"]
            or protocol.get("primary_fairness_hash") != manifest["primary_fairness_hash"]
            or protocol.get("development_control_family_sha256")
            != manifest["development_control_family_sha256"]
            or protocol.get("physical_mechanism_family_sha256")
            != manifest["physical_mechanism_family_sha256"]
            or int(cfg.get("data", {}).get("fold_index", -1)) != int(job["fold_index"])
            or source_contract != manifest["source_contract"]
        ):
            raise ValueError(
                f"Physical final checkpoint contract failed: {job['job_id']}; "
                f"completion_mismatch={completion_mismatch}."
            )
    if incomplete:
        raise ValueError(f"Physical mechanism training grid is incomplete: {incomplete}.")


def validate_predictions_complete(manifest: dict) -> None:
    _validate_prediction_directory(manifest, use_staging=False)


def _validate_prediction_directory(manifest: dict, *, use_staging: bool) -> None:
    field = "prediction_staging_csv" if use_staging else "prediction_csv"
    expected_files: set[Path] = set()
    missing: list[str] = []
    fold_reference: dict[int, pd.DataFrame] = {}
    role_coverage: dict[str, set[str]] = {
        role: set() for role in PHYSICAL_MECHANISM_ROLES
    }
    for job in manifest["jobs"]:
        path = Path(job[field])
        provenance = path.with_name(f"{path.stem}_provenance.json")
        expected_files.update({path.resolve(), provenance.resolve()})
        if not path.is_file() or not provenance.is_file():
            missing.append(job["job_id"])
            continue
        bundle = load_verified_prediction_bundle(
            path,
            expected_recorded_csv_path=job["prediction_csv"],
        )
        frame = bundle.frame
        artifact = bundle.artifact
        contract = artifact.get("primary_contract") or {}
        role = str(job["role"])
        fold = int(job["fold_index"])
        expected_contract = {
            "execution_job_id": job["job_id"],
            "execution_manifest_content_sha256": manifest["grid_content_sha256"],
            "lockbox_reveal_receipt_sha256": manifest["decision_receipt_sha256"],
            "planned_config_sha256": job["resolved_config_sha256"],
            "training_completion_sha256": file_sha256(
                Path(job["output_dir"]) / "training_completion.json"
            ),
            "primary_ablation_role": role,
            "primary_fairness_hash": manifest["primary_fairness_hash"],
            "optimization_seed": job["seed"],
            "fold_index": fold,
            "conditioning_architecture": "physical_hybrid_v1",
            "execution_phase": "development",
        }
        provenance_mismatch = {
            key: {"expected": value, "observed": contract.get(key)}
            for key, value in expected_contract.items()
            if str(contract.get(key)) != str(value)
        }
        observed_checkpoint = Path(
            str(artifact.get("checkpoint_path", ""))
        ).expanduser().resolve()
        expected_checkpoint = (Path(job["output_dir"]) / "final.pt").resolve()
        if observed_checkpoint != expected_checkpoint:
            provenance_mismatch["checkpoint_path"] = {
                "expected": str(expected_checkpoint),
                "observed": str(observed_checkpoint),
            }
        if (
            provenance_mismatch
            or len(frame) != 240
            or frame["sample_id"].astype(str).duplicated().any()
            or set(frame["label"].astype(int)) != set(range(12))
            or not frame.groupby(frame["label"].astype(int)).size().eq(20).all()
            or not frame["prediction"].astype(int).between(0, 11).all()
            or set(frame["selection_split"].astype(str)) != {"val"}
            or set(frame["primary_ablation_role"].astype(str)) != {role}
            or set(frame["optimization_seed"].astype(int)) != {int(job["seed"])}
            or set(frame["fold_index"].astype(int)) != {fold}
            or set(frame["conditioning_architecture"].astype(str))
            != {"physical_hybrid_v1"}
        ):
            raise ValueError(
                f"Physical prediction contract mismatch: {job['job_id']}; "
                f"provenance={provenance_mismatch}."
            )
        identity = frame[["sample_id", "label"]].copy()
        identity["sample_id"] = identity["sample_id"].astype(str)
        identity["label"] = identity["label"].astype(int)
        identity = identity.sort_values("sample_id").reset_index(drop=True)
        reference = fold_reference.setdefault(fold, identity)
        if not identity.equals(reference):
            raise ValueError(f"Physical fold sample/label pairing differs for fold={fold}.")
        ids = set(identity["sample_id"])
        if role_coverage[role] & ids:
            raise ValueError(f"Physical held-out samples overlap between folds for role={role}.")
        role_coverage[role].update(ids)
    if missing:
        raise ValueError(f"Physical mechanism prediction grid is incomplete: {missing}.")
    if any(len(ids) != 720 for ids in role_coverage.values()):
        raise ValueError("Physical prediction roles do not each cover exactly 720 held-out rows.")
    directory = Path(manifest["jobs"][0][field]).parent
    observed = {path.resolve() for path in directory.iterdir() if path.is_file()}
    if observed != expected_files:
        raise ValueError("Physical prediction directory has missing or undeclared artifacts.")


def validate_interventions_complete(manifest: dict) -> None:
    root = Path(manifest["canonical_output_root"])
    authorization = _private_staging_authorization(
        manifest, manifest_path=root / "grid_manifest.json"
    )
    _validate_intervention_directory(
        manifest,
        use_staging=False,
        authorization=authorization,
    )


def _validate_intervention_directory(
    manifest: dict,
    *,
    use_staging: bool,
    authorization: dict,
) -> None:
    field = "staging_output_csv" if use_staging else "output_csv"
    expected_files: set[Path] = set()
    expected_scenarios: set[str] | None = None
    interventions = list(manifest.get("interventions") or [])
    if not interventions:
        raise ValueError("Physical A2 intervention matrix is empty.")
    processed_dir = resolve_processed_dir(interventions[0]["config"])
    asset_manifest = pd.read_parquet(Path(processed_dir) / "manifest.parquet")
    for intervention in manifest.get("interventions") or []:
        output = Path(intervention[field])
        provenance_path = output.with_name(f"{output.stem}_provenance.json")
        if not output.is_file() or not provenance_path.is_file():
            raise ValueError(
                f"Physical A2 intervention bundle is incomplete: fold "
                f"{intervention['fold_index']}."
            )
        frame = pd.read_csv(output)
        scenarios = {
            "clean",
            *(str(spec["name"]) for spec in intervention["config"]["perturbations"]),
        }
        if expected_scenarios is None:
            expected_scenarios = scenarios
        elif expected_scenarios != scenarios:
            raise ValueError("Physical A2 intervention scenarios differ between folds.")
        if "balanced_accuracy" not in frame:
            raise ValueError("Physical A2 intervention result lacks balanced_accuracy.")
        balanced_accuracy = pd.to_numeric(frame["balanced_accuracy"], errors="coerce")
        if (
            len(frame) != len(scenarios)
            or set(frame["scenario"].astype(str)) != scenarios
            or frame["scenario"].duplicated().any()
            or balanced_accuracy.isna().any()
            or not balanced_accuracy.between(0.0, 1.0).all()
        ):
            raise ValueError("Physical A2 intervention scenario matrix is incomplete.")
        provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
        matching_jobs = [
            job for job in manifest["jobs"] if job["job_id"] == intervention["job_id"]
        ]
        if len(matching_jobs) != 1:
            raise ValueError("Physical intervention has no unique Full A2 grid job.")
        full_job = matching_jobs[0]
        checkpoint = torch.load(
            intervention["checkpoint_path"], map_location="cpu", weights_only=False
        )
        checkpoint_runtime = (checkpoint.get("config") or {}).get("runtime_contract") or {}
        runtime_config = copy.deepcopy(intervention["config"])
        runtime_config["development_authorization"] = authorization
        scenario_records = list(provenance.get("scenarios") or [])
        scenario_record_names = [str(item.get("scenario")) for item in scenario_records]
        sample_identity_hashes = {
            str(item.get("sample_id_identity_sha256") or "") for item in scenario_records
        }
        if (
            provenance.get("schema") != "cfeg.evaluation-provenance.v1"
            or provenance.get("mode") != "robustness"
            or provenance.get("confirmatory_claim_allowed") is not False
            or Path(str(provenance.get("checkpoint_path", ""))).resolve()
            != Path(intervention["checkpoint_path"]).resolve()
            or provenance.get("checkpoint_sha256")
            != file_sha256(Path(intervention["checkpoint_path"]))
            or provenance.get("checkpoint_runtime_contract") != checkpoint_runtime
            or provenance.get("evaluation_config_sha256") != sha256_json(runtime_config)
            or provenance.get("current_source_revision") != manifest["source_contract"]
            or len(scenario_records) != len(scenarios)
            or len(scenario_record_names) != len(set(scenario_record_names))
            or set(scenario_record_names) != scenarios
            or any(int(item.get("n_samples", -1)) != 240 for item in scenario_records)
            or len(sample_identity_hashes) != 1
            or any(
                len(value) != 64 or any(character not in "0123456789abcdef" for character in value)
                for value in sample_identity_hashes
            )
        ):
            raise ValueError("Physical A2 intervention provenance contract failed.")
        artifacts = provenance.get("artifacts") or {}
        donor_scenarios = {
            str(spec["name"])
            for spec in intervention["config"]["perturbations"]
            if spec.get("type") in {"metadata_shuffle", "metadata_counterfactual_wet_dry"}
        }
        expected_artifact_names = {
            output.name,
            f"{output.stem}_confusion.csv",
            *(f"{output.stem}_{scenario}_predictions.csv" for scenario in scenarios),
            *(f"{output.stem}_{scenario}_donors.csv" for scenario in donor_scenarios),
        }
        if set(artifacts) != expected_artifact_names:
            raise ValueError("Physical A2 intervention artifact inventory is incomplete.")
        expected_files.update({output.resolve(), provenance_path.resolve()})
        for name, contract in artifacts.items():
            artifact_path = output.parent / str(name)
            if (
                not artifact_path.is_file()
                or contract.get("sha256") != file_sha256(artifact_path)
                or int(contract.get("size_bytes", -1)) != artifact_path.stat().st_size
            ):
                raise ValueError("Physical A2 intervention artifact hash is invalid.")
            expected_files.add(artifact_path.resolve())
        expected_identity_hash = next(iter(sample_identity_hashes))
        scenario_predictions: dict[str, pd.DataFrame] = {}
        for scenario in scenarios:
            predictions_path = output.with_name(
                f"{output.stem}_{scenario}_predictions.csv"
            )
            predictions = pd.read_csv(predictions_path)
            if (
                len(predictions) != 240
                or predictions["sample_id"].astype(str).duplicated().any()
                or set(predictions["label"].astype(int)) != set(range(12))
                or not predictions.groupby(predictions["label"].astype(int)).size().eq(20).all()
                or not predictions["prediction"].astype(int).between(0, 11).all()
                or _sample_identity_sha256(predictions["sample_id"].astype(str))
                != expected_identity_hash
            ):
                raise ValueError(
                    f"Physical A2 intervention predictions are incomplete: {scenario}."
                )
            normalized = predictions[["sample_id", "label", "prediction"]].copy()
            normalized["sample_id"] = normalized["sample_id"].astype(str)
            normalized[["label", "prediction"]] = normalized[
                ["label", "prediction"]
            ].astype(int)
            scenario_predictions[scenario] = normalized.sort_values("sample_id").reset_index(
                drop=True
            )
        clean_identity = scenario_predictions["clean"][["sample_id", "label"]]
        for scenario, predictions in scenario_predictions.items():
            if not predictions[["sample_id", "label"]].equals(clean_identity):
                raise ValueError(
                    f"Physical intervention sample/label pairing changed in {scenario}."
                )
        main_field = "prediction_staging_csv" if use_staging else "prediction_csv"
        main_clean = pd.read_csv(full_job[main_field])[
            ["sample_id", "label", "prediction"]
        ].copy()
        main_clean["sample_id"] = main_clean["sample_id"].astype(str)
        main_clean[["label", "prediction"]] = main_clean[["label", "prediction"]].astype(int)
        main_clean = main_clean.sort_values("sample_id").reset_index(drop=True)
        if not main_clean.equals(scenario_predictions["clean"]):
            raise ValueError(
                "Full A2 clean prediction differs between the main and intervention bundles."
            )
        confusion = pd.read_csv(output.with_name(f"{output.stem}_confusion.csv"))
        if set(confusion["scenario"].astype(str)) != scenarios:
            raise ValueError("Physical A2 intervention confusion matrix is incomplete.")
        result_by_scenario = frame.set_index(frame["scenario"].astype(str), drop=False)
        provenance_by_scenario = {
            str(item["scenario"]): item for item in scenario_records
        }
        for scenario in donor_scenarios:
            donors_path = output.with_name(f"{output.stem}_{scenario}_donors.csv")
            donors = pd.read_csv(donors_path)
            result = result_by_scenario.loc[scenario]
            audit = provenance_by_scenario[scenario].get("metadata_control") or {}
            changed_fraction = float(
                result["metadata_control_effective_changed_fraction"]
            )
            expected_control, expected_scope = _expected_intervention_control_contract(
                scenario
            )
            rebuilt_control = _rebuild_donor_control_contract(
                asset_manifest,
                donors,
                control=expected_control,
                seed=int(intervention["config"]["seed"]),
                observed_mapping_sha256=file_sha256(donors_path),
            )
            expected_mapping_sha256 = str(rebuilt_control["donor_mapping_sha256"])
            expected_changed_fraction = float(
                rebuilt_control["effective_external_metadata_changed_fraction"]
            )
            if (
                len(donors) != 240
                or donors["sample_id"].astype(str).duplicated().any()
                or donors["donor_sample_id"].astype(str).duplicated().any()
                or (donors["sample_id"].astype(str) == donors["donor_sample_id"].astype(str)).any()
                or not donors["target_label"].astype(int).equals(
                    donors["donor_label"].astype(int)
                )
                or set(donors["sample_id"].astype(str)) != set(clean_identity["sample_id"])
                or result["metadata_control_name"] != expected_control
                or result["metadata_control_scope"] != expected_scope
                or int(result["metadata_control_seed"]) != int(intervention["config"]["seed"])
                or result["metadata_control_donor_mapping_sha256"]
                != file_sha256(donors_path)
                or file_sha256(donors_path) != expected_mapping_sha256
                or audit.get("metadata_control_name") != expected_control
                or audit.get("metadata_control_scope") != expected_scope
                or int(audit.get("metadata_control_seed", -1))
                != int(intervention["config"]["seed"])
                or audit.get("metadata_control_donor_mapping_sha256")
                != file_sha256(donors_path)
                or float(audit.get("metadata_control_effective_changed_fraction", -1.0))
                != changed_fraction
                or changed_fraction != expected_changed_fraction
                or not math.isfinite(changed_fraction)
                or not 0.0 <= changed_fraction <= 1.0
            ):
                raise ValueError(
                    f"Physical A2 intervention donor integrity failed: {scenario}."
                )
            if scenario == "joint_wet_dry_counterfactual":
                condition_flip = float(
                    result["metadata_control_condition_flip_fraction"]
                )
                expected_condition_flip = float(
                    rebuilt_control["condition_flip_fraction"]
                )
                if (
                    not math.isfinite(condition_flip)
                    or not 0.0 <= condition_flip <= 1.0
                    or float(
                        audit.get("metadata_control_condition_flip_fraction", -1.0)
                    )
                    != condition_flip
                    or condition_flip != expected_condition_flip
                ):
                    raise ValueError("Physical counterfactual condition-flip potency failed.")
    if len(manifest.get("interventions") or []) != 3 or expected_scenarios is None:
        raise ValueError("Physical A2 intervention matrix must contain exactly three folds.")
    directory = Path(manifest["interventions"][0][field]).parent
    observed_files = {path.resolve() for path in directory.iterdir() if path.is_file()}
    if observed_files != expected_files:
        raise ValueError("Physical intervention directory has missing or undeclared artifacts.")


def _expected_intervention_control_contract(scenario: str) -> tuple[str, str]:
    contracts = {
        "block_coherent_metadata_shuffle": (
            "within_class_shuffle",
            "within_dataset_block_derangement_label_aligned",
        ),
        "joint_wet_dry_counterfactual": (
            "counterfactual_wet_dry",
            "same_dataset_subject_label_block_window_opposite_electrode",
        ),
    }
    try:
        return contracts[scenario]
    except KeyError as exc:
        raise ValueError(f"Unknown physical donor scenario: {scenario}.") from exc


def _rebuild_donor_control_contract(
    asset_manifest: pd.DataFrame,
    donors: pd.DataFrame,
    *,
    control: str,
    seed: int,
    observed_mapping_sha256: str,
) -> dict:
    """Recreate one donor map from canonical rows instead of trusting its claims."""

    if "sample_id" not in asset_manifest or asset_manifest["sample_id"].astype(str).duplicated().any():
        raise ValueError("Canonical physical asset sample identities are invalid.")
    if "sample_id" not in donors or "donor_sample_id" not in donors:
        raise ValueError("Physical donor mapping lacks sample identities.")
    sample_ids = asset_manifest["sample_id"].astype(str)
    lookup = pd.Series(range(len(asset_manifest)), index=sample_ids)
    target_ids = donors["sample_id"].astype(str)
    donor_ids = donors["donor_sample_id"].astype(str)
    positions = lookup.reindex(target_ids)
    if positions.isna().any() or lookup.reindex(donor_ids).isna().any():
        raise ValueError("Physical donor mapping references a non-canonical sample.")
    rebuilt = build_development_control_plan(
        asset_manifest,
        {"val": positions.astype(int).to_numpy()},
        control=control,
        seed=int(seed),
    )
    if rebuilt.contract.get("donor_mapping_sha256") != observed_mapping_sha256:
        raise ValueError("Physical donor mapping does not match canonical reconstruction.")
    return rebuilt.contract


def _aggregate_intervention_results(
    manifest: dict, *, use_staging: bool = False
) -> tuple[pd.DataFrame, dict]:
    rows: list[pd.DataFrame] = []
    for intervention in manifest["interventions"]:
        field = "staging_output_csv" if use_staging else "output_csv"
        frame = pd.read_csv(intervention[field])
        frame.insert(0, "fold_index", int(intervention["fold_index"]))
        frame.insert(0, "job_id", str(intervention["job_id"]))
        rows.append(frame)
    combined = pd.concat(rows, ignore_index=True).sort_values(
        ["scenario", "fold_index"]
    )
    means = (
        combined.groupby("scenario", sort=True)["balanced_accuracy"].mean().astype(float).to_dict()
    )
    clean = float(means["clean"])
    deltas = {
        f"clean_minus_{scenario}": clean - float(value)
        for scenario, value in means.items()
        if scenario != "clean"
    }
    summary = {
        "status": "descriptive_three_subject_safety_assay_only",
        "population_inference_allowed": False,
        "mean_balanced_accuracy": means,
        "clean_minus_intervention_delta": deltas,
    }
    return combined.reset_index(drop=True), summary


def _training_control_rows(manifest: dict) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for job in manifest["jobs"]:
        control_path = Path(job["output_dir"]) / "development_control.json"
        control = json.loads(control_path.read_text(encoding="utf-8"))
        rows.append(
            {
                "job_id": job["job_id"],
                "fold_index": int(job["fold_index"]),
                "role": str(job["role"]),
                "control_name": control.get("name"),
                "effective_changed_fraction": control.get(
                    "effective_external_metadata_changed_fraction"
                ),
                "condition_flip_fraction": control.get("condition_flip_fraction"),
                "donor_mapping_sha256": control.get("donor_mapping_sha256"),
            }
        )
    return pd.DataFrame(rows)


def _private_staging_authorization(manifest: dict, *, manifest_path: Path) -> dict:
    receipt_path = _repository_path(str(manifest["decision_receipt"]))
    if (
        receipt_path != CANONICAL_DECISION_RECEIPT.resolve()
        or not receipt_path.is_file()
        or file_sha256(receipt_path) != manifest["decision_receipt_sha256"]
    ):
        raise ValueError("Physical private-staging decision receipt is missing or stale.")
    return {
        "grid_manifest": str(manifest_path),
        "decision_receipt": str(receipt_path),
        "decision_receipt_sha256": manifest["decision_receipt_sha256"],
        "reveal_index": str(manifest["outcome_reveal_index"]),
        "authorization_phase": "private_complete_bundle_staging",
    }


def _write_staged_analysis(
    staging_dir: Path,
    *,
    subjects: pd.DataFrame,
    interventions: pd.DataFrame,
    summary: dict,
) -> None:
    analysis_dir = staging_dir / "analysis"
    temporary = staging_dir / ".analysis-build"
    if temporary.exists():
        shutil.rmtree(temporary)
    temporary.mkdir(mode=0o700)
    subjects.to_csv(temporary / "revealed_subjects.csv", index=False)
    interventions.to_csv(temporary / "revealed_interventions.csv", index=False)
    _atomic_save_json(temporary / "revealed_summary.json", summary)
    if analysis_dir.exists():
        if _directory_file_contract(analysis_dir) != _directory_file_contract(temporary):
            raise ValueError("Existing private physical analysis differs from recomputation.")
        shutil.rmtree(temporary)
        return
    os.replace(temporary, analysis_dir)


def _directory_file_contract(directory: Path) -> dict[str, dict[str, object]]:
    return {
        path.relative_to(directory).as_posix(): {
            "sha256": file_sha256(path),
            "size_bytes": path.stat().st_size,
        }
        for path in sorted(directory.rglob("*"))
        if path.is_file()
    }


def _bundle_manifest_payload(bundle: Path, manifest: dict) -> dict:
    if bundle.is_symlink():
        raise ValueError("Physical reveal bundle root cannot be a symlink.")
    files: dict[str, dict[str, object]] = {}
    manifest_path = bundle / "bundle_manifest.json"
    allowed_directories = {
        bundle / "predictions",
        bundle / "interventions",
        bundle / "analysis",
    }
    for path in sorted(bundle.rglob("*")):
        if path.is_symlink():
            raise ValueError("Physical reveal bundle cannot contain symlinks.")
        if path.is_dir():
            if path not in allowed_directories:
                raise ValueError("Physical reveal bundle cannot contain nested directories.")
            continue
        if not path.is_file():
            raise ValueError("Physical reveal bundle contains a non-regular artifact.")
        if path == manifest_path:
            continue
        relative = path.relative_to(bundle).as_posix()
        files[relative] = {
            "sha256": file_sha256(path),
            "size_bytes": path.stat().st_size,
        }
    return {
        "schema": "cfeg.physical-reveal-bundle.v1",
        "grid_content_sha256": manifest["grid_content_sha256"],
        "decision_receipt_sha256": manifest["decision_receipt_sha256"],
        "reveal_index": manifest["outcome_reveal_index"],
        "files": files,
    }


def _write_bundle_manifest(bundle: Path, manifest: dict) -> str:
    path = bundle / "bundle_manifest.json"
    _cleanup_known_atomic_json_temps(path)
    if path.exists():
        return _validate_bundle_manifest(bundle, manifest)
    payload = _bundle_manifest_payload(bundle, manifest)
    payload["bundle_content_sha256"] = sha256_json(payload)
    _atomic_save_json(path, payload)
    return str(payload["bundle_content_sha256"])


def _validate_bundle_manifest(
    bundle: Path,
    manifest: dict,
    *,
    expected_digest: str | None = None,
) -> str:
    path = bundle / "bundle_manifest.json"
    if not path.is_file():
        raise ValueError("Physical reveal bundle manifest is missing.")
    observed = json.loads(path.read_text(encoding="utf-8"))
    recorded = observed.pop("bundle_content_sha256", None)
    expected = _bundle_manifest_payload(bundle, manifest)
    digest = sha256_json(expected)
    if observed != expected or recorded != digest or (
        expected_digest is not None and digest != expected_digest
    ):
        raise ValueError("Physical reveal bundle tree hash is invalid or stale.")
    return digest


def _validate_complete_publication(manifest: dict, *, root: Path) -> None:
    bundle = root / "reveal-bundle"
    _validate_reveal_bundle_layout(manifest, use_staging=False)
    bundle_digest = _validate_bundle_manifest(bundle, manifest)
    validate_predictions_complete(manifest)
    validate_interventions_complete(manifest)
    analysis_dir = bundle / "analysis"
    summary = json.loads((analysis_dir / "revealed_summary.json").read_text(encoding="utf-8"))
    if (
        summary.get("grid_manifest_sha256") != manifest["grid_content_sha256"]
        or summary.get("decision_receipt_sha256") != manifest["decision_receipt_sha256"]
        or summary.get("reveal_index") != manifest["outcome_reveal_index"]
        or not isinstance(summary.get("predeclared_gate_evaluation"), dict)
    ):
        raise ValueError("Physical reveal analysis summary is incomplete or stale.")
    _validate_published_reveal_receipt(manifest, root=root, bundle_digest=bundle_digest)


def _validate_reveal_bundle_layout(manifest: dict, *, use_staging: bool) -> None:
    root = Path(manifest["canonical_output_root"])
    bundle = root / (".reveal-staging" if use_staging else "reveal-bundle")
    if root.is_symlink() or bundle.is_symlink():
        raise ValueError("Physical reveal root and bundle paths cannot be symlinks.")
    expected = {
        (bundle / "predictions").resolve(),
        (bundle / "interventions").resolve(),
        (bundle / "analysis").resolve(),
        (bundle / "bundle_manifest.json").resolve(),
    }
    observed = {path.resolve() for path in bundle.iterdir()}
    expected_dirs = expected - {(bundle / "bundle_manifest.json").resolve()}
    if (
        observed != expected
        or any(not path.is_dir() for path in expected_dirs)
        or any(path.is_symlink() for path in bundle.iterdir())
        or any(
            child.is_symlink() or not child.is_file()
            for directory in expected_dirs
            for child in directory.iterdir()
        )
        or not (bundle / "bundle_manifest.json").is_file()
        or {
            path.name for path in (bundle / "analysis").iterdir() if path.is_file()
        }
        != {
            "revealed_subjects.csv",
            "revealed_interventions.csv",
            "revealed_summary.json",
        }
    ):
        raise ValueError("Physical reveal bundle has undeclared top-level artifacts.")


def _sample_identity_sha256(sample_ids) -> str:
    digest = hashlib.sha256()
    for sample_id in sorted(str(value) for value in sample_ids):
        encoded = sample_id.encode("utf-8")
        digest.update(len(encoded).to_bytes(8, byteorder="big", signed=False))
        digest.update(encoded)
    return digest.hexdigest()


@contextmanager
def _exclusive_reveal_publication(root: Path):
    """Serialize ledger registration, staging writes, and atomic publication."""

    if root.is_symlink():
        raise ValueError("Physical reveal root cannot be a symlink.")
    root.mkdir(parents=True, exist_ok=True)
    lock_path = root / ".reveal-publication.lock"
    if lock_path.is_symlink():
        raise ValueError("Physical reveal publication lock cannot be a symlink.")
    with lock_path.open("a+", encoding="utf-8") as lock:
        lock_path.chmod(0o600)
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(lock.fileno(), fcntl.LOCK_UN)


def _preflight_global_reveal(
    manifest: dict,
    *,
    root: Path,
    allow_receipt_recovery: bool = True,
) -> str | None:
    historical = _load_reveal_ledger(Path(manifest["historical_reveal_ledger"]))
    global_path = Path(manifest["global_reveal_ledger"])
    if global_path.is_symlink() or (global_path.exists() and not global_path.is_file()):
        raise ValueError("Physical global reveal ledger path is invalid.")
    global_ledger = (
        _load_reveal_ledger(global_path)
        if global_path.exists()
        else {"schema": "cfeg.development-reveal-ledger.v1", "entries": []}
    )
    combined = [*historical["entries"], *global_ledger["entries"]]
    indices = [int(item["reveal_index"]) for item in combined]
    if sorted(indices) != list(range(1, len(indices) + 1)) or len(indices) != len(set(indices)):
        raise ValueError("S1-S3 reveal ledger indices are not unique and contiguous.")
    existing = [
        item
        for item in combined
        if item.get("grid_content_sha256") == manifest["grid_content_sha256"]
    ]
    if len(existing) > 1:
        raise ValueError("Physical grid appears more than once in the reveal ledger.")
    if existing:
        entry = existing[0]
        previous = [
            item
            for item in combined
            if int(item["reveal_index"]) == int(entry["reveal_index"]) - 1
        ]
        bundle_digest = str(entry.get("bundle_content_sha256") or "")
        if len(previous) != 1:
            raise ValueError("Precommitted physical reveal has no unique predecessor.")
        _validate_physical_reveal_entry(
            entry,
            manifest,
            previous_event_sha256=_reveal_event_sha256(previous[0]),
            bundle_content_sha256=bundle_digest,
        )
        _write_or_validate_precommit_reveal_receipt(
            manifest,
            root=root,
            entry=entry,
            allow_create=allow_receipt_recovery,
        )
        final_receipt = root / "reveal_receipt.json"
        if (final_receipt.exists() or final_receipt.is_symlink()) and not (
            root / "reveal-bundle"
        ).is_dir():
            raise ValueError(
                "Physical final reveal receipt exists before public bundle publication."
            )
        return bundle_digest
    precommit_receipt = root / "reveal_precommit_receipt.json"
    final_receipt = root / "reveal_receipt.json"
    if precommit_receipt.exists() or precommit_receipt.is_symlink():
        raise ValueError("Physical precommit receipt has no matching reveal ledger event.")
    if final_receipt.exists() or final_receipt.is_symlink():
        raise ValueError("Physical final reveal receipt has no matching reveal ledger event.")
    expected_index = len(combined) + 1
    if expected_index != int(manifest["outcome_reveal_index"]):
        raise ValueError(
            f"Physical reveal index must be {manifest['outcome_reveal_index']}; "
            f"global history implies {expected_index}."
        )
    if expected_index > int(manifest["recommended_max_outcome_reveals"]):
        raise ValueError("Recommended S1-S3 outcome-reveal budget is exhausted.")
    if expected_index > int(manifest["absolute_max_outcome_reveals"]):
        raise ValueError("Absolute S1-S3 outcome-reveal budget is exhausted.")
    return None


def register_global_reveal(
    manifest: dict,
    root: Path,
    *,
    bundle_path: Path,
    bundle_content_sha256: str,
) -> dict:
    expected_private_bundle = (root / ".reveal-staging").resolve()
    if (
        not bundle_path.is_symlink()
        and bundle_path.resolve() == expected_private_bundle
        and bundle_path.is_dir()
    ):
        _validate_reveal_bundle_layout(manifest, use_staging=True)
    if (
        bundle_path.resolve() != expected_private_bundle
        or bundle_path.is_symlink()
        or not bundle_path.is_dir()
        or _validate_bundle_manifest(bundle_path, manifest) != bundle_content_sha256
    ):
        raise ValueError("The complete canonical private bundle is required for precommit.")
    historical = _load_reveal_ledger(Path(manifest["historical_reveal_ledger"]))
    global_path = Path(manifest["global_reveal_ledger"])
    global_path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = global_path.with_suffix(f"{global_path.suffix}.lock")
    with lock_path.open("a+", encoding="utf-8") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        global_ledger = (
            _load_reveal_ledger(global_path)
            if global_path.exists()
            else {"schema": "cfeg.development-reveal-ledger.v1", "entries": []}
        )
        grid_hash = manifest["grid_content_sha256"]
        combined = [*historical["entries"], *global_ledger["entries"]]
        indices = [int(item["reveal_index"]) for item in combined]
        if sorted(indices) != list(range(1, len(indices) + 1)) or len(indices) != len(
            set(indices)
        ):
            raise ValueError("S1-S3 reveal ledger indices are not unique and contiguous.")
        existing = [item for item in combined if item.get("grid_content_sha256") == grid_hash]
        if len(existing) > 1:
            raise ValueError("Physical grid appears more than once in the reveal ledger.")
        if existing:
            entry = existing[0]
            entry_index = int(entry["reveal_index"])
            previous_entry = next(
                item for item in combined if int(item["reveal_index"]) == entry_index - 1
            )
            _validate_physical_reveal_entry(
                entry,
                manifest,
                previous_event_sha256=_reveal_event_sha256(previous_entry),
                bundle_content_sha256=bundle_content_sha256,
            )
        else:
            expected_index = len(combined) + 1
            if expected_index != int(manifest["outcome_reveal_index"]):
                raise ValueError(
                    f"Physical reveal index must be {manifest['outcome_reveal_index']}; "
                    f"global history implies {expected_index}."
                )
            if expected_index > int(manifest["recommended_max_outcome_reveals"]):
                raise ValueError("Recommended S1-S3 outcome-reveal budget is exhausted.")
            if expected_index > int(manifest["absolute_max_outcome_reveals"]):
                raise ValueError("Absolute S1-S3 outcome-reveal budget is exhausted.")
            previous_event_sha256 = _reveal_event_sha256(combined[-1])
            entry = {
                "reveal_index": expected_index,
                "grid_content_sha256": grid_hash,
                "grid_id": manifest["grid_id"],
                "registered_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
                "event_type": "reveal_budget_precommit",
                "publication_state": "complete_private_bundle_precommitted",
                "scope": (
                    "complete-18-clean-plus-3-intervention-plus-predeclared-analysis-bundle"
                ),
                "publication_protocol": (
                    "complete_private_bundle_digest_precommit_then_atomic_directory_rename"
                ),
                "confirmatory_claim_allowed": False,
                "population_inference_allowed": False,
                "decision_receipt_sha256": manifest["decision_receipt_sha256"],
                "bundle_content_sha256": bundle_content_sha256,
                "historical_reveal_ledger_sha256": manifest[
                    "historical_reveal_ledger_sha256"
                ],
                "previous_reveal_event_sha256": previous_event_sha256,
            }
            entry["reveal_event_sha256"] = _reveal_event_sha256(entry)
            global_ledger["entries"].append(entry)
            _atomic_save_json(global_path, global_ledger)
        _write_or_validate_precommit_reveal_receipt(
            manifest,
            root=root,
            entry=entry,
            allow_create=True,
        )
    return entry


def _validate_physical_reveal_entry(
    entry: dict,
    manifest: dict,
    *,
    previous_event_sha256: str,
    bundle_content_sha256: str,
) -> None:
    expected = {
        "reveal_index": int(manifest["outcome_reveal_index"]),
        "grid_content_sha256": manifest["grid_content_sha256"],
        "grid_id": manifest["grid_id"],
        "event_type": "reveal_budget_precommit",
        "publication_state": "complete_private_bundle_precommitted",
        "scope": "complete-18-clean-plus-3-intervention-plus-predeclared-analysis-bundle",
        "publication_protocol": (
            "complete_private_bundle_digest_precommit_then_atomic_directory_rename"
        ),
        "confirmatory_claim_allowed": False,
        "population_inference_allowed": False,
        "decision_receipt_sha256": manifest["decision_receipt_sha256"],
        "bundle_content_sha256": bundle_content_sha256,
        "historical_reveal_ledger_sha256": manifest["historical_reveal_ledger_sha256"],
        "previous_reveal_event_sha256": previous_event_sha256,
    }
    if (
        any(entry.get(key) != value for key, value in expected.items())
        or not _is_sha256_hex(bundle_content_sha256)
        or not _is_utc_timestamp(entry.get("registered_at_utc"))
        or entry.get("reveal_event_sha256") != _reveal_event_sha256(entry)
    ):
        raise ValueError("Existing physical reveal ledger entry is invalid or forged.")


def _validated_physical_reveal_event(
    manifest: dict, *, bundle_digest: str
) -> dict:
    historical = _load_reveal_ledger(Path(manifest["historical_reveal_ledger"]))
    global_ledger = _load_reveal_ledger(Path(manifest["global_reveal_ledger"]))
    combined = [*historical["entries"], *global_ledger["entries"]]
    matches = [
        entry
        for entry in combined
        if entry.get("grid_content_sha256") == manifest["grid_content_sha256"]
    ]
    if len(matches) != 1:
        raise ValueError("Physical reveal ledger has no unique published event.")
    entry = matches[0]
    previous = [
        item
        for item in combined
        if int(item["reveal_index"]) == int(entry["reveal_index"]) - 1
    ]
    if len(previous) != 1:
        raise ValueError("Physical reveal ledger has no unique predecessor.")
    _validate_physical_reveal_entry(
        entry,
        manifest,
        previous_event_sha256=_reveal_event_sha256(previous[0]),
        bundle_content_sha256=bundle_digest,
    )
    return entry


def _precommit_reveal_receipt_payload(
    manifest: dict,
    *,
    entry: dict,
) -> dict:
    return {
        "schema": "cfeg.physical-mechanism-reveal-precommit.v1",
        **entry,
        "global_reveal_ledger": str(Path(manifest["global_reveal_ledger"])),
    }


def _write_or_validate_precommit_reveal_receipt(
    manifest: dict,
    *,
    root: Path,
    entry: dict,
    allow_create: bool,
) -> None:
    expected = _precommit_reveal_receipt_payload(manifest, entry=entry)
    receipt_path = root / "reveal_precommit_receipt.json"
    if receipt_path.is_symlink():
        raise ValueError("Physical reveal precommit receipt cannot be a symlink.")
    if receipt_path.exists():
        if json.loads(receipt_path.read_text(encoding="utf-8")) != expected:
            raise ValueError(
                "Physical reveal precommit receipt is immutable and does not match the event."
            )
        return
    if not allow_create:
        raise ValueError("Physical reveal precommit receipt is missing.")
    _atomic_save_json(receipt_path, expected)


def _validate_precommit_reveal_receipt(
    manifest: dict, *, root: Path, bundle_digest: str
) -> dict:
    entry = _validated_physical_reveal_event(
        manifest,
        bundle_digest=bundle_digest,
    )
    _write_or_validate_precommit_reveal_receipt(
        manifest,
        root=root,
        entry=entry,
        allow_create=False,
    )
    return entry


def _finalize_reveal_publication(
    manifest: dict, *, root: Path, bundle_digest: str
) -> None:
    public_bundle = root / "reveal-bundle"
    if not public_bundle.is_dir() or (root / ".reveal-staging").exists():
        raise ValueError(
            "Physical publication can be finalized only after the private bundle rename."
        )
    _validate_reveal_bundle_layout(manifest, use_staging=False)
    if _validate_bundle_manifest(public_bundle, manifest) != bundle_digest:
        raise ValueError("Published physical bundle differs from its precommitted digest.")
    entry = _validate_precommit_reveal_receipt(
        manifest,
        root=root,
        bundle_digest=bundle_digest,
    )
    receipt_path = root / "reveal_receipt.json"
    if receipt_path.is_symlink():
        raise ValueError("Physical final reveal receipt cannot be a symlink.")
    if receipt_path.exists():
        _validate_published_reveal_receipt(
            manifest,
            root=root,
            bundle_digest=bundle_digest,
        )
        return
    receipt = {
        "schema": "cfeg.physical-mechanism-reveal-receipt.v4",
        **entry,
        "global_reveal_ledger": str(Path(manifest["global_reveal_ledger"])),
        "public_bundle": str(public_bundle.resolve()),
        "final_publication_state": "public_bundle_published",
        "publication_completed_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
    }
    _atomic_save_json(receipt_path, receipt)


def _validate_published_reveal_receipt(
    manifest: dict, *, root: Path, bundle_digest: str
) -> None:
    public_bundle = root / "reveal-bundle"
    if not public_bundle.is_dir():
        raise ValueError("Physical public reveal bundle is missing.")
    if _validate_bundle_manifest(public_bundle, manifest) != bundle_digest:
        raise ValueError("Physical public reveal bundle digest is invalid.")
    entry = _validate_precommit_reveal_receipt(
        manifest,
        root=root,
        bundle_digest=bundle_digest,
    )
    receipt_path = root / "reveal_receipt.json"
    if receipt_path.is_symlink():
        raise ValueError("Physical final reveal receipt cannot be a symlink.")
    if not receipt_path.is_file():
        raise ValueError("Physical final reveal receipt is missing.")
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    timestamp = receipt.get("publication_completed_at_utc")
    expected = {
        "schema": "cfeg.physical-mechanism-reveal-receipt.v4",
        **entry,
        "global_reveal_ledger": str(Path(manifest["global_reveal_ledger"])),
        "public_bundle": str(public_bundle.resolve()),
        "final_publication_state": "public_bundle_published",
        "publication_completed_at_utc": timestamp,
    }
    if receipt != expected or not _is_utc_timestamp(timestamp):
        raise ValueError("Physical final reveal receipt is invalid or stale.")


def _is_utc_timestamp(value: object) -> bool:
    try:
        parsed = dt.datetime.fromisoformat(str(value))
    except (TypeError, ValueError):
        return False
    return parsed.tzinfo is not None and parsed.utcoffset() == dt.timedelta(0)


def _is_sha256_hex(value: object) -> bool:
    normalized = str(value)
    return len(normalized) == 64 and all(
        character in "0123456789abcdef" for character in normalized
    )


def _reveal_event_sha256(entry: dict) -> str:
    event = dict(entry)
    event.pop("reveal_event_sha256", None)
    return sha256_json(event)


def _atomic_save_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    _cleanup_known_atomic_json_temps(path)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    encoded = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8")
    with temporary.open("wb") as handle:
        handle.write(encoded)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)
    directory_fd = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)


def _cleanup_known_atomic_json_temps(path: Path) -> None:
    """Remove only stale temp names belonging to one known private JSON artifact."""

    for pattern in (f".{path.name}.tmp-*", f".{path.name}.*.tmp"):
        for temporary in path.parent.glob(pattern):
            if temporary.is_file():
                temporary.unlink()


def _fsync_directory(path: Path) -> None:
    directory_fd = os.open(path, os.O_RDONLY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)


def _fsync_bundle_tree(bundle: Path) -> None:
    """Durably flush a validated private bundle before reveal-budget precommit."""

    if bundle.is_symlink():
        raise ValueError("Physical reveal bundle root cannot be a symlink.")
    directories = [bundle]
    for path in sorted(bundle.rglob("*")):
        if path.is_symlink():
            raise ValueError("Physical reveal bundle cannot contain symlinks.")
        if path.is_dir():
            directories.append(path)
            continue
        if not path.is_file():
            raise ValueError("Physical reveal bundle contains a non-regular artifact.")
        file_descriptor = os.open(path, os.O_RDONLY)
        try:
            os.fsync(file_descriptor)
        finally:
            os.close(file_descriptor)
    for directory in sorted(directories, key=lambda item: len(item.parts), reverse=True):
        _fsync_directory(directory)
    _fsync_directory(bundle.parent)


def resolve_processed_dir(cfg: dict) -> str:
    value = os.path.expandvars(str(cfg["data"]["processed_dirs"][0]))
    if "${env:" in value or not Path(value).is_dir():
        raise ValueError("EEG_DATA_ROOT must resolve the audited wearable_v3 processed asset.")
    return value


def print_status(manifest: dict, root: Path) -> None:
    trained = sum(
        (Path(job["output_dir"]) / "training_completion.json").is_file() for job in manifest["jobs"]
    )
    predicted = sum(Path(job["prediction_csv"]).is_file() for job in manifest["jobs"])
    intervened = sum(
        Path(item["output_csv"]).is_file() for item in manifest.get("interventions") or []
    )
    precommitted_digest = _preflight_global_reveal(
        manifest,
        root=root,
        allow_receipt_recovery=False,
    )
    reveal_budget_consumed = precommitted_digest is not None
    public_bundle = root / "reveal-bundle"
    final_receipt = root / "reveal_receipt.json"
    public_present = public_bundle.exists() or public_bundle.is_symlink()
    receipt_present = final_receipt.exists() or final_receipt.is_symlink()
    if public_present != receipt_present:
        raise ValueError("Physical public bundle/final receipt state is incomplete.")
    if public_present and (
        public_bundle.is_symlink()
        or final_receipt.is_symlink()
        or not public_bundle.is_dir()
        or not final_receipt.is_file()
    ):
        raise ValueError("Physical public bundle/final receipt types are invalid.")
    public_bundle_published = public_present
    if public_bundle_published:
        _validate_complete_publication(manifest, root=root)
    print(
        json.dumps(
            {
                "design_status": manifest["design_status"],
                "trained": trained,
                "expected_training_jobs": manifest["expected_job_count"],
                "predicted": predicted,
                "intervention_bundles": intervened,
                "expected_intervention_bundles": manifest.get("expected_intervention_count"),
                "reveal_budget_consumed": reveal_budget_consumed,
                "outcomes_revealed": public_bundle_published,
                "public_bundle_published": public_bundle_published,
                "outcome_reveal_index": manifest["outcome_reveal_index"],
                "canonical_root": str(root),
            },
            sort_keys=True,
        )
    )


def _load_reveal_ledger(path: Path) -> dict:
    ledger = json.loads(path.read_text(encoding="utf-8"))
    if ledger.get("schema") != "cfeg.development-reveal-ledger.v1" or not isinstance(
        ledger.get("entries"), list
    ):
        raise ValueError(f"Invalid S1-S3 reveal ledger: {path}.")
    indices: list[int] = []
    hashes: list[str] = []
    for entry in ledger["entries"]:
        try:
            index = int(entry["reveal_index"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"Invalid S1-S3 reveal entry in {path}.") from exc
        digest = str(entry.get("grid_content_sha256") or "")
        if index < 1 or len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
            raise ValueError(f"Invalid S1-S3 reveal entry in {path}.")
        indices.append(index)
        hashes.append(digest)
    if len(indices) != len(set(indices)) or len(hashes) != len(set(hashes)):
        raise ValueError(f"Duplicate S1-S3 reveal entry in {path}.")
    return ledger


def _repository_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (REPO / path).resolve()


def _reject_symlink_components(path: Path) -> None:
    """Reject lexical path aliases before callers erase them with resolve()."""

    absolute = Path(os.path.abspath(path))
    current = Path(absolute.anchor)
    for component in absolute.parts[1:]:
        current /= component
        if current.is_symlink():
            raise ValueError(f"Physical canonical output path contains a symlink: {current}.")


def file_sha256(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


if __name__ == "__main__":
    main()
