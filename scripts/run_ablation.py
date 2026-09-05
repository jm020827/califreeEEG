#!/usr/bin/env python
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import tempfile
import traceback
from pathlib import Path

import pandas as pd
from _bootstrap import add_src_to_path

add_src_to_path()

from cfeg.analysis.synthetic_reliability_stage1 import (
    finalize_synthetic_reliability_stage1,
    validate_synthetic_reliability_stage1_publication_inputs,
)
from cfeg.execution_manifest import validate_primary_pair_and_hash
from cfeg.governance import (
    GovernanceError,
    current_source_revision_contract,
    implementation_contract_sha256,
)
from cfeg.reliability_contract import (
    RELIABILITY_ROLES,
    validate_reliability_family,
    validate_reliability_runtime_family,
    validate_reliability_stage0_gate,
)
from cfeg.train_loop import _RELIABILITY_ORCHESTRATOR_AUTHORIZATION, run_training
from cfeg.utils.config import load_config, merge_overrides

REPO = Path(__file__).resolve().parents[1]
CANONICAL_RELIABILITY_SUITE = REPO / "configs/train/synthetic_reliability_2x2.yaml"
CANONICAL_RELIABILITY_BASE = REPO / "configs/train/synthetic_reliability_candidate.yaml"
CANONICAL_RELIABILITY_STAGE1_ROOT = (
    REPO / "outputs/engineering/reliability-spatial-v1/stage1"
)
CANONICAL_RELIABILITY_DRY_RUN_PARENT = (
    REPO / "outputs/dry-runs/reliability-spatial-v1"
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Execute A0-A5 ablation training variants.")
    parser.add_argument("--config", default="configs/train/ablation.yaml")
    parser.add_argument("--base-config", default=None)
    parser.add_argument("--output-root", default="outputs/ablation")
    parser.add_argument(
        "--override",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Override the base config before applying each variant (repeatable).",
    )
    parser.add_argument("--only", default=None, help="Comma-separated variant names.")
    parser.add_argument("--include-optional", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--resume-exact", action="store_true")
    parser.add_argument("--continue-on-error", action="store_true")
    args = parser.parse_args()

    suite = load_config(args.config, strict_env=False)
    base_config = args.base_config or os.environ.get(
        "CFEG_ABLATION_BASE_CONFIG", suite["base_config"]
    )
    if suite.get("reliability_family") is not None:
        _validate_reliability_orchestrator_args(
            args,
            base_config=base_config,
            dry_run=args.dry_run,
        )
    base = merge_overrides(load_config(base_config, strict_env=False), args.override)
    _validate_development_grid_execution(base, dry_run=args.dry_run)
    _apply_runtime_environment(base)
    primary_fairness_hash = validate_primary_pair_and_hash(
        base,
        suite.get("variants", {}),
        enforce_frozen_architecture=_requires_frozen_primary_architecture(base),
    )
    control_family_hash = _validate_development_control_family(base, suite)
    mechanism_family_hash = _validate_physical_mechanism_family(base, suite)
    reliability_family_hash = validate_reliability_family(base, suite)
    reliability_stage0_receipt_sha256 = (
        validate_reliability_stage0_gate(base, require_receipt=not args.dry_run)
        if reliability_family_hash is not None
        else None
    )
    canonical_output_root = Path(args.output_root)
    artifact_output_root = canonical_output_root
    reliability_staging_root: Path | None = None
    reliability_reservation: Path | None = None
    if reliability_family_hash is not None and not args.dry_run:
        canonical_output_root.parent.mkdir(parents=True, exist_ok=True)
        reliability_reservation = _reserve_reliability_stage1_execution(
            canonical_output_root,
            contract=_reliability_attempt_contract(
                base,
                reliability_family_hash=reliability_family_hash,
                stage0_receipt_sha256=reliability_stage0_receipt_sha256,
            ),
        )
        reliability_staging_root = Path(
            tempfile.mkdtemp(
                prefix=f".{canonical_output_root.name}.staging-",
                dir=canonical_output_root.parent,
            )
        )
        reliability_staging_root.chmod(0o700)
        artifact_output_root = reliability_staging_root
    selected = set(args.only.split(",")) if args.only else None
    optional = set(suite.get("optional_variants", []))
    rows = []
    failed = False
    for name, overrides in suite.get("variants", {}).items():
        if selected is not None and name not in selected:
            continue
        if name in optional and not args.include_optional:
            rows.append({"variant": name, "status": "skipped_optional"})
            continue
        cfg = merge_overrides(base, [f"{key}={value!r}" for key, value in overrides.items()])
        cfg["run_name"] = name
        cfg["output_dir"] = str(Path(args.output_root) / name)
        if control_family_hash is not None:
            cfg.setdefault("protocol", {})["development_control_family_sha256"] = (
                control_family_hash
            )
        if mechanism_family_hash is not None:
            cfg.setdefault("protocol", {})["physical_mechanism_family_sha256"] = (
                mechanism_family_hash
            )
        if reliability_family_hash is not None:
            cfg.setdefault("protocol", {})["reliability_mechanism_family_sha256"] = (
                reliability_family_hash
            )
            if name in RELIABILITY_ROLES:
                cfg["protocol"]["reliability_family_role"] = name
                cfg["protocol"]["reliability_stage0_receipt_sha256"] = (
                    reliability_stage0_receipt_sha256
                )
        if name in {"A0_eeg_only", "A2_structured_condition_prompt"}:
            protocol_cfg = cfg.setdefault("protocol", {})
            protocol_cfg["primary_fairness_hash"] = primary_fairness_hash
            protocol_cfg["primary_ablation_role"] = name
        wandb_cfg = cfg.setdefault("tracking", {}).setdefault("wandb", {})
        tags = list(wandb_cfg.get("tags") or [])
        wandb_cfg["tags"] = sorted(set(tags + ["ablation", name]))
        try:
            result = run_training(
                cfg,
                dry_run=args.dry_run,
                resume_exact=args.resume_exact,
                _artifact_output_dir=(
                    artifact_output_root / name
                    if reliability_staging_root is not None
                    else None
                ),
                _reliability_orchestrator_authorization=(
                    _RELIABILITY_ORCHESTRATOR_AUTHORIZATION
                    if reliability_family_hash is not None
                    else None
                ),
            )
            runtime_metrics = result.get("runtime_metrics") or {}
            control = result.get("development_control") or {}
            row = {
                "variant": name,
                "seed": cfg.get("seed"),
                "split_seed": cfg.get("data", {}).get("split_seed", 42),
                "n_folds": cfg.get("data", {}).get("n_folds"),
                "fold_index": cfg.get("data", {}).get("fold_index"),
                "base_config": base_config,
                "candidate_id": cfg.get("protocol", {}).get("candidate_id"),
                "candidate_plan_sha256": cfg.get("protocol", {}).get(
                    "candidate_plan_sha256"
                ),
                "predecessor_retirement_receipt_sha256": cfg.get("protocol", {}).get(
                    "predecessor_retirement_receipt_sha256"
                ),
                "reliability_query_feature_schema": cfg.get("protocol", {}).get(
                    "reliability_query_feature_schema"
                ),
                "primary_fairness_hash": (
                    primary_fairness_hash
                    if name in {"A0_eeg_only", "A2_structured_condition_prompt"}
                    else None
                ),
                "reliability_mechanism_family_sha256": (
                    reliability_family_hash if name in RELIABILITY_ROLES else None
                ),
                "reliability_family_role": name if name in RELIABILITY_ROLES else None,
                "reliability_stage0_receipt_sha256": (
                    reliability_stage0_receipt_sha256
                    if name in RELIABILITY_ROLES
                    else None
                ),
                "query_qc_mode": cfg.get("model", {})
                .get("conditioning", {})
                .get("spatial_reliability", {})
                .get("query_qc_mode"),
                "external_metadata_mode": cfg.get("model", {})
                .get("condition_encoder", {})
                .get("external_metadata_mode"),
                "status": "dry_run" if args.dry_run else "completed",
                "best_validation_accuracy": result.get(
                    "best_validation_accuracy", result.get("best_accuracy")
                ),
                "total_parameters": (result.get("params") or {}).get("total"),
                "trainable_parameters": (result.get("params") or {}).get("trainable"),
                "parameter_schema_sha256": (result.get("runtime_contract") or {}).get(
                    "parameter_schema_sha256"
                ),
                "initial_trainable_state_sha256": (result.get("runtime_contract") or {}).get(
                    "initial_trainable_state_sha256"
                ),
                "split_assignment_sha256": (result.get("runtime_contract") or {}).get(
                    "split_assignment_sha256"
                ),
                "vocabulary_sha256": (result.get("runtime_contract") or {}).get(
                    "vocabulary_sha256"
                ),
                "asset_provenance_sha256": (result.get("runtime_contract") or {}).get(
                    "asset_provenance_sha256"
                ),
                "source_commit_sha": (result.get("runtime_contract") or {}).get(
                    "source_commit_sha"
                ),
                "source_dirty": (result.get("runtime_contract") or {}).get("source_dirty"),
                "source_tree_sha256": (result.get("runtime_contract") or {}).get(
                    "source_tree_sha256"
                ),
                "environment_sha256": (result.get("runtime_contract") or {}).get(
                    "environment_sha256"
                ),
                "execution_phase": (result.get("runtime_contract") or {}).get("execution_phase"),
                "analysis_plan_status": (result.get("runtime_contract") or {}).get(
                    "analysis_plan_status"
                ),
                "cohort_role": (result.get("runtime_contract") or {}).get("cohort_role"),
                "loader_settings_sha256": (result.get("runtime_contract") or {}).get(
                    "loader_settings_sha256"
                ),
                "analysis_plan_sha256": (result.get("runtime_contract") or {}).get(
                    "analysis_plan_sha256"
                ),
                "cohort_roles_sha256": (result.get("runtime_contract") or {}).get(
                    "cohort_roles_sha256"
                ),
                "cohort_sha256": (result.get("runtime_contract") or {}).get("cohort_sha256"),
                "cohort_subject_count": (result.get("runtime_contract") or {}).get(
                    "cohort_subject_count"
                ),
                "cohort_sample_count": (result.get("runtime_contract") or {}).get(
                    "cohort_sample_count"
                ),
                "outer_test_access_during_training": (result.get("runtime_contract") or {}).get(
                    "outer_test_access_during_training"
                ),
                "development_control_name": control.get("name", "none"),
                "development_control_sha256": (result.get("runtime_contract") or {}).get(
                    "development_control_sha256"
                ),
                "control_donor_scope": control.get("donor_scope"),
                "control_donor_mapping_sha256": control.get("donor_mapping_sha256"),
                "control_mapping_row_count": control.get("mapping_row_count"),
                "control_fixed_point_count": control.get("fixed_point_count"),
                "control_pair_coverage": control.get("pair_coverage"),
                "control_effective_changed_fraction": control.get(
                    "effective_external_metadata_changed_fraction"
                ),
                "control_condition_flip_fraction": control.get("condition_flip_fraction"),
                "control_field_changed_fraction": json.dumps(
                    control.get("field_changed_fraction", {}), sort_keys=True
                ),
                "natural_missingness_pattern_count": control.get(
                    "natural_missingness_pattern_count"
                ),
                "control_assay_potency": control.get("assay_potency"),
                "runtime_metrics_schema": runtime_metrics.get("schema"),
                "runtime_metrics_path": result.get("runtime_metrics_path"),
                "runtime_metrics_sha256": result.get("runtime_metrics_sha256"),
                "resolved_device": runtime_metrics.get("resolved_device"),
                "device_name": runtime_metrics.get("device_name"),
                "compute_capability": json.dumps(runtime_metrics.get("compute_capability")),
                "device_total_memory_bytes": runtime_metrics.get("device_total_memory_bytes"),
                "cuda_peak_allocated_bytes": runtime_metrics.get("peak_memory_allocated_bytes"),
                "cuda_peak_reserved_bytes": runtime_metrics.get("peak_memory_reserved_bytes"),
                "cuda_oom": runtime_metrics.get("cuda_oom"),
                "elapsed_time_sec": result.get("elapsed_time_sec"),
                "total_elapsed_time_sec": result.get("total_elapsed_time_sec"),
                "output_dir": result.get("output_dir", cfg["output_dir"]),
            }
            if "test" in result:
                row["test_accuracy"] = (result.get("test") or {}).get("accuracy")
                row["test_balanced_accuracy"] = (result.get("test") or {}).get("balanced_accuracy")
            rows.append(row)
        except GovernanceError as exc:
            if reliability_staging_root is None:
                raise
            failed = True
            failure_metrics = _load_failure_runtime_metrics(
                artifact_output_root / name
            )
            rows.append(
                {
                    "variant": name,
                    "seed": cfg.get("seed"),
                    "split_seed": cfg.get("data", {}).get("split_seed", 42),
                    "base_config": base_config,
                    "status": "failed",
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                    "output_dir": str(artifact_output_root / name),
                    **_flatten_runtime_metrics(failure_metrics),
                }
            )
            traceback.print_exc()
            break
        except Exception as exc:  # noqa: BLE001 - preserve and report failed research runs
            failed = True
            failure_metrics = _load_failure_runtime_metrics(
                artifact_output_root / name
                if reliability_staging_root is not None
                else Path(cfg["output_dir"])
            )
            rows.append(
                {
                    "variant": name,
                    "seed": cfg.get("seed"),
                    "split_seed": cfg.get("data", {}).get("split_seed", 42),
                    "fold_index": cfg.get("data", {}).get("fold_index"),
                    "base_config": base_config,
                    "status": "failed",
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                    "output_dir": str(
                        artifact_output_root / name
                        if reliability_staging_root is not None
                        else Path(cfg["output_dir"])
                    ),
                    "development_control_name": cfg.get("protocol", {}).get(
                        "development_control", "none"
                    ),
                    **_flatten_runtime_metrics(failure_metrics),
                }
            )
            traceback.print_exc()
            if not args.continue_on_error:
                break

    try:
        pair_status = _validate_primary_runtime_pair(rows)
        for row in rows:
            if row.get("variant") in {
                "A0_eeg_only",
                "A2_structured_condition_prompt",
            }:
                row["primary_pair_runtime_status"] = pair_status
    except ValueError as exc:
        failed = True
        rows.append(
            {
                "variant": "__primary_pair_guard__",
                "status": "failed",
                "error": str(exc),
            }
        )

    try:
        control_status = _validate_development_control_runtime_family(
            rows, suite.get("control_family")
        )
        for row in rows:
            if row.get("variant") in set(
                (suite.get("control_family") or {}).get("training_variants", [])
            ):
                row["control_family_runtime_status"] = control_status
    except ValueError as exc:
        failed = True
        rows.append(
            {
                "variant": "__development_control_family_guard__",
                "status": "failed",
                "error": str(exc),
            }
        )

    try:
        reliability_status = validate_reliability_runtime_family(
            rows, suite.get("reliability_family")
        )
        for row in rows:
            if row.get("variant") in RELIABILITY_ROLES:
                row["reliability_family_runtime_status"] = reliability_status
    except ValueError as exc:
        failed = True
        rows.append(
            {
                "variant": "__reliability_family_guard__",
                "status": "failed",
                "error": str(exc),
            }
        )

    if (
        reliability_family_hash is not None
        and not args.dry_run
        and not failed
        and reliability_status == "verified_equal"
    ):
        try:
            candidate_plan_path = REPO / str(base["protocol"]["candidate_plan"])
            candidate_plan = load_config(candidate_plan_path, strict_env=False)
            analysis_receipt = finalize_synthetic_reliability_stage1(
                artifact_output_root,
                runtime_rows=rows,
                candidate_plan=candidate_plan,
                candidate_plan_path=candidate_plan_path,
                stage0_receipt_sha256=str(reliability_stage0_receipt_sha256),
            )
            analysis_receipt_sha256 = _sha256_file(analysis_receipt)
            for row in rows:
                if row.get("variant") in RELIABILITY_ROLES:
                    row["stage1_analysis_receipt"] = str(analysis_receipt)
                    row["stage1_analysis_receipt_sha256"] = analysis_receipt_sha256
        except Exception as exc:  # noqa: BLE001 - publish a terminal failed attempt
            failed = True
            rows.append(
                {
                    "variant": "__stage1_analysis_guard__",
                    "status": "failed",
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                }
            )
            traceback.print_exc()

    out = artifact_output_root / "summary.csv"
    try:
        summary_rows = copy.deepcopy(rows)
        if reliability_staging_root is not None and not failed:
            _rewrite_reliability_success_paths(
                summary_rows,
                staging_root=reliability_staging_root,
                canonical_root=canonical_output_root,
            )
        _write_summary_csv(summary_rows, out)
    except Exception as exc:
        if reliability_staging_root is None:
            raise
        rows.append(
            {
                "variant": "__stage1_summary_guard__",
                "status": "failed",
                "error_type": type(exc).__name__,
                "error": str(exc),
            }
        )
        # If even this minimal failure summary cannot be persisted, the fixed
        # reservation remains as the explicit crash/review boundary.
        _write_summary_csv(rows, out)
        attempt_receipt = _write_failed_reliability_attempt(
            reliability_staging_root,
            canonical_root=canonical_output_root,
            rows=rows,
            attempt_contract=_reliability_attempt_contract(
                base,
                reliability_family_hash=reliability_family_hash,
                stage0_receipt_sha256=reliability_stage0_receipt_sha256,
            ),
        )
        traceback.print_exc()
        print(attempt_receipt)
        print(out)
        raise SystemExit(1) from exc
    if failed:
        if reliability_staging_root is not None:
            attempt_receipt = _write_failed_reliability_attempt(
                reliability_staging_root,
                canonical_root=canonical_output_root,
                rows=rows,
                attempt_contract=_reliability_attempt_contract(
                    base,
                    reliability_family_hash=reliability_family_hash,
                    stage0_receipt_sha256=reliability_stage0_receipt_sha256,
                ),
            )
            print(attempt_receipt)
        print(out)
        raise SystemExit(1)
    if reliability_staging_root is not None:
        try:
            _publish_reliability_stage1_root(
                reliability_staging_root,
                canonical_root=canonical_output_root,
            )
        except Exception as exc:
            if not reliability_staging_root.exists():
                raise
            rows.append(
                {
                    "variant": "__stage1_publication_guard__",
                    "status": "failed",
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                }
            )
            _write_summary_csv(rows, out)
            attempt_receipt = _write_failed_reliability_attempt(
                reliability_staging_root,
                canonical_root=canonical_output_root,
                rows=rows,
                attempt_contract=_reliability_attempt_contract(
                    base,
                    reliability_family_hash=reliability_family_hash,
                    stage0_receipt_sha256=reliability_stage0_receipt_sha256,
                ),
            )
            traceback.print_exc()
            print(attempt_receipt)
            print(out)
            raise SystemExit(1) from exc
        out = canonical_output_root / "summary.csv"
        if reliability_reservation is None:  # pragma: no cover - coupled above
            raise RuntimeError("Reliability Stage-1 publication lacks its reservation.")
        _release_reliability_stage1_execution(reliability_reservation)
    print(out)


def _apply_runtime_environment(cfg: dict) -> None:
    backbone = os.environ.get("CFEG_BACKBONE")
    if backbone:
        backbone_cfg = cfg.setdefault("model", {}).setdefault("backbone", {})
        backbone_cfg["name"] = backbone
        if backbone == "reve":
            backbone_cfg.update(
                {
                    "hf_model": "brain-bzh/reve-base",
                    "hf_positions": "brain-bzh/reve-positions",
                    "cache_dir": os.environ.get("HF_HUB_CACHE"),
                    "local_files_only": True,
                    "freeze": True,
                }
            )

    mode = os.environ.get("WANDB_MODE")
    if not mode:
        mode = "online" if os.environ.get("WANDB_API_KEY") else "disabled"
    wandb_cfg = cfg.setdefault("tracking", {}).setdefault("wandb", {})
    wandb_cfg["enabled"] = mode != "disabled"
    wandb_cfg["mode"] = mode
    wandb_cfg["project"] = os.environ.get(
        "WANDB_PROJECT", wandb_cfg.get("project", "calibration-free-eeg")
    )
    if os.environ.get("WANDB_ENTITY"):
        wandb_cfg["entity"] = os.environ["WANDB_ENTITY"]


def _validate_reliability_orchestrator_args(
    args: argparse.Namespace,
    *,
    base_config: str,
    dry_run: bool,
) -> None:
    suite_path = Path(args.config).resolve()
    base_path = Path(base_config).resolve()
    if suite_path != CANONICAL_RELIABILITY_SUITE.resolve():
        raise GovernanceError(
            "reliability-spatial-v1 requires its canonical four-arm suite."
        )
    if base_path != CANONICAL_RELIABILITY_BASE.resolve():
        raise GovernanceError(
            "reliability-spatial-v1 requires its canonical frozen base config."
        )
    if args.override:
        raise GovernanceError(
            "reliability-spatial-v1 does not permit in-memory recipe overrides."
        )
    if args.only is not None:
        raise GovernanceError(
            "Reliability Stage-1 must execute all four arms together; --only is forbidden."
        )
    if args.include_optional or args.continue_on_error or args.resume_exact:
        raise GovernanceError(
            "Reliability Stage-1 forbids optional, continue-on-error, and resume overrides."
        )
    output_root = Path(args.output_root).resolve()
    if dry_run:
        if (
            output_root == CANONICAL_RELIABILITY_STAGE1_ROOT.resolve()
            or CANONICAL_RELIABILITY_STAGE1_ROOT.resolve() in output_root.parents
        ):
            raise GovernanceError(
                "Reliability dry-run cannot target or create the canonical Stage-1 root."
            )
        if (
            output_root.parent != CANONICAL_RELIABILITY_DRY_RUN_PARENT.resolve()
            or not output_root.name.startswith("stage1-contract-")
            or not output_root.name.removeprefix("stage1-contract-")
        ):
            raise GovernanceError(
                "Reliability dry-run must use a direct fresh child named "
                "stage1-contract-* under its dedicated dry-runs directory."
            )
        if output_root.exists() or output_root.is_symlink():
            raise GovernanceError(
                "Reliability dry-run output root must be fresh."
            )
    else:
        if output_root != CANONICAL_RELIABILITY_STAGE1_ROOT.resolve():
            raise GovernanceError(
                "Reliability Stage-1 must use its canonical output root."
            )
        if output_root.exists() or output_root.is_symlink():
            raise GovernanceError(
                "Reliability Stage-1 output root already exists; refusing an overwrite or rerun."
            )
        reservation = _reliability_stage1_reservation_path(output_root)
        if reservation.exists() or reservation.is_symlink():
            raise GovernanceError(
                "A Reliability Stage-1 execution reservation already exists; retry "
                "requires a new owner decision after reservation review."
            )
        failed_attempts = _noncanonical_reliability_attempts(output_root)
        if failed_attempts:
            raise GovernanceError(
                "A prior noncanonical Reliability Stage-1 attempt exists; retry requires "
                "a new owner decision after attempt review: "
                f"{[str(path) for path in failed_attempts]}."
            )


def _validate_development_grid_execution(base: dict, *, dry_run: bool) -> None:
    if dry_run:
        return
    protocol = base.get("protocol", {})
    data = base.get("data", {})
    governed_development = (
        protocol.get("governance_required") is True
        and protocol.get("execution_phase") == "development"
        and data.get("cohort_role") == "development"
    )
    design = base.get("development_grid")
    if not governed_development and not isinstance(design, dict):
        return
    if isinstance(design, dict) and design.get("schema") == "cfeg.development-loso-design.v1":
        raise GovernanceError(
            "The first S1-S3 grid is a completed historical prompt experiment; "
            "it cannot be recreated through the mutable shared ablation suite."
        )
    raise GovernanceError(
        "Governed S1-S3 outcome training must run through its canonical grid orchestrator; "
        "CLI overrides cannot disable or rename this gate."
    )


def _requires_frozen_primary_architecture(base: dict) -> bool:
    protocol = base.get("protocol", {})
    data = base.get("data", {})
    return bool(
        protocol.get("governance_required") is True
        and data.get("expected_revisions", {}).get("wearable") == "wearable_v3"
    )


def _validate_development_control_family(base: dict, suite: dict) -> str | None:
    family = suite.get("control_family")
    if family is None:
        return None
    if family.get("schema") != "cfeg.development-control-family.v1":
        raise ValueError("Unknown development control-family schema.")
    names = list(family.get("training_variants") or [])
    variants = suite.get("variants") or {}
    missing = sorted(set(names) - set(variants))
    if missing:
        raise ValueError(f"Development control family is missing variants: {missing}.")
    if family.get("reference_variant") not in names:
        raise ValueError("Development control family reference must be a training variant.")
    if (
        family.get("cohort_role") != "development"
        or family.get("confirmatory_claim_allowed") is not False
    ):
        raise ValueError("Development controls must be S1-S3-only and non-confirmatory.")
    if (
        base.get("protocol", {}).get("execution_phase") != "development"
        or base.get("data", {}).get("cohort_role") != "development"
    ):
        raise ValueError("Development control suite requires the governed development base.")

    allowed_training_controls = {
        "none",
        "metadata_only",
        "missingness_only",
        "within_class_shuffle",
    }
    signatures: dict[str, str] = {}
    observed_controls: dict[str, str] = {}
    for name in names:
        cfg = merge_overrides(
            copy.deepcopy(base),
            [f"{key}={value!r}" for key, value in variants[name].items()],
        )
        protocol = cfg.get("protocol", {})
        model_control = str(protocol.get("development_control", "none"))
        training_control = str(protocol.get("training_external_metadata_control", "none"))
        if model_control != "none" and training_control != "none":
            raise ValueError(f"Training variant {name!r} declares two development controls.")
        control = training_control if training_control != "none" else model_control
        if control not in allowed_training_controls:
            raise ValueError(
                f"Training variant {name!r} has invalid development control {control!r}."
            )
        observed_controls[name] = control
        normalized = copy.deepcopy(cfg)
        normalized.setdefault("protocol", {})["development_control"] = "<control-treatment>"
        normalized["protocol"]["training_external_metadata_control"] = "<control-treatment>"
        normalized["protocol"].pop("development_control_seed", None)
        payload = json.dumps(normalized, sort_keys=True, separators=(",", ":"))
        signatures[name] = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    if observed_controls[family["reference_variant"]] != "none":
        raise ValueError("Development control-family reference must use correct metadata.")
    if len(set(observed_controls.values())) != len(observed_controls):
        raise ValueError(f"Development control roles must be unique: {observed_controls}.")
    if len(set(signatures.values())) != 1:
        raise ValueError(
            f"Development control configs differ outside the declared treatment: {signatures}."
        )
    payload = {
        "family": family,
        "normalized_config_sha256": next(iter(signatures.values())),
        "variant_controls": observed_controls,
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _validate_physical_mechanism_family(base: dict, suite: dict) -> str | None:
    family = suite.get("mechanism_family")
    if family is None:
        return None
    if family.get("schema") != "cfeg.physical-mechanism-family.v1":
        raise ValueError("Unknown physical mechanism-family schema.")
    expected_roles = [
        "A0_eeg_only",
        "A2_structured_condition_prompt",
        "M1_global_only",
        "M2_channel_only",
        "M3_full_shuffle_train",
        "M4_metadata_only",
    ]
    if list(family.get("roles") or []) != expected_roles:
        raise ValueError("Physical mechanism family must declare the exact ordered six roles.")
    if family.get("missingness_only_policy") != "invalid_assay_single_natural_pattern":
        raise ValueError("Physical mechanism family must mark missingness-only as invalid.")
    if family.get("population_inference_allowed") is not False:
        raise ValueError("Physical mechanism family cannot authorize population inference.")
    variants = suite.get("variants") or {}
    missing = sorted(set(expected_roles) - set(variants))
    if missing:
        raise ValueError(f"Physical mechanism family is missing variants: {missing}.")

    expected_treatments = {
        "A0_eeg_only": ("null", True, True, "none"),
        "A2_structured_condition_prompt": ("observed", True, True, "none"),
        "M1_global_only": ("observed", True, False, "none"),
        "M2_channel_only": ("observed", False, True, "none"),
        "M3_full_shuffle_train": ("observed", True, True, "within_class_shuffle"),
        "M4_metadata_only": ("observed", True, True, "metadata_only"),
    }
    signatures: dict[str, str] = {}
    observed_treatments: dict[str, tuple[str, bool, bool, str]] = {}
    for role in expected_roles:
        cfg = merge_overrides(
            copy.deepcopy(base),
            [f"{key}={value!r}" for key, value in variants[role].items()],
        )
        model = cfg.get("model", {})
        conditioning = model.get("conditioning", {})
        protocol = cfg.get("protocol", {})
        model_control = str(protocol.get("development_control", "none"))
        training_control = str(protocol.get("training_external_metadata_control", "none"))
        effective_control = training_control if training_control != "none" else model_control
        treatment = (
            str(model.get("condition_encoder", {}).get("external_metadata_mode", "observed")),
            bool(conditioning.get("external_global_film", {}).get("enabled", False)),
            bool(conditioning.get("channel_quality", {}).get("enabled", False)),
            effective_control,
        )
        observed_treatments[role] = treatment
        if treatment != expected_treatments[role]:
            raise ValueError(
                f"Physical mechanism treatment mismatch for {role}: "
                f"expected {expected_treatments[role]}, got {treatment}."
            )
        if conditioning.get("architecture") != "physical_hybrid_v1" or not bool(
            conditioning.get("common_query_film", {}).get("enabled", False)
        ):
            raise ValueError("Every physical mechanism role must retain the shared query FiLM.")
        normalized = copy.deepcopy(cfg)
        normalized["model"]["condition_encoder"]["external_metadata_mode"] = "<external-mode>"
        normalized["model"]["conditioning"]["external_global_film"]["enabled"] = "<global-enabled>"
        normalized["model"]["conditioning"]["channel_quality"]["enabled"] = "<channel-enabled>"
        normalized.setdefault("protocol", {})["development_control"] = "<control-treatment>"
        normalized["protocol"]["training_external_metadata_control"] = "<control-treatment>"
        normalized["protocol"].pop("development_control_seed", None)
        payload = json.dumps(normalized, sort_keys=True, separators=(",", ":"))
        signatures[role] = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    if len(set(signatures.values())) != 1:
        raise ValueError(
            "Physical mechanism configs differ outside their declared treatment masks: "
            f"{signatures}."
        )
    payload = {
        "family": family,
        "normalized_config_sha256": next(iter(signatures.values())),
        "role_treatments": observed_treatments,
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _load_failure_runtime_metrics(output_dir: Path) -> dict:
    path = output_dir / "runtime_metrics.json"
    if not path.exists():
        return {}
    with path.open(encoding="utf-8") as handle:
        metrics = json.load(handle)
    metrics["_path"] = str(path)
    metrics["_sha256"] = _sha256_file(path)
    return metrics


def _flatten_runtime_metrics(metrics: dict) -> dict:
    return {
        "runtime_metrics_schema": metrics.get("schema"),
        "runtime_metrics_path": metrics.get("_path"),
        "runtime_metrics_sha256": metrics.get("_sha256"),
        "resolved_device": metrics.get("resolved_device"),
        "device_name": metrics.get("device_name"),
        "compute_capability": json.dumps(metrics.get("compute_capability")),
        "device_total_memory_bytes": metrics.get("device_total_memory_bytes"),
        "cuda_peak_allocated_bytes": metrics.get("peak_memory_allocated_bytes"),
        "cuda_peak_reserved_bytes": metrics.get("peak_memory_reserved_bytes"),
        "cuda_oom": metrics.get("cuda_oom"),
        "elapsed_time_sec": metrics.get("elapsed_time_sec"),
    }


def _rewrite_reliability_success_paths(
    rows: list[dict],
    *,
    staging_root: Path,
    canonical_root: Path,
) -> None:
    for row in rows:
        for field in (
            "output_dir",
            "runtime_metrics_path",
            "stage1_analysis_receipt",
        ):
            value = row.get(field)
            if not value:
                continue
            observed = Path(str(value)).resolve()
            try:
                relative = observed.relative_to(staging_root.resolve())
            except ValueError as exc:
                raise ValueError(
                    f"Reliability success path {field!r} escaped its staging root."
                ) from exc
            row[field] = str(canonical_root / relative)


def _write_summary_csv(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.tmp-",
        dir=path.parent,
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as handle:
            pd.DataFrame(rows).to_csv(handle, index=False)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        _fsync_directory(path.parent)
    finally:
        if temporary.exists():
            temporary.unlink()


def _reliability_attempt_contract(
    base: dict,
    *,
    reliability_family_hash: str | None,
    stage0_receipt_sha256: str | None,
) -> dict[str, object]:
    return {
        "candidate_id": base.get("protocol", {}).get("candidate_id"),
        "candidate_plan_sha256": base.get("protocol", {}).get(
            "candidate_plan_sha256"
        ),
        "reliability_mechanism_family_sha256": reliability_family_hash,
        "stage0_receipt_sha256": stage0_receipt_sha256,
        "implementation_contract_sha256": implementation_contract_sha256(),
        **current_source_revision_contract(),
    }


def _reliability_stage1_reservation_path(canonical_root: Path) -> Path:
    return canonical_root.parent / f".{canonical_root.name}.execution-reservation"


def _reserve_reliability_stage1_execution(
    canonical_root: Path,
    *,
    contract: dict[str, object],
) -> Path:
    reservation = _reliability_stage1_reservation_path(canonical_root)
    try:
        reservation.mkdir(mode=0o700)
    except FileExistsError as exc:
        raise GovernanceError(
            "Reliability Stage-1 is already reserved by another or prior execution."
        ) from exc
    if canonical_root.exists() or canonical_root.is_symlink():
        reservation.rmdir()
        _fsync_directory(reservation.parent)
        raise GovernanceError(
            "Reliability Stage-1 canonical output appeared before reservation "
            "acquisition; refusing another execution."
        )
    receipt = reservation / "reservation.json"
    payload = {
        "schema": "cfeg.synthetic-reliability-stage1-reservation.v1",
        "status": "active_until_complete_atomic_publication",
        "canonical_output_root": str(canonical_root),
        "process_id": os.getpid(),
        "retry_authorized": False,
        "retry_requires_new_owner_decision_after_crash": True,
        "contract": contract,
    }
    with receipt.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.flush()
        os.fsync(handle.fileno())
    _fsync_directory(reservation)
    _fsync_directory(reservation.parent)
    return reservation


def _release_reliability_stage1_execution(reservation: Path) -> None:
    receipt = reservation / "reservation.json"
    entries = list(reservation.iterdir()) if reservation.is_dir() else []
    if (
        reservation.is_symlink()
        or len(entries) != 1
        or entries[0] != receipt
        or not receipt.is_file()
        or receipt.is_symlink()
    ):
        raise ValueError("Reliability Stage-1 reservation is unsafe to release.")
    payload = json.loads(receipt.read_text(encoding="utf-8"))
    if (
        payload.get("schema")
        != "cfeg.synthetic-reliability-stage1-reservation.v1"
        or payload.get("status") != "active_until_complete_atomic_publication"
        or payload.get("process_id") != os.getpid()
    ):
        raise ValueError("Reliability Stage-1 reservation ownership changed.")
    receipt.unlink()
    reservation.rmdir()
    _fsync_directory(reservation.parent)


def _write_failed_reliability_attempt(
    staging_root: Path,
    *,
    canonical_root: Path,
    rows: list[dict],
    attempt_contract: dict,
) -> Path:
    summary = staging_root / "summary.csv"
    if not summary.is_file() or summary.is_symlink():
        raise ValueError("Failed reliability attempt lacks a safe summary artifact.")
    receipt = staging_root / "stage1_attempt_receipt.json"
    payload = {
        "schema": "cfeg.synthetic-reliability-stage1-attempt.v1",
        "status": "failed_noncanonical_attempt",
        "canonical_output_root": str(canonical_root),
        "staging_output_root": str(staging_root),
        "canonical_published": False,
        "retry_authorized": False,
        "retry_requires_new_owner_decision": True,
        "summary_sha256": _sha256_file(summary),
        "runtime_rows_sha256": _sha256_json(rows),
        "attempt_contract": attempt_contract,
        "asset_provenance_sha256": sorted(
            {
                str(row["asset_provenance_sha256"])
                for row in rows
                if row.get("asset_provenance_sha256")
            }
        ),
        "roles_completed": sorted(
            str(row["variant"])
            for row in rows
            if row.get("variant") in RELIABILITY_ROLES
            and row.get("status") == "completed"
        ),
        "failures": [
            {
                "variant": row.get("variant"),
                "error_type": row.get("error_type"),
                "error": row.get("error"),
            }
            for row in rows
            if row.get("status") == "failed"
        ],
        "claims": {
            "synthetic_integration_claim_allowed": False,
            "human_eeg_claim_allowed": False,
            "population_inference_allowed": False,
        },
        "artifacts": _fingerprint_noncanonical_attempt(staging_root),
    }
    with receipt.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.flush()
        os.fsync(handle.fileno())
    with summary.open("rb") as handle:
        os.fsync(handle.fileno())
    _fsync_directory(staging_root)
    return receipt


def _noncanonical_reliability_attempts(canonical_root: Path) -> list[Path]:
    if not canonical_root.parent.is_dir():
        return []
    return sorted(
        path
        for path in canonical_root.parent.glob(f".{canonical_root.name}.staging-*")
        if path.exists() or path.is_symlink()
    )


def _fingerprint_noncanonical_attempt(staging_root: Path) -> dict[str, dict[str, object]]:
    artifacts: dict[str, dict[str, object]] = {}
    for path in sorted(staging_root.rglob("*")):
        if path.is_symlink():
            raise ValueError("Failed reliability attempt contains a symlink.")
        if path.is_file():
            relative = path.relative_to(staging_root).as_posix()
            if relative == "stage1_attempt_receipt.json":
                continue
            artifacts[relative] = {
                "sha256": _sha256_file(path),
                "size_bytes": path.stat().st_size,
            }
        elif not path.is_dir():
            raise ValueError("Failed reliability attempt contains a special file.")
    return artifacts


def _publish_reliability_stage1_root(
    staging_root: Path,
    *,
    canonical_root: Path,
) -> None:
    expected = {*RELIABILITY_ROLES, "analysis", "summary.csv"}
    entries = list(staging_root.iterdir())
    if {path.name for path in entries} != expected:
        raise ValueError("Reliability Stage-1 staging root is incomplete or unexpected.")
    if any(
        (path.name == "summary.csv" and (not path.is_file() or path.is_symlink()))
        or (path.name != "summary.csv" and (not path.is_dir() or path.is_symlink()))
        for path in entries
    ):
        raise ValueError("Reliability Stage-1 staging root contains an unsafe entry.")
    _validate_reliability_stage1_summary(
        staging_root,
        canonical_root=canonical_root,
    )
    _fsync_tree(staging_root)
    validate_synthetic_reliability_stage1_publication_inputs(staging_root)
    _validate_reliability_stage1_summary(
        staging_root,
        canonical_root=canonical_root,
    )
    if canonical_root.exists() or canonical_root.is_symlink():
        raise FileExistsError(
            f"Reliability Stage-1 canonical root appeared before publication: {canonical_root}."
        )
    os.replace(staging_root, canonical_root)
    _fsync_directory(canonical_root.parent)


def _validate_reliability_stage1_summary(
    staging_root: Path,
    *,
    canonical_root: Path,
) -> None:
    path = staging_root / "summary.csv"
    if not path.is_file() or path.is_symlink():
        raise ValueError("Reliability Stage-1 summary is missing or unsafe.")
    frame = pd.read_csv(path, dtype=str, keep_default_na=False)
    required = {
        "variant",
        "status",
        "reliability_family_runtime_status",
        "output_dir",
        "runtime_metrics_path",
        "stage1_analysis_receipt",
        "stage1_analysis_receipt_sha256",
    }
    if (
        not required.issubset(frame.columns)
        or len(frame) != len(RELIABILITY_ROLES)
        or set(frame["variant"]) != set(RELIABILITY_ROLES)
        or frame["variant"].duplicated().any()
        or set(frame["status"]) != {"completed"}
        or set(frame["reliability_family_runtime_status"]) != {"verified_equal"}
    ):
        raise ValueError("Reliability Stage-1 summary does not contain four completed arms.")
    receipt_path = canonical_root / "analysis/stage1_receipt.json"
    receipt_sha256 = _sha256_file(staging_root / "analysis/stage1_receipt.json")
    for row in frame.to_dict(orient="records"):
        role = str(row["variant"])
        if (
            row["output_dir"] != str(canonical_root / role)
            or row["runtime_metrics_path"]
            != str(canonical_root / role / "runtime_metrics.json")
            or row["stage1_analysis_receipt"] != str(receipt_path)
            or row["stage1_analysis_receipt_sha256"] != receipt_sha256
        ):
            raise ValueError("Reliability Stage-1 summary contains stale artifact paths.")
    if ".stage1.staging-" in path.read_text(encoding="utf-8"):
        raise ValueError("Reliability Stage-1 summary exposes a staging path.")


def _fsync_tree(root: Path) -> None:
    paths = list(root.rglob("*"))
    if any(path.is_symlink() for path in paths):
        raise ValueError("Reliability Stage-1 staging tree cannot contain symlinks.")
    for path in paths:
        if path.is_file():
            with path.open("rb") as handle:
                os.fsync(handle.fileno())
        elif not path.is_dir():
            raise ValueError("Reliability Stage-1 staging tree contains a special file.")
    directories = sorted(
        (path for path in paths if path.is_dir()),
        key=lambda path: len(path.parts),
        reverse=True,
    )
    for directory in [*directories, root]:
        _fsync_directory(directory)


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_json(value) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
        default=str,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _validate_primary_ablation_contract(base: dict, variants: dict) -> str | None:
    """Backward-compatible wrapper used by downstream scripts/tests."""

    return validate_primary_pair_and_hash(base, variants)


def _validate_primary_runtime_pair(rows: list[dict]) -> str:
    names = ("A0_eeg_only", "A2_structured_condition_prompt")
    primary = {row.get("variant"): row for row in rows if row.get("variant") in names}
    if set(primary) != set(names):
        return "not_run_together"
    if any(primary[name].get("status") not in {"dry_run", "completed"} for name in names):
        return "incomplete"
    equality_fields = (
        "total_parameters",
        "trainable_parameters",
        "parameter_schema_sha256",
        "initial_trainable_state_sha256",
        "split_assignment_sha256",
        "vocabulary_sha256",
        "asset_provenance_sha256",
        "source_commit_sha",
        "source_dirty",
        "source_tree_sha256",
        "environment_sha256",
        "execution_phase",
        "analysis_plan_sha256",
        "cohort_roles_sha256",
        "cohort_sha256",
        "cohort_subject_count",
        "cohort_sample_count",
        "outer_test_access_during_training",
    )
    missing = {
        name: [field for field in equality_fields if primary[name].get(field) is None]
        for name in names
    }
    missing = {name: fields for name, fields in missing.items() if fields}
    if missing:
        raise ValueError(f"Primary runtime contract is incomplete: {missing}.")
    mismatched = [
        field for field in equality_fields if primary[names[0]][field] != primary[names[1]][field]
    ]
    if mismatched:
        raise ValueError(
            f"Primary A0/A2 runtime contracts differ outside treatment access: {mismatched}."
        )
    return "verified_equal"


def _validate_development_control_runtime_family(rows: list[dict], family: dict | None) -> str:
    if not family:
        return "not_configured"
    names = tuple(family.get("training_variants") or [])
    observed = {row.get("variant"): row for row in rows if row.get("variant") in names}
    if set(observed) != set(names):
        return "not_run_together"
    if any(observed[name].get("status") not in {"dry_run", "completed"} for name in names):
        return "incomplete"
    equality_fields = (
        "total_parameters",
        "trainable_parameters",
        "parameter_schema_sha256",
        "initial_trainable_state_sha256",
        "split_assignment_sha256",
        "vocabulary_sha256",
        "asset_provenance_sha256",
        "source_commit_sha",
        "source_dirty",
        "source_tree_sha256",
        "environment_sha256",
        "execution_phase",
        "analysis_plan_sha256",
        "cohort_roles_sha256",
        "cohort_sha256",
        "cohort_subject_count",
        "cohort_sample_count",
        "outer_test_access_during_training",
        "runtime_metrics_schema",
        "resolved_device",
    )
    missing = {
        name: [field for field in equality_fields if observed[name].get(field) is None]
        for name in names
    }
    missing = {name: fields for name, fields in missing.items() if fields}
    if missing:
        raise ValueError(f"Development control runtime contract is incomplete: {missing}.")
    reference = names[0]
    mismatched = [
        field
        for field in equality_fields
        if any(observed[name][field] != observed[reference][field] for name in names[1:])
    ]
    if mismatched:
        raise ValueError(
            f"Development control runtime contracts differ outside treatment: {mismatched}."
        )
    return "verified_equal"


if __name__ == "__main__":
    main()
