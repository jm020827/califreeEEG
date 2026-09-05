from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from cfeg.constants import (
    PROTOCOL_V04_EXTERNAL_CATEGORICAL_FIELDS,
    PROTOCOL_V04_EXTERNAL_CONTINUOUS_FIELDS,
)
from cfeg.data.collate import _norm_impedance, overwrite_external_metadata
from cfeg.eval_loop import _checkpoint_split_indices, _loader, load_evaluation_context
from cfeg.governance import (
    current_source_revision_contract,
    implementation_contract_sha256,
)
from cfeg.metrics import classification_metrics
from cfeg.reliability_contract import (
    RELIABILITY_BASE_CONFIG,
    RELIABILITY_CANDIDATE_PLAN,
    RELIABILITY_ROLES,
    validate_reliability_stage0_gate,
    validate_reliability_training_preflight,
)
from cfeg.train_loop import _to_device
from cfeg.utils.config import load_config

STAGE1_ANALYSIS_SCHEMA = "cfeg.synthetic-reliability-stage1-analysis.v1"
STAGE1_RECEIPT_SCHEMA = "cfeg.synthetic-reliability-stage1-receipt.v1"
ANALYSIS_FILES = (
    "validation_predictions.csv",
    "validation_subject_metrics.csv",
    "validation_group_metrics.csv",
    "stage1_subject_contrasts.csv",
    "metadata_intervention_predictions.csv",
    "metadata_intervention_mapping.csv",
    "metadata_intervention_subject_metrics.csv",
    "stage1_metadata_safety.csv",
    "metadata_only_probe_predictions.csv",
    "metadata_only_probe_subject_metrics.csv",
    "metadata_only_probe_model.json",
)
OUTPUT_FILES = (*ANALYSIS_FILES, "stage1_receipt.json")
M_FIELDS = (
    *PROTOCOL_V04_EXTERNAL_CATEGORICAL_FIELDS,
    *PROTOCOL_V04_EXTERNAL_CONTINUOUS_FIELDS,
    "impedance_kohm_by_channel",
)
PREDICTOR_FORBIDDEN = {
    "label",
    "stimulus_frequency_hz",
    "stimulus_phase_rad",
    "subject_id",
    "session_id",
    "run_id",
    "trial_id",
    "sample_id",
    "acquisition_block_id",
}
_AUX_FIELDS = (
    "metadata_residual_alpha",
    "spatial_operator_frobenius_norm",
    "query_spatial_operator_frobenius_norm",
    "metadata_spatial_residual_frobenius_norm",
    "spatial_self_gain_min",
    "spatial_self_gain_max",
    "spatial_offdiagonal_row_l1_max",
    "channel_query_available_count",
    "channel_metadata_available_count",
)
_RUNTIME_ROW_FIELDS = (
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
    "analysis_plan_status",
    "analysis_plan_sha256",
    "cohort_role",
    "cohort_roles_sha256",
    "cohort_sha256",
    "cohort_subject_count",
    "cohort_sample_count",
    "outer_test_access_during_training",
    "loader_settings_sha256",
    "development_control_sha256",
)
_PROTOCOL_ROW_FIELDS = (
    "candidate_id",
    "candidate_plan_sha256",
    "predecessor_retirement_receipt_sha256",
    "reliability_query_feature_schema",
    "reliability_mechanism_family_sha256",
    "reliability_family_role",
    "reliability_stage0_receipt_sha256",
)
REPO = Path(__file__).resolve().parents[3]


def finalize_synthetic_reliability_stage1(
    output_root: str | Path,
    *,
    runtime_rows: list[dict[str, Any]],
    candidate_plan: Mapping[str, Any],
    candidate_plan_path: str | Path,
    stage0_receipt_sha256: str,
) -> Path:
    """Evaluate and atomically publish the frozen synthetic Stage-1 analysis."""

    root = Path(output_root)
    analysis_dir = root / "analysis"
    if analysis_dir.exists():
        raise FileExistsError("Reliability Stage-1 analysis output already exists.")
    result = analyze_synthetic_reliability_stage1(
        root,
        runtime_rows=runtime_rows,
        candidate_plan=candidate_plan,
        candidate_plan_path=candidate_plan_path,
        stage0_receipt_sha256=stage0_receipt_sha256,
    )
    return write_stage1_analysis(result, analysis_dir)


def analyze_synthetic_reliability_stage1(
    output_root: str | Path,
    *,
    runtime_rows: list[dict[str, Any]],
    candidate_plan: Mapping[str, Any],
    candidate_plan_path: str | Path,
    stage0_receipt_sha256: str,
) -> dict[str, Any]:
    root = Path(output_root)
    plan_path = Path(candidate_plan_path).resolve()
    if not plan_path.is_file() or plan_path.is_symlink():
        raise ValueError("Stage-1 candidate plan path is missing or unsafe.")
    observed_plan = load_config(plan_path, strict_env=False)
    if _json_roundtrip(candidate_plan) != _json_roundtrip(observed_plan):
        raise ValueError("Stage-1 in-memory candidate plan differs from its frozen file.")
    candidate_plan_file_sha256 = _sha256_file(plan_path)
    analysis_contract = dict(candidate_plan["stage1"]["analysis"])
    _validate_analysis_contract(analysis_contract)
    rows_by_role = {
        str(row.get("variant")): row
        for row in runtime_rows
        if row.get("variant") in RELIABILITY_ROLES
    }
    if set(rows_by_role) != set(RELIABILITY_ROLES) or any(
        rows_by_role[role].get("status") != "completed" for role in RELIABILITY_ROLES
    ):
        raise ValueError("Stage-1 analysis requires four completed reliability arms.")
    if any(
        rows_by_role[role].get("reliability_family_runtime_status") != "verified_equal"
        for role in RELIABILITY_ROLES
    ):
        raise ValueError("Stage-1 analysis requires a verified-equal runtime family.")
    observed_stage0 = {
        rows_by_role[role].get("reliability_stage0_receipt_sha256")
        for role in RELIABILITY_ROLES
    }
    if observed_stage0 != {stage0_receipt_sha256}:
        raise ValueError("Stage-1 arms do not share the authorized Stage-0 receipt.")
    if {
        rows_by_role[role].get("candidate_plan_sha256")
        for role in RELIABILITY_ROLES
    } != {candidate_plan_file_sha256}:
        raise ValueError("Stage-1 arms do not share the current frozen candidate plan.")
    bound_source = {
        field: rows_by_role["A0"].get(field)
        for field in ("source_commit_sha", "source_dirty", "source_tree_sha256")
    }
    if any(
        any(rows_by_role[role].get(field) != bound_source[field] for field in bound_source)
        for role in RELIABILITY_ROLES
    ):
        raise ValueError("Stage-1 arms do not share one source revision contract.")
    if current_source_revision_contract() != bound_source:
        raise ValueError("Stage-1 source differs from the source bound by its runtime rows.")
    bound_implementation_sha256 = implementation_contract_sha256()

    contexts: dict[str, dict[str, Any]] = {}
    arm_frames: list[pd.DataFrame] = []
    checkpoint_fingerprints: dict[str, dict[str, Any]] = {}
    runtime_artifacts: dict[str, dict[str, Any]] = {}
    runtime_attempt_artifacts: dict[str, dict[str, Any]] = {}
    training_artifacts: dict[str, dict[str, dict[str, Any]]] = {}
    for role in RELIABILITY_ROLES:
        checkpoint_path = root / role / "best.pt"
        if not checkpoint_path.is_file() or checkpoint_path.is_symlink():
            raise ValueError(f"Stage-1 checkpoint is missing or unsafe for {role}.")
        context = load_evaluation_context(
            {"data": {"split": "val"}},
            checkpoint_path,
            access_route="reliability_stage1_analysis",
        )
        observed_gate = validate_reliability_training_preflight(
            context["train_config"], dry_run=False
        )
        if observed_gate != stage0_receipt_sha256:
            raise ValueError(f"Stage-1 checkpoint gate differs for {role}.")
        if context["train_config"].get("protocol", {}).get(
            "reliability_family_role"
        ) != role:
            raise ValueError(f"Stage-1 checkpoint role differs for {role}.")
        indices = _checkpoint_split_indices(context, split="val")
        checkpoint_sha256 = _sha256_file(checkpoint_path)
        checkpoint = context["checkpoint"]
        checkpoint_runtime = context["train_config"].get("runtime_contract") or {}
        checkpoint_protocol = context["train_config"].get("protocol") or {}
        mismatched_runtime = [
            field
            for field in _RUNTIME_ROW_FIELDS
            if checkpoint_runtime.get(field) != rows_by_role[role].get(field)
        ]
        mismatched_protocol = [
            field
            for field in _PROTOCOL_ROW_FIELDS
            if checkpoint_protocol.get(field) != rows_by_role[role].get(field)
        ]
        if mismatched_runtime or mismatched_protocol:
            raise ValueError(
                f"Stage-1 checkpoint/row contract differs for {role}: "
                f"runtime={mismatched_runtime}, protocol={mismatched_protocol}."
            )
        if any(
            checkpoint_runtime.get(field) != bound_source[field]
            or rows_by_role[role].get(field) != bound_source[field]
            for field in ("source_commit_sha", "source_dirty", "source_tree_sha256")
        ):
            raise ValueError(f"Stage-1 source changed before analysis for {role}.")
        if (
            checkpoint.get("checkpoint_role") != "source_validation_best"
            or checkpoint.get("selection_split") != "val"
            or checkpoint.get("selection_metric") != "accuracy"
        ):
            raise ValueError(f"Stage-1 best checkpoint contract is invalid for {role}.")
        split_path = root / role / "split.csv"
        metrics_path = root / role / "metrics_val.csv"
        _validate_best_metric_artifact(
            metrics_path,
            checkpoint_epoch=int(checkpoint["epoch"]),
            checkpoint_metric=float(checkpoint["best_metric"]),
        )
        training_artifacts[role] = {
            "split": _bound_file_contract(
                split_path,
                relative=f"{role}/split.csv",
            ),
            "validation_metrics": _bound_file_contract(
                metrics_path,
                relative=f"{role}/metrics_val.csv",
            ),
        }
        frame = _collect_prediction_frame(
            context,
            indices,
            role=role,
            scenario="correct",
            checkpoint_sha256=checkpoint_sha256,
            best_epoch=int(checkpoint["epoch"]),
        )
        contexts[role] = context
        arm_frames.append(frame)
        observed_accuracy = float(
            (frame["label"].to_numpy(int) == frame["predicted_label"].to_numpy(int)).mean()
        )
        expected_accuracy = float(checkpoint["best_metric"])
        row_accuracy = float(rows_by_role[role]["best_validation_accuracy"])
        if not (
            abs(observed_accuracy - expected_accuracy) <= 1e-12
            and abs(observed_accuracy - row_accuracy) <= 1e-12
        ):
            raise ValueError(
                f"Stage-1 clean accuracy does not reproduce the selected checkpoint for {role}."
            )
        checkpoint_fingerprints[role] = {
            "path": f"{role}/best.pt",
            "sha256": checkpoint_sha256,
            "size_bytes": checkpoint_path.stat().st_size,
            "epoch": int(checkpoint["epoch"]),
            "selection_split": checkpoint.get("selection_split"),
            "selection_metric": checkpoint.get("selection_metric"),
        }
        runtime_path = Path(str(rows_by_role[role].get("runtime_metrics_path", "")))
        expected_runtime_path = (root / role / "runtime_metrics.json").resolve()
        if (
            not runtime_path.is_file()
            or runtime_path.is_symlink()
            or runtime_path.resolve() != expected_runtime_path
            or rows_by_role[role].get("runtime_metrics_sha256")
            != _sha256_file(runtime_path)
        ):
            raise ValueError(f"Stage-1 runtime artifact is missing or stale for {role}.")
        runtime_metrics = json.loads(runtime_path.read_text(encoding="utf-8"))
        if (
            runtime_metrics.get("schema") != "cfeg.runtime-metrics.v1"
            or runtime_metrics.get("status") != "completed"
            or runtime_metrics.get("run_mode") != "training_and_validation"
            or runtime_metrics.get("requested_device") != "cuda"
            or runtime_metrics.get("execution_device_type") != "cuda"
            or str(runtime_metrics.get("resolved_device", "")).split(":", 1)[0]
            != "cuda"
            or runtime_metrics.get("cuda_oom") is not False
            or rows_by_role[role].get("runtime_metrics_schema")
            != runtime_metrics.get("schema")
            or rows_by_role[role].get("resolved_device")
            != runtime_metrics.get("resolved_device")
        ):
            raise ValueError(f"Stage-1 runtime evidence is invalid for {role}.")
        runtime_attempt_artifacts[role] = _validate_runtime_attempts(
            runtime_metrics, root / role
        )
        runtime_artifacts[role] = {
            "path": f"{role}/runtime_metrics.json",
            "sha256": _sha256_file(runtime_path),
            "size_bytes": runtime_path.stat().st_size,
        }

    validation_predictions = pd.concat(arm_frames, ignore_index=True)
    _validate_arm_prediction_parity(validation_predictions)
    validation_subject_metrics = _group_metrics(
        validation_predictions,
        group_columns=["role", "subject_id"],
    )
    validation_group_metrics = _group_metrics(
        validation_predictions,
        group_columns=["role", "subject_id", "electrode_type"],
    )
    subject_contrasts = _subject_contrasts(validation_subject_metrics)

    aqm_context = contexts["A_QM"]
    aqm_indices = _checkpoint_split_indices(aqm_context, split="val")
    aqm_manifest = aqm_context["manifest"].iloc[aqm_indices].reset_index(drop=True)
    overrides, intervention_mapping = _build_intervention_overrides(aqm_manifest)
    intervention_frames = [
        validation_predictions.loc[validation_predictions["role"].eq("A_QM")]
        .assign(scenario="correct")
        .reset_index(drop=True)
    ]
    for scenario in ("missing", "stale_block", "wrong_interface"):
        intervention_frames.append(
            _collect_prediction_frame(
                aqm_context,
                aqm_indices,
                role="A_QM",
                scenario=scenario,
                checkpoint_sha256=checkpoint_fingerprints["A_QM"]["sha256"],
                best_epoch=checkpoint_fingerprints["A_QM"]["epoch"],
                external_overrides=overrides[scenario],
            )
        )
    intervention_predictions = pd.concat(intervention_frames, ignore_index=True)
    _validate_intervention_predictions(
        intervention_predictions,
        intervention_mapping,
        tolerance=float(analysis_contract["integrity_tolerance"]),
    )
    intervention_subject_metrics = _group_metrics(
        intervention_predictions,
        group_columns=["scenario", "subject_id"],
    )
    metadata_safety = _metadata_safety(
        intervention_subject_metrics,
        validation_subject_metrics.loc[
            validation_subject_metrics["role"].eq("A_Q"),
            ["subject_id", "balanced_accuracy"],
        ],
    )

    reference_context = contexts["A0"]
    train_indices = _checkpoint_split_indices(reference_context, split="train")
    val_indices = _checkpoint_split_indices(reference_context, split="val")
    metadata_probe = _run_metadata_only_probe(
        reference_context["manifest"],
        train_indices=train_indices,
        val_indices=val_indices,
        n_classes=int(reference_context["train_config"]["model"]["n_classes"]),
        active_channel_ids=list(
            analysis_contract["metadata_only_probe"]["active_canonical_channel_ids"]
        ),
        ridge_alphas=tuple(
            float(value)
            for value in analysis_contract["metadata_only_probe"]["ridge_alphas"]
        ),
        invariance_tolerance=float(
            analysis_contract["metadata_only_probe"][
                "block_logit_invariance_tolerance"
            ]
        ),
        chance_tolerance=float(
            analysis_contract["metadata_only_probe"][
                "balanced_accuracy_chance_tolerance"
            ]
        ),
    )

    frames = {
        "validation_predictions.csv": validation_predictions,
        "validation_subject_metrics.csv": validation_subject_metrics,
        "validation_group_metrics.csv": validation_group_metrics,
        "stage1_subject_contrasts.csv": subject_contrasts,
        "metadata_intervention_predictions.csv": intervention_predictions,
        "metadata_intervention_mapping.csv": intervention_mapping,
        "metadata_intervention_subject_metrics.csv": intervention_subject_metrics,
        "stage1_metadata_safety.csv": metadata_safety,
        "metadata_only_probe_predictions.csv": metadata_probe["predictions"],
        "metadata_only_probe_subject_metrics.csv": metadata_probe["subject_metrics"],
    }
    if any(
        not np.isfinite(frame.select_dtypes(include="number").to_numpy(float)).all()
        for frame in frames.values()
    ):
        raise ValueError("Stage-1 analysis contains non-finite numeric output.")
    contrast_summary = {
        name: float(subject_contrasts[name].mean())
        for name in (
            "delta_metadata_given_q",
            "delta_metadata_without_q",
            "interaction",
        )
    }
    safety_summary = {
        name: float(metadata_safety[name].mean())
        for name in (
            "correct_minus_missing",
            "correct_minus_stale_block",
            "correct_minus_wrong_interface",
            "correct_minus_A_Q",
            "missing_minus_A_Q",
            "stale_block_minus_A_Q",
            "wrong_interface_minus_A_Q",
            "worst_intervened_balanced_accuracy",
        )
    }
    receipt = {
        "schema": STAGE1_RECEIPT_SCHEMA,
        "analysis_schema": STAGE1_ANALYSIS_SCHEMA,
        "analysis_contract": analysis_contract,
        "candidate_id": candidate_plan["candidate_id"],
        "status": "completed_integrity_verified_descriptive_only",
        "scope": "synthetic_engineering_only",
        "candidate_plan_file_sha256": candidate_plan_file_sha256,
        "candidate_plan_semantic_sha256": _sha256_json(candidate_plan),
        "stage0_receipt_sha256": stage0_receipt_sha256,
        "runtime_family_status": "verified_equal",
        "reliability_mechanism_family_sha256": rows_by_role["A0"][
            "reliability_mechanism_family_sha256"
        ],
        "checkpoint_fingerprints": checkpoint_fingerprints,
        "runtime_artifacts": runtime_artifacts,
        "runtime_attempt_artifacts": runtime_attempt_artifacts,
        "training_artifacts": training_artifacts,
        "validation_sample_identity_sha256": _sample_identity_sha256(
            validation_predictions.loc[
                validation_predictions["role"].eq("A0"), "sample_id"
            ].tolist()
        ),
        "n_validation_subjects": int(
            validation_predictions["subject_id"].nunique()
        ),
        "n_validation_samples_per_arm": int(
            validation_predictions.loc[validation_predictions["role"].eq("A0")].shape[0]
        ),
        "statistical_unit": "participant",
        "checkpoint_selection_metric": "accuracy",
        "score_scope": "checkpoint_selection_validation_set",
        "substantive_effect_gate": "none_descriptive_only",
        "primary_contrast": "A_QM_minus_A_Q_subject_macro_balanced_accuracy",
        "secondary_contrast": "A_M_minus_A0_subject_macro_balanced_accuracy",
        "contrast_summary": contrast_summary,
        "metadata_safety_summary": safety_summary,
        "metadata_intervention_mapping_sha256": _sha256_json(
            intervention_mapping.to_dict(orient="records")
        ),
        "metadata_only_probe": {
            "subject_macro_balanced_accuracy": float(
                metadata_probe["subject_metrics"]["balanced_accuracy"].mean()
            ),
            "chance_balanced_accuracy": 1.0
            / int(reference_context["train_config"]["model"]["n_classes"]),
            "block_logit_invariance_verified": True,
            "predictor_allowlist_only": True,
        },
        "integrity": {
            "all_checks_passed": True,
            "four_arm_sample_label_parity": True,
            "four_arm_runtime_parity": True,
            "finite_logits": True,
            "same_aqm_checkpoint_all_interventions": True,
            "query_branch_invariant_across_interventions": True,
            "a_q_reference_comparison_present": True,
            "missing_metadata_available_count_zero": True,
            "missing_metadata_residual_norm_zero": True,
            "stale_block_mapping_fixed_point_free": True,
            "wrong_interface_mapping_fixed_point_free": True,
            "metadata_only_block_invariance": True,
            "metadata_only_subject_macro_chance": True,
        },
        "claims": {
            "synthetic_integration_claim_allowed": True,
            "human_eeg_claim_allowed": False,
            "population_inference_allowed": False,
            "confirmatory_execution_authorized": False,
            "architecture_superiority_claim_allowed": False,
        },
        "interpretation": (
            "Selection-set synthetic engineering analysis only. Effect contrasts are "
            "descriptive because the same two held-out subjects selected checkpoints; "
            "no human, OOD, population, or architecture-superiority inference is allowed."
        ),
        "implementation_contract_sha256": bound_implementation_sha256,
        **bound_source,
    }
    return {
        "frames": frames,
        "metadata_only_probe_model": metadata_probe["model"],
        "receipt": receipt,
        "publish_guard": {
            "candidate_plan_path": str(plan_path),
            "base_config_path": str(
                (REPO / str(candidate_plan["stage1"]["base_config"])).resolve()
            ),
            "output_root": str(root.resolve()),
            "candidate_plan_file_sha256": candidate_plan_file_sha256,
            "stage0_receipt_sha256": stage0_receipt_sha256,
            "checkpoint_fingerprints": checkpoint_fingerprints,
            "runtime_artifacts": runtime_artifacts,
            "runtime_attempt_artifacts": runtime_attempt_artifacts,
            "training_artifacts": training_artifacts,
            "implementation_contract_sha256": receipt[
                "implementation_contract_sha256"
            ],
            **{
                field: receipt[field]
                for field in (
                    "source_commit_sha",
                    "source_dirty",
                    "source_tree_sha256",
                )
            },
        },
    }


def write_stage1_analysis(result: dict[str, Any], output_dir: str | Path) -> Path:
    _validate_stage1_publish_guard(result)
    output = Path(output_dir)
    if output.exists():
        raise FileExistsError(f"Stage-1 analysis directory already exists: {output}.")
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(prefix=f".{output.name}.staging-", dir=output.parent)
    )
    try:
        for name, frame in result["frames"].items():
            _write_csv_fsynced(frame, staging / name)
        _write_json_fsynced(
            staging / "metadata_only_probe_model.json",
            result["metadata_only_probe_model"],
        )
        receipt = dict(result["receipt"])
        receipt["artifacts"] = {
            name: {
                "path": name,
                "sha256": _sha256_file(staging / name),
                "size_bytes": (staging / name).stat().st_size,
            }
            for name in ANALYSIS_FILES
        }
        _write_json_fsynced(staging / "stage1_receipt.json", receipt)
        validate_stage1_analysis_tree(result, staging)
        _validate_stage1_publish_guard(result)
        _fsync_directory(staging)
        os.replace(staging, output)
        _fsync_directory(output.parent)
    finally:
        if staging.exists():
            shutil.rmtree(staging)
    return output / "stage1_receipt.json"


def _validate_stage1_publish_guard(result: Mapping[str, Any]) -> None:
    guard = result.get("publish_guard")
    if not isinstance(guard, Mapping):
        raise TypeError("Stage-1 result lacks its publication guard.")
    _validate_bound_stage1_inputs(guard)


def validate_synthetic_reliability_stage1_publication_inputs(
    output_root: str | Path,
) -> None:
    """Rehash every bound Stage-1 input immediately before suite publication."""

    root = Path(output_root)
    receipt_path = root / "analysis/stage1_receipt.json"
    if not receipt_path.is_file() or receipt_path.is_symlink():
        raise ValueError("Stage-1 publication receipt is missing or unsafe.")
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    if (
        receipt.get("schema") != STAGE1_RECEIPT_SCHEMA
        or receipt.get("status") != "completed_integrity_verified_descriptive_only"
        or receipt.get("scope") != "synthetic_engineering_only"
        or not (receipt.get("integrity") or {}).get("all_checks_passed")
        or receipt.get("claims")
        != {
            "synthetic_integration_claim_allowed": True,
            "human_eeg_claim_allowed": False,
            "population_inference_allowed": False,
            "confirmatory_execution_authorized": False,
            "architecture_superiority_claim_allowed": False,
        }
    ):
        raise ValueError("Stage-1 publication receipt is incomplete or unsuccessful.")
    artifacts = receipt.get("artifacts")
    if not isinstance(artifacts, Mapping) or set(artifacts) != set(ANALYSIS_FILES):
        raise ValueError("Stage-1 publication receipt has an invalid analysis inventory.")
    for name, contract in artifacts.items():
        _validate_bound_file(
            root / "analysis",
            expected_relative=str(name),
            contract=contract,
        )
    plan_path = REPO / RELIABILITY_CANDIDATE_PLAN
    plan = load_config(plan_path, strict_env=False)
    guard = {
        "candidate_plan_path": str(plan_path.resolve()),
        "base_config_path": str((REPO / RELIABILITY_BASE_CONFIG).resolve()),
        "output_root": str(root.resolve()),
        "candidate_plan_file_sha256": receipt.get("candidate_plan_file_sha256"),
        "stage0_receipt_sha256": receipt.get("stage0_receipt_sha256"),
        "checkpoint_fingerprints": receipt.get("checkpoint_fingerprints"),
        "runtime_artifacts": receipt.get("runtime_artifacts"),
        "runtime_attempt_artifacts": receipt.get("runtime_attempt_artifacts"),
        "training_artifacts": receipt.get("training_artifacts"),
        "implementation_contract_sha256": receipt.get(
            "implementation_contract_sha256"
        ),
        **{
            field: receipt.get(field)
            for field in ("source_commit_sha", "source_dirty", "source_tree_sha256")
        },
    }
    if receipt.get("candidate_plan_semantic_sha256") != _sha256_json(plan):
        raise ValueError("Stage-1 candidate-plan semantic digest changed before publication.")
    _validate_bound_stage1_inputs(guard)


def _validate_bound_stage1_inputs(guard: Mapping[str, Any]) -> None:
    plan_path = Path(str(guard.get("candidate_plan_path", "")))
    base_path = Path(str(guard.get("base_config_path", "")))
    root = Path(str(guard.get("output_root", "")))
    if (
        not plan_path.is_file()
        or plan_path.is_symlink()
        or plan_path.resolve() != (REPO / RELIABILITY_CANDIDATE_PLAN).resolve()
        or _sha256_file(plan_path) != guard.get("candidate_plan_file_sha256")
        or not base_path.is_file()
        or base_path.is_symlink()
        or base_path.resolve() != (REPO / RELIABILITY_BASE_CONFIG).resolve()
        or not root.is_dir()
        or root.is_symlink()
        or implementation_contract_sha256()
        != guard.get("implementation_contract_sha256")
    ):
        raise ValueError("Stage-1 implementation or candidate plan changed before publication.")
    current_source = current_source_revision_contract()
    if any(
        current_source[field] != guard.get(field)
        for field in ("source_commit_sha", "source_dirty", "source_tree_sha256")
    ):
        raise ValueError("Stage-1 source changed before publication.")
    observed_stage0 = validate_reliability_stage0_gate(
        load_config(base_path, strict_env=False), require_receipt=True
    )
    if observed_stage0 != guard.get("stage0_receipt_sha256"):
        raise ValueError("Stage-1 Stage-0/asset binding changed before publication.")
    for field, suffix in (
        ("checkpoint_fingerprints", "best.pt"),
        ("runtime_artifacts", "runtime_metrics.json"),
        ("runtime_attempt_artifacts", "runtime_attempts/attempt_000.json"),
    ):
        contracts = guard.get(field)
        if not isinstance(contracts, Mapping) or set(contracts) != set(RELIABILITY_ROLES):
            raise ValueError(f"Stage-1 {field} inventory is incomplete or unexpected.")
        for role in RELIABILITY_ROLES:
            _validate_bound_file(
                root,
                expected_relative=f"{role}/{suffix}",
                contract=contracts[role],
            )
    training_artifacts = guard.get("training_artifacts")
    if (
        not isinstance(training_artifacts, Mapping)
        or set(training_artifacts) != set(RELIABILITY_ROLES)
    ):
        raise ValueError("Stage-1 training artifact inventory is incomplete or unexpected.")
    for role in RELIABILITY_ROLES:
        role_contract = training_artifacts[role]
        if not isinstance(role_contract, Mapping) or set(role_contract) != {
            "split",
            "validation_metrics",
        }:
            raise ValueError(f"Stage-1 training artifact contract is invalid for {role}.")
        _validate_bound_file(
            root,
            expected_relative=f"{role}/split.csv",
            contract=role_contract["split"],
        )
        _validate_bound_file(
            root,
            expected_relative=f"{role}/metrics_val.csv",
            contract=role_contract["validation_metrics"],
        )


def _validate_bound_file(
    root: Path,
    *,
    expected_relative: str,
    contract: Any,
) -> None:
    if not isinstance(contract, Mapping) or contract.get("path") != expected_relative:
        raise ValueError(f"Stage-1 bound path contract is invalid: {expected_relative}.")
    path = root / expected_relative
    if (
        not path.is_file()
        or path.is_symlink()
        or path.resolve() != (root.resolve() / expected_relative)
        or contract.get("sha256") != _sha256_file(path)
        or contract.get("size_bytes") != path.stat().st_size
    ):
        raise ValueError(f"Stage-1 bound artifact changed: {expected_relative}.")


def _bound_file_contract(path: Path, *, relative: str) -> dict[str, Any]:
    if not path.is_file() or path.is_symlink():
        raise ValueError(f"Stage-1 training artifact is missing or unsafe: {relative}.")
    return {
        "path": relative,
        "sha256": _sha256_file(path),
        "size_bytes": path.stat().st_size,
    }


def _validate_best_metric_artifact(
    path: Path,
    *,
    checkpoint_epoch: int,
    checkpoint_metric: float,
) -> None:
    if not path.is_file() or path.is_symlink():
        raise ValueError("Stage-1 validation metrics artifact is missing or unsafe.")
    metrics = pd.read_csv(path)
    if (
        not {"epoch", "val_accuracy"}.issubset(metrics.columns)
        or metrics.empty
        or metrics["epoch"].duplicated().any()
    ):
        raise ValueError("Stage-1 validation metrics artifact is malformed.")
    epochs = metrics["epoch"].to_numpy(float)
    values = metrics["val_accuracy"].to_numpy(float)
    if (
        not np.isfinite(epochs).all()
        or not np.isfinite(values).all()
        or np.any(epochs != np.floor(epochs))
    ):
        raise ValueError("Stage-1 validation metrics contain invalid values.")
    best_index = int(np.argmax(values))
    if (
        int(epochs[best_index]) != checkpoint_epoch
        or abs(float(values[best_index]) - checkpoint_metric) > 1e-12
    ):
        raise ValueError(
            "Stage-1 checkpoint selection does not match metrics_val.csv."
        )


def _validate_runtime_attempts(
    runtime_metrics: Mapping[str, Any], role_root: Path
) -> dict[str, Any]:
    attempts = runtime_metrics.get("runtime_attempts")
    if (
        runtime_metrics.get("attempt_count") != 1
        or not isinstance(attempts, list)
        or len(attempts) != 1
    ):
        raise ValueError("Stage-1 requires exactly one fresh runtime attempt per arm.")
    contract = attempts[0]
    if not isinstance(contract, Mapping):
        raise TypeError("Stage-1 runtime-attempt contract must be a mapping.")
    relative = contract.get("path")
    if relative != "runtime_attempts/attempt_000.json":
        raise ValueError("Stage-1 runtime-attempt path is noncanonical.")
    path = role_root / str(relative)
    if (
        not path.is_file()
        or path.is_symlink()
        or path.resolve().parent.parent != role_root.resolve()
        or contract.get("sha256") != _sha256_file(path)
    ):
        raise ValueError("Stage-1 runtime-attempt artifact is missing or stale.")
    attempt = json.loads(path.read_text(encoding="utf-8"))
    if (
        attempt.get("schema") != "cfeg.runtime-metrics.v1"
        or attempt.get("status") != "completed"
        or attempt.get("run_mode") != "training_and_validation"
        or attempt.get("requested_device") != "cuda"
        or attempt.get("execution_device_type") != "cuda"
        or attempt.get("resolved_device") != runtime_metrics.get("resolved_device")
        or attempt.get("cuda_oom") is not False
        or contract.get("status") != attempt.get("status")
        or contract.get("elapsed_time_sec") != attempt.get("elapsed_time_sec")
    ):
        raise ValueError("Stage-1 runtime-attempt content is invalid.")
    return {
        "path": f"{role_root.name}/runtime_attempts/attempt_000.json",
        "sha256": _sha256_file(path),
        "size_bytes": path.stat().st_size,
    }


def validate_stage1_analysis_tree(
    result: dict[str, Any], output_dir: str | Path
) -> dict[str, Any]:
    output = Path(output_dir)
    entries = list(output.iterdir())
    if {path.name for path in entries} != set(OUTPUT_FILES) or any(
        path.is_symlink() or not path.is_file() for path in entries
    ):
        raise ValueError("Stage-1 analysis tree has a missing or unexpected entry.")
    for name, expected in result["frames"].items():
        observed = pd.read_csv(output / name)
        try:
            pd.testing.assert_frame_equal(
                observed,
                expected.reset_index(drop=True),
                check_dtype=False,
                check_exact=False,
                rtol=1e-10,
                atol=1e-12,
            )
        except AssertionError as exc:
            raise ValueError(f"Stage-1 artifact {name} differs from recomputation.") from exc
    observed_model = json.loads(
        (output / "metadata_only_probe_model.json").read_text(encoding="utf-8")
    )
    if observed_model != _json_roundtrip(result["metadata_only_probe_model"]):
        raise ValueError("Stage-1 metadata-only model differs from recomputation.")
    receipt = json.loads((output / "stage1_receipt.json").read_text(encoding="utf-8"))
    if set(receipt) != {*result["receipt"], "artifacts"}:
        raise ValueError("Stage-1 receipt has a missing or unexpected field.")
    artifacts = receipt.get("artifacts")
    if not isinstance(artifacts, dict) or set(artifacts) != set(ANALYSIS_FILES):
        raise ValueError("Stage-1 receipt artifact inventory is invalid.")
    for name, contract in artifacts.items():
        path = output / name
        if (
            not isinstance(contract, dict)
            or contract.get("sha256") != _sha256_file(path)
            or contract.get("size_bytes") != path.stat().st_size
        ):
            raise ValueError("Stage-1 receipt contains a stale artifact fingerprint.")
    for field, expected in result["receipt"].items():
        if receipt.get(field) != _json_roundtrip(expected):
            raise ValueError(f"Stage-1 receipt field {field!r} differs from recomputation.")
    return receipt


@torch.no_grad()
def _collect_prediction_frame(
    context: dict[str, Any],
    indices: np.ndarray,
    *,
    role: str,
    scenario: str,
    checkpoint_sha256: str,
    best_epoch: int,
    external_overrides: Mapping[str, Mapping[str, Any]] | None = None,
) -> pd.DataFrame:
    model = context["model"]
    device = context["device"]
    model.eval()
    rows: list[dict[str, Any]] = []
    manifest = context["manifest"].set_index("sample_id", drop=False)
    for batch in _loader(context, indices):
        batch = _to_device(batch, device)
        if external_overrides is not None:
            batch["cond"] = overwrite_external_metadata(
                batch["cond"],
                list(batch["sample_id"]),
                external_overrides,
                context["vocab"],
            )
        out = model(batch["x"], batch["cond"], use_latent=False)
        logits = out.logits.detach().float().cpu().numpy()
        if logits.ndim != 2 or not np.isfinite(logits).all():
            raise ValueError("Stage-1 prediction logits must be finite [N,K].")
        probabilities = _softmax(logits)
        labels = batch["y"].detach().cpu().numpy().astype(int)
        aux = _per_sample_aux(
            out.aux,
            len(labels),
            channel_mask=batch["cond"]["channel_mask"],
        )
        for index, sample_id in enumerate(batch["sample_id"]):
            meta = manifest.loc[str(sample_id)]
            row = {
                "role": role,
                "scenario": scenario,
                "sample_id": str(sample_id),
                "subject_id": str(meta["subject_id"]),
                "electrode_type": str(meta["electrode_type"]),
                "acquisition_block_id": str(meta["acquisition_block_id"]),
                "label": int(labels[index]),
                "predicted_label": int(np.argmax(logits[index])),
                "confidence": float(probabilities[index].max()),
                "best_epoch": int(best_epoch),
                "checkpoint_sha256": checkpoint_sha256,
                **{
                    f"logit_{class_index}": float(value)
                    for class_index, value in enumerate(logits[index])
                },
                **{key: values[index] for key, values in aux.items()},
            }
            rows.append(row)
    frame = pd.DataFrame(rows)
    if frame["sample_id"].duplicated().any() or len(frame) != len(indices):
        raise ValueError("Stage-1 prediction sample identities are incomplete or duplicated.")
    return frame


def _per_sample_aux(
    aux: Mapping[str, torch.Tensor],
    n_samples: int,
    *,
    channel_mask: torch.Tensor,
) -> dict[str, list[float]]:
    result: dict[str, list[float]] = {}
    for key in _AUX_FIELDS:
        value = aux.get(key)
        if not torch.is_tensor(value):
            raise ValueError(f"Stage-1 reliability output is missing diagnostic {key!r}.")
        flattened = value.detach().float().reshape(-1).cpu().numpy()
        if len(flattened) == 1:
            flattened = np.repeat(flattened, n_samples)
        if len(flattened) != n_samples or not np.isfinite(flattened).all():
            raise ValueError(f"Stage-1 diagnostic {key!r} is not one finite value per sample.")
        result[key] = [float(item) for item in flattened]
    gain = aux.get("channel_spatial_self_gain")
    if not torch.is_tensor(gain) or gain.shape != channel_mask.shape:
        raise ValueError("Stage-1 reliability output lacks per-channel self gain.")
    active_gain = gain.detach().float().masked_fill(~channel_mask.bool(), torch.nan)
    active_min = torch.nan_to_num(active_gain, nan=float("inf")).min(dim=-1).values
    active_max = torch.nan_to_num(active_gain, nan=float("-inf")).max(dim=-1).values
    if not torch.isfinite(active_min).all() or not torch.isfinite(active_max).all():
        raise ValueError("Stage-1 active-channel gain diagnostics are non-finite.")
    result["active_spatial_self_gain_min"] = [
        float(item) for item in active_min.cpu().numpy()
    ]
    result["active_spatial_self_gain_max"] = [
        float(item) for item in active_max.cpu().numpy()
    ]
    query_noise = aux.get("channel_query_noise_logit")
    if not torch.is_tensor(query_noise) or query_noise.shape != channel_mask.shape:
        raise ValueError("Stage-1 reliability output lacks per-channel Q logits.")
    query_noise = query_noise.detach().float().cpu().contiguous().numpy()
    result["channel_query_noise_logit_sha256"] = [
        hashlib.sha256(row.tobytes()).hexdigest() for row in query_noise
    ]
    for channel_index in range(query_noise.shape[1]):
        result[f"query_noise_logit_{channel_index}"] = [
            float(item) for item in query_noise[:, channel_index]
        ]
    return result


def _validate_arm_prediction_parity(frame: pd.DataFrame) -> None:
    reference = frame.loc[frame["role"].eq(RELIABILITY_ROLES[0])].sort_values(
        "sample_id"
    )
    keys = reference[["sample_id", "label", "subject_id", "electrode_type"]].reset_index(
        drop=True
    )
    if len(keys) == 0:
        raise ValueError("Stage-1 validation prediction set is empty.")
    for role in RELIABILITY_ROLES[1:]:
        observed = frame.loc[frame["role"].eq(role)].sort_values("sample_id")
        if not observed[
            ["sample_id", "label", "subject_id", "electrode_type"]
        ].reset_index(drop=True).equals(keys):
            raise ValueError("Stage-1 arms do not use identical validation samples and labels.")


def _group_metrics(frame: pd.DataFrame, *, group_columns: list[str]) -> pd.DataFrame:
    logit_columns = sorted(
        (name for name in frame.columns if name.startswith("logit_")),
        key=lambda name: int(name.split("_")[1]),
    )
    rows: list[dict[str, Any]] = []
    grouping: str | list[str] = group_columns[0] if len(group_columns) == 1 else group_columns
    for key, group in frame.groupby(grouping, sort=True):
        key_values = (key,) if len(group_columns) == 1 else tuple(key)
        metrics = classification_metrics(
            group["label"].to_numpy(int),
            group[logit_columns].to_numpy(float),
        )
        rows.append({**dict(zip(group_columns, key_values)), **metrics})
    return pd.DataFrame(rows)


def _subject_contrasts(subject_metrics: pd.DataFrame) -> pd.DataFrame:
    pivot = subject_metrics.pivot(
        index="subject_id", columns="role", values="balanced_accuracy"
    )
    if set(pivot.columns) != set(RELIABILITY_ROLES) or pivot.isna().any().any():
        raise ValueError("Stage-1 subject contrast grid is incomplete.")
    output = pivot.loc[:, list(RELIABILITY_ROLES)].reset_index()
    output.columns = ["subject_id", *[f"balanced_accuracy_{role}" for role in RELIABILITY_ROLES]]
    output["delta_metadata_given_q"] = (
        output["balanced_accuracy_A_QM"] - output["balanced_accuracy_A_Q"]
    )
    output["delta_metadata_without_q"] = (
        output["balanced_accuracy_A_M"] - output["balanced_accuracy_A0"]
    )
    output["interaction"] = (
        output["delta_metadata_given_q"] - output["delta_metadata_without_q"]
    )
    return output


def _build_intervention_overrides(
    manifest: pd.DataFrame,
) -> tuple[dict[str, dict[str, dict[str, Any]]], pd.DataFrame]:
    required = {
        "sample_id",
        "subject_id",
        "run_id",
        "electrode_type",
        "acquisition_block_id",
        "metadata_measurement_id",
        *M_FIELDS,
    }
    missing = sorted(required - set(manifest.columns))
    if missing:
        raise ValueError(f"Stage-1 intervention manifest is missing {missing}.")
    block_rows: dict[tuple[str, str, str], dict[str, Any]] = {}
    for key, group in manifest.groupby(
        ["subject_id", "electrode_type", "run_id"], sort=True
    ):
        hashes = {_metadata_bundle_sha256(row) for _, row in group.iterrows()}
        if len(hashes) != 1:
            raise ValueError("Stage-1 intervention metadata is not block-constant.")
        block_rows[tuple(str(value) for value in key)] = group.iloc[0].to_dict()
    grouped_runs: dict[tuple[str, str], list[str]] = {}
    for subject, electrode, run_id in block_rows:
        grouped_runs.setdefault((subject, electrode), []).append(run_id)
    for key, values in grouped_runs.items():
        runs = sorted(set(values))
        if len(runs) < 2:
            raise ValueError(f"Stage-1 stale-block control is not identifiable for {key}.")
        grouped_runs[key] = runs

    overrides = {name: {} for name in ("missing", "stale_block", "wrong_interface")}
    mapping_rows: list[dict[str, Any]] = []
    for _, target in manifest.sort_values("sample_id").iterrows():
        sample_id = str(target["sample_id"])
        subject = str(target["subject_id"])
        electrode = str(target["electrode_type"])
        run_id = str(target["run_id"])
        target_hash = _metadata_bundle_sha256(target)
        missing_source = target.to_dict()
        for field in M_FIELDS:
            missing_source[field] = None
        overrides["missing"][sample_id] = missing_source
        mapping_rows.append(
            _mapping_row(
                target,
                scenario="missing",
                donor=None,
                target_hash=target_hash,
                donor_hash=_metadata_bundle_sha256(missing_source),
            )
        )

        runs = grouped_runs[(subject, electrode)]
        donor_run = runs[(runs.index(run_id) + 1) % len(runs)]
        stale_donor = block_rows[(subject, electrode, donor_run)]
        stale_source = _replace_metadata_bundle(target, stale_donor)
        overrides["stale_block"][sample_id] = stale_source
        mapping_rows.append(
            _mapping_row(
                target,
                scenario="stale_block",
                donor=stale_donor,
                target_hash=target_hash,
                donor_hash=_metadata_bundle_sha256(stale_source),
            )
        )

        opposite = "dry" if electrode == "wet" else "wet"
        wrong_key = (subject, opposite, run_id)
        if wrong_key not in block_rows:
            raise ValueError("Stage-1 wrong-interface donor block is missing.")
        wrong_donor = block_rows[wrong_key]
        wrong_source = _replace_metadata_bundle(target, wrong_donor)
        overrides["wrong_interface"][sample_id] = wrong_source
        mapping_rows.append(
            _mapping_row(
                target,
                scenario="wrong_interface",
                donor=wrong_donor,
                target_hash=target_hash,
                donor_hash=_metadata_bundle_sha256(wrong_source),
            )
        )
    mapping = pd.DataFrame(mapping_rows)
    for scenario in ("stale_block", "wrong_interface"):
        selected = mapping.loc[mapping["scenario"].eq(scenario)]
        if not selected["bundle_changed"].all() or selected["mapping_fixed_point"].any():
            raise ValueError(f"Stage-1 {scenario} intervention lacks full mapping potency.")
    return overrides, mapping


def _replace_metadata_bundle(
    target: pd.Series, donor: Mapping[str, Any]
) -> dict[str, Any]:
    output = target.to_dict()
    for field in M_FIELDS:
        output[field] = donor[field]
    return output


def _mapping_row(
    target: pd.Series,
    *,
    scenario: str,
    donor: Mapping[str, Any] | None,
    target_hash: str,
    donor_hash: str,
) -> dict[str, Any]:
    donor_block = None if donor is None else str(donor["acquisition_block_id"])
    donor_subject = None if donor is None else str(donor["subject_id"])
    donor_electrode = None if donor is None else str(donor["electrode_type"])
    donor_run = None if donor is None else str(donor["run_id"])
    donor_measurement = None if donor is None else str(donor["metadata_measurement_id"])
    target_block = str(target["acquisition_block_id"])
    target_electrode = str(target["electrode_type"])
    target_payload = _metadata_bundle_payload(target)
    donor_payload = (
        {field: None for field in M_FIELDS}
        if donor is None
        else _metadata_bundle_payload(donor)
    )
    return {
        "scenario": scenario,
        "target_sample_id": str(target["sample_id"]),
        "subject_id": str(target["subject_id"]),
        "donor_subject_id": donor_subject,
        "target_acquisition_block_id": target_block,
        "target_metadata_measurement_id": str(target["metadata_measurement_id"]),
        "target_run_id": str(target["run_id"]),
        "target_electrode_type": target_electrode,
        "donor_acquisition_block_id": donor_block,
        "donor_metadata_measurement_id": donor_measurement,
        "donor_run_id": donor_run,
        "donor_electrode_type": donor_electrode,
        "target_metadata_bundle_sha256": target_hash,
        "donor_metadata_bundle_sha256": donor_hash,
        "bundle_changed": target_hash != donor_hash,
        "mapping_fixed_point": bool(
            donor_block == target_block and donor_electrode == target_electrode
        ),
        **{
            f"{field}_changed": target_payload[field] != donor_payload[field]
            for field in M_FIELDS
        },
        "label_used_for_pairing": False,
    }


def _metadata_bundle_sha256(row: Mapping[str, Any] | pd.Series) -> str:
    return _sha256_json(_metadata_bundle_payload(row))


def _metadata_bundle_payload(row: Mapping[str, Any] | pd.Series) -> dict[str, Any]:
    return {field: _semantic_value(row.get(field)) for field in M_FIELDS}


def _validate_intervention_predictions(
    predictions: pd.DataFrame,
    mapping: pd.DataFrame,
    *,
    tolerance: float,
) -> None:
    scenarios = ("correct", "missing", "stale_block", "wrong_interface")
    reference = predictions.loc[predictions["scenario"].eq("correct")].sort_values(
        "sample_id"
    )
    checkpoint_hashes = set(reference["checkpoint_sha256"])
    for scenario in scenarios[1:]:
        observed = predictions.loc[predictions["scenario"].eq(scenario)].sort_values(
            "sample_id"
        )
        if not observed[["sample_id", "label"]].reset_index(drop=True).equals(
            reference[["sample_id", "label"]].reset_index(drop=True)
        ):
            raise ValueError("Stage-1 metadata interventions changed sample or label identity.")
        checkpoint_hashes.update(observed["checkpoint_sha256"])
    if len(checkpoint_hashes) != 1:
        raise ValueError("Stage-1 metadata interventions used different A_QM checkpoints.")
    scalar_query_invariants = (
        "query_spatial_operator_frobenius_norm",
        "channel_query_available_count",
        "metadata_residual_alpha",
    )
    vector_query_invariants = sorted(
        name for name in predictions.columns if name.startswith("query_noise_logit_")
    )
    query_invariants = (*scalar_query_invariants, *vector_query_invariants)
    reference_query = reference[["sample_id", *query_invariants]].reset_index(drop=True)
    for scenario in scenarios[1:]:
        observed_query = (
            predictions.loc[predictions["scenario"].eq(scenario)]
            .sort_values("sample_id")[["sample_id", *query_invariants]]
            .reset_index(drop=True)
        )
        if not observed_query["sample_id"].equals(reference_query["sample_id"]) or not np.allclose(
            observed_query[list(query_invariants)].to_numpy(float),
            reference_query[list(query_invariants)].to_numpy(float),
            rtol=0.0,
            atol=tolerance,
        ):
            raise ValueError("Stage-1 metadata intervention changed the independent Q branch.")
    if mapping.duplicated(["scenario", "target_sample_id"]).any() or len(mapping) != (
        3 * len(reference)
    ):
        raise ValueError("Stage-1 intervention mapping coverage is incomplete or duplicated.")
    missing = predictions.loc[predictions["scenario"].eq("missing")]
    if (
        missing["channel_metadata_available_count"].abs().max() > tolerance
        or missing["metadata_spatial_residual_frobenius_norm"].abs().max()
        > tolerance
    ):
        raise ValueError("Stage-1 missing-M intervention is not an exact metadata fallback.")
    for scenario in ("stale_block", "wrong_interface"):
        selected = mapping.loc[mapping["scenario"].eq(scenario)]
        if len(selected) != len(reference):
            raise ValueError("Stage-1 intervention mapping row count is incomplete.")
        if (
            selected["mapping_fixed_point"].astype(bool).any()
            or not selected["bundle_changed"].astype(bool).all()
            or selected["label_used_for_pairing"].astype(bool).any()
        ):
            raise ValueError("Stage-1 intervention mapping integrity failed.")
    stale = mapping.loc[mapping["scenario"].eq("stale_block")]
    if not (
        stale["subject_id"].eq(stale["donor_subject_id"]).all()
        and
        stale["target_electrode_type"].eq(stale["donor_electrode_type"]).all()
        and stale["target_run_id"].ne(stale["donor_run_id"]).all()
    ):
        raise ValueError("Stage-1 stale-block mapping crossed its subject/interface scope.")
    wrong = mapping.loc[mapping["scenario"].eq("wrong_interface")]
    if not (
        wrong["subject_id"].eq(wrong["donor_subject_id"]).all()
        and
        wrong["target_run_id"].eq(wrong["donor_run_id"]).all()
        and wrong["target_electrode_type"].ne(wrong["donor_electrode_type"]).all()
    ):
        raise ValueError("Stage-1 wrong-interface mapping lacks its exact interface flip.")


def _metadata_safety(
    subject_metrics: pd.DataFrame,
    a_q_subject_metrics: pd.DataFrame,
) -> pd.DataFrame:
    pivot = subject_metrics.pivot(
        index="subject_id", columns="scenario", values="balanced_accuracy"
    )
    scenarios = ["correct", "missing", "stale_block", "wrong_interface"]
    if set(pivot.columns) != set(scenarios) or pivot.isna().any().any():
        raise ValueError("Stage-1 metadata-safety subject grid is incomplete.")
    output = pivot.loc[:, scenarios].reset_index()
    output.columns = ["subject_id", *[f"balanced_accuracy_{name}" for name in scenarios]]
    a_q = a_q_subject_metrics.rename(
        columns={"balanced_accuracy": "balanced_accuracy_A_Q"}
    )
    if a_q["subject_id"].duplicated().any() or set(a_q["subject_id"]) != set(
        output["subject_id"]
    ):
        raise ValueError("Stage-1 A_Q reference subject grid is incomplete.")
    output = output.merge(a_q, on="subject_id", validate="one_to_one")
    output["correct_minus_missing"] = (
        output["balanced_accuracy_correct"] - output["balanced_accuracy_missing"]
    )
    output["correct_minus_stale_block"] = (
        output["balanced_accuracy_correct"] - output["balanced_accuracy_stale_block"]
    )
    output["correct_minus_wrong_interface"] = (
        output["balanced_accuracy_correct"]
        - output["balanced_accuracy_wrong_interface"]
    )
    for scenario in scenarios:
        output[f"{scenario}_minus_A_Q"] = (
            output[f"balanced_accuracy_{scenario}"] - output["balanced_accuracy_A_Q"]
        )
    output["worst_intervened_balanced_accuracy"] = output[
        [
            "balanced_accuracy_missing",
            "balanced_accuracy_stale_block",
            "balanced_accuracy_wrong_interface",
        ]
    ].min(axis=1)
    return output


def _run_metadata_only_probe(
    manifest: pd.DataFrame,
    *,
    train_indices: np.ndarray,
    val_indices: np.ndarray,
    n_classes: int,
    active_channel_ids: list[int],
    ridge_alphas: tuple[float, ...],
    invariance_tolerance: float,
    chance_tolerance: float,
) -> dict[str, Any]:
    features, feature_names = _metadata_feature_matrix(
        manifest,
        active_channel_ids=active_channel_ids,
    )
    if set(feature_names) & PREDICTOR_FORBIDDEN:
        raise ValueError("Metadata-only probe feature allowlist contains a forbidden predictor.")
    labels = manifest["label"].to_numpy(int)
    subjects = manifest["subject_id"].astype(str).to_numpy()
    train_subjects = sorted(set(subjects[train_indices]))
    inner_scores: dict[str, float] = {}
    for alpha in ridge_alphas:
        fold_scores = []
        for held_subject in train_subjects:
            inner_test = train_indices[subjects[train_indices] == held_subject]
            inner_train = train_indices[subjects[train_indices] != held_subject]
            model = _fit_multiclass_ridge(
                features[inner_train], labels[inner_train], n_classes=n_classes, alpha=alpha
            )
            logits = _predict_multiclass_ridge(model, features[inner_test])
            one_hot = np.eye(n_classes, dtype=float)[labels[inner_test]]
            fold_scores.append(float(np.mean(np.square(logits - one_hot))))
        inner_scores[f"{alpha:.12g}"] = float(np.mean(fold_scores))
    selected_alpha = min(
        ridge_alphas,
        key=lambda alpha: (inner_scores[f"{alpha:.12g}"], alpha),
    )
    model = _fit_multiclass_ridge(
        features[train_indices],
        labels[train_indices],
        n_classes=n_classes,
        alpha=selected_alpha,
    )
    logits = _predict_multiclass_ridge(model, features[val_indices])
    predictions = manifest.iloc[val_indices][
        [
            "sample_id",
            "subject_id",
            "electrode_type",
            "acquisition_block_id",
            "label",
        ]
    ].reset_index(drop=True)
    predictions["predicted_label"] = logits.argmax(axis=1)
    for class_index in range(n_classes):
        predictions[f"logit_{class_index}"] = logits[:, class_index]
    logit_columns = [f"logit_{index}" for index in range(n_classes)]
    block_ranges = predictions.groupby("acquisition_block_id")[logit_columns].agg(
        lambda values: float(values.max() - values.min())
    )
    if block_ranges.to_numpy(float).max(initial=0.0) > invariance_tolerance:
        raise ValueError("Metadata-only probe logits vary within an acquisition block.")
    subject_metrics = _group_metrics(predictions, group_columns=["subject_id"])
    chance = 1.0 / n_classes
    if not np.allclose(
        subject_metrics["balanced_accuracy"].to_numpy(float),
        chance,
        rtol=0.0,
        atol=chance_tolerance,
    ):
        raise ValueError("Metadata-only probe does not equal the balanced-class chance rate.")
    return {
        "predictions": predictions,
        "subject_metrics": subject_metrics,
        "model": {
            "schema": "cfeg.synthetic-reliability-metadata-only-ridge.v1",
            "feature_names": feature_names,
            "forbidden_predictors": sorted(PREDICTOR_FORBIDDEN),
            "selected_alpha": float(selected_alpha),
            "inner_subject_macro_mse_by_alpha": inner_scores,
            "mean": model["mean"].tolist(),
            "scale": model["scale"].tolist(),
            "coefficient": model["coefficient"].tolist(),
            "predictor_allowlist_only": True,
            "block_logit_invariance_verified": True,
            "balanced_accuracy_equals_chance": True,
        },
    }


def _metadata_feature_matrix(
    manifest: pd.DataFrame, *, active_channel_ids: list[int]
) -> tuple[np.ndarray, list[str]]:
    names = [
        "m_impedance_mean",
        "m_impedance_max",
        "m_impedance_mean_available",
        "m_impedance_max_available",
        "m_electrode_wet",
        "m_electrode_dry",
    ]
    names.extend(f"m_channel_{channel_id}_impedance" for channel_id in active_channel_ids)
    names.extend(
        f"m_channel_{channel_id}_impedance_available" for channel_id in active_channel_ids
    )
    rows: list[list[float]] = []
    for _, row in manifest.iterrows():
        mean = _finite_or_none(row["impedance_mean_kohm"])
        maximum = _finite_or_none(row["impedance_max_kohm"])
        canonical_ids = np.asarray(row["canonical_channel_ids"], dtype=int)
        impedance = np.asarray(row["impedance_kohm_by_channel"], dtype=float)
        channel_values: list[float] = []
        channel_available: list[float] = []
        for channel_id in active_channel_ids:
            slots = np.flatnonzero(canonical_ids == int(channel_id))
            if len(slots) != 1:
                raise ValueError("Metadata-only probe channel identity is missing or duplicated.")
            value = _finite_or_none(impedance[int(slots[0])])
            channel_values.append(_norm_impedance(value))
            channel_available.append(float(value is not None))
        electrode = str(row["electrode_type"])
        rows.append(
            [
                _norm_impedance(mean),
                _norm_impedance(maximum),
                float(mean is not None),
                float(maximum is not None),
                float(electrode == "wet"),
                float(electrode == "dry"),
                *channel_values,
                *channel_available,
            ]
        )
    output = np.asarray(rows, dtype=float)
    if not np.isfinite(output).all():
        raise ValueError("Metadata-only probe features contain non-finite values.")
    return output, names


def _fit_multiclass_ridge(
    x: np.ndarray,
    y: np.ndarray,
    *,
    n_classes: int,
    alpha: float,
) -> dict[str, np.ndarray | float]:
    mean = x.mean(axis=0)
    scale = x.std(axis=0)
    scale = np.where(scale > 1e-8, scale, 1.0)
    design = np.column_stack([np.ones(len(x)), (x - mean) / scale])
    target = np.eye(n_classes, dtype=float)[np.asarray(y, dtype=int)]
    penalty = np.eye(design.shape[1], dtype=float) * float(alpha)
    penalty[0, 0] = 0.0
    coefficient = np.linalg.pinv(design.T @ design + penalty) @ design.T @ target
    return {
        "mean": mean,
        "scale": scale,
        "coefficient": coefficient,
        "alpha": float(alpha),
    }


def _predict_multiclass_ridge(
    model: Mapping[str, Any], x: np.ndarray
) -> np.ndarray:
    design = np.column_stack(
        [np.ones(len(x)), (x - model["mean"]) / model["scale"]]
    )
    return np.asarray(design @ model["coefficient"], dtype=float)


def _validate_analysis_contract(contract: Mapping[str, Any]) -> None:
    expected = {
        "schema": STAGE1_ANALYSIS_SCHEMA,
        "score_scope": "checkpoint_selection_validation_set",
        "statistical_unit": "participant",
        "primary_metric": "subject_macro_balanced_accuracy",
        "substantive_effect_gate": "none_descriptive_only",
        "interventions": ["missing", "stale_block", "wrong_interface"],
        "external_metadata_bundle_fields": [
            "reference",
            "electrode_type",
            "cap_type",
            "impedance_mean_kohm",
            "impedance_max_kohm",
            "impedance_kohm_by_channel",
        ],
        "missing_intervention": "full_external_bundle_null",
        "stale_block_donor_scope": "same_subject_same_interface_other_run",
        "wrong_interface_donor_scope": "same_subject_same_run_opposite_interface",
        "donor_pairing_uses_label": False,
        "same_aqm_checkpoint_for_interventions": True,
        "missing_fallback_interpretation": (
            "same_aqm_checkpoint_with_metadata_operator_zero_not_separately_trained_a_q"
        ),
        "a_q_comparator": "descriptive_subject_level",
        "integrity_tolerance": 1e-6,
        "metadata_only_probe": {
            "method": "participant_disjoint_multiclass_ridge",
            "ridge_alphas": [1e-6, 0.0001, 0.01, 1.0, 10.0, 100.0],
            "active_canonical_channel_ids": [48, 55, 57, 56, 53, 61, 62, 63],
            "block_logit_invariance_tolerance": 1e-12,
            "balanced_accuracy_chance_tolerance": 1e-12,
        },
        "output_files": list(OUTPUT_FILES),
    }
    if dict(contract) != expected:
        raise ValueError("Stage-1 analysis contract is incomplete or stale.")


def _softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - logits.max(axis=1, keepdims=True)
    probabilities = np.exp(shifted)
    return probabilities / probabilities.sum(axis=1, keepdims=True)


def _finite_or_none(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if np.isfinite(number) else None


def _sample_identity_sha256(sample_ids: list[str]) -> str:
    return _sha256_json([str(value) for value in sample_ids])


def _semantic_value(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return [_semantic_value(item) for item in value.tolist()]
    if isinstance(value, (list, tuple)):
        return [_semantic_value(item) for item in value]
    if isinstance(value, np.generic):
        return _semantic_value(value.item())
    if value is None:
        return None
    try:
        if bool(pd.isna(value)):
            return None
    except (TypeError, ValueError):
        pass
    if isinstance(value, float):
        return round(value, 12)
    return value


def _json_roundtrip(value: Any) -> Any:
    return json.loads(json.dumps(value, sort_keys=True, allow_nan=False, default=_json_default))


def _json_default(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"Value is not JSON serializable: {type(value).__name__}.")


def _sha256_json(value: Any) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
        default=_json_default,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_csv_fsynced(frame: pd.DataFrame, path: Path) -> None:
    frame.to_csv(path, index=False)
    with path.open("rb") as handle:
        os.fsync(handle.fileno())


def _write_json_fsynced(path: Path, value: Any) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, allow_nan=False, default=_json_default)
        handle.flush()
        os.fsync(handle.fileno())


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
