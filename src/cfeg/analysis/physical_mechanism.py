from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Sequence
from pathlib import Path

import pandas as pd

from cfeg.metrics import balanced_accuracy

PHYSICAL_MECHANISM_ROLES = (
    "A0_eeg_only",
    "A2_structured_condition_prompt",
    "M1_global_only",
    "M2_channel_only",
    "M3_full_shuffle_train",
    "M4_metadata_only",
)

_ROLE_COLUMNS = {
    "A0_eeg_only": "a0",
    "A2_structured_condition_prompt": "full_a2",
    "M1_global_only": "global_only",
    "M2_channel_only": "channel_only",
    "M3_full_shuffle_train": "shuffle_train",
    "M4_metadata_only": "metadata_only",
}

_DIAGNOSTIC_FOLDS = (0, 1, 2)
_NUMERIC_COMPARISON_TOLERANCE = 1e-12


def aggregate_complete_physical_mechanism(
    jobs: Sequence[dict],
    manifest: pd.DataFrame,
    *,
    expected_subjects: Sequence[str],
    expected_n_samples: int,
) -> tuple[pd.DataFrame, dict[str, object]]:
    """Aggregate one complete six-role S1--S3 mechanism grid.

    This is a wiring/reliance diagnostic. N=3 population inference is forbidden.
    """

    expected_keys = {(42, fold, role) for fold in range(3) for role in PHYSICAL_MECHANISM_ROLES}
    observed_keys = {(int(job["seed"]), int(job["fold_index"]), str(job["role"])) for job in jobs}
    if len(jobs) != len(expected_keys) or observed_keys != expected_keys:
        raise ValueError("Physical mechanism grid must contain the exact 18 fold-role jobs.")

    metadata = manifest.loc[
        manifest["subject_id"].astype(str).isin({str(value) for value in expected_subjects})
    ].copy()
    metadata["sample_id"] = metadata["sample_id"].astype(str)
    if len(metadata) != expected_n_samples or metadata["sample_id"].duplicated().any():
        raise ValueError("Development manifest does not match the frozen S1-S3 cohort.")

    frames: dict[tuple[int, str], pd.DataFrame] = {}
    coverage: dict[str, set[str]] = {role: set() for role in PHYSICAL_MECHANISM_ROLES}
    held_subjects: dict[str, list[str]] = {role: [] for role in PHYSICAL_MECHANISM_ROLES}
    for job in jobs:
        path = Path(job["prediction_csv"])
        provenance_path = path.with_name(f"{path.stem}_provenance.json")
        if not path.is_file() or not provenance_path.is_file():
            raise ValueError("Physical mechanism predictions are incomplete; outcomes stay sealed.")
        provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
        if provenance.get("prediction_csv_sha256") != _sha256_file(path):
            raise ValueError(f"Development prediction artifact changed: {path}.")
        frame = pd.read_csv(path)
        required = {
            "sample_id",
            "label",
            "prediction",
            "selection_split",
            "primary_ablation_role",
            "optimization_seed",
            "fold_index",
            "checkpoint_role",
            "checkpoint_selection_metric",
            "conditioning_architecture",
        }
        missing = required - set(frame.columns)
        if missing:
            raise ValueError(f"Physical prediction is missing columns: {sorted(missing)}.")
        role = str(job["role"])
        fold = int(job["fold_index"])
        if (
            set(frame["selection_split"].astype(str)) != {"val"}
            or set(frame["primary_ablation_role"].astype(str)) != {role}
            or set(frame["optimization_seed"].astype(int)) != {42}
            or set(frame["fold_index"].astype(int)) != {fold}
            or set(frame["checkpoint_role"].astype(str)) != {"development_fixed_epoch"}
            or set(frame["checkpoint_selection_metric"].astype(str)) != {"fixed_epoch"}
            or set(frame["conditioning_architecture"].astype(str)) != {"physical_hybrid_v1"}
        ):
            raise ValueError(f"Physical prediction contract mismatch for fold={fold}, role={role}.")
        frame["sample_id"] = frame["sample_id"].astype(str)
        if frame["sample_id"].duplicated().any() or coverage[role] & set(frame["sample_id"]):
            raise ValueError(f"Development held-out samples overlap for role={role}.")
        joined = frame.merge(
            metadata[["sample_id", "subject_id"]], on="sample_id", validate="one_to_one"
        )
        if len(joined) != len(frame) or joined["subject_id"].nunique() != 1:
            raise ValueError("Each physical mechanism fold must hold out exactly one subject.")
        coverage[role].update(frame["sample_id"])
        held_subjects[role].append(str(joined["subject_id"].iloc[0]))
        frames[(fold, role)] = joined

    expected_ids = set(metadata["sample_id"])
    for role in PHYSICAL_MECHANISM_ROLES:
        if coverage[role] != expected_ids or sorted(held_subjects[role]) != sorted(
            str(value) for value in expected_subjects
        ):
            raise ValueError(f"Physical mechanism coverage is incomplete for role={role}.")

    rows: list[dict[str, object]] = []
    for fold in range(3):
        reference = frames[(fold, PHYSICAL_MECHANISM_ROLES[0])][
            ["sample_id", "subject_id", "label"]
        ].copy()
        row: dict[str, object] = {
            "subject_id": str(reference["subject_id"].iloc[0]),
            "fold_index": fold,
            "n_samples": len(reference),
        }
        for role in PHYSICAL_MECHANISM_ROLES:
            frame = frames[(fold, role)]
            paired = reference[["sample_id", "label"]].merge(
                frame[["sample_id", "label", "prediction"]],
                on="sample_id",
                suffixes=("_reference", "_role"),
                validate="one_to_one",
            )
            if len(paired) != len(reference) or not paired["label_reference"].equals(
                paired["label_role"]
            ):
                raise ValueError(f"Physical mechanism labels differ for fold={fold}, role={role}.")
            row[f"{_ROLE_COLUMNS[role]}_balanced_accuracy"] = balanced_accuracy(
                paired["label_reference"], paired["prediction"], 12
            )
        row.update(_mechanism_deltas(row))
        rows.append(row)

    subjects = pd.DataFrame(rows).sort_values("subject_id").reset_index(drop=True)
    metric_columns = [column for column in subjects if column.endswith("balanced_accuracy")]
    delta_columns = [column for column in subjects if column.endswith("_delta")]
    summary = {
        "schema": "cfeg.physical-mechanism-summary.v1",
        "status": "descriptive_mechanism_assay_only",
        "confirmatory_claim_allowed": False,
        "population_inference_allowed": False,
        "n_subjects": 3,
        "n_samples_per_role": expected_n_samples,
        "n_training_jobs": len(expected_keys),
        "missingness_only_status": "invalid_assay_single_natural_pattern",
        "mean_metrics": {
            column: float(subjects[column].mean()) for column in [*metric_columns, *delta_columns]
        },
        "warning": (
            "N=3 can diagnose wiring, reliance, and catastrophic harm only; it cannot "
            "establish population benefit, equivalence, or architecture superiority."
        ),
    }
    return subjects, summary


def _mechanism_deltas(row: dict[str, object]) -> dict[str, float]:
    a0 = float(row["a0_balanced_accuracy"])
    full = float(row["full_a2_balanced_accuracy"])
    global_only = float(row["global_only_balanced_accuracy"])
    channel_only = float(row["channel_only_balanced_accuracy"])
    shuffle_train = float(row["shuffle_train_balanced_accuracy"])
    metadata_only = float(row["metadata_only_balanced_accuracy"])
    return {
        "full_minus_a0_delta": full - a0,
        "global_minus_a0_delta": global_only - a0,
        "channel_minus_a0_delta": channel_only - a0,
        "full_minus_global_delta": full - global_only,
        "full_minus_channel_delta": full - channel_only,
        "full_minus_shuffle_train_delta": full - shuffle_train,
        "metadata_only_minus_chance_delta": metadata_only - (1.0 / 12.0),
    }


def evaluate_predeclared_physical_gates(
    subjects: pd.DataFrame,
    interventions: pd.DataFrame,
    training_controls: pd.DataFrame,
    control_plan: dict,
) -> dict[str, object]:
    """Apply the frozen N=3 mechanism/safety gates without population inference.

    Every effect gate uses the least-favourable observed development subject/fold.
    Passing is therefore a development diagnostic needed before owner review, not a
    confidence interval, a population claim, or permission to open the lockbox.
    """

    margin_keys = (
        "clean_a2_mean_minimum_delta",
        "clean_a2_subject_harm_margin",
        "shortcut_equivalence_margin",
        "pairing_mechanism_margin",
        "counterfactual_mechanism_margin",
        "wrong_metadata_safety_harm_margin",
        "minimum_bundle_changed_fraction",
        "minimum_condition_flip_fraction",
    )
    try:
        margins = {key: float(control_plan[key]) for key in margin_keys}
        chance = float(control_plan["chance_balanced_accuracy"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("Physical diagnostic margins must be frozen numeric values.") from exc
    for key in margin_keys[:6]:
        if not 0.0 <= margins[key] <= 1.0:
            raise ValueError(f"{key} must be within [0, 1].")
    for key in margin_keys[4:]:
        if not 0.0 < margins[key] <= 1.0:
            raise ValueError(f"{key} must be within (0, 1].")
    if not 0.0 < chance < 1.0:
        raise ValueError("chance_balanced_accuracy must be within (0, 1).")
    if control_plan.get("confirmatory_control_policy") != "development_gate_only":
        raise ValueError("Only development_gate_only is implemented for physical controls.")

    subject_required = {
        "subject_id",
        "fold_index",
        "a0_balanced_accuracy",
        "full_a2_balanced_accuracy",
        "shuffle_train_balanced_accuracy",
        "metadata_only_balanced_accuracy",
    }
    missing = subject_required - set(subjects.columns)
    if missing:
        raise ValueError(f"Physical subject table is missing gate columns: {sorted(missing)}.")
    subject_rows = subjects.copy()
    subject_rows["fold_index"] = subject_rows["fold_index"].astype(int)
    _require_exact_fold_rows(subject_rows, name="physical subject table")

    intervention_required = {
        "fold_index",
        "scenario",
        "balanced_accuracy",
        "metadata_control_effective_changed_fraction",
        "metadata_control_condition_flip_fraction",
    }
    missing = intervention_required - set(interventions.columns)
    if missing:
        raise ValueError(
            f"Physical intervention table is missing gate columns: {sorted(missing)}."
        )
    intervention_rows = interventions.copy()
    intervention_rows["fold_index"] = intervention_rows["fold_index"].astype(int)
    scenario_matrix = intervention_rows.pivot(
        index="fold_index", columns="scenario", values="balanced_accuracy"
    )
    required_scenarios = {
        "clean",
        "block_coherent_metadata_shuffle",
        "joint_wet_dry_counterfactual",
    }
    if (
        set(scenario_matrix.index.astype(int)) != set(_DIAGNOSTIC_FOLDS)
        or not required_scenarios.issubset(set(scenario_matrix.columns.astype(str)))
    ):
        raise ValueError("Physical intervention table lacks the exact three diagnostic folds.")

    training_required = {"fold_index", "role", "effective_changed_fraction"}
    missing = training_required - set(training_controls.columns)
    if missing:
        raise ValueError(
            f"Physical training-control table is missing gate columns: {sorted(missing)}."
        )
    m3 = training_controls.loc[
        training_controls["role"].astype(str) == "M3_full_shuffle_train"
    ].copy()
    m3["fold_index"] = m3["fold_index"].astype(int)
    _require_exact_fold_rows(m3, name="M3 training-control table")

    subject_rows = subject_rows.sort_values("fold_index")
    scenario_matrix = scenario_matrix.sort_index()
    m3 = m3.sort_values("fold_index")
    shortcut_values = (
        pd.to_numeric(subject_rows["metadata_only_balanced_accuracy"]) - chance
    ).tolist()
    clean_a2_values = (
        pd.to_numeric(subject_rows["full_a2_balanced_accuracy"])
        - pd.to_numeric(subject_rows["a0_balanced_accuracy"])
    ).tolist()
    clean_a2_harm_values = [-value for value in clean_a2_values]
    training_pairing_values = (
        pd.to_numeric(subject_rows["full_a2_balanced_accuracy"])
        - pd.to_numeric(subject_rows["shuffle_train_balanced_accuracy"])
    ).tolist()
    inference_pairing_values = (
        pd.to_numeric(scenario_matrix["clean"])
        - pd.to_numeric(scenario_matrix["block_coherent_metadata_shuffle"])
    ).tolist()
    counterfactual_values = (
        pd.to_numeric(scenario_matrix["clean"])
        - pd.to_numeric(scenario_matrix["joint_wet_dry_counterfactual"])
    ).tolist()
    safety_values = (
        pd.to_numeric(subject_rows["a0_balanced_accuracy"]).reset_index(drop=True)
        - pd.to_numeric(
            scenario_matrix["joint_wet_dry_counterfactual"]
        ).reset_index(drop=True)
    ).tolist()
    training_potency_values = pd.to_numeric(m3["effective_changed_fraction"]).tolist()

    donor_rows = intervention_rows.loc[
        intervention_rows["scenario"].astype(str).isin(
            {"block_coherent_metadata_shuffle", "joint_wet_dry_counterfactual"}
        )
    ].copy()
    if len(donor_rows) != 6 or donor_rows.groupby("scenario")["fold_index"].nunique().ne(3).any():
        raise ValueError("Physical intervention donor potency is incomplete.")
    inference_potency_values = pd.to_numeric(
        donor_rows["metadata_control_effective_changed_fraction"]
    ).tolist()
    counterfactual_rows = intervention_rows.loc[
        intervention_rows["scenario"].astype(str) == "joint_wet_dry_counterfactual"
    ].sort_values("fold_index")
    condition_flip_values = pd.to_numeric(
        counterfactual_rows["metadata_control_condition_flip_fraction"]
    ).tolist()

    gates = {
        "clean_a2_directional_mean": _mean_lower_gate(
            clean_a2_values, margins["clean_a2_mean_minimum_delta"]
        ),
        "clean_a2_observed_subject_safety_audit": {
            **_upper_gate(
                clean_a2_harm_values, margins["clean_a2_subject_harm_margin"]
            ),
            "redundancy_note": (
                "At the frozen thresholds this is an explicit audit metric but is "
                "mathematically implied by the joint counterfactual-reliance and "
                "wrong-metadata-safety gates."
            ),
        },
        "shortcut_metadata_only": _upper_gate(
            shortcut_values, margins["shortcut_equivalence_margin"]
        ),
        "training_pairing_shuffle": _lower_gate(
            training_pairing_values, margins["pairing_mechanism_margin"]
        ),
        "inference_pairing_shuffle": _lower_gate(
            inference_pairing_values, margins["pairing_mechanism_margin"]
        ),
        "counterfactual_reliance": _lower_gate(
            counterfactual_values, margins["counterfactual_mechanism_margin"]
        ),
        "wrong_metadata_safety": _upper_gate(
            safety_values, margins["wrong_metadata_safety_harm_margin"]
        ),
        "training_shuffle_potency": _lower_gate(
            training_potency_values, margins["minimum_bundle_changed_fraction"]
        ),
        "inference_bundle_potency": _lower_gate(
            inference_potency_values, margins["minimum_bundle_changed_fraction"]
        ),
        "counterfactual_condition_flip_potency": _lower_gate(
            condition_flip_values, margins["minimum_condition_flip_fraction"]
        ),
    }
    potency_gate_names = {
        "training_shuffle_potency",
        "inference_bundle_potency",
        "counterfactual_condition_flip_potency",
    }
    failed_potency = sorted(
        name for name in potency_gate_names if not bool(gates[name]["passed"])
    )
    assay_valid = not failed_potency
    substantive_gate_names = [name for name in gates if name not in potency_gate_names]
    failed_substantive = sorted(
        name for name in substantive_gate_names if not bool(gates[name]["passed"])
    )
    substantive_pass = assay_valid and not failed_substantive
    if not assay_valid:
        status = "invalid_assay_control_potency_failure"
    elif failed_substantive:
        status = "diagnostic_no_go_one_or_more_substantive_gates_failed"
    else:
        status = "pass_all_predeclared_diagnostic_gates"
    return {
        "schema": "cfeg.physical-development-gates.v2",
        "status": status,
        "assay_valid": assay_valid,
        "failed_potency_gates": failed_potency,
        "substantive_interpretation_allowed": assay_valid,
        "substantive_diagnostic_gates_pass": substantive_pass,
        "failed_substantive_gates": failed_substantive,
        "all_predeclared_diagnostic_gates_pass": substantive_pass,
        "owner_review_required_regardless": True,
        "confirmatory_execution_authorized": False,
        "population_inference_allowed": False,
        "n_development_subjects": 3,
        "decision_rule": (
            "clean_direction_uses_observed_three_fold_mean; all_other_effect_and_safety_"
            "gates_use_least_favourable_observed_subject_or_fold"
        ),
        "numeric_comparison_tolerance": _NUMERIC_COMPARISON_TOLERANCE,
        "gates": gates,
        "invalid_assays": {
            "missingness_only": "invalid_single_natural_missingness_pattern",
            "physical_mechanism_bundle": (
                "valid" if assay_valid else "invalid_control_potency_failure"
            ),
        },
    }


def _require_exact_fold_rows(frame: pd.DataFrame, *, name: str) -> None:
    folds = frame["fold_index"].astype(int)
    if len(frame) != 3 or folds.duplicated().any() or set(folds) != set(_DIAGNOSTIC_FOLDS):
        raise ValueError(f"{name} must contain one row for each fold 0, 1, and 2.")


def _lower_gate(values: Sequence[float], threshold: float) -> dict[str, object]:
    normalized = _finite_values(values)
    observed = min(normalized)
    return {
        "direction": "all_observed_values_greater_than_or_equal",
        "threshold": float(threshold),
        "least_favourable_value": observed,
        "observed_values": normalized,
        "comparison_tolerance": _NUMERIC_COMPARISON_TOLERANCE,
        "passed": observed + _NUMERIC_COMPARISON_TOLERANCE >= float(threshold),
    }


def _upper_gate(values: Sequence[float], threshold: float) -> dict[str, object]:
    normalized = _finite_values(values)
    observed = max(normalized)
    return {
        "direction": "all_observed_values_less_than_or_equal",
        "threshold": float(threshold),
        "least_favourable_value": observed,
        "observed_values": normalized,
        "comparison_tolerance": _NUMERIC_COMPARISON_TOLERANCE,
        "passed": observed <= float(threshold) + _NUMERIC_COMPARISON_TOLERANCE,
    }


def _mean_lower_gate(values: Sequence[float], threshold: float) -> dict[str, object]:
    normalized = _finite_values(values)
    observed = float(sum(normalized) / len(normalized))
    return {
        "direction": "observed_mean_greater_than_or_equal",
        "threshold": float(threshold),
        "observed_mean": observed,
        "observed_values": normalized,
        "comparison_tolerance": _NUMERIC_COMPARISON_TOLERANCE,
        "passed": observed + _NUMERIC_COMPARISON_TOLERANCE >= float(threshold),
    }


def _finite_values(values: Sequence[float]) -> list[float]:
    normalized = [float(value) for value in values]
    if not normalized or any(not math.isfinite(value) for value in normalized):
        raise ValueError("Physical diagnostic gates require finite observed values.")
    return normalized


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
