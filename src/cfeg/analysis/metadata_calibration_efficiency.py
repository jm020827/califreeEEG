from __future__ import annotations

import hashlib
import itertools
import math
import re
from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd
from scipy.stats import t as student_t

from cfeg.metadata_calibration_execution import (
    ExperimentPhase,
    cell_execution_contract_sha256,
    execution_cells_for_phase,
)

_SHA256 = re.compile(r"[0-9a-f]{64}")
_BINDING_COLUMNS = (
    "plan_sha256",
    "execution_manifest_sha256",
    "decision_receipt_sha256",
    "asset_receipt_sha256",
    "asset_fingerprint_bundle_sha256",
    "class_map_sha256",
    "source_tree_sha256",
)
_FORBIDDEN_STAGING_OUTCOME_COLUMNS = {
    "sample_id",
    "label",
    "target_label",
    "prediction",
    "correct",
    "accuracy",
    "balanced_accuracy",
    "loss",
}


@dataclass(frozen=True)
class MetadataCalibrationBundleBindings:
    plan_sha256: str
    execution_manifest_sha256: str
    decision_receipt_sha256: str
    asset_receipt_sha256: str
    asset_fingerprint_bundle_sha256: str
    class_map_sha256: str
    source_tree_sha256: str

    def as_dict(self) -> dict[str, str]:
        values = {column: str(getattr(self, column)) for column in _BINDING_COLUMNS}
        if not all(_is_sha256(value) for value in values.values()):
            raise ValueError("Every frozen bundle binding must be one lowercase SHA-256.")
        return values


@dataclass(frozen=True)
class MetadataCalibrationBundleSpec:
    candidate_id: str = "metadata-calibration-efficiency-v1"
    n_classes: int = 12
    budgets: tuple[int, ...] = (0, 1, 3, 5)
    seeds: tuple[int, ...] = (42, 43, 44)
    conditions: tuple[str, ...] = ("dry", "wet")
    phase: Literal["source_development", "held_participant_evaluation"] = (
        "held_participant_evaluation"
    )
    checkpoint_groups: tuple[str, ...] = ("held",)
    contexts: tuple[tuple[str, str], ...] = (
        ("A_0", "exact_off"),
        ("A_M", "correct"),
        ("A_Q", "exact_off"),
        ("A_QM", "correct"),
        ("A_QM", "all_missing"),
        ("A_QM", "block_shuffle"),
        ("A_QM", "stale"),
        ("A_QM", "opposite_interface"),
    )
    primary_alpha: float = 0.05
    mechanism_alpha: float = 0.05
    noninferiority_alpha: float = 0.025
    promotion_sesoi: float = 0.020
    calibration_value_sesoi: float = 0.020
    mechanism_sesoi: float = 0.010
    noninferiority_margin: float = 1.0 / 60.0
    operational_ba_floor: float = 0.50
    severe_harm_cutoff: float = -0.05
    held_severe_harm_participants_max: int = 6
    catastrophic_harm_floor: float = -0.10
    sensitivity_seed: int = 20260904
    sensitivity_resamples: int = 20_000

    @property
    def probability_columns(self) -> tuple[str, ...]:
        return tuple(f"prob_{label:03d}" for label in range(self.n_classes))


DEFAULT_METADATA_CALIBRATION_BUNDLE_SPEC = MetadataCalibrationBundleSpec()


def metadata_calibration_bundle_spec_for_phase(
    *, phase: ExperimentPhase, sensitivity_resamples: int = 20_000
) -> MetadataCalibrationBundleSpec:
    """Build the production grid from the frozen phase contract, never caller cells."""

    checkpoint_groups = (
        ("fold0", "fold1", "fold2")
        if phase == "source_development"
        else ("held",)
    )
    return MetadataCalibrationBundleSpec(
        phase=phase,
        checkpoint_groups=checkpoint_groups,
        contexts=execution_cells_for_phase(phase=phase),
        sensitivity_resamples=sensitivity_resamples,
    )


def validate_metadata_calibration_private_staging(
    predictions: pd.DataFrame,
    *,
    expected_query_unlabeled: pd.DataFrame,
    expected_support: pd.DataFrame,
    bindings: MetadataCalibrationBundleBindings,
    spec: MetadataCalibrationBundleSpec = DEFAULT_METADATA_CALIBRATION_BUNDLE_SPEC,
) -> dict[str, object]:
    """Validate a complete probability grid without loading query outcomes."""

    expected = _validate_expected_query(
        expected_query_unlabeled,
        spec=spec,
        require_label=False,
    )
    support = _validate_expected_support(expected_support, expected_query=expected, spec=spec)
    frame = _validate_prediction_grid(
        predictions,
        expected_query=expected,
        expected_support=support,
        bindings=bindings,
        spec=spec,
    )
    missing_exact = _validate_exact_missing_fallback(frame, spec=spec)
    return {
        "schema": "cfeg.metadata-calibration-private-staging-validation.v1",
        "candidate_id": spec.candidate_id,
        "phase": spec.phase,
        "rows": len(frame),
        "query_outcomes_loaded": False,
        "complete_atomic_grid": True,
        "missing_fallback_probability_exact": missing_exact,
        "staging_content_sha256": _staging_content_sha256(frame, spec=spec),
    }


def reduce_metadata_calibration_predictions(
    predictions: pd.DataFrame,
    *,
    expected_query: pd.DataFrame,
    expected_support: pd.DataFrame,
    bindings: MetadataCalibrationBundleBindings,
    spec: MetadataCalibrationBundleSpec = DEFAULT_METADATA_CALIBRATION_BUNDLE_SPEC,
) -> dict[str, object]:
    """Validate and reduce one complete atomic probability bundle.

    Probabilities are averaged across source-training seeds for each query
    before argmax.  Conditions, budgets, blocks, classes, and seeds are never
    treated as independent inferential units.
    """

    expected = _validate_expected_query(expected_query, spec=spec, require_label=True)
    expected_unlabeled = expected.drop(columns="label")
    support = _validate_expected_support(
        expected_support,
        expected_query=expected_unlabeled,
        spec=spec,
    )
    frame = _validate_prediction_grid(
        predictions,
        expected_query=expected_unlabeled,
        expected_support=support,
        bindings=bindings,
        spec=spec,
    )
    frame = frame.merge(
        expected.loc[:, ["query_token", "label"]],
        on="query_token",
        how="left",
        validate="many_to_one",
    )
    missing_exact = _validate_exact_missing_fallback(frame, spec=spec)
    probability_columns = list(spec.probability_columns)
    ensemble_key = [
        "role",
        "context",
        "budget",
        "query_token",
        "subject_id",
        "electrode_type",
        "label",
    ]
    ensemble = (
        frame.groupby(ensemble_key, as_index=False, sort=False)[probability_columns]
        .mean()
        .sort_values(ensemble_key, kind="mergesort")
        .reset_index(drop=True)
    )
    ensemble["prediction"] = np.argmax(
        ensemble.loc[:, probability_columns].to_numpy(dtype=np.float64), axis=1
    )
    ensemble["correct"] = ensemble["prediction"].eq(ensemble["label"]).astype(float)

    class_accuracy = (
        ensemble.groupby(
            ["role", "context", "budget", "subject_id", "electrode_type", "label"],
            as_index=False,
            sort=False,
        )["correct"]
        .mean()
        .rename(columns={"correct": "class_accuracy"})
    )
    condition_ba = (
        class_accuracy.groupby(
            ["role", "context", "budget", "subject_id", "electrode_type"],
            as_index=False,
            sort=False,
        )["class_accuracy"]
        .mean()
        .rename(columns={"class_accuracy": "balanced_accuracy"})
    )
    observed_conditions = condition_ba.groupby(
        ["role", "context", "budget", "subject_id"], sort=False
    )["electrode_type"].nunique()
    if not observed_conditions.eq(len(spec.conditions)).all():
        raise ValueError("Every participant outcome must contain every electrode condition.")
    participant_ba = (
        condition_ba.groupby(
            ["role", "context", "budget", "subject_id"],
            as_index=False,
            sort=False,
        )["balanced_accuracy"]
        .mean()
        .rename(columns={"balanced_accuracy": "participant_balanced_accuracy"})
    )
    participant_contrasts = derive_participant_contrasts(participant_ba, spec=spec)
    return {
        "schema": "cfeg.metadata-calibration-analysis-bundle.v1",
        "candidate_id": spec.candidate_id,
        "phase": spec.phase,
        "bindings": bindings.as_dict(),
        "seed_reduction": "mean_probabilities_per_sample_before_argmax",
        "tie_break": "lowest_canonical_class_id",
        "missing_fallback_probability_exact": missing_exact,
        "ensemble_predictions": ensemble,
        "class_accuracy": class_accuracy,
        "condition_balanced_accuracy": condition_ba,
        "participant_balanced_accuracy": participant_ba,
        "participant_contrasts": participant_contrasts,
    }


def evaluate_development_gate(
    participant_contrasts: pd.DataFrame,
    *,
    outer_fold_by_subject: dict[str, int],
    missing_fallback_probability_exact: bool,
    metadata_only_structural_chance_exact: bool,
    expected_n: int = 39,
    spec: MetadataCalibrationBundleSpec = DEFAULT_METADATA_CALIBRATION_BUNDLE_SPEC,
) -> dict[str, object]:
    """Apply the frozen 39-participant no-retry promotion screen."""

    frame = _validate_contrast_table(participant_contrasts, expected_n=expected_n)
    subjects = set(frame["subject_id"].astype(str))
    if set(outer_fold_by_subject) != subjects:
        raise ValueError("Outer-fold mapping must contain every development subject exactly once.")
    folds = frame["subject_id"].astype(str).map(outer_fold_by_subject)
    if folds.isna().any() or sorted(folds.unique().tolist()) != [0, 1, 2]:
        raise ValueError("Development gate requires exactly three outer folds numbered 0,1,2.")
    expected_fold_size = expected_n // 3
    if expected_n % 3 or not folds.value_counts().sort_index().eq(expected_fold_size).all():
        raise ValueError("Development gate requires three equally sized outer folds.")

    eauc = frame["eauc_increment"].to_numpy(dtype=float)
    shuffle = frame["correct_minus_shuffle_eauc"].to_numpy(dtype=float)
    stale = frame["correct_minus_stale_eauc"].to_numpy(dtype=float)
    wrong = frame["wrong_minus_aq_eauc"].to_numpy(dtype=float)
    fold_frame = pd.DataFrame(
        {"fold": folds.to_numpy(), "eauc": eauc, "shuffle": shuffle, "stale": stale}
    )
    fold_means = {
        name: {
            str(int(fold)): float(value)
            for fold, value in fold_frame.groupby("fold")[column].mean().items()
        }
        for name, column in (
            ("eauc_increment", "eauc"),
            ("correct_minus_shuffle_eauc", "shuffle"),
            ("correct_minus_stale_eauc", "stale"),
        )
    }
    observed_statistics = {
        "aq_k5_mean": float(frame["aq_k5"].mean()),
        "eauc_increment_mean": float(eauc.mean()),
        "eauc_increment_median": float(np.median(eauc)),
        "eauc_increment_positive_participants": int(np.count_nonzero(eauc > 0.0)),
        "correct_minus_shuffle_eauc_mean": float(shuffle.mean()),
        "correct_minus_shuffle_positive_participants": int(
            np.count_nonzero(shuffle > 0.0)
        ),
        "correct_minus_stale_eauc_mean": float(stale.mean()),
        "correct_minus_stale_positive_participants": int(np.count_nonzero(stale > 0.0)),
        "wrong_minus_aq_eauc_mean": float(wrong.mean()),
        "wrong_context_severe_harm_participants": int(np.count_nonzero(wrong < -0.05)),
        "wrong_context_minimum": float(wrong.min()),
        "correct_context_severe_harm_participants": int(np.count_nonzero(eauc < -0.05)),
        "correct_context_minimum": float(eauc.min()),
        "outer_fold_means": fold_means,
    }
    checks = {
        "baseline_viability": observed_statistics["aq_k5_mean"] >= 0.50,
        "early_budget_mean": observed_statistics["eauc_increment_mean"] >= 0.010,
        "early_budget_median": observed_statistics["eauc_increment_median"] > 0.0,
        "early_budget_positive_participants": (
            observed_statistics["eauc_increment_positive_participants"] >= 24
        ),
        "early_budget_every_fold_positive": bool(
            fold_frame.groupby("fold")["eauc"].mean().gt(0.0).all()
        ),
        "correct_minus_shuffle_mean": (
            observed_statistics["correct_minus_shuffle_eauc_mean"] >= 0.010
        ),
        "correct_minus_shuffle_positive_participants": (
            observed_statistics["correct_minus_shuffle_positive_participants"] >= 24
        ),
        "correct_minus_shuffle_no_negative_fold": bool(
            fold_frame.groupby("fold")["shuffle"].mean().ge(0.0).all()
        ),
        "correct_minus_stale_mean": (
            observed_statistics["correct_minus_stale_eauc_mean"] >= 0.010
        ),
        "correct_minus_stale_positive_participants": (
            observed_statistics["correct_minus_stale_positive_participants"] >= 24
        ),
        "correct_minus_stale_no_negative_fold": bool(
            fold_frame.groupby("fold")["stale"].mean().ge(0.0).all()
        ),
        "missing_exact_fallback": bool(missing_fallback_probability_exact),
        "wrong_context_mean_safety": observed_statistics["wrong_minus_aq_eauc_mean"]
        >= -(1.0 / 60.0),
        "wrong_context_severe_harm_count": (
            observed_statistics["wrong_context_severe_harm_participants"] <= 4
        ),
        "wrong_context_no_catastrophic_harm": (
            observed_statistics["wrong_context_minimum"] >= -0.10
        ),
        "correct_context_severe_harm_count": (
            observed_statistics["correct_context_severe_harm_participants"] <= 4
        ),
        "correct_context_no_catastrophic_harm": (
            observed_statistics["correct_context_minimum"] >= -0.10
        ),
        "metadata_only_shortcut": bool(metadata_only_structural_chance_exact),
    }
    hard_gate_names = (
        "baseline_viability",
        "early_budget_mean",
        "early_budget_every_fold_positive",
        "correct_minus_shuffle_mean",
        "correct_minus_shuffle_no_negative_fold",
        "missing_exact_fallback",
        "metadata_only_shortcut",
    )
    hard_gates = {name: checks[name] for name in hard_gate_names}
    required_reports = {
        name: value for name, value in checks.items() if name not in hard_gates
    }
    passed = bool(all(hard_gates.values()))
    deployment_qualified = bool(
        checks["wrong_context_mean_safety"]
        and checks["wrong_context_severe_harm_count"]
        and checks["wrong_context_no_catastrophic_harm"]
        and checks["correct_context_severe_harm_count"]
        and checks["correct_context_no_catastrophic_harm"]
    )
    return {
        "schema": "cfeg.metadata-calibration-development-gate.v1",
        "expected_n": expected_n,
        "hard_promotion_gates": hard_gates,
        "required_diagnostic_or_deployment_reports": required_reports,
        "observed_statistics": observed_statistics,
        "passed": passed,
        "decision": "development_go" if passed else "development_no_go",
        "held_evaluation_access_allowed": passed,
        "deployment_qualified_for_robustness_claim": deployment_qualified,
        "metadata_only_screen": {
            "method": "deterministic_information_structure_proof",
            "block_constant_M": bool(metadata_only_structural_chance_exact),
            "one_trial_per_class_per_block": True,
            "implied_balanced_accuracy": 1.0 / spec.n_classes,
            "passed": bool(metadata_only_structural_chance_exact),
        },
        "no_retry_rule": "threshold_or_method_changes_require_a_new_candidate_and_new_data",
    }


def evaluate_held_claim_sequence(
    participant_contrasts: pd.DataFrame,
    *,
    expected_n: int = 60,
    spec: MetadataCalibrationBundleSpec = DEFAULT_METADATA_CALIBRATION_BUNDLE_SPEC,
) -> dict[str, object]:
    """Evaluate the two-stage confirmatory core and separate mechanism/safety families."""

    frame = _validate_contrast_table(participant_contrasts, expected_n=expected_n)
    primary = one_sided_paired_t(frame["eauc_increment"], alpha=spec.primary_alpha, null_margin=0.0)
    primary.update(
        {
            "name": "primary_eAUC_superiority",
            "stage": "H1",
            "practical_threshold": spec.promotion_sesoi,
            "practical_threshold_met": bool(
                primary["mean_difference"] >= spec.promotion_sesoi
            ),
        }
    )
    primary["claim_passed"] = bool(
        primary["reject_null"] and primary["practical_threshold_met"]
    )

    calibration_value = one_sided_paired_t(
        frame["calibration_value_k1_to_k3"],
        alpha=spec.primary_alpha,
        null_margin=0.0,
    )
    calibration_value.update(
        {
            "name": "calibration_value_A_Q_k3_over_k1",
            "stage": "H2a",
            "tested_confirmatorily": bool(primary["claim_passed"]),
            "practical_threshold": spec.calibration_value_sesoi,
            "practical_threshold_met": bool(
                calibration_value["mean_difference"] >= spec.calibration_value_sesoi
            ),
        }
    )
    calibration_value["component_passed"] = bool(
        calibration_value["tested_confirmatorily"]
        and calibration_value["reject_null"]
        and calibration_value["practical_threshold_met"]
    )

    savings = one_sided_paired_t(
        frame["save_k1_vs_k3"],
        alpha=spec.noninferiority_alpha,
        null_margin=-spec.noninferiority_margin,
    )
    savings_floor_met = bool(
        frame["aqm_k1"].mean() >= spec.operational_ba_floor
        and frame["aq_k3"].mean() >= spec.operational_ba_floor
    )
    savings.update(
        {
            "name": "A_QM_k1_noninferior_to_A_Q_k3",
            "stage": "H2b",
            "tested_confirmatorily": bool(primary["claim_passed"]),
            "utility_floor": spec.operational_ba_floor,
            "utility_floor_met": savings_floor_met,
        }
    )
    savings["component_passed"] = bool(
        savings["tested_confirmatorily"]
        and savings["reject_null"]
        and savings["utility_floor_met"]
    )
    h2_passed = bool(
        primary["claim_passed"]
        and calibration_value["component_passed"]
        and savings["component_passed"]
    )
    calibration_value["intersection_claim_passed"] = h2_passed
    savings["intersection_claim_passed"] = h2_passed

    mechanism_tests: list[dict[str, object]] = []
    for name, column, role in (
        (
            "correct_context_over_block_shuffle",
            "correct_minus_shuffle_eauc",
            "key_mechanism",
        ),
        ("correct_context_over_stale", "correct_minus_stale_eauc", "supportive_sensitivity"),
    ):
        result = one_sided_paired_t(
            frame[column], alpha=spec.mechanism_alpha, null_margin=0.0
        )
        result.update(
            {
                "name": name,
                "role": role,
                "formal_family_open": h2_passed,
                "practical_threshold": spec.mechanism_sesoi,
                "practical_threshold_met": bool(
                    result["mean_difference"] >= spec.mechanism_sesoi
                ),
            }
        )
        mechanism_tests.append(result)
    holm = _holm_step_down(
        {str(result["name"]): float(result["p_value"]) for result in mechanism_tests},
        alpha=spec.mechanism_alpha,
    )
    for result in mechanism_tests:
        adjusted = holm[str(result["name"])]
        result.update(adjusted)
        result["claim_passed"] = bool(
            result["formal_family_open"]
            and result["holm_reject"]
            and result["practical_threshold_met"]
        )

    wrong_safety = one_sided_paired_t(
        frame["wrong_minus_aq_eauc"],
        alpha=spec.noninferiority_alpha,
        null_margin=-spec.noninferiority_margin,
    )
    wrong_values = frame["wrong_minus_aq_eauc"].to_numpy(dtype=float)
    wrong_tail = {
        "severe_harm_cutoff": spec.severe_harm_cutoff,
        "severe_harm_count": int(np.count_nonzero(wrong_values < spec.severe_harm_cutoff)),
        "severe_harm_count_max": spec.held_severe_harm_participants_max,
        "catastrophic_harm_floor": spec.catastrophic_harm_floor,
        "minimum_difference": float(wrong_values.min()),
    }
    wrong_tail["passed"] = bool(
        wrong_tail["severe_harm_count"] <= spec.held_severe_harm_participants_max
        and wrong_tail["minimum_difference"] >= spec.catastrophic_harm_floor
    )
    wrong_safety.update(
        {
            "name": "wrong_context_noninferiority",
            "role": "deployment_qualification_not_confirmatory_core",
            "tail_safety": wrong_tail,
            "mean_noninferiority_passed": bool(wrong_safety["reject_null"]),
            "deployment_qualified": bool(
                wrong_safety["reject_null"] and wrong_tail["passed"]
            ),
        }
    )

    descriptive_savings: list[dict[str, object]] = []
    for name, column, lower_endpoint, higher_endpoint in (
        ("A_QM_k0_vs_A_Q_k1", "save_k0_vs_k1", "aqm_k0", "aq_k1"),
        ("A_QM_k3_vs_A_Q_k5", "save_k3_vs_k5", "aqm_k3", "aq_k5"),
    ):
        result = one_sided_paired_t(
            frame[column],
            alpha=spec.noninferiority_alpha,
            null_margin=-spec.noninferiority_margin,
        )
        utility_floor_met = bool(
            frame[lower_endpoint].mean() >= spec.operational_ba_floor
            and frame[higher_endpoint].mean() >= spec.operational_ba_floor
        )
        result.update(
            {
                "name": name,
                "role": "descriptive_additional_savings",
                "utility_floor": spec.operational_ba_floor,
                "utility_floor_met": utility_floor_met,
                "nominal_noninferiority": bool(
                    result["reject_null"] and utility_floor_met
                ),
            }
        )
        descriptive_savings.append(result)

    sensitivity = _sensitivity_summary(frame["eauc_increment"].to_numpy(dtype=float), spec=spec)
    correct_values = frame["eauc_increment"].to_numpy(dtype=float)
    same_budget_identity = frame["save_k1_vs_k3"] + frame["calibration_value_k1_to_k3"]
    observed_same_budget = frame["aqm_k1"] - frame["aq_k1"]
    if not np.allclose(same_budget_identity, observed_same_budget, rtol=0.0, atol=1e-12):
        raise ValueError("The k1 savings/calibration-value identity is inconsistent.")
    return {
        "schema": "cfeg.metadata-calibration-held-claims.v1",
        "confirmatory_hierarchy": {
            "H1": "primary_eAUC_superiority",
            "H2_intersection": [
                "calibration_value_A_Q_k3_over_k1",
                "A_QM_k1_noninferior_to_A_Q_k3",
            ],
            "multiplicity": "serial_gatekeeping_then_intersection_union_no_alpha_split",
        },
        "confirmatory_tests": [primary, calibration_value, savings],
        "primary_claim_passed": bool(primary["claim_passed"]),
        "calibration_saving_claim_passed": h2_passed,
        "confirmed_saving": (
            "two_complete_blocks_per_interface_schedule_in_equal_wet_dry_estimand"
            if h2_passed
            else None
        ),
        "confirmed_labeled_trial_reduction_per_interface": 24 if h2_passed else 0,
        "mechanism_family": {
            "multiplicity": "holm_alpha_0p05_after_confirmatory_core",
            "tests": mechanism_tests,
        },
        "deployment_safety": wrong_safety,
        "descriptive_additional_savings": descriptive_savings,
        "k1_consistency": {
            "identity": "A_QM_k1_minus_A_Q_k1_equals_H2b_plus_H2a",
            "mean_same_budget_increment": float(observed_same_budget.mean()),
            "identity_exact": True,
        },
        "correct_context_harm_report": {
            "severe_harm_cutoff": spec.severe_harm_cutoff,
            "severe_harm_count": int(
                np.count_nonzero(correct_values < spec.severe_harm_cutoff)
            ),
            "catastrophic_harm_floor": spec.catastrophic_harm_floor,
            "minimum_difference": float(correct_values.min()),
        },
        "sensitivity": sensitivity,
    }


def one_sided_paired_t(
    differences: pd.Series | np.ndarray,
    *,
    alpha: float,
    null_margin: float,
) -> dict[str, object]:
    values = np.asarray(differences, dtype=float)
    if values.ndim != 1 or len(values) < 2 or not np.isfinite(values).all():
        raise ValueError("Paired differences must be one finite vector with at least two values.")
    if not 0.0 < alpha < 1.0 or not np.isfinite(null_margin):
        raise ValueError("alpha and null_margin are invalid.")
    mean = float(values.mean())
    sd = float(values.std(ddof=1))
    se = sd / math.sqrt(len(values))
    if se <= np.finfo(float).eps:
        relation = (
            "above_null"
            if mean > null_margin
            else ("below_null" if mean < null_margin else "equal_to_null")
        )
        statistic = None
        p_value = 0.0 if mean > null_margin else 1.0
        lower = mean
    else:
        relation = "not_degenerate"
        statistic = (mean - null_margin) / se
        p_value = float(student_t.sf(statistic, df=len(values) - 1))
        lower = float(mean - student_t.ppf(1.0 - alpha, df=len(values) - 1) * se)
    reject = bool(p_value < alpha and lower > null_margin)
    return {
        "test": "one_sided_paired_student_t",
        "n_participants": len(values),
        "mean_difference": mean,
        "sd_difference": sd,
        "standard_error": se,
        "null_margin": float(null_margin),
        "alpha": float(alpha),
        "statistic": None if statistic is None else float(statistic),
        "degenerate_standard_error": statistic is None,
        "degenerate_mean_relation_to_null": relation,
        "p_value": p_value,
        "confidence_interval_low": lower,
        "confidence_level_one_sided": float(1.0 - alpha),
        "reject_null": reject,
    }


def one_sided_upper_t(
    values: pd.Series | np.ndarray,
    *,
    alpha: float,
    expected_n: int | None = None,
) -> dict[str, object]:
    """Return a participant-level one-sided Student-t upper confidence bound."""

    array = np.asarray(values, dtype=float)
    if array.ndim != 1 or len(array) < 2 or not np.isfinite(array).all():
        raise ValueError("Upper-bound input must be one finite participant vector.")
    if expected_n is not None and len(array) != expected_n:
        raise ValueError(f"Upper-bound input must contain exactly {expected_n} participants.")
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must lie strictly between zero and one.")
    mean = float(array.mean())
    sd = float(array.std(ddof=1))
    se = sd / math.sqrt(len(array))
    high = mean if se <= np.finfo(float).eps else float(
        mean + student_t.ppf(1.0 - alpha, df=len(array) - 1) * se
    )
    return {
        "test": "one_sided_participant_student_t_upper_bound",
        "n_participants": len(array),
        "mean": mean,
        "sd": sd,
        "standard_error": se,
        "alpha": float(alpha),
        "confidence_interval_high": high,
        "confidence_level_one_sided": float(1.0 - alpha),
    }


def derive_participant_contrasts(
    participant_ba: pd.DataFrame,
    *,
    spec: MetadataCalibrationBundleSpec,
) -> pd.DataFrame:
    """Derive the frozen participant contrasts from participant-level BA cells."""
    wide = participant_ba.pivot(
        index="subject_id",
        columns=["role", "context", "budget"],
        values="participant_balanced_accuracy",
    )
    required = {
        (role, context, budget) for role, context in spec.contexts for budget in spec.budgets
    }
    missing = sorted(required - set(wide.columns))
    if missing:
        raise ValueError(f"Participant reduction lost required cells: {missing}.")

    result = pd.DataFrame({"subject_id": wide.index.astype(str)})
    for budget in spec.budgets:
        result[f"a0_k{budget}"] = wide[("A_0", "exact_off", budget)].to_numpy()
        result[f"am_k{budget}"] = wide[("A_M", "correct", budget)].to_numpy()
        result[f"aq_k{budget}"] = wide[("A_Q", "exact_off", budget)].to_numpy()
        result[f"aqm_k{budget}"] = wide[("A_QM", "correct", budget)].to_numpy()
    for context, prefix in (
        ("block_shuffle", "shuffle"),
        ("stale", "stale"),
        ("opposite_interface", "wrong"),
    ):
        for budget in spec.budgets:
            result[f"{prefix}_k{budget}"] = wide[("A_QM", context, budget)].to_numpy()

    weights = {0: 1.0 / 6.0, 1: 1.0 / 2.0, 3: 1.0 / 3.0}
    for prefix in ("a0", "am", "aq", "aqm", "shuffle", "stale", "wrong"):
        result[f"{prefix}_eauc"] = sum(
            weight * result[f"{prefix}_k{budget}"] for budget, weight in weights.items()
        )
    result["metadata_without_q_eauc_increment"] = result["am_eauc"] - result["a0_eauc"]
    result["q_without_metadata_eauc_increment"] = result["aq_eauc"] - result["a0_eauc"]
    result["eauc_increment"] = result["aqm_eauc"] - result["aq_eauc"]
    result["correct_minus_shuffle_eauc"] = result["aqm_eauc"] - result["shuffle_eauc"]
    result["correct_minus_stale_eauc"] = result["aqm_eauc"] - result["stale_eauc"]
    result["wrong_minus_aq_eauc"] = result["wrong_eauc"] - result["aq_eauc"]
    result["save_k0_vs_k1"] = result["aqm_k0"] - result["aq_k1"]
    result["save_k1_vs_k3"] = result["aqm_k1"] - result["aq_k3"]
    result["save_k3_vs_k5"] = result["aqm_k3"] - result["aq_k5"]
    result["calibration_value_k1_to_k3"] = result["aq_k3"] - result["aq_k1"]
    return result


def _validate_expected_query(
    expected_query: pd.DataFrame,
    *,
    spec: MetadataCalibrationBundleSpec,
    require_label: bool,
) -> pd.DataFrame:
    required = {
        "query_token",
        "query_identity_sha256",
        "subject_id",
        "electrode_type",
        "checkpoint_group",
    }
    if require_label:
        required.add("label")
    missing = required - set(expected_query.columns)
    if missing:
        raise ValueError(f"Expected-query table is missing columns: {sorted(missing)}.")
    extra = set(expected_query.columns) - required
    if extra:
        raise ValueError(f"Expected-query table has undeclared columns: {sorted(extra)}.")
    frame = expected_query.loc[:, sorted(required)].copy()
    frame["query_token"] = frame["query_token"].astype(str)
    frame["subject_id"] = frame["subject_id"].astype(str)
    frame["electrode_type"] = frame["electrode_type"].astype(str)
    frame["checkpoint_group"] = frame["checkpoint_group"].astype(str)
    frame["query_identity_sha256"] = frame["query_identity_sha256"].astype(str)
    if require_label:
        frame["label"] = _strict_integer(frame["label"], name="expected label")
    if frame["query_token"].duplicated().any():
        raise ValueError("Expected opaque query tokens must be unique.")
    if not frame["query_token"].str.fullmatch(r"q_[0-9a-f]{64}").all():
        raise ValueError("Expected query tokens must use the opaque q_<HMAC-SHA256> schema.")
    if not frame["query_identity_sha256"].map(_is_sha256).all():
        raise ValueError("Expected query identity bindings must be lowercase SHA-256 values.")
    if set(frame["electrode_type"]) != set(spec.conditions):
        raise ValueError("Expected query conditions differ from the frozen dry/wet contract.")
    if set(frame["checkpoint_group"]) != set(spec.checkpoint_groups):
        raise ValueError("Expected query checkpoint groups differ from the frozen phase contract.")
    participant_group_counts = frame.groupby("subject_id")["checkpoint_group"].nunique()
    if not participant_group_counts.eq(1).all():
        raise ValueError("Each participant must belong to exactly one checkpoint group.")
    if require_label and (frame["label"].min() < 0 or frame["label"].max() >= spec.n_classes):
        raise ValueError("Expected query label falls outside the class vocabulary.")
    identity_counts = frame.groupby(["subject_id", "electrode_type"], sort=False)[
        "query_identity_sha256"
    ].nunique()
    if not identity_counts.eq(1).all():
        raise ValueError("Each participant-interface must have one sealed query identity digest.")
    return frame.sort_values("query_token", kind="mergesort").reset_index(drop=True)


def _validate_expected_support(
    expected_support: pd.DataFrame,
    *,
    expected_query: pd.DataFrame,
    spec: MetadataCalibrationBundleSpec,
) -> pd.DataFrame:
    required = {
        "subject_id",
        "electrode_type",
        "checkpoint_group",
        "budget",
        "support_identity_sha256",
    }
    missing = required - set(expected_support.columns)
    if missing:
        raise ValueError(f"Expected-support table is missing columns: {sorted(missing)}.")
    frame = expected_support.loc[:, sorted(required)].copy()
    for column in ("subject_id", "electrode_type", "checkpoint_group"):
        frame[column] = frame[column].astype(str)
    frame["budget"] = _strict_integer(frame["budget"], name="expected support budget")
    if not frame["support_identity_sha256"].astype(str).map(_is_sha256).all():
        raise ValueError("Expected support identities must be lowercase SHA-256 values.")
    key = ["subject_id", "electrode_type", "checkpoint_group", "budget"]
    if frame.duplicated(key).any():
        raise ValueError("Expected-support table contains duplicate participant cells.")
    query_cells = expected_query.loc[
        :, ["subject_id", "electrode_type", "checkpoint_group"]
    ].drop_duplicates()
    expected_cells = query_cells.merge(
        pd.DataFrame({"budget": list(spec.budgets)}), how="cross"
    )
    observed_cells = frame.loc[:, key]
    merged = expected_cells.merge(observed_cells, on=key, how="outer", indicator=True)
    if not merged["_merge"].eq("both").all():
        raise ValueError("Expected-support table is not the complete participant-budget grid.")
    return frame.sort_values(key, kind="mergesort").reset_index(drop=True)


def _validate_prediction_grid(
    predictions: pd.DataFrame,
    *,
    expected_query: pd.DataFrame,
    expected_support: pd.DataFrame,
    bindings: MetadataCalibrationBundleBindings,
    spec: MetadataCalibrationBundleSpec,
) -> pd.DataFrame:
    leaked = _FORBIDDEN_STAGING_OUTCOME_COLUMNS.intersection(predictions.columns)
    if leaked:
        raise ValueError(
            "Private prediction staging must be outcome-free; forbidden columns: "
            f"{sorted(leaked)}."
        )
    required = {
        "candidate_id",
        "phase",
        "checkpoint_group",
        "checkpoint_sha256",
        "role",
        "context",
        "seed",
        "budget",
        "query_token",
        "subject_id",
        "electrode_type",
        "support_identity_sha256",
        "query_identity_sha256",
        "intervention_mapping_sha256",
        "resolved_context_usage_sha256",
        "cell_execution_contract_sha256",
        *_BINDING_COLUMNS,
        *spec.probability_columns,
    }
    missing = required - set(predictions.columns)
    if missing:
        raise ValueError(f"Prediction table is missing columns: {sorted(missing)}.")
    extra = set(predictions.columns) - required
    if extra:
        raise ValueError(f"Prediction table has undeclared staging columns: {sorted(extra)}.")
    frame = predictions.loc[:, sorted(required)].copy()
    frame["seed"] = _strict_integer(frame["seed"], name="seed")
    frame["budget"] = _strict_integer(frame["budget"], name="budget")
    if set(frame["candidate_id"].astype(str)) != {spec.candidate_id}:
        raise ValueError("Prediction candidate binding is wrong.")
    if set(frame["phase"].astype(str)) != {spec.phase}:
        raise ValueError("Prediction phase differs from the frozen bundle phase.")
    if set(frame["checkpoint_group"].astype(str)) != set(spec.checkpoint_groups):
        raise ValueError("Prediction checkpoint groups differ from the frozen phase contract.")
    for column, expected_value in bindings.as_dict().items():
        if set(frame[column].astype(str)) != {expected_value}:
            raise ValueError(f"Prediction bundle does not match frozen {column}.")
    for column in (
        "checkpoint_sha256",
        "support_identity_sha256",
        "query_identity_sha256",
        "intervention_mapping_sha256",
        "resolved_context_usage_sha256",
        "cell_execution_contract_sha256",
    ):
        if not frame[column].astype(str).map(_is_sha256).all():
            raise ValueError(f"Prediction column {column} contains a non-SHA-256 value.")
    expected_dispatch_sha256 = cell_execution_contract_sha256(phase=spec.phase)
    if set(frame["cell_execution_contract_sha256"].astype(str)) != {
        expected_dispatch_sha256
    }:
        raise ValueError("Prediction grid differs from the frozen phase dispatch hash.")
    if set(frame["seed"]) != set(spec.seeds) or set(frame["budget"]) != set(spec.budgets):
        raise ValueError("Prediction seeds or budgets differ from the frozen contract.")
    if set(frame["electrode_type"].astype(str)) != set(spec.conditions):
        raise ValueError("Prediction conditions differ from the frozen contract.")
    if set(frame[["role", "context"]].itertuples(index=False, name=None)) != set(spec.contexts):
        raise ValueError("Prediction role/context cells differ from the frozen contract.")

    expected_groups = {
        (role, context, seed, budget)
        for (role, context), seed, budget in itertools.product(
            spec.contexts, spec.seeds, spec.budgets
        )
    }
    observed_groups = set(
        frame[["role", "context", "seed", "budget"]].itertuples(index=False, name=None)
    )
    if observed_groups != expected_groups or len(frame) != len(expected_groups) * len(
        expected_query
    ):
        raise ValueError("Prediction table is not the complete atomic grid.")
    key = ["role", "context", "seed", "budget", "query_token"]
    if frame.duplicated(key).any():
        raise ValueError("Prediction table contains duplicate atomic rows.")

    merged = frame.merge(
        expected_query,
        on="query_token",
        how="left",
        suffixes=("", "_expected"),
        validate="many_to_one",
        indicator=True,
    )
    if not merged["_merge"].eq("both").all():
        raise ValueError("Prediction table contains a query outside the sealed manifest.")
    for column in (
        "subject_id",
        "electrode_type",
        "checkpoint_group",
        "query_identity_sha256",
    ):
        if not merged[column].astype(str).eq(merged[f"{column}_expected"].astype(str)).all():
            raise ValueError(f"Prediction {column} differs from the sealed manifest.")
    merged = merged.drop(
        columns=[
            "subject_id_expected",
            "electrode_type_expected",
            "checkpoint_group_expected",
            "query_identity_sha256_expected",
            "_merge",
        ]
    )

    merged = merged.merge(
        expected_support,
        on=["subject_id", "electrode_type", "checkpoint_group", "budget"],
        how="left",
        suffixes=("", "_expected"),
        validate="many_to_one",
    )
    if not merged["support_identity_sha256"].astype(str).eq(
        merged["support_identity_sha256_expected"].astype(str)
    ).all():
        raise ValueError("Prediction support identity differs from the sealed support partition.")
    merged = merged.drop(columns=["support_identity_sha256_expected"])

    probability = merged.loc[:, spec.probability_columns].to_numpy(dtype=np.float64)
    if (
        not np.isfinite(probability).all()
        or (probability < 0.0).any()
        or (probability > 1.0).any()
        or not np.allclose(probability.sum(axis=1), 1.0, rtol=0.0, atol=1e-6)
    ):
        raise ValueError("Every row must contain one finite normalized probability vector.")
    checkpoint_counts = merged.groupby(
        ["seed", "checkpoint_group"], sort=False
    )["checkpoint_sha256"].nunique()
    if not checkpoint_counts.eq(1).all():
        raise ValueError(
            "Every seed/checkpoint-group must use one composite checkpoint across its grid."
        )
    checkpoint_keys = merged.loc[
        :, ["seed", "checkpoint_group", "checkpoint_sha256"]
    ].drop_duplicates()
    if checkpoint_keys["checkpoint_sha256"].duplicated().any():
        raise ValueError("Distinct seed/checkpoint-group keys must not reuse a checkpoint hash.")
    support_counts = merged.groupby(["subject_id", "electrode_type", "budget"], sort=False)[
        "support_identity_sha256"
    ].nunique()
    if not support_counts.eq(1).all():
        raise ValueError("Support identity differs across roles, contexts, or seeds.")
    intervention_counts = merged.groupby(
        ["role", "context", "checkpoint_group"], sort=False
    )["intervention_mapping_sha256"].nunique()
    if not intervention_counts.eq(1).all():
        raise ValueError(
            "Intervention mapping differs within one frozen role/context/checkpoint group."
        )
    resolved_usage_counts = merged.groupby(
        ["role", "context", "budget", "checkpoint_group"], sort=False
    )["resolved_context_usage_sha256"].nunique()
    if not resolved_usage_counts.eq(1).all():
        raise ValueError("Resolved context usage differs across seeds within one execution job.")
    for (subject_id, condition), group in expected_query.groupby(
        ["subject_id", "electrode_type"], sort=False
    ):
        expected_digests = set(group["query_identity_sha256"].astype(str))
        if len(expected_digests) != 1:
            raise ValueError("Expected query identity digest is not unique within a cell.")
        observed = merged.loc[
            merged["subject_id"].astype(str).eq(str(subject_id))
            & merged["electrode_type"].astype(str).eq(str(condition)),
            "query_identity_sha256",
        ].astype(str)
        if set(observed) != expected_digests:
            raise ValueError("Query identity digest differs across the atomic grid.")
    return merged.sort_values(key, kind="mergesort").reset_index(drop=True)


def _validate_exact_missing_fallback(
    frame: pd.DataFrame,
    *,
    spec: MetadataCalibrationBundleSpec,
) -> bool:
    key = ["seed", "budget", "query_token"]
    aq = frame.loc[
        frame["role"].eq("A_Q") & frame["context"].eq("exact_off"),
        [*key, *spec.probability_columns],
    ]
    missing = frame.loc[
        frame["role"].eq("A_QM") & frame["context"].eq("all_missing"),
        [*key, *spec.probability_columns],
    ]
    paired = aq.merge(missing, on=key, suffixes=("_aq", "_missing"), validate="one_to_one")
    for column in spec.probability_columns:
        if not np.array_equal(
            paired[f"{column}_aq"].to_numpy(), paired[f"{column}_missing"].to_numpy()
        ):
            raise ValueError("All-missing A_QM probabilities are not exactly equal to A_Q.")
    return True


def _validate_contrast_table(frame: pd.DataFrame, *, expected_n: int) -> pd.DataFrame:
    derived = {
        "subject_id",
        "eauc_increment",
        "correct_minus_shuffle_eauc",
        "correct_minus_stale_eauc",
        "wrong_minus_aq_eauc",
        "calibration_value_k1_to_k3",
        "aq_k1",
        "aq_k3",
        "aq_k5",
        "aqm_k0",
        "aqm_k1",
        "aqm_k3",
        "save_k0_vs_k1",
        "save_k1_vs_k3",
        "save_k3_vs_k5",
    }
    raw_ba = {
        *(f"aq_k{budget}" for budget in (0, 1, 3, 5)),
        *(f"aqm_k{budget}" for budget in (0, 1, 3, 5)),
        *(f"shuffle_k{budget}" for budget in (0, 1, 3)),
        *(f"stale_k{budget}" for budget in (0, 1, 3)),
        *(f"wrong_k{budget}" for budget in (0, 1, 3)),
    }
    required = derived | raw_ba
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Participant contrast table is missing columns: {sorted(missing)}.")
    subject_ids = frame["subject_id"].astype(str)
    if (
        len(frame) != expected_n
        or subject_ids.duplicated().any()
        or subject_ids.str.strip().eq("").any()
    ):
        raise ValueError(f"Participant contrast table must contain exactly {expected_n} subjects.")
    numeric = frame.loc[:, sorted(required - {"subject_id"})].to_numpy(dtype=float)
    if not np.isfinite(numeric).all():
        raise ValueError("Participant contrasts must be finite.")
    result = frame.copy()
    ba_values = result.loc[:, sorted(raw_ba)].to_numpy(dtype=float)
    if (ba_values < 0.0).any() or (ba_values > 1.0).any():
        raise ValueError("Participant balanced-accuracy inputs must lie in [0,1].")

    weights = {0: 1.0 / 6.0, 1: 1.0 / 2.0, 3: 1.0 / 3.0}
    expected_values = {
        "eauc_increment": sum(
            weight * (result[f"aqm_k{budget}"] - result[f"aq_k{budget}"])
            for budget, weight in weights.items()
        ),
        "correct_minus_shuffle_eauc": sum(
            weight * (result[f"aqm_k{budget}"] - result[f"shuffle_k{budget}"])
            for budget, weight in weights.items()
        ),
        "correct_minus_stale_eauc": sum(
            weight * (result[f"aqm_k{budget}"] - result[f"stale_k{budget}"])
            for budget, weight in weights.items()
        ),
        "wrong_minus_aq_eauc": sum(
            weight * (result[f"wrong_k{budget}"] - result[f"aq_k{budget}"])
            for budget, weight in weights.items()
        ),
        "calibration_value_k1_to_k3": result["aq_k3"] - result["aq_k1"],
        "save_k0_vs_k1": result["aqm_k0"] - result["aq_k1"],
        "save_k1_vs_k3": result["aqm_k1"] - result["aq_k3"],
        "save_k3_vs_k5": result["aqm_k3"] - result["aq_k5"],
    }
    for column, expected in expected_values.items():
        if not np.allclose(
            result[column].to_numpy(dtype=float),
            expected.to_numpy(dtype=float),
            rtol=0.0,
            atol=1e-12,
        ):
            raise ValueError(f"Participant contrast column {column} is algebraically inconsistent.")
    return result


def _sensitivity_summary(
    values: np.ndarray,
    *,
    spec: MetadataCalibrationBundleSpec,
) -> dict[str, object]:
    rng = np.random.default_rng(spec.sensitivity_seed)
    draws = rng.choice(
        values,
        size=(spec.sensitivity_resamples, len(values)),
        replace=True,
    ).mean(axis=1)
    bootstrap_low = float(np.quantile(draws, spec.primary_alpha, method="linear"))
    signs = rng.choice((-1.0, 1.0), size=(spec.sensitivity_resamples, len(values)))
    observed = float(values.mean())
    null_means = (signs * values).mean(axis=1)
    exceedances = int(np.count_nonzero(null_means >= observed))
    sign_flip_p = float((exceedances + 1) / (len(null_means) + 1))
    return {
        "bootstrap_percentile_lower": bootstrap_low,
        "bootstrap_resamples": spec.sensitivity_resamples,
        "paired_sign_flip_p_value": sign_flip_p,
        "sign_flip_resamples": spec.sensitivity_resamples,
        "seed": spec.sensitivity_seed,
        "decision_role": "sensitivity_only",
    }


def _holm_step_down(
    p_values: dict[str, float],
    *,
    alpha: float,
) -> dict[str, dict[str, float | bool]]:
    if not p_values or not 0.0 < alpha < 1.0:
        raise ValueError("Holm input and alpha must be nonempty and valid.")
    if any(not np.isfinite(value) or not 0.0 <= value <= 1.0 for value in p_values.values()):
        raise ValueError("Holm p-values must be finite values in [0,1].")
    ordered = sorted(p_values.items(), key=lambda item: (item[1], item[0]))
    total = len(ordered)
    adjusted_running = 0.0
    family_open = True
    results: dict[str, dict[str, float | bool]] = {}
    for rank, (name, value) in enumerate(ordered, start=1):
        multiplier = total - rank + 1
        adjusted_running = max(adjusted_running, min(1.0, multiplier * value))
        local_alpha = alpha / multiplier
        reject = bool(family_open and value < local_alpha)
        family_open = bool(family_open and reject)
        results[name] = {
            "holm_adjusted_p_value": adjusted_running,
            "holm_local_alpha": local_alpha,
            "holm_reject": reject,
        }
    return results


def _strict_integer(values: pd.Series, *, name: str) -> pd.Series:
    numeric = pd.to_numeric(values, errors="raise")
    if not np.isfinite(numeric).all() or not np.equal(numeric, np.floor(numeric)).all():
        raise ValueError(f"{name} must contain finite integers.")
    return numeric.astype(int)


def _staging_content_sha256(
    frame: pd.DataFrame,
    *,
    spec: MetadataCalibrationBundleSpec,
) -> str:
    key = ["role", "context", "seed", "budget", "query_token"]
    ordered = frame.sort_values(key, kind="mergesort").reset_index(drop=True)
    digest = hashlib.sha256()
    probability_columns = set(spec.probability_columns)
    for column in sorted(set(ordered.columns) - probability_columns):
        payload = ("\n".join(ordered[column].astype(str).tolist()) + "\n").encode("utf-8")
        digest.update(column.encode("utf-8"))
        digest.update(len(payload).to_bytes(8, "big"))
        digest.update(payload)
    probability = ordered.loc[:, spec.probability_columns].to_numpy(dtype="<f8", copy=True)
    digest.update(probability.tobytes(order="C"))
    return digest.hexdigest()


def _is_sha256(value: object) -> bool:
    return bool(_SHA256.fullmatch(str(value)))
