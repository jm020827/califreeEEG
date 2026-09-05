from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from typing import Literal

import numpy as np
from scipy.stats import nct
from scipy.stats import t as student_t

DistributionFamily = Literal["normal", "standardized_t5", "twenty_percent_harmed_mixture"]

LEGACY_SEVEN_ENDPOINTS = (
    "primary_eAUC_superiority",
    "correct_context_over_block_shuffle",
    "correct_context_over_stale",
    "wrong_context_noninferiority",
    "A_QM_k0_noninferior_to_A_Q_k1",
    "A_QM_k1_noninferior_to_A_Q_k3",
    "A_QM_k3_noninferior_to_A_Q_k5",
)
HELD_CORE_ENDPOINTS = (
    "primary_eAUC_superiority",
    "calibration_value_A_Q_k3_over_k1",
    "A_QM_k1_noninferior_to_A_Q_k3",
)
HELD_MECHANISM_ENDPOINTS = (
    "correct_context_over_block_shuffle",
    "correct_context_over_stale",
)
DEVELOPMENT_CHECKS = (
    "baseline_viability",
    "early_budget_mean",
    "early_budget_median",
    "early_budget_positive_participants",
    "early_budget_every_fold_positive",
    "correct_minus_shuffle_mean",
    "correct_minus_shuffle_positive_participants",
    "correct_minus_shuffle_no_negative_fold",
    "correct_minus_stale_mean",
    "correct_minus_stale_positive_participants",
    "correct_minus_stale_no_negative_fold",
    "wrong_context_mean_safety",
    "wrong_context_severe_harm_count",
    "wrong_context_no_catastrophic_harm",
    "correct_context_severe_harm_count",
    "correct_context_no_catastrophic_harm",
    "missing_exact_fallback",
    "metadata_only_shortcut",
)
DEVELOPMENT_HARD_GATES = (
    "baseline_viability",
    "early_budget_mean",
    "early_budget_every_fold_positive",
    "correct_minus_shuffle_mean",
    "correct_minus_shuffle_no_negative_fold",
    "missing_exact_fallback",
    "metadata_only_shortcut",
)
DEVELOPMENT_REQUIRED_REPORTS = tuple(
    name for name in DEVELOPMENT_CHECKS if name not in DEVELOPMENT_HARD_GATES
)


@dataclass(frozen=True)
class MonteCarloSpec:
    draws: int = 200_000
    chunk_size: int = 10_000
    master_seed: int = 20260904

    def __post_init__(self) -> None:
        if self.draws <= 0 or self.chunk_size <= 0:
            raise ValueError("Monte Carlo draws and chunk size must be positive.")


def one_sample_t_power(
    *,
    n: int,
    true_mean: float,
    sd: float,
    null_margin: float,
    alpha: float,
) -> float:
    """Exact normal-theory rejection power for a one-sided paired/one-sample t test."""

    if n < 2 or sd <= 0.0 or not 0.0 < alpha < 1.0:
        raise ValueError("n, sd, or alpha is invalid.")
    critical = student_t.ppf(1.0 - alpha, df=n - 1)
    noncentrality = (true_mean - null_margin) * math.sqrt(n) / sd
    return float(nct.sf(critical, df=n - 1, nc=noncentrality))


def simulate_marginal_claim(
    *,
    n: int,
    true_mean: float,
    sd: float,
    null_margin: float,
    alpha: float,
    practical_threshold: float | None,
    family: DistributionFamily,
    simulation: MonteCarloSpec,
) -> dict[str, float | int | str | None]:
    """Separate inferential rejection power from the full practical claim probability."""

    cell = {
        "kind": "marginal",
        "n": n,
        "true_mean": true_mean,
        "sd": sd,
        "null_margin": null_margin,
        "alpha": alpha,
        "practical_threshold": practical_threshold,
        "family": family,
    }
    rng = np.random.default_rng(_cell_seed(simulation.master_seed, cell))
    rejection_count = claim_count = 0
    realized_sum = realized_square_sum = 0.0
    realized_n = 0
    for chunk in _chunk_sizes(simulation.draws, simulation.chunk_size):
        standardized = _standardized_draws(rng, shape=(chunk, n, 1), family=family)[..., 0]
        values = true_mean + sd * standardized
        reject, sample_mean = _one_sided_t_rejections(
            values,
            null_margin=null_margin,
            alpha=alpha,
        )
        claim = reject if practical_threshold is None else reject & (sample_mean >= practical_threshold)
        rejection_count += int(reject.sum())
        claim_count += int(claim.sum())
        realized_sum += float(values.sum())
        realized_square_sum += float(np.square(values).sum())
        realized_n += values.size
    realized_mean, realized_sd = _moments(realized_sum, realized_square_sum, realized_n)
    return {
        **cell,
        "draws": simulation.draws,
        "cell_seed": _cell_seed(simulation.master_seed, cell),
        "inferential_rejection_power": rejection_count / simulation.draws,
        "full_claim_probability": claim_count / simulation.draws,
        "realized_mean": realized_mean,
        "realized_sd": realized_sd,
        "maximum_nominal_mc_standard_error": 0.5 / math.sqrt(simulation.draws),
    }


def simulate_held_core_hierarchy(
    *,
    true_means: tuple[float, float, float],
    sds: tuple[float, float, float],
    correlation_matrix: tuple[tuple[float, float, float], ...],
    family: DistributionFamily,
    simulation: MonteCarloSpec,
    n: int = 60,
    utility_floor_assumed: bool = True,
) -> dict[str, object]:
    """Simulate H1 followed by the H2a/H2b intersection calibration-saving claim."""

    means_array = np.asarray(true_means, dtype=float)
    sds_array = np.asarray(sds, dtype=float)
    correlation_array = np.asarray(correlation_matrix, dtype=float)
    if means_array.shape != (3,) or sds_array.shape != (3,):
        raise ValueError("Held core simulation requires three means and three SDs.")
    if np.any(sds_array <= 0.0):
        raise ValueError("Held core SDs must be positive.")
    _validate_correlation_matrix(correlation_array, dimension=3)
    cell = {
        "kind": "held_core_hierarchy",
        "n": n,
        "true_means": means_array.tolist(),
        "sds": sds_array.tolist(),
        "correlation_matrix": correlation_array.tolist(),
        "family": family,
        "utility_floor_assumed": utility_floor_assumed,
    }
    rng = np.random.default_rng(_cell_seed(simulation.master_seed, cell))
    local_counts = np.zeros(3, dtype=np.int64)
    h1_count = h2_reached_count = h2_count = consistency_count = 0
    harmed_sum = total_participants = 0
    for chunk in _chunk_sizes(simulation.draws, simulation.chunk_size):
        values, harmed = _correlated_draws_from_matrix(
            rng,
            draws=chunk,
            n=n,
            means=means_array,
            sds=sds_array,
            correlation_matrix=correlation_array,
            family=family,
            harmed_direction=np.asarray([1.0, 0.0, 1.0]),
        )
        means = values.mean(axis=1)
        h1_reject, _ = _one_sided_t_rejections(
            values[:, :, 0], null_margin=0.0, alpha=0.05
        )
        h2a_reject, _ = _one_sided_t_rejections(
            values[:, :, 1], null_margin=0.0, alpha=0.05
        )
        h2b_reject, _ = _one_sided_t_rejections(
            values[:, :, 2], null_margin=-(1.0 / 60.0), alpha=0.025
        )
        h1_local = h1_reject & (means[:, 0] >= 0.020)
        h2a_local = h2a_reject & (means[:, 1] >= 0.020)
        h2b_local = h2b_reject & bool(utility_floor_assumed)
        local = np.column_stack([h1_local, h2a_local, h2b_local])
        h2 = h1_local & h2a_local & h2b_local
        local_counts += local.sum(axis=0)
        h1_count += int(h1_local.sum())
        h2_reached_count += int(h1_local.sum())
        h2_count += int(h2.sum())
        consistency_count += int((h2 & ((means[:, 1] + means[:, 2]) > 0.0)).sum())
        harmed_sum += harmed
        total_participants += chunk * n
    return {
        **cell,
        "draws": simulation.draws,
        "cell_seed": _cell_seed(simulation.master_seed, cell),
        "endpoint_order": list(HELD_CORE_ENDPOINTS),
        "hierarchy": {
            "H1": "primary_eAUC_superiority",
            "H2_intersection": [
                "calibration_value_A_Q_k3_over_k1",
                "A_QM_k1_noninferior_to_A_Q_k3",
            ],
        },
        "local_claim_probability": {
            name: float(value / simulation.draws)
            for name, value in zip(HELD_CORE_ENDPOINTS, local_counts)
        },
        "H1_claim_probability": h1_count / simulation.draws,
        "H2_reach_probability": h2_reached_count / simulation.draws,
        "calibration_saving_full_claim_probability": h2_count / simulation.draws,
        "same_budget_positive_identity_probability_among_H2": (
            consistency_count / h2_count if h2_count else None
        ),
        "utility_floor_role": "assumption_not_identified_by_contrast_only_simulation",
        "realized_harmed_fraction": (
            harmed_sum / total_participants
            if family == "twenty_percent_harmed_mixture"
            else None
        ),
        "maximum_nominal_mc_standard_error": 0.5 / math.sqrt(simulation.draws),
    }


def simulate_held_mechanism_holm(
    *,
    true_means: tuple[float, float],
    sds: tuple[float, float],
    endpoint_correlation: float,
    family: DistributionFamily,
    simulation: MonteCarloSpec,
    n: int = 60,
    practical_threshold: float = 0.010,
) -> dict[str, object]:
    """Power sensitivity for the two held mechanism endpoints after core opens."""

    means_array = np.asarray(true_means, dtype=float)
    sds_array = np.asarray(sds, dtype=float)
    if means_array.shape != (2,) or sds_array.shape != (2,) or np.any(sds_array <= 0.0):
        raise ValueError("Held mechanism sensitivity requires two means and positive SDs.")
    if not -1.0 < endpoint_correlation < 1.0:
        raise ValueError("Held mechanism endpoint correlation must lie strictly inside (-1,1).")
    correlation_matrix = np.asarray(
        [[1.0, endpoint_correlation], [endpoint_correlation, 1.0]], dtype=float
    )
    cell = {
        "kind": "held_mechanism_holm",
        "n": n,
        "true_means": means_array.tolist(),
        "sds": sds_array.tolist(),
        "endpoint_correlation": endpoint_correlation,
        "family": family,
        "practical_threshold": practical_threshold,
    }
    rng = np.random.default_rng(_cell_seed(simulation.master_seed, cell))
    individual_counts = np.zeros(2, dtype=np.int64)
    any_count = both_count = harmed_sum = total_participants = 0
    for chunk in _chunk_sizes(simulation.draws, simulation.chunk_size):
        values, harmed = _correlated_draws_from_matrix(
            rng,
            draws=chunk,
            n=n,
            means=means_array,
            sds=sds_array,
            correlation_matrix=correlation_matrix,
            family=family,
            harmed_direction=np.ones(2, dtype=float),
        )
        sample_means = values.mean(axis=1)
        p_values = _one_sided_t_p_values(values, null_margin=0.0)
        order = np.argsort(p_values, axis=1, kind="stable")
        ordered_p = np.take_along_axis(p_values, order, axis=1)
        smallest_rejects = ordered_p[:, 0] < 0.025
        largest_rejects = smallest_rejects & (ordered_p[:, 1] < 0.05)
        holm_reject = np.zeros_like(p_values, dtype=bool)
        row_indices = np.arange(chunk)
        holm_reject[row_indices[smallest_rejects], order[smallest_rejects, 0]] = True
        holm_reject[largest_rejects, :] = True
        claims = holm_reject & (sample_means >= practical_threshold)
        individual_counts += claims.sum(axis=0)
        any_count += int(claims.any(axis=1).sum())
        both_count += int(claims.all(axis=1).sum())
        harmed_sum += harmed
        total_participants += chunk * n
    return {
        **cell,
        "draws": simulation.draws,
        "cell_seed": _cell_seed(simulation.master_seed, cell),
        "endpoint_order": list(HELD_MECHANISM_ENDPOINTS),
        "holm_family_alpha": 0.05,
        "conditional_on_confirmatory_core_open": True,
        "not_a_joint_pipeline_probability": True,
        "individual_holm_claim_probability": {
            name: float(value / simulation.draws)
            for name, value in zip(HELD_MECHANISM_ENDPOINTS, individual_counts)
        },
        "any_mechanism_claim_probability": any_count / simulation.draws,
        "both_mechanism_claim_probability": both_count / simulation.draws,
        "realized_harmed_fraction": (
            harmed_sum / total_participants
            if family == "twenty_percent_harmed_mixture"
            else None
        ),
        "maximum_nominal_mc_standard_error": 0.5 / math.sqrt(simulation.draws),
    }


def simulate_legacy_seven_endpoint_sequence(
    *,
    true_means: tuple[float, ...],
    sds: tuple[float, ...],
    endpoint_correlation: float,
    family: DistributionFamily,
    simulation: MonteCarloSpec,
    n: int = 60,
    utility_floor_assumed: bool = True,
) -> dict[str, object]:
    """Retained diagnostic for the superseded seven-gate hierarchy.

    Calibration-saving probabilities are explicitly conditional on the supplied
    utility-floor assumption; outcome-free power analysis cannot infer endpoint
    BA levels from contrast means alone.
    """

    dimension = len(LEGACY_SEVEN_ENDPOINTS)
    if len(true_means) != dimension or len(sds) != dimension:
        raise ValueError(f"Held simulation requires {dimension} means and SDs.")
    if any(sd <= 0.0 for sd in sds) or not 0.0 <= endpoint_correlation < 1.0:
        raise ValueError("Held SDs or endpoint correlation are invalid.")
    cell = {
        "kind": "held_fixed_sequence",
        "n": n,
        "true_means": list(true_means),
        "sds": list(sds),
        "endpoint_correlation": endpoint_correlation,
        "family": family,
        "utility_floor_assumed": utility_floor_assumed,
    }
    rng = np.random.default_rng(_cell_seed(simulation.master_seed, cell))
    raw_counts = np.zeros(dimension, dtype=np.int64)
    claim_counts = np.zeros(dimension, dtype=np.int64)
    reach_counts = np.zeros(dimension, dtype=np.int64)
    saving_counts = np.zeros(4, dtype=np.int64)
    wrong_tail_safe_count = 0
    harmed_sum = 0
    total_participants = 0
    for chunk in _chunk_sizes(simulation.draws, simulation.chunk_size):
        values, harmed = _correlated_draws(
            rng,
            draws=chunk,
            n=n,
            means=np.asarray(true_means, dtype=float),
            sds=np.asarray(sds, dtype=float),
            correlation=endpoint_correlation,
            family=family,
        )
        raw = np.zeros((chunk, dimension), dtype=bool)
        means = values.mean(axis=1)
        for endpoint in range(dimension):
            null_margin = 0.0 if endpoint < 3 else -(1.0 / 60.0)
            alpha = 0.05 if endpoint < 3 else 0.025
            raw[:, endpoint], _ = _one_sided_t_rejections(
                values[:, :, endpoint],
                null_margin=null_margin,
                alpha=alpha,
            )
        practical = np.ones_like(raw)
        practical[:, 0] = means[:, 0] >= 0.020
        practical[:, 1:3] = means[:, 1:3] >= 0.010
        wrong = values[:, :, 3]
        tail_safe = (wrong < -0.05).sum(axis=1) <= 6
        tail_safe &= wrong.min(axis=1) >= -0.10
        wrong_tail_safe_count += int(tail_safe.sum())
        practical[:, 4:] = bool(utility_floor_assumed)
        local_claim = raw & practical
        confirmed = np.zeros_like(local_claim)
        sequence_open = np.ones(chunk, dtype=bool)
        for endpoint in range(dimension):
            reach_counts[endpoint] += int(sequence_open.sum())
            confirmed[:, endpoint] = sequence_open & local_claim[:, endpoint]
            sequence_open = confirmed[:, endpoint]
        raw_counts += raw.sum(axis=0)
        claim_counts += confirmed.sum(axis=0)
        number_savings = confirmed[:, 4:].sum(axis=1)
        for count in range(4):
            saving_counts[count] += int(np.count_nonzero(number_savings == count))
        harmed_sum += harmed
        total_participants += chunk * n
    return {
        **cell,
        "draws": simulation.draws,
        "cell_seed": _cell_seed(simulation.master_seed, cell),
        "endpoint_order": list(LEGACY_SEVEN_ENDPOINTS),
        "raw_rejection_probability": {
            name: float(value / simulation.draws)
            for name, value in zip(LEGACY_SEVEN_ENDPOINTS, raw_counts)
        },
        "stage_reach_probability": {
            name: float(value / simulation.draws)
            for name, value in zip(LEGACY_SEVEN_ENDPOINTS, reach_counts)
        },
        "confirmatory_claim_probability": {
            name: float(value / simulation.draws)
            for name, value in zip(LEGACY_SEVEN_ENDPOINTS, claim_counts)
        },
        "confirmed_savings_count_probability": {
            str(count): saving_counts[count] / simulation.draws for count in range(4)
        },
        "utility_floor_role": "assumption_not_inferred_from_contrast_distribution",
        "wrong_context_tail_report_pass_probability": (
            wrong_tail_safe_count / simulation.draws
        ),
        "wrong_context_tail_report_role": "required_report_not_fixed_sequence_gate",
        "realized_harmed_fraction": (
            harmed_sum / total_participants
            if family == "twenty_percent_harmed_mixture"
            else None
        ),
        "maximum_nominal_mc_standard_error": 0.5 / math.sqrt(simulation.draws),
    }


def simulate_development_gate(
    *,
    contrast_means: tuple[float, float, float, float],
    contrast_sds: tuple[float, float, float, float],
    baseline_mean: float,
    baseline_sd: float,
    endpoint_correlation: float,
    family: DistributionFamily,
    simulation: MonteCarloSpec,
    n: int = 39,
) -> dict[str, object]:
    """Estimate promotion probability for the exact three-by-thirteen development gate."""

    if n != 39:
        raise ValueError("The frozen development promotion gate requires exactly 39 participants.")
    means_vector = np.asarray([*contrast_means, baseline_mean], dtype=float)
    sds_vector = np.asarray([*contrast_sds, baseline_sd], dtype=float)
    if np.any(sds_vector <= 0.0) or not 0.0 <= endpoint_correlation < 1.0:
        raise ValueError("Development SDs or endpoint correlation are invalid.")
    cell = {
        "kind": "development_gate",
        "n": n,
        "contrast_means": list(contrast_means),
        "contrast_sds": list(contrast_sds),
        "baseline_mean": baseline_mean,
        "baseline_sd": baseline_sd,
        "endpoint_correlation": endpoint_correlation,
        "family": family,
    }
    rng = np.random.default_rng(_cell_seed(simulation.master_seed, cell))
    check_counts = np.zeros(len(DEVELOPMENT_CHECKS), dtype=np.int64)
    hard_first_failure_counts = np.zeros(len(DEVELOPMENT_HARD_GATES) + 1, dtype=np.int64)
    hard_gate_count = all_checks_count = 0
    harmed_sum = total_participants = 0
    for chunk in _chunk_sizes(simulation.draws, simulation.chunk_size):
        generated, harmed = _correlated_draws(
            rng,
            draws=chunk,
            n=n,
            means=means_vector,
            sds=sds_vector,
            correlation=endpoint_correlation,
            family=family,
        )
        eauc, shuffle, stale, wrong = (generated[:, :, index] for index in range(4))
        baseline = np.clip(generated[:, :, 4], 0.0, 1.0)
        folds = (
            eauc.reshape(chunk, 3, 13),
            shuffle.reshape(chunk, 3, 13),
            stale.reshape(chunk, 3, 13),
        )
        gate = np.column_stack(
            [
                baseline.mean(axis=1) >= 0.50,
                eauc.mean(axis=1) >= 0.010,
                np.median(eauc, axis=1) > 0.0,
                (eauc > 0.0).sum(axis=1) >= 24,
                folds[0].mean(axis=2).min(axis=1) > 0.0,
                shuffle.mean(axis=1) >= 0.010,
                (shuffle > 0.0).sum(axis=1) >= 24,
                folds[1].mean(axis=2).min(axis=1) >= 0.0,
                stale.mean(axis=1) >= 0.010,
                (stale > 0.0).sum(axis=1) >= 24,
                folds[2].mean(axis=2).min(axis=1) >= 0.0,
                wrong.mean(axis=1) >= -(1.0 / 60.0),
                (wrong < -0.05).sum(axis=1) <= 4,
                wrong.min(axis=1) >= -0.10,
                (eauc < -0.05).sum(axis=1) <= 4,
                eauc.min(axis=1) >= -0.10,
                np.ones(chunk, dtype=bool),
                np.ones(chunk, dtype=bool),
            ]
        )
        check_counts += gate.sum(axis=0)
        check_index = {name: index for index, name in enumerate(DEVELOPMENT_CHECKS)}
        hard = gate[:, [check_index[name] for name in DEVELOPMENT_HARD_GATES]]
        hard_pass = hard.all(axis=1)
        hard_gate_count += int(hard_pass.sum())
        all_checks_count += int(gate.all(axis=1).sum())
        first_failure = np.where(
            hard_pass,
            len(DEVELOPMENT_HARD_GATES),
            np.argmin(hard, axis=1),
        )
        hard_first_failure_counts += np.bincount(
            first_failure, minlength=len(DEVELOPMENT_HARD_GATES) + 1
        )
        harmed_sum += harmed
        total_participants += chunk * n
    return {
        **cell,
        "draws": simulation.draws,
        "cell_seed": _cell_seed(simulation.master_seed, cell),
        "all_missing_gate_role": "deterministic_unit_invariant_assumed_true",
        "metadata_only_gate_role": (
            "deterministic_block_constant_M_and_complete_class_balance_assumed_true"
        ),
        "individual_gate_pass_probability": {
            name: float(value / simulation.draws)
            for name, value in zip(DEVELOPMENT_CHECKS, check_counts)
        },
        "hard_promotion_gates": list(DEVELOPMENT_HARD_GATES),
        "required_diagnostic_or_deployment_reports": list(DEVELOPMENT_REQUIRED_REPORTS),
        "hard_gate_promotion_probability": hard_gate_count / simulation.draws,
        "all_checks_pass_probability": all_checks_count / simulation.draws,
        "hard_first_failure_probability": {
            **{
                name: float(hard_first_failure_counts[index] / simulation.draws)
                for index, name in enumerate(DEVELOPMENT_HARD_GATES)
            },
            "all_pass": float(hard_first_failure_counts[-1] / simulation.draws),
        },
        "balanced_accuracy_generation": "normal_scale_draw_then_clip_to_unit_interval",
        "realized_harmed_fraction": (
            harmed_sum / total_participants
            if family == "twenty_percent_harmed_mixture"
            else None
        ),
        "maximum_nominal_mc_standard_error": 0.5 / math.sqrt(simulation.draws),
    }


def _one_sided_t_rejections(
    values: np.ndarray,
    *,
    null_margin: float,
    alpha: float,
) -> tuple[np.ndarray, np.ndarray]:
    if values.ndim != 2 or values.shape[1] < 2:
        raise ValueError("Vectorized t input must have shape [draw,participant>=2].")
    means = values.mean(axis=1)
    sd = values.std(axis=1, ddof=1)
    se = sd / math.sqrt(values.shape[1])
    critical = student_t.ppf(1.0 - alpha, df=values.shape[1] - 1)
    statistic = np.divide(
        means - null_margin,
        se,
        out=np.where(means > null_margin, np.inf, -np.inf),
        where=se > np.finfo(float).eps,
    )
    return statistic > critical, means


def _one_sided_t_p_values(values: np.ndarray, *, null_margin: float) -> np.ndarray:
    if values.ndim != 3 or values.shape[1] < 2:
        raise ValueError("Vectorized multi-endpoint t input must be [draw,participant,endpoint].")
    means = values.mean(axis=1)
    sd = values.std(axis=1, ddof=1)
    se = sd / math.sqrt(values.shape[1])
    statistic = np.divide(
        means - null_margin,
        se,
        out=np.where(means > null_margin, np.inf, -np.inf),
        where=se > np.finfo(float).eps,
    )
    return student_t.sf(statistic, df=values.shape[1] - 1)


def _standardized_draws(
    rng: np.random.Generator,
    *,
    shape: tuple[int, int, int],
    family: DistributionFamily,
) -> np.ndarray:
    if family == "normal":
        return rng.standard_normal(shape)
    if family == "standardized_t5":
        normal = rng.standard_normal(shape)
        scale = np.sqrt(3.0 / rng.chisquare(5, size=shape[:-1] + (1,)))
        return normal * scale
    if family == "twenty_percent_harmed_mixture":
        harmed = rng.random(shape) < 0.20
        shift = np.where(harmed, -0.5, 0.125)
        return shift + math.sqrt(0.9375) * rng.standard_normal(shape)
    raise ValueError(f"Unknown distribution family {family!r}.")


def _correlated_draws(
    rng: np.random.Generator,
    *,
    draws: int,
    n: int,
    means: np.ndarray,
    sds: np.ndarray,
    correlation: float,
    family: DistributionFamily,
) -> tuple[np.ndarray, int]:
    dimension = len(means)
    correlation_matrix = (1.0 - correlation) * np.eye(dimension) + correlation * np.ones(
        (dimension, dimension)
    )
    covariance = np.outer(sds, sds) * correlation_matrix
    harmed_count = 0
    if family == "twenty_percent_harmed_mixture":
        shift_covariance = 0.0625 * np.outer(sds, sds)
        residual_covariance = covariance - shift_covariance
        residual = _covariance_factor(residual_covariance)
        normal = rng.standard_normal((draws, n, dimension)) @ residual.T
        harmed = rng.random((draws, n, 1)) < 0.20
        harmed_count = int(harmed.sum())
        shift = np.where(harmed, -0.5, 0.125) * sds.reshape(1, 1, -1)
        return means.reshape(1, 1, -1) + shift + normal, harmed_count
    factor = _covariance_factor(covariance)
    values = rng.standard_normal((draws, n, dimension)) @ factor.T
    if family == "standardized_t5":
        values *= np.sqrt(3.0 / rng.chisquare(5, size=(draws, n, 1)))
    elif family != "normal":
        raise ValueError(f"Unknown distribution family {family!r}.")
    return means.reshape(1, 1, -1) + values, harmed_count


def _correlated_draws_from_matrix(
    rng: np.random.Generator,
    *,
    draws: int,
    n: int,
    means: np.ndarray,
    sds: np.ndarray,
    correlation_matrix: np.ndarray,
    family: DistributionFamily,
    harmed_direction: np.ndarray,
) -> tuple[np.ndarray, int]:
    dimension = len(means)
    _validate_correlation_matrix(correlation_matrix, dimension=dimension)
    if harmed_direction.shape != (dimension,) or not np.isin(
        harmed_direction, [-1.0, 0.0, 1.0]
    ).all():
        raise ValueError("Harmed directions must be one -1/0/1 value per endpoint.")
    covariance = np.outer(sds, sds) * correlation_matrix
    harmed_count = 0
    if family == "twenty_percent_harmed_mixture":
        shift_vector = sds * harmed_direction
        residual_covariance = covariance - 0.0625 * np.outer(
            shift_vector, shift_vector
        )
        residual = _covariance_factor(residual_covariance)
        normal = rng.standard_normal((draws, n, dimension)) @ residual.T
        harmed = rng.random((draws, n, 1)) < 0.20
        harmed_count = int(harmed.sum())
        centered_shift = np.where(harmed, -0.5, 0.125) * shift_vector.reshape(
            1, 1, -1
        )
        return means.reshape(1, 1, -1) + centered_shift + normal, harmed_count
    factor = _covariance_factor(covariance)
    values = rng.standard_normal((draws, n, dimension)) @ factor.T
    if family == "standardized_t5":
        values *= np.sqrt(3.0 / rng.chisquare(5, size=(draws, n, 1)))
    elif family != "normal":
        raise ValueError(f"Unknown distribution family {family!r}.")
    return means.reshape(1, 1, -1) + values, harmed_count


def _validate_correlation_matrix(matrix: np.ndarray, *, dimension: int) -> None:
    if matrix.shape != (dimension, dimension):
        raise ValueError("Correlation matrix has the wrong dimension.")
    if not np.isfinite(matrix).all() or not np.allclose(matrix, matrix.T, atol=1e-12):
        raise ValueError("Correlation matrix must be finite and symmetric.")
    if not np.allclose(np.diag(matrix), 1.0, atol=1e-12):
        raise ValueError("Correlation matrix diagonal must be exactly one.")
    if (np.abs(matrix) > 1.0 + 1e-12).any():
        raise ValueError("Correlation entries must lie in [-1,1].")
    _covariance_factor(matrix)


def _covariance_factor(covariance: np.ndarray) -> np.ndarray:
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    tolerance = np.finfo(float).eps * max(1.0, float(np.abs(eigenvalues).max())) * 100
    if float(eigenvalues.min()) < -tolerance:
        raise ValueError("Requested mixture/correlation covariance is not positive semidefinite.")
    return eigenvectors @ np.diag(np.sqrt(np.clip(eigenvalues, 0.0, None)))


def _cell_seed(master_seed: int, cell: dict[str, object]) -> int:
    payload = json.dumps(cell, sort_keys=True, separators=(",", ":"), allow_nan=False)
    digest = hashlib.sha256(f"{master_seed}:{payload}".encode()).digest()
    return int.from_bytes(digest[:8], "big", signed=False)


def _chunk_sizes(total: int, chunk_size: int):
    remaining = total
    while remaining:
        current = min(remaining, chunk_size)
        yield current
        remaining -= current


def _moments(total: float, square_total: float, count: int) -> tuple[float, float]:
    mean = total / count
    variance = max(0.0, (square_total - count * mean * mean) / (count - 1))
    return mean, math.sqrt(variance)
