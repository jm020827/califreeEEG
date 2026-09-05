from __future__ import annotations

import math

import numpy as np
from scipy import stats

from cfeg.analysis.query_reliability_statistics import (
    analytic_one_sided_mean_t_power,
)


def analytic_power_grid(
    *,
    true_means: tuple[float, ...],
    standard_deviations: tuple[float, ...],
    null_margin: float,
    n_participants: int = 20,
    alpha: float = 0.05,
) -> list[dict[str, float | int]]:
    return [
        {
            "n_participants": int(n_participants),
            "alpha": float(alpha),
            "true_mean": float(mean),
            "null_margin": float(null_margin),
            "standard_deviation": float(sd),
            "power": analytic_one_sided_mean_t_power(
                n_participants=n_participants,
                true_mean=mean,
                null_margin=null_margin,
                standard_deviation=sd,
                alpha=alpha,
            ),
        }
        for sd in standard_deviations
        for mean in true_means
    ]


def simulate_primary_promotion_probability(
    *,
    true_mean: float,
    standard_deviation: float,
    draws: int,
    seed: int,
    n_participants: int = 20,
    alpha: float = 0.05,
    practical_effect_min: float = 0.02,
    required_positive: int = 12,
    chunk_size: int = 20_000,
) -> dict[str, float | int]:
    _validate_simulation(draws, n_participants, standard_deviation, chunk_size)
    rng = np.random.default_rng(seed)
    critical = float(stats.t.ppf(1.0 - alpha, n_participants - 1))
    passes = 0
    completed = 0
    while completed < draws:
        size = min(chunk_size, draws - completed)
        values = rng.normal(
            loc=true_mean,
            scale=standard_deviation,
            size=(size, n_participants),
        )
        mean = values.mean(axis=1)
        standard_error = values.std(axis=1, ddof=1) / math.sqrt(n_participants)
        lower = mean - critical * standard_error
        positive = (values > 0.0).sum(axis=1)
        passes += int(
            ((lower > 0.0) & (mean >= practical_effect_min) & (positive >= required_positive)).sum()
        )
        completed += size
    return _simulation_result(passes, draws)


def simulate_joint_promotion_probability(
    *,
    primary_mean: float,
    clean_simple_mean: float,
    clean_deployment_mean: float,
    standard_deviation: float,
    correlation: np.ndarray,
    draws: int,
    seed: int,
    n_participants: int = 20,
    alpha: float = 0.05,
    practical_effect_min: float = 0.02,
    clean_margin: float = -0.01,
    required_positive: int = 12,
    chunk_size: int = 20_000,
) -> dict[str, float | int]:
    _validate_simulation(draws, n_participants, standard_deviation, chunk_size)
    matrix = np.asarray(correlation, dtype=np.float64)
    if matrix.shape != (3, 3) or not np.allclose(matrix, matrix.T, rtol=0.0, atol=1e-12):
        raise ValueError("correlation must be one symmetric 3x3 matrix.")
    eigenvalues = np.linalg.eigvalsh(matrix)
    if not np.isfinite(matrix).all() or eigenvalues.min() <= 0.0:
        raise ValueError("correlation must be finite and positive definite.")
    if not np.allclose(np.diag(matrix), 1.0, rtol=0.0, atol=1e-12):
        raise ValueError("correlation diagonal must equal one.")

    means = np.asarray([primary_mean, clean_simple_mean, clean_deployment_mean], dtype=np.float64)
    rng = np.random.default_rng(seed)
    cholesky = np.linalg.cholesky(matrix)
    critical = float(stats.t.ppf(1.0 - alpha, n_participants - 1))
    passes = 0
    completed = 0
    while completed < draws:
        size = min(chunk_size, draws - completed)
        standard = rng.standard_normal((size, n_participants, 3))
        values = means + standard_deviation * (standard @ cholesky.T)
        mean = values.mean(axis=1)
        standard_error = values.std(axis=1, ddof=1) / math.sqrt(n_participants)
        lower = mean - critical * standard_error
        primary_positive = (values[:, :, 0] > 0.0).sum(axis=1)
        passes += int(
            (
                (lower[:, 0] > 0.0)
                & (mean[:, 0] >= practical_effect_min)
                & (primary_positive >= required_positive)
                & (lower[:, 1] > clean_margin)
                & (lower[:, 2] > clean_margin)
            ).sum()
        )
        completed += size
    return _simulation_result(passes, draws)


def build_query_reliability_sensitivity_audit(
    *,
    draws: int = 200_000,
    seed: int = 20260904,
) -> dict:
    correlations = {
        "independent": np.eye(3),
        "equicorrelation_0_25": np.full((3, 3), 0.25) + np.eye(3) * 0.75,
        "equicorrelation_0_50": np.full((3, 3), 0.50) + np.eye(3) * 0.50,
        "equicorrelation_0_80": np.full((3, 3), 0.80) + np.eye(3) * 0.20,
    }
    return {
        "schema": "cfeg.query-reliability-power-sensitivity.v1",
        "outcome_data_used": False,
        "status": "design_sensitivity_not_execution_authority",
        "seed": int(seed),
        "monte_carlo_draws_per_cell": int(draws),
        "analytic_primary": analytic_power_grid(
            true_means=(0.0, 0.01, 0.02, 0.03, 0.04),
            standard_deviations=(0.02, 0.04, 0.06, 0.08),
            null_margin=0.0,
        ),
        "analytic_clean_noninferiority": analytic_power_grid(
            true_means=(-0.01, -0.005, 0.0, 0.005, 0.01, 0.02),
            standard_deviations=(0.01, 0.015, 0.02, 0.03, 0.04, 0.06, 0.08),
            null_margin=-0.01,
        ),
        "analytic_mechanism": analytic_power_grid(
            true_means=(0.0, 0.005, 0.01, 0.02, 0.03),
            standard_deviations=(0.02, 0.04, 0.06),
            null_margin=0.0,
        ),
        "primary_full_gate_normal": [
            {
                "true_mean": mean,
                "standard_deviation": sd,
                **simulate_primary_promotion_probability(
                    true_mean=mean,
                    standard_deviation=sd,
                    draws=draws,
                    seed=seed + index,
                ),
            }
            for index, (mean, sd) in enumerate(
                (mean, sd) for mean in (0.02, 0.03) for sd in (0.02, 0.04, 0.06)
            )
        ],
        "joint_gate_normal_primary_0_03_clean_0_sd_0_04": {
            name: simulate_joint_promotion_probability(
                primary_mean=0.03,
                clean_simple_mean=0.0,
                clean_deployment_mean=0.0,
                standard_deviation=0.04,
                correlation=correlation,
                draws=draws,
                seed=seed + 100 + index,
            )
            for index, (name, correlation) in enumerate(correlations.items())
        },
    }


def _validate_simulation(
    draws: int,
    n_participants: int,
    standard_deviation: float,
    chunk_size: int,
) -> None:
    if draws <= 0 or chunk_size <= 0:
        raise ValueError("draws and chunk_size must be positive.")
    if n_participants < 2:
        raise ValueError("Simulation requires at least two participants.")
    if not np.isfinite(standard_deviation) or standard_deviation <= 0.0:
        raise ValueError("standard_deviation must be finite and positive.")


def _simulation_result(passes: int, draws: int) -> dict[str, float | int]:
    probability = float(passes / draws)
    return {
        "passes": int(passes),
        "draws": int(draws),
        "probability": probability,
        "monte_carlo_standard_error": math.sqrt(probability * (1.0 - probability) / draws),
    }
