from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
from scipy.stats import t as student_t


def run_target_free_inference_simulation(
    config: Mapping[str, Any],
) -> dict[str, object]:
    """Stress-test the planned subject-level test without opening EEG outcomes.

    The simulation is expressed on effects after subtracting the eventual SESOI.
    Consequently, its null calibration and power curve are translation-invariant
    to the numerical SESOI later frozen in the confirmatory plan. No observed EEG
    metric is accepted by this function.
    """

    _validate_simulation_config(config)
    seed = int(config["seed"])
    n_simulations = int(config["n_simulations"])
    n_subjects = int(config["n_subjects"])
    n_folds = int(config["n_folds"])
    n_seeds = int(config["n_optimization_seeds"])
    alphas = tuple(float(value) for value in config["alpha_grid"])
    effect_grid = tuple(float(value) for value in config["effect_above_sesoi_grid"])
    n_sign_flips = int(config["sign_flip_resamples"])
    n_bootstraps = int(config["bootstrap_resamples"])
    max_null_rejection_rate = float(config["acceptance"]["max_null_rejection_rate"])
    power_effect = float(config["acceptance"]["power_effect_above_sesoi"])
    min_power = float(config["acceptance"]["minimum_power"])

    fold_ids = np.arange(n_subjects, dtype=np.int64) % n_folds
    master = np.random.default_rng(seed)
    signs = master.choice(
        np.asarray([-1.0, 1.0]), size=(n_sign_flips, n_subjects), replace=True
    )
    bootstrap_indices = master.integers(
        0, n_subjects, size=(n_bootstraps, n_subjects), endpoint=False
    )
    bootstrap_counts = np.zeros((n_bootstraps, n_subjects), dtype=np.float64)
    np.add.at(
        bootstrap_counts,
        (
            np.repeat(np.arange(n_bootstraps), n_subjects),
            bootstrap_indices.reshape(-1),
        ),
        1.0,
    )
    scenario_rows: list[dict[str, object]] = []
    freeze_eligible = bool(config.get("freeze_eligible", False))
    all_pass = True
    for scenario_index, raw_scenario in enumerate(config["scenarios"]):
        scenario = dict(raw_scenario)
        scenario_seed = seed + 10_000 * (scenario_index + 1)
        rng = np.random.default_rng(scenario_seed)
        centered = _simulate_centered_subject_effects(
            rng,
            n_simulations=n_simulations,
            n_subjects=n_subjects,
            n_folds=n_folds,
            n_seeds=n_seeds,
            fold_ids=fold_ids,
            scenario=scenario,
        )
        scenario_result: dict[str, object] = {
            "name": str(scenario["name"]),
            "distribution": str(scenario["distribution"]),
            "scenario_seed": scenario_seed,
            "subject_sd": float(scenario["subject_sd"]),
            "seed_subject_sd": float(scenario.get("seed_subject_sd", 0.0)),
            "seed_global_sd": float(scenario.get("seed_global_sd", 0.0)),
            "fold_seed_sd": float(scenario.get("fold_seed_sd", 0.0)),
            "fold_icc": float(scenario.get("fold_icc", 0.0)),
            "overlap_coupling": float(scenario.get("overlap_coupling", 0.0)),
            "effect_at_bound_fraction": float(
                np.count_nonzero((centered <= -1.0) | (centered >= 1.0)) / centered.size
            ),
            "null": {},
            "power": {},
        }
        for alpha in alphas:
            null_result = _evaluate_simulated_effects(
                centered,
                alpha=alpha,
                signs=signs,
                bootstrap_counts=bootstrap_counts,
                true_effect=0.0,
            )
            null_pass = (
                float(null_result["rejection_rate_wilson95_high"])
                <= max_null_rejection_rate
            )
            minimum_coverage = float(
                config["acceptance"].get("minimum_one_sided_coverage", 0.0)
            )
            maximum_bias = float(
                config["acceptance"].get("maximum_absolute_mean_bias", float("inf"))
            )
            maximum_mcse = float(
                config["acceptance"].get("maximum_type_i_mcse", float("inf"))
            )
            null_pass = bool(
                null_pass
                and float(null_result["coverage_wilson95_low"]) >= minimum_coverage
                and abs(float(null_result["mean_estimator_bias"])) <= maximum_bias
                and float(null_result["rejection_rate_mcse"]) <= maximum_mcse
            )
            scenario_result["null"][str(alpha)] = {
                **null_result,
                "maximum_allowed_rejection_rate": max_null_rejection_rate,
                "minimum_required_coverage": minimum_coverage,
                "maximum_allowed_absolute_mean_bias": maximum_bias,
                "maximum_allowed_type_i_mcse": maximum_mcse,
                "accepted": null_pass,
            }
            if str(scenario.get("acceptance_role", "core")) == "core":
                all_pass = all_pass and null_pass
            scenario_power: dict[str, object] = {}
            for effect in effect_grid:
                power_result = _evaluate_simulated_effects(
                    centered + effect,
                    alpha=alpha,
                    signs=signs,
                    bootstrap_counts=bootstrap_counts,
                    true_effect=effect,
                )
                power_result["effect_above_sesoi"] = effect
                scenario_power[str(effect)] = power_result
            selected = scenario_power[str(power_effect)]
            selected_pass = (
                float(selected["rejection_rate_wilson95_low"]) >= min_power
            )
            selected["minimum_required_power"] = min_power
            selected["accepted"] = selected_pass
            if str(scenario.get("acceptance_role", "core")) == "core":
                all_pass = all_pass and selected_pass
            scenario_result["power"][str(alpha)] = scenario_power
        scenario_result["acceptance_role"] = str(
            scenario.get("acceptance_role", "core")
        )
        if scenario_result["acceptance_role"] != "core":
            # Keep all diagnostics visible without allowing an explicitly
            # report-only stress envelope to veto the frozen core DGP family.
            for result in scenario_result["null"].values():
                result["accepted_for_freeze"] = None
        scenario_rows.append(scenario_result)

    return {
        "schema": "cfeg.target-free-inference-simulation.v1",
        "status": (
            "accepted"
            if freeze_eligible and all_pass
            else "rejected"
            if freeze_eligible
            else "diagnostic_complete"
        ),
        "freeze_eligible": freeze_eligible,
        "acceptance_checks_passed": all_pass,
        "target_outcomes_accessed": False,
        "simulation_scale": "balanced_accuracy_delta_minus_future_sesoi",
        "sesoi_translation_invariant": True,
        "estimand": "mean_seed_averaged_subject_balanced_accuracy_delta",
        "n_simulations": n_simulations,
        "n_subjects": n_subjects,
        "n_folds": n_folds,
        "n_optimization_seeds": n_seeds,
        "fold_assignment": "deterministic_balanced_index_modulo_fold",
        "alpha_grid": list(alphas),
        "effect_above_sesoi_grid": list(effect_grid),
        "sign_flip_resamples": n_sign_flips,
        "bootstrap_resamples": n_bootstraps,
        "simulation_seed": seed,
        "acceptance": dict(config["acceptance"]),
        "scenarios": scenario_rows,
        "limitations": [
            "Simulation cannot establish the scientific value of a future SESOI.",
            "Validity remains conditional on the declared effect distributions and dependence stressors.",
            "The five overlapping training folds are not independent experimental replicates.",
            "Missing confirmatory predictions are fail-closed rather than imputed or simulated.",
        ],
    }


def _simulate_centered_subject_effects(
    rng: np.random.Generator,
    *,
    n_simulations: int,
    n_subjects: int,
    n_folds: int,
    n_seeds: int,
    fold_ids: np.ndarray,
    scenario: Mapping[str, Any],
) -> np.ndarray:
    distribution = str(scenario["distribution"])
    subject_sd = float(scenario["subject_sd"])
    shape = (n_simulations, n_subjects)
    if distribution == "normal":
        subject = rng.normal(0.0, subject_sd, size=shape)
    elif distribution == "student_t3":
        # Var(t_3)=3; rescale so subject_sd retains the same interpretation.
        subject = rng.standard_t(3.0, size=shape) * (subject_sd / math.sqrt(3.0))
    elif distribution == "centered_lognormal":
        sigma = float(scenario.get("skew_sigma", 0.8))
        raw = rng.lognormal(mean=0.0, sigma=sigma, size=shape)
        raw -= math.exp(0.5 * sigma**2)
        raw_sd = math.sqrt((math.exp(sigma**2) - 1.0) * math.exp(sigma**2))
        subject = raw * (subject_sd / raw_sd)
    elif distribution == "bounded_beta":
        alpha = float(scenario.get("beta_alpha", 2.0))
        beta = float(scenario.get("beta_beta", 5.0))
        raw = rng.beta(alpha, beta, size=shape)
        raw -= alpha / (alpha + beta)
        raw_sd = math.sqrt(
            alpha * beta / ((alpha + beta) ** 2 * (alpha + beta + 1.0))
        )
        subject = raw * (subject_sd / raw_sd)
    elif distribution == "negative_responder_mixture":
        probability = float(scenario.get("negative_responder_probability", 0.2))
        negative_location = -float(scenario.get("negative_responder_shift", 0.08))
        positive_location = -probability * negative_location / (1.0 - probability)
        negative = rng.random(size=shape) < probability
        locations = np.where(negative, negative_location, positive_location)
        residual_sd = float(scenario.get("mixture_residual_sd", subject_sd / 2.0))
        raw = locations + rng.normal(0.0, residual_sd, size=shape)
        mixture_variance = (
            probability * negative_location**2
            + (1.0 - probability) * positive_location**2
            + residual_sd**2
        )
        subject = raw * (subject_sd / math.sqrt(mixture_variance))
    elif distribution == "paired_binomial":
        trials = int(scenario.get("binomial_trials", 240))
        probability = float(scenario.get("binomial_probability", 0.5))
        candidate = rng.binomial(trials, probability, size=shape)
        baseline = rng.binomial(trials, probability, size=shape)
        raw = (candidate - baseline) / trials
        raw_sd = math.sqrt(2.0 * probability * (1.0 - probability) / trials)
        subject = raw * (subject_sd / raw_sd)
    else:  # pragma: no cover - rejected during validation
        raise ValueError(f"Unknown simulation distribution: {distribution}.")

    seed_subject_sd = float(scenario.get("seed_subject_sd", 0.0))
    seed_global_sd = float(scenario.get("seed_global_sd", 0.0))
    fold_seed_sd = float(scenario.get("fold_seed_sd", 0.0))
    fold_icc = float(scenario.get("fold_icc", 0.0))
    if fold_icc:
        fold_seed_sd = math.sqrt(fold_icc / (1.0 - fold_icc)) * subject_sd
    overlap_coupling = float(scenario.get("overlap_coupling", 0.0))
    seed_subject = rng.normal(
        0.0, seed_subject_sd, size=(n_simulations, n_seeds, n_subjects)
    )
    seed_global = rng.normal(0.0, seed_global_sd, size=(n_simulations, n_seeds, 1))
    if fold_seed_sd == 0.0:
        fold_seed = np.zeros((n_simulations, n_seeds, n_folds), dtype=np.float64)
    else:
        fold_correlation = _fold_overlap_correlation(fold_ids, n_folds)
        covariance = fold_seed_sd**2 * (
            (1.0 - overlap_coupling) * np.eye(n_folds)
            + overlap_coupling * fold_correlation
        )
        fold_seed = rng.multivariate_normal(
            np.zeros(n_folds), covariance, size=(n_simulations, n_seeds)
        )
    per_seed = (
        subject[:, None, :]
        + seed_subject
        + seed_global
        + fold_seed[:, :, fold_ids]
    )
    return np.clip(per_seed.mean(axis=1), -1.0, 1.0)


def _fold_overlap_correlation(fold_ids: np.ndarray, n_folds: int) -> np.ndarray:
    if n_folds == 1:
        return np.ones((1, 1), dtype=np.float64)
    training_incidence = np.stack(
        [(fold_ids != fold).astype(np.float64) for fold in range(n_folds)]
    )
    overlap = training_incidence @ training_incidence.T
    scale = np.sqrt(np.outer(np.diag(overlap), np.diag(overlap)))
    return overlap / scale


def _evaluate_simulated_effects(
    effects: np.ndarray,
    *,
    alpha: float,
    signs: np.ndarray,
    bootstrap_counts: np.ndarray,
    true_effect: float,
    batch_size: int = 32,
) -> dict[str, object]:
    n_simulations, n_subjects = effects.shape
    rejected = np.zeros(n_simulations, dtype=bool)
    p_values = np.empty(n_simulations, dtype=np.float64)
    lower_bounds = np.empty(n_simulations, dtype=np.float64)
    sign_transpose = signs.T
    for start in range(0, n_simulations, batch_size):
        stop = min(start + batch_size, n_simulations)
        batch = effects[start:stop]
        observed = batch.mean(axis=1)
        null_statistics = batch @ sign_transpose / n_subjects
        exceedances = np.count_nonzero(
            null_statistics >= observed[:, None], axis=1
        )
        p_value = (exceedances + 1.0) / (signs.shape[0] + 1.0)
        # Multinomial counts exactly encode each nonparametric bootstrap draw
        # while avoiding a [simulation, bootstrap, subject] materialization.
        bootstrap_means = batch @ bootstrap_counts.T / n_subjects
        observed_se = batch.std(axis=1, ddof=1) / math.sqrt(n_subjects)
        bootstrap_second = np.square(batch) @ bootstrap_counts.T
        bootstrap_variance = np.maximum(
            (bootstrap_second - n_subjects * np.square(bootstrap_means))
            / (n_subjects - 1),
            0.0,
        )
        bootstrap_se = np.sqrt(bootstrap_variance / n_subjects)
        studentized = np.divide(
            bootstrap_means - observed[:, None],
            bootstrap_se,
            out=np.zeros_like(bootstrap_means),
            where=bootstrap_se > np.finfo(float).eps,
        )
        upper_t = np.quantile(studentized, 1.0 - alpha, axis=1, method="linear")
        bootstrap_lower = observed - upper_t * observed_se
        constant = observed_se <= np.finfo(float).eps
        bootstrap_lower[constant] = observed[constant]
        lower = observed - float(student_t.ppf(1.0 - alpha, n_subjects - 1)) * observed_se
        p_values[start:stop] = p_value
        lower_bounds[start:stop] = lower
        rejected[start:stop] = (p_value <= alpha) & (lower > 0.0)
    count = int(rejected.sum())
    rate = float(count / n_simulations)
    interval_low, interval_high = _wilson_interval(count, n_simulations)
    covered = int(np.count_nonzero(lower_bounds <= true_effect))
    coverage = float(covered / n_simulations)
    coverage_low, coverage_high = _wilson_interval(covered, n_simulations)
    return {
        "rejections": count,
        "rejection_rate": rate,
        "rejection_rate_wilson95_low": interval_low,
        "rejection_rate_wilson95_high": interval_high,
        "rejection_rate_mcse": float(math.sqrt(rate * (1.0 - rate) / n_simulations)),
        "one_sided_lower_bound_coverage": coverage,
        "coverage_wilson95_low": coverage_low,
        "coverage_wilson95_high": coverage_high,
        "mean_estimator_bias": float(effects.mean() - true_effect),
        "median_p_value": float(np.median(p_values)),
        "median_primary_t_lower_bound": float(np.median(lower_bounds)),
        "median_bootstrap_t_sensitivity_lower_bound": float(
            np.median(bootstrap_lower)
        ),
    }


def _wilson_interval(successes: int, trials: int, z: float = 1.959963984540054) -> tuple[float, float]:
    proportion = successes / trials
    denominator = 1.0 + z**2 / trials
    center = (proportion + z**2 / (2.0 * trials)) / denominator
    half = (
        z
        * math.sqrt(
            proportion * (1.0 - proportion) / trials + z**2 / (4.0 * trials**2)
        )
        / denominator
    )
    return float(max(0.0, center - half)), float(min(1.0, center + half))


def _validate_simulation_config(config: Mapping[str, Any]) -> None:
    if config.get("schema") != "cfeg.target-free-inference-simulation-plan.v1":
        raise ValueError("Unknown target-free inference-simulation plan schema.")
    positive_integer_fields = (
        "seed",
        "n_simulations",
        "n_subjects",
        "n_folds",
        "n_optimization_seeds",
        "sign_flip_resamples",
        "bootstrap_resamples",
    )
    for field in positive_integer_fields:
        if int(config.get(field, 0)) < 1:
            raise ValueError(f"{field} must be a positive integer.")
    alphas = _positive_finite_sequence(config.get("alpha_grid"), "alpha_grid")
    if any(value >= 1.0 for value in alphas):
        raise ValueError("alpha_grid values must lie within (0, 1).")
    effects = _positive_finite_sequence(
        config.get("effect_above_sesoi_grid"), "effect_above_sesoi_grid"
    )
    acceptance = config.get("acceptance") or {}
    for field in ("max_null_rejection_rate", "minimum_power"):
        value = float(acceptance.get(field, -1.0))
        if not 0.0 < value < 1.0:
            raise ValueError(f"acceptance.{field} must lie within (0, 1).")
    for field in ("minimum_one_sided_coverage",):
        if field in acceptance and not 0.0 < float(acceptance[field]) < 1.0:
            raise ValueError(f"acceptance.{field} must lie within (0, 1).")
    for field in ("maximum_absolute_mean_bias", "maximum_type_i_mcse"):
        if field in acceptance and float(acceptance[field]) <= 0.0:
            raise ValueError(f"acceptance.{field} must be positive.")
    selected_effect = float(acceptance.get("power_effect_above_sesoi", -1.0))
    if selected_effect not in effects:
        raise ValueError(
            "acceptance.power_effect_above_sesoi must be present in the effect grid."
        )
    scenarios = config.get("scenarios")
    if not isinstance(scenarios, Sequence) or not scenarios:
        raise ValueError("At least one simulation scenario is required.")
    names: set[str] = set()
    for raw in scenarios:
        scenario = dict(raw)
        name = str(scenario.get("name", ""))
        if not name or name in names:
            raise ValueError("Simulation scenario names must be non-empty and unique.")
        names.add(name)
        if scenario.get("distribution") not in {
            "normal",
            "student_t3",
            "centered_lognormal",
            "bounded_beta",
            "negative_responder_mixture",
            "paired_binomial",
        }:
            raise ValueError(f"Unknown distribution in scenario {name!r}.")
        if float(scenario.get("subject_sd", 0.0)) <= 0.0:
            raise ValueError(f"Scenario {name!r} must define subject_sd > 0.")
        for field in ("seed_subject_sd", "seed_global_sd", "fold_seed_sd"):
            if float(scenario.get(field, 0.0)) < 0.0:
                raise ValueError(f"Scenario {name!r} has negative {field}.")
        fold_icc = float(scenario.get("fold_icc", 0.0))
        coupling = float(scenario.get("overlap_coupling", 0.0))
        if not 0.0 <= fold_icc < 1.0:
            raise ValueError(f"Scenario {name!r} has invalid fold_icc.")
        if not 0.0 <= coupling <= 1.0:
            raise ValueError(f"Scenario {name!r} has invalid overlap_coupling.")


def _positive_finite_sequence(value: Any, name: str) -> tuple[float, ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)) or not value:
        raise ValueError(f"{name} must be a non-empty sequence.")
    result = tuple(float(item) for item in value)
    if any(not np.isfinite(item) or item <= 0.0 for item in result):
        raise ValueError(f"{name} values must be positive and finite.")
    if len(set(result)) != len(result):
        raise ValueError(f"{name} values must be unique.")
    return result
