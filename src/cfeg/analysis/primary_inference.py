from __future__ import annotations

import itertools

import numpy as np
import pandas as pd
from scipy.stats import t as student_t


def primary_subject_inference(
    subjects: pd.DataFrame,
    *,
    alpha: float,
    alternative: str,
    null_margin_ba: float,
    seed: int,
    n_resamples: int,
) -> dict[str, object]:
    """Inference on one seed-averaged paired difference per participant.

    The confirmatory directional claim is H0: E[D] <= null_margin_ba. The
    sign-flip calibration assumes exchangeability of centered paired effects;
    target-free simulation must validate robustness before this method is frozen.
    """

    required = {"subject_id", "balanced_accuracy_delta"}
    missing = required - set(subjects.columns)
    if missing:
        raise ValueError(f"Subject table missing columns: {sorted(missing)}")
    subject_ids = subjects["subject_id"].astype(str)
    if subject_ids.duplicated().any():
        duplicates = sorted(subject_ids[subject_ids.duplicated(keep=False)].unique())
        raise ValueError(f"Subject IDs must be unique for primary inference: {duplicates[:5]}.")
    differences = subjects["balanced_accuracy_delta"].to_numpy(dtype=float)
    if len(differences) < 2 or not np.isfinite(differences).all():
        raise ValueError("Subject differences must contain at least two finite values.")
    if not 0.0 < float(alpha) < 1.0:
        raise ValueError("alpha must lie within (0, 1).")
    if alternative not in {"greater", "two-sided"}:
        raise ValueError("alternative must be greater or two-sided.")
    if not np.isfinite(null_margin_ba):
        raise ValueError("null_margin_ba must be finite.")
    if n_resamples < 1:
        raise ValueError("n_resamples must be positive.")

    bootstrap_seed, sign_seed = np.random.SeedSequence(seed).spawn(2)
    bootstrap_rng = np.random.default_rng(bootstrap_seed)
    sign_rng = np.random.default_rng(sign_seed)
    draws = bootstrap_rng.choice(
        differences, size=(n_resamples, len(differences)), replace=True
    )
    bootstrap_means = draws.mean(axis=1)
    observed_mean = float(differences.mean())
    observed_se = float(differences.std(ddof=1) / np.sqrt(len(differences))) if len(
        differences
    ) > 1 else 0.0
    bootstrap_se = (
        draws.std(axis=1, ddof=1) / np.sqrt(len(differences))
        if len(differences) > 1
        else np.zeros(n_resamples, dtype=float)
    )
    valid_studentized = bootstrap_se > np.finfo(float).eps
    studentized = np.zeros(n_resamples, dtype=float)
    studentized[valid_studentized] = (
        bootstrap_means[valid_studentized] - observed_mean
    ) / bootstrap_se[valid_studentized]
    if alternative == "greater":
        if observed_se <= np.finfo(float).eps:
            ci_low = observed_mean
        else:
            upper_t = float(
                np.quantile(studentized[valid_studentized], 1.0 - alpha, method="linear")
            )
            bootstrap_sensitivity_low = float(observed_mean - upper_t * observed_se)
        if observed_se <= np.finfo(float).eps:
            bootstrap_sensitivity_low = observed_mean
        ci_low = float(
            observed_mean
            - float(student_t.ppf(1.0 - alpha, df=len(differences) - 1)) * observed_se
        )
        ci_high = None
    else:
        if observed_se <= np.finfo(float).eps:
            bootstrap_sensitivity_low = bootstrap_sensitivity_high = observed_mean
        else:
            lower_t, upper_t = np.quantile(
                studentized[valid_studentized],
                [alpha / 2.0, 1.0 - alpha / 2.0],
                method="linear",
            )
            bootstrap_sensitivity_low = float(observed_mean - upper_t * observed_se)
            bootstrap_sensitivity_high = float(observed_mean - lower_t * observed_se)
        critical = float(student_t.ppf(1.0 - alpha / 2.0, df=len(differences) - 1))
        ci_low = float(observed_mean - critical * observed_se)
        ci_high = float(observed_mean + critical * observed_se)

    centered = differences - float(null_margin_ba)
    observed = float(centered.mean())
    exact = len(centered) <= 16
    if exact:
        signs = np.asarray(list(itertools.product((-1.0, 1.0), repeat=len(centered))))
    else:
        signs = sign_rng.choice((-1.0, 1.0), size=(n_resamples, len(centered)))
    null_statistics = (signs * centered).mean(axis=1)
    if alternative == "greater":
        exceedances = int(np.count_nonzero(null_statistics >= observed))
    else:
        exceedances = int(np.count_nonzero(np.abs(null_statistics) >= abs(observed)))
    p_value = (
        float(exceedances / len(null_statistics))
        if exact
        else float((exceedances + 1) / (len(null_statistics) + 1))
    )
    reject = bool(p_value <= alpha and ci_low > null_margin_ba)
    return {
        "inference_schema": "cfeg.primary-subject-inference.v1",
        "estimand": "mean_seed_averaged_subject_balanced_accuracy_delta",
        "n_subjects": len(differences),
        "mean_subject_balanced_accuracy_delta": float(differences.mean()),
        "median_subject_balanced_accuracy_delta": float(np.median(differences)),
        "null_margin_balanced_accuracy": float(null_margin_ba),
        "alpha": float(alpha),
        "alternative": alternative,
        "confidence_level": float(1.0 - alpha),
        "confidence_interval_low": ci_low,
        "confidence_interval_high": ci_high,
        "confidence_interval_method": "student_t_mean_difference",
        "bootstrap_t_sensitivity_low": bootstrap_sensitivity_low,
        "bootstrap_t_sensitivity_high": (
            bootstrap_sensitivity_high if alternative == "two-sided" else None
        ),
        "paired_sign_flip_p_value": p_value,
        "sign_flip_exact": exact,
        "reject_null": reject,
        "minimum_effect_claim_supported": reject if alternative == "greater" else None,
        "assumption_warning": (
            "Sign-flip validity requires exchangeable centered paired effects; "
            "cross-fold training overlap is handled by target-free stress simulation, "
            "not by treating folds as independent samples."
        ),
        "inference_seed": int(seed),
        "inference_resamples": int(n_resamples),
        "random_streams": "numpy_seedsequence_bootstrap_and_sign_flip_independent",
        "bootstrap_method": "studentized_percentile_t",
        "bootstrap_quantile_method": "linear",
    }
