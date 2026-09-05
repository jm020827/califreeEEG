from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import numpy as np
from scipy import stats


@dataclass(frozen=True)
class OneSidedMeanResult:
    n_participants: int
    mean: float
    standard_deviation_ddof1: float
    standard_error: float
    null_margin: float
    alpha: float
    lower_confidence_bound: float
    p_value_greater: float

    def contract(self) -> dict[str, float | int]:
        return asdict(self)


def analytic_one_sided_mean_t_power(
    *,
    n_participants: int,
    true_mean: float,
    null_margin: float,
    standard_deviation: float,
    alpha: float = 0.05,
) -> float:
    """Outcome-free design sensitivity for the frozen one-sided mean t test."""

    if n_participants < 2:
        raise ValueError("Power calculation requires at least two participants.")
    if not np.isfinite([true_mean, null_margin, standard_deviation, alpha]).all():
        raise ValueError("Power inputs must all be finite.")
    if standard_deviation <= 0.0:
        raise ValueError("standard_deviation must be positive.")
    if not 0.0 < alpha < 0.5:
        raise ValueError("alpha must lie within (0,0.5).")
    degrees_of_freedom = int(n_participants) - 1
    critical = float(stats.t.ppf(1.0 - alpha, df=degrees_of_freedom))
    noncentrality = (
        (float(true_mean) - float(null_margin))
        * math.sqrt(int(n_participants))
        / float(standard_deviation)
    )
    return float(stats.nct.sf(critical, degrees_of_freedom, noncentrality))


def one_sided_paired_mean_result(
    differences,
    *,
    null_margin: float,
    alpha: float = 0.05,
    expected_n: int = 20,
) -> OneSidedMeanResult:
    """Frozen participant-level one-sample t procedure for paired contrasts.

    There is deliberately no data-dependent normality test or method switch.
    A zero-variance vector is handled deterministically rather than divided by
    zero. Exact sign-flip and bootstrap analyses, if later implemented, are
    sensitivity reports and cannot replace this promotion procedure.
    """

    if (
        isinstance(expected_n, (bool, np.bool_))
        or not isinstance(expected_n, (int, np.integer))
        or int(expected_n) < 2
    ):
        raise ValueError("expected_n must be an integer of at least two participants.")
    if not np.isfinite([null_margin, alpha]).all():
        raise ValueError("null_margin and alpha must be finite.")
    values = np.asarray(differences, dtype=np.float64).reshape(-1)
    if len(values) != int(expected_n):
        raise ValueError(f"Expected exactly {expected_n} participant contrasts, got {len(values)}.")
    if not np.isfinite(values).all():
        raise ValueError("Participant contrasts must all be finite; no complete-case fallback.")
    if not 0.0 < alpha < 0.5:
        raise ValueError("alpha must lie within (0,0.5).")

    mean = float(values.mean())
    standard_deviation = float(values.std(ddof=1))
    if float(np.ptp(values)) == 0.0:
        standard_deviation = 0.0
    standard_error = standard_deviation / math.sqrt(len(values))
    if standard_error == 0.0:
        lower = mean
        p_value = 0.0 if mean > float(null_margin) else 1.0
    else:
        critical = float(stats.t.ppf(1.0 - alpha, df=len(values) - 1))
        lower = mean - critical * standard_error
        statistic = (mean - float(null_margin)) / standard_error
        p_value = float(stats.t.sf(statistic, df=len(values) - 1))
    return OneSidedMeanResult(
        n_participants=len(values),
        mean=mean,
        standard_deviation_ddof1=standard_deviation,
        standard_error=standard_error,
        null_margin=float(null_margin),
        alpha=float(alpha),
        lower_confidence_bound=float(lower),
        p_value_greater=p_value,
    )


def evaluate_query_reliability_promotion_gates(
    *,
    corrupted_q1_aug_minus_q0_aug,
    clean_q1_aug_minus_q0_aug,
    clean_q1_aug_minus_q0_clean,
    expected_n: int = 20,
    alpha: float = 0.05,
    practical_effect_min: float = 0.02,
    clean_noninferiority_margin: float = -0.01,
    deployment_clean_noninferiority_margin: float | None = None,
    positive_fraction_min: float = 0.60,
) -> dict:
    """Evaluate the draft three-test IUT plus its non-inferential consistency gate."""

    if deployment_clean_noninferiority_margin is None:
        deployment_clean_noninferiority_margin = clean_noninferiority_margin
    thresholds = np.asarray(
        [
            alpha,
            practical_effect_min,
            clean_noninferiority_margin,
            deployment_clean_noninferiority_margin,
            positive_fraction_min,
        ],
        dtype=np.float64,
    )
    if not np.isfinite(thresholds).all():
        raise ValueError("Promotion thresholds must all be finite.")
    if not 0.0 < practical_effect_min <= 1.0:
        raise ValueError("practical_effect_min must lie within (0,1].")
    if not -1.0 <= clean_noninferiority_margin < 0.0:
        raise ValueError("clean_noninferiority_margin must lie within [-1,0).")
    if not -1.0 <= deployment_clean_noninferiority_margin < 0.0:
        raise ValueError("deployment_clean_noninferiority_margin must lie within [-1,0).")
    if not 0.0 < positive_fraction_min <= 1.0:
        raise ValueError("positive_fraction_min must lie within (0,1].")
    corrupted = _bounded_balanced_accuracy_contrast(
        corrupted_q1_aug_minus_q0_aug, field="corrupted primary contrast"
    )
    clean_simple_values = _bounded_balanced_accuracy_contrast(
        clean_q1_aug_minus_q0_aug, field="clean simple contrast"
    )
    clean_deployment_values = _bounded_balanced_accuracy_contrast(
        clean_q1_aug_minus_q0_clean, field="clean deployment contrast"
    )
    primary = one_sided_paired_mean_result(
        corrupted, null_margin=0.0, alpha=alpha, expected_n=expected_n
    )
    clean_simple = one_sided_paired_mean_result(
        clean_simple_values,
        null_margin=clean_noninferiority_margin,
        alpha=alpha,
        expected_n=expected_n,
    )
    clean_deployment = one_sided_paired_mean_result(
        clean_deployment_values,
        null_margin=deployment_clean_noninferiority_margin,
        alpha=alpha,
        expected_n=expected_n,
    )
    positive_count = int((corrupted > 0.0).sum())
    required_positive = math.ceil(expected_n * positive_fraction_min)
    gates = {
        "corrupted_superiority": primary.lower_confidence_bound > 0.0,
        "practical_effect": primary.mean >= practical_effect_min,
        "positive_participant_consistency": positive_count >= required_positive,
        "clean_simple_noninferiority": (
            clean_simple.lower_confidence_bound > clean_noninferiority_margin
        ),
        "clean_deployment_noninferiority": (
            clean_deployment.lower_confidence_bound > deployment_clean_noninferiority_margin
        ),
    }
    return {
        "schema": "cfeg.query-reliability-promotion-gates.v1",
        "contract_status": "draft_provisional_not_frozen",
        "grants_outcome_execution_or_publication_authority": False,
        "statistical_unit": "participant",
        "inference": "fixed_one_sided_paired_mean_t_lcb_ddof1_no_method_switch",
        "primary": primary.contract(),
        "clean_simple": clean_simple.contract(),
        "clean_deployment": clean_deployment.contract(),
        "positive_participant_count": positive_count,
        "required_positive_participant_count": required_positive,
        "positive_fraction_is_inferential_test": False,
        "thresholds": {
            "alpha": float(alpha),
            "primary_inferential_null_margin": 0.0,
            "practical_effect_min": float(practical_effect_min),
            "clean_noninferiority_margin": float(clean_noninferiority_margin),
            "deployment_clean_noninferiority_margin": float(deployment_clean_noninferiority_margin),
            "positive_fraction_min": float(positive_fraction_min),
        },
        "gates": gates,
        "promotion_pass": all(gates.values()),
    }


def _bounded_balanced_accuracy_contrast(values, *, field: str) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64).reshape(-1)
    if not np.isfinite(array).all():
        raise ValueError(f"{field} must contain only finite values.")
    if ((array < -1.0) | (array > 1.0)).any():
        raise ValueError(f"{field} must lie within the balanced-accuracy range [-1,1].")
    return array
