"""Pure scalar oracle for the prospective known-zero inference rule.

No truth inputs, fitting, data loaders, production/core imports or execution
entrypoint. This implementation skips FULL-equivalent omissions while scanning
positive utilities; it does not call or reproduce vectorized gain projection.
"""

from __future__ import annotations

import numpy as np

PLAN_SHA256 = "9f3ed2309b9dc2efff81460a91c49e131722595ef1ff8de4b24f9291f66c9e55"
MODES = ("Q", "QM", "SHAM_REFIT", "M_STALE")
FALLBACK_MODES = ("QM", "SHAM_REFIT", "M_STALE")


def _predictions(expert_predictions):
    if (
        not isinstance(expert_predictions, np.ndarray)
        or expert_predictions.ndim != 2
        or not np.issubdtype(expert_predictions.dtype, np.integer)
        or expert_predictions.shape[0] not in (4, 6)
        or expert_predictions.shape[1] == 0
        or np.any(expert_predictions < 0)
        or np.any(expert_predictions > 11)
    ):
        raise ValueError("Predictions must be integer class IDs0..11 with shape[k+1,q], k3/5, q>0")
    return expert_predictions.shape[0] - 1, expert_predictions.shape[1]


def _gains(rounded_gains, shape):
    if (
        not isinstance(rounded_gains, np.ndarray)
        or rounded_gains.dtype != np.float64
        or rounded_gains.shape != shape
        or not np.isfinite(rounded_gains).all()
    ):
        raise ValueError("Gains must be finite float64 with shape[k,q]")
    # The caller must supply frozen predict_gain outputs, not unrounded values.
    # Overflow in this validation is rejection, never a replacement or cast.
    with np.errstate(over="ignore", invalid="ignore"):
        if not np.array_equal(rounded_gains, np.round(rounded_gains, 10)):
            raise ValueError("Gains must already be rounded to ten decimals")


def _fallback(values, queries):
    if not isinstance(values, np.ndarray) or values.dtype != np.bool_ or values.shape != (queries,):
        raise ValueError("Fallback must be a bool array with shape[q]")


def choose_known_zero_scalar(rounded_gains, expert_predictions):
    """Choose each query independently, ignoring FULL aliases and nonpositive gains."""
    omissions, queries = _predictions(expert_predictions)
    _gains(rounded_gains, (omissions, queries))
    actions = np.zeros(queries, dtype=np.int64)
    for query in range(queries):
        full_prediction = int(expert_predictions[0, query])
        best_gain, best_action = 0.0, 0
        for expert in range(1, omissions + 1):
            if int(expert_predictions[expert, query]) == full_prediction:
                continue
            gain = float(rounded_gains[expert - 1, query])
            if gain > best_gain:
                best_gain, best_action = gain, expert
        actions[query] = best_action
    return actions


def route_family_scalar(
    expert_predictions,
    gains_by_mode,
    fallback_by_mode,
    shuffled_gains,
    shuffled_fallbacks,
):
    """Apply one shared alias rule, followed by exact corrected-Q fallback."""
    omissions, queries = _predictions(expert_predictions)
    if not isinstance(gains_by_mode, dict) or set(gains_by_mode) != set(MODES):
        raise ValueError("Exactly Q/QM/SHAM_REFIT/M_STALE gains are required")
    if not isinstance(fallback_by_mode, dict) or set(fallback_by_mode) != set(FALLBACK_MODES):
        raise ValueError("Exactly QM/SHAM_REFIT/M_STALE fallback masks are required")
    if (
        not isinstance(shuffled_gains, (list, tuple))
        or not isinstance(shuffled_fallbacks, (list, tuple))
        or len(shuffled_gains) == 0
        or len(shuffled_gains) != len(shuffled_fallbacks)
    ):
        raise ValueError("At least one matched shuffle gain/fallback intervention is required")
    for mode in MODES:
        _gains(gains_by_mode[mode], (omissions, queries))
    for mode in FALLBACK_MODES:
        _fallback(fallback_by_mode[mode], queries)
    for gains, fallback in zip(shuffled_gains, shuffled_fallbacks):
        _gains(gains, (omissions, queries))
        _fallback(fallback, queries)

    actions = {"Q": choose_known_zero_scalar(gains_by_mode["Q"], expert_predictions)}
    for mode in FALLBACK_MODES:
        corrected = choose_known_zero_scalar(gains_by_mode[mode], expert_predictions)
        for query in range(queries):
            if fallback_by_mode[mode][query]:
                corrected[query] = actions["Q"][query]
        actions[mode] = corrected
    actions["M_MISSING"] = actions["Q"].copy()
    shuffle_actions = []
    for gains, fallback in zip(shuffled_gains, shuffled_fallbacks):
        corrected = choose_known_zero_scalar(gains, expert_predictions)
        for query in range(queries):
            if fallback[query]:
                corrected[query] = actions["Q"][query]
        shuffle_actions.append(corrected)
    return {"actions": actions, "shuffle_actions": shuffle_actions}
