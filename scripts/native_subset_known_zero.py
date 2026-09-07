"""Label-free, inference-only FULL-equivalence projection; no I/O or fitting.

The original study and its closed entrypoints are not modified. Callers must
provide unchanged, already rounded gains and native-score argmax predictions.
This module alone is not a human-outcome execution pipeline.
"""

import numpy as np

MODES = ("Q", "QM", "SHAM_REFIT", "M_STALE")
M_MODES = MODES[1:]


def _validate(rounded_gains, expert_predictions):
    if not isinstance(rounded_gains, np.ndarray) or rounded_gains.dtype != np.float64:
        raise ValueError("Gains must be a float64 ndarray")
    if rounded_gains.ndim != 2:
        raise ValueError("Gains must have shape [k, queries]")
    k, queries = rounded_gains.shape
    if k not in (3, 5) or queries < 1:
        raise ValueError("Only unpadded k3/k5 and nonempty queries are allowed")
    if not np.isfinite(rounded_gains).all():
        raise ValueError("Nonfinite gain, including a FULL-equivalent candidate")
    if not np.array_equal(rounded_gains, np.round(rounded_gains, 10)):
        raise ValueError("Use unchanged gains already rounded to 10 decimals")
    if (
        not isinstance(expert_predictions, np.ndarray)
        or not np.issubdtype(expert_predictions.dtype, np.integer)
        or expert_predictions.shape != (k + 1, queries)
    ):
        raise ValueError("Predictions must be integer ndarray [k+1, queries]")
    if np.any(expert_predictions < 0) or np.any(expert_predictions >= 12):
        raise ValueError("Prediction class must lie in the fixed 0..11 alphabet")


def project_gains(rounded_gains, expert_predictions):
    """Return a new gain array: FULL-equivalent omissions have exact gain zero."""
    _validate(rounded_gains, expert_predictions)
    equivalent = expert_predictions[1:] == expert_predictions[0]
    return np.where(equivalent, 0.0, rounded_gains)


def choose_known_zero(rounded_gains, expert_predictions):
    """FULL is action0; positive ties keep the earliest original omission."""
    projected = project_gains(rounded_gains, expert_predictions)
    candidates = np.column_stack([np.zeros(projected.shape[1]), projected.T])
    return candidates.argmax(axis=-1)


def _fallback(value, queries):
    if not isinstance(value, np.ndarray) or value.dtype != np.bool_ or value.shape != (queries,):
        raise ValueError("Fallback must be a bool ndarray [queries]")
    return value


def route_family(
    expert_predictions, gains_by_mode, fallback_by_mode, shuffled_gains, shuffled_fallbacks
):
    """Apply the same projection to matched Q and every M/control intervention.

    Gains/scalers/fallback masks and shuffle maps are supplied by the unchanged
    parent computation. At least one shuffle evaluation is required, including
    the old identity evaluation when no admissible map exists. Availability and
    per-map metric averaging belong to the future evaluation caller, not here.
    """
    if not isinstance(gains_by_mode, dict) or set(gains_by_mode) != set(MODES):
        raise ValueError("Require exactly Q/QM/SHAM_REFIT/M_STALE gains")
    if not isinstance(fallback_by_mode, dict) or set(fallback_by_mode) != set(M_MODES):
        raise ValueError("Require exactly QM/SHAM_REFIT/M_STALE fallback masks")
    if (
        not isinstance(shuffled_gains, (list, tuple))
        or not isinstance(shuffled_fallbacks, (list, tuple))
        or not shuffled_gains
        or len(shuffled_gains) != len(shuffled_fallbacks)
    ):
        raise ValueError("Require matching nonempty lists of shuffle gains and masks")
    q_action = choose_known_zero(gains_by_mode["Q"], expert_predictions)
    queries = len(q_action)
    actions = {"Q": q_action}
    for mode in M_MODES:
        action = choose_known_zero(gains_by_mode[mode], expert_predictions)
        fallback = _fallback(fallback_by_mode[mode], queries)
        actions[mode] = np.where(fallback, q_action, action)
    actions["M_MISSING"] = q_action.copy()
    shuffle_actions = []
    for gains, fallback in zip(shuffled_gains, shuffled_fallbacks):
        action = choose_known_zero(gains, expert_predictions)
        shuffle_actions.append(np.where(_fallback(fallback, queries), q_action, action))
    return {"actions": actions, "shuffle_actions": shuffle_actions}
