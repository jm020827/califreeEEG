"""Pure, fail-closed MAMEM-I recorded-event parser for the frozen v1 protocol.

DIN rows 2 and 4 contain timestamps and one-based samples, respectively. Rows 1
and 3 are deliberately never inspected. Labels are inferred from continuous
within-group event periods, not trial order or legacy integer frequency keys.
The two support features describe recorded events, not verified physical jitter.
"""

from itertools import pairwise
from numbers import Integral, Real

import numpy as np

FREQUENCIES = np.array([6.66, 7.50, 8.57, 10.00, 12.00], dtype=np.float64)
FREQUENCIES.setflags(write=False)
_SPACINGS = np.abs(FREQUENCIES[:, None] - FREQUENCIES[None, :])
np.fill_diagonal(_SPACINGS, np.inf)
_WIDTHS = _SPACINGS.min(axis=1) / 4.0
_LOWER = FREQUENCIES - _WIDTHS
_UPPER = FREQUENCIES + _WIDTHS
_TAU = 1000.0 / 60.0
_ZERO = 1e-9


def _require(condition: bool, reason: str) -> None:
    if not condition:
        raise ValueError(reason)


def _scalar(cell: object) -> Real:
    """Accept a numeric scalar or a one-element numeric MATLAB cell payload."""
    if isinstance(cell, np.ndarray):
        _require(cell.size == 1 and cell.dtype.kind in "iuf", "invalid_numeric_cell")
        cell = cell.item()
    _require(
        isinstance(cell, Real) and not isinstance(cell, (bool, np.bool_)), "invalid_numeric_cell"
    )
    try:
        finite = np.isfinite(float(cell))
    except (OverflowError, TypeError, ValueError):
        raise ValueError("invalid_numeric_cell") from None
    _require(bool(finite), "nonfinite_numeric_cell")
    return cell


def _frequency_label(frequency: float) -> int:
    """Closed, disjoint guardbands; deliberately no nearest-label fallback."""
    matches = np.flatnonzero((frequency >= _LOWER) & (frequency <= _UPPER))
    _require(len(matches) == 1, "frequency_outside_unique_guardband")
    return int(matches[0])


def marker_features(timestamps: object) -> np.ndarray:
    """Return frame-grid residual MAD (ms) and lag-1 Pearson correlation.

    For each positive interval d, r=d-tau*rint(d/tau), tau=1000/60 ms.
    Residuals with abs(r)<1e-9 become zero before median centering. MAD is
    median(abs(centered residual)); lag-1 is zero if either shifted vector's
    population standard deviation is below 1e-9. No absolute time, mean period,
    frame-count sequence, event count, or trial order is returned as metadata.
    """
    try:
        values = np.asarray(timestamps)
    except (TypeError, ValueError, OverflowError):
        raise ValueError("invalid_timestamps") from None
    _require(
        values.ndim == 1 and values.size >= 4 and values.dtype.kind in "iuf", "invalid_timestamps"
    )
    with np.errstate(over="ignore", invalid="ignore"):
        values = values.astype(np.float64)
        delta = np.diff(values)
    _require(bool(np.isfinite(values).all() and np.isfinite(delta).all()), "nonfinite_timestamps")
    _require(bool((delta > 0).all()), "nonincreasing_timestamps")
    with np.errstate(over="ignore", invalid="ignore"):
        residual = delta - _TAU * np.rint(delta / _TAU)
    _require(bool(np.isfinite(residual).all()), "invalid_marker_features")
    residual[np.abs(residual) < _ZERO] = 0.0
    centered = residual - np.median(residual)
    mad = float(np.median(np.abs(centered)))
    a, b = centered[:-1], centered[1:]
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        scales = np.array([np.std(a), np.std(b)])
        _require(bool(np.isfinite(scales).all()), "invalid_marker_features")
        correlation = 0.0 if scales.min() < _ZERO else float(np.corrcoef(a, b)[0, 1])
    result = np.array([mad, correlation], dtype=np.float64)
    _require(bool(np.isfinite(result).all()), "invalid_marker_features")
    return result


def parse_main_trials(
    din: np.ndarray, total_samples: int, include_metadata: bool = True
) -> list[dict]:
    """Validate all 23 groups and return the 15 main trials in sample order.

    Python windows are [first_sample-1+250, first_sample-1+750). The last
    recorded sample is inclusive, so end0<=last_sample is the containment test.
    All groups, including adaptation, must be complete; nothing is dropped.
    Metadata is computed only for returned support trials and is never called
    when include_metadata=False. Query DIN is for inferred labels, not prediction.
    """
    _require(
        isinstance(total_samples, Integral)
        and not isinstance(total_samples, (bool, np.bool_))
        and 0 < total_samples <= np.iinfo(np.int64).max,
        "invalid_total_samples",
    )
    _require(isinstance(include_metadata, (bool, np.bool_)), "invalid_metadata_flag")
    _require(
        isinstance(din, np.ndarray)
        and din.dtype == np.dtype(object)
        and din.ndim == 2
        and din.shape[0] == 4
        and 0 < din.shape[1] <= 10000,
        "invalid_din_shape_or_dtype",
    )
    total_samples = int(total_samples)

    # Do not coerce/copy all DIN cells: descriptor payloads are out of scope.
    timestamps = np.array([float(_scalar(cell)) for cell in din[1, :]], dtype=np.float64)
    sample_values = []
    for cell in din[3, :]:
        value = _scalar(cell)
        sample = int(value)
        _require(value == sample and 1 <= sample <= total_samples, "invalid_sample_index")
        sample_values.append(sample)
    samples = np.array(sample_values, dtype=np.int64)
    with np.errstate(over="ignore", invalid="ignore"):
        deltas = np.diff(timestamps)
    _require(bool(np.isfinite(deltas).all() and (deltas > 0).all()), "nonincreasing_timestamps")
    _require(bool((np.diff(samples) > 0).all()), "nonincreasing_samples")

    boundaries = np.r_[0, np.flatnonzero(deltas > 2000.0) + 1, timestamps.size]
    _require(boundaries.size == 24, "expected_23_groups")
    records = []
    selected_timestamps = []
    for group_index, (left, right) in enumerate(pairwise(boundaries)):
        group_times = timestamps[left:right]
        group_samples = samples[left:right]
        _require(group_times.size >= 2, "incomplete_group")
        frequency = 1000.0 / (2.0 * float(np.diff(group_times).mean()))
        label = _frequency_label(frequency)
        first_sample = int(group_samples[0])
        start0 = first_sample - 1 + 250
        end0 = start0 + 500
        trial_end0 = first_sample - 1 + 1250
        _require(
            end0 <= int(group_samples[-1]) and end0 <= total_samples, "incomplete_analysis_window"
        )
        _require(trial_end0 <= total_samples, "incomplete_trial_cost_window")
        inside = ((group_samples - 1) >= start0) & ((group_samples - 1) < end0)
        event_count = int(inside.sum())
        _require(event_count >= 4, "insufficient_window_events")
        records.append(
            {
                "group_index": group_index,
                "label": label,
                "start0": start0,
                "end0": end0,
                "event_count": event_count,
                "metadata": None,
                "trial_end0": trial_end0,
            }
        )
        selected_timestamps.append(group_times[inside])

    main = records[8:23]
    labels = np.array([record["label"] for record in main], dtype=np.int64)
    _require(bool((np.bincount(labels, minlength=5) == 3).all()), "main_class_coverage_mismatch")
    blocks = labels.reshape(5, 3)
    _require(bool((blocks == blocks[:, :1]).all()), "main_class_blocks_mismatch")
    if include_metadata:
        for record, times in zip(main, selected_timestamps[8:23]):
            record["metadata"] = marker_features(times)
    return main
