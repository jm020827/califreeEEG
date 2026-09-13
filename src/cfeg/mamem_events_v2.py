"""MAMEM-I v2: recorded-event labels under the frozen MOABB convention.

This changes only v1's label decoder. It does not widen nominal guardbands,
infer labels from trial order, recover physical stimulus frequencies, or repair
unsupported keys. Nominal reference frequencies remain separate from class keys.
"""

from itertools import pairwise
from numbers import Integral

import numpy as np

from cfeg.mamem_events_v1 import FREQUENCIES, _require, _scalar, marker_features

__all__ = ["FREQUENCIES", "marker_features", "parse_main_trials"]

_KEY_TO_LABEL = {6: 0, 7: 1, 8: 2, 9: 3, 11: 4}


def _group_label(timestamps: np.ndarray) -> int:
    """Decode a validated group, without nominal-Hz or ordering fallbacks.

    MOABB mamem_event, retained blob eef0203f06c86f18f6db53f7cc67cbd4a6af685b,
    lines 76/87/89-96: floor the mean interval, then integer-divide 1000 by
    twice that integer. This is NOT floor(1000/(2*mean_interval)).
    """
    _require(timestamps.size >= 2, "incomplete_group")
    mean_interval = float(np.sum(np.diff(timestamps))) / (timestamps.size - 1)
    period_integer_ms = int(np.floor(mean_interval))
    _require(period_integer_ms >= 1, "nonpositive_integer_period")
    key = 1000 // (2 * period_integer_ms)
    _require(key in _KEY_TO_LABEL, "unsupported_frequency_key")
    return _KEY_TO_LABEL[key]


def parse_main_trials(
    din: np.ndarray, total_samples: int, include_metadata: bool = True
) -> list[dict]:
    """Return 15 main trials only after validating all 23 aligned groups.

    Only DIN rows 2/4 (timestamps/one-based samples) are inspected. A Python
    analysis window is [first_sample-1+250, first_sample-1+750); the final
    marker is inclusive. All v1 bounds, adaptation removal, three-per-class
    contiguous coverage, and support-only metadata rules are retained.
    Unknown class keys stop before any metadata extraction, even in adaptation.
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

    # Never coerce the full DIN array: descriptor cells are outside this API.
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
        label = _group_label(group_times)
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
