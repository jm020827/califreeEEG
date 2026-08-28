from __future__ import annotations

from pathlib import Path

import numpy as np
import yaml

from cfeg.data.prepare_wearable import (
    _impedance_electrode_index,
    _impedance_for,
    _validate_required_impedance,
    _wearable_data,
)


def test_wearable_axis_detection_accepts_matlab_and_reversed_storage():
    expected = np.zeros((8, 710, 2, 10, 12), dtype=np.float32)
    assert _wearable_data({"data": expected}).shape == expected.shape
    reversed_data = np.moveaxis(expected, range(5), tuple(reversed(range(5))))
    assert _wearable_data({"data": reversed_data}).shape == expected.shape


def test_wearable_impedance_selects_block_electrode_subject():
    impedance = np.arange(8 * 10 * 2 * 102, dtype=np.float32).reshape(8, 10, 2, 102)
    selected = _impedance_for(impedance, subject=4, electrode=1, block=3)
    np.testing.assert_array_equal(selected, impedance[:, 3, 1, 4])


def test_wearable_eeg_and_impedance_electrode_orders_are_opposite():
    assert _impedance_electrode_index("dry", ["wet", "dry"]) == 1
    assert _impedance_electrode_index("wet", ["wet", "dry"]) == 0


def test_wearable_target_axis_matches_official_stimulation_order():
    with Path("configs/data/wearable.yaml").open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle)

    assert config["class_frequencies"] == [
        9.25,
        11.25,
        13.25,
        9.75,
        11.75,
        13.75,
        10.25,
        12.25,
        14.25,
        10.75,
        12.75,
        14.75,
    ]
    assert config["class_phases"] == [
        0.0,
        0.0,
        0.0,
        np.pi / 2,
        np.pi / 2,
        np.pi / 2,
        np.pi,
        np.pi,
        np.pi,
        3 * np.pi / 2,
        3 * np.pi / 2,
        3 * np.pi / 2,
    ]


def test_required_wearable_impedance_fails_fast_when_missing() -> None:
    with np.testing.assert_raises_regex(ValueError, "requires impedance metadata"):
        _validate_required_impedance(None, {"expected": {"has_impedance": True}})
