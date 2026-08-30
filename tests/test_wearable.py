from __future__ import annotations

from pathlib import Path

import numpy as np
import yaml
from scipy.io import savemat

from cfeg.data.prepare_wearable import (
    _condition_period,
    _impedance_electrode_index,
    _impedance_for,
    _load_headband_orders,
    _validate_headband_orders,
    _validate_impedance_signature,
    _validate_required_impedance,
    _validate_wearable_config,
    _wearable_data,
)
from cfeg.data.preprocess import CanonicalChannelMap, PreprocessConfig


def test_wearable_axis_detection_accepts_matlab_and_reversed_storage():
    expected = np.zeros((8, 710, 2, 10, 12), dtype=np.float32)
    assert _wearable_data({"data": expected}).shape == expected.shape
    reversed_data = np.moveaxis(expected, range(5), tuple(reversed(range(5))))
    assert _wearable_data({"data": reversed_data}).shape == expected.shape


def test_wearable_impedance_selects_block_electrode_subject():
    impedance = np.arange(8 * 10 * 2 * 102, dtype=np.float32).reshape(8, 10, 2, 102)
    selected = _impedance_for(impedance, subject=4, electrode=1, block=3)
    np.testing.assert_array_equal(selected, impedance[:, 3, 1, 4])


def test_wearable_impedance_axis_uses_empirically_verified_dry_wet_order():
    assert _impedance_electrode_index("dry", ["dry", "wet"]) == 0
    assert _impedance_electrode_index("wet", ["dry", "wet"]) == 1


def test_wearable_impedance_numeric_signature_rejects_reversed_mapping():
    impedance = np.empty((8, 10, 2, 102), dtype=np.float32)
    impedance[:, :, 0, :] = 261.6749
    impedance[:, :, 1, :] = 19.6317
    config = {
        "expected": {
            "impedance_mean_kohm": {"dry": 261.6749, "wet": 19.6317},
            "impedance_mean_tolerance_kohm": 0.1,
        }
    }

    signature = _validate_impedance_signature(impedance, ["dry", "wet"], config)
    assert signature["axis_labels"] == ["dry", "wet"]
    with np.testing.assert_raises_regex(ValueError, "empirically verified"):
        _validate_impedance_signature(impedance, ["wet", "dry"], config)


def test_required_impedance_signature_cannot_omit_expected_condition():
    impedance = np.ones((8, 10, 2, 102), dtype=np.float32)
    config = {
        "expected": {
            "has_impedance": True,
            "impedance_mean_kohm": {"dry": 1.0},
            "impedance_mean_tolerance_kohm": 0.1,
        }
    }

    with np.testing.assert_raises_regex(ValueError, "must define exactly"):
        _validate_impedance_signature(impedance, ["dry", "wet"], config)


def test_wearable_impedance_numeric_signature_rejects_nonfinite_axis():
    impedance = np.empty((8, 10, 2, 102), dtype=np.float32)
    impedance[:, :, 0, :] = np.nan
    impedance[:, :, 1, :] = 19.6317
    config = {
        "expected": {
            "impedance_mean_kohm": {"dry": 261.6749, "wet": 19.6317},
            "impedance_mean_tolerance_kohm": 0.1,
        }
    }

    with np.testing.assert_raises_regex(ValueError, "not finite"):
        _validate_impedance_signature(impedance, ["dry", "wet"], config)


def test_wearable_impedance_preserves_missing_channel_position():
    impedance = np.ones((8, 10, 2, 102), dtype=np.float32)
    impedance[3, 2, 0, 4] = np.nan

    selected = _impedance_for(impedance, subject=4, electrode=0, block=2)

    assert selected.shape == (8,)
    assert np.isnan(selected[3])


def test_wearable_headband_order_parser_and_period(tmp_path):
    table = np.empty((103, 2), dtype=object)
    table[0] = ["", "Headband to wear first"]
    for subject in range(1, 103):
        table[subject] = [
            f"sub{subject}",
            "Dry" if subject <= 53 else "Wet",
        ]
    savemat(tmp_path / "Subjects_Information.mat", {"Subjects_Information": table})

    orders = _load_headband_orders(tmp_path)
    _validate_headband_orders(
        orders,
        {
            "expected": {
                "has_headband_order": True,
                "n_subjects": 102,
                "headband_order_counts": {"dry": 53, "wet": 49},
            }
        },
    )

    assert orders[1] == "dry"
    assert orders[102] == "wet"
    assert _condition_period("dry", "dry") == "first"
    assert _condition_period("dry", "wet") == "second"


def test_wearable_target_axis_matches_official_stimulation_order():
    with Path("configs/data/wearable.yaml").open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle)

    assert config["dataset_revision"] == "wearable_v3"
    assert config["impedance_electrode_types"] == ["dry", "wet"]

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

    _validate_wearable_config(
        config,
        CanonicalChannelMap.from_yaml(),
        config["channel_names"],
        config["class_frequencies"],
        config["class_phases"],
        config["electrode_types"],
        config["impedance_electrode_types"],
        PreprocessConfig.from_dict(config["preprocess"]).c_max,
    )


def test_wearable_asset_registry_enforces_v3_manifest_contract():
    with Path("configs/assets.yaml").open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle)

    wearable = config["datasets"]["wearable"]
    assert wearable["dataset_revision"] == "wearable_v3"
    assert wearable["required_manifest_columns"] == [
        "query_signal_std",
        "query_signal_std_by_channel",
        "impedance_kohm_by_channel",
        "headband_order",
        "condition_period",
    ]
    assert "required_manifest_columns" not in config["datasets"]["synthetic"]


def test_required_wearable_impedance_fails_fast_when_missing() -> None:
    with np.testing.assert_raises_regex(ValueError, "requires impedance metadata"):
        _validate_required_impedance(None, {"expected": {"has_impedance": True}})
