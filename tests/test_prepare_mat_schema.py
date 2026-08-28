from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import yaml

from cfeg.data.prepare_mat import (
    _trial_block_indices,
    _validate_known_subject_trials,
    _validate_source_schema,
)

OFFICIAL_WANG_CHANNEL_ORDER = [
    "Fp1", "Fpz", "Fp2", "AF3", "AF4", "F7", "F5", "F3", "F1", "Fz",
    "F2", "F4", "F6", "F8", "FT7", "FC5", "FC3", "FC1", "FCz", "FC2",
    "FC4", "FC6", "FT8", "T7", "C5", "C3", "C1", "Cz", "C2", "C4",
    "C6", "T8", "M1", "TP7", "CP5", "CP3", "CP1", "CPz", "CP2", "CP4",
    "CP6", "TP8", "M2", "P7", "P5", "P3", "P1", "Pz", "P2", "P4",
    "P6", "P8", "PO7", "PO5", "PO3", "POz", "PO4", "PO6", "PO8", "CB1",
    "O1", "Oz", "O2", "CB2",
]


def test_wang_config_uses_official_raw_channel_axis_order() -> None:
    with Path("configs/data/wang.yaml").open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle)

    assert config["channel_names"] == OFFICIAL_WANG_CHANNEL_ORDER
    assert config["source_schema"] == [64, 1500, 40, 6]


def test_dong2023_config_has_exact_40_class_schema_and_native_channels() -> None:
    with Path("configs/data/dong2023.yaml").open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle)

    assert config["source_schema"] == [8, 1250, 40, 4]
    assert config["channel_names"] == ["POz", "PO3", "PO4", "PO7", "PO8", "Oz", "O1", "O2"]
    np.testing.assert_allclose(config["class_frequencies"], np.arange(8.0, 16.0, 0.2))
    assert len(config["class_phases"]) == 40


def test_known_schema_refuses_silent_axis_inference() -> None:
    data = np.zeros((2, 10, 4, 3), dtype=np.float32)
    config = {"source_schema_variants": [[2, 10, 4, 3]]}

    assert _validate_source_schema(data, config, "example", Path("S1.mat")) == (
        2,
        10,
        4,
        3,
    )
    with pytest.raises(ValueError, match="Refusing heuristic axis/label inference"):
        _validate_source_schema(
            np.zeros((2, 10, 3, 4), dtype=np.float32),
            config,
            "example",
            Path("S1.mat"),
        )


def test_block_indices_follow_each_dataset_axis_order() -> None:
    wang_order = _trial_block_indices(
        np.empty((2, 10, 3, 4)), expected_channels=2, n_targets=3
    )
    beta_order = _trial_block_indices(
        np.empty((2, 10, 4, 3)), expected_channels=2, n_targets=3
    )

    np.testing.assert_array_equal(wang_order, np.tile(np.arange(4), 3))
    np.testing.assert_array_equal(beta_order, np.repeat(np.arange(4), 3))


def test_known_subject_requires_every_target_in_every_block() -> None:
    labels = np.repeat(np.arange(3), 4)
    _validate_known_subject_trials(
        labels,
        len(labels),
        n_targets=3,
        n_blocks=4,
        dataset_id="wang",
        file=Path("S1.mat"),
    )

    bad = labels.copy()
    bad[-1] = 0
    with pytest.raises(ValueError, match="label counts"):
        _validate_known_subject_trials(
            bad,
            len(bad),
            n_targets=3,
            n_blocks=4,
            dataset_id="wang",
            file=Path("S1.mat"),
        )
