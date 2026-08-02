from __future__ import annotations

from pathlib import Path

import numpy as np

from cfeg.data.prepare_mat import (
    FREQUENCY_KEYS,
    _dedupe_files,
    _extract_numeric_vector,
    _labels_from_data_shape,
    _select_data_array,
    _strings_from_value,
)


def test_dedupe_files_skips_symlink_duplicate(tmp_path: Path):
    raw = tmp_path / "raw"
    raw.mkdir()
    original = raw / "S1.mat"
    original.write_bytes(b"placeholder")
    link_dir = raw / "moabb_links"
    link_dir.mkdir()
    link = link_dir / "S1.mat"
    link.symlink_to(original)

    unique, duplicates = _dedupe_files(sorted([original, link]))

    assert unique == [original]
    assert duplicates == [link]


def test_extract_channel_names_prefers_label_column_over_numeric_strings():
    chan = np.asarray(
        [
            ["1", "-17.926", "0.51499", "FP1"],
            ["2", "0", "0.50669", "FPZ"],
            ["3", "17.926", "0.51499", "FP2"],
        ],
        dtype=object,
    )

    assert _strings_from_value(chan, expected_channels=3) == ["FP1", "FPZ", "FP2"]


def test_extract_numeric_vector_from_nested_key():
    arrays = {"data.suppl_info.freqs": np.asarray([8.6, 8.8, 9.0])}

    assert _extract_numeric_vector(arrays, FREQUENCY_KEYS, expected_len=3) == [8.6, 8.8, 9.0]


def test_labels_follow_target_axis_before_block_axis():
    data = np.zeros((64, 750, 40, 4), dtype=np.float32)
    labels = _labels_from_data_shape(data, expected_channels=64, n_targets=40)
    np.testing.assert_array_equal(labels, np.repeat(np.arange(40), 4))


def test_labels_follow_block_before_target_axis():
    data = np.zeros((6, 40, 64, 750), dtype=np.float32)
    labels = _labels_from_data_shape(data, expected_channels=64, n_targets=40)
    np.testing.assert_array_equal(labels, np.tile(np.arange(40), 6))


def test_select_data_array_rejects_frequency_support_mat():
    arrays = {"freqs": np.arange(40, dtype=np.float32)[None, :]}

    try:
        _select_data_array(arrays, expected_channels=64)
    except KeyError as exc:
        assert "No numeric EEG data array" in str(exc)
    else:
        raise AssertionError("support MAT was incorrectly selected as EEG")
