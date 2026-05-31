from __future__ import annotations

from pathlib import Path

import numpy as np

from cfeg.data.prepare_mat import FREQUENCY_KEYS, _dedupe_files, _extract_numeric_vector, _strings_from_value


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
