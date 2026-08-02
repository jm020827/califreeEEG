from __future__ import annotations

import json
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

from cfeg.data.migrate_labels import migrate_processed_label_alignment
from cfeg.data.schema import REQUIRED_MANIFEST_COLUMNS, load_manifest, write_manifest


def _legacy_processed(root):
    rows = []
    for index, (label, frequency) in enumerate([(0, 8.2), (1, 8.0)]):
        row = {column: None for column in REQUIRED_MANIFEST_COLUMNS}
        row.update(
            {
                "sample_id": f"sample-{index}",
                "h5_index": index,
                "dataset_id": "beta",
                "subject_id": "sub001",
                "label": label,
                "stimulus_frequency_hz": frequency,
            }
        )
        rows.append(row)
    manifest = pd.DataFrame(rows, columns=REQUIRED_MANIFEST_COLUMNS)
    write_manifest(manifest, root)
    with h5py.File(root / "signals.h5", "w") as h5:
        h5.create_dataset("x", data=np.zeros((2, 2, 4), dtype=np.float32))
        h5.create_dataset("channel_mask", data=np.ones((2, 2), dtype=bool))
        h5.create_dataset("y", data=np.asarray([0, 1], dtype=np.int64))
    (root / "class_map.json").write_text(
        json.dumps(
            {
                "0": {"label": 0, "stimulus_frequency_hz": 8.2},
                "1": {"label": 1, "stimulus_frequency_hz": 8.0},
            }
        ),
        encoding="utf-8",
    )


def test_label_alignment_migration_is_dry_run_by_default(tmp_path):
    _legacy_processed(tmp_path)
    result = migrate_processed_label_alignment(tmp_path)
    assert result["status"] == "dry_run"
    with h5py.File(tmp_path / "signals.h5", "r") as h5:
        np.testing.assert_array_equal(h5["y"][:], [0, 1])


def test_label_alignment_migration_updates_metadata_and_h5_labels(tmp_path):
    _legacy_processed(tmp_path)
    result = migrate_processed_label_alignment(tmp_path, apply=True)
    assert result["status"] == "migrated"
    assert result["changed_labels"] == 2
    np.testing.assert_array_equal(load_manifest(tmp_path)["label"].to_numpy(), [1, 0])
    with h5py.File(tmp_path / "signals.h5", "r") as h5:
        np.testing.assert_array_equal(h5["y"][:], [1, 0])
    class_map = json.loads((tmp_path / "class_map.json").read_text(encoding="utf-8"))
    assert class_map["0"]["stimulus_frequency_hz"] == 8.0
    assert class_map["39"]["stimulus_frequency_hz"] == 15.8
    assert Path(str(result["backup_dir"])).is_dir()
