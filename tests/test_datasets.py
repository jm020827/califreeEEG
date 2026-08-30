from __future__ import annotations

import hashlib
import json
import os
import pickle

import numpy as np
import pandas as pd
import pytest

from cfeg.data.datasets import (
    EEGProcessedDataset,
    _contiguous_index_runs,
    _to_float_array,
    _to_int_array,
    _validate_audit_receipt,
    _validate_manifest_class_map,
    _validate_processed_revision,
    _validate_protocol_contract,
)
from cfeg.data.io_hdf5 import HDF5SampleReader, write_processed_hdf5


def _sha256(path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_manifest_class_map_rejects_stale_label_alignment(tmp_path):
    class_map = {"0": {"label": 0, "stimulus_frequency_hz": 8.0}}
    (tmp_path / "class_map.json").write_text(json.dumps(class_map), encoding="utf-8")
    manifest = pd.DataFrame({"label": [0, 0], "stimulus_frequency_hz": [8.6, 8.6]})

    with pytest.raises(ValueError, match="canonical frequency alignment"):
        _validate_manifest_class_map(tmp_path, manifest)


def test_manifest_class_map_accepts_canonical_alignment(tmp_path):
    class_map = {"0": {"label": 0, "stimulus_frequency_hz": 8.0}}
    (tmp_path / "class_map.json").write_text(json.dumps(class_map), encoding="utf-8")
    manifest = pd.DataFrame({"label": [0, 0], "stimulus_frequency_hz": [8.0, 8.0]})

    _validate_manifest_class_map(tmp_path, manifest)


def test_processed_revision_guard_rejects_stale_wearable_asset(tmp_path):
    (tmp_path / "asset_info.json").write_text(
        json.dumps({"dataset_id": "wearable", "dataset_revision": "wearable_v2"}),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="revision mismatch"):
        _validate_processed_revision(tmp_path, {"wearable": "wearable_v3"})


def test_processed_revision_guard_rejects_unexpected_asset_id(tmp_path):
    (tmp_path / "asset_info.json").write_text(
        json.dumps({"dataset_id": "typo", "dataset_revision": "wearable_v3"}),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="unexpected dataset_id"):
        _validate_processed_revision(tmp_path, {"wearable": "wearable_v3"})


def test_protocol_v04_rejects_missing_query_qc_columns(tmp_path):
    manifest = pd.DataFrame({"sample_id": ["x"]})

    with pytest.raises(ValueError, match="missing protocol/revision columns"):
        _validate_protocol_contract(
            tmp_path,
            manifest,
            {
                "query_qc_extractor_version": ("filtered_cropped_pre_zscore_channel_std_median_v1"),
                "external_continuous_schema": "impedance_mean_max_v1",
            },
            {
                "metadata_contract_version": "0.4-dev",
                "query_qc_extractor_version": ("filtered_cropped_pre_zscore_channel_std_median_v1"),
                "external_continuous_schema": "impedance_mean_max_v1",
            },
        )


def test_optional_channel_vector_restores_null_as_nan():
    restored = _to_float_array([1.0, None, 3.0], length=4)

    np.testing.assert_allclose(restored[[0, 2]], [1.0, 3.0])
    assert np.isnan(restored[1])
    assert np.isnan(restored[3])


def test_missing_channel_identity_is_not_synthesized_from_tensor_slot():
    restored = _to_int_array(None, length=4, mask=np.asarray([True, True, False, False]))

    assert restored.tolist() == [0, 0, 0, 0]


def test_audit_receipt_binds_exact_signals_file_even_if_mtime_is_restored(tmp_path):
    metadata_names = [
        "asset_info.json",
        "manifest.jsonl",
        "preprocess_config.yaml",
        "class_map.json",
    ]
    for name in metadata_names:
        (tmp_path / name).write_text(f"contents:{name}", encoding="utf-8")
    signals = tmp_path / "signals.h5"
    signals.write_bytes(b"original-signals")
    original_mtime_ns = signals.stat().st_mtime_ns
    receipt = {
        "status": "accepted",
        "dataset_revision": "wearable_v3",
        "n_rows": 1,
        "raw_alignment": {
            "status": "deep_verified_atol_1e-6",
            "signals_content_sha256": "0" * 64,
            "signals_file_sha256": _sha256(signals),
            "processed_metadata_sha256": {
                name: _sha256(tmp_path / name) for name in metadata_names
            },
        },
    }
    receipt_path = tmp_path / "wearable_v3_audit_receipt.json"
    receipt_path.write_text(json.dumps(receipt), encoding="utf-8")
    manifest = pd.DataFrame({"sample_id": ["sample"]})
    info = {"dataset_revision": "wearable_v3"}

    _validate_audit_receipt(tmp_path, manifest, info)
    signals.write_bytes(b"tampered-signals")
    assert signals.stat().st_size == len(b"original-signals")
    os.utime(signals, ns=(original_mtime_ns, original_mtime_ns))

    with pytest.raises(ValueError, match="signals.h5 changed"):
        _validate_audit_receipt(tmp_path, manifest, info)


def test_persistent_hdf5_reader_reuses_handle_and_drops_it_when_pickled(tmp_path):
    path = write_processed_hdf5(
        tmp_path,
        np.arange(12, dtype=np.float32).reshape(2, 2, 3),
        np.ones((2, 2), dtype=bool),
        np.asarray([0, 1], dtype=np.int64),
    )
    reader = HDF5SampleReader(persistent=True)

    first = reader.read(path, 0)
    handle = reader._handles[path.resolve()]
    second = reader.read(path, 1)

    assert reader._handles[path.resolve()] is handle
    assert first[2] == 0
    assert second[2] == 1
    restored = pickle.loads(pickle.dumps(reader))
    assert restored.persistent is True
    assert restored._handles == {}
    reader.close()
    assert not handle.id.valid


def test_persistent_hdf5_reader_reopens_after_pid_change(tmp_path, monkeypatch):
    path = write_processed_hdf5(
        tmp_path,
        np.arange(12, dtype=np.float32).reshape(2, 2, 3),
        np.ones((2, 2), dtype=bool),
        np.asarray([0, 1], dtype=np.int64),
    )
    reader = HDF5SampleReader(persistent=True)
    reader.read(path, 0)
    inherited = reader._handles[path.resolve()]
    monkeypatch.setattr("cfeg.data.io_hdf5.os.getpid", lambda: reader._pid + 1)

    _, _, label = reader.read(path, 1)

    assert not inherited.id.valid
    assert reader._handles[path.resolve()] is not inherited
    assert label == 1
    reader.close()


def test_persistent_hdf5_reader_keeps_one_handle_per_processed_root(tmp_path):
    paths = []
    for root_name, label in (("first", 0), ("second", 1)):
        path = write_processed_hdf5(
            tmp_path / root_name,
            np.full((1, 2, 3), label, dtype=np.float32),
            np.ones((1, 2), dtype=bool),
            np.asarray([label], dtype=np.int64),
        )
        paths.append(path)
    reader = HDF5SampleReader(persistent=True)

    assert [reader.read(path, 0)[2] for path in paths] == [0, 1]
    assert len(reader._handles) == 2
    reader.close()


def test_contiguous_index_runs_preserve_positions_and_split_gaps() -> None:
    runs = _contiguous_index_runs([(4, 1), (1, 2), (3, 5), (0, 6)])

    assert runs == [[(4, 1), (1, 2)], [(3, 5), (0, 6)]]


def test_partial_preload_does_not_read_unselected_hdf_rows(tmp_path) -> None:
    write_processed_hdf5(
        tmp_path,
        np.arange(24, dtype=np.float32).reshape(4, 2, 3),
        np.ones((4, 2), dtype=bool),
        np.asarray([0, 1, 0, 1], dtype=np.int64),
    )
    # This test exercises the cache boundary directly; metadata validation is
    # orthogonal and the entries are the only fields __getitem__ needs below.
    dataset = object.__new__(EEGProcessedDataset)
    dataset.entries = [
        (
            tmp_path,
            index,
            {
                "sample_id": f"s{index}",
                "dataset_id": "synthetic",
                "subject_id": f"sub{index}",
                "session_id": "session",
                "canonical_channel_ids": [1, 2],
                "sfreq_processed": 3.0,
            },
        )
        for index in range(4)
    ]
    dataset._preloaded_samples = {}
    dataset._sample_reader = HDF5SampleReader(persistent=False)

    dataset.preload_indices([0, 2])

    assert set(dataset._preloaded_samples) == {0, 2}
    assert dataset[0].y == 0
    assert dataset[1].y == 1
