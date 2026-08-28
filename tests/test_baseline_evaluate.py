from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from cfeg.baselines.evaluate import (
    _prediction_metrics,
    _resolve_channel_ids,
    _select_eeg_channels,
    evaluate_frequency_baseline,
)
from cfeg.data.io_hdf5 import write_processed_hdf5


def test_select_eeg_channels_uses_mask_and_canonical_ids() -> None:
    x = np.arange(20, dtype=np.float32).reshape(4, 5)
    selected = _select_eeg_channels(
        x,
        np.asarray([True, True, False, True]),
        np.asarray([55, 61, 62, 63]),
        {61, 62, 63},
    )

    np.testing.assert_array_equal(selected, x[[1, 3]])


def test_select_eeg_channels_rejects_missing_requested_set() -> None:
    with pytest.raises(ValueError, match="no channels"):
        _select_eeg_channels(
            np.ones((2, 5)),
            np.ones(2, dtype=bool),
            np.asarray([1, 2]),
            {61, 62},
        )


def test_select_eeg_channels_strict_mode_rejects_partial_named_set() -> None:
    with pytest.raises(ValueError, match="strict channel set is incomplete"):
        _select_eeg_channels(
            np.ones((2, 5)),
            np.ones(2, dtype=bool),
            np.asarray([61, 62]),
            {61, 62, 63},
            strict=True,
        )


def test_prediction_metrics_use_balanced_accuracy() -> None:
    frame = pd.DataFrame({"label": [0, 0, 1, 1], "prediction": [0, 0, 1, 0]})
    metrics = _prediction_metrics(frame, n_classes=2, trial_time_sec=2.0)

    assert metrics["accuracy"] == 0.75
    assert metrics["balanced_accuracy"] == 0.75
    assert metrics["n_samples"] == 4


def test_frequency_baseline_uses_only_selected_test_sample_ids(tmp_path) -> None:
    processed = tmp_path / "processed"
    sfreq = 200.0
    time = np.arange(200) / sfreq
    labels = np.asarray([0, 0, 1, 1])
    signals = np.stack(
        [
            np.stack([np.sin(2 * np.pi * (10.0 + label * 2.0) * time)] * 2)
            for label in labels
        ]
    ).astype(np.float32)
    write_processed_hdf5(
        processed,
        signals,
        np.ones((4, 2), dtype=bool),
        labels,
    )
    manifest = pd.DataFrame(
        {
            "sample_id": ["train-0", "val-0", "test-0", "test-1"],
            "h5_index": np.arange(4),
            "dataset_id": ["synthetic"] * 4,
            "subject_id": ["s1", "s2", "s3", "s3"],
            "label": labels,
            "stimulus_frequency_hz": [10.0, 10.0, 12.0, 12.0],
            "sfreq_processed": [sfreq] * 4,
            "canonical_channel_ids": [[61, 62]] * 4,
        }
    )
    manifest.to_json(processed / "manifest.jsonl", orient="records", lines=True)
    (processed / "class_map.json").write_text(
        json.dumps(
            {
                "0": {"stimulus_frequency_hz": 10.0},
                "1": {"stimulus_frequency_hz": 12.0},
            }
        ),
        encoding="utf-8",
    )
    split_csv = tmp_path / "split.csv"
    pd.DataFrame(
        {
            "sample_id": ["train-0", "val-0", "test-0", "test-1"],
            "split": ["train", "val", "test", "test"],
        }
    ).to_csv(split_csv, index=False)

    predictions, subjects, summary = evaluate_frequency_baseline(
        processed,
        method="cca",
        selection_csv=split_csv,
        channel_set="all",
        n_harmonics=1,
    )

    assert predictions["sample_id"].tolist() == ["test-0", "test-1"]
    assert subjects["subject_id"].tolist() == ["s3"]
    assert summary["n_samples"] == 2
    assert summary["selection_split"] == "test"
    assert summary["sample_identity_sha256"] == _expected_sample_hash(
        ["test-0", "test-1"]
    )


def test_wearable_channel_set_contains_all_official_electrodes() -> None:
    ids = _resolve_channel_ids("wearable_8", "configs/channel_sets.yaml")

    assert ids is not None
    assert len(ids) == 8


def _expected_sample_hash(sample_ids: list[str]) -> str:
    import hashlib

    digest = hashlib.sha256()
    for sample_id in sorted(sample_ids):
        encoded = sample_id.encode()
        digest.update(len(encoded).to_bytes(8, "big"))
        digest.update(encoded)
    return digest.hexdigest()
