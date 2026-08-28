from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from cfeg import prediction
from cfeg.prediction import OOD_REQUIRED_COLUMNS, run_prediction


def _context(checkpoint_path, sample_ids: list[str]) -> dict:
    return {
        "checkpoint_path": checkpoint_path,
        "checkpoint": {
            "class_map": {
                "0": {"stimulus_frequency_hz": 9.0},
                "1": {"stimulus_frequency_hz": 10.0},
            }
        },
        "manifest": pd.DataFrame(
            {"sample_id": sample_ids, "dataset_id": ["wearable"] * len(sample_ids)}
        ),
        "dataset": list(sample_ids),
        "model": object(),
        "device": "cpu",
    }


def test_prediction_writes_paired_ood_schema_from_checkpoint_test_split(
    tmp_path, monkeypatch
) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    checkpoint = run_dir / "best.pt"
    checkpoint.write_bytes(b"checkpoint fixture")
    pd.DataFrame(
        {
            "sample_id": ["train-1", "test-1", "test-2"],
            "split": ["train", "test", "test"],
        }
    ).to_csv(run_dir / "split.csv", index=False)
    context = _context(checkpoint, ["test-2", "train-1", "test-1"])
    observed: dict[str, object] = {}

    monkeypatch.setattr(prediction, "load_evaluation_context", lambda *_: context)

    def fake_loader(_context, indices):
        observed["indices"] = indices.tolist()
        return "loader"

    monkeypatch.setattr(prediction, "_loader", fake_loader)
    monkeypatch.setattr(
        prediction,
        "collect_predictions",
        lambda *_: (
            np.asarray([1, 0]),
            np.asarray([[0.0, 3.0], [3.0, 0.0]]),
            ["test-2", "test-1"],
        ),
    )

    output = tmp_path / "predictions.csv"
    result = run_prediction(
        checkpoint, tmp_path / "processed", output, scenario="clean", split="test"
    )

    frame = pd.read_csv(output)
    provenance = json.loads((tmp_path / "predictions_provenance.json").read_text())
    assert observed["indices"] == [0, 2]
    assert set(OOD_REQUIRED_COLUMNS) <= set(frame.columns)
    assert "true_label" not in frame and "predicted_label" not in frame
    assert frame["sample_id"].tolist() == ["test-2", "test-1"]
    assert frame["label"].tolist() == [1, 0]
    assert frame["prediction"].tolist() == [1, 0]
    assert frame["scenario"].unique().tolist() == ["clean"]
    assert frame["selection_split"].unique().tolist() == ["test"]
    assert provenance["n_samples"] == 2
    assert provenance["selection_split"] == "test"
    assert provenance["sample_identity_sha256"] == frame["sample_identity_sha256"].iloc[0]
    assert provenance["checkpoint_sha256"] == frame["checkpoint_sha256"].iloc[0]
    assert provenance["prediction_csv_sha256"]
    assert provenance["split_manifest_sha256"]
    assert result["provenance_json"].endswith("predictions_provenance.json")


def test_prediction_defaults_to_safe_test_selection_and_requires_split_manifest(
    tmp_path, monkeypatch
) -> None:
    checkpoint = tmp_path / "best.pt"
    checkpoint.write_bytes(b"checkpoint fixture")
    context = _context(checkpoint, ["sample-1"])
    monkeypatch.setattr(prediction, "load_evaluation_context", lambda *_: context)

    with pytest.raises(FileNotFoundError, match="split manifest is missing"):
        run_prediction(checkpoint, tmp_path / "processed", tmp_path / "predictions.csv")


def test_whole_manifest_prediction_requires_explicit_all_selection(
    tmp_path, monkeypatch
) -> None:
    checkpoint = tmp_path / "best.pt"
    checkpoint.write_bytes(b"checkpoint fixture")
    context = _context(checkpoint, ["sample-1", "sample-2"])
    observed: dict[str, object] = {}
    monkeypatch.setattr(prediction, "load_evaluation_context", lambda *_: context)

    def fake_loader(_context, indices):
        observed["indices"] = indices.tolist()
        return "loader"

    monkeypatch.setattr(prediction, "_loader", fake_loader)
    monkeypatch.setattr(
        prediction,
        "collect_predictions",
        lambda *_: (
            np.asarray([0, 1]),
            np.asarray([[2.0, 0.0], [0.0, 2.0]]),
            ["sample-1", "sample-2"],
        ),
    )

    result = run_prediction(
        checkpoint,
        tmp_path / "processed",
        tmp_path / "all.csv",
        split="all",
        scenario="external_inference",
    )

    assert observed["indices"] == [0, 1]
    assert result["selection_split"] == "all"
    assert result["split_manifest_path"] is None
    assert result["split_manifest_sha256"] is None
