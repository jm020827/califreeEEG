from __future__ import annotations

import hashlib
import json
import shutil

import numpy as np
import pandas as pd
import pytest

from cfeg import prediction
from cfeg.analysis.provenance import PRIMARY_ANALYSIS_MANIFEST_COLUMNS
from cfeg.prediction import (
    OOD_REQUIRED_COLUMNS,
    PRIMARY_CONTRACT_COLUMNS,
    load_verified_prediction_bundle,
    run_prediction,
)

TEST_SOURCE_CONTRACT = {
    "source_commit_sha": "a" * 40,
    "source_dirty": False,
    "source_tree_sha256": "7" * 64,
}


def _context(checkpoint_path, sample_ids: list[str]) -> dict:
    manifest = pd.DataFrame({"sample_id": sample_ids})
    for column in PRIMARY_ANALYSIS_MANIFEST_COLUMNS:
        if column not in manifest:
            manifest[column] = 0
    manifest["dataset_id"] = "wearable"
    manifest["subject_id"] = "sub001"
    manifest["electrode_type"] = "dry"
    return {
        "checkpoint_path": checkpoint_path,
        "checkpoint": {
            "checkpoint_role": "source_validation_best",
            "selection_split": "val",
            "selection_metric": "accuracy",
            "class_map": {
                "0": {"stimulus_frequency_hz": 9.0},
                "1": {"stimulus_frequency_hz": 10.0},
            },
            "config": {
                "seed": 42,
                "data": {"split_seed": 73, "n_folds": 5, "fold_index": 0},
                "protocol": {
                    "metadata_contract_version": "0.4-dev",
                    "primary_ablation_role": "A0_eeg_only",
                    "primary_fairness_hash": "f" * 64,
                },
                "model": {
                    "conditioning": {"architecture": "physical_hybrid_v1"},
                    "condition_encoder": {"external_metadata_mode": "null"},
                },
                "runtime_contract": {
                    "parameter_schema_sha256": "1" * 64,
                    "initial_trainable_state_sha256": "2" * 64,
                    "split_assignment_sha256": "3" * 64,
                    "vocabulary_sha256": "4" * 64,
                    "asset_provenance_sha256": "5" * 64,
                    "execution_phase": "confirmatory_training",
                    "analysis_plan_sha256": "a" * 64,
                    "cohort_roles_sha256": "b" * 64,
                    "cohort_sha256": "d" * 64,
                    "outer_test_access_during_training": False,
                    "source_commit_sha": "a" * 40,
                    "source_dirty": False,
                    "source_tree_sha256": "7" * 64,
                    "environment_sha256": "8" * 64,
                },
            },
        },
        "manifest": manifest,
        "dataset": list(sample_ids),
        "model": object(),
        "device": "cpu",
    }


def _assignment_hash(rows: list[tuple[str, str]]) -> str:
    assignments = [{"sample_id": sample_id, "split": split} for sample_id, split in rows]
    assignments.sort(key=lambda row: row["sample_id"])
    payload = json.dumps(assignments, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def test_prediction_writes_paired_ood_schema_from_checkpoint_test_split(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.setattr(prediction, "_source_revision_contract", lambda: TEST_SOURCE_CONTRACT)
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
    context["checkpoint"]["config"]["runtime_contract"]["split_assignment_sha256"] = (
        _assignment_hash([("train-1", "train"), ("test-1", "test"), ("test-2", "test")])
    )
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
    assert set(PRIMARY_CONTRACT_COLUMNS) <= set(frame.columns)
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
    assert provenance["primary_contract"]["primary_ablation_role"] == "A0_eeg_only"
    assert provenance["prediction_csv_sha256"]
    assert provenance["split_manifest_sha256"]
    assert result["provenance_json"].endswith("predictions_provenance.json")

    verified = load_verified_prediction_bundle(output)
    assert verified.frame["prediction"].tolist() == [1, 0]

    staging = tmp_path / ".predictions.contract.staging"
    staging.mkdir()
    staged_csv = staging / "predictions.csv"
    staged_provenance = staging / "predictions_provenance.json"
    shutil.copy2(output, staged_csv)
    shutil.copy2(tmp_path / "predictions_provenance.json", staged_provenance)
    staged = load_verified_prediction_bundle(
        staged_csv,
        expected_recorded_csv_path=output,
    )
    assert staged.frame["prediction"].tolist() == [1, 0]
    with pytest.raises(ValueError, match="different CSV path"):
        load_verified_prediction_bundle(
            staged_csv,
            expected_recorded_csv_path=tmp_path / "other.csv",
        )

    frame.loc[0, "prediction"] = 0
    frame.to_csv(output, index=False)
    with pytest.raises(ValueError, match="changed after provenance"):
        load_verified_prediction_bundle(output)


def test_prediction_defaults_to_safe_test_selection_and_requires_split_manifest(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.setattr(prediction, "_source_revision_contract", lambda: TEST_SOURCE_CONTRACT)
    checkpoint = tmp_path / "best.pt"
    checkpoint.write_bytes(b"checkpoint fixture")
    context = _context(checkpoint, ["sample-1"])
    monkeypatch.setattr(prediction, "load_evaluation_context", lambda *_: context)

    with pytest.raises(FileNotFoundError, match="split manifest is missing"):
        run_prediction(checkpoint, tmp_path / "processed", tmp_path / "predictions.csv")


def test_development_val_prediction_binds_and_verifies_split_manifest(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.setattr(prediction, "_source_revision_contract", lambda: TEST_SOURCE_CONTRACT)
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    checkpoint = run_dir / "best.pt"
    checkpoint.write_bytes(b"checkpoint fixture")
    pd.DataFrame(
        {"sample_id": ["train-1", "val-1"], "split": ["train", "val"]}
    ).to_csv(run_dir / "split.csv", index=False)
    context = _context(checkpoint, ["val-1", "train-1"])
    context["checkpoint"]["config"]["runtime_contract"]["execution_phase"] = "development"
    context["checkpoint"]["config"]["runtime_contract"]["split_assignment_sha256"] = (
        _assignment_hash([("train-1", "train"), ("val-1", "val")])
    )
    monkeypatch.setattr(prediction, "load_evaluation_context", lambda *_: context)
    monkeypatch.setattr(prediction, "_loader", lambda *_: "loader")
    monkeypatch.setattr(
        prediction,
        "collect_predictions",
        lambda *_: (
            np.asarray([0]),
            np.asarray([[2.0, 0.0]]),
            ["val-1"],
        ),
    )

    output = tmp_path / "val.csv"
    result = run_prediction(checkpoint, tmp_path / "processed", output, split="val")

    assert result["split_manifest_sha256"]
    assert result["split_manifest_path"] == str(run_dir / "split.csv")
    verified = load_verified_prediction_bundle(output)
    assert verified.frame["selection_split"].tolist() == ["val"]


def test_whole_manifest_prediction_requires_explicit_all_selection(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(prediction, "_source_revision_contract", lambda: TEST_SOURCE_CONTRACT)
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
