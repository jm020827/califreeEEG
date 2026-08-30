from __future__ import annotations

import hashlib
from pathlib import Path

import pandas as pd
import pytest

from cfeg.analysis.ood_coverage import compare_ood_coverage
from cfeg.analysis.provenance import (
    PRIMARY_ANALYSIS_MANIFEST_COLUMNS,
    PRIMARY_ANALYSIS_MANIFEST_SCHEMA,
    VerifiedPredictionBundle,
    primary_analysis_manifest_sha256,
)


def _predictions(first: list[int], second: list[int]) -> tuple[pd.DataFrame, pd.DataFrame]:
    sample_ids = [f"sample-{index}" for index in range(8)]
    labels = [0, 0, 1, 1, 0, 0, 1, 1]
    return (
        pd.DataFrame({"sample_id": sample_ids, "label": labels, "prediction": first}),
        pd.DataFrame({"sample_id": sample_ids, "label": labels, "prediction": second}),
    )


def _manifest() -> pd.DataFrame:
    manifest = pd.DataFrame(
        {
            "sample_id": [f"sample-{index}" for index in range(8)],
            "subject_id": ["s1"] * 4 + ["s2"] * 4,
            "electrode_type": ["dry"] * 4 + ["wet"] * 4,
            "window_duration_sec": [1.0] * 8,
            "label": [0, 0, 1, 1, 0, 0, 1, 1],
        }
    )
    for column in PRIMARY_ANALYSIS_MANIFEST_COLUMNS:
        if column not in manifest:
            manifest[column] = 0
    manifest["dataset_id"] = "wearable"
    return manifest


def _contracted(frame: pd.DataFrame, *, role: str, mode: str, checkpoint: str) -> pd.DataFrame:
    result = frame.copy()
    digest = hashlib.sha256()
    for sample_id in sorted(result["sample_id"].astype(str)):
        encoded = sample_id.encode("utf-8")
        digest.update(len(encoded).to_bytes(8, "big"))
        digest.update(encoded)
    constants = {
        "prediction_schema_version": "cfeg.predictions.v3",
        "scenario": "clean",
        "selection_split": "test",
        "sample_identity_sha256": digest.hexdigest(),
        "checkpoint_sha256": checkpoint,
        "execution_job_id": f"seed42-fold0-{role}",
        "execution_contract_sha256": "e" * 64,
        "execution_manifest_content_sha256": "1" * 64,
        "lockbox_reveal_receipt_sha256": "2" * 64,
        "planned_config_sha256": ("0" if role == "A0_eeg_only" else "a") * 64,
        "training_completion_sha256": ("c" if role == "A0_eeg_only" else "d") * 64,
        "primary_ablation_role": role,
        "primary_fairness_hash": "f" * 64,
        "metadata_contract_version": "0.4-dev",
        "conditioning_architecture": "physical_hybrid_v1",
        "external_metadata_mode": mode,
        "optimization_seed": 42,
        "split_seed": 73,
        "n_folds": 5,
        "fold_index": 0,
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
        "source_commit_sha": "c" * 40,
        "source_dirty": False,
        "source_tree_sha256": "7" * 64,
        "environment_sha256": "8" * 64,
        "inference_environment_sha256": "9" * 64,
        "checkpoint_role": "confirmatory_fixed_epoch",
        "checkpoint_selection_split": "none",
        "checkpoint_selection_metric": "fixed_epoch",
        "analysis_manifest_schema": PRIMARY_ANALYSIS_MANIFEST_SCHEMA,
        "analysis_manifest_sha256": primary_analysis_manifest_sha256(
            _manifest(), result["sample_id"].astype(str)
        ),
        "split_manifest_sha256": "6" * 64,
    }
    for name, value in constants.items():
        result[name] = value
    return result


def _verified(frame: pd.DataFrame) -> VerifiedPredictionBundle:
    return VerifiedPredictionBundle(frame=frame, artifact={}, path=Path("fixture.csv"))


def test_ood_coverage_counts_learned_and_forgotten_cells() -> None:
    baseline, candidate = _predictions(
        [0, 1, 1, 0, 0, 0, 1, 1],
        [0, 0, 1, 1, 0, 1, 1, 0],
    )
    cells, subjects, summary = compare_ood_coverage(
        _verified(baseline),
        _verified(candidate),
        _manifest(),
        cell_columns=["subject_id", "electrode_type", "window_duration_sec"],
        success_threshold=0.75,
        tags={"scenario": "clean"},
        require_primary_contract=False,
    )

    assert cells.set_index("subject_id")["transition"].to_dict() == {
        "s1": "learned",
        "s2": "forgotten",
    }
    assert summary["learned_cells"] == 1
    assert summary["forgotten_cells"] == 1
    assert summary["learned_minus_forgotten"] == 0
    assert summary["n_subjects"] == 2
    assert list(subjects["subject_id"]) == ["s1", "s2"]


def test_ood_coverage_requires_exactly_paired_sample_sets() -> None:
    baseline, candidate = _predictions([0] * 8, [0] * 8)
    candidate = candidate.iloc[:-1]
    with pytest.raises(ValueError, match="sample sets differ"):
        compare_ood_coverage(
            baseline,
            candidate,
            _manifest(),
            cell_columns=["subject_id"],
            success_threshold=0.5,
            require_primary_contract=False,
        )


def test_ood_coverage_rejects_label_disagreement() -> None:
    baseline, candidate = _predictions([0] * 8, [0] * 8)
    candidate.loc[0, "label"] = 1
    with pytest.raises(ValueError, match="labels disagree"):
        compare_ood_coverage(
            baseline,
            candidate,
            _manifest(),
            cell_columns=["subject_id"],
            success_threshold=0.5,
            require_primary_contract=False,
        )


def test_subject_inference_keeps_same_subject_id_separate_across_datasets() -> None:
    labels = [0, 0, 1, 1] * 2
    sample_ids = [f"sample-{index}" for index in range(8)]
    baseline = pd.DataFrame(
        {
            "sample_id": sample_ids,
            "label": labels,
            "prediction": [0, 1, 1, 0, 0, 0, 1, 1],
        }
    )
    candidate = pd.DataFrame(
        {
            "sample_id": sample_ids,
            "label": labels,
            "prediction": [0, 0, 1, 1, 0, 1, 1, 0],
        }
    )
    manifest = pd.DataFrame(
        {
            "sample_id": sample_ids,
            "dataset_id": ["dataset-a"] * 4 + ["dataset-b"] * 4,
            "subject_id": ["s1"] * 8,
        }
    )

    _, subjects, summary = compare_ood_coverage(
        baseline,
        candidate,
        manifest,
        cell_columns=["dataset_id", "subject_id"],
        success_threshold=0.75,
        require_primary_contract=False,
    )

    assert list(subjects[["dataset_id", "subject_id"]].itertuples(index=False, name=None)) == [
        ("dataset-a", "s1"),
        ("dataset-b", "s1"),
    ]
    assert subjects["balanced_accuracy_delta"].tolist() == [0.5, -0.5]
    assert summary["n_subjects"] == 2


def test_incomplete_label_cells_are_excluded_from_coverage_counts() -> None:
    baseline, candidate = _predictions(
        [0, 1, 1, 0, 0, 0, 1, 1],
        [0, 0, 1, 1, 0, 1, 1, 0],
    )
    manifest = _manifest()
    manifest.loc[0, "electrode_type"] = "partial"

    cells, _, summary = compare_ood_coverage(
        baseline,
        candidate,
        manifest,
        cell_columns=["subject_id", "electrode_type"],
        success_threshold=0.75,
        expected_n_labels=2,
        require_primary_contract=False,
    )

    assert (cells["transition"] == "insufficient").sum() == 1
    assert summary["insufficient_cells"] == 1
    assert summary["n_valid_cells"] == len(cells) - 1


def test_tags_cannot_overwrite_subject_identity() -> None:
    baseline, candidate = _predictions([0] * 8, [0] * 8)

    with pytest.raises(ValueError, match="may not overwrite"):
        compare_ood_coverage(
            baseline,
            candidate,
            _manifest(),
            cell_columns=["subject_id"],
            success_threshold=0.5,
            tags={"subject_id": "corrupted"},
            require_primary_contract=False,
        )


def test_primary_coverage_requires_matching_a0_a2_runtime_contract() -> None:
    baseline, candidate = _predictions([0] * 8, [0] * 8)
    baseline = _contracted(baseline, role="A0_eeg_only", mode="null", checkpoint="a" * 64)
    candidate = _contracted(
        candidate,
        role="A2_structured_condition_prompt",
        mode="observed",
        checkpoint="b" * 64,
    )

    _, _, summary = compare_ood_coverage(
        _verified(baseline),
        _verified(candidate),
        _manifest(),
        cell_columns=["subject_id"],
        success_threshold=0.5,
    )

    assert summary["primary_contract"]["baseline_role"] == "A0_eeg_only"
    assert summary["primary_contract"]["candidate_role"] == ("A2_structured_condition_prompt")


def test_primary_coverage_rejects_scenario_or_runtime_mismatch() -> None:
    baseline, candidate = _predictions([0] * 8, [0] * 8)
    baseline = _contracted(baseline, role="A0_eeg_only", mode="null", checkpoint="a" * 64)
    candidate = _contracted(
        candidate,
        role="A2_structured_condition_prompt",
        mode="observed",
        checkpoint="b" * 64,
    )
    candidate["scenario"] = "channel_drop_50"
    candidate["asset_provenance_sha256"] = "7" * 64

    with pytest.raises(ValueError, match="differ outside treatment access"):
        compare_ood_coverage(
            _verified(baseline),
            _verified(candidate),
            _manifest(),
            cell_columns=["subject_id"],
            success_threshold=0.5,
        )
