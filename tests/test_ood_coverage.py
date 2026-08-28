from __future__ import annotations

import pandas as pd
import pytest

from cfeg.analysis.ood_coverage import compare_ood_coverage


def _predictions(first: list[int], second: list[int]) -> tuple[pd.DataFrame, pd.DataFrame]:
    sample_ids = [f"sample-{index}" for index in range(8)]
    labels = [0, 0, 1, 1, 0, 0, 1, 1]
    return (
        pd.DataFrame({"sample_id": sample_ids, "label": labels, "prediction": first}),
        pd.DataFrame({"sample_id": sample_ids, "label": labels, "prediction": second}),
    )


def _manifest() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "sample_id": [f"sample-{index}" for index in range(8)],
            "subject_id": ["s1"] * 4 + ["s2"] * 4,
            "electrode_type": ["dry"] * 4 + ["wet"] * 4,
            "window_duration_sec": [1.0] * 8,
        }
    )


def test_ood_coverage_counts_learned_and_forgotten_cells() -> None:
    baseline, candidate = _predictions(
        [0, 1, 1, 0, 0, 0, 1, 1],
        [0, 0, 1, 1, 0, 1, 1, 0],
    )
    cells, subjects, summary = compare_ood_coverage(
        baseline,
        candidate,
        _manifest(),
        cell_columns=["subject_id", "electrode_type", "window_duration_sec"],
        success_threshold=0.75,
        tags={"scenario": "clean"},
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
        )
