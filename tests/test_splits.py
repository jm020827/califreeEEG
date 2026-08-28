from __future__ import annotations

import pandas as pd
import pytest

from cfeg.data.splits import (
    make_cross_condition_split,
    make_cross_dataset_split,
    make_cross_subject_fold_split,
    make_joint_subject_condition_split,
)
from cfeg.train_loop import _resolve_split_seed


def _manifest():
    rows = []
    for dataset in ("wang", "beta"):
        for subject in ("sub001", "sub002", "sub003"):
            for electrode in ("dry", "wet"):
                rows.append(
                    {
                        "dataset_id": dataset,
                        "subject_id": subject,
                        "electrode_type": electrode,
                    }
                )
    return pd.DataFrame(rows)


def test_cross_dataset_has_source_validation_and_untouched_target():
    manifest = _manifest()
    split = make_cross_dataset_split(
        manifest, ["wang"], ["beta"], seed=3, val_ratio=0.34
    )
    assert set(manifest.iloc[split.train].dataset_id) == {"wang"}
    assert set(manifest.iloc[split.val].dataset_id) == {"wang"}
    assert set(manifest.iloc[split.test].dataset_id) == {"beta"}
    train_subjects = set(manifest.iloc[split.train].subject_id)
    val_subjects = set(manifest.iloc[split.val].subject_id)
    assert train_subjects.isdisjoint(val_subjects)


def test_cross_condition_separates_wet_and_dry():
    manifest = _manifest()
    split = make_cross_condition_split(
        manifest,
        {"dataset_id": "wang", "electrode_type": "dry"},
        {"dataset_id": "wang", "electrode_type": "wet"},
    )
    assert set(manifest.iloc[split.train].electrode_type) == {"dry"}
    assert set(manifest.iloc[split.test].electrode_type) == {"wet"}


def test_joint_subject_condition_separates_subjects_and_electrodes():
    manifest = _manifest()
    split = make_joint_subject_condition_split(
        manifest,
        {"dataset_id": "wang", "electrode_type": "dry"},
        {"dataset_id": "wang", "electrode_type": "wet"},
        seed=3,
        val_ratio=0.34,
        test_ratio=0.34,
    )

    assert set(manifest.iloc[split.train].electrode_type) == {"dry"}
    assert set(manifest.iloc[split.val].electrode_type) == {"dry"}
    assert set(manifest.iloc[split.test].electrode_type) == {"wet"}
    train_subjects = set(manifest.iloc[split.train].subject_id)
    val_subjects = set(manifest.iloc[split.val].subject_id)
    test_subjects = set(manifest.iloc[split.test].subject_id)
    assert train_subjects.isdisjoint(val_subjects)
    assert train_subjects.isdisjoint(test_subjects)
    assert val_subjects.isdisjoint(test_subjects)


def test_joint_subject_condition_requires_paired_conditions():
    manifest = _manifest()
    manifest = manifest.loc[
        ~(
            manifest["subject_id"].eq("sub003")
            & manifest["electrode_type"].eq("wet")
        )
    ].reset_index(drop=True)

    with pytest.raises(ValueError, match="at least three subjects"):
        make_joint_subject_condition_split(
            manifest,
            {"dataset_id": "wang", "electrode_type": "dry"},
            {"dataset_id": "wang", "electrode_type": "wet"},
        )


def test_cross_dataset_refuses_missing_source_validation():
    manifest = pd.DataFrame(
        [
            {"dataset_id": "wang", "subject_id": "sub001", "electrode_type": "wet"},
            {"dataset_id": "beta", "subject_id": "sub001", "electrode_type": "wet"},
        ]
    )
    with pytest.raises(ValueError, match="val=0"):
        make_cross_dataset_split(manifest, ["wang"], ["beta"])


def test_split_seed_is_independent_of_training_seed() -> None:
    first = {"seed": 1, "data": {"split_seed": 73}}
    second = {"seed": 999, "data": {"split_seed": 73}}

    assert _resolve_split_seed(first) == _resolve_split_seed(second) == 73


def test_cross_subject_folds_cover_each_subject_exactly_once() -> None:
    manifest = pd.DataFrame(
        {
            "dataset_id": ["wearable"] * 20,
            "subject_id": [f"s{index:02d}" for index in range(10) for _ in range(2)],
        }
    )
    test_subject_counts: dict[str, int] = {}
    for fold_index in range(5):
        split = make_cross_subject_fold_split(
            manifest,
            seed=17,
            n_folds=5,
            fold_index=fold_index,
            val_ratio=0.2,
        )
        train_subjects = set(manifest.iloc[split.train]["subject_id"])
        val_subjects = set(manifest.iloc[split.val]["subject_id"])
        test_subjects = set(manifest.iloc[split.test]["subject_id"])
        assert train_subjects.isdisjoint(val_subjects | test_subjects)
        assert val_subjects.isdisjoint(test_subjects)
        for subject in test_subjects:
            test_subject_counts[subject] = test_subject_counts.get(subject, 0) + 1

    assert test_subject_counts == {f"s{index:02d}": 1 for index in range(10)}
