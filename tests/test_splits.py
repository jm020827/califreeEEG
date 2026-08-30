from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from cfeg.data.splits import (
    make_confirmatory_lockbox_split,
    make_cross_condition_split,
    make_cross_dataset_split,
    make_cross_subject_fold_split,
    make_development_subject_fold_split,
    make_joint_subject_condition_split,
)
from cfeg.train_loop import (
    _resolve_split_seed,
    _resolve_training_selection,
    _source_revision_contract,
    _validate_primary_split_contract,
)


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
    split = make_cross_dataset_split(manifest, ["wang"], ["beta"], seed=3, val_ratio=0.34)
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
        ~(manifest["subject_id"].eq("sub003") & manifest["electrode_type"].eq("wet"))
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


def test_source_revision_contract_binds_commit_and_worktree() -> None:
    contract = _source_revision_contract()

    assert len(str(contract["source_commit_sha"])) in {40, 64}
    assert isinstance(contract["source_dirty"], bool)
    assert len(str(contract["source_tree_sha256"])) == 64


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


def test_development_subject_folds_rotate_validation_without_test() -> None:
    manifest = pd.DataFrame(
        {
            "dataset_id": ["wearable"] * 6,
            "subject_id": ["sub001", "sub001", "sub002", "sub002", "sub003", "sub003"],
        }
    )
    validation_counts: dict[str, int] = {}
    for fold_index in range(3):
        split = make_development_subject_fold_split(
            manifest,
            seed=42,
            n_folds=3,
            fold_index=fold_index,
        )
        train_subjects = set(manifest.iloc[split.train]["subject_id"])
        val_subjects = set(manifest.iloc[split.val]["subject_id"])
        assert len(train_subjects) == 2
        assert len(val_subjects) == 1
        assert train_subjects.isdisjoint(val_subjects)
        assert len(split.test) == 0
        subject = next(iter(val_subjects))
        validation_counts[subject] = validation_counts.get(subject, 0) + 1

    assert validation_counts == {"sub001": 1, "sub002": 1, "sub003": 1}


def test_development_subject_fold_requires_one_fold_per_subject() -> None:
    manifest = pd.DataFrame(
        {
            "dataset_id": ["wearable"] * 3,
            "subject_id": ["sub001", "sub002", "sub003"],
        }
    )
    with pytest.raises(ValueError, match="one validation fold per development subject"):
        make_development_subject_fold_split(
            manifest,
            seed=42,
            n_folds=2,
            fold_index=0,
        )


def test_confirmatory_lockbox_partitions_subjects_without_validation() -> None:
    manifest = pd.DataFrame(
        {
            "dataset_id": ["wearable"] * 8,
            "subject_id": [f"sub{subject:03d}" for subject in range(4) for _ in range(2)],
        }
    )
    split = make_confirmatory_lockbox_split(
        manifest,
        training_subject_ids=["sub000", "sub001"],
        lockbox_subject_ids=["sub002", "sub003"],
    )

    assert set(manifest.iloc[split.train]["subject_id"]) == {"sub000", "sub001"}
    assert set(manifest.iloc[split.test]["subject_id"]) == {"sub002", "sub003"}
    assert len(split.val) == 0


def test_fixed_epoch_blind_never_schedules_validation() -> None:
    policy, validation_epochs = _resolve_training_selection(
        {"train": {"checkpoint_selection": "fixed_epoch_blind"}},
        max_epochs=7,
    )

    assert policy == "fixed_epoch_blind"
    assert validation_epochs == set()


def test_fixed_epoch_blind_rejects_validation_milestones() -> None:
    with pytest.raises(ValueError, match="forbids train.validation_epochs"):
        _resolve_training_selection(
            {
                "train": {
                    "checkpoint_selection": "fixed_epoch_blind",
                    "validation_epochs": [7],
                }
            },
            max_epochs=7,
        )


def test_primary_split_contract_rejects_participant_leakage() -> None:
    manifest = pd.DataFrame(
        {
            "dataset_id": ["wearable"] * 12,
            "subject_id": [f"s{index:02d}" for index in range(6) for _ in range(2)],
        }
    )
    split = make_cross_subject_fold_split(
        manifest, seed=42, n_folds=3, fold_index=0, val_ratio=0.25
    )
    cfg = {
        "protocol": {
            "metadata_contract_version": "0.4-dev",
            "primary_ablation_role": "A0_eeg_only",
        },
        "data": {"split": "cross_subject_fold"},
    }
    _validate_primary_split_contract(cfg, manifest, split)
    split.test = np.append(split.test, split.train[0])

    with pytest.raises(ValueError, match="assign every eligible cohort row exactly once"):
        _validate_primary_split_contract(cfg, manifest, split)
