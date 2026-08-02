from __future__ import annotations

import pandas as pd
import pytest

from cfeg.data.splits import make_cross_condition_split, make_cross_dataset_split


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


def test_cross_dataset_refuses_missing_source_validation():
    manifest = pd.DataFrame(
        [
            {"dataset_id": "wang", "subject_id": "sub001", "electrode_type": "wet"},
            {"dataset_id": "beta", "subject_id": "sub001", "electrode_type": "wet"},
        ]
    )
    with pytest.raises(ValueError, match="val=0"):
        make_cross_dataset_split(manifest, ["wang"], ["beta"])
