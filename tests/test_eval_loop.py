from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import torch

from cfeg.eval_loop import (
    _calibration_support_indices,
    _held_out_evaluation_indices,
    _relative_drop,
    _robustness_perturbation,
    _sample_id_identity_sha256,
    _scenarios,
    _subject_calibration_indices,
    _subject_calibration_partition,
)


def _condition(batch: int = 4, channels: int = 8):
    cond = {
        "channel_ids": torch.arange(1, channels + 1).repeat(batch, 1),
        "channel_mask": torch.ones(batch, channels, dtype=torch.bool),
        "continuous": torch.ones(batch, 5),
        "continuous_missing": torch.zeros(batch, 5, dtype=torch.bool),
    }
    for field in [
        "dataset_id",
        "reference",
        "hardware_id",
        "electrode_type",
        "cap_type",
        "reattach_flag",
    ]:
        cond[field] = torch.ones(batch, dtype=torch.long)
    return cond


def test_metadata_missing_masks_prompt_channels_not_backbone_channels():
    x = torch.ones(4, 8, 20)
    cond = _condition()
    _, masked = _robustness_perturbation(
        {"type": "metadata_missing", "name": "missing"}
    )(x, cond)

    assert masked["channel_mask"].all()
    assert masked["channel_ids"].ne(0).all()
    assert not masked["condition_channel_mask"].any()
    assert not masked["condition_channel_ids"].any()
    assert masked["continuous_missing"].all()


def test_generalization_drop_is_relative_to_reference():
    assert _relative_drop(0.8, 0.6) == pytest.approx(0.25)
    assert _relative_drop(0.0, 0.0) == 0.0


def _calibration_manifest() -> pd.DataFrame:
    rows = []
    for label in [0, 1]:
        for trial in range(6):
            rows.append(
                {
                    "sample_id": f"wearable_s01_l{label}_t{trial}",
                    "dataset_id": "wearable",
                    "subject_id": "s01",
                    "label": label,
                }
            )
    return pd.DataFrame(rows)


def _selected_sample_ids(manifest: pd.DataFrame, indices: np.ndarray) -> set[str]:
    return set(manifest.iloc[indices]["sample_id"].astype(str))


def _ordered_sample_ids(manifest: pd.DataFrame, indices: np.ndarray) -> list[str]:
    return manifest.iloc[indices]["sample_id"].astype(str).tolist()


def test_calibration_uses_one_disjoint_fixed_query_across_all_budgets():
    manifest = _calibration_manifest()
    support_by_label, query = _subject_calibration_partition(
        manifest, np.arange(len(manifest)), max_budget=3, seed=42
    )
    query_ids = _selected_sample_ids(manifest, query)
    query_hash = _sample_id_identity_sha256(list(query_ids))

    supports = []
    for budget in [0, 1, 3]:
        support = _calibration_support_indices(support_by_label, budget)
        support_ids = _selected_sample_ids(manifest, support)
        assert support_ids.isdisjoint(query_ids)
        assert _sample_id_identity_sha256(list(query_ids)) == query_hash
        assert len(query) == 6
        assert len(support) == 2 * budget
        supports.append(support_ids)

    assert supports[0] == set()
    assert supports[1] < supports[2]


def test_calibration_partition_is_deterministic_under_manifest_reordering():
    manifest = _calibration_manifest()
    first_support, first_query = _subject_calibration_partition(
        manifest, np.arange(len(manifest)), max_budget=3, seed=7
    )
    shuffled = manifest.sample(frac=1.0, random_state=99).reset_index(drop=True)
    second_support, second_query = _subject_calibration_partition(
        shuffled, np.arange(len(shuffled)), max_budget=3, seed=7
    )

    assert _selected_sample_ids(manifest, first_query) == _selected_sample_ids(
        shuffled, second_query
    )
    for label in first_support:
        assert _ordered_sample_ids(manifest, first_support[label]) == _ordered_sample_ids(
            shuffled, second_support[label]
        )


def test_calibration_fails_when_max_budget_would_consume_the_query():
    manifest = _calibration_manifest()

    with pytest.raises(ValueError, match="would leave no fixed query trial"):
        _subject_calibration_indices(
            manifest,
            np.arange(len(manifest)),
            budget=6,
            max_budget=6,
            seed=42,
        )


def test_held_out_selection_uses_checkpoint_split_csv_sample_ids(tmp_path):
    checkpoint_path = tmp_path / "run" / "best.pt"
    checkpoint_path.parent.mkdir()
    pd.DataFrame(
        {
            "sample_id": ["train-1", "val-1", "test-1", "test-2"],
            "split": ["train", "val", "test", "test"],
        }
    ).to_csv(checkpoint_path.parent / "split.csv", index=False)
    context = {
        "checkpoint_path": checkpoint_path,
        "manifest": pd.DataFrame(
            {
                "sample_id": ["test-2", "train-1", "test-1", "external"],
                "dataset_id": ["wearable"] * 4,
            }
        ),
    }

    selected = _held_out_evaluation_indices(
        {"data": {"split": "test"}}, context, mode="robustness"
    )

    assert selected.tolist() == [0, 2]


def test_robustness_refuses_ambiguous_whole_manifest(tmp_path):
    context = {
        "checkpoint_path": tmp_path / "best.pt",
        "manifest": pd.DataFrame(
            {
                "sample_id": ["train-1", "test-1"],
                "dataset_id": ["wearable", "wearable"],
            }
        ),
    }

    with pytest.raises(ValueError, match="no safe held-out selection"):
        _scenarios({}, context, "robustness")
