from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import torch

from cfeg.eval_loop import (
    _calibration_prediction_rows,
    _calibration_adaptation_seed,
    _calibration_support_indices,
    _held_out_evaluation_indices,
    _metadata_shuffle_indices,
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


def test_calibration_prediction_rows_record_query_and_subject_identity():
    rows = _calibration_prediction_rows(
        dataset_id="wearable",
        subject_id="s01",
        budget=3,
        query_identity_sha256="abc123",
        sample_ids=["sample-1", "sample-2"],
        y_true=np.asarray([0, 1]),
        logits=np.asarray([[3.0, 0.0], [0.0, 3.0]]),
    )

    assert [row["sample_id"] for row in rows] == ["sample-1", "sample-2"]
    assert [row["prediction"] for row in rows] == [0, 1]
    assert {row["calibration_trials_per_class"] for row in rows} == {3}
    assert {row["query_identity_sha256"] for row in rows} == {"abc123"}


def test_calibration_adaptation_seed_is_order_independent_and_specific() -> None:
    first = _calibration_adaptation_seed(
        42, dataset_id="wearable", subject_id="s01", budget=3
    )

    assert first == _calibration_adaptation_seed(
        42, dataset_id="wearable", subject_id="s01", budget=3
    )
    assert first != _calibration_adaptation_seed(
        42, dataset_id="wearable", subject_id="s02", budget=3
    )
    assert first != _calibration_adaptation_seed(
        42, dataset_id="wearable", subject_id="s01", budget=5
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


def test_calibration_selection_intersects_test_split_and_electrode_filter(tmp_path):
    checkpoint_path = tmp_path / "joint-condition" / "best.pt"
    checkpoint_path.parent.mkdir()
    pd.DataFrame(
        {
            "sample_id": [
                "train-dry",
                "train-wet",
                "val-wet",
                "test-dry",
                "test-wet",
            ],
            "split": ["train", "train", "val", "test", "test"],
        }
    ).to_csv(checkpoint_path.parent / "split.csv", index=False)
    context = {
        "checkpoint_path": checkpoint_path,
        "manifest": pd.DataFrame(
            {
                "sample_id": [
                    "train-dry",
                    "train-wet",
                    "val-wet",
                    "test-dry",
                    "test-wet",
                ],
                "dataset_id": ["wearable"] * 5,
                "subject_id": ["train", "train", "val", "test", "test"],
                "electrode_type": ["dry", "wet", "wet", "dry", "wet"],
            }
        ),
    }

    selected = _held_out_evaluation_indices(
        {
            "data": {"split": "test"},
            "test_datasets": ["wearable"],
            "test_filter": {"electrode_type": "wet"},
        },
        context,
        mode="calibration",
    )

    selected_rows = context["manifest"].iloc[selected]
    assert selected_rows["sample_id"].tolist() == ["test-wet"]
    assert set(selected_rows["subject_id"]) == {"test"}


def test_cross_condition_scenario_uses_only_test_subject_target_condition(tmp_path):
    checkpoint_path = tmp_path / "joint-condition" / "best.pt"
    checkpoint_path.parent.mkdir()
    pd.DataFrame(
        {
            "sample_id": ["train-wet", "val-wet", "test-dry", "test-wet"],
            "split": ["train", "val", "test", "test"],
        }
    ).to_csv(checkpoint_path.parent / "split.csv", index=False)
    context = {
        "checkpoint_path": checkpoint_path,
        "manifest": pd.DataFrame(
            {
                "sample_id": ["train-wet", "val-wet", "test-dry", "test-wet"],
                "dataset_id": ["wearable"] * 4,
                "subject_id": ["train", "val", "test", "test"],
                "electrode_type": ["wet", "wet", "dry", "wet"],
            }
        ),
    }

    scenarios = _scenarios(
        {
            "data": {"split": "test"},
            "test_filter": {"dataset_id": "wearable", "electrode_type": "wet"},
        },
        context,
        "cross_condition",
    )

    _, indices, perturbation = scenarios[0]
    assert context["manifest"].iloc[indices]["sample_id"].tolist() == ["test-wet"]
    assert perturbation is None


def test_cross_dataset_scenario_intersects_checkpoint_test_and_target_dataset(tmp_path):
    checkpoint_path = tmp_path / "wang-to-beta" / "best.pt"
    checkpoint_path.parent.mkdir()
    pd.DataFrame(
        {
            "sample_id": ["wang-train", "beta-test"],
            "split": ["train", "test"],
        }
    ).to_csv(checkpoint_path.parent / "split.csv", index=False)
    context = {
        "checkpoint_path": checkpoint_path,
        "manifest": pd.DataFrame(
            {
                "sample_id": ["wang-train", "beta-test"],
                "dataset_id": ["wang", "beta"],
                "label": [0, 0],
            }
        ),
        "train_config": {"data": {"train_datasets": ["wang"]}},
    }

    scenarios = _scenarios(
        {
            "data": {"split": "test"},
            "test_datasets": ["beta"],
            "common_label_subset_only": False,
        },
        context,
        "cross_dataset",
    )

    assert scenarios[0][0] == "zero_shot_beta"
    assert context["manifest"].iloc[scenarios[0][1]]["sample_id"].tolist() == [
        "beta-test"
    ]


def test_standard_scenario_refuses_implicit_whole_manifest(tmp_path):
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
        _scenarios({}, context, "standard")


def test_metadata_shuffle_is_global_deterministic_and_changes_conditions() -> None:
    manifest = pd.DataFrame(
        {
            "sample_id": [f"sample-{index}" for index in range(12)],
            "dataset_id": ["wearable"] * 12,
            "electrode_type": ["dry"] * 6 + ["wet"] * 6,
            "impedance_mean_kohm": list(range(12)),
        }
    )
    indices = np.arange(len(manifest))

    first = _metadata_shuffle_indices(manifest, indices, seed=42)
    second = _metadata_shuffle_indices(manifest, indices, seed=42)

    np.testing.assert_array_equal(first, second)
    assert np.count_nonzero(first != indices) == len(indices)
    donor_electrodes = manifest.iloc[first]["electrode_type"].to_numpy()
    target_electrodes = manifest.iloc[indices]["electrode_type"].to_numpy()
    assert np.count_nonzero(donor_electrodes != target_electrodes) >= len(indices) // 2


def test_held_out_selection_rejects_incomplete_checkpoint_test_manifest(tmp_path):
    checkpoint_path = tmp_path / "run" / "best.pt"
    checkpoint_path.parent.mkdir()
    pd.DataFrame(
        {"sample_id": ["test-1", "test-2"], "split": ["test", "test"]}
    ).to_csv(checkpoint_path.parent / "split.csv", index=False)
    context = {
        "checkpoint_path": checkpoint_path,
        "manifest": pd.DataFrame(
            {"sample_id": ["test-1"], "dataset_id": ["wearable"]}
        ),
    }

    with pytest.raises(ValueError, match="missing 1 sample_id"):
        _held_out_evaluation_indices(
            {"data": {"split": "test"}}, context, mode="robustness"
        )


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
