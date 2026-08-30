from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import torch

from cfeg.eval_loop import (
    _calibration_adaptation_seed,
    _calibration_prediction_rows,
    _calibration_support_indices,
    _held_out_evaluation_indices,
    _metadata_shuffle_indices,
    _relative_drop,
    _robustness_perturbation,
    _sample_id_identity_sha256,
    _scenarios,
    _subject_calibration_indices,
    _subject_calibration_partition,
    _validate_checkpoint_asset_revisions,
    _validate_checkpoint_dataset_provenance,
    _validate_fresh_governed_evaluation_output,
    _validate_split_assignment_contract,
)
from cfeg.governance import GovernanceError
from cfeg.train_loop import _sha256_json


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


def _condition_v04(batch: int = 4, channels: int = 8):
    cond = _condition(batch=batch, channels=channels)
    cond.update(
        {
            "metadata_contract_version": "0.4-dev",
            "external_continuous": torch.ones(batch, 2),
            "external_continuous_missing": torch.zeros(batch, 2, dtype=torch.bool),
            "query_qc": torch.ones(batch, 1),
            "query_qc_missing": torch.zeros(batch, 1, dtype=torch.bool),
            "channel_query_qc": torch.ones(batch, channels),
            "channel_query_qc_missing": torch.zeros(batch, channels, dtype=torch.bool),
            "channel_impedance": torch.ones(batch, channels),
            "channel_impedance_missing": torch.zeros(batch, channels, dtype=torch.bool),
        }
    )
    return cond


def test_metadata_missing_masks_prompt_channels_not_backbone_channels():
    x = torch.ones(4, 8, 20)
    cond = _condition()
    _, masked = _robustness_perturbation({"type": "metadata_missing", "name": "missing"})(x, cond)

    assert masked["channel_mask"].all()
    assert masked["channel_ids"].ne(0).all()
    assert not masked["condition_channel_mask"].any()
    assert not masked["condition_channel_ids"].any()
    assert masked["continuous_missing"].all()


def test_protocol_v04_metadata_missing_preserves_structure_and_query_qc():
    x = torch.ones(4, 8, 20)
    cond = _condition_v04()
    _, masked = _robustness_perturbation({"type": "metadata_missing", "name": "missing"})(x, cond)

    torch.testing.assert_close(masked["channel_ids"], cond["channel_ids"])
    torch.testing.assert_close(masked["channel_mask"], cond["channel_mask"])
    torch.testing.assert_close(masked["query_qc"], cond["query_qc"])
    torch.testing.assert_close(masked["query_qc_missing"], cond["query_qc_missing"])
    assert "condition_channel_ids" not in masked
    assert masked["external_continuous"].eq(0).all()
    assert masked["external_continuous_missing"].all()
    assert masked["channel_impedance"].eq(0).all()
    assert masked["channel_impedance_missing"].all()
    assert masked["reference"].eq(0).all()
    assert masked["electrode_type"].eq(0).all()
    assert masked["cap_type"].eq(0).all()


def test_protocol_v04_query_qc_missing_is_separate_from_external_metadata():
    x = torch.ones(4, 8, 20)
    cond = _condition_v04()
    _, masked = _robustness_perturbation(
        {
            "type": "metadata_group_missing",
            "name": "query_qc_missing",
            "query_qc_indices": [0],
        }
    )(x, cond)

    assert masked["query_qc"].eq(0).all()
    assert masked["query_qc_missing"].all()
    torch.testing.assert_close(masked["external_continuous"], cond["external_continuous"])


def test_generalization_drop_is_relative_to_reference():
    assert _relative_drop(0.8, 0.6) == pytest.approx(0.25)
    assert _relative_drop(0.0, 0.0) == 0.0


def test_governed_evaluation_refuses_existing_output_prefix(tmp_path):
    output = tmp_path / "robustness.csv"
    output.with_name("robustness_clean_predictions.csv").write_text(
        "sample_id\nexisting\n", encoding="utf-8"
    )
    context = {
        "research_access": type("Access", (), {"governed": True})(),
        "train_config": {},
    }

    with pytest.raises(GovernanceError, match="output prefix is not fresh"):
        _validate_fresh_governed_evaluation_output(
            {"output_csv": str(output)}, context, "robustness"
        )


def test_evaluation_rejects_stale_checkpoint_revision():
    with pytest.raises(ValueError, match="Checkpoint/data revision mismatch"):
        _validate_checkpoint_asset_revisions(
            {
                "asset_info": {
                    "datasets": [
                        {
                            "dataset_id": "wearable",
                            "dataset_revision": "wearable_v2",
                        }
                    ]
                }
            },
            {"wearable": "wearable_v3"},
        )


def test_evaluation_compares_matching_wearable_runtime_provenance(monkeypatch):
    signature = {"axis_labels": ["dry", "wet"], "axis_mean_kohm": [261.67, 19.63]}
    entry = {
        "dataset_id": "wearable",
        "dataset_revision": "wearable_v3",
        "query_qc_extractor_version": "filtered_cropped_pre_zscore_channel_std_median_v1",
        "external_continuous_schema": "impedance_mean_max_v1",
        "impedance_numeric_signature": signature,
        "processed_subject_count": 102,
    }
    asset_info = {"processed_dirs": [], "datasets": [entry]}
    checkpoint = {
        "config": {
            "protocol": {"metadata_contract_version": "0.4-dev"},
            "data": {"expected_revisions": {"wearable": "wearable_v3"}},
            "runtime_contract": {"asset_provenance_sha256": _sha256_json(asset_info)},
        },
        "asset_info": asset_info,
    }
    dataset = type("Dataset", (), {"asset_infos": [dict(entry)], "roots": []})()
    monkeypatch.setattr("cfeg.eval_loop._processed_asset_provenance", lambda roots: asset_info)

    _validate_checkpoint_dataset_provenance(checkpoint, dataset)

    dataset.asset_infos[0]["processed_subject_count"] = 3
    with pytest.raises(ValueError, match="processed_subject_count"):
        _validate_checkpoint_dataset_provenance(checkpoint, dataset)


def test_evaluation_rejects_legacy_checkpoint_without_claimed_dataset_provenance():
    checkpoint = {
        "config": {"data": {"processed_dirs": ["data/processed/wearable_v2"]}},
        "asset_info": {"processed_dirs": ["data/processed/wearable_v2"]},
    }
    dataset = type(
        "Dataset",
        (),
        {"asset_infos": [{"dataset_id": "wearable", "dataset_revision": "wearable_v3"}]},
    )()

    with pytest.raises(ValueError, match="no revisioned dataset provenance"):
        _validate_checkpoint_dataset_provenance(checkpoint, dataset)


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
    first = _calibration_adaptation_seed(42, dataset_id="wearable", subject_id="s01", budget=3)

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

    selected = _held_out_evaluation_indices({"data": {"split": "test"}}, context, mode="robustness")

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
    assert context["manifest"].iloc[scenarios[0][1]]["sample_id"].tolist() == ["beta-test"]


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


def test_protocol_v04_rejects_split_csv_not_bound_to_checkpoint(tmp_path):
    split_path = tmp_path / "split.csv"
    table = pd.DataFrame({"sample_id": ["train-1", "test-1"], "split": ["test", "test"]})
    context = {
        "train_config": {
            "protocol": {"metadata_contract_version": "0.4-dev"},
            "runtime_contract": {"split_assignment_sha256": "0" * 64},
        }
    }

    with pytest.raises(ValueError, match="split-assignment contract"):
        _validate_split_assignment_contract(context, table, split_path)


def test_metadata_shuffle_is_global_deterministic_and_changes_conditions() -> None:
    manifest = pd.DataFrame(
        {
            "sample_id": [f"sample-{index}" for index in range(12)],
            "dataset_id": ["wearable"] * 12,
            "subject_id": [f"sub{index:03d}" for index in range(1, 5) for _ in range(3)],
            "run_id": ["block01"] * 12,
            "label": [0, 1, 2] * 4,
            "electrode_type": [value for value in ("dry", "wet", "dry", "wet") for _ in range(3)],
            "impedance_mean_kohm": [value for value in range(4) for _ in range(3)],
        }
    )
    indices = np.arange(len(manifest))

    first = _metadata_shuffle_indices(manifest, indices, seed=42)
    second = _metadata_shuffle_indices(manifest, indices, seed=42)

    np.testing.assert_array_equal(first, second)
    assert np.count_nonzero(first != indices) == len(indices)
    np.testing.assert_array_equal(
        manifest.iloc[first]["label"].to_numpy(),
        manifest.iloc[indices]["label"].to_numpy(),
    )


def test_held_out_selection_rejects_incomplete_checkpoint_test_manifest(tmp_path):
    checkpoint_path = tmp_path / "run" / "best.pt"
    checkpoint_path.parent.mkdir()
    pd.DataFrame({"sample_id": ["test-1", "test-2"], "split": ["test", "test"]}).to_csv(
        checkpoint_path.parent / "split.csv", index=False
    )
    context = {
        "checkpoint_path": checkpoint_path,
        "manifest": pd.DataFrame({"sample_id": ["test-1"], "dataset_id": ["wearable"]}),
    }

    with pytest.raises(ValueError, match="missing 1 sample_id"):
        _held_out_evaluation_indices({"data": {"split": "test"}}, context, mode="robustness")


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
