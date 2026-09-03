from __future__ import annotations

from types import SimpleNamespace

import pandas as pd
import pytest
import torch

from cfeg.eval_loop import (
    _calibration_target_mask,
    _relative_drop,
    _robustness_perturbation,
    _saved_split_scenarios,
    _scenario_reference_metrics,
    _select_prediction_logits,
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


def test_saved_split_reference_resets_for_each_dataset_clean_scenario():
    reference_accuracy, reference_itr = _scenario_reference_metrics(
        "saved_split",
        None,
        {"accuracy": 0.8, "itr_bits_per_min": 80.0},
        baseline_accuracy=None,
        baseline_itr=None,
    )
    reference_accuracy, reference_itr = _scenario_reference_metrics(
        "saved_split",
        object(),
        {"accuracy": 0.6, "itr_bits_per_min": 50.0},
        baseline_accuracy=reference_accuracy,
        baseline_itr=reference_itr,
    )
    assert reference_accuracy == pytest.approx(0.8)
    assert reference_itr == pytest.approx(80.0)

    reference_accuracy, reference_itr = _scenario_reference_metrics(
        "saved_split",
        None,
        {"accuracy": 0.5, "itr_bits_per_min": 40.0},
        baseline_accuracy=reference_accuracy,
        baseline_itr=reference_itr,
    )
    assert reference_accuracy == pytest.approx(0.5)
    assert reference_itr == pytest.approx(40.0)


def test_saved_split_scenarios_keep_only_held_out_samples_by_dataset(tmp_path):
    manifest = pd.DataFrame(
        [
            {"sample_id": "w_train", "dataset_id": "wang"},
            {"sample_id": "w_test", "dataset_id": "wang"},
            {"sample_id": "b_train", "dataset_id": "beta"},
            {"sample_id": "b_test", "dataset_id": "beta"},
        ]
    )
    split_csv = tmp_path / "split.csv"
    pd.DataFrame(
        [
            {"sample_id": "w_train", "split": "train"},
            {"sample_id": "w_test", "split": "test"},
            {"sample_id": "b_train", "split": "train"},
            {"sample_id": "b_test", "split": "test"},
        ]
    ).to_csv(split_csv, index=False)

    scenarios = _saved_split_scenarios(
        {
            "split_csv": str(split_csv),
            "split_name": "test",
            "test_datasets": ["wang", "beta"],
        },
        {"manifest": manifest},
    )

    assert [(name, indices.tolist()) for name, indices, _ in scenarios] == [
        ("saved_test_wang", [1]),
        ("saved_test_beta", [3]),
    ]


def test_calibration_target_mask_keeps_only_held_out_dataset_samples(tmp_path):
    manifest = pd.DataFrame(
        [
            {"sample_id": "b_train", "dataset_id": "beta"},
            {"sample_id": "b_test", "dataset_id": "beta"},
            {"sample_id": "w_test", "dataset_id": "wang"},
        ]
    )
    split_csv = tmp_path / "split.csv"
    pd.DataFrame(
        [
            {"sample_id": "b_train", "split": "train"},
            {"sample_id": "b_test", "split": "test"},
            {"sample_id": "w_test", "split": "test"},
        ]
    ).to_csv(split_csv, index=False)

    mask = _calibration_target_mask(
        {
            "split_csv": str(split_csv),
            "split_name": "test",
            "test_datasets": ["beta"],
        },
        manifest,
    )

    assert mask.tolist() == [False, True, False]


def test_prediction_branch_separates_learned_and_spectral_logits():
    spectral = torch.tensor([[0.5, -0.5]])
    learned = torch.tensor([[1.0, 2.0]])
    output = SimpleNamespace(
        logits=learned + spectral,
        aux={"spectral_logits": spectral},
    )

    assert torch.equal(_select_prediction_logits(output, "combined"), learned + spectral)
    assert torch.equal(_select_prediction_logits(output, "learned"), learned)
    assert torch.equal(_select_prediction_logits(output, "spectral"), spectral)


def test_saved_split_scenarios_add_noise_only_to_held_out_samples(tmp_path):
    manifest = pd.DataFrame(
        [
            {"sample_id": "train", "dataset_id": "beta"},
            {"sample_id": "test", "dataset_id": "beta"},
        ]
    )
    split_csv = tmp_path / "split.csv"
    pd.DataFrame(
        [
            {"sample_id": "train", "split": "train"},
            {"sample_id": "test", "split": "test"},
        ]
    ).to_csv(split_csv, index=False)

    scenarios = _saved_split_scenarios(
        {
            "split_csv": str(split_csv),
            "test_datasets": ["beta"],
            "perturbations": [
                {"name": "noise_01", "type": "gaussian_noise", "std": 0.1}
            ],
        },
        {"manifest": manifest},
    )

    assert [(name, indices.tolist()) for name, indices, _ in scenarios] == [
        ("saved_test_beta", [1]),
        ("saved_test_beta_noise_01", [1]),
    ]
    assert scenarios[0][2] is None
    assert callable(scenarios[1][2])
