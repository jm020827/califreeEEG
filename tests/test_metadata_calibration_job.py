from __future__ import annotations

import pytest

from cfeg.metadata_calibration_job import _validate_source_selection_history


def _recipe() -> dict:
    return {
        "stage_1_common_Q": {
            "minimum_epochs": 3,
            "maximum_epochs": 5,
            "early_stop_patience": 2,
        },
        "stage_2_bounded_M": {
            "minimum_epochs": 2,
            "maximum_epochs": 4,
            "early_stop_patience": 2,
        },
    }


def _selection() -> dict:
    return {
        "stage1_epochs": 3,
        "stage2_epochs": 2,
        "stage1_history": [
            {
                "epoch": epoch,
                "train_loss": 1.0 / epoch,
                "validation_eauc": value,
                "validation_nll": nll,
            }
            for epoch, value, nll in (
                (1, 0.1, 2.0),
                (2, 0.2, 1.9),
                (3, 0.4, 1.5),
                (4, 0.39, 1.4),
                (5, 0.38, 1.3),
            )
        ],
        "stage2_history": [
            {
                "epoch": epoch,
                "train_loss": 1.0 / epoch,
                "validation_increment_eauc": value,
                "validation_correct_minus_shuffle_eauc": shuffle,
            }
            for epoch, value, shuffle in (
                (1, 0.0, 0.0),
                (2, 0.03, 0.02),
                (3, 0.02, 0.03),
                (4, 0.01, 0.04),
            )
        ],
    }


def test_source_epoch_history_replays_exact_eligibility_and_early_stop() -> None:
    _validate_source_selection_history(_selection(), recipe=_recipe())


def test_source_epoch_history_rejects_arbitrary_selected_epoch() -> None:
    selection = _selection()
    selection["stage1_epochs"] = 4
    with pytest.raises(ValueError, match="lexicographic optimum"):
        _validate_source_selection_history(selection, recipe=_recipe())


def test_source_epoch_history_rejects_truncation_before_frozen_stop() -> None:
    selection = _selection()
    selection["stage1_history"] = selection["stage1_history"][:-1]
    with pytest.raises(ValueError, match="ended before or after"):
        _validate_source_selection_history(selection, recipe=_recipe())
