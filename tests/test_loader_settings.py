from __future__ import annotations

import pytest

from cfeg.data.loader import resolve_loader_settings


def test_loader_settings_separate_training_and_evaluation_batches() -> None:
    settings = resolve_loader_settings(
        {
            "batch_size": 32,
            "eval_batch_size": 8,
            "num_workers": 2,
            "persistent_workers": True,
            "persistent_hdf5_handles": True,
            "preload_hdf5_to_memory": True,
            "preload_train_indices_to_memory": False,
        },
        device_type="cuda",
    )

    assert settings.train_batch_size == 32
    assert settings.eval_batch_size == 8
    assert settings.num_workers == 2
    assert settings.persistent_workers is True
    assert settings.pin_memory is True
    assert settings.persistent_hdf5_handles is True
    assert settings.preload_hdf5_to_memory is True
    assert settings.preload_train_indices_to_memory is False


def test_loader_settings_disable_persistent_workers_without_workers() -> None:
    settings = resolve_loader_settings(
        {"batch_size": 4, "num_workers": 0, "persistent_workers": True},
        device_type="cpu",
    )

    assert settings.persistent_workers is False
    assert settings.pin_memory is False


@pytest.mark.parametrize(
    "data_cfg",
    [
        {"batch_size": 0},
        {"batch_size": 1, "eval_batch_size": 0},
        {"batch_size": 1, "num_workers": -1},
    ],
)
def test_loader_settings_reject_invalid_values(data_cfg) -> None:
    with pytest.raises(ValueError):
        resolve_loader_settings(data_cfg, device_type="cpu")
