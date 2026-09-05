from __future__ import annotations

import inspect

import pytest
import torch

from cfeg.constants import METADATA_CONTRACT_V04_DEV
from cfeg.data.metadata_calibration_features import (
    WEARABLE_CALIBRATION_CHANNEL_IDS,
    WEARABLE_METADATA_FEATURE_NAMES,
    WEARABLE_Q_FEATURE_NAMES,
    apply_metadata_feature_ablation,
    build_wearable_metadata_calibration_features,
    build_wearable_metadata_features,
    build_wearable_q_features,
    with_all_metadata_missing,
)


def _batch() -> tuple[torch.Tensor, dict[str, object]]:
    torch.manual_seed(13)
    batch, channels, time = 2, 64, 400
    x = torch.zeros(batch, channels, time)
    channel_ids = torch.zeros(batch, channels, dtype=torch.long)
    channel_mask = torch.zeros(batch, channels, dtype=torch.bool)
    query_qc = torch.zeros(batch, channels)
    query_qc_missing = torch.ones(batch, channels, dtype=torch.bool)
    impedance = torch.zeros(batch, channels)
    impedance_missing = torch.ones(batch, channels, dtype=torch.bool)
    for offset, channel_id in enumerate(WEARABLE_CALIBRATION_CHANNEL_IDS):
        slot = channel_id - 1
        x[:, slot] = torch.randn(batch, time) + 0.01 * offset
        channel_ids[:, slot] = channel_id
        channel_mask[:, slot] = True
        query_qc[:, slot] = 0.1 + 0.01 * offset
        query_qc_missing[:, slot] = False
        impedance[:, slot] = 0.2 + 0.1 * offset
        impedance_missing[:, slot] = False
    cond: dict[str, object] = {
        "channel_ids": channel_ids,
        "channel_mask": channel_mask,
        "channel_query_qc": query_qc,
        "channel_query_qc_missing": query_qc_missing,
        "channel_impedance": impedance,
        "channel_impedance_missing": impedance_missing,
        "sfreq_processed_float": torch.full((batch,), 200.0),
        "metadata_contract_version": METADATA_CONTRACT_V04_DEV,
    }
    return x, cond


def test_builds_frozen_q_and_external_metadata_schemas() -> None:
    x, cond = _batch()
    features = build_wearable_metadata_calibration_features(
        x, cond, electrode_types=["dry", "wet"]
    )

    assert features.q_features.shape == (2, 56)
    assert features.metadata_values.shape == (2, 18)
    assert features.metadata_missing.shape == (2, 18)
    assert len(WEARABLE_Q_FEATURE_NAMES) == 56
    assert len(WEARABLE_METADATA_FEATURE_NAMES) == 18
    torch.testing.assert_close(
        features.metadata_values[:, :2], torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    )
    torch.testing.assert_close(
        features.metadata_values[0, 2:10],
        torch.tensor([0.2 + 0.1 * index for index in range(8)]),
    )
    assert features.metadata_values[:, 10:].eq(1.0).all()
    assert not features.metadata_missing.any()


def test_channel_tensor_permutation_cannot_change_named_feature_order() -> None:
    x, cond = _batch()
    first = build_wearable_metadata_calibration_features(
        x, cond, electrode_types=["dry", "wet"]
    )
    permutation = torch.randperm(x.shape[1])
    permuted_cond = {
        key: (value[:, permutation] if torch.is_tensor(value) and value.ndim == 2 else value)
        for key, value in cond.items()
    }
    second = build_wearable_metadata_calibration_features(
        x[:, permutation], permuted_cond, electrode_types=["dry", "wet"]
    )

    torch.testing.assert_close(first.q_features, second.q_features, rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(first.metadata_values, second.metadata_values)
    assert torch.equal(first.metadata_missing, second.metadata_missing)


def test_q_and_m_builders_have_disjoint_information_rights() -> None:
    x, cond = _batch()
    q_only_cond = {
        key: value
        for key, value in cond.items()
        if key not in {"channel_impedance", "channel_impedance_missing"}
    }
    q = build_wearable_q_features(x, q_only_cond)

    m_only_cond = {
        key: value
        for key, value in cond.items()
        if key not in {"channel_query_qc", "channel_query_qc_missing", "sfreq_processed_float"}
    }
    values, missing = build_wearable_metadata_features(
        m_only_cond, electrode_types=["dry", "wet"]
    )
    assert q.shape == (2, 56)
    assert values.shape == missing.shape == (2, 18)


def test_impedance_missingness_is_explicit_and_all_missing_nulls_every_m_feature() -> None:
    x, cond = _batch()
    missing_channel = WEARABLE_CALIBRATION_CHANNEL_IDS[3] - 1
    cond["channel_impedance"][0, missing_channel] = float("nan")
    cond["channel_impedance_missing"][0, missing_channel] = True
    features = build_wearable_metadata_calibration_features(
        x, cond, electrode_types=["unknown", "wet"]
    )

    assert features.metadata_missing[0, :2].all()
    assert features.metadata_values[0, 2 + 3].item() == 0.0
    assert features.metadata_missing[0, 2 + 3].item() is True
    assert features.metadata_values[0, 10 + 3].item() == 0.0
    assert features.metadata_missing[0, 10 + 3].item() is False

    null = with_all_metadata_missing(features)
    assert torch.equal(null.q_features, features.q_features)
    assert null.metadata_values.eq(0.0).all()
    assert null.metadata_missing.all()


def test_builder_rejects_channel_identity_drift_and_has_no_label_or_id_api() -> None:
    x, cond = _batch()
    cond["channel_ids"][:, WEARABLE_CALIBRATION_CHANNEL_IDS[3] - 1] = (
        WEARABLE_CALIBRATION_CHANNEL_IDS[0]
    )
    with pytest.raises(ValueError, match="exactly once"):
        build_wearable_metadata_calibration_features(
            x, cond, electrode_types=["dry", "wet"]
        )

    parameters = set(inspect.signature(build_wearable_metadata_calibration_features).parameters)
    assert parameters == {"x", "cond", "electrode_types", "target_sfreq"}
    assert set(inspect.signature(build_wearable_q_features).parameters) == {
        "x",
        "cond",
        "target_sfreq",
    }
    assert set(inspect.signature(build_wearable_metadata_features).parameters) == {
        "cond",
        "electrode_types",
    }


def test_interface_and_impedance_ablation_masks_are_exact() -> None:
    _x, cond = _batch()
    values, missing = build_wearable_metadata_features(
        cond, electrode_types=["dry", "wet"]
    )
    interface_values, interface_missing = apply_metadata_feature_ablation(
        values, missing, ablation="interface_only"
    )
    assert torch.equal(interface_values[:, :2], values[:, :2])
    assert interface_values[:, 2:].eq(0.0).all()
    assert interface_missing[:, 2:].all()

    impedance_values, impedance_missing = apply_metadata_feature_ablation(
        values, missing, ablation="impedance_only"
    )
    assert impedance_values[:, :2].eq(0.0).all()
    assert impedance_missing[:, :2].all()
    assert torch.equal(impedance_values[:, 2:], values[:, 2:])
    assert torch.equal(impedance_missing[:, 2:], missing[:, 2:])
