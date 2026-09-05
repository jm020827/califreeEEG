from __future__ import annotations

import hashlib
from dataclasses import replace

import h5py
import numpy as np
import pandas as pd
import pytest
import torch

from cfeg.data.metadata_calibration_features import WEARABLE_CALIBRATION_CHANNEL_IDS
from cfeg.data.metadata_calibration_interventions import (
    metadata_calibration_base_input_sha256,
)
from cfeg.data.unlabeled_query import (
    OpaqueTokenDomain,
    UnlabeledQueryReader,
    collate_metadata_calibration_queries,
    opaque_metadata_row_token,
    opaque_query_token,
    token_secret_commitment_sha256,
)
from cfeg.models.backbones.spectral_transformer import SpectralEEGTransformerBackbone
from cfeg.models.metadata_calibration_composite import MetadataCalibrationComposite
from cfeg.models.metadata_calibration_prior import BoundedMetadataCalibrationPrior


def _channel_ids() -> list[int]:
    values = [0] * 64
    for channel_id in WEARABLE_CALIBRATION_CHANNEL_IDS:
        values[channel_id - 1] = channel_id
    return values


def _channel_mask() -> np.ndarray:
    return np.asarray(_channel_ids(), dtype=np.int64) > 0


def _domain(*, checkpoint_group: str = "held") -> OpaqueTokenDomain:
    return OpaqueTokenDomain(
        phase="held_participant_evaluation",
        checkpoint_group=checkpoint_group,
        purpose="evaluation_query",
        raw_asset_fingerprint_sha256="a" * 64,
        seal_decision_sha256="b" * 64,
    )


def _view() -> pd.DataFrame:
    key = b"k" * 32
    base_input_sha256 = metadata_calibration_base_input_sha256(
        np.ones((64, 400), dtype=np.float32),
        _channel_mask(),
    )
    rows = [
        {
            "query_token": opaque_query_token(
                "subject_dry_target07", secret_key=key, domain=_domain()
            ),
            "signal_index": 0,
            "subject_id": "sub001",
            "electrode_type": "dry",
            "checkpoint_group": "held",
            "run_id": "block06",
            "canonical_channel_ids": _channel_ids(),
            "sfreq_processed": 200.0,
            "query_signal_std_by_channel": [1.0] * 64,
            "query_signal_std_missing_by_channel": [False] * 64,
            "impedance_kohm_by_channel": [0.5] * 64,
            "impedance_missing_by_channel": [False] * 64,
            "base_input_sha256": base_input_sha256,
        },
        {
            "query_token": opaque_query_token(
                "subject_wet_target09", secret_key=key, domain=_domain()
            ),
            "signal_index": 1,
            "subject_id": "sub001",
            "electrode_type": "wet",
            "checkpoint_group": "held",
            "run_id": "block06",
            "canonical_channel_ids": _channel_ids(),
            "sfreq_processed": 200.0,
            "query_signal_std_by_channel": [1.0] * 64,
            "query_signal_std_missing_by_channel": [False] * 64,
            "impedance_kohm_by_channel": [0.0] + [0.25] * 63,
            "impedance_missing_by_channel": [True] + [False] * 63,
            "base_input_sha256": base_input_sha256,
        },
    ]
    rows.sort(key=lambda row: row["query_token"])
    for signal_index, row in enumerate(rows):
        row["signal_index"] = signal_index
    return pd.DataFrame(rows)


def test_unlabeled_reader_succeeds_when_hdf5_has_no_y_dataset(tmp_path) -> None:
    path = tmp_path / "signals.h5"
    with h5py.File(path, "w") as handle:
        handle.create_dataset("x", data=np.ones((2, 64, 400), dtype=np.float32))
        handle.create_dataset("channel_mask", data=np.stack([_channel_mask()] * 2))
        assert "y" not in handle

    reader = UnlabeledQueryReader(path, _view())
    sample = reader[0]
    assert sample.x.shape == (64, 400)
    assert sample.channel_mask.shape == (64,)
    assert sample.query_token.startswith("q_")
    assert not hasattr(sample, "y")
    assert not hasattr(sample, "sample_id")
    wet = next(reader[index] for index in range(len(reader)) if reader[index].electrode_type == "wet")
    assert wet.impedance_missing_by_channel[0]
    assert wet.impedance_kohm_by_channel[0] == 0.0
    reader.close()


def test_query_collator_normalizes_values_without_adding_outcomes(tmp_path) -> None:
    path = tmp_path / "signals.h5"
    with h5py.File(path, "w") as handle:
        handle.create_dataset("x", data=np.ones((2, 64, 400), dtype=np.float32))
        handle.create_dataset("channel_mask", data=np.stack([_channel_mask()] * 2))
    reader = UnlabeledQueryReader(path, _view())
    batch = collate_metadata_calibration_queries([reader[0], reader[1]])
    assert batch.x.shape == (2, 64, 400)
    assert not hasattr(batch, "y")
    assert not hasattr(batch, "sample_id")
    assert set(batch.cond) == {
        "channel_ids",
        "channel_mask",
        "sfreq_processed_float",
        "channel_query_qc",
        "channel_query_qc_missing",
        "channel_impedance",
        "channel_impedance_missing",
        "metadata_contract_version",
    }
    assert batch.cond["channel_impedance_missing"][1, 0]
    assert batch.cond["channel_impedance"][1, 0] == 0.0
    backbone = SpectralEEGTransformerBackbone(
        c_max=64,
        t_len=400,
        target_sfreq=200.0,
        d_model=16,
        depth=1,
        n_heads=4,
        dropout=0.0,
    )
    prior = BoundedMetadataCalibrationPrior(
        embedding_dim=16,
        n_classes=12,
        q_feature_dim=56,
        metadata_feature_dim=18,
        hidden_dim=8,
    )
    output = MetadataCalibrationComposite(backbone=backbone, prior=prior).eval()(
        query_x=batch.x,
        query_cond=batch.cond,
        support_x=torch.empty(0, 64, 400),
        support_cond={},
        support_labels=torch.empty(0, dtype=torch.long),
        query_q_mode="observed",
        support_q_mode="observed",
        query_metadata_mode="observed",
        support_metadata_mode="observed",
        query_electrode_types=batch.electrode_types,
        support_electrode_types=(),
    )
    assert output.probabilities.shape == (2, 12)
    reader.close()


def test_unlabeled_reader_rejects_label_proxies_and_nonopaque_tokens(tmp_path) -> None:
    path = tmp_path / "signals.h5"
    with h5py.File(path, "w") as handle:
        handle.create_dataset("x", data=np.ones((2, 64, 400), dtype=np.float32))
        handle.create_dataset("channel_mask", data=np.stack([_channel_mask()] * 2))

    with pytest.raises(ValueError, match="forbidden columns"):
        UnlabeledQueryReader(path, _view().assign(sample_id="contains_target07"))
    with pytest.raises(ValueError, match="forbidden columns"):
        UnlabeledQueryReader(path, _view().assign(h5_index=[7, 9]))
    bad = _view()
    bad.loc[0, "query_token"] = "subject_target07"
    with pytest.raises(ValueError, match="opaque"):
        UnlabeledQueryReader(path, bad)

    not_zero_filled = _view()
    wet_row = not_zero_filled.index[not_zero_filled["electrode_type"].eq("wet")][0]
    not_zero_filled.at[wet_row, "impedance_kohm_by_channel"] = [0.5] + [0.25] * 63
    reader = UnlabeledQueryReader(path, not_zero_filled)
    with pytest.raises(ValueError, match="zero-filled"):
        _ = reader[int(wet_row)]
    reader.close()


def test_unlabeled_reader_rejects_source_hdf5_with_label_dataset(tmp_path) -> None:
    path = tmp_path / "source_signals.h5"
    with h5py.File(path, "w") as handle:
        handle.create_dataset("x", data=np.ones((2, 64, 400), dtype=np.float32))
        handle.create_dataset("channel_mask", data=np.stack([_channel_mask()] * 2))
        handle.create_dataset("y", data=np.asarray([7, 9], dtype=np.int64))
    with pytest.raises(ValueError, match="only x and channel_mask"):
        UnlabeledQueryReader(path, _view())


@pytest.mark.parametrize(
    ("x_dtype", "mask_dtype", "message"),
    [
        (np.float64, bool, "exact float32"),
        (np.float32, np.uint8, "exact boolean"),
    ],
)
def test_unlabeled_reader_rejects_noncanonical_hdf5_dtypes(
    tmp_path,
    x_dtype,
    mask_dtype,
    message,
) -> None:
    path = tmp_path / "bad_dtype_signals.h5"
    with h5py.File(path, "w") as handle:
        handle.create_dataset("x", data=np.ones((2, 64, 400), dtype=x_dtype))
        handle.create_dataset(
            "channel_mask",
            data=np.stack([_channel_mask()] * 2).astype(mask_dtype),
        )
    with pytest.raises(ValueError, match=message):
        UnlabeledQueryReader(path, _view())


@pytest.mark.parametrize("kind", ["x", "channel_mask"])
def test_unlabeled_reader_rehashes_decoded_signal_and_mask_bytes(
    tmp_path,
    kind: str,
) -> None:
    path = tmp_path / f"tampered_{kind}.h5"
    with h5py.File(path, "w") as handle:
        handle.create_dataset("x", data=np.ones((2, 64, 400), dtype=np.float32))
        handle.create_dataset("channel_mask", data=np.stack([_channel_mask()] * 2))
    reader = UnlabeledQueryReader(path, _view())
    active_slot = int(np.flatnonzero(_channel_mask())[0])
    with h5py.File(path, "r+") as handle:
        if kind == "x":
            handle["x"][0, active_slot, 0] = np.float32(2.0)
        else:
            handle["channel_mask"][0, active_slot] = False
    with pytest.raises(ValueError, match="signal bytes differ"):
        _ = reader[0]
    reader.close()


def test_unlabeled_reader_rejects_noncanonical_signal_shape(tmp_path) -> None:
    path = tmp_path / "bad_shape_signals.h5"
    with h5py.File(path, "w") as handle:
        handle.create_dataset("x", data=np.ones((2, 64, 399), dtype=np.float32))
        handle.create_dataset("channel_mask", data=np.stack([_channel_mask()] * 2))
    with pytest.raises(ValueError, match="exact shape"):
        UnlabeledQueryReader(path, _view())


def test_unlabeled_reader_rejects_source_order_even_with_local_indices(tmp_path) -> None:
    path = tmp_path / "query_signals.h5"
    with h5py.File(path, "w") as handle:
        handle.create_dataset("x", data=np.ones((2, 64, 400), dtype=np.float32))
        handle.create_dataset("channel_mask", data=np.stack([_channel_mask()] * 2))
    source_order = _view().sort_values("query_token", ascending=False).reset_index(drop=True)
    source_order["signal_index"] = [0, 1]
    with pytest.raises(ValueError, match="opaque-token order"):
        UnlabeledQueryReader(path, source_order)


def test_opaque_query_token_requires_secret_and_hides_identity() -> None:
    token = opaque_query_token(
        "wearable_sub001_target07", secret_key=b"s" * 32, domain=_domain()
    )
    assert token.startswith("q_") and len(token) == 66
    assert "target07" not in token
    assert token == opaque_query_token(
        "wearable_sub001_target07", secret_key=b"s" * 32, domain=_domain()
    )
    with pytest.raises(ValueError, match="at least 32"):
        opaque_query_token(
            "wearable_sub001_target07", secret_key=b"short", domain=_domain()
        )

    different_group = opaque_query_token(
        "wearable_sub001_target07",
        secret_key=b"s" * 32,
        domain=_domain(checkpoint_group="fold0"),
    )
    support_domain = replace(_domain(), purpose="evaluation_support")
    support = opaque_metadata_row_token(
        "wearable_sub001_target07",
        secret_key=b"s" * 32,
        domain=support_domain,
    )
    assert different_group != token
    assert support.startswith("s_") and support[2:] != token[2:]
    assert token_secret_commitment_sha256(b"s" * 32) != hashlib.sha256(b"s" * 32).hexdigest()
