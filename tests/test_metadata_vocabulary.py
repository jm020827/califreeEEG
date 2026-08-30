from __future__ import annotations

import numpy as np
import torch

from cfeg.constants import CATEGORICAL_VOCABS, METADATA_CONTRACT_V04_DEV
from cfeg.data.collate import build_vocabularies, collate_eeg
from cfeg.data.schema import EEGSample


def _sample(*, dataset_id: str = "wang") -> EEGSample:
    return EEGSample(
        x=np.ones((2, 8), dtype=np.float32),
        y=0,
        sample_id=f"{dataset_id}-sample",
        dataset_id=dataset_id,
        subject_id="subject-1",
        session_id="session-1",
        channel_mask=np.asarray([True, True]),
        canonical_channel_ids=np.asarray([1, 2], dtype=np.int64),
        sfreq=200.0,
        reference="average",
        hardware_id="neuroscan_synamp2",
        electrode_type="wet",
        cap_type="wet_cap",
        n_channels_used=2,
        impedance_mean_kohm=5.0,
        impedance_max_kohm=10.0,
        reattach_flag=False,
        time_since_last_session_hours=12.0,
        query_signal_std=25.0,
        query_signal_std_by_channel=np.asarray([20.0, 30.0], dtype=np.float32),
        impedance_kohm_by_channel=np.asarray([5.0, 10.0], dtype=np.float32),
    )


def test_training_vocabulary_excludes_unseen_global_categories():
    training_manifest_rows = [vars(_sample(dataset_id="wang"))]
    vocab = build_vocabularies(training_manifest_rows)

    assert all(field_vocab["unknown"] == 0 for field_vocab in vocab.values())
    assert vocab["dataset_id"] == {"unknown": 0, "wang": 1}
    assert "beta" not in vocab["dataset_id"]

    target_batch = collate_eeg([_sample(dataset_id="beta")], vocabularies=vocab)
    assert target_batch["cond"]["dataset_id"].item() == 0


def test_legacy_registry_fallback_keeps_unknown_at_zero():
    vocab = build_vocabularies()

    assert all(field_vocab["unknown"] == 0 for field_vocab in vocab.values())
    assert vocab["dataset_id"]["wang"] == 2
    assert vocab["dataset_id"]["beta"] == 3


def test_categorical_dropout_only_replaces_categorical_ids_with_unknown():
    sample = _sample()
    vocab = build_vocabularies([sample])
    baseline = collate_eeg(
        [sample],
        vocabularies=vocab,
        categorical_metadata_dropout_prob=0.0,
    )
    dropped = collate_eeg(
        [sample],
        vocabularies=vocab,
        categorical_metadata_dropout_prob=1.0,
    )

    for field in CATEGORICAL_VOCABS:
        assert baseline["cond"][field].item() != 0
        assert dropped["cond"][field].item() == 0

    for field in (
        "channel_ids",
        "channel_mask",
        "continuous",
        "continuous_missing",
        "sfreq_processed_float",
    ):
        torch.testing.assert_close(dropped["cond"][field], baseline["cond"][field])


def test_categorical_dropout_probability_is_validated():
    sample = _sample()
    vocab = build_vocabularies([sample])

    for invalid_probability in (-0.01, 1.01):
        try:
            collate_eeg(
                [sample],
                vocabularies=vocab,
                categorical_metadata_dropout_prob=invalid_probability,
            )
        except ValueError as exc:
            assert "between 0 and 1" in str(exc)
        else:
            raise AssertionError("invalid categorical dropout probability was accepted")


def test_protocol_v04_nulls_only_external_metadata():
    sample = _sample()
    vocab = build_vocabularies([sample])
    observed = collate_eeg(
        [sample],
        vocabularies=vocab,
        metadata_contract_version=METADATA_CONTRACT_V04_DEV,
        external_metadata_mode="observed",
    )["cond"]
    null = collate_eeg(
        [sample],
        vocabularies=vocab,
        metadata_contract_version=METADATA_CONTRACT_V04_DEV,
        external_metadata_mode="null",
    )["cond"]

    for field in (
        "channel_ids",
        "channel_mask",
        "continuous",
        "continuous_missing",
        "sfreq_processed_float",
        "query_qc",
        "query_qc_missing",
        "channel_query_qc",
        "channel_query_qc_missing",
    ):
        torch.testing.assert_close(null[field], observed[field])

    assert observed["external_continuous_missing"].logical_not().all()
    assert null["external_continuous_missing"].all()
    assert null["external_continuous"].eq(0).all()
    assert null["channel_impedance_missing"].all()
    assert observed["reference"].item() != 0
    assert observed["electrode_type"].item() != 0
    assert observed["cap_type"].item() != 0
    assert null["reference"].item() == 0
    assert null["electrode_type"].item() == 0
    assert null["cap_type"].item() == 0
    # The compatibility tensor retains structure only; impedance/elapsed time
    # must not remain available as an ungoverned side channel.
    assert observed["continuous"][0, :2].ne(0).all()
    assert observed["continuous"][0, 2:].eq(0).all()
    assert observed["continuous_missing"][0, 2:].all()
    # Dataset/hardware IDs and reattachment are forbidden proxies/confounds in v0.4.
    assert observed["dataset_id"].item() == 0
    assert observed["hardware_id"].item() == 0
    assert observed["reattach_flag"].item() == 0


def test_query_qc_is_independent_of_batch_partners():
    target = _sample(dataset_id="wang")
    donor = _sample(dataset_id="beta")
    donor.query_signal_std = 80.0
    vocab = build_vocabularies([target, donor])

    alone = collate_eeg(
        [target],
        vocabularies=vocab,
        metadata_contract_version=METADATA_CONTRACT_V04_DEV,
    )["cond"]["query_qc"][0]
    together = collate_eeg(
        [target, donor],
        vocabularies=vocab,
        metadata_contract_version=METADATA_CONTRACT_V04_DEV,
    )["cond"]["query_qc"][0]

    torch.testing.assert_close(alone, together)
