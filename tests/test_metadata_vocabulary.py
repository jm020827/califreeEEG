from __future__ import annotations

import numpy as np
import torch

from cfeg.constants import CATEGORICAL_VOCABS
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
