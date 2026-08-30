from __future__ import annotations

import pytest
import torch

from cfeg.constants import CATEGORICAL_VOCABS, METADATA_CONTRACT_V04_DEV
from cfeg.models.condition_encoder import ConditionEncoder


def _cond(batch=2, c=64):
    cond = {
        "continuous": torch.zeros(batch, 5),
        "continuous_missing": torch.ones(batch, 5, dtype=torch.bool),
        "channel_ids": torch.arange(c).unsqueeze(0).repeat(batch, 1),
        "channel_mask": torch.ones(batch, c, dtype=torch.bool),
        "external_continuous": torch.zeros(batch, 2),
        "external_continuous_missing": torch.ones(batch, 2, dtype=torch.bool),
        "query_qc": torch.zeros(batch, 1),
        "query_qc_missing": torch.zeros(batch, 1, dtype=torch.bool),
        "channel_impedance": torch.zeros(batch, c),
        "channel_impedance_missing": torch.ones(batch, c, dtype=torch.bool),
    }
    for name in CATEGORICAL_VOCABS:
        cond[name] = torch.zeros(batch, dtype=torch.long)
    return cond


def test_condition_encoder_prompt_shape():
    enc = ConditionEncoder(
        d_model=32,
        n_prompt_tokens=3,
        vocab_sizes={k: len(v) for k, v in CATEGORICAL_VOCABS.items()},
        n_cont_features=5,
        channel_vocab_size=65,
    )
    prompt, cond_vec = enc(_cond())
    assert prompt.shape == (2, 3, 32)
    assert cond_vec.shape == (2, 32)


def test_condition_encoder_zero_prompt():
    enc = ConditionEncoder(
        d_model=32,
        n_prompt_tokens=0,
        vocab_sizes={k: len(v) for k, v in CATEGORICAL_VOCABS.items()},
        n_cont_features=5,
        channel_vocab_size=65,
    )
    prompt, cond_vec = enc(_cond())
    assert prompt is None
    assert cond_vec.shape == (2, 32)


def test_condition_encoder_uses_dynamic_training_vocabulary_sizes():
    vocab_sizes = {name: 1 for name in CATEGORICAL_VOCABS}
    vocab_sizes["dataset_id"] = 2
    enc = ConditionEncoder(
        d_model=32,
        n_prompt_tokens=2,
        vocab_sizes=vocab_sizes,
        n_cont_features=5,
        channel_vocab_size=65,
    )

    assert enc.cat_embeddings["dataset_id"].num_embeddings == 2
    for name in set(CATEGORICAL_VOCABS) - {"dataset_id"}:
        assert enc.cat_embeddings[name].num_embeddings == 1


def test_dataset_id_only_encoder_ignores_continuous_and_channel_metadata():
    enc = ConditionEncoder(
        d_model=32,
        n_prompt_tokens=2,
        vocab_sizes={k: len(v) for k, v in CATEGORICAL_VOCABS.items()},
        n_cont_features=5,
        channel_vocab_size=65,
        fields=["dataset_id"],
        include_continuous=False,
        include_channels=False,
    ).eval()
    first = _cond()
    second = _cond()
    second["continuous"].normal_()
    second["continuous_missing"].logical_not_()
    second["channel_ids"] = torch.flip(second["channel_ids"], dims=[1])
    with torch.no_grad():
        prompt_a, vector_a = enc(first)
        prompt_b, vector_b = enc(second)
    torch.testing.assert_close(prompt_a, prompt_b)
    torch.testing.assert_close(vector_a, vector_b)


def test_condition_channel_override_masks_prompt_metadata_only():
    enc = ConditionEncoder(
        d_model=32,
        n_prompt_tokens=2,
        vocab_sizes={k: len(v) for k, v in CATEGORICAL_VOCABS.items()},
        n_cont_features=5,
        channel_vocab_size=65,
        fields=[],
        include_continuous=False,
        include_channels=True,
    ).eval()
    cond = _cond()
    masked = {key: value.clone() for key, value in cond.items()}
    masked["condition_channel_ids"] = torch.zeros_like(cond["channel_ids"])
    masked["condition_channel_mask"] = torch.zeros_like(cond["channel_mask"])

    with torch.no_grad():
        prompt_full, _ = enc(cond)
        prompt_masked, _ = enc(masked)
    assert not torch.allclose(prompt_full, prompt_masked)
    torch.testing.assert_close(masked["channel_ids"], cond["channel_ids"])
    torch.testing.assert_close(masked["channel_mask"], cond["channel_mask"])


def test_force_missing_is_an_architecture_matched_metadata_null_control():
    encoder = ConditionEncoder(
        d_model=16,
        n_prompt_tokens=2,
        vocab_sizes={name: len(values) for name, values in CATEGORICAL_VOCABS.items()},
        n_cont_features=5,
        channel_vocab_size=65,
        fields=["reference", "electrode_type", "cap_type", "reattach_flag"],
        force_missing=True,
    ).eval()
    first = _cond(batch=2, c=4)
    second = {key: value.clone() for key, value in first.items()}
    for name in CATEGORICAL_VOCABS:
        second[name].fill_(1)
    second["continuous"].fill_(9.0)
    second["continuous_missing"].fill_(False)
    second["channel_ids"] = torch.flip(second["channel_ids"], dims=[1])
    second["channel_mask"][:, 0] = False

    first_prompt, first_vector = encoder(first)
    second_prompt, second_vector = encoder(second)

    torch.testing.assert_close(first_prompt, second_prompt)
    torch.testing.assert_close(first_vector, second_vector)


def _protocol_v04_encoder(mode: str) -> ConditionEncoder:
    return ConditionEncoder(
        d_model=16,
        n_prompt_tokens=2,
        vocab_sizes={name: len(values) for name, values in CATEGORICAL_VOCABS.items()},
        n_cont_features=5,
        channel_vocab_size=65,
        fields=["reference", "electrode_type", "cap_type"],
        include_continuous=True,
        include_channels=False,
        metadata_contract_version=METADATA_CONTRACT_V04_DEV,
        external_metadata_mode=mode,
    ).eval()


def test_protocol_v04_null_encoder_ignores_external_but_uses_query_qc():
    encoder = _protocol_v04_encoder("null")
    baseline = _cond(batch=2, c=4)
    changed_external = {key: value.clone() for key, value in baseline.items()}
    for name in ["reference", "electrode_type", "cap_type"]:
        changed_external[name].fill_(1)
    changed_external["external_continuous"].fill_(1.5)
    changed_external["external_continuous_missing"].fill_(False)
    changed_external["channel_impedance"].fill_(1.0)
    changed_external["channel_impedance_missing"].fill_(False)

    baseline_prompt, baseline_vector = encoder(baseline)
    external_prompt, external_vector = encoder(changed_external)
    torch.testing.assert_close(baseline_prompt, external_prompt)
    torch.testing.assert_close(baseline_vector, external_vector)

    changed_query = {key: value.clone() for key, value in baseline.items()}
    changed_query["query_qc"].fill_(1.0)
    query_prompt, query_vector = encoder(changed_query)
    assert not torch.allclose(baseline_prompt, query_prompt)
    assert not torch.allclose(baseline_vector, query_vector)


def test_protocol_v04_observed_encoder_responds_to_external_metadata():
    encoder = _protocol_v04_encoder("observed")
    baseline = _cond(batch=2, c=4)
    changed = {key: value.clone() for key, value in baseline.items()}
    changed["reference"].fill_(1)
    changed["external_continuous"].fill_(1.0)
    changed["external_continuous_missing"].fill_(False)

    baseline_prompt, _ = encoder(baseline)
    changed_prompt, _ = encoder(changed)

    assert not torch.allclose(baseline_prompt, changed_prompt)


def test_protocol_v04_null_and_observed_have_identical_parameter_contracts():
    null = _protocol_v04_encoder("null")
    observed = _protocol_v04_encoder("observed")

    assert {name: tuple(value.shape) for name, value in null.state_dict().items()} == {
        name: tuple(value.shape) for name, value in observed.state_dict().items()
    }


def test_protocol_v04_rejects_forbidden_identifier_or_confound_fields():
    with pytest.raises(ValueError, match="forbidden/confounded"):
        ConditionEncoder(
            d_model=16,
            n_prompt_tokens=2,
            vocab_sizes={name: len(values) for name, values in CATEGORICAL_VOCABS.items()},
            n_cont_features=5,
            channel_vocab_size=65,
            fields=["dataset_id", "reference"],
            include_channels=False,
            metadata_contract_version=METADATA_CONTRACT_V04_DEV,
        )
