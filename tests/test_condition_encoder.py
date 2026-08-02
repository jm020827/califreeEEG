from __future__ import annotations

import torch

from cfeg.constants import CATEGORICAL_VOCABS
from cfeg.models.condition_encoder import ConditionEncoder


def _cond(batch=2, c=64):
    cond = {
        "continuous": torch.zeros(batch, 5),
        "continuous_missing": torch.ones(batch, 5, dtype=torch.bool),
        "channel_ids": torch.arange(c).unsqueeze(0).repeat(batch, 1),
        "channel_mask": torch.ones(batch, c, dtype=torch.bool),
    }
    for name, values in CATEGORICAL_VOCABS.items():
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
