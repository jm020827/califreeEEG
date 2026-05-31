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

