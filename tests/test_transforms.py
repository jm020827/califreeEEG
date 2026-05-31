from __future__ import annotations

import torch

from cfeg.data.transforms import RandomChannelDropout


def test_channel_dropout_keeps_min_channels_and_updates_cond():
    x = torch.ones(2, 8, 10)
    cond = {
        "channel_mask": torch.ones(2, 8, dtype=torch.bool),
        "channel_ids": torch.arange(8).unsqueeze(0).repeat(2, 1),
        "continuous": torch.zeros(2, 5),
        "continuous_missing": torch.ones(2, 5, dtype=torch.bool),
    }
    x2, cond2 = RandomChannelDropout(drop_prob=1.0, min_channels=3)(x, cond)
    assert cond2["channel_mask"].sum(dim=-1).min() >= 3
    assert x2.shape == x.shape
    assert not cond2["continuous_missing"][:, 1].any()

