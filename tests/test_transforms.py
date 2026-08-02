from __future__ import annotations

import torch

from cfeg.data.transforms import RandomChannelDropout, RandomChannelSubset


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


def test_random_channel_subset_reaches_two_channel_condition():
    x = torch.ones(2, 8, 20)
    cond = {
        "channel_ids": torch.arange(1, 9).repeat(2, 1),
        "channel_mask": torch.ones(2, 8, dtype=torch.bool),
        "continuous": torch.zeros(2, 5),
        "continuous_missing": torch.ones(2, 5, dtype=torch.bool),
    }
    reduced_x, reduced_cond = RandomChannelSubset([[1, 2]], p=1.0)(x, cond)
    assert reduced_cond["channel_mask"].sum(dim=1).tolist() == [2, 2]
    assert reduced_x[:, 2:].eq(0).all()
