from __future__ import annotations

import random

import torch


class ChannelSubset:
    def __init__(self, channel_ids: list[int], p: float = 1.0):
        self.channel_ids = set(int(x) for x in channel_ids)
        self.p = p

    def __call__(self, x: torch.Tensor, cond: dict) -> tuple[torch.Tensor, dict]:
        if random.random() > self.p:
            return x, cond
        out_cond = _clone_cond(cond)
        keep = torch.zeros_like(out_cond["channel_mask"])
        for cid in self.channel_ids:
            keep |= out_cond["channel_ids"].eq(cid)
        keep &= out_cond["channel_mask"]
        if keep.sum(dim=-1).min().item() == 0:
            return x, cond
        x2 = x * keep.unsqueeze(-1).to(x.dtype)
        out_cond["channel_mask"] = keep
        _update_n_channels(out_cond)
        return x2, out_cond


class RandomChannelDropout:
    def __init__(self, drop_prob: float = 0.2, min_channels: int = 4):
        self.drop_prob = drop_prob
        self.min_channels = min_channels

    def __call__(self, x: torch.Tensor, cond: dict) -> tuple[torch.Tensor, dict]:
        out_cond = _clone_cond(cond)
        mask = out_cond["channel_mask"].clone()
        rand = torch.rand_like(mask.float()) > self.drop_prob
        keep = mask & rand
        for b in range(mask.shape[0]):
            if keep[b].sum() < self.min_channels:
                available = torch.nonzero(mask[b], as_tuple=False).flatten()
                if len(available) > 0:
                    keep[b, available[: min(self.min_channels, len(available))]] = True
        x2 = x * keep.unsqueeze(-1).to(x.dtype)
        out_cond["channel_mask"] = keep
        _update_n_channels(out_cond)
        return x2, out_cond


class GaussianNoise:
    def __init__(self, std_range: tuple[float, float] = (0.01, 0.05), p: float = 0.5):
        self.std_range = std_range
        self.p = p

    def __call__(self, x: torch.Tensor, cond: dict) -> tuple[torch.Tensor, dict]:
        if random.random() > self.p:
            return x, cond
        std = random.uniform(*self.std_range)
        noise = torch.randn_like(x) * std
        return x + noise * cond["channel_mask"].unsqueeze(-1).to(x.dtype), cond


class TimeShift:
    def __init__(self, max_shift_samples: int = 8, p: float = 0.5):
        self.max_shift_samples = max_shift_samples
        self.p = p

    def __call__(self, x: torch.Tensor, cond: dict) -> tuple[torch.Tensor, dict]:
        if random.random() > self.p or self.max_shift_samples <= 0:
            return x, cond
        shift = random.randint(-self.max_shift_samples, self.max_shift_samples)
        return torch.roll(x, shifts=shift, dims=-1), cond


def make_two_views(
    x: torch.Tensor,
    cond: dict,
    channel_dropout_prob: float = 0.2,
    min_channels: int = 4,
    noise_std_range: tuple[float, float] = (0.01, 0.05),
    time_shift_samples: int = 8,
) -> tuple[tuple[torch.Tensor, dict], tuple[torch.Tensor, dict]]:
    weak_x, weak_cond = GaussianNoise((0.0, 0.01), p=0.5)(x, cond)
    strong_x, strong_cond = RandomChannelDropout(channel_dropout_prob, min_channels)(x, cond)
    strong_x, strong_cond = GaussianNoise(noise_std_range, p=0.8)(strong_x, strong_cond)
    strong_x, strong_cond = TimeShift(time_shift_samples, p=0.8)(strong_x, strong_cond)
    return (weak_x, weak_cond), (strong_x, strong_cond)


def _clone_cond(cond: dict) -> dict:
    return {k: v.clone() if torch.is_tensor(v) else v for k, v in cond.items()}


def _update_n_channels(cond: dict) -> None:
    n_channels = cond["channel_mask"].sum(dim=-1).float()
    c_max = cond["channel_mask"].shape[-1]
    cond["continuous"] = cond["continuous"].clone()
    cond["continuous"][:, 1] = torch.log1p(n_channels) / torch.log1p(torch.tensor(float(c_max)))
    cond["continuous_missing"] = cond["continuous_missing"].clone()
    cond["continuous_missing"][:, 1] = False
