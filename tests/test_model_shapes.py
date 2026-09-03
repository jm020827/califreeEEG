from __future__ import annotations

import torch

from cfeg.constants import CATEGORICAL_VOCABS
from cfeg.models.full_model import ConditionedEEGDecoder
from cfeg.models.heads import HarmonicPowerPrior


def test_full_model_forward_shape():
    cfg = {
        "model": {
            "n_classes": 4,
            "c_max": 64,
            "t_len": 400,
            "d_model": 32,
            "patch_size": 20,
            "depth": 1,
            "n_heads": 4,
            "backbone": {"name": "tiny_transformer"},
            "condition_encoder": {"enabled": True, "n_prompt_tokens": 2},
            "adapter": {"enabled": True, "bottleneck_dim": 8},
            "latent": {"enabled": True, "z_dim": 4},
            "spectral_prior": {"enabled": True, "n_harmonics": 3, "channel_ids": [48, 56]},
        }
    }
    model = ConditionedEEGDecoder(cfg)
    cond = {
        "continuous": torch.zeros(2, 5),
        "continuous_missing": torch.ones(2, 5, dtype=torch.bool),
        "channel_ids": torch.arange(64).unsqueeze(0).repeat(2, 1),
        "channel_mask": torch.ones(2, 64, dtype=torch.bool),
        "sfreq_processed_float": torch.full((2,), 200.0),
    }
    for name in CATEGORICAL_VOCABS:
        cond[name] = torch.zeros(2, dtype=torch.long)
    out = model(torch.randn(2, 64, 400), cond)
    assert out.logits.shape == (2, 4)
    assert out.logits_zero.shape == (2, 4)
    assert out.aux["spectral_logits"].shape == (2, 4)


def test_harmonic_power_prior_prefers_matching_frequency_and_respects_mask():
    prior = HarmonicPowerPrior(
        frequencies_hz=[8.0, 10.0],
        sample_rate_hz=200.0,
        n_samples=400,
        n_harmonics=3,
        channel_ids=[1],
        trainable_scale=False,
        harmonic_weighting="uniform",
    )
    time = torch.arange(400) / 200.0
    x = torch.zeros(2, 2, 400)
    x[0, 0] = torch.sin(2 * torch.pi * 8.0 * time)
    x[1, 0] = torch.sin(2 * torch.pi * 10.0 * time)
    mask = torch.tensor([[True, True], [True, False]])
    logits = prior(x, mask)
    assert logits.argmax(dim=-1).tolist() == [0, 1]
