from __future__ import annotations

import pytest
import torch

from cfeg.models.backbones.spectral_transformer import SpectralEEGTransformerBackbone


def _condition(batch: int, channels: int):
    return {
        "channel_ids": torch.arange(1, channels + 1).repeat(batch, 1),
        "channel_mask": torch.ones(batch, channels, dtype=torch.bool),
    }


def test_spectral_backbone_shape_frequency_contract_and_gradient() -> None:
    model = SpectralEEGTransformerBackbone(
        c_max=8,
        t_len=400,
        target_sfreq=200.0,
        d_model=32,
        depth=2,
        n_heads=4,
        min_frequency_hz=6.0,
        max_frequency_hz=60.0,
    )
    x = torch.randn(3, 8, 400, requires_grad=True)

    output = model(x, _condition(3, 8), return_tokens=True)
    output.h.square().mean().backward()

    assert output.h.shape == (3, 32)
    assert output.tokens.shape == (3, 9, 32)
    assert model.frequency_hz[0].item() == 6.0
    assert model.frequency_hz[-1].item() == 60.0
    assert x.grad is not None and torch.isfinite(x.grad).all()


def test_spectral_backbone_accepts_prompt_tokens_and_masks_channels() -> None:
    model = SpectralEEGTransformerBackbone(
        c_max=4,
        t_len=200,
        target_sfreq=200.0,
        d_model=16,
        depth=1,
        n_heads=4,
    ).eval()
    condition = _condition(2, 4)
    condition["channel_mask"][:, -1] = False
    prompt = torch.randn(2, 3, 16)

    with torch.no_grad():
        output = model(torch.randn(2, 4, 200), condition, prompt, return_tokens=True)

    assert output.h.shape == (2, 16)
    assert output.tokens.shape == (2, 8, 16)


def test_spectral_channel_gain_ones_is_exact_noop_and_shape_is_checked() -> None:
    model = SpectralEEGTransformerBackbone(
        c_max=4,
        t_len=200,
        target_sfreq=200.0,
        d_model=16,
        depth=1,
        n_heads=4,
        dropout=0.0,
    ).eval()
    x = torch.randn(2, 4, 200)
    condition = _condition(2, 4)

    with torch.no_grad():
        plain = model(x, condition).h
        neutral = model(x, condition, channel_gain=torch.ones(2, 4)).h
    torch.testing.assert_close(plain, neutral, rtol=0.0, atol=0.0)

    with pytest.raises(ValueError, match="shape"):
        model(x, condition, channel_gain=torch.ones(2, 3))
    with pytest.raises(ValueError, match="finite positive"):
        model(x, condition, channel_gain=torch.zeros(2, 4))
