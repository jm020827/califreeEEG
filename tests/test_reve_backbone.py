from __future__ import annotations

import torch
from torch import nn

from cfeg.data.preprocess import CanonicalChannelMap
from cfeg.models.backbones.reve import REVEBackbone, extract_reve_tokens


class _FakePosBank:
    def __call__(self, names):
        known = [name for name in names if name != "M1"]
        return torch.ones((len(known), 3), dtype=torch.float32)


def test_resolve_positions_drops_missing_channels():
    backbone = REVEBackbone.__new__(REVEBackbone)
    backbone.pos_bank = _FakePosBank()

    keep_indices, positions, missing_names = backbone._resolve_positions(["O1", "M1", "Oz"])

    assert keep_indices == [0, 2]
    assert positions.shape == (2, 3)
    assert missing_names == ["M1"]


def test_reve_exposes_prompt_fusion_contract_and_token_shape():
    assert REVEBackbone.supports_prompt_tokens is True
    tensor = torch.randn(2, 3, 5, 7)
    tokens = extract_reve_tokens({"last_hidden_state": tensor})
    assert tokens.shape == (2, 15, 7)


def test_reve_token_extraction_falls_back_when_last_hidden_state_is_none():
    class Output:
        last_hidden_state = None
        pooler_output = torch.randn(2, 7)

    assert extract_reve_tokens(Output()).shape == (2, 1, 7)


class _DeterministicPosBank:
    def __call__(self, names):
        values = {"O1": 1.0, "Oz": 2.0, "O2": 3.0}
        return torch.tensor([[values[name], 0.0, 0.0] for name in names], dtype=torch.float32)


class _MaskSensitiveFakeREVE(nn.Module):
    def forward(self, x, positions):
        return {"last_hidden_state": x.mean(dim=-1, keepdim=True) + positions[..., :1]}


class _InPlacePositionFakeREVE(nn.Module):
    def forward(self, x, positions):
        positions.add_(1.0)
        return {"last_hidden_state": x.mean(dim=-1, keepdim=True) + positions[..., :1]}


def _fake_reve_backbone() -> REVEBackbone:
    backbone = REVEBackbone.__new__(REVEBackbone)
    nn.Module.__init__(backbone)
    backbone.required_sample_rate_hz = 200.0
    backbone.pos_bank = _DeterministicPosBank()
    backbone.reve = _MaskSensitiveFakeREVE()
    backbone.d_model = 1
    backbone.canonical_map = CanonicalChannelMap(
        unknown_id=0,
        name_to_id={"O1": 1, "OZ": 2, "O2": 3},
        id_to_name={1: "O1", 2: "Oz", 3: "O2"},
    )
    backbone.output_proj = None
    backbone._position_cache = {}
    backbone.freeze = True
    return backbone


def test_reve_sample_is_invariant_to_mixed_mask_batch_partner():
    backbone = _fake_reve_backbone()
    x_target = torch.tensor([[[1.0, 1.0], [2.0, 2.0], [99.0, 99.0]]])
    cond_target = {
        "channel_ids": torch.tensor([[1, 2, 3]]),
        "channel_mask": torch.tensor([[True, True, False]]),
        "sfreq_processed_float": torch.tensor([200.0]),
    }
    alone = backbone(x_target, cond_target).h[0]

    x_partner = torch.tensor([[[0.0, 0.0], [0.0, 0.0], [4.0, 4.0]]])
    paired_cond = {
        "channel_ids": torch.tensor([[1, 2, 3], [1, 2, 3]]),
        "channel_mask": torch.tensor([[True, True, False], [False, False, True]]),
        "sfreq_processed_float": torch.tensor([200.0, 200.0]),
    }
    paired = backbone(torch.cat([x_target, x_partner]), paired_cond).h[0]

    torch.testing.assert_close(alone, paired)


def test_reve_rejects_active_unknown_channel_instead_of_silently_dropping_it():
    backbone = _fake_reve_backbone()
    x = torch.ones(1, 3, 2)
    cond = {
        "channel_ids": torch.tensor([[1, 0, 3]]),
        "channel_mask": torch.tensor([[True, True, False]]),
        "sfreq_processed_float": torch.tensor([200.0]),
    }

    try:
        backbone(x, cond)
    except ValueError as exc:
        assert "unknown canonical ID 0" in str(exc)
    else:
        raise AssertionError("active unknown channel was silently accepted by REVE")


def test_reve_position_bank_stays_eval_when_reve_is_finetuned():
    backbone = _fake_reve_backbone()
    backbone.pos_bank = nn.Sequential(nn.Dropout(p=0.5))
    backbone.reve = nn.Sequential(nn.Identity())
    backbone.freeze = False

    backbone.train()

    assert backbone.reve.training
    assert not backbone.pos_bank.training


def test_reve_trainable_inplace_position_noise_does_not_mutate_cached_base():
    backbone = _fake_reve_backbone()
    backbone.reve = _InPlacePositionFakeREVE()
    backbone.freeze = False
    x = torch.ones(2, 3, 2)
    cond = {
        "channel_ids": torch.tensor([[1, 2, 3], [1, 2, 3]]),
        "channel_mask": torch.tensor([[True, True, False], [True, True, False]]),
        "sfreq_processed_float": torch.tensor([200.0, 200.0]),
    }

    backbone(x, cond)
    cached = backbone._position_cache[("O1", "Oz")][1]

    torch.testing.assert_close(
        cached,
        torch.tensor([[1.0, 0.0, 0.0], [2.0, 0.0, 0.0]]),
    )
