from __future__ import annotations

import torch

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
