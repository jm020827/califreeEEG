from __future__ import annotations

import torch

from cfeg.models.backbones.reve import REVEBackbone


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
