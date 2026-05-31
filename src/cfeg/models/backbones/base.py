from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


@dataclass
class BackboneOutput:
    h: torch.Tensor
    tokens: torch.Tensor | None
    aux: dict[str, torch.Tensor]


class EEGBackbone(nn.Module):
    d_model: int
    supports_prompt_tokens: bool = False

    def forward(
        self,
        x: torch.Tensor,
        cond: dict[str, torch.Tensor],
        prompt_tokens: torch.Tensor | None = None,
        return_tokens: bool = False,
    ) -> BackboneOutput:
        raise NotImplementedError

