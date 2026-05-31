from __future__ import annotations

import torch
from torch import nn


class BottleneckAdapter(nn.Module):
    def __init__(self, d_model: int, bottleneck_dim: int = 32, dropout: float = 0.1):
        super().__init__()
        self.ln = nn.LayerNorm(d_model)
        self.down = nn.Linear(d_model, bottleneck_dim)
        self.act = nn.GELU()
        self.up = nn.Linear(bottleneck_dim, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.dropout(self.up(self.act(self.down(self.ln(x)))))


class ConditionedAdapter(nn.Module):
    def __init__(self, d_model: int, bottleneck_dim: int = 32, dropout: float = 0.1):
        super().__init__()
        self.adapter = BottleneckAdapter(d_model, bottleneck_dim, dropout)
        self.gate = nn.Linear(d_model, d_model)

    def forward(self, h: torch.Tensor, cond_vec: torch.Tensor | None = None) -> torch.Tensor:
        if cond_vec is None:
            return self.adapter(h)
        gate = torch.sigmoid(self.gate(cond_vec))
        return h + gate * (self.adapter(h) - h)

