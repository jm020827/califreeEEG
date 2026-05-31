from __future__ import annotations

import torch
from torch import nn


class ClassificationHead(nn.Module):
    def __init__(self, h_dim: int, n_classes: int, z_dim: int = 0, dropout: float = 0.1):
        super().__init__()
        self.z_dim = z_dim
        self.net = nn.Sequential(
            nn.LayerNorm(h_dim + z_dim),
            nn.Linear(h_dim + z_dim, h_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(h_dim, n_classes),
        )

    def forward(self, h: torch.Tensor, z: torch.Tensor | None = None) -> torch.Tensor:
        if self.z_dim > 0:
            if z is None:
                z = torch.zeros((h.shape[0], self.z_dim), device=h.device, dtype=h.dtype)
            h = torch.cat([h, z], dim=-1)
        return self.net(h)

