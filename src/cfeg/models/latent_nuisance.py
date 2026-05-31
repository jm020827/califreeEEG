from __future__ import annotations

import torch
from torch import nn


class LatentNuisanceEncoder(nn.Module):
    def __init__(self, h_dim: int, cond_dim: int, z_dim: int = 16):
        super().__init__()
        self.z_dim = z_dim
        self.net = nn.Sequential(
            nn.LayerNorm(h_dim + cond_dim),
            nn.Linear(h_dim + cond_dim, h_dim),
            nn.GELU(),
            nn.Linear(h_dim, z_dim * 2),
        )

    def forward(self, h: torch.Tensor, cond_vec: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        inp = torch.cat([h.detach(), cond_vec], dim=-1)
        mu, logvar = self.net(inp).chunk(2, dim=-1)
        return mu, logvar.clamp(min=-8.0, max=8.0)

    def sample(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        eps = torch.randn_like(mu)
        return mu + eps * torch.exp(0.5 * logvar)

