"""Source-episodic, support-conditioned spectral metric; no data reader.

Q, Q2, QM and SHAM use the exact same graph. Their allowed context construction
is outside this module. Neither query labels nor participant IDs are accepted.
"""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class JointHarmonicPrototype(nn.Module):
    def __init__(
        self,
        *,
        n_channels: int = 8,
        n_classes: int = 12,
        n_harmonics: int = 3,
        channel_width: int = 32,
        embedding_dim: int = 64,
        context_width: int = 32,
        affine_bound: float = 0.5,
        temperature: float = 10.0,
    ) -> None:
        super().__init__()
        if min(n_channels, n_classes, n_harmonics, channel_width,
               embedding_dim, context_width) <= 0:
            raise ValueError("Model dimensions must be positive.")
        if not 0 < affine_bound < 1 or not 0 < temperature < float("inf"):
            raise ValueError("Invalid scale bound or temperature.")
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.n_features = 2 * n_classes * n_harmonics
        self.affine_bound = affine_bound
        self.temperature = temperature
        self.channel_encoder = nn.Linear(self.n_features, channel_width)
        self.conditioner = nn.Sequential(
            nn.Linear(10, context_width), nn.GELU(),
            nn.Linear(context_width, 2 * channel_width),
        )
        nn.init.zeros_(self.conditioner[-1].weight)
        nn.init.zeros_(self.conditioner[-1].bias)
        self.spatial_encoder = nn.Linear(n_channels * channel_width, embedding_dim)

    def encode(self, spectra: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        """[episode,trial,channel,feature] -> unit [episode,trial,embedding]."""
        if spectra.ndim != 4 or spectra.shape[-2:] != (
            self.n_channels, self.n_features
        ):
            raise ValueError("Spectra shape does not match the frozen model dimensions.")
        if context.shape != (spectra.shape[0], self.n_channels, 10):
            raise ValueError("Context must be [episode,channel,10].")
        if not torch.isfinite(spectra).all() or not torch.isfinite(context).all():
            raise ValueError("Spectra and sanitized context must be finite.")
        scale, shift = self.conditioner(context).chunk(2, dim=-1)
        hidden = F.gelu(self.channel_encoder(spectra))
        hidden = F.gelu(
            (1 + self.affine_bound * scale.tanh()[:, None]) * hidden
            + self.affine_bound * shift.tanh()[:, None]
        )
        embedded = self.spatial_encoder(hidden.flatten(start_dim=-2))
        return F.normalize(embedded, p=2, dim=-1, eps=1e-8)

    def forward(
        self,
        support: torch.Tensor,
        support_labels: torch.Tensor,
        query: torch.Tensor,
        context: torch.Tensor,
    ) -> torch.Tensor:
        """Joint gradients flow through paid support and source-episode queries."""
        if support.ndim != 4 or query.ndim != 4:
            raise ValueError("Support and query must both be four-dimensional.")
        if support.shape[0] != query.shape[0] or support_labels.shape != support.shape[:2]:
            raise ValueError("Episode or support-label shapes differ.")
        if support_labels.dtype != torch.long or support_labels.numel() == 0:
            raise ValueError("Nonempty support labels must have dtype long.")
        if support_labels.min() < 0 or support_labels.max() >= self.n_classes:
            raise ValueError("Support label outside fixed class bank.")
        assignments = F.one_hot(support_labels, self.n_classes).to(support.dtype)
        counts = assignments.sum(dim=1)
        if (counts == 0).any():
            raise ValueError("Every episode requires support for every class.")
        support_embedding = self.encode(support, context)
        query_embedding = self.encode(query, context)
        prototypes = torch.einsum("bnj,bnd->bjd", assignments, support_embedding)
        prototypes = F.normalize(prototypes / counts[..., None], dim=-1, eps=1e-8)
        return self.temperature * torch.einsum("bnd,bjd->bnj", query_embedding, prototypes)
