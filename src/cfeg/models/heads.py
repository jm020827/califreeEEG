from __future__ import annotations

import math

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


class HarmonicResidualGate(nn.Module):
    """Produce a bounded, standardized correction to harmonic logits."""

    def __init__(
        self,
        h_dim: int,
        *,
        initial_gate: float = 0.02,
        max_gate: float = 0.35,
        normalize_logits: bool = True,
    ):
        super().__init__()
        if max_gate <= 0:
            raise ValueError("residual_gate.max_gate must be positive")
        if not 0 < initial_gate < max_gate:
            raise ValueError("residual_gate.initial_gate must be between zero and max_gate")
        self.max_gate = float(max_gate)
        self.normalize_logits = bool(normalize_logits)
        self.norm = nn.LayerNorm(h_dim)
        self.projection = nn.Linear(h_dim, 1)
        nn.init.zeros_(self.projection.weight)
        ratio = float(initial_gate) / self.max_gate
        nn.init.constant_(self.projection.bias, math.log(ratio / (1.0 - ratio)))

    def forward(
        self, h: torch.Tensor, learned_logits: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        residual = learned_logits
        if self.normalize_logits:
            residual = (residual - residual.mean(dim=-1, keepdim=True)) / residual.std(
                dim=-1, keepdim=True, unbiased=False
            ).clamp_min(1e-6)
        gate = self.max_gate * torch.sigmoid(self.projection(self.norm(h)))
        return gate * residual, residual, gate


class HarmonicPowerPrior(nn.Module):
    """Phase-invariant SSVEP logits from fixed sinusoidal references."""

    def __init__(
        self,
        frequencies_hz: list[float],
        sample_rate_hz: float,
        n_samples: int,
        n_harmonics: int = 5,
        channel_ids: list[int] | None = None,
        logit_scale: float = 1.0,
        trainable_scale: bool = True,
        harmonic_weighting: str = "inverse",
    ):
        super().__init__()
        if not frequencies_hz:
            raise ValueError("spectral_prior.frequencies_hz must not be empty")
        if n_harmonics < 1:
            raise ValueError("spectral_prior.n_harmonics must be positive")
        time = torch.arange(n_samples, dtype=torch.float32) / float(sample_rate_hz)
        references = []
        for frequency in frequencies_hz:
            rows = []
            for harmonic in range(1, n_harmonics + 1):
                angle = 2.0 * torch.pi * harmonic * float(frequency) * time
                rows.extend((torch.sin(angle), torch.cos(angle)))
            references.append(torch.stack(rows))
        reference = torch.stack(references)
        reference = reference / reference.norm(dim=-1, keepdim=True).clamp_min(1e-8)
        self.register_buffer("reference", reference, persistent=True)
        harmonic_index = torch.arange(1, n_harmonics + 1, dtype=torch.float32)
        if harmonic_weighting == "uniform":
            harmonic_weights = torch.ones_like(harmonic_index)
        elif harmonic_weighting == "inverse":
            harmonic_weights = 1.0 / harmonic_index
        elif harmonic_weighting == "inverse_square":
            harmonic_weights = 1.0 / harmonic_index.square()
        else:
            raise ValueError(
                "spectral_prior.harmonic_weighting must be uniform, inverse, or inverse_square"
            )
        self.register_buffer("harmonic_weights", harmonic_weights, persistent=True)
        selected = (
            [] if channel_ids is None else [int(channel_id) - 1 for channel_id in channel_ids]
        )
        self.register_buffer(
            "selected_indices", torch.tensor(selected, dtype=torch.long), persistent=True
        )
        initial_log_scale = math.log(max(float(logit_scale), 1e-6))
        value = torch.tensor(initial_log_scale, dtype=torch.float32)
        if trainable_scale:
            self.log_scale = nn.Parameter(value)
        else:
            self.register_buffer("log_scale", value, persistent=True)

    def forward(self, x: torch.Tensor, channel_mask: torch.Tensor) -> torch.Tensor:
        # Keep the small projection in FP32 even when the REVE path uses autocast.
        x = x.float()
        mask = channel_mask.bool()
        if self.selected_indices.numel() > 0:
            indices = self.selected_indices.to(x.device)
            x = x.index_select(1, indices)
            mask = mask.index_select(1, indices)
        batch, channels, _ = x.shape
        n_classes, rows, _ = self.reference.shape
        n_harmonics = rows // 2
        projection = torch.einsum("bct,fkt->bcfk", x, self.reference.float())
        projection = projection.view(batch, channels, n_classes, n_harmonics, 2)
        power = projection.square().sum(dim=-1)
        power = (power * self.harmonic_weights.view(1, 1, 1, -1)).sum(dim=-1)
        weights = mask.float().unsqueeze(-1)
        scores = (power * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)
        scores = (scores - scores.mean(dim=-1, keepdim=True)) / scores.std(
            dim=-1, keepdim=True, unbiased=False
        ).clamp_min(1e-6)
        return scores * self.log_scale.exp()
