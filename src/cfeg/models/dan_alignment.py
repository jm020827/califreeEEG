"""Paper-grounded, independently written spatial waveform alignment core.

Reference: SSVEP-DAN, arXiv:2311.12666v1, III-A--C. This is not an author-code
port or a claim of exact reproduction. Source-only BN/evaluation semantics are
explicit; this module has no data reader, participant lookup or experiment runner.
"""

from __future__ import annotations

import torch
from torch import nn


class DanAlignment(nn.Module):
    def __init__(self, channels: int, samples: int):
        super().__init__()
        if channels < 1 or samples < 2:
            raise ValueError("Positive channels and at least two samples required.")
        self.channels, self.samples = channels, samples
        self.spatial = nn.Linear(channels, channels, bias=False)
        self.normalization = nn.BatchNorm1d(channels * samples, eps=1e-5, momentum=0.1)
        self.hidden = nn.Linear(channels, channels)
        self.output = nn.Linear(channels, channels)

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        if waveform.ndim != 3 or waveform.shape[1:] != (self.channels, self.samples):
            raise ValueError("Expected [trial,channel,time] with the fixed waveform geometry.")
        if not waveform.is_floating_point() or not torch.isfinite(waveform).all():
            raise ValueError("Waveforms must be finite real floating point values.")
        if self.training and len(waveform) < 2:
            raise ValueError("Training BatchNorm requires at least two source trials.")
        hidden = self.spatial(waveform.transpose(1, 2))
        hidden = self.normalization(hidden.reshape(len(waveform), -1)).reshape(
            len(waveform), self.samples, self.channels)
        aligned = self.output(torch.tanh(self.hidden(hidden)))
        return aligned.transpose(1, 2)

    @torch.inference_mode()
    def transform_frozen(self, waveform: torch.Tensor) -> torch.Tensor:
        if self.training:
            raise RuntimeError("Freeze BatchNorm/model with eval() before source transformation.")
        return self(waveform)


def weighted_teacher(support: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    """Complete-block [k,class,channel,time] -> [class,channel,time] target.

    The original real support remains unchanged for downstream eTRCA. This
    teacher affects alignment training only, not query-dependent scoring.
    """
    if support.ndim != 4 or support.shape[0] < 2 or min(support.shape[1:]) < 1:
        raise ValueError("At least two complete support blocks are required.")
    if not support.is_floating_point() or not torch.isfinite(support).all():
        raise ValueError("Support must be finite floating point.")
    if weights.shape != (support.shape[0], support.shape[2]):
        raise ValueError("Expected one class-independent weight per block and channel.")
    if not weights.is_floating_point() or not torch.isfinite(weights).all() or (weights < 0).any():
        raise ValueError("Weights must be finite and nonnegative.")
    tolerance = 8 * torch.finfo(weights.dtype).eps
    if not torch.allclose(weights.sum(0), torch.ones_like(weights[0]), atol=tolerance, rtol=0):
        raise ValueError("Block weights must sum to one in every channel.")
    weights = weights.to(dtype=support.dtype, device=support.device)
    return torch.einsum("bc,bjct->jct", weights, support)


def alignment_loss(model: DanAlignment, source: torch.Tensor,
                   source_labels: torch.Tensor, teacher: torch.Tensor) -> torch.Tensor:
    """Cross-stimulus waveform MSE with labels from source trials, never query labels."""
    if source_labels.dtype != torch.long or source_labels.shape != (len(source),):
        raise ValueError("Need one integer class label per source trial.")
    if teacher.ndim != 3 or teacher.shape[1:] != source.shape[1:]:
        raise ValueError("Teacher must preserve the source channel/time geometry.")
    if not teacher.is_floating_point() or not torch.isfinite(teacher).all():
        raise ValueError("Teacher must be finite floating point.")
    if source_labels.numel() == 0 or source_labels.min() < 0 or source_labels.max() >= len(teacher):
        raise ValueError("Source label outside the target stimulus set.")
    return torch.mean((model(source) - teacher[source_labels]) ** 2)
