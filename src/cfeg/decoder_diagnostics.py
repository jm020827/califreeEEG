"""Small decoder controls with explicit subject splits and no metadata inputs.

EEGNet architecture reference (independent PyTorch implementation):
https://github.com/vlawhern/arl-eegmodels/blob/master/EEGModels.py
The temporal kernel is 100 samples for the project's 200 Hz input.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from torch import nn

from cfeg.data.splits import SplitIndices, make_cross_dataset_split, make_cross_subject_split

OCCIPITAL_IDS = [48, 55, 57, 56, 53, 61, 62, 63]


def subject_protocols(manifest: pd.DataFrame, seed: int) -> dict[str, SplitIndices]:
    """Reuse identical held-out subjects in single-dataset and pooled controls."""
    parts = {}
    for dataset in ("wang", "beta"):
        indices = np.flatnonzero(manifest.dataset_id.eq(dataset).to_numpy())
        local = make_cross_subject_split(
            manifest.iloc[indices], seed=seed, val_ratio=0.2, test_ratio=0.2
        )
        parts[dataset] = SplitIndices(
            *(indices[getattr(local, name)] for name in ("train", "val", "test"))
        )
    parts["pooled"] = SplitIndices(
        *(
            np.concatenate([getattr(parts[d], name) for d in ("wang", "beta")])
            for name in ("train", "val", "test")
        )
    )
    parts["wang_to_beta"] = make_cross_dataset_split(
        manifest, ["wang"], ["beta"], seed=seed, val_ratio=0.2
    )
    for split in parts.values():
        assert_subject_disjoint(manifest, split)
    return parts


def assert_subject_disjoint(manifest: pd.DataFrame, split: SplitIndices) -> None:
    groups = manifest.dataset_id.astype(str) + "::" + manifest.subject_id.astype(str)
    selections = [set(groups.iloc[getattr(split, name)]) for name in ("train", "val", "test")]
    if any(not x for x in selections):
        raise ValueError("Empty subject split")
    if any(selections[i] & selections[j] for i, j in ((0, 1), (0, 2), (1, 2))):
        raise ValueError("Subject leakage between train, validation, and test")


def calibration_partition(labels: np.ndarray, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Reserve one trial/class; use the same remaining query for k=0 and k=1."""
    rng = np.random.default_rng(seed)
    support = []
    for label in np.unique(labels):
        candidates = np.flatnonzero(labels == label)
        if len(candidates) < 2:
            raise ValueError("Calibration needs at least two trials per class")
        support.append(int(rng.choice(candidates)))
    support = np.asarray(support, dtype=int)
    query = np.setdiff1d(np.arange(len(labels)), support)
    return support, query


class ReferenceCCA(nn.Module):
    """Unregularized reference CCA via orthonormal subspaces, without labels.

    This is plain sine/cosine CCA, not filter-bank CCA or a learned template.
    QR/SVD implements the same maximum-correlation objective as standard CCA.
    """

    def __init__(self, frequencies: list[float], sfreq: float, n_times: int, harmonics=5):
        super().__init__()
        t = torch.arange(n_times, dtype=torch.float64) / sfreq
        refs = []
        for freq in frequencies:
            rows = [
                fn(2 * torch.pi * freq * h * t)
                for h in range(1, harmonics + 1)
                for fn in (torch.sin, torch.cos)
            ]
            reference = torch.stack(rows, dim=-1)
            reference = reference - reference.mean(dim=0, keepdim=True)
            q, _ = torch.linalg.qr(reference, mode="reduced")
            refs.append(q)
        self.register_buffer("reference_q", torch.stack(refs))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # SVD suppresses artificial directions for rank-deficient/zero channels.
        centered = x.double() - x.double().mean(dim=-1, keepdim=True)
        u, singular, _ = torch.linalg.svd(centered.transpose(-1, -2), full_matrices=False)
        keep = singular > singular[:, :1].clamp_min(1e-12) * 1e-8
        u = u * keep.unsqueeze(1)
        overlap = torch.einsum("btc,ftk->bfck", u, self.reference_q)
        return torch.linalg.svdvals(overlap)[..., 0].float()


class EEGNetDecoder(nn.Module):
    """EEGNet-8,2 style control; learns directly from EEG, no REVE or harmonic."""

    def __init__(self, channels=64, n_times=400, n_classes=40):
        super().__init__()
        self.temporal = nn.Conv2d(1, 8, (1, 100), padding="same", bias=False)
        self.temporal_norm = nn.BatchNorm2d(8, eps=1e-3, momentum=0.01)
        self.spatial = nn.Conv2d(8, 16, (channels, 1), groups=8, bias=False)
        self.rest = nn.Sequential(
            nn.BatchNorm2d(16, eps=1e-3, momentum=0.01),
            nn.ELU(),
            nn.AvgPool2d((1, 4)),
            nn.Dropout(0.5),
            nn.Conv2d(16, 16, (1, 16), groups=16, padding="same", bias=False),
            nn.Conv2d(16, 16, 1, bias=False),
            nn.BatchNorm2d(16, eps=1e-3, momentum=0.01),
            nn.ELU(),
            nn.AvgPool2d((1, 8)),
            nn.Dropout(0.5),
            nn.Flatten(),
        )
        self.head = nn.Linear(16 * (n_times // 32), n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.rest(self.spatial(self.temporal_norm(self.temporal(x[:, None])))))

    @torch.no_grad()
    def constrain_weights(self) -> None:
        for weight, limit in ((self.spatial.weight, 1.0), (self.head.weight, 0.25)):
            norm = weight.flatten(1).norm(dim=1)
            scale = (limit / norm.clamp_min(1e-8)).clamp(max=1)
            weight.mul_(scale.view(-1, *([1] * (weight.ndim - 1))))
