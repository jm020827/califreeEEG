"""Pure M-blind source-template target; no reader, fitting, or study authority.

The caller must enforce source-only block, identity, preprocessing and label
rights. Array shapes cannot establish these facts. This target is a diagnostic
single-channel template margin, not a learned spatial score or calibration cost.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

NORM_FLOOR = 1e-12


def _finite(value, name):
    raw = np.asarray(value)
    if raw.dtype.kind not in "fiu" or not np.isfinite(raw).all():
        raise ValueError(f"{name} must be finite real numbers")
    return np.asarray(raw, dtype=np.float64)


def _unit_centered(value, name):
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        centered = value - value.mean(axis=-1, keepdims=True)
        norm = np.linalg.norm(centered, axis=-1, keepdims=True)
    if not np.isfinite(norm).all() or np.any(norm <= NORM_FLOOR):
        raise ValueError(f"{name} has undefined centered correlation")
    return centered / norm


@dataclass(frozen=True)
class ChannelMargin:
    """Class axes follow the supplied class IDs, not a best-label selection."""

    correlations: np.ndarray  # [true class, template class, band, channel]
    class_margin: np.ndarray  # [true class, band, channel]
    channel_margin: np.ndarray  # [band, channel]
    channel_shape: np.ndarray  # [band, channel], primary
    level: np.ndarray  # [band], descriptive only


def source_channel_margin(support, source_block, support_class_ids, source_class_ids):
    """Build class-balanced true-minus-strongest-wrong signed Pearson margin.

    support: [3,12,5,8,N>=2]; source_block: [12,5,8,N]. Class IDs must be
    matching permutations of 0..11. Any degenerate waveform rejects the whole
    target, rather than silently assigning zero correlation or dropping a row.
    Input arrays are not modified. Numeric M and final-query inputs do not exist.
    """
    s, x = _finite(support, "support"), _finite(source_block, "source_block")
    if s.ndim != 5 or s.shape[:4] != (3, 12, 5, 8) or s.shape[-1] < 2:
        raise ValueError("support must be [3,12,5,8,N>=2]")
    if x.shape != s.shape[1:]:
        raise ValueError("source_block must match [12,5,8,N]")
    ids = []
    for name, value in (
        ("support_class_ids", support_class_ids),
        ("source_class_ids", source_class_ids),
    ):
        labels = np.asarray(value)
        if (
            labels.shape != (12,)
            or labels.dtype.kind not in "iu"
            or not np.array_equal(np.sort(labels), np.arange(12))
        ):
            raise ValueError(f"{name} must be a permutation of integer IDs 0..11")
        ids.append(labels)
    if not np.array_equal(*ids):
        raise ValueError("support and source class order must match")
    with np.errstate(over="ignore", invalid="ignore"):
        template = s.mean(axis=0)
    t_unit = _unit_centered(template, "template")
    x_unit = _unit_centered(x, "source_block")
    correlation = np.einsum("ybcn,jbcn->yjbc", x_unit, t_unit, optimize=True)
    if not np.isfinite(correlation).all() or np.any(np.abs(correlation) > 1.0 + 1e-12):
        raise ValueError("invalid derived correlation")
    correlation = np.clip(correlation, -1.0, 1.0)
    diagonal = np.arange(12)
    wrong = correlation.copy()
    wrong[diagonal, diagonal] = -np.inf
    margin = correlation[diagonal, diagonal] - wrong.max(axis=1)
    channel = margin.mean(axis=0)
    level = channel.mean(axis=-1)
    return ChannelMargin(correlation, margin, channel, channel - level[:, None], level)


def participant_margin_mse(prediction, channel_shape):
    """Equal-person losses, never class/channel counts as independent people.

    Inputs are [participant,2 interfaces,5 bands,8 channels]. Center predictions
    within channels; require an already centered target. No pooled inference.
    """
    p, y = _finite(prediction, "prediction"), _finite(channel_shape, "channel_shape")
    if p.ndim != 4 or p.shape[0] < 1 or p.shape[1:] != (2, 5, 8) or y.shape != p.shape:
        raise ValueError("prediction and target must be [participant,2,5,8]")
    if np.max(np.abs(y.mean(axis=-1))) > 1e-12:
        raise ValueError("target must be channel centered")
    with np.errstate(over="ignore", invalid="ignore"):
        error = (p - p.mean(axis=-1, keepdims=True) - y) ** 2
        result = error.mean(axis=(1, 2, 3))
    return _finite(result, "participant MSE")
