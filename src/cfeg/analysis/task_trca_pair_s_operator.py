"""Pure C2 pair-weighted numerator and C-normalized temporal projectors.

Generated engineering only: this module does not authorize a human experiment.
S0, pair cross-products and C are constants; pair logits and the resulting S
retain first-order gradients. The unchanged leading-projector guards apply to
the raw two-solve H. There is no ridge, jitter, anchor or numerical rescue.
"""

from __future__ import annotations

import torch
from torch import Tensor

from cfeg.analysis.task_trca_shape_operator import (
    LOGIT_BOUND,
    _same_device,
    _symmetric,
    _tensor,
    leading_projector,
)


def pair_distribution(logits: Tensor) -> tuple[Tensor, Tensor]:
    """Return bounded probabilities and mean-relative deviations [...,n].

    Only the native pair widths n=3/10 are accepted. Computing deviations from
    the unnormalized exponential makes uniform logits yield exact zero without
    disconnecting their contrast gradients.
    """
    logits = _tensor(logits, "logits")
    if logits.ndim < 1 or logits.shape[-1] not in (3, 10):
        raise ValueError("logits must have a final native pair axis of 3 or 10")
    if (logits.abs() > LOGIT_BOUND + 1e-15).any():
        raise ValueError("logits exceed the frozen log(2)/2 bound")
    u = torch.exp(logits - logits.amax(dim=-1, keepdim=True))
    mean = u.mean(dim=-1, keepdim=True)
    p, d = u / u.sum(dim=-1, keepdim=True), (u - mean) / mean
    _tensor(p, "pair probabilities")
    _tensor(d, "pair deviations")
    n = logits.shape[-1]
    if (p <= 0).any() or ((p.sum(dim=-1) - 1).abs() > 1e-12).any():
        raise ValueError("pair probabilities must be positive and normalized")
    if (p.amax(dim=-1) / p.amin(dim=-1) > 2 + 1e-12).any():
        raise ValueError("pair probability max/min ratio exceeds two")
    if ((p < 1 / (2 * n - 1) - 1e-12) | (p > 2 / (n + 1) + 1e-12)).any():
        raise ValueError("pair probabilities violate the native-width bounds")
    return p, d


def weighted_numerator(s0: Tensor, pair_cross: Tensor, logits: Tensor) -> Tensor:
    """Return S0 + sum_pair(d A), retaining exact S0 at uniform logits.

    Shapes must agree without broadcasting: S0 [...,b,12,8,8], A
    [...,b,n,12,8,8], logits [...,b,n]. Symmetry validation deliberately does
    not replace S0/A by their averages; native floating construction is kept.
    """
    _symmetric(s0, "s0", constant=True)
    _symmetric(pair_cross, "pair_cross", constant=True)
    p, d = pair_distribution(logits)
    _same_device(s0, ("pair_cross", pair_cross), ("logits", logits))
    n = p.shape[-1]
    if (
        s0.ndim < 4
        or s0.shape[-3] != 12
        or pair_cross.shape != s0.shape[:-3] + (n,) + s0.shape[-3:]
        or logits.shape != s0.shape[:-3] + (n,)
    ):
        raise ValueError("S0/A/logits must have exactly matched batch/band/pair/12-class axes")
    result = s0 + (d[..., None, None, None] * pair_cross).sum(dim=-4)
    _tensor(result, "weighted numerator")
    return result


def c_projectors(s: Tensor, c: Tensor) -> Tensor:
    """Return rank-one F=K/tr(CK) from differentiable S and constant SPD C.

    B is exactly C. H is passed directly to the original leading_projector,
    including its symmetry and top-root-gap checks. tr(CK) is positive finite;
    its exact-arithmetic value one is a diagnostic, not an added runtime gate.
    """
    s, c = _symmetric(s, "s"), _symmetric(c, "c", constant=True)
    _same_device(s, ("c", c))
    if s.shape != c.shape:
        raise ValueError("s/c shapes must agree exactly")
    try:
        minimum = torch.linalg.eigvalsh(c)[..., 0]
    except RuntimeError as exc:
        raise ValueError("C eigensolver failed") from exc
    if not torch.isfinite(minimum).all() or (minimum <= 0).any():
        raise ValueError("c must be strictly positive definite; no jitter is permitted")
    try:
        lower = torch.linalg.cholesky(c)
    except RuntimeError as exc:
        raise ValueError("C Cholesky failed; no jitter is permitted") from exc
    left = torch.linalg.solve_triangular(lower, s, upper=False)
    h = torch.linalg.solve_triangular(lower, left.mT, upper=False).mT
    p = leading_projector(h)
    left = torch.linalg.solve_triangular(lower.mT, p, upper=True)
    k = torch.linalg.solve_triangular(lower.mT, left.mT, upper=True).mT
    k = (k + k.mT) / 2
    norm = torch.einsum("...ij,...ji->...", c, k)
    _tensor(norm, "projector C trace")
    if (norm <= 0).any():
        raise ValueError("projector C trace must be strictly positive")
    result = k / norm[..., None, None]
    _tensor(result, "C-normalized projector")
    return result


def pair_projectors(s0: Tensor, c: Tensor, pair_cross: Tensor, logits: Tensor) -> Tensor:
    """Compose the pair-weighted numerator and the unchanged B=C geometry."""
    return c_projectors(weighted_numerator(s0, pair_cross, logits), c)
