"""Generated-only N1 adapter for the unchanged temporal denominator-R mechanism.

Only the numerical projector route changes. Original S/C are checked before
averaging and remain detached CPU constants. R and F retain their first-order
gradients. No archive access, CUDA, fallback, jitter or legacy-policy mutation.
"""

from __future__ import annotations

import torch
from torch import Tensor

from cfeg.analysis import numerical_stability_operator as numerical
from cfeg.analysis.task_trca_shape_operator import ETA, R_MAX, _check_prior
from cfeg.analysis.task_trca_shape_signfree import (
    TemporalGramStatistics,
    score_temporal_gram,
    temporal_statistics,
)

METHOD = "N1_SYM_CHOLESKY"

__all__ = [
    "METHOD",
    "TemporalGramStatistics",
    "bounded_projectors",
    "diagnostics",
    "score_temporal_gram",
    "temporal_statistics",
]


def _denominator(s: Tensor, c: Tensor, r: Tensor) -> Tensor:
    # N1 checks ALL raw S/C before averaging; never call old _symmetric here.
    _, symmetric_c, _ = numerical._inputs(s, c, c)
    if s.requires_grad or c.requires_grad:
        raise ValueError("S/C must be detached support constants")
    _check_prior(r)
    if r.device.type != "cpu":
        raise ValueError("R must be CPU-only")
    try:
        r = torch.broadcast_to(r, s.shape[:-1])
    except RuntimeError as error:
        raise ValueError("R must broadcast to support matrix axes") from error
    minimum = torch.linalg.eigvalsh(symmetric_c)[..., 0]
    tau = ETA * minimum / R_MAX
    if not torch.isfinite(tau).all() or (tau <= 0).any():
        raise ValueError("positive penalty mass is numerically unrepresentable")
    # Keep the original raw C in B and in the N1 residual/normalization checks.
    return c + torch.diag_embed(tau.unsqueeze(-1) * r)


def _normalization_ratio(c: Tensor, b: Tensor, f: Tensor) -> Tensor:
    trace_c = torch.einsum("...ij,...ji->...", c, f)
    trace_b = torch.einsum("...ij,...ji->...", b, f)
    if (
        not torch.isfinite(trace_c).all()
        or not torch.isfinite(trace_b).all()
        or (trace_c <= 0).any()
        or (trace_b <= 0).any()
    ):
        raise ValueError("Invalid positive C/B projector trace")
    ratio = trace_c / trace_b
    if (
        not torch.isfinite(ratio).all()
        or (ratio < 1 / (1 + ETA) - 1e-10).any()
        or (ratio > 1 + 1e-10).any()
    ):
        raise ValueError("projector C/B trace violates the bounded-denominator identity")
    return ratio


def bounded_projectors(s: Tensor, c: Tensor, r: Tensor) -> Tensor:
    """N1(S, C+tau diag(R), C), with the prospectively frozen trace ratio guard."""
    b = _denominator(s, c, r)
    f = numerical.projector(s, b, c, METHOD)
    _normalization_ratio(c, b, f)
    return f


@torch.no_grad()
def diagnostics(s: Tensor, c: Tensor, r: Tensor, f: Tensor) -> dict[str, Tensor]:
    """Detached N1 metrics and tr(CF)/tr(BF); raw H skew is diagnostic only."""
    b = _denominator(s, c, r)
    result = numerical.diagnostics(s, b, c, f)
    result["normalization_ratio"] = _normalization_ratio(c, b, f)
    return {name: value.detach() for name, value in result.items()}
