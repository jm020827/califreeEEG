"""Isolated sign-free engineering candidate; NOT the frozen v1 human scorer.

Temporal centering removes each projected component's time mean before ensemble
Pearson. On inputs with nonzero channel means this changes native/global Pearson.
No data readers, training entry points, native-zero wrapper or execution adapters
are provided. Support matrices/statistics are detached; only R carries gradients.
The shared top projector implements first derivatives only, with a simple top root.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from cfeg.analysis.task_trca_shape_operator import (
    CHANNELS,
    ETA,
    R_MAX,
    _check_prior,
    _same_device,
    _scale_trials,
    _score_inputs,
    _score_outputs,
    _symmetric,
    _tensor,
    leading_projector,
)


def bounded_projectors(s: Tensor, c: Tensor, r: Tensor) -> Tensor:
    """Return C-normalized rank-one PSD matrices [...,8,8], without an anchor.

    B=C+tau diag(R), tau=.1 lambda_min(C)/(16/9), H=L^-1 S L^-T.
    For K=L^-T P_top(H) L^-1 return F=K/tr(CK). F equals ww^T with
    w^T C w=1. Unlike an anchor projection, tr(CK) lies in [1/1.1,1]
    in exact arithmetic. This does NOT bound the filter angle or top-root gap.
    """
    s, c = _symmetric(s, "s", constant=True), _symmetric(c, "c", constant=True)
    _check_prior(r)
    _same_device(s, ("c", c), ("r", r))
    if s.shape != c.shape:
        raise ValueError("s/c shapes must agree")
    try:
        r = torch.broadcast_to(r, s.shape[:-1])
    except RuntimeError as exc:
        raise ValueError("r does not broadcast to support matrix axes") from exc
    minimum = torch.linalg.eigvalsh(c)[..., 0]
    if not torch.isfinite(minimum).all() or (minimum <= 0).any():
        raise ValueError("c must be strictly positive definite; no jitter is permitted")
    tau = ETA * minimum / R_MAX
    if not torch.isfinite(tau).all() or (tau <= 0).any():
        raise ValueError("positive penalty mass is numerically unrepresentable")
    b = c + torch.diag_embed(tau.unsqueeze(-1) * r)
    _tensor(b, "bounded denominator")
    try:
        lower = torch.linalg.cholesky(b)
    except RuntimeError as exc:
        raise ValueError("bounded denominator Cholesky failed; no jitter is permitted") from exc
    left = torch.linalg.solve_triangular(lower, s, upper=False)
    h = torch.linalg.solve_triangular(lower, left.mT, upper=False).mT
    p = leading_projector(h)
    left = torch.linalg.solve_triangular(lower.mT, p, upper=True)
    k = torch.linalg.solve_triangular(lower.mT, left.mT, upper=True).mT
    k = (k + k.mT) / 2
    norm = torch.einsum("...ij,...ij->...", c, k)
    _tensor(norm, "projector C trace")
    if ((norm < 1 / (1 + ETA) - 1e-10) | (norm > 1 + 1e-10)).any():
        raise ValueError("projector C trace violates the bounded-denominator identity")
    result = k / norm[..., None, None]
    _tensor(result, "C-normalized projector")
    return result


@dataclass(frozen=True)
class TemporalGramStatistics:
    """Explicit NEW score contract; cannot be passed to old global-Pearson code.

    query_gram[n,b,8,8], template_gram[c,b,8,8], cross_gram[n,c,b,8,8].
    Each trial/band is positively scaled and time-centered per channel. No mean
    correction is stored. No claim is made that native preprocessing does this.
    """

    query_gram: Tensor
    template_gram: Tensor
    cross_gram: Tensor
    samples: int


def temporal_statistics(templates: Tensor, query: Tensor) -> TemporalGramStatistics:
    """Build centered covariance sums; a constant time series stays exactly zero.

    Subtracting the first sample before the mean is algebraically equivalent to
    centering. It prevents reduction-roundoff on a constant scaled value from
    creating artificial positive temporal variance. No variance floor is added.
    """
    _, _, samples, _ = _score_inputs(templates, query)
    t, x = _scale_trials(templates, "templates"), _scale_trials(query, "query")
    t, x = t - t[..., :1], x - x[..., :1]
    t, x = t - t.mean(-1, keepdim=True), x - x.mean(-1, keepdim=True)
    return TemporalGramStatistics(
        x @ x.mT, t @ t.mT, torch.einsum("nbit,cbjt->ncbij", x, t), samples
    )


def score_temporal_gram(
    projectors: Tensor, stats: TemporalGramStatistics, weights: Tensor
) -> tuple[Tensor, Tensor]:
    """Signed linear filterbank Pearson from J_b=sum_component F_b.

    Projectors have shape [bands,classes,8,8]. They must be symmetric PSD and
    nonzero. The constructor supplies rank-one C-normalized matrices. Centering
    is NOT per-component variance normalization or averaging component Pearson.
    """
    if not isinstance(stats, TemporalGramStatistics):
        raise TypeError("stats must be TemporalGramStatistics, not native/global statistics")
    x, t, xt = stats.query_gram, stats.template_gram, stats.cross_gram
    for name, value in (("query_gram", x), ("template_gram", t), ("cross_gram", xt)):
        _tensor(value, name, constant=True)
        _same_device(x, (name, value))
    if x.ndim != 4 or t.ndim != 4:
        raise ValueError("temporal Gram arrays must have trial,band,8,8 axes")
    n, bands = x.shape[:2]
    classes = t.shape[0]
    if (
        x.shape != (n, bands, CHANNELS, CHANNELS)
        or t.shape != (classes, bands, CHANNELS, CHANNELS)
        or xt.shape != (n, classes, bands, CHANNELS, CHANNELS)
        or isinstance(stats.samples, bool)
        or not isinstance(stats.samples, int)
        or stats.samples < 2
    ):
        raise ValueError("temporal Gram shapes or sample count disagree")
    projectors = _symmetric(projectors, "projectors")
    _tensor(weights, "weights", constant=True)
    _same_device(x, ("projectors", projectors), ("weights", weights))
    if projectors.shape != (bands, classes, CHANNELS, CHANNELS) or weights.shape != (bands,):
        raise ValueError("projectors/weights shapes must be (bands,classes,8,8)/(bands,)")
    if (weights <= 0).any():
        raise ValueError("band weights must be positive; signed refers to correlations")
    roots = torch.linalg.eigvalsh(projectors.detach())
    scale = roots.abs().amax(dim=-1)
    if (roots[..., 0] < -1e-12 * scale).any() or (roots.sum(dim=-1) <= 0).any():
        raise ValueError("projectors must be nonzero positive semidefinite")
    j = projectors.sum(dim=1)
    dot = torch.einsum("bij,ncbij->nbc", j, xt)
    xx = torch.einsum("bij,nbij->nb", j, x)
    tt = torch.einsum("bij,cbij->bc", j, t)
    return _score_outputs(dot, xx, tt, weights)
