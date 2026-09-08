"""Synthetic-gated differentiable operator for the frozen task-shape v1 design.

This module has no data reader or executable-study entry point. Support S/C,
native anchors, and score statistics are constants; only the bounded shape and
filters carry learning gradients. No numerical rescue or sample dropping is
performed. The custom projector supports first derivatives only.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch
from torch import Tensor
from torch.autograd.function import once_differentiable

CHANNELS = 8
LOGIT_BOUND = math.log(2.0) / 2.0
R_MAX = 16.0 / 9.0
ETA = 0.1
TOP_GAP_MIN = 1e-10
ANCHOR_COSINE_MIN = 1e-6


def _tensor(value: Tensor, name: str, *, constant: bool = False) -> Tensor:
    if not isinstance(value, Tensor) or value.dtype != torch.float64:
        raise ValueError(f"{name} must be a float64 tensor")
    if any(size == 0 for size in value.shape) or not torch.isfinite(value).all():
        raise ValueError(f"{name} must be nonempty and finite")
    if constant and value.requires_grad:
        raise ValueError(f"{name} must be a detached support/statistics constant")
    return value


def _same_device(reference: Tensor, *named: tuple[str, Tensor]) -> None:
    for name, value in named:
        if value.device != reference.device:
            raise ValueError(f"{name} must be on the same device")


def _symmetric(value: Tensor, name: str, *, constant: bool = False) -> Tensor:
    value = _tensor(value, name, constant=constant)
    if value.ndim < 2 or value.shape[-2:] != (CHANNELS, CHANNELS):
        raise ValueError(f"{name} must have final shape (8,8)")
    scale = torch.linalg.matrix_norm(value, ord="fro", dim=(-2, -1))
    error = torch.linalg.matrix_norm(value - value.transpose(-1, -2), ord="fro", dim=(-2, -1))
    if not torch.isfinite(scale).all() or not torch.isfinite(error).all():
        raise ValueError(f"{name} symmetry arithmetic is nonfinite")
    if (error > 1e-12 * torch.clamp_min(scale, 1.0)).any():
        raise ValueError(f"{name} must be symmetric within the frozen tolerance")
    return (value + value.transpose(-1, -2)) / 2.0


class _LeadingProjector(torch.autograd.Function):
    @staticmethod
    def forward(ctx, h: Tensor) -> Tensor:
        try:
            values, vectors = torch.linalg.eigh(h)
        except RuntimeError as exc:
            raise ValueError("leading projector eigensolver failed") from exc
        _tensor(values, "eigenvalues")
        _tensor(vectors, "eigenvectors")
        gaps = values[..., -1:] - values[..., :-1]
        tolerance = TOP_GAP_MIN * torch.clamp_min(values.abs().amax(dim=-1), 1.0)
        if (gaps[..., -1] <= tolerance).any():
            raise ValueError("leading eigenspace is degenerate or nearly degenerate")
        ctx.save_for_backward(vectors, gaps)
        top = vectors[..., -1]
        return top.unsqueeze(-1) * top.unsqueeze(-2)

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output: Tensor) -> tuple[Tensor]:
        _tensor(grad_output, "projector upstream gradient")
        vectors, gaps = ctx.saved_tensors
        top = vectors[..., -1]
        lower = vectors[..., :-1]
        action = (grad_output + grad_output.transpose(-1, -2)) @ top.unsqueeze(-1)
        coefficients = (lower.transpose(-1, -2) @ action).squeeze(-1) / gaps
        h = (lower @ coefficients.unsqueeze(-1)).squeeze(-1)
        grad = (h.unsqueeze(-1) * top.unsqueeze(-2) + top.unsqueeze(-1) * h.unsqueeze(-2)) / 2.0
        _tensor(grad, "projector gradient")
        return (grad,)


def leading_projector(h: Tensor) -> Tensor:
    """Return the top rank-one projector, allowing repeated *lower* roots.

    Input is symmetric float64 [...,8,8]. Gradcheck must parameterize symmetric
    perturbations, e.g. ``lambda x: leading_projector((x+x.mT)/2)``. Gradients
    involve only top-versus-rest gaps, not differences among lower roots.
    """
    return _LeadingProjector.apply(_symmetric(h, "h"))


def shape_prior(logits: Tensor) -> Tensor:
    """Map bounded logits [...,8] to a positive trace-eight diagonal shape."""
    logits = _tensor(logits, "logits")
    if logits.ndim < 1 or logits.shape[-1] != CHANNELS:
        raise ValueError("logits must have a final eight-channel axis")
    if (logits.abs() > LOGIT_BOUND + 1e-15).any():
        raise ValueError("logits exceed the frozen log(2)/2 bound")
    result = CHANNELS * torch.softmax(logits, dim=-1)
    _check_prior(result)
    return result


def _check_prior(r: Tensor) -> None:
    r = _tensor(r, "r")
    if r.ndim < 1 or r.shape[-1] != CHANNELS:
        raise ValueError("r must have a final eight-channel axis")
    if (r <= 0).any():
        raise ValueError("r must be strictly positive")
    if ((r.sum(dim=-1) - CHANNELS).abs() > 8e-12).any():
        raise ValueError("r must have trace eight")
    if (r.amax(dim=-1) / r.amin(dim=-1) > 2.0 + 1e-12).any():
        raise ValueError("r max/min ratio exceeds two")
    if (r > R_MAX + 1e-12).any():
        raise ValueError("r exceeds the frozen 16/9 maximum")


def bounded_filters(
    s: Tensor, c: Tensor, anchors: Tensor, r: Tensor, *, eta: float = ETA
) -> Tensor:
    """Return anchored C-normalized filters [...,8] for fixed positive eta=.1.

    S/C have identical shape [...,8,8], native FULL anchors [...,8], and R
    broadcasts to [...,8]. Tau depends only on C and the fixed bound, never on
    actual R. Eta zero is deliberately a separate native wrapper, not a
    numerically reconstructed version of the old solver.
    """
    if isinstance(eta, bool) or not isinstance(eta, (int, float)) or eta != ETA:
        raise ValueError("bounded_filters requires frozen eta=0.1; use native_zero_scores for zero")
    s = _symmetric(s, "s", constant=True)
    c = _symmetric(c, "c", constant=True)
    anchors = _tensor(anchors, "anchors", constant=True)
    _check_prior(r)
    _same_device(s, ("c", c), ("anchors", anchors), ("r", r))
    if s.shape != c.shape or anchors.shape != s.shape[:-1]:
        raise ValueError("S/C/anchor shapes must agree exactly")
    try:
        r = torch.broadcast_to(r, anchors.shape)
    except RuntimeError as exc:
        raise ValueError("r must broadcast to the support class/band/channel axes") from exc
    try:
        minimum = torch.linalg.eigvalsh(c)[..., 0]
    except RuntimeError as exc:
        raise ValueError("C eigensolver failed") from exc
    if not torch.isfinite(minimum).all() or (minimum <= 0).any():
        raise ValueError("C must be positive definite")
    anchor_norm2 = torch.einsum("...i,...ij,...j->...", anchors, c, anchors)
    if not torch.isfinite(anchor_norm2).all() or (anchor_norm2 <= 0).any():
        raise ValueError("anchor must have positive finite C norm")
    anchor = anchors / anchor_norm2.sqrt().unsqueeze(-1)
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
    h = torch.linalg.solve_triangular(lower, left.transpose(-1, -2), upper=False).transpose(-1, -2)
    projector = leading_projector(h)
    anchor_h = torch.linalg.solve_triangular(lower, (c @ anchor.unsqueeze(-1)), upper=False)
    raw = torch.linalg.solve_triangular(
        lower.transpose(-1, -2), projector @ anchor_h, upper=True
    ).squeeze(-1)
    norm2 = torch.einsum("...i,...ij,...j->...", raw, c, raw)
    if not torch.isfinite(norm2).all() or (norm2 <= 0).any():
        raise ValueError("anchor projection has nonpositive or nonfinite C norm")
    norm = norm2.sqrt()
    cosine = torch.einsum("...i,...ij,...j->...", raw, c, anchor).abs() / norm
    if not torch.isfinite(cosine).all() or (cosine <= ANCHOR_COSINE_MIN).any():
        raise ValueError("native anchor is nearly C-orthogonal to the leading direction")
    result = raw / norm.unsqueeze(-1)
    _tensor(result, "C-normalized filters")
    return result


@dataclass(frozen=True)
class GramStatistics:
    """Detached score sufficient statistics; all covariance sums are unscaled.

    Array axes: query_gram[n,b,i,j], template_gram[c,b,i,j],
    cross_gram[n,c,b,i,j], query_mean[n,b,i], template_mean[c,b,i].
    Inputs are independently divided by their positive max-absolute value per
    query/template and band before computing these statistics. Pearson and its
    filter derivative are unchanged; scale factors are not learned or reused.
    """

    query_gram: Tensor
    template_gram: Tensor
    cross_gram: Tensor
    query_mean: Tensor
    template_mean: Tensor
    samples: int


def _score_inputs(templates: Tensor, query: Tensor) -> tuple[int, int, int, int]:
    templates = _tensor(templates, "templates", constant=True)
    query = _tensor(query, "query", constant=True)
    _same_device(templates, ("query", query))
    if templates.ndim != 4 or templates.shape[2] != CHANNELS or templates.shape[3] < 2:
        raise ValueError("templates must have shape (classes,bands,8,samples>=2)")
    classes, bands, _, samples = templates.shape
    if query.ndim != 4 or query.shape[1:] != (bands, CHANNELS, samples):
        raise ValueError("query must have shape (n,bands,8,samples) matching templates")
    return classes, bands, samples, query.shape[0]


def _scale_trials(value: Tensor, name: str) -> Tensor:
    maximum = value.abs().amax(dim=(-2, -1), keepdim=True)
    if (maximum <= 0).any():
        raise ValueError(f"{name} has zero input energy")
    return value / maximum


def gram_statistics(templates: Tensor, query: Tensor) -> GramStatistics:
    """Build detached, stable global-Pearson statistics without changing means."""
    _, _, samples, _ = _score_inputs(templates, query)
    t = _scale_trials(templates, "templates")
    x = _scale_trials(query, "query")
    mt, mx = t.mean(dim=-1), x.mean(dim=-1)
    tc, xc = t - mt.unsqueeze(-1), x - mx.unsqueeze(-1)
    return GramStatistics(
        query_gram=xc @ xc.transpose(-1, -2),
        template_gram=tc @ tc.transpose(-1, -2),
        cross_gram=torch.einsum("nbit,cbjt->ncbij", xc, tc),
        query_mean=mx,
        template_mean=mt,
        samples=samples,
    )


def _scoring_filters(
    filters: Tensor, weights: Tensor, classes: int, bands: int, reference: Tensor
) -> None:
    _tensor(filters, "filters")
    _tensor(weights, "weights", constant=True)
    _same_device(reference, ("filters", filters), ("weights", weights))
    if filters.shape != (bands, CHANNELS, classes) or weights.shape != (bands,):
        raise ValueError("filters/weights must have shapes (bands,8,classes)/(bands,)")


def _score_outputs(dot: Tensor, xx: Tensor, tt: Tensor, weights: Tensor) -> tuple[Tensor, Tensor]:
    _tensor(dot, "projected cross product")
    _tensor(xx, "query projected variance")
    _tensor(tt, "template projected variance")
    if (xx <= 0).any() or (tt <= 0).any():
        raise ValueError(
            "query/template has nonpositive projected variance; no variance floor is permitted"
        )
    # Take the square roots separately to avoid an unnecessary xx*tt overflow.
    correlation = dot / xx.sqrt().unsqueeze(-1) / tt.sqrt().unsqueeze(0)
    _tensor(correlation, "correlations before clipping")
    correlation = torch.clamp(correlation, -1.0, 1.0)
    scores = torch.einsum("nbc,b->nc", correlation, weights)
    _tensor(scores, "weighted scores")
    return scores, correlation


def score_gram(filters: Tensor, stats: GramStatistics, weights: Tensor) -> tuple[Tensor, Tensor]:
    """Score native signed ensemble Pearson from detached Gram statistics."""
    if not isinstance(stats, GramStatistics):
        raise TypeError("stats must be GramStatistics")
    for name in ("query_gram", "template_gram", "cross_gram", "query_mean", "template_mean"):
        _tensor(getattr(stats, name), name, constant=True)
        _same_device(stats.query_mean, (name, getattr(stats, name)))
    if stats.query_mean.ndim != 3 or stats.template_mean.ndim != 3:
        raise ValueError("Gram means must have query/template,band,channel axes")
    n, bands, channels = stats.query_mean.shape
    classes = stats.template_mean.shape[0]
    if (
        channels != CHANNELS
        or stats.template_mean.shape != (classes, bands, CHANNELS)
        or stats.query_gram.shape != (n, bands, CHANNELS, CHANNELS)
        or stats.template_gram.shape != (classes, bands, CHANNELS, CHANNELS)
        or stats.cross_gram.shape != (n, classes, bands, CHANNELS, CHANNELS)
        or isinstance(stats.samples, bool)
        or not isinstance(stats.samples, int)
        or stats.samples < 2
    ):
        raise ValueError("Gram statistics shapes or sample count disagree")
    _scoring_filters(filters, weights, classes, bands, stats.query_mean)
    j = filters @ filters.transpose(-1, -2)
    centered_filters = filters - filters.mean(dim=-1, keepdim=True)
    h = centered_filters @ centered_filters.transpose(-1, -2)
    mx, mt = stats.query_mean, stats.template_mean
    dot = torch.einsum("bij,ncbij->nbc", j, stats.cross_gram)
    dot = dot + stats.samples * torch.einsum("nbi,bij,cbj->nbc", mx, h, mt)
    xx = torch.einsum("bij,nbij->nb", j, stats.query_gram)
    xx = xx + stats.samples * torch.einsum("nbi,bij,nbj->nb", mx, h, mx)
    tt = torch.einsum("bij,cbij->bc", j, stats.template_gram)
    tt = tt + stats.samples * torch.einsum("cbi,bij,cbj->bc", mt, h, mt)
    return _score_outputs(dot, xx, tt, weights)


def score_flattened(
    filters: Tensor, templates: Tensor, query: Tensor, weights: Tensor
) -> tuple[Tensor, Tensor]:
    """Independent literal flattened ensemble Pearson for numerical checks."""
    classes, bands, _, _ = _score_inputs(templates, query)
    _scoring_filters(filters, weights, classes, bands, templates)
    x = torch.einsum("bik,nbit->nbkt", filters, _scale_trials(query, "query")).flatten(start_dim=2)
    t = torch.einsum("bik,cbit->cbkt", filters, _scale_trials(templates, "templates")).flatten(
        start_dim=2
    )
    _tensor(x, "projected query")
    _tensor(t, "projected template")
    x = x - x.mean(dim=-1, keepdim=True)
    t = t - t.mean(dim=-1, keepdim=True)
    dot = torch.einsum("nbt,cbt->nbc", x, t)
    xx = (x * x).sum(dim=-1)
    tt = (t * t).sum(dim=-1).transpose(0, 1)
    return _score_outputs(dot, xx, tt, weights)


def native_zero_scores(support, query, weights):
    """Exact delegation to the unchanged project gamma-zero fit/score path.

    This NumPy reference wrapper does not enter the differentiable positive-mass
    path and is not equivalent to a zero shape head (which produces ISO).
    """
    from cfeg.analysis.metadata_trca_prior import fit_trca, score_trca

    raw = np.asarray(support)
    if raw.ndim != 5 or raw.shape[3] != CHANNELS:
        raise ValueError("support must have shape (k,classes,bands,8,samples)")
    model = fit_trca(support, np.ones((raw.shape[2], CHANNELS), dtype=np.float64), gamma=0.0)
    return score_trca(model, query, weights)
