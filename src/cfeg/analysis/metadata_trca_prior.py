"""Pure-array, trace-matched regularization for the synthetic TRCA prior study.

The S/C matrices, SciPy generalized eigenvector convention, unweighted templates,
and signed flattened ensemble correlation follow SSVEP-Analysis-Toolbox revision
3344bd199daf78888e364d9db00ae7d8128d2b5f. This module neither preprocesses EEG nor
loads any data. A diagonal prior is a statistical penalty, not measured noise.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy import linalg

FloatArray = NDArray[np.float64]
_TINY = np.finfo(np.float64).tiny
_TRACE_RTOL = 1e-12


def _real_array(value: ArrayLike, name: str, *, finite: bool = True) -> FloatArray:
    raw = np.asarray(value)
    if raw.dtype.kind not in "fiu" or np.iscomplexobj(raw):
        raise ValueError(f"{name} must contain real numeric values")
    result = np.asarray(raw, dtype=np.float64)
    if finite and not np.isfinite(result).all():
        raise ValueError(f"{name} must contain only finite values")
    return result


def _scalar(value: float, name: str) -> float:
    result = _real_array(value, name)
    if result.ndim != 0 or result < 0:
        raise ValueError(f"{name} must be a finite nonnegative scalar")
    return float(result)


def _check_diagonal(prior: FloatArray, name: str) -> None:
    if prior.ndim < 1 or any(size == 0 for size in prior.shape):
        raise ValueError(f"{name} must have a nonempty channel axis")
    if np.any(prior <= 0):
        raise ValueError(f"{name} must be strictly positive")
    if not np.allclose(prior.mean(axis=-1), 1.0, rtol=_TRACE_RTOL, atol=0.0):
        raise ValueError(f"{name} trace must equal the channel count")


def trace_normalize(values: ArrayLike) -> FloatArray:
    """Return positive diagonals with trace d, preserving arbitrary leading axes.

    Scale before summing to avoid overflow. Dynamic ranges too large to retain
    strictly positive float64 outputs are rejected instead of silently floored.
    """
    diagonal = _real_array(values, "values")
    if diagonal.ndim < 1 or any(size == 0 for size in diagonal.shape):
        raise ValueError("values must have a nonempty channel axis")
    if np.any(diagonal <= 0):
        raise ValueError("values must be strictly positive")
    scaled = diagonal / diagonal.max(axis=-1, keepdims=True)
    normalized = scaled / scaled.mean(axis=-1, keepdims=True)
    if not np.isfinite(normalized).all() or np.any(normalized <= 0):
        raise ValueError("values have an unrepresentable float64 dynamic range")
    return normalized


def residual_prior(
    q_prior: ArrayLike, delta: ArrayLike, available: ArrayLike, bound: float = 0.2
) -> FloatArray:
    """Apply a bounded log residual, with exact all-missing band fallback.

    Masks must be boolean and have exactly the diagonal shape; no broadcasting
    can accidentally align a different channel. NaN is accepted only at missing
    delta entries. Common trace normalization may change partially missing
    channels. The final QM/Q ratio is bounded by exp(+/-2*bound).
    """
    q = _real_array(q_prior, "q_prior")
    _check_diagonal(q, "q_prior")
    residual = _real_array(delta, "delta", finite=False)
    mask = np.asarray(available)
    if mask.dtype != np.bool_:
        raise ValueError("available must have boolean dtype")
    if residual.shape != q.shape or mask.shape != q.shape:
        raise ValueError("q_prior, delta and available must have identical shapes")
    if np.isinf(residual).any() or not np.isfinite(residual[mask]).all():
        raise ValueError("delta must be finite when available and never infinite")
    limit = _scalar(bound, "bound")
    clipped = np.where(mask, np.clip(residual, -limit, limit), 0.0)
    # Subtract a bandwise maximum before exponentiating. This cancels under
    # trace normalization and avoids overflow at large, still-finite bounds.
    log_prior = np.log(q) + clipped
    log_prior -= log_prior.max(axis=-1, keepdims=True)
    result = trace_normalize(np.exp(log_prior))
    all_missing = ~mask.any(axis=-1, keepdims=True)
    result = np.where(all_missing, q, result)
    return result


def trca_matrices(trials: ArrayLike) -> tuple[FloatArray, FloatArray]:
    """Return native cross-repeat S and concatenated-temporal-centered C.

    trials has shape (repeats, channels, samples), with at least two entries on
    each axis. Individual trials are deliberately not centered before forming S.
    """
    x = _real_array(trials, "trials")
    if x.ndim != 3 or any(size < 2 for size in x.shape):
        raise ValueError("trials must have shape (k>=2, channels>=2, samples>=2)")
    summed = np.zeros_like(x[0])
    for repeat in x:
        summed = summed + repeat
    concatenated = np.concatenate([repeat.T for repeat in x], axis=0)
    with np.errstate(over="ignore", invalid="ignore", under="ignore"):
        s = summed @ summed.T - concatenated.T @ concatenated
        centered = concatenated - concatenated.mean(axis=0)
        c = centered.T @ centered
    if not np.isfinite(s).all() or not np.isfinite(c).all():
        raise ValueError("TRCA matrix arithmetic produced nonfinite values")
    energy = float(np.trace(c))
    if not np.isfinite(energy) or energy <= _TINY:
        raise ValueError("trials have zero or numerically unrepresentable centered energy")
    return s, c


@dataclass(frozen=True)
class TrcaModel:
    """Ensemble filters [band,channel,class], unweighted templates and diagnostics."""

    filters: FloatArray
    templates: FloatArray
    diagnostics: dict[str, object]


def fit_trca(support: ArrayLike, prior: ArrayLike, gamma: float) -> TrcaModel:
    """Fit classwise generalized TRCA filters with a class-blind diagonal prior.

    The denominator is C + gamma*trace(C)/channels*diag(prior). No jitter,
    pseudoinverse, eigenspace tie breaker, or automatic ridge rescue is applied.
    """
    x = _real_array(support, "support")
    if (
        x.ndim != 5
        or x.shape[0] < 2
        or x.shape[1] < 1
        or x.shape[2] < 1
        or x.shape[3] < 2
        or x.shape[4] < 2
    ):
        raise ValueError("support must have shape (k>=2, classes, bands, channels>=2, samples>=2)")
    _, classes, bands, channels, _ = x.shape
    diagonal = _real_array(prior, "prior")
    if diagonal.shape != (bands, channels):
        raise ValueError("prior must have shape (bands, channels)")
    _check_diagonal(diagonal, "prior")
    strength = _scalar(gamma, "gamma")
    filters = np.empty((bands, channels, classes), dtype=np.float64)
    top_eigenvalues = np.empty((classes, bands), dtype=np.float64)
    top_gaps = np.empty_like(top_eigenvalues)
    denominator_min = np.empty_like(top_eigenvalues)
    norm_error = np.empty_like(top_eigenvalues)
    penalty_trace = np.empty_like(top_eigenvalues)
    for label in range(classes):
        for band in range(bands):
            s, c = trca_matrices(x[:, label, band])
            with np.errstate(over="ignore", invalid="ignore"):
                penalty = strength * (np.trace(c) / channels) * diagonal[band]
                denominator = c + np.diag(penalty)
            if not np.isfinite(denominator).all():
                raise ValueError("TRCA denominator contains nonfinite values")
            minimum = float(linalg.eigvalsh(denominator)[0])
            if minimum <= 0:
                raise ValueError("TRCA denominator must be positive definite")
            eigenvalues, eigenvectors = linalg.eig(s, denominator)
            if (
                not np.isfinite(eigenvalues).all()
                or not np.isfinite(eigenvectors).all()
                or np.iscomplex(eigenvalues).any()
                or np.iscomplex(eigenvectors).any()
            ):
                raise ValueError("TRCA eigensystem must be finite and real")
            order = np.argsort(eigenvalues)[::-1]
            sorted_values = np.real(eigenvalues[order])
            gap = float(sorted_values[0] - sorted_values[1])
            tolerance = 1e-10 * max(1.0, float(np.abs(sorted_values).max()))
            if gap <= tolerance:
                raise ValueError("TRCA leading eigenspace is degenerate or nearly degenerate")
            vector = np.real(eigenvectors[:, order[0]])
            square_norm = float(vector @ denominator @ vector)
            if not np.isfinite(square_norm) or square_norm <= 0:
                raise ValueError("TRCA filter has nonpositive or nonfinite denominator norm")
            vector = vector / np.sqrt(square_norm)
            if not np.isfinite(vector).all():
                raise ValueError("TRCA filter normalization produced nonfinite values")
            filters[band, :, label] = vector
            top_eigenvalues[label, band] = sorted_values[0]
            top_gaps[label, band] = gap
            denominator_min[label, band] = minimum
            norm_error[label, band] = abs(float(vector @ denominator @ vector) - 1.0)
            penalty_trace[label, band] = float(penalty.sum())
    templates = x.mean(axis=0)
    if not np.isfinite(templates).all():
        raise ValueError("TRCA template arithmetic produced nonfinite values")
    diagnostics: dict[str, object] = {
        "gamma": strength,
        "prior_trace": diagonal.sum(axis=-1).tolist(),
        "top_eigenvalues": top_eigenvalues.tolist(),
        "top_eigenvalue_gaps": top_gaps.tolist(),
        "denominator_min_eigenvalues": denominator_min.tolist(),
        "denominator_norm_errors": norm_error.tolist(),
        "penalty_trace": penalty_trace.tolist(),
    }
    return TrcaModel(filters=filters, templates=templates, diagnostics=diagnostics)


def _unit_centered(vector: FloatArray, name: str) -> FloatArray:
    if not np.isfinite(vector).all():
        raise ValueError(f"{name} projected values must be finite")
    # Scaling before centering protects against overflow/underflow in variance.
    maximum = float(np.max(np.abs(vector)))
    if maximum == 0:
        raise ValueError(f"{name} has zero projected variance")
    centered = vector / maximum
    centered = centered - centered.mean()
    norm = float(np.linalg.norm(centered))
    if norm == 0 or not np.isfinite(norm):
        raise ValueError(f"{name} has zero or nonfinite projected variance")
    return centered / norm


def score_trca(
    model: TrcaModel, query: ArrayLike, filter_weights: ArrayLike
) -> tuple[FloatArray, FloatArray]:
    """Return signed linear scores [n,class] and correlations [n,band,class]."""
    if not isinstance(model, TrcaModel):
        raise TypeError("model must be a TrcaModel")
    filters = _real_array(model.filters, "model.filters")
    templates = _real_array(model.templates, "model.templates")
    if templates.ndim != 4 or any(size == 0 for size in templates.shape):
        raise ValueError("model.templates must have shape (classes, bands, channels, samples)")
    classes, bands, channels, samples = templates.shape
    if channels < 2 or samples < 2 or filters.shape != (bands, channels, classes):
        raise ValueError("model filter and template dimensions disagree")
    x = _real_array(query, "query")
    if x.ndim != 4 or x.shape[0] < 1 or x.shape[1:] != (bands, channels, samples):
        raise ValueError("query must have shape (n>=1, bands, channels, samples) matching model")
    weights = _real_array(filter_weights, "filter_weights")
    if weights.shape != (bands,):
        raise ValueError("filter_weights must have shape (bands,)")
    projected_templates = np.empty((classes, bands, classes * samples), dtype=np.float64)
    for label in range(classes):
        for band in range(bands):
            with np.errstate(over="ignore", invalid="ignore"):
                projected = (filters[band].T @ templates[label, band]).reshape(-1)
            projected_templates[label, band] = _unit_centered(projected, "template")
    correlations = np.empty((len(x), bands, classes), dtype=np.float64)
    for index, trial in enumerate(x):
        for band in range(bands):
            with np.errstate(over="ignore", invalid="ignore"):
                projected = (filters[band].T @ trial[band]).reshape(-1)
            unit = _unit_centered(projected, "query")
            correlations[index, band] = np.clip(projected_templates[:, band] @ unit, -1, 1)
    with np.errstate(over="ignore", invalid="ignore"):
        scores = np.einsum("nbc,b->nc", correlations, weights)
    if not np.isfinite(scores).all():
        raise ValueError("weighted TRCA scores must be finite")
    return scores, correlations
