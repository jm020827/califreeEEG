"""Differentiable k>=1 reference ridge primitives; no data reader or trained model."""

import math

import torch


def _require(ok: bool, reason: str) -> None:
    if not ok:
        raise ValueError(reason)


def _finite(value: torch.Tensor, ndim: int) -> None:
    _require(value.ndim == ndim and value.is_floating_point(), "tensor_shape_or_dtype")
    _require(bool(torch.isfinite(value).all()), "nonfinite_tensor")


def center(value: torch.Tensor) -> torch.Tensor:
    return value - value.mean(dim=-1, keepdim=True)


def references(frequency: float, fs: float, samples: int, harmonics: int = 3) -> torch.Tensor:
    _require(
        math.isfinite(frequency) and math.isfinite(fs) and 0 < frequency * harmonics < fs / 2,
        "reference_frequency",
    )
    _require(
        isinstance(samples, int) and samples > 2 * harmonics and harmonics > 0, "reference_samples"
    )
    time = torch.arange(samples, dtype=torch.float64) / fs
    phase = (
        2
        * math.pi
        * frequency
        * torch.arange(1, harmonics + 1, dtype=torch.float64)[:, None]
        * time
    )
    return center(torch.stack((phase.sin(), phase.cos()), dim=1).reshape(2 * harmonics, samples))


def q_diagonal(logits: torch.Tensor, epsilon: float = 0.25) -> torch.Tensor:
    _finite(logits, 1)
    _require(logits.numel() > 0 and 0 < epsilon < 1, "prior_arguments")
    return epsilon + (1 - epsilon) * logits.numel() * torch.softmax(logits, dim=0)


def residual_diagonal(
    q_diag: torch.Tensor, logits: torch.Tensor, epsilon: float = 0.25, alpha: float = 0.4
) -> torch.Tensor:
    _finite(q_diag, 1)
    _finite(logits, 1)
    _require(q_diag.shape == logits.shape and q_diag.numel() > 0, "residual_shape")
    _require(0 < epsilon < 1 and 0 <= alpha < 1, "residual_bounds")
    _require(bool((q_diag >= epsilon).all()), "q_floor_not_satisfied")
    v = torch.tanh(logits)
    return q_diag.detach() + (alpha * epsilon / 2) * (v - v.mean())


def fit_reference_ridge(
    support: torch.Tensor,
    reference: torch.Tensor,
    prior: torch.Tensor,
    ridge: float,
    energy_floor: float = 1e-12,
) -> torch.Tensor:
    _finite(support, 3)
    _finite(reference, 2)
    _finite(prior, 2)
    k, channels, samples = support.shape
    _require(k >= 1 and channels >= 1 and samples >= 2, "empty_support")
    _require(reference.shape[1] == samples and prior.shape == (channels, channels), "ridge_shapes")
    _require(
        math.isfinite(ridge) and ridge > 0 and math.isfinite(energy_floor) and energy_floor > 0,
        "ridge_scale",
    )
    _require(bool(torch.allclose(prior, prior.T, atol=1e-12, rtol=0)), "prior_not_symmetric")
    _require(bool((torch.linalg.cholesky_ex(prior).info == 0).all()), "prior_not_spd")
    x = center(support).transpose(0, 1).reshape(channels, k * samples)
    y = center(reference).repeat(1, k)
    covariance = (x @ x.T) / (k * samples)
    cross = (x @ y.T) / (k * samples)
    scale = torch.diagonal(covariance).mean().clamp_min(energy_floor)
    return torch.linalg.solve(covariance + ridge * scale * prior, cross)


def reference_projection_score(
    weights: torch.Tensor, query: torch.Tensor, reference: torch.Tensor, energy_floor: float = 1e-12
) -> torch.Tensor:
    _finite(weights, 2)
    _finite(query, 2)
    _finite(reference, 2)
    _require(
        weights.shape == (query.shape[0], reference.shape[0])
        and query.shape[1] == reference.shape[1],
        "score_shapes",
    )
    _require(math.isfinite(energy_floor) and energy_floor > 0, "score_floor")
    y = center(reference)
    gram = y @ y.T
    _require(bool((torch.linalg.cholesky_ex(gram).info == 0).all()), "reference_gram_not_spd")
    z = weights.T @ center(query)
    cross = z @ y.T
    numerator = (cross * torch.linalg.solve(gram, cross.T).T).sum()
    energy = z.square().sum()
    score = torch.where(
        energy > energy_floor, numerator / energy.clamp_min(energy_floor), energy * 0
    )
    _require(
        bool(torch.isfinite(score)) and bool(score >= -1e-8) and bool(score <= 1 + 1e-8),
        "projection_score_out_of_range",
    )
    return score.clamp(0, 1)
