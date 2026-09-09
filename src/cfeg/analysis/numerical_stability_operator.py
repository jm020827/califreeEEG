"""Two prospectively frozen, CPU-only numerical formulations; no study I/O.

These operators do not reopen C1/C2 or authorize human input. All raw inputs
pass finite/symmetry checks before averaging; strict SPD is then checked on
the accepted symmetric B/C, never obtained with jitter. Residuals refer to the
original supplied matrices. Only first derivatives are supported.
"""

from __future__ import annotations

import numpy as np
import torch
from scipy import linalg
from torch import Tensor
from torch.autograd.function import once_differentiable

from cfeg.analysis.task_trca_shape_operator import leading_projector

METHODS = ("N1_SYM_CHOLESKY", "N2_GVD_IMPLICIT")
SYMMETRY_MAX = 1e-12
TOP_GAP_MIN = 1e-10
RESIDUAL_MAX = 1e-12
TINY = torch.finfo(torch.float64).tiny


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _tensor(value, name):
    _require(
        isinstance(value, Tensor) and value.dtype == torch.float64,
        f"{name} must be a float64 tensor",
    )
    _require(value.device.type == "cpu", f"{name} must be CPU-only")
    _require(
        value.ndim >= 2 and value.shape[-2:] == (8, 8) and all(value.shape),
        f"{name} must be nonempty [...,8,8]",
    )
    _require(bool(torch.isfinite(value).all()), f"{name} must be finite")
    return value


def _sym(value):
    return (value + value.mT) / 2


def _fro(value):
    return torch.linalg.matrix_norm(value, ord="fro", dim=(-2, -1))


def _skew(value):
    norm, error = _fro(value), _fro(value - value.mT)
    _require(
        bool(torch.isfinite(norm).all() and torch.isfinite(error).all()),
        "Nonfinite symmetry arithmetic",
    )
    return error / torch.clamp_min(norm, TINY)


def _inputs(s, b, c):
    for name, value in (("s", s), ("b", b), ("c", c)):
        _tensor(value, name)
    _require(s.shape == b.shape == c.shape, "S/B/C shapes must agree without broadcasting")
    for name, value in (("s", s), ("b", b), ("c", c)):
        _require(
            bool((_skew(value) <= SYMMETRY_MAX).all()),
            f"{name} symmetry exceeds frozen relative tolerance",
        )
    # No raw input is averaged until every input symmetry check has passed.
    s, b, c = _sym(s), _sym(b), _sym(c)
    for name, value in (("b", b), ("c", c)):
        try:
            roots = torch.linalg.eigvalsh(value)
        except RuntimeError as error:
            raise ValueError(f"{name} SPD eigensolver failed") from error
        _require(
            bool(torch.isfinite(roots).all() and (roots[..., 0] > 0).all()),
            f"{name} must be strictly positive definite; no jitter",
        )
    return s, b, c


def _trace_product(left, right):
    return torch.einsum("...ij,...ji->...", left, right)


def _raw_h(s, b):
    lower = torch.linalg.cholesky(b)
    left = torch.linalg.solve_triangular(lower, s, upper=False)
    raw = torch.linalg.solve_triangular(lower, left.mT, upper=False).mT
    _require(bool(torch.isfinite(raw).all()), "Nonfinite raw H")
    return lower, raw


def _n1(s, b):
    lower, raw = _raw_h(s, b)
    # The new explicit correction is prospective and never hides input skew.
    p = leading_projector(_sym(raw))
    left = torch.linalg.solve_triangular(lower.mT, p, upper=True)
    k = torch.linalg.solve_triangular(lower.mT, left.mT, upper=True).mT
    return _sym(k)


class _GeneralizedProjector(torch.autograd.Function):
    @staticmethod
    def forward(ctx, s, b):
        roots = np.empty((*s.shape[:-2], 8), dtype=np.float64)
        vectors = np.empty(tuple(s.shape), dtype=np.float64)
        for index, (ss, bb) in enumerate(
            zip(
                s.detach().numpy().reshape(-1, 8, 8),
                b.detach().numpy().reshape(-1, 8, 8),
                strict=True,
            )
        ):
            try:
                values, basis = linalg.eigh(
                    ss,
                    bb,
                    type=1,
                    lower=True,
                    driver="gvd",
                    check_finite=True,
                    overwrite_a=False,
                    overwrite_b=False,
                )
            except (ValueError, linalg.LinAlgError) as error:
                raise ValueError("N2 generalized eigensolver failed; no fallback") from error
            roots.reshape(-1, 8)[index] = values
            vectors.reshape(-1, 8, 8)[index] = basis
        values, basis = torch.from_numpy(roots), torch.from_numpy(vectors)
        _require(
            bool(torch.isfinite(values).all() and torch.isfinite(basis).all()),
            "Nonfinite generalized eigensystem",
        )
        gaps = values[..., -1:] - values[..., :-1]
        tolerance = TOP_GAP_MIN * torch.clamp_min(values.abs().amax(-1), 1.0)
        _require(
            bool((gaps[..., -1] > tolerance).all()),
            "leading eigenspace is degenerate or nearly degenerate",
        )
        ctx.save_for_backward(values, basis)
        top = basis[..., -1]
        return top[..., :, None] * top[..., None, :]

    @staticmethod
    @once_differentiable
    def backward(ctx, upstream):
        _tensor(upstream, "projector upstream gradient")
        values, basis = ctx.saved_tensors
        top, rest = basis[..., -1], basis[..., :-1]
        action = (upstream + upstream.mT) @ top[..., None]
        coefficients = (rest.mT @ action).squeeze(-1) / (values[..., -1:] - values[..., :-1])
        t = (rest @ coefficients[..., None]).squeeze(-1)
        gs = _sym(t[..., :, None] * top[..., None, :])
        q = torch.einsum("...i,...ij,...j->...", top, upstream, top)
        gb = -values[..., -1, None, None] * gs - q[..., None, None] * (
            top[..., :, None] * top[..., None, :]
        )
        _tensor(gs, "S gradient")
        _tensor(gb, "B gradient")
        return gs, gb


@torch.no_grad()
def _metrics(s, b, c, f):
    sf, bf = s @ f, b @ f
    trace_b = _trace_product(b, f)
    _require(
        bool(torch.isfinite(trace_b).all() and (trace_b != 0).all()),
        "Undefined generalized Rayleigh quotient",
    )
    eigenvalue = _trace_product(s, f) / trace_b
    denominator = (_fro(s) + eigenvalue.abs() * _fro(b)) * _fro(f)
    _require(
        bool(torch.isfinite(denominator).all() and (denominator > 0).all()),
        "Undefined original-pair residual",
    )
    result = {
        "lambda": eigenvalue,
        "residual": _fro(sf - eigenvalue[..., None, None] * bf) / denominator,
        "normalization_error": (_trace_product(c, f) - 1).abs(),
        "symmetry_error": _skew(f),
    }
    _require(
        all(bool(torch.isfinite(value).all()) for value in result.values()), "Nonfinite diagnostics"
    )
    return result


def projector(s: Tensor, b: Tensor, c: Tensor, method: str) -> Tensor:
    """Return differentiable, C-normalized F using exactly the named method.

    Independent S/B/C parameters and a shared underlying B=C parameter are
    both supported. No fallback, gap clipping, jitter or input broadcasting.
    """
    _require(isinstance(method, str) and method in METHODS, "Unknown frozen numerical method")
    ss, bb, cc = _inputs(s, b, c)
    try:
        k = _n1(ss, bb) if method == METHODS[0] else _GeneralizedProjector.apply(ss, bb)
    except RuntimeError as error:
        raise ValueError(f"{method} solver failed; no fallback") from error
    norm = _trace_product(cc, k)
    _require(
        bool(torch.isfinite(norm).all() and (norm > 0).all()),
        "Nonpositive or nonfinite C normalization",
    )
    result = k / norm[..., None, None]
    _tensor(result, "C-normalized projector")
    residual = _metrics(s, b, c, result)["residual"]
    _require(
        bool((residual <= RESIDUAL_MAX).all()), "Original-pair residual exceeds frozen tolerance"
    )
    return result


@torch.no_grad()
def diagnostics(s: Tensor, b: Tensor, c: Tensor, f: Tensor) -> dict[str, Tensor]:
    """Detached batch scalars; zero eigen-residual does not certify topness.

    The raw-H skew is a diagnostic only, computed before N1 H symmetrization.
    Rayleigh quotient/residual/normalization use original supplied inputs.
    """
    ss, bb, _ = _inputs(s, b, c)
    _tensor(f, "f")
    _require(f.shape == s.shape, "F shape must match S/B/C exactly")
    _, raw = _raw_h(ss, bb)
    return {**_metrics(s, b, c, f), "raw_h_skew": _skew(raw).detach()}
