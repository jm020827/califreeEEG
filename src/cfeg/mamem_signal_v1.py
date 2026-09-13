"""Geometry-free, common 256-row signal features for the frozen MAMEM v1 study.

Only ``analyze_window`` is public. No I/O, metadata, labels, filtering, re-reference,
montage assignment, or unit conversion occurs here. Structural checks precede a
float64 copy of rows [0:256] and exactly 500 selected samples; all numerical checks
and calculations follow that copy. The excluded row and times are never examined.

Each selected channel is demeaned, then all channels share one population RMS.
RMS is evaluated after division by the centered absolute maximum to avoid squaring
tiny/huge physical units. Only exactly zero energy is rejected; there is no absolute
amplitude threshold. Overflow of an unrepresentable centered value is rejected.

References are column-centered sine/cosine pairs, QR-orthonormalized within their
own window. Factors are REAL quadrature factors, not complex temporal templates.
Q = [tr(H)/tr(C), log((tr(H)+f)/(mean_neighbor_energy+f)), entropy_rank(C)/256,
     ||H_first/tr(H_first)-H_second/tr(H_second)||_F]. Neighbors are harmonic Hz
plus/minus 0.5 Hz. Entropy rank is exp(-sum(p*log(p))) for covariance eigenvalues p.
Q2 = [log1p(max(diag(C))/mean(diag(C))), cosine(X[:,:-1], X[:,1:])], repeated for
the five candidate frequencies. Split-half references use their respective times;
both halves retain the common full-window normalization.

Frozen numerical conventions, in normalized (not physical) units:
* log-energy floor f = 1e-12 * full-window tr(C);
* half projection <= 1e-12 * half total energy uses I/256 for its normalized H;
* negative covariance eigenvalues within 1e-10 * tr(C) are roundoff, clipped to 0;
* CCA uses C + 1e-6 * tr(C)/256 * I, and a 1e-7 dimensionless bound tolerance;
* an exactly zero lag-vector norm returns correlation 0.
All returned arrays own their data and are finite float64. Standard reference
frequencies and 250 Hz are protocol constants, not measured timing/geometry claims.
"""

from functools import lru_cache

import numpy as np
from scipy.linalg import cholesky, eigvalsh, solve_triangular


_CHANNELS = 256
_SAMPLES = 500
_RATE = 250.0
_FREQUENCIES = (6.66, 7.50, 8.57, 10.00, 12.00)
_ENERGY_FLOOR = 1e-12
_EIGEN_TOL = 1e-10
_CCA_RIDGE = 1e-6
_BOUND_TOL = 1e-7


def _require(condition, reason):
    if not condition:
        raise ValueError(reason)


def _quadrature(frequencies, count, first=0):
    time = np.arange(first, first + count, dtype=np.float64) / _RATE
    columns = []
    for frequency in frequencies:
        phase = (2.0 * np.pi * frequency) * time
        columns.extend((np.sin(phase), np.cos(phase)))
    references = np.column_stack(columns)
    references -= references.mean(axis=0)
    basis, _ = np.linalg.qr(references, mode="reduced")
    return basis


@lru_cache(maxsize=1)
def _reference_banks():
    full, neighbor, first, second, joint = [], [], [], [], []
    for frequency in _FREQUENCIES:
        for harmonic in (1, 2):
            hz = frequency * harmonic
            full.append(_quadrature((hz,), _SAMPLES))
            neighbor.extend(_quadrature((hz + delta,), _SAMPLES) for delta in (-0.5, 0.5))
            first.append(_quadrature((hz,), _SAMPLES // 2))
            second.append(_quadrature((hz,), _SAMPLES // 2, _SAMPLES // 2))
        joint.append(_quadrature((frequency, 2.0 * frequency), _SAMPLES))
    banks = tuple(np.column_stack(parts) for parts in (full, neighbor, first, second, joint))
    for bank in banks:
        bank.setflags(write=False)
    return banks


def _normalized_psd(factor, total_energy):
    energy = float(np.sum(factor * factor))
    if energy <= _ENERGY_FLOOR * total_energy:
        return np.eye(_CHANNELS, dtype=np.float64) / _CHANNELS
    return (factor @ factor.T) / energy


def _center_and_scale(window):
    # Computing the mean in each channel's relative units avoids sum overflow.
    row_peak = np.max(np.abs(window), axis=1, keepdims=True)
    safe_peak = np.where(row_peak == 0.0, 1.0, row_peak)
    means = np.mean(window / safe_peak, axis=1, keepdims=True) * safe_peak
    with np.errstate(over="ignore", invalid="ignore"):
        window -= means
    _require(bool(np.isfinite(window).all()), "nonfinite_centered_window")
    peak = float(np.max(np.abs(window)))
    _require(peak > 0.0, "zero_centered_energy")
    window /= peak
    rms = float(np.sqrt(np.mean(window * window)))
    _require(np.isfinite(rms) and rms > 0.0, "invalid_centered_rms")
    window /= rms
    return window


def analyze_window(eeg, start0, end0):
    """Return covariance, quadrature factors, Q/Q2 and zero-shot CCA scores.

    ``eeg`` must be a real float64 ndarray of shape (257, total_samples).
    ``start0:end0`` is a zero-based, half-open interval of exactly 500 samples.
    Errors are ``ValueError`` with a static reason. Caller must enforce source,
    support/query, file provenance and chronological roles outside this pure API.
    """
    _require(isinstance(eeg, np.ndarray), "ndarray_required")
    _require(eeg.ndim == 2 and eeg.shape[0] == 257, "raw_shape_257_by_time_required")
    _require(eeg.dtype == np.dtype(np.float64), "float64_required")
    _require(isinstance(start0, (int, np.integer)) and not isinstance(start0, (bool, np.bool_)),
             "integer_start_required")
    _require(isinstance(end0, (int, np.integer)) and not isinstance(end0, (bool, np.bool_)),
             "integer_end_required")
    start0, end0 = int(start0), int(end0)
    _require(0 <= start0 < end0 <= eeg.shape[1] and end0 - start0 == _SAMPLES,
             "invalid_500_sample_window")
    # Never run finite checks, reductions or normalization over the raw input.
    selected = np.array(eeg[:_CHANNELS, start0:end0], dtype=np.float64, order="C", copy=True)
    _require(bool(np.isfinite(selected).all()), "nonfinite_selected_window")
    x = _center_and_scale(selected)
    covariance = x @ x.T / _SAMPLES
    total = float(np.trace(covariance))
    _require(np.isfinite(total) and total > 0.0, "invalid_covariance_energy")

    full, neighbor, half_first, half_second, joint = _reference_banks()
    flat = (x @ full) / np.sqrt(_SAMPLES)
    factors = flat.reshape(_CHANNELS, 5, 2, 2).transpose(1, 2, 0, 3).copy()
    energies = np.sum(factors * factors, axis=(2, 3))
    near = (x @ neighbor) / np.sqrt(_SAMPLES)
    neighbor_energies = np.sum(near * near, axis=0).reshape(5, 2, 2, 2).sum(axis=-1).mean(axis=-1)
    relative = energies / total
    _require(bool(((relative >= -_BOUND_TOL) & (relative <= 1.0 + _BOUND_TOL)).all()),
             "projection_energy_out_of_bounds")

    eigenvalues = eigvalsh(covariance, check_finite=False)
    _require(float(eigenvalues[0]) >= -_EIGEN_TOL * total, "covariance_not_psd")
    eigenvalues = np.maximum(eigenvalues, 0.0)
    probabilities = eigenvalues[eigenvalues > 0] / eigenvalues.sum()
    rank_fraction = float(np.exp(-np.sum(probabilities * np.log(probabilities))) / _CHANNELS)
    first_x, second_x = x[:, :_SAMPLES // 2], x[:, _SAMPLES // 2:]
    first_f = (first_x @ half_first / np.sqrt(_SAMPLES // 2)).reshape(_CHANNELS, 5, 2, 2)
    second_f = (second_x @ half_second / np.sqrt(_SAMPLES // 2)).reshape(_CHANNELS, 5, 2, 2)
    first_energy = float(np.sum(first_x * first_x) / (_SAMPLES // 2))
    second_energy = float(np.sum(second_x * second_x) / (_SAMPLES // 2))
    split = np.empty((5, 2), dtype=np.float64)
    for candidate in range(5):
        for harmonic in range(2):
            a = _normalized_psd(first_f[:, candidate, harmonic], first_energy)
            b = _normalized_psd(second_f[:, candidate, harmonic], second_energy)
            split[candidate, harmonic] = np.linalg.norm(a - b, ord="fro")
    q = np.empty((5, 2, 4), dtype=np.float64)
    q[..., 0] = np.clip(relative, 0.0, 1.0)
    q[..., 1] = np.log((energies + _ENERGY_FLOOR * total)
                       / (neighbor_energies + _ENERGY_FLOOR * total))
    q[..., 2] = rank_fraction
    q[..., 3] = split

    power = np.diag(covariance)
    channel_concentration = float(np.log1p(power.max() / power.mean()))
    left, right = x[:, :-1], x[:, 1:]
    norm_product = float(np.linalg.norm(left) * np.linalg.norm(right))
    lag = 0.0 if norm_product == 0.0 else float(np.sum(left * right) / norm_product)
    _require(-1.0 - _BOUND_TOL <= lag <= 1.0 + _BOUND_TOL, "lag_out_of_bounds")
    q2 = np.tile(np.array([channel_concentration, np.clip(lag, -1.0, 1.0)]), (5, 1)).copy()

    # Joint reference whitening is per candidate (four columns), not per harmonic.
    cross = (x @ joint) / np.sqrt(_SAMPLES)
    regularized = covariance + (_CCA_RIDGE * total / _CHANNELS) * np.eye(_CHANNELS)
    try:
        lower = cholesky(regularized, lower=True, check_finite=False)
        whitened = solve_triangular(lower, cross, lower=True, check_finite=False)
    except np.linalg.LinAlgError as error:
        raise ValueError("cca_factorization_failed") from error
    zero_shot = np.empty(5, dtype=np.float64)
    for candidate in range(5):
        block = whitened[:, 4 * candidate:4 * (candidate + 1)]
        largest = float(eigvalsh(block.T @ block, check_finite=False)[-1])
        _require(-_BOUND_TOL <= largest <= 1.0 + _BOUND_TOL, "cca_out_of_bounds")
        zero_shot[candidate] = np.clip(largest, 0.0, 1.0)
    result = dict(covariance=covariance, factors=factors, q=q, q2=q2, zero_shot=zero_shot)
    _require(all(value.dtype == np.float64 and bool(np.isfinite(value).all())
                 for value in result.values()), "nonfinite_features")
    return result
