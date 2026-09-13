"""Synthetic-qualified reference-sensitivity primitive, not an author recipe.

Independent projection-algebra implementation. No readers, labels, fitting,
channel search, filtering or learned metadata. See the synthetic v1 contract.
An eventual E126 reader would select MATLAB row126 / Python125; this module
receives only one already selected channel and cannot establish its provenance.
"""

from __future__ import annotations

import numpy as np
from scipy.linalg import orth

SAMPLE_RATE = 250
SAMPLES = 500
HARMONICS = 2
CHANNEL_INDEX = 125
NOMINAL_FREQUENCIES = (6.66, 7.5, 8.57, 10.0, 12.0)
_TOL = 1e-12


def _require(condition: bool, reason: str) -> None:
    if not condition:
        raise ValueError(reason)


def _real_array(value, shape: tuple[int, ...], name: str) -> np.ndarray:
    array = np.asarray(value)
    _require(array.shape == shape and array.dtype.kind in "fiu", f"{name}_shape_dtype")
    array = np.asarray(array, dtype=np.float64)
    _require(bool(np.isfinite(array).all()), f"{name}_finite")
    return array


def _frequencies(value) -> np.ndarray:
    frequencies = _real_array(value, (5,), "frequencies")
    _require(bool((frequencies > 0).all()
                  and (frequencies < SAMPLE_RATE / (2 * HARMONICS)).all()
                  and (np.diff(frequencies) > 0).all()), "frequency_range_order")
    return frequencies


def reference_bank(frequencies) -> np.ndarray:
    """Return five centered orthonormal (500,4) harmonic reference subspaces."""
    frequencies = _frequencies(frequencies)
    time = np.arange(1, SAMPLES + 1, dtype=np.float64) / SAMPLE_RATE
    bank = []
    for frequency in frequencies:
        angle = 2 * np.pi * time[:, None] * frequency * np.arange(1, HARMONICS + 1)
        reference = np.stack((np.sin(angle), np.cos(angle)), axis=2).reshape(SAMPLES, 4)
        reference -= reference.mean(axis=0, keepdims=True)
        basis = orth(reference)
        _require(basis.shape == (SAMPLES, 4), "reference_rank")
        bank.append(basis)
    result = np.stack(bank)
    _validate_bank(result)
    return result


def sample_clock_frequencies(samples_by_class) -> np.ndarray:
    """Convert exactly one support event-index sequence per ordered class to Hz.

    Two events per cycle is an assumption. Caller must prove support-only origin
    and window membership; this function verifies span, not absolute EEG bounds.
    Timestamp fields, query truth, other support repeats and EEG are not inputs.
    """
    _require(isinstance(samples_by_class, (tuple, list)) and len(samples_by_class) == 5,
             "five_support_sequences")
    frequencies = []
    for value in samples_by_class:
        samples = np.asarray(value)
        _require(samples.ndim == 1 and 4 <= samples.size <= SAMPLES
                 and samples.dtype.kind in "fiu", "sample_shape_dtype")
        # Exact float64 integer range also prevents signed integer diff overflow.
        values = np.asarray(samples, dtype=np.float64)
        _require(bool(np.isfinite(values).all() and (values >= 1).all()
                      and (values <= 2**53 - 1).all()
                      and (values == np.floor(values)).all()), "sample_integer_range")
        differences = np.diff(values)
        span = float(values[-1] - values[0])
        _require(bool((differences > 0).all()) and span < SAMPLES, "sample_order_span")
        frequencies.append(SAMPLE_RATE * (values.size - 1) / (2 * span))
    return _frequencies(frequencies)


def _validate_bank(value) -> np.ndarray:
    bank = _real_array(value, (5, SAMPLES, 4), "bank")
    gram = bank.transpose(0, 2, 1) @ bank
    _require(bool(np.max(np.abs(gram - np.eye(4))) <= _TOL), "bank_orthonormal")
    _require(bool(np.max(np.abs(bank.mean(axis=1))) <= _TOL), "bank_centered")
    return bank


def scores(query, bank) -> np.ndarray:
    """Squared one-channel CCA; no query-label or query-DIN input.

    This is the ratio of centered query energy projected on each reference
    subspace. Scaling before centering avoids squaring raw extreme amplitudes.
    Numerical checks reject invalid subspaces; only <=1e-12 roundoff is clipped.
    """
    query = _real_array(query, (SAMPLES,), "query")
    bank = _validate_bank(bank)
    scale = float(np.max(np.abs(query)))
    _require(scale > 0, "constant_query")
    centered = query / scale
    centered = centered - centered.mean()
    denominator = float(centered @ centered)
    _require(np.isfinite(denominator) and denominator > 0, "constant_query")
    normalized = centered / np.sqrt(denominator)
    projected = bank.transpose(0, 2, 1) @ normalized
    result = np.sum(projected**2, axis=1)
    _require(bool(np.isfinite(result).all() and (result >= -_TOL).all()
                  and (result <= 1 + _TOL).all()), "score_range")
    return np.clip(result, 0, 1)


def predict(query, bank) -> int:
    """Full-bank argmax, with the smallest index winning an exact tie."""
    return int(np.argmax(scores(query, bank)))
