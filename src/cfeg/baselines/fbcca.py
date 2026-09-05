from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from numbers import Integral, Real
from typing import Any

import numpy as np
from scipy import signal


@dataclass(frozen=True)
class FilterBankConfig:
    """Configuration for :func:`predict_fbcca`.

    ``bands`` contains ``(low_hz, high_hz)`` band-pass limits. When ``weights``
    is omitted, subband ``m`` receives the commonly used power-law FBCCA
    weight ``m**(-weight_exponent) + weight_offset`` (with one-based ``m``).
    The Butterworth filters and finite-trial handling used here are generic;
    they are not intended as a bitwise reproduction of a particular toolbox.
    """

    bands: Sequence[Sequence[float]] | None = None
    weights: Sequence[float] | None = None
    order: int = 4
    n_harmonics: int = 3
    regularization: float = 1e-8
    weight_exponent: float = 1.25
    weight_offset: float = 0.25
    filter_family: str = "butterworth"
    passband_ripple_db: float = 0.5
    reproduction_contract: str = "generic"


def _positive_integer(value: Any, name: str) -> int:
    if not isinstance(value, Integral) or isinstance(value, bool) or value <= 0:
        raise ValueError(f"{name} must be a positive integer, got {value!r}")
    return int(value)


def _positive_float(value: Any, name: str) -> float:
    if not isinstance(value, Real) or isinstance(value, bool):
        raise TypeError(f"{name} must be a positive finite number, got {value!r}")
    result = float(value)
    if not np.isfinite(result) or result <= 0.0:
        raise ValueError(f"{name} must be a positive finite number, got {value!r}")
    return result


def _nonnegative_float(value: Any, name: str) -> float:
    if not isinstance(value, Real) or isinstance(value, bool):
        raise TypeError(f"{name} must be a non-negative finite number, got {value!r}")
    result = float(value)
    if not np.isfinite(result) or result < 0.0:
        raise ValueError(f"{name} must be a non-negative finite number, got {value!r}")
    return result


def _frequencies(freqs: Sequence[float] | np.ndarray, sfreq: float) -> np.ndarray:
    values = np.asarray(freqs, dtype=np.float64)
    if values.ndim != 1 or values.size == 0:
        raise ValueError("freqs must be a non-empty one-dimensional sequence")
    if not np.isfinite(values).all() or np.any(values <= 0.0):
        raise ValueError("freqs must contain only positive finite frequencies")
    nyquist = sfreq / 2.0
    if np.any(values >= nyquist):
        raise ValueError(f"all frequencies must be below Nyquist ({nyquist:g} Hz)")
    return values


def _matrix(value: np.ndarray, name: str) -> np.ndarray:
    result = np.asarray(value, dtype=np.float64)
    if result.ndim != 2:
        raise ValueError(f"{name} must have shape (channels/components, samples)")
    if result.shape[0] == 0 or result.shape[1] < 2:
        raise ValueError(f"{name} must contain at least one row and two samples")
    if not np.isfinite(result).all():
        raise ValueError(f"{name} must contain only finite values")
    return result


def make_reference_signals(freqs, sfreq, n_samples, n_harmonics=3):
    """Create sine/cosine SSVEP references.

    Returns an array shaped ``(n_frequencies, 2 * n_harmonics, n_samples)``.
    """

    sampling_frequency = _positive_float(sfreq, "sfreq")
    samples = _positive_integer(n_samples, "n_samples")
    if samples < 2:
        raise ValueError("n_samples must be at least 2")
    harmonics = _positive_integer(n_harmonics, "n_harmonics")
    frequencies = _frequencies(freqs, sampling_frequency)
    max_component = float(frequencies.max()) * harmonics
    if max_component >= sampling_frequency / 2.0:
        raise ValueError(
            "the highest requested harmonic must be below Nyquist; "
            f"got {max_component:g} Hz at sfreq={sampling_frequency:g} Hz"
        )

    t = np.arange(samples, dtype=np.float64) / sampling_frequency
    harmonic_numbers = np.arange(1, harmonics + 1, dtype=np.float64)
    phase = (
        2.0
        * np.pi
        * frequencies[:, None, None]
        * harmonic_numbers[None, :, None]
        * t[None, None, :]
    )
    refs = np.stack((np.sin(phase), np.cos(phase)), axis=2)
    return refs.reshape(len(frequencies), 2 * harmonics, samples).astype(np.float32)


def _inverse_sqrt_psd(matrix: np.ndarray) -> np.ndarray:
    eigenvalues, eigenvectors = np.linalg.eigh(matrix)
    largest = max(float(eigenvalues[-1]), 0.0)
    tolerance = max(np.finfo(np.float64).eps, largest * 1e-12)
    inverse_roots = np.zeros_like(eigenvalues)
    keep = eigenvalues > tolerance
    inverse_roots[keep] = 1.0 / np.sqrt(eigenvalues[keep])
    return (eigenvectors * inverse_roots) @ eigenvectors.T


def cca_score(x, ref, regularization=1e-8):
    """Return the largest regularized canonical correlation.

    Time samples are observations and EEG channels/reference components are
    variables. Covariance whitening, rather than a flattened dot product, lets
    CCA select the best spatial and harmonic mixtures. Relative ridge terms
    keep the calculation stable for duplicated channels and other rank-deficient
    inputs. Constant inputs have a defined score of zero.
    """

    eeg = _matrix(x, "x")
    reference = _matrix(ref, "ref")
    if eeg.shape[1] != reference.shape[1]:
        raise ValueError(
            "x and ref must have the same number of samples; "
            f"got {eeg.shape[1]} and {reference.shape[1]}"
        )
    ridge_fraction = _nonnegative_float(regularization, "regularization")

    eeg = eeg - eeg.mean(axis=1, keepdims=True)
    reference = reference - reference.mean(axis=1, keepdims=True)
    eeg_energy = float(np.linalg.norm(eeg))
    reference_energy = float(np.linalg.norm(reference))
    numerical_floor = np.finfo(np.float64).eps * np.sqrt(eeg.shape[1]) * 10.0
    if eeg_energy <= numerical_floor or reference_energy <= numerical_floor:
        return 0.0

    denominator = float(eeg.shape[1] - 1)
    covariance_xx = eeg @ eeg.T / denominator
    covariance_yy = reference @ reference.T / denominator
    covariance_xy = eeg @ reference.T / denominator

    scale_x = float(np.trace(covariance_xx)) / covariance_xx.shape[0]
    scale_y = float(np.trace(covariance_yy)) / covariance_yy.shape[0]
    if scale_x <= 0.0 or scale_y <= 0.0:
        return 0.0
    covariance_xx = covariance_xx + ridge_fraction * scale_x * np.eye(eeg.shape[0])
    covariance_yy = covariance_yy + ridge_fraction * scale_y * np.eye(reference.shape[0])

    whitened_cross_covariance = (
        _inverse_sqrt_psd(covariance_xx)
        @ covariance_xy
        @ _inverse_sqrt_psd(covariance_yy)
    )
    singular_values = np.linalg.svd(whitened_cross_covariance, compute_uv=False)
    if singular_values.size == 0 or not np.isfinite(singular_values[0]):
        return 0.0
    return float(np.clip(singular_values[0], 0.0, 1.0))


def predict_cca(x, freqs, sfreq, n_harmonics=3, regularization=1e-8):
    """Classify one multichannel trial with plain multi-harmonic CCA."""

    eeg = _matrix(x, "x")
    sampling_frequency = _positive_float(sfreq, "sfreq")
    references = make_reference_signals(
        freqs,
        sampling_frequency,
        eeg.shape[-1],
        n_harmonics=n_harmonics,
    )
    scores = np.asarray(
        [cca_score(eeg, reference, regularization=regularization) for reference in references],
        dtype=np.float64,
    )
    return int(np.argmax(scores)), scores


def _default_bands(sfreq: float) -> tuple[tuple[float, float], ...]:
    nyquist = sfreq / 2.0
    high = min(90.0, 0.95 * nyquist)
    bands = tuple((float(low), high) for low in (6, 14, 22, 30, 38) if low < high)
    if bands:
        return bands
    # Keep the fallback strictly inside Nyquist for unusually low-rate input.
    return ((high / 4.0, high),)


def _is_sequence(value: Any) -> bool:
    return isinstance(value, (Sequence, np.ndarray)) and not isinstance(value, (str, bytes))


def _coerce_filterbank(filterbank: Any) -> FilterBankConfig:
    if filterbank is None:
        return FilterBankConfig()
    if isinstance(filterbank, FilterBankConfig):
        return filterbank
    if isinstance(filterbank, Mapping):
        allowed = {
            "bands",
            "weights",
            "order",
            "n_harmonics",
            "regularization",
            "weight_exponent",
            "weight_offset",
            "filter_family",
            "passband_ripple_db",
            "reproduction_contract",
        }
        unknown = set(filterbank) - allowed
        if unknown:
            raise ValueError(f"unknown filterbank options: {sorted(unknown, key=str)}")
        return FilterBankConfig(**filterbank)
    if _is_sequence(filterbank):
        return FilterBankConfig(bands=filterbank)
    raise ValueError("filterbank must be None, a FilterBankConfig, a mapping, or band pairs")


def _validated_filterbank(
    filterbank: Any, sfreq: float
) -> tuple[
    tuple[tuple[float, float], ...],
    np.ndarray,
    int,
    int,
    float,
    str,
    float,
    str,
]:
    config = _coerce_filterbank(filterbank)
    raw_bands = _default_bands(sfreq) if config.bands is None else config.bands
    if not _is_sequence(raw_bands):
        raise ValueError("filterbank bands must be a non-empty sequence of (low, high) pairs")
    if len(raw_bands) == 0:
        raise ValueError("filterbank bands must not be empty")

    nyquist = sfreq / 2.0
    bands: list[tuple[float, float]] = []
    for index, raw_band in enumerate(raw_bands):
        if not _is_sequence(raw_band):
            raise ValueError(f"filterbank band {index} must be a (low, high) pair")
        if len(raw_band) != 2:
            raise ValueError(f"filterbank band {index} must be a (low, high) pair")
        low = _positive_float(raw_band[0], f"filterbank band {index} low")
        high = _positive_float(raw_band[1], f"filterbank band {index} high")
        if not low < high < nyquist:
            raise ValueError(
                f"filterbank band {index} must satisfy 0 < low < high < "
                f"Nyquist ({nyquist:g} Hz)"
            )
        bands.append((low, high))

    exponent = _nonnegative_float(config.weight_exponent, "weight_exponent")
    offset = _nonnegative_float(config.weight_offset, "weight_offset")
    if config.weights is None:
        indices = np.arange(1, len(bands) + 1, dtype=np.float64)
        weights = indices ** (-exponent) + offset
    else:
        weights = np.asarray(config.weights, dtype=np.float64)
        if weights.ndim != 1 or len(weights) != len(bands):
            raise ValueError("filterbank weights must have one value per band")
        if not np.isfinite(weights).all() or np.any(weights < 0.0) or not np.any(weights > 0.0):
            raise ValueError("filterbank weights must be finite, non-negative, and not all zero")

    order = _positive_integer(config.order, "filterbank order")
    harmonics = _positive_integer(config.n_harmonics, "n_harmonics")
    regularization = _nonnegative_float(config.regularization, "regularization")
    family = str(config.filter_family)
    if family not in {"butterworth", "chebyshev1"}:
        raise ValueError("filter_family must be butterworth or chebyshev1.")
    ripple = _positive_float(config.passband_ripple_db, "passband_ripple_db")
    contract = str(config.reproduction_contract).strip()
    if not contract:
        raise ValueError("reproduction_contract must be non-empty.")
    return (
        tuple(bands),
        weights,
        order,
        harmonics,
        regularization,
        family,
        ripple,
        contract,
    )


def _bandpass(
    x: np.ndarray,
    sfreq: float,
    band: tuple[float, float],
    order: int,
    *,
    family: str,
    ripple_db: float,
) -> np.ndarray:
    if family == "chebyshev1":
        sos = signal.cheby1(
            order, ripple_db, band, btype="bandpass", fs=sfreq, output="sos"
        )
    else:
        sos = signal.butter(order, band, btype="bandpass", fs=sfreq, output="sos")
    # Explicitly cap padding so short, otherwise valid trials fail deterministically.
    default_padlen = 3 * (2 * len(sos) + 1)
    padlen = min(default_padlen, x.shape[-1] - 1)
    return signal.sosfiltfilt(sos, x, axis=-1, padlen=padlen)


def apply_filterbank(
    x: np.ndarray,
    *,
    sfreq: float,
    filterbank: Any,
) -> tuple[np.ndarray, dict[str, object]]:
    """Apply the exact configured filter bank to one or more EEG trials.

    The returned array prepends a subband dimension and otherwise preserves the
    input shape. This public helper lets FBCCA and calibration baselines share
    one frozen DSP implementation.
    """

    values = np.asarray(x, dtype=np.float64)
    if values.ndim < 2 or values.shape[-1] < 2 or not np.isfinite(values).all():
        raise ValueError("Filter-bank EEG must be finite with [...,channel,time] shape.")
    sampling_frequency = _positive_float(sfreq, "sfreq")
    (
        bands,
        weights,
        order,
        harmonics,
        regularization,
        family,
        ripple,
        contract,
    ) = _validated_filterbank(filterbank, sampling_frequency)
    filtered = np.stack(
        [
            _bandpass(
                values,
                sampling_frequency,
                band,
                order,
                family=family,
                ripple_db=ripple,
            )
            for band in bands
        ],
        axis=0,
    )
    parameters = {
        "bands": [list(band) for band in bands],
        "weights": weights.tolist(),
        "order": order,
        "n_harmonics": harmonics,
        "regularization": regularization,
        "filter_family": family,
        "passband_ripple_db": ripple,
        "reproduction_contract": contract,
    }
    return filtered, parameters


def resolve_filterbank_parameters(filterbank: Any, sfreq: float) -> dict[str, object]:
    """Return the exact effective FBCCA parameters for provenance artifacts."""
    sampling_frequency = _positive_float(sfreq, "sfreq")
    (
        bands,
        weights,
        order,
        harmonics,
        regularization,
        family,
        ripple,
        contract,
    ) = _validated_filterbank(filterbank, sampling_frequency)
    return {
        "filter_family": f"zero_phase_{family}_sos",
        "bands_hz": [list(band) for band in bands],
        "weights": weights.tolist(),
        "order": order,
        "n_harmonics": harmonics,
        "regularization": regularization,
        "passband_ripple_db": ripple if family == "chebyshev1" else None,
        "score_fusion": "sum_weighted_squared_canonical_correlations",
        "reproduction_contract": contract,
    }


def predict_fbcca(x, freqs, sfreq, filterbank=None):
    """Classify one trial with filter-bank canonical correlation analysis.

    Parameters
    ----------
    x:
        EEG shaped ``(channels, samples)``.
    freqs:
        Candidate stimulus frequencies in Hz.
    sfreq:
        Sampling frequency in Hz.
    filterbank:
        ``None`` for Nyquist-safe defaults, a sequence of ``(low, high)``
        bands, a :class:`FilterBankConfig`, or an equivalent mapping.

    Returns
    -------
    predicted_index, scores:
        The index into ``freqs`` and weighted sums of squared canonical
        correlations. This implementation follows the standard FBCCA scoring
        structure but does not claim exact reproduction of external toolboxes.
    """

    eeg = _matrix(x, "x")
    sampling_frequency = _positive_float(sfreq, "sfreq")
    frequencies = _frequencies(freqs, sampling_frequency)
    (
        bands,
        weights,
        order,
        harmonics,
        regularization,
        family,
        ripple,
        _,
    ) = _validated_filterbank(filterbank, sampling_frequency)
    references = make_reference_signals(
        frequencies,
        sampling_frequency,
        eeg.shape[-1],
        n_harmonics=harmonics,
    )

    scores = np.zeros(len(frequencies), dtype=np.float64)
    for weight, band in zip(weights, bands):
        filtered = _bandpass(
            eeg,
            sampling_frequency,
            band,
            order,
            family=family,
            ripple_db=ripple,
        )
        correlations = np.asarray(
            [
                cca_score(filtered, reference, regularization=regularization)
                for reference in references
            ],
            dtype=np.float64,
        )
        scores += weight * correlations**2
    return int(np.argmax(scores)), scores
