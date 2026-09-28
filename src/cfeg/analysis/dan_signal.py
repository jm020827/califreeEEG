"""Frozen DAN study preprocessing and independently implemented classical decoders.

No file IO, participant lookup, parameter search or target-label scoring here.
Input arrays are already restricted to the caller's permitted role and time prefix.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import linalg, signal


def _finite(x: np.ndarray, ndim: int | None = None) -> np.ndarray:
    x = np.ascontiguousarray(x, dtype=np.float64)
    if (ndim is not None and x.ndim != ndim) or not x.size or not np.isfinite(x).all():
        raise ValueError("Expected a nonempty finite waveform array with the declared rank.")
    return x


def filter_prefix(prefix: np.ndarray, config: dict) -> np.ndarray:
    """[...,channel,535] -> [band,...,channel,375], never accepts later samples."""
    x = _finite(prefix)
    p = config["preprocessing"]
    end = p["available_prefix_end_exclusive"]
    if x.ndim < 2 or x.shape[-2:] != (len(config["channels"]), end):
        raise ValueError("Pass exactly the available channel/time prefix, not the full trial.")
    begin, count, sfreq = config["start_sample"], config["n_samples"], config["sfreq"]
    if begin + count != end or begin < 0 or count < 2:
        raise ValueError("Inconsistent decision crop.")
    b, a = signal.iirnotch(p["notch_hz"], p["notch_q"], fs=sfreq)
    notched = signal.filtfilt(b, a, x, axis=-1)
    bands = []
    for low, stop in zip(p["pass_low_hz"], p["stop_low_hz"]):
        order, critical = signal.cheb1ord(
            [low, p["pass_high_hz"]], [stop, p["stop_high_hz"]],
            p["pass_ripple_db"], p["stop_attenuation_db"], fs=sfreq)
        sos = signal.cheby1(order, p["pass_ripple_db"], critical,
                           btype="bandpass", fs=sfreq, output="sos")
        band = signal.sosfiltfilt(sos, notched, axis=-1)[..., begin:end].copy()
        band -= band.mean(axis=-1, keepdims=True)
        scale = band.std(axis=-1, ddof=p["channel_standardization_ddof"], keepdims=True)
        if not np.isfinite(scale).all() or np.any(scale <= 1e-12):
            raise ValueError("Zero-power/nonfinite channel cannot be silently removed.")
        bands.append(band / scale)
    result = np.stack(bands)
    if result.shape[0] != 3 or not np.isfinite(result).all():
        raise ValueError("Expected three finite filter bands.")
    return result


def trca_covariances(trials: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    x = _finite(trials, 3)
    if len(x) < 2 or min(x.shape[1:]) < 2:
        raise ValueError("TRCA needs >=2 repeated multichannel trials.")
    x = x - x.mean(-1, keepdims=True)
    within = np.einsum("nct,ndt->cd", x, x)
    total = x.sum(axis=0)
    between = total @ total.T - within
    return (between + between.T) / 2, (within + within.T) / 2


def _ridge(covariance: np.ndarray, ridge: float) -> np.ndarray:
    variance = np.trace(covariance) / len(covariance)
    if not np.isfinite(ridge) or ridge <= 0 or not np.isfinite(variance) or variance <= 0:
        raise ValueError("Positive trace and ridge required.")
    return covariance + np.eye(len(covariance)) * ridge * variance


def _band_weights(n: int) -> np.ndarray:
    return np.arange(1, n + 1, dtype=float) ** -1.25 + 0.25


@dataclass(frozen=True)
class EnsembleTRCA:
    filters: np.ndarray  # [band,channel,class]
    templates: np.ndarray  # [band,class,channel,time]

    def scores(self, query: np.ndarray) -> np.ndarray:
        x = _finite(query, 4)
        if (x.shape[0] != len(self.filters)
                or x.shape[2:] != self.templates.shape[2:]):
            raise ValueError("Query geometry differs from the frozen decoder.")
        x = x - x.mean(-1, keepdims=True)
        result = np.zeros((x.shape[1], self.templates.shape[1]))
        for band in range(len(self.filters)):
            projected = np.einsum("ck,nct->nkt", self.filters[band], x[band])
            references = np.einsum("ck,jct->jkt", self.filters[band], self.templates[band])
            left = projected.reshape(len(projected), -1)
            right = references.reshape(len(references), -1)
            left -= left.mean(-1, keepdims=True)
            right -= right.mean(-1, keepdims=True)
            denominator = np.linalg.norm(left, axis=1)[:, None] * np.linalg.norm(right, axis=1)
            if np.any(denominator <= 1e-24):
                raise ValueError("Undefined correlation: silent projection/template.")
            correlation = np.clip((left @ right.T) / denominator, -1, 1)
            result += _band_weights(len(self.filters))[band] * correlation * abs(correlation)
        return result


def fit_ensemble_trca(calibration: np.ndarray, *, ridge: float = 1e-8) -> EnsembleTRCA:
    """[band,repeat,class,channel,time]; no query argument exists."""
    x = _finite(calibration, 5)
    if len(x) != 3 or x.shape[1] < 2:
        raise ValueError("Three bands and >=2 complete labeled repeats required.")
    x = x - x.mean(-1, keepdims=True)
    filters = np.empty((len(x), x.shape[3], x.shape[2]))
    for band in range(len(x)):
        for label in range(x.shape[2]):
            between, within = trca_covariances(x[band, :, label])
            _, vectors = linalg.eigh(between, _ridge(within, ridge), check_finite=True)
            vector = vectors[:, -1]
            vector /= np.linalg.norm(vector)
            if vector[np.argmax(abs(vector))] < 0:
                vector *= -1
            filters[band, :, label] = vector
    templates = x.mean(axis=1)
    filters.setflags(write=False)
    templates.setflags(write=False)
    return EnsembleTRCA(filters, templates)


def _whitener(covariance: np.ndarray, ridge: float) -> np.ndarray:
    values, vectors = linalg.eigh(_ridge(covariance, ridge))
    if np.any(values <= 0):
        raise ValueError("Regularized covariance is not positive definite.")
    return (vectors / np.sqrt(values)) @ vectors.T


def fbcca_scores(query: np.ndarray, frequencies: np.ndarray, *, sfreq: float = 250,
                 harmonics: int = 3, ridge: float = 1e-8) -> np.ndarray:
    x = _finite(query, 4)
    frequencies = _finite(frequencies, 1)
    if (len(x) != 3 or sfreq <= 0 or harmonics < 1 or np.any(frequencies <= 0)
            or frequencies.max() * harmonics >= sfreq / 2):
        raise ValueError("Invalid fixed reference bank.")
    time = np.arange(x.shape[-1]) / sfreq
    references = []
    for frequency in frequencies:
        angle = 2 * np.pi * frequency * np.arange(1, harmonics + 1)[:, None] * time
        y = np.concatenate((np.sin(angle), np.cos(angle)))
        y -= y.mean(-1, keepdims=True)
        references.append((_whitener(y @ y.T, ridge), y))
    scores = np.zeros((x.shape[1], len(frequencies)))
    for band in range(len(x)):
        for row in range(x.shape[1]):
            eeg = x[band, row] - x[band, row].mean(-1, keepdims=True)
            wx = _whitener(eeg @ eeg.T, ridge)
            for label, (wy, y) in enumerate(references):
                rho = linalg.svdvals(wx @ eeg @ y.T @ wy)[0]
                scores[row, label] += _band_weights(len(x))[band] * min(float(rho), 1) ** 2
    return scores
