"""No-IO reliability features and target teachers for the prospective DAN study."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

ARMS = ("U", "Q", "Q2", "QM", "SHAM")


def logged_impedance(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    impedance = np.asarray(values, dtype=np.float64)
    if impedance.ndim != 2 or min(impedance.shape) < 1:
        raise ValueError("Impedance must be [paid block,channel].")
    if np.isinf(impedance).any() or np.any(impedance[np.isfinite(impedance)] < 0):
        raise ValueError("Only nonnegative impedance or NaN missing values are allowed.")
    observed = np.isfinite(impedance)
    return np.log1p(np.where(observed, impedance, 0)), observed


@dataclass(frozen=True)
class SourceChannelScaler:
    mean: np.ndarray
    scale: np.ndarray
    fit_subject_ids: tuple[int, ...]

    @classmethod
    def fit(cls, values: np.ndarray, observed: np.ndarray, *,
            row_subject_ids: Sequence[int], allowed_fit_ids: Sequence[int]) -> SourceChannelScaler:
        values = np.ascontiguousarray(values, dtype=np.float64)
        observed = np.asarray(observed)
        people = tuple(sorted(allowed_fit_ids))
        if not people or len(set(people)) != len(people):
            raise ValueError("Distinct fit subject IDs required.")
        if values.ndim != 2 or min(values.shape) < 1 or observed.shape != values.shape or observed.dtype != bool:
            raise ValueError("Expected row/channel values and a boolean availability mask.")
        if len(row_subject_ids) != len(values) or set(row_subject_ids) != set(people):
            raise ValueError("Scaler rows must contain exactly the allowed source fit people.")
        if not np.isfinite(values[observed]).all():
            raise ValueError("Observed values must be finite.")
        mean, scale = np.zeros(values.shape[1]), np.ones(values.shape[1])
        for channel in range(values.shape[1]):
            sample = values[:, channel][observed[:, channel]]
            if len(sample):
                mean[channel] = sample.mean(dtype=np.float64)
                deviation = sample.std(dtype=np.float64, ddof=0)
                scale[channel] = deviation if deviation >= 1e-8 else 1.0
        mean.setflags(write=False)
        scale.setflags(write=False)
        return cls(mean, scale, people)

    def transform(self, values: np.ndarray, observed: np.ndarray) -> np.ndarray:
        values, observed = np.asarray(values, dtype=np.float64), np.asarray(observed)
        if values.ndim != 2 or values.shape[1] != len(self.mean):
            raise ValueError("Scaler channel schema differs.")
        if observed.shape != values.shape or observed.dtype != bool or not np.isfinite(values[observed]).all():
            raise ValueError("Invalid observed feature packet.")
        safe = np.where(observed, values, self.mean)
        return np.where(observed, (safe - self.mean) / self.scale, 0.0)


def teacher_block_weights(q: np.ndarray, arm: str, *, auxiliary: np.ndarray | None = None,
                         observed: np.ndarray | None = None) -> np.ndarray:
    """Input features must already be standardized using source-fit statistics.

    U/Q intentionally ignore auxiliary arguments. A missing or block-constant
    auxiliary contributes exactly zero, avoiding accidental Q rescaling.
    """
    q = np.ascontiguousarray(q, dtype=np.float64)
    if q.ndim != 2 or q.shape[0] < 2 or q.shape[1] < 1 or not np.isfinite(q).all():
        raise ValueError("Finite [k>=2,channel] paid-support Q required.")
    if arm not in ARMS:
        raise ValueError("Unknown arm.")
    if arm == "U":
        return np.full_like(q, 1.0 / len(q))
    auxiliary_centered = np.zeros_like(q)
    if arm in ("Q2", "QM", "SHAM"):
        if auxiliary is None or observed is None:
            raise ValueError("This arm requires an explicit auxiliary packet/mask.")
        auxiliary = np.ascontiguousarray(auxiliary, dtype=np.float64)
        observed = np.asarray(observed)
        if auxiliary.shape != q.shape or observed.shape != q.shape or observed.dtype != bool:
            raise ValueError("Auxiliary shape or availability type differs.")
        if not np.isfinite(auxiliary[observed]).all():
            raise ValueError("Observed auxiliary must be finite.")
        for channel in range(q.shape[1]):
            mask = observed[:, channel]
            if mask.any():
                sample = auxiliary[mask, channel]
                # Subtract an anchor first: constant packets yield bit-exact zero.
                shifted = sample - sample[0]
                auxiliary_centered[mask, channel] = shifted - shifted.mean(dtype=np.float64)
    shifted_q = q - q[:1]
    logits = np.clip(-(shifted_q - shifted_q.mean(axis=0)) - 0.5 * auxiliary_centered, -3, 3)
    mass = np.exp(logits - logits.max(axis=0))
    mass /= mass.sum(axis=0)
    return 0.5 / len(q) + 0.5 * mass


def support_quality(support: np.ndarray, *, frequencies: Sequence[float],
                    sfreq: float, n_harmonics: int = 3) -> tuple[np.ndarray, np.ndarray]:
    """Paid [block,class,channel,time] -> two [block,channel] EEG-only features."""
    x = np.ascontiguousarray(support, dtype=np.float64)
    frequencies = np.asarray(frequencies, dtype=np.float64)
    if (x.ndim != 4 or x.shape[0] < 2 or x.shape[-1] < 8 or min(x.shape[1:]) < 1
            or frequencies.shape != (x.shape[1],) or not np.isfinite(x).all()):
        raise ValueError("Invalid complete labeled support geometry.")
    if (not np.isfinite(frequencies).all() or np.any(frequencies <= 0)
            or not np.isfinite(sfreq) or sfreq <= 0 or n_harmonics < 1
            or frequencies.max() * n_harmonics >= sfreq / 2
            or 2 * n_harmonics > x.shape[-1]):
        raise ValueError("Invalid known stimulus bank.")
    x = x - x.mean(axis=-1, keepdims=True)
    power = np.mean(x * x, axis=-1)
    if np.any(power <= 1e-24):
        raise ValueError("Silent zero-power channel removal is forbidden.")
    residual_log = np.empty(power.shape)
    time = np.arange(x.shape[-1]) / sfreq
    for label, frequency in enumerate(frequencies):
        angles = 2 * np.pi * frequency * time[:, None] * np.arange(1, n_harmonics + 1)
        references = np.concatenate((np.sin(angles), np.cos(angles)), axis=1)
        references -= references.mean(axis=0)
        basis, _ = np.linalg.qr(references, mode="reduced")
        signals = x[:, label]
        residual = signals - (signals @ basis) @ basis.T
        residual_log[:, label] = np.log(np.maximum(np.mean(residual**2, axis=-1) / power[:, label], 1e-24))
    fourth = np.mean(x**4, axis=-1) / power**2
    return residual_log.mean(axis=1), np.log(np.maximum(fourth, 1e-24)).mean(axis=1)
