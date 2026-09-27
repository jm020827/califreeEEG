"""Pure numerical and information-boundary helpers for the frozen joint program.

This file never opens participant files. Public callers must supply role-filtered
arrays; the human runner must additionally enforce the immutable source allowlist.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np

ARMS = ("Q", "Q2", "QM", "SHAM")


def reference_bank(
    frequencies: Sequence[float], sfreq: float, n_samples: int, n_harmonics: int
) -> np.ndarray:
    frequencies = np.asarray(frequencies, dtype=np.float64)
    if frequencies.ndim != 1 or not len(frequencies) or not np.isfinite(frequencies).all():
        raise ValueError("Need a finite one-dimensional frequency bank.")
    if len(np.unique(frequencies)) != len(frequencies) or np.any(frequencies <= 0):
        raise ValueError("Frequencies must be positive and distinct.")
    if not np.isfinite(sfreq) or sfreq <= 0 or n_samples < 2 or n_harmonics < 1:
        raise ValueError("Invalid time or harmonic specification.")
    if frequencies.max() * n_harmonics >= sfreq / 2:
        raise ValueError("Harmonics must be strictly below Nyquist.")
    time = np.arange(n_samples, dtype=np.float64) / sfreq
    angle = 2 * np.pi * frequencies[:, None, None] * (
        np.arange(1, n_harmonics + 1)[None, :, None] * time
    )
    return np.exp(-1j * angle)


def spectral_features(
    trials: np.ndarray, *, frequencies: Sequence[float], sfreq: float, n_harmonics: int = 3
) -> dict[str, np.ndarray]:
    """All-candidate direct Fourier features without labels or cohort statistics.

    Result spectra order is channel, frequency, harmonic, real/imaginary. Q/Q2
    are per-trial; only support rows may be pooled into the episode context.
    """
    x = np.asarray(trials, dtype=np.float64)
    if x.ndim < 3 or not x.shape[-2] or not np.isfinite(x).all():
        raise ValueError("Expected finite [...,channel,time] trials.")
    bank = reference_bank(frequencies, sfreq, x.shape[-1], n_harmonics)
    x = x - x.mean(axis=-1, keepdims=True)
    power = np.mean(x * x, axis=-1)
    global_rms = np.sqrt(power.mean(axis=-1))
    if np.any(global_rms <= 1e-12):
        raise ValueError("Zero or invalid trial RMS; silent trial dropping is forbidden.")
    coefficients = (2 / x.shape[-1]) * np.einsum("...ct,jht->...cjh", x, bank)
    normalized = coefficients / global_rms[..., None, None, None]
    spectra = np.stack((normalized.real, normalized.imag), axis=-1)
    spectra = spectra.reshape(*x.shape[:-1], -1)
    rms = np.sqrt(np.maximum(power, 1e-24))
    relative_power = np.sum(np.abs(coefficients) ** 2, axis=(-2, -1)) / np.maximum(
        power, 1e-24
    )
    q = np.stack((np.log(rms), np.log(np.maximum(relative_power, 1e-24))), axis=-1)
    differences = np.sqrt(np.mean(np.diff(x, axis=-1) ** 2, axis=-1)) / rms
    fourth = np.mean((x / rms[..., None]) ** 4, axis=-1)
    q2 = np.stack(
        (np.log(np.maximum(differences, 1e-24)), np.log(np.maximum(fourth, 1e-24))),
        axis=-1,
    )
    return {"spectra": spectra, "q": q, "q2": q2}


def impedance_summary(prefix: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """[paid block,channel] -> [channel,mean_logZ/std_logZ] and availability."""
    z = np.asarray(prefix, dtype=np.float64)
    if z.ndim != 2 or not z.shape[0] or not z.shape[1]:
        raise ValueError("Impedance must be a nonempty paid prefix by channel.")
    if np.isinf(z).any() or np.any(z[np.isfinite(z)] < 0):
        raise ValueError("Impedance must be nonnegative or NaN for missing.")
    observed = np.isfinite(z)
    count = observed.sum(axis=0)
    logged = np.log1p(np.where(observed, z, 0))
    mean = logged.sum(axis=0) / np.maximum(count, 1)
    variance = (np.where(observed, logged - mean, 0) ** 2).sum(axis=0) / np.maximum(count, 1)
    values = np.stack((mean, np.sqrt(variance)), axis=-1)
    availability = np.repeat((count > 0)[:, None], 2, axis=1)
    return values, availability


def make_context(
    q: np.ndarray,
    *,
    common: np.ndarray,
    arm: str,
    q2: np.ndarray | None = None,
    metadata: np.ndarray | None = None,
    available: np.ndarray | None = None,
) -> np.ndarray:
    """[channel,10]: Q2, auxiliary2, auxiliary_available2, common4.

    Q deliberately ignores all M arguments, even poison values. Duplicate Q
    keeps the second input branch trainable at equal nominal model capacity.
    """
    q = np.asarray(q, dtype=np.float64)
    common = np.asarray(common, dtype=np.float64)
    if q.ndim != 2 or q.shape[1] != 2 or not np.isfinite(q).all():
        raise ValueError("Q must contain two finite values per channel.")
    if common.shape != (4,) or not np.isfinite(common).all():
        raise ValueError("Exactly four finite common fields are required.")
    if arm not in ARMS:
        raise ValueError("Unknown arm.")
    if arm == "Q":
        auxiliary, observed = q, np.ones_like(q, dtype=bool)
    elif arm == "Q2":
        if q2 is None:
            raise ValueError("Q2 requires extra EEG features.")
        auxiliary, observed = np.asarray(q2), np.ones_like(q, dtype=bool)
    else:
        if metadata is None or available is None:
            raise ValueError("QM/SHAM require explicitly paired metadata and flags.")
        auxiliary, observed = np.asarray(metadata), np.asarray(available)
        if observed.dtype != np.bool_:
            raise ValueError("Metadata availability must be boolean.")
    if auxiliary.shape != q.shape or observed.shape != q.shape:
        raise ValueError("Auxiliary values/flags do not match channels.")
    if not np.isfinite(auxiliary[observed]).all():
        raise ValueError("Observed auxiliary values must be finite.")
    safe = np.where(observed, auxiliary, 0)
    return np.concatenate((q, safe, observed.astype(float),
                           np.broadcast_to(common, (len(q), 4))), axis=-1)


@dataclass(frozen=True)
class ContextScaler:
    mean: np.ndarray
    scale: np.ndarray
    fit_subject_ids: tuple[int, ...]

    @classmethod
    def fit(
        cls,
        contexts: np.ndarray,
        *,
        subject_ids: Sequence[int],
        allowed_fit_ids: Sequence[int],
    ) -> ContextScaler:
        x = np.asarray(contexts, dtype=np.float64)
        allowed = tuple(sorted(allowed_fit_ids))
        if not allowed or len(set(allowed)) != len(allowed):
            raise ValueError("Fit role must have distinct participant IDs.")
        if x.ndim != 3 or x.shape[-1] != 10 or len(subject_ids) != len(x):
            raise ValueError("Context rows and participant IDs must align.")
        if set(subject_ids) != set(allowed):
            raise ValueError("Normalizer fit rows must contain exactly the fit role.")
        if not np.isfinite(x).all() or not np.isin(x[..., 4:6], [0, 1]).all():
            raise ValueError("Context must be sanitized with binary flags.")
        mask = np.concatenate((np.ones_like(x[..., :2], dtype=bool), x[..., 4:6] > 0), -1)
        count = mask.sum(axis=0)
        mean = np.where(mask, x[..., :4], 0).sum(axis=0) / np.maximum(count, 1)
        variance = np.where(mask, (x[..., :4] - mean) ** 2, 0).sum(axis=0)
        scale = np.sqrt(variance / np.maximum(count, 1))
        scale = np.where(scale < 1e-8, 1.0, scale)
        mean.setflags(write=False)
        scale.setflags(write=False)
        return cls(mean, scale, allowed)

    def transform(self, contexts: np.ndarray) -> np.ndarray:
        x = np.asarray(contexts, dtype=np.float64)
        if x.ndim < 2 or x.shape[-2:] != (self.mean.shape[0], 10):
            raise ValueError("Transform context dimensions differ.")
        if not np.isfinite(x).all() or not np.isin(x[..., 4:6], [0, 1]).all():
            raise ValueError("Invalid normalized-context input.")
        result = x.copy()
        mask = np.concatenate((np.ones_like(x[..., :2], dtype=bool), x[..., 4:6] > 0), -1)
        result[..., :4] = np.where(mask, (x[..., :4] - self.mean) / self.scale, 0)
        return result


def participant_splits(subject_ids: Sequence[int], n_folds: int = 3) -> list[dict]:
    ids = sorted(subject_ids)
    if n_folds != 3 or len(ids) < 9 or len(ids) % n_folds or len(set(ids)) != len(ids):
        raise ValueError("Need unique participants in three equal outer folds.")
    result = []
    for fold in range(n_folds):
        test = ids[fold::n_folds]
        train = [s for s in ids if s not in test]
        result.append({"fold": fold, "outer_fit": train, "outer_query": test,
                       "inner_fit": train[::2], "inner_validation": train[1::2]})
    return result


def sham_donors(
    role_ids: Sequence[int], *, headband_orders: Mapping[int, str], seed: int
) -> dict[int, int]:
    ids = sorted(role_ids)
    if not ids or len(set(ids)) != len(ids) or set(headband_orders) != set(ids):
        raise ValueError("SHAM nuisance mapping must be confined to the exact role.")
    if set(headband_orders.values()) - {"dry", "wet"}:
        raise ValueError("Unknown headband order.")
    rng = np.random.default_rng(seed)
    donors = {}
    for nuisance in ("dry", "wet"):
        group = [s for s in ids if headband_orders[s] == nuisance]
        if len(group) == 1:
            raise ValueError("Singleton SHAM stratum: cannot preserve nuisance and derange.")
        if not group:
            continue
        permuted = rng.permutation(group).tolist()
        donors.update(zip(permuted, permuted[1:] + permuted[:1]))
    return donors


def raw_prototype_scores(
    support: np.ndarray, labels: np.ndarray, query: np.ndarray, *,
    n_classes: int, channel_gain: np.ndarray | None = None,
) -> np.ndarray:
    support, query = np.asarray(support, float), np.asarray(query, float)
    labels = np.asarray(labels)
    if support.ndim != 3 or query.ndim != 3 or support.shape[1:] != query.shape[1:]:
        raise ValueError("Spectral support/query dimensions differ.")
    if labels.shape != (len(support),) or labels.dtype.kind not in "iu":
        raise ValueError("Support labels must be aligned integer labels.")
    if set(labels) != set(range(n_classes)):
        raise ValueError("Every and only valid classes must have support.")
    if not np.isfinite(support).all() or not np.isfinite(query).all():
        raise ValueError("Nonfinite spectral feature.")
    if channel_gain is not None:
        gain = np.asarray(channel_gain, float)
        if gain.shape != (support.shape[1],) or not np.isfinite(gain).all() or np.any(gain <= 0):
            raise ValueError("Channel gains must be finite and positive.")
        support, query = support * gain[None, :, None], query * gain[None, :, None]
    s, q = support.reshape(len(support), -1), query.reshape(len(query), -1)
    prototype = np.stack([s[labels == j].mean(axis=0) for j in range(n_classes)])
    prototype /= np.maximum(np.linalg.norm(prototype, axis=-1, keepdims=True), 1e-12)
    q = q / np.maximum(np.linalg.norm(q, axis=-1, keepdims=True), 1e-12)
    return q @ prototype.T


def direct_impedance_gain(prefix: np.ndarray) -> np.ndarray:
    impedance_summary(prefix)  # same validity and missingness rules
    z = np.asarray(prefix, float)
    observed = np.isfinite(z)
    count = observed.sum(axis=0)
    mean = np.where(observed, z, 0).sum(axis=0) / np.maximum(count, 1)
    log_gain = -0.5 * np.log1p(mean)
    present = count > 0
    if present.any():
        log_gain = log_gain - log_gain[present].mean()
    return np.where(present, np.clip(np.exp(log_gain), 0.5, 2), 1)


def regularized_cca_scores(
    trials: np.ndarray, *, frequencies: Sequence[float], sfreq: float,
    n_harmonics: int = 3, ridge: float = 1e-6,
) -> np.ndarray:
    """Fixed common k0 comparator; not a filter-bank or literature reproduction."""
    x = np.asarray(trials, float)
    if x.ndim != 3 or not np.isfinite(x).all() or ridge <= 0 or not np.isfinite(ridge):
        raise ValueError("CCA needs finite [trial,channel,time] and positive ridge.")
    n = x.shape[-1]
    bank = reference_bank(frequencies, sfreq, n, n_harmonics)
    y = np.stack((bank.real, -bank.imag), axis=2).reshape(len(frequencies), -1, n)
    x, y = x - x.mean(axis=-1, keepdims=True), y - y.mean(axis=-1, keepdims=True)

    def inverse_root(covariance: np.ndarray) -> np.ndarray:
        dimension = covariance.shape[-1]
        scale = np.trace(covariance, axis1=-2, axis2=-1) / dimension
        if np.any(scale <= 1e-24):
            raise ValueError("Zero covariance cannot be silently classified.")
        values, vectors = np.linalg.eigh(
            covariance + (ridge * scale)[..., None, None] * np.eye(dimension)
        )
        if np.any(values <= 0):
            raise ValueError("CCA whitening covariance is not positive definite.")
        return (vectors * values[..., None, :] ** -0.5) @ vectors.swapaxes(-1, -2)

    ix = inverse_root(x @ x.swapaxes(-1, -2) / n)
    iy = inverse_root(y @ y.swapaxes(-1, -2) / n)
    cross = np.einsum("nct,jht->njch", x, y) / n
    whitened = ix[:, None] @ cross @ iy[None]
    return np.linalg.svd(whitened, compute_uv=False)[..., 0]


def select_learning_rate(validation: Mapping[float, float]) -> float:
    if set(validation) != {0.001, 0.0003} or not np.isfinite(list(validation.values())).all():
        raise ValueError("Exactly two fixed finite LR results required.")
    return min(validation, key=lambda lr: (-validation[lr], lr))


def select_policy(accuracy_by_k: Mapping[int, float], target: float = 0.8) -> dict:
    if set(accuracy_by_k) != {0, 1, 3, 5} or not 0 < target <= 1:
        raise ValueError("Policy requires the fixed calibration grid and valid target.")
    values = np.asarray(list(accuracy_by_k.values()), float)
    if not np.isfinite(values).all() or np.any((values < 0) | (values > 1)):
        raise ValueError("Invalid source-validation accuracy.")
    passing = [k for k in sorted(accuracy_by_k) if accuracy_by_k[k] >= target]
    k = passing[0] if passing else 5
    return {"k": k, "acquired_trials": 12 * k, "fallback": not passing}


def observed_threshold_cost(accuracy: np.ndarray, target: float = 0.8) -> np.ndarray:
    """Descriptive only: unmatched/unattained costs remain NaN, not zero."""
    values = np.asarray(accuracy, float)
    if values.shape[-1:] != (4,) or not 0 < target <= 1:
        raise ValueError("Last axis must be the 0/1/3/5 calibration grid.")
    if not np.isfinite(values).all() or np.any((values < 0) | (values > 1)):
        raise ValueError("Invalid development accuracy.")
    reached = values >= target
    result = np.array([0, 12, 36, 60], float)[reached.argmax(axis=-1)]
    return np.where(reached.any(axis=-1), result, np.nan)
