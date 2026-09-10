"""Fixed support-only features and sufficient statistics; no filesystem access."""

import numpy as np
from scipy.signal import butter, sosfilt

FLOOR = 1e-12
FREQUENCIES = (5.45, 8.57, 12.0)


def references(fs=500, samples=500):
    t = np.arange(samples) / fs
    phase = 2 * np.pi * np.asarray(FREQUENCIES)[:, None, None] * np.arange(1, 4)[None, :, None] * t
    y = np.stack((np.sin(phase), np.cos(phase)), axis=2).reshape(3, 6, samples)
    return y - y.mean(-1, keepdims=True)


def eeg_trials(raw, markers, indices, channels, fs=500):
    if fs != 500:
        raise ValueError("fixed_eeg_rate")
    sos = butter(4, [4, 40], btype="bandpass", fs=fs, output="sos")
    rows = []
    for trial in indices:
        start, end = int(markers[trial]) - fs // 2, int(markers[trial]) + 3 * fs // 2
        if start < 0 or end > raw.shape[0]:
            raise ValueError("eeg_window_bounds")
        segment = np.asarray(raw[start:end, channels], dtype=np.float64)
        if not np.isfinite(segment).all():
            raise ValueError("nonfinite_selected_eeg")
        x = sosfilt(sos, segment, axis=0)[fs:].T
        x = x - x.mean(-1, keepdims=True)
        if not np.isfinite(x).all():
            raise ValueError("nonfinite_filtered_eeg")
        rows.append(x)
    return np.stack(rows)


def gyro_trials(raw, markers, indices, channels, fs=128):
    if fs != 128:
        raise ValueError("fixed_imu_rate")
    rows = []
    for trial in indices:
        start, end = int(markers[trial]) + fs // 2, int(markers[trial]) + 3 * fs // 2
        if start < 0 or end > raw.shape[0]:
            raise ValueError("imu_window_bounds")
        segment = np.asarray(raw[start:end, channels], dtype=np.float64).T
        if not np.isfinite(segment).all():
            raise ValueError("nonfinite_selected_imu")
        rows.append(segment)
    return np.stack(rows)


def support_statistics(x, labels, motion, reference=None):
    """x: exactly 3k centered trials, labels 0..2, motion: same selected trials."""
    y = references() if reference is None else reference
    labels = np.asarray(labels, dtype=int)
    if x.ndim != 3 or x.shape[0] != labels.size or motion.shape != (labels.size, 3, 128):
        raise ValueError("support_shape")
    if not np.isfinite(x).all() or not np.isfinite(motion).all():
        raise ValueError("nonfinite_support")
    counts = np.bincount(labels, minlength=3)
    if counts.shape != (3,) or min(counts) < 1 or len(set(counts)) != 1:
        raise ValueError("support_class_balance")
    floors = {}

    def floor(a, key):
        a = np.asarray(a)
        floors[key] = int((a < FLOOR).sum())
        return np.maximum(a, FLOOR)

    x = x - x.mean(-1, keepdims=True)
    grams = y @ y.transpose(0, 2, 1)
    residual = np.stack(
        [trial - trial @ y[c].T @ np.linalg.solve(grams[c], y[c]) for trial, c in zip(x, labels)]
    )
    total_var = np.mean(x * x, axis=(0, 2))
    residual_var = np.mean(residual * residual, axis=(0, 2))
    covariance = np.einsum("nct,ndt->cd", residual, residual) / (x.shape[0] * x.shape[2])
    residual_safe = floor(residual_var, "residual_variance")
    correlation = covariance / np.sqrt(residual_safe[:, None] * residual_safe[None, :])
    q = np.concatenate(
        (
            np.log(floor(total_var, "total_variance")),
            np.log(
                floor(residual_var / floor(total_var, "ratio_total_variance"), "residual_ratio")
            ),
            correlation[np.triu_indices(x.shape[1], k=1)],
        )
    )
    residual_norm = np.linalg.norm(residual, axis=1)
    rms = floor(np.sqrt(np.mean(residual_norm**2, axis=1)), "residual_norm_rms")
    q2 = np.array(
        [
            np.log(floor(residual_norm.std(axis=1) / rms, "q2_variation")).mean(),
            np.log(floor(np.quantile(residual_norm, 0.95, axis=1) / rms, "q2_tail")).mean(),
        ]
    )
    motion_norm = np.linalg.norm(motion, axis=1)
    motion_rms = floor(np.sqrt(np.mean(motion_norm**2, axis=1)), "motion_rms")
    m = np.array(
        [
            np.log(motion_rms).mean(),
            np.log(floor(motion_norm.std(axis=1) / motion_rms, "motion_variation")).mean(),
        ]
    )
    covariances, crosses = [], []
    for c in range(3):
        xc = x[labels == c]
        denominator = xc.shape[0] * xc.shape[-1]
        covariances.append(np.einsum("nct,ndt->cd", xc, xc) / denominator)
        crosses.append(np.einsum("nct,ht->ch", xc, y[c]) / denominator)
    result = {
        "q": q,
        "q2": q2,
        "m": m,
        "support_cov": np.stack(covariances),
        "support_cross": np.stack(crosses),
    }
    if not all(np.isfinite(a).all() for a in result.values()):
        raise ValueError("nonfinite_features")
    return result, floors


def query_statistics(x, reference=None):
    y = references() if reference is None else reference
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError("nonfinite_query_input")
    x = x - x.mean(-1, keepdims=True)
    result = {
        "query_cov": np.einsum("nct,ndt->ncd", x, x),
        "query_cross": np.einsum("nct,aht->nach", x, y),
        "reference_gram": y @ y.transpose(0, 2, 1),
    }
    if not all(np.isfinite(value).all() for value in result.values()):
        raise ValueError("nonfinite_query_statistics")
    return result
