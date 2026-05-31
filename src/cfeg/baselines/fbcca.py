from __future__ import annotations

import numpy as np


def make_reference_signals(freqs, sfreq, n_samples, n_harmonics=3):
    t = np.arange(n_samples) / float(sfreq)
    refs = []
    for freq in freqs:
        rows = []
        for h in range(1, n_harmonics + 1):
            rows.append(np.sin(2 * np.pi * h * freq * t))
            rows.append(np.cos(2 * np.pi * h * freq * t))
        refs.append(np.stack(rows, axis=0))
    return np.stack(refs, axis=0).astype(np.float32)


def cca_score(x, ref):
    x = x - x.mean(axis=-1, keepdims=True)
    ref = ref - ref.mean(axis=-1, keepdims=True)
    x_flat = x.reshape(-1)
    ref_proj = ref.mean(axis=0)
    denom = np.linalg.norm(x_flat[: len(ref_proj)]) * np.linalg.norm(ref_proj)
    if denom <= 0:
        return 0.0
    return float(abs(np.dot(x_flat[: len(ref_proj)], ref_proj) / denom))


def predict_fbcca(x, freqs, sfreq, filterbank=None):
    refs = make_reference_signals(freqs, sfreq, x.shape[-1])
    scores = [cca_score(x, ref) for ref in refs]
    return int(np.argmax(scores)), np.asarray(scores)

