from __future__ import annotations

import math

import numpy as np


def accuracy(y_true, y_pred) -> float:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    return float((y_true == y_pred).mean()) if len(y_true) else 0.0


def itr_bits_per_min(n_classes: int, acc: float, trial_time_sec: float) -> float:
    if n_classes <= 1 or trial_time_sec <= 0:
        return 0.0
    p = min(max(acc, 1e-12), 1.0 - 1e-12)
    if acc <= 1.0 / n_classes:
        return 0.0
    bits = (
        math.log2(n_classes)
        + p * math.log2(p)
        + (1 - p) * math.log2((1 - p) / (n_classes - 1))
    )
    return float(bits * 60.0 / trial_time_sec)

