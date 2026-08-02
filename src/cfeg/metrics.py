from __future__ import annotations

import math

import numpy as np


def accuracy(y_true, y_pred) -> float:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    return float((y_true == y_pred).mean()) if len(y_true) else 0.0


def confusion_matrix(y_true, y_pred, n_classes: int | None = None) -> np.ndarray:
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    if n_classes is None:
        largest = max(y_true.max(initial=0), y_pred.max(initial=0))
        n_classes = int(largest) + 1
    matrix = np.zeros((n_classes, n_classes), dtype=np.int64)
    np.add.at(matrix, (y_true, y_pred), 1)
    return matrix


def balanced_accuracy(y_true, y_pred, n_classes: int | None = None) -> float:
    matrix = confusion_matrix(y_true, y_pred, n_classes)
    support = matrix.sum(axis=1)
    present = support > 0
    recall = np.divide(
        np.diag(matrix), support, out=np.zeros_like(support, dtype=float), where=present
    )
    return float(recall[present].mean()) if present.any() else 0.0


def macro_f1(y_true, y_pred, n_classes: int | None = None) -> float:
    matrix = confusion_matrix(y_true, y_pred, n_classes)
    support = matrix.sum(axis=1)
    present = support > 0
    tp = np.diag(matrix).astype(float)
    fp = matrix.sum(axis=0) - tp
    fn = matrix.sum(axis=1) - tp
    denom = 2 * tp + fp + fn
    f1 = np.divide(2 * tp, denom, out=np.zeros_like(tp), where=denom > 0)
    return float(f1[present].mean()) if present.any() else 0.0


def expected_calibration_error(
    probabilities: np.ndarray, y_true: np.ndarray, n_bins: int = 15
) -> float:
    probabilities = np.asarray(probabilities, dtype=float)
    y_true = np.asarray(y_true, dtype=int)
    if not len(y_true):
        return 0.0
    confidence = probabilities.max(axis=1)
    predicted = probabilities.argmax(axis=1)
    correct = predicted == y_true
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    total = float(len(y_true))
    ece = 0.0
    for lower, upper in zip(edges[:-1], edges[1:]):
        selected = (confidence > lower) & (confidence <= upper)
        if lower == 0.0:
            selected |= confidence == 0.0
        if selected.any():
            ece += selected.sum() / total * abs(correct[selected].mean() - confidence[selected].mean())
    return float(ece)


def classification_metrics(
    y_true: np.ndarray,
    logits: np.ndarray,
    *,
    trial_time_sec: float | None = None,
    n_bins: int = 15,
) -> dict[str, float]:
    y_true = np.asarray(y_true, dtype=int)
    logits = np.asarray(logits, dtype=float)
    if logits.ndim != 2:
        raise ValueError(f"logits must be [N,K], got {logits.shape}")
    shifted = logits - logits.max(axis=1, keepdims=True)
    probabilities = np.exp(shifted)
    probabilities /= probabilities.sum(axis=1, keepdims=True)
    predicted = probabilities.argmax(axis=1)
    n_classes = logits.shape[1]
    picked = probabilities[np.arange(len(y_true)), y_true].clip(1e-12, 1.0)
    acc = accuracy(y_true, predicted)
    metrics = {
        "accuracy": acc,
        "balanced_accuracy": balanced_accuracy(y_true, predicted, n_classes),
        "macro_f1": macro_f1(y_true, predicted, n_classes),
        "nll": float(-np.log(picked).mean()) if len(picked) else 0.0,
        "ece": expected_calibration_error(probabilities, y_true, n_bins=n_bins),
        "n_samples": int(len(y_true)),
    }
    if trial_time_sec is not None:
        metrics["itr_bits_per_min"] = itr_bits_per_min(n_classes, acc, trial_time_sec)
    return metrics


def itr_bits_per_min(n_classes: int, acc: float, trial_time_sec: float) -> float:
    if n_classes <= 1 or trial_time_sec <= 0 or acc <= 1.0 / n_classes:
        return 0.0
    p = min(max(float(acc), 0.0), 1.0)
    error_term = 0.0 if p >= 1.0 else (1.0 - p) * math.log2((1.0 - p) / (n_classes - 1))
    correct_term = 0.0 if p <= 0.0 else p * math.log2(p)
    bits = math.log2(n_classes) + correct_term + error_term
    return float(bits * 60.0 / trial_time_sec)
