from __future__ import annotations

import numpy as np

from cfeg.metrics import classification_metrics, itr_bits_per_min


def test_classification_metrics_perfect_predictions():
    labels = np.array([0, 1, 2, 0])
    logits = np.eye(3)[labels] * 10.0
    metrics = classification_metrics(labels, logits, trial_time_sec=2.0)
    assert metrics["accuracy"] == 1.0
    assert metrics["balanced_accuracy"] == 1.0
    assert metrics["macro_f1"] == 1.0
    assert metrics["itr_bits_per_min"] > 0
    assert metrics["ece"] < 0.001


def test_itr_handles_chance_and_perfect_edges():
    assert itr_bits_per_min(4, 0.25, 2.0) == 0.0
    assert itr_bits_per_min(4, 1.0, 2.0) == 60.0
