"""No human-input tests of the standalone frozen-prior arithmetic."""

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "frozen_prior_diagnosis_test", ROOT / "scripts/diagnose_task_trca_temporal_frozen_prior.py"
)
diagnosis = importlib.util.module_from_spec(spec)
spec.loader.exec_module(diagnosis)


def fixture():
    scaler = lambda n: diagnosis.features.FeatureScaler(np.zeros(n), np.ones(n), (4, 6))
    pipeline = SimpleNamespace(
        q_scaler=scaler(15),
        q2_scaler=scaler(2),
        m_scaler=scaler(2),
        q=SimpleNamespace(coefficients=np.zeros(16)),
        residuals={
            arm: SimpleNamespace(coefficients=np.zeros(3)) for arm in ("Q2", "QM", "SHAM_REFIT")
        },
    )
    packet = np.broadcast_to(np.arange(1, 6)[None, :, None], (2, 5, 8)).copy().astype(float)
    packet[1] *= 2
    m = np.stack([diagnosis.features.metadata_features(row[:3])[0] for row in packet])
    data = {
        "keys": np.array([[4, 1, 125, 3], [6, 1, 125, 3]]),
        "orders": np.array([0, 0]),
        "packet5": packet,
        "q": np.zeros((2, 5, 8, 15)),
        "m": m,
        "available": np.ones((2, 8), bool),
    }
    return data, pipeline


def test_generated_zero_heads_are_uniform_and_exact_missing():
    data, pipeline = fixture()
    priors, donor = diagnosis.frozen_priors(data, pipeline, 0)
    assert donor == 6 and tuple(priors) == diagnosis.evaluation.POSITIVE_ARMS
    for value in priors.values():
        assert torch.equal(value, torch.ones((5, 8), dtype=torch.float64))


def test_generated_missing_residual_gate_includes_bias():
    data, pipeline = fixture()
    data["available"][:] = False
    for head in pipeline.residuals.values():
        head.coefficients[-1] = 1
    priors, _ = diagnosis.frozen_priors(data, pipeline, 0)
    for arm in ("Q", "Q2", "QM", "SHAM_REFIT", "PERMUTED", "MISSING"):
        assert torch.equal(priors[arm], priors["Q"])


def test_output_collision_precedes_all_input_reads(tmp_path):
    with pytest.raises(ValueError, match="New diagnostic output"):
        diagnosis.run(tmp_path)
