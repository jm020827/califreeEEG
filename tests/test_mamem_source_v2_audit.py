import importlib.util
from pathlib import Path

import numpy as np
import pytest

SPEC = importlib.util.spec_from_file_location(
    "audit",
    Path(__file__).resolve().parents[1] / "scripts/analysis/audit_mamem_source_v2_results.py",
)
audit = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(audit)


def fixture():
    x = np.array([[1.0, 0.0], [2.0, 0.0], [3.0, 0.0]])
    y = np.full((3, 2), 0.4)
    scale = x.std(axis=0)
    scale[scale < 1e-9] = 1
    model = {
        "mean": x.mean(axis=0),
        "scale": scale,
        "coef": np.zeros((2, 2)),
        "intercept": [0.4, 0.4],
    }
    return x, y, x[:1], model, np.full((1, 2), 0.4)


def test_closed_form_fixture_without_fitting():
    residual, error = audit.audit_model(*fixture())
    assert residual < 1e-12 and error == 0


@pytest.mark.parametrize("field", ["mean", "scale", "intercept", "coef"])
def test_model_mutations_are_detected(field):
    args = list(fixture())
    args[3][field] = np.asarray(args[3][field]) + 0.1
    with pytest.raises(ValueError):
        audit.audit_model(*args)


def test_prediction_mutation_is_detected():
    args = list(fixture())
    args[-1] += 0.1
    with pytest.raises(ValueError, match="lambda_replay"):
        audit.audit_model(*args)
