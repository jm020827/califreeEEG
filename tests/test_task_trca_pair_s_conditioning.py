"""Toy structural checks of the conditioning CLI, not the frozen ladder run."""

import importlib.util
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "pair_conditioning_test", ROOT / "scripts/check_task_trca_pair_s_conditioning.py"
)
gate = importlib.util.module_from_spec(spec)
spec.loader.exec_module(gate)


def toy():
    s0 = np.broadcast_to(np.diag(np.arange(1, 9)), (5, 12, 8, 8)).copy().astype(float)
    c = np.broadcast_to(np.eye(8), s0.shape).copy()
    cross = np.zeros((5, 3, 12, 8, 8))
    return s0, c, cross, np.zeros((5, 3))


def test_toy_uniform_exact_and_scalar_batch_agreement():
    row, arrays = gate.measure(*toy())
    assert row["status"] == "PASS"
    assert row["uniform_s0_exact"]
    assert row["batch"]["maximum_symmetry_ratio"] == 0
    assert row["scalar_batch_f_max_error"] == 0
    np.testing.assert_array_equal(arrays["f_batch"], arrays["f_scalar"])


def test_rejection_retained_without_fallback(monkeypatch):
    def reject(*args):
        raise ValueError("fixed sentinel validity rejection")

    monkeypatch.setattr(gate.pair, "pair_projectors", reject)
    monkeypatch.setattr(gate.pair, "c_projectors", reject)
    row, arrays = gate.measure(*toy())
    assert row["status"] == "REJECTED"
    assert len(row["scalar_failures"]) == 60
    assert row["batch"]["error"] == "fixed sentinel validity rejection"
    assert "f_batch" not in arrays and "f_scalar" not in arrays
    assert "h_batch" in arrays and "h_scalar" in arrays
    assert row["first_failure"]["backend"] == "batch_projector"
    assert row["first_failure"]["band"] is None  # Injected error has no invented matrix.


def test_cli_status_mapping_does_not_report_rejection_as_success():
    assert gate.exit_code({"status": "GENERATED_CONDITIONING_PASS"}) == 0
    assert gate.exit_code({"status": "GENERATED_CONDITIONING_VALIDITY_FAILURE"}) == 2


def test_fresh_output_required_before_any_execution(tmp_path, monkeypatch):
    def forbidden():
        raise AssertionError("No generated draw after invalid output")

    monkeypatch.setattr(gate, "generated_support", forbidden)
    with pytest.raises(ValueError, match="Fresh unaliased"):
        gate.run(tmp_path)


def test_cpu_environment_gate_before_output_creation(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENBLAS_NUM_THREADS", "2")
    output = tmp_path / "new"
    with pytest.raises(ValueError, match="CPU1"):
        gate.run(output)
    assert not output.exists()


def test_frozen_recipe_shape_and_draw_order_without_operator_execution():
    rng = np.random.default_rng(20260909)
    expected = rng.normal(size=(12, 5, 8, 125)) + rng.normal(size=(5, 12, 5, 8, 125))
    np.testing.assert_array_equal(gate.generated_support(), expected)
    assert gate.CONDITIONS == (1, 1000, 100000, 120000)


def test_json_receipt_never_overwritten_and_rejects_nan(tmp_path):
    path = tmp_path / "receipt.json"
    gate.write_json(path, {"value": 1})
    assert path.stat().st_mode & 0o777 == 0o400
    with pytest.raises(FileExistsError):
        gate.write_json(path, {"value": 2})
    with pytest.raises(ValueError):
        gate.write_json(tmp_path / "nonfinite.json", {"value": float("nan")})
