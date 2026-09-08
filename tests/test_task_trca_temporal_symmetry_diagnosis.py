"""Generated-only checks of the narrowly scoped postfailure replay."""

import hashlib
import importlib.util
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "symmetry_diagnosis_test", ROOT / "scripts/diagnose_task_trca_temporal_symmetry.py"
)
diagnosis = importlib.util.module_from_spec(spec)
spec.loader.exec_module(diagnosis)


def test_generated_symmetric_spd_conditions_pass_without_policy_change():
    s = np.broadcast_to(np.diag(np.arange(1, 9)), (5, 12, 8, 8)).copy()
    c = np.broadcast_to(np.eye(8), s.shape).copy()
    result = diagnosis.measure(s, c)
    assert set(result["operators"]) == {"C1_ISO", "C2_UNIFORM_NECESSARY_CONDITION"}
    for row in result["operators"].values():
        assert row["maximum_ratio"] == 0
        assert row["strict_projector_status"] == "PASS"


def test_generated_nonsymmetric_input_is_not_silently_repaired():
    s = np.broadcast_to(np.diag(np.arange(1, 9)), (5, 12, 8, 8)).copy().astype(float)
    c = np.broadcast_to(np.eye(8), s.shape).copy()
    s[0, 0, 0, 1] = 1
    with pytest.raises(ValueError, match="s must be symmetric"):
        diagnosis.measure(s, c)


def test_diagnostic_overwrite_rejected_before_human_inputs(tmp_path):
    with pytest.raises(ValueError, match="New unaliased"):
        diagnosis.run(tmp_path)


def test_saved_source_only_decodes_explicit_s_c_fields(tmp_path, monkeypatch):
    path = tmp_path / "generated.npz"
    s = np.broadcast_to(np.eye(8), (1, 5, 12, 8, 8)).copy()
    np.savez(path, keys=np.array([diagnosis.KEY]), s=s, c=s, numeric_metadata=np.ones(8))
    seen = []
    original = np.lib.npyio.NpzFile.__getitem__

    def read(archive, name):
        seen.append(name)
        assert name in ("keys", "s", "c")
        return original(archive, name)

    monkeypatch.setattr(np.lib.npyio.NpzFile, "__getitem__", read)
    item = {
        "path": str(path),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "bytes": path.stat().st_size,
    }
    result = diagnosis.source_matrices(item)
    assert seen == ["keys", "s", "c"]
    np.testing.assert_array_equal(result["s"], s[0])
