"""Toy-only producer checks; registered data/seeds are not executed here."""

import importlib.util
from pathlib import Path

import numpy as np
import pytest


@pytest.fixture(scope="module")
def runner():
    path = Path(__file__).resolve().parents[1] / "scripts/run_numerical_stability.py"
    spec = importlib.util.spec_from_file_location("numerical_runner_toy", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_grid_count_and_identity_repetition(runner):
    cases = list(runner.new_cases(17, "toy"))
    assert len(cases) == 96
    for arrays, meta in cases:
        _, b, c, g, *_ = arrays
        assert all(a.shape == (8, 8) and a.dtype == np.float64 for a in arrays)
        assert all(np.array_equal(a, a.T) for a in arrays)
        assert np.linalg.eigvalsh(b)[0] > 0 and np.linalg.eigvalsh(c)[0] > 0
        if meta["condition"] == 1:
            assert np.array_equal(c, meta["scale"] * np.eye(8))
        assert np.linalg.norm(g) > 0


def test_common_scale_uses_same_problem(runner):
    cases = list(runner.new_cases(19, "toy"))
    base = {
        (m["condition"], m["spectrum"], m["gap"], m["denominator"]): a
        for a, m in cases
        if m["scale"] == 1
    }
    for arrays, meta in cases:
        unscaled = base[(meta["condition"], meta["spectrum"], meta["gap"], meta["denominator"])]
        for value, original in zip(arrays, unscaled):
            np.testing.assert_allclose(value / meta["scale"], original, rtol=1e-12, atol=1e-14)


def test_nine_negative_cases_exact_positions(runner):
    s, b, c = runner.negative_cases()
    assert s.shape == b.shape == c.shape == (9, 8, 8)
    assert s[0, 0, 1] == b[1, 0, 1] == c[2, 0, 1] == 0.001
    assert np.isnan(s[3, 0, 0])
    assert b[4, 0, 0] == 0 and b[5, 0, 0] == -1 and c[6, 0, 0] == 0
    assert s[7, 6, 6] == 1 and s[8, 6, 6] == 1 - 5e-12


def test_descriptor_readonly_and_symlink_guards(runner, tmp_path):
    path = tmp_path / "test.json"
    runner.write_json(path, {"value": 1})
    assert runner.regular_readonly(path) == path
    assert runner.descriptor(path)["sha256"] == runner.sha(path)
    with pytest.raises(FileExistsError):
        runner.write_json(path, {})
    link = tmp_path / "link.json"
    link.symlink_to(path)
    with pytest.raises(ValueError, match="Unaliased"):
        runner.regular_readonly(link)
    path.chmod(0o600)
    with pytest.raises(ValueError, match="0400"):
        runner.regular_readonly(path)


def test_legacy_path_rejects_before_access(runner):
    config = {"known_generated_regression": {"result_path": "/forbidden/human/result.json"}}
    with pytest.raises(ValueError, match="Unauthorized legacy"):
        list(runner.load_known(config))


def test_gradient_selection_is_exact_without_registered_inputs():
    mask = np.zeros(1152, dtype=bool)
    mask[::60][:16] = True
    mask[851] = True
    mask[960:] = True
    assert mask.sum() == 209
    assert set(np.flatnonzero(mask[:960])) == set(range(0, 960, 60)) | {851}
