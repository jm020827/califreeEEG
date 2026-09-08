"""Unrelated seeds only. Never execute run_suite or expose its evaluation seed."""

import ast
import hashlib
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from cfeg.analysis import metadata_trca_prior_synthetic as syn


@pytest.fixture
def artificial():
    return syn.generate_participant(4701, "informative", 0)


@pytest.mark.parametrize("scenario", syn.SCENARIOS)
def test_generator_reproducible_and_artificial_geometry(scenario):
    a = syn.generate_participant(4701, scenario, 1)
    b = syn.generate_participant(4701, scenario, 1)
    np.testing.assert_array_equal(a.eeg, b.eeg)
    np.testing.assert_array_equal(a.impedance, b.impedance)
    assert a.eeg.shape == (8, 12, 1, 8, 125)
    assert np.isfinite(a.eeg).all() and (a.impedance > 0).all()
    np.testing.assert_allclose(a.eeg.mean(-1), 0, atol=1e-14)
    assert (np.any(a.oracle_q != 0)) == (scenario == "q_sufficient")


@pytest.mark.parametrize("k", syn.BUDGETS)
def test_feature_proxy_temporal_separation_and_units(artificial, k):
    p = artificial
    q = syn.support_features(p.eeg[:k], p.impedance[:k], p.oracle_q)
    y = syn.proxy_target(p.eeg[:k], p.eeg[5])
    assert q.shape == (1, 8, 6) and y.shape == (1, 8)
    changed = p.eeg.copy()
    changed[k:] = 123
    np.testing.assert_array_equal(q, syn.support_features(changed[:k], p.impedance[:k], p.oracle_q))
    assert not np.allclose(y, syn.proxy_target(changed[:k], changed[5]))
    np.testing.assert_allclose(
        q, syn.support_features(p.eeg[:k] * 100, p.impedance[:k], p.oracle_q), atol=1e-12
    )
    np.testing.assert_allclose(y, syn.proxy_target(p.eeg[:k] * 100, p.eeg[5] * 100), atol=1e-12)
    np.testing.assert_array_equal(q[..., 4], np.ones((1, 8)))


def test_metadata_missing_population_sd_and_units():
    x = np.array([[[1, np.nan, 4]], [[np.e, np.nan, np.nan]]])
    features, mask = syn.metadata_features(x)
    np.testing.assert_array_equal(mask, [[True, False, True]])
    assert features[0, 0, 1] == pytest.approx(0.5)
    assert features[0, 2, 1] == 0
    np.testing.assert_array_equal(features[0, 1], [0, 0])
    np.testing.assert_allclose(syn.metadata_features(x * 100)[0], features, atol=1e-14)
    all_missing, mask = syn.metadata_features(np.full((3, 1, 8), np.nan))
    assert not mask.any()
    np.testing.assert_array_equal(all_missing, np.zeros((1, 8, 2)))


@pytest.mark.parametrize("bad", [0, -1, np.inf, -np.inf])
def test_invalid_observed_metadata_rejected(bad):
    with pytest.raises(ValueError):
        syn.metadata_features(np.full((3, 1, 8), bad))


def test_natural_missing_changes_common_q_mask(artificial):
    packet = artificial.impedance[:3].copy()
    packet[:, :, 0] = np.nan
    q = syn.support_features(artificial.eeg[:3], packet, artificial.oracle_q)
    assert q[0, 0, 4] == 0 and np.all(q[0, 1:, 4] == 1)


def test_ridge_matches_independent_augmented_least_squares():
    rng = np.random.default_rng(731)
    x, y = rng.normal(size=(35, 3)), rng.normal(size=35)
    for intercept in (False, True):
        fit = syn.fit_linear(x, y, intercept=intercept)
        z = (x - x.mean(0)) / x.std(0)
        bias = y.mean() if intercept else 0
        expanded = np.vstack([z / np.sqrt(len(x)), np.sqrt(syn.RIDGE) * np.eye(3)])
        target = np.r_[(y - bias) / np.sqrt(len(x)), np.zeros(3)]
        expected = np.linalg.lstsq(expanded, target, rcond=None)[0]
        np.testing.assert_allclose(fit.coefficient, expected, atol=1e-13)
        assert fit.intercept == pytest.approx(bias)
    constant = syn.fit_linear(np.ones((10, 2)), np.arange(10), intercept=True)
    np.testing.assert_array_equal(constant.coefficient, np.zeros(2))


def test_derangement_stays_within_participant_partition():
    ids = np.arange(1, 24, 3)
    mapping = syn.derangement(ids, 483)
    assert set(mapping) == set(mapping.values()) == set(ids)
    assert all(a != b for a, b in mapping.items())
    assert mapping == syn.derangement(ids, 483)
    with pytest.raises(ValueError):
        syn.derangement(np.array([1]), 1)


def test_fit_receives_only_source_subjects_and_preserves_inputs():
    ids = np.array([1, 2, 4, 5])
    data = [syn.generate_participant(8843, "informative", int(i)) for i in ids]
    before = [p.eeg.copy() for p in data]
    models, receipt = syn.fit_fold(data, ids, 0, 0)
    assert receipt["rows"] == 4 * 2 * 8
    assert set(receipt["fit_ids"]) == set(ids)
    for a, p in zip(before, data):
        np.testing.assert_array_equal(a, p.eeg)
    prior, proxy = syn.priors_for_participant(data[0], data[1], 3, models)
    np.testing.assert_array_equal(prior["MISSING"], prior["Q"])
    np.testing.assert_array_equal(proxy["MISSING"], proxy["Q"])
    assert proxy["FULL"] is None and proxy["ISO"] is None
    for arm in syn.ARMS:
        np.testing.assert_allclose(prior[arm].sum(-1), 8, atol=1e-13)
    assert np.all(np.abs(proxy["QM"] - proxy["Q"]) <= syn.BOUND + 1e-14)
    with pytest.raises(ValueError):
        syn.fit_fold(data, np.array([0, 1, 2, 3]), 0, 0)


def test_channel_permutation_is_synchronized(artificial):
    order = np.array([4, 0, 2, 7, 1, 5, 3, 6])
    p = artificial
    q = syn.support_features(p.eeg[:3], p.impedance[:3], p.oracle_q)
    perm = syn.support_features(
        p.eeg[:3, :, :, order, :], p.impedance[:3, :, order], p.oracle_q[:, order]
    )
    np.testing.assert_allclose(perm, q[:, order], atol=1e-13)


def test_missing_donor_cannot_create_a_numeric_residual(artificial):
    base = syn.LinearModel(np.zeros(6), np.ones(6), np.zeros(6), 0.0)
    extra = syn.LinearModel(np.ones(2), np.ones(2), np.ones(2), 0.0)
    models = {"Q": base, "Q2": base, "QM": extra, "SHAM_REFIT": extra}
    donor = syn.ArtificialParticipant(
        artificial.eeg, np.full_like(artificial.impedance, np.nan), artificial.oracle_q
    )
    priors, proxy = syn.priors_for_participant(artificial, donor, 3, models)
    for arm in ("SHAM_REFIT", "PERMUTED"):
        np.testing.assert_array_equal(priors[arm], priors["Q"])
        np.testing.assert_array_equal(proxy[arm], proxy["Q"])


def test_summary_counts_and_descriptive_screen_from_handcrafted_rows():
    rows = []
    for scenario in syn.SCENARIOS:
        for k in syn.BUDGETS:
            for pid in range(24):
                for arm in syn.ARMS:
                    good = scenario == "informative" and arm == "QM"
                    rows.append(
                        {
                            "scenario": scenario,
                            "k": k,
                            "participant": pid,
                            "arm": arm,
                            "accuracy": 0.75 if good else 0.5,
                            "proxy_mse": None if arm in ("FULL", "ISO") else 0.1 if good else 0.2,
                            "predictions": [1 if good else 0] * 24,
                        }
                    )
    summary, contrasts, screen = syn.summarize(rows)
    assert len(summary) == 72 and len(contrasts) == 24
    assert screen["status"] == "SYNTHETIC_SCREEN_SUPPORTED"
    assert not screen["human_promotion"]
    record = next(
        r for r in summary if (r["scenario"], r["k"], r["arm"]) == ("informative", 3, "QM")
    )
    assert record["help_vs_Q"] == 24 and record["prediction_changes_vs_Q"] == 576
    for row in rows:
        if row["arm"] == "QM":
            row["accuracy"] = 0.5
    assert syn.summarize(rows)[2]["status"] == "SYNTHETIC_CANDIDATE_NOT_ESTABLISHED"


def test_synthetic_modules_have_no_data_loading_imports():
    tree = ast.parse(Path(syn.__file__).read_text())
    names = {node.module for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)}
    assert names == {"__future__", "dataclasses", "cfeg.analysis"}
    imports = {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    }
    assert imports == {"numpy"}
    assert "open(" not in Path(syn.__file__).read_text()
    assert "np.testing" not in Path(syn.__file__).read_text()


def test_frozen_json_and_runner_guards(tmp_path):
    root = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location(
        "prior_synthetic_runner", root / "scripts/run_metadata_trca_prior_synthetic.py"
    )
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)
    assert hashlib.sha256(runner.PLAN.read_bytes()).hexdigest() == runner.PLAN_SHA
    plan = json.loads(runner.PLAN.read_text())
    assert tuple(plan["arms"]) == syn.ARMS and plan["seed"] == syn.SEED
    for event, args in [
        ("open", ("/nonexistent/file.mat", "r", 0)),
        ("open", ("/nonexistent/file.npz", "r", 0)),
        ("socket.connect", (None,)),
        ("subprocess.Popen", (None,)),
    ]:
        with pytest.raises(RuntimeError):
            runner.guard(event, args)
    result = tmp_path / "tiny.json"
    digest = runner.publish(result, {"only": "artificial-test"})
    assert hashlib.sha256(result.read_bytes()).hexdigest() == digest
    with pytest.raises(FileExistsError):
        runner.publish(result, {})
    assert result.stat().st_mode & 0o777 == 0o400


def test_fresh_child_control_flow_without_testing_preimports():
    # This is a cold control-flow fixture, not the declared numerical suite:
    # duplicate four seed4701 participants and stub the eigensolver/scoring.
    root = Path(__file__).resolve().parents[1]
    code = """
import json, sys
from pathlib import Path
from types import SimpleNamespace
sys.path.insert(0,str(Path.cwd()/"src"))
sys.path.insert(0,str(Path.cwd()/"scripts"))
import numpy as np
from cfeg.analysis import metadata_trca_prior_synthetic as syn
from run_metadata_trca_prior_synthetic import guard
fixtures={s:syn.generate_participant(4701,s,0) for s in syn.SCENARIOS}
syn.generate_participant=lambda seed,scenario,participant: fixtures[scenario]
syn.op.fit_trca=lambda *args: SimpleNamespace(diagnostics={
    "denominator_min_eigenvalues":[[1.0]], "top_eigenvalue_gaps":[[1.0]],
    "denominator_norm_errors":[[0.0]], "penalty_trace":[[1.0]]})
syn.op.score_trca=lambda *args: (np.zeros((24,12)),np.zeros((24,1,12)))
assert "numpy.testing" not in sys.modules
sys.addaudithook(guard)
out=syn.run_suite()
assert "numpy.testing" not in sys.modules
print(json.dumps({"rows":len(out["rows"]),"attainment":len(out["attainment"])}))
"""
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=root,
        text=True,
        capture_output=True,
        timeout=30,
        check=True,
    )
    assert json.loads(result.stdout) == {"rows": 1728, "attainment": 864}
