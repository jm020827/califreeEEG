"""Generated mathematics only. No actual MAT, M training, or human accuracy."""

import copy
import hashlib
import importlib.util
import json
from functools import lru_cache
from pathlib import Path

import numpy as np
import pytest
from scipy.linalg import eigh

from cfeg import mamem_events_v1, mamem_events_v2
from cfeg import mamem_signal_diagnostic_v1 as diagnostic
from cfeg.mamem_signal_v1 import _center_and_scale, _quadrature


def references(frequency, harmonics=(1, 2)):
    time = np.arange(500) / 250
    basis = np.column_stack([fun(2 * np.pi * frequency * h * time)
                             for h in harmonics for fun in (np.sin, np.cos)])
    return basis - basis.mean(axis=0)


@lru_cache(maxsize=1)
def full_rank_canaries():
    all_refs = np.column_stack([np.ones(500), *[references(f) for f in diagnostic.FREQUENCIES]])
    assert np.linalg.matrix_rank(all_refs) == 21
    complete, _ = np.linalg.qr(all_refs, mode="complete")
    complement = complete[:, 21:21 + 256]
    target = np.linalg.qr(references(10.0), mode="reduced")[0]
    positive = np.vstack([2 * target.T, complement[:, :252].T])
    negative = complement.T
    assert np.linalg.matrix_rank(positive) == np.linalg.matrix_rank(negative) == 256
    return positive, negative, target


def window_result(selected):
    return diagnostic.diagnose_window(selected, {"nominal": 10., "time": 10., "sample": 10.},
                                      3, diagnostic.fixed_permutation())


@pytest.fixture(scope="module")
def canary_results():
    positive, negative, target = full_rank_canaries()
    return positive, negative, target, window_result(positive), window_result(negative)


def test_full_rank_positive_and_negative_have_exact_algebra(canary_results):
    _, _, target, positive, negative = canary_results
    eta = 4 / (4 + 1e-6 * 268 / 256)
    assert positive["original"]["cca"]["values"][3] == pytest.approx(eta, abs=1e-10)
    assert positive["reference"]["relative_energy_h1_h2"]["nominal"] == pytest.approx([8/268]*2)
    assert max(negative["original"]["cca"]["values"]) < 1e-20
    assert max(negative["original"]["weighted_energy"]["values"]) < 1e-25
    for j, f in enumerate(diagnostic.FREQUENCIES):
        uj = np.linalg.qr(references(f), mode="reduced")[0]
        expected = eta * np.linalg.svd(target.T @ uj, compute_uv=False)[0] ** 2
        assert positive["original"]["cca"]["values"][j] == pytest.approx(expected, abs=1e-10)
    assert positive["original"]["cca"]["inferred_label_margin"] > 0.5
    # A time permutation is not obliged to lower every negative example's score.
    assert max(negative["permuted"]["cca"]["values"]) > 0.1


def test_independent_gram_projection_and_eigen_whitening_match(canary_results):
    raw, _, _, report, _ = canary_results
    x = _center_and_scale(raw.copy())
    c = x @ x.T / 500
    lam, vectors = eigh(c)
    whitening = (vectors / np.sqrt(lam + 1e-6 * np.trace(c) / 256)) @ vectors.T
    for j, f in enumerate(diagnostic.FREQUENCIES):
        basis = references(f)
        gram_values, gram_vectors = eigh(basis.T @ basis)
        reference_white = (gram_vectors / np.sqrt(gram_values)) @ gram_vectors.T
        cross = whitening @ x @ basis @ reference_white / np.sqrt(500)
        expected = np.linalg.svd(cross, compute_uv=False)[0] ** 2
        assert report["original"]["cca"]["values"][j] == pytest.approx(expected, abs=1e-10)
        for h in (1, 2):
            v = references(f, (h,))
            xv = x @ v
            energy = np.trace(np.linalg.solve(v.T @ v, xv.T @ xv)) / np.sum(x*x)
            assert report["original"]["relative_energy_h1_h2"][j][h-1] == pytest.approx(energy)


def test_permutation_covariance_spectrum_and_cross_window_gram(canary_results):
    positive, negative, _, report, _ = canary_results
    p = diagnostic.fixed_permutation()
    np.testing.assert_array_equal(p, diagnostic.fixed_permutation())
    a, b = positive[:, p], negative[:, p]
    np.testing.assert_allclose(a @ a.T, positive @ positive.T, atol=1e-12)
    np.testing.assert_allclose(a @ b.T, positive @ negative.T, atol=1e-12)
    first = diagnostic.covariance_summary(positive @ positive.T / 500)
    second = diagnostic.covariance_summary(a @ a.T / 500)
    for key in first:
        np.testing.assert_allclose(first[key], second[key], atol=1e-10)
    assert report["permutation_covariance_relative_error"] < 1e-12


def test_overlap_symmetry_identity_and_fixed_shift():
    x = full_rank_canaries()[0]
    result = diagnostic.reference_diagnostics(x, {"nominal": 10., "time": 9.6, "sample": 10.})
    for entry in result["reference_overlap_h1_h2"]["nominal_sample"]:
        assert entry["mean_squared_overlap"] == pytest.approx(1)
    for h, entry in enumerate(result["reference_overlap_h1_h2"]["nominal_time"], 1):
        ua, ub = _quadrature((10*h,), 500), _quadrature((9.6*h,), 500)
        assert entry["mean_squared_overlap"] == pytest.approx(np.linalg.norm(ub.T @ ua)**2/2)
        assert 0 <= entry["mean_squared_overlap"] < .1


def test_common_quadrature_rotation_preserves_psd_not_arbitrary_scaling():
    b = np.random.default_rng(151).standard_normal((256, 2))
    angle = np.pi / 3
    rotation = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    moved = b @ rotation
    np.testing.assert_allclose(moved @ moved.T, b @ b.T, atol=1e-12)
    changed = b @ np.diag([2, 1])
    assert np.linalg.norm(changed @ changed.T - b @ b.T) > 1


def test_signed_margin_different_from_top_gap():
    result = diagnostic.score_summary([.8, .1, .2, .3, .4], 1)
    assert result["top_gap"] == pytest.approx(.4)
    assert result["inferred_label_margin"] == pytest.approx(-.7)


@pytest.mark.parametrize("change", ["scale", "offset", "channels"])
def test_window_invariances(change, canary_results):
    raw, _, _, original, _ = canary_results
    changed = (raw * -17 if change == "scale" else raw + np.arange(256)[:, None]
               if change == "offset" else raw[::-1])
    result = window_result(changed)
    for variant in ("original", "permuted"):
        for metric in ("cca", "weighted_energy"):
            np.testing.assert_allclose(result[variant][metric]["values"],
                                       original[variant][metric]["values"], atol=1e-9)


class Poison:
    def __float__(self):
        raise AssertionError("descriptor touched")


def generated_din():
    parts = []
    labels = [0, 1, 2, 3, 4, 0, 1, 2] + [j for j in range(5) for _ in range(3)]
    periods = [75.5, 66.5, 58.5, 52.5, 43.5]
    for i, j in enumerate(labels):
        rel = np.arange(int(5000 / periods[j]) + 1) * periods[j]
        group = np.empty((4, len(rel)), dtype=object)
        group[0] = Poison()
        group[2] = Poison()
        group[1] = 1000 + 10000 * i + rel
        group[3] = 101 + 2500 * i + np.rint(rel / 4).astype(int)
        parts.append(group)
    return np.concatenate(parts, axis=1)


@pytest.fixture(scope="module")
def integrated_result():
    din = generated_din()
    trials = mamem_events_v2.parse_main_trials(din, 60000, include_metadata=False)
    eeg = np.full((257, 60000), np.nan)
    for trial in trials:
        eeg[:256, trial["start0"]:trial["end0"]] = full_rank_canaries()[0]
    eeg.setflags(write=False)
    result = diagnostic.diagnose(eeg, din)
    return result


def test_complete_generated_pipeline_and_no_excluded_values(integrated_result):
    assert diagnostic.validate_report(integrated_result)
    assert integrated_result["scope"] == diagnostic.SCOPE
    assert len(integrated_result["groups"]) == 15
    for row in integrated_result["groups"]:
        assert row["mean_ms_per_sample"] == pytest.approx(4, abs=.01)
        assert row["max_interval_clock_residual_ms"] <= 4
        f = row["frequencies_hz"]
        assert f["time"] / f["sample"] == pytest.approx(4 / row["mean_ms_per_sample"])


def test_metadata_never_called_and_constant_sample_offset_cancels_in_clock(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("M called")
    monkeypatch.setattr(mamem_events_v1, "marker_features", forbidden)
    monkeypatch.setattr(mamem_events_v2, "marker_features", forbidden)
    monkeypatch.setattr(diagnostic, "diagnose_window", lambda *args: {"diagnostic_stub": True})
    eeg = np.zeros((257, 60000))
    din = generated_din()
    din[3] = [int(v) + 100 for v in din[3]]
    result = diagnostic.diagnose(eeg, din)
    assert result["groups"][0]["start0"] == 20450
    assert result["scope"]["M_features_computed"] is False


@pytest.mark.parametrize("field", ["scope", "permutation", "group", "score", "margin"])
def test_parent_rejects_tampered_report(field, integrated_result):
    result = copy.deepcopy(integrated_result)
    if field == "scope":
        result["scope"]["fits"] = 1
    elif field == "permutation":
        result["permutation"][0] = -1
    elif field == "group":
        result["groups"][0]["group_index"] = 7
    elif field == "score":
        result["groups"][0]["original"]["cca"]["values"] = [0] * 4
    else:
        result["groups"][0]["original"]["cca"]["inferred_label_margin"] = 17
    with pytest.raises(ValueError):
        diagnostic.validate_report(result)


@pytest.mark.parametrize("bad", [np.zeros((256, 500)), np.full((256, 500), np.inf),
                                  np.ones((255, 500)), np.ones((256, 500), dtype=np.float32)])
def test_bad_selected_input_stops(bad):
    with pytest.raises(ValueError):
        window_result(bad)


def test_float_permutation_stops_before_indexing():
    with pytest.raises(ValueError, match="permutation"):
        diagnostic.diagnose_window(full_rank_canaries()[0], {"nominal":10., "time":10., "sample":10.},
                                    3, diagnostic.fixed_permutation().astype(float))


@pytest.mark.parametrize("field", ["clock", "covariance", "overlap", "energy", "q2"])
def test_parent_rejects_more_saved_scalar_corruption(field, integrated_result):
    result = copy.deepcopy(integrated_result)
    row = result["groups"][0]
    if field == "clock":
        row["frequencies_hz"]["time"] = -1
    elif field == "covariance":
        row["covariance"]["entropy_effective_rank"] = -1
    elif field == "overlap":
        row["reference"]["reference_overlap_h1_h2"]["nominal_time"][0]["squared_singular_values"][0] = 2
    elif field == "energy":
        row["original"]["relative_energy_h1_h2"] = [[1, 1]]*5
    else:
        row["permuted"]["q2_channel_concentration_lag1"] = [1, 2]
    with pytest.raises(ValueError):
        diagnostic.validate_report(result)


def test_independent_saved_scalar_auditor_with_generated_report(tmp_path, monkeypatch, capsys,
                                                              integrated_result):
    root = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location("signal_audit", root / "scripts/analysis/audit_mamem_signal_validity_v1.py")
    auditor = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(auditor)
    monkeypatch.setattr(auditor, "RUN", tmp_path)
    paths = ["scripts/analysis/diagnose_mamem_signal_v1.py", "src/cfeg/mamem_signal_diagnostic_v1.py",
             "docs/mamem_signal_validity_v1_contract.md", "src/cfeg/mamem_events_v1.py",
             "src/cfeg/mamem_events_v2.py", "src/cfeg/mamem_signal_v1.py"]
    mat_sha = "57a72c3fde0ff3bc9aaae10721cd7cb450bda63eb5299a96f41704ea696ad10a"
    report = dict(integrated_result, mat_sha256=mat_sha)
    (tmp_path / "diagnostic.json").write_text(json.dumps(report))
    terminal = {"status": "COMPLETE", "fits": 0, "attempts": 1, "windows": 15,
                "diagnostic_sha256": hashlib.sha256((tmp_path / "diagnostic.json").read_bytes()).hexdigest()}
    (tmp_path / "terminal.json").write_text(json.dumps(terminal))
    (tmp_path / "manifest.json").write_text(json.dumps({"fits": 0, "attempt_budget": 1,
        "subject": "S001", "mat_sha256": mat_sha,
        "code_sha256": {p: hashlib.sha256((root/p).read_bytes()).hexdigest() for p in paths}}))
    auditor.main()
    assert json.loads(capsys.readouterr().out)["status"] == "PASS_SAVED_SCALARS_AND_RECORDED_SCOPE"
    report["groups"][0]["frequencies_hz"]["time"] += 1
    (tmp_path / "diagnostic.json").write_text(json.dumps(report))
    terminal["diagnostic_sha256"] = hashlib.sha256((tmp_path / "diagnostic.json").read_bytes()).hexdigest()
    (tmp_path / "terminal.json").write_text(json.dumps(terminal))
    with pytest.raises(ValueError, match="saved_scalar_arithmetic"):
        auditor.main()
