"""Unrelated synthetic engineering fixtures; no human study inputs or outcomes."""

from __future__ import annotations

import copy
import inspect
import json
import stat
import time
from pathlib import Path

import numpy as np
import pytest
import yaml
from scipy import signal
from scipy.linalg import qr, solve_triangular, svd

from cfeg.analysis import reference_calibration_source as study
from cfeg.baselines.fbcca import apply_filterbank, cca_score, predict_fbcca

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def plan():
    return json.loads(
        (ROOT / "configs/analysis/reference_calibration_source39_v1.json").read_text()
    )


@pytest.fixture
def bank(plan):
    return yaml.safe_load((ROOT / plan["filterbank_config"]).read_text())


def scalar_pair(x, y, ridge):
    """Independent scalar covariance-SVD plus original-waveform Pearson oracle."""
    x, y = x - x.mean(axis=-1, keepdims=True), y - y.mean(axis=-1, keepdims=True)
    if min(np.linalg.norm(x), np.linalg.norm(y)) <= np.finfo(float).eps * np.sqrt(x.shape[-1]) * 10:
        return np.zeros(len(x)), np.zeros(len(y)), 0.0

    def inverse(value):
        covariance = value @ value.T / (value.shape[-1] - 1)
        covariance += ridge * np.trace(covariance) / len(covariance) * np.eye(len(covariance))
        d, v = np.linalg.eigh(covariance)
        roots = np.zeros_like(d)
        mask = d > max(np.finfo(float).eps, max(d[-1], 0) * 1e-12)
        roots[mask] = 1 / np.sqrt(d[mask])
        return (v * roots) @ v.T

    ix, iy = inverse(x), inverse(y)
    u, _, vh = np.linalg.svd(ix @ (x @ y.T / (x.shape[-1] - 1)) @ iy, full_matrices=False)
    wx, wy = ix @ u[:, 0], iy @ vh[0]
    return wx, wy, scalar_corr(wx @ x, wy @ y)


def scalar_corr(x, y):
    x, y = x - x.mean(), y - y.mean()
    denominator = np.linalg.norm(x) * np.linalg.norm(y)
    return 0.0 if denominator <= 0 else float(np.clip(x @ y / denominator, -1, 1))


def scalar_ecca(x, template, reference, ridge):
    u1, v1, _ = scalar_pair(x, reference, ridge)
    u2, _, _ = scalar_pair(x, template, ridge)
    u3, _, _ = scalar_pair(template, reference, ridge)
    return np.asarray(
        [
            scalar_corr(u1 @ x, v1 @ reference),
            scalar_corr(u2 @ x, u2 @ template),
            scalar_corr(u1 @ x, u1 @ template),
            scalar_corr(u3 @ x, u3 @ template),
        ]
    )


def test_scalar_four_feature_and_fixed_reference_wrong_label_oracle(plan, bank):
    rng = np.random.default_rng(928101)
    active = copy.deepcopy(plan)
    active["n_classes"], active["frequencies"] = 3, plan["frequencies"][:3]
    workspace = study.build_workspace(active, bank, 125)
    support, query = rng.normal(size=(3, 3, 8, 125)), rng.normal(size=(4, 8, 125))
    q = study.prepare_query(query, workspace)
    model = study.fit_ecca_templates(support, workspace)
    actual = study.ecca_query_features(q, model)
    expected = np.array(
        [
            [
                scalar_ecca(x, support.mean(axis=0)[c], r, 1e-8)
                for c, r in enumerate(workspace["references"])
            ]
            for x in query
        ]
    )
    np.testing.assert_allclose(actual["components"], expected, atol=1e-10, rtol=1e-10)
    correct_scores = np.sum(np.sign(expected) * expected**2, axis=-1)
    for shift in (1, 2):
        shifted = np.array(
            [
                [
                    scalar_ecca(x, support.mean(axis=0)[(c - shift) % 3], r, 1e-8)
                    for c, r in enumerate(workspace["references"])
                ]
                for x in query
            ]
        )
        expected_scores = np.sum(np.sign(shifted) * shifted**2, axis=-1)
        np.testing.assert_allclose(
            actual["wrong_band_scores"][shift - 1], expected_scores, atol=1e-10
        )
        assert not np.allclose(expected_scores, np.roll(correct_scores, shift, axis=-1), atol=1e-5)
        np.testing.assert_array_equal(shifted[..., 0], expected[..., 0])
    pair = study.canonical_pairwise(
        study.covariance_statistics(query, 0.01),
        study.covariance_statistics(workspace["references"], 0.01),
    )
    assert np.max(np.abs(pair["pearson"] - pair["singular"])) > 0.001


def test_unregularized_full_rank_qr_oracle_and_degenerate_pairs():
    rng = np.random.default_rng(928102)
    x, y = rng.normal(size=(5, 90)), rng.normal(size=(6, 90))
    xc, yc = x - x.mean(axis=1, keepdims=True), y - y.mean(axis=1, keepdims=True)
    qx, rx = qr(xc.T, mode="economic")
    qy, ry = qr(yc.T, mode="economic")
    u, singular, vh = svd(qx.T @ qy, full_matrices=False)
    wx, wy = solve_triangular(rx, u[:, 0]), solve_triangular(ry, vh[0])
    result = study.canonical_pairwise(
        study.covariance_statistics(x[None], 0), study.covariance_statistics(y[None], 0)
    )
    assert result["pearson"][0, 0] == pytest.approx(scalar_corr(wx @ x, wy @ y), abs=1e-12)
    assert result["singular"][0, 0] == pytest.approx(singular[0], abs=1e-12)
    for value in (np.zeros_like(x), np.ones_like(x) * 10):
        pair = study.canonical_pairwise(
            study.covariance_statistics(value[None], 1e-8),
            study.covariance_statistics(y[None], 1e-8),
        )
        assert not pair["u"].any() and not pair["v"].any() and not pair["pearson"].any()
    duplicate = np.vstack((x, x[0]))
    pair = study.canonical_pairwise(
        study.covariance_statistics(duplicate[None], 0.01),
        study.covariance_statistics(y[None], 0.01),
    )
    assert pair["singular"][0, 0] == pytest.approx(cca_score(duplicate, y, 0.01), abs=1e-12)


def test_full_bank_fixture_benchmark_and_parity(plan, bank):
    rng = np.random.default_rng(928103)
    crop = rng.normal(size=(10, 12, 8, 125))
    workspace = study.build_workspace(plan, bank, 125)
    begin = time.perf_counter()
    actual = study.extract_cell(crop, workspace, include_legacy=True)
    elapsed = time.perf_counter() - begin
    print(f"\nsynthetic full12class/7band/N125 cell elapsed={elapsed:.3f}s")
    assert actual["ecca_components"].shape == (3, 7, 60, 12, 4)
    assert actual["reference_projection_fraction"].shape == (7, 10, 12, 12)
    for k in range(3):
        np.testing.assert_array_equal(actual["ecca_components"][k, ..., 0], actual["aref_band_rho"])
    filtered, _ = apply_filterbank(crop.reshape(-1, 8, 125), sfreq=250, filterbank=bank)
    grid = filtered.reshape(7, 10, 12, 8, 125)
    for band in (0, 3, 6):
        for k_index, k in enumerate(study.BUDGETS):
            expected = scalar_ecca(
                grid[band, 5, 0], grid[band, :k, 0].mean(axis=0), workspace["references"][0], 1e-8
            )
            np.testing.assert_allclose(
                actual["ecca_components"][k_index, band, 0, 0], expected, atol=1e-10
            )
    expected = predict_fbcca(crop[5, 0], plan["frequencies"], 250, filterbank=bank)[1]
    np.testing.assert_allclose(
        actual["legacy_a0_scores"][0], expected / np.sum(bank["weights"]), atol=1e-12
    )


def test_causal_notch_prefix_trial_independence_and_future_invariance(plan):
    rng = np.random.default_rng(928104)
    raw = rng.normal(size=tuple(plan["raw_shape"]))
    short = {"n_samples": 125, "interface_index": 0}
    long = {"n_samples": 250, "interface_index": 0}
    result = study.prepare_pipeline_crop(raw, short, plan, "causal_notch50")
    longer = study.prepare_pipeline_crop(raw, long, plan, "causal_notch50")
    np.testing.assert_array_equal(result, longer[..., :125])
    b, a = signal.iirnotch(50, 35, fs=250)
    expected = signal.lfilter(b, a, raw[:, :285, 0, 5, 2].astype(float), axis=-1)[:, 160:]
    np.testing.assert_array_equal(result[5, 2], expected)
    changed = raw.copy()
    changed[:, 285:] = rng.normal(size=changed[:, 285:].shape) * 500
    np.testing.assert_array_equal(
        result, study.prepare_pipeline_crop(changed, short, plan, "causal_notch50")
    )
    changed = raw.copy()
    changed[:, :, 0, 6, 2] *= 100
    np.testing.assert_array_equal(
        result[5, 2], study.prepare_pipeline_crop(changed, short, plan, "causal_notch50")[5, 2]
    )
    no_notch = study.prepare_pipeline_crop(raw, short, plan, "no_notch")
    np.testing.assert_array_equal(no_notch, raw[:, 160:285, 0].transpose(2, 3, 0, 1))
    assert no_notch.flags.c_contiguous and no_notch.dtype == np.float64


def test_descriptive_projection_and_support_pearson_independent_oracle(plan, bank):
    rng = np.random.default_rng(928105)
    workspace = study.build_workspace(plan, bank, 125)
    crop = rng.normal(size=(10, 12, 8, 125))
    filtered, _ = apply_filterbank(crop, sfreq=250, filterbank=bank)
    diagnostic = study.signal_diagnostics(crop, filtered, workspace)
    assert np.all(
        (diagnostic["reference_projection_fraction"] >= 0)
        & (diagnostic["reference_projection_fraction"] <= 1)
    )
    x = filtered[2, 1, 4]
    x = x - x.mean(axis=-1, keepdims=True)
    reference = workspace["references"][7]
    centered_ref = reference - reference.mean(axis=-1, keepdims=True)
    projected = x @ np.linalg.pinv(centered_ref) @ centered_ref
    assert diagnostic["reference_projection_fraction"][2, 1, 4, 7] == pytest.approx(
        np.sum(projected**2) / np.sum(x**2), abs=1e-12
    )
    y = filtered[2, 3, 4]
    y = y - y.mean(axis=-1, keepdims=True)
    assert diagnostic["support_correlations"][2, 4, 1, 3] == pytest.approx(
        scalar_corr(x.ravel(), y.ravel()), abs=1e-12
    )
    zero = study.signal_diagnostics(np.zeros_like(crop), np.zeros_like(filtered), workspace)
    assert all(not value.any() for value in zero.values())


def artificial_features(plan, subjects=6):
    rng = np.random.default_rng(928106)
    p, a, l, k, f, q, c = subjects, 2, len(study.spatial.cells(plan)), 3, 2, 24, 12
    r1 = rng.uniform(0.1, 0.7, (p, a, l, f, q, c))
    components = rng.uniform(-0.3, 0.3, (p, a, l, k, f, q, c, 4))
    components[..., 0] = r1[:, :, :, None]
    return {
        "schema": np.asarray(study.FEATURE_SCHEMA),
        "subject_ids": np.arange(201, 201 + p),
        "pipeline_ids": np.asarray(study.PIPELINES),
        "cell_ids": np.asarray([cell["cell_id"] for cell in study.spatial.cells(plan)]),
        "sample_counts": np.asarray([cell["n_samples"] for cell in study.spatial.cells(plan)]),
        "interface_indices": np.asarray(
            [cell["interface_index"] for cell in study.spatial.cells(plan)]
        ),
        "query_labels": np.tile(np.arange(c), 2),
        "query_block_ids": np.repeat([5, 6], c),
        "support_block_ids": np.arange(5),
        "calibration_budgets": np.asarray(study.BUDGETS),
        "wrong_label_shifts": np.arange(1, c),
        "filter_weights": np.asarray([1.25, 0.7]),
        "ecca_components": components,
        "wrong_ecca_scores": rng.uniform(0, 1, (p, a, l, k, 11, q, c)),
        "itcca_band_rho": rng.uniform(0.1, 0.7, (p, a, l, k, f, q, c)),
        "aref_band_rho": r1,
        "legacy_a0_scores": rng.uniform(0, 1, (p, l, q, c)),
        "support_correlations": np.zeros((p, a, l, f, c, 5, 5)),
        "reference_projection_fraction": np.zeros((p, a, l, f, 10, c, c)),
        "line50_fraction": np.zeros((p, a, l, 10, c)),
        "raw_file_sha256": np.asarray(["a" * 64] * p),
        "crop_sha256": np.full((p, a, l), "b" * 64),
    }


def test_prefix_support_query_and_shared_filter_sign_locality(plan, bank):
    rng = np.random.default_rng(928107)
    active = copy.deepcopy(plan)
    active["n_classes"], active["frequencies"] = 3, plan["frequencies"][:3]
    workspace = study.build_workspace(active, bank, 125)
    crop = rng.normal(size=(10, 3, 8, 125))
    original = study.extract_cell(crop, workspace)
    modified = crop.copy()
    modified[3:5] *= 7
    changed = study.extract_cell(modified, workspace)
    for key in ("ecca_components", "wrong_ecca_scores", "itcca_band_rho"):
        np.testing.assert_array_equal(original[key][:2], changed[key][:2])
    np.testing.assert_array_equal(original["aref_band_rho"], changed["aref_band_rho"])
    modified = crop.copy()
    modified[5, 0] *= 7
    changed = study.extract_cell(modified, workspace)
    np.testing.assert_array_equal(
        original["ecca_components"][..., 1:, :, :], changed["ecca_components"][..., 1:, :, :]
    )
    np.testing.assert_array_equal(
        original["wrong_ecca_scores"][..., 1:, :], changed["wrong_ecca_scores"][..., 1:, :]
    )
    query = study.prepare_query(crop[5:].reshape(-1, 8, 125), workspace)
    model = study.fit_ecca_templates(crop[:3], workspace)
    before = study.ecca_query_features(query, model)
    changed_query, changed_model = copy.deepcopy(query), copy.deepcopy(model)
    changed_query["reference_pair"]["u"] *= -1
    changed_query["reference_pair"]["v"] *= -1
    changed_model["reference_pair"]["u"] *= -1
    after = study.ecca_query_features(changed_query, changed_model)
    np.testing.assert_array_equal(before["components"], after["components"])
    single = study.ecca_query_features(study.prepare_query(crop[5, :1], workspace), model)
    np.testing.assert_allclose(single["components"][0], before["components"][0], atol=1e-12)


def test_extraction_shape_and_pipeline_axis_without_future_value_inspection(plan, monkeypatch):
    rng = np.random.default_rng(928108)
    raw = rng.normal(size=tuple(plan["raw_shape"]))
    captured = []

    def extract(crop, workspace, *, include_legacy=False):
        captured.append(crop.copy())
        result = {key: np.zeros(1) for key in study.FEATURE_KEYS}
        if include_legacy:
            result["legacy_a0_scores"] = np.zeros(1)
        return result

    monkeypatch.setattr(study, "extract_cell", extract)
    raw[:, 410:] = np.nan  # Outside every frozen available prefix.
    features = study.extract_features(raw, plan, {n: None for n in plan["sample_counts"]})
    assert len(captured) == 12 and features["crop_sha256"].shape == (2, 6)
    for c, cell in enumerate(study.spatial.cells(plan)):
        expected = raw[:, 160 : 160 + cell["n_samples"], cell["interface_index"]].transpose(
            2, 3, 0, 1
        )
        np.testing.assert_array_equal(captured[c], expected)
        assert captured[c].flags.c_contiguous
    np.testing.assert_array_equal(captured[6], captured[8][..., :125])


def test_all_temperature_objectives_fit_only_and_diagnostic_independence(plan):
    features = artificial_features(plan)
    positions = np.arange(len(features["subject_ids"]))
    fit_indices = np.flatnonzero(positions % 3 != 0)
    freeze = study.fit_fold(features, fit_indices, plan, fold_id=0)
    changed = copy.deepcopy(features)
    for key in ("ecca_components", "itcca_band_rho", "aref_band_rho", "legacy_a0_scores"):
        changed[key][[0, 3]] *= -3
    for key in (
        "support_correlations",
        "reference_projection_fraction",
        "line50_fraction",
        "wrong_ecca_scores",
    ):
        changed[key] += 20
    assert study.fit_fold(changed, fit_indices, plan, fold_id=0) == freeze
    count = sum(len(fit["objectives"]) for fit in freeze["legacy_a0_temperatures"].values())
    for pipeline in freeze["pipelines"].values():
        for window in pipeline["windows"].values():
            count += len(window["aref_temperature"]["objectives"])
            for budget in window["budgets"].values():
                count += len(budget["ecca_temperature"]["objectives"])
                count += len(budget["itcca_temperature"]["objectives"])
    assert count * 3 == plan["fitting"]["expected_objective_count"] == 4185
    scores = study.decoder_scores(features)
    scalar = scores["ECCA"][fit_indices, 1][:, [0, 1], 0]
    expected = [
        study.spatial.nll(study.numeric.softmax(scalar, t), features["query_labels"])
        for t in np.logspace(-4, 1, 31)
    ]
    assert (
        freeze["pipelines"]["causal_notch50"]["windows"]["125"]["budgets"]["1"]["ecca_temperature"][
            "objectives"
        ]
        == expected
    )
    with pytest.raises(ValueError):
        study.fit_fold(features, positions, plan, fold_id=0)


def test_full_row_grid_exact_aliases_and_permutation_metrics(plan):
    features = artificial_features(plan)
    positions = np.arange(len(features["subject_ids"]))
    rows = []
    for fold in range(3):
        freeze = study.fit_fold(features, np.flatnonzero(positions % 3 != fold), plan, fold_id=fold)
        rows.extend(
            study.evaluate_fold(features, np.flatnonzero(positions % 3 == fold), freeze, plan)
        )
    assert len(rows) == 6 * 6 * 34
    assert len(rows) / 6 * 39 == plan["reporting"]["expected_rows"] == 7956
    lookup = {
        (r["subject_id"], r["pipeline"], r["cell_id"], r["method"], r["budget"]): r for r in rows
    }
    for row in rows:
        key = (row["subject_id"], row["pipeline"], row["cell_id"])
        if row["budget"] == 0 and row["method"] in ("ECCA", "wrong_label_support"):
            assert row["posterior_sha256"] == lookup[(*key, "A0_reference", 0)]["posterior_sha256"]
        if row["method"] == "legacy_A0":
            assert row["pipeline"] == "no_notch"
        if row["method"] == "wrong_label_support":
            assert row["permutations"] == (0 if row["budget"] == 0 else 11)
            if row["budget"]:
                assert row["posterior_sha256"] is None
                assert row["balanced_accuracy"] == np.mean(
                    [p["balanced_accuracy"] for p in row["permutation_metrics"]]
                )
    summary = study.summarize_results(rows, plan)
    assert summary["status"] == "DIAGNOSTIC_COMPLETE" and not summary["human_held_unlock"]
    assert len(summary["paired_pipeline_diagnostics"]) == 7
    for pipeline, nested in summary["pipeline_summaries"].items():
        assert len(nested["aggregate_rows"]) == (19 if pipeline == "no_notch" else 15) * 6
        assert all(record["n"] == 6 for record in nested["comparisons"].values())
        assert not any(r["method"] in ("A0", "AQ_ITCCA") for r in nested["aggregate_rows"])
        for record in nested["calibration_attainment"]:
            expected = [
                12 * k * 0.64 if isinstance(k, int) else None
                for k in record["first_observed_budget"]
            ]
            assert record["history_seconds"] == expected
    expected = np.asarray(
        summary["pipeline_summaries"]["causal_notch50"]["comparisons"]["AQ_eauc_gain"][
            "participant_values"
        ]
    ) - np.asarray(
        summary["pipeline_summaries"]["no_notch"]["comparisons"]["AQ_eauc_gain"][
            "participant_values"
        ]
    )
    np.testing.assert_array_equal(
        summary["paired_pipeline_diagnostics"]["calibration_gain_interaction"][
            "participant_values"
        ],
        expected,
    )


def test_boundary_allowlist_and_no_querylabel_metadata_interfaces(plan, monkeypatch):
    assert (
        study.spatial.file_sha(ROOT / "configs/analysis/reference_calibration_source39_v1.json")
        == study.PLAN_SHA256
    )
    assert tuple(plan["source_subject_ids"]) == study.SOURCE_IDS and len(study.SOURCE_IDS) == 39
    for subject in (1, 2, 3, 5, 103, True, "4"):
        with pytest.raises(ValueError):
            study.allowed_raw_path(subject, plan)
    with monkeypatch.context() as mock:
        mock.setattr(Path, "is_symlink", lambda _: False)
        mock.setattr(Path, "resolve", lambda self: self)
        assert study.allowed_raw_path(4, plan) == study.RAW_ROOT / "S004.mat"
    changed = copy.deepcopy(plan)
    changed["source_subject_ids"] = [4]
    with pytest.raises(ValueError):
        study.allowed_raw_path(4, changed)
    monkeypatch.setattr(study, "_WORKER", None)
    with pytest.raises(RuntimeError):
        study.load_participant(4)
    for function in (study.prepare_query, study.fit_ecca_templates, study.ecca_query_features):
        assert not any(
            "label" in name or "metadata" in name for name in inspect.signature(function).parameters
        )
    source = inspect.getsource(study)
    for forbidden in ("np.random", "read_parquet", "manifest_path", "rglob("):
        assert forbidden not in source


def test_observed_attainment_history_and_no_automatic_pipeline_promotion(plan):
    rows = []
    for pipeline in study.PIPELINES:
        methods = dict(plan["decoder"]["method_budgets"])
        if pipeline == "no_notch":
            methods["legacy_A0"] = plan["decoder"]["legacy_budgets"]
        for subject in (201, 202, 203):
            for cell in study.spatial.cells(plan):
                for method, budgets in methods.items():
                    for k in budgets:
                        accuracy = 0.6
                        if method == "ECCA" and pipeline == "no_notch":
                            accuracy = {0: 0.6, 1: 0.8, 3: 1.0, 5: 1.0}[k]
                        if method == "ITCCA":
                            accuracy = {1: 0.4, 3: 0.6, 5: 0.8}[k]
                        if method == "legacy_A0":
                            accuracy = 0.8
                        rows.append(
                            {
                                "subject_id": subject,
                                "pipeline": pipeline,
                                **{
                                    key: value
                                    for key, value in cell.items()
                                    if key != "interface_index"
                                },
                                "method": method,
                                "budget": k,
                                "balanced_accuracy": accuracy,
                                "correct_log_probability": -1.0,
                                "true_class_margin": 0.0,
                                "prediction_flips_vs_A0": 0.0,
                                "correct_count": 60 * accuracy,
                                "query_count": 60,
                                "class_correct_counts": [5 * accuracy] * 12,
                                "class_query_counts": [5] * 12,
                            }
                        )
    summary = study.summarize_results(rows, plan)
    assert summary["status"] == "DIAGNOSTIC_COMPLETE" and not summary["human_held_unlock"]
    nonotch = summary["pipeline_summaries"]["no_notch"]
    assert nonotch["status"] == "USEFUL_EEG_CALIBRATION_DEVELOPMENT"
    assert summary["pipeline_summaries"]["causal_notch50"]["status"] == "AQ_NOT_ESTABLISHED"
    attainment = {
        r["method"]: r for r in nonotch["calibration_attainment"] if r["cell_id"] == "dry-n125"
    }
    assert attainment["ECCA"]["first_observed_budget"] == [1] * 3
    assert attainment["ECCA"]["labeled_trials"] == [12] * 3
    assert attainment["ECCA"]["EEG_seconds"] == [6.0] * 3
    assert attainment["ECCA"]["history_seconds"] == [7.68] * 3
    assert attainment["ITCCA"]["first_observed_budget"] == [5] * 3
    assert attainment["ITCCA"]["history_seconds"] == [38.4] * 3
    assert attainment["legacy_A0"]["first_observed_budget"] == [0] * 3
    assert attainment["legacy_A0"]["history_seconds"] == [0.0] * 3
    assert attainment["A0_reference"]["first_observed_budget"] == [">5"] * 3
    assert attainment["A0_reference"]["history_seconds"] == [None] * 3


def test_lifecycle_all_freezes_before_eval_and_four_immutable_artifacts(
    plan, tmp_path, monkeypatch
):
    features = artificial_features(plan)
    active = copy.deepcopy(plan)
    active["reporting"]["expected_rows"] = 6 * 6 * 34
    start = {"source_commit": "fixture-commit", "plan_sha256": "fixture-plan"}
    start_sha = study.spatial.write_json(tmp_path / "start.json", start, 100000)
    events = []

    def collect(*args):
        assert (tmp_path / "start.json").exists()
        events.append("collect")
        return features

    original_fit, original_evaluate = study.fit_fold, study.evaluate_fold

    def fit(*args, **kwargs):
        events.append("fit")
        return original_fit(*args, **kwargs)

    def evaluate(*args, **kwargs):
        assert events.count("fit") == 3
        freeze = json.loads((tmp_path / "fold-freezes.json").read_text())
        assert len(freeze["folds"]) == 3
        events.append("evaluate")
        return original_evaluate(*args, **kwargs)

    monkeypatch.setattr(study, "collect_features", collect)
    monkeypatch.setattr(study, "fit_fold", fit)
    monkeypatch.setattr(study, "evaluate_fold", evaluate)
    result = study.execute_phases(active, {}, tmp_path, start, start_sha, 1)
    assert result["status"] == "DIAGNOSTIC_COMPLETE"
    assert sorted(p.name for p in tmp_path.iterdir()) == sorted(plan["execution"]["artifacts"])
    for path in tmp_path.iterdir():
        assert stat.S_IMODE(path.stat().st_mode) == 0o400
    saved = json.loads((tmp_path / "result.json").read_text())
    assert saved["start_sha256"] == start_sha
    assert saved["features_sha256"] == study.spatial.file_sha(tmp_path / "features.npz")
    assert saved["fold_freezes_sha256"] == study.spatial.file_sha(tmp_path / "fold-freezes.json")
    with pytest.raises(FileExistsError):
        study.spatial.write_scores(tmp_path / "features.npz", features, 1 << 30)
    with pytest.raises(RuntimeError):
        study.spatial.write_json(tmp_path / "overquota.json", {"x": 1}, 1)


def test_start_before_input_and_consumed_attempt_guard(plan, tmp_path, monkeypatch):
    active = copy.deepcopy(plan)
    active["execution"]["output_root"] = str(tmp_path)
    monkeypatch.setattr(study.json, "loads", lambda *args, **kwargs: active)
    monkeypatch.setattr(
        study.subprocess,
        "check_output",
        lambda args, **kwargs: "" if "status" in args else "a" * 40,
    )
    for key in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS"):
        monkeypatch.setenv(key, "1")

    def execute(*args):
        assert (tmp_path / "start.json").exists()
        return {"fixture": True}

    monkeypatch.setattr(study, "execute_phases", execute)
    path = ROOT / "configs/analysis/reference_calibration_source39_v1.json"
    assert study.run_study(path, tmp_path, 4) == {"fixture": True}
    with pytest.raises(FileExistsError, match="consumes"):
        study.run_study(path, tmp_path, 4)


@pytest.mark.parametrize("failure", ["dirty", "pin", "pipeline", "workers"])
def test_preflight_failures_never_open_inputs_or_start(plan, tmp_path, monkeypatch, failure):
    active = copy.deepcopy(plan)
    active["execution"]["output_root"] = str(tmp_path)
    monkeypatch.setattr(study.json, "loads", lambda *args, **kwargs: active)
    monkeypatch.setattr(
        study.subprocess,
        "check_output",
        lambda *args, **kwargs: " M fixture.py" if failure == "dirty" else "",
    )
    monkeypatch.setattr(
        study, "execute_phases", lambda *args: pytest.fail("No input access permitted")
    )
    if failure == "pin":
        active["pinned_files"] = {"src/cfeg/baselines/fbcca.py": "0" * 64}
    if failure == "pipeline":
        active["pipeline_ids"].reverse()
    with pytest.raises((ValueError, RuntimeError)):
        study.run_study(
            ROOT / "configs/analysis/reference_calibration_source39_v1.json",
            tmp_path,
            1 if failure == "workers" else 4,
        )
    assert not (tmp_path / "start.json").exists()
