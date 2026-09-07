"""Unrelated deterministic engineering fixtures; never open human study inputs."""

from __future__ import annotations

import copy
import inspect
import json
import stat
from pathlib import Path

import numpy as np
import pytest
import yaml
from scipy.linalg import eigh

from cfeg.analysis import spatial_calibration_source as study
from cfeg.baselines.calibration import predict_filterbank_ensemble_trca
from cfeg.baselines.fbcca import cca_score, predict_fbcca

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def plan():
    return json.loads((ROOT / "configs/analysis/spatial_calibration_source39_v1.json").read_text())


@pytest.fixture
def bank(plan):
    return yaml.safe_load((ROOT / plan["filterbank_config"]).read_text())


def artificial_scores(plan, subjects=6):
    rng = np.random.default_rng(718204)  # Unit-fixture namespace only.
    classes, queries, count = plan["n_classes"], 2 * plan["n_classes"], len(study.cells(plan))
    shape = (subjects, count)
    labels = np.tile(np.arange(classes), 2)
    a0 = rng.uniform(0, 0.4, (*shape, queries, classes))
    itcca = rng.uniform(0, 0.3, (*shape, 3, queries, classes))
    for k in range(3):
        itcca[:, :, k, np.arange(queries), labels] += 0.02 * (k + 1)
    return {
        "subject_ids": np.arange(201, 201 + subjects),
        "sample_counts": np.asarray([c["n_samples"] for c in study.cells(plan)]),
        "query_labels": labels,
        "a0_scores": a0,
        "itcca_scores": itcca,
        "etrca_unit_scores": rng.uniform(-0.2, 0.5, (*shape, 2, queries, classes)),
        "legacy_etrca_scores": rng.uniform(-0.3, 2, (*shape, 2, queries, classes)),
    }


def test_frozen_plan_and_input_allowlist(plan):
    assert (
        study.file_sha(ROOT / "configs/analysis/spatial_calibration_source39_v1.json")
        == study.PLAN_SHA256
    )
    assert tuple(plan["source_subject_ids"]) == study.SOURCE_IDS
    assert len(study.SOURCE_IDS) == 39
    for subject in (1, 2, 3, 5, 103, True, "4"):
        with pytest.raises(ValueError):
            study.allowed_raw_path(subject, plan)
    changed = copy.deepcopy(plan)
    changed["source_subject_ids"] = [4]
    with pytest.raises(ValueError):
        study.allowed_raw_path(4, changed)
    changed = copy.deepcopy(plan)
    changed["raw_root"] = "/tmp/not-study-data"
    with pytest.raises(ValueError):
        study.allowed_raw_path(4, changed)
    assert study.allowed_raw_path(4, plan) == study.RAW_ROOT / "S004.mat"


def test_pairwise_cca_matches_public_rank_deficient_and_constant():
    rng = np.random.default_rng(718205)
    query, templates = rng.normal(size=(4, 8, 125)), rng.normal(size=(3, 8, 125))
    query[1, 1] = query[1, 0]
    query[2] = 12
    templates[2] = 0
    for scale in (1, 1e-7, 1000):
        actual = study.pairwise_regularized_cca(query * scale, templates * scale, 0.01)
        expected = np.array(
            [[cca_score(x * scale, y * scale, 0.01) for y in templates] for x in query]
        )
        np.testing.assert_allclose(actual, expected, atol=2e-12, rtol=2e-12)
    assert np.array_equal(actual[2], np.zeros(3))


def test_cca_query_locality_offsets_and_shape_validation():
    rng = np.random.default_rng(718206)
    query, templates = rng.normal(size=(4, 5, 80)), rng.normal(size=(3, 5, 80))
    full = study.pairwise_regularized_cca(query, templates)
    for i in range(len(query)):
        np.testing.assert_allclose(
            study.pairwise_regularized_cca(query[i : i + 1], templates)[0], full[i], atol=1e-14
        )
    np.testing.assert_allclose(
        full,
        study.pairwise_regularized_cca(query + np.arange(5)[None, :, None], templates - 9),
        atol=1e-13,
    )
    for invalid in (query[0], query[..., :1], query * np.nan):
        with pytest.raises(ValueError):
            study.pairwise_regularized_cca(invalid, templates)
    with pytest.raises(ValueError):
        study.pairwise_regularized_cca(query, templates[..., :-1])
    with pytest.raises(ValueError):
        study.pairwise_regularized_cca(query, templates, -1)


def test_cyclic_support_labels_exact_score_column_permutation():
    rng = np.random.default_rng(718207)
    support, query = rng.normal(size=(5, 12, 4, 70)), rng.normal(size=(7, 4, 70))
    for k in study.BUDGETS:
        scores = study.pairwise_regularized_cca(query, support[:k].mean(axis=0))
        for shift in range(1, 12):
            relabeled = np.roll(support[:k], shift, axis=1)
            expected = study.pairwise_regularized_cca(query, relabeled.mean(axis=0))
            np.testing.assert_array_equal(np.roll(scores, shift, axis=-1), expected)


def test_unit_trca_manual_operator_sign_dc_and_scale_invariance():
    rng = np.random.default_rng(718208)
    support, query = rng.normal(size=(3, 4, 5, 90)), rng.normal(size=(6, 5, 90))
    templates, filters = study.unit_trca_filters(support)
    np.testing.assert_allclose(np.linalg.norm(filters, axis=0), 1, atol=1e-14)
    centered = support - support.mean(axis=-1, keepdims=True)
    for c in range(4):
        x = centered[:, c]
        q = sum(v @ v.T for v in x)
        s = sum(x[b] @ x[d].T for b in range(3) for d in range(3) if b != d)
        ridge = 0.001 * max(np.trace(q) / len(q), 1e-12)
        eigenvalues, eigenvectors = eigh(s, q + ridge * np.eye(5))
        oracle = eigenvectors[:, np.argmax(eigenvalues)]
        oracle /= np.linalg.norm(oracle)
        assert abs(float(oracle @ filters[:, c])) == pytest.approx(1, abs=1e-12)
    scores = study.unit_trca_scores(query, templates, filters)
    np.testing.assert_allclose(
        scores, study.unit_trca_scores(query, templates, filters * [1, -1, -1, 1]), atol=1e-14
    )
    shifted = study.unit_trca_filters(support + np.arange(5)[None, None, :, None] * 10)
    np.testing.assert_allclose(scores, study.unit_trca_scores(query - 100, *shifted), atol=1e-12)
    np.testing.assert_allclose(
        scores,
        study.unit_trca_scores(query * 0.001, *study.unit_trca_filters(support * 0.001)),
        atol=1e-12,
    )
    zeros = np.zeros_like(support)
    np.testing.assert_array_equal(
        study.unit_trca_scores(np.zeros_like(query), *study.unit_trca_filters(zeros)),
        np.zeros((6, 4)),
    )
    with pytest.raises(ValueError):
        study.unit_trca_filters(support[:1])


def test_full_bank_a0_legacy_parity_and_prefix_locality(plan, bank):
    rng = np.random.default_rng(718209)
    small = copy.deepcopy(plan)
    small["n_classes"] = 3
    small["frequencies"] = plan["frequencies"][:3]
    crop = rng.normal(size=(10, 3, 8, 125))
    workspace = study.build_workspace(small, bank, 125)
    result = study.extract_cell(crop, workspace)
    query = crop[5:].reshape(-1, 8, 125)
    expected = np.stack(
        [predict_fbcca(trial, small["frequencies"], 250, filterbank=bank)[1] for trial in query]
    ) / np.sum(bank["weights"])
    np.testing.assert_allclose(result["a0_scores"], expected, atol=1e-12)
    for j, k in enumerate(study.TRCA_BUDGETS):
        expected = predict_filterbank_ensemble_trca(
            query,
            crop[:k].reshape(-1, 8, 125),
            np.tile(np.arange(3), k),
            n_classes=3,
            sfreq=250,
            filterbank=bank,
        ).scores
        np.testing.assert_array_equal(result["legacy_etrca_scores"][j], expected)
    changed = crop.copy()
    changed[3:5] *= 3
    after = study.extract_cell(changed, workspace)
    np.testing.assert_array_equal(result["a0_scores"], after["a0_scores"])
    np.testing.assert_array_equal(result["itcca_scores"][:2], after["itcca_scores"][:2])
    np.testing.assert_array_equal(result["etrca_unit_scores"][0], after["etrca_unit_scores"][0])
    changed = crop.copy()
    changed[5, 0] *= 5
    after = study.extract_cell(changed, workspace)
    for key in study.SCORE_KEYS:
        np.testing.assert_array_equal(result[key][..., 1:, :], after[key][..., 1:, :])


def test_native_crop_before_filter_and_cell_order(plan, monkeypatch):
    rng = np.random.default_rng(718210)
    raw = rng.normal(size=tuple(plan["raw_shape"]))
    captured = []

    def capture(crop, workspace):
        captured.append(crop.copy())
        assert crop.flags.c_contiguous and crop.dtype == np.float64
        return {key: np.array([0.0]) for key in study.SCORE_KEYS}

    monkeypatch.setattr(study, "extract_cell", capture)
    result = study.extract_scores(raw, plan, {n: None for n in plan["sample_counts"]})
    assert result["crop_sha256"].shape == (6,)
    for i, cell in enumerate(study.cells(plan)):
        expected = raw[:, 160 : 160 + cell["n_samples"], cell["interface_index"]].transpose(
            2, 3, 0, 1
        )
        np.testing.assert_array_equal(captured[i], expected)
    np.testing.assert_array_equal(captured[0], captured[2][..., :125])
    with pytest.raises(ValueError):
        study.extract_scores(raw[..., :-1], plan, {})


def test_fusion_exact_fallback_and_uniform_decisions():
    rng = np.random.default_rng(718211)
    p0 = study.numeric.softmax(rng.normal(size=(20, 12)), 0.1)
    assert study.fused_probability(p0, np.full_like(p0, np.nan), None, 0) is p0
    for strength in (0.25, 0.5, 0.75):
        uniform = (1 - strength) * p0 + strength / 12
        np.testing.assert_array_equal(p0.argmax(axis=-1), uniform.argmax(axis=-1))
    with pytest.raises(ValueError):
        study.fused_probability(p0, p0, 1, 1)


def test_fit_uses_actual_mixture_nll_and_deterministic_ties(plan, monkeypatch):
    rng = np.random.default_rng(718212)
    labels = np.tile(np.arange(12), 2)
    scores = rng.normal(size=(2, 2, 24, 12))
    p0 = study.numeric.softmax(rng.normal(size=scores.shape), 0.5)
    fit = study.fusion_fit(p0, scores, labels, plan)
    assert len(fit["candidates"]) == 94
    assert fit["candidates"][0]["temperature"] is None
    assert [fit["candidates"][i]["lambda"] for i in (0, 1, 32, 63)] == [0, 0.25, 0.5, 0.75]
    for candidate in fit["candidates"]:
        probability = study.fused_probability(
            p0, scores, candidate["temperature"], candidate["lambda"]
        )
        truth = probability[..., np.arange(24), labels]
        assert candidate["nll"] == pytest.approx(
            float(-np.log(np.maximum(truth, 1e-300)).mean()), abs=1e-14
        )
    monkeypatch.setattr(study, "nll", lambda *args: 2.5)
    assert study.temperature_fit(scores, labels)["selected_index"] == 0
    assert study.fusion_fit(p0, scores, labels, plan)["selected_index"] == 0


def test_all_fits_source_only_window_pooling_and_objective_count(plan):
    cache = artificial_scores(plan)
    fit_indices = np.flatnonzero(np.arange(6) % 3 != 0)
    fit = study.fit_fold(cache, fit_indices, plan, fold_id=0)
    changed = copy.deepcopy(cache)
    for key in study.SCORE_KEYS:
        changed[key][[0, 3]] += 1000
    assert fit == study.fit_fold(changed, fit_indices, plan, fold_id=0)
    changed = copy.deepcopy(cache)
    for key in study.SCORE_KEYS:
        changed[key][:, [0, 1]] = changed[key][:, [1, 0]]
    swapped = study.fit_fold(changed, fit_indices, plan, fold_id=0)
    for n in plan["sample_counts"]:
        np.testing.assert_allclose(
            fit["windows"][str(n)]["a0_temperature"]["objectives"],
            swapped["windows"][str(n)]["a0_temperature"]["objectives"],
            atol=1e-14,
        )
    count = 0
    for window in fit["windows"].values():
        count += len(window["a0_temperature"]["objectives"])
        for budget in window["budgets"].values():
            count += len(budget["fusion_fit"]["candidates"])
            count += sum(len(t["objectives"]) for t in budget["standalone_temperatures"].values())
    assert count * 3 == plan["fitting"]["expected_objective_count"]
    with pytest.raises(ValueError):
        study.fit_fold(cache, np.arange(6), plan, fold_id=0)


def test_all_rows_controls_aggregation_and_available_budgets(plan):
    cache = artificial_scores(plan)
    rows = []
    for fold in range(3):
        freeze = study.fit_fold(cache, np.flatnonzero(np.arange(6) % 3 != fold), plan, fold_id=fold)
        rows.extend(
            study.evaluate_fold(cache, np.flatnonzero(np.arange(6) % 3 == fold), freeze, plan)
        )
    assert len(rows) == 6 * 6 * 23
    lookup = {(r["subject_id"], r["cell_id"], r["method"], r["budget"]): r for r in rows}
    for row in rows:
        a0 = lookup[row["subject_id"], row["cell_id"], "A0", 0]
        if row["budget"] == 0:
            assert row["posterior_sha256"] == a0["posterior_sha256"]
        if row["method"] == "uniform_shrinkage":
            assert row["balanced_accuracy"] == a0["balanced_accuracy"]
        if row["method"] == "wrong_label_support" and row["budget"] > 0:
            assert row["permutations"] == 11 and row["posterior_sha256"] is None
            assert row["balanced_accuracy"] == np.mean(
                [m["balanced_accuracy"] for m in row["permutation_metrics"]]
            )
    summary = study.summarize_results(rows, plan)
    assert len(summary["aggregate_rows"]) == 6 * 23
    assert len(summary["participant_harm"]) == 6 * 23
    assert len(summary["calibration_attainment"]) == 6 * 7
    assert all(c["n"] == 6 for c in summary["comparisons"].values())
    assert not summary["human_held_unlock"]
    for record in summary["calibration_attainment"]:
        if record["method"] == "ITCCA":
            assert list(record["participant_accuracy_curve"]) == ["1", "3", "5"]
        if record["method"] == "wrong_label_support":
            assert (
                record["interpretation"] == "expected_permutation_metric_not_deployable_predictor"
            )


def test_decoding_apis_have_no_query_label_or_metadata_arguments():
    for function in (
        study.extract_cell,
        study.pairwise_regularized_cca,
        study.unit_trca_filters,
        study.unit_trca_scores,
    ):
        arguments = inspect.signature(function).parameters
        assert not any("label" in name or "metadata" in name for name in arguments)
    source = inspect.getsource(study)
    assert "read_parquet" not in source and "manifest_path" not in source
    assert "np.random" not in source and "rglob(" not in source


def test_exclusive_publication_permissions_hash_and_quota(tmp_path):
    payload = {"sample": np.arange(8, dtype=np.float64), "schema": np.asarray("fixture")}
    path = tmp_path / "scores.npz"
    digest = study.write_scores(path, payload, 100000)
    assert digest == study.file_sha(path)
    assert stat.S_IMODE(path.stat().st_mode) == 0o400
    with np.load(path, allow_pickle=False) as loaded:
        np.testing.assert_array_equal(loaded["sample"], payload["sample"])
    with pytest.raises(FileExistsError):
        study.write_scores(path, payload, 100000)
    with pytest.raises(RuntimeError):
        study.write_json(tmp_path / "over-quota.json", {"x": 1}, 1)
    assert not (tmp_path / "over-quota.json").exists()


def test_lifecycle_all_fit_freezes_precede_evaluation(plan, tmp_path, monkeypatch):
    cache = artificial_scores(plan)
    active = copy.deepcopy(plan)
    active["reporting"]["expected_rows"] = 6 * 6 * 23
    events = []
    start = {"source_commit": "fixture-commit", "plan_sha256": "fixture-plan"}
    study.write_json(tmp_path / "start.json", start, 100000)
    monkeypatch.setattr(study, "collect_scores", lambda *args: events.append("collect") or cache)
    original_fit, original_evaluate = study.fit_fold, study.evaluate_fold

    def fit(*args, **kwargs):
        events.append("fit")
        return original_fit(*args, **kwargs)

    def evaluate(*args, **kwargs):
        assert events.count("fit") == 3
        frozen = json.loads((tmp_path / "fold-freezes.json").read_text())
        assert len(frozen["folds"]) == 3
        events.append("evaluate")
        return original_evaluate(*args, **kwargs)

    monkeypatch.setattr(study, "fit_fold", fit)
    monkeypatch.setattr(study, "evaluate_fold", evaluate)
    study.execute_phases(active, {}, tmp_path, start, study.file_sha(tmp_path / "start.json"), 1)
    assert sorted(p.name for p in tmp_path.iterdir()) == sorted(plan["execution"]["artifacts"])
    result = json.loads((tmp_path / "result.json").read_text())
    assert result["scores_sha256"] == study.file_sha(tmp_path / "scores.npz")
    assert result["fold_freezes_sha256"] == study.file_sha(tmp_path / "fold-freezes.json")


def test_run_guard_start_before_inputs_and_existing_start_refusal(plan, tmp_path, monkeypatch):
    active = copy.deepcopy(plan)
    active["execution"]["output_root"] = str(tmp_path)
    monkeypatch.setattr(study.json, "loads", lambda *args, **kwargs: active)
    monkeypatch.setattr(
        study.subprocess,
        "check_output",
        lambda args, **kwargs: "" if "status" in args else "fixture-commit",
    )
    for key in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS"):
        monkeypatch.setenv(key, "1")

    def execute(*args):
        assert (tmp_path / "start.json").exists()
        assert stat.S_IMODE((tmp_path / "start.json").stat().st_mode) == 0o400
        return {"fixture": True}

    monkeypatch.setattr(study, "execute_phases", execute)
    path = ROOT / "configs/analysis/spatial_calibration_source39_v1.json"
    assert study.run_study(path, tmp_path, 4) == {"fixture": True}
    with pytest.raises(FileExistsError, match="consumes"):
        study.run_study(path, tmp_path, 4)


def test_run_rejects_dirty_source_before_inputs(plan, tmp_path, monkeypatch):
    active = copy.deepcopy(plan)
    active["execution"]["output_root"] = str(tmp_path)
    monkeypatch.setattr(study.json, "loads", lambda *args, **kwargs: active)
    monkeypatch.setattr(study.subprocess, "check_output", lambda *args, **kwargs: " M fixture.py")
    monkeypatch.setattr(
        study, "execute_phases", lambda *args: pytest.fail("No input access permitted")
    )
    with pytest.raises(RuntimeError, match="Clean committed"):
        study.run_study(ROOT / "configs/analysis/spatial_calibration_source39_v1.json", tmp_path, 4)
    assert not (tmp_path / "start.json").exists()
