from __future__ import annotations

import copy
import inspect
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from cfeg.analysis import context_template_source as study
from cfeg.baselines.calibration import predict_filterbank_ensemble_trca
from cfeg.baselines.fbcca import predict_fbcca

FIXTURE_SEED = 131519  # Synthetic unit arrays only; no human files or metadata are read.
FILTERBANK = {
    "bands": [[6, 45], [14, 45]],
    "weights": [1.0, 0.5],
    "order": 2,
    "n_harmonics": 3,
    "regularization": 1e-8,
    "filter_family": "butterworth",
}


@pytest.fixture(autouse=True)
def prohibit_human_inputs(monkeypatch):
    def forbidden(*_args, **_kwargs):
        pytest.fail("Unit tests must never open human EEG or acquisition metadata")

    original = study.file_sha

    def checked_sha(path):
        if str(path).startswith("/home/whwovy/eeg-data/"):
            forbidden()
        return original(path)

    monkeypatch.setattr(study, "_load_arrays", forbidden)
    monkeypatch.setattr(study.pd, "read_parquet", forbidden)
    monkeypatch.setattr(study, "file_sha", checked_sha)


@pytest.fixture(scope="module")
def plan():
    root = Path(__file__).resolve().parents[1]
    value = json.loads((root / "configs/analysis/context_template_source39_v1.json").read_text())
    value.update(
        source_subject_ids=list(study.SOURCE_IDS[:6]),
        raw_shape=[8, 80, 2, 10, 3],
        start_sample=10,
        sample_counts=[32, 48],
        n_classes=3,
        frequencies=[9.25, 11.25, 13.25],
        phases_pi=[0, 0, 0],
        expected_metadata_packets=120,
    )
    value["reporting"]["cost_window_samples"] = 48
    return value


def fixture_raw(plan, seed=FIXTURE_SEED):
    rng = np.random.Generator(np.random.PCG64DXSM(seed))
    raw = rng.normal(size=plan["raw_shape"])
    time = np.arange(plan["raw_shape"][1]) / plan["sfreq"]
    raw += (
        2
        * np.sin(2 * np.pi * time[:, None] * np.asarray(plan["frequencies"])[None])[
            None, :, None, None, :
        ]
    )
    return raw


@pytest.fixture(scope="module")
def features(plan):
    workspaces = {n: study.build_workspace(plan, FILTERBANK, n) for n in plan["sample_counts"]}
    records = [
        study.extract_features(fixture_raw(plan, FIXTURE_SEED + i), plan, workspaces)
        for i in range(6)
    ]
    rng = np.random.Generator(np.random.PCG64DXSM(FIXTURE_SEED + 100))
    cs = study.cells(plan)
    return {
        **{
            key: np.stack([r[key] for r in records]) for key in (*study.FEATURE_KEYS, "crop_sha256")
        },
        "schema": np.asarray(study.FEATURE_SCHEMA),
        "subject_ids": np.asarray(plan["source_subject_ids"]),
        "cell_ids": np.asarray([c["cell_id"] for c in cs]),
        "sample_counts": np.asarray([c["n_samples"] for c in cs]),
        "interface_indices": np.asarray([c["interface_index"] for c in cs]),
        "query_labels": np.tile(np.arange(3), 5),
        "query_block_ids": np.repeat(np.arange(5, 10), 3),
        "support_block_ids": np.arange(5),
        "raw_file_sha256": np.asarray(["a" * 64] * 6),
        "filter_weights": np.asarray(FILTERBANK["weights"]),
        "impedance": np.exp(rng.normal(3, 1, (6, 2, 10, 8))),
    }


@pytest.fixture(scope="module")
def freeze(features, plan):
    return study.fit_fold(features, np.flatnonzero(np.arange(6) % 3 != 0), plan, fold_id=0)


def test_gram_equals_direct_weighted_waveform_and_does_not_mutate():
    rng = np.random.Generator(np.random.PCG64DXSM(FIXTURE_SEED))
    query = rng.normal(size=(2, 6, 8, 48))
    support = rng.normal(size=(2, 5, 3, 8, 48))
    original_query, original_support = query.copy(), support.copy()
    gram = study.gram_statistics(query, support)
    weights = rng.uniform(0.1, 1, (6, 5))
    weights /= weights.sum(axis=-1, keepdims=True)
    actual = study.gram_template_scores(
        gram["gram_cross"], gram["gram_support"], gram["query_norm2"], weights, np.array([1.0, 0.5])
    )
    expected = np.zeros((6, 3))
    for f, fw in enumerate((1.0, 0.5)):
        for q in range(6):
            for c in range(3):
                x = query[f, q].ravel()
                x = x - x.mean()
                t = np.einsum("b,bht->ht", weights[q], support[f, :, c]).ravel()
                t = t - t.mean()
                corr = np.dot(x, t) / (np.linalg.norm(x) * np.linalg.norm(t))
                expected[q, c] += fw * np.sign(corr) * corr**2 / 1.5
    assert np.allclose(actual, expected, atol=1e-14, rtol=0)
    assert np.array_equal(query, original_query)
    assert np.array_equal(support, original_support)
    zeros = study.gram_statistics(np.zeros_like(query), np.zeros_like(support))
    assert not study.gram_template_scores(
        zeros["gram_cross"],
        zeros["gram_support"],
        zeros["query_norm2"],
        weights,
        np.array([1.0, 0.5]),
    ).any()


def test_covariance_is_current_query_only_and_rank_safe():
    rng = np.random.Generator(np.random.PCG64DXSM(FIXTURE_SEED))
    x = rng.normal(size=(3, 8, 32))
    x[:, 7] = x[:, 0]
    full, diag = study.covariance_descriptors(x)
    solo = study.covariance_descriptors(x[:1])
    assert np.array_equal(full[:1], solo[0]) and np.array_equal(diag[:1], solo[1])
    assert np.allclose(full, full.transpose(0, 2, 1), atol=1e-12)
    assert np.isfinite(study.covariance_descriptors(np.zeros_like(x))[0]).all()
    changed = x.copy()
    changed[1:] *= 100
    assert np.array_equal(full[:1], study.covariance_descriptors(changed)[0][:1])
    with pytest.raises(ValueError):
        study.covariance_descriptors(np.full_like(x, np.nan))


def test_raw_crop_boundary_and_public_a0_etrca_parity(plan):
    raw = fixture_raw(plan)
    workspace = {n: study.build_workspace(plan, FILTERBANK, n) for n in plan["sample_counts"]}
    original = study.extract_features(raw, plan, workspace)
    changed = raw.copy()
    changed[:, 58:] += 100
    repeated = study.extract_features(changed, plan, workspace)
    assert all(np.array_equal(original[k], repeated[k]) for k in original)
    crop = raw[:, 10:42, 0].transpose(2, 3, 0, 1)
    query = crop[5:].reshape(15, 8, 32)
    a0 = np.stack([predict_fbcca(x, plan["frequencies"], 250, FILTERBANK)[1] for x in query]) / 1.5
    assert np.allclose(a0, original["a0_scores"][0], atol=1e-10, rtol=0)
    for index, k in enumerate((3, 5)):
        exact = predict_filterbank_ensemble_trca(
            query,
            crop[:k].reshape(-1, 8, 32),
            np.tile(np.arange(3), k),
            n_classes=3,
            sfreq=250,
            filterbank=FILTERBANK,
            ridge_ratio=1e-6,
        ).scores
        assert np.array_equal(exact, original["etrca_scores"][0, index])


def test_query_perturbation_has_no_other_query_effect(plan):
    crop = fixture_raw(plan)[:, 10:42, 0].transpose(2, 3, 0, 1)
    workspace = study.build_workspace(plan, FILTERBANK, 32)
    first = study.extract_cell(crop, workspace)
    other = crop.copy()
    other[6:] *= -4
    second = study.extract_cell(other, workspace)
    assert np.array_equal(first["a0_scores"][:3], second["a0_scores"][:3])
    assert np.array_equal(first["gram_cross"][:, :3], second["gram_cross"][:, :3])
    assert np.array_equal(first["q_full_query"][:3], second["q_full_query"][:3])
    assert np.array_equal(first["q_diag_query"][:3], second["q_diag_query"][:3])
    assert np.array_equal(first["gram_support"], second["gram_support"])


def test_q_only_apis_have_no_metadata_or_query_labels(features, plan):
    for function in (
        study.covariance_descriptors,
        study.extract_cell,
        study.gram_template_scores,
        study.q_weights,
    ):
        assert not set(inspect.signature(function).parameters) & {
            "query_labels",
            "labels",
            "metadata",
            "impedance",
            "subject_id",
        }
    one = study.one_cell(features, 0, 0)
    for candidate in plan["q"]["candidates"]:
        w = study.q_weights(one, candidate, 3, plan)
        assert w.shape == (15, 3) and np.all(w >= 0) and np.allclose(w.sum(axis=1), 1)
        assert np.array_equal(study.q_weights(one, candidate, 1, plan), np.ones((15, 1)))
    modified = dict(one, impedance=np.full((10, 8), 1e100))
    assert np.array_equal(
        study.q_weights(one, plan["q"]["candidates"][-1], 3, plan),
        study.q_weights(modified, plan["q"]["candidates"][-1], 3, plan),
    )


def test_metadata_scale_unique_packets_validity_and_missing():
    values = np.ones((4, 2, 10, 8))
    values[:, 0, :, 0] = np.arange(40).reshape(4, 10)
    values[:, 1, :, 3] = np.nan
    scale, available = study.fit_metadata_scale(values)
    assert scale[0, 0] == max(np.diff(np.percentile(np.log1p(np.arange(40)), [25, 75]))[0], 0.1)
    assert scale[1, 3] == 0.1 and not available[1, 3]
    assert np.all(scale >= 0.1)
    for bad in (-1, np.inf, -np.inf):
        with pytest.raises(ValueError, match="Impedance"):
            study.validate_impedance(np.array([bad]))
    assert study.validate_impedance(np.array([0, np.nan]))[0] == 0


def test_mixed_missing_rows_beta_zero_and_single_block_are_exact():
    q = np.array([[0.2, 0.3, 0.5000000000000001], [0.6, 0.3, 0.1]])
    support = np.array([[2.0, 4.0], [50.0, 80.0], [4.0, 6.0]])
    query = np.array([[np.nan, np.nan], [2.0, 4.0]])
    actual = study.metadata_weights(q, support, query, np.ones(2), np.ones(2, dtype=bool), 1.0)
    assert np.array_equal(actual[0], q[0])
    assert not np.array_equal(actual[1], q[1])
    assert study.metadata_weights(q, support, query, np.ones(2), np.ones(2, dtype=bool), 0.0) is q
    assert study.metadata_weights(q, support, query, np.ones(2), np.zeros(2, dtype=bool), 1.0) is q
    assert (
        study.metadata_weights(
            q, support, np.full_like(query, np.nan), np.ones(2), np.ones(2, dtype=bool), 1.0
        )
        is q
    )
    one = np.ones((2, 1))
    assert (
        study.metadata_weights(one, support[:1], query, np.ones(2), np.ones(2, dtype=bool), 1.0)
        is one
    )
    ordered = study.order_weights(q, np.array([5, 9]), np.arange(5), 16)
    assert not np.array_equal(ordered, q)


def fixture_frame(plan):
    rows = []
    slots = np.asarray(plan["canonical_channel_ids"]) - 1
    for s in plan["source_subject_ids"]:
        for interface in plan["interfaces"]:
            for b in range(10):
                z = np.full(64, np.nan)
                z[slots] = np.arange(8) + s + b
                record = {
                    "subject_id": f"sub{s:03d}",
                    "electrode_type": interface,
                    "run_id": f"block{b + 1:02d}",
                    "impedance_kohm_by_channel": z,
                    "headband_order": "dry",
                    "condition_period": "first" if interface == "dry" else "second",
                }
                rows.extend([record.copy() for _ in range(12)])
    return pd.DataFrame(rows)


def test_projection_aligns_native_slots_and_rejects_forbidden_rows(plan):
    frame = fixture_frame(plan)
    projection = study.metadata_projection(frame, plan)
    assert projection["returned_packets"] == 120
    assert projection["packets"][0]["impedance_kohm"] == (np.arange(8) + 4).tolist()
    bad = frame.copy()
    bad.loc[0, "subject_id"] = "sub005"
    with pytest.raises(ValueError, match="non-source"):
        study.metadata_projection(bad, plan)
    with pytest.raises(ValueError, match="columns"):
        study.metadata_projection(frame.assign(query_signal_std=1), plan)
    bad = frame.copy()
    z = bad.iloc[0]["impedance_kohm_by_channel"].copy()
    z[55] += 1
    bad.at[0, "impedance_kohm_by_channel"] = z
    with pytest.raises(ValueError, match="disagree"):
        study.metadata_projection(bad, plan)
    bad = frame.copy()
    bad.loc[:11, "headband_order"] = "wet"
    with pytest.raises(ValueError, match="constant"):
        study.metadata_projection(bad, plan)
    bad = frame.copy()
    bad.loc[:11, "condition_period"] = "second"
    with pytest.raises(ValueError, match="contradicts"):
        study.metadata_projection(bad, plan)


def test_production_projection_requests_only_authorized_rows_columns(monkeypatch, plan):
    full_plan = copy.deepcopy(plan)
    full_plan.update(source_subject_ids=list(study.SOURCE_IDS), expected_metadata_packets=780)
    frame = fixture_frame(full_plan)
    observed = {}

    def read(path, *, columns, filters):
        observed.update(path=path, columns=columns, filters=filters)
        return frame

    monkeypatch.setattr(study.pd, "read_parquet", read)
    monkeypatch.setattr(study, "file_sha", lambda _p: full_plan["manifest_sha256"])
    projected = study.project_source_metadata(full_plan)
    assert projected["returned_subject_ids"] == list(study.SOURCE_IDS)
    assert observed["columns"] == list(study.MANIFEST_COLUMNS)
    assert observed["filters"] == [("subject_id", "in", [f"sub{s:03d}" for s in study.SOURCE_IDS])]


def test_source_raw_allowlist_rejects_held_and_retired_before_load(plan):
    full_plan = dict(plan, source_subject_ids=list(study.SOURCE_IDS))
    for subject in (1, 2, 3, 5, 7, 103, "004"):
        with pytest.raises(ValueError, match="allowlist"):
            study.allowed_raw_path(subject, full_plan)
    assert study.allowed_raw_path(4, full_plan) == study.RAW_ROOT / "S004.mat"
    with pytest.raises(ValueError, match="root"):
        study.allowed_raw_path(4, dict(full_plan, raw_root="/tmp/forbidden"))


def test_fold_source_only_fit_aliases_and_actual_fused_nll(features, freeze, plan):
    assert set(freeze["fit_subject_ids"]).isdisjoint(freeze["evaluation_subject_ids"])
    assert len(freeze["fit_subject_ids"]) == 4
    for candidate in freeze["support_temperatures"]:
        assert (
            freeze["support_temperatures"][candidate]["1"]
            == freeze["support_temperatures"]["uniform"]["1"]
        )
    chosen_indices = np.flatnonzero(np.arange(6) % 3 != 0)
    fit = study.subset_features(features, chosen_indices)
    scores = study.score_weights(fit, study.q_weights(fit, plan["q"]["candidates"][0], 3, plan))
    t = freeze["temperature_grid"][12]
    p0 = study.numeric.softmax(fit["a0_scores"], freeze["a0_temperature"]["temperature"])
    expected = study.nll(
        0.5 * p0 + 0.5 * study.numeric.softmax(scores, t), features["query_labels"]
    )
    assert expected == pytest.approx(
        freeze["support_temperatures"]["uniform"]["3"]["objective_values"][12]
    )
    changed = copy.deepcopy(features)
    for key in study.FEATURE_KEYS:
        changed[key][np.arange(6) % 3 == 0] *= 7
    changed["impedance"][np.arange(6) % 3 == 0] *= 100
    assert study.fit_fold(changed, chosen_indices, plan, fold_id=0) == freeze
    with pytest.raises(ValueError, match="complement"):
        study.fit_fold(features, np.arange(6), plan, fold_id=0)


def test_q_selection_cannot_depend_on_metadata(features, freeze, plan):
    changed = copy.deepcopy(features)
    changed["impedance"] *= 1000
    other = study.fit_fold(changed, np.flatnonzero(np.arange(6) % 3 != 0), plan, fold_id=0)
    for key in (
        "a0_temperature",
        "support_temperatures",
        "q_selection",
        "diag_selection",
        "full_selection",
    ):
        assert other[key] == freeze[key]


def test_temperature_ties_take_first_grid_index():
    scores = np.zeros((2, 3, 4))
    labels = np.array([0, 1, 2])
    result = study.temperature_fit(scores, labels, np.logspace(-4, 1, 31))
    assert result["selected_index"] == 0


def test_all_methods_exact_nulls_and_permutation_metric_controls(features, freeze, plan):
    rows = study.evaluate_fold(features, np.array([0, 3]), freeze, plan)
    assert len(rows) == 2 * 4 * 42
    lookup = {(r["subject_id"], r["cell_id"], r["method"], r["budget"]): r for r in rows}
    for subject in freeze["evaluation_subject_ids"]:
        for cell in study.cells(plan):
            cid = cell["cell_id"]
            for k in (0, 1, 3, 5):
                aq = lookup[subject, cid, "AQ", k]
                assert (
                    lookup[subject, cid, "M_missing", k]["posterior_sha256"]
                    == aq["posterior_sha256"]
                )
                if k <= 1:
                    for method in ("AQM", "M_stale", "order_control"):
                        assert (
                            lookup[subject, cid, method, k]["posterior_sha256"]
                            == aq["posterior_sha256"]
                        )
                assert (
                    lookup[subject, cid, "M_shuffle", k]["permutations"]
                    == {0: 0, 1: 0, 3: 2, 5: 44}[k]
                )
                assert lookup[subject, cid, "M_shuffle", k]["posterior_sha256"] is None
    zero = copy.deepcopy(freeze)
    zero["m_selection"]["beta"] = 0
    zero_rows = study.evaluate_fold(features, np.array([0, 3]), zero, plan)
    grouped = {
        (r["subject_id"], r["cell_id"], r["budget"]): r for r in zero_rows if r["method"] == "AQ"
    }
    for row in zero_rows:
        if row["method"] in ("AQM", "M_stale"):
            assert (
                row["posterior_sha256"]
                == grouped[row["subject_id"], row["cell_id"], row["budget"]]["posterior_sha256"]
            )


def test_summary_participant_aggregation_and_sparse_etrca(features, freeze, plan):
    rows = study.evaluate_fold(features, np.array([0, 3]), freeze, plan)
    summary = study.summarize_results(rows, plan)
    assert summary["comparisons"]["M_eauc_gain"]["n"] == 2
    assert len(summary["aggregate_rows"]) == 4 * 42
    assert len(summary["participant_harm"]) == 2 * 42
    assert len(summary["calibration_attainment"]) == 4 * 11
    for item in summary["calibration_attainment"]:
        if item["method"] == "FB_eTRCA":
            assert set(item["participant_accuracy_curve"]) == {"3", "5"}
        if item["method"] == "M_shuffle":
            assert item["interpretation"] == "expected_permutation_metric_not_deployable_predictor"
    assert summary["human_held_unlock"] is False
    assert len(summary["screen_components"]) == 9


def test_exclusive_artifacts_quota_and_pickle_free_readback(features, tmp_path):
    path = tmp_path / "fixture.npz"
    digest = study.write_features(path, features, 10**8)
    assert digest == study.file_sha(path)
    assert path.stat().st_mode & 0o777 == 0o400
    with np.load(path, allow_pickle=False) as restored:
        assert set(restored.files) == set(features)
    with pytest.raises(FileExistsError):
        study.write_features(path, features, 10**8)
    with pytest.raises(RuntimeError, match="quota"):
        study.write_json(tmp_path / "too-large.json", {"a": "b"}, 1)
    assert not (tmp_path / "too-large.json").exists()


def test_all_fit_freezes_are_published_before_any_outer_eval(features, plan, monkeypatch, tmp_path):
    events = []
    real_evaluate = study.evaluate_fold

    def projection(_plan):
        assert (tmp_path / "start.json").exists()
        events.append("projection")
        return study.metadata_projection(fixture_frame(plan), plan)

    def collect(*_args):
        assert (tmp_path / "source-projection.json").exists()
        events.append("features")
        return features

    def evaluate(cache, indices, freeze, _plan):
        frozen = json.loads((tmp_path / "fold-freezes.json").read_text())
        assert len(frozen["folds"]) == 3
        events.append(f"eval{freeze['fold_id']}")
        return real_evaluate(cache, indices, freeze, _plan)

    monkeypatch.setattr(study, "project_source_metadata", projection)
    monkeypatch.setattr(study, "collect_features", collect)
    monkeypatch.setattr(study, "evaluate_fold", evaluate)
    start = {"source_commit": "fixture-only", "plan_sha256": "fixture-plan"}
    digest = study.write_json(tmp_path / "start.json", start, 10**8)
    result = study.execute_phases(plan, FILTERBANK, tmp_path, start, digest, 1)
    assert result["rows"] == 6 * 4 * 42
    assert events == ["projection", "features", "eval0", "eval1", "eval2"]
    assert {p.name for p in tmp_path.iterdir()} == set(plan["execution"]["artifacts"])
    assert all(p.stat().st_mode & 0o777 == 0o400 for p in tmp_path.iterdir())


def test_existing_start_is_refused_before_human_access(monkeypatch):
    root = Path(__file__).resolve().parents[1]
    plan_path = root / "configs/analysis/context_template_source39_v1.json"
    actual = json.loads(plan_path.read_text())
    original = Path.exists
    monkeypatch.setattr(
        Path,
        "exists",
        lambda p: (
            True if p == Path(actual["execution"]["output_root"]) / "start.json" else original(p)
        ),
    )
    with pytest.raises(FileExistsError, match="consumes"):
        study.run_study(plan_path, Path(actual["execution"]["output_root"]), 4)
