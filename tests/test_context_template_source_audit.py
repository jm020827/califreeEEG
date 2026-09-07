"""Independent arithmetic fixtures, never source EEG or declared human metadata."""

from __future__ import annotations

import copy
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "context_audit_fixture", ROOT / "scripts/audit_context_template_source.py"
)
audit = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(audit)


@pytest.fixture(scope="module")
def cache():
    rng = np.random.default_rng(817234)  # Synthetic sufficient-statistic fixture, not study data.
    queries = rng.normal(size=(7, 60, 16))
    support = rng.normal(size=(7, 12, 5, 16))
    queries -= queries.mean(-1, keepdims=True)
    support -= support.mean(-1, keepdims=True)
    cross = np.einsum("fqt,fcbt->fqcb", queries, support)
    gram = np.einsum("fcbt,fcdt->fcbd", support, support)
    full_support = rng.normal(size=(5, 8, 8))
    full_support = (full_support + full_support.swapaxes(-1, -2)) / 2
    full_query = rng.normal(size=(60, 8, 8))
    full_query = (full_query + full_query.swapaxes(-1, -2)) / 2
    return {
        "gram_cross": cross[None, None],
        "gram_support": gram[None, None],
        "query_norm2": (queries**2).sum(-1)[None, None],
        "filter_weights": np.arange(1, 8, dtype=float),
        "query_labels": np.tile(np.arange(12), 5),
        "query_block_ids": np.repeat(np.arange(5, 10), 12),
        "interface_indices": np.array([0]),
        "reliability": rng.uniform(0, 1, (1, 1, 5)),
        "q_full_support": full_support[None, None],
        "q_full_query": full_query[None, None],
        "q_diag_support": np.diagonal(full_support, axis1=-2, axis2=-1)[None, None],
        "q_diag_query": np.diagonal(full_query, axis1=-2, axis2=-1)[None, None],
        "impedance": rng.uniform(0, 200, (1, 2, 10, 8)),
        "fixture_queries": queries,
        "fixture_support": support,
    }


def test_gram_audit_matches_direct_weighted_vectors(cache):
    weights = audit.qweights(cache, 0, 0, {"kind": "full", "tau": 0.75}, 3)
    actual = audit.template_scores(cache, 0, 0, weights)
    templates = np.einsum("qb,fcbt->fqct", weights, cache["fixture_support"][:, :, :3])
    query = cache["fixture_queries"][:, :, None]
    corr = (query * templates).sum(-1) / np.sqrt((query**2).sum(-1) * (templates**2).sum(-1))
    expected = (
        np.sum(cache["filter_weights"][:, None, None] * corr * np.abs(corr), axis=0)
        / cache["filter_weights"].sum()
    )
    np.testing.assert_allclose(actual, expected, atol=1e-14)


def test_metadata_missing_fit_channel_remains_neutral(cache):
    z = cache["impedance"].copy()
    z[..., 0] = np.nan
    scale, available = audit.metadata_scale(z)
    assert not available[:, 0].any()
    assert np.all(scale[:, 0] == 0.1)
    first = copy.deepcopy(cache)
    first["impedance"] = z.copy()
    second = copy.deepcopy(first)
    second["impedance"][..., 0] = np.arange(10)[None, None] * 999
    qw = audit.qweights(cache, 0, 0, {"kind": "uniform"}, 3)
    np.testing.assert_array_equal(
        audit.mweights(first, 0, 0, qw, scale, available, 1),
        audit.mweights(second, 0, 0, qw, scale, available, 1),
    )


def test_zero_impedance_is_valid_and_missing_is_not(cache):
    z = np.full_like(cache["impedance"], np.nan)
    z[..., 0] = 0
    scale, available = audit.metadata_scale(z)
    assert available[:, 0].all() and not available[:, 1:].any()
    assert np.all(scale == 0.1)


@pytest.mark.parametrize("budget", [0, 1, 3, 5])
def test_null_metadata_and_beta_zero(cache, budget):
    qw = audit.qweights(cache, 0, 0, {"kind": "reliability"}, budget)
    scale, available = audit.metadata_scale(cache["impedance"])
    np.testing.assert_array_equal(audit.mweights(cache, 0, 0, qw, scale, available, 0), qw)
    missing = copy.deepcopy(cache)
    missing["impedance"][:] = np.nan
    np.testing.assert_allclose(
        audit.mweights(missing, 0, 0, qw, scale, available, 1), qw, atol=1e-16
    )


def test_current_query_locality(cache):
    candidate = {"kind": "full", "tau": 0.75}
    before = audit.qweights(cache, 0, 0, candidate, 3)
    changed = copy.deepcopy(cache)
    changed["q_full_query"][0, 0, 7] += 10
    after = audit.qweights(changed, 0, 0, candidate, 3)
    np.testing.assert_array_equal(np.delete(before, 7, axis=0), np.delete(after, 7, axis=0))
    assert not np.allclose(before[7], after[7])


def test_audit_detects_bad_fit_index():
    record = {"objective_values": [0.5, 0.5, 0.6], "selected_index": 1, "temperature": 2}
    with pytest.raises(ValueError, match="first minimum"):
        audit.verify_fit_record(record, [0.5, 0.5, 0.6], [1, 2, 3], "fixture")


def test_metric_average_is_not_posterior_average():
    first = np.array([[0.6, 0.4], [0.9, 0.1]])
    second = np.array([[0.01, 0.99], [0.1, 0.9]])
    truth = np.array([0, 1])
    metric_mean = np.mean(
        [audit.metrics(p, truth, first, first)["balanced_accuracy"] for p in (first, second)]
    )
    probability_mean = audit.metrics((first + second) / 2, truth, first, first)["balanced_accuracy"]
    assert metric_mean != probability_mean


def test_paired_ci_and_harm_units():
    record = audit.paired([0.0, 0.1, -0.1, 0.2])
    assert record["n"] == 4
    assert record["mean"] == pytest.approx(0.05)
    assert record["standard_error"] == pytest.approx(np.std([0, 0.1, -0.1, 0.2], ddof=1) / 2)


def test_plan_exact_prespecified_boundary():
    plan = json.loads((ROOT / "configs/analysis/context_template_source39_v1.json").read_text())
    assert len(plan["source_subject_ids"]) == 39
    assert not set(plan["source_subject_ids"]) & {1, 2, 3}
    assert plan["canonical_channel_ids"] == [56, 55, 57, 54, 58, 62, 61, 63]
    assert plan["m"]["order_beta_grid"] == [0, 1, 4, 16]
    assert plan["reporting"]["human_held_unlock"] is False


@pytest.fixture(scope="module")
def producer_bundle(cache):
    from cfeg.analysis import context_template_source as producer

    plan = json.loads((ROOT / "configs/analysis/context_template_source39_v1.json").read_text())
    plan.update(source_subject_ids=[4, 6, 8], sample_counts=[250], expected_metadata_packets=60)
    rng = np.random.default_rng(992844)  # Unrelated artificial-score fixture only.
    values = {}
    for name in (
        "gram_cross",
        "gram_support",
        "query_norm2",
        "q_full_support",
        "q_full_query",
        "q_diag_support",
        "q_diag_query",
        "reliability",
    ):
        values[name] = np.broadcast_to(cache[name], (3, 2, *cache[name].shape[2:])).copy()
    values.update(
        schema=np.asarray("cfeg.context-template-source.features.v1"),
        subject_ids=np.asarray([4, 6, 8]),
        cell_ids=np.asarray(["dry-n250", "wet-n250"]),
        sample_counts=np.asarray([250, 250]),
        interface_indices=np.asarray([0, 1]),
        query_labels=cache["query_labels"],
        query_block_ids=cache["query_block_ids"],
        support_block_ids=np.arange(5),
        filter_weights=cache["filter_weights"],
        impedance=rng.uniform(0, 200, (3, 2, 10, 8)),
        a0_scores=rng.uniform(0, 1, (3, 2, 60, 12)),
        etrca_scores=rng.uniform(-1, 1, (3, 2, 2, 60, 12)),
        raw_file_sha256=np.asarray(["a" * 64, "b" * 64, "c" * 64]),
        crop_sha256=np.full((3, 2), "d" * 64),
    )
    values["impedance"][0, 0, 6] = np.nan
    folds = [
        producer.fit_fold(values, np.flatnonzero(np.arange(3) % 3 != fold), plan, fold_id=fold)
        for fold in range(3)
    ]
    freeze = {"folds": folds}
    rows = [
        row
        for fold in folds
        for row in producer.evaluate_fold(
            values, np.flatnonzero(np.arange(3) % 3 == fold["fold_id"]), fold, plan
        )
    ]
    result = {"rows": rows, "summary": producer.summarize_results(rows, plan)}
    return plan, values, freeze, result


def test_independent_producer_feature_fit_row_summary_agreement(producer_bundle):
    plan, values, freeze, result = producer_bundle
    audit.verify_features(values, plan)
    assert audit.verify_folds(values, freeze, plan) == 2532
    assert audit.verify_rows(values, freeze, result, plan) == 252
    report = audit.verify_summary(result, plan)
    assert report["aggregate_rows_checked"] == 84
    assert report["participant_harm_checked"] == 126
    assert report["attainment_rows_checked"] == 22


@pytest.mark.parametrize(
    "change",
    [
        "row",
        "duplicate",
        "shuffle",
        "alias",
        "screen",
        "aggregate",
        "harm",
        "attainment",
        "curve",
        "cost",
    ],
)
def test_independent_audit_rejects_corruption(producer_bundle, change):
    plan, values, freeze, original = producer_bundle
    result = copy.deepcopy(original)
    summary_only = False
    if change == "row":
        result["rows"][0]["balanced_accuracy"] += 0.1
    elif change == "duplicate":
        result["rows"].append(copy.deepcopy(result["rows"][0]))
    elif change == "shuffle":
        next(r for r in result["rows"] if r["method"] == "M_shuffle" and r["budget"] == 5)[
            "permutations"
        ] = 43
    elif change == "alias":
        next(r for r in result["rows"] if r["method"] == "M_missing")["posterior_sha256"] = "bad"
    else:
        summary_only = True
        if change == "screen":
            result["summary"]["screen_components"]["AQ_utility"] = not result["summary"][
                "screen_components"
            ]["AQ_utility"]
        elif change == "aggregate":
            result["summary"]["aggregate_rows"].pop()
        elif change == "harm":
            result["summary"]["participant_harm"][0]["delta_vs_AQ"] += 1
        elif change == "attainment":
            result["summary"]["calibration_attainment"][0]["first_observed_budget"][0] = 999
        elif change == "curve":
            result["summary"]["calibration_attainment"][0]["participant_accuracy_curve"]["0"][
                0
            ] += 0.1
        elif change == "cost":
            result["summary"]["calibration_attainment"][0]["labeled_trials"][0] = 999
    with pytest.raises(ValueError):
        if summary_only:
            audit.verify_summary(result, plan)
        else:
            audit.verify_rows(values, freeze, result, plan)


def test_posthoc_diagnosis_is_read_only_and_does_not_reselect(
    producer_bundle, tmp_path, monkeypatch
):
    import sys

    plan, values, freeze, result = producer_bundle
    monkeypatch.setitem(sys.modules, "audit_context_template_source", audit)
    spec = importlib.util.spec_from_file_location(
        "context_posthoc_fixture", ROOT / "scripts/diagnose_context_template_source.py"
    )
    diagnostic = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(diagnostic)
    np.savez_compressed(tmp_path / "features.npz", **values)
    (tmp_path / "fold-freezes.json").write_text(json.dumps(freeze))
    (tmp_path / "result.json").write_text(json.dumps(result))
    before = {p.name: audit.sha(p) for p in tmp_path.iterdir()}
    report = diagnostic.diagnose(tmp_path, plan)
    assert len(report["rows"]) == 6
    assert report["human_raw_access"] is False
    assert report["changes_to_frozen_study"] is False
    assert report["evidence_role"].startswith("posthoc_saved_feature_diagnostic")
    assert {p.name: audit.sha(p) for p in tmp_path.iterdir()} == before
    row = report["rows"][0]
    expected = []
    for p in range(3):
        fold = freeze["folds"][p % 3]
        candidate = next(
            c for c in plan["q"]["candidates"] if c["id"] == fold["q_selection"]["candidate_id"]
        )
        score = audit.template_scores(values, p, 0, audit.qweights(values, p, 0, candidate, 1))
        expected.append(float((score.argmax(-1) == values["query_labels"]).mean()))
    assert row["standalone_template_participant_BA"] == expected
