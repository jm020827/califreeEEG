"""Artificial score fixtures only; independent auditor never imports decoders."""

from __future__ import annotations

import copy
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "spatial_audit_fixture", ROOT / "scripts/audit_spatial_calibration_source.py"
)
audit = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(audit)


def test_exact_fusion_zero_and_uniform_argmax():
    a0 = np.array([[0.1, 0.6, 0.3], [0.5, 0.2, 0.3]])
    assert audit.mixture(a0, None, 0, None) is a0
    for weight in (0.25, 0.5, 0.75):
        uniform = (1 - weight) * a0 + weight / 3
        np.testing.assert_array_equal(uniform.argmax(-1), a0.argmax(-1))


def test_metric_average_not_probability_average():
    p = np.array([[0.6, 0.4], [0.9, 0.1]])
    q = np.array([[0.01, 0.99], [0.1, 0.9]])
    labels = np.array([0, 1])
    metrics = np.mean([audit.metrics(v, labels, p)["balanced_accuracy"] for v in (p, q)])
    assert metrics != audit.metrics((p + q) / 2, labels, p)["balanced_accuracy"]


def test_participant_ci_and_nll():
    values = np.array([0.1, 0, -0.1, 0.2])
    result = audit.paired(values)
    assert result["n"] == 4 and result["mean"] == pytest.approx(0.05)
    assert result["standard_error"] == pytest.approx(values.std(ddof=1) / 2)
    probability = np.broadcast_to(np.array([[0.7, 0.3], [0.2, 0.8]]), (3, 2, 2, 2))
    assert audit.nll(probability, np.array([0, 1])) == pytest.approx(-np.log([0.7, 0.8]).mean())


def test_temperature_first_exact_tie():
    scores = np.zeros((2, 2, 60, 12))
    labels = np.tile(np.arange(12), 5)
    grid = np.logspace(-4, 1, 31)
    objectives = [audit.nll(audit.softmax(scores, t), labels) for t in grid]
    record = {
        "temperatures": grid.tolist(),
        "objectives": objectives,
        "selected_index": 0,
        "temperature": grid[0],
        "nll": objectives[0],
    }
    assert audit.verify_temperature(record, scores, labels, "fixture") == 31
    record.update(selected_index=1, temperature=grid[1])
    with pytest.raises(ValueError, match="first minimum"):
        audit.verify_temperature(record, scores, labels, "fixture")


def test_plan_boundary():
    plan = json.loads((ROOT / "configs/analysis/spatial_calibration_source39_v1.json").read_text())
    assert len(plan["source_subject_ids"]) == 39
    assert not set(plan["source_subject_ids"]) & {1, 2, 3}
    assert plan["authority"]["metadata_access"] is False
    assert plan["authority"]["held_eeg_access"] is False
    assert plan["fitting"]["expected_objective_count"] == 4770
    assert plan["reporting"]["expected_rows"] == 5382
    assert (
        plan["decoder"]["FB_eTRCA_unit_ridge_formula"] == ".001 * max(trace(Q)/channels, 1e-12) * I"
    )


@pytest.fixture(scope="module")
def producer_bundle():
    from cfeg.analysis import spatial_calibration_source as producer

    plan = json.loads((ROOT / "configs/analysis/spatial_calibration_source39_v1.json").read_text())
    plan.update(source_subject_ids=[4, 6, 8], sample_counts=[250])
    rng = np.random.default_rng(805319)  # Unrelated artificial scores, no study draws/data.
    cache = {
        "schema": np.asarray("cfeg.spatial-calibration-source.scores.v1"),
        "subject_ids": np.array([4, 6, 8]),
        "cell_ids": np.array(["dry-n250", "wet-n250"]),
        "sample_counts": np.array([250, 250]),
        "interface_indices": np.array([0, 1]),
        "query_labels": np.tile(np.arange(12), 5),
        "query_block_ids": np.repeat(np.arange(5, 10), 12),
        "support_block_ids": np.arange(5),
        "calibration_budgets": np.array([1, 3, 5]),
        "trca_budgets": np.array([3, 5]),
        "a0_scores": rng.uniform(0, 1, (3, 2, 60, 12)),
        "itcca_scores": rng.uniform(0, 1, (3, 2, 3, 60, 12)),
        "etrca_unit_scores": rng.uniform(-1, 1, (3, 2, 2, 60, 12)),
        "legacy_etrca_scores": rng.uniform(-1, 1, (3, 2, 2, 60, 12)),
        "raw_file_sha256": np.asarray(["a" * 64, "b" * 64, "c" * 64]),
        "crop_sha256": np.full((3, 2), "d" * 64),
        "filter_weights": np.arange(1, 8, dtype=float),
    }
    folds = [
        producer.fit_fold(cache, np.flatnonzero(np.arange(3) % 3 != f), plan, fold_id=f)
        for f in range(3)
    ]
    rows = [
        r
        for fold in folds
        for r in producer.evaluate_fold(
            cache, np.flatnonzero(np.arange(3) % 3 == fold["fold_id"]), fold, plan
        )
    ]
    return (
        plan,
        cache,
        {"folds": folds},
        {"rows": rows, "summary": producer.summarize_results(rows, plan)},
    )


def test_independent_all_layers_agree(producer_bundle):
    plan, cache, freeze, result = producer_bundle
    audit.verify_scores(cache, plan)
    assert audit.verify_folds(cache, freeze, plan) == 1590
    assert audit.verify_rows(cache, freeze, result, plan) == 138
    report = audit.verify_summary(result, plan)
    assert report["aggregate_rows_checked"] == 46
    assert report["participant_harm_checked"] == 69
    assert report["attainment_rows_checked"] == 14


@pytest.mark.parametrize(
    "change",
    [
        "row",
        "duplicate",
        "permutation",
        "shift",
        "alias",
        "gate",
        "aggregate",
        "harm",
        "attainment",
        "cost",
        "curve",
    ],
)
def test_auditor_rejects_corruption(producer_bundle, change):
    plan, cache, freeze, original = producer_bundle
    result = copy.deepcopy(original)
    rows = result["rows"]
    summary_only = False
    if change == "row":
        rows[0]["balanced_accuracy"] += 0.1
    elif change == "duplicate":
        rows.append(copy.deepcopy(rows[0]))
    elif change == "permutation":
        next(r for r in rows if r["method"] == "wrong_label_support" and r["budget"] == 3)[
            "permutations"
        ] = 10
    elif change == "shift":
        next(r for r in rows if r["method"] == "wrong_label_support" and r["budget"] == 1)[
            "permutation_metrics"
        ][0]["balanced_accuracy"] += 0.1
    elif change == "alias":
        next(r for r in rows if r["method"] == "AQ_ITCCA" and r["budget"] == 0)[
            "posterior_sha256"
        ] = "e" * 64
    else:
        summary_only = True
        summary = result["summary"]
        if change == "gate":
            summary["screen_components"]["AQ_utility"] = not summary["screen_components"][
                "AQ_utility"
            ]
        elif change == "aggregate":
            summary["aggregate_rows"][0]["mean_metrics"]["balanced_accuracy"] += 0.1
        elif change == "harm":
            summary["participant_harm"][0]["delta_vs_A0"] += 0.1
        elif change == "attainment":
            summary["calibration_attainment"][0]["first_observed_budget"][0] = -1
        elif change == "cost":
            summary["calibration_attainment"][0]["EEG_seconds"][0] = -1
        elif change == "curve":
            key = next(iter(summary["calibration_attainment"][0]["participant_accuracy_curve"]))
            summary["calibration_attainment"][0]["participant_accuracy_curve"][key][0] += 0.1
    with pytest.raises(ValueError):
        audit.verify_summary(result, plan) if summary_only else audit.verify_rows(
            cache, freeze, result, plan
        )


@pytest.mark.parametrize(
    "change", ["fit_ids", "eval_ids", "objective", "lambda", "standalone_T", "grid"]
)
def test_auditor_rejects_fit_corruption(producer_bundle, change):
    plan, cache, original, _ = producer_bundle
    freeze = copy.deepcopy(original)
    fold = freeze["folds"][0]
    window = fold["windows"]["250"]
    fit = window["budgets"]["1"]["fusion_fit"]
    if change == "fit_ids":
        fold["fit_subject_ids"][0] = 4
    elif change == "eval_ids":
        fold["evaluation_subject_ids"][0] = 6
    elif change == "objective":
        fit["candidates"][0]["nll"] += 0.1
    elif change == "lambda":
        fit["lambda"] = 0.99
    elif change == "standalone_T":
        window["budgets"]["1"]["standalone_temperatures"]["ITCCA"]["temperature"] = 0.123456
    elif change == "grid":
        fit["candidates"].reverse()
    with pytest.raises(ValueError):
        audit.verify_folds(cache, freeze, plan)


def test_no_producer_or_human_import_in_auditor():
    import ast

    source = (ROOT / "scripts/audit_spatial_calibration_source.py").read_text()
    tree = ast.parse(source)
    imports = [n.module or "" for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)]
    assert not any(n.startswith("cfeg") for n in imports)
    assert "loadmat(" not in source and "read_parquet(" not in source


@pytest.fixture
def provenance_bundle(tmp_path, monkeypatch):
    import hashlib

    root = tmp_path / "artifacts"
    root.mkdir()
    plan_path = tmp_path / "plan.json"
    helper = b"synthetic pinned source fixture"
    plan = {
        "study_id": "fixture-only",
        "source_subject_ids": [4, 6, 8],
        "evidence_role": "artificial-fixture",
        "execution": {
            "artifacts": ["start.json", "scores.npz", "fold-freezes.json", "result.json"],
            "resource_budget_bytes": 100000,
            "workers": 4,
        },
        "pinned_files": {"synthetic.py": hashlib.sha256(helper).hexdigest()},
    }
    plan_path.write_text(json.dumps(plan))
    start = {
        "schema": "cfeg.spatial-calibration-source.start.v1",
        "study_id": plan["study_id"],
        "plan_sha256": audit.sha(plan_path),
        "source_commit": "a" * 40,
        "source_tree": "b" * 40,
        "source_subject_ids": plan["source_subject_ids"],
        "workers": 4,
        "metadata_access": False,
        "human_held_access": False,
        "pinned_files": plan["pinned_files"],
        "runtime": {
            "blas_threads": {
                k: "1" for k in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS")
            }
        },
    }
    (root / "start.json").write_text(json.dumps(start))
    (root / "scores.npz").write_bytes(b"only provenance fixture, not a score archive")
    provenance = {
        "study_id": plan["study_id"],
        "plan_sha256": audit.sha(plan_path),
        "source_commit": start["source_commit"],
        "start_sha256": audit.sha(root / "start.json"),
        "scores_sha256": audit.sha(root / "scores.npz"),
    }
    freeze = {"schema": "cfeg.spatial-calibration-source.fold-freezes.v1", **provenance}
    (root / "fold-freezes.json").write_text(json.dumps(freeze))
    result = {
        "schema": "cfeg.spatial-calibration-source.result.v1",
        **provenance,
        "fold_freezes_sha256": audit.sha(root / "fold-freezes.json"),
        "evidence_role": plan["evidence_role"],
        "summary": {"human_held_unlock": False},
    }
    (root / "result.json").write_text(json.dumps(result))
    for path in root.iterdir():
        path.chmod(0o400)

    def git(command):
        if command[3] == "rev-parse":
            return (start["source_tree"] + "\n").encode()
        if command[4].endswith("synthetic.py"):
            return helper
        return plan_path.read_bytes()

    monkeypatch.setattr(audit.subprocess, "check_output", git)
    return root, plan_path, plan, start, freeze, result, ROOT


def test_provenance_chain(provenance_bundle):
    audit.verify_provenance(*provenance_bundle)


@pytest.mark.parametrize(
    "change",
    [
        "metadata",
        "held",
        "commit",
        "score_hash",
        "freeze_hash",
        "plan_hash",
        "mode",
        "extra",
        "helpers",
    ],
)
def test_provenance_rejects_corruption(provenance_bundle, change):
    root, plan_path, plan, start, freeze, result, repo = provenance_bundle
    if change == "metadata":
        start["metadata_access"] = True
    elif change == "held":
        result["summary"]["human_held_unlock"] = True
    elif change == "commit":
        freeze["source_commit"] = "c" * 40
    elif change == "score_hash":
        freeze["scores_sha256"] = "d" * 64
    elif change == "freeze_hash":
        result["fold_freezes_sha256"] = "d" * 64
    elif change == "plan_hash":
        result["plan_sha256"] = "d" * 64
    elif change == "mode":
        (root / "scores.npz").chmod(0o600)
    elif change == "extra":
        (root / "unexpected.txt").write_text("fixture")
    elif change == "helpers":
        start["pinned_files"] = {}
    with pytest.raises(ValueError):
        audit.verify_provenance(root, plan_path, plan, start, freeze, result, repo)
