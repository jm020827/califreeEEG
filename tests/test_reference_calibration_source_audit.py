"""Artificial feature arithmetic and provenance fixtures only; no human-data execution."""

from __future__ import annotations

import ast
import copy
import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "reference_auditor_fixture", ROOT / "scripts/audit_reference_calibration_source.py"
)
audit = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(audit)


@pytest.fixture(scope="module")
def features():
    plan = json.loads(
        (ROOT / "configs/analysis/reference_calibration_source39_v1.json").read_text()
    )
    plan.update(source_subject_ids=[4, 6, 8], sample_counts=[125])
    rng = np.random.default_rng(7113509)  # Artificial fixture only, not a study RNG.
    r = rng.uniform(-0.6, 0.6, (3, 2, 2, 3, 7, 60, 12, 4))
    aref = rng.uniform(0.05, 0.65, (3, 2, 2, 7, 60, 12))
    for k in range(3):
        r[:, :, :, k, ..., 0] = aref
    cache = {
        "schema": np.asarray("cfeg.reference-calibration-source.features.v1"),
        "subject_ids": np.array([4, 6, 8]),
        "pipeline_ids": np.array(plan["pipeline_ids"]),
        "cell_ids": np.array(["dry-n125", "wet-n125"]),
        "sample_counts": np.array([125, 125]),
        "interface_indices": np.array([0, 1]),
        "query_labels": np.tile(np.arange(12), 5),
        "query_block_ids": np.repeat(np.arange(5, 10), 12),
        "support_block_ids": np.arange(5),
        "calibration_budgets": np.array([1, 3, 5]),
        "wrong_label_shifts": np.arange(1, 12),
        "filter_weights": np.arange(1, 8, dtype=float),
        "ecca_components": r,
        "aref_band_rho": aref,
        "wrong_ecca_scores": rng.uniform(-1, 1, (3, 2, 2, 3, 11, 60, 12)),
        "itcca_band_rho": rng.uniform(0, 1, (3, 2, 2, 3, 7, 60, 12)),
        "legacy_a0_scores": rng.uniform(0, 1, (3, 2, 60, 12)),
        "support_correlations": np.broadcast_to(np.eye(5), (3, 2, 2, 7, 12, 5, 5)).copy(),
        "reference_projection_fraction": rng.uniform(0, 1, (3, 2, 2, 7, 10, 12, 12)),
        "line50_fraction": rng.uniform(0, 1, (3, 2, 2, 10, 12)),
        "raw_file_sha256": np.array(["a" * 64, "b" * 64, "c" * 64]),
        "crop_sha256": np.full((3, 2, 2), "d" * 64),
    }
    return plan, cache


def test_feature_inventory_and_independent_score_formula(features):
    plan, cache = features
    audit.verify_features(cache, plan)
    scores = audit.reconstructed_scores(cache)
    r = cache["ecca_components"][0, 1, 0, 2]
    w = cache["filter_weights"]
    direct = sum(
        w[f] * sum(np.sign(r[f, ..., i]) * r[f, ..., i] ** 2 for i in range(4)) for f in range(7)
    ) / sum(w)
    np.testing.assert_allclose(scores["ECCA"][0, 1, 0, 2], direct, rtol=0, atol=2e-16)
    assert np.any(scores["ECCA"] < 0), "Signed score must not be squared a second time"


@pytest.mark.parametrize(
    "change", ["descriptor", "missing", "r1", "bounds", "subject", "shift", "symmetry"]
)
def test_feature_corruption(features, change):
    plan, original = features
    cache = copy.deepcopy(original)
    if change == "descriptor":
        cache["impedance"] = np.ones(3)
    elif change == "missing":
        del cache["reference_projection_fraction"]
    elif change == "r1":
        cache["ecca_components"][0, 0, 0, 1, 0, 0, 0, 0] += 0.01
    elif change == "bounds":
        cache["line50_fraction"][0, 0, 0, 0, 0] = 1.5
    elif change == "subject":
        cache["subject_ids"][0] = 1
    elif change == "shift":
        cache["wrong_label_shifts"][0] = 0
    else:
        cache["support_correlations"][0, 0, 0, 0, 0, 0, 1] = 0.5
    with pytest.raises(ValueError):
        audit.verify_features(cache, plan)


def test_every_band_diagnostic_and_true_reference_axis(features):
    _, original = features
    cache = copy.deepcopy(original)
    fraction = cache["reference_projection_fraction"]
    fraction[...] = 0.1
    for c in range(12):
        fraction[..., c, c] = 0.8
    report = audit.diagnostic_report(cache)
    assert len(report["score_rows"]) == 2 * 2 * 7 * 7
    assert len(report["signal_rows"]) == 2 * 2 * 7
    assert len(report["line50_rows"]) == 4
    row = report["signal_rows"][0]
    assert row["support_offdiagonal_correlation"]["mean"] == 0
    assert row["query_matched_reference_fraction"]["mean"] == pytest.approx(0.8)
    assert row["support_reference_margin"]["mean"] == pytest.approx(0.7)
    assert len(row["query_reference_margin"]["participant_values"]) == 3


@pytest.fixture(scope="module")
def producer_bundle(features):
    from cfeg.analysis import reference_calibration_source as producer

    plan, cache = features
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


def test_producer_and_independent_auditor_agree(producer_bundle):
    plan, cache, freeze, result = producer_bundle
    assert audit.verify_folds(cache, freeze, plan) == 1395
    assert audit.verify_rows(cache, freeze, result, plan) == 204
    report = audit.verify_summary(result, plan)
    assert report["no_notch"]["aggregate_rows_checked"] == 38
    assert report["causal_notch50"]["attainment_rows_checked"] == 8


@pytest.mark.parametrize(
    "change",
    [
        "metric",
        "duplicate",
        "wrong_shift",
        "wrong_metric",
        "alias",
        "global_status",
        "gate",
        "aggregate",
        "harm",
        "attainment",
        "history",
        "interaction",
        "pipeline",
    ],
)
def test_result_corruption(producer_bundle, change):
    plan, cache, freeze, original = producer_bundle
    result = copy.deepcopy(original)
    rows, summary = result["rows"], result["summary"]
    nested = summary["pipeline_summaries"]["no_notch"]
    summary_only = False
    if change == "metric":
        rows[0]["balanced_accuracy"] += 0.1
    elif change == "duplicate":
        rows.append(copy.deepcopy(rows[0]))
    elif change in ("wrong_shift", "wrong_metric"):
        row = next(r for r in rows if r["method"] == "wrong_label_support" and r["budget"] == 3)
        row["permutation_metrics"][0]["shift" if change == "wrong_shift" else "correct_count"] += 1
    elif change == "alias":
        next(r for r in rows if r["method"] == "ECCA" and r["budget"] == 0)["posterior_sha256"] = (
            "f" * 64
        )
    else:
        summary_only = True
        if change == "global_status":
            summary["status"] = "USEFUL_EEG_CALIBRATION_DEVELOPMENT"
        elif change == "gate":
            nested["screen_components"]["AQ_utility"] = not nested["screen_components"][
                "AQ_utility"
            ]
        elif change == "aggregate":
            nested["aggregate_rows"][0]["mean_metrics"]["balanced_accuracy"] += 0.1
        elif change == "harm":
            nested["participant_harm"][0]["delta_vs_A0"] += 0.1
        elif change == "attainment":
            nested["calibration_attainment"][0]["first_observed_budget"][0] = -1
        elif change == "history":
            nested["calibration_attainment"][0]["history_seconds"][0] = -1
        elif change == "interaction":
            summary["paired_pipeline_diagnostics"]["calibration_gain_interaction"]["mean"] += 0.1
        elif change == "pipeline":
            del summary["pipeline_summaries"]["causal_notch50"]
    with pytest.raises(ValueError):
        audit.verify_summary(result, plan) if summary_only else audit.verify_rows(
            cache, freeze, result, plan
        )


@pytest.mark.parametrize(
    "change", ["fit_ids", "eval_ids", "objective", "T", "grid", "extra", "legacy"]
)
def test_fit_corruption(producer_bundle, change):
    plan, cache, original, _ = producer_bundle
    freeze = copy.deepcopy(original)
    fold = freeze["folds"][0]
    window = fold["pipelines"]["no_notch"]["windows"]["125"]
    fit = window["budgets"]["1"]["ecca_temperature"]
    if change == "fit_ids":
        fold["fit_subject_ids"][0] = 1
    elif change == "eval_ids":
        fold["evaluation_subject_ids"][0] = 6
    elif change == "objective":
        fit["objectives"][0] += 0.1
    elif change == "T":
        fit["temperature"] = 0.123456
    elif change == "grid":
        fit["temperatures"].reverse()
    elif change == "extra":
        window["budgets"]["1"]["lambda"] = 0.5
    else:
        fold["legacy_a0_temperatures"]["125"]["nll"] += 0.1
    with pytest.raises(ValueError):
        audit.verify_folds(cache, freeze, plan)


def test_no_decoder_loader_import_and_frozen_boundary():
    text = (ROOT / "scripts/audit_reference_calibration_source.py").read_text()
    imports = [n.module or "" for n in ast.walk(ast.parse(text)) if isinstance(n, ast.ImportFrom)]
    assert not any(n.startswith("cfeg") for n in imports)
    assert "loadmat(" not in text and "read_parquet(" not in text
    plan_path = ROOT / "configs/analysis/reference_calibration_source39_v1.json"
    assert (
        audit.sha(plan_path) == "d7f89ef223d0ad0637f449bc021711167f9f1d5b8dabeba068b6bf8c11131e21"
    )
    plan = json.loads(plan_path.read_text())
    assert len(plan["source_subject_ids"]) == 39 and not set(plan["source_subject_ids"]) & {1, 2, 3}
    assert not plan["authority"]["metadata_access"] and not plan["authority"]["held_eeg_access"]


@pytest.fixture
def provenance_bundle(tmp_path, monkeypatch):
    root = tmp_path / "artifacts"
    root.mkdir()
    plan_path = tmp_path / "plan.json"
    helper = b"synthetic reference fixture helper"
    plan = {
        "study_id": "fixture",
        "source_subject_ids": [4, 6, 8],
        "evidence_role": "artificial-only",
        "execution": {
            "artifacts": ["start.json", "features.npz", "fold-freezes.json", "result.json"],
            "resource_budget_bytes": 100000,
            "workers": 4,
        },
        "pinned_files": {"synthetic.py": hashlib.sha256(helper).hexdigest()},
    }
    plan_path.write_text(json.dumps(plan))
    start = {
        "schema": "cfeg.reference-calibration-source.start.v1",
        "study_id": "fixture",
        "plan_sha256": audit.sha(plan_path),
        "source_commit": "a" * 40,
        "source_tree": "b" * 40,
        "source_subject_ids": [4, 6, 8],
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
    (root / "features.npz").write_bytes(b"provenance fixture, not actual feature archive")
    shared = {
        "study_id": "fixture",
        "plan_sha256": audit.sha(plan_path),
        "source_commit": start["source_commit"],
        "start_sha256": audit.sha(root / "start.json"),
        "features_sha256": audit.sha(root / "features.npz"),
    }
    freeze = {"schema": "cfeg.reference-calibration-source.fold-freezes.v1", **shared}
    (root / "fold-freezes.json").write_text(json.dumps(freeze))
    result = {
        "schema": "cfeg.reference-calibration-source.result.v1",
        **shared,
        "fold_freezes_sha256": audit.sha(root / "fold-freezes.json"),
        "evidence_role": "artificial-only",
        "summary": {"human_held_unlock": False},
    }
    (root / "result.json").write_text(json.dumps(result))
    for path in root.iterdir():
        path.chmod(0o400)

    def git(command):
        if command[3] == "rev-parse":
            return (start["source_tree"] + "\n").encode()
        return helper if command[4].endswith("synthetic.py") else plan_path.read_bytes()

    monkeypatch.setattr(audit.subprocess, "check_output", git)
    return root, plan_path, plan, start, freeze, result, ROOT


def test_provenance_chain(provenance_bundle):
    audit.verify_provenance(*provenance_bundle)


@pytest.mark.parametrize(
    "change", ["metadata", "held", "commit", "features", "freeze", "plan", "mode", "extra", "pins"]
)
def test_provenance_corruption(provenance_bundle, change):
    root, path, plan, start, freeze, result, repo = provenance_bundle
    if change == "metadata":
        start["metadata_access"] = True
    elif change == "held":
        result["summary"]["human_held_unlock"] = True
    elif change == "commit":
        freeze["source_commit"] = "e" * 40
    elif change == "features":
        freeze["features_sha256"] = "e" * 64
    elif change == "freeze":
        result["fold_freezes_sha256"] = "e" * 64
    elif change == "plan":
        result["plan_sha256"] = "e" * 64
    elif change == "mode":
        (root / "features.npz").chmod(0o600)
    elif change == "extra":
        (root / "extra").write_text("fixture")
    else:
        start["pinned_files"] = {}
    with pytest.raises(ValueError):
        audit.verify_provenance(root, path, plan, start, freeze, result, repo)
