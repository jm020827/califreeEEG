"""Independent score-only audit fixtures; no declared study RNG or EEG generation."""

from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

from cfeg.analysis import metadata_calibration_v4_aq_dev as study

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "aq_audit_fixture", ROOT / "scripts/audit_metadata_calibration_v4_aq_study.py"
)
audit = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(audit)


@pytest.fixture(scope="module")
def bundle():
    plan = json.loads((ROOT / "configs/analysis/metadata_calibration_v4_aq_study.json").read_text())
    plan.update(
        source_participants=4,
        evaluation_participants=4,
        n_classes=3,
        query_blocks=2,
        sample_counts=[75],
        noise_multipliers=[1.0],
        seed_namespace="independent-auditor-unit-fixture-only",
    )
    rng = np.random.default_rng(314159)  # Arbitrary score fixture, not an EEG study root.

    def cache(split):
        labels = np.tile(np.arange(3), (4, 7, 1))
        truth = labels[:, 5:].reshape(4, 6)
        a0 = np.zeros((4, 2, 6, 3))
        for p in range(4):
            for c in range(2):
                prediction = truth[p].copy()
                prediction[:2] = (prediction[:2] + 1) % 3
                a0[p, c, np.arange(6), prediction] = 0.4
        block = rng.uniform(-0.2, 0.3, (4, 2, 6, 5, 3))
        pooled = rng.uniform(-0.2, 0.3, (4, 2, 3, 6, 3))
        pooled[:, :, 0] = block[:, :, :, 0]
        digest = lambda text: hashlib.sha256(text.encode()).hexdigest()
        return {
            "schema": np.asarray(study.CACHE_SCHEMA),
            "split": np.asarray(split),
            "a0_scores": a0,
            "block_scores": block,
            "pooled_scores": pooled,
            "block_reliability": rng.uniform(0, 1, (4, 2, 5)),
            "labels": labels,
            "participant_ids": np.arange(4),
            "cell_ids": np.asarray([c["cell_id"] for c in study.cells(plan)]),
            "base_sha256": np.asarray([digest(f"{split}/{p}") for p in range(4)]),
            "eeg_sha256": np.asarray(
                [[digest(f"{split}/{p}/{c}") for c in range(2)] for p in range(4)]
            ),
        }

    source, evaluation = cache("source"), cache("evaluation")
    freeze = {**study.fit_temperatures(source, plan), **study.source_difficulty(source, plan)}
    rows = study.evaluate_rows(evaluation, freeze, plan)
    result = {"rows": rows, "summary": study.summarize_results(rows, freeze, plan)}
    return plan, source, evaluation, freeze, result


def test_independent_full_arithmetic_and_inventories(bundle):
    plan, source, evaluation, freeze, result = bundle
    audit.verify_source(source, freeze, plan)
    report = audit.verify_evaluation(evaluation, freeze, result, plan)
    assert report["audit"] == "PASS"
    assert report["metric_rows_checked"] == 320
    assert report["attainment_rows_checked"] == 20
    assert report["diagnostics_checked"] == 80
    assert report["selected_stable_cells"] == ["stable-n075-m1"]


@pytest.mark.parametrize(
    "change",
    [
        "row",
        "row_duplicate",
        "aggregate_empty",
        "aggregate_missing",
        "attainment",
        "cost",
        "diagnostic",
        "screen",
        "human",
        "selection",
        "alias",
    ],
)
def test_independent_audit_rejects_corruption(bundle, change):
    plan, _, evaluation, freeze, original = bundle
    result = copy.deepcopy(original)
    summary = result["summary"]
    if change == "row":
        result["rows"][0]["balanced_accuracy"] += 0.01
    elif change == "row_duplicate":
        result["rows"].append(copy.deepcopy(result["rows"][0]))
    elif change == "aggregate_empty":
        summary["aggregate_rows"][0]["mean_metrics"] = {}
    elif change == "aggregate_missing":
        summary["aggregate_rows"].pop()
    elif change == "attainment":
        summary["calibration_attainment"][0]["first_observed_budget"][0] = 0
    elif change == "cost":
        summary["calibration_attainment"][0]["labeled_trials"][0] = 72
    elif change == "diagnostic":
        summary["diagnostics"][0]["n"] = 8
    elif change == "screen":
        screen = summary["primary"]["screen_components"]
        screen["eauc_mean"] = not screen["eauc_mean"]
    elif change == "human":
        summary["human_unlock"] = True
    elif change == "selection":
        summary["selected_stable_cells"] = []
    elif change == "alias":
        next(r for r in result["rows"] if r["method"] == "pooled_raw" and r["budget"] == 0)[
            "posterior_sha256"
        ] = "0" * 64
    with pytest.raises(ValueError):
        audit.verify_evaluation(evaluation, freeze, result, plan)


def test_independent_source_fit_and_no_informative_cells(bundle):
    plan, source, evaluation, freeze, _ = bundle
    bad = copy.deepcopy(freeze)
    bad["fits"]["block_k3"]["selected_index"] += 1
    with pytest.raises(ValueError, match="minimum"):
        audit.verify_source(source, bad, plan)
    bad = copy.deepcopy(freeze)
    bad["source_difficulty_rows"].append(bad["source_difficulty_rows"][0])
    with pytest.raises(ValueError, match="Duplicate"):
        audit.verify_source(source, bad, plan)
    none = {**freeze, "selected_stable_cells": []}
    rows = study.evaluate_rows(evaluation, none, plan)
    result = {"rows": rows, "summary": study.summarize_results(rows, none, plan)}
    assert (
        audit.verify_evaluation(evaluation, none, result, plan)["status"]
        == "NO_SOURCE_INFORMATIVE_CELLS"
    )


def test_exclusive_fixture_artifact_provenance(bundle, tmp_path, monkeypatch):
    plan, source, evaluation, _, _ = bundle
    plan = copy.deepcopy(plan)
    # Only the fixture namespace above is hashed; no study seed or DGP is used.
    plan_path = tmp_path / "fixture-plan.json"
    plan_path.write_text(json.dumps(plan))
    output = tmp_path / "output"
    output.mkdir()
    start = {
        "schema": "cfeg.v4-aq-study.start.v1",
        "study_id": plan["study_id"],
        "evidence_role": plan["evidence_role"],
        "source_commit": "fixture-commit",
        "source_tree": "fixture-tree",
        "plan_sha256": audit.sha(plan_path),
        "human_data_access": False,
        "seed_namespace": plan["seed_namespace"],
        "workers": 4,
        "filterbank_sha256": plan["filterbank_sha256"],
        "pure_helper_module_sha256": plan["pure_helper_module_sha256"],
        "runtime": {
            "blas_threads": {
                k: "1" for k in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS")
            }
        },
    }
    start_sha = study.numeric.write_json_exclusive(output / "start.json", start)
    monkeypatch.setattr(
        study,
        "collect_scores",
        lambda p, f, split, seed, w: source if split == "source" else evaluation,
    )
    study.execute_phases(plan, {}, output, start, start_sha, 4)
    freeze, evaluation_start, result = [
        json.loads((output / name).read_text())
        for name in ("source-freeze.json", "evaluation-start.json", "result.json")
    ]

    def git_fixture(args):
        if args[-2] == "rev-parse":
            return b"fixture-tree\n"
        path = args[-1].split(":", 1)[1]
        return (
            plan_path.read_bytes()
            if path.endswith("metadata_calibration_v4_aq_study.json")
            else (ROOT / path).read_bytes()
        )

    monkeypatch.setattr(audit.subprocess, "check_output", git_fixture)
    audit.verify_provenance(output, plan_path, plan, start, freeze, evaluation_start, result, ROOT)
    bad = {**result, "start_sha256": "0" * 64}
    with pytest.raises(ValueError, match="Start binding"):
        audit.verify_provenance(output, plan_path, plan, start, freeze, evaluation_start, bad, ROOT)
    bad = {**evaluation_start, "source_commit": "wrong"}
    with pytest.raises(ValueError, match="Source commit"):
        audit.verify_provenance(output, plan_path, plan, start, freeze, bad, result, ROOT)
