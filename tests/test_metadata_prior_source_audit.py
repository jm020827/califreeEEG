"""Artificial-array and provenance corruption checks; never read source39 artifacts."""

from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "metadata_prior_independent_audit", ROOT / "scripts/audit_metadata_prior_source.py"
)
audit = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(audit)


@pytest.fixture
def plan():
    value = json.loads((ROOT / "configs/analysis/metadata_prior_source39_v1.json").read_text())
    value["source_subject_ids"] = [4, 6, 8]
    value["sample_counts"] = [125]
    return value


def test_paired_statistics_are_participant_t_not_query_intervals():
    result = audit.paired_stats([-0.03, -0.01, 0.01, 0.03])
    radius = 3.182446305284263 * np.sqrt(0.002 / 3) / 2
    assert result["n"] == 4
    assert result["mean"] == pytest.approx(0)
    assert result["lower"] == pytest.approx(-radius)
    assert result["upper"] == pytest.approx(radius)
    assert (result["help"], result["tie"], result["harm"]) == (2, 0, 2)
    tied = audit.paired_stats([0, 0, 0])
    assert tied["lower"] == tied["upper"] == 0
    assert tied["tie"] == 3


def test_accuracy_balanced_queries_and_first_tie_argmax(plan):
    scores = np.zeros((3, 2, 1, 2, 9, 48, 12))
    anchor = np.zeros((3, 2, 1, 48, 12))
    for query in range(48):
        scores[..., query, query % 12] = 1
    scores[0, 0, 0, 0, 0, 0, :] = 0
    scores[0, 0, 0, 0, 0, 1, :] = 0
    accuracy, a0, predictions, a0_predictions = audit.accuracy_arrays(scores, anchor, plan)
    assert accuracy[0, 0, 0, 0, 0] == 47 / 48
    assert accuracy[2, 1, 0, 1, 8] == 1
    np.testing.assert_array_equal(a0, np.full((3, 2, 1), 1 / 12))
    assert predictions[0, 0, 0, 0, 0, :2].tolist() == [0, 0]
    assert np.all(a0_predictions == 0)


def test_cost_preserves_unreached_and_nonmonotonic_first_observed(plan):
    accuracy = np.zeros((3, 2, 1, 2, 9))
    anchor = np.zeros((3, 2, 1))
    anchor[0, 0, 0] = 0.8
    accuracy[1, 0, 0, 0] = 0.8
    accuracy[2, 0, 0, 1] = 0.9
    first, costs = audit.cost_arrays(accuracy, anchor, plan)
    assert np.all(first[0, 0, 0] == 0)
    assert np.all(first[1, 0, 0] == 3)
    assert np.all(first[2, 0, 0] == 5)
    assert np.all(costs[:, 0, 0] == np.array([0, 36, 60])[:, None])
    assert np.isnan(first[:, 1]).all()
    assert np.isnan(costs[:, 1]).all()


@pytest.mark.parametrize("alpha", [0, 0.0001, 0.1, 1])
def test_selected_ridge_replay_matches_independent_augmented_fit(alpha):
    x = np.array([[1, 2, 1], [3, 1, 1], [-2, 4, 1], [0, 0, 1], [4, -2, 1]], dtype=float)
    y = np.array([1, -1, 3, -2, 0.5])
    mean, scale = x.mean(0), x.std(0)
    scale[scale < 1e-12] = 1
    z = (x - mean) / scale
    augmented = np.vstack((z / np.sqrt(len(x)), np.sqrt(alpha) * np.eye(3)))
    response = np.concatenate(((y - y.mean()) / np.sqrt(len(x)), np.zeros(3)))
    record = {
        "mean": mean.tolist(),
        "scale": scale.tolist(),
        "coefficient": np.linalg.lstsq(augmented, response, rcond=None)[0].tolist(),
        "intercept": float(y.mean()),
        "alpha": alpha,
        "design_rank": int(np.linalg.matrix_rank(z / np.sqrt(len(x)))),
    }
    recovered = audit.verify_ridge(record, x, y, alpha=alpha)
    np.testing.assert_allclose(
        audit.predict_record(record, x), z @ recovered["coefficient"] + recovered["intercept"]
    )
    bad = copy.deepcopy(record)
    bad["coefficient"][0] += 0.01
    with pytest.raises(ValueError, match="coefficient"):
        audit.verify_ridge(bad, x, y, alpha=alpha)


def test_context_does_not_pool_participants_or_budgets():
    q = np.arange(3 * 2 * 2 * 4.0).reshape(3, 2, 2, 4, 1)
    context = audit.channel_design(q, "context")
    np.testing.assert_array_equal(context[..., :1], q)
    np.testing.assert_allclose(
        context[..., 1:], np.broadcast_to(q.mean(-2, keepdims=True), q.shape)
    )
    changed = q.copy()
    changed[1, 1] += 100
    np.testing.assert_array_equal(audit.channel_design(changed, "context")[0], context[0])
    np.testing.assert_array_equal(audit.channel_design(changed, "context")[1, 0], context[1, 0])


@pytest.mark.parametrize("value", [np.nan, np.inf, 1j, "1", True])
def test_nonfinite_and_nonreal_input_rejected(value):
    with pytest.raises(ValueError):
        audit.finite([value], "test")


def test_invalid_geometries_and_alpha(plan):
    with pytest.raises(ValueError):
        audit.paired_stats([1])
    with pytest.raises(ValueError):
        audit.accuracy_arrays(np.zeros((1, 2)), np.zeros(1), plan)
    with pytest.raises(ValueError):
        audit.cost_arrays(np.zeros((2, 2, 9)), np.zeros(3), plan)
    with pytest.raises(ValueError):
        audit.independent_ridge([[1], [2]], [1, 2], -1)


def augmented_record(x, y, alpha=0.1):
    x = x.reshape(-1, x.shape[-1])
    y = y.ravel()
    mean, scale = x.mean(0), x.std(0)
    scale[scale < 1e-12] = 1
    z = (x - mean) / scale
    augmented = np.vstack((z / np.sqrt(len(x)), np.sqrt(alpha) * np.eye(x.shape[-1])))
    response = np.r_[(y - y.mean()) / np.sqrt(len(x)), np.zeros(x.shape[-1])]
    return {
        "mean": mean.tolist(),
        "scale": scale.tolist(),
        "coefficient": np.linalg.lstsq(augmented, response, rcond=None)[0].tolist(),
        "intercept": float(y.mean()),
        "alpha": alpha,
        "design_rank": int(np.linalg.matrix_rank(z / np.sqrt(len(x)))),
    }


def selected_fixture(q, y, ids):
    selected = {
        "representation": "local",
        "feature_count": q.shape[-1],
        "ridge": augmented_record(q, y),
    }
    assigned = np.arange(len(ids)) % 3
    return {
        "participant_ids": ids.tolist(),
        "splits": [
            {
                "fold": f,
                "fit_ids": ids[assigned != f].tolist(),
                "validation_ids": ids[assigned == f].tolist(),
            }
            for f in range(3)
        ],
        "candidates": [
            {
                "representation": rep,
                "alpha": alpha,
                "participant_mse": [float(0.1 if (rep, alpha) == ("local", 0.1) else 1)] * len(ids),
                "mean_participant_mse": float(0.1 if (rep, alpha) == ("local", 0.1) else 1),
            }
            for rep in ("local", "context")
            for alpha in (1, 0.1, 0.01, 0.001, 0.0001, 0)
        ],
        "selected": selected,
    }


@pytest.fixture(scope="module")
def artificial_bundle():
    plan = json.loads((ROOT / "configs/analysis/metadata_prior_source39_v1.json").read_text())
    plan["source_subject_ids"] = list(range(10, 19))
    plan["sample_counts"] = [125]
    rng = np.random.default_rng(826031)
    q = rng.normal(size=(9, 2, 10, 8, 15))
    z = rng.uniform(0, 40, size=(9, 2, 10, 8))
    order = np.zeros(9, dtype=int)
    target = 0.2 * q[..., 0] - 0.1 * q[..., 1] + rng.normal(0, 0.03, size=q.shape[:-1])
    for bi, k in enumerate(plan["budgets"]):
        q[:, bi, ..., 3] = np.log(k)
        q[:, bi, ..., 4] = 1
        q[:, bi, ..., 5] = 0
        q[:, bi, :5, :, 6] = 0
        q[:, bi, 5:, :, 6] = 1
        q[:, bi, ..., 7:] = np.eye(8)
    features = {"q": q, "target": target, "z": z, "order": order}
    ids = np.asarray(plan["source_subject_ids"])
    folds = []
    for f in range(3):
        fit, ev = np.flatnonzero(np.arange(9) % 3 != f), np.flatnonzero(np.arange(9) % 3 == f)
        train_maps = np.tile(np.roll(fit, -1), (2, 2, 1))
        eval_maps = np.tile(np.roll(ev, -1), (2, 2, 1))
        cells = []
        for ci in range(10):
            x, y = q[fit, :, ci : ci + 1], target[fit, :, ci : ci + 1]
            assignment = np.arange(len(fit)) % 3
            oof = np.empty_like(y)
            oof_folds = []
            for ff in range(3):
                train, held = assignment != ff, assignment == ff
                inner = selected_fixture(x[train], y[train], ids[fit][train])
                oof[held] = audit.predict_record(inner["selected"]["ridge"], x[held])
                oof_folds.append(
                    {
                        "fold": ff,
                        "fit_ids": ids[fit][train].tolist(),
                        "oof_ids": ids[fit][held].tolist(),
                        "inner_selection": inner,
                    }
                )
            final = selected_fixture(x, y, ids[fit])
            residual = y - oof
            interface = ci // 5
            m = np.stack(
                [audit.metadata_features(z[fit, interface, :k])[0] for k in (3, 5)], axis=1
            )[:, :, None]
            sham = np.stack(
                [
                    audit.metadata_features(z[train_maps[interface, bi], interface, :k])[0]
                    for bi, k in enumerate((3, 5))
                ],
                axis=1,
            )[:, :, None]
            receipt = {
                "oof_folds": oof_folds,
                "final_selection": final,
                "training_size_mismatch": {
                    "oof_fit_participants": [4, 4, 4],
                    "final_fit_participants": 6,
                },
            }
            cells.append(
                {
                    "cell": ci,
                    "Q": final["selected"],
                    "oof_prediction": oof.tolist(),
                    "q_receipt": receipt,
                    "residuals": {
                        "Q2": augmented_record(x, residual),
                        "QM": augmented_record(m, residual),
                        "SHAM_REFIT": augmented_record(sham, residual),
                    },
                }
            )
        folds.append(
            {
                "fold": f,
                "fit_ids": ids[fit].tolist(),
                "eval_ids": ids[ev].tolist(),
                "train_maps": train_maps.tolist(),
                "eval_maps": eval_maps.tolist(),
                "cells": cells,
            }
        )
    freeze = {
        "schema": "cfeg.metadata-prior-source.freezes.v1",
        "study_id": plan["study_id"],
        "plan_sha256": audit.PLAN_SHA256,
        "folds": folds,
    }
    return plan, features, freeze


def test_selected_oof_residual_and_prior_numeric_replay(artificial_bundle):
    plan, features, freeze = artificial_bundle
    result = audit.verify_freezes(features, freeze, plan)
    assert result["selected_fit_replays"] == 3 * 10 * 7
    assert result["coverage_changed_fraction"] == 1
    np.testing.assert_allclose(result["priors"].sum(-2), 8, atol=1e-12)
    np.testing.assert_array_equal(
        result["priors"][..., audit.ARMS.index("MISSING")],
        result["priors"][..., audit.ARMS.index("Q")],
    )
    assert np.isnan(result["proxy"][..., :2]).all()


@pytest.mark.parametrize(
    "corruption",
    [
        "subject",
        "donor",
        "inner_leak",
        "coefficient",
        "oof",
        "selected",
        "candidate_loss",
        "common_mask",
        "period",
        "schema",
    ],
)
def test_fit_and_feature_corruption_detected(artificial_bundle, corruption):
    plan, original_features, original_freeze = artificial_bundle
    features, freeze = copy.deepcopy(original_features), copy.deepcopy(original_freeze)
    cell = freeze["folds"][0]["cells"][0]
    if corruption == "subject":
        freeze["folds"][0]["fit_ids"][0] = 10
    elif corruption == "donor":
        freeze["folds"][0]["eval_maps"][0][0][0] = 1
    elif corruption == "inner_leak":
        cell["q_receipt"]["oof_folds"][0]["inner_selection"]["splits"][0]["fit_ids"].append(10)
    elif corruption == "coefficient":
        cell["residuals"]["QM"]["coefficient"][0] += 0.05
    elif corruption == "oof":
        cell["oof_prediction"][0][0][0][0] += 0.05
    elif corruption == "selected":
        cell["q_receipt"]["final_selection"]["selected"]["ridge"]["alpha"] = 1
    elif corruption == "candidate_loss":
        cell["q_receipt"]["final_selection"]["candidates"][0]["mean_participant_mse"] = 0.25
    elif corruption == "common_mask":
        features["q"][0, 0, 0, 0, 4] = 0
    elif corruption == "period":
        features["q"][0, 0, 0, 0, 6] = 1
    else:
        freeze["schema"] += ".wrong"
    with pytest.raises(ValueError):
        audit.verify_freezes(features, freeze, plan)


def test_whole_prefix_mask_donor_strata_and_valid_zero(plan):
    z = np.ones((3, 2, 10, 8))
    z[0, 0, 0, 0] = 0
    z[2, 0, 1, 0] = np.nan
    donor = audit.donor_map(z, [0, 0, 0], [0, 1, 2], 0, 3, plan["source_subject_ids"])
    assert donor.tolist() == [1, 0, 2]
    mf, mask = audit.metadata_features(z[:, 0, :3])
    assert mask.all()
    assert np.isfinite(mf).all()
    assert mf[0, 0, 1] > 0


@pytest.fixture(scope="module")
def artificial_result(artificial_bundle):
    plan, features, freeze = artificial_bundle
    replay = audit.verify_freezes(features, freeze, plan)
    scores = np.zeros((9, 2, 1, 2, 9, 48, 12))
    anchors = np.zeros((9, 2, 1, 48, 12))
    for query in range(48):
        anchors[..., query, query % 12 if query < 24 else (query + 1) % 12] = 1
        for bi in range(2):
            for ai, arm in enumerate(audit.ARMS):
                correct = 42 if bi else 36
                if arm == "QM":
                    correct = 44 if bi else 42
                scores[
                    :, :, :, bi, ai, query, query % 12 if query < correct else (query + 1) % 12
                ] = 1
    accuracy, a0, predicted, a0pred = audit.accuracy_arrays(scores, anchors, plan)
    rows = []
    for pi, participant in enumerate(plan["source_subject_ids"]):
        for ei, interface in enumerate(plan["interfaces"]):
            common = {"participant": participant, "interface": interface, "n_samples": 125}
            rows.append(
                {
                    **common,
                    "k": 0,
                    "arm": "A0",
                    "accuracy": float(a0[pi, ei, 0]),
                    "prior": None,
                    "proxy_components": None,
                    "prediction_changes_vs_Q": None,
                    "diagnostics": {},
                }
            )
            for bi, k in enumerate((3, 5)):
                which = slice(ei * 5, (ei + 1) * 5)
                for ai, arm in enumerate(audit.ARMS):
                    error = (
                        replay["proxy"][pi, bi, which, :, ai] - features["target"][pi, bi, which]
                    )
                    level = error.mean(-1, keepdims=True)
                    pieces = (
                        None
                        if arm in ("FULL", "ISO")
                        else {
                            "raw_mse": float((error**2).mean()),
                            "level_mse": float((level**2).mean()),
                            "shape_mse": float(((error - level) ** 2).mean()),
                        }
                    )
                    rows.append(
                        {
                            **common,
                            "k": k,
                            "arm": arm,
                            "accuracy": float(accuracy[pi, ei, 0, bi, ai]),
                            "prior": replay["priors"][pi, bi, which, :, ai].tolist(),
                            "proxy_components": pieces,
                            "prediction_changes_vs_Q": int(
                                np.count_nonzero(
                                    predicted[pi, ei, 0, bi, ai]
                                    != predicted[pi, ei, 0, bi, audit.ARMS.index("Q")]
                                )
                            ),
                            "diagnostics": {},
                        }
                    )
    summary, contrasts, attainment, calibration, verdict = audit.expected_statistics(
        accuracy, a0, predicted, a0pred, plan, replay["coverage_changed_fraction"]
    )
    for entry in summary:
        group = [
            r for r in rows if all(r[k] == entry[k] for k in ("interface", "n_samples", "k", "arm"))
        ]
        for part in ("raw", "level", "shape"):
            entry[f"proxy_{part}_mse"] = (
                None
                if group[0]["proxy_components"] is None
                else float(np.mean([r["proxy_components"][f"{part}_mse"] for r in group]))
            )
    result = {
        "schema": "cfeg.metadata-prior-source.result.v1",
        "study_id": plan["study_id"],
        "rows": rows,
        "summary": summary,
        "contrasts": contrasts,
        "attainment": attainment,
        "calibration": calibration,
        "verdict": verdict,
        "compatibility": {"max_native_correlation_error": 1e-12, "native_prediction_mismatches": 0},
    }
    return {"scores": scores, "a0_scores": anchors}, result


def test_complete_artificial_saved_result_audit(artificial_bundle, artificial_result):
    plan, features, freeze = artificial_bundle
    scores, result = artificial_result
    report = audit.verify_results(scores, features, freeze, result, plan)
    assert report["rows_checked"] == 9 * 2 * 19
    assert report["summary_checked"] == 2 * 19
    assert report["contrasts_checked"] == 3 * 11
    assert report["attainment_checked"] == 9 * 2 * 9
    assert report["verdict"]["status"] == "DEVELOPMENT_CALIBRATION_BENEFIT_CANDIDATE"
    primary = next(
        r
        for r in result["contrasts"]
        if r["scope"] == "ALL" and r["k"] == 3 and r["comparator"] == "Q"
    )
    assert primary["mean_delta"] == primary["ci_low"] == primary["ci_high"] == 0.125
    assert result["calibration"]["both_reached"] == 18
    assert result["calibration"]["mean_label_delta_both"] == -24
    assert result["calibration"]["transitions"] == {"5->3": 18}


@pytest.mark.parametrize(
    "corruption",
    [
        "row_accuracy",
        "prior",
        "proxy",
        "summary",
        "contrast",
        "attainment",
        "calibration",
        "verdict",
        "compatibility",
        "duplicate",
        "missing_score",
        "diagnostic",
    ],
)
def test_saved_result_corruption_is_rejected(artificial_bundle, artificial_result, corruption):
    plan, features, freeze = artificial_bundle
    original_scores, original_result = artificial_result
    scores, result = copy.deepcopy(original_scores), copy.deepcopy(original_result)
    row = next(r for r in result["rows"] if r["arm"] == "QM")
    if corruption == "row_accuracy":
        row["accuracy"] += 0.01
    elif corruption == "prior":
        row["prior"][0][0] += 0.01
    elif corruption == "proxy":
        row["proxy_components"]["shape_mse"] += 0.01
    elif corruption == "summary":
        result["summary"][0]["accuracy"] += 0.01
    elif corruption == "contrast":
        result["contrasts"][0]["ci_low"] += 0.01
    elif corruption == "attainment":
        result["attainment"][0]["labels"] = 24
    elif corruption == "calibration":
        result["calibration"]["mean_label_delta_both"] = 0
    elif corruption == "verdict":
        result["verdict"]["metadata_increment"] = False
    elif corruption == "compatibility":
        result["compatibility"]["native_prediction_mismatches"] = 1
    elif corruption == "duplicate":
        result["rows"].append(copy.deepcopy(result["rows"][0]))
    elif corruption == "diagnostic":
        row["diagnostics"]["max_score_change"] = 999
    else:
        scores["scores"][0, 0, 0, 0, audit.ARMS.index("MISSING")] += 0.01
    with pytest.raises(ValueError):
        audit.verify_results(scores, features, freeze, result, plan)


def test_no_coverage_blocks_increment_and_no_finite_pairs_means_no_fake_cost(plan):
    accuracy = np.full((3, 2, 1, 2, 9), 0.5)
    accuracy[..., 0, audit.ARMS.index("QM")] = 0.75
    anchors = np.full((3, 2, 1), 0.25)
    predictions = np.zeros((3, 2, 1, 2, 9, 48), dtype=int)
    a0pred = np.zeros((3, 2, 1, 48), dtype=int)
    *_, calibration, verdict = audit.expected_statistics(
        accuracy, anchors, predictions, a0pred, plan, 1
    )
    assert verdict["status"] == "CLASSIFICATION_INCREMENT_ONLY"
    assert calibration["mean_label_delta_both"] is None
    assert calibration["neither_reached"] == 6
    *_, blocked = audit.expected_statistics(accuracy, anchors, predictions, a0pred, plan, 0.49)
    assert blocked["status"] == "METADATA_INCREMENT_NOT_ESTABLISHED"


def projection_fixture(plan):
    spec = plan["source_projection"]
    value = {
        **spec["envelope_provenance"],
        "manifest_sha256": spec["manifest_sha256"],
        "returned_rows": spec["returned_rows"],
        "returned_packets": len(plan["source_subject_ids"]) * 20,
        "returned_subject_ids": plan["source_subject_ids"],
        "columns": spec["columns"],
        "packets": [],
    }
    for pi, subject in enumerate(plan["source_subject_ids"]):
        first = plan["interfaces"][pi % 2]
        for interface in plan["interfaces"]:
            for block in range(10):
                value["packets"].append(
                    {
                        "subject_id": subject,
                        "interface": interface,
                        "block_id": block,
                        "impedance_kohm": [0, None, 1, 2, 3, 4, 5, 6],
                        "headband_order": first,
                        "condition_period": "first" if interface == first else "second",
                    }
                )
    return value


def test_projection_envelope_exact_values_and_missingness(plan):
    z, order = audit.projection_arrays(projection_fixture(plan), plan)
    assert z.shape == (3, 2, 10, 8)
    assert order.tolist() == [0, 1, 0]
    assert np.all(z[..., 0] == 0)
    assert np.isnan(z[..., 1]).all()


@pytest.mark.parametrize(
    "corruption", ["extra", "provenance", "subject", "duplicate", "period", "bool", "negative"]
)
def test_projection_corruption_rejected(plan, corruption):
    value = projection_fixture(plan)
    if corruption == "extra":
        value["held"] = True
    elif corruption == "provenance":
        value["source_commit"] = "0" * 40
    elif corruption == "subject":
        value["packets"][0]["subject_id"] = 1
    elif corruption == "duplicate":
        value["packets"][1] = copy.deepcopy(value["packets"][0])
    elif corruption == "period":
        value["packets"][0]["condition_period"] = "second"
    elif corruption == "bool":
        value["packets"][0]["impedance_kohm"][0] = True
    else:
        value["packets"][0]["impedance_kohm"][0] = -1
    with pytest.raises(ValueError):
        audit.projection_arrays(value, plan)


@pytest.fixture
def provenance_fixture(tmp_path, plan):
    root = tmp_path / "source"
    sources = [
        "src/cfeg/analysis/metadata_prior_source.py",
        "scripts/run_metadata_prior_source.py",
        "scripts/export_metadata_prior_source.py",
        plan["operator"]["path"],
        plan["q_helper"]["path"],
    ]
    source_hashes = {}
    for name in sources:
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(name.encode())
        source_hashes[name] = hashlib.sha256(name.encode()).hexdigest()
    plan = copy.deepcopy(plan)
    for role in ("operator", "q_helper"):
        plan[role]["sha256"] = source_hashes[plan[role]["path"]]
    common = {
        "plan_sha256": audit.PLAN_SHA256,
        "source_commit": "a" * 40,
        "source_hashes": source_hashes,
        "python": "3.10.12",
        "numpy": "1.26.4",
        "scipy": "1.15.3",
    }
    start = {
        "schema": "cfeg.metadata-prior-source.start.v1",
        "study_id": plan["study_id"],
        **common,
    }
    hashes = {
        name: hashlib.sha256(name.encode()).hexdigest()
        for name in (
            "start.json",
            "features.npz",
            "fold-freezes.json",
            "scores.npz",
            "native/start.json",
            "native/result.json",
            "source_projection",
        )
    }
    hashes["source_projection"] = plan["source_projection"]["sha256"]
    inputs = {
        "start_sha256": hashes["start.json"],
        "features_sha256": hashes["features.npz"],
        "freezes_sha256": hashes["fold-freezes.json"],
        "scores_sha256": hashes["scores.npz"],
        "native_manifest_sha256": hashes["native/result.json"],
        "projection_sha256": hashes["source_projection"],
    }
    result = {"provenance": {**common, **inputs}}
    freeze = {"plan_sha256": audit.PLAN_SHA256}
    native_common = {
        **common,
        "study_id": plan["study_id"],
        "source_subject_ids": plan["source_subject_ids"],
        "sample_counts": plan["sample_counts"],
    }
    native_start = {**native_common, "schema": "cfeg.metadata-prior-source.native-start.v1"}
    native_manifest = {
        **native_common,
        "schema": "cfeg.metadata-prior-source.native-result.v1",
        "status": "COMPLETE",
        "start_sha256": hashes["native/start.json"],
        "files": [
            {
                "subject": subject,
                "filename": f"S{subject:03d}.npz",
                "bytes": 123,
                "sha256": "f" * 64,
            }
            for subject in plan["source_subject_ids"]
        ],
        "total_bytes": len(plan["source_subject_ids"]) * 123,
    }
    return start, result, freeze, native_start, native_manifest, hashes, plan, root


def test_start_input_and_source_hash_chains(provenance_fixture):
    audit.verify_provenance(*provenance_fixture)


@pytest.mark.parametrize(
    "corruption",
    [
        "start_hash",
        "native_hash",
        "features_hash",
        "source_hash",
        "source_escape",
        "incomplete_native",
        "cache_escape",
        "byte_total",
    ],
)
def test_provenance_corruption_rejected(provenance_fixture, corruption):
    args = copy.deepcopy(provenance_fixture)
    start, result, _, _, manifest, _, _, _ = args
    if corruption == "start_hash":
        result["provenance"]["start_sha256"] = "0" * 64
    elif corruption == "native_hash":
        manifest["start_sha256"] = "0" * 64
    elif corruption == "features_hash":
        result["provenance"]["features_sha256"] = "0" * 64
    elif corruption == "source_hash":
        start["source_hashes"]["src/cfeg/analysis/metadata_prior_source.py"] = "0" * 64
    elif corruption == "source_escape":
        start["source_hashes"]["../../credentials"] = "0" * 64
    elif corruption == "incomplete_native":
        manifest["status"] = "STARTED"
    elif corruption == "cache_escape":
        manifest["files"][0]["filename"] = "../../somewhere.npz"
    else:
        manifest["total_bytes"] += 1
    with pytest.raises(ValueError):
        audit.verify_provenance(*args)


def test_exclusive_publication_regular_inputs_and_hash_rejection(tmp_path):
    path = tmp_path / "audit.json"
    expected = audit.publish_exclusive(path, {"status": "PASS"})
    value, actual = audit.read_document(path, expected)
    assert value == {"status": "PASS"} and actual == expected
    assert path.stat().st_mode & 0o777 == 0o400
    with pytest.raises(FileExistsError):
        audit.publish_exclusive(path, {"status": "CHANGED"})
    with pytest.raises(ValueError):
        audit.read_document(path, "0" * 64)
    link = tmp_path / "alias.json"
    link.symlink_to(path)
    with pytest.raises(ValueError, match="symlink"):
        audit.regular_bytes(link)
    assert audit.read_document(path)[0] == {"status": "PASS"}


def test_cold_cli_help_reads_no_study_inputs():
    completed = subprocess.run(
        [sys.executable, str(ROOT / "scripts/audit_metadata_prior_source.py"), "--help"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    assert "--analysis-root" in completed.stdout


def test_existing_audit_stops_before_study_inputs(tmp_path, plan, monkeypatch):
    directory = tmp_path / "analysis"
    directory.mkdir()
    previous = directory / "audit.json"
    previous.write_text("preserved")
    plan["execution"]["analysis_root"] = str(directory)
    plan_path = tmp_path / "plan.json"
    plan_path.write_text(json.dumps(plan))
    monkeypatch.setattr(audit, "PLAN_SHA256", hashlib.sha256(plan_path.read_bytes()).hexdigest())
    with pytest.raises(ValueError, match="already exists"):
        audit.run(directory, plan_path)
    assert previous.read_text() == "preserved"
