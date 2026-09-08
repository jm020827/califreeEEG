"""Pure count/sign/correction fixtures only; never open immutable source39 artifacts."""

from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "source_exact_counts", ROOT / "scripts/audit_metadata_prior_source_counts.py"
)
audit = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(audit)


@pytest.fixture
def plan():
    value = json.loads((ROOT / "configs/analysis/metadata_prior_source39_v1.json").read_text())
    value["source_subject_ids"] = [4, 6, 8]
    value["sample_counts"] = [125]
    return value


def test_integer_cancellation_does_not_become_a_tiny_help():
    difference = (10 / 48 - 9 / 48) + (2 / 48 - 3 / 48)
    assert difference > 0
    result = audit.paired_counts(np.array([0, 1, -1]), 96)
    assert result["participant_net_correct"] == [0, 1, -1]
    assert result["help"] == result["tie"] == result["harm"] == 1
    assert result["mean_delta"] == 0
    assert result["ci_low"] < 0 < result["ci_high"]


def test_known_student_interval_and_exact_zero_variance():
    result = audit.paired_counts(np.array([-3, -1, 1, 3]), 100)
    radius = 3.182446305284263 * np.sqrt(0.002 / 3) / 2
    assert result["ci_low"] == pytest.approx(-radius)
    assert result["ci_high"] == pytest.approx(radius)
    equal = audit.paired_counts(np.array([1, 1, 1]), 48)
    assert equal["mean_delta"] == equal["ci_low"] == equal["ci_high"] == 1 / 48
    assert equal["help"] == 3 and equal["tie"] == equal["harm"] == 0


def test_all_scope_sums_integer_cell_counts_before_dividing(plan):
    counts = np.full((3, 2, 1, 2, 9), 3, dtype=int)
    anchors = np.zeros((3, 2, 1), dtype=int)
    qi, mi = audit.ARMS.index("Q"), audit.ARMS.index("QM")
    counts[0, 0, 0, :, qi] = 9
    counts[0, 0, 0, :, mi] = 10
    counts[0, 1, 0, :, qi] = 3
    counts[0, 1, 0, :, mi] = 2
    counts[1, 0, 0, :, mi] = 4
    counts[2, 0, 0, :, mi] = 2
    result = audit.exact_contrasts(counts, anchors, plan)
    selected = next(x for x in result if (x["scope"], x["k"], x["comparator"]) == ("ALL", 5, "Q"))
    assert selected["participant_net_correct"] == [0, 1, -1]
    assert selected["per_participant_denominator"] == 96
    assert (selected["help"], selected["tie"], selected["harm"]) == (1, 1, 1)


def test_full_portfolio_has_99_contrasts_and_correct_denominators():
    plan = json.loads((ROOT / "configs/analysis/metadata_prior_source39_v1.json").read_text())
    counts = np.zeros((39, 2, 4, 2, 9), dtype=np.int64)
    anchors = np.zeros((39, 2, 4), dtype=np.int64)
    result = audit.exact_contrasts(counts, anchors, plan)
    assert len(result) == 99
    assert sum(x["scope"] == "ALL" for x in result) == 11
    assert all(
        x["per_participant_denominator"] == (384 if x["scope"] == "ALL" else 48) for x in result
    )
    assert all(x["tie"] == 39 and x["help"] == x["harm"] == 0 for x in result)
    assert all(x["k"] == 3 for x in result if x["comparator"] == "Q5")


@pytest.fixture
def score_fixture(plan):
    scores = np.zeros((3, 2, 1, 2, 9, 48, 12), dtype=float)
    anchors = np.zeros((3, 2, 1, 48, 12), dtype=float)
    for query in range(48):
        scores[..., query, query % 12] = 1
        anchors[..., query, query % 12] = 1
    counts, a0 = audit.classify_counts(scores, anchors, plan)
    contrasts = audit.exact_contrasts(counts, a0, plan)
    calibration = audit.calibration_counts(counts, a0, plan)
    verdict = audit.corrected_verdict(contrasts, counts, calibration, 1.0, plan)
    result = {
        "schema": "cfeg.metadata-prior-source.result.v1",
        "study_id": plan["study_id"],
        "contrasts": copy.deepcopy(contrasts),
        "calibration": calibration,
        "verdict": verdict,
    }
    prior = {"status": "PASS", "study_id": plan["study_id"], "verdict": copy.deepcopy(verdict)}
    return scores, anchors, result, prior


def test_reporting_correction_preserves_means_calibration_and_verdict(plan, score_fixture):
    scores, anchors, result, prior = score_fixture
    row = next(
        x for x in result["contrasts"] if (x["scope"], x["k"], x["comparator"]) == ("ALL", 5, "Q")
    )
    row.update(help=1, tie=1, harm=1, mean_delta=1e-19)
    report = audit.compare_reporting(scores, anchors, result, prior, plan)
    assert report["status"] == "REPORTING_CORRECTION_VERIFIED"
    assert report["changed_help_tie_harm_count"] == 1
    assert report["changed_contrasts"][0]["corrected"] == {"help": 0, "tie": 3, "harm": 0}
    assert report["max_mean_or_ci_absolute_change"] == 1e-19
    assert report["verdict"] == result["verdict"]
    assert report["calibration"]["mean_label_delta_both"] == 0
    assert row["help"] == 1, "Original result must not be mutated"


@pytest.mark.parametrize(
    "corruption", ["mean", "ci", "duplicate", "verdict", "calibration", "prior_audit", "n"]
)
def test_material_result_change_is_not_accepted_as_roundoff(plan, score_fixture, corruption):
    scores, anchors, result, prior = score_fixture
    if corruption == "mean":
        result["contrasts"][0]["mean_delta"] += 0.01
    elif corruption == "ci":
        result["contrasts"][0]["ci_low"] += 0.01
    elif corruption == "duplicate":
        result["contrasts"].append(copy.deepcopy(result["contrasts"][0]))
    elif corruption == "verdict":
        result["verdict"]["metadata_increment"] = True
    elif corruption == "calibration":
        result["calibration"]["both_reached"] -= 1
    elif corruption == "n":
        result["contrasts"][0]["n"] += 1
    else:
        prior["status"] = "FAIL"
    with pytest.raises(ValueError):
        audit.compare_reporting(scores, anchors, result, prior, plan)


def test_class_ties_and_invalid_score_geometry(plan, score_fixture):
    scores, anchors, _, _ = score_fixture
    scores[..., 1, :] = 0
    counts, _ = audit.classify_counts(scores, anchors, plan)
    assert np.all(counts == 47)
    with pytest.raises(ValueError):
        audit.classify_counts(scores[0], anchors, plan)
    scores.flat[0] = np.nan
    with pytest.raises(ValueError):
        audit.classify_counts(scores, anchors, plan)


def test_first_observed_reachability_keeps_unreached_missing(plan):
    counts = np.zeros((3, 2, 1, 2, 9), dtype=int)
    anchors = np.zeros((3, 2, 1), dtype=int)
    qi, mi = audit.ARMS.index("Q"), audit.ARMS.index("QM")
    counts[0, 0, 0, 1, qi] = 39
    counts[0, 0, 0, 0, mi] = 39
    counts[1, 0, 0, 0, mi] = 39
    counts[2, 0, 0, 1, qi] = 39
    result = audit.calibration_counts(counts, anchors, plan)
    assert result == {
        "both_reached": 1,
        "new_reach": 1,
        "lost_reach": 1,
        "neither_reached": 3,
        "mean_label_delta_both": -24.0,
    }
    empty = audit.calibration_counts(np.zeros_like(counts), anchors, plan)
    assert empty["mean_label_delta_both"] is None


@pytest.mark.parametrize(
    "net,denominator",
    [
        ([0.0, 1.0], 48),
        ([1], 48),
        ([0, 49], 48),
        ([0, -49], 48),
        ([0, np.iinfo(np.int64).min], 48),
        ([0, 1], 0),
        ([0, 1], 48.0),
    ],
)
def test_invalid_net_counts_rejected(net, denominator):
    with pytest.raises(ValueError):
        audit.paired_counts(np.asarray(net), denominator)


def test_exact_input_hash_and_exclusive_receipt(tmp_path):
    path = tmp_path / "exact-counts.json"
    digest = audit.publish(path, {"status": "fixture"})
    data, actual = audit.read_regular(path, digest)
    assert actual == hashlib.sha256(data).hexdigest()
    assert path.stat().st_mode & 0o777 == 0o400
    with pytest.raises(FileExistsError):
        audit.publish(path, {"changed": True})
    with pytest.raises(ValueError):
        audit.read_regular(path, "0" * 64)
    linked = tmp_path / "link.json"
    linked.symlink_to(path)
    with pytest.raises(ValueError):
        audit.read_regular(linked)
    assert json.loads(path.read_text()) == {"status": "fixture"}


def test_all_four_declared_input_pins_are_explicit():
    assert audit.RESULT_SHA256 == "f1d8158c0228cfefdf5bd762bdc1d960f7268f79f93c24268a519e46d6460ce6"
    assert audit.AUDIT_SHA256 == "ce5e355674f940a6d256226c573df0727780676e927ccbbe3e0626e124616be8"
    assert all(
        len(value) == 64
        for value in (
            audit.PLAN_SHA256,
            audit.RESULT_SHA256,
            audit.SCORES_SHA256,
            audit.AUDIT_SHA256,
        )
    )
    assert (
        hashlib.sha256(
            (ROOT / "configs/analysis/metadata_prior_source39_v1.json").read_bytes()
        ).hexdigest()
        == audit.PLAN_SHA256
    )
