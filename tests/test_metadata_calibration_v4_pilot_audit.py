from __future__ import annotations

import copy
import importlib.util
import json
from pathlib import Path

import pytest

from cfeg.analysis import metadata_calibration_v4_pilot as pilot


@pytest.fixture(scope="module")
def auditor():
    path = Path(__file__).resolve().parents[1] / "scripts/audit_metadata_calibration_v4_pilot.py"
    spec = importlib.util.spec_from_file_location("v4_independent_audit", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def fixture_result():
    root = Path(__file__).resolve().parents[1]
    plan = json.loads((root / "configs/analysis/metadata_calibration_v4_pilot.json").read_text())
    plan.update(participants=4, n_classes=3, n_channels=4, n_samples=96, query_blocks=2)
    plan["dgp"]["frequency_step"] = 2.0
    filterbank = {
        "bands": [[6, 45]],
        "weights": [1.0],
        "order": 2,
        "n_harmonics": 3,
        "regularization": 1e-8,
        "filter_family": "butterworth",
    }
    pilot.initialize_workspace(plan, filterbank)
    # Independent fixture seed, never the SHA-derived study namespace.
    participants = [pilot.evaluate_participant(p, 672023) for p in range(4)]
    return plan, {
        "participant_results": participants,
        "summary": pilot.summarize_results(participants, plan),
    }


def test_independent_audit_accepts_fixture_roundtrip(auditor, fixture_result):
    plan, result = fixture_result
    report = auditor.audit_result(plan, json.loads(json.dumps(result)))
    assert report["audit"] == "PASS"
    assert report["metric_rows"] == 4 * 84
    assert report["prediction_records_checked"] == 4 * 72
    assert report["permutation_metric_records_checked"] == 4 * 3 * 2 * (2 + 44)


@pytest.mark.parametrize("mutation", ["count", "interval", "screen", "prediction", "permutation"])
def test_independent_audit_rejects_corruption(auditor, fixture_result, mutation):
    plan, original = fixture_result
    result = copy.deepcopy(original)
    if mutation == "count":
        result["summary"]["participant_metric_rows"][0]["correct_count"] += 1
    elif mutation == "interval":
        result["summary"]["comparisons"]["stable_AQ_minus_A0_eauc"]["one_sided_95_LCB"] += 0.1
    elif mutation == "screen":
        result["summary"]["human_unlock"] = True
    elif mutation == "prediction":
        predictions = result["participant_results"][0]["prediction_evidence"][0]["predictions"]
        predictions[0] = (predictions[0] + 1) % plan["n_classes"]
    else:
        result["participant_results"][0]["derangement_evidence"][0]["permutations"].pop()
    with pytest.raises(ValueError):
        auditor.audit_result(plan, result)
