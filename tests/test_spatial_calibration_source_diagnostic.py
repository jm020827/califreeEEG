"""Saved-score diagnostics do not establish a raw-signal mechanism."""

import importlib.util
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "spatial_diagnostic_fixture", ROOT / "scripts/diagnose_spatial_calibration_source.py"
)
diagnostic = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(diagnostic)


def test_high_correlation_can_have_no_label_information():
    labels = np.tile(np.arange(12), 5)
    scores = np.full((3, 1, 1, 60, 12), 0.9)
    cache = {
        "query_labels": labels,
        "cell_ids": np.array(["fixture"]),
        "calibration_budgets": np.array([5]),
        "a0_scores": np.zeros((3, 1, 60, 12)),
        "itcca_scores": scores,
    }
    fits = {"folds": [{"windows": {"250": {"budgets": {"5": {"fusion_fit": {"lambda": 0}}}}}}]}
    result = {"summary": {"aggregate_rows": []}}
    report = diagnostic.diagnose(cache, fits, result)
    row = report["ITCCA_score_diagnostics"][0]
    assert row["mean_correct_template_score"] == pytest.approx(0.9)
    assert row["mean_wrong_template_score"] == pytest.approx(0.9)
    assert row["median_participant_BA"] == pytest.approx(1 / 12)
    assert row["maximum_participant_modal_fraction"] == 1
    assert row["mean_correct_minus_max_wrong_score"] == 0
    assert report["zero_lambda_fits"] == 1


def test_lineage_exact_and_score_change_detected():
    cache = {
        key: np.array([1])
        for key in (
            "subject_ids",
            "cell_ids",
            "raw_file_sha256",
            "crop_sha256",
            "a0_scores",
            "legacy_etrca_scores",
        )
    }
    previous = {key: cache[key].copy() for key in cache if key != "legacy_etrca_scores"}
    previous["etrca_scores"] = cache["legacy_etrca_scores"].copy()
    assert all(diagnostic.lineage(cache, previous).values())
    previous["a0_scores"][0] = 2
    assert not diagnostic.lineage(cache, previous)["a0_scores"]
