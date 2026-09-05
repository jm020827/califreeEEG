from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import pandas as pd
import pytest

from cfeg.analysis.metadata_calibration_baselines import (
    summarize_baseline_participant_accuracy,
)
from cfeg.analysis.metadata_calibration_efficiency import (
    derive_participant_contrasts,
    evaluate_development_gate,
    metadata_calibration_bundle_spec_for_phase,
)
from cfeg.metadata_calibration_authority import canonical_json_bytes, canonical_json_sha256
from cfeg.metadata_calibration_contract import METADATA_CALIBRATION_SOURCE_RECIPE
from cfeg.metadata_calibration_sealer import validate_metadata_calibration_result_bundle
from cfeg.utils.config import load_config

REPO = Path(__file__).resolve().parents[1]


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, value: dict) -> None:
    path.write_bytes(canonical_json_bytes(value) + b"\n")


def _source_result_tree(tmp_path: Path) -> Path:
    root = tmp_path / "result"
    root.mkdir(mode=0o700)
    recipe = load_config(REPO / METADATA_CALIBRATION_SOURCE_RECIPE, strict_env=False)
    outer_fold_by_subject = {
        str(subject): fold
        for fold in (0, 1, 2)
        for subject in recipe["data"]["folds"][fold]["outer_evaluation"]
    }
    subjects = sorted(outer_fold_by_subject)
    candidate_cells = (
        ("A_0", "exact_off"),
        ("A_M", "correct"),
        ("A_Q", "exact_off"),
        ("A_QM", "all_missing"),
        ("A_QM", "block_shuffle"),
        ("A_QM", "correct"),
        ("A_QM", "impedance_only"),
        ("A_QM", "interface_only"),
        ("A_QM", "opposite_interface"),
        ("A_QM", "query_metadata_only"),
        ("A_QM", "stale"),
        ("A_QM", "support_metadata_only"),
    )
    candidate_ba = pd.DataFrame.from_records(
        {
            "role": role,
            "context": context,
            "budget": budget,
            "subject_id": subject,
            "participant_balanced_accuracy": 0.5,
        }
        for role, context in candidate_cells
        for budget in (0, 1, 3, 5)
        for subject in subjects
    )
    spec = metadata_calibration_bundle_spec_for_phase(phase="source_development")
    contrasts = derive_participant_contrasts(
        candidate_ba,
        spec=spec,
    )
    baseline_budgets = {
        "strict_FBCCA": (0,),
        "target_template_correlation": (1, 3, 5),
        "target_filterbank_eTRCA": (3, 5),
        "same3_filterbank_eTRCA": (1,),
        "chiang2021_LST_filterbank_eTRCA": (1, 3, 5),
    }
    baseline_ba = pd.DataFrame.from_records(
        {
            "adapter_id": adapter,
            "budget": budget,
            "subject_id": subject,
            "participant_balanced_accuracy": 0.5,
        }
        for adapter, budgets in baseline_budgets.items()
        for budget in budgets
        for subject in subjects
    )
    baseline_summary = summarize_baseline_participant_accuracy(baseline_ba)
    for name, frame in {
        "candidate_participant_balanced_accuracy.csv": candidate_ba,
        "candidate_participant_contrasts.csv": contrasts,
        "baseline_participant_balanced_accuracy.csv": baseline_ba,
        "baseline_summary.csv": baseline_summary,
    }.items():
        frame.to_csv(root / name, index=False)

    bindings = {
        "plan_sha256": "a" * 64,
        "execution_manifest_sha256": "b" * 64,
        "decision_receipt_sha256": "c" * 64,
        "asset_receipt_sha256": "d" * 64,
        "asset_fingerprint_bundle_sha256": "e" * 64,
        "class_map_sha256": "f" * 64,
        "source_tree_sha256": "1" * 64,
    }
    gate = evaluate_development_gate(
        contrasts,
        outer_fold_by_subject=outer_fold_by_subject,
        missing_fallback_probability_exact=True,
        metadata_only_structural_chance_exact=True,
        expected_n=39,
        spec=spec,
    )
    selected_epochs = {
        str(seed): {"stage1_epochs": 10, "stage2_epochs": 5}
        for seed in (42, 43, 44)
    }
    gate.update(
        {
            "held_selected_epochs_by_seed": selected_epochs,
            "held_selected_epochs_by_seed_sha256": canonical_json_sha256(selected_epochs),
            "held_epoch_selection_derivation_sha256": "b" * 64,
        }
    )
    _write_json(
        root / "analysis_metadata.json",
        {
            "schema": "cfeg.metadata-calibration-published-analysis-metadata.v1",
            "phase": "source_development",
            "candidate": {
                "schema": "cfeg.metadata-calibration-analysis-bundle.v1",
                "candidate_id": "metadata-calibration-efficiency-v1",
                "phase": "source_development",
                "bindings": bindings,
                "seed_reduction": "mean_probabilities_per_sample_before_argmax",
                "tie_break": "lowest_canonical_class_id",
                "missing_fallback_probability_exact": True,
            },
            "baseline": {
                "schema": "cfeg.metadata-calibration-baseline-analysis-bundle.v1",
                "candidate_id": "metadata-calibration-efficiency-v1",
                "phase": "source_development",
                "bindings": bindings,
                "score_interpretation": "uncalibrated_raw_class_score_argmax_only",
                "seed_reduction": "none_deterministic_once",
                "tie_break": "lowest_zero_based_class_id",
                "primary_inference_role": "descriptive_resource_matched_context_only",
            },
            "bindings": bindings,
            "query_level_outputs_published": False,
        },
    )
    _write_json(root / "gate_or_claim_result.json", gate)
    _write_json(
        root / "analysis_covariate_summary.json",
        {
            "schema": "cfeg.metadata-calibration-analysis-covariate-summary.v1",
            "n_query_rows": 39 * 2 * 60,
            "query_tokens_published": False,
            "headband_order_counts": {"dry": 20 * 120, "wet": 19 * 120},
            "condition_period_counts": {"first": 2340, "second": 2340},
            "block_number_counts": {
                str(block): 39 * 2 * 12 for block in range(6, 11)
            },
        },
    )
    _write_json(root / "completion_receipt.json", {"development_gate_result": gate})
    records = [
        {
            "relative_path": path.name,
            "size_bytes": path.stat().st_size,
            "file_sha256": _file_sha256(path),
        }
        for path in sorted(root.iterdir())
        if path.name != "completion_receipt.json"
    ]
    manifest = {
        "schema": "cfeg.metadata-calibration-result-bundle.v1",
        "candidate_id": "metadata-calibration-efficiency-v1",
        "phase": "source_development",
        "query_outcomes_loaded": True,
        "query_level_outputs_published": False,
        "completion_receipt_excluded_from_result_content_hash": True,
        "files": records,
    }
    manifest["result_bundle_sha256"] = canonical_json_sha256(manifest)
    _write_json(root / "result_bundle_manifest.json", manifest)
    for path in root.iterdir():
        os.chmod(path, 0o400)
    os.chmod(root, 0o700)
    return root


def test_result_bundle_requires_exact_aggregate_files_and_bound_gate(tmp_path: Path) -> None:
    root = _source_result_tree(tmp_path)
    observed = validate_metadata_calibration_result_bundle(
        root,
        completion_receipt_path=root / "completion_receipt.json",
        expected_phase="source_development",
    )
    manifest = json.loads((root / "result_bundle_manifest.json").read_text())
    assert observed == manifest["result_bundle_sha256"]


def test_result_bundle_rejects_declared_query_level_file(tmp_path: Path) -> None:
    root = _source_result_tree(tmp_path)
    os.chmod(root, 0o700)
    query = root / "query_predictions.csv"
    query.write_text("query_token,label\nq_secret,1\n", encoding="utf-8")
    os.chmod(query, 0o400)
    manifest_path = root / "result_bundle_manifest.json"
    os.chmod(manifest_path, 0o600)
    manifest = json.loads(manifest_path.read_text())
    manifest["files"].append(
        {
            "relative_path": query.name,
            "size_bytes": query.stat().st_size,
            "file_sha256": _file_sha256(query),
        }
    )
    manifest.pop("result_bundle_sha256")
    manifest["result_bundle_sha256"] = canonical_json_sha256(manifest)
    _write_json(manifest_path, manifest)
    os.chmod(manifest_path, 0o400)
    with pytest.raises(PermissionError, match="aggregate-only allowlist"):
        validate_metadata_calibration_result_bundle(
            root,
            completion_receipt_path=root / "completion_receipt.json",
            expected_phase="source_development",
        )


def test_result_bundle_rejects_gate_receipt_mismatch(tmp_path: Path) -> None:
    root = _source_result_tree(tmp_path)
    completion = root / "completion_receipt.json"
    os.chmod(completion, 0o600)
    _write_json(
        completion,
        {
            "development_gate_result": {
                "schema": "cfeg.metadata-calibration-development-gate.v1",
                "passed": True,
            }
        },
    )
    os.chmod(completion, 0o400)
    with pytest.raises(PermissionError, match="signed completion receipt"):
        validate_metadata_calibration_result_bundle(
            root,
            completion_receipt_path=completion,
            expected_phase="source_development",
        )
