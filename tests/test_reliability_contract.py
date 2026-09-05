from __future__ import annotations

import copy
from pathlib import Path

import pytest

from cfeg.governance import GovernanceError
from cfeg.reliability_contract import (
    RELIABILITY_ROLES,
    validate_reliability_candidate_bindings,
    validate_reliability_family,
    validate_reliability_runtime_family,
    validate_reliability_stage0_gate,
    validate_reliability_training_preflight,
)
from cfeg.train_loop import run_training
from cfeg.utils.config import load_config, merge_overrides

REPO = Path(__file__).resolve().parents[1]


def _configs() -> tuple[dict, dict]:
    base = load_config(
        REPO / "configs/train/synthetic_reliability_candidate.yaml",
        strict_env=False,
    )
    suite = load_config(
        REPO / "configs/train/synthetic_reliability_2x2.yaml",
        strict_env=False,
    )
    return base, suite


def _runtime_rows() -> list[dict]:
    shared = {
        "reliability_mechanism_family_sha256": "a" * 64,
        "candidate_id": "reliability-spatial-v1",
        "candidate_plan_sha256": "6" * 64,
        "predecessor_retirement_receipt_sha256": "7" * 64,
        "reliability_query_feature_schema": "query_window_reliability_features_v1",
        "reliability_stage0_receipt_sha256": None,
        "status": "dry_run",
        "seed": 42,
        "split_seed": 42,
        "n_folds": None,
        "fold_index": None,
        "total_parameters": 100,
        "trainable_parameters": 100,
        "parameter_schema_sha256": "b" * 64,
        "initial_trainable_state_sha256": "c" * 64,
        "split_assignment_sha256": "d" * 64,
        "vocabulary_sha256": "e" * 64,
        "asset_provenance_sha256": "f" * 64,
        "source_commit_sha": "1" * 40,
        "source_dirty": True,
        "source_tree_sha256": "2" * 64,
        "environment_sha256": "3" * 64,
        "execution_phase": "synthetic_engineering",
        # Ungoverned synthetic runs have no wearable analysis plan.  None is
        # therefore an explicit shared state, not a missing family binding.
        "analysis_plan_status": None,
        "cohort_role": "simulation_only",
        "loader_settings_sha256": "4" * 64,
        "development_control_sha256": "5" * 64,
        "runtime_metrics_schema": "cfeg.runtime-metrics.v1",
        "resolved_device": "cuda",
    }
    treatments = {
        "A0": ("null", "null"),
        "A_M": ("null", "observed"),
        "A_Q": ("observed", "null"),
        "A_QM": ("observed", "observed"),
    }
    return [
        {
            "variant": role,
            "reliability_family_role": role,
            "query_qc_mode": treatments[role][0],
            "external_metadata_mode": treatments[role][1],
            **shared,
        }
        for role in RELIABILITY_ROLES
    ]


def test_synthetic_reliability_family_is_exact_versioned_2x2() -> None:
    base, suite = _configs()

    digest = validate_reliability_family(base, suite)

    assert digest is not None and len(digest) == 64
    assert tuple(suite["variants"]) == RELIABILITY_ROLES
    assert set(suite["variants"]).isdisjoint(
        {"A0_eeg_only", "A2_structured_condition_prompt"}
    )
    assert validate_reliability_candidate_bindings(base)["status"] == (
        "frozen_for_synthetic_engineering"
    )
    assert validate_reliability_stage0_gate(base, require_receipt=False) is None


def test_reliability_family_rejects_treatment_or_recipe_drift() -> None:
    base, suite = _configs()
    wrong_treatment = copy.deepcopy(suite)
    wrong_treatment["variants"]["A_M"][
        "model.condition_encoder.external_metadata_mode"
    ] = "null"
    with pytest.raises(ValueError, match="treatment mismatch"):
        validate_reliability_family(base, wrong_treatment)

    recipe_drift = copy.deepcopy(suite)
    recipe_drift["variants"]["A_QM"]["train.lr"] = 0.01
    with pytest.raises(ValueError, match="outside the two declared access axes"):
        validate_reliability_family(base, recipe_drift)

    base_drift = copy.deepcopy(base)
    base_drift["train"]["lr"] = 0.01
    with pytest.raises(ValueError, match="frozen full contract"):
        validate_reliability_family(base_drift, suite)

    suite_drift = copy.deepcopy(suite)
    for overrides in suite_drift["variants"].values():
        overrides["train.weight_decay"] = 0.02
    with pytest.raises(ValueError, match="frozen canonical file"):
        validate_reliability_family(base, suite_drift)


def test_reliability_family_cannot_target_wearable_or_authorize_inference() -> None:
    base, suite = _configs()
    wearable = copy.deepcopy(base)
    wearable["data"]["expected_revisions"] = {"wearable": "wearable_v3"}
    with pytest.raises(ValueError, match="only synthetic_quality_v1"):
        validate_reliability_family(wearable, suite)

    promoted = copy.deepcopy(suite)
    promoted["reliability_family"]["confirmatory_execution_authorized"] = True
    with pytest.raises(ValueError, match="confirmatory_execution_authorized=false"):
        validate_reliability_family(base, promoted)

    stale = copy.deepcopy(base)
    stale["protocol"]["predecessor_retirement_receipt_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="retirement digest"):
        validate_reliability_family(stale, suite)

    with pytest.raises(ValueError, match="canonical four-arm orchestrator"):
        validate_reliability_training_preflight(base, dry_run=True)


def test_reliability_training_preflight_requires_exact_orchestrated_recipe() -> None:
    base, suite = _configs()
    family_sha256 = validate_reliability_family(base, suite)
    cfg = merge_overrides(
        base,
        [f"{key}={value!r}" for key, value in suite["variants"]["A_Q"].items()],
    )
    cfg["protocol"].update(
        {
            "reliability_mechanism_family_sha256": family_sha256,
            "reliability_family_role": "A_Q",
            "reliability_stage0_receipt_sha256": None,
        }
    )
    assert validate_reliability_training_preflight(cfg, dry_run=True) is None

    checkpoint_cfg = copy.deepcopy(cfg)
    checkpoint_cfg["runtime_contract"] = {"source_tree_sha256": "recorded"}
    checkpoint_cfg["augment"]["channel_subset_ids"] = [[48, 55, 57, 56, 53, 61, 62, 63]]
    assert validate_reliability_training_preflight(checkpoint_cfg, dry_run=True) is None

    drifted = copy.deepcopy(cfg)
    drifted["train"]["epochs"] += 1
    with pytest.raises(ValueError, match="frozen"):
        validate_reliability_training_preflight(drifted, dry_run=True)

    with pytest.raises(GovernanceError, match="direct train.py/API execution"):
        run_training(cfg, dry_run=True)


def test_reliability_runtime_parity_requires_identical_graph_state_and_data() -> None:
    _base, suite = _configs()
    rows = _runtime_rows()

    assert (
        validate_reliability_runtime_family(rows, suite["reliability_family"])
        == "verified_equal"
    )
    changed = copy.deepcopy(rows)
    changed[-1]["initial_trainable_state_sha256"] = "9" * 64
    with pytest.raises(ValueError, match="initial_trainable_state_sha256"):
        validate_reliability_runtime_family(changed, suite["reliability_family"])
    assert (
        validate_reliability_runtime_family(rows[:-1], suite["reliability_family"])
        == "not_run_together"
    )
    completed_without_receipt = copy.deepcopy(rows)
    for row in completed_without_receipt:
        row["status"] = "completed"
    with pytest.raises(ValueError, match="non-null Stage-0 receipt"):
        validate_reliability_runtime_family(
            completed_without_receipt, suite["reliability_family"]
        )
