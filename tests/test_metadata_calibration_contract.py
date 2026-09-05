from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from cfeg import eval_loop
from cfeg.governance import GovernanceError, resolve_research_access
from cfeg.identity import canonical_identity_sha256
from cfeg.metadata_calibration_contract import (
    METADATA_CALIBRATION_BUDGETS,
    audit_complete_block_cohort_partition,
    build_complete_block_calibration_partition,
    validate_metadata_calibration_baseline_ledger,
    validate_metadata_calibration_input_artifact_contract,
    validate_metadata_calibration_plan,
    validate_metadata_calibration_power_receipt,
)
from cfeg.utils.config import load_config

REPO = Path(__file__).resolve().parents[1]


def _manifest() -> pd.DataFrame:
    rows = []
    for block in range(1, 11):
        for label in range(12):
            rows.append(
                {
                    "sample_id": f"wearable_sub004_dry_block{block:02d}_target{label:02d}",
                    "dataset_id": "wearable",
                    "subject_id": "sub004",
                    "electrode_type": "dry",
                    "run_id": f"block{block:02d}",
                    "label": label,
                }
            )
    return pd.DataFrame(rows)


def _two_condition_manifest(subject_ids: tuple[str, ...] = ("sub004",)) -> pd.DataFrame:
    frames = []
    for subject_id in subject_ids:
        for condition in ("dry", "wet"):
            frame = _manifest().copy()
            frame["subject_id"] = subject_id
            frame["electrode_type"] = condition
            frame["sample_id"] = (
                "wearable_"
                + subject_id
                + "_"
                + condition
                + "_"
                + frame["run_id"].astype(str)
                + "_target"
                + frame["label"].astype(str).str.zfill(2)
            )
            frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def _safe_manifest() -> pd.DataFrame:
    rows = []
    for block in range(1, 11):
        prefix = "s" if block <= 5 else "q"
        for row in range(12):
            digest = hashlib.sha256(f"sealed:{block}:{row}".encode()).hexdigest()
            rows.append(
                {
                    "row_token": f"{prefix}_{digest}",
                    "dataset_id": "wearable",
                    "subject_id": "sub004",
                    "electrode_type": "dry",
                    "run_id": f"block{block:02d}",
                }
            )
    return pd.DataFrame(rows)


def test_frozen_plan_authorizes_only_the_signed_phase_chain() -> None:
    binding = validate_metadata_calibration_plan()

    assert binding["candidate_id"] == "metadata-calibration-efficiency-v1"
    assert binding["status"] == "frozen_execution_ready_signed_phase_authority_required"
    assert binding["outcome_execution_authorized"] is True
    assert binding["authorization_basis"].startswith("응 그렇게 해보자")
    assert len(binding["plan_sha256"]) == 64
    assert len(binding["input_artifact_contract_sha256"]) == 64


def test_input_artifact_contract_is_phase_scoped_and_grants_no_authority() -> None:
    binding = validate_metadata_calibration_input_artifact_contract()
    assert binding["human_input_access_authorized"] is False
    assert binding["schema"] == "cfeg.metadata-calibration-input-artifacts.v1"
    contract = load_config(
        REPO / "configs/analysis/metadata_calibration_input_artifacts_v1.yaml",
        strict_env=False,
    )
    source = contract["phase_contract"]["source_development"]
    held = contract["phase_contract"]["held_participant_evaluation"]
    assert (
        source["final_refit_source_participants"]
        == "forbidden_in_source_development_phase"
    )
    assert held["final_refit_source_participants"] == 39


def test_baseline_ledger_rejects_source_information_rights_drift(
    tmp_path: Path,
) -> None:
    source = REPO / "configs/analysis/metadata_calibration_baseline_ledger_v1.yaml"
    ledger = yaml.safe_load(source.read_text(encoding="utf-8"))
    mandatory = {
        item["id"]: item for item in ledger["mandatory_before_held_reveal"]
    }
    mandatory["proposed_common_A_Q"]["source_labeled_eeg"] = (
        "exact_39_source_development_participants"
    )
    drifted = tmp_path / "baseline_ledger.yaml"
    drifted.write_text(yaml.safe_dump(ledger, sort_keys=False), encoding="utf-8")
    with pytest.raises(ValueError, match="budget or external-context rights drifted"):
        validate_metadata_calibration_baseline_ledger(drifted)


def test_baseline_ledger_rejects_silent_direct_comparator_demotion(
    tmp_path: Path,
) -> None:
    source = REPO / "configs/analysis/metadata_calibration_baseline_ledger_v1.yaml"
    ledger = yaml.safe_load(source.read_text(encoding="utf-8"))
    direct = {item["id"]: item for item in ledger["direct_prior_art_comparators"]}
    direct["CSDuDoFN_OS_SSVEP"]["priority"] = "optional"
    drifted = tmp_path / "baseline_ledger.yaml"
    drifted.write_text(yaml.safe_dump(ledger, sort_keys=False), encoding="utf-8")
    with pytest.raises(ValueError, match="direct prior-art comparator ledger"):
        validate_metadata_calibration_baseline_ledger(drifted)


def test_input_artifact_contract_rejects_unvalidated_artifact_set_drift(
    tmp_path: Path,
) -> None:
    source = REPO / "configs/analysis/metadata_calibration_input_artifacts_v1.yaml"
    contract = yaml.safe_load(source.read_text(encoding="utf-8"))
    contract["artifact_sets"]["target_query"]["runner"].remove(
        "expected_unlabeled.jsonl"
    )
    drifted = tmp_path / "input_artifacts.yaml"
    drifted.write_text(yaml.safe_dump(contract, sort_keys=False), encoding="utf-8")
    with pytest.raises(ValueError, match="contract bytes drifted"):
        validate_metadata_calibration_input_artifact_contract(drifted)


def test_power_receipt_rejects_rehashed_dgp_tampering(tmp_path: Path) -> None:
    plan_path = REPO / "configs/analysis/metadata_calibration_efficiency_v1.yaml"
    plan = load_config(plan_path, strict_env=False)
    source = REPO / "configs/governance/metadata_calibration_power_receipt.json"
    receipt = json.loads(source.read_text(encoding="utf-8"))
    receipt.pop("receipt_payload_sha256")
    receipt["held_core_hierarchy_monte_carlo"][0]["true_means"] = [0.9, 0.9, 0.9]
    receipt["receipt_payload_sha256"] = hashlib.sha256(
        json.dumps(
            receipt, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode()
    ).hexdigest()
    tampered = tmp_path / "power_receipt.json"
    tampered.write_text(json.dumps(receipt, sort_keys=True), encoding="utf-8")
    with pytest.raises(ValueError, match="cell DGP"):
        validate_metadata_calibration_power_receipt(
            tampered,
            expected_plan_sha256=hashlib.sha256(plan_path.read_bytes()).hexdigest(),
            expected_power_config=plan["power_sensitivity"],
        )


def test_plan_rejects_allocation_or_nested_fallback_drift(tmp_path: Path) -> None:
    plan = yaml.safe_load(
        (REPO / "configs/analysis/metadata_calibration_efficiency_v1.yaml").read_text()
    )
    allocation = plan["data_contract"]["proposed_unopened_allocation"]
    allocation["held_participant_evaluation_subject_ids"][0] = "sub004"
    drifted = tmp_path / "allocation_drift.yaml"
    drifted.write_text(yaml.safe_dump(plan, sort_keys=False), encoding="utf-8")
    with pytest.raises(ValueError, match="exact outcome-blind 39/60 allocation"):
        validate_metadata_calibration_plan(drifted)

    plan = yaml.safe_load(
        (REPO / "configs/analysis/metadata_calibration_efficiency_v1.yaml").read_text()
    )
    plan["method_contract"]["fairness"]["missing_metadata_fallback"] = (
        "separately_trained_checkpoint"
    )
    drifted = tmp_path / "fallback_drift.yaml"
    drifted.write_text(yaml.safe_dump(plan, sort_keys=False), encoding="utf-8")
    with pytest.raises(ValueError, match="exact-fallback"):
        validate_metadata_calibration_plan(drifted)

    plan = yaml.safe_load(
        (REPO / "configs/analysis/metadata_calibration_efficiency_v1.yaml").read_text()
    )
    plan["estimands"]["primary"]["alternative"] = "two_sided"
    drifted = tmp_path / "estimand_drift.yaml"
    drifted.write_text(yaml.safe_dump(plan, sort_keys=False), encoding="utf-8")
    with pytest.raises(ValueError, match="primary estimand"):
        validate_metadata_calibration_plan(drifted)

    plan = yaml.safe_load(
        (REPO / "configs/analysis/metadata_calibration_efficiency_v1.yaml").read_text()
    )
    plan["method_contract"]["parameter_bounds"]["metadata_precision_ratio"] = [0.5, 2.0]
    drifted = tmp_path / "formula_drift.yaml"
    drifted.write_text(yaml.safe_dump(plan, sort_keys=False), encoding="utf-8")
    with pytest.raises(ValueError, match="diagonal-Gaussian"):
        validate_metadata_calibration_plan(drifted)

    plan = yaml.safe_load(
        (REPO / "configs/analysis/metadata_calibration_efficiency_v1.yaml").read_text()
    )
    plan["source_development_gate"]["numeric_thresholds"]["eAUC_increment_mean_min"] = 0.0
    drifted = tmp_path / "development_gate_drift.yaml"
    drifted.write_text(yaml.safe_dump(plan, sort_keys=False), encoding="utf-8")
    with pytest.raises(ValueError, match="source-development gate"):
        validate_metadata_calibration_plan(drifted)


def test_complete_blocks_are_nested_and_query_is_fixed() -> None:
    manifest = _safe_manifest().sample(frac=1.0, random_state=7).reset_index(drop=True)
    partition = build_complete_block_calibration_partition(manifest, np.arange(len(manifest)))

    assert tuple(partition.support_by_budget) == METADATA_CALIBRATION_BUDGETS
    assert {budget: len(indices) for budget, indices in partition.support_by_budget.items()} == {
        0: 0,
        1: 12,
        3: 36,
        5: 60,
    }
    assert len(partition.query_indices) == 60
    assert len(partition.query_identity_sha256) == 64
    query_ids = manifest.iloc[partition.query_indices]["row_token"].astype(str)
    assert partition.query_identity_sha256 == canonical_identity_sha256(query_ids)

    query_runs = set(manifest.iloc[partition.query_indices]["run_id"])
    assert query_runs == {f"block{block:02d}" for block in range(6, 11)}
    support_runs = {
        budget: set(manifest.iloc[indices]["run_id"])
        for budget, indices in partition.support_by_budget.items()
    }
    assert support_runs == {
        0: set(),
        1: {"block01"},
        3: {"block01", "block02", "block03"},
        5: {f"block{block:02d}" for block in range(1, 6)},
    }
    assert set(partition.support_by_budget[1]) < set(partition.support_by_budget[3])
    assert set(partition.support_by_budget[3]) < set(partition.support_by_budget[5])
    assert not set(partition.support_by_budget[5]).intersection(partition.query_indices)


def test_partition_rejects_incomplete_or_mixed_condition_blocks() -> None:
    incomplete = _safe_manifest().drop(index=0).reset_index(drop=True)
    with pytest.raises(ValueError, match="does not contain exactly"):
        build_complete_block_calibration_partition(incomplete, np.arange(len(incomplete)))

    mixed = _safe_manifest()
    mixed.loc[mixed.index[-1], "electrode_type"] = "wet"
    with pytest.raises(ValueError, match="one participant-condition group"):
        build_complete_block_calibration_partition(mixed, np.arange(len(mixed)))


def test_partition_rejects_classwise_or_budget_drift() -> None:
    manifest = _safe_manifest()

    with pytest.raises(ValueError, match="exactly \\(0, 1, 3, 5\\)"):
        build_complete_block_calibration_partition(
            manifest,
            np.arange(len(manifest)),
            budgets=(0, 1, 2, 5),
        )

    block_mixed = manifest.copy()
    block_mixed.loc[0, "run_id"] = "block02"
    with pytest.raises(ValueError, match="does not contain exactly"):
        build_complete_block_calibration_partition(block_mixed, np.arange(len(block_mixed)))

    for forbidden in ("sample_id", "label", "trial_id"):
        with pytest.raises(ValueError, match="exact allowlist"):
            build_complete_block_calibration_partition(
                manifest.assign(**{forbidden: "forbidden"}), np.arange(len(manifest))
            )


def test_cohort_binding_requires_exact_wet_dry_membership_and_query_identity() -> None:
    plan = yaml.safe_load(
        (REPO / "configs/analysis/metadata_calibration_efficiency_v1.yaml").read_text()
    )
    allocation = plan["data_contract"]["proposed_unopened_allocation"]
    subject_ids = tuple(allocation["source_development_subject_ids"])
    manifest = _two_condition_manifest(subject_ids)
    first = audit_complete_block_cohort_partition(
        manifest,
        role="source_development",
    )

    assert first.subject_ids == tuple(sorted(subject_ids))
    assert first.participant_condition_groups == 78
    assert first.query_count == 4680
    assert first.query_identity_bundle_sha256 == allocation["source_query_identity_bundle_sha256"]
    repeated = audit_complete_block_cohort_partition(
        manifest.sample(frac=1.0, random_state=11).reset_index(drop=True),
        role="source_development",
    )
    assert repeated.query_identity_bundle_sha256 == first.query_identity_bundle_sha256

    missing_subject = subject_ids[-1]
    missing_wet = manifest.loc[
        ~(manifest["subject_id"].eq(missing_subject) & manifest["electrode_type"].eq("wet"))
    ].reset_index(drop=True)
    with pytest.raises(ValueError, match="exact dry/wet conditions"):
        audit_complete_block_cohort_partition(
            missing_wet,
            role="source_development",
        )

    drifted_query = manifest.copy()
    query_row = drifted_query.index[drifted_query["run_id"].eq("block06")][0]
    drifted_query.loc[query_row, "sample_id"] = "wearable_drifted_query_identity"
    with pytest.raises(ValueError, match="query identity drifted"):
        audit_complete_block_cohort_partition(
            drifted_query,
            role="source_development",
        )

    with pytest.raises(ValueError, match="role must be"):
        audit_complete_block_cohort_partition(manifest, role="caller_selected")


@pytest.mark.parametrize("asset_receipt", ["valid", "missing", "malformed"])
def test_generic_checkpoint_cannot_override_target_to_wearable_v3_before_manifest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, asset_receipt: str
) -> None:
    target = tmp_path / "renamed_target"
    target.mkdir()
    (target / "manifest.jsonl").write_text(
        json.dumps({"dataset_id": "wearable", "sample_id": "sealed"}) + "\n",
        encoding="utf-8",
    )
    if asset_receipt == "valid":
        (target / "asset_info.json").write_text(
            json.dumps({"dataset_id": "wearable", "dataset_revision": "wearable_v3"}),
            encoding="utf-8",
        )
    elif asset_receipt == "malformed":
        (target / "asset_info.json").write_text("{broken", encoding="utf-8")
    generic = load_config(REPO / "configs/train/debug.yaml", strict_env=False)
    generic["runtime"] = {"device": "cpu"}
    manifest_loaded = False

    def forbidden_manifest(*args, **kwargs):
        nonlocal manifest_loaded
        manifest_loaded = True
        raise AssertionError("wearable manifest must remain sealed")

    monkeypatch.setattr(
        eval_loop,
        "load_checkpoint",
        lambda *args, **kwargs: {"config": generic},
    )
    monkeypatch.setattr(eval_loop, "load_manifest", forbidden_manifest)

    with pytest.raises(GovernanceError, match="DEC-20260905-006"):
        eval_loop.load_evaluation_context(
            {"data": {"processed_dirs": [str(target)]}},
            "unused.pt",
        )
    assert manifest_loaded is False


@pytest.mark.parametrize("asset_receipt", ["missing", "malformed"])
def test_renamed_wearable_manifest_cannot_enter_ungoverned_training(
    tmp_path: Path, asset_receipt: str
) -> None:
    target = tmp_path / "renamed_target"
    target.mkdir()
    (target / "manifest.jsonl").write_text(
        json.dumps({"dataset_id": "wearable", "sample_id": "sealed"}) + "\n",
        encoding="utf-8",
    )
    if asset_receipt == "malformed":
        (target / "asset_info.json").write_text("{broken", encoding="utf-8")
    generic = load_config(REPO / "configs/train/debug.yaml", strict_env=False)
    generic["data"]["processed_dirs"] = [str(target)]
    generic["data"]["expected_revisions"] = {}
    generic["protocol"] = {"governance_required": False, "execution_phase": "legacy"}

    with pytest.raises(GovernanceError, match="governance_required=true"):
        resolve_research_access(generic)
