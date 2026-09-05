from __future__ import annotations

import copy
import ctypes
import errno
import hashlib
import json
import os
import re
import shutil
import stat
import subprocess
import uuid
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import h5py
import numpy as np
import pandas as pd

from cfeg.data.metadata_calibration_features import WEARABLE_CALIBRATION_CHANNEL_IDS
from cfeg.data.metadata_calibration_interventions import (
    metadata_calibration_base_input_sha256,
)
from cfeg.data.unlabeled_query import (
    OpaqueTokenDomain,
    opaque_metadata_row_token,
    token_secret_commitment_sha256,
)
from cfeg.governance import current_source_revision_contract
from cfeg.identity import canonical_identity_sha256
from cfeg.metadata_calibration_attempt import validate_outcome_access_claim_evidence
from cfeg.metadata_calibration_authority import (
    LABEL_ENCRYPTION_ALGORITHM,
    canonical_json_bytes,
    canonical_json_sha256,
    encrypt_json_for_finalizer,
    issue_signed_record,
    public_key_fingerprint_sha256,
    read_json_object,
    verify_signed_record,
)
from cfeg.metadata_calibration_contract import (
    METADATA_CALIBRATION_INPUT_ARTIFACT_CONTRACT,
    METADATA_CALIBRATION_PLAN,
    METADATA_CALIBRATION_SOURCE_RECIPE,
    audit_complete_block_cohort_partition,
    validate_metadata_calibration_plan,
)
from cfeg.utils.config import load_config

ExperimentPhase = Literal["source_development", "held_participant_evaluation"]
INPUT_DECISION_SCHEMA = "cfeg.metadata-calibration-input-seal-decision.v1"
INPUT_RECEIPT_SCHEMA = "cfeg.metadata-calibration-input-seal-receipt.v1"
FILESYSTEM_PRIVATE_SEAL_SCHEMA = (
    "cfeg.metadata-calibration-filesystem-private-seal.v1"
)
DEVELOPMENT_GATE_RECEIPT_SCHEMA = (
    "cfeg.metadata-calibration-development-gate-receipt.v1"
)
_SHA256 = re.compile(r"[0-9a-f]{64}")
_REPOSITORY = Path(__file__).resolve().parents[2]
_INPUT_DECISION_FIELDS = {
    "schema",
    "candidate_id",
    "decision_id",
    "decision_date",
    "phase",
    "input_seal_only_approved",
    "approved_action",
    "approved_target_subject_ids_sha256",
    "approved_source_fit_subject_ids_by_checkpoint_group_sha256",
    "plan_sha256",
    "source_recipe_sha256",
    "baseline_ledger_sha256",
    "input_artifact_contract_sha256",
    "input_sealer_implementation_bundle_sha256",
    "token_domain_contract_sha256",
    "source_commit",
    "source_tree_sha256",
    "source_tag",
    "processed_asset_root",
    "raw_asset_receipt_sha256",
    "raw_asset_fingerprint_bundle_sha256",
    "class_map_sha256",
    "output_root",
    "token_secret_domain_separated_commitment_sha256",
    "source_development_gate_receipt_sha256",
    "source_development_private_seal_sha256",
    "source_development_gate_signed_record_sha256",
    "source_development_private_seal_signed_record_sha256",
    "source_development_gate_receipt_path",
    "source_development_private_seal_path",
    "signing_key_id",
    "signer_role",
    "authorization_basis",
    "signature_algorithm",
    "label_capability_boundary",
    "outcome_privacy_scope",
    "decision_payload_sha256",
    "signature",
    "signed_record_sha256",
}
_INPUT_RECEIPT_FIELDS = {
    "schema",
    "candidate_id",
    "phase",
    "status",
    "input_seal_decision_signed_record_sha256",
    "plan_sha256",
    "source_recipe_sha256",
    "baseline_ledger_sha256",
    "input_artifact_contract_sha256",
    "token_domain_contract_sha256",
    "token_secret_domain_separated_commitment_sha256",
    "raw_asset_receipt_sha256",
    "raw_asset_fingerprint_bundle_sha256",
    "class_map_sha256",
    "input_sealer_implementation_bundle_sha256",
    "source_commit",
    "source_tree_sha256",
    "source_tag",
    "sealed_root",
    "checkpoint_groups",
    "exact_target_subject_ids_sha256",
    "cohort_audit",
    "label_capability_boundary",
    "outcome_privacy_scope",
    "artifact_records",
    "artifact_tree_sha256",
    "query_outcomes_loaded_by_runner",
    "training_scoring_or_metrics_performed",
    "signing_key_id",
    "signer_role",
    "authorization_basis",
    "signature_algorithm",
    "receipt_payload_sha256",
    "signature",
    "signed_record_sha256",
}
_ARTIFACT_RECORD_FIELDS = {
    "relative_path",
    "artifact_kind",
    "size_bytes",
    "file_sha256",
    "semantic_sha256",
}
_MODEL_VIEW_COLUMNS = (
    "base_input_sha256",
    "canonical_channel_ids",
    "checkpoint_group",
    "electrode_type",
    "impedance_kohm_by_channel",
    "impedance_missing_by_channel",
    "query_signal_std_by_channel",
    "query_signal_std_missing_by_channel",
    "run_id",
    "sfreq_processed",
    "signal_index",
    "subject_id",
)
_BASELINE_VIEW_COLUMNS = (
    "base_input_sha256",
    "checkpoint_group",
    "electrode_type",
    "run_id",
    "signal_index",
    "subject_id",
)
_FILESYSTEM_PRIVATE_SEAL_FIELDS = {
    "schema",
    "candidate_id",
    "phase",
    "status",
    "phase_manifest_sha256",
    "outcome_decision_signed_record_sha256",
    "authorization_envelope_sha256",
    "input_seal_receipt_signed_record_sha256",
    "input_artifact_tree_sha256",
    "execution_contract_sha256",
    "execution_output_root",
    "expected_job_count",
    "completed_job_ids",
    "producer_receipt_file_sha256s_by_job",
    "held_selected_epochs_by_seed",
    "held_selected_epochs_by_seed_sha256",
    "held_epoch_selection_derivation",
    "private_staging_seal",
    "private_staging_semantic_sha256",
    "private_artifact_records",
    "filesystem_private_tree_sha256",
    "complete_candidate_and_mandatory_baseline_grid",
    "query_outcomes_loaded",
    "source_commit",
    "source_tree_sha256",
    "source_tag",
    "signing_key_id",
    "signer_role",
    "authorization_basis",
    "signature_algorithm",
    "private_seal_payload_sha256",
    "signature",
    "signed_record_sha256",
}
_PRIVATE_ARTIFACT_RECORD_FIELDS = {
    "job_id",
    "relative_path",
    "artifact_kind",
    "size_bytes",
    "file_sha256",
    "mode_octal",
}
_DEVELOPMENT_GATE_RECEIPT_FIELDS = {
    "schema",
    "candidate_id",
    "phase",
    "development_gate_status",
    "source_phase_manifest_sha256",
    "source_outcome_decision_signed_record_sha256",
    "source_input_seal_receipt_signed_record_sha256",
    "source_input_artifact_tree_sha256",
    "source_execution_contract_sha256",
    "source_execution_output_root",
    "filesystem_private_seal_file_sha256",
    "filesystem_private_seal_signed_record_sha256",
    "filesystem_private_tree_sha256",
    "source_result_bundle_root",
    "source_result_bundle_sha256",
    "development_gate_result",
    "outcome_finalization_performed",
    "outcome_access_claim_path",
    "outcome_access_claim_file_sha256",
    "outcome_access_claim_signed_record_sha256",
    "outcome_access_scope_sha256",
    "held_input_access_authorized",
    "source_commit",
    "source_tree_sha256",
    "source_tag",
    "signing_key_id",
    "signer_role",
    "authorization_basis",
    "signature_algorithm",
    "gate_payload_sha256",
    "signature",
    "signed_record_sha256",
}
_RESULT_ANALYSIS_FILES = {
    "candidate_participant_balanced_accuracy.csv",
    "candidate_participant_contrasts.csv",
    "baseline_participant_balanced_accuracy.csv",
    "baseline_summary.csv",
    "analysis_metadata.json",
    "gate_or_claim_result.json",
    "analysis_covariate_summary.json",
}
_CANDIDATE_PARTICIPANT_BA_COLUMNS = {
    "role",
    "context",
    "budget",
    "subject_id",
    "participant_balanced_accuracy",
}
_BASELINE_PARTICIPANT_BA_COLUMNS = {
    "adapter_id",
    "budget",
    "subject_id",
    "participant_balanced_accuracy",
}
_BASELINE_SUMMARY_COLUMNS = {
    "adapter_id",
    "budget",
    "n_participants",
    "mean",
    "sd",
    "median",
    "q25",
    "q75",
    "minimum",
    "maximum",
}
_PARTICIPANT_CONTRAST_COLUMNS = {
    "subject_id",
    *(f"{prefix}_k{budget}" for prefix in ("a0", "am", "aq", "aqm", "shuffle", "stale", "wrong") for budget in (0, 1, 3, 5)),
    *(f"{prefix}_eauc" for prefix in ("a0", "am", "aq", "aqm", "shuffle", "stale", "wrong")),
    "metadata_without_q_eauc_increment",
    "q_without_metadata_eauc_increment",
    "eauc_increment",
    "correct_minus_shuffle_eauc",
    "correct_minus_stale_eauc",
    "wrong_minus_aq_eauc",
    "save_k0_vs_k1",
    "save_k1_vs_k3",
    "save_k3_vs_k5",
    "calibration_value_k1_to_k3",
}


@dataclass(frozen=True)
class InputSealDecisionBinding:
    phase: ExperimentPhase
    signed_record_sha256: str
    output_root: Path
    processed_asset_root: Path
    finalizer_public_key_fingerprint_sha256: str
    token_secret_commitment_sha256: str


@dataclass(frozen=True)
class InputSealBinding:
    phase: ExperimentPhase
    input_seal_receipt_signed_record_sha256: str
    artifact_tree_sha256: str
    sealed_root: Path
    checkpoint_groups: tuple[str, ...]


@dataclass(frozen=True)
class SourceGateBinding:
    gate_receipt_file_sha256: str
    private_seal_file_sha256: str
    gate_signed_record_sha256: str
    private_seal_signed_record_sha256: str
    held_selected_epochs_by_seed: dict[str, dict[str, int]]
    held_selected_epochs_by_seed_sha256: str


@dataclass(frozen=True)
class SourcePrivateEpochBinding:
    private_seal_file_sha256: str
    private_seal_signed_record_sha256: str
    held_selected_epochs_by_seed: dict[str, dict[str, int]]
    held_selected_epochs_by_seed_sha256: str
    held_epoch_selection_derivation_sha256: str


def input_sealer_implementation_bundle_sha256() -> str:
    paths = (
        "src/cfeg/metadata_calibration_authority.py",
        "src/cfeg/metadata_calibration_sealer.py",
        "src/cfeg/data/unlabeled_query.py",
        "src/cfeg/metadata_calibration_contract.py",
    )
    records = [
        {"path": relative, "file_sha256": _sha256_file(_REPOSITORY / relative)}
        for relative in paths
    ]
    return canonical_json_sha256({"files": records})


def build_input_seal_decision(
    *,
    phase: ExperimentPhase,
    decision_id: str,
    decision_date: str,
    processed_asset_root: str | Path,
    output_root: str | Path,
    token_secret: bytes,
    finalizer_public_key_path: str | Path,
    trusted_signing_public_key_path: str | Path,
    signing_key_id: str,
    source_tag: str,
    authorization_basis: str,
    signer_role: str = "delegated_automation_authority",
    source_development_gate_receipt_path: str | Path | None = None,
    source_development_private_seal_path: str | Path | None = None,
) -> dict[str, Any]:
    """Build the unsigned, outcome-free authority record for one input seal."""

    if phase not in {"source_development", "held_participant_evaluation"}:
        raise ValueError("Input-seal phase is invalid.")
    binding = validate_metadata_calibration_plan(_REPOSITORY / METADATA_CALIBRATION_PLAN)
    recipe = load_config(
        _REPOSITORY / METADATA_CALIBRATION_SOURCE_RECIPE,
        strict_env=False,
    )
    plan = load_config(_REPOSITORY / METADATA_CALIBRATION_PLAN, strict_env=False)
    source = current_source_revision_contract()
    if source.get("source_dirty") is not False:
        raise RuntimeError("Input-seal decisions require a clean source tree.")
    source_commit = str(source["source_commit_sha"])
    _verify_source_tag(source_tag, source_commit)
    group_sources, group_targets = _phase_subject_groups(
        phase=phase,
        recipe=recipe,
        plan=plan,
    )
    gate_receipt_path: Path | None = None
    private_seal_path: Path | None = None
    gate_receipt_sha256: str | None = None
    private_seal_sha256: str | None = None
    if phase == "held_participant_evaluation":
        if source_development_gate_receipt_path is None or (
            source_development_private_seal_path is None
        ):
            raise ValueError("Held sealing requires actual passing source artifacts.")
        gate_receipt_path = _absolute_no_symlink_path(
            Path(source_development_gate_receipt_path), must_exist=True
        )
        private_seal_path = _absolute_no_symlink_path(
            Path(source_development_private_seal_path), must_exist=True
        )
        source_gate = validate_source_gate_artifacts(
            gate_receipt_path,
            private_seal_path,
            trusted_signing_public_key_path=trusted_signing_public_key_path,
        )
        gate_receipt_sha256 = source_gate.gate_receipt_file_sha256
        private_seal_sha256 = source_gate.private_seal_file_sha256
        gate_signed_record_sha256 = source_gate.gate_signed_record_sha256
        private_seal_signed_record_sha256 = (
            source_gate.private_seal_signed_record_sha256
        )
        privacy_scope = "phase_wide_label_capability_isolation"
    else:
        if (
            source_development_gate_receipt_path is not None
            or source_development_private_seal_path is not None
        ):
            raise ValueError("Source sealing must not carry held-prerequisite receipts.")
        gate_signed_record_sha256 = None
        private_seal_signed_record_sha256 = None
        privacy_scope = "checkpoint_group_scientific_job_isolation"
    # For held, the PASS boundary above is checked before any processed human
    # asset receipt, manifest, signal, support label, or query outcome is read.
    processed_root = _absolute_no_symlink_path(Path(processed_asset_root), must_exist=True)
    output = _absolute_no_symlink_path(Path(output_root), must_exist=False)
    if os.path.lexists(output):
        raise FileExistsError("Input-seal output root must be completely new.")
    asset = _load_asset_receipt_metadata(processed_root)
    public_key = _load_public_key_for_fingerprint(finalizer_public_key_path)
    token_contract = load_config(
        _REPOSITORY / METADATA_CALIBRATION_INPUT_ARTIFACT_CONTRACT,
        strict_env=False,
    )["token_contract"]
    return {
        "schema": INPUT_DECISION_SCHEMA,
        "candidate_id": "metadata-calibration-efficiency-v1",
        "decision_id": str(decision_id),
        "decision_date": str(decision_date),
        "phase": phase,
        "input_seal_only_approved": True,
        "approved_action": "seal_inputs_only_no_training_scoring_or_outcome_reduction",
        "approved_target_subject_ids_sha256": canonical_identity_sha256(
            value for values in group_targets.values() for value in values
        ),
        "approved_source_fit_subject_ids_by_checkpoint_group_sha256": {
            group: canonical_identity_sha256(values)
            for group, values in group_sources.items()
        },
        "plan_sha256": binding["plan_sha256"],
        "source_recipe_sha256": binding["source_recipe_sha256"],
        "baseline_ledger_sha256": binding["baseline_ledger_sha256"],
        "input_artifact_contract_sha256": binding["input_artifact_contract_sha256"],
        "input_sealer_implementation_bundle_sha256": (
            input_sealer_implementation_bundle_sha256()
        ),
        "token_domain_contract_sha256": canonical_json_sha256(token_contract),
        "source_commit": source_commit,
        "source_tree_sha256": source["source_tree_sha256"],
        "source_tag": source_tag,
        "processed_asset_root": str(processed_root),
        "raw_asset_receipt_sha256": asset["raw_asset_receipt_sha256"],
        "raw_asset_fingerprint_bundle_sha256": asset[
            "raw_asset_fingerprint_bundle_sha256"
        ],
        "class_map_sha256": asset["class_map_sha256"],
        "output_root": str(output),
        "token_secret_domain_separated_commitment_sha256": (
            token_secret_commitment_sha256(token_secret)
        ),
        "source_development_gate_receipt_sha256": gate_receipt_sha256,
        "source_development_private_seal_sha256": private_seal_sha256,
        "source_development_gate_signed_record_sha256": gate_signed_record_sha256,
        "source_development_private_seal_signed_record_sha256": (
            private_seal_signed_record_sha256
        ),
        "source_development_gate_receipt_path": (
            None if gate_receipt_path is None else str(gate_receipt_path)
        ),
        "source_development_private_seal_path": (
            None if private_seal_path is None else str(private_seal_path)
        ),
        "signing_key_id": signing_key_id,
        "signer_role": signer_role,
        "authorization_basis": authorization_basis,
        "signature_algorithm": "ed25519",
        "label_capability_boundary": {
            "mode": "finalizer_public_key_encryption",
            "encryption_algorithm": LABEL_ENCRYPTION_ALGORITHM,
            "finalizer_public_key_fingerprint_sha256": (
                public_key_fingerprint_sha256(public_key)
            ),
            "encrypted_sidecar_plaintext_hash_exposed": False,
        },
        "outcome_privacy_scope": privacy_scope,
    }


def validate_input_seal_decision(
    decision: Mapping[str, Any],
    *,
    trusted_signing_public_key_path: str | Path,
    finalizer_public_key_path: str | Path,
) -> InputSealDecisionBinding:
    """Validate all authority and outcome-free dependencies before raw asset access."""

    decision_hash = verify_signed_record(
        decision,
        public_key_path=trusted_signing_public_key_path,
        hash_field="decision_payload_sha256",
        expected_schema=INPUT_DECISION_SCHEMA,
    )
    if set(decision) != _INPUT_DECISION_FIELDS:
        raise ValueError("Input-seal decision does not use its exact field schema.")
    if type(decision["input_seal_only_approved"]) is not bool:
        raise TypeError("Input-seal approval must be an exact JSON boolean.")
    if decision["input_seal_only_approved"] is not True or decision.get(
        "approved_action"
    ) != "seal_inputs_only_no_training_scoring_or_outcome_reduction":
        raise PermissionError("Input sealing is not approved by this record.")
    phase = str(decision.get("phase"))
    if phase not in {"source_development", "held_participant_evaluation"}:
        raise ValueError("Input-seal decision phase is invalid.")
    binding = validate_metadata_calibration_plan(_REPOSITORY / METADATA_CALIBRATION_PLAN)
    for field in (
        "plan_sha256",
        "source_recipe_sha256",
        "baseline_ledger_sha256",
        "input_artifact_contract_sha256",
    ):
        if decision.get(field) != binding[field]:
            raise ValueError(f"Input-seal decision {field} drifted from the repository.")
    if decision.get("input_sealer_implementation_bundle_sha256") != (
        input_sealer_implementation_bundle_sha256()
    ):
        raise ValueError("Input-sealer implementation bundle drifted.")
    token_contract = load_config(
        _REPOSITORY / METADATA_CALIBRATION_INPUT_ARTIFACT_CONTRACT,
        strict_env=False,
    )["token_contract"]
    if decision.get("token_domain_contract_sha256") != canonical_json_sha256(
        token_contract
    ):
        raise ValueError("Opaque-token domain contract drifted.")
    source = current_source_revision_contract()
    if source.get("source_dirty") is not False:
        raise RuntimeError("Input sealer refuses a dirty source tree.")
    if decision.get("source_commit") != source.get("source_commit_sha") or decision.get(
        "source_tree_sha256"
    ) != source.get("source_tree_sha256"):
        raise ValueError("Input-seal decision source revision drifted.")
    _verify_source_tag(str(decision.get("source_tag")), str(source["source_commit_sha"]))

    # Held authorization is re-proven from the actual PASS artifacts before
    # even the processed-asset receipt path is opened.
    if phase == "held_participant_evaluation":
        gate_path = _absolute_no_symlink_path(
            Path(str(decision.get("source_development_gate_receipt_path"))),
            must_exist=True,
        )
        seal_path = _absolute_no_symlink_path(
            Path(str(decision.get("source_development_private_seal_path"))),
            must_exist=True,
        )
        source_gate = validate_source_gate_artifacts(
            gate_path,
            seal_path,
            trusted_signing_public_key_path=trusted_signing_public_key_path,
        )
        if source_gate.gate_receipt_file_sha256 != decision.get(
            "source_development_gate_receipt_sha256"
        ) or source_gate.private_seal_file_sha256 != decision.get(
            "source_development_private_seal_sha256"
        ) or source_gate.gate_signed_record_sha256 != decision.get(
            "source_development_gate_signed_record_sha256"
        ) or source_gate.private_seal_signed_record_sha256 != decision.get(
            "source_development_private_seal_signed_record_sha256"
        ):
            raise ValueError("Held source PASS artifacts differ from the signed decision.")

    processed_root = _absolute_no_symlink_path(
        Path(str(decision.get("processed_asset_root"))), must_exist=True
    )
    output_root = _absolute_no_symlink_path(
        Path(str(decision.get("output_root"))), must_exist=False
    )
    if os.path.lexists(output_root):
        raise FileExistsError("Signed input-seal destination is no longer new.")
    asset = _load_asset_receipt_metadata(processed_root)
    for field in (
        "raw_asset_receipt_sha256",
        "raw_asset_fingerprint_bundle_sha256",
        "class_map_sha256",
    ):
        if decision.get(field) != asset[field]:
            raise ValueError(f"Input-seal decision {field} differs from the audit receipt.")
    label_boundary = decision.get("label_capability_boundary")
    expected_fingerprint = public_key_fingerprint_sha256(
        _load_public_key_for_fingerprint(finalizer_public_key_path)
    )
    if not isinstance(label_boundary, Mapping) or dict(label_boundary) != {
        "mode": "finalizer_public_key_encryption",
        "encryption_algorithm": LABEL_ENCRYPTION_ALGORITHM,
        "finalizer_public_key_fingerprint_sha256": expected_fingerprint,
        "encrypted_sidecar_plaintext_hash_exposed": False,
    }:
        raise ValueError("Input-seal label capability boundary is not the frozen public-key mode.")
    token_commitment = str(
        decision.get("token_secret_domain_separated_commitment_sha256")
    )
    if not _is_sha256(token_commitment):
        raise ValueError("Input-seal token-secret commitment is invalid.")

    plan = load_config(_REPOSITORY / METADATA_CALIBRATION_PLAN, strict_env=False)
    recipe = load_config(_REPOSITORY / METADATA_CALIBRATION_SOURCE_RECIPE, strict_env=False)
    group_sources, group_targets = _phase_subject_groups(
        phase=phase,  # type: ignore[arg-type]
        recipe=recipe,
        plan=plan,
    )
    if decision.get("approved_target_subject_ids_sha256") != canonical_identity_sha256(
        value for values in group_targets.values() for value in values
    ) or decision.get("approved_source_fit_subject_ids_by_checkpoint_group_sha256") != {
        group: canonical_identity_sha256(values) for group, values in group_sources.items()
    }:
        raise ValueError("Input-seal approved subject identities drifted.")
    if phase == "source_development":
        if decision.get("outcome_privacy_scope") != (
            "checkpoint_group_scientific_job_isolation"
        ) or any(
            decision.get(field) is not None
            for field in (
                "source_development_gate_receipt_sha256",
                "source_development_private_seal_sha256",
                "source_development_gate_signed_record_sha256",
                "source_development_private_seal_signed_record_sha256",
                "source_development_gate_receipt_path",
                "source_development_private_seal_path",
            )
        ):
            raise ValueError("Source input decision has incorrect privacy or prior-phase fields.")
    else:
        if decision.get("outcome_privacy_scope") != (
            "phase_wide_label_capability_isolation"
        ) or not all(
            _is_sha256(decision.get(field))
            for field in (
                "source_development_gate_receipt_sha256",
                "source_development_private_seal_sha256",
                "source_development_gate_signed_record_sha256",
                "source_development_private_seal_signed_record_sha256",
            )
        ):
            raise ValueError("Held input decision lacks exact passing-source bindings.")
        gate_path = _absolute_no_symlink_path(
            Path(str(decision.get("source_development_gate_receipt_path"))),
            must_exist=True,
        )
        seal_path = _absolute_no_symlink_path(
            Path(str(decision.get("source_development_private_seal_path"))),
            must_exist=True,
        )
        source_gate = validate_source_gate_artifacts(
            gate_path,
            seal_path,
            trusted_signing_public_key_path=trusted_signing_public_key_path,
        )
        if source_gate.gate_receipt_file_sha256 != decision.get(
            "source_development_gate_receipt_sha256"
        ) or source_gate.private_seal_file_sha256 != decision.get(
            "source_development_private_seal_sha256"
        ) or source_gate.gate_signed_record_sha256 != decision.get(
            "source_development_gate_signed_record_sha256"
        ) or source_gate.private_seal_signed_record_sha256 != decision.get(
            "source_development_private_seal_signed_record_sha256"
        ):
            raise ValueError("Held source PASS artifacts differ from the signed decision.")
    return InputSealDecisionBinding(
        phase=phase,  # type: ignore[arg-type]
        signed_record_sha256=decision_hash,
        output_root=output_root,
        processed_asset_root=processed_root,
        finalizer_public_key_fingerprint_sha256=expected_fingerprint,
        token_secret_commitment_sha256=token_commitment,
    )


def seal_metadata_calibration_inputs(
    decision: Mapping[str, Any],
    *,
    token_secret: bytes,
    trusted_signing_public_key_path: str | Path,
    finalizer_public_key_path: str | Path,
    receipt_signing_private_key_path: str | Path,
    receipt_signing_private_key_password: bytes | None = None,
) -> InputSealBinding:
    """Create a phase-scoped tokenized no-y bundle after signed authorization."""

    authority = validate_input_seal_decision(
        decision,
        trusted_signing_public_key_path=trusted_signing_public_key_path,
        finalizer_public_key_path=finalizer_public_key_path,
    )
    if token_secret_commitment_sha256(token_secret) != (
        authority.token_secret_commitment_sha256
    ):
        raise ValueError("In-memory token secret differs from the signed commitment.")

    # No manifest, signal, or label-bearing file is opened before the authority
    # validation above has completed.
    asset = _verify_raw_assets(authority.processed_asset_root)
    plan = load_config(_REPOSITORY / METADATA_CALIBRATION_PLAN, strict_env=False)
    recipe = load_config(_REPOSITORY / METADATA_CALIBRATION_SOURCE_RECIPE, strict_env=False)
    source_by_group, target_by_group = _phase_subject_groups(
        phase=authority.phase,
        recipe=recipe,
        plan=plan,
    )
    authorized_subjects = tuple(
        sorted(
            set().union(
                *(set(values) for values in source_by_group.values()),
                *(set(values) for values in target_by_group.values()),
            )
        )
    )
    manifest = _read_authorized_manifest(
        authority.processed_asset_root / "manifest.parquet",
        authorized_subjects=authorized_subjects,
    )
    audit = audit_complete_block_cohort_partition(manifest, role=authority.phase)
    _validate_raw_manifest_and_h5(
        manifest,
        authority.processed_asset_root / "signals.h5",
        expected_subjects=authorized_subjects,
    )
    staging = _new_sibling_staging_root(authority.output_root)
    artifact_records: list[dict[str, Any]] = []
    try:
        raw_indices = np.sort(manifest["h5_index"].to_numpy(dtype=np.int64))
        with h5py.File(authority.processed_asset_root / "signals.h5", "r") as raw_h5:
            # Never materialize an unauthorized participant signal in this
            # phase. Whole-file hashing above is an integrity-only byte stream.
            raw_x = np.asarray(raw_h5["x"][raw_indices], dtype=np.float32)
            raw_mask = np.asarray(raw_h5["channel_mask"][raw_indices], dtype=bool)
        local_index = {int(value): index for index, value in enumerate(raw_indices)}
        manifest = manifest.copy()
        manifest["h5_index"] = manifest["h5_index"].map(local_index)
        if manifest["h5_index"].isna().any():
            raise RuntimeError("Authorized signal index remapping was incomplete.")
        manifest["h5_index"] = manifest["h5_index"].astype(int)
        for checkpoint_group in source_by_group:
            group_root = staging / checkpoint_group
            group_root.mkdir(mode=0o700)
            records = _seal_checkpoint_group(
                manifest=manifest,
                raw_x=raw_x,
                raw_mask=raw_mask,
                phase=authority.phase,
                checkpoint_group=checkpoint_group,
                source_subjects=source_by_group[checkpoint_group],
                target_subjects=target_by_group[checkpoint_group],
                token_secret=token_secret,
                decision_signed_record_sha256=authority.signed_record_sha256,
                raw_asset_fingerprint_bundle_sha256=asset[
                    "raw_asset_fingerprint_bundle_sha256"
                ],
                token_secret_commitment_sha256=authority.token_secret_commitment_sha256,
                finalizer_public_key_path=Path(finalizer_public_key_path),
                finalizer_public_key_fingerprint_sha256=(
                    authority.finalizer_public_key_fingerprint_sha256
                ),
                group_root=group_root,
            )
            artifact_records.extend(records)

        decision_path = staging / "input_seal_decision.json"
        _write_json_file(decision_path, dict(decision), mode=0o400)
        artifact_records.append(_artifact_record(staging, decision_path, "signed_decision"))
        artifact_records = sorted(artifact_records, key=lambda value: value["relative_path"])
        artifact_tree_sha256 = canonical_json_sha256({"artifacts": artifact_records})
        receipt: dict[str, Any] = {
            "schema": INPUT_RECEIPT_SCHEMA,
            "candidate_id": "metadata-calibration-efficiency-v1",
            "phase": authority.phase,
            "status": "sealed_inputs_only_no_training_or_scoring",
            "input_seal_decision_signed_record_sha256": authority.signed_record_sha256,
            "plan_sha256": decision["plan_sha256"],
            "source_recipe_sha256": decision["source_recipe_sha256"],
            "baseline_ledger_sha256": decision["baseline_ledger_sha256"],
            "input_artifact_contract_sha256": decision[
                "input_artifact_contract_sha256"
            ],
            "token_domain_contract_sha256": decision["token_domain_contract_sha256"],
            "token_secret_domain_separated_commitment_sha256": (
                authority.token_secret_commitment_sha256
            ),
            "raw_asset_receipt_sha256": asset["raw_asset_receipt_sha256"],
            "raw_asset_fingerprint_bundle_sha256": asset[
                "raw_asset_fingerprint_bundle_sha256"
            ],
            "class_map_sha256": asset["class_map_sha256"],
            "input_sealer_implementation_bundle_sha256": (
                input_sealer_implementation_bundle_sha256()
            ),
            "source_commit": decision["source_commit"],
            "source_tree_sha256": decision["source_tree_sha256"],
            "source_tag": decision["source_tag"],
            "sealed_root": str(authority.output_root),
            "checkpoint_groups": list(source_by_group),
            "exact_target_subject_ids_sha256": decision[
                "approved_target_subject_ids_sha256"
            ],
            "cohort_audit": {
                "participant_condition_groups": audit.participant_condition_groups,
                "query_count": audit.query_count,
                "historical_query_identity_bundle_sha256": (
                    audit.query_identity_bundle_sha256
                ),
            },
            "label_capability_boundary": copy.deepcopy(
                decision["label_capability_boundary"]
            ),
            "outcome_privacy_scope": decision["outcome_privacy_scope"],
            "artifact_records": artifact_records,
            "artifact_tree_sha256": artifact_tree_sha256,
            "query_outcomes_loaded_by_runner": False,
            "training_scoring_or_metrics_performed": False,
            "signing_key_id": decision["signing_key_id"],
            "signer_role": decision["signer_role"],
            "authorization_basis": decision["authorization_basis"],
            "signature_algorithm": "ed25519",
        }
        signed_receipt = issue_signed_record(
            receipt,
            private_key_path=receipt_signing_private_key_path,
            private_key_password=receipt_signing_private_key_password,
            hash_field="receipt_payload_sha256",
        )
        receipt_path = staging / "input_seal_receipt.json"
        _write_json_file(receipt_path, signed_receipt, mode=0o400)
        _fsync_tree(staging)
        _validate_input_seal_tree(
            staging,
            declared_root=authority.output_root,
            trusted_signing_public_key_path=trusted_signing_public_key_path,
        )
        _publish_new_tree(staging, authority.output_root)
    except BaseException:
        _remove_owned_staging(staging, authority.output_root)
        raise
    return validate_input_seal_root(
        authority.output_root,
        trusted_signing_public_key_path=trusted_signing_public_key_path,
    )


def validate_input_seal_root(
    sealed_root: str | Path,
    *,
    trusted_signing_public_key_path: str | Path,
) -> InputSealBinding:
    root = _absolute_no_symlink_path(Path(sealed_root), must_exist=True)
    return _validate_input_seal_tree(
        root,
        declared_root=root,
        trusted_signing_public_key_path=trusted_signing_public_key_path,
    )


def _validate_input_seal_tree(
    root: Path,
    *,
    declared_root: Path,
    trusted_signing_public_key_path: str | Path,
) -> InputSealBinding:
    receipt = read_json_object(root / "input_seal_receipt.json")
    receipt_hash = verify_signed_record(
        receipt,
        public_key_path=trusted_signing_public_key_path,
        hash_field="receipt_payload_sha256",
        expected_schema=INPUT_RECEIPT_SCHEMA,
    )
    if set(receipt) != _INPUT_RECEIPT_FIELDS:
        raise ValueError("Input-seal receipt does not use its exact field schema.")
    if (
        receipt.get("candidate_id") != "metadata-calibration-efficiency-v1"
        or receipt.get("status") != "sealed_inputs_only_no_training_or_scoring"
        or type(receipt.get("query_outcomes_loaded_by_runner")) is not bool
        or receipt.get("query_outcomes_loaded_by_runner") is not False
        or type(receipt.get("training_scoring_or_metrics_performed")) is not bool
        or receipt.get("training_scoring_or_metrics_performed") is not False
    ):
        raise ValueError("Input-seal receipt status or outcome-free flags are invalid.")
    if receipt.get("sealed_root") != str(declared_root):
        raise ValueError("Input-seal receipt root differs from its actual location.")
    records = receipt.get("artifact_records")
    if not isinstance(records, list) or not records:
        raise TypeError("Input-seal receipt is missing artifact records.")
    if any(not isinstance(value, Mapping) or set(value) != _ARTIFACT_RECORD_FIELDS for value in records):
        raise ValueError("Input-seal artifact records do not use the exact schema.")
    record_paths = [str(value["relative_path"]) for value in records]
    if len(record_paths) != len(set(record_paths)):
        raise ValueError("Input-seal artifact records contain duplicate paths.")
    groups = receipt.get("checkpoint_groups")
    phase = str(receipt.get("phase"))
    expected_groups = (
        ["fold0", "fold1", "fold2"]
        if phase == "source_development"
        else ["held"]
        if phase == "held_participant_evaluation"
        else None
    )
    if groups != expected_groups:
        raise ValueError("Input-seal checkpoint groups differ from the phase contract.")
    expected_kinds = _expected_input_artifact_kinds(expected_groups)
    if {str(value["relative_path"]): str(value["artifact_kind"]) for value in records} != (
        expected_kinds
    ):
        raise ValueError("Input-seal artifact path/kind allowlist drifted.")
    actual_paths = _validate_sealed_tree_entries(root)
    expected_paths = set(record_paths)
    if actual_paths - {"input_seal_receipt.json"} != expected_paths:
        raise ValueError("Sealed input tree has missing or undeclared files.")
    for record in records:
        path = root / str(record["relative_path"])
        _validate_sealed_regular_file(path)
        if (
            path.stat().st_size != record.get("size_bytes")
            or _sha256_file(path) != record.get("file_sha256")
            or _semantic_artifact_sha256(path) != record.get("semantic_sha256")
        ):
            raise ValueError(f"Sealed input file hash drifted: {path}.")
        if path.suffix == ".h5":
            _validate_sealed_hdf5(path)
    expected_tree = canonical_json_sha256(
        {"artifacts": sorted(records, key=lambda value: value["relative_path"])}
    )
    if receipt.get("artifact_tree_sha256") != expected_tree:
        raise ValueError("Sealed input artifact tree hash drifted.")
    decision = read_json_object(root / "input_seal_decision.json")
    decision_hash = verify_signed_record(
        decision,
        public_key_path=trusted_signing_public_key_path,
        hash_field="decision_payload_sha256",
        expected_schema=INPUT_DECISION_SCHEMA,
    )
    if decision_hash != receipt.get("input_seal_decision_signed_record_sha256") or (
        set(decision) != _INPUT_DECISION_FIELDS
    ):
        raise ValueError("Input-seal decision binding inside the tree is invalid.")
    plan_binding = validate_metadata_calibration_plan(_REPOSITORY / METADATA_CALIBRATION_PLAN)
    for field in (
        "plan_sha256",
        "source_recipe_sha256",
        "baseline_ledger_sha256",
        "input_artifact_contract_sha256",
    ):
        if receipt.get(field) != plan_binding[field] or decision.get(field) != plan_binding[field]:
            raise ValueError(f"Input-seal tree repository binding drifted for {field}.")
    if receipt.get("input_sealer_implementation_bundle_sha256") != (
        input_sealer_implementation_bundle_sha256()
    ):
        raise ValueError("Input-seal receipt implementation bundle drifted.")
    return InputSealBinding(
        phase=phase,  # type: ignore[arg-type]
        input_seal_receipt_signed_record_sha256=receipt_hash,
        artifact_tree_sha256=expected_tree,
        sealed_root=declared_root,
        checkpoint_groups=tuple(expected_groups),
    )


def write_signed_decision_exclusive(path: str | Path, decision: Mapping[str, Any]) -> None:
    target = _absolute_no_symlink_path(Path(path), must_exist=False)
    if os.path.lexists(target):
        raise FileExistsError("Decision record path already exists.")
    _ensure_durable_private_parent(target.parent)
    staging = target.parent / f".{target.name}.staging-{uuid.uuid4().hex}"
    try:
        _write_json_file(staging, dict(decision), mode=0o400)
        _publish_new_tree(staging, target)
    except BaseException:
        if os.path.lexists(staging):
            staging.unlink()
            _fsync_directory(staging.parent)
        raise


def _ensure_durable_private_parent(parent: Path) -> None:
    """Create missing private ancestors and durably persist every new link."""

    parent = _absolute_no_symlink_path(parent, must_exist=False)
    missing: list[Path] = []
    current = parent
    while not current.exists():
        missing.append(current)
        current = current.parent
    for path in reversed(missing):
        try:
            path.mkdir(mode=0o700)
        except FileExistsError:
            pass
        observed = path.lstat()
        if not stat.S_ISDIR(observed.st_mode) or stat.S_ISLNK(observed.st_mode):
            raise ValueError("Decision parent hierarchy contains a non-directory component.")
        os.chmod(path, 0o700)
        _fsync_directory(path)
        _fsync_directory(path.parent)
    observed_parent = parent.lstat()
    if not stat.S_ISDIR(observed_parent.st_mode) or stat.S_ISLNK(observed_parent.st_mode):
        raise ValueError("Decision parent must be one real directory.")
    os.chmod(parent, 0o700)
    _fsync_directory(parent)
    _fsync_directory(parent.parent)


def _seal_checkpoint_group(
    *,
    manifest: pd.DataFrame,
    raw_x: np.ndarray,
    raw_mask: np.ndarray,
    phase: ExperimentPhase,
    checkpoint_group: str,
    source_subjects: Sequence[str],
    target_subjects: Sequence[str],
    token_secret: bytes,
    decision_signed_record_sha256: str,
    raw_asset_fingerprint_bundle_sha256: str,
    token_secret_commitment_sha256: str,
    finalizer_public_key_path: Path,
    finalizer_public_key_fingerprint_sha256: str,
    group_root: Path,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    source_rows = manifest.loc[
        manifest["dataset_id"].astype(str).eq("wearable")
        & manifest["subject_id"].astype(str).isin(source_subjects)
    ].copy()
    target_rows = manifest.loc[
        manifest["dataset_id"].astype(str).eq("wearable")
        & manifest["subject_id"].astype(str).isin(target_subjects)
    ].copy()
    if set(source_rows["subject_id"].astype(str)) != set(source_subjects):
        raise ValueError("Source-fit cohort is incomplete in the raw manifest.")
    if set(target_rows["subject_id"].astype(str)) != set(target_subjects):
        raise ValueError("Target cohort is incomplete in the raw manifest.")
    if set(source_subjects) & set(target_subjects):
        raise ValueError("One checkpoint group mixed source-fit and target participants.")

    source_domain = OpaqueTokenDomain(
        phase=phase,
        checkpoint_group=checkpoint_group,
        purpose="source_fit",
        raw_asset_fingerprint_sha256=raw_asset_fingerprint_bundle_sha256,
        seal_decision_sha256=decision_signed_record_sha256,
    )
    source_rows["row_token"] = [
        opaque_metadata_row_token(
            str(value), secret_key=token_secret, domain=source_domain
        )
        for value in source_rows["sample_id"]
    ]
    source_rows = source_rows.sort_values("row_token", kind="mergesort").reset_index(
        drop=True
    )
    source_dir = group_root / "source_fit"
    source_dir.mkdir(mode=0o700)
    source_view = _build_model_view(
        source_rows,
        raw_x=raw_x,
        raw_mask=raw_mask,
        checkpoint_group=checkpoint_group,
        token_field="source_token",
        include_label=True,
    )
    source_h5 = source_dir / "signals.h5"
    _write_sealed_hdf5(source_h5, source_rows, raw_x=raw_x, raw_mask=raw_mask)
    source_json = source_dir / "labeled_view.jsonl"
    source_baseline_json = source_dir / "baseline_view.jsonl"
    _write_jsonl_file(source_json, source_view)
    _write_jsonl_file(
        source_baseline_json,
        _build_baseline_view(source_view, token_field="source_token", include_label=True),
    )
    records.extend(
        (
            _artifact_record(group_root.parents[0], source_h5, "source_fit_signals"),
            _artifact_record(group_root.parents[0], source_json, "source_fit_labeled_view"),
            _artifact_record(
                group_root.parents[0],
                source_baseline_json,
                "source_fit_baseline_view",
            ),
        )
    )

    support_rows = target_rows.loc[
        target_rows["run_id"].astype(str).isin(
            [f"block{index:02d}" for index in range(1, 6)]
        )
    ].copy()
    query_rows = target_rows.loc[
        target_rows["run_id"].astype(str).isin(
            [f"block{index:02d}" for index in range(6, 11)]
        )
    ].copy()
    support_domain = OpaqueTokenDomain(
        phase=phase,
        checkpoint_group=checkpoint_group,
        purpose="evaluation_support",
        raw_asset_fingerprint_sha256=raw_asset_fingerprint_bundle_sha256,
        seal_decision_sha256=decision_signed_record_sha256,
    )
    query_domain = OpaqueTokenDomain(
        phase=phase,
        checkpoint_group=checkpoint_group,
        purpose="evaluation_query",
        raw_asset_fingerprint_sha256=raw_asset_fingerprint_bundle_sha256,
        seal_decision_sha256=decision_signed_record_sha256,
    )
    support_rows["row_token"] = [
        opaque_metadata_row_token(
            str(value), secret_key=token_secret, domain=support_domain
        )
        for value in support_rows["sample_id"]
    ]
    query_rows["row_token"] = [
        opaque_metadata_row_token(
            str(value), secret_key=token_secret, domain=query_domain
        )
        for value in query_rows["sample_id"]
    ]
    support_rows = support_rows.sort_values("row_token", kind="mergesort").reset_index(
        drop=True
    )
    query_rows = query_rows.sort_values("row_token", kind="mergesort").reset_index(
        drop=True
    )
    _validate_complete_target_rows(support_rows, query_rows, target_subjects)

    support_dir = group_root / "target_support"
    support_dir.mkdir(mode=0o700)
    support_view = _build_model_view(
        support_rows,
        raw_x=raw_x,
        raw_mask=raw_mask,
        checkpoint_group=checkpoint_group,
        token_field="support_token",
        include_label=False,
    )
    support_labels = _expanded_support_label_rows(
        support_rows,
        support_view=support_view,
        checkpoint_group=checkpoint_group,
    )
    support_h5 = support_dir / "signals.h5"
    support_view_path = support_dir / "model_view.jsonl"
    support_baseline_view_path = support_dir / "baseline_view.jsonl"
    support_label_path = support_dir / "support_labels.jsonl"
    _write_sealed_hdf5(support_h5, support_rows, raw_x=raw_x, raw_mask=raw_mask)
    _write_jsonl_file(support_view_path, support_view)
    _write_jsonl_file(
        support_baseline_view_path,
        _build_baseline_view(
            support_view,
            token_field="support_token",
            include_label=False,
        ),
    )
    _write_jsonl_file(support_label_path, support_labels)

    query_dir = group_root / "target_query"
    query_dir.mkdir(mode=0o700)
    query_view = _build_model_view(
        query_rows,
        raw_x=raw_x,
        raw_mask=raw_mask,
        checkpoint_group=checkpoint_group,
        token_field="query_token",
        include_label=False,
    )
    query_identity = {
        (subject, interface): canonical_identity_sha256(
            query_rows.loc[
                query_rows["subject_id"].astype(str).eq(subject)
                & query_rows["electrode_type"].astype(str).eq(interface),
                "row_token",
            ].astype(str)
        )
        for subject in sorted(set(query_rows["subject_id"].astype(str)))
        for interface in ("dry", "wet")
    }
    expected_unlabeled = [
        {
            "query_token": str(row["row_token"]),
            "query_identity_sha256": query_identity[
                (str(row["subject_id"]), str(row["electrode_type"]))
            ],
            "subject_id": str(row["subject_id"]),
            "electrode_type": str(row["electrode_type"]),
            "checkpoint_group": checkpoint_group,
        }
        for row in query_rows.to_dict(orient="records")
    ]
    query_h5 = query_dir / "signals.h5"
    query_view_path = query_dir / "model_view.jsonl"
    query_baseline_view_path = query_dir / "baseline_view.jsonl"
    expected_path = query_dir / "expected_unlabeled.jsonl"
    _write_sealed_hdf5(query_h5, query_rows, raw_x=raw_x, raw_mask=raw_mask)
    _write_jsonl_file(query_view_path, query_view)
    _write_jsonl_file(
        query_baseline_view_path,
        _build_baseline_view(
            query_view,
            token_field="query_token",
            include_label=False,
        ),
    )
    _write_jsonl_file(expected_path, expected_unlabeled)

    label_plaintext = {
        "schema": "cfeg.metadata-calibration-query-labels.v1",
        "candidate_id": "metadata-calibration-efficiency-v1",
        "phase": phase,
        "checkpoint_group": checkpoint_group,
        "rows": [
            {"query_token": str(row["row_token"]), "label": int(row["label"])}
            for row in query_rows.to_dict(orient="records")
        ],
    }
    covariate_plaintext = {
        "schema": "cfeg.metadata-calibration-query-analysis-covariates.v1",
        "candidate_id": "metadata-calibration-efficiency-v1",
        "phase": phase,
        "checkpoint_group": checkpoint_group,
        "rows": [
            {
                "query_token": str(row["row_token"]),
                "headband_order": _nullable_string(row.get("headband_order")),
                "condition_period": _nullable_string(row.get("condition_period")),
                "block_number": _block_number(str(row["run_id"])),
            }
            for row in query_rows.to_dict(orient="records")
        ],
    }
    for name, plaintext in (
        ("query_labels.jsonl.enc", label_plaintext),
        ("query_analysis_covariates.jsonl.enc", covariate_plaintext),
    ):
        relative_path = f"{checkpoint_group}/target_query/{name}"
        aad = {
            "candidate_id": "metadata-calibration-efficiency-v1",
            "phase": phase,
            "checkpoint_group": checkpoint_group,
            "input_seal_decision_signed_record_sha256": (
                decision_signed_record_sha256
            ),
            "token_secret_commitment_sha256": token_secret_commitment_sha256,
            "artifact_relative_path": relative_path,
            "plaintext_schema": plaintext["schema"],
            "finalizer_public_key_fingerprint_sha256": (
                finalizer_public_key_fingerprint_sha256
            ),
        }
        encrypted = encrypt_json_for_finalizer(
            plaintext,
            public_key_path=finalizer_public_key_path,
            associated_data=aad,
        )
        if encrypted.get("public_key_fingerprint_sha256") != (
            finalizer_public_key_fingerprint_sha256
        ):
            raise ValueError("Finalizer public key changed during input encryption.")
        encrypted_path = query_dir / name
        _write_json_file(encrypted_path, encrypted, mode=0o400)

    block_context = _build_block_context(target_rows)
    block_context_path = group_root / "block_context.jsonl"
    _write_jsonl_file(block_context_path, block_context)

    for path, kind in (
        (support_h5, "target_support_signals"),
        (support_view_path, "target_support_model_view"),
        (support_baseline_view_path, "target_support_baseline_view"),
        (support_label_path, "target_support_labels"),
        (query_h5, "target_query_signals"),
        (query_view_path, "target_query_model_view"),
        (query_baseline_view_path, "target_query_baseline_view"),
        (expected_path, "target_query_expected_unlabeled"),
        (query_dir / "query_labels.jsonl.enc", "encrypted_query_labels"),
        (
            query_dir / "query_analysis_covariates.jsonl.enc",
            "encrypted_query_analysis_covariates",
        ),
        (block_context_path, "target_block_context"),
    ):
        records.append(_artifact_record(group_root.parents[0], path, kind))
    return records


def _build_model_view(
    rows: pd.DataFrame,
    *,
    raw_x: np.ndarray,
    raw_mask: np.ndarray,
    checkpoint_group: str,
    token_field: str,
    include_label: bool,
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for local_index, row in enumerate(rows.to_dict(orient="records")):
        raw_index = _strict_int(row["h5_index"], name="raw h5_index")
        x = np.asarray(raw_x[raw_index], dtype=np.float32)
        mask = np.asarray(raw_mask[raw_index], dtype=bool)
        query_values, query_missing = _nullable_float_vector(
            row["query_signal_std_by_channel"], expected=64
        )
        impedance_values, impedance_missing = _nullable_float_vector(
            row["impedance_kohm_by_channel"], expected=64
        )
        channel_ids = _strict_int_vector(row["canonical_channel_ids"], expected=64)
        record: dict[str, Any] = {
            token_field: str(row["row_token"]),
            "signal_index": local_index,
            "subject_id": str(row["subject_id"]),
            "electrode_type": str(row["electrode_type"]),
            "checkpoint_group": checkpoint_group,
            "run_id": str(row["run_id"]),
            "canonical_channel_ids": channel_ids.tolist(),
            "sfreq_processed": float(row["sfreq_processed"]),
            "query_signal_std_by_channel": query_values.tolist(),
            "query_signal_std_missing_by_channel": query_missing.tolist(),
            "impedance_kohm_by_channel": impedance_values.tolist(),
            "impedance_missing_by_channel": impedance_missing.tolist(),
            "base_input_sha256": metadata_calibration_base_input_sha256(x, mask),
        }
        if include_label:
            record["label"] = _strict_int(row["label"], name="source label")
        expected_fields = set(_MODEL_VIEW_COLUMNS) | {token_field}
        if include_label:
            expected_fields.add("label")
        if set(record) != expected_fields:
            raise RuntimeError("Internal sealed model-view schema drifted.")
        output.append(record)
    return output


def _build_baseline_view(
    model_view: Sequence[Mapping[str, Any]],
    *,
    token_field: str,
    include_label: bool,
) -> list[dict[str, Any]]:
    expected = set(_BASELINE_VIEW_COLUMNS) | {token_field}
    if include_label:
        expected.add("label")
    output: list[dict[str, Any]] = []
    for source in model_view:
        row = {field: source[field] for field in _BASELINE_VIEW_COLUMNS}
        row[token_field] = source[token_field]
        if include_label:
            row["label"] = source["label"]
        if set(row) != expected:
            raise RuntimeError("Internal baseline-only view schema drifted.")
        output.append(row)
    return output


def _expanded_support_label_rows(
    rows: pd.DataFrame,
    *,
    support_view: Sequence[Mapping[str, Any]],
    checkpoint_group: str,
) -> list[dict[str, Any]]:
    view_by_token = {str(value["support_token"]): value for value in support_view}
    output: list[dict[str, Any]] = []
    for row in rows.to_dict(orient="records"):
        token = str(row["row_token"])
        block = _block_number(str(row["run_id"]))
        for budget in (1, 3, 5):
            if block > budget:
                continue
            output.append(
                {
                    "checkpoint_group": checkpoint_group,
                    "subject_id": str(row["subject_id"]),
                    "electrode_type": str(row["electrode_type"]),
                    "budget": budget,
                    "support_token": token,
                    "run_id": str(row["run_id"]),
                    "support_label": _strict_int(row["label"], name="support label"),
                    "base_input_sha256": str(view_by_token[token]["base_input_sha256"]),
                }
            )
    return sorted(
        output,
        key=lambda value: (
            str(value["checkpoint_group"]),
            str(value["subject_id"]),
            str(value["electrode_type"]),
            int(value["budget"]),
            str(value["support_token"]),
        ),
    )


def _build_block_context(rows: pd.DataFrame) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    key = ["subject_id", "electrode_type", "run_id"]
    for values, group in rows.groupby(key, sort=True):
        if len(group) != 12:
            raise ValueError("Every target block must contain exactly twelve classes.")
        vectors = [
            _nullable_float_vector(value, expected=64)
            for value in group["impedance_kohm_by_channel"]
        ]
        first_values, first_missing = vectors[0]
        if any(
            not np.array_equal(first_values, value)
            or not np.array_equal(first_missing, missing)
            for value, missing in vectors[1:]
        ):
            raise ValueError("Block-level impedance changed within a class-complete block.")
        active = np.asarray(WEARABLE_CALIBRATION_CHANNEL_IDS, dtype=int) - 1
        selected = first_values[active]
        if first_missing[active].any():
            raise ValueError("Official wearable block impedance is unexpectedly missing.")
        output.append(
            {
                "dataset_id": "wearable",
                "subject_id": str(values[0]),
                "electrode_type": str(values[1]),
                "run_id": str(values[2]),
                "impedance_channel_ids": list(WEARABLE_CALIBRATION_CHANNEL_IDS),
                "impedance_kohm_by_channel": selected.tolist(),
            }
        )
    return output


def _validate_complete_target_rows(
    support: pd.DataFrame,
    query: pd.DataFrame,
    target_subjects: Sequence[str],
) -> None:
    if set(support["row_token"].astype(str)) & set(query["row_token"].astype(str)):
        raise ValueError("Support and query opaque tokens overlap.")
    if len(support) != len(target_subjects) * 2 * 5 * 12 or len(query) != (
        len(target_subjects) * 2 * 5 * 12
    ):
        raise ValueError("Target support/query row count differs from the exact block contract.")
    for frame, runs, prefix in (
        (support, range(1, 6), "s_"),
        (query, range(6, 11), "q_"),
    ):
        if not frame["row_token"].astype(str).str.startswith(prefix).all():
            raise ValueError("Sealed target token prefix drifted.")
        for (subject, interface, run), group in frame.groupby(
            ["subject_id", "electrode_type", "run_id"], sort=False
        ):
            if str(interface) not in {"dry", "wet"} or _block_number(str(run)) not in runs:
                raise ValueError("Sealed target participant/interface block is invalid.")
            labels = sorted(_strict_int(value, name="target label") for value in group["label"])
            if len(group) != 12 or labels != list(range(12)):
                raise ValueError(
                    f"Target {subject}/{interface}/{run} is not one class-complete block."
                )


def _read_authorized_manifest(
    path: Path, *, authorized_subjects: Sequence[str]
) -> pd.DataFrame:
    required = [
        "sample_id",
        "h5_index",
        "dataset_id",
        "subject_id",
        "electrode_type",
        "run_id",
        "label",
        "sfreq_processed",
        "canonical_channel_ids",
        "query_signal_std_by_channel",
        "impedance_kohm_by_channel",
        "headband_order",
        "condition_period",
    ]
    subjects = tuple(sorted(str(value) for value in authorized_subjects))
    if not subjects:
        raise ValueError("An input-seal phase must authorize at least one subject.")
    # PyArrow applies the exact participant predicate at the dataset read API;
    # the returned application object never contains another cohort's labels.
    frame = pd.read_parquet(
        path,
        columns=required,
        filters=[("dataset_id", "==", "wearable"), ("subject_id", "in", list(subjects))],
        engine="pyarrow",
    )
    observed = tuple(sorted(frame["subject_id"].astype(str).unique()))
    if observed != subjects or len(frame) != len(subjects) * 240:
        raise ValueError("Authorized manifest projection differs from its exact subject scope.")
    if set(frame["dataset_id"].astype(str)) != {"wearable"}:
        raise ValueError("Authorized manifest projection contains another dataset.")
    return frame.reset_index(drop=True)


def _validate_raw_manifest_and_h5(
    manifest: pd.DataFrame,
    h5_path: Path,
    *,
    expected_subjects: Sequence[str],
) -> None:
    required = {
        "sample_id",
        "h5_index",
        "dataset_id",
        "subject_id",
        "electrode_type",
        "run_id",
        "label",
        "sfreq_processed",
        "canonical_channel_ids",
        "query_signal_std_by_channel",
        "impedance_kohm_by_channel",
        "headband_order",
        "condition_period",
    }
    missing = required - set(manifest)
    if missing:
        raise ValueError(f"Raw wearable manifest lacks sealer columns: {sorted(missing)}.")
    subjects = tuple(sorted(str(value) for value in expected_subjects))
    if (
        len(manifest) != len(subjects) * 240
        or tuple(sorted(manifest["subject_id"].astype(str).unique())) != subjects
        or manifest["sample_id"].astype(str).duplicated().any()
    ):
        raise ValueError("Authorized wearable manifest row, subject, or identity count drifted.")
    orders = manifest["headband_order"]
    periods = manifest["condition_period"]
    interfaces = manifest["electrode_type"].astype(str)
    if (
        orders.isna().any()
        or periods.isna().any()
        or set(orders.astype(str)) != {"dry", "wet"}
        or set(periods.astype(str)) != {"first", "second"}
        or set(interfaces) != {"dry", "wet"}
        or (manifest.groupby("subject_id")["headband_order"].nunique() != 1).any()
    ):
        raise ValueError("Wearable acquisition-order covariates have an invalid domain.")
    expected_periods = np.where(
        interfaces.to_numpy() == orders.astype(str).to_numpy(), "first", "second"
    )
    if not np.array_equal(periods.astype(str).to_numpy(), expected_periods):
        raise ValueError("Wearable acquisition period disagrees with headband order.")
    indices = np.asarray(
        [_strict_int(value, name="raw h5_index") for value in manifest["h5_index"]],
        dtype=int,
    )
    if len(np.unique(indices)) != len(indices) or indices.min() < 0 or indices.max() >= 24_480:
        raise ValueError("Authorized wearable HDF5 indices are invalid or duplicated.")
    with h5py.File(h5_path, "r") as handle:
        if set(handle) != {"x", "channel_mask", "y"}:
            raise ValueError("Raw wearable HDF5 dataset schema drifted.")
        if handle["x"].shape != (24_480, 64, 400) or handle[
            "x"
        ].dtype != np.dtype(np.float32):
            raise ValueError("Raw wearable x storage shape or dtype drifted.")
        if handle["channel_mask"].shape != (24_480, 64) or handle[
            "channel_mask"
        ].dtype != np.dtype(bool):
            raise ValueError("Raw wearable channel mask shape or dtype drifted.")
        if handle["y"].shape != (24_480,):
            raise ValueError("Raw wearable label dataset shape drifted.")
        order = np.argsort(indices)
        observed_labels = np.asarray(handle["y"][indices[order]], dtype=np.int64)
        expected_labels = np.asarray(
            [_strict_int(value, name="raw manifest label") for value in manifest["label"]],
            dtype=np.int64,
        )[order]
        if not np.array_equal(observed_labels, expected_labels):
            raise ValueError("Authorized manifest labels differ from selected HDF5 labels.")


def _write_sealed_hdf5(
    path: Path,
    rows: pd.DataFrame,
    *,
    raw_x: np.ndarray,
    raw_mask: np.ndarray,
) -> None:
    indices = np.asarray(
        [_strict_int(value, name="raw h5_index") for value in rows["h5_index"]],
        dtype=int,
    )
    x = np.asarray(raw_x[indices], dtype=np.float32)
    mask = np.asarray(raw_mask[indices], dtype=bool)
    if x.shape != (len(rows), 64, 400) or mask.shape != (len(rows), 64):
        raise ValueError("Sealed signal selection shape drifted.")
    inactive = ~mask
    if not np.all(x[inactive] == 0.0):
        raise ValueError("Inactive canonical channel slots must be exact zero before sealing.")
    with h5py.File(path, "x") as handle:
        handle.create_dataset("channel_mask", data=mask, compression="gzip")
        handle.create_dataset("x", data=x, compression="gzip")
        handle.flush()
    os.chmod(path, 0o400)
    _fsync_file(path)


def _write_jsonl_file(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    encoded = b"".join(canonical_json_bytes(dict(row)) + b"\n" for row in rows)
    _write_new_bytes(path, encoded, mode=0o400)


def _write_json_file(path: Path, value: Mapping[str, Any], *, mode: int) -> None:
    _write_new_bytes(path, canonical_json_bytes(value) + b"\n", mode=mode)


def _write_new_bytes(path: Path, data: bytes, *, mode: int) -> None:
    descriptor = os.open(
        path,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0),
        mode,
    )
    try:
        offset = 0
        while offset < len(data):
            offset += os.write(descriptor, data[offset:])
        os.fchmod(descriptor, mode)
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _artifact_record(root: Path, path: Path, kind: str) -> dict[str, Any]:
    relative = path.relative_to(root).as_posix()
    return {
        "relative_path": relative,
        "artifact_kind": kind,
        "size_bytes": path.stat().st_size,
        "file_sha256": _sha256_file(path),
        "semantic_sha256": _semantic_artifact_sha256(path),
    }


def _semantic_artifact_sha256(path: Path) -> str:
    if path.suffix == ".h5":
        with h5py.File(path, "r") as handle:
            digest = hashlib.sha256()
            for name in ("channel_mask", "x"):
                value = np.ascontiguousarray(handle[name][:])
                encoded_name = name.encode("ascii")
                digest.update(len(encoded_name).to_bytes(8, "big"))
                digest.update(encoded_name)
                digest.update(str(value.dtype).encode("ascii"))
                digest.update(np.asarray(value.shape, dtype=np.int64).tobytes())
                digest.update(value.tobytes())
            return digest.hexdigest()
    if path.name.endswith(".jsonl"):
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
        return canonical_json_sha256({"rows": rows})
    value = read_json_object(path)
    return canonical_json_sha256(value)


def _phase_subject_groups(
    *,
    phase: ExperimentPhase,
    recipe: Mapping[str, Any],
    plan: Mapping[str, Any],
) -> tuple[dict[str, tuple[str, ...]], dict[str, tuple[str, ...]]]:
    allocation = plan["data_contract"]["proposed_unopened_allocation"]
    if phase == "held_participant_evaluation":
        return (
            {"held": tuple(str(value) for value in allocation["source_development_subject_ids"])},
            {
                "held": tuple(
                    str(value)
                    for value in allocation["held_participant_evaluation_subject_ids"]
                )
            },
        )
    sources: dict[str, tuple[str, ...]] = {}
    targets: dict[str, tuple[str, ...]] = {}
    folds = recipe["data"]["folds"]
    for index in (0, 1, 2):
        spec = folds[index]
        group = f"fold{index}"
        targets[group] = tuple(str(value) for value in spec["outer_evaluation"])
        sources[group] = tuple(
            str(value) for value in (*spec["inner_fit"], *spec["inner_validation"])
        )
    return sources, targets


def _load_asset_receipt_metadata(processed_root: Path) -> dict[str, str]:
    receipt_path = processed_root / "wearable_v3_audit_receipt.json"
    receipt = read_json_object(receipt_path)
    if (
        receipt.get("status") != "accepted"
        or receipt.get("dataset_revision") != "wearable_v3"
        or receipt.get("n_rows") != 24_480
        or receipt.get("n_subjects") != 102
        or (receipt.get("raw_alignment") or {}).get("status")
        != "deep_verified_atol_1e-6"
    ):
        raise ValueError("Wearable audit receipt is not the accepted deep-verified asset.")
    alignment = receipt["raw_alignment"]
    metadata = alignment["processed_metadata_sha256"]
    fingerprint_payload = {
        "schema": "cfeg.wearable-v3-asset-fingerprint-bundle.v1",
        "dataset_revision": "wearable_v3",
        "n_rows": 24_480,
        "signals_file_sha256": alignment["signals_file_sha256"],
        "signals_content_sha256": alignment["signals_content_sha256"],
        "manifest_parquet_sha256": metadata["manifest.parquet"],
        "manifest_jsonl_sha256": metadata["manifest.jsonl"],
        "class_map_sha256": metadata["class_map.json"],
    }
    return {
        "raw_asset_receipt_sha256": _sha256_file(receipt_path),
        "raw_asset_fingerprint_bundle_sha256": canonical_json_sha256(
            fingerprint_payload
        ),
        "class_map_sha256": str(metadata["class_map.json"]),
        "signals_file_sha256": str(alignment["signals_file_sha256"]),
        "manifest_parquet_sha256": str(metadata["manifest.parquet"]),
        "manifest_jsonl_sha256": str(metadata["manifest.jsonl"]),
    }


def validate_source_private_epoch_artifacts(
    private_seal_path: str | Path,
    *,
    trusted_signing_public_key_path: str | Path,
) -> SourcePrivateEpochBinding:
    """Validate source epoch selection against every immutable private artifact."""

    private_seal_path = Path(private_seal_path)
    _validate_unlinked_regular_file(private_seal_path)
    private_seal = read_json_object(private_seal_path)
    private_signed_hash = verify_signed_record(
        private_seal,
        public_key_path=trusted_signing_public_key_path,
        hash_field="private_seal_payload_sha256",
        expected_schema=FILESYSTEM_PRIVATE_SEAL_SCHEMA,
    )
    staging = private_seal.get("private_staging_seal")
    if set(private_seal) != _FILESYSTEM_PRIVATE_SEAL_FIELDS or (
        private_seal.get("candidate_id") != "metadata-calibration-efficiency-v1"
        or private_seal.get("phase") != "source_development"
        or private_seal.get("status") != "complete_private_grid_signed_before_outcomes"
        or private_seal.get("complete_candidate_and_mandatory_baseline_grid") is not True
        or private_seal.get("query_outcomes_loaded") is not False
        or not isinstance(staging, Mapping)
        or staging.get("schema") != "cfeg.metadata-calibration-private-seal.v1"
        or staging.get("complete_candidate_and_mandatory_baseline_grid") is not True
        or staging.get("query_outcomes_loaded") is not False
        or staging.get("combined_private_tree_sha256")
        != private_seal.get("private_staging_semantic_sha256")
    ):
        raise PermissionError("Source filesystem private seal is invalid or incomplete.")
    for field in (
        "phase_manifest_sha256",
        "outcome_decision_signed_record_sha256",
        "authorization_envelope_sha256",
        "input_seal_receipt_signed_record_sha256",
        "input_artifact_tree_sha256",
        "execution_contract_sha256",
        "source_tree_sha256",
    ):
        if not _is_sha256(private_seal.get(field)):
            raise PermissionError(f"Source private seal has an invalid {field} binding.")
    completed = private_seal.get("completed_job_ids")
    receipt_hashes = private_seal.get("producer_receipt_file_sha256s_by_job")
    if not isinstance(completed, list) or len(completed) != int(
        private_seal.get("expected_job_count", -1)
    ) or len(set(map(str, completed))) != len(completed) or not isinstance(
        receipt_hashes, Mapping
    ) or set(map(str, receipt_hashes)) != set(map(str, completed)) or not all(
        _is_sha256(value) for value in receipt_hashes.values()
    ):
        raise PermissionError("Source private seal does not bind every exact producer receipt.")
    artifact_records = _validate_private_execution_artifacts(
        private_seal,
        completed_job_ids=tuple(map(str, completed)),
        producer_receipt_hashes=receipt_hashes,
    )
    selected_epochs = _validate_held_selected_epochs(
        private_seal.get("held_selected_epochs_by_seed")
    )
    selected_epochs_sha256 = canonical_json_sha256(selected_epochs)
    if private_seal.get("held_selected_epochs_by_seed_sha256") != (
        selected_epochs_sha256
    ):
        raise PermissionError("Source private seal has an invalid held epoch-selection digest.")
    derivation_sha256 = _validate_epoch_selection_derivation(
        private_seal.get("held_epoch_selection_derivation"),
        selected_epochs=selected_epochs,
        producer_receipt_hashes=receipt_hashes,
        private_artifact_records=artifact_records,
        execution_output_root=Path(str(private_seal["execution_output_root"])),
    )
    private_file_hash = _sha256_file(private_seal_path)
    return SourcePrivateEpochBinding(
        private_seal_file_sha256=private_file_hash,
        private_seal_signed_record_sha256=private_signed_hash,
        held_selected_epochs_by_seed=selected_epochs,
        held_selected_epochs_by_seed_sha256=selected_epochs_sha256,
        held_epoch_selection_derivation_sha256=derivation_sha256,
    )


def validate_source_gate_artifacts(
    gate_receipt_path: str | Path,
    private_seal_path: str | Path,
    *,
    trusted_signing_public_key_path: str | Path,
    physical_result_bundle_root: str | Path | None = None,
    expected_result_bundle_root: str | Path | None = None,
) -> SourceGateBinding:
    gate_receipt_path = Path(gate_receipt_path)
    private_seal_path = Path(private_seal_path)
    _validate_unlinked_regular_file(gate_receipt_path)
    private_binding = validate_source_private_epoch_artifacts(
        private_seal_path,
        trusted_signing_public_key_path=trusted_signing_public_key_path,
    )
    private_seal = read_json_object(private_seal_path)
    private_signed_hash = private_binding.private_seal_signed_record_sha256
    selected_epochs = private_binding.held_selected_epochs_by_seed
    selected_epochs_sha256 = private_binding.held_selected_epochs_by_seed_sha256
    derivation_sha256 = private_binding.held_epoch_selection_derivation_sha256
    private_file_hash = private_binding.private_seal_file_sha256

    gate = read_json_object(gate_receipt_path)
    gate_signed_hash = verify_signed_record(
        gate,
        public_key_path=trusted_signing_public_key_path,
        hash_field="gate_payload_sha256",
        expected_schema=DEVELOPMENT_GATE_RECEIPT_SCHEMA,
    )
    if set(gate) != _DEVELOPMENT_GATE_RECEIPT_FIELDS or (
        gate.get("candidate_id") != "metadata-calibration-efficiency-v1"
        or gate.get("phase") != "source_development"
        or gate.get("development_gate_status") != "pass"
        or gate.get("filesystem_private_seal_file_sha256") != private_file_hash
        or gate.get("filesystem_private_seal_signed_record_sha256")
        != private_signed_hash
        or gate.get("filesystem_private_tree_sha256")
        != private_seal.get("filesystem_private_tree_sha256")
        or gate.get("source_phase_manifest_sha256")
        != private_seal.get("phase_manifest_sha256")
        or gate.get("source_outcome_decision_signed_record_sha256")
        != private_seal.get("outcome_decision_signed_record_sha256")
        or gate.get("source_input_seal_receipt_signed_record_sha256")
        != private_seal.get("input_seal_receipt_signed_record_sha256")
        or gate.get("source_input_artifact_tree_sha256")
        != private_seal.get("input_artifact_tree_sha256")
        or gate.get("source_execution_contract_sha256")
        != private_seal.get("execution_contract_sha256")
        or gate.get("source_execution_output_root")
        != private_seal.get("execution_output_root")
        or gate.get("source_commit") != private_seal.get("source_commit")
        or gate.get("source_tree_sha256") != private_seal.get("source_tree_sha256")
        or gate.get("source_tag") != private_seal.get("source_tag")
        or gate.get("outcome_finalization_performed") is not True
        or gate.get("held_input_access_authorized") is not True
        or not isinstance(gate.get("source_result_bundle_root"), str)
        or not _is_sha256(gate.get("source_result_bundle_sha256"))
        or not isinstance(gate.get("development_gate_result"), Mapping)
        or gate["development_gate_result"].get("passed") is not True
        or gate["development_gate_result"].get("decision") != "development_go"
        or gate["development_gate_result"].get(
            "held_evaluation_access_allowed"
        ) is not True
        or gate["development_gate_result"].get("held_selected_epochs_by_seed")
        != selected_epochs
        or gate["development_gate_result"].get(
            "held_selected_epochs_by_seed_sha256"
        )
        != selected_epochs_sha256
        or gate["development_gate_result"].get(
            "held_epoch_selection_derivation_sha256"
        )
        != derivation_sha256
    ):
        raise PermissionError("Source development gate is not an exact bound PASS receipt.")
    outcome_claim, _ = validate_outcome_access_claim_evidence(
        str(gate["outcome_access_claim_path"]),
        expected_phase="source_development",
        signed_private_seal=private_seal,
        private_seal_path=private_seal_path,
        trusted_signing_public_key_path=trusted_signing_public_key_path,
    )
    if (
        gate.get("outcome_access_claim_file_sha256")
        != outcome_claim.claim_file_sha256
        or gate.get("outcome_access_claim_signed_record_sha256")
        != outcome_claim.signed_record_sha256
        or gate.get("outcome_access_scope_sha256")
        != outcome_claim.outcome_access_scope_sha256
    ):
        raise PermissionError("Source gate does not bind its one-shot outcome claim.")
    declared_result_root = _absolute_no_symlink_path(
        Path(str(gate["source_result_bundle_root"])), must_exist=False
    )
    expected_declared_root = (
        declared_result_root
        if expected_result_bundle_root is None
        else _absolute_no_symlink_path(Path(expected_result_bundle_root), must_exist=False)
    )
    if declared_result_root != expected_declared_root:
        raise PermissionError("Source gate declares an unexpected canonical result root.")
    physical_result_root = (
        declared_result_root
        if physical_result_bundle_root is None
        else _absolute_no_symlink_path(Path(physical_result_bundle_root), must_exist=True)
    )
    observed_result_hash = validate_metadata_calibration_result_bundle(
        physical_result_root,
        completion_receipt_path=gate_receipt_path,
        expected_phase="source_development",
    )
    if observed_result_hash != gate["source_result_bundle_sha256"]:
        raise PermissionError("Source gate does not bind its physically published result bundle.")
    return SourceGateBinding(
        gate_receipt_file_sha256=_sha256_file(gate_receipt_path),
        private_seal_file_sha256=private_file_hash,
        gate_signed_record_sha256=gate_signed_hash,
        private_seal_signed_record_sha256=private_signed_hash,
        held_selected_epochs_by_seed=selected_epochs,
        held_selected_epochs_by_seed_sha256=selected_epochs_sha256,
    )


def validate_metadata_calibration_result_bundle(
    result_bundle_root: str | Path,
    *,
    completion_receipt_path: str | Path,
    expected_phase: ExperimentPhase,
    expected_bindings: Mapping[str, str] | None = None,
) -> str:
    """Validate one atomically published aggregate-only result directory."""

    root = _absolute_no_symlink_path(Path(result_bundle_root), must_exist=True)
    completion = _absolute_no_symlink_path(Path(completion_receipt_path), must_exist=True)
    if completion != root / "completion_receipt.json":
        raise PermissionError("Completion receipt must be inside its atomic result directory.")
    observed_root = root.lstat()
    if not stat.S_ISDIR(observed_root.st_mode) or stat.S_IMODE(observed_root.st_mode) != 0o700:
        raise PermissionError("Published result root must be one mode-0700 directory.")

    manifest_path = root / "result_bundle_manifest.json"
    _validate_unlinked_regular_file(manifest_path)
    manifest = read_json_object(manifest_path)
    exact_manifest_fields = {
        "schema",
        "candidate_id",
        "phase",
        "query_outcomes_loaded",
        "query_level_outputs_published",
        "completion_receipt_excluded_from_result_content_hash",
        "files",
        "result_bundle_sha256",
    }
    records = manifest.get("files")
    if set(manifest) != exact_manifest_fields or (
        manifest.get("schema") != "cfeg.metadata-calibration-result-bundle.v1"
        or manifest.get("candidate_id") != "metadata-calibration-efficiency-v1"
        or manifest.get("phase") != expected_phase
        or manifest.get("query_outcomes_loaded") is not True
        or manifest.get("query_level_outputs_published") is not False
        or manifest.get("completion_receipt_excluded_from_result_content_hash") is not True
        or not isinstance(records, list)
        or not records
    ):
        raise PermissionError("Published result manifest has an invalid exact schema.")
    payload = dict(manifest)
    claimed_hash = payload.pop("result_bundle_sha256", None)
    if not _is_sha256(claimed_hash) or claimed_hash != canonical_json_sha256(payload):
        raise PermissionError("Published result bundle content hash is invalid.")

    expected_files: set[str] = set()
    for record in records:
        if not isinstance(record, Mapping) or set(record) != {
            "relative_path",
            "size_bytes",
            "file_sha256",
        }:
            raise PermissionError("Published result file record has an invalid exact schema.")
        relative = str(record["relative_path"])
        pure = Path(relative)
        if (
            pure.is_absolute()
            or len(pure.parts) != 1
            or relative in {
                "result_bundle_manifest.json",
                "completion_receipt.json",
            }
            or relative in expected_files
        ):
            raise PermissionError("Published result file path is not a unique immediate child.")
        expected_files.add(relative)
        path = root / relative
        _validate_unlinked_regular_file(path)
        observed = path.lstat()
        if (
            stat.S_IMODE(observed.st_mode) != 0o400
            or type(record.get("size_bytes")) is not int
            or observed.st_size != record["size_bytes"]
            or not _is_sha256(record.get("file_sha256"))
            or _sha256_file(path) != record["file_sha256"]
        ):
            raise PermissionError("Published result artifact hash, size, or mode drifted.")
    if expected_files != _RESULT_ANALYSIS_FILES:
        raise PermissionError("Published result files differ from the aggregate-only allowlist.")

    actual_files: set[str] = set()
    for path in root.iterdir():
        _validate_unlinked_regular_file(path)
        if stat.S_IMODE(path.lstat().st_mode) != 0o400:
            raise PermissionError("Every published result file must use mode 0400.")
        actual_files.add(path.name)
    if actual_files != {
        *expected_files,
        "result_bundle_manifest.json",
        "completion_receipt.json",
    }:
        raise PermissionError("Published result directory has missing or undeclared files.")
    completion_record = read_json_object(completion)
    _validate_published_result_analysis(
        root,
        phase=expected_phase,
        completion_record=completion_record,
        expected_bindings=expected_bindings,
    )
    return str(claimed_hash)


def _validate_published_result_analysis(
    root: Path,
    *,
    phase: ExperimentPhase,
    completion_record: Mapping[str, Any],
    expected_bindings: Mapping[str, str] | None,
) -> None:
    expected_n = 39 if phase == "source_development" else 60
    csv_contracts = {
        "candidate_participant_balanced_accuracy.csv": (
            _CANDIDATE_PARTICIPANT_BA_COLUMNS
        ),
        "candidate_participant_contrasts.csv": _PARTICIPANT_CONTRAST_COLUMNS,
        "baseline_participant_balanced_accuracy.csv": (
            _BASELINE_PARTICIPANT_BA_COLUMNS
        ),
        "baseline_summary.csv": _BASELINE_SUMMARY_COLUMNS,
    }
    frames: dict[str, pd.DataFrame] = {}
    for name, columns in csv_contracts.items():
        frame = pd.read_csv(root / name, float_precision="round_trip")
        if set(frame) != columns or frame.empty:
            raise PermissionError(f"Published aggregate CSV has a wrong schema: {name}.")
        forbidden = {"query_token", "label", "prediction", "correct"}.intersection(frame)
        if forbidden:
            raise PermissionError("Published aggregate CSV contains query-level columns.")
        frames[name] = frame
    contrast = frames["candidate_participant_contrasts.csv"]
    if len(contrast) != expected_n or contrast["subject_id"].astype(str).duplicated().any():
        raise PermissionError("Published participant contrast table has a wrong cohort size.")
    candidate_ba = frames["candidate_participant_balanced_accuracy.csv"]
    baseline_ba = frames["baseline_participant_balanced_accuracy.csv"]
    contrast_subjects = set(contrast["subject_id"].astype(str))
    if set(candidate_ba["subject_id"].astype(str)) != contrast_subjects or set(
        baseline_ba["subject_id"].astype(str)
    ) != contrast_subjects:
        raise PermissionError("Published participant aggregates have different cohorts.")
    candidate_cells = {
        ("A_0", "exact_off"),
        ("A_M", "correct"),
        ("A_Q", "exact_off"),
        ("A_QM", "all_missing"),
        ("A_QM", "block_shuffle"),
        ("A_QM", "correct"),
        ("A_QM", "opposite_interface"),
        ("A_QM", "stale"),
    }
    if phase == "source_development":
        candidate_cells.update(
            {
                ("A_QM", "impedance_only"),
                ("A_QM", "interface_only"),
                ("A_QM", "query_metadata_only"),
                ("A_QM", "support_metadata_only"),
            }
        )
    candidate_key = ["role", "context", "budget", "subject_id"]
    expected_candidate = {
        (role, context, budget, subject)
        for role, context in candidate_cells
        for budget in (0, 1, 3, 5)
        for subject in contrast_subjects
    }
    observed_candidate = set(candidate_ba[candidate_key].itertuples(index=False, name=None))
    if candidate_ba.duplicated(candidate_key).any() or observed_candidate != expected_candidate:
        raise PermissionError("Published candidate aggregate is not the exact cell cross-product.")
    baseline_budgets = {
        "strict_FBCCA": {0},
        "target_template_correlation": {1, 3, 5},
        "target_filterbank_eTRCA": {3, 5},
        "same3_filterbank_eTRCA": {1},
        "chiang2021_LST_filterbank_eTRCA": {1, 3, 5},
    }
    baseline_key = ["adapter_id", "budget", "subject_id"]
    expected_baseline = {
        (adapter, budget, subject)
        for adapter, budgets in baseline_budgets.items()
        for budget in budgets
        for subject in contrast_subjects
    }
    observed_baseline = set(baseline_ba[baseline_key].itertuples(index=False, name=None))
    if baseline_ba.duplicated(baseline_key).any() or observed_baseline != expected_baseline:
        raise PermissionError("Published baseline aggregate is not the exact method-budget grid.")
    summary = frames["baseline_summary.csv"]
    summary_key = ["adapter_id", "budget"]
    expected_summary = {
        (adapter, budget) for adapter, budgets in baseline_budgets.items() for budget in budgets
    }
    if summary.duplicated(summary_key).any() or set(
        summary[summary_key].itertuples(index=False, name=None)
    ) != expected_summary or not summary["n_participants"].eq(expected_n).all():
        raise PermissionError("Published baseline summary is not the exact method-budget grid.")
    for name, frame in frames.items():
        numeric = frame.drop(columns=[column for column in ("subject_id", "role", "context", "adapter_id") if column in frame])
        values = numeric.apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float64)
        if not np.isfinite(values).all():
            raise PermissionError(f"Published aggregate CSV contains non-finite data: {name}.")
    for frame, column in (
        (candidate_ba, "participant_balanced_accuracy"),
        (baseline_ba, "participant_balanced_accuracy"),
    ):
        values = frame[column].to_numpy(dtype=np.float64)
        if (values < 0.0).any() or (values > 1.0).any():
            raise PermissionError("Published balanced accuracy lies outside [0,1].")
    contrast_values = contrast.drop(columns="subject_id").to_numpy(dtype=np.float64)
    if (contrast_values < -1.0 - 1e-12).any() or (contrast_values > 1.0 + 1e-12).any():
        raise PermissionError("Published participant contrast lies outside its mathematical range.")

    from cfeg.analysis.metadata_calibration_baselines import (
        summarize_baseline_participant_accuracy,
    )
    from cfeg.analysis.metadata_calibration_efficiency import (
        derive_participant_contrasts,
        metadata_calibration_bundle_spec_for_phase,
    )

    spec = metadata_calibration_bundle_spec_for_phase(phase=phase)
    recomputed_contrast = derive_participant_contrasts(candidate_ba, spec=spec)
    observed_contrast = contrast.sort_values("subject_id", kind="mergesort").reset_index(
        drop=True
    )
    recomputed_contrast = recomputed_contrast.sort_values(
        "subject_id", kind="mergesort"
    ).reset_index(drop=True)
    contrast_columns = list(recomputed_contrast.columns)
    if (
        set(contrast_columns) != _PARTICIPANT_CONTRAST_COLUMNS
        or not observed_contrast["subject_id"].astype(str).equals(
            recomputed_contrast["subject_id"].astype(str)
        )
        or not np.array_equal(
            observed_contrast.loc[:, contrast_columns[1:]].to_numpy(dtype=np.float64),
            recomputed_contrast.loc[:, contrast_columns[1:]].to_numpy(dtype=np.float64),
        )
    ):
        raise PermissionError(
            "Published participant contrasts do not derive exactly from participant BA."
        )

    recomputed_summary = summarize_baseline_participant_accuracy(baseline_ba)
    observed_summary = summary.sort_values(
        ["adapter_id", "budget"], kind="mergesort"
    ).reset_index(drop=True)
    recomputed_summary = recomputed_summary.sort_values(
        ["adapter_id", "budget"], kind="mergesort"
    ).reset_index(drop=True)
    summary_columns = list(recomputed_summary.columns)
    if (
        set(summary_columns) != _BASELINE_SUMMARY_COLUMNS
        or not observed_summary[["adapter_id", "budget"]].equals(
            recomputed_summary[["adapter_id", "budget"]]
        )
        or not np.array_equal(
            observed_summary.loc[:, summary_columns[2:]].to_numpy(dtype=np.float64),
            recomputed_summary.loc[:, summary_columns[2:]].to_numpy(dtype=np.float64),
        )
    ):
        raise PermissionError(
            "Published baseline summary does not derive exactly from participant BA."
        )

    analysis = read_json_object(root / "analysis_metadata.json")
    if set(analysis) != {
        "schema",
        "phase",
        "candidate",
        "baseline",
        "bindings",
        "query_level_outputs_published",
    } or (
        analysis.get("schema")
        != "cfeg.metadata-calibration-published-analysis-metadata.v1"
        or analysis.get("phase") != phase
        or analysis.get("query_level_outputs_published") is not False
        or not isinstance(analysis.get("candidate"), Mapping)
        or not isinstance(analysis.get("baseline"), Mapping)
        or not isinstance(analysis.get("bindings"), Mapping)
    ):
        raise PermissionError("Published analysis metadata has a wrong aggregate schema.")
    candidate_metadata = analysis["candidate"]
    baseline_metadata = analysis["baseline"]
    if set(candidate_metadata) != {
        "schema",
        "candidate_id",
        "phase",
        "bindings",
        "seed_reduction",
        "tie_break",
        "missing_fallback_probability_exact",
    } or candidate_metadata.get("schema") != "cfeg.metadata-calibration-analysis-bundle.v1" or (
        candidate_metadata.get("candidate_id") != "metadata-calibration-efficiency-v1"
        or candidate_metadata.get("phase") != phase
        or candidate_metadata.get("seed_reduction")
        != "mean_probabilities_per_sample_before_argmax"
        or candidate_metadata.get("tie_break") != "lowest_canonical_class_id"
        or candidate_metadata.get("missing_fallback_probability_exact") is not True
    ) or set(baseline_metadata) != {
        "schema",
        "candidate_id",
        "phase",
        "bindings",
        "score_interpretation",
        "seed_reduction",
        "tie_break",
        "primary_inference_role",
    } or baseline_metadata.get("schema") != (
        "cfeg.metadata-calibration-baseline-analysis-bundle.v1"
    ) or baseline_metadata.get("candidate_id") != (
        "metadata-calibration-efficiency-v1"
    ) or baseline_metadata.get("phase") != phase or baseline_metadata.get(
        "score_interpretation"
    ) != "uncalibrated_raw_class_score_argmax_only" or baseline_metadata.get(
        "seed_reduction"
    ) != "none_deterministic_once" or baseline_metadata.get(
        "tie_break"
    ) != "lowest_zero_based_class_id" or baseline_metadata.get(
        "primary_inference_role"
    ) != "descriptive_resource_matched_context_only":
        raise PermissionError("Published candidate/baseline metadata schema drifted.")
    if not (
        analysis["bindings"]
        == candidate_metadata.get("bindings")
        == baseline_metadata.get("bindings")
    ):
        raise PermissionError("Published analysis provenance bindings disagree.")
    observed_bindings = analysis["bindings"]
    binding_fields = {
        "plan_sha256",
        "execution_manifest_sha256",
        "decision_receipt_sha256",
        "asset_receipt_sha256",
        "asset_fingerprint_bundle_sha256",
        "class_map_sha256",
        "source_tree_sha256",
    }
    if set(observed_bindings) != binding_fields or not all(
        _is_sha256(value) for value in observed_bindings.values()
    ):
        raise PermissionError("Published analysis provenance binding schema is invalid.")
    if expected_bindings is not None and dict(observed_bindings) != dict(
        expected_bindings
    ):
        raise PermissionError("Published analysis provenance differs from phase authority.")

    covariates = read_json_object(root / "analysis_covariate_summary.json")
    if set(covariates) != {
        "schema",
        "n_query_rows",
        "query_tokens_published",
        "headband_order_counts",
        "condition_period_counts",
        "block_number_counts",
    } or (
        covariates.get("schema")
        != "cfeg.metadata-calibration-analysis-covariate-summary.v1"
        or covariates.get("n_query_rows") != expected_n * 2 * 60
        or covariates.get("query_tokens_published") is not False
    ):
        raise PermissionError("Published covariate summary has a wrong aggregate schema.")
    headband_counts = covariates.get("headband_order_counts")
    period_counts = covariates.get("condition_period_counts")
    block_counts = covariates.get("block_number_counts")
    if (
        not isinstance(headband_counts, Mapping)
        or set(headband_counts) != {"dry", "wet"}
        or any(type(value) is not int or value <= 0 or value % 120 != 0 for value in headband_counts.values())
        or sum(headband_counts.values()) != expected_n * 120
        or not isinstance(period_counts, Mapping)
        or period_counts != {"first": expected_n * 60, "second": expected_n * 60}
        or not isinstance(block_counts, Mapping)
        or block_counts
        != {str(block): expected_n * 24 for block in range(6, 11)}
    ):
        raise PermissionError("Published covariate counts violate the closed design domain.")

    gate_or_claim = read_json_object(root / "gate_or_claim_result.json")
    expected_result = (
        completion_record.get("development_gate_result")
        if phase == "source_development"
        else completion_record.get("held_claim_result")
    )
    expected_schema = (
        "cfeg.metadata-calibration-development-gate.v1"
        if phase == "source_development"
        else "cfeg.metadata-calibration-held-claims.v1"
    )
    if (
        not isinstance(expected_result, Mapping)
        or gate_or_claim != dict(expected_result)
        or gate_or_claim.get("schema") != expected_schema
    ):
        raise PermissionError("Published gate/claim result differs from its signed completion receipt.")
    from cfeg.analysis.metadata_calibration_efficiency import (
        evaluate_development_gate,
        evaluate_held_claim_sequence,
    )
    if phase == "source_development":
        recipe = load_config(
            _REPOSITORY / METADATA_CALIBRATION_SOURCE_RECIPE,
            strict_env=False,
        )
        outer_fold_by_subject = {
            str(subject): fold
            for fold in (0, 1, 2)
            for subject in recipe["data"]["folds"][fold]["outer_evaluation"]
        }
        recomputed = evaluate_development_gate(
            contrast,
            outer_fold_by_subject=outer_fold_by_subject,
            missing_fallback_probability_exact=bool(
                candidate_metadata["missing_fallback_probability_exact"]
            ),
            metadata_only_structural_chance_exact=True,
            expected_n=39,
            spec=spec,
        )
        epoch_fields = {
            "held_selected_epochs_by_seed",
            "held_selected_epochs_by_seed_sha256",
            "held_epoch_selection_derivation_sha256",
        }
        if set(gate_or_claim) != {*recomputed, *epoch_fields} or any(
            gate_or_claim.get(key) != value for key, value in recomputed.items()
        ):
            raise PermissionError("Published source gate does not recompute from its contrasts.")
        if not isinstance(gate_or_claim.get("held_selected_epochs_by_seed"), Mapping) or not (
            _is_sha256(gate_or_claim.get("held_selected_epochs_by_seed_sha256"))
            and _is_sha256(gate_or_claim.get("held_epoch_selection_derivation_sha256"))
        ):
            raise PermissionError("Published source gate has an invalid epoch-selection binding.")
    else:
        recomputed = evaluate_held_claim_sequence(
            contrast,
            expected_n=60,
            spec=spec,
        )
        if gate_or_claim != recomputed:
            raise PermissionError("Published held claims do not recompute from their contrasts.")


def _validate_held_selected_epochs(value: object) -> dict[str, dict[str, int]]:
    if not isinstance(value, Mapping) or set(map(str, value)) != {"42", "43", "44"}:
        raise PermissionError("Held epoch selection must bind exact seeds 42, 43, and 44.")
    output: dict[str, dict[str, int]] = {}
    for seed in (42, 43, 44):
        observed = value.get(str(seed))
        if not isinstance(observed, Mapping) or set(observed) != {
            "stage1_epochs",
            "stage2_epochs",
        }:
            raise PermissionError("Held epoch selection has an unexpected schema.")
        stage1 = observed.get("stage1_epochs")
        stage2 = observed.get("stage2_epochs")
        if type(stage1) is not int or not 10 <= stage1 <= 40:
            raise PermissionError("Held Stage-1 epoch selection is outside [10,40].")
        if type(stage2) is not int or not 5 <= stage2 <= 25:
            raise PermissionError("Held Stage-2 epoch selection is outside [5,25].")
        output[str(seed)] = {
            "stage1_epochs": stage1,
            "stage2_epochs": stage2,
        }
    return output


def _validate_private_execution_artifacts(
    private_seal: Mapping[str, Any],
    *,
    completed_job_ids: Sequence[str],
    producer_receipt_hashes: Mapping[str, Any],
) -> tuple[dict[str, Any], ...]:
    records = private_seal.get("private_artifact_records")
    if not isinstance(records, list) or not records or any(
        not isinstance(record, Mapping)
        or set(record) != _PRIVATE_ARTIFACT_RECORD_FIELDS
        for record in records
    ):
        raise PermissionError("Private execution artifacts use an invalid exact schema.")
    normalized = tuple(
        dict(record)
        for record in sorted(records, key=lambda item: str(item["relative_path"]))
    )
    paths = [str(record["relative_path"]) for record in normalized]
    if len(paths) != len(set(paths)):
        raise PermissionError("Private execution artifacts repeat a relative path.")
    expected = _expected_source_private_artifacts()
    observed = {
        str(record["relative_path"]): {
            "job_id": str(record["job_id"]),
            "artifact_kind": str(record["artifact_kind"]),
        }
        for record in normalized
    }
    if set(completed_job_ids) != {
        value["job_id"] for value in expected.values()
    } or observed != expected:
        raise PermissionError("Source private artifacts differ from the exact 24-job manifest.")
    root = _absolute_no_symlink_path(
        Path(str(private_seal.get("execution_output_root"))), must_exist=True
    )
    if stat.S_IMODE(root.lstat().st_mode) != 0o700:
        raise PermissionError("Private execution output root must use mode 0700.")
    actual_paths: set[str] = set()
    for path in sorted(root.rglob("*")):
        observed_stat = path.lstat()
        if stat.S_ISDIR(observed_stat.st_mode):
            if stat.S_IMODE(observed_stat.st_mode) != 0o700:
                raise PermissionError("Private execution directories must use mode 0700.")
            continue
        if not stat.S_ISREG(observed_stat.st_mode) or observed_stat.st_nlink != 1:
            raise PermissionError("Private execution tree contains a special or linked file.")
        actual_paths.add(path.relative_to(root).as_posix())
    if actual_paths != set(paths):
        raise PermissionError("Private execution tree has missing or undeclared files.")
    receipt_records: dict[str, dict[str, Any]] = {}
    for record in normalized:
        relative = str(record["relative_path"])
        pure = Path(relative)
        if pure.is_absolute() or ".." in pure.parts or record.get("mode_octal") != "0400":
            raise PermissionError("Private execution artifact path or mode is invalid.")
        path = root / pure
        observed_stat = path.lstat()
        if stat.S_IMODE(observed_stat.st_mode) != 0o400 or (
            observed_stat.st_size != record.get("size_bytes")
        ) or _sha256_file(path) != record.get("file_sha256"):
            raise PermissionError("Private execution artifact hash, size, or mode drifted.")
        if record["artifact_kind"] == "producer_receipt":
            job_id = str(record["job_id"])
            if record["file_sha256"] != producer_receipt_hashes.get(job_id):
                raise PermissionError("Private producer-receipt file hash binding drifted.")
            receipt = read_json_object(path)
            claimed = receipt.get("producer_receipt_sha256")
            payload = dict(receipt)
            payload.pop("producer_receipt_sha256", None)
            if (
                receipt.get("job_id") != job_id
                or receipt.get("phase") != "source_development"
                or receipt.get("candidate_id")
                != "metadata-calibration-efficiency-v1"
                or receipt.get("query_outcomes_loaded") is not False
                or receipt.get("raw_sample_ids_loaded") is not False
                or claimed != canonical_json_sha256(payload)
            ):
                raise PermissionError("Private producer receipt content is invalid.")
            receipt_records[job_id] = receipt
    if set(receipt_records) != set(completed_job_ids):
        raise PermissionError("Private execution tree lacks one producer receipt per job.")
    expected_tree = canonical_json_sha256({"artifacts": list(normalized)})
    if private_seal.get("filesystem_private_tree_sha256") != expected_tree:
        raise PermissionError("Private filesystem artifact-tree digest is invalid.")
    return normalized


def _expected_source_private_artifacts() -> dict[str, dict[str, str]]:
    output: dict[str, dict[str, str]] = {}
    candidate_kinds = {
        "probabilities.parquet": "candidate_probability_grid",
        "composite.safetensors": "full_composite_checkpoint",
        "composite.receipt.json": "full_composite_checkpoint_receipt",
        "training_selection.json": "epoch_selection_record",
        "producer_receipt.json": "producer_receipt",
    }
    baseline_kinds = {
        "scores.parquet": "baseline_raw_score_grid",
        "producer_receipt.json": "producer_receipt",
    }
    adapters = (
        "strict_FBCCA",
        "target_template_correlation",
        "target_filterbank_eTRCA",
        "same3_filterbank_eTRCA",
        "chiang2021_LST_filterbank_eTRCA",
    )
    for fold in ("fold0", "fold1", "fold2"):
        for seed in (42, 43, 44):
            job_id = f"candidate-{fold}-seed{seed}"
            for name, kind in candidate_kinds.items():
                output[f"candidate/{fold}/seed{seed}/{name}"] = {
                    "job_id": job_id,
                    "artifact_kind": kind,
                }
        for adapter in adapters:
            job_id = f"baseline-{adapter}-{fold}"
            for name, kind in baseline_kinds.items():
                output[f"baseline/{adapter}/{fold}/{name}"] = {
                    "job_id": job_id,
                    "artifact_kind": kind,
                }
    return output


def _validate_epoch_selection_derivation(
    value: object,
    *,
    selected_epochs: Mapping[str, Mapping[str, int]],
    producer_receipt_hashes: Mapping[str, Any],
    private_artifact_records: Sequence[Mapping[str, Any]],
    execution_output_root: Path,
) -> str:
    from cfeg.metadata_calibration_job import _validate_source_selection_history

    recipe = load_config(
        _REPOSITORY / METADATA_CALIBRATION_SOURCE_RECIPE,
        strict_env=False,
    )
    if not isinstance(value, Mapping) or set(value) != {
        "schema",
        "rule",
        "seeds",
        "held_selected_epochs_by_seed_sha256",
    } or value.get("schema") != (
        "cfeg.metadata-calibration-held-epoch-selection-derivation.v1"
    ) or value.get("rule") != (
        "per_seed_integer_median_across_exact_fold0_fold1_fold2_inner_selected_epochs"
    ) or value.get("held_selected_epochs_by_seed_sha256") != canonical_json_sha256(
        selected_epochs
    ):
        raise PermissionError("Held epoch-selection derivation has an invalid schema.")
    seeds = value.get("seeds")
    if not isinstance(seeds, Mapping) or set(map(str, seeds)) != {"42", "43", "44"}:
        raise PermissionError("Held epoch derivation must contain exact seeds 42,43,44.")
    for seed in (42, 43, 44):
        observed = seeds.get(str(seed))
        if not isinstance(observed, Mapping) or set(observed) != {"folds", "derived"}:
            raise PermissionError("Held epoch derivation seed record has a wrong schema.")
        folds = observed.get("folds")
        if not isinstance(folds, Mapping) or set(folds) != {"fold0", "fold1", "fold2"}:
            raise PermissionError("Held epoch derivation lacks one exact outer fold.")
        stage1_values: list[int] = []
        stage2_values: list[int] = []
        for fold in ("fold0", "fold1", "fold2"):
            record = folds.get(fold)
            if not isinstance(record, Mapping) or set(record) != {
                "job_id",
                "producer_receipt_file_sha256",
                "training_selection_file_sha256",
                "stage1_epochs",
                "stage2_epochs",
            }:
                raise PermissionError("Held epoch fold contributor has a wrong schema.")
            job_id = f"candidate-{fold}-seed{seed}"
            if record.get("job_id") != job_id or record.get(
                "producer_receipt_file_sha256"
            ) != producer_receipt_hashes.get(job_id):
                raise PermissionError("Held epoch contributor receipt binding is invalid.")
            matching = [
                candidate
                for candidate in private_artifact_records
                if candidate.get("job_id") == job_id
                and candidate.get("artifact_kind") == "epoch_selection_record"
            ]
            if len(matching) != 1 or record.get(
                "training_selection_file_sha256"
            ) != matching[0].get("file_sha256"):
                raise PermissionError("Held epoch contributor selection-file binding is invalid.")
            selection = read_json_object(
                execution_output_root / str(matching[0]["relative_path"])
            )
            if set(selection) != {
                "stage1_epochs",
                "stage2_epochs",
                "stage1_history",
                "stage2_history",
            }:
                raise PermissionError("Held epoch contributor selection has a wrong schema.")
            try:
                _validate_source_selection_history(selection, recipe=recipe)
            except (TypeError, ValueError) as error:
                raise PermissionError(
                    "Held epoch contributor does not replay the frozen selection rule."
                ) from error
            stage1 = record.get("stage1_epochs")
            stage2 = record.get("stage2_epochs")
            if (
                type(stage1) is not int
                or type(stage2) is not int
                or selection.get("stage1_epochs") != stage1
                or selection.get("stage2_epochs") != stage2
            ):
                raise PermissionError("Held epoch contributor differs from its selection file.")
            receipt_record = next(
                candidate
                for candidate in private_artifact_records
                if candidate.get("job_id") == job_id
                and candidate.get("artifact_kind") == "producer_receipt"
            )
            producer_receipt = read_json_object(
                execution_output_root / str(receipt_record["relative_path"])
            )
            if producer_receipt.get("selected_stage1_epochs") != stage1 or (
                producer_receipt.get("selected_stage2_epochs") != stage2
            ):
                raise PermissionError("Held epoch contributor differs from producer receipt.")
            stage1_values.append(stage1)
            stage2_values.append(stage2)
        derived = {
            "stage1_epochs": sorted(stage1_values)[1],
            "stage2_epochs": sorted(stage2_values)[1],
        }
        if observed.get("derived") != derived or dict(selected_epochs[str(seed)]) != derived:
            raise PermissionError("Held epoch selection is not the exact three-fold median.")
    return canonical_json_sha256(value)


def _verify_raw_assets(processed_root: Path) -> dict[str, str]:
    asset = _load_asset_receipt_metadata(processed_root)
    expected = {
        "signals.h5": asset["signals_file_sha256"],
        "manifest.parquet": asset["manifest_parquet_sha256"],
        "manifest.jsonl": asset["manifest_jsonl_sha256"],
        "class_map.json": asset["class_map_sha256"],
    }
    for name, expected_hash in expected.items():
        path = processed_root / name
        _validate_unlinked_regular_file(path)
        if _sha256_file(path) != expected_hash:
            raise ValueError(f"Raw wearable asset hash drifted: {name}.")
    return asset


def _load_public_key_for_fingerprint(path: str | Path) -> object:
    from cryptography.hazmat.primitives import serialization
    from cryptography.hazmat.primitives.asymmetric import rsa

    target = _absolute_no_symlink_path(Path(path), must_exist=True)
    _validate_unlinked_regular_file(target)
    descriptor = os.open(target, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    try:
        before = target.lstat()
        after = os.fstat(descriptor)
        if (before.st_dev, before.st_ino) != (after.st_dev, after.st_ino):
            raise ValueError("Finalizer public key changed while being opened.")
        chunks: list[bytes] = []
        while True:
            chunk = os.read(descriptor, 65536)
            if not chunk:
                break
            chunks.append(chunk)
    finally:
        os.close(descriptor)
    key = serialization.load_pem_public_key(b"".join(chunks))
    if not isinstance(key, rsa.RSAPublicKey) or key.key_size < 3072:
        raise ValueError("Finalizer public key must be RSA with at least 3072 bits.")
    return key


def _verify_source_tag(tag: str, source_commit: str) -> None:
    if not tag.strip() or tag != tag.strip() or tag.startswith("refs/"):
        raise ValueError("Source freeze tag must be nonempty.")
    try:
        subprocess.run(
            ["git", "check-ref-format", f"refs/tags/{tag}"],
            cwd=_REPOSITORY,
            check=True,
            capture_output=True,
            text=True,
        )
        observed = subprocess.run(
            ["git", "rev-parse", "--verify", f"refs/tags/{tag}^{{commit}}"],
            cwd=_REPOSITORY,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError) as error:
        raise ValueError("Source freeze tag does not exist.") from error
    if observed != source_commit:
        raise ValueError("Source freeze tag does not resolve to the current commit.")


def _new_sibling_staging_root(output_root: Path) -> Path:
    parent = output_root.parent
    parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    if os.path.lexists(output_root):
        raise FileExistsError("Canonical input-seal root already exists.")
    staging = parent / f".{output_root.name}.staging-{uuid.uuid4().hex}"
    staging.mkdir(mode=0o700)
    return staging


def _publish_new_tree(staging: Path, destination: Path) -> None:
    if staging.parent != destination.parent or os.path.lexists(destination):
        raise FileExistsError("Atomic input publication destination is not new and local.")
    # Linux renameat2(RENAME_NOREPLACE) closes the check/rename race even for a
    # second writer that ignores our advisory parent lock.
    directory_fd = os.open(
        destination.parent,
        os.O_RDONLY | os.O_DIRECTORY | getattr(os, "O_NOFOLLOW", 0),
    )
    try:
        import fcntl

        fcntl.flock(directory_fd, fcntl.LOCK_EX)
        if os.path.lexists(destination):
            raise FileExistsError("Input-seal destination appeared before publication.")
        libc = ctypes.CDLL(None, use_errno=True)
        renameat2 = getattr(libc, "renameat2", None)
        if renameat2 is None:
            raise OSError(errno.ENOSYS, "renameat2 is required for no-replace publication")
        renameat2.argtypes = [
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_uint,
        ]
        renameat2.restype = ctypes.c_int
        result = renameat2(
            directory_fd,
            staging.name.encode(),
            directory_fd,
            destination.name.encode(),
            1,  # RENAME_NOREPLACE
        )
        if result != 0:
            error_number = ctypes.get_errno()
            if error_number == errno.EEXIST:
                raise FileExistsError(destination)
            raise OSError(error_number, os.strerror(error_number), destination)
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)


def _remove_owned_staging(staging: Path, destination: Path) -> None:
    if (
        staging.parent == destination.parent
        and staging.name.startswith(f".{destination.name}.staging-")
        and staging.exists()
    ):
        shutil.rmtree(staging)


def _fsync_tree(root: Path) -> None:
    for path in sorted(
        (value for value in root.rglob("*") if value.is_dir()),
        key=lambda value: len(value.parts),
        reverse=True,
    ):
        _fsync_directory(path)
    _fsync_directory(root)


def _fsync_file(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(
        path,
        os.O_RDONLY | os.O_DIRECTORY | getattr(os, "O_NOFOLLOW", 0),
    )
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _absolute_no_symlink_path(path: Path, *, must_exist: bool) -> Path:
    absolute = path.absolute()
    component = Path(absolute.anchor)
    for name in absolute.parts[1:]:
        component /= name
        try:
            observed = component.lstat()
        except FileNotFoundError:
            if must_exist:
                raise FileNotFoundError(component) from None
            continue
        if stat.S_ISLNK(observed.st_mode):
            raise ValueError(f"Path contains a symlink component: {component}.")
    if must_exist and not absolute.exists():
        raise FileNotFoundError(absolute)
    return absolute


def _validate_unlinked_regular_file(path: Path) -> None:
    observed = path.lstat()
    if not stat.S_ISREG(observed.st_mode) or observed.st_nlink != 1:
        raise ValueError(f"Expected a regular, non-hard-linked file: {path}.")


def _expected_input_artifact_kinds(groups: Sequence[str]) -> dict[str, str]:
    relative = {
        "source_fit/signals.h5": "source_fit_signals",
        "source_fit/labeled_view.jsonl": "source_fit_labeled_view",
        "source_fit/baseline_view.jsonl": "source_fit_baseline_view",
        "target_support/signals.h5": "target_support_signals",
        "target_support/model_view.jsonl": "target_support_model_view",
        "target_support/baseline_view.jsonl": "target_support_baseline_view",
        "target_support/support_labels.jsonl": "target_support_labels",
        "target_query/signals.h5": "target_query_signals",
        "target_query/model_view.jsonl": "target_query_model_view",
        "target_query/baseline_view.jsonl": "target_query_baseline_view",
        "target_query/expected_unlabeled.jsonl": "target_query_expected_unlabeled",
        "target_query/query_labels.jsonl.enc": "encrypted_query_labels",
        "target_query/query_analysis_covariates.jsonl.enc": (
            "encrypted_query_analysis_covariates"
        ),
        "block_context.jsonl": "target_block_context",
    }
    output = {
        f"{group}/{path}": kind
        for group in groups
        for path, kind in relative.items()
    }
    output["input_seal_decision.json"] = "signed_decision"
    return output


def _validate_sealed_tree_entries(root: Path) -> set[str]:
    observed_files: set[str] = set()
    for current, directories, files in os.walk(root, topdown=True, followlinks=False):
        current_path = Path(current)
        current_stat = current_path.lstat()
        if not stat.S_ISDIR(current_stat.st_mode) or stat.S_IMODE(current_stat.st_mode) != 0o700:
            raise PermissionError(f"Sealed directory must be a real mode-0700 directory: {current_path}.")
        for name in list(directories):
            child = current_path / name
            child_stat = child.lstat()
            if not stat.S_ISDIR(child_stat.st_mode) or stat.S_ISLNK(child_stat.st_mode):
                raise ValueError(f"Sealed tree contains a non-directory or symlink: {child}.")
        for name in files:
            child = current_path / name
            child_stat = child.lstat()
            if not stat.S_ISREG(child_stat.st_mode) or child_stat.st_nlink != 1:
                raise ValueError(f"Sealed tree contains a special or hard-linked file: {child}.")
            if stat.S_IMODE(child_stat.st_mode) != 0o400:
                raise PermissionError(f"Sealed file must have mode 0400: {child}.")
            observed_files.add(child.relative_to(root).as_posix())
    return observed_files


def _validate_sealed_hdf5(path: Path) -> None:
    with h5py.File(path, "r") as handle:
        if set(handle) != {"channel_mask", "x"}:
            raise ValueError("Sealed HDF5 must contain exactly channel_mask and x.")
        for name in ("channel_mask", "x"):
            link = handle.get(name, getlink=True)
            if not isinstance(link, h5py.HardLink):
                raise TypeError("Sealed HDF5 must not use soft or external dataset links.")
        x = handle["x"]
        mask = handle["channel_mask"]
        if (
            x.ndim != 3
            or x.shape[1:] != (64, 400)
            or x.dtype != np.dtype(np.float32)
            or mask.shape != x.shape[:2]
            or mask.dtype != np.dtype(bool)
        ):
            raise ValueError("Sealed HDF5 shape or dtype drifted.")
        x_values = np.asarray(x[:], dtype=np.float32)
        mask_values = np.asarray(mask[:], dtype=bool)
        if not np.isfinite(x_values[mask_values]).all() or not np.all(
            x_values[~mask_values] == 0.0
        ):
            raise ValueError("Sealed HDF5 signal values violate the finite/zero-mask contract.")


def _validate_sealed_regular_file(path: Path) -> None:
    _validate_unlinked_regular_file(path)
    mode = stat.S_IMODE(path.lstat().st_mode)
    if mode != 0o400:
        raise PermissionError(f"Sealed artifact mode must be 0400: {path}.")


def _nullable_float_vector(value: object, *, expected: int) -> tuple[np.ndarray, np.ndarray]:
    raw = list(value) if isinstance(value, (list, tuple, np.ndarray)) else []
    if len(raw) != expected:
        raise ValueError(f"Nullable vector must have exact width {expected}.")
    missing = np.asarray(
        [item is None or (isinstance(item, float) and not np.isfinite(item)) for item in raw],
        dtype=bool,
    )
    values = np.asarray([0.0 if miss else float(item) for item, miss in zip(raw, missing)])
    if not np.isfinite(values).all() or np.any(values[missing] != 0.0):
        raise ValueError("Nullable vectors must zero-fill missing values.")
    return values.astype(np.float32), missing


def _strict_int_vector(value: object, *, expected: int) -> np.ndarray:
    values = np.asarray(value, dtype=np.float64)
    if (
        values.shape != (expected,)
        or not np.isfinite(values).all()
        or not np.equal(values, np.floor(values)).all()
    ):
        raise ValueError("Canonical channel IDs must be one exact integer vector.")
    return values.astype(np.int64)


def _strict_int(value: object, *, name: str) -> int:
    numeric = float(value)
    if not np.isfinite(numeric) or numeric != np.floor(numeric):
        raise ValueError(f"{name} must be an exact finite integer.")
    return int(numeric)


def _block_number(run_id: str) -> int:
    match = re.fullmatch(r"block(\d{2})", run_id)
    if match is None:
        raise ValueError("Run ID is not a canonical wearable block.")
    return int(match.group(1))


def _nullable_string(value: object) -> str | None:
    if value is None or (isinstance(value, float) and not np.isfinite(value)):
        return None
    text = str(value)
    return None if text.lower() == "nan" else text


def _is_sha256(value: object) -> bool:
    return isinstance(value, str) and bool(_SHA256.fullmatch(value))


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    try:
        observed = os.fstat(descriptor)
        if not stat.S_ISREG(observed.st_mode) or observed.st_nlink != 1:
            raise ValueError(f"Cannot hash a nonregular or hard-linked file: {path}.")
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    finally:
        os.close(descriptor)
    return digest.hexdigest()
