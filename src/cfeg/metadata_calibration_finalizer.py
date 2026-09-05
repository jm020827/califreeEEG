from __future__ import annotations

import hashlib
import json
import os
import shutil
import stat
import uuid
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from cfeg.analysis.metadata_calibration_baselines import (
    BaselineAnalysisBundle,
    reduce_metadata_calibration_baseline_scores,
)
from cfeg.analysis.metadata_calibration_efficiency import (
    MetadataCalibrationBundleBindings,
    evaluate_development_gate,
    evaluate_held_claim_sequence,
    metadata_calibration_bundle_spec_for_phase,
    reduce_metadata_calibration_predictions,
)
from cfeg.data.metadata_calibration_baseline_sealed import (
    exact_source_pool_identities,
    load_baseline_source_fit,
)
from cfeg.data.metadata_calibration_interventions import (
    build_metadata_intervention_mapping,
    resolve_metadata_intervention_batch,
    resolved_context_usage_sha256,
)
from cfeg.data.metadata_calibration_sealed import (
    load_sealed_target_group,
    support_indices_for_budget,
    tensor_batch,
)
from cfeg.metadata_calibration_atomic import (
    PrivateStagingSeal,
    seal_metadata_calibration_private_staging,
    validate_and_summarize_support_rows,
)
from cfeg.metadata_calibration_attempt import (
    CANONICAL_OUTCOME_ACCESS_REGISTRY_ROOT,
    OutcomeAccessClaimBinding,
    claim_outcome_access,
    validate_outcome_access_claim,
)
from cfeg.metadata_calibration_authority import (
    canonical_json_bytes,
    canonical_json_sha256,
    decrypt_json_for_finalizer,
    issue_signed_record,
    read_json_object,
    verify_signed_record,
)
from cfeg.metadata_calibration_execution import (
    execution_cells_for_phase,
    resolve_cell_execution_spec,
)
from cfeg.metadata_calibration_job import (
    AuthorizedMetadataCalibrationJob,
    CompletedMetadataCalibrationJob,
    publish_tree_noreplace,
    resolve_authorized_metadata_calibration_job,
    validate_published_metadata_calibration_job,
)
from cfeg.metadata_calibration_manifest import validate_authorization_envelope
from cfeg.metadata_calibration_sealer import (
    DEVELOPMENT_GATE_RECEIPT_SCHEMA,
    FILESYSTEM_PRIVATE_SEAL_SCHEMA,
    validate_metadata_calibration_result_bundle,
    validate_source_gate_artifacts,
    validate_source_private_epoch_artifacts,
    write_signed_decision_exclusive,
)

HELD_COMPLETION_RECEIPT_SCHEMA = (
    "cfeg.metadata-calibration-held-completion-receipt.v1"
)
_DEVELOPMENT_COMPLETION_FIELDS = {
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
_PRIVATE_SEAL_FIELDS = {
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
_HELD_COMPLETION_FIELDS = {
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
    "filesystem_private_seal_file_sha256",
    "filesystem_private_seal_signed_record_sha256",
    "filesystem_private_tree_sha256",
    "held_result_bundle_root",
    "held_result_bundle_sha256",
    "held_claim_result",
    "outcome_finalization_performed",
    "query_outcomes_loaded",
    "outcome_access_claim_path",
    "outcome_access_claim_file_sha256",
    "outcome_access_claim_signed_record_sha256",
    "outcome_access_scope_sha256",
    "source_development_gate_binding",
    "source_commit",
    "source_tree_sha256",
    "source_tag",
    "signing_key_id",
    "signer_role",
    "authorization_basis",
    "signature_algorithm",
    "completion_payload_sha256",
    "signature",
    "signed_record_sha256",
}


@dataclass(frozen=True)
class PhaseFinalizationResult:
    phase: str
    filesystem_private_seal_path: Path
    result_bundle_root: Path
    completion_receipt_path: Path
    result_bundle_sha256: str
    gate_or_claim_result: dict[str, Any]
    held_evaluation_access_authorized: bool


@dataclass(frozen=True)
class _CollectedPhase:
    manifest: dict[str, Any]
    decision: dict[str, Any]
    envelope: dict[str, Any]
    jobs: tuple[AuthorizedMetadataCalibrationJob, ...]
    completed: tuple[CompletedMetadataCalibrationJob, ...]
    bindings: MetadataCalibrationBundleBindings
    candidate_predictions: pd.DataFrame
    baseline_scores: pd.DataFrame
    support_rows: pd.DataFrame
    expected_query_unlabeled: pd.DataFrame
    expected_source_pools: dict[tuple[str, str], str]
    private_artifact_records: tuple[dict[str, Any], ...]
    trusted_signing_public_key_path: Path


def finalize_metadata_calibration_phase(
    *,
    manifest_path: str | Path,
    outcome_decision_path: str | Path,
    authorization_envelope_path: str | Path,
    trusted_signing_public_key_path: str | Path,
    authority_signing_private_key_path: str | Path,
    authority_signing_private_key_password: bytes | None = None,
    finalizer_private_key_path: str | Path,
    finalizer_private_key_password: bytes,
    filesystem_private_seal_path: str | Path,
    result_bundle_root: str | Path,
    completion_receipt_path: str | Path,
    authorization_basis: str,
    outcome_access_registry_root: str | Path = CANONICAL_OUTCOME_ACCESS_REGISTRY_ROOT,
) -> PhaseFinalizationResult:
    """Finalize exactly once after a complete signed filesystem-private grid."""

    if len(finalizer_private_key_password) < 32:
        raise ValueError("Finalizer password must remain a strong in-memory capability.")
    registry_root = _absolute_no_symlink_path(
        Path(outcome_access_registry_root), must_exist=False
    )
    if registry_root != CANONICAL_OUTCOME_ACCESS_REGISTRY_ROOT:
        raise ValueError("Finalization requires the canonical cross-run outcome registry.")
    private_seal_path = _absolute_no_symlink_path(
        Path(filesystem_private_seal_path), must_exist=False
    )
    result_root = _absolute_no_symlink_path(Path(result_bundle_root), must_exist=False)
    completion_path = _absolute_no_symlink_path(
        Path(completion_receipt_path), must_exist=False
    )
    if completion_path != result_root / "completion_receipt.json":
        raise ValueError("Completion receipt must be the canonical child of the result root.")
    if _is_relative_to(private_seal_path, result_root):
        raise ValueError("The pre-label private seal must remain outside the result bundle.")
    for path in (private_seal_path, result_root):
        if os.path.lexists(path):
            raise FileExistsError("Finalization outputs must be completely new.")

    collected = _collect_complete_phase(
        manifest_path=manifest_path,
        outcome_decision_path=outcome_decision_path,
        authorization_envelope_path=authorization_envelope_path,
        trusted_signing_public_key_path=trusted_signing_public_key_path,
    )
    if authorization_basis != collected.decision.get("authorization_basis"):
        raise PermissionError(
            "Finalizer authorization_basis differs from the signed outcome decision."
        )
    execution_root = _absolute_no_symlink_path(
        Path(collected.manifest["execution_output_root"]), must_exist=True
    )
    for path in (private_seal_path, result_root):
        if _is_relative_to(path, execution_root):
            raise ValueError("Finalization records must not alter the sealed execution tree.")

    phase = str(collected.manifest["phase"])
    spec = metadata_calibration_bundle_spec_for_phase(phase=phase)  # type: ignore[arg-type]
    _validate_candidate_context_ledgers(collected)
    private_staging = seal_metadata_calibration_private_staging(
        collected.candidate_predictions,
        collected.baseline_scores,
        collected.support_rows,
        expected_query_unlabeled=collected.expected_query_unlabeled,
        expected_source_pools=collected.expected_source_pools,
        bindings=collected.bindings,
        spec=spec,
    )
    selected_epochs, epoch_derivation = _phase_epoch_selection(collected)
    unsigned_private = _build_filesystem_private_seal(
        collected,
        private_staging=private_staging,
        selected_epochs=selected_epochs,
        epoch_derivation=epoch_derivation,
        authorization_basis=authorization_basis,
    )
    signed_private = issue_signed_record(
        unsigned_private,
        private_key_path=authority_signing_private_key_path,
        private_key_password=authority_signing_private_key_password,
        hash_field="private_seal_payload_sha256",
    )
    write_signed_decision_exclusive(private_seal_path, signed_private)
    _validate_durable_private_seal(
        private_seal_path,
        collected=collected,
        private_staging=private_staging,
        trusted_signing_public_key_path=trusted_signing_public_key_path,
    )

    outcome_access_claim = claim_outcome_access(
        manifest=collected.manifest,
        decision=collected.decision,
        signed_private_seal=signed_private,
        private_seal_path=private_seal_path,
        registry_root=registry_root,
        trusted_signing_public_key_path=trusted_signing_public_key_path,
        authority_signing_private_key_path=authority_signing_private_key_path,
        authority_signing_private_key_password=authority_signing_private_key_password,
        authorization_basis=authorization_basis,
    )

    # This is the first analysis/finalization process permitted to decrypt or
    # reduce query outcomes after the permanent cohort claim. The trusted input
    # sealer saw raw labels earlier only to create non-reporting encrypted sidecars.
    expected_query, covariates = _decrypt_and_validate_query_sidecars(
        collected,
        finalizer_private_key_path=finalizer_private_key_path,
        finalizer_private_key_password=finalizer_private_key_password,
    )
    expected_support, _ = validate_and_summarize_support_rows(
        collected.support_rows,
        expected_query_unlabeled=collected.expected_query_unlabeled,
        spec=spec,
    )
    candidate = reduce_metadata_calibration_predictions(
        collected.candidate_predictions,
        expected_query=expected_query,
        expected_support=expected_support,
        bindings=collected.bindings,
        spec=spec,
    )
    baseline = reduce_metadata_calibration_baseline_scores(
        collected.baseline_scores,
        expected_query_with_label=expected_query,
        expected_support=expected_support,
        expected_source_pools=collected.expected_source_pools,
        bindings=collected.bindings,
        spec=spec,
    )
    if phase == "source_development":
        gate = evaluate_development_gate(
            candidate["participant_contrasts"],
            outer_fold_by_subject=_outer_fold_by_subject(collected.manifest),
            missing_fallback_probability_exact=bool(
                candidate["missing_fallback_probability_exact"]
            ),
            metadata_only_structural_chance_exact=_metadata_only_structure_proof(
                collected,
                expected_query,
            ),
            expected_n=39,
            spec=spec,
        )
        gate["held_selected_epochs_by_seed"] = selected_epochs
        gate["held_selected_epochs_by_seed_sha256"] = canonical_json_sha256(
            selected_epochs
        )
        gate["held_epoch_selection_derivation_sha256"] = canonical_json_sha256(
            epoch_derivation
        )
        gate_or_claim = gate
    else:
        gate_or_claim = evaluate_held_claim_sequence(
            candidate["participant_contrasts"],
            expected_n=60,
            spec=spec,
        )

    result_staging, result_bundle_sha256 = _stage_result_bundle(
        result_root,
        phase=phase,
        candidate=candidate,
        baseline=baseline,
        covariates=covariates,
        gate_or_claim=gate_or_claim,
        bindings=collected.bindings,
    )
    try:
        if phase == "source_development":
            unsigned_completion = _build_development_gate_receipt(
                collected,
                signed_private=signed_private,
                private_seal_path=private_seal_path,
                result_bundle_root=result_root,
                result_bundle_sha256=result_bundle_sha256,
                development_gate_result=gate_or_claim,
                outcome_access_claim=outcome_access_claim,
                authorization_basis=authorization_basis,
            )
            signed_completion = issue_signed_record(
                unsigned_completion,
                private_key_path=authority_signing_private_key_path,
                private_key_password=authority_signing_private_key_password,
                hash_field="gate_payload_sha256",
            )
            held_authorized = bool(gate_or_claim["held_evaluation_access_allowed"])
        else:
            unsigned_completion = _build_held_completion_receipt(
                collected,
                signed_private=signed_private,
                private_seal_path=private_seal_path,
                result_bundle_root=result_root,
                result_bundle_sha256=result_bundle_sha256,
                held_claim_result=gate_or_claim,
                outcome_access_claim=outcome_access_claim,
                authorization_basis=authorization_basis,
            )
            signed_completion = issue_signed_record(
                unsigned_completion,
                private_key_path=authority_signing_private_key_path,
                private_key_password=authority_signing_private_key_password,
                hash_field="completion_payload_sha256",
            )
            held_authorized = False
        write_signed_decision_exclusive(
            result_staging / "completion_receipt.json",
            signed_completion,
        )
        _lock_and_fsync_result_tree(result_staging)
        if phase == "source_development":
            validate_development_completion_receipt(
                result_staging / "completion_receipt.json",
                private_seal_path=private_seal_path,
                manifest=collected.manifest,
                trusted_signing_public_key_path=trusted_signing_public_key_path,
                physical_result_bundle_root=result_staging,
                expected_result_bundle_root=result_root,
            )
        else:
            validate_held_completion_receipt(
                result_staging / "completion_receipt.json",
                private_seal_path=private_seal_path,
                manifest=collected.manifest,
                trusted_signing_public_key_path=trusted_signing_public_key_path,
                physical_result_bundle_root=result_staging,
                expected_result_bundle_root=result_root,
            )
        publish_tree_noreplace(result_staging, result_root)
        if phase == "source_development":
            validate_development_completion_receipt(
                completion_path,
                private_seal_path=private_seal_path,
                manifest=collected.manifest,
                trusted_signing_public_key_path=trusted_signing_public_key_path,
                physical_result_bundle_root=result_root,
                expected_result_bundle_root=result_root,
            )
        else:
            validate_held_completion_receipt(
                completion_path,
                private_seal_path=private_seal_path,
                manifest=collected.manifest,
                trusted_signing_public_key_path=trusted_signing_public_key_path,
                physical_result_bundle_root=result_root,
                expected_result_bundle_root=result_root,
            )
    except BaseException:
        if os.path.lexists(result_staging):
            shutil.rmtree(result_staging)
        raise
    return PhaseFinalizationResult(
        phase=phase,
        filesystem_private_seal_path=private_seal_path,
        result_bundle_root=result_root,
        completion_receipt_path=completion_path,
        result_bundle_sha256=result_bundle_sha256,
        gate_or_claim_result=dict(gate_or_claim),
        held_evaluation_access_authorized=held_authorized,
    )


def validate_development_completion_receipt(
    receipt_path: str | Path,
    *,
    private_seal_path: str | Path,
    manifest: Mapping[str, Any],
    trusted_signing_public_key_path: str | Path,
    physical_result_bundle_root: str | Path | None = None,
    expected_result_bundle_root: str | Path | None = None,
) -> str:
    """Validate both source PASS and terminal FAIL without granting on FAIL."""

    receipt_path = Path(receipt_path).absolute()
    private_seal_path = Path(private_seal_path).absolute()
    receipt = read_json_object(receipt_path)
    signed_hash = verify_signed_record(
        receipt,
        public_key_path=trusted_signing_public_key_path,
        hash_field="gate_payload_sha256",
        expected_schema=DEVELOPMENT_GATE_RECEIPT_SCHEMA,
    )
    private = read_json_object(private_seal_path)
    private_signed_hash = verify_signed_record(
        private,
        public_key_path=trusted_signing_public_key_path,
        hash_field="private_seal_payload_sha256",
        expected_schema=FILESYSTEM_PRIVATE_SEAL_SCHEMA,
    )
    expected_ids = [
        str(job["job_id"]) for job in manifest["execution_contract"]["jobs"]
    ]
    manifest_payload = dict(manifest)
    manifest_hash = manifest_payload.pop("phase_manifest_sha256", None)
    artifact_records = private.get("private_artifact_records")
    receipt_hashes = private.get("producer_receipt_file_sha256s_by_job")
    if (
        set(receipt) != _DEVELOPMENT_COMPLETION_FIELDS
        or set(private) != _PRIVATE_SEAL_FIELDS
        or manifest_hash != canonical_json_sha256(manifest_payload)
        or manifest.get("phase") != "source_development"
        or receipt.get("candidate_id") != "metadata-calibration-efficiency-v1"
        or receipt.get("phase") != "source_development"
        or private.get("candidate_id") != "metadata-calibration-efficiency-v1"
        or private.get("phase") != "source_development"
        or private.get("status") != "complete_private_grid_signed_before_outcomes"
        or receipt.get("source_phase_manifest_sha256") != manifest_hash
        or private.get("phase_manifest_sha256") != manifest_hash
        or receipt.get("source_input_seal_receipt_signed_record_sha256")
        != manifest.get("input_seal_receipt_signed_record_sha256")
        or private.get("input_seal_receipt_signed_record_sha256")
        != manifest.get("input_seal_receipt_signed_record_sha256")
        or receipt.get("source_input_artifact_tree_sha256")
        != manifest.get("input_artifact_tree_sha256")
        or private.get("input_artifact_tree_sha256")
        != manifest.get("input_artifact_tree_sha256")
        or receipt.get("source_execution_contract_sha256")
        != manifest.get("execution_contract_sha256")
        or private.get("execution_contract_sha256")
        != manifest.get("execution_contract_sha256")
        or receipt.get("source_execution_output_root")
        != manifest.get("execution_output_root")
        or private.get("execution_output_root") != manifest.get("execution_output_root")
        or receipt.get("source_outcome_decision_signed_record_sha256")
        != private.get("outcome_decision_signed_record_sha256")
        or receipt.get("filesystem_private_seal_file_sha256")
        != _sha256_file(private_seal_path)
        or receipt.get("filesystem_private_seal_signed_record_sha256")
        != private_signed_hash
        or receipt.get("filesystem_private_tree_sha256")
        != private.get("filesystem_private_tree_sha256")
        or private.get("completed_job_ids") != expected_ids
        or private.get("expected_job_count") != len(expected_ids)
        or not isinstance(receipt_hashes, Mapping)
        or set(map(str, receipt_hashes)) != set(expected_ids)
        or not all(_is_sha256(value) for value in receipt_hashes.values())
        or not isinstance(artifact_records, list)
        or not artifact_records
        or private.get("complete_candidate_and_mandatory_baseline_grid") is not True
        or private.get("query_outcomes_loaded") is not False
        or receipt.get("outcome_finalization_performed") is not True
        or receipt.get("source_commit") != manifest.get("source_commit")
        or receipt.get("source_tree_sha256") != manifest.get("source_tree_sha256")
        or receipt.get("source_tag") != manifest.get("source_tag")
        or private.get("source_commit") != manifest.get("source_commit")
        or private.get("source_tree_sha256") != manifest.get("source_tree_sha256")
        or private.get("source_tag") != manifest.get("source_tag")
        or receipt.get("signing_key_id") != private.get("signing_key_id")
        or receipt.get("signer_role") != private.get("signer_role")
        or receipt.get("authorization_basis") != private.get("authorization_basis")
        or receipt.get("signature_algorithm") != "ed25519"
        or private.get("signature_algorithm") != "ed25519"
    ):
        raise PermissionError("Source completion receipt/private seal binding is invalid.")
    outcome_claim = validate_outcome_access_claim(
        str(receipt["outcome_access_claim_path"]),
        manifest=manifest,
        signed_private_seal=private,
        private_seal_path=private_seal_path,
        trusted_signing_public_key_path=trusted_signing_public_key_path,
    )
    if (
        receipt.get("outcome_access_claim_file_sha256")
        != outcome_claim.claim_file_sha256
        or receipt.get("outcome_access_claim_signed_record_sha256")
        != outcome_claim.signed_record_sha256
        or receipt.get("outcome_access_scope_sha256")
        != outcome_claim.outcome_access_scope_sha256
    ):
        raise PermissionError("Source completion does not bind its one-shot outcome claim.")
    physical_hash = _validate_manifest_bound_private_grid(manifest, private)
    if physical_hash != private["filesystem_private_tree_sha256"]:
        raise PermissionError("Source completion references a drifted private execution tree.")
    validate_source_private_epoch_artifacts(
        private_seal_path,
        trusted_signing_public_key_path=trusted_signing_public_key_path,
    )

    gate = receipt.get("development_gate_result")
    status = receipt.get("development_gate_status")
    passed = status == "pass"
    selected_epochs = _validate_private_selected_epochs(
        private.get("held_selected_epochs_by_seed")
    )
    epoch_derivation = private.get("held_epoch_selection_derivation")
    if status not in {"pass", "fail"} or not isinstance(gate, Mapping) or (
        gate.get("schema") != "cfeg.metadata-calibration-development-gate.v1"
        or gate.get("passed") is not passed
        or gate.get("decision")
        != ("development_go" if passed else "development_no_go")
        or gate.get("held_evaluation_access_allowed") is not passed
        or receipt.get("held_input_access_authorized") is not passed
        or private.get("held_selected_epochs_by_seed_sha256")
        != canonical_json_sha256(selected_epochs)
        or gate.get("held_selected_epochs_by_seed") != selected_epochs
        or gate.get("held_selected_epochs_by_seed_sha256")
        != canonical_json_sha256(selected_epochs)
        or not isinstance(epoch_derivation, Mapping)
        or gate.get("held_epoch_selection_derivation_sha256")
        != canonical_json_sha256(epoch_derivation)
    ):
        raise PermissionError("Source completion gate status is internally inconsistent.")
    declared_result_root = _absolute_no_symlink_path(
        Path(str(receipt["source_result_bundle_root"])), must_exist=False
    )
    expected_declared_root = (
        declared_result_root
        if expected_result_bundle_root is None
        else _absolute_no_symlink_path(Path(expected_result_bundle_root), must_exist=False)
    )
    if declared_result_root != expected_declared_root:
        raise PermissionError("Source completion declares an unexpected canonical result root.")
    physical_result_root = (
        declared_result_root
        if physical_result_bundle_root is None
        else _absolute_no_symlink_path(Path(physical_result_bundle_root), must_exist=True)
    )
    observed_result_hash = validate_metadata_calibration_result_bundle(
        physical_result_root,
        completion_receipt_path=receipt_path,
        expected_phase="source_development",
        expected_bindings=_expected_result_bindings(manifest, private),
    )
    if observed_result_hash != receipt.get("source_result_bundle_sha256"):
        raise PermissionError("Source completion does not bind its physical result bundle.")
    if passed:
        validate_source_gate_artifacts(
            receipt_path,
            private_seal_path,
            trusted_signing_public_key_path=trusted_signing_public_key_path,
            physical_result_bundle_root=physical_result_root,
            expected_result_bundle_root=declared_result_root,
        )
    return signed_hash


def _validate_private_selected_epochs(value: object) -> dict[str, dict[str, int]]:
    if not isinstance(value, Mapping) or set(map(str, value)) != {"42", "43", "44"}:
        raise PermissionError("Private epoch selection lacks exact seeds 42, 43, and 44.")
    output: dict[str, dict[str, int]] = {}
    for seed in (42, 43, 44):
        selected = value.get(str(seed))
        if not isinstance(selected, Mapping) or set(selected) != {
            "stage1_epochs",
            "stage2_epochs",
        }:
            raise PermissionError("Private epoch selection has an invalid exact schema.")
        stage1 = selected.get("stage1_epochs")
        stage2 = selected.get("stage2_epochs")
        if type(stage1) is not int or not 10 <= stage1 <= 40 or type(
            stage2
        ) is not int or not 5 <= stage2 <= 25:
            raise PermissionError("Private selected epoch lies outside the frozen range.")
        output[str(seed)] = {"stage1_epochs": stage1, "stage2_epochs": stage2}
    return output


def validate_held_completion_receipt(
    receipt_path: str | Path,
    *,
    private_seal_path: str | Path,
    manifest: Mapping[str, Any],
    trusted_signing_public_key_path: str | Path,
    physical_result_bundle_root: str | Path | None = None,
    expected_result_bundle_root: str | Path | None = None,
) -> str:
    receipt = read_json_object(receipt_path)
    signed_hash = verify_signed_record(
        receipt,
        public_key_path=trusted_signing_public_key_path,
        hash_field="completion_payload_sha256",
        expected_schema=HELD_COMPLETION_RECEIPT_SCHEMA,
    )
    private = read_json_object(private_seal_path)
    private_signed_hash = verify_signed_record(
        private,
        public_key_path=trusted_signing_public_key_path,
        hash_field="private_seal_payload_sha256",
        expected_schema=FILESYSTEM_PRIVATE_SEAL_SCHEMA,
    )
    manifest_payload = dict(manifest)
    manifest_hash = manifest_payload.pop("phase_manifest_sha256", None)
    expected_ids = [
        str(job["job_id"]) for job in manifest["execution_contract"]["jobs"]
    ]
    artifact_records = private.get("private_artifact_records")
    expected_epochs = manifest["execution_contract"]["phase_spec"][
        "held_selected_epochs_by_seed"
    ]
    if set(receipt) != _HELD_COMPLETION_FIELDS or set(private) != _PRIVATE_SEAL_FIELDS or (
        manifest_hash != canonical_json_sha256(manifest_payload)
        or manifest.get("phase") != "held_participant_evaluation"
        or receipt.get("candidate_id") != "metadata-calibration-efficiency-v1"
        or receipt.get("phase") != "held_participant_evaluation"
        or receipt.get("status") != "held_outcomes_finalized_once"
        or receipt.get("phase_manifest_sha256") != manifest_hash
        or private.get("candidate_id") != "metadata-calibration-efficiency-v1"
        or private.get("phase") != "held_participant_evaluation"
        or private.get("status") != "complete_private_grid_signed_before_outcomes"
        or private.get("phase_manifest_sha256") != manifest_hash
        or receipt.get("outcome_decision_signed_record_sha256")
        != private.get("outcome_decision_signed_record_sha256")
        or receipt.get("authorization_envelope_sha256")
        != private.get("authorization_envelope_sha256")
        or receipt.get("input_seal_receipt_signed_record_sha256")
        != manifest.get("input_seal_receipt_signed_record_sha256")
        or private.get("input_seal_receipt_signed_record_sha256")
        != manifest.get("input_seal_receipt_signed_record_sha256")
        or receipt.get("input_artifact_tree_sha256")
        != manifest.get("input_artifact_tree_sha256")
        or private.get("input_artifact_tree_sha256")
        != manifest.get("input_artifact_tree_sha256")
        or receipt.get("execution_contract_sha256")
        != manifest.get("execution_contract_sha256")
        or private.get("execution_contract_sha256")
        != manifest.get("execution_contract_sha256")
        or private.get("execution_output_root") != manifest.get("execution_output_root")
        or receipt.get("filesystem_private_seal_file_sha256")
        != _sha256_file(Path(private_seal_path))
        or receipt.get("filesystem_private_seal_signed_record_sha256")
        != private_signed_hash
        or receipt.get("filesystem_private_tree_sha256")
        != private.get("filesystem_private_tree_sha256")
        or private.get("completed_job_ids") != expected_ids
        or private.get("expected_job_count") != len(expected_ids)
        or not isinstance(artifact_records, list)
        or not artifact_records
        or private.get("held_selected_epochs_by_seed") != expected_epochs
        or private.get("held_selected_epochs_by_seed_sha256")
        != canonical_json_sha256(expected_epochs)
        or private.get("complete_candidate_and_mandatory_baseline_grid") is not True
        or private.get("query_outcomes_loaded") is not False
        or receipt.get("outcome_finalization_performed") is not True
        or receipt.get("query_outcomes_loaded") is not True
        or not isinstance(receipt.get("held_claim_result"), Mapping)
        or receipt["held_claim_result"].get("schema")
        != "cfeg.metadata-calibration-held-claims.v1"
        or receipt.get("source_development_gate_binding")
        != manifest.get("source_development_gate_binding")
        or receipt.get("source_commit") != manifest.get("source_commit")
        or receipt.get("source_tree_sha256") != manifest.get("source_tree_sha256")
        or receipt.get("source_tag") != manifest.get("source_tag")
        or private.get("source_commit") != manifest.get("source_commit")
        or private.get("source_tree_sha256") != manifest.get("source_tree_sha256")
        or private.get("source_tag") != manifest.get("source_tag")
        or receipt.get("signing_key_id") != private.get("signing_key_id")
        or receipt.get("signer_role") != private.get("signer_role")
        or receipt.get("authorization_basis") != private.get("authorization_basis")
        or receipt.get("signature_algorithm") != "ed25519"
        or private.get("signature_algorithm") != "ed25519"
        or not isinstance(receipt.get("held_result_bundle_root"), str)
        or not _is_sha256(receipt.get("held_result_bundle_sha256"))
    ):
        raise PermissionError("Held completion receipt is not an exact finalized record.")
    outcome_claim = validate_outcome_access_claim(
        str(receipt["outcome_access_claim_path"]),
        manifest=manifest,
        signed_private_seal=private,
        private_seal_path=private_seal_path,
        trusted_signing_public_key_path=trusted_signing_public_key_path,
    )
    if (
        receipt.get("outcome_access_claim_file_sha256")
        != outcome_claim.claim_file_sha256
        or receipt.get("outcome_access_claim_signed_record_sha256")
        != outcome_claim.signed_record_sha256
        or receipt.get("outcome_access_scope_sha256")
        != outcome_claim.outcome_access_scope_sha256
    ):
        raise PermissionError("Held completion does not bind its one-shot outcome claim.")
    physical_hash = _validate_manifest_bound_private_grid(manifest, private)
    if physical_hash != private["filesystem_private_tree_sha256"]:
        raise PermissionError("Held completion references a drifted private execution tree.")
    declared_result_root = _absolute_no_symlink_path(
        Path(str(receipt["held_result_bundle_root"])), must_exist=False
    )
    expected_declared_root = (
        declared_result_root
        if expected_result_bundle_root is None
        else _absolute_no_symlink_path(Path(expected_result_bundle_root), must_exist=False)
    )
    if declared_result_root != expected_declared_root:
        raise PermissionError("Held completion declares an unexpected canonical result root.")
    physical_result_root = (
        declared_result_root
        if physical_result_bundle_root is None
        else _absolute_no_symlink_path(Path(physical_result_bundle_root), must_exist=True)
    )
    result_hash = validate_metadata_calibration_result_bundle(
        physical_result_root,
        completion_receipt_path=receipt_path,
        expected_phase="held_participant_evaluation",
        expected_bindings=_expected_result_bindings(manifest, private),
    )
    if result_hash != receipt["held_result_bundle_sha256"]:
        raise PermissionError("Held completion does not bind its physical result bundle.")
    return signed_hash


def _expected_result_bindings(
    manifest: Mapping[str, Any],
    private_seal: Mapping[str, Any],
) -> dict[str, str]:
    """Derive published-analysis provenance from retained phase authority."""

    contract = manifest.get("execution_contract")
    if not isinstance(contract, Mapping) or not isinstance(
        contract.get("global_contracts"), Mapping
    ):
        raise PermissionError("Phase manifest lacks exact global result bindings.")
    input_root = Path(str(manifest.get("input_seal_root"))).absolute()
    input_receipt = read_json_object(input_root / "input_seal_receipt.json")
    globals_ = contract["global_contracts"]
    bindings = MetadataCalibrationBundleBindings(
        plan_sha256=str(globals_["plan_sha256"]),
        execution_manifest_sha256=str(manifest["phase_manifest_sha256"]),
        decision_receipt_sha256=str(
            private_seal["outcome_decision_signed_record_sha256"]
        ),
        asset_receipt_sha256=str(input_receipt["raw_asset_receipt_sha256"]),
        asset_fingerprint_bundle_sha256=str(
            input_receipt["raw_asset_fingerprint_bundle_sha256"]
        ),
        class_map_sha256=str(input_receipt["class_map_sha256"]),
        source_tree_sha256=str(manifest["source_tree_sha256"]),
    )
    return bindings.as_dict()


def _collect_complete_phase(
    *,
    manifest_path: str | Path,
    outcome_decision_path: str | Path,
    authorization_envelope_path: str | Path,
    trusted_signing_public_key_path: str | Path,
) -> _CollectedPhase:
    manifest = read_json_object(manifest_path)
    decision = read_json_object(outcome_decision_path)
    envelope = read_json_object(authorization_envelope_path)
    validate_authorization_envelope(
        envelope,
        manifest=manifest,
        outcome_decision=decision,
        trusted_signing_public_key_path=trusted_signing_public_key_path,
    )
    manifest_jobs = manifest["execution_contract"]["jobs"]
    jobs: list[AuthorizedMetadataCalibrationJob] = []
    completed: list[CompletedMetadataCalibrationJob] = []
    candidate_frames: list[pd.DataFrame] = []
    baseline_frames: list[pd.DataFrame] = []
    artifact_records: list[dict[str, Any]] = []
    for job_spec in manifest_jobs:
        job = resolve_authorized_metadata_calibration_job(
            manifest_path=manifest_path,
            outcome_decision_path=outcome_decision_path,
            authorization_envelope_path=authorization_envelope_path,
            trusted_signing_public_key_path=trusted_signing_public_key_path,
            job_id=str(job_spec["job_id"]),
        )
        done = validate_published_metadata_calibration_job(job)
        jobs.append(job)
        completed.append(done)
        if done.producer_kind == "candidate":
            candidate_frames.append(
                pd.read_parquet(done.canonical_job_root / "probabilities.parquet")
            )
        else:
            baseline_frames.append(
                pd.read_parquet(done.canonical_job_root / "scores.parquet")
            )
        for relative, kind in job.required_artifacts.items():
            path = job.execution_output_root / relative
            observed = path.lstat()
            artifact_records.append(
                {
                    "job_id": str(job.job_spec["job_id"]),
                    "relative_path": relative,
                    "artifact_kind": kind,
                    "size_bytes": observed.st_size,
                    "file_sha256": _sha256_file(path),
                    "mode_octal": f"{stat.S_IMODE(observed.st_mode):04o}",
                }
            )
    expected_ids = [str(job["job_id"]) for job in manifest_jobs]
    if [item.job_id for item in completed] != expected_ids or len(set(expected_ids)) != len(
        expected_ids
    ):
        raise ValueError("Completed job set differs from the exact ordered manifest.")
    if not candidate_frames or not baseline_frames:
        raise ValueError("Complete phase requires candidate and mandatory baseline outputs.")
    bindings = jobs[0].bindings
    if any(job.bindings != bindings for job in jobs[1:]):
        raise ValueError("Phase jobs do not share one exact immutable binding set.")

    group_roots = {str(job.job_spec["checkpoint_group"]): job.group_root for job in jobs}
    support_rows = pd.concat(
        [
            _read_jsonl(root / "target_support/support_labels.jsonl")
            for root in group_roots.values()
        ],
        ignore_index=True,
    )
    expected_query = pd.concat(
        [
            _read_jsonl(root / "target_query/expected_unlabeled.jsonl")
            for root in group_roots.values()
        ],
        ignore_index=True,
    )
    source_pools: dict[tuple[str, str], str] = {}
    for group, root in group_roots.items():
        identities = exact_source_pool_identities(load_baseline_source_fit(root))
        for interface, identity in identities.items():
            source_pools[(group, interface)] = identity
    return _CollectedPhase(
        manifest=manifest,
        decision=decision,
        envelope=envelope,
        jobs=tuple(jobs),
        completed=tuple(completed),
        bindings=bindings,
        candidate_predictions=pd.concat(candidate_frames, ignore_index=True),
        baseline_scores=pd.concat(baseline_frames, ignore_index=True),
        support_rows=support_rows,
        expected_query_unlabeled=expected_query,
        expected_source_pools=source_pools,
        private_artifact_records=tuple(
            sorted(artifact_records, key=lambda value: value["relative_path"])
        ),
        trusted_signing_public_key_path=Path(
            trusted_signing_public_key_path
        ).absolute(),
    )


def _phase_epoch_selection(
    collected: _CollectedPhase,
) -> tuple[dict[str, dict[str, int]], dict[str, Any]]:
    if collected.manifest["phase"] == "source_development":
        receipt_by_id = {item.job_id: item for item in collected.completed}
        records_by_job = {
            (record["job_id"], record["artifact_kind"]): record
            for record in collected.private_artifact_records
        }
        selected: dict[str, dict[str, int]] = {}
        seeds: dict[str, Any] = {}
        for seed in (42, 43, 44):
            folds: dict[str, Any] = {}
            stage1_values: list[int] = []
            stage2_values: list[int] = []
            for fold in ("fold0", "fold1", "fold2"):
                job_id = f"candidate-{fold}-seed{seed}"
                receipt = receipt_by_id[job_id]
                stage1 = receipt.producer_receipt["selected_stage1_epochs"]
                stage2 = receipt.producer_receipt["selected_stage2_epochs"]
                if type(stage1) is not int or type(stage2) is not int:
                    raise ValueError("Source selected epochs must be exact JSON integers.")
                selection_record = records_by_job[(job_id, "epoch_selection_record")]
                folds[fold] = {
                    "job_id": job_id,
                    "producer_receipt_file_sha256": (
                        receipt.producer_receipt_file_sha256
                    ),
                    "training_selection_file_sha256": selection_record["file_sha256"],
                    "stage1_epochs": stage1,
                    "stage2_epochs": stage2,
                }
                stage1_values.append(stage1)
                stage2_values.append(stage2)
            derived = {
                "stage1_epochs": sorted(stage1_values)[1],
                "stage2_epochs": sorted(stage2_values)[1],
            }
            selected[str(seed)] = derived
            seeds[str(seed)] = {"folds": folds, "derived": derived}
        derivation = {
            "schema": "cfeg.metadata-calibration-held-epoch-selection-derivation.v1",
            "rule": (
                "per_seed_integer_median_across_exact_fold0_fold1_fold2_"
                "inner_selected_epochs"
            ),
            "seeds": seeds,
            "held_selected_epochs_by_seed_sha256": canonical_json_sha256(selected),
        }
        return selected, derivation

    selected = dict(
        collected.manifest["execution_contract"]["phase_spec"][
            "held_selected_epochs_by_seed"
        ]
    )
    input_decision = read_json_object(
        collected.jobs[0].group_root.parent / "input_seal_decision.json"
    )
    validate_source_gate_artifacts(
        str(input_decision["source_development_gate_receipt_path"]),
        str(input_decision["source_development_private_seal_path"]),
        trusted_signing_public_key_path=collected.trusted_signing_public_key_path,
    )
    source_private = read_json_object(
        str(input_decision["source_development_private_seal_path"])
    )
    if source_private["held_selected_epochs_by_seed"] != selected:
        raise ValueError("Held manifest epoch selection differs from the source private seal.")
    return selected, dict(source_private["held_epoch_selection_derivation"])


def _build_filesystem_private_seal(
    collected: _CollectedPhase,
    *,
    private_staging: PrivateStagingSeal,
    selected_epochs: Mapping[str, Mapping[str, int]],
    epoch_derivation: Mapping[str, Any],
    authorization_basis: str,
) -> dict[str, Any]:
    records = [dict(value) for value in collected.private_artifact_records]
    tree_sha256 = canonical_json_sha256({"artifacts": records})
    completed_ids = [str(job["job_id"]) for job in collected.manifest["execution_contract"]["jobs"]]
    completed = {value.job_id: value for value in collected.completed}
    record = {
        "schema": FILESYSTEM_PRIVATE_SEAL_SCHEMA,
        "candidate_id": "metadata-calibration-efficiency-v1",
        "phase": collected.manifest["phase"],
        "status": "complete_private_grid_signed_before_outcomes",
        "phase_manifest_sha256": collected.manifest["phase_manifest_sha256"],
        "outcome_decision_signed_record_sha256": collected.decision[
            "signed_record_sha256"
        ],
        "authorization_envelope_sha256": collected.envelope[
            "authorization_envelope_sha256"
        ],
        "input_seal_receipt_signed_record_sha256": collected.manifest[
            "input_seal_receipt_signed_record_sha256"
        ],
        "input_artifact_tree_sha256": collected.manifest[
            "input_artifact_tree_sha256"
        ],
        "execution_contract_sha256": collected.manifest[
            "execution_contract_sha256"
        ],
        "execution_output_root": collected.manifest["execution_output_root"],
        "expected_job_count": collected.manifest["expected_job_count"],
        "completed_job_ids": completed_ids,
        "producer_receipt_file_sha256s_by_job": {
            job_id: completed[job_id].producer_receipt_file_sha256
            for job_id in completed_ids
        },
        "held_selected_epochs_by_seed": {
            str(seed): dict(value) for seed, value in selected_epochs.items()
        },
        "held_selected_epochs_by_seed_sha256": canonical_json_sha256(
            selected_epochs
        ),
        "held_epoch_selection_derivation": dict(epoch_derivation),
        "private_staging_seal": private_staging.as_dict(),
        "private_staging_semantic_sha256": (
            private_staging.combined_private_tree_sha256
        ),
        "private_artifact_records": records,
        "filesystem_private_tree_sha256": tree_sha256,
        "complete_candidate_and_mandatory_baseline_grid": True,
        "query_outcomes_loaded": False,
        "source_commit": collected.manifest["source_commit"],
        "source_tree_sha256": collected.manifest["source_tree_sha256"],
        "source_tag": collected.manifest["source_tag"],
        "signing_key_id": collected.decision["signing_key_id"],
        "signer_role": collected.decision["signer_role"],
        "authorization_basis": authorization_basis,
        "signature_algorithm": "ed25519",
    }
    return record


def _validate_durable_private_seal(
    path: Path,
    *,
    collected: _CollectedPhase,
    private_staging: PrivateStagingSeal,
    trusted_signing_public_key_path: str | Path,
) -> str:
    value = read_json_object(path)
    signed_hash = verify_signed_record(
        value,
        public_key_path=trusted_signing_public_key_path,
        hash_field="private_seal_payload_sha256",
        expected_schema=FILESYSTEM_PRIVATE_SEAL_SCHEMA,
    )
    expected_common = {
        "candidate_id": "metadata-calibration-efficiency-v1",
        "phase": collected.manifest["phase"],
        "status": "complete_private_grid_signed_before_outcomes",
        "phase_manifest_sha256": collected.manifest["phase_manifest_sha256"],
        "outcome_decision_signed_record_sha256": collected.decision[
            "signed_record_sha256"
        ],
        "authorization_envelope_sha256": collected.envelope[
            "authorization_envelope_sha256"
        ],
        "input_seal_receipt_signed_record_sha256": collected.manifest[
            "input_seal_receipt_signed_record_sha256"
        ],
        "input_artifact_tree_sha256": collected.manifest[
            "input_artifact_tree_sha256"
        ],
        "execution_contract_sha256": collected.manifest[
            "execution_contract_sha256"
        ],
        "execution_output_root": collected.manifest["execution_output_root"],
        "expected_job_count": collected.manifest["expected_job_count"],
        "complete_candidate_and_mandatory_baseline_grid": True,
        "query_outcomes_loaded": False,
        "source_commit": collected.manifest["source_commit"],
        "source_tree_sha256": collected.manifest["source_tree_sha256"],
        "source_tag": collected.manifest["source_tag"],
        "signing_key_id": collected.decision["signing_key_id"],
        "signer_role": collected.decision["signer_role"],
        "authorization_basis": collected.decision["authorization_basis"],
        "signature_algorithm": "ed25519",
    }
    if set(value) != _PRIVATE_SEAL_FIELDS or any(
        value.get(key) != expected for key, expected in expected_common.items()
    ):
        raise PermissionError("Durable filesystem private seal has a wrong exact binding.")
    expected_ids = [str(job["job_id"]) for job in collected.manifest["execution_contract"]["jobs"]]
    expected_receipts = {
        item.job_id: item.producer_receipt_file_sha256 for item in collected.completed
    }
    expected_epochs, expected_derivation = _phase_epoch_selection(collected)
    if value.get("completed_job_ids") != expected_ids or value.get(
        "producer_receipt_file_sha256s_by_job"
    ) != expected_receipts or value.get(
        "held_selected_epochs_by_seed"
    ) != expected_epochs or value.get(
        "held_selected_epochs_by_seed_sha256"
    ) != canonical_json_sha256(expected_epochs) or value.get(
        "held_epoch_selection_derivation"
    ) != expected_derivation or value.get("private_staging_seal") != private_staging.as_dict() or value.get(
        "private_staging_semantic_sha256"
    ) != private_staging.combined_private_tree_sha256 or value.get(
        "private_artifact_records"
    ) != [dict(record) for record in collected.private_artifact_records]:
        raise PermissionError("Durable private seal differs from the validated complete grid.")
    observed_tree = _validate_physical_private_tree(
        Path(collected.manifest["execution_output_root"]),
        collected.private_artifact_records,
    )
    if value.get("filesystem_private_tree_sha256") != observed_tree:
        raise PermissionError("Durable private seal physical tree hash is invalid.")
    return signed_hash


def _validate_physical_private_tree(
    root: Path,
    records: Sequence[Mapping[str, Any]],
) -> str:
    root = root.absolute()
    observed_root = root.lstat()
    if not stat.S_ISDIR(observed_root.st_mode) or stat.S_IMODE(observed_root.st_mode) != 0o700:
        raise PermissionError("Execution private root must be one mode-0700 directory.")
    expected_paths = {str(record["relative_path"]) for record in records}
    actual_paths: set[str] = set()
    for path in sorted(root.rglob("*")):
        observed = path.lstat()
        if stat.S_ISDIR(observed.st_mode):
            if stat.S_IMODE(observed.st_mode) != 0o700:
                raise PermissionError("Execution private subdirectories must use mode 0700.")
            continue
        if not stat.S_ISREG(observed.st_mode) or observed.st_nlink != 1:
            raise PermissionError("Execution private tree contains a linked or special file.")
        relative = path.relative_to(root).as_posix()
        actual_paths.add(relative)
    if actual_paths != expected_paths:
        raise PermissionError("Execution private tree has missing or undeclared files.")
    for record in records:
        path = root / str(record["relative_path"])
        observed = path.lstat()
        if (
            record.get("mode_octal") != "0400"
            or stat.S_IMODE(observed.st_mode) != 0o400
            or observed.st_size != record.get("size_bytes")
            or _sha256_file(path) != record.get("file_sha256")
        ):
            raise PermissionError("Execution private artifact hash, size, or mode drifted.")
    return canonical_json_sha256(
        {"artifacts": [dict(value) for value in records]}
    )


def _validate_manifest_bound_private_grid(
    manifest: Mapping[str, Any],
    private: Mapping[str, Any],
) -> str:
    """Bind every private file, kind, job, and producer receipt to the manifest."""

    jobs = manifest["execution_contract"]["jobs"]
    expected_ids = [str(job["job_id"]) for job in jobs]
    expected: dict[str, tuple[str, str]] = {}
    for job in jobs:
        job_id = str(job["job_id"])
        for artifact in job["required_artifacts"]:
            relative = str(artifact["relative_path"])
            if relative in expected:
                raise PermissionError("Execution manifest repeats a private artifact path.")
            expected[relative] = (job_id, str(artifact["artifact_kind"]))

    records = private.get("private_artifact_records")
    receipt_hashes = private.get("producer_receipt_file_sha256s_by_job")
    if (
        private.get("completed_job_ids") != expected_ids
        or private.get("expected_job_count") != len(expected_ids)
        or not isinstance(records, list)
        or len(records) != len(expected)
        or not isinstance(receipt_hashes, Mapping)
        or set(map(str, receipt_hashes)) != set(expected_ids)
        or not all(_is_sha256(value) for value in receipt_hashes.values())
    ):
        raise PermissionError("Private grid does not contain the exact manifest job set.")
    if records != sorted(records, key=lambda record: str(record.get("relative_path"))):
        raise PermissionError("Private artifact records are not in canonical path order.")

    observed: dict[str, tuple[str, str]] = {}
    producer_receipt_records: dict[str, Mapping[str, Any]] = {}
    for record in records:
        if not isinstance(record, Mapping) or set(record) != _PRIVATE_ARTIFACT_RECORD_FIELDS:
            raise PermissionError("Private artifact record has an invalid exact schema.")
        relative = str(record["relative_path"])
        if relative in observed or type(record.get("size_bytes")) is not int or (
            record.get("mode_octal") != "0400"
        ) or not _is_sha256(record.get("file_sha256")):
            raise PermissionError("Private artifact record path, size, mode, or hash is invalid.")
        job_id = str(record["job_id"])
        kind = str(record["artifact_kind"])
        observed[relative] = (job_id, kind)
        if kind == "producer_receipt":
            if job_id in producer_receipt_records:
                raise PermissionError("Private grid repeats a producer receipt for one job.")
            producer_receipt_records[job_id] = record
    if observed != expected or set(producer_receipt_records) != set(expected_ids):
        raise PermissionError("Private artifact paths/kinds differ from the exact manifest.")
    if any(
        record["file_sha256"] != receipt_hashes[job_id]
        for job_id, record in producer_receipt_records.items()
    ):
        raise PermissionError("Private producer-receipt hashes differ from the signed seal map.")
    return _validate_physical_private_tree(
        Path(str(manifest["execution_output_root"])),
        records,
    )


def _validate_candidate_context_ledgers(collected: _CollectedPhase) -> None:
    """Recompute every label-free M mapping and usage digest from sealed rows."""

    phase = str(collected.manifest["phase"])
    predictions = collected.candidate_predictions
    group_roots = {
        str(job.job_spec["checkpoint_group"]): job.group_root for job in collected.jobs
    }
    for group, root in group_roots.items():
        target = load_sealed_target_group(root)
        interventions = {
            resolve_cell_execution_spec(
                role=role,
                context=context,
                phase=phase,  # type: ignore[arg-type]
            ).intervention
            for role, context in execution_cells_for_phase(phase=phase)  # type: ignore[arg-type]
        }
        mappings = {
            intervention: build_metadata_intervention_mapping(
                target.block_context,
                context=intervention,
                shuffle_seed=20260904,
            )
            for intervention in interventions
        }
        group_frame = predictions.loc[
            predictions["checkpoint_group"].astype(str).eq(group)
        ]
        for role, context in execution_cells_for_phase(phase=phase):  # type: ignore[arg-type]
            cell = resolve_cell_execution_spec(
                role=role,
                context=context,
                phase=phase,  # type: ignore[arg-type]
            )
            mapping = mappings[cell.intervention]
            observed_mapping = set(
                group_frame.loc[
                    group_frame["role"].astype(str).eq(role)
                    & group_frame["context"].astype(str).eq(context),
                    "intervention_mapping_sha256",
                ].astype(str)
            )
            if observed_mapping != {mapping.sha256}:
                raise PermissionError("Candidate intervention map differs from sealed context.")
            for budget in (0, 1, 3, 5):
                batches = []
                expected_tokens: list[str] = []
                for subject, interface in target.query.episode_keys():
                    query_indices = target.query.indices_for_episode(subject, interface)
                    query = tensor_batch(target.query, query_indices, device="cpu")
                    batches.append(
                        resolve_metadata_intervention_batch(
                            subject_ids=query.subject_ids,
                            electrode_types=query.electrode_types,
                            run_ids=query.run_ids,
                            row_tokens=query.tokens,
                            base_input_sha256s=query.base_input_sha256s,
                            mapping=mapping,
                            sealed_block_context=target.block_context,
                            cell_spec=cell,
                            phase=phase,  # type: ignore[arg-type]
                            batch_kind="query",
                        )
                    )
                    expected_tokens.extend(query.tokens)
                    if budget:
                        support_indices = support_indices_for_budget(
                            target,
                            subject_id=subject,
                            electrode_type=interface,
                            budget=budget,
                        )
                        support = tensor_batch(
                            target.support,
                            support_indices,
                            device="cpu",
                        )
                        batches.append(
                            resolve_metadata_intervention_batch(
                                subject_ids=support.subject_ids,
                                electrode_types=support.electrode_types,
                                run_ids=support.run_ids,
                                row_tokens=support.tokens,
                                base_input_sha256s=support.base_input_sha256s,
                                mapping=mapping,
                                sealed_block_context=target.block_context,
                                cell_spec=cell,
                                phase=phase,  # type: ignore[arg-type]
                                batch_kind="support",
                            )
                        )
                        expected_tokens.extend(support.tokens)
                expected_usage = resolved_context_usage_sha256(
                    batches,
                    expected_row_tokens=expected_tokens,
                )
                observed_usage = set(
                    group_frame.loc[
                        group_frame["role"].astype(str).eq(role)
                        & group_frame["context"].astype(str).eq(context)
                        & pd.to_numeric(group_frame["budget"], errors="raise").eq(
                            budget
                        ),
                        "resolved_context_usage_sha256",
                    ].astype(str)
                )
                if observed_usage != {expected_usage}:
                    raise PermissionError(
                        "Candidate context-usage ledger differs from sealed row values."
                    )


def _decrypt_and_validate_query_sidecars(
    collected: _CollectedPhase,
    *,
    finalizer_private_key_path: str | Path,
    finalizer_private_key_password: bytes,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    input_root = collected.jobs[0].group_root.parent
    decision = read_json_object(input_root / "input_seal_decision.json")
    input_receipt = read_json_object(input_root / "input_seal_receipt.json")
    label_frames: list[pd.DataFrame] = []
    covariate_frames: list[pd.DataFrame] = []
    group_roots = {str(job.job_spec["checkpoint_group"]): job.group_root for job in collected.jobs}
    for group, root in group_roots.items():
        for filename, plaintext_schema in (
            ("query_labels.jsonl.enc", "cfeg.metadata-calibration-query-labels.v1"),
            (
                "query_analysis_covariates.jsonl.enc",
                "cfeg.metadata-calibration-query-analysis-covariates.v1",
            ),
        ):
            relative = f"{group}/target_query/{filename}"
            aad = {
                "candidate_id": "metadata-calibration-efficiency-v1",
                "phase": collected.manifest["phase"],
                "checkpoint_group": group,
                "input_seal_decision_signed_record_sha256": decision[
                    "signed_record_sha256"
                ],
                "token_secret_commitment_sha256": input_receipt[
                    "token_secret_domain_separated_commitment_sha256"
                ],
                "artifact_relative_path": relative,
                "plaintext_schema": plaintext_schema,
                "finalizer_public_key_fingerprint_sha256": decision[
                    "label_capability_boundary"
                ]["finalizer_public_key_fingerprint_sha256"],
            }
            plaintext = decrypt_json_for_finalizer(
                read_json_object(root / f"target_query/{filename}"),
                private_key_path=finalizer_private_key_path,
                private_key_password=finalizer_private_key_password,
                expected_associated_data=aad,
            )
            if set(plaintext) != {
                "schema",
                "candidate_id",
                "phase",
                "checkpoint_group",
                "rows",
            } or plaintext.get("schema") != plaintext_schema or plaintext.get(
                "candidate_id"
            ) != "metadata-calibration-efficiency-v1" or plaintext.get(
                "phase"
            ) != collected.manifest["phase"] or plaintext.get(
                "checkpoint_group"
            ) != group or not isinstance(plaintext.get("rows"), list):
                raise PermissionError("Decrypted query sidecar has an invalid exact envelope.")
            frame = pd.DataFrame(plaintext["rows"])
            if filename.startswith("query_labels"):
                if set(frame) != {"query_token", "label"}:
                    raise PermissionError("Decrypted query-label rows have a wrong schema.")
                label_frames.append(frame)
            else:
                if set(frame) != {
                    "query_token",
                    "headband_order",
                    "condition_period",
                    "block_number",
                }:
                    raise PermissionError("Decrypted covariate rows have a wrong schema.")
                if any(
                    type(row.get("headband_order")) is not str
                    or type(row.get("condition_period")) is not str
                    or type(row.get("block_number")) is not int
                    for row in plaintext["rows"]
                ):
                    raise PermissionError(
                        "Decrypted covariates contain a noncanonical primitive type."
                    )
                covariate_frames.append(frame)
    labels = pd.concat(label_frames, ignore_index=True)
    covariates = pd.concat(covariate_frames, ignore_index=True)
    expected_tokens = set(collected.expected_query_unlabeled["query_token"].astype(str))
    if labels["query_token"].astype(str).duplicated().any() or set(
        labels["query_token"].astype(str)
    ) != expected_tokens or covariates["query_token"].astype(str).duplicated().any() or set(
        covariates["query_token"].astype(str)
    ) != expected_tokens:
        raise PermissionError("Decrypted sidecar token coverage differs from the private seal.")
    numeric = pd.to_numeric(labels["label"], errors="raise").to_numpy()
    if not np.equal(numeric, np.floor(numeric)).all() or (
        (numeric < 0) | (numeric >= 12)
    ).any():
        raise PermissionError("Decrypted query labels fall outside the class vocabulary.")
    labels["label"] = numeric.astype(np.int64)
    expected = collected.expected_query_unlabeled.merge(
        labels,
        on="query_token",
        how="left",
        validate="one_to_one",
    )
    if expected["label"].isna().any():
        raise PermissionError("Finalizer query-label join is incomplete.")
    _validate_analysis_covariates(collected, covariates)
    _validate_query_block_class_completeness(collected, labels)
    return expected, covariates


def _validate_analysis_covariates(
    collected: _CollectedPhase, covariates: pd.DataFrame
) -> None:
    if (
        set(covariates["headband_order"].astype(str)) != {"dry", "wet"}
        or set(covariates["condition_period"].astype(str)) != {"first", "second"}
        or set(covariates["block_number"].tolist()) != {6, 7, 8, 9, 10}
    ):
        raise PermissionError("Decrypted analysis covariates have an invalid closed domain.")
    group_roots = {
        str(job.job_spec["checkpoint_group"]): job.group_root
        for job in collected.jobs
    }
    context_frames = []
    for root in group_roots.values():
        context_frames.append(
            _read_jsonl(root / "target_query/baseline_view.jsonl").loc[
                :, ["query_token", "subject_id", "electrode_type", "run_id"]
            ]
        )
    context = pd.concat(context_frames, ignore_index=True)
    merged = context.merge(
        covariates, on="query_token", how="left", validate="one_to_one"
    )
    expected_block = merged["run_id"].astype(str).str.removeprefix("block").astype(int)
    if not np.array_equal(
        merged["block_number"].to_numpy(dtype=int), expected_block.to_numpy(dtype=int)
    ):
        raise PermissionError("Decrypted block-number covariate disagrees with query context.")
    if (merged.groupby("subject_id")["headband_order"].nunique() != 1).any():
        raise PermissionError("A participant has more than one decrypted headband order.")
    expected_period = np.where(
        merged["electrode_type"].astype(str).to_numpy()
        == merged["headband_order"].astype(str).to_numpy(),
        "first",
        "second",
    )
    if not np.array_equal(
        merged["condition_period"].astype(str).to_numpy(), expected_period
    ):
        raise PermissionError("Decrypted condition period disagrees with headband order.")


def _validate_query_block_class_completeness(
    collected: _CollectedPhase,
    labels: pd.DataFrame,
) -> None:
    group_roots = {str(job.job_spec["checkpoint_group"]): job.group_root for job in collected.jobs}
    views = []
    for group, root in group_roots.items():
        view = _read_jsonl(root / "target_query/baseline_view.jsonl")
        view["checkpoint_group"] = group
        views.append(view)
    query_view = pd.concat(views, ignore_index=True)
    merged = query_view.loc[
        :, ["query_token", "subject_id", "electrode_type", "run_id", "checkpoint_group"]
    ].merge(labels, on="query_token", how="left", validate="one_to_one")
    for _key, block in merged.groupby(
        ["checkpoint_group", "subject_id", "electrode_type", "run_id"],
        sort=False,
    ):
        if len(block) != 12 or sorted(block["label"].astype(int).tolist()) != list(
            range(12)
        ):
            raise PermissionError("Decrypted query block is not exactly class complete.")


def _outer_fold_by_subject(manifest: Mapping[str, Any]) -> dict[str, int]:
    groups = manifest["execution_contract"]["phase_spec"][
        "target_subject_ids_by_checkpoint_group"
    ]
    output: dict[str, int] = {}
    for fold in (0, 1, 2):
        for subject in groups[f"fold{fold}"]:
            if str(subject) in output:
                raise ValueError("Source participant appears in more than one outer fold.")
            output[str(subject)] = fold
    return output


def _metadata_only_structure_proof(
    collected: _CollectedPhase,
    expected_query: pd.DataFrame,
) -> bool:
    _validate_query_block_class_completeness(
        collected,
        expected_query.loc[:, ["query_token", "label"]],
    )
    group_roots = {str(job.job_spec["checkpoint_group"]): job.group_root for job in collected.jobs}
    for root in group_roots.values():
        context = _read_jsonl(root / "block_context.jsonl")
        if set(context) != {
            "dataset_id",
            "subject_id",
            "electrode_type",
            "run_id",
            "impedance_channel_ids",
            "impedance_kohm_by_channel",
        } or context.duplicated(["subject_id", "electrode_type", "run_id"]).any():
            raise PermissionError("Block-context structure is not one vector per block.")
        per_cell = context.groupby(["subject_id", "electrode_type"], sort=False)[
            "run_id"
        ].agg(lambda values: set(map(str, values)))
        expected_runs = {f"block{index:02d}" for index in range(1, 11)}
        if not per_cell.map(expected_runs.__eq__).all():
            raise PermissionError("Block context does not cover exact blocks 01 through 10.")
        for row in context.itertuples(index=False):
            channels = list(row.impedance_channel_ids)
            values = np.asarray(row.impedance_kohm_by_channel, dtype=float)
            if len(channels) != 8 or values.shape != (8,) or not np.isfinite(values).all():
                raise PermissionError("Block context is not one finite official-channel vector.")
    return True


def _stage_result_bundle(
    destination: Path,
    *,
    phase: str,
    candidate: Mapping[str, Any],
    baseline: BaselineAnalysisBundle,
    covariates: pd.DataFrame,
    gate_or_claim: Mapping[str, Any],
    bindings: MetadataCalibrationBundleBindings,
) -> tuple[Path, str]:
    destination = destination.absolute()
    _ensure_durable_private_directory(destination.parent)
    staging = destination.parent / f".{destination.name}.staging-{uuid.uuid4().hex}"
    staging.mkdir(mode=0o700)
    _fsync_directory(staging)
    _fsync_directory(staging.parent)
    try:
        frames = {
            "candidate_participant_balanced_accuracy.csv": candidate[
                "participant_balanced_accuracy"
            ],
            "candidate_participant_contrasts.csv": candidate[
                "participant_contrasts"
            ],
            "baseline_participant_balanced_accuracy.csv": (
                baseline.participant_balanced_accuracy
            ),
            "baseline_summary.csv": baseline.summary,
        }
        for name, frame in frames.items():
            assert isinstance(frame, pd.DataFrame)
            frame.sort_values(list(frame.columns), kind="mergesort").to_csv(
                staging / name,
                index=False,
                float_format="%.17g",
            )
        analysis_metadata = {
            key: value
            for key, value in candidate.items()
            if not isinstance(value, pd.DataFrame)
        }
        _write_json(
            staging / "analysis_metadata.json",
            {
                "schema": "cfeg.metadata-calibration-published-analysis-metadata.v1",
                "phase": phase,
                "candidate": analysis_metadata,
                "baseline": baseline.metadata,
                "bindings": bindings.as_dict(),
                "query_level_outputs_published": False,
            },
        )
        _write_json(staging / "gate_or_claim_result.json", dict(gate_or_claim))
        _write_json(
            staging / "analysis_covariate_summary.json",
            _covariate_summary(covariates),
        )
        file_records = [
            {
                "relative_path": path.name,
                "size_bytes": path.stat().st_size,
                "file_sha256": _sha256_file(path),
            }
            for path in sorted(staging.iterdir())
        ]
        bundle = {
            "schema": "cfeg.metadata-calibration-result-bundle.v1",
            "candidate_id": "metadata-calibration-efficiency-v1",
            "phase": phase,
            "query_outcomes_loaded": True,
            "query_level_outputs_published": False,
            "completion_receipt_excluded_from_result_content_hash": True,
            "files": file_records,
        }
        bundle["result_bundle_sha256"] = canonical_json_sha256(bundle)
        _write_json(staging / "result_bundle_manifest.json", bundle)
        return staging, str(bundle["result_bundle_sha256"])
    except BaseException:
        if os.path.lexists(staging):
            shutil.rmtree(staging)
        raise


def _covariate_summary(covariates: pd.DataFrame) -> dict[str, Any]:
    output: dict[str, Any] = {
        "schema": "cfeg.metadata-calibration-analysis-covariate-summary.v1",
        "n_query_rows": len(covariates),
        "query_tokens_published": False,
    }
    for column in ("headband_order", "condition_period", "block_number"):
        values = covariates[column].map(
            lambda value: "<missing>" if pd.isna(value) else str(value)
        )
        output[f"{column}_counts"] = {
            key: int(value) for key, value in values.value_counts().sort_index().items()
        }
    return output


def _build_development_gate_receipt(
    collected: _CollectedPhase,
    *,
    signed_private: Mapping[str, Any],
    private_seal_path: Path,
    result_bundle_root: Path,
    result_bundle_sha256: str,
    development_gate_result: Mapping[str, Any],
    outcome_access_claim: OutcomeAccessClaimBinding,
    authorization_basis: str,
) -> dict[str, Any]:
    passed = development_gate_result.get("passed") is True
    return {
        "schema": DEVELOPMENT_GATE_RECEIPT_SCHEMA,
        "candidate_id": "metadata-calibration-efficiency-v1",
        "phase": "source_development",
        "development_gate_status": "pass" if passed else "fail",
        "source_phase_manifest_sha256": collected.manifest[
            "phase_manifest_sha256"
        ],
        "source_outcome_decision_signed_record_sha256": collected.decision[
            "signed_record_sha256"
        ],
        "source_input_seal_receipt_signed_record_sha256": collected.manifest[
            "input_seal_receipt_signed_record_sha256"
        ],
        "source_input_artifact_tree_sha256": collected.manifest[
            "input_artifact_tree_sha256"
        ],
        "source_execution_contract_sha256": collected.manifest[
            "execution_contract_sha256"
        ],
        "source_execution_output_root": collected.manifest[
            "execution_output_root"
        ],
        "filesystem_private_seal_file_sha256": _sha256_file(private_seal_path),
        "filesystem_private_seal_signed_record_sha256": signed_private[
            "signed_record_sha256"
        ],
        "filesystem_private_tree_sha256": signed_private[
            "filesystem_private_tree_sha256"
        ],
        "source_result_bundle_root": str(result_bundle_root),
        "source_result_bundle_sha256": result_bundle_sha256,
        "development_gate_result": dict(development_gate_result),
        "outcome_finalization_performed": True,
        "outcome_access_claim_path": str(outcome_access_claim.claim_path),
        "outcome_access_claim_file_sha256": outcome_access_claim.claim_file_sha256,
        "outcome_access_claim_signed_record_sha256": (
            outcome_access_claim.signed_record_sha256
        ),
        "outcome_access_scope_sha256": outcome_access_claim.outcome_access_scope_sha256,
        "held_input_access_authorized": passed,
        "source_commit": collected.manifest["source_commit"],
        "source_tree_sha256": collected.manifest["source_tree_sha256"],
        "source_tag": collected.manifest["source_tag"],
        "signing_key_id": collected.decision["signing_key_id"],
        "signer_role": collected.decision["signer_role"],
        "authorization_basis": authorization_basis,
        "signature_algorithm": "ed25519",
    }


def _build_held_completion_receipt(
    collected: _CollectedPhase,
    *,
    signed_private: Mapping[str, Any],
    private_seal_path: Path,
    result_bundle_root: Path,
    result_bundle_sha256: str,
    held_claim_result: Mapping[str, Any],
    outcome_access_claim: OutcomeAccessClaimBinding,
    authorization_basis: str,
) -> dict[str, Any]:
    return {
        "schema": HELD_COMPLETION_RECEIPT_SCHEMA,
        "candidate_id": "metadata-calibration-efficiency-v1",
        "phase": "held_participant_evaluation",
        "status": "held_outcomes_finalized_once",
        "phase_manifest_sha256": collected.manifest["phase_manifest_sha256"],
        "outcome_decision_signed_record_sha256": collected.decision[
            "signed_record_sha256"
        ],
        "authorization_envelope_sha256": collected.envelope[
            "authorization_envelope_sha256"
        ],
        "input_seal_receipt_signed_record_sha256": collected.manifest[
            "input_seal_receipt_signed_record_sha256"
        ],
        "input_artifact_tree_sha256": collected.manifest[
            "input_artifact_tree_sha256"
        ],
        "execution_contract_sha256": collected.manifest[
            "execution_contract_sha256"
        ],
        "filesystem_private_seal_file_sha256": _sha256_file(private_seal_path),
        "filesystem_private_seal_signed_record_sha256": signed_private[
            "signed_record_sha256"
        ],
        "filesystem_private_tree_sha256": signed_private[
            "filesystem_private_tree_sha256"
        ],
        "held_result_bundle_root": str(result_bundle_root),
        "held_result_bundle_sha256": result_bundle_sha256,
        "held_claim_result": dict(held_claim_result),
        "outcome_finalization_performed": True,
        "query_outcomes_loaded": True,
        "outcome_access_claim_path": str(outcome_access_claim.claim_path),
        "outcome_access_claim_file_sha256": outcome_access_claim.claim_file_sha256,
        "outcome_access_claim_signed_record_sha256": (
            outcome_access_claim.signed_record_sha256
        ),
        "outcome_access_scope_sha256": outcome_access_claim.outcome_access_scope_sha256,
        "source_development_gate_binding": collected.manifest[
            "source_development_gate_binding"
        ],
        "source_commit": collected.manifest["source_commit"],
        "source_tree_sha256": collected.manifest["source_tree_sha256"],
        "source_tag": collected.manifest["source_tag"],
        "signing_key_id": collected.decision["signing_key_id"],
        "signer_role": collected.decision["signer_role"],
        "authorization_basis": authorization_basis,
        "signature_algorithm": "ed25519",
    }


def _lock_and_fsync_result_tree(root: Path) -> None:
    for path in root.iterdir():
        os.chmod(path, 0o400)
        descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    os.chmod(root, 0o700)
    descriptor = os.open(root, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _ensure_durable_private_directory(path: Path) -> None:
    path = _absolute_no_symlink_path(path, must_exist=False)
    missing: list[Path] = []
    current = path
    while not current.exists():
        missing.append(current)
        current = current.parent
    for target in reversed(missing):
        try:
            target.mkdir(mode=0o700)
        except FileExistsError:
            pass
        observed = target.lstat()
        if not stat.S_ISDIR(observed.st_mode) or stat.S_ISLNK(observed.st_mode):
            raise ValueError("Result hierarchy contains a non-directory component.")
        os.chmod(target, 0o700)
        _fsync_directory(target)
        _fsync_directory(target.parent)
    observed = path.lstat()
    if not stat.S_ISDIR(observed.st_mode) or stat.S_ISLNK(observed.st_mode):
        raise ValueError("Result parent must be one real directory.")
    os.chmod(path, 0o700)
    _fsync_directory(path)
    _fsync_directory(path.parent)


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(
        path,
        os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0),
    )
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.write_bytes(canonical_json_bytes(value) + b"\n")


def _read_jsonl(path: Path) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            value = json.loads(line)
            if not isinstance(value, dict):
                raise TypeError("Finalizer JSONL rows must be objects.")
            rows.append(value)
    if not rows:
        raise ValueError("Finalizer JSONL input must be nonempty.")
    return pd.DataFrame(rows)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    try:
        while True:
            chunk = os.read(descriptor, 1 << 20)
            if not chunk:
                break
            digest.update(chunk)
    finally:
        os.close(descriptor)
    return digest.hexdigest()


def _is_sha256(value: object) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(
        character in "0123456789abcdef" for character in value
    )


def _is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


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
            raise ValueError(f"Finalization path contains a symlink component: {component}.")
    if must_exist and not absolute.exists():
        raise FileNotFoundError(absolute)
    return absolute
