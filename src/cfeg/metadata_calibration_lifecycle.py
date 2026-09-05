from __future__ import annotations

import ctypes
import errno
import hashlib
import json
import os
import resource
import shutil
import stat
import subprocess
import sys
import uuid
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import torch

from cfeg.metadata_calibration_attempt import (
    CANONICAL_OUTCOME_ACCESS_REGISTRY_ROOT,
    OutcomeAccessAlreadyClaimed,
    audit_outcome_access_registry_for_run,
    hold_outcome_access_scope_lease,
)
from cfeg.metadata_calibration_authority import (
    generate_authority_keypair,
    generate_finalizer_keypair,
    issue_signed_record,
    read_json_object,
    verify_signed_record,
)
from cfeg.metadata_calibration_contract import validate_metadata_calibration_plan
from cfeg.metadata_calibration_finalizer import (
    PhaseFinalizationResult,
    finalize_metadata_calibration_phase,
    validate_development_completion_receipt,
    validate_held_completion_receipt,
)
from cfeg.metadata_calibration_isolation import (
    WORKER_ISOLATION_CONTRACT_SHA256,
)
from cfeg.metadata_calibration_job import (
    build_metadata_calibration_job_capability,
    publish_validated_metadata_calibration_job_spool,
    resolve_authorized_metadata_calibration_job,
    runner_input_relative_paths,
    validate_metadata_calibration_job_staging,
    validate_published_metadata_calibration_job,
)
from cfeg.metadata_calibration_manifest import (
    build_authorization_envelope,
    build_metadata_calibration_execution_contract,
    build_outcome_execution_decision,
    build_phase_execution_manifest,
)
from cfeg.metadata_calibration_sealer import (
    build_input_seal_decision,
    seal_metadata_calibration_inputs,
    write_signed_decision_exclusive,
)

ExperimentPhase = Literal["source_development", "held_participant_evaluation"]
_REPOSITORY = Path(__file__).resolve().parents[2]
_LIFECYCLE_COMPLETION_FIELDS = {
    "schema",
    "candidate_id",
    "status",
    "run_root",
    "source_completion_receipt_file_sha256",
    "source_result_bundle_sha256",
    "source_gate_passed",
    "held_phase_opened",
    "held_completion_receipt_file_sha256",
    "held_result_bundle_sha256",
    "private_key_absence_required_for_canonical_publication",
    "completion_publication_precondition",
    "worker_isolation_contract_sha256",
    "worker_isolation",
    "worker_code_snapshot",
    "authority_private_key_encrypted",
    "finalizer_private_key_encrypted",
    "label_capable_parent_dumpable",
    "core_dump_limit_bytes",
    "python_secret_memory_zeroization_guaranteed",
    "decision_date",
    "signing_key_id",
    "signer_role",
    "authorization_basis",
    "signature_algorithm",
    "lifecycle_payload_sha256",
    "signature",
    "signed_record_sha256",
}
_LIFECYCLE_FAILURE_FIELDS = {
    "schema",
    "candidate_id",
    "status",
    "current_run_outcome_access_started",
    "failed_phase",
    "claims_by_phase",
    "failed_phase_claimed",
    "recovery_disposition",
    "blocked_by_prior_outcome_claim",
    "outcome_registry_integrity_failure",
    "same_cohort_automatic_retry_permitted",
    "blocking_claim_path",
    "blocking_scope_sha256",
    "blocking_claim_file_sha256",
    "run_root",
    "error_type",
    "error_message",
    "error_detail_sha256",
    "post_outcome_error_detail_redacted",
    "decision_date",
    "signing_key_id",
    "signer_role",
    "authorization_basis",
    "signature_algorithm",
    "lifecycle_payload_sha256",
    "signature",
    "signed_record_sha256",
}


@dataclass(frozen=True)
class MetadataCalibrationLifecycleResult:
    run_root: Path
    lifecycle_receipt_path: Path
    source: PhaseFinalizationResult
    held: PhaseFinalizationResult | None
    source_gate_passed: bool
    generated_private_keys_destroyed: bool


class PostOutcomeFinalizationFailure(RuntimeError):
    """Constant outward failure after an irreversible outcome claim."""


def run_metadata_calibration_lifecycle(
    *,
    run_root: str | Path,
    processed_asset_root: str | Path,
    source_tag: str,
    decision_date: str,
    authorization_basis: str,
    python_executable: str | Path | None = None,
) -> MetadataCalibrationLifecycleResult:
    """Run source once and conditionally run held once under signed authority."""

    plan_binding = validate_metadata_calibration_plan()
    executable = _validate_runtime(python_executable)
    asset_root = _absolute_no_symlink_path(Path(processed_asset_root), must_exist=True)
    requested_run_root = _absolute_no_symlink_path(Path(run_root), must_exist=False)
    registry_root = _absolute_no_symlink_path(
        CANONICAL_OUTCOME_ACCESS_REGISTRY_ROOT, must_exist=False
    )
    _validate_capability_path_disjointness(
        repository=_REPOSITORY,
        processed_asset_root=asset_root,
        run_root=requested_run_root,
        outcome_access_registry_root=registry_root,
    )
    if plan_binding.get("outcome_execution_authorized") is not True:
        raise PermissionError("The frozen research plan does not authorize outcome execution.")
    if authorization_basis != plan_binding.get("authorization_basis"):
        raise PermissionError("Lifecycle authorization differs from the frozen owner directive.")
    _harden_label_capable_parent_process()
    root = _new_private_run_root(requested_run_root)
    worker_code_snapshot = _prepare_worker_code_snapshot(
        root / "worker-code-snapshot", source_tag=source_tag
    )

    keys = root / "keys"
    authority_private = keys / "authority-signing-private.pem"
    authority_public = keys / "authority-signing-public.pem"
    finalizer_private = keys / "finalizer-label-private.pem"
    finalizer_public = keys / "finalizer-label-public.pem"
    authority_password = bytearray(os.urandom(48))
    finalizer_password = bytearray(os.urandom(48))
    authority_key_id: str | None = None
    source_result: PhaseFinalizationResult | None = None
    held_result: PhaseFinalizationResult | None = None
    lifecycle_receipt = root / "lifecycle_completion.json"
    lifecycle_pending = root / "lifecycle_completion.pending.json"
    private_keys_destroyed = False
    signed_summary: dict[str, Any] | None = None
    active_phase: ExperimentPhase = "source_development"
    try:
        authority_key_id = generate_authority_keypair(
            private_key_path=authority_private,
            public_key_path=authority_public,
            password=bytes(authority_password),
        )
        generate_finalizer_keypair(
            private_key_path=finalizer_private,
            public_key_path=finalizer_public,
            key_size=3072,
            password=bytes(finalizer_password),
        )
        _emit("keys_ready", run_root=str(root), authority_key_id=authority_key_id)

        source_result = _run_phase(
            phase="source_development",
            phase_root=root / "source",
            processed_asset_root=asset_root,
            source_tag=source_tag,
            decision_date=decision_date,
            authorization_basis=authorization_basis,
            authority_key_id=authority_key_id,
            authority_private_key=authority_private,
            authority_public_key=authority_public,
            finalizer_private_key=finalizer_private,
            finalizer_public_key=finalizer_public,
            finalizer_password=bytes(finalizer_password),
            authority_password=bytes(authority_password),
            python_executable=executable,
            worker_code_snapshot=worker_code_snapshot,
        )
        source_completion = read_json_object(source_result.completion_receipt_path)
        source_gate_passed = source_completion["development_gate_status"] == "pass"
        _emit(
            "source_finalized",
            gate_status=source_completion["development_gate_status"],
            result_bundle_sha256=source_result.result_bundle_sha256,
        )
        if source_gate_passed:
            active_phase = "held_participant_evaluation"
            held_result = _run_phase(
                phase="held_participant_evaluation",
                phase_root=root / "held",
                processed_asset_root=asset_root,
                source_tag=source_tag,
                decision_date=decision_date,
                authorization_basis=authorization_basis,
                authority_key_id=authority_key_id,
                authority_private_key=authority_private,
                authority_public_key=authority_public,
                finalizer_private_key=finalizer_private,
                finalizer_public_key=finalizer_public,
                finalizer_password=bytes(finalizer_password),
                authority_password=bytes(authority_password),
                python_executable=executable,
                worker_code_snapshot=worker_code_snapshot,
                source_gate_receipt=source_result.completion_receipt_path,
                source_private_seal=source_result.filesystem_private_seal_path,
            )
            _emit(
                "held_finalized",
                result_bundle_sha256=held_result.result_bundle_sha256,
            )
        else:
            _emit("held_not_opened", reason="source_development_gate_failed")

        summary = {
            "schema": "cfeg.metadata-calibration-lifecycle-completion.v1",
            "candidate_id": "metadata-calibration-efficiency-v1",
            "status": "completed",
            "run_root": str(root),
            "source_completion_receipt_file_sha256": _sha256_file(
                source_result.completion_receipt_path
            ),
            "source_result_bundle_sha256": source_result.result_bundle_sha256,
            "source_gate_passed": source_gate_passed,
            "held_phase_opened": held_result is not None,
            "held_completion_receipt_file_sha256": (
                None
                if held_result is None
                else _sha256_file(held_result.completion_receipt_path)
            ),
            "held_result_bundle_sha256": (
                None if held_result is None else held_result.result_bundle_sha256
            ),
            "private_key_absence_required_for_canonical_publication": True,
            "completion_publication_precondition": (
                "both_generated_private_keys_absent_and_parent_directories_fsynced"
            ),
            "worker_isolation_contract_sha256": WORKER_ISOLATION_CONTRACT_SHA256,
            "worker_isolation": "bubblewrap_per_job_fresh_spool_no_raw_or_key_mount",
            "worker_code_snapshot": "clean_local_clone_single_source_tag_commit",
            "authority_private_key_encrypted": True,
            "finalizer_private_key_encrypted": True,
            "label_capable_parent_dumpable": False,
            "core_dump_limit_bytes": 0,
            "python_secret_memory_zeroization_guaranteed": False,
            "decision_date": decision_date,
            "signing_key_id": authority_key_id,
            "signer_role": "delegated_automation_authority",
            "authorization_basis": authorization_basis,
            "signature_algorithm": "ed25519",
        }
        signed_summary = issue_signed_record(
            summary,
            private_key_path=authority_private,
            private_key_password=bytes(authority_password),
            hash_field="lifecycle_payload_sha256",
        )
        write_signed_decision_exclusive(lifecycle_pending, signed_summary)
    except BaseException as error:
        try:
            registry_audit = audit_outcome_access_registry_for_run(
                root, registry_root=registry_root
            )
        except BaseException:  # noqa: BLE001 - registry uncertainty must redact fail-closed
            raise PostOutcomeFinalizationFailure(
                "Lifecycle failure could not safely audit the permanent outcome registry; "
                "manual security review is required."
            ) from None
        redact_outward = (
            registry_audit.current_run_claim_found
            or registry_audit.registry_integrity_failure
            or isinstance(error, OutcomeAccessAlreadyClaimed)
        )
        if authority_key_id is not None and authority_private.exists():
            try:
                _write_failure_receipt(
                    root,
                    error=error,
                    decision_date=decision_date,
                    authorization_basis=authorization_basis,
                    authority_key_id=authority_key_id,
                    authority_private_key=authority_private,
                    authority_password=bytes(authority_password),
                    current_run_outcome_access_started=(
                        registry_audit.current_run_claim_found
                    ),
                    failed_phase=active_phase,
                    current_run_claimed_phases=(
                        registry_audit.current_run_claimed_phases
                    ),
                    registry_integrity_failure=(
                        registry_audit.registry_integrity_failure
                    ),
                    blocked_by_prior_claim=isinstance(
                        error, OutcomeAccessAlreadyClaimed
                    ),
                    blocking_claim_path=(
                        error.claim_path
                        if isinstance(error, OutcomeAccessAlreadyClaimed)
                        else None
                    ),
                    blocking_scope_sha256=(
                        error.outcome_access_scope_sha256
                        if isinstance(error, OutcomeAccessAlreadyClaimed)
                        else None
                    ),
                )
            except BaseException:
                if redact_outward:
                    raise PostOutcomeFinalizationFailure(
                        "Outcome finalization failed after an irreversible claim; "
                        "failure-receipt publication also failed."
                    ) from None
                raise
        elif redact_outward:
            raise PostOutcomeFinalizationFailure(
                "Outcome finalization failed after an irreversible claim, but the "
                "run signing key was unavailable for a failure receipt."
            ) from None
        if redact_outward:
            raise PostOutcomeFinalizationFailure(
                "Outcome finalization failed after an irreversible claim; inspect the "
                "signed lifecycle failure receipt under the private run root."
            ) from None
        raise
    finally:
        private_keys_destroyed = _destroy_generated_private_keys(
            (authority_private, finalizer_private)
        )
        if not private_keys_destroyed:
            try:
                _write_key_destruction_failure(root, (authority_private, finalizer_private))
            except (OSError, TypeError, ValueError) as error:
                _emit(
                    "private_key_destruction_failure_receipt_error",
                    error_type=type(error).__name__,
                    error_message=str(error),
                )
        for index in range(len(authority_password)):
            authority_password[index] = 0
        for index in range(len(finalizer_password)):
            finalizer_password[index] = 0

    if source_result is None:
        raise RuntimeError("Lifecycle exited without a source finalization result.")
    if not private_keys_destroyed:
        raise RuntimeError(
            "Lifecycle results were finalized, but private-key destruction failed; "
            "completion was not published."
        )
    if signed_summary is None:
        raise RuntimeError("Lifecycle exited without a signed completion record.")
    _validate_metadata_calibration_lifecycle_record(lifecycle_pending, pending=True)
    try:
        _publish_signed_pending_noreplace(
            lifecycle_pending,
            lifecycle_receipt,
            private_key_paths=(authority_private, finalizer_private),
        )
    except BaseException as error:
        _write_completion_publication_failure_marker(
            root,
            pending_path=lifecycle_pending,
            canonical_path=lifecycle_receipt,
            error=error,
        )
        raise
    validate_metadata_calibration_lifecycle_completion(lifecycle_receipt)
    return MetadataCalibrationLifecycleResult(
        run_root=root,
        lifecycle_receipt_path=lifecycle_receipt,
        source=source_result,
        held=held_result,
        source_gate_passed=source_result.held_evaluation_access_authorized,
        generated_private_keys_destroyed=private_keys_destroyed,
    )


def validate_metadata_calibration_lifecycle_completion(
    receipt_path: str | Path,
) -> str:
    """Authenticate the canonical lifecycle and every phase artifact it binds."""

    return _validate_metadata_calibration_lifecycle_record(
        Path(receipt_path).absolute(),
        pending=False,
    )


def recover_metadata_calibration_lifecycle_completion(
    run_root: str | Path,
) -> str:
    """Publish a fully validated signed pending completion after a crash."""

    root = _absolute_no_symlink_path(Path(run_root), must_exist=True)
    pending = root / "lifecycle_completion.pending.json"
    canonical = root / "lifecycle_completion.json"
    _validate_metadata_calibration_lifecycle_record(pending, pending=True)
    private_paths = (
        root / "keys" / "authority-signing-private.pem",
        root / "keys" / "finalizer-label-private.pem",
    )
    _publish_signed_pending_noreplace(
        pending,
        canonical,
        private_key_paths=private_paths,
    )
    return validate_metadata_calibration_lifecycle_completion(canonical)


def _validate_metadata_calibration_lifecycle_record(
    path: Path,
    *,
    pending: bool,
) -> str:
    """Authenticate a canonical or recoverable pending lifecycle record."""

    root = path.parent
    expected_name = (
        "lifecycle_completion.pending.json"
        if pending
        else "lifecycle_completion.json"
    )
    if path != root / expected_name:
        raise PermissionError("Lifecycle completion path is noncanonical.")
    observed = path.lstat()
    if (
        not stat.S_ISREG(observed.st_mode)
        or stat.S_ISLNK(observed.st_mode)
        or observed.st_nlink != 1
        or stat.S_IMODE(observed.st_mode) != 0o400
    ):
        raise PermissionError("Lifecycle completion is not one immutable regular file.")
    if pending:
        if os.path.lexists(root / "lifecycle_completion.json"):
            raise PermissionError("Pending lifecycle recovery cannot replace completion.")
    elif os.path.lexists(root / "lifecycle_completion.pending.json"):
        raise PermissionError("Canonical lifecycle completion retains a pending duplicate.")
    public_key = root / "keys" / "authority-signing-public.pem"
    receipt = read_json_object(path)
    signed_hash = verify_signed_record(
        receipt,
        public_key_path=public_key,
        hash_field="lifecycle_payload_sha256",
        expected_schema="cfeg.metadata-calibration-lifecycle-completion.v1",
    )
    expected_common = {
        "candidate_id": "metadata-calibration-efficiency-v1",
        "status": "completed",
        "run_root": str(root),
        "private_key_absence_required_for_canonical_publication": True,
        "completion_publication_precondition": (
            "both_generated_private_keys_absent_and_parent_directories_fsynced"
        ),
        "worker_isolation_contract_sha256": WORKER_ISOLATION_CONTRACT_SHA256,
        "worker_isolation": "bubblewrap_per_job_fresh_spool_no_raw_or_key_mount",
        "worker_code_snapshot": "clean_local_clone_single_source_tag_commit",
        "authority_private_key_encrypted": True,
        "finalizer_private_key_encrypted": True,
        "label_capable_parent_dumpable": False,
        "core_dump_limit_bytes": 0,
        "python_secret_memory_zeroization_guaranteed": False,
        "signer_role": "delegated_automation_authority",
        "signature_algorithm": "ed25519",
    }
    if set(receipt) != _LIFECYCLE_COMPLETION_FIELDS or any(
        receipt.get(key) != value for key, value in expected_common.items()
    ):
        raise PermissionError("Lifecycle completion has a wrong exact schema or state.")
    for private_name in (
        "authority-signing-private.pem",
        "finalizer-label-private.pem",
    ):
        if os.path.lexists(root / "keys" / private_name):
            raise PermissionError("Lifecycle completion was published with a private key.")

    source_root = root / "source"
    source_receipt_path = source_root / "results" / "completion_receipt.json"
    source_private = source_root / "private" / "filesystem-private-seal.json"
    source_manifest = read_json_object(source_root / "authority" / "phase-manifest.json")
    validate_development_completion_receipt(
        source_receipt_path,
        private_seal_path=source_private,
        manifest=source_manifest,
        trusted_signing_public_key_path=public_key,
        physical_result_bundle_root=source_root / "results",
        expected_result_bundle_root=source_root / "results",
    )
    source_receipt = read_json_object(source_receipt_path)
    source_passed = source_receipt.get("development_gate_status") == "pass"
    if (
        receipt.get("source_completion_receipt_file_sha256")
        != _sha256_file(source_receipt_path)
        or receipt.get("source_result_bundle_sha256")
        != source_receipt.get("source_result_bundle_sha256")
        or receipt.get("source_gate_passed") is not source_passed
        or receipt.get("held_phase_opened") is not source_passed
        or receipt.get("signing_key_id") != source_receipt.get("signing_key_id")
        or receipt.get("authorization_basis")
        != source_receipt.get("authorization_basis")
    ):
        raise PermissionError("Lifecycle completion differs from the source phase.")
    if not source_passed:
        if (
            receipt.get("held_completion_receipt_file_sha256") is not None
            or receipt.get("held_result_bundle_sha256") is not None
            or os.path.lexists(root / "held")
        ):
            raise PermissionError("A failed source gate cannot bind or expose held results.")
        return signed_hash

    held_root = root / "held"
    held_receipt_path = held_root / "results" / "completion_receipt.json"
    held_private = held_root / "private" / "filesystem-private-seal.json"
    held_manifest = read_json_object(held_root / "authority" / "phase-manifest.json")
    validate_held_completion_receipt(
        held_receipt_path,
        private_seal_path=held_private,
        manifest=held_manifest,
        trusted_signing_public_key_path=public_key,
        physical_result_bundle_root=held_root / "results",
        expected_result_bundle_root=held_root / "results",
    )
    held_receipt = read_json_object(held_receipt_path)
    if (
        receipt.get("held_completion_receipt_file_sha256")
        != _sha256_file(held_receipt_path)
        or receipt.get("held_result_bundle_sha256")
        != held_receipt.get("held_result_bundle_sha256")
        or receipt.get("signing_key_id") != held_receipt.get("signing_key_id")
        or receipt.get("authorization_basis")
        != held_receipt.get("authorization_basis")
    ):
        raise PermissionError("Lifecycle completion differs from the held phase.")
    return signed_hash


def _run_phase(
    *,
    phase: ExperimentPhase,
    phase_root: Path,
    processed_asset_root: Path,
    source_tag: str,
    decision_date: str,
    authorization_basis: str,
    authority_key_id: str,
    authority_private_key: Path,
    authority_public_key: Path,
    finalizer_private_key: Path,
    finalizer_public_key: Path,
    finalizer_password: bytes,
    authority_password: bytes,
    python_executable: Path,
    worker_code_snapshot: Path,
    source_gate_receipt: Path | None = None,
    source_private_seal: Path | None = None,
) -> PhaseFinalizationResult:
    scope_manifest = {
        "candidate_id": "metadata-calibration-efficiency-v1",
        "phase": phase,
        "execution_contract": build_metadata_calibration_execution_contract(
            phase=phase
        ),
    }
    with hold_outcome_access_scope_lease(
        scope_manifest,
        registry_root=CANONICAL_OUTCOME_ACCESS_REGISTRY_ROOT,
    ):
        return _run_phase_under_lease(
            phase=phase,
            phase_root=phase_root,
            processed_asset_root=processed_asset_root,
            source_tag=source_tag,
            decision_date=decision_date,
            authorization_basis=authorization_basis,
            authority_key_id=authority_key_id,
            authority_private_key=authority_private_key,
            authority_public_key=authority_public_key,
            finalizer_private_key=finalizer_private_key,
            finalizer_public_key=finalizer_public_key,
            finalizer_password=finalizer_password,
            authority_password=authority_password,
            python_executable=python_executable,
            worker_code_snapshot=worker_code_snapshot,
            source_gate_receipt=source_gate_receipt,
            source_private_seal=source_private_seal,
        )


def _run_phase_under_lease(
    *,
    phase: ExperimentPhase,
    phase_root: Path,
    processed_asset_root: Path,
    source_tag: str,
    decision_date: str,
    authorization_basis: str,
    authority_key_id: str,
    authority_private_key: Path,
    authority_public_key: Path,
    finalizer_private_key: Path,
    finalizer_public_key: Path,
    finalizer_password: bytes,
    authority_password: bytes,
    python_executable: Path,
    worker_code_snapshot: Path,
    source_gate_receipt: Path | None = None,
    source_private_seal: Path | None = None,
) -> PhaseFinalizationResult:
    phase_slug = "source" if phase == "source_development" else "held"
    authority_root = phase_root / "authority"
    input_root = phase_root / "input-seal"
    execution_root = phase_root / "execution"
    private_seal = phase_root / "private" / "filesystem-private-seal.json"
    result_root = phase_root / "results"
    input_decision_path = authority_root / "input-seal-decision.json"
    manifest_path = authority_root / "phase-manifest.json"
    outcome_decision_path = authority_root / "outcome-decision.json"
    envelope_path = authority_root / "authorization-envelope.json"
    token_secret = os.urandom(32)

    unsigned_input = build_input_seal_decision(
        phase=phase,
        decision_id=f"DEC-{decision_date.replace('-', '')}-MC-{phase_slug.upper()}-SEAL",
        decision_date=decision_date,
        processed_asset_root=processed_asset_root,
        output_root=input_root,
        token_secret=token_secret,
        finalizer_public_key_path=finalizer_public_key,
        trusted_signing_public_key_path=authority_public_key,
        signing_key_id=authority_key_id,
        source_tag=source_tag,
        authorization_basis=authorization_basis,
        source_development_gate_receipt_path=source_gate_receipt,
        source_development_private_seal_path=source_private_seal,
    )
    signed_input = issue_signed_record(
        unsigned_input,
        private_key_path=authority_private_key,
        private_key_password=authority_password,
        hash_field="decision_payload_sha256",
    )
    write_signed_decision_exclusive(input_decision_path, signed_input)
    seal_metadata_calibration_inputs(
        signed_input,
        token_secret=token_secret,
        trusted_signing_public_key_path=authority_public_key,
        finalizer_public_key_path=finalizer_public_key,
        receipt_signing_private_key_path=authority_private_key,
        receipt_signing_private_key_password=authority_password,
    )
    del token_secret
    _emit("input_sealed", phase=phase, input_root=str(input_root))

    manifest = build_phase_execution_manifest(
        phase=phase,
        input_seal_root=input_root,
        execution_output_root=execution_root,
        trusted_signing_public_key_path=authority_public_key,
        source_tag=source_tag,
        source_development_gate_receipt_path=source_gate_receipt,
        source_development_private_seal_path=source_private_seal,
    )
    write_signed_decision_exclusive(manifest_path, manifest)
    unsigned_outcome = build_outcome_execution_decision(
        manifest,
        trusted_signing_public_key_path=authority_public_key,
        signing_key_id=authority_key_id,
        decision_id=f"DEC-{decision_date.replace('-', '')}-MC-{phase_slug.upper()}-EXECUTE",
        decision_date=decision_date,
        authorization_basis=authorization_basis,
    )
    signed_outcome = issue_signed_record(
        unsigned_outcome,
        private_key_path=authority_private_key,
        private_key_password=authority_password,
        hash_field="decision_payload_sha256",
    )
    write_signed_decision_exclusive(outcome_decision_path, signed_outcome)
    envelope = build_authorization_envelope(
        manifest=manifest,
        outcome_decision=signed_outcome,
        trusted_signing_public_key_path=authority_public_key,
    )
    write_signed_decision_exclusive(envelope_path, envelope)
    _emit("phase_authorized", phase=phase, jobs=manifest["expected_job_count"])

    _run_workers(
        phase=phase,
        phase_root=phase_root,
        manifest_path=manifest_path,
        outcome_decision_path=outcome_decision_path,
        envelope_path=envelope_path,
        authority_public_key=authority_public_key,
        jobs=manifest["execution_contract"]["jobs"],
        python_executable=python_executable,
        worker_code_snapshot=worker_code_snapshot,
        authority_private_key=authority_private_key,
        authority_password=authority_password,
        authorization_basis=authorization_basis,
    )
    _emit("private_grid_complete", phase=phase)
    return finalize_metadata_calibration_phase(
        manifest_path=manifest_path,
        outcome_decision_path=outcome_decision_path,
        authorization_envelope_path=envelope_path,
        trusted_signing_public_key_path=authority_public_key,
        authority_signing_private_key_path=authority_private_key,
        authority_signing_private_key_password=authority_password,
        finalizer_private_key_path=finalizer_private_key,
        finalizer_private_key_password=finalizer_password,
        filesystem_private_seal_path=private_seal,
        result_bundle_root=result_root,
        completion_receipt_path=result_root / "completion_receipt.json",
        authorization_basis=authorization_basis,
        outcome_access_registry_root=CANONICAL_OUTCOME_ACCESS_REGISTRY_ROOT,
    )


def _run_workers(
    *,
    phase: ExperimentPhase,
    phase_root: Path,
    manifest_path: Path,
    outcome_decision_path: Path,
    envelope_path: Path,
    authority_public_key: Path,
    jobs: list[dict[str, Any]],
    python_executable: Path,
    worker_code_snapshot: Path,
    authority_private_key: Path,
    authority_password: bytes,
    authorization_basis: str,
) -> None:
    log_root = phase_root / "logs"
    _ensure_private_directory(log_root)
    spool_parent = phase_root / ".worker-spool"
    _ensure_private_directory(spool_parent)
    total = len(jobs)
    manifest = read_json_object(manifest_path)
    outcome_decision = read_json_object(outcome_decision_path)
    envelope = read_json_object(envelope_path)
    for index, job_spec in enumerate(jobs, start=1):
        job_id = str(job_spec["job_id"])
        _emit("job_started", phase=phase, job_id=job_id, index=index, total=total)
        authorized_job = resolve_authorized_metadata_calibration_job(
            manifest_path=manifest_path,
            outcome_decision_path=outcome_decision_path,
            authorization_envelope_path=envelope_path,
            trusted_signing_public_key_path=authority_public_key,
            job_id=job_id,
        )
        capability_path = (
            manifest_path.parent / "job-capabilities" / f"{job_id}.json"
        )
        unsigned_capability = build_metadata_calibration_job_capability(
            authorized_job,
            phase_manifest_sha256=str(manifest["phase_manifest_sha256"]),
            outcome_decision_signed_record_sha256=str(
                outcome_decision["signed_record_sha256"]
            ),
            authorization_envelope_sha256=str(
                envelope["authorization_envelope_sha256"]
            ),
            signing_key_id=str(outcome_decision["signing_key_id"]),
            authorization_basis=authorization_basis,
        )
        signed_capability = issue_signed_record(
            unsigned_capability,
            private_key_path=authority_private_key,
            private_key_password=authority_password,
            hash_field="capability_payload_sha256",
        )
        write_signed_decision_exclusive(capability_path, signed_capability)
        spool_root = spool_parent / f"{index:02d}-{uuid.uuid4().hex}"
        _ensure_private_directory(spool_root)
        spool_observed = spool_root.lstat()
        spool_identity = (spool_observed.st_dev, spool_observed.st_ino)
        command = _bubblewrap_worker_command(
            phase=phase,
            phase_root=phase_root,
            spool_root=spool_root,
            manifest_path=manifest_path,
            authority_public_key=authority_public_key,
            python_executable=python_executable,
            worker_code_snapshot=worker_code_snapshot,
            checkpoint_group=str(authorized_job.job_spec["checkpoint_group"]),
            runner_input_paths=runner_input_relative_paths(
                authorized_job.job_spec
            ),
            job_capability_path=capability_path,
            job_id=job_id,
        )
        completed = subprocess.run(
            command,
            cwd=_REPOSITORY,
            env={"PATH": "/usr/bin:/bin", "LANG": "C.UTF-8"},
            capture_output=True,
            stdin=subprocess.DEVNULL,
            text=True,
            check=False,
            close_fds=True,
        )
        log_path = log_root / f"{index:02d}-{job_id}.json"
        _write_exclusive_json(
            log_path,
            {
                "schema": "cfeg.metadata-calibration-worker-log.v1",
                "phase": phase,
                "job_id": job_id,
                "index": index,
                "total": total,
                "returncode": completed.returncode,
                "stdout": completed.stdout,
                "stderr": completed.stderr,
                "worker_isolation_contract_sha256": (
                    WORKER_ISOLATION_CONTRACT_SHA256
                ),
                "bubblewrap_command_sha256": hashlib.sha256(
                    json.dumps(command, separators=(",", ":")).encode("utf-8")
                ).hexdigest(),
            },
        )
        if completed.returncode != 0:
            raise RuntimeError(f"Worker {job_id} failed; inspect {log_path}.")
        worker_report = _parse_worker_report(
            completed.stdout,
            expected_job_id=job_id,
            expected_producer_kind=str(authorized_job.job_spec["producer_kind"]),
            expected_canonical_job_root=authorized_job.canonical_job_root,
        )
        relative_job_root = authorized_job.canonical_job_root.relative_to(
            authorized_job.execution_output_root
        )
        spool_job_root = spool_root / relative_job_root
        _validate_exact_worker_spool(
            spool_root,
            relative_job_root=relative_job_root,
            expected_files={Path(path).name for path in authorized_job.required_artifacts},
        )
        _require_directory_identity(spool_root, spool_identity)
        nested_observed = spool_job_root.lstat()
        nested_spool_identity = (nested_observed.st_dev, nested_observed.st_ino)
        validate_metadata_calibration_job_staging(
            authorized_job,
            spool_job_root,
            require_locked_modes=True,
        )
        promoted = publish_validated_metadata_calibration_job_spool(
            authorized_job,
            spool_job_root,
            expected_spool_identity=nested_spool_identity,
        )
        republished = validate_published_metadata_calibration_job(authorized_job)
        if (
            promoted.job_id != job_id
            or worker_report["producer_receipt_file_sha256"]
            != promoted.producer_receipt_file_sha256
            or republished != promoted
        ):
            raise PermissionError("Promoted job differs from its isolated worker report.")
        shutil.rmtree(spool_root)
        _fsync_directory(spool_parent)
        _emit("job_completed", phase=phase, job_id=job_id, index=index, total=total)
    if any(spool_parent.iterdir()):
        raise PermissionError("Worker spool parent is not empty after complete promotion.")
    spool_parent.rmdir()
    _fsync_directory(spool_parent.parent)


def _validate_runtime(python_executable: str | Path | None) -> Path:
    executable = Path(
        sys.executable if python_executable is None else python_executable
    ).absolute()
    expected = (_REPOSITORY / ".venv/bin/python").absolute()
    if executable != expected:
        raise RuntimeError(f"Lifecycle must use the repository CUDA venv Python: {expected}.")
    if Path(sys.prefix).absolute() != (_REPOSITORY / ".venv").absolute() or not os.path.samefile(
        executable, expected
    ):
        raise RuntimeError("Lifecycle is not running inside the exact repository CUDA venv.")
    if (
        torch.__version__ != "2.2.2+cu121"
        or not torch.cuda.is_available()
        or torch.cuda.device_count() != 1
        or torch.cuda.get_device_name(0) != "NVIDIA GeForce RTX 4090"
    ):
        raise RuntimeError("Lifecycle runtime differs from the frozen single-RTX-4090 contract.")
    return executable


def _prepare_worker_code_snapshot(path: Path, *, source_tag: str) -> Path:
    """Clone only the frozen tag so ignored data and live edits are never mounted."""

    target = _absolute_no_symlink_path(path, must_exist=False)
    if os.path.lexists(target):
        raise FileExistsError("Worker code snapshot path must be completely new.")
    expected_commit = subprocess.run(
        ["/usr/bin/git", "rev-parse", "--verify", f"refs/tags/{source_tag}^{{commit}}"],
        cwd=_REPOSITORY,
        check=True,
        capture_output=True,
        text=True,
        env={"PATH": "/usr/bin:/bin", "LANG": "C.UTF-8"},
    ).stdout.strip()
    clone = subprocess.run(
        [
            "/usr/bin/git",
            "clone",
            "--no-local",
            "--depth",
            "1",
            "--single-branch",
            "--branch",
            source_tag,
            f"file://{_REPOSITORY}",
            str(target),
        ],
        check=False,
        capture_output=True,
        text=True,
        env={"PATH": "/usr/bin:/bin", "LANG": "C.UTF-8"},
    )
    if clone.returncode != 0:
        raise RuntimeError(f"Could not create frozen worker code snapshot: {clone.stderr}")
    observed_commit = subprocess.run(
        ["/usr/bin/git", "-C", str(target), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
        env={"PATH": "/usr/bin:/bin", "LANG": "C.UTF-8"},
    ).stdout.strip()
    commit_count = subprocess.run(
        ["/usr/bin/git", "-C", str(target), "rev-list", "--all", "--count"],
        check=True,
        capture_output=True,
        text=True,
        env={"PATH": "/usr/bin:/bin", "LANG": "C.UTF-8"},
    ).stdout.strip()
    if observed_commit != expected_commit or commit_count != "1":
        raise PermissionError("Worker code snapshot differs from the frozen single commit.")
    tracked_sensitive = subprocess.run(
        [
            "/usr/bin/git",
            "-C",
            str(target),
            "ls-files",
            "checkpoints",
            "data",
            "outputs",
            ".local",
        ],
        check=True,
        capture_output=True,
        text=True,
        env={"PATH": "/usr/bin:/bin", "LANG": "C.UTF-8"},
    ).stdout.splitlines()
    if set(tracked_sensitive) != {
        "checkpoints/.gitkeep",
        "data/README.md",
        "data/manifests/.gitkeep",
        "data/processed/.gitkeep",
        "data/raw/.gitkeep",
        "outputs/.gitkeep",
    }:
        raise PermissionError("Frozen source tag contains unexpected data or result artifacts.")
    (target / ".venv").mkdir(mode=0o500)
    for child in sorted(target.rglob("*"), reverse=True):
        observed = child.lstat()
        if stat.S_ISLNK(observed.st_mode):
            raise PermissionError("Worker code snapshot contains a symbolic link.")
        os.chmod(child, 0o500 if stat.S_ISDIR(observed.st_mode) else 0o400)
    os.chmod(target, 0o500)
    _fsync_directory(target.parent)
    return target


def _harden_label_capable_parent_process() -> None:
    """Prevent worker-side memory inspection before any secret is created."""

    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    libc = ctypes.CDLL(None, use_errno=True)
    prctl = getattr(libc, "prctl", None)
    if prctl is None:
        raise RuntimeError("The label-capable lifecycle requires Linux prctl.")
    prctl.restype = ctypes.c_int
    if prctl(4, 0, 0, 0, 0) != 0:  # PR_SET_DUMPABLE
        error = ctypes.get_errno()
        raise OSError(error, os.strerror(error), "prctl(PR_SET_DUMPABLE)")
    if prctl(3, 0, 0, 0, 0) != 0:  # PR_GET_DUMPABLE must now return zero.
        raise RuntimeError("The label-capable lifecycle process remains dumpable.")
    soft, hard = resource.getrlimit(resource.RLIMIT_CORE)
    if (soft, hard) != (0, 0):
        raise RuntimeError("The label-capable lifecycle retains a core-dump allowance.")


def _bubblewrap_worker_command(
    *,
    phase: ExperimentPhase,
    phase_root: Path,
    spool_root: Path,
    manifest_path: Path,
    authority_public_key: Path,
    python_executable: Path,
    worker_code_snapshot: Path,
    checkpoint_group: str,
    runner_input_paths: tuple[str, ...],
    job_capability_path: Path,
    job_id: str,
) -> list[str]:
    bubblewrap = Path("/usr/bin/bwrap")
    if not bubblewrap.is_file() or not os.access(bubblewrap, os.X_OK):
        raise RuntimeError("The frozen worker boundary requires executable /usr/bin/bwrap.")
    manifest = read_json_object(manifest_path)
    execution_root = Path(str(manifest["execution_output_root"])).absolute()
    input_root = Path(str(manifest["input_seal_root"])).absolute()
    if execution_root != phase_root / "execution" or input_root != phase_root / "input-seal":
        raise PermissionError("Manifest paths differ from the lifecycle sandbox layout.")
    command = [
        str(bubblewrap),
        "--unshare-all",
        "--die-with-parent",
        "--new-session",
        "--clearenv",
        "--cap-drop",
        "ALL",
    ]
    environment = {
        "PATH": "/usr/bin:/bin",
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONNOUSERSITE": "1",
        "HOME": "/tmp/home",
        "TMPDIR": "/tmp",
        "PYTHONPATH": str(_REPOSITORY / "src"),
        "CUDA_VISIBLE_DEVICES": "0",
        "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
        "PYTHONHASHSEED": "0",
        "OMP_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "NUMEXPR_NUM_THREADS": "1",
        "LANG": "C.UTF-8",
        "CFEG_WORKER_ISOLATION_CONTRACT_SHA256": (
            WORKER_ISOLATION_CONTRACT_SHA256
        ),
        "CFEG_LIFECYCLE_HOST_PID": str(os.getpid()),
    }
    for name, value in environment.items():
        command.extend(("--setenv", name, value))
    for source in ("/usr", "/bin", "/lib", "/lib64", "/etc", "/sys"):
        command.extend(("--ro-bind", source, source))
    command.extend(("--dev", "/dev"))
    for device in (
        "/dev/nvidia0",
        "/dev/nvidiactl",
        "/dev/nvidia-uvm",
        "/dev/nvidia-uvm-tools",
    ):
        observed = Path(device).lstat()
        if not stat.S_ISCHR(observed.st_mode):
            raise RuntimeError(f"Frozen CUDA device is unavailable: {device}.")
        command.extend(("--dev-bind", device, device))
    command.extend(("--proc", "/proc", "--tmpfs", "/tmp", "--dir", "/tmp/home"))
    for directory in _sandbox_directory_chain(phase_root, authority_public_key.parent):
        command.extend(("--dir", str(directory)))
    command.extend(
        (
            "--tmpfs",
            str(manifest_path.parent),
            "--dir",
            str(job_capability_path.parent),
            "--tmpfs",
            str(authority_public_key.parent),
        )
    )
    command.extend(
        (
            "--ro-bind",
            str(worker_code_snapshot),
            str(_REPOSITORY),
            "--ro-bind",
            str(_REPOSITORY / ".venv"),
            str(_REPOSITORY / ".venv"),
            "--ro-bind",
            str(job_capability_path),
            str(job_capability_path),
            "--tmpfs",
            str(input_root),
            "--ro-bind",
            str(input_root / "input_seal_decision.json"),
            str(input_root / "input_seal_decision.json"),
            "--ro-bind",
            str(input_root / "input_seal_receipt.json"),
            str(input_root / "input_seal_receipt.json"),
            "--ro-bind",
            str(authority_public_key),
            str(authority_public_key),
        )
    )
    group_root = input_root / checkpoint_group
    command.extend(("--tmpfs", str(group_root)))
    projection_directories = sorted(
        {
            Path(relative).parent.as_posix()
            for relative in runner_input_paths
            if Path(relative).parent != Path(".")
        }
    )
    for directory in projection_directories:
        command.extend(("--dir", str(group_root / directory)))
    for relative in runner_input_paths:
        artifact = group_root / relative
        command.extend(("--ro-bind", str(artifact), str(artifact)))
    command.extend(
        (
            "--remount-ro",
            str(group_root),
            "--remount-ro",
            str(input_root),
            "--remount-ro",
            str(manifest_path.parent),
            "--remount-ro",
            str(authority_public_key.parent),
        )
    )
    command.extend(
        (
            "--bind",
            str(spool_root),
            str(execution_root),
            "--chdir",
            str(_REPOSITORY),
            "--",
            str(python_executable),
            str(_REPOSITORY / "scripts/run_metadata_calibration_worker.py"),
            "--trusted-signing-public-key",
            str(authority_public_key),
            "--job-capability",
            str(job_capability_path),
            "--checkpoint-group",
            checkpoint_group,
            "--job-id",
            job_id,
        )
    )
    return command


def _sandbox_directory_chain(phase_root: Path, key_root: Path) -> tuple[Path, ...]:
    base = Path("/home/whwovy")
    directories = {Path("/home"), base}
    for leaf in (phase_root, key_root):
        if not leaf.is_relative_to(base):
            raise ValueError("Sandbox research paths must remain beneath /home/whwovy.")
        current = base
        for part in leaf.relative_to(base).parts:
            current /= part
            directories.add(current)
    return tuple(sorted(directories, key=lambda path: (len(path.parts), str(path))))


def _parse_worker_report(
    stdout: str,
    *,
    expected_job_id: str,
    expected_producer_kind: str,
    expected_canonical_job_root: Path,
) -> dict[str, Any]:
    lines = [line for line in stdout.splitlines() if line.strip()]
    if not lines:
        raise PermissionError("Isolated worker emitted no completion report.")
    try:
        report = json.loads(lines[-1])
    except json.JSONDecodeError as error:
        raise PermissionError("Isolated worker completion report is not JSON.") from error
    report_fields = {
        "job_id",
        "producer_kind",
        "canonical_job_root",
        "producer_receipt_file_sha256",
        "isolation_attestation",
    }
    isolation_fields = {
        "schema",
        "isolation_contract_sha256",
        "processed_asset_root_absent",
        "private_keys_absent",
        "input_and_code_read_only",
        "prior_same_phase_outputs_and_ignored_data_absent",
        "other_checkpoint_groups_absent",
        "finalizer_only_ciphertexts_absent",
        "only_current_job_capability_visible",
        "source_prerequisite_phase_access_policy_exact",
        "fresh_execution_spool_mounted",
        "execution_spool_empty_on_entry",
        "no_new_privileges",
        "linux_capabilities_all_zero",
        "host_pid_namespace_absent",
        "secret_file_descriptors_absent",
    }
    isolation = report.get("isolation_attestation") if isinstance(report, dict) else None
    if (
        not isinstance(report, dict)
        or set(report) != report_fields
        or report.get("job_id") != expected_job_id
        or report.get("producer_kind") != expected_producer_kind
        or Path(str(report.get("canonical_job_root"))).absolute()
        != expected_canonical_job_root
        or not _is_sha256(report.get("producer_receipt_file_sha256"))
        or not isinstance(isolation, dict)
        or set(isolation) != isolation_fields
        or isolation.get("schema")
        != "cfeg.metadata-calibration-worker-isolation-attestation.v1"
        or isolation.get("isolation_contract_sha256")
        != WORKER_ISOLATION_CONTRACT_SHA256
        or any(
            isolation.get(field) is not True
            for field in isolation_fields - {"schema", "isolation_contract_sha256"}
        )
    ):
        raise PermissionError("Isolated worker completion attestation is invalid.")
    return report


def _validate_exact_worker_spool(
    spool_root: Path,
    *,
    relative_job_root: Path,
    expected_files: set[str],
) -> None:
    expected_directories: set[Path] = set()
    current = spool_root
    for part in relative_job_root.parts:
        current /= part
        expected_directories.add(current)
    expected_paths = {current / name for name in expected_files}
    actual_directories: set[Path] = set()
    actual_files: set[Path] = set()
    for path in spool_root.rglob("*"):
        observed = path.lstat()
        if stat.S_ISDIR(observed.st_mode) and not stat.S_ISLNK(observed.st_mode):
            if stat.S_IMODE(observed.st_mode) != 0o700:
                raise PermissionError("Worker spool directory is not mode 0700.")
            actual_directories.add(path)
        elif stat.S_ISREG(observed.st_mode) and observed.st_nlink == 1:
            if stat.S_IMODE(observed.st_mode) != 0o400:
                raise PermissionError("Worker spool artifact is not mode 0400.")
            actual_files.add(path)
        else:
            raise PermissionError("Worker spool contains a link or special file.")
    if actual_directories != expected_directories or actual_files != expected_paths:
        raise PermissionError("Worker spool contains missing or undeclared paths.")


def _require_directory_identity(path: Path, expected: tuple[int, int]) -> None:
    observed = path.lstat()
    if (
        not stat.S_ISDIR(observed.st_mode)
        or stat.S_ISLNK(observed.st_mode)
        or (observed.st_dev, observed.st_ino) != expected
    ):
        raise PermissionError("Worker spool root identity changed during execution.")


def _new_private_run_root(path: Path) -> Path:
    root = _absolute_no_symlink_path(path, must_exist=False)
    if os.path.lexists(root):
        raise FileExistsError("Lifecycle run root must be completely new.")
    _ensure_private_directory(root.parent)
    root.mkdir(mode=0o700)
    _fsync_directory(root)
    _fsync_directory(root.parent)
    return root


def _validate_capability_path_disjointness(
    *,
    repository: Path,
    processed_asset_root: Path,
    run_root: Path,
    outcome_access_registry_root: Path,
) -> None:
    registry = outcome_access_registry_root
    named_paths = (
        ("repository", repository),
        ("processed asset", processed_asset_root),
        ("lifecycle run root", run_root),
        ("outcome-access registry", registry),
        (
            "outcome-access registry locks",
            _absolute_no_symlink_path(
                registry.with_name(f"{registry.name}-locks"), must_exist=False
            ),
        ),
        (
            "outcome-access registry staging",
            _absolute_no_symlink_path(
                registry.with_name(f"{registry.name}-staging"), must_exist=False
            ),
        ),
    )
    for index, (left_name, left) in enumerate(named_paths):
        for right_name, right in named_paths[index + 1 :]:
            label = f"{left_name} and {right_name}"
            if left == right or left.is_relative_to(right) or right.is_relative_to(left):
                raise ValueError(f"Capability-bearing paths must be disjoint: {label}.")


def _ensure_private_directory(path: Path) -> None:
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
            raise ValueError("Lifecycle output hierarchy contains a non-directory component.")
        os.chmod(target, 0o700)
        _fsync_directory(target)
        _fsync_directory(target.parent)
    observed = path.lstat()
    if not stat.S_ISDIR(observed.st_mode) or stat.S_ISLNK(observed.st_mode):
        raise ValueError("Lifecycle output parent must be one real directory.")


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
            raise ValueError(f"Lifecycle path contains a symlink component: {component}.")
    if must_exist and not absolute.exists():
        raise FileNotFoundError(absolute)
    return absolute


def _write_exclusive_json(path: Path, value: dict[str, Any]) -> None:
    write_signed_decision_exclusive(path, value)


def _write_failure_receipt(
    root: Path,
    *,
    error: BaseException,
    decision_date: str,
    authorization_basis: str,
    authority_key_id: str,
    authority_private_key: Path,
    authority_password: bytes,
    current_run_outcome_access_started: bool,
    failed_phase: ExperimentPhase,
    current_run_claimed_phases: tuple[str, ...],
    registry_integrity_failure: bool,
    blocked_by_prior_claim: bool,
    blocking_claim_path: Path | None,
    blocking_scope_sha256: str | None,
) -> None:
    path = root / "lifecycle_failure.json"
    if os.path.lexists(path):
        return
    claims_by_phase = {
        phase: phase in current_run_claimed_phases
        for phase in ("source_development", "held_participant_evaluation")
    }
    failed_phase_claimed = claims_by_phase[failed_phase]
    status, recovery_disposition, retry_permitted, redact_detail = (
        _derive_failure_policy(
            failed_phase=failed_phase,
            current_run_outcome_access_started=current_run_outcome_access_started,
            registry_integrity_failure=registry_integrity_failure,
            blocked_by_prior_claim=blocked_by_prior_claim,
        )
    )
    record = {
        "schema": "cfeg.metadata-calibration-lifecycle-failure.v1",
        "candidate_id": "metadata-calibration-efficiency-v1",
        "status": status,
        "current_run_outcome_access_started": current_run_outcome_access_started,
        "failed_phase": failed_phase,
        "claims_by_phase": claims_by_phase,
        "failed_phase_claimed": failed_phase_claimed,
        "recovery_disposition": recovery_disposition,
        "blocked_by_prior_outcome_claim": blocked_by_prior_claim,
        "outcome_registry_integrity_failure": registry_integrity_failure,
        "same_cohort_automatic_retry_permitted": retry_permitted,
        "blocking_claim_path": (
            None if blocking_claim_path is None else str(blocking_claim_path)
        ),
        "blocking_scope_sha256": blocking_scope_sha256,
        "blocking_claim_file_sha256": _optional_regular_file_sha256(
            blocking_claim_path
        ),
        "run_root": str(root),
        "error_type": type(error).__name__,
        "error_message": (
            None if redact_detail else str(error)
        ),
        "error_detail_sha256": (
            None
            if redact_detail
            else hashlib.sha256(str(error).encode("utf-8")).hexdigest()
        ),
        "post_outcome_error_detail_redacted": redact_detail,
        "decision_date": decision_date,
        "signing_key_id": authority_key_id,
        "signer_role": "delegated_automation_authority",
        "authorization_basis": authorization_basis,
        "signature_algorithm": "ed25519",
    }
    signed = issue_signed_record(
        record,
        private_key_path=authority_private_key,
        private_key_password=authority_password,
        hash_field="lifecycle_payload_sha256",
    )
    write_signed_decision_exclusive(path, signed)
    validate_metadata_calibration_lifecycle_failure(path)


def _derive_failure_policy(
    *,
    failed_phase: ExperimentPhase,
    current_run_outcome_access_started: bool,
    registry_integrity_failure: bool,
    blocked_by_prior_claim: bool,
) -> tuple[str, str, bool, bool]:
    if registry_integrity_failure:
        return (
            "outcome_registry_integrity_failure_manual_audit_required",
            "manual_audit_registry_integrity_failure_no_retry",
            False,
            True,
        )
    if blocked_by_prior_claim:
        return (
            "blocked_by_prior_outcome_claim_no_same_cohort_retry",
            "manual_audit_prior_claim_blocks_run",
            False,
            True,
        )
    if current_run_outcome_access_started:
        return (
            "manual_audit_required_no_new_same_cohort_run",
            "terminal_fail_stop_after_any_current_run_claim_no_automatic_retry",
            False,
            True,
        )
    if failed_phase == "source_development":
        return (
            "failed_before_outcome_access_new_exact_run_permitted",
            "fresh_full_exact_run_permitted",
            True,
            False,
        )
    return (
        "inconsistent_held_failure_without_source_claim_manual_audit_required",
        "manual_audit_inconsistent_held_failure_without_source_claim",
        False,
        False,
    )


def validate_metadata_calibration_lifecycle_failure(
    receipt_path: str | Path,
) -> str:
    """Authenticate one exact terminal or preclaim lifecycle failure record."""

    path = Path(receipt_path).absolute()
    root = path.parent
    if path != root / "lifecycle_failure.json":
        raise PermissionError("Lifecycle failure path is noncanonical.")
    observed = path.lstat()
    if (
        not stat.S_ISREG(observed.st_mode)
        or stat.S_ISLNK(observed.st_mode)
        or observed.st_nlink != 1
        or stat.S_IMODE(observed.st_mode) != 0o400
    ):
        raise PermissionError("Lifecycle failure is not one immutable regular file.")
    record = read_json_object(path)
    signed_hash = verify_signed_record(
        record,
        public_key_path=root / "keys" / "authority-signing-public.pem",
        hash_field="lifecycle_payload_sha256",
        expected_schema="cfeg.metadata-calibration-lifecycle-failure.v1",
    )
    claims = record.get("claims_by_phase")
    failed_phase = record.get("failed_phase")
    if (
        set(record) != _LIFECYCLE_FAILURE_FIELDS
        or record.get("candidate_id") != "metadata-calibration-efficiency-v1"
        or record.get("run_root") != str(root)
        or failed_phase not in {
            "source_development",
            "held_participant_evaluation",
        }
        or not isinstance(claims, Mapping)
        or set(claims) != {
            "source_development",
            "held_participant_evaluation",
        }
        or any(type(value) is not bool for value in claims.values())
        or type(record.get("current_run_outcome_access_started")) is not bool
        or type(record.get("failed_phase_claimed")) is not bool
        or type(record.get("blocked_by_prior_outcome_claim")) is not bool
        or type(record.get("outcome_registry_integrity_failure")) is not bool
        or type(record.get("same_cohort_automatic_retry_permitted")) is not bool
        or type(record.get("post_outcome_error_detail_redacted")) is not bool
        or not isinstance(record.get("error_type"), str)
        or not record["error_type"]
        or not isinstance(record.get("decision_date"), str)
        or not record["decision_date"]
        or record.get("signer_role") != "delegated_automation_authority"
        or record.get("signature_algorithm") != "ed25519"
        or not isinstance(record.get("authorization_basis"), str)
        or not record["authorization_basis"]
        or not _is_sha256(record.get("signing_key_id"))
    ):
        raise PermissionError("Lifecycle failure has a wrong exact schema.")
    current_claim = any(claims.values())
    blocked = bool(record["blocked_by_prior_outcome_claim"])
    registry_failure = bool(record["outcome_registry_integrity_failure"])
    status, recovery, retry, redact = _derive_failure_policy(
        failed_phase=failed_phase,
        current_run_outcome_access_started=current_claim,
        registry_integrity_failure=registry_failure,
        blocked_by_prior_claim=blocked,
    )
    if (
        record.get("current_run_outcome_access_started") is not current_claim
        or record.get("failed_phase_claimed") is not claims[failed_phase]
        or record.get("status") != status
        or record.get("recovery_disposition") != recovery
        or record.get("same_cohort_automatic_retry_permitted") is not retry
        or record.get("post_outcome_error_detail_redacted") is not redact
    ):
        raise PermissionError("Lifecycle failure policy fields are inconsistent.")
    blocking_path = record.get("blocking_claim_path")
    if blocked:
        if (
            not isinstance(blocking_path, str)
            or not Path(blocking_path).is_absolute()
            or Path(blocking_path).parent
            != CANONICAL_OUTCOME_ACCESS_REGISTRY_ROOT.absolute()
            or not _is_sha256(record.get("blocking_scope_sha256"))
            or not _is_sha256(record.get("blocking_claim_file_sha256"))
            or record.get("blocking_claim_file_sha256")
            != _optional_regular_file_sha256(Path(blocking_path))
        ):
            raise PermissionError("Lifecycle failure lacks its blocking claim evidence.")
    elif any(
        record.get(field) is not None
        for field in (
            "blocking_claim_path",
            "blocking_scope_sha256",
            "blocking_claim_file_sha256",
        )
    ):
        raise PermissionError("Lifecycle failure carries undeclared blocking evidence.")
    message = record.get("error_message")
    detail_hash = record.get("error_detail_sha256")
    if redact:
        if message is not None or detail_hash is not None:
            raise PermissionError("Sensitive lifecycle failure detail was not redacted.")
    elif (
        not isinstance(message, str)
        or detail_hash != hashlib.sha256(message.encode("utf-8")).hexdigest()
    ):
        raise PermissionError("Preclaim lifecycle failure detail is invalid.")
    return signed_hash


def _destroy_generated_private_keys(paths: tuple[Path, ...]) -> bool:
    destroyed = True
    for path in paths:
        try:
            observed = path.lstat()
        except FileNotFoundError:
            continue
        if not stat.S_ISREG(observed.st_mode) or observed.st_nlink != 1:
            destroyed = False
            continue
        try:
            path.unlink()
            _fsync_directory(path.parent)
        except OSError:
            destroyed = False
    return destroyed and all(not os.path.lexists(path) for path in paths)


def _publish_signed_pending_noreplace(
    pending: Path,
    canonical: Path,
    *,
    private_key_paths: tuple[Path, ...],
) -> None:
    """Publish an already-fsynced signed completion only after key destruction."""

    observed = pending.lstat()
    if (
        not stat.S_ISREG(observed.st_mode)
        or stat.S_ISLNK(observed.st_mode)
        or observed.st_nlink != 1
        or stat.S_IMODE(observed.st_mode) != 0o400
    ):
        raise PermissionError("Signed lifecycle pending record is not immutable.")
    if any(os.path.lexists(path) for path in private_key_paths):
        raise PermissionError("Lifecycle completion cannot publish while a private key exists.")
    if pending.parent.lstat().st_dev != canonical.parent.lstat().st_dev:
        raise OSError(errno.EXDEV, "Lifecycle publication requires one filesystem.")
    if os.path.lexists(canonical):
        raise FileExistsError("Lifecycle completion never replaces a canonical record.")
    libc = ctypes.CDLL(None, use_errno=True)
    renameat2 = getattr(libc, "renameat2", None)
    if renameat2 is None:
        raise RuntimeError("Lifecycle publication requires Linux renameat2.")
    renameat2.argtypes = [
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_uint,
    ]
    renameat2.restype = ctypes.c_int
    result = renameat2(
        -100,
        os.fsencode(pending),
        -100,
        os.fsencode(canonical),
        1,
    )
    if result != 0:
        error = ctypes.get_errno()
        if error == errno.EEXIST:
            raise FileExistsError("Lifecycle completion destination appeared.")
        raise OSError(error, os.strerror(error), str(canonical))
    _fsync_directory(canonical.parent)


def _write_completion_publication_failure_marker(
    root: Path,
    *,
    pending_path: Path,
    canonical_path: Path,
    error: BaseException,
) -> None:
    marker = root / "lifecycle_completion_publication_failure.json"
    if os.path.lexists(marker):
        return
    _write_exclusive_json(
        marker,
        {
            "schema": "cfeg.metadata-calibration-completion-publication-failure.v1",
            "status": "signed_pending_retained_manual_recovery_required",
            "run_root": str(root),
            "signed_pending_path": str(pending_path),
            "signed_pending_file_sha256": _optional_regular_file_sha256(
                pending_path
            ),
            "canonical_completion_path": str(canonical_path),
            "generated_private_keys_absent": True,
            "error_type": type(error).__name__,
        },
    )


def _optional_regular_file_sha256(path: Path | None) -> str | None:
    if path is None:
        return None
    try:
        observed = path.lstat()
    except FileNotFoundError:
        return None
    if not stat.S_ISREG(observed.st_mode) or stat.S_ISLNK(observed.st_mode):
        return None
    return _sha256_file(path)


def _write_key_destruction_failure(root: Path, paths: tuple[Path, ...]) -> None:
    remaining = [str(path) for path in paths if os.path.lexists(path)]
    _write_exclusive_json(
        root / "private_key_destruction_failure.json",
        {
            "schema": "cfeg.metadata-calibration-key-destruction-failure.v1",
            "status": "completion_not_published_manual_security_audit_required",
            "run_root": str(root),
            "remaining_private_key_paths": remaining,
            "generated_private_keys_destroyed": False,
        },
    )


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(
        path,
        os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0),
    )
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _is_sha256(value: object) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(
        character in "0123456789abcdef" for character in value
    )


def _emit(event: str, **payload: Any) -> None:
    print(
        json.dumps({"event": event, **payload}, sort_keys=True, separators=(",", ":")),
        flush=True,
    )
