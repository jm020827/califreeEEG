from __future__ import annotations

import ctypes
import errno
import fcntl
import hashlib
import os
import stat
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from cfeg.metadata_calibration_authority import (
    canonical_json_bytes,
    canonical_json_sha256,
    issue_signed_record,
    read_json_object,
    verify_signed_record,
)

OUTCOME_ACCESS_CLAIM_SCHEMA = "cfeg.metadata-calibration-outcome-access-claim.v1"
OUTCOME_ACCESS_SCOPE_SCHEMA = "cfeg.metadata-calibration-outcome-access-scope.v1"
OUTCOME_ACCESS_ATTEMPT_KEY_SCHEMA = (
    "cfeg.metadata-calibration-outcome-access-attempt-key.permanent-v1"
)
CANONICAL_OUTCOME_ACCESS_REGISTRY_ROOT = Path(
    "/home/whwovy/eeg-results/cfeg-wearable-v3-outcome-access-registry-v1"
)
_CLAIM_FIELDS = {
    "schema",
    "candidate_id",
    "phase",
    "status",
    "outcome_access_scope",
    "outcome_access_scope_sha256",
    "outcome_access_attempt_key_sha256",
    "claim_path",
    "run_root",
    "phase_manifest_sha256",
    "outcome_decision_signed_record_sha256",
    "execution_contract_sha256",
    "filesystem_private_seal_file_sha256",
    "filesystem_private_seal_signed_record_sha256",
    "filesystem_private_tree_sha256",
    "source_commit",
    "source_tree_sha256",
    "source_tag",
    "signing_key_id",
    "signer_role",
    "authorization_basis",
    "signature_algorithm",
    "claim_payload_sha256",
    "signature",
    "signed_record_sha256",
}
_PERMANENT_IDENTITY_FIELDS = {
    "phase",
    "primary_dataset_id",
    "conditions",
    "query_blocks",
    "target_subject_ids",
    "canonical_target_subject_ids_sha256",
    "query_identity_bundle_sha256",
    "trial_inclusion_rule",
    "outcome_field",
    "outcome_analysis_unit",
}
_SCOPE_FIELDS = {
    "schema",
    "candidate_id",
    "permanent_attempt_key",
    "permanent_attempt_key_sha256",
    *_PERMANENT_IDENTITY_FIELDS,
    "allocation_decision_id",
    "n_classes",
    "outcome_scope",
}
_PHASE_PERMANENT_EXPECTATIONS = {
    "source_development": {
        "subject_count": 39,
        "target_subject_ids_sha256": (
            "02f673ab1efa5cd35665ea0ed41f2ab70cc290cc28cdae5fcce942d0c6c4f75a"
        ),
        "query_identity_bundle_sha256": (
            "224d8239aef5d0f9dd04c388f8ed8d2f961fc150c1cf72741f5ab22b756f80de"
        ),
    },
    "held_participant_evaluation": {
        "subject_count": 60,
        "target_subject_ids_sha256": (
            "52c1f2813bf98ab06fc1f89f9fbd70c93e0c7075e1baa780cb0e66689d5f3d14"
        ),
        "query_identity_bundle_sha256": (
            "feffdcc005f2f4818bb496c61b683dc7afd735166c2f50cf7892262ef55d2a16"
        ),
    },
}


@dataclass(frozen=True)
class OutcomeAccessClaimBinding:
    claim_path: Path
    claim_file_sha256: str
    signed_record_sha256: str
    outcome_access_scope_sha256: str
    run_root: Path


class OutcomeAccessAlreadyClaimed(PermissionError):
    """Raised when a stable cohort outcome slot was consumed by any prior run."""

    def __init__(self, claim_path: Path, outcome_access_scope_sha256: str) -> None:
        self.claim_path = claim_path
        self.outcome_access_scope_sha256 = outcome_access_scope_sha256
        super().__init__(
            "Outcome access was already claimed for this phase and cohort; "
            f"automatic same-cohort retry is forbidden: {claim_path}."
        )


@dataclass(frozen=True)
class OutcomeAccessRegistryAudit:
    current_run_claim_found: bool
    registry_integrity_failure: bool
    current_run_claimed_phases: tuple[str, ...] = ()


class OutcomeAccessRegistryIntegrityError(PermissionError):
    """Raised before data access when any permanent registry entry is invalid."""


@contextmanager
def hold_outcome_access_scope_lease(
    manifest: Mapping[str, Any],
    *,
    registry_root: str | Path = CANONICAL_OUTCOME_ACCESS_REGISTRY_ROOT,
) -> Iterator[None]:
    """Serialize same-cohort attempts while preserving retry before a durable claim."""

    registry = Path(registry_root).absolute()
    lock_root = registry.with_name(f"{registry.name}-locks")
    lock_root.mkdir(parents=True, mode=0o700, exist_ok=True)
    root_stat = lock_root.lstat()
    if (
        not stat.S_ISDIR(root_stat.st_mode)
        or stat.S_ISLNK(root_stat.st_mode)
        or stat.S_IMODE(root_stat.st_mode) != 0o700
    ):
        raise PermissionError("Outcome-access lease root is not one private directory.")
    attempt_key_sha256 = canonical_json_sha256(
        build_outcome_access_attempt_key(manifest)
    )
    phase = str(manifest["phase"])
    lock_path = lock_root / f"{phase}-{attempt_key_sha256}.lock"
    descriptor = os.open(
        lock_path,
        os.O_RDWR | os.O_CREAT | getattr(os, "O_NOFOLLOW", 0),
        0o600,
    )
    try:
        os.fchmod(descriptor, 0o600)
        observed = os.fstat(descriptor)
        if (
            not stat.S_ISREG(observed.st_mode)
            or observed.st_nlink != 1
            or stat.S_IMODE(observed.st_mode) != 0o600
        ):
            raise PermissionError("Outcome-access lease is not one private regular file.")
        fcntl.flock(descriptor, fcntl.LOCK_EX)
        path_observed = lock_path.lstat()
        fd_observed = os.fstat(descriptor)
        if (
            stat.S_ISLNK(path_observed.st_mode)
            or not stat.S_ISREG(path_observed.st_mode)
            or path_observed.st_nlink != 1
            or stat.S_IMODE(path_observed.st_mode) != 0o600
            or (path_observed.st_dev, path_observed.st_ino)
            != (fd_observed.st_dev, fd_observed.st_ino)
        ):
            raise PermissionError("Outcome-access lease path changed while acquiring its lock.")
        refuse_prior_outcome_access_claim(manifest, registry_root=registry)
        yield
    finally:
        try:
            fcntl.flock(descriptor, fcntl.LOCK_UN)
        finally:
            os.close(descriptor)


def build_outcome_access_attempt_key(manifest: Mapping[str, Any]) -> dict[str, Any]:
    """Build the permanent cohort key; this namespace must never be version-bumped."""

    scope = _build_outcome_access_identity(manifest)
    key = {
        "schema": OUTCOME_ACCESS_ATTEMPT_KEY_SCHEMA,
        **scope,
    }
    _validate_permanent_attempt_key(key)
    return key


def build_outcome_access_scope(manifest: Mapping[str, Any]) -> dict[str, Any]:
    """Build the stable candidate/cohort identity that survives run-root changes."""

    identity = _build_outcome_access_identity(manifest)
    attempt_key = {
        "schema": OUTCOME_ACCESS_ATTEMPT_KEY_SCHEMA,
        **identity,
    }
    return {
        "schema": OUTCOME_ACCESS_SCOPE_SCHEMA,
        "candidate_id": "metadata-calibration-efficiency-v1",
        "permanent_attempt_key": attempt_key,
        "permanent_attempt_key_sha256": canonical_json_sha256(attempt_key),
        **identity,
        "allocation_decision_id": "DEC-20260830-002",
        "n_classes": 12,
        "outcome_scope": "paired_dry_wet_12_class_query_labels_only",
    }


def _build_outcome_access_identity(manifest: Mapping[str, Any]) -> dict[str, Any]:
    """Validate and return the immutable scientific identity shared by both records."""

    contract = manifest.get("execution_contract")
    if not isinstance(contract, Mapping):
        raise TypeError("Outcome-access scope requires an execution contract.")
    phase_spec = contract.get("phase_spec")
    if not isinstance(phase_spec, Mapping):
        raise TypeError("Outcome-access scope requires a phase specification.")
    phase = str(manifest.get("phase"))
    if phase not in {"source_development", "held_participant_evaluation"}:
        raise ValueError("Outcome-access scope has an invalid phase.")
    target_groups = phase_spec.get("target_subject_ids_by_checkpoint_group")
    if not isinstance(target_groups, Mapping):
        raise TypeError("Outcome-access scope requires target participant groups.")
    if any(not isinstance(subjects, list) for subjects in target_groups.values()):
        raise ValueError("Outcome-access target participant groups must be lists.")
    flattened_subject_ids = [
        str(subject) for subjects in target_groups.values() for subject in subjects
    ]
    target_subject_ids = sorted(
        set(flattened_subject_ids)
    )
    expected_group_names = (
        {"fold0", "fold1", "fold2"}
        if phase == "source_development"
        else {"held"}
    )
    expected_subject_count = 39 if phase == "source_development" else 60
    identity = {
        "phase": phase,
        "primary_dataset_id": "wearable_v3_paired_dry_wet",
        "conditions": phase_spec.get("conditions"),
        "query_blocks": ["block06", "block07", "block08", "block09", "block10"],
        "target_subject_ids": target_subject_ids,
        "canonical_target_subject_ids_sha256": canonical_json_sha256(
            {"subject_ids": target_subject_ids}
        ),
        "query_identity_bundle_sha256": phase_spec.get(
            "query_identity_bundle_sha256"
        ),
        "trial_inclusion_rule": "five_query_blocks_each_12_class_complete",
        "outcome_field": "query_label",
        "outcome_analysis_unit": "participant_equal_interface_balanced_accuracy",
    }
    if (
        manifest.get("candidate_id") != "metadata-calibration-efficiency-v1"
        or contract.get("candidate_id") != "metadata-calibration-efficiency-v1"
        or contract.get("phase") != phase
        or phase_spec.get("n_classes") != 12
        or identity["conditions"] != ["dry", "wet"]
        or set(map(str, target_groups)) != expected_group_names
        or len(flattened_subject_ids) != expected_subject_count
        or len(target_subject_ids) != expected_subject_count
    ):
        raise ValueError("Outcome-access scope differs from the frozen cohort contract.")
    _validate_permanent_attempt_key(
        {"schema": OUTCOME_ACCESS_ATTEMPT_KEY_SCHEMA, **identity}
    )
    return identity


def _validate_permanent_attempt_key(value: Mapping[str, Any]) -> dict[str, Any]:
    """Validate the one permanent, non-extensible physical outcome-universe key."""

    if set(value) != {"schema", *_PERMANENT_IDENTITY_FIELDS} or value.get(
        "schema"
    ) != OUTCOME_ACCESS_ATTEMPT_KEY_SCHEMA:
        raise PermissionError("Permanent outcome-attempt key has a noncanonical schema.")
    phase = str(value.get("phase"))
    expected = _PHASE_PERMANENT_EXPECTATIONS.get(phase)
    subjects = value.get("target_subject_ids")
    if expected is None or not isinstance(subjects, list) or any(
        not isinstance(subject, str) for subject in subjects
    ):
        raise PermissionError("Permanent outcome-attempt key has an invalid cohort.")
    subject_hash = canonical_json_sha256({"subject_ids": subjects})
    if (
        subjects != sorted(set(subjects))
        or len(subjects) != expected["subject_count"]
        or value.get("primary_dataset_id") != "wearable_v3_paired_dry_wet"
        or value.get("conditions") != ["dry", "wet"]
        or value.get("query_blocks")
        != ["block06", "block07", "block08", "block09", "block10"]
        or value.get("canonical_target_subject_ids_sha256") != subject_hash
        or subject_hash != expected["target_subject_ids_sha256"]
        or value.get("query_identity_bundle_sha256")
        != expected["query_identity_bundle_sha256"]
        or value.get("trial_inclusion_rule")
        != "five_query_blocks_each_12_class_complete"
        or value.get("outcome_field") != "query_label"
        or value.get("outcome_analysis_unit")
        != "participant_equal_interface_balanced_accuracy"
    ):
        raise PermissionError("Permanent outcome-attempt key differs from the frozen universe.")
    return {key: value[key] for key in _PERMANENT_IDENTITY_FIELDS}


def _validate_outcome_access_scope(value: Mapping[str, Any]) -> dict[str, Any]:
    """Validate exact scope evidence and its embedded permanent key."""

    if set(value) != _SCOPE_FIELDS or value.get("schema") != OUTCOME_ACCESS_SCOPE_SCHEMA:
        raise PermissionError("Outcome-access scope has a noncanonical schema.")
    attempt_key = value.get("permanent_attempt_key")
    if not isinstance(attempt_key, Mapping):
        raise PermissionError("Outcome-access scope has no permanent attempt key.")
    identity = _validate_permanent_attempt_key(attempt_key)
    if (
        value.get("candidate_id") != "metadata-calibration-efficiency-v1"
        or value.get("permanent_attempt_key_sha256")
        != canonical_json_sha256(attempt_key)
        or any(value.get(key) != expected for key, expected in identity.items())
        or value.get("allocation_decision_id") != "DEC-20260830-002"
        or value.get("n_classes") != 12
        or value.get("outcome_scope")
        != "paired_dry_wet_12_class_query_labels_only"
    ):
        raise PermissionError("Outcome-access scope evidence is noncanonical.")
    return dict(identity)


def claim_outcome_access(
    *,
    manifest: Mapping[str, Any],
    decision: Mapping[str, Any],
    signed_private_seal: Mapping[str, Any],
    private_seal_path: str | Path,
    registry_root: str | Path,
    trusted_signing_public_key_path: str | Path,
    authority_signing_private_key_path: str | Path,
    authority_signing_private_key_password: bytes | None,
    authorization_basis: str,
) -> OutcomeAccessClaimBinding:
    """Durably consume the one-shot outcome-access right before decryption."""

    root = Path(registry_root).absolute()
    scope = build_outcome_access_scope(manifest)
    scope_sha256 = canonical_json_sha256(scope)
    attempt_key_sha256 = canonical_json_sha256(
        build_outcome_access_attempt_key(manifest)
    )
    phase = str(manifest["phase"])
    claim_path = root / f"{phase}-{attempt_key_sha256}.json"
    execution_root = Path(str(manifest["execution_output_root"])).absolute()
    run_root = execution_root.parent.parent
    private_path = Path(private_seal_path).absolute()
    phase_directory = "source" if phase == "source_development" else "held"
    expected_private_path = (
        run_root / phase_directory / "private" / "filesystem-private-seal.json"
    )
    expected_public_key = run_root / "keys" / "authority-signing-public.pem"
    persisted_private = read_json_object(private_path)
    if (
        private_path != expected_private_path
        or Path(trusted_signing_public_key_path).absolute() != expected_public_key
        or persisted_private != dict(signed_private_seal)
    ):
        raise PermissionError("Outcome claim lacks the canonical retained authority layout.")
    verify_signed_record(
        persisted_private,
        public_key_path=expected_public_key,
        hash_field="private_seal_payload_sha256",
        expected_schema="cfeg.metadata-calibration-filesystem-private-seal.v1",
    )
    record = {
        "schema": OUTCOME_ACCESS_CLAIM_SCHEMA,
        "candidate_id": "metadata-calibration-efficiency-v1",
        "phase": phase,
        "status": "outcome_access_claimed_irreversible_no_same_cohort_retry",
        "outcome_access_scope": scope,
        "outcome_access_scope_sha256": scope_sha256,
        "outcome_access_attempt_key_sha256": attempt_key_sha256,
        "claim_path": str(claim_path),
        "run_root": str(run_root),
        "phase_manifest_sha256": manifest["phase_manifest_sha256"],
        "outcome_decision_signed_record_sha256": decision["signed_record_sha256"],
        "execution_contract_sha256": manifest["execution_contract_sha256"],
        "filesystem_private_seal_file_sha256": _sha256_file(private_path),
        "filesystem_private_seal_signed_record_sha256": signed_private_seal[
            "signed_record_sha256"
        ],
        "filesystem_private_tree_sha256": signed_private_seal[
            "filesystem_private_tree_sha256"
        ],
        "source_commit": manifest["source_commit"],
        "source_tree_sha256": manifest["source_tree_sha256"],
        "source_tag": manifest["source_tag"],
        "signing_key_id": decision["signing_key_id"],
        "signer_role": decision["signer_role"],
        "authorization_basis": authorization_basis,
        "signature_algorithm": "ed25519",
    }
    signed = issue_signed_record(
        record,
        private_key_path=authority_signing_private_key_path,
        private_key_password=authority_signing_private_key_password,
        hash_field="claim_payload_sha256",
    )
    try:
        _write_claim_exclusive(claim_path, signed)
    except FileExistsError as error:
        raise OutcomeAccessAlreadyClaimed(claim_path, scope_sha256) from error
    return validate_outcome_access_claim(
        claim_path,
        manifest=manifest,
        signed_private_seal=signed_private_seal,
        private_seal_path=private_path,
        trusted_signing_public_key_path=trusted_signing_public_key_path,
        expected_registry_root=root,
    )


def refuse_prior_outcome_access_claim(
    manifest: Mapping[str, Any],
    *,
    registry_root: str | Path = CANONICAL_OUTCOME_ACCESS_REGISTRY_ROOT,
) -> None:
    """Fail before any data access when the stable cohort slot already exists."""

    root = Path(registry_root).absolute()
    audit = audit_outcome_access_registry_for_run(Path("/__cfeg_registry_preflight__"), registry_root=root)
    if audit.registry_integrity_failure:
        raise OutcomeAccessRegistryIntegrityError(
            "Outcome-access registry integrity failed before any phase data access."
        )
    scope_sha256 = canonical_json_sha256(build_outcome_access_scope(manifest))
    attempt_key_sha256 = canonical_json_sha256(
        build_outcome_access_attempt_key(manifest)
    )
    claim_path = root / f"{manifest['phase']}-{attempt_key_sha256}.json"
    if os.path.lexists(claim_path):
        raise OutcomeAccessAlreadyClaimed(claim_path, scope_sha256)


def validate_outcome_access_claim(
    claim_path: str | Path,
    *,
    manifest: Mapping[str, Any],
    signed_private_seal: Mapping[str, Any],
    private_seal_path: str | Path,
    trusted_signing_public_key_path: str | Path,
    expected_registry_root: str | Path = CANONICAL_OUTCOME_ACCESS_REGISTRY_ROOT,
) -> OutcomeAccessClaimBinding:
    binding, claim = validate_outcome_access_claim_evidence(
        claim_path,
        expected_phase=str(manifest["phase"]),
        signed_private_seal=signed_private_seal,
        private_seal_path=private_seal_path,
        trusted_signing_public_key_path=trusted_signing_public_key_path,
        expected_registry_root=expected_registry_root,
    )
    scope = build_outcome_access_scope(manifest)
    scope_sha256 = canonical_json_sha256(scope)
    attempt_key_sha256 = canonical_json_sha256(
        build_outcome_access_attempt_key(manifest)
    )
    path = binding.claim_path
    execution_root = Path(str(manifest["execution_output_root"])).absolute()
    private_path = Path(private_seal_path).absolute()
    expected = {
        "candidate_id": "metadata-calibration-efficiency-v1",
        "phase": manifest["phase"],
        "status": "outcome_access_claimed_irreversible_no_same_cohort_retry",
        "outcome_access_scope": scope,
        "outcome_access_scope_sha256": scope_sha256,
        "outcome_access_attempt_key_sha256": attempt_key_sha256,
        "claim_path": str(path),
        "run_root": str(execution_root.parent.parent),
        "phase_manifest_sha256": manifest["phase_manifest_sha256"],
        "outcome_decision_signed_record_sha256": signed_private_seal[
            "outcome_decision_signed_record_sha256"
        ],
        "execution_contract_sha256": manifest["execution_contract_sha256"],
        "filesystem_private_seal_file_sha256": _sha256_file(private_path),
        "filesystem_private_seal_signed_record_sha256": signed_private_seal[
            "signed_record_sha256"
        ],
        "filesystem_private_tree_sha256": signed_private_seal[
            "filesystem_private_tree_sha256"
        ],
        "source_commit": manifest["source_commit"],
        "source_tree_sha256": manifest["source_tree_sha256"],
        "source_tag": manifest["source_tag"],
        "signing_key_id": signed_private_seal["signing_key_id"],
        "signer_role": signed_private_seal["signer_role"],
        "authorization_basis": signed_private_seal["authorization_basis"],
        "signature_algorithm": "ed25519",
    }
    if any(claim.get(key) != value for key, value in expected.items()):
        raise PermissionError("Outcome-access claim differs from the sealed phase authority.")
    return binding


def validate_outcome_access_claim_evidence(
    claim_path: str | Path,
    *,
    expected_phase: str,
    signed_private_seal: Mapping[str, Any],
    private_seal_path: str | Path,
    trusted_signing_public_key_path: str | Path,
    expected_registry_root: str | Path = CANONICAL_OUTCOME_ACCESS_REGISTRY_ROOT,
) -> tuple[OutcomeAccessClaimBinding, dict[str, Any]]:
    """Validate signed one-shot evidence without needing the full execution manifest."""

    path = Path(claim_path).absolute()
    root = Path(expected_registry_root).absolute()
    observed = path.lstat()
    if (
        not stat.S_ISREG(observed.st_mode)
        or stat.S_ISLNK(observed.st_mode)
        or observed.st_nlink != 1
        or stat.S_IMODE(observed.st_mode) != 0o400
    ):
        raise PermissionError("Outcome-access claim is not one immutable regular file.")
    claim = read_json_object(path)
    signed_hash = verify_signed_record(
        claim,
        public_key_path=trusted_signing_public_key_path,
        hash_field="claim_payload_sha256",
        expected_schema=OUTCOME_ACCESS_CLAIM_SCHEMA,
    )
    scope = claim.get("outcome_access_scope")
    scope_sha256 = claim.get("outcome_access_scope_sha256")
    if not isinstance(scope, Mapping) or scope_sha256 != canonical_json_sha256(scope):
        raise PermissionError("Outcome-access claim has an invalid stable scope.")
    _validate_outcome_access_scope(scope)
    attempt_key = scope.get("permanent_attempt_key") if isinstance(scope, Mapping) else None
    attempt_key_sha256 = claim.get("outcome_access_attempt_key_sha256")
    if (
        not isinstance(attempt_key, Mapping)
        or scope.get("permanent_attempt_key_sha256")
        != canonical_json_sha256(attempt_key)
        or attempt_key_sha256 != canonical_json_sha256(attempt_key)
    ):
        raise PermissionError("Outcome-access claim has an invalid permanent attempt key.")
    expected_path = root / f"{expected_phase}-{attempt_key_sha256}.json"
    private_path = Path(private_seal_path).absolute()
    if (
        path != expected_path
        or set(claim) != _CLAIM_FIELDS
        or claim.get("candidate_id") != "metadata-calibration-efficiency-v1"
        or claim.get("phase") != expected_phase
        or claim.get("status")
        != "outcome_access_claimed_irreversible_no_same_cohort_retry"
        or claim.get("claim_path") != str(path)
        or claim.get("filesystem_private_seal_file_sha256")
        != _sha256_file(private_path)
        or claim.get("filesystem_private_seal_signed_record_sha256")
        != signed_private_seal.get("signed_record_sha256")
        or claim.get("filesystem_private_tree_sha256")
        != signed_private_seal.get("filesystem_private_tree_sha256")
        or claim.get("outcome_decision_signed_record_sha256")
        != signed_private_seal.get("outcome_decision_signed_record_sha256")
        or claim.get("execution_contract_sha256")
        != signed_private_seal.get("execution_contract_sha256")
        or claim.get("signing_key_id") != signed_private_seal.get("signing_key_id")
        or claim.get("signer_role") != signed_private_seal.get("signer_role")
        or claim.get("authorization_basis")
        != signed_private_seal.get("authorization_basis")
        or claim.get("signature_algorithm") != "ed25519"
    ):
        raise PermissionError("Outcome-access claim evidence is not bound to the private seal.")
    run_root = Path(str(claim.get("run_root"))).absolute()
    return (
        OutcomeAccessClaimBinding(
            claim_path=path,
            claim_file_sha256=_sha256_file(path),
            signed_record_sha256=signed_hash,
            outcome_access_scope_sha256=str(scope_sha256),
            run_root=run_root,
        ),
        claim,
    )


def outcome_access_started_for_run(
    run_root: str | Path,
    *,
    registry_root: str | Path = CANONICAL_OUTCOME_ACCESS_REGISTRY_ROOT,
) -> bool:
    """Detect a durable claim issued by one lifecycle run."""

    return audit_outcome_access_registry_for_run(
        run_root, registry_root=registry_root
    ).current_run_claim_found


def audit_outcome_access_registry_for_run(
    run_root: str | Path,
    *,
    registry_root: str | Path = CANONICAL_OUTCOME_ACCESS_REGISTRY_ROOT,
) -> OutcomeAccessRegistryAudit:
    """Separate current-run evidence from unrelated registry corruption."""

    expected_run = str(Path(run_root).absolute())
    root = Path(registry_root).absolute()
    if not root.exists():
        return OutcomeAccessRegistryAudit(False, False, ())
    observed = root.lstat()
    if (
        not stat.S_ISDIR(observed.st_mode)
        or stat.S_ISLNK(observed.st_mode)
        or stat.S_IMODE(observed.st_mode) != 0o700
    ):
        return OutcomeAccessRegistryAudit(False, True, ())
    integrity_failure = False
    current_run_claimed_phases: set[str] = set()
    for path in root.iterdir():
        try:
            claim_run_root, claim_phase = _audit_registry_claim(
                path, registry_root=root
            )
        except (OSError, TypeError, ValueError, PermissionError):
            integrity_failure = True
            continue
        if claim_run_root == expected_run:
            current_run_claimed_phases.add(claim_phase)
    phases = tuple(sorted(current_run_claimed_phases))
    return OutcomeAccessRegistryAudit(bool(phases), integrity_failure, phases)


def _audit_registry_claim(path: Path, *, registry_root: Path) -> tuple[str, str]:
    """Authenticate one retained registry claim and all durable authority it cites."""

    item = path.lstat()
    if (
        not stat.S_ISREG(item.st_mode)
        or stat.S_ISLNK(item.st_mode)
        or item.st_nlink != 1
        or stat.S_IMODE(item.st_mode) != 0o400
    ):
        raise PermissionError("Registry entry is not one immutable regular file.")
    claim = read_json_object(path)
    if set(claim) != _CLAIM_FIELDS:
        raise PermissionError("Registry claim does not have the exact schema.")
    phase = claim.get("phase")
    phase_directory = {
        "source_development": "source",
        "held_participant_evaluation": "held",
    }.get(str(phase))
    if phase_directory is None:
        raise PermissionError("Registry claim has an invalid phase.")
    scope = claim.get("outcome_access_scope")
    if not isinstance(scope, Mapping):
        raise PermissionError("Registry claim has no stable scope.")
    identity = _validate_outcome_access_scope(scope)
    scope_sha256 = canonical_json_sha256(scope)
    attempt_key = scope.get("permanent_attempt_key")
    if not isinstance(attempt_key, Mapping):
        raise PermissionError("Registry claim has no permanent attempt key.")
    attempt_key_sha256 = canonical_json_sha256(attempt_key)
    if (
        claim.get("schema") != OUTCOME_ACCESS_CLAIM_SCHEMA
        or claim.get("candidate_id") != "metadata-calibration-efficiency-v1"
        or claim.get("status")
        != "outcome_access_claimed_irreversible_no_same_cohort_retry"
        or scope.get("schema") != OUTCOME_ACCESS_SCOPE_SCHEMA
        or scope.get("permanent_attempt_key_sha256") != attempt_key_sha256
        or claim.get("outcome_access_scope_sha256") != scope_sha256
        or claim.get("outcome_access_attempt_key_sha256") != attempt_key_sha256
        or path != registry_root / f"{phase}-{attempt_key_sha256}.json"
        or claim.get("claim_path") != str(path)
    ):
        raise PermissionError("Registry claim path or permanent identity is invalid.")
    if dict(attempt_key) != {
        "schema": OUTCOME_ACCESS_ATTEMPT_KEY_SCHEMA,
        **identity,
    }:
        raise PermissionError("Registry scope and permanent attempt key disagree.")

    run_root_text = claim.get("run_root")
    if not isinstance(run_root_text, str):
        raise PermissionError("Registry claim has no absolute run root.")
    run_root = Path(run_root_text)
    if not run_root.is_absolute() or str(run_root.absolute()) != run_root_text:
        raise PermissionError("Registry claim run root is noncanonical.")
    run_stat = run_root.lstat()
    if not stat.S_ISDIR(run_stat.st_mode) or stat.S_ISLNK(run_stat.st_mode):
        raise PermissionError("Registry claim run root is not retained safely.")
    public_key = run_root / "keys" / "authority-signing-public.pem"
    key_stat = public_key.lstat()
    if (
        not stat.S_ISREG(key_stat.st_mode)
        or stat.S_ISLNK(key_stat.st_mode)
        or key_stat.st_nlink != 1
        or stat.S_IMODE(key_stat.st_mode) != 0o444
    ):
        raise PermissionError("Registry claim public authority key is invalid.")
    verify_signed_record(
        claim,
        public_key_path=public_key,
        hash_field="claim_payload_sha256",
        expected_schema=OUTCOME_ACCESS_CLAIM_SCHEMA,
    )
    private_path = (
        run_root / phase_directory / "private" / "filesystem-private-seal.json"
    )
    private = read_json_object(private_path)
    private_signed_hash = verify_signed_record(
        private,
        public_key_path=public_key,
        hash_field="private_seal_payload_sha256",
        expected_schema="cfeg.metadata-calibration-filesystem-private-seal.v1",
    )
    execution_root = Path(str(private.get("execution_output_root")))
    if (
        not execution_root.is_absolute()
        or execution_root != run_root / phase_directory / "execution"
        or claim.get("phase") != private.get("phase")
        or claim.get("filesystem_private_seal_file_sha256")
        != _sha256_file(private_path)
        or claim.get("filesystem_private_seal_signed_record_sha256")
        != private_signed_hash
        or claim.get("filesystem_private_tree_sha256")
        != private.get("filesystem_private_tree_sha256")
        or claim.get("outcome_decision_signed_record_sha256")
        != private.get("outcome_decision_signed_record_sha256")
        or claim.get("execution_contract_sha256")
        != private.get("execution_contract_sha256")
        or claim.get("source_commit") != private.get("source_commit")
        or claim.get("source_tree_sha256") != private.get("source_tree_sha256")
        or claim.get("source_tag") != private.get("source_tag")
        or claim.get("signing_key_id") != private.get("signing_key_id")
        or claim.get("signer_role") != private.get("signer_role")
        or claim.get("authorization_basis") != private.get("authorization_basis")
        or claim.get("signature_algorithm") != "ed25519"
        or private.get("signature_algorithm") != "ed25519"
    ):
        raise PermissionError("Registry claim is not bound to its retained private seal.")
    return run_root_text, str(phase)


def _write_claim_exclusive(path: Path, claim: Mapping[str, Any]) -> None:
    root = path.parent
    root.mkdir(parents=True, mode=0o700, exist_ok=True)
    observed = root.lstat()
    if (
        not stat.S_ISDIR(observed.st_mode)
        or stat.S_ISLNK(observed.st_mode)
        or stat.S_IMODE(observed.st_mode) != 0o700
    ):
        raise PermissionError("Outcome-access registry is not one real directory.")
    staging_root = root.with_name(f"{root.name}-staging")
    staging_root.mkdir(parents=True, mode=0o700, exist_ok=True)
    staging_root_stat = staging_root.lstat()
    if (
        not stat.S_ISDIR(staging_root_stat.st_mode)
        or stat.S_ISLNK(staging_root_stat.st_mode)
        or stat.S_IMODE(staging_root_stat.st_mode) != 0o700
    ):
        raise PermissionError("Outcome-access staging root is not one private directory.")
    staging = staging_root / f"{path.name}.staging-{os.getpid()}-{os.urandom(8).hex()}"
    descriptor = os.open(
        staging,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0),
        0o400,
    )
    try:
        os.fchmod(descriptor, 0o400)
        payload = canonical_json_bytes(claim) + b"\n"
        written = 0
        while written < len(payload):
            written += os.write(descriptor, payload[written:])
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    try:
        _rename_no_replace(staging, path)
        _fsync_directory(root)
        _fsync_directory(root.parent)
        _fsync_directory(staging_root)
    finally:
        if os.path.lexists(staging):
            staging.unlink()
            _fsync_directory(staging_root)


def _rename_no_replace(source: Path, destination: Path) -> None:
    """Atomically publish one claim without a hard-link crash window."""

    libc = ctypes.CDLL(None, use_errno=True)
    renameat2 = getattr(libc, "renameat2", None)
    if renameat2 is None:
        raise RuntimeError("Outcome-claim publication requires Linux renameat2.")
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
        os.fsencode(source),
        -100,
        os.fsencode(destination),
        1,
    )
    if result != 0:
        number = ctypes.get_errno()
        if number == errno.EEXIST:
            raise FileExistsError(number, os.strerror(number), destination)
        raise OSError(number, os.strerror(number), destination)


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
