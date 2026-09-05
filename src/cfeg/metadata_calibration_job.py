from __future__ import annotations

import ctypes
import errno
import hashlib
import json
import os
import shutil
import stat
import uuid
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

import numpy as np
import pandas as pd
from pandas.api.types import is_integer_dtype

from cfeg.analysis.metadata_calibration_efficiency import (
    MetadataCalibrationBundleBindings,
    metadata_calibration_bundle_spec_for_phase,
)
from cfeg.data.metadata_calibration_baseline_sealed import (
    exact_source_pool_identities,
    load_baseline_source_fit,
)
from cfeg.identity import canonical_identity_sha256
from cfeg.metadata_calibration_atomic import validate_and_summarize_support_rows
from cfeg.metadata_calibration_authority import verify_signed_record
from cfeg.metadata_calibration_baseline_contract import baseline_execution_binding
from cfeg.metadata_calibration_checkpoint import verify_full_composite_checkpoint
from cfeg.metadata_calibration_execution import cell_execution_contract_sha256
from cfeg.metadata_calibration_manifest import validate_authorization_envelope
from cfeg.metadata_calibration_sealer import validate_input_seal_root
from cfeg.utils.config import load_config

_REPOSITORY = Path(__file__).resolve().parents[2]
_PRODUCER_RECEIPT_SCHEMA = "cfeg.metadata-calibration-producer-receipt.v1"
JOB_CAPABILITY_SCHEMA = "cfeg.metadata-calibration-job-capability.v1"
_CANDIDATE_RUNNER_INPUTS = (
    "block_context.jsonl",
    "source_fit/labeled_view.jsonl",
    "source_fit/signals.h5",
    "target_query/expected_unlabeled.jsonl",
    "target_query/model_view.jsonl",
    "target_query/signals.h5",
    "target_support/model_view.jsonl",
    "target_support/signals.h5",
    "target_support/support_labels.jsonl",
)
_BASELINE_QUERY_INPUTS = (
    "target_query/baseline_view.jsonl",
    "target_query/expected_unlabeled.jsonl",
    "target_query/signals.h5",
)
_BASELINE_SUPPORT_INPUTS = (
    "target_support/baseline_view.jsonl",
    "target_support/signals.h5",
    "target_support/support_labels.jsonl",
)
_BASELINE_SOURCE_INPUTS = (
    "source_fit/baseline_view.jsonl",
    "source_fit/signals.h5",
)
_RENAME_NOREPLACE = 1
_JOB_CAPABILITY_FIELDS = {
    "schema",
    "candidate_id",
    "phase",
    "job_id",
    "job_spec",
    "group_root",
    "execution_output_root",
    "canonical_job_root",
    "required_artifacts",
    "input_checkpoint_group_tree_sha256",
    "runner_projection_tree_sha256",
    "runner_input_relative_paths",
    "bindings",
    "source_commit",
    "source_tree_sha256",
    "source_tag",
    "phase_manifest_sha256",
    "outcome_decision_signed_record_sha256",
    "authorization_envelope_sha256",
    "input_seal_decision_signed_record_sha256",
    "input_seal_receipt_signed_record_sha256",
    "processed_asset_root",
    "signing_key_id",
    "signer_role",
    "authorization_basis",
    "signature_algorithm",
    "capability_payload_sha256",
    "signature",
    "signed_record_sha256",
}
_CANDIDATE_RECEIPT_FIELDS = {
    "schema",
    "candidate_id",
    "phase",
    "producer_kind",
    "job_id",
    "job_contract_sha256",
    "checkpoint_group",
    "input_checkpoint_group_tree_sha256",
    "bindings",
    "output_files",
    "rows",
    "query_outcomes_loaded",
    "raw_sample_ids_loaded",
    "checkpoint_sha256",
    "selected_stage1_epochs",
    "selected_stage2_epochs",
    "runtime",
    "producer_receipt_sha256",
}
_BASELINE_RECEIPT_FIELDS = {
    "schema",
    "candidate_id",
    "phase",
    "producer_kind",
    "job_id",
    "job_contract_sha256",
    "checkpoint_group",
    "input_checkpoint_group_tree_sha256",
    "bindings",
    "output_files",
    "rows",
    "query_outcomes_loaded",
    "raw_sample_ids_loaded",
    "adapter_id",
    "deterministic_seed_column_present",
    "producer_receipt_sha256",
}
_CANDIDATE_FIXED_COLUMNS = {
    "candidate_id",
    "phase",
    "checkpoint_group",
    "checkpoint_sha256",
    "role",
    "context",
    "seed",
    "budget",
    "query_token",
    "subject_id",
    "electrode_type",
    "support_identity_sha256",
    "query_identity_sha256",
    "intervention_mapping_sha256",
    "resolved_context_usage_sha256",
    "cell_execution_contract_sha256",
}
_BASELINE_FIXED_COLUMNS = {
    "candidate_id",
    "phase",
    "checkpoint_group",
    "adapter_id",
    "budget",
    "query_token",
    "subject_id",
    "electrode_type",
    "support_identity_sha256",
    "query_identity_sha256",
    "source_pool_identity_sha256",
    "implementation_sha256",
    "config_sha256",
    "adapter_contract_sha256",
    "score_schema",
    "score_interpretation",
}


@dataclass(frozen=True)
class AuthorizedMetadataCalibrationJob:
    phase: str
    job_spec: dict[str, Any]
    group_root: Path
    execution_output_root: Path
    canonical_job_root: Path
    required_artifacts: dict[str, str]
    input_checkpoint_group_tree_sha256: str
    bindings: MetadataCalibrationBundleBindings
    source_commit: str
    source_tree_sha256: str
    source_tag: str


@dataclass(frozen=True)
class CompletedMetadataCalibrationJob:
    job_id: str
    producer_kind: str
    canonical_job_root: Path
    producer_receipt_file_sha256: str
    producer_receipt: dict[str, Any]


def build_metadata_calibration_job_capability(
    job: AuthorizedMetadataCalibrationJob,
    *,
    phase_manifest_sha256: str,
    outcome_decision_signed_record_sha256: str,
    authorization_envelope_sha256: str,
    signing_key_id: str,
    authorization_basis: str,
) -> dict[str, Any]:
    """Describe one least-capability worker job for authority signing."""

    input_root = job.group_root.parent
    input_decision = _read_regular_json(input_root / "input_seal_decision.json")
    input_receipt = _read_regular_json(input_root / "input_seal_receipt.json")
    return {
        "schema": JOB_CAPABILITY_SCHEMA,
        "candidate_id": "metadata-calibration-efficiency-v1",
        "phase": job.phase,
        "job_id": str(job.job_spec["job_id"]),
        "job_spec": dict(job.job_spec),
        "group_root": str(job.group_root),
        "execution_output_root": str(job.execution_output_root),
        "canonical_job_root": str(job.canonical_job_root),
        "required_artifacts": dict(job.required_artifacts),
        "input_checkpoint_group_tree_sha256": job.input_checkpoint_group_tree_sha256,
        "runner_input_relative_paths": list(
            runner_input_relative_paths(job.job_spec)
        ),
        "runner_projection_tree_sha256": _runner_projection_tree_sha256(
            job.group_root, runner_input_relative_paths(job.job_spec)
        ),
        "bindings": job.bindings.as_dict(),
        "source_commit": job.source_commit,
        "source_tree_sha256": job.source_tree_sha256,
        "source_tag": job.source_tag,
        "phase_manifest_sha256": phase_manifest_sha256,
        "outcome_decision_signed_record_sha256": outcome_decision_signed_record_sha256,
        "authorization_envelope_sha256": authorization_envelope_sha256,
        "input_seal_decision_signed_record_sha256": input_decision[
            "signed_record_sha256"
        ],
        "input_seal_receipt_signed_record_sha256": input_receipt[
            "signed_record_sha256"
        ],
        "processed_asset_root": input_decision["processed_asset_root"],
        "signing_key_id": signing_key_id,
        "signer_role": "delegated_automation_authority",
        "authorization_basis": authorization_basis,
        "signature_algorithm": "ed25519",
    }


def resolve_projected_metadata_calibration_job(
    *,
    job_capability_path: str | Path,
    trusted_signing_public_key_path: str | Path,
    job_id: str,
) -> AuthorizedMetadataCalibrationJob:
    """Resolve one signed job against its isolated single-group input projection."""

    capability_path = Path(job_capability_path).absolute()
    capability = _read_regular_json(capability_path)
    verify_signed_record(
        capability,
        public_key_path=trusted_signing_public_key_path,
        hash_field="capability_payload_sha256",
        expected_schema=JOB_CAPABILITY_SCHEMA,
    )
    if (
        set(capability) != _JOB_CAPABILITY_FIELDS
        or capability.get("candidate_id") != "metadata-calibration-efficiency-v1"
        or capability.get("job_id") != job_id
        or capability_path.name != f"{job_id}.json"
        or capability.get("signer_role") != "delegated_automation_authority"
        or capability.get("signature_algorithm") != "ed25519"
    ):
        raise PermissionError("Projected job capability has an invalid exact schema.")
    job_spec = capability.get("job_spec")
    required = capability.get("required_artifacts")
    binding_values = capability.get("bindings")
    if (
        not isinstance(job_spec, Mapping)
        or job_spec.get("job_id") != job_id
        or not isinstance(required, Mapping)
        or not isinstance(binding_values, Mapping)
    ):
        raise PermissionError("Projected job capability lacks an exact job binding.")
    artifacts = _exact_required_artifacts(job_spec)
    if dict(required) != artifacts:
        raise PermissionError("Projected job artifacts differ from the job contract.")
    phase = str(capability["phase"])
    runner_paths = capability.get("runner_input_relative_paths")
    if (
        not isinstance(runner_paths, list)
        or runner_paths != list(runner_input_relative_paths(job_spec))
    ):
        raise PermissionError("Projected job has an invalid least-capability allowlist.")
    checkpoint_group = str(job_spec["checkpoint_group"])
    group_root = Path(str(capability["group_root"])).absolute()
    execution_root = Path(str(capability["execution_output_root"])).absolute()
    canonical_root = Path(str(capability["canonical_job_root"])).absolute()
    if group_root.name != checkpoint_group:
        raise PermissionError("Projected input group path differs from the signed job.")
    _require_real_directory(group_root)
    receipt_relative = PurePosixPath(str(job_spec["required_receipt"]))
    expected_canonical = execution_root.joinpath(*receipt_relative.parent.parts)
    if canonical_root != expected_canonical:
        raise PermissionError("Projected output path differs from the signed job.")
    _require_beneath(canonical_root, execution_root)
    runner_tree = _runner_projection_tree_sha256(
        group_root, tuple(runner_paths), require_exact=True
    )
    if runner_tree != capability.get("runner_projection_tree_sha256"):
        raise PermissionError("Runner input projection differs from its signed capability.")
    group_tree = str(capability.get("input_checkpoint_group_tree_sha256"))
    if not _is_sha256(group_tree):
        raise PermissionError("Projected capability lacks the sealed full-group commitment.")
    bindings = MetadataCalibrationBundleBindings(**dict(binding_values))
    bindings.as_dict()
    input_receipt = _read_regular_json(group_root.parent / "input_seal_receipt.json")
    if (
        capability.get("phase_manifest_sha256")
        != bindings.execution_manifest_sha256
        or capability.get("outcome_decision_signed_record_sha256")
        != bindings.decision_receipt_sha256
        or capability.get("source_tree_sha256") != bindings.source_tree_sha256
        or capability.get("input_seal_receipt_signed_record_sha256")
        != input_receipt.get("signed_record_sha256")
        or input_receipt.get("raw_asset_receipt_sha256")
        != bindings.asset_receipt_sha256
        or input_receipt.get("raw_asset_fingerprint_bundle_sha256")
        != bindings.asset_fingerprint_bundle_sha256
        or input_receipt.get("class_map_sha256") != bindings.class_map_sha256
        or input_receipt.get("signing_key_id") != capability.get("signing_key_id")
        or input_receipt.get("authorization_basis")
        != capability.get("authorization_basis")
    ):
        raise PermissionError("Projected capability provenance chain is internally inconsistent.")
    return AuthorizedMetadataCalibrationJob(
        phase=phase,
        job_spec=dict(job_spec),
        group_root=group_root,
        execution_output_root=execution_root,
        canonical_job_root=canonical_root,
        required_artifacts=artifacts,
        input_checkpoint_group_tree_sha256=group_tree,
        bindings=bindings,
        source_commit=str(capability["source_commit"]),
        source_tree_sha256=str(capability["source_tree_sha256"]),
        source_tag=str(capability["source_tag"]),
    )


def resolve_authorized_metadata_calibration_job(
    *,
    manifest_path: str | Path,
    outcome_decision_path: str | Path,
    authorization_envelope_path: str | Path,
    trusted_signing_public_key_path: str | Path,
    job_id: str,
) -> AuthorizedMetadataCalibrationJob:
    """Resolve one exact job from the signed manifest chain, never caller payloads."""

    manifest = _read_regular_json(manifest_path)
    decision = _read_regular_json(outcome_decision_path)
    envelope = _read_regular_json(authorization_envelope_path)
    phase_binding = validate_authorization_envelope(
        envelope,
        manifest=manifest,
        outcome_decision=decision,
        trusted_signing_public_key_path=trusted_signing_public_key_path,
    )
    contract = manifest["execution_contract"]
    jobs = contract["jobs"]
    if not isinstance(jobs, list) or len({str(job.get("job_id")) for job in jobs}) != len(
        jobs
    ):
        raise ValueError("Execution manifest contains duplicate or invalid job IDs.")
    selected = [dict(job) for job in jobs if str(job.get("job_id")) == str(job_id)]
    if len(selected) != 1:
        raise ValueError("Requested job is not one exact manifest member.")
    job = selected[0]
    phase = str(phase_binding.phase)
    checkpoint_group = str(job["checkpoint_group"])

    input_binding = validate_input_seal_root(
        str(manifest["input_seal_root"]),
        trusted_signing_public_key_path=trusted_signing_public_key_path,
    )
    if input_binding.phase != phase or checkpoint_group not in input_binding.checkpoint_groups:
        raise ValueError("Authorized job checkpoint group differs from the input seal.")
    group_root = input_binding.sealed_root / checkpoint_group
    _require_real_directory(group_root)
    group_tree = _tree_sha256(group_root)

    execution_root = Path(phase_binding.execution_output_root).absolute()
    artifacts = _exact_required_artifacts(job)
    receipt_relative = PurePosixPath(str(job["required_receipt"]))
    job_relative = receipt_relative.parent
    if any(PurePosixPath(path).parent != job_relative for path in artifacts):
        raise ValueError("One manifest job must publish into exactly one canonical directory.")
    canonical_job_root = execution_root.joinpath(*job_relative.parts)
    _require_beneath(canonical_job_root, execution_root)

    input_receipt = _read_regular_json(input_binding.sealed_root / "input_seal_receipt.json")
    globals_ = contract["global_contracts"]
    bindings = MetadataCalibrationBundleBindings(
        plan_sha256=str(globals_["plan_sha256"]),
        execution_manifest_sha256=str(manifest["phase_manifest_sha256"]),
        decision_receipt_sha256=str(
            phase_binding.outcome_decision_signed_record_sha256
        ),
        asset_receipt_sha256=str(input_receipt["raw_asset_receipt_sha256"]),
        asset_fingerprint_bundle_sha256=str(
            input_receipt["raw_asset_fingerprint_bundle_sha256"]
        ),
        class_map_sha256=str(input_receipt["class_map_sha256"]),
        source_tree_sha256=str(manifest["source_tree_sha256"]),
    )
    bindings.as_dict()
    return AuthorizedMetadataCalibrationJob(
        phase=phase,
        job_spec=job,
        group_root=group_root,
        execution_output_root=execution_root,
        canonical_job_root=canonical_job_root,
        required_artifacts=artifacts,
        input_checkpoint_group_tree_sha256=group_tree,
        bindings=bindings,
        source_commit=str(manifest["source_commit"]),
        source_tree_sha256=str(manifest["source_tree_sha256"]),
        source_tag=str(manifest["source_tag"]),
    )


def run_authorized_metadata_calibration_job(
    *,
    manifest_path: str | Path,
    outcome_decision_path: str | Path,
    authorization_envelope_path: str | Path,
    trusted_signing_public_key_path: str | Path,
    job_id: str,
) -> CompletedMetadataCalibrationJob:
    """Reject the legacy full-input execution path outside the isolated worker."""

    del (
        manifest_path,
        outcome_decision_path,
        authorization_envelope_path,
        trusted_signing_public_key_path,
        job_id,
    )
    raise PermissionError(
        "Direct full-input job execution is disabled; use the signed single-group "
        "Bubblewrap worker capability."
    )


def run_projected_metadata_calibration_job(
    *,
    job_capability_path: str | Path,
    trusted_signing_public_key_path: str | Path,
    repository_root: str | Path,
    checkpoint_group: str,
    job_id: str,
) -> tuple[CompletedMetadataCalibrationJob, dict[str, Any]]:
    """Run one signed job whose namespace contains only its checkpoint group."""

    from cfeg.metadata_calibration_isolation import verify_bubblewrap_worker_isolation

    isolation, job = verify_bubblewrap_worker_isolation(
        job_capability_path=job_capability_path,
        trusted_signing_public_key_path=trusted_signing_public_key_path,
        repository_root=repository_root,
        expected_checkpoint_group=checkpoint_group,
        expected_job_id=job_id,
    )
    if str(job.job_spec["checkpoint_group"]) != checkpoint_group:
        raise PermissionError("Projected job checkpoint group differs from its capability.")
    return _run_resolved_metadata_calibration_job(job), isolation


def _run_resolved_metadata_calibration_job(
    job: AuthorizedMetadataCalibrationJob,
) -> CompletedMetadataCalibrationJob:
    from cfeg.metadata_calibration_producer import run_baseline_job, run_candidate_job

    if os.path.lexists(job.canonical_job_root):
        raise FileExistsError("Canonical job output already exists; in-place retry is forbidden.")
    _make_private_parents(job.execution_output_root, job.canonical_job_root.parent)
    staging = job.canonical_job_root.parent / (
        f".{job.canonical_job_root.name}.staging-{uuid.uuid4().hex}"
    )
    try:
        if job.job_spec["producer_kind"] == "candidate":
            run_candidate_job(
                group_root=job.group_root,
                job_root=staging,
                job_spec=job.job_spec,
                phase=job.phase,  # type: ignore[arg-type]
                bindings=job.bindings,
                source_commit=job.source_commit,
                source_tree_sha256=job.source_tree_sha256,
                source_tag=job.source_tag,
                input_checkpoint_group_tree_sha256=(
                    job.input_checkpoint_group_tree_sha256
                ),
            )
        elif job.job_spec["producer_kind"] == "baseline":
            run_baseline_job(
                group_root=job.group_root,
                job_root=staging,
                job_spec=job.job_spec,
                phase=job.phase,  # type: ignore[arg-type]
                bindings=job.bindings,
                input_checkpoint_group_tree_sha256=(
                    job.input_checkpoint_group_tree_sha256
                ),
            )
        else:
            raise ValueError("Manifest job has an unknown producer kind.")
        completed = validate_metadata_calibration_job_staging(job, staging)
        _lock_down_and_fsync_tree(staging)
        publish_tree_noreplace(staging, job.canonical_job_root)
        return CompletedMetadataCalibrationJob(
            job_id=completed.job_id,
            producer_kind=completed.producer_kind,
            canonical_job_root=job.canonical_job_root,
            producer_receipt_file_sha256=_sha256_file(
                job.canonical_job_root / "producer_receipt.json"
            ),
            producer_receipt=completed.producer_receipt,
        )
    except BaseException:
        if os.path.lexists(staging):
            shutil.rmtree(staging)
        raise


def validate_published_metadata_calibration_job(
    job: AuthorizedMetadataCalibrationJob,
) -> CompletedMetadataCalibrationJob:
    completed = validate_metadata_calibration_job_staging(
        job,
        job.canonical_job_root,
        require_locked_modes=True,
    )
    return CompletedMetadataCalibrationJob(
        job_id=completed.job_id,
        producer_kind=completed.producer_kind,
        canonical_job_root=job.canonical_job_root,
        producer_receipt_file_sha256=_sha256_file(
            job.canonical_job_root / "producer_receipt.json"
        ),
        producer_receipt=completed.producer_receipt,
    )


def publish_validated_metadata_calibration_job_spool(
    job: AuthorizedMetadataCalibrationJob,
    spool_job_root: str | Path,
    *,
    expected_spool_identity: tuple[int, int] | None = None,
) -> CompletedMetadataCalibrationJob:
    """Validate an isolated worker spool, then no-replace publish it as the real job."""

    spool = Path(spool_job_root).absolute()
    _require_expected_directory_identity(spool, expected_spool_identity)
    if os.path.lexists(job.canonical_job_root):
        raise FileExistsError("Canonical job output already exists; retry is forbidden.")
    completed = validate_metadata_calibration_job_staging(
        job,
        spool,
        require_locked_modes=True,
    )
    _require_expected_directory_identity(spool, expected_spool_identity)
    _make_private_parents(job.execution_output_root, job.canonical_job_root.parent)
    _require_expected_directory_identity(spool, expected_spool_identity)
    publish_tree_noreplace(spool, job.canonical_job_root)
    return CompletedMetadataCalibrationJob(
        job_id=completed.job_id,
        producer_kind=completed.producer_kind,
        canonical_job_root=job.canonical_job_root,
        producer_receipt_file_sha256=_sha256_file(
            job.canonical_job_root / "producer_receipt.json"
        ),
        producer_receipt=completed.producer_receipt,
    )


def _require_expected_directory_identity(
    path: Path, expected: tuple[int, int] | None
) -> None:
    if expected is None:
        return
    observed = path.lstat()
    if (
        not stat.S_ISDIR(observed.st_mode)
        or stat.S_ISLNK(observed.st_mode)
        or (observed.st_dev, observed.st_ino) != expected
    ):
        raise PermissionError("Worker spool directory identity changed during validation.")


def validate_metadata_calibration_job_staging(
    job: AuthorizedMetadataCalibrationJob,
    root: str | Path,
    *,
    require_locked_modes: bool = False,
) -> CompletedMetadataCalibrationJob:
    target = Path(root).absolute()
    _validate_exact_job_tree(
        target,
        expected_names={PurePosixPath(path).name for path in job.required_artifacts},
        require_locked_modes=require_locked_modes,
    )
    receipt_path = target / "producer_receipt.json"
    receipt = _read_regular_json(receipt_path)
    expected_receipt_fields = (
        _CANDIDATE_RECEIPT_FIELDS
        if job.job_spec["producer_kind"] == "candidate"
        else _BASELINE_RECEIPT_FIELDS
    )
    if set(receipt) != expected_receipt_fields:
        raise ValueError("Producer receipt does not use its exact kind-specific schema.")
    claimed = receipt.get("producer_receipt_sha256")
    payload = dict(receipt)
    payload.pop("producer_receipt_sha256", None)
    if claimed != _json_sha256(payload):
        raise ValueError("Producer receipt self-hash is invalid.")
    exact_common = {
        "schema": _PRODUCER_RECEIPT_SCHEMA,
        "candidate_id": "metadata-calibration-efficiency-v1",
        "phase": job.phase,
        "producer_kind": job.job_spec["producer_kind"],
        "job_id": job.job_spec["job_id"],
        "job_contract_sha256": job.job_spec["job_contract_sha256"],
        "checkpoint_group": job.job_spec["checkpoint_group"],
        "input_checkpoint_group_tree_sha256": (
            job.input_checkpoint_group_tree_sha256
        ),
        "bindings": job.bindings.as_dict(),
        "query_outcomes_loaded": False,
        "raw_sample_ids_loaded": False,
    }
    if any(receipt.get(field) != value for field, value in exact_common.items()):
        raise ValueError("Producer receipt differs from its authorized manifest/input chain.")
    _validate_output_file_records(job, target, receipt)
    if job.job_spec["producer_kind"] == "candidate":
        _validate_candidate_artifacts(job, target, receipt)
    else:
        _validate_baseline_artifacts(job, target, receipt)
    return CompletedMetadataCalibrationJob(
        job_id=str(receipt["job_id"]),
        producer_kind=str(receipt["producer_kind"]),
        canonical_job_root=target,
        producer_receipt_file_sha256=_sha256_file(receipt_path),
        producer_receipt=receipt,
    )


def _validate_candidate_artifacts(
    job: AuthorizedMetadataCalibrationJob,
    root: Path,
    receipt: Mapping[str, Any],
) -> None:
    checkpoint_receipt = verify_full_composite_checkpoint(
        root / "composite.safetensors",
        root / "composite.receipt.json",
    )
    recipe = load_config(
        _REPOSITORY / "configs/train/metadata_calibration_efficiency_v1.yaml",
        strict_env=False,
    )
    exact_checkpoint = {
        "candidate_id": "metadata-calibration-efficiency-v1",
        "phase": job.phase,
        "seed": job.job_spec["seed"],
        "outer_fold": job.job_spec["outer_fold"],
        "plan_sha256": job.bindings.plan_sha256,
        "decision_receipt_sha256": job.bindings.decision_receipt_sha256,
        "asset_receipt_sha256": job.bindings.asset_receipt_sha256,
        "asset_fingerprint_bundle_sha256": (
            job.bindings.asset_fingerprint_bundle_sha256
        ),
        "class_map_sha256": job.bindings.class_map_sha256,
        "source_commit": job.source_commit,
        "source_tree_sha256": job.source_tree_sha256,
        "source_tag": job.source_tag,
        "stage1_recipe_sha256": _json_sha256(recipe["stage_1_common_Q"]),
        "stage2_recipe_sha256": _json_sha256(recipe["stage_2_bounded_M"]),
    }
    if any(checkpoint_receipt.get(key) != value for key, value in exact_checkpoint.items()):
        raise ValueError("Composite checkpoint receipt differs from the authorized job.")
    if receipt.get("checkpoint_sha256") != checkpoint_receipt["checkpoint_sha256"]:
        raise ValueError("Producer receipt does not bind its verified composite checkpoint.")

    selection = _read_regular_json(root / "training_selection.json")
    if set(selection) != {
        "stage1_epochs",
        "stage2_epochs",
        "stage1_history",
        "stage2_history",
    } or type(selection["stage1_epochs"]) is not int or type(
        selection["stage2_epochs"]
    ) is not int or not isinstance(selection["stage1_history"], list) or not isinstance(
        selection["stage2_history"], list
    ):
        raise ValueError("Training-selection artifact has an invalid exact schema.")
    if receipt.get("selected_stage1_epochs") != selection["stage1_epochs"] or receipt.get(
        "selected_stage2_epochs"
    ) != selection["stage2_epochs"]:
        raise ValueError("Producer receipt epoch selection differs from its artifact.")
    if job.phase == "held_participant_evaluation":
        exact_epochs = job.job_spec["held_selected_epochs"]
        if {
            "stage1_epochs": selection["stage1_epochs"],
            "stage2_epochs": selection["stage2_epochs"],
        } != exact_epochs or selection["stage1_history"] or selection["stage2_history"]:
            raise ValueError("Held refit did not use only its source-gate epoch counts.")
    else:
        _validate_source_selection_history(selection, recipe=recipe)

    runtime = receipt.get("runtime")
    runtime_exact = {
        "device": "cuda:0",
        "visible_device_count": 1,
        "device_name": "NVIDIA GeForce RTX 4090",
        "cublas_workspace_config": ":4096:8",
        "stage1_amp_training": True,
        "stage2_amp_training": False,
        "amp_evaluation": False,
        "deterministic_algorithms": True,
        "cudnn_benchmark": False,
        "cudnn_deterministic": True,
        "cuda_matmul_allow_tf32": False,
        "cudnn_allow_tf32": False,
        "float32_matmul_precision": "highest",
    }
    if not isinstance(runtime, Mapping) or set(runtime) != {
        *runtime_exact,
        "torch_version",
        "safetensors_version",
        "torch_cuda_version",
        "cudnn_version",
    } or runtime.get("safetensors_version") != "0.8.0" or any(
        runtime.get(key) != value for key, value in runtime_exact.items()
    ):
        raise ValueError("Candidate runtime differs from the frozen CUDA contract.")

    frame = pd.read_parquet(root / "probabilities.parquet")
    spec = metadata_calibration_bundle_spec_for_phase(phase=job.phase)  # type: ignore[arg-type]
    probability_columns = {f"prob_{label:03d}" for label in range(spec.n_classes)}
    expected_columns = {
        *_CANDIDATE_FIXED_COLUMNS,
        *job.bindings.as_dict(),
        *probability_columns,
    }
    if set(frame) != expected_columns:
        raise ValueError("Candidate probability artifact has an invalid exact schema.")
    expected_query, expected_support = _expected_group_tables(job)
    cells = {(str(cell["role"]), str(cell["context"])) for cell in job.job_spec["cells"]}
    if set(frame["candidate_id"].astype(str)) != {"metadata-calibration-efficiency-v1"} or set(
        frame["phase"].astype(str)
    ) != {job.phase} or set(frame["checkpoint_group"].astype(str)) != {
        str(job.job_spec["checkpoint_group"])
    } or not is_integer_dtype(frame["seed"].dtype) or not is_integer_dtype(
        frame["budget"].dtype
    ) or set(frame["seed"]) != {
        int(job.job_spec["seed"])
    } or set(frame["budget"]) != set(
        job.job_spec["budgets"]
    ) or set(frame[["role", "context"]].itertuples(index=False, name=None)) != cells:
        raise ValueError("Candidate probability grid differs from the exact manifest job.")
    for key, value in job.bindings.as_dict().items():
        if set(frame[key].astype(str)) != {value}:
            raise ValueError(f"Candidate probability grid has a wrong {key} binding.")
    if set(frame["checkpoint_sha256"].astype(str)) != {
        str(receipt["checkpoint_sha256"])
    }:
        raise ValueError("Candidate probability grid uses an unbound checkpoint.")
    expected_rows = len(expected_query) * len(cells) * len(job.job_spec["budgets"])
    key = ["role", "context", "budget", "query_token"]
    if len(frame) != expected_rows or frame.duplicated(key).any():
        raise ValueError("Candidate probability artifact is not its complete atomic job grid.")
    expected_atomic_keys = {
        (role, context, int(budget), str(token))
        for role, context in cells
        for budget in job.job_spec["budgets"]
        for token in expected_query["query_token"]
    }
    observed_atomic_keys = set(frame[key].itertuples(index=False, name=None))
    if observed_atomic_keys != expected_atomic_keys:
        raise ValueError("Candidate artifact does not contain the exact query-cell cross product.")
    _validate_query_and_support_bindings(frame, expected_query, expected_support)
    probability = frame.loc[:, sorted(probability_columns)].to_numpy(dtype=np.float64)
    if not np.isfinite(probability).all() or (probability < 0).any() or (
        probability > 1
    ).any() or not np.allclose(probability.sum(axis=1), 1.0, rtol=0.0, atol=1e-6):
        raise ValueError("Candidate artifact contains an invalid probability vector.")
    expected_cell_hash = cell_execution_contract_sha256(
        phase=job.phase  # type: ignore[arg-type]
    )
    if set(frame["cell_execution_contract_sha256"].astype(str)) != {
        expected_cell_hash
    }:
        raise ValueError("Candidate artifact differs from the frozen cell dispatch contract.")
    if not frame["intervention_mapping_sha256"].astype(str).str.fullmatch(
        r"[0-9a-f]{64}"
    ).all() or not frame["resolved_context_usage_sha256"].astype(str).str.fullmatch(
        r"[0-9a-f]{64}"
    ).all():
        raise ValueError("Candidate intervention binding is not SHA-256.")
    if not frame.groupby(["role", "context"])[
        "intervention_mapping_sha256"
    ].nunique().eq(1).all() or not frame.groupby(["role", "context", "budget"])[
        "resolved_context_usage_sha256"
    ].nunique().eq(1).all():
        raise ValueError("Candidate intervention binding changes within one job cell.")
    fallback_key = ["budget", "query_token"]
    aq = frame.loc[
        frame["role"].eq("A_Q") & frame["context"].eq("exact_off"),
        [*fallback_key, *sorted(probability_columns)],
    ]
    missing = frame.loc[
        frame["role"].eq("A_QM") & frame["context"].eq("all_missing"),
        [*fallback_key, *sorted(probability_columns)],
    ]
    paired = aq.merge(
        missing,
        on=fallback_key,
        suffixes=("_aq", "_missing"),
        validate="one_to_one",
    )
    for column in sorted(probability_columns):
        if not np.array_equal(
            paired[f"{column}_aq"].to_numpy(),
            paired[f"{column}_missing"].to_numpy(),
        ):
            raise ValueError("Candidate all-missing fallback differs bitwise from A_Q.")
    if type(receipt.get("rows")) is not int or receipt["rows"] != len(frame):
        raise ValueError("Candidate producer receipt row count is invalid.")


def _validate_baseline_artifacts(
    job: AuthorizedMetadataCalibrationJob,
    root: Path,
    receipt: Mapping[str, Any],
) -> None:
    adapter_id = str(job.job_spec["adapter_id"])
    if receipt.get("adapter_id") != adapter_id or receipt.get(
        "deterministic_seed_column_present"
    ) is not False:
        raise ValueError("Baseline producer receipt differs from its deterministic job.")
    execution = baseline_execution_binding(adapter_id)
    frame = pd.read_parquet(root / "scores.parquet")
    score_columns = {f"score_{label:03d}" for label in range(12)}
    expected_columns = {
        *_BASELINE_FIXED_COLUMNS,
        *job.bindings.as_dict(),
        *score_columns,
    }
    if set(frame) != expected_columns or "seed" in frame:
        raise ValueError("Baseline score artifact has an invalid exact schema.")
    expected_query, expected_support = _expected_group_tables(job)
    if set(frame["candidate_id"].astype(str)) != {"metadata-calibration-efficiency-v1"} or set(
        frame["phase"].astype(str)
    ) != {job.phase} or set(frame["checkpoint_group"].astype(str)) != {
        str(job.job_spec["checkpoint_group"])
    } or set(frame["adapter_id"].astype(str)) != {adapter_id} or not is_integer_dtype(
        frame["budget"].dtype
    ) or set(frame["budget"]) != set(job.job_spec["budgets"]):
        raise ValueError("Baseline score grid differs from the exact manifest job.")
    for key, value in job.bindings.as_dict().items():
        if set(frame[key].astype(str)) != {value}:
            raise ValueError(f"Baseline score grid has a wrong {key} binding.")
    exact_adapter = {
        "implementation_sha256": execution.implementation_sha256,
        "config_sha256": execution.config_sha256,
        "adapter_contract_sha256": execution.adapter_contract_sha256,
        "score_schema": execution.score_schema,
        "score_interpretation": execution.score_interpretation,
    }
    if any(set(frame[key].astype(str)) != {value} for key, value in exact_adapter.items()):
        raise ValueError("Baseline score grid differs from its canonical adapter binding.")
    expected_rows = len(expected_query) * len(job.job_spec["budgets"])
    if len(frame) != expected_rows or frame.duplicated(
        ["adapter_id", "budget", "query_token"]
    ).any():
        raise ValueError("Baseline score artifact is not its complete atomic job grid.")
    expected_atomic_keys = {
        (adapter_id, int(budget), str(token))
        for budget in job.job_spec["budgets"]
        for token in expected_query["query_token"]
    }
    observed_atomic_keys = set(
        frame[["adapter_id", "budget", "query_token"]].itertuples(
            index=False,
            name=None,
        )
    )
    if observed_atomic_keys != expected_atomic_keys:
        raise ValueError("Baseline artifact does not contain the exact query-budget grid.")
    _validate_query_and_support_bindings(frame, expected_query, expected_support)
    scores = frame.loc[:, sorted(score_columns)].to_numpy(dtype=np.float64)
    if not np.isfinite(scores).all():
        raise ValueError("Baseline score artifact contains a non-finite raw score.")
    if adapter_id == "chiang2021_LST_filterbank_eTRCA":
        exact_sources = exact_source_pool_identities(load_baseline_source_fit(job.group_root))
        expected_source = frame["electrode_type"].astype(str).map(exact_sources)
        if not frame["source_pool_identity_sha256"].astype(str).eq(expected_source).all():
            raise ValueError("LST score artifact differs from its exact source sample pools.")
    elif set(frame["source_pool_identity_sha256"].astype(str)) != {
        canonical_identity_sha256([])
    }:
        raise ValueError("A target-only baseline unexpectedly binds labeled source EEG.")
    if type(receipt.get("rows")) is not int or receipt["rows"] != len(frame):
        raise ValueError("Baseline producer receipt row count is invalid.")


def _validate_source_selection_history(
    selection: Mapping[str, Any],
    *,
    recipe: Mapping[str, Any],
) -> None:
    stage_specs = (
        (
            "stage1_epochs",
            "stage1_history",
            "stage_1_common_Q",
            {"epoch", "train_loss", "validation_eauc", "validation_nll"},
            ("validation_eauc", "validation_nll"),
        ),
        (
            "stage2_epochs",
            "stage2_history",
            "stage_2_bounded_M",
            {
                "epoch",
                "train_loss",
                "validation_increment_eauc",
                "validation_correct_minus_shuffle_eauc",
            },
            (
                "validation_increment_eauc",
                "validation_correct_minus_shuffle_eauc",
            ),
        ),
    )
    for (
        epoch_field,
        history_field,
        recipe_field,
        exact_fields,
        selection_metrics,
    ) in stage_specs:
        selected_epoch = selection[epoch_field]
        stage = recipe[recipe_field]
        if not int(stage["minimum_epochs"]) <= selected_epoch <= int(
            stage["maximum_epochs"]
        ):
            raise ValueError("Source selected epoch lies outside the frozen recipe range.")
        history = selection[history_field]
        if not history or len(history) > int(stage["maximum_epochs"]):
            raise ValueError("Source epoch-selection history has an invalid length.")
        for expected_epoch, row in enumerate(history, start=1):
            if not isinstance(row, Mapping) or set(row) != exact_fields or type(
                row.get("epoch")
            ) is not int or row["epoch"] != expected_epoch:
                raise ValueError("Source epoch-selection history has a wrong exact schema.")
            metrics = [value for key, value in row.items() if key != "epoch"]
            if any(type(value) not in {int, float} for value in metrics) or not np.isfinite(
                np.asarray(metrics, dtype=np.float64)
            ).all():
                raise ValueError("Source epoch-selection history has a non-finite metric.")
        recomputed_epoch, expected_history_length = _recompute_selected_epoch(
            history,
            minimum_epochs=int(stage["minimum_epochs"]),
            maximum_epochs=int(stage["maximum_epochs"]),
            early_stop_patience=int(stage["early_stop_patience"]),
            selection_metrics=selection_metrics,
            minimize_second_metric=history_field == "stage1_history",
        )
        if len(history) != expected_history_length:
            raise ValueError("Source epoch-selection history ended before or after the frozen stop.")
        if selected_epoch != recomputed_epoch:
            raise ValueError("Source selected epoch differs from the frozen lexicographic optimum.")


def _recompute_selected_epoch(
    history: list[Mapping[str, Any]],
    *,
    minimum_epochs: int,
    maximum_epochs: int,
    early_stop_patience: int,
    selection_metrics: tuple[str, str],
    minimize_second_metric: bool,
) -> tuple[int, int]:
    """Independently replay the frozen eligible-epoch and early-stop rule."""

    best_key: tuple[float, float] | None = None
    best_epoch = 0
    patience = 0
    expected_history_length = maximum_epochs
    for row in history:
        epoch = int(row["epoch"])
        second = float(row[selection_metrics[1]])
        key = (
            float(row[selection_metrics[0]]),
            -second if minimize_second_metric else second,
        )
        if epoch >= minimum_epochs and _strictly_better_epoch_key(key, best_key):
            best_key = key
            best_epoch = epoch
            patience = 0
        else:
            patience += 1
        if epoch >= minimum_epochs and patience >= early_stop_patience:
            expected_history_length = epoch
            break
    if best_key is None or best_epoch < minimum_epochs:
        raise ValueError("Source history contains no checkpoint-eligible optimum.")
    return best_epoch, expected_history_length


def _strictly_better_epoch_key(
    candidate: tuple[float, float],
    incumbent: tuple[float, float] | None,
) -> bool:
    if incumbent is None:
        return True
    for left, right in zip(candidate, incumbent):
        if left > right + 1e-12:
            return True
        if left < right - 1e-12:
            return False
    return False


def _expected_group_tables(
    job: AuthorizedMetadataCalibrationJob,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    expected_query = _read_jsonl(
        job.group_root / "target_query/expected_unlabeled.jsonl"
    )
    if (
        job.job_spec.get("producer_kind") == "baseline"
        and job.job_spec.get("adapter_id") == "strict_FBCCA"
    ):
        expected_support = expected_query.loc[
            :, ["subject_id", "electrode_type", "checkpoint_group"]
        ].astype(str).drop_duplicates()
        expected_support["budget"] = 0
        expected_support["support_identity_sha256"] = _json_sha256([])
        return expected_query, expected_support
    support_rows = _read_jsonl(
        job.group_root / "target_support/support_labels.jsonl"
    )
    spec = metadata_calibration_bundle_spec_for_phase(phase=job.phase)  # type: ignore[arg-type]
    expected_support, _ = validate_and_summarize_support_rows(
        support_rows,
        expected_query_unlabeled=expected_query,
        spec=spec,
    )
    return expected_query, expected_support


def _validate_query_and_support_bindings(
    frame: pd.DataFrame,
    expected_query: pd.DataFrame,
    expected_support: pd.DataFrame,
) -> None:
    query = expected_query.loc[
        :,
        [
            "query_token",
            "subject_id",
            "electrode_type",
            "checkpoint_group",
            "query_identity_sha256",
        ],
    ]
    merged = frame.merge(
        query,
        on="query_token",
        how="left",
        suffixes=("", "_expected"),
        validate="many_to_one",
        indicator=True,
    )
    if not merged["_merge"].eq("both").all():
        raise ValueError("Job output contains an unsealed query token.")
    for column in (
        "subject_id",
        "electrode_type",
        "checkpoint_group",
        "query_identity_sha256",
    ):
        if not merged[column].astype(str).eq(
            merged[f"{column}_expected"].astype(str)
        ).all():
            raise ValueError(f"Job output has a wrong sealed {column} binding.")
    merged = merged.merge(
        expected_support,
        on=["subject_id", "electrode_type", "checkpoint_group", "budget"],
        how="left",
        suffixes=("", "_expected_support"),
        validate="many_to_one",
    )
    if not merged["support_identity_sha256"].astype(str).eq(
        merged["support_identity_sha256_expected_support"].astype(str)
    ).all():
        raise ValueError("Job output has a wrong sealed support identity.")


def _validate_output_file_records(
    job: AuthorizedMetadataCalibrationJob,
    root: Path,
    receipt: Mapping[str, Any],
) -> None:
    records = receipt.get("output_files")
    if not isinstance(records, list):
        raise TypeError("Producer receipt output_files must be a list.")
    expected_names = {
        PurePosixPath(path).name
        for path, kind in job.required_artifacts.items()
        if kind != "producer_receipt"
    }
    if len(records) != len(expected_names) or {
        str(record.get("name")) for record in records if isinstance(record, Mapping)
    } != expected_names:
        raise ValueError("Producer receipt output allowlist differs from the manifest.")
    for record in records:
        if not isinstance(record, Mapping) or set(record) != {
            "name",
            "size_bytes",
            "file_sha256",
        }:
            raise ValueError("Producer output-file record has an invalid exact schema.")
        path = root / str(record["name"])
        if type(record.get("size_bytes")) is not int or (
            path.stat().st_size != record["size_bytes"]
        ) or _sha256_file(path) != record[
            "file_sha256"
        ]:
            raise ValueError("Producer output-file receipt hash or size is invalid.")


def _exact_required_artifacts(job: Mapping[str, Any]) -> dict[str, str]:
    records = job.get("required_artifacts")
    if not isinstance(records, list) or not records:
        raise ValueError("Manifest job has no exact required-artifact allowlist.")
    output: dict[str, str] = {}
    for record in records:
        if not isinstance(record, Mapping) or set(record) != {
            "relative_path",
            "artifact_kind",
        }:
            raise ValueError("Manifest required artifact has an invalid exact schema.")
        relative = str(record["relative_path"])
        pure = PurePosixPath(relative)
        if pure.is_absolute() or ".." in pure.parts or "." in pure.parts:
            raise ValueError("Manifest required artifact path is not canonical relative POSIX.")
        if relative in output:
            raise ValueError("Manifest job repeats a required artifact path.")
        output[relative] = str(record["artifact_kind"])
    if str(job["required_output"]) not in output or str(job["required_receipt"]) not in output:
        raise ValueError("Manifest required output/receipt is missing from its artifact allowlist.")
    return output


def _validate_exact_job_tree(
    root: Path,
    *,
    expected_names: set[str],
    require_locked_modes: bool,
) -> None:
    _require_real_directory(root)
    actual: set[str] = set()
    for path in root.iterdir():
        observed = path.lstat()
        if not stat.S_ISREG(observed.st_mode) or observed.st_nlink != 1:
            raise ValueError("Job output tree may contain only non-hard-linked regular files.")
        if require_locked_modes and stat.S_IMODE(observed.st_mode) != 0o400:
            raise ValueError("Published job files must be read-only mode 0400.")
        actual.add(path.name)
    if actual != expected_names:
        raise ValueError("Job output tree has missing or undeclared artifacts.")
    if require_locked_modes and stat.S_IMODE(root.lstat().st_mode) != 0o700:
        raise ValueError("Published job directory must use private mode 0700.")


def _make_private_parents(root: Path, leaf_parent: Path) -> None:
    root = root.absolute()
    leaf_parent = leaf_parent.absolute()
    _require_beneath(leaf_parent, root)
    current = Path(root.anchor)
    for part in root.parts[1:]:
        current /= part
        created = False
        try:
            current.mkdir(mode=0o700)
            created = True
        except FileExistsError:
            pass
        _require_real_directory(current)
        if created:
            os.chmod(current, 0o700)
            _fsync_directory(current)
            _fsync_directory(current.parent)
    relative = leaf_parent.relative_to(root)
    current = root
    for part in relative.parts:
        current /= part
        created = False
        try:
            current.mkdir(mode=0o700)
            created = True
        except FileExistsError:
            pass
        _require_real_directory(current)
        os.chmod(current, 0o700)
        _fsync_directory(current)
        if created:
            _fsync_directory(current.parent)
    os.chmod(root, 0o700)
    _fsync_directory(root)
    _fsync_directory(root.parent)


def _lock_down_and_fsync_tree(root: Path) -> None:
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


def publish_tree_noreplace(staging: Path, destination: Path) -> None:
    staging = staging.absolute()
    destination = destination.absolute()
    _require_real_directory(staging)
    _require_real_directory(staging.parent)
    _require_real_directory(destination.parent)
    if staging.parent.lstat().st_dev != destination.parent.lstat().st_dev:
        raise OSError(errno.EXDEV, "Atomic tree publication requires one filesystem.")
    if os.path.lexists(destination):
        raise FileExistsError("Atomic tree publication never replaces a destination.")
    libc = ctypes.CDLL(None, use_errno=True)
    renameat2 = getattr(libc, "renameat2", None)
    if renameat2 is None:
        raise RuntimeError("Atomic no-replace publication requires Linux renameat2.")
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
        os.fsencode(staging),
        -100,
        os.fsencode(destination),
        _RENAME_NOREPLACE,
    )
    if result != 0:
        error = ctypes.get_errno()
        if error == errno.EEXIST:
            raise FileExistsError("Canonical job destination appeared during publication.")
        raise OSError(error, os.strerror(error), str(destination))
    _fsync_directory(destination.parent)
    if staging.parent != destination.parent:
        _fsync_directory(staging.parent)


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _read_regular_json(path: str | Path) -> dict[str, Any]:
    target = Path(path).absolute()
    observed = target.lstat()
    if not stat.S_ISREG(observed.st_mode) or observed.st_nlink != 1:
        raise ValueError(f"Expected one non-hard-linked regular JSON file: {target}.")
    descriptor = os.open(target, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    try:
        current = os.fstat(descriptor)
        if (current.st_dev, current.st_ino) != (observed.st_dev, observed.st_ino):
            raise ValueError("JSON artifact changed while being opened.")
        chunks: list[bytes] = []
        while True:
            chunk = os.read(descriptor, 1 << 20)
            if not chunk:
                break
            chunks.append(chunk)
    finally:
        os.close(descriptor)
    value = json.loads(b"".join(chunks))
    if not isinstance(value, dict):
        raise TypeError("Governed JSON artifact must contain one object.")
    return value


def _read_jsonl(path: Path) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            value = json.loads(line)
            if not isinstance(value, dict):
                raise TypeError("Governed JSONL row must be one object.")
            rows.append(value)
    if not rows:
        raise ValueError("Governed JSONL artifact must be nonempty.")
    return pd.DataFrame(rows)


def _tree_sha256(root: Path) -> str:
    records = []
    for path in sorted(root.rglob("*")):
        observed = path.lstat()
        if stat.S_ISDIR(observed.st_mode):
            continue
        if not stat.S_ISREG(observed.st_mode) or observed.st_nlink != 1:
            raise ValueError("Input checkpoint group contains a non-regular artifact.")
        records.append(
            {
                "path": path.relative_to(root).as_posix(),
                "file_sha256": _sha256_file(path),
            }
        )
    return _json_sha256({"files": records})


def _runner_projection_tree_sha256(
    root: Path,
    relative_paths: tuple[str, ...],
    *,
    require_exact: bool = False,
) -> str:
    expected = set(relative_paths)
    if require_exact:
        observed = {
            path.relative_to(root).as_posix()
            for path in root.rglob("*")
            if path.is_file()
        }
        if observed != expected:
            raise PermissionError(
                "Runner projection contains missing or finalizer-only artifacts."
            )
    records = []
    for relative in relative_paths:
        path = root / relative
        observed = path.lstat()
        if (
            not stat.S_ISREG(observed.st_mode)
            or stat.S_ISLNK(observed.st_mode)
            or observed.st_nlink != 1
        ):
            raise PermissionError("Runner projection contains a non-regular artifact.")
        records.append({"path": relative, "file_sha256": _sha256_file(path)})
    return _json_sha256({"files": records})


def runner_input_relative_paths(job_spec: Mapping[str, Any]) -> tuple[str, ...]:
    """Return the exact runner-visible files needed by one signed producer job."""

    kind = str(job_spec.get("producer_kind"))
    if kind == "candidate":
        return _CANDIDATE_RUNNER_INPUTS
    if kind != "baseline":
        raise ValueError("Cannot derive an input projection for an unknown producer kind.")
    adapter = str(job_spec.get("adapter_id"))
    paths = list(_BASELINE_QUERY_INPUTS)
    if adapter != "strict_FBCCA":
        paths.extend(_BASELINE_SUPPORT_INPUTS)
    if adapter == "chiang2021_LST_filterbank_eTRCA":
        paths.extend(_BASELINE_SOURCE_INPUTS)
    return tuple(sorted(paths))


def _require_real_directory(path: Path) -> None:
    observed = path.lstat()
    if not stat.S_ISDIR(observed.st_mode) or stat.S_ISLNK(observed.st_mode):
        raise ValueError(f"Expected one real directory: {path}.")


def _require_beneath(path: Path, root: Path) -> None:
    try:
        path.absolute().relative_to(root.absolute())
    except ValueError as error:
        raise ValueError("Governed path escapes its authorized root.") from error


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


def _json_sha256(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode(
            "utf-8"
        )
    ).hexdigest()


def _is_sha256(value: object) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(
        character in "0123456789abcdef" for character in value
    )
