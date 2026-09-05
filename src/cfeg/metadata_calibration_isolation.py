from __future__ import annotations

import hashlib
import json
import os
import subprocess
from pathlib import Path
from typing import Any

from cfeg.metadata_calibration_authority import verify_signed_record

WORKER_ISOLATION_CONTRACT = {
    "schema": "cfeg.metadata-calibration-worker-isolation.v1",
    "mechanism": "bubblewrap_mount_pid_user_network_namespace",
    "repository": "read_only_clean_single_commit_source_tag_snapshot",
    "phase_authority": "current_signed_job_capability_only",
    "sealed_input": "runner_allowlist_projection_only_finalizer_ciphertexts_absent",
    "processed_asset_root": "absent",
    "run_private_keys": "absent",
    "execution_output": "fresh_per_job_spool_bind_at_signed_logical_path",
    "prior_same_phase_job_outputs_and_ignored_data": "absent",
    "source_prerequisite_phase": "absent_signed_job_capability_only",
    "linux_capabilities": "all_zero",
    "no_new_privileges": True,
    "host_pid_namespace": "absent",
    "inherited_secret_file_descriptors": "forbidden",
}
WORKER_ISOLATION_CONTRACT_SHA256 = hashlib.sha256(
    json.dumps(
        WORKER_ISOLATION_CONTRACT,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
).hexdigest()


def verify_bubblewrap_worker_isolation(
    *,
    job_capability_path: str | Path,
    trusted_signing_public_key_path: str | Path,
    repository_root: str | Path,
    expected_checkpoint_group: str,
    expected_job_id: str,
) -> tuple[dict[str, Any], Any]:
    """Fail closed unless this worker has the exact label-blind mount capability."""

    if os.environ.get("CFEG_WORKER_ISOLATION_CONTRACT_SHA256") != (
        WORKER_ISOLATION_CONTRACT_SHA256
    ):
        raise PermissionError("Worker lacks the frozen Bubblewrap isolation capability.")
    host_pid_text = os.environ.get("CFEG_LIFECYCLE_HOST_PID", "")
    if not host_pid_text.isdigit() or int(host_pid_text) <= 2:
        raise PermissionError("Worker lacks a valid hidden lifecycle host-PID assertion.")

    from cfeg.metadata_calibration_job import (
        resolve_projected_metadata_calibration_job,
        runner_input_relative_paths,
    )

    capability_path = Path(job_capability_path).absolute()
    public_key = Path(trusted_signing_public_key_path).absolute()
    repository = Path(repository_root).absolute()
    job = resolve_projected_metadata_calibration_job(
        job_capability_path=capability_path,
        trusted_signing_public_key_path=public_key,
        job_id=expected_job_id,
    )
    capability = _read_json(capability_path)
    runner_paths = runner_input_relative_paths(job.job_spec)
    if str(job.job_spec.get("checkpoint_group")) != expected_checkpoint_group:
        raise PermissionError("Worker group differs from its signed job capability.")
    selected_group = job.group_root
    input_root = selected_group.parent
    execution_root = job.execution_output_root
    input_decision = _read_json(input_root / "input_seal_decision.json")
    input_receipt = _read_json(input_root / "input_seal_receipt.json")
    decision_hash = verify_signed_record(
        input_decision,
        public_key_path=public_key,
        hash_field="decision_payload_sha256",
        expected_schema="cfeg.metadata-calibration-input-seal-decision.v1",
    )
    receipt_hash = verify_signed_record(
        input_receipt,
        public_key_path=public_key,
        hash_field="receipt_payload_sha256",
        expected_schema="cfeg.metadata-calibration-input-seal-receipt.v1",
    )
    processed_root = Path(str(capability.get("processed_asset_root"))).absolute()
    if (
        decision_hash
        != capability.get("input_seal_decision_signed_record_sha256")
        or receipt_hash
        != capability.get("input_seal_receipt_signed_record_sha256")
        or input_decision.get("processed_asset_root") != str(processed_root)
        or input_decision.get("output_root") != str(input_root)
        or input_receipt.get("sealed_root") != str(input_root)
        or input_receipt.get("input_seal_decision_signed_record_sha256")
        != decision_hash
        or expected_checkpoint_group not in input_receipt.get("checkpoint_groups", [])
    ):
        raise PermissionError("Worker input projection differs from its signed seal chain.")

    if os.path.lexists(processed_root):
        raise PermissionError("Worker namespace exposes the processed label-bearing asset root.")
    private_names = {
        "authority-signing-private.pem",
        "finalizer-label-private.pem",
    }
    if any(os.path.lexists(public_key.parent / name) for name in private_names):
        raise PermissionError("Worker namespace exposes a research private key.")
    for path in (
        repository,
        capability_path.parent,
        capability_path.parent.parent,
        capability_path,
        input_root,
        selected_group,
        input_root / "input_seal_decision.json",
        input_root / "input_seal_receipt.json",
        public_key,
        public_key.parent,
        *(selected_group / relative for relative in runner_paths),
    ):
        if not path.exists() or not _is_read_only_mount(path):
            raise PermissionError(f"Worker read-only bind is missing or writable: {path}.")
    if set(capability_path.parent.iterdir()) != {capability_path} or set(
        capability_path.parent.parent.iterdir()
    ) != {capability_path.parent}:
        raise PermissionError("Worker namespace exposes another phase authority record.")
    if set(public_key.parent.iterdir()) != {public_key}:
        raise PermissionError("Worker namespace exposes another run key.")
    checkpoint_groups = input_receipt["checkpoint_groups"]
    if any(
        os.path.lexists(input_root / str(group))
        for group in checkpoint_groups
        if str(group) != expected_checkpoint_group
    ):
        raise PermissionError("Worker namespace exposes another checkpoint group.")
    if set(input_root.iterdir()) != {
        input_root / "input_seal_decision.json",
        input_root / "input_seal_receipt.json",
        selected_group,
    }:
        raise PermissionError("Worker input root exposes a non-projected artifact.")
    for name in (
        "query_labels.jsonl.enc",
        "query_analysis_covariates.jsonl.enc",
    ):
        if os.path.lexists(selected_group / "target_query" / name):
            raise PermissionError("Worker namespace exposes finalizer-only ciphertext.")
    expected_projection_directories = {
        selected_group / Path(relative).parent
        for relative in runner_paths
        if Path(relative).parent != Path(".")
    }
    observed_projection_directories = {
        path for path in selected_group.rglob("*") if path.is_dir()
    }
    if observed_projection_directories != expected_projection_directories:
        raise PermissionError("Worker projection contains an undeclared directory.")
    _validate_clean_worker_code_snapshot(
        repository, {"source_commit": job.source_commit}
    )
    if not execution_root.is_dir() or not os.path.ismount(execution_root) or (
        _is_read_only_mount(execution_root)
    ):
        raise PermissionError("Worker output is not one fresh writable mountpoint.")
    if any(execution_root.iterdir()):
        raise PermissionError("Worker execution spool was not empty on entry.")
    phase = job.phase
    source_prerequisite = execution_root.parent.parent / "source"
    if phase not in {"source_development", "held_participant_evaluation"}:
        raise PermissionError("Worker isolation manifest has an invalid phase.")
    if phase == "held_participant_evaluation" and os.path.lexists(source_prerequisite):
        raise PermissionError("Worker namespace exposes the prior source phase.")

    status = _proc_status()
    if status.get("NoNewPrivs") != "1":
        raise PermissionError("Worker namespace does not enforce no-new-privileges.")
    capability_fields = ("CapInh", "CapPrm", "CapEff", "CapBnd", "CapAmb")
    if any(int(status.get(field, "1"), 16) != 0 for field in capability_fields):
        raise PermissionError("Worker namespace retains a Linux capability.")
    if os.path.exists(f"/proc/{host_pid_text}"):
        raise PermissionError("Worker can observe the lifecycle process in its PID namespace.")

    forbidden_prefixes = (processed_root, public_key.parent)
    for fd_path in Path("/proc/self/fd").iterdir():
        try:
            resolved = Path(os.readlink(fd_path)).absolute()
        except (FileNotFoundError, OSError):
            continue
        if resolved == public_key:
            continue
        if any(_is_relative_to(resolved, prefix) for prefix in forbidden_prefixes):
            raise PermissionError("Worker inherited a file descriptor into a forbidden capability.")

    mountinfo = Path("/proc/self/mountinfo").read_text(encoding="utf-8")
    if str(processed_root) in mountinfo or any(name in mountinfo for name in private_names):
        raise PermissionError("Worker mount table exposes a forbidden data or key capability.")
    return (
        {
            "schema": "cfeg.metadata-calibration-worker-isolation-attestation.v1",
            "isolation_contract_sha256": WORKER_ISOLATION_CONTRACT_SHA256,
            "processed_asset_root_absent": True,
            "private_keys_absent": True,
            "input_and_code_read_only": True,
            "prior_same_phase_outputs_and_ignored_data_absent": True,
            "other_checkpoint_groups_absent": True,
            "finalizer_only_ciphertexts_absent": True,
            "only_current_job_capability_visible": True,
            "source_prerequisite_phase_access_policy_exact": True,
            "fresh_execution_spool_mounted": True,
            "execution_spool_empty_on_entry": True,
            "no_new_privileges": True,
            "linux_capabilities_all_zero": True,
            "host_pid_namespace_absent": True,
            "secret_file_descriptors_absent": True,
        },
        job,
    )


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"Expected one JSON object at {path}.")
    return value


def _is_read_only_mount(path: Path) -> bool:
    return bool(os.statvfs(path).f_flag & getattr(os, "ST_RDONLY", 1))


def _validate_clean_worker_code_snapshot(
    repository: Path, manifest: dict[str, Any]
) -> None:
    forbidden = (
        repository / ".local",
        repository / ".pytest-temp",
        repository / ".pytest_cache",
        repository / ".ruff_cache",
        repository / ".venv-accidental-20260905-1826",
    )
    if any(os.path.lexists(path) for path in forbidden):
        raise PermissionError("Worker code snapshot exposes a local cache or ignored tree.")
    allowed_sensitive_files = {
        "checkpoints/.gitkeep",
        "data/README.md",
        "data/manifests/.gitkeep",
        "data/processed/.gitkeep",
        "data/raw/.gitkeep",
        "outputs/.gitkeep",
    }
    observed_sensitive = {
        path.relative_to(repository).as_posix()
        for top in ("checkpoints", "data", "outputs")
        for path in (repository / top).rglob("*")
        if path.is_file()
    }
    if observed_sensitive != allowed_sensitive_files:
        raise PermissionError("Worker code snapshot contains data, checkpoints, or prior outputs.")
    completed = subprocess.run(
        ["/usr/bin/git", "-C", str(repository), "rev-list", "--all", "--count"],
        check=True,
        capture_output=True,
        text=True,
        env={"PATH": "/usr/bin:/bin", "LANG": "C.UTF-8"},
    )
    head = subprocess.run(
        ["/usr/bin/git", "-C", str(repository), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
        env={"PATH": "/usr/bin:/bin", "LANG": "C.UTF-8"},
    )
    if completed.stdout.strip() != "1" or head.stdout.strip() != manifest.get(
        "source_commit"
    ):
        raise PermissionError("Worker code snapshot is not the exact single frozen commit.")


def _proc_status() -> dict[str, str]:
    output: dict[str, str] = {}
    for line in Path("/proc/self/status").read_text(encoding="utf-8").splitlines():
        if ":" in line:
            key, value = line.split(":", 1)
            parts = value.strip().split()
            output[key] = parts[0] if parts else ""
    return output


def _is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
    except ValueError:
        return False
    return True
