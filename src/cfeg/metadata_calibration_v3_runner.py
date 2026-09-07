"""Restartable, fail-closed operator for the frozen V3 synthetic workflow.

The command surface deliberately contains no scientific, human-EEG, network,
seed, path, overwrite, or test-command controls.  Durable state is inferred
only from the fixed artifact paths owned by the governance module.  Every
existing artifact is reopened through its authoritative validator before the
next write-once phase may run.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import stat
import subprocess
import sys
import sysconfig
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from enum import Enum
from pathlib import Path, PurePosixPath
from types import ModuleType
from typing import Any, Protocol, TextIO

V3_SOURCE_REPOSITORY = Path("/home/whwovy/califreeEEG")
V3_PYTHON_EXECUTABLE = V3_SOURCE_REPOSITORY / ".venv/bin/python"
V3_SITE_PACKAGES = V3_SOURCE_REPOSITORY / ".venv/lib/python3.10/site-packages"
V3_REPOSITORY_SRC = V3_SOURCE_REPOSITORY / "src"
V3_RUNNER_PATH = V3_REPOSITORY_SRC / "cfeg/metadata_calibration_v3_runner.py"
V3_GIT_EXECUTABLE = Path("/usr/bin/git")
V3_DEVELOPMENT_BUNDLE_PATH = Path(
    "/home/whwovy/v3-artifacts/metadata-calibration-efficiency-v3/"
    "development-v5/development-bundle.json"
)

_EXPECTED_STANDARD_LIBRARY_PATH = (
    "/usr/lib/python310.zip",
    "/usr/lib/python3.10",
    "/usr/lib/python3.10/lib-dynload",
)
EXPECTED_RUNNER_SYS_PATH = (
    *_EXPECTED_STANDARD_LIBRARY_PATH,
    os.fspath(V3_REPOSITORY_SRC),
    os.fspath(V3_SITE_PACKAGES),
)

_DEVELOPMENT_BUNDLE_SCHEMA = "cfeg.metadata-calibration-efficiency-v3.development-bundle.v5"
_DEVELOPMENT_BUNDLE_FIELDS = frozenset(
    {
        "artifact_schema_inventory_sha256",
        "artifact_schemas",
        "candidate_id",
        "canonical_path",
        "clean_git_commit",
        "clean_git_tree",
        "created_at_UTC",
        "development_attempt_id",
        "development_start_receipt_path",
        "development_start_receipt_policy",
        "development_start_receipt_schema",
        "focused_test_argv",
        "focused_test_collected_tests",
        "focused_test_environment_sha256",
        "focused_test_error_tests",
        "focused_test_executable_file_sha256",
        "focused_test_failed_tests",
        "focused_test_junit_report_base64",
        "focused_test_junit_report_file_sha256",
        "focused_test_junit_report_path",
        "focused_test_passed_tests",
        "focused_test_process_id",
        "focused_test_scope",
        "focused_test_skipped_tests",
        "focused_test_stderr_sha256",
        "focused_test_stdout_sha256",
        "focused_test_working_directory",
        "focused_test_xfailed_tests",
        "focused_test_xpassed_tests",
        "focused_tests_passed",
        "human_EEG_outcome_authorized",
        "master_plan_file_sha256",
        "master_plan_path",
        "numerical_runtime_fingerprint_sha256",
        "numerical_runtime_inventory",
        "original_preoutcome_amendment_file_sha256",
        "original_preoutcome_amendment_path",
        "prior_recovery_amendment_file_sha256",
        "prior_recovery_amendment_path",
        "owner_authority_public_key_file_sha256",
        "owner_authority_public_key_path",
        "owner_authority_ssh_fingerprint",
        "payload_sha256",
        "preoutcome_amendment_file_sha256",
        "preoutcome_amendment_path",
        "protocol_revision",
        "retired_v1_artifact_inventory_schema",
        "retired_v1_artifact_inventory_sha256",
        "retired_v1_context_reference_file_sha256",
        "retired_v1_context_reference_path",
        "retired_v1_context_reference_payload_sha256",
        "retired_v1_context_reference_root_seed",
        "retired_v1_context_reference_schema",
        "retired_v1_continuation_authorized",
        "retired_v1_development_DGP_executed",
        "retired_v1_development_bundle_file_sha256",
        "retired_v1_development_bundle_path",
        "retired_v1_development_bundle_payload_sha256",
        "retired_v1_development_bundle_schema",
        "retired_v1_development_result_present",
        "retired_v1_development_root_seed",
        "retired_v1_development_seedsequence_created",
        "retired_v1_experimental_outcome_observed",
        "retired_v1_governed_network_accessed",
        "retired_v1_incident_error",
        "retired_v1_incident_message",
        "retired_v1_incident_output_sha256",
        "retired_v1_source_commit",
        "retired_v1_source_tree",
        "retired_v2_artifact_inventory_schema",
        "retired_v2_artifact_inventory_sha256",
        "retired_v2_context_reference_present",
        "retired_v2_context_reference_replay_count",
        "retired_v2_context_reference_root_seed",
        "retired_v2_continuation_authorized",
        "retired_v2_development_DGP_executed",
        "retired_v2_development_bundle_file_sha256",
        "retired_v2_development_bundle_path",
        "retired_v2_development_bundle_payload_sha256",
        "retired_v2_development_bundle_schema",
        "retired_v2_development_result_present",
        "retired_v2_development_rng_authority_issued",
        "retired_v2_development_root_seed",
        "retired_v2_development_seedsequence_created",
        "retired_v2_experimental_outcome_observed",
        "retired_v2_governed_network_accessed",
        "retired_v2_incident_error",
        "retired_v2_incident_message",
        "retired_v2_incident_output_sha256",
        "retired_v2_source_bundle_sha256",
        "retired_v2_source_commit",
        "retired_v2_source_tree",
        "schema",
        "scientific_contract_projection_sha256",
        "scientific_lockbox_authorized",
        "source_bundle_sha256",
        "synthetic_plan_file_sha256",
        "synthetic_plan_path",
        "tracked_source_file_inventory",
        "v2_deny_overlay_file_sha256",
        "v2_deny_overlay_path",
        "v2_terminal_audit_file_sha256",
        "v2_terminal_audit_path",
    }
) | frozenset(
    {
        "earlier_recovery_amendment_file_sha256",
        "earlier_recovery_amendment_path",
        "replacement_development_evidence_role",
        "replacement_development_root_seed",
        "replacement_seed_digest_split",
        "replacement_seed_outcome_used",
        "replacement_seed_preimage_schema",
        "replacement_seed_preimage_sha256",
        "replacement_seed_selected_word_index",
        "replacement_seed_selection_rule",
        "retired_scientific_contract_projection_sha256",
        "retired_v3_artifact_inventory_schema",
        "retired_v3_artifact_inventory_sha256",
        "retired_v3_canary_present",
        "retired_v3_complete_grid_gate_reports_computed_in_memory",
        "retired_v3_context_reference_file_sha256",
        "retired_v3_context_reference_path",
        "retired_v3_context_reference_payload_sha256",
        "retired_v3_context_reference_root_seed",
        "retired_v3_context_reference_schema",
        "retired_v3_continuation_authorized",
        "retired_v3_control_flow_inference_classification",
        "retired_v3_development_bundle_file_sha256",
        "retired_v3_development_bundle_path",
        "retired_v3_development_bundle_payload_sha256",
        "retired_v3_development_bundle_schema",
        "retired_v3_development_result_present",
        "retired_v3_development_rng_authority_issued_in_memory",
        "retired_v3_development_root_seed",
        "retired_v3_development_seed_consumed",
        "retired_v3_development_seedsequence_and_DGP_executed",
        "retired_v3_development_selection_computed_in_memory",
        "retired_v3_development_start_creator_process_id",
        "retired_v3_development_start_file_sha256",
        "retired_v3_development_start_path",
        "retired_v3_development_start_payload_sha256",
        "retired_v3_development_start_schema",
        "retired_v3_development_started_at_UTC",
        "retired_v3_experimental_outcome_observed",
        "retired_v3_governed_network_accessed",
        "retired_v3_human_EEG_outcome_accessed",
        "retired_v3_incident_boundary",
        "retired_v3_incident_error",
        "retired_v3_incident_message",
        "retired_v3_incident_output_sha256",
        "retired_v3_numerical_runtime_fingerprint_sha256",
        "retired_v3_selected_method_present",
        "retired_v3_source_bundle_sha256",
        "retired_v3_source_commit",
        "retired_v3_source_tree",
        "retired_v3_test_evidence_present",
        "development_result_endpoint_order_policy",
        "first_recovery_amendment_file_sha256",
        "first_recovery_amendment_path",
        "original_retired_scientific_contract_projection_sha256",
        "retired_v4_artifact_inventory_schema",
        "retired_v4_artifact_inventory_sha256",
        "retired_v4_canary_present",
        "retired_v4_complete_grid_gate_reports_computed_in_memory",
        "retired_v4_context_reference_file_sha256",
        "retired_v4_context_reference_path",
        "retired_v4_context_reference_payload_sha256",
        "retired_v4_context_reference_root_seed",
        "retired_v4_context_reference_schema",
        "retired_v4_continuation_authorized",
        "retired_v4_control_flow_inference_classification",
        "retired_v4_development_bundle_file_sha256",
        "retired_v4_development_bundle_path",
        "retired_v4_development_bundle_payload_sha256",
        "retired_v4_development_bundle_schema",
        "retired_v4_development_result_present",
        "retired_v4_development_rng_authority_issued_in_memory",
        "retired_v4_development_root_seed",
        "retired_v4_development_seed_consumed",
        "retired_v4_development_seedsequence_and_DGP_executed",
        "retired_v4_development_selection_computed_in_memory",
        "retired_v4_development_start_creator_process_id",
        "retired_v4_development_start_file_sha256",
        "retired_v4_development_start_path",
        "retired_v4_development_start_payload_sha256",
        "retired_v4_development_start_schema",
        "retired_v4_development_started_at_UTC",
        "retired_v4_experimental_outcome_observed",
        "retired_v4_governed_network_accessed",
        "retired_v4_human_EEG_outcome_accessed",
        "retired_v4_incident_boundary",
        "retired_v4_incident_error",
        "retired_v4_incident_message",
        "retired_v4_incident_output_sha256",
        "retired_v4_numerical_runtime_fingerprint_sha256",
        "retired_v4_selected_method_present",
        "retired_v4_source_bundle_sha256",
        "retired_v4_source_commit",
        "retired_v4_source_tree",
        "retired_v4_test_evidence_present",
    }
)
_PROTOCOL_REVISION = "V3.4"
_DEVELOPMENT_ATTEMPT_ID = "development-v5"
_RECOVERY_AMENDMENT_PATH = "configs/governance/metadata_calibration_v3_4_recovery_amendment.json"
_RECOVERY_AMENDMENT_SHA256 = "129c81f1a806e17d68ced5055d094faf22a9e0f017e834175394c5b40e79ee4e"
_PRIOR_RECOVERY_AMENDMENT_PATH = (
    "configs/governance/metadata_calibration_v3_3_recovery_amendment.json"
)
_PRIOR_RECOVERY_AMENDMENT_SHA256 = (
    "d00fbb2722444fd92235eae97c403c6fc0b9bdf0157d571c2c7717c8afa09c11"
)
_EARLIER_RECOVERY_AMENDMENT_PATH = (
    "configs/governance/metadata_calibration_v3_2_recovery_amendment.json"
)
_EARLIER_RECOVERY_AMENDMENT_SHA256 = (
    "1db2d118376975c1817e329f4203dbaf9929d4f2bcb30bfede990af26c2c1ceb"
)
_FIRST_RECOVERY_AMENDMENT_PATH = (
    "configs/governance/metadata_calibration_v3_1_recovery_amendment.json"
)
_FIRST_RECOVERY_AMENDMENT_SHA256 = (
    "deddca9286f07c6293925409d01b3df1313e89f2ce89688c4a25c14a8ec01238"
)
_ORIGINAL_AMENDMENT_PATH = "configs/governance/metadata_calibration_v3_preoutcome_amendment.json"
_ORIGINAL_AMENDMENT_SHA256 = "3238761a0a9a257032d6582286953a4d20981e5e646c9f1322c5c0357bd0c202"
_RETIRED_V1_COMMIT = "569394da6894a2efed37b9dde162cbe4fe60534d"
_RETIRED_V1_TREE = "fe472b6ae07d2a95d87d8ae5f3ec266a5347e8dd"
_RETIRED_V1_INVENTORY_SHA256 = "ece2fc0822fc33d952b926efacb076586d90a74e1bb5132b110789359837055b"
_RETIRED_V1_BUNDLE_PAYLOAD_SHA256 = (
    "959ce56a97e3ece034b52c44a993d393ef168bbb1bdbad9a72dee86cdf3430b3"
)
_RETIRED_V1_BUNDLE_FILE_SHA256 = "9f6027918110cfd877a1b170f3472d897d4b06f6fca7f0991b9f15f6768c4a1a"
_RETIRED_V1_CONTEXT_PAYLOAD_SHA256 = (
    "c475e9d0ce4f8e436b49c50585eaee37965bba7c052eefdee862bc50f70a9253"
)
_RETIRED_V1_CONTEXT_FILE_SHA256 = "7b78b3092f5824b6e247d97f4b9cd9f9ec22dbfa74a182e2f32c34fb1c8cf3c8"
_RETIRED_V1_INCIDENT_OUTPUT_SHA256 = (
    "8cdcfe3961979b2d691d4a241878abbf68aa743d8dd08ccbf01ffe1ec03a796a"
)
_RETIRED_V2_COMMIT = "82e26d60fd17464380697126bb64ad73b36708a2"
_RETIRED_V2_TREE = "f7f59bda48610aeaa5bd1b6ffd288a7b56a4bb85"
_RETIRED_V2_SOURCE_BUNDLE_SHA256 = (
    "a74d3a86f207520fd97c746809378146d75614fe287b1eaab96e50b15da3dbc0"
)
_RETIRED_V2_INVENTORY_SHA256 = "7ae6a99b7ec2033b2ce0cdfbfedc05851b77d9e5dfeb21b485ebe4215f18c865"
_RETIRED_V2_BUNDLE_PAYLOAD_SHA256 = (
    "ce9628ecc5a2b000aa1ce370bba540e6d0fa2d46d3ae068cbcc5b43f089ba312"
)
_RETIRED_V2_BUNDLE_FILE_SHA256 = "a616945040ea45a78c2eed58bf251f1f379f9a82a320ed4f39c26f5efb073cac"
_RETIRED_V2_INCIDENT_OUTPUT_SHA256 = (
    "5e21a934ee692d13e3717502442c412d58121547eb2d05ecaf3c1bd827abd04c"
)
_RETIRED_V3_COMMIT = "adab7acda840c5e6ed9a994d82c8098094284c6b"
_RETIRED_V3_TREE = "b06b43a130f599e0b85b4ee91856954f6061bea7"
_RETIRED_V3_SOURCE_BUNDLE_SHA256 = (
    "5e7ff5ad96c010d73870857caf6f3544e227e49099fbb0c1ad839649e30cd609"
)
_RETIRED_V3_RUNTIME_FINGERPRINT_SHA256 = (
    "d68d72b909b5eed289de51464c3b5a6340854ac189fecdc394fc27edaad9d633"
)
_RETIRED_V3_INVENTORY_SHA256 = "59502b324417c2be6c03aa5483532646fb8630af92de1218e56d862ce72cd6e7"
_RETIRED_V3_BUNDLE_PAYLOAD_SHA256 = (
    "b47c55453117c30af0507b10ef7e894b3721738d4670b7ea8afa40f861f9fa1f"
)
_RETIRED_V3_BUNDLE_FILE_SHA256 = "dee4068d2fdce5ec1031de17bf9e004f0a4629f88bfcc0f13c39211d4383d741"
_RETIRED_V3_START_PAYLOAD_SHA256 = (
    "5c26e0e95175fa374f80b9c7e5cd11af66c28d4a53f693dec31a0ff8f4ed6486"
)
_RETIRED_V3_START_FILE_SHA256 = "2183cc2506cbf4b7d6b62bf442c2a03a89fbbdaace5d66e5e73afb5869fd87ff"
_RETIRED_V4_COMMIT = "198158663efd1c4481b23dd3d012b9d854ada8d1"
_RETIRED_V4_TREE = "ac7882494955a7abb5046a4dd950bd98d6fdd6b1"
_RETIRED_V4_SOURCE_BUNDLE_SHA256 = (
    "c57c81a2ffae4511e4de6159d11ea5ef482e13b8fdebd916eef98bdf7235250c"
)
_RETIRED_V4_RUNTIME_FINGERPRINT_SHA256 = _RETIRED_V3_RUNTIME_FINGERPRINT_SHA256
_RETIRED_V4_INVENTORY_SHA256 = "511de6688fa315830ddf98ed5a78eefde4e0e4ba6f15977c347281403e908dfa"
_RETIRED_V4_BUNDLE_PAYLOAD_SHA256 = (
    "0fbb581d320a256c1722f6004810dbfb6f501f5eecf6e95c9264be9657f09e4c"
)
_RETIRED_V4_BUNDLE_FILE_SHA256 = "d257a2f4ea83bc355399ddafea5f7deca8e95bb9e4de7d4a5e3ccb99a1602dd9"
_RETIRED_V4_START_PAYLOAD_SHA256 = (
    "b33666e840f9a6669713b79b10f75dbf1100d939b2c96e25bce22d7fe60ec8b2"
)
_RETIRED_V4_START_FILE_SHA256 = "1bfafb3b265c3cd7c459a8c7d09be4275743588addb2680e0685a1501389c068"
_RETIRED_V4_INCIDENT_OUTPUT_SHA256 = (
    "5c6a3fa68152b88b3f32745f04afcc52f3c636dd709eb56a937470b45e7ddbcc"
)
_ORIGINAL_SCIENTIFIC_CONTRACT_PROJECTION_SHA256 = (
    "0702d01be1e055d3203a3c1b78777db6456b8d527e5525b6d468fb52522f8a79"
)
_RETIRED_SCIENTIFIC_CONTRACT_PROJECTION_SHA256 = (
    "5885f39908d8a33b130e0dc923abe560cc4587ebe7c58ed242647ce028f94712"
)
_SCIENTIFIC_CONTRACT_PROJECTION_SHA256 = (
    "670b61c767ff2a4b02c3a582aa8ac7e46b5632ada2806ffc109e2e79326d05ca"
)
_REPLACEMENT_SEED_PREIMAGE_SHA256 = (
    "11f503bdca78e8a7d0e853f9a7b484510dd749018c7735bce61ff7965b70b1cb"
)
_REPLACEMENT_DEVELOPMENT_ROOT_SEED = 301_269_949
_PYTHON_SITE_INVENTORY_SCHEMA = "cfeg.metadata-calibration-efficiency-v3.python-site-inventory.v1"
_CANDIDATE_ID = "metadata-calibration-efficiency-v3"
_SELECTED_METHOD_RELATIVE_PATH = (
    "configs/governance/metadata_calibration_v3_selected_method_freeze.json"
)
_PROTECTED_SOURCE_ROOTS = ("scripts", "src", "tests")
_ROOT_IMPORT_CONTROLS = frozenset(
    {
        ".pytest.ini",
        ".pytest.toml",
        ".pythonrc.py",
        "conftest.py",
        "pyproject.toml",
        "pytest.ini",
        "pytest.toml",
        "setup.cfg",
        "setup.py",
        "sitecustomize.py",
        "tox.ini",
        "usercustomize.py",
    }
)
_COMMANDS = ("status", "resume", "emit-selection")
_EXPECTED_GOVERNED_ENVIRONMENT = {
    "BLIS_NUM_THREADS": "1",
    "LANG": "C.UTF-8",
    "LC_ALL": "C.UTF-8",
    "MKL_DYNAMIC": "FALSE",
    "MKL_NUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
    "OMP_DYNAMIC": "FALSE",
    "OMP_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "PATH": "/usr/bin:/bin",
    "PYTHONNOUSERSITE": "1",
    "PYTHONDONTWRITEBYTECODE": "1",
    "PYTEST_DISABLE_PLUGIN_AUTOLOAD": "1",
    "VECLIB_MAXIMUM_THREADS": "1",
}
_SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")
_GIT_OBJECT_RE = re.compile(r"[0-9a-f]{40}\Z")
_RFC3339_SECONDS_RE = re.compile(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z\Z")
_STRICT_GIT_ENVIRONMENT = {
    "GIT_CONFIG_GLOBAL": "/dev/null",
    "GIT_CONFIG_NOSYSTEM": "1",
    "GIT_NO_REPLACE_OBJECTS": "1",
    "GIT_OPTIONAL_LOCKS": "0",
    "GIT_TERMINAL_PROMPT": "0",
    "LANG": "C.UTF-8",
    "LC_ALL": "C.UTF-8",
    "PATH": "/usr/bin:/bin",
}
_PREFLIGHT_ISSUER = object()


class RunnerError(RuntimeError):
    """The fixed runner state or transition is invalid."""


@dataclass(frozen=True, slots=True)
class _PreflightConfig:
    """Fixed production paths, with an explicit isolated-test construction seam."""

    repository: Path
    python_executable: Path
    site_packages: Path
    repository_src: Path
    runner_path: Path
    runner_repository_path: str
    bundle_path: Path
    standard_library_path: tuple[str, ...]
    protected_source_roots: tuple[str, ...]
    root_import_controls: frozenset[str]
    selected_method_relative_path: str
    require_direct_invocation: bool


@dataclass(frozen=True, slots=True)
class _TrackedFile:
    path: str
    kind: str
    file_sha256: str

    def record(self) -> dict[str, str]:
        return {
            "file_sha256": self.file_sha256,
            "kind": self.kind,
            "path": self.path,
        }


@dataclass(frozen=True, slots=True)
class _SourceFacts:
    commit: str
    tree: str
    tracked_files: tuple[_TrackedFile, ...]
    source_bundle_sha256: str
    git_executable_file_sha256: str
    git_version: str
    runner_file_sha256: str


@dataclass(frozen=True, slots=True)
class _PreflightFacts:
    config: _PreflightConfig
    command: str | None
    source: _SourceFacts
    python: Mapping[str, Any]
    python_site_inventory: Mapping[str, Any]
    bundle_payload: Mapping[str, Any] | None
    _issuer: object


_PRODUCTION_PREFLIGHT_CONFIG = _PreflightConfig(
    repository=V3_SOURCE_REPOSITORY,
    python_executable=V3_PYTHON_EXECUTABLE,
    site_packages=V3_SITE_PACKAGES,
    repository_src=V3_REPOSITORY_SRC,
    runner_path=V3_RUNNER_PATH,
    runner_repository_path="src/cfeg/metadata_calibration_v3_runner.py",
    bundle_path=V3_DEVELOPMENT_BUNDLE_PATH,
    standard_library_path=_EXPECTED_STANDARD_LIBRARY_PATH,
    protected_source_roots=_PROTECTED_SOURCE_ROOTS,
    root_import_controls=_ROOT_IMPORT_CONTROLS,
    selected_method_relative_path=_SELECTED_METHOD_RELATIVE_PATH,
    require_direct_invocation=True,
)


def _plain_json(value: Any) -> Any:
    if isinstance(value, Mapping):
        if any(not isinstance(key, str) for key in value):
            raise RunnerError("JSON object keys must be strings")
        return {key: _plain_json(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_plain_json(item) for item in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise RunnerError(f"value is not JSON-compatible: {type(value).__name__}")


def _canonical_json_bytes(value: Mapping[str, Any]) -> bytes:
    try:
        return json.dumps(
            _plain_json(value),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise RunnerError("value is not canonical JSON") from exc


def _detach_retired_v1_context_payload(
    governance: ModuleType,
    retired_v1: object,
) -> dict[str, Any]:
    """Cross the frozen-parser boundary without changing the exact context bytes."""

    frozen = governance.retired_v1_context_reference_payload(retired_v1)
    detached = _plain_json(frozen)
    if type(detached) is not dict:
        raise RunnerError("retired V1 context did not detach to an exact JSON object")
    exact_bytes = governance.retired_v1_context_reference_bytes(retired_v1)
    if _canonical_json_bytes(detached) + b"\n" != exact_bytes:
        raise RunnerError("detached retired V1 context differs from its immutable bytes")
    return detached


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _stat_identity(value: os.stat_result) -> tuple[int, ...]:
    return (
        value.st_dev,
        value.st_ino,
        value.st_mode,
        value.st_uid,
        value.st_nlink,
        value.st_size,
        value.st_mtime_ns,
        value.st_ctime_ns,
    )


def _require_safe_directory_stat(value: os.stat_result, label: str) -> None:
    mode = stat.S_IMODE(value.st_mode)
    if (
        not stat.S_ISDIR(value.st_mode)
        or value.st_uid != os.geteuid()
        or mode & stat.S_IWOTH
        or mode & 0o7000
    ):
        raise RunnerError(f"{label} is not a safe owner-controlled directory")


def _require_safe_regular_stat(
    value: os.stat_result,
    label: str,
    *,
    require_current_owner: bool,
    require_single_link: bool,
) -> None:
    mode = stat.S_IMODE(value.st_mode)
    if (
        not stat.S_ISREG(value.st_mode)
        or (require_current_owner and value.st_uid != os.geteuid())
        or value.st_nlink < 1
        or (require_single_link and value.st_nlink != 1)
        or not mode & stat.S_IRUSR
        or mode & stat.S_IWOTH
        or mode & 0o7000
    ):
        raise RunnerError(f"{label} is not a safe owner-controlled regular file")


def _path_parts(path: Path) -> tuple[str, ...]:
    raw = os.fspath(path)
    pure = PurePosixPath(raw)
    if not pure.is_absolute() or raw != pure.as_posix():
        raise RunnerError("preflight path is not exact canonical absolute POSIX text")
    parts = pure.parts
    if not parts or parts[0] != "/" or any(part in {"", ".", ".."} for part in parts[1:]):
        raise RunnerError("preflight path contains an unsafe component")
    return tuple(parts[1:])


def _open_absolute_directory(path: Path) -> int:
    if not hasattr(os, "O_DIRECTORY") or not hasattr(os, "O_NOFOLLOW"):
        raise RunnerError("directory-fd and no-follow primitives are required")
    descriptor = os.open("/", os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        for part in _path_parts(path):
            child = os.open(
                part,
                os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW,
                dir_fd=descriptor,
            )
            os.close(descriptor)
            descriptor = child
        return descriptor
    except Exception:
        os.close(descriptor)
        raise


def _hash_open_regular_file(
    descriptor: int,
    *,
    label: str,
    require_current_owner: bool = True,
    require_single_link: bool,
) -> tuple[str, os.stat_result]:
    before = os.fstat(descriptor)
    _require_safe_regular_stat(
        before,
        label,
        require_current_owner=require_current_owner,
        require_single_link=require_single_link,
    )
    digest = hashlib.sha256()
    while True:
        chunk = os.read(descriptor, 1 << 20)
        if not chunk:
            break
        digest.update(chunk)
    after = os.fstat(descriptor)
    if _stat_identity(before) != _stat_identity(after):
        raise RunnerError(f"{label} changed while it was hashed")
    return digest.hexdigest(), before


def _hash_absolute_regular_file(
    path: Path,
    *,
    label: str,
    require_current_owner: bool = True,
    require_single_link: bool,
) -> str:
    parts = _path_parts(path)
    if not parts:
        raise RunnerError(f"{label} cannot be filesystem root")
    parent = Path("/", *parts[:-1])
    directory = _open_absolute_directory(parent)
    descriptor: int | None = None
    try:
        descriptor = os.open(
            parts[-1],
            os.O_RDONLY | os.O_NOFOLLOW,
            dir_fd=directory,
        )
        digest, _ = _hash_open_regular_file(
            descriptor,
            label=label,
            require_current_owner=require_current_owner,
            require_single_link=require_single_link,
        )
        return digest
    finally:
        if descriptor is not None:
            os.close(descriptor)
        os.close(directory)


def _read_open_file(descriptor: int, *, maximum_bytes: int, label: str) -> bytes:
    chunks: list[bytes] = []
    size = 0
    while True:
        chunk = os.read(descriptor, min(1 << 20, maximum_bytes - size + 1))
        if not chunk:
            break
        chunks.append(chunk)
        size += len(chunk)
        if size > maximum_bytes:
            raise RunnerError(f"{label} exceeds its pre-import size limit")
    return b"".join(chunks)


def _read_optional_artifact(path: Path) -> bytes | None:
    parts = _path_parts(path)
    if not parts:
        raise RunnerError("bundle path cannot be filesystem root")
    try:
        directory = _open_absolute_directory(Path("/", *parts[:-1]))
    except FileNotFoundError:
        return None
    descriptor: int | None = None
    try:
        try:
            descriptor = os.open(
                parts[-1],
                os.O_RDONLY | os.O_NOFOLLOW,
                dir_fd=directory,
            )
        except FileNotFoundError:
            return None
        before = os.fstat(descriptor)
        if (
            not stat.S_ISREG(before.st_mode)
            or before.st_uid != os.geteuid()
            or before.st_nlink != 1
            or stat.S_IMODE(before.st_mode) != 0o400
        ):
            raise RunnerError("development bundle is not the exact immutable artifact type")
        data = _read_open_file(
            descriptor,
            maximum_bytes=64 * 1024 * 1024,
            label="development bundle",
        )
        if _stat_identity(before) != _stat_identity(os.fstat(descriptor)):
            raise RunnerError("development bundle changed while it was read")
        return data
    finally:
        if descriptor is not None:
            os.close(descriptor)
        os.close(directory)


def _run_git(config: _PreflightConfig, arguments: Sequence[str]) -> bytes:
    completed = subprocess.run(
        [
            os.fspath(V3_GIT_EXECUTABLE),
            "--no-optional-locks",
            "-c",
            "core.fsmonitor=false",
            "-c",
            "core.hooksPath=/dev/null",
            "-C",
            os.fspath(config.repository),
            *arguments,
        ],
        check=False,
        stdin=subprocess.DEVNULL,
        capture_output=True,
        env=dict(_STRICT_GIT_ENVIRONMENT),
    )
    if completed.returncode != 0 or completed.stderr:
        detail = completed.stderr.decode("utf-8", errors="replace").strip()
        raise RunnerError(f"strict Git preflight command failed: {detail}")
    return completed.stdout


def _require_git_object(value: str, label: str) -> str:
    if not _GIT_OBJECT_RE.fullmatch(value):
        raise RunnerError(f"{label} is not an exact SHA-1 Git object ID")
    return value


def _decode_git_path(raw: bytes) -> str:
    try:
        path = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise RunnerError("Git source path is not UTF-8") from exc
    pure = PurePosixPath(path)
    if (
        not path
        or "\x00" in path
        or pure.is_absolute()
        or pure.as_posix() != path
        or any(part in {"", ".", ".."} for part in pure.parts)
    ):
        raise RunnerError("Git source path is not canonical repository-relative text")
    return path


def _git_repository_precheck(config: _PreflightConfig) -> None:
    repository = config.repository
    repository_descriptor = _open_absolute_directory(repository)
    try:
        _require_safe_directory_stat(os.fstat(repository_descriptor), "repository root")
    finally:
        os.close(repository_descriptor)
    try:
        top = _run_git(config, ("rev-parse", "--show-toplevel")).decode("utf-8").strip()
        git_dir = _run_git(config, ("rev-parse", "--absolute-git-dir")).decode("utf-8").strip()
    except UnicodeDecodeError as exc:
        raise RunnerError("Git repository identity paths are not UTF-8") from exc
    if top != os.fspath(repository) or git_dir != os.fspath(repository / ".git"):
        raise RunnerError("repository is not the exact local non-worktree Git root")
    git_descriptor = _open_absolute_directory(repository / ".git")
    os.close(git_descriptor)
    try:
        os.lstat(repository / ".git/objects/info/alternates")
    except FileNotFoundError:
        pass
    else:
        raise RunnerError("Git alternate object stores are prohibited")
    if _run_git(config, ("for-each-ref", "--format=%(refname)", "refs/replace")):
        raise RunnerError("Git replacement objects are prohibited")
    raw_config = _run_git(config, ("config", "--local", "--null", "--list"))
    for raw_record in raw_config.split(b"\x00"):
        if not raw_record:
            continue
        raw_key, separator, raw_value = raw_record.partition(b"\n")
        try:
            key = raw_key.decode("utf-8").casefold()
            value = raw_value.decode("utf-8").casefold()
        except UnicodeDecodeError as exc:
            raise RunnerError("local Git configuration is not UTF-8") from exc
        if key in {"core.sparsecheckout", "core.sparsecheckoutcone"}:
            if not separator or value not in {
                "true",
                "false",
                "yes",
                "no",
                "on",
                "off",
                "1",
                "0",
            }:
                raise RunnerError("sparse-checkout setting is not canonical Boolean")
            if value in {"true", "yes", "on", "1"}:
                raise RunnerError("sparse checkout is prohibited")
    try:
        os.lstat(repository / ".git/info/sparse-checkout")
    except FileNotFoundError:
        pass
    else:
        raise RunnerError("sparse-checkout pattern state is prohibited")


def _capture_committed_files(
    config: _PreflightConfig,
    commit: str,
) -> tuple[str, tuple[_TrackedFile, ...], str]:
    commit = _require_git_object(commit, "source commit")
    try:
        resolved = (
            _run_git(config, ("rev-parse", "--verify", f"{commit}^{{commit}}"))
            .decode("ascii")
            .strip()
        )
        tree = (
            _run_git(config, ("rev-parse", "--verify", f"{commit}^{{tree}}"))
            .decode("ascii")
            .strip()
        )
    except UnicodeDecodeError as exc:
        raise RunnerError("Git object identity is not ASCII") from exc
    if resolved != commit:
        raise RunnerError("source commit does not resolve exactly")
    _require_git_object(tree, "source tree")
    raw_tree = _run_git(config, ("ls-tree", "-r", "-z", "--full-tree", commit))
    records: list[_TrackedFile] = []
    for raw_record in raw_tree.split(b"\x00"):
        if not raw_record:
            continue
        try:
            metadata, raw_path = raw_record.split(b"\t", 1)
            raw_mode, raw_type, raw_object = metadata.split(b" ")
            mode = raw_mode.decode("ascii")
            object_type = raw_type.decode("ascii")
            object_id = raw_object.decode("ascii")
        except (ValueError, UnicodeDecodeError) as exc:
            raise RunnerError("Git tree contains a malformed record") from exc
        if mode not in {"100644", "100755"} or object_type != "blob":
            raise RunnerError("source tree permits regular tracked blobs only")
        path = _decode_git_path(raw_path)
        blob = _run_git(config, ("cat-file", "blob", object_id))
        records.append(
            _TrackedFile(
                path=path,
                kind="executable_file" if mode == "100755" else "regular_file",
                file_sha256=_sha256_bytes(blob),
            )
        )
    records.sort(key=lambda item: item.path)
    paths = tuple(item.path for item in records)
    if not records or len(paths) != len(set(paths)):
        raise RunnerError("committed source inventory is empty or contains duplicate paths")
    inventory = tuple(records)
    digest = _sha256_bytes(_canonical_json_bytes({"files": [item.record() for item in inventory]}))
    return tree, inventory, digest


def _require_index_matches_commit(
    config: _PreflightConfig,
    tracked_files: tuple[_TrackedFile, ...],
) -> None:
    expected_paths = tuple(item.path for item in tracked_files)
    visible_paths: list[str] = []
    for raw_record in _run_git(config, ("ls-files", "-v", "-z")).split(b"\x00"):
        if not raw_record:
            continue
        if len(raw_record) < 3 or raw_record[:2] != b"H ":
            raise RunnerError("Git index contains hidden or special path state")
        visible_paths.append(_decode_git_path(raw_record[2:]))
    if tuple(visible_paths) != expected_paths:
        raise RunnerError("Git index path inventory differs from HEAD")
    staged_paths: list[str] = []
    for raw_record in _run_git(config, ("ls-files", "--stage", "-z")).split(b"\x00"):
        if not raw_record:
            continue
        try:
            metadata, raw_path = raw_record.split(b"\t", 1)
            mode, object_id, stage = metadata.decode("ascii").split(" ")
        except (ValueError, UnicodeDecodeError) as exc:
            raise RunnerError("Git staged inventory contains a malformed record") from exc
        if mode not in {"100644", "100755"} or stage != "0":
            raise RunnerError("Git index contains a noncanonical stage or mode")
        _require_git_object(object_id, "index blob")
        staged_paths.append(_decode_git_path(raw_path))
    if tuple(staged_paths) != expected_paths:
        raise RunnerError("Git staged path inventory differs from HEAD")


def _open_relative_directory(root_descriptor: int, parts: Sequence[str]) -> int:
    descriptor = os.dup(root_descriptor)
    try:
        for part in parts:
            child = os.open(
                part,
                os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW,
                dir_fd=descriptor,
            )
            _require_safe_directory_stat(os.fstat(child), "source directory")
            os.close(descriptor)
            descriptor = child
        return descriptor
    except Exception:
        os.close(descriptor)
        raise


def _hash_tracked_worktree_file(
    root_descriptor: int,
    item: _TrackedFile,
) -> str:
    pure = PurePosixPath(item.path)
    directory = _open_relative_directory(root_descriptor, pure.parts[:-1])
    descriptor: int | None = None
    try:
        descriptor = os.open(
            pure.parts[-1],
            os.O_RDONLY | os.O_NOFOLLOW,
            dir_fd=directory,
        )
        digest, observed = _hash_open_regular_file(
            descriptor,
            label=f"tracked source {item.path}",
            require_single_link=True,
        )
        mode = stat.S_IMODE(observed.st_mode)
        executable = mode & 0o111 == 0o111
        if item.kind == "executable_file":
            if not executable:
                raise RunnerError(f"tracked executable mode differs for {item.path}")
        elif executable or mode & 0o111:
            raise RunnerError(f"tracked regular-file mode differs for {item.path}")
        return digest
    except OSError as exc:
        raise RunnerError(f"tracked source cannot be opened safely: {item.path}") from exc
    finally:
        if descriptor is not None:
            os.close(descriptor)
        os.close(directory)


def _require_exact_protected_tree(
    root_descriptor: int,
    root_name: str,
    expected_files: Mapping[str, _TrackedFile],
) -> None:
    expected_directories = {root_name}
    for path in expected_files:
        parts = PurePosixPath(path).parts
        for index in range(1, len(parts)):
            expected_directories.add(PurePosixPath(*parts[:index]).as_posix())
    root = _open_relative_directory(root_descriptor, (root_name,))
    seen_files: set[str] = set()
    seen_directories = {root_name}

    def visit(directory: int, relative: str) -> None:
        before = os.fstat(directory)
        _require_safe_directory_stat(before, f"protected source directory {relative}")
        with os.scandir(directory) as scanner:
            entries = sorted(scanner, key=lambda entry: entry.name)
        for entry in entries:
            try:
                entry.name.encode("utf-8", errors="strict")
            except UnicodeEncodeError as exc:
                raise RunnerError("protected source entry name is not UTF-8") from exc
            if entry.name in {"", ".", ".."} or "/" in entry.name or "\x00" in entry.name:
                raise RunnerError("protected source entry name is not canonical")
            path = PurePosixPath(relative, entry.name).as_posix()
            observed = entry.stat(follow_symlinks=False)
            if stat.S_ISDIR(observed.st_mode):
                if path not in expected_directories:
                    raise RunnerError(f"uncommitted directory in protected source tree: {path}")
                child = os.open(
                    entry.name,
                    os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW,
                    dir_fd=directory,
                )
                try:
                    if (observed.st_dev, observed.st_ino) != (
                        os.fstat(child).st_dev,
                        os.fstat(child).st_ino,
                    ):
                        raise RunnerError(f"protected source directory raced: {path}")
                    seen_directories.add(path)
                    visit(child, path)
                finally:
                    os.close(child)
            elif stat.S_ISREG(observed.st_mode):
                item = expected_files.get(path)
                if item is None:
                    raise RunnerError(f"uncommitted file in protected source tree: {path}")
                descriptor = os.open(
                    entry.name,
                    os.O_RDONLY | os.O_NOFOLLOW,
                    dir_fd=directory,
                )
                try:
                    digest, opened = _hash_open_regular_file(
                        descriptor,
                        label=f"protected source {path}",
                        require_single_link=True,
                    )
                    if (observed.st_dev, observed.st_ino) != (
                        opened.st_dev,
                        opened.st_ino,
                    ):
                        raise RunnerError(f"protected source file raced: {path}")
                    if digest != item.file_sha256:
                        raise RunnerError(f"protected source bytes differ from Git: {path}")
                finally:
                    os.close(descriptor)
                seen_files.add(path)
            else:
                raise RunnerError(f"symlink or special entry in protected source tree: {path}")
        if _stat_identity(before) != _stat_identity(os.fstat(directory)):
            raise RunnerError(f"protected source directory changed during scan: {relative}")

    try:
        visit(root, root_name)
    finally:
        os.close(root)
    if seen_files != set(expected_files) or seen_directories != expected_directories:
        raise RunnerError(f"protected source tree is incomplete: {root_name}")


def _require_root_import_controls(
    config: _PreflightConfig,
    root_descriptor: int,
    tracked_by_path: Mapping[str, _TrackedFile],
) -> None:
    for name in sorted(config.root_import_controls):
        if PurePosixPath(name).parts != (name,):
            raise RunnerError("root import-control name is not canonical")
        expected = tracked_by_path.get(name)
        try:
            observed = os.stat(name, dir_fd=root_descriptor, follow_symlinks=False)
        except FileNotFoundError:
            if expected is not None:
                raise RunnerError(f"tracked root import control is missing: {name}")
            continue
        if expected is None:
            raise RunnerError(f"uncommitted root import control is prohibited: {name}")
        if not stat.S_ISREG(observed.st_mode):
            raise RunnerError(f"root import control is not a regular file: {name}")
        if _hash_tracked_worktree_file(root_descriptor, expected) != expected.file_sha256:
            raise RunnerError(f"root import control differs from Git: {name}")


def _capture_source_facts(
    config: _PreflightConfig,
    *,
    executing_file: Path,
) -> _SourceFacts:
    git_hash = _hash_absolute_regular_file(
        V3_GIT_EXECUTABLE,
        label="pinned Git executable",
        require_current_owner=False,
        require_single_link=False,
    )
    _git_repository_precheck(config)
    before_commit_raw = _run_git(config, ("rev-parse", "--verify", "HEAD^{commit}"))
    try:
        commit = before_commit_raw.decode("ascii").strip()
    except UnicodeDecodeError as exc:
        raise RunnerError("HEAD identity is not ASCII") from exc
    _require_git_object(commit, "HEAD")
    tree, tracked_files, source_digest = _capture_committed_files(config, commit)
    _require_index_matches_commit(config, tracked_files)
    if _run_git(config, ("status", "--porcelain=v1", "--untracked-files=all", "-z")):
        raise RunnerError("runner requires an exactly clean Git worktree")
    tracked_by_path = {item.path: item for item in tracked_files}
    root = _open_absolute_directory(config.repository)
    try:
        for item in tracked_files:
            if _hash_tracked_worktree_file(root, item) != item.file_sha256:
                raise RunnerError(f"tracked working-tree bytes differ from Git: {item.path}")
        for root_name in config.protected_source_roots:
            prefix = root_name + "/"
            expected = {
                path: item for path, item in tracked_by_path.items() if path.startswith(prefix)
            }
            if not expected:
                raise RunnerError(f"protected source root has no committed files: {root_name}")
            _require_exact_protected_tree(root, root_name, expected)
        _require_root_import_controls(config, root, tracked_by_path)
    finally:
        os.close(root)
    runner = tracked_by_path.get(config.runner_repository_path)
    if runner is None or runner.kind != "regular_file":
        raise RunnerError("runner is not the exact committed regular source file")
    executing_hash = _hash_absolute_regular_file(
        executing_file,
        label="executing runner source",
        require_single_link=True,
    )
    canonical_runner_hash = _hash_absolute_regular_file(
        config.runner_path,
        label="canonical runner source",
        require_single_link=True,
    )
    if executing_hash != runner.file_sha256 or canonical_runner_hash != runner.file_sha256:
        raise RunnerError("executing runner bytes differ from committed canonical runner")
    _require_index_matches_commit(config, tracked_files)
    after_commit = _run_git(config, ("rev-parse", "--verify", "HEAD^{commit}"))
    after_tree = _run_git(config, ("rev-parse", "--verify", "HEAD^{tree}"))
    after_status = _run_git(
        config,
        ("status", "--porcelain=v1", "--untracked-files=all", "-z"),
    )
    if (
        after_commit != before_commit_raw
        or after_tree.decode("ascii").strip() != tree
        or after_status
    ):
        raise RunnerError("Git source changed during pre-import capture")
    git_hash_after = _hash_absolute_regular_file(
        V3_GIT_EXECUTABLE,
        label="pinned Git executable",
        require_current_owner=False,
        require_single_link=False,
    )
    if git_hash_after != git_hash:
        raise RunnerError("pinned Git executable changed during source capture")
    try:
        git_version = _run_git(config, ("--version",)).decode("ascii").strip()
    except UnicodeDecodeError as exc:
        raise RunnerError("pinned Git version is not ASCII") from exc
    if not git_version.startswith("git version ") or "\n" in git_version:
        raise RunnerError("pinned Git version output is malformed")
    return _SourceFacts(
        commit=commit,
        tree=tree,
        tracked_files=tracked_files,
        source_bundle_sha256=source_digest,
        git_executable_file_sha256=git_hash,
        git_version=git_version,
        runner_file_sha256=runner.file_sha256,
    )


def _site_mode(value: int) -> str:
    return f"0{stat.S_IMODE(value):03o}"


def _capture_python_site_inventory(site_packages: Path) -> Mapping[str, Any]:
    """Hash every installed site entry without importing or executing any of it."""

    root = _open_absolute_directory(site_packages)
    entries: list[dict[str, Any]] = []

    def visit(directory: int, relative: str) -> None:
        before = os.fstat(directory)
        _require_safe_directory_stat(before, f"Python site directory {relative}")
        entries.append(
            {
                "kind": "directory",
                "mode": _site_mode(before.st_mode),
                "path": relative,
            }
        )
        with os.scandir(directory) as scanner:
            children = sorted(scanner, key=lambda entry: entry.name)
        for child in children:
            try:
                child.name.encode("utf-8", errors="strict")
            except UnicodeEncodeError as exc:
                raise RunnerError("Python site entry name is not UTF-8") from exc
            if child.name in {"", ".", ".."} or "/" in child.name or "\x00" in child.name:
                raise RunnerError("Python site entry name is not canonical")
            path = child.name if relative == "." else f"{relative}/{child.name}"
            observed = child.stat(follow_symlinks=False)
            if stat.S_ISDIR(observed.st_mode):
                descriptor = os.open(
                    child.name,
                    os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW,
                    dir_fd=directory,
                )
                try:
                    opened = os.fstat(descriptor)
                    if (observed.st_dev, observed.st_ino) != (
                        opened.st_dev,
                        opened.st_ino,
                    ):
                        raise RunnerError(f"Python site directory raced: {path}")
                    visit(descriptor, path)
                finally:
                    os.close(descriptor)
            elif stat.S_ISREG(observed.st_mode):
                descriptor = os.open(
                    child.name,
                    os.O_RDONLY | os.O_NOFOLLOW,
                    dir_fd=directory,
                )
                try:
                    digest, opened = _hash_open_regular_file(
                        descriptor,
                        label=f"Python site file {path}",
                        require_single_link=False,
                    )
                    if (observed.st_dev, observed.st_ino) != (
                        opened.st_dev,
                        opened.st_ino,
                    ):
                        raise RunnerError(f"Python site file raced: {path}")
                    entries.append(
                        {
                            "file_sha256": digest,
                            "kind": "regular_file",
                            "link_count": opened.st_nlink,
                            "mode": _site_mode(opened.st_mode),
                            "path": path,
                            "size_bytes": opened.st_size,
                        }
                    )
                finally:
                    os.close(descriptor)
            else:
                raise RunnerError(f"Python site contains a symlink or special entry: {path}")
        if _stat_identity(before) != _stat_identity(os.fstat(directory)):
            raise RunnerError(f"Python site directory changed during scan: {relative}")

    try:
        visit(root, ".")
    finally:
        os.close(root)
    entries.sort(key=lambda item: str(item["path"]))
    directory_count = sum(item["kind"] == "directory" for item in entries)
    regular_files = tuple(item for item in entries if item["kind"] == "regular_file")
    preimage = {
        "entries": entries,
        "root_path": os.fspath(site_packages),
        "schema": _PYTHON_SITE_INVENTORY_SCHEMA,
    }
    return {
        "directory_count": directory_count,
        "inventory_sha256": _sha256_bytes(_canonical_json_bytes(preimage)),
        "regular_file_count": len(regular_files),
        "root_path": os.fspath(site_packages),
        "schema": _PYTHON_SITE_INVENTORY_SCHEMA,
        "total_regular_file_bytes": sum(int(item["size_bytes"]) for item in regular_files),
    }


def _process_executable_path_and_hash() -> tuple[str, str]:
    descriptor = os.open(f"/proc/{os.getpid()}/exe", os.O_RDONLY)
    try:
        digest, _ = _hash_open_regular_file(
            descriptor,
            label="current Python executable",
            require_current_owner=False,
            require_single_link=False,
        )
        target = os.readlink(f"/proc/{os.getpid()}/exe")
    finally:
        os.close(descriptor)
    if target.endswith(" (deleted)"):
        raise RunnerError("current Python executable was deleted")
    resolved = Path(target).resolve(strict=True)
    return os.fspath(resolved), digest


def _capture_python_record(config: _PreflightConfig) -> Mapping[str, Any]:
    executable_path, executable_hash = _process_executable_path_and_hash()
    configured = config.python_executable.resolve(strict=True)
    configured_hash = _hash_absolute_regular_file(
        configured,
        label="configured Python executable",
        require_current_owner=False,
        require_single_link=False,
    )
    if os.fspath(configured) != executable_path or configured_hash != executable_hash:
        raise RunnerError("configured Python and the executing image differ")
    return {
        "configured_executable_file_sha256": configured_hash,
        "configured_executable_path": os.fspath(configured),
        "executable_file_sha256": executable_hash,
        "executable_path": executable_path,
        "implementation_cache_tag": sys.implementation.cache_tag,
        "implementation_name": sys.implementation.name,
        "multiarch": sysconfig.get_config_var("MULTIARCH"),
        "soabi": sysconfig.get_config_var("SOABI"),
        "version": sys.version,
    }


def _strict_json_object(data: bytes, *, label: str) -> Mapping[str, Any]:
    if not data.endswith(b"\n") or data.endswith(b"\n\n"):
        raise RunnerError(f"{label} is not canonical JSON plus one LF")

    def exact_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        value: dict[str, Any] = {}
        for key, item in pairs:
            if key in value:
                raise RunnerError(f"{label} contains a duplicate JSON key")
            value[key] = item
        return value

    def reject_constant(value: str) -> None:
        raise RunnerError(f"{label} contains non-finite JSON: {value}")

    try:
        parsed = json.loads(
            data[:-1].decode("utf-8"),
            object_pairs_hook=exact_object,
            parse_constant=reject_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RunnerError(f"{label} is not strict UTF-8 JSON") from exc
    if not isinstance(parsed, dict) or _canonical_json_bytes(parsed) + b"\n" != data:
        raise RunnerError(f"{label} bytes are not canonical")
    return parsed


def _validate_payload_self_hash(payload: Mapping[str, Any], *, label: str) -> None:
    claimed = payload.get("payload_sha256")
    if not isinstance(claimed, str) or not _SHA256_RE.fullmatch(claimed):
        raise RunnerError(f"{label} payload hash is malformed")
    preimage = dict(payload)
    preimage.pop("payload_sha256")
    if _sha256_bytes(_canonical_json_bytes(preimage)) != claimed:
        raise RunnerError(f"{label} payload self-hash differs")


def _tracked_records(value: Any, *, label: str) -> tuple[_TrackedFile, ...]:
    if not isinstance(value, list) or not value:
        raise RunnerError(f"{label} is not a nonempty source inventory")
    records: list[_TrackedFile] = []
    for record in value:
        if not isinstance(record, dict) or set(record) != {"file_sha256", "kind", "path"}:
            raise RunnerError(f"{label} has a malformed record")
        path = record["path"]
        kind = record["kind"]
        digest = record["file_sha256"]
        if not isinstance(path, str):
            raise RunnerError(f"{label} path is not text")
        _decode_git_path(path.encode("utf-8"))
        if kind not in {"regular_file", "executable_file"}:
            raise RunnerError(f"{label} kind is invalid")
        if not isinstance(digest, str) or not _SHA256_RE.fullmatch(digest):
            raise RunnerError(f"{label} file hash is invalid")
        records.append(_TrackedFile(path=path, kind=kind, file_sha256=digest))
    records_tuple = tuple(records)
    paths = tuple(item.path for item in records_tuple)
    if paths != tuple(sorted(paths)) or len(paths) != len(set(paths)):
        raise RunnerError(f"{label} is not uniquely sorted")
    return records_tuple


def _require_current_source_compatible_with_bundle(
    config: _PreflightConfig,
    current: _SourceFacts,
    bundle_source: _SourceFacts,
) -> None:
    if (
        current.commit == bundle_source.commit
        and current.tree == bundle_source.tree
        and current.tracked_files == bundle_source.tracked_files
        and current.source_bundle_sha256 == bundle_source.source_bundle_sha256
    ):
        return
    before = {item.path: item for item in bundle_source.tracked_files}
    after = {item.path: item for item in current.tracked_files}
    added = set(after) - set(before)
    if (
        set(before) - set(after)
        or added != {config.selected_method_relative_path}
        or any(after[path] != item for path, item in before.items())
        or after[config.selected_method_relative_path].kind != "regular_file"
    ):
        raise RunnerError("current source is neither bundle A nor exact selected-only B")
    _run_git(
        config,
        ("merge-base", "--is-ancestor", bundle_source.commit, current.commit),
    )


def _require_v3_4_bundle_recovery_identity(
    payload: Mapping[str, Any],
    *,
    canonical_path: Path,
) -> None:
    expected: dict[str, Any] = {
        "schema": _DEVELOPMENT_BUNDLE_SCHEMA,
        "candidate_id": _CANDIDATE_ID,
        "protocol_revision": _PROTOCOL_REVISION,
        "development_attempt_id": _DEVELOPMENT_ATTEMPT_ID,
        "canonical_path": os.fspath(canonical_path),
        "preoutcome_amendment_path": _RECOVERY_AMENDMENT_PATH,
        "preoutcome_amendment_file_sha256": _RECOVERY_AMENDMENT_SHA256,
        "original_preoutcome_amendment_path": _ORIGINAL_AMENDMENT_PATH,
        "original_preoutcome_amendment_file_sha256": _ORIGINAL_AMENDMENT_SHA256,
        "prior_recovery_amendment_path": _PRIOR_RECOVERY_AMENDMENT_PATH,
        "prior_recovery_amendment_file_sha256": _PRIOR_RECOVERY_AMENDMENT_SHA256,
        "earlier_recovery_amendment_path": _EARLIER_RECOVERY_AMENDMENT_PATH,
        "earlier_recovery_amendment_file_sha256": _EARLIER_RECOVERY_AMENDMENT_SHA256,
        "first_recovery_amendment_path": _FIRST_RECOVERY_AMENDMENT_PATH,
        "first_recovery_amendment_file_sha256": _FIRST_RECOVERY_AMENDMENT_SHA256,
        "retired_v1_source_commit": _RETIRED_V1_COMMIT,
        "retired_v1_source_tree": _RETIRED_V1_TREE,
        "retired_v1_artifact_inventory_schema": (
            "cfeg.metadata-calibration-efficiency-v3.retired-development-inventory.v1"
        ),
        "retired_v1_artifact_inventory_sha256": _RETIRED_V1_INVENTORY_SHA256,
        "retired_v1_development_bundle_path": (
            "/home/whwovy/v3-artifacts/metadata-calibration-efficiency-v3/"
            "development-v1/development-bundle.json"
        ),
        "retired_v1_development_bundle_schema": (
            "cfeg.metadata-calibration-efficiency-v3.development-bundle.v1"
        ),
        "retired_v1_development_bundle_payload_sha256": (_RETIRED_V1_BUNDLE_PAYLOAD_SHA256),
        "retired_v1_development_bundle_file_sha256": _RETIRED_V1_BUNDLE_FILE_SHA256,
        "retired_v1_context_reference_path": (
            "/home/whwovy/v3-artifacts/metadata-calibration-efficiency-v3/"
            "development-v1/context-reference.json"
        ),
        "retired_v1_context_reference_schema": (
            "cfeg.metadata-calibration-efficiency-v3.context-reference.v1"
        ),
        "retired_v1_context_reference_payload_sha256": (_RETIRED_V1_CONTEXT_PAYLOAD_SHA256),
        "retired_v1_context_reference_file_sha256": _RETIRED_V1_CONTEXT_FILE_SHA256,
        "retired_v1_context_reference_root_seed": 20_260_910,
        "retired_v1_development_root_seed": 20_260_909,
        "retired_v1_incident_error": "AuthorityError",
        "retired_v1_incident_message": ("this governed process role cannot mutate canonical state"),
        "retired_v1_incident_output_sha256": _RETIRED_V1_INCIDENT_OUTPUT_SHA256,
        "retired_v1_development_seedsequence_created": False,
        "retired_v1_development_DGP_executed": False,
        "retired_v1_development_result_present": False,
        "retired_v1_experimental_outcome_observed": False,
        "retired_v1_governed_network_accessed": False,
        "retired_v1_continuation_authorized": False,
        "retired_v2_source_commit": _RETIRED_V2_COMMIT,
        "retired_v2_source_tree": _RETIRED_V2_TREE,
        "retired_v2_source_bundle_sha256": _RETIRED_V2_SOURCE_BUNDLE_SHA256,
        "retired_v2_artifact_inventory_schema": (
            "cfeg.metadata-calibration-efficiency-v3.retired-development-inventory.v2"
        ),
        "retired_v2_artifact_inventory_sha256": _RETIRED_V2_INVENTORY_SHA256,
        "retired_v2_development_bundle_path": (
            "/home/whwovy/v3-artifacts/metadata-calibration-efficiency-v3/"
            "development-v2/development-bundle.json"
        ),
        "retired_v2_development_bundle_schema": (
            "cfeg.metadata-calibration-efficiency-v3.development-bundle.v2"
        ),
        "retired_v2_development_bundle_payload_sha256": (_RETIRED_V2_BUNDLE_PAYLOAD_SHA256),
        "retired_v2_development_bundle_file_sha256": _RETIRED_V2_BUNDLE_FILE_SHA256,
        "retired_v2_incident_error": "TypeError",
        "retired_v2_incident_message": ("Object of type mappingproxy is not JSON serializable"),
        "retired_v2_incident_output_sha256": _RETIRED_V2_INCIDENT_OUTPUT_SHA256,
        "retired_v2_context_reference_root_seed": 20_260_910,
        "retired_v2_context_reference_replay_count": 2,
        "retired_v2_context_reference_present": False,
        "retired_v2_development_root_seed": 20_260_909,
        "retired_v2_development_rng_authority_issued": False,
        "retired_v2_development_seedsequence_created": False,
        "retired_v2_development_DGP_executed": False,
        "retired_v2_development_result_present": False,
        "retired_v2_experimental_outcome_observed": False,
        "retired_v2_governed_network_accessed": False,
        "retired_v2_continuation_authorized": False,
        "retired_v3_source_commit": _RETIRED_V3_COMMIT,
        "retired_v3_source_tree": _RETIRED_V3_TREE,
        "retired_v3_source_bundle_sha256": _RETIRED_V3_SOURCE_BUNDLE_SHA256,
        "retired_v3_numerical_runtime_fingerprint_sha256": (_RETIRED_V3_RUNTIME_FINGERPRINT_SHA256),
        "retired_v3_artifact_inventory_schema": (
            "cfeg.metadata-calibration-efficiency-v3.retired-development-inventory.v3"
        ),
        "retired_v3_artifact_inventory_sha256": _RETIRED_V3_INVENTORY_SHA256,
        "retired_v3_development_bundle_path": (
            "/home/whwovy/v3-artifacts/metadata-calibration-efficiency-v3/"
            "development-v3/development-bundle.json"
        ),
        "retired_v3_development_bundle_schema": (
            "cfeg.metadata-calibration-efficiency-v3.development-bundle.v3"
        ),
        "retired_v3_development_bundle_payload_sha256": _RETIRED_V3_BUNDLE_PAYLOAD_SHA256,
        "retired_v3_development_bundle_file_sha256": _RETIRED_V3_BUNDLE_FILE_SHA256,
        "retired_v3_context_reference_path": (
            "/home/whwovy/v3-artifacts/metadata-calibration-efficiency-v3/"
            "development-v3/context-reference.json"
        ),
        "retired_v3_context_reference_schema": (
            "cfeg.metadata-calibration-efficiency-v3.context-reference.v1"
        ),
        "retired_v3_context_reference_payload_sha256": _RETIRED_V1_CONTEXT_PAYLOAD_SHA256,
        "retired_v3_context_reference_file_sha256": _RETIRED_V1_CONTEXT_FILE_SHA256,
        "retired_v3_development_start_path": (
            "/home/whwovy/v3-artifacts/metadata-calibration-efficiency-v3/"
            "development-v3/development-start.json"
        ),
        "retired_v3_development_start_schema": (
            "cfeg.metadata-calibration-efficiency-v3.development-start.v1"
        ),
        "retired_v3_development_start_payload_sha256": _RETIRED_V3_START_PAYLOAD_SHA256,
        "retired_v3_development_start_file_sha256": _RETIRED_V3_START_FILE_SHA256,
        "retired_v3_development_started_at_UTC": "2026-09-06T21:06:11Z",
        "retired_v3_development_start_creator_process_id": 550_174,
        "retired_v3_incident_error": "TypeError",
        "retired_v3_incident_message": "Object of type mappingproxy is not JSON serializable",
        "retired_v3_incident_output_sha256": _RETIRED_V2_INCIDENT_OUTPUT_SHA256,
        "retired_v3_incident_boundary": (
            "after_execute_complete_development_returned_during_governance_frozen_parse_to_"
            "core_publication_validation_before_result_O_EXCL"
        ),
        "retired_v3_context_reference_root_seed": 20_260_910,
        "retired_v3_development_root_seed": 20_260_909,
        "retired_v3_development_seed_consumed": True,
        "retired_v3_control_flow_inference_classification": (
            "strong_control_flow_inference_not_observed_outcome_or_durable_attestation"
        ),
        "retired_v3_development_rng_authority_issued_in_memory": True,
        "retired_v3_development_seedsequence_and_DGP_executed": True,
        "retired_v3_complete_grid_gate_reports_computed_in_memory": True,
        "retired_v3_development_selection_computed_in_memory": True,
        "retired_v3_development_result_present": False,
        "retired_v3_selected_method_present": False,
        "retired_v3_test_evidence_present": False,
        "retired_v3_canary_present": False,
        "retired_v3_experimental_outcome_observed": False,
        "retired_v3_governed_network_accessed": False,
        "retired_v3_human_EEG_outcome_accessed": False,
        "retired_v3_continuation_authorized": False,
        "retired_v4_source_commit": _RETIRED_V4_COMMIT,
        "retired_v4_source_tree": _RETIRED_V4_TREE,
        "retired_v4_source_bundle_sha256": _RETIRED_V4_SOURCE_BUNDLE_SHA256,
        "retired_v4_numerical_runtime_fingerprint_sha256": (_RETIRED_V4_RUNTIME_FINGERPRINT_SHA256),
        "retired_v4_artifact_inventory_schema": (
            "cfeg.metadata-calibration-efficiency-v3.retired-development-inventory.v4"
        ),
        "retired_v4_artifact_inventory_sha256": _RETIRED_V4_INVENTORY_SHA256,
        "retired_v4_development_bundle_path": (
            "/home/whwovy/v3-artifacts/metadata-calibration-efficiency-v3/"
            "development-v4/development-bundle.json"
        ),
        "retired_v4_development_bundle_schema": (
            "cfeg.metadata-calibration-efficiency-v3.development-bundle.v4"
        ),
        "retired_v4_development_bundle_payload_sha256": _RETIRED_V4_BUNDLE_PAYLOAD_SHA256,
        "retired_v4_development_bundle_file_sha256": _RETIRED_V4_BUNDLE_FILE_SHA256,
        "retired_v4_context_reference_path": (
            "/home/whwovy/v3-artifacts/metadata-calibration-efficiency-v3/"
            "development-v4/context-reference.json"
        ),
        "retired_v4_context_reference_schema": (
            "cfeg.metadata-calibration-efficiency-v3.context-reference.v1"
        ),
        "retired_v4_context_reference_payload_sha256": _RETIRED_V1_CONTEXT_PAYLOAD_SHA256,
        "retired_v4_context_reference_file_sha256": _RETIRED_V1_CONTEXT_FILE_SHA256,
        "retired_v4_development_start_path": (
            "/home/whwovy/v3-artifacts/metadata-calibration-efficiency-v3/"
            "development-v4/development-start.json"
        ),
        "retired_v4_development_start_schema": (
            "cfeg.metadata-calibration-efficiency-v3.development-start.v1"
        ),
        "retired_v4_development_start_payload_sha256": _RETIRED_V4_START_PAYLOAD_SHA256,
        "retired_v4_development_start_file_sha256": _RETIRED_V4_START_FILE_SHA256,
        "retired_v4_development_started_at_UTC": "2026-09-06T23:35:38Z",
        "retired_v4_development_start_creator_process_id": 1_234_958,
        "retired_v4_incident_error": "TypeError",
        "retired_v4_incident_message": (
            "primary_values_by_endpoint must be an exact endpoint-ordered dictionary."
        ),
        "retired_v4_incident_output_sha256": _RETIRED_V4_INCIDENT_OUTPUT_SHA256,
        "retired_v4_incident_boundary": (
            "after_execute_complete_development_returned_during_governance_canonical_parse_to_"
            "core_publication_validation_before_result_O_EXCL"
        ),
        "retired_v4_context_reference_root_seed": 20_260_910,
        "retired_v4_development_root_seed": 3_156_745_110,
        "retired_v4_development_seed_consumed": True,
        "retired_v4_control_flow_inference_classification": (
            "strong_control_flow_inference_not_observed_outcome_or_durable_attestation"
        ),
        "retired_v4_development_rng_authority_issued_in_memory": True,
        "retired_v4_development_seedsequence_and_DGP_executed": True,
        "retired_v4_complete_grid_gate_reports_computed_in_memory": True,
        "retired_v4_development_selection_computed_in_memory": True,
        "retired_v4_development_result_present": False,
        "retired_v4_selected_method_present": False,
        "retired_v4_test_evidence_present": False,
        "retired_v4_canary_present": False,
        "retired_v4_experimental_outcome_observed": False,
        "retired_v4_governed_network_accessed": False,
        "retired_v4_human_EEG_outcome_accessed": False,
        "retired_v4_continuation_authorized": False,
        "original_retired_scientific_contract_projection_sha256": (
            _ORIGINAL_SCIENTIFIC_CONTRACT_PROJECTION_SHA256
        ),
        "retired_scientific_contract_projection_sha256": (
            _RETIRED_SCIENTIFIC_CONTRACT_PROJECTION_SHA256
        ),
        "replacement_seed_preimage_schema": (
            "cfeg.metadata-calibration-efficiency-v3.replacement-development-seed-preimage.v2"
        ),
        "replacement_seed_preimage_sha256": _REPLACEMENT_SEED_PREIMAGE_SHA256,
        "replacement_seed_digest_split": "eight_big_endian_uint32_words",
        "replacement_seed_selection_rule": (
            "first_nonzero_word_outside_forbidden_seed_set_without_reroll_or_counter"
        ),
        "replacement_seed_selected_word_index": 0,
        "replacement_development_root_seed": _REPLACEMENT_DEVELOPMENT_ROOT_SEED,
        "replacement_seed_outcome_used": False,
        "replacement_development_evidence_role": (
            "transparent_synthetic_development_and_model_selection_not_human_confirmatory_evidence"
        ),
        "development_result_endpoint_order_policy": (
            "endpoint_order_array_is_authoritative_JSON_object_keys_are_an_exact_unordered_set_"
            "and_all_validation_iteration_uses_endpoint_order"
        ),
        "development_start_receipt_schema": (
            "cfeg.metadata-calibration-efficiency-v3.development-start.v1"
        ),
        "development_start_receipt_path": (
            "/home/whwovy/v3-artifacts/metadata-calibration-efficiency-v3/"
            "development-v5/development-start.json"
        ),
        "development_start_receipt_policy": (
            "O_EXCL_before_development_authority_and_first_RNG_draw_creator_process_only"
        ),
        "scientific_contract_projection_sha256": (_SCIENTIFIC_CONTRACT_PROJECTION_SHA256),
        "scientific_lockbox_authorized": False,
        "human_EEG_outcome_authorized": False,
    }
    if frozenset(payload) != _DEVELOPMENT_BUNDLE_FIELDS:
        raise RunnerError("development bundle field inventory differs before import")
    for name, expected_value in expected.items():
        observed = payload.get(name)
        if (type(expected_value) is bool and observed is not expected_value) or (
            type(expected_value) is not bool and observed != expected_value
        ):
            raise RunnerError(f"development bundle recovery identity differs: {name}")


def _validate_existing_bundle(
    config: _PreflightConfig,
    data: bytes,
    *,
    source: _SourceFacts,
    python: Mapping[str, Any],
    site_inventory: Mapping[str, Any],
) -> Mapping[str, Any]:
    payload = _strict_json_object(data, label="development bundle")
    _validate_payload_self_hash(payload, label="development bundle")
    _require_v3_4_bundle_recovery_identity(payload, canonical_path=config.bundle_path)
    if (
        payload["schema"] != _DEVELOPMENT_BUNDLE_SCHEMA
        or payload["candidate_id"] != _CANDIDATE_ID
        or payload["protocol_revision"] != _PROTOCOL_REVISION
        or payload["development_attempt_id"] != _DEVELOPMENT_ATTEMPT_ID
        or payload["canonical_path"] != os.fspath(config.bundle_path)
        or payload["preoutcome_amendment_path"] != _RECOVERY_AMENDMENT_PATH
        or payload["preoutcome_amendment_file_sha256"] != _RECOVERY_AMENDMENT_SHA256
        or payload["original_preoutcome_amendment_path"] != _ORIGINAL_AMENDMENT_PATH
        or payload["original_preoutcome_amendment_file_sha256"] != _ORIGINAL_AMENDMENT_SHA256
        or payload["retired_v1_source_commit"] != _RETIRED_V1_COMMIT
        or payload["retired_v1_source_tree"] != _RETIRED_V1_TREE
        or payload["retired_v1_artifact_inventory_schema"]
        != "cfeg.metadata-calibration-efficiency-v3.retired-development-inventory.v1"
        or payload["retired_v1_artifact_inventory_sha256"] != _RETIRED_V1_INVENTORY_SHA256
        or payload["retired_v1_development_bundle_path"]
        != (
            "/home/whwovy/v3-artifacts/metadata-calibration-efficiency-v3/"
            "development-v1/development-bundle.json"
        )
        or payload["retired_v1_development_bundle_schema"]
        != "cfeg.metadata-calibration-efficiency-v3.development-bundle.v1"
        or payload["retired_v1_development_bundle_payload_sha256"]
        != _RETIRED_V1_BUNDLE_PAYLOAD_SHA256
        or payload["retired_v1_development_bundle_file_sha256"] != _RETIRED_V1_BUNDLE_FILE_SHA256
        or payload["retired_v1_context_reference_payload_sha256"]
        != _RETIRED_V1_CONTEXT_PAYLOAD_SHA256
        or payload["retired_v1_context_reference_file_sha256"] != _RETIRED_V1_CONTEXT_FILE_SHA256
        or payload["retired_v1_context_reference_path"]
        != (
            "/home/whwovy/v3-artifacts/metadata-calibration-efficiency-v3/"
            "development-v1/context-reference.json"
        )
        or payload["retired_v1_context_reference_schema"]
        != "cfeg.metadata-calibration-efficiency-v3.context-reference.v1"
        or payload["retired_v1_context_reference_root_seed"] != 20_260_910
        or payload["retired_v1_development_root_seed"] != 20_260_909
        or payload["retired_v1_incident_error"] != "AuthorityError"
        or payload["retired_v1_incident_message"]
        != "this governed process role cannot mutate canonical state"
        or payload["retired_v1_incident_output_sha256"] != _RETIRED_V1_INCIDENT_OUTPUT_SHA256
        or payload["retired_v1_development_seedsequence_created"] is not False
        or payload["retired_v1_development_DGP_executed"] is not False
        or payload["retired_v1_development_result_present"] is not False
        or payload["retired_v1_experimental_outcome_observed"] is not False
        or payload["retired_v1_governed_network_accessed"] is not False
        or payload["retired_v1_continuation_authorized"] is not False
        or payload["scientific_contract_projection_sha256"]
        != _SCIENTIFIC_CONTRACT_PROJECTION_SHA256
        or payload["scientific_lockbox_authorized"] is not False
        or payload["human_EEG_outcome_authorized"] is not False
    ):
        raise RunnerError("development bundle has an invalid frozen identity")
    commit = payload["clean_git_commit"]
    tree = payload["clean_git_tree"]
    source_digest = payload["source_bundle_sha256"]
    if not isinstance(commit, str) or not isinstance(tree, str):
        raise RunnerError("development bundle Git identity is malformed")
    if not isinstance(source_digest, str) or not _SHA256_RE.fullmatch(source_digest):
        raise RunnerError("development bundle source digest is malformed")
    committed_tree, committed_files, committed_digest = _capture_committed_files(config, commit)
    bundle_records = _tracked_records(
        payload["tracked_source_file_inventory"],
        label="development bundle source inventory",
    )
    if (
        tree != committed_tree
        or bundle_records != committed_files
        or source_digest != committed_digest
    ):
        raise RunnerError("development bundle source binding differs from Git objects")
    bundle_runner = next(
        (
            item
            for item in committed_files
            if item.path == config.runner_repository_path and item.kind == "regular_file"
        ),
        None,
    )
    if bundle_runner is None:
        raise RunnerError("development bundle source lacks the exact runner file")
    bundle_source = _SourceFacts(
        commit=commit,
        tree=committed_tree,
        tracked_files=committed_files,
        source_bundle_sha256=committed_digest,
        git_executable_file_sha256=source.git_executable_file_sha256,
        git_version=source.git_version,
        runner_file_sha256=bundle_runner.file_sha256,
    )
    _require_current_source_compatible_with_bundle(config, source, bundle_source)
    runtime = payload["numerical_runtime_inventory"]
    fingerprint = payload["numerical_runtime_fingerprint_sha256"]
    if not isinstance(runtime, dict):
        raise RunnerError("development bundle numerical runtime is not an object")
    if not isinstance(fingerprint, str) or not _SHA256_RE.fullmatch(fingerprint):
        raise RunnerError("development bundle numerical runtime hash is malformed")
    if _sha256_bytes(_canonical_json_bytes(runtime)) != fingerprint:
        raise RunnerError("development bundle numerical runtime self-hash differs")
    if runtime.get("python") != dict(python):
        raise RunnerError("development bundle Python runtime differs before import")
    if runtime.get("python_site_inventory") != dict(site_inventory):
        raise RunnerError("development bundle Python site inventory differs before import")
    git_record = runtime.get("git")
    expected_git = {
        "executable_file_sha256": source.git_executable_file_sha256,
        "executable_path": os.fspath(V3_GIT_EXECUTABLE),
        "version": source.git_version,
    }
    if git_record != expected_git:
        raise RunnerError("development bundle Git runtime differs before import")
    return payload


def _require_process_bootstrap(
    config: _PreflightConfig,
    *,
    executing_file: Path,
) -> str | None:
    flags = sys.flags
    if (
        flags.isolated != 1
        or flags.ignore_environment != 1
        or flags.dont_write_bytecode != 1
        or flags.no_user_site != 1
        or flags.no_site != 1
        or flags.hash_randomization != 1
    ):
        raise RunnerError("V3 runner requires -I -B -S from interpreter startup")
    if dict(os.environ) != _EXPECTED_GOVERNED_ENVIRONMENT:
        raise RunnerError("V3 runner environment is not the exact governed environment")
    if sys.executable != os.fspath(config.python_executable):
        raise RunnerError("V3 runner executable is not the exact configured Python")
    if tuple(sys.path) != config.standard_library_path:
        raise RunnerError("V3 pre-import sys.path is not the pristine standard library")
    if config.require_direct_invocation:
        if (
            Path(executing_file) != config.runner_path
            or Path(__file__) != config.runner_path
            or __name__ != "__main__"
            or __spec__ is not None
            or __package__ is not None
        ):
            raise RunnerError("V3 runner is not executing as the exact direct script")
        if len(sys.argv) != 2 or sys.argv[0] != os.fspath(config.runner_path):
            raise RunnerError("V3 runner argv is not the exact direct-script form")
        command = sys.argv[1]
        expected_orig = (
            os.fspath(config.python_executable),
            "-I",
            "-B",
            "-S",
            os.fspath(config.runner_path),
            command,
        )
        if command not in _COMMANDS or tuple(sys.orig_argv) != expected_orig:
            raise RunnerError("V3 runner original argv is not an authorized command role")
        return command
    return None


def _run_preimport_preflight(
    config: _PreflightConfig,
    *,
    executing_file: Path,
) -> _PreflightFacts:
    command = _require_process_bootstrap(config, executing_file=executing_file)
    source = _capture_source_facts(config, executing_file=executing_file)
    python = _capture_python_record(config)
    site_inventory = _capture_python_site_inventory(config.site_packages)
    bundle_data = _read_optional_artifact(config.bundle_path)
    bundle = None
    if bundle_data is not None:
        bundle = _validate_existing_bundle(
            config,
            bundle_data,
            source=source,
            python=python,
            site_inventory=site_inventory,
        )
    source_after = _capture_source_facts(config, executing_file=executing_file)
    if source_after != source:
        raise RunnerError("source changed across complete pre-import validation")
    if _capture_python_record(config) != python:
        raise RunnerError("Python executable changed across pre-import validation")
    if _capture_python_site_inventory(config.site_packages) != site_inventory:
        raise RunnerError("Python site changed across complete pre-import validation")
    return _PreflightFacts(
        config=config,
        command=command,
        source=source,
        python=python,
        python_site_inventory=site_inventory,
        bundle_payload=bundle,
        _issuer=_PREFLIGHT_ISSUER,
    )


def _require_preflight_unchanged(
    facts: _PreflightFacts,
    *,
    include_site: bool = True,
) -> None:
    if type(facts) is not _PreflightFacts or facts._issuer is not _PREFLIGHT_ISSUER:
        raise RunnerError("exact internally-issued preflight facts are required")
    expected_path = (
        *facts.config.standard_library_path,
        os.fspath(facts.config.repository_src),
        os.fspath(facts.config.site_packages),
    )
    if tuple(sys.path) != expected_path:
        raise RunnerError("active sys.path differs from the pre-import authority")
    if facts.command is not None:
        expected_orig = (
            os.fspath(facts.config.python_executable),
            "-I",
            "-B",
            "-S",
            os.fspath(facts.config.runner_path),
            facts.command,
        )
        if tuple(sys.orig_argv) != expected_orig or tuple(sys.argv) != (
            os.fspath(facts.config.runner_path),
            facts.command,
        ):
            raise RunnerError("active process argv differs from pre-import authority")
    source = _capture_source_facts(facts.config, executing_file=facts.config.runner_path)
    if source != facts.source:
        raise RunnerError("source differs from the pre-import authority")
    if _capture_python_record(facts.config) != facts.python:
        raise RunnerError("Python executable differs from the pre-import authority")
    if include_site:
        current_site = _capture_python_site_inventory(facts.config.site_packages)
        if current_site != facts.python_site_inventory:
            raise RunnerError("Python site differs from the pre-import authority")
    bundle_data = _read_optional_artifact(facts.config.bundle_path)
    if facts.bundle_payload is None:
        if bundle_data is not None:
            raise RunnerError("development bundle appeared after pre-import validation")
    else:
        if bundle_data is None:
            raise RunnerError("development bundle disappeared after pre-import validation")
        current_bundle = _validate_existing_bundle(
            facts.config,
            bundle_data,
            source=source,
            python=facts.python,
            site_inventory=facts.python_site_inventory,
        )
        if current_bundle != facts.bundle_payload:
            raise RunnerError("development bundle differs from pre-import validation")


def _activate_exact_import_path(facts: _PreflightFacts) -> None:
    if tuple(sys.path) != facts.config.standard_library_path:
        raise RunnerError("sys.path changed before exact import activation")
    sys.path[:] = [
        *facts.config.standard_library_path,
        os.fspath(facts.config.repository_src),
        os.fspath(facts.config.site_packages),
    ]
    expected = (
        *facts.config.standard_library_path,
        os.fspath(facts.config.repository_src),
        os.fspath(facts.config.site_packages),
    )
    if tuple(sys.path) != expected:
        raise RunnerError("failed to establish exact post-preflight sys.path")


class RunStage(str, Enum):
    BUNDLE_PENDING = "BUNDLE_PENDING"
    CONTEXT_REFERENCE_PENDING = "CONTEXT_REFERENCE_PENDING"
    DEVELOPMENT_PENDING = "DEVELOPMENT_PENDING"
    DEVELOPMENT_ATTEMPT_CONSUMED = "DEVELOPMENT_ATTEMPT_CONSUMED"
    DEVELOPMENT_NO_GO = "DEVELOPMENT_NO_GO"
    SELECTION_HANDOFF = "SELECTION_HANDOFF"
    TEST_EVIDENCE_PENDING = "TEST_EVIDENCE_PENDING"
    CANARY_AUTHORIZATION_PENDING = "CANARY_AUTHORIZATION_PENDING"
    CANARY_CLAIM_PENDING = "CANARY_CLAIM_PENDING"
    CANARY_BEACON_PENDING = "CANARY_BEACON_PENDING"
    CANARY_RESULT_PENDING = "CANARY_RESULT_PENDING"
    FRESH_CANARY_AUDIT_PENDING = "FRESH_CANARY_AUDIT_PENDING"
    CANARY_PASSED = "CANARY_PASSED"


_TERMINAL_STAGES = frozenset(
    {
        RunStage.DEVELOPMENT_ATTEMPT_CONSUMED,
        RunStage.DEVELOPMENT_NO_GO,
        RunStage.CANARY_PASSED,
    }
)
_ADVANCEABLE_STAGES = frozenset(
    {
        RunStage.BUNDLE_PENDING,
        RunStage.CONTEXT_REFERENCE_PENDING,
        RunStage.DEVELOPMENT_PENDING,
        RunStage.TEST_EVIDENCE_PENDING,
        RunStage.CANARY_AUTHORIZATION_PENDING,
        RunStage.CANARY_CLAIM_PENDING,
        RunStage.CANARY_BEACON_PENDING,
        RunStage.CANARY_RESULT_PENDING,
        RunStage.FRESH_CANARY_AUDIT_PENDING,
    }
)
_RESUME_TRANSITIONS = {
    RunStage.BUNDLE_PENDING: frozenset({RunStage.CONTEXT_REFERENCE_PENDING}),
    RunStage.CONTEXT_REFERENCE_PENDING: frozenset({RunStage.DEVELOPMENT_PENDING}),
    RunStage.DEVELOPMENT_PENDING: frozenset(
        {RunStage.DEVELOPMENT_NO_GO, RunStage.SELECTION_HANDOFF}
    ),
    RunStage.TEST_EVIDENCE_PENDING: frozenset({RunStage.CANARY_AUTHORIZATION_PENDING}),
    RunStage.CANARY_AUTHORIZATION_PENDING: frozenset({RunStage.CANARY_CLAIM_PENDING}),
    RunStage.CANARY_CLAIM_PENDING: frozenset({RunStage.CANARY_BEACON_PENDING}),
    RunStage.CANARY_BEACON_PENDING: frozenset({RunStage.CANARY_RESULT_PENDING}),
    RunStage.CANARY_RESULT_PENDING: frozenset({RunStage.FRESH_CANARY_AUDIT_PENDING}),
    RunStage.FRESH_CANARY_AUDIT_PENDING: frozenset({RunStage.CANARY_PASSED}),
}


@dataclass(frozen=True, slots=True)
class WorkflowInspection:
    """Small outcome-free view of the exact validated workflow prefix."""

    stage: RunStage
    selection_status: str | None = None
    selected_grid_cell_id: str | None = None

    def __post_init__(self) -> None:
        if type(self.stage) is not RunStage:
            raise TypeError("stage must be an exact RunStage")
        if self.selection_status not in {
            None,
            "SELECTED_METHOD_PROPOSED",
            "DEVELOPMENT_NO_GO",
        }:
            raise RunnerError("unknown development selection status")
        if self.selection_status == "SELECTED_METHOD_PROPOSED":
            if not isinstance(self.selected_grid_cell_id, str) or not self.selected_grid_cell_id:
                raise RunnerError("selected development state requires one grid-cell ID")
        elif self.selected_grid_cell_id is not None:
            raise RunnerError("non-selected development state cannot carry a grid-cell ID")

    @property
    def terminal(self) -> bool:
        return self.stage in _TERMINAL_STAGES

    @property
    def next_command(self) -> str | None:
        if self.stage is RunStage.SELECTION_HANDOFF:
            return "emit-selection"
        if self.terminal:
            return None
        return "resume"


class WorkflowFacade(Protocol):
    """Narrow seam used by the state machine and by side-effect-free unit fakes."""

    def inspect(self) -> WorkflowInspection:
        """Validate and reopen the complete existing durable prefix."""

    def advance(self, expected_stage: RunStage) -> WorkflowInspection:
        """Advance one logical durable phase.

        ``DEVELOPMENT_PENDING`` may create its start receipt and result in one call.
        """

    def selection_bytes(self) -> bytes:
        """Return exact selected-method artifact bytes without writing them."""


@dataclass(slots=True)
class _ValidatedGraph:
    presence: Mapping[str, bool]
    retired_v1: Any | None = None
    retired_v2: Any | None = None
    retired_v3: Any | None = None
    retired_v4: Any | None = None
    bundle: Any | None = None
    snapshot: Any | None = None
    reference_rng: Any | None = None
    reference_value: Any | None = None
    reference_proof: Any | None = None
    context_reference: Any | None = None
    development_start: Any | None = None
    development_result: Any | None = None
    audited_development_result: Any | None = None
    recovery: Any | None = None
    selected_method_freeze: Any | None = None
    test_evidence: Any | None = None
    canary_authorization: Any | None = None
    canary_claim: Any | None = None
    canary_seed: Any | None = None
    canary_beacon: Any | None = None
    published_canary_result: Any | None = None
    fresh_canary_audit: Any | None = None


def _require_activated_production_path(facts: _PreflightFacts) -> None:
    if type(facts) is not _PreflightFacts or facts._issuer is not _PREFLIGHT_ISSUER:
        raise RunnerError("production facade requires exact pre-import authority")
    if facts.config is not _PRODUCTION_PREFLIGHT_CONFIG:
        raise RunnerError("production facade rejects test preflight configuration")
    if tuple(sys.path) != EXPECTED_RUNNER_SYS_PATH:
        raise RunnerError("V3 post-preflight sys.path is not exact")


def _require_loaded_module_at_source(
    facts: _PreflightFacts,
    module: ModuleType,
    relative_path: str,
) -> None:
    expected = next(
        (item for item in facts.source.tracked_files if item.path == relative_path),
        None,
    )
    path = getattr(module, "__file__", None)
    canonical = facts.config.repository / relative_path
    if (
        expected is None
        or expected.kind != "regular_file"
        or path != os.fspath(canonical)
        or _hash_absolute_regular_file(
            canonical,
            label=f"loaded module {module.__name__}",
            require_single_link=True,
        )
        != expected.file_sha256
    ):
        raise RunnerError(f"loaded module is not exact committed source: {module.__name__}")


def _rfc3339_now_after(previous: str | None = None) -> str:
    now = datetime.now(timezone.utc).replace(microsecond=0)
    if previous is not None:
        if not isinstance(previous, str) or not _RFC3339_SECONDS_RE.fullmatch(previous):
            raise RunnerError("prior artifact timestamp is not strict RFC3339 seconds UTC")
        try:
            prior = datetime.fromisoformat(previous[:-1] + "+00:00")
        except ValueError as exc:
            raise RunnerError("prior artifact timestamp is malformed") from exc
        if prior.tzinfo is None:
            raise RunnerError("prior artifact timestamp lacks a timezone")
        if now <= prior:
            try:
                now = prior.astimezone(timezone.utc) + timedelta(seconds=1)
            except OverflowError as exc:
                raise RunnerError("prior artifact timestamp cannot be advanced") from exc
    return now.strftime("%Y-%m-%dT%H:%M:%SZ")


def _artifact_payload(capability: Any, name: str) -> Mapping[str, Any]:
    payload = getattr(capability, "_validated_payload", None)
    if not isinstance(payload, Mapping):
        raise RunnerError(f"{name} capability lacks validated payload bytes")
    return payload


class ProductionFacade:
    """Lazy adapter around the V3 core and governance public APIs."""

    _ARTIFACT_ORDER = (
        "bundle",
        "context_reference",
        "development_start",
        "development_result",
        "selected_method_freeze",
        "test_evidence",
        "canary_authorization",
        "canary_claim",
        "canary_beacon",
        "canary_result",
        "fresh_canary_audit",
    )

    def __init__(self, preflight: _PreflightFacts) -> None:
        _require_activated_production_path(preflight)
        # Governance is now a stdlib-only import.  It establishes the nominal
        # direct-script role before either numerical workflow module may import
        # anything from the frozen site-packages tree.
        import cfeg
        from cfeg import metadata_calibration_v3_governance as governance

        _require_loaded_module_at_source(preflight, cfeg, "src/cfeg/__init__.py")
        _require_loaded_module_at_source(
            preflight,
            governance,
            "src/cfeg/metadata_calibration_v3_governance.py",
        )
        if preflight.command is None:
            raise RunnerError("production preflight lacks a governed command role")
        expected_command = tuple(governance.governed_runner_command(preflight.command))
        if tuple(sys.orig_argv) != expected_command:
            raise RunnerError("governance and runner disagree on direct bootstrap argv")
        governed_process = governance.establish_governed_process(
            expected_python_site_inventory_sha256=str(
                preflight.python_site_inventory["inventory_sha256"]
            )
        )
        role_by_command = {
            "emit-selection": "runner_emit_selection",
            "resume": "runner_resume",
            "status": "runner_status",
        }
        governance.require_governed_process_capability(
            governed_process,
            expected_role=role_by_command[preflight.command],
        )
        _require_preflight_unchanged(preflight, include_site=False)

        from cfeg.analysis import metadata_calibration_v3_synthetic as synthetic
        from cfeg.models import metadata_calibration_v3 as model

        _require_loaded_module_at_source(
            preflight,
            synthetic,
            "src/cfeg/analysis/metadata_calibration_v3_synthetic.py",
        )
        _require_loaded_module_at_source(
            preflight,
            model,
            "src/cfeg/models/metadata_calibration_v3.py",
        )
        _require_preflight_unchanged(preflight)

        self._governance: ModuleType = governance
        self._model: ModuleType = model
        self._synthetic: ModuleType = synthetic
        self._preflight = preflight
        self._governed_process = governed_process
        self._inspection: WorkflowInspection | None = None
        self._graph: _ValidatedGraph | None = None

    def _artifact_paths(self) -> Mapping[str, Path]:
        governance = self._governance
        return {
            "bundle": Path(governance.DEVELOPMENT_BUNDLE_CANONICAL_PATH),
            "context_reference": Path(governance.CONTEXT_REFERENCE_CANONICAL_PATH),
            "development_start": Path(governance.DEVELOPMENT_START_CANONICAL_PATH),
            "development_result": Path(governance.DEVELOPMENT_RESULT_CANONICAL_PATH),
            "selected_method_freeze": (
                Path(governance.V3_SOURCE_REPOSITORY)
                / governance.SELECTED_METHOD_FREEZE_REPOSITORY_PATH
            ),
            "test_evidence": Path(governance.TEST_EVIDENCE_CANONICAL_PATH),
            "canary_authorization": Path(governance.CANARY_CANONICAL_ROOT) / "authorization.json",
            "canary_claim": Path(governance.CANARY_CANONICAL_ROOT) / "claim.json",
            "canary_beacon": Path(governance.CANARY_CANONICAL_ROOT) / "beacon.json",
            "canary_result": Path(governance.CANARY_RESULT_CANONICAL_PATH),
            "fresh_canary_audit": Path(governance.CANARY_FRESH_AUDIT_CANONICAL_PATH),
        }

    def _presence(self) -> Mapping[str, bool]:
        return {
            name: os.path.lexists(os.fspath(path)) for name, path in self._artifact_paths().items()
        }

    def _require_active_process(self) -> None:
        self._governance.require_governed_process_capability(
            self._governed_process,
        )

    def _require_bundle_matches_preflight(self, bundle: Any) -> None:
        payload = _artifact_payload(bundle, "development bundle")
        if self._preflight.bundle_payload is None:
            raise RunnerError("governance reopened a bundle absent from pre-import authority")
        if _canonical_json_bytes(payload) != _canonical_json_bytes(self._preflight.bundle_payload):
            raise RunnerError("governance bundle differs from stdlib pre-import validation")
        if _plain_json(getattr(bundle, "numerical_runtime_inventory", None)) != _plain_json(
            payload["numerical_runtime_inventory"]
        ) or payload["numerical_runtime_inventory"].get("python_site_inventory") != dict(
            self._preflight.python_site_inventory
        ):
            raise RunnerError("governance bundle runtime differs from pre-import authority")

    def _require_built_bundle_matches_preflight(
        self,
        value: Mapping[str, Any],
    ) -> None:
        facts = self._preflight
        if facts.bundle_payload is not None:
            raise RunnerError("cannot build a replacement for an existing development bundle")
        _validate_payload_self_hash(value, label="built development bundle")
        _require_v3_4_bundle_recovery_identity(
            value,
            canonical_path=facts.config.bundle_path,
        )
        required_source = [item.record() for item in facts.source.tracked_files]
        runtime = value.get("numerical_runtime_inventory")
        fingerprint = value.get("numerical_runtime_fingerprint_sha256")
        if not isinstance(runtime, Mapping):
            raise RunnerError("built bundle lacks a numerical runtime inventory")
        if (
            value.get("schema") != _DEVELOPMENT_BUNDLE_SCHEMA
            or value.get("candidate_id") != _CANDIDATE_ID
            or value.get("canonical_path") != os.fspath(facts.config.bundle_path)
            or value.get("clean_git_commit") != facts.source.commit
            or value.get("clean_git_tree") != facts.source.tree
            or value.get("source_bundle_sha256") != facts.source.source_bundle_sha256
            or _plain_json(value.get("tracked_source_file_inventory")) != required_source
            or value.get("scientific_lockbox_authorized") is not False
            or value.get("human_EEG_outcome_authorized") is not False
            or _plain_json(runtime.get("python")) != dict(facts.python)
            or _plain_json(runtime.get("python_site_inventory"))
            != dict(facts.python_site_inventory)
        ):
            raise RunnerError("built bundle differs from stdlib pre-import facts")
        expected_git = {
            "executable_file_sha256": facts.source.git_executable_file_sha256,
            "executable_path": os.fspath(V3_GIT_EXECUTABLE),
            "version": facts.source.git_version,
        }
        if _plain_json(runtime.get("git")) != expected_git:
            raise RunnerError("built bundle Git runtime differs from pre-import facts")
        if (
            not isinstance(fingerprint, str)
            or not _SHA256_RE.fullmatch(fingerprint)
            or _sha256_bytes(_canonical_json_bytes(runtime)) != fingerprint
        ):
            raise RunnerError("built bundle numerical runtime fingerprint is invalid")

    def _require_snapshot_matches_preflight(self, snapshot: Any) -> None:
        identity = getattr(snapshot, "identity", None)
        tracked = getattr(snapshot, "tracked_files", None)
        if identity is None or tracked is None:
            raise RunnerError("governance source snapshot lacks exact bindings")
        observed_records = [
            {
                "file_sha256": getattr(item, "file_sha256", None),
                "kind": getattr(item, "kind", None),
                "path": getattr(item, "path", None),
            }
            for item in tracked
        ]
        facts = self._preflight.source
        if (
            getattr(identity, "commit", None) != facts.commit
            or getattr(identity, "tree", None) != facts.tree
            or getattr(snapshot, "source_bundle_sha256", None) != facts.source_bundle_sha256
            or observed_records != [item.record() for item in facts.tracked_files]
        ):
            raise RunnerError("governance source snapshot differs from stdlib preflight")

    def _adopt_published_bundle(self, data: bytes) -> None:
        facts = self._preflight
        payload = _validate_existing_bundle(
            facts.config,
            data,
            source=facts.source,
            python=facts.python,
            site_inventory=facts.python_site_inventory,
        )
        self._preflight = _PreflightFacts(
            config=facts.config,
            command=facts.command,
            source=facts.source,
            python=facts.python,
            python_site_inventory=facts.python_site_inventory,
            bundle_payload=payload,
            _issuer=_PREFLIGHT_ISSUER,
        )

    def _require_no_gap_before(self, presence: Mapping[str, bool], name: str) -> None:
        index = self._ARTIFACT_ORDER.index(name)
        later = self._ARTIFACT_ORDER[index + 1 :]
        unexpected = tuple(item for item in later if presence[item])
        if unexpected:
            joined = ",".join(unexpected)
            raise RunnerError(f"durable artifact prefix has a gap before {name}: {joined}")

    def _cache(
        self,
        stage: RunStage,
        graph: _ValidatedGraph,
        *,
        selection_status: str | None = None,
        selected_grid_cell_id: str | None = None,
    ) -> WorkflowInspection:
        inspection = WorkflowInspection(
            stage=stage,
            selection_status=selection_status,
            selected_grid_cell_id=selected_grid_cell_id,
        )
        self._graph = graph
        self._inspection = inspection
        return inspection

    def _reopen_a_context(self, graph: _ValidatedGraph) -> None:
        model = self._model
        synthetic = self._synthetic
        governance = self._governance
        if graph.retired_v1 is None:
            raise RunnerError("context reopen lacks the exact retired V1 inventory")
        reference_rng = model.issue_bundle_bound_context_reference_rng_authority(
            graph.bundle,
            expected_commit=graph.bundle.clean_commit,
            expected_tree=graph.bundle.clean_tree,
        )
        payload = _detach_retired_v1_context_payload(governance, graph.retired_v1)
        replayed = synthetic.replay_context_reference_payload(rng_authority=reference_rng)
        if _canonical_json_bytes(replayed) != _canonical_json_bytes(payload):
            raise RunnerError("development-v5 reference differs from deterministic V1 replay")
        reference = model.validate_context_reference_payload(payload)
        proof = model.validate_context_reference_for_publication(
            payload,
            rng_authority=reference_rng,
        )
        artifact = governance.validate_context_reference(
            development_bundle=graph.bundle,
            rng_authority=reference_rng,
            core_capability=proof,
        )
        graph.reference_rng = reference_rng
        graph.reference_value = reference
        graph.reference_proof = proof
        graph.context_reference = artifact

    def _reopen_b_selection_graph(self, graph: _ValidatedGraph) -> None:
        governance = self._governance
        snapshot = governance.capture_clean_source_snapshot(governance.V3_SOURCE_REPOSITORY)
        recovery = governance.validate_bundle_selected_delta_recovery(
            development_bundle=graph.bundle,
            snapshot=snapshot,
        )
        context = governance.reopen_context_reference(
            development_bundle=graph.bundle,
            recovery_capability=recovery,
        )
        result = governance.reopen_development_result(
            development_bundle=graph.bundle,
            context_reference=context,
            recovery_capability=recovery,
        )
        selected = governance.reopen_selected_method_freeze(
            development_bundle=graph.bundle,
            development_result=result,
            recovery_capability=recovery,
        )
        result_payload = _artifact_payload(result, "development result")
        if result_payload.get("selection_status") != "SELECTED_METHOD_PROPOSED":
            raise RunnerError("selected B is impossible after DEVELOPMENT_NO_GO")
        graph.snapshot = snapshot
        graph.recovery = recovery
        graph.context_reference = context
        graph.development_result = result
        graph.selected_method_freeze = selected

    def _reopen_canary_seed(self, graph: _ValidatedGraph) -> Any:
        if graph.snapshot is None:
            raise RunnerError("canary seed requires selected-source snapshot")
        if graph.canary_seed is None:
            graph.canary_seed = self._governance.reopen_historical_canary_seed(
                snapshot=graph.snapshot
            )
        return graph.canary_seed

    def _bridge(self, graph: _ValidatedGraph) -> None:
        governance = self._governance
        required = (
            graph.bundle,
            graph.development_result,
            graph.selected_method_freeze,
            graph.test_evidence,
            graph.canary_authorization,
            graph.canary_claim,
            graph.canary_beacon,
            graph.published_canary_result,
            graph.fresh_canary_audit,
        )
        if any(item is None for item in required):
            raise RunnerError("fresh-audit bridge graph is incomplete")
        seed = self._reopen_canary_seed(graph)
        scientific = governance.ScientificLifecycle()
        scientific = scientific.advance(
            governance.ScientificState.BUNDLE_FROZEN,
            authority=graph.bundle,
        )
        scientific = scientific.advance(
            governance.ScientificState.DEVELOPMENT_COMPLETE,
            authority=graph.development_result,
        )
        scientific = scientific.advance(
            governance.ScientificState.SELECTED_METHOD_FROZEN,
            authority=graph.selected_method_freeze,
        )
        scientific = scientific.advance(
            governance.ScientificState.VERIFIED,
            authority=graph.test_evidence,
        )
        canary = governance.CanaryLifecycle()
        canary = canary.advance(
            governance.CanaryState.CANARY_AUTHORIZED,
            authority=graph.canary_authorization,
        )
        canary = canary.advance(
            governance.CanaryState.CANARY_CLAIMED,
            authority=graph.canary_claim,
        )
        canary = canary.advance(
            governance.CanaryState.CANARY_BEACON_BOUND,
            authority=graph.canary_beacon,
        )
        canary = canary.advance(
            governance.CanaryState.CANARY_PAYLOAD_VALIDATED,
            authority=seed,
        )
        canary = canary.advance(
            governance.CanaryState.CANARY_RESULT_PUBLISHED,
            authority=graph.published_canary_result,
        )
        canary = canary.advance(
            governance.CanaryState.CANARY_FRESH_AUDITED,
            authority=graph.fresh_canary_audit,
        )
        bridged = governance.bridge_canary_audit(
            scientific,
            canary,
            graph.fresh_canary_audit,
        )
        if bridged.state is not governance.ScientificState.CANARY_PASSED:
            raise RunnerError("fresh canary audit did not reach CANARY_PASSED")

    def inspect(self) -> WorkflowInspection:
        governance = self._governance
        presence = self._presence()
        graph = _ValidatedGraph(presence=presence)
        (
            graph.retired_v1,
            graph.retired_v2,
            graph.retired_v3,
            graph.retired_v4,
        ) = governance.observe_retired_development_attempts()

        if not presence["bundle"]:
            self._require_no_gap_before(presence, "bundle")
            return self._cache(RunStage.BUNDLE_PENDING, graph)
        graph.bundle = governance.reopen_development_bundle()
        self._require_bundle_matches_preflight(graph.bundle)

        if not presence["context_reference"]:
            self._require_no_gap_before(presence, "context_reference")
            return self._cache(RunStage.CONTEXT_REFERENCE_PENDING, graph)
        if not presence["development_start"]:
            self._require_no_gap_before(presence, "development_start")
            self._reopen_a_context(graph)
            return self._cache(RunStage.DEVELOPMENT_PENDING, graph)
        self._reopen_a_context(graph)
        graph.development_start = governance.reopen_development_start_receipt(
            development_bundle=graph.bundle,
            context_reference=graph.context_reference,
        )
        if not presence["development_result"]:
            self._require_no_gap_before(presence, "development_result")
            return self._cache(RunStage.DEVELOPMENT_ATTEMPT_CONSUMED, graph)

        if not presence["selected_method_freeze"]:
            self._require_no_gap_before(presence, "selected_method_freeze")
            result, audited = governance.recover_canonical_development_result_at_bundle_source(
                development_bundle=graph.bundle,
                context_reference=graph.context_reference,
            )
            graph.development_result = result
            graph.audited_development_result = audited
            selection_status = getattr(audited, "selection_status", None)
            selected_grid_cell_id = getattr(audited, "selected_grid_cell_id", None)
            if selection_status == "DEVELOPMENT_NO_GO":
                scientific = governance.ScientificLifecycle()
                scientific = scientific.advance(
                    governance.ScientificState.BUNDLE_FROZEN,
                    authority=graph.bundle,
                )
                scientific = scientific.advance(
                    governance.ScientificState.DEVELOPMENT_COMPLETE,
                    authority=result,
                )
                scientific = scientific.advance(
                    governance.ScientificState.DEVELOPMENT_NO_GO,
                    authority=result,
                )
                if scientific.state is not governance.ScientificState.DEVELOPMENT_NO_GO:
                    raise RunnerError("development no-go lifecycle validation failed")
                return self._cache(
                    RunStage.DEVELOPMENT_NO_GO,
                    graph,
                    selection_status=selection_status,
                )
            if selection_status != "SELECTED_METHOD_PROPOSED":
                raise RunnerError("canonical development result has unknown selection status")
            return self._cache(
                RunStage.SELECTION_HANDOFF,
                graph,
                selection_status=selection_status,
                selected_grid_cell_id=selected_grid_cell_id,
            )

        self._reopen_b_selection_graph(graph)
        selected_id = _artifact_payload(graph.development_result, "development result")[
            "selected_grid_cell_id"
        ]

        if not presence["test_evidence"]:
            self._require_no_gap_before(presence, "test_evidence")
            return self._cache(
                RunStage.TEST_EVIDENCE_PENDING,
                graph,
                selection_status="SELECTED_METHOD_PROPOSED",
                selected_grid_cell_id=selected_id,
            )
        graph.test_evidence = governance.reopen_test_evidence(
            selected_method_freeze=graph.selected_method_freeze,
            snapshot=graph.snapshot,
        )

        if not presence["canary_authorization"]:
            self._require_no_gap_before(presence, "canary_authorization")
            return self._cache(
                RunStage.CANARY_AUTHORIZATION_PENDING,
                graph,
                selection_status="SELECTED_METHOD_PROPOSED",
                selected_grid_cell_id=selected_id,
            )
        graph.canary_authorization = governance.validate_canary_authorization(
            selected_method_freeze=graph.selected_method_freeze,
            test_evidence=graph.test_evidence,
        )

        if not presence["canary_claim"]:
            self._require_no_gap_before(presence, "canary_claim")
            return self._cache(
                RunStage.CANARY_CLAIM_PENDING,
                graph,
                selection_status="SELECTED_METHOD_PROPOSED",
                selected_grid_cell_id=selected_id,
            )
        graph.canary_claim = governance.validate_canary_claim(
            authorization=graph.canary_authorization
        )

        if not presence["canary_beacon"]:
            self._require_no_gap_before(presence, "canary_beacon")
            return self._cache(
                RunStage.CANARY_BEACON_PENDING,
                graph,
                selection_status="SELECTED_METHOD_PROPOSED",
                selected_grid_cell_id=selected_id,
            )
        seed = self._reopen_canary_seed(graph)
        graph.canary_beacon = governance.validate_canary_beacon(
            claim=graph.canary_claim,
            seed=seed,
        )

        if not presence["canary_result"]:
            self._require_no_gap_before(presence, "canary_result")
            return self._cache(
                RunStage.CANARY_RESULT_PENDING,
                graph,
                selection_status="SELECTED_METHOD_PROPOSED",
                selected_grid_cell_id=selected_id,
            )
        graph.published_canary_result = governance.reopen_published_canary_result(
            beacon=graph.canary_beacon,
            seed=seed,
        )

        if not presence["fresh_canary_audit"]:
            return self._cache(
                RunStage.FRESH_CANARY_AUDIT_PENDING,
                graph,
                selection_status="SELECTED_METHOD_PROPOSED",
                selected_grid_cell_id=selected_id,
            )
        # The sole high-level fresh-exec API first reopens an existing exact
        # artifact; it cannot republish because the path is already consumed.
        graph.fresh_canary_audit = governance.run_fresh_canary_audit_in_exec_subprocess()
        self._bridge(graph)
        return self._cache(
            RunStage.CANARY_PASSED,
            graph,
            selection_status="SELECTED_METHOD_PROPOSED",
            selected_grid_cell_id=selected_id,
        )

    def _require_cached_stage(self, expected_stage: RunStage) -> _ValidatedGraph:
        if expected_stage not in _ADVANCEABLE_STAGES:
            raise RunnerError(f"stage cannot create a durable artifact: {expected_stage.value}")
        if self._inspection is None or self._graph is None:
            raise RunnerError("advance requires a preceding exact inspection")
        if self._inspection.stage is not expected_stage:
            raise RunnerError(
                f"workflow changed before advance: expected {expected_stage.value}, "
                f"observed {self._inspection.stage.value}"
            )
        return self._graph

    def _publish_bundle(self, graph: _ValidatedGraph) -> None:
        del graph
        governance = self._governance
        self._require_active_process()
        _require_preflight_unchanged(self._preflight)
        snapshot = governance.capture_clean_source_snapshot(governance.V3_SOURCE_REPOSITORY)
        self._require_snapshot_matches_preflight(snapshot)
        focused = governance.run_observed_test("focused_v3", snapshot=snapshot)
        frozen = {
            path: (Path(governance.V3_SOURCE_REPOSITORY) / path).read_bytes()
            for path in governance.FROZEN_FILE_SHA256
        }
        created_at = _rfc3339_now_after()
        built = governance.build_development_bundle(
            snapshot=snapshot,
            frozen_file_bytes=frozen,
            created_at_UTC=created_at,
            focused_test_run=focused,
        )
        self._require_built_bundle_matches_preflight(built)
        _require_preflight_unchanged(self._preflight)
        governance.publish_development_bundle(
            snapshot=snapshot,
            frozen_file_bytes=frozen,
            created_at_UTC=created_at,
            focused_test_run=focused,
        )
        published = _read_optional_artifact(self._preflight.config.bundle_path)
        if published is None:
            raise RunnerError("development bundle publication returned without exact bytes")
        if published != _canonical_json_bytes(built) + b"\n":
            raise RunnerError("published development bundle differs from prevalidated bytes")
        self._adopt_published_bundle(published)
        governance.validate_development_bundle(
            snapshot=snapshot,
            frozen_file_bytes=frozen,
            focused_test_run=focused,
        )
        _require_preflight_unchanged(self._preflight)

    def _publish_context_reference(self, graph: _ValidatedGraph) -> None:
        if graph.bundle is None or graph.retired_v1 is None:
            raise RunnerError("context-reference phase lacks bundle or retired V1 receipt")
        model = self._model
        synthetic = self._synthetic
        governance = self._governance
        reference_rng = model.issue_bundle_bound_context_reference_rng_authority(
            graph.bundle,
            expected_commit=graph.bundle.clean_commit,
            expected_tree=graph.bundle.clean_tree,
        )
        payload = _detach_retired_v1_context_payload(governance, graph.retired_v1)
        replayed = synthetic.replay_context_reference_payload(rng_authority=reference_rng)
        if _canonical_json_bytes(replayed) != _canonical_json_bytes(payload):
            raise RunnerError("retired V1 reference is not the exact deterministic replay")
        proof = model.validate_context_reference_for_publication(
            payload,
            rng_authority=reference_rng,
        )
        governance.publish_context_reference(
            payload,
            development_bundle=graph.bundle,
            rng_authority=reference_rng,
            core_capability=proof,
        )
        governance.validate_context_reference(
            development_bundle=graph.bundle,
            rng_authority=reference_rng,
            core_capability=proof,
        )

    def _publish_development_result(self, graph: _ValidatedGraph) -> None:
        if any(
            item is None
            for item in (
                graph.bundle,
                graph.reference_value,
                graph.reference_proof,
                graph.context_reference,
            )
        ):
            raise RunnerError("development phase lacks an exact validated A graph")
        synthetic = self._synthetic
        model = self._model
        governance = self._governance
        contract = synthetic.validate_synthetic_contract()
        synthetic.validate_development_contract(contract)
        _require_preflight_unchanged(self._preflight)
        development_start = governance.publish_development_start(
            development_bundle=graph.bundle,
            context_reference=graph.context_reference,
            started_at_UTC=_rfc3339_now_after(),
        )
        graph.development_start = development_start
        development_rng = model.issue_bundle_bound_development_rng_authority(
            graph.bundle,
            graph.context_reference,
            development_start,
            expected_commit=graph.bundle.clean_commit,
            expected_tree=graph.bundle.clean_tree,
        )
        payload, proof = synthetic.execute_complete_development(
            contract,
            graph.reference_value,
            rng_authority=development_rng,
            validated_reference=graph.reference_proof,
            context_reference_file_sha256=graph.context_reference.file_sha256,
        )
        governance.publish_development_result(
            payload,
            development_bundle=graph.bundle,
            context_reference=graph.context_reference,
            development_start=development_start,
            development_rng_authority=development_rng,
            validated_context_reference=graph.reference_proof,
            core_capability=proof,
        )
        governance.validate_development_result(
            development_bundle=graph.bundle,
            context_reference=graph.context_reference,
            development_start=development_start,
            development_rng_authority=development_rng,
            validated_context_reference=graph.reference_proof,
            core_capability=proof,
        )

    def _publish_test_evidence(self, graph: _ValidatedGraph) -> None:
        if graph.snapshot is None or graph.selected_method_freeze is None:
            raise RunnerError("test-evidence phase lacks exact selected B")
        governance = self._governance
        focused = governance.run_observed_test("focused_v3", snapshot=graph.snapshot)
        full = governance.run_observed_test("repository_full", snapshot=graph.snapshot)
        runs = (focused, full)
        payload = governance.seal_test_evidence(
            selected_method_freeze=graph.selected_method_freeze,
            observed_runs=runs,
            snapshot=graph.snapshot,
            recorded_at_UTC=_rfc3339_now_after(),
        )
        governance.publish_test_evidence(
            payload,
            selected_method_freeze=graph.selected_method_freeze,
            observed_runs=runs,
            snapshot=graph.snapshot,
        )
        governance.validate_test_evidence(
            selected_method_freeze=graph.selected_method_freeze,
            observed_runs=runs,
            snapshot=graph.snapshot,
        )

    def _publish_canary_authorization(self, graph: _ValidatedGraph) -> None:
        if graph.selected_method_freeze is None or graph.test_evidence is None:
            raise RunnerError("canary authorization phase lacks selected test evidence")
        governance = self._governance
        tests = _artifact_payload(graph.test_evidence, "test evidence")
        payload = governance.build_canary_authorization(
            selected_method_freeze=graph.selected_method_freeze,
            test_evidence=graph.test_evidence,
            authorized_at_UTC=_rfc3339_now_after(str(tests["recorded_at_UTC"])),
        )
        governance.publish_canary_authorization(
            payload,
            selected_method_freeze=graph.selected_method_freeze,
            test_evidence=graph.test_evidence,
        )
        governance.validate_canary_authorization(
            selected_method_freeze=graph.selected_method_freeze,
            test_evidence=graph.test_evidence,
        )

    def _publish_canary_claim(self, graph: _ValidatedGraph) -> None:
        if graph.canary_authorization is None:
            raise RunnerError("canary claim phase lacks authorization")
        governance = self._governance
        authorization = _artifact_payload(graph.canary_authorization, "canary authorization")
        payload = governance.build_canary_claim(
            graph.canary_authorization,
            claimed_at_UTC=_rfc3339_now_after(str(authorization["authorized_at_UTC"])),
        )
        governance.publish_canary_claim(
            payload,
            authorization=graph.canary_authorization,
        )
        governance.validate_canary_claim(authorization=graph.canary_authorization)

    def _publish_canary_beacon(self, graph: _ValidatedGraph) -> None:
        if graph.canary_claim is None:
            raise RunnerError("canary beacon phase lacks claim")
        governance = self._governance
        seed = self._reopen_canary_seed(graph)
        payload = governance.build_canary_beacon(graph.canary_claim, seed)
        governance.publish_canary_beacon(
            payload,
            claim=graph.canary_claim,
            seed=seed,
        )
        governance.validate_canary_beacon(claim=graph.canary_claim, seed=seed)

    def _publish_canary_result(self, graph: _ValidatedGraph) -> None:
        if graph.canary_beacon is None:
            raise RunnerError("canary result phase lacks beacon")
        governance = self._governance
        seed = self._reopen_canary_seed(graph)
        probe = governance.canary_probe_digest(
            seed,
            prevalidation_trace=tuple(governance.CanaryState)[:4],
        )
        governance.publish_canary_result(
            beacon=graph.canary_beacon,
            seed=seed,
            probe_digest_sha256=probe,
        )

    def _publish_fresh_canary_audit(self, graph: _ValidatedGraph) -> None:
        if graph.published_canary_result is None:
            raise RunnerError("fresh audit phase lacks published canary result")
        # Do not call, serialize, or reproduce any private child protocol here.
        graph.fresh_canary_audit = self._governance.run_fresh_canary_audit_in_exec_subprocess()
        self._bridge(graph)

    def advance(self, expected_stage: RunStage) -> WorkflowInspection:
        graph = self._require_cached_stage(expected_stage)
        self._require_active_process()
        _require_preflight_unchanged(self._preflight)
        dispatch = {
            RunStage.BUNDLE_PENDING: self._publish_bundle,
            RunStage.CONTEXT_REFERENCE_PENDING: self._publish_context_reference,
            RunStage.DEVELOPMENT_PENDING: self._publish_development_result,
            RunStage.TEST_EVIDENCE_PENDING: self._publish_test_evidence,
            RunStage.CANARY_AUTHORIZATION_PENDING: self._publish_canary_authorization,
            RunStage.CANARY_CLAIM_PENDING: self._publish_canary_claim,
            RunStage.CANARY_BEACON_PENDING: self._publish_canary_beacon,
            RunStage.CANARY_RESULT_PENDING: self._publish_canary_result,
            RunStage.FRESH_CANARY_AUDIT_PENDING: self._publish_fresh_canary_audit,
        }
        dispatch[expected_stage](graph)
        # Reopen and validate the just-published artifact from its canonical
        # location.  If publication returned only partially, this fails closed.
        return self.inspect()

    def selection_bytes(self) -> bytes:
        inspection = self.inspect()
        if inspection.stage is not RunStage.SELECTION_HANDOFF or self._graph is None:
            raise RunnerError("selection bytes are available only at SELECTION_HANDOFF")
        graph = self._graph
        if graph.development_result is None or graph.audited_development_result is None:
            raise RunnerError("selection handoff lacks exact A recovery proof")
        selected = self._governance.build_selected_method_freeze_from_canonical_development_result(
            development_result=graph.development_result,
            audited_development_result=graph.audited_development_result,
        )
        return self._governance.artifact_bytes(selected)


def status_report(facade: WorkflowFacade) -> Mapping[str, Any]:
    inspection = facade.inspect()
    return {
        "command": "status",
        "stage": inspection.stage.value,
        "terminal": inspection.terminal,
        "selection_status": inspection.selection_status,
        "selected_grid_cell_id": inspection.selected_grid_cell_id,
        "next_command": inspection.next_command,
        "durable_phases_advanced": 0,
    }


def resume_once(facade: WorkflowFacade) -> Mapping[str, Any]:
    before = facade.inspect()
    if before.stage is RunStage.SELECTION_HANDOFF or before.terminal:
        after = before
        advanced = 0
    else:
        if before.stage not in _ADVANCEABLE_STAGES:
            raise RunnerError(f"unhandled workflow stage: {before.stage.value}")
        after = facade.advance(before.stage)
        if after.stage not in _RESUME_TRANSITIONS[before.stage]:
            raise RunnerError(
                f"resume crossed an invalid durable boundary: "
                f"{before.stage.value}->{after.stage.value}"
            )
        advanced = 1
    return {
        "command": "resume",
        "stage_before": before.stage.value,
        "stage": after.stage.value,
        "terminal": after.terminal,
        "selection_status": after.selection_status,
        "selected_grid_cell_id": after.selected_grid_cell_id,
        "next_command": after.next_command,
        "durable_phases_advanced": advanced,
    }


def emit_selection_bytes(facade: WorkflowFacade) -> bytes:
    inspection = facade.inspect()
    if inspection.stage is not RunStage.SELECTION_HANDOFF:
        raise RunnerError("emit-selection requires the exact A selection handoff")
    data = facade.selection_bytes()
    if not isinstance(data, bytes) or not data or not data.endswith(b"\n"):
        raise RunnerError("selection handoff did not return canonical artifact bytes")
    return data


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="run_metadata_calibration_v3",
        description="Resume exactly one frozen V3 synthetic/governance phase.",
        allow_abbrev=False,
    )
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("status", allow_abbrev=False)
    commands.add_parser("resume", allow_abbrev=False)
    commands.add_parser("emit-selection", allow_abbrev=False)
    return parser


def _report_bytes(value: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(
            dict(value),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("utf-8")
        + b"\n"
    )


def run_cli(
    argv: Sequence[str],
    *,
    facade: WorkflowFacade,
    text_stdout: TextIO,
    binary_stdout: Any,
) -> int:
    args = _build_parser().parse_args(tuple(argv))
    if args.command == "status":
        text_stdout.write(_report_bytes(status_report(facade)).decode("ascii"))
    elif args.command == "resume":
        text_stdout.write(_report_bytes(resume_once(facade)).decode("ascii"))
    elif args.command == "emit-selection":
        binary_stdout.write(emit_selection_bytes(facade))
        binary_stdout.flush()
    else:  # pragma: no cover - argparse owns the closed command vocabulary
        raise RunnerError("unknown runner command")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    arguments = tuple(sys.argv[1:] if argv is None else argv)
    try:
        preflight = _run_preimport_preflight(
            _PRODUCTION_PREFLIGHT_CONFIG,
            executing_file=Path(__file__),
        )
        if argv is not None and arguments != tuple(sys.argv[1:]):
            raise RunnerError("production main arguments must be the exact process argv")
        _activate_exact_import_path(preflight)
        facade = ProductionFacade(preflight)
        return run_cli(
            arguments,
            facade=facade,
            text_stdout=sys.stdout,
            binary_stdout=sys.stdout.buffer,
        )
    except (RunnerError, OSError, ValueError, TypeError, RuntimeError) as exc:
        error = {
            "command": arguments[0] if arguments else None,
            "error": type(exc).__name__,
            "message": str(exc),
            "status": "FAIL_CLOSED",
        }
        sys.stderr.write(_report_bytes(error).decode("ascii"))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "EXPECTED_RUNNER_SYS_PATH",
    "ProductionFacade",
    "RunStage",
    "RunnerError",
    "WorkflowFacade",
    "WorkflowInspection",
    "emit_selection_bytes",
    "main",
    "resume_once",
    "run_cli",
    "status_report",
]
