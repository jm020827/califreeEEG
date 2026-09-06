"""Outcome-free governance primitives for metadata-calibration V3.

This module is deliberately standalone.  It does not import any V1/V2 runtime,
does not fetch a beacon, and does not grant scientific or human-outcome access.
Pure builders are usable by unit tests; publication and execution remain behind
explicit, validated capabilities.
"""

from __future__ import annotations

import base64
import binascii
import contextlib
import hashlib
import importlib
import importlib.metadata
import importlib.util
import io
import json
import math
import os
import platform
import re
import select
import shutil
import stat
import subprocess
import sys
import sysconfig
import tempfile
import time
import xml.etree.ElementTree as ET
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path, PurePosixPath
from types import MappingProxyType
from typing import TYPE_CHECKING, Any
from urllib.parse import urlsplit

if TYPE_CHECKING:
    import numpy as np

_GOVERNANCE_IMPORT_SOURCE_PATH = Path(__file__).resolve(strict=True)
_GOVERNANCE_IMPORT_SOURCE_FILE_SHA256 = hashlib.sha256(
    _GOVERNANCE_IMPORT_SOURCE_PATH.read_bytes()
).hexdigest()

CANDIDATE_ID = "metadata-calibration-efficiency-v3"
CANARY_CANDIDATE_ID = "metadata-calibration-efficiency-v3-governance-canary"

DEVELOPMENT_BUNDLE_SCHEMA = "cfeg.metadata-calibration-efficiency-v3.development-bundle.v3"
RETIRED_V1_DEVELOPMENT_BUNDLE_SCHEMA = (
    "cfeg.metadata-calibration-efficiency-v3.development-bundle.v1"
)
RETIRED_V2_DEVELOPMENT_BUNDLE_SCHEMA = (
    "cfeg.metadata-calibration-efficiency-v3.development-bundle.v2"
)
RECOVERY_AMENDMENT_SCHEMA = (
    "cfeg.metadata-calibration-efficiency-v3.preoutcome-recovery-amendment.v2"
)
PROTOCOL_REVISION = "V3.2"
DEVELOPMENT_ATTEMPT_ID = "development-v3"
CANARY_SEED_CAPABILITY_SCHEMA = "cfeg.metadata-calibration-efficiency-v3.canary-seed-capability.v1"
FRESH_CANARY_AUDIT_SCHEMA = "cfeg.metadata-calibration-efficiency-v3.canary-fresh-audit.v1"
FRESH_CANARY_AUDIT_CAPABILITY_SCHEMA = (
    "cfeg.metadata-calibration-efficiency-v3.fresh-canary-audit-capability.v1"
)
ATTEMPT_MANIFEST_SCHEMA = "cfeg.metadata-calibration-efficiency-v3.attempt-manifest.v1"
EXECUTION_AUTHORIZATION_SCHEMA = (
    "cfeg.metadata-calibration-efficiency-v3.execution-authorization.v1"
)
EXECUTION_AUTHORIZATION_PAYLOAD_SCHEMA = (
    "cfeg.metadata-calibration-efficiency-v3.execution-authorization-payload.v1"
)
GLOBAL_CLAIM_SCHEMA = "cfeg.metadata-calibration-efficiency-v3.claim.v1"
SCIENTIFIC_SEED_CAPABILITY_SCHEMA = (
    "cfeg.metadata-calibration-efficiency-v3.scientific-seed-capability.v1"
)
SEED_STATEMENT_SCHEMA = "cfeg.metadata-calibration-efficiency-v3.seed-statement.v1"
BEACON_SEED_DERIVATION_SCHEMA = "cfeg.metadata-calibration-efficiency-v3.beacon-seed-derivation.v1"

ARTIFACT_SCHEMAS = MappingProxyType(
    {
        "canary_authorization": ("cfeg.metadata-calibration-efficiency-v3.canary-authorization.v1"),
        "canary_beacon": "cfeg.metadata-calibration-efficiency-v3.canary-beacon.v1",
        "canary_claim": "cfeg.metadata-calibration-efficiency-v3.canary-claim.v1",
        "canary_fresh_audit": FRESH_CANARY_AUDIT_SCHEMA,
        "canary_result": "cfeg.metadata-calibration-efficiency-v3.canary-result.v1",
        "canary_seed_capability": CANARY_SEED_CAPABILITY_SCHEMA,
        "context_preflight_capability": (
            "cfeg.metadata-calibration-efficiency-v3.context-preflight-capability.v1"
        ),
        "context_reference": "cfeg.metadata-calibration-efficiency-v3.context-reference.v1",
        "context_trust_capability": (
            "cfeg.metadata-calibration-efficiency-v3.context-trust-capability.v1"
        ),
        "development_bundle": DEVELOPMENT_BUNDLE_SCHEMA,
        "development_start": "cfeg.metadata-calibration-efficiency-v3.development-start.v1",
        "development_result": "cfeg.metadata-calibration-efficiency-v3.development-result.v1",
        "fresh_canary_audit_capability": FRESH_CANARY_AUDIT_CAPABILITY_SCHEMA,
        "scientific_attempt_manifest": ATTEMPT_MANIFEST_SCHEMA,
        "scientific_beacon": "cfeg.metadata-calibration-efficiency-v3.beacon.v1",
        "scientific_claim": GLOBAL_CLAIM_SCHEMA,
        "scientific_execution_authorization": EXECUTION_AUTHORIZATION_SCHEMA,
        "scientific_result": ("cfeg.metadata-calibration-efficiency-v3.synthetic-result.v1"),
        "scientific_seed_capability": SCIENTIFIC_SEED_CAPABILITY_SCHEMA,
        "scientific_terminal": "cfeg.metadata-calibration-efficiency-v3.terminal.v1",
        "selected_method_freeze": (
            "cfeg.metadata-calibration-efficiency-v3.selected-method-freeze.v1"
        ),
        "support_reliability_capability": (
            "cfeg.metadata-calibration-efficiency-v3.support-reliability-capability.v1"
        ),
        "test_evidence": "cfeg.metadata-calibration-efficiency-v3.test-evidence.v1",
    }
)

MASTER_PLAN_PATH = "configs/analysis/metadata_calibration_efficiency_v3.yaml"
SYNTHETIC_PLAN_PATH = "configs/analysis/metadata_calibration_v3_synthetic.yaml"
PREOUTCOME_AMENDMENT_PATH = "configs/governance/metadata_calibration_v3_2_recovery_amendment.json"
PRIOR_RECOVERY_AMENDMENT_PATH = (
    "configs/governance/metadata_calibration_v3_1_recovery_amendment.json"
)
ORIGINAL_PREOUTCOME_AMENDMENT_PATH = (
    "configs/governance/metadata_calibration_v3_preoutcome_amendment.json"
)
OWNER_AUTHORITY_PUBLIC_KEY_PATH = "configs/governance/metadata_calibration_v3_owner_authority.pub"
V2_TERMINAL_AUDIT_PATH = (
    "configs/governance/metadata_calibration_v2_synthetic_v11_terminal_audit.json"
)
V2_DENY_OVERLAY_PATH = "configs/governance/metadata_calibration_v2_terminal_deny_overlay.json"

FROZEN_FILE_SHA256 = MappingProxyType(
    {
        MASTER_PLAN_PATH: "ff82c78ab6765c2bf58062195d246df7eb1654cabd1eb413fa4c40ca1f461446",
        SYNTHETIC_PLAN_PATH: "172de79a5315ef6daba9d623f05ee6ebe37595fc8f9dc3c913b6f63c8119f693",
        PREOUTCOME_AMENDMENT_PATH: (
            "1db2d118376975c1817e329f4203dbaf9929d4f2bcb30bfede990af26c2c1ceb"
        ),
        PRIOR_RECOVERY_AMENDMENT_PATH: (
            "deddca9286f07c6293925409d01b3df1313e89f2ce89688c4a25c14a8ec01238"
        ),
        ORIGINAL_PREOUTCOME_AMENDMENT_PATH: (
            "3238761a0a9a257032d6582286953a4d20981e5e646c9f1322c5c0357bd0c202"
        ),
        OWNER_AUTHORITY_PUBLIC_KEY_PATH: (
            "ce868830f9c47948a6abe2068863c8f39b12cba470e5493ff32a7c3c7178374b"
        ),
        V2_TERMINAL_AUDIT_PATH: (
            "46db1fceb67b98157d25c06afb4aae15c2436c3b57cc5fbf691c290a637aa4a0"
        ),
        V2_DENY_OVERLAY_PATH: ("dd4541765b6b021f4dc4c83fa8fc0b7c2eb76b997a093a0cff479f4fa9b3f7d0"),
    }
)
OWNER_AUTHORITY_FINGERPRINT = "SHA256:tm6CDH5eVtjTKNqBUwrBYwbq5RhZ48wo1QjP9c+mR+g"
OWNER_SIGNATURE_DOMAIN = "cfeg-metadata-calibration-v3-execution-authorization"
OWNER_SIGNER_ROLE = "active_workspace_owner"

CANARY_FIXTURE_SHA256 = "bccc6bb11467ecb13b535966a9e22de219294a586a41c3bd716ddc64b2bc6646"
CANARY_FIXTURE_REPOSITORY_PATH = PurePosixPath(
    "tests/fixtures/nist_beacon_v2_20260905T180000Z.json"
)
CANARY_TIMESTAMP_UTC = "2026-09-05T18:00:00.000Z"
CANARY_CHAIN_INDEX = 2
CANARY_PULSE_INDEX = 1_928_426
CANARY_OUTPUT_VALUE = (
    "BADC50F6F9E38477950DA01DBF00C0F637FEB4A2F5A1BC9232B0CCF084DE2612"
    "5539608A6138566AB5B904B816229D3D9B322B7F0B894C35EDAD1C4A057BEB03"
)
CANARY_SEED_DOMAIN = "cfeg.metadata-calibration-efficiency-v3.canary-seed.v1"
CANARY_SEED_DIGEST = "137bf8d1430e12af656bc01bb9a27f8223d24822394632ec432bc287db42b4bf"
CANARY_ROOT_SEED = int(CANARY_SEED_DIGEST, 16)
CANARY_PROBE_DOMAIN = "cfeg.metadata-calibration-efficiency-v3.canary-probe.v1"
CANARY_PROBE_DIGEST = "9b1d33abdf86747e7e384a1b656e62f017d6fb094b9e0127634c54eed77dc4d2"
SCIENTIFIC_SEED_DOMAIN = "cfeg.metadata-calibration-efficiency-v3.scientific-seed.v1"

SCIENTIFIC_LOCKBOX_AUTHORIZED = False
HUMAN_EEG_OUTCOME_AUTHORIZED = False
FUTURE_BEACON_SELECTED = False

V3_CANONICAL_ROOT = Path("/home/whwovy/v3-artifacts/metadata-calibration-efficiency-v3")
V3_SOURCE_REPOSITORY = Path("/home/whwovy/califreeEEG")
V3_PYTHON_EXECUTABLE = Path("/home/whwovy/califreeEEG/.venv/bin/python")
V3_SITE_PACKAGES = Path("/home/whwovy/califreeEEG/.venv/lib/python3.10/site-packages")
V3_GIT_EXECUTABLE = Path("/usr/bin/git")
_GIT_IMPORT_EXECUTABLE_FILE_SHA256 = hashlib.sha256(V3_GIT_EXECUTABLE.read_bytes()).hexdigest()
CANARY_CANONICAL_ROOT = V3_CANONICAL_ROOT / "governance-canary-v1"
CANARY_RESULT_CANONICAL_PATH = CANARY_CANONICAL_ROOT / "result.json"
CANARY_FRESH_AUDIT_CANONICAL_PATH = CANARY_CANONICAL_ROOT / "fresh-audit.json"
SCIENTIFIC_GLOBAL_CLAIM_CANONICAL_PATH = V3_CANONICAL_ROOT / "scientific/global-claim.json"
RETIRED_V1_DEVELOPMENT_ROOT = V3_CANONICAL_ROOT / "development-v1"
RETIRED_V1_DEVELOPMENT_BUNDLE_PATH = RETIRED_V1_DEVELOPMENT_ROOT / "development-bundle.json"
RETIRED_V1_CONTEXT_REFERENCE_PATH = RETIRED_V1_DEVELOPMENT_ROOT / "context-reference.json"
RETIRED_V2_DEVELOPMENT_ROOT = V3_CANONICAL_ROOT / "development-v2"
RETIRED_V2_DEVELOPMENT_BUNDLE_PATH = RETIRED_V2_DEVELOPMENT_ROOT / "development-bundle.json"
DEVELOPMENT_ROOT = V3_CANONICAL_ROOT / DEVELOPMENT_ATTEMPT_ID
CONTEXT_REFERENCE_CANONICAL_PATH = DEVELOPMENT_ROOT / "context-reference.json"
DEVELOPMENT_START_CANONICAL_PATH = DEVELOPMENT_ROOT / "development-start.json"
DEVELOPMENT_RESULT_CANONICAL_PATH = DEVELOPMENT_ROOT / "development-result.json"
RETIRED_V1_SOURCE_COMMIT = "569394da6894a2efed37b9dde162cbe4fe60534d"
RETIRED_V1_SOURCE_TREE = "fe472b6ae07d2a95d87d8ae5f3ec266a5347e8dd"
RETIRED_V1_BUNDLE_PAYLOAD_SHA256 = (
    "959ce56a97e3ece034b52c44a993d393ef168bbb1bdbad9a72dee86cdf3430b3"
)
RETIRED_V1_BUNDLE_FILE_SHA256 = "9f6027918110cfd877a1b170f3472d897d4b06f6fca7f0991b9f15f6768c4a1a"
RETIRED_V1_CONTEXT_PAYLOAD_SHA256 = (
    "c475e9d0ce4f8e436b49c50585eaee37965bba7c052eefdee862bc50f70a9253"
)
RETIRED_V1_CONTEXT_FILE_SHA256 = "7b78b3092f5824b6e247d97f4b9cd9f9ec22dbfa74a182e2f32c34fb1c8cf3c8"
RETIRED_V1_ARTIFACT_INVENTORY_SCHEMA = (
    "cfeg.metadata-calibration-efficiency-v3.retired-development-inventory.v1"
)
RETIRED_V1_ARTIFACT_INVENTORY_SHA256 = (
    "ece2fc0822fc33d952b926efacb076586d90a74e1bb5132b110789359837055b"
)
RETIRED_V1_INCIDENT_OUTPUT_SHA256 = (
    "8cdcfe3961979b2d691d4a241878abbf68aa743d8dd08ccbf01ffe1ec03a796a"
)
RETIRED_V2_SOURCE_COMMIT = "82e26d60fd17464380697126bb64ad73b36708a2"
RETIRED_V2_SOURCE_TREE = "f7f59bda48610aeaa5bd1b6ffd288a7b56a4bb85"
RETIRED_V2_SOURCE_BUNDLE_SHA256 = "a74d3a86f207520fd97c746809378146d75614fe287b1eaab96e50b15da3dbc0"
RETIRED_V2_BUNDLE_PAYLOAD_SHA256 = (
    "ce9628ecc5a2b000aa1ce370bba540e6d0fa2d46d3ae068cbcc5b43f089ba312"
)
RETIRED_V2_BUNDLE_FILE_SHA256 = "a616945040ea45a78c2eed58bf251f1f379f9a82a320ed4f39c26f5efb073cac"
RETIRED_V2_ARTIFACT_INVENTORY_SCHEMA = (
    "cfeg.metadata-calibration-efficiency-v3.retired-development-inventory.v2"
)
RETIRED_V2_ARTIFACT_INVENTORY_SHA256 = (
    "7ae6a99b7ec2033b2ce0cdfbfedc05851b77d9e5dfeb21b485ebe4215f18c865"
)
RETIRED_V2_INCIDENT_OUTPUT_SHA256 = (
    "5e21a934ee692d13e3717502442c412d58121547eb2d05ecaf3c1bd827abd04c"
)
SCIENTIFIC_CONTRACT_PROJECTION_SHA256 = (
    "0702d01be1e055d3203a3c1b78777db6456b8d527e5525b6d468fb52522f8a79"
)
SCIENTIFIC_CONTRACT_PROJECTION_SCHEMA = (
    "cfeg.metadata-calibration-efficiency-v3.scientific-contract-projection.v1"
)
_SCIENTIFIC_PROJECTION_MASTER_KEYS = (
    "candidate_id",
    "scientific_candidate_id_template",
    "objective",
    "research_question",
    "claim_order",
    "interpretation_boundary",
    "information_rights",
    "roles",
    "anchor",
    "score_normalization",
    "support_residual",
    "fusion",
    "metadata_trust",
    "target_local_gate",
    "calibration_contract",
    "factor_ablation",
    "data_roles",
    "estimands",
    "promotion_logic",
    "stopping_rules",
)
_SCIENTIFIC_PROJECTION_SYNTHETIC_KEYS = (
    "candidate_id",
    "generator_revision",
    "purpose",
    "disclosed_prior_seed_exposure",
    "rng",
    "population",
    "waveform",
    "operator_grid",
    "context_reference",
    "families",
    "context_controls",
    "invariants",
    "complete_grid",
    "metrics",
    "promotion_requirements",
    "development_selection",
    "governance_canary",
    "scientific_lockbox",
    "stopping_rules",
)
_SCIENTIFIC_PROJECTION_UNCHANGED_COMPONENTS = (
    "objective",
    "operator_equations",
    "DGP",
    "three_by_three_grid",
    "thresholds",
    "promotion_gates",
    "selection_rank",
    "context_reference_root_seed_20260910",
    "development_root_seed_20260909",
    "evidence_roles_and_cohorts",
)
RETIRED_V1_INCIDENT_ERROR = "AuthorityError"
RETIRED_V1_INCIDENT_MESSAGE = "this governed process role cannot mutate canonical state"
RETIRED_V1_INCIDENT_OUTPUT_BYTES = (
    b'{"command":"status","error":"AuthorityError","message":"this governed process '
    b'role cannot mutate canonical state","status":"FAIL_CLOSED"}\n'
)
RETIRED_V2_INCIDENT_ERROR = "TypeError"
RETIRED_V2_INCIDENT_MESSAGE = "Object of type mappingproxy is not JSON serializable"
RETIRED_V2_INCIDENT_OUTPUT_BYTES = (
    b'{"command":"resume","error":"TypeError","message":"Object of type mappingproxy '
    b'is not JSON serializable","status":"FAIL_CLOSED"}\n'
)
TEST_EVIDENCE_CANONICAL_PATH = CANARY_CANONICAL_ROOT / "test-evidence.json"
SELECTED_METHOD_FREEZE_REPOSITORY_PATH = PurePosixPath(
    "configs/governance/metadata_calibration_v3_selected_method_freeze.json"
)
DEVELOPMENT_RESULT_MAXIMUM_BYTES = 512 * 1024 * 1024
DEVELOPMENT_START_EXECUTION_CAPABILITY_SCHEMA = (
    "cfeg.metadata-calibration-efficiency-v3.development-start-execution-capability.v1"
)
_CANONICAL_PYTHON_EXECUTION_ROOTS = ("src", "tests", "scripts")
_ROOT_PYTEST_IMPORT_CONTROL_NAMES = frozenset(
    {
        ".python-version",
        ".pytest.ini",
        ".pytest.toml",
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

_OBSERVED_TEST_ENVIRONMENT = MappingProxyType(
    {
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
)
_STRICT_GIT_ENVIRONMENT = MappingProxyType(
    {
        "GIT_CONFIG_GLOBAL": "/dev/null",
        "GIT_CONFIG_NOSYSTEM": "1",
        "GIT_NO_REPLACE_OBJECTS": "1",
        "GIT_OPTIONAL_LOCKS": "0",
        "GIT_TERMINAL_PROMPT": "0",
        "LANG": "C.UTF-8",
        "LC_ALL": "C.UTF-8",
        "PATH": "/usr/bin:/bin",
    }
)
_SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")
_GIT_OBJECT_RE = re.compile(r"[0-9a-f]{40}\Z")
_ATTEMPT_ID_RE = re.compile(r"sha256-[0-9a-f]{64}\Z")
_RFC3339_SECONDS_RE = re.compile(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z\Z")
_OUTPUT_VALUE_RE = re.compile(r"[0-9A-F]{128}\Z")
_CAPABILITY_ISSUER = object()
_PUBLICATION_ISSUER = object()
_FRESH_EXEC_OBSERVER_ISSUER = object()
_DEVELOPMENT_START_EXECUTION_ISSUER = object()

FRESH_CANARY_EXEC_RECEIPT_SCHEMA = (
    "cfeg.metadata-calibration-efficiency-v3.fresh-canary-exec-receipt.v1"
)
_FRESH_CANARY_EXEC_PHASE = "_fresh-canary-audit-internal"
_ISOLATED_STDLIB_SYS_PATH = (
    "/usr/lib/python310.zip",
    "/usr/lib/python3.10",
    "/usr/lib/python3.10/lib-dynload",
)
_GOVERNED_PYTHON_SUBPROCESS_PREFIX = (
    os.fspath(V3_PYTHON_EXECUTABLE),
    "-I",
    "-B",
    "-S",
)
V3_RUNNER_PATH = V3_SOURCE_REPOSITORY / "src/cfeg/metadata_calibration_v3_runner.py"
_GOVERNED_RUNNER_COMMANDS = frozenset({"status", "resume", "emit-selection"})
_GOVERNED_PYTHON_BOOTSTRAP_PREAMBLE = (
    "import os,pathlib,sys\n"
    f"_expected_stdlib_path={_ISOLATED_STDLIB_SYS_PATH!r}\n"
    f"_expected_python=pathlib.Path({os.fspath(V3_PYTHON_EXECUTABLE)!r})\n"
    f"_source=pathlib.Path({os.fspath(V3_SOURCE_REPOSITORY / 'src')!r})\n"
    f"_site=pathlib.Path({os.fspath(V3_SITE_PACKAGES)!r})\n"
    f"_expected_environment={dict(_OBSERVED_TEST_ENVIRONMENT)!r}\n"
    "if tuple(sys.path)!=_expected_stdlib_path:\n"
    " raise RuntimeError('governed Python did not start with pristine stdlib sys.path')\n"
    "if not (sys.flags.isolated==1 and sys.flags.dont_write_bytecode==1 "
    "and sys.flags.no_site==1 and sys.flags.ignore_environment==1 "
    "and sys.flags.no_user_site==1 and sys.flags.hash_randomization==1):\n"
    " raise RuntimeError('governed Python requires exact -I -B -S flags')\n"
    "if pathlib.Path(sys.executable)!=_expected_python:\n"
    " raise RuntimeError('governed Python executable path mismatch')\n"
    "if dict(os.environ)!=_expected_environment:\n"
    " raise RuntimeError('governed Python environment mismatch')\n"
    "if _source.resolve(strict=True)!=_source or _site.resolve(strict=True)!=_site:\n"
    " raise RuntimeError('governed Python import roots are not exact nonsymlink paths')\n"
    "sys.path.append(str(_source))\n"
    "sys.path.append(str(_site))\n"
    "if tuple(sys.path)!=(*_expected_stdlib_path,str(_source),str(_site)):\n"
    " raise RuntimeError('governed Python explicit import path construction failed')\n"
)
_FRESH_CANARY_EXEC_BOOTSTRAP = (
    _GOVERNED_PYTHON_BOOTSTRAP_PREAMBLE
    + "import cfeg\n"
    + "import cfeg.metadata_calibration_v3_governance as g\n"
    + "raise SystemExit(g._fresh_canary_audit_exec_child_main())\n"
)
_FRESH_CANARY_EXEC_COMMAND = (
    *_GOVERNED_PYTHON_SUBPROCESS_PREFIX,
    "-c",
    _FRESH_CANARY_EXEC_BOOTSTRAP,
    _FRESH_CANARY_EXEC_PHASE,
)


class GovernanceError(RuntimeError):
    """Base class for V3 governance failures."""


class ValidationError(GovernanceError):
    """An input did not match the frozen contract."""


class AuthorityError(GovernanceError):
    """A requested action has no valid typed authority."""


class TransitionError(GovernanceError):
    """A lifecycle transition is skipped, reversed, duplicated, or cross-domain."""


class PublicationError(GovernanceError):
    """A write-once artifact could not be published safely."""


def _plain_json(value: Any) -> Any:
    if isinstance(value, Mapping):
        if any(not isinstance(key, str) for key in value):
            raise ValidationError("canonical JSON object keys must already be strings")
        return {key: _plain_json(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_plain_json(item) for item in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise ValidationError(f"value is not canonical-JSON compatible: {type(value).__name__}")


def _freeze_json(value: Any) -> Any:
    if isinstance(value, Mapping):
        if any(not isinstance(key, str) for key in value):
            raise ValidationError("immutable JSON object keys must already be strings")
        return MappingProxyType({key: _freeze_json(item) for key, item in value.items()})
    if isinstance(value, list):
        return tuple(_freeze_json(item) for item in value)
    return value


def canonical_json_bytes(value: Mapping[str, Any]) -> bytes:
    """Return the contract's canonical JSON bytes, without a trailing newline."""

    try:
        text = json.dumps(
            _plain_json(value),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        raise ValidationError("value cannot be canonically serialized") from exc
    return text.encode("utf-8")


def file_sha256(value: bytes) -> str:
    if not isinstance(value, bytes):
        raise ValidationError("file hashing requires immutable bytes")
    return hashlib.sha256(value).hexdigest()


def payload_sha256(value: Mapping[str, Any]) -> str:
    """Hash a payload after removing only its top-level ``payload_sha256``."""

    payload = dict(value)
    payload.pop("payload_sha256", None)
    return hashlib.sha256(canonical_json_bytes(payload)).hexdigest()


def seal_payload(value: Mapping[str, Any]) -> Mapping[str, Any]:
    if "payload_sha256" in value:
        raise ValidationError("builder input already contains payload_sha256")
    result = dict(_plain_json(value))
    result["payload_sha256"] = payload_sha256(result)
    return MappingProxyType(result)


def validate_payload_hash(value: Mapping[str, Any]) -> str:
    claimed = value.get("payload_sha256")
    if not isinstance(claimed, str) or not _SHA256_RE.fullmatch(claimed):
        raise ValidationError("missing or malformed payload_sha256")
    expected = payload_sha256(value)
    if claimed != expected:
        raise ValidationError("payload_sha256 mismatch")
    return claimed


def artifact_bytes(value: Mapping[str, Any]) -> bytes:
    validate_payload_hash(value)
    return canonical_json_bytes(value) + b"\n"


def _require_exact_keys(value: Mapping[str, Any], keys: frozenset[str], label: str) -> None:
    observed = frozenset(value)
    if observed != keys:
        missing = sorted(keys - observed)
        extra = sorted(observed - keys)
        raise ValidationError(f"{label} key mismatch; missing={missing}, extra={extra}")


def _require_sha256(value: Any, label: str) -> str:
    if not isinstance(value, str) or not _SHA256_RE.fullmatch(value):
        raise ValidationError(f"{label} must be 64 lowercase hexadecimal characters")
    return value


def _require_git_object(value: Any, label: str) -> str:
    if not isinstance(value, str) or not _GIT_OBJECT_RE.fullmatch(value):
        raise ValidationError(f"{label} must be a 40-character lowercase git object ID")
    return value


def _parse_rfc3339_seconds(value: Any, label: str) -> datetime:
    if not isinstance(value, str) or not _RFC3339_SECONDS_RE.fullmatch(value):
        raise ValidationError(f"{label} must be strict RFC3339 UTC with seconds and Z")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValidationError(f"{label} is not a calendar timestamp") from exc
    return parsed


def _ascii_lines(fields: Sequence[str]) -> bytes:
    if not fields:
        raise ValidationError("ASCII line preimage cannot be empty")
    if any(not isinstance(item, str) or "\r" in item or "\n" in item for item in fields):
        raise ValidationError("ASCII line fields must be strings without CR or LF")
    try:
        return ("\n".join(fields) + "\n").encode("ascii")
    except UnicodeEncodeError as exc:
        raise ValidationError("line-hash fields must be strict ASCII") from exc


@dataclass(frozen=True, slots=True, init=False)
class _ArtifactReference:
    schema: str
    payload_sha256: str
    file_sha256: str
    _validation_marker: object = field(repr=False, compare=False)

    def __init__(
        self,
        *,
        schema: str,
        payload_sha256: str,
        file_sha256: str,
        _issuer: object,
    ) -> None:
        if _issuer is not _CAPABILITY_ISSUER:
            raise AuthorityError("artifact reference has no valid capability issuer")
        if not isinstance(schema, str) or not schema:
            raise ValidationError("artifact schema must be a nonempty string")
        _require_sha256(payload_sha256, "artifact payload_sha256")
        _require_sha256(file_sha256, "artifact file_sha256")
        object.__setattr__(self, "schema", schema)
        object.__setattr__(self, "payload_sha256", payload_sha256)
        object.__setattr__(self, "file_sha256", file_sha256)
        object.__setattr__(self, "_validation_marker", _issuer)


def _require_artifact_reference(
    value: object,
    *,
    expected_schema: str,
) -> _ArtifactReference:
    if type(value) is not _ArtifactReference:
        raise AuthorityError("exact internally-issued artifact reference required")
    reference = value
    if reference._validation_marker is not _CAPABILITY_ISSUER:
        raise AuthorityError("artifact reference marker is invalid")
    if reference.schema != expected_schema:
        raise AuthorityError("artifact reference schema mismatch")
    _require_sha256(reference.payload_sha256, "artifact reference payload_sha256")
    _require_sha256(reference.file_sha256, "artifact reference file_sha256")
    return reference


@dataclass(frozen=True, slots=True)
class CleanGitIdentity:
    commit: str
    tree: str

    def __post_init__(self) -> None:
        _require_git_object(self.commit, "clean commit")
        _require_git_object(self.tree, "clean tree")


@dataclass(frozen=True, slots=True, init=False)
class DevelopmentBundleCapability:
    """Nominal authority issued only by the exact development-bundle validator."""

    schema: str
    candidate_id: str
    payload_sha256: str
    file_sha256: str
    clean_commit: str
    clean_tree: str
    source_bundle_sha256: str
    tracked_source_files: tuple[TrackedSourceFile, ...]
    numerical_runtime_fingerprint_sha256: str
    numerical_runtime_inventory: Mapping[str, Any] = field(repr=False, compare=False)
    canonical_path: Path
    scope: str
    binding_sha256: str
    _validated_payload: Mapping[str, Any] = field(repr=False, compare=False)
    _validation_marker: object = field(repr=False, compare=False)

    def __init__(
        self,
        *,
        schema: str,
        candidate_id: str,
        payload_sha256: str,
        file_sha256: str,
        clean_commit: str,
        clean_tree: str,
        source_bundle_sha256: str,
        tracked_source_files: tuple[TrackedSourceFile, ...],
        numerical_runtime_fingerprint_sha256: str,
        numerical_runtime_inventory: Mapping[str, Any],
        canonical_path: Path,
        scope: str,
        validated_payload: Mapping[str, Any],
        _issuer: object,
    ) -> None:
        if _issuer is not _CAPABILITY_ISSUER:
            raise AuthorityError("DevelopmentBundleCapability has no valid issuer")
        object.__setattr__(self, "schema", schema)
        object.__setattr__(self, "candidate_id", candidate_id)
        object.__setattr__(self, "payload_sha256", payload_sha256)
        object.__setattr__(self, "file_sha256", file_sha256)
        object.__setattr__(self, "clean_commit", clean_commit)
        object.__setattr__(self, "clean_tree", clean_tree)
        object.__setattr__(self, "source_bundle_sha256", source_bundle_sha256)
        object.__setattr__(self, "tracked_source_files", tracked_source_files)
        object.__setattr__(
            self,
            "numerical_runtime_fingerprint_sha256",
            numerical_runtime_fingerprint_sha256,
        )
        object.__setattr__(
            self,
            "numerical_runtime_inventory",
            _freeze_json(numerical_runtime_inventory),
        )
        object.__setattr__(self, "canonical_path", canonical_path)
        object.__setattr__(self, "scope", scope)
        binding = {
            "schema": schema,
            "candidate_id": candidate_id,
            "payload_sha256": payload_sha256,
            "file_sha256": file_sha256,
            "clean_commit": clean_commit,
            "clean_tree": clean_tree,
            "source_bundle_sha256": source_bundle_sha256,
            "tracked_source_file_inventory": [item.as_record() for item in tracked_source_files],
            "numerical_runtime_fingerprint_sha256": (numerical_runtime_fingerprint_sha256),
            "numerical_runtime_inventory": numerical_runtime_inventory,
            "canonical_path": os.fspath(canonical_path),
            "scope": scope,
        }
        object.__setattr__(self, "binding_sha256", _artifact_binding_sha256(binding))
        object.__setattr__(self, "_validated_payload", _freeze_json(validated_payload))
        object.__setattr__(self, "_validation_marker", _issuer)


def require_development_bundle_capability(
    value: object,
    *,
    expected_commit: str,
    expected_tree: str,
) -> DevelopmentBundleCapability:
    """Require the exact nominal bundle capability and its bound source identity."""

    _require_git_object(expected_commit, "expected commit")
    _require_git_object(expected_tree, "expected tree")
    if type(value) is not DevelopmentBundleCapability:
        raise AuthorityError("exact DevelopmentBundleCapability required")
    capability = value
    if capability._validation_marker is not _CAPABILITY_ISSUER:
        raise AuthorityError("development-bundle capability marker is invalid")
    if capability.schema != DEVELOPMENT_BUNDLE_SCHEMA or capability.candidate_id != CANDIDATE_ID:
        raise AuthorityError("development-bundle capability identity mismatch")
    if capability.scope != "canonical" or capability.canonical_path != Path(
        DEVELOPMENT_BUNDLE_CANONICAL_PATH
    ):
        raise AuthorityError("test-only or noncanonical development bundle has no authority")
    _require_sha256(capability.payload_sha256, "bundle capability payload_sha256")
    _require_sha256(capability.file_sha256, "bundle capability file_sha256")
    if (
        not capability.tracked_source_files
        or _source_inventory_digest(capability.tracked_source_files)
        != capability.source_bundle_sha256
    ):
        raise AuthorityError("development-bundle source inventory binding is invalid")
    payload = capability._validated_payload
    expected_binding = {
        "schema": capability.schema,
        "candidate_id": capability.candidate_id,
        "payload_sha256": capability.payload_sha256,
        "file_sha256": capability.file_sha256,
        "clean_commit": capability.clean_commit,
        "clean_tree": capability.clean_tree,
        "source_bundle_sha256": capability.source_bundle_sha256,
        "tracked_source_file_inventory": [
            item.as_record() for item in capability.tracked_source_files
        ],
        "numerical_runtime_fingerprint_sha256": (capability.numerical_runtime_fingerprint_sha256),
        "numerical_runtime_inventory": capability.numerical_runtime_inventory,
        "canonical_path": os.fspath(capability.canonical_path),
        "scope": capability.scope,
    }
    if (
        capability.binding_sha256 != _artifact_binding_sha256(expected_binding)
        or validate_payload_hash(payload) != capability.payload_sha256
        or file_sha256(artifact_bytes(payload)) != capability.file_sha256
        or payload.get("schema") != capability.schema
        or payload.get("candidate_id") != capability.candidate_id
        or payload.get("clean_git_commit") != capability.clean_commit
        or payload.get("clean_git_tree") != capability.clean_tree
        or payload.get("source_bundle_sha256") != capability.source_bundle_sha256
        or payload.get("numerical_runtime_fingerprint_sha256")
        != capability.numerical_runtime_fingerprint_sha256
        or _plain_json(payload.get("numerical_runtime_inventory"))
        != _plain_json(capability.numerical_runtime_inventory)
        or tuple(payload.get("tracked_source_file_inventory", ()))
        != tuple(_freeze_json(item.as_record()) for item in capability.tracked_source_files)
    ):
        raise AuthorityError("development-bundle capability sealed binding mismatch")
    if capability.clean_commit != expected_commit or capability.clean_tree != expected_tree:
        raise AuthorityError("development-bundle capability is bound to another source identity")
    return capability


def require_current_numerical_runtime_fingerprint(
    value: object,
) -> DevelopmentBundleCapability:
    """Require the bundle's exact numerical package/ABI environment right now."""

    if type(value) is not DevelopmentBundleCapability:
        raise AuthorityError("exact DevelopmentBundleCapability required")
    capability = require_development_bundle_capability(
        value,
        expected_commit=value.clean_commit,
        expected_tree=value.clean_tree,
    )
    _require_current_numerical_runtime_binding(
        capability.numerical_runtime_inventory,
        capability.numerical_runtime_fingerprint_sha256,
    )
    return capability


def _require_rng_bundle_capability_for_role(
    value: object,
    *,
    expected_commit: str,
    expected_tree: str,
    allowed_process_roles: frozenset[str],
) -> DevelopmentBundleCapability:
    process = _require_active_governed_process(
        recheck_site_packages=True,
        recheck_source=True,
    )
    if process.role not in allowed_process_roles:
        raise AuthorityError("this governed process role cannot use the requested bundle authority")
    _require_frozen_numerical_executor_environment()
    capability = require_current_numerical_runtime_fingerprint(value)
    if capability.clean_commit != expected_commit or capability.clean_tree != expected_tree:
        raise AuthorityError("development RNG bundle source identity mismatch")
    current = capture_clean_source_snapshot(V3_SOURCE_REPOSITORY)
    if (
        current.identity.commit != capability.clean_commit
        or current.identity.tree != capability.clean_tree
        or current.source_bundle_sha256 != capability.source_bundle_sha256
        or current.tracked_files != capability.tracked_source_files
    ):
        raise AuthorityError("RNG authority requires current source exactly at bundle A")
    _require_loaded_governance_source_identity(capability)
    observe_retired_development_attempts()
    return capability


def require_context_reference_rng_bundle_capability(
    value: object,
    *,
    expected_commit: str,
    expected_tree: str,
) -> DevelopmentBundleCapability:
    """Validate exact bundle A for the read-only context-reference RNG role."""

    return _require_rng_bundle_capability_for_role(
        value,
        expected_commit=expected_commit,
        expected_tree=expected_tree,
        allowed_process_roles=frozenset(
            {"runner_status", "runner_resume", "runner_emit_selection"}
        ),
    )


def require_development_recovery_bundle_capability(
    value: object,
    *,
    expected_commit: str,
    expected_tree: str,
) -> DevelopmentBundleCapability:
    """Validate exact bundle A for read-only post-result recovery and status."""

    return _require_rng_bundle_capability_for_role(
        value,
        expected_commit=expected_commit,
        expected_tree=expected_tree,
        allowed_process_roles=frozenset(
            {"runner_status", "runner_resume", "runner_emit_selection"}
        ),
    )


def require_development_rng_bundle_capability(
    value: object,
    *,
    expected_commit: str,
    expected_tree: str,
) -> DevelopmentBundleCapability:
    """Validate bundle A for JIT development RNG issuance in a mutation role."""

    return _require_rng_bundle_capability_for_role(
        value,
        expected_commit=expected_commit,
        expected_tree=expected_tree,
        allowed_process_roles=frozenset({"runner_resume"}),
    )


class ScientificState(str, Enum):
    DECLARED = "DECLARED"
    BUNDLE_FROZEN = "BUNDLE_FROZEN"
    DEVELOPMENT_COMPLETE = "DEVELOPMENT_COMPLETE"
    SELECTED_METHOD_FROZEN = "SELECTED_METHOD_FROZEN"
    VERIFIED = "VERIFIED"
    CANARY_PASSED = "CANARY_PASSED"
    ATTEMPT_MANIFESTED = "ATTEMPT_MANIFESTED"
    AUTHORIZED = "AUTHORIZED"
    CLAIMED = "CLAIMED"
    BEACON_BOUND = "BEACON_BOUND"
    SCIENTIFIC_VALIDATED = "SCIENTIFIC_VALIDATED"
    RESULT_PUBLISHED = "RESULT_PUBLISHED"
    DEVELOPMENT_NO_GO = "DEVELOPMENT_NO_GO"
    ERROR_TERMINAL_PUBLISHED = "ERROR_TERMINAL_PUBLISHED"
    CONSUMED_MISSING_TERMINAL = "CONSUMED_MISSING_TERMINAL"


class CanaryState(str, Enum):
    CANARY_DECLARED = "CANARY_DECLARED"
    CANARY_AUTHORIZED = "CANARY_AUTHORIZED"
    CANARY_CLAIMED = "CANARY_CLAIMED"
    CANARY_BEACON_BOUND = "CANARY_BEACON_BOUND"
    CANARY_PAYLOAD_VALIDATED = "CANARY_PAYLOAD_VALIDATED"
    CANARY_RESULT_PUBLISHED = "CANARY_RESULT_PUBLISHED"
    CANARY_FRESH_AUDITED = "CANARY_FRESH_AUDITED"


_SCIENTIFIC_TRANSITIONS = frozenset(
    {
        (ScientificState.DECLARED, ScientificState.BUNDLE_FROZEN),
        (ScientificState.BUNDLE_FROZEN, ScientificState.DEVELOPMENT_COMPLETE),
        (ScientificState.DEVELOPMENT_COMPLETE, ScientificState.SELECTED_METHOD_FROZEN),
        (ScientificState.DEVELOPMENT_COMPLETE, ScientificState.DEVELOPMENT_NO_GO),
        (ScientificState.SELECTED_METHOD_FROZEN, ScientificState.VERIFIED),
        (ScientificState.CANARY_PASSED, ScientificState.ATTEMPT_MANIFESTED),
        (ScientificState.ATTEMPT_MANIFESTED, ScientificState.AUTHORIZED),
        (ScientificState.AUTHORIZED, ScientificState.CLAIMED),
        (ScientificState.CLAIMED, ScientificState.BEACON_BOUND),
        (ScientificState.BEACON_BOUND, ScientificState.SCIENTIFIC_VALIDATED),
        (ScientificState.SCIENTIFIC_VALIDATED, ScientificState.RESULT_PUBLISHED),
        (ScientificState.CLAIMED, ScientificState.ERROR_TERMINAL_PUBLISHED),
        (ScientificState.BEACON_BOUND, ScientificState.ERROR_TERMINAL_PUBLISHED),
        (ScientificState.SCIENTIFIC_VALIDATED, ScientificState.ERROR_TERMINAL_PUBLISHED),
        (ScientificState.CLAIMED, ScientificState.CONSUMED_MISSING_TERMINAL),
    }
)
_CANARY_TRANSITIONS = frozenset(zip(tuple(CanaryState)[:-1], tuple(CanaryState)[1:]))


def _lifecycle_chain_binding(
    *,
    state: ScientificState | CanaryState,
    trace: Sequence[ScientificState | CanaryState],
    artifact_bindings: Mapping[str, tuple[str, str]],
    clean_commit: str | None,
    clean_tree: str | None,
) -> Mapping[str, Any]:
    return MappingProxyType(
        {
            "domain": "scientific" if type(state) is ScientificState else "canary",
            "state": state.value,
            "trace": [item.value for item in trace],
            "artifact_bindings": {
                name: [binding[0], binding[1]]
                for name, binding in sorted(artifact_bindings.items())
            },
            "clean_commit": clean_commit,
            "clean_tree": clean_tree,
        }
    )


@dataclass(frozen=True, slots=True, init=False)
class ScientificLifecycle:
    state: ScientificState
    trace: tuple[ScientificState, ...]
    artifact_bindings: Mapping[str, tuple[str, str]]
    clean_commit: str | None
    clean_tree: str | None
    _sealed_chain: Mapping[str, Any] = field(repr=False, compare=False)
    _validation_marker: object = field(repr=False, compare=False)

    def __init__(self) -> None:
        object.__setattr__(self, "state", ScientificState.DECLARED)
        object.__setattr__(self, "trace", (ScientificState.DECLARED,))
        object.__setattr__(self, "artifact_bindings", MappingProxyType({}))
        object.__setattr__(self, "clean_commit", None)
        object.__setattr__(self, "clean_tree", None)
        object.__setattr__(
            self,
            "_sealed_chain",
            _lifecycle_chain_binding(
                state=ScientificState.DECLARED,
                trace=(ScientificState.DECLARED,),
                artifact_bindings={},
                clean_commit=None,
                clean_tree=None,
            ),
        )
        object.__setattr__(self, "_validation_marker", _CAPABILITY_ISSUER)

    def advance(
        self, target: ScientificState, *, authority: object | None = None
    ) -> ScientificLifecycle:
        advanced = _transition_kernel(
            self,
            target,
            authority=authority,
        )
        if type(advanced) is not ScientificLifecycle:
            raise AssertionError("scientific transition returned the wrong lifecycle type")
        return advanced


@dataclass(frozen=True, slots=True, init=False)
class CanaryLifecycle:
    state: CanaryState
    trace: tuple[CanaryState, ...]
    artifact_bindings: Mapping[str, tuple[str, str]]
    clean_commit: str | None
    clean_tree: str | None
    _sealed_chain: Mapping[str, Any] = field(repr=False, compare=False)
    _validation_marker: object = field(repr=False, compare=False)

    def __init__(self) -> None:
        object.__setattr__(self, "state", CanaryState.CANARY_DECLARED)
        object.__setattr__(self, "trace", (CanaryState.CANARY_DECLARED,))
        object.__setattr__(self, "artifact_bindings", MappingProxyType({}))
        object.__setattr__(self, "clean_commit", None)
        object.__setattr__(self, "clean_tree", None)
        object.__setattr__(
            self,
            "_sealed_chain",
            _lifecycle_chain_binding(
                state=CanaryState.CANARY_DECLARED,
                trace=(CanaryState.CANARY_DECLARED,),
                artifact_bindings={},
                clean_commit=None,
                clean_tree=None,
            ),
        )
        object.__setattr__(self, "_validation_marker", _CAPABILITY_ISSUER)

    def advance(self, target: CanaryState, *, authority: object | None = None) -> CanaryLifecycle:
        advanced = _transition_kernel(
            self,
            target,
            authority=authority,
        )
        if type(advanced) is not CanaryLifecycle:
            raise AssertionError("canary transition returned the wrong lifecycle type")
        return advanced


def _exact_binding(capability: object) -> tuple[str, str]:
    payload = getattr(capability, "payload_sha256", None)
    file_hash = getattr(capability, "file_sha256", None)
    _require_sha256(payload, "transition payload_sha256")
    _require_sha256(file_hash, "transition file_sha256")
    return payload, file_hash


def _require_prior_binding(
    bindings: Mapping[str, tuple[str, str]],
    name: str,
    *,
    payload_sha256_value: Any,
    file_sha256_value: Any,
) -> None:
    expected = bindings.get(name)
    observed = (payload_sha256_value, file_sha256_value)
    if expected is None or observed != expected:
        raise AuthorityError(f"transition authority is not bound to prior {name}")


def _require_exact_lifecycle_chain(
    lifecycle: object,
) -> ScientificLifecycle | CanaryLifecycle:
    if (
        type(lifecycle) not in {ScientificLifecycle, CanaryLifecycle}
        or getattr(lifecycle, "_validation_marker", None) is not _CAPABILITY_ISSUER
        or not getattr(lifecycle, "trace", ())
        or lifecycle.trace[-1] is not getattr(lifecycle, "state", None)
    ):
        raise AuthorityError("exact issued lifecycle capability required")
    observed_chain = _lifecycle_chain_binding(
        state=lifecycle.state,
        trace=lifecycle.trace,
        artifact_bindings=lifecycle.artifact_bindings,
        clean_commit=lifecycle.clean_commit,
        clean_tree=lifecycle.clean_tree,
    )
    if canonical_json_bytes(observed_chain) != canonical_json_bytes(lifecycle._sealed_chain):
        raise AuthorityError("sealed lifecycle chain binding is invalid")
    return lifecycle


def _transition_kernel(
    lifecycle: ScientificLifecycle | CanaryLifecycle,
    target: ScientificState | CanaryState,
    *,
    authority: object | None,
) -> ScientificLifecycle | CanaryLifecycle:
    """Only nominal lifecycle instances may use the shared transition kernel."""

    _require_exact_lifecycle_chain(lifecycle)
    if type(lifecycle) is CanaryLifecycle:
        if lifecycle.trace != tuple(CanaryState)[: len(lifecycle.trace)]:
            raise AuthorityError("canary lifecycle trace is not an exact state prefix")
    else:
        allowed_edges = _SCIENTIFIC_TRANSITIONS | {
            (ScientificState.VERIFIED, ScientificState.CANARY_PASSED)
        }
        if lifecycle.trace[0] is not ScientificState.DECLARED or any(
            edge not in allowed_edges for edge in zip(lifecycle.trace, lifecycle.trace[1:])
        ):
            raise AuthorityError("scientific lifecycle trace is not a valid frozen path")
    current = lifecycle.state
    if type(current) is not type(target):
        raise TransitionError("cross-domain lifecycle transition rejected")
    transitions = (
        _SCIENTIFIC_TRANSITIONS if type(current) is ScientificState else _CANARY_TRANSITIONS
    )
    if (current, target) not in transitions:
        raise TransitionError(f"invalid lifecycle transition: {current.value}->{target.value}")

    bindings = dict(lifecycle.artifact_bindings)
    clean_commit = lifecycle.clean_commit
    clean_tree = lifecycle.clean_tree
    if type(current) is ScientificState:
        if (current, target) == (ScientificState.DECLARED, ScientificState.BUNDLE_FROZEN):
            if type(authority) is not DevelopmentBundleCapability:
                raise AuthorityError("BUNDLE_FROZEN requires DevelopmentBundleCapability")
            bundle = require_development_bundle_capability(
                authority,
                expected_commit=authority.clean_commit,
                expected_tree=authority.clean_tree,
            )
            bindings["development_bundle"] = _exact_binding(bundle)
            clean_commit = bundle.clean_commit
            clean_tree = bundle.clean_tree
        elif (current, target) == (
            ScientificState.BUNDLE_FROZEN,
            ScientificState.DEVELOPMENT_COMPLETE,
        ):
            result = require_validated_artifact_capability(
                authority,
                expected_schema=ARTIFACT_SCHEMAS["development_result"],
            )
            payload = result._validated_payload
            _require_prior_binding(
                bindings,
                "development_bundle",
                payload_sha256_value=payload["development_bundle_payload_sha256"],
                file_sha256_value=payload["development_bundle_file_sha256"],
            )
            bindings["development_result"] = _exact_binding(result)
        elif current is ScientificState.DEVELOPMENT_COMPLETE and target in {
            ScientificState.SELECTED_METHOD_FROZEN,
            ScientificState.DEVELOPMENT_NO_GO,
        }:
            expected_schema = (
                ARTIFACT_SCHEMAS["selected_method_freeze"]
                if target is ScientificState.SELECTED_METHOD_FROZEN
                else ARTIFACT_SCHEMAS["development_result"]
            )
            artifact = require_validated_artifact_capability(
                authority,
                expected_schema=expected_schema,
            )
            if target is ScientificState.DEVELOPMENT_NO_GO:
                if _exact_binding(artifact) != bindings.get("development_result"):
                    raise AuthorityError("DEVELOPMENT_NO_GO requires the prior result capability")
            else:
                payload = artifact._validated_payload
                _require_prior_binding(
                    bindings,
                    "development_bundle",
                    payload_sha256_value=payload["development_bundle_payload_sha256"],
                    file_sha256_value=payload["development_bundle_file_sha256"],
                )
                _require_prior_binding(
                    bindings,
                    "development_result",
                    payload_sha256_value=payload["development_result_payload_sha256"],
                    file_sha256_value=payload["development_result_file_sha256"],
                )
                bindings["selected_method_freeze"] = _exact_binding(artifact)
                clean_commit = artifact.source_commit
                clean_tree = artifact.source_tree
        elif (current, target) == (
            ScientificState.SELECTED_METHOD_FROZEN,
            ScientificState.VERIFIED,
        ):
            tests = require_validated_artifact_capability(
                authority,
                expected_schema=ARTIFACT_SCHEMAS["test_evidence"],
            )
            payload = tests._validated_payload
            _require_prior_binding(
                bindings,
                "selected_method_freeze",
                payload_sha256_value=payload["selected_method_freeze_payload_sha256"],
                file_sha256_value=payload["selected_method_freeze_file_sha256"],
            )
            if payload["clean_commit"] != clean_commit or payload["clean_tree"] != clean_tree:
                raise AuthorityError("test evidence clean identity differs from selected freeze")
            bindings["test_evidence"] = _exact_binding(tests)
        elif (current, target) == (
            ScientificState.CANARY_PASSED,
            ScientificState.ATTEMPT_MANIFESTED,
        ):
            manifest = require_attempt_manifest_capability(authority)
            manifest_payload = manifest._validated_payload
            for name in ("development_bundle", "selected_method_freeze", "test_evidence"):
                _require_prior_binding(
                    bindings,
                    name,
                    payload_sha256_value=manifest_payload[f"{name}_payload_sha256"],
                    file_sha256_value=manifest_payload[f"{name}_file_sha256"],
                )
            if manifest.clean_commit != clean_commit or manifest.clean_tree != clean_tree:
                raise AuthorityError("attempt manifest clean identity differs from lifecycle")
            bindings["attempt_manifest"] = _exact_binding(manifest)
        elif (current, target) == (
            ScientificState.ATTEMPT_MANIFESTED,
            ScientificState.AUTHORIZED,
        ):
            if type(authority) is not ExecutionAuthorizationCapability or (
                authority._validation_marker is not _CAPABILITY_ISSUER
            ):
                raise AuthorityError("AUTHORIZED requires validated execution authorization")
            _require_prior_binding(
                bindings,
                "attempt_manifest",
                payload_sha256_value=authority.attempt_manifest_payload_sha256,
                file_sha256_value=authority.attempt_manifest_file_sha256,
            )
            bindings["execution_authorization"] = _exact_binding(authority)
        elif (current, target) == (ScientificState.AUTHORIZED, ScientificState.CLAIMED):
            claim = require_published_global_claim_capability(authority)
            _require_prior_binding(
                bindings,
                "attempt_manifest",
                payload_sha256_value=claim.attempt_manifest_payload_sha256,
                file_sha256_value=claim.attempt_manifest_file_sha256,
            )
            _require_prior_binding(
                bindings,
                "execution_authorization",
                payload_sha256_value=claim.execution_authorization_payload_sha256,
                file_sha256_value=claim.execution_authorization_file_sha256,
            )
            bindings["global_claim"] = _exact_binding(claim)
        elif (current, target) == (ScientificState.CLAIMED, ScientificState.BEACON_BOUND):
            beacon = require_scientific_beacon_capability(authority)
            _require_prior_binding(
                bindings,
                "attempt_manifest",
                payload_sha256_value=beacon.attempt_manifest_payload_sha256,
                file_sha256_value=beacon.attempt_manifest_file_sha256,
            )
            _require_prior_binding(
                bindings,
                "global_claim",
                payload_sha256_value=beacon.global_claim_payload_sha256,
                file_sha256_value=beacon.global_claim_file_sha256,
            )
            bindings["scientific_beacon"] = _exact_binding(beacon)
        elif target in {
            ScientificState.SCIENTIFIC_VALIDATED,
            ScientificState.RESULT_PUBLISHED,
            ScientificState.ERROR_TERMINAL_PUBLISHED,
            ScientificState.CONSUMED_MISSING_TERMINAL,
        }:
            raise AuthorityError(
                "current frozen contract grants no scientific result or terminal authority"
            )
        else:
            raise AuthorityError("scientific transition has no exact artifact authority")
    else:
        if (current, target) == (
            CanaryState.CANARY_DECLARED,
            CanaryState.CANARY_AUTHORIZED,
        ):
            authorization = require_canary_artifact_capability(
                authority,
                expected_schema=ARTIFACT_SCHEMAS["canary_authorization"],
            )
            payload = authorization._validated_payload
            bindings["selected_method_freeze"] = (
                payload["selected_method_freeze_payload_sha256"],
                payload["selected_method_freeze_file_sha256"],
            )
            bindings["test_evidence"] = (
                payload["test_evidence_payload_sha256"],
                payload["test_evidence_file_sha256"],
            )
            bindings["canary_authorization"] = _exact_binding(authorization)
            clean_commit = str(payload["clean_commit"])
            clean_tree = str(payload["clean_tree"])
        elif (current, target) == (
            CanaryState.CANARY_AUTHORIZED,
            CanaryState.CANARY_CLAIMED,
        ):
            claim = require_canary_artifact_capability(
                authority,
                expected_schema=ARTIFACT_SCHEMAS["canary_claim"],
            )
            payload = claim._validated_payload
            _require_prior_binding(
                bindings,
                "canary_authorization",
                payload_sha256_value=payload["canary_authorization_payload_sha256"],
                file_sha256_value=payload["canary_authorization_file_sha256"],
            )
            bindings["canary_claim"] = _exact_binding(claim)
        elif (current, target) == (
            CanaryState.CANARY_CLAIMED,
            CanaryState.CANARY_BEACON_BOUND,
        ):
            beacon = require_canary_artifact_capability(
                authority,
                expected_schema=ARTIFACT_SCHEMAS["canary_beacon"],
            )
            payload = beacon._validated_payload
            _require_prior_binding(
                bindings,
                "canary_claim",
                payload_sha256_value=payload["canary_claim_payload_sha256"],
                file_sha256_value=payload["canary_claim_file_sha256"],
            )
            bindings["canary_beacon"] = _exact_binding(beacon)
        elif (current, target) == (
            CanaryState.CANARY_BEACON_BOUND,
            CanaryState.CANARY_PAYLOAD_VALIDATED,
        ):
            seed = _require_canary_seed_capability(authority)
            if seed.seed_digest_sha256 != CANARY_SEED_DIGEST:
                raise AuthorityError("canary payload validation seed mismatch")
        elif (current, target) == (
            CanaryState.CANARY_PAYLOAD_VALIDATED,
            CanaryState.CANARY_RESULT_PUBLISHED,
        ):
            publication = require_published_canary_result_capability(authority)
            result = publication.artifact
            payload = result._validated_payload
            _require_prior_binding(
                bindings,
                "canary_beacon",
                payload_sha256_value=payload["canary_beacon_payload_sha256"],
                file_sha256_value=payload["canary_beacon_file_sha256"],
            )
            bindings["canary_result"] = _exact_binding(result)
        elif (current, target) == (
            CanaryState.CANARY_RESULT_PUBLISHED,
            CanaryState.CANARY_FRESH_AUDITED,
        ):
            fresh = require_fresh_canary_audit_capability(authority)
            for name in ("selected_method_freeze", "test_evidence", "canary_result"):
                _require_prior_binding(
                    bindings,
                    name,
                    payload_sha256_value=getattr(fresh, f"{name}_payload_sha256"),
                    file_sha256_value=getattr(fresh, f"{name}_file_sha256"),
                )
            if fresh.clean_commit != clean_commit or fresh.clean_tree != clean_tree:
                raise AuthorityError("fresh audit clean identity differs from canary lifecycle")
            bindings["canary_fresh_audit"] = _exact_binding(fresh)
        else:
            raise AuthorityError("canary transition has no exact artifact authority")
    cls = ScientificLifecycle if type(lifecycle) is ScientificLifecycle else CanaryLifecycle
    advanced = object.__new__(cls)
    object.__setattr__(advanced, "state", target)
    object.__setattr__(advanced, "trace", (*lifecycle.trace, target))
    object.__setattr__(advanced, "artifact_bindings", MappingProxyType(bindings))
    object.__setattr__(advanced, "clean_commit", clean_commit)
    object.__setattr__(advanced, "clean_tree", clean_tree)
    object.__setattr__(
        advanced,
        "_sealed_chain",
        _lifecycle_chain_binding(
            state=target,
            trace=(*lifecycle.trace, target),
            artifact_bindings=bindings,
            clean_commit=clean_commit,
            clean_tree=clean_tree,
        ),
    )
    object.__setattr__(advanced, "_validation_marker", _CAPABILITY_ISSUER)
    return advanced


@dataclass(frozen=True, slots=True)
class ArtifactSpec:
    schema: str
    exact_fields: frozenset[str] | None = None


def parse_artifact_bytes(data: bytes, specification: ArtifactSpec) -> Mapping[str, Any]:
    """Parse one canonical artifact from supplied bytes without performing I/O."""

    if not isinstance(data, bytes):
        raise ValidationError("artifact parser accepts bytes only")
    if not data.endswith(b"\n") or data.endswith(b"\n\n"):
        raise ValidationError("artifact file must have exactly one trailing LF")
    try:
        value = json.loads(data[:-1].decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValidationError("artifact is not valid UTF-8 JSON") from exc
    if not isinstance(value, dict):
        raise ValidationError("artifact root must be a JSON object")
    if canonical_json_bytes(value) + b"\n" != data:
        raise ValidationError("artifact bytes are not canonical JSON plus one LF")
    if value.get("schema") != specification.schema:
        raise ValidationError("artifact schema mismatch")
    if specification.exact_fields is not None:
        _require_exact_keys(value, specification.exact_fields, specification.schema)
    validate_payload_hash(value)
    return _freeze_json(value)


def parse_exact_artifacts(
    bytes_by_path: Mapping[str, bytes],
    specifications: Mapping[str, ArtifactSpec],
) -> Mapping[str, Mapping[str, Any]]:
    """Pure exact-set parser; it cannot discover or load undeclared artifacts."""

    if frozenset(bytes_by_path) != frozenset(specifications):
        raise ValidationError("artifact bytes must match the exact declared path set")
    parsed = {
        path: parse_artifact_bytes(bytes_by_path[path], specifications[path])
        for path in sorted(specifications)
    }
    return MappingProxyType(parsed)


def _require_dirfd_primitives() -> None:
    required = ("O_DIRECTORY", "O_NOFOLLOW")
    if any(not hasattr(os, name) for name in required):
        raise PublicationError("directory-fd O_DIRECTORY/O_NOFOLLOW guarantees unavailable")
    if os.open not in os.supports_dir_fd or os.mkdir not in os.supports_dir_fd:
        raise PublicationError("required dir_fd operations are unavailable")


def _canonical_absolute_path(path: str | Path) -> Path:
    if not isinstance(path, (str, Path)):
        raise ValidationError("artifact path must be a string or Path")
    raw = os.fspath(path)
    if not isinstance(raw, str) or not raw or "\x00" in raw:
        raise ValidationError("artifact path must be nonempty canonical text")
    canonical = PurePosixPath(raw).as_posix()
    if raw != canonical:
        raise ValidationError("artifact path text is not canonical POSIX syntax")
    return Path(raw)


def _path_parts(path: Path) -> tuple[str, ...]:
    if not path.is_absolute():
        raise ValidationError("artifact paths must be absolute")
    parts = PurePosixPath(path.as_posix()).parts
    if not parts or parts[0] != "/" or any(part in ("", ".", "..") for part in parts[1:]):
        raise ValidationError("artifact path is not canonical absolute POSIX syntax")
    return tuple(parts[1:])


def _is_same_or_descendant(path: Path, boundary: Path) -> bool:
    try:
        path.relative_to(boundary)
    except ValueError:
        return False
    return True


def _relative_parts(path: str | PurePosixPath) -> tuple[str, ...]:
    if not isinstance(path, (str, PurePosixPath)):
        raise ValidationError("publication path must be canonical POSIX text")
    raw = os.fspath(path)
    if not isinstance(raw, str) or not raw or "\x00" in raw:
        raise ValidationError("publication path must be canonical nonempty text")
    pure = PurePosixPath(raw)
    if (
        raw != pure.as_posix()
        or pure.is_absolute()
        or not pure.parts
        or any(part in ("", ".", "..") for part in pure.parts)
    ):
        raise ValidationError("publication path must be a canonical nonempty relative path")
    return tuple(pure.parts)


def _validate_directory_fd(descriptor: int, *, exact_private_mode: bool) -> os.stat_result:
    observed = os.fstat(descriptor)
    if not stat.S_ISDIR(observed.st_mode):
        raise PublicationError("path component is not a directory")
    if exact_private_mode:
        if observed.st_uid != os.geteuid():
            raise PublicationError("canonical artifact directory has an unexpected owner")
        if stat.S_IMODE(observed.st_mode) != 0o700:
            raise PublicationError("canonical artifact directory mode must be 0700")
    return observed


def _open_absolute_directory(
    path: Path,
    *,
    create: bool,
    require_private_final: bool = True,
) -> int:
    """Open an absolute directory without following any component symlink."""

    _require_dirfd_primitives()
    parts = _path_parts(path)
    flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
    descriptor = os.open("/", flags)
    try:
        if not parts:
            _validate_directory_fd(descriptor, exact_private_mode=require_private_final)
        for index, part in enumerate(parts):
            try:
                child = os.open(part, flags, dir_fd=descriptor)
            except FileNotFoundError:
                if not create:
                    raise
                try:
                    os.mkdir(part, 0o700, dir_fd=descriptor)
                except FileExistsError:
                    # A concurrent creator is acceptable only if the no-follow
                    # open and fstat checks below validate the exact directory.
                    pass
                child = os.open(part, flags, dir_fd=descriptor)
            _validate_directory_fd(
                child,
                exact_private_mode=require_private_final and index == len(parts) - 1,
            )
            os.close(descriptor)
            descriptor = child
        return descriptor
    except Exception:
        os.close(descriptor)
        raise


def _reject_prohibited_artifact_root(path: Path) -> None:
    prohibited = Path("/home/whwovy/v2-artifacts")
    try:
        path.relative_to(prohibited)
    except ValueError:
        return
    raise PublicationError("V2 artifact root is prohibited for V3")


def _read_all(descriptor: int, *, maximum_bytes: int) -> bytes:
    chunks: list[bytes] = []
    total = 0
    while True:
        chunk = os.read(descriptor, min(1 << 20, maximum_bytes - total + 1))
        if not chunk:
            break
        chunks.append(chunk)
        total += len(chunk)
        if total > maximum_bytes:
            raise ValidationError("artifact exceeds the explicit loader size limit")
    return b"".join(chunks)


def _current_executable_file_sha256() -> str:
    return _process_executable_file_sha256(os.getpid())


def _process_executable_file_sha256(process_id: int) -> str:
    if not isinstance(process_id, int) or isinstance(process_id, bool) or process_id <= 0:
        raise ValidationError("process ID must be a positive integer")
    descriptor = os.open(f"/proc/{process_id}/exe", os.O_RDONLY)
    try:
        observed = os.fstat(descriptor)
        if not stat.S_ISREG(observed.st_mode):
            raise ValidationError("current executable is not a regular file")
        return file_sha256(_read_all(descriptor, maximum_bytes=256 * 1024 * 1024))
    finally:
        os.close(descriptor)


def _process_command_line(process_id: int) -> tuple[str, ...]:
    if not isinstance(process_id, int) or isinstance(process_id, bool) or process_id <= 0:
        raise ValidationError("process ID must be a positive integer")
    descriptor = os.open(f"/proc/{process_id}/cmdline", os.O_RDONLY | os.O_NOFOLLOW)
    try:
        raw = _read_all(descriptor, maximum_bytes=64 * 1024)
    finally:
        os.close(descriptor)
    if not raw.endswith(b"\x00"):
        raise AuthorityError("observed process command line is incomplete")
    fields = raw[:-1].split(b"\x00")
    try:
        decoded = tuple(field.decode("utf-8") for field in fields)
    except UnicodeDecodeError as exc:
        raise AuthorityError("observed process command line is not UTF-8") from exc
    if not decoded or any(not field or "\x00" in field for field in decoded):
        raise AuthorityError("observed process command line is malformed")
    return decoded


_NUMERICAL_DISTRIBUTIONS = (
    ("numpy", "numpy"),
    ("scipy", "scipy"),
    ("PyYAML", "yaml"),
    ("cryptography", "cryptography"),
    ("pytest", "pytest"),
    # Active Python 3.10 dependency closure of pytest and cryptography.  These
    # are security/runtime inputs even when the top-level distributions do not
    # expose their versions in a scientific artifact field of their own.
    ("exceptiongroup", "exceptiongroup"),
    ("iniconfig", "iniconfig"),
    ("packaging", "packaging"),
    ("pluggy", "pluggy"),
    ("Pygments", "pygments"),
    ("tomli", "tomli"),
    ("typing_extensions", "typing_extensions"),
    ("cffi", "cffi"),
    ("pycparser", "pycparser"),
)
_NUMERICAL_ENVIRONMENT_KEYS = (
    "BLIS_NUM_THREADS",
    "CUDA_VISIBLE_DEVICES",
    "CUBLAS_WORKSPACE_CONFIG",
    "MKL_DYNAMIC",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "OMP_DYNAMIC",
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
)
_COMPLETE_SITE_PACKAGES_INVENTORY_SCHEMA = (
    "cfeg.metadata-calibration-efficiency-v3.python-site-inventory.v1"
)
_PYTHON_SITE_INVENTORY_FIELDS = frozenset(
    {
        "schema",
        "root_path",
        "directory_count",
        "regular_file_count",
        "total_regular_file_bytes",
        "inventory_sha256",
    }
)
_PYTHON_SITE_INVENTORY_ARG_PREFIX = "--cfeg-site-inventory-sha256="


def _require_python_site_inventory(value: object) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise AuthorityError("exact Python site inventory summary is required")
    _require_exact_keys(value, _PYTHON_SITE_INVENTORY_FIELDS, "Python site inventory")
    if value.get("schema") != _COMPLETE_SITE_PACKAGES_INVENTORY_SCHEMA or value.get(
        "root_path"
    ) != os.fspath(V3_SITE_PACKAGES):
        raise AuthorityError("Python site inventory identity is invalid")
    for field_name in (
        "directory_count",
        "regular_file_count",
        "total_regular_file_bytes",
    ):
        observed = value.get(field_name)
        if not isinstance(observed, int) or isinstance(observed, bool) or observed <= 0:
            raise AuthorityError(f"Python site inventory {field_name} is invalid")
    _require_sha256(value.get("inventory_sha256"), "Python site inventory digest")
    return value


def _python_site_inventory_sha256(runtime_inventory: Mapping[str, Any]) -> str:
    summary = _require_python_site_inventory(runtime_inventory.get("python_site_inventory"))
    return str(summary["inventory_sha256"])


def _hash_runtime_file_descriptor(descriptor: int) -> tuple[int, str]:
    """Stream-hash one unbounded runtime file while detecting in-place races."""

    before = os.fstat(descriptor)
    if not stat.S_ISREG(before.st_mode):
        raise AuthorityError("site-packages inventory encountered a non-regular file")
    digest = hashlib.sha256()
    size = 0
    while True:
        block = os.read(descriptor, 8 * 1024 * 1024)
        if not block:
            break
        digest.update(block)
        size += len(block)
    after = os.fstat(descriptor)
    before_identity = (
        before.st_dev,
        before.st_ino,
        before.st_size,
        before.st_mtime_ns,
        before.st_ctime_ns,
        before.st_mode,
        before.st_uid,
    )
    after_identity = (
        after.st_dev,
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
        after.st_ctime_ns,
        after.st_mode,
        after.st_uid,
    )
    if before_identity != after_identity or size != before.st_size:
        raise AuthorityError("site-packages file changed while it was hashed")
    return size, digest.hexdigest()


def _require_safe_runtime_entry(
    observed: os.stat_result,
    *,
    expect_directory: bool,
) -> None:
    mode = stat.S_IMODE(observed.st_mode)
    expected_type = stat.S_ISDIR if expect_directory else stat.S_ISREG
    if (
        not expected_type(observed.st_mode)
        or observed.st_uid != os.geteuid()
        or mode & 0o7000
        or mode & stat.S_IWOTH
    ):
        raise AuthorityError("site-packages contains an unsafe type, owner, or permission mode")


def _capture_complete_site_packages_inventory() -> Mapping[str, Any]:
    """Hash every site-packages entry without Git/RECORD/ignore semantics."""

    root = V3_SITE_PACKAGES
    if root.resolve(strict=True) != root:
        raise AuthorityError("site-packages must be an exact nonsymlink directory")
    root_descriptor = _open_absolute_directory(
        root,
        create=False,
        require_private_final=False,
    )
    entries: list[dict[str, Any]] = []
    directory_count = 0
    regular_file_count = 0
    total_regular_file_bytes = 0

    def visit(descriptor: int, relative: PurePosixPath) -> None:
        nonlocal directory_count, regular_file_count, total_regular_file_bytes
        before = os.fstat(descriptor)
        _require_safe_runtime_entry(before, expect_directory=True)
        directory_count += 1
        entries.append(
            {
                "kind": "directory",
                "mode": f"{stat.S_IMODE(before.st_mode):04o}",
                "path": relative.as_posix(),
            }
        )
        try:
            names = tuple(sorted(os.listdir(descriptor)))
        except OSError as exc:
            raise AuthorityError("site-packages directory could not be enumerated") from exc
        for name in names:
            if (
                not isinstance(name, str)
                or not name
                or name in {".", ".."}
                or "/" in name
                or "\x00" in name
            ):
                raise AuthorityError("site-packages contains noncanonical path syntax")
            child_relative = relative / name
            try:
                listed = os.stat(name, dir_fd=descriptor, follow_symlinks=False)
            except OSError as exc:
                raise AuthorityError("site-packages entry could not be inspected") from exc
            if stat.S_ISDIR(listed.st_mode):
                _require_safe_runtime_entry(listed, expect_directory=True)
                try:
                    child_descriptor = os.open(
                        name,
                        os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW,
                        dir_fd=descriptor,
                    )
                except OSError as exc:
                    raise AuthorityError(
                        "site-packages directory could not be opened without following links"
                    ) from exc
                try:
                    opened = os.fstat(child_descriptor)
                    if (listed.st_dev, listed.st_ino) != (opened.st_dev, opened.st_ino):
                        raise AuthorityError("site-packages directory changed during traversal")
                    visit(child_descriptor, child_relative)
                finally:
                    os.close(child_descriptor)
                continue
            _require_safe_runtime_entry(listed, expect_directory=False)
            try:
                file_descriptor = os.open(
                    name,
                    os.O_RDONLY | os.O_NOFOLLOW,
                    dir_fd=descriptor,
                )
            except OSError as exc:
                raise AuthorityError(
                    "site-packages file could not be opened without following links"
                ) from exc
            try:
                opened = os.fstat(file_descriptor)
                _require_safe_runtime_entry(opened, expect_directory=False)
                if (
                    listed.st_dev,
                    listed.st_ino,
                    listed.st_mode,
                    listed.st_uid,
                    listed.st_nlink,
                    listed.st_size,
                    listed.st_mtime_ns,
                    listed.st_ctime_ns,
                ) != (
                    opened.st_dev,
                    opened.st_ino,
                    opened.st_mode,
                    opened.st_uid,
                    opened.st_nlink,
                    opened.st_size,
                    opened.st_mtime_ns,
                    opened.st_ctime_ns,
                ):
                    raise AuthorityError("site-packages file changed before hashing")
                size, digest = _hash_runtime_file_descriptor(file_descriptor)
                entries.append(
                    {
                        "file_sha256": digest,
                        "kind": "regular_file",
                        "link_count": opened.st_nlink,
                        "mode": f"{stat.S_IMODE(opened.st_mode):04o}",
                        "path": child_relative.as_posix(),
                        "size_bytes": size,
                    }
                )
                regular_file_count += 1
                total_regular_file_bytes += size
            finally:
                os.close(file_descriptor)
        after = os.fstat(descriptor)
        if (
            before.st_dev,
            before.st_ino,
            before.st_mtime_ns,
            before.st_ctime_ns,
            before.st_mode,
            before.st_uid,
        ) != (
            after.st_dev,
            after.st_ino,
            after.st_mtime_ns,
            after.st_ctime_ns,
            after.st_mode,
            after.st_uid,
        ):
            raise AuthorityError("site-packages directory changed during traversal")

    try:
        visit(root_descriptor, PurePosixPath("."))
    finally:
        os.close(root_descriptor)
    entries.sort(key=lambda value: str(value["path"]))
    inventory_sha256 = hashlib.sha256(
        canonical_json_bytes(
            {
                "schema": _COMPLETE_SITE_PACKAGES_INVENTORY_SCHEMA,
                "root_path": os.fspath(root),
                "entries": entries,
            }
        )
    ).hexdigest()
    return MappingProxyType(
        {
            "schema": _COMPLETE_SITE_PACKAGES_INVENTORY_SCHEMA,
            "root_path": os.fspath(root),
            "directory_count": directory_count,
            "regular_file_count": regular_file_count,
            "total_regular_file_bytes": total_regular_file_bytes,
            "inventory_sha256": inventory_sha256,
        }
    )


def _hash_installed_regular_file(path: Path) -> str:
    descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
    try:
        observed = os.fstat(descriptor)
        if not stat.S_ISREG(observed.st_mode):
            raise AuthorityError("runtime dependency inventory contains a non-regular file")
        return file_sha256(_read_all(descriptor, maximum_bytes=256 * 1024 * 1024))
    finally:
        os.close(descriptor)


def _run_git_version() -> str:
    completed = subprocess.run(
        [os.fspath(V3_GIT_EXECUTABLE), "--version"],
        check=False,
        stdin=subprocess.DEVNULL,
        capture_output=True,
        env=dict(_STRICT_GIT_ENVIRONMENT),
    )
    if completed.returncode != 0 or completed.stderr:
        raise AuthorityError("pinned Git executable could not report its version cleanly")
    try:
        version = completed.stdout.decode("ascii").strip()
    except UnicodeDecodeError as exc:
        raise AuthorityError("pinned Git version is not ASCII") from exc
    if not version.startswith("git version ") or "\n" in version or "\r" in version:
        raise AuthorityError("pinned Git version output is malformed")
    return version


def _require_pinned_git_executable() -> None:
    if _hash_installed_regular_file(V3_GIT_EXECUTABLE) != _GIT_IMPORT_EXECUTABLE_FILE_SHA256:
        raise AuthorityError("pinned Git executable changed after governance import")


def _distribution_runtime_record(
    distribution_name: str,
    module_name: str,
) -> Mapping[str, Any]:
    try:
        distribution = importlib.metadata.distribution(distribution_name)
        module_spec = importlib.util.find_spec(module_name)
    except (ImportError, importlib.metadata.PackageNotFoundError) as exc:
        raise AuthorityError(
            f"required numerical distribution is unavailable: {distribution_name}"
        ) from exc
    if module_spec is None or not isinstance(module_spec.origin, str):
        raise AuthorityError(f"required module has no import origin: {module_name}")
    files = tuple(distribution.files or ())
    record_files = tuple(
        entry for entry in files if entry.name == "RECORD" and ".dist-info" in entry.as_posix()
    )
    if len(record_files) != 1:
        raise AuthorityError(
            f"required numerical distribution has no unique RECORD: {distribution_name}"
        )
    record_path = Path(distribution.locate_file(record_files[0]))
    sorted_entries = tuple(sorted(files, key=lambda item: item.as_posix()))
    installed_file_inventory = []
    allowed_roots = (
        V3_SITE_PACKAGES,
        V3_PYTHON_EXECUTABLE.parent,
    )
    for entry in sorted_entries:
        located = Path(distribution.locate_file(entry)).resolve(strict=True)
        if not any(_is_same_or_descendant(located, root) for root in allowed_roots):
            raise AuthorityError(
                f"required distribution file is outside the frozen venv: {distribution_name}"
            )
        descriptor = os.open(located, os.O_RDONLY | os.O_NOFOLLOW)
        try:
            observed = os.fstat(descriptor)
            if not stat.S_ISREG(observed.st_mode):
                raise AuthorityError(
                    f"required distribution contains a non-regular file: {distribution_name}"
                )
            data = _read_all(descriptor, maximum_bytes=256 * 1024 * 1024)
            after = os.fstat(descriptor)
            if (
                observed.st_dev,
                observed.st_ino,
                observed.st_size,
                observed.st_mtime_ns,
                observed.st_ctime_ns,
                observed.st_mode,
            ) != (
                after.st_dev,
                after.st_ino,
                after.st_size,
                after.st_mtime_ns,
                after.st_ctime_ns,
                after.st_mode,
            ):
                raise AuthorityError(
                    f"required distribution file changed during hashing: {distribution_name}"
                )
            installed_file_inventory.append(
                {
                    "path": entry.as_posix(),
                    "size_bytes": observed.st_size,
                    "mode": f"{stat.S_IMODE(observed.st_mode):04o}",
                    "file_sha256": file_sha256(data),
                }
            )
        finally:
            os.close(descriptor)
    native_inventory = [
        entry
        for entry in installed_file_inventory
        if entry["path"].endswith((".so", ".dylib", ".dll", ".pyd"))
    ]
    raw_module_path = Path(module_spec.origin)
    module_path = raw_module_path.resolve(strict=True)
    if module_path != raw_module_path or not _is_same_or_descendant(module_path, V3_SITE_PACKAGES):
        raise AuthorityError(
            f"required module resolved outside the exact frozen site-packages: {module_name}"
        )
    return MappingProxyType(
        {
            "distribution": distribution_name,
            "module": module_name,
            "version": str(distribution.version),
            "module_version": str(distribution.version),
            "module_file_path": os.fspath(module_path),
            "record_file_sha256": _hash_installed_regular_file(record_path),
            "module_file_sha256": _hash_installed_regular_file(module_path),
            "installed_file_inventory": installed_file_inventory,
            "installed_file_inventory_sha256": hashlib.sha256(
                canonical_json_bytes({"files": installed_file_inventory})
            ).hexdigest(),
            "native_binary_inventory": native_inventory,
            "native_binary_inventory_sha256": hashlib.sha256(
                canonical_json_bytes({"files": native_inventory})
            ).hexdigest(),
        }
    )


def _capture_module_report(module_name: str, function_name: str) -> str:
    module = importlib.import_module(module_name)
    reporter = getattr(module, function_name, None)
    if not callable(reporter):
        raise AuthorityError(f"{module_name} has no callable {function_name} reporter")
    output = io.StringIO()
    with contextlib.redirect_stdout(output):
        reporter()
    text = output.getvalue()
    if not text or "\x00" in text:
        raise AuthorityError(f"{module_name} {function_name} output is invalid")
    return text


def _capture_numerical_runtime_inventory() -> Mapping[str, Any]:
    """Capture the exact Python/package/native numerical environment."""

    executable = Path(sys.executable).resolve(strict=True)
    configured_executable = V3_PYTHON_EXECUTABLE.resolve(strict=True)
    complete_site_packages_inventory = _capture_complete_site_packages_inventory()
    distributions = [
        dict(_distribution_runtime_record(distribution, module))
        for distribution, module in _NUMERICAL_DISTRIBUTIONS
    ]
    numpy_config = _capture_module_report("numpy", "show_config")
    numpy_runtime = _capture_module_report("numpy", "show_runtime")
    scipy_config = _capture_module_report("scipy", "show_config")
    return MappingProxyType(
        {
            "schema": ("cfeg.metadata-calibration-efficiency-v3.numerical-runtime-inventory.v1"),
            "fixed_subprocess_environment": dict(_OBSERVED_TEST_ENVIRONMENT),
            "effective_numerical_environment": {
                key: _OBSERVED_TEST_ENVIRONMENT.get(key) for key in _NUMERICAL_ENVIRONMENT_KEYS
            },
            "platform": {
                "platform": platform.platform(),
                "machine": platform.machine(),
                "processor": platform.processor(),
                "libc": list(platform.libc_ver()),
                "uname": list(platform.uname()),
            },
            "python": {
                "version": sys.version,
                "implementation_name": sys.implementation.name,
                "implementation_cache_tag": sys.implementation.cache_tag,
                "soabi": sysconfig.get_config_var("SOABI"),
                "multiarch": sysconfig.get_config_var("MULTIARCH"),
                "executable_path": os.fspath(executable),
                "executable_file_sha256": _hash_installed_regular_file(executable),
                "configured_executable_path": os.fspath(configured_executable),
                "configured_executable_file_sha256": _hash_installed_regular_file(
                    configured_executable
                ),
            },
            "git": {
                "executable_path": os.fspath(V3_GIT_EXECUTABLE),
                "executable_file_sha256": _hash_installed_regular_file(V3_GIT_EXECUTABLE),
                "version": _run_git_version(),
            },
            "python_site_inventory": complete_site_packages_inventory,
            "distributions": distributions,
            "numpy_build_configuration": numpy_config,
            "numpy_build_configuration_sha256": file_sha256(numpy_config.encode("utf-8")),
            "numpy_runtime_configuration": numpy_runtime,
            "numpy_runtime_configuration_sha256": file_sha256(numpy_runtime.encode("utf-8")),
            "scipy_build_configuration": scipy_config,
            "scipy_build_configuration_sha256": file_sha256(scipy_config.encode("utf-8")),
        }
    )


def numerical_runtime_fingerprint_sha256(inventory: Mapping[str, Any]) -> str:
    return hashlib.sha256(canonical_json_bytes(inventory)).hexdigest()


def _current_numerical_runtime() -> tuple[Mapping[str, Any], str]:
    inventory = _capture_numerical_runtime_inventory()
    return inventory, numerical_runtime_fingerprint_sha256(inventory)


def _require_current_numerical_runtime_binding(
    expected_inventory: Mapping[str, Any],
    expected_fingerprint: str,
) -> None:
    _require_sha256(expected_fingerprint, "numerical runtime fingerprint")
    if numerical_runtime_fingerprint_sha256(expected_inventory) != expected_fingerprint:
        raise AuthorityError("stored numerical runtime inventory hash is invalid")
    expected_site_packages = _require_python_site_inventory(
        expected_inventory.get("python_site_inventory")
    )
    current_site_packages = _capture_complete_site_packages_inventory()
    if _plain_json(current_site_packages) != _plain_json(expected_site_packages):
        raise AuthorityError(
            "complete site-packages changed before external runtime code was trusted"
        )
    current_inventory, current_fingerprint = _current_numerical_runtime()
    if current_fingerprint != expected_fingerprint or _plain_json(current_inventory) != _plain_json(
        expected_inventory
    ):
        raise AuthorityError("current numerical runtime differs from frozen inventory")


def _require_frozen_numerical_executor_environment() -> None:
    for key in _NUMERICAL_ENVIRONMENT_KEYS:
        if os.environ.get(key) != _OBSERVED_TEST_ENVIRONMENT.get(key):
            raise AuthorityError(f"numerical executor environment differs at frozen key {key}")


def frozen_numerical_executor_environment() -> Mapping[str, str]:
    """Return the immutable environment for fresh production phase execs."""

    return MappingProxyType(dict(_OBSERVED_TEST_ENVIRONMENT))


def governed_python_subprocess_prefix() -> tuple[str, ...]:
    """Return the only Python executable/isolation prefix allowed by V3."""

    return _GOVERNED_PYTHON_SUBPROCESS_PREFIX


def _load_one_exact_file(path: Path, *, maximum_bytes: int) -> bytes:
    parts = _path_parts(path)
    if not parts:
        raise ValidationError("artifact path cannot be filesystem root")
    parent = Path("/", *parts[:-1])
    parent_fd = _open_absolute_directory(parent, create=False, require_private_final=True)
    try:
        descriptor = os.open(parts[-1], os.O_RDONLY | os.O_NOFOLLOW, dir_fd=parent_fd)
        try:
            observed = os.fstat(descriptor)
            if (
                not stat.S_ISREG(observed.st_mode)
                or observed.st_nlink != 1
                or observed.st_uid != os.geteuid()
                or stat.S_IMODE(observed.st_mode) != 0o400
            ):
                raise ValidationError(
                    "artifact must be owner-controlled, mode-0400, and singly linked"
                )
            return _read_all(descriptor, maximum_bytes=maximum_bytes)
        finally:
            os.close(descriptor)
    finally:
        os.close(parent_fd)


def load_exact_artifact_bytes(
    requested_paths: Sequence[str | Path],
    *,
    declared_paths: Sequence[str | Path],
    maximum_bytes_per_file: int = 64 * 1024 * 1024,
) -> Mapping[str, bytes]:
    """Load exactly the declared files through no-follow directory descriptors."""

    if maximum_bytes_per_file <= 0:
        raise ValidationError("loader size limit must be positive")
    requested_paths_checked = tuple(_canonical_absolute_path(path) for path in requested_paths)
    declared_paths_checked = tuple(_canonical_absolute_path(path) for path in declared_paths)
    requested = tuple(os.fspath(path) for path in requested_paths_checked)
    declared = tuple(os.fspath(path) for path in declared_paths_checked)
    if len(set(requested)) != len(requested) or len(set(declared)) != len(declared):
        raise ValidationError("duplicate exact artifact path")
    if frozenset(requested) != frozenset(declared):
        raise ValidationError("requested paths differ from the exact declared path set")
    loaded: dict[str, bytes] = {}
    for raw_path in sorted(requested):
        path = Path(raw_path)
        _reject_prohibited_artifact_root(path)
        loaded[raw_path] = _load_one_exact_file(path, maximum_bytes=maximum_bytes_per_file)
    return MappingProxyType(loaded)


@dataclass(frozen=True, slots=True)
class PublishedArtifact:
    path: Path
    file_sha256: str
    size_bytes: int

    def __post_init__(self) -> None:
        if not self.path.is_absolute():
            raise ValidationError("published artifact path must be absolute")
        _require_sha256(self.file_sha256, "published artifact file_sha256")
        if self.size_bytes < 0:
            raise ValidationError("published artifact size cannot be negative")


def _publish_write_once(
    trusted_root: str | Path,
    relative_path: str | PurePosixPath,
    data: bytes,
    *,
    create_root: bool = False,
    require_governed_process: bool = True,
    _publisher: object,
) -> PublishedArtifact:
    """Publish bytes once using a single validated parent directory descriptor.

    A failure after final-file creation deliberately leaves that path consumed;
    callers must never replace or retry it.
    """

    if _publisher is not _PUBLICATION_ISSUER:
        raise AuthorityError("write-once publication requires an internal validated publisher")
    if require_governed_process:
        _require_active_governed_process(
            recheck_site_packages=True,
            recheck_source=True,
            require_mutation_role=True,
        )
    if not isinstance(data, bytes) or not data:
        raise ValidationError("write-once publication requires nonempty immutable bytes")
    root = _canonical_absolute_path(trusted_root)
    _reject_prohibited_artifact_root(root)
    parts = _relative_parts(relative_path)
    root_fd = _open_absolute_directory(root, create=create_root)
    descriptor = root_fd
    try:
        flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
        for part in parts[:-1]:
            try:
                child = os.open(part, flags, dir_fd=descriptor)
            except FileNotFoundError:
                try:
                    os.mkdir(part, 0o700, dir_fd=descriptor)
                except FileExistsError:
                    pass
                child = os.open(part, flags, dir_fd=descriptor)
            _validate_directory_fd(child, exact_private_mode=True)
            if descriptor != root_fd:
                os.close(descriptor)
            descriptor = child

        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW
        try:
            output = os.open(parts[-1], flags, 0o400, dir_fd=descriptor)
        except FileExistsError as exc:
            raise PublicationError("write-once artifact path is already consumed") from exc
        try:
            observed = os.fstat(output)
            if (
                not stat.S_ISREG(observed.st_mode)
                or observed.st_nlink != 1
                or observed.st_uid != os.geteuid()
                or stat.S_IMODE(observed.st_mode) != 0o400
            ):
                raise PublicationError("new artifact failed regular-file ownership/mode checks")
            view = memoryview(data)
            offset = 0
            while offset < len(view):
                written = os.write(output, view[offset:])
                if written <= 0:
                    raise PublicationError("short write consumed the artifact path")
                offset += written
            os.fsync(output)
        finally:
            os.close(output)
        os.fsync(descriptor)
    finally:
        if descriptor != root_fd:
            os.close(descriptor)
        os.close(root_fd)
    return PublishedArtifact(
        path=root.joinpath(*parts),
        file_sha256=file_sha256(data),
        size_bytes=len(data),
    )


def _publish_write_once_under_test_root(
    test_root: str | Path,
    relative_path: str | PurePosixPath,
    data: bytes,
    *,
    create_root: bool = False,
) -> PublishedArtifact:
    """Exercise the filesystem primitive without granting canonical publication access."""

    root = _canonical_absolute_path(test_root)
    _reject_prohibited_artifact_root(root)
    parts = _relative_parts(relative_path)
    destination = root.joinpath(*parts)
    try:
        resolved_root = root.resolve(strict=False)
    except (OSError, RuntimeError) as exc:
        raise AuthorityError("test-only publication root cannot be resolved safely") from exc
    resolved_destination = resolved_root.joinpath(*parts)
    for protected in (V3_CANONICAL_ROOT, V3_SOURCE_REPOSITORY):
        resolved_protected = protected.resolve(strict=False)
        if any(
            _is_same_or_descendant(candidate, boundary)
            for candidate in (root, destination, resolved_root, resolved_destination)
            for boundary in (protected, resolved_protected)
        ):
            raise AuthorityError("test-only publisher cannot target a production namespace")
    temporary_root = Path("/tmp").resolve(strict=True)
    if resolved_root == temporary_root or not _is_same_or_descendant(
        resolved_root,
        temporary_root,
    ):
        raise AuthorityError(
            "test-only publication root must be an owner-private namespace below /tmp"
        )
    return _publish_write_once(
        root,
        PurePosixPath(*parts),
        data,
        create_root=create_root,
        require_governed_process=False,
        _publisher=_PUBLICATION_ISSUER,
    )


@dataclass(frozen=True, slots=True)
class TrackedSourceFile:
    path: str
    kind: str
    file_sha256: str

    def __post_init__(self) -> None:
        pure = PurePosixPath(self.path)
        if (
            pure.is_absolute()
            or not pure.parts
            or pure.as_posix() != self.path
            or any(part in ("", ".", "..") for part in pure.parts)
            or "\x00" in self.path
        ):
            raise ValidationError("tracked source path must be canonical and repository-relative")
        if self.kind not in {"regular_file", "executable_file"}:
            raise ValidationError("unsupported tracked source kind")
        _require_sha256(self.file_sha256, "tracked source file_sha256")

    def as_record(self) -> dict[str, str]:
        return {"path": self.path, "kind": self.kind, "file_sha256": self.file_sha256}


def _require_loaded_governance_source_identity(
    bundle: DevelopmentBundleCapability,
) -> None:
    relative = "src/cfeg/metadata_calibration_v3_governance.py"
    expected_path = V3_SOURCE_REPOSITORY / relative
    expected_hash = next(
        (
            entry.file_sha256
            for entry in bundle.tracked_source_files
            if entry.path == relative and entry.kind == "regular_file"
        ),
        None,
    )
    if (
        _GOVERNANCE_IMPORT_SOURCE_PATH != expected_path
        or expected_hash is None
        or _GOVERNANCE_IMPORT_SOURCE_FILE_SHA256 != expected_hash
        or _hash_installed_regular_file(expected_path) != expected_hash
    ):
        raise AuthorityError(
            "loaded governance module does not match the exact bundle-A source bytes"
        )


def _require_loaded_governance_source_at_snapshot(
    snapshot: CleanSourceSnapshot,
) -> None:
    """Bind this already-loaded module to an exact current source snapshot."""

    snapshot = _require_clean_source_snapshot(snapshot)
    relative = "src/cfeg/metadata_calibration_v3_governance.py"
    expected_path = V3_SOURCE_REPOSITORY / relative
    expected_hash = next(
        (
            entry.file_sha256
            for entry in snapshot.tracked_files
            if entry.path == relative and entry.kind == "regular_file"
        ),
        None,
    )
    if (
        _GOVERNANCE_IMPORT_SOURCE_PATH != expected_path
        or expected_hash is None
        or _GOVERNANCE_IMPORT_SOURCE_FILE_SHA256 != expected_hash
        or _hash_installed_regular_file(expected_path) != expected_hash
    ):
        raise AuthorityError(
            "loaded governance module differs from the exact current source snapshot"
        )


def _source_inventory_digest(files: Sequence[TrackedSourceFile]) -> str:
    records = [entry.as_record() for entry in files]
    return hashlib.sha256(canonical_json_bytes({"files": records})).hexdigest()


@dataclass(frozen=True, slots=True, init=False)
class CleanSourceSnapshot:
    identity: CleanGitIdentity
    tracked_files: tuple[TrackedSourceFile, ...]
    source_bundle_sha256: str
    _validation_marker: object = field(repr=False, compare=False)

    def __init__(
        self,
        *,
        identity: CleanGitIdentity,
        tracked_files: tuple[TrackedSourceFile, ...],
        _issuer: object,
    ) -> None:
        if _issuer is not _CAPABILITY_ISSUER:
            raise AuthorityError("CleanSourceSnapshot has no valid clean-git issuer")
        if not tracked_files:
            raise ValidationError("tracked source inventory cannot be empty")
        paths = tuple(entry.path for entry in tracked_files)
        if paths != tuple(sorted(paths)) or len(set(paths)) != len(paths):
            raise ValidationError("tracked source inventory must be complete, unique, and sorted")
        object.__setattr__(self, "identity", identity)
        object.__setattr__(self, "tracked_files", tracked_files)
        object.__setattr__(self, "source_bundle_sha256", _source_inventory_digest(tracked_files))
        object.__setattr__(self, "_validation_marker", _issuer)


def _require_clean_source_snapshot(value: object) -> CleanSourceSnapshot:
    if type(value) is not CleanSourceSnapshot or value._validation_marker is not _CAPABILITY_ISSUER:
        raise AuthorityError("exact validator-issued CleanSourceSnapshot required")
    return value


def _run_git(repository: Path, arguments: Sequence[str]) -> bytes:
    root = _canonical_absolute_path(repository)
    if root.resolve(strict=True) != root:
        raise ValidationError("git repository path must not traverse a symlink")
    completed = subprocess.run(
        [
            os.fspath(V3_GIT_EXECUTABLE),
            "--no-optional-locks",
            "-c",
            "core.fsmonitor=false",
            "-c",
            "core.hooksPath=/dev/null",
            "-C",
            os.fspath(root),
            *arguments,
        ],
        check=False,
        stdin=subprocess.DEVNULL,
        capture_output=True,
        env=dict(_STRICT_GIT_ENVIRONMENT),
    )
    if completed.returncode != 0:
        detail = completed.stderr.decode("utf-8", errors="replace").strip()
        raise ValidationError(f"git command failed: {detail}")
    return completed.stdout


def _require_nonsparse_unmodified_index(repository: Path) -> tuple[str, ...]:
    """Reject index hints that can hide working-tree bytes from Git status."""

    raw_config = _run_git(repository, ("config", "--local", "--null", "--list"))
    for raw_record in raw_config.split(b"\x00"):
        if not raw_record:
            continue
        raw_key, separator, raw_value = raw_record.partition(b"\n")
        try:
            key = raw_key.decode("utf-8").casefold()
            value = raw_value.decode("utf-8").casefold()
        except UnicodeDecodeError as exc:
            raise ValidationError("local Git configuration is not UTF-8") from exc
        if key not in {"core.sparsecheckout", "core.sparsecheckoutcone"}:
            continue
        if not separator or value not in {"true", "false", "yes", "no", "on", "off", "1", "0"}:
            raise ValidationError("sparse-checkout Git configuration is not canonical boolean")
        if value in {"true", "yes", "on", "1"}:
            raise AuthorityError("sparse checkout is prohibited for source authority")

    sparse_patterns = repository / ".git/info/sparse-checkout"
    try:
        os.lstat(sparse_patterns)
    except FileNotFoundError:
        pass
    else:
        raise AuthorityError("sparse-checkout pattern state is prohibited for source authority")

    raw_index = _run_git(repository, ("ls-files", "-v", "-z"))
    paths: list[str] = []
    for raw_record in raw_index.split(b"\x00"):
        if not raw_record:
            continue
        if len(raw_record) < 3 or raw_record[1:2] != b" ":
            raise ValidationError("Git index emitted a malformed tracked-path record")
        if raw_record[:1] != b"H":
            raise AuthorityError(
                "assume-unchanged, skip-worktree, or special Git index state is prohibited"
            )
        try:
            path = raw_record[2:].decode("utf-8")
        except UnicodeDecodeError as exc:
            raise ValidationError("Git index contains a non-UTF-8 tracked path") from exc
        # Reuse the canonical path validation without trusting an index hash.
        TrackedSourceFile(path=path, kind="regular_file", file_sha256="0" * 64)
        paths.append(path)
    if tuple(paths) != tuple(sorted(paths)) or len(paths) != len(set(paths)):
        raise ValidationError("Git index paths must be unique and sorted")
    return tuple(paths)


def _tracked_worktree_file_sha256(
    root_descriptor: int,
    entry: TrackedSourceFile,
) -> str:
    """Hash one tracked disk file through no-follow directory descriptors."""

    descriptor = os.dup(root_descriptor)
    file_descriptor: int | None = None
    try:
        flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
        for component in PurePosixPath(entry.path).parts[:-1]:
            child = os.open(component, flags, dir_fd=descriptor)
            _validate_directory_fd(child, exact_private_mode=False)
            os.close(descriptor)
            descriptor = child
        file_descriptor = os.open(
            PurePosixPath(entry.path).parts[-1],
            os.O_RDONLY | os.O_NOFOLLOW,
            dir_fd=descriptor,
        )
        before = os.fstat(file_descriptor)
        observed_mode = stat.S_IMODE(before.st_mode)
        expected_executable = entry.kind == "executable_file"
        observed_executable = (observed_mode & 0o111) == 0o111
        if (
            not stat.S_ISREG(before.st_mode)
            or before.st_uid != os.geteuid()
            or before.st_nlink != 1
            or not observed_mode & stat.S_IRUSR
            or observed_mode & 0o7000
            or observed_mode & stat.S_IWOTH
            or observed_executable != expected_executable
            or (not expected_executable and observed_mode & 0o111)
        ):
            raise AuthorityError(
                "tracked working-tree file type, owner, link count, or Git mode is unsafe"
            )
        data = _read_all(file_descriptor, maximum_bytes=512 * 1024 * 1024)
        after = os.fstat(file_descriptor)
        before_identity = (
            before.st_dev,
            before.st_ino,
            before.st_size,
            before.st_mtime_ns,
            before.st_ctime_ns,
            before.st_mode,
            before.st_uid,
            before.st_nlink,
        )
        after_identity = (
            after.st_dev,
            after.st_ino,
            after.st_size,
            after.st_mtime_ns,
            after.st_ctime_ns,
            after.st_mode,
            after.st_uid,
            after.st_nlink,
        )
        if before_identity != after_identity:
            raise AuthorityError("tracked working-tree file changed while it was hashed")
        return file_sha256(data)
    except OSError as exc:
        raise AuthorityError(
            f"tracked working-tree path could not be opened safely: {entry.path}"
        ) from exc
    finally:
        if file_descriptor is not None:
            os.close(file_descriptor)
        os.close(descriptor)


def _require_worktree_matches_committed_inventory(
    repository: Path,
    entries: Sequence[TrackedSourceFile],
) -> None:
    root_descriptor = _open_absolute_directory(
        repository,
        create=False,
        require_private_final=False,
    )
    try:
        for entry in entries:
            if _tracked_worktree_file_sha256(root_descriptor, entry) != entry.file_sha256:
                raise AuthorityError(
                    f"tracked working-tree bytes differ from committed blob: {entry.path}"
                )
    finally:
        os.close(root_descriptor)


def _require_safe_source_directory(
    observed: os.stat_result,
    *,
    description: str,
) -> None:
    mode = stat.S_IMODE(observed.st_mode)
    if (
        not stat.S_ISDIR(observed.st_mode)
        or observed.st_uid != os.geteuid()
        or mode & 0o7000
        or mode & stat.S_IWOTH
    ):
        raise AuthorityError(f"{description} is not an owner-controlled source directory")


def _execution_inventory_trees(
    entries: Sequence[TrackedSourceFile],
) -> Mapping[str, Mapping[str, Any]]:
    trees: dict[str, dict[str, Any]] = {
        root_name: {} for root_name in _CANONICAL_PYTHON_EXECUTION_ROOTS
    }
    for entry in entries:
        parts = PurePosixPath(entry.path).parts
        if parts[0] not in trees:
            continue
        if len(parts) == 1:
            raise AuthorityError("canonical Python execution roots must be directories")
        node = trees[parts[0]]
        for component in parts[1:-1]:
            existing = node.get(component)
            if isinstance(existing, TrackedSourceFile):
                raise AuthorityError("tracked source inventory has a file/directory collision")
            if existing is None:
                existing = {}
                node[component] = existing
            node = existing
        leaf = parts[-1]
        if leaf in node:
            raise AuthorityError("tracked source inventory has a duplicate tree entry")
        node[leaf] = entry
    return MappingProxyType(trees)


def _scan_exact_execution_directory(
    descriptor: int,
    expected: Mapping[str, Any],
    *,
    relative_path: PurePosixPath,
) -> None:
    try:
        actual_names = tuple(sorted(os.listdir(descriptor)))
    except OSError as exc:
        raise AuthorityError("canonical execution directory could not be enumerated") from exc
    if actual_names != tuple(sorted(expected)):
        raise AuthorityError(
            f"canonical execution tree contains untracked or missing entries: {relative_path}"
        )
    directory_flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
    for name in actual_names:
        child_path = relative_path / name
        expected_child = expected[name]
        try:
            listed = os.stat(name, dir_fd=descriptor, follow_symlinks=False)
        except OSError as exc:
            raise AuthorityError(
                f"canonical execution entry could not be inspected safely: {child_path}"
            ) from exc
        if isinstance(expected_child, Mapping):
            _require_safe_source_directory(
                listed,
                description=f"canonical execution directory {child_path}",
            )
            try:
                child_descriptor = os.open(name, directory_flags, dir_fd=descriptor)
            except OSError as exc:
                raise AuthorityError(
                    f"canonical execution directory could not be opened safely: {child_path}"
                ) from exc
            try:
                opened = os.fstat(child_descriptor)
                if (opened.st_dev, opened.st_ino) != (listed.st_dev, listed.st_ino):
                    raise AuthorityError(
                        f"canonical execution directory changed during traversal: {child_path}"
                    )
                _require_safe_source_directory(
                    opened,
                    description=f"canonical execution directory {child_path}",
                )
                _scan_exact_execution_directory(
                    child_descriptor,
                    expected_child,
                    relative_path=child_path,
                )
            finally:
                os.close(child_descriptor)
        elif isinstance(expected_child, TrackedSourceFile):
            if not stat.S_ISREG(listed.st_mode):
                raise AuthorityError(
                    f"canonical execution file is not a no-follow regular file: {child_path}"
                )
        else:  # pragma: no cover - constructed only by _execution_inventory_trees.
            raise AuthorityError("tracked execution inventory node has an invalid type")


def _require_exact_python_execution_filesystem(
    repository: Path,
    entries: Sequence[TrackedSourceFile],
) -> None:
    """Reject ignored or untracked code and import controls without asking Git."""

    root_descriptor = _open_absolute_directory(
        repository,
        create=False,
        require_private_final=False,
    )
    try:
        root_stat = os.fstat(root_descriptor)
        _require_safe_source_directory(root_stat, description="source repository root")
        trees = _execution_inventory_trees(entries)
        tracked_by_path = {entry.path: entry for entry in entries}
        directory_flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
        for root_name in _CANONICAL_PYTHON_EXECUTION_ROOTS:
            expected_tree = trees[root_name]
            try:
                listed = os.stat(
                    root_name,
                    dir_fd=root_descriptor,
                    follow_symlinks=False,
                )
            except FileNotFoundError:
                if expected_tree:
                    raise AuthorityError(
                        f"tracked canonical execution root is absent: {root_name}"
                    ) from None
                continue
            if not expected_tree:
                raise AuthorityError(
                    f"canonical execution root is external to tracked inventory: {root_name}"
                )
            _require_safe_source_directory(
                listed,
                description=f"canonical execution root {root_name}",
            )
            try:
                child_descriptor = os.open(
                    root_name,
                    directory_flags,
                    dir_fd=root_descriptor,
                )
            except OSError as exc:
                raise AuthorityError(
                    f"canonical execution root could not be opened safely: {root_name}"
                ) from exc
            try:
                opened = os.fstat(child_descriptor)
                if (opened.st_dev, opened.st_ino) != (listed.st_dev, listed.st_ino):
                    raise AuthorityError(
                        f"canonical execution root changed during traversal: {root_name}"
                    )
                _scan_exact_execution_directory(
                    child_descriptor,
                    expected_tree,
                    relative_path=PurePosixPath(root_name),
                )
            finally:
                os.close(child_descriptor)

        try:
            root_names = tuple(os.listdir(root_descriptor))
        except OSError as exc:
            raise AuthorityError("source repository root could not be enumerated") from exc
        controls = _ROOT_PYTEST_IMPORT_CONTROL_NAMES | {
            name for name in root_names if name.endswith(".pth")
        }
        for name in controls:
            record = tracked_by_path.get(name)
            try:
                observed = os.stat(
                    name,
                    dir_fd=root_descriptor,
                    follow_symlinks=False,
                )
            except FileNotFoundError:
                if record is not None:
                    raise AuthorityError(f"tracked root Python control is absent: {name}") from None
                continue
            if record is None or record.kind != "regular_file":
                raise AuthorityError(
                    f"root Python/pytest control is external to tracked inventory: {name}"
                )
            if not stat.S_ISREG(observed.st_mode):
                raise AuthorityError(f"root Python/pytest control is not regular: {name}")
    finally:
        os.close(root_descriptor)


def _validate_git_repository_root(repository: Path) -> Path:
    _require_pinned_git_executable()
    root = _canonical_absolute_path(repository)
    if root.resolve(strict=True) != root:
        raise ValidationError("git repository root must be an exact nonsymlink path")
    try:
        top = Path(_run_git(root, ("rev-parse", "--show-toplevel")).decode("utf-8").strip())
        git_directory = Path(
            _run_git(root, ("rev-parse", "--absolute-git-dir")).decode("utf-8").strip()
        )
    except UnicodeDecodeError as exc:
        raise ValidationError("git repository paths were not UTF-8") from exc
    if top != root or top.resolve(strict=True) != root:
        raise AuthorityError("git top-level differs from the exact requested repository")
    expected_git_directory = root / ".git"
    if git_directory != expected_git_directory or not git_directory.is_dir():
        raise AuthorityError("git directory differs from the exact repository-local .git")
    alternates = git_directory / "objects/info/alternates"
    if alternates.exists():
        raise AuthorityError("alternate Git object stores are prohibited")
    if _run_git(root, ("for-each-ref", "--format=%(refname)", "refs/replace")):
        raise AuthorityError("Git replace references are prohibited")
    return root


def capture_clean_source_snapshot(repository: str | Path) -> CleanSourceSnapshot:
    """Capture every tracked blob from an already-clean git commit.

    The inventory comes from committed objects, then every corresponding disk
    file is independently opened without following symlinks and byte-compared.
    This prevents assume-unchanged, skip-worktree, and sparse-index state from
    hiding source drift from the otherwise clean Git porcelain report.
    """

    root = _validate_git_repository_root(_canonical_absolute_path(repository))
    index_paths = _require_nonsparse_unmodified_index(root)
    status = _run_git(root, ("status", "--porcelain=v1", "--untracked-files=all", "-z"))
    if status:
        raise ValidationError("development bundle requires an already-clean worktree")
    commit = _run_git(root, ("rev-parse", "--verify", "HEAD^{commit}"))
    tree = _run_git(root, ("rev-parse", "--verify", "HEAD^{tree}"))
    try:
        identity = CleanGitIdentity(
            commit=commit.decode("ascii").strip(),
            tree=tree.decode("ascii").strip(),
        )
    except UnicodeDecodeError as exc:
        raise ValidationError("git object IDs were not ASCII") from exc

    raw_tree = _run_git(root, ("ls-tree", "-r", "-z", "--full-tree", identity.commit))
    entries: list[TrackedSourceFile] = []
    for record in raw_tree.split(b"\x00"):
        if not record:
            continue
        try:
            metadata, raw_path = record.split(b"\t", 1)
            mode, object_type, object_id = metadata.decode("ascii").split(" ")
            path = raw_path.decode("utf-8")
        except (ValueError, UnicodeDecodeError) as exc:
            raise ValidationError("git tree contains an unsupported path record") from exc
        if object_type != "blob" or mode not in {"100644", "100755"}:
            raise ValidationError("V3 source inventory permits regular tracked blobs only")
        blob = _run_git(root, ("cat-file", "blob", object_id))
        entries.append(
            TrackedSourceFile(
                path=path,
                kind="executable_file" if mode == "100755" else "regular_file",
                file_sha256=file_sha256(blob),
            )
        )
    entries.sort(key=lambda item: item.path)
    if tuple(entry.path for entry in entries) != index_paths:
        raise AuthorityError("Git index path set differs from the committed source tree")
    _require_worktree_matches_committed_inventory(root, entries)
    _require_exact_python_execution_filesystem(root, entries)
    post_index_paths = _require_nonsparse_unmodified_index(root)
    post_commit = _run_git(root, ("rev-parse", "--verify", "HEAD^{commit}"))
    post_tree = _run_git(root, ("rev-parse", "--verify", "HEAD^{tree}"))
    post_status = _run_git(
        root,
        ("status", "--porcelain=v1", "--untracked-files=all", "-z"),
    )
    if post_commit != commit or post_tree != tree or post_status or post_index_paths != index_paths:
        raise ValidationError("git identity or worktree changed during source capture")
    _require_exact_python_execution_filesystem(root, entries)
    return CleanSourceSnapshot(
        identity=identity,
        tracked_files=tuple(entries),
        _issuer=_CAPABILITY_ISSUER,
    )


def _require_prepublication_current_source(
    expected: CleanSourceSnapshot,
) -> CleanSourceSnapshot:
    """Repeat clean-disk and loaded-code checks immediately before O_EXCL."""

    expected = _require_clean_source_snapshot(expected)
    current = capture_clean_source_snapshot(V3_SOURCE_REPOSITORY)
    if not _same_source_snapshot(current, expected):
        raise AuthorityError("current source differs immediately before publication")
    _require_loaded_governance_source_at_snapshot(current)
    return current


def _require_current_source_identity(
    *,
    commit: str,
    tree: str,
) -> CleanSourceSnapshot:
    """Require the exact clean B identity when only durable IDs are upstream."""

    _require_git_object(commit, "prepublication source commit")
    _require_git_object(tree, "prepublication source tree")
    current = capture_clean_source_snapshot(V3_SOURCE_REPOSITORY)
    if current.identity.commit != commit or current.identity.tree != tree:
        raise AuthorityError("current clean source identity differs before publication")
    _require_loaded_governance_source_at_snapshot(current)
    return current


def _capture_committed_source_snapshot(
    repository: Path,
    commit: str,
) -> CleanSourceSnapshot:
    """Reconstruct an immutable historical source snapshot from Git objects."""

    root = _validate_git_repository_root(_canonical_absolute_path(repository))
    _require_git_object(commit, "historical source commit")
    resolved = _run_git(root, ("rev-parse", "--verify", f"{commit}^{{commit}}"))
    if resolved.decode("ascii").strip() != commit:
        raise AuthorityError("historical source commit did not resolve exactly")
    tree = _run_git(root, ("rev-parse", "--verify", f"{commit}^{{tree}}"))
    identity = CleanGitIdentity(commit=commit, tree=tree.decode("ascii").strip())
    raw_tree = _run_git(root, ("ls-tree", "-r", "-z", "--full-tree", commit))
    entries: list[TrackedSourceFile] = []
    for record in raw_tree.split(b"\x00"):
        if not record:
            continue
        try:
            metadata, raw_path = record.split(b"\t", 1)
            mode, object_type, object_id = metadata.decode("ascii").split(" ")
            path = raw_path.decode("utf-8")
        except (ValueError, UnicodeDecodeError) as exc:
            raise ValidationError("git tree contains an unsupported path record") from exc
        if object_type != "blob" or mode not in {"100644", "100755"}:
            raise ValidationError("V3 source inventory permits regular tracked blobs only")
        blob = _run_git(root, ("cat-file", "blob", object_id))
        entries.append(
            TrackedSourceFile(
                path=path,
                kind="executable_file" if mode == "100755" else "regular_file",
                file_sha256=file_sha256(blob),
            )
        )
    return CleanSourceSnapshot(
        identity=identity,
        tracked_files=tuple(sorted(entries, key=lambda item: item.path)),
        _issuer=_CAPABILITY_ISSUER,
    )


def _load_committed_source_blob(
    repository: Path,
    relative_path: PurePosixPath,
    *,
    snapshot: CleanSourceSnapshot,
    production: bool,
) -> bytes:
    """Read one exact tracked blob while proving the clean snapshot stayed fixed."""

    snapshot = _require_clean_source_snapshot(snapshot)
    root = _canonical_absolute_path(repository)
    if production and root != V3_SOURCE_REPOSITORY:
        raise AuthorityError("production committed-source validation requires canonical repository")
    if relative_path.is_absolute() or any(part in {"", ".", ".."} for part in relative_path.parts):
        raise ValidationError("committed source path must be exact and repository-relative")
    before = capture_clean_source_snapshot(root)
    if not _same_source_snapshot(before, snapshot):
        raise AuthorityError("committed-source repository differs from approved snapshot")
    path_text = relative_path.as_posix()
    record = next((item for item in snapshot.tracked_files if item.path == path_text), None)
    if record is None or record.kind != "regular_file":
        raise AuthorityError("exact committed source artifact is absent or not a regular blob")
    blob = _run_git(root, ("show", f"{snapshot.identity.commit}:{path_text}"))
    if file_sha256(blob) != record.file_sha256:
        raise AuthorityError("committed source blob differs from captured inventory")
    after = capture_clean_source_snapshot(root)
    if not _same_source_snapshot(after, snapshot):
        raise AuthorityError("repository changed during committed-source validation")
    return blob


def artifact_schema_inventory_sha256() -> str:
    return hashlib.sha256(canonical_json_bytes(ARTIFACT_SCHEMAS)).hexdigest()


def _ssh_public_key_fingerprint(public_key_bytes: bytes) -> str:
    try:
        fields = public_key_bytes.strip().split()
        if len(fields) < 2 or fields[0] != b"ssh-rsa":
            raise ValidationError("owner authority must be an ssh-rsa public key")
        wire = base64.b64decode(fields[1], validate=True)
    except (binascii.Error, ValueError) as exc:
        raise ValidationError("owner authority SSH public key encoding is malformed") from exc
    return "SHA256:" + base64.b64encode(hashlib.sha256(wire).digest()).decode("ascii").rstrip("=")


def _validate_owner_public_key_bytes(public_key_bytes: bytes) -> tuple[int, int]:
    expected_sha = FROZEN_FILE_SHA256[OWNER_AUTHORITY_PUBLIC_KEY_PATH]
    if file_sha256(public_key_bytes) != expected_sha:
        raise ValidationError("owner authority public-key file hash mismatch")
    if _ssh_public_key_fingerprint(public_key_bytes) != OWNER_AUTHORITY_FINGERPRINT:
        raise ValidationError("owner authority SSH fingerprint mismatch")
    try:
        fields = public_key_bytes.strip().split()
        wire = base64.b64decode(fields[1], validate=True)
        offset = 0

        def read_field() -> bytes:
            nonlocal offset
            if offset + 4 > len(wire):
                raise ValueError
            length = int.from_bytes(wire[offset : offset + 4], "big")
            offset += 4
            if length <= 0 or offset + length > len(wire):
                raise ValueError
            value = wire[offset : offset + length]
            offset += length
            return value

        algorithm = read_field()
        exponent_bytes = read_field()
        modulus_bytes = read_field()
        if offset != len(wire) or algorithm != b"ssh-rsa":
            raise ValueError
        for encoded_integer in (exponent_bytes, modulus_bytes):
            if encoded_integer[0] & 0x80:
                raise ValueError
            if (
                len(encoded_integer) > 1
                and encoded_integer[0] == 0
                and not encoded_integer[1] & 0x80
            ):
                raise ValueError
        exponent = int.from_bytes(exponent_bytes, "big", signed=False)
        modulus = int.from_bytes(modulus_bytes, "big", signed=False)
    except (binascii.Error, IndexError, TypeError, ValueError) as exc:
        raise ValidationError("owner authority public key could not be parsed") from exc
    if exponent != 65_537 or modulus.bit_length() != 4096:
        raise ValidationError("owner authority must be the pinned RSA-4096 key")
    return exponent, modulus


def _validate_frozen_bundle_files(files: Mapping[str, bytes]) -> None:
    if frozenset(files) != frozenset(FROZEN_FILE_SHA256):
        raise ValidationError("bundle evidence must contain the exact frozen file set")
    for path, expected in FROZEN_FILE_SHA256.items():
        data = files[path]
        if not isinstance(data, bytes) or file_sha256(data) != expected:
            raise ValidationError(f"frozen bundle evidence hash mismatch: {path}")
    _validate_owner_public_key_bytes(files[OWNER_AUTHORITY_PUBLIC_KEY_PATH])


def _require_frozen_files_in_source_snapshot(
    snapshot: CleanSourceSnapshot,
    frozen_file_bytes: Mapping[str, bytes],
) -> None:
    """Bind caller bytes to the exact frozen records in a validated commit."""

    snapshot = _require_clean_source_snapshot(snapshot)
    _validate_frozen_bundle_files(frozen_file_bytes)
    _validate_recovery_amendment_bytes(frozen_file_bytes[PREOUTCOME_AMENDMENT_PATH])
    records = {entry.path: entry for entry in snapshot.tracked_files}
    for path, expected_sha256 in FROZEN_FILE_SHA256.items():
        record = records.get(path)
        if (
            record is None
            or record.kind != "regular_file"
            or record.file_sha256 != expected_sha256
            or file_sha256(frozen_file_bytes[path]) != record.file_sha256
        ):
            raise AuthorityError(
                f"frozen file is not the exact regular blob in the source snapshot: {path}"
            )


def _validate_recovery_amendment_bytes(data: bytes) -> Mapping[str, Any]:
    """Validate the transparent V3.2 recovery record before bundle authority exists."""

    if not isinstance(data, bytes):
        raise ValidationError("recovery amendment must be immutable bytes")

    def unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, item in pairs:
            if key in result:
                raise ValidationError("recovery amendment contains a duplicate JSON key")
            result[key] = item
        return result

    def reject_constant(value: str) -> None:
        raise ValidationError(f"recovery amendment contains nonfinite JSON number: {value}")

    try:
        value = json.loads(
            data.decode("utf-8"),
            object_pairs_hook=unique_object,
            parse_constant=reject_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValidationError("recovery amendment is not strict UTF-8 JSON") from exc
    if not isinstance(value, Mapping):
        raise ValidationError("recovery amendment must be a JSON object")
    owner = value.get("owner_approval")
    retired_v1 = value.get("retired_development_v1")
    retired_v2 = value.get("retired_development_v2")
    recovery = value.get("authorized_recovery")
    projection = value.get("scientific_contract_projection")
    original = value.get("original_preoutcome_amendment")
    prior = value.get("prior_recovery_amendment")
    immutability = value.get("immutability")
    if not all(
        isinstance(item, Mapping)
        for item in (
            owner,
            retired_v1,
            retired_v2,
            recovery,
            projection,
            original,
            prior,
            immutability,
        )
    ):
        raise ValidationError("recovery amendment is missing a required object")
    assert isinstance(owner, Mapping)
    assert isinstance(retired_v1, Mapping)
    assert isinstance(retired_v2, Mapping)
    assert isinstance(recovery, Mapping)
    assert isinstance(projection, Mapping)
    assert isinstance(original, Mapping)
    assert isinstance(prior, Mapping)
    assert isinstance(immutability, Mapping)
    failure = retired_v2.get("observed_resume_failure")
    observations = retired_v2.get("execution_observations")
    retired_v1_bundle = retired_v1.get("development_bundle")
    retired_v1_context = retired_v1.get("context_reference")
    retired_v2_bundle = retired_v2.get("development_bundle")
    if not all(
        isinstance(item, Mapping)
        for item in (
            failure,
            observations,
            retired_v1_bundle,
            retired_v1_context,
            retired_v2_bundle,
        )
    ):
        raise ValidationError("recovery amendment lacks incident observations")
    assert isinstance(failure, Mapping)
    assert isinstance(observations, Mapping)
    assert isinstance(retired_v1_bundle, Mapping)
    assert isinstance(retired_v1_context, Mapping)
    assert isinstance(retired_v2_bundle, Mapping)
    encoded = failure.get("emitted_json_line_utf8_base64")
    try:
        emitted = base64.b64decode(encoded, validate=True)
    except (TypeError, binascii.Error) as exc:
        raise ValidationError("recovery incident output is not canonical base64") from exc
    definition = projection.get("canonical_preimage_definition")
    expected_definition = {
        "top_level_keys": ["candidate_id", "master", "schema", "synthetic"],
        "candidate_id_value": CANDIDATE_ID,
        "master_value": ("mapping_of_the_exact_master_keys_below_to_their_parsed_YAML_values"),
        "synthetic_value": (
            "mapping_of_the_exact_synthetic_keys_below_to_their_parsed_YAML_values"
        ),
        "serialization": (
            "UTF-8_JSON_sort_keys_true_separators_comma_colon_ensure_ascii_true_"
            "allow_nan_false_without_trailing_newline"
        ),
    }
    expected_key_sets = (
        (
            value,
            {
                "authorized_recovery",
                "candidate_id",
                "decision_id",
                "immutability",
                "original_preoutcome_amendment",
                "owner_approval",
                "prior_recovery_amendment",
                "protocol_revision",
                "recorded_at_is_amendment_time_not_incident_timestamp",
                "recorded_at_utc",
                "retired_development_v1",
                "retired_development_v2",
                "schema",
                "scientific_contract_projection",
                "scientific_objective_changed",
                "status",
            },
        ),
        (
            projection,
            {
                "canonical_preimage_definition",
                "master_keys",
                "schema",
                "sha256",
                "synthetic_keys",
                "unchanged_components",
            },
        ),
        (definition, set(expected_definition)),
        (
            owner,
            {"approval_basis", "approval_scope", "approved_by", "attestation_note"},
        ),
        (original, {"file_sha256", "must_remain_byte_identical", "path"}),
        (prior, {"file_sha256", "must_remain_byte_identical", "path"}),
        (
            retired_v1,
            {
                "artifact_inventory_schema",
                "artifact_inventory_sha256",
                "attempt_id",
                "canonical_root",
                "context_reference",
                "continuation_authorized",
                "development_bundle",
                "exact_directory_inventory_must_remain_two_files",
                "exact_file_names",
                "source_commit",
                "source_tree",
            },
        ),
        (retired_v1_bundle, {"file_sha256", "path", "payload_sha256", "schema"}),
        (
            retired_v1_context,
            {"file_sha256", "path", "payload_sha256", "schema"},
        ),
        (
            retired_v2,
            {
                "artifact_inventory_schema",
                "artifact_inventory_sha256",
                "attempt_id",
                "canonical_root",
                "continuation_authorized",
                "development_bundle",
                "exact_directory_inventory_must_remain_one_file",
                "exact_file_names",
                "execution_observations",
                "observed_resume_failure",
                "scientific_interpretation",
                "source_bundle_sha256",
                "source_commit",
                "source_tree",
            },
        ),
        (retired_v2_bundle, {"file_sha256", "path", "payload_sha256", "schema"}),
        (
            failure,
            {
                "argv",
                "emitted_json_line_sha256",
                "emitted_json_line_utf8_base64",
                "exception_type",
                "exit_code",
                "failure_boundary",
                "message",
            },
        ),
        (
            observations,
            {
                "canary_artifact_count",
                "context_reference_artifact_count",
                "context_reference_replay_count",
                "context_reference_seed",
                "context_reference_seedsequence_and_covariate_only_replay_executed",
                "development_DGP_executed",
                "development_rng_authority_object_instantiated_in_memory",
                "development_result_artifact_count",
                "development_seed",
                "development_seedsequence_created",
                "development_v2_experimental_workflow_outcome_observation_count",
                "development_v2_governed_execution_network_access_count",
                "selection_artifact_count",
            },
        ),
        (
            recovery,
            {
                "attempt_id",
                "canonical_root",
                "context_reference_authority_policy",
                "context_reference_policy",
                "context_reference_transport_policy",
                "development_authority_policy",
                "development_bundle_schema",
                "development_result_must_bind_start_receipt",
                "development_start_receipt_path",
                "development_start_receipt_policy",
                "development_start_receipt_schema",
                "human_EEG_outcome_authorized",
                "network_access_authorized",
                "only_authorized_development_attempt",
                "receipt_with_result_restart_policy",
                "receipt_without_result_restart_policy",
                "requires_exact_retired_v1_two_file_inventory",
                "requires_exact_retired_v2_one_file_inventory",
                "scientific_lockbox_authorized",
            },
        ),
        (
            immutability,
            {
                "development_v1_files_may_not_be_modified_deleted_or_extended",
                "development_v1_or_v2_schema_may_not_be_reinterpreted_as_v3",
                "development_v2_files_may_not_be_modified_deleted_or_extended",
                "any_failure_after_development_start_receipt_consumes_attempt",
                "future_target_bound_RSA_PSS_authorization_remains_separate_and_unissued",
                "operator_DGP_grid_threshold_gate_rank_seed_or_evidence_change_requires_new_scientific_revision",
                "original_preoutcome_amendment_may_not_be_modified",
                "prior_V3_1_recovery_amendment_may_not_be_modified",
            },
        ),
    )
    if any(not isinstance(item, Mapping) or set(item) != keys for item, keys in expected_key_sets):
        raise ValidationError("recovery amendment field inventory is not exact")
    if (
        value.get("schema") != RECOVERY_AMENDMENT_SCHEMA
        or value.get("candidate_id") != CANDIDATE_ID
        or value.get("decision_id") != "COR-20260907-030"
        or value.get("protocol_revision") != PROTOCOL_REVISION
        or value.get("status") != "preoutcome_infrastructure_recovery_only"
        or value.get("recorded_at_is_amendment_time_not_incident_timestamp") is not True
        or value.get("scientific_objective_changed") is not False
        or _parse_rfc3339_seconds(value.get("recorded_at_utc"), "recovery recorded_at_utc")
        > datetime.now(timezone.utc)
        or projection.get("schema") != SCIENTIFIC_CONTRACT_PROJECTION_SCHEMA
        or projection.get("sha256") != SCIENTIFIC_CONTRACT_PROJECTION_SHA256
        or _plain_json(definition) != expected_definition
        or projection.get("master_keys") != list(_SCIENTIFIC_PROJECTION_MASTER_KEYS)
        or projection.get("synthetic_keys") != list(_SCIENTIFIC_PROJECTION_SYNTHETIC_KEYS)
        or projection.get("unchanged_components")
        != list(_SCIENTIFIC_PROJECTION_UNCHANGED_COMPONENTS)
        or owner
        != {
            "approved_by": "active_workspace_owner",
            "approval_scope": (
                "tracked_preoutcome_V3_2_development_v3_infrastructure_recovery_only"
            ),
            "approval_basis": (
                "explicit_user_instruction_before_any_development_v3_bundle_context_or_"
                "development_execution"
            ),
            "attestation_note": (
                "This repository-local hash-bound receipt records the active user's explicit "
                "approval; it is not a cryptographic signature or a future target-bound "
                "execution authorization."
            ),
        }
        or original.get("path") != ORIGINAL_PREOUTCOME_AMENDMENT_PATH
        or original.get("file_sha256") != FROZEN_FILE_SHA256[ORIGINAL_PREOUTCOME_AMENDMENT_PATH]
        or original.get("must_remain_byte_identical") is not True
        or prior.get("path") != PRIOR_RECOVERY_AMENDMENT_PATH
        or prior.get("file_sha256") != FROZEN_FILE_SHA256[PRIOR_RECOVERY_AMENDMENT_PATH]
        or prior.get("must_remain_byte_identical") is not True
        or retired_v1.get("attempt_id") != "development-v1"
        or retired_v1.get("canonical_root") != os.fspath(RETIRED_V1_DEVELOPMENT_ROOT)
        or retired_v1.get("source_commit") != RETIRED_V1_SOURCE_COMMIT
        or retired_v1.get("source_tree") != RETIRED_V1_SOURCE_TREE
        or retired_v1.get("artifact_inventory_schema") != RETIRED_V1_ARTIFACT_INVENTORY_SCHEMA
        or retired_v1.get("artifact_inventory_sha256") != RETIRED_V1_ARTIFACT_INVENTORY_SHA256
        or retired_v1.get("exact_file_names")
        != ["context-reference.json", "development-bundle.json"]
        or retired_v1_bundle.get("path") != os.fspath(RETIRED_V1_DEVELOPMENT_BUNDLE_PATH)
        or retired_v1_bundle.get("schema") != RETIRED_V1_DEVELOPMENT_BUNDLE_SCHEMA
        or retired_v1_bundle.get("payload_sha256") != RETIRED_V1_BUNDLE_PAYLOAD_SHA256
        or retired_v1_bundle.get("file_sha256") != RETIRED_V1_BUNDLE_FILE_SHA256
        or retired_v1_context.get("path") != os.fspath(RETIRED_V1_CONTEXT_REFERENCE_PATH)
        or retired_v1_context.get("schema") != ARTIFACT_SCHEMAS["context_reference"]
        or retired_v1_context.get("payload_sha256") != RETIRED_V1_CONTEXT_PAYLOAD_SHA256
        or retired_v1_context.get("file_sha256") != RETIRED_V1_CONTEXT_FILE_SHA256
        or retired_v1.get("continuation_authorized") is not False
        or retired_v1.get("exact_directory_inventory_must_remain_two_files") is not True
        or retired_v2.get("attempt_id") != "development-v2"
        or retired_v2.get("canonical_root") != os.fspath(RETIRED_V2_DEVELOPMENT_ROOT)
        or retired_v2.get("source_commit") != RETIRED_V2_SOURCE_COMMIT
        or retired_v2.get("source_tree") != RETIRED_V2_SOURCE_TREE
        or retired_v2.get("source_bundle_sha256") != RETIRED_V2_SOURCE_BUNDLE_SHA256
        or retired_v2.get("artifact_inventory_schema") != RETIRED_V2_ARTIFACT_INVENTORY_SCHEMA
        or retired_v2.get("artifact_inventory_sha256") != RETIRED_V2_ARTIFACT_INVENTORY_SHA256
        or retired_v2.get("exact_file_names") != ["development-bundle.json"]
        or retired_v2_bundle.get("path") != os.fspath(RETIRED_V2_DEVELOPMENT_BUNDLE_PATH)
        or retired_v2_bundle.get("schema") != RETIRED_V2_DEVELOPMENT_BUNDLE_SCHEMA
        or retired_v2_bundle.get("payload_sha256") != RETIRED_V2_BUNDLE_PAYLOAD_SHA256
        or retired_v2_bundle.get("file_sha256") != RETIRED_V2_BUNDLE_FILE_SHA256
        or failure.get("argv") != ["scripts/run_metadata_calibration_v3", "resume"]
        or failure.get("exit_code") != 2
        or failure.get("exception_type") != RETIRED_V2_INCIDENT_ERROR
        or failure.get("message") != RETIRED_V2_INCIDENT_MESSAGE
        or failure.get("emitted_json_line_sha256") != RETIRED_V2_INCIDENT_OUTPUT_SHA256
        or failure.get("failure_boundary")
        != "after_context_seed_20260910_reference_replay_before_core_proof_and_context_publication"
        or emitted != RETIRED_V2_INCIDENT_OUTPUT_BYTES
        or file_sha256(emitted) != RETIRED_V2_INCIDENT_OUTPUT_SHA256
        or observations.get("context_reference_seed") != 20_260_910
        or observations.get("context_reference_seedsequence_and_covariate_only_replay_executed")
        is not True
        or observations.get("context_reference_replay_count") != 2
        or observations.get("context_reference_artifact_count") != 0
        or observations.get("development_rng_authority_object_instantiated_in_memory") is not False
        or observations.get("development_seed") != 20_260_909
        or observations.get("development_seedsequence_created") is not False
        or observations.get("development_DGP_executed") is not False
        or observations.get("development_result_artifact_count") != 0
        or observations.get("selection_artifact_count") != 0
        or observations.get("canary_artifact_count") != 0
        or observations.get("development_v2_experimental_workflow_outcome_observation_count") != 0
        or observations.get("development_v2_governed_execution_network_access_count") != 0
        or retired_v2.get("scientific_interpretation")
        != "infrastructure_no_go_pre_development_not_a_scientific_pass_or_fail"
        or retired_v2.get("continuation_authorized") is not False
        or retired_v2.get("exact_directory_inventory_must_remain_one_file") is not True
        or recovery.get("attempt_id") != DEVELOPMENT_ATTEMPT_ID
        or recovery.get("development_bundle_schema") != DEVELOPMENT_BUNDLE_SCHEMA
        or recovery.get("canonical_root") != os.fspath(DEVELOPMENT_ROOT)
        or recovery.get("development_start_receipt_schema") != ARTIFACT_SCHEMAS["development_start"]
        or recovery.get("development_start_receipt_path")
        != os.fspath(DEVELOPMENT_START_CANONICAL_PATH)
        or recovery.get("development_start_receipt_policy")
        != "O_EXCL_before_development_authority_and_first_RNG_draw_creator_process_only"
        or recovery.get("receipt_without_result_restart_policy")
        != "terminal_DEVELOPMENT_ATTEMPT_CONSUMED_no_resume"
        or recovery.get("receipt_with_result_restart_policy") != "read_only_reopen_and_audit"
        or recovery.get("development_result_must_bind_start_receipt")
        != ["schema", "payload_sha256", "file_sha256"]
        or recovery.get("only_authorized_development_attempt") is not True
        or recovery.get("requires_exact_retired_v1_two_file_inventory") is not True
        or recovery.get("requires_exact_retired_v2_one_file_inventory") is not True
        or recovery.get("context_reference_policy")
        != "copy_and_revalidate_exact_development_v1_context_bytes_without_refit"
        or recovery.get("context_reference_transport_policy")
        != (
            "deep_detach_parsed_frozen_JSON_to_ordinary_dict_list_before_core_boundary_"
            "while_preserving_exact_source_bytes"
        )
        or recovery.get("context_reference_authority_policy")
        != "issue_reference_only_authority_for_status_and_context_publication"
        or recovery.get("development_authority_policy")
        != "issue_development_only_authority_just_in_time_during_DEVELOPMENT_PENDING_resume"
        or recovery.get("scientific_lockbox_authorized") is not False
        or recovery.get("human_EEG_outcome_authorized") is not False
        or recovery.get("network_access_authorized") is not False
        or immutability
        != {
            "development_v1_files_may_not_be_modified_deleted_or_extended": True,
            "development_v2_files_may_not_be_modified_deleted_or_extended": True,
            "development_v1_or_v2_schema_may_not_be_reinterpreted_as_v3": True,
            "any_failure_after_development_start_receipt_consumes_attempt": True,
            "original_preoutcome_amendment_may_not_be_modified": True,
            "prior_V3_1_recovery_amendment_may_not_be_modified": True,
            "future_target_bound_RSA_PSS_authorization_remains_separate_and_unissued": True,
            "operator_DGP_grid_threshold_gate_rank_seed_or_evidence_change_requires_new_scientific_revision": True,
        }
    ):
        raise ValidationError("recovery amendment differs from the frozen V3.2 facts")
    return MappingProxyType(dict(value))


def _scientific_contract_projection_bytes(
    *,
    master_bytes: bytes,
    synthetic_bytes: bytes,
    amendment: Mapping[str, Any],
) -> bytes:
    """Derive the exact outcome-relevant projection recorded by the amendment."""

    try:
        import yaml

        master = yaml.safe_load(master_bytes.decode("utf-8"))
        synthetic = yaml.safe_load(synthetic_bytes.decode("utf-8"))
    except (UnicodeDecodeError, ValueError, yaml.YAMLError) as exc:
        raise ValidationError("scientific plans are not valid UTF-8 YAML") from exc
    projection = amendment["scientific_contract_projection"]
    master_keys = projection.get("master_keys")
    synthetic_keys = projection.get("synthetic_keys")
    if (
        not isinstance(master, Mapping)
        or not isinstance(synthetic, Mapping)
        or not isinstance(master_keys, list)
        or not master_keys
        or any(not isinstance(key, str) for key in master_keys)
        or len(set(master_keys)) != len(master_keys)
        or not isinstance(synthetic_keys, list)
        or not synthetic_keys
        or any(not isinstance(key, str) for key in synthetic_keys)
        or len(set(synthetic_keys)) != len(synthetic_keys)
        or any(key not in master for key in master_keys)
        or any(key not in synthetic for key in synthetic_keys)
    ):
        raise ValidationError("scientific projection key inventory is invalid")
    preimage = {
        "schema": projection["schema"],
        "candidate_id": CANDIDATE_ID,
        "master": {key: master[key] for key in master_keys},
        "synthetic": {key: synthetic[key] for key in synthetic_keys},
    }
    try:
        return json.dumps(
            preimage,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise ValidationError("scientific projection is not canonical-JSON compatible") from exc


def _validate_scientific_contract_projection(
    frozen_file_bytes: Mapping[str, bytes],
) -> None:
    """Prove V3.2 changed governance only, against old V3 and current plans."""

    amendment = _validate_recovery_amendment_bytes(frozen_file_bytes[PREOUTCOME_AMENDMENT_PATH])
    current = _scientific_contract_projection_bytes(
        master_bytes=frozen_file_bytes[MASTER_PLAN_PATH],
        synthetic_bytes=frozen_file_bytes[SYNTHETIC_PLAN_PATH],
        amendment=amendment,
    )
    retired_v1 = _scientific_contract_projection_bytes(
        master_bytes=_run_git(
            V3_SOURCE_REPOSITORY,
            ("show", f"{RETIRED_V1_SOURCE_COMMIT}:{MASTER_PLAN_PATH}"),
        ),
        synthetic_bytes=_run_git(
            V3_SOURCE_REPOSITORY,
            ("show", f"{RETIRED_V1_SOURCE_COMMIT}:{SYNTHETIC_PLAN_PATH}"),
        ),
        amendment=amendment,
    )
    retired_v2 = _scientific_contract_projection_bytes(
        master_bytes=_run_git(
            V3_SOURCE_REPOSITORY,
            ("show", f"{RETIRED_V2_SOURCE_COMMIT}:{MASTER_PLAN_PATH}"),
        ),
        synthetic_bytes=_run_git(
            V3_SOURCE_REPOSITORY,
            ("show", f"{RETIRED_V2_SOURCE_COMMIT}:{SYNTHETIC_PLAN_PATH}"),
        ),
        amendment=amendment,
    )
    expected = SCIENTIFIC_CONTRACT_PROJECTION_SHA256
    if (
        current != retired_v1
        or current != retired_v2
        or file_sha256(current) != expected
        or amendment["scientific_contract_projection"]["sha256"] != expected
    ):
        raise AuthorityError("V3.2 scientific-contract projection differs from retired V1 or V2")


def _require_frozen_files_in_current_repository(
    snapshot: CleanSourceSnapshot,
    frozen_file_bytes: Mapping[str, bytes],
) -> None:
    """Byte-compare frozen inputs with canonical Git objects before authority use."""

    _require_frozen_files_in_source_snapshot(snapshot, frozen_file_bytes)
    before = capture_clean_source_snapshot(V3_SOURCE_REPOSITORY)
    if not _same_source_snapshot(before, snapshot):
        raise AuthorityError("canonical repository differs from the frozen source snapshot")
    for path in FROZEN_FILE_SHA256:
        committed = _run_git(
            V3_SOURCE_REPOSITORY,
            ("show", f"{snapshot.identity.commit}:{path}"),
        )
        if committed != frozen_file_bytes[path]:
            raise AuthorityError(f"caller frozen bytes differ from committed blob: {path}")
    _validate_scientific_contract_projection(frozen_file_bytes)
    after = capture_clean_source_snapshot(V3_SOURCE_REPOSITORY)
    if not _same_source_snapshot(after, snapshot):
        raise AuthorityError("canonical repository changed during frozen-file validation")


_RETIRED_V1_DEVELOPMENT_BUNDLE_FIELDS = frozenset(
    {
        "schema",
        "candidate_id",
        "created_at_UTC",
        "canonical_path",
        "master_plan_path",
        "master_plan_file_sha256",
        "synthetic_plan_path",
        "synthetic_plan_file_sha256",
        "preoutcome_amendment_path",
        "preoutcome_amendment_file_sha256",
        "clean_git_commit",
        "clean_git_tree",
        "tracked_source_file_inventory",
        "source_bundle_sha256",
        "numerical_runtime_inventory",
        "numerical_runtime_fingerprint_sha256",
        "artifact_schemas",
        "artifact_schema_inventory_sha256",
        "owner_authority_public_key_path",
        "owner_authority_public_key_file_sha256",
        "owner_authority_ssh_fingerprint",
        "v2_terminal_audit_path",
        "v2_terminal_audit_file_sha256",
        "v2_deny_overlay_path",
        "v2_deny_overlay_file_sha256",
        "focused_tests_passed",
        "focused_test_argv",
        "focused_test_process_id",
        "focused_test_collected_tests",
        "focused_test_passed_tests",
        "focused_test_failed_tests",
        "focused_test_skipped_tests",
        "focused_test_xfailed_tests",
        "focused_test_xpassed_tests",
        "focused_test_error_tests",
        "focused_test_executable_file_sha256",
        "focused_test_stdout_sha256",
        "focused_test_stderr_sha256",
        "focused_test_junit_report_file_sha256",
        "focused_test_junit_report_path",
        "focused_test_junit_report_base64",
        "focused_test_working_directory",
        "focused_test_environment_sha256",
        "focused_test_scope",
        "scientific_lockbox_authorized",
        "human_EEG_outcome_authorized",
        "payload_sha256",
    }
)
_RETIRED_V2_DEVELOPMENT_BUNDLE_FIELDS = _RETIRED_V1_DEVELOPMENT_BUNDLE_FIELDS | frozenset(
    {
        "protocol_revision",
        "development_attempt_id",
        "original_preoutcome_amendment_path",
        "original_preoutcome_amendment_file_sha256",
        "retired_v1_source_commit",
        "retired_v1_source_tree",
        "retired_v1_artifact_inventory_schema",
        "retired_v1_artifact_inventory_sha256",
        "retired_v1_development_bundle_path",
        "retired_v1_development_bundle_schema",
        "retired_v1_development_bundle_payload_sha256",
        "retired_v1_development_bundle_file_sha256",
        "retired_v1_context_reference_path",
        "retired_v1_context_reference_schema",
        "retired_v1_context_reference_payload_sha256",
        "retired_v1_context_reference_file_sha256",
        "retired_v1_incident_error",
        "retired_v1_incident_message",
        "retired_v1_incident_output_sha256",
        "retired_v1_context_reference_root_seed",
        "retired_v1_development_root_seed",
        "retired_v1_development_seedsequence_created",
        "retired_v1_development_DGP_executed",
        "retired_v1_development_result_present",
        "retired_v1_experimental_outcome_observed",
        "retired_v1_governed_network_accessed",
        "retired_v1_continuation_authorized",
        "scientific_contract_projection_sha256",
    }
)
_DEVELOPMENT_BUNDLE_FIELDS = _RETIRED_V2_DEVELOPMENT_BUNDLE_FIELDS | frozenset(
    {
        "prior_recovery_amendment_path",
        "prior_recovery_amendment_file_sha256",
        "retired_v2_source_commit",
        "retired_v2_source_tree",
        "retired_v2_source_bundle_sha256",
        "retired_v2_artifact_inventory_schema",
        "retired_v2_artifact_inventory_sha256",
        "retired_v2_development_bundle_path",
        "retired_v2_development_bundle_schema",
        "retired_v2_development_bundle_payload_sha256",
        "retired_v2_development_bundle_file_sha256",
        "retired_v2_incident_error",
        "retired_v2_incident_message",
        "retired_v2_incident_output_sha256",
        "retired_v2_context_reference_root_seed",
        "retired_v2_context_reference_replay_count",
        "retired_v2_context_reference_present",
        "retired_v2_development_root_seed",
        "retired_v2_development_rng_authority_issued",
        "retired_v2_development_seedsequence_created",
        "retired_v2_development_DGP_executed",
        "retired_v2_development_result_present",
        "retired_v2_experimental_outcome_observed",
        "retired_v2_governed_network_accessed",
        "retired_v2_continuation_authorized",
        "development_start_receipt_schema",
        "development_start_receipt_path",
        "development_start_receipt_policy",
    }
)
DEVELOPMENT_BUNDLE_CANONICAL_PATH = (
    "/home/whwovy/v3-artifacts/metadata-calibration-efficiency-v3/"
    "development-v3/development-bundle.json"
)


def _retired_v1_inventory_preimage() -> Mapping[str, Any]:
    return {
        "schema": RETIRED_V1_ARTIFACT_INVENTORY_SCHEMA,
        "attempt": "development-v1",
        "artifacts": [
            {
                "name": "development_bundle",
                "path": os.fspath(RETIRED_V1_DEVELOPMENT_BUNDLE_PATH),
                "schema": RETIRED_V1_DEVELOPMENT_BUNDLE_SCHEMA,
                "payload_sha256": RETIRED_V1_BUNDLE_PAYLOAD_SHA256,
                "file_sha256": RETIRED_V1_BUNDLE_FILE_SHA256,
            },
            {
                "name": "context_reference",
                "path": os.fspath(RETIRED_V1_CONTEXT_REFERENCE_PATH),
                "schema": ARTIFACT_SCHEMAS["context_reference"],
                "payload_sha256": RETIRED_V1_CONTEXT_PAYLOAD_SHA256,
                "file_sha256": RETIRED_V1_CONTEXT_FILE_SHA256,
            },
        ],
    }


def retired_v1_artifact_inventory_sha256() -> str:
    return hashlib.sha256(canonical_json_bytes(_retired_v1_inventory_preimage())).hexdigest()


def _retired_v2_inventory_preimage() -> Mapping[str, Any]:
    return {
        "schema": RETIRED_V2_ARTIFACT_INVENTORY_SCHEMA,
        "attempt": "development-v2",
        "artifacts": [
            {
                "name": "development_bundle",
                "path": os.fspath(RETIRED_V2_DEVELOPMENT_BUNDLE_PATH),
                "schema": RETIRED_V2_DEVELOPMENT_BUNDLE_SCHEMA,
                "payload_sha256": RETIRED_V2_BUNDLE_PAYLOAD_SHA256,
                "file_sha256": RETIRED_V2_BUNDLE_FILE_SHA256,
            }
        ],
    }


def retired_v2_artifact_inventory_sha256() -> str:
    return hashlib.sha256(canonical_json_bytes(_retired_v2_inventory_preimage())).hexdigest()


def _retired_v1_stat_identity(value: os.stat_result) -> tuple[int, ...]:
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


def _load_exact_retired_v1_directory_bytes(root: Path) -> Mapping[str, bytes]:
    """Read one exact two-file retirement inventory without following links."""

    root = _canonical_absolute_path(root)
    descriptor = _open_absolute_directory(root, create=False, require_private_final=True)
    expected_names = frozenset({"development-bundle.json", "context-reference.json"})
    loaded: dict[str, bytes] = {}
    try:
        before = os.fstat(descriptor)
        try:
            names = frozenset(os.listdir(descriptor))
        except OSError as exc:
            raise AuthorityError("retired development-v1 inventory cannot be enumerated") from exc
        if names != expected_names:
            raise AuthorityError("retired development-v1 must remain the exact two-file inventory")
        for name in sorted(expected_names):
            try:
                child = os.open(name, os.O_RDONLY | os.O_NOFOLLOW, dir_fd=descriptor)
            except OSError as exc:
                raise AuthorityError(
                    "retired development-v1 files must be owner-controlled mode-0400 regular files"
                ) from exc
            try:
                observed = os.fstat(child)
                if (
                    not stat.S_ISREG(observed.st_mode)
                    or observed.st_uid != os.geteuid()
                    or observed.st_nlink != 1
                    or stat.S_IMODE(observed.st_mode) != 0o400
                ):
                    raise AuthorityError(
                        "retired development-v1 files must be owner-controlled mode-0400 regular files"
                    )
                loaded[name] = _read_all(child, maximum_bytes=64 * 1024 * 1024)
                after_file = os.fstat(child)
                if _retired_v1_stat_identity(observed) != _retired_v1_stat_identity(after_file):
                    raise AuthorityError("retired development-v1 file changed while read")
            finally:
                os.close(child)
        after = os.fstat(descriptor)
        if (
            _retired_v1_stat_identity(before) != _retired_v1_stat_identity(after)
            or frozenset(os.listdir(descriptor)) != expected_names
        ):
            raise AuthorityError("retired development-v1 directory changed while observed")
    finally:
        os.close(descriptor)
    return MappingProxyType(loaded)


def _load_exact_retired_v2_directory_bytes(root: Path) -> Mapping[str, bytes]:
    """Read the exact bundle-only V2 retirement inventory without following links."""

    root = _canonical_absolute_path(root)
    descriptor = _open_absolute_directory(root, create=False, require_private_final=True)
    expected_names = frozenset({"development-bundle.json"})
    loaded: dict[str, bytes] = {}
    try:
        before = os.fstat(descriptor)
        try:
            names = frozenset(os.listdir(descriptor))
        except OSError as exc:
            raise AuthorityError("retired development-v2 inventory cannot be enumerated") from exc
        if names != expected_names:
            raise AuthorityError("retired development-v2 must remain the exact one-file inventory")
        try:
            child = os.open(
                "development-bundle.json",
                os.O_RDONLY | os.O_NOFOLLOW,
                dir_fd=descriptor,
            )
        except OSError as exc:
            raise AuthorityError(
                "retired development-v2 bundle must be an owner-controlled mode-0400 regular file"
            ) from exc
        try:
            observed = os.fstat(child)
            if (
                not stat.S_ISREG(observed.st_mode)
                or observed.st_uid != os.geteuid()
                or observed.st_nlink != 1
                or stat.S_IMODE(observed.st_mode) != 0o400
            ):
                raise AuthorityError(
                    "retired development-v2 bundle must be an owner-controlled mode-0400 regular file"
                )
            loaded["development-bundle.json"] = _read_all(
                child,
                maximum_bytes=64 * 1024 * 1024,
            )
            after_file = os.fstat(child)
            if _retired_v1_stat_identity(observed) != _retired_v1_stat_identity(after_file):
                raise AuthorityError("retired development-v2 bundle changed while read")
        finally:
            os.close(child)
        after = os.fstat(descriptor)
        if (
            _retired_v1_stat_identity(before) != _retired_v1_stat_identity(after)
            or frozenset(os.listdir(descriptor)) != expected_names
        ):
            raise AuthorityError("retired development-v2 directory changed while observed")
    finally:
        os.close(descriptor)
    return MappingProxyType(loaded)


@dataclass(frozen=True, slots=True, init=False)
class RetiredV1ArtifactsCapability:
    """Read-only proof that the failed V1 attempt remains the exact two-file prefix."""

    schema: str
    candidate_id: str
    attempt_id: str
    source_commit: str
    source_tree: str
    artifact_inventory_sha256: str
    development_bundle_schema: str
    development_bundle_payload_sha256: str
    development_bundle_file_sha256: str
    context_reference_schema: str
    context_reference_payload_sha256: str
    context_reference_file_sha256: str
    incident_error: str
    incident_output_sha256: str
    continuation_authorized: bool
    binding_sha256: str
    _development_bundle_bytes: bytes = field(repr=False, compare=False)
    _context_reference_bytes: bytes = field(repr=False, compare=False)
    _validation_marker: object = field(repr=False, compare=False)

    def __init__(
        self,
        *,
        development_bundle_bytes: bytes,
        context_reference_bytes: bytes,
        _issuer: object,
    ) -> None:
        if _issuer is not _CAPABILITY_ISSUER:
            raise AuthorityError("RetiredV1ArtifactsCapability has no valid issuer")
        bindings = {
            "schema": RETIRED_V1_ARTIFACT_INVENTORY_SCHEMA,
            "candidate_id": CANDIDATE_ID,
            "attempt_id": "development-v1",
            "source_commit": RETIRED_V1_SOURCE_COMMIT,
            "source_tree": RETIRED_V1_SOURCE_TREE,
            "artifact_inventory_sha256": RETIRED_V1_ARTIFACT_INVENTORY_SHA256,
            "development_bundle_schema": RETIRED_V1_DEVELOPMENT_BUNDLE_SCHEMA,
            "development_bundle_payload_sha256": RETIRED_V1_BUNDLE_PAYLOAD_SHA256,
            "development_bundle_file_sha256": RETIRED_V1_BUNDLE_FILE_SHA256,
            "context_reference_schema": ARTIFACT_SCHEMAS["context_reference"],
            "context_reference_payload_sha256": RETIRED_V1_CONTEXT_PAYLOAD_SHA256,
            "context_reference_file_sha256": RETIRED_V1_CONTEXT_FILE_SHA256,
            "incident_error": RETIRED_V1_INCIDENT_ERROR,
            "incident_output_sha256": RETIRED_V1_INCIDENT_OUTPUT_SHA256,
            "continuation_authorized": False,
        }
        for name, value in bindings.items():
            object.__setattr__(self, name, value)
        object.__setattr__(
            self,
            "binding_sha256",
            hashlib.sha256(canonical_json_bytes(bindings)).hexdigest(),
        )
        object.__setattr__(self, "_development_bundle_bytes", bytes(development_bundle_bytes))
        object.__setattr__(self, "_context_reference_bytes", bytes(context_reference_bytes))
        object.__setattr__(self, "_validation_marker", _issuer)


def require_retired_v1_artifacts_capability(
    value: object,
) -> RetiredV1ArtifactsCapability:
    if type(value) is not RetiredV1ArtifactsCapability:
        raise AuthorityError("exact RetiredV1ArtifactsCapability required")
    capability = value
    if getattr(capability, "_validation_marker", None) is not _CAPABILITY_ISSUER:
        raise AuthorityError("retired development-v1 capability marker is invalid")
    bindings = {
        "schema": capability.schema,
        "candidate_id": capability.candidate_id,
        "attempt_id": capability.attempt_id,
        "source_commit": capability.source_commit,
        "source_tree": capability.source_tree,
        "artifact_inventory_sha256": capability.artifact_inventory_sha256,
        "development_bundle_schema": capability.development_bundle_schema,
        "development_bundle_payload_sha256": capability.development_bundle_payload_sha256,
        "development_bundle_file_sha256": capability.development_bundle_file_sha256,
        "context_reference_schema": capability.context_reference_schema,
        "context_reference_payload_sha256": capability.context_reference_payload_sha256,
        "context_reference_file_sha256": capability.context_reference_file_sha256,
        "incident_error": capability.incident_error,
        "incident_output_sha256": capability.incident_output_sha256,
        "continuation_authorized": capability.continuation_authorized,
    }
    expected = {
        "schema": RETIRED_V1_ARTIFACT_INVENTORY_SCHEMA,
        "candidate_id": CANDIDATE_ID,
        "attempt_id": "development-v1",
        "source_commit": RETIRED_V1_SOURCE_COMMIT,
        "source_tree": RETIRED_V1_SOURCE_TREE,
        "artifact_inventory_sha256": RETIRED_V1_ARTIFACT_INVENTORY_SHA256,
        "development_bundle_schema": RETIRED_V1_DEVELOPMENT_BUNDLE_SCHEMA,
        "development_bundle_payload_sha256": RETIRED_V1_BUNDLE_PAYLOAD_SHA256,
        "development_bundle_file_sha256": RETIRED_V1_BUNDLE_FILE_SHA256,
        "context_reference_schema": ARTIFACT_SCHEMAS["context_reference"],
        "context_reference_payload_sha256": RETIRED_V1_CONTEXT_PAYLOAD_SHA256,
        "context_reference_file_sha256": RETIRED_V1_CONTEXT_FILE_SHA256,
        "incident_error": RETIRED_V1_INCIDENT_ERROR,
        "incident_output_sha256": RETIRED_V1_INCIDENT_OUTPUT_SHA256,
        "continuation_authorized": False,
    }
    if (
        bindings != expected
        or capability.binding_sha256 != hashlib.sha256(canonical_json_bytes(expected)).hexdigest()
        or retired_v1_artifact_inventory_sha256() != RETIRED_V1_ARTIFACT_INVENTORY_SHA256
        or file_sha256(capability._development_bundle_bytes) != RETIRED_V1_BUNDLE_FILE_SHA256
        or file_sha256(capability._context_reference_bytes) != RETIRED_V1_CONTEXT_FILE_SHA256
    ):
        raise AuthorityError("retired development-v1 capability binding is invalid")
    return capability


def _observe_retired_v1_artifacts_from_exact_root(root: Path) -> RetiredV1ArtifactsCapability:
    loaded = _load_exact_retired_v1_directory_bytes(root)
    bundle_bytes = loaded["development-bundle.json"]
    context_bytes = loaded["context-reference.json"]
    bundle = parse_artifact_bytes(
        bundle_bytes,
        ArtifactSpec(
            schema=RETIRED_V1_DEVELOPMENT_BUNDLE_SCHEMA,
            exact_fields=_RETIRED_V1_DEVELOPMENT_BUNDLE_FIELDS,
        ),
    )
    context = parse_artifact_bytes(
        context_bytes,
        ArtifactSpec(
            schema=ARTIFACT_SCHEMAS["context_reference"],
            exact_fields=_CONTEXT_REFERENCE_FIELDS,
        ),
    )
    if (
        root != RETIRED_V1_DEVELOPMENT_ROOT
        or bundle["candidate_id"] != CANDIDATE_ID
        or bundle["canonical_path"] != os.fspath(RETIRED_V1_DEVELOPMENT_BUNDLE_PATH)
        or bundle["clean_git_commit"] != RETIRED_V1_SOURCE_COMMIT
        or bundle["clean_git_tree"] != RETIRED_V1_SOURCE_TREE
        or bundle["payload_sha256"] != RETIRED_V1_BUNDLE_PAYLOAD_SHA256
        or file_sha256(bundle_bytes) != RETIRED_V1_BUNDLE_FILE_SHA256
        or context["candidate_id"] != CANDIDATE_ID
        or context["payload_sha256"] != RETIRED_V1_CONTEXT_PAYLOAD_SHA256
        or file_sha256(context_bytes) != RETIRED_V1_CONTEXT_FILE_SHA256
    ):
        raise AuthorityError("retired development-v1 artifact identity mismatch")
    return RetiredV1ArtifactsCapability(
        development_bundle_bytes=bundle_bytes,
        context_reference_bytes=context_bytes,
        _issuer=_CAPABILITY_ISSUER,
    )


def observe_retired_v1_artifacts() -> RetiredV1ArtifactsCapability:
    """Observe the immutable failed V1 prefix; this grants no RNG or mutation authority."""

    _require_active_governed_process(
        recheck_site_packages=True,
        recheck_source=True,
    )
    return _observe_retired_v1_artifacts_from_exact_root(RETIRED_V1_DEVELOPMENT_ROOT)


def retired_v1_context_reference_bytes(value: object) -> bytes:
    """Return exact retired context bytes after re-observing the fixed two-file inventory."""

    capability = require_retired_v1_artifacts_capability(value)
    observed = observe_retired_v1_artifacts()
    if observed.binding_sha256 != capability.binding_sha256:
        raise AuthorityError("retired development-v1 inventory changed after observation")
    return bytes(observed._context_reference_bytes)


def retired_v1_context_reference_payload(value: object) -> Mapping[str, Any]:
    """Parse the exact retired reference bytes without fitting or selecting a new reference."""

    data = retired_v1_context_reference_bytes(value)
    return parse_artifact_bytes(
        data,
        ArtifactSpec(
            schema=ARTIFACT_SCHEMAS["context_reference"],
            exact_fields=_CONTEXT_REFERENCE_FIELDS,
        ),
    )


@dataclass(frozen=True, slots=True, init=False)
class RetiredV2ArtifactsCapability:
    """Read-only proof that failed V2 remains its exact bundle-only prefix."""

    schema: str
    candidate_id: str
    attempt_id: str
    source_commit: str
    source_tree: str
    source_bundle_sha256: str
    artifact_inventory_sha256: str
    development_bundle_schema: str
    development_bundle_payload_sha256: str
    development_bundle_file_sha256: str
    incident_error: str
    incident_output_sha256: str
    context_reference_replay_count: int
    context_reference_present: bool
    development_seedsequence_created: bool
    development_DGP_executed: bool
    continuation_authorized: bool
    binding_sha256: str
    _development_bundle_bytes: bytes = field(repr=False, compare=False)
    _validation_marker: object = field(repr=False, compare=False)

    def __init__(self, *, development_bundle_bytes: bytes, _issuer: object) -> None:
        if _issuer is not _CAPABILITY_ISSUER:
            raise AuthorityError("RetiredV2ArtifactsCapability has no valid issuer")
        bindings = _retired_v2_capability_bindings()
        for name, item in bindings.items():
            object.__setattr__(self, name, item)
        object.__setattr__(
            self,
            "binding_sha256",
            hashlib.sha256(canonical_json_bytes(bindings)).hexdigest(),
        )
        object.__setattr__(self, "_development_bundle_bytes", bytes(development_bundle_bytes))
        object.__setattr__(self, "_validation_marker", _issuer)


def _retired_v2_capability_bindings() -> Mapping[str, Any]:
    return {
        "schema": RETIRED_V2_ARTIFACT_INVENTORY_SCHEMA,
        "candidate_id": CANDIDATE_ID,
        "attempt_id": "development-v2",
        "source_commit": RETIRED_V2_SOURCE_COMMIT,
        "source_tree": RETIRED_V2_SOURCE_TREE,
        "source_bundle_sha256": RETIRED_V2_SOURCE_BUNDLE_SHA256,
        "artifact_inventory_sha256": RETIRED_V2_ARTIFACT_INVENTORY_SHA256,
        "development_bundle_schema": RETIRED_V2_DEVELOPMENT_BUNDLE_SCHEMA,
        "development_bundle_payload_sha256": RETIRED_V2_BUNDLE_PAYLOAD_SHA256,
        "development_bundle_file_sha256": RETIRED_V2_BUNDLE_FILE_SHA256,
        "incident_error": RETIRED_V2_INCIDENT_ERROR,
        "incident_output_sha256": RETIRED_V2_INCIDENT_OUTPUT_SHA256,
        "context_reference_replay_count": 2,
        "context_reference_present": False,
        "development_seedsequence_created": False,
        "development_DGP_executed": False,
        "continuation_authorized": False,
    }


def require_retired_v2_artifacts_capability(
    value: object,
) -> RetiredV2ArtifactsCapability:
    if type(value) is not RetiredV2ArtifactsCapability:
        raise AuthorityError("exact RetiredV2ArtifactsCapability required")
    capability = value
    if getattr(capability, "_validation_marker", None) is not _CAPABILITY_ISSUER:
        raise AuthorityError("retired development-v2 capability marker is invalid")
    expected = _retired_v2_capability_bindings()
    observed = {name: getattr(capability, name) for name in expected}
    if (
        observed != expected
        or capability.binding_sha256 != hashlib.sha256(canonical_json_bytes(expected)).hexdigest()
        or retired_v2_artifact_inventory_sha256() != RETIRED_V2_ARTIFACT_INVENTORY_SHA256
        or file_sha256(capability._development_bundle_bytes) != RETIRED_V2_BUNDLE_FILE_SHA256
    ):
        raise AuthorityError("retired development-v2 capability binding is invalid")
    return capability


def _observe_retired_v2_artifacts_from_exact_root(root: Path) -> RetiredV2ArtifactsCapability:
    loaded = _load_exact_retired_v2_directory_bytes(root)
    bundle_bytes = loaded["development-bundle.json"]
    bundle = parse_artifact_bytes(
        bundle_bytes,
        ArtifactSpec(
            schema=RETIRED_V2_DEVELOPMENT_BUNDLE_SCHEMA,
            exact_fields=_RETIRED_V2_DEVELOPMENT_BUNDLE_FIELDS,
        ),
    )
    if (
        root != RETIRED_V2_DEVELOPMENT_ROOT
        or bundle["candidate_id"] != CANDIDATE_ID
        or bundle["canonical_path"] != os.fspath(RETIRED_V2_DEVELOPMENT_BUNDLE_PATH)
        or bundle["clean_git_commit"] != RETIRED_V2_SOURCE_COMMIT
        or bundle["clean_git_tree"] != RETIRED_V2_SOURCE_TREE
        or bundle["source_bundle_sha256"] != RETIRED_V2_SOURCE_BUNDLE_SHA256
        or bundle["protocol_revision"] != "V3.1"
        or bundle["development_attempt_id"] != "development-v2"
        or bundle["preoutcome_amendment_path"] != PRIOR_RECOVERY_AMENDMENT_PATH
        or bundle["preoutcome_amendment_file_sha256"]
        != FROZEN_FILE_SHA256[PRIOR_RECOVERY_AMENDMENT_PATH]
        or bundle["payload_sha256"] != RETIRED_V2_BUNDLE_PAYLOAD_SHA256
        or file_sha256(bundle_bytes) != RETIRED_V2_BUNDLE_FILE_SHA256
    ):
        raise AuthorityError("retired development-v2 artifact identity mismatch")
    return RetiredV2ArtifactsCapability(
        development_bundle_bytes=bundle_bytes,
        _issuer=_CAPABILITY_ISSUER,
    )


def observe_retired_v2_artifacts() -> RetiredV2ArtifactsCapability:
    """Observe immutable failed V2; this grants no RNG or mutation authority."""

    _require_active_governed_process(
        recheck_site_packages=True,
        recheck_source=True,
    )
    return _observe_retired_v2_artifacts_from_exact_root(RETIRED_V2_DEVELOPMENT_ROOT)


def observe_retired_development_attempts() -> tuple[
    RetiredV1ArtifactsCapability,
    RetiredV2ArtifactsCapability,
]:
    """Re-observe both immutable failed prefixes before any V3 authority use."""

    retired_v1 = observe_retired_v1_artifacts()
    retired_v2 = observe_retired_v2_artifacts()
    return retired_v1, retired_v2


def _retired_attempt_bundle_fields() -> Mapping[str, Any]:
    return {
        "protocol_revision": PROTOCOL_REVISION,
        "development_attempt_id": DEVELOPMENT_ATTEMPT_ID,
        "original_preoutcome_amendment_path": ORIGINAL_PREOUTCOME_AMENDMENT_PATH,
        "original_preoutcome_amendment_file_sha256": FROZEN_FILE_SHA256[
            ORIGINAL_PREOUTCOME_AMENDMENT_PATH
        ],
        "prior_recovery_amendment_path": PRIOR_RECOVERY_AMENDMENT_PATH,
        "prior_recovery_amendment_file_sha256": FROZEN_FILE_SHA256[PRIOR_RECOVERY_AMENDMENT_PATH],
        "retired_v1_source_commit": RETIRED_V1_SOURCE_COMMIT,
        "retired_v1_source_tree": RETIRED_V1_SOURCE_TREE,
        "retired_v1_artifact_inventory_schema": RETIRED_V1_ARTIFACT_INVENTORY_SCHEMA,
        "retired_v1_artifact_inventory_sha256": RETIRED_V1_ARTIFACT_INVENTORY_SHA256,
        "retired_v1_development_bundle_path": os.fspath(RETIRED_V1_DEVELOPMENT_BUNDLE_PATH),
        "retired_v1_development_bundle_schema": RETIRED_V1_DEVELOPMENT_BUNDLE_SCHEMA,
        "retired_v1_development_bundle_payload_sha256": (RETIRED_V1_BUNDLE_PAYLOAD_SHA256),
        "retired_v1_development_bundle_file_sha256": RETIRED_V1_BUNDLE_FILE_SHA256,
        "retired_v1_context_reference_path": os.fspath(RETIRED_V1_CONTEXT_REFERENCE_PATH),
        "retired_v1_context_reference_schema": ARTIFACT_SCHEMAS["context_reference"],
        "retired_v1_context_reference_payload_sha256": (RETIRED_V1_CONTEXT_PAYLOAD_SHA256),
        "retired_v1_context_reference_file_sha256": RETIRED_V1_CONTEXT_FILE_SHA256,
        "retired_v1_incident_error": RETIRED_V1_INCIDENT_ERROR,
        "retired_v1_incident_message": RETIRED_V1_INCIDENT_MESSAGE,
        "retired_v1_incident_output_sha256": RETIRED_V1_INCIDENT_OUTPUT_SHA256,
        "retired_v1_context_reference_root_seed": 20_260_910,
        "retired_v1_development_root_seed": 20_260_909,
        "retired_v1_development_seedsequence_created": False,
        "retired_v1_development_DGP_executed": False,
        "retired_v1_development_result_present": False,
        "retired_v1_experimental_outcome_observed": False,
        "retired_v1_governed_network_accessed": False,
        "retired_v1_continuation_authorized": False,
        "retired_v2_source_commit": RETIRED_V2_SOURCE_COMMIT,
        "retired_v2_source_tree": RETIRED_V2_SOURCE_TREE,
        "retired_v2_source_bundle_sha256": RETIRED_V2_SOURCE_BUNDLE_SHA256,
        "retired_v2_artifact_inventory_schema": RETIRED_V2_ARTIFACT_INVENTORY_SCHEMA,
        "retired_v2_artifact_inventory_sha256": RETIRED_V2_ARTIFACT_INVENTORY_SHA256,
        "retired_v2_development_bundle_path": os.fspath(RETIRED_V2_DEVELOPMENT_BUNDLE_PATH),
        "retired_v2_development_bundle_schema": RETIRED_V2_DEVELOPMENT_BUNDLE_SCHEMA,
        "retired_v2_development_bundle_payload_sha256": (RETIRED_V2_BUNDLE_PAYLOAD_SHA256),
        "retired_v2_development_bundle_file_sha256": RETIRED_V2_BUNDLE_FILE_SHA256,
        "retired_v2_incident_error": RETIRED_V2_INCIDENT_ERROR,
        "retired_v2_incident_message": RETIRED_V2_INCIDENT_MESSAGE,
        "retired_v2_incident_output_sha256": RETIRED_V2_INCIDENT_OUTPUT_SHA256,
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
        "scientific_contract_projection_sha256": SCIENTIFIC_CONTRACT_PROJECTION_SHA256,
        "development_start_receipt_schema": ARTIFACT_SCHEMAS["development_start"],
        "development_start_receipt_path": os.fspath(DEVELOPMENT_START_CANONICAL_PATH),
        "development_start_receipt_policy": (
            "O_EXCL_before_development_authority_and_first_RNG_draw_creator_process_only"
        ),
    }


def _build_development_bundle(
    *,
    snapshot: CleanSourceSnapshot,
    frozen_file_bytes: Mapping[str, bytes],
    created_at_UTC: str,
    focused_test_run: ObservedTestRunCapability,
    expected_test_scope: str,
) -> Mapping[str, Any]:
    """Build, but do not publish, the exact V3 development-bundle payload."""

    snapshot = _require_clean_source_snapshot(snapshot)
    _parse_rfc3339_seconds(created_at_UTC, "development bundle created_at_UTC")
    runtime_inventory, runtime_fingerprint = _current_numerical_runtime()
    focused = _require_observed_test_run_capability(
        focused_test_run,
        expected_name="focused_v3",
        snapshot=snapshot,
        expected_scope=expected_test_scope,
        expected_environment_sha256=runtime_fingerprint,
        expected_python_site_inventory_sha256=_python_site_inventory_sha256(runtime_inventory),
    )
    _require_frozen_files_in_source_snapshot(snapshot, frozen_file_bytes)
    records = [entry.as_record() for entry in snapshot.tracked_files]
    bundle = {
        "schema": DEVELOPMENT_BUNDLE_SCHEMA,
        "candidate_id": CANDIDATE_ID,
        **dict(_retired_attempt_bundle_fields()),
        "created_at_UTC": created_at_UTC,
        "canonical_path": DEVELOPMENT_BUNDLE_CANONICAL_PATH,
        "master_plan_path": MASTER_PLAN_PATH,
        "master_plan_file_sha256": FROZEN_FILE_SHA256[MASTER_PLAN_PATH],
        "synthetic_plan_path": SYNTHETIC_PLAN_PATH,
        "synthetic_plan_file_sha256": FROZEN_FILE_SHA256[SYNTHETIC_PLAN_PATH],
        "preoutcome_amendment_path": PREOUTCOME_AMENDMENT_PATH,
        "preoutcome_amendment_file_sha256": FROZEN_FILE_SHA256[PREOUTCOME_AMENDMENT_PATH],
        "clean_git_commit": snapshot.identity.commit,
        "clean_git_tree": snapshot.identity.tree,
        "tracked_source_file_inventory": records,
        "source_bundle_sha256": snapshot.source_bundle_sha256,
        "numerical_runtime_inventory": runtime_inventory,
        "numerical_runtime_fingerprint_sha256": runtime_fingerprint,
        "artifact_schemas": dict(ARTIFACT_SCHEMAS),
        "artifact_schema_inventory_sha256": artifact_schema_inventory_sha256(),
        "owner_authority_public_key_path": OWNER_AUTHORITY_PUBLIC_KEY_PATH,
        "owner_authority_public_key_file_sha256": FROZEN_FILE_SHA256[
            OWNER_AUTHORITY_PUBLIC_KEY_PATH
        ],
        "owner_authority_ssh_fingerprint": OWNER_AUTHORITY_FINGERPRINT,
        "v2_terminal_audit_path": V2_TERMINAL_AUDIT_PATH,
        "v2_terminal_audit_file_sha256": FROZEN_FILE_SHA256[V2_TERMINAL_AUDIT_PATH],
        "v2_deny_overlay_path": V2_DENY_OVERLAY_PATH,
        "v2_deny_overlay_file_sha256": FROZEN_FILE_SHA256[V2_DENY_OVERLAY_PATH],
        "focused_tests_passed": True,
        "focused_test_argv": list(focused.argv),
        "focused_test_process_id": focused.process_id,
        "focused_test_collected_tests": focused.collected_tests,
        "focused_test_passed_tests": focused.passed_tests,
        "focused_test_failed_tests": focused.failed_tests,
        "focused_test_skipped_tests": focused.skipped_tests,
        "focused_test_xfailed_tests": focused.xfailed_tests,
        "focused_test_xpassed_tests": focused.xpassed_tests,
        "focused_test_error_tests": focused.error_tests,
        "focused_test_executable_file_sha256": focused.executable_file_sha256,
        "focused_test_stdout_sha256": focused.stdout_sha256,
        "focused_test_stderr_sha256": focused.stderr_sha256,
        "focused_test_junit_report_file_sha256": focused.junit_report_file_sha256,
        "focused_test_junit_report_path": focused.junit_report_path,
        "focused_test_junit_report_base64": base64.b64encode(focused.junit_report_bytes).decode(
            "ascii"
        ),
        "focused_test_working_directory": focused.working_directory,
        "focused_test_environment_sha256": focused.environment_sha256,
        "focused_test_scope": focused.scope,
        "scientific_lockbox_authorized": False,
        "human_EEG_outcome_authorized": False,
    }
    return seal_payload(bundle)


def build_development_bundle(
    *,
    snapshot: CleanSourceSnapshot,
    frozen_file_bytes: Mapping[str, bytes],
    created_at_UTC: str,
    focused_test_run: ObservedTestRunCapability,
) -> Mapping[str, Any]:
    """Build the production bundle from a canonical observed test receipt."""

    _require_active_governed_process(
        recheck_site_packages=True,
        recheck_source=True,
        require_mutation_role=True,
    )
    observe_retired_development_attempts()
    value = _build_development_bundle(
        snapshot=snapshot,
        frozen_file_bytes=frozen_file_bytes,
        created_at_UTC=created_at_UTC,
        focused_test_run=focused_test_run,
        expected_test_scope="canonical",
    )
    _require_frozen_files_in_current_repository(snapshot, frozen_file_bytes)
    return value


def _validate_development_bundle_bytes(
    value: Mapping[str, Any],
    *,
    file_bytes: bytes,
    snapshot: CleanSourceSnapshot,
    frozen_file_bytes: Mapping[str, bytes],
    focused_test_run: ObservedTestRunCapability,
    expected_test_scope: str,
) -> Mapping[str, Any]:
    """Validate exact bundle bytes without issuing execution authority."""

    snapshot = _require_clean_source_snapshot(snapshot)
    parsed = parse_artifact_bytes(
        file_bytes,
        ArtifactSpec(schema=DEVELOPMENT_BUNDLE_SCHEMA, exact_fields=_DEVELOPMENT_BUNDLE_FIELDS),
    )
    if canonical_json_bytes(parsed) != canonical_json_bytes(value):
        raise ValidationError("development-bundle value differs from supplied file bytes")
    expected = _build_development_bundle(
        snapshot=snapshot,
        frozen_file_bytes=frozen_file_bytes,
        created_at_UTC=str(parsed["created_at_UTC"]),
        focused_test_run=focused_test_run,
        expected_test_scope=expected_test_scope,
    )
    if canonical_json_bytes(parsed) != canonical_json_bytes(expected):
        raise ValidationError("development-bundle payload differs from exact reconstructed bundle")
    return parsed


def _validate_recorded_development_bundle(
    parsed: Mapping[str, Any],
    *,
    file_bytes: bytes,
) -> CleanSourceSnapshot:
    observe_retired_development_attempts()
    parsed = parse_artifact_bytes(
        file_bytes,
        ArtifactSpec(schema=DEVELOPMENT_BUNDLE_SCHEMA, exact_fields=_DEVELOPMENT_BUNDLE_FIELDS),
    )
    snapshot = _capture_committed_source_snapshot(
        V3_SOURCE_REPOSITORY,
        str(parsed["clean_git_commit"]),
    )
    expected_inventory = tuple(_freeze_json(item.as_record()) for item in snapshot.tracked_files)
    fixed = {
        "candidate_id": CANDIDATE_ID,
        **dict(_retired_attempt_bundle_fields()),
        "canonical_path": DEVELOPMENT_BUNDLE_CANONICAL_PATH,
        "master_plan_path": MASTER_PLAN_PATH,
        "master_plan_file_sha256": FROZEN_FILE_SHA256[MASTER_PLAN_PATH],
        "synthetic_plan_path": SYNTHETIC_PLAN_PATH,
        "synthetic_plan_file_sha256": FROZEN_FILE_SHA256[SYNTHETIC_PLAN_PATH],
        "preoutcome_amendment_path": PREOUTCOME_AMENDMENT_PATH,
        "preoutcome_amendment_file_sha256": FROZEN_FILE_SHA256[PREOUTCOME_AMENDMENT_PATH],
        "clean_git_tree": snapshot.identity.tree,
        "source_bundle_sha256": snapshot.source_bundle_sha256,
        "numerical_runtime_fingerprint_sha256": numerical_runtime_fingerprint_sha256(
            parsed["numerical_runtime_inventory"]
        ),
        "artifact_schemas": dict(ARTIFACT_SCHEMAS),
        "artifact_schema_inventory_sha256": artifact_schema_inventory_sha256(),
        "owner_authority_public_key_path": OWNER_AUTHORITY_PUBLIC_KEY_PATH,
        "owner_authority_public_key_file_sha256": FROZEN_FILE_SHA256[
            OWNER_AUTHORITY_PUBLIC_KEY_PATH
        ],
        "owner_authority_ssh_fingerprint": OWNER_AUTHORITY_FINGERPRINT,
        "v2_terminal_audit_path": V2_TERMINAL_AUDIT_PATH,
        "v2_terminal_audit_file_sha256": FROZEN_FILE_SHA256[V2_TERMINAL_AUDIT_PATH],
        "v2_deny_overlay_path": V2_DENY_OVERLAY_PATH,
        "v2_deny_overlay_file_sha256": FROZEN_FILE_SHA256[V2_DENY_OVERLAY_PATH],
        "focused_tests_passed": True,
        "focused_test_working_directory": os.fspath(V3_SOURCE_REPOSITORY),
        "focused_test_environment_sha256": parsed["numerical_runtime_fingerprint_sha256"],
        "focused_test_scope": "canonical",
        "scientific_lockbox_authorized": False,
        "human_EEG_outcome_authorized": False,
    }
    if any(_plain_json(parsed[key]) != _plain_json(expected) for key, expected in fixed.items()):
        raise AuthorityError("recorded development bundle has a frozen binding mismatch")
    _require_current_numerical_runtime_binding(
        parsed["numerical_runtime_inventory"],
        str(parsed["numerical_runtime_fingerprint_sha256"]),
    )
    if tuple(parsed["tracked_source_file_inventory"]) != expected_inventory:
        raise AuthorityError("recorded development bundle inventory differs from Git objects")
    _parse_rfc3339_seconds(parsed["created_at_UTC"], "development bundle created_at_UTC")
    _require_frozen_test_argv(
        "focused_v3",
        parsed["focused_test_argv"],
        str(parsed["focused_test_junit_report_path"]),
        expected_python_site_inventory_sha256=_python_site_inventory_sha256(
            parsed["numerical_runtime_inventory"]
        ),
    )
    try:
        junit_data = base64.b64decode(
            str(parsed["focused_test_junit_report_base64"]),
            validate=True,
        )
    except (binascii.Error, ValueError) as exc:
        raise ValidationError("bundle JUnit evidence is not strict base64") from exc
    summary = _parse_pytest_junit_report(junit_data)
    if (
        base64.b64encode(junit_data).decode("ascii") != parsed["focused_test_junit_report_base64"]
        or file_sha256(junit_data) != parsed["focused_test_junit_report_file_sha256"]
        or parsed["focused_test_collected_tests"] != summary["collected"]
        or parsed["focused_test_passed_tests"] != summary["passed"]
        or parsed["focused_test_failed_tests"] != summary["failed"]
        or parsed["focused_test_skipped_tests"] != summary["skipped"]
        or parsed["focused_test_error_tests"] != summary["errors"]
        or parsed["focused_test_xfailed_tests"] != 0
        or parsed["focused_test_xpassed_tests"] != 0
        or summary["collected"] <= 0
        or summary["passed"] != summary["collected"]
        or not isinstance(parsed["focused_test_process_id"], int)
        or isinstance(parsed["focused_test_process_id"], bool)
        or parsed["focused_test_process_id"] <= 0
    ):
        raise AuthorityError("recorded bundle JUnit all-pass evidence mismatch")
    for name in (
        "focused_test_executable_file_sha256",
        "focused_test_stdout_sha256",
        "focused_test_stderr_sha256",
        "focused_test_junit_report_file_sha256",
    ):
        _require_sha256(parsed[name], f"development bundle {name}")
    frozen_bytes = {
        path: _run_git(
            V3_SOURCE_REPOSITORY,
            ("show", f"{snapshot.identity.commit}:{path}"),
        )
        for path in FROZEN_FILE_SHA256
    }
    _validate_frozen_bundle_files(frozen_bytes)
    return snapshot


def _issue_development_bundle_capability(
    parsed: Mapping[str, Any],
    *,
    file_bytes: bytes,
    snapshot: CleanSourceSnapshot,
    _observer: object,
) -> DevelopmentBundleCapability:
    if _observer is not _CAPABILITY_ISSUER:
        raise AuthorityError("development-bundle issuance requires canonical observation")
    _require_active_governed_process(
        recheck_site_packages=True,
        recheck_source=True,
    )
    return DevelopmentBundleCapability(
        schema=DEVELOPMENT_BUNDLE_SCHEMA,
        candidate_id=CANDIDATE_ID,
        payload_sha256=str(parsed["payload_sha256"]),
        file_sha256=file_sha256(file_bytes),
        clean_commit=snapshot.identity.commit,
        clean_tree=snapshot.identity.tree,
        source_bundle_sha256=snapshot.source_bundle_sha256,
        tracked_source_files=snapshot.tracked_files,
        numerical_runtime_fingerprint_sha256=str(parsed["numerical_runtime_fingerprint_sha256"]),
        numerical_runtime_inventory=parsed["numerical_runtime_inventory"],
        canonical_path=Path(DEVELOPMENT_BUNDLE_CANONICAL_PATH),
        scope="canonical",
        validated_payload=parsed,
        _issuer=_CAPABILITY_ISSUER,
    )


@dataclass(frozen=True, slots=True, init=False)
class TestOnlyArtifactReceipt:
    schema: str
    payload_sha256: str
    file_sha256: str
    path: Path
    _validation_marker: object = field(repr=False, compare=False)

    def __init__(
        self,
        *,
        schema: str,
        payload_sha256: str,
        file_sha256: str,
        path: Path,
        _issuer: object,
    ) -> None:
        if _issuer is not _CAPABILITY_ISSUER:
            raise AuthorityError("TestOnlyArtifactReceipt has no valid issuer")
        object.__setattr__(self, "schema", schema)
        object.__setattr__(self, "payload_sha256", payload_sha256)
        object.__setattr__(self, "file_sha256", file_sha256)
        object.__setattr__(self, "path", path)
        object.__setattr__(self, "_validation_marker", _issuer)


def publish_development_bundle(
    *,
    snapshot: CleanSourceSnapshot,
    frozen_file_bytes: Mapping[str, bytes],
    created_at_UTC: str,
    focused_test_run: ObservedTestRunCapability,
) -> PublishedArtifact:
    _require_active_governed_process(
        recheck_site_packages=True,
        recheck_source=True,
        require_mutation_role=True,
    )
    value = build_development_bundle(
        snapshot=snapshot,
        frozen_file_bytes=frozen_file_bytes,
        created_at_UTC=created_at_UTC,
        focused_test_run=focused_test_run,
    )
    _require_prepublication_current_source(snapshot)
    _require_frozen_files_in_current_repository(snapshot, frozen_file_bytes)
    observe_retired_development_attempts()
    return _publish_write_once(
        DEVELOPMENT_ROOT,
        "development-bundle.json",
        artifact_bytes(value),
        create_root=True,
        _publisher=_PUBLICATION_ISSUER,
    )


def _publish_development_bundle_under_test_root(
    test_root: Path,
    *,
    snapshot: CleanSourceSnapshot,
    frozen_file_bytes: Mapping[str, bytes],
    created_at_UTC: str,
    focused_test_run: ObservedTestRunCapability,
) -> PublishedArtifact:
    value = _build_development_bundle(
        snapshot=snapshot,
        frozen_file_bytes=frozen_file_bytes,
        created_at_UTC=created_at_UTC,
        focused_test_run=focused_test_run,
        expected_test_scope="test-only",
    )
    return _publish_write_once_under_test_root(
        test_root,
        "development-bundle.json",
        artifact_bytes(value),
        create_root=True,
    )


def _validate_development_bundle_at_path(
    path: Path,
    *,
    snapshot: CleanSourceSnapshot,
    frozen_file_bytes: Mapping[str, bytes],
    focused_test_run: ObservedTestRunCapability,
    scope: str,
) -> DevelopmentBundleCapability | TestOnlyArtifactReceipt:
    path = _canonical_absolute_path(path)
    if scope not in {"canonical", "test-only"}:
        raise AuthorityError("development-bundle validation scope is not recognized")
    if scope == "canonical" and path != Path(DEVELOPMENT_BUNDLE_CANONICAL_PATH):
        raise AuthorityError(
            "canonical development-bundle authority requires the exact canonical path"
        )
    if scope == "canonical":
        _require_active_governed_process(
            recheck_site_packages=True,
            recheck_source=True,
        )
        observe_retired_development_attempts()
    if scope == "canonical":
        current = capture_clean_source_snapshot(V3_SOURCE_REPOSITORY)
        if not _same_source_snapshot(current, snapshot):
            raise AuthorityError("canonical repository differs from bundle source snapshot")
        _require_frozen_files_in_current_repository(snapshot, frozen_file_bytes)
    loaded = load_exact_artifact_bytes([path], declared_paths=[path])
    file_bytes = loaded[os.fspath(path)]
    parsed = parse_artifact_bytes(
        file_bytes,
        ArtifactSpec(schema=DEVELOPMENT_BUNDLE_SCHEMA, exact_fields=_DEVELOPMENT_BUNDLE_FIELDS),
    )
    parsed = _validate_development_bundle_bytes(
        parsed,
        file_bytes=file_bytes,
        snapshot=snapshot,
        frozen_file_bytes=frozen_file_bytes,
        focused_test_run=focused_test_run,
        expected_test_scope=scope,
    )
    if scope == "canonical":
        current = capture_clean_source_snapshot(V3_SOURCE_REPOSITORY)
        if not _same_source_snapshot(current, snapshot):
            raise AuthorityError("canonical repository changed during bundle validation")
    if scope == "test-only":
        return TestOnlyArtifactReceipt(
            schema=DEVELOPMENT_BUNDLE_SCHEMA,
            payload_sha256=str(parsed["payload_sha256"]),
            file_sha256=file_sha256(file_bytes),
            path=path,
            _issuer=_CAPABILITY_ISSUER,
        )
    return _issue_development_bundle_capability(
        parsed,
        file_bytes=file_bytes,
        snapshot=snapshot,
        _observer=_CAPABILITY_ISSUER,
    )


def validate_development_bundle(
    *,
    snapshot: CleanSourceSnapshot,
    frozen_file_bytes: Mapping[str, bytes],
    focused_test_run: ObservedTestRunCapability,
) -> DevelopmentBundleCapability:
    """Issue authority only after observing exact canonical published bytes."""

    _require_active_governed_process(
        recheck_site_packages=True,
        recheck_source=True,
    )
    observe_retired_development_attempts()
    capability = _validate_development_bundle_at_path(
        Path(DEVELOPMENT_BUNDLE_CANONICAL_PATH),
        snapshot=snapshot,
        frozen_file_bytes=frozen_file_bytes,
        focused_test_run=focused_test_run,
        scope="canonical",
    )
    if type(capability) is not DevelopmentBundleCapability:
        raise AssertionError("canonical bundle validator returned a test receipt")
    return capability


def reopen_development_bundle() -> DevelopmentBundleCapability:
    """Reconstruct durable bundle authority after restart from path plus Git objects."""

    _require_active_governed_process(
        recheck_site_packages=True,
        recheck_source=True,
    )
    path = Path(DEVELOPMENT_BUNDLE_CANONICAL_PATH)
    file_bytes = load_exact_artifact_bytes([path], declared_paths=[path])[os.fspath(path)]
    parsed = parse_artifact_bytes(
        file_bytes,
        ArtifactSpec(schema=DEVELOPMENT_BUNDLE_SCHEMA, exact_fields=_DEVELOPMENT_BUNDLE_FIELDS),
    )
    snapshot = _validate_recorded_development_bundle(parsed, file_bytes=file_bytes)
    current = capture_clean_source_snapshot(V3_SOURCE_REPOSITORY)
    if not _same_source_snapshot(current, snapshot):
        _require_exact_selected_inventory_delta(
            snapshot.tracked_files,
            current.tracked_files,
        )
        _run_git(
            V3_SOURCE_REPOSITORY,
            ("merge-base", "--is-ancestor", snapshot.identity.commit, current.identity.commit),
        )
    _require_loaded_governance_source_at_snapshot(current)
    return _issue_development_bundle_capability(
        parsed,
        file_bytes=file_bytes,
        snapshot=snapshot,
        _observer=_CAPABILITY_ISSUER,
    )


def _validate_development_bundle_under_test_root(
    test_root: Path,
    *,
    snapshot: CleanSourceSnapshot,
    frozen_file_bytes: Mapping[str, bytes],
    focused_test_run: ObservedTestRunCapability,
) -> TestOnlyArtifactReceipt:
    receipt = _validate_development_bundle_at_path(
        _canonical_absolute_path(test_root) / "development-bundle.json",
        snapshot=snapshot,
        frozen_file_bytes=frozen_file_bytes,
        focused_test_run=focused_test_run,
        scope="test-only",
    )
    if type(receipt) is not TestOnlyArtifactReceipt:
        raise AssertionError("test bundle validator returned production authority")
    return receipt


@dataclass(frozen=True, slots=True, init=False)
class ValidatedArtifactCapability:
    """Nominal authority for one fully parsed, file-backed V3 artifact."""

    schema: str
    candidate_id: str
    payload_sha256: str
    file_sha256: str
    binding_sha256: str
    capability_seal_sha256: str
    canonical_path: Path
    scope: str
    source_commit: str | None
    source_tree: str | None
    _validated_payload: Mapping[str, Any] = field(repr=False, compare=False)
    _validated_bindings: Mapping[str, Any] = field(repr=False, compare=False)
    _validation_marker: object = field(repr=False, compare=False)

    def __init__(
        self,
        *,
        schema: str,
        candidate_id: str,
        payload_sha256: str,
        file_sha256: str,
        binding_sha256: str,
        validated_bindings: Mapping[str, Any],
        canonical_path: Path,
        scope: str,
        source_commit: str | None,
        source_tree: str | None,
        validated_payload: Mapping[str, Any],
        _issuer: object,
    ) -> None:
        if _issuer is not _CAPABILITY_ISSUER:
            raise AuthorityError("ValidatedArtifactCapability has no valid issuer")
        object.__setattr__(self, "schema", schema)
        object.__setattr__(self, "candidate_id", candidate_id)
        object.__setattr__(self, "payload_sha256", payload_sha256)
        object.__setattr__(self, "file_sha256", file_sha256)
        object.__setattr__(self, "binding_sha256", binding_sha256)
        object.__setattr__(self, "canonical_path", canonical_path)
        object.__setattr__(self, "scope", scope)
        object.__setattr__(self, "source_commit", source_commit)
        object.__setattr__(self, "source_tree", source_tree)
        seal = {
            "schema": schema,
            "candidate_id": candidate_id,
            "payload_sha256": payload_sha256,
            "file_sha256": file_sha256,
            "binding_sha256": binding_sha256,
            "canonical_path": os.fspath(canonical_path),
            "scope": scope,
            "source_commit": source_commit,
            "source_tree": source_tree,
        }
        object.__setattr__(self, "capability_seal_sha256", _artifact_binding_sha256(seal))
        object.__setattr__(self, "_validated_payload", _freeze_json(validated_payload))
        object.__setattr__(self, "_validated_bindings", _freeze_json(validated_bindings))
        object.__setattr__(self, "_validation_marker", _issuer)


def _artifact_binding_sha256(bindings: Mapping[str, Any]) -> str:
    return hashlib.sha256(canonical_json_bytes({"bindings": bindings})).hexdigest()


def require_validated_artifact_capability(
    value: object,
    *,
    expected_schema: str,
    expected_bindings: Mapping[str, Any] | None = None,
) -> ValidatedArtifactCapability:
    if type(value) is not ValidatedArtifactCapability:
        raise AuthorityError("exact ValidatedArtifactCapability required")
    capability = value
    if capability._validation_marker is not _CAPABILITY_ISSUER:
        raise AuthorityError("artifact capability marker is invalid")
    if capability.schema != expected_schema or capability.candidate_id != CANDIDATE_ID:
        raise AuthorityError("artifact capability identity mismatch")
    expected_paths = {
        ARTIFACT_SCHEMAS["context_reference"]: CONTEXT_REFERENCE_CANONICAL_PATH,
        ARTIFACT_SCHEMAS["development_start"]: DEVELOPMENT_START_CANONICAL_PATH,
        ARTIFACT_SCHEMAS["development_result"]: DEVELOPMENT_RESULT_CANONICAL_PATH,
        ARTIFACT_SCHEMAS["selected_method_freeze"]: (
            V3_SOURCE_REPOSITORY / SELECTED_METHOD_FREEZE_REPOSITORY_PATH
        ),
        ARTIFACT_SCHEMAS["test_evidence"]: TEST_EVIDENCE_CANONICAL_PATH,
    }
    expected_path = expected_paths.get(expected_schema)
    expected_scope = (
        "committed-source"
        if expected_schema == ARTIFACT_SCHEMAS["selected_method_freeze"]
        else "canonical-publication"
    )
    if (
        expected_path is None
        or capability.scope != expected_scope
        or capability.canonical_path != expected_path
    ):
        raise AuthorityError("artifact capability lacks exact production provenance")
    if expected_scope == "committed-source":
        _require_git_object(capability.source_commit, "artifact source commit")
        _require_git_object(capability.source_tree, "artifact source tree")
    elif capability.source_commit is not None or capability.source_tree is not None:
        raise AuthorityError("external artifact capability has invalid source provenance")
    _require_sha256(capability.payload_sha256, "artifact capability payload_sha256")
    _require_sha256(capability.file_sha256, "artifact capability file_sha256")
    seal = {
        "schema": capability.schema,
        "candidate_id": capability.candidate_id,
        "payload_sha256": capability.payload_sha256,
        "file_sha256": capability.file_sha256,
        "binding_sha256": capability.binding_sha256,
        "canonical_path": os.fspath(capability.canonical_path),
        "scope": capability.scope,
        "source_commit": capability.source_commit,
        "source_tree": capability.source_tree,
    }
    if (
        validate_payload_hash(capability._validated_payload) != capability.payload_sha256
        or file_sha256(artifact_bytes(capability._validated_payload)) != capability.file_sha256
        or capability.binding_sha256 != _artifact_binding_sha256(capability._validated_bindings)
        or capability.capability_seal_sha256 != _artifact_binding_sha256(seal)
        or any(
            key not in capability._validated_payload
            or _plain_json(capability._validated_payload[key]) != _plain_json(expected)
            for key, expected in capability._validated_bindings.items()
        )
    ):
        raise AuthorityError("artifact capability payload no longer matches its hash")
    if expected_bindings is not None and capability.binding_sha256 != _artifact_binding_sha256(
        expected_bindings
    ):
        raise AuthorityError("artifact capability binding mismatch")
    return capability


def _seal_schema_artifact(schema: str, content: Mapping[str, Any]) -> Mapping[str, Any]:
    reserved = {"schema", "candidate_id", "payload_sha256"}
    if reserved.intersection(content):
        raise ValidationError("schema-specific artifact content contains a reserved field")
    return seal_payload({"schema": schema, "candidate_id": CANDIDATE_ID, **dict(content)})


def _validate_schema_artifact(
    value: Mapping[str, Any],
    *,
    file_bytes: bytes,
    schema: str,
    exact_fields: frozenset[str],
    expected_bindings: Mapping[str, Any],
    canonical_path: Path,
    scope: str,
    source_commit: str | None = None,
    source_tree: str | None = None,
    _observer: object,
) -> ValidatedArtifactCapability:
    if _observer is not _CAPABILITY_ISSUER:
        raise AuthorityError("artifact issuance requires exact canonical observation")
    if scope != "test-only":
        _require_active_governed_process(
            recheck_site_packages=True,
            recheck_source=True,
        )
    parsed = parse_artifact_bytes(
        file_bytes, ArtifactSpec(schema=schema, exact_fields=exact_fields)
    )
    if canonical_json_bytes(parsed) != canonical_json_bytes(value):
        raise ValidationError("artifact value differs from exact supplied bytes")
    if parsed["candidate_id"] != CANDIDATE_ID:
        raise ValidationError("artifact candidate_id mismatch")
    for key, expected in expected_bindings.items():
        if key not in parsed or _plain_json(parsed[key]) != _plain_json(expected):
            raise ValidationError(f"artifact frozen binding mismatch: {key}")
    return ValidatedArtifactCapability(
        schema=schema,
        candidate_id=CANDIDATE_ID,
        payload_sha256=str(parsed["payload_sha256"]),
        file_sha256=file_sha256(file_bytes),
        binding_sha256=_artifact_binding_sha256(expected_bindings),
        validated_bindings=expected_bindings,
        canonical_path=canonical_path,
        scope=scope,
        source_commit=source_commit,
        source_tree=source_tree,
        validated_payload=parsed,
        _issuer=_CAPABILITY_ISSUER,
    )


def _require_core_context_reference_capability(
    value: object,
    *,
    parsed: Mapping[str, Any],
    development_bundle: DevelopmentBundleCapability,
    rng_authority: object,
) -> object:
    try:
        from cfeg.models.metadata_calibration_v3 import (
            require_context_reference_rng_authority,
            require_validated_context_reference,
        )
    except ImportError as exc:
        raise AuthorityError("V3 core context validator is unavailable") from exc
    bundle = require_context_reference_rng_bundle_capability(
        development_bundle,
        expected_commit=development_bundle.clean_commit,
        expected_tree=development_bundle.clean_tree,
    )
    authority = require_context_reference_rng_authority(rng_authority)
    capability = require_validated_context_reference(
        value,
        expected_payload_sha256=str(parsed["payload_sha256"]),
    )
    if (
        capability.candidate_id != CANDIDATE_ID
        or capability.payload_schema != ARTIFACT_SCHEMAS["context_reference"]
        or capability.payload_sha256 != parsed["payload_sha256"]
        or capability.domain != parsed["domain"]
        or tuple(capability.canonical_channels) != tuple(parsed["canonical_channels"])
        or tuple(capability.interface_lookup_keys)
        != tuple(table["interface_lookup_key"] for table in parsed["interface_tables"])
        or capability.minimum_per_interface_channel_count
        != parsed["minimum_per_interface_channel_count"]
        or capability.pooled_per_channel_fallback_minimum_count
        != parsed["pooled_per_channel_fallback_minimum_count"]
        or capability.scale_floor != parsed["scale_floor"]
        or authority.development_bundle_payload_sha256 != bundle.payload_sha256
        or authority.development_bundle_file_sha256 != bundle.file_sha256
        or authority.development_bundle_source_bundle_sha256 != bundle.source_bundle_sha256
        or authority.numerical_runtime_fingerprint_sha256
        != bundle.numerical_runtime_fingerprint_sha256
        or authority.clean_commit != bundle.clean_commit
        or authority.clean_tree != bundle.clean_tree
        or capability.context_reference_rng_authority_schema != authority.schema
        or capability.context_reference_rng_authority_semantic_binding_sha256
        != authority.semantic_binding_sha256
        or capability.development_bundle_payload_sha256 != bundle.payload_sha256
        or capability.development_bundle_file_sha256 != bundle.file_sha256
        or capability.development_bundle_source_bundle_sha256 != bundle.source_bundle_sha256
        or capability.numerical_runtime_fingerprint_sha256
        != bundle.numerical_runtime_fingerprint_sha256
        or capability.reference_root_seed != 20_260_910
        or capability.rng_key_map_sha256 != authority.key_map_sha256
    ):
        raise AuthorityError("V3 core context capability does not bind the exact artifact")
    return capability


def _require_core_development_result_capability(
    value: object,
    *,
    parsed: Mapping[str, Any],
    development_bundle: DevelopmentBundleCapability,
    context_reference: ValidatedArtifactCapability,
    development_start: DevelopmentStartExecutionCapability,
    development_rng_authority: object,
    validated_context_reference: object,
) -> object:
    try:
        from cfeg.analysis.metadata_calibration_v3_synthetic import (
            require_validated_development_result,
            validate_development_result_for_publication,
        )
        from cfeg.models.metadata_calibration_v3 import (
            require_development_rng_authority,
            require_validated_context_reference,
        )
    except ImportError as exc:
        raise AuthorityError("V3 core development validator is unavailable") from exc
    bundle = require_development_rng_bundle_capability(
        development_bundle,
        expected_commit=development_bundle.clean_commit,
        expected_tree=development_bundle.clean_tree,
    )
    reference_artifact = require_validated_artifact_capability(
        context_reference,
        expected_schema=ARTIFACT_SCHEMAS["context_reference"],
    )
    start = require_development_start_execution_capability(development_start)
    authority = require_development_rng_authority(development_rng_authority)
    reference_proof = require_validated_context_reference(
        validated_context_reference,
        expected_payload_sha256=reference_artifact.payload_sha256,
    )
    capability = validate_development_result_for_publication(
        parsed,
        validated_result=value,
    )
    capability = require_validated_development_result(
        capability,
        expected_payload_sha256=str(parsed["payload_sha256"]),
    )
    reports = parsed["complete_grid_gate_report"]
    shared_bindings = {
        "candidate_id": CANDIDATE_ID,
        "payload_schema": ARTIFACT_SCHEMAS["development_result"],
        "payload_sha256": parsed["payload_sha256"],
        "master_plan_file_sha256": parsed["master_plan_file_sha256"],
        "synthetic_plan_file_sha256": parsed["synthetic_plan_file_sha256"],
        "development_bundle_schema": parsed["development_bundle_schema"],
        "development_bundle_payload_sha256": parsed["development_bundle_payload_sha256"],
        "development_bundle_file_sha256": parsed["development_bundle_file_sha256"],
        "context_reference_schema": parsed["context_reference_schema"],
        "context_reference_payload_sha256": parsed["context_reference_payload_sha256"],
        "context_reference_file_sha256": parsed["context_reference_file_sha256"],
        "development_start_schema": parsed["development_start_schema"],
        "development_start_payload_sha256": parsed["development_start_payload_sha256"],
        "development_start_file_sha256": parsed["development_start_file_sha256"],
        "development_rng_primitive_schema": parsed["development_rng_primitive_schema"],
        "development_root_seed": parsed["development_root_seed"],
        "participant_count": parsed["participant_count"],
        "b4_source_range_stress_participant_indices": tuple(
            parsed["B4_source_range_stress_participant_indices"]
        ),
        "grid_cell_count": len(parsed["grid_cells"]),
        "metric_row_count": len(parsed["participant_metric_rows"]),
        "invariant_row_count": len(parsed["invariant_rows"]),
        "total_row_count": len(parsed["participant_metric_rows"]) + len(parsed["invariant_rows"]),
        "participant_metric_rows_sha256": hashlib.sha256(
            canonical_json_bytes(
                {
                    "schema": (
                        "cfeg.metadata-calibration-efficiency-v3.participant-metric-row-set.v1"
                    ),
                    "rows": parsed["participant_metric_rows"],
                }
            )
        ).hexdigest(),
        "invariant_rows_sha256": hashlib.sha256(
            canonical_json_bytes(
                {
                    "schema": ("cfeg.metadata-calibration-efficiency-v3.invariant-row-set.v1"),
                    "rows": parsed["invariant_rows"],
                }
            )
        ).hexdigest(),
        "gate_report_count": len(parsed["complete_grid_gate_report"]),
        "complete_grid_gate_report_sha256": hashlib.sha256(
            canonical_json_bytes(
                {
                    "schema": (
                        "cfeg.metadata-calibration-efficiency-v3.complete-grid-gate-report.v1"
                    ),
                    "reports": parsed["complete_grid_gate_report"],
                }
            )
        ).hexdigest(),
        "uniform_block_weight_sensitivity_sha256": hashlib.sha256(
            canonical_json_bytes(
                {
                    "schema": (
                        "cfeg.metadata-calibration-efficiency-v3."
                        "complete-uniform-block-weight-sensitivity.v1"
                    ),
                    "reports": [report["uniform_block_weight_sensitivity"] for report in reports],
                }
            )
        ).hexdigest(),
        "resampling_sensitivity_sha256": hashlib.sha256(
            canonical_json_bytes(
                {
                    "schema": (
                        "cfeg.metadata-calibration-efficiency-v3.complete-resampling-sensitivity.v1"
                    ),
                    "reports": [report["resampling_sensitivity"] for report in reports],
                }
            )
        ).hexdigest(),
        "sensitivity_participant_vectors_sha256": hashlib.sha256(
            canonical_json_bytes(
                {
                    "schema": (
                        "cfeg.metadata-calibration-efficiency-v3."
                        "complete-sensitivity-participant-vectors.v1"
                    ),
                    "reports": [report["sensitivity_participant_vectors"] for report in reports],
                }
            )
        ).hexdigest(),
        "selection_status": parsed["selection_status"],
        "selected_grid_cell_id": parsed["selected_grid_cell_id"],
    }
    if any(getattr(capability, name) != expected for name, expected in shared_bindings.items()):
        raise AuthorityError("V3 core development capability does not bind the exact artifact")
    if (
        authority.development_bundle_payload_sha256 != bundle.payload_sha256
        or authority.development_bundle_file_sha256 != bundle.file_sha256
        or authority.development_bundle_source_bundle_sha256 != bundle.source_bundle_sha256
        or authority.numerical_runtime_fingerprint_sha256
        != bundle.numerical_runtime_fingerprint_sha256
        or authority.clean_commit != bundle.clean_commit
        or authority.clean_tree != bundle.clean_tree
        or capability.development_bundle_source_bundle_sha256 != bundle.source_bundle_sha256
        or capability.numerical_runtime_fingerprint_sha256
        != bundle.numerical_runtime_fingerprint_sha256
        or capability.development_rng_authority_schema != authority.schema
        or capability.development_rng_authority_semantic_binding_sha256
        != authority.semantic_binding_sha256
        or capability.rng_key_map_sha256 != authority.key_map_sha256
        or capability.context_reference_payload_sha256 != reference_artifact.payload_sha256
        or capability.context_reference_file_sha256 != reference_artifact.file_sha256
        or authority.development_start_schema != start.receipt_schema
        or authority.development_start_payload_sha256 != start.receipt_payload_sha256
        or authority.development_start_file_sha256 != start.receipt_file_sha256
        or parsed["development_start_schema"] != start.receipt_schema
        or parsed["development_start_payload_sha256"] != start.receipt_payload_sha256
        or parsed["development_start_file_sha256"] != start.receipt_file_sha256
        or capability.validated_context_reference_semantic_binding_sha256
        != reference_proof.semantic_binding_sha256
        or reference_proof.development_bundle_payload_sha256 != bundle.payload_sha256
        or reference_proof.development_bundle_file_sha256 != bundle.file_sha256
        or reference_proof.development_bundle_source_bundle_sha256 != bundle.source_bundle_sha256
        or reference_proof.numerical_runtime_fingerprint_sha256
        != bundle.numerical_runtime_fingerprint_sha256
    ):
        raise AuthorityError("V3 development proof is not one bundle/RNG/reference execution chain")
    return capability


def _require_core_selected_method_proposal(
    value: object,
    *,
    parsed: Mapping[str, Any],
) -> object:
    try:
        from cfeg.analysis.metadata_calibration_v3_synthetic import (
            require_selected_method_proposal,
        )
    except ImportError as exc:
        raise AuthorityError("V3 core selected-method validator is unavailable") from exc
    capability = require_selected_method_proposal(
        value,
        expected_payload_sha256=str(parsed["payload_sha256"]),
    )
    rank = parsed["selection_rank_tuple"]
    shared_bindings = {
        "candidate_id": CANDIDATE_ID,
        "payload_schema": ARTIFACT_SCHEMAS["selected_method_freeze"],
        "payload_sha256": parsed["payload_sha256"],
        "scientific_candidate_id": parsed["scientific_candidate_id"],
        "selected_grid_cell_index": parsed["selected_grid_cell_index"],
        "selected_grid_cell_id": parsed["selected_grid_cell_id"],
        "selected_prototype_prior_pseudocount": parsed["selected_prototype_prior_pseudocount"],
        "selected_lambda_max": parsed["selected_lambda_max"],
        "selected_operator_instance_sha256": parsed["selected_operator_instance_sha256"],
        "master_plan_file_sha256": parsed["master_plan_file_sha256"],
        "synthetic_plan_file_sha256": parsed["synthetic_plan_file_sha256"],
        "development_bundle_schema": parsed["development_bundle_schema"],
        "development_bundle_payload_sha256": parsed["development_bundle_payload_sha256"],
        "development_bundle_file_sha256": parsed["development_bundle_file_sha256"],
        "development_result_schema": parsed["development_result_schema"],
        "development_result_payload_sha256": parsed["development_result_payload_sha256"],
        "development_result_file_sha256": parsed["development_result_file_sha256"],
        "complete_grid_gate_report_sha256": parsed["complete_grid_gate_report_sha256"],
        "minimum_mandatory_observed_gain": rank["minimum_mandatory_observed_gain"],
        "minimum_corresponding_one_sided_LCB": rank["minimum_corresponding_one_sided_LCB"],
        "clean_commit": parsed["clean_commit"],
        "clean_tree": parsed["clean_tree"],
    }
    if any(getattr(capability, name) != expected for name, expected in shared_bindings.items()):
        raise AuthorityError("V3 core selected-method capability does not bind exact artifact")
    return capability


_CONTEXT_REFERENCE_FIELDS = frozenset(
    {
        "schema",
        "candidate_id",
        "domain",
        "canonical_channels",
        "interface_tables",
        "pooled_table",
        "minimum_per_interface_channel_count",
        "pooled_per_channel_fallback_minimum_count",
        "scale_floor",
        "payload_sha256",
    }
)
_CONTEXT_TABLE_FIELDS = frozenset({"interface_lookup_key", "centers", "scales", "observed_counts"})
_SYNTHETIC_CHANNELS = tuple(f"ch{index:02d}" for index in range(8))
_SYNTHETIC_INTERFACES = ("neutral", "wet", "dry")


def _validate_context_reference_content(value: Mapping[str, Any]) -> None:
    _require_exact_keys(value, _CONTEXT_REFERENCE_FIELDS, "context reference")
    if value.get("schema") != ARTIFACT_SCHEMAS["context_reference"]:
        raise ValidationError("context-reference schema mismatch")
    if value.get("candidate_id") != CANDIDATE_ID:
        raise ValidationError("context-reference candidate mismatch")
    if value.get("domain") != "synthetic_development_and_future_synthetic":
        raise AuthorityError(
            "current V3 contract permits only the outcome-free synthetic reference"
        )
    channels = tuple(value.get("canonical_channels", ()))
    if channels != _SYNTHETIC_CHANNELS:
        raise ValidationError("synthetic context-reference channels must be ch00 through ch07")
    tables = value.get("interface_tables")
    if not isinstance(tables, (list, tuple)) or len(tables) != 3:
        raise ValidationError("synthetic context reference requires exactly three interface tables")
    if tuple(table.get("interface_lookup_key") for table in tables) != _SYNTHETIC_INTERFACES:
        raise ValidationError("synthetic context interface order must be neutral, wet, dry")
    all_tables = (*tables, value.get("pooled_table"))
    for index, table in enumerate(all_tables):
        if not isinstance(table, Mapping):
            raise ValidationError("context-reference table must be an object")
        _require_exact_keys(table, _CONTEXT_TABLE_FIELDS, "context-reference table")
        expected_key = "__pooled__" if index == 3 else _SYNTHETIC_INTERFACES[index]
        if table["interface_lookup_key"] != expected_key:
            raise ValidationError("context-reference table lookup key mismatch")
        for field_name in ("centers", "scales", "observed_counts"):
            if not isinstance(table[field_name], (list, tuple)) or len(table[field_name]) != 8:
                raise ValidationError(
                    "context-reference vectors must match eight canonical channels"
                )
        for scale in table["scales"]:
            if not isinstance(scale, (int, float)) or isinstance(scale, bool) or scale < 0.05:
                raise ValidationError("context-reference scales must satisfy the frozen floor")
        minimum = 256 if index == 3 else 128
        for count in table["observed_counts"]:
            if not isinstance(count, int) or isinstance(count, bool) or count < minimum:
                raise ValidationError(
                    "context-reference observed count is below the frozen minimum"
                )
    if (
        value["minimum_per_interface_channel_count"] != 128
        or value["pooled_per_channel_fallback_minimum_count"] != 256
        or value["scale_floor"] != 0.05
    ):
        raise ValidationError("context-reference robust-estimator constants changed")


def seal_context_reference(content: Mapping[str, Any]) -> Mapping[str, Any]:
    value = _seal_schema_artifact(ARTIFACT_SCHEMAS["context_reference"], content)
    _validate_context_reference_content(value)
    return value


def _validate_context_reference_bytes(
    value: Mapping[str, Any],
    *,
    file_bytes: bytes,
    development_bundle: DevelopmentBundleCapability,
    rng_authority: object,
    core_capability: object,
) -> Mapping[str, Any]:
    _validate_context_reference_content(value)
    parsed = parse_artifact_bytes(
        file_bytes,
        ArtifactSpec(
            schema=ARTIFACT_SCHEMAS["context_reference"],
            exact_fields=_CONTEXT_REFERENCE_FIELDS,
        ),
    )
    if canonical_json_bytes(parsed) != canonical_json_bytes(value):
        raise ValidationError("context-reference value differs from supplied bytes")
    _require_core_context_reference_capability(
        core_capability,
        parsed=parsed,
        development_bundle=development_bundle,
        rng_authority=rng_authority,
    )
    return parsed


def publish_context_reference(
    value: Mapping[str, Any],
    *,
    development_bundle: DevelopmentBundleCapability,
    rng_authority: object,
    core_capability: object,
) -> PublishedArtifact:
    _require_active_governed_process(
        recheck_site_packages=True,
        recheck_source=True,
        require_mutation_role=True,
    )
    data = artifact_bytes(value)
    retired_v1, _ = observe_retired_development_attempts()
    if data != retired_v1_context_reference_bytes(retired_v1):
        raise AuthorityError(
            "development-v3 must adopt the exact retired development-v1 context bytes"
        )
    _validate_context_reference_bytes(
        value,
        file_bytes=data,
        development_bundle=development_bundle,
        rng_authority=rng_authority,
        core_capability=core_capability,
    )
    require_context_reference_rng_bundle_capability(
        development_bundle,
        expected_commit=development_bundle.clean_commit,
        expected_tree=development_bundle.clean_tree,
    )
    retired_v1_context_reference_bytes(retired_v1)
    return _publish_write_once(
        DEVELOPMENT_ROOT,
        "context-reference.json",
        data,
        create_root=True,
        _publisher=_PUBLICATION_ISSUER,
    )


def _publish_context_reference_under_test_root(
    test_root: Path,
    value: Mapping[str, Any],
    *,
    development_bundle: DevelopmentBundleCapability,
    rng_authority: object,
    core_capability: object,
) -> PublishedArtifact:
    data = artifact_bytes(value)
    _validate_context_reference_bytes(
        value,
        file_bytes=data,
        development_bundle=development_bundle,
        rng_authority=rng_authority,
        core_capability=core_capability,
    )
    return _publish_write_once_under_test_root(
        test_root,
        "context-reference.json",
        data,
        create_root=True,
    )


def _validate_context_reference_at_path(
    path: Path,
    *,
    development_bundle: DevelopmentBundleCapability,
    rng_authority: object,
    core_capability: object,
    scope: str,
) -> ValidatedArtifactCapability | TestOnlyArtifactReceipt:
    path = _canonical_absolute_path(path)
    if scope not in {"canonical", "test-only"}:
        raise AuthorityError("context-reference validation scope is not recognized")
    if scope == "canonical" and path != CONTEXT_REFERENCE_CANONICAL_PATH:
        raise AuthorityError(
            "canonical context-reference authority requires the exact canonical path"
        )
    if scope == "canonical":
        _require_active_governed_process(
            recheck_site_packages=True,
            recheck_source=True,
        )
    file_bytes = load_exact_artifact_bytes([path], declared_paths=[path])[os.fspath(path)]
    parsed = parse_artifact_bytes(
        file_bytes,
        ArtifactSpec(
            schema=ARTIFACT_SCHEMAS["context_reference"],
            exact_fields=_CONTEXT_REFERENCE_FIELDS,
        ),
    )
    if scope == "canonical":
        retired_v1, _ = observe_retired_development_attempts()
        if file_bytes != retired_v1_context_reference_bytes(retired_v1):
            raise AuthorityError(
                "development-v3 context differs from the exact retired V1 reference"
            )
    parsed = _validate_context_reference_bytes(
        parsed,
        file_bytes=file_bytes,
        development_bundle=development_bundle,
        rng_authority=rng_authority,
        core_capability=core_capability,
    )
    bindings = {
        "domain": "synthetic_development_and_future_synthetic",
        "canonical_channels": list(_SYNTHETIC_CHANNELS),
        "minimum_per_interface_channel_count": 128,
        "pooled_per_channel_fallback_minimum_count": 256,
        "scale_floor": 0.05,
    }
    if scope == "test-only":
        return TestOnlyArtifactReceipt(
            schema=ARTIFACT_SCHEMAS["context_reference"],
            payload_sha256=str(parsed["payload_sha256"]),
            file_sha256=file_sha256(file_bytes),
            path=path,
            _issuer=_CAPABILITY_ISSUER,
        )
    return _validate_schema_artifact(
        parsed,
        file_bytes=file_bytes,
        schema=ARTIFACT_SCHEMAS["context_reference"],
        exact_fields=_CONTEXT_REFERENCE_FIELDS,
        expected_bindings=bindings,
        canonical_path=path,
        scope="canonical-publication",
        _observer=_CAPABILITY_ISSUER,
    )


def validate_context_reference(
    *,
    development_bundle: DevelopmentBundleCapability,
    rng_authority: object,
    core_capability: object,
) -> ValidatedArtifactCapability:
    _require_active_governed_process(
        recheck_site_packages=True,
        recheck_source=True,
    )
    capability = _validate_context_reference_at_path(
        CONTEXT_REFERENCE_CANONICAL_PATH,
        development_bundle=development_bundle,
        rng_authority=rng_authority,
        core_capability=core_capability,
        scope="canonical",
    )
    if type(capability) is not ValidatedArtifactCapability:
        raise AssertionError("canonical context validator returned test receipt")
    return capability


_DEVELOPMENT_START_FIELDS = frozenset(
    {
        "schema",
        "candidate_id",
        "protocol_revision",
        "development_attempt_id",
        "started_at_UTC",
        "canonical_path",
        "creator_process_id",
        "creator_executable_file_sha256",
        "clean_git_commit",
        "clean_git_tree",
        "development_bundle_schema",
        "development_bundle_payload_sha256",
        "development_bundle_file_sha256",
        "development_bundle_source_bundle_sha256",
        "numerical_runtime_fingerprint_sha256",
        "context_reference_schema",
        "context_reference_payload_sha256",
        "context_reference_file_sha256",
        "development_rng_primitive_schema",
        "development_root_seed",
        "development_result_schema",
        "development_result_canonical_path",
        "development_rng_authority_issued_before_receipt",
        "development_seedsequence_created_before_receipt",
        "development_DGP_executed_before_receipt",
        "presence_without_result_terminal",
        "scientific_lockbox_authorized",
        "human_EEG_outcome_authorized",
        "payload_sha256",
    }
)


@dataclass(frozen=True, slots=True, init=False)
class DevelopmentStartExecutionCapability:
    """Creator-process-only authority proving the durable development boundary."""

    schema: str
    candidate_id: str
    receipt_schema: str
    receipt_payload_sha256: str
    receipt_file_sha256: str
    development_bundle_payload_sha256: str
    development_bundle_file_sha256: str
    context_reference_payload_sha256: str
    context_reference_file_sha256: str
    clean_commit: str
    clean_tree: str
    creator_process_id: int
    semantic_binding_sha256: str
    _bundle: DevelopmentBundleCapability = field(repr=False, compare=False)
    _context_reference: ValidatedArtifactCapability = field(repr=False, compare=False)
    _receipt: ValidatedArtifactCapability = field(repr=False, compare=False)
    _governed_process: GovernedProcessCapability = field(repr=False, compare=False)
    _instance_identity: object = field(repr=False, compare=False)
    _validation_marker: object = field(repr=False, compare=False)

    def __new__(  # noqa: PYI034 -- Python 3.10 does not provide typing.Self.
        cls,
        *_args: object,
        **_kwargs: object,
    ) -> DevelopmentStartExecutionCapability:
        raise TypeError(
            "DevelopmentStartExecutionCapability is issued only by the canonical "
            "write-once start publisher"
        )


_ACTIVE_DEVELOPMENT_START_CAPABILITIES: dict[int, object] = {}


def _development_start_execution_binding(
    *,
    bundle: DevelopmentBundleCapability,
    context: ValidatedArtifactCapability,
    receipt: ValidatedArtifactCapability,
    process: GovernedProcessCapability,
) -> Mapping[str, Any]:
    return MappingProxyType(
        {
            "schema": DEVELOPMENT_START_EXECUTION_CAPABILITY_SCHEMA,
            "candidate_id": CANDIDATE_ID,
            "receipt_schema": receipt.schema,
            "receipt_payload_sha256": receipt.payload_sha256,
            "receipt_file_sha256": receipt.file_sha256,
            "development_bundle_payload_sha256": bundle.payload_sha256,
            "development_bundle_file_sha256": bundle.file_sha256,
            "context_reference_payload_sha256": context.payload_sha256,
            "context_reference_file_sha256": context.file_sha256,
            "clean_commit": bundle.clean_commit,
            "clean_tree": bundle.clean_tree,
            "creator_process_id": process.process_id,
        }
    )


def _require_exact_development_attempt_inventory(expected_names: frozenset[str]) -> None:
    """Reject gaps, aliases, and unplanned files in the active attempt directory."""

    descriptor = _open_absolute_directory(
        DEVELOPMENT_ROOT,
        create=False,
        require_private_final=True,
    )
    try:
        before = os.fstat(descriptor)
        try:
            names = frozenset(os.listdir(descriptor))
        except OSError as exc:
            raise AuthorityError("active development inventory cannot be enumerated") from exc
        after = os.fstat(descriptor)
        if (
            names != expected_names
            or _retired_v1_stat_identity(before) != _retired_v1_stat_identity(after)
            or frozenset(os.listdir(descriptor)) != expected_names
        ):
            raise AuthorityError("active development directory inventory is not exact")
    finally:
        os.close(descriptor)


def _require_development_bundle_and_context(
    development_bundle: object,
    context_reference: object,
    *,
    expected_commit: str,
    expected_tree: str,
) -> tuple[DevelopmentBundleCapability, ValidatedArtifactCapability]:
    bundle = require_development_rng_bundle_capability(
        development_bundle,
        expected_commit=expected_commit,
        expected_tree=expected_tree,
    )
    context = require_validated_artifact_capability(
        context_reference,
        expected_schema=ARTIFACT_SCHEMAS["context_reference"],
    )
    data = load_exact_artifact_bytes(
        [CONTEXT_REFERENCE_CANONICAL_PATH],
        declared_paths=[CONTEXT_REFERENCE_CANONICAL_PATH],
    )[os.fspath(CONTEXT_REFERENCE_CANONICAL_PATH)]
    retired, _ = observe_retired_development_attempts()
    parsed = parse_artifact_bytes(
        data,
        ArtifactSpec(
            schema=ARTIFACT_SCHEMAS["context_reference"],
            exact_fields=_CONTEXT_REFERENCE_FIELDS,
        ),
    )
    if (
        data != retired_v1_context_reference_bytes(retired)
        or context.payload_sha256 != parsed["payload_sha256"]
        or context.file_sha256 != file_sha256(data)
        or canonical_json_bytes(context._validated_payload) != canonical_json_bytes(parsed)
    ):
        raise AuthorityError(
            "development RNG requires the exact validated development-v3 reference"
        )
    return bundle, context


def _development_start_bindings(
    bundle: DevelopmentBundleCapability,
    context: ValidatedArtifactCapability,
) -> Mapping[str, Any]:
    return MappingProxyType(
        {
            "protocol_revision": PROTOCOL_REVISION,
            "development_attempt_id": DEVELOPMENT_ATTEMPT_ID,
            "canonical_path": os.fspath(DEVELOPMENT_START_CANONICAL_PATH),
            "clean_git_commit": bundle.clean_commit,
            "clean_git_tree": bundle.clean_tree,
            "development_bundle_schema": bundle.schema,
            "development_bundle_payload_sha256": bundle.payload_sha256,
            "development_bundle_file_sha256": bundle.file_sha256,
            "development_bundle_source_bundle_sha256": bundle.source_bundle_sha256,
            "numerical_runtime_fingerprint_sha256": (bundle.numerical_runtime_fingerprint_sha256),
            "context_reference_schema": context.schema,
            "context_reference_payload_sha256": context.payload_sha256,
            "context_reference_file_sha256": context.file_sha256,
            "development_rng_primitive_schema": (
                "cfeg.metadata-calibration-efficiency-v3.development-rng-binding.v1"
            ),
            "development_root_seed": 20_260_909,
            "development_result_schema": ARTIFACT_SCHEMAS["development_result"],
            "development_result_canonical_path": os.fspath(DEVELOPMENT_RESULT_CANONICAL_PATH),
            "development_rng_authority_issued_before_receipt": False,
            "development_seedsequence_created_before_receipt": False,
            "development_DGP_executed_before_receipt": False,
            "presence_without_result_terminal": True,
            "scientific_lockbox_authorized": False,
            "human_EEG_outcome_authorized": False,
        }
    )


def _validate_development_start_content(
    value: Mapping[str, Any],
    *,
    bundle: DevelopmentBundleCapability,
    context: ValidatedArtifactCapability,
) -> Mapping[str, Any]:
    _require_exact_keys(value, _DEVELOPMENT_START_FIELDS, "development start receipt")
    if (
        value.get("schema") != ARTIFACT_SCHEMAS["development_start"]
        or value.get("candidate_id") != CANDIDATE_ID
    ):
        raise ValidationError("development start receipt identity mismatch")
    _parse_rfc3339_seconds(value.get("started_at_UTC"), "development start time")
    process_id = value.get("creator_process_id")
    if not isinstance(process_id, int) or isinstance(process_id, bool) or process_id <= 0:
        raise ValidationError("development start creator PID is invalid")
    _require_sha256(
        value.get("creator_executable_file_sha256"),
        "development start creator executable",
    )
    bindings = _development_start_bindings(bundle, context)
    for name, expected in bindings.items():
        if _plain_json(value.get(name)) != _plain_json(expected):
            raise ValidationError(f"development start binding mismatch: {name}")
    python_runtime = bundle.numerical_runtime_inventory.get("python")
    if not isinstance(python_runtime, Mapping) or value[
        "creator_executable_file_sha256"
    ] != python_runtime.get("executable_file_sha256"):
        raise AuthorityError("development start creator executable differs from the bundle")
    validate_payload_hash(value)
    return value


def _validate_development_start_bytes(
    data: bytes,
    *,
    bundle: DevelopmentBundleCapability,
    context: ValidatedArtifactCapability,
) -> ValidatedArtifactCapability:
    parsed = parse_artifact_bytes(
        data,
        ArtifactSpec(
            schema=ARTIFACT_SCHEMAS["development_start"],
            exact_fields=_DEVELOPMENT_START_FIELDS,
        ),
    )
    _validate_development_start_content(parsed, bundle=bundle, context=context)
    bindings = {
        **dict(_development_start_bindings(bundle, context)),
        "creator_process_id": parsed["creator_process_id"],
        "creator_executable_file_sha256": parsed["creator_executable_file_sha256"],
        "started_at_UTC": parsed["started_at_UTC"],
    }
    return _validate_schema_artifact(
        parsed,
        file_bytes=data,
        schema=ARTIFACT_SCHEMAS["development_start"],
        exact_fields=_DEVELOPMENT_START_FIELDS,
        expected_bindings=bindings,
        canonical_path=DEVELOPMENT_START_CANONICAL_PATH,
        scope="canonical-publication",
        _observer=_CAPABILITY_ISSUER,
    )


def reopen_development_start_receipt(
    *,
    development_bundle: DevelopmentBundleCapability,
    context_reference: ValidatedArtifactCapability,
) -> ValidatedArtifactCapability:
    """Reopen receipt evidence without reconstructing execution authority."""

    _require_active_governed_process(recheck_site_packages=True, recheck_source=True)
    bundle = require_development_recovery_bundle_capability(
        development_bundle,
        expected_commit=development_bundle.clean_commit,
        expected_tree=development_bundle.clean_tree,
    )
    context = require_validated_artifact_capability(
        context_reference,
        expected_schema=ARTIFACT_SCHEMAS["context_reference"],
    )
    data = load_exact_artifact_bytes(
        [DEVELOPMENT_START_CANONICAL_PATH],
        declared_paths=[DEVELOPMENT_START_CANONICAL_PATH],
    )[os.fspath(DEVELOPMENT_START_CANONICAL_PATH)]
    return _validate_development_start_bytes(data, bundle=bundle, context=context)


def publish_development_start(
    *,
    development_bundle: DevelopmentBundleCapability,
    context_reference: ValidatedArtifactCapability,
    started_at_UTC: str,
) -> DevelopmentStartExecutionCapability:
    """Consume the attempt immediately before any development authority or RNG."""

    process = _require_active_governed_process(
        recheck_site_packages=True,
        recheck_source=True,
        require_mutation_role=True,
    )
    if process.role != "runner_resume":
        raise AuthorityError("development start requires the exact runner-resume role")
    bundle, context = _require_development_bundle_and_context(
        development_bundle,
        context_reference,
        expected_commit=development_bundle.clean_commit,
        expected_tree=development_bundle.clean_tree,
    )
    _require_exact_development_attempt_inventory(
        frozenset({"development-bundle.json", "context-reference.json"})
    )
    _parse_rfc3339_seconds(started_at_UTC, "development start time")
    content = {
        **dict(_development_start_bindings(bundle, context)),
        "started_at_UTC": started_at_UTC,
        "creator_process_id": process.process_id,
        "creator_executable_file_sha256": process.executable_file_sha256,
    }
    value = _seal_schema_artifact(ARTIFACT_SCHEMAS["development_start"], content)
    _validate_development_start_content(value, bundle=bundle, context=context)
    data = artifact_bytes(value)
    _publish_write_once(
        DEVELOPMENT_ROOT,
        DEVELOPMENT_START_CANONICAL_PATH.name,
        data,
        _publisher=_PUBLICATION_ISSUER,
    )
    _require_exact_development_attempt_inventory(
        frozenset({"development-bundle.json", "context-reference.json", "development-start.json"})
    )
    observed = load_exact_artifact_bytes(
        [DEVELOPMENT_START_CANONICAL_PATH],
        declared_paths=[DEVELOPMENT_START_CANONICAL_PATH],
    )[os.fspath(DEVELOPMENT_START_CANONICAL_PATH)]
    if observed != data:
        raise AuthorityError("development start receipt differs after write-once publication")
    receipt = _validate_development_start_bytes(observed, bundle=bundle, context=context)
    binding = _development_start_execution_binding(
        bundle=bundle,
        context=context,
        receipt=receipt,
        process=process,
    )
    capability = object.__new__(DevelopmentStartExecutionCapability)
    for name, item in binding.items():
        object.__setattr__(capability, name, item)
    identity = object()
    object.__setattr__(capability, "semantic_binding_sha256", _artifact_binding_sha256(binding))
    object.__setattr__(capability, "_bundle", bundle)
    object.__setattr__(capability, "_context_reference", context)
    object.__setattr__(capability, "_receipt", receipt)
    object.__setattr__(capability, "_governed_process", process)
    object.__setattr__(capability, "_instance_identity", identity)
    object.__setattr__(
        capability,
        "_validation_marker",
        _DEVELOPMENT_START_EXECUTION_ISSUER,
    )
    _ACTIVE_DEVELOPMENT_START_CAPABILITIES[id(capability)] = identity
    return require_development_start_execution_capability(capability)


def require_development_start_execution_capability(
    value: object,
) -> DevelopmentStartExecutionCapability:
    """Validate the unique live creator-process authority and exact receipt bytes."""

    if type(value) is not DevelopmentStartExecutionCapability:
        raise AuthorityError("exact DevelopmentStartExecutionCapability required")
    capability = value
    try:
        marker = capability._validation_marker
        instance_identity = capability._instance_identity
    except AttributeError as exc:
        raise AuthorityError("development start capability is incomplete") from exc
    if (
        marker is not _DEVELOPMENT_START_EXECUTION_ISSUER
        or _ACTIVE_DEVELOPMENT_START_CAPABILITIES.get(id(capability)) is not instance_identity
        or capability.creator_process_id != os.getpid()
    ):
        raise AuthorityError("development start capability is not the live creator instance")
    process = require_governed_process_capability(
        capability._governed_process,
        expected_role="runner_resume",
    )
    bundle = require_development_rng_bundle_capability(
        capability._bundle,
        expected_commit=capability.clean_commit,
        expected_tree=capability.clean_tree,
    )
    context = require_validated_artifact_capability(
        capability._context_reference,
        expected_schema=ARTIFACT_SCHEMAS["context_reference"],
    )
    data = load_exact_artifact_bytes(
        [DEVELOPMENT_START_CANONICAL_PATH],
        declared_paths=[DEVELOPMENT_START_CANONICAL_PATH],
    )[os.fspath(DEVELOPMENT_START_CANONICAL_PATH)]
    receipt = _validate_development_start_bytes(data, bundle=bundle, context=context)
    binding = _development_start_execution_binding(
        bundle=bundle,
        context=context,
        receipt=receipt,
        process=process,
    )
    if (
        any(getattr(capability, name) != expected for name, expected in binding.items())
        or capability.semantic_binding_sha256 != _artifact_binding_sha256(binding)
        or capability._receipt is not receipt
        and (
            capability._receipt.payload_sha256 != receipt.payload_sha256
            or capability._receipt.file_sha256 != receipt.file_sha256
        )
    ):
        raise AuthorityError("development start capability binding changed")
    return capability


def require_development_rng_prerequisites(
    development_bundle: object,
    context_reference: object,
    development_start: object,
    *,
    expected_commit: str,
    expected_tree: str,
) -> tuple[
    DevelopmentBundleCapability,
    ValidatedArtifactCapability,
    DevelopmentStartExecutionCapability,
]:
    """Gate initial JIT development authority on the just-created durable receipt."""

    bundle, context = _require_development_bundle_and_context(
        development_bundle,
        context_reference,
        expected_commit=expected_commit,
        expected_tree=expected_tree,
    )
    start = require_development_start_execution_capability(development_start)
    if start._bundle is not bundle or start._context_reference is not context:
        raise AuthorityError("development start does not bind the exact prerequisite instances")
    _require_exact_development_attempt_inventory(
        frozenset({"development-bundle.json", "context-reference.json", "development-start.json"})
    )
    return bundle, context, start


def revalidate_development_rng_prerequisites(
    development_bundle: object,
    context_reference: object,
    development_start: object,
    *,
    expected_commit: str,
    expected_tree: str,
) -> tuple[
    DevelopmentBundleCapability,
    ValidatedArtifactCapability,
    DevelopmentStartExecutionCapability,
]:
    """Revalidate live development authority before or after result publication."""

    bundle, context = _require_development_bundle_and_context(
        development_bundle,
        context_reference,
        expected_commit=expected_commit,
        expected_tree=expected_tree,
    )
    start = require_development_start_execution_capability(development_start)
    if start._bundle is not bundle or start._context_reference is not context:
        raise AuthorityError("development start does not bind the exact prerequisite instances")
    names = {"development-bundle.json", "context-reference.json", "development-start.json"}
    if os.path.lexists(DEVELOPMENT_RESULT_CANONICAL_PATH):
        names.add("development-result.json")
    _require_exact_development_attempt_inventory(frozenset(names))
    return bundle, context, start


def _validate_context_reference_under_test_root(
    test_root: Path,
    *,
    development_bundle: DevelopmentBundleCapability,
    rng_authority: object,
    core_capability: object,
) -> TestOnlyArtifactReceipt:
    receipt = _validate_context_reference_at_path(
        _canonical_absolute_path(test_root) / "context-reference.json",
        development_bundle=development_bundle,
        rng_authority=rng_authority,
        core_capability=core_capability,
        scope="test-only",
    )
    if type(receipt) is not TestOnlyArtifactReceipt:
        raise AssertionError("test context validator returned production authority")
    return receipt


_GRID_CELLS = (
    (
        1,
        "p3-nu_1-lambda_0p10",
        "1",
        "0p10",
        1.0,
        0.10,
        "63ad5a68d79dbad9cfe4443a207eefd96a15d643bf6bbb4914c7a487bb59e5f9",
    ),
    (
        2,
        "p3-nu_1-lambda_0p20",
        "1",
        "0p20",
        1.0,
        0.20,
        "60b4fdb0f0039ac4e0a08c2ebb7e11e36e0aa43b0578362ada8e0bed81e8a43f",
    ),
    (
        3,
        "p3-nu_1-lambda_0p30",
        "1",
        "0p30",
        1.0,
        0.30,
        "16c9d0b71e855df9190bc961b78b0fdc1fd9123628ace1d856ddd7b0faad2786",
    ),
    (
        4,
        "p3-nu_4-lambda_0p10",
        "4",
        "0p10",
        4.0,
        0.10,
        "38c8ffbcc9746bfcfb0a8157012111412f1eecb9c9ed590135763fd548c29613",
    ),
    (
        5,
        "p3-nu_4-lambda_0p20",
        "4",
        "0p20",
        4.0,
        0.20,
        "5723702a4df184cc70dc6195a8031904eb57d0300e555dbf8cfa6345dc9b8347",
    ),
    (
        6,
        "p3-nu_4-lambda_0p30",
        "4",
        "0p30",
        4.0,
        0.30,
        "73668e3f34272a69d8a0ba7265acb3b53b40aeb0c4ea5889bc910d7d878d3558",
    ),
    (
        7,
        "p3-nu_16-lambda_0p10",
        "16",
        "0p10",
        16.0,
        0.10,
        "a0a46034f0d857410c9a70661cce5c73e25a31bd7b51515d0b8adf8ebd2c3201",
    ),
    (
        8,
        "p3-nu_16-lambda_0p20",
        "16",
        "0p20",
        16.0,
        0.20,
        "035c524182fc1c4fca0ae5ccb273d09b4499955784e89b948861312511620d08",
    ),
    (
        9,
        "p3-nu_16-lambda_0p30",
        "16",
        "0p30",
        16.0,
        0.30,
        "d0db62d57756ea1513b910d6c7e2b85b82d0a0b8aa1b76a4a7f9a03ea32e3bfa",
    ),
)
_GRID_CELL_FIELDS = (
    "index",
    "id",
    "nu_token",
    "lambda_token",
    "prototype_prior_pseudocount",
    "lambda_max",
    "operator_instance_sha256",
)
_DEVELOPMENT_RESULT_FIELDS = frozenset(
    {
        "schema",
        "candidate_id",
        "generator_revision",
        "master_plan_file_sha256",
        "synthetic_plan_file_sha256",
        "development_bundle_schema",
        "development_bundle_payload_sha256",
        "development_bundle_file_sha256",
        "context_reference_schema",
        "context_reference_payload_sha256",
        "context_reference_file_sha256",
        "development_start_schema",
        "development_start_payload_sha256",
        "development_start_file_sha256",
        "development_rng_primitive_schema",
        "development_root_seed",
        "participant_count",
        "B4_source_range_stress_participant_indices",
        "grid_cells",
        "participant_metric_rows",
        "invariant_rows",
        "complete_grid_gate_report",
        "selection_status",
        "selected_grid_cell_id",
        "payload_sha256",
    }
)


def _validate_grid_cells(value: Any) -> None:
    if not isinstance(value, (list, tuple)) or len(value) != len(_GRID_CELLS):
        raise ValidationError("development result must contain all nine grid cells")
    for record, expected in zip(value, _GRID_CELLS, strict=True):
        if not isinstance(record, Mapping):
            raise ValidationError("grid cell must be an object")
        _require_exact_keys(record, frozenset(_GRID_CELL_FIELDS), "development grid cell")
        observed = tuple(record[field] for field in _GRID_CELL_FIELDS)
        if observed != expected:
            raise ValidationError("development grid cell differs from the frozen table")


def _validate_development_result_content(value: Mapping[str, Any]) -> None:
    _require_exact_keys(value, _DEVELOPMENT_RESULT_FIELDS, "development result")
    if value.get("schema") != ARTIFACT_SCHEMAS["development_result"]:
        raise ValidationError("development-result schema mismatch")
    if value.get("candidate_id") != CANDIDATE_ID:
        raise ValidationError("development-result candidate mismatch")
    if (
        value["generator_revision"] != "v1_single_insertion_context_trust"
        or value["master_plan_file_sha256"] != FROZEN_FILE_SHA256[MASTER_PLAN_PATH]
        or value["synthetic_plan_file_sha256"] != FROZEN_FILE_SHA256[SYNTHETIC_PLAN_PATH]
        or value["development_start_schema"] != ARTIFACT_SCHEMAS["development_start"]
        or value["development_rng_primitive_schema"]
        != "cfeg.metadata-calibration-efficiency-v3.development-rng-binding.v1"
        or value["development_root_seed"] != 20_260_909
        or value["participant_count"] != 48
    ):
        raise ValidationError("development-result frozen execution constants changed")
    _require_sha256(value["development_start_payload_sha256"], "development start payload")
    _require_sha256(value["development_start_file_sha256"], "development start file")
    _validate_grid_cells(value["grid_cells"])
    stress = value["B4_source_range_stress_participant_indices"]
    if (
        not isinstance(stress, (list, tuple))
        or len(stress) != 5
        or tuple(sorted(set(stress))) != tuple(stress)
        or any(
            not isinstance(item, int) or isinstance(item, bool) or not 0 <= item < 48
            for item in stress
        )
    ):
        raise ValidationError("development B4 stress indices must be five sorted integers")
    if (
        not isinstance(value["participant_metric_rows"], (list, tuple))
        or len(value["participant_metric_rows"]) != 88_992
        or not isinstance(value["invariant_rows"], (list, tuple))
        or len(value["invariant_rows"]) != 90
    ):
        raise ValidationError("development result must carry all 89,082 exact rows")
    if (
        not isinstance(value["complete_grid_gate_report"], (list, tuple))
        or len(value["complete_grid_gate_report"]) != 9
    ):
        raise ValidationError("complete grid gate report must contain all nine cells")
    status = value["selection_status"]
    selected = value["selected_grid_cell_id"]
    ids = {cell[1] for cell in _GRID_CELLS}
    if status == "SELECTED_METHOD_PROPOSED":
        if selected not in ids:
            raise ValidationError("selected development result needs one frozen grid cell ID")
    elif status == "DEVELOPMENT_NO_GO":
        if selected is not None:
            raise ValidationError("DEVELOPMENT_NO_GO must have null selected grid cell")
    else:
        raise ValidationError("unknown development selection status")


def seal_development_result(content: Mapping[str, Any]) -> Mapping[str, Any]:
    value = _seal_schema_artifact(ARTIFACT_SCHEMAS["development_result"], content)
    _validate_development_result_content(value)
    return value


def _validate_development_result_bytes(
    value: Mapping[str, Any],
    *,
    file_bytes: bytes,
    development_bundle: DevelopmentBundleCapability,
    context_reference: ValidatedArtifactCapability,
    development_start: DevelopmentStartExecutionCapability,
    development_rng_authority: object,
    validated_context_reference: object,
    core_capability: object,
) -> tuple[Mapping[str, Any], Mapping[str, Any]]:
    _validate_development_result_content(value)
    parsed = parse_artifact_bytes(
        file_bytes,
        ArtifactSpec(
            schema=ARTIFACT_SCHEMAS["development_result"],
            exact_fields=_DEVELOPMENT_RESULT_FIELDS,
        ),
    )
    if canonical_json_bytes(parsed) != canonical_json_bytes(value):
        raise ValidationError("development-result value differs from supplied bytes")
    core_capability = _require_core_development_result_capability(
        core_capability,
        parsed=parsed,
        development_bundle=development_bundle,
        context_reference=context_reference,
        development_start=development_start,
        development_rng_authority=development_rng_authority,
        validated_context_reference=validated_context_reference,
    )
    development_bundle = require_development_rng_bundle_capability(
        development_bundle,
        expected_commit=development_bundle.clean_commit,
        expected_tree=development_bundle.clean_tree,
    )
    context_reference = require_validated_artifact_capability(
        context_reference,
        expected_schema=ARTIFACT_SCHEMAS["context_reference"],
    )
    development_start = require_development_start_execution_capability(development_start)
    bindings = {
        "master_plan_file_sha256": FROZEN_FILE_SHA256[MASTER_PLAN_PATH],
        "synthetic_plan_file_sha256": FROZEN_FILE_SHA256[SYNTHETIC_PLAN_PATH],
        "development_bundle_schema": development_bundle.schema,
        "development_bundle_payload_sha256": development_bundle.payload_sha256,
        "development_bundle_file_sha256": development_bundle.file_sha256,
        "context_reference_schema": context_reference.schema,
        "context_reference_payload_sha256": context_reference.payload_sha256,
        "context_reference_file_sha256": context_reference.file_sha256,
        "development_start_schema": development_start.receipt_schema,
        "development_start_payload_sha256": development_start.receipt_payload_sha256,
        "development_start_file_sha256": development_start.receipt_file_sha256,
        "development_root_seed": 20_260_909,
    }
    for key, expected in bindings.items():
        if _plain_json(parsed[key]) != _plain_json(expected):
            raise ValidationError(f"development-result binding mismatch: {key}")
    return parsed, MappingProxyType(bindings)


def publish_development_result(
    value: Mapping[str, Any],
    *,
    development_bundle: DevelopmentBundleCapability,
    context_reference: ValidatedArtifactCapability,
    development_start: DevelopmentStartExecutionCapability,
    development_rng_authority: object,
    validated_context_reference: object,
    core_capability: object,
) -> PublishedArtifact:
    _require_active_governed_process(
        recheck_site_packages=True,
        recheck_source=True,
        require_mutation_role=True,
    )
    data = artifact_bytes(value)
    if len(data) > DEVELOPMENT_RESULT_MAXIMUM_BYTES:
        raise ValidationError("development result exceeds its frozen safe size ceiling")
    _validate_development_result_bytes(
        value,
        file_bytes=data,
        development_bundle=development_bundle,
        context_reference=context_reference,
        development_start=development_start,
        development_rng_authority=development_rng_authority,
        validated_context_reference=validated_context_reference,
        core_capability=core_capability,
    )
    require_development_rng_bundle_capability(
        development_bundle,
        expected_commit=development_bundle.clean_commit,
        expected_tree=development_bundle.clean_tree,
    )
    require_development_start_execution_capability(development_start)
    _require_exact_development_attempt_inventory(
        frozenset({"development-bundle.json", "context-reference.json", "development-start.json"})
    )
    return _publish_write_once(
        DEVELOPMENT_RESULT_CANONICAL_PATH.parent,
        DEVELOPMENT_RESULT_CANONICAL_PATH.name,
        data,
        create_root=True,
        _publisher=_PUBLICATION_ISSUER,
    )


def validate_development_result(
    *,
    development_bundle: DevelopmentBundleCapability,
    context_reference: ValidatedArtifactCapability,
    development_start: DevelopmentStartExecutionCapability,
    development_rng_authority: object,
    validated_context_reference: object,
    core_capability: object,
) -> ValidatedArtifactCapability:
    """Issue result authority only from exact canonical published bytes."""

    _require_active_governed_process(
        recheck_site_packages=True,
        recheck_source=True,
    )
    path = DEVELOPMENT_RESULT_CANONICAL_PATH
    data = load_exact_artifact_bytes(
        [path],
        declared_paths=[path],
        maximum_bytes_per_file=DEVELOPMENT_RESULT_MAXIMUM_BYTES,
    )[os.fspath(path)]
    parsed = parse_artifact_bytes(
        data,
        ArtifactSpec(
            schema=ARTIFACT_SCHEMAS["development_result"],
            exact_fields=_DEVELOPMENT_RESULT_FIELDS,
        ),
    )
    parsed, bindings = _validate_development_result_bytes(
        parsed,
        file_bytes=data,
        development_bundle=development_bundle,
        context_reference=context_reference,
        development_start=development_start,
        development_rng_authority=development_rng_authority,
        validated_context_reference=validated_context_reference,
        core_capability=core_capability,
    )
    _require_exact_development_attempt_inventory(
        frozenset(
            {
                "development-bundle.json",
                "context-reference.json",
                "development-start.json",
                "development-result.json",
            }
        )
    )
    return _validate_schema_artifact(
        parsed,
        file_bytes=data,
        schema=ARTIFACT_SCHEMAS["development_result"],
        exact_fields=_DEVELOPMENT_RESULT_FIELDS,
        expected_bindings=bindings,
        canonical_path=path,
        scope="canonical-publication",
        _observer=_CAPABILITY_ISSUER,
    )


CANONICAL_DEVELOPMENT_RESULT_RECOVERY_CAPABILITY_SCHEMA = (
    "cfeg.metadata-calibration-efficiency-v3.canonical-development-result-recovery-capability.v1"
)


@dataclass(frozen=True, slots=True, init=False)
class CanonicalDevelopmentResultRecoveryCapability:
    """A-only durable observation of a proof-gated canonical result."""

    schema: str
    candidate_id: str
    development_bundle_payload_sha256: str
    development_bundle_file_sha256: str
    development_bundle_source_bundle_sha256: str
    numerical_runtime_fingerprint_sha256: str
    context_reference_schema: str
    context_reference_payload_sha256: str
    context_reference_file_sha256: str
    development_start_schema: str
    development_start_payload_sha256: str
    development_start_file_sha256: str
    development_result_schema: str
    development_result_payload_sha256: str
    development_result_file_sha256: str
    clean_commit: str
    clean_tree: str
    semantic_binding_sha256: str
    _bundle: DevelopmentBundleCapability = field(repr=False, compare=False)
    _context_reference: ValidatedArtifactCapability = field(repr=False, compare=False)
    _development_start: ValidatedArtifactCapability = field(repr=False, compare=False)
    _validated_payload: Mapping[str, Any] = field(repr=False, compare=False)
    _snapshot: CleanSourceSnapshot = field(repr=False, compare=False)
    _validation_marker: object = field(repr=False, compare=False)

    def __new__(  # noqa: PYI034 -- Python 3.10 does not provide typing.Self.
        cls,
        *_args: object,
        **_kwargs: object,
    ) -> CanonicalDevelopmentResultRecoveryCapability:
        raise TypeError(
            "CanonicalDevelopmentResultRecoveryCapability is issued only by "
            "canonical A-stage disk observation"
        )


def _canonical_development_result_recovery_binding(
    *,
    bundle: DevelopmentBundleCapability,
    context_reference: ValidatedArtifactCapability,
    development_start: ValidatedArtifactCapability,
    result: Mapping[str, Any],
    result_file_bytes: bytes,
) -> Mapping[str, Any]:
    return MappingProxyType(
        {
            "schema": CANONICAL_DEVELOPMENT_RESULT_RECOVERY_CAPABILITY_SCHEMA,
            "candidate_id": CANDIDATE_ID,
            "development_bundle_payload_sha256": bundle.payload_sha256,
            "development_bundle_file_sha256": bundle.file_sha256,
            "development_bundle_source_bundle_sha256": bundle.source_bundle_sha256,
            "numerical_runtime_fingerprint_sha256": (bundle.numerical_runtime_fingerprint_sha256),
            "context_reference_schema": context_reference.schema,
            "context_reference_payload_sha256": context_reference.payload_sha256,
            "context_reference_file_sha256": context_reference.file_sha256,
            "development_start_schema": development_start.schema,
            "development_start_payload_sha256": development_start.payload_sha256,
            "development_start_file_sha256": development_start.file_sha256,
            "development_result_schema": ARTIFACT_SCHEMAS["development_result"],
            "development_result_payload_sha256": result["payload_sha256"],
            "development_result_file_sha256": file_sha256(result_file_bytes),
            "clean_commit": bundle.clean_commit,
            "clean_tree": bundle.clean_tree,
        }
    )


def _issue_canonical_development_result_recovery_capability(
    *,
    bundle: DevelopmentBundleCapability,
    context_reference: ValidatedArtifactCapability,
    development_start: ValidatedArtifactCapability,
    result: Mapping[str, Any],
    result_file_bytes: bytes,
    snapshot: CleanSourceSnapshot,
    _observer: object,
) -> CanonicalDevelopmentResultRecoveryCapability:
    if _observer is not _CAPABILITY_ISSUER:
        raise AuthorityError("development-result recovery issuance requires canonical observation")
    _require_active_governed_process(
        recheck_site_packages=True,
        recheck_source=True,
    )
    binding = _canonical_development_result_recovery_binding(
        bundle=bundle,
        context_reference=context_reference,
        development_start=development_start,
        result=result,
        result_file_bytes=result_file_bytes,
    )
    capability = object.__new__(CanonicalDevelopmentResultRecoveryCapability)
    for name, value in binding.items():
        object.__setattr__(capability, name, value)
    object.__setattr__(
        capability,
        "semantic_binding_sha256",
        _artifact_binding_sha256(binding),
    )
    object.__setattr__(capability, "_bundle", bundle)
    object.__setattr__(capability, "_context_reference", context_reference)
    object.__setattr__(capability, "_development_start", development_start)
    object.__setattr__(capability, "_validated_payload", _freeze_json(result))
    object.__setattr__(capability, "_snapshot", snapshot)
    object.__setattr__(capability, "_validation_marker", _CAPABILITY_ISSUER)
    return capability


def _load_canonical_development_result_for_a_recovery(
    *,
    development_bundle: DevelopmentBundleCapability,
    context_reference: ValidatedArtifactCapability,
) -> tuple[
    DevelopmentBundleCapability,
    ValidatedArtifactCapability,
    ValidatedArtifactCapability,
    CleanSourceSnapshot,
    Mapping[str, Any],
    bytes,
]:
    """Observe the exact A source, canonical context, and canonical result bytes."""

    bundle = require_development_recovery_bundle_capability(
        development_bundle,
        expected_commit=development_bundle.clean_commit,
        expected_tree=development_bundle.clean_tree,
    )
    context = require_validated_artifact_capability(
        context_reference,
        expected_schema=ARTIFACT_SCHEMAS["context_reference"],
    )
    context_data = load_exact_artifact_bytes(
        [CONTEXT_REFERENCE_CANONICAL_PATH],
        declared_paths=[CONTEXT_REFERENCE_CANONICAL_PATH],
    )[os.fspath(CONTEXT_REFERENCE_CANONICAL_PATH)]
    retired_v1, _ = observe_retired_development_attempts()
    if context_data != retired_v1_context_reference_bytes(retired_v1):
        raise AuthorityError("A-stage development recovery context differs from retired V1 bytes")
    context_payload = parse_artifact_bytes(
        context_data,
        ArtifactSpec(
            schema=ARTIFACT_SCHEMAS["context_reference"],
            exact_fields=_CONTEXT_REFERENCE_FIELDS,
        ),
    )
    if (
        context.payload_sha256 != context_payload["payload_sha256"]
        or context.file_sha256 != file_sha256(context_data)
        or canonical_json_bytes(context._validated_payload) != canonical_json_bytes(context_payload)
    ):
        raise AuthorityError("A-stage development recovery context differs from canonical bytes")
    start_data = load_exact_artifact_bytes(
        [DEVELOPMENT_START_CANONICAL_PATH],
        declared_paths=[DEVELOPMENT_START_CANONICAL_PATH],
    )[os.fspath(DEVELOPMENT_START_CANONICAL_PATH)]
    development_start = _validate_development_start_bytes(
        start_data,
        bundle=bundle,
        context=context,
    )
    result_data = load_exact_artifact_bytes(
        [DEVELOPMENT_RESULT_CANONICAL_PATH],
        declared_paths=[DEVELOPMENT_RESULT_CANONICAL_PATH],
        maximum_bytes_per_file=DEVELOPMENT_RESULT_MAXIMUM_BYTES,
    )[os.fspath(DEVELOPMENT_RESULT_CANONICAL_PATH)]
    result = parse_artifact_bytes(
        result_data,
        ArtifactSpec(
            schema=ARTIFACT_SCHEMAS["development_result"],
            exact_fields=_DEVELOPMENT_RESULT_FIELDS,
        ),
    )
    _validate_development_result_content(result)
    bindings = {
        "master_plan_file_sha256": FROZEN_FILE_SHA256[MASTER_PLAN_PATH],
        "synthetic_plan_file_sha256": FROZEN_FILE_SHA256[SYNTHETIC_PLAN_PATH],
        "development_bundle_schema": bundle.schema,
        "development_bundle_payload_sha256": bundle.payload_sha256,
        "development_bundle_file_sha256": bundle.file_sha256,
        "context_reference_schema": context.schema,
        "context_reference_payload_sha256": context.payload_sha256,
        "context_reference_file_sha256": context.file_sha256,
        "development_start_schema": development_start.schema,
        "development_start_payload_sha256": development_start.payload_sha256,
        "development_start_file_sha256": development_start.file_sha256,
        "development_root_seed": 20_260_909,
    }
    if any(_plain_json(result[key]) != _plain_json(value) for key, value in bindings.items()):
        raise AuthorityError("A-stage development result durable bindings differ")
    snapshot = capture_clean_source_snapshot(V3_SOURCE_REPOSITORY)
    if (
        snapshot.identity.commit != bundle.clean_commit
        or snapshot.identity.tree != bundle.clean_tree
        or snapshot.source_bundle_sha256 != bundle.source_bundle_sha256
        or snapshot.tracked_files != bundle.tracked_source_files
    ):
        raise AuthorityError("development-result recovery requires exact bundle A")
    _require_loaded_governance_source_at_snapshot(snapshot)
    _require_exact_development_attempt_inventory(
        frozenset(
            {
                "development-bundle.json",
                "context-reference.json",
                "development-start.json",
                "development-result.json",
            }
        )
    )
    return bundle, context, development_start, snapshot, result, result_data


def observe_canonical_development_result_for_a_recovery(
    *,
    development_bundle: DevelopmentBundleCapability,
    context_reference: ValidatedArtifactCapability,
) -> CanonicalDevelopmentResultRecoveryCapability:
    """Issue audit-only A recovery authority from immutable canonical bytes."""

    _require_active_governed_process(
        recheck_site_packages=True,
        recheck_source=True,
    )
    bundle, context, development_start, snapshot, result, result_data = (
        _load_canonical_development_result_for_a_recovery(
            development_bundle=development_bundle,
            context_reference=context_reference,
        )
    )
    return _issue_canonical_development_result_recovery_capability(
        bundle=bundle,
        context_reference=context,
        development_start=development_start,
        result=result,
        result_file_bytes=result_data,
        snapshot=snapshot,
        _observer=_CAPABILITY_ISSUER,
    )


def _require_canonical_development_result_recovery_seal(
    value: object,
) -> CanonicalDevelopmentResultRecoveryCapability:
    if type(value) is not CanonicalDevelopmentResultRecoveryCapability:
        raise AuthorityError("exact CanonicalDevelopmentResultRecoveryCapability required")
    capability = value
    if capability._validation_marker is not _CAPABILITY_ISSUER:
        raise AuthorityError("canonical development-result recovery marker is invalid")
    bundle = require_development_bundle_capability(
        capability._bundle,
        expected_commit=capability.clean_commit,
        expected_tree=capability.clean_tree,
    )
    context = require_validated_artifact_capability(
        capability._context_reference,
        expected_schema=ARTIFACT_SCHEMAS["context_reference"],
    )
    development_start = require_validated_artifact_capability(
        capability._development_start,
        expected_schema=ARTIFACT_SCHEMAS["development_start"],
    )
    binding = _canonical_development_result_recovery_binding(
        bundle=bundle,
        context_reference=context,
        development_start=development_start,
        result=capability._validated_payload,
        result_file_bytes=artifact_bytes(capability._validated_payload),
    )
    if any(
        getattr(capability, name) != expected for name, expected in binding.items()
    ) or capability.semantic_binding_sha256 != _artifact_binding_sha256(binding):
        raise AuthorityError("canonical development-result recovery seal mismatch")
    return capability


def require_canonical_development_result_recovery_capability(
    value: object,
) -> CanonicalDevelopmentResultRecoveryCapability:
    """Revalidate an A recovery capability against source, runtime, and disk."""

    capability = _require_canonical_development_result_recovery_seal(value)
    bundle, context, development_start, snapshot, result, result_data = (
        _load_canonical_development_result_for_a_recovery(
            development_bundle=capability._bundle,
            context_reference=capability._context_reference,
        )
    )
    binding = _canonical_development_result_recovery_binding(
        bundle=bundle,
        context_reference=context,
        development_start=development_start,
        result=result,
        result_file_bytes=result_data,
    )
    if (
        snapshot != capability._snapshot
        or canonical_json_bytes(result) != canonical_json_bytes(capability._validated_payload)
        or any(getattr(capability, name) != expected for name, expected in binding.items())
        or capability.semantic_binding_sha256 != _artifact_binding_sha256(binding)
    ):
        raise AuthorityError("canonical development-result recovery no longer matches exact disk")
    return capability


def development_result_recovery_source_file_sha256(
    value: object,
    *,
    repository_relative_path: str,
) -> str:
    """Return one exact bundle-A source hash from a sealed recovery receipt."""

    capability = _require_canonical_development_result_recovery_seal(value)
    pure = PurePosixPath(repository_relative_path)
    if (
        pure.is_absolute()
        or pure.as_posix() != repository_relative_path
        or any(part in ("", ".", "..") for part in pure.parts)
    ):
        raise ValidationError("development recovery source path is not canonical")
    matches = tuple(
        item.file_sha256
        for item in capability._bundle.tracked_source_files
        if item.path == repository_relative_path and item.kind == "regular_file"
    )
    if len(matches) != 1:
        raise AuthorityError("development recovery bundle lacks one exact regular source blob")
    return matches[0]


def _development_result_audit_hashes(
    value: Mapping[str, Any],
) -> Mapping[str, str]:
    reports = value["complete_grid_gate_report"]
    return MappingProxyType(
        {
            "participant_metric_rows_sha256": hashlib.sha256(
                canonical_json_bytes(
                    {
                        "schema": (
                            "cfeg.metadata-calibration-efficiency-v3.participant-metric-row-set.v1"
                        ),
                        "rows": value["participant_metric_rows"],
                    }
                )
            ).hexdigest(),
            "invariant_rows_sha256": hashlib.sha256(
                canonical_json_bytes(
                    {
                        "schema": ("cfeg.metadata-calibration-efficiency-v3.invariant-row-set.v1"),
                        "rows": value["invariant_rows"],
                    }
                )
            ).hexdigest(),
            "complete_grid_gate_report_sha256": hashlib.sha256(
                canonical_json_bytes(
                    {
                        "schema": (
                            "cfeg.metadata-calibration-efficiency-v3.complete-grid-gate-report.v1"
                        ),
                        "reports": reports,
                    }
                )
            ).hexdigest(),
            "uniform_block_weight_sensitivity_sha256": hashlib.sha256(
                canonical_json_bytes(
                    {
                        "schema": (
                            "cfeg.metadata-calibration-efficiency-v3."
                            "complete-uniform-block-weight-sensitivity.v1"
                        ),
                        "reports": [
                            report["uniform_block_weight_sensitivity"] for report in reports
                        ],
                    }
                )
            ).hexdigest(),
            "resampling_sensitivity_sha256": hashlib.sha256(
                canonical_json_bytes(
                    {
                        "schema": (
                            "cfeg.metadata-calibration-efficiency-v3."
                            "complete-resampling-sensitivity.v1"
                        ),
                        "reports": [report["resampling_sensitivity"] for report in reports],
                    }
                )
            ).hexdigest(),
            "sensitivity_participant_vectors_sha256": hashlib.sha256(
                canonical_json_bytes(
                    {
                        "schema": (
                            "cfeg.metadata-calibration-efficiency-v3."
                            "complete-sensitivity-participant-vectors.v1"
                        ),
                        "reports": [
                            report["sensitivity_participant_vectors"] for report in reports
                        ],
                    }
                )
            ).hexdigest(),
        }
    )


def _require_core_audited_development_result(
    value: object,
    *,
    recovery: CanonicalDevelopmentResultRecoveryCapability,
) -> object:
    try:
        from cfeg.analysis.metadata_calibration_v3_synthetic import (
            require_audited_development_result,
        )
    except ImportError as exc:
        raise AuthorityError("V3 core A recovery proof checker is unavailable") from exc
    capability = require_audited_development_result(
        value,
        expected_payload_sha256=recovery.development_result_payload_sha256,
    )
    expected = {
        "schema": (
            "cfeg.metadata-calibration-efficiency-v3.audited-development-result-capability.v1"
        ),
        "candidate_id": CANDIDATE_ID,
        "payload_schema": recovery.development_result_schema,
        "payload_sha256": recovery.development_result_payload_sha256,
        "development_result_file_sha256": recovery.development_result_file_sha256,
        "development_bundle_payload_sha256": (recovery.development_bundle_payload_sha256),
        "development_bundle_file_sha256": recovery.development_bundle_file_sha256,
        "development_bundle_source_bundle_sha256": (
            recovery.development_bundle_source_bundle_sha256
        ),
        "numerical_runtime_fingerprint_sha256": (recovery.numerical_runtime_fingerprint_sha256),
        "context_reference_schema": recovery.context_reference_schema,
        "context_reference_payload_sha256": recovery.context_reference_payload_sha256,
        "context_reference_file_sha256": recovery.context_reference_file_sha256,
        "clean_commit": recovery.clean_commit,
        "clean_tree": recovery.clean_tree,
        **_development_result_audit_hashes(recovery._validated_payload),
        "selection_status": recovery._validated_payload["selection_status"],
        "selected_grid_cell_id": recovery._validated_payload["selected_grid_cell_id"],
    }
    if any(
        getattr(capability, name) != expected_value for name, expected_value in expected.items()
    ):
        raise AuthorityError("core A recovery proof differs from canonical evidence")
    _require_sha256(
        capability.semantic_binding_sha256,
        "audited development result semantic binding",
    )
    return capability


def recover_canonical_development_result_at_bundle_source(
    *,
    development_bundle: DevelopmentBundleCapability,
    context_reference: ValidatedArtifactCapability,
) -> tuple[ValidatedArtifactCapability, object]:
    """Recover canonical A result authority plus a sealed core audit proof."""

    _require_active_governed_process(
        recheck_site_packages=True,
        recheck_source=True,
    )
    recovery = observe_canonical_development_result_for_a_recovery(
        development_bundle=development_bundle,
        context_reference=context_reference,
    )
    recovery = require_canonical_development_result_recovery_capability(recovery)
    try:
        from cfeg.analysis.metadata_calibration_v3_synthetic import (
            validate_development_result_for_a_recovery,
        )
    except ImportError as exc:
        raise AuthorityError("V3 core A recovery validator is unavailable") from exc
    audited = validate_development_result_for_a_recovery(
        recovery._validated_payload,
        context_reference=recovery._context_reference._validated_payload,
        recovery_capability=recovery,
    )
    recovery = require_canonical_development_result_recovery_capability(recovery)
    audited = _require_core_audited_development_result(audited, recovery=recovery)
    bindings = {
        "master_plan_file_sha256": FROZEN_FILE_SHA256[MASTER_PLAN_PATH],
        "synthetic_plan_file_sha256": FROZEN_FILE_SHA256[SYNTHETIC_PLAN_PATH],
        "development_bundle_schema": recovery._bundle.schema,
        "development_bundle_payload_sha256": recovery.development_bundle_payload_sha256,
        "development_bundle_file_sha256": recovery.development_bundle_file_sha256,
        "context_reference_schema": recovery.context_reference_schema,
        "context_reference_payload_sha256": recovery.context_reference_payload_sha256,
        "context_reference_file_sha256": recovery.context_reference_file_sha256,
        "development_root_seed": 20_260_909,
    }
    artifact = _validate_schema_artifact(
        recovery._validated_payload,
        file_bytes=artifact_bytes(recovery._validated_payload),
        schema=ARTIFACT_SCHEMAS["development_result"],
        exact_fields=_DEVELOPMENT_RESULT_FIELDS,
        expected_bindings=bindings,
        canonical_path=DEVELOPMENT_RESULT_CANONICAL_PATH,
        scope="canonical-publication",
        _observer=_CAPABILITY_ISSUER,
    )
    return artifact, audited


def reopen_development_result_at_bundle_source(
    *,
    development_bundle: DevelopmentBundleCapability,
    context_reference: ValidatedArtifactCapability,
) -> ValidatedArtifactCapability:
    """Recover a canonical development result at clean A, including NO_GO."""

    _require_active_governed_process(
        recheck_site_packages=True,
        recheck_source=True,
    )
    artifact, _ = recover_canonical_development_result_at_bundle_source(
        development_bundle=development_bundle,
        context_reference=context_reference,
    )
    return artifact


def build_selected_method_freeze_from_canonical_development_result(
    *,
    development_result: ValidatedArtifactCapability,
    audited_development_result: object,
) -> Mapping[str, Any]:
    """Rebuild PASS selection bytes only from the sealed canonical A audit."""

    _require_active_governed_process(
        recheck_site_packages=True,
        recheck_source=True,
    )
    result = require_validated_artifact_capability(
        development_result,
        expected_schema=ARTIFACT_SCHEMAS["development_result"],
    )
    try:
        from cfeg.analysis.metadata_calibration_v3_synthetic import (
            build_selected_method_freeze_payload_for_a_recovery,
            require_audited_development_result,
        )
    except ImportError as exc:
        raise AuthorityError("V3 core A recovery selector is unavailable") from exc
    audited = require_audited_development_result(
        audited_development_result,
        expected_payload_sha256=result.payload_sha256,
    )
    if (
        audited.payload_schema != result.schema
        or audited.payload_sha256 != result.payload_sha256
        or audited.development_result_file_sha256 != result.file_sha256
        or audited.selection_status != "SELECTED_METHOD_PROPOSED"
        or audited.selected_grid_cell_id != result._validated_payload["selected_grid_cell_id"]
    ):
        raise AuthorityError("selected recovery proof differs from result authority")
    selected = build_selected_method_freeze_payload_for_a_recovery(
        result._validated_payload,
        audited_development_result=audited,
    )
    selected = parse_artifact_bytes(
        artifact_bytes(selected),
        ArtifactSpec(
            schema=ARTIFACT_SCHEMAS["selected_method_freeze"],
            exact_fields=_SELECTED_FREEZE_FIELDS,
        ),
    )
    _validate_selected_method_freeze_content(selected)
    gate_report_sha256 = hashlib.sha256(
        canonical_json_bytes(
            {
                "schema": ("cfeg.metadata-calibration-efficiency-v3.complete-grid-gate-report.v1"),
                "reports": result._validated_payload["complete_grid_gate_report"],
            }
        )
    ).hexdigest()
    bindings = {
        "master_plan_file_sha256": FROZEN_FILE_SHA256[MASTER_PLAN_PATH],
        "synthetic_plan_file_sha256": FROZEN_FILE_SHA256[SYNTHETIC_PLAN_PATH],
        "development_bundle_schema": DEVELOPMENT_BUNDLE_SCHEMA,
        "development_bundle_payload_sha256": audited.development_bundle_payload_sha256,
        "development_bundle_file_sha256": audited.development_bundle_file_sha256,
        "development_result_schema": result.schema,
        "development_result_payload_sha256": result.payload_sha256,
        "development_result_file_sha256": result.file_sha256,
        "complete_grid_gate_report_sha256": gate_report_sha256,
        "clean_commit": audited.clean_commit,
        "clean_tree": audited.clean_tree,
    }
    if selected["selected_grid_cell_id"] != audited.selected_grid_cell_id or any(
        _plain_json(selected[name]) != _plain_json(expected) for name, expected in bindings.items()
    ):
        raise AuthorityError("A recovery selector differs from the canonical development result")
    return selected


_SELECTED_FREEZE_FIELDS = frozenset(
    {
        "schema",
        "candidate_id",
        "scientific_candidate_id",
        "selected_grid_cell_index",
        "selected_grid_cell_id",
        "selected_prototype_prior_pseudocount",
        "selected_lambda_max",
        "selected_operator_instance_sha256",
        "master_plan_file_sha256",
        "synthetic_plan_file_sha256",
        "development_bundle_schema",
        "development_bundle_payload_sha256",
        "development_bundle_file_sha256",
        "development_result_schema",
        "development_result_payload_sha256",
        "development_result_file_sha256",
        "complete_grid_gate_report_sha256",
        "selection_rank_tuple",
        "clean_commit",
        "clean_tree",
        "payload_sha256",
    }
)
_SELECTION_RANK_FIELDS = frozenset(
    {"minimum_mandatory_observed_gain", "minimum_corresponding_one_sided_LCB"}
)


def _selected_cell_from_payload(value: Mapping[str, Any]) -> tuple[Any, ...]:
    observed = (
        value["selected_grid_cell_index"],
        value["selected_grid_cell_id"],
        None,
        None,
        value["selected_prototype_prior_pseudocount"],
        value["selected_lambda_max"],
        value["selected_operator_instance_sha256"],
    )
    for cell in _GRID_CELLS:
        comparable = (cell[0], cell[1], None, None, cell[4], cell[5], cell[6])
        if observed == comparable:
            return cell
    raise ValidationError("selected-method freeze cell differs from all frozen grid cells")


def _validate_selected_method_freeze_content(value: Mapping[str, Any]) -> None:
    _require_exact_keys(value, _SELECTED_FREEZE_FIELDS, "selected-method freeze")
    if value.get("schema") != ARTIFACT_SCHEMAS["selected_method_freeze"]:
        raise ValidationError("selected-method-freeze schema mismatch")
    if value.get("candidate_id") != CANDIDATE_ID:
        raise ValidationError("selected-method-freeze candidate mismatch")
    cell = _selected_cell_from_payload(value)
    expected_scientific_id = f"{CANDIDATE_ID}-p3-nu_{cell[2]}-lambda_{cell[3]}"
    if value["scientific_candidate_id"] != expected_scientific_id:
        raise ValidationError("scientific candidate ID differs from the selected frozen cell")
    if (
        value["master_plan_file_sha256"] != FROZEN_FILE_SHA256[MASTER_PLAN_PATH]
        or value["synthetic_plan_file_sha256"] != FROZEN_FILE_SHA256[SYNTHETIC_PLAN_PATH]
    ):
        raise ValidationError("selected-method freeze plan hashes changed")
    for name in (
        "development_bundle_payload_sha256",
        "development_bundle_file_sha256",
        "development_result_payload_sha256",
        "development_result_file_sha256",
        "complete_grid_gate_report_sha256",
    ):
        _require_sha256(value[name], f"selected-method freeze {name}")
    rank = value["selection_rank_tuple"]
    if not isinstance(rank, Mapping):
        raise ValidationError("selection rank tuple must be an object")
    _require_exact_keys(rank, _SELECTION_RANK_FIELDS, "selection rank tuple")
    if any(
        not isinstance(rank[name], float) or not math.isfinite(rank[name])
        for name in _SELECTION_RANK_FIELDS
    ):
        raise ValidationError("selection rank tuple components must be finite floats")
    _require_git_object(value["clean_commit"], "selected-method clean commit")
    _require_git_object(value["clean_tree"], "selected-method clean tree")


def seal_selected_method_freeze(content: Mapping[str, Any]) -> Mapping[str, Any]:
    value = _seal_schema_artifact(ARTIFACT_SCHEMAS["selected_method_freeze"], content)
    _validate_selected_method_freeze_content(value)
    return value


def _require_exact_selected_inventory_delta(
    before: Sequence[TrackedSourceFile],
    after: Sequence[TrackedSourceFile],
) -> None:
    old = {item.path: item for item in before}
    new = {item.path: item for item in after}
    selected_path = SELECTED_METHOD_FREEZE_REPOSITORY_PATH.as_posix()
    if selected_path in old:
        raise AuthorityError("selected-freeze path already existed in the development bundle")
    selected_record = new.pop(selected_path, None)
    if selected_record is None or selected_record.kind != "regular_file" or new != old:
        raise AuthorityError(
            "selected snapshot may differ from the bundle only by one regular selected-freeze blob"
        )


def _require_selected_freeze_only_source_delta(
    snapshot: CleanSourceSnapshot,
    development_bundle: DevelopmentBundleCapability,
) -> None:
    """Permit only the deterministic selected-freeze blob after bundle commit A."""

    snapshot = _require_clean_source_snapshot(snapshot)
    bundle = require_development_bundle_capability(
        development_bundle,
        expected_commit=development_bundle.clean_commit,
        expected_tree=development_bundle.clean_tree,
    )
    _require_exact_selected_inventory_delta(
        bundle.tracked_source_files,
        snapshot.tracked_files,
    )
    _run_git(
        V3_SOURCE_REPOSITORY,
        ("merge-base", "--is-ancestor", bundle.clean_commit, snapshot.identity.commit),
    )


def _validate_selected_method_freeze_bytes(
    value: Mapping[str, Any],
    *,
    file_bytes: bytes,
    development_bundle: DevelopmentBundleCapability,
    development_result: ValidatedArtifactCapability,
    core_capability: object,
) -> tuple[Mapping[str, Any], Mapping[str, Any]]:
    _validate_selected_method_freeze_content(value)
    parsed = parse_artifact_bytes(
        file_bytes,
        ArtifactSpec(
            schema=ARTIFACT_SCHEMAS["selected_method_freeze"],
            exact_fields=_SELECTED_FREEZE_FIELDS,
        ),
    )
    if canonical_json_bytes(parsed) != canonical_json_bytes(value):
        raise ValidationError("selected-method value differs from supplied bytes")
    _require_core_selected_method_proposal(core_capability, parsed=parsed)
    development_bundle = require_development_bundle_capability(
        development_bundle,
        expected_commit=development_bundle.clean_commit,
        expected_tree=development_bundle.clean_tree,
    )
    development_result = require_validated_artifact_capability(
        development_result,
        expected_schema=ARTIFACT_SCHEMAS["development_result"],
    )
    result_payload = development_result._validated_payload
    if result_payload["selection_status"] != "SELECTED_METHOD_PROPOSED":
        raise AuthorityError("selected-method freeze cannot follow DEVELOPMENT_NO_GO")
    gate_report_sha256 = hashlib.sha256(
        canonical_json_bytes(
            {
                "schema": ("cfeg.metadata-calibration-efficiency-v3.complete-grid-gate-report.v1"),
                "reports": result_payload["complete_grid_gate_report"],
            }
        )
    ).hexdigest()
    bindings = {
        "master_plan_file_sha256": FROZEN_FILE_SHA256[MASTER_PLAN_PATH],
        "synthetic_plan_file_sha256": FROZEN_FILE_SHA256[SYNTHETIC_PLAN_PATH],
        "development_bundle_schema": development_bundle.schema,
        "development_bundle_payload_sha256": development_bundle.payload_sha256,
        "development_bundle_file_sha256": development_bundle.file_sha256,
        "development_result_schema": development_result.schema,
        "development_result_payload_sha256": development_result.payload_sha256,
        "development_result_file_sha256": development_result.file_sha256,
        "complete_grid_gate_report_sha256": gate_report_sha256,
        "clean_commit": development_bundle.clean_commit,
        "clean_tree": development_bundle.clean_tree,
    }
    if value["selected_grid_cell_id"] != result_payload["selected_grid_cell_id"]:
        raise ValidationError("selected freeze cell differs from development selector output")
    for key, expected in bindings.items():
        if _plain_json(parsed[key]) != _plain_json(expected):
            raise ValidationError(f"selected-method binding mismatch: {key}")
    return parsed, MappingProxyType(bindings)


def validate_selected_method_freeze(
    *,
    snapshot: CleanSourceSnapshot,
    development_bundle: DevelopmentBundleCapability,
    development_result: ValidatedArtifactCapability,
    recovery_capability: BundleSelectedDeltaRecoveryCapability,
) -> ValidatedArtifactCapability:
    """Validate selected B without reusing an A-only ephemeral RNG proposal."""

    _require_active_governed_process(
        recheck_site_packages=True,
        recheck_source=True,
    )
    snapshot = _require_clean_source_snapshot(snapshot)
    recovery = require_bundle_selected_delta_recovery_capability(recovery_capability)
    if (
        recovery.selected_commit != snapshot.identity.commit
        or recovery.selected_tree != snapshot.identity.tree
        or recovery.selected_source_bundle_sha256 != snapshot.source_bundle_sha256
    ):
        raise AuthorityError("selected validator snapshot differs from recovery B")
    return reopen_selected_method_freeze(
        development_bundle=development_bundle,
        development_result=development_result,
        recovery_capability=recovery,
    )


BUNDLE_SELECTED_DELTA_RECOVERY_CAPABILITY_SCHEMA = (
    "cfeg.metadata-calibration-efficiency-v3.bundle-selected-delta-recovery-capability.v1"
)


@dataclass(frozen=True, slots=True, init=False)
class BundleSelectedDeltaRecoveryCapability:
    """Audit-only proof that B is exactly A plus the selected-freeze blob."""

    schema: str
    candidate_id: str
    development_bundle_payload_sha256: str
    development_bundle_file_sha256: str
    development_bundle_source_bundle_sha256: str
    numerical_runtime_fingerprint_sha256: str
    bundle_commit: str
    bundle_tree: str
    selected_method_freeze_payload_sha256: str
    selected_method_freeze_file_sha256: str
    selected_commit: str
    selected_tree: str
    selected_source_bundle_sha256: str
    semantic_binding_sha256: str
    _bundle: DevelopmentBundleCapability = field(repr=False, compare=False)
    _selected_payload: Mapping[str, Any] = field(repr=False, compare=False)
    _snapshot: CleanSourceSnapshot = field(repr=False, compare=False)
    _validation_marker: object = field(repr=False, compare=False)

    def __new__(  # noqa: PYI034 -- Python 3.10 does not provide typing.Self.
        cls,
        *_args: object,
        **_kwargs: object,
    ) -> BundleSelectedDeltaRecoveryCapability:
        raise TypeError(
            "BundleSelectedDeltaRecoveryCapability is issued only by exact Git-delta validation"
        )


def _bundle_selected_delta_binding(
    bundle: DevelopmentBundleCapability,
    selected_payload: Mapping[str, Any],
    selected_file_sha256: str,
    snapshot: CleanSourceSnapshot,
) -> Mapping[str, Any]:
    return MappingProxyType(
        {
            "schema": BUNDLE_SELECTED_DELTA_RECOVERY_CAPABILITY_SCHEMA,
            "candidate_id": CANDIDATE_ID,
            "development_bundle_payload_sha256": bundle.payload_sha256,
            "development_bundle_file_sha256": bundle.file_sha256,
            "development_bundle_source_bundle_sha256": bundle.source_bundle_sha256,
            "numerical_runtime_fingerprint_sha256": (bundle.numerical_runtime_fingerprint_sha256),
            "bundle_commit": bundle.clean_commit,
            "bundle_tree": bundle.clean_tree,
            "selected_method_freeze_payload_sha256": selected_payload["payload_sha256"],
            "selected_method_freeze_file_sha256": selected_file_sha256,
            "selected_commit": snapshot.identity.commit,
            "selected_tree": snapshot.identity.tree,
            "selected_source_bundle_sha256": snapshot.source_bundle_sha256,
        }
    )


def _issue_bundle_selected_delta_recovery_capability(
    *,
    bundle: DevelopmentBundleCapability,
    selected_payload: Mapping[str, Any],
    selected_file_sha256: str,
    snapshot: CleanSourceSnapshot,
    _observer: object,
) -> BundleSelectedDeltaRecoveryCapability:
    if _observer is not _CAPABILITY_ISSUER:
        raise AuthorityError("selected-delta recovery issuance requires canonical observation")
    _require_active_governed_process(
        recheck_site_packages=True,
        recheck_source=True,
    )
    binding = _bundle_selected_delta_binding(
        bundle,
        selected_payload,
        selected_file_sha256,
        snapshot,
    )
    value = object.__new__(BundleSelectedDeltaRecoveryCapability)
    for name, item in binding.items():
        object.__setattr__(value, name, item)
    object.__setattr__(value, "semantic_binding_sha256", _artifact_binding_sha256(binding))
    object.__setattr__(value, "_bundle", bundle)
    object.__setattr__(value, "_selected_payload", _freeze_json(selected_payload))
    object.__setattr__(value, "_snapshot", snapshot)
    object.__setattr__(value, "_validation_marker", _CAPABILITY_ISSUER)
    return value


def validate_bundle_selected_delta_recovery(
    *,
    development_bundle: DevelopmentBundleCapability,
    snapshot: CleanSourceSnapshot,
) -> BundleSelectedDeltaRecoveryCapability:
    """Issue audit-only recovery proof from exact current B and canonical selected bytes."""

    _require_active_governed_process(
        recheck_site_packages=True,
        recheck_source=True,
    )
    _require_frozen_numerical_executor_environment()
    bundle = require_current_numerical_runtime_fingerprint(development_bundle)
    snapshot = _require_clean_source_snapshot(snapshot)
    current = capture_clean_source_snapshot(V3_SOURCE_REPOSITORY)
    if not _same_source_snapshot(current, snapshot):
        raise AuthorityError("selected-delta recovery requires the exact current clean B")
    _require_loaded_governance_source_at_snapshot(current)
    _require_selected_freeze_only_source_delta(snapshot, bundle)
    selected_data = _load_committed_source_blob(
        V3_SOURCE_REPOSITORY,
        SELECTED_METHOD_FREEZE_REPOSITORY_PATH,
        snapshot=snapshot,
        production=True,
    )
    selected_payload = parse_artifact_bytes(
        selected_data,
        ArtifactSpec(
            schema=ARTIFACT_SCHEMAS["selected_method_freeze"],
            exact_fields=_SELECTED_FREEZE_FIELDS,
        ),
    )
    _validate_selected_method_freeze_content(selected_payload)
    if (
        selected_payload["development_bundle_payload_sha256"] != bundle.payload_sha256
        or selected_payload["development_bundle_file_sha256"] != bundle.file_sha256
        or selected_payload["clean_commit"] != bundle.clean_commit
        or selected_payload["clean_tree"] != bundle.clean_tree
    ):
        raise AuthorityError("selected-delta recovery chain bindings differ")
    return _issue_bundle_selected_delta_recovery_capability(
        bundle=bundle,
        selected_payload=selected_payload,
        selected_file_sha256=file_sha256(selected_data),
        snapshot=snapshot,
        _observer=_CAPABILITY_ISSUER,
    )


def require_bundle_selected_delta_recovery_capability(
    value: object,
    *,
    expected_selected_commit: str | None = None,
    expected_selected_tree: str | None = None,
) -> BundleSelectedDeltaRecoveryCapability:
    _require_frozen_numerical_executor_environment()
    if type(value) is not BundleSelectedDeltaRecoveryCapability:
        raise AuthorityError("exact BundleSelectedDeltaRecoveryCapability required")
    capability = value
    if capability._validation_marker is not _CAPABILITY_ISSUER:
        raise AuthorityError("selected-delta recovery marker is invalid")
    bundle = require_current_numerical_runtime_fingerprint(capability._bundle)
    snapshot = _require_clean_source_snapshot(capability._snapshot)
    current = capture_clean_source_snapshot(V3_SOURCE_REPOSITORY)
    if not _same_source_snapshot(current, snapshot):
        raise AuthorityError("selected-delta recovery source is no longer exact current B")
    _require_loaded_governance_source_at_snapshot(current)
    _require_selected_freeze_only_source_delta(snapshot, bundle)
    selected_data = _load_committed_source_blob(
        V3_SOURCE_REPOSITORY,
        SELECTED_METHOD_FREEZE_REPOSITORY_PATH,
        snapshot=snapshot,
        production=True,
    )
    selected_payload = parse_artifact_bytes(
        selected_data,
        ArtifactSpec(
            schema=ARTIFACT_SCHEMAS["selected_method_freeze"],
            exact_fields=_SELECTED_FREEZE_FIELDS,
        ),
    )
    if canonical_json_bytes(selected_payload) != canonical_json_bytes(capability._selected_payload):
        raise AuthorityError("selected-delta recovery selected payload changed")
    binding = _bundle_selected_delta_binding(
        bundle,
        selected_payload,
        file_sha256(selected_data),
        snapshot,
    )
    if any(getattr(capability, name) != expected for name, expected in binding.items()):
        raise AuthorityError("selected-delta recovery fields differ from sealed authorities")
    if capability.semantic_binding_sha256 != _artifact_binding_sha256(binding):
        raise AuthorityError("selected-delta recovery semantic binding mismatch")
    if expected_selected_commit is not None:
        _require_git_object(expected_selected_commit, "expected selected commit")
        if capability.selected_commit != expected_selected_commit:
            raise AuthorityError("selected-delta recovery commit mismatch")
    if expected_selected_tree is not None:
        _require_git_object(expected_selected_tree, "expected selected tree")
        if capability.selected_tree != expected_selected_tree:
            raise AuthorityError("selected-delta recovery tree mismatch")
    return capability


def recovery_bundle_source_file_sha256(
    value: object,
    *,
    repository_relative_path: str,
) -> str:
    """Return one exact A blob hash after full B-recovery revalidation."""

    capability = require_bundle_selected_delta_recovery_capability(value)
    pure = PurePosixPath(repository_relative_path)
    if (
        pure.is_absolute()
        or pure.as_posix() != repository_relative_path
        or any(part in ("", ".", "..") for part in pure.parts)
    ):
        raise ValidationError("recovery source lookup path is not canonical")
    matches = tuple(
        item.file_sha256
        for item in capability._bundle.tracked_source_files
        if item.path == repository_relative_path and item.kind == "regular_file"
    )
    if len(matches) != 1:
        raise AuthorityError("recovery bundle lacks one exact regular source blob")
    return matches[0]


def _load_durable_development_selection_graph(
    recovery_capability: BundleSelectedDeltaRecoveryCapability,
) -> tuple[
    BundleSelectedDeltaRecoveryCapability,
    Mapping[str, Any],
    bytes,
    Mapping[str, Any],
    bytes,
]:
    """Load exact immutable dev-result and committed selected bytes at B."""

    recovery = require_bundle_selected_delta_recovery_capability(recovery_capability)
    selected_data = _load_committed_source_blob(
        V3_SOURCE_REPOSITORY,
        SELECTED_METHOD_FREEZE_REPOSITORY_PATH,
        snapshot=recovery._snapshot,
        production=True,
    )
    selected = parse_artifact_bytes(
        selected_data,
        ArtifactSpec(
            schema=ARTIFACT_SCHEMAS["selected_method_freeze"],
            exact_fields=_SELECTED_FREEZE_FIELDS,
        ),
    )
    result_data = load_exact_artifact_bytes(
        [DEVELOPMENT_RESULT_CANONICAL_PATH],
        declared_paths=[DEVELOPMENT_RESULT_CANONICAL_PATH],
        maximum_bytes_per_file=DEVELOPMENT_RESULT_MAXIMUM_BYTES,
    )[os.fspath(DEVELOPMENT_RESULT_CANONICAL_PATH)]
    result = parse_artifact_bytes(
        result_data,
        ArtifactSpec(
            schema=ARTIFACT_SCHEMAS["development_result"],
            exact_fields=_DEVELOPMENT_RESULT_FIELDS,
        ),
    )
    if (
        selected["payload_sha256"] != recovery.selected_method_freeze_payload_sha256
        or file_sha256(selected_data) != recovery.selected_method_freeze_file_sha256
        or result["payload_sha256"] != selected["development_result_payload_sha256"]
        or file_sha256(result_data) != selected["development_result_file_sha256"]
    ):
        raise AuthorityError("durable development/selection file graph differs")
    return recovery, selected, selected_data, result, result_data


def reopen_context_reference(
    *,
    development_bundle: DevelopmentBundleCapability,
    recovery_capability: BundleSelectedDeltaRecoveryCapability,
) -> ValidatedArtifactCapability:
    """Reopen canonical reference authority at B through the durable result chain."""

    _require_active_governed_process(
        recheck_site_packages=True,
        recheck_source=True,
    )
    recovery, selected, _, result, result_data = _load_durable_development_selection_graph(
        recovery_capability
    )
    bundle = require_development_bundle_capability(
        development_bundle,
        expected_commit=recovery.bundle_commit,
        expected_tree=recovery.bundle_tree,
    )
    if (
        bundle.payload_sha256 != recovery.development_bundle_payload_sha256
        or bundle.file_sha256 != recovery.development_bundle_file_sha256
        or bundle.source_bundle_sha256 != recovery.development_bundle_source_bundle_sha256
    ):
        raise AuthorityError("reference reopen bundle differs from B recovery")
    reference_data = load_exact_artifact_bytes(
        [CONTEXT_REFERENCE_CANONICAL_PATH],
        declared_paths=[CONTEXT_REFERENCE_CANONICAL_PATH],
    )[os.fspath(CONTEXT_REFERENCE_CANONICAL_PATH)]
    retired_v1, _ = observe_retired_development_attempts()
    if reference_data != retired_v1_context_reference_bytes(retired_v1):
        raise AuthorityError(
            "durable development-v3 context differs from the exact retired V1 bytes"
        )
    reference = parse_artifact_bytes(
        reference_data,
        ArtifactSpec(
            schema=ARTIFACT_SCHEMAS["context_reference"],
            exact_fields=_CONTEXT_REFERENCE_FIELDS,
        ),
    )
    try:
        from cfeg.analysis.metadata_calibration_v3_synthetic import (
            validate_context_reference_for_audit,
        )
    except ImportError as exc:
        raise AuthorityError("V3 core context audit validator is unavailable") from exc
    core_reference = validate_context_reference_for_audit(
        reference,
        context_reference_file_sha256=file_sha256(reference_data),
        development_result=result,
        development_result_file_sha256=file_sha256(result_data),
        selected_method_freeze=selected,
        recovery_capability=recovery,
    )
    if (
        core_reference.payload_sha256 != reference["payload_sha256"]
        or core_reference.domain != reference["domain"]
        or tuple(core_reference.canonical_channels) != tuple(reference["canonical_channels"])
        or tuple(table.interface_lookup_key for table in core_reference.interface_tables)
        != tuple(table["interface_lookup_key"] for table in reference["interface_tables"])
    ):
        raise AuthorityError("core context audit differs from canonical reference bytes")
    bindings = {
        "domain": "synthetic_development_and_future_synthetic",
        "canonical_channels": list(_SYNTHETIC_CHANNELS),
        "minimum_per_interface_channel_count": 128,
        "pooled_per_channel_fallback_minimum_count": 256,
        "scale_floor": 0.05,
    }
    return _validate_schema_artifact(
        reference,
        file_bytes=reference_data,
        schema=ARTIFACT_SCHEMAS["context_reference"],
        exact_fields=_CONTEXT_REFERENCE_FIELDS,
        expected_bindings=bindings,
        canonical_path=CONTEXT_REFERENCE_CANONICAL_PATH,
        scope="canonical-publication",
        _observer=_CAPABILITY_ISSUER,
    )


def reopen_development_result(
    *,
    development_bundle: DevelopmentBundleCapability,
    context_reference: ValidatedArtifactCapability,
    recovery_capability: BundleSelectedDeltaRecoveryCapability,
) -> ValidatedArtifactCapability:
    """Reopen canonical development authority without recreating RNG authority."""

    _require_active_governed_process(
        recheck_site_packages=True,
        recheck_source=True,
    )
    recovery, selected, _, result, result_data = _load_durable_development_selection_graph(
        recovery_capability
    )
    bundle = require_development_bundle_capability(
        development_bundle,
        expected_commit=recovery.bundle_commit,
        expected_tree=recovery.bundle_tree,
    )
    reference = require_validated_artifact_capability(
        context_reference,
        expected_schema=ARTIFACT_SCHEMAS["context_reference"],
    )
    development_start = reopen_development_start_receipt(
        development_bundle=bundle,
        context_reference=reference,
    )
    if (
        result["development_start_schema"] != development_start.schema
        or result["development_start_payload_sha256"] != development_start.payload_sha256
        or result["development_start_file_sha256"] != development_start.file_sha256
    ):
        raise AuthorityError("durable development result differs from its start receipt")
    try:
        from cfeg.analysis.metadata_calibration_v3_synthetic import (
            validate_development_result_for_audit,
        )
    except ImportError as exc:
        raise AuthorityError("V3 core development audit validator is unavailable") from exc
    audited = validate_development_result_for_audit(
        result,
        development_result_file_sha256=file_sha256(result_data),
        selected_method_freeze=selected,
        recovery_capability=recovery,
    )
    if canonical_json_bytes(audited) != canonical_json_bytes(result):
        raise AuthorityError("core development audit differs from canonical result bytes")
    bindings = {
        "master_plan_file_sha256": FROZEN_FILE_SHA256[MASTER_PLAN_PATH],
        "synthetic_plan_file_sha256": FROZEN_FILE_SHA256[SYNTHETIC_PLAN_PATH],
        "development_bundle_schema": bundle.schema,
        "development_bundle_payload_sha256": bundle.payload_sha256,
        "development_bundle_file_sha256": bundle.file_sha256,
        "context_reference_schema": reference.schema,
        "context_reference_payload_sha256": reference.payload_sha256,
        "context_reference_file_sha256": reference.file_sha256,
        "development_start_schema": development_start.schema,
        "development_start_payload_sha256": development_start.payload_sha256,
        "development_start_file_sha256": development_start.file_sha256,
        "development_root_seed": 20_260_909,
    }
    return _validate_schema_artifact(
        result,
        file_bytes=result_data,
        schema=ARTIFACT_SCHEMAS["development_result"],
        exact_fields=_DEVELOPMENT_RESULT_FIELDS,
        expected_bindings=bindings,
        canonical_path=DEVELOPMENT_RESULT_CANONICAL_PATH,
        scope="canonical-publication",
        _observer=_CAPABILITY_ISSUER,
    )


def reopen_selected_method_freeze(
    *,
    development_bundle: DevelopmentBundleCapability,
    development_result: ValidatedArtifactCapability,
    recovery_capability: BundleSelectedDeltaRecoveryCapability,
) -> ValidatedArtifactCapability:
    """Reopen selected B from exact A→B delta plus durable dev semantics."""

    _require_active_governed_process(
        recheck_site_packages=True,
        recheck_source=True,
    )
    recovery, selected, selected_data, result, result_data = (
        _load_durable_development_selection_graph(recovery_capability)
    )
    bundle = require_development_bundle_capability(
        development_bundle,
        expected_commit=recovery.bundle_commit,
        expected_tree=recovery.bundle_tree,
    )
    result_capability = require_validated_artifact_capability(
        development_result,
        expected_schema=ARTIFACT_SCHEMAS["development_result"],
    )
    if result_capability.payload_sha256 != result[
        "payload_sha256"
    ] or result_capability.file_sha256 != file_sha256(result_data):
        raise AuthorityError("selected reopen development capability differs from disk")
    try:
        from cfeg.analysis.metadata_calibration_v3_synthetic import (
            validate_selected_method_freeze_for_audit,
        )
    except ImportError as exc:
        raise AuthorityError("V3 core selected-method audit validator is unavailable") from exc
    audited = validate_selected_method_freeze_for_audit(
        selected,
        development_result=result,
        development_result_file_sha256=file_sha256(result_data),
        recovery_capability=recovery,
    )
    if canonical_json_bytes(audited) != canonical_json_bytes(selected):
        raise AuthorityError("core selected audit differs from committed selected bytes")
    gate_report_sha256 = hashlib.sha256(
        canonical_json_bytes(
            {
                "schema": ("cfeg.metadata-calibration-efficiency-v3.complete-grid-gate-report.v1"),
                "reports": result["complete_grid_gate_report"],
            }
        )
    ).hexdigest()
    bindings = {
        "master_plan_file_sha256": FROZEN_FILE_SHA256[MASTER_PLAN_PATH],
        "synthetic_plan_file_sha256": FROZEN_FILE_SHA256[SYNTHETIC_PLAN_PATH],
        "development_bundle_schema": bundle.schema,
        "development_bundle_payload_sha256": bundle.payload_sha256,
        "development_bundle_file_sha256": bundle.file_sha256,
        "development_result_schema": result_capability.schema,
        "development_result_payload_sha256": result_capability.payload_sha256,
        "development_result_file_sha256": result_capability.file_sha256,
        "complete_grid_gate_report_sha256": gate_report_sha256,
        "clean_commit": bundle.clean_commit,
        "clean_tree": bundle.clean_tree,
    }
    return _validate_schema_artifact(
        selected,
        file_bytes=selected_data,
        schema=ARTIFACT_SCHEMAS["selected_method_freeze"],
        exact_fields=_SELECTED_FREEZE_FIELDS,
        expected_bindings=bindings,
        canonical_path=V3_SOURCE_REPOSITORY / SELECTED_METHOD_FREEZE_REPOSITORY_PATH,
        scope="committed-source",
        source_commit=recovery.selected_commit,
        source_tree=recovery.selected_tree,
        _observer=_CAPABILITY_ISSUER,
    )


_PYTEST_ISOLATED_BOOTSTRAP = (
    _GOVERNED_PYTHON_BOOTSTRAP_PREAMBLE
    + "import cfeg\n"
    + "import cfeg.metadata_calibration_v3_governance as g\n"
    + "_site_args=[(i,v) for i,v in enumerate(sys.argv) "
    + "if v.startswith(g._PYTHON_SITE_INVENTORY_ARG_PREFIX)]\n"
    + "if len(_site_args)!=1:\n"
    + " raise RuntimeError('governed pytest requires one site-inventory digest')\n"
    + "_site_index,_site_arg=_site_args[0]\n"
    + "_site_digest=_site_arg[len(g._PYTHON_SITE_INVENTORY_ARG_PREFIX):]\n"
    + "del sys.argv[_site_index]\n"
    + "g.establish_governed_process("
    + "expected_python_site_inventory_sha256=_site_digest)\n"
    + "import pytest\n"
    + "g._require_loaded_modules_within_governed_roots(require_pytest=True)\n"
    + "_pytest_exit_code=pytest.main(sys.argv[1:])\n"
    + "g._require_active_governed_process(recheck_site_packages=True)\n"
    + "raise SystemExit(_pytest_exit_code)\n"
)
_TEST_COMMANDS = MappingProxyType(
    {
        "focused_v3": (
            *_GOVERNED_PYTHON_SUBPROCESS_PREFIX,
            "-c",
            _PYTEST_ISOLATED_BOOTSTRAP,
            "-q",
            "-c",
            "pyproject.toml",
            "--rootdir=.",
            "-o",
            "addopts=",
            "-o",
            "pythonpath=",
            "-o",
            "xfail_strict=true",
            "--import-mode=importlib",
            "-p",
            "no:cacheprovider",
            "tests/test_metadata_calibration_v3.py",
            "tests/test_metadata_calibration_v3_synthetic.py",
            "tests/test_metadata_calibration_v3_governance.py",
            "tests/test_metadata_calibration_v3_isolation.py",
            "tests/test_metadata_calibration_v3_contract.py",
            "tests/test_metadata_calibration_v3_runner.py",
        ),
        "repository_full": (
            *_GOVERNED_PYTHON_SUBPROCESS_PREFIX,
            "-c",
            _PYTEST_ISOLATED_BOOTSTRAP,
            "-q",
            "-c",
            "pyproject.toml",
            "--rootdir=.",
            "-o",
            "addopts=",
            "-o",
            "pythonpath=",
            "-o",
            "xfail_strict=true",
            "--import-mode=importlib",
            "-p",
            "no:cacheprovider",
        ),
    }
)


def _require_governed_python_command_prefix(argv: Sequence[str]) -> None:
    prefix_length = len(_GOVERNED_PYTHON_SUBPROCESS_PREFIX)
    if (
        len(argv) <= prefix_length + 1
        or tuple(argv[:prefix_length]) != _GOVERNED_PYTHON_SUBPROCESS_PREFIX
        or argv[prefix_length] != "-c"
        or not isinstance(argv[prefix_length + 1], str)
        or not argv[prefix_length + 1]
    ):
        raise AuthorityError("governed Python command requires exact python -I -B -S -c")


def _require_loaded_module_path(module_name: str, expected_path: Path) -> None:
    module = sys.modules.get(module_name)
    module_path = getattr(module, "__file__", None)
    if not isinstance(module_path, str):
        raise AuthorityError(f"governed Python did not import required module: {module_name}")
    observed = Path(module_path)
    if (
        observed != expected_path
        or observed.resolve(strict=True) != expected_path
        or expected_path.resolve(strict=True) != expected_path
    ):
        raise AuthorityError(f"governed Python imported {module_name} outside its exact root")


def _require_loaded_modules_within_governed_roots(
    *,
    require_pytest: bool,
    allow_site_packages: bool = True,
) -> None:
    allowed_roots = [
        V3_SOURCE_REPOSITORY / "src",
        V3_SOURCE_REPOSITORY / "tests",
        V3_SOURCE_REPOSITORY / "scripts",
        Path("/usr/lib/python3.10"),
    ]
    if allow_site_packages:
        allowed_roots.append(V3_SITE_PACKAGES)
    for module_name, module in tuple(sys.modules.items()):
        module_file = getattr(module, "__file__", None)
        if module_file is None:
            continue
        if not isinstance(module_file, str) or not module_file:
            raise AuthorityError(f"loaded module has malformed source identity: {module_name}")
        raw_path = Path(module_file)
        try:
            resolved_path = raw_path.resolve(strict=True)
        except (OSError, RuntimeError) as exc:
            raise AuthorityError(
                f"loaded module source cannot be resolved exactly: {module_name}"
            ) from exc
        if raw_path != resolved_path or not any(
            _is_same_or_descendant(resolved_path, root) for root in allowed_roots
        ):
            raise AuthorityError(f"loaded module is outside governed roots: {module_name}")
    if require_pytest:
        _require_loaded_module_path(
            "pytest",
            V3_SITE_PACKAGES / "pytest/__init__.py",
        )


GOVERNED_PROCESS_CAPABILITY_SCHEMA = (
    "cfeg.metadata-calibration-efficiency-v3.governed-process-capability.v1"
)


@dataclass(frozen=True, slots=True, init=False)
class GovernedProcessCapability:
    schema: str
    process_id: int
    role: str
    python_site_inventory_sha256: str
    executable_file_sha256: str
    clean_commit: str
    clean_tree: str
    source_bundle_sha256: str
    environment_sha256: str
    capability_seal_sha256: str
    _validation_marker: object = field(repr=False, compare=False)

    def __init__(
        self,
        *,
        role: str,
        python_site_inventory_sha256: str,
        executable_file_sha256: str,
        snapshot: CleanSourceSnapshot,
        _issuer: object,
    ) -> None:
        if _issuer is not _CAPABILITY_ISSUER:
            raise AuthorityError("GovernedProcessCapability has no valid issuer")
        bindings = {
            "schema": GOVERNED_PROCESS_CAPABILITY_SCHEMA,
            "process_id": os.getpid(),
            "role": role,
            "python_site_inventory_sha256": python_site_inventory_sha256,
            "executable_file_sha256": executable_file_sha256,
            "clean_commit": snapshot.identity.commit,
            "clean_tree": snapshot.identity.tree,
            "source_bundle_sha256": snapshot.source_bundle_sha256,
            "environment_sha256": hashlib.sha256(
                canonical_json_bytes(_OBSERVED_TEST_ENVIRONMENT)
            ).hexdigest(),
        }
        for name, value in bindings.items():
            object.__setattr__(self, name, value)
        object.__setattr__(
            self,
            "capability_seal_sha256",
            hashlib.sha256(canonical_json_bytes(bindings)).hexdigest(),
        )
        object.__setattr__(self, "_validation_marker", _issuer)


_ACTIVE_GOVERNED_PROCESS_CAPABILITY: GovernedProcessCapability | None = None


def governed_runner_command(command: str) -> tuple[str, ...]:
    if command not in _GOVERNED_RUNNER_COMMANDS:
        raise ValidationError("unknown governed V3 runner command")
    return (*_GOVERNED_PYTHON_SUBPROCESS_PREFIX, os.fspath(V3_RUNNER_PATH), command)


def _governed_process_role(expected_site_inventory_sha256: str) -> str:
    original = tuple(getattr(sys, "orig_argv", ()))
    if original == _FRESH_CANARY_EXEC_COMMAND:
        if tuple(sys.argv) != ("-c", _FRESH_CANARY_EXEC_PHASE):
            raise AuthorityError("fresh-audit child sys.argv is not exact")
        return "fresh_canary_audit"
    for command in _GOVERNED_RUNNER_COMMANDS:
        expected = governed_runner_command(command)
        if original == expected:
            if tuple(sys.argv) != (os.fspath(V3_RUNNER_PATH), command):
                raise AuthorityError("governed runner sys.argv is not exact")
            return f"runner_{command.replace('-', '_')}"
    for name in _TEST_COMMANDS:
        if (
            len(original) >= 3
            and original[-3]
            == f"{_PYTHON_SITE_INVENTORY_ARG_PREFIX}{expected_site_inventory_sha256}"
            and original[-2].startswith("--basetemp=")
            and original[-1].startswith("--junitxml=")
        ):
            junit_path = original[-1].split("=", 1)[1]
            try:
                _require_frozen_test_argv(
                    name,
                    original,
                    junit_path,
                    expected_python_site_inventory_sha256=(expected_site_inventory_sha256),
                )
            except AuthorityError:
                continue
            if not sys.argv or sys.argv[0] != "-c":
                raise AuthorityError("governed pytest sys.argv is not exact")
            return f"pytest_{name}"
    raise AuthorityError("process orig_argv is not a frozen governed V3 role")


def _assert_governed_python_subprocess_runtime(
    *,
    expected_python_site_inventory_sha256: str,
) -> tuple[str, CleanSourceSnapshot, Mapping[str, Any]]:
    """Validate one pre-import isolated process and its complete site tree."""

    expected_path = (
        *_ISOLATED_STDLIB_SYS_PATH,
        os.fspath(V3_SOURCE_REPOSITORY / "src"),
        os.fspath(V3_SITE_PACKAGES),
    )
    if (
        sys.flags.isolated != 1
        or sys.flags.dont_write_bytecode != 1
        or sys.flags.no_site != 1
        or sys.flags.ignore_environment != 1
        or sys.flags.no_user_site != 1
        or sys.flags.hash_randomization != 1
        or Path(sys.executable) != V3_PYTHON_EXECUTABLE
        or tuple(sys.path) != expected_path
        or dict(os.environ) != dict(_OBSERVED_TEST_ENVIRONMENT)
    ):
        raise AuthorityError("governed Python runtime differs from exact -I -B -S bootstrap")
    _require_sha256(
        expected_python_site_inventory_sha256,
        "expected governed Python site inventory",
    )
    role = _governed_process_role(expected_python_site_inventory_sha256)
    site_inventory = _capture_complete_site_packages_inventory()
    if site_inventory["inventory_sha256"] != expected_python_site_inventory_sha256:
        raise AuthorityError("pre-import Python site inventory differs from trusted digest")
    if _current_executable_file_sha256() != _hash_installed_regular_file(
        V3_PYTHON_EXECUTABLE.resolve(strict=True)
    ):
        raise AuthorityError("governed Python executable bytes differ from the pinned binary")
    _require_loaded_module_path(
        "cfeg",
        V3_SOURCE_REPOSITORY / "src/cfeg/__init__.py",
    )
    _require_loaded_module_path(
        "cfeg.metadata_calibration_v3_governance",
        V3_SOURCE_REPOSITORY / "src/cfeg/metadata_calibration_v3_governance.py",
    )
    _require_loaded_modules_within_governed_roots(
        require_pytest=False,
        allow_site_packages=False,
    )
    snapshot = capture_clean_source_snapshot(V3_SOURCE_REPOSITORY)
    _require_loaded_governance_source_at_snapshot(snapshot)
    init_record = next(
        (entry for entry in snapshot.tracked_files if entry.path == "src/cfeg/__init__.py"),
        None,
    )
    if (
        init_record is None
        or init_record.kind != "regular_file"
        or init_record.file_sha256
        != _hash_installed_regular_file(V3_SOURCE_REPOSITORY / "src/cfeg/__init__.py")
    ):
        raise AuthorityError("loaded cfeg package root differs from tracked source inventory")
    return role, snapshot, site_inventory


def establish_governed_process(
    *,
    expected_python_site_inventory_sha256: str,
) -> GovernedProcessCapability:
    """Issue one process-local authority after stdlib-only pre-import validation."""

    global _ACTIVE_GOVERNED_PROCESS_CAPABILITY
    role, snapshot, _ = _assert_governed_python_subprocess_runtime(
        expected_python_site_inventory_sha256=expected_python_site_inventory_sha256,
    )
    capability = GovernedProcessCapability(
        role=role,
        python_site_inventory_sha256=expected_python_site_inventory_sha256,
        executable_file_sha256=_current_executable_file_sha256(),
        snapshot=snapshot,
        _issuer=_CAPABILITY_ISSUER,
    )
    _ACTIVE_GOVERNED_PROCESS_CAPABILITY = capability
    return capability


def _require_active_governed_process(
    *,
    recheck_site_packages: bool = False,
    recheck_source: bool = False,
    require_mutation_role: bool = False,
) -> GovernedProcessCapability:
    value = _ACTIVE_GOVERNED_PROCESS_CAPABILITY
    if type(value) is not GovernedProcessCapability:
        raise AuthorityError("active exact GovernedProcessCapability required")
    capability = value
    bindings = {
        "schema": capability.schema,
        "process_id": capability.process_id,
        "role": capability.role,
        "python_site_inventory_sha256": capability.python_site_inventory_sha256,
        "executable_file_sha256": capability.executable_file_sha256,
        "clean_commit": capability.clean_commit,
        "clean_tree": capability.clean_tree,
        "source_bundle_sha256": capability.source_bundle_sha256,
        "environment_sha256": capability.environment_sha256,
    }
    expected_path = (
        *_ISOLATED_STDLIB_SYS_PATH,
        os.fspath(V3_SOURCE_REPOSITORY / "src"),
        os.fspath(V3_SITE_PACKAGES),
    )
    if (
        capability._validation_marker is not _CAPABILITY_ISSUER
        or capability.schema != GOVERNED_PROCESS_CAPABILITY_SCHEMA
        or capability.process_id != os.getpid()
        or capability.environment_sha256
        != hashlib.sha256(canonical_json_bytes(_OBSERVED_TEST_ENVIRONMENT)).hexdigest()
        or capability.capability_seal_sha256
        != hashlib.sha256(canonical_json_bytes(bindings)).hexdigest()
        or capability.executable_file_sha256 != _current_executable_file_sha256()
        or sys.flags.isolated != 1
        or sys.flags.dont_write_bytecode != 1
        or sys.flags.no_site != 1
        or sys.flags.ignore_environment != 1
        or sys.flags.no_user_site != 1
        or sys.flags.hash_randomization != 1
        or Path(sys.executable) != V3_PYTHON_EXECUTABLE
        or tuple(sys.path) != expected_path
        or dict(os.environ) != dict(_OBSERVED_TEST_ENVIRONMENT)
        or _governed_process_role(capability.python_site_inventory_sha256) != capability.role
    ):
        raise AuthorityError("governed-process authority binding is invalid")
    if require_mutation_role and capability.role not in {
        "runner_resume",
        "fresh_canary_audit",
    }:
        raise AuthorityError("this governed process role cannot mutate canonical state")
    _require_loaded_modules_within_governed_roots(require_pytest=False)
    if recheck_site_packages:
        current = _capture_complete_site_packages_inventory()
        if current["inventory_sha256"] != capability.python_site_inventory_sha256:
            raise AuthorityError("Python site inventory drifted after process attestation")
    if recheck_source:
        current_source = capture_clean_source_snapshot(V3_SOURCE_REPOSITORY)
        if (
            current_source.identity.commit != capability.clean_commit
            or current_source.identity.tree != capability.clean_tree
            or current_source.source_bundle_sha256 != capability.source_bundle_sha256
        ):
            raise AuthorityError("source drifted after governed-process attestation")
        _require_loaded_governance_source_at_snapshot(current_source)
    return capability


def require_governed_process_capability(
    value: object,
    *,
    expected_role: str | None = None,
) -> GovernedProcessCapability:
    """Validate the nominal capability installed for this exact live process."""

    capability = _require_active_governed_process()
    if value is not capability:
        raise AuthorityError("governed-process capability is not the active exact instance")
    if expected_role is not None and capability.role != expected_role:
        raise AuthorityError("governed-process capability role mismatch")
    return capability


@dataclass(frozen=True, slots=True, init=False)
class ObservedTestRunCapability:
    name: str
    argv: tuple[str, ...]
    exit_code: int
    collected_tests: int
    passed_tests: int
    failed_tests: int
    skipped_tests: int
    xfailed_tests: int
    xpassed_tests: int
    error_tests: int
    process_id: int
    executable_file_sha256: str
    stdout_sha256: str
    stderr_sha256: str
    junit_report_file_sha256: str
    junit_report_path: str
    junit_report_bytes: bytes = field(repr=False, compare=False)
    clean_commit: str
    clean_tree: str
    source_bundle_sha256: str
    working_directory: str
    environment_sha256: str
    scope: str
    _validation_marker: object = field(repr=False, compare=False)

    def __init__(
        self,
        *,
        name: str,
        argv: tuple[str, ...],
        exit_code: int,
        collected_tests: int,
        passed_tests: int,
        failed_tests: int,
        skipped_tests: int,
        xfailed_tests: int,
        xpassed_tests: int,
        error_tests: int,
        process_id: int,
        executable_file_sha256: str,
        stdout_sha256: str,
        stderr_sha256: str,
        junit_report_file_sha256: str,
        junit_report_path: str,
        junit_report_bytes: bytes,
        snapshot: CleanSourceSnapshot,
        working_directory: str,
        environment_sha256: str,
        scope: str,
        _issuer: object,
    ) -> None:
        if _issuer is not _CAPABILITY_ISSUER:
            raise AuthorityError("ObservedTestRunCapability has no valid runner issuer")
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "argv", argv)
        object.__setattr__(self, "exit_code", exit_code)
        object.__setattr__(self, "collected_tests", collected_tests)
        object.__setattr__(self, "passed_tests", passed_tests)
        object.__setattr__(self, "failed_tests", failed_tests)
        object.__setattr__(self, "skipped_tests", skipped_tests)
        object.__setattr__(self, "xfailed_tests", xfailed_tests)
        object.__setattr__(self, "xpassed_tests", xpassed_tests)
        object.__setattr__(self, "error_tests", error_tests)
        object.__setattr__(self, "process_id", process_id)
        object.__setattr__(self, "executable_file_sha256", executable_file_sha256)
        object.__setattr__(self, "stdout_sha256", stdout_sha256)
        object.__setattr__(self, "stderr_sha256", stderr_sha256)
        object.__setattr__(self, "junit_report_file_sha256", junit_report_file_sha256)
        object.__setattr__(self, "junit_report_path", junit_report_path)
        object.__setattr__(self, "junit_report_bytes", junit_report_bytes)
        object.__setattr__(self, "clean_commit", snapshot.identity.commit)
        object.__setattr__(self, "clean_tree", snapshot.identity.tree)
        object.__setattr__(self, "source_bundle_sha256", snapshot.source_bundle_sha256)
        object.__setattr__(self, "working_directory", working_directory)
        object.__setattr__(self, "environment_sha256", environment_sha256)
        object.__setattr__(self, "scope", scope)
        object.__setattr__(self, "_validation_marker", _issuer)

    def as_record(self) -> Mapping[str, Any]:
        return MappingProxyType(
            {
                "name": self.name,
                "argv": list(self.argv),
                "exit_code": self.exit_code,
                "collected_tests": self.collected_tests,
                "passed_tests": self.passed_tests,
                "failed_tests": self.failed_tests,
                "skipped_tests": self.skipped_tests,
                "xfailed_tests": self.xfailed_tests,
                "xpassed_tests": self.xpassed_tests,
                "error_tests": self.error_tests,
                "process_id": self.process_id,
                "executable_file_sha256": self.executable_file_sha256,
                "stdout_sha256": self.stdout_sha256,
                "stderr_sha256": self.stderr_sha256,
                "junit_report_file_sha256": self.junit_report_file_sha256,
                "junit_report_path": self.junit_report_path,
                "junit_report_base64": base64.b64encode(self.junit_report_bytes).decode("ascii"),
                "source_bundle_sha256": self.source_bundle_sha256,
                "working_directory": self.working_directory,
                "environment_sha256": self.environment_sha256,
                "scope": self.scope,
            }
        )


def _same_source_snapshot(left: CleanSourceSnapshot, right: CleanSourceSnapshot) -> bool:
    return (
        left.identity == right.identity
        and left.source_bundle_sha256 == right.source_bundle_sha256
        and left.tracked_files == right.tracked_files
    )


def _parse_pytest_junit_report(data: bytes) -> Mapping[str, int]:
    if b"<!DOCTYPE" in data or b"<!ENTITY" in data:
        raise AuthorityError("pytest JUnit report may not contain DTD or entity declarations")
    try:
        root = ET.fromstring(data)
    except ET.ParseError as exc:
        raise AuthorityError("pytest JUnit report is malformed") from exc
    if root.tag == "testsuite":
        suites = (root,)
    elif root.tag == "testsuites":
        suites = tuple(root.findall("./testsuite"))
    else:
        raise AuthorityError("pytest JUnit report has an unexpected root element")
    if not suites:
        raise AuthorityError("pytest JUnit report contains no test suites")
    totals = {"collected": 0, "failed": 0, "skipped": 0, "errors": 0}
    attributes = {
        "collected": "tests",
        "failed": "failures",
        "skipped": "skipped",
        "errors": "errors",
    }
    try:
        for suite in suites:
            for name, attribute in attributes.items():
                raw = suite.attrib[attribute]
                if not raw.isascii() or not raw.isdecimal():
                    raise ValueError
                totals[name] += int(raw)
    except (KeyError, ValueError) as exc:
        raise AuthorityError("pytest JUnit report has invalid exact count attributes") from exc
    totals["passed"] = totals["collected"] - totals["failed"] - totals["skipped"] - totals["errors"]
    if totals["passed"] < 0:
        raise AuthorityError("pytest JUnit report counts are inconsistent")
    return MappingProxyType(totals)


def _read_and_seal_junit_report(path: Path) -> bytes:
    parent_fd = _open_absolute_directory(path.parent, create=False, require_private_final=True)
    try:
        try:
            descriptor = os.open(path.name, os.O_RDONLY | os.O_NOFOLLOW, dir_fd=parent_fd)
        except FileNotFoundError as exc:
            raise AuthorityError("allowlisted test command emitted no JUnit report") from exc
        try:
            observed = os.fstat(descriptor)
            if (
                not stat.S_ISREG(observed.st_mode)
                or observed.st_uid != os.geteuid()
                or observed.st_nlink != 1
            ):
                raise AuthorityError("pytest JUnit report is not one owner-controlled regular file")
            os.fchmod(descriptor, 0o400)
            sealed = os.fstat(descriptor)
            if stat.S_IMODE(sealed.st_mode) != 0o400:
                raise AuthorityError("pytest JUnit report could not be sealed mode-0400")
            return _read_all(descriptor, maximum_bytes=64 * 1024 * 1024)
        finally:
            os.close(descriptor)
    finally:
        os.close(parent_fd)


def _run_observed_test_at_repository(
    name: str,
    *,
    repository: Path,
    snapshot: CleanSourceSnapshot,
    scope: str,
) -> ObservedTestRunCapability:
    """Run one frozen command and bind observed process/output/source facts."""

    snapshot = _require_clean_source_snapshot(snapshot)
    if name not in _TEST_COMMANDS:
        raise ValidationError("test role is not in the frozen command allowlist")
    _require_governed_python_command_prefix(_TEST_COMMANDS[name])
    if scope not in {"canonical", "test-only"}:
        raise ValidationError("unknown observed-test scope")
    root = _canonical_absolute_path(repository)
    if scope == "canonical" and root != V3_SOURCE_REPOSITORY:
        raise AuthorityError("production test observation requires the canonical repository")
    before = capture_clean_source_snapshot(root)
    if not _same_source_snapshot(before, snapshot):
        raise AuthorityError("test runner source differs from the approved clean snapshot")
    before_runtime_inventory, before_runtime_fingerprint = _current_numerical_runtime()
    site_inventory_sha256 = _python_site_inventory_sha256(before_runtime_inventory)
    junit_directory = Path(tempfile.mkdtemp(prefix="cfeg-v3-junit-", dir="/tmp"))
    os.chmod(junit_directory, 0o700)
    junit_path = junit_directory / "report.xml"
    basetemp_path = junit_directory / "pytest-temp"
    argv = (
        *_TEST_COMMANDS[name],
        f"{_PYTHON_SITE_INVENTORY_ARG_PREFIX}{site_inventory_sha256}",
        f"--basetemp={basetemp_path}",
        f"--junitxml={junit_path}",
    )
    executable = _canonical_absolute_path(argv[0]).resolve(strict=True)
    executable_sha256 = file_sha256(executable.read_bytes())
    process: subprocess.Popen[bytes] | None = None
    try:
        process = subprocess.Popen(
            argv,
            cwd=root,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=dict(_OBSERVED_TEST_ENVIRONMENT),
        )
        observed_executable_sha256 = _process_executable_file_sha256(process.pid)
        if observed_executable_sha256 != executable_sha256:
            raise AuthorityError("spawned test process executable differs from frozen argv")
        stdout, stderr = process.communicate()
        after = capture_clean_source_snapshot(root)
        if not _same_source_snapshot(after, snapshot):
            raise AuthorityError("source changed while the allowlisted test command ran")
        after_runtime_inventory, after_runtime_fingerprint = _current_numerical_runtime()
        if before_runtime_fingerprint != after_runtime_fingerprint or _plain_json(
            before_runtime_inventory
        ) != _plain_json(after_runtime_inventory):
            raise AuthorityError("numerical runtime changed while the test command ran")
        junit_data = _read_and_seal_junit_report(junit_path)
        receipt = _parse_pytest_junit_report(junit_data)
        junit_sha256 = file_sha256(junit_data)
    except BaseException:
        if process is not None and process.poll() is None:
            process.terminate()
            try:
                process.communicate(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.communicate()
        raise
    finally:
        shutil.rmtree(junit_directory)
    if (
        process.returncode != 0
        or receipt["collected"] <= 0
        or receipt["passed"] != receipt["collected"]
        or any(receipt[name] != 0 for name in ("failed", "skipped", "errors"))
    ):
        raise AuthorityError("observed allowlisted test command did not pass completely")
    return ObservedTestRunCapability(
        name=name,
        argv=argv,
        exit_code=process.returncode,
        collected_tests=receipt["collected"],
        passed_tests=receipt["passed"],
        failed_tests=receipt["failed"],
        skipped_tests=receipt["skipped"],
        xfailed_tests=0,
        xpassed_tests=0,
        error_tests=receipt["errors"],
        process_id=process.pid,
        executable_file_sha256=executable_sha256,
        stdout_sha256=file_sha256(stdout),
        stderr_sha256=file_sha256(stderr),
        junit_report_file_sha256=junit_sha256,
        junit_report_path=os.fspath(junit_path),
        junit_report_bytes=junit_data,
        snapshot=snapshot,
        working_directory=os.fspath(root),
        environment_sha256=before_runtime_fingerprint,
        scope=scope,
        _issuer=_CAPABILITY_ISSUER,
    )


def run_observed_test(
    name: str,
    *,
    snapshot: CleanSourceSnapshot,
) -> ObservedTestRunCapability:
    """Run a frozen test command only in the canonical source repository."""

    _require_active_governed_process(
        recheck_site_packages=True,
        recheck_source=True,
        require_mutation_role=True,
    )
    return _run_observed_test_at_repository(
        name,
        repository=V3_SOURCE_REPOSITORY,
        snapshot=snapshot,
        scope="canonical",
    )


def _run_observed_test_under_test_root(
    name: str,
    *,
    repository: Path,
    snapshot: CleanSourceSnapshot,
) -> ObservedTestRunCapability:
    return _run_observed_test_at_repository(
        name,
        repository=repository,
        snapshot=snapshot,
        scope="test-only",
    )


def _require_frozen_test_argv(
    name: str,
    argv: Sequence[str],
    junit_report_path: str,
    *,
    expected_python_site_inventory_sha256: str,
) -> None:
    if name not in _TEST_COMMANDS:
        raise AuthorityError("test role is not in the frozen command allowlist")
    _require_governed_python_command_prefix(_TEST_COMMANDS[name])
    path = _canonical_absolute_path(junit_report_path)
    if (
        path.parent.parent != Path("/tmp")
        or not path.parent.name.startswith("cfeg-v3-junit-")
        or path.name != "report.xml"
    ):
        raise AuthorityError("JUnit report path is outside the runner-owned temp namespace")
    basetemp_path = path.parent / "pytest-temp"
    _require_sha256(
        expected_python_site_inventory_sha256,
        "expected test Python site inventory",
    )
    expected = (
        *_TEST_COMMANDS[name],
        f"{_PYTHON_SITE_INVENTORY_ARG_PREFIX}{expected_python_site_inventory_sha256}",
        f"--basetemp={basetemp_path}",
        f"--junitxml={path}",
    )
    if tuple(argv) != expected:
        raise AuthorityError("observed test argv differs from the frozen command template")


def _require_observed_test_run_capability(
    value: object,
    *,
    expected_name: str,
    snapshot: CleanSourceSnapshot,
    expected_scope: str,
    expected_environment_sha256: str,
    expected_python_site_inventory_sha256: str,
) -> ObservedTestRunCapability:
    snapshot = _require_clean_source_snapshot(snapshot)
    expected_root = (
        os.fspath(V3_SOURCE_REPOSITORY)
        if expected_scope == "canonical"
        else getattr(value, "working_directory", None)
    )
    if type(value) is not ObservedTestRunCapability:
        raise AuthorityError("exact ObservedTestRunCapability required")
    capability = value
    if capability._validation_marker is not _CAPABILITY_ISSUER:
        raise AuthorityError("observed test-run marker is invalid")
    if (
        expected_name not in _TEST_COMMANDS
        or capability.name != expected_name
        or capability.exit_code != 0
        or capability.collected_tests <= 0
        or capability.passed_tests != capability.collected_tests
        or capability.failed_tests != 0
        or capability.skipped_tests != 0
        or capability.xfailed_tests != 0
        or capability.xpassed_tests != 0
        or capability.error_tests != 0
        or capability.process_id <= 0
        or capability.clean_commit != snapshot.identity.commit
        or capability.clean_tree != snapshot.identity.tree
        or capability.source_bundle_sha256 != snapshot.source_bundle_sha256
        or capability.working_directory != expected_root
        or capability.environment_sha256 != expected_environment_sha256
        or capability.scope != expected_scope
    ):
        raise AuthorityError("observed test-run capability binding mismatch")
    _require_frozen_test_argv(
        expected_name,
        capability.argv,
        capability.junit_report_path,
        expected_python_site_inventory_sha256=(expected_python_site_inventory_sha256),
    )
    expected_executable = _canonical_absolute_path(_TEST_COMMANDS[expected_name][0]).resolve(
        strict=True
    )
    if capability.executable_file_sha256 != file_sha256(expected_executable.read_bytes()):
        raise AuthorityError("observed test executable changed after the run")
    junit_summary = _parse_pytest_junit_report(capability.junit_report_bytes)
    if (
        file_sha256(capability.junit_report_bytes) != capability.junit_report_file_sha256
        or junit_summary["collected"] != capability.collected_tests
        or junit_summary["passed"] != capability.passed_tests
        or junit_summary["failed"] != capability.failed_tests
        or junit_summary["skipped"] != capability.skipped_tests
        or junit_summary["errors"] != capability.error_tests
    ):
        raise AuthorityError("observed test JUnit evidence binding mismatch")
    for field_name in (
        "executable_file_sha256",
        "stdout_sha256",
        "stderr_sha256",
        "junit_report_file_sha256",
        "source_bundle_sha256",
        "environment_sha256",
    ):
        _require_sha256(getattr(capability, field_name), f"observed test {field_name}")
    return capability


def require_observed_test_run_capability(
    value: object,
    *,
    expected_name: str,
    snapshot: CleanSourceSnapshot,
) -> ObservedTestRunCapability:
    """Require a production observation; test-root receipts never satisfy it."""

    runtime_inventory, runtime_fingerprint = _current_numerical_runtime()
    return _require_observed_test_run_capability(
        value,
        expected_name=expected_name,
        snapshot=snapshot,
        expected_scope="canonical",
        expected_environment_sha256=runtime_fingerprint,
        expected_python_site_inventory_sha256=_python_site_inventory_sha256(runtime_inventory),
    )


_TEST_RUN_FIELDS = frozenset(
    {
        "name",
        "argv",
        "exit_code",
        "collected_tests",
        "passed_tests",
        "failed_tests",
        "skipped_tests",
        "xfailed_tests",
        "xpassed_tests",
        "error_tests",
        "process_id",
        "executable_file_sha256",
        "stdout_sha256",
        "stderr_sha256",
        "junit_report_file_sha256",
        "junit_report_path",
        "junit_report_base64",
        "source_bundle_sha256",
        "working_directory",
        "environment_sha256",
        "scope",
    }
)
_TEST_EVIDENCE_FIELDS = frozenset(
    {
        "schema",
        "candidate_id",
        "scientific_candidate_id",
        "selected_method_freeze_schema",
        "selected_method_freeze_payload_sha256",
        "selected_method_freeze_file_sha256",
        "clean_commit",
        "clean_tree",
        "recorded_at_UTC",
        "test_runs",
        "all_tests_passed",
        "public_fixture_file_sha256",
        "governance_module_file_sha256",
        "payload_sha256",
    }
)


def _validate_test_evidence_content(value: Mapping[str, Any]) -> None:
    _require_exact_keys(value, _TEST_EVIDENCE_FIELDS, "test evidence")
    if value.get("schema") != ARTIFACT_SCHEMAS["test_evidence"]:
        raise ValidationError("test-evidence schema mismatch")
    if value.get("candidate_id") != CANDIDATE_ID:
        raise ValidationError("test-evidence candidate mismatch")
    _parse_rfc3339_seconds(value["recorded_at_UTC"], "test evidence recorded_at_UTC")
    if value["all_tests_passed"] is not True:
        raise AuthorityError("test evidence must report an all-pass result")
    if value["public_fixture_file_sha256"] != CANARY_FIXTURE_SHA256:
        raise ValidationError("test evidence public fixture hash mismatch")
    _require_sha256(value["governance_module_file_sha256"], "governance module file_sha256")
    _require_git_object(value["clean_commit"], "test-evidence clean commit")
    _require_git_object(value["clean_tree"], "test-evidence clean tree")
    runs = value["test_runs"]
    if not isinstance(runs, (list, tuple)) or tuple(run.get("name") for run in runs) != (
        "focused_v3",
        "repository_full",
    ):
        raise ValidationError("test evidence requires focused_v3 then repository_full runs")
    runtime_inventory, runtime_fingerprint = _current_numerical_runtime()
    for run in runs:
        if not isinstance(run, Mapping):
            raise ValidationError("test run must be an object")
        _require_exact_keys(run, _TEST_RUN_FIELDS, "test run")
        argv = run["argv"]
        if (
            not isinstance(argv, (list, tuple))
            or not argv
            or any(not isinstance(item, str) or not item for item in argv)
        ):
            raise ValidationError("test run argv must be a nonempty string vector")
        _require_frozen_test_argv(
            str(run["name"]),
            argv,
            str(run["junit_report_path"]),
            expected_python_site_inventory_sha256=_python_site_inventory_sha256(runtime_inventory),
        )
        try:
            junit_data = base64.b64decode(str(run["junit_report_base64"]), validate=True)
        except (binascii.Error, ValueError) as exc:
            raise ValidationError("test JUnit evidence is not strict base64") from exc
        if base64.b64encode(junit_data).decode("ascii") != run["junit_report_base64"]:
            raise ValidationError("test JUnit evidence base64 is noncanonical")
        junit_summary = _parse_pytest_junit_report(junit_data)
        if (
            run["working_directory"] != os.fspath(V3_SOURCE_REPOSITORY)
            or run["environment_sha256"] != runtime_fingerprint
            or run["scope"] != "canonical"
            or run["exit_code"] != 0
            or not isinstance(run["collected_tests"], int)
            or isinstance(run["collected_tests"], bool)
            or run["collected_tests"] <= 0
            or run["passed_tests"] != run["collected_tests"]
            or run["failed_tests"] != 0
            or run["skipped_tests"] != 0
            or run["xfailed_tests"] != 0
            or run["xpassed_tests"] != 0
            or run["error_tests"] != 0
            or not isinstance(run["process_id"], int)
            or isinstance(run["process_id"], bool)
            or run["process_id"] <= 0
            or file_sha256(junit_data) != run["junit_report_file_sha256"]
            or junit_summary["collected"] != run["collected_tests"]
            or junit_summary["passed"] != run["passed_tests"]
            or junit_summary["failed"] != run["failed_tests"]
            or junit_summary["skipped"] != run["skipped_tests"]
            or junit_summary["errors"] != run["error_tests"]
        ):
            raise AuthorityError("test run does not prove every collected test passed")
        for hash_name in (
            "executable_file_sha256",
            "stdout_sha256",
            "stderr_sha256",
            "source_bundle_sha256",
            "environment_sha256",
            "junit_report_file_sha256",
        ):
            _require_sha256(run[hash_name], f"test run {hash_name}")


def seal_test_evidence(
    *,
    selected_method_freeze: ValidatedArtifactCapability,
    observed_runs: Sequence[ObservedTestRunCapability],
    snapshot: CleanSourceSnapshot,
    recorded_at_UTC: str,
) -> Mapping[str, Any]:
    selected = require_validated_artifact_capability(
        selected_method_freeze,
        expected_schema=ARTIFACT_SCHEMAS["selected_method_freeze"],
    )
    snapshot = _require_clean_source_snapshot(snapshot)
    if len(observed_runs) != 2:
        raise AuthorityError("test evidence requires exactly two observed run capabilities")
    runtime_inventory, runtime_fingerprint = _current_numerical_runtime()
    runs = tuple(
        _require_observed_test_run_capability(
            observed_runs[index],
            expected_name=name,
            snapshot=snapshot,
            expected_scope="canonical",
            expected_environment_sha256=runtime_fingerprint,
            expected_python_site_inventory_sha256=_python_site_inventory_sha256(runtime_inventory),
        )
        for index, name in enumerate(("focused_v3", "repository_full"))
    )
    selected_payload = selected._validated_payload
    if (
        selected.source_commit != snapshot.identity.commit
        or selected.source_tree != snapshot.identity.tree
    ):
        raise AuthorityError("test snapshot differs from selected-method clean identity")
    module_hash = next(
        (
            item.file_sha256
            for item in snapshot.tracked_files
            if item.path == "src/cfeg/metadata_calibration_v3_governance.py"
        ),
        None,
    )
    if module_hash is None:
        raise AuthorityError("clean source inventory lacks the V3 governance module")
    value = _seal_schema_artifact(
        ARTIFACT_SCHEMAS["test_evidence"],
        {
            "scientific_candidate_id": selected_payload["scientific_candidate_id"],
            "selected_method_freeze_schema": selected.schema,
            "selected_method_freeze_payload_sha256": selected.payload_sha256,
            "selected_method_freeze_file_sha256": selected.file_sha256,
            "clean_commit": snapshot.identity.commit,
            "clean_tree": snapshot.identity.tree,
            "recorded_at_UTC": recorded_at_UTC,
            "test_runs": [dict(run.as_record()) for run in runs],
            "all_tests_passed": True,
            "public_fixture_file_sha256": CANARY_FIXTURE_SHA256,
            "governance_module_file_sha256": module_hash,
        },
    )
    _validate_test_evidence_content(value)
    return value


def _validate_test_evidence_bytes(
    value: Mapping[str, Any],
    *,
    file_bytes: bytes,
    selected_method_freeze: ValidatedArtifactCapability,
    observed_runs: Sequence[ObservedTestRunCapability],
    snapshot: CleanSourceSnapshot,
) -> tuple[Mapping[str, Any], Mapping[str, Any]]:
    _validate_test_evidence_content(value)
    selected_method_freeze = require_validated_artifact_capability(
        selected_method_freeze,
        expected_schema=ARTIFACT_SCHEMAS["selected_method_freeze"],
    )
    selected_payload = selected_method_freeze._validated_payload
    expected = seal_test_evidence(
        selected_method_freeze=selected_method_freeze,
        observed_runs=observed_runs,
        snapshot=snapshot,
        recorded_at_UTC=str(value["recorded_at_UTC"]),
    )
    if canonical_json_bytes(value) != canonical_json_bytes(expected):
        raise AuthorityError("test evidence differs from observed subprocess receipts")
    bindings = {
        "scientific_candidate_id": selected_payload["scientific_candidate_id"],
        "selected_method_freeze_schema": selected_method_freeze.schema,
        "selected_method_freeze_payload_sha256": selected_method_freeze.payload_sha256,
        "selected_method_freeze_file_sha256": selected_method_freeze.file_sha256,
        "clean_commit": selected_method_freeze.source_commit,
        "clean_tree": selected_method_freeze.source_tree,
        "public_fixture_file_sha256": CANARY_FIXTURE_SHA256,
        "governance_module_file_sha256": value["governance_module_file_sha256"],
    }
    parsed = parse_artifact_bytes(
        file_bytes,
        ArtifactSpec(
            schema=ARTIFACT_SCHEMAS["test_evidence"],
            exact_fields=_TEST_EVIDENCE_FIELDS,
        ),
    )
    if canonical_json_bytes(parsed) != canonical_json_bytes(value):
        raise ValidationError("test evidence differs from exact supplied bytes")
    for key, expected_value in bindings.items():
        if _plain_json(parsed[key]) != _plain_json(expected_value):
            raise ValidationError(f"test-evidence binding mismatch: {key}")
    return parsed, MappingProxyType(bindings)


def publish_test_evidence(
    value: Mapping[str, Any],
    *,
    selected_method_freeze: ValidatedArtifactCapability,
    observed_runs: Sequence[ObservedTestRunCapability],
    snapshot: CleanSourceSnapshot,
) -> PublishedArtifact:
    _require_active_governed_process(
        recheck_site_packages=True,
        recheck_source=True,
        require_mutation_role=True,
    )
    data = artifact_bytes(value)
    _validate_test_evidence_bytes(
        value,
        file_bytes=data,
        selected_method_freeze=selected_method_freeze,
        observed_runs=observed_runs,
        snapshot=snapshot,
    )
    _require_prepublication_current_source(snapshot)
    return _publish_write_once(
        TEST_EVIDENCE_CANONICAL_PATH.parent,
        TEST_EVIDENCE_CANONICAL_PATH.name,
        data,
        create_root=True,
        _publisher=_PUBLICATION_ISSUER,
    )


def validate_test_evidence(
    *,
    selected_method_freeze: ValidatedArtifactCapability,
    observed_runs: Sequence[ObservedTestRunCapability],
    snapshot: CleanSourceSnapshot,
) -> ValidatedArtifactCapability:
    """Issue test authority only from the exact canonical publication."""

    _require_active_governed_process(
        recheck_site_packages=True,
        recheck_source=True,
    )
    path = TEST_EVIDENCE_CANONICAL_PATH
    data = load_exact_artifact_bytes([path], declared_paths=[path])[os.fspath(path)]
    parsed = parse_artifact_bytes(
        data,
        ArtifactSpec(
            schema=ARTIFACT_SCHEMAS["test_evidence"],
            exact_fields=_TEST_EVIDENCE_FIELDS,
        ),
    )
    parsed, bindings = _validate_test_evidence_bytes(
        parsed,
        file_bytes=data,
        selected_method_freeze=selected_method_freeze,
        observed_runs=observed_runs,
        snapshot=snapshot,
    )
    return _validate_schema_artifact(
        parsed,
        file_bytes=data,
        schema=ARTIFACT_SCHEMAS["test_evidence"],
        exact_fields=_TEST_EVIDENCE_FIELDS,
        expected_bindings=bindings,
        canonical_path=path,
        scope="canonical-publication",
        _observer=_CAPABILITY_ISSUER,
    )


def reopen_test_evidence(
    *,
    selected_method_freeze: ValidatedArtifactCapability,
    snapshot: CleanSourceSnapshot,
) -> ValidatedArtifactCapability:
    """Reopen immutable all-pass evidence without recreating historical PIDs."""

    _require_active_governed_process(
        recheck_site_packages=True,
        recheck_source=True,
    )
    selected = require_validated_artifact_capability(
        selected_method_freeze,
        expected_schema=ARTIFACT_SCHEMAS["selected_method_freeze"],
    )
    snapshot = _require_clean_source_snapshot(snapshot)
    current = capture_clean_source_snapshot(V3_SOURCE_REPOSITORY)
    if not _same_source_snapshot(current, snapshot):
        raise AuthorityError("test-evidence reopen requires the current exact clean snapshot")
    _require_loaded_governance_source_at_snapshot(current)
    if (
        selected.source_commit != snapshot.identity.commit
        or selected.source_tree != snapshot.identity.tree
    ):
        raise AuthorityError("test-evidence snapshot differs from selected committed source")
    path = TEST_EVIDENCE_CANONICAL_PATH
    data = load_exact_artifact_bytes([path], declared_paths=[path])[os.fspath(path)]
    parsed = parse_artifact_bytes(
        data,
        ArtifactSpec(
            schema=ARTIFACT_SCHEMAS["test_evidence"],
            exact_fields=_TEST_EVIDENCE_FIELDS,
        ),
    )
    _validate_test_evidence_content(parsed)
    module_hash = next(
        (
            item.file_sha256
            for item in snapshot.tracked_files
            if item.path == "src/cfeg/metadata_calibration_v3_governance.py"
        ),
        None,
    )
    bindings = {
        "scientific_candidate_id": selected._validated_payload["scientific_candidate_id"],
        "selected_method_freeze_schema": selected.schema,
        "selected_method_freeze_payload_sha256": selected.payload_sha256,
        "selected_method_freeze_file_sha256": selected.file_sha256,
        "clean_commit": snapshot.identity.commit,
        "clean_tree": snapshot.identity.tree,
        "public_fixture_file_sha256": CANARY_FIXTURE_SHA256,
        "governance_module_file_sha256": module_hash,
    }
    if any(_plain_json(parsed[key]) != _plain_json(expected) for key, expected in bindings.items()):
        raise AuthorityError("durable test evidence differs from selected clean source")
    return _validate_schema_artifact(
        parsed,
        file_bytes=data,
        schema=ARTIFACT_SCHEMAS["test_evidence"],
        exact_fields=_TEST_EVIDENCE_FIELDS,
        expected_bindings=bindings,
        canonical_path=path,
        scope="canonical-publication",
        _observer=_CAPABILITY_ISSUER,
    )


def _artifact_reference_from_capability(
    value: DevelopmentBundleCapability | ValidatedArtifactCapability | CanaryArtifactCapability,
) -> _ArtifactReference:
    if type(value) is DevelopmentBundleCapability:
        require_development_bundle_capability(
            value,
            expected_commit=value.clean_commit,
            expected_tree=value.clean_tree,
        )
    elif type(value) is ValidatedArtifactCapability:
        require_validated_artifact_capability(value, expected_schema=value.schema)
    elif type(value) is CanaryArtifactCapability:
        require_canary_artifact_capability(value, expected_schema=value.schema)
    else:
        raise AuthorityError("validated file-backed capability required for artifact reference")
    return _ArtifactReference(
        schema=value.schema,
        payload_sha256=value.payload_sha256,
        file_sha256=value.file_sha256,
        _issuer=_CAPABILITY_ISSUER,
    )


@dataclass(frozen=True, slots=True, init=False)
class CanaryArtifactCapability:
    schema: str
    candidate_id: str
    payload_sha256: str
    file_sha256: str
    canonical_path: Path
    source_commit: str
    source_tree: str
    capability_seal_sha256: str
    _validated_payload: Mapping[str, Any] = field(repr=False, compare=False)
    _validation_marker: object = field(repr=False, compare=False)

    def __init__(
        self,
        *,
        schema: str,
        payload_sha256: str,
        file_sha256: str,
        validated_payload: Mapping[str, Any],
        canonical_path: Path,
        source_commit: str,
        source_tree: str,
        _issuer: object,
    ) -> None:
        if _issuer is not _CAPABILITY_ISSUER:
            raise AuthorityError("CanaryArtifactCapability has no valid issuer")
        object.__setattr__(self, "schema", schema)
        object.__setattr__(self, "candidate_id", CANARY_CANDIDATE_ID)
        object.__setattr__(self, "payload_sha256", payload_sha256)
        object.__setattr__(self, "file_sha256", file_sha256)
        object.__setattr__(self, "canonical_path", canonical_path)
        object.__setattr__(self, "source_commit", source_commit)
        object.__setattr__(self, "source_tree", source_tree)
        object.__setattr__(
            self,
            "capability_seal_sha256",
            _artifact_binding_sha256(
                {
                    "schema": schema,
                    "candidate_id": CANARY_CANDIDATE_ID,
                    "payload_sha256": payload_sha256,
                    "file_sha256": file_sha256,
                    "canonical_path": os.fspath(canonical_path),
                    "source_commit": source_commit,
                    "source_tree": source_tree,
                }
            ),
        )
        object.__setattr__(self, "_validated_payload", _freeze_json(validated_payload))
        object.__setattr__(self, "_validation_marker", _issuer)


def require_canary_artifact_capability(
    value: object,
    *,
    expected_schema: str,
) -> CanaryArtifactCapability:
    if type(value) is not CanaryArtifactCapability:
        raise AuthorityError("exact CanaryArtifactCapability required")
    capability = value
    if capability._validation_marker is not _CAPABILITY_ISSUER:
        raise AuthorityError("canary artifact capability marker is invalid")
    if capability.schema != expected_schema or capability.candidate_id != CANARY_CANDIDATE_ID:
        raise AuthorityError("canary artifact capability identity mismatch")
    expected_path = {
        ARTIFACT_SCHEMAS["canary_authorization"]: CANARY_CANONICAL_ROOT / "authorization.json",
        ARTIFACT_SCHEMAS["canary_claim"]: CANARY_CANONICAL_ROOT / "claim.json",
        ARTIFACT_SCHEMAS["canary_beacon"]: CANARY_CANONICAL_ROOT / "beacon.json",
        ARTIFACT_SCHEMAS["canary_result"]: CANARY_RESULT_CANONICAL_PATH,
    }.get(expected_schema)
    if expected_path is None or capability.canonical_path != expected_path:
        raise AuthorityError("canary artifact lacks exact canonical publication provenance")
    expected_seal = _artifact_binding_sha256(
        {
            "schema": capability.schema,
            "candidate_id": capability.candidate_id,
            "payload_sha256": capability.payload_sha256,
            "file_sha256": capability.file_sha256,
            "canonical_path": os.fspath(capability.canonical_path),
            "source_commit": capability.source_commit,
            "source_tree": capability.source_tree,
        }
    )
    if (
        validate_payload_hash(capability._validated_payload) != capability.payload_sha256
        or file_sha256(artifact_bytes(capability._validated_payload)) != capability.file_sha256
        or capability.capability_seal_sha256 != expected_seal
    ):
        raise AuthorityError("canary artifact capability payload hash mismatch")
    _require_sha256(capability.file_sha256, "canary artifact file_sha256")
    _require_git_object(capability.source_commit, "canary artifact source commit")
    _require_git_object(capability.source_tree, "canary artifact source tree")
    return capability


def _validate_canary_artifact_bytes(
    value: Mapping[str, Any],
    *,
    file_bytes: bytes,
    schema: str,
    exact_fields: frozenset[str],
) -> Mapping[str, Any]:
    parsed = parse_artifact_bytes(
        file_bytes,
        ArtifactSpec(schema=schema, exact_fields=exact_fields),
    )
    if canonical_json_bytes(parsed) != canonical_json_bytes(value):
        raise ValidationError("canary artifact value differs from supplied bytes")
    if parsed["candidate_id"] != CANARY_CANDIDATE_ID:
        raise ValidationError("canary artifact candidate mismatch")
    return parsed


def _issue_canary_artifact_from_canonical_path(
    path: Path,
    *,
    value: Mapping[str, Any],
    file_bytes: bytes,
    schema: str,
    exact_fields: frozenset[str],
    source_commit: str,
    source_tree: str,
    _observer: object,
) -> CanaryArtifactCapability:
    if _observer is not _CAPABILITY_ISSUER:
        raise AuthorityError("canary issuance requires exact canonical observation")
    _require_active_governed_process(
        recheck_site_packages=True,
        recheck_source=True,
    )
    _require_git_object(source_commit, "canary source commit")
    _require_git_object(source_tree, "canary source tree")
    parsed = _validate_canary_artifact_bytes(
        value,
        file_bytes=file_bytes,
        schema=schema,
        exact_fields=exact_fields,
    )
    expected_path = {
        ARTIFACT_SCHEMAS["canary_authorization"]: CANARY_CANONICAL_ROOT / "authorization.json",
        ARTIFACT_SCHEMAS["canary_claim"]: CANARY_CANONICAL_ROOT / "claim.json",
        ARTIFACT_SCHEMAS["canary_beacon"]: CANARY_CANONICAL_ROOT / "beacon.json",
        ARTIFACT_SCHEMAS["canary_result"]: CANARY_RESULT_CANONICAL_PATH,
    }.get(schema)
    if path != expected_path:
        raise AuthorityError("canary authority may be issued only from its canonical path")
    return CanaryArtifactCapability(
        schema=schema,
        payload_sha256=str(parsed["payload_sha256"]),
        file_sha256=file_sha256(file_bytes),
        validated_payload=parsed,
        canonical_path=path,
        source_commit=source_commit,
        source_tree=source_tree,
        _issuer=_CAPABILITY_ISSUER,
    )


@dataclass(frozen=True, slots=True)
class HistoricalCanaryClock:
    timestamp_UTC: str = CANARY_TIMESTAMP_UTC

    def __post_init__(self) -> None:
        if self.timestamp_UTC != CANARY_TIMESTAMP_UTC:
            raise ValidationError(
                "historical canary clock must bind the declared fixture timestamp"
            )


@dataclass(frozen=True, slots=True, init=False)
class LiveFutureBeaconClock:
    target_timestamp_UTC: str
    observed_at_UTC: str
    _validation_marker: object = field(repr=False, compare=False)

    def __init__(self, *_args: object, **_kwargs: object) -> None:
        raise AuthorityError(
            "no live future-beacon clock can be issued before an exact beacon selection"
        )


@dataclass(frozen=True, slots=True, init=False)
class CanarySeedCapability:
    schema: str
    candidate_id: str
    public_fixture_file_sha256: str
    timestamp_UTC: str
    chain_index: int
    pulse_index: int
    output_value: str
    seed_digest_sha256: str
    full_unsigned_256_bit_root_seed: int
    _validation_marker: object = field(repr=False, compare=False)

    def __init__(
        self,
        *,
        public_fixture_file_sha256: str,
        timestamp_UTC: str,
        chain_index: int,
        pulse_index: int,
        output_value: str,
        seed_digest_sha256: str,
        full_unsigned_256_bit_root_seed: int,
        _issuer: object,
    ) -> None:
        if _issuer is not _CAPABILITY_ISSUER:
            raise AuthorityError("CanarySeedCapability has no valid issuer")
        object.__setattr__(self, "schema", CANARY_SEED_CAPABILITY_SCHEMA)
        object.__setattr__(self, "candidate_id", CANARY_CANDIDATE_ID)
        object.__setattr__(self, "public_fixture_file_sha256", public_fixture_file_sha256)
        object.__setattr__(self, "timestamp_UTC", timestamp_UTC)
        object.__setattr__(self, "chain_index", chain_index)
        object.__setattr__(self, "pulse_index", pulse_index)
        object.__setattr__(self, "output_value", output_value)
        object.__setattr__(self, "seed_digest_sha256", seed_digest_sha256)
        object.__setattr__(self, "full_unsigned_256_bit_root_seed", full_unsigned_256_bit_root_seed)
        object.__setattr__(self, "_validation_marker", _issuer)


def _require_canary_seed_capability(value: object) -> CanarySeedCapability:
    if type(value) is not CanarySeedCapability:
        raise AuthorityError("exact CanarySeedCapability required")
    capability = value
    if capability._validation_marker is not _CAPABILITY_ISSUER:
        raise AuthorityError("canary seed capability marker is invalid")
    if (
        capability.schema != CANARY_SEED_CAPABILITY_SCHEMA
        or capability.candidate_id != CANARY_CANDIDATE_ID
        or capability.public_fixture_file_sha256 != CANARY_FIXTURE_SHA256
        or capability.timestamp_UTC != CANARY_TIMESTAMP_UTC
        or capability.chain_index != CANARY_CHAIN_INDEX
        or capability.pulse_index != CANARY_PULSE_INDEX
        or capability.output_value != CANARY_OUTPUT_VALUE
        or capability.seed_digest_sha256 != CANARY_SEED_DIGEST
        or capability.full_unsigned_256_bit_root_seed != CANARY_ROOT_SEED
    ):
        raise AuthorityError("canary seed capability bindings are invalid")
    return capability


def derive_canary_seed(
    *,
    public_fixture_file_sha256: str,
    timestamp_UTC: str,
    chain_index: int,
    pulse_index: int,
    output_value: str,
    clock: HistoricalCanaryClock,
) -> CanarySeedCapability:
    if type(clock) is not HistoricalCanaryClock or clock.timestamp_UTC != timestamp_UTC:
        raise AuthorityError("exact HistoricalCanaryClock required for the canary")
    if (
        public_fixture_file_sha256 != CANARY_FIXTURE_SHA256
        or timestamp_UTC != CANARY_TIMESTAMP_UTC
        or chain_index != CANARY_CHAIN_INDEX
        or pulse_index != CANARY_PULSE_INDEX
        or output_value != CANARY_OUTPUT_VALUE
    ):
        raise ValidationError("canary pulse differs from the frozen public fixture oracle")
    digest = hashlib.sha256(
        _ascii_lines(
            (
                CANARY_SEED_DOMAIN,
                public_fixture_file_sha256,
                timestamp_UTC,
                str(chain_index),
                str(pulse_index),
                output_value,
            )
        )
    ).hexdigest()
    root_seed = int.from_bytes(bytes.fromhex(digest), byteorder="big", signed=False)
    if digest != CANARY_SEED_DIGEST or root_seed != CANARY_ROOT_SEED:
        raise ValidationError("canary seed oracle mismatch")
    return CanarySeedCapability(
        public_fixture_file_sha256=public_fixture_file_sha256,
        timestamp_UTC=timestamp_UTC,
        chain_index=chain_index,
        pulse_index=pulse_index,
        output_value=output_value,
        seed_digest_sha256=digest,
        full_unsigned_256_bit_root_seed=root_seed,
        _issuer=_CAPABILITY_ISSUER,
    )


def validate_canary_fixture(
    fixture_bytes: bytes,
    *,
    clock: HistoricalCanaryClock,
) -> CanarySeedCapability:
    if file_sha256(fixture_bytes) != CANARY_FIXTURE_SHA256:
        raise ValidationError("historical canary fixture file hash mismatch")
    try:
        fixture = json.loads(fixture_bytes.decode("utf-8"))
        response = fixture["response"]
        pulse = response["pulse"]
    except (UnicodeDecodeError, json.JSONDecodeError, KeyError, TypeError) as exc:
        raise ValidationError("historical canary fixture structure is malformed") from exc
    if (
        not isinstance(fixture, dict)
        or not isinstance(response, dict)
        or not isinstance(pulse, dict)
    ):
        raise ValidationError("historical canary fixture objects are malformed")
    response_digest = hashlib.sha256(canonical_json_bytes(response)).hexdigest()
    if fixture.get("parsed_response_canonical_sha256") != response_digest:
        raise ValidationError("historical canary response canonical digest mismatch")
    return derive_canary_seed(
        public_fixture_file_sha256=CANARY_FIXTURE_SHA256,
        timestamp_UTC=pulse.get("timeStamp"),
        chain_index=pulse.get("chainIndex"),
        pulse_index=pulse.get("pulseIndex"),
        output_value=pulse.get("outputValue"),
        clock=clock,
    )


def reopen_historical_canary_seed(
    *,
    snapshot: CleanSourceSnapshot,
) -> CanarySeedCapability:
    """Load and validate the frozen canary fixture from exact committed B."""

    _require_active_governed_process(
        recheck_site_packages=True,
        recheck_source=True,
    )
    snapshot = _require_clean_source_snapshot(snapshot)
    current = capture_clean_source_snapshot(V3_SOURCE_REPOSITORY)
    if not _same_source_snapshot(current, snapshot):
        raise AuthorityError("canary fixture requires the exact current clean source")
    _require_loaded_governance_source_at_snapshot(current)
    fixture_bytes = _load_committed_source_blob(
        V3_SOURCE_REPOSITORY,
        CANARY_FIXTURE_REPOSITORY_PATH,
        snapshot=current,
        production=True,
    )
    return validate_canary_fixture(
        fixture_bytes,
        clock=HistoricalCanaryClock(),
    )


_CANARY_PREVALIDATION_TRACE = (
    CanaryState.CANARY_DECLARED,
    CanaryState.CANARY_AUTHORIZED,
    CanaryState.CANARY_CLAIMED,
    CanaryState.CANARY_BEACON_BOUND,
)
_CANARY_FULL_TRACE = tuple(CanaryState)


def canary_probe_digest(
    capability: CanarySeedCapability,
    *,
    prevalidation_trace: Sequence[CanaryState],
) -> str:
    _require_canary_seed_capability(capability)
    if tuple(prevalidation_trace) != _CANARY_PREVALIDATION_TRACE:
        raise TransitionError("canary probe requires the exact pre-validation state trace")
    trace = ",".join(state.value for state in prevalidation_trace)
    digest = hashlib.sha256(
        _ascii_lines((CANARY_PROBE_DOMAIN, capability.seed_digest_sha256, trace))
    ).hexdigest()
    if digest != CANARY_PROBE_DIGEST:
        raise ValidationError("canary probe oracle mismatch")
    return digest


_CANARY_AUTHORIZATION_FIELDS = frozenset(
    {
        "schema",
        "candidate_id",
        "scientific_candidate_id",
        "selected_method_freeze_schema",
        "selected_method_freeze_payload_sha256",
        "selected_method_freeze_file_sha256",
        "test_evidence_schema",
        "test_evidence_payload_sha256",
        "test_evidence_file_sha256",
        "clean_commit",
        "clean_tree",
        "public_fixture_file_sha256",
        "authorized_at_UTC",
        "payload_sha256",
    }
)
_CANARY_CLAIM_FIELDS = frozenset(
    {
        "schema",
        "candidate_id",
        "canary_authorization_schema",
        "canary_authorization_payload_sha256",
        "canary_authorization_file_sha256",
        "claimed_at_UTC",
        "payload_sha256",
    }
)
_CANARY_BEACON_FIELDS = frozenset(
    {
        "schema",
        "candidate_id",
        "canary_claim_schema",
        "canary_claim_payload_sha256",
        "canary_claim_file_sha256",
        "public_fixture_file_sha256",
        "timestamp_UTC",
        "chain_index",
        "pulse_index",
        "output_value",
        "payload_sha256",
    }
)
_CANARY_RESULT_FIELDS = frozenset(
    {
        "schema",
        "candidate_id",
        "canary_beacon_schema",
        "canary_beacon_payload_sha256",
        "canary_beacon_file_sha256",
        "seed_digest_sha256",
        "full_unsigned_256_bit_root_seed",
        "probe_digest_sha256",
        "exact_CANARY_state_trace_through_result",
        "publisher_process_id",
        "publisher_executable_file_sha256",
        "payload_sha256",
    }
)


def build_canary_authorization(
    *,
    selected_method_freeze: ValidatedArtifactCapability,
    test_evidence: ValidatedArtifactCapability,
    authorized_at_UTC: str,
) -> Mapping[str, Any]:
    selected = require_validated_artifact_capability(
        selected_method_freeze,
        expected_schema=ARTIFACT_SCHEMAS["selected_method_freeze"],
    )
    tests = require_validated_artifact_capability(
        test_evidence,
        expected_schema=ARTIFACT_SCHEMAS["test_evidence"],
    )
    authorized = _parse_rfc3339_seconds(authorized_at_UTC, "canary authorization time")
    tested = _parse_rfc3339_seconds(
        tests._validated_payload["recorded_at_UTC"],
        "test evidence recorded_at_UTC",
    )
    if not tested < authorized:
        raise ValidationError("canary authorization must be after full test evidence")
    selected_payload = selected._validated_payload
    if (
        tests._validated_payload["selected_method_freeze_payload_sha256"] != selected.payload_sha256
        or tests._validated_payload["selected_method_freeze_file_sha256"] != selected.file_sha256
        or tests._validated_payload["clean_commit"] != selected.source_commit
        or tests._validated_payload["clean_tree"] != selected.source_tree
    ):
        raise AuthorityError("canary test evidence is not bound to selected-method freeze")
    return seal_payload(
        {
            "schema": ARTIFACT_SCHEMAS["canary_authorization"],
            "candidate_id": CANARY_CANDIDATE_ID,
            "scientific_candidate_id": selected_payload["scientific_candidate_id"],
            "selected_method_freeze_schema": selected.schema,
            "selected_method_freeze_payload_sha256": selected.payload_sha256,
            "selected_method_freeze_file_sha256": selected.file_sha256,
            "test_evidence_schema": tests.schema,
            "test_evidence_payload_sha256": tests.payload_sha256,
            "test_evidence_file_sha256": tests.file_sha256,
            "clean_commit": selected.source_commit,
            "clean_tree": selected.source_tree,
            "public_fixture_file_sha256": CANARY_FIXTURE_SHA256,
            "authorized_at_UTC": authorized_at_UTC,
        }
    )


def _validate_canary_authorization_bytes(
    value: Mapping[str, Any],
    *,
    file_bytes: bytes,
    selected_method_freeze: ValidatedArtifactCapability,
    test_evidence: ValidatedArtifactCapability,
) -> Mapping[str, Any]:
    expected = build_canary_authorization(
        selected_method_freeze=selected_method_freeze,
        test_evidence=test_evidence,
        authorized_at_UTC=str(value.get("authorized_at_UTC")),
    )
    if canonical_json_bytes(value) != canonical_json_bytes(expected):
        raise ValidationError("canary authorization differs from exact reconstruction")
    return _validate_canary_artifact_bytes(
        value,
        file_bytes=file_bytes,
        schema=ARTIFACT_SCHEMAS["canary_authorization"],
        exact_fields=_CANARY_AUTHORIZATION_FIELDS,
    )


def publish_canary_authorization(
    value: Mapping[str, Any],
    *,
    selected_method_freeze: ValidatedArtifactCapability,
    test_evidence: ValidatedArtifactCapability,
) -> PublishedArtifact:
    _require_active_governed_process(
        recheck_site_packages=True,
        recheck_source=True,
        require_mutation_role=True,
    )
    data = artifact_bytes(value)
    _validate_canary_authorization_bytes(
        value,
        file_bytes=data,
        selected_method_freeze=selected_method_freeze,
        test_evidence=test_evidence,
    )
    selected = require_validated_artifact_capability(
        selected_method_freeze,
        expected_schema=ARTIFACT_SCHEMAS["selected_method_freeze"],
    )
    _require_current_source_identity(
        commit=str(selected.source_commit),
        tree=str(selected.source_tree),
    )
    return _publish_write_once(
        CANARY_CANONICAL_ROOT,
        "authorization.json",
        data,
        create_root=True,
        _publisher=_PUBLICATION_ISSUER,
    )


def validate_canary_authorization(
    *,
    selected_method_freeze: ValidatedArtifactCapability,
    test_evidence: ValidatedArtifactCapability,
) -> CanaryArtifactCapability:
    _require_active_governed_process(
        recheck_site_packages=True,
        recheck_source=True,
    )
    path = CANARY_CANONICAL_ROOT / "authorization.json"
    data = load_exact_artifact_bytes([path], declared_paths=[path])[os.fspath(path)]
    parsed = parse_artifact_bytes(
        data,
        ArtifactSpec(
            schema=ARTIFACT_SCHEMAS["canary_authorization"],
            exact_fields=_CANARY_AUTHORIZATION_FIELDS,
        ),
    )
    _validate_canary_authorization_bytes(
        parsed,
        file_bytes=data,
        selected_method_freeze=selected_method_freeze,
        test_evidence=test_evidence,
    )
    selected_method_freeze = require_validated_artifact_capability(
        selected_method_freeze,
        expected_schema=ARTIFACT_SCHEMAS["selected_method_freeze"],
    )
    _require_current_source_identity(
        commit=str(selected_method_freeze.source_commit),
        tree=str(selected_method_freeze.source_tree),
    )
    return _issue_canary_artifact_from_canonical_path(
        path,
        value=parsed,
        file_bytes=data,
        schema=ARTIFACT_SCHEMAS["canary_authorization"],
        exact_fields=_CANARY_AUTHORIZATION_FIELDS,
        source_commit=str(selected_method_freeze.source_commit),
        source_tree=str(selected_method_freeze.source_tree),
        _observer=_CAPABILITY_ISSUER,
    )


def build_canary_claim(
    authorization: CanaryArtifactCapability,
    *,
    claimed_at_UTC: str,
) -> Mapping[str, Any]:
    authorization = require_canary_artifact_capability(
        authorization,
        expected_schema=ARTIFACT_SCHEMAS["canary_authorization"],
    )
    claimed = _parse_rfc3339_seconds(claimed_at_UTC, "canary claim time")
    authorized = _parse_rfc3339_seconds(
        authorization._validated_payload["authorized_at_UTC"], "canary authorization time"
    )
    if not authorized < claimed:
        raise ValidationError("canary claim time must be after canary authorization")
    return seal_payload(
        {
            "schema": ARTIFACT_SCHEMAS["canary_claim"],
            "candidate_id": CANARY_CANDIDATE_ID,
            "canary_authorization_schema": authorization.schema,
            "canary_authorization_payload_sha256": authorization.payload_sha256,
            "canary_authorization_file_sha256": authorization.file_sha256,
            "claimed_at_UTC": claimed_at_UTC,
        }
    )


def _validate_canary_claim_bytes(
    value: Mapping[str, Any],
    *,
    file_bytes: bytes,
    authorization: CanaryArtifactCapability,
) -> Mapping[str, Any]:
    expected = build_canary_claim(
        authorization,
        claimed_at_UTC=str(value.get("claimed_at_UTC")),
    )
    if canonical_json_bytes(value) != canonical_json_bytes(expected):
        raise ValidationError("canary claim differs from exact reconstruction")
    return _validate_canary_artifact_bytes(
        value,
        file_bytes=file_bytes,
        schema=ARTIFACT_SCHEMAS["canary_claim"],
        exact_fields=_CANARY_CLAIM_FIELDS,
    )


def publish_canary_claim(
    value: Mapping[str, Any],
    *,
    authorization: CanaryArtifactCapability,
) -> PublishedArtifact:
    _require_active_governed_process(
        recheck_site_packages=True,
        recheck_source=True,
        require_mutation_role=True,
    )
    data = artifact_bytes(value)
    _validate_canary_claim_bytes(
        value,
        file_bytes=data,
        authorization=authorization,
    )
    authorization = require_canary_artifact_capability(
        authorization,
        expected_schema=ARTIFACT_SCHEMAS["canary_authorization"],
    )
    _require_current_source_identity(
        commit=authorization.source_commit,
        tree=authorization.source_tree,
    )
    return _publish_write_once(
        CANARY_CANONICAL_ROOT,
        "claim.json",
        data,
        create_root=False,
        _publisher=_PUBLICATION_ISSUER,
    )


def validate_canary_claim(
    *,
    authorization: CanaryArtifactCapability,
) -> CanaryArtifactCapability:
    _require_active_governed_process(
        recheck_site_packages=True,
        recheck_source=True,
    )
    path = CANARY_CANONICAL_ROOT / "claim.json"
    data = load_exact_artifact_bytes([path], declared_paths=[path])[os.fspath(path)]
    parsed = parse_artifact_bytes(
        data,
        ArtifactSpec(
            schema=ARTIFACT_SCHEMAS["canary_claim"],
            exact_fields=_CANARY_CLAIM_FIELDS,
        ),
    )
    _validate_canary_claim_bytes(parsed, file_bytes=data, authorization=authorization)
    authorization = require_canary_artifact_capability(
        authorization,
        expected_schema=ARTIFACT_SCHEMAS["canary_authorization"],
    )
    _require_current_source_identity(
        commit=authorization.source_commit,
        tree=authorization.source_tree,
    )
    return _issue_canary_artifact_from_canonical_path(
        path,
        value=parsed,
        file_bytes=data,
        schema=ARTIFACT_SCHEMAS["canary_claim"],
        exact_fields=_CANARY_CLAIM_FIELDS,
        source_commit=authorization.source_commit,
        source_tree=authorization.source_tree,
        _observer=_CAPABILITY_ISSUER,
    )


def build_canary_beacon(
    claim: CanaryArtifactCapability,
    seed: CanarySeedCapability,
) -> Mapping[str, Any]:
    claim = require_canary_artifact_capability(
        claim,
        expected_schema=ARTIFACT_SCHEMAS["canary_claim"],
    )
    seed = _require_canary_seed_capability(seed)
    return seal_payload(
        {
            "schema": ARTIFACT_SCHEMAS["canary_beacon"],
            "candidate_id": CANARY_CANDIDATE_ID,
            "canary_claim_schema": claim.schema,
            "canary_claim_payload_sha256": claim.payload_sha256,
            "canary_claim_file_sha256": claim.file_sha256,
            "public_fixture_file_sha256": seed.public_fixture_file_sha256,
            "timestamp_UTC": seed.timestamp_UTC,
            "chain_index": seed.chain_index,
            "pulse_index": seed.pulse_index,
            "output_value": seed.output_value,
        }
    )


def _validate_canary_beacon_bytes(
    value: Mapping[str, Any],
    *,
    file_bytes: bytes,
    claim: CanaryArtifactCapability,
    seed: CanarySeedCapability,
) -> Mapping[str, Any]:
    expected = build_canary_beacon(claim, seed)
    if canonical_json_bytes(value) != canonical_json_bytes(expected):
        raise ValidationError("canary beacon differs from exact reconstruction")
    return _validate_canary_artifact_bytes(
        value,
        file_bytes=file_bytes,
        schema=ARTIFACT_SCHEMAS["canary_beacon"],
        exact_fields=_CANARY_BEACON_FIELDS,
    )


def publish_canary_beacon(
    value: Mapping[str, Any],
    *,
    claim: CanaryArtifactCapability,
    seed: CanarySeedCapability,
) -> PublishedArtifact:
    _require_active_governed_process(
        recheck_site_packages=True,
        recheck_source=True,
        require_mutation_role=True,
    )
    data = artifact_bytes(value)
    _validate_canary_beacon_bytes(value, file_bytes=data, claim=claim, seed=seed)
    claim = require_canary_artifact_capability(
        claim,
        expected_schema=ARTIFACT_SCHEMAS["canary_claim"],
    )
    _require_current_source_identity(
        commit=claim.source_commit,
        tree=claim.source_tree,
    )
    return _publish_write_once(
        CANARY_CANONICAL_ROOT,
        "beacon.json",
        data,
        create_root=False,
        _publisher=_PUBLICATION_ISSUER,
    )


def validate_canary_beacon(
    *,
    claim: CanaryArtifactCapability,
    seed: CanarySeedCapability,
) -> CanaryArtifactCapability:
    _require_active_governed_process(
        recheck_site_packages=True,
        recheck_source=True,
    )
    path = CANARY_CANONICAL_ROOT / "beacon.json"
    data = load_exact_artifact_bytes([path], declared_paths=[path])[os.fspath(path)]
    parsed = parse_artifact_bytes(
        data,
        ArtifactSpec(
            schema=ARTIFACT_SCHEMAS["canary_beacon"],
            exact_fields=_CANARY_BEACON_FIELDS,
        ),
    )
    _validate_canary_beacon_bytes(parsed, file_bytes=data, claim=claim, seed=seed)
    claim = require_canary_artifact_capability(
        claim,
        expected_schema=ARTIFACT_SCHEMAS["canary_claim"],
    )
    _require_current_source_identity(
        commit=claim.source_commit,
        tree=claim.source_tree,
    )
    return _issue_canary_artifact_from_canonical_path(
        path,
        value=parsed,
        file_bytes=data,
        schema=ARTIFACT_SCHEMAS["canary_beacon"],
        exact_fields=_CANARY_BEACON_FIELDS,
        source_commit=claim.source_commit,
        source_tree=claim.source_tree,
        _observer=_CAPABILITY_ISSUER,
    )


def _canary_result_payload(
    beacon: CanaryArtifactCapability,
    seed: CanarySeedCapability,
    *,
    probe_digest_sha256: str,
    publisher_process_id: int,
    publisher_executable_file_sha256: str,
) -> Mapping[str, Any]:
    beacon = require_canary_artifact_capability(
        beacon,
        expected_schema=ARTIFACT_SCHEMAS["canary_beacon"],
    )
    seed = _require_canary_seed_capability(seed)
    if probe_digest_sha256 != CANARY_PROBE_DIGEST:
        raise ValidationError("canary result probe digest differs from the oracle")
    if (
        not isinstance(publisher_process_id, int)
        or isinstance(publisher_process_id, bool)
        or publisher_process_id <= 0
    ):
        raise ValidationError("canary publisher process ID must be a positive integer")
    _require_sha256(
        publisher_executable_file_sha256,
        "canary publisher executable file_sha256",
    )
    return seal_payload(
        {
            "schema": ARTIFACT_SCHEMAS["canary_result"],
            "candidate_id": CANARY_CANDIDATE_ID,
            "canary_beacon_schema": beacon.schema,
            "canary_beacon_payload_sha256": beacon.payload_sha256,
            "canary_beacon_file_sha256": beacon.file_sha256,
            "seed_digest_sha256": seed.seed_digest_sha256,
            "full_unsigned_256_bit_root_seed": seed.full_unsigned_256_bit_root_seed,
            "probe_digest_sha256": probe_digest_sha256,
            "exact_CANARY_state_trace_through_result": [
                state.value for state in tuple(CanaryState)[:6]
            ],
            "publisher_process_id": publisher_process_id,
            "publisher_executable_file_sha256": publisher_executable_file_sha256,
        }
    )


def build_canary_result(
    beacon: CanaryArtifactCapability,
    seed: CanarySeedCapability,
    *,
    probe_digest_sha256: str,
) -> Mapping[str, Any]:
    """Build a result bound to the actual current publisher process."""

    return _canary_result_payload(
        beacon,
        seed,
        probe_digest_sha256=probe_digest_sha256,
        publisher_process_id=os.getpid(),
        publisher_executable_file_sha256=_current_executable_file_sha256(),
    )


def _validate_canary_result_bytes(
    value: Mapping[str, Any],
    *,
    file_bytes: bytes,
    beacon: CanaryArtifactCapability,
    seed: CanarySeedCapability,
    require_current_publisher: bool,
    require_live_publisher: bool,
) -> Mapping[str, Any]:
    publisher_process_id = value.get("publisher_process_id")
    if require_current_publisher and publisher_process_id != os.getpid():
        raise AuthorityError("canary result must be validated by its actual publisher process")
    recorded_executable = _require_sha256(
        value.get("publisher_executable_file_sha256"),
        "canary result publisher executable",
    )
    if require_live_publisher:
        publisher_executable = _process_executable_file_sha256(publisher_process_id)
        if recorded_executable != publisher_executable:
            raise AuthorityError("canary result publisher executable observation mismatch")
    expected = _canary_result_payload(
        beacon,
        seed,
        probe_digest_sha256=str(value.get("probe_digest_sha256")),
        publisher_process_id=publisher_process_id,
        publisher_executable_file_sha256=recorded_executable,
    )
    if canonical_json_bytes(value) != canonical_json_bytes(expected):
        raise ValidationError("canary result differs from exact reconstruction")
    return _validate_canary_artifact_bytes(
        value,
        file_bytes=file_bytes,
        schema=ARTIFACT_SCHEMAS["canary_result"],
        exact_fields=_CANARY_RESULT_FIELDS,
    )


@dataclass(frozen=True, slots=True, init=False)
class PublishedCanaryResultCapability:
    artifact: CanaryArtifactCapability
    path: Path
    publisher_process_id: int
    publisher_executable_file_sha256: str
    publisher_liveness_observed: bool
    payload_sha256: str
    file_sha256: str
    _validation_marker: object = field(repr=False, compare=False)

    def __init__(
        self,
        *,
        artifact: CanaryArtifactCapability,
        path: Path,
        publisher_process_id: int,
        publisher_liveness_observed: bool,
        _issuer: object,
    ) -> None:
        if _issuer is not _CAPABILITY_ISSUER:
            raise AuthorityError("PublishedCanaryResultCapability has no valid issuer")
        object.__setattr__(self, "artifact", artifact)
        object.__setattr__(self, "path", path)
        object.__setattr__(self, "publisher_process_id", publisher_process_id)
        object.__setattr__(
            self,
            "publisher_executable_file_sha256",
            str(artifact._validated_payload["publisher_executable_file_sha256"]),
        )
        object.__setattr__(self, "publisher_liveness_observed", publisher_liveness_observed)
        object.__setattr__(self, "payload_sha256", artifact.payload_sha256)
        object.__setattr__(self, "file_sha256", artifact.file_sha256)
        object.__setattr__(self, "_validation_marker", _issuer)


def require_published_canary_result_capability(
    value: object,
) -> PublishedCanaryResultCapability:
    if type(value) is not PublishedCanaryResultCapability:
        raise AuthorityError("exact PublishedCanaryResultCapability required")
    capability = value
    if capability._validation_marker is not _CAPABILITY_ISSUER:
        raise AuthorityError("published canary-result marker is invalid")
    artifact = require_canary_artifact_capability(
        capability.artifact,
        expected_schema=ARTIFACT_SCHEMAS["canary_result"],
    )
    if (
        capability.payload_sha256 != artifact.payload_sha256
        or capability.file_sha256 != artifact.file_sha256
        or capability.publisher_process_id != artifact._validated_payload["publisher_process_id"]
        or capability.publisher_executable_file_sha256
        != artifact._validated_payload["publisher_executable_file_sha256"]
        or capability.path != CANARY_RESULT_CANONICAL_PATH
        or not isinstance(capability.publisher_liveness_observed, bool)
    ):
        raise AuthorityError("published canary-result capability binding mismatch")
    if capability.publisher_liveness_observed and (
        capability.publisher_executable_file_sha256
        != _process_executable_file_sha256(capability.publisher_process_id)
    ):
        raise AuthorityError("live canary publisher executable changed")
    return capability


def publish_canary_result(
    *,
    beacon: CanaryArtifactCapability,
    seed: CanarySeedCapability,
    probe_digest_sha256: str,
) -> PublishedCanaryResultCapability:
    """Publish only at the frozen canonical canary result path."""

    _require_active_governed_process(
        recheck_site_packages=True,
        recheck_source=True,
        require_mutation_role=True,
    )
    value = build_canary_result(
        beacon,
        seed,
        probe_digest_sha256=probe_digest_sha256,
    )
    data = artifact_bytes(value)
    _validate_canary_result_bytes(
        value,
        file_bytes=data,
        beacon=beacon,
        seed=seed,
        require_current_publisher=True,
        require_live_publisher=True,
    )
    beacon = require_canary_artifact_capability(
        beacon,
        expected_schema=ARTIFACT_SCHEMAS["canary_beacon"],
    )
    _require_current_source_identity(
        commit=beacon.source_commit,
        tree=beacon.source_tree,
    )
    _publish_write_once(
        CANARY_CANONICAL_ROOT,
        "result.json",
        data,
        create_root=False,
        _publisher=_PUBLICATION_ISSUER,
    )
    return _observe_canary_result_at_path(
        CANARY_RESULT_CANONICAL_PATH,
        beacon=beacon,
        seed=seed,
        require_live_publisher=True,
    )


def _publish_canary_result_under_test_root(
    test_root: Path,
    *,
    beacon: CanaryArtifactCapability,
    seed: CanarySeedCapability,
    probe_digest_sha256: str,
) -> TestOnlyArtifactReceipt:
    """Internal tmp-path publication; it deliberately cannot advance canary state."""

    value = build_canary_result(beacon, seed, probe_digest_sha256=probe_digest_sha256)
    data = artifact_bytes(value)
    _validate_canary_result_bytes(
        value,
        file_bytes=data,
        beacon=beacon,
        seed=seed,
        require_current_publisher=True,
        require_live_publisher=True,
    )
    published = _publish_write_once_under_test_root(
        test_root,
        "result.json",
        data,
        create_root=True,
    )
    return TestOnlyArtifactReceipt(
        schema=ARTIFACT_SCHEMAS["canary_result"],
        payload_sha256=str(value["payload_sha256"]),
        file_sha256=published.file_sha256,
        path=published.path,
        _issuer=_CAPABILITY_ISSUER,
    )


def _observe_canary_result_at_path(
    path: Path,
    *,
    beacon: CanaryArtifactCapability,
    seed: CanarySeedCapability,
    require_live_publisher: bool,
) -> PublishedCanaryResultCapability:
    if path != CANARY_RESULT_CANONICAL_PATH:
        raise AuthorityError("published canary result authority requires canonical path")
    beacon = require_canary_artifact_capability(
        beacon,
        expected_schema=ARTIFACT_SCHEMAS["canary_beacon"],
    )
    _require_current_source_identity(
        commit=beacon.source_commit,
        tree=beacon.source_tree,
    )
    loaded = load_exact_artifact_bytes([path], declared_paths=[path])
    data = loaded[os.fspath(path)]
    parsed = parse_artifact_bytes(
        data,
        ArtifactSpec(
            schema=ARTIFACT_SCHEMAS["canary_result"],
            exact_fields=_CANARY_RESULT_FIELDS,
        ),
    )
    expected = _canary_result_payload(
        beacon,
        seed,
        probe_digest_sha256=str(parsed["probe_digest_sha256"]),
        publisher_process_id=parsed["publisher_process_id"],
        publisher_executable_file_sha256=str(parsed["publisher_executable_file_sha256"]),
    )
    if canonical_json_bytes(parsed) != canonical_json_bytes(expected):
        raise ValidationError("published canary result differs from exact reconstruction")
    parsed = _validate_canary_result_bytes(
        parsed,
        file_bytes=data,
        beacon=beacon,
        seed=seed,
        require_current_publisher=False,
        require_live_publisher=require_live_publisher,
    )
    artifact = _issue_canary_artifact_from_canonical_path(
        path,
        value=parsed,
        file_bytes=data,
        schema=ARTIFACT_SCHEMAS["canary_result"],
        exact_fields=_CANARY_RESULT_FIELDS,
        source_commit=beacon.source_commit,
        source_tree=beacon.source_tree,
        _observer=_CAPABILITY_ISSUER,
    )
    return PublishedCanaryResultCapability(
        artifact=artifact,
        path=path,
        publisher_process_id=parsed["publisher_process_id"],
        publisher_liveness_observed=require_live_publisher,
        _issuer=_CAPABILITY_ISSUER,
    )


def observe_published_canary_result(
    *,
    beacon: CanaryArtifactCapability,
    seed: CanarySeedCapability,
) -> PublishedCanaryResultCapability:
    _require_active_governed_process(
        recheck_site_packages=True,
        recheck_source=True,
    )
    return _observe_canary_result_at_path(
        CANARY_RESULT_CANONICAL_PATH,
        beacon=beacon,
        seed=seed,
        require_live_publisher=True,
    )


def reopen_published_canary_result(
    *,
    beacon: CanaryArtifactCapability,
    seed: CanarySeedCapability,
) -> PublishedCanaryResultCapability:
    """Reopen a durable result even when its validated publisher has exited."""

    _require_active_governed_process(
        recheck_site_packages=True,
        recheck_source=True,
    )
    return _observe_canary_result_at_path(
        CANARY_RESULT_CANONICAL_PATH,
        beacon=beacon,
        seed=seed,
        require_live_publisher=False,
    )


@dataclass(frozen=True, slots=True)
class _CanonicalCanaryGraph:
    snapshot: CleanSourceSnapshot
    development_bundle: DevelopmentBundleCapability
    recovery: BundleSelectedDeltaRecoveryCapability
    context_reference: ValidatedArtifactCapability
    development_result: ValidatedArtifactCapability
    selected_method_freeze: ValidatedArtifactCapability
    test_evidence: ValidatedArtifactCapability
    canary_authorization: CanaryArtifactCapability
    canary_claim: CanaryArtifactCapability
    canary_beacon: CanaryArtifactCapability
    published_canary_result: PublishedCanaryResultCapability
    canary_seed: CanarySeedCapability
    public_fixture_bytes: bytes = field(repr=False)


def _reopen_canonical_canary_graph() -> _CanonicalCanaryGraph:
    """Reconstruct the complete durable graph solely from fixed disk/Git roots."""

    _require_frozen_numerical_executor_environment()
    snapshot = capture_clean_source_snapshot(V3_SOURCE_REPOSITORY)
    development_bundle = reopen_development_bundle()
    recovery = validate_bundle_selected_delta_recovery(
        development_bundle=development_bundle,
        snapshot=snapshot,
    )
    context_reference = reopen_context_reference(
        development_bundle=development_bundle,
        recovery_capability=recovery,
    )
    development_result = reopen_development_result(
        development_bundle=development_bundle,
        context_reference=context_reference,
        recovery_capability=recovery,
    )
    selected_method_freeze = reopen_selected_method_freeze(
        development_bundle=development_bundle,
        development_result=development_result,
        recovery_capability=recovery,
    )
    test_evidence = reopen_test_evidence(
        selected_method_freeze=selected_method_freeze,
        snapshot=snapshot,
    )
    canary_authorization = validate_canary_authorization(
        selected_method_freeze=selected_method_freeze,
        test_evidence=test_evidence,
    )
    canary_claim = validate_canary_claim(authorization=canary_authorization)
    public_fixture_bytes = _load_committed_source_blob(
        V3_SOURCE_REPOSITORY,
        CANARY_FIXTURE_REPOSITORY_PATH,
        snapshot=snapshot,
        production=True,
    )
    canary_seed = validate_canary_fixture(
        public_fixture_bytes,
        clock=HistoricalCanaryClock(),
    )
    canary_beacon = validate_canary_beacon(claim=canary_claim, seed=canary_seed)
    published_canary_result = reopen_published_canary_result(
        beacon=canary_beacon,
        seed=canary_seed,
    )
    return _CanonicalCanaryGraph(
        snapshot=snapshot,
        development_bundle=development_bundle,
        recovery=recovery,
        context_reference=context_reference,
        development_result=development_result,
        selected_method_freeze=selected_method_freeze,
        test_evidence=test_evidence,
        canary_authorization=canary_authorization,
        canary_claim=canary_claim,
        canary_beacon=canary_beacon,
        published_canary_result=published_canary_result,
        canary_seed=canary_seed,
        public_fixture_bytes=public_fixture_bytes,
    )


_FRESH_AUDIT_REFERENCE_NAMES = (
    "selected_method_freeze",
    "test_evidence",
    "canary_authorization",
    "canary_claim",
    "canary_beacon",
    "canary_result",
)
_FRESH_AUDIT_REFERENCE_SCHEMAS = MappingProxyType(
    {
        "selected_method_freeze": ARTIFACT_SCHEMAS["selected_method_freeze"],
        "test_evidence": ARTIFACT_SCHEMAS["test_evidence"],
        "canary_authorization": ARTIFACT_SCHEMAS["canary_authorization"],
        "canary_claim": ARTIFACT_SCHEMAS["canary_claim"],
        "canary_beacon": ARTIFACT_SCHEMAS["canary_beacon"],
        "canary_result": ARTIFACT_SCHEMAS["canary_result"],
    }
)
_FRESH_AUDIT_CANONICAL_PATHS = MappingProxyType(
    {
        "selected_method_freeze": V3_SOURCE_REPOSITORY / SELECTED_METHOD_FREEZE_REPOSITORY_PATH,
        "test_evidence": CANARY_CANONICAL_ROOT / "test-evidence.json",
        "canary_authorization": CANARY_CANONICAL_ROOT / "authorization.json",
        "canary_claim": CANARY_CANONICAL_ROOT / "claim.json",
        "canary_beacon": CANARY_CANONICAL_ROOT / "beacon.json",
        "canary_result": CANARY_RESULT_CANONICAL_PATH,
    }
)
_FRESH_AUDIT_FIELDS = frozenset(
    {
        "schema",
        "candidate_id",
        "clean_commit",
        "clean_tree",
        "public_fixture_file_sha256",
        "exact_CANARY_state_trace",
        "expected_and_observed_seed_digest",
        "expected_and_observed_probe_digest",
        "publisher_process_id",
        "distinct_auditor_process_id",
        "auditor_executable_file_sha256",
        "payload_sha256",
        *(
            f"{name}_{suffix}"
            for name in _FRESH_AUDIT_REFERENCE_NAMES
            for suffix in ("schema", "payload_sha256", "file_sha256")
        ),
    }
)


@dataclass(frozen=True, slots=True, init=False)
class FreshCanaryAuditCapability:
    schema: str
    candidate_id: str
    payload_sha256: str
    file_sha256: str
    clean_commit: str
    clean_tree: str
    exact_state_trace: tuple[CanaryState, ...]
    selected_method_freeze_payload_sha256: str
    selected_method_freeze_file_sha256: str
    test_evidence_payload_sha256: str
    test_evidence_file_sha256: str
    canary_result_payload_sha256: str
    canary_result_file_sha256: str
    publisher_process_id: int
    auditor_process_id: int
    auditor_executable_file_sha256: str
    auditor_liveness_observed: bool
    scope: str
    capability_seal_sha256: str
    _validated_payload: Mapping[str, Any] = field(repr=False, compare=False)
    _validation_marker: object = field(repr=False, compare=False)

    def __init__(
        self,
        *,
        payload_sha256: str,
        file_sha256: str,
        clean_commit: str,
        clean_tree: str,
        exact_state_trace: tuple[CanaryState, ...],
        selected_method_freeze_payload_sha256: str,
        selected_method_freeze_file_sha256: str,
        test_evidence_payload_sha256: str,
        test_evidence_file_sha256: str,
        canary_result_payload_sha256: str,
        canary_result_file_sha256: str,
        publisher_process_id: int,
        auditor_process_id: int,
        auditor_executable_file_sha256: str,
        auditor_liveness_observed: bool,
        scope: str,
        validated_payload: Mapping[str, Any],
        _issuer: object,
    ) -> None:
        if _issuer is not _CAPABILITY_ISSUER:
            raise AuthorityError("FreshCanaryAuditCapability has no valid issuer")
        object.__setattr__(self, "schema", FRESH_CANARY_AUDIT_CAPABILITY_SCHEMA)
        object.__setattr__(self, "candidate_id", CANARY_CANDIDATE_ID)
        object.__setattr__(self, "payload_sha256", payload_sha256)
        object.__setattr__(self, "file_sha256", file_sha256)
        object.__setattr__(self, "clean_commit", clean_commit)
        object.__setattr__(self, "clean_tree", clean_tree)
        object.__setattr__(self, "exact_state_trace", exact_state_trace)
        object.__setattr__(
            self,
            "selected_method_freeze_payload_sha256",
            selected_method_freeze_payload_sha256,
        )
        object.__setattr__(
            self, "selected_method_freeze_file_sha256", selected_method_freeze_file_sha256
        )
        object.__setattr__(self, "test_evidence_payload_sha256", test_evidence_payload_sha256)
        object.__setattr__(self, "test_evidence_file_sha256", test_evidence_file_sha256)
        object.__setattr__(self, "canary_result_payload_sha256", canary_result_payload_sha256)
        object.__setattr__(self, "canary_result_file_sha256", canary_result_file_sha256)
        object.__setattr__(self, "publisher_process_id", publisher_process_id)
        object.__setattr__(self, "auditor_process_id", auditor_process_id)
        object.__setattr__(
            self,
            "auditor_executable_file_sha256",
            auditor_executable_file_sha256,
        )
        object.__setattr__(
            self,
            "auditor_liveness_observed",
            auditor_liveness_observed,
        )
        object.__setattr__(self, "scope", scope)
        seal = {
            "schema": FRESH_CANARY_AUDIT_CAPABILITY_SCHEMA,
            "candidate_id": CANARY_CANDIDATE_ID,
            "payload_sha256": payload_sha256,
            "file_sha256": file_sha256,
            "clean_commit": clean_commit,
            "clean_tree": clean_tree,
            "exact_state_trace": [state.value for state in exact_state_trace],
            "selected_method_freeze_payload_sha256": (selected_method_freeze_payload_sha256),
            "selected_method_freeze_file_sha256": selected_method_freeze_file_sha256,
            "test_evidence_payload_sha256": test_evidence_payload_sha256,
            "test_evidence_file_sha256": test_evidence_file_sha256,
            "canary_result_payload_sha256": canary_result_payload_sha256,
            "canary_result_file_sha256": canary_result_file_sha256,
            "publisher_process_id": publisher_process_id,
            "auditor_process_id": auditor_process_id,
            "auditor_executable_file_sha256": auditor_executable_file_sha256,
            "auditor_liveness_observed": auditor_liveness_observed,
            "scope": scope,
        }
        object.__setattr__(
            self,
            "capability_seal_sha256",
            _artifact_binding_sha256(seal),
        )
        object.__setattr__(self, "_validated_payload", _freeze_json(validated_payload))
        object.__setattr__(self, "_validation_marker", _issuer)


_FRESH_CANARY_EXEC_RECEIPT_FIELDS = frozenset(
    {
        "schema",
        "candidate_id",
        "auditor_process_id",
        "auditor_executable_file_sha256",
        "exact_orig_argv_sha256",
        "clean_commit",
        "clean_tree",
        "source_bundle_sha256",
        "fresh_audit_payload_sha256",
        "fresh_audit_file_sha256",
        "fresh_audit_file_base64",
        "payload_sha256",
    }
)


@dataclass(frozen=True, slots=True, init=False)
class ObservedFreshCanaryExecCapability:
    """Ephemeral parent observation of the one pinned isolated audit child."""

    schema: str
    candidate_id: str
    auditor_process_id: int
    auditor_executable_file_sha256: str
    exact_orig_argv_sha256: str
    clean_commit: str
    clean_tree: str
    source_bundle_sha256: str
    fresh_audit_payload_sha256: str
    fresh_audit_file_sha256: str
    receipt_payload_sha256: str
    capability_seal_sha256: str
    _fresh_audit_file_bytes: bytes = field(repr=False, compare=False)
    _process: subprocess.Popen[bytes] = field(repr=False, compare=False)
    _validation_marker: object = field(repr=False, compare=False)

    def __new__(  # noqa: PYI034 -- Python 3.10 does not provide typing.Self.
        cls,
        *_args: object,
        **_kwargs: object,
    ) -> ObservedFreshCanaryExecCapability:
        raise TypeError(
            "ObservedFreshCanaryExecCapability is issued only by the pinned parent launcher"
        )


def _fresh_exec_command_sha256() -> str:
    return hashlib.sha256(
        canonical_json_bytes({"orig_argv": list(_FRESH_CANARY_EXEC_COMMAND)})
    ).hexdigest()


def _canonical_bundle_python_site_inventory_sha256() -> str:
    """Read only the sealed bundle's stdlib-verifiable pre-import site digest."""

    path = Path(DEVELOPMENT_BUNDLE_CANONICAL_PATH)
    file_bytes = load_exact_artifact_bytes([path], declared_paths=[path])[os.fspath(path)]
    parsed = parse_artifact_bytes(
        file_bytes,
        ArtifactSpec(
            schema=DEVELOPMENT_BUNDLE_SCHEMA,
            exact_fields=_DEVELOPMENT_BUNDLE_FIELDS,
        ),
    )
    runtime_inventory = parsed.get("numerical_runtime_inventory")
    if not isinstance(runtime_inventory, Mapping):
        raise AuthorityError("canonical bundle runtime inventory is malformed")
    claimed_fingerprint = parsed.get("numerical_runtime_fingerprint_sha256")
    _require_sha256(claimed_fingerprint, "canonical bundle runtime fingerprint")
    if numerical_runtime_fingerprint_sha256(runtime_inventory) != claimed_fingerprint:
        raise AuthorityError("canonical bundle runtime inventory hash is invalid")
    return _python_site_inventory_sha256(runtime_inventory)


def _assert_fresh_canary_exec_child_context() -> None:
    """Reject direct calls, forks, non-isolated imports, and variable child argv."""

    if sys.flags.isolated != 1 or sys.flags.dont_write_bytecode != 1 or sys.flags.no_site != 1:
        raise AuthorityError("fresh audit child requires the exact -I -B -S bootstrap")
    if tuple(getattr(sys, "orig_argv", ())) != _FRESH_CANARY_EXEC_COMMAND:
        raise AuthorityError("fresh audit child command line differs from the frozen role")
    if tuple(sys.argv) != ("-c", _FRESH_CANARY_EXEC_PHASE):
        raise AuthorityError("fresh audit child received an unexpected argument")
    if _process_command_line(os.getpid()) != _FRESH_CANARY_EXEC_COMMAND:
        raise AuthorityError("fresh audit child /proc command line mismatch")
    if _GOVERNANCE_IMPORT_SOURCE_PATH != (
        V3_SOURCE_REPOSITORY / "src/cfeg/metadata_calibration_v3_governance.py"
    ):
        raise AuthorityError("fresh audit child imported governance outside canonical source")
    if _current_executable_file_sha256() != _hash_installed_regular_file(
        V3_PYTHON_EXECUTABLE.resolve(strict=True)
    ):
        raise AuthorityError("fresh audit child did not execute the pinned Python binary")
    if dict(os.environ) != dict(_OBSERVED_TEST_ENVIRONMENT):
        raise AuthorityError("fresh audit child environment differs from the exact allowlist")
    expected_site_inventory_sha256 = _canonical_bundle_python_site_inventory_sha256()
    process = establish_governed_process(
        expected_python_site_inventory_sha256=expected_site_inventory_sha256,
    )
    require_governed_process_capability(
        process,
        expected_role="fresh_canary_audit",
    )


def _build_fresh_exec_receipt(
    *,
    graph: _CanonicalCanaryGraph,
    fresh_audit_file_bytes: bytes,
) -> Mapping[str, Any]:
    parsed = parse_artifact_bytes(
        fresh_audit_file_bytes,
        ArtifactSpec(
            schema=FRESH_CANARY_AUDIT_SCHEMA,
            exact_fields=_FRESH_AUDIT_FIELDS,
        ),
    )
    return seal_payload(
        {
            "schema": FRESH_CANARY_EXEC_RECEIPT_SCHEMA,
            "candidate_id": CANARY_CANDIDATE_ID,
            "auditor_process_id": os.getpid(),
            "auditor_executable_file_sha256": _current_executable_file_sha256(),
            "exact_orig_argv_sha256": _fresh_exec_command_sha256(),
            "clean_commit": graph.snapshot.identity.commit,
            "clean_tree": graph.snapshot.identity.tree,
            "source_bundle_sha256": graph.snapshot.source_bundle_sha256,
            "fresh_audit_payload_sha256": parsed["payload_sha256"],
            "fresh_audit_file_sha256": file_sha256(fresh_audit_file_bytes),
            "fresh_audit_file_base64": base64.b64encode(fresh_audit_file_bytes).decode("ascii"),
        }
    )


def _issue_observed_fresh_exec_capability(
    *,
    process: subprocess.Popen[bytes],
    receipt_bytes: bytes,
    snapshot: CleanSourceSnapshot,
    _observer: object,
) -> ObservedFreshCanaryExecCapability:
    if _observer is not _FRESH_EXEC_OBSERVER_ISSUER:
        raise AuthorityError("fresh exec issuance requires the pinned parent observer")
    _require_active_governed_process(
        recheck_site_packages=True,
        recheck_source=True,
        require_mutation_role=True,
    )
    snapshot = _require_clean_source_snapshot(snapshot)
    if process.pid == os.getpid() or process.poll() is not None:
        raise AuthorityError("fresh audit child must be a distinct live process")
    if tuple(process.args) != _FRESH_CANARY_EXEC_COMMAND:
        raise AuthorityError("fresh audit Popen arguments differ from the frozen command")
    if _process_command_line(process.pid) != _FRESH_CANARY_EXEC_COMMAND:
        raise AuthorityError("fresh audit child command line observation mismatch")
    executable_sha256 = _process_executable_file_sha256(process.pid)
    expected_executable_sha256 = _hash_installed_regular_file(
        V3_PYTHON_EXECUTABLE.resolve(strict=True)
    )
    if executable_sha256 != expected_executable_sha256:
        raise AuthorityError("fresh audit child executable differs from pinned Python")
    receipt = parse_artifact_bytes(
        receipt_bytes,
        ArtifactSpec(
            schema=FRESH_CANARY_EXEC_RECEIPT_SCHEMA,
            exact_fields=_FRESH_CANARY_EXEC_RECEIPT_FIELDS,
        ),
    )
    if receipt["candidate_id"] != CANARY_CANDIDATE_ID:
        raise AuthorityError("fresh audit exec receipt candidate mismatch")
    try:
        audit_bytes = base64.b64decode(
            receipt["fresh_audit_file_base64"],
            validate=True,
        )
    except (ValueError, binascii.Error) as exc:
        raise ValidationError("fresh audit exec receipt base64 is invalid") from exc
    parsed_audit = parse_artifact_bytes(
        audit_bytes,
        ArtifactSpec(
            schema=FRESH_CANARY_AUDIT_SCHEMA,
            exact_fields=_FRESH_AUDIT_FIELDS,
        ),
    )
    if (
        receipt["auditor_process_id"] != process.pid
        or receipt["auditor_executable_file_sha256"] != executable_sha256
        or receipt["exact_orig_argv_sha256"] != _fresh_exec_command_sha256()
        or receipt["clean_commit"] != snapshot.identity.commit
        or receipt["clean_tree"] != snapshot.identity.tree
        or receipt["source_bundle_sha256"] != snapshot.source_bundle_sha256
        or receipt["fresh_audit_payload_sha256"] != parsed_audit["payload_sha256"]
        or receipt["fresh_audit_file_sha256"] != file_sha256(audit_bytes)
        or parsed_audit["distinct_auditor_process_id"] != process.pid
        or parsed_audit["auditor_executable_file_sha256"] != executable_sha256
        or parsed_audit["clean_commit"] != snapshot.identity.commit
        or parsed_audit["clean_tree"] != snapshot.identity.tree
    ):
        raise AuthorityError("fresh audit exec receipt differs from live child/source")
    binding = {
        "schema": FRESH_CANARY_EXEC_RECEIPT_SCHEMA,
        "candidate_id": CANARY_CANDIDATE_ID,
        "auditor_process_id": process.pid,
        "auditor_executable_file_sha256": executable_sha256,
        "exact_orig_argv_sha256": receipt["exact_orig_argv_sha256"],
        "clean_commit": snapshot.identity.commit,
        "clean_tree": snapshot.identity.tree,
        "source_bundle_sha256": snapshot.source_bundle_sha256,
        "fresh_audit_payload_sha256": parsed_audit["payload_sha256"],
        "fresh_audit_file_sha256": file_sha256(audit_bytes),
        "receipt_payload_sha256": receipt["payload_sha256"],
    }
    value = object.__new__(ObservedFreshCanaryExecCapability)
    for name, item in binding.items():
        object.__setattr__(value, name, item)
    object.__setattr__(value, "capability_seal_sha256", _artifact_binding_sha256(binding))
    object.__setattr__(value, "_fresh_audit_file_bytes", bytes(audit_bytes))
    object.__setattr__(value, "_process", process)
    object.__setattr__(value, "_validation_marker", _FRESH_EXEC_OBSERVER_ISSUER)
    return value


def _require_observed_fresh_exec_capability(
    value: object,
) -> ObservedFreshCanaryExecCapability:
    if type(value) is not ObservedFreshCanaryExecCapability:
        raise AuthorityError("exact ObservedFreshCanaryExecCapability required")
    capability = value
    if capability._validation_marker is not _FRESH_EXEC_OBSERVER_ISSUER:
        raise AuthorityError("fresh exec observer marker is invalid")
    binding = {
        "schema": capability.schema,
        "candidate_id": capability.candidate_id,
        "auditor_process_id": capability.auditor_process_id,
        "auditor_executable_file_sha256": capability.auditor_executable_file_sha256,
        "exact_orig_argv_sha256": capability.exact_orig_argv_sha256,
        "clean_commit": capability.clean_commit,
        "clean_tree": capability.clean_tree,
        "source_bundle_sha256": capability.source_bundle_sha256,
        "fresh_audit_payload_sha256": capability.fresh_audit_payload_sha256,
        "fresh_audit_file_sha256": capability.fresh_audit_file_sha256,
        "receipt_payload_sha256": capability.receipt_payload_sha256,
    }
    audit = parse_artifact_bytes(
        capability._fresh_audit_file_bytes,
        ArtifactSpec(
            schema=FRESH_CANARY_AUDIT_SCHEMA,
            exact_fields=_FRESH_AUDIT_FIELDS,
        ),
    )
    if (
        capability.schema != FRESH_CANARY_EXEC_RECEIPT_SCHEMA
        or capability.candidate_id != CANARY_CANDIDATE_ID
        or capability.capability_seal_sha256 != _artifact_binding_sha256(binding)
        or capability._process.pid != capability.auditor_process_id
        or capability._process.poll() is not None
        or tuple(capability._process.args) != _FRESH_CANARY_EXEC_COMMAND
        or _process_command_line(capability.auditor_process_id) != _FRESH_CANARY_EXEC_COMMAND
        or _process_executable_file_sha256(capability.auditor_process_id)
        != capability.auditor_executable_file_sha256
        or capability.exact_orig_argv_sha256 != _fresh_exec_command_sha256()
        or audit["payload_sha256"] != capability.fresh_audit_payload_sha256
        or file_sha256(capability._fresh_audit_file_bytes) != capability.fresh_audit_file_sha256
    ):
        raise AuthorityError("fresh exec capability no longer matches its live child")
    return capability


def _validated_fresh_audit_inputs(
    *,
    selected_method_freeze: ValidatedArtifactCapability,
    test_evidence: ValidatedArtifactCapability,
    canary_authorization: CanaryArtifactCapability,
    canary_claim: CanaryArtifactCapability,
    canary_beacon: CanaryArtifactCapability,
    published_canary_result: PublishedCanaryResultCapability,
) -> Mapping[
    str, DevelopmentBundleCapability | ValidatedArtifactCapability | CanaryArtifactCapability
]:
    selected = require_validated_artifact_capability(
        selected_method_freeze,
        expected_schema=ARTIFACT_SCHEMAS["selected_method_freeze"],
    )
    tests = require_validated_artifact_capability(
        test_evidence,
        expected_schema=ARTIFACT_SCHEMAS["test_evidence"],
    )
    authorization = require_canary_artifact_capability(
        canary_authorization,
        expected_schema=ARTIFACT_SCHEMAS["canary_authorization"],
    )
    claim = require_canary_artifact_capability(
        canary_claim,
        expected_schema=ARTIFACT_SCHEMAS["canary_claim"],
    )
    beacon = require_canary_artifact_capability(
        canary_beacon,
        expected_schema=ARTIFACT_SCHEMAS["canary_beacon"],
    )
    result_publication = require_published_canary_result_capability(published_canary_result)
    result = result_publication.artifact
    for name, capability in (
        ("canary authorization", authorization),
        ("canary claim", claim),
        ("canary beacon", beacon),
        ("canary result", result),
    ):
        if (
            capability.source_commit != selected.source_commit
            or capability.source_tree != selected.source_tree
        ):
            raise AuthorityError(f"{name} source identity differs from selected B")
    tests_payload = tests._validated_payload
    authorization_payload = authorization._validated_payload
    claim_payload = claim._validated_payload
    beacon_payload = beacon._validated_payload
    result_payload = result._validated_payload
    if (
        tests_payload["selected_method_freeze_payload_sha256"] != selected.payload_sha256
        or tests_payload["selected_method_freeze_file_sha256"] != selected.file_sha256
        or authorization_payload["selected_method_freeze_payload_sha256"] != selected.payload_sha256
        or authorization_payload["selected_method_freeze_file_sha256"] != selected.file_sha256
        or authorization_payload["test_evidence_payload_sha256"] != tests.payload_sha256
        or authorization_payload["test_evidence_file_sha256"] != tests.file_sha256
        or authorization_payload["clean_commit"] != selected.source_commit
        or authorization_payload["clean_tree"] != selected.source_tree
    ):
        raise AuthorityError("fresh audit selection, tests, and authorization are not one chain")
    if (
        claim_payload["canary_authorization_payload_sha256"] != authorization.payload_sha256
        or claim_payload["canary_authorization_file_sha256"] != authorization.file_sha256
        or beacon_payload["canary_claim_payload_sha256"] != claim.payload_sha256
        or beacon_payload["canary_claim_file_sha256"] != claim.file_sha256
        or result_payload["canary_beacon_payload_sha256"] != beacon.payload_sha256
        or result_payload["canary_beacon_file_sha256"] != beacon.file_sha256
    ):
        raise AuthorityError("fresh audit canary artifacts are not one validated chain")
    if (
        beacon_payload["public_fixture_file_sha256"] != CANARY_FIXTURE_SHA256
        or beacon_payload["timestamp_UTC"] != CANARY_TIMESTAMP_UTC
        or beacon_payload["chain_index"] != CANARY_CHAIN_INDEX
        or beacon_payload["pulse_index"] != CANARY_PULSE_INDEX
        or beacon_payload["output_value"] != CANARY_OUTPUT_VALUE
        or result_payload["seed_digest_sha256"] != CANARY_SEED_DIGEST
        or result_payload["probe_digest_sha256"] != CANARY_PROBE_DIGEST
    ):
        raise AuthorityError("fresh audit canary oracle binding mismatch")
    return MappingProxyType(
        {
            "selected_method_freeze": selected,
            "test_evidence": tests,
            "canary_authorization": authorization,
            "canary_claim": claim,
            "canary_beacon": beacon,
            "canary_result": result,
        }
    )


def _build_fresh_canary_audit_for_process(
    *,
    selected_method_freeze: ValidatedArtifactCapability,
    test_evidence: ValidatedArtifactCapability,
    canary_authorization: CanaryArtifactCapability,
    canary_claim: CanaryArtifactCapability,
    canary_beacon: CanaryArtifactCapability,
    published_canary_result: PublishedCanaryResultCapability,
    clean_commit: str,
    clean_tree: str,
    state_trace: Sequence[CanaryState],
    observed_seed_digest: str,
    observed_probe_digest: str,
    auditor_process_id: int,
    auditor_executable_file_sha256: str,
) -> Mapping[str, Any]:
    _require_git_object(clean_commit, "canary clean commit")
    _require_git_object(clean_tree, "canary clean tree")
    capabilities = _validated_fresh_audit_inputs(
        selected_method_freeze=selected_method_freeze,
        test_evidence=test_evidence,
        canary_authorization=canary_authorization,
        canary_claim=canary_claim,
        canary_beacon=canary_beacon,
        published_canary_result=published_canary_result,
    )
    result_publication = require_published_canary_result_capability(published_canary_result)
    result_payload = result_publication.artifact._validated_payload
    if (
        clean_commit != selected_method_freeze.source_commit
        or clean_tree != selected_method_freeze.source_tree
    ):
        raise AuthorityError("fresh canary audit clean identity differs from selected freeze")
    if tuple(state_trace) != _CANARY_FULL_TRACE:
        raise TransitionError("fresh canary audit requires the exact complete canary trace")
    if observed_seed_digest != CANARY_SEED_DIGEST or observed_probe_digest != CANARY_PROBE_DIGEST:
        raise ValidationError("fresh canary audit oracle mismatch")
    publisher_process_id = result_publication.publisher_process_id
    if (
        not isinstance(auditor_process_id, int)
        or isinstance(auditor_process_id, bool)
        or auditor_process_id <= 0
    ):
        raise ValidationError("fresh auditor process ID must be a positive integer")
    _require_sha256(
        auditor_executable_file_sha256,
        "fresh auditor executable file_sha256",
    )
    if publisher_process_id == auditor_process_id:
        raise AuthorityError("fresh audit must run in a process distinct from the publisher")
    if result_payload["publisher_process_id"] != publisher_process_id:
        raise AuthorityError("fresh audit publisher differs from validated canary result")
    content: dict[str, Any] = {
        "schema": FRESH_CANARY_AUDIT_SCHEMA,
        "candidate_id": CANARY_CANDIDATE_ID,
        "clean_commit": clean_commit,
        "clean_tree": clean_tree,
        "public_fixture_file_sha256": CANARY_FIXTURE_SHA256,
        "exact_CANARY_state_trace": [state.value for state in state_trace],
        "expected_and_observed_seed_digest": {
            "expected": CANARY_SEED_DIGEST,
            "observed": observed_seed_digest,
        },
        "expected_and_observed_probe_digest": {
            "expected": CANARY_PROBE_DIGEST,
            "observed": observed_probe_digest,
        },
        "publisher_process_id": publisher_process_id,
        "distinct_auditor_process_id": auditor_process_id,
        "auditor_executable_file_sha256": auditor_executable_file_sha256,
    }
    for name in _FRESH_AUDIT_REFERENCE_NAMES:
        reference = _artifact_reference_from_capability(capabilities[name])
        _require_artifact_reference(
            reference,
            expected_schema=_FRESH_AUDIT_REFERENCE_SCHEMAS[name],
        )
        content[f"{name}_schema"] = reference.schema
        content[f"{name}_payload_sha256"] = reference.payload_sha256
        content[f"{name}_file_sha256"] = reference.file_sha256
    return seal_payload(content)


def build_fresh_canary_audit(
    *,
    selected_method_freeze: ValidatedArtifactCapability,
    test_evidence: ValidatedArtifactCapability,
    canary_authorization: CanaryArtifactCapability,
    canary_claim: CanaryArtifactCapability,
    canary_beacon: CanaryArtifactCapability,
    published_canary_result: PublishedCanaryResultCapability,
    clean_commit: str,
    clean_tree: str,
    state_trace: Sequence[CanaryState],
    observed_seed_digest: str,
    observed_probe_digest: str,
) -> Mapping[str, Any]:
    """Build a fresh-audit payload bound to the actual current auditor."""

    return _build_fresh_canary_audit_for_process(
        selected_method_freeze=selected_method_freeze,
        test_evidence=test_evidence,
        canary_authorization=canary_authorization,
        canary_claim=canary_claim,
        canary_beacon=canary_beacon,
        published_canary_result=published_canary_result,
        clean_commit=clean_commit,
        clean_tree=clean_tree,
        state_trace=state_trace,
        observed_seed_digest=observed_seed_digest,
        observed_probe_digest=observed_probe_digest,
        auditor_process_id=os.getpid(),
        auditor_executable_file_sha256=_current_executable_file_sha256(),
    )


def _fresh_canary_audit_exec_child_main() -> int:
    """Pinned hidden exec role: reopen everything, emit bytes, await parent ACK."""

    _assert_fresh_canary_exec_child_context()
    graph = _reopen_canonical_canary_graph()
    audit = _build_fresh_canary_audit_for_process(
        selected_method_freeze=graph.selected_method_freeze,
        test_evidence=graph.test_evidence,
        canary_authorization=graph.canary_authorization,
        canary_claim=graph.canary_claim,
        canary_beacon=graph.canary_beacon,
        published_canary_result=graph.published_canary_result,
        clean_commit=graph.snapshot.identity.commit,
        clean_tree=graph.snapshot.identity.tree,
        state_trace=_CANARY_FULL_TRACE,
        observed_seed_digest=CANARY_SEED_DIGEST,
        observed_probe_digest=CANARY_PROBE_DIGEST,
        auditor_process_id=os.getpid(),
        auditor_executable_file_sha256=_current_executable_file_sha256(),
    )
    audit_bytes = artifact_bytes(audit)
    receipt = _build_fresh_exec_receipt(
        graph=graph,
        fresh_audit_file_bytes=audit_bytes,
    )
    sys.stdout.buffer.write(artifact_bytes(receipt))
    sys.stdout.buffer.flush()
    if sys.stdin.buffer.readline(16) != b"ACK\n":
        raise AuthorityError("fresh audit parent barrier acknowledgement is invalid")
    return 0


def _read_fresh_exec_receipt(
    process: subprocess.Popen[bytes],
    *,
    timeout_seconds: float = 3_600.0,
) -> bytes:
    if process.stdout is None:
        raise AuthorityError("fresh audit child stdout pipe is unavailable")
    descriptor = process.stdout.fileno()
    deadline = time.monotonic() + timeout_seconds
    chunks: list[bytes] = []
    total = 0
    while True:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise AuthorityError("fresh audit child receipt timed out")
        readable, _, _ = select.select((descriptor,), (), (), min(1.0, remaining))
        if not readable:
            if process.poll() is not None:
                raise AuthorityError("fresh audit child exited before its receipt")
            continue
        chunk = os.read(descriptor, 64 * 1024)
        if not chunk:
            raise AuthorityError("fresh audit child closed stdout before its receipt")
        chunks.append(chunk)
        total += len(chunk)
        if total > 2 * 1024 * 1024:
            raise AuthorityError("fresh audit child receipt exceeds its fixed limit")
        joined = b"".join(chunks)
        if b"\n" in joined:
            if joined.count(b"\n") != 1 or not joined.endswith(b"\n"):
                raise AuthorityError("fresh audit child emitted non-receipt stdout")
            return joined


def _stop_fresh_exec_child(process: subprocess.Popen[bytes]) -> None:
    if process.poll() is None:
        process.terminate()
        try:
            process.wait(timeout=10.0)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=10.0)


def _publish_fresh_canary_audit_from_observed_exec(
    observed_exec: ObservedFreshCanaryExecCapability,
    *,
    graph: _CanonicalCanaryGraph,
) -> PublishedArtifact:
    observed = _require_observed_fresh_exec_capability(observed_exec)
    _require_prepublication_current_source(graph.snapshot)
    expected = _build_fresh_canary_audit_for_process(
        selected_method_freeze=graph.selected_method_freeze,
        test_evidence=graph.test_evidence,
        canary_authorization=graph.canary_authorization,
        canary_claim=graph.canary_claim,
        canary_beacon=graph.canary_beacon,
        published_canary_result=graph.published_canary_result,
        clean_commit=graph.snapshot.identity.commit,
        clean_tree=graph.snapshot.identity.tree,
        state_trace=_CANARY_FULL_TRACE,
        observed_seed_digest=CANARY_SEED_DIGEST,
        observed_probe_digest=CANARY_PROBE_DIGEST,
        auditor_process_id=observed.auditor_process_id,
        auditor_executable_file_sha256=observed.auditor_executable_file_sha256,
    )
    expected_bytes = artifact_bytes(expected)
    if expected_bytes != observed._fresh_audit_file_bytes:
        raise AuthorityError("parent reconstruction differs from fresh child audit bytes")
    return _publish_write_once(
        CANARY_CANONICAL_ROOT,
        CANARY_FRESH_AUDIT_CANONICAL_PATH.name,
        expected_bytes,
        create_root=False,
        _publisher=_PUBLICATION_ISSUER,
    )


def _validate_fresh_canary_audit_at_paths(
    *,
    audited_artifact_paths: Mapping[str, Path],
    audit_path: Path,
    scope: str,
    public_fixture_bytes: bytes,
    selected_method_freeze: ValidatedArtifactCapability,
    test_evidence: ValidatedArtifactCapability,
    canary_authorization: CanaryArtifactCapability,
    canary_claim: CanaryArtifactCapability,
    canary_beacon: CanaryArtifactCapability,
    published_canary_result: PublishedCanaryResultCapability,
    selected_source_snapshot: CleanSourceSnapshot,
    expected_auditor_process_id: int | None,
    require_current_auditor: bool,
    require_live_auditor: bool,
    _observer: object,
) -> FreshCanaryAuditCapability:
    if _observer is not _CAPABILITY_ISSUER:
        raise AuthorityError("fresh-audit issuance requires exact canonical observation")
    _require_active_governed_process(
        recheck_site_packages=True,
        recheck_source=True,
    )
    if frozenset(audited_artifact_paths) != frozenset(_FRESH_AUDIT_REFERENCE_NAMES):
        raise ValidationError("fresh audit requires the exact audited artifact path set")
    publication = require_published_canary_result_capability(published_canary_result)
    if publication.path != audited_artifact_paths["canary_result"] or scope != "canonical":
        raise AuthorityError("fresh audit result publication scope or path mismatch")
    capabilities = _validated_fresh_audit_inputs(
        selected_method_freeze=selected_method_freeze,
        test_evidence=test_evidence,
        canary_authorization=canary_authorization,
        canary_claim=canary_claim,
        canary_beacon=canary_beacon,
        published_canary_result=published_canary_result,
    )
    exact_fields = {
        "selected_method_freeze": _SELECTED_FREEZE_FIELDS,
        "test_evidence": _TEST_EVIDENCE_FIELDS,
        "canary_authorization": _CANARY_AUTHORIZATION_FIELDS,
        "canary_claim": _CANARY_CLAIM_FIELDS,
        "canary_beacon": _CANARY_BEACON_FIELDS,
        "canary_result": _CANARY_RESULT_FIELDS,
    }
    selected_capability = capabilities["selected_method_freeze"]
    selected_source_snapshot = _require_clean_source_snapshot(selected_source_snapshot)
    if (
        selected_capability.source_commit != selected_source_snapshot.identity.commit
        or selected_capability.source_tree != selected_source_snapshot.identity.tree
    ):
        raise AuthorityError("fresh audit selected source snapshot binding mismatch")
    selected_data = _load_committed_source_blob(
        V3_SOURCE_REPOSITORY,
        SELECTED_METHOD_FREEZE_REPOSITORY_PATH,
        snapshot=selected_source_snapshot,
        production=True,
    )
    external_names = tuple(
        name for name in _FRESH_AUDIT_REFERENCE_NAMES if name != "selected_method_freeze"
    )
    external_paths = [*(audited_artifact_paths[name] for name in external_names), audit_path]
    loaded = load_exact_artifact_bytes(external_paths, declared_paths=external_paths)
    for name in _FRESH_AUDIT_REFERENCE_NAMES:
        data = (
            selected_data
            if name == "selected_method_freeze"
            else loaded[os.fspath(audited_artifact_paths[name])]
        )
        parsed_artifact = parse_artifact_bytes(
            data,
            ArtifactSpec(
                schema=_FRESH_AUDIT_REFERENCE_SCHEMAS[name],
                exact_fields=exact_fields[name],
            ),
        )
        capability = capabilities[name]
        if (
            str(parsed_artifact["payload_sha256"]) != capability.payload_sha256
            or file_sha256(data) != capability.file_sha256
            or canonical_json_bytes(parsed_artifact)
            != canonical_json_bytes(capability._validated_payload)
        ):
            raise AuthorityError(f"fresh audit {name} bytes differ from validated capability")
    validate_canary_fixture(public_fixture_bytes, clock=HistoricalCanaryClock())
    file_bytes = loaded[os.fspath(audit_path)]
    parsed = parse_artifact_bytes(
        file_bytes,
        ArtifactSpec(schema=FRESH_CANARY_AUDIT_SCHEMA, exact_fields=_FRESH_AUDIT_FIELDS),
    )
    auditor_process_id = parsed["distinct_auditor_process_id"]
    if (
        not isinstance(auditor_process_id, int)
        or isinstance(auditor_process_id, bool)
        or auditor_process_id <= 0
    ):
        raise ValidationError("fresh audit auditor process ID is invalid")
    if require_current_auditor and auditor_process_id != os.getpid():
        raise AuthorityError("fresh audit artifact was not produced by the current auditor")
    if (
        expected_auditor_process_id is not None
        and auditor_process_id != expected_auditor_process_id
    ):
        raise AuthorityError("fresh audit artifact differs from the observed child PID")
    auditor_executable = _require_sha256(
        parsed["auditor_executable_file_sha256"],
        "fresh audit executable file_sha256",
    )
    if require_live_auditor and (
        auditor_executable != _process_executable_file_sha256(auditor_process_id)
    ):
        raise AuthorityError("fresh audit executable observation mismatch")
    expected = _build_fresh_canary_audit_for_process(
        selected_method_freeze=selected_method_freeze,
        test_evidence=test_evidence,
        canary_authorization=canary_authorization,
        canary_claim=canary_claim,
        canary_beacon=canary_beacon,
        published_canary_result=published_canary_result,
        clean_commit=str(parsed["clean_commit"]),
        clean_tree=str(parsed["clean_tree"]),
        state_trace=tuple(CanaryState(item) for item in parsed["exact_CANARY_state_trace"]),
        observed_seed_digest=str(parsed["expected_and_observed_seed_digest"]["observed"]),
        observed_probe_digest=str(parsed["expected_and_observed_probe_digest"]["observed"]),
        auditor_process_id=auditor_process_id,
        auditor_executable_file_sha256=auditor_executable,
    )
    if canonical_json_bytes(parsed) != canonical_json_bytes(expected):
        raise ValidationError("fresh canary audit differs from exact reconstruction")
    selected = capabilities["selected_method_freeze"]
    tests = capabilities["test_evidence"]
    result = capabilities["canary_result"]
    return FreshCanaryAuditCapability(
        payload_sha256=str(parsed["payload_sha256"]),
        file_sha256=file_sha256(file_bytes),
        clean_commit=str(parsed["clean_commit"]),
        clean_tree=str(parsed["clean_tree"]),
        exact_state_trace=_CANARY_FULL_TRACE,
        selected_method_freeze_payload_sha256=selected.payload_sha256,
        selected_method_freeze_file_sha256=selected.file_sha256,
        test_evidence_payload_sha256=tests.payload_sha256,
        test_evidence_file_sha256=tests.file_sha256,
        canary_result_payload_sha256=result.payload_sha256,
        canary_result_file_sha256=result.file_sha256,
        publisher_process_id=publication.publisher_process_id,
        auditor_process_id=auditor_process_id,
        auditor_executable_file_sha256=auditor_executable,
        auditor_liveness_observed=require_live_auditor,
        scope=scope,
        validated_payload=parsed,
        _issuer=_CAPABILITY_ISSUER,
    )


def validate_fresh_canary_audit(
    *,
    public_fixture_bytes: bytes,
    selected_method_freeze: ValidatedArtifactCapability,
    test_evidence: ValidatedArtifactCapability,
    canary_authorization: CanaryArtifactCapability,
    canary_claim: CanaryArtifactCapability,
    canary_beacon: CanaryArtifactCapability,
    published_canary_result: PublishedCanaryResultCapability,
    selected_source_snapshot: CleanSourceSnapshot,
) -> FreshCanaryAuditCapability:
    """Validate only the exact canonical fresh-audit publication graph."""

    _require_active_governed_process(
        recheck_site_packages=True,
        recheck_source=True,
    )
    return _validate_fresh_canary_audit_at_paths(
        audited_artifact_paths=_FRESH_AUDIT_CANONICAL_PATHS,
        audit_path=CANARY_FRESH_AUDIT_CANONICAL_PATH,
        scope="canonical",
        public_fixture_bytes=public_fixture_bytes,
        selected_method_freeze=selected_method_freeze,
        test_evidence=test_evidence,
        canary_authorization=canary_authorization,
        canary_claim=canary_claim,
        canary_beacon=canary_beacon,
        published_canary_result=published_canary_result,
        selected_source_snapshot=selected_source_snapshot,
        expected_auditor_process_id=os.getpid(),
        require_current_auditor=True,
        require_live_auditor=True,
        _observer=_CAPABILITY_ISSUER,
    )


def observe_child_fresh_canary_audit(
    *,
    auditor_process_id: int,
    public_fixture_bytes: bytes,
    selected_method_freeze: ValidatedArtifactCapability,
    test_evidence: ValidatedArtifactCapability,
    canary_authorization: CanaryArtifactCapability,
    canary_claim: CanaryArtifactCapability,
    canary_beacon: CanaryArtifactCapability,
    published_canary_result: PublishedCanaryResultCapability,
    selected_source_snapshot: CleanSourceSnapshot,
) -> FreshCanaryAuditCapability:
    """Observe a still-live exec child's canonical audit from the parent."""

    _require_active_governed_process(
        recheck_site_packages=True,
        recheck_source=True,
        require_mutation_role=True,
    )
    if auditor_process_id == os.getpid():
        raise AuthorityError("fresh audit child must be distinct from its parent")
    return _validate_fresh_canary_audit_at_paths(
        audited_artifact_paths=_FRESH_AUDIT_CANONICAL_PATHS,
        audit_path=CANARY_FRESH_AUDIT_CANONICAL_PATH,
        scope="canonical",
        public_fixture_bytes=public_fixture_bytes,
        selected_method_freeze=selected_method_freeze,
        test_evidence=test_evidence,
        canary_authorization=canary_authorization,
        canary_claim=canary_claim,
        canary_beacon=canary_beacon,
        published_canary_result=published_canary_result,
        selected_source_snapshot=selected_source_snapshot,
        expected_auditor_process_id=auditor_process_id,
        require_current_auditor=False,
        require_live_auditor=True,
        _observer=_CAPABILITY_ISSUER,
    )


def reopen_fresh_canary_audit(
    *,
    public_fixture_bytes: bytes,
    selected_method_freeze: ValidatedArtifactCapability,
    test_evidence: ValidatedArtifactCapability,
    canary_authorization: CanaryArtifactCapability,
    canary_claim: CanaryArtifactCapability,
    canary_beacon: CanaryArtifactCapability,
    published_canary_result: PublishedCanaryResultCapability,
    selected_source_snapshot: CleanSourceSnapshot,
) -> FreshCanaryAuditCapability:
    """Reopen durable fresh-audit authority after both recorded processes exit."""

    _require_active_governed_process(
        recheck_site_packages=True,
        recheck_source=True,
    )
    return _validate_fresh_canary_audit_at_paths(
        audited_artifact_paths=_FRESH_AUDIT_CANONICAL_PATHS,
        audit_path=CANARY_FRESH_AUDIT_CANONICAL_PATH,
        scope="canonical",
        public_fixture_bytes=public_fixture_bytes,
        selected_method_freeze=selected_method_freeze,
        test_evidence=test_evidence,
        canary_authorization=canary_authorization,
        canary_claim=canary_claim,
        canary_beacon=canary_beacon,
        published_canary_result=published_canary_result,
        selected_source_snapshot=selected_source_snapshot,
        expected_auditor_process_id=None,
        require_current_auditor=False,
        require_live_auditor=False,
        _observer=_CAPABILITY_ISSUER,
    )


def run_fresh_canary_audit_in_exec_subprocess() -> FreshCanaryAuditCapability:
    """Run or reopen the fixed canonical audit through a clean isolated exec."""

    _require_active_governed_process(
        recheck_site_packages=True,
        recheck_source=True,
    )
    graph = _reopen_canonical_canary_graph()
    try:
        return reopen_fresh_canary_audit(
            public_fixture_bytes=graph.public_fixture_bytes,
            selected_method_freeze=graph.selected_method_freeze,
            test_evidence=graph.test_evidence,
            canary_authorization=graph.canary_authorization,
            canary_claim=graph.canary_claim,
            canary_beacon=graph.canary_beacon,
            published_canary_result=graph.published_canary_result,
            selected_source_snapshot=graph.snapshot,
        )
    except FileNotFoundError:
        pass

    _require_active_governed_process(
        recheck_site_packages=True,
        recheck_source=True,
        require_mutation_role=True,
    )
    process = subprocess.Popen(
        _FRESH_CANARY_EXEC_COMMAND,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        cwd=V3_SOURCE_REPOSITORY,
        env=dict(_OBSERVED_TEST_ENVIRONMENT),
        close_fds=True,
        bufsize=0,
    )
    try:
        receipt_bytes = _read_fresh_exec_receipt(process)
        observed = _issue_observed_fresh_exec_capability(
            process=process,
            receipt_bytes=receipt_bytes,
            snapshot=graph.snapshot,
            _observer=_FRESH_EXEC_OBSERVER_ISSUER,
        )
        _publish_fresh_canary_audit_from_observed_exec(observed, graph=graph)
        capability = _validate_fresh_canary_audit_at_paths(
            audited_artifact_paths=_FRESH_AUDIT_CANONICAL_PATHS,
            audit_path=CANARY_FRESH_AUDIT_CANONICAL_PATH,
            scope="canonical",
            public_fixture_bytes=graph.public_fixture_bytes,
            selected_method_freeze=graph.selected_method_freeze,
            test_evidence=graph.test_evidence,
            canary_authorization=graph.canary_authorization,
            canary_claim=graph.canary_claim,
            canary_beacon=graph.canary_beacon,
            published_canary_result=graph.published_canary_result,
            selected_source_snapshot=graph.snapshot,
            expected_auditor_process_id=process.pid,
            require_current_auditor=False,
            require_live_auditor=True,
            _observer=_CAPABILITY_ISSUER,
        )
        if process.stdin is None or process.stdout is None or process.stderr is None:
            raise AuthorityError("fresh audit child pipes disappeared")
        process.stdin.write(b"ACK\n")
        process.stdin.flush()
        process.stdin.close()
        if process.wait(timeout=30.0) != 0:
            raise AuthorityError("fresh audit child failed after parent publication")
        trailing_stdout = process.stdout.read()
        stderr = process.stderr.read()
        if trailing_stdout or stderr:
            raise AuthorityError("fresh audit child emitted unexpected trailing output")
        return capability
    finally:
        _stop_fresh_exec_child(process)


def _validate_fresh_canary_audit_under_test_root(
    test_root: Path,
    *,
    public_fixture_bytes: bytes,
    selected_method_freeze: ValidatedArtifactCapability,
    test_evidence: ValidatedArtifactCapability,
    canary_authorization: CanaryArtifactCapability,
    canary_claim: CanaryArtifactCapability,
    canary_beacon: CanaryArtifactCapability,
    published_canary_result: PublishedCanaryResultCapability,
) -> FreshCanaryAuditCapability:
    del (
        test_root,
        public_fixture_bytes,
        selected_method_freeze,
        test_evidence,
        canary_authorization,
        canary_claim,
        canary_beacon,
        published_canary_result,
    )
    raise AuthorityError("tmp-path fresh audits can never issue a capability")


def require_fresh_canary_audit_capability(value: object) -> FreshCanaryAuditCapability:
    if type(value) is not FreshCanaryAuditCapability:
        raise AuthorityError("exact FreshCanaryAuditCapability required")
    capability = value
    if capability._validation_marker is not _CAPABILITY_ISSUER:
        raise AuthorityError("fresh canary audit marker is invalid")
    payload = capability._validated_payload
    seal = {
        "schema": capability.schema,
        "candidate_id": capability.candidate_id,
        "payload_sha256": capability.payload_sha256,
        "file_sha256": capability.file_sha256,
        "clean_commit": capability.clean_commit,
        "clean_tree": capability.clean_tree,
        "exact_state_trace": [state.value for state in capability.exact_state_trace],
        "selected_method_freeze_payload_sha256": (capability.selected_method_freeze_payload_sha256),
        "selected_method_freeze_file_sha256": (capability.selected_method_freeze_file_sha256),
        "test_evidence_payload_sha256": capability.test_evidence_payload_sha256,
        "test_evidence_file_sha256": capability.test_evidence_file_sha256,
        "canary_result_payload_sha256": capability.canary_result_payload_sha256,
        "canary_result_file_sha256": capability.canary_result_file_sha256,
        "publisher_process_id": capability.publisher_process_id,
        "auditor_process_id": capability.auditor_process_id,
        "auditor_executable_file_sha256": capability.auditor_executable_file_sha256,
        "auditor_liveness_observed": capability.auditor_liveness_observed,
        "scope": capability.scope,
    }
    if (
        capability.schema != FRESH_CANARY_AUDIT_CAPABILITY_SCHEMA
        or capability.candidate_id != CANARY_CANDIDATE_ID
        or capability.exact_state_trace != _CANARY_FULL_TRACE
        or validate_payload_hash(payload) != capability.payload_sha256
        or file_sha256(artifact_bytes(payload)) != capability.file_sha256
        or capability.scope != "canonical"
        or capability.publisher_process_id != payload["publisher_process_id"]
        or capability.auditor_process_id != payload["distinct_auditor_process_id"]
        or capability.auditor_executable_file_sha256 != payload["auditor_executable_file_sha256"]
        or capability.publisher_process_id == capability.auditor_process_id
        or not isinstance(capability.auditor_liveness_observed, bool)
        or capability.clean_commit != payload["clean_commit"]
        or capability.clean_tree != payload["clean_tree"]
        or [state.value for state in capability.exact_state_trace]
        != payload["exact_CANARY_state_trace"]
        or capability.selected_method_freeze_payload_sha256
        != payload["selected_method_freeze_payload_sha256"]
        or capability.selected_method_freeze_file_sha256
        != payload["selected_method_freeze_file_sha256"]
        or capability.test_evidence_payload_sha256 != payload["test_evidence_payload_sha256"]
        or capability.test_evidence_file_sha256 != payload["test_evidence_file_sha256"]
        or capability.canary_result_payload_sha256 != payload["canary_result_payload_sha256"]
        or capability.canary_result_file_sha256 != payload["canary_result_file_sha256"]
        or capability.capability_seal_sha256 != _artifact_binding_sha256(seal)
    ):
        raise AuthorityError("fresh canary audit capability binding mismatch")
    for name in (
        "payload_sha256",
        "file_sha256",
        "selected_method_freeze_payload_sha256",
        "selected_method_freeze_file_sha256",
        "test_evidence_payload_sha256",
        "test_evidence_file_sha256",
        "canary_result_payload_sha256",
        "canary_result_file_sha256",
        "auditor_executable_file_sha256",
    ):
        _require_sha256(getattr(capability, name), f"fresh canary capability {name}")
    _require_git_object(capability.clean_commit, "fresh canary clean commit")
    _require_git_object(capability.clean_tree, "fresh canary clean tree")
    return capability


def bridge_canary_audit(
    scientific: ScientificLifecycle,
    canary: CanaryLifecycle,
    capability: FreshCanaryAuditCapability,
) -> ScientificLifecycle:
    validated = require_fresh_canary_audit_capability(capability)
    if type(scientific) is not ScientificLifecycle or type(canary) is not CanaryLifecycle:
        raise AuthorityError("bridge requires exact issued lifecycle capabilities")
    _require_exact_lifecycle_chain(scientific)
    _require_exact_lifecycle_chain(canary)
    if validated.scope != "canonical":
        raise AuthorityError("test-only fresh audit cannot enter the scientific lifecycle")
    if scientific.state is not ScientificState.VERIFIED:
        raise TransitionError("canary bridge requires scientific VERIFIED state")
    if canary.state is not CanaryState.CANARY_FRESH_AUDITED:
        raise TransitionError("canary bridge requires CANARY_FRESH_AUDITED state")
    if canary.trace != validated.exact_state_trace:
        raise TransitionError("canary lifecycle trace differs from fresh audit capability")
    if (
        scientific.clean_commit != validated.clean_commit
        or scientific.clean_tree != validated.clean_tree
        or canary.clean_commit != validated.clean_commit
        or canary.clean_tree != validated.clean_tree
    ):
        raise AuthorityError("bridge clean identity differs across scientific and canary chains")
    for name in ("selected_method_freeze", "test_evidence"):
        expected = scientific.artifact_bindings.get(name)
        canary_observed = canary.artifact_bindings.get(name)
        audit_observed = (
            getattr(validated, f"{name}_payload_sha256"),
            getattr(validated, f"{name}_file_sha256"),
        )
        if expected is None or expected != canary_observed or expected != audit_observed:
            raise AuthorityError(f"bridge {name} binding mismatch")
    bindings = dict(scientific.artifact_bindings)
    bindings["canary_result"] = (
        validated.canary_result_payload_sha256,
        validated.canary_result_file_sha256,
    )
    bindings["canary_fresh_audit"] = _exact_binding(validated)
    bridged = object.__new__(ScientificLifecycle)
    object.__setattr__(bridged, "state", ScientificState.CANARY_PASSED)
    object.__setattr__(
        bridged,
        "trace",
        (*scientific.trace, ScientificState.CANARY_PASSED),
    )
    object.__setattr__(bridged, "artifact_bindings", MappingProxyType(bindings))
    object.__setattr__(bridged, "clean_commit", scientific.clean_commit)
    object.__setattr__(bridged, "clean_tree", scientific.clean_tree)
    object.__setattr__(
        bridged,
        "_sealed_chain",
        _lifecycle_chain_binding(
            state=ScientificState.CANARY_PASSED,
            trace=(*scientific.trace, ScientificState.CANARY_PASSED),
            artifact_bindings=bindings,
            clean_commit=scientific.clean_commit,
            clean_tree=scientific.clean_tree,
        ),
    )
    object.__setattr__(bridged, "_validation_marker", _CAPABILITY_ISSUER)
    return bridged


def _validate_https_endpoint(
    value: Any,
    *,
    exact_target_timestamp_UTC: str,
) -> str:
    if not isinstance(value, str):
        raise ValidationError("future beacon endpoint must be a string")
    try:
        value.encode("ascii")
    except UnicodeEncodeError as exc:
        raise ValidationError("future beacon endpoint must be ASCII") from exc
    try:
        parsed = urlsplit(value)
        parsed_port = parsed.port
    except ValueError as exc:
        raise ValidationError("future beacon endpoint is malformed") from exc
    target = _parse_rfc3339_seconds(exact_target_timestamp_UTC, "future beacon endpoint target")
    expected_path = f"/beacon/2.0/pulse/time/{int(target.timestamp()) * 1000}"
    if (
        parsed.scheme != "https"
        or parsed.netloc != "beacon.nist.gov"
        or parsed.hostname != "beacon.nist.gov"
        or parsed_port is not None
        or parsed.username is not None
        or parsed.password is not None
        or parsed.path != expected_path
        or parsed.query
        or parsed.fragment
    ):
        raise ValidationError("future beacon endpoint must be the exact target-bound NIST URL")
    return value


def build_seed_statement(
    *,
    scientific_candidate_id: str,
    selected_method_freeze_payload_sha256: str,
    exact_target_timestamp_UTC: str,
    exact_HTTPS_endpoint: str,
    beacon_seed_derivation_primitive_schema: str = BEACON_SEED_DERIVATION_SCHEMA,
) -> bytes:
    if not isinstance(scientific_candidate_id, str) or not scientific_candidate_id.startswith(
        f"{CANDIDATE_ID}-p3-nu_"
    ):
        raise ValidationError("scientific candidate ID is not a selected V3 operator ID")
    _require_sha256(
        selected_method_freeze_payload_sha256,
        "selected-method-freeze payload_sha256",
    )
    _parse_rfc3339_seconds(exact_target_timestamp_UTC, "future beacon target")
    _validate_https_endpoint(
        exact_HTTPS_endpoint,
        exact_target_timestamp_UTC=exact_target_timestamp_UTC,
    )
    if beacon_seed_derivation_primitive_schema != BEACON_SEED_DERIVATION_SCHEMA:
        raise ValidationError("beacon seed derivation primitive schema mismatch")
    return _ascii_lines(
        (
            SEED_STATEMENT_SCHEMA,
            CANDIDATE_ID,
            scientific_candidate_id,
            selected_method_freeze_payload_sha256,
            exact_target_timestamp_UTC,
            exact_HTTPS_endpoint,
            beacon_seed_derivation_primitive_schema,
        )
    )


_MANIFEST_REFERENCE_NAMES = (
    "selected_method_freeze",
    "development_bundle",
    "test_evidence",
    "canary_result",
    "canary_fresh_audit",
)
_MANIFEST_REFERENCE_SCHEMAS = MappingProxyType(
    {
        "selected_method_freeze": ARTIFACT_SCHEMAS["selected_method_freeze"],
        "development_bundle": DEVELOPMENT_BUNDLE_SCHEMA,
        "test_evidence": ARTIFACT_SCHEMAS["test_evidence"],
        "canary_result": ARTIFACT_SCHEMAS["canary_result"],
        "canary_fresh_audit": FRESH_CANARY_AUDIT_SCHEMA,
    }
)
_ATTEMPT_MANIFEST_FIELDS = frozenset(
    {
        "schema",
        "candidate_id",
        "scientific_candidate_id",
        "manifest_created_at_UTC",
        "clean_commit",
        "clean_tree",
        "exact_target_timestamp_UTC",
        "exact_HTTPS_endpoint",
        "beacon_seed_derivation_primitive_schema",
        "exact_seed_statement_base64",
        "payload_sha256",
        *(
            f"{name}_{suffix}"
            for name in _MANIFEST_REFERENCE_NAMES
            for suffix in ("schema", "payload_sha256", "file_sha256")
        ),
    }
)


@dataclass(frozen=True, slots=True, init=False)
class AttemptManifestCapability:
    schema: str
    candidate_id: str
    scientific_candidate_id: str
    payload_sha256: str
    file_sha256: str
    attempt_id: str
    manifest_created_at_UTC: str
    exact_target_timestamp_UTC: str
    exact_HTTPS_endpoint: str
    clean_commit: str
    clean_tree: str
    _validated_payload: Mapping[str, Any] = field(repr=False, compare=False)
    _validation_marker: object = field(repr=False, compare=False)

    def __init__(
        self,
        *,
        scientific_candidate_id: str,
        payload_sha256: str,
        file_sha256: str,
        manifest_created_at_UTC: str,
        exact_target_timestamp_UTC: str,
        exact_HTTPS_endpoint: str,
        clean_commit: str,
        clean_tree: str,
        validated_payload: Mapping[str, Any],
        _issuer: object,
    ) -> None:
        if _issuer is not _CAPABILITY_ISSUER:
            raise AuthorityError("AttemptManifestCapability has no valid issuer")
        object.__setattr__(self, "schema", ATTEMPT_MANIFEST_SCHEMA)
        object.__setattr__(self, "candidate_id", CANDIDATE_ID)
        object.__setattr__(self, "scientific_candidate_id", scientific_candidate_id)
        object.__setattr__(self, "payload_sha256", payload_sha256)
        object.__setattr__(self, "file_sha256", file_sha256)
        object.__setattr__(self, "attempt_id", f"sha256-{payload_sha256}")
        object.__setattr__(self, "manifest_created_at_UTC", manifest_created_at_UTC)
        object.__setattr__(self, "exact_target_timestamp_UTC", exact_target_timestamp_UTC)
        object.__setattr__(self, "exact_HTTPS_endpoint", exact_HTTPS_endpoint)
        object.__setattr__(self, "clean_commit", clean_commit)
        object.__setattr__(self, "clean_tree", clean_tree)
        object.__setattr__(self, "_validated_payload", _freeze_json(validated_payload))
        object.__setattr__(self, "_validation_marker", _issuer)


def require_attempt_manifest_capability(value: object) -> AttemptManifestCapability:
    raise AuthorityError("no exact future beacon is selected by the frozen contract")
    # The code below is retained as a reviewable future-amendment specification,
    # but is deliberately unreachable in this frozen authority revision.
    if type(value) is not AttemptManifestCapability:
        raise AuthorityError("exact AttemptManifestCapability required")
    capability = value
    if capability._validation_marker is not _CAPABILITY_ISSUER:
        raise AuthorityError("attempt manifest capability marker is invalid")
    if capability.schema != ATTEMPT_MANIFEST_SCHEMA or capability.candidate_id != CANDIDATE_ID:
        raise AuthorityError("attempt manifest capability identity mismatch")
    if validate_payload_hash(capability._validated_payload) != capability.payload_sha256:
        raise AuthorityError("attempt manifest capability payload binding mismatch")
    _require_sha256(capability.payload_sha256, "attempt manifest payload_sha256")
    _require_sha256(capability.file_sha256, "attempt manifest file_sha256")
    if (
        capability.attempt_id != f"sha256-{capability.payload_sha256}"
        or not _ATTEMPT_ID_RE.fullmatch(capability.attempt_id)
    ):
        raise AuthorityError("attempt ID is not derived from the exact manifest payload hash")
    created = _parse_rfc3339_seconds(
        capability.manifest_created_at_UTC, "attempt manifest creation time"
    )
    target = _parse_rfc3339_seconds(
        capability.exact_target_timestamp_UTC, "attempt manifest target time"
    )
    if not created < target:
        raise AuthorityError("attempt manifest must be created strictly before target")
    _validate_https_endpoint(
        capability.exact_HTTPS_endpoint,
        exact_target_timestamp_UTC=capability.exact_target_timestamp_UTC,
    )
    _require_git_object(capability.clean_commit, "attempt manifest clean commit")
    _require_git_object(capability.clean_tree, "attempt manifest clean tree")
    return capability


def _validated_manifest_references(
    *,
    clean_identity: CleanGitIdentity,
    selected_method_freeze: ValidatedArtifactCapability,
    development_bundle: DevelopmentBundleCapability,
    test_evidence: ValidatedArtifactCapability,
    canary_result: CanaryArtifactCapability,
    fresh_canary_audit: FreshCanaryAuditCapability,
) -> Mapping[str, _ArtifactReference]:
    selected = require_validated_artifact_capability(
        selected_method_freeze,
        expected_schema=ARTIFACT_SCHEMAS["selected_method_freeze"],
    )
    development_bundle = require_development_bundle_capability(
        development_bundle,
        expected_commit=development_bundle.clean_commit,
        expected_tree=development_bundle.clean_tree,
    )
    tests = require_validated_artifact_capability(
        test_evidence,
        expected_schema=ARTIFACT_SCHEMAS["test_evidence"],
    )
    result = require_canary_artifact_capability(
        canary_result,
        expected_schema=ARTIFACT_SCHEMAS["canary_result"],
    )
    fresh_canary_audit = require_fresh_canary_audit_capability(fresh_canary_audit)
    if fresh_canary_audit.scope != "canonical":
        raise AuthorityError("test-only fresh audit cannot authorize an attempt manifest")
    if (
        fresh_canary_audit.clean_commit != clean_identity.commit
        or fresh_canary_audit.clean_tree != clean_identity.tree
    ):
        raise AuthorityError("manifest clean source differs from fresh canary audit")
    if (
        selected.source_commit != clean_identity.commit
        or selected.source_tree != clean_identity.tree
        or tests._validated_payload["selected_method_freeze_payload_sha256"]
        != selected.payload_sha256
        or tests._validated_payload["selected_method_freeze_file_sha256"] != selected.file_sha256
        or selected.payload_sha256 != fresh_canary_audit.selected_method_freeze_payload_sha256
        or selected.file_sha256 != fresh_canary_audit.selected_method_freeze_file_sha256
        or tests.payload_sha256 != fresh_canary_audit.test_evidence_payload_sha256
        or tests.file_sha256 != fresh_canary_audit.test_evidence_file_sha256
        or result.payload_sha256 != fresh_canary_audit.canary_result_payload_sha256
        or result.file_sha256 != fresh_canary_audit.canary_result_file_sha256
    ):
        raise AuthorityError("manifest evidence references differ from fresh canary capability")
    return MappingProxyType(
        {
            "selected_method_freeze": _artifact_reference_from_capability(selected),
            "development_bundle": _artifact_reference_from_capability(development_bundle),
            "test_evidence": _artifact_reference_from_capability(tests),
            "canary_result": _artifact_reference_from_capability(result),
            "canary_fresh_audit": _ArtifactReference(
                schema=FRESH_CANARY_AUDIT_SCHEMA,
                payload_sha256=fresh_canary_audit.payload_sha256,
                file_sha256=fresh_canary_audit.file_sha256,
                _issuer=_CAPABILITY_ISSUER,
            ),
        }
    )


def build_attempt_manifest(
    *,
    scientific_candidate_id: str,
    manifest_created_at_UTC: str,
    exact_target_timestamp_UTC: str,
    exact_HTTPS_endpoint: str,
    clean_identity: CleanGitIdentity,
    selected_method_freeze: ValidatedArtifactCapability,
    development_bundle: DevelopmentBundleCapability,
    test_evidence: ValidatedArtifactCapability,
    canary_result: CanaryArtifactCapability,
    fresh_canary_audit: FreshCanaryAuditCapability,
) -> Mapping[str, Any]:
    created = _parse_rfc3339_seconds(manifest_created_at_UTC, "attempt manifest creation time")
    target = _parse_rfc3339_seconds(exact_target_timestamp_UTC, "attempt manifest target time")
    if not created < target:
        raise ValidationError("attempt manifest creation time must be strictly before target")
    _validate_https_endpoint(
        exact_HTTPS_endpoint,
        exact_target_timestamp_UTC=exact_target_timestamp_UTC,
    )
    references = _validated_manifest_references(
        clean_identity=clean_identity,
        selected_method_freeze=selected_method_freeze,
        development_bundle=development_bundle,
        test_evidence=test_evidence,
        canary_result=canary_result,
        fresh_canary_audit=fresh_canary_audit,
    )
    if (
        scientific_candidate_id
        != selected_method_freeze._validated_payload["scientific_candidate_id"]
    ):
        raise AuthorityError("manifest scientific candidate differs from selected freeze")
    selected = references["selected_method_freeze"]
    statement = build_seed_statement(
        scientific_candidate_id=scientific_candidate_id,
        selected_method_freeze_payload_sha256=selected.payload_sha256,
        exact_target_timestamp_UTC=exact_target_timestamp_UTC,
        exact_HTTPS_endpoint=exact_HTTPS_endpoint,
    )
    content: dict[str, Any] = {
        "schema": ATTEMPT_MANIFEST_SCHEMA,
        "candidate_id": CANDIDATE_ID,
        "scientific_candidate_id": scientific_candidate_id,
        "manifest_created_at_UTC": manifest_created_at_UTC,
        "clean_commit": clean_identity.commit,
        "clean_tree": clean_identity.tree,
        "exact_target_timestamp_UTC": exact_target_timestamp_UTC,
        "exact_HTTPS_endpoint": exact_HTTPS_endpoint,
        "beacon_seed_derivation_primitive_schema": BEACON_SEED_DERIVATION_SCHEMA,
        "exact_seed_statement_base64": base64.b64encode(statement).decode("ascii"),
    }
    for name in _MANIFEST_REFERENCE_NAMES:
        reference = _require_artifact_reference(
            references[name],
            expected_schema=_MANIFEST_REFERENCE_SCHEMAS[name],
        )
        content[f"{name}_schema"] = reference.schema
        content[f"{name}_payload_sha256"] = reference.payload_sha256
        content[f"{name}_file_sha256"] = reference.file_sha256
    return seal_payload(content)


def validate_attempt_manifest(
    value: Mapping[str, Any],
    *,
    file_bytes: bytes,
    referenced_artifact_bytes: Mapping[str, bytes],
    selected_method_freeze: ValidatedArtifactCapability,
    development_bundle: DevelopmentBundleCapability,
    test_evidence: ValidatedArtifactCapability,
    canary_result: CanaryArtifactCapability,
    fresh_canary_audit: FreshCanaryAuditCapability,
) -> AttemptManifestCapability:
    raise AuthorityError("attempt-manifest authority is disabled until beacon selection")
    # The code below is retained as a reviewable future-amendment specification,
    # but is deliberately unreachable in this frozen authority revision.
    if frozenset(referenced_artifact_bytes) != frozenset(_MANIFEST_REFERENCE_NAMES):
        raise ValidationError("manifest parser requires the exact referenced artifact byte set")
    references = _validated_manifest_references(
        clean_identity=CleanGitIdentity(
            commit=fresh_canary_audit.clean_commit,
            tree=fresh_canary_audit.clean_tree,
        ),
        selected_method_freeze=selected_method_freeze,
        development_bundle=development_bundle,
        test_evidence=test_evidence,
        canary_result=canary_result,
        fresh_canary_audit=fresh_canary_audit,
    )
    capability_inputs = {
        "selected_method_freeze": selected_method_freeze,
        "development_bundle": development_bundle,
        "test_evidence": test_evidence,
        "canary_result": canary_result,
        "canary_fresh_audit": fresh_canary_audit,
    }
    exact_fields = {
        "selected_method_freeze": _SELECTED_FREEZE_FIELDS,
        "development_bundle": _DEVELOPMENT_BUNDLE_FIELDS,
        "test_evidence": _TEST_EVIDENCE_FIELDS,
        "canary_result": _CANARY_RESULT_FIELDS,
        "canary_fresh_audit": _FRESH_AUDIT_FIELDS,
    }
    for name in _MANIFEST_REFERENCE_NAMES:
        data = referenced_artifact_bytes[name]
        parsed_reference = parse_artifact_bytes(
            data,
            ArtifactSpec(
                schema=_MANIFEST_REFERENCE_SCHEMAS[name],
                exact_fields=exact_fields[name],
            ),
        )
        reference = references[name]
        if (
            parsed_reference["payload_sha256"] != reference.payload_sha256
            or file_sha256(data) != reference.file_sha256
        ):
            raise AuthorityError(f"manifest {name} bytes differ from validated capability")
        capability = capability_inputs[name]
        if type(capability) in {
            ValidatedArtifactCapability,
            CanaryArtifactCapability,
            FreshCanaryAuditCapability,
        } and canonical_json_bytes(parsed_reference) != canonical_json_bytes(
            capability._validated_payload
        ):
            raise AuthorityError(f"manifest {name} payload differs from validated capability")
    parsed = parse_artifact_bytes(
        file_bytes,
        ArtifactSpec(schema=ATTEMPT_MANIFEST_SCHEMA, exact_fields=_ATTEMPT_MANIFEST_FIELDS),
    )
    if canonical_json_bytes(parsed) != canonical_json_bytes(value):
        raise ValidationError("attempt manifest value differs from supplied canonical bytes")
    clean_identity = CleanGitIdentity(
        commit=str(parsed["clean_commit"]), tree=str(parsed["clean_tree"])
    )
    expected = build_attempt_manifest(
        scientific_candidate_id=str(parsed["scientific_candidate_id"]),
        manifest_created_at_UTC=str(parsed["manifest_created_at_UTC"]),
        exact_target_timestamp_UTC=str(parsed["exact_target_timestamp_UTC"]),
        exact_HTTPS_endpoint=str(parsed["exact_HTTPS_endpoint"]),
        clean_identity=clean_identity,
        selected_method_freeze=selected_method_freeze,
        development_bundle=development_bundle,
        test_evidence=test_evidence,
        canary_result=canary_result,
        fresh_canary_audit=fresh_canary_audit,
    )
    if canonical_json_bytes(parsed) != canonical_json_bytes(expected):
        raise ValidationError("attempt manifest differs from exact noncircular reconstruction")
    encoded = str(parsed["exact_seed_statement_base64"])
    try:
        decoded = base64.b64decode(encoded, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise ValidationError("attempt seed statement is not strict RFC4648 base64") from exc
    if base64.b64encode(decoded).decode("ascii") != encoded:
        raise ValidationError("attempt seed statement base64 is not canonical with padding")
    statement = build_seed_statement(
        scientific_candidate_id=str(parsed["scientific_candidate_id"]),
        selected_method_freeze_payload_sha256=references["selected_method_freeze"].payload_sha256,
        exact_target_timestamp_UTC=str(parsed["exact_target_timestamp_UTC"]),
        exact_HTTPS_endpoint=str(parsed["exact_HTTPS_endpoint"]),
    )
    if decoded != statement:
        raise ValidationError("attempt seed statement does not match reconstructed bytes")
    return AttemptManifestCapability(
        scientific_candidate_id=str(parsed["scientific_candidate_id"]),
        payload_sha256=str(parsed["payload_sha256"]),
        file_sha256=file_sha256(file_bytes),
        manifest_created_at_UTC=str(parsed["manifest_created_at_UTC"]),
        exact_target_timestamp_UTC=str(parsed["exact_target_timestamp_UTC"]),
        exact_HTTPS_endpoint=str(parsed["exact_HTTPS_endpoint"]),
        clean_commit=clean_identity.commit,
        clean_tree=clean_identity.tree,
        validated_payload=parsed,
        _issuer=_CAPABILITY_ISSUER,
    )


_AUTHORIZATION_SIGNED_FIELDS = frozenset(
    {
        "schema",
        "signature_domain",
        "signer_role",
        "candidate_id",
        "scientific_candidate_id",
        "attempt_id",
        "attempt_manifest_schema",
        "attempt_manifest_payload_sha256",
        "attempt_manifest_file_sha256",
        "exact_target_timestamp_UTC",
        "exact_HTTPS_endpoint",
        "signed_at_UTC",
        "one_time_execution_true",
    }
)
_AUTHORIZATION_FIELDS = frozenset(
    {
        "schema",
        "candidate_id",
        "signed_payload",
        "detached_signature_base64",
        "owner_authority_public_key_file_sha256",
        "owner_authority_ssh_fingerprint",
        "payload_sha256",
    }
)


@dataclass(frozen=True, slots=True, init=False)
class ExecutionAuthorizationCapability:
    schema: str
    candidate_id: str
    scientific_candidate_id: str
    payload_sha256: str
    file_sha256: str
    attempt_id: str
    attempt_manifest_payload_sha256: str
    attempt_manifest_file_sha256: str
    exact_target_timestamp_UTC: str
    exact_HTTPS_endpoint: str
    signed_at_UTC: str
    _validation_marker: object = field(repr=False, compare=False)

    def __init__(
        self,
        *,
        scientific_candidate_id: str,
        payload_sha256: str,
        file_sha256: str,
        attempt_id: str,
        attempt_manifest_payload_sha256: str,
        attempt_manifest_file_sha256: str,
        exact_target_timestamp_UTC: str,
        exact_HTTPS_endpoint: str,
        signed_at_UTC: str,
        _issuer: object,
    ) -> None:
        if _issuer is not _CAPABILITY_ISSUER:
            raise AuthorityError("ExecutionAuthorizationCapability has no valid issuer")
        object.__setattr__(self, "schema", EXECUTION_AUTHORIZATION_SCHEMA)
        object.__setattr__(self, "candidate_id", CANDIDATE_ID)
        object.__setattr__(self, "scientific_candidate_id", scientific_candidate_id)
        object.__setattr__(self, "payload_sha256", payload_sha256)
        object.__setattr__(self, "file_sha256", file_sha256)
        object.__setattr__(self, "attempt_id", attempt_id)
        object.__setattr__(self, "attempt_manifest_payload_sha256", attempt_manifest_payload_sha256)
        object.__setattr__(self, "attempt_manifest_file_sha256", attempt_manifest_file_sha256)
        object.__setattr__(self, "exact_target_timestamp_UTC", exact_target_timestamp_UTC)
        object.__setattr__(self, "exact_HTTPS_endpoint", exact_HTTPS_endpoint)
        object.__setattr__(self, "signed_at_UTC", signed_at_UTC)
        object.__setattr__(self, "_validation_marker", _issuer)


def require_execution_authorization_capability(
    value: object,
    *,
    manifest: AttemptManifestCapability,
) -> ExecutionAuthorizationCapability:
    manifest = require_attempt_manifest_capability(manifest)
    if type(value) is not ExecutionAuthorizationCapability:
        raise AuthorityError("exact ExecutionAuthorizationCapability required")
    capability = value
    if capability._validation_marker is not _CAPABILITY_ISSUER:
        raise AuthorityError("execution authorization capability marker is invalid")
    if (
        capability.schema != EXECUTION_AUTHORIZATION_SCHEMA
        or capability.candidate_id != CANDIDATE_ID
        or capability.scientific_candidate_id != manifest.scientific_candidate_id
        or capability.attempt_id != manifest.attempt_id
        or capability.attempt_manifest_payload_sha256 != manifest.payload_sha256
        or capability.attempt_manifest_file_sha256 != manifest.file_sha256
        or capability.exact_target_timestamp_UTC != manifest.exact_target_timestamp_UTC
        or capability.exact_HTTPS_endpoint != manifest.exact_HTTPS_endpoint
    ):
        raise AuthorityError("execution authorization is not bound to the exact manifest")
    _require_sha256(capability.payload_sha256, "execution authorization payload_sha256")
    _require_sha256(capability.file_sha256, "execution authorization file_sha256")
    signed = _parse_rfc3339_seconds(capability.signed_at_UTC, "authorization signed_at_UTC")
    created = _parse_rfc3339_seconds(manifest.manifest_created_at_UTC, "manifest created_at_UTC")
    target = _parse_rfc3339_seconds(manifest.exact_target_timestamp_UTC, "manifest target")
    if not created < signed < target:
        raise AuthorityError("authorization signature time is outside the exact open interval")
    return capability


def build_execution_authorization_signed_payload(
    manifest: AttemptManifestCapability,
    *,
    signed_at_UTC: str,
) -> Mapping[str, Any]:
    manifest = require_attempt_manifest_capability(manifest)
    signed = _parse_rfc3339_seconds(signed_at_UTC, "authorization signed_at_UTC")
    created = _parse_rfc3339_seconds(manifest.manifest_created_at_UTC, "manifest created_at_UTC")
    target = _parse_rfc3339_seconds(manifest.exact_target_timestamp_UTC, "manifest target")
    if not created < signed < target:
        raise ValidationError("authorization must be signed after manifest and before target")
    return MappingProxyType(
        {
            "schema": EXECUTION_AUTHORIZATION_PAYLOAD_SCHEMA,
            "signature_domain": OWNER_SIGNATURE_DOMAIN,
            "signer_role": OWNER_SIGNER_ROLE,
            "candidate_id": CANDIDATE_ID,
            "scientific_candidate_id": manifest.scientific_candidate_id,
            "attempt_id": manifest.attempt_id,
            "attempt_manifest_schema": manifest.schema,
            "attempt_manifest_payload_sha256": manifest.payload_sha256,
            "attempt_manifest_file_sha256": manifest.file_sha256,
            "exact_target_timestamp_UTC": manifest.exact_target_timestamp_UTC,
            "exact_HTTPS_endpoint": manifest.exact_HTTPS_endpoint,
            "signed_at_UTC": signed_at_UTC,
            "one_time_execution_true": True,
        }
    )


def build_execution_authorization_record(
    manifest: AttemptManifestCapability,
    *,
    signed_at_UTC: str,
    detached_signature_base64: str,
) -> Mapping[str, Any]:
    signed_payload = build_execution_authorization_signed_payload(
        manifest, signed_at_UTC=signed_at_UTC
    )
    try:
        signature = base64.b64decode(detached_signature_base64, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise ValidationError("detached owner signature is not strict RFC4648 base64") from exc
    if not signature or base64.b64encode(signature).decode("ascii") != detached_signature_base64:
        raise ValidationError("detached owner signature base64 is not canonical with padding")
    return seal_payload(
        {
            "schema": EXECUTION_AUTHORIZATION_SCHEMA,
            "candidate_id": CANDIDATE_ID,
            "signed_payload": dict(signed_payload),
            "detached_signature_base64": detached_signature_base64,
            "owner_authority_public_key_file_sha256": FROZEN_FILE_SHA256[
                OWNER_AUTHORITY_PUBLIC_KEY_PATH
            ],
            "owner_authority_ssh_fingerprint": OWNER_AUTHORITY_FINGERPRINT,
        }
    )


def validate_execution_authorization(
    value: Mapping[str, Any],
    *,
    file_bytes: bytes,
    manifest: AttemptManifestCapability,
    owner_public_key_bytes: bytes,
) -> ExecutionAuthorizationCapability:
    manifest = require_attempt_manifest_capability(manifest)
    parsed = parse_artifact_bytes(
        file_bytes,
        ArtifactSpec(
            schema=EXECUTION_AUTHORIZATION_SCHEMA,
            exact_fields=_AUTHORIZATION_FIELDS,
        ),
    )
    if canonical_json_bytes(parsed) != canonical_json_bytes(value):
        raise ValidationError("execution authorization value differs from supplied bytes")
    signed_payload = parsed["signed_payload"]
    if not isinstance(signed_payload, Mapping):
        raise ValidationError("execution authorization signed_payload must be an object")
    _require_exact_keys(
        signed_payload,
        _AUTHORIZATION_SIGNED_FIELDS,
        EXECUTION_AUTHORIZATION_PAYLOAD_SCHEMA,
    )
    expected_signed = build_execution_authorization_signed_payload(
        manifest, signed_at_UTC=str(signed_payload["signed_at_UTC"])
    )
    if canonical_json_bytes(signed_payload) != canonical_json_bytes(expected_signed):
        raise ValidationError("execution authorization signed payload binding mismatch")
    if (
        parsed["candidate_id"] != CANDIDATE_ID
        or parsed["owner_authority_public_key_file_sha256"]
        != FROZEN_FILE_SHA256[OWNER_AUTHORITY_PUBLIC_KEY_PATH]
        or parsed["owner_authority_ssh_fingerprint"] != OWNER_AUTHORITY_FINGERPRINT
    ):
        raise ValidationError("execution authorization trust-root record mismatch")
    exponent, modulus = _validate_owner_public_key_bytes(owner_public_key_bytes)
    from cryptography.exceptions import InvalidSignature
    from cryptography.hazmat.primitives import hashes
    from cryptography.hazmat.primitives.asymmetric import padding, rsa

    public_key = rsa.RSAPublicNumbers(exponent, modulus).public_key()
    encoded = str(parsed["detached_signature_base64"])
    try:
        signature = base64.b64decode(encoded, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise ValidationError("execution authorization signature encoding is malformed") from exc
    if base64.b64encode(signature).decode("ascii") != encoded:
        raise ValidationError("execution authorization signature encoding is noncanonical")
    try:
        public_key.verify(
            signature,
            canonical_json_bytes(signed_payload),
            padding.PSS(mgf=padding.MGF1(hashes.SHA256()), salt_length=hashes.SHA256().digest_size),
            hashes.SHA256(),
        )
    except InvalidSignature as exc:
        raise AuthorityError("owner RSA-PSS execution authorization signature is invalid") from exc
    capability = ExecutionAuthorizationCapability(
        scientific_candidate_id=manifest.scientific_candidate_id,
        payload_sha256=str(parsed["payload_sha256"]),
        file_sha256=file_sha256(file_bytes),
        attempt_id=manifest.attempt_id,
        attempt_manifest_payload_sha256=manifest.payload_sha256,
        attempt_manifest_file_sha256=manifest.file_sha256,
        exact_target_timestamp_UTC=manifest.exact_target_timestamp_UTC,
        exact_HTTPS_endpoint=manifest.exact_HTTPS_endpoint,
        signed_at_UTC=str(signed_payload["signed_at_UTC"]),
        _issuer=_CAPABILITY_ISSUER,
    )
    return require_execution_authorization_capability(capability, manifest=manifest)


_GLOBAL_CLAIM_FIELDS = frozenset(
    {
        "schema",
        "candidate_id",
        "scientific_candidate_id",
        "attempt_id",
        "attempt_manifest_schema",
        "attempt_manifest_payload_sha256",
        "attempt_manifest_file_sha256",
        "execution_authorization_schema",
        "execution_authorization_payload_sha256",
        "execution_authorization_file_sha256",
        "exact_target_timestamp_UTC",
        "exact_HTTPS_endpoint",
        "claimed_at_UTC",
        "payload_sha256",
    }
)


@dataclass(frozen=True, slots=True, init=False)
class GlobalClaimCapability:
    schema: str
    candidate_id: str
    scientific_candidate_id: str
    payload_sha256: str
    file_sha256: str
    attempt_id: str
    attempt_manifest_payload_sha256: str
    attempt_manifest_file_sha256: str
    execution_authorization_payload_sha256: str
    execution_authorization_file_sha256: str
    exact_target_timestamp_UTC: str
    exact_HTTPS_endpoint: str
    claimed_at_UTC: str
    _validation_marker: object = field(repr=False, compare=False)

    def __init__(
        self,
        *,
        scientific_candidate_id: str,
        payload_sha256: str,
        file_sha256: str,
        attempt_id: str,
        attempt_manifest_payload_sha256: str,
        attempt_manifest_file_sha256: str,
        execution_authorization_payload_sha256: str,
        execution_authorization_file_sha256: str,
        exact_target_timestamp_UTC: str,
        exact_HTTPS_endpoint: str,
        claimed_at_UTC: str,
        _issuer: object,
    ) -> None:
        if _issuer is not _CAPABILITY_ISSUER:
            raise AuthorityError("GlobalClaimCapability has no valid issuer")
        object.__setattr__(self, "schema", GLOBAL_CLAIM_SCHEMA)
        object.__setattr__(self, "candidate_id", CANDIDATE_ID)
        object.__setattr__(self, "scientific_candidate_id", scientific_candidate_id)
        object.__setattr__(self, "payload_sha256", payload_sha256)
        object.__setattr__(self, "file_sha256", file_sha256)
        object.__setattr__(self, "attempt_id", attempt_id)
        object.__setattr__(self, "attempt_manifest_payload_sha256", attempt_manifest_payload_sha256)
        object.__setattr__(self, "attempt_manifest_file_sha256", attempt_manifest_file_sha256)
        object.__setattr__(
            self,
            "execution_authorization_payload_sha256",
            execution_authorization_payload_sha256,
        )
        object.__setattr__(
            self,
            "execution_authorization_file_sha256",
            execution_authorization_file_sha256,
        )
        object.__setattr__(self, "exact_target_timestamp_UTC", exact_target_timestamp_UTC)
        object.__setattr__(self, "exact_HTTPS_endpoint", exact_HTTPS_endpoint)
        object.__setattr__(self, "claimed_at_UTC", claimed_at_UTC)
        object.__setattr__(self, "_validation_marker", _issuer)


def build_global_claim(
    manifest: AttemptManifestCapability,
    authorization: ExecutionAuthorizationCapability,
    *,
    claimed_at_UTC: str,
) -> Mapping[str, Any]:
    manifest = require_attempt_manifest_capability(manifest)
    authorization = require_execution_authorization_capability(authorization, manifest=manifest)
    claimed = _parse_rfc3339_seconds(claimed_at_UTC, "global claim claimed_at_UTC")
    target = _parse_rfc3339_seconds(manifest.exact_target_timestamp_UTC, "global claim target")
    if claimed < target:
        raise ValidationError("global claim cannot be created before the exact target")
    return seal_payload(
        {
            "schema": GLOBAL_CLAIM_SCHEMA,
            "candidate_id": CANDIDATE_ID,
            "scientific_candidate_id": manifest.scientific_candidate_id,
            "attempt_id": manifest.attempt_id,
            "attempt_manifest_schema": manifest.schema,
            "attempt_manifest_payload_sha256": manifest.payload_sha256,
            "attempt_manifest_file_sha256": manifest.file_sha256,
            "execution_authorization_schema": authorization.schema,
            "execution_authorization_payload_sha256": authorization.payload_sha256,
            "execution_authorization_file_sha256": authorization.file_sha256,
            "exact_target_timestamp_UTC": manifest.exact_target_timestamp_UTC,
            "exact_HTTPS_endpoint": manifest.exact_HTTPS_endpoint,
            "claimed_at_UTC": claimed_at_UTC,
        }
    )


def validate_global_claim(
    value: Mapping[str, Any],
    *,
    file_bytes: bytes,
    manifest: AttemptManifestCapability,
    authorization: ExecutionAuthorizationCapability,
) -> GlobalClaimCapability:
    parsed = parse_artifact_bytes(
        file_bytes,
        ArtifactSpec(schema=GLOBAL_CLAIM_SCHEMA, exact_fields=_GLOBAL_CLAIM_FIELDS),
    )
    if canonical_json_bytes(parsed) != canonical_json_bytes(value):
        raise ValidationError("global claim value differs from supplied canonical bytes")
    expected = build_global_claim(
        manifest, authorization, claimed_at_UTC=str(parsed["claimed_at_UTC"])
    )
    if canonical_json_bytes(parsed) != canonical_json_bytes(expected):
        raise ValidationError("global claim differs from exact reconstruction")
    return GlobalClaimCapability(
        scientific_candidate_id=manifest.scientific_candidate_id,
        payload_sha256=str(parsed["payload_sha256"]),
        file_sha256=file_sha256(file_bytes),
        attempt_id=manifest.attempt_id,
        attempt_manifest_payload_sha256=manifest.payload_sha256,
        attempt_manifest_file_sha256=manifest.file_sha256,
        execution_authorization_payload_sha256=authorization.payload_sha256,
        execution_authorization_file_sha256=authorization.file_sha256,
        exact_target_timestamp_UTC=manifest.exact_target_timestamp_UTC,
        exact_HTTPS_endpoint=manifest.exact_HTTPS_endpoint,
        claimed_at_UTC=str(parsed["claimed_at_UTC"]),
        _issuer=_CAPABILITY_ISSUER,
    )


def require_global_claim_capability(
    value: object,
    *,
    manifest: AttemptManifestCapability,
    authorization: ExecutionAuthorizationCapability,
) -> GlobalClaimCapability:
    manifest = require_attempt_manifest_capability(manifest)
    authorization = require_execution_authorization_capability(authorization, manifest=manifest)
    if type(value) is not GlobalClaimCapability:
        raise AuthorityError("exact GlobalClaimCapability required")
    capability = value
    if capability._validation_marker is not _CAPABILITY_ISSUER:
        raise AuthorityError("global claim capability marker is invalid")
    if (
        capability.schema != GLOBAL_CLAIM_SCHEMA
        or capability.candidate_id != CANDIDATE_ID
        or capability.scientific_candidate_id != manifest.scientific_candidate_id
        or capability.attempt_id != manifest.attempt_id
        or capability.attempt_manifest_payload_sha256 != manifest.payload_sha256
        or capability.attempt_manifest_file_sha256 != manifest.file_sha256
        or capability.execution_authorization_payload_sha256 != authorization.payload_sha256
        or capability.execution_authorization_file_sha256 != authorization.file_sha256
        or capability.exact_target_timestamp_UTC != manifest.exact_target_timestamp_UTC
        or capability.exact_HTTPS_endpoint != manifest.exact_HTTPS_endpoint
    ):
        raise AuthorityError("global claim capability binding mismatch")
    _require_sha256(capability.payload_sha256, "global claim payload_sha256")
    _require_sha256(capability.file_sha256, "global claim file_sha256")
    return capability


@dataclass(frozen=True, slots=True, init=False)
class PublishedGlobalClaimCapability:
    claim: GlobalClaimCapability = field(repr=False, compare=False)
    path: Path
    payload_sha256: str
    file_sha256: str
    attempt_manifest_payload_sha256: str
    attempt_manifest_file_sha256: str
    execution_authorization_payload_sha256: str
    execution_authorization_file_sha256: str
    _validation_marker: object = field(repr=False, compare=False)

    def __init__(
        self,
        *,
        claim: GlobalClaimCapability,
        _issuer: object,
    ) -> None:
        if _issuer is not _CAPABILITY_ISSUER:
            raise AuthorityError("PublishedGlobalClaimCapability has no valid issuer")
        object.__setattr__(self, "claim", claim)
        object.__setattr__(self, "path", SCIENTIFIC_GLOBAL_CLAIM_CANONICAL_PATH)
        object.__setattr__(self, "payload_sha256", claim.payload_sha256)
        object.__setattr__(self, "file_sha256", claim.file_sha256)
        object.__setattr__(
            self,
            "attempt_manifest_payload_sha256",
            claim.attempt_manifest_payload_sha256,
        )
        object.__setattr__(
            self,
            "attempt_manifest_file_sha256",
            claim.attempt_manifest_file_sha256,
        )
        object.__setattr__(
            self,
            "execution_authorization_payload_sha256",
            claim.execution_authorization_payload_sha256,
        )
        object.__setattr__(
            self,
            "execution_authorization_file_sha256",
            claim.execution_authorization_file_sha256,
        )
        object.__setattr__(self, "_validation_marker", _issuer)


def require_published_global_claim_capability(
    value: object,
) -> PublishedGlobalClaimCapability:
    if type(value) is not PublishedGlobalClaimCapability:
        raise AuthorityError("exact PublishedGlobalClaimCapability required")
    capability = value
    claim = capability.claim
    if (
        capability._validation_marker is not _CAPABILITY_ISSUER
        or type(claim) is not GlobalClaimCapability
        or claim._validation_marker is not _CAPABILITY_ISSUER
        or capability.path != SCIENTIFIC_GLOBAL_CLAIM_CANONICAL_PATH
        or capability.payload_sha256 != claim.payload_sha256
        or capability.file_sha256 != claim.file_sha256
        or capability.attempt_manifest_payload_sha256 != claim.attempt_manifest_payload_sha256
        or capability.attempt_manifest_file_sha256 != claim.attempt_manifest_file_sha256
        or capability.execution_authorization_payload_sha256
        != claim.execution_authorization_payload_sha256
        or capability.execution_authorization_file_sha256
        != claim.execution_authorization_file_sha256
    ):
        raise AuthorityError("published global-claim capability binding mismatch")
    return capability


@dataclass(frozen=True, slots=True, init=False)
class ScientificBeaconCapability:
    """Intrinsic-beacon proof; no issuer exists under the frozen contract."""

    schema: str
    candidate_id: str
    scientific_candidate_id: str
    attempt_id: str
    payload_sha256: str
    file_sha256: str
    attempt_manifest_payload_sha256: str
    attempt_manifest_file_sha256: str
    global_claim_payload_sha256: str
    global_claim_file_sha256: str
    exact_target_timestamp_UTC: str
    chain_index: int
    pulse_index: int
    output_value: str
    _validation_marker: object = field(repr=False, compare=False)

    def __init__(self, *_args: object, **_kwargs: object) -> None:
        raise AuthorityError(
            "scientific beacon authority is unavailable before exact future selection"
        )


def require_scientific_beacon_capability(value: object) -> ScientificBeaconCapability:
    if type(value) is not ScientificBeaconCapability:
        raise AuthorityError("exact ScientificBeaconCapability required")
    capability = value
    if (
        capability._validation_marker is not _CAPABILITY_ISSUER
        or capability.schema != ARTIFACT_SCHEMAS["scientific_beacon"]
        or capability.candidate_id != CANDIDATE_ID
        or capability.attempt_id != f"sha256-{capability.attempt_manifest_payload_sha256}"
    ):
        raise AuthorityError("scientific beacon capability identity mismatch")
    for name in (
        "payload_sha256",
        "file_sha256",
        "attempt_manifest_payload_sha256",
        "attempt_manifest_file_sha256",
        "global_claim_payload_sha256",
        "global_claim_file_sha256",
    ):
        _require_sha256(getattr(capability, name), f"scientific beacon {name}")
    _scientific_seed_digest(
        attempt_manifest_payload_sha256=capability.attempt_manifest_payload_sha256,
        exact_target_timestamp_UTC=capability.exact_target_timestamp_UTC,
        chain_index=capability.chain_index,
        pulse_index=capability.pulse_index,
        output_value=capability.output_value,
    )
    return capability


def validate_scientific_beacon(*_args: object, **_kwargs: object) -> ScientificBeaconCapability:
    """Fail closed until a future amendment pins an intrinsic NIST verifier."""

    raise AuthorityError(
        "no exact future beacon/endpoint is selected; intrinsic validation is unavailable"
    )


@dataclass(frozen=True, slots=True, init=False)
class ScientificSeedCapability:
    schema: str
    candidate_id: str
    scientific_candidate_id: str
    attempt_id: str
    attempt_manifest_payload_sha256: str
    exact_target_timestamp_UTC: str
    chain_index: int
    pulse_index: int
    output_value: str
    seed_digest_sha256: str
    full_unsigned_256_bit_root_seed: int
    seed_sequence_entropy: int
    bit_generator: str
    _validation_marker: object = field(repr=False, compare=False)

    def __init__(
        self,
        *,
        scientific_candidate_id: str,
        attempt_id: str,
        attempt_manifest_payload_sha256: str,
        exact_target_timestamp_UTC: str,
        chain_index: int,
        pulse_index: int,
        output_value: str,
        seed_digest_sha256: str,
        full_unsigned_256_bit_root_seed: int,
        _issuer: object,
    ) -> None:
        if _issuer is not _CAPABILITY_ISSUER:
            raise AuthorityError("ScientificSeedCapability has no valid issuer")
        object.__setattr__(self, "schema", SCIENTIFIC_SEED_CAPABILITY_SCHEMA)
        object.__setattr__(self, "candidate_id", CANDIDATE_ID)
        object.__setattr__(self, "scientific_candidate_id", scientific_candidate_id)
        object.__setattr__(self, "attempt_id", attempt_id)
        object.__setattr__(self, "attempt_manifest_payload_sha256", attempt_manifest_payload_sha256)
        object.__setattr__(self, "exact_target_timestamp_UTC", exact_target_timestamp_UTC)
        object.__setattr__(self, "chain_index", chain_index)
        object.__setattr__(self, "pulse_index", pulse_index)
        object.__setattr__(self, "output_value", output_value)
        object.__setattr__(self, "seed_digest_sha256", seed_digest_sha256)
        object.__setattr__(self, "full_unsigned_256_bit_root_seed", full_unsigned_256_bit_root_seed)
        object.__setattr__(self, "seed_sequence_entropy", full_unsigned_256_bit_root_seed)
        object.__setattr__(self, "bit_generator", "PCG64DXSM")
        object.__setattr__(self, "_validation_marker", _issuer)


def _scientific_seed_digest(
    *,
    attempt_manifest_payload_sha256: str,
    exact_target_timestamp_UTC: str,
    chain_index: int,
    pulse_index: int,
    output_value: str,
) -> str:
    _require_sha256(attempt_manifest_payload_sha256, "attempt manifest payload_sha256")
    _parse_rfc3339_seconds(exact_target_timestamp_UTC, "scientific seed target")
    if (
        not isinstance(chain_index, int)
        or isinstance(chain_index, bool)
        or chain_index < 0
        or not isinstance(pulse_index, int)
        or isinstance(pulse_index, bool)
        or pulse_index < 0
    ):
        raise ValidationError("beacon chain and pulse indexes must be nonnegative integers")
    if not isinstance(output_value, str) or not _OUTPUT_VALUE_RE.fullmatch(output_value):
        raise ValidationError("beacon outputValue must be exactly 128 uppercase hex characters")
    return hashlib.sha256(
        _ascii_lines(
            (
                SCIENTIFIC_SEED_DOMAIN,
                attempt_manifest_payload_sha256,
                exact_target_timestamp_UTC,
                str(chain_index),
                str(pulse_index),
                output_value,
            )
        )
    ).hexdigest()


def derive_scientific_seed(
    *,
    beacon: ScientificBeaconCapability,
) -> ScientificSeedCapability:
    """Derive only from a nominal intrinsically validated scientific beacon."""

    beacon = require_scientific_beacon_capability(beacon)
    digest = _scientific_seed_digest(
        attempt_manifest_payload_sha256=beacon.attempt_manifest_payload_sha256,
        exact_target_timestamp_UTC=beacon.exact_target_timestamp_UTC,
        chain_index=beacon.chain_index,
        pulse_index=beacon.pulse_index,
        output_value=beacon.output_value,
    )
    root_seed = int.from_bytes(bytes.fromhex(digest), byteorder="big", signed=False)
    return ScientificSeedCapability(
        scientific_candidate_id=beacon.scientific_candidate_id,
        attempt_id=beacon.attempt_id,
        attempt_manifest_payload_sha256=beacon.attempt_manifest_payload_sha256,
        exact_target_timestamp_UTC=beacon.exact_target_timestamp_UTC,
        chain_index=beacon.chain_index,
        pulse_index=beacon.pulse_index,
        output_value=beacon.output_value,
        seed_digest_sha256=digest,
        full_unsigned_256_bit_root_seed=root_seed,
        _issuer=_CAPABILITY_ISSUER,
    )


def require_scientific_seed_capability(
    value: object,
    *,
    manifest: AttemptManifestCapability | None = None,
) -> ScientificSeedCapability:
    """Reject canary capabilities, mappings, and subclasses at the scientific seam."""

    if type(value) is not ScientificSeedCapability:
        raise AuthorityError("exact ScientificSeedCapability required")
    capability = value
    if capability._validation_marker is not _CAPABILITY_ISSUER:
        raise AuthorityError("scientific seed capability marker is invalid")
    if (
        capability.schema != SCIENTIFIC_SEED_CAPABILITY_SCHEMA
        or capability.candidate_id != CANDIDATE_ID
    ):
        raise AuthorityError("scientific seed capability identity mismatch")
    expected = _scientific_seed_digest(
        attempt_manifest_payload_sha256=capability.attempt_manifest_payload_sha256,
        exact_target_timestamp_UTC=capability.exact_target_timestamp_UTC,
        chain_index=capability.chain_index,
        pulse_index=capability.pulse_index,
        output_value=capability.output_value,
    )
    root_seed = int.from_bytes(bytes.fromhex(expected), byteorder="big", signed=False)
    if (
        capability.seed_digest_sha256 != expected
        or capability.full_unsigned_256_bit_root_seed != root_seed
        or capability.seed_sequence_entropy != root_seed
        or capability.bit_generator != "PCG64DXSM"
        or capability.attempt_id != f"sha256-{capability.attempt_manifest_payload_sha256}"
    ):
        raise AuthorityError("scientific seed capability derivation binding mismatch")
    if manifest is not None:
        manifest = require_attempt_manifest_capability(manifest)
        if (
            capability.scientific_candidate_id != manifest.scientific_candidate_id
            or capability.attempt_id != manifest.attempt_id
            or capability.attempt_manifest_payload_sha256 != manifest.payload_sha256
            or capability.exact_target_timestamp_UTC != manifest.exact_target_timestamp_UTC
        ):
            raise AuthorityError("scientific seed capability is bound to another manifest")
    return capability


def scientific_seed_sequence(capability: ScientificSeedCapability) -> np.random.SeedSequence:
    capability = require_scientific_seed_capability(capability)
    import numpy as np

    return np.random.SeedSequence(entropy=capability.seed_sequence_entropy)


def assert_scientific_execution_authorized() -> None:
    raise AuthorityError("current V3 contract does not authorize scientific execution")


def assert_human_EEG_outcome_authorized() -> None:
    raise AuthorityError("current V3 contract does not authorize human EEG outcomes")


def scientific_rng(capability: ScientificSeedCapability) -> np.random.Generator:
    """Scientific executor boundary; current pre-outcome contract always rejects."""

    capability = require_scientific_seed_capability(capability)
    assert_scientific_execution_authorized()
    import numpy as np

    return np.random.Generator(np.random.PCG64DXSM(scientific_seed_sequence(capability)))


def publish_scientific_global_claim(
    manifest: AttemptManifestCapability,
    authorization: ExecutionAuthorizationCapability,
) -> PublishedGlobalClaimCapability:
    """Observe claim time and publish once at the fixed candidate-wide path."""

    assert_scientific_execution_authorized()
    manifest = require_attempt_manifest_capability(manifest)
    authorization = require_execution_authorization_capability(
        authorization,
        manifest=manifest,
    )
    claimed_at_UTC = (
        datetime.now(timezone.utc).replace(microsecond=0).strftime("%Y-%m-%dT%H:%M:%SZ")
    )
    value = build_global_claim(
        manifest,
        authorization,
        claimed_at_UTC=claimed_at_UTC,
    )
    data = artifact_bytes(value)
    claim = validate_global_claim(
        value,
        file_bytes=data,
        manifest=manifest,
        authorization=authorization,
    )
    published = _publish_write_once(
        V3_CANONICAL_ROOT,
        "scientific/global-claim.json",
        data,
        create_root=True,
        _publisher=_PUBLICATION_ISSUER,
    )
    if published.path != SCIENTIFIC_GLOBAL_CLAIM_CANONICAL_PATH:
        raise PublicationError("global claim resolved outside the fixed canonical path")
    return PublishedGlobalClaimCapability(
        claim=claim,
        _issuer=_CAPABILITY_ISSUER,
    )
