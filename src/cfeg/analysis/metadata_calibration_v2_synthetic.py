from __future__ import annotations

import hashlib
import inspect
import io
import json
import math
import os
import platform as platform_module
import re
import stat
import subprocess
import sys
import urllib.error
import urllib.request
from collections.abc import Mapping, Sequence
from contextlib import redirect_stdout
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import scipy
import yaml
from scipy.stats import beta as beta_distribution
from scipy.stats import t as student_t

from cfeg.baselines.fbcca import (
    apply_filterbank,
    cca_score,
    make_reference_signals,
    resolve_filterbank_parameters,
)
from cfeg.models import metadata_calibration_v2 as v2_operator

SYNTHETIC_PLAN_SCHEMA = "cfeg.metadata-calibration-v2-synthetic-plan.v1"
SYNTHETIC_RESULT_SCHEMA = "cfeg.metadata-calibration-v2-synthetic-result.v1"
SYNTHETIC_PREPARATION_SCHEMA = "cfeg.metadata-calibration-v2-synthetic-preparation.v1"
SYNTHETIC_DEVELOPMENT_EVIDENCE_SCHEMA = (
    "cfeg.metadata-calibration-v2-synthetic-development-evidence.v1"
)
SYNTHETIC_LOCKBOX_AUTHORIZATION_SCHEMA = (
    "cfeg.metadata-calibration-v2-synthetic-lockbox-authorization.v1"
)
SYNTHETIC_LOCKBOX_CLAIM_SCHEMA = "cfeg.metadata-calibration-v2-synthetic-lockbox-claim.v1"
SYNTHETIC_FBCCA_CACHE_SCHEMA = "cfeg.metadata-calibration-v2-synthetic-fbcca-cache.v1"
SYNTHETIC_TEST_EVIDENCE_SCHEMA = "cfeg.metadata-calibration-v2-synthetic-test-evidence.v1"
SYNTHETIC_BEACON_RECEIPT_SCHEMA = "cfeg.metadata-calibration-v2-synthetic-beacon.v1"
SYNTHETIC_TERMINAL_RECEIPT_SCHEMA = "cfeg.metadata-calibration-v2-synthetic-terminal.v1"

EXPECTED_SYNTHETIC_PLAN_SHA256 = "9a9683141ccd3d1fe9cf094aad955c6e317d112a73e35fbcfcdd2ef7d06e1f8c"
EXPECTED_FILTERBANK_SHA256 = "b8c1ce4477d980b40f359bc3dc97f3125313191d24401f29a7e2dc473d0d1380"
DEVELOPMENT_ROOT_SEED = 20260906

FAMILY_NAMES = (
    "B1_participant_specific_class_confusion",
    "B2_participant_phase_spatial_shift",
    "B3_context_dependent_support_query_shift",
    "N1_perfect_clean_anchor",
    "N2_random_fit_support_labels",
    "N3_nonstationary_corrupted_calibration",
    "N4_context_independent_of_signal",
)
CONTEXT_CONTROLS = ("correct", "pair_shuffled", "stale", "opposite_interface", "all_missing")
SUPPORT_BUDGETS = (0, 1, 3, 5)
EAUC_BUDGET_WEIGHTS = {0: 1.0 / 6.0, 1: 1.0 / 2.0, 3: 1.0 / 3.0}
P2_SUBBAND_WEIGHTS = np.asarray(
    [
        1.25,
        0.6704482076268572,
        0.5032785618838642,
        0.42677669529663687,
        0.38374806099528436,
        0.35649051737437876,
        0.33782687899303776,
    ],
    dtype=np.float64,
)

_REPOSITORY = Path(__file__).resolve().parents[3]
DEFAULT_SYNTHETIC_PLAN_PATH = (
    _REPOSITORY / "configs/analysis/metadata_calibration_v2_synthetic.yaml"
)
DEFAULT_FILTERBANK_PATH = (
    _REPOSITORY / "configs/baselines/fbcca_chen2015_m3_v2_explicit_weights.yaml"
)
_SHA256_RE = re.compile(r"[0-9a-f]{64}")
_GIT_OBJECT_RE = re.compile(r"[0-9a-f]{40,64}")
_UPPER_HEX_128_RE = re.compile(r"[0-9A-F]{128}")
_LOWER_HEX_128_RE = re.compile(r"[0-9a-f]{128}")
_UPPER_HEX_1024_RE = re.compile(r"[0-9A-F]{1024}")
_GOVERNED_SEED_SENTINEL = object()
_DYNAMIC_RESERVED_SEEDS: set[int] = set()
_VALIDATED_BEACON_SEED_CACHE: dict[str, tuple[tuple[int, int, int], int]] = {}
_THREAD_ENVIRONMENT_NAMES = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "BLIS_NUM_THREADS",
)


@dataclass(frozen=True)
class SyntheticContract:
    plan_path: Path
    plan_sha256: str
    filterbank_path: Path
    filterbank_sha256: str
    diagnostic_operator_configs_sha256: str
    plan: Mapping[str, Any]
    filterbank: Mapping[str, Any]


@dataclass(frozen=True)
class SyntheticParticipant:
    family: str
    participant_index: int
    root_seed: int
    signals: np.ndarray
    true_labels: np.ndarray
    recorded_support_labels: np.ndarray
    interfaces: np.ndarray
    impedance_kohm: np.ndarray
    signal_states: np.ndarray
    context_states: np.ndarray
    generated_target_classes: np.ndarray
    generated_cross_classes: np.ndarray
    innovation_sds: np.ndarray
    cross_extra_phases: np.ndarray
    partition_sha256s: tuple[str, ...]


@dataclass(frozen=True)
class StrictFBCCAProduct:
    scores: np.ndarray
    subbands: np.ndarray
    subband_weights: np.ndarray
    producer_sha256: str


@dataclass(frozen=True)
class GitIdentity:
    commit: str
    tree: str
    source_bundle_sha256: str
    clean: bool
    commit_timestamp_utc: str


@dataclass(frozen=True)
class _GovernedSeedContext:
    """Ordinary accidental-use guard, not a Python-introspection security boundary."""

    contract_sha256: str
    phase: str
    root_seed: int
    sentinel: object


class HardInvariantFailure(RuntimeError):
    pass


class _RejectHTTPRedirect(urllib.request.HTTPRedirectHandler):
    """Fail before issuing a second request to a redirect Location."""

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        del newurl
        raise urllib.error.HTTPError(
            req.full_url,
            code,
            "Redirect forbidden by the frozen NIST transport contract",
            headers,
            fp,
        )


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _open_parent_directory_nofollow(path: str | Path, name: str) -> tuple[int, Path]:
    """Pin every absolute parent component without following symbolic links."""

    target = Path(path).expanduser().absolute()
    if target.name in {"", ".", ".."}:
        raise ValueError(f"{name} must name a file below an absolute parent directory.")
    flags = (
        os.O_RDONLY
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_CLOEXEC", 0)
    )
    descriptor = os.open(os.sep, flags)
    try:
        for component in target.parent.parts[1:]:
            next_descriptor = os.open(component, flags, dir_fd=descriptor)
            os.close(descriptor)
            descriptor = next_descriptor
        parent_stat = os.fstat(descriptor)
        if not stat.S_ISDIR(parent_stat.st_mode):
            raise ValueError(f"{name} parent is not a directory.")
        return descriptor, target
    except OSError as error:
        os.close(descriptor)
        raise ValueError(f"{name} has an unavailable or symlinked parent component.") from error
    except BaseException:
        os.close(descriptor)
        raise


def _safe_regular_file_stat(path: str | Path, name: str) -> tuple[Path, os.stat_result]:
    parent_descriptor, target = _open_parent_directory_nofollow(path, name)
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_CLOEXEC", 0)
    try:
        descriptor = os.open(target.name, flags, dir_fd=parent_descriptor)
        try:
            observed = os.fstat(descriptor)
            if not stat.S_ISREG(observed.st_mode):
                raise ValueError(f"{name} must be one nonsymlink regular file: {target}")
            return target, observed
        finally:
            os.close(descriptor)
    except OSError as error:
        raise ValueError(f"{name} must be one existing nonsymlink regular file: {target}") from error
    finally:
        os.close(parent_descriptor)


def _read_bytes_nofollow(path: str | Path, name: str) -> bytes:
    parent_descriptor, target = _open_parent_directory_nofollow(path, name)
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_CLOEXEC", 0)
    try:
        descriptor = os.open(target.name, flags, dir_fd=parent_descriptor)
        try:
            if not stat.S_ISREG(os.fstat(descriptor).st_mode):
                raise ValueError(f"{name} must be a nonsymlink regular file.")
            chunks: list[bytes] = []
            while True:
                chunk = os.read(descriptor, 1024 * 1024)
                if not chunk:
                    break
                chunks.append(chunk)
            return b"".join(chunks)
        finally:
            os.close(descriptor)
    except OSError as error:
        raise ValueError(f"{name} must be one existing nonsymlink regular file: {target}") from error
    finally:
        os.close(parent_descriptor)


def _sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    parent_descriptor, target = _open_parent_directory_nofollow(path, "hashed file")
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_CLOEXEC", 0)
    try:
        descriptor = os.open(target.name, flags, dir_fd=parent_descriptor)
        try:
            if not stat.S_ISREG(os.fstat(descriptor).st_mode):
                raise ValueError(f"Hashed path is not a regular file: {target}")
            while True:
                chunk = os.read(descriptor, 1024 * 1024)
                if not chunk:
                    break
                digest.update(chunk)
        finally:
            os.close(descriptor)
    finally:
        os.close(parent_descriptor)
    return digest.hexdigest()


def _canonical_json_bytes(value: Mapping[str, Any]) -> bytes:
    return json.dumps(
        dict(value),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _canonical_json_sha256(value: Mapping[str, Any]) -> str:
    return _sha256_bytes(_canonical_json_bytes(value))


def _array_sha256(value: np.ndarray) -> str:
    array = np.asarray(value)
    if array.dtype == object:
        payload = {
            "dtype": "object",
            "shape": list(array.shape),
            "values": [None if item is None else str(item) for item in array.ravel().tolist()],
        }
        return _canonical_json_sha256(payload)
    contiguous = np.ascontiguousarray(array)
    header = json.dumps(
        {"dtype": contiguous.dtype.str, "shape": list(contiguous.shape)},
        sort_keys=True,
        separators=(",", ":"),
    ).encode("ascii")
    return _sha256_bytes(header + b"\n" + contiguous.tobytes(order="C"))


def _require_exact_keys(value: Mapping[str, Any], expected: set[str], name: str) -> None:
    observed = set(value)
    if observed != expected:
        raise ValueError(
            f"{name} must have its exact frozen fields; "
            f"missing={sorted(expected - observed)}, extra={sorted(observed - expected)}."
        )


def _reject_symlink_ancestors(path: str | Path, name: str) -> None:
    absolute = Path(path).expanduser().absolute()
    for candidate in (absolute, *absolute.parents):
        try:
            mode = candidate.lstat().st_mode
        except FileNotFoundError:
            continue
        if stat.S_ISLNK(mode):
            raise ValueError(f"{name} has a forbidden symlink ancestor: {candidate}")


def _regular_nonsymlink_file(path: str | Path, name: str) -> Path:
    resolved, _ = _safe_regular_file_stat(path, name)
    return resolved


def _readonly_receipt_file(path: str | Path, name: str) -> Path:
    source, observed = _safe_regular_file_stat(path, name)
    if stat.S_IMODE(observed.st_mode) != stat.S_IRUSR:
        raise ValueError(f"{name} must have exact read-only owner mode 0400.")
    return source


def _reserved_seed_values(contract: SyntheticContract) -> frozenset[int]:
    rng = contract.plan["rng"]
    values = {
        int(rng["development_root_seed"]),
        int(rng["superseded_unexecuted_plaintext_lockbox_root_seed"]),
        *_DYNAMIC_RESERVED_SEEDS,
    }
    beacon_path = canonical_execution_path(contract, "beacon_receipt")
    if beacon_path.exists() or beacon_path.is_symlink():
        # Re-derive from the canonical receipt on every fresh-process public API
        # path.  The process-local set is only a cache, never the source of truth.
        file_stat = beacon_path.stat()
        fingerprint = (file_stat.st_ino, file_stat.st_size, file_stat.st_mtime_ns)
        cached = _VALIDATED_BEACON_SEED_CACHE.get(str(beacon_path))
        if cached is None or cached[0] != fingerprint:
            derived = _derive_lockbox_seed_from_beacon_receipt(contract, beacon_path)
            _VALIDATED_BEACON_SEED_CACHE[str(beacon_path)] = (fingerprint, derived)
        else:
            derived = cached[1]
        _DYNAMIC_RESERVED_SEEDS.add(derived)
        values.add(derived)
    return frozenset(values)


def _governed_seed_context(
    contract: SyntheticContract,
    *,
    phase: str,
    root_seed: int,
) -> _GovernedSeedContext:
    return _GovernedSeedContext(
        contract_sha256=contract.plan_sha256,
        phase=phase,
        root_seed=root_seed,
        sentinel=_GOVERNED_SEED_SENTINEL,
    )


def _require_seed_access(
    contract: SyntheticContract,
    root_seed: int,
    governed_context: _GovernedSeedContext | None,
) -> None:
    """Block accidental reserved-seed use through ordinary public APIs.

    This deliberately is not presented as protection from hostile Python
    introspection.  It is a fail-closed guard against an analyst accidentally
    reaching a reserved stream outside the named governed phase.
    """

    if root_seed not in _reserved_seed_values(contract):
        return
    if not (
        isinstance(governed_context, _GovernedSeedContext)
        and governed_context.sentinel is _GOVERNED_SEED_SENTINEL
        and governed_context.contract_sha256 == contract.plan_sha256
        and governed_context.root_seed == root_seed
        and governed_context.phase in {"development", "lockbox", "development-validation"}
    ):
        raise ValueError("Reserved seed requires an internal governed execution context.")


def canonical_execution_path(contract: SyntheticContract, name: str) -> Path:
    paths = contract.plan["freeze_and_stopping"]["canonical_execution_paths"]
    if name not in paths:
        raise ValueError(f"Unknown canonical synthetic artifact {name!r}.")
    path = Path(str(paths[name]))
    if not path.is_absolute():
        raise ValueError("Frozen synthetic execution paths must be absolute.")
    return path


def prepare_canonical_artifact_directories(contract: SyntheticContract) -> None:
    """Create only the immediate governed artifact directories as private directories."""

    paths = {
        canonical_execution_path(contract, name).parent
        for name in contract.plan["freeze_and_stopping"]["canonical_execution_paths"]
    }
    for directory in sorted(paths, key=str):
        _reject_symlink_ancestors(directory, "canonical artifact directory")
        directory.mkdir(mode=0o700, parents=True, exist_ok=True)
        _reject_symlink_ancestors(directory, "canonical artifact directory")
        if stat.S_IMODE(directory.stat().st_mode) != 0o700:
            raise ValueError(f"Canonical artifact directory must have exact mode 0700: {directory}")


def _require_canonical_path(contract: SyntheticContract, name: str, path: str | Path) -> Path:
    expected = canonical_execution_path(contract, name)
    observed = Path(path).expanduser().absolute()
    if observed != expected:
        raise ValueError(f"{name} must use the exact canonical path {expected}.")
    return observed


def validate_synthetic_contract(
    plan_path: str | Path = DEFAULT_SYNTHETIC_PLAN_PATH,
) -> SyntheticContract:
    """Load the byte-frozen v11 plan and its separately frozen FBCCA contract."""

    path = _regular_nonsymlink_file(plan_path, "synthetic plan")
    plan_bytes = _read_bytes_nofollow(path, "synthetic plan")
    observed_sha = _sha256_bytes(plan_bytes)
    if observed_sha != EXPECTED_SYNTHETIC_PLAN_SHA256:
        raise ValueError(
            "Synthetic plan byte hash drifted from frozen v11: "
            f"expected {EXPECTED_SYNTHETIC_PLAN_SHA256}, observed {observed_sha}."
        )
    decoded = yaml.safe_load(plan_bytes.decode("utf-8"))
    if not isinstance(decoded, Mapping):
        raise TypeError("Synthetic plan must decode to one mapping.")
    plan = dict(decoded)
    if (
        plan.get("schema") != SYNTHETIC_PLAN_SCHEMA
        or plan.get("candidate_id") != "metadata-calibration-efficiency-v2"
        or plan.get("generator_revision") != "v11_terminal_receipt_precedence"
        or plan.get("status")
        != "frozen_after_v10_interrupted_development_before_any_lockbox_seed_or_outcome"
    ):
        raise ValueError("Synthetic plan identity or pre-outcome freeze status is invalid.")

    rng = plan.get("rng")
    population = plan.get("population")
    waveform = plan.get("waveform")
    diagnostics = plan.get("diagnostic_operator_configs")
    families = plan.get("families")
    if not all(
        isinstance(item, Mapping) for item in (rng, population, waveform, diagnostics, families)
    ):
        raise TypeError("Synthetic plan sections must be mappings.")
    if (
        rng.get("bit_generator") != "numpy.random.PCG64DXSM"
        or rng.get("construction") != "numpy.random.SeedSequence"
        or rng.get("key_order")
        != [
            "root_seed",
            "family_code",
            "participant_index",
            "block",
            "class_index",
            "component_code",
        ]
        or rng.get("development_root_seed") != DEVELOPMENT_ROOT_SEED
        or not isinstance(rng.get("superseded_unexecuted_plaintext_lockbox_root_seed"), int)
        or not isinstance(rng.get("independent_lockbox_root_seed"), Mapping)
        or tuple(rng.get("family_codes", ())) != FAMILY_NAMES
    ):
        raise ValueError("Synthetic RNG contract differs from the frozen keyed construction.")
    if (
        population.get("participants_per_family_per_seed") != 64
        or population.get("classes") != 12
        or population.get("complete_blocks") != 10
        or population.get("trials_per_class_per_block") != 1
        or tuple(population.get("support_budgets", ())) != SUPPORT_BUDGETS
        or population.get("immutable_query_blocks") != [6, 7, 8, 9, 10]
    ):
        raise ValueError("Synthetic population or budget contract drifted.")
    if tuple(families) != FAMILY_NAMES:
        raise ValueError("Synthetic family order or membership drifted.")
    b3 = families["B3_context_dependent_support_query_shift"]
    n4 = families["N4_context_independent_of_signal"]
    if (
        b3.get("state_minus_one_odd_block_regime")
        != {
            "cross_frequency_class": "matched_cross_frequency_class",
            "extra_phase_radians": 0.0,
            "noise_innovation_sd": 0.85,
        }
        or b3.get("state_plus_one_even_block_regime")
        != {
            "cross_frequency_class": "opposite_cross_frequency_class",
            "extra_phase_radians": 1.5707963267948966,
            "noise_innovation_sd": 1.3,
        }
        or n4.get("signal_state_draw") != "keyed_Bernoulli_one_half_mapped_to_minus_one_or_plus_one"
        or n4.get("context_state_draw")
        != "separate_keyed_Bernoulli_one_half_mapped_to_minus_one_or_plus_one"
    ):
        raise ValueError("Synthetic B3/N4 state mapping is not exact.")
    expected_axes = {
        "spatial_signature": "participant_by_intended_class_by_channel_shared_across_blocks",
        "class_phase": "participant_by_frequency_class_shared_across_blocks",
        "block_phase_drift": "participant_by_block_shared_across_classes_and_channels",
        "channel_gain": "participant_by_block_by_channel_shared_across_all_class_trials",
        "cross_frequency_spatial_signature": "intended_class_spatial_signature",
        "impedance_noise": "participant_by_block_by_channel_shared_across_all_class_trials",
    }
    if waveform.get("random_variable_axes") != expected_axes:
        raise ValueError("Synthetic v4 random-variable axis contract is not exact.")
    if waveform.get("component_phase_formulas") != {
        "target_harmonic_h": (
            "h_times_parenthesized_target_frequency_class_phase_plus_block_phase_drift"
        ),
        "cross_frequency_harmonic_h": (
            "h_times_parenthesized_cross_frequency_class_phase_plus_cross_extra_phase_plus_"
            "block_phase_drift"
        ),
    }:
        raise ValueError("Synthetic v5 component phase formulas are not exact.")
    if waveform.get("exact_signal_composition") != {
        "target": (
            "target_amplitude times intended_class_spatial_signature times the sum of "
            "harmonic 1 and 0.35 times harmonic 2 at generated_target_class frequency "
            "and phase"
        ),
        "cross_frequency": (
            "cross_amplitude times the same intended_class_spatial_signature times the "
            "sum of harmonic 1 and 0.35 times harmonic 2 at generated_cross_class "
            "frequency and phase plus cross_extra_phase"
        ),
        "channel_gain_placement": "multiply_target_plus_cross_only_not_AR1_noise",
        "final": ("channel_gain_times_parenthesized_target_plus_cross_plus_stationary_AR1_noise"),
    }:
        raise ValueError("Synthetic v9 signal-composition contract is not exact.")
    if waveform.get("noise") != {
        "process": "stationary_AR1_per_channel",
        "rho": 0.55,
        "initialization": "epsilon_0_equals_sigma_times_z_0_div_sqrt_1_minus_rho_squared",
        "recurrence": "epsilon_t_equals_rho_times_epsilon_t_minus_1_plus_sigma_times_z_t",
    }:
        raise ValueError("Synthetic v9 AR(1) contract is not exact.")
    beacon = dict(rng["independent_lockbox_root_seed"])
    if beacon != {
        "derivation_revision": "nist_beacon_v1",
        "provider": "NIST_Randomness_Beacon_2.0",
        "official_specification": (
            "https://csrc.nist.gov/Projects/interoperable-randomness-beacons/beacon-20"
        ),
        "target_timestamp_utc": "2026-09-05T21:30:00.000Z",
        "target_timestamp_unix_milliseconds": 1788643800000,
        "exact_endpoint": ("https://beacon.nist.gov/beacon/2.0/pulse/time/1788643800000"),
        "required_exact_pulse_timestamp": "2026-09-05T21:30:00.000Z",
        "statement_utf8": (
            "cfeg.metadata-calibration-efficiency-v2|synthetic-lockbox|v11|2026-09-05T21:30:00.000Z"
        ),
        "statement_sha256": ("33d8e8a1ff6b6415ccce2fcd3fbd47794b975e65e725acba66000c2fe1092fe7"),
        "output_value_encoding": ("exact_128_uppercase_hex_characters_decoded_to_64_bytes"),
        "digest_formula": "SHA256_UTF8_statement_then_LF_then_raw_output_value_bytes",
        "root_seed_formula": "unsigned_big_endian_integer_of_all_32_digest_bytes",
        "retrieval_not_before_target_timestamp": True,
        "require_https_nist_host_exact_pulse_fields_and_full_response_receipt": True,
        "authorization_must_be_sealed_before_target_timestamp": True,
        "standalone_beacon_fetch_phase": "forbidden",
        "transaction_order": [
            "validate_pretarget_authorization_and_all_prelockbox_evidence",
            "create_canonical_global_O_EXCL_claim_before_network_access",
            "fetch_exact_NIST_HTTPS_endpoint_without_dependency_injection_or_redirect",
            "verify_exact_pulse_and_deployed_protocol_SHA512_output_value",
            "derive_seed_material_and_persist_claim_bound_beacon_receipt",
            "execute_reserved_seed_once",
            "persist_terminal_result",
        ],
        "deployed_json_serialization_revision": (
            "nist_v2_cipher0_four_byte_length_prefixes_raw_signature"
        ),
        "transport_authentication": (
            "default_verified_TLS_to_exact_beacon_dot_nist_dot_gov"
        ),
        "local_RSA_signature_verification": (
            "unavailable_with_current_service_certificate_signature_size_mismatch"
        ),
    }:
        raise ValueError("Synthetic v11 future-beacon derivation contract is not exact.")
    expected_paths = {
        "preparation_receipt",
        "development_result",
        "full_suite_test_evidence",
        "beacon_receipt",
        "lockbox_authorization",
        "lockbox_result",
        "lockbox_terminal_receipt",
        "seed_global_lockbox_claim",
    }
    path_values = plan.get("freeze_and_stopping", {}).get("canonical_execution_paths", {})
    if set(path_values) != expected_paths or any(
        not Path(str(value)).is_absolute() for value in path_values.values()
    ):
        raise ValueError("Synthetic v11 canonical execution paths are not exact absolute paths.")
    severe = plan.get("promotion_requirements", {}).get("severe_harm", {})
    if severe.get("contrast_ids") != [
        "B1:P1_A_Q-A0:eAUC",
        "B2:P2_A_Q-A0:eAUC",
        "B3:P1_A_QM_correct-P1_A_Q:eAUC",
        "N1:P1_A_Q-A0:eAUC",
        "N1:P2_A_Q-A0:eAUC",
        "N2:P1_A_Q-A0:eAUC",
        "N3:P1_A_Q-A0:eAUC",
        "N4:P1_A_QM_correct-P1_A_Q:eAUC",
    ]:
        raise ValueError("Synthetic v11 severe-harm estimand set is not exact.")
    if (
        waveform.get("filterbank_config")
        != "configs/baselines/fbcca_chen2015_m3_v2_explicit_weights.yaml"
        or waveform.get("filterbank_config_sha256") != EXPECTED_FILTERBANK_SHA256
        or not np.array_equal(
            np.asarray(waveform.get("exact_subband_weights"), dtype=np.float64),
            P2_SUBBAND_WEIGHTS,
        )
    ):
        raise ValueError("Synthetic waveform FBCCA binding drifted.")
    filterbank_path = _regular_nonsymlink_file(
        _REPOSITORY / str(waveform["filterbank_config"]), "FBCCA config"
    )
    filterbank_sha = _sha256_file(filterbank_path)
    if filterbank_sha != EXPECTED_FILTERBANK_SHA256:
        raise ValueError("Frozen strict-FBCCA config byte hash drifted.")
    filterbank = yaml.safe_load(
        _read_bytes_nofollow(filterbank_path, "frozen filterbank config").decode("utf-8")
    )
    if not isinstance(filterbank, Mapping):
        raise TypeError("FBCCA config must decode to one mapping.")
    resolved = resolve_filterbank_parameters(filterbank, sfreq=250.0)
    if (
        len(resolved["bands_hz"]) != 7
        or resolved["n_harmonics"] != 5
        or not np.array_equal(np.asarray(resolved["weights"]), P2_SUBBAND_WEIGHTS)
    ):
        raise ValueError("Resolved FBCCA parameters are not the frozen seven-band contract.")
    return SyntheticContract(
        plan_path=path.resolve(),
        plan_sha256=observed_sha,
        filterbank_path=filterbank_path.resolve(),
        filterbank_sha256=filterbank_sha,
        diagnostic_operator_configs_sha256=_canonical_json_sha256(dict(diagnostics)),
        plan=plan,
        filterbank=dict(filterbank),
    )


def keyed_rng(
    contract: SyntheticContract,
    *,
    root_seed: int,
    family: str,
    participant_index: int,
    block: int,
    class_index: int,
    component: str,
    _governed_context: _GovernedSeedContext | None = None,
) -> np.random.Generator:
    """Construct one loop-order-independent PCG64DXSM stream from all six keys."""

    _require_seed_access(contract, root_seed, _governed_context)
    if family not in FAMILY_NAMES:
        raise ValueError(f"Unknown synthetic family {family!r}.")
    values = (root_seed, participant_index, block, class_index)
    if any(type(value) is not int or value < 0 for value in values):
        raise ValueError("RNG integer keys must be non-negative exact integers.")
    rng_plan = contract.plan["rng"]
    component_codes = rng_plan["component_codes"]
    if component not in component_codes:
        raise ValueError(f"Unknown synthetic random component {component!r}.")
    entropy = [
        root_seed,
        int(rng_plan["family_codes"][family]),
        participant_index,
        block,
        class_index,
        int(component_codes[component]),
    ]
    return np.random.Generator(np.random.PCG64DXSM(np.random.SeedSequence(entropy)))


def _oscillation(
    frequency_hz: float,
    phase: float,
    drift: float,
    timeline: np.ndarray,
) -> np.ndarray:
    angle = 2.0 * np.pi * frequency_hz * timeline + phase + drift
    return np.sin(angle) + 0.35 * np.sin(2.0 * angle)


def _stationary_ar1_noise(
    standard_normal: np.ndarray,
    *,
    innovation_sd: float,
    rho: float,
) -> np.ndarray:
    innovations = np.asarray(standard_normal, dtype=np.float64) * innovation_sd
    noise = np.empty_like(innovations)
    noise[:, 0] = innovations[:, 0] / math.sqrt(1.0 - rho**2)
    for sample in range(1, innovations.shape[1]):
        noise[:, sample] = rho * noise[:, sample - 1] + innovations[:, sample]
    return noise


def _regime_for_state(
    state: int,
    *,
    class_index: int,
    participant_index: int,
    n_classes: int,
) -> tuple[int, float, float]:
    offset = participant_index % 3
    if state == -1:
        return (class_index + 1 + offset) % n_classes, 0.0, 0.85
    if state == 1:
        return (class_index + 4 + offset) % n_classes, 0.5 * np.pi, 1.30
    raise ValueError("Synthetic state must be -1 or +1.")


def generate_synthetic_participant(
    contract: SyntheticContract,
    *,
    root_seed: int,
    family: str,
    participant_index: int,
    _governed_context: _GovernedSeedContext | None = None,
) -> SyntheticParticipant:
    """Generate one participant; no family-sized EEG tensor is ever materialized."""

    _require_seed_access(contract, root_seed, _governed_context)
    if family not in FAMILY_NAMES:
        raise ValueError(f"Unknown synthetic family {family!r}.")
    if type(participant_index) is not int or participant_index < 0:
        raise ValueError("participant_index must be a non-negative exact integer.")
    population = contract.plan["population"]
    waveform = contract.plan["waveform"]
    n_classes = int(population["classes"])
    n_blocks = int(population["complete_blocks"])
    n_channels = int(waveform["channels"])
    sfreq = float(waveform["sampling_rate_hz"])
    n_samples = round(sfreq * float(waveform["duration_seconds"]))
    timeline = np.arange(n_samples, dtype=np.float64) / sfreq
    frequencies = 8.0 + 0.4 * np.arange(n_classes, dtype=np.float64)

    spatial = np.empty((n_classes, n_channels), dtype=np.float64)
    phases = np.empty(n_classes, dtype=np.float64)
    for class_index in range(n_classes):
        vector = keyed_rng(
            contract,
            root_seed=root_seed,
            family=family,
            participant_index=participant_index,
            block=0,
            class_index=class_index,
            component="spatial_signature",
            _governed_context=_governed_context,
        ).standard_normal(n_channels)
        norm = float(np.linalg.norm(vector))
        if norm == 0.0:  # practically impossible, but deterministic failure is preferable.
            raise RuntimeError("A synthetic spatial signature had zero norm.")
        spatial[class_index] = vector / norm
        phases[class_index] = keyed_rng(
            contract,
            root_seed=root_seed,
            family=family,
            participant_index=participant_index,
            block=0,
            class_index=class_index,
            component="class_phase",
            _governed_context=_governed_context,
        ).uniform(-np.pi, np.pi)

    signals = np.empty((n_blocks, n_classes, n_channels, n_samples), dtype=np.float64)
    true_labels = np.broadcast_to(np.arange(n_classes), (n_blocks, n_classes)).copy()
    recorded_labels = true_labels.copy()
    interfaces = np.full((n_blocks, n_classes), None, dtype=object)
    impedance = np.full((n_blocks, n_classes, n_channels), np.nan, dtype=np.float64)
    signal_states = np.zeros(n_blocks, dtype=np.int8)
    context_states = np.zeros(n_blocks, dtype=np.int8)
    generated_target_classes = np.empty((n_blocks, n_classes), dtype=np.int64)
    generated_cross_classes = np.full((n_blocks, n_classes), -1, dtype=np.int64)
    innovation_sds = np.empty((n_blocks, n_classes), dtype=np.float64)
    cross_extra_phases = np.zeros((n_blocks, n_classes), dtype=np.float64)
    interface_value = f"synthetic-interface-{participant_index % 3}"

    for block_index in range(n_blocks):
        block = block_index + 1
        drift = keyed_rng(
            contract,
            root_seed=root_seed,
            family=family,
            participant_index=participant_index,
            block=block,
            class_index=0,
            component="block_phase_drift",
            _governed_context=_governed_context,
        ).normal(0.0, 0.08)
        gains = keyed_rng(
            contract,
            root_seed=root_seed,
            family=family,
            participant_index=participant_index,
            block=block,
            class_index=0,
            component="channel_gain",
            _governed_context=_governed_context,
        ).lognormal(mean=0.0, sigma=0.12, size=n_channels)

        if family == "B3_context_dependent_support_query_shift":
            signal_state = context_state = -1 if block % 2 else 1
        elif family == "N4_context_independent_of_signal":
            signal_draw = keyed_rng(
                contract,
                root_seed=root_seed,
                family=family,
                participant_index=participant_index,
                block=block,
                class_index=0,
                component="label_permutation",
                _governed_context=_governed_context,
            ).integers(0, 2)
            context_draw = keyed_rng(
                contract,
                root_seed=root_seed,
                family=family,
                participant_index=participant_index,
                block=block,
                class_index=0,
                component="metadata_shuffle",
                _governed_context=_governed_context,
            ).integers(0, 2)
            signal_state = -1 if signal_draw == 0 else 1
            context_state = -1 if context_draw == 0 else 1
        else:
            signal_state = context_state = 0
        signal_states[block_index] = signal_state
        context_states[block_index] = context_state

        if family in {
            "B3_context_dependent_support_query_shift",
            "N4_context_independent_of_signal",
        }:
            interfaces[block_index, :] = interface_value
            metadata_noise = keyed_rng(
                contract,
                root_seed=root_seed,
                family=family,
                participant_index=participant_index,
                block=block,
                class_index=0,
                component="metadata_noise",
                _governed_context=_governed_context,
            ).normal(0.0, 0.05, size=n_channels)
            impedance_packet = np.exp(np.log(6.0) + 0.65 * context_state + metadata_noise) - 1.0
            impedance[block_index, :, :] = impedance_packet[None, :]

        if family == "N2_random_fit_support_labels" and block <= 5:
            shift = int(
                keyed_rng(
                    contract,
                    root_seed=root_seed,
                    family=family,
                    participant_index=participant_index,
                    block=block,
                    class_index=0,
                    component="label_permutation",
                    _governed_context=_governed_context,
                ).integers(1, n_classes)
            )
            recorded_labels[block_index] = (true_labels[block_index] + shift) % n_classes

        for class_index in range(n_classes):
            target_class = class_index
            target_amplitude = 1.0
            cross_class: int | None = None
            cross_amplitude = 0.0
            cross_extra_phase = 0.0
            innovation_sd = 0.85
            if family in {
                "B1_participant_specific_class_confusion",
                "N2_random_fit_support_labels",
            }:
                cross_class = (class_index + 1 + participant_index % 3) % n_classes
                cross_amplitude = 0.90
            elif family == "B2_participant_phase_spatial_shift":
                target_amplitude = 0.65
                innovation_sd = 1.10
            elif family in {
                "B3_context_dependent_support_query_shift",
                "N4_context_independent_of_signal",
            }:
                cross_class, cross_extra_phase, innovation_sd = _regime_for_state(
                    int(signal_state),
                    class_index=class_index,
                    participant_index=participant_index,
                    n_classes=n_classes,
                )
                cross_amplitude = 0.75
            elif family == "N1_perfect_clean_anchor":
                target_amplitude = 2.0
                innovation_sd = 0.05
            elif family == "N3_nonstationary_corrupted_calibration":
                if block <= 5:
                    target_class = (class_index + block) % n_classes
                innovation_sd = 0.85

            generated_target_classes[block_index, class_index] = target_class
            generated_cross_classes[block_index, class_index] = (
                -1 if cross_class is None else cross_class
            )
            innovation_sds[block_index, class_index] = innovation_sd
            cross_extra_phases[block_index, class_index] = cross_extra_phase

            waveform_value = (
                target_amplitude
                * spatial[class_index, :, None]
                * _oscillation(frequencies[target_class], phases[target_class], drift, timeline)[
                    None, :
                ]
            )
            if cross_class is not None:
                waveform_value += (
                    cross_amplitude
                    * spatial[class_index, :, None]
                    * _oscillation(
                        frequencies[cross_class],
                        phases[cross_class] + cross_extra_phase,
                        drift,
                        timeline,
                    )[None, :]
                )
            standard_normal = keyed_rng(
                contract,
                root_seed=root_seed,
                family=family,
                participant_index=participant_index,
                block=block,
                class_index=class_index,
                component="innovation_noise",
                _governed_context=_governed_context,
            ).standard_normal((n_channels, n_samples))
            noise = _stationary_ar1_noise(
                standard_normal,
                innovation_sd=innovation_sd,
                rho=0.55,
            )
            signals[block_index, class_index] = gains[:, None] * waveform_value + noise

    partition_sha256s: list[str] = []
    for block_index in range(n_blocks):
        partition_sha256s.append(
            _canonical_json_sha256(
                {
                    "schema": "cfeg.metadata-calibration-v2-synthetic-partition.v1",
                    "plan_sha256": contract.plan_sha256,
                    "root_seed": root_seed,
                    "family": family,
                    "participant_index": participant_index,
                    "block": block_index + 1,
                    "signals_sha256": _array_sha256(signals[block_index]),
                    "recorded_support_labels_sha256": _array_sha256(recorded_labels[block_index]),
                    "interfaces_sha256": _array_sha256(interfaces[block_index]),
                    "impedance_sha256": _array_sha256(impedance[block_index]),
                    "generated_target_classes_sha256": _array_sha256(
                        generated_target_classes[block_index]
                    ),
                    "generated_cross_classes_sha256": _array_sha256(
                        generated_cross_classes[block_index]
                    ),
                    "innovation_sds_sha256": _array_sha256(innovation_sds[block_index]),
                    "cross_extra_phases_sha256": _array_sha256(cross_extra_phases[block_index]),
                }
            )
        )
    for array in (
        signals,
        true_labels,
        recorded_labels,
        interfaces,
        impedance,
        signal_states,
        context_states,
        generated_target_classes,
        generated_cross_classes,
        innovation_sds,
        cross_extra_phases,
    ):
        array.setflags(write=False)
    return SyntheticParticipant(
        family=family,
        participant_index=participant_index,
        root_seed=root_seed,
        signals=signals,
        true_labels=true_labels,
        recorded_support_labels=recorded_labels,
        interfaces=interfaces,
        impedance_kohm=impedance,
        signal_states=signal_states,
        context_states=context_states,
        generated_target_classes=generated_target_classes,
        generated_cross_classes=generated_cross_classes,
        innovation_sds=innovation_sds,
        cross_extra_phases=cross_extra_phases,
        partition_sha256s=tuple(partition_sha256s),
    )


def produce_strict_fbcca(
    contract: SyntheticContract,
    signals: np.ndarray,
) -> StrictFBCCAProduct:
    """Produce label-free strict FBCCA scores and the exact P2 subbands."""

    trials = np.asarray(signals, dtype=np.float64)
    if trials.ndim != 4 or trials.shape[2:] != (8, 500):
        raise ValueError("Synthetic signals must have [block,class,8,500] shape.")
    if not np.isfinite(trials).all():
        raise ValueError("Synthetic FBCCA producer accepts only finite EEG.")
    n_blocks, n_classes = trials.shape[:2]
    if n_classes != 12:
        raise ValueError("Synthetic strict FBCCA requires the frozen 12-class codebook.")
    flat = trials.reshape(n_blocks * n_classes, 8, 500)
    subbands, parameters = apply_filterbank(flat, sfreq=250.0, filterbank=contract.filterbank)
    weights = np.asarray(parameters["weights"], dtype=np.float64)
    if subbands.shape != (7, len(flat), 8, 500) or not np.array_equal(weights, P2_SUBBAND_WEIGHTS):
        raise RuntimeError("Strict FBCCA did not produce the exact seven frozen subbands.")
    frequencies = 8.0 + 0.4 * np.arange(n_classes, dtype=np.float64)
    references = make_reference_signals(frequencies, 250.0, 500, n_harmonics=5)
    scores = np.zeros((len(flat), n_classes), dtype=np.float64)
    for band_index, weight in enumerate(weights):
        for trial_index in range(len(flat)):
            correlations = np.asarray(
                [
                    cca_score(
                        subbands[band_index, trial_index],
                        reference,
                        regularization=1.0e-8,
                    )
                    for reference in references
                ],
                dtype=np.float64,
            )
            scores[trial_index] += weight * correlations**2
    scores = scores.reshape(n_blocks, n_classes, n_classes)
    subbands = subbands.reshape(7, n_blocks, n_classes, 8, 500)
    if not np.isfinite(scores).all() or not np.isfinite(subbands).all():
        raise RuntimeError("Strict FBCCA product contains non-finite values.")
    producer_sha = _canonical_json_sha256(
        {
            "schema": "cfeg.metadata-calibration-v2-strict-fbcca-product.v1",
            "plan_sha256": contract.plan_sha256,
            "filterbank_sha256": contract.filterbank_sha256,
            "signals_sha256": _array_sha256(trials),
            "scores_sha256": _array_sha256(scores),
            "subbands_sha256": _array_sha256(subbands),
            "weights": weights.tolist(),
        }
    )
    for array in (scores, subbands, weights):
        array.setflags(write=False)
    return StrictFBCCAProduct(
        scores=scores,
        subbands=subbands,
        subband_weights=weights,
        producer_sha256=producer_sha,
    )


def write_strict_fbcca_cache_exclusive(
    path: str | Path,
    product: StrictFBCCAProduct,
    *,
    contract: SyntheticContract,
    participant: SyntheticParticipant,
    _governed_context: _GovernedSeedContext | None = None,
) -> Path:
    """Write one participant-only cache without permitting overwrite or pickle payloads."""

    _require_seed_access(contract, participant.root_seed, _governed_context)
    target = _validated_unused_output_path(path)
    manifest = {
        "schema": SYNTHETIC_FBCCA_CACHE_SCHEMA,
        "plan_sha256": contract.plan_sha256,
        "filterbank_sha256": contract.filterbank_sha256,
        "root_seed": participant.root_seed,
        "family": participant.family,
        "participant_index": participant.participant_index,
        "signals_sha256": _array_sha256(participant.signals),
        "scores_sha256": _array_sha256(product.scores),
        "subbands_sha256": _array_sha256(product.subbands),
        "subband_weights_sha256": _array_sha256(product.subband_weights),
        "producer_sha256": product.producer_sha256,
    }
    manifest["cache_manifest_sha256"] = _canonical_json_sha256(manifest)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(target, flags, stat.S_IRUSR | stat.S_IWUSR)
    with os.fdopen(descriptor, "wb", closefd=True) as handle:
        np.savez_compressed(
            handle,
            manifest=np.frombuffer(_canonical_json_bytes(manifest), dtype=np.uint8),
            scores=product.scores,
            subbands=product.subbands,
            subband_weights=product.subband_weights,
        )
        handle.flush()
        os.fsync(handle.fileno())
    return target


def load_strict_fbcca_cache(
    path: str | Path,
    *,
    contract: SyntheticContract,
    participant: SyntheticParticipant,
    _governed_context: _GovernedSeedContext | None = None,
) -> StrictFBCCAProduct:
    _require_seed_access(contract, participant.root_seed, _governed_context)
    source = _regular_nonsymlink_file(path, "synthetic FBCCA cache")
    with np.load(source, allow_pickle=False) as archive:
        if set(archive.files) != {"manifest", "scores", "subbands", "subband_weights"}:
            raise ValueError("Synthetic FBCCA cache members are not exact.")
        manifest_bytes = np.asarray(archive["manifest"], dtype=np.uint8).tobytes()
        decoded = json.loads(manifest_bytes.decode("utf-8"))
        scores = np.asarray(archive["scores"], dtype=np.float64)
        subbands = np.asarray(archive["subbands"], dtype=np.float64)
        weights = np.asarray(archive["subband_weights"], dtype=np.float64)
    if not isinstance(decoded, Mapping):
        raise TypeError("Synthetic FBCCA cache manifest must be a mapping.")
    manifest = dict(decoded)
    claimed_hash = manifest.pop("cache_manifest_sha256", None)
    if claimed_hash != _canonical_json_sha256(manifest):
        raise ValueError("Synthetic FBCCA cache manifest self-hash mismatch.")
    expected = {
        "schema": SYNTHETIC_FBCCA_CACHE_SCHEMA,
        "plan_sha256": contract.plan_sha256,
        "filterbank_sha256": contract.filterbank_sha256,
        "root_seed": participant.root_seed,
        "family": participant.family,
        "participant_index": participant.participant_index,
        "signals_sha256": _array_sha256(participant.signals),
        "scores_sha256": _array_sha256(scores),
        "subbands_sha256": _array_sha256(subbands),
        "subband_weights_sha256": _array_sha256(weights),
        "producer_sha256": manifest.get("producer_sha256"),
    }
    if manifest != expected:
        raise ValueError("Synthetic FBCCA cache provenance or array hash drifted.")
    if (
        scores.shape != (10, 12, 12)
        or subbands.shape != (7, 10, 12, 8, 500)
        or not np.array_equal(weights, P2_SUBBAND_WEIGHTS)
        or not np.isfinite(scores).all()
        or not np.isfinite(subbands).all()
    ):
        raise ValueError("Synthetic FBCCA cache shape, weights, or finiteness is invalid.")
    for array in (scores, subbands, weights):
        array.setflags(write=False)
    return StrictFBCCAProduct(
        scores=scores,
        subbands=subbands,
        subband_weights=weights,
        producer_sha256=str(manifest["producer_sha256"]),
    )


def _flatten_blocks(value: np.ndarray, blocks: Sequence[int]) -> np.ndarray:
    indices = np.asarray([int(block) - 1 for block in blocks], dtype=np.int64)
    if np.any(indices < 0) or np.any(indices >= value.shape[0]):
        raise ValueError("Requested block is outside the synthetic participant.")
    selected = value[indices]
    return selected.reshape((-1, *value.shape[2:]))


def _flatten_subband_blocks(value: np.ndarray, blocks: Sequence[int]) -> np.ndarray:
    indices = np.asarray([int(block) - 1 for block in blocks], dtype=np.int64)
    if np.any(indices < 0) or np.any(indices >= value.shape[1]):
        raise ValueError("Requested subband block is outside the synthetic participant.")
    selected = value[:, indices]
    return selected.reshape((value.shape[0], -1, *value.shape[3:]))


def _metadata_source_blocks(
    participant: SyntheticParticipant,
    support_blocks: Sequence[int],
    control: str,
    contract: SyntheticContract,
    governed_context: _GovernedSeedContext | None = None,
) -> tuple[int, ...]:
    _require_seed_access(contract, participant.root_seed, governed_context)
    support = tuple(int(block) for block in support_blocks)
    if not support or support != tuple(range(1, len(support) + 1)) or len(support) > 5:
        raise ValueError(
            "Support metadata controls require the exact current prefix in blocks 1..5."
        )
    if control == "correct" or control == "opposite_interface" or control == "all_missing":
        sources = support
    if control == "stale":
        sources = support[-1:] + support[:-1]
    elif control == "pair_shuffled":
        participant_global_order = keyed_rng(
            contract,
            root_seed=participant.root_seed,
            family=participant.family,
            participant_index=participant.participant_index,
            block=0,
            class_index=0,
            component="metadata_shuffle",
            _governed_context=governed_context,
        ).permutation(np.arange(1, 6, dtype=np.int64))
        sources = tuple(int(block) for block in participant_global_order if int(block) in support)
    elif control not in {"correct", "opposite_interface", "all_missing"}:
        raise ValueError(f"Unknown metadata pairing control {control!r}.")
    if len(sources) != len(support) or set(sources) != set(support):
        raise RuntimeError("Every metadata donor must be exactly one member of the support prefix.")
    return sources


def _context_arguments(
    participant: SyntheticParticipant,
    contract: SyntheticContract,
    *,
    query_blocks: Sequence[int],
    support_blocks: Sequence[int],
    control: str,
    governed_context: _GovernedSeedContext | None = None,
) -> tuple[dict[str, Any], str]:
    if control not in CONTEXT_CONTROLS:
        raise ValueError(f"Unknown metadata pairing control {control!r}.")
    source_blocks = _metadata_source_blocks(
        participant,
        support_blocks,
        control,
        contract,
        governed_context,
    )
    if set(source_blocks) != {int(value) for value in support_blocks} or set(source_blocks) & {
        int(value) for value in query_blocks
    }:
        raise RuntimeError("Context pairing referenced evaluation, query, or future metadata.")
    pairing = {
        "schema": "cfeg.metadata-calibration-v2-synthetic-context-pairing.v1",
        "plan_sha256": contract.plan_sha256,
        "root_seed": participant.root_seed,
        "family": participant.family,
        "participant_index": participant.participant_index,
        "control": control,
        "query_blocks": [int(value) for value in query_blocks],
        "support_blocks": [int(value) for value in support_blocks],
        "support_metadata_source_blocks": list(source_blocks),
    }
    pairing_sha = _canonical_json_sha256(pairing)
    if control == "all_missing":
        return {}, pairing_sha
    query_interfaces = _flatten_blocks(participant.interfaces, query_blocks).reshape(-1)
    query_impedance = _flatten_blocks(participant.impedance_kohm, query_blocks)
    support_interfaces = _flatten_blocks(participant.interfaces, source_blocks).reshape(-1)
    support_impedance = _flatten_blocks(participant.impedance_kohm, source_blocks)
    if control == "opposite_interface":
        support_interfaces = np.asarray(
            [f"opposite::{value}" for value in support_interfaces], dtype=object
        )
    return (
        {
            "query_interfaces": query_interfaces,
            "support_interfaces": support_interfaces,
            "query_impedance_kohm": query_impedance,
            "support_impedance_kohm": support_impedance,
        },
        pairing_sha,
    )


def _operator_config(contract: SyntheticContract, operator: str) -> v2_operator.V2OperatorConfig:
    diagnostic = contract.plan["diagnostic_operator_configs"]
    p1 = diagnostic["P1"]
    p2 = diagnostic["P2"]
    aqm = diagnostic["A_QM"]
    selected = p1 if operator == "score_prototype_shrinkage" else p2
    return v2_operator.V2OperatorConfig(
        ideal_prototype_smoothing=float(p1["ideal_prototype_smoothing"]),
        prototype_prior_pseudocount=float(p1["prototype_prior_pseudocount"]),
        lambda_max=float(selected["lambda_max"]),
        different_interface_affinity=float(aqm["different_interface_affinity"]),
        entropy_scaling=bool(selected["entropy_scaling"]),
    )


def _operator_payload(
    participant: SyntheticParticipant,
    product: StrictFBCCAProduct,
    contract: SyntheticContract,
    *,
    query_blocks: Sequence[int],
    support_blocks: Sequence[int],
    operator: str,
    variant: str,
    context_control: str,
    governed_context: _GovernedSeedContext | None = None,
) -> dict[str, Any]:
    """Build support-only capabilities. This function has no query-label argument."""

    support_scores = _flatten_blocks(product.scores, support_blocks)
    support_labels = _flatten_blocks(participant.recorded_support_labels, support_blocks).reshape(
        -1
    )
    payload: dict[str, Any] = {
        "support_fbcca_scores": support_scores,
        "support_labels": support_labels,
    }
    if operator == "filterbank_target_template_residual":
        payload.update(
            {
                "template_query_subbands": _flatten_subband_blocks(product.subbands, query_blocks),
                "template_support_subbands": _flatten_subband_blocks(
                    product.subbands, support_blocks
                ),
                "template_subband_weights": product.subband_weights,
            }
        )
    if variant == "A_QM":
        context, pairing_sha = _context_arguments(
            participant,
            contract,
            query_blocks=query_blocks,
            support_blocks=support_blocks,
            control=context_control,
            governed_context=governed_context,
        )
        payload.update(context)
        payload["relative_context_pairing_sha256"] = pairing_sha
    return payload


def _prequential_gate(
    participant: SyntheticParticipant,
    product: StrictFBCCAProduct,
    contract: SyntheticContract,
    *,
    budget: int,
    operator: str,
    variant: str,
    context_control: str,
    governed_context: _GovernedSeedContext | None = None,
) -> v2_operator.PrequentialGateDecision:
    _require_seed_access(contract, participant.root_seed, governed_context)
    if budget not in {3, 5}:
        raise ValueError("Synthetic prequential gate is defined only for k=3 or k=5.")
    base_rows: list[np.ndarray] = []
    candidate_rows: list[np.ndarray] = []
    evaluation_labels: list[np.ndarray] = []
    block_rows: list[np.ndarray] = []
    provenance_rows: list[v2_operator.PrequentialFoldProvenance] = []
    for evaluation_block in range(2, budget + 1):
        fit_blocks = tuple(range(1, evaluation_block))
        provenance = v2_operator.PrequentialFoldProvenance(
            evaluation_block=evaluation_block,
            fit_blocks=fit_blocks,
            fit_block_partition_sha256s=tuple(
                participant.partition_sha256s[index - 1] for index in fit_blocks
            ),
            evaluation_partition_sha256=participant.partition_sha256s[evaluation_block - 1],
        )
        evaluation_scores = _flatten_blocks(product.scores, (evaluation_block,))
        payload = _operator_payload(
            participant,
            product,
            contract,
            query_blocks=(evaluation_block,),
            support_blocks=fit_blocks,
            operator=operator,
            variant=variant,
            context_control=context_control,
            governed_context=governed_context,
        )
        output = v2_operator.apply_v2_prequential_operator(
            evaluation_scores,
            final_budget=budget,
            fold_provenance=provenance,
            operator=operator,
            variant=variant,
            config=_operator_config(contract, operator),
            **payload,
        )
        if output.prequential_fold_provenance is not provenance:
            raise RuntimeError("Prequential operator did not echo the exact typed provenance.")
        base_rows.append(output.base_probabilities)
        candidate_rows.append(output.fused_probabilities)
        # Labels are joined only after the operator has returned its probabilities.
        evaluation_labels.append(participant.true_labels[evaluation_block - 1].copy())
        block_rows.append(np.full(12, evaluation_block, dtype=np.int64))
        provenance_rows.append(provenance)
    return v2_operator.prequential_gate_decision(
        np.concatenate(base_rows),
        np.concatenate(candidate_rows),
        np.concatenate(evaluation_labels),
        np.concatenate(block_rows),
        budget=budget,
        fold_provenance=tuple(provenance_rows),
    )


def _apply_final_query_operator(
    participant: SyntheticParticipant,
    product: StrictFBCCAProduct,
    contract: SyntheticContract,
    *,
    budget: int,
    operator: str,
    variant: str,
    context_control: str = "correct",
    governed_context: _GovernedSeedContext | None = None,
) -> tuple[v2_operator.V2OperatorOutput, v2_operator.PrequentialGateDecision | None]:
    """Score immutable query blocks without exposing their labels to the operator."""

    _require_seed_access(contract, participant.root_seed, governed_context)
    query_blocks = tuple(
        int(value) for value in contract.plan["population"]["immutable_query_blocks"]
    )
    query_scores = _flatten_blocks(product.scores, query_blocks)
    config = _operator_config(contract, operator)
    if budget == 0:
        output = v2_operator.apply_v2_safe_operator(
            query_scores,
            budget=0,
            operator=operator,
            variant=variant,
            config=config,
        )
        return output, None
    if budget == 1:
        gate_enabled = True
        gate = None
    else:
        gate = _prequential_gate(
            participant,
            product,
            contract,
            budget=budget,
            operator=operator,
            variant=variant,
            context_control=context_control,
            governed_context=governed_context,
        )
        gate_enabled = gate.enabled
    if not gate_enabled:
        # Fail closed before constructing or passing final-query support/context capabilities.
        output = v2_operator.apply_v2_safe_operator(
            query_scores,
            budget=budget,
            operator=operator,
            variant=variant,
            config=config,
            gate_enabled=False,
        )
        return output, gate
    support_blocks = tuple(
        int(value) for value in contract.plan["population"]["nested_support"][budget]
    )
    payload = _operator_payload(
        participant,
        product,
        contract,
        query_blocks=query_blocks,
        support_blocks=support_blocks,
        operator=operator,
        variant=variant,
        context_control=context_control,
        governed_context=governed_context,
    )
    output = v2_operator.apply_v2_safe_operator(
        query_scores,
        budget=budget,
        operator=operator,
        variant=variant,
        config=config,
        gate_enabled=True,
        **payload,
    )
    return output, gate


def _balanced_accuracy(predictions: np.ndarray, labels: np.ndarray, n_classes: int = 12) -> float:
    predicted = np.asarray(predictions)
    observed = np.asarray(labels)
    if predicted.shape != observed.shape or predicted.ndim != 1:
        raise ValueError("Predictions and labels must be aligned one-dimensional arrays.")
    recalls = []
    for class_index in range(n_classes):
        rows = observed == class_index
        if not rows.any():
            raise ValueError("Balanced accuracy requires every frozen class.")
        recalls.append(float(np.mean(predicted[rows] == observed[rows])))
    return float(np.mean(recalls))


def _method_specs_for_family(family: str) -> tuple[tuple[str, str, str, str], ...]:
    p1 = "score_prototype_shrinkage"
    p2 = "filterbank_target_template_residual"
    if family == "B1_participant_specific_class_confusion":
        return (("P1_A_Q", p1, "A_Q", "correct"),)
    if family == "B2_participant_phase_spatial_shift":
        return (("P2_A_Q", p2, "A_Q", "correct"),)
    if family == "B3_context_dependent_support_query_shift":
        return (
            ("P1_A_Q", p1, "A_Q", "correct"),
            *((f"P1_A_QM_{control}", p1, "A_QM", control) for control in CONTEXT_CONTROLS),
        )
    if family == "N1_perfect_clean_anchor":
        return (
            ("P1_A_Q", p1, "A_Q", "correct"),
            ("P2_A_Q", p2, "A_Q", "correct"),
        )
    if family in {
        "N2_random_fit_support_labels",
        "N3_nonstationary_corrupted_calibration",
    }:
        return (("P1_A_Q", p1, "A_Q", "correct"),)
    if family == "N4_context_independent_of_signal":
        return (
            ("P1_A_Q", p1, "A_Q", "correct"),
            ("P1_A_QM_correct", p1, "A_QM", "correct"),
            ("P1_A_QM_pair_shuffled", p1, "A_QM", "pair_shuffled"),
            ("P1_A_QM_all_missing", p1, "A_QM", "all_missing"),
        )
    raise ValueError(f"Unknown synthetic family {family!r}.")


def evaluate_synthetic_participant(
    participant: SyntheticParticipant,
    product: StrictFBCCAProduct,
    contract: SyntheticContract,
    *,
    _governed_context: _GovernedSeedContext | None = None,
) -> list[dict[str, Any]]:
    """Return scalar participant metrics after label-free operator predictions exist."""

    _require_seed_access(contract, participant.root_seed, _governed_context)
    query_blocks = tuple(contract.plan["population"]["immutable_query_blocks"])
    query_labels = _flatten_blocks(participant.true_labels, query_blocks).reshape(-1)
    query_scores = _flatten_blocks(product.scores, query_blocks)
    _, base_probabilities = v2_operator.normalize_fbcca_scores(query_scores)
    base_predictions = np.argmax(base_probabilities, axis=1)
    base_ba = _balanced_accuracy(base_predictions, query_labels)
    rows: list[dict[str, Any]] = []
    for budget in SUPPORT_BUDGETS:
        rows.append(
            {
                "family": participant.family,
                "participant_index": participant.participant_index,
                "method": "A0",
                "budget": budget,
                "balanced_accuracy": base_ba,
                "gate_enabled": None,
                "gate_reason": "strict_fbcca_anchor",
                "exact_fallback": True,
                "prediction_sha256": _array_sha256(base_predictions),
                "producer_sha256": product.producer_sha256,
                "operator_schema": None,
                "support_depth": 0,
                "relative_context_pairing_sha256": None,
                "comparable_context_pair_count": 0,
                "prequential": None,
            }
        )
    for method, operator, variant, control in _method_specs_for_family(participant.family):
        for budget in SUPPORT_BUDGETS:
            output, gate = _apply_final_query_operator(
                participant,
                product,
                contract,
                budget=budget,
                operator=operator,
                variant=variant,
                context_control=control,
                governed_context=_governed_context,
            )
            # This is the first point at which immutable query labels meet predictions.
            ba = _balanced_accuracy(output.predictions, query_labels)
            rows.append(
                {
                    "family": participant.family,
                    "participant_index": participant.participant_index,
                    "method": method,
                    "budget": budget,
                    "balanced_accuracy": ba,
                    "gate_enabled": None if gate is None else gate.enabled,
                    "gate_reason": (
                        "synthetic_global_k1_enabled"
                        if budget == 1
                        else output.fallback_reason
                        if gate is None
                        else gate.reason
                    ),
                    "exact_fallback": output.exact_fallback,
                    "prediction_sha256": _array_sha256(output.predictions),
                    "producer_sha256": product.producer_sha256,
                    "operator_schema": output.schema,
                    "support_depth": output.support_depth,
                    "relative_context_pairing_sha256": output.relative_context_pairing_sha256,
                    "comparable_context_pair_count": output.comparable_context_pair_count,
                    "prequential": (
                        None
                        if gate is None
                        else {
                            "evaluated_blocks": list(gate.evaluated_blocks),
                            "block_balanced_accuracy_deltas": list(
                                gate.block_balanced_accuracy_deltas
                            ),
                            "block_log_probability_deltas": list(gate.block_log_probability_deltas),
                            "mean_balanced_accuracy_delta": gate.mean_balanced_accuracy_delta,
                            "mean_log_probability_delta": gate.mean_log_probability_delta,
                            "fold_provenance": [
                                {
                                    "schema": item.schema,
                                    "evaluation_block": item.evaluation_block,
                                    "fit_blocks": list(item.fit_blocks),
                                    "fit_block_partition_sha256s": list(
                                        item.fit_block_partition_sha256s
                                    ),
                                    "fit_partition_sha256": item.fit_partition_sha256,
                                    "evaluation_partition_sha256": (
                                        item.evaluation_partition_sha256
                                    ),
                                }
                                for item in gate.fold_provenance
                            ],
                        }
                    ),
                }
            )
    return rows


def validate_hard_assertions(
    participant: SyntheticParticipant,
    product: StrictFBCCAProduct,
    contract: SyntheticContract,
    *,
    _governed_context: _GovernedSeedContext | None = None,
) -> dict[str, bool]:
    """Execute every frozen hard invariant before any efficacy metric is recorded."""

    _require_seed_access(contract, participant.root_seed, _governed_context)
    query_block = (6,)
    support_block = (1,)
    query_scores = _flatten_blocks(product.scores, query_block)
    support_scores = _flatten_blocks(product.scores, support_block)
    support_labels = _flatten_blocks(participant.recorded_support_labels, support_block).reshape(-1)
    p1_config = _operator_config(contract, "score_prototype_shrinkage")
    p2_config = _operator_config(contract, "filterbank_target_template_residual")
    _, expected_base = v2_operator.normalize_fbcca_scores(query_scores)

    k0 = v2_operator.apply_v2_safe_operator(
        query_scores,
        budget=0,
        operator="score_prototype_shrinkage",
        variant="A_QM",
        config=p1_config,
        support_fbcca_scores=object(),  # type: ignore[arg-type]
        support_labels=object(),  # type: ignore[arg-type]
        query_interfaces=object(),  # type: ignore[arg-type]
        relative_context_pairing_sha256="deliberately-not-a-digest",
    )
    if not (
        k0.exact_fallback
        and np.array_equal(k0.fused_probabilities, expected_base)
        and k0.relative_context_pairing_sha256 is None
    ):
        raise RuntimeError("k=0 was not a bitwise exact strict-FBCCA fallback.")

    missing_gate = v2_operator.apply_v2_safe_operator(
        query_scores,
        budget=1,
        operator="score_prototype_shrinkage",
        variant="A_QM",
        config=p1_config,
        support_fbcca_scores=object(),  # type: ignore[arg-type]
        support_labels=object(),  # type: ignore[arg-type]
        query_interfaces=object(),  # type: ignore[arg-type]
        relative_context_pairing_sha256="deliberately-not-a-digest",
    )
    false_gate = v2_operator.apply_v2_safe_operator(
        query_scores,
        budget=1,
        operator="score_prototype_shrinkage",
        variant="A_QM",
        config=p1_config,
        gate_enabled=False,
        support_fbcca_scores=object(),  # type: ignore[arg-type]
        support_labels=object(),  # type: ignore[arg-type]
        query_interfaces=object(),  # type: ignore[arg-type]
        relative_context_pairing_sha256="deliberately-not-a-digest",
    )
    if not all(
        output.exact_fallback
        and np.array_equal(output.fused_probabilities, expected_base)
        and output.relative_context_pairing_sha256 is None
        for output in (missing_gate, false_gate)
    ):
        raise RuntimeError("Missing/false gate did not fail closed before support access.")

    aq = v2_operator.apply_v2_safe_operator(
        query_scores,
        budget=1,
        operator="score_prototype_shrinkage",
        variant="A_Q",
        config=p1_config,
        gate_enabled=True,
        support_fbcca_scores=support_scores,
        support_labels=support_labels,
    )
    missing_pairing_sha = _canonical_json_sha256(
        {"schema": "cfeg.synthetic-all-missing-probe.v1", "plan_sha256": contract.plan_sha256}
    )
    aqm_missing = v2_operator.apply_v2_safe_operator(
        query_scores,
        budget=1,
        operator="score_prototype_shrinkage",
        variant="A_QM",
        config=p1_config,
        gate_enabled=True,
        support_fbcca_scores=support_scores,
        support_labels=support_labels,
        relative_context_pairing_sha256=missing_pairing_sha,
    )
    for field in (
        "base_probabilities",
        "support_probabilities",
        "fused_probabilities",
        "predictions",
        "lambdas",
        "affinities",
    ):
        if not np.array_equal(getattr(aq, field), getattr(aqm_missing, field)):
            raise RuntimeError("All-missing A_QM was not bitwise identical to A_Q.")

    query_order = np.arange(len(query_scores) - 1, -1, -1)
    support_order = np.roll(np.arange(len(support_scores)), 5)
    reordered = v2_operator.apply_v2_safe_operator(
        query_scores[query_order],
        budget=1,
        operator="score_prototype_shrinkage",
        variant="A_Q",
        config=p1_config,
        gate_enabled=True,
        support_fbcca_scores=support_scores[support_order],
        support_labels=support_labels[support_order],
    )
    if not (
        np.array_equal(reordered.fused_probabilities, aq.fused_probabilities[query_order])
        and np.array_equal(reordered.predictions, aq.predictions[query_order])
    ):
        raise RuntimeError("Query/support row reorder invariance failed.")

    permutation = np.asarray([4, 7, 1, 10, 0, 9, 3, 11, 2, 6, 8, 5], dtype=np.int64)
    inverse = np.argsort(permutation)
    class_permuted = v2_operator.apply_v2_safe_operator(
        query_scores[:, permutation],
        budget=1,
        operator="score_prototype_shrinkage",
        variant="A_Q",
        config=p1_config,
        gate_enabled=True,
        support_fbcca_scores=support_scores[:, permutation],
        support_labels=inverse[support_labels],
    )
    if not np.allclose(
        class_permuted.fused_probabilities,
        aq.fused_probabilities[:, permutation],
        rtol=0.0,
        atol=1.0e-15,
    ):
        raise RuntimeError("Class/codebook equivariance failed.")

    displacement = np.abs(aq.fused_probabilities - aq.base_probabilities).sum(axis=1)
    if not (
        np.isfinite(aq.fused_probabilities).all()
        and np.allclose(aq.fused_probabilities.sum(axis=1), 1.0, rtol=0.0, atol=1.0e-12)
        and np.all(displacement <= 2.0 * aq.lambdas + 1.0e-12)
    ):
        raise RuntimeError("Probability simplex or two-lambda displacement invariant failed.")

    query_subbands = _flatten_subband_blocks(product.subbands, query_block)
    support_subbands = _flatten_subband_blocks(product.subbands, support_block)
    v2_operator.apply_v2_safe_operator(
        query_scores,
        budget=1,
        operator="filterbank_target_template_residual",
        variant="A_Q",
        config=p2_config,
        gate_enabled=True,
        support_fbcca_scores=support_scores,
        support_labels=support_labels,
        template_query_subbands=query_subbands,
        template_support_subbands=support_subbands,
        template_subband_weights=product.subband_weights,
    )
    wrong_weights_rejected = False
    try:
        v2_operator.apply_v2_safe_operator(
            query_scores,
            budget=1,
            operator="filterbank_target_template_residual",
            variant="A_Q",
            config=p2_config,
            gate_enabled=True,
            support_fbcca_scores=support_scores,
            support_labels=support_labels,
            template_query_subbands=query_subbands,
            template_support_subbands=support_subbands,
            template_subband_weights=np.ones(7),
        )
    except ValueError:
        wrong_weights_rejected = True
    if not wrong_weights_rejected or not np.array_equal(
        product.subband_weights, P2_SUBBAND_WEIGHTS
    ):
        raise RuntimeError("P2 did not enforce the exact seven frozen weights.")

    template_scores = v2_operator.template_residual_scores_from_subbands(
        query_subbands,
        support_subbands,
        support_labels,
        product.subband_weights,
    )
    template_provenance = v2_operator.TemplateScoreProvenance(
        filterbank_sha256=contract.filterbank_sha256,
        preprocessing_sha256=product.producer_sha256,
        query_partition_sha256=participant.partition_sha256s[5],
        support_partition_sha256=participant.partition_sha256s[0],
    )
    template_missing_rejected = False
    try:
        v2_operator.apply_v2_safe_operator(
            query_scores,
            budget=1,
            operator="filterbank_target_template_residual",
            variant="A_Q",
            config=p2_config,
            gate_enabled=True,
            support_fbcca_scores=support_scores,
            support_labels=support_labels,
            template_query_class_scores=template_scores,
        )
    except ValueError:
        template_missing_rejected = True
    template_echo = v2_operator.apply_v2_safe_operator(
        query_scores,
        budget=1,
        operator="filterbank_target_template_residual",
        variant="A_Q",
        config=p2_config,
        gate_enabled=True,
        support_fbcca_scores=support_scores,
        support_labels=support_labels,
        template_query_class_scores=template_scores,
        template_score_provenance=template_provenance,
    )
    if not template_missing_rejected or template_echo.template_score_provenance is not (
        template_provenance
    ):
        raise RuntimeError("P2 typed template provenance rejection/echo failed.")

    context, pairing_sha = _context_arguments(
        participant,
        contract,
        query_blocks=query_block,
        support_blocks=support_block,
        control="all_missing",
        governed_context=_governed_context,
    )
    pairing_echo = v2_operator.apply_v2_safe_operator(
        query_scores,
        budget=1,
        operator="score_prototype_shrinkage",
        variant="A_QM",
        config=p1_config,
        gate_enabled=True,
        support_fbcca_scores=support_scores,
        support_labels=support_labels,
        relative_context_pairing_sha256=pairing_sha,
        **context,
    )
    bad_pairing_rejected = False
    try:
        v2_operator.apply_v2_safe_operator(
            query_scores,
            budget=1,
            operator="score_prototype_shrinkage",
            variant="A_QM",
            config=p1_config,
            gate_enabled=True,
            support_fbcca_scores=support_scores,
            support_labels=support_labels,
            relative_context_pairing_sha256="invalid",
        )
    except (TypeError, ValueError):
        bad_pairing_rejected = True
    gate = _prequential_gate(
        participant,
        product,
        contract,
        budget=3,
        operator="score_prototype_shrinkage",
        variant="A_Q",
        context_control="correct",
        governed_context=_governed_context,
    )
    if (
        pairing_echo.relative_context_pairing_sha256 != pairing_sha
        or not bad_pairing_rejected
        or len(gate.fold_provenance) != 2
        or tuple(item.evaluation_block for item in gate.fold_provenance) != (2, 3)
    ):
        raise RuntimeError("Prefix or pairing provenance rejection/echo failed.")

    operator_parameters = set(inspect.signature(v2_operator.apply_v2_safe_operator).parameters)
    prequential_parameters = set(
        inspect.signature(v2_operator.apply_v2_prequential_operator).parameters
    )
    forbidden_query_capabilities = {
        "query_label",
        "query_labels",
        "query_identity",
        "query_identities",
        "participant_index",
        "sample_id",
        "row_id",
    }
    if operator_parameters & forbidden_query_capabilities or (
        prequential_parameters & forbidden_query_capabilities
    ):
        raise RuntimeError("A V2 operator API exposes query labels or identities.")

    for depth in range(1, 6):
        prefix = tuple(range(1, depth + 1))
        for control in CONTEXT_CONTROLS:
            donors = _metadata_source_blocks(
                participant,
                prefix,
                control,
                contract,
                _governed_context,
            )
            if len(donors) != depth or set(donors) != set(prefix):
                raise RuntimeError("A context donor escaped the current support prefix.")
            if depth == 1 and donors != prefix:
                raise RuntimeError("Depth-one context controls must be exact no-ops.")

    for reserved_seed in _reserved_seed_values(contract):
        try:
            keyed_rng(
                contract,
                root_seed=reserved_seed,
                family=FAMILY_NAMES[0],
                participant_index=0,
                block=0,
                class_index=0,
                component="spatial_signature",
            )
        except ValueError:
            pass
        else:  # pragma: no cover - protects the governance implementation itself.
            raise RuntimeError("A public RNG primitive accepted a reserved seed.")

    canonical_paths = contract.plan["freeze_and_stopping"]["canonical_execution_paths"]
    if Path(canonical_paths["seed_global_lockbox_claim"]) == Path(
        canonical_paths["lockbox_result"]
    ) or not bool(
        contract.plan["freeze_and_stopping"][
            "alternate_result_path_cannot_create_a_second_lockbox_claim"
        ]
    ):
        raise RuntimeError("The frozen global claim binding is not canonical.")

    assertion_names = tuple(contract.plan["hard_assertions_before_efficacy"])
    return {name: True for name in assertion_names}


def _eauc(rows: Mapping[tuple[str, int], float], method: str) -> float:
    return float(
        sum(EAUC_BUDGET_WEIGHTS[budget] * rows[(method, budget)] for budget in EAUC_BUDGET_WEIGHTS)
    )


def _contrast_vectors(
    metric_rows: Sequence[Mapping[str, Any]],
    families: Sequence[str],
) -> dict[str, tuple[str, np.ndarray, float]]:
    by_participant: dict[tuple[str, int], dict[tuple[str, int], float]] = {}
    for row in metric_rows:
        key = (str(row["family"]), int(row["participant_index"]))
        method_key = (str(row["method"]), int(row["budget"]))
        if method_key in by_participant.setdefault(key, {}):
            raise RuntimeError("Duplicate synthetic participant metric cell.")
        by_participant[key][method_key] = float(row["balanced_accuracy"])

    available = set(families)
    vectors: dict[str, tuple[str, np.ndarray, float]] = {}

    def participant_values(family: str, function: Any) -> np.ndarray:
        keys = sorted(key for key in by_participant if key[0] == family)
        if not keys:
            raise RuntimeError(f"No participant metrics for {family}.")
        return np.asarray([function(by_participant[key]) for key in keys], dtype=np.float64)

    b1 = "B1_participant_specific_class_confusion"
    if b1 in available:
        vectors["B1:P1_A_Q-A0:eAUC"] = (
            b1,
            participant_values(b1, lambda rows: _eauc(rows, "P1_A_Q") - _eauc(rows, "A0")),
            0.0,
        )
    b2 = "B2_participant_phase_spatial_shift"
    if b2 in available:
        vectors["B2:P2_A_Q-A0:eAUC"] = (
            b2,
            participant_values(b2, lambda rows: _eauc(rows, "P2_A_Q") - _eauc(rows, "A0")),
            0.0,
        )
    b3 = "B3_context_dependent_support_query_shift"
    if b3 in available:
        vectors["B3:P1_A_QM_correct-P1_A_Q:eAUC"] = (
            b3,
            participant_values(
                b3,
                lambda rows: _eauc(rows, "P1_A_QM_correct") - _eauc(rows, "P1_A_Q"),
            ),
            0.0,
        )
        vectors["B3:P1_A_QM_correct-P1_A_QM_pair_shuffled:eAUC"] = (
            b3,
            participant_values(
                b3,
                lambda rows: _eauc(rows, "P1_A_QM_correct") - _eauc(rows, "P1_A_QM_pair_shuffled"),
            ),
            0.0,
        )
    n1 = "N1_perfect_clean_anchor"
    if n1 in available:
        for method in ("P1_A_Q", "P2_A_Q"):
            vectors[f"N1:{method}-A0:eAUC"] = (
                n1,
                participant_values(
                    n1,
                    lambda rows, method=method: _eauc(rows, method) - _eauc(rows, "A0"),
                ),
                -1.0 / 60.0,
            )
            for budget in (1, 3, 5):
                name = f"N1:{method}-A0:k{budget}"
                vectors[name] = (
                    n1,
                    participant_values(
                        n1,
                        lambda rows, method=method, budget=budget: (
                            rows[(method, budget)] - rows[("A0", budget)]
                        ),
                    ),
                    -1.0 / 60.0,
                )
    for short, adversarial_family in (
        ("N2", "N2_random_fit_support_labels"),
        ("N3", "N3_nonstationary_corrupted_calibration"),
    ):
        if adversarial_family in available:
            vectors[f"{short}:P1_A_Q-A0:eAUC"] = (
                adversarial_family,
                participant_values(
                    adversarial_family,
                    lambda rows: _eauc(rows, "P1_A_Q") - _eauc(rows, "A0"),
                ),
                0.0,
            )
    n4 = "N4_context_independent_of_signal"
    if n4 in available:
        vectors["N4:P1_A_QM_correct-P1_A_Q:eAUC"] = (
            n4,
            participant_values(
                n4,
                lambda rows: _eauc(rows, "P1_A_QM_correct") - _eauc(rows, "P1_A_Q"),
            ),
            -1.0 / 60.0,
        )
        for budget in (1, 3, 5):
            name = f"N4:P1_A_QM_correct-P1_A_Q:k{budget}"
            vectors[name] = (
                n4,
                participant_values(
                    n4,
                    lambda rows, budget=budget: (
                        rows[("P1_A_QM_correct", budget)] - rows[("P1_A_Q", budget)]
                    ),
                ),
                -1.0 / 60.0,
            )
        vectors["N4:P1_A_QM_correct-P1_A_QM_pair_shuffled:eAUC"] = (
            n4,
            participant_values(
                n4,
                lambda rows: _eauc(rows, "P1_A_QM_correct") - _eauc(rows, "P1_A_QM_pair_shuffled"),
            ),
            -1.0 / 60.0,
        )
    return vectors


def _one_sided_t_lower_bound(values: np.ndarray, confidence: float = 0.95) -> float:
    observed = np.asarray(values, dtype=np.float64)
    if observed.ndim != 1 or len(observed) < 2 or not np.isfinite(observed).all():
        raise ValueError("Paired-t lower bound requires at least two finite participant deltas.")
    mean = float(np.mean(observed))
    standard_deviation = float(np.std(observed, ddof=1))
    if standard_deviation == 0.0:
        return mean
    critical = float(student_t.ppf(confidence, df=len(observed) - 1))
    return mean - critical * standard_deviation / math.sqrt(len(observed))


def _clopper_pearson_upper(successes: int, total: int, confidence: float = 0.95) -> float:
    if type(successes) is not int or type(total) is not int or not 0 <= successes <= total:
        raise ValueError("Exact binomial counts are invalid.")
    if total < 1:
        raise ValueError("Exact binomial interval requires observations.")
    if successes == total:
        return 1.0
    return float(beta_distribution.ppf(confidence, successes + 1, total - successes))


def _sensitivity_summary(
    values: np.ndarray,
    *,
    contract: SyntheticContract,
    root_seed: int,
    family: str,
    contrast_index: int,
    sign_flip_draws: int,
    bootstrap_draws: int,
    governed_context: _GovernedSeedContext | None = None,
) -> dict[str, Any]:
    observed = np.asarray(values, dtype=np.float64)
    mean = float(np.mean(observed))
    sign_rng = keyed_rng(
        contract,
        root_seed=root_seed,
        family=family,
        participant_index=0,
        block=1,
        class_index=contrast_index,
        component="sensitivity_resampling",
        _governed_context=governed_context,
    )
    exceed = 0
    complete = 0
    while complete < sign_flip_draws:
        count = min(4096, sign_flip_draws - complete)
        signs = sign_rng.integers(0, 2, size=(count, len(observed)), dtype=np.int8)
        signs = signs.astype(np.float64) * 2.0 - 1.0
        null_means = np.mean(signs * observed[None, :], axis=1)
        exceed += int(np.count_nonzero(null_means >= mean))
        complete += count
    sign_p = (exceed + 1.0) / (sign_flip_draws + 1.0)

    bootstrap_rng = keyed_rng(
        contract,
        root_seed=root_seed,
        family=family,
        participant_index=0,
        block=2,
        class_index=contrast_index,
        component="sensitivity_resampling",
        _governed_context=governed_context,
    )
    bootstrap_means = np.empty(bootstrap_draws, dtype=np.float64)
    complete = 0
    while complete < bootstrap_draws:
        count = min(2048, bootstrap_draws - complete)
        indices = bootstrap_rng.integers(0, len(observed), size=(count, len(observed)))
        bootstrap_means[complete : complete + count] = np.mean(observed[indices], axis=1)
        complete += count
    return {
        "sign_flip": {
            "draws": sign_flip_draws,
            "one_sided_p_greater_than_zero": float(sign_p),
        },
        "participant_bootstrap": {
            "draws": bootstrap_draws,
            "two_sided_95_percent_interval": [
                float(np.quantile(bootstrap_means, 0.025)),
                float(np.quantile(bootstrap_means, 0.975)),
            ],
        },
    }


def summarize_synthetic_metrics(
    metric_rows: Sequence[Mapping[str, Any]],
    *,
    contract: SyntheticContract,
    root_seed: int,
    families: Sequence[str],
    sign_flip_draws: int,
    bootstrap_draws: int,
    _governed_context: _GovernedSeedContext | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    _require_seed_access(contract, root_seed, _governed_context)
    vectors = _contrast_vectors(metric_rows, families)
    severe_harm_ids = tuple(contract.plan["promotion_requirements"]["severe_harm"]["contrast_ids"])
    if set(severe_harm_ids) - set(vectors) and set(families) == set(FAMILY_NAMES):
        raise RuntimeError("Frozen severe-harm contrasts are missing from the complete grid.")
    summaries: list[dict[str, Any]] = []
    for contrast_index, name in enumerate(sorted(vectors)):
        family, values, margin = vectors[name]
        severe_harm_applicable = name in severe_harm_ids
        harmful = int(np.count_nonzero(values < -0.10)) if severe_harm_applicable else 0
        summary = {
            "contrast": name,
            "family": family,
            "n_participants": len(values),
            "participant_deltas": values.tolist(),
            "observed_mean": float(np.mean(values)),
            "paired_t_one_sided_95_percent_lower_bound": _one_sided_t_lower_bound(values),
            "decision_margin": margin,
            "severe_harm": {
                "definition": "participant_delta_less_than_-0.10",
                "scale": "participant_eAUC_delta",
                "applicable": severe_harm_applicable,
                "count": harmful if severe_harm_applicable else None,
                "exact_one_sided_95_percent_upper_rate": (
                    _clopper_pearson_upper(harmful, len(values)) if severe_harm_applicable else None
                ),
            },
            "sensitivity_only": _sensitivity_summary(
                values,
                contract=contract,
                root_seed=root_seed,
                family=family,
                contrast_index=contrast_index,
                sign_flip_draws=sign_flip_draws,
                bootstrap_draws=bootstrap_draws,
                governed_context=_governed_context,
            ),
        }
        summaries.append(summary)

    by_name = {str(item["contrast"]): item for item in summaries}
    required_families = set(FAMILY_NAMES)
    participant_counts = {
        family: len(
            {int(row["participant_index"]) for row in metric_rows if str(row["family"]) == family}
        )
        for family in set(families)
    }
    complete = set(families) == required_families and all(
        participant_counts.get(family) == 64 for family in required_families
    )
    if not complete:
        return summaries, {
            "evaluated": False,
            "reason": "diagnostic_subset_cannot_evaluate_frozen_promotion_requirements",
            "all_requirements_passed": False,
        }

    helpful_names = {
        "B1": "B1:P1_A_Q-A0:eAUC",
        "B2": "B2:P2_A_Q-A0:eAUC",
        "B3": "B3:P1_A_QM_correct-P1_A_Q:eAUC",
    }
    helpful: dict[str, Any] = {}
    for short, name in helpful_names.items():
        item = by_name[name]
        passed = (
            item["paired_t_one_sided_95_percent_lower_bound"] > 0.0
            and item["observed_mean"] >= 0.01
        )
        helpful[short] = {"contrast": name, "passed": bool(passed)}
    helpful_count = sum(bool(item["passed"]) for item in helpful.values())
    helpful_pass = helpful_count >= 2

    b3_mechanism_name = "B3:P1_A_QM_correct-P1_A_QM_pair_shuffled:eAUC"
    b3_mechanism = by_name[b3_mechanism_name]["observed_mean"] >= 0.01

    n1_names = [name for name in by_name if name.startswith("N1:") and ":k" in name]
    n4_names = [name for name in by_name if name.startswith("N4:P1_A_QM_correct-P1_A_Q:k")]
    n1_noninferiority = all(
        by_name[name]["paired_t_one_sided_95_percent_lower_bound"] > -1.0 / 60.0
        for name in n1_names
    )
    n4_noninferiority = all(
        by_name[name]["paired_t_one_sided_95_percent_lower_bound"] > -1.0 / 60.0
        for name in n4_names
    )

    adversarial_rows = [
        row
        for row in metric_rows
        if row["family"]
        in {"N2_random_fit_support_labels", "N3_nonstationary_corrupted_calibration"}
        and row["method"] == "P1_A_Q"
        and row["budget"] in {3, 5}
    ]
    expected_adversarial = 2 * 64 * 2
    if len(adversarial_rows) != expected_adversarial:
        raise RuntimeError("Adversarial abstention denominator is not the frozen complete grid.")
    abstentions = sum(row["gate_enabled"] is False for row in adversarial_rows)
    abstention_rate = abstentions / len(adversarial_rows)
    adversarial_pass = abstention_rate >= 0.95

    observed_safety_set = {
        str(item["contrast"]) for item in summaries if item["severe_harm"]["applicable"]
    }
    if observed_safety_set != set(severe_harm_ids):
        raise RuntimeError("Severe-harm applicability must equal the exact frozen eight-ID set.")
    severe_harm_pass = all(
        item["severe_harm"]["exact_one_sided_95_percent_upper_rate"] < 0.10
        for item in summaries
        if item["severe_harm"]["applicable"]
    )
    all_pass = all(
        (
            helpful_pass,
            b3_mechanism,
            n1_noninferiority,
            n4_noninferiority,
            adversarial_pass,
            severe_harm_pass,
        )
    )
    return summaries, {
        "evaluated": True,
        "helpful_families": {
            "details": helpful,
            "passed_count": helpful_count,
            "required_count": 2,
            "passed": helpful_pass,
        },
        "B3_pairing_mechanism": {
            "contrast": b3_mechanism_name,
            "observed_minimum_gain": 0.01,
            "passed": bool(b3_mechanism),
        },
        "null_noninferiority": {
            "N1": {"contrasts": sorted(n1_names), "passed": bool(n1_noninferiority)},
            "N4": {"contrasts": sorted(n4_names), "passed": bool(n4_noninferiority)},
            "passed": bool(n1_noninferiority and n4_noninferiority),
        },
        "adversarial_prequential_abstention": {
            "abstentions": abstentions,
            "denominator": len(adversarial_rows),
            "rate": abstention_rate,
            "minimum_rate": 0.95,
            "passed": bool(adversarial_pass),
        },
        "severe_harm": {
            "contrast_ids": list(severe_harm_ids),
            "all_declared_safety_contrasts_exact_upper_rate_below_0.10": bool(severe_harm_pass),
            "passed": bool(severe_harm_pass),
        },
        "all_requirements_passed": bool(all_pass),
    }


def _execute_synthetic(
    contract: SyntheticContract,
    *,
    phase: str,
    root_seed: int,
    participants_per_family: int,
    families: Sequence[str],
    sign_flip_draws: int,
    bootstrap_draws: int,
    fbcca_producer: Any = produce_strict_fbcca,
    governed_context: _GovernedSeedContext | None = None,
) -> dict[str, Any]:
    _require_seed_access(contract, root_seed, governed_context)
    if type(participants_per_family) is not int or participants_per_family < 2:
        raise ValueError("Synthetic execution requires at least two participants per family.")
    selected_families = tuple(families)
    if len(set(selected_families)) != len(selected_families) or any(
        family not in FAMILY_NAMES for family in selected_families
    ):
        raise ValueError("Synthetic execution family selection is invalid.")
    if type(sign_flip_draws) is not int or sign_flip_draws < 1:
        raise ValueError("sign_flip_draws must be a positive exact integer.")
    if type(bootstrap_draws) is not int or bootstrap_draws < 1:
        raise ValueError("bootstrap_draws must be a positive exact integer.")
    if root_seed in _reserved_seed_values(contract):
        sensitivity = contract.plan["metrics"]["sensitivity_only"]
        if (
            governed_context is None
            or governed_context.phase != phase
            or participants_per_family != 64
            or selected_families != FAMILY_NAMES
            or sign_flip_draws != int(sensitivity["sign_flip"]["draws"])
            or bootstrap_draws != int(sensitivity["participant_bootstrap"]["draws"])
            or fbcca_producer is not produce_strict_fbcca
        ):
            raise ValueError(
                "Reserved seed execution has no alternate size, family, RNG, or producer path."
            )

    metric_rows: list[dict[str, Any]] = []
    hard_assertions: dict[str, bool] | None = None
    for family in selected_families:
        for participant_index in range(participants_per_family):
            participant = generate_synthetic_participant(
                contract,
                root_seed=root_seed,
                family=family,
                participant_index=participant_index,
                _governed_context=governed_context,
            )
            product = fbcca_producer(contract, participant.signals)
            if not isinstance(product, StrictFBCCAProduct):
                raise TypeError("FBCCA producer must return StrictFBCCAProduct.")
            if hard_assertions is None:
                try:
                    hard_assertions = validate_hard_assertions(
                        participant,
                        product,
                        contract,
                        _governed_context=governed_context,
                    )
                except (MemoryError, OSError):
                    raise
                except (AssertionError, RuntimeError, TypeError, ValueError) as error:
                    raise HardInvariantFailure(str(error)) from error
                expected_assertions = {
                    name: True for name in contract.plan["hard_assertions_before_efficacy"]
                }
                if hard_assertions != expected_assertions:
                    raise HardInvariantFailure(
                        "Hard-assertion key set/value set differs from the frozen plan."
                    )
            # Efficacy metrics are not evaluated until all hard assertions have passed.
            metric_rows.extend(
                evaluate_synthetic_participant(
                    participant,
                    product,
                    contract,
                    _governed_context=governed_context,
                )
            )
    if hard_assertions is None or not all(hard_assertions.values()):
        raise HardInvariantFailure(
            "Synthetic hard assertions did not complete before efficacy metrics."
        )
    contrasts, promotion = summarize_synthetic_metrics(
        metric_rows,
        contract=contract,
        root_seed=root_seed,
        families=selected_families,
        sign_flip_draws=sign_flip_draws,
        bootstrap_draws=bootstrap_draws,
        _governed_context=governed_context,
    )
    status = (
        "engineering_only"
        if phase in {"diagnostic", "development"}
        else "terminal_pass"
        if promotion["all_requirements_passed"]
        else "terminal_scientific_fail"
    )
    result: dict[str, Any] = {
        "schema": SYNTHETIC_RESULT_SCHEMA,
        "candidate_id": "metadata-calibration-efficiency-v2",
        "phase": phase,
        "status": status,
        "human_claim_boundary": "synthetic_pass_is_necessary_but_never_sufficient",
        "plan_sha256": contract.plan_sha256,
        "filterbank_sha256": contract.filterbank_sha256,
        "diagnostic_operator_configs_sha256": contract.diagnostic_operator_configs_sha256,
        "root_seed": root_seed,
        "rng": {
            "bit_generator": "numpy.random.PCG64DXSM",
            "construction": "numpy.random.SeedSequence",
            "complete_key_order": list(contract.plan["rng"]["key_order"]),
        },
        "participants_per_family": participants_per_family,
        "families": list(selected_families),
        "support_budgets": list(SUPPORT_BUDGETS),
        "query_blocks": list(contract.plan["population"]["immutable_query_blocks"]),
        "hard_assertions_completed_before_metrics": True,
        "hard_assertions": hard_assertions,
        "participant_metrics": metric_rows,
        "primary_contrasts": contrasts,
        "promotion": promotion,
        "development_outputs_are_engineering_only": phase == "development",
        "lockbox_terminal": phase == "lockbox",
    }
    result["result_sha256"] = _canonical_json_sha256(result)
    return result


def run_synthetic_diagnostic(
    *,
    root_seed: int,
    participants_per_family: int = 2,
    families: Sequence[str] = FAMILY_NAMES,
    sign_flip_draws: int = 128,
    bootstrap_draws: int = 128,
    plan_path: str | Path = DEFAULT_SYNTHETIC_PLAN_PATH,
    fbcca_producer: Any = produce_strict_fbcca,
) -> dict[str, Any]:
    """Run a small non-reserved engineering diagnostic for unit/property tests."""

    contract = validate_synthetic_contract(plan_path)
    if root_seed in _reserved_seed_values(contract):
        raise ValueError("Reserved development/lockbox seeds cannot use the diagnostic path.")
    return _execute_synthetic(
        contract,
        phase="diagnostic",
        root_seed=root_seed,
        participants_per_family=participants_per_family,
        families=families,
        sign_flip_draws=sign_flip_draws,
        bootstrap_draws=bootstrap_draws,
        fbcca_producer=fbcca_producer,
    )


def _json_payload_bytes(payload: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(
            dict(payload),
            sort_keys=True,
            indent=2,
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
        + b"\n"
    )


def _validated_unused_output_path(path: str | Path) -> Path:
    parent_descriptor, candidate = _open_parent_directory_nofollow(path, "governed output")
    try:
        parent_stat = os.fstat(parent_descriptor)
        if (
            stat.S_IMODE(parent_stat.st_mode) != 0o700
            or parent_stat.st_uid != os.geteuid()
        ):
            raise ValueError("Governed output parent must be owner-controlled mode 0700.")
        try:
            os.stat(candidate.name, dir_fd=parent_descriptor, follow_symlinks=False)
        except FileNotFoundError:
            return candidate
        raise FileExistsError(f"Refusing to overwrite existing output: {candidate}")
    finally:
        os.close(parent_descriptor)


def _write_bytes_exclusive(path: str | Path, payload: bytes) -> Path:
    parent_descriptor, target = _open_parent_directory_nofollow(path, "governed output")
    try:
        parent_stat = os.fstat(parent_descriptor)
        if (
            stat.S_IMODE(parent_stat.st_mode) != 0o700
            or parent_stat.st_uid != os.geteuid()
        ):
            raise ValueError("Governed output parent must be owner-controlled mode 0700.")
        flags = (
            os.O_WRONLY
            | os.O_CREAT
            | os.O_EXCL
            | getattr(os, "O_NOFOLLOW", 0)
            | getattr(os, "O_CLOEXEC", 0)
        )
        descriptor = os.open(
            target.name,
            flags,
            stat.S_IRUSR | stat.S_IWUSR,
            dir_fd=parent_descriptor,
        )
        with os.fdopen(descriptor, "wb", closefd=True) as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
            os.fchmod(handle.fileno(), stat.S_IRUSR)
            os.fsync(handle.fileno())
        os.fsync(parent_descriptor)
        verification_descriptor, _ = _open_parent_directory_nofollow(
            target, "governed output verification"
        )
        try:
            verification_stat = os.fstat(verification_descriptor)
            if (verification_stat.st_dev, verification_stat.st_ino) != (
                parent_stat.st_dev,
                parent_stat.st_ino,
            ):
                raise RuntimeError("Governed output parent changed during exclusive write.")
        finally:
            os.close(verification_descriptor)
    finally:
        os.close(parent_descriptor)
    return target


def write_json_exclusive(path: str | Path, payload: Mapping[str, Any]) -> Path:
    """Write a JSON receipt with O_EXCL and never replace an existing artifact."""

    return _write_bytes_exclusive(path, _json_payload_bytes(payload))


def load_json_object(path: str | Path, *, name: str) -> dict[str, Any]:
    source = _regular_nonsymlink_file(path, name)
    decoded = json.loads(_read_bytes_nofollow(source, name).decode("utf-8"))
    if not isinstance(decoded, Mapping):
        raise TypeError(f"{name} must decode to one JSON object.")
    return dict(decoded)


def current_environment_receipt() -> dict[str, Any]:
    configuration = io.StringIO()
    with redirect_stdout(configuration):
        np.show_config()
    fields: dict[str, Any] = {
        "python_implementation": platform_module.python_implementation(),
        "python_version": sys.version,
        "python_executable": os.path.abspath(sys.executable),
        "python_executable_realpath": os.path.realpath(sys.executable),
        "python_prefix": os.path.abspath(sys.prefix),
        "python_base_prefix": os.path.abspath(sys.base_prefix),
        "platform": platform_module.platform(),
        "numpy_version": np.__version__,
        "numpy_configuration_sha256": _sha256_bytes(configuration.getvalue().encode("utf-8")),
        "scipy_version": scipy.__version__,
        "thread_environment": {name: os.environ.get(name) for name in _THREAD_ENVIRONMENT_NAMES},
    }
    required = set(
        validate_synthetic_contract().plan["freeze_and_stopping"]["execution_identity"][
            "required_environment_fields"
        ]
    )
    if set(fields) != required:
        raise RuntimeError("Runtime environment fields differ from the frozen contract.")
    fields["environment_sha256"] = _canonical_json_sha256(fields)
    return fields


def _validate_environment_receipt(environment: Mapping[str, Any]) -> dict[str, Any]:
    observed = dict(environment)
    expected_fields = {
        "python_implementation",
        "python_version",
        "python_executable",
        "python_executable_realpath",
        "python_prefix",
        "python_base_prefix",
        "platform",
        "numpy_version",
        "numpy_configuration_sha256",
        "scipy_version",
        "thread_environment",
        "environment_sha256",
    }
    _require_exact_keys(observed, expected_fields, "execution environment")
    claimed = observed.pop("environment_sha256")
    if not isinstance(claimed, str) or not _SHA256_RE.fullmatch(claimed):
        raise ValueError("Environment receipt requires a lowercase SHA-256 self-hash.")
    if claimed != _canonical_json_sha256(observed):
        raise ValueError("Environment receipt self-hash mismatch.")
    observed["environment_sha256"] = claimed
    return observed


def current_git_identity(repository: str | Path = _REPOSITORY) -> GitIdentity:
    """Bind exact HEAD, HEAD tree, every tracked file, and full worktree status."""

    root = Path(repository).resolve()

    def git_text(*arguments: str) -> str:
        completed = subprocess.run(
            ["git", *arguments],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        )
        return completed.stdout.strip()

    commit = git_text("rev-parse", "HEAD")
    tree = git_text("rev-parse", "HEAD^{tree}")
    commit_timestamp = git_text("show", "-s", "--format=%cI", "HEAD")
    if not _GIT_OBJECT_RE.fullmatch(commit) or not _GIT_OBJECT_RE.fullmatch(tree):
        raise RuntimeError("Git did not return valid exact HEAD/tree object IDs.")
    status_output = git_text("status", "--porcelain=v1", "--untracked-files=all")
    tracked = subprocess.run(
        ["git", "ls-files", "-z"],
        cwd=root,
        check=True,
        capture_output=True,
    ).stdout
    records: list[dict[str, str]] = []
    for raw_relative in tracked.split(b"\0"):
        if not raw_relative:
            continue
        relative_text = raw_relative.decode("utf-8", errors="surrogateescape")
        source = root / relative_text
        if source.is_symlink():
            digest = _sha256_bytes(os.readlink(source).encode("utf-8"))
            kind = "symlink-target"
        elif source.is_file():
            digest = _sha256_file(source)
            kind = "regular-file"
        else:
            raise RuntimeError(f"Tracked source is not materialized safely: {relative_text}")
        records.append({"path": relative_text, "kind": kind, "sha256": digest})
    source_bundle = {
        "schema": "cfeg.metadata-calibration-v2-synthetic-source-bundle.v2",
        "head": commit,
        "tree": tree,
        "files": records,
    }
    return GitIdentity(
        commit=commit,
        tree=tree,
        source_bundle_sha256=_canonical_json_sha256(source_bundle),
        clean=status_output == "",
        commit_timestamp_utc=(
            datetime.fromisoformat(commit_timestamp.replace("Z", "+00:00"))
            .astimezone(timezone.utc)
            .isoformat(timespec="seconds")
            .replace("+00:00", "Z")
        ),
    )


def _identity_mapping(identity: GitIdentity) -> dict[str, Any]:
    return {
        "generator_commit": identity.commit,
        "generator_tree": identity.tree,
        "generator_source_bundle_sha256": identity.source_bundle_sha256,
        "generator_commit_timestamp_utc": identity.commit_timestamp_utc,
        "git_worktree_clean": identity.clean,
    }


def _require_clean_identity(identity: GitIdentity) -> None:
    if not identity.clean:
        raise ValueError("Governed synthetic execution requires the entire git worktree clean.")


def _source_hashes(repository: Path = _REPOSITORY) -> dict[str, str]:
    root = Path(repository).resolve()
    completed = subprocess.run(["git", "ls-files", "-z"], cwd=root, check=True, capture_output=True)
    result: dict[str, str] = {}
    for raw_relative in completed.stdout.split(b"\0"):
        if raw_relative:
            relative = raw_relative.decode("utf-8", errors="surrogateescape")
            source = root / relative
            result[relative] = (
                _sha256_bytes(os.readlink(source).encode("utf-8"))
                if source.is_symlink()
                else _sha256_file(source)
            )
    return result


_PREPARATION_FIELDS = {
    "schema",
    "candidate_id",
    "phase",
    "status",
    "plan_path",
    "plan_sha256",
    "filterbank_path",
    "filterbank_sha256",
    "diagnostic_operator_configs_sha256",
    "generator_commit",
    "generator_tree",
    "generator_source_bundle_sha256",
    "git_worktree_clean",
    "generator_commit_timestamp_utc",
    "created_at_utc",
    "environment",
    "canonical_execution_paths",
    "lockbox_seed_source",
    "reserved_seed_executed",
    "requested_output_path",
    "completion_receipt_sha256",
}


def build_preparation_receipt(
    contract: SyntheticContract,
    *,
    output_path: str | Path,
) -> dict[str, Any]:
    target = _require_canonical_path(contract, "preparation_receipt", output_path)
    _validated_unused_output_path(target)
    identity = current_git_identity()
    _require_clean_identity(identity)
    created = datetime.now(timezone.utc)
    commit_time = datetime.fromisoformat(
        identity.commit_timestamp_utc.replace("Z", "+00:00")
    )
    if created >= _target_datetime(contract) or commit_time >= _target_datetime(contract):
        raise ValueError("Preparation and exact HEAD commit must both predate the beacon target.")
    environment = _validate_environment_receipt(current_environment_receipt())
    receipt: dict[str, Any] = {
        "schema": SYNTHETIC_PREPARATION_SCHEMA,
        "candidate_id": "metadata-calibration-efficiency-v2",
        "phase": "prepare",
        "status": "validated_no_outcome_execution",
        "plan_path": str(contract.plan_path),
        "plan_sha256": contract.plan_sha256,
        "filterbank_path": str(contract.filterbank_path),
        "filterbank_sha256": contract.filterbank_sha256,
        "diagnostic_operator_configs_sha256": contract.diagnostic_operator_configs_sha256,
        **_identity_mapping(identity),
        "created_at_utc": created.isoformat(timespec="milliseconds").replace("+00:00", "Z"),
        "environment": environment,
        "canonical_execution_paths": dict(
            contract.plan["freeze_and_stopping"]["canonical_execution_paths"]
        ),
        "lockbox_seed_source": "exact_future_public_randomness_beacon_pulse",
        "reserved_seed_executed": False,
        "requested_output_path": str(target),
    }
    receipt["completion_receipt_sha256"] = _canonical_json_sha256(receipt)
    return receipt


def _validate_preparation_receipt(
    contract: SyntheticContract,
    *,
    identity: GitIdentity,
    environment: Mapping[str, Any],
) -> tuple[dict[str, Any], Path]:
    path = canonical_execution_path(contract, "preparation_receipt")
    _readonly_receipt_file(path, "synthetic preparation receipt")
    _require_file_strictly_predates_target(path, contract, "synthetic preparation receipt")
    receipt = load_json_object(path, name="synthetic preparation receipt")
    _require_exact_keys(receipt, _PREPARATION_FIELDS, "synthetic preparation receipt")
    claimed = receipt.pop("completion_receipt_sha256")
    if claimed != _canonical_json_sha256(receipt):
        raise ValueError("Synthetic preparation receipt self-hash mismatch.")
    created_at = receipt.get("created_at_utc")
    if not isinstance(created_at, str):
        raise TypeError("Preparation receipt lacks its exact creation time.")
    created_time = datetime.fromisoformat(created_at.replace("Z", "+00:00"))
    commit_time = datetime.fromisoformat(identity.commit_timestamp_utc.replace("Z", "+00:00"))
    if created_time >= _target_datetime(contract) or commit_time >= _target_datetime(contract):
        raise ValueError("Preparation/source freeze did not strictly predate the beacon target.")
    expected = {
        "schema": SYNTHETIC_PREPARATION_SCHEMA,
        "candidate_id": "metadata-calibration-efficiency-v2",
        "phase": "prepare",
        "status": "validated_no_outcome_execution",
        "plan_path": str(contract.plan_path),
        "plan_sha256": contract.plan_sha256,
        "filterbank_path": str(contract.filterbank_path),
        "filterbank_sha256": contract.filterbank_sha256,
        "diagnostic_operator_configs_sha256": contract.diagnostic_operator_configs_sha256,
        **_identity_mapping(identity),
        "created_at_utc": created_at,
        "environment": dict(environment),
        "canonical_execution_paths": dict(
            contract.plan["freeze_and_stopping"]["canonical_execution_paths"]
        ),
        "lockbox_seed_source": "exact_future_public_randomness_beacon_pulse",
        "reserved_seed_executed": False,
        "requested_output_path": str(path),
    }
    if receipt != expected:
        raise ValueError("Synthetic preparation receipt is incomplete, stale, or noncanonical.")
    receipt["completion_receipt_sha256"] = claimed
    return receipt, path


_PARTICIPANT_METRIC_FIELDS = {
    "family",
    "participant_index",
    "method",
    "budget",
    "balanced_accuracy",
    "gate_enabled",
    "gate_reason",
    "exact_fallback",
    "prediction_sha256",
    "producer_sha256",
    "operator_schema",
    "support_depth",
    "relative_context_pairing_sha256",
    "comparable_context_pair_count",
    "prequential",
}
_PREQUENTIAL_FIELDS = {
    "evaluated_blocks",
    "block_balanced_accuracy_deltas",
    "block_log_probability_deltas",
    "mean_balanced_accuracy_delta",
    "mean_log_probability_delta",
    "fold_provenance",
}
_FOLD_PROVENANCE_FIELDS = {
    "schema",
    "evaluation_block",
    "fit_blocks",
    "fit_block_partition_sha256s",
    "fit_partition_sha256",
    "evaluation_partition_sha256",
}
_CONTRAST_FIELDS = {
    "contrast",
    "family",
    "n_participants",
    "participant_deltas",
    "observed_mean",
    "paired_t_one_sided_95_percent_lower_bound",
    "decision_margin",
    "severe_harm",
    "sensitivity_only",
}
_SEVERE_HARM_FIELDS = {
    "definition",
    "scale",
    "applicable",
    "count",
    "exact_one_sided_95_percent_upper_rate",
}
_SENSITIVITY_FIELDS = {"sign_flip", "participant_bootstrap"}
_RESULT_BASE_FIELDS = {
    "schema",
    "candidate_id",
    "phase",
    "status",
    "human_claim_boundary",
    "plan_sha256",
    "filterbank_sha256",
    "diagnostic_operator_configs_sha256",
    "root_seed",
    "rng",
    "participants_per_family",
    "families",
    "support_budgets",
    "query_blocks",
    "hard_assertions_completed_before_metrics",
    "hard_assertions",
    "participant_metrics",
    "primary_contrasts",
    "promotion",
    "development_outputs_are_engineering_only",
    "lockbox_terminal",
}
_DEVELOPMENT_RESULT_FIELDS = _RESULT_BASE_FIELDS | {
    "generator_commit",
    "generator_tree",
    "generator_source_bundle_sha256",
    "generator_commit_timestamp_utc",
    "environment",
    "preparation_receipt_sha256",
    "result_sha256",
}
_LOCKBOX_RESULT_FIELDS = _RESULT_BASE_FIELDS | {
    "generator_commit",
    "generator_tree",
    "generator_source_bundle_sha256",
    "generator_commit_timestamp_utc",
    "environment",
    "authorization_receipt_sha256",
    "beacon_receipt_sha256",
    "one_time_claim_sha256",
    "result_sha256",
}


def _expected_method_grid() -> dict[str, set[tuple[str, int]]]:
    result: dict[str, set[tuple[str, int]]] = {}
    for family in FAMILY_NAMES:
        methods = {"A0", *(item[0] for item in _method_specs_for_family(family))}
        result[family] = {(method, budget) for method in methods for budget in SUPPORT_BUDGETS}
    return result


def _validate_metric_row_schema(row: Mapping[str, Any], index: int) -> None:
    _require_exact_keys(dict(row), _PARTICIPANT_METRIC_FIELDS, f"participant metric row {index}")
    if (
        row["family"] not in FAMILY_NAMES
        or type(row["participant_index"]) is not int
        or not 0 <= row["participant_index"] < 64
        or type(row["budget"]) is not int
        or row["budget"] not in SUPPORT_BUDGETS
        or not isinstance(row["method"], str)
        or not isinstance(row["balanced_accuracy"], (int, float))
        or not math.isfinite(float(row["balanced_accuracy"]))
        or not 0.0 <= float(row["balanced_accuracy"]) <= 1.0
        or row["gate_enabled"] not in {None, True, False}
        or type(row["exact_fallback"]) is not bool
        or type(row["support_depth"]) is not int
        or type(row["comparable_context_pair_count"]) is not int
    ):
        raise ValueError(f"Participant metric row {index} contains invalid frozen values.")
    for digest_name in ("prediction_sha256", "producer_sha256"):
        if not isinstance(row[digest_name], str) or not _SHA256_RE.fullmatch(row[digest_name]):
            raise ValueError(f"Participant metric row {index} has invalid {digest_name}.")
    prequential = row["prequential"]
    if prequential is not None:
        if not isinstance(prequential, Mapping):
            raise TypeError("Prequential metric evidence must be a mapping or null.")
        _require_exact_keys(
            dict(prequential), _PREQUENTIAL_FIELDS, f"participant metric prequential {index}"
        )
        if not isinstance(prequential["fold_provenance"], list):
            raise TypeError("Prequential fold provenance must be a list.")
        for fold_index, fold in enumerate(prequential["fold_provenance"]):
            if not isinstance(fold, Mapping):
                raise TypeError("Prequential fold provenance entries must be mappings.")
            _require_exact_keys(
                dict(fold),
                _FOLD_PROVENANCE_FIELDS,
                f"participant metric fold {index}:{fold_index}",
            )


def _validate_complete_metric_grid(rows: Any) -> list[Mapping[str, Any]]:
    if not isinstance(rows, list) or len(rows) != 5888:
        raise ValueError("Development result must contain exactly 5888 participant metric rows.")
    grid = _expected_method_grid()
    observed: dict[tuple[str, int], set[tuple[str, int]]] = {}
    cells: set[tuple[str, int, str, int]] = set()
    for index, row in enumerate(rows):
        if not isinstance(row, Mapping):
            raise TypeError("Every participant metric row must be a mapping.")
        _validate_metric_row_schema(row, index)
        family = str(row["family"])
        participant = int(row["participant_index"])
        cell = (family, participant, str(row["method"]), int(row["budget"]))
        if cell in cells:
            raise ValueError("Development participant metric grid contains a duplicate cell.")
        cells.add(cell)
        observed.setdefault((family, participant), set()).add((cell[2], cell[3]))
    if set(observed) != {
        (family, participant) for family in FAMILY_NAMES for participant in range(64)
    }:
        raise ValueError("Development participant IDs must be exact zero through 63 per family.")
    for (family, _participant), method_budget in observed.items():
        if method_budget != grid[family]:
            raise ValueError("Development family/method/budget grid is incomplete or extra.")
    return rows


def _validate_contrast_schema(contrasts: Any) -> list[Mapping[str, Any]]:
    if not isinstance(contrasts, list) or len(contrasts) != 19:
        raise ValueError("Development result must contain exactly 19 primary contrasts.")
    names: set[str] = set()
    for index, item in enumerate(contrasts):
        if not isinstance(item, Mapping):
            raise TypeError("Every primary contrast must be a mapping.")
        _require_exact_keys(dict(item), _CONTRAST_FIELDS, f"primary contrast {index}")
        name = item["contrast"]
        if not isinstance(name, str) or name in names:
            raise ValueError("Primary contrast IDs must be unique strings.")
        names.add(name)
        if item["n_participants"] != 64 or not isinstance(item["participant_deltas"], list):
            raise ValueError("Every primary contrast must contain 64 participant deltas.")
        deltas = np.asarray(item["participant_deltas"], dtype=np.float64)
        if deltas.shape != (64,) or not np.isfinite(deltas).all():
            raise ValueError("Every primary contrast must contain 64 finite participant deltas.")
        harm = item["severe_harm"]
        sensitivity = item["sensitivity_only"]
        if not isinstance(harm, Mapping) or not isinstance(sensitivity, Mapping):
            raise TypeError("Contrast harm and sensitivity evidence must be mappings.")
        _require_exact_keys(dict(harm), _SEVERE_HARM_FIELDS, f"contrast harm {index}")
        _require_exact_keys(dict(sensitivity), _SENSITIVITY_FIELDS, f"sensitivity {index}")
        sign_flip = sensitivity["sign_flip"]
        bootstrap = sensitivity["participant_bootstrap"]
        if not isinstance(sign_flip, Mapping) or set(sign_flip) != {
            "draws",
            "one_sided_p_greater_than_zero",
        }:
            raise ValueError("Sign-flip sensitivity schema is not exact.")
        if not isinstance(bootstrap, Mapping) or set(bootstrap) != {
            "draws",
            "two_sided_95_percent_interval",
        }:
            raise ValueError("Bootstrap sensitivity schema is not exact.")
    return contrasts


def _validate_promotion_schema(promotion: Any) -> Mapping[str, Any]:
    if not isinstance(promotion, Mapping):
        raise TypeError("Development promotion result must be a mapping.")
    _require_exact_keys(
        dict(promotion),
        {
            "evaluated",
            "helpful_families",
            "B3_pairing_mechanism",
            "null_noninferiority",
            "adversarial_prequential_abstention",
            "severe_harm",
            "all_requirements_passed",
        },
        "development promotion",
    )
    if promotion["evaluated"] is not True or type(promotion["all_requirements_passed"]) is not bool:
        raise ValueError("Complete development promotion must be evaluated exactly once.")
    expected_nested = {
        "helpful_families": {"details", "passed_count", "required_count", "passed"},
        "B3_pairing_mechanism": {"contrast", "observed_minimum_gain", "passed"},
        "null_noninferiority": {"N1", "N4", "passed"},
        "adversarial_prequential_abstention": {
            "abstentions",
            "denominator",
            "rate",
            "minimum_rate",
            "passed",
        },
        "severe_harm": {
            "contrast_ids",
            "all_declared_safety_contrasts_exact_upper_rate_below_0.10",
            "passed",
        },
    }
    for name, keys in expected_nested.items():
        value = promotion[name]
        if not isinstance(value, Mapping):
            raise TypeError(f"Promotion {name} must be a mapping.")
        _require_exact_keys(dict(value), keys, f"promotion {name}")
    helpful = promotion["helpful_families"]["details"]
    if not isinstance(helpful, Mapping) or set(helpful) != {"B1", "B2", "B3"}:
        raise ValueError("Helpful-family promotion details are not exact.")
    for name, value in helpful.items():
        if not isinstance(value, Mapping) or set(value) != {"contrast", "passed"}:
            raise ValueError(f"Helpful-family detail {name} schema is not exact.")
    nulls = promotion["null_noninferiority"]
    for name in ("N1", "N4"):
        if not isinstance(nulls[name], Mapping) or set(nulls[name]) != {"contrasts", "passed"}:
            raise ValueError(f"Null-noninferiority detail {name} schema is not exact.")
    return promotion


def validate_development_result(
    result: Mapping[str, Any],
    *,
    contract: SyntheticContract,
    identity: GitIdentity,
    environment: Mapping[str, Any],
    preparation_receipt_sha256: str,
) -> dict[str, Any]:
    """Reject anything short of the complete frozen development outcome."""

    observed = dict(result)
    _require_exact_keys(observed, _DEVELOPMENT_RESULT_FIELDS, "development result")
    claimed_hash = observed.pop("result_sha256")
    if not isinstance(claimed_hash, str) or claimed_hash != _canonical_json_sha256(observed):
        raise ValueError("Synthetic development result self-hash is invalid.")
    expected_assertions = {name: True for name in contract.plan["hard_assertions_before_efficacy"]}
    if (
        observed["schema"] != SYNTHETIC_RESULT_SCHEMA
        or observed["candidate_id"] != "metadata-calibration-efficiency-v2"
        or observed["phase"] != "development"
        or observed["status"] != "engineering_only"
        or observed["human_claim_boundary"] != "synthetic_pass_is_necessary_but_never_sufficient"
        or observed["plan_sha256"] != contract.plan_sha256
        or observed["filterbank_sha256"] != contract.filterbank_sha256
        or observed["diagnostic_operator_configs_sha256"]
        != contract.diagnostic_operator_configs_sha256
        or observed["root_seed"] != DEVELOPMENT_ROOT_SEED
        or observed["participants_per_family"] != 64
        or observed["families"] != list(FAMILY_NAMES)
        or observed["support_budgets"] != list(SUPPORT_BUDGETS)
        or observed["query_blocks"] != list(contract.plan["population"]["immutable_query_blocks"])
        or observed["hard_assertions_completed_before_metrics"] is not True
        or observed["hard_assertions"] != expected_assertions
        or observed["development_outputs_are_engineering_only"] is not True
        or observed["lockbox_terminal"] is not False
        or observed["generator_commit"] != identity.commit
        or observed["generator_tree"] != identity.tree
        or observed["generator_source_bundle_sha256"] != identity.source_bundle_sha256
        or observed["generator_commit_timestamp_utc"] != identity.commit_timestamp_utc
        or observed["environment"] != dict(environment)
        or observed["preparation_receipt_sha256"] != preparation_receipt_sha256
        or observed["rng"]
        != {
            "bit_generator": "numpy.random.PCG64DXSM",
            "construction": "numpy.random.SeedSequence",
            "complete_key_order": list(contract.plan["rng"]["key_order"]),
        }
    ):
        raise ValueError("Synthetic development result binding or hard assertions are not exact.")

    metric_rows = _validate_complete_metric_grid(observed["participant_metrics"])
    contrasts = _validate_contrast_schema(observed["primary_contrasts"])
    promotion = _validate_promotion_schema(observed["promotion"])
    metrics = contract.plan["metrics"]["sensitivity_only"]
    context = _governed_seed_context(
        contract,
        phase="development-validation",
        root_seed=DEVELOPMENT_ROOT_SEED,
    )
    recomputed_contrasts, recomputed_promotion = summarize_synthetic_metrics(
        metric_rows,
        contract=contract,
        root_seed=DEVELOPMENT_ROOT_SEED,
        families=FAMILY_NAMES,
        sign_flip_draws=int(metrics["sign_flip"]["draws"]),
        bootstrap_draws=int(metrics["participant_bootstrap"]["draws"]),
        _governed_context=context,
    )
    if _canonical_json_bytes({"value": contrasts}) != _canonical_json_bytes(
        {"value": recomputed_contrasts}
    ) or _canonical_json_bytes({"value": promotion}) != _canonical_json_bytes(
        {"value": recomputed_promotion}
    ):
        raise ValueError("Development contrasts/promotion do not canonically recompute from rows.")
    observed["result_sha256"] = claimed_hash
    return observed


def _development_evidence(
    *,
    contract: SyntheticContract,
    identity: GitIdentity,
    environment: Mapping[str, Any],
    preparation_receipt_sha256: str,
) -> dict[str, Any]:
    path = canonical_execution_path(contract, "development_result")
    _readonly_receipt_file(path, "synthetic development result")
    result = load_json_object(path, name="synthetic development result")
    validated = validate_development_result(
        result,
        contract=contract,
        identity=identity,
        environment=environment,
        preparation_receipt_sha256=preparation_receipt_sha256,
    )
    return {
        "schema": SYNTHETIC_DEVELOPMENT_EVIDENCE_SCHEMA,
        "status": "recorded_complete_engineering_only",
        "path": str(path),
        "file_sha256": _sha256_file(path),
        "result_sha256": validated["result_sha256"],
    }


def run_development(
    contract: SyntheticContract,
    *,
    _governed_context: _GovernedSeedContext | None = None,
) -> dict[str, Any]:
    """Execute development only when the governed artifact entry point authorizes it."""

    _require_seed_access(contract, DEVELOPMENT_ROOT_SEED, _governed_context)
    if _governed_context is None or _governed_context.phase != "development":
        raise ValueError("Development requires its exact internal governed phase context.")
    identity = current_git_identity()
    _require_clean_identity(identity)
    metrics = contract.plan["metrics"]["sensitivity_only"]
    return _execute_synthetic(
        contract,
        phase="development",
        root_seed=DEVELOPMENT_ROOT_SEED,
        participants_per_family=64,
        families=FAMILY_NAMES,
        sign_flip_draws=int(metrics["sign_flip"]["draws"]),
        bootstrap_draws=int(metrics["participant_bootstrap"]["draws"]),
        fbcca_producer=produce_strict_fbcca,
        governed_context=_governed_context,
    )


def run_development_to_path(
    contract: SyntheticContract,
    *,
    output_path: str | Path,
) -> Path:
    target = _require_canonical_path(contract, "development_result", output_path)
    _validated_unused_output_path(target)
    identity = current_git_identity()
    _require_clean_identity(identity)
    environment = _validate_environment_receipt(current_environment_receipt())
    preparation, _ = _validate_preparation_receipt(
        contract,
        identity=identity,
        environment=environment,
    )
    context = _governed_seed_context(
        contract,
        phase="development",
        root_seed=DEVELOPMENT_ROOT_SEED,
    )
    result = run_development(contract, _governed_context=context)
    result.pop("result_sha256")
    result.update(
        {
            "generator_commit": identity.commit,
            "generator_tree": identity.tree,
            "generator_source_bundle_sha256": identity.source_bundle_sha256,
            "generator_commit_timestamp_utc": identity.commit_timestamp_utc,
            "environment": environment,
            "preparation_receipt_sha256": preparation["completion_receipt_sha256"],
        }
    )
    result["result_sha256"] = _canonical_json_sha256(result)
    return write_json_exclusive(target, result)


_TEST_EVIDENCE_FIELDS = {
    "schema",
    "candidate_id",
    "phase",
    "status",
    "plan_sha256",
    "argv",
    "working_directory",
    "scope",
    "exit_code",
    "stdout_path",
    "stdout_sha256",
    "stderr_path",
    "stderr_sha256",
    "generator_commit",
    "generator_tree",
    "generator_source_bundle_sha256",
    "git_worktree_clean",
    "generator_commit_timestamp_utc",
    "environment",
    "preparation_receipt_sha256",
    "development_result_sha256",
    "test_evidence_sha256",
}


def _test_transcript_paths(receipt_path: Path) -> tuple[Path, Path]:
    return (
        receipt_path.with_name(receipt_path.name + ".stdout.txt"),
        receipt_path.with_name(receipt_path.name + ".stderr.txt"),
    )


def run_full_suite_test_evidence(
    contract: SyntheticContract,
    *,
    output_path: str | Path,
) -> Path:
    """Run the actual complete suite with the exact current Python entry point."""

    target = _require_canonical_path(contract, "full_suite_test_evidence", output_path)
    _validated_unused_output_path(target)
    stdout_path, stderr_path = _test_transcript_paths(target)
    _validated_unused_output_path(stdout_path)
    _validated_unused_output_path(stderr_path)
    identity = current_git_identity()
    _require_clean_identity(identity)
    environment = _validate_environment_receipt(current_environment_receipt())
    preparation, _ = _validate_preparation_receipt(
        contract,
        identity=identity,
        environment=environment,
    )
    development = _development_evidence(
        contract=contract,
        identity=identity,
        environment=environment,
        preparation_receipt_sha256=preparation["completion_receipt_sha256"],
    )
    argv = [str(environment["python_executable"]), "-m", "pytest", "-q"]
    completed = subprocess.run(
        argv,
        cwd=_REPOSITORY,
        check=False,
        capture_output=True,
    )
    stdout = (
        completed.stdout.encode("utf-8")
        if isinstance(completed.stdout, str)
        else bytes(completed.stdout)
    )
    stderr = (
        completed.stderr.encode("utf-8")
        if isinstance(completed.stderr, str)
        else bytes(completed.stderr)
    )
    after_identity = current_git_identity()
    after_environment = _validate_environment_receipt(current_environment_receipt())
    if after_identity != identity or after_environment != environment:
        raise ValueError("Full-suite execution changed the clean source or numerical environment.")
    _write_bytes_exclusive(stdout_path, stdout)
    _write_bytes_exclusive(stderr_path, stderr)
    receipt: dict[str, Any] = {
        "schema": SYNTHETIC_TEST_EVIDENCE_SCHEMA,
        "candidate_id": "metadata-calibration-efficiency-v2",
        "phase": "test-evidence",
        "status": "passed" if completed.returncode == 0 else "failed",
        "plan_sha256": contract.plan_sha256,
        "argv": argv,
        "working_directory": str(_REPOSITORY),
        "scope": "complete_repository_test_suite",
        "exit_code": int(completed.returncode),
        "stdout_path": str(stdout_path),
        "stdout_sha256": _sha256_bytes(stdout),
        "stderr_path": str(stderr_path),
        "stderr_sha256": _sha256_bytes(stderr),
        **_identity_mapping(identity),
        "environment": environment,
        "preparation_receipt_sha256": preparation["completion_receipt_sha256"],
        "development_result_sha256": development["result_sha256"],
    }
    receipt["test_evidence_sha256"] = _canonical_json_sha256(receipt)
    return write_json_exclusive(target, receipt)


def _validate_test_evidence(
    contract: SyntheticContract,
    *,
    identity: GitIdentity,
    environment: Mapping[str, Any],
    preparation_receipt_sha256: str,
    development_result_sha256: str,
) -> dict[str, Any]:
    path = canonical_execution_path(contract, "full_suite_test_evidence")
    _readonly_receipt_file(path, "synthetic full-suite test evidence")
    receipt = load_json_object(path, name="synthetic full-suite test evidence")
    _require_exact_keys(receipt, _TEST_EVIDENCE_FIELDS, "full-suite test evidence")
    claimed = receipt.pop("test_evidence_sha256")
    if claimed != _canonical_json_sha256(receipt):
        raise ValueError("Full-suite test evidence self-hash mismatch.")
    stdout_path, stderr_path = _test_transcript_paths(path)
    stdout = _readonly_receipt_file(stdout_path, "captured pytest stdout")
    stderr = _readonly_receipt_file(stderr_path, "captured pytest stderr")
    expected = {
        "schema": SYNTHETIC_TEST_EVIDENCE_SCHEMA,
        "candidate_id": "metadata-calibration-efficiency-v2",
        "phase": "test-evidence",
        "status": "passed",
        "plan_sha256": contract.plan_sha256,
        "argv": [str(environment["python_executable"]), "-m", "pytest", "-q"],
        "working_directory": str(_REPOSITORY),
        "scope": "complete_repository_test_suite",
        "exit_code": 0,
        "stdout_path": str(stdout_path),
        "stdout_sha256": _sha256_file(stdout),
        "stderr_path": str(stderr_path),
        "stderr_sha256": _sha256_file(stderr),
        **_identity_mapping(identity),
        "environment": dict(environment),
        "preparation_receipt_sha256": preparation_receipt_sha256,
        "development_result_sha256": development_result_sha256,
    }
    if receipt != expected:
        raise ValueError("Full-suite test evidence is failed, forged, stale, or noncanonical.")
    receipt["test_evidence_sha256"] = claimed
    return receipt


_BEACON_RECEIPT_FIELDS = {
    "schema",
    "candidate_id",
    "phase",
    "status",
    "plan_sha256",
    "provider",
    "endpoint",
    "target_timestamp_utc",
    "fetched_at_utc",
    "statement_utf8",
    "statement_sha256",
    "pulse_timestamp",
    "pulse_uri",
    "pulse_version",
    "pulse_period_milliseconds",
    "pulse_cipher_suite",
    "certificate_id",
    "chain_index",
    "pulse_index",
    "local_random_value",
    "precommitment_value",
    "pulse_status_code",
    "signature_value",
    "output_value",
    "deployed_json_serialization_revision",
    "signed_message_sha512",
    "output_value_recomputed",
    "output_value_protocol_verified",
    "transport_authentication",
    "output_value_raw_sha256",
    "full_response_utf8",
    "full_response_sha256",
    "derived_seed_sha256",
    "root_seed_formula",
    "generator_commit",
    "generator_tree",
    "generator_source_bundle_sha256",
    "git_worktree_clean",
    "generator_commit_timestamp_utc",
    "environment",
    "preparation_receipt_sha256",
    "development_result_sha256",
    "test_evidence_sha256",
    "one_time_claim_sha256",
    "beacon_receipt_sha256",
}
_LOCKBOX_CLAIM_FIELDS = {
    "schema",
    "candidate_id",
    "phase",
    "status",
    "plan_sha256",
    "claim_created_at_utc",
    "future_beacon_target_utc",
    "future_beacon_statement_sha256",
    "network_access_before_claim",
    "authorization_path",
    "authorization_file_sha256",
    "authorization_receipt_sha256",
    "authorization_created_at_utc",
    "generator_commit",
    "generator_tree",
    "generator_source_bundle_sha256",
    "git_worktree_clean",
    "generator_commit_timestamp_utc",
    "environment_sha256",
    "beacon_receipt_path",
    "output_path",
    "terminal_receipt_path",
    "claim_sha256",
}


def _target_datetime(contract: SyntheticContract) -> datetime:
    text = contract.plan["rng"]["independent_lockbox_root_seed"]["target_timestamp_utc"]
    return datetime.fromisoformat(str(text).replace("Z", "+00:00"))


def _require_file_strictly_predates_target(
    path: str | Path, contract: SyntheticContract, name: str
) -> None:
    source = _readonly_receipt_file(path, name)
    target_ns = int(_target_datetime(contract).timestamp() * 1_000_000_000)
    _, observed = _safe_regular_file_stat(source, name)
    if observed.st_mtime_ns >= target_ns or observed.st_ctime_ns >= target_ns:
        raise ValueError(f"{name} filesystem timestamps do not predate the beacon target.")


def _parse_exact_utc(value: Any, name: str) -> datetime:
    if not isinstance(value, str) or not value.endswith("Z"):
        raise ValueError(f"{name} must be an exact UTC timestamp ending in Z.")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as error:
        raise ValueError(f"{name} is not a valid UTC timestamp.") from error
    if parsed.tzinfo is None or parsed.utcoffset() != timezone.utc.utcoffset(parsed):
        raise ValueError(f"{name} must resolve to UTC.")
    return parsed


def _validate_pre_network_claim(
    contract: SyntheticContract,
    *,
    claim_path: str | Path,
    authorization_path: str | Path,
    authorization: Mapping[str, Any],
    identity: GitIdentity,
    environment: Mapping[str, Any],
) -> tuple[dict[str, Any], str]:
    """Read back and validate the durable claim that precedes beacon access."""

    path = _require_canonical_path(contract, "seed_global_lockbox_claim", claim_path)
    _readonly_receipt_file(path, "canonical pre-network lockbox claim")
    claim = load_json_object(path, name="canonical pre-network lockbox claim")
    _require_exact_keys(claim, _LOCKBOX_CLAIM_FIELDS, "pre-network lockbox claim")
    claimed = claim.pop("claim_sha256")
    if not isinstance(claimed, str) or claimed != _canonical_json_sha256(claim):
        raise ValueError("Pre-network lockbox claim self-hash mismatch.")

    auth_path = _require_canonical_path(
        contract, "lockbox_authorization", authorization_path
    )
    _readonly_receipt_file(auth_path, "synthetic lockbox authorization")
    _require_file_strictly_predates_target(
        auth_path, contract, "synthetic lockbox authorization"
    )
    auth_receipt = authorization.get("authorization_receipt_sha256")
    auth_created = authorization.get("authorization_created_at_utc")
    validated_authorization_sha = _validate_consumed_authorization_binding(
        authorization,
        contract=contract,
        identity=identity,
        environment=environment,
    )
    if auth_receipt != validated_authorization_sha:
        raise ValueError("Global claim authorization digest is not a valid sealed record.")
    claim_created = _parse_exact_utc(claim["claim_created_at_utc"], "claim creation time")
    authorization_created = _parse_exact_utc(auth_created, "authorization creation time")
    target = _target_datetime(contract)
    if authorization_created >= target or claim_created < target:
        raise ValueError("Authorization/claim timestamps do not straddle the beacon target.")
    if (
        claim["schema"] != SYNTHETIC_LOCKBOX_CLAIM_SCHEMA
        or claim["candidate_id"] != "metadata-calibration-efficiency-v2"
        or claim["phase"] != "lockbox-pre-network-claim"
        or claim["status"]
        != "one_time_global_claimed_before_beacon_fetch_seed_or_rng"
        or claim["plan_sha256"] != contract.plan_sha256
        or claim["future_beacon_target_utc"]
        != contract.plan["rng"]["independent_lockbox_root_seed"]["target_timestamp_utc"]
        or claim["future_beacon_statement_sha256"]
        != contract.plan["rng"]["independent_lockbox_root_seed"]["statement_sha256"]
        or claim["network_access_before_claim"] is not False
        or claim["authorization_path"] != str(auth_path)
        or claim["authorization_file_sha256"] != _sha256_file(auth_path)
        or claim["authorization_receipt_sha256"] != auth_receipt
        or claim["authorization_created_at_utc"] != auth_created
        or claim["generator_commit"] != identity.commit
        or claim["generator_tree"] != identity.tree
        or claim["generator_source_bundle_sha256"] != identity.source_bundle_sha256
        or claim["git_worktree_clean"] is not True
        or claim["generator_commit_timestamp_utc"] != identity.commit_timestamp_utc
        or claim["environment_sha256"] != environment["environment_sha256"]
        or claim["beacon_receipt_path"]
        != str(canonical_execution_path(contract, "beacon_receipt"))
        or claim["output_path"]
        != str(canonical_execution_path(contract, "lockbox_result"))
        or claim["terminal_receipt_path"]
        != str(canonical_execution_path(contract, "lockbox_terminal_receipt"))
    ):
        raise ValueError("Pre-network lockbox claim is forged, stale, or noncanonical.")
    return claim, claimed


def _extract_beacon_pulse(full_response: Mapping[str, Any]) -> Mapping[str, Any]:
    if set(full_response) != {"pulse"} or not isinstance(full_response["pulse"], Mapping):
        raise ValueError("NIST beacon response must have the exact top-level pulse field.")
    pulse = full_response["pulse"]
    required = {
        "uri",
        "version",
        "cipherSuite",
        "period",
        "certificateId",
        "chainIndex",
        "pulseIndex",
        "timeStamp",
        "localRandomValue",
        "precommitmentValue",
        "statusCode",
        "signatureValue",
        "outputValue",
        "external",
        "listValues",
    }
    if set(pulse) != required:
        raise ValueError("NIST beacon pulse lacks the exact deployed v2 JSON field set.")
    chain_index = pulse["chainIndex"]
    pulse_index = pulse["pulseIndex"]
    if (
        pulse["version"] != "2.0"
        or pulse["cipherSuite"] != 0
        or pulse["period"] != 60000
        or pulse["statusCode"] != 0
        or type(chain_index) is not int
        or chain_index <= 0
        or type(pulse_index) is not int
        or pulse_index <= 0
    ):
        raise ValueError("NIST pulse version/period/status/index fields are invalid.")
    expected_uri = f"https://beacon.nist.gov/beacon/2.0/chain/{chain_index}/pulse/{pulse_index}"
    if pulse["uri"] != expected_uri:
        raise ValueError("NIST pulse URI does not exactly bind its chain and pulse indices.")
    certificate_id = pulse["certificateId"]
    if not isinstance(certificate_id, str) or not _LOWER_HEX_128_RE.fullmatch(certificate_id):
        raise ValueError("NIST pulse certificateId must be exact lowercase 128-hex.")
    for name in ("localRandomValue", "precommitmentValue", "outputValue"):
        if not isinstance(pulse[name], str) or not _UPPER_HEX_128_RE.fullmatch(pulse[name]):
            raise ValueError(f"NIST pulse {name} must be exact uppercase 128-hex.")
    signature = pulse["signatureValue"]
    if not isinstance(signature, str) or not _UPPER_HEX_1024_RE.fullmatch(signature):
        raise ValueError("NIST cipher-suite-0 signatureValue must be exact uppercase 1024-hex.")
    external = pulse["external"]
    if not isinstance(external, Mapping) or set(external) != {"sourceId", "statusCode", "value"}:
        raise ValueError("NIST pulse external packet must have its exact deployed fields.")
    if type(external["statusCode"]) is not int or external["statusCode"] < 0:
        raise ValueError("NIST external statusCode must be a nonnegative exact integer.")
    for name in ("sourceId", "value"):
        if not isinstance(external[name], str) or not _UPPER_HEX_128_RE.fullmatch(external[name]):
            raise ValueError(f"NIST external {name} must be exact uppercase 128-hex.")
    list_values = pulse["listValues"]
    if not isinstance(list_values, list) or len(list_values) != 5:
        raise ValueError("NIST pulse must contain exactly five skip-list values.")
    expected_types = {"previous", "hour", "day", "month", "year"}
    observed_types: set[str] = set()
    for item in list_values:
        if not isinstance(item, Mapping) or set(item) != {"uri", "type", "value"}:
            raise ValueError("Every NIST skip-list item must have exact uri/type/value fields.")
        item_type = item["type"]
        if not isinstance(item_type, str) or item_type not in expected_types:
            raise ValueError("NIST skip-list type is invalid.")
        if item_type in observed_types:
            raise ValueError("NIST skip-list types must be unique.")
        observed_types.add(item_type)
        if (
            not isinstance(item["uri"], str)
            or not item["uri"].startswith(
                f"https://beacon.nist.gov/beacon/2.0/chain/{chain_index}/pulse/"
            )
            or not isinstance(item["value"], str)
            or not _UPPER_HEX_128_RE.fullmatch(item["value"])
        ):
            raise ValueError("NIST skip-list URI or value is invalid.")
    if observed_types != expected_types:
        raise ValueError("NIST skip-list type set is incomplete.")
    return pulse


def _uint_bytes(value: Any, width: int, name: str) -> bytes:
    if type(value) is not int or value < 0 or value >= 1 << (8 * width):
        raise ValueError(f"{name} does not fit its exact unsigned integer width.")
    return value.to_bytes(width, byteorder="big", signed=False)


def _deployed_length_prefixed_bytes(value: bytes) -> bytes:
    return _uint_bytes(len(value), 4, "deployed JSON field length") + value


def _deployed_length_prefixed_string(value: Any, name: str) -> bytes:
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a string.")
    return _deployed_length_prefixed_bytes(value.encode("utf-8"))


def _deployed_length_prefixed_hex(value: Any, name: str) -> bytes:
    if not isinstance(value, str) or len(value) % 2 or re.fullmatch(r"[0-9A-Fa-f]+", value) is None:
        raise ValueError(f"{name} must be nonempty hexadecimal.")
    return _deployed_length_prefixed_bytes(bytes.fromhex(value))


def _serialize_deployed_nist_signed_message(pulse: Mapping[str, Any]) -> bytes:
    """Serialize fields 1--19 as deployed by the NIST v2 JSON service."""

    list_by_type = {str(item["type"]): item["value"] for item in pulse["listValues"]}
    external = pulse["external"]
    return b"".join(
        (
            _deployed_length_prefixed_string(pulse["uri"], "pulse uri"),
            _deployed_length_prefixed_string(pulse["version"], "pulse version"),
            _uint_bytes(pulse["cipherSuite"], 4, "cipherSuite"),
            _uint_bytes(pulse["period"], 4, "period"),
            _deployed_length_prefixed_hex(pulse["certificateId"], "certificateId"),
            _uint_bytes(pulse["chainIndex"], 8, "chainIndex"),
            _uint_bytes(pulse["pulseIndex"], 8, "pulseIndex"),
            _deployed_length_prefixed_string(pulse["timeStamp"], "timeStamp"),
            _deployed_length_prefixed_hex(pulse["localRandomValue"], "localRandomValue"),
            _deployed_length_prefixed_hex(external["sourceId"], "external.sourceId"),
            _uint_bytes(external["statusCode"], 4, "external.statusCode"),
            _deployed_length_prefixed_hex(external["value"], "external.value"),
            *(
                _deployed_length_prefixed_hex(list_by_type[name], f"listValues.{name}")
                for name in ("previous", "hour", "day", "month", "year")
            ),
            _deployed_length_prefixed_hex(
                pulse["precommitmentValue"], "precommitmentValue"
            ),
            _uint_bytes(pulse["statusCode"], 4, "statusCode"),
        )
    )


def _verify_deployed_nist_output_value(pulse: Mapping[str, Any]) -> tuple[str, str]:
    signed_message = _serialize_deployed_nist_signed_message(pulse)
    signature = bytes.fromhex(str(pulse["signatureValue"]))
    recomputed = hashlib.sha512(signed_message + signature).hexdigest().upper()
    if recomputed != pulse["outputValue"]:
        raise ValueError("NIST outputValue does not match its deployed protocol SHA-512 preimage.")
    return hashlib.sha512(signed_message).hexdigest(), recomputed


def _derive_seed_material(
    contract: SyntheticContract,
    *,
    pulse_timestamp: Any,
    output_value: Any,
) -> tuple[int, str, str]:
    beacon = contract.plan["rng"]["independent_lockbox_root_seed"]
    if pulse_timestamp != beacon["required_exact_pulse_timestamp"]:
        raise ValueError("NIST pulse timestamp does not equal the precommitted target.")
    if not isinstance(output_value, str) or not _UPPER_HEX_128_RE.fullmatch(output_value):
        raise ValueError("NIST outputValue must be exactly 128 uppercase hexadecimal characters.")
    statement = str(beacon["statement_utf8"])
    if _sha256_bytes(statement.encode("utf-8")) != beacon["statement_sha256"]:
        raise ValueError("Frozen NIST derivation statement hash mismatch.")
    raw_output = bytes.fromhex(output_value)
    if len(raw_output) != 64:
        raise ValueError("NIST outputValue must decode to exactly 64 bytes.")
    digest = hashlib.sha256(statement.encode("utf-8") + b"\n" + raw_output).digest()
    return (
        int.from_bytes(digest, byteorder="big", signed=False),
        digest.hex(),
        _sha256_bytes(raw_output),
    )


def _derive_lockbox_seed_from_beacon_receipt(
    contract: SyntheticContract,
    receipt_path: str | Path,
) -> int:
    path = _require_canonical_path(contract, "beacon_receipt", receipt_path)
    _readonly_receipt_file(path, "canonical NIST beacon receipt")
    receipt = load_json_object(path, name="canonical NIST beacon receipt")
    _require_exact_keys(receipt, _BEACON_RECEIPT_FIELDS, "NIST beacon receipt")
    claimed = receipt.pop("beacon_receipt_sha256")
    if not isinstance(claimed, str) or claimed != _canonical_json_sha256(receipt):
        raise ValueError("NIST beacon receipt self-hash mismatch.")
    authorization_path = canonical_execution_path(contract, "lockbox_authorization")
    authorization = load_json_object(
        authorization_path, name="synthetic lockbox authorization"
    )
    receipt_identity = GitIdentity(
        commit=str(receipt["generator_commit"]),
        tree=str(receipt["generator_tree"]),
        source_bundle_sha256=str(receipt["generator_source_bundle_sha256"]),
        clean=receipt["git_worktree_clean"] is True,
        commit_timestamp_utc=str(receipt["generator_commit_timestamp_utc"]),
    )
    receipt_environment = receipt["environment"]
    if not isinstance(receipt_environment, Mapping):
        raise TypeError("NIST beacon receipt environment must be a mapping.")
    claim, claim_digest = _validate_pre_network_claim(
        contract,
        claim_path=canonical_execution_path(contract, "seed_global_lockbox_claim"),
        authorization_path=authorization_path,
        authorization=authorization,
        identity=receipt_identity,
        environment=receipt_environment,
    )
    if (
        claim["beacon_receipt_path"] != str(path)
        or claim["output_path"] != str(canonical_execution_path(contract, "lockbox_result"))
        or receipt["one_time_claim_sha256"] != claim_digest
    ):
        raise ValueError("NIST beacon receipt is not bound to the canonical pre-network claim.")
    raw_response = receipt["full_response_utf8"]
    if (
        not isinstance(raw_response, str)
        or _sha256_bytes(raw_response.encode("utf-8")) != receipt["full_response_sha256"]
    ):
        raise ValueError("NIST full response hash mismatch.")
    decoded = json.loads(raw_response)
    if not isinstance(decoded, Mapping):
        raise TypeError("NIST full response must decode to one object.")
    pulse = _extract_beacon_pulse(decoded)
    if (
        pulse["timeStamp"] != receipt["pulse_timestamp"]
        or pulse["outputValue"] != receipt["output_value"]
        or pulse["uri"] != receipt["pulse_uri"]
        or pulse["version"] != receipt["pulse_version"]
        or pulse["period"] != receipt["pulse_period_milliseconds"]
        or pulse["cipherSuite"] != receipt["pulse_cipher_suite"]
        or pulse["certificateId"] != receipt["certificate_id"]
        or pulse["chainIndex"] != receipt["chain_index"]
        or pulse["pulseIndex"] != receipt["pulse_index"]
        or pulse["localRandomValue"] != receipt["local_random_value"]
        or pulse["precommitmentValue"] != receipt["precommitment_value"]
        or pulse["statusCode"] != receipt["pulse_status_code"]
        or pulse["signatureValue"] != receipt["signature_value"]
    ):
        raise ValueError("NIST receipt pulse fields differ from its full response.")
    root_seed, digest, raw_sha = _derive_seed_material(
        contract,
        pulse_timestamp=pulse["timeStamp"],
        output_value=pulse["outputValue"],
    )
    signed_message_sha512, recomputed_output = _verify_deployed_nist_output_value(pulse)
    beacon = contract.plan["rng"]["independent_lockbox_root_seed"]
    fetched_at = receipt["fetched_at_utc"]
    fetched_time = _parse_exact_utc(fetched_at, "beacon fetch time")
    claim_time = _parse_exact_utc(claim["claim_created_at_utc"], "claim creation time")
    if fetched_time < _target_datetime(contract) or fetched_time < claim_time:
        raise ValueError("NIST beacon receipt claims retrieval before the frozen target.")
    claim_path = _readonly_receipt_file(
        canonical_execution_path(contract, "seed_global_lockbox_claim"),
        "canonical pre-network lockbox claim",
    )
    _, claim_stat = _safe_regular_file_stat(
        claim_path, "canonical pre-network lockbox claim"
    )
    _, beacon_stat = _safe_regular_file_stat(path, "canonical NIST beacon receipt")
    if claim_stat.st_ctime_ns > beacon_stat.st_ctime_ns:
        raise ValueError("Beacon receipt filesystem creation predates the global claim.")
    if (
        receipt["schema"] != SYNTHETIC_BEACON_RECEIPT_SCHEMA
        or receipt["candidate_id"] != "metadata-calibration-efficiency-v2"
        or receipt["phase"] != "lockbox-beacon-after-claim"
        or receipt["status"] != "exact_future_pulse_recorded"
        or receipt["plan_sha256"] != contract.plan_sha256
        or receipt["provider"] != beacon["provider"]
        or receipt["endpoint"] != beacon["exact_endpoint"]
        or receipt["target_timestamp_utc"] != beacon["target_timestamp_utc"]
        or receipt["statement_utf8"] != beacon["statement_utf8"]
        or receipt["statement_sha256"] != beacon["statement_sha256"]
        or receipt["deployed_json_serialization_revision"]
        != beacon["deployed_json_serialization_revision"]
        or receipt["signed_message_sha512"] != signed_message_sha512
        or receipt["output_value_recomputed"] != recomputed_output
        or receipt["output_value_protocol_verified"] is not True
        or receipt["transport_authentication"] != beacon["transport_authentication"]
        or receipt["output_value_raw_sha256"] != raw_sha
        or receipt["derived_seed_sha256"] != digest
        or receipt["root_seed_formula"] != beacon["root_seed_formula"]
        or not isinstance(receipt["one_time_claim_sha256"], str)
        or not _SHA256_RE.fullmatch(receipt["one_time_claim_sha256"])
    ):
        raise ValueError("NIST beacon receipt is not the exact frozen derivation.")
    return root_seed


def fetch_nist_beacon_receipt(
    contract: SyntheticContract,
    *,
    output_path: str | Path,
) -> Path:
    """Standalone seed-bearing beacon retrieval is forbidden by the V11 contract."""

    _require_canonical_path(contract, "beacon_receipt", output_path)
    raise RuntimeError(
        "V11 forbids standalone beacon fetch; use the one-time lockbox transaction."
    )


def _fetch_nist_beacon_after_global_claim(
    contract: SyntheticContract,
    *,
    identity: GitIdentity,
    environment: Mapping[str, Any],
    preparation_receipt_sha256: str,
    development_result_sha256: str,
    test_evidence_sha256: str,
    one_time_claim_sha256: str,
) -> tuple[Path, dict[str, Any], int]:
    """Fetch and persist the future pulse only after the global claim exists."""

    target = canonical_execution_path(contract, "beacon_receipt")
    _validated_unused_output_path(target)
    authorization_path = canonical_execution_path(contract, "lockbox_authorization")
    authorization = load_json_object(
        authorization_path, name="synthetic lockbox authorization"
    )
    _, validated_claim_sha = _validate_pre_network_claim(
        contract,
        claim_path=canonical_execution_path(contract, "seed_global_lockbox_claim"),
        authorization_path=authorization_path,
        authorization=authorization,
        identity=identity,
        environment=environment,
    )
    if (
        validated_claim_sha != one_time_claim_sha256
        or authorization["preparation_evidence"]["receipt_sha256"]
        != preparation_receipt_sha256
        or authorization["development_evidence"]["receipt_sha256"]
        != development_result_sha256
        or authorization["test_evidence"]["receipt_sha256"] != test_evidence_sha256
    ):
        raise ValueError("Beacon fetch is not bound to the durable global claim and evidence.")
    observed_now = datetime.now(timezone.utc)
    if observed_now < _target_datetime(contract):
        raise ValueError("The exact NIST pulse cannot be fetched before its target timestamp.")
    beacon = contract.plan["rng"]["independent_lockbox_root_seed"]
    endpoint = str(beacon["exact_endpoint"])
    if not endpoint.startswith("https://beacon.nist.gov/"):
        raise ValueError("Frozen beacon endpoint must use exact NIST HTTPS host.")
    request = urllib.request.Request(endpoint, headers={"Accept": "application/json"})
    opener = urllib.request.build_opener(_RejectHTTPRedirect())
    response = opener.open(request, timeout=30)
    try:
        response_url = response.geturl() if hasattr(response, "geturl") else endpoint
        status_code = getattr(response, "status", 200)
        response_bytes = response.read()
    finally:
        close = getattr(response, "close", None)
        if callable(close):
            close()
    if response_url != endpoint or status_code != 200:
        raise ValueError("NIST beacon retrieval redirected or did not return HTTP 200.")
    try:
        response_text = response_bytes.decode("utf-8")
        decoded = json.loads(response_text)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("NIST beacon response is not valid UTF-8 JSON.") from error
    if not isinstance(decoded, Mapping):
        raise TypeError("NIST beacon response must decode to one mapping.")
    pulse = _extract_beacon_pulse(decoded)
    root_seed, derived_digest, raw_sha = _derive_seed_material(
        contract,
        pulse_timestamp=pulse["timeStamp"],
        output_value=pulse["outputValue"],
    )
    signed_message_sha512, recomputed_output = _verify_deployed_nist_output_value(pulse)
    _DYNAMIC_RESERVED_SEEDS.add(root_seed)
    receipt: dict[str, Any] = {
        "schema": SYNTHETIC_BEACON_RECEIPT_SCHEMA,
        "candidate_id": "metadata-calibration-efficiency-v2",
        "phase": "lockbox-beacon-after-claim",
        "status": "exact_future_pulse_recorded",
        "plan_sha256": contract.plan_sha256,
        "provider": beacon["provider"],
        "endpoint": endpoint,
        "target_timestamp_utc": beacon["target_timestamp_utc"],
        "fetched_at_utc": datetime.now(timezone.utc)
        .isoformat(timespec="milliseconds")
        .replace("+00:00", "Z"),
        "statement_utf8": beacon["statement_utf8"],
        "statement_sha256": beacon["statement_sha256"],
        "pulse_timestamp": pulse["timeStamp"],
        "pulse_uri": pulse["uri"],
        "pulse_version": pulse["version"],
        "pulse_period_milliseconds": pulse["period"],
        "pulse_cipher_suite": pulse["cipherSuite"],
        "certificate_id": pulse["certificateId"],
        "chain_index": pulse["chainIndex"],
        "pulse_index": pulse["pulseIndex"],
        "local_random_value": pulse["localRandomValue"],
        "precommitment_value": pulse["precommitmentValue"],
        "pulse_status_code": pulse["statusCode"],
        "signature_value": pulse["signatureValue"],
        "output_value": pulse["outputValue"],
        "deployed_json_serialization_revision": beacon[
            "deployed_json_serialization_revision"
        ],
        "signed_message_sha512": signed_message_sha512,
        "output_value_recomputed": recomputed_output,
        "output_value_protocol_verified": True,
        "transport_authentication": beacon["transport_authentication"],
        "output_value_raw_sha256": raw_sha,
        "full_response_utf8": response_text,
        "full_response_sha256": _sha256_bytes(response_bytes),
        "derived_seed_sha256": derived_digest,
        "root_seed_formula": beacon["root_seed_formula"],
        **_identity_mapping(identity),
        "environment": environment,
        "preparation_receipt_sha256": preparation_receipt_sha256,
        "development_result_sha256": development_result_sha256,
        "test_evidence_sha256": test_evidence_sha256,
        "one_time_claim_sha256": one_time_claim_sha256,
    }
    receipt["beacon_receipt_sha256"] = _canonical_json_sha256(receipt)
    written = write_json_exclusive(target, receipt)
    return written, receipt, root_seed


def _validate_beacon_evidence(
    contract: SyntheticContract,
    *,
    identity: GitIdentity,
    environment: Mapping[str, Any],
    preparation_receipt_sha256: str,
    development_result_sha256: str,
    test_evidence_sha256: str,
    one_time_claim_sha256: str,
) -> tuple[dict[str, Any], int]:
    path = canonical_execution_path(contract, "beacon_receipt")
    root_seed = _derive_lockbox_seed_from_beacon_receipt(contract, path)
    receipt = load_json_object(path, name="canonical NIST beacon receipt")
    if (
        receipt["generator_commit"] != identity.commit
        or receipt["generator_tree"] != identity.tree
        or receipt["generator_source_bundle_sha256"] != identity.source_bundle_sha256
        or receipt["generator_commit_timestamp_utc"] != identity.commit_timestamp_utc
        or receipt["git_worktree_clean"] is not True
        or receipt["environment"] != dict(environment)
        or receipt["preparation_receipt_sha256"] != preparation_receipt_sha256
        or receipt["development_result_sha256"] != development_result_sha256
        or receipt["test_evidence_sha256"] != test_evidence_sha256
        or receipt["one_time_claim_sha256"] != one_time_claim_sha256
    ):
        raise ValueError("NIST beacon receipt source/environment/evidence binding drifted.")
    _DYNAMIC_RESERVED_SEEDS.add(root_seed)
    return receipt, root_seed


_EVIDENCE_REFERENCE_FIELDS = {"path", "file_sha256", "receipt_sha256"}
_AUTHORIZATION_FIELDS = {
    "schema",
    "candidate_id",
    "phase",
    "status",
    "authorized",
    "authorized_by",
    "authorization_basis",
    "authorization_created_at_utc",
    "plan_sha256",
    "filterbank_sha256",
    "diagnostic_operator_configs_sha256",
    "generator_commit",
    "generator_tree",
    "generator_source_bundle_sha256",
    "git_worktree_clean",
    "generator_commit_timestamp_utc",
    "environment",
    "preparation_evidence",
    "development_evidence",
    "test_evidence",
    "future_beacon_commitment",
    "unused_one_time_output",
    "one_time_nonce_sha256",
    "authorization_receipt_sha256",
}
_OUTPUT_BINDING_FIELDS = {
    "path",
    "terminal_receipt_path",
    "beacon_receipt_path",
    "seed_global_claim_path",
    "observed_output_absent",
    "observed_terminal_receipt_absent",
    "observed_beacon_absent",
    "observed_global_claim_absent",
    "never_overwrite",
}
_FUTURE_BEACON_COMMITMENT_FIELDS = {
    "provider",
    "target_timestamp_utc",
    "target_timestamp_unix_milliseconds",
    "exact_endpoint",
    "statement_utf8",
    "statement_sha256",
    "digest_formula",
    "root_seed_formula",
    "standalone_beacon_fetch_phase",
    "transaction_order",
}


def _evidence_reference(path: Path, receipt_sha256: str) -> dict[str, str]:
    return {
        "path": str(path),
        "file_sha256": _sha256_file(path),
        "receipt_sha256": receipt_sha256,
    }


def _prelockbox_authorization_evidence(
    contract: SyntheticContract,
    identity: GitIdentity,
    environment: Mapping[str, Any],
) -> dict[str, dict[str, str]]:
    preparation, preparation_path = _validate_preparation_receipt(
        contract,
        identity=identity,
        environment=environment,
    )
    development_path = canonical_execution_path(contract, "development_result")
    _require_file_strictly_predates_target(
        development_path, contract, "synthetic development result"
    )
    development = _development_evidence(
        contract=contract,
        identity=identity,
        environment=environment,
        preparation_receipt_sha256=preparation["completion_receipt_sha256"],
    )
    test_path = canonical_execution_path(contract, "full_suite_test_evidence")
    _require_file_strictly_predates_target(
        test_path, contract, "synthetic full-suite test evidence"
    )
    for transcript, name in zip(
        _test_transcript_paths(test_path),
        ("captured pytest stdout", "captured pytest stderr"),
    ):
        _require_file_strictly_predates_target(transcript, contract, name)
    tests = _validate_test_evidence(
        contract,
        identity=identity,
        environment=environment,
        preparation_receipt_sha256=preparation["completion_receipt_sha256"],
        development_result_sha256=development["result_sha256"],
    )
    references = {
        "preparation_evidence": _evidence_reference(
            preparation_path, preparation["completion_receipt_sha256"]
        ),
        "development_evidence": _evidence_reference(
            development_path,
            development["result_sha256"],
        ),
        "test_evidence": _evidence_reference(
            test_path,
            tests["test_evidence_sha256"],
        ),
    }
    return references


def _future_beacon_commitment(contract: SyntheticContract) -> dict[str, Any]:
    beacon = contract.plan["rng"]["independent_lockbox_root_seed"]
    return {
        name: beacon[name]
        for name in (
            "provider",
            "target_timestamp_utc",
            "target_timestamp_unix_milliseconds",
            "exact_endpoint",
            "statement_utf8",
            "statement_sha256",
            "digest_formula",
            "root_seed_formula",
            "standalone_beacon_fetch_phase",
            "transaction_order",
        )
    }


def _validate_consumed_authorization_binding(
    authorization: Mapping[str, Any],
    *,
    contract: SyntheticContract,
    identity: GitIdentity,
    environment: Mapping[str, Any],
) -> str:
    """Validate a sealed authorization after its bound outputs have been claimed."""

    record = dict(authorization)
    _require_exact_keys(record, _AUTHORIZATION_FIELDS, "lockbox authorization")
    claimed = record.pop("authorization_receipt_sha256")
    if (
        not isinstance(claimed, str)
        or not _SHA256_RE.fullmatch(claimed)
        or claimed != _canonical_json_sha256(record)
    ):
        raise ValueError("Lockbox authorization self-hash mismatch.")
    created = _parse_exact_utc(
        record["authorization_created_at_utc"], "authorization creation time"
    )
    if (
        record["schema"] != SYNTHETIC_LOCKBOX_AUTHORIZATION_SCHEMA
        or record["candidate_id"] != "metadata-calibration-efficiency-v2"
        or record["phase"] != "lockbox"
        or record["status"] != "authorized_for_one_time_execution"
        or record["authorized"] is not True
        or not isinstance(record["authorized_by"], str)
        or not record["authorized_by"].strip()
        or not isinstance(record["authorization_basis"], str)
        or not record["authorization_basis"].strip()
        or not isinstance(record["one_time_nonce_sha256"], str)
        or not _SHA256_RE.fullmatch(record["one_time_nonce_sha256"])
        or created >= _target_datetime(contract)
        or record["plan_sha256"] != contract.plan_sha256
        or record["filterbank_sha256"] != contract.filterbank_sha256
        or record["diagnostic_operator_configs_sha256"]
        != contract.diagnostic_operator_configs_sha256
        or record["generator_commit"] != identity.commit
        or record["generator_tree"] != identity.tree
        or record["generator_source_bundle_sha256"] != identity.source_bundle_sha256
        or record["generator_commit_timestamp_utc"] != identity.commit_timestamp_utc
        or record["git_worktree_clean"] is not True
        or record["environment"] != dict(environment)
    ):
        raise ValueError("Lockbox authorization source, scope, or time binding is invalid.")
    references = _prelockbox_authorization_evidence(contract, identity, environment)
    for name, expected in references.items():
        observed = record[name]
        if not isinstance(observed, Mapping):
            raise TypeError(f"Authorization {name} must be a mapping.")
        _require_exact_keys(dict(observed), _EVIDENCE_REFERENCE_FIELDS, name)
        if dict(observed) != expected:
            raise ValueError(f"Authorization {name} is forged, stale, or noncanonical.")
    future = record["future_beacon_commitment"]
    if not isinstance(future, Mapping):
        raise TypeError("Authorization future-beacon commitment must be a mapping.")
    _require_exact_keys(
        dict(future), _FUTURE_BEACON_COMMITMENT_FIELDS, "future-beacon commitment"
    )
    if dict(future) != _future_beacon_commitment(contract):
        raise ValueError("Authorization future-beacon commitment drifted.")
    target = canonical_execution_path(contract, "lockbox_result")
    terminal_path = canonical_execution_path(contract, "lockbox_terminal_receipt")
    claim_path = canonical_execution_path(contract, "seed_global_lockbox_claim")
    beacon_path = canonical_execution_path(contract, "beacon_receipt")
    output_binding = record["unused_one_time_output"]
    if not isinstance(output_binding, Mapping):
        raise TypeError("Authorization unused-output binding must be a mapping.")
    _require_exact_keys(dict(output_binding), _OUTPUT_BINDING_FIELDS, "unused lockbox output")
    if dict(output_binding) != {
        "path": str(target),
        "terminal_receipt_path": str(terminal_path),
        "beacon_receipt_path": str(beacon_path),
        "seed_global_claim_path": str(claim_path),
        "observed_output_absent": True,
        "observed_terminal_receipt_absent": True,
        "observed_beacon_absent": True,
        "observed_global_claim_absent": True,
        "never_overwrite": True,
    }:
        raise ValueError("Authorization does not bind the canonical one-time outputs.")
    return claimed


def build_lockbox_authorization_template(
    contract: SyntheticContract,
    *,
    output_path: str | Path,
) -> dict[str, Any]:
    """Build a pre-target record binding all evidence and the future pulse commitment."""

    target = _require_canonical_path(contract, "lockbox_result", output_path)
    _validated_unused_output_path(target)
    terminal_path = canonical_execution_path(contract, "lockbox_terminal_receipt")
    _validated_unused_output_path(terminal_path)
    claim_path = canonical_execution_path(contract, "seed_global_lockbox_claim")
    _validated_unused_output_path(claim_path)
    beacon_path = canonical_execution_path(contract, "beacon_receipt")
    _validated_unused_output_path(beacon_path)
    created = datetime.now(timezone.utc)
    if created >= _target_datetime(contract):
        raise ValueError("Lockbox authorization must be created before the beacon target.")
    identity = current_git_identity()
    _require_clean_identity(identity)
    environment = _validate_environment_receipt(current_environment_receipt())
    references = _prelockbox_authorization_evidence(contract, identity, environment)
    record: dict[str, Any] = {
        "schema": SYNTHETIC_LOCKBOX_AUTHORIZATION_SCHEMA,
        "candidate_id": "metadata-calibration-efficiency-v2",
        "phase": "lockbox",
        "status": "authorization_required",
        "authorized": False,
        "authorized_by": "",
        "authorization_basis": "",
        "authorization_created_at_utc": created.isoformat(timespec="milliseconds").replace(
            "+00:00", "Z"
        ),
        "plan_sha256": contract.plan_sha256,
        "filterbank_sha256": contract.filterbank_sha256,
        "diagnostic_operator_configs_sha256": contract.diagnostic_operator_configs_sha256,
        **_identity_mapping(identity),
        "environment": environment,
        **references,
        "future_beacon_commitment": _future_beacon_commitment(contract),
        "unused_one_time_output": {
            "path": str(target),
            "terminal_receipt_path": str(terminal_path),
            "beacon_receipt_path": str(beacon_path),
            "seed_global_claim_path": str(claim_path),
            "observed_output_absent": True,
            "observed_terminal_receipt_absent": True,
            "observed_beacon_absent": True,
            "observed_global_claim_absent": True,
            "never_overwrite": True,
        },
        "one_time_nonce_sha256": None,
    }
    record["authorization_receipt_sha256"] = _canonical_json_sha256(record)
    return record


def seal_authorization_record(record: Mapping[str, Any]) -> dict[str, Any]:
    """Self-hash a separately authorized record; hashing alone grants no authority."""

    sealed = dict(record)
    sealed.pop("authorization_receipt_sha256", None)
    sealed["authorization_receipt_sha256"] = _canonical_json_sha256(sealed)
    return sealed


def validate_lockbox_authorization(
    authorization: Mapping[str, Any],
    *,
    contract: SyntheticContract,
    output_path: str | Path,
) -> str:
    identity = current_git_identity()
    _require_clean_identity(identity)
    environment = _validate_environment_receipt(current_environment_receipt())
    claimed = _validate_consumed_authorization_binding(
        authorization,
        contract=contract,
        identity=identity,
        environment=environment,
    )
    target = _require_canonical_path(contract, "lockbox_result", output_path)
    _validated_unused_output_path(target)
    terminal_path = canonical_execution_path(contract, "lockbox_terminal_receipt")
    _validated_unused_output_path(terminal_path)
    claim_path = canonical_execution_path(contract, "seed_global_lockbox_claim")
    _validated_unused_output_path(claim_path)
    beacon_path = canonical_execution_path(contract, "beacon_receipt")
    _validated_unused_output_path(beacon_path)
    return claimed


def _terminal_receipt(
    *,
    status: str,
    contract: SyntheticContract,
    identity: GitIdentity,
    environment: Mapping[str, Any],
    authorization_sha256: str,
    beacon_receipt_sha256: str | None,
    claim_sha256: str,
    error: BaseException,
) -> dict[str, Any]:
    receipt: dict[str, Any] = {
        "schema": SYNTHETIC_TERMINAL_RECEIPT_SCHEMA,
        "candidate_id": "metadata-calibration-efficiency-v2",
        "phase": "lockbox",
        "status": status,
        "human_claim_boundary": "synthetic_pass_is_necessary_but_never_sufficient",
        "plan_sha256": contract.plan_sha256,
        "generator_commit": identity.commit,
        "generator_tree": identity.tree,
        "generator_source_bundle_sha256": identity.source_bundle_sha256,
        "generator_commit_timestamp_utc": identity.commit_timestamp_utc,
        "environment": dict(environment),
        "authorization_receipt_sha256": authorization_sha256,
        "beacon_receipt_sha256": beacon_receipt_sha256,
        "one_time_claim_sha256": claim_sha256,
        "error_type": type(error).__name__,
        "error_message": str(error),
        "automatic_retry": "forbidden",
    }
    receipt["result_sha256"] = _canonical_json_sha256(receipt)
    return receipt


def run_lockbox_to_path(
    contract: SyntheticContract,
    *,
    authorization_path: str | Path,
    output_path: str | Path,
) -> Path:
    """Consume the global claim before network access, seed derivation, or RNG."""

    target = _require_canonical_path(contract, "lockbox_result", output_path)
    auth_path = _require_canonical_path(contract, "lockbox_authorization", authorization_path)
    claim_path = canonical_execution_path(contract, "seed_global_lockbox_claim")
    beacon_path = canonical_execution_path(contract, "beacon_receipt")
    terminal_path = canonical_execution_path(contract, "lockbox_terminal_receipt")
    # A global claim consumes the seed even if a prior process never wrote a terminal result.
    _validated_unused_output_path(claim_path)
    _validated_unused_output_path(beacon_path)
    _validated_unused_output_path(target)
    _validated_unused_output_path(terminal_path)
    _readonly_receipt_file(auth_path, "synthetic lockbox authorization")
    _require_file_strictly_predates_target(
        auth_path, contract, "synthetic lockbox authorization"
    )
    if datetime.now(timezone.utc) < _target_datetime(contract):
        raise ValueError("The one-time transaction cannot start before the beacon target.")
    authorization = load_json_object(auth_path, name="synthetic lockbox authorization")
    identity = current_git_identity()
    _require_clean_identity(identity)
    environment = _validate_environment_receipt(current_environment_receipt())
    authorization_sha = validate_lockbox_authorization(
        authorization,
        contract=contract,
        output_path=target,
    )
    if (
        current_git_identity() != identity
        or _validate_environment_receipt(current_environment_receipt()) != environment
    ):
        raise ValueError("Source or numerical environment changed during preclaim validation.")
    # The claim is O_EXCL before the first network byte or seed-bearing value is obtained.
    beacon_commitment = authorization["future_beacon_commitment"]
    claim: dict[str, Any] = {
        "schema": SYNTHETIC_LOCKBOX_CLAIM_SCHEMA,
        "candidate_id": "metadata-calibration-efficiency-v2",
        "phase": "lockbox-pre-network-claim",
        "status": "one_time_global_claimed_before_beacon_fetch_seed_or_rng",
        "plan_sha256": contract.plan_sha256,
        "claim_created_at_utc": datetime.now(timezone.utc)
        .isoformat(timespec="milliseconds")
        .replace("+00:00", "Z"),
        "future_beacon_target_utc": beacon_commitment["target_timestamp_utc"],
        "future_beacon_statement_sha256": beacon_commitment["statement_sha256"],
        "network_access_before_claim": False,
        "authorization_path": str(auth_path),
        "authorization_file_sha256": _sha256_file(auth_path),
        "authorization_receipt_sha256": authorization_sha,
        "authorization_created_at_utc": authorization["authorization_created_at_utc"],
        "generator_commit": identity.commit,
        "generator_tree": identity.tree,
        "generator_source_bundle_sha256": identity.source_bundle_sha256,
        "git_worktree_clean": identity.clean,
        "generator_commit_timestamp_utc": identity.commit_timestamp_utc,
        "environment_sha256": environment["environment_sha256"],
        "beacon_receipt_path": str(beacon_path),
        "output_path": str(target),
        "terminal_receipt_path": str(terminal_path),
    }
    claim["claim_sha256"] = _canonical_json_sha256(claim)
    try:
        write_json_exclusive(claim_path, claim)
    except FileExistsError:
        raise
    except Exception as error:
        if not (claim_path.exists() or claim_path.is_symlink()):
            raise
        terminal = _terminal_receipt(
            status="consumed_inconclusive_claim_write_error",
            contract=contract,
            identity=identity,
            environment=environment,
            authorization_sha256=authorization_sha,
            beacon_receipt_sha256=None,
            claim_sha256=claim["claim_sha256"],
            error=error,
        )
        return write_json_exclusive(terminal_path, terminal)

    metrics = contract.plan["metrics"]["sensitivity_only"]
    beacon_receipt_sha256: str | None = None
    try:
        _, validated_claim_sha = _validate_pre_network_claim(
            contract,
            claim_path=claim_path,
            authorization_path=auth_path,
            authorization=authorization,
            identity=identity,
            environment=environment,
        )
        if validated_claim_sha != claim["claim_sha256"]:
            raise ValueError("Read-back global claim digest differs before beacon access.")
        _, fetched_beacon, _ = _fetch_nist_beacon_after_global_claim(
            contract,
            identity=identity,
            environment=environment,
            preparation_receipt_sha256=authorization["preparation_evidence"][
                "receipt_sha256"
            ],
            development_result_sha256=authorization["development_evidence"][
                "receipt_sha256"
            ],
            test_evidence_sha256=authorization["test_evidence"]["receipt_sha256"],
            one_time_claim_sha256=claim["claim_sha256"],
        )
        beacon_receipt_sha256 = fetched_beacon["beacon_receipt_sha256"]
        beacon, root_seed = _validate_beacon_evidence(
            contract,
            identity=identity,
            environment=environment,
            preparation_receipt_sha256=authorization["preparation_evidence"][
                "receipt_sha256"
            ],
            development_result_sha256=authorization["development_evidence"][
                "receipt_sha256"
            ],
            test_evidence_sha256=authorization["test_evidence"]["receipt_sha256"],
            one_time_claim_sha256=claim["claim_sha256"],
        )
        context = _governed_seed_context(contract, phase="lockbox", root_seed=root_seed)
        result = _execute_synthetic(
            contract,
            phase="lockbox",
            root_seed=root_seed,
            participants_per_family=64,
            families=FAMILY_NAMES,
            sign_flip_draws=int(metrics["sign_flip"]["draws"]),
            bootstrap_draws=int(metrics["participant_bootstrap"]["draws"]),
            fbcca_producer=produce_strict_fbcca,
            governed_context=context,
        )
        if (
            current_git_identity() != identity
            or _validate_environment_receipt(current_environment_receipt()) != environment
        ):
            raise RuntimeError("Source or numerical environment changed during lockbox execution.")
        result.pop("result_sha256")
        result.update(
            {
                "generator_commit": identity.commit,
                "generator_tree": identity.tree,
                "generator_source_bundle_sha256": identity.source_bundle_sha256,
                "generator_commit_timestamp_utc": identity.commit_timestamp_utc,
                "environment": environment,
                "authorization_receipt_sha256": authorization_sha,
                "beacon_receipt_sha256": beacon["beacon_receipt_sha256"],
                "one_time_claim_sha256": claim["claim_sha256"],
            }
        )
        result["result_sha256"] = _canonical_json_sha256(result)
        encoded = _json_payload_bytes(result)
        written = _write_bytes_exclusive(target, encoded)
        validate_lockbox_terminal_artifact(contract, artifact_path=written)
        return written
    except HardInvariantFailure as error:
        terminal = _terminal_receipt(
            status="terminal_invariant_fail",
            contract=contract,
            identity=identity,
            environment=environment,
            authorization_sha256=authorization_sha,
            beacon_receipt_sha256=beacon_receipt_sha256,
            claim_sha256=claim["claim_sha256"],
            error=error,
        )
        return write_json_exclusive(terminal_path, terminal)
    except Exception as error:  # noqa: BLE001 - every caught post-claim failure is terminal.
        terminal = _terminal_receipt(
            status="consumed_inconclusive_infrastructure_error",
            contract=contract,
            identity=identity,
            environment=environment,
            authorization_sha256=authorization_sha,
            beacon_receipt_sha256=beacon_receipt_sha256,
            claim_sha256=claim["claim_sha256"],
            error=error,
        )
        return write_json_exclusive(terminal_path, terminal)


_TERMINAL_RECEIPT_FIELDS = {
    "schema",
    "candidate_id",
    "phase",
    "status",
    "human_claim_boundary",
    "plan_sha256",
    "generator_commit",
    "generator_tree",
    "generator_source_bundle_sha256",
    "generator_commit_timestamp_utc",
    "environment",
    "authorization_receipt_sha256",
    "beacon_receipt_sha256",
    "one_time_claim_sha256",
    "error_type",
    "error_message",
    "automatic_retry",
    "result_sha256",
}


def _live_lockbox_bindings(
    contract: SyntheticContract,
) -> tuple[GitIdentity, dict[str, Any], dict[str, Any], str]:
    identity = current_git_identity()
    _require_clean_identity(identity)
    environment = _validate_environment_receipt(current_environment_receipt())
    authorization_path = canonical_execution_path(contract, "lockbox_authorization")
    _readonly_receipt_file(authorization_path, "synthetic lockbox authorization")
    _require_file_strictly_predates_target(
        authorization_path, contract, "synthetic lockbox authorization"
    )
    authorization = load_json_object(
        authorization_path, name="synthetic lockbox authorization"
    )
    authorization_sha = _validate_consumed_authorization_binding(
        authorization,
        contract=contract,
        identity=identity,
        environment=environment,
    )
    return identity, environment, authorization, authorization_sha


def _validate_lockbox_scientific_result(
    contract: SyntheticContract,
    path: Path,
) -> dict[str, Any]:
    terminal_path = canonical_execution_path(contract, "lockbox_terminal_receipt")
    try:
        _validated_unused_output_path(terminal_path)
    except FileExistsError as error:
        raise ValueError(
            "A consumed/error terminal receipt has strict precedence over any scientific result."
        ) from error
    _readonly_receipt_file(path, "synthetic lockbox scientific result")
    observed = load_json_object(path, name="synthetic lockbox scientific result")
    _require_exact_keys(observed, _LOCKBOX_RESULT_FIELDS, "lockbox scientific result")
    claimed = observed.pop("result_sha256")
    if not isinstance(claimed, str) or claimed != _canonical_json_sha256(observed):
        raise ValueError("Synthetic lockbox result self-hash mismatch.")
    identity, environment, authorization, authorization_sha = _live_lockbox_bindings(
        contract
    )
    claim, claim_sha = _validate_pre_network_claim(
        contract,
        claim_path=canonical_execution_path(contract, "seed_global_lockbox_claim"),
        authorization_path=canonical_execution_path(contract, "lockbox_authorization"),
        authorization=authorization,
        identity=identity,
        environment=environment,
    )
    beacon, root_seed = _validate_beacon_evidence(
        contract,
        identity=identity,
        environment=environment,
        preparation_receipt_sha256=authorization["preparation_evidence"][
            "receipt_sha256"
        ],
        development_result_sha256=authorization["development_evidence"][
            "receipt_sha256"
        ],
        test_evidence_sha256=authorization["test_evidence"]["receipt_sha256"],
        one_time_claim_sha256=claim_sha,
    )
    expected_assertions = {name: True for name in contract.plan["hard_assertions_before_efficacy"]}
    if (
        observed["schema"] != SYNTHETIC_RESULT_SCHEMA
        or observed["candidate_id"] != "metadata-calibration-efficiency-v2"
        or observed["phase"] != "lockbox"
        or observed["status"] not in {"terminal_pass", "terminal_scientific_fail"}
        or observed["human_claim_boundary"]
        != "synthetic_pass_is_necessary_but_never_sufficient"
        or observed["plan_sha256"] != contract.plan_sha256
        or observed["filterbank_sha256"] != contract.filterbank_sha256
        or observed["diagnostic_operator_configs_sha256"]
        != contract.diagnostic_operator_configs_sha256
        or observed["root_seed"] != root_seed
        or observed["participants_per_family"] != 64
        or observed["families"] != list(FAMILY_NAMES)
        or observed["support_budgets"] != list(SUPPORT_BUDGETS)
        or observed["query_blocks"]
        != list(contract.plan["population"]["immutable_query_blocks"])
        or observed["hard_assertions_completed_before_metrics"] is not True
        or observed["hard_assertions"] != expected_assertions
        or observed["development_outputs_are_engineering_only"] is not False
        or observed["lockbox_terminal"] is not True
        or observed["generator_commit"] != identity.commit
        or observed["generator_tree"] != identity.tree
        or observed["generator_source_bundle_sha256"] != identity.source_bundle_sha256
        or observed["generator_commit_timestamp_utc"] != identity.commit_timestamp_utc
        or observed["environment"] != environment
        or observed["authorization_receipt_sha256"] != authorization_sha
        or observed["beacon_receipt_sha256"] != beacon["beacon_receipt_sha256"]
        or observed["one_time_claim_sha256"] != claim_sha
        or claim["output_path"] != str(path)
        or observed["rng"]
        != {
            "bit_generator": "numpy.random.PCG64DXSM",
            "construction": "numpy.random.SeedSequence",
            "complete_key_order": list(contract.plan["rng"]["key_order"]),
        }
    ):
        raise ValueError("Synthetic lockbox result source, evidence, or grid binding drifted.")
    metric_rows = _validate_complete_metric_grid(observed["participant_metrics"])
    contrasts = _validate_contrast_schema(observed["primary_contrasts"])
    promotion = _validate_promotion_schema(observed["promotion"])
    metrics = contract.plan["metrics"]["sensitivity_only"]
    context = _governed_seed_context(contract, phase="lockbox", root_seed=root_seed)
    recomputed_contrasts, recomputed_promotion = summarize_synthetic_metrics(
        metric_rows,
        contract=contract,
        root_seed=root_seed,
        families=FAMILY_NAMES,
        sign_flip_draws=int(metrics["sign_flip"]["draws"]),
        bootstrap_draws=int(metrics["participant_bootstrap"]["draws"]),
        _governed_context=context,
    )
    expected_status = (
        "terminal_pass" if recomputed_promotion["all_requirements_passed"] else "terminal_scientific_fail"
    )
    if (
        observed["status"] != expected_status
        or _canonical_json_bytes({"value": contrasts})
        != _canonical_json_bytes({"value": recomputed_contrasts})
        or _canonical_json_bytes({"value": promotion})
        != _canonical_json_bytes({"value": recomputed_promotion})
    ):
        raise ValueError("Lockbox contrasts, promotion, or terminal status do not recompute.")
    observed["result_sha256"] = claimed
    return observed


def _validate_lockbox_error_receipt(
    contract: SyntheticContract,
    path: Path,
) -> dict[str, Any]:
    _readonly_receipt_file(path, "synthetic lockbox terminal receipt")
    receipt = load_json_object(path, name="synthetic lockbox terminal receipt")
    _require_exact_keys(receipt, _TERMINAL_RECEIPT_FIELDS, "lockbox terminal receipt")
    claimed = receipt.pop("result_sha256")
    if not isinstance(claimed, str) or claimed != _canonical_json_sha256(receipt):
        raise ValueError("Synthetic terminal receipt self-hash mismatch.")
    identity, environment, authorization, authorization_sha = _live_lockbox_bindings(
        contract
    )
    allowed = {
        "terminal_invariant_fail",
        "consumed_inconclusive_infrastructure_error",
        "consumed_inconclusive_claim_write_error",
    }
    if (
        receipt["schema"] != SYNTHETIC_TERMINAL_RECEIPT_SCHEMA
        or receipt["candidate_id"] != "metadata-calibration-efficiency-v2"
        or receipt["phase"] != "lockbox"
        or receipt["status"] not in allowed
        or receipt["human_claim_boundary"]
        != "synthetic_pass_is_necessary_but_never_sufficient"
        or receipt["plan_sha256"] != contract.plan_sha256
        or receipt["generator_commit"] != identity.commit
        or receipt["generator_tree"] != identity.tree
        or receipt["generator_source_bundle_sha256"] != identity.source_bundle_sha256
        or receipt["generator_commit_timestamp_utc"] != identity.commit_timestamp_utc
        or receipt["environment"] != environment
        or receipt["authorization_receipt_sha256"] != authorization_sha
        or not isinstance(receipt["one_time_claim_sha256"], str)
        or not _SHA256_RE.fullmatch(receipt["one_time_claim_sha256"])
        or receipt["automatic_retry"] != "forbidden"
        or not isinstance(receipt["error_type"], str)
        or not receipt["error_type"]
        or not isinstance(receipt["error_message"], str)
    ):
        raise ValueError("Synthetic terminal receipt source or evidence binding drifted.")
    claim_path = canonical_execution_path(contract, "seed_global_lockbox_claim")
    if receipt["status"] == "consumed_inconclusive_claim_write_error":
        _regular_nonsymlink_file(claim_path, "partial canonical global claim")
    else:
        _, claim_sha = _validate_pre_network_claim(
            contract,
            claim_path=claim_path,
            authorization_path=canonical_execution_path(contract, "lockbox_authorization"),
            authorization=authorization,
            identity=identity,
            environment=environment,
        )
        if receipt["one_time_claim_sha256"] != claim_sha:
            raise ValueError("Terminal receipt does not bind the durable global claim.")
    beacon_sha = receipt["beacon_receipt_sha256"]
    if beacon_sha is not None and (
        not isinstance(beacon_sha, str) or not _SHA256_RE.fullmatch(beacon_sha)
    ):
        raise ValueError("Terminal receipt beacon digest must be null or SHA-256.")
    if beacon_sha is not None:
        beacon, _ = _validate_beacon_evidence(
            contract,
            identity=identity,
            environment=environment,
            preparation_receipt_sha256=authorization["preparation_evidence"][
                "receipt_sha256"
            ],
            development_result_sha256=authorization["development_evidence"][
                "receipt_sha256"
            ],
            test_evidence_sha256=authorization["test_evidence"]["receipt_sha256"],
            one_time_claim_sha256=receipt["one_time_claim_sha256"],
        )
        if beacon["beacon_receipt_sha256"] != beacon_sha:
            raise ValueError("Terminal receipt does not bind the canonical beacon receipt.")
    receipt["result_sha256"] = claimed
    return receipt


def validate_lockbox_terminal_artifact(
    contract: SyntheticContract,
    *,
    artifact_path: str | Path,
) -> dict[str, Any]:
    """Validate the single scientific result or separate consumed-error receipt."""

    observed = Path(artifact_path).expanduser().absolute()
    result_path = canonical_execution_path(contract, "lockbox_result")
    terminal_path = canonical_execution_path(contract, "lockbox_terminal_receipt")
    if observed == result_path:
        return _validate_lockbox_scientific_result(contract, observed)
    if observed == terminal_path:
        return _validate_lockbox_error_receipt(contract, observed)
    raise ValueError("Lockbox terminal artifact must use one exact canonical result path.")
