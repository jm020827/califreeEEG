from __future__ import annotations

import hashlib
import inspect
import json
import math
import os
import re
import stat
import subprocess
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
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

EXPECTED_SYNTHETIC_PLAN_SHA256 = "e7b9bf107b91a2e43d4456f11758ba5afc9ba6a0495e72ada63ac9d7858bf9ac"
EXPECTED_FILTERBANK_SHA256 = "b8c1ce4477d980b40f359bc3dc97f3125313191d24401f29a7e2dc473d0d1380"
DEVELOPMENT_ROOT_SEED = 20260906
LOCKBOX_ROOT_SEED = 20260907
RESERVED_ROOT_SEEDS = frozenset((DEVELOPMENT_ROOT_SEED, LOCKBOX_ROOT_SEED))

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
_OWNED_SOURCE_PATHS = (
    Path("src/cfeg/analysis/metadata_calibration_v2_synthetic.py"),
    Path("scripts/run_metadata_calibration_v2_synthetic.py"),
    Path("tests/test_metadata_calibration_v2_synthetic.py"),
)
_EXECUTION_SOURCE_PATHS = (
    *_OWNED_SOURCE_PATHS,
    Path("src/cfeg/models/metadata_calibration_v2.py"),
    Path("src/cfeg/baselines/fbcca.py"),
    Path("configs/analysis/metadata_calibration_v2_synthetic.yaml"),
    Path("configs/baselines/fbcca_chen2015_m3_v2_explicit_weights.yaml"),
)
_SHA256_RE = re.compile(r"[0-9a-f]{64}")
_GIT_OBJECT_RE = re.compile(r"[0-9a-f]{40,64}")
_RESERVED_EXECUTION_CAPABILITY = object()


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


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
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


def _regular_nonsymlink_file(path: str | Path, name: str) -> Path:
    resolved = Path(path)
    if not resolved.exists() or resolved.is_symlink() or not resolved.is_file():
        raise ValueError(f"{name} must be one existing nonsymlink regular file: {resolved}")
    return resolved


def validate_synthetic_contract(
    plan_path: str | Path = DEFAULT_SYNTHETIC_PLAN_PATH,
) -> SyntheticContract:
    """Load the byte-frozen r2 plan and its separately frozen FBCCA contract."""

    path = _regular_nonsymlink_file(plan_path, "synthetic plan")
    observed_sha = _sha256_file(path)
    if observed_sha != EXPECTED_SYNTHETIC_PLAN_SHA256:
        raise ValueError(
            "Synthetic plan byte hash drifted from frozen r2: "
            f"expected {EXPECTED_SYNTHETIC_PLAN_SHA256}, observed {observed_sha}."
        )
    decoded = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(decoded, Mapping):
        raise TypeError("Synthetic plan must decode to one mapping.")
    plan = dict(decoded)
    if (
        plan.get("schema") != SYNTHETIC_PLAN_SCHEMA
        or plan.get("candidate_id") != "metadata-calibration-efficiency-v2"
        or plan.get("generator_revision") != "v5_pre_outcome_exact_component_phase_formulas"
        or plan.get("status") != "frozen_before_any_synthetic_outcome"
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
        or rng.get("independent_lockbox_root_seed") != LOCKBOX_ROOT_SEED
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
        raise ValueError("Synthetic r2 B3/N4 state mapping is not exact.")
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
    filterbank = yaml.safe_load(filterbank_path.read_text(encoding="utf-8"))
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
) -> np.random.Generator:
    """Construct one loop-order-independent PCG64DXSM stream from all six keys."""

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
) -> SyntheticParticipant:
    """Generate one participant; no family-sized EEG tensor is ever materialized."""

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
        ).normal(0.0, 0.08)
        gains = keyed_rng(
            contract,
            root_seed=root_seed,
            family=family,
            participant_index=participant_index,
            block=block,
            class_index=0,
            component="channel_gain",
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
            ).integers(0, 2)
            context_draw = keyed_rng(
                contract,
                root_seed=root_seed,
                family=family,
                participant_index=participant_index,
                block=block,
                class_index=0,
                component="metadata_shuffle",
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
) -> Path:
    """Write one participant-only cache without permitting overwrite or pickle payloads."""

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
) -> StrictFBCCAProduct:
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
) -> tuple[int, ...]:
    if control == "correct" or control == "opposite_interface" or control == "all_missing":
        return tuple(int(block) for block in support_blocks)
    if control == "stale":
        return tuple(10 if int(block) == 1 else int(block) - 1 for block in support_blocks)
    if control == "pair_shuffled":
        permutation = keyed_rng(
            contract,
            root_seed=participant.root_seed,
            family=participant.family,
            participant_index=participant.participant_index,
            block=0,
            class_index=0,
            component="metadata_shuffle",
        ).permutation(10)
        return tuple(int(permutation[int(block) - 1]) + 1 for block in support_blocks)
    raise ValueError(f"Unknown metadata pairing control {control!r}.")


def _context_arguments(
    participant: SyntheticParticipant,
    contract: SyntheticContract,
    *,
    query_blocks: Sequence[int],
    support_blocks: Sequence[int],
    control: str,
) -> tuple[dict[str, Any], str]:
    if control not in CONTEXT_CONTROLS:
        raise ValueError(f"Unknown metadata pairing control {control!r}.")
    source_blocks = _metadata_source_blocks(participant, support_blocks, control, contract)
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
) -> v2_operator.PrequentialGateDecision:
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
) -> tuple[v2_operator.V2OperatorOutput, v2_operator.PrequentialGateDecision | None]:
    """Score immutable query blocks without exposing their labels to the operator."""

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
) -> list[dict[str, Any]]:
    """Return scalar participant metrics after label-free operator predictions exist."""

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
) -> dict[str, bool]:
    """Execute every frozen hard invariant before any efficacy metric is recorded."""

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
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    vectors = _contrast_vectors(metric_rows, families)
    summaries: list[dict[str, Any]] = []
    for contrast_index, name in enumerate(sorted(vectors)):
        family, values, margin = vectors[name]
        severe_harm_applicable = name.endswith(":eAUC")
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
            "all_primary_contrasts_exact_upper_rate_below_0.10": bool(severe_harm_pass),
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
    reserved_execution_capability: object | None = None,
) -> dict[str, Any]:
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
    if root_seed in RESERVED_ROOT_SEEDS:
        sensitivity = contract.plan["metrics"]["sensitivity_only"]
        expected_phase = "development" if root_seed == DEVELOPMENT_ROOT_SEED else "lockbox"
        if (
            reserved_execution_capability is not _RESERVED_EXECUTION_CAPABILITY
            or phase != expected_phase
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
            )
            product = fbcca_producer(contract, participant.signals)
            if not isinstance(product, StrictFBCCAProduct):
                raise TypeError("FBCCA producer must return StrictFBCCAProduct.")
            if hard_assertions is None:
                hard_assertions = validate_hard_assertions(participant, product, contract)
            # Efficacy metrics are not evaluated until all hard assertions have passed.
            metric_rows.extend(evaluate_synthetic_participant(participant, product, contract))
    if hard_assertions is None or not all(hard_assertions.values()):
        raise RuntimeError("Synthetic hard assertions did not complete before efficacy metrics.")
    contrasts, promotion = summarize_synthetic_metrics(
        metric_rows,
        contract=contract,
        root_seed=root_seed,
        families=selected_families,
        sign_flip_draws=sign_flip_draws,
        bootstrap_draws=bootstrap_draws,
    )
    status = (
        "engineering_only"
        if phase in {"diagnostic", "development"}
        else "terminal_pass"
        if promotion["all_requirements_passed"]
        else "terminal_fail"
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

    if root_seed in RESERVED_ROOT_SEEDS:
        raise ValueError("Reserved development/lockbox seeds cannot use the diagnostic path.")
    contract = validate_synthetic_contract(plan_path)
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


def run_development(
    contract: SyntheticContract,
) -> dict[str, Any]:
    """Run the frozen development seed with no alternate size or resampling path."""

    identity = current_git_identity()
    if not identity.clean:
        raise ValueError("Frozen development execution requires a clean git worktree.")
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
        reserved_execution_capability=_RESERVED_EXECUTION_CAPABILITY,
    )


def current_git_identity(repository: str | Path = _REPOSITORY) -> GitIdentity:
    root = Path(repository).resolve()

    def git(*arguments: str) -> str:
        completed = subprocess.run(
            ["git", *arguments],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        )
        return completed.stdout.strip()

    relative_paths = [path.as_posix() for path in _EXECUTION_SOURCE_PATHS]
    # Bind the latest commit touching any execution dependency, rather than an unrelated
    # later research-log commit. The tree is the complete repository tree at that anchor.
    commit = git("log", "-1", "--format=%H", "--", *relative_paths)
    tree = git("rev-parse", f"{commit}^{{tree}}")
    if not _GIT_OBJECT_RE.fullmatch(commit) or not _GIT_OBJECT_RE.fullmatch(tree):
        raise RuntimeError("Git did not return valid commit/tree object IDs.")
    status_output = git("status", "--porcelain=v1", "--untracked-files=all", "--", *relative_paths)
    bundle_records = []
    for relative in _EXECUTION_SOURCE_PATHS:
        path = _regular_nonsymlink_file(root / relative, f"generator source {relative}")
        bundle_records.append({"path": relative.as_posix(), "sha256": _sha256_file(path)})
    return GitIdentity(
        commit=commit,
        tree=tree,
        source_bundle_sha256=_canonical_json_sha256(
            {
                "schema": "cfeg.metadata-calibration-v2-synthetic-source-bundle.v1",
                "files": bundle_records,
            }
        ),
        clean=status_output == "",
    )


def _source_hashes(repository: Path = _REPOSITORY) -> dict[str, str]:
    values: dict[str, str] = {}
    for relative in _EXECUTION_SOURCE_PATHS:
        values[relative.as_posix()] = _sha256_file(
            _regular_nonsymlink_file(repository / relative, f"generator source {relative}")
        )
    return values


def build_preparation_receipt(
    contract: SyntheticContract,
    *,
    output_path: str | Path,
    identity: GitIdentity | None = None,
) -> dict[str, Any]:
    resolved_identity = current_git_identity() if identity is None else identity
    target = _validated_unused_output_path(output_path)
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
        "generator_commit": resolved_identity.commit,
        "generator_tree": resolved_identity.tree,
        "generator_source_bundle_sha256": resolved_identity.source_bundle_sha256,
        "git_worktree_clean": resolved_identity.clean,
        "development_root_seed": DEVELOPMENT_ROOT_SEED,
        "lockbox_root_seed": LOCKBOX_ROOT_SEED,
        "reserved_seed_executed": False,
        "requested_output_path": str(target),
        "lockbox_authorization_schema": SYNTHETIC_LOCKBOX_AUTHORIZATION_SCHEMA,
        "lockbox_authorized": False,
    }
    receipt["completion_receipt_sha256"] = _canonical_json_sha256(receipt)
    return receipt


def _development_evidence(
    path: str | Path | None,
    *,
    contract: SyntheticContract,
    identity: GitIdentity,
    permit_pending: bool,
) -> dict[str, Any]:
    if path is None:
        if not permit_pending:
            raise ValueError("Lockbox authorization requires a recorded development result.")
        return {
            "schema": SYNTHETIC_DEVELOPMENT_EVIDENCE_SCHEMA,
            "status": "pending",
            "path": None,
            "file_sha256": None,
            "result_sha256": None,
            "plan_sha256": contract.plan_sha256,
            "filterbank_sha256": contract.filterbank_sha256,
            "diagnostic_operator_configs_sha256": contract.diagnostic_operator_configs_sha256,
            "root_seed": DEVELOPMENT_ROOT_SEED,
            "participants_per_family": 64,
            "families": list(FAMILY_NAMES),
            "generator_commit": identity.commit,
            "generator_tree": identity.tree,
            "generator_source_bundle_sha256": identity.source_bundle_sha256,
        }
    source = _regular_nonsymlink_file(path, "synthetic development result").resolve()
    result = load_json_object(source, name="synthetic development result")
    result_sha = result.get("result_sha256")
    unsigned = dict(result)
    unsigned.pop("result_sha256", None)
    if not isinstance(result_sha, str) or result_sha != _canonical_json_sha256(unsigned):
        raise ValueError("Synthetic development result self-hash is invalid.")
    if (
        result.get("schema") != SYNTHETIC_RESULT_SCHEMA
        or result.get("phase") != "development"
        or result.get("status") != "engineering_only"
        or result.get("plan_sha256") != contract.plan_sha256
        or result.get("filterbank_sha256") != contract.filterbank_sha256
        or result.get("diagnostic_operator_configs_sha256")
        != contract.diagnostic_operator_configs_sha256
        or result.get("root_seed") != DEVELOPMENT_ROOT_SEED
        or result.get("participants_per_family") != 64
        or result.get("families") != list(FAMILY_NAMES)
        or result.get("generator_commit") != identity.commit
        or result.get("generator_tree") != identity.tree
        or result.get("generator_source_bundle_sha256") != identity.source_bundle_sha256
    ):
        raise ValueError("Synthetic development result is incomplete, stale, or not frozen.")
    return {
        "schema": SYNTHETIC_DEVELOPMENT_EVIDENCE_SCHEMA,
        "status": "recorded_engineering_only",
        "path": str(source),
        "file_sha256": _sha256_file(source),
        "result_sha256": result_sha,
        "plan_sha256": contract.plan_sha256,
        "filterbank_sha256": contract.filterbank_sha256,
        "diagnostic_operator_configs_sha256": contract.diagnostic_operator_configs_sha256,
        "root_seed": DEVELOPMENT_ROOT_SEED,
        "participants_per_family": 64,
        "families": list(FAMILY_NAMES),
        "generator_commit": identity.commit,
        "generator_tree": identity.tree,
        "generator_source_bundle_sha256": identity.source_bundle_sha256,
    }


def build_lockbox_authorization_template(
    contract: SyntheticContract,
    *,
    output_path: str | Path,
    development_result_path: str | Path | None = None,
    identity: GitIdentity | None = None,
) -> dict[str, Any]:
    """Build a non-authorizing template that a separate decision must complete."""

    resolved_identity = current_git_identity() if identity is None else identity
    target = _validated_unused_output_path(output_path)
    claim = _claim_path(target)
    source_hashes = _source_hashes()
    development_evidence = _development_evidence(
        development_result_path,
        contract=contract,
        identity=resolved_identity,
        permit_pending=True,
    )
    record: dict[str, Any] = {
        "schema": SYNTHETIC_LOCKBOX_AUTHORIZATION_SCHEMA,
        "candidate_id": "metadata-calibration-efficiency-v2",
        "phase": "lockbox",
        "status": "authorization_required",
        "authorized": False,
        "authorized_by": "",
        "authorization_basis": "",
        "plan_sha256": contract.plan_sha256,
        "filterbank_sha256": contract.filterbank_sha256,
        "diagnostic_operator_configs_sha256": contract.diagnostic_operator_configs_sha256,
        "lockbox_root_seed": LOCKBOX_ROOT_SEED,
        "generator_commit": resolved_identity.commit,
        "generator_tree": resolved_identity.tree,
        "generator_source_bundle_sha256": resolved_identity.source_bundle_sha256,
        "test_evidence": {
            "schema": "cfeg.metadata-calibration-v2-synthetic-test-evidence.v1",
            "status": "pending",
            "command": "python -m pytest -q tests/test_metadata_calibration_v2_synthetic.py",
            "exit_code": None,
            "generator_commit": resolved_identity.commit,
            "source_sha256s": source_hashes,
        },
        "development_evidence": development_evidence,
        "unused_one_time_output": {
            "path": str(target),
            "claim_path": str(claim),
            "observed_output_absent": not target.exists(),
            "observed_claim_absent": not claim.exists(),
            "never_overwrite": True,
        },
        "one_time_nonce_sha256": None,
    }
    record["authorization_receipt_sha256"] = _canonical_json_sha256(record)
    return record


_AUTHORIZATION_FIELDS = {
    "schema",
    "candidate_id",
    "phase",
    "status",
    "authorized",
    "authorized_by",
    "authorization_basis",
    "plan_sha256",
    "filterbank_sha256",
    "diagnostic_operator_configs_sha256",
    "lockbox_root_seed",
    "generator_commit",
    "generator_tree",
    "generator_source_bundle_sha256",
    "test_evidence",
    "development_evidence",
    "unused_one_time_output",
    "one_time_nonce_sha256",
    "authorization_receipt_sha256",
}
_TEST_EVIDENCE_FIELDS = {
    "schema",
    "status",
    "command",
    "exit_code",
    "generator_commit",
    "source_sha256s",
}
_OUTPUT_BINDING_FIELDS = {
    "path",
    "claim_path",
    "observed_output_absent",
    "observed_claim_absent",
    "never_overwrite",
}
_DEVELOPMENT_EVIDENCE_FIELDS = {
    "schema",
    "status",
    "path",
    "file_sha256",
    "result_sha256",
    "plan_sha256",
    "filterbank_sha256",
    "diagnostic_operator_configs_sha256",
    "root_seed",
    "participants_per_family",
    "families",
    "generator_commit",
    "generator_tree",
    "generator_source_bundle_sha256",
}


def seal_authorization_record(record: Mapping[str, Any]) -> dict[str, Any]:
    """Recompute a record self-hash; this does not itself grant authorization."""

    sealed = dict(record)
    sealed.pop("authorization_receipt_sha256", None)
    sealed["authorization_receipt_sha256"] = _canonical_json_sha256(sealed)
    return sealed


def validate_lockbox_authorization(
    authorization: Mapping[str, Any],
    *,
    contract: SyntheticContract,
    output_path: str | Path,
    identity: GitIdentity | None = None,
) -> str:
    """Fail closed unless an exact, fresh, clean-tree one-time receipt is present."""

    record = dict(authorization)
    _require_exact_keys(record, _AUTHORIZATION_FIELDS, "lockbox authorization")
    claimed_hash = record.pop("authorization_receipt_sha256")
    if not isinstance(claimed_hash, str) or not _SHA256_RE.fullmatch(claimed_hash):
        raise ValueError("Lockbox authorization self-hash must be lowercase SHA-256.")
    if claimed_hash != _canonical_json_sha256(record):
        raise ValueError("Lockbox authorization self-hash mismatch.")
    if (
        record["schema"] != SYNTHETIC_LOCKBOX_AUTHORIZATION_SCHEMA
        or record["candidate_id"] != "metadata-calibration-efficiency-v2"
        or record["phase"] != "lockbox"
        or record["status"] != "authorized_for_one_time_execution"
        or record["authorized"] is not True
    ):
        raise ValueError("Lockbox receipt does not explicitly authorize one-time execution.")
    if not isinstance(record["authorized_by"], str) or not record["authorized_by"].strip():
        raise ValueError("Lockbox receipt must name a separate authorizer.")
    if (
        not isinstance(record["authorization_basis"], str)
        or not record["authorization_basis"].strip()
    ):
        raise ValueError("Lockbox receipt must state a non-empty authorization basis.")
    if (
        record["plan_sha256"] != contract.plan_sha256
        or record["filterbank_sha256"] != contract.filterbank_sha256
        or record["diagnostic_operator_configs_sha256"]
        != contract.diagnostic_operator_configs_sha256
        or record["lockbox_root_seed"] != LOCKBOX_ROOT_SEED
    ):
        raise ValueError("Lockbox receipt differs from a frozen config/seed binding.")

    resolved_identity = current_git_identity() if identity is None else identity
    if not resolved_identity.clean:
        raise ValueError("Lockbox execution requires a clean git worktree.")
    if (
        record["generator_commit"] != resolved_identity.commit
        or record["generator_tree"] != resolved_identity.tree
        or record["generator_source_bundle_sha256"] != resolved_identity.source_bundle_sha256
    ):
        raise ValueError("Lockbox generator commit/tree/source binding drifted.")

    evidence = record["test_evidence"]
    if not isinstance(evidence, Mapping):
        raise TypeError("Lockbox test evidence must be one mapping.")
    evidence = dict(evidence)
    _require_exact_keys(evidence, _TEST_EVIDENCE_FIELDS, "lockbox test evidence")
    if (
        evidence["schema"] != "cfeg.metadata-calibration-v2-synthetic-test-evidence.v1"
        or evidence["status"] != "passed"
        or evidence["exit_code"] != 0
        or evidence["command"]
        != "python -m pytest -q tests/test_metadata_calibration_v2_synthetic.py"
        or evidence["generator_commit"] != resolved_identity.commit
        or evidence["source_sha256s"] != _source_hashes()
    ):
        raise ValueError("Lockbox test evidence is missing, stale, or not exact.")

    development_evidence = record["development_evidence"]
    if not isinstance(development_evidence, Mapping):
        raise TypeError("Lockbox development evidence must be one mapping.")
    development_evidence = dict(development_evidence)
    _require_exact_keys(
        development_evidence,
        _DEVELOPMENT_EVIDENCE_FIELDS,
        "lockbox development evidence",
    )
    observed_development = _development_evidence(
        development_evidence.get("path"),
        contract=contract,
        identity=resolved_identity,
        permit_pending=False,
    )
    if development_evidence != observed_development:
        raise ValueError("Lockbox development evidence is missing, stale, or not exact.")

    target = _validated_unused_output_path(output_path)
    claim = _claim_path(target)
    output_binding = record["unused_one_time_output"]
    if not isinstance(output_binding, Mapping):
        raise TypeError("Lockbox unused-output binding must be one mapping.")
    output_binding = dict(output_binding)
    _require_exact_keys(output_binding, _OUTPUT_BINDING_FIELDS, "unused lockbox output")
    if output_binding != {
        "path": str(target),
        "claim_path": str(claim),
        "observed_output_absent": True,
        "observed_claim_absent": True,
        "never_overwrite": True,
    }:
        raise ValueError("Lockbox receipt does not bind this exact unused one-time output.")
    nonce = record["one_time_nonce_sha256"]
    if not isinstance(nonce, str) or not _SHA256_RE.fullmatch(nonce):
        raise ValueError("Lockbox receipt requires a unique SHA-256 nonce commitment.")
    return claimed_hash


def _validated_unused_output_path(path: str | Path) -> Path:
    candidate = Path(path).expanduser().absolute()
    parent = candidate.parent
    if not parent.exists() or parent.is_symlink() or not parent.is_dir():
        raise ValueError("Output parent must be one existing nonsymlink directory.")
    if candidate.exists() or candidate.is_symlink():
        raise FileExistsError(f"Refusing to overwrite existing output: {candidate}")
    return candidate


def _claim_path(output_path: Path) -> Path:
    return output_path.with_name(output_path.name + ".lockbox-claim.json")


def write_json_exclusive(path: str | Path, payload: Mapping[str, Any]) -> Path:
    target = _validated_unused_output_path(path)
    encoded = (
        json.dumps(
            dict(payload), sort_keys=True, indent=2, ensure_ascii=False, allow_nan=False
        ).encode("utf-8")
        + b"\n"
    )
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(target, flags, stat.S_IRUSR | stat.S_IWUSR)
    # The exclusive artifact is deliberately retained on write failure; callers must never
    # silently reuse a partially consumed one-time output path.
    with os.fdopen(descriptor, "wb", closefd=True) as handle:
        handle.write(encoded)
        handle.flush()
        os.fsync(handle.fileno())
    return target


def run_development_to_path(
    contract: SyntheticContract,
    *,
    output_path: str | Path,
) -> Path:
    target = _validated_unused_output_path(output_path)
    identity = current_git_identity()
    if not identity.clean:
        raise ValueError("Frozen development execution requires a clean git worktree.")
    result = run_development(contract)
    result.pop("result_sha256")
    result.update(
        {
            "generator_commit": identity.commit,
            "generator_tree": identity.tree,
            "generator_source_bundle_sha256": identity.source_bundle_sha256,
        }
    )
    result["result_sha256"] = _canonical_json_sha256(result)
    return write_json_exclusive(target, result)


def run_lockbox_to_path(
    contract: SyntheticContract,
    *,
    authorization: Mapping[str, Any],
    output_path: str | Path,
) -> Path:
    target = _validated_unused_output_path(output_path)
    identity = current_git_identity()
    authorization_sha = validate_lockbox_authorization(
        authorization,
        contract=contract,
        output_path=target,
        identity=identity,
    )
    claim_path = _claim_path(target)
    claim: dict[str, Any] = {
        "schema": SYNTHETIC_LOCKBOX_CLAIM_SCHEMA,
        "status": "one_time_execution_claimed_before_outcome_generation",
        "authorization_receipt_sha256": authorization_sha,
        "plan_sha256": contract.plan_sha256,
        "generator_commit": identity.commit,
        "generator_tree": identity.tree,
        "output_path": str(target),
    }
    claim["claim_sha256"] = _canonical_json_sha256(claim)
    write_json_exclusive(claim_path, claim)

    metrics = contract.plan["metrics"]["sensitivity_only"]
    result = _execute_synthetic(
        contract,
        phase="lockbox",
        root_seed=LOCKBOX_ROOT_SEED,
        participants_per_family=64,
        families=FAMILY_NAMES,
        sign_flip_draws=int(metrics["sign_flip"]["draws"]),
        bootstrap_draws=int(metrics["participant_bootstrap"]["draws"]),
        fbcca_producer=produce_strict_fbcca,
        reserved_execution_capability=_RESERVED_EXECUTION_CAPABILITY,
    )
    result.pop("result_sha256")
    result.update(
        {
            "generator_commit": identity.commit,
            "generator_tree": identity.tree,
            "generator_source_bundle_sha256": identity.source_bundle_sha256,
            "authorization_receipt_sha256": authorization_sha,
            "one_time_claim_sha256": claim["claim_sha256"],
        }
    )
    result["result_sha256"] = _canonical_json_sha256(result)
    return write_json_exclusive(target, result)


def load_json_object(path: str | Path, *, name: str) -> dict[str, Any]:
    source = _regular_nonsymlink_file(path, name)
    decoded = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(decoded, Mapping):
        raise TypeError(f"{name} must decode to one JSON object.")
    return dict(decoded)
