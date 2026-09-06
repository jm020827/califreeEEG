"""Outcome-free scientific core for metadata-calibration V3.

The module deliberately has no dependency on any V2 implementation.  Raw
acquisition context is consumed only by :func:`preflight_context`; the
scientific operator accepts the resulting immutable, hash-bound capability.
This makes the ``M`` insertion point inspectable rather than conventional.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from dataclasses import field as dataclass_field
from functools import lru_cache
from itertools import permutations
from numbers import Integral, Real
from pathlib import Path
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, Literal

import numpy as np

if TYPE_CHECKING:
    from typing_extensions import Self

CANDIDATE_ID = "metadata-calibration-efficiency-v3"
FILTERBANK_SHA256 = "b8c1ce4477d980b40f359bc3dc97f3125313191d24401f29a7e2dc473d0d1380"
OPERATOR_ID = "P3_blockwise_score_prototype_reliability_mixture"
SCORE_NORMALIZATION_ID = "overflow_safe_row_RMS_softmax_T1_v1"
SUPPORT_WEIGHTING_ID = "strict_FBCCA_observed_label_relative_block_mixture_v1"
FUSION_ID = "bounded_entropy_budget_support_residual_v1"
CONTEXT_TRUST_ID = "interface_scaled_impedance_affinity_single_insertion_v1"

CONTEXT_REFERENCE_SCHEMA = "cfeg.metadata-calibration-efficiency-v3.context-reference.v1"
VALIDATED_CONTEXT_REFERENCE_SCHEMA = (
    "cfeg.metadata-calibration-efficiency-v3."
    "validated-context-reference-capability.v1"
)
CONTEXT_PREFLIGHT_SCHEMA = (
    "cfeg.metadata-calibration-efficiency-v3.context-preflight-capability.v1"
)
SUPPORT_RELIABILITY_SCHEMA = (
    "cfeg.metadata-calibration-efficiency-v3.support-reliability-capability.v1"
)
CONTEXT_TRUST_SCHEMA = "cfeg.metadata-calibration-efficiency-v3.context-trust-capability.v1"
SUPPORT_PRODUCT_SCHEMA = "cfeg.metadata-calibration-efficiency-v3.support-product.v1"
GATE_SCHEMA = "cfeg.metadata-calibration-efficiency-v3.prequential-gate.v1"
OPERATOR_OUTPUT_SCHEMA = "cfeg.metadata-calibration-efficiency-v3.operator-output.v1"
PAIRING_SCHEMA = "cfeg.metadata-calibration-efficiency-v3.context-pairing.v1"
CONTEXT_REFERENCE_RNG_AUTHORITY_SCHEMA = (
    "cfeg.metadata-calibration-efficiency-v3.context-reference-rng-authority.v1"
)
DEVELOPMENT_RNG_AUTHORITY_SCHEMA = (
    "cfeg.metadata-calibration-efficiency-v3.development-rng-authority.v1"
)
DEVELOPMENT_BUNDLE_SCHEMA = (
    "cfeg.metadata-calibration-efficiency-v3.development-bundle.v1"
)
RNG_PRIMITIVE_SCHEMA = (
    "cfeg.metadata-calibration-efficiency-v3.development-rng-binding.v1"
)

MASTER_PLAN_REPOSITORY_PATH = "configs/analysis/metadata_calibration_efficiency_v3.yaml"
SYNTHETIC_PLAN_REPOSITORY_PATH = "configs/analysis/metadata_calibration_v3_synthetic.yaml"
MODEL_MODULE_REPOSITORY_PATH = "src/cfeg/models/metadata_calibration_v3.py"
MASTER_PLAN_SHA256 = "29de9a4772da34769806ee4f1633f0cbe50d088945a1205507f43c2befa143db"
SYNTHETIC_PLAN_SHA256 = "df676c0b38b65450a611ab652567531df8818aa83223d0aebd314ca3759a971f"
CONTEXT_REFERENCE_ROOT_SEED = 20_260_910
DEVELOPMENT_ROOT_SEED = 20_260_909
RNG_KEY_ORDER = (
    "root_seed",
    "family_code",
    "participant_index",
    "block",
    "class_index",
    "component_code",
)
RNG_FAMILY_CODES = (
    ("B1_participant_class_confusion", 3101),
    ("B2_participant_phase_spatial_shift", 3102),
    ("B3_impedance_linked_transfer_shift", 3103),
    ("B4_interface_calibrated_impedance_shift", 3104),
    ("N1_clean_anchor", 3201),
    ("N2_random_support_labels", 3202),
    ("N3_nonstationary_calibration", 3203),
    ("N4_context_null", 3204),
    ("N5_invalid_context", 3205),
)
RNG_COMPONENT_CODES = (
    ("spatial_signature", 1),
    ("class_phase", 2),
    ("block_phase_drift", 3),
    ("channel_gain", 4),
    ("innovation_noise", 5),
    ("confuser_state", 6),
    ("support_label_permutation", 7),
    ("impedance_measurement_noise", 8),
    ("interface_assignment", 9),
    ("context_control", 10),
    ("sensitivity_resampling", 11),
)

_IMPORTED_MODEL_MODULE_SHA256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()

FINAL_BUDGETS = (0, 1, 3, 5)
PREQUENTIAL_BUDGETS = (3, 5)
CANONICAL_CHANNELS = tuple(f"ch{index:02d}" for index in range(8))

V3Variant = Literal["A_Q", "A_QM"]
ContextLookupMode = Literal["interface", "pooled", "wrong_interface"]

_OPERATOR_CONSTANTS = {
    "schema": "cfeg.metadata-calibration-efficiency-v3.operator-instance.v1",
    "candidate_id": CANDIDATE_ID,
    "operator_id": OPERATOR_ID,
    "filterbank_sha256": FILTERBANK_SHA256,
    "score_normalization_id": SCORE_NORMALIZATION_ID,
    "support_weighting_id": SUPPORT_WEIGHTING_ID,
    "fusion_id": FUSION_ID,
    "context_trust_id": CONTEXT_TRUST_ID,
}

_GRID_ROWS = (
    (1, "p3-nu_1-lambda_0p10", "1", "0p10", 1.0, 0.10, "63ad5a68d79dbad9cfe4443a207eefd96a15d643bf6bbb4914c7a487bb59e5f9"),
    (2, "p3-nu_1-lambda_0p20", "1", "0p20", 1.0, 0.20, "60b4fdb0f0039ac4e0a08c2ebb7e11e36e0aa43b0578362ada8e0bed81e8a43f"),
    (3, "p3-nu_1-lambda_0p30", "1", "0p30", 1.0, 0.30, "16c9d0b71e855df9190bc961b78b0fdc1fd9123628ace1d856ddd7b0faad2786"),
    (4, "p3-nu_4-lambda_0p10", "4", "0p10", 4.0, 0.10, "38c8ffbcc9746bfcfb0a8157012111412f1eecb9c9ed590135763fd548c29613"),
    (5, "p3-nu_4-lambda_0p20", "4", "0p20", 4.0, 0.20, "5723702a4df184cc70dc6195a8031904eb57d0300e555dbf8cfa6345dc9b8347"),
    (6, "p3-nu_4-lambda_0p30", "4", "0p30", 4.0, 0.30, "73668e3f34272a69d8a0ba7265acb3b53b40aeb0c4ea5889bc910d7d878d3558"),
    (7, "p3-nu_16-lambda_0p10", "16", "0p10", 16.0, 0.10, "a0a46034f0d857410c9a70661cce5c73e25a31bd7b51515d0b8adf8ebd2c3201"),
    (8, "p3-nu_16-lambda_0p20", "16", "0p20", 16.0, 0.20, "035c524182fc1c4fca0ae5ccb273d09b4499955784e89b948861312511620d08"),
    (9, "p3-nu_16-lambda_0p30", "16", "0p30", 16.0, 0.30, "d0db62d57756ea1513b910d6c7e2b85b82d0a0b8aa1b76a4a7f9a03ea32e3bfa"),
)


def canonical_json_bytes(value: Mapping[str, Any]) -> bytes:
    """Return the contract's exact canonical JSON encoding."""

    if not isinstance(value, Mapping):
        raise TypeError("canonical JSON input must be a mapping.")
    return json.dumps(
        dict(value),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def canonical_payload_sha256(value: Mapping[str, Any]) -> str:
    """Hash a payload after removing only its top-level self hash."""

    payload = dict(value)
    payload.pop("payload_sha256", None)
    return hashlib.sha256(canonical_json_bytes(payload)).hexdigest()


def _canonical_plain_sha256(value: Mapping[str, Any]) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _array_sha256(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    payload = {
        "schema": "cfeg.ndarray.v1",
        "dtype": array.dtype.str,
        "shape": list(array.shape),
        "bytes_sha256": hashlib.sha256(array.tobytes(order="C")).hexdigest(),
    }
    return _canonical_plain_sha256(payload)


def rng_key_map_sha256() -> str:
    """Return the exact frozen RNG vocabulary digest shared by both roles."""

    return _canonical_plain_sha256(
        {
            "schema": RNG_PRIMITIVE_SCHEMA,
            "bit_generator": "numpy.random.PCG64DXSM",
            "construction": "numpy.random.SeedSequence",
            "key_order": list(RNG_KEY_ORDER),
            "family_codes": [[name, code] for name, code in RNG_FAMILY_CODES],
            "component_codes": [[name, code] for name, code in RNG_COMPONENT_CODES],
        }
    )


_CONTEXT_REFERENCE_RNG_AUTHORITY_ISSUER = object()
_DEVELOPMENT_RNG_AUTHORITY_ISSUER = object()


@dataclass(frozen=True, slots=True, init=False)
class ContextReferenceRNGAuthority:
    """Nominal bundle-bound authority for the exact reference seed only."""

    schema: str
    candidate_id: str
    role: str
    development_bundle_schema: str
    development_bundle_payload_sha256: str
    development_bundle_file_sha256: str
    development_bundle_source_bundle_sha256: str
    numerical_runtime_fingerprint_sha256: str
    clean_commit: str
    clean_tree: str
    master_plan_file_sha256: str
    synthetic_plan_file_sha256: str
    rng_primitive_schema: str
    root_seed: int
    key_order: tuple[str, ...]
    family_codes: tuple[tuple[str, int], ...]
    component_codes: tuple[tuple[str, int], ...]
    key_map_sha256: str
    semantic_binding_sha256: str
    _bundle_capability: object = dataclass_field(repr=False, compare=False)
    _issuer: object = dataclass_field(repr=False, compare=False)

    def __new__(cls, *_args: object, **_kwargs: object) -> Self:
        raise TypeError("ContextReferenceRNGAuthority is issued only from a canonical bundle.")


@dataclass(frozen=True, slots=True, init=False)
class DevelopmentRNGAuthority:
    """Nominal bundle-bound authority for development seed 20260909 only."""

    schema: str
    candidate_id: str
    role: str
    development_bundle_schema: str
    development_bundle_payload_sha256: str
    development_bundle_file_sha256: str
    development_bundle_source_bundle_sha256: str
    numerical_runtime_fingerprint_sha256: str
    clean_commit: str
    clean_tree: str
    master_plan_file_sha256: str
    synthetic_plan_file_sha256: str
    rng_primitive_schema: str
    root_seed: int
    key_order: tuple[str, ...]
    family_codes: tuple[tuple[str, int], ...]
    component_codes: tuple[tuple[str, int], ...]
    key_map_sha256: str
    semantic_binding_sha256: str
    _bundle_capability: object = dataclass_field(repr=False, compare=False)
    _issuer: object = dataclass_field(repr=False, compare=False)

    def __new__(cls, *_args: object, **_kwargs: object) -> Self:
        raise TypeError("DevelopmentRNGAuthority is issued only from a canonical bundle.")


def _rng_authority_binding(value: ContextReferenceRNGAuthority | DevelopmentRNGAuthority) -> dict[str, Any]:
    return {
        "schema": value.schema,
        "candidate_id": value.candidate_id,
        "role": value.role,
        "development_bundle_schema": value.development_bundle_schema,
        "development_bundle_payload_sha256": value.development_bundle_payload_sha256,
        "development_bundle_file_sha256": value.development_bundle_file_sha256,
        "development_bundle_source_bundle_sha256": (
            value.development_bundle_source_bundle_sha256
        ),
        "numerical_runtime_fingerprint_sha256": (
            value.numerical_runtime_fingerprint_sha256
        ),
        "clean_commit": value.clean_commit,
        "clean_tree": value.clean_tree,
        "master_plan_file_sha256": value.master_plan_file_sha256,
        "synthetic_plan_file_sha256": value.synthetic_plan_file_sha256,
        "rng_primitive_schema": value.rng_primitive_schema,
        "root_seed": value.root_seed,
        "key_order": list(value.key_order),
        "family_codes": [[name, code] for name, code in value.family_codes],
        "component_codes": [[name, code] for name, code in value.component_codes],
        "key_map_sha256": value.key_map_sha256,
    }


def issue_bundle_bound_rng_authorities(
    development_bundle_capability: object,
    *,
    expected_commit: str,
    expected_tree: str,
) -> tuple[ContextReferenceRNGAuthority, DevelopmentRNGAuthority]:
    """Issue the two disjoint RNG roles after canonical bundle revalidation."""

    _git_object_id(expected_commit, "expected_commit")
    _git_object_id(expected_tree, "expected_tree")
    try:
        from cfeg.metadata_calibration_v3_governance import (
            require_development_rng_bundle_capability,
        )
    except ImportError as error:  # pragma: no cover - integration installation failure
        raise RuntimeError("V3 governance bundle validator is unavailable.") from error
    bundle = require_development_rng_bundle_capability(
        development_bundle_capability,
        expected_commit=expected_commit,
        expected_tree=expected_tree,
    )
    inventory = {
        entry.path: entry.file_sha256 for entry in bundle.tracked_source_files
    }
    if (
        inventory.get(MASTER_PLAN_REPOSITORY_PATH) != MASTER_PLAN_SHA256
        or inventory.get(SYNTHETIC_PLAN_REPOSITORY_PATH) != SYNTHETIC_PLAN_SHA256
    ):
        raise ValueError("canonical bundle does not bind the frozen V3 plan files.")
    common = {
        "candidate_id": CANDIDATE_ID,
        "development_bundle_schema": DEVELOPMENT_BUNDLE_SCHEMA,
        "development_bundle_payload_sha256": bundle.payload_sha256,
        "development_bundle_file_sha256": bundle.file_sha256,
        "development_bundle_source_bundle_sha256": bundle.source_bundle_sha256,
        "numerical_runtime_fingerprint_sha256": (
            bundle.numerical_runtime_fingerprint_sha256
        ),
        "clean_commit": bundle.clean_commit,
        "clean_tree": bundle.clean_tree,
        "master_plan_file_sha256": MASTER_PLAN_SHA256,
        "synthetic_plan_file_sha256": SYNTHETIC_PLAN_SHA256,
        "rng_primitive_schema": RNG_PRIMITIVE_SCHEMA,
        "key_order": RNG_KEY_ORDER,
        "family_codes": RNG_FAMILY_CODES,
        "component_codes": RNG_COMPONENT_CODES,
        "key_map_sha256": rng_key_map_sha256(),
        "_bundle_capability": bundle,
    }
    reference = _issue_rng_authority(
        ContextReferenceRNGAuthority,
        schema=CONTEXT_REFERENCE_RNG_AUTHORITY_SCHEMA,
        role="context_reference",
        root_seed=CONTEXT_REFERENCE_ROOT_SEED,
        issuer=_CONTEXT_REFERENCE_RNG_AUTHORITY_ISSUER,
        common=common,
    )
    development = _issue_rng_authority(
        DevelopmentRNGAuthority,
        schema=DEVELOPMENT_RNG_AUTHORITY_SCHEMA,
        role="development",
        root_seed=DEVELOPMENT_ROOT_SEED,
        issuer=_DEVELOPMENT_RNG_AUTHORITY_ISSUER,
        common=common,
    )
    return (
        require_context_reference_rng_authority(reference),
        require_development_rng_authority(development),
    )


def _issue_rng_authority(
    capability_type: type[ContextReferenceRNGAuthority | DevelopmentRNGAuthority],
    *,
    schema: str,
    role: str,
    root_seed: int,
    issuer: object,
    common: Mapping[str, Any],
) -> ContextReferenceRNGAuthority | DevelopmentRNGAuthority:
    expected_issuer = (
        _CONTEXT_REFERENCE_RNG_AUTHORITY_ISSUER
        if capability_type is ContextReferenceRNGAuthority
        else _DEVELOPMENT_RNG_AUTHORITY_ISSUER
        if capability_type is DevelopmentRNGAuthority
        else None
    )
    if issuer is not expected_issuer:
        raise TypeError("RNG authority issuer token/type pairing is invalid.")
    value = object.__new__(capability_type)
    fields = {
        **dict(common),
        "schema": schema,
        "role": role,
        "root_seed": root_seed,
    }
    for name, item in fields.items():
        object.__setattr__(value, name, item)
    object.__setattr__(
        value,
        "semantic_binding_sha256",
        _canonical_plain_sha256(_rng_authority_binding(value)),
    )
    object.__setattr__(value, "_issuer", issuer)
    return value


def require_context_reference_rng_authority(value: object) -> ContextReferenceRNGAuthority:
    """Revalidate the exact reference role and its originating bundle."""

    if type(value) is not ContextReferenceRNGAuthority:
        raise TypeError("an exact ContextReferenceRNGAuthority is required.")
    _require_rng_authority(
        value,
        schema=CONTEXT_REFERENCE_RNG_AUTHORITY_SCHEMA,
        role="context_reference",
        root_seed=CONTEXT_REFERENCE_ROOT_SEED,
        issuer=_CONTEXT_REFERENCE_RNG_AUTHORITY_ISSUER,
    )
    return value


def require_development_rng_authority(value: object) -> DevelopmentRNGAuthority:
    """Revalidate the exact development role and its originating bundle."""

    if type(value) is not DevelopmentRNGAuthority:
        raise TypeError("an exact DevelopmentRNGAuthority is required.")
    _require_rng_authority(
        value,
        schema=DEVELOPMENT_RNG_AUTHORITY_SCHEMA,
        role="development",
        root_seed=DEVELOPMENT_ROOT_SEED,
        issuer=_DEVELOPMENT_RNG_AUTHORITY_ISSUER,
    )
    return value


def _require_rng_authority(
    value: ContextReferenceRNGAuthority | DevelopmentRNGAuthority,
    *,
    schema: str,
    role: str,
    root_seed: int,
    issuer: object,
    revalidate_bundle: bool = True,
) -> None:
    if value._issuer is not issuer:
        raise TypeError("RNG authority issuer marker is invalid.")
    if (
        value.schema != schema
        or value.candidate_id != CANDIDATE_ID
        or value.role != role
        or value.development_bundle_schema != DEVELOPMENT_BUNDLE_SCHEMA
        or value.master_plan_file_sha256 != MASTER_PLAN_SHA256
        or value.synthetic_plan_file_sha256 != SYNTHETIC_PLAN_SHA256
        or value.rng_primitive_schema != RNG_PRIMITIVE_SCHEMA
        or type(value.root_seed) is not int
        or value.root_seed != root_seed
        or type(value.key_order) is not tuple
        or value.key_order != RNG_KEY_ORDER
        or type(value.family_codes) is not tuple
        or value.family_codes != RNG_FAMILY_CODES
        or type(value.component_codes) is not tuple
        or value.component_codes != RNG_COMPONENT_CODES
        or value.key_map_sha256 != rng_key_map_sha256()
    ):
        raise ValueError("RNG authority identity, role, seed, or key map is invalid.")
    for name in (
        "development_bundle_payload_sha256",
        "development_bundle_file_sha256",
        "development_bundle_source_bundle_sha256",
        "numerical_runtime_fingerprint_sha256",
        "master_plan_file_sha256",
        "synthetic_plan_file_sha256",
        "key_map_sha256",
        "semantic_binding_sha256",
    ):
        _sha256(getattr(value, name), name)
    _git_object_id(value.clean_commit, "clean_commit")
    _git_object_id(value.clean_tree, "clean_tree")
    if value.semantic_binding_sha256 != _canonical_plain_sha256(
        _rng_authority_binding(value)
    ):
        raise ValueError("RNG authority semantic binding is invalid.")
    if not revalidate_bundle:
        return
    try:
        from cfeg.metadata_calibration_v3_governance import (
            require_development_rng_bundle_capability,
        )
    except ImportError as error:  # pragma: no cover - integration installation failure
        raise RuntimeError("V3 governance bundle validator is unavailable.") from error
    bundle = require_development_rng_bundle_capability(
        value._bundle_capability,
        expected_commit=value.clean_commit,
        expected_tree=value.clean_tree,
    )
    inventory = {
        entry.path: entry.file_sha256 for entry in bundle.tracked_source_files
    }
    current_module_sha256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    if (
        current_module_sha256 != _IMPORTED_MODEL_MODULE_SHA256
        or inventory.get(MODEL_MODULE_REPOSITORY_PATH)
        != _IMPORTED_MODEL_MODULE_SHA256
    ):
        raise RuntimeError(
            "loaded V3 model code does not equal the current bundle-bound source bytes."
        )
    if (
        bundle.payload_sha256 != value.development_bundle_payload_sha256
        or bundle.file_sha256 != value.development_bundle_file_sha256
        or bundle.source_bundle_sha256
        != value.development_bundle_source_bundle_sha256
        or bundle.numerical_runtime_fingerprint_sha256
        != value.numerical_runtime_fingerprint_sha256
    ):
        raise ValueError("RNG authority differs from its revalidated development bundle.")


def _require_context_reference_rng_authority_semantics(
    value: object,
) -> ContextReferenceRNGAuthority:
    if type(value) is not ContextReferenceRNGAuthority:
        raise TypeError("an exact ContextReferenceRNGAuthority is required.")
    _require_rng_authority(
        value,
        schema=CONTEXT_REFERENCE_RNG_AUTHORITY_SCHEMA,
        role="context_reference",
        root_seed=CONTEXT_REFERENCE_ROOT_SEED,
        issuer=_CONTEXT_REFERENCE_RNG_AUTHORITY_ISSUER,
        revalidate_bundle=False,
    )
    return value


def _require_development_rng_authority_semantics(
    value: object,
) -> DevelopmentRNGAuthority:
    if type(value) is not DevelopmentRNGAuthority:
        raise TypeError("an exact DevelopmentRNGAuthority is required.")
    _require_rng_authority(
        value,
        schema=DEVELOPMENT_RNG_AUTHORITY_SCHEMA,
        role="development",
        root_seed=DEVELOPMENT_ROOT_SEED,
        issuer=_DEVELOPMENT_RNG_AUTHORITY_ISSUER,
        revalidate_bundle=False,
    )
    return value


@dataclass(frozen=True)
class V3GridCell:
    index: int
    grid_cell_id: str
    nu_token: str
    lambda_token: str
    prototype_prior_pseudocount: float
    lambda_max: float
    operator_instance_sha256: str

    def __post_init__(self) -> None:
        _positive_integer(self.index, "index")
        _nonempty_ascii(self.grid_cell_id, "grid_cell_id")
        _nonempty_ascii(self.nu_token, "nu_token")
        _nonempty_ascii(self.lambda_token, "lambda_token")
        _positive_finite(self.prototype_prior_pseudocount, "prototype_prior_pseudocount")
        maximum = _finite_float(self.lambda_max, "lambda_max")
        if not 0.0 <= maximum <= 1.0:
            raise ValueError("lambda_max must be in [0, 1].")
        _sha256(self.operator_instance_sha256, "operator_instance_sha256")
        expected = operator_instance_sha256(
            grid_cell_id=self.grid_cell_id,
            nu_token=self.nu_token,
            lambda_token=self.lambda_token,
        )
        if self.operator_instance_sha256 != expected:
            raise ValueError("operator-instance digest does not match the exact V3 payload.")


@dataclass(frozen=True)
class V3OperatorConfig:
    grid_cell: V3GridCell
    score_epsilon: float = 1.0e-12
    ideal_prototype_smoothing: float = 0.05

    def __post_init__(self) -> None:
        if type(self.grid_cell) is not V3GridCell:
            raise TypeError("grid_cell must be an exact V3GridCell.")
        if self.grid_cell != grid_cell_by_id(self.grid_cell.grid_cell_id):
            raise ValueError("grid_cell must equal one exact canonical V3 grid cell.")
        if _positive_finite(self.score_epsilon, "score_epsilon") != 1.0e-12:
            raise ValueError("V3 score_epsilon is frozen at 1e-12.")
        smoothing = _finite_float(self.ideal_prototype_smoothing, "ideal_prototype_smoothing")
        if smoothing != 0.05:
            raise ValueError("V3 ideal_prototype_smoothing is frozen at 0.05.")


def operator_instance_sha256(*, grid_cell_id: str, nu_token: str, lambda_token: str) -> str:
    """Compute the frozen string-only operator-instance digest."""

    payload = {
        **_OPERATOR_CONSTANTS,
        "grid_cell_id": _nonempty_ascii(grid_cell_id, "grid_cell_id"),
        "nu_token": _nonempty_ascii(nu_token, "nu_token"),
        "lambda_token": _nonempty_ascii(lambda_token, "lambda_token"),
    }
    if set(payload) != {
        "schema",
        "candidate_id",
        "operator_id",
        "filterbank_sha256",
        "score_normalization_id",
        "support_weighting_id",
        "fusion_id",
        "context_trust_id",
        "grid_cell_id",
        "nu_token",
        "lambda_token",
    } or not all(isinstance(item, str) for item in payload.values()):
        raise RuntimeError("operator-instance digest payload is not exact.")
    return _canonical_plain_sha256(payload)


@lru_cache(maxsize=1)
def canonical_operator_grid() -> tuple[V3GridCell, ...]:
    """Return and revalidate the exact ordered 3-by-3 development grid."""

    cells = tuple(V3GridCell(*row) for row in _GRID_ROWS)
    if tuple(cell.index for cell in cells) != tuple(range(1, 10)):
        raise RuntimeError("V3 grid indices are not consecutive.")
    if len({cell.grid_cell_id for cell in cells}) != 9:
        raise RuntimeError("V3 grid IDs are not unique.")
    if {(cell.prototype_prior_pseudocount, cell.lambda_max) for cell in cells} != {
        (nu, maximum) for nu in (1.0, 4.0, 16.0) for maximum in (0.10, 0.20, 0.30)
    }:
        raise RuntimeError("V3 grid is not the exact frozen Cartesian product.")
    return cells


def grid_cell_by_id(grid_cell_id: str) -> V3GridCell:
    requested = _nonempty_ascii(grid_cell_id, "grid_cell_id")
    matches = tuple(cell for cell in canonical_operator_grid() if cell.grid_cell_id == requested)
    if len(matches) != 1:
        raise ValueError(f"unknown V3 grid cell {requested!r}.")
    return matches[0]


def _require_exact_operator_config(value: object) -> V3OperatorConfig:
    if type(value) is not V3OperatorConfig:
        raise TypeError("config must be an exact V3OperatorConfig.")
    cell = value.grid_cell
    if type(cell) is not V3GridCell or cell != grid_cell_by_id(cell.grid_cell_id):
        raise ValueError("config grid cell is not the exact canonical instance value.")
    if cell.operator_instance_sha256 != operator_instance_sha256(
        grid_cell_id=cell.grid_cell_id,
        nu_token=cell.nu_token,
        lambda_token=cell.lambda_token,
    ):
        raise ValueError("config operator-instance digest is invalid.")
    if value.score_epsilon != 1.0e-12 or value.ideal_prototype_smoothing != 0.05:
        raise ValueError("config numeric constants differ from the frozen V3 contract.")
    return value


def normalize_fbcca_scores(
    scores: np.ndarray,
    *,
    epsilon: float = 1.0e-12,
) -> tuple[np.ndarray, np.ndarray]:
    """Overflow-safe, row-local RMS normalization followed by T=1 softmax."""

    values = _finite_matrix(scores, "scores")
    eps = _positive_finite(epsilon, "epsilon")
    as_float = values.astype(np.float64, copy=False)
    scale = np.max(np.abs(as_float), axis=1, keepdims=True)
    safe_scale = np.where(scale > 0.0, scale, 1.0)
    scaled = as_float / safe_scale
    centered = scaled - scaled.mean(axis=1, keepdims=True)
    rms = np.sqrt(np.mean(np.square(centered), axis=1, keepdims=True))
    normalized = centered / np.maximum(rms, eps / safe_scale)
    normalized -= normalized.mean(axis=1, keepdims=True)
    probabilities = _softmax(normalized)
    return _readonly(normalized), _readonly(probabilities)


def normalized_entropy(probabilities: np.ndarray) -> np.ndarray:
    values = _probability_matrix(probabilities, "probabilities")
    classes = values.shape[1]
    if classes <= 1:
        raise ValueError("normalized entropy requires at least two classes.")
    terms = np.zeros_like(values)
    positive = values > 0.0
    terms[positive] = values[positive] * np.log(values[positive])
    return _readonly(np.clip(-terms.sum(axis=1) / math.log(classes), 0.0, 1.0))


def jensen_shannon_divergence(
    first: np.ndarray,
    second: np.ndarray,
    *,
    epsilon: float = 1.0e-12,
) -> np.ndarray:
    """Rowwise natural-log Jensen--Shannon divergence."""

    left = _probability_matrix(first, "first")
    right = _probability_matrix(second, "second")
    if left.shape != right.shape:
        raise ValueError("Jensen-Shannon operands must have identical shapes.")
    eps = _positive_finite(epsilon, "epsilon")
    midpoint = 0.5 * (left + right)
    return _readonly(0.5 * _rowwise_kl(left, midpoint, eps) + 0.5 * _rowwise_kl(right, midpoint, eps))


_SUPPORT_RELIABILITY_ISSUER = object()
_SUPPORT_PRODUCT_ISSUER = object()


@dataclass(frozen=True, slots=True, init=False)
class MFreeSupportReliabilityCapability:
    candidate_id: str
    active_grid_cell_id: str
    exact_operator_instance_digest: str
    anchor_normalizer_sha256s: tuple[str, ...]
    ordered_support_block_keys: tuple[str, ...]
    block_reliabilities: tuple[float, ...]
    support_eeg_label_manifest_sha256: str
    payload_sha256: str
    schema: str = SUPPORT_RELIABILITY_SCHEMA
    _issuer: object = dataclass_field(repr=False, compare=False)

    def __new__(cls, *_args: object, **_kwargs: object) -> Self:
        raise TypeError("MFreeSupportReliabilityCapability has no public constructor.")

    def __post_init__(self) -> None:
        if self.schema != SUPPORT_RELIABILITY_SCHEMA or self.candidate_id != CANDIDATE_ID:
            raise ValueError("support-reliability capability identity is invalid.")
        grid = grid_cell_by_id(self.active_grid_cell_id)
        if self.exact_operator_instance_digest != grid.operator_instance_sha256:
            raise ValueError("support-reliability operator binding is invalid.")
        if (
            type(self.anchor_normalizer_sha256s) is not tuple
            or type(self.ordered_support_block_keys) is not tuple
            or type(self.block_reliabilities) is not tuple
        ):
            raise TypeError("support-reliability vectors must remain exact tuples.")
        keys = _keys_tuple(self.ordered_support_block_keys, "ordered_support_block_keys")
        if not keys:
            raise ValueError("support-reliability capability requires at least one block.")
        if len(self.anchor_normalizer_sha256s) != len(keys):
            raise ValueError("one anchor-normalizer digest is required per support block.")
        for index, digest in enumerate(self.anchor_normalizer_sha256s):
            _sha256(digest, f"anchor_normalizer_sha256s[{index}]")
        if len(self.block_reliabilities) != len(keys):
            raise ValueError("one reliability is required per support block.")
        for index, value in enumerate(self.block_reliabilities):
            resolved = _finite_float(value, f"block_reliabilities[{index}]")
            if not 1.0e-12 <= resolved <= 1.0:
                raise ValueError("support reliability must be in [1e-12, 1].")
        _sha256(self.support_eeg_label_manifest_sha256, "support_eeg_label_manifest_sha256")
        _validate_capability_hash(self, self._payload())

    def _payload(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "candidate_id": self.candidate_id,
            "active_grid_cell_id": self.active_grid_cell_id,
            "exact_operator_instance_digest": self.exact_operator_instance_digest,
            "anchor_normalizer_sha256s": list(self.anchor_normalizer_sha256s),
            "ordered_support_block_keys": list(self.ordered_support_block_keys),
            "block_reliabilities": list(self.block_reliabilities),
            "support_eeg_label_manifest_sha256": self.support_eeg_label_manifest_sha256,
        }


@dataclass(frozen=True, slots=True, init=False)
class V3SupportProduct:
    query_key: str
    active_grid_cell_id: str
    exact_operator_instance_digest: str
    ordered_support_block_keys: tuple[str, ...]
    query_normalizer_sha256: str
    base_probabilities: np.ndarray
    per_block_support_probabilities: np.ndarray
    support_probabilities: np.ndarray
    block_reliabilities: np.ndarray
    normalized_block_weights: np.ndarray
    reliability_capability: MFreeSupportReliabilityCapability
    block_weight_mode: Literal["reliability", "uniform"]
    payload_sha256: str
    schema: str = SUPPORT_PRODUCT_SCHEMA
    _issuer: object = dataclass_field(repr=False, compare=False)

    def __new__(cls, *_args: object, **_kwargs: object) -> Self:
        raise TypeError("V3SupportProduct has no public constructor.")

    def __post_init__(self) -> None:
        if self.schema != SUPPORT_PRODUCT_SCHEMA:
            raise ValueError("support-product schema is invalid.")
        _nonempty_ascii(self.query_key, "query_key")
        grid = grid_cell_by_id(self.active_grid_cell_id)
        if self.exact_operator_instance_digest != grid.operator_instance_sha256:
            raise ValueError("support-product operator binding is invalid.")
        if type(self.ordered_support_block_keys) is not tuple:
            raise TypeError("support-product keys must remain an exact tuple.")
        keys = _keys_tuple(self.ordered_support_block_keys, "ordered_support_block_keys")
        _sha256(self.query_normalizer_sha256, "query_normalizer_sha256")
        capability = require_m_free_support_reliability(
            self.reliability_capability
        )
        if (
            capability.active_grid_cell_id != self.active_grid_cell_id
            or capability.exact_operator_instance_digest != self.exact_operator_instance_digest
            or capability.ordered_support_block_keys != keys
        ):
            raise ValueError("support product and reliability capability bindings differ.")
        base = _probability_matrix(self.base_probabilities, "base_probabilities")
        per_block = np.asarray(self.per_block_support_probabilities, dtype=np.float64)
        if per_block.shape != (len(keys), *base.shape):
            raise ValueError("per-block support probabilities have the wrong shape.")
        if not np.isfinite(per_block).all() or np.any(per_block < 0.0):
            raise ValueError("per-block support probabilities are invalid.")
        if not np.allclose(per_block.sum(axis=2), 1.0, rtol=0.0, atol=1.0e-12):
            raise ValueError("per-block support probabilities must sum to one.")
        support = _probability_matrix(self.support_probabilities, "support_probabilities")
        if support.shape != base.shape:
            raise ValueError("support probability shape differs from the anchor.")
        weights = _probability_vector(
            self.normalized_block_weights, len(keys), "normalized_block_weights"
        )
        reliabilities = _positive_vector(
            self.block_reliabilities, len(keys), "block_reliabilities"
        )
        if not np.array_equal(reliabilities, np.asarray(capability.block_reliabilities)):
            raise ValueError("support-product reliabilities differ from the capability.")
        if self.block_weight_mode == "reliability":
            expected_weights = reliabilities / reliabilities.sum()
        elif self.block_weight_mode == "uniform":
            expected_weights = np.full(len(keys), 1.0 / len(keys), dtype=np.float64)
        else:
            raise ValueError("unknown block weight mode.")
        if not np.array_equal(weights, expected_weights):
            raise ValueError("normalized block weights are not the exact relative reliabilities.")
        expected_support = np.sum(weights[:, None, None] * per_block, axis=0)
        if not np.array_equal(support, expected_support):
            raise ValueError("support probability is not the exact blockwise P3 mixture.")
        for array in (
            self.base_probabilities,
            self.per_block_support_probabilities,
            self.support_probabilities,
            self.block_reliabilities,
            self.normalized_block_weights,
        ):
            if not isinstance(array, np.ndarray) or array.flags.writeable:
                raise ValueError("support-product arrays must be immutable NumPy arrays.")
        _validate_capability_hash(self, self._payload())

    def _payload(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "query_key": self.query_key,
            "active_grid_cell_id": self.active_grid_cell_id,
            "exact_operator_instance_digest": self.exact_operator_instance_digest,
            "ordered_support_block_keys": list(self.ordered_support_block_keys),
            "query_normalizer_sha256": self.query_normalizer_sha256,
            "base_probabilities_sha256": _array_sha256(self.base_probabilities),
            "per_block_support_probabilities_sha256": _array_sha256(
                self.per_block_support_probabilities
            ),
            "support_probabilities_sha256": _array_sha256(self.support_probabilities),
            "block_reliabilities": self.block_reliabilities.tolist(),
            "normalized_block_weights": self.normalized_block_weights.tolist(),
            "support_reliability_capability_sha256": self.reliability_capability.payload_sha256,
            "block_weight_mode": self.block_weight_mode,
        }


def require_m_free_support_reliability(
    value: object,
) -> MFreeSupportReliabilityCapability:
    if type(value) is not MFreeSupportReliabilityCapability:
        raise TypeError("an exact MFreeSupportReliabilityCapability is required.")
    if value._issuer is not _SUPPORT_RELIABILITY_ISSUER:
        raise TypeError("support-reliability capability issuer is invalid.")
    MFreeSupportReliabilityCapability.__post_init__(value)
    return value


def require_v3_support_product(value: object) -> V3SupportProduct:
    if type(value) is not V3SupportProduct:
        raise TypeError("an exact V3SupportProduct is required.")
    if value._issuer is not _SUPPORT_PRODUCT_ISSUER:
        raise TypeError("support-product issuer is invalid.")
    V3SupportProduct.__post_init__(value)
    return value


def blockwise_p3_support(
    query_fbcca_scores: np.ndarray,
    support_fbcca_scores_by_block: np.ndarray,
    support_labels_by_block: np.ndarray,
    *,
    query_key: str,
    ordered_support_block_keys: Sequence[str],
    support_eeg_label_manifest_sha256: str,
    config: V3OperatorConfig,
) -> V3SupportProduct:
    """Build the common M-free blockwise P3 mixture and reliability capability."""

    return _blockwise_p3_support(
        query_fbcca_scores,
        support_fbcca_scores_by_block,
        support_labels_by_block,
        query_key=query_key,
        ordered_support_block_keys=ordered_support_block_keys,
        support_eeg_label_manifest_sha256=support_eeg_label_manifest_sha256,
        config=config,
        block_weight_mode="reliability",
    )


def blockwise_p3_support_uniform_sensitivity(
    query_fbcca_scores: np.ndarray,
    support_fbcca_scores_by_block: np.ndarray,
    support_labels_by_block: np.ndarray,
    *,
    query_key: str,
    ordered_support_block_keys: Sequence[str],
    support_eeg_label_manifest_sha256: str,
    config: V3OperatorConfig,
) -> V3SupportProduct:
    """Build the predeclared uniform-block-weight sensitivity product."""

    return _blockwise_p3_support(
        query_fbcca_scores,
        support_fbcca_scores_by_block,
        support_labels_by_block,
        query_key=query_key,
        ordered_support_block_keys=ordered_support_block_keys,
        support_eeg_label_manifest_sha256=support_eeg_label_manifest_sha256,
        config=config,
        block_weight_mode="uniform",
    )


def _blockwise_p3_support(
    query_fbcca_scores: np.ndarray,
    support_fbcca_scores_by_block: np.ndarray,
    support_labels_by_block: np.ndarray,
    *,
    query_key: str,
    ordered_support_block_keys: Sequence[str],
    support_eeg_label_manifest_sha256: str,
    config: V3OperatorConfig,
    block_weight_mode: Literal["reliability", "uniform"],
) -> V3SupportProduct:

    config = _require_exact_operator_config(config)
    query_key = _nonempty_ascii(query_key, "query_key")
    keys = _keys_tuple(ordered_support_block_keys, "ordered_support_block_keys")
    _sha256(support_eeg_label_manifest_sha256, "support_eeg_label_manifest_sha256")
    query_scores = _finite_matrix(query_fbcca_scores, "query_fbcca_scores")
    support_scores = np.asarray(support_fbcca_scores_by_block, dtype=np.float64)
    labels = np.asarray(support_labels_by_block)
    if support_scores.ndim != 3:
        raise ValueError("support scores must have shape [block, trial, class].")
    blocks, trials, classes = support_scores.shape
    if blocks == 0 or len(keys) != blocks:
        raise ValueError("support block count must be positive and match the ordered keys.")
    if query_scores.shape[1] != classes or trials != classes:
        raise ValueError("each complete support block must contain one trial per class.")
    if labels.shape != (blocks, trials):
        raise ValueError("support labels must have shape [block, class-balanced trial].")
    if not np.isfinite(support_scores).all():
        raise ValueError("support scores must contain only finite values.")
    label_values = _integer_labels(labels, classes, "support_labels_by_block")
    for block_index in range(blocks):
        if not np.array_equal(np.sort(label_values[block_index]), np.arange(classes)):
            raise ValueError("each support block must contain every class exactly once.")

    _, base = normalize_fbcca_scores(query_scores, epsilon=config.score_epsilon)
    support_probabilities = np.empty_like(support_scores, dtype=np.float64)
    normalizer_hashes: list[str] = []
    for block_index in range(blocks):
        _, block_probability = normalize_fbcca_scores(
            support_scores[block_index], epsilon=config.score_epsilon
        )
        support_probabilities[block_index] = block_probability
        normalizer_hashes.append(_array_sha256(block_probability))

    smoothing = config.ideal_prototype_smoothing
    prior = config.grid_cell.prototype_prior_pseudocount
    per_block = np.empty((blocks, len(query_scores), classes), dtype=np.float64)
    reliabilities = np.empty(blocks, dtype=np.float64)
    for block_index in range(blocks):
        observed = support_probabilities[block_index]
        block_labels = label_values[block_index]
        reliabilities[block_index] = np.clip(
            np.mean(observed[np.arange(trials), block_labels]), 1.0e-12, 1.0
        )
        prototypes = np.empty((classes, classes), dtype=np.float64)
        for class_index in range(classes):
            support_row = int(np.flatnonzero(block_labels == class_index)[0])
            ideal = np.full(classes, smoothing / classes, dtype=np.float64)
            ideal[class_index] += 1.0 - smoothing
            prototypes[class_index] = (
                prior * ideal + observed[support_row]
            ) / (prior + 1.0)
        scores = np.empty((len(base), classes), dtype=np.float64)
        for class_index in range(classes):
            tiled = np.broadcast_to(prototypes[class_index], base.shape)
            scores[:, class_index] = -jensen_shannon_divergence(
                base, tiled, epsilon=config.score_epsilon
            )
        per_block[block_index] = _softmax(scores)

    if block_weight_mode == "reliability":
        weights = reliabilities / reliabilities.sum()
    elif block_weight_mode == "uniform":
        weights = np.full(blocks, 1.0 / blocks, dtype=np.float64)
    else:  # pragma: no cover - callers pass one literal
        raise ValueError("unknown block weight mode.")
    mixture = np.sum(weights[:, None, None] * per_block, axis=0)
    readonly_reliabilities = _readonly(reliabilities)
    readonly_weights = _readonly(weights)
    readonly_per_block = _readonly(per_block)
    readonly_mixture = _readonly(mixture)
    reliability_payload = {
        "schema": SUPPORT_RELIABILITY_SCHEMA,
        "candidate_id": CANDIDATE_ID,
        "active_grid_cell_id": config.grid_cell.grid_cell_id,
        "exact_operator_instance_digest": config.grid_cell.operator_instance_sha256,
        "anchor_normalizer_sha256s": normalizer_hashes,
        "ordered_support_block_keys": list(keys),
        "block_reliabilities": readonly_reliabilities.tolist(),
        "support_eeg_label_manifest_sha256": support_eeg_label_manifest_sha256,
    }
    reliability = _issue_support_reliability(
        {
            "candidate_id": CANDIDATE_ID,
            "active_grid_cell_id": config.grid_cell.grid_cell_id,
            "exact_operator_instance_digest": (
                config.grid_cell.operator_instance_sha256
            ),
            "anchor_normalizer_sha256s": tuple(normalizer_hashes),
            "ordered_support_block_keys": keys,
            "block_reliabilities": tuple(
                float(value) for value in readonly_reliabilities
            ),
            "support_eeg_label_manifest_sha256": (
                support_eeg_label_manifest_sha256
            ),
            "payload_sha256": canonical_payload_sha256(reliability_payload),
            "schema": SUPPORT_RELIABILITY_SCHEMA,
        },
        _issuer=_SUPPORT_RELIABILITY_ISSUER,
    )
    product_payload = {
        "schema": SUPPORT_PRODUCT_SCHEMA,
        "query_key": query_key,
        "active_grid_cell_id": config.grid_cell.grid_cell_id,
        "exact_operator_instance_digest": config.grid_cell.operator_instance_sha256,
        "ordered_support_block_keys": list(keys),
        "query_normalizer_sha256": _array_sha256(base),
        "base_probabilities_sha256": _array_sha256(base),
        "per_block_support_probabilities_sha256": _array_sha256(readonly_per_block),
        "support_probabilities_sha256": _array_sha256(readonly_mixture),
        "block_reliabilities": readonly_reliabilities.tolist(),
        "normalized_block_weights": readonly_weights.tolist(),
        "support_reliability_capability_sha256": reliability.payload_sha256,
        "block_weight_mode": block_weight_mode,
    }
    return _issue_support_product(
        {
            "query_key": query_key,
            "active_grid_cell_id": config.grid_cell.grid_cell_id,
            "exact_operator_instance_digest": (
                config.grid_cell.operator_instance_sha256
            ),
            "ordered_support_block_keys": keys,
            "query_normalizer_sha256": _array_sha256(base),
            "base_probabilities": base,
            "per_block_support_probabilities": readonly_per_block,
            "support_probabilities": readonly_mixture,
            "block_reliabilities": readonly_reliabilities,
            "normalized_block_weights": readonly_weights,
            "reliability_capability": reliability,
            "block_weight_mode": block_weight_mode,
            "payload_sha256": canonical_payload_sha256(product_payload),
            "schema": SUPPORT_PRODUCT_SCHEMA,
        },
        _issuer=_SUPPORT_PRODUCT_ISSUER,
    )


def _issue_support_reliability(
    fields: Mapping[str, Any],
    *,
    _issuer: object,
) -> MFreeSupportReliabilityCapability:
    if _issuer is not _SUPPORT_RELIABILITY_ISSUER:
        raise TypeError("support-reliability issuer token is invalid.")
    value = object.__new__(MFreeSupportReliabilityCapability)
    for name, item in fields.items():
        object.__setattr__(value, name, item)
    object.__setattr__(value, "_issuer", _SUPPORT_RELIABILITY_ISSUER)
    return require_m_free_support_reliability(value)


def _issue_support_product(
    fields: Mapping[str, Any],
    *,
    _issuer: object,
) -> V3SupportProduct:
    if _issuer is not _SUPPORT_PRODUCT_ISSUER:
        raise TypeError("support-product issuer token is invalid.")
    value = object.__new__(V3SupportProduct)
    for name, item in fields.items():
        object.__setattr__(value, name, item)
    object.__setattr__(value, "_issuer", _SUPPORT_PRODUCT_ISSUER)
    return require_v3_support_product(value)


@dataclass(frozen=True)
class ContextScaleTable:
    """One immutable interface/channel robust center-and-scale table."""

    interface_lookup_key: str
    centers: tuple[float | None, ...]
    scales: tuple[float | None, ...]
    observed_counts: tuple[int, ...]

    def __post_init__(self) -> None:
        _nonempty_ascii(self.interface_lookup_key, "interface_lookup_key")
        if (
            type(self.centers) is not tuple
            or type(self.scales) is not tuple
            or type(self.observed_counts) is not tuple
        ):
            raise TypeError("context scale vectors must remain exact tuples.")
        if not (
            len(self.centers) == len(self.scales) == len(self.observed_counts) > 0
        ):
            raise ValueError("context scale table fields must have one common positive length.")
        for index, (center, scale, count) in enumerate(
            zip(self.centers, self.scales, self.observed_counts, strict=True)
        ):
            if type(count) is not int:
                raise TypeError(f"observed_counts[{index}] must be an exact int.")
            _nonnegative_integer(count, f"observed_counts[{index}]")
            if (center is None) != (scale is None):
                raise ValueError("context center and scale must be available as a pair.")
            if center is not None:
                _finite_float(center, f"centers[{index}]")
                if _positive_finite(scale, f"scales[{index}]") < 0.05:
                    raise ValueError("every available context scale must be at least 0.05.")


@dataclass(frozen=True)
class ContextReference:
    """Outcome-free, domain-scoped frozen context reference."""

    domain: str
    canonical_channels: tuple[str, ...]
    interface_tables: tuple[ContextScaleTable, ...]
    pooled_table: ContextScaleTable
    minimum_per_interface_channel_count: int
    pooled_per_channel_fallback_minimum_count: int
    scale_floor: float
    payload_sha256: str
    schema: str = CONTEXT_REFERENCE_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != CONTEXT_REFERENCE_SCHEMA:
            raise ValueError("context-reference schema is invalid.")
        _nonempty_ascii(self.domain, "domain")
        if type(self.canonical_channels) is not tuple:
            raise TypeError("canonical_channels must remain an exact tuple.")
        channels = _keys_tuple(self.canonical_channels, "canonical_channels")
        if channels != CANONICAL_CHANNELS:
            raise ValueError("V3 canonical channels must be exactly ch00 through ch07.")
        if type(self.interface_tables) is not tuple or not self.interface_tables:
            raise ValueError("context reference requires at least one interface table.")
        if any(type(table) is not ContextScaleTable for table in self.interface_tables):
            raise TypeError("interface_tables must contain exact ContextScaleTable objects.")
        if len({table.interface_lookup_key for table in self.interface_tables}) != len(
            self.interface_tables
        ):
            raise ValueError("context interface lookup keys must be unique.")
        if any(
            table.interface_lookup_key == "__pooled__"
            for table in self.interface_tables
        ):
            raise ValueError("__pooled__ is reserved for the pooled context table.")
        if any(len(table.centers) != len(channels) for table in self.interface_tables):
            raise ValueError("every context interface table must match canonical channels.")
        if type(self.pooled_table) is not ContextScaleTable:
            raise TypeError("pooled_table must be an exact ContextScaleTable.")
        if self.pooled_table.interface_lookup_key != "__pooled__":
            raise ValueError("pooled context table must use the reserved __pooled__ key.")
        if len(self.pooled_table.centers) != len(channels):
            raise ValueError("pooled context table must match canonical channels.")
        if _positive_integer(
            self.minimum_per_interface_channel_count,
            "minimum_per_interface_channel_count",
        ) != 128:
            raise ValueError("V3 per-interface minimum count is frozen at 128.")
        if _positive_integer(
            self.pooled_per_channel_fallback_minimum_count,
            "pooled_per_channel_fallback_minimum_count",
        ) != 256:
            raise ValueError("V3 pooled minimum count is frozen at 256.")
        if _positive_finite(self.scale_floor, "scale_floor") != 0.05:
            raise ValueError("V3 context scale floor is frozen at 0.05.")
        for table in self.interface_tables:
            for center, scale, count in zip(
                table.centers,
                table.scales,
                table.observed_counts,
                strict=True,
            ):
                available = center is not None and scale is not None
                if available != (count >= self.minimum_per_interface_channel_count):
                    raise ValueError(
                        "interface center/scale availability must exactly follow count>=128."
                    )
        for center, scale, count in zip(
            self.pooled_table.centers,
            self.pooled_table.scales,
            self.pooled_table.observed_counts,
            strict=True,
        ):
            available = center is not None and scale is not None
            if available != (count >= self.pooled_per_channel_fallback_minimum_count):
                raise ValueError(
                    "pooled center/scale availability must exactly follow count>=256."
                )
        _validate_capability_hash(self, self._payload())

    @property
    def table_by_interface(self) -> Mapping[str, ContextScaleTable]:
        return MappingProxyType(
            {table.interface_lookup_key: table for table in self.interface_tables}
        )

    def _payload(self) -> dict[str, Any]:
        def table_payload(table: ContextScaleTable) -> dict[str, Any]:
            return {
                "interface_lookup_key": table.interface_lookup_key,
                "centers": list(table.centers),
                "scales": list(table.scales),
                "observed_counts": list(table.observed_counts),
            }

        return {
            "schema": self.schema,
            "candidate_id": CANDIDATE_ID,
            "domain": self.domain,
            "canonical_channels": list(self.canonical_channels),
            "interface_tables": [table_payload(table) for table in self.interface_tables],
            "pooled_table": table_payload(self.pooled_table),
            "minimum_per_interface_channel_count": self.minimum_per_interface_channel_count,
            "pooled_per_channel_fallback_minimum_count": (
                self.pooled_per_channel_fallback_minimum_count
            ),
            "scale_floor": self.scale_floor,
        }


def _require_context_reference(value: object) -> ContextReference:
    if type(value) is not ContextReference:
        raise TypeError("reference must be an exact ContextReference.")
    ContextReference.__post_init__(value)
    return value


_VALIDATED_CONTEXT_REFERENCE_ISSUER = object()


@dataclass(frozen=True, slots=True, init=False)
class ValidatedContextReference:
    """Nominal proof that the exact context-reference semantics were checked.

    Instances can only be issued by
    :func:`validate_context_reference_for_publication`.  Publication code must
    call :func:`require_validated_context_reference` immediately before using
    the proof; a nominal instance alone is deliberately insufficient because
    Python permits low-level ``object.__new__`` construction.
    """

    schema: str
    candidate_id: str
    payload_schema: str
    payload_sha256: str
    domain: str
    canonical_channels: tuple[str, ...]
    interface_lookup_keys: tuple[str, ...]
    minimum_per_interface_channel_count: int
    pooled_per_channel_fallback_minimum_count: int
    scale_floor: float
    context_reference_rng_authority_schema: str
    context_reference_rng_authority_semantic_binding_sha256: str
    development_bundle_payload_sha256: str
    development_bundle_file_sha256: str
    development_bundle_source_bundle_sha256: str
    numerical_runtime_fingerprint_sha256: str
    reference_root_seed: int
    rng_key_map_sha256: str
    semantic_binding_sha256: str
    _rng_authority: ContextReferenceRNGAuthority = dataclass_field(
        repr=False,
        compare=False,
    )
    _issuer: object = dataclass_field(repr=False, compare=False)

    def __new__(cls, *_args: object, **_kwargs: object) -> Self:
        raise TypeError(
            "ValidatedContextReference is issued only by exact semantic validation."
        )


def _validated_context_reference_binding(
    value: ValidatedContextReference,
) -> dict[str, Any]:
    return {
        "schema": value.schema,
        "candidate_id": value.candidate_id,
        "payload_schema": value.payload_schema,
        "payload_sha256": value.payload_sha256,
        "domain": value.domain,
        "canonical_channels": list(value.canonical_channels),
        "interface_lookup_keys": list(value.interface_lookup_keys),
        "minimum_per_interface_channel_count": (
            value.minimum_per_interface_channel_count
        ),
        "pooled_per_channel_fallback_minimum_count": (
            value.pooled_per_channel_fallback_minimum_count
        ),
        "scale_floor": value.scale_floor,
        "context_reference_rng_authority_schema": (
            value.context_reference_rng_authority_schema
        ),
        "context_reference_rng_authority_semantic_binding_sha256": (
            value.context_reference_rng_authority_semantic_binding_sha256
        ),
        "development_bundle_payload_sha256": (
            value.development_bundle_payload_sha256
        ),
        "development_bundle_file_sha256": value.development_bundle_file_sha256,
        "development_bundle_source_bundle_sha256": (
            value.development_bundle_source_bundle_sha256
        ),
        "numerical_runtime_fingerprint_sha256": (
            value.numerical_runtime_fingerprint_sha256
        ),
        "reference_root_seed": value.reference_root_seed,
        "rng_key_map_sha256": value.rng_key_map_sha256,
    }


def _issue_validated_context_reference(
    reference: ContextReference,
    rng_authority: ContextReferenceRNGAuthority,
    *,
    _issuer: object,
) -> ValidatedContextReference:
    if _issuer is not _VALIDATED_CONTEXT_REFERENCE_ISSUER:
        raise TypeError("validated-reference issuer token is invalid.")
    reference = _require_context_reference(reference)
    authority = _require_context_reference_rng_authority_semantics(rng_authority)
    value = object.__new__(ValidatedContextReference)
    fields: dict[str, Any] = {
        "schema": VALIDATED_CONTEXT_REFERENCE_SCHEMA,
        "candidate_id": CANDIDATE_ID,
        "payload_schema": reference.schema,
        "payload_sha256": reference.payload_sha256,
        "domain": reference.domain,
        "canonical_channels": reference.canonical_channels,
        "interface_lookup_keys": tuple(
            table.interface_lookup_key for table in reference.interface_tables
        ),
        "minimum_per_interface_channel_count": (
            reference.minimum_per_interface_channel_count
        ),
        "pooled_per_channel_fallback_minimum_count": (
            reference.pooled_per_channel_fallback_minimum_count
        ),
        "scale_floor": reference.scale_floor,
        "context_reference_rng_authority_schema": authority.schema,
        "context_reference_rng_authority_semantic_binding_sha256": (
            authority.semantic_binding_sha256
        ),
        "development_bundle_payload_sha256": (
            authority.development_bundle_payload_sha256
        ),
        "development_bundle_file_sha256": authority.development_bundle_file_sha256,
        "development_bundle_source_bundle_sha256": (
            authority.development_bundle_source_bundle_sha256
        ),
        "numerical_runtime_fingerprint_sha256": (
            authority.numerical_runtime_fingerprint_sha256
        ),
        "reference_root_seed": authority.root_seed,
        "rng_key_map_sha256": authority.key_map_sha256,
        "_rng_authority": authority,
    }
    for name, item in fields.items():
        object.__setattr__(value, name, item)
    object.__setattr__(
        value,
        "semantic_binding_sha256",
        _canonical_plain_sha256(_validated_context_reference_binding(value)),
    )
    object.__setattr__(value, "_issuer", _VALIDATED_CONTEXT_REFERENCE_ISSUER)
    return _require_validated_context_reference_semantics(value)


def fit_context_reference(
    packets: Sequence[Mapping[str, Any]],
    *,
    domain: str,
    canonical_channels: Sequence[str] = CANONICAL_CHANNELS,
) -> ContextReference:
    """Fit robust context scales using acquisition covariates and nothing else.

    Every input mapping must contain exactly the three contract-authorized
    fields.  In particular this API has no EEG, label, prediction, or outcome
    parameter through which leakage could occur.
    """

    domain = _nonempty_ascii(domain, "domain")
    channels = _keys_tuple(canonical_channels, "canonical_channels")
    if len(channels) != 8:
        raise ValueError("V3 requires exactly eight canonical channels.")
    if isinstance(packets, (str, bytes)) or not isinstance(packets, Sequence) or not packets:
        raise ValueError("context-reference packets must be a non-empty sequence.")
    allowed = {"interface", "impedance_kohm_by_channel", "channel_availability"}
    grouped: dict[str, list[list[float]]] = {}
    pooled: list[list[float]] = [[] for _ in channels]
    for packet_index, packet in enumerate(packets):
        if not isinstance(packet, Mapping) or set(packet) != allowed:
            raise ValueError(
                f"context-reference packet {packet_index} must contain exactly {sorted(allowed)}."
            )
        interface = _nonempty_ascii(packet["interface"], f"packets[{packet_index}].interface")
        if interface == "__pooled__":
            raise ValueError("__pooled__ is reserved and cannot be a packet interface.")
        impedance = _sequence_length(
            packet["impedance_kohm_by_channel"],
            len(channels),
            f"packets[{packet_index}].impedance_kohm_by_channel",
        )
        availability = _boolean_sequence(
            packet["channel_availability"],
            len(channels),
            f"packets[{packet_index}].channel_availability",
        )
        by_channel = grouped.setdefault(interface, [[] for _ in channels])
        for channel_index, (raw_value, available) in enumerate(
            zip(impedance, availability, strict=True)
        ):
            resolved = _available_impedance(raw_value, available)
            if resolved is None:
                continue
            log_value = math.log1p(resolved)
            by_channel[channel_index].append(log_value)
            pooled[channel_index].append(log_value)

    def robust_table(
        interface: str,
        values_by_channel: Sequence[Sequence[float]],
        minimum: int,
    ) -> ContextScaleTable:
        centers: list[float | None] = []
        scales: list[float | None] = []
        counts: list[int] = []
        for values in values_by_channel:
            count = len(values)
            counts.append(count)
            if count < minimum:
                centers.append(None)
                scales.append(None)
                continue
            array = np.asarray(values, dtype=np.float64)
            center = float(np.median(array))
            raw_scale = float(1.4826 * np.median(np.abs(array - center)))
            centers.append(center)
            scales.append(max(raw_scale, 0.05))
        return ContextScaleTable(interface, tuple(centers), tuple(scales), tuple(counts))

    interface_tables = tuple(
        robust_table(interface, values, 128) for interface, values in grouped.items()
    )
    pooled_table = robust_table("__pooled__", pooled, 256)
    provisional = {
        "schema": CONTEXT_REFERENCE_SCHEMA,
        "candidate_id": CANDIDATE_ID,
        "domain": domain,
        "canonical_channels": list(channels),
        "interface_tables": [
            {
                "interface_lookup_key": table.interface_lookup_key,
                "centers": list(table.centers),
                "scales": list(table.scales),
                "observed_counts": list(table.observed_counts),
            }
            for table in interface_tables
        ],
        "pooled_table": {
            "interface_lookup_key": pooled_table.interface_lookup_key,
            "centers": list(pooled_table.centers),
            "scales": list(pooled_table.scales),
            "observed_counts": list(pooled_table.observed_counts),
        },
        "minimum_per_interface_channel_count": 128,
        "pooled_per_channel_fallback_minimum_count": 256,
        "scale_floor": 0.05,
    }
    return ContextReference(
        domain=domain,
        canonical_channels=channels,
        interface_tables=interface_tables,
        pooled_table=pooled_table,
        minimum_per_interface_channel_count=128,
        pooled_per_channel_fallback_minimum_count=256,
        scale_floor=0.05,
        payload_sha256=canonical_payload_sha256(provisional),
    )


def context_reference_payload(reference: ContextReference) -> dict[str, Any]:
    """Return a schema-ready in-memory payload for governance sealing."""

    reference = _require_context_reference(reference)
    payload = reference._payload()
    payload["payload_sha256"] = reference.payload_sha256
    validate_context_reference_payload(payload)
    return payload


def validate_context_reference_payload(payload: Mapping[str, Any]) -> ContextReference:
    """Purely validate and reconstruct an exact V3 context-reference payload."""

    expected = {
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
    _require_exact_keys(payload, expected, "context-reference payload")
    if payload["schema"] != CONTEXT_REFERENCE_SCHEMA or payload["candidate_id"] != CANDIDATE_ID:
        raise ValueError("context-reference payload identity is invalid.")
    _sha256(payload["payload_sha256"], "payload_sha256")
    if canonical_payload_sha256(payload) != payload["payload_sha256"]:
        raise ValueError("context-reference payload hash is invalid.")

    def parse_table(value: Any, name: str) -> ContextScaleTable:
        if not isinstance(value, Mapping):
            raise TypeError(f"{name} must be a mapping.")
        _require_exact_keys(
            value,
            {"interface_lookup_key", "centers", "scales", "observed_counts"},
            name,
        )
        if not all(isinstance(value[field], list) for field in ("centers", "scales", "observed_counts")):
            raise TypeError(f"{name} vector fields must be lists.")
        return ContextScaleTable(
            interface_lookup_key=value["interface_lookup_key"],
            centers=tuple(value["centers"]),
            scales=tuple(value["scales"]),
            observed_counts=tuple(value["observed_counts"]),
        )

    raw_tables = payload["interface_tables"]
    if not isinstance(raw_tables, list):
        raise TypeError("interface_tables must be a list.")
    return ContextReference(
        domain=payload["domain"],
        canonical_channels=tuple(payload["canonical_channels"]),
        interface_tables=tuple(
            parse_table(value, f"interface_tables[{index}]")
            for index, value in enumerate(raw_tables)
        ),
        pooled_table=parse_table(payload["pooled_table"], "pooled_table"),
        minimum_per_interface_channel_count=payload["minimum_per_interface_channel_count"],
        pooled_per_channel_fallback_minimum_count=payload[
            "pooled_per_channel_fallback_minimum_count"
        ],
        scale_floor=payload["scale_floor"],
        payload_sha256=payload["payload_sha256"],
    )


def validate_context_reference_for_publication(
    payload: Mapping[str, Any],
    *,
    rng_authority: ContextReferenceRNGAuthority,
) -> ValidatedContextReference:
    """Replay seed 20260910 exactly, byte-compare, and issue nominal proof."""

    authority = require_context_reference_rng_authority(rng_authority)
    try:
        from cfeg.analysis.metadata_calibration_v3_synthetic import (
            _replay_context_reference_payload_from_validated_authority,
        )
    except ImportError as error:  # pragma: no cover - integration installation failure
        raise RuntimeError("V3 synthetic context replay is unavailable.") from error
    replayed = _replay_context_reference_payload_from_validated_authority(authority)
    if canonical_json_bytes(payload) != canonical_json_bytes(replayed):
        raise ValueError("context-reference payload differs from exact authorized replay.")
    reference = validate_context_reference_payload(payload)
    if reference.domain != "synthetic_development_and_future_synthetic":
        raise ValueError("production reference proof requires the exact synthetic domain.")
    return _issue_validated_context_reference(
        reference,
        authority,
        _issuer=_VALIDATED_CONTEXT_REFERENCE_ISSUER,
    )


def require_validated_context_reference(
    value: object,
    *,
    expected_payload_sha256: str | None = None,
) -> ValidatedContextReference:
    """Require an authentic, internally consistent exact validation proof."""

    return _require_validated_context_reference(
        value,
        expected_payload_sha256=expected_payload_sha256,
        revalidate_bundle=True,
    )


def _require_validated_context_reference_semantics(
    value: object,
    *,
    expected_payload_sha256: str | None = None,
) -> ValidatedContextReference:
    """Cheap exact proof check after a public boundary revalidated its bundle."""

    return _require_validated_context_reference(
        value,
        expected_payload_sha256=expected_payload_sha256,
        revalidate_bundle=False,
    )


def _require_validated_context_reference(
    value: object,
    *,
    expected_payload_sha256: str | None,
    revalidate_bundle: bool,
) -> ValidatedContextReference:

    if type(value) is not ValidatedContextReference:
        raise TypeError("value must be an exact ValidatedContextReference.")
    try:
        issuer = value._issuer
    except AttributeError as error:
        raise TypeError("ValidatedContextReference was not issued by this module.") from error
    if issuer is not _VALIDATED_CONTEXT_REFERENCE_ISSUER:
        raise TypeError("ValidatedContextReference issuer marker is invalid.")
    if (
        value.schema != VALIDATED_CONTEXT_REFERENCE_SCHEMA
        or value.candidate_id != CANDIDATE_ID
        or value.payload_schema != CONTEXT_REFERENCE_SCHEMA
        or value.domain != "synthetic_development_and_future_synthetic"
        or value.context_reference_rng_authority_schema
        != CONTEXT_REFERENCE_RNG_AUTHORITY_SCHEMA
        or value.reference_root_seed != CONTEXT_REFERENCE_ROOT_SEED
        or value.rng_key_map_sha256 != rng_key_map_sha256()
    ):
        raise ValueError("ValidatedContextReference identity is invalid.")
    for name in (
        "payload_sha256",
        "context_reference_rng_authority_semantic_binding_sha256",
        "development_bundle_payload_sha256",
        "development_bundle_file_sha256",
        "development_bundle_source_bundle_sha256",
        "numerical_runtime_fingerprint_sha256",
        "rng_key_map_sha256",
    ):
        _sha256(getattr(value, name), name)
    _nonempty_ascii(value.domain, "domain")
    channels = _keys_tuple(value.canonical_channels, "canonical_channels")
    if len(channels) != 8:
        raise ValueError("validated context reference requires exactly eight channels.")
    interface_keys = _keys_tuple(value.interface_lookup_keys, "interface_lookup_keys")
    if interface_keys != ("neutral", "wet", "dry"):
        raise ValueError(
            "validated synthetic reference interfaces must be neutral, wet, dry in order."
        )
    if (
        _positive_integer(
            value.minimum_per_interface_channel_count,
            "minimum_per_interface_channel_count",
        )
        != 128
        or _positive_integer(
            value.pooled_per_channel_fallback_minimum_count,
            "pooled_per_channel_fallback_minimum_count",
        )
        != 256
        or _positive_finite(value.scale_floor, "scale_floor") != 0.05
    ):
        raise ValueError("validated context-reference constants are invalid.")
    _sha256(value.semantic_binding_sha256, "semantic_binding_sha256")
    expected_binding = _canonical_plain_sha256(
        _validated_context_reference_binding(value)
    )
    if value.semantic_binding_sha256 != expected_binding:
        raise ValueError("ValidatedContextReference semantic binding is invalid.")
    authority = (
        require_context_reference_rng_authority(value._rng_authority)
        if revalidate_bundle
        else _require_context_reference_rng_authority_semantics(
            value._rng_authority
        )
    )
    if (
        value.context_reference_rng_authority_semantic_binding_sha256
        != authority.semantic_binding_sha256
        or value.development_bundle_payload_sha256
        != authority.development_bundle_payload_sha256
        or value.development_bundle_file_sha256
        != authority.development_bundle_file_sha256
        or value.development_bundle_source_bundle_sha256
        != authority.development_bundle_source_bundle_sha256
        or value.numerical_runtime_fingerprint_sha256
        != authority.numerical_runtime_fingerprint_sha256
        or value.reference_root_seed != authority.root_seed
        or value.rng_key_map_sha256 != authority.key_map_sha256
    ):
        raise ValueError("validated reference differs from its RNG provenance authority.")
    if expected_payload_sha256 is not None:
        _sha256(expected_payload_sha256, "expected_payload_sha256")
        if value.payload_sha256 != expected_payload_sha256:
            raise ValueError("ValidatedContextReference binds a different payload.")
    return value


@dataclass(frozen=True)
class ContextPacket:
    """Validated raw adapter input; invalid channel values become unavailable."""

    packet_key: str
    interface_lookup_key: str | None
    impedance_kohm_by_channel: tuple[float | None, ...]
    channel_availability: tuple[bool, ...]

    def __post_init__(self) -> None:
        _nonempty_ascii(self.packet_key, "packet_key")
        if self.interface_lookup_key is not None:
            _nonempty_ascii(self.interface_lookup_key, "interface_lookup_key")
        if len(self.impedance_kohm_by_channel) != 8 or len(self.channel_availability) != 8:
            raise ValueError("V3 context packets require exactly eight channels.")
        for index, (value, available) in enumerate(
            zip(self.impedance_kohm_by_channel, self.channel_availability, strict=True)
        ):
            if type(available) is not bool:
                raise TypeError(f"channel_availability[{index}] must be an exact bool.")
            if value is not None:
                resolved = _finite_float(value, f"impedance_kohm_by_channel[{index}]")
                if resolved < 0.0:
                    raise ValueError("validated impedance cannot be negative.")
                if not available:
                    raise ValueError("an unavailable channel cannot retain impedance.")

    @classmethod
    def from_raw(cls, value: Mapping[str, Any], *, name: str = "context packet") -> ContextPacket:
        expected = {
            "packet_key",
            "interface_lookup_key",
            "impedance_kohm_by_channel",
            "channel_availability",
        }
        _require_exact_keys(value, expected, name)
        impedance = _sequence_length(
            value["impedance_kohm_by_channel"], 8, f"{name}.impedance_kohm_by_channel"
        )
        availability = _boolean_sequence(
            value["channel_availability"], 8, f"{name}.channel_availability"
        )
        cleaned: list[float | None] = []
        cleaned_availability: list[bool] = []
        for raw, available in zip(impedance, availability, strict=True):
            resolved = _available_impedance(raw, available)
            cleaned.append(resolved)
            cleaned_availability.append(resolved is not None)
        interface = value["interface_lookup_key"]
        if interface is not None:
            interface = _nonempty_ascii(interface, f"{name}.interface_lookup_key")
        return cls(
            packet_key=_nonempty_ascii(value["packet_key"], f"{name}.packet_key"),
            interface_lookup_key=interface,
            impedance_kohm_by_channel=tuple(cleaned),
            channel_availability=tuple(cleaned_availability),
        )


def context_pairing_sha256(
    *,
    query_key: str,
    ordered_support_block_keys: Sequence[str],
    query_packet_key: str,
    ordered_support_packet_keys: Sequence[str],
) -> str:
    """Bind logical EEG slots to the exact raw context-packet assignment."""

    slots = _keys_tuple(ordered_support_block_keys, "ordered_support_block_keys")
    packets = _keys_tuple(ordered_support_packet_keys, "ordered_support_packet_keys")
    if len(slots) != len(packets):
        raise ValueError("support slots and context packets must have the same length.")
    return _canonical_plain_sha256(
        {
            "schema": PAIRING_SCHEMA,
            "candidate_id": CANDIDATE_ID,
            "query_key": _nonempty_ascii(query_key, "query_key"),
            "ordered_support_block_keys": list(slots),
            "query_packet_key": _nonempty_ascii(query_packet_key, "query_packet_key"),
            "ordered_support_packet_keys": list(packets),
        }
    )


_CONTEXT_PREFLIGHT_ISSUER = object()


@dataclass(frozen=True, slots=True, init=False)
class ContextPreflightCapability:
    candidate_id: str
    query_key: str
    ordered_support_block_keys: tuple[str, ...]
    affinity_by_support_block: tuple[float, ...]
    comparable_channel_counts: tuple[int, ...]
    decision_reason: str
    context_reference_receipt_sha256: str
    pairing_sha256: str
    payload_sha256: str
    schema: str = CONTEXT_PREFLIGHT_SCHEMA
    _issuer: object = dataclass_field(repr=False, compare=False)

    def __new__(cls, *_args: object, **_kwargs: object) -> Self:
        raise TypeError("ContextPreflightCapability has no public constructor.")

    def __post_init__(self) -> None:
        if self.schema != CONTEXT_PREFLIGHT_SCHEMA or self.candidate_id != CANDIDATE_ID:
            raise ValueError("context-preflight capability identity is invalid.")
        _nonempty_ascii(self.query_key, "query_key")
        if (
            type(self.ordered_support_block_keys) is not tuple
            or type(self.affinity_by_support_block) is not tuple
            or type(self.comparable_channel_counts) is not tuple
        ):
            raise TypeError("context-preflight vectors must remain exact tuples.")
        keys = _keys_tuple(self.ordered_support_block_keys, "ordered_support_block_keys")
        if len(self.affinity_by_support_block) != len(keys) or len(
            self.comparable_channel_counts
        ) != len(keys):
            raise ValueError("preflight affinity/count vectors must match support blocks.")
        for index, affinity in enumerate(self.affinity_by_support_block):
            resolved = _finite_float(affinity, f"affinity_by_support_block[{index}]")
            if not 0.0 <= resolved <= 1.0:
                raise ValueError("context affinity must be in [0, 1].")
        for index, count in enumerate(self.comparable_channel_counts):
            if _nonnegative_integer(count, f"comparable_channel_counts[{index}]") > 8:
                raise ValueError("comparable context channel count cannot exceed eight.")
        if self.decision_reason not in {
            "ready",
            "all_missing_exact_A_Q",
            "malformed_packet_exact_A0",
            "pairing_mismatch_exact_A0",
            "cross_interface_exact_A0",
            "query_context_OOD_exact_A0",
        }:
            raise ValueError("unknown context preflight decision reason.")
        _sha256(self.context_reference_receipt_sha256, "context_reference_receipt_sha256")
        _sha256(self.pairing_sha256, "pairing_sha256")
        _validate_capability_hash(self, self._payload())

    @property
    def rejects_before_support_access(self) -> bool:
        return self.decision_reason.endswith("exact_A0")

    def _payload(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "candidate_id": self.candidate_id,
            "query_key": self.query_key,
            "ordered_support_block_keys": list(self.ordered_support_block_keys),
            "affinity_by_support_block": list(self.affinity_by_support_block),
            "comparable_channel_counts": list(self.comparable_channel_counts),
            "decision_reason": self.decision_reason,
            "context_reference_receipt_sha256": self.context_reference_receipt_sha256,
            "pairing_sha256": self.pairing_sha256,
        }


def require_context_preflight(value: object) -> ContextPreflightCapability:
    if type(value) is not ContextPreflightCapability:
        raise TypeError("an exact ContextPreflightCapability is required.")
    if value._issuer is not _CONTEXT_PREFLIGHT_ISSUER:
        raise TypeError("context-preflight capability issuer is invalid.")
    ContextPreflightCapability.__post_init__(value)
    return value


def preflight_context(
    *,
    reference: ContextReference,
    validated_reference: ValidatedContextReference,
    query_key: str,
    ordered_support_block_keys: Sequence[str],
    query_packet: Mapping[str, Any] | ContextPacket,
    support_packets: Sequence[Mapping[str, Any] | ContextPacket],
    pairing_sha256: str,
    lookup_mode: ContextLookupMode = "interface",
    wrong_interface_lookup_key: str | None = None,
) -> ContextPreflightCapability:
    """Validate raw context using a nominally authorized reference receipt."""

    return _preflight_context_with_validated_reference(
        reference=reference,
        validated_reference=validated_reference,
        query_key=query_key,
        ordered_support_block_keys=ordered_support_block_keys,
        query_packet=query_packet,
        support_packets=support_packets,
        pairing_sha256=pairing_sha256,
        lookup_mode=lookup_mode,
        wrong_interface_lookup_key=wrong_interface_lookup_key,
        revalidate_bundle=True,
    )


def _preflight_context_after_authorized_boundary(
    *,
    reference: ContextReference,
    validated_reference: ValidatedContextReference,
    query_key: str,
    ordered_support_block_keys: Sequence[str],
    query_packet: Mapping[str, Any] | ContextPacket,
    support_packets: Sequence[Mapping[str, Any] | ContextPacket],
    pairing_sha256: str,
    lookup_mode: ContextLookupMode = "interface",
    wrong_interface_lookup_key: str | None = None,
) -> ContextPreflightCapability:
    """Cheap per-query proof validation after one public governance check."""

    return _preflight_context_with_validated_reference(
        reference=reference,
        validated_reference=validated_reference,
        query_key=query_key,
        ordered_support_block_keys=ordered_support_block_keys,
        query_packet=query_packet,
        support_packets=support_packets,
        pairing_sha256=pairing_sha256,
        lookup_mode=lookup_mode,
        wrong_interface_lookup_key=wrong_interface_lookup_key,
        revalidate_bundle=False,
    )


def _preflight_context_with_validated_reference(
    *,
    reference: ContextReference,
    validated_reference: ValidatedContextReference,
    query_key: str,
    ordered_support_block_keys: Sequence[str],
    query_packet: Mapping[str, Any] | ContextPacket,
    support_packets: Sequence[Mapping[str, Any] | ContextPacket],
    pairing_sha256: str,
    lookup_mode: ContextLookupMode,
    wrong_interface_lookup_key: str | None,
    revalidate_bundle: bool,
) -> ContextPreflightCapability:

    reference = _require_context_reference(reference)
    proof = _require_validated_context_reference(
        validated_reference,
        expected_payload_sha256=reference.payload_sha256,
        revalidate_bundle=revalidate_bundle,
    )
    if (
        proof.domain != reference.domain
        or proof.canonical_channels != reference.canonical_channels
        or proof.interface_lookup_keys
        != tuple(table.interface_lookup_key for table in reference.interface_tables)
    ):
        raise ValueError("validated context-reference semantics differ from reference.")
    return _preflight_context_impl(
        reference=reference,
        query_key=query_key,
        ordered_support_block_keys=ordered_support_block_keys,
        query_packet=query_packet,
        support_packets=support_packets,
        pairing_sha256=pairing_sha256,
        lookup_mode=lookup_mode,
        wrong_interface_lookup_key=wrong_interface_lookup_key,
    )


def _preflight_context_unit(
    *,
    reference: ContextReference,
    query_key: str,
    ordered_support_block_keys: Sequence[str],
    query_packet: Mapping[str, Any] | ContextPacket,
    support_packets: Sequence[Mapping[str, Any] | ContextPacket],
    pairing_sha256: str,
    lookup_mode: ContextLookupMode = "interface",
    wrong_interface_lookup_key: str | None = None,
) -> ContextPreflightCapability:
    """Explicit unit-fixture seam that carries no publication authority."""

    return _preflight_context_impl(
        reference=reference,
        query_key=query_key,
        ordered_support_block_keys=ordered_support_block_keys,
        query_packet=query_packet,
        support_packets=support_packets,
        pairing_sha256=pairing_sha256,
        lookup_mode=lookup_mode,
        wrong_interface_lookup_key=wrong_interface_lookup_key,
    )


def _preflight_context_impl(
    *,
    reference: ContextReference,
    query_key: str,
    ordered_support_block_keys: Sequence[str],
    query_packet: Mapping[str, Any] | ContextPacket,
    support_packets: Sequence[Mapping[str, Any] | ContextPacket],
    pairing_sha256: str,
    lookup_mode: ContextLookupMode = "interface",
    wrong_interface_lookup_key: str | None = None,
) -> ContextPreflightCapability:
    """Shared pure implementation; callers choose production or unit authority."""

    reference = _require_context_reference(reference)
    query_key = _nonempty_ascii(query_key, "query_key")
    slots = _keys_tuple(ordered_support_block_keys, "ordered_support_block_keys")
    if not isinstance(support_packets, Sequence) or isinstance(support_packets, (str, bytes)):
        return _rejected_preflight(
            reference,
            query_key,
            slots,
            pairing_sha256,
            "malformed_packet_exact_A0",
            _issuer=_CONTEXT_PREFLIGHT_ISSUER,
        )
    if len(support_packets) != len(slots):
        return _rejected_preflight(
            reference,
            query_key,
            slots,
            pairing_sha256,
            "malformed_packet_exact_A0",
            _issuer=_CONTEXT_PREFLIGHT_ISSUER,
        )
    if lookup_mode not in {"interface", "pooled", "wrong_interface"}:
        return _rejected_preflight(
            reference,
            query_key,
            slots,
            pairing_sha256,
            "malformed_packet_exact_A0",
            _issuer=_CONTEXT_PREFLIGHT_ISSUER,
        )
    try:
        query = (
            query_packet
            if type(query_packet) is ContextPacket
            else ContextPacket.from_raw(query_packet, name="query_packet")
        )
        supports = tuple(
            packet
            if type(packet) is ContextPacket
            else ContextPacket.from_raw(packet, name=f"support_packets[{index}]")
            for index, packet in enumerate(support_packets)
        )
        supplied_pairing = _sha256(pairing_sha256, "pairing_sha256")
    except (TypeError, ValueError, OverflowError):
        return _rejected_preflight(
            reference,
            query_key,
            slots,
            pairing_sha256,
            "malformed_packet_exact_A0",
            _issuer=_CONTEXT_PREFLIGHT_ISSUER,
        )

    expected_pairing = context_pairing_sha256(
        query_key=query_key,
        ordered_support_block_keys=slots,
        query_packet_key=query.packet_key,
        ordered_support_packet_keys=tuple(packet.packet_key for packet in supports),
    )
    if supplied_pairing != expected_pairing:
        return _rejected_preflight(
            reference,
            query_key,
            slots,
            supplied_pairing,
            "pairing_mismatch_exact_A0",
            _issuer=_CONTEXT_PREFLIGHT_ISSUER,
        )
    if query.packet_key != query_key:
        return _rejected_preflight(
            reference,
            query_key,
            slots,
            supplied_pairing,
            "pairing_mismatch_exact_A0",
            _issuer=_CONTEXT_PREFLIGHT_ISSUER,
        )
    if query.interface_lookup_key is not None and any(
        packet.interface_lookup_key not in {None, query.interface_lookup_key} for packet in supports
    ):
        return _rejected_preflight(
            reference,
            query_key,
            slots,
            supplied_pairing,
            "cross_interface_exact_A0",
            _issuer=_CONTEXT_PREFLIGHT_ISSUER,
        )

    try:
        centers, scales = _lookup_context_scale(
            reference,
            query.interface_lookup_key,
            lookup_mode=lookup_mode,
            wrong_interface_lookup_key=wrong_interface_lookup_key,
        )
    except (TypeError, ValueError):
        return _rejected_preflight(
            reference,
            query_key,
            slots,
            supplied_pairing,
            "malformed_packet_exact_A0",
            _issuer=_CONTEXT_PREFLIGHT_ISSUER,
        )
    query_z = _standardized_packet(query, centers, scales)
    valid_query_z = tuple(value for value in query_z if value is not None)
    if valid_query_z and sum(abs(value) > 8.0 for value in valid_query_z) > len(valid_query_z) / 2:
        comparable_counts = tuple(
            sum(
                query_value is not None and support_value is not None
                for query_value, support_value in zip(
                    query_z,
                    _standardized_packet(packet, centers, scales),
                    strict=True,
                )
            )
            for packet in supports
        )
        return _issue_preflight(
            reference=reference,
            query_key=query_key,
            slots=slots,
            affinities=tuple(1.0 for _ in slots),
            comparable_counts=comparable_counts,
            decision_reason="query_context_OOD_exact_A0",
            pairing_sha256=supplied_pairing,
            _issuer=_CONTEXT_PREFLIGHT_ISSUER,
        )

    affinities: list[float] = []
    comparable_counts: list[int] = []
    for packet in supports:
        support_z = _standardized_packet(packet, centers, scales)
        differences = [
            abs(query_value - support_value)
            for query_value, support_value in zip(query_z, support_z, strict=True)
            if query_value is not None and support_value is not None
        ]
        comparable_counts.append(len(differences))
        if not differences:
            affinities.append(1.0)
        else:
            distance = float(np.median(np.asarray(differences, dtype=np.float64)))
            affinities.append(float(2.0 ** (-np.clip(distance, 0.0, 16.0))))
    reason = "all_missing_exact_A_Q" if not any(comparable_counts) else "ready"
    return _issue_preflight(
        reference=reference,
        query_key=query_key,
        slots=slots,
        affinities=tuple(affinities),
        comparable_counts=tuple(comparable_counts),
        decision_reason=reason,
        pairing_sha256=supplied_pairing,
        _issuer=_CONTEXT_PREFLIGHT_ISSUER,
    )


_CONTEXT_TRUST_ISSUER = object()


@dataclass(frozen=True, slots=True, init=False)
class ContextTrustCapability:
    candidate_id: str
    active_grid_cell_id: str
    exact_operator_instance_digest: str
    query_key: str
    ordered_support_block_keys: tuple[str, ...]
    g_M_by_query: float
    comparable_block_count_by_query: int
    decision_reason: str
    context_reference_receipt_sha256: str
    context_preflight_capability_sha256: str
    support_reliability_capability_sha256: str
    pairing_sha256: str
    payload_sha256: str
    schema: str = CONTEXT_TRUST_SCHEMA
    _issuer: object = dataclass_field(repr=False, compare=False)

    def __new__(cls, *_args: object, **_kwargs: object) -> Self:
        raise TypeError("ContextTrustCapability has no public constructor.")

    def __post_init__(self) -> None:
        if self.schema != CONTEXT_TRUST_SCHEMA or self.candidate_id != CANDIDATE_ID:
            raise ValueError("context-trust capability identity is invalid.")
        grid = grid_cell_by_id(self.active_grid_cell_id)
        if self.exact_operator_instance_digest != grid.operator_instance_sha256:
            raise ValueError("context-trust operator binding is invalid.")
        _nonempty_ascii(self.query_key, "query_key")
        if type(self.ordered_support_block_keys) is not tuple:
            raise TypeError("context-trust support keys must remain an exact tuple.")
        keys = _keys_tuple(self.ordered_support_block_keys, "ordered_support_block_keys")
        trust = _finite_float(self.g_M_by_query, "g_M_by_query")
        if not 0.0 <= trust <= 1.0:
            raise ValueError("g_M_by_query must be in [0, 1].")
        comparable = _nonnegative_integer(
            self.comparable_block_count_by_query, "comparable_block_count_by_query"
        )
        if comparable > len(keys):
            raise ValueError("comparable block count exceeds support depth.")
        if self.decision_reason not in {"context_weighted", "exact_neutral_A_Q"}:
            raise ValueError("unknown context-trust decision reason.")
        for name in (
            "context_reference_receipt_sha256",
            "context_preflight_capability_sha256",
            "support_reliability_capability_sha256",
            "pairing_sha256",
        ):
            _sha256(getattr(self, name), name)
        _validate_capability_hash(self, self._payload())

    def _payload(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "candidate_id": self.candidate_id,
            "active_grid_cell_id": self.active_grid_cell_id,
            "exact_operator_instance_digest": self.exact_operator_instance_digest,
            "query_key": self.query_key,
            "ordered_support_block_keys": list(self.ordered_support_block_keys),
            "g_M_by_query": self.g_M_by_query,
            "comparable_block_count_by_query": self.comparable_block_count_by_query,
            "decision_reason": self.decision_reason,
            "context_reference_receipt_sha256": self.context_reference_receipt_sha256,
            "context_preflight_capability_sha256": self.context_preflight_capability_sha256,
            "support_reliability_capability_sha256": (
                self.support_reliability_capability_sha256
            ),
            "pairing_sha256": self.pairing_sha256,
        }


def require_context_trust(value: object) -> ContextTrustCapability:
    if type(value) is not ContextTrustCapability:
        raise TypeError("an exact ContextTrustCapability is required.")
    if value._issuer is not _CONTEXT_TRUST_ISSUER:
        raise TypeError("context-trust capability issuer is invalid.")
    ContextTrustCapability.__post_init__(value)
    return value


def _issue_context_trust(
    fields: Mapping[str, Any],
    *,
    _issuer: object,
) -> ContextTrustCapability:
    if _issuer is not _CONTEXT_TRUST_ISSUER:
        raise TypeError("context-trust issuer token is invalid.")
    value = object.__new__(ContextTrustCapability)
    for name, item in fields.items():
        object.__setattr__(value, name, item)
    object.__setattr__(value, "_issuer", _issuer)
    return require_context_trust(value)


def finalize_context_trust(
    preflight: ContextPreflightCapability,
    support_reliability: MFreeSupportReliabilityCapability,
) -> ContextTrustCapability:
    """Combine the preflight and the only authorized M-free signal summary."""

    preflight = require_context_preflight(preflight)
    support_reliability = require_m_free_support_reliability(
        support_reliability
    )
    weights = np.asarray(support_reliability.block_reliabilities, dtype=np.float64)
    weights = weights / weights.sum()
    return _finalize_context_trust_with_weights(
        preflight,
        support_reliability,
        weights,
        _issuer=_CONTEXT_TRUST_ISSUER,
    )


def finalize_context_trust_uniform_sensitivity(
    preflight: ContextPreflightCapability,
    support_product: V3SupportProduct,
) -> ContextTrustCapability:
    """Apply uniform common weights to both P3 mixing and context affinity."""

    preflight = require_context_preflight(preflight)
    product = require_v3_support_product(support_product)
    if product.block_weight_mode != "uniform":
        raise ValueError("uniform sensitivity requires an exact uniform support product.")
    return _finalize_context_trust_with_weights(
        preflight,
        product.reliability_capability,
        product.normalized_block_weights,
        _issuer=_CONTEXT_TRUST_ISSUER,
    )


def _finalize_context_trust_with_weights(
    preflight: ContextPreflightCapability,
    support_reliability: MFreeSupportReliabilityCapability,
    block_weights: np.ndarray,
    *,
    _issuer: object,
) -> ContextTrustCapability:
    if _issuer is not _CONTEXT_TRUST_ISSUER:
        raise TypeError("context-trust finalizer issuer token is invalid.")
    preflight = require_context_preflight(preflight)
    support_reliability = require_m_free_support_reliability(
        support_reliability
    )
    if preflight.rejects_before_support_access:
        raise ValueError("a rejecting preflight cannot be finalized after support access.")
    if (
        preflight.candidate_id != support_reliability.candidate_id
        or preflight.ordered_support_block_keys
        != support_reliability.ordered_support_block_keys
    ):
        raise ValueError("preflight and support-reliability bindings differ.")
    affinity = np.asarray(preflight.affinity_by_support_block, dtype=np.float64)
    weights = _probability_vector(
        block_weights,
        len(preflight.ordered_support_block_keys),
        "block_weights",
    )
    trust = float(np.sum(weights * affinity))
    if np.array_equal(affinity, np.ones_like(affinity)) or trust == 1.0:
        trust = 1.0
        reason = "exact_neutral_A_Q"
    else:
        trust = float(np.clip(trust, 0.0, 1.0))
        reason = "context_weighted"
    payload = {
        "schema": CONTEXT_TRUST_SCHEMA,
        "candidate_id": CANDIDATE_ID,
        "active_grid_cell_id": support_reliability.active_grid_cell_id,
        "exact_operator_instance_digest": (
            support_reliability.exact_operator_instance_digest
        ),
        "query_key": preflight.query_key,
        "ordered_support_block_keys": list(preflight.ordered_support_block_keys),
        "g_M_by_query": trust,
        "comparable_block_count_by_query": sum(
            count > 0 for count in preflight.comparable_channel_counts
        ),
        "decision_reason": reason,
        "context_reference_receipt_sha256": preflight.context_reference_receipt_sha256,
        "context_preflight_capability_sha256": preflight.payload_sha256,
        "support_reliability_capability_sha256": support_reliability.payload_sha256,
        "pairing_sha256": preflight.pairing_sha256,
    }
    return _issue_context_trust(
        {
            "candidate_id": CANDIDATE_ID,
            "active_grid_cell_id": support_reliability.active_grid_cell_id,
            "exact_operator_instance_digest": (
                support_reliability.exact_operator_instance_digest
            ),
            "query_key": preflight.query_key,
            "ordered_support_block_keys": preflight.ordered_support_block_keys,
            "g_M_by_query": trust,
            "comparable_block_count_by_query": payload[
                "comparable_block_count_by_query"
            ],
            "decision_reason": reason,
            "context_reference_receipt_sha256": (
                preflight.context_reference_receipt_sha256
            ),
            "context_preflight_capability_sha256": preflight.payload_sha256,
            "support_reliability_capability_sha256": (
                support_reliability.payload_sha256
            ),
            "pairing_sha256": preflight.pairing_sha256,
            "payload_sha256": canonical_payload_sha256(payload),
            "schema": CONTEXT_TRUST_SCHEMA,
        },
        _issuer=_CONTEXT_TRUST_ISSUER,
    )


_PREQUENTIAL_GATE_ISSUER = object()


@dataclass(frozen=True, slots=True, init=False)
class PrequentialGateDecision:
    candidate_id: str
    active_grid_cell_id: str
    exact_operator_instance_digest: str
    budget: int
    enabled: bool
    evaluated_blocks: tuple[int, ...]
    fit_block_prefixes: tuple[tuple[int, ...], ...]
    block_balanced_accuracy_deltas: tuple[float, ...]
    block_log_probability_deltas: tuple[float, ...]
    block_partition_sha256s: tuple[str, ...]
    reason: str
    payload_sha256: str
    schema: str = GATE_SCHEMA
    _issuer: object = dataclass_field(repr=False, compare=False)

    def __new__(cls, *_args: object, **_kwargs: object) -> Self:
        raise TypeError("PrequentialGateDecision has no public constructor.")

    def __post_init__(self) -> None:
        if self.schema != GATE_SCHEMA or self.candidate_id != CANDIDATE_ID:
            raise ValueError("prequential gate identity is invalid.")
        grid = grid_cell_by_id(self.active_grid_cell_id)
        if self.exact_operator_instance_digest != grid.operator_instance_sha256:
            raise ValueError("prequential gate operator binding is invalid.")
        budget = _final_budget(self.budget)
        if budget not in PREQUENTIAL_BUDGETS:
            raise ValueError("prequential gates exist only at k=3 and k=5.")
        if type(self.enabled) is not bool:
            raise TypeError("prequential gate enabled must be an exact bool.")
        if any(
            type(value) is not tuple
            for value in (
                self.evaluated_blocks,
                self.fit_block_prefixes,
                self.block_balanced_accuracy_deltas,
                self.block_log_probability_deltas,
                self.block_partition_sha256s,
            )
        ) or any(type(value) is not tuple for value in self.fit_block_prefixes):
            raise TypeError("prequential gate vectors must remain exact tuples.")
        expected_evaluated = tuple(range(2, budget + 1))
        if self.evaluated_blocks != expected_evaluated:
            raise ValueError("prequential evaluated blocks are not the exact chronology.")
        expected_prefixes = tuple(tuple(range(1, block)) for block in expected_evaluated)
        if self.fit_block_prefixes != expected_prefixes:
            raise ValueError("prequential fit prefixes are not exact.")
        if not (
            len(self.block_balanced_accuracy_deltas)
            == len(self.block_log_probability_deltas)
            == len(expected_evaluated)
        ):
            raise ValueError("prequential metric vectors have the wrong length.")
        for name, values in (
            ("block_balanced_accuracy_deltas", self.block_balanced_accuracy_deltas),
            ("block_log_probability_deltas", self.block_log_probability_deltas),
        ):
            for index, value in enumerate(values):
                _finite_float(value, f"{name}[{index}]")
        if len(self.block_partition_sha256s) != budget:
            raise ValueError("one partition digest is required per authorized support block.")
        for index, digest in enumerate(self.block_partition_sha256s):
            _sha256(digest, f"block_partition_sha256s[{index}]")
        if self.reason not in {"all_prequential_requirements_pass", "prequential_abstention"}:
            raise ValueError("unknown prequential gate reason.")
        expected_enabled = (
            all(value >= 0.0 for value in self.block_balanced_accuracy_deltas)
            and all(value > 0.0 for value in self.block_log_probability_deltas)
            and any(value > 0.0 for value in self.block_balanced_accuracy_deltas)
        )
        if self.enabled != expected_enabled:
            raise ValueError("prequential enabled bit does not match its M-free metrics.")
        _validate_capability_hash(self, self._payload())

    def _payload(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "candidate_id": self.candidate_id,
            "active_grid_cell_id": self.active_grid_cell_id,
            "exact_operator_instance_digest": self.exact_operator_instance_digest,
            "budget": self.budget,
            "enabled": self.enabled,
            "evaluated_blocks": list(self.evaluated_blocks),
            "fit_block_prefixes": [list(values) for values in self.fit_block_prefixes],
            "block_balanced_accuracy_deltas": list(self.block_balanced_accuracy_deltas),
            "block_log_probability_deltas": list(self.block_log_probability_deltas),
            "block_partition_sha256s": list(self.block_partition_sha256s),
            "reason": self.reason,
        }


def require_prequential_gate(value: object) -> PrequentialGateDecision:
    if type(value) is not PrequentialGateDecision:
        raise TypeError("an exact PrequentialGateDecision is required.")
    if value._issuer is not _PREQUENTIAL_GATE_ISSUER:
        raise TypeError("prequential gate issuer is invalid.")
    PrequentialGateDecision.__post_init__(value)
    return value


def _issue_prequential_gate(
    fields: Mapping[str, Any],
    *,
    _issuer: object,
) -> PrequentialGateDecision:
    if _issuer is not _PREQUENTIAL_GATE_ISSUER:
        raise TypeError("prequential-gate issuer token is invalid.")
    value = object.__new__(PrequentialGateDecision)
    for name, item in fields.items():
        object.__setattr__(value, name, item)
    object.__setattr__(value, "_issuer", _issuer)
    return require_prequential_gate(value)


def prequential_gate_decision(
    calibration_fbcca_scores_by_block: np.ndarray,
    observed_labels_by_block: np.ndarray,
    *,
    budget: int,
    block_partition_sha256s: Sequence[str],
    config: V3OperatorConfig,
) -> PrequentialGateDecision:
    """Fit/evaluate the chronological gate using A_Q versus A0 and no metadata."""

    config = _require_exact_operator_config(config)
    budget = _final_budget(budget)
    if budget not in PREQUENTIAL_BUDGETS:
        raise ValueError("prequential gate budget must be 3 or 5.")
    scores = np.asarray(calibration_fbcca_scores_by_block, dtype=np.float64)
    labels = np.asarray(observed_labels_by_block)
    if scores.ndim != 3 or scores.shape[0] != budget:
        raise ValueError("calibration scores must contain exactly the authorized prefix.")
    if scores.shape[1] != scores.shape[2] or scores.shape[1] < 2:
        raise ValueError("each calibration block must be complete and class-balanced.")
    classes = scores.shape[2]
    resolved_labels = _integer_labels(labels, classes, "observed_labels_by_block")
    if resolved_labels.shape != scores.shape[:2]:
        raise ValueError("observed labels must match calibration score blocks.")
    partitions = tuple(block_partition_sha256s)
    if len(partitions) != budget:
        raise ValueError("block_partition_sha256s must match the gate budget.")
    for index, digest in enumerate(partitions):
        _sha256(digest, f"block_partition_sha256s[{index}]")

    evaluated = tuple(range(2, budget + 1))
    ba_deltas: list[float] = []
    log_deltas: list[float] = []
    for evaluation_block in evaluated:
        fit_depth = evaluation_block - 1
        query_scores = scores[evaluation_block - 1]
        evaluation_labels = resolved_labels[evaluation_block - 1].copy()
        support_manifest = _canonical_plain_sha256(
            {
                "schema": "cfeg.metadata-calibration-efficiency-v3.gate-prefix.v1",
                "partition_sha256s": list(partitions[:fit_depth]),
            }
        )
        product = blockwise_p3_support(
            query_scores,
            scores[:fit_depth],
            resolved_labels[:fit_depth],
            query_key=f"prequential-block-{evaluation_block:02d}",
            ordered_support_block_keys=tuple(
                f"block{block:02d}" for block in range(1, evaluation_block)
            ),
            support_eeg_label_manifest_sha256=support_manifest,
            config=config,
        )
        lambda_q = _lambda_q(product.base_probabilities, fit_depth, config.grid_cell.lambda_max)
        aq = _convex_fusion(product.base_probabilities, product.support_probabilities, lambda_q)
        anchor_prediction = np.argmax(query_scores, axis=1)
        aq_prediction = np.argmax(aq, axis=1)
        ba_deltas.append(
            _balanced_accuracy(aq_prediction, evaluation_labels, classes)
            - _balanced_accuracy(anchor_prediction, evaluation_labels, classes)
        )
        log_deltas.append(
            _correct_log_probability(aq, evaluation_labels)
            - _correct_log_probability(product.base_probabilities, evaluation_labels)
        )
    enabled = (
        all(value >= 0.0 for value in ba_deltas)
        and all(value > 0.0 for value in log_deltas)
        and any(value > 0.0 for value in ba_deltas)
    )
    reason = "all_prequential_requirements_pass" if enabled else "prequential_abstention"
    payload = {
        "schema": GATE_SCHEMA,
        "candidate_id": CANDIDATE_ID,
        "active_grid_cell_id": config.grid_cell.grid_cell_id,
        "exact_operator_instance_digest": config.grid_cell.operator_instance_sha256,
        "budget": budget,
        "enabled": enabled,
        "evaluated_blocks": list(evaluated),
        "fit_block_prefixes": [list(range(1, block)) for block in evaluated],
        "block_balanced_accuracy_deltas": ba_deltas,
        "block_log_probability_deltas": log_deltas,
        "block_partition_sha256s": list(partitions),
        "reason": reason,
    }
    return _issue_prequential_gate(
        {
            "candidate_id": CANDIDATE_ID,
            "active_grid_cell_id": config.grid_cell.grid_cell_id,
            "exact_operator_instance_digest": (
                config.grid_cell.operator_instance_sha256
            ),
            "budget": budget,
            "enabled": enabled,
            "evaluated_blocks": evaluated,
            "fit_block_prefixes": tuple(
                tuple(range(1, block)) for block in evaluated
            ),
            "block_balanced_accuracy_deltas": tuple(ba_deltas),
            "block_log_probability_deltas": tuple(log_deltas),
            "block_partition_sha256s": partitions,
            "reason": reason,
            "payload_sha256": canonical_payload_sha256(payload),
            "schema": GATE_SCHEMA,
        },
        _issuer=_PREQUENTIAL_GATE_ISSUER,
    )


@dataclass(frozen=True)
class V3OperatorOutput:
    base_scores: np.ndarray
    base_probabilities: np.ndarray
    support_probabilities: np.ndarray
    fused_probabilities: np.ndarray
    predictions: np.ndarray
    lambdas_Q: np.ndarray
    effective_lambdas: np.ndarray
    variant: V3Variant
    budget: int
    gate_enabled: bool
    g_M: float
    exact_fallback: bool
    fallback_reason: str | None
    support_reliability_capability_sha256: str | None
    context_trust_capability_sha256: str | None
    schema: str = OPERATOR_OUTPUT_SCHEMA


def apply_v3_operator(
    query_fbcca_scores: np.ndarray,
    *,
    query_key: str,
    budget: int,
    variant: V3Variant,
    config: V3OperatorConfig,
    support_product: V3SupportProduct | object | None = None,
    gate_decision: PrequentialGateDecision | object | None = None,
    gate_mode: Literal["deployed", "forced_on"] = "deployed",
    context_trust: ContextTrustCapability | object | None = None,
) -> V3OperatorOutput:
    """Apply A_Q or A_QM; raw acquisition fields are impossible at this boundary."""

    scores = _finite_matrix(query_fbcca_scores, "query_fbcca_scores")
    query_key = _nonempty_ascii(query_key, "query_key")
    budget = _final_budget(budget)
    if variant not in {"A_Q", "A_QM"}:
        raise ValueError("variant must be A_Q or A_QM.")
    config = _require_exact_operator_config(config)
    if variant == "A_Q" and context_trust is not None:
        raise ValueError("A_Q rejects every metadata/context capability input.")
    if gate_mode not in {"deployed", "forced_on"}:
        raise ValueError("gate_mode must be deployed or forced_on.")
    _, base = normalize_fbcca_scores(scores, epsilon=config.score_epsilon)
    if budget == 0:
        return _fallback_output(scores, base, variant, budget, True, "k0_exact_A0")

    if gate_mode == "forced_on":
        gate_enabled = True
    elif budget == 1:
        if gate_decision is not None:
            raise ValueError("k=1 uses the frozen global enable bit, not a local gate object.")
        gate_enabled = True
    else:
        if type(gate_decision) is not PrequentialGateDecision:
            return _fallback_output(
                scores, base, variant, budget, False, "missing_or_invalid_M_free_gate"
            )
        try:
            gate = require_prequential_gate(gate_decision)
        except (AttributeError, TypeError, ValueError, OverflowError):
            return _fallback_output(
                scores, base, variant, budget, False, "missing_or_invalid_M_free_gate"
            )
        if (
            gate.budget != budget
            or gate.active_grid_cell_id != config.grid_cell.grid_cell_id
            or gate.exact_operator_instance_digest
            != config.grid_cell.operator_instance_sha256
        ):
            return _fallback_output(
                scores, base, variant, budget, False, "M_free_gate_binding_mismatch"
            )
        gate_enabled = gate.enabled
    if not gate_enabled:
        return _fallback_output(scores, base, variant, budget, False, "M_free_gate_abstained")

    product = require_v3_support_product(support_product)
    if (
        product.query_key != query_key
        or len(product.ordered_support_block_keys) != budget
        or product.active_grid_cell_id != config.grid_cell.grid_cell_id
        or product.exact_operator_instance_digest != config.grid_cell.operator_instance_sha256
        or not np.array_equal(product.base_probabilities, base)
        or product.query_normalizer_sha256 != _array_sha256(base)
    ):
        raise ValueError("support product does not bind the current signal state.")
    lambda_q = _lambda_q(base, budget, config.grid_cell.lambda_max)
    aq_probability = _convex_fusion(base, product.support_probabilities, lambda_q)
    if variant == "A_Q":
        return _operator_output(
            scores=scores,
            base=base,
            support=product.support_probabilities,
            fused=aq_probability,
            lambda_q=lambda_q,
            effective_lambda=lambda_q,
            variant=variant,
            budget=budget,
            gate_enabled=True,
            g_m=1.0,
            fallback_reason=None,
            reliability_sha=product.reliability_capability.payload_sha256,
            context_sha=None,
        )

    trust = require_context_trust(context_trust)
    if (
        trust.candidate_id != CANDIDATE_ID
        or trust.active_grid_cell_id != product.active_grid_cell_id
        or trust.exact_operator_instance_digest != product.exact_operator_instance_digest
        or trust.query_key != query_key
        or trust.ordered_support_block_keys != product.ordered_support_block_keys
        or trust.support_reliability_capability_sha256
        != product.reliability_capability.payload_sha256
    ):
        raise ValueError("ContextTrustCapability does not bind the current signal state.")
    g_m = trust.g_M_by_query
    if g_m == 1.0:
        fused = aq_probability
        effective = lambda_q
        fallback_reason = "context_exact_neutral_A_Q"
    elif g_m == 0.0:
        fused = base
        effective = _readonly(np.zeros_like(lambda_q))
        fallback_reason = "context_zero_trust_exact_A0"
    else:
        effective = _readonly(lambda_q * g_m)
        fused = _convex_fusion(base, product.support_probabilities, effective)
        fallback_reason = None
    return _operator_output(
        scores=scores,
        base=base,
        support=product.support_probabilities,
        fused=fused,
        lambda_q=lambda_q,
        effective_lambda=effective,
        variant=variant,
        budget=budget,
        gate_enabled=True,
        g_m=g_m,
        fallback_reason=fallback_reason,
        reliability_sha=product.reliability_capability.payload_sha256,
        context_sha=trust.payload_sha256,
    )


def apply_v3_after_preflight(
    query_fbcca_scores: np.ndarray,
    *,
    query_key: str,
    budget: int,
    config: V3OperatorConfig,
    preflight: ContextPreflightCapability,
    expected_ordered_support_block_keys: Sequence[str],
    expected_pairing_sha256: str,
    support_loader: Callable[[], V3SupportProduct],
    gate_decision: PrequentialGateDecision | object | None = None,
    gate_mode: Literal["deployed", "forced_on"] = "deployed",
) -> V3OperatorOutput:
    """Enforce the preflight-before-support-access sequencing for A_QM."""

    scores = _finite_matrix(query_fbcca_scores, "query_fbcca_scores")
    query_key = _nonempty_ascii(query_key, "query_key")
    budget = _final_budget(budget)
    config = _require_exact_operator_config(config)
    _, base = normalize_fbcca_scores(scores, epsilon=config.score_epsilon)
    if budget == 0:
        return _fallback_output(scores, base, "A_QM", budget, True, "k0_exact_A0")

    try:
        current_support_keys = _keys_tuple(
            expected_ordered_support_block_keys,
            "expected_ordered_support_block_keys",
        )
        current_pairing = _sha256(
            expected_pairing_sha256,
            "expected_pairing_sha256",
        )
    except (TypeError, ValueError):
        return _fallback_output(
            scores,
            base,
            "A_QM",
            budget,
            False,
            "invalid_current_context_binding_exact_A0",
        )
    if len(current_support_keys) != budget:
        return _fallback_output(
            scores,
            base,
            "A_QM",
            budget,
            False,
            "invalid_current_context_binding_exact_A0",
        )

    if type(preflight) is not ContextPreflightCapability:
        return _fallback_output(
            scores,
            base,
            "A_QM",
            budget,
            False,
            "invalid_preflight_capability_exact_A0",
        )
    try:
        require_context_preflight(preflight)
        support_keys = _keys_tuple(
            preflight.ordered_support_block_keys,
            "preflight.ordered_support_block_keys",
        )
        if type(preflight.ordered_support_block_keys) is not tuple:
            raise TypeError("preflight support keys must remain an exact tuple.")
        _sha256(preflight.pairing_sha256, "preflight.pairing_sha256")
    except (AttributeError, TypeError, ValueError, OverflowError):
        return _fallback_output(
            scores,
            base,
            "A_QM",
            budget,
            False,
            "invalid_preflight_capability_exact_A0",
        )
    if (
        preflight.candidate_id != CANDIDATE_ID
        or preflight.query_key != query_key
        or len(support_keys) != budget
        or support_keys != current_support_keys
        or preflight.pairing_sha256 != current_pairing
    ):
        return _fallback_output(
            scores,
            base,
            "A_QM",
            budget,
            False,
            "preflight_current_binding_mismatch_exact_A0",
        )
    if preflight.rejects_before_support_access:
        return _fallback_output(
            scores,
            base,
            "A_QM",
            budget,
            False,
            preflight.decision_reason,
        )
    if not callable(support_loader):
        raise TypeError("support_loader must be callable after an accepting preflight.")
    product = support_loader()
    product = require_v3_support_product(product)
    if product.block_weight_mode == "uniform":
        trust = finalize_context_trust_uniform_sensitivity(preflight, product)
    else:
        trust = finalize_context_trust(preflight, product.reliability_capability)
    return apply_v3_operator(
        scores,
        query_key=query_key,
        budget=budget,
        variant="A_QM",
        config=config,
        support_product=product,
        gate_decision=gate_decision,
        gate_mode=gate_mode,
        context_trust=trust,
    )


def exhaustive_derangements(depth: int) -> tuple[tuple[int, ...], ...]:
    """Return lexicographically ordered whole-packet derangements."""

    resolved = _positive_integer(depth, "depth")
    if resolved not in {3, 5}:
        raise ValueError("the frozen pairing assay defines derangements only for k=3 or k=5.")
    identity = tuple(range(resolved))
    values = tuple(
        candidate
        for candidate in permutations(identity)
        if all(candidate[index] != index for index in identity)
    )
    expected = 2 if resolved == 3 else 44
    if len(values) != expected or tuple(sorted(values)) != values:
        raise RuntimeError("derangement enumeration violated the frozen exact count/order.")
    return values


def uniform_weight_support_probability(product: V3SupportProduct) -> np.ndarray:
    """Predeclared sensitivity using uniform, rather than reliability, block weights."""

    product = require_v3_support_product(product)
    return _readonly(np.mean(product.per_block_support_probabilities, axis=0))


def _issue_preflight(
    *,
    reference: ContextReference,
    query_key: str,
    slots: tuple[str, ...],
    affinities: tuple[float, ...],
    comparable_counts: tuple[int, ...],
    decision_reason: str,
    pairing_sha256: str,
    _issuer: object,
) -> ContextPreflightCapability:
    if _issuer is not _CONTEXT_PREFLIGHT_ISSUER:
        raise TypeError("context-preflight builder issuer token is invalid.")
    payload = {
        "schema": CONTEXT_PREFLIGHT_SCHEMA,
        "candidate_id": CANDIDATE_ID,
        "query_key": query_key,
        "ordered_support_block_keys": list(slots),
        "affinity_by_support_block": list(affinities),
        "comparable_channel_counts": list(comparable_counts),
        "decision_reason": decision_reason,
        "context_reference_receipt_sha256": reference.payload_sha256,
        "pairing_sha256": pairing_sha256,
    }
    return _issue_context_preflight(
        {
            "candidate_id": CANDIDATE_ID,
            "query_key": query_key,
            "ordered_support_block_keys": slots,
            "affinity_by_support_block": affinities,
            "comparable_channel_counts": comparable_counts,
            "decision_reason": decision_reason,
            "context_reference_receipt_sha256": reference.payload_sha256,
            "pairing_sha256": pairing_sha256,
            "payload_sha256": canonical_payload_sha256(payload),
            "schema": CONTEXT_PREFLIGHT_SCHEMA,
        },
        _issuer=_CONTEXT_PREFLIGHT_ISSUER,
    )


def _issue_context_preflight(
    fields: Mapping[str, Any],
    *,
    _issuer: object,
) -> ContextPreflightCapability:
    if _issuer is not _CONTEXT_PREFLIGHT_ISSUER:
        raise TypeError("context-preflight issuer token is invalid.")
    value = object.__new__(ContextPreflightCapability)
    for name, item in fields.items():
        object.__setattr__(value, name, item)
    object.__setattr__(value, "_issuer", _issuer)
    return require_context_preflight(value)


def _rejected_preflight(
    reference: ContextReference,
    query_key: str,
    slots: tuple[str, ...],
    supplied_pairing: object,
    reason: str,
    *,
    _issuer: object,
) -> ContextPreflightCapability:
    if _issuer is not _CONTEXT_PREFLIGHT_ISSUER:
        raise TypeError("rejected-preflight issuer token is invalid.")
    pairing = (
        supplied_pairing
        if isinstance(supplied_pairing, str)
        and len(supplied_pairing) == 64
        and all(character in "0123456789abcdef" for character in supplied_pairing)
        else "0" * 64
    )
    return _issue_preflight(
        reference=reference,
        query_key=query_key,
        slots=slots,
        affinities=tuple(1.0 for _ in slots),
        comparable_counts=tuple(0 for _ in slots),
        decision_reason=reason,
        pairing_sha256=pairing,
        _issuer=_CONTEXT_PREFLIGHT_ISSUER,
    )


def _lookup_context_scale(
    reference: ContextReference,
    interface: str | None,
    *,
    lookup_mode: ContextLookupMode,
    wrong_interface_lookup_key: str | None,
) -> tuple[tuple[float | None, ...], tuple[float | None, ...]]:
    tables = reference.table_by_interface
    if lookup_mode == "pooled":
        selected = reference.pooled_table
    elif lookup_mode == "wrong_interface":
        if wrong_interface_lookup_key is None:
            raise ValueError("wrong-interface lookup requires an explicit frozen lookup key.")
        key = _nonempty_ascii(wrong_interface_lookup_key, "wrong_interface_lookup_key")
        if key not in tables:
            raise ValueError("wrong-interface lookup key is absent from the frozen reference.")
        selected = tables[key]
    elif interface is not None and interface in tables:
        selected = tables[interface]
    else:
        selected = reference.pooled_table
    centers: list[float | None] = []
    scales: list[float | None] = []
    for index, (center, scale) in enumerate(zip(selected.centers, selected.scales, strict=True)):
        if center is not None and scale is not None:
            centers.append(center)
            scales.append(scale)
        else:
            pooled_center = reference.pooled_table.centers[index]
            pooled_scale = reference.pooled_table.scales[index]
            centers.append(pooled_center)
            scales.append(pooled_scale)
    return tuple(centers), tuple(scales)


def _standardized_packet(
    packet: ContextPacket,
    centers: Sequence[float | None],
    scales: Sequence[float | None],
) -> tuple[float | None, ...]:
    result: list[float | None] = []
    for impedance, available, center, scale in zip(
        packet.impedance_kohm_by_channel,
        packet.channel_availability,
        centers,
        scales,
        strict=True,
    ):
        if not available or impedance is None or center is None or scale is None:
            result.append(None)
        else:
            result.append((math.log1p(impedance) - center) / scale)
    return tuple(result)


def _lambda_q(probabilities: np.ndarray, budget: int, lambda_max: float) -> np.ndarray:
    if budget <= 0:
        return _readonly(np.zeros(len(probabilities), dtype=np.float64))
    values = lambda_max * (budget / (budget + 1.0)) * normalized_entropy(probabilities)
    return _readonly(values)


def _convex_fusion(
    base: np.ndarray,
    support: np.ndarray,
    weight: np.ndarray | float,
) -> np.ndarray:
    left = _probability_matrix(base, "base")
    right = _probability_matrix(support, "support")
    if left.shape != right.shape:
        raise ValueError("fusion operands must have identical shapes.")
    values = np.asarray(weight, dtype=np.float64)
    if values.ndim == 0:
        values = np.full(len(left), float(values), dtype=np.float64)
    if values.shape != (len(left),) or not np.isfinite(values).all():
        raise ValueError("fusion weights must be a finite scalar or one value per row.")
    if np.any(values < 0.0) or np.any(values > 1.0):
        raise ValueError("fusion weights must be in [0, 1].")
    if np.all(values == 0.0):
        return base
    result = (1.0 - values[:, None]) * left + values[:, None] * right
    return _readonly(result)


def _operator_output(
    *,
    scores: np.ndarray,
    base: np.ndarray,
    support: np.ndarray,
    fused: np.ndarray,
    lambda_q: np.ndarray,
    effective_lambda: np.ndarray,
    variant: V3Variant,
    budget: int,
    gate_enabled: bool,
    g_m: float,
    fallback_reason: str | None,
    reliability_sha: str | None,
    context_sha: str | None,
) -> V3OperatorOutput:
    predictions = _readonly(np.argmax(fused, axis=1).astype(np.int64, copy=False))
    return V3OperatorOutput(
        base_scores=scores,
        base_probabilities=base,
        support_probabilities=support,
        fused_probabilities=fused,
        predictions=predictions,
        lambdas_Q=lambda_q,
        effective_lambdas=effective_lambda,
        variant=variant,
        budget=budget,
        gate_enabled=gate_enabled,
        g_M=g_m,
        exact_fallback=fused is base,
        fallback_reason=fallback_reason,
        support_reliability_capability_sha256=reliability_sha,
        context_trust_capability_sha256=context_sha,
    )


def _fallback_output(
    scores: np.ndarray,
    base: np.ndarray,
    variant: V3Variant,
    budget: int,
    gate_enabled: bool,
    reason: str,
) -> V3OperatorOutput:
    zeros = _readonly(np.zeros(len(base), dtype=np.float64))
    predictions = _readonly(np.argmax(scores, axis=1).astype(np.int64, copy=False))
    return V3OperatorOutput(
        base_scores=scores,
        base_probabilities=base,
        support_probabilities=base,
        fused_probabilities=base,
        predictions=predictions,
        lambdas_Q=zeros,
        effective_lambdas=zeros,
        variant=variant,
        budget=budget,
        gate_enabled=gate_enabled,
        g_M=0.0,
        exact_fallback=True,
        fallback_reason=reason,
        support_reliability_capability_sha256=None,
        context_trust_capability_sha256=None,
    )


def _balanced_accuracy(predictions: np.ndarray, labels: np.ndarray, classes: int) -> float:
    prediction_values = _integer_labels(predictions, classes, "predictions").reshape(-1)
    label_values = _integer_labels(labels, classes, "labels").reshape(-1)
    if prediction_values.shape != label_values.shape:
        raise ValueError("predictions and labels must have identical shapes.")
    recalls = []
    for class_index in range(classes):
        mask = label_values == class_index
        if not np.any(mask):
            raise ValueError("balanced accuracy requires every class.")
        recalls.append(float(np.mean(prediction_values[mask] == class_index)))
    return float(np.mean(recalls))


def _correct_log_probability(probabilities: np.ndarray, labels: np.ndarray) -> float:
    values = _probability_matrix(probabilities, "probabilities")
    label_values = _integer_labels(labels, values.shape[1], "labels").reshape(-1)
    if len(label_values) != len(values):
        raise ValueError("probabilities and labels must have the same row count.")
    correct = np.clip(values[np.arange(len(values)), label_values], 1.0e-12, 1.0)
    return float(np.mean(np.log(correct)))


def _rowwise_kl(first: np.ndarray, second: np.ndarray, epsilon: float) -> np.ndarray:
    left = np.clip(first, epsilon, 1.0)
    right = np.clip(second, epsilon, 1.0)
    return np.sum(left * (np.log(left) - np.log(right)), axis=1)


def _softmax(values: np.ndarray) -> np.ndarray:
    shifted = values - np.max(values, axis=1, keepdims=True)
    exponent = np.exp(shifted)
    return exponent / exponent.sum(axis=1, keepdims=True)


def _readonly(value: np.ndarray) -> np.ndarray:
    result = np.array(value, copy=True)
    result.setflags(write=False)
    return result


def _finite_matrix(value: np.ndarray, name: str) -> np.ndarray:
    result = np.asarray(value)
    if result.ndim != 2 or result.shape[0] == 0 or result.shape[1] < 2:
        raise ValueError(f"{name} must be a non-empty [row,class] matrix.")
    if result.dtype.kind not in "fiu" or not np.isfinite(result).all():
        raise ValueError(f"{name} must contain only finite numeric values.")
    return result.astype(np.float64, copy=False)


def _probability_matrix(value: np.ndarray, name: str) -> np.ndarray:
    result = np.asarray(value, dtype=np.float64)
    if result.ndim != 2 or result.shape[0] == 0 or result.shape[1] < 2:
        raise ValueError(f"{name} must be a non-empty [row,class] matrix.")
    if not np.isfinite(result).all() or np.any(result < 0.0) or np.any(result > 1.0):
        raise ValueError(f"{name} must contain finite probabilities in [0, 1].")
    if not np.allclose(result.sum(axis=1), 1.0, rtol=0.0, atol=1.0e-12):
        raise ValueError(f"{name} rows must sum to one.")
    return result


def _probability_vector(value: np.ndarray, length: int, name: str) -> np.ndarray:
    result = np.asarray(value, dtype=np.float64)
    if result.shape != (length,) or not np.isfinite(result).all() or np.any(result < 0.0):
        raise ValueError(f"{name} must be a finite nonnegative vector of length {length}.")
    if not np.isclose(result.sum(), 1.0, rtol=0.0, atol=1.0e-12):
        raise ValueError(f"{name} must sum to one.")
    return result


def _positive_vector(value: np.ndarray, length: int, name: str) -> np.ndarray:
    result = np.asarray(value, dtype=np.float64)
    if result.shape != (length,) or not np.isfinite(result).all() or np.any(result <= 0.0):
        raise ValueError(f"{name} must be a finite positive vector of length {length}.")
    return result


def _integer_labels(value: np.ndarray, classes: int, name: str) -> np.ndarray:
    result = np.asarray(value)
    if result.dtype.kind not in "iu" or result.size == 0:
        raise ValueError(f"{name} must contain exact integer class indices.")
    if np.any(result < 0) or np.any(result >= classes):
        raise ValueError(f"{name} contains an out-of-range class index.")
    return result.astype(np.int64, copy=False)


def _keys_tuple(value: Sequence[str], name: str) -> tuple[str, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise TypeError(f"{name} must be a sequence of strings.")
    result = tuple(_nonempty_ascii(item, f"{name}[{index}]") for index, item in enumerate(value))
    if len(set(result)) != len(result):
        raise ValueError(f"{name} must not contain duplicates.")
    return result


def _sequence_length(value: Any, length: int, name: str) -> tuple[Any, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence | np.ndarray):
        raise TypeError(f"{name} must be a sequence.")
    result = tuple(value)
    if len(result) != length:
        raise ValueError(f"{name} must contain exactly {length} values.")
    return result


def _boolean_sequence(value: Any, length: int, name: str) -> tuple[bool, ...]:
    raw = _sequence_length(value, length, name)
    result: list[bool] = []
    for index, item in enumerate(raw):
        if not isinstance(item, (bool, np.bool_)):
            raise TypeError(f"{name}[{index}] must be boolean.")
        result.append(bool(item))
    return tuple(result)


def _available_impedance(value: Any, available: bool) -> float | None:
    if not available or not isinstance(value, Real) or isinstance(value, bool):
        return None
    resolved = float(value)
    if not math.isfinite(resolved) or resolved < 0.0:
        return None
    return resolved


def _require_exact_keys(value: Mapping[str, Any], expected: set[str], name: str) -> None:
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be a mapping.")
    observed = set(value)
    if observed != expected:
        missing = sorted(expected - observed)
        extra = sorted(observed - expected)
        raise ValueError(f"{name} keys differ; missing={missing}, extra={extra}.")


def _validate_capability_hash(capability: object, payload: Mapping[str, Any]) -> None:
    observed = getattr(capability, "payload_sha256", None)
    _sha256(observed, "payload_sha256")
    if observed != canonical_payload_sha256(payload):
        raise ValueError("typed capability payload hash is invalid.")


def _nonempty_ascii(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise TypeError(f"{name} must be a non-empty string.")
    try:
        value.encode("ascii")
    except UnicodeEncodeError as error:
        raise ValueError(f"{name} must be strict ASCII.") from error
    return value


def _sha256(value: Any, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{name} must be 64 lowercase hexadecimal characters.")
    return value


def _git_object_id(value: Any, name: str) -> str:
    if (
        type(value) is not str
        or len(value) != 40
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{name} must be an exact lowercase 40-hex Git object ID.")
    return value


def _finite_float(value: Any, name: str) -> float:
    if not isinstance(value, Real) or isinstance(value, bool):
        raise TypeError(f"{name} must be a finite real number.")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite.")
    return result


def _positive_finite(value: Any, name: str) -> float:
    result = _finite_float(value, name)
    if result <= 0.0:
        raise ValueError(f"{name} must be positive.")
    return result


def _positive_integer(value: Any, name: str) -> int:
    if not isinstance(value, Integral) or isinstance(value, bool) or int(value) <= 0:
        raise ValueError(f"{name} must be a positive exact integer.")
    return int(value)


def _nonnegative_integer(value: Any, name: str) -> int:
    if not isinstance(value, Integral) or isinstance(value, bool) or int(value) < 0:
        raise ValueError(f"{name} must be a nonnegative exact integer.")
    return int(value)


def _final_budget(value: Any) -> int:
    if not isinstance(value, Integral) or isinstance(value, bool) or int(value) not in FINAL_BUDGETS:
        raise ValueError(f"budget must be one of {FINAL_BUDGETS}.")
    return int(value)
