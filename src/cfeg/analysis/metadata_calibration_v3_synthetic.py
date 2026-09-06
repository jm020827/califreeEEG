"""Pure, outcome-free synthetic engine for metadata-calibration V3.

The module intentionally performs no artifact publication and exposes no
raw-seed development runner.  A governance-owned wrapper must validate its
nominal development-bundle capability, construct the keyed RNG factory, and
then call the pure generation/evaluation/reduction seams below.
"""

from __future__ import annotations

import hashlib
import json
import marshal
import math
from collections import Counter, defaultdict
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from dataclasses import field as dataclass_field
from functools import cache
from pathlib import Path
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
import yaml
from scipy import stats

from cfeg.baselines import fbcca as _fbcca_module
from cfeg.baselines.fbcca import (
    apply_filterbank,
    cca_score,
    make_reference_signals,
    resolve_filterbank_parameters,
)
from cfeg.models import metadata_calibration_v3 as v3

if TYPE_CHECKING:
    from typing_extensions import Self

_SYNTHETIC_MODULE_IMPORT_PATH = Path(__file__).resolve()
_REPOSITORY = _SYNTHETIC_MODULE_IMPORT_PATH.parents[3]
SYNTHETIC_MODULE_REPOSITORY_PATH = (
    "src/cfeg/analysis/metadata_calibration_v3_synthetic.py"
)
_IMPORTED_SYNTHETIC_MODULE_SHA256 = hashlib.sha256(
    _SYNTHETIC_MODULE_IMPORT_PATH.read_bytes()
).hexdigest()
FBCCA_MODULE_REPOSITORY_PATH = "src/cfeg/baselines/fbcca.py"
FILTERBANK_REPOSITORY_PATH = (
    "configs/baselines/fbcca_chen2015_m3_v2_explicit_weights.yaml"
)
_FBCCA_MODULE_IMPORT_PATH = Path(_fbcca_module.__file__).resolve()
_IMPORTED_FBCCA_MODULE_SHA256 = hashlib.sha256(
    _FBCCA_MODULE_IMPORT_PATH.read_bytes()
).hexdigest()
DEFAULT_SYNTHETIC_PLAN_PATH = (
    _REPOSITORY / "configs/analysis/metadata_calibration_v3_synthetic.yaml"
)
DEFAULT_MASTER_PLAN_PATH = (
    _REPOSITORY / "configs/analysis/metadata_calibration_efficiency_v3.yaml"
)
DEFAULT_FILTERBANK_PATH = _REPOSITORY / FILTERBANK_REPOSITORY_PATH


def _callable_code_sha256(value: object) -> str:
    code = getattr(value, "__code__", None)
    if code is None:
        raise TypeError("bound scientific callable must expose Python bytecode.")
    payload = marshal.dumps(code)
    payload += repr(getattr(value, "__defaults__", None)).encode("utf-8")
    payload += repr(getattr(value, "__kwdefaults__", None)).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


_IMPORTED_FBCCA_CALLABLE_BINDINGS = tuple(
    (name, value, _callable_code_sha256(value))
    for name, value in vars(_fbcca_module).items()
    if callable(value)
    and getattr(value, "__module__", None) == _fbcca_module.__name__
    and hasattr(value, "__code__")
)
_IMPORTED_FBCCA_NUMPY_MODULE = _fbcca_module.np
_IMPORTED_FBCCA_SCIPY_SIGNAL_MODULE = _fbcca_module.signal

EXPECTED_SYNTHETIC_PLAN_SHA256 = (
    "172de79a5315ef6daba9d623f05ee6ebe37595fc8f9dc3c913b6f63c8119f693"
)
EXPECTED_MASTER_PLAN_SHA256 = (
    "ff82c78ab6765c2bf58062195d246df7eb1654cabd1eb413fa4c40ca1f461446"
)
EXPECTED_FILTERBANK_SHA256 = v3.FILTERBANK_SHA256
SYNTHETIC_PLAN_SCHEMA = "cfeg.metadata-calibration-efficiency-v3.synthetic-plan.v1"
DEVELOPMENT_RESULT_SCHEMA = "cfeg.metadata-calibration-efficiency-v3.development-result.v1"
SELECTED_METHOD_FREEZE_SCHEMA = (
    "cfeg.metadata-calibration-efficiency-v3.selected-method-freeze.v1"
)
VALIDATED_DEVELOPMENT_RESULT_SCHEMA = (
    "cfeg.metadata-calibration-efficiency-v3."
    "validated-development-result-capability.v1"
)
AUDITED_DEVELOPMENT_RESULT_SCHEMA = (
    "cfeg.metadata-calibration-efficiency-v3."
    "audited-development-result-capability.v1"
)
SELECTED_METHOD_PROPOSAL_SCHEMA = (
    "cfeg.metadata-calibration-efficiency-v3.selected-method-proposal-capability.v1"
)
DEVELOPMENT_RNG_PRIMITIVE_SCHEMA = (
    "cfeg.metadata-calibration-efficiency-v3.development-rng-binding.v1"
)
PARTICIPANT_METRIC_SCHEMA = (
    "cfeg.metadata-calibration-efficiency-v3.synthetic-participant-metric.v1"
)
INVARIANT_ROW_SCHEMA = "cfeg.metadata-calibration-efficiency-v3.synthetic-invariant-row.v1"

FAMILY_NAMES = (
    "B1_participant_class_confusion",
    "B2_participant_phase_spatial_shift",
    "B3_impedance_linked_transfer_shift",
    "B4_interface_calibrated_impedance_shift",
    "N1_clean_anchor",
    "N2_random_support_labels",
    "N3_nonstationary_calibration",
    "N4_context_null",
)
RNG_FAMILY_NAMES = (*FAMILY_NAMES, "N5_invalid_context")
COMPONENT_NAMES = (
    "spatial_signature",
    "class_phase",
    "block_phase_drift",
    "channel_gain",
    "innovation_noise",
    "confuser_state",
    "support_label_permutation",
    "impedance_measurement_noise",
    "interface_assignment",
    "context_control",
    "sensitivity_resampling",
)
CONTEXT_REFERENCE_INTERFACES = ("neutral", "wet", "dry")
SUPPORT_BUDGETS = (0, 1, 3, 5)
QUERY_BLOCKS = (6, 7, 8, 9, 10)
DEVELOPMENT_PARTICIPANTS = 48
FUTURE_PARTICIPANTS = 96
DEVELOPMENT_METRIC_ROWS = 88_992
DEVELOPMENT_INVARIANT_ROWS = 90
DEVELOPMENT_TOTAL_ROWS = 89_082
FUTURE_SELECTED_TOTAL_ROWS = 19_786
SENSITIVITY_SIGN_FLIP_DRAWS = 100_000
SENSITIVITY_PARTICIPANT_BOOTSTRAP_DRAWS = 10_000
SENSITIVITY_ENDPOINT_IDS = (
    "B1_A_Q_minus_A0_eAUC",
    "B2_A_Q_minus_A0_eAUC",
    "B3_A_QM_minus_A_Q_eAUC",
    "B4_A_QM_minus_A_Q_eAUC",
    "B3_k3_correct_minus_shuffle",
    "B4_k3_correct_minus_shuffle",
)
_SENSITIVITY_ENDPOINT_CODES = {
    endpoint: index for index, endpoint in enumerate(SENSITIVITY_ENDPOINT_IDS, start=1)
}

RngFactory = Callable[[str, int, int, int, str], np.random.Generator]

_UNIT_TEST_RNG_AUTHORITY_ISSUER = object()
_SYNTHETIC_RNG_PRODUCER_ISSUER = object()
_RESERVED_RNG_ROOT_SEEDS = frozenset(
    {
        20_260_906,
        20_260_907,
        20_260_908,
        v3.DEVELOPMENT_ROOT_SEED,
        v3.CONTEXT_REFERENCE_ROOT_SEED,
        8812983586834372979543294859684702645563544465387352627235733726918051280063,
    }
)


@dataclass(frozen=True, slots=True, init=False)
class _UnitTestRNGAuthority:
    """Nonproduction keyed-RNG authority that cannot carry a reserved seed."""

    root_seed: int
    semantic_binding_sha256: str
    _issuer: object = dataclass_field(repr=False, compare=False)

    def __new__(cls, *_args: object, **_kwargs: object) -> Self:
        raise TypeError("unit-test RNG authority has no public constructor.")


def _issue_unit_test_rng_authority(root_seed: int) -> _UnitTestRNGAuthority:
    """Issue an explicit unit authority while rejecting all disclosed roots."""

    if type(root_seed) is not int or root_seed < 0:
        raise TypeError("unit-test root_seed must be an exact nonnegative int.")
    if root_seed in _RESERVED_RNG_ROOT_SEEDS:
        raise ValueError("reserved development/reference/beacon seed is forbidden in tests.")
    value = object.__new__(_UnitTestRNGAuthority)
    object.__setattr__(value, "root_seed", root_seed)
    object.__setattr__(
        value,
        "semantic_binding_sha256",
        _canonical_sha256(
            {
                "schema": (
                    "cfeg.metadata-calibration-efficiency-v3.unit-test-rng-authority.v1"
                ),
                "root_seed": root_seed,
                "scope": "unit_test_only_never_production",
            }
        ),
    )
    object.__setattr__(value, "_issuer", _UNIT_TEST_RNG_AUTHORITY_ISSUER)
    return _require_unit_test_rng_authority(value)


def _require_unit_test_rng_authority(value: object) -> _UnitTestRNGAuthority:
    if type(value) is not _UnitTestRNGAuthority:
        raise TypeError("an exact _UnitTestRNGAuthority is required.")
    if value._issuer is not _UNIT_TEST_RNG_AUTHORITY_ISSUER:
        raise TypeError("unit-test RNG authority issuer is invalid.")
    if type(value.root_seed) is not int or value.root_seed < 0:
        raise TypeError("unit-test root_seed must remain an exact nonnegative int.")
    if value.root_seed in _RESERVED_RNG_ROOT_SEEDS:
        raise ValueError("reserved development/reference/beacon seed is forbidden in tests.")
    expected = _canonical_sha256(
        {
            "schema": "cfeg.metadata-calibration-efficiency-v3.unit-test-rng-authority.v1",
            "root_seed": value.root_seed,
            "scope": "unit_test_only_never_production",
        }
    )
    if value.semantic_binding_sha256 != expected:
        raise ValueError("unit-test RNG authority binding is invalid.")
    return value


_SYNTHETIC_CONTRACT_ISSUER = object()


@dataclass(frozen=True, slots=True, init=False)
class SyntheticV3Contract:
    plan_path: Path
    plan_sha256: str
    master_plan_path: Path
    master_plan_sha256: str
    filterbank_path: Path
    filterbank_sha256: str
    plan: Mapping[str, Any]
    filterbank: Mapping[str, Any]
    plan_content_sha256: str
    filterbank_content_sha256: str
    semantic_binding_sha256: str
    _issuer: object = dataclass_field(repr=False, compare=False)

    def __new__(cls, *_args: object, **_kwargs: object) -> Self:
        raise TypeError("SyntheticV3Contract is issued only by exact file validation.")


def _deep_freeze_json(value: Any) -> Any:
    if isinstance(value, Mapping):
        if any(type(key) not in {str, int} for key in value):
            raise TypeError("contract mappings require exact string or integer keys.")
        return MappingProxyType(
            {key: _deep_freeze_json(item) for key, item in value.items()}
        )
    if isinstance(value, list):
        return tuple(_deep_freeze_json(item) for item in value)
    if value is None or type(value) in {bool, int, float, str}:
        return value
    raise TypeError("contract payload contains a non-JSON value.")


def _deep_thaw_json(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {key: _deep_thaw_json(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_deep_thaw_json(item) for item in value]
    return value


def _contract_semantic_binding(value: SyntheticV3Contract) -> dict[str, Any]:
    return {
        "schema": "cfeg.metadata-calibration-efficiency-v3.synthetic-contract.v1",
        "plan_path": str(value.plan_path),
        "plan_sha256": value.plan_sha256,
        "master_plan_path": str(value.master_plan_path),
        "master_plan_sha256": value.master_plan_sha256,
        "filterbank_path": str(value.filterbank_path),
        "filterbank_sha256": value.filterbank_sha256,
        "plan_content_sha256": value.plan_content_sha256,
        "filterbank_content_sha256": value.filterbank_content_sha256,
    }


def _issue_synthetic_contract(
    fields: Mapping[str, Any],
    *,
    _issuer: object,
) -> SyntheticV3Contract:
    if _issuer is not _SYNTHETIC_CONTRACT_ISSUER:
        raise TypeError("synthetic-contract issuer token is invalid.")
    value = object.__new__(SyntheticV3Contract)
    for name, item in fields.items():
        object.__setattr__(value, name, item)
    object.__setattr__(
        value,
        "semantic_binding_sha256",
        _canonical_sha256(_contract_semantic_binding(value)),
    )
    object.__setattr__(value, "_issuer", _SYNTHETIC_CONTRACT_ISSUER)
    return _require_exact_contract(value)


_SYNTHETIC_PARTICIPANT_ISSUER = object()
_SOURCE_RANGE_STRESS_ISSUER = object()


@dataclass(frozen=True, slots=True, init=False)
class SyntheticParticipant:
    family: str
    participant_index: int
    conditions: tuple[str, ...]
    signals: np.ndarray
    true_labels: np.ndarray
    recorded_support_labels: np.ndarray
    context_packets_by_condition: tuple[tuple[str, tuple[v3.ContextPacket, ...]], ...]
    signal_states: np.ndarray
    context_states_by_condition: tuple[tuple[str, tuple[int, ...]], ...]
    generated_target_classes: np.ndarray
    generated_confuser_classes: np.ndarray
    innovation_sds: np.ndarray
    partition_sha256s: tuple[str, ...]
    source_range_stress: bool = False
    generation_scope: Literal["development", "unit_test"]
    rng_authority_semantic_binding_sha256: str
    payload_sha256: str
    _rng_authority: object = dataclass_field(repr=False, compare=False)
    _issuer: object = dataclass_field(repr=False, compare=False)

    def __new__(cls, *_args: object, **_kwargs: object) -> Self:
        raise TypeError("SyntheticParticipant has no public constructor.")

    def __post_init__(self) -> None:
        if self.family not in FAMILY_NAMES:
            raise ValueError("unknown synthetic family.")
        _nonnegative_exact_int(self.participant_index, "participant_index")
        expected_conditions = ("wet", "dry") if self.family.startswith("B4_") else ("neutral",)
        if self.conditions != expected_conditions:
            raise ValueError("synthetic participant conditions violate the family contract.")
        if self.signals.shape != (10, 12, 8, 500) or self.signals.dtype != np.float64:
            raise ValueError("synthetic EEG must have exact [10,12,8,500] float64 shape.")
        if not np.isfinite(self.signals).all():
            raise ValueError("synthetic EEG contains a non-finite value.")
        if self.true_labels.shape != (10, 12) or self.recorded_support_labels.shape != (10, 12):
            raise ValueError("synthetic label arrays must have [10,12] shape.")
        if self.signal_states.shape != (10,):
            raise ValueError("synthetic signal states must have shape [10].")
        if self.generated_target_classes.shape != (10, 12):
            raise ValueError("generated target classes must have [10,12] shape.")
        if self.generated_confuser_classes.shape != (10, 12):
            raise ValueError("generated confuser classes must have [10,12] shape.")
        if self.innovation_sds.shape != (10, 12):
            raise ValueError("innovation SDs must have [10,12] shape.")
        if tuple(name for name, _ in self.context_packets_by_condition) != self.conditions:
            raise ValueError("context packet condition order is invalid.")
        if tuple(name for name, _ in self.context_states_by_condition) != self.conditions:
            raise ValueError("context state condition order is invalid.")
        if any(len(packets) != 10 for _, packets in self.context_packets_by_condition):
            raise ValueError("each condition requires exactly ten acquisition packets.")
        if any(len(states) != 10 for _, states in self.context_states_by_condition):
            raise ValueError("each condition requires exactly ten context states.")
        if len(self.partition_sha256s) != 10:
            raise ValueError("one partition hash is required per block.")
        for digest in self.partition_sha256s:
            _sha256(digest, "partition_sha256")
        for array in (
            self.signals,
            self.true_labels,
            self.recorded_support_labels,
            self.signal_states,
            self.generated_target_classes,
            self.generated_confuser_classes,
            self.innovation_sds,
        ):
            if array.flags.writeable:
                raise ValueError("synthetic participant arrays must be immutable.")
        if type(self.source_range_stress) is not bool:
            raise TypeError("source_range_stress must be an exact bool.")
        _sha256(
            self.rng_authority_semantic_binding_sha256,
            "rng_authority_semantic_binding_sha256",
        )
        if self.generation_scope == "development":
            authority = v3._require_development_rng_authority_semantics(
                self._rng_authority
            )
        elif self.generation_scope == "unit_test":
            authority = _require_unit_test_rng_authority(self._rng_authority)
        else:
            raise ValueError("unknown synthetic participant generation scope.")
        if (
            self.rng_authority_semantic_binding_sha256
            != authority.semantic_binding_sha256
        ):
            raise ValueError("synthetic participant RNG authority binding is invalid.")
        _sha256(self.payload_sha256, "synthetic participant payload_sha256")
        if self.payload_sha256 != _canonical_sha256(
            _synthetic_participant_binding(self)
        ):
            raise ValueError("synthetic participant content binding is invalid.")

    def context_packets(self, condition: str) -> tuple[v3.ContextPacket, ...]:
        matches = tuple(packets for name, packets in self.context_packets_by_condition if name == condition)
        if len(matches) != 1:
            raise ValueError(f"unknown participant condition {condition!r}.")
        return matches[0]


def _synthetic_participant_binding(value: SyntheticParticipant) -> dict[str, Any]:
    return {
        "schema": "cfeg.metadata-calibration-efficiency-v3.synthetic-participant.v1",
        "family": value.family,
        "participant_index": value.participant_index,
        "conditions": list(value.conditions),
        "signals_sha256": _array_sha256(value.signals),
        "true_labels_sha256": _array_sha256(value.true_labels),
        "recorded_support_labels_sha256": _array_sha256(
            value.recorded_support_labels
        ),
        "context_packets_by_condition": [
            {
                "condition": condition,
                "packets": [
                    {
                        "packet_key": packet.packet_key,
                        "interface_lookup_key": packet.interface_lookup_key,
                        "impedance_kohm_by_channel": list(
                            packet.impedance_kohm_by_channel
                        ),
                        "channel_availability": list(packet.channel_availability),
                    }
                    for packet in packets
                ],
            }
            for condition, packets in value.context_packets_by_condition
        ],
        "signal_states_sha256": _array_sha256(value.signal_states),
        "context_states_by_condition": [
            {"condition": condition, "states": list(states)}
            for condition, states in value.context_states_by_condition
        ],
        "generated_target_classes_sha256": _array_sha256(
            value.generated_target_classes
        ),
        "generated_confuser_classes_sha256": _array_sha256(
            value.generated_confuser_classes
        ),
        "innovation_sds_sha256": _array_sha256(value.innovation_sds),
        "partition_sha256s": list(value.partition_sha256s),
        "source_range_stress": value.source_range_stress,
        "generation_scope": value.generation_scope,
        "rng_authority_semantic_binding_sha256": (
            value.rng_authority_semantic_binding_sha256
        ),
    }


def _issue_synthetic_participant(
    fields: Mapping[str, Any],
    *,
    _issuer: object,
) -> SyntheticParticipant:
    if _issuer is not _SYNTHETIC_PARTICIPANT_ISSUER:
        raise TypeError("synthetic-participant issuer token is invalid.")
    value = object.__new__(SyntheticParticipant)
    for name, item in fields.items():
        object.__setattr__(value, name, item)
    object.__setattr__(
        value,
        "payload_sha256",
        _canonical_sha256(_synthetic_participant_binding(value)),
    )
    object.__setattr__(value, "_issuer", _SYNTHETIC_PARTICIPANT_ISSUER)
    return _require_synthetic_participant_identity(value)


def _require_synthetic_participant_identity(value: object) -> SyntheticParticipant:
    if type(value) is not SyntheticParticipant:
        raise TypeError("participant must be an exact SyntheticParticipant.")
    if value._issuer is not _SYNTHETIC_PARTICIPANT_ISSUER:
        raise TypeError("synthetic participant issuer is invalid.")
    SyntheticParticipant.__post_init__(value)
    return value


_STRICT_FBCCA_PRODUCT_ISSUER = object()
_STRICT_FBCCA_SUPPORT_PRODUCT_ISSUER = object()
_STRICT_FBCCA_COMPUTATION_ISSUER = object()


@dataclass(frozen=True, slots=True, init=False)
class StrictFBCCAProduct:
    """Query-only strict-FBCCA anchors for immutable blocks 6..10."""

    scores: np.ndarray
    query_eeg_sha256s: tuple[str, ...]
    plan_sha256: str
    filterbank_sha256: str
    scores_sha256: str
    producer_sha256: str
    _issuer: object = dataclass_field(repr=False, compare=False)

    def __new__(cls, *_args: object, **_kwargs: object) -> Self:
        raise TypeError("StrictFBCCAProduct has no public constructor.")

    def __post_init__(self) -> None:
        if (
            type(self.scores) is not np.ndarray
            or self.scores.shape != (5, 12, 12)
            or self.scores.dtype != np.float64
        ):
            raise ValueError(
                "query strict-FBCCA scores must have exact [5,12,12] float64 shape."
            )
        if not np.isfinite(self.scores).all() or self.scores.flags.writeable:
            raise ValueError("query strict-FBCCA scores must be finite and immutable.")
        if type(self.query_eeg_sha256s) is not tuple or len(self.query_eeg_sha256s) != 5:
            raise ValueError("query product requires five exact EEG hashes.")
        for digest in self.query_eeg_sha256s:
            _sha256(digest, "query_eeg_sha256")
        if self.plan_sha256 != EXPECTED_SYNTHETIC_PLAN_SHA256:
            raise ValueError("query product synthetic-plan binding is invalid.")
        if self.filterbank_sha256 != EXPECTED_FILTERBANK_SHA256:
            raise ValueError("query product filterbank binding is invalid.")
        if self.scores_sha256 != _array_sha256(self.scores):
            raise ValueError("query product score bytes differ from their bound hash.")
        expected = _canonical_sha256(
            {
                "schema": (
                    "cfeg.metadata-calibration-efficiency-v3."
                    "strict-fbcca-query-product.v1"
                ),
                "plan_sha256": self.plan_sha256,
                "filterbank_sha256": self.filterbank_sha256,
                "query_blocks": list(QUERY_BLOCKS),
                "query_eeg_sha256s": list(self.query_eeg_sha256s),
                "scores_sha256": self.scores_sha256,
            }
        )
        if self.producer_sha256 != expected:
            raise ValueError("query strict-FBCCA producer binding is invalid.")

    def score_for_query_block(self, block: int) -> np.ndarray:
        require_strict_fbcca_product(self)
        if type(block) is not int or block not in QUERY_BLOCKS:
            raise ValueError("query score access is restricted to blocks 6..10.")
        return self.scores[block - QUERY_BLOCKS[0]]


@dataclass(frozen=True, slots=True, init=False)
class StrictFBCCASupportProduct:
    """Lazily produced support scores, labels, and exact partition bindings."""

    budget: int
    scores: np.ndarray
    recorded_support_labels: np.ndarray
    partition_sha256s: tuple[str, ...]
    support_eeg_label_manifest_sha256: str
    plan_sha256: str
    filterbank_sha256: str
    scores_sha256: str
    producer_sha256: str
    _issuer: object = dataclass_field(repr=False, compare=False)

    def __new__(cls, *_args: object, **_kwargs: object) -> Self:
        raise TypeError("StrictFBCCASupportProduct has no public constructor.")

    def __post_init__(self) -> None:
        if type(self.budget) is not int or self.budget not in (1, 3, 5):
            raise ValueError("support FBCCA budget must be exactly 1, 3, or 5.")
        if (
            type(self.scores) is not np.ndarray
            or self.scores.shape != (self.budget, 12, 12)
            or self.scores.dtype != np.float64
        ):
            raise ValueError("support FBCCA scores have the wrong frozen shape/dtype.")
        if (
            self.recorded_support_labels.shape != (self.budget, 12)
            or self.recorded_support_labels.dtype.kind not in "iu"
        ):
            raise ValueError("support labels have the wrong frozen shape/dtype.")
        if (
            not np.isfinite(self.scores).all()
            or self.scores.flags.writeable
            or self.recorded_support_labels.flags.writeable
        ):
            raise ValueError("support scores and labels must be finite and immutable.")
        if np.any(self.recorded_support_labels < 0) or np.any(
            self.recorded_support_labels >= 12
        ):
            raise ValueError("support labels contain an out-of-range class.")
        if type(self.partition_sha256s) is not tuple or len(self.partition_sha256s) != self.budget:
            raise ValueError("support product requires one partition hash per block.")
        for digest in self.partition_sha256s:
            _sha256(digest, "partition_sha256")
        _sha256(
            self.support_eeg_label_manifest_sha256,
            "support_eeg_label_manifest_sha256",
        )
        if self.plan_sha256 != EXPECTED_SYNTHETIC_PLAN_SHA256:
            raise ValueError("support product synthetic-plan binding is invalid.")
        if self.filterbank_sha256 != EXPECTED_FILTERBANK_SHA256:
            raise ValueError("support product filterbank binding is invalid.")
        if self.scores_sha256 != _array_sha256(self.scores):
            raise ValueError("support product score bytes differ from their bound hash.")
        expected_manifest = _canonical_sha256(
            {
                "schema": (
                    "cfeg.metadata-calibration-efficiency-v3.support-manifest.v1"
                ),
                "participant_partition_sha256s": list(self.partition_sha256s),
                "recorded_support_labels_sha256": _array_sha256(
                    self.recorded_support_labels
                ),
            }
        )
        if self.support_eeg_label_manifest_sha256 != expected_manifest:
            raise ValueError("support labels/partitions differ from their bound manifest.")
        expected = _canonical_sha256(
            {
                "schema": (
                    "cfeg.metadata-calibration-efficiency-v3."
                    "strict-fbcca-support-product.v1"
                ),
                "plan_sha256": self.plan_sha256,
                "filterbank_sha256": self.filterbank_sha256,
                "support_blocks": list(range(1, self.budget + 1)),
                "support_eeg_label_manifest_sha256": (
                    self.support_eeg_label_manifest_sha256
                ),
                "scores_sha256": self.scores_sha256,
            }
        )
        if self.producer_sha256 != expected:
            raise ValueError("support strict-FBCCA producer binding is invalid.")


def require_strict_fbcca_product(value: object) -> StrictFBCCAProduct:
    if type(value) is not StrictFBCCAProduct:
        raise TypeError("an exact StrictFBCCAProduct is required.")
    if value._issuer is not _STRICT_FBCCA_PRODUCT_ISSUER:
        raise TypeError("query strict-FBCCA product issuer is invalid.")
    StrictFBCCAProduct.__post_init__(value)
    return value


def require_strict_fbcca_support_product(
    value: object,
) -> StrictFBCCASupportProduct:
    if type(value) is not StrictFBCCASupportProduct:
        raise TypeError("an exact StrictFBCCASupportProduct is required.")
    if value._issuer is not _STRICT_FBCCA_SUPPORT_PRODUCT_ISSUER:
        raise TypeError("support strict-FBCCA product issuer is invalid.")
    StrictFBCCASupportProduct.__post_init__(value)
    return value


def _issue_strict_fbcca_product(
    fields: Mapping[str, Any],
    *,
    _issuer: object,
) -> StrictFBCCAProduct:
    if _issuer is not _STRICT_FBCCA_PRODUCT_ISSUER:
        raise TypeError("query strict-FBCCA issuer token is invalid.")
    value = object.__new__(StrictFBCCAProduct)
    for name, item in fields.items():
        object.__setattr__(value, name, item)
    object.__setattr__(value, "_issuer", _STRICT_FBCCA_PRODUCT_ISSUER)
    return require_strict_fbcca_product(value)


def _issue_strict_fbcca_support_product(
    fields: Mapping[str, Any],
    *,
    _issuer: object,
) -> StrictFBCCASupportProduct:
    if _issuer is not _STRICT_FBCCA_SUPPORT_PRODUCT_ISSUER:
        raise TypeError("support strict-FBCCA issuer token is invalid.")
    value = object.__new__(StrictFBCCASupportProduct)
    for name, item in fields.items():
        object.__setattr__(value, name, item)
    object.__setattr__(value, "_issuer", _STRICT_FBCCA_SUPPORT_PRODUCT_ISSUER)
    return require_strict_fbcca_support_product(value)


@dataclass(frozen=True)
class MethodSpec:
    condition: str
    role: Literal["A0", "A_Q", "A_QM"]
    control: str
    gate: Literal["none", "deployed", "forced_on"]
    budget: int


_VALIDATED_DEVELOPMENT_RESULT_ISSUER = object()
_AUDITED_DEVELOPMENT_RESULT_ISSUER = object()
_SELECTED_METHOD_PROPOSAL_ISSUER = object()
_SELECTED_FREEZE_ASSEMBLY_ISSUER = object()


@dataclass(frozen=True, slots=True, init=False)
class ValidatedDevelopmentResult:
    """Nominal proof of complete-row validation and exact V3 reduction."""

    schema: str
    candidate_id: str
    payload_schema: str
    payload_sha256: str
    master_plan_file_sha256: str
    synthetic_plan_file_sha256: str
    development_bundle_schema: str
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
    development_rng_primitive_schema: str
    development_root_seed: int
    development_rng_authority_schema: str
    development_rng_authority_semantic_binding_sha256: str
    rng_key_map_sha256: str
    validated_context_reference_semantic_binding_sha256: str
    participant_count: int
    b4_source_range_stress_participant_indices: tuple[int, ...]
    grid_cell_count: int
    metric_row_count: int
    invariant_row_count: int
    total_row_count: int
    participant_metric_rows_sha256: str
    invariant_rows_sha256: str
    gate_report_count: int
    complete_grid_gate_report_sha256: str
    uniform_block_weight_sensitivity_sha256: str
    resampling_sensitivity_sha256: str
    sensitivity_participant_vectors_sha256: str
    eligible_grid_cell_ids: tuple[str, ...]
    selection_status: Literal["SELECTED_METHOD_PROPOSED", "DEVELOPMENT_NO_GO"]
    selected_grid_cell_id: str | None
    semantic_binding_sha256: str
    _rng_authority: v3.DevelopmentRNGAuthority = dataclass_field(
        repr=False,
        compare=False,
    )
    _validated_reference: v3.ValidatedContextReference = dataclass_field(
        repr=False,
        compare=False,
    )
    _issuer: object = dataclass_field(repr=False, compare=False)

    def __new__(cls, *_args: object, **_kwargs: object) -> Self:
        raise TypeError(
            "ValidatedDevelopmentResult is issued only by exact semantic validation."
        )


@dataclass(frozen=True, slots=True, init=False)
class AuditedDevelopmentResult:
    """Nominal clean-A recovery proof that grants no RNG or generation access."""

    schema: str
    candidate_id: str
    payload_schema: str
    payload_sha256: str
    development_result_file_sha256: str
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
    clean_commit: str
    clean_tree: str
    participant_metric_rows_sha256: str
    invariant_rows_sha256: str
    complete_grid_gate_report_sha256: str
    uniform_block_weight_sensitivity_sha256: str
    resampling_sensitivity_sha256: str
    sensitivity_participant_vectors_sha256: str
    selection_status: Literal["SELECTED_METHOD_PROPOSED", "DEVELOPMENT_NO_GO"]
    selected_grid_cell_id: str | None
    semantic_binding_sha256: str
    _recovery_capability: object = dataclass_field(repr=False, compare=False)
    _issuer: object = dataclass_field(repr=False, compare=False)

    def __new__(cls, *_args: object, **_kwargs: object) -> Self:
        raise TypeError(
            "AuditedDevelopmentResult is issued only by exact clean-A recovery."
        )


@dataclass(frozen=True, slots=True, init=False)
class SelectedMethodProposal:
    """Nominal proof that a selected-method payload follows an eligible result."""

    schema: str
    candidate_id: str
    payload_schema: str
    payload_sha256: str
    scientific_candidate_id: str
    selected_grid_cell_index: int
    selected_grid_cell_id: str
    selected_prototype_prior_pseudocount: float
    selected_lambda_max: float
    selected_operator_instance_sha256: str
    master_plan_file_sha256: str
    synthetic_plan_file_sha256: str
    development_bundle_schema: str
    development_bundle_payload_sha256: str
    development_bundle_file_sha256: str
    development_result_schema: str
    development_result_payload_sha256: str
    development_result_file_sha256: str
    validated_development_result_semantic_binding_sha256: str
    complete_grid_gate_report_sha256: str
    minimum_mandatory_observed_gain: float
    minimum_corresponding_one_sided_LCB: float
    clean_commit: str
    clean_tree: str
    semantic_binding_sha256: str
    _validated_result: ValidatedDevelopmentResult = dataclass_field(
        repr=False,
        compare=False,
    )
    _issuer: object = dataclass_field(repr=False, compare=False)

    def __new__(cls, *_args: object, **_kwargs: object) -> Self:
        raise TypeError(
            "SelectedMethodProposal is issued only by exact semantic validation."
        )


def validate_synthetic_contract(
    plan_path: str | Path = DEFAULT_SYNTHETIC_PLAN_PATH,
) -> SyntheticV3Contract:
    """Load and validate the byte-frozen V3 synthetic development contract."""

    path = _regular_file(plan_path, "synthetic plan")
    plan_bytes = path.read_bytes()
    observed_sha = hashlib.sha256(plan_bytes).hexdigest()
    if observed_sha != EXPECTED_SYNTHETIC_PLAN_SHA256:
        raise ValueError(
            f"V3 synthetic plan hash drifted: expected {EXPECTED_SYNTHETIC_PLAN_SHA256}, "
            f"observed {observed_sha}."
        )
    decoded = yaml.safe_load(plan_bytes.decode("utf-8"))
    if not isinstance(decoded, Mapping):
        raise TypeError("synthetic plan must decode to one mapping.")
    plan = dict(decoded)
    if (
        plan.get("schema") != SYNTHETIC_PLAN_SCHEMA
        or plan.get("candidate_id") != v3.CANDIDATE_ID
        or plan.get("generator_revision") != "v1_single_insertion_context_trust"
        or plan.get("status") != "V3_2_recovery_development_v3_not_yet_bundle_frozen_or_executable"
    ):
        raise ValueError("synthetic plan identity/status is invalid.")
    master_section = plan.get("master_plan")
    if not isinstance(master_section, Mapping):
        raise TypeError("master_plan must be a mapping.")
    master_path = _regular_file(_REPOSITORY / str(master_section.get("path")), "master plan")
    master_sha = hashlib.sha256(master_path.read_bytes()).hexdigest()
    if (
        master_sha != EXPECTED_MASTER_PLAN_SHA256
        or master_section.get("file_sha256") != EXPECTED_MASTER_PLAN_SHA256
    ):
        raise ValueError("synthetic-to-master hash binding is invalid.")
    waveform = _mapping(plan, "waveform")
    filterbank_path = _regular_file(
        _REPOSITORY / str(waveform.get("filterbank_config")), "filterbank config"
    )
    filterbank_bytes = filterbank_path.read_bytes()
    filterbank_sha = hashlib.sha256(filterbank_bytes).hexdigest()
    if (
        filterbank_sha != EXPECTED_FILTERBANK_SHA256
        or waveform.get("filterbank_config_sha256") != EXPECTED_FILTERBANK_SHA256
    ):
        raise ValueError("frozen strict-FBCCA config binding is invalid.")
    filterbank = yaml.safe_load(filterbank_bytes.decode("utf-8"))
    if not isinstance(filterbank, Mapping):
        raise TypeError("filterbank config must decode to one mapping.")
    _validate_resolved_filterbank(filterbank)
    _validate_exact_plan_sections(plan)
    frozen_plan = _deep_freeze_json(plan)
    frozen_filterbank = _deep_freeze_json(dict(filterbank))
    return _issue_synthetic_contract(
        {
            "plan_path": path.resolve(),
            "plan_sha256": observed_sha,
            "master_plan_path": master_path.resolve(),
            "master_plan_sha256": master_sha,
            "filterbank_path": filterbank_path.resolve(),
            "filterbank_sha256": filterbank_sha,
            "plan": frozen_plan,
            "filterbank": frozen_filterbank,
            "plan_content_sha256": _canonical_sha256({"value": plan}),
            "filterbank_content_sha256": _canonical_sha256(
                {"value": dict(filterbank)}
            ),
        },
        _issuer=_SYNTHETIC_CONTRACT_ISSUER,
    )


def validate_development_contract(
    contract: SyntheticV3Contract | None = None,
) -> dict[str, int]:
    """Validate the exact grid arithmetic without generating any outcome."""

    resolved = (
        validate_synthetic_contract()
        if contract is None
        else _require_current_contract_files(contract)
    )
    counts = development_row_counts(DEVELOPMENT_PARTICIPANTS, grid_cells=9)
    expected = {
        "condition_level_cells_per_parameter_pair": 172,
        "B4_composite_cells_per_parameter_pair": 34,
        "participant_summary_cells_per_parameter_pair": 206,
        "development_participant_rows_per_parameter_pair": 9_888,
        "development_participant_metric_rows": DEVELOPMENT_METRIC_ROWS,
        "development_invariant_rows": DEVELOPMENT_INVARIANT_ROWS,
        "total_development_rows": DEVELOPMENT_TOTAL_ROWS,
    }
    if counts != expected:
        raise RuntimeError(f"development row arithmetic drifted: {counts!r}.")
    plan_counts = _mapping(_deep_thaw_json(resolved.plan), "complete_grid")
    for name, value in expected.items():
        if plan_counts.get(name) != value:
            raise ValueError(f"synthetic plan count {name} is not exact.")
    if plan_counts.get("future_selected_lockbox_total_rows") != FUTURE_SELECTED_TOTAL_ROWS:
        raise ValueError("future selected lockbox row count is not exact.")
    v3.canonical_operator_grid()
    return counts


def development_row_counts(participants: int, *, grid_cells: int) -> dict[str, int]:
    participants = _positive_exact_int(participants, "participants")
    grid_cells = _positive_exact_int(grid_cells, "grid_cells")
    condition_cells = sum(len(method_specs_for_family(family)) for family in FAMILY_NAMES)
    b4_composites = len(method_specs_for_family("B4_interface_calibrated_impedance_shift")) // 2
    summaries = condition_cells + b4_composites
    participant_rows_per_pair = summaries * participants
    metric_rows = participant_rows_per_pair * grid_cells
    invariant_rows = (6 + 4) * grid_cells
    return {
        "condition_level_cells_per_parameter_pair": condition_cells,
        "B4_composite_cells_per_parameter_pair": b4_composites,
        "participant_summary_cells_per_parameter_pair": summaries,
        "development_participant_rows_per_parameter_pair": participant_rows_per_pair,
        "development_participant_metric_rows": metric_rows,
        "development_invariant_rows": invariant_rows,
        "total_development_rows": metric_rows + invariant_rows,
    }


def generate_covariate_reference_packets(
    *,
    rng_authority: v3.ContextReferenceRNGAuthority,
) -> tuple[dict[str, Any], ...]:
    """Generate the exact reference packets under canonical bundle authority."""

    authority = v3.require_context_reference_rng_authority(rng_authority)
    _require_loaded_synthetic_source(authority)
    return _generate_covariate_reference_packets_with_rng_factory(
        _authorized_rng_factory(authority),
        _issuer=_SYNTHETIC_RNG_PRODUCER_ISSUER,
    )


def _generate_covariate_reference_packets_for_test(
    rng_authority: _UnitTestRNGAuthority,
) -> tuple[dict[str, Any], ...]:
    """Generate reference packets from an explicitly nonreserved unit seed."""

    authority = _require_unit_test_rng_authority(rng_authority)
    return _generate_covariate_reference_packets_with_rng_factory(
        _unit_test_rng_factory(authority),
        _issuer=_SYNTHETIC_RNG_PRODUCER_ISSUER,
    )


def _generate_covariate_reference_packets_with_rng_factory(
    rng_factory: RngFactory,
    *,
    _issuer: object,
) -> tuple[dict[str, Any], ...]:
    """Internal reference implementation after authority validation."""

    if _issuer is not _SYNTHETIC_RNG_PRODUCER_ISSUER:
        raise TypeError("synthetic RNG producer issuer token is invalid.")
    _require_rng_factory(rng_factory)
    formulas = {
        "neutral": (math.log1p(6.0), 0.65, 0.05),
        "wet": (math.log1p(20.0), 0.20, 0.03),
        "dry": (math.log1p(262.0), 0.80, 0.08),
    }
    packets: list[dict[str, Any]] = []
    for interface_index, interface in enumerate(CONTEXT_REFERENCE_INTERFACES):
        center, slope, noise_sd = formulas[interface]
        state_rng = _rng(rng_factory, "N5_invalid_context", interface_index, 0, 0, "context_control")
        states = np.asarray([-1] * 128 + [1] * 128, dtype=np.int8)
        states = states[state_rng.permutation(256)]
        for packet_index, state in enumerate(states):
            generator = _rng(
                rng_factory,
                "N5_invalid_context",
                interface_index,
                packet_index + 1,
                0,
                "impedance_measurement_noise",
            )
            offsets = 0.04 * (np.arange(8, dtype=np.float64) - 3.5)
            log_values = center + slope * int(state) + offsets + generator.normal(0.0, noise_sd, 8)
            availability = generator.random(8) >= 0.05
            impedance = np.expm1(log_values)
            packets.append(
                {
                    "interface": interface,
                    "impedance_kohm_by_channel": impedance.tolist(),
                    "channel_availability": availability.tolist(),
                }
            )
    return tuple(packets)


def fit_synthetic_context_reference(
    *,
    rng_authority: v3.ContextReferenceRNGAuthority,
) -> v3.ContextReference:
    authority = v3.require_context_reference_rng_authority(rng_authority)
    _require_loaded_synthetic_source(authority)
    return _fit_synthetic_context_reference_with_rng_factory(
        _authorized_rng_factory(authority),
        _issuer=_SYNTHETIC_RNG_PRODUCER_ISSUER,
    )


def _fit_synthetic_context_reference_for_test(
    rng_authority: _UnitTestRNGAuthority,
) -> v3.ContextReference:
    authority = _require_unit_test_rng_authority(rng_authority)
    return _fit_synthetic_context_reference_with_rng_factory(
        _unit_test_rng_factory(authority),
        _issuer=_SYNTHETIC_RNG_PRODUCER_ISSUER,
    )


def _fit_synthetic_context_reference_with_rng_factory(
    rng_factory: RngFactory,
    *,
    _issuer: object,
) -> v3.ContextReference:
    if _issuer is not _SYNTHETIC_RNG_PRODUCER_ISSUER:
        raise TypeError("synthetic RNG producer issuer token is invalid.")
    packets = _generate_covariate_reference_packets_with_rng_factory(
        rng_factory,
        _issuer=_SYNTHETIC_RNG_PRODUCER_ISSUER,
    )
    return v3.fit_context_reference(
        packets,
        domain="synthetic_development_and_future_synthetic",
    )


def replay_context_reference_payload(
    *,
    rng_authority: v3.ContextReferenceRNGAuthority,
) -> dict[str, Any]:
    """Deterministically replay and self-hash the authorized reference payload."""

    authority = v3.require_context_reference_rng_authority(rng_authority)
    return _replay_context_reference_payload_from_validated_authority(authority)


def _replay_context_reference_payload_from_validated_authority(
    rng_authority: v3.ContextReferenceRNGAuthority,
) -> dict[str, Any]:
    """Replay after the caller has performed the expensive governance check."""

    authority = v3._require_context_reference_rng_authority_semantics(
        rng_authority
    )
    _require_loaded_synthetic_source(authority)
    reference = _fit_synthetic_context_reference_with_rng_factory(
        _authorized_rng_factory(authority),
        _issuer=_SYNTHETIC_RNG_PRODUCER_ISSUER,
    )
    return v3.context_reference_payload(reference)


def stress_participant_indices(
    *,
    participant_count: int,
    rng_authority: v3.DevelopmentRNGAuthority,
) -> tuple[int, ...]:
    """Select the frozen stress ranks under canonical development authority."""

    authority = v3.require_development_rng_authority(rng_authority)
    _require_loaded_synthetic_source(authority)
    return _stress_participant_indices_with_rng_factory(
        _authorized_rng_factory(authority),
        participant_count=participant_count,
        _issuer=_SYNTHETIC_RNG_PRODUCER_ISSUER,
    )


def _stress_participant_indices_for_test(
    rng_authority: _UnitTestRNGAuthority,
    *,
    participant_count: int,
) -> tuple[int, ...]:
    authority = _require_unit_test_rng_authority(rng_authority)
    return _stress_participant_indices_with_rng_factory(
        _unit_test_rng_factory(authority),
        participant_count=participant_count,
        _issuer=_SYNTHETIC_RNG_PRODUCER_ISSUER,
    )


def _stress_participant_indices_with_rng_factory(
    rng_factory: RngFactory,
    *,
    participant_count: int,
    _issuer: object,
) -> tuple[int, ...]:
    """Select exactly 5/48 or 10/96 participants by smallest keyed rank."""

    if _issuer is not _SYNTHETIC_RNG_PRODUCER_ISSUER:
        raise TypeError("synthetic RNG producer issuer token is invalid.")
    count = _positive_exact_int(participant_count, "participant_count")
    if count == DEVELOPMENT_PARTICIPANTS:
        stress_count = 5
    elif count == FUTURE_PARTICIPANTS:
        stress_count = 10
    else:
        raise ValueError("source-range stress is defined only for 48 or 96 participants.")
    ranked = []
    for participant_index in range(count):
        value = float(
            _rng(
                rng_factory,
                "B4_interface_calibrated_impedance_shift",
                participant_index,
                0,
                0,
                "context_control",
            ).random()
        )
        ranked.append((value, participant_index))
    return tuple(sorted(index for _, index in sorted(ranked)[:stress_count]))


def generate_synthetic_participant(
    contract: SyntheticV3Contract,
    *,
    family: str,
    participant_index: int,
    rng_authority: v3.DevelopmentRNGAuthority,
) -> SyntheticParticipant:
    """Generate one participant under exact canonical development authority."""

    contract = _require_current_contract_files(contract)
    authority = v3.require_development_rng_authority(rng_authority)
    _require_loaded_synthetic_source(authority)
    return _generate_synthetic_participant_with_rng_factory(
        contract,
        family=family,
        participant_index=participant_index,
        rng_factory=_authorized_rng_factory(authority),
        rng_authority=authority,
        generation_scope="development",
        _issuer=_SYNTHETIC_RNG_PRODUCER_ISSUER,
    )


def _generate_synthetic_participant_for_test(
    contract: SyntheticV3Contract,
    *,
    family: str,
    participant_index: int,
    rng_authority: _UnitTestRNGAuthority,
) -> SyntheticParticipant:
    authority = _require_unit_test_rng_authority(rng_authority)
    return _generate_synthetic_participant_with_rng_factory(
        contract,
        family=family,
        participant_index=participant_index,
        rng_factory=_unit_test_rng_factory(authority),
        rng_authority=authority,
        generation_scope="unit_test",
        _issuer=_SYNTHETIC_RNG_PRODUCER_ISSUER,
    )


def _generate_synthetic_participant_with_rng_factory(
    contract: SyntheticV3Contract,
    *,
    family: str,
    participant_index: int,
    rng_factory: RngFactory,
    rng_authority: v3.DevelopmentRNGAuthority | _UnitTestRNGAuthority,
    generation_scope: Literal["development", "unit_test"],
    _issuer: object,
) -> SyntheticParticipant:
    """Generate one participant without materializing a family-sized EEG tensor."""

    if _issuer is not _SYNTHETIC_RNG_PRODUCER_ISSUER:
        raise TypeError("synthetic RNG producer issuer token is invalid.")
    if type(contract) is not SyntheticV3Contract:
        raise TypeError("contract must be an exact SyntheticV3Contract.")
    if family not in FAMILY_NAMES:
        raise ValueError(f"unknown synthetic family {family!r}.")
    participant_index = _nonnegative_exact_int(participant_index, "participant_index")
    _require_rng_factory(rng_factory)
    classes, blocks, channels, samples = 12, 10, 8, 500
    timeline = np.arange(samples, dtype=np.float64) / 250.0
    frequencies = 8.0 + 0.4 * np.arange(classes, dtype=np.float64)
    conditions = ("wet", "dry") if family.startswith("B4_") else ("neutral",)

    spatial = np.empty((classes, channels), dtype=np.float64)
    phases = np.empty(classes, dtype=np.float64)
    for class_index in range(classes):
        vector = _rng(
            rng_factory,
            family,
            participant_index,
            0,
            class_index,
            "spatial_signature",
        ).standard_normal(channels)
        norm = float(np.linalg.norm(vector))
        if norm == 0.0:
            raise RuntimeError("synthetic spatial signature had zero norm.")
        spatial[class_index] = vector / norm
        phases[class_index] = _rng(
            rng_factory,
            family,
            participant_index,
            0,
            class_index,
            "class_phase",
        ).uniform(-np.pi, np.pi)

    if family in {
        "B3_impedance_linked_transfer_shift",
        "B4_interface_calibrated_impedance_shift",
        "N4_context_null",
    }:
        signal_states = _balanced_states(
            _rng(rng_factory, family, participant_index, 0, 0, "confuser_state")
        )
    else:
        signal_states = np.zeros(blocks, dtype=np.int8)
    context_states: dict[str, np.ndarray] = {}
    for condition_index, condition in enumerate(conditions):
        if family in {
            "B3_impedance_linked_transfer_shift",
            "B4_interface_calibrated_impedance_shift",
        }:
            context_states[condition] = signal_states.copy()
        elif family == "N4_context_null":
            context_states[condition] = _balanced_states(
                _rng(
                    rng_factory,
                    family,
                    participant_index,
                    0,
                    condition_index,
                    "context_control",
                )
            )
        else:
            context_states[condition] = np.zeros(blocks, dtype=np.int8)

    signals = np.empty((blocks, classes, channels, samples), dtype=np.float64)
    true_labels = np.broadcast_to(np.arange(classes), (blocks, classes)).copy()
    recorded_labels = true_labels.copy()
    generated_targets = np.empty((blocks, classes), dtype=np.int64)
    generated_confusers = np.full((blocks, classes), -1, dtype=np.int64)
    innovation_sds = np.empty((blocks, classes), dtype=np.float64)

    for block_index in range(blocks):
        block = block_index + 1
        drift = float(
            _rng(
                rng_factory,
                family,
                participant_index,
                block,
                0,
                "block_phase_drift",
            ).normal(0.0, 0.08)
        )
        gains = _rng(
            rng_factory, family, participant_index, block, 0, "channel_gain"
        ).lognormal(mean=0.0, sigma=0.12, size=channels)
        if family == "N2_random_support_labels" and block <= 5:
            shift = int(
                _rng(
                    rng_factory,
                    family,
                    participant_index,
                    block,
                    0,
                    "support_label_permutation",
                ).integers(1, classes)
            )
            recorded_labels[block_index] = (true_labels[block_index] + shift) % classes

        for class_index in range(classes):
            target_class = class_index
            target_amplitude = 1.0
            confuser_class: int | None = None
            confuser_amplitude = 0.0
            extra_phase = 0.0
            innovation_sd = 0.85
            if family in {"B1_participant_class_confusion", "N2_random_support_labels"}:
                confuser_class = (class_index + 1 + participant_index % 3) % classes
                confuser_amplitude = 0.90
            elif family == "B2_participant_phase_spatial_shift":
                target_amplitude = 0.65
                innovation_sd = 1.10
            elif family in {
                "B3_impedance_linked_transfer_shift",
                "N4_context_null",
            }:
                confuser_class, extra_phase, innovation_sd = _regime(
                    int(signal_states[block_index]),
                    participant_index,
                    class_index,
                    classes,
                    plus_noise=1.30,
                )
                confuser_amplitude = 0.75
            elif family == "B4_interface_calibrated_impedance_shift":
                confuser_class, extra_phase, innovation_sd = _regime(
                    int(signal_states[block_index]),
                    participant_index,
                    class_index,
                    classes,
                    plus_noise=1.20,
                )
                confuser_amplitude = 0.70
            elif family == "N1_clean_anchor":
                target_amplitude = 2.0
                innovation_sd = 0.05
            elif family == "N3_nonstationary_calibration":
                if block <= 5:
                    target_class = (class_index + block) % classes
                innovation_sd = 0.85

            generated_targets[block_index, class_index] = target_class
            generated_confusers[block_index, class_index] = (
                -1 if confuser_class is None else confuser_class
            )
            innovation_sds[block_index, class_index] = innovation_sd
            target_waveform = (
                target_amplitude
                * spatial[class_index, :, None]
                * _oscillation(
                    frequencies[target_class], phases[target_class], drift, timeline
                )[None, :]
            )
            confuser_waveform = np.zeros((channels, samples), dtype=np.float64)
            if confuser_class is not None:
                confuser_waveform = (
                    confuser_amplitude
                    * spatial[class_index, :, None]
                    * _oscillation(
                        frequencies[confuser_class],
                        phases[confuser_class] + extra_phase,
                        drift,
                        timeline,
                    )[None, :]
                )
            white = _rng(
                rng_factory,
                family,
                participant_index,
                block,
                class_index,
                "innovation_noise",
            ).standard_normal((channels, samples))
            noise = _stationary_ar1(white, innovation_sd=innovation_sd, rho=0.55)
            signals[block_index, class_index] = (
                gains[:, None] * target_waveform + confuser_waveform + noise
            )

    contexts = tuple(
        (
            condition,
            _generate_condition_context_packets(
                family=family,
                condition=condition,
                participant_index=participant_index,
                states=context_states[condition],
                rng_factory=rng_factory,
            ),
        )
        for condition in conditions
    )
    partition_hashes = []
    for block_index in range(blocks):
        partition_hashes.append(
            _canonical_sha256(
                {
                    "schema": "cfeg.metadata-calibration-efficiency-v3.synthetic-partition.v1",
                    "plan_sha256": contract.plan_sha256,
                    "family": family,
                    "participant_index": participant_index,
                    "block": block_index + 1,
                    "signals_sha256": _array_sha256(signals[block_index]),
                    "true_labels_sha256": _array_sha256(true_labels[block_index]),
                    "recorded_labels_sha256": _array_sha256(recorded_labels[block_index]),
                }
            )
        )

    arrays = (
        signals,
        true_labels,
        recorded_labels,
        signal_states,
        generated_targets,
        generated_confusers,
        innovation_sds,
    )
    readonly = tuple(_readonly(array) for array in arrays)
    if generation_scope == "development":
        authority_semantics = v3._require_development_rng_authority_semantics(
            rng_authority
        )
    elif generation_scope == "unit_test":
        authority_semantics = _require_unit_test_rng_authority(rng_authority)
    else:  # pragma: no cover - private callers pass one literal
        raise ValueError("unknown synthetic generation scope.")
    return _issue_synthetic_participant(
        {
            "family": family,
            "participant_index": participant_index,
            "conditions": conditions,
            "signals": readonly[0],
            "true_labels": readonly[1],
            "recorded_support_labels": readonly[2],
            "context_packets_by_condition": contexts,
            "signal_states": readonly[3],
            "context_states_by_condition": tuple(
                (condition, tuple(int(value) for value in context_states[condition]))
                for condition in conditions
            ),
            "generated_target_classes": readonly[4],
            "generated_confuser_classes": readonly[5],
            "innovation_sds": readonly[6],
            "partition_sha256s": tuple(partition_hashes),
            "source_range_stress": False,
            "generation_scope": generation_scope,
            "rng_authority_semantic_binding_sha256": (
                authority_semantics.semantic_binding_sha256
            ),
            "_rng_authority": authority_semantics,
        },
        _issuer=_SYNTHETIC_PARTICIPANT_ISSUER,
    )


def apply_source_range_stress(
    participant: SyntheticParticipant,
    reference: v3.ContextReference,
    *,
    contract: SyntheticV3Contract,
    participant_count: int,
    rng_authority: v3.DevelopmentRNGAuthority,
    validated_reference: v3.ValidatedContextReference,
) -> SyntheticParticipant:
    """Set every B4 query valid-channel source z coordinate to exactly +10."""

    contract = _require_current_contract_files(contract)
    authority = v3.require_development_rng_authority(rng_authority)
    _require_loaded_synthetic_source(authority)
    proof = v3.require_validated_context_reference(
        validated_reference,
        expected_payload_sha256=reference.payload_sha256,
    )
    _require_shared_bundle_provenance(authority, proof)
    participant = _require_synthetic_participant(
        participant,
        contract=contract,
        expected_scope="development",
        expected_rng_authority=authority,
    )
    selected = _stress_participant_indices_with_rng_factory(
        _authorized_rng_factory(authority),
        participant_count=participant_count,
        _issuer=_SYNTHETIC_RNG_PRODUCER_ISSUER,
    )
    if participant.participant_index not in selected:
        raise ValueError("participant is not one of the exact authorized stress ranks.")
    return _apply_source_range_stress_impl(
        participant,
        reference,
        _issuer=_SOURCE_RANGE_STRESS_ISSUER,
    )


def _apply_source_range_stress_for_test(
    participant: SyntheticParticipant,
    reference: v3.ContextReference,
) -> SyntheticParticipant:
    """Explicit non-authoritative unit seam for deterministic stress semantics."""

    participant = _require_synthetic_participant_identity(participant)
    if participant.generation_scope != "unit_test":
        raise TypeError("unit stress seam requires a unit-test participant.")
    return _apply_source_range_stress_impl(
        participant,
        reference,
        _issuer=_SOURCE_RANGE_STRESS_ISSUER,
    )


def _apply_source_range_stress_impl(
    participant: SyntheticParticipant,
    reference: v3.ContextReference,
    *,
    _issuer: object,
) -> SyntheticParticipant:
    if _issuer is not _SOURCE_RANGE_STRESS_ISSUER:
        raise TypeError("source-range stress issuer token is invalid.")
    participant = _require_synthetic_participant_identity(participant)
    if participant.family != (
        "B4_interface_calibrated_impedance_shift"
    ):
        raise TypeError("source-range stress accepts only an exact B4 participant.")
    if type(reference) is not v3.ContextReference:
        raise TypeError("reference must be an exact ContextReference.")
    updated: list[tuple[str, tuple[v3.ContextPacket, ...]]] = []
    for condition, packets in participant.context_packets_by_condition:
        table = reference.table_by_interface.get(condition)
        if table is None:
            raise ValueError("B4 stress interface is absent from the source reference.")
        transformed = list(packets)
        for block in QUERY_BLOCKS:
            packet = packets[block - 1]
            impedance: list[float | None] = []
            availability: list[bool] = []
            for channel_index, (center, scale) in enumerate(
                zip(table.centers, table.scales, strict=True)
            ):
                if center is None or scale is None:
                    center = reference.pooled_table.centers[channel_index]
                    scale = reference.pooled_table.scales[channel_index]
                if center is None or scale is None:
                    impedance.append(None)
                    availability.append(False)
                    continue
                impedance.append(float(np.expm1(center + 10.0 * scale)))
                availability.append(True)
            transformed[block - 1] = v3.ContextPacket(
                packet_key=packet.packet_key,
                interface_lookup_key=packet.interface_lookup_key,
                impedance_kohm_by_channel=tuple(impedance),
                channel_availability=tuple(availability),
            )
        updated.append((condition, tuple(transformed)))
    return _issue_synthetic_participant(
        {
            "family": participant.family,
            "participant_index": participant.participant_index,
            "conditions": participant.conditions,
            "signals": participant.signals,
            "true_labels": participant.true_labels,
            "recorded_support_labels": participant.recorded_support_labels,
            "context_packets_by_condition": tuple(updated),
            "signal_states": participant.signal_states,
            "context_states_by_condition": participant.context_states_by_condition,
            "generated_target_classes": participant.generated_target_classes,
            "generated_confuser_classes": participant.generated_confuser_classes,
            "innovation_sds": participant.innovation_sds,
            "partition_sha256s": participant.partition_sha256s,
            "source_range_stress": True,
            "generation_scope": participant.generation_scope,
            "rng_authority_semantic_binding_sha256": (
                participant.rng_authority_semantic_binding_sha256
            ),
            "_rng_authority": participant._rng_authority,
        },
        _issuer=_SYNTHETIC_PARTICIPANT_ISSUER,
    )


def produce_strict_fbcca(
    contract: SyntheticV3Contract,
    participant: SyntheticParticipant,
) -> StrictFBCCAProduct:
    """Produce query-only anchors without reading any support waveform or label."""

    contract = _require_current_contract_files(contract)
    participant = _require_synthetic_participant(participant, contract=contract)
    if participant.generation_scope == "development":
        authority = v3.require_development_rng_authority(
            participant._rng_authority
        )
        _require_loaded_synthetic_source(authority)
    return _produce_strict_fbcca_after_authorized_boundary(
        contract,
        participant,
        _issuer=_STRICT_FBCCA_COMPUTATION_ISSUER,
    )


def _produce_strict_fbcca_after_authorized_boundary(
    contract: SyntheticV3Contract,
    participant: SyntheticParticipant,
    *,
    _issuer: object,
) -> StrictFBCCAProduct:
    if _issuer is not _STRICT_FBCCA_COMPUTATION_ISSUER:
        raise TypeError("strict-FBCCA computation issuer token is invalid.")
    contract = _require_exact_contract(contract)
    participant = _require_synthetic_participant(participant, contract=contract)
    scores = _score_strict_fbcca_blocks(
        contract,
        participant.signals[5:10],
    )
    query_eeg_sha256s = tuple(
        _array_sha256(participant.signals[block - 1]) for block in QUERY_BLOCKS
    )
    producer_sha = _canonical_sha256(
        {
            "schema": (
                "cfeg.metadata-calibration-efficiency-v3."
                "strict-fbcca-query-product.v1"
            ),
            "plan_sha256": contract.plan_sha256,
            "filterbank_sha256": contract.filterbank_sha256,
            "query_blocks": list(QUERY_BLOCKS),
            "query_eeg_sha256s": list(query_eeg_sha256s),
            "scores_sha256": _array_sha256(scores),
        }
    )
    return _issue_strict_fbcca_product(
        {
            "scores": scores,
            "query_eeg_sha256s": query_eeg_sha256s,
            "plan_sha256": contract.plan_sha256,
            "filterbank_sha256": contract.filterbank_sha256,
            "scores_sha256": _array_sha256(scores),
            "producer_sha256": producer_sha,
        },
        _issuer=_STRICT_FBCCA_PRODUCT_ISSUER,
    )


def produce_strict_fbcca_support(
    contract: SyntheticV3Contract,
    participant: SyntheticParticipant,
    *,
    budget: int,
) -> StrictFBCCASupportProduct:
    """Read and score a support prefix only when a caller invokes its loader."""

    contract = _require_current_contract_files(contract)
    participant = _require_synthetic_participant(participant, contract=contract)
    if participant.generation_scope == "development":
        authority = v3.require_development_rng_authority(
            participant._rng_authority
        )
        _require_loaded_synthetic_source(authority)
    return _produce_strict_fbcca_support_after_authorized_boundary(
        contract,
        participant,
        budget=budget,
        _issuer=_STRICT_FBCCA_COMPUTATION_ISSUER,
    )


def _produce_strict_fbcca_support_after_authorized_boundary(
    contract: SyntheticV3Contract,
    participant: SyntheticParticipant,
    *,
    budget: int,
    _issuer: object,
) -> StrictFBCCASupportProduct:
    if _issuer is not _STRICT_FBCCA_COMPUTATION_ISSUER:
        raise TypeError("strict-FBCCA computation issuer token is invalid.")
    contract = _require_exact_contract(contract)
    participant = _require_synthetic_participant(participant, contract=contract)
    if type(budget) is not int or budget not in (1, 3, 5):
        raise ValueError("support FBCCA budget must be exactly 1, 3, or 5.")
    scores = _score_strict_fbcca_blocks(
        contract,
        participant.signals[:budget],
    )
    return _strict_fbcca_support_product_from_scores(
        contract,
        participant,
        budget=budget,
        scores=scores,
        _issuer=_STRICT_FBCCA_SUPPORT_PRODUCT_ISSUER,
    )


def produce_strict_fbcca_support_prefixes(
    contract: SyntheticV3Contract,
    participant: SyntheticParticipant,
) -> tuple[StrictFBCCASupportProduct, StrictFBCCASupportProduct, StrictFBCCASupportProduct]:
    """Score support blocks 1..5 once and issue exact k=1/3/5 products.

    Callers remain responsible for invoking this only from a loader after an
    accepting context preflight.  The isolated AQM operator never invokes it
    on a rejecting preflight.
    """

    contract = _require_current_contract_files(contract)
    participant = _require_synthetic_participant(participant, contract=contract)
    if participant.generation_scope == "development":
        authority = v3.require_development_rng_authority(
            participant._rng_authority
        )
        _require_loaded_synthetic_source(authority)
    return _produce_strict_fbcca_support_prefixes_after_authorized_boundary(
        contract,
        participant,
        _issuer=_STRICT_FBCCA_COMPUTATION_ISSUER,
    )


def _produce_strict_fbcca_support_prefixes_after_authorized_boundary(
    contract: SyntheticV3Contract,
    participant: SyntheticParticipant,
    *,
    _issuer: object,
) -> tuple[StrictFBCCASupportProduct, StrictFBCCASupportProduct, StrictFBCCASupportProduct]:
    if _issuer is not _STRICT_FBCCA_COMPUTATION_ISSUER:
        raise TypeError("strict-FBCCA computation issuer token is invalid.")
    contract = _require_exact_contract(contract)
    participant = _require_synthetic_participant(participant, contract=contract)
    all_scores = _score_strict_fbcca_blocks(contract, participant.signals[:5])
    return tuple(
        _strict_fbcca_support_product_from_scores(
            contract,
            participant,
            budget=budget,
            scores=_readonly(all_scores[:budget]),
            _issuer=_STRICT_FBCCA_SUPPORT_PRODUCT_ISSUER,
        )
        for budget in (1, 3, 5)
    )  # type: ignore[return-value]


def _incremental_strict_fbcca_support_loader_after_authorized_boundary(
    contract: SyntheticV3Contract,
    participant: SyntheticParticipant,
    *,
    _issuer: object,
) -> Callable[[int], StrictFBCCASupportProduct]:
    """Cache only support blocks already requested by an accepting arm."""

    if _issuer is not _STRICT_FBCCA_COMPUTATION_ISSUER:
        raise TypeError("incremental support-loader issuer token is invalid.")
    contract = _require_exact_contract(contract)
    participant = _require_synthetic_participant(participant, contract=contract)
    maximum_scored = 0
    cached_scores = np.empty((0, 12, 12), dtype=np.float64)
    products: dict[int, StrictFBCCASupportProduct] = {}

    def load(budget: int) -> StrictFBCCASupportProduct:
        nonlocal maximum_scored, cached_scores
        if type(budget) is not int or budget not in (1, 3, 5):
            raise ValueError("support FBCCA budget must be exactly 1, 3, or 5.")
        if budget not in products:
            if budget > maximum_scored:
                new_scores = _score_strict_fbcca_blocks(
                    contract,
                    participant.signals[maximum_scored:budget],
                )
                cached_scores = _readonly(
                    np.concatenate((cached_scores, new_scores), axis=0)
                )
                maximum_scored = budget
            products[budget] = _strict_fbcca_support_product_from_scores(
                contract,
                participant,
                budget=budget,
                scores=_readonly(cached_scores[:budget]),
                _issuer=_STRICT_FBCCA_SUPPORT_PRODUCT_ISSUER,
            )
        return products[budget]

    return load


def _strict_fbcca_support_product_from_scores(
    contract: SyntheticV3Contract,
    participant: SyntheticParticipant,
    *,
    budget: int,
    scores: np.ndarray,
    _issuer: object,
) -> StrictFBCCASupportProduct:
    if _issuer is not _STRICT_FBCCA_SUPPORT_PRODUCT_ISSUER:
        raise TypeError("support-product computation issuer token is invalid.")
    if type(budget) is not int or budget not in (1, 3, 5):
        raise ValueError("support FBCCA budget must be exactly 1, 3, or 5.")
    score_values = np.asarray(scores)
    if score_values.shape != (budget, 12, 12) or score_values.dtype != np.float64:
        raise ValueError("precomputed support scores have the wrong shape/dtype.")
    scores = _readonly(score_values)
    labels = _readonly(participant.recorded_support_labels[:budget])
    partitions = participant.partition_sha256s[:budget]
    manifest = _canonical_sha256(
        {
            "schema": "cfeg.metadata-calibration-efficiency-v3.support-manifest.v1",
            "participant_partition_sha256s": list(partitions),
            "recorded_support_labels_sha256": _array_sha256(labels),
        }
    )
    producer_sha = _canonical_sha256(
        {
            "schema": (
                "cfeg.metadata-calibration-efficiency-v3."
                "strict-fbcca-support-product.v1"
            ),
            "plan_sha256": contract.plan_sha256,
            "filterbank_sha256": contract.filterbank_sha256,
            "support_blocks": list(range(1, budget + 1)),
            "support_eeg_label_manifest_sha256": manifest,
            "scores_sha256": _array_sha256(scores),
        }
    )
    return _issue_strict_fbcca_support_product(
        {
            "budget": budget,
            "scores": scores,
            "recorded_support_labels": labels,
            "partition_sha256s": partitions,
            "support_eeg_label_manifest_sha256": manifest,
            "plan_sha256": contract.plan_sha256,
            "filterbank_sha256": contract.filterbank_sha256,
            "scores_sha256": _array_sha256(scores),
            "producer_sha256": producer_sha,
        },
        _issuer=_STRICT_FBCCA_SUPPORT_PRODUCT_ISSUER,
    )


def _score_strict_fbcca_blocks(
    contract: SyntheticV3Contract,
    block_signals: np.ndarray,
) -> np.ndarray:
    """Score only the waveform blocks explicitly supplied by the caller."""

    values = np.asarray(block_signals)
    if (
        values.ndim != 4
        or not 1 <= values.shape[0] <= 5
        or values.shape[1:] != (12, 8, 500)
        or values.dtype != np.float64
        or not np.isfinite(values).all()
    ):
        raise ValueError("strict FBCCA block input has the wrong frozen shape/dtype.")
    block_count = values.shape[0]
    flat = values.reshape(block_count * 12, 8, 500)
    subbands, parameters = apply_filterbank(flat, sfreq=250.0, filterbank=contract.filterbank)
    weights = np.asarray(parameters["weights"], dtype=np.float64)
    if subbands.shape != (7, block_count * 12, 8, 500) or len(weights) != 7:
        raise RuntimeError("strict FBCCA did not produce the frozen seven subbands.")
    references = make_reference_signals(
        8.0 + 0.4 * np.arange(12), 250.0, 500, n_harmonics=5
    )
    scores = np.zeros((block_count * 12, 12), dtype=np.float64)
    for band_index, weight in enumerate(weights):
        for trial_index in range(block_count * 12):
            correlations = np.asarray(
                [
                    cca_score(
                        subbands[band_index, trial_index],
                        reference_signal,
                        regularization=1.0e-8,
                    )
                    for reference_signal in references
                ],
                dtype=np.float64,
            )
            scores[trial_index] += weight * correlations**2
    return _readonly(scores.reshape(block_count, 12, 12))


@cache
def method_specs_for_family(family: str) -> tuple[MethodSpec, ...]:
    """Return the exact frozen condition-level cell list for one family."""

    if family not in FAMILY_NAMES:
        raise ValueError(f"unknown synthetic family {family!r}.")

    def specs_for_condition(condition: str) -> list[MethodSpec]:
        specs = [MethodSpec(condition, "A0", "none", "none", budget) for budget in SUPPORT_BUDGETS]
        specs += [
            MethodSpec(condition, "A_Q", "none", "deployed", budget)
            for budget in SUPPORT_BUDGETS
        ]
        if family in {
            "B1_participant_class_confusion",
            "B2_participant_phase_spatial_shift",
            "B3_impedance_linked_transfer_shift",
            "B4_interface_calibrated_impedance_shift",
            "N2_random_support_labels",
            "N3_nonstationary_calibration",
        }:
            specs += [
                MethodSpec(condition, "A_Q", "none", "forced_on", budget)
                for budget in (1, 3, 5)
            ]
        if family in {
            "B3_impedance_linked_transfer_shift",
            "B4_interface_calibrated_impedance_shift",
            "N1_clean_anchor",
            "N4_context_null",
        }:
            specs += [
                MethodSpec(condition, "A_QM", "correct", "deployed", budget)
                for budget in SUPPORT_BUDGETS
            ]
            specs += [
                MethodSpec(condition, "A_QM", "all_missing", "deployed", budget)
                for budget in SUPPORT_BUDGETS
            ]
        if family in {
            "B3_impedance_linked_transfer_shift",
            "B4_interface_calibrated_impedance_shift",
            "N4_context_null",
        }:
            specs.append(
                MethodSpec(
                    condition,
                    "A_QM",
                    "support_impedance_channel_rotation",
                    "deployed",
                    1,
                )
            )
            specs += [
                MethodSpec(
                    condition,
                    "A_QM",
                    "within_prefix_packet_shuffle",
                    "deployed",
                    budget,
                )
                for budget in (3, 5)
            ]
        if family == "B3_impedance_linked_transfer_shift":
            specs += [
                MethodSpec(condition, "A_QM", "correct", "forced_on", budget)
                for budget in (1, 3, 5)
            ]
            # Preserve the YAML's exact order: forced-on follows correct deployed.
            correct_start = next(
                index
                for index, spec in enumerate(specs)
                if spec.role == "A_QM" and spec.control == "correct"
            )
            forced = specs[-3:]
            del specs[-3:]
            specs[correct_start + 4 : correct_start + 4] = forced
        if family == "B4_interface_calibrated_impedance_shift":
            specs += [
                MethodSpec(condition, "A_QM", "correct", "forced_on", budget)
                for budget in (1, 3, 5)
            ]
            correct_start = next(
                index
                for index, spec in enumerate(specs)
                if spec.role == "A_QM" and spec.control == "correct"
            )
            forced = specs[-3:]
            del specs[-3:]
            specs[correct_start + 4 : correct_start + 4] = forced
            for control in ("pooled_scale", "wrong_interface_scale", "interface_only"):
                specs += [
                    MethodSpec(condition, "A_QM", control, "deployed", budget)
                    for budget in (1, 3, 5)
                ]
        return specs

    conditions = ("wet", "dry") if family.startswith("B4_") else ("neutral",)
    result = tuple(spec for condition in conditions for spec in specs_for_condition(condition))
    expected = {
        "B1_participant_class_confusion": 11,
        "B2_participant_phase_spatial_shift": 11,
        "B3_impedance_linked_transfer_shift": 25,
        "B4_interface_calibrated_impedance_shift": 68,
        "N1_clean_anchor": 16,
        "N2_random_support_labels": 11,
        "N3_nonstationary_calibration": 11,
        "N4_context_null": 19,
    }[family]
    if len(result) != expected or len(set(result)) != expected:
        raise RuntimeError(f"method-cell construction drifted for {family}.")
    return result


def evaluate_synthetic_participant(
    contract: SyntheticV3Contract,
    participant: SyntheticParticipant,
    fbcca: StrictFBCCAProduct,
    reference: v3.ContextReference,
    grid_cell: v3.V3GridCell,
    *,
    rng_authority: v3.DevelopmentRNGAuthority,
    validated_reference: v3.ValidatedContextReference,
    support_loader: Callable[[int], StrictFBCCASupportProduct],
) -> list[dict[str, Any]]:
    """Evaluate one primary grid cell under exact bundle-bound authority."""

    contract = _require_current_contract_files(contract)
    authority = v3.require_development_rng_authority(rng_authority)
    _require_loaded_synthetic_source(authority)
    proof = v3.require_validated_context_reference(
        validated_reference,
        expected_payload_sha256=reference.payload_sha256,
    )
    _require_shared_bundle_provenance(authority, proof)
    return _evaluate_synthetic_participant_impl(
        contract,
        participant,
        fbcca,
        reference,
        grid_cell,
        validated_reference=proof,
        support_loader=support_loader,
        block_weight_mode="reliability",
        production_rng_authority=authority,
    )


def evaluate_uniform_block_weight_sensitivity(
    contract: SyntheticV3Contract,
    participant: SyntheticParticipant,
    fbcca: StrictFBCCAProduct,
    reference: v3.ContextReference,
    grid_cell: v3.V3GridCell,
    *,
    rng_authority: v3.DevelopmentRNGAuthority,
    validated_reference: v3.ValidatedContextReference,
    support_loader: Callable[[int], StrictFBCCASupportProduct],
) -> list[dict[str, Any]]:
    """Evaluate the report-only common-uniform-block-weight sensitivity."""

    contract = _require_current_contract_files(contract)
    authority = v3.require_development_rng_authority(rng_authority)
    _require_loaded_synthetic_source(authority)
    proof = v3.require_validated_context_reference(
        validated_reference,
        expected_payload_sha256=reference.payload_sha256,
    )
    _require_shared_bundle_provenance(authority, proof)
    return _evaluate_synthetic_participant_impl(
        contract,
        participant,
        fbcca,
        reference,
        grid_cell,
        validated_reference=proof,
        support_loader=support_loader,
        block_weight_mode="uniform",
        production_rng_authority=authority,
    )


def _evaluate_synthetic_participant_for_test(
    contract: SyntheticV3Contract,
    participant: SyntheticParticipant,
    fbcca: StrictFBCCAProduct,
    reference: v3.ContextReference,
    grid_cell: v3.V3GridCell,
    *,
    support_loader: Callable[[int], StrictFBCCASupportProduct],
    block_weight_mode: Literal["reliability", "uniform"] = "reliability",
) -> list[dict[str, Any]]:
    """Explicit nonproduction evaluator seam for deterministic unit fixtures."""

    return _evaluate_synthetic_participant_impl(
        contract,
        participant,
        fbcca,
        reference,
        grid_cell,
        validated_reference=None,
        support_loader=support_loader,
        block_weight_mode=block_weight_mode,
        production_rng_authority=None,
    )


def _evaluate_synthetic_participant_impl(
    contract: SyntheticV3Contract,
    participant: SyntheticParticipant,
    fbcca: StrictFBCCAProduct,
    reference: v3.ContextReference,
    grid_cell: v3.V3GridCell,
    *,
    validated_reference: v3.ValidatedContextReference | None,
    support_loader: Callable[[int], StrictFBCCASupportProduct],
    block_weight_mode: Literal["reliability", "uniform"],
    production_rng_authority: v3.DevelopmentRNGAuthority | None,
) -> list[dict[str, Any]]:
    """Shared evaluator; the public wrapper establishes production authority."""

    _require_exact_contract(contract)
    participant = _require_synthetic_participant(
        participant,
        contract=contract,
        expected_scope=(
            "development" if production_rng_authority is not None else "unit_test"
        ),
        expected_rng_authority=(
            production_rng_authority
            if production_rng_authority is not None
            else participant._rng_authority
        ),
    )
    fbcca = require_strict_fbcca_product(fbcca)
    reference = v3._require_context_reference(reference)
    if block_weight_mode not in {"reliability", "uniform"}:
        raise ValueError("block_weight_mode must be reliability or uniform.")
    if type(grid_cell) is not v3.V3GridCell:
        raise TypeError("grid_cell must be an exact V3GridCell.")
    if fbcca.query_eeg_sha256s != tuple(
        _array_sha256(participant.signals[block - 1]) for block in QUERY_BLOCKS
    ):
        raise ValueError("query FBCCA product does not bind this participant.")
    if not callable(support_loader):
        raise TypeError("support_loader must be callable.")
    config = v3.V3OperatorConfig(grid_cell)
    products: dict[tuple[str, int, int], v3.V3SupportProduct] = {}
    gates: dict[int, v3.PrequentialGateDecision] = {}
    support_prefixes: dict[int, StrictFBCCASupportProduct] = {}

    def support_for(budget: int) -> StrictFBCCASupportProduct:
        if budget not in support_prefixes:
            loaded = require_strict_fbcca_support_product(support_loader(budget))
            if (
                loaded.budget != budget
                or loaded.partition_sha256s != participant.partition_sha256s[:budget]
                or not np.array_equal(
                    loaded.recorded_support_labels,
                    participant.recorded_support_labels[:budget],
                )
            ):
                raise ValueError("lazy support product does not bind this participant/prefix.")
            support_prefixes[budget] = loaded
        return support_prefixes[budget]

    def gate_for(budget: int) -> v3.PrequentialGateDecision | None:
        if budget not in (3, 5):
            return None
        if budget not in gates:
            support = support_for(budget)
            gates[budget] = v3.prequential_gate_decision(
                support.scores,
                support.recorded_support_labels,
                budget=budget,
                block_partition_sha256s=support.partition_sha256s,
                config=config,
            )
        return gates[budget]

    def product_for(condition: str, budget: int, query_block: int) -> v3.V3SupportProduct:
        key = (condition, budget, query_block)
        if key not in products:
            packets = participant.context_packets(condition)
            support_keys = tuple(packet.packet_key for packet in packets[:budget])
            support = support_for(budget)
            producer = (
                v3.blockwise_p3_support
                if block_weight_mode == "reliability"
                else v3.blockwise_p3_support_uniform_sensitivity
            )
            products[key] = producer(
                fbcca.score_for_query_block(query_block),
                support.scores,
                support.recorded_support_labels,
                query_key=packets[query_block - 1].packet_key,
                ordered_support_block_keys=support_keys,
                support_eeg_label_manifest_sha256=(
                    support.support_eeg_label_manifest_sha256
                ),
                config=config,
            )
        return products[key]

    rows: list[dict[str, Any]] = []
    for spec in method_specs_for_family(participant.family):
        rows.append(
            _evaluate_method_spec(
                participant=participant,
                fbcca=fbcca,
                reference=reference,
                validated_reference=validated_reference,
                config=config,
                spec=spec,
                product_for=product_for,
                gate_for=gate_for,
            )
        )
    if participant.family == "B4_interface_calibrated_impedance_shift":
        rows.extend(_equal_condition_composites(rows))
    expected = len(method_specs_for_family(participant.family)) + (
        34 if participant.family == "B4_interface_calibrated_impedance_shift" else 0
    )
    if len(rows) != expected:
        raise RuntimeError("participant family evaluation emitted the wrong row count.")
    for row in rows:
        validate_participant_metric_row(row)
    return rows


def _evaluate_method_spec(
    *,
    participant: SyntheticParticipant,
    fbcca: StrictFBCCAProduct,
    reference: v3.ContextReference,
    validated_reference: v3.ValidatedContextReference | None,
    config: v3.V3OperatorConfig,
    spec: MethodSpec,
    product_for: Callable[[str, int, int], v3.V3SupportProduct],
    gate_for: Callable[[int], v3.PrequentialGateDecision | None],
) -> dict[str, Any]:
    probabilities_by_query: list[np.ndarray] = []
    exact_a0_flags: list[bool] = []
    support_enabled_flags: list[bool] = []
    g_values: list[float] = []
    affinity_values: list[float] = []
    pairing_hashes: list[str] = []
    reliability_hashes: list[str] = []
    pairing_changes: list[float] = []
    scientifically_changed: list[bool] = []
    covered_context_flags: list[bool] = []

    for query_block in QUERY_BLOCKS:
        query_scores = fbcca.score_for_query_block(query_block)
        if spec.role == "A0":
            base = v3.normalize_fbcca_scores(query_scores)[1]
            outputs = [
                _MetricOutput(
                    probabilities=base,
                    exact_a0=True,
                    support_enabled=False,
                    g_m=0.0,
                    affinities=(),
                    comparable_counts=(),
                    pairing_sha256=None,
                    reliability_sha256=None,
                )
            ]
        elif spec.role == "A_Q":
            if spec.budget == 0:
                output = v3.apply_v3_operator(
                    query_scores,
                    query_key=f"unused-k0-{query_block}",
                    budget=0,
                    variant="A_Q",
                    config=config,
                )
            else:
                product = product_for(spec.condition, spec.budget, query_block)
                output = v3.apply_v3_operator(
                    query_scores,
                    query_key=product.query_key,
                    budget=spec.budget,
                    variant="A_Q",
                    config=config,
                    support_product=product,
                    gate_decision=gate_for(spec.budget),
                    gate_mode=spec.gate,
                )
            outputs = [_metric_output_from_operator(output)]
        else:
            outputs = _evaluate_aqm_query(
                participant=participant,
                query_block=query_block,
                query_scores=query_scores,
                reference=reference,
                validated_reference=validated_reference,
                config=config,
                spec=spec,
                product_for=product_for,
                gate_for=gate_for,
            )

        query_labels = participant.true_labels[query_block - 1]
        block_bas = [_balanced_accuracy_from_probabilities(item.probabilities, query_labels) for item in outputs]
        block_logs = [_correct_log_probability(item.probabilities, query_labels) for item in outputs]
        # For exhaustive derangements the metric, not the probability vector,
        # is averaged within the participant/query unit.
        probabilities_by_query.append(
            np.asarray([float(np.mean(block_bas)), float(np.mean(block_logs))])
        )
        exact_a0_flags.append(all(item.exact_a0 for item in outputs))
        support_enabled_flags.append(any(item.support_enabled for item in outputs))
        if spec.role == "A_QM":
            g_values.append(float(np.mean([item.g_m for item in outputs])))
            affinity_values.extend(
                affinity for item in outputs for affinity in item.affinities
            )
            pairing_hashes.extend(
                item.pairing_sha256
                for item in outputs
                if item.pairing_sha256 is not None
            )
            reliability_hashes.extend(
                item.reliability_sha256
                for item in outputs
                if item.reliability_sha256 is not None
            )
            covered_context_flags.append(
                any(any(count > 0 for count in item.comparable_counts) for item in outputs)
            )
            if spec.control == "within_prefix_packet_shuffle":
                correct = _evaluate_aqm_assignment(
                    participant=participant,
                    condition=spec.condition,
                    budget=spec.budget,
                    query_block=query_block,
                    query_scores=query_scores,
                    reference=reference,
                    validated_reference=validated_reference,
                    config=config,
                    control="correct",
                    assignment=tuple(range(spec.budget)),
                    product_for=product_for,
                    gate_for=gate_for,
                    gate_mode=spec.gate,
                )
                changes = [abs(correct.g_m - item.g_m) for item in outputs]
                mean_change = float(np.mean(changes))
                pairing_changes.append(mean_change)
                scientifically_changed.append(mean_change >= 0.05)

    metric_values = np.vstack(probabilities_by_query)
    helpful_eligible = (
        participant.family
        in {
            "B3_impedance_linked_transfer_shift",
            "B4_interface_calibrated_impedance_shift",
        }
        and spec.role == "A_QM"
        and spec.control == "correct"
        and spec.budget in (1, 3)
    )
    helpful_count = (
        sum(
            enabled and abs(g_m - 1.0) > 1.0e-12
            for enabled, g_m in zip(support_enabled_flags, g_values, strict=True)
        )
        if helpful_eligible
        else 0
    )
    row = {
        "schema": PARTICIPANT_METRIC_SCHEMA,
        "row_type": "participant_metric",
        "grid_cell_index": config.grid_cell.index,
        "grid_cell_id": config.grid_cell.grid_cell_id,
        "operator_instance_sha256": config.grid_cell.operator_instance_sha256,
        "family": participant.family,
        "participant_index": participant.participant_index,
        "condition": spec.condition,
        "source_range_stress": participant.source_range_stress,
        "role": spec.role,
        "control": spec.control,
        "gate": spec.gate,
        "budget": spec.budget,
        "balanced_accuracy": float(np.mean(metric_values[:, 0])),
        "correct_log_probability": float(np.mean(metric_values[:, 1])),
        "support_enabled": any(support_enabled_flags),
        "exact_A0_query_block_fraction": float(np.mean(exact_a0_flags)),
        "helpful_metadata_use_count": int(helpful_count),
        "helpful_metadata_use_denominator": 5 if helpful_eligible else 0,
        "g_M_by_query_block": g_values,
        "affinity_min": min(affinity_values) if affinity_values else None,
        "affinity_max": max(affinity_values) if affinity_values else None,
        "affinity_standard_deviation": (
            float(np.std(affinity_values, ddof=0)) if affinity_values else None
        ),
        "pairing_sha256s": pairing_hashes,
        "support_reliability_capability_sha256s": reliability_hashes,
        "pairing_packet_binding_changed_fraction": (
            1.0 if spec.control == "within_prefix_packet_shuffle" else None
        ),
        "pairing_mean_derangement_abs_g_M_change_by_query_block": pairing_changes,
        "pairing_scientifically_changed_count": (
            sum(scientifically_changed)
            if spec.control == "within_prefix_packet_shuffle"
            else 0
        ),
        "pairing_potential_unit_count": (
            len(QUERY_BLOCKS) if spec.control == "within_prefix_packet_shuffle" else 0
        ),
        "context_packet_covered_count": (
            sum(covered_context_flags)
            if spec.control == "within_prefix_packet_shuffle"
            else 0
        ),
    }
    return row


@dataclass(frozen=True)
class _MetricOutput:
    probabilities: np.ndarray
    exact_a0: bool
    support_enabled: bool
    g_m: float
    affinities: tuple[float, ...]
    comparable_counts: tuple[int, ...]
    pairing_sha256: str | None
    reliability_sha256: str | None


def _metric_output_from_operator(
    output: v3.V3OperatorOutput,
    *,
    affinities: Sequence[float] = (),
    comparable_counts: Sequence[int] = (),
    pairing_sha256: str | None = None,
) -> _MetricOutput:
    return _MetricOutput(
        probabilities=output.fused_probabilities,
        exact_a0=output.fused_probabilities is output.base_probabilities,
        support_enabled=(
            output.gate_enabled
            and output.budget > 0
            and output.fused_probabilities is not output.base_probabilities
        ),
        g_m=output.g_M,
        affinities=tuple(float(value) for value in affinities),
        comparable_counts=tuple(int(value) for value in comparable_counts),
        pairing_sha256=pairing_sha256,
        reliability_sha256=output.support_reliability_capability_sha256,
    )


def _evaluate_aqm_query(
    *,
    participant: SyntheticParticipant,
    query_block: int,
    query_scores: np.ndarray,
    reference: v3.ContextReference,
    validated_reference: v3.ValidatedContextReference | None,
    config: v3.V3OperatorConfig,
    spec: MethodSpec,
    product_for: Callable[[str, int, int], v3.V3SupportProduct],
    gate_for: Callable[[int], v3.PrequentialGateDecision | None],
) -> list[_MetricOutput]:
    if spec.budget == 0:
        output = v3.apply_v3_operator(
            query_scores,
            query_key=f"unused-k0-{query_block}",
            budget=0,
            variant="A_QM",
            config=config,
        )
        return [_metric_output_from_operator(output)]
    if spec.control == "within_prefix_packet_shuffle":
        return [
            _evaluate_aqm_assignment(
                participant=participant,
                condition=spec.condition,
                budget=spec.budget,
                query_block=query_block,
                query_scores=query_scores,
                reference=reference,
                validated_reference=validated_reference,
                config=config,
                control=spec.control,
                assignment=assignment,
                product_for=product_for,
                gate_for=gate_for,
                gate_mode=spec.gate,
            )
            for assignment in v3.exhaustive_derangements(spec.budget)
        ]
    return [
        _evaluate_aqm_assignment(
            participant=participant,
            condition=spec.condition,
            budget=spec.budget,
            query_block=query_block,
            query_scores=query_scores,
            reference=reference,
            validated_reference=validated_reference,
            config=config,
            control=spec.control,
            assignment=tuple(range(spec.budget)),
            product_for=product_for,
            gate_for=gate_for,
            gate_mode=spec.gate,
        )
    ]


def _evaluate_aqm_assignment(
    *,
    participant: SyntheticParticipant,
    condition: str,
    budget: int,
    query_block: int,
    query_scores: np.ndarray,
    reference: v3.ContextReference,
    validated_reference: v3.ValidatedContextReference | None,
    config: v3.V3OperatorConfig,
    control: str,
    assignment: tuple[int, ...],
    product_for: Callable[[str, int, int], v3.V3SupportProduct],
    gate_for: Callable[[int], v3.PrequentialGateDecision | None],
    gate_mode: str,
) -> _MetricOutput:
    packets = participant.context_packets(condition)
    query_packet = packets[query_block - 1]
    support_packets = tuple(packets[index] for index in assignment)
    lookup_mode: v3.ContextLookupMode = "interface"
    wrong_interface: str | None = None
    if control in {"all_missing", "interface_only"}:
        query_packet = _missing_packet(
            query_packet,
            retain_interface=(control == "interface_only"),
            suffix=control,
        )
        support_packets = tuple(
            _missing_packet(
                packet,
                retain_interface=(control == "interface_only"),
                suffix=control,
            )
            for packet in support_packets
        )
    elif control == "support_impedance_channel_rotation":
        if budget != 1:
            raise ValueError("support channel rotation is defined only at k=1.")
        support_packets = (_rotate_packet(support_packets[0]),)
    elif control == "pooled_scale":
        lookup_mode = "pooled"
    elif control == "wrong_interface_scale":
        lookup_mode = "wrong_interface"
        wrong_interface = "dry" if condition == "wet" else "wet"
    elif control not in {"correct", "within_prefix_packet_shuffle"}:
        raise ValueError(f"unknown A_QM context control {control!r}.")

    logical_support_keys = tuple(packet.packet_key for packet in packets[:budget])
    pairing = v3.context_pairing_sha256(
        query_key=query_packet.packet_key,
        ordered_support_block_keys=logical_support_keys,
        query_packet_key=query_packet.packet_key,
        ordered_support_packet_keys=tuple(packet.packet_key for packet in support_packets),
    )
    preflight_arguments = {
        "reference": reference,
        "query_key": query_packet.packet_key,
        "ordered_support_block_keys": logical_support_keys,
        "query_packet": query_packet,
        "support_packets": support_packets,
        "pairing_sha256": pairing,
        "lookup_mode": lookup_mode,
        "wrong_interface_lookup_key": wrong_interface,
    }
    if validated_reference is None:
        preflight = v3._preflight_context_unit(**preflight_arguments)
    else:
        preflight = v3._preflight_context_after_authorized_boundary(
            validated_reference=validated_reference,
            **preflight_arguments,
        )
    if preflight.rejects_before_support_access:
        output = v3.apply_v3_after_preflight(
            query_scores,
            query_key=query_packet.packet_key,
            budget=budget,
            config=config,
            preflight=preflight,
            expected_ordered_support_block_keys=logical_support_keys,
            expected_pairing_sha256=pairing,
            support_loader=lambda: product_for(condition, budget, query_block),
            gate_decision=None,
            gate_mode=gate_mode,  # type: ignore[arg-type]
        )
        return _metric_output_from_operator(
            output,
            affinities=preflight.affinity_by_support_block,
            comparable_counts=preflight.comparable_channel_counts,
            pairing_sha256=preflight.pairing_sha256,
        )
    gate_decision = gate_for(budget)
    loaded_product: v3.V3SupportProduct | None = None

    def load_product() -> v3.V3SupportProduct:
        nonlocal loaded_product
        loaded_product = product_for(condition, budget, query_block)
        return loaded_product

    output = v3.apply_v3_after_preflight(
        query_scores,
        query_key=query_packet.packet_key,
        budget=budget,
        config=config,
        preflight=preflight,
        expected_ordered_support_block_keys=logical_support_keys,
        expected_pairing_sha256=pairing,
        support_loader=load_product,
        gate_decision=gate_decision,
        gate_mode=gate_mode,  # type: ignore[arg-type]
    )
    if loaded_product is None:  # pragma: no cover - accepted preflight always loads once
        raise RuntimeError("accepted context preflight did not load support.")
    if loaded_product.block_weight_mode == "uniform":
        trust = v3.finalize_context_trust_uniform_sensitivity(
            preflight,
            loaded_product,
        )
    else:
        trust = v3.finalize_context_trust(
            preflight,
            loaded_product.reliability_capability,
        )
    metric = _metric_output_from_operator(
        output,
        affinities=preflight.affinity_by_support_block,
        comparable_counts=preflight.comparable_channel_counts,
        pairing_sha256=preflight.pairing_sha256,
    )
    return _MetricOutput(
        probabilities=metric.probabilities,
        exact_a0=metric.exact_a0,
        support_enabled=metric.support_enabled,
        g_m=trust.g_M_by_query,
        affinities=metric.affinities,
        comparable_counts=metric.comparable_counts,
        pairing_sha256=metric.pairing_sha256,
        reliability_sha256=loaded_product.reliability_capability.payload_sha256,
    )


def _missing_packet(
    packet: v3.ContextPacket,
    *,
    retain_interface: bool,
    suffix: str,
) -> v3.ContextPacket:
    del suffix  # the immutable packet identity remains bound while fields are ablated
    return v3.ContextPacket(
        packet_key=packet.packet_key,
        interface_lookup_key=packet.interface_lookup_key if retain_interface else None,
        impedance_kohm_by_channel=(None,) * 8,
        channel_availability=(False,) * 8,
    )


def _rotate_packet(packet: v3.ContextPacket) -> v3.ContextPacket:
    return v3.ContextPacket(
        packet_key=f"{packet.packet_key}:channel-rotation",
        interface_lookup_key=packet.interface_lookup_key,
        impedance_kohm_by_channel=(
            packet.impedance_kohm_by_channel[-1],
            *packet.impedance_kohm_by_channel[:-1],
        ),
        channel_availability=(
            packet.channel_availability[-1],
            *packet.channel_availability[:-1],
        ),
    )


def _equal_condition_composites(
    rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    by_method: dict[tuple[str, str, str, int], dict[str, Mapping[str, Any]]] = defaultdict(dict)
    for row in rows:
        key = (row["role"], row["control"], row["gate"], row["budget"])
        by_method[key][row["condition"]] = row
    if len(by_method) != 34 or any(set(value) != {"wet", "dry"} for value in by_method.values()):
        raise RuntimeError("B4 condition rows cannot form the exact 34 paired composites.")
    composites = []
    for key in sorted(by_method, key=lambda value: (value[0], value[1], value[2], value[3])):
        wet = by_method[key]["wet"]
        dry = by_method[key]["dry"]
        composite = dict(wet)
        composite["condition"] = "equal_condition_composite"
        for field in (
            "balanced_accuracy",
            "correct_log_probability",
            "exact_A0_query_block_fraction",
        ):
            composite[field] = 0.5 * (float(wet[field]) + float(dry[field]))
        composite["support_enabled"] = bool(wet["support_enabled"] or dry["support_enabled"])
        composite["helpful_metadata_use_count"] = int(wet["helpful_metadata_use_count"]) + int(
            dry["helpful_metadata_use_count"]
        )
        composite["helpful_metadata_use_denominator"] = int(
            wet["helpful_metadata_use_denominator"]
        ) + int(dry["helpful_metadata_use_denominator"])
        wet_g = list(wet["g_M_by_query_block"])
        dry_g = list(dry["g_M_by_query_block"])
        composite["g_M_by_query_block"] = (
            [0.5 * (left + right) for left, right in zip(wet_g, dry_g, strict=True)]
            if wet_g
            else []
        )
        affinity_values = [
            value
            for row in (wet, dry)
            for value in (row["affinity_min"], row["affinity_max"])
            if value is not None
        ]
        composite["affinity_min"] = min(affinity_values) if affinity_values else None
        composite["affinity_max"] = max(affinity_values) if affinity_values else None
        # The full raw affinity vector is intentionally absent from result rows;
        # report the equal-condition mean of within-condition population SDs.
        sd_values = [
            value
            for value in (wet["affinity_standard_deviation"], dry["affinity_standard_deviation"])
            if value is not None
        ]
        composite["affinity_standard_deviation"] = (
            float(np.mean(sd_values)) if sd_values else None
        )
        composite["pairing_sha256s"] = [*wet["pairing_sha256s"], *dry["pairing_sha256s"]]
        composite["support_reliability_capability_sha256s"] = [
            *wet["support_reliability_capability_sha256s"],
            *dry["support_reliability_capability_sha256s"],
        ]
        binding_values = [
            value
            for value in (
                wet["pairing_packet_binding_changed_fraction"],
                dry["pairing_packet_binding_changed_fraction"],
            )
            if value is not None
        ]
        composite["pairing_packet_binding_changed_fraction"] = (
            float(np.mean(binding_values)) if binding_values else None
        )
        composite["pairing_mean_derangement_abs_g_M_change_by_query_block"] = [
            *wet["pairing_mean_derangement_abs_g_M_change_by_query_block"],
            *dry["pairing_mean_derangement_abs_g_M_change_by_query_block"],
        ]
        composite["pairing_scientifically_changed_count"] = int(
            wet["pairing_scientifically_changed_count"]
        ) + int(dry["pairing_scientifically_changed_count"])
        composite["pairing_potential_unit_count"] = int(wet["pairing_potential_unit_count"]) + int(
            dry["pairing_potential_unit_count"]
        )
        composite["context_packet_covered_count"] = int(wet["context_packet_covered_count"]) + int(
            dry["context_packet_covered_count"]
        )
        composites.append(composite)
    return composites


_PARTICIPANT_ROW_KEYS = {
    "schema",
    "row_type",
    "grid_cell_index",
    "grid_cell_id",
    "operator_instance_sha256",
    "family",
    "participant_index",
    "condition",
    "source_range_stress",
    "role",
    "control",
    "gate",
    "budget",
    "balanced_accuracy",
    "correct_log_probability",
    "support_enabled",
    "exact_A0_query_block_fraction",
    "helpful_metadata_use_count",
    "helpful_metadata_use_denominator",
    "g_M_by_query_block",
    "affinity_min",
    "affinity_max",
    "affinity_standard_deviation",
    "pairing_sha256s",
    "support_reliability_capability_sha256s",
    "pairing_packet_binding_changed_fraction",
    "pairing_mean_derangement_abs_g_M_change_by_query_block",
    "pairing_scientifically_changed_count",
    "pairing_potential_unit_count",
    "context_packet_covered_count",
}


def validate_participant_metric_row(row: Mapping[str, Any]) -> None:
    """Validate one exact participant-metric row without external state."""

    _require_exact_keys(row, _PARTICIPANT_ROW_KEYS, "participant metric row")
    if row["schema"] != PARTICIPANT_METRIC_SCHEMA or row["row_type"] != "participant_metric":
        raise ValueError("participant metric row identity is invalid.")
    cell = v3.grid_cell_by_id(row["grid_cell_id"])
    if (
        row["grid_cell_index"] != cell.index
        or row["operator_instance_sha256"] != cell.operator_instance_sha256
    ):
        raise ValueError("participant row grid binding is invalid.")
    family = row["family"]
    if family not in FAMILY_NAMES:
        raise ValueError("participant row family is invalid.")
    _nonnegative_exact_int(row["participant_index"], "participant_index")
    if type(row["source_range_stress"]) is not bool:
        raise TypeError("source_range_stress must be an exact bool.")
    if row["source_range_stress"] and family != "B4_interface_calibrated_impedance_shift":
        raise ValueError("source-range stress is defined only for B4.")
    key = (row["condition"], row["role"], row["control"], row["gate"], row["budget"])
    allowed = {
        (spec.condition, spec.role, spec.control, spec.gate, spec.budget)
        for spec in method_specs_for_family(family)
    }
    if family == "B4_interface_calibrated_impedance_shift":
        allowed |= {
            ("equal_condition_composite", spec.role, spec.control, spec.gate, spec.budget)
            for spec in method_specs_for_family(family)
            if spec.condition == "wet"
        }
    if key not in allowed:
        raise ValueError("participant metric row is not an exact declared method cell.")
    ba = _finite_float(row["balanced_accuracy"], "balanced_accuracy")
    if not 0.0 <= ba <= 1.0:
        raise ValueError("balanced_accuracy must be in [0, 1].")
    log_score = _finite_float(row["correct_log_probability"], "correct_log_probability")
    if log_score > 1.0e-12:
        raise ValueError("correct log probability cannot be positive.")
    if type(row["support_enabled"]) is not bool:
        raise TypeError("support_enabled must be an exact bool.")
    fallback = _finite_float(
        row["exact_A0_query_block_fraction"], "exact_A0_query_block_fraction"
    )
    if not 0.0 <= fallback <= 1.0:
        raise ValueError("exact A0 fraction must be in [0, 1].")
    for name in (
        "helpful_metadata_use_count",
        "helpful_metadata_use_denominator",
        "pairing_scientifically_changed_count",
        "pairing_potential_unit_count",
        "context_packet_covered_count",
    ):
        _nonnegative_exact_int(row[name], name)
    if row["helpful_metadata_use_count"] > row["helpful_metadata_use_denominator"]:
        raise ValueError("helpful metadata count exceeds its denominator.")
    if row["pairing_scientifically_changed_count"] > row["pairing_potential_unit_count"]:
        raise ValueError("pairing changed count exceeds its potential units.")
    if row["context_packet_covered_count"] > row["pairing_potential_unit_count"]:
        raise ValueError("context coverage count exceeds pairing potential units.")
    for name in (
        "g_M_by_query_block",
        "pairing_sha256s",
        "support_reliability_capability_sha256s",
        "pairing_mean_derangement_abs_g_M_change_by_query_block",
    ):
        if not isinstance(row[name], list):
            raise TypeError(f"{name} must be a list.")
    for value in row["g_M_by_query_block"]:
        resolved = _finite_float(value, "g_M_by_query_block item")
        if not 0.0 <= resolved <= 1.0:
            raise ValueError("g_M values must be in [0, 1].")
    for name in ("pairing_sha256s", "support_reliability_capability_sha256s"):
        for value in row[name]:
            _sha256(value, f"{name} item")
    for value in row["pairing_mean_derangement_abs_g_M_change_by_query_block"]:
        resolved = _finite_float(value, "pairing mean g_M change")
        if not 0.0 <= resolved <= 1.0:
            raise ValueError("pairing g_M change must be in [0, 1].")
    nullability = ("affinity_min", "affinity_max", "affinity_standard_deviation")
    if any(row[name] is None for name in nullability) != all(
        row[name] is None for name in nullability
    ):
        raise ValueError("affinity summary fields must be jointly null or non-null.")
    if row["affinity_min"] is not None:
        minimum = _finite_float(row["affinity_min"], "affinity_min")
        maximum = _finite_float(row["affinity_max"], "affinity_max")
        deviation = _finite_float(
            row["affinity_standard_deviation"], "affinity_standard_deviation"
        )
        if not 0.0 <= minimum <= maximum <= 1.0 or deviation < 0.0:
            raise ValueError("affinity summary is outside its valid range.")
    binding = row["pairing_packet_binding_changed_fraction"]
    if row["control"] == "within_prefix_packet_shuffle":
        if binding != 1.0 or row["pairing_potential_unit_count"] not in {5, 10}:
            raise ValueError("shuffle row lacks exact packet-binding/potential evidence.")
    elif binding is not None or row["pairing_potential_unit_count"] != 0:
        raise ValueError("non-shuffle row carries pairing potency evidence.")


def build_invariant_rows(
    contract: SyntheticV3Contract,
    reference: v3.ContextReference,
    grid_cell: v3.V3GridCell,
    *,
    rng_authority: v3.DevelopmentRNGAuthority,
    validated_reference: v3.ValidatedContextReference,
) -> list[dict[str, Any]]:
    """Execute the six N5 and four N6 deterministic implementation guards."""

    contract = _require_current_contract_files(contract)
    authority = v3.require_development_rng_authority(rng_authority)
    _require_loaded_synthetic_source(authority)
    proof = v3.require_validated_context_reference(
        validated_reference,
        expected_payload_sha256=reference.payload_sha256,
    )
    _require_shared_bundle_provenance(authority, proof)
    return _build_invariant_rows_impl(
        reference,
        grid_cell,
        validated_reference=proof,
    )


def _build_invariant_rows_for_test(
    reference: v3.ContextReference,
    grid_cell: v3.V3GridCell,
) -> list[dict[str, Any]]:
    """Explicit nonproduction invariant seam for unit fixtures."""

    return _build_invariant_rows_impl(
        reference,
        grid_cell,
        validated_reference=None,
    )


def _build_invariant_rows_impl(
    reference: v3.ContextReference,
    grid_cell: v3.V3GridCell,
    *,
    validated_reference: v3.ValidatedContextReference | None,
) -> list[dict[str, Any]]:
    if type(reference) is not v3.ContextReference or type(grid_cell) is not v3.V3GridCell:
        raise TypeError("invariant rows require exact reference and grid-cell objects.")
    v3.context_reference_payload(reference)
    config = v3.V3OperatorConfig(grid_cell)
    query_scores = np.eye(12, dtype=np.float64)
    support_scores = np.stack([np.eye(12, dtype=np.float64)] * 3)
    labels = np.stack([np.arange(12, dtype=np.int64)] * 3)
    product = v3.blockwise_p3_support(
        query_scores,
        support_scores[:1],
        labels[:1],
        query_key="N5-query",
        ordered_support_block_keys=("N5-support",),
        support_eeg_label_manifest_sha256="a" * 64,
        config=config,
    )
    aq = v3.apply_v3_operator(
        query_scores,
        query_key="N5-query",
        budget=1,
        variant="A_Q",
        config=config,
        support_product=product,
    )
    n5_results: dict[str, bool] = {}

    base_query = _packet_at_reference_z(reference, "N5-query", "neutral", 0.0)
    base_support = _packet_at_reference_z(reference, "N5-support", "neutral", 0.5)
    unknown_query = _packet_at_reference_z(reference, "N5-query", "__pooled__", 0.0, "unknown")
    unknown_support = _packet_at_reference_z(
        reference, "N5-support", "__pooled__", 0.5, "unknown"
    )
    unknown = _one_block_preflight(
        reference,
        validated_reference,
        unknown_query,
        unknown_support,
    )
    n5_results["unknown_interface_category"] = not unknown.rejects_before_support_access

    for case_id, invalid in (
        ("negative_impedance", -1.0),
        ("nonfinite_impedance", float("nan")),
    ):
        raw_query = _context_packet_mapping(base_query)
        raw_support = _context_packet_mapping(base_support)
        raw_query["impedance_kohm_by_channel"] = [invalid] * 8
        raw_support["impedance_kohm_by_channel"] = [invalid] * 8
        preflight = _one_block_preflight(
            reference,
            validated_reference,
            raw_query,
            raw_support,
        )
        trust = v3.finalize_context_trust(preflight, product.reliability_capability)
        aqm = v3.apply_v3_operator(
            query_scores,
            query_key="N5-query",
            budget=1,
            variant="A_QM",
            config=config,
            support_product=product,
            context_trust=trust,
        )
        n5_results[case_id] = (
            preflight.decision_reason == "all_missing_exact_A_Q"
            and np.array_equal(aqm.fused_probabilities, aq.fused_probabilities)
        )

    wrong_count = _context_packet_mapping(base_query)
    wrong_count["impedance_kohm_by_channel"] = wrong_count["impedance_kohm_by_channel"][:-1]
    wrong = _one_block_preflight(
        reference,
        validated_reference,
        wrong_count,
        base_support,
    )
    n5_results["wrong_channel_count"] = wrong.decision_reason == "malformed_packet_exact_A0"
    mismatched = _one_block_preflight(
        reference,
        validated_reference,
        base_query,
        base_support,
        pairing_override="b" * 64,
    )
    n5_results["packet_key_or_pairing_digest_mismatch"] = (
        mismatched.decision_reason == "pairing_mismatch_exact_A0"
    )
    ood_query = _packet_at_reference_z(reference, "N5-query", "neutral", 10.0)
    ood = _one_block_preflight(
        reference,
        validated_reference,
        ood_query,
        base_support,
    )
    n5_results["query_source_standardized_OOD"] = (
        ood.decision_reason == "query_context_OOD_exact_A0"
    )

    uniform_scores = np.zeros((3, 12, 12), dtype=np.float64)
    uniform = v3.blockwise_p3_support(
        query_scores,
        uniform_scores,
        labels,
        query_key="N6-query",
        ordered_support_block_keys=("N6-1", "N6-2", "N6-3"),
        support_eeg_label_manifest_sha256="c" * 64,
        config=config,
    )
    below_scores = np.zeros((3, 12, 12), dtype=np.float64)
    for block in range(3):
        below_scores[block, np.arange(12), np.arange(12)] = -100.0
    below = v3.blockwise_p3_support(
        query_scores,
        below_scores,
        labels,
        query_key="N6-query",
        ordered_support_block_keys=("N6-1", "N6-2", "N6-3"),
        support_eeg_label_manifest_sha256="c" * 64,
        config=config,
    )
    corrupt_scores = np.stack([np.eye(12), np.eye(12), -np.eye(12)])
    corrupt = v3.blockwise_p3_support(
        query_scores,
        corrupt_scores,
        labels,
        query_key="N6-query",
        ordered_support_block_keys=("N6-1", "N6-2", "N6-3"),
        support_eeg_label_manifest_sha256="c" * 64,
        config=config,
    )
    n6_results = {
        "uniform_chance_probabilities": (
            np.array_equal(uniform.normalized_block_weights, np.full(3, 1.0 / 3.0))
            and np.allclose(uniform.block_reliabilities, 1.0 / 12.0)
        ),
        "all_observed_label_probabilities_below_chance": (
            np.all(below.block_reliabilities > 0.0)
            and np.isclose(below.normalized_block_weights.sum(), 1.0)
        ),
        "one_corrupt_block_among_three": (
            corrupt.block_reliabilities[2] < corrupt.block_reliabilities[0]
            and np.array_equal(
                corrupt.support_probabilities,
                np.sum(
                    corrupt.normalized_block_weights[:, None, None]
                    * corrupt.per_block_support_probabilities,
                    axis=0,
                ),
            )
        ),
        "confident_wrong_anchor_probabilities": (
            np.all(below.block_reliabilities >= 1.0e-12)
            and not np.array_equal(below.support_probabilities, below.base_probabilities)
        ),
    }
    rows = []
    for family, results in (("N5_invalid_context", n5_results), ("N6_support_reliability_invariants", n6_results)):
        for case_id, passed in results.items():
            detail = {
                "schema": "cfeg.metadata-calibration-efficiency-v3.invariant-detail.v1",
                "grid_cell_id": grid_cell.grid_cell_id,
                "family": family,
                "case_id": case_id,
                "passed": bool(passed),
            }
            rows.append(
                {
                    "schema": INVARIANT_ROW_SCHEMA,
                    "row_type": "invariant",
                    "grid_cell_index": grid_cell.index,
                    "grid_cell_id": grid_cell.grid_cell_id,
                    "operator_instance_sha256": grid_cell.operator_instance_sha256,
                    "family": family,
                    "case_id": case_id,
                    "passed": bool(passed),
                    "detail_sha256": _canonical_sha256(detail),
                }
            )
    if len(rows) != 10:
        raise RuntimeError("invariant suite must emit exactly ten rows per grid cell.")
    for row in rows:
        validate_invariant_row(row)
    return rows


_INVARIANT_ROW_KEYS = {
    "schema",
    "row_type",
    "grid_cell_index",
    "grid_cell_id",
    "operator_instance_sha256",
    "family",
    "case_id",
    "passed",
    "detail_sha256",
}


def validate_invariant_row(row: Mapping[str, Any]) -> None:
    _require_exact_keys(row, _INVARIANT_ROW_KEYS, "invariant row")
    if row["schema"] != INVARIANT_ROW_SCHEMA or row["row_type"] != "invariant":
        raise ValueError("invariant row identity is invalid.")
    cell = v3.grid_cell_by_id(row["grid_cell_id"])
    if row["grid_cell_index"] != cell.index or row["operator_instance_sha256"] != (
        cell.operator_instance_sha256
    ):
        raise ValueError("invariant row grid binding is invalid.")
    cases = {
        "N5_invalid_context": {
            "unknown_interface_category",
            "negative_impedance",
            "nonfinite_impedance",
            "wrong_channel_count",
            "packet_key_or_pairing_digest_mismatch",
            "query_source_standardized_OOD",
        },
        "N6_support_reliability_invariants": {
            "uniform_chance_probabilities",
            "all_observed_label_probabilities_below_chance",
            "one_corrupt_block_among_three",
            "confident_wrong_anchor_probabilities",
        },
    }
    if row["family"] not in cases or row["case_id"] not in cases[row["family"]]:
        raise ValueError("invariant row family/case is invalid.")
    if type(row["passed"]) is not bool:
        raise TypeError("invariant passed must be an exact bool.")
    _sha256(row["detail_sha256"], "detail_sha256")


def validate_complete_development_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    expected_stress_participant_indices: Sequence[int],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Reject any partial, duplicated, or extra row before reduction/publication."""

    if isinstance(rows, (str, bytes)) or not isinstance(rows, Sequence):
        raise TypeError("development rows must be a sequence.")
    if len(rows) != DEVELOPMENT_TOTAL_ROWS:
        raise ValueError(
            f"complete V3 development requires {DEVELOPMENT_TOTAL_ROWS} rows, got {len(rows)}."
        )
    stress = tuple(expected_stress_participant_indices)
    if (
        len(stress) != 5
        or tuple(sorted(set(stress))) != stress
        or any(type(index) is not int or not 0 <= index < DEVELOPMENT_PARTICIPANTS for index in stress)
    ):
        raise ValueError("development B4 stress indices must be five sorted unique indices in [0,48).")
    metric_rows: list[dict[str, Any]] = []
    invariant_rows: list[dict[str, Any]] = []
    for row in rows:
        if not isinstance(row, Mapping):
            raise TypeError("every development row must be a mapping.")
        copied = dict(row)
        if row.get("row_type") == "participant_metric":
            validate_participant_metric_row(copied)
            metric_rows.append(copied)
        elif row.get("row_type") == "invariant":
            validate_invariant_row(copied)
            invariant_rows.append(copied)
        else:
            raise ValueError("development row_type is invalid.")
    if len(metric_rows) != DEVELOPMENT_METRIC_ROWS or len(invariant_rows) != DEVELOPMENT_INVARIANT_ROWS:
        raise ValueError("development metric/invariant row partition is incomplete.")

    observed_metric = Counter(_metric_identity(row) for row in metric_rows)
    expected_metric = Counter(_expected_metric_identities())
    if observed_metric != expected_metric:
        missing = list((expected_metric - observed_metric).elements())[:3]
        extra = list((observed_metric - expected_metric).elements())[:3]
        raise ValueError(f"development participant grid differs; missing={missing}, extra={extra}.")
    observed_invariant = Counter(_invariant_identity(row) for row in invariant_rows)
    expected_invariant = Counter(_expected_invariant_identities())
    if observed_invariant != expected_invariant:
        missing = list((expected_invariant - observed_invariant).elements())[:3]
        extra = list((observed_invariant - expected_invariant).elements())[:3]
        raise ValueError(f"development invariant grid differs; missing={missing}, extra={extra}.")

    expected_stress = set(stress)
    for row in metric_rows:
        should_stress = (
            row["family"] == "B4_interface_calibrated_impedance_shift"
            and row["participant_index"] in expected_stress
        )
        if row["source_range_stress"] != should_stress:
            raise ValueError("B4 source-range stress flag differs from the bound participant ranks.")
    canonical_metric = sorted(metric_rows, key=_participant_row_sort_key)
    canonical_invariant = sorted(invariant_rows, key=_invariant_row_sort_key)
    if [dict(row) for row in rows] != [*canonical_metric, *canonical_invariant]:
        raise ValueError("development rows are not in the exact canonical order.")
    return canonical_metric, canonical_invariant


def _validate_complete_uniform_sensitivity_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    expected_stress_participant_indices: Sequence[int],
) -> list[dict[str, Any]]:
    """Validate a report-only second evaluation without changing primary count."""

    if isinstance(rows, (str, bytes)) or not isinstance(rows, Sequence):
        raise TypeError("uniform sensitivity rows must be a sequence.")
    if len(rows) != DEVELOPMENT_METRIC_ROWS:
        raise ValueError(
            "uniform sensitivity requires exactly 88,992 report-only metric rows."
        )
    stress = tuple(expected_stress_participant_indices)
    if (
        len(stress) != 5
        or tuple(sorted(set(stress))) != stress
        or any(
            type(index) is not int
            or not 0 <= index < DEVELOPMENT_PARTICIPANTS
            for index in stress
        )
    ):
        raise ValueError("uniform sensitivity stress ranks are invalid.")
    copied = [dict(row) for row in rows]
    for row in copied:
        validate_participant_metric_row(row)
    if Counter(_metric_identity(row) for row in copied) != Counter(
        _expected_metric_identities()
    ):
        raise ValueError("uniform sensitivity metric grid is incomplete.")
    expected_stress = set(stress)
    for row in copied:
        should_stress = (
            row["family"] == "B4_interface_calibrated_impedance_shift"
            and row["participant_index"] in expected_stress
        )
        if row["source_range_stress"] != should_stress:
            raise ValueError("uniform sensitivity stress flags are invalid.")
    canonical = sorted(copied, key=_participant_row_sort_key)
    if copied != canonical:
        raise ValueError("uniform sensitivity rows are not canonically ordered.")
    return canonical


def canonicalize_development_rows(
    participant_rows: Iterable[Mapping[str, Any]],
    invariant_rows: Iterable[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Sort separately generated rows into the one canonical result order."""

    metrics = [dict(row) for row in participant_rows]
    invariants = [dict(row) for row in invariant_rows]
    for row in metrics:
        validate_participant_metric_row(row)
    for row in invariants:
        validate_invariant_row(row)
    return [
        *sorted(metrics, key=_participant_row_sort_key),
        *sorted(invariants, key=_invariant_row_sort_key),
    ]


def summarize_complete_development(
    metric_rows: Sequence[Mapping[str, Any]],
    invariant_rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Compute every frozen promotion component and rank tuple for all nine cells."""

    if len(metric_rows) != DEVELOPMENT_METRIC_ROWS or len(invariant_rows) != DEVELOPMENT_INVARIANT_ROWS:
        raise ValueError("complete development rows are required for summary.")
    reports = []
    for cell in v3.canonical_operator_grid():
        cell_metrics = [row for row in metric_rows if row["grid_cell_index"] == cell.index]
        cell_invariants = [row for row in invariant_rows if row["grid_cell_index"] == cell.index]
        if len(cell_metrics) != 9_888 or len(cell_invariants) != 10:
            raise ValueError("one grid cell is incomplete before reduction.")
        reports.append(_summarize_grid_cell(cell, cell_metrics, cell_invariants))
    return reports


def _summarize_complete_development_authorized(
    metric_rows: Sequence[Mapping[str, Any]],
    invariant_rows: Sequence[Mapping[str, Any]],
    uniform_metric_rows: Sequence[Mapping[str, Any]],
    *,
    rng_authority: v3.DevelopmentRNGAuthority,
    _issuer: object,
) -> list[dict[str, Any]]:
    """Add report-only sensitivities after one public authority check."""

    if _issuer is not _SYNTHETIC_RNG_PRODUCER_ISSUER:
        raise TypeError("authorized development reducer issuer token is invalid.")
    authority = v3._require_development_rng_authority_semantics(rng_authority)
    if (
        len(metric_rows) != DEVELOPMENT_METRIC_ROWS
        or len(invariant_rows) != DEVELOPMENT_INVARIANT_ROWS
        or len(uniform_metric_rows) != DEVELOPMENT_METRIC_ROWS
    ):
        raise ValueError("authorized development reduction requires all exact rows.")
    rng_factory = _authorized_rng_factory(authority)
    reports = []
    for cell in v3.canonical_operator_grid():
        cell_metrics = [
            row for row in metric_rows if row["grid_cell_index"] == cell.index
        ]
        cell_invariants = [
            row for row in invariant_rows if row["grid_cell_index"] == cell.index
        ]
        cell_uniform = [
            row for row in uniform_metric_rows if row["grid_cell_index"] == cell.index
        ]
        if (
            len(cell_metrics) != 9_888
            or len(cell_invariants) != 10
            or len(cell_uniform) != 9_888
        ):
            raise ValueError("one authorized grid cell is incomplete before reduction.")
        reports.append(
            _summarize_grid_cell(
                cell,
                cell_metrics,
                cell_invariants,
                uniform_rows=cell_uniform,
                sensitivity_rng_factory=rng_factory,
                _issuer=_SYNTHETIC_RNG_PRODUCER_ISSUER,
            )
        )
    return reports


def select_grid_cell(gate_reports: Sequence[Mapping[str, Any]]) -> Mapping[str, Any] | None:
    """Apply the exact no-rounding V3 selector to complete gate reports."""

    if len(gate_reports) != 9:
        raise ValueError("selection requires exactly nine grid-cell gate reports.")
    by_index = {report.get("grid_cell_index"): report for report in gate_reports}
    if set(by_index) != set(range(1, 10)):
        raise ValueError("gate reports do not cover exact grid indices 1..9.")
    for report in gate_reports:
        _validate_gate_report(report)
    eligible = [report for report in gate_reports if report.get("eligible") is True]
    if not eligible:
        return None
    return min(
        eligible,
        key=lambda report: (
            -float(report["selection_rank_tuple"]["minimum_mandatory_observed_gain"]),
            -float(report["selection_rank_tuple"]["minimum_corresponding_one_sided_LCB"]),
            float(report["lambda_max"]),
            -float(report["prototype_prior_pseudocount"]),
            int(report["grid_cell_index"]),
        ),
    )


def _summarize_grid_cell(
    cell: v3.V3GridCell,
    rows: Sequence[Mapping[str, Any]],
    invariant_rows: Sequence[Mapping[str, Any]],
    *,
    uniform_rows: Sequence[Mapping[str, Any]] | None = None,
    sensitivity_rng_factory: RngFactory | None = None,
    _issuer: object | None = None,
) -> dict[str, Any]:
    if (uniform_rows is None) != (sensitivity_rng_factory is None):
        raise ValueError("uniform and resampling sensitivities must be reported together.")
    if uniform_rows is not None and _issuer is not _SYNTHETIC_RNG_PRODUCER_ISSUER:
        raise TypeError("authorized gate-report enrichment issuer token is invalid.")
    index = {_metric_lookup_key(row): row for row in rows}
    if len(index) != len(rows):
        raise ValueError("duplicate participant rows reached grid reduction.")

    def vector(
        family: str,
        condition: str,
        left: tuple[str, str, str],
        right: tuple[str, str, str],
        *,
        metric: str = "balanced_accuracy",
        budget: int | None = None,
        eauc: bool = False,
        participant_subset: Sequence[int] = tuple(range(DEVELOPMENT_PARTICIPANTS)),
    ) -> np.ndarray:
        values = []
        for participant in participant_subset:
            if eauc:
                left_value = _participant_eauc(
                    index, family, participant, condition, *left, metric=metric
                )
                right_value = _participant_eauc(
                    index, family, participant, condition, *right, metric=metric
                )
            else:
                if budget is None:
                    raise ValueError("a point contrast requires an exact budget.")
                left_value = float(
                    index[(family, participant, condition, *left, budget)][metric]
                )
                right_value = float(
                    index[(family, participant, condition, *right, budget)][metric]
                )
            values.append(left_value - right_value)
        return np.asarray(values, dtype=np.float64)

    aq_stats: dict[str, dict[str, Any]] = {}
    aq_vectors: dict[str, np.ndarray] = {}
    for family in ("B1_participant_class_confusion", "B2_participant_phase_spatial_shift"):
        delta = vector(
            family,
            "neutral",
            ("A_Q", "none", "deployed"),
            ("A0", "none", "none"),
            eauc=True,
        )
        k1 = vector(
            family,
            "neutral",
            ("A_Q", "none", "deployed"),
            ("A0", "none", "none"),
            budget=1,
        )
        summary = _one_sided_summary(delta)
        summary["k1_observed_mean"] = float(np.mean(k1))
        summary["passed"] = bool(
            summary["one_sided_95_LCB"] > 0.0
            and summary["observed_mean"] >= 0.01
            and summary["k1_observed_mean"] >= 0.0
        )
        aq_stats[family] = summary
        aq_vectors[family] = delta
    aq_viability_pass = any(summary["passed"] for summary in aq_stats.values())

    metadata_stats: dict[str, dict[str, Any]] = {}
    metadata_vectors: dict[str, np.ndarray] = {}
    for family, condition in (
        ("B3_impedance_linked_transfer_shift", "neutral"),
        ("B4_interface_calibrated_impedance_shift", "equal_condition_composite"),
    ):
        delta = vector(
            family,
            condition,
            ("A_QM", "correct", "deployed"),
            ("A_Q", "none", "deployed"),
            eauc=True,
        )
        summary = _one_sided_summary(delta)
        summary["passed"] = bool(
            summary["one_sided_95_LCB"] > 0.0 and summary["observed_mean"] >= 0.01
        )
        metadata_stats[family] = summary
        metadata_vectors[family] = delta

    b4_in_reference = tuple(
        participant
        for participant in range(DEVELOPMENT_PARTICIPANTS)
        if not bool(
            index[
                (
                    "B4_interface_calibrated_impedance_shift",
                    participant,
                    "equal_condition_composite",
                    "A_QM",
                    "correct",
                    "deployed",
                    1,
                )
            ]["source_range_stress"]
        )
    )
    if len(b4_in_reference) != 43:
        raise ValueError("B4 in-reference development subset must contain exactly 43 participants.")
    b4_in_reference_delta = vector(
        "B4_interface_calibrated_impedance_shift",
        "equal_condition_composite",
        ("A_QM", "correct", "deployed"),
        ("A_Q", "none", "deployed"),
        eauc=True,
        participant_subset=b4_in_reference,
    )
    b4_in_reference_summary = _one_sided_summary(b4_in_reference_delta)
    b4_in_reference_summary["passed"] = bool(
        b4_in_reference_summary["one_sided_95_LCB"] > 0.0
        and b4_in_reference_summary["observed_mean"] >= 0.01
    )

    scale_stats = {}
    for control in ("pooled_scale", "wrong_interface_scale"):
        delta = vector(
            "B4_interface_calibrated_impedance_shift",
            "equal_condition_composite",
            ("A_QM", "correct", "deployed"),
            ("A_QM", control, "deployed"),
            eauc=True,
            participant_subset=b4_in_reference,
        )
        summary = _one_sided_summary(delta)
        summary["passed"] = bool(
            summary["one_sided_95_LCB"] > 0.0 and summary["observed_mean"] > 0.0
        )
        scale_stats[control] = summary
    scale_value_pass = all(summary["passed"] for summary in scale_stats.values())

    mechanism_stats = {}
    mechanism_vectors = {}
    for family, condition in (
        ("B3_impedance_linked_transfer_shift", "neutral"),
        ("B4_interface_calibrated_impedance_shift", "equal_condition_composite"),
    ):
        delta = vector(
            family,
            condition,
            ("A_QM", "correct", "deployed"),
            ("A_QM", "within_prefix_packet_shuffle", "deployed"),
            budget=3,
        )
        summary = _one_sided_summary(delta)
        summary["passed"] = bool(
            summary["one_sided_95_LCB"] > 0.0 and summary["observed_mean"] >= 0.01
        )
        mechanism_stats[family] = summary
        mechanism_vectors[family] = delta

    mechanism_holm = holm_step_down(
        {
            family: summary["one_sided_t_p_value"]
            for family, summary in mechanism_stats.items()
        },
        familywise_alpha=0.05,
    )
    for family, summary in mechanism_stats.items():
        correction = mechanism_holm[family]
        summary.update(correction)
        summary["passed"] = bool(
            summary["one_sided_95_LCB"] > 0.0
            and summary["observed_mean"] >= 0.01
            and correction["holm_reject"]
        )

    deployment_stats = {}
    for family, condition in (
        ("B3_impedance_linked_transfer_shift", "neutral"),
        ("B4_interface_calibrated_impedance_shift", "equal_condition_composite"),
    ):
        delta = vector(
            family,
            condition,
            ("A_QM", "correct", "deployed"),
            ("A0", "none", "none"),
            eauc=True,
        )
        summary = _one_sided_summary(delta)
        summary["passed"] = bool(
            summary["one_sided_95_LCB"] > -(1.0 / 60.0)
            and summary["observed_mean"] >= 0.0
        )
        deployment_stats[family] = summary

    n1_stats = {}
    for contrast_id, left, right in (
        ("A_Q_minus_A0", ("A_Q", "none", "deployed"), ("A0", "none", "none")),
        (
            "A_QM_correct_minus_A_Q",
            ("A_QM", "correct", "deployed"),
            ("A_Q", "none", "deployed"),
        ),
    ):
        by_budget = {}
        for budget in (1, 3):
            summary = _one_sided_summary(
                vector("N1_clean_anchor", "neutral", left, right, budget=budget)
            )
            summary["passed"] = summary["one_sided_95_LCB"] > -(1.0 / 60.0)
            by_budget[str(budget)] = summary
        n1_stats[contrast_id] = by_budget
    n1_pass = all(
        summary["passed"] for by_budget in n1_stats.values() for summary in by_budget.values()
    )

    n4_stats = {}
    for contrast_id, left, right, budgets in (
        (
            "A_QM_correct_minus_A_Q",
            ("A_QM", "correct", "deployed"),
            ("A_Q", "none", "deployed"),
            (1, 3),
        ),
        (
            "correct_minus_support_channel_rotation",
            ("A_QM", "correct", "deployed"),
            ("A_QM", "support_impedance_channel_rotation", "deployed"),
            (1,),
        ),
        (
            "correct_minus_within_prefix_shuffle",
            ("A_QM", "correct", "deployed"),
            ("A_QM", "within_prefix_packet_shuffle", "deployed"),
            (3,),
        ),
    ):
        by_budget = {}
        for budget in budgets:
            values = vector("N4_context_null", "neutral", left, right, budget=budget)
            lower, upper = _paired_90_ci(values)
            by_budget[str(budget)] = {
                "n": len(values),
                "observed_mean": float(np.mean(values)),
                "paired_90_CI_lower": lower,
                "paired_90_CI_upper": upper,
                "passed": bool(lower > -(1.0 / 60.0) and upper < (1.0 / 60.0)),
            }
        n4_stats[contrast_id] = by_budget
    n4_pass = all(
        summary["passed"] for by_budget in n4_stats.values() for summary in by_budget.values()
    )

    potency_stats = {}
    for family, conditions, expected_units in (
        ("B3_impedance_linked_transfer_shift", ("neutral",), 240),
        ("B4_interface_calibrated_impedance_shift", ("wet", "dry"), 480),
    ):
        potency_rows = [
            row
            for row in rows
            if row["family"] == family
            and row["condition"] in conditions
            and row["role"] == "A_QM"
            and row["control"] == "within_prefix_packet_shuffle"
            and row["gate"] == "deployed"
            and row["budget"] == 3
        ]
        potential = sum(int(row["pairing_potential_unit_count"]) for row in potency_rows)
        changed = sum(int(row["pairing_scientifically_changed_count"]) for row in potency_rows)
        covered = sum(int(row["context_packet_covered_count"]) for row in potency_rows)
        changes = [
            float(value)
            for row in potency_rows
            for value in row["pairing_mean_derangement_abs_g_M_change_by_query_block"]
        ]
        binding_ok = all(row["pairing_packet_binding_changed_fraction"] == 1.0 for row in potency_rows)
        if potential != expected_units or len(changes) != expected_units:
            raise ValueError("pairing potency denominator differs from the frozen contract.")
        summary = {
            "potential_units": potential,
            "context_packet_coverage_fraction": covered / potential,
            "scientifically_changed_fraction": changed / potential,
            "median_mean_derangement_abs_g_M_change": float(np.median(changes)),
            "packet_binding_changed_fraction": 1.0 if binding_ok else 0.0,
        }
        summary["passed"] = bool(
            binding_ok
            and summary["context_packet_coverage_fraction"] >= 0.90
            and summary["scientifically_changed_fraction"] >= 0.50
            and summary["median_mean_derangement_abs_g_M_change"] >= 0.05
        )
        potency_stats[family] = summary

    anti_triviality = {}
    for family, conditions, expected_denominator in (
        ("B3_impedance_linked_transfer_shift", ("neutral",), 480),
        ("B4_interface_calibrated_impedance_shift", ("wet", "dry"), 960),
    ):
        helpful_rows = [
            row
            for row in rows
            if row["family"] == family
            and row["condition"] in conditions
            and row["role"] == "A_QM"
            and row["control"] == "correct"
            and row["gate"] == "deployed"
            and row["budget"] in (1, 3)
        ]
        numerator = sum(int(row["helpful_metadata_use_count"]) for row in helpful_rows)
        denominator = sum(int(row["helpful_metadata_use_denominator"]) for row in helpful_rows)
        if denominator != expected_denominator:
            raise ValueError("helpful-use denominator differs from the frozen contract.")
        observed_rate = numerator / denominator
        anti_triviality[family] = {
            "numerator": numerator,
            "denominator": denominator,
            "observed_rate": observed_rate,
            "passed": observed_rate >= 0.20,
        }

    abstention = {}
    for family in ("N2_random_support_labels", "N3_nonstationary_calibration"):
        relevant = [
            row
            for row in rows
            if row["family"] == family
            and row["condition"] == "neutral"
            and row["role"] == "A_Q"
            and row["control"] == "none"
            and row["gate"] == "deployed"
            and row["budget"] in (3, 5)
        ]
        if len(relevant) != 96:
            raise ValueError("adversarial abstention denominator must be 96.")
        successes = sum(row["exact_A0_query_block_fraction"] == 1.0 for row in relevant)
        rate = successes / len(relevant)
        abstention[family] = {
            "numerator": successes,
            "denominator": len(relevant),
            "observed_rate": rate,
            "one_sided_exact_95_LCB": _clopper_pearson_lower(successes, len(relevant)),
            "passed": rate >= 0.95,
        }

    b4_stress_rows = [
        row
        for row in rows
        if row["family"] == "B4_interface_calibrated_impedance_shift"
        and row["condition"] in {"wet", "dry"}
        and row["source_range_stress"]
        and row["role"] == "A_QM"
        and row["control"] == "correct"
        and row["gate"] == "deployed"
        and row["budget"] in (1, 3, 5)
    ]
    if len(b4_stress_rows) != 5 * 2 * 3:
        raise ValueError("B4 stress safety denominator must be exactly 30 rows.")
    b4_stress_safety = {
        "rows": len(b4_stress_rows),
        "exact_A0_rows": sum(
            row["exact_A0_query_block_fraction"] == 1.0 for row in b4_stress_rows
        ),
        "helpful_metadata_use_count": sum(
            int(row["helpful_metadata_use_count"]) for row in b4_stress_rows
        ),
    }
    b4_stress_safety["passed"] = bool(
        b4_stress_safety["exact_A0_rows"] == len(b4_stress_rows)
        and b4_stress_safety["helpful_metadata_use_count"] == 0
    )

    harm_specs = (
        ("B1_A_Q_minus_A0", "B1_participant_class_confusion", "neutral", ("A_Q", "none", "deployed"), ("A0", "none", "none")),
        ("B2_A_Q_minus_A0", "B2_participant_phase_spatial_shift", "neutral", ("A_Q", "none", "deployed"), ("A0", "none", "none")),
        ("B3_A_QM_minus_A_Q", "B3_impedance_linked_transfer_shift", "neutral", ("A_QM", "correct", "deployed"), ("A_Q", "none", "deployed")),
        ("B3_A_QM_minus_A0", "B3_impedance_linked_transfer_shift", "neutral", ("A_QM", "correct", "deployed"), ("A0", "none", "none")),
        ("B4_A_QM_minus_A_Q", "B4_interface_calibrated_impedance_shift", "equal_condition_composite", ("A_QM", "correct", "deployed"), ("A_Q", "none", "deployed")),
        ("B4_A_QM_minus_A0", "B4_interface_calibrated_impedance_shift", "equal_condition_composite", ("A_QM", "correct", "deployed"), ("A0", "none", "none")),
        ("B4_pooled_minus_A_Q", "B4_interface_calibrated_impedance_shift", "equal_condition_composite", ("A_QM", "pooled_scale", "deployed"), ("A_Q", "none", "deployed")),
        ("B4_wrong_minus_A_Q", "B4_interface_calibrated_impedance_shift", "equal_condition_composite", ("A_QM", "wrong_interface_scale", "deployed"), ("A_Q", "none", "deployed")),
    )
    harm = {}
    for contrast_id, family, condition, left, right in harm_specs:
        values = vector(family, condition, left, right, eauc=True)
        harmed = int(np.sum(values < -0.10))
        upper = _clopper_pearson_upper(harmed, len(values))
        harm[contrast_id] = {
            "severely_harmed": harmed,
            "denominator": len(values),
            "one_sided_exact_95_upper_rate": upper,
            "passed": upper < 0.10,
        }

    invariants_pass = len(invariant_rows) == 10 and all(row["passed"] for row in invariant_rows)
    components = {
        "invariant_suite": invariants_pass,
        "pairing_potency": all(summary["passed"] for summary in potency_stats.values()),
        "A_Q_viability_B1_or_B2": aq_viability_pass,
        "B3_metadata_efficacy": metadata_stats["B3_impedance_linked_transfer_shift"]["passed"],
        "B4_metadata_efficacy": metadata_stats["B4_interface_calibrated_impedance_shift"]["passed"],
        "B4_in_reference_metadata_efficacy": b4_in_reference_summary["passed"],
        "B4_source_range_stress_safety": b4_stress_safety["passed"],
        "B4_interface_scale_value": scale_value_pass,
        "k3_pairing_mechanism": all(summary["passed"] for summary in mechanism_stats.values()),
        "B3_B4_deployment_viability": all(summary["passed"] for summary in deployment_stats.values()),
        "N1_null_noninferiority": n1_pass,
        "N4_null_equivalence": n4_pass,
        "severe_harm": all(summary["passed"] for summary in harm.values()),
        "adversarial_abstention": all(summary["passed"] for summary in abstention.values()),
        "helpful_use_anti_triviality": all(
            summary["passed"] for summary in anti_triviality.values()
        ),
    }
    eligible = all(components.values())
    q_mean = max(aq_stats[family]["observed_mean"] for family in aq_stats)
    q_lcb = max(aq_stats[family]["one_sided_95_LCB"] for family in aq_stats)
    rank_mean = min(
        q_mean,
        metadata_stats["B3_impedance_linked_transfer_shift"]["observed_mean"],
        metadata_stats["B4_interface_calibrated_impedance_shift"]["observed_mean"],
        mechanism_stats["B3_impedance_linked_transfer_shift"]["observed_mean"],
        mechanism_stats["B4_interface_calibrated_impedance_shift"]["observed_mean"],
    )
    rank_lcb = min(
        q_lcb,
        metadata_stats["B3_impedance_linked_transfer_shift"]["one_sided_95_LCB"],
        metadata_stats["B4_interface_calibrated_impedance_shift"]["one_sided_95_LCB"],
        mechanism_stats["B3_impedance_linked_transfer_shift"]["one_sided_95_LCB"],
        mechanism_stats["B4_interface_calibrated_impedance_shift"]["one_sided_95_LCB"],
    )
    report = {
        "schema": "cfeg.metadata-calibration-efficiency-v3.grid-gate-report.v1",
        "grid_cell_index": cell.index,
        "grid_cell_id": cell.grid_cell_id,
        "operator_instance_sha256": cell.operator_instance_sha256,
        "prototype_prior_pseudocount": cell.prototype_prior_pseudocount,
        "lambda_max": cell.lambda_max,
        "components": components,
        "A_Q_viability": aq_stats,
        "metadata_efficacy": metadata_stats,
        "B4_in_reference_metadata_efficacy": b4_in_reference_summary,
        "B4_source_range_stress_safety": b4_stress_safety,
        "B4_interface_scale_value": scale_stats,
        "pairing_mechanism": mechanism_stats,
        "deployment_viability": deployment_stats,
        "pairing_potency": potency_stats,
        "N1_null_noninferiority": n1_stats,
        "N4_null_equivalence": n4_stats,
        "severe_harm": harm,
        "adversarial_abstention": abstention,
        "helpful_use_anti_triviality": anti_triviality,
        "selection_rank_tuple": {
            "minimum_mandatory_observed_gain": rank_mean,
            "minimum_corresponding_one_sided_LCB": rank_lcb,
        },
        "eligible": eligible,
    }
    if uniform_rows is not None and sensitivity_rng_factory is not None:
        primary_sensitivity_vectors = _sensitivity_vectors_from_metric_rows(
            rows,
            grid_cell_index=cell.index,
        )
        uniform_sensitivity_vectors = _sensitivity_vectors_from_metric_rows(
            uniform_rows,
            grid_cell_index=cell.index,
        )
        primary_values_by_endpoint = {
            endpoint: primary_sensitivity_vectors[endpoint].tolist()
            for endpoint in SENSITIVITY_ENDPOINT_IDS
        }
        uniform_values_by_endpoint = {
            endpoint: uniform_sensitivity_vectors[endpoint].tolist()
            for endpoint in SENSITIVITY_ENDPOINT_IDS
        }
        report["sensitivity_participant_vectors"] = {
            "schema": (
                "cfeg.metadata-calibration-efficiency-v3."
                "sensitivity-participant-vectors.v1"
            ),
            "participant_count": DEVELOPMENT_PARTICIPANTS,
            "endpoint_order": list(SENSITIVITY_ENDPOINT_IDS),
            "primary_values_by_endpoint": primary_values_by_endpoint,
            "uniform_values_by_endpoint": uniform_values_by_endpoint,
            "primary_values_sha256": _sensitivity_vector_values_sha256(
                "primary", primary_values_by_endpoint
            ),
            "uniform_values_sha256": _sensitivity_vector_values_sha256(
                "uniform", uniform_values_by_endpoint
            ),
        }
        report["uniform_block_weight_sensitivity"] = {
            "schema": (
                "cfeg.metadata-calibration-efficiency-v3."
                "uniform-block-weight-sensitivity.v1"
            ),
            "promotion_or_selection_use": False,
            "common_weight_rule": "both_p_support_and_g_M_use_exact_one_over_k",
            "endpoints": {
                endpoint: {
                    "metric": (
                        "participant_eAUC"
                        if endpoint.endswith("eAUC")
                        else "participant_k3_balanced_accuracy"
                    ),
                    "n": DEVELOPMENT_PARTICIPANTS,
                    "reliability_weighted_observed_mean": float(
                        np.mean(primary_sensitivity_vectors[endpoint])
                    ),
                    "uniform_weight_observed_mean": float(
                        np.mean(uniform_sensitivity_vectors[endpoint])
                    ),
                    "paired_weighted_minus_uniform_mean": float(
                        np.mean(
                            primary_sensitivity_vectors[endpoint]
                            - uniform_sensitivity_vectors[endpoint]
                        )
                    ),
                }
                for endpoint in SENSITIVITY_ENDPOINT_IDS
            },
        }
        report["resampling_sensitivity"] = _sensitivity_resampling_with_rng_factory(
            primary_sensitivity_vectors,
            grid_cell_index=cell.index,
            rng_factory=sensitivity_rng_factory,
            sign_flip_draws=SENSITIVITY_SIGN_FLIP_DRAWS,
            participant_bootstrap_draws=(
                SENSITIVITY_PARTICIPANT_BOOTSTRAP_DRAWS
            ),
            require_complete=True,
            _issuer=_SYNTHETIC_RNG_PRODUCER_ISSUER,
        )
    _validate_gate_report(report)
    return report


def execute_complete_development(
    contract: SyntheticV3Contract,
    reference: v3.ContextReference,
    *,
    rng_authority: v3.DevelopmentRNGAuthority,
    validated_reference: v3.ValidatedContextReference,
    context_reference_file_sha256: str,
    progress_callback: Callable[[Mapping[str, Any]], None] | None = None,
) -> tuple[dict[str, Any], ValidatedDevelopmentResult]:
    """Execute the sole authoritative 48x8x9 development pipeline in memory."""

    contract = _require_current_contract_files(contract)
    authority = v3.require_development_rng_authority(rng_authority)
    _require_loaded_synthetic_source(authority)
    reference = v3._require_context_reference(reference)
    proof = v3.require_validated_context_reference(
        validated_reference,
        expected_payload_sha256=reference.payload_sha256,
    )
    _require_shared_bundle_provenance(authority, proof)
    _sha256(context_reference_file_sha256, "context_reference_file_sha256")
    if progress_callback is not None and not callable(progress_callback):
        raise TypeError("progress_callback must be callable or None.")

    rng_factory = _authorized_rng_factory(authority)
    stress_indices = _stress_participant_indices_with_rng_factory(
        rng_factory,
        participant_count=DEVELOPMENT_PARTICIPANTS,
        _issuer=_SYNTHETIC_RNG_PRODUCER_ISSUER,
    )
    stress_set = set(stress_indices)
    primary_rows: list[dict[str, Any]] = []
    uniform_rows: list[dict[str, Any]] = []
    progress_events: list[Mapping[str, Any]] = []
    for family in FAMILY_NAMES:
        for participant_index in range(DEVELOPMENT_PARTICIPANTS):
            participant = _generate_synthetic_participant_with_rng_factory(
                contract,
                family=family,
                participant_index=participant_index,
                rng_factory=rng_factory,
                rng_authority=authority,
                generation_scope="development",
                _issuer=_SYNTHETIC_RNG_PRODUCER_ISSUER,
            )
            if (
                family == "B4_interface_calibrated_impedance_shift"
                and participant_index in stress_set
            ):
                participant = _apply_source_range_stress_impl(
                    participant,
                    reference,
                    _issuer=_SOURCE_RANGE_STRESS_ISSUER,
                )
            query_product = _produce_strict_fbcca_after_authorized_boundary(
                contract,
                participant,
                _issuer=_STRICT_FBCCA_COMPUTATION_ISSUER,
            )
            support_loader = (
                _incremental_strict_fbcca_support_loader_after_authorized_boundary(
                    contract,
                    participant,
                    _issuer=_STRICT_FBCCA_COMPUTATION_ISSUER,
                )
            )

            for cell in v3.canonical_operator_grid():
                primary_rows.extend(
                    _evaluate_synthetic_participant_impl(
                        contract,
                        participant,
                        query_product,
                        reference,
                        cell,
                        validated_reference=proof,
                        support_loader=support_loader,
                        block_weight_mode="reliability",
                        production_rng_authority=authority,
                    )
                )
                uniform_rows.extend(
                    _evaluate_synthetic_participant_impl(
                        contract,
                        participant,
                        query_product,
                        reference,
                        cell,
                        validated_reference=proof,
                        support_loader=support_loader,
                        block_weight_mode="uniform",
                        production_rng_authority=authority,
                    )
                )
            if progress_callback is not None:
                progress_events.append(
                    MappingProxyType(
                        {
                            "stage": "participant_complete",
                            "family": family,
                            "participant_index": participant_index,
                            "participant_count": DEVELOPMENT_PARTICIPANTS,
                        }
                    )
                )

    invariant_rows = [
        row
        for cell in v3.canonical_operator_grid()
        for row in _build_invariant_rows_impl(
            reference,
            cell,
            validated_reference=proof,
        )
    ]
    canonical_primary = canonicalize_development_rows(primary_rows, invariant_rows)
    metric_rows, canonical_invariants = validate_complete_development_rows(
        canonical_primary,
        expected_stress_participant_indices=stress_indices,
    )
    canonical_uniform = sorted(uniform_rows, key=_participant_row_sort_key)
    canonical_uniform = _validate_complete_uniform_sensitivity_rows(
        canonical_uniform,
        expected_stress_participant_indices=stress_indices,
    )
    reports = _summarize_complete_development_authorized(
        metric_rows,
        canonical_invariants,
        canonical_uniform,
        rng_authority=authority,
        _issuer=_SYNTHETIC_RNG_PRODUCER_ISSUER,
    )
    # A long execution may outlive a source/runtime checkout.  Revalidate the
    # exact current-A governance authority and both loaded module byte images
    # immediately before assembling or issuing the authoritative proof.
    contract = _require_current_contract_files(contract)
    authority = v3.require_development_rng_authority(authority)
    _require_loaded_synthetic_source(authority)
    proof = v3.require_validated_context_reference(
        proof,
        expected_payload_sha256=reference.payload_sha256,
    )
    _require_shared_bundle_provenance(authority, proof)
    payload = _assemble_development_result_payload(
        contract,
        metric_rows=metric_rows,
        invariant_rows=canonical_invariants,
        reports=reports,
        expected_stress_participant_indices=stress_indices,
        rng_authority=authority,
        context_reference=reference,
        context_reference_file_sha256=context_reference_file_sha256,
        _issuer=_VALIDATED_DEVELOPMENT_RESULT_ISSUER,
    )
    result_proof = _issue_validated_development_result(
        payload,
        rng_authority=authority,
        validated_reference=proof,
        _issuer=_VALIDATED_DEVELOPMENT_RESULT_ISSUER,
    )
    # Callbacks are delayed until every result byte and nominal proof is fixed;
    # arbitrary callback behavior therefore cannot affect computation.
    if progress_callback is not None:
        for event in progress_events:
            progress_callback(event)
    return payload, result_proof


def _execute_development_smoke_for_test(
    contract: SyntheticV3Contract,
    reference: v3.ContextReference,
    *,
    rng_authority: _UnitTestRNGAuthority,
    families: Sequence[str],
    participant_count: int,
    grid_cell_indices: Sequence[int],
) -> dict[str, Any]:
    """Exercise real orchestration on a tiny scope without production authority."""

    contract = _require_exact_contract(contract)
    authority = _require_unit_test_rng_authority(rng_authority)
    reference = v3._require_context_reference(reference)
    if (
        isinstance(families, (str, bytes))
        or not isinstance(families, Sequence)
        or not families
        or tuple(dict.fromkeys(families)) != tuple(families)
        or any(family not in FAMILY_NAMES for family in families)
    ):
        raise ValueError("smoke families must be a unique non-empty declared sequence.")
    participants = _positive_exact_int(participant_count, "participant_count")
    if participants > 2:
        raise ValueError("unit orchestration smoke permits at most two participants.")
    if isinstance(grid_cell_indices, (str, bytes)) or not isinstance(
        grid_cell_indices, Sequence
    ):
        raise TypeError("grid_cell_indices must be a sequence.")
    indices = tuple(grid_cell_indices)
    if (
        not indices
        or tuple(sorted(set(indices))) != indices
        or any(type(index) is not int or index not in range(1, 10) for index in indices)
    ):
        raise ValueError("smoke grid indices must be sorted unique values in 1..9.")

    rng_factory = _unit_test_rng_factory(authority)
    primary_rows: list[dict[str, Any]] = []
    uniform_rows: list[dict[str, Any]] = []
    for family in families:
        for participant_index in range(participants):
            participant = _generate_synthetic_participant_with_rng_factory(
                contract,
                family=family,
                participant_index=participant_index,
                rng_factory=rng_factory,
                rng_authority=authority,
                generation_scope="unit_test",
                _issuer=_SYNTHETIC_RNG_PRODUCER_ISSUER,
            )
            query_product = _produce_strict_fbcca_after_authorized_boundary(
                contract,
                participant,
                _issuer=_STRICT_FBCCA_COMPUTATION_ISSUER,
            )
            support_loader = (
                _incremental_strict_fbcca_support_loader_after_authorized_boundary(
                    contract,
                    participant,
                    _issuer=_STRICT_FBCCA_COMPUTATION_ISSUER,
                )
            )

            for index in indices:
                cell = v3.canonical_operator_grid()[index - 1]
                primary_rows.extend(
                    _evaluate_synthetic_participant_impl(
                        contract,
                        participant,
                        query_product,
                        reference,
                        cell,
                        validated_reference=None,
                        support_loader=support_loader,
                        block_weight_mode="reliability",
                        production_rng_authority=None,
                    )
                )
                uniform_rows.extend(
                    _evaluate_synthetic_participant_impl(
                        contract,
                        participant,
                        query_product,
                        reference,
                        cell,
                        validated_reference=None,
                        support_loader=support_loader,
                        block_weight_mode="uniform",
                        production_rng_authority=None,
                    )
                )
    report: dict[str, Any] = {
        "schema": (
            "cfeg.metadata-calibration-efficiency-v3.unit-orchestration-smoke.v1"
        ),
        "scope": "unit_test_only_never_selection_or_publication",
        "root_seed": authority.root_seed,
        "families": list(families),
        "participant_count": participants,
        "grid_cell_indices": list(indices),
        "primary_row_count": len(primary_rows),
        "uniform_sensitivity_row_count": len(uniform_rows),
        "primary_rows_sha256": _canonical_sha256(
            {"schema": PARTICIPANT_METRIC_SCHEMA, "rows": primary_rows}
        ),
        "uniform_sensitivity_rows_sha256": _canonical_sha256(
            {"schema": PARTICIPANT_METRIC_SCHEMA, "rows": uniform_rows}
        ),
        "selection_status": "FORBIDDEN_IN_UNIT_SMOKE",
        "nominal_development_proof_issued": False,
    }
    report["payload_sha256"] = _payload_sha256(report)
    return report


def _build_development_result_payload_from_rows_for_test(
    contract: SyntheticV3Contract,
    *,
    rows: Sequence[Mapping[str, Any]],
    expected_stress_participant_indices: Sequence[int],
    development_bundle_reference: Mapping[str, Any],
    development_start_reference: Mapping[str, Any],
    context_reference: v3.ContextReference,
    context_reference_file_sha256: str,
) -> dict[str, Any]:
    """Pure unit helper; caller-supplied rows can never mint a nominal proof."""

    if type(contract) is not SyntheticV3Contract:
        raise TypeError("contract must be an exact SyntheticV3Contract.")
    validate_development_contract(contract)
    bundle = _artifact_reference(
        development_bundle_reference,
        "development bundle reference",
        expected_schema="cfeg.metadata-calibration-efficiency-v3.development-bundle.v3",
    )
    start = _artifact_reference(
        development_start_reference,
        "development start reference",
        expected_schema="cfeg.metadata-calibration-efficiency-v3.development-start.v1",
    )
    if type(context_reference) is not v3.ContextReference:
        raise TypeError("context_reference must be an exact ContextReference.")
    _sha256(context_reference_file_sha256, "context_reference_file_sha256")
    canonical_rows = [dict(row) for row in rows]
    metric_rows, invariant_rows = validate_complete_development_rows(
        canonical_rows,
        expected_stress_participant_indices=expected_stress_participant_indices,
    )
    reports = summarize_complete_development(metric_rows, invariant_rows)
    selected = select_grid_cell(reports)
    grid_records = [_grid_record(cell) for cell in v3.canonical_operator_grid()]
    payload: dict[str, Any] = {
        "schema": DEVELOPMENT_RESULT_SCHEMA,
        "candidate_id": v3.CANDIDATE_ID,
        "generator_revision": "v1_single_insertion_context_trust",
        "master_plan_file_sha256": contract.master_plan_sha256,
        "synthetic_plan_file_sha256": contract.plan_sha256,
        "development_bundle_schema": bundle["schema"],
        "development_bundle_payload_sha256": bundle["payload_sha256"],
        "development_bundle_file_sha256": bundle["file_sha256"],
        "context_reference_schema": v3.CONTEXT_REFERENCE_SCHEMA,
        "context_reference_payload_sha256": context_reference.payload_sha256,
        "context_reference_file_sha256": context_reference_file_sha256,
        "development_start_schema": start["schema"],
        "development_start_payload_sha256": start["payload_sha256"],
        "development_start_file_sha256": start["file_sha256"],
        "development_rng_primitive_schema": DEVELOPMENT_RNG_PRIMITIVE_SCHEMA,
        "development_root_seed": 20260909,
        "participant_count": DEVELOPMENT_PARTICIPANTS,
        "B4_source_range_stress_participant_indices": list(
            expected_stress_participant_indices
        ),
        "grid_cells": grid_records,
        "participant_metric_rows": metric_rows,
        "invariant_rows": invariant_rows,
        "complete_grid_gate_report": reports,
        "selection_status": (
            "SELECTED_METHOD_PROPOSED" if selected is not None else "DEVELOPMENT_NO_GO"
        ),
        "selected_grid_cell_id": selected["grid_cell_id"] if selected is not None else None,
    }
    payload["payload_sha256"] = _payload_sha256(payload)
    validate_development_result_payload(payload)
    return payload


def _assemble_development_result_payload(
    contract: SyntheticV3Contract,
    *,
    metric_rows: Sequence[Mapping[str, Any]],
    invariant_rows: Sequence[Mapping[str, Any]],
    reports: Sequence[Mapping[str, Any]],
    expected_stress_participant_indices: Sequence[int],
    rng_authority: v3.DevelopmentRNGAuthority,
    context_reference: v3.ContextReference,
    context_reference_file_sha256: str,
    _issuer: object,
) -> dict[str, Any]:
    """Assemble only results emitted by the private authoritative executor."""

    if _issuer is not _VALIDATED_DEVELOPMENT_RESULT_ISSUER:
        raise TypeError("development-result assembly issuer token is invalid.")
    contract = _require_exact_contract(contract)
    authority = v3._require_development_rng_authority_semantics(rng_authority)
    reference = v3._require_context_reference(context_reference)
    _sha256(context_reference_file_sha256, "context_reference_file_sha256")
    if len(reports) != 9:
        raise ValueError("authorized development result requires nine gate reports.")
    report_values = [dict(report) for report in reports]
    if any(set(report) != _ENRICHED_GATE_REPORT_KEYS for report in report_values):
        raise ValueError("authorized gate reports must contain both sensitivities.")
    for report in report_values:
        _validate_gate_report(report)
    selected = select_grid_cell(report_values)
    payload: dict[str, Any] = {
        "schema": DEVELOPMENT_RESULT_SCHEMA,
        "candidate_id": v3.CANDIDATE_ID,
        "generator_revision": "v1_single_insertion_context_trust",
        "master_plan_file_sha256": contract.master_plan_sha256,
        "synthetic_plan_file_sha256": contract.plan_sha256,
        "development_bundle_schema": authority.development_bundle_schema,
        "development_bundle_payload_sha256": (
            authority.development_bundle_payload_sha256
        ),
        "development_bundle_file_sha256": authority.development_bundle_file_sha256,
        "context_reference_schema": v3.CONTEXT_REFERENCE_SCHEMA,
        "context_reference_payload_sha256": reference.payload_sha256,
        "context_reference_file_sha256": context_reference_file_sha256,
        "development_start_schema": authority.development_start_schema,
        "development_start_payload_sha256": authority.development_start_payload_sha256,
        "development_start_file_sha256": authority.development_start_file_sha256,
        "development_rng_primitive_schema": authority.rng_primitive_schema,
        "development_root_seed": authority.root_seed,
        "participant_count": DEVELOPMENT_PARTICIPANTS,
        "B4_source_range_stress_participant_indices": list(
            expected_stress_participant_indices
        ),
        "grid_cells": [_grid_record(cell) for cell in v3.canonical_operator_grid()],
        "participant_metric_rows": [dict(row) for row in metric_rows],
        "invariant_rows": [dict(row) for row in invariant_rows],
        "complete_grid_gate_report": report_values,
        "selection_status": (
            "SELECTED_METHOD_PROPOSED"
            if selected is not None
            else "DEVELOPMENT_NO_GO"
        ),
        "selected_grid_cell_id": (
            selected["grid_cell_id"] if selected is not None else None
        ),
    }
    payload["payload_sha256"] = _payload_sha256(payload)
    validate_development_result_payload(payload, require_enriched=True)
    return payload


_DEVELOPMENT_RESULT_KEYS = {
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


def validate_development_result_payload(
    payload: Mapping[str, Any],
    *,
    require_enriched: bool = False,
) -> dict[str, Any]:
    """Pure complete-case validation before a governance layer opens an output path."""

    if type(require_enriched) is not bool:
        raise TypeError("require_enriched must be an exact bool.")
    _require_exact_keys(payload, _DEVELOPMENT_RESULT_KEYS, "development result payload")
    if (
        payload["schema"] != DEVELOPMENT_RESULT_SCHEMA
        or payload["candidate_id"] != v3.CANDIDATE_ID
        or payload["generator_revision"] != "v1_single_insertion_context_trust"
        or payload["master_plan_file_sha256"] != EXPECTED_MASTER_PLAN_SHA256
        or payload["synthetic_plan_file_sha256"] != EXPECTED_SYNTHETIC_PLAN_SHA256
        or payload["development_bundle_schema"]
        != "cfeg.metadata-calibration-efficiency-v3.development-bundle.v3"
        or payload["context_reference_schema"] != v3.CONTEXT_REFERENCE_SCHEMA
        or payload["development_start_schema"]
        != "cfeg.metadata-calibration-efficiency-v3.development-start.v1"
        or payload["development_rng_primitive_schema"] != DEVELOPMENT_RNG_PRIMITIVE_SCHEMA
        or payload["development_root_seed"] != 20260909
        or payload["participant_count"] != DEVELOPMENT_PARTICIPANTS
    ):
        raise ValueError("development result identity/bindings are invalid.")
    for name in (
        "development_bundle_payload_sha256",
        "development_bundle_file_sha256",
        "context_reference_payload_sha256",
        "context_reference_file_sha256",
        "development_start_payload_sha256",
        "development_start_file_sha256",
        "payload_sha256",
    ):
        _sha256(payload[name], name)
    if _payload_sha256(payload) != payload["payload_sha256"]:
        raise ValueError("development result payload hash is invalid.")
    if payload["grid_cells"] != [_grid_record(cell) for cell in v3.canonical_operator_grid()]:
        raise ValueError("development result grid records are not exact.")
    stress = payload["B4_source_range_stress_participant_indices"]
    if not isinstance(stress, list):
        raise TypeError("B4 stress participant indices must be a list.")
    raw_metrics = payload["participant_metric_rows"]
    raw_invariants = payload["invariant_rows"]
    if not isinstance(raw_metrics, list) or not isinstance(raw_invariants, list):
        raise TypeError("development result row fields must be lists.")
    all_rows = [*raw_metrics, *raw_invariants]
    metrics, invariants = validate_complete_development_rows(
        all_rows,
        expected_stress_participant_indices=stress,
    )
    reports = summarize_complete_development(metrics, invariants)
    stored_reports = payload["complete_grid_gate_report"]
    if not isinstance(stored_reports, list) or len(stored_reports) != 9:
        raise ValueError("stored grid gate report must contain exactly nine reports.")
    enriched_flags = []
    for expected, stored in zip(reports, stored_reports, strict=True):
        if not isinstance(stored, Mapping):
            raise TypeError("each stored grid gate report must be a mapping.")
        _validate_gate_report(stored)
        if {name: stored[name] for name in _GATE_REPORT_KEYS} != expected:
            raise ValueError("stored primary gate report differs from exact recomputation.")
        enriched = set(stored) == _ENRICHED_GATE_REPORT_KEYS
        enriched_flags.append(enriched)
        if enriched:
            stored_primary, _ = _validated_sensitivity_participant_vectors(
                stored["sensitivity_participant_vectors"]
            )
            expected_primary = _sensitivity_vectors_from_metric_rows(
                metrics,
                grid_cell_index=int(stored["grid_cell_index"]),
            )
            for endpoint in SENSITIVITY_ENDPOINT_IDS:
                if not np.array_equal(
                    stored_primary[endpoint], expected_primary[endpoint]
                ):
                    raise ValueError(
                        "stored primary sensitivity vector differs from metric rows."
                    )
    if any(enriched_flags) and not all(enriched_flags):
        raise ValueError("gate-report sensitivities must be present for all or no cells.")
    if require_enriched and not all(enriched_flags):
        raise ValueError("canonical development results require all frozen sensitivities.")
    selected = select_grid_cell(stored_reports)
    expected_status = "SELECTED_METHOD_PROPOSED" if selected is not None else "DEVELOPMENT_NO_GO"
    expected_id = selected["grid_cell_id"] if selected is not None else None
    if payload["selection_status"] != expected_status or payload["selected_grid_cell_id"] != expected_id:
        raise ValueError("development result selection fields are invalid.")
    return dict(payload)


def _validated_development_result_binding(
    value: ValidatedDevelopmentResult,
) -> dict[str, Any]:
    return {
        "schema": value.schema,
        "candidate_id": value.candidate_id,
        "payload_schema": value.payload_schema,
        "payload_sha256": value.payload_sha256,
        "master_plan_file_sha256": value.master_plan_file_sha256,
        "synthetic_plan_file_sha256": value.synthetic_plan_file_sha256,
        "development_bundle_schema": value.development_bundle_schema,
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
        "context_reference_schema": value.context_reference_schema,
        "context_reference_payload_sha256": value.context_reference_payload_sha256,
        "context_reference_file_sha256": value.context_reference_file_sha256,
        "development_start_schema": value.development_start_schema,
        "development_start_payload_sha256": value.development_start_payload_sha256,
        "development_start_file_sha256": value.development_start_file_sha256,
        "development_rng_primitive_schema": value.development_rng_primitive_schema,
        "development_root_seed": value.development_root_seed,
        "development_rng_authority_schema": value.development_rng_authority_schema,
        "development_rng_authority_semantic_binding_sha256": (
            value.development_rng_authority_semantic_binding_sha256
        ),
        "rng_key_map_sha256": value.rng_key_map_sha256,
        "validated_context_reference_semantic_binding_sha256": (
            value.validated_context_reference_semantic_binding_sha256
        ),
        "participant_count": value.participant_count,
        "B4_source_range_stress_participant_indices": list(
            value.b4_source_range_stress_participant_indices
        ),
        "grid_cell_count": value.grid_cell_count,
        "metric_row_count": value.metric_row_count,
        "invariant_row_count": value.invariant_row_count,
        "total_row_count": value.total_row_count,
        "participant_metric_rows_sha256": value.participant_metric_rows_sha256,
        "invariant_rows_sha256": value.invariant_rows_sha256,
        "gate_report_count": value.gate_report_count,
        "complete_grid_gate_report_sha256": value.complete_grid_gate_report_sha256,
        "uniform_block_weight_sensitivity_sha256": (
            value.uniform_block_weight_sensitivity_sha256
        ),
        "resampling_sensitivity_sha256": value.resampling_sensitivity_sha256,
        "sensitivity_participant_vectors_sha256": (
            value.sensitivity_participant_vectors_sha256
        ),
        "eligible_grid_cell_ids": list(value.eligible_grid_cell_ids),
        "selection_status": value.selection_status,
        "selected_grid_cell_id": value.selected_grid_cell_id,
    }


def _issue_validated_development_result(
    result: Mapping[str, Any],
    *,
    rng_authority: v3.DevelopmentRNGAuthority,
    validated_reference: v3.ValidatedContextReference,
    _issuer: object,
) -> ValidatedDevelopmentResult:
    if _issuer is not _VALIDATED_DEVELOPMENT_RESULT_ISSUER:
        raise TypeError("validated-development-result issuer token is invalid.")
    result = validate_development_result_payload(result, require_enriched=True)
    if any(
        set(report) != _ENRICHED_GATE_REPORT_KEYS
        for report in result["complete_grid_gate_report"]
    ):
        raise ValueError("nominal development proof requires both frozen sensitivities.")
    authority = v3._require_development_rng_authority_semantics(rng_authority)
    reference = v3._require_validated_context_reference_semantics(
        validated_reference,
        expected_payload_sha256=result["context_reference_payload_sha256"],
    )
    _require_shared_bundle_provenance(authority, reference)
    if (
        result["development_bundle_schema"] != authority.development_bundle_schema
        or result["development_bundle_payload_sha256"]
        != authority.development_bundle_payload_sha256
        or result["development_bundle_file_sha256"]
        != authority.development_bundle_file_sha256
        or result["development_rng_primitive_schema"]
        != authority.rng_primitive_schema
        or result["development_start_schema"] != authority.development_start_schema
        or result["development_start_payload_sha256"] != authority.development_start_payload_sha256
        or result["development_start_file_sha256"] != authority.development_start_file_sha256
        or result["development_root_seed"] != authority.root_seed
    ):
        raise ValueError("development result differs from its RNG authority.")
    value = object.__new__(ValidatedDevelopmentResult)
    reports = result["complete_grid_gate_report"]
    fields: dict[str, Any] = {
        "schema": VALIDATED_DEVELOPMENT_RESULT_SCHEMA,
        "candidate_id": v3.CANDIDATE_ID,
        "payload_schema": result["schema"],
        "payload_sha256": result["payload_sha256"],
        "master_plan_file_sha256": result["master_plan_file_sha256"],
        "synthetic_plan_file_sha256": result["synthetic_plan_file_sha256"],
        "development_bundle_schema": result["development_bundle_schema"],
        "development_bundle_payload_sha256": result[
            "development_bundle_payload_sha256"
        ],
        "development_bundle_file_sha256": result[
            "development_bundle_file_sha256"
        ],
        "development_bundle_source_bundle_sha256": (
            authority.development_bundle_source_bundle_sha256
        ),
        "numerical_runtime_fingerprint_sha256": (
            authority.numerical_runtime_fingerprint_sha256
        ),
        "context_reference_schema": result["context_reference_schema"],
        "context_reference_payload_sha256": result[
            "context_reference_payload_sha256"
        ],
        "context_reference_file_sha256": result["context_reference_file_sha256"],
        "development_start_schema": result["development_start_schema"],
        "development_start_payload_sha256": result[
            "development_start_payload_sha256"
        ],
        "development_start_file_sha256": result["development_start_file_sha256"],
        "development_rng_primitive_schema": result[
            "development_rng_primitive_schema"
        ],
        "development_root_seed": result["development_root_seed"],
        "development_rng_authority_schema": authority.schema,
        "development_rng_authority_semantic_binding_sha256": (
            authority.semantic_binding_sha256
        ),
        "rng_key_map_sha256": authority.key_map_sha256,
        "validated_context_reference_semantic_binding_sha256": (
            reference.semantic_binding_sha256
        ),
        "participant_count": result["participant_count"],
        "b4_source_range_stress_participant_indices": tuple(
            result["B4_source_range_stress_participant_indices"]
        ),
        "grid_cell_count": len(result["grid_cells"]),
        "metric_row_count": len(result["participant_metric_rows"]),
        "invariant_row_count": len(result["invariant_rows"]),
        "total_row_count": len(result["participant_metric_rows"])
        + len(result["invariant_rows"]),
        "participant_metric_rows_sha256": _canonical_sha256(
            {
                "schema": (
                    "cfeg.metadata-calibration-efficiency-v3."
                    "participant-metric-row-set.v1"
                ),
                "rows": result["participant_metric_rows"],
            }
        ),
        "invariant_rows_sha256": _canonical_sha256(
            {
                "schema": (
                    "cfeg.metadata-calibration-efficiency-v3.invariant-row-set.v1"
                ),
                "rows": result["invariant_rows"],
            }
        ),
        "gate_report_count": len(reports),
        "complete_grid_gate_report_sha256": _canonical_sha256(
            {
                "schema": (
                    "cfeg.metadata-calibration-efficiency-v3."
                    "complete-grid-gate-report.v1"
                ),
                "reports": reports,
            }
        ),
        "uniform_block_weight_sensitivity_sha256": _canonical_sha256(
            {
                "schema": (
                    "cfeg.metadata-calibration-efficiency-v3."
                    "complete-uniform-block-weight-sensitivity.v1"
                ),
                "reports": [
                    report["uniform_block_weight_sensitivity"]
                    for report in reports
                ],
            }
        ),
        "resampling_sensitivity_sha256": _canonical_sha256(
            {
                "schema": (
                    "cfeg.metadata-calibration-efficiency-v3."
                    "complete-resampling-sensitivity.v1"
                ),
                "reports": [report["resampling_sensitivity"] for report in reports],
            }
        ),
        "sensitivity_participant_vectors_sha256": (
            _complete_sensitivity_participant_vectors_sha256(reports)
        ),
        "eligible_grid_cell_ids": tuple(
            report["grid_cell_id"] for report in reports if report["eligible"]
        ),
        "selection_status": result["selection_status"],
        "selected_grid_cell_id": result["selected_grid_cell_id"],
        "_rng_authority": authority,
        "_validated_reference": reference,
    }
    for name, item in fields.items():
        object.__setattr__(value, name, item)
    object.__setattr__(
        value,
        "semantic_binding_sha256",
        _canonical_sha256(_validated_development_result_binding(value)),
    )
    object.__setattr__(value, "_issuer", _VALIDATED_DEVELOPMENT_RESULT_ISSUER)
    return _require_validated_development_result(
        value,
        expected_payload_sha256=result["payload_sha256"],
        revalidate_bundle=False,
    )


def validate_development_result_for_publication(
    payload: Mapping[str, Any],
    *,
    validated_result: ValidatedDevelopmentResult,
) -> ValidatedDevelopmentResult:
    """Match a payload to the proof issued only by the full executor."""

    proof = require_validated_development_result(
        validated_result,
        expected_payload_sha256=None,
    )
    result = validate_development_result_payload(payload, require_enriched=True)
    if proof.payload_sha256 != result["payload_sha256"]:
        raise ValueError("ValidatedDevelopmentResult binds a different payload.")
    if (
        proof.participant_metric_rows_sha256
        != _canonical_sha256(
            {
                "schema": (
                    "cfeg.metadata-calibration-efficiency-v3."
                    "participant-metric-row-set.v1"
                ),
                "rows": result["participant_metric_rows"],
            }
        )
        or proof.invariant_rows_sha256
        != _canonical_sha256(
            {
                "schema": (
                    "cfeg.metadata-calibration-efficiency-v3.invariant-row-set.v1"
                ),
                "rows": result["invariant_rows"],
            }
        )
        or proof.complete_grid_gate_report_sha256
        != _canonical_sha256(
            {
                "schema": (
                    "cfeg.metadata-calibration-efficiency-v3."
                    "complete-grid-gate-report.v1"
                ),
                "reports": result["complete_grid_gate_report"],
            }
        )
        or proof.uniform_block_weight_sensitivity_sha256
        != _canonical_sha256(
            {
                "schema": (
                    "cfeg.metadata-calibration-efficiency-v3."
                    "complete-uniform-block-weight-sensitivity.v1"
                ),
                "reports": [
                    report["uniform_block_weight_sensitivity"]
                    for report in result["complete_grid_gate_report"]
                ],
            }
        )
        or proof.resampling_sensitivity_sha256
        != _canonical_sha256(
            {
                "schema": (
                    "cfeg.metadata-calibration-efficiency-v3."
                    "complete-resampling-sensitivity.v1"
                ),
                "reports": [
                    report["resampling_sensitivity"]
                    for report in result["complete_grid_gate_report"]
                ],
            }
        )
        or proof.sensitivity_participant_vectors_sha256
        != _complete_sensitivity_participant_vectors_sha256(
            result["complete_grid_gate_report"]
        )
    ):
        raise ValueError("development proof differs from payload evidence bytes.")
    return proof


def require_validated_development_result(
    value: object,
    *,
    expected_payload_sha256: str | None = None,
) -> ValidatedDevelopmentResult:
    """Require an authentic and internally bound exact development proof."""

    return _require_validated_development_result(
        value,
        expected_payload_sha256=expected_payload_sha256,
        revalidate_bundle=True,
    )


def _require_validated_development_result(
    value: object,
    *,
    expected_payload_sha256: str | None,
    revalidate_bundle: bool,
) -> ValidatedDevelopmentResult:

    if type(value) is not ValidatedDevelopmentResult:
        raise TypeError("value must be an exact ValidatedDevelopmentResult.")
    try:
        issuer = value._issuer
    except AttributeError as error:
        raise TypeError("ValidatedDevelopmentResult was not issued by this module.") from error
    if issuer is not _VALIDATED_DEVELOPMENT_RESULT_ISSUER:
        raise TypeError("ValidatedDevelopmentResult issuer marker is invalid.")
    if (
        value.schema != VALIDATED_DEVELOPMENT_RESULT_SCHEMA
        or value.candidate_id != v3.CANDIDATE_ID
        or value.payload_schema != DEVELOPMENT_RESULT_SCHEMA
        or value.master_plan_file_sha256 != EXPECTED_MASTER_PLAN_SHA256
        or value.synthetic_plan_file_sha256 != EXPECTED_SYNTHETIC_PLAN_SHA256
        or value.development_bundle_schema
        != "cfeg.metadata-calibration-efficiency-v3.development-bundle.v3"
        or value.context_reference_schema != v3.CONTEXT_REFERENCE_SCHEMA
        or value.development_rng_primitive_schema != DEVELOPMENT_RNG_PRIMITIVE_SCHEMA
        or value.development_start_schema
        != "cfeg.metadata-calibration-efficiency-v3.development-start.v1"
        or value.development_root_seed != 20260909
        or value.development_rng_authority_schema
        != v3.DEVELOPMENT_RNG_AUTHORITY_SCHEMA
        or value.rng_key_map_sha256 != v3.rng_key_map_sha256()
        or value.participant_count != DEVELOPMENT_PARTICIPANTS
    ):
        raise ValueError("ValidatedDevelopmentResult identity/bindings are invalid.")
    for name in (
        "payload_sha256",
        "master_plan_file_sha256",
        "synthetic_plan_file_sha256",
        "development_bundle_payload_sha256",
        "development_bundle_file_sha256",
        "development_bundle_source_bundle_sha256",
        "numerical_runtime_fingerprint_sha256",
        "context_reference_payload_sha256",
        "context_reference_file_sha256",
        "development_start_payload_sha256",
        "development_start_file_sha256",
        "development_rng_authority_semantic_binding_sha256",
        "rng_key_map_sha256",
        "validated_context_reference_semantic_binding_sha256",
        "participant_metric_rows_sha256",
        "invariant_rows_sha256",
        "complete_grid_gate_report_sha256",
        "uniform_block_weight_sensitivity_sha256",
        "resampling_sensitivity_sha256",
        "sensitivity_participant_vectors_sha256",
        "semantic_binding_sha256",
    ):
        _sha256(getattr(value, name), name)
    stress = value.b4_source_range_stress_participant_indices
    if (
        type(stress) is not tuple
        or len(stress) != 5
        or tuple(sorted(set(stress))) != stress
        or any(type(index) is not int or not 0 <= index < DEVELOPMENT_PARTICIPANTS for index in stress)
    ):
        raise ValueError("validated development stress indices are invalid.")
    if (
        value.grid_cell_count != 9
        or value.metric_row_count != DEVELOPMENT_METRIC_ROWS
        or value.invariant_row_count != DEVELOPMENT_INVARIANT_ROWS
        or value.total_row_count != DEVELOPMENT_TOTAL_ROWS
        or value.gate_report_count != 9
    ):
        raise ValueError("validated development counts are invalid.")
    canonical_ids = tuple(cell.grid_cell_id for cell in v3.canonical_operator_grid())
    eligible = value.eligible_grid_cell_ids
    if (
        type(eligible) is not tuple
        or len(set(eligible)) != len(eligible)
        or tuple(item for item in canonical_ids if item in set(eligible)) != eligible
    ):
        raise ValueError("validated development eligible IDs are invalid.")
    if value.selection_status == "SELECTED_METHOD_PROPOSED":
        if value.selected_grid_cell_id not in eligible:
            raise ValueError("selected cell must be an eligible grid cell.")
    elif value.selection_status == "DEVELOPMENT_NO_GO":
        if value.selected_grid_cell_id is not None or eligible:
            raise ValueError("DEVELOPMENT_NO_GO cannot bind an eligible/selected cell.")
    else:
        raise ValueError("validated development selection status is invalid.")
    expected_binding = _canonical_sha256(
        _validated_development_result_binding(value)
    )
    if value.semantic_binding_sha256 != expected_binding:
        raise ValueError("ValidatedDevelopmentResult semantic binding is invalid.")
    authority = (
        v3.require_development_rng_authority(value._rng_authority)
        if revalidate_bundle
        else v3._require_development_rng_authority_semantics(value._rng_authority)
    )
    if revalidate_bundle:
        _require_loaded_synthetic_source(authority)
        reference = v3.require_validated_context_reference(
            value._validated_reference,
            expected_payload_sha256=value.context_reference_payload_sha256,
        )
    else:
        reference = v3._require_validated_context_reference_semantics(
            value._validated_reference,
            expected_payload_sha256=value.context_reference_payload_sha256,
        )
    _require_shared_bundle_provenance(authority, reference)
    if (
        value.development_bundle_payload_sha256
        != authority.development_bundle_payload_sha256
        or value.development_bundle_file_sha256
        != authority.development_bundle_file_sha256
        or value.development_bundle_source_bundle_sha256
        != authority.development_bundle_source_bundle_sha256
        or value.numerical_runtime_fingerprint_sha256
        != authority.numerical_runtime_fingerprint_sha256
        or value.development_rng_authority_semantic_binding_sha256
        != authority.semantic_binding_sha256
        or value.rng_key_map_sha256 != authority.key_map_sha256
        or value.development_start_schema != authority.development_start_schema
        or value.development_start_payload_sha256 != authority.development_start_payload_sha256
        or value.development_start_file_sha256 != authority.development_start_file_sha256
        or value.validated_context_reference_semantic_binding_sha256
        != reference.semantic_binding_sha256
    ):
        raise ValueError("ValidatedDevelopmentResult authority provenance is invalid.")
    if expected_payload_sha256 is not None:
        _sha256(expected_payload_sha256, "expected_payload_sha256")
        if value.payload_sha256 != expected_payload_sha256:
            raise ValueError("ValidatedDevelopmentResult binds a different payload.")
    return value


def build_selected_method_freeze_payload(
    development_result: Mapping[str, Any],
    *,
    validated_development_result: ValidatedDevelopmentResult,
    development_result_file_sha256: str,
    clean_commit: str,
    clean_tree: str,
) -> dict[str, Any]:
    """Build the deterministic selected-method freeze after a development PASS."""

    validate_development_result_for_publication(
        development_result,
        validated_result=validated_development_result,
    )
    result = validate_development_result_payload(
        development_result,
        require_enriched=True,
    )
    return _assemble_selected_method_freeze_payload(
        result,
        development_result_file_sha256=development_result_file_sha256,
        clean_commit=clean_commit,
        clean_tree=clean_tree,
        _issuer=_SELECTED_FREEZE_ASSEMBLY_ISSUER,
    )


def _assemble_selected_method_freeze_payload(
    result: Mapping[str, Any],
    *,
    development_result_file_sha256: str,
    clean_commit: str,
    clean_tree: str,
    _issuer: object,
) -> dict[str, Any]:
    if _issuer is not _SELECTED_FREEZE_ASSEMBLY_ISSUER:
        raise TypeError("selected-freeze assembly issuer token is invalid.")
    _sha256(development_result_file_sha256, "development_result_file_sha256")
    _git_object_id(clean_commit, "clean_commit")
    _git_object_id(clean_tree, "clean_tree")
    if result["selection_status"] != "SELECTED_METHOD_PROPOSED":
        raise ValueError("a DEVELOPMENT_NO_GO result cannot produce a selected-method freeze.")
    report = next(
        item
        for item in result["complete_grid_gate_report"]
        if item["grid_cell_id"] == result["selected_grid_cell_id"]
    )
    cell = v3.grid_cell_by_id(report["grid_cell_id"])
    payload: dict[str, Any] = {
        "schema": SELECTED_METHOD_FREEZE_SCHEMA,
        "candidate_id": v3.CANDIDATE_ID,
        "scientific_candidate_id": (
            f"{v3.CANDIDATE_ID}-p3-nu_{cell.nu_token}-lambda_{cell.lambda_token}"
        ),
        "selected_grid_cell_index": cell.index,
        "selected_grid_cell_id": cell.grid_cell_id,
        "selected_prototype_prior_pseudocount": cell.prototype_prior_pseudocount,
        "selected_lambda_max": cell.lambda_max,
        "selected_operator_instance_sha256": cell.operator_instance_sha256,
        "master_plan_file_sha256": result["master_plan_file_sha256"],
        "synthetic_plan_file_sha256": result["synthetic_plan_file_sha256"],
        "development_bundle_schema": result["development_bundle_schema"],
        "development_bundle_payload_sha256": result["development_bundle_payload_sha256"],
        "development_bundle_file_sha256": result["development_bundle_file_sha256"],
        "development_result_schema": result["schema"],
        "development_result_payload_sha256": result["payload_sha256"],
        "development_result_file_sha256": development_result_file_sha256,
        "complete_grid_gate_report_sha256": _canonical_sha256(
            {
                "schema": "cfeg.metadata-calibration-efficiency-v3.complete-grid-gate-report.v1",
                "reports": result["complete_grid_gate_report"],
            }
        ),
        "selection_rank_tuple": dict(report["selection_rank_tuple"]),
        "clean_commit": clean_commit,
        "clean_tree": clean_tree,
    }
    payload["payload_sha256"] = _payload_sha256(payload)
    validate_selected_method_freeze_payload(payload, development_result=result)
    return payload


_SELECTED_FREEZE_KEYS = {
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


def validate_selected_method_freeze_payload(
    payload: Mapping[str, Any],
    *,
    development_result: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    _require_exact_keys(payload, _SELECTED_FREEZE_KEYS, "selected-method freeze payload")
    if (
        payload["schema"] != SELECTED_METHOD_FREEZE_SCHEMA
        or payload["candidate_id"] != v3.CANDIDATE_ID
        or payload["master_plan_file_sha256"] != EXPECTED_MASTER_PLAN_SHA256
        or payload["synthetic_plan_file_sha256"] != EXPECTED_SYNTHETIC_PLAN_SHA256
        or payload["development_bundle_schema"]
        != "cfeg.metadata-calibration-efficiency-v3.development-bundle.v3"
        or payload["development_result_schema"] != DEVELOPMENT_RESULT_SCHEMA
    ):
        raise ValueError("selected-method freeze identity is invalid.")
    for name in (
        "selected_operator_instance_sha256",
        "master_plan_file_sha256",
        "synthetic_plan_file_sha256",
        "development_bundle_payload_sha256",
        "development_bundle_file_sha256",
        "development_result_payload_sha256",
        "development_result_file_sha256",
        "complete_grid_gate_report_sha256",
        "payload_sha256",
    ):
        _sha256(payload[name], name)
    if _payload_sha256(payload) != payload["payload_sha256"]:
        raise ValueError("selected-method freeze payload hash is invalid.")
    _git_object_id(payload["clean_commit"], "clean_commit")
    _git_object_id(payload["clean_tree"], "clean_tree")
    cell = v3.grid_cell_by_id(payload["selected_grid_cell_id"])
    expected_scientific_id = (
        f"{v3.CANDIDATE_ID}-p3-nu_{cell.nu_token}-lambda_{cell.lambda_token}"
    )
    if (
        payload["selected_grid_cell_index"] != cell.index
        or payload["selected_prototype_prior_pseudocount"]
        != cell.prototype_prior_pseudocount
        or payload["selected_lambda_max"] != cell.lambda_max
        or payload["selected_operator_instance_sha256"] != cell.operator_instance_sha256
        or payload["scientific_candidate_id"] != expected_scientific_id
    ):
        raise ValueError("selected-method freeze cell/scientific ID binding is invalid.")
    rank = payload["selection_rank_tuple"]
    _require_exact_keys(
        rank,
        {"minimum_mandatory_observed_gain", "minimum_corresponding_one_sided_LCB"},
        "selection_rank_tuple",
    )
    for value in rank.values():
        _finite_float(value, "selection rank component")
    if development_result is not None:
        result = validate_development_result_payload(
            development_result,
            require_enriched=True,
        )
        if (
            result["selection_status"] != "SELECTED_METHOD_PROPOSED"
            or result["selected_grid_cell_id"] != cell.grid_cell_id
            or result["payload_sha256"] != payload["development_result_payload_sha256"]
            or result["development_bundle_payload_sha256"]
            != payload["development_bundle_payload_sha256"]
            or result["development_bundle_file_sha256"]
            != payload["development_bundle_file_sha256"]
        ):
            raise ValueError("selected-method freeze differs from its development result.")
        report = next(
            item
            for item in result["complete_grid_gate_report"]
            if item["grid_cell_id"] == cell.grid_cell_id
        )
        expected_report_sha = _canonical_sha256(
            {
                "schema": "cfeg.metadata-calibration-efficiency-v3.complete-grid-gate-report.v1",
                "reports": result["complete_grid_gate_report"],
            }
        )
        if (
            rank != report["selection_rank_tuple"]
            or payload["complete_grid_gate_report_sha256"] != expected_report_sha
        ):
            raise ValueError("selected-method freeze rank/gate-report binding is invalid.")
    return dict(payload)


def _require_bundle_selected_delta_recovery(value: object) -> object:
    """Lazily validate governance's exact B=A+selected-only audit receipt."""

    try:
        from cfeg.metadata_calibration_v3_governance import (
            require_bundle_selected_delta_recovery_capability,
        )
    except ImportError as error:  # pragma: no cover - integration install failure
        raise RuntimeError("V3 governance recovery validator is unavailable.") from error
    return require_bundle_selected_delta_recovery_capability(value)


def _require_loaded_sources_for_recovery(recovery_capability: object) -> object:
    """Bind audit execution to the exact loaded and current bundle-A sources."""

    recovery = _require_bundle_selected_delta_recovery(recovery_capability)
    try:
        from cfeg.metadata_calibration_v3_governance import (
            recovery_bundle_source_file_sha256,
        )
    except ImportError as error:  # pragma: no cover - integration install failure
        raise RuntimeError("V3 recovery source validator is unavailable.") from error

    inventory = {
        path: recovery_bundle_source_file_sha256(
            recovery,
            repository_relative_path=path,
        )
        for path in (
            v3.MODEL_MODULE_REPOSITORY_PATH,
            SYNTHETIC_MODULE_REPOSITORY_PATH,
            FBCCA_MODULE_REPOSITORY_PATH,
            v3.MASTER_PLAN_REPOSITORY_PATH,
            v3.SYNTHETIC_PLAN_REPOSITORY_PATH,
            FILTERBANK_REPOSITORY_PATH,
        )
    }
    _require_loaded_bundle_a_source_inventory(inventory)
    return recovery


def _require_loaded_bundle_a_source_inventory(
    inventory: Mapping[str, str],
) -> None:
    """Compare loaded callables and current dependency bytes to bundle A."""

    model_path = Path(v3.__file__).resolve()
    synthetic_path = Path(__file__).resolve()
    if (
        model_path != (_REPOSITORY / v3.MODEL_MODULE_REPOSITORY_PATH).resolve()
        or model_path != v3._MODEL_MODULE_IMPORT_PATH
        or hashlib.sha256(model_path.read_bytes()).hexdigest()
        != v3._IMPORTED_MODEL_MODULE_SHA256
        or inventory[v3.MODEL_MODULE_REPOSITORY_PATH]
        != v3._IMPORTED_MODEL_MODULE_SHA256
        or synthetic_path
        != (_REPOSITORY / SYNTHETIC_MODULE_REPOSITORY_PATH).resolve()
        or synthetic_path != _SYNTHETIC_MODULE_IMPORT_PATH
        or hashlib.sha256(synthetic_path.read_bytes()).hexdigest()
        != _IMPORTED_SYNTHETIC_MODULE_SHA256
        or inventory[SYNTHETIC_MODULE_REPOSITORY_PATH]
        != _IMPORTED_SYNTHETIC_MODULE_SHA256
    ):
        raise RuntimeError(
            "loaded V3 audit modules differ from bundle-A source bytes."
        )
    for path, expected_sha256 in (
        (v3.MASTER_PLAN_REPOSITORY_PATH, EXPECTED_MASTER_PLAN_SHA256),
        (v3.SYNTHETIC_PLAN_REPOSITORY_PATH, EXPECTED_SYNTHETIC_PLAN_SHA256),
        (FILTERBANK_REPOSITORY_PATH, EXPECTED_FILTERBANK_SHA256),
    ):
        current_path = (_REPOSITORY / path).resolve()
        if (
            not current_path.is_file()
            or current_path.is_symlink()
            or hashlib.sha256(current_path.read_bytes()).hexdigest()
            != expected_sha256
            or inventory[path] != expected_sha256
        ):
            raise RuntimeError(f"audit dependency {path!r} differs from bundle A.")
    _require_fbcca_runtime_identity(inventory)


def _require_canonical_development_result_recovery(value: object) -> object:
    """Lazily validate governance's exact clean-A canonical-result receipt."""

    try:
        from cfeg.metadata_calibration_v3_governance import (
            require_canonical_development_result_recovery_capability,
        )
    except ImportError as error:  # pragma: no cover - integration install failure
        raise RuntimeError(
            "V3 canonical development-result recovery validator is unavailable."
        ) from error
    return require_canonical_development_result_recovery_capability(value)


def _require_loaded_sources_for_a_recovery(recovery_capability: object) -> object:
    """Bind clean-A recovery to current runtime and loaded bundle-A sources."""

    recovery = _require_canonical_development_result_recovery(recovery_capability)
    try:
        from cfeg.metadata_calibration_v3_governance import (
            development_result_recovery_source_file_sha256,
        )
    except ImportError as error:  # pragma: no cover - integration install failure
        raise RuntimeError("V3 clean-A recovery source validator is unavailable.") from error
    inventory = {
        path: development_result_recovery_source_file_sha256(
            recovery,
            repository_relative_path=path,
        )
        for path in (
            v3.MODEL_MODULE_REPOSITORY_PATH,
            SYNTHETIC_MODULE_REPOSITORY_PATH,
            FBCCA_MODULE_REPOSITORY_PATH,
            v3.MASTER_PLAN_REPOSITORY_PATH,
            v3.SYNTHETIC_PLAN_REPOSITORY_PATH,
            FILTERBANK_REPOSITORY_PATH,
        )
    }
    _require_loaded_bundle_a_source_inventory(inventory)
    return recovery


def _crosscheck_recovery_selected_and_result(
    development_result: Mapping[str, Any],
    *,
    development_result_file_sha256: str,
    selected_method_freeze: Mapping[str, Any],
    recovery_capability: object,
) -> tuple[dict[str, Any], dict[str, Any], object]:
    """Pure B-audit cross-check; deliberately issues no scientific authority."""

    recovery = _require_loaded_sources_for_recovery(recovery_capability)
    result_file_sha256 = _sha256(
        development_result_file_sha256,
        "development_result_file_sha256",
    )
    result = validate_development_result_payload(
        development_result,
        require_enriched=True,
    )
    selected = validate_selected_method_freeze_payload(
        selected_method_freeze,
        development_result=result,
    )
    if (
        recovery.candidate_id != v3.CANDIDATE_ID
        or recovery.development_bundle_payload_sha256
        != result["development_bundle_payload_sha256"]
        or recovery.development_bundle_file_sha256
        != result["development_bundle_file_sha256"]
        or selected["development_bundle_payload_sha256"]
        != recovery.development_bundle_payload_sha256
        or selected["development_bundle_file_sha256"]
        != recovery.development_bundle_file_sha256
        or selected["development_result_payload_sha256"]
        != result["payload_sha256"]
        or selected["development_result_file_sha256"] != result_file_sha256
        or selected["payload_sha256"]
        != recovery.selected_method_freeze_payload_sha256
        or selected["clean_commit"] != recovery.bundle_commit
        or selected["clean_tree"] != recovery.bundle_tree
    ):
        raise ValueError(
            "B-audit recovery, development result, and selected freeze differ."
        )
    _validate_frozen_sensitivity_replay_for_audit(result)
    _require_loaded_sources_for_recovery(recovery)
    return result, selected, recovery


def _frozen_sensitivity_audit_rng_factory(
    family: str,
    participant_index: int,
    block: int,
    class_index: int,
    component: str,
) -> np.random.Generator:
    """Private fixed-key replay used only after a governance recovery check."""

    family_codes = dict(v3.RNG_FAMILY_CODES)
    component_codes = dict(v3.RNG_COMPONENT_CODES)
    entropy = [
        v3.DEVELOPMENT_ROOT_SEED,
        family_codes[family],
        participant_index,
        block,
        class_index,
        component_codes[component],
    ]
    return np.random.Generator(
        np.random.PCG64DXSM(np.random.SeedSequence(entropy))
    )


def _validate_frozen_sensitivity_replay_for_audit(
    development_result: Mapping[str, Any],
) -> None:
    """Replay only the frozen component-11 report; expose no RNG or draws."""

    reports = development_result["complete_grid_gate_report"]
    for report in reports:
        primary, _ = _validated_sensitivity_participant_vectors(
            report["sensitivity_participant_vectors"]
        )
        expected = _sensitivity_resampling_with_rng_factory(
            primary,
            grid_cell_index=int(report["grid_cell_index"]),
            rng_factory=_frozen_sensitivity_audit_rng_factory,
            sign_flip_draws=SENSITIVITY_SIGN_FLIP_DRAWS,
            participant_bootstrap_draws=SENSITIVITY_PARTICIPANT_BOOTSTRAP_DRAWS,
            require_complete=True,
            _issuer=_SYNTHETIC_RNG_PRODUCER_ISSUER,
        )
        if report["resampling_sensitivity"] != expected:
            raise ValueError(
                "stored resampling sensitivity differs from exact component-11 replay."
            )


def _audited_development_result_binding(
    value: AuditedDevelopmentResult,
) -> dict[str, Any]:
    return {
        "schema": value.schema,
        "candidate_id": value.candidate_id,
        "payload_schema": value.payload_schema,
        "payload_sha256": value.payload_sha256,
        "development_result_file_sha256": value.development_result_file_sha256,
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
        "context_reference_schema": value.context_reference_schema,
        "context_reference_payload_sha256": value.context_reference_payload_sha256,
        "context_reference_file_sha256": value.context_reference_file_sha256,
        "development_start_schema": value.development_start_schema,
        "development_start_payload_sha256": value.development_start_payload_sha256,
        "development_start_file_sha256": value.development_start_file_sha256,
        "clean_commit": value.clean_commit,
        "clean_tree": value.clean_tree,
        "participant_metric_rows_sha256": value.participant_metric_rows_sha256,
        "invariant_rows_sha256": value.invariant_rows_sha256,
        "complete_grid_gate_report_sha256": (
            value.complete_grid_gate_report_sha256
        ),
        "uniform_block_weight_sensitivity_sha256": (
            value.uniform_block_weight_sensitivity_sha256
        ),
        "resampling_sensitivity_sha256": value.resampling_sensitivity_sha256,
        "sensitivity_participant_vectors_sha256": (
            value.sensitivity_participant_vectors_sha256
        ),
        "selection_status": value.selection_status,
        "selected_grid_cell_id": value.selected_grid_cell_id,
    }


def _issue_audited_development_result(
    result: Mapping[str, Any],
    recovery: object,
    *,
    _issuer: object,
) -> AuditedDevelopmentResult:
    if _issuer is not _AUDITED_DEVELOPMENT_RESULT_ISSUER:
        raise TypeError("audited-development-result issuer token is invalid.")
    reports = result["complete_grid_gate_report"]
    value = object.__new__(AuditedDevelopmentResult)
    fields: dict[str, Any] = {
        "schema": AUDITED_DEVELOPMENT_RESULT_SCHEMA,
        "candidate_id": v3.CANDIDATE_ID,
        "payload_schema": DEVELOPMENT_RESULT_SCHEMA,
        "payload_sha256": result["payload_sha256"],
        "development_result_file_sha256": (
            recovery.development_result_file_sha256
        ),
        "development_bundle_payload_sha256": (
            recovery.development_bundle_payload_sha256
        ),
        "development_bundle_file_sha256": (
            recovery.development_bundle_file_sha256
        ),
        "development_bundle_source_bundle_sha256": (
            recovery.development_bundle_source_bundle_sha256
        ),
        "numerical_runtime_fingerprint_sha256": (
            recovery.numerical_runtime_fingerprint_sha256
        ),
        "context_reference_schema": recovery.context_reference_schema,
        "context_reference_payload_sha256": (
            recovery.context_reference_payload_sha256
        ),
        "context_reference_file_sha256": recovery.context_reference_file_sha256,
        "development_start_schema": recovery.development_start_schema,
        "development_start_payload_sha256": recovery.development_start_payload_sha256,
        "development_start_file_sha256": recovery.development_start_file_sha256,
        "clean_commit": recovery.clean_commit,
        "clean_tree": recovery.clean_tree,
        "participant_metric_rows_sha256": _canonical_sha256(
            {
                "schema": (
                    "cfeg.metadata-calibration-efficiency-v3."
                    "participant-metric-row-set.v1"
                ),
                "rows": result["participant_metric_rows"],
            }
        ),
        "invariant_rows_sha256": _canonical_sha256(
            {
                "schema": (
                    "cfeg.metadata-calibration-efficiency-v3.invariant-row-set.v1"
                ),
                "rows": result["invariant_rows"],
            }
        ),
        "complete_grid_gate_report_sha256": _canonical_sha256(
            {
                "schema": (
                    "cfeg.metadata-calibration-efficiency-v3."
                    "complete-grid-gate-report.v1"
                ),
                "reports": reports,
            }
        ),
        "uniform_block_weight_sensitivity_sha256": _canonical_sha256(
            {
                "schema": (
                    "cfeg.metadata-calibration-efficiency-v3."
                    "complete-uniform-block-weight-sensitivity.v1"
                ),
                "reports": [
                    report["uniform_block_weight_sensitivity"]
                    for report in reports
                ],
            }
        ),
        "resampling_sensitivity_sha256": _canonical_sha256(
            {
                "schema": (
                    "cfeg.metadata-calibration-efficiency-v3."
                    "complete-resampling-sensitivity.v1"
                ),
                "reports": [report["resampling_sensitivity"] for report in reports],
            }
        ),
        "sensitivity_participant_vectors_sha256": (
            _complete_sensitivity_participant_vectors_sha256(reports)
        ),
        "selection_status": result["selection_status"],
        "selected_grid_cell_id": result["selected_grid_cell_id"],
        "_recovery_capability": recovery,
    }
    for name, item in fields.items():
        object.__setattr__(value, name, item)
    object.__setattr__(
        value,
        "semantic_binding_sha256",
        _canonical_sha256(_audited_development_result_binding(value)),
    )
    object.__setattr__(value, "_issuer", _AUDITED_DEVELOPMENT_RESULT_ISSUER)
    return _require_audited_development_result(
        value,
        expected_payload_sha256=result["payload_sha256"],
    )


def require_audited_development_result(
    value: object,
    *,
    expected_payload_sha256: str | None = None,
) -> AuditedDevelopmentResult:
    """Require an exact audit-only proof and recheck its clean-A receipt."""

    return _require_audited_development_result(
        value,
        expected_payload_sha256=expected_payload_sha256,
    )


def _require_audited_development_result(
    value: object,
    *,
    expected_payload_sha256: str | None,
) -> AuditedDevelopmentResult:
    if type(value) is not AuditedDevelopmentResult:
        raise TypeError("value must be an exact AuditedDevelopmentResult.")
    try:
        issuer = value._issuer
    except AttributeError as error:
        raise TypeError("AuditedDevelopmentResult was not issued by this module.") from error
    if issuer is not _AUDITED_DEVELOPMENT_RESULT_ISSUER:
        raise TypeError("AuditedDevelopmentResult issuer marker is invalid.")
    if (
        value.schema != AUDITED_DEVELOPMENT_RESULT_SCHEMA
        or value.candidate_id != v3.CANDIDATE_ID
        or value.payload_schema != DEVELOPMENT_RESULT_SCHEMA
        or value.context_reference_schema != v3.CONTEXT_REFERENCE_SCHEMA
        or value.development_start_schema
        != "cfeg.metadata-calibration-efficiency-v3.development-start.v1"
        or value.selection_status
        not in {"SELECTED_METHOD_PROPOSED", "DEVELOPMENT_NO_GO"}
        or (
            value.selection_status == "DEVELOPMENT_NO_GO"
            and value.selected_grid_cell_id is not None
        )
        or (
            value.selection_status == "SELECTED_METHOD_PROPOSED"
            and value.selected_grid_cell_id
            not in {cell.grid_cell_id for cell in v3.canonical_operator_grid()}
        )
    ):
        raise ValueError("AuditedDevelopmentResult identity/selection is invalid.")
    for name in (
        "payload_sha256",
        "development_result_file_sha256",
        "development_bundle_payload_sha256",
        "development_bundle_file_sha256",
        "development_bundle_source_bundle_sha256",
        "numerical_runtime_fingerprint_sha256",
        "context_reference_payload_sha256",
        "context_reference_file_sha256",
        "development_start_payload_sha256",
        "development_start_file_sha256",
        "participant_metric_rows_sha256",
        "invariant_rows_sha256",
        "complete_grid_gate_report_sha256",
        "uniform_block_weight_sensitivity_sha256",
        "resampling_sensitivity_sha256",
        "sensitivity_participant_vectors_sha256",
        "semantic_binding_sha256",
    ):
        _sha256(getattr(value, name), name)
    _git_object_id(value.clean_commit, "clean_commit")
    _git_object_id(value.clean_tree, "clean_tree")
    if value.semantic_binding_sha256 != _canonical_sha256(
        _audited_development_result_binding(value)
    ):
        raise ValueError("AuditedDevelopmentResult semantic binding is invalid.")
    recovery = _require_loaded_sources_for_a_recovery(value._recovery_capability)
    if (
        value.development_result_file_sha256
        != recovery.development_result_file_sha256
        or value.development_bundle_payload_sha256
        != recovery.development_bundle_payload_sha256
        or value.development_bundle_file_sha256
        != recovery.development_bundle_file_sha256
        or value.development_bundle_source_bundle_sha256
        != recovery.development_bundle_source_bundle_sha256
        or value.numerical_runtime_fingerprint_sha256
        != recovery.numerical_runtime_fingerprint_sha256
        or value.context_reference_schema != recovery.context_reference_schema
        or value.context_reference_payload_sha256
        != recovery.context_reference_payload_sha256
        or value.context_reference_file_sha256
        != recovery.context_reference_file_sha256
        or value.development_start_schema != recovery.development_start_schema
        or value.development_start_payload_sha256 != recovery.development_start_payload_sha256
        or value.development_start_file_sha256 != recovery.development_start_file_sha256
        or value.payload_sha256 != recovery.development_result_payload_sha256
        or value.clean_commit != recovery.clean_commit
        or value.clean_tree != recovery.clean_tree
    ):
        raise ValueError("AuditedDevelopmentResult differs from its recovery receipt.")
    if expected_payload_sha256 is not None:
        _sha256(expected_payload_sha256, "expected_payload_sha256")
        if value.payload_sha256 != expected_payload_sha256:
            raise ValueError("AuditedDevelopmentResult binds a different payload.")
    return value


def _require_result_evidence_matches_audited_proof(
    result: Mapping[str, Any],
    proof: AuditedDevelopmentResult,
) -> None:
    reports = result["complete_grid_gate_report"]
    if (
        proof.development_bundle_payload_sha256
        != result["development_bundle_payload_sha256"]
        or proof.development_bundle_file_sha256
        != result["development_bundle_file_sha256"]
        or proof.context_reference_schema != result["context_reference_schema"]
        or proof.context_reference_payload_sha256
        != result["context_reference_payload_sha256"]
        or proof.context_reference_file_sha256
        != result["context_reference_file_sha256"]
        or proof.development_start_schema != result["development_start_schema"]
        or proof.development_start_payload_sha256 != result["development_start_payload_sha256"]
        or proof.development_start_file_sha256 != result["development_start_file_sha256"]
        or proof.participant_metric_rows_sha256
        != _canonical_sha256(
            {
                "schema": (
                    "cfeg.metadata-calibration-efficiency-v3."
                    "participant-metric-row-set.v1"
                ),
                "rows": result["participant_metric_rows"],
            }
        )
        or proof.invariant_rows_sha256
        != _canonical_sha256(
            {
                "schema": (
                    "cfeg.metadata-calibration-efficiency-v3.invariant-row-set.v1"
                ),
                "rows": result["invariant_rows"],
            }
        )
        or proof.complete_grid_gate_report_sha256
        != _canonical_sha256(
            {
                "schema": (
                    "cfeg.metadata-calibration-efficiency-v3."
                    "complete-grid-gate-report.v1"
                ),
                "reports": reports,
            }
        )
        or proof.uniform_block_weight_sensitivity_sha256
        != _canonical_sha256(
            {
                "schema": (
                    "cfeg.metadata-calibration-efficiency-v3."
                    "complete-uniform-block-weight-sensitivity.v1"
                ),
                "reports": [
                    report["uniform_block_weight_sensitivity"]
                    for report in reports
                ],
            }
        )
        or proof.resampling_sensitivity_sha256
        != _canonical_sha256(
            {
                "schema": (
                    "cfeg.metadata-calibration-efficiency-v3."
                    "complete-resampling-sensitivity.v1"
                ),
                "reports": [report["resampling_sensitivity"] for report in reports],
            }
        )
        or proof.sensitivity_participant_vectors_sha256
        != _complete_sensitivity_participant_vectors_sha256(reports)
        or proof.selection_status != result["selection_status"]
        or proof.selected_grid_cell_id != result["selected_grid_cell_id"]
    ):
        raise ValueError("audited proof differs from development-result evidence.")


def _validate_exact_synthetic_context_reference(
    payload: Mapping[str, Any],
) -> v3.ContextReference:
    reference = v3.validate_context_reference_payload(payload)
    if (
        reference.domain != "synthetic_development_and_future_synthetic"
        or tuple(
            table.interface_lookup_key for table in reference.interface_tables
        )
        != CONTEXT_REFERENCE_INTERFACES
        or reference.canonical_channels != v3.CANONICAL_CHANNELS
    ):
        raise ValueError("synthetic context-reference identity/order is invalid.")
    return reference


def validate_development_result_for_a_recovery(
    payload: Mapping[str, Any],
    *,
    context_reference: Mapping[str, Any],
    recovery_capability: object,
) -> AuditedDevelopmentResult:
    """Audit a proof-gated canonical result at clean A without minting authority."""

    recovery = _require_loaded_sources_for_a_recovery(recovery_capability)
    result = validate_development_result_payload(payload, require_enriched=True)
    reference = _validate_exact_synthetic_context_reference(context_reference)
    if (
        recovery.candidate_id != v3.CANDIDATE_ID
        or recovery.development_bundle_payload_sha256
        != result["development_bundle_payload_sha256"]
        or recovery.development_bundle_file_sha256
        != result["development_bundle_file_sha256"]
        or recovery.context_reference_schema != v3.CONTEXT_REFERENCE_SCHEMA
        or recovery.context_reference_payload_sha256 != reference.payload_sha256
        or recovery.context_reference_payload_sha256
        != result["context_reference_payload_sha256"]
        or recovery.context_reference_file_sha256
        != result["context_reference_file_sha256"]
        or recovery.development_start_schema != result["development_start_schema"]
        or recovery.development_start_payload_sha256
        != result["development_start_payload_sha256"]
        or recovery.development_start_file_sha256
        != result["development_start_file_sha256"]
        or recovery.development_result_schema != DEVELOPMENT_RESULT_SCHEMA
        or recovery.development_result_payload_sha256 != result["payload_sha256"]
    ):
        raise ValueError(
            "clean-A recovery differs from canonical context/development bytes."
        )
    _validate_frozen_sensitivity_replay_for_audit(result)
    return _issue_audited_development_result(
        result,
        recovery,
        _issuer=_AUDITED_DEVELOPMENT_RESULT_ISSUER,
    )


def build_selected_method_freeze_payload_for_a_recovery(
    development_result: Mapping[str, Any],
    *,
    audited_development_result: AuditedDevelopmentResult,
) -> dict[str, Any]:
    """Recover the selected payload from proof-gated bytes at exact clean A."""

    proof = require_audited_development_result(
        audited_development_result,
        expected_payload_sha256=None,
    )
    result = validate_development_result_payload(
        development_result,
        require_enriched=True,
    )
    if proof.payload_sha256 != result["payload_sha256"]:
        raise ValueError("AuditedDevelopmentResult binds a different payload.")
    _require_result_evidence_matches_audited_proof(result, proof)
    return _assemble_selected_method_freeze_payload(
        result,
        development_result_file_sha256=proof.development_result_file_sha256,
        clean_commit=proof.clean_commit,
        clean_tree=proof.clean_tree,
        _issuer=_SELECTED_FREEZE_ASSEMBLY_ISSUER,
    )


def validate_development_result_for_audit(
    payload: Mapping[str, Any],
    *,
    development_result_file_sha256: str,
    selected_method_freeze: Mapping[str, Any],
    recovery_capability: object,
) -> dict[str, Any]:
    """Validate canonical development bytes at B without recreating RNG authority."""

    result, _, _ = _crosscheck_recovery_selected_and_result(
        payload,
        development_result_file_sha256=development_result_file_sha256,
        selected_method_freeze=selected_method_freeze,
        recovery_capability=recovery_capability,
    )
    return result


def validate_context_reference_for_audit(
    payload: Mapping[str, Any],
    *,
    context_reference_file_sha256: str,
    development_result: Mapping[str, Any],
    development_result_file_sha256: str,
    selected_method_freeze: Mapping[str, Any],
    recovery_capability: object,
) -> v3.ContextReference:
    """Validate the canonical reference through its durable dev/selected chain."""

    result, _, _ = _crosscheck_recovery_selected_and_result(
        development_result,
        development_result_file_sha256=development_result_file_sha256,
        selected_method_freeze=selected_method_freeze,
        recovery_capability=recovery_capability,
    )
    reference_file_sha256 = _sha256(
        context_reference_file_sha256,
        "context_reference_file_sha256",
    )
    reference = _validate_exact_synthetic_context_reference(payload)
    if (
        reference.payload_sha256 != result["context_reference_payload_sha256"]
        or reference_file_sha256 != result["context_reference_file_sha256"]
    ):
        raise ValueError(
            "B-audit context reference differs from the durable development chain."
        )
    return reference


def validate_selected_method_freeze_for_audit(
    payload: Mapping[str, Any],
    *,
    development_result: Mapping[str, Any],
    development_result_file_sha256: str,
    recovery_capability: object,
) -> dict[str, Any]:
    """Recompute selection at B without issuing/requiring an A-only proposal."""

    _, selected, _ = _crosscheck_recovery_selected_and_result(
        development_result,
        development_result_file_sha256=development_result_file_sha256,
        selected_method_freeze=payload,
        recovery_capability=recovery_capability,
    )
    return selected


def _selected_method_proposal_binding(
    value: SelectedMethodProposal,
) -> dict[str, Any]:
    return {
        "schema": value.schema,
        "candidate_id": value.candidate_id,
        "payload_schema": value.payload_schema,
        "payload_sha256": value.payload_sha256,
        "scientific_candidate_id": value.scientific_candidate_id,
        "selected_grid_cell_index": value.selected_grid_cell_index,
        "selected_grid_cell_id": value.selected_grid_cell_id,
        "selected_prototype_prior_pseudocount": (
            value.selected_prototype_prior_pseudocount
        ),
        "selected_lambda_max": value.selected_lambda_max,
        "selected_operator_instance_sha256": value.selected_operator_instance_sha256,
        "master_plan_file_sha256": value.master_plan_file_sha256,
        "synthetic_plan_file_sha256": value.synthetic_plan_file_sha256,
        "development_bundle_schema": value.development_bundle_schema,
        "development_bundle_payload_sha256": (
            value.development_bundle_payload_sha256
        ),
        "development_bundle_file_sha256": value.development_bundle_file_sha256,
        "development_result_schema": value.development_result_schema,
        "development_result_payload_sha256": value.development_result_payload_sha256,
        "development_result_file_sha256": value.development_result_file_sha256,
        "validated_development_result_semantic_binding_sha256": (
            value.validated_development_result_semantic_binding_sha256
        ),
        "complete_grid_gate_report_sha256": value.complete_grid_gate_report_sha256,
        "minimum_mandatory_observed_gain": value.minimum_mandatory_observed_gain,
        "minimum_corresponding_one_sided_LCB": (
            value.minimum_corresponding_one_sided_LCB
        ),
        "clean_commit": value.clean_commit,
        "clean_tree": value.clean_tree,
    }


def _issue_selected_method_proposal(
    selected: Mapping[str, Any],
    *,
    validated_development_result: ValidatedDevelopmentResult,
    _issuer: object,
) -> SelectedMethodProposal:
    if _issuer is not _SELECTED_METHOD_PROPOSAL_ISSUER:
        raise TypeError("selected-method proposal issuer token is invalid.")
    result_proof = require_validated_development_result(
        validated_development_result,
        expected_payload_sha256=selected["development_result_payload_sha256"],
    )
    if (
        result_proof.development_bundle_payload_sha256
        != selected["development_bundle_payload_sha256"]
        or result_proof.development_bundle_file_sha256
        != selected["development_bundle_file_sha256"]
        or result_proof.complete_grid_gate_report_sha256
        != selected["complete_grid_gate_report_sha256"]
        or result_proof.selection_status != "SELECTED_METHOD_PROPOSED"
        or result_proof.selected_grid_cell_id != selected["selected_grid_cell_id"]
    ):
        raise ValueError("selected proposal differs from authorized development proof.")
    value = object.__new__(SelectedMethodProposal)
    rank = selected["selection_rank_tuple"]
    fields: dict[str, Any] = {
        "schema": SELECTED_METHOD_PROPOSAL_SCHEMA,
        "candidate_id": v3.CANDIDATE_ID,
        "payload_schema": selected["schema"],
        "payload_sha256": selected["payload_sha256"],
        "scientific_candidate_id": selected["scientific_candidate_id"],
        "selected_grid_cell_index": selected["selected_grid_cell_index"],
        "selected_grid_cell_id": selected["selected_grid_cell_id"],
        "selected_prototype_prior_pseudocount": selected[
            "selected_prototype_prior_pseudocount"
        ],
        "selected_lambda_max": selected["selected_lambda_max"],
        "selected_operator_instance_sha256": selected[
            "selected_operator_instance_sha256"
        ],
        "master_plan_file_sha256": selected["master_plan_file_sha256"],
        "synthetic_plan_file_sha256": selected["synthetic_plan_file_sha256"],
        "development_bundle_schema": selected["development_bundle_schema"],
        "development_bundle_payload_sha256": selected[
            "development_bundle_payload_sha256"
        ],
        "development_bundle_file_sha256": selected[
            "development_bundle_file_sha256"
        ],
        "development_result_schema": selected["development_result_schema"],
        "development_result_payload_sha256": selected[
            "development_result_payload_sha256"
        ],
        "development_result_file_sha256": selected[
            "development_result_file_sha256"
        ],
        "validated_development_result_semantic_binding_sha256": (
            result_proof.semantic_binding_sha256
        ),
        "complete_grid_gate_report_sha256": selected[
            "complete_grid_gate_report_sha256"
        ],
        "minimum_mandatory_observed_gain": rank[
            "minimum_mandatory_observed_gain"
        ],
        "minimum_corresponding_one_sided_LCB": rank[
            "minimum_corresponding_one_sided_LCB"
        ],
        "clean_commit": selected["clean_commit"],
        "clean_tree": selected["clean_tree"],
        "_validated_result": result_proof,
    }
    for name, item in fields.items():
        object.__setattr__(value, name, item)
    object.__setattr__(
        value,
        "semantic_binding_sha256",
        _canonical_sha256(_selected_method_proposal_binding(value)),
    )
    object.__setattr__(value, "_issuer", _SELECTED_METHOD_PROPOSAL_ISSUER)
    return _require_selected_method_proposal(
        value,
        expected_payload_sha256=selected["payload_sha256"],
        revalidate_development=False,
    )


def validate_selected_method_proposal(
    payload: Mapping[str, Any],
    *,
    development_result: Mapping[str, Any],
    validated_development_result: ValidatedDevelopmentResult,
) -> SelectedMethodProposal:
    """Issue a proposal proof only after exact selected/result cross-validation."""

    selected = validate_selected_method_freeze_payload(
        payload,
        development_result=development_result,
    )
    return _issue_selected_method_proposal(
        selected,
        validated_development_result=validated_development_result,
        _issuer=_SELECTED_METHOD_PROPOSAL_ISSUER,
    )


def require_selected_method_proposal(
    value: object,
    *,
    expected_payload_sha256: str | None = None,
) -> SelectedMethodProposal:
    """Require an authentic and internally bound exact selection proposal."""

    return _require_selected_method_proposal(
        value,
        expected_payload_sha256=expected_payload_sha256,
        revalidate_development=True,
    )


def _require_selected_method_proposal(
    value: object,
    *,
    expected_payload_sha256: str | None,
    revalidate_development: bool,
) -> SelectedMethodProposal:

    if type(value) is not SelectedMethodProposal:
        raise TypeError("value must be an exact SelectedMethodProposal.")
    try:
        issuer = value._issuer
    except AttributeError as error:
        raise TypeError("SelectedMethodProposal was not issued by this module.") from error
    if issuer is not _SELECTED_METHOD_PROPOSAL_ISSUER:
        raise TypeError("SelectedMethodProposal issuer marker is invalid.")
    if (
        value.schema != SELECTED_METHOD_PROPOSAL_SCHEMA
        or value.candidate_id != v3.CANDIDATE_ID
        or value.payload_schema != SELECTED_METHOD_FREEZE_SCHEMA
        or value.master_plan_file_sha256 != EXPECTED_MASTER_PLAN_SHA256
        or value.synthetic_plan_file_sha256 != EXPECTED_SYNTHETIC_PLAN_SHA256
        or value.development_bundle_schema
        != "cfeg.metadata-calibration-efficiency-v3.development-bundle.v3"
        or value.development_result_schema != DEVELOPMENT_RESULT_SCHEMA
    ):
        raise ValueError("SelectedMethodProposal identity/bindings are invalid.")
    for name in (
        "payload_sha256",
        "selected_operator_instance_sha256",
        "master_plan_file_sha256",
        "synthetic_plan_file_sha256",
        "development_bundle_payload_sha256",
        "development_bundle_file_sha256",
        "development_result_payload_sha256",
        "development_result_file_sha256",
        "validated_development_result_semantic_binding_sha256",
        "complete_grid_gate_report_sha256",
        "semantic_binding_sha256",
    ):
        _sha256(getattr(value, name), name)
    _git_object_id(value.clean_commit, "clean_commit")
    _git_object_id(value.clean_tree, "clean_tree")
    cell = v3.grid_cell_by_id(value.selected_grid_cell_id)
    expected_scientific_id = (
        f"{v3.CANDIDATE_ID}-p3-nu_{cell.nu_token}-lambda_{cell.lambda_token}"
    )
    if (
        value.selected_grid_cell_index != cell.index
        or value.selected_prototype_prior_pseudocount
        != cell.prototype_prior_pseudocount
        or value.selected_lambda_max != cell.lambda_max
        or value.selected_operator_instance_sha256 != cell.operator_instance_sha256
        or value.scientific_candidate_id != expected_scientific_id
    ):
        raise ValueError("SelectedMethodProposal grid-cell binding is invalid.")
    _finite_float(
        value.minimum_mandatory_observed_gain,
        "minimum_mandatory_observed_gain",
    )
    _finite_float(
        value.minimum_corresponding_one_sided_LCB,
        "minimum_corresponding_one_sided_LCB",
    )
    expected_binding = _canonical_sha256(_selected_method_proposal_binding(value))
    if value.semantic_binding_sha256 != expected_binding:
        raise ValueError("SelectedMethodProposal semantic binding is invalid.")
    result_proof = _require_validated_development_result(
        value._validated_result,
        expected_payload_sha256=value.development_result_payload_sha256,
        revalidate_bundle=revalidate_development,
    )
    if (
        value.validated_development_result_semantic_binding_sha256
        != result_proof.semantic_binding_sha256
        or value.development_bundle_payload_sha256
        != result_proof.development_bundle_payload_sha256
        or value.development_bundle_file_sha256
        != result_proof.development_bundle_file_sha256
        or value.complete_grid_gate_report_sha256
        != result_proof.complete_grid_gate_report_sha256
        or value.selected_grid_cell_id != result_proof.selected_grid_cell_id
    ):
        raise ValueError("SelectedMethodProposal development proof binding is invalid.")
    if expected_payload_sha256 is not None:
        _sha256(expected_payload_sha256, "expected_payload_sha256")
        if value.payload_sha256 != expected_payload_sha256:
            raise ValueError("SelectedMethodProposal binds a different payload.")
    return value


def _metric_identity(row: Mapping[str, Any]) -> tuple[Any, ...]:
    return (
        row["grid_cell_index"],
        row["family"],
        row["participant_index"],
        row["condition"],
        row["role"],
        row["control"],
        row["gate"],
        row["budget"],
    )


def _expected_metric_identities() -> Iterable[tuple[Any, ...]]:
    for cell in v3.canonical_operator_grid():
        for family in FAMILY_NAMES:
            specs = method_specs_for_family(family)
            for participant in range(DEVELOPMENT_PARTICIPANTS):
                for spec in specs:
                    yield (
                        cell.index,
                        family,
                        participant,
                        spec.condition,
                        spec.role,
                        spec.control,
                        spec.gate,
                        spec.budget,
                    )
                if family == "B4_interface_calibrated_impedance_shift":
                    for spec in specs[:34]:
                        yield (
                            cell.index,
                            family,
                            participant,
                            "equal_condition_composite",
                            spec.role,
                            spec.control,
                            spec.gate,
                            spec.budget,
                        )


_INVARIANT_CASE_ORDER = {
    "N5_invalid_context": (
        "unknown_interface_category",
        "negative_impedance",
        "nonfinite_impedance",
        "wrong_channel_count",
        "packet_key_or_pairing_digest_mismatch",
        "query_source_standardized_OOD",
    ),
    "N6_support_reliability_invariants": (
        "uniform_chance_probabilities",
        "all_observed_label_probabilities_below_chance",
        "one_corrupt_block_among_three",
        "confident_wrong_anchor_probabilities",
    ),
}


def _invariant_identity(row: Mapping[str, Any]) -> tuple[Any, ...]:
    return row["grid_cell_index"], row["family"], row["case_id"]


def _expected_invariant_identities() -> Iterable[tuple[Any, ...]]:
    for cell in v3.canonical_operator_grid():
        for family, cases in _INVARIANT_CASE_ORDER.items():
            for case_id in cases:
                yield cell.index, family, case_id


def _participant_row_sort_key(row: Mapping[str, Any]) -> tuple[int, ...]:
    family_index = FAMILY_NAMES.index(row["family"])
    specs = method_specs_for_family(row["family"])
    if row["condition"] == "equal_condition_composite":
        method_index = 68 + next(
            index
            for index, spec in enumerate(specs[:34])
            if (spec.role, spec.control, spec.gate, spec.budget)
            == (row["role"], row["control"], row["gate"], row["budget"])
        )
    else:
        method_index = next(
            index
            for index, spec in enumerate(specs)
            if (spec.condition, spec.role, spec.control, spec.gate, spec.budget)
            == (
                row["condition"],
                row["role"],
                row["control"],
                row["gate"],
                row["budget"],
            )
        )
    return (
        int(row["grid_cell_index"]),
        family_index,
        int(row["participant_index"]),
        method_index,
    )


def _invariant_row_sort_key(row: Mapping[str, Any]) -> tuple[int, ...]:
    family_index = tuple(_INVARIANT_CASE_ORDER).index(row["family"])
    case_index = _INVARIANT_CASE_ORDER[row["family"]].index(row["case_id"])
    return int(row["grid_cell_index"]), family_index, case_index


def _metric_lookup_key(row: Mapping[str, Any]) -> tuple[Any, ...]:
    return (
        row["family"],
        row["participant_index"],
        row["condition"],
        row["role"],
        row["control"],
        row["gate"],
        row["budget"],
    )


def _participant_eauc(
    index: Mapping[tuple[Any, ...], Mapping[str, Any]],
    family: str,
    participant: int,
    condition: str,
    role: str,
    control: str,
    gate: str,
    *,
    metric: str,
) -> float:
    values = {}
    for budget in (0, 1, 3):
        key = (family, participant, condition, role, control, gate, budget)
        if key not in index and budget == 0:
            key = (family, participant, condition, "A0", "none", "none", 0)
        if key not in index:
            raise ValueError(f"eAUC method cell is missing: {key!r}.")
        values[budget] = float(index[key][metric])
    return values[0] / 6.0 + values[1] / 2.0 + values[3] / 3.0


def _sensitivity_vectors_from_metric_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    grid_cell_index: int,
) -> dict[str, np.ndarray]:
    """Derive the six ordered 48-participant sensitivity vectors from rows."""

    cell_index = _positive_exact_int(grid_cell_index, "grid_cell_index")
    if cell_index not in range(1, 10):
        raise ValueError("grid_cell_index must be in the canonical range 1..9.")
    selected_rows = [
        row for row in rows if row["grid_cell_index"] == cell_index
    ]
    index = {_metric_lookup_key(row): row for row in selected_rows}
    if len(index) != len(selected_rows):
        raise ValueError("duplicate metric rows reached sensitivity reduction.")

    def vector(
        family: str,
        condition: str,
        left: tuple[str, str, str],
        right: tuple[str, str, str],
        *,
        budget: int | None = None,
        eauc: bool,
    ) -> np.ndarray:
        values: list[float] = []
        for participant_index in range(DEVELOPMENT_PARTICIPANTS):
            if eauc:
                left_value = _participant_eauc(
                    index,
                    family,
                    participant_index,
                    condition,
                    *left,
                    metric="balanced_accuracy",
                )
                right_value = _participant_eauc(
                    index,
                    family,
                    participant_index,
                    condition,
                    *right,
                    metric="balanced_accuracy",
                )
            else:
                if budget is None:
                    raise ValueError("point sensitivity contrast requires a budget.")
                left_value = float(
                    index[
                        (family, participant_index, condition, *left, budget)
                    ]["balanced_accuracy"]
                )
                right_value = float(
                    index[
                        (family, participant_index, condition, *right, budget)
                    ]["balanced_accuracy"]
                )
            values.append(left_value - right_value)
        result = np.asarray(values, dtype=np.float64)
        result.setflags(write=False)
        return result

    vectors = {
        "B1_A_Q_minus_A0_eAUC": vector(
            "B1_participant_class_confusion",
            "neutral",
            ("A_Q", "none", "deployed"),
            ("A0", "none", "none"),
            eauc=True,
        ),
        "B2_A_Q_minus_A0_eAUC": vector(
            "B2_participant_phase_spatial_shift",
            "neutral",
            ("A_Q", "none", "deployed"),
            ("A0", "none", "none"),
            eauc=True,
        ),
        "B3_A_QM_minus_A_Q_eAUC": vector(
            "B3_impedance_linked_transfer_shift",
            "neutral",
            ("A_QM", "correct", "deployed"),
            ("A_Q", "none", "deployed"),
            eauc=True,
        ),
        "B4_A_QM_minus_A_Q_eAUC": vector(
            "B4_interface_calibrated_impedance_shift",
            "equal_condition_composite",
            ("A_QM", "correct", "deployed"),
            ("A_Q", "none", "deployed"),
            eauc=True,
        ),
        "B3_k3_correct_minus_shuffle": vector(
            "B3_impedance_linked_transfer_shift",
            "neutral",
            ("A_QM", "correct", "deployed"),
            ("A_QM", "within_prefix_packet_shuffle", "deployed"),
            budget=3,
            eauc=False,
        ),
        "B4_k3_correct_minus_shuffle": vector(
            "B4_interface_calibrated_impedance_shift",
            "equal_condition_composite",
            ("A_QM", "correct", "deployed"),
            ("A_QM", "within_prefix_packet_shuffle", "deployed"),
            budget=3,
            eauc=False,
        ),
    }
    if tuple(vectors) != SENSITIVITY_ENDPOINT_IDS:
        raise RuntimeError("sensitivity vector order drifted.")
    return vectors


def _sensitivity_vector_values_sha256(
    kind: str,
    values_by_endpoint: Mapping[str, Sequence[float]],
) -> str:
    if kind not in {"primary", "uniform"}:
        raise ValueError("sensitivity vector kind must be primary or uniform.")
    return _canonical_sha256(
        {
            "schema": (
                "cfeg.metadata-calibration-efficiency-v3."
                f"{kind}-sensitivity-participant-vector-set.v1"
            ),
            "participant_count": DEVELOPMENT_PARTICIPANTS,
            "endpoint_order": list(SENSITIVITY_ENDPOINT_IDS),
            "values_by_endpoint": dict(values_by_endpoint),
        }
    )


def _complete_sensitivity_participant_vectors_sha256(
    reports: Sequence[Mapping[str, Any]],
) -> str:
    return _canonical_sha256(
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


def _validated_sensitivity_participant_vectors(
    value: Mapping[str, Any],
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    """Validate and materialize the exact report-only participant evidence."""

    if type(value) is not dict:
        raise TypeError("sensitivity participant vectors must be an exact dictionary.")
    _require_exact_keys(
        value,
        {
            "schema",
            "participant_count",
            "endpoint_order",
            "primary_values_by_endpoint",
            "uniform_values_by_endpoint",
            "primary_values_sha256",
            "uniform_values_sha256",
        },
        "sensitivity participant vectors",
    )
    if (
        value["schema"]
        != (
            "cfeg.metadata-calibration-efficiency-v3."
            "sensitivity-participant-vectors.v1"
        )
        or value["participant_count"] != DEVELOPMENT_PARTICIPANTS
        or type(value["endpoint_order"]) is not list
        or tuple(value["endpoint_order"]) != SENSITIVITY_ENDPOINT_IDS
    ):
        raise ValueError("sensitivity participant-vector identity/order is invalid.")

    resolved: dict[str, dict[str, np.ndarray]] = {}
    for kind in ("primary", "uniform"):
        field = f"{kind}_values_by_endpoint"
        raw_mapping = value[field]
        if type(raw_mapping) is not dict or tuple(raw_mapping) != SENSITIVITY_ENDPOINT_IDS:
            raise TypeError(
                f"{field} must be an exact endpoint-ordered dictionary."
            )
        raw_for_hash: dict[str, list[float]] = {}
        arrays: dict[str, np.ndarray] = {}
        for endpoint in SENSITIVITY_ENDPOINT_IDS:
            raw_vector = raw_mapping[endpoint]
            if type(raw_vector) is not list or len(raw_vector) != DEVELOPMENT_PARTICIPANTS:
                raise ValueError(
                    f"{field}[{endpoint!r}] must contain exactly 48 values."
                )
            if any(type(item) is not float or not math.isfinite(item) for item in raw_vector):
                raise TypeError(
                    f"{field}[{endpoint!r}] must contain exact finite floats."
                )
            raw_for_hash[endpoint] = list(raw_vector)
            array = np.asarray(raw_vector, dtype=np.float64)
            array.setflags(write=False)
            arrays[endpoint] = array
        expected_hash = _sensitivity_vector_values_sha256(kind, raw_for_hash)
        if value[f"{kind}_values_sha256"] != expected_hash:
            raise ValueError(f"{kind} sensitivity participant-vector hash is invalid.")
        resolved[kind] = arrays
    return resolved["primary"], resolved["uniform"]


def _one_sided_summary(values: np.ndarray) -> dict[str, Any]:
    array = _finite_vector(values, "paired contrast")
    return {
        "n": len(array),
        "observed_mean": float(np.mean(array)),
        "one_sided_95_LCB": _one_sided_t_lower_bound(array),
        "one_sided_t_p_value": _one_sided_t_p_value(array),
    }


def _one_sided_t_lower_bound(values: np.ndarray) -> float:
    array = _finite_vector(values, "paired contrast")
    if len(array) < 2:
        raise ValueError("paired t lower bound requires at least two participants.")
    mean = float(np.mean(array))
    standard_error = float(stats.sem(array))
    if standard_error == 0.0:
        return mean
    return mean - float(stats.t.ppf(0.95, len(array) - 1)) * standard_error


def _one_sided_t_p_value(values: np.ndarray) -> float:
    array = _finite_vector(values, "paired contrast")
    if len(array) < 2:
        raise ValueError("paired t test requires at least two participants.")
    mean = float(np.mean(array))
    standard_error = float(stats.sem(array))
    if standard_error == 0.0:
        return 0.0 if mean > 0.0 else 1.0
    statistic = mean / standard_error
    return float(stats.t.sf(statistic, len(array) - 1))


def holm_step_down(
    p_values: Mapping[str, float],
    *,
    familywise_alpha: float = 0.05,
) -> dict[str, dict[str, Any]]:
    """Return deterministic Holm step-down decisions for one named family."""

    if not isinstance(p_values, Mapping) or not p_values:
        raise ValueError("Holm correction requires a non-empty p-value mapping.")
    alpha = _finite_float(familywise_alpha, "familywise_alpha")
    if not 0.0 < alpha < 1.0:
        raise ValueError("familywise_alpha must lie strictly inside (0,1).")
    resolved: dict[str, float] = {}
    for endpoint, raw in p_values.items():
        if not isinstance(endpoint, str) or not endpoint:
            raise TypeError("Holm endpoint IDs must be non-empty strings.")
        value = _finite_float(raw, f"p_values[{endpoint!r}]")
        if not 0.0 <= value <= 1.0:
            raise ValueError("Holm p-values must lie in [0,1].")
        resolved[endpoint] = value
    ordered = sorted(resolved.items(), key=lambda item: (item[1], item[0]))
    count = len(ordered)
    continue_rejecting = True
    adjusted_running = 0.0
    result: dict[str, dict[str, Any]] = {}
    for rank, (endpoint, raw) in enumerate(ordered, start=1):
        threshold = alpha / (count - rank + 1)
        rejected = continue_rejecting and raw <= threshold
        if not rejected:
            continue_rejecting = False
        adjusted_running = max(adjusted_running, (count - rank + 1) * raw)
        result[endpoint] = {
            "raw_p_value": raw,
            "holm_rank": rank,
            "holm_threshold": threshold,
            "holm_adjusted_p_value": min(1.0, adjusted_running),
            "holm_reject": rejected,
        }
    return {endpoint: result[endpoint] for endpoint in sorted(result)}


def sensitivity_resampling_report(
    paired_vectors: Mapping[str, np.ndarray],
    *,
    grid_cell_index: int,
    rng_authority: v3.DevelopmentRNGAuthority,
) -> dict[str, Any]:
    """Run the frozen report-only 100k sign flips and 10k bootstraps."""

    authority = v3.require_development_rng_authority(rng_authority)
    _require_loaded_synthetic_source(authority)
    return _sensitivity_resampling_with_rng_factory(
        paired_vectors,
        grid_cell_index=grid_cell_index,
        rng_factory=_authorized_rng_factory(authority),
        sign_flip_draws=SENSITIVITY_SIGN_FLIP_DRAWS,
        participant_bootstrap_draws=SENSITIVITY_PARTICIPANT_BOOTSTRAP_DRAWS,
        require_complete=True,
        _issuer=_SYNTHETIC_RNG_PRODUCER_ISSUER,
    )


def _sensitivity_resampling_report_for_test(
    paired_vectors: Mapping[str, np.ndarray],
    *,
    grid_cell_index: int,
    rng_authority: _UnitTestRNGAuthority,
    sign_flip_draws: int,
    participant_bootstrap_draws: int,
) -> dict[str, Any]:
    """Small-draw sensitivity seam restricted to a nonreserved unit authority."""

    authority = _require_unit_test_rng_authority(rng_authority)
    return _sensitivity_resampling_with_rng_factory(
        paired_vectors,
        grid_cell_index=grid_cell_index,
        rng_factory=_unit_test_rng_factory(authority),
        sign_flip_draws=sign_flip_draws,
        participant_bootstrap_draws=participant_bootstrap_draws,
        require_complete=False,
        _issuer=_SYNTHETIC_RNG_PRODUCER_ISSUER,
    )


def _sensitivity_resampling_with_rng_factory(
    paired_vectors: Mapping[str, np.ndarray],
    *,
    grid_cell_index: int,
    rng_factory: RngFactory,
    sign_flip_draws: int,
    participant_bootstrap_draws: int,
    require_complete: bool,
    _issuer: object,
) -> dict[str, Any]:
    if _issuer is not _SYNTHETIC_RNG_PRODUCER_ISSUER:
        raise TypeError("sensitivity RNG producer issuer token is invalid.")
    cell_index = _positive_exact_int(grid_cell_index, "grid_cell_index")
    if cell_index not in range(1, 10):
        raise ValueError("grid_cell_index must be in the canonical range 1..9.")
    sign_draws = _positive_exact_int(sign_flip_draws, "sign_flip_draws")
    bootstrap_draws = _positive_exact_int(
        participant_bootstrap_draws,
        "participant_bootstrap_draws",
    )
    if require_complete:
        if tuple(paired_vectors) != SENSITIVITY_ENDPOINT_IDS:
            raise ValueError("production sensitivity requires every endpoint in exact order.")
        if (
            sign_draws != SENSITIVITY_SIGN_FLIP_DRAWS
            or bootstrap_draws != SENSITIVITY_PARTICIPANT_BOOTSTRAP_DRAWS
        ):
            raise ValueError("production sensitivity resampling counts are frozen.")
    elif not paired_vectors:
        raise ValueError("unit sensitivity requires at least one named endpoint.")

    endpoint_reports: dict[str, Any] = {}
    for endpoint, raw_vector in paired_vectors.items():
        if endpoint not in _SENSITIVITY_ENDPOINT_CODES:
            raise ValueError(f"unknown sensitivity endpoint {endpoint!r}.")
        values = _finite_vector(raw_vector, endpoint)
        if require_complete and len(values) != DEVELOPMENT_PARTICIPANTS:
            raise ValueError("production sensitivity vectors require 48 participants.")
        family = _sensitivity_family(endpoint)
        endpoint_code = _SENSITIVITY_ENDPOINT_CODES[endpoint]
        observed_mean = float(np.mean(values))

        sign_rng = _rng(
            rng_factory,
            family,
            cell_index,
            endpoint_code,
            0,
            "sensitivity_resampling",
        )
        extreme = 0
        remaining = sign_draws
        while remaining:
            chunk = min(4_096, remaining)
            signs = sign_rng.integers(0, 2, size=(chunk, len(values)), dtype=np.int8)
            signs = signs.astype(np.float64) * 2.0 - 1.0
            null_means = np.mean(signs * values[None, :], axis=1)
            extreme += int(np.sum(null_means >= observed_mean))
            remaining -= chunk
        sign_p = (extreme + 1.0) / (sign_draws + 1.0)

        bootstrap_rng = _rng(
            rng_factory,
            family,
            cell_index,
            endpoint_code,
            1,
            "sensitivity_resampling",
        )
        bootstrap_means = np.empty(bootstrap_draws, dtype=np.float64)
        remaining = bootstrap_draws
        offset = 0
        while remaining:
            chunk = min(4_096, remaining)
            indices = bootstrap_rng.integers(
                0,
                len(values),
                size=(chunk, len(values)),
            )
            bootstrap_means[offset : offset + chunk] = np.mean(
                values[indices], axis=1
            )
            offset += chunk
            remaining -= chunk
        lower, upper = np.quantile(bootstrap_means, [0.025, 0.975])
        endpoint_reports[endpoint] = {
            "n": len(values),
            "observed_mean": observed_mean,
            "sign_flip_draws": sign_draws,
            "sign_flip_extreme_count": extreme,
            "sign_flip_one_sided_p_value": float(sign_p),
            "participant_bootstrap_draws": bootstrap_draws,
            "participant_bootstrap_95_CI_lower": float(lower),
            "participant_bootstrap_95_CI_upper": float(upper),
            "sign_flip_rng_key": _sensitivity_rng_key(
                family, cell_index, endpoint_code, 0
            ),
            "participant_bootstrap_rng_key": _sensitivity_rng_key(
                family, cell_index, endpoint_code, 1
            ),
        }
    return {
        "schema": (
            "cfeg.metadata-calibration-efficiency-v3.sensitivity-resampling-report.v1"
        ),
        "component": "sensitivity_resampling",
        "promotion_or_selection_use": False,
        "endpoints": endpoint_reports,
    }


def _sensitivity_family(endpoint: str) -> str:
    prefix = endpoint.split("_", 1)[0]
    return {
        "B1": "B1_participant_class_confusion",
        "B2": "B2_participant_phase_spatial_shift",
        "B3": "B3_impedance_linked_transfer_shift",
        "B4": "B4_interface_calibrated_impedance_shift",
    }[prefix]


def _sensitivity_rng_key(
    family: str,
    grid_cell_index: int,
    endpoint_code: int,
    stream_code: int,
) -> dict[str, Any]:
    return {
        "family": family,
        "participant_index": grid_cell_index,
        "block": endpoint_code,
        "class_index": stream_code,
        "component": "sensitivity_resampling",
    }


def _paired_90_ci(values: np.ndarray) -> tuple[float, float]:
    array = _finite_vector(values, "equivalence contrast")
    if len(array) < 2:
        raise ValueError("paired equivalence interval requires at least two participants.")
    mean = float(np.mean(array))
    standard_error = float(stats.sem(array))
    if standard_error == 0.0:
        return mean, mean
    radius = float(stats.t.ppf(0.95, len(array) - 1)) * standard_error
    return mean - radius, mean + radius


def _clopper_pearson_upper(successes: int, total: int) -> float:
    successes = _nonnegative_exact_int(successes, "successes")
    total = _positive_exact_int(total, "total")
    if successes > total:
        raise ValueError("successes cannot exceed total.")
    if successes == total:
        return 1.0
    return float(stats.beta.ppf(0.95, successes + 1, total - successes))


def _clopper_pearson_lower(successes: int, total: int) -> float:
    successes = _nonnegative_exact_int(successes, "successes")
    total = _positive_exact_int(total, "total")
    if successes > total:
        raise ValueError("successes cannot exceed total.")
    if successes == 0:
        return 0.0
    return float(stats.beta.ppf(0.05, successes, total - successes + 1))


_GATE_REPORT_KEYS = {
    "schema",
    "grid_cell_index",
    "grid_cell_id",
    "operator_instance_sha256",
    "prototype_prior_pseudocount",
    "lambda_max",
    "components",
    "A_Q_viability",
    "metadata_efficacy",
    "B4_in_reference_metadata_efficacy",
    "B4_source_range_stress_safety",
    "B4_interface_scale_value",
    "pairing_mechanism",
    "deployment_viability",
    "pairing_potency",
    "N1_null_noninferiority",
    "N4_null_equivalence",
    "severe_harm",
    "adversarial_abstention",
    "helpful_use_anti_triviality",
    "selection_rank_tuple",
    "eligible",
}
_ENRICHED_GATE_REPORT_KEYS = {
    *_GATE_REPORT_KEYS,
    "sensitivity_participant_vectors",
    "uniform_block_weight_sensitivity",
    "resampling_sensitivity",
}


def _validate_gate_report(report: Mapping[str, Any]) -> None:
    report_keys = set(report)
    if report_keys != _GATE_REPORT_KEYS and report_keys != _ENRICHED_GATE_REPORT_KEYS:
        raise ValueError("grid gate report has missing or extra fields.")
    if report["schema"] != "cfeg.metadata-calibration-efficiency-v3.grid-gate-report.v1":
        raise ValueError("grid gate report schema is invalid.")
    cell = v3.grid_cell_by_id(report["grid_cell_id"])
    if (
        report["grid_cell_index"] != cell.index
        or report["operator_instance_sha256"] != cell.operator_instance_sha256
        or report["prototype_prior_pseudocount"] != cell.prototype_prior_pseudocount
        or report["lambda_max"] != cell.lambda_max
    ):
        raise ValueError("grid gate report cell binding is invalid.")
    expected_components = {
        "invariant_suite",
        "pairing_potency",
        "A_Q_viability_B1_or_B2",
        "B3_metadata_efficacy",
        "B4_metadata_efficacy",
        "B4_in_reference_metadata_efficacy",
        "B4_source_range_stress_safety",
        "B4_interface_scale_value",
        "k3_pairing_mechanism",
        "B3_B4_deployment_viability",
        "N1_null_noninferiority",
        "N4_null_equivalence",
        "severe_harm",
        "adversarial_abstention",
        "helpful_use_anti_triviality",
    }
    _require_exact_keys(report["components"], expected_components, "promotion components")
    if any(type(value) is not bool for value in report["components"].values()):
        raise TypeError("promotion components must be exact booleans.")
    if type(report["eligible"]) is not bool or report["eligible"] != all(
        report["components"].values()
    ):
        raise ValueError("grid eligibility is not the exact component intersection.")
    rank = report["selection_rank_tuple"]
    _require_exact_keys(
        rank,
        {"minimum_mandatory_observed_gain", "minimum_corresponding_one_sided_LCB"},
        "selection rank tuple",
    )
    for value in rank.values():
        _finite_float(value, "selection rank component")
    if report_keys == _ENRICHED_GATE_REPORT_KEYS:
        _validate_gate_report_sensitivities(report)


def _validate_gate_report_sensitivities(
    report: Mapping[str, Any],
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    primary_vectors, uniform_vectors = _validated_sensitivity_participant_vectors(
        report["sensitivity_participant_vectors"]
    )
    uniform = report["uniform_block_weight_sensitivity"]
    if type(uniform) is not dict:
        raise TypeError("uniform block-weight sensitivity must be a dictionary.")
    _require_exact_keys(
        uniform,
        {"schema", "promotion_or_selection_use", "common_weight_rule", "endpoints"},
        "uniform block-weight sensitivity",
    )
    if (
        uniform["schema"]
        != "cfeg.metadata-calibration-efficiency-v3.uniform-block-weight-sensitivity.v1"
        or uniform["promotion_or_selection_use"] is not False
        or uniform["common_weight_rule"]
        != "both_p_support_and_g_M_use_exact_one_over_k"
        or type(uniform["endpoints"]) is not dict
        or tuple(uniform["endpoints"]) != SENSITIVITY_ENDPOINT_IDS
    ):
        raise ValueError("uniform block-weight sensitivity identity/order is invalid.")
    for endpoint, summary in uniform["endpoints"].items():
        if type(summary) is not dict:
            raise TypeError("uniform sensitivity endpoint summaries must be dictionaries.")
        _require_exact_keys(
            summary,
            {
                "metric",
                "n",
                "reliability_weighted_observed_mean",
                "uniform_weight_observed_mean",
                "paired_weighted_minus_uniform_mean",
            },
            f"uniform sensitivity endpoint {endpoint}",
        )
        expected_metric = (
            "participant_eAUC"
            if endpoint.endswith("eAUC")
            else "participant_k3_balanced_accuracy"
        )
        if summary["metric"] != expected_metric or summary["n"] != DEVELOPMENT_PARTICIPANTS:
            raise ValueError("uniform sensitivity endpoint identity/count is invalid.")
        for name in (
            "reliability_weighted_observed_mean",
            "uniform_weight_observed_mean",
            "paired_weighted_minus_uniform_mean",
        ):
            if type(summary[name]) is not float:
                raise TypeError(f"{endpoint}.{name} must be an exact float.")
            _finite_float(summary[name], f"{endpoint}.{name}")
        primary = primary_vectors[endpoint]
        uniform_values = uniform_vectors[endpoint]
        paired = primary - uniform_values
        if (
            summary["reliability_weighted_observed_mean"]
            != float(np.mean(primary))
            or summary["uniform_weight_observed_mean"]
            != float(np.mean(uniform_values))
            or summary["paired_weighted_minus_uniform_mean"]
            != float(np.mean(paired))
        ):
            raise ValueError(
                "uniform sensitivity summaries differ from participant vectors."
            )

    resampling = report["resampling_sensitivity"]
    if type(resampling) is not dict:
        raise TypeError("resampling sensitivity must be a dictionary.")
    _require_exact_keys(
        resampling,
        {"schema", "component", "promotion_or_selection_use", "endpoints"},
        "resampling sensitivity",
    )
    if (
        resampling["schema"]
        != "cfeg.metadata-calibration-efficiency-v3.sensitivity-resampling-report.v1"
        or resampling["component"] != "sensitivity_resampling"
        or resampling["promotion_or_selection_use"] is not False
        or type(resampling["endpoints"]) is not dict
        or tuple(resampling["endpoints"]) != SENSITIVITY_ENDPOINT_IDS
    ):
        raise ValueError("resampling sensitivity identity/order is invalid.")
    for endpoint, summary in resampling["endpoints"].items():
        if type(summary) is not dict:
            raise TypeError("resampling endpoint summaries must be dictionaries.")
        _require_exact_keys(
            summary,
            {
                "n",
                "observed_mean",
                "sign_flip_draws",
                "sign_flip_extreme_count",
                "sign_flip_one_sided_p_value",
                "participant_bootstrap_draws",
                "participant_bootstrap_95_CI_lower",
                "participant_bootstrap_95_CI_upper",
                "sign_flip_rng_key",
                "participant_bootstrap_rng_key",
            },
            f"resampling endpoint {endpoint}",
        )
        if (
            summary["n"] != DEVELOPMENT_PARTICIPANTS
            or summary["sign_flip_draws"] != SENSITIVITY_SIGN_FLIP_DRAWS
            or summary["participant_bootstrap_draws"]
            != SENSITIVITY_PARTICIPANT_BOOTSTRAP_DRAWS
        ):
            raise ValueError("resampling sensitivity frozen counts are invalid.")
        extreme = _nonnegative_exact_int(
            summary["sign_flip_extreme_count"],
            "sign_flip_extreme_count",
        )
        if extreme > SENSITIVITY_SIGN_FLIP_DRAWS:
            raise ValueError("sign-flip extreme count exceeds draw count.")
        for name in (
            "observed_mean",
            "sign_flip_one_sided_p_value",
            "participant_bootstrap_95_CI_lower",
            "participant_bootstrap_95_CI_upper",
        ):
            if type(summary[name]) is not float:
                raise TypeError(f"{endpoint}.{name} must be an exact float.")
            _finite_float(summary[name], f"{endpoint}.{name}")
        expected_sign_p = (extreme + 1.0) / (
            SENSITIVITY_SIGN_FLIP_DRAWS + 1.0
        )
        if (
            summary["observed_mean"] != float(np.mean(primary_vectors[endpoint]))
            or summary["sign_flip_one_sided_p_value"] != expected_sign_p
            or not 0.0 <= summary["sign_flip_one_sided_p_value"] <= 1.0
            or summary["participant_bootstrap_95_CI_lower"]
            > summary["participant_bootstrap_95_CI_upper"]
        ):
            raise ValueError(
                "resampling sensitivity statistics differ from exact formulas."
            )
        family = _sensitivity_family(endpoint)
        code = _SENSITIVITY_ENDPOINT_CODES[endpoint]
        if summary["sign_flip_rng_key"] != _sensitivity_rng_key(
            family, int(report["grid_cell_index"]), code, 0
        ) or summary["participant_bootstrap_rng_key"] != _sensitivity_rng_key(
            family, int(report["grid_cell_index"]), code, 1
        ):
            raise ValueError("sensitivity RNG key binding is invalid.")
    return primary_vectors, uniform_vectors


def _artifact_reference(
    value: Mapping[str, Any],
    name: str,
    *,
    expected_schema: str,
) -> dict[str, str]:
    _require_exact_keys(value, {"schema", "payload_sha256", "file_sha256"}, name)
    if value["schema"] != expected_schema:
        raise ValueError(f"{name} schema is invalid.")
    _sha256(value["payload_sha256"], f"{name}.payload_sha256")
    _sha256(value["file_sha256"], f"{name}.file_sha256")
    return dict(value)


def _grid_record(cell: v3.V3GridCell) -> dict[str, Any]:
    return {
        "index": cell.index,
        "id": cell.grid_cell_id,
        "nu_token": cell.nu_token,
        "lambda_token": cell.lambda_token,
        "prototype_prior_pseudocount": cell.prototype_prior_pseudocount,
        "lambda_max": cell.lambda_max,
        "operator_instance_sha256": cell.operator_instance_sha256,
    }


def _payload_sha256(value: Mapping[str, Any]) -> str:
    payload = dict(value)
    payload.pop("payload_sha256", None)
    return _canonical_sha256(payload)


def _git_object_id(value: Any, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 40
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{name} must be a 40-character lowercase Git object ID.")
    return value


def _packet_at_reference_z(
    reference: v3.ContextReference,
    packet_key: str,
    scale_key: str,
    z_value: float,
    interface_lookup_key: str | None = None,
) -> v3.ContextPacket:
    table = (
        reference.pooled_table
        if scale_key == "__pooled__"
        else reference.table_by_interface[scale_key]
    )
    impedance = []
    availability = []
    for center, scale in zip(table.centers, table.scales, strict=True):
        if center is None or scale is None:
            impedance.append(None)
            availability.append(False)
        else:
            impedance.append(float(np.expm1(center + z_value * scale)))
            availability.append(True)
    return v3.ContextPacket(
        packet_key=packet_key,
        interface_lookup_key=(scale_key if interface_lookup_key is None else interface_lookup_key),
        impedance_kohm_by_channel=tuple(impedance),
        channel_availability=tuple(availability),
    )


def _context_packet_mapping(packet: v3.ContextPacket) -> dict[str, Any]:
    return {
        "packet_key": packet.packet_key,
        "interface_lookup_key": packet.interface_lookup_key,
        "impedance_kohm_by_channel": list(packet.impedance_kohm_by_channel),
        "channel_availability": list(packet.channel_availability),
    }


def _one_block_preflight(
    reference: v3.ContextReference,
    validated_reference: v3.ValidatedContextReference | None,
    query: Mapping[str, Any] | v3.ContextPacket,
    support: Mapping[str, Any] | v3.ContextPacket,
    *,
    pairing_override: str | None = None,
) -> v3.ContextPreflightCapability:
    query_key = query.packet_key if type(query) is v3.ContextPacket else str(query["packet_key"])
    support_key = (
        support.packet_key if type(support) is v3.ContextPacket else str(support["packet_key"])
    )
    pairing = pairing_override or v3.context_pairing_sha256(
        query_key=query_key,
        ordered_support_block_keys=("N5-support",),
        query_packet_key=query_key,
        ordered_support_packet_keys=(support_key,),
    )
    arguments = {
        "reference": reference,
        "query_key": query_key,
        "ordered_support_block_keys": ("N5-support",),
        "query_packet": query,
        "support_packets": (support,),
        "pairing_sha256": pairing,
    }
    if validated_reference is None:
        return v3._preflight_context_unit(**arguments)
    return v3._preflight_context_after_authorized_boundary(
        validated_reference=validated_reference,
        **arguments,
    )


def _balanced_accuracy_from_probabilities(
    probabilities: np.ndarray,
    labels: np.ndarray,
) -> float:
    values = np.asarray(probabilities, dtype=np.float64)
    label_values = np.asarray(labels, dtype=np.int64)
    if values.ndim != 2 or label_values.shape != (len(values),):
        raise ValueError("probability/label metric shapes are invalid.")
    predictions = np.argmax(values, axis=1)
    recalls = []
    for class_index in range(values.shape[1]):
        mask = label_values == class_index
        if not np.any(mask):
            raise ValueError("balanced accuracy requires every class.")
        recalls.append(float(np.mean(predictions[mask] == class_index)))
    return float(np.mean(recalls))


def _correct_log_probability(probabilities: np.ndarray, labels: np.ndarray) -> float:
    values = np.asarray(probabilities, dtype=np.float64)
    label_values = np.asarray(labels, dtype=np.int64)
    if values.ndim != 2 or label_values.shape != (len(values),):
        raise ValueError("probability/label metric shapes are invalid.")
    correct = np.clip(values[np.arange(len(values)), label_values], 1.0e-12, 1.0)
    return float(np.mean(np.log(correct)))


def _finite_vector(value: np.ndarray, name: str) -> np.ndarray:
    result = np.asarray(value, dtype=np.float64)
    if result.ndim != 1 or result.size == 0 or not np.isfinite(result).all():
        raise ValueError(f"{name} must be a non-empty finite vector.")
    return result


def _finite_float(value: Any, name: str) -> float:
    if not isinstance(value, (int, float, np.integer, np.floating)) or isinstance(value, bool):
        raise TypeError(f"{name} must be a finite real number.")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite.")
    return result


def _require_exact_keys(value: Mapping[str, Any], expected: set[str], name: str) -> None:
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be a mapping.")
    observed = set(value)
    if observed != expected:
        raise ValueError(
            f"{name} keys differ; missing={sorted(expected-observed)}, "
            f"extra={sorted(observed-expected)}."
        )


def _validate_exact_plan_sections(plan: Mapping[str, Any]) -> None:
    rng = _mapping(plan, "rng")
    if (
        rng.get("bit_generator") != "numpy.random.PCG64DXSM"
        or rng.get("construction") != "numpy.random.SeedSequence"
        or rng.get("development_root_seed") != 20260909
        or rng.get("key_order")
        != [
            "root_seed",
            "family_code",
            "participant_index",
            "block",
            "class_index",
            "component_code",
        ]
        or rng.get("family_codes")
        != {
            "B1_participant_class_confusion": 3101,
            "B2_participant_phase_spatial_shift": 3102,
            "B3_impedance_linked_transfer_shift": 3103,
            "B4_interface_calibrated_impedance_shift": 3104,
            "N1_clean_anchor": 3201,
            "N2_random_support_labels": 3202,
            "N3_nonstationary_calibration": 3203,
            "N4_context_null": 3204,
            "N5_invalid_context": 3205,
        }
        or tuple(rng.get("component_codes", {})) != COMPONENT_NAMES
        or tuple(rng.get("family_codes", {})) != RNG_FAMILY_NAMES
    ):
        raise ValueError("synthetic RNG contract is not exact.")
    if rng.get("component_codes") != {
        name: index for index, name in enumerate(COMPONENT_NAMES, start=1)
    }:
        raise ValueError("synthetic component-code mapping is not exact.")
    population = _mapping(plan, "population")
    expected_population = {
        "development_participants_per_family": 48,
        "future_scientific_lockbox_participants_per_family": 96,
        "classes": 12,
        "complete_blocks": 10,
        "trials_per_class_per_block": 1,
        "primary_support_budgets": [0, 1, 3],
        "descriptive_support_budget": 5,
        "immutable_query_blocks": [6, 7, 8, 9, 10],
    }
    if any(population.get(key) != value for key, value in expected_population.items()):
        raise ValueError("synthetic population contract is not exact.")
    waveform = _mapping(plan, "waveform")
    if (
        waveform.get("dtype") != "float64"
        or waveform.get("sampling_rate_hz") != 250
        or waveform.get("duration_seconds") != 2.0
        or waveform.get("channels") != 8
        or waveform.get("class_frequency_hz") != "8.0_plus_0.4_times_zero_based_class"
        or waveform.get("harmonic_amplitudes") != {1: 1.0, 2: 0.35}
        or waveform.get("noise")
        != {
            "process": "stationary_AR1_per_channel",
            "rho": 0.55,
            "initialization": "epsilon_0_equals_sigma_times_z_0_div_sqrt_1_minus_rho_squared",
            "recurrence": "epsilon_t_equals_rho_times_epsilon_t_minus_1_plus_sigma_times_z_t",
        }
    ):
        raise ValueError("synthetic waveform contract is not exact.")
    reference = _mapping(plan, "context_reference")
    if (
        reference.get("root_seed") != 20260910
        or reference.get("interface_categories") != ["neutral", "wet", "dry"]
        or reference.get("packets_per_interface") != 256
        or reference.get("channels") != 8
        or reference.get("minimum_observed_packets_per_interface_channel") != 128
        or reference.get("robust_scale_floor") != 0.05
        or reference.get("pooled_per_channel_fallback_minimum_count") != 256
    ):
        raise ValueError("synthetic context-reference contract is not exact.")
    if tuple(_mapping(plan, "families")) != (
        *FAMILY_NAMES,
        "N5_invalid_context",
        "N6_support_reliability_invariants",
    ):
        raise ValueError("synthetic family order/membership is not exact.")
    complete = _mapping(plan, "complete_grid")
    expected_counts = {
        "condition_level_cells_per_parameter_pair": 172,
        "B4_composite_cells_per_parameter_pair": 34,
        "participant_summary_cells_per_parameter_pair": 206,
        "development_participant_rows_per_parameter_pair": 9_888,
        "development_parameter_pairs": 9,
        "development_participant_metric_rows": DEVELOPMENT_METRIC_ROWS,
        "N5_context_invariant_cases_per_parameter_pair": 6,
        "N6_support_reliability_invariant_cases_per_parameter_pair": 4,
        "development_invariant_rows": DEVELOPMENT_INVARIANT_ROWS,
        "total_development_rows": DEVELOPMENT_TOTAL_ROWS,
        "future_selected_lockbox_total_rows": FUTURE_SELECTED_TOTAL_ROWS,
    }
    if any(complete.get(key) != value for key, value in expected_counts.items()):
        raise ValueError("synthetic complete-grid arithmetic is not exact.")
    amendment = _mapping(plan, "preoutcome_amendment")
    if amendment != {
        "path": "configs/governance/metadata_calibration_v3_2_recovery_amendment.json",
        "file_sha256": ("1db2d118376975c1817e329f4203dbaf9929d4f2bcb30bfede990af26c2c1ceb"),
        "protocol_revision": "V3.2",
        "development_attempt_id": "development-v3",
        "original_path": ("configs/governance/metadata_calibration_v3_preoutcome_amendment.json"),
        "prior_recovery_file_sha256": (
            "deddca9286f07c6293925409d01b3df1313e89f2ce89688c4a25c14a8ec01238"
        ),
        "prior_recovery_path": (
            "configs/governance/metadata_calibration_v3_1_recovery_amendment.json"
        ),
        "original_file_sha256": (
            "3238761a0a9a257032d6582286953a4d20981e5e646c9f1322c5c0357bd0c202"
        ),
        "scientific_contract_projection_sha256": (
            "0702d01be1e055d3203a3c1b78777db6456b8d527e5525b6d468fb52522f8a79"
        ),
    }:
        raise ValueError("V3.2 recovery amendment binding is not exact.")
    bundle = _mapping(plan, "development_bundle")
    if (
        bundle.get("current_status") != "absent"
        or bundle.get("protocol_revision") != "V3.2"
        or bundle.get("development_attempt_id") != "development-v3"
        or bundle.get("schema")
        != "cfeg.metadata-calibration-efficiency-v3.development-bundle.v3"
        or bundle.get("canonical_path")
        != "/home/whwovy/v3-artifacts/metadata-calibration-efficiency-v3/development-v3/development-bundle.json"
        or bundle.get("context_reference_canonical_path")
        != "/home/whwovy/v3-artifacts/metadata-calibration-efficiency-v3/development-v3/context-reference.json"
        or bundle.get("development_result_canonical_path")
        != "/home/whwovy/v3-artifacts/metadata-calibration-efficiency-v3/development-v3/development-result.json"
    ):
        raise ValueError("development bundle must remain absent in the frozen plan.")
    start = _mapping(bundle, "development_start_receipt")
    if start != {
        "schema": "cfeg.metadata-calibration-efficiency-v3.development-start.v1",
        "canonical_path": (
            "/home/whwovy/v3-artifacts/metadata-calibration-efficiency-v3/"
            "development-v3/development-start.json"
        ),
        "creation": "O_EXCL_after_contract_validation_before_development_authority_or_RNG",
        "execution_capability": "private_and_creator_process_only",
        "fresh_receipt_without_result": ("terminal_DEVELOPMENT_ATTEMPT_CONSUMED_no_resume"),
        "fresh_receipt_with_result": "read_only_reopen_and_audit",
        "development_result_must_bind": [
            "schema",
            "payload_sha256",
            "file_sha256",
        ],
    }:
        raise ValueError("development-start receipt contract is not exact.")
    recovery = _mapping(bundle, "recovery_prerequisite")
    if (
        recovery.get("amendment")
        != "configs/governance/metadata_calibration_v3_2_recovery_amendment.json"
        or recovery.get("retired_v1_attempt_id") != "development-v1"
        or recovery.get("retired_v1_exact_file_count") != 2
        or recovery.get("retired_v1_bundle_file_sha256")
        != "9f6027918110cfd877a1b170f3472d897d4b06f6fca7f0991b9f15f6768c4a1a"
        or recovery.get("retired_v1_context_file_sha256")
        != "7b78b3092f5824b6e247d97f4b9cd9f9ec22dbfa74a182e2f32c34fb1c8cf3c8"
        or recovery.get("retired_v2_attempt_id") != "development-v2"
        or recovery.get("retired_v2_exact_file_count") != 1
        or recovery.get("retired_v2_bundle_file_sha256")
        != "a616945040ea45a78c2eed58bf251f1f379f9a82a320ed4f39c26f5efb073cac"
        or recovery.get("retired_v2_context_reference_artifact_count") != 0
        or recovery.get("retired_v2_context_reference_seedsequence_and_replay_executed") is not True
        or recovery.get("retired_v2_context_reference_replay_count") != 2
        or recovery.get("retired_v2_development_seedsequence_created") is not False
        or recovery.get("retired_v1_and_v2_continuation_authorized") is not False
        or recovery.get("adopt_retired_context_exact_bytes") is not True
    ):
        raise ValueError("V3.2 recovery prerequisite is not exact.")
    lockbox = _mapping(plan, "scientific_lockbox")
    if lockbox.get("current_authority") is not False or lockbox.get("future_beacon_selected") is not False:
        raise ValueError("scientific lockbox authority must be false.")


def _generate_condition_context_packets(
    *,
    family: str,
    condition: str,
    participant_index: int,
    states: np.ndarray,
    rng_factory: RngFactory,
) -> tuple[v3.ContextPacket, ...]:
    packets: list[v3.ContextPacket] = []
    informative = family in {
        "B3_impedance_linked_transfer_shift",
        "B4_interface_calibrated_impedance_shift",
        "N4_context_null",
    }
    if family == "B4_interface_calibrated_impedance_shift":
        condition_index = 0 if condition == "wet" else 1
        assignment_rng = _rng(
            rng_factory,
            family,
            participant_index,
            0,
            condition_index,
            "interface_assignment",
        )
        participant_offset = float(assignment_rng.normal(0.0, 0.15))
        slope_multiplier = float(assignment_rng.lognormal(0.0, 0.20))
        noise_multiplier = float(assignment_rng.lognormal(0.0, 0.15))
    else:
        participant_offset = 0.0
        slope_multiplier = 1.0
        noise_multiplier = 1.0
    for block_index in range(10):
        block = block_index + 1
        key = f"participant-{participant_index:03d}:{condition}:block{block:02d}"
        if not informative:
            packets.append(
                v3.ContextPacket(
                    packet_key=key,
                    interface_lookup_key=None,
                    impedance_kohm_by_channel=(None,) * 8,
                    channel_availability=(False,) * 8,
                )
            )
            continue
        generator = _rng(
            rng_factory,
            family,
            participant_index,
            block,
            0 if condition != "dry" else 1,
            "impedance_measurement_noise",
        )
        state = int(states[block_index])
        offsets = 0.04 * (np.arange(8, dtype=np.float64) - 3.5)
        if condition == "neutral":
            center, slope, noise_sd = math.log1p(6.0), 0.65, 0.05
            interface = "neutral"
        elif condition == "wet":
            center, slope, noise_sd = math.log1p(20.0), 0.20, 0.03
            interface = "wet"
        elif condition == "dry":
            center, slope, noise_sd = math.log1p(262.0), 0.80, 0.08
            interface = "dry"
        else:
            raise ValueError("unknown synthetic condition.")
        log_values = (
            center
            + participant_offset
            + slope_multiplier * slope * state
            + offsets
            + generator.normal(0.0, noise_sd * noise_multiplier, 8)
        )
        impedance = np.expm1(log_values)
        if np.any(impedance < 0.0) or not np.isfinite(impedance).all():
            raise RuntimeError("synthetic impedance generation left the valid domain.")
        packets.append(
            v3.ContextPacket(
                packet_key=key,
                interface_lookup_key=interface,
                impedance_kohm_by_channel=tuple(float(value) for value in impedance),
                channel_availability=(True,) * 8,
            )
        )
    return tuple(packets)


def _balanced_states(generator: np.random.Generator) -> np.ndarray:
    states = np.asarray([-1] * 5 + [1] * 5, dtype=np.int8)
    return states[generator.permutation(10)]


def _regime(
    state: int,
    participant_index: int,
    class_index: int,
    classes: int,
    *,
    plus_noise: float,
) -> tuple[int, float, float]:
    if state == -1:
        return (class_index + 1 + participant_index % 3) % classes, 0.0, 0.85
    if state == 1:
        return (class_index + 4 + participant_index % 3) % classes, 0.5 * np.pi, plus_noise
    raise ValueError("context-linked signal state must be -1 or +1.")


def _oscillation(
    frequency_hz: float,
    phase: float,
    drift: float,
    timeline: np.ndarray,
) -> np.ndarray:
    angle = 2.0 * np.pi * frequency_hz * timeline + phase + drift
    return np.sin(angle) + 0.35 * np.sin(2.0 * angle)


def _stationary_ar1(
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


def _rng(
    factory: RngFactory,
    family: str,
    participant_index: int,
    block: int,
    class_index: int,
    component: str,
) -> np.random.Generator:
    if family not in RNG_FAMILY_NAMES or component not in COMPONENT_NAMES:
        raise ValueError("RNG key is outside the frozen family/component vocabulary.")
    for value, name in (
        (participant_index, "participant_index"),
        (block, "block"),
        (class_index, "class_index"),
    ):
        _nonnegative_exact_int(value, name)
    result = factory(family, participant_index, block, class_index, component)
    if type(result) is not np.random.Generator or type(result.bit_generator) is not np.random.PCG64DXSM:
        raise TypeError("RNG factory must return an exact Generator(PCG64DXSM).")
    return result


def _authorized_rng_factory(
    authority: v3.ContextReferenceRNGAuthority | v3.DevelopmentRNGAuthority,
) -> RngFactory:
    """Construct keyed PCG64DXSM only after revalidating authority per draw."""

    if type(authority) is v3.ContextReferenceRNGAuthority:
        checker: Callable[[object], object] = (
            v3._require_context_reference_rng_authority_semantics
        )
    elif type(authority) is v3.DevelopmentRNGAuthority:
        checker = v3._require_development_rng_authority_semantics
    else:
        raise TypeError("an exact context-reference or development RNG authority is required.")

    def factory(
        family: str,
        participant_index: int,
        block: int,
        class_index: int,
        component: str,
    ) -> np.random.Generator:
        current = checker(authority)
        family_codes = dict(current.family_codes)  # type: ignore[attr-defined]
        component_codes = dict(current.component_codes)  # type: ignore[attr-defined]
        entropy = [
            current.root_seed,  # type: ignore[attr-defined]
            family_codes[family],
            participant_index,
            block,
            class_index,
            component_codes[component],
        ]
        return np.random.Generator(
            np.random.PCG64DXSM(np.random.SeedSequence(entropy))
        )

    return factory


def _require_loaded_synthetic_source(
    authority: v3.ContextReferenceRNGAuthority | v3.DevelopmentRNGAuthority,
) -> None:
    """Reject old imported code even when another process restored clean disk A."""

    if type(authority) is v3.ContextReferenceRNGAuthority:
        current = v3._require_context_reference_rng_authority_semantics(authority)
    elif type(authority) is v3.DevelopmentRNGAuthority:
        current = v3._require_development_rng_authority_semantics(authority)
    else:
        raise TypeError("an exact production RNG authority is required.")
    current_path = Path(__file__).resolve()
    current_sha256 = hashlib.sha256(current_path.read_bytes()).hexdigest()
    inventory = {
        entry.path: entry.file_sha256
        for entry in current._bundle_capability.tracked_source_files
    }
    if (
        current_path
        != (_REPOSITORY / SYNTHETIC_MODULE_REPOSITORY_PATH).resolve()
        or current_sha256 != _IMPORTED_SYNTHETIC_MODULE_SHA256
        or inventory.get(SYNTHETIC_MODULE_REPOSITORY_PATH)
        != _IMPORTED_SYNTHETIC_MODULE_SHA256
    ):
        raise RuntimeError(
            "loaded V3 synthetic code does not equal current bundle-bound source bytes."
        )
    _require_fbcca_runtime_identity(inventory)


def _require_fbcca_runtime_identity(inventory: Mapping[str, str]) -> None:
    current_path = Path(_fbcca_module.__file__).resolve()
    current_sha256 = hashlib.sha256(current_path.read_bytes()).hexdigest()
    if (
        current_path != (_REPOSITORY / FBCCA_MODULE_REPOSITORY_PATH).resolve()
        or _FBCCA_MODULE_IMPORT_PATH != current_path
        or current_sha256 != _IMPORTED_FBCCA_MODULE_SHA256
        or inventory.get(FBCCA_MODULE_REPOSITORY_PATH)
        != _IMPORTED_FBCCA_MODULE_SHA256
        or inventory.get(FILTERBANK_REPOSITORY_PATH) != EXPECTED_FILTERBANK_SHA256
        or _fbcca_module.np is not _IMPORTED_FBCCA_NUMPY_MODULE
        or _fbcca_module.signal is not _IMPORTED_FBCCA_SCIPY_SIGNAL_MODULE
        or apply_filterbank is not _fbcca_module.apply_filterbank
        or cca_score is not _fbcca_module.cca_score
        or make_reference_signals is not _fbcca_module.make_reference_signals
        or resolve_filterbank_parameters
        is not _fbcca_module.resolve_filterbank_parameters
    ):
        raise RuntimeError(
            "loaded strict-FBCCA code differs from the bundle-bound module."
        )
    for name, imported_callable, imported_code_sha256 in (
        _IMPORTED_FBCCA_CALLABLE_BINDINGS
    ):
        current_callable = getattr(_fbcca_module, name, None)
        if (
            current_callable is not imported_callable
            or _callable_code_sha256(current_callable) != imported_code_sha256
        ):
            raise RuntimeError(
                f"loaded strict-FBCCA callable {name!r} changed after import."
            )


def _unit_test_rng_factory(authority: _UnitTestRNGAuthority) -> RngFactory:
    """Construct keyed PCG64DXSM from a nonreserved unit-test authority."""

    authority = _require_unit_test_rng_authority(authority)
    family_codes = dict(v3.RNG_FAMILY_CODES)
    component_codes = dict(v3.RNG_COMPONENT_CODES)

    def factory(
        family: str,
        participant_index: int,
        block: int,
        class_index: int,
        component: str,
    ) -> np.random.Generator:
        current = _require_unit_test_rng_authority(authority)
        entropy = [
            current.root_seed,
            family_codes[family],
            participant_index,
            block,
            class_index,
            component_codes[component],
        ]
        return np.random.Generator(
            np.random.PCG64DXSM(np.random.SeedSequence(entropy))
        )

    return factory


def _require_exact_contract(value: object) -> SyntheticV3Contract:
    if type(value) is not SyntheticV3Contract:
        raise TypeError("contract must be an exact SyntheticV3Contract.")
    try:
        issuer = value._issuer
    except AttributeError as error:
        raise TypeError("SyntheticV3Contract was not issued by this module.") from error
    if issuer is not _SYNTHETIC_CONTRACT_ISSUER:
        raise TypeError("synthetic contract issuer is invalid.")
    if (
        value.plan_sha256 != EXPECTED_SYNTHETIC_PLAN_SHA256
        or value.master_plan_sha256 != EXPECTED_MASTER_PLAN_SHA256
        or value.filterbank_sha256 != EXPECTED_FILTERBANK_SHA256
        or value.plan_path != DEFAULT_SYNTHETIC_PLAN_PATH.resolve()
        or value.master_plan_path != DEFAULT_MASTER_PLAN_PATH.resolve()
        or value.filterbank_path != DEFAULT_FILTERBANK_PATH.resolve()
    ):
        raise ValueError("synthetic contract identity or plan binding is invalid.")
    _require_deep_frozen_json(value.plan, "contract.plan")
    _require_deep_frozen_json(value.filterbank, "contract.filterbank")
    thawed_plan = _deep_thaw_json(value.plan)
    thawed_filterbank = _deep_thaw_json(value.filterbank)
    if (
        value.plan_content_sha256
        != _canonical_sha256({"value": thawed_plan})
        or value.filterbank_content_sha256
        != _canonical_sha256({"value": thawed_filterbank})
        or value.semantic_binding_sha256
        != _canonical_sha256(_contract_semantic_binding(value))
    ):
        raise ValueError("synthetic contract in-memory content binding is invalid.")
    _validate_exact_plan_sections(thawed_plan)
    _validate_resolved_filterbank(thawed_filterbank)
    return value


def _require_deep_frozen_json(value: Any, name: str) -> None:
    if type(value) is MappingProxyType:
        if any(type(key) not in {str, int} for key in value):
            raise TypeError(f"{name} contains an invalid mapping key.")
        for key, item in value.items():
            _require_deep_frozen_json(item, f"{name}[{key!r}]")
        return
    if type(value) is tuple:
        for index, item in enumerate(value):
            _require_deep_frozen_json(item, f"{name}[{index}]")
        return
    if value is None or type(value) in {bool, int, float, str}:
        return
    raise TypeError(f"{name} is not deeply immutable JSON data.")


def _validate_resolved_filterbank(filterbank: Mapping[str, Any]) -> None:
    resolved = resolve_filterbank_parameters(filterbank, sfreq=250.0)
    if (
        len(resolved["bands_hz"]) != 7
        or resolved["n_harmonics"] != 5
        or resolved["weights"]
        != [
            1.25,
            0.6704482076268572,
            0.5032785618838642,
            0.42677669529663687,
            0.38374806099528436,
            0.35649051737437876,
            0.33782687899303776,
        ]
    ):
        raise ValueError("resolved strict-FBCCA parameters are not frozen V3 values.")


def _require_current_contract_files(value: object) -> SyntheticV3Contract:
    """Expensive public-boundary validation of all three frozen input files."""

    contract = _require_exact_contract(value)
    paths_and_hashes = (
        (contract.plan_path, EXPECTED_SYNTHETIC_PLAN_SHA256, "synthetic plan"),
        (contract.master_plan_path, EXPECTED_MASTER_PLAN_SHA256, "master plan"),
        (contract.filterbank_path, EXPECTED_FILTERBANK_SHA256, "filterbank"),
    )
    for path, expected_sha256, name in paths_and_hashes:
        if not path.is_file() or path.is_symlink():
            raise ValueError(f"{name} must remain one canonical regular file.")
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected_sha256:
            raise ValueError(f"{name} bytes differ from the frozen contract.")
    decoded_plan = yaml.safe_load(contract.plan_path.read_text(encoding="utf-8"))
    decoded_filterbank = yaml.safe_load(
        contract.filterbank_path.read_text(encoding="utf-8")
    )
    if not isinstance(decoded_plan, Mapping) or not isinstance(
        decoded_filterbank, Mapping
    ):
        raise TypeError("frozen contract files must decode to mappings.")
    if (
        decoded_plan != _deep_thaw_json(contract.plan)
        or dict(decoded_filterbank) != _deep_thaw_json(contract.filterbank)
    ):
        raise ValueError("in-memory contract differs from canonical file content.")
    _validate_exact_plan_sections(decoded_plan)
    _validate_resolved_filterbank(decoded_filterbank)
    return contract


def _require_shared_bundle_provenance(
    authority: v3.DevelopmentRNGAuthority,
    reference: v3.ValidatedContextReference,
) -> None:
    """Require reference and development execution to share one sealed bundle."""

    if type(authority) is not v3.DevelopmentRNGAuthority:
        raise TypeError("an exact DevelopmentRNGAuthority is required.")
    if type(reference) is not v3.ValidatedContextReference:
        raise TypeError("an exact ValidatedContextReference is required.")
    if (
        reference.development_bundle_payload_sha256
        != authority.development_bundle_payload_sha256
        or reference.development_bundle_file_sha256
        != authority.development_bundle_file_sha256
        or reference.development_bundle_source_bundle_sha256
        != authority.development_bundle_source_bundle_sha256
        or reference.numerical_runtime_fingerprint_sha256
        != authority.numerical_runtime_fingerprint_sha256
        or reference.rng_key_map_sha256 != authority.key_map_sha256
    ):
        raise ValueError(
            "development authority and context-reference provenance differ."
        )


def _require_synthetic_participant(
    value: object,
    *,
    contract: SyntheticV3Contract,
    expected_scope: Literal["development", "unit_test"] | None = None,
    expected_rng_authority: (
        v3.DevelopmentRNGAuthority | _UnitTestRNGAuthority | None
    ) = None,
) -> SyntheticParticipant:
    contract = _require_exact_contract(contract)
    value = _require_synthetic_participant_identity(value)
    if expected_scope is not None and value.generation_scope != expected_scope:
        raise TypeError(
            f"participant requires exact {expected_scope!r} generation scope."
        )
    if expected_rng_authority is not None:
        if value._rng_authority is not expected_rng_authority:
            raise ValueError("participant binds a different RNG authority instance.")
        if (
            value.rng_authority_semantic_binding_sha256
            != expected_rng_authority.semantic_binding_sha256
        ):
            raise ValueError("participant binds a different RNG authority semantic hash.")
    expected_partitions = tuple(
        _canonical_sha256(
            {
                "schema": (
                    "cfeg.metadata-calibration-efficiency-v3.synthetic-partition.v1"
                ),
                "plan_sha256": contract.plan_sha256,
                "family": value.family,
                "participant_index": value.participant_index,
                "block": block_index + 1,
                "signals_sha256": _array_sha256(value.signals[block_index]),
                "true_labels_sha256": _array_sha256(value.true_labels[block_index]),
                "recorded_labels_sha256": _array_sha256(
                    value.recorded_support_labels[block_index]
                ),
            }
        )
        for block_index in range(10)
    )
    if value.partition_sha256s != expected_partitions:
        raise ValueError("synthetic participant arrays differ from partition provenance.")
    return value


def _require_rng_factory(value: object) -> None:
    if not callable(value):
        raise TypeError("rng_factory must be callable.")


def _context_packet_sha256(packet: v3.ContextPacket) -> str:
    return _canonical_sha256(
        {
            "schema": "cfeg.metadata-calibration-efficiency-v3.context-packet.v1",
            "packet_key": packet.packet_key,
            "interface_lookup_key": packet.interface_lookup_key,
            "impedance_kohm_by_channel": list(packet.impedance_kohm_by_channel),
            "channel_availability": list(packet.channel_availability),
        }
    )


def _array_sha256(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    return _canonical_sha256(
        {
            "schema": "cfeg.ndarray.v1",
            "dtype": array.dtype.str,
            "shape": list(array.shape),
            "bytes_sha256": hashlib.sha256(array.tobytes(order="C")).hexdigest(),
        }
    )


def _canonical_bytes(value: Mapping[str, Any]) -> bytes:
    return json.dumps(
        dict(value),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def _canonical_sha256(value: Mapping[str, Any]) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _readonly(value: np.ndarray) -> np.ndarray:
    result = np.array(value, copy=True)
    result.setflags(write=False)
    return result


def _regular_file(value: str | Path, name: str) -> Path:
    path = Path(value)
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"{name} must be an existing regular non-symlink file.")
    return path


def _mapping(value: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    result = value.get(key)
    if not isinstance(result, Mapping):
        raise TypeError(f"{key} must be a mapping.")
    return result


def _positive_exact_int(value: Any, name: str) -> int:
    if type(value) is not int or value <= 0:
        raise ValueError(f"{name} must be a positive exact integer.")
    return value


def _nonnegative_exact_int(value: Any, name: str) -> int:
    if type(value) is not int or value < 0:
        raise ValueError(f"{name} must be a nonnegative exact integer.")
    return value


def _sha256(value: Any, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{name} must be 64 lowercase hexadecimal characters.")
    return value
