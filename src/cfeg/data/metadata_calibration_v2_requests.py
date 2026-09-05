from __future__ import annotations

import hashlib
import hmac
import itertools
import json
import math
import os
import re
import stat
import uuid
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass
from multiprocessing import get_context
from pathlib import Path
from typing import Any, Literal

import h5py
import numpy as np
import pandas as pd
import yaml

from cfeg.baselines.fbcca import apply_filterbank, predict_fbcca, resolve_filterbank_parameters
from cfeg.data.io_hdf5 import HDF5SampleReader
from cfeg.metadata_calibration_v2_terminal import deny_v2_terminal_operational_action
from cfeg.models.metadata_calibration_v2 import (
    PrequentialFoldProvenance,
    TemplateScoreProvenance,
    V2OperatorConfig,
    prequential_gate_decision,
    template_residual_scores_from_subbands,
)

V2_CANDIDATE_ID = "metadata-calibration-efficiency-v2"
V2_PLAN_SCHEMA = "cfeg.metadata-calibration-efficiency-plan.v2"
V2_ALLOCATION_SCHEMA = "cfeg.metadata-calibration-efficiency-v2.external-allocation.v1"
V2_SCORE_REQUEST_SCHEMA = "cfeg.metadata-calibration-v2-score-request.v2"
V2_INVENTORY_RECEIPT_SCHEMA = "cfeg.metadata-calibration-v2-external-inventory.v1"
V2_SCORE_CACHE_SCHEMA = "cfeg.metadata-calibration-v2-fbcca-cache.v1"
V2_SUBBAND_CACHE_SCHEMA = "cfeg.metadata-calibration-v2-subband-cache.v1"
V2_TEMPLATE_CACHE_SCHEMA = "cfeg.metadata-calibration-v2-template-score-cache.v1"
V2_LABEL_JOIN_SCHEMA = "cfeg.metadata-calibration-v2-label-join.v1"

_REPOSITORY = Path(__file__).resolve().parents[3]
DEFAULT_V2_PLAN_PATH = _REPOSITORY / "configs/analysis/metadata_calibration_efficiency_v2.yaml"
DEFAULT_V2_ALLOCATION_PATH = (
    _REPOSITORY / "configs/governance/metadata_calibration_v2_external_allocation.json"
)
DEFAULT_V2_FILTERBANK_PATH = (
    _REPOSITORY / "configs/baselines/fbcca_chen2015_m3_v2_explicit_weights.yaml"
)

V2_FILTERBANK_SHA256 = "b8c1ce4477d980b40f359bc3dc97f3125313191d24401f29a7e2dc473d0d1380"
V2_ALLOCATION_SHA256 = "3722449446183a7d5a4b7006b6a28c38cbcf536bbeac7bc74a07317143077cf4"
V2_N_CLASSES = 40
V2_BUDGETS = (0, 1, 3)
V2_CODEBOOK_HZ = tuple(8.0 + 0.2 * index for index in range(V2_N_CLASSES))
V2_CHANNEL_NAMES = ("PO7", "PO3", "POz", "PO4", "PO8", "O1", "Oz", "O2")
V2_CANONICAL_CHANNEL_IDS = (53, 55, 56, 57, 59, 61, 62, 63)
V2_CANONICAL_CHANNEL_POSITIONS = tuple(value - 1 for value in V2_CANONICAL_CHANNEL_IDS)
V2_SUBBAND_WEIGHTS = (
    1.25,
    0.6704482076268572,
    0.5032785618838642,
    0.42677669529663687,
    0.38374806099528436,
    0.35649051737437876,
    0.33782687899303776,
)

_DATASET_SPECS: dict[str, dict[str, Any]] = {
    "beta_v1": {
        "manifest_dataset_id": "beta",
        "n_subjects": 70,
        "manifest_sha256": "ec4792cbaa418297f9ca883a4b17c60ea33ebae6af89caadf8f8662f57582b19",
        "signals_sha256": "ee6f3b324eb7cf5c8349ac7ae09d51def13b6d12c4968472b4bb70ed43cb4818",
        "excluded_exposed": ("sub001", "sub016"),
        "seed": 20260906,
        "development_count": 23,
        "independent_gate_count": 45,
    },
    "dong2023_v1": {
        "manifest_dataset_id": "dong2023",
        "n_subjects": 59,
        "manifest_sha256": "618e1b16cd8e32c69fe725af43f347b4754981369c1885f6f9a587526b1d26d7",
        "signals_sha256": "c5893ba3ac2fe6a0af4a6ebf6b17a3604db40707aafa786605589b89040cec6e",
        "excluded_exposed": ("sub001",),
        "seed": 20260907,
        "development_count": 19,
        "independent_gate_count": 39,
    },
}

_FORBIDDEN_PATH_PARTS = re.compile(
    r"(?:^|[^a-z0-9])(?:wearable(?:_v3)?|held(?:60)?|source(?:_?39)?|exposed)(?:[^a-z0-9]|$)",
    re.IGNORECASE,
)
_SHA256_RE = re.compile(r"[0-9a-f]{64}")
_SUBJECT_RE = re.compile(r"sub[0-9]{3}")
_QUERY_FORBIDDEN_FIELDS = frozenset(
    {
        "query_label",
        "query_labels",
        "label",
        "labels",
        "target",
        "targets",
        "query_sample_id",
        "query_sample_ids",
        "sample_id",
        "sample_ids",
        "query_h5_index",
        "query_h5_indices",
        "h5_index",
        "h5_indices",
        "trial_id",
        "trial_ids",
        "source_file",
        "source_files",
    }
)
_SELECTION_RECEIPT_FIELDS = frozenset(
    {
        "schema",
        "candidate_id",
        "phase",
        "status",
        "plan_sha256",
        "plan_content_sha256",
        "allocation_sha256",
        "allocation_file_sha256",
        "candidate_grid_sha256",
        "participant_delta_sha256",
        "criterion_order",
        "candidate_summaries",
        "selected_candidate",
        "k1_global_enabled",
        "candidate_and_parameters_immutable",
        "n_candidates",
        "n_participants",
        "independent_gate_authorized",
        "held_access_authorized",
        "completion_receipt_sha256",
    }
)
_SELECTION_CRITERION_ORDER = (
    "reject_any_candidate_failing_datasetwise_k1_mean_ge_0",
    "reject_any_candidate_failing_pooled_k1_lower_CI_gt_minus_0p025",
    "maximize_equal_dataset_mean_eAUC_A_Q_minus_A0",
    "minimize_lambda_max",
    "prefer_score_prototype_shrinkage",
    "minimize_prototype_prior_pseudocount",
)


@dataclass(frozen=True)
class V2CandidateSpec:
    candidate_key: str
    operator: Literal["score_prototype_shrinkage", "filterbank_target_template_residual"]
    lambda_max: float
    prototype_prior_pseudocount: float | None

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class V2ExternalContract:
    plan_sha256: str
    plan_content_sha256: str
    allocation_file_sha256: str
    allocation_sha256: str
    plan: Mapping[str, Any]
    allocation: Mapping[str, Any]
    filterbank: Mapping[str, Any]
    asset_bindings: Mapping[str, tuple[str, str]]

    def subjects(
        self,
        dataset_id: str,
        phase: Literal["development", "independent_gate"],
    ) -> tuple[str, ...]:
        _dataset_id(dataset_id)
        _phase(phase)
        return tuple(str(value) for value in self.allocation["allocation"][dataset_id][phase])

    def asset_hashes(self, dataset_id: str) -> tuple[str, str]:
        resolved = _dataset_id(dataset_id)
        value = self.asset_bindings.get(resolved)
        if (
            not isinstance(value, tuple)
            or len(value) != 2
            or not all(_is_sha256(item) for item in value)
        ):
            raise ValueError("V2 contract has an invalid asset-hash binding.")
        return value


@dataclass(frozen=True)
class DevelopmentSelectionAuthorization:
    candidate: V2CandidateSpec
    receipt: Mapping[str, Any]
    receipt_sha256: str


@dataclass(frozen=True)
class SupportRow:
    block: int
    label: int
    h5_index: int


@dataclass(frozen=True)
class ParticipantPartition:
    subject_id: str
    support_rows: tuple[SupportRow, ...]
    query_h5_indices: tuple[int, ...]
    sfreq: float

    def support_prefix(self, budget: int) -> tuple[SupportRow, ...]:
        resolved = _budget(budget)
        return tuple(row for row in self.support_rows if row.block <= resolved)


@dataclass(frozen=True)
class ValidatedExternalAsset:
    dataset_id: str
    phase: Literal["development", "independent_gate"]
    root: Path
    manifest_path: Path
    signals_path: Path
    manifest_sha256: str
    signals_sha256: str
    preprocessing_sha256: str
    participants: tuple[ParticipantPartition, ...]
    manifest_file_identity: tuple[int, int, int, int]
    signals_file_identity: tuple[int, int, int, int]

    def participant(self, subject_id: str) -> ParticipantPartition:
        matches = tuple(value for value in self.participants if value.subject_id == subject_id)
        if len(matches) != 1:
            raise PermissionError("Participant is outside the validated frozen allocation.")
        return matches[0]


@dataclass(frozen=True)
class FBCCAScoreCache:
    metadata: Mapping[str, Any]
    support_scores: np.ndarray
    support_labels: np.ndarray
    support_blocks: np.ndarray
    query_scores: np.ndarray
    query_tokens: tuple[str, ...]
    path: Path


@dataclass(frozen=True)
class SubbandCache:
    metadata: Mapping[str, Any]
    support_subbands: np.ndarray
    support_labels: np.ndarray
    support_blocks: np.ndarray
    query_subbands: np.ndarray
    query_tokens: tuple[str, ...]
    path: Path


@dataclass(frozen=True)
class TemplateScoreCache:
    metadata: Mapping[str, Any]
    query_class_scores: np.ndarray
    provenance: TemplateScoreProvenance
    path: Path


def load_v2_external_contract(
    plan_path: str | Path = DEFAULT_V2_PLAN_PATH,
    allocation_path: str | Path = DEFAULT_V2_ALLOCATION_PATH,
    filterbank_path: str | Path = DEFAULT_V2_FILTERBANK_PATH,
) -> V2ExternalContract:
    """Load and replay the frozen r6 external-data contract.

    This loader intentionally validates only producer-owned invariants. The
    analysis runner performs a second, independently implemented validation.
    """

    plan_file = _regular_file(plan_path, "V2 plan")
    allocation_file = _regular_file(allocation_path, "V2 allocation")
    filterbank_file = _regular_file(filterbank_path, "V2 filterbank")
    plan = yaml.safe_load(plan_file.read_text(encoding="utf-8")) or {}
    allocation = json.loads(allocation_file.read_text(encoding="utf-8"))
    filterbank = yaml.safe_load(filterbank_file.read_text(encoding="utf-8")) or {}
    if not all(isinstance(value, Mapping) for value in (plan, allocation, filterbank)):
        raise TypeError("V2 plan, allocation, and filterbank must be mappings.")
    if (
        plan.get("schema") != V2_PLAN_SCHEMA
        or plan.get("candidate_id") != V2_CANDIDATE_ID
        or plan.get("decision_id") != "DEC-20260906-001"
        or plan.get("method_revision") != "r6_pre_outcome_explicit_filterbank_weights"
    ):
        raise ValueError("V2 producer plan header drifted from the frozen design.")
    anchor = plan.get("anchor") or {}
    external = plan.get("external_allocation") or {}
    calibration = plan.get("calibration_contract") or {}
    template = (plan.get("support_operators") or {}).get(
        "filterbank_target_template_residual"
    ) or {}
    if (
        anchor.get("filterbank_config")
        != "configs/baselines/fbcca_chen2015_m3_v2_explicit_weights.yaml"
        or anchor.get("filterbank_config_sha256") != V2_FILTERBANK_SHA256
        or anchor.get("score_schema") != "strict_fbcca_bound_filterbank_raw_scores_v1"
        or anchor.get("candidate_codebook_rule") != "all_classes_same_order_for_every_row"
        or calibration.get("external_four_block_support")
        != {1: ["block01"], 3: ["block01", "block02", "block03"]}
        or calibration.get("external_four_block_query") != ["block04"]
        or calibration.get("support_query_disjoint") is not True
        or calibration.get("classwise_mixed_block_sampling") != "forbidden"
        or tuple(template.get("subband_weights") or ()) != V2_SUBBAND_WEIGHTS
    ):
        raise ValueError("V2 producer signal/partition contract drifted.")
    if _sha256_file(filterbank_file) != V2_FILTERBANK_SHA256:
        raise ValueError("V2 filterbank file does not match its frozen SHA-256.")
    _validate_filterbank(filterbank)
    allocation_sha = _validate_allocation(allocation)
    if external.get("allocation_sha256") != allocation_sha:
        raise ValueError("V2 plan does not bind the replayed external allocation.")
    for dataset_id, spec in _DATASET_SPECS.items():
        declared = external.get(dataset_id) or {}
        observed = allocation["allocation"][dataset_id]
        if (
            declared.get("manifest_sha256") != spec["manifest_sha256"]
            or declared.get("signals_sha256") != spec["signals_sha256"]
            or tuple(declared.get("excluded_exposed") or ()) != spec["excluded_exposed"]
            or declared.get("development") != observed["development"]
            or declared.get("independent_gate") != observed["independent_gate"]
        ):
            raise ValueError(f"V2 {dataset_id} asset/allocation binding drifted.")
    return V2ExternalContract(
        plan_sha256=_sha256_file(plan_file),
        plan_content_sha256=_canonical_json_sha256(plan),
        allocation_file_sha256=_sha256_file(allocation_file),
        allocation_sha256=allocation_sha,
        plan=dict(plan),
        allocation=dict(allocation),
        filterbank=dict(filterbank),
        asset_bindings={
            dataset_id: (str(spec["manifest_sha256"]), str(spec["signals_sha256"]))
            for dataset_id, spec in _DATASET_SPECS.items()
        },
    )


def candidate_grid(contract: V2ExternalContract) -> tuple[V2CandidateSpec, ...]:
    support = contract.plan.get("support_operators") or {}
    pseudocounts = tuple(
        float(value)
        for value in (support.get("score_prototype_shrinkage") or {}).get(
            "prototype_prior_pseudocount_grid", ()
        )
    )
    lambdas = tuple(
        float(value) for value in (support.get("fusion") or {}).get("lambda_max_grid", ())
    )
    if pseudocounts != (1.0, 4.0, 16.0) or lambdas != (0.1, 0.2, 0.3):
        raise ValueError("V2 producer candidate grid drifted.")
    values = [
        V2CandidateSpec(
            candidate_key=f"score-prototype-pc{pseudocount:g}-lambda{value:.2f}".replace(".", "p"),
            operator="score_prototype_shrinkage",
            lambda_max=value,
            prototype_prior_pseudocount=pseudocount,
        )
        for value, pseudocount in itertools.product(lambdas, pseudocounts)
    ]
    values.extend(
        V2CandidateSpec(
            candidate_key=f"template-residual-lambda{value:.2f}".replace(".", "p"),
            operator="filterbank_target_template_residual",
            lambda_max=value,
            prototype_prior_pseudocount=None,
        )
        for value in lambdas
    )
    result = tuple(sorted(values, key=lambda value: value.candidate_key))
    if len(result) != 12 or len({value.candidate_key for value in result}) != 12:
        raise RuntimeError("V2 producer did not obtain the exact 12-candidate grid.")
    return result


def validate_external_asset(
    processed_dir: str | Path,
    *,
    dataset_id: str,
    phase: Literal["development", "independent_gate"],
    contract: V2ExternalContract,
    selection_authorization: DevelopmentSelectionAuthorization | None = None,
) -> ValidatedExternalAsset:
    """Validate an exact BETA/Dong asset without reading query labels or HDF5 ``y``.

    Support labels are calibration inputs and are projected separately. Query
    rows are projected without ``label`` or ``stimulus_frequency_hz``. Their
    class-completeness check is deliberately deferred to the post-prediction
    label join.
    """

    resolved_dataset = _dataset_id(dataset_id)
    resolved_phase = _phase(phase)
    if resolved_phase == "development":
        if selection_authorization is not None:
            raise PermissionError("Development inventory must not consume a selection receipt.")
    else:
        if not isinstance(selection_authorization, DevelopmentSelectionAuthorization):
            raise PermissionError(
                "Independent-gate inventory requires typed development authorization."
            )
        verified_authorization = _validate_selection_receipt_mapping(
            selection_authorization.receipt,
            contract=contract,
        )
        if verified_authorization != selection_authorization:
            raise PermissionError("Independent-gate selection authorization drifted.")
    root = _safe_external_path(processed_dir)
    if root.is_symlink() or not root.is_dir():
        raise ValueError("V2 processed asset must be a nonsymlink directory.")
    manifest_path = _regular_file(root / "manifest.parquet", "processed manifest")
    signals_path = _regular_file(root / "signals.h5", "processed signals")
    class_map_path = _regular_file(root / "class_map.json", "processed class map")
    asset_info_path = _regular_file(root / "asset_info.json", "processed asset info")
    expected_manifest, expected_signals = contract.asset_hashes(resolved_dataset)
    observed_manifest = _sha256_file(manifest_path)
    if observed_manifest != expected_manifest:
        raise ValueError("V2 processed manifest SHA-256 differs from the frozen asset.")
    observed_signals = _sha256_file(signals_path)
    if observed_signals != expected_signals:
        raise ValueError("V2 processed signals SHA-256 differs from the frozen asset.")
    _validate_class_map(class_map_path)
    _validate_asset_info(asset_info_path, resolved_dataset)
    _validate_h5_container(signals_path, _DATASET_SPECS[resolved_dataset]["n_subjects"] * 160)

    public_columns = [
        "h5_index",
        "dataset_id",
        "subject_id",
        "session_id",
        "run_id",
        "sfreq_processed",
        "n_channels_used",
        "canonical_channel_ids",
    ]
    public = pd.read_parquet(manifest_path, columns=public_columns)
    _validate_public_manifest(public, resolved_dataset)
    allocated = contract.subjects(resolved_dataset, resolved_phase)
    observed_subjects = tuple(sorted(set(public["subject_id"].astype(str))))
    expected_subjects = tuple(
        f"sub{index:03d}" for index in range(1, _DATASET_SPECS[resolved_dataset]["n_subjects"] + 1)
    )
    if observed_subjects != expected_subjects:
        raise ValueError("V2 processed asset has an unexpected participant inventory.")
    selected = public.loc[public["subject_id"].astype(str).isin(allocated)].copy()
    if set(selected["subject_id"].astype(str)) != set(allocated):
        raise PermissionError("V2 asset is missing a participant in the frozen phase allocation.")
    if set(selected["subject_id"].astype(str)) & set(
        _DATASET_SPECS[resolved_dataset]["excluded_exposed"]
    ):
        raise PermissionError("V2 phase inventory includes an exposed participant.")

    participants: list[ParticipantPartition] = []
    with h5py.File(signals_path, "r") as signals:
        support_y = signals["y"]
        for subject_id in allocated:
            rows = selected.loc[selected["subject_id"].astype(str).eq(subject_id)]
            if len(rows) != 160 or set(rows["run_id"].astype(str)) != {
                "block01",
                "block02",
                "block03",
                "block04",
            }:
                raise ValueError("Every V2 external participant must have exact blocks01-04.")
            counts = rows.groupby("run_id", sort=False).size()
            if not counts.eq(V2_N_CLASSES).all():
                raise ValueError("Every V2 external participant block must contain 40 rows.")
            subject_support = rows.loc[
                rows["run_id"].astype(str).isin(("block01", "block02", "block03")),
                ["h5_index", "run_id"],
            ].copy()
            subject_support["block"] = (
                subject_support["run_id"].astype(str).str.removeprefix("block").astype(int)
            )
            # Only calibration labels are indexed. The query slice of HDF5 y
            # is never materialized by the inventory/prediction capability.
            subject_support["label"] = [
                int(support_y[int(index)]) for index in subject_support["h5_index"]
            ]
            subject_support["label"] = _exact_integer_series(
                subject_support["label"], name="support label"
            )
            for block in (1, 2, 3):
                block_labels = subject_support.loc[subject_support["block"].eq(block), "label"]
                if tuple(sorted(block_labels.tolist())) != tuple(range(V2_N_CLASSES)):
                    raise ValueError("Every V2 support block must contain each class exactly once.")
            subject_support = subject_support.sort_values(
                ["block", "label", "h5_index"], kind="mergesort"
            )
            support_rows = tuple(
                SupportRow(int(row.block), int(row.label), int(row.h5_index))
                for row in subject_support.itertuples(index=False)
            )
            query = rows.loc[rows["run_id"].astype(str).eq("block04")].sort_values(
                "h5_index", kind="mergesort"
            )
            query_indices = tuple(int(value) for value in query["h5_index"])
            if len(query_indices) != V2_N_CLASSES or set(query_indices) & {
                row.h5_index for row in support_rows
            }:
                raise ValueError("V2 support/query partitions are not exact and disjoint.")
            sfreq_values = tuple(sorted(set(rows["sfreq_processed"].astype(float))))
            if sfreq_values != (200.0,):
                raise ValueError(
                    "V2 external participant must use the frozen 200 Hz preprocessing."
                )
            participants.append(
                ParticipantPartition(
                    subject_id=subject_id,
                    support_rows=support_rows,
                    query_h5_indices=query_indices,
                    sfreq=sfreq_values[0],
                )
            )
    preprocessing_sha = _canonical_json_sha256(
        {
            "schema": "cfeg.metadata-calibration-v2-preprocessing-binding.v1",
            "manifest_sha256": observed_manifest,
            "signals_sha256": observed_signals,
            "sfreq": 200.0,
            "channel_names": V2_CHANNEL_NAMES,
            "canonical_channel_ids": V2_CANONICAL_CHANNEL_IDS,
            "channel_positions": V2_CANONICAL_CHANNEL_POSITIONS,
        }
    )
    return ValidatedExternalAsset(
        dataset_id=resolved_dataset,
        phase=resolved_phase,
        root=root,
        manifest_path=manifest_path,
        signals_path=signals_path,
        manifest_sha256=observed_manifest,
        signals_sha256=observed_signals,
        preprocessing_sha256=preprocessing_sha,
        participants=tuple(participants),
        manifest_file_identity=_file_identity(manifest_path),
        signals_file_identity=_file_identity(signals_path),
    )


def build_inventory_receipt(
    asset: ValidatedExternalAsset,
    *,
    contract: V2ExternalContract,
) -> dict[str, Any]:
    expected = contract.subjects(asset.dataset_id, asset.phase)
    observed = tuple(value.subject_id for value in asset.participants)
    if observed != expected:
        raise ValueError("V2 inventory participant order drifted from the allocation.")
    payload: dict[str, Any] = {
        "schema": V2_INVENTORY_RECEIPT_SCHEMA,
        "candidate_id": V2_CANDIDATE_ID,
        "phase": asset.phase,
        "status": "schema_and_inventory_validated_without_query_outcomes",
        "plan_sha256": contract.plan_sha256,
        "allocation_sha256": contract.allocation_sha256,
        "dataset_id": asset.dataset_id,
        "asset_manifest_sha256": asset.manifest_sha256,
        "asset_signals_sha256": asset.signals_sha256,
        "preprocessing_sha256": asset.preprocessing_sha256,
        "participant_ids_sha256": _canonical_json_sha256({"subject_ids": observed}),
        "n_participants": len(observed),
        "n_support_rows": sum(len(value.support_rows) for value in asset.participants),
        "n_query_rows": sum(len(value.query_h5_indices) for value in asset.participants),
        "support_blocks": [1, 2, 3],
        "query_blocks": [4],
        "support_labels_loaded": True,
        "query_labels_loaded": False,
        "query_hdf5_y_loaded": False,
        "v1_source39_accessed": False,
        "wearable_held60_accessed": False,
        "held_access_authorized": False,
    }
    return _with_hash(payload, "inventory_receipt_sha256")


def deterministic_query_tokens(
    secret: bytes,
    *,
    asset: ValidatedExternalAsset,
    participant: ParticipantPartition,
) -> tuple[str, ...]:
    if not isinstance(secret, bytes) or len(secret) < 32:
        raise ValueError("V2 query-token secret must contain at least 32 bytes.")
    if participant.subject_id not in {value.subject_id for value in asset.participants}:
        raise PermissionError("Cannot tokenize a participant outside the validated inventory.")
    tokens: list[str] = []
    for ordinal, h5_index in enumerate(participant.query_h5_indices):
        message = _canonical_json_bytes(
            {
                "domain": "cfeg-v2-external-query-token-v1",
                "manifest_sha256": asset.manifest_sha256,
                "dataset_id": asset.dataset_id,
                "phase": asset.phase,
                "subject_id": participant.subject_id,
                "query_ordinal": ordinal,
                "internal_h5_index": h5_index,
            }
        )
        tokens.append("qv2_" + hmac.new(secret, message, hashlib.sha256).hexdigest())
    if len(tokens) != V2_N_CLASSES or len(set(tokens)) != V2_N_CLASSES:
        raise RuntimeError("V2 query-token derivation did not yield 40 unique opaque tokens.")
    return tuple(tokens)


def build_fbcca_score_cache(
    asset: ValidatedExternalAsset,
    *,
    participant: ParticipantPartition,
    contract: V2ExternalContract,
    token_secret: bytes,
    cache_dir: str | Path,
    workers: int = 1,
) -> FBCCAScoreCache:
    """Produce or verify a content-addressed strict-FBCCA cache.

    Query signals are loaded through ``HDF5SampleReader.read_unlabeled``. This function
    neither accepts nor returns query labels, raw sample IDs, or query HDF5
    indices.
    """

    _require_participant(asset, participant)
    _assert_asset_unchanged(asset)
    resolved_workers = _workers(workers)
    query_tokens = deterministic_query_tokens(token_secret, asset=asset, participant=participant)
    support_partition_sha = _support_partition_sha(asset, participant.support_rows)
    query_partition_sha = _query_partition_sha(asset, participant, query_tokens)
    metadata = {
        "schema": V2_SCORE_CACHE_SCHEMA,
        "plan_sha256": contract.plan_sha256,
        "allocation_sha256": contract.allocation_sha256,
        "dataset_id": asset.dataset_id,
        "phase": asset.phase,
        "subject_id": participant.subject_id,
        "asset_manifest_sha256": asset.manifest_sha256,
        "asset_signals_sha256": asset.signals_sha256,
        "preprocessing_sha256": asset.preprocessing_sha256,
        "filterbank_sha256": V2_FILTERBANK_SHA256,
        "score_schema": "strict_fbcca_bound_filterbank_raw_scores_v1",
        "codebook_hz": V2_CODEBOOK_HZ,
        "channel_names": V2_CHANNEL_NAMES,
        "support_partition_sha256": support_partition_sha,
        "query_partition_sha256": query_partition_sha,
        "query_token_commitment_sha256": _canonical_json_sha256({"query_tokens": query_tokens}),
        "query_labels_loaded": False,
        "hdf5_y_loaded_for_query": False,
    }
    cache_key = _canonical_json_sha256(metadata)
    cache_path = _cache_path(cache_dir, "fbcca", cache_key)

    def compute() -> Mapping[str, np.ndarray]:
        support_signals, query_signals = _read_partition_signals(asset, participant)
        combined = np.concatenate((support_signals, query_signals), axis=0)
        scores = _map_fbcca(
            combined,
            sfreq=participant.sfreq,
            filterbank=dict(contract.filterbank),
            workers=resolved_workers,
        )
        if scores.shape != (160, V2_N_CLASSES) or not np.isfinite(scores).all():
            raise RuntimeError("Strict-FBCCA producer returned an invalid full score matrix.")
        return {
            "support_scores": scores[:120],
            "support_labels": np.asarray(
                [row.label for row in participant.support_rows], dtype=np.int64
            ),
            "support_blocks": np.asarray(
                [row.block for row in participant.support_rows], dtype=np.int64
            ),
            "query_scores": scores[120:],
            "query_tokens": np.asarray(query_tokens, dtype="U68"),
        }

    values = _load_or_create_npz(cache_path, metadata=metadata, compute=compute)
    _assert_asset_unchanged(asset)
    return _score_cache_from_values(cache_path, metadata, values)


def build_subband_cache(
    asset: ValidatedExternalAsset,
    *,
    participant: ParticipantPartition,
    contract: V2ExternalContract,
    token_secret: bytes,
    cache_dir: str | Path,
) -> SubbandCache:
    """Cache exact seven-band P2 inputs without opening query outcomes."""

    _require_participant(asset, participant)
    _assert_asset_unchanged(asset)
    query_tokens = deterministic_query_tokens(token_secret, asset=asset, participant=participant)
    support_partition_sha = _support_partition_sha(asset, participant.support_rows)
    query_partition_sha = _query_partition_sha(asset, participant, query_tokens)
    metadata = {
        "schema": V2_SUBBAND_CACHE_SCHEMA,
        "plan_sha256": contract.plan_sha256,
        "allocation_sha256": contract.allocation_sha256,
        "dataset_id": asset.dataset_id,
        "phase": asset.phase,
        "subject_id": participant.subject_id,
        "asset_manifest_sha256": asset.manifest_sha256,
        "asset_signals_sha256": asset.signals_sha256,
        "preprocessing_sha256": asset.preprocessing_sha256,
        "filterbank_sha256": V2_FILTERBANK_SHA256,
        "subband_weights": V2_SUBBAND_WEIGHTS,
        "subband_dtype": "float64",
        "support_partition_sha256": support_partition_sha,
        "query_partition_sha256": query_partition_sha,
        "query_token_commitment_sha256": _canonical_json_sha256({"query_tokens": query_tokens}),
        "query_labels_loaded": False,
        "hdf5_y_loaded_for_query": False,
    }
    cache_key = _canonical_json_sha256(metadata)
    cache_path = _cache_path(cache_dir, "subbands", cache_key)

    def compute() -> Mapping[str, np.ndarray]:
        support_signals, query_signals = _read_partition_signals(asset, participant)
        support_subbands, support_parameters = apply_filterbank(
            support_signals, sfreq=participant.sfreq, filterbank=contract.filterbank
        )
        query_subbands, query_parameters = apply_filterbank(
            query_signals, sfreq=participant.sfreq, filterbank=contract.filterbank
        )
        if support_parameters != query_parameters:
            raise RuntimeError("P2 support/query filterbank parameters unexpectedly differ.")
        if tuple(float(value) for value in support_parameters["weights"]) != V2_SUBBAND_WEIGHTS:
            raise RuntimeError("P2 runtime subband weights differ from frozen r6 values.")
        return {
            "support_subbands": support_subbands.astype(np.float64, copy=False),
            "support_labels": np.asarray(
                [row.label for row in participant.support_rows], dtype=np.int64
            ),
            "support_blocks": np.asarray(
                [row.block for row in participant.support_rows], dtype=np.int64
            ),
            "query_subbands": query_subbands.astype(np.float64, copy=False),
            "query_tokens": np.asarray(query_tokens, dtype="U68"),
        }

    values = _load_or_create_npz(cache_path, metadata=metadata, compute=compute)
    _assert_asset_unchanged(asset)
    return _subband_cache_from_values(cache_path, metadata, values)


def build_template_score_cache(
    subbands: SubbandCache,
    *,
    budget: Literal[1, 3],
    cache_dir: str | Path,
) -> TemplateScoreCache:
    """Cache P2 query-class scores for one exact support prefix."""

    resolved_budget = _budget(budget)
    if resolved_budget == 0:
        raise ValueError("P2 has no support score at k=0.")
    _verify_subband_cache(subbands)
    keep = subbands.support_blocks <= resolved_budget
    support = subbands.support_subbands[:, keep]
    labels = subbands.support_labels[keep]
    expected = resolved_budget * V2_N_CLASSES
    if support.shape[1] != expected or labels.shape != (expected,):
        raise ValueError("P2 subband cache lacks the exact support prefix.")
    support_partition_sha = _template_support_partition_sha(
        str(subbands.metadata["support_partition_sha256"]),
        budget=resolved_budget,
        labels=labels,
    )
    provenance = TemplateScoreProvenance(
        filterbank_sha256=V2_FILTERBANK_SHA256,
        preprocessing_sha256=str(subbands.metadata["preprocessing_sha256"]),
        query_partition_sha256=str(subbands.metadata["query_partition_sha256"]),
        support_partition_sha256=support_partition_sha,
    )
    metadata = {
        "schema": V2_TEMPLATE_CACHE_SCHEMA,
        "parent_cache_sha256": _sha256_file(subbands.path),
        "dataset_id": subbands.metadata["dataset_id"],
        "phase": subbands.metadata["phase"],
        "subject_id": subbands.metadata["subject_id"],
        "budget": resolved_budget,
        "subband_weights": V2_SUBBAND_WEIGHTS,
        "template_score_provenance": asdict(provenance),
        "query_token_commitment_sha256": subbands.metadata["query_token_commitment_sha256"],
        "query_labels_loaded": False,
    }
    cache_key = _canonical_json_sha256(metadata)
    cache_path = _cache_path(cache_dir, "template_scores", cache_key)

    def compute() -> Mapping[str, np.ndarray]:
        scores = template_residual_scores_from_subbands(
            subbands.query_subbands,
            support,
            labels,
            np.asarray(V2_SUBBAND_WEIGHTS, dtype=np.float64),
        )
        return {"query_class_scores": scores}

    values = _load_or_create_npz(cache_path, metadata=metadata, compute=compute)
    scores = np.asarray(values.get("query_class_scores"), dtype=np.float64)
    if scores.shape != (V2_N_CLASSES, V2_N_CLASSES) or not np.isfinite(scores).all():
        raise ValueError("P2 template score cache has an invalid matrix.")
    scores.setflags(write=False)
    return TemplateScoreCache(metadata, scores, provenance, cache_path)


def build_external_score_request(
    *,
    score_cache: FBCCAScoreCache,
    candidate: V2CandidateSpec,
    budget: Literal[0, 1, 3],
    contract: V2ExternalContract,
    gate_enabled: bool | None,
    template_cache: TemplateScoreCache | None = None,
    selection_authorization: DevelopmentSelectionAuthorization | None = None,
) -> dict[str, Any]:
    """Build the runner's exact JSON adapter without any query identity/outcome."""

    deny_v2_terminal_operational_action("beta_dong_score_request_or_prediction")
    resolved_budget = _budget(budget)
    score_cache = _verify_score_cache_binding(score_cache, contract=contract)
    frozen = {value.candidate_key: value for value in candidate_grid(contract)}
    if frozen.get(candidate.candidate_key) != candidate:
        raise ValueError("V2 request candidate is outside the frozen grid.")
    phase = str(score_cache.metadata["phase"])
    if phase == "development":
        if selection_authorization is not None:
            raise PermissionError("Development requests must not consume a selection receipt.")
        selection_receipt: dict[str, Any] | None = None
    elif phase == "independent_gate":
        if not isinstance(selection_authorization, DevelopmentSelectionAuthorization):
            raise PermissionError("Independent request requires typed development authorization.")
        verified = _validate_selection_receipt_mapping(
            selection_authorization.receipt,
            contract=contract,
        )
        if (
            verified.candidate != candidate
            or verified.receipt_sha256 != selection_authorization.receipt_sha256
        ):
            raise PermissionError("Independent request candidate/receipt capability mismatched.")
        selection_receipt = dict(verified.receipt)
    else:  # Metadata binding validation above should already have rejected this.
        raise PermissionError("V2 request cache has an unknown phase.")
    if type(gate_enabled) is not bool and gate_enabled is not None:
        raise TypeError("gate_enabled must be an exact bool or None.")
    support_authorized = resolved_budget > 0 and gate_enabled is True
    support_keep = score_cache.support_blocks <= resolved_budget
    support_scores = score_cache.support_scores[support_keep]
    support_labels = score_cache.support_labels[support_keep]
    if support_authorized and (
        support_scores.shape != (resolved_budget * V2_N_CLASSES, V2_N_CLASSES)
        or tuple(sorted(support_labels.tolist())) != tuple(range(V2_N_CLASSES)) * resolved_budget
    ):
        # The count check below is authoritative; the tuple comparison catches
        # accidental ordering drift only after stable sorting.
        counts = np.bincount(support_labels, minlength=V2_N_CLASSES)
        if not np.array_equal(counts, np.full(V2_N_CLASSES, resolved_budget)):
            raise ValueError("V2 request support prefix is not class complete.")
    if not support_authorized:
        support_scores_value = None
        support_labels_value = None
    else:
        support_scores_value = support_scores.tolist()
        support_labels_value = support_labels.astype(int).tolist()
    template_scores: list[list[float]] | None = None
    template_provenance: dict[str, Any] | None = None
    if candidate.operator == "filterbank_target_template_residual" and support_authorized:
        if template_cache is None or template_cache.metadata.get("budget") != resolved_budget:
            raise ValueError("Active P2 request requires its exact-budget template score cache.")
        if template_cache.metadata.get("query_token_commitment_sha256") != _canonical_json_sha256(
            {"query_tokens": score_cache.query_tokens}
        ):
            raise ValueError("P2 and FBCCA caches bind different query partitions.")
        _verify_template_cache(template_cache)
        expected_template_support = _template_support_partition_sha(
            str(score_cache.metadata["support_partition_sha256"]),
            budget=resolved_budget,
            labels=support_labels,
        )
        if (
            template_cache.metadata.get("dataset_id") != score_cache.metadata["dataset_id"]
            or template_cache.metadata.get("phase") != score_cache.metadata["phase"]
            or template_cache.metadata.get("subject_id") != score_cache.metadata["subject_id"]
            or template_cache.provenance.preprocessing_sha256
            != score_cache.metadata["preprocessing_sha256"]
            or template_cache.provenance.query_partition_sha256
            != score_cache.metadata["query_partition_sha256"]
            or template_cache.provenance.support_partition_sha256 != expected_template_support
        ):
            raise ValueError("P2 template cache differs from the exact FBCCA request partition.")
        template_scores = template_cache.query_class_scores.tolist()
        template_provenance = asdict(template_cache.provenance)
    elif template_cache is not None:
        raise ValueError("P1 or fallback requests must not receive a P2 cache.")
    request: dict[str, Any] = {
        "schema": V2_SCORE_REQUEST_SCHEMA,
        "candidate_id": V2_CANDIDATE_ID,
        "plan_sha256": contract.plan_sha256,
        "allocation_sha256": contract.allocation_sha256,
        "asset_manifest_sha256": score_cache.metadata["asset_manifest_sha256"],
        "asset_signals_sha256": score_cache.metadata["asset_signals_sha256"],
        "dataset_id": score_cache.metadata["dataset_id"],
        "cohort": score_cache.metadata["phase"],
        "candidate": candidate.as_dict(),
        "variant": "A_Q",
        "subject_id": score_cache.metadata["subject_id"],
        "budget": resolved_budget,
        "query_tokens": list(score_cache.query_tokens),
        "query_fbcca_scores": score_cache.query_scores.tolist(),
        "support_fbcca_scores": support_scores_value,
        "support_labels": support_labels_value,
        "base_probabilities": None,
        "gate_enabled": gate_enabled,
        "template_query_class_scores": template_scores,
        "template_query_class_probabilities": None,
        "template_score_provenance": template_provenance,
        "template_query_subbands": None,
        "template_support_subbands": None,
        "template_subband_weights": None,
        "query_interfaces": None,
        "support_interfaces": None,
        "query_impedance_kohm": None,
        "support_impedance_kohm": None,
        "relative_context_pairing_sha256": None,
        "development_selection_receipt": selection_receipt,
    }
    _assert_query_blind_request(request)
    return request


def validate_independent_selection_receipt(
    receipt_path: str | Path,
    *,
    contract: V2ExternalContract,
) -> DevelopmentSelectionAuthorization:
    """Require one immutable development selection before opening gate inventory."""

    path = _regular_file(receipt_path, "development selection receipt")
    if stat.S_IMODE(path.stat().st_mode) & 0o222:
        raise PermissionError("Independent gate requires a read-only selection receipt.")
    receipt = json.loads(path.read_text(encoding="utf-8"))
    return _validate_selection_receipt_mapping(receipt, contract=contract)


def _validate_selection_receipt_mapping(
    receipt: Mapping[str, Any],
    *,
    contract: V2ExternalContract,
) -> DevelopmentSelectionAuthorization:
    if not isinstance(receipt, Mapping) or set(receipt) != _SELECTION_RECEIPT_FIELDS:
        raise ValueError("Development selection receipt has a wrong exact schema.")
    claimed = receipt.get("completion_receipt_sha256")
    payload = {key: value for key, value in receipt.items() if key != "completion_receipt_sha256"}
    if not _is_sha256(claimed) or claimed != _canonical_json_sha256(payload):
        raise ValueError("Development selection completion hash is invalid.")
    selected = receipt.get("selected_candidate")
    if (
        receipt.get("schema") != "cfeg.metadata-calibration-v2-development-selection-completion.v1"
        or receipt.get("candidate_id") != V2_CANDIDATE_ID
        or receipt.get("phase") != "development_selection"
        or receipt.get("status") != "selected_candidate_frozen"
        or receipt.get("plan_sha256") != contract.plan_sha256
        or receipt.get("plan_content_sha256") != contract.plan_content_sha256
        or receipt.get("allocation_sha256") != contract.allocation_sha256
        or receipt.get("allocation_file_sha256") != contract.allocation_file_sha256
        or receipt.get("criterion_order") != list(_SELECTION_CRITERION_ORDER)
        or receipt.get("n_candidates") != 12
        or receipt.get("n_participants") != 42
        or receipt.get("candidate_and_parameters_immutable") is not True
        or receipt.get("k1_global_enabled") is not True
        or receipt.get("independent_gate_authorized") is not True
        or receipt.get("held_access_authorized") is not False
        or not isinstance(selected, Mapping)
    ):
        raise PermissionError("Development selection does not authorize the independent gate.")
    candidates = {value.candidate_key: value for value in candidate_grid(contract)}
    expected_grid_sha256 = _canonical_json_sha256(
        {"candidates": [value.as_dict() for value in candidates.values()]}
    )
    if receipt.get("candidate_grid_sha256") != expected_grid_sha256:
        raise ValueError("Development receipt candidate-grid SHA-256 drifted.")
    candidate = candidates.get(str(selected.get("candidate_key")))
    if candidate is None or candidate.as_dict() != dict(selected):
        raise ValueError("Development receipt selected a candidate outside the frozen grid.")
    return DevelopmentSelectionAuthorization(candidate, dict(receipt), str(claimed))


def resolve_phase_candidates(
    *,
    phase: Literal["development", "independent_gate"],
    contract: V2ExternalContract,
    selection_receipt: Mapping[str, Any] | str | Path | None = None,
) -> tuple[V2CandidateSpec, ...]:
    resolved = _phase(phase)
    if resolved == "development":
        if selection_receipt is not None:
            raise PermissionError("Development must not consume a selection receipt.")
        return candidate_grid(contract)
    if selection_receipt is None:
        raise PermissionError("Independent gate requires the immutable development receipt.")
    if isinstance(selection_receipt, Mapping):
        raise PermissionError("Independent gate requires an immutable receipt file, not a mapping.")
    authorization = validate_independent_selection_receipt(selection_receipt, contract=contract)
    return (authorization.candidate,)


def evaluate_query_labels_after_predictions(
    score_receipts: Sequence[Mapping[str, Any]],
    *,
    asset: ValidatedExternalAsset,
    participant: ParticipantPartition,
    token_secret: bytes,
) -> dict[str, Any]:
    """Join query labels only after complete immutable prediction receipts exist.

    The returned label rows are for the privileged evaluator. They must never
    be routed back to :func:`build_external_score_request`.
    """

    deny_v2_terminal_operational_action("beta_dong_query_label_join")
    if not score_receipts:
        raise ValueError("Label join requires at least one completed score receipt.")
    expected_tokens = deterministic_query_tokens(token_secret, asset=asset, participant=participant)
    for receipt in score_receipts:
        if (
            receipt.get("status") != "complete_query_outcome_free"
            or receipt.get("dataset_id") != asset.dataset_id
            or receipt.get("cohort") != asset.phase
            or receipt.get("subject_id") != participant.subject_id
            or tuple(receipt.get("query_tokens") or ()) != expected_tokens
            or receipt.get("query_outcomes_loaded") is not False
            or receipt.get("held_access_authorized") is not False
            or not _is_sha256(receipt.get("completion_receipt_sha256"))
        ):
            raise PermissionError("Label join received an incomplete or mismatched score receipt.")
        payload = {
            key: value for key, value in receipt.items() if key != "completion_receipt_sha256"
        }
        if receipt["completion_receipt_sha256"] != _canonical_json_sha256(payload):
            raise ValueError("Label join received a score receipt with an invalid hash.")

    labels = pd.read_parquet(
        asset.manifest_path,
        columns=["h5_index", "subject_id", "run_id", "label"],
        filters=[("run_id", "==", "block04")],
    )
    labels = labels.loc[labels["subject_id"].astype(str).eq(participant.subject_id)].sort_values(
        "h5_index", kind="mergesort"
    )
    if tuple(int(value) for value in labels["h5_index"]) != participant.query_h5_indices:
        raise ValueError("Privileged query-label rows do not match the sealed query partition.")
    values = _exact_integer_series(labels["label"], name="query label").to_numpy(np.int64)
    if tuple(sorted(values.tolist())) != tuple(range(V2_N_CLASSES)):
        raise ValueError("Query block is not exactly class complete.")
    rows = [
        {"query_token": token, "label": int(label)}
        for token, label in zip(expected_tokens, values, strict=True)
    ]
    payload = {
        "schema": V2_LABEL_JOIN_SCHEMA,
        "status": "labels_joined_after_predictions",
        "dataset_id": asset.dataset_id,
        "phase": asset.phase,
        "subject_id": participant.subject_id,
        "asset_manifest_sha256": asset.manifest_sha256,
        "score_receipt_sha256": sorted(
            str(value["completion_receipt_sha256"]) for value in score_receipts
        ),
        "rows": rows,
        "query_labels_loaded_only_after_predictions": True,
        "held_access_authorized": False,
    }
    return _with_hash(payload, "label_join_receipt_sha256")


def write_json_exclusive(path: str | Path, value: Mapping[str, Any]) -> Path:
    """Atomically publish a read-only JSON request/receipt without replacement."""

    target = _safe_external_path(path)
    if os.path.lexists(target):
        raise FileExistsError(f"Refusing to replace existing V2 artifact: {target}")
    target.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    if target.parent.is_symlink() or not target.parent.is_dir():
        raise ValueError("V2 artifact parent must be a real directory.")
    serialized = (
        json.dumps(dict(value), sort_keys=True, indent=2, ensure_ascii=False, allow_nan=False)
        + "\n"
    )
    staging = target.parent / f".{target.name}.staging-{uuid.uuid4().hex}"
    descriptor = os.open(staging, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o400)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            descriptor = -1
            handle.write(serialized)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(staging, target)
        _fsync_directory(target.parent)
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        if os.path.lexists(staging):
            staging.unlink()
            _fsync_directory(target.parent)
    return target


def derive_prequential_gate(
    *args: Any,
    **kwargs: Any,
) -> bool:
    """Thin integration point for the frozen r5 prefix-depth operator.

    r4 cannot express the two-support-block prefix needed for the second k=3
    fold. The root integration owner is freezing r5 before outcomes. Keeping
    this adapter explicit prevents accidentally approximating the fold with a
    final-budget call. Until r5 is integrated, it fails closed.
    """

    try:
        from cfeg.models.metadata_calibration_v2 import apply_v2_prequential_operator
    except ImportError as error:  # pragma: no cover - removed when r5 lands.
        raise RuntimeError(
            "V2 r5 prequential prefix-depth API is required before k=3 requests."
        ) from error
    return _derive_prequential_gate_r5(apply_v2_prequential_operator, *args, **kwargs)


def _derive_prequential_gate_r5(
    prequential_operator: Callable[..., Any],
    *,
    score_cache: FBCCAScoreCache,
    candidate: V2CandidateSpec,
    contract: V2ExternalContract,
    template_subbands: SubbandCache | None = None,
) -> bool:
    """Run block2/block3 prefix predictions; isolated for r5 API adaptation."""

    score_cache = _verify_score_cache_binding(score_cache, contract=contract)
    frozen = {value.candidate_key: value for value in candidate_grid(contract)}
    if frozen.get(candidate.candidate_key) != candidate:
        raise ValueError("Prequential candidate is outside the frozen V2 grid.")
    if candidate.operator == "filterbank_target_template_residual":
        if template_subbands is None:
            raise ValueError("P2 prequential gate requires its subband cache.")
        _verify_subband_cache(template_subbands)
        _assert_subband_score_cache_binding(template_subbands, score_cache)
    elif template_subbands is not None:
        raise ValueError("P1 prequential gate must not receive a P2 subband cache.")
    base_rows: list[np.ndarray] = []
    candidate_rows: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    blocks: list[np.ndarray] = []
    provenance: list[PrequentialFoldProvenance] = []
    config = _operator_config(contract, candidate)
    for evaluation_block in (2, 3):
        support_depth = evaluation_block - 1
        support_keep = score_cache.support_blocks <= support_depth
        evaluation_keep = score_cache.support_blocks == evaluation_block
        block_sha256s = tuple(
            _prequential_block_sha256(score_cache, block) for block in range(1, evaluation_block)
        )
        evaluation_sha256 = _prequential_block_sha256(score_cache, evaluation_block)
        fold = PrequentialFoldProvenance(
            evaluation_block=evaluation_block,
            fit_blocks=tuple(range(1, evaluation_block)),
            fit_block_partition_sha256s=block_sha256s,
            evaluation_partition_sha256=evaluation_sha256,
        )
        operator_kwargs: dict[str, Any] = {
            "evaluation_fbcca_scores": score_cache.support_scores[evaluation_keep],
            "final_budget": 3,
            "operator": candidate.operator,
            "variant": "A_Q",
            "config": config,
            "support_fbcca_scores": score_cache.support_scores[support_keep],
            "support_labels": score_cache.support_labels[support_keep],
            "fold_provenance": fold,
        }
        if candidate.operator == "filterbank_target_template_residual":
            assert template_subbands is not None
            operator_kwargs.update(
                {
                    "template_query_subbands": template_subbands.support_subbands[
                        :, evaluation_keep
                    ],
                    "template_support_subbands": template_subbands.support_subbands[
                        :, support_keep
                    ],
                    "template_subband_weights": np.asarray(V2_SUBBAND_WEIGHTS, dtype=np.float64),
                }
            )
        output = prequential_operator(**operator_kwargs)
        base_rows.append(np.asarray(output.base_probabilities, dtype=np.float64))
        candidate_rows.append(np.asarray(output.fused_probabilities, dtype=np.float64))
        labels.append(score_cache.support_labels[evaluation_keep])
        blocks.append(np.full(V2_N_CLASSES, evaluation_block, dtype=np.int64))
        provenance.append(fold)
    decision = prequential_gate_decision(
        np.concatenate(base_rows),
        np.concatenate(candidate_rows),
        np.concatenate(labels),
        np.concatenate(blocks),
        budget=3,
        fold_provenance=provenance,
    )
    return bool(decision.enabled)


def _prequential_block_sha256(score_cache: FBCCAScoreCache, block: int) -> str:
    keep = score_cache.support_blocks == block
    scores = score_cache.support_scores[keep]
    labels = score_cache.support_labels[keep]
    if scores.shape != (V2_N_CLASSES, V2_N_CLASSES) or tuple(sorted(labels.tolist())) != tuple(
        range(V2_N_CLASSES)
    ):
        raise ValueError("Prequential block partition is not exactly class complete.")
    return _canonical_json_sha256(
        {
            "domain": "cfeg-v2-prequential-block-partition-v1",
            "parent_support_partition_sha256": score_cache.metadata["support_partition_sha256"],
            "block": block,
            "labels": labels.tolist(),
            "fbcca_scores_sha256": _array_mapping_sha256({"scores": scores}),
        }
    )


def _operator_config(
    contract: V2ExternalContract,
    candidate: V2CandidateSpec,
) -> V2OperatorConfig:
    support = contract.plan["support_operators"]
    return V2OperatorConfig(
        score_epsilon=float(contract.plan["score_normalization"]["epsilon"]),
        ideal_prototype_smoothing=float(
            support["score_prototype_shrinkage"]["ideal_prototype_smoothing"]
        ),
        prototype_prior_pseudocount=(
            4.0
            if candidate.prototype_prior_pseudocount is None
            else candidate.prototype_prior_pseudocount
        ),
        lambda_max=candidate.lambda_max,
        different_interface_affinity=float(
            contract.plan["relative_context"]["interface_affinity"]["different"]
        ),
        entropy_scaling=True,
    )


def _score_cache_from_values(
    path: Path,
    metadata: Mapping[str, Any],
    values: Mapping[str, np.ndarray],
) -> FBCCAScoreCache:
    support_scores = np.asarray(values.get("support_scores"), dtype=np.float64)
    support_labels = _exact_integer_array(values.get("support_labels"), name="support labels")
    support_blocks = _exact_integer_array(values.get("support_blocks"), name="support blocks")
    query_scores = np.asarray(values.get("query_scores"), dtype=np.float64)
    raw_tokens = np.asarray(values.get("query_tokens"))
    query_tokens = (
        tuple(str(value) for value in raw_tokens)
        if raw_tokens.shape == (V2_N_CLASSES,) and raw_tokens.dtype.kind in {"U", "S"}
        else ()
    )
    if (
        support_scores.shape != (120, V2_N_CLASSES)
        or support_labels.shape != (120,)
        or support_blocks.shape != (120,)
        or query_scores.shape != (V2_N_CLASSES, V2_N_CLASSES)
        or len(query_tokens) != V2_N_CLASSES
        or len(set(query_tokens)) != V2_N_CLASSES
        or any(re.fullmatch(r"qv2_[0-9a-f]{64}", value) is None for value in query_tokens)
        or not np.isfinite(support_scores).all()
        or not np.isfinite(query_scores).all()
        or (support_labels < 0).any()
        or (support_labels >= V2_N_CLASSES).any()
        or not np.array_equal(
            np.bincount(support_labels, minlength=V2_N_CLASSES),
            np.full(V2_N_CLASSES, 3),
        )
        or tuple(sorted(set(support_blocks.tolist()))) != (1, 2, 3)
        or any(
            not np.array_equal(
                np.bincount(support_labels[support_blocks == block], minlength=V2_N_CLASSES),
                np.ones(V2_N_CLASSES, dtype=np.int64),
            )
            for block in (1, 2, 3)
        )
    ):
        raise ValueError("Strict-FBCCA cache content is incomplete or malformed.")
    if metadata.get("query_token_commitment_sha256") != _canonical_json_sha256(
        {"query_tokens": query_tokens}
    ):
        raise ValueError("Strict-FBCCA cache query-token commitment drifted.")
    for value in (
        support_scores,
        support_labels,
        support_blocks,
        query_scores,
    ):
        value.setflags(write=False)
    return FBCCAScoreCache(
        metadata,
        support_scores,
        support_labels,
        support_blocks,
        query_scores,
        query_tokens,
        path,
    )


def _subband_cache_from_values(
    path: Path,
    metadata: Mapping[str, Any],
    values: Mapping[str, np.ndarray],
) -> SubbandCache:
    support = np.asarray(values.get("support_subbands"), dtype=np.float64)
    labels = _exact_integer_array(values.get("support_labels"), name="support labels")
    blocks = _exact_integer_array(values.get("support_blocks"), name="support blocks")
    query = np.asarray(values.get("query_subbands"), dtype=np.float64)
    raw_tokens = np.asarray(values.get("query_tokens"))
    tokens = (
        tuple(str(value) for value in raw_tokens)
        if raw_tokens.shape == (V2_N_CLASSES,) and raw_tokens.dtype.kind in {"U", "S"}
        else ()
    )
    if (
        support.shape != (7, 120, 8, 400)
        or query.shape != (7, 40, 8, 400)
        or labels.shape != (120,)
        or blocks.shape != (120,)
        or len(tokens) != 40
        or len(set(tokens)) != V2_N_CLASSES
        or any(re.fullmatch(r"qv2_[0-9a-f]{64}", value) is None for value in tokens)
        or not np.isfinite(support).all()
        or not np.isfinite(query).all()
        or (labels < 0).any()
        or (labels >= V2_N_CLASSES).any()
        or tuple(sorted(set(blocks.tolist()))) != (1, 2, 3)
        or any(
            not np.array_equal(
                np.bincount(labels[blocks == block], minlength=V2_N_CLASSES),
                np.ones(V2_N_CLASSES, dtype=np.int64),
            )
            for block in (1, 2, 3)
        )
    ):
        raise ValueError("P2 subband cache content is incomplete or malformed.")
    if metadata.get("query_token_commitment_sha256") != _canonical_json_sha256(
        {"query_tokens": tokens}
    ):
        raise ValueError("P2 subband query-token commitment drifted.")
    for value in (support, labels, blocks, query):
        value.setflags(write=False)
    return SubbandCache(metadata, support, labels, blocks, query, tokens, path)


def _verify_score_cache_binding(
    cache: FBCCAScoreCache,
    *,
    contract: V2ExternalContract,
) -> FBCCAScoreCache:
    metadata = cache.metadata
    dataset_id = _dataset_id(str(metadata.get("dataset_id")))
    phase = _phase(str(metadata.get("phase")))
    subject_id = str(metadata.get("subject_id"))
    manifest_sha, signals_sha = contract.asset_hashes(dataset_id)
    if (
        metadata.get("schema") != V2_SCORE_CACHE_SCHEMA
        or metadata.get("plan_sha256") != contract.plan_sha256
        or metadata.get("allocation_sha256") != contract.allocation_sha256
        or subject_id not in contract.subjects(dataset_id, phase)
        or metadata.get("asset_manifest_sha256") != manifest_sha
        or metadata.get("asset_signals_sha256") != signals_sha
        or metadata.get("filterbank_sha256") != V2_FILTERBANK_SHA256
        or metadata.get("score_schema") != "strict_fbcca_bound_filterbank_raw_scores_v1"
        or tuple(metadata.get("codebook_hz") or ()) != V2_CODEBOOK_HZ
        or tuple(metadata.get("channel_names") or ()) != V2_CHANNEL_NAMES
        or metadata.get("query_labels_loaded") is not False
        or metadata.get("hdf5_y_loaded_for_query") is not False
        or not all(
            _is_sha256(metadata.get(field))
            for field in (
                "preprocessing_sha256",
                "support_partition_sha256",
                "query_partition_sha256",
                "query_token_commitment_sha256",
            )
        )
    ):
        raise ValueError("Strict-FBCCA cache contract binding drifted.")
    verified = _load_npz(cache.path, expected_metadata=metadata)
    loaded = _score_cache_from_values(cache.path, metadata, verified)
    for name in (
        "support_scores",
        "support_labels",
        "support_blocks",
        "query_scores",
    ):
        if not np.array_equal(getattr(cache, name), getattr(loaded, name)):
            raise ValueError("In-memory strict-FBCCA cache differs from its immutable file.")
    if cache.query_tokens != loaded.query_tokens:
        raise ValueError("In-memory query tokens differ from the immutable score cache.")
    return loaded


def _verify_template_cache(cache: TemplateScoreCache) -> None:
    if (
        cache.metadata.get("schema") != V2_TEMPLATE_CACHE_SCHEMA
        or cache.metadata.get("query_labels_loaded") is not False
        or cache.metadata.get("template_score_provenance") != asdict(cache.provenance)
        or tuple(cache.metadata.get("subband_weights") or ()) != V2_SUBBAND_WEIGHTS
        or cache.metadata.get("budget") not in {1, 3}
        or not _is_sha256(cache.metadata.get("parent_cache_sha256"))
        or not _is_sha256(cache.metadata.get("query_token_commitment_sha256"))
    ):
        raise ValueError("P2 template cache contract binding drifted.")
    values = _load_npz(cache.path, expected_metadata=cache.metadata)
    stored = np.asarray(values.get("query_class_scores"), dtype=np.float64)
    if not np.array_equal(stored, cache.query_class_scores):
        raise ValueError("In-memory P2 template scores differ from their immutable cache.")


def _verify_subband_cache(cache: SubbandCache) -> None:
    if (
        cache.metadata.get("schema") != V2_SUBBAND_CACHE_SCHEMA
        or cache.metadata.get("query_labels_loaded") is not False
        or cache.metadata.get("hdf5_y_loaded_for_query") is not False
        or cache.metadata.get("filterbank_sha256") != V2_FILTERBANK_SHA256
        or tuple(cache.metadata.get("subband_weights") or ()) != V2_SUBBAND_WEIGHTS
        or cache.metadata.get("subband_dtype") != "float64"
        or not all(
            _is_sha256(cache.metadata.get(field))
            for field in (
                "plan_sha256",
                "allocation_sha256",
                "asset_manifest_sha256",
                "asset_signals_sha256",
                "preprocessing_sha256",
                "support_partition_sha256",
                "query_partition_sha256",
                "query_token_commitment_sha256",
            )
        )
    ):
        raise ValueError("P2 subband cache contract binding drifted.")
    values = _load_npz(cache.path, expected_metadata=cache.metadata)
    loaded = _subband_cache_from_values(cache.path, cache.metadata, values)
    for name in ("support_subbands", "support_labels", "support_blocks", "query_subbands"):
        if not np.array_equal(getattr(cache, name), getattr(loaded, name)):
            raise ValueError("In-memory P2 subband cache differs from its immutable file.")
    if cache.query_tokens != loaded.query_tokens:
        raise ValueError("In-memory P2 tokens differ from the immutable subband cache.")


def _assert_subband_score_cache_binding(
    subbands: SubbandCache,
    scores: FBCCAScoreCache,
) -> None:
    shared_fields = (
        "plan_sha256",
        "allocation_sha256",
        "dataset_id",
        "phase",
        "subject_id",
        "asset_manifest_sha256",
        "asset_signals_sha256",
        "preprocessing_sha256",
        "filterbank_sha256",
        "support_partition_sha256",
        "query_partition_sha256",
        "query_token_commitment_sha256",
    )
    if any(subbands.metadata.get(field) != scores.metadata.get(field) for field in shared_fields):
        raise ValueError("P2 subband and FBCCA caches bind different governed partitions.")
    if (
        subbands.query_tokens != scores.query_tokens
        or not np.array_equal(subbands.support_labels, scores.support_labels)
        or not np.array_equal(subbands.support_blocks, scores.support_blocks)
    ):
        raise ValueError("P2 subband and FBCCA cache rows are not exactly aligned.")


def _load_or_create_npz(
    path: Path,
    *,
    metadata: Mapping[str, Any],
    compute: Callable[[], Mapping[str, np.ndarray]],
) -> dict[str, np.ndarray]:
    if os.path.lexists(path):
        return _load_npz(path, expected_metadata=metadata)
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    if path.parent.is_symlink() or not path.parent.is_dir():
        raise ValueError("V2 cache parent must be a real directory.")
    values = {key: np.asarray(value) for key, value in compute().items()}
    payload_hash = _array_mapping_sha256(values)
    stored_metadata = {**dict(metadata), "payload_sha256": payload_hash}
    staging = path.parent / f".{path.name}.staging-{uuid.uuid4().hex}"
    descriptor = os.open(staging, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o400)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            descriptor = -1
            np.savez_compressed(
                handle,
                metadata_json=np.asarray(
                    json.dumps(stored_metadata, sort_keys=True, separators=(",", ":"))
                ),
                **values,
            )
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(staging, path)
            _fsync_directory(path.parent)
        except FileExistsError:
            # A concurrent producer won. Its content must match exactly.
            return _load_npz(path, expected_metadata=metadata)
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        if os.path.lexists(staging):
            staging.unlink()
            _fsync_directory(path.parent)
    return _load_npz(path, expected_metadata=metadata)


def _load_npz(path: Path, *, expected_metadata: Mapping[str, Any]) -> dict[str, np.ndarray]:
    if path.is_symlink() or not path.is_file():
        raise ValueError("V2 cache path must be a nonsymlink regular file.")
    try:
        with np.load(path, allow_pickle=False) as loaded:
            if "metadata_json" not in loaded.files:
                raise ValueError("V2 cache lacks metadata.")
            raw_metadata = loaded["metadata_json"]
            if raw_metadata.shape != ():
                raise ValueError("V2 cache metadata is not scalar JSON.")
            metadata = json.loads(str(raw_metadata.item()))
            values = {key: loaded[key] for key in loaded.files if key != "metadata_json"}
    except (OSError, EOFError, ValueError, json.JSONDecodeError) as error:
        raise ValueError(
            "V2 cache is partial or unreadable; refusing recomputation in place."
        ) from error
    claimed_payload = metadata.pop("payload_sha256", None)
    if _canonical_json_sha256(metadata) != _canonical_json_sha256(expected_metadata):
        raise FileExistsError("V2 content-addressed cache collision or metadata drift.")
    if not _is_sha256(claimed_payload) or claimed_payload != _array_mapping_sha256(values):
        raise ValueError("V2 cache payload hash is invalid.")
    return values


def _cache_path(cache_dir: str | Path, kind: str, key: str) -> Path:
    root = _safe_external_path(cache_dir)
    if not _is_sha256(key) or not re.fullmatch(r"[a-z_]+", kind):
        raise ValueError("Invalid V2 content-addressed cache key.")
    return root / kind / f"{key}.npz"


def _read_channel_view(
    asset: ValidatedExternalAsset,
    h5_index: int,
    *,
    query: bool,
    expected_label: int | None = None,
    reader: HDF5SampleReader,
) -> np.ndarray:
    if query:
        if expected_label is not None:
            raise PermissionError("Query signal reader forbids a label argument.")
        signal, mask = reader.read_unlabeled(asset.signals_path, h5_index)
    else:
        if expected_label is None:
            raise ValueError("Support signal reader requires its calibration label.")
        signal, mask, observed_label = reader.read(asset.signals_path, h5_index)
        if observed_label != expected_label:
            raise ValueError("Support manifest label differs from HDF5 y.")
    positions = np.asarray(V2_CANONICAL_CHANNEL_POSITIONS, dtype=np.int64)
    if signal.shape != (64, 400) or mask.shape != (64,):
        raise ValueError("V2 external signal has the wrong processed shape.")
    if not mask[positions].all():
        raise ValueError("V2 external signal lacks a canonical posterior channel.")
    view = np.asarray(signal[positions], dtype=np.float64)
    if view.shape != (8, 400) or not np.isfinite(view).all():
        raise ValueError("V2 canonical eight-channel signal view is invalid.")
    return view


def _read_partition_signals(
    asset: ValidatedExternalAsset,
    participant: ParticipantPartition,
) -> tuple[np.ndarray, np.ndarray]:
    reader = HDF5SampleReader(persistent=True)
    try:
        support = np.stack(
            [
                _read_channel_view(
                    asset,
                    row.h5_index,
                    query=False,
                    expected_label=row.label,
                    reader=reader,
                )
                for row in participant.support_rows
            ]
        )
        query = np.stack(
            [
                _read_channel_view(
                    asset,
                    h5_index,
                    query=True,
                    reader=reader,
                )
                for h5_index in participant.query_h5_indices
            ]
        )
    finally:
        reader.close()
    return support, query


def _map_fbcca(
    signals: np.ndarray,
    *,
    sfreq: float,
    filterbank: Mapping[str, Any],
    workers: int,
) -> np.ndarray:
    tasks = [
        (np.asarray(signal, dtype=np.float64), sfreq, tuple(V2_CODEBOOK_HZ), dict(filterbank))
        for signal in signals
    ]
    if workers == 1:
        return np.stack([_fbcca_worker(value) for value in tasks])
    thread_names = (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    )
    previous = {name: os.environ.get(name) for name in thread_names}
    try:
        # Spawned workers import NumPy/SciPy before their initializer, so the
        # fixed limits must already be present in their inherited environment.
        for name in thread_names:
            os.environ[name] = "1"
        with ProcessPoolExecutor(
            max_workers=workers,
            mp_context=get_context("spawn"),
            initializer=_limit_worker_threads,
        ) as executor:
            return np.stack(list(executor.map(_fbcca_worker, tasks, chunksize=4)))
    finally:
        for name, value in previous.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


def _fbcca_worker(args: tuple[np.ndarray, float, tuple[float, ...], dict[str, Any]]) -> np.ndarray:
    signal, sfreq, codebook, filterbank = args
    return np.asarray(predict_fbcca(signal, codebook, sfreq, filterbank=filterbank)[1])


def _limit_worker_threads() -> None:
    for name in (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        os.environ[name] = "1"


def _validate_public_manifest(frame: pd.DataFrame, dataset_id: str) -> None:
    expected_columns = {
        "h5_index",
        "dataset_id",
        "subject_id",
        "session_id",
        "run_id",
        "sfreq_processed",
        "n_channels_used",
        "canonical_channel_ids",
    }
    if (
        set(frame.columns) != expected_columns
        or len(frame) != _DATASET_SPECS[dataset_id]["n_subjects"] * 160
    ):
        raise ValueError("V2 public manifest projection has a wrong schema or row count.")
    indices = _exact_integer_series(frame["h5_index"], name="h5 index").to_numpy()
    if not np.array_equal(indices, np.arange(len(frame))):
        raise ValueError("V2 manifest HDF5 indices are not contiguous.")
    spec = _DATASET_SPECS[dataset_id]
    if set(frame["dataset_id"].astype(str)) != {spec["manifest_dataset_id"]}:
        raise ValueError("V2 manifest dataset identity drifted.")
    if set(frame["session_id"].astype(str)) != {"session01"}:
        raise ValueError("V2 external assets must have the frozen single-session schema.")
    if set(frame["run_id"].astype(str)) != {
        "block01",
        "block02",
        "block03",
        "block04",
    }:
        raise ValueError("V2 external asset has unexpected block names.")
    if set(frame["sfreq_processed"].astype(float)) != {200.0}:
        raise ValueError("V2 external asset sampling rate drifted.")
    expected_used = 64 if dataset_id == "beta_v1" else 8
    if set(_exact_integer_series(frame["n_channels_used"], name="n_channels_used")) != {
        expected_used
    }:
        raise ValueError("V2 external asset channel count drifted.")
    positions = np.asarray(V2_CANONICAL_CHANNEL_POSITIONS)
    for ids in frame["canonical_channel_ids"]:
        values = np.asarray(ids)
        if (
            values.shape != (64,)
            or tuple(values[positions].astype(int)) != V2_CANONICAL_CHANNEL_IDS
        ):
            raise ValueError("V2 external asset lacks the frozen canonical channel mapping.")


def _validate_class_map(path: Path) -> None:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping) or tuple(value) != tuple(str(index) for index in range(40)):
        raise ValueError("V2 class map does not contain exact labels 0-39 in order.")
    for index, frequency in enumerate(V2_CODEBOOK_HZ):
        row = value[str(index)]
        if (
            not isinstance(row, Mapping)
            or row.get("label") != index
            or not math.isclose(
                float(row.get("stimulus_frequency_hz", math.nan)),
                frequency,
                rel_tol=0.0,
                abs_tol=1e-12,
            )
        ):
            raise ValueError("V2 class map differs from the complete frozen codebook.")


def _validate_asset_info(path: Path, dataset_id: str) -> None:
    value = json.loads(path.read_text(encoding="utf-8"))
    spec = _DATASET_SPECS[dataset_id]
    frequencies = tuple(float(item) for item in value.get("canonical_class_frequencies", ()))
    if (
        value.get("dataset_id") != spec["manifest_dataset_id"]
        or len(frequencies) != 40
        or not np.allclose(frequencies, V2_CODEBOOK_HZ, rtol=0.0, atol=1e-12)
    ):
        raise ValueError("V2 asset-info identity/codebook drifted.")


def _validate_h5_container(path: Path, expected_rows: int) -> None:
    with h5py.File(path, "r") as handle:
        if set(handle.keys()) != {"x", "channel_mask", "y"}:
            raise ValueError("V2 signals HDF5 has an unexpected dataset inventory.")
        if (
            handle["x"].shape != (expected_rows, 64, 400)
            or handle["channel_mask"].shape != (expected_rows, 64)
            or handle["y"].shape != (expected_rows,)
        ):
            raise ValueError("V2 signals HDF5 shapes differ from the frozen asset schema.")
        # Do not index or materialize y here: query outcomes remain unopened.


def _validate_filterbank(value: Mapping[str, Any]) -> None:
    if (
        tuple(tuple(float(x) for x in band) for band in value.get("bands", ()))
        != tuple((float(low), 90.0) for low in (6, 14, 22, 30, 38, 46, 54))
        or value.get("order") != 4
        or value.get("n_harmonics") != 5
        or float(value.get("regularization", math.nan)) != 1.0e-8
        or float(value.get("weight_exponent", math.nan)) != 1.25
        or float(value.get("weight_offset", math.nan)) != 0.25
        or tuple(value.get("weights", ())) != V2_SUBBAND_WEIGHTS
        or value.get("filter_family") != "chebyshev1"
        or float(value.get("passband_ripple_db", math.nan)) != 0.5
        or value.get("reproduction_contract")
        != "chen2015_m3_v2_explicit_weights_numpy_version_independent"
    ):
        raise ValueError("V2 filterbank payload drifted from r6.")
    resolved = resolve_filterbank_parameters(value, 200.0)
    if not np.array_equal(
        np.asarray(resolved["weights"], dtype=np.float64),
        np.asarray(V2_SUBBAND_WEIGHTS, dtype=np.float64),
    ):
        raise ValueError("V2 runtime filterbank weights drifted from the exact r6 vector.")


def _validate_allocation(receipt: Mapping[str, Any]) -> str:
    if set(receipt) != {"schema", "method", "seeds", "allocation"}:
        raise ValueError("V2 allocation has a wrong exact schema.")
    if (
        receipt.get("schema") != V2_ALLOCATION_SCHEMA
        or receipt.get("method")
        != "dataset_stratified_numpy_default_rng_permutation_sorted_ids_dev_first"
        or set(receipt.get("seeds") or {}) != set(_DATASET_SPECS)
        or set(receipt.get("allocation") or {}) != set(_DATASET_SPECS)
    ):
        raise ValueError("V2 allocation header drifted.")
    for dataset_id, spec in _DATASET_SPECS.items():
        if receipt["seeds"].get(dataset_id) != spec["seed"]:
            raise ValueError(f"V2 {dataset_id} allocation seed drifted.")
        row = receipt["allocation"].get(dataset_id)
        if not isinstance(row, Mapping) or set(row) != {
            "excluded_exposed",
            "development",
            "independent_gate",
        }:
            raise ValueError(f"V2 {dataset_id} allocation row has a wrong schema.")
        excluded = _subject_list(row["excluded_exposed"])
        development = _subject_list(row["development"])
        gate = _subject_list(row["independent_gate"])
        if (
            excluded != spec["excluded_exposed"]
            or len(development) != spec["development_count"]
            or len(gate) != spec["independent_gate_count"]
        ):
            raise ValueError(f"V2 {dataset_id} allocation count/exclusion drifted.")
        universe = tuple(f"sub{index:03d}" for index in range(1, spec["n_subjects"] + 1))
        eligible = tuple(value for value in universe if value not in set(excluded))
        shuffled = np.asarray(eligible, dtype=str)
        np.random.default_rng(spec["seed"]).shuffle(shuffled)
        if development != tuple(sorted(shuffled[: len(development)].tolist())) or gate != tuple(
            sorted(shuffled[len(development) :].tolist())
        ):
            raise ValueError(f"V2 {dataset_id} allocation RNG replay failed.")
    observed = _canonical_json_sha256(receipt)
    if observed != V2_ALLOCATION_SHA256:
        raise ValueError("V2 allocation canonical SHA-256 drifted.")
    return observed


def _support_partition_sha(
    asset: ValidatedExternalAsset,
    rows: Sequence[SupportRow],
) -> str:
    return _canonical_json_sha256(
        {
            "domain": "cfeg-v2-support-partition-v1",
            "manifest_sha256": asset.manifest_sha256,
            "rows": [asdict(row) for row in rows],
        }
    )


def _template_support_partition_sha(
    parent_support_partition_sha256: str,
    *,
    budget: int,
    labels: np.ndarray,
) -> str:
    if not _is_sha256(parent_support_partition_sha256):
        raise ValueError("P2 parent support partition digest is invalid.")
    resolved_budget = _budget(budget)
    if resolved_budget == 0:
        raise ValueError("P2 template support partition requires k=1 or k=3.")
    exact_labels = _exact_integer_array(labels, name="P2 support labels")
    if exact_labels.shape != (resolved_budget * V2_N_CLASSES,):
        raise ValueError("P2 template support labels have the wrong budget length.")
    return _canonical_json_sha256(
        {
            "parent_support_partition_sha256": parent_support_partition_sha256,
            "budget": resolved_budget,
            "support_blocks": list(range(1, resolved_budget + 1)),
            "support_labels": exact_labels.tolist(),
        }
    )


def _query_partition_sha(
    asset: ValidatedExternalAsset,
    participant: ParticipantPartition,
    tokens: Sequence[str],
) -> str:
    return _canonical_json_sha256(
        {
            "domain": "cfeg-v2-query-partition-v1",
            "manifest_sha256": asset.manifest_sha256,
            "internal_h5_indices": participant.query_h5_indices,
            "query_tokens": tuple(tokens),
        }
    )


def _array_mapping_sha256(values: Mapping[str, np.ndarray]) -> str:
    digest = hashlib.sha256()
    for key in sorted(values):
        array = np.ascontiguousarray(values[key])
        digest.update(key.encode("utf-8"))
        digest.update(b"\0")
        digest.update(array.dtype.str.encode("ascii"))
        digest.update(b"\0")
        digest.update(_canonical_json_bytes({"shape": array.shape}))
        digest.update(memoryview(array).cast("B"))
    return digest.hexdigest()


def _assert_query_blind_request(request: Mapping[str, Any]) -> None:
    forbidden = set(request) & _QUERY_FORBIDDEN_FIELDS
    if forbidden:
        raise PermissionError(
            f"V2 operator request contains query identity/outcome fields: {forbidden}"
        )
    tokens = request.get("query_tokens")
    if not isinstance(tokens, list) or any(
        not re.fullmatch(r"qv2_[0-9a-f]{64}", x) for x in tokens
    ):
        raise ValueError("V2 request query tokens are not canonical opaque HMAC tokens.")
    if request.get("variant") != "A_Q" or any(
        request.get(field) is not None
        for field in (
            "query_interfaces",
            "support_interfaces",
            "query_impedance_kohm",
            "support_impedance_kohm",
            "relative_context_pairing_sha256",
        )
    ):
        raise PermissionError("External BETA/Dong requests are signal-only A_Q.")


def _require_participant(
    asset: ValidatedExternalAsset,
    participant: ParticipantPartition,
) -> None:
    if asset.participant(participant.subject_id) != participant:
        raise PermissionError("Participant partition differs from the validated inventory.")


def _assert_asset_unchanged(asset: ValidatedExternalAsset) -> None:
    if (
        _file_identity(asset.manifest_path) != asset.manifest_file_identity
        or _file_identity(asset.signals_path) != asset.signals_file_identity
    ):
        raise PermissionError(
            "Validated V2 input changed after its SHA-256 audit; revalidation is required."
        )


def _file_identity(path: Path) -> tuple[int, int, int, int]:
    if path.is_symlink() or not path.is_file():
        raise PermissionError("Validated V2 input disappeared or became a symlink.")
    observed = path.stat()
    return (observed.st_dev, observed.st_ino, observed.st_size, observed.st_mtime_ns)


def _regular_file(path: str | Path, name: str) -> Path:
    value = _safe_external_path(path)
    if value.is_symlink() or not value.is_file():
        raise ValueError(f"{name} must be one existing nonsymlink regular file.")
    return value


def _safe_external_path(path: str | Path) -> Path:
    value = Path(path).expanduser().absolute()
    if _FORBIDDEN_PATH_PARTS.search(str(value)):
        raise PermissionError("V2 external producer refuses source39/wearable/held paths.")
    for parent in (value, *value.parents):
        if parent.is_symlink():
            raise PermissionError("V2 external producer refuses symlink path aliases.")
    return value


def _dataset_id(value: str) -> str:
    if value not in _DATASET_SPECS:
        raise PermissionError("V2 producer accepts only beta_v1 or dong2023_v1.")
    return value


def _phase(value: str) -> Literal["development", "independent_gate"]:
    if value not in {"development", "independent_gate"}:
        raise PermissionError("V2 producer phase must be development or independent_gate.")
    return value  # type: ignore[return-value]


def _budget(value: int) -> Literal[0, 1, 3]:
    if type(value) is not int or value not in V2_BUDGETS:
        raise ValueError("V2 external budget must be exactly 0, 1, or 3.")
    return value  # type: ignore[return-value]


def _workers(value: int) -> int:
    if type(value) is not int or value < 1 or value > 32:
        raise ValueError("V2 workers must be an integer in [1,32].")
    return value


def _subject_list(value: Any) -> tuple[str, ...]:
    if (
        not isinstance(value, list)
        or any(type(item) is not str or not _SUBJECT_RE.fullmatch(item) for item in value)
        or len(set(value)) != len(value)
        or value != sorted(value)
    ):
        raise ValueError("V2 subject allocation must be a sorted unique subject-ID list.")
    return tuple(value)


def _exact_integer_series(value: pd.Series, *, name: str) -> pd.Series:
    numeric = pd.to_numeric(value, errors="raise")
    array = numeric.to_numpy(dtype=np.float64)
    if not np.isfinite(array).all() or not np.equal(array, np.floor(array)).all():
        raise ValueError(f"{name} must contain exact finite integers.")
    return pd.Series(array.astype(np.int64), index=value.index)


def _exact_integer_array(value: Any, *, name: str) -> np.ndarray:
    raw = np.asarray(value)
    if not np.issubdtype(raw.dtype, np.number) or np.issubdtype(raw.dtype, np.bool_):
        raise TypeError(f"{name} must be a numeric non-boolean array.")
    numeric = raw.astype(np.float64)
    if not np.isfinite(numeric).all() or not np.equal(numeric, np.floor(numeric)).all():
        raise ValueError(f"{name} must contain exact finite integers.")
    return numeric.astype(np.int64)


def _is_sha256(value: Any) -> bool:
    return type(value) is str and _SHA256_RE.fullmatch(value) is not None


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _canonical_json_sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_json_bytes(value)).hexdigest()


def _with_hash(payload: Mapping[str, Any], field: str) -> dict[str, Any]:
    result = dict(payload)
    result[field] = _canonical_json_sha256(result)
    return result


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
