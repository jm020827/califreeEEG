from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from typing import Any, Literal

import numpy as np
import pandas as pd
import torch

from cfeg.data.metadata_calibration_features import WEARABLE_CALIBRATION_CHANNEL_IDS

MetadataIntervention = Literal[
    "exact_off",
    "correct",
    "all_missing",
    "block_shuffle",
    "stale",
    "opposite_interface",
]

_SUPPORT_RUNS = tuple(f"block{index:02d}" for index in range(1, 6))
_QUERY_RUNS = tuple(f"block{index:02d}" for index in range(6, 11))
_ALL_RUNS = (*_SUPPORT_RUNS, *_QUERY_RUNS)
_MAPPING_COLUMNS = (
    "subject_id",
    "electrode_type",
    "run_id",
    "context",
    "metadata_mode",
    "donor_subject_id",
    "donor_electrode_type",
    "donor_run_id",
)
_ROW_TOKEN = re.compile(r"[qs]_[0-9a-f]{64}")
_SHA256 = re.compile(r"[0-9a-f]{64}")


@dataclass(frozen=True)
class MetadataInterventionMapping:
    context: str
    frame: pd.DataFrame
    sha256: str
    shuffle_seed: int


@dataclass(frozen=True)
class ResolvedMetadataInterventionBatch:
    """Exact acquisition-context values consumed by one model batch.

    Impedance is already transformed to the frozen model scale. Missing values
    are zero-filled and carried separately in ``impedance_missing_by_channel``.
    Opaque row tokens are included to prove exact coverage. Raw sample IDs,
    labels, and predictions are deliberately not part of this boundary.
    """

    electrode_types: tuple[str | None, ...]
    impedance_by_channel: np.ndarray
    impedance_missing_by_channel: np.ndarray
    metadata_modes: tuple[str, ...]
    row_tokens: tuple[str, ...]
    base_input_sha256s: tuple[str, ...]
    target_block_keys: tuple[tuple[str, str, str], ...]
    cell_execution_spec: tuple[tuple[str, str], ...]
    phase: str
    batch_kind: str
    mapping_sha256: str
    resolved_context_values_sha256: str


@dataclass(frozen=True)
class ResolvedMetadataModelInputs:
    """The two arguments that must travel together into the M feature builder."""

    condition: dict[str, Any]
    electrode_types: tuple[str | None, ...]
    metadata_modes: tuple[str, ...]
    resolved_context_values_sha256: str


def build_metadata_intervention_mapping(
    manifest: pd.DataFrame,
    *,
    context: MetadataIntervention,
    shuffle_seed: int = 20260904,
) -> MetadataInterventionMapping:
    """Construct a block-level M donor map without reading labels or EEG.

    The block-shuffle donor always stays within participant, interface, and
    support/query pool.  Stale context is the true previous block and therefore
    has an explicit all-missing row for block01.  Opposite-interface context is
    paired within participant and block.
    """

    allowed_contexts = {
        "exact_off",
        "correct",
        "all_missing",
        "block_shuffle",
        "stale",
        "opposite_interface",
    }
    if context not in allowed_contexts:
        raise ValueError(f"Unknown metadata intervention context {context!r}.")
    required = {
        "dataset_id",
        "subject_id",
        "electrode_type",
        "run_id",
        "impedance_channel_ids",
        "impedance_kohm_by_channel",
    }
    missing = required - set(manifest.columns)
    extra = set(manifest.columns) - required
    if missing or extra:
        raise ValueError(
            "Block-context view must use its exact label-free allowlist; "
            f"missing={sorted(missing)}, extra={sorted(extra)}."
        )
    frame = manifest.loc[:, sorted(required)].copy()
    for column in ("dataset_id", "subject_id", "electrode_type", "run_id"):
        frame[column] = frame[column].astype(str)
    if set(frame["dataset_id"]) != {"wearable"}:
        raise ValueError("Metadata intervention mapping is frozen to wearable_v3.")
    block_key = ["subject_id", "electrode_type", "run_id"]
    if frame.duplicated(block_key).any():
        raise ValueError("Block-context view must contain exactly one row per block.")

    block_lookup: set[tuple[str, str, str]] = set()
    participants = tuple(sorted(frame["subject_id"].unique().tolist()))
    for subject_id in participants:
        subject = frame.loc[frame["subject_id"].eq(subject_id)]
        if set(subject["electrode_type"]) != {"dry", "wet"}:
            raise ValueError(f"Subject {subject_id} must contain exact dry and wet interfaces.")
        for electrode_type in ("dry", "wet"):
            condition = subject.loc[subject["electrode_type"].eq(electrode_type)]
            if set(condition["run_id"]) != set(_ALL_RUNS):
                raise ValueError("Every participant-interface must contain exact block01-block10.")
            for run_id in _ALL_RUNS:
                block = condition.loc[condition["run_id"].eq(run_id)]
                if len(block) != 1:
                    raise ValueError("Block-context view must contain exactly one row per block.")
                _canonical_metadata_value(block.iloc[0]["impedance_kohm_by_channel"])
                _validate_impedance_channel_ids(block.iloc[0]["impedance_channel_ids"])
                block_lookup.add((subject_id, electrode_type, run_id))

    rows: list[dict[str, object]] = []
    for target in frame.sort_values(block_key, kind="mergesort").itertuples(index=False):
        donor = _donor_block(
            subject_id=target.subject_id,
            electrode_type=target.electrode_type,
            run_id=target.run_id,
            context=context,
            shuffle_seed=shuffle_seed,
        )
        if donor is None:
            donor_subject = donor_interface = donor_run = None
            metadata_mode = "off" if context == "exact_off" else "all_missing"
        else:
            donor_subject, donor_interface, donor_run = donor
            if (donor_subject, donor_interface, donor_run) not in block_lookup:
                raise ValueError("Metadata intervention donor block is absent from the sealed view.")
            metadata_mode = "observed"
        rows.append(
            {
                "subject_id": target.subject_id,
                "electrode_type": target.electrode_type,
                "run_id": target.run_id,
                "context": context,
                "metadata_mode": metadata_mode,
                "donor_subject_id": donor_subject,
                "donor_electrode_type": donor_interface,
                "donor_run_id": donor_run,
            }
        )
    mapping = pd.DataFrame(rows, columns=_MAPPING_COLUMNS)
    digest = metadata_intervention_mapping_sha256(
        mapping,
        shuffle_seed=shuffle_seed,
    )
    return MetadataInterventionMapping(
        context=context,
        frame=mapping,
        sha256=digest,
        shuffle_seed=int(shuffle_seed),
    )


def metadata_intervention_mapping_sha256(
    mapping: pd.DataFrame,
    *,
    shuffle_seed: int = 20260904,
) -> str:
    missing = set(_MAPPING_COLUMNS) - set(mapping.columns)
    if missing:
        raise ValueError(f"Intervention mapping is missing columns {sorted(missing)}.")
    extra = set(mapping.columns) - set(_MAPPING_COLUMNS)
    if extra:
        raise ValueError(f"Intervention mapping has undeclared columns {sorted(extra)}.")
    canonical = mapping.loc[:, _MAPPING_COLUMNS].sort_values(
        ["subject_id", "electrode_type", "run_id"], kind="mergesort"
    )
    records = [
        {key: (None if pd.isna(value) else value) for key, value in row.items()}
        for row in canonical.to_dict(orient="records")
    ]
    payload = json.dumps(
        {
            "schema": "cfeg.metadata-calibration-intervention-mapping.v1",
            "shuffle_seed": int(shuffle_seed),
            "rows": records,
        },
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def resolve_metadata_intervention_batch(
    *,
    subject_ids: Sequence[str],
    electrode_types: Sequence[str],
    run_ids: Sequence[str],
    row_tokens: Sequence[str],
    base_input_sha256s: Sequence[str],
    mapping: MetadataInterventionMapping,
    sealed_block_context: pd.DataFrame,
    cell_spec: Any,
    phase: Literal["source_development", "held_participant_evaluation"],
    batch_kind: Literal["query", "support"],
) -> ResolvedMetadataInterventionBatch:
    """Resolve a frozen cell dispatch into the exact M values for a batch.

    This function is label-free. It joins only participant/interface/block keys
    against the sealed one-row-per-block context view. The cell dispatch, donor
    mapping, effective missingness, ablation, and final float32 values all enter
    the returned digest so a producer receipt can bind what the model consumed.
    """

    # Imported lazily because the execution table imports the intervention
    # Literal from this module.
    from cfeg.metadata_calibration_execution import (
        CellExecutionSpec,
        resolve_cell_execution_spec,
    )

    if not isinstance(cell_spec, CellExecutionSpec):
        raise TypeError("cell_spec must be one frozen CellExecutionSpec.")
    if batch_kind not in {"query", "support"}:
        raise ValueError("batch_kind must be 'query' or 'support'.")
    canonical_spec = resolve_cell_execution_spec(
        role=cell_spec.role,
        context=cell_spec.context,
        phase=phase,
    )
    if cell_spec != canonical_spec:
        raise ValueError("Cell specification differs from the frozen execution dispatch.")
    lengths = {
        len(subject_ids),
        len(electrode_types),
        len(run_ids),
        len(row_tokens),
        len(base_input_sha256s),
    }
    if len(lengths) != 1 or len(subject_ids) == 0:
        raise ValueError("Batch block keys must be nonempty and have equal lengths.")
    expected_token_prefix = "q_" if batch_kind == "query" else "s_"
    normalized_tokens = tuple(str(value) for value in row_tokens)
    if len(set(normalized_tokens)) != len(normalized_tokens) or not all(
        _ROW_TOKEN.fullmatch(value) and value.startswith(expected_token_prefix)
        for value in normalized_tokens
    ):
        raise ValueError("Resolved rows require unique purpose-matched opaque row tokens.")
    normalized_input_hashes = tuple(str(value) for value in base_input_sha256s)
    if not all(_SHA256.fullmatch(value) for value in normalized_input_hashes):
        raise ValueError("Every resolved row must bind one base input SHA-256.")
    if mapping.context != cell_spec.intervention:
        raise ValueError("Intervention mapping does not match the frozen cell dispatch.")
    recomputed_mapping_sha256 = metadata_intervention_mapping_sha256(
        mapping.frame,
        shuffle_seed=mapping.shuffle_seed,
    )
    if recomputed_mapping_sha256 != mapping.sha256:
        raise ValueError("Intervention mapping content drifted after its digest was computed.")
    if set(mapping.frame["context"].astype(str)) != {mapping.context}:
        raise ValueError("Intervention mapping rows disagree with their declared context.")
    _validate_mapping_semantics(mapping)

    context_lookup = _validated_block_context_lookup(sealed_block_context)
    mapping_frame = mapping.frame.set_index(
        ["subject_id", "electrode_type", "run_id"], verify_integrity=True
    )
    if set(mapping_frame.index.tolist()) != set(context_lookup):
        raise ValueError(
            "Intervention mapping and sealed context must cover the same block keys."
        )
    declared_mode = getattr(cell_spec, f"{batch_kind}_metadata_mode")
    ablation = getattr(cell_spec, f"{batch_kind}_metadata_ablation")
    resolved_interfaces: list[str | None] = []
    resolved_modes: list[str] = []
    impedance_rows: list[np.ndarray] = []
    missing_rows: list[np.ndarray] = []

    for raw_key in zip(subject_ids, electrode_types, run_ids):
        key = tuple(str(value) for value in raw_key)
        if key not in mapping_frame.index:
            raise ValueError(f"Batch block {key!r} is absent from the intervention mapping.")
        donor = mapping_frame.loc[key]
        if isinstance(donor, pd.DataFrame):
            raise TypeError("Intervention mapping must contain one row per target block.")

        effective_mode = str(donor["metadata_mode"])
        if declared_mode in {"off", "all_missing"}:
            effective_mode = str(declared_mode)
        elif declared_mode != "observed":
            raise ValueError(f"Unsupported metadata mode {declared_mode!r}.")

        if effective_mode in {"off", "all_missing"}:
            resolved_interface: str | None = None
            impedance = np.zeros(8, dtype=np.float32)
            impedance_missing = np.ones(8, dtype=bool)
        elif effective_mode == "observed":
            donor_key = (
                str(donor["donor_subject_id"]),
                str(donor["donor_electrode_type"]),
                str(donor["donor_run_id"]),
            )
            if donor_key not in context_lookup:
                raise ValueError(
                    f"Mapped donor block {donor_key!r} is absent from the sealed context view."
                )
            resolved_interface = donor_key[1]
            impedance, impedance_missing = _model_impedance(
                context_lookup[donor_key]
            )
        else:
            raise ValueError(f"Unsupported mapped metadata mode {effective_mode!r}.")

        if ablation == "interface_only":
            impedance = np.zeros(8, dtype=np.float32)
            impedance_missing = np.ones(8, dtype=bool)
        elif ablation == "impedance_only":
            resolved_interface = None
        elif ablation != "full":
            raise ValueError(f"Unsupported metadata ablation {ablation!r}.")

        resolved_interfaces.append(resolved_interface)
        resolved_modes.append(effective_mode)
        impedance_rows.append(impedance)
        missing_rows.append(impedance_missing)

    impedance_array = np.stack(impedance_rows).astype(np.float32, copy=False)
    missing_array = np.stack(missing_rows).astype(bool, copy=False)
    target_block_keys = tuple(
        tuple(str(value) for value in key)
        for key in zip(subject_ids, electrode_types, run_ids)
    )
    cell_contract = tuple(sorted(cell_spec.as_dict().items()))
    impedance_array.setflags(write=False)
    missing_array.setflags(write=False)
    provisional = ResolvedMetadataInterventionBatch(
        electrode_types=tuple(resolved_interfaces),
        impedance_by_channel=impedance_array,
        impedance_missing_by_channel=missing_array,
        metadata_modes=tuple(resolved_modes),
        row_tokens=normalized_tokens,
        base_input_sha256s=normalized_input_hashes,
        target_block_keys=target_block_keys,
        cell_execution_spec=cell_contract,
        phase=phase,
        batch_kind=batch_kind,
        mapping_sha256=mapping.sha256,
        resolved_context_values_sha256="",
    )
    return replace(
        provisional,
        resolved_context_values_sha256=_resolved_context_batch_sha256(provisional),
    )


def apply_resolved_metadata_to_condition(
    x: torch.Tensor,
    cond: Mapping[str, Any],
    resolved: ResolvedMetadataInterventionBatch,
) -> ResolvedMetadataModelInputs:
    """Install M and return interface plus impedance as one indivisible input."""

    return _apply_resolved_metadata_to_condition(
        x,
        cond,
        resolved,
        prevalidated_base_input_sha256s=None,
    )


def apply_prevalidated_resolved_metadata_to_condition(
    x: torch.Tensor,
    cond: Mapping[str, Any],
    resolved: ResolvedMetadataInterventionBatch,
    *,
    prevalidated_base_input_sha256s: Sequence[str],
) -> ResolvedMetadataModelInputs:
    """Install M after a prepared batch has already bound every EEG row once."""

    return _apply_resolved_metadata_to_condition(
        x,
        cond,
        resolved,
        prevalidated_base_input_sha256s=prevalidated_base_input_sha256s,
    )


def _apply_resolved_metadata_to_condition(
    x: torch.Tensor,
    cond: Mapping[str, Any],
    resolved: ResolvedMetadataInterventionBatch,
    *,
    prevalidated_base_input_sha256s: Sequence[str] | None,
) -> ResolvedMetadataModelInputs:
    """Shared installer with either immediate or prepared base-input proof."""

    if _resolved_context_batch_sha256(resolved) != resolved.resolved_context_values_sha256:
        raise ValueError("Resolved metadata values drifted after their digest was computed.")

    channel_ids = cond.get("channel_ids")
    channel_mask = cond.get("channel_mask")
    if not torch.is_tensor(channel_ids) or not torch.is_tensor(channel_mask):
        raise ValueError("Condition must contain channel_ids and channel_mask tensors.")
    if channel_ids.ndim != 2 or channel_mask.shape != channel_ids.shape:
        raise ValueError("Condition channel IDs and masks must be aligned rank-two tensors.")
    batch, channels = channel_ids.shape
    if x.dtype != torch.float32 or x.shape[:2] != (batch, channels) or x.ndim != 3:
        raise ValueError("EEG must be an aligned rank-three float32 tensor.")
    if prevalidated_base_input_sha256s is None:
        for row_index in range(batch):
            observed_base_sha256 = metadata_calibration_base_input_sha256(
                x[row_index],
                channel_mask[row_index],
            )
            if observed_base_sha256 != resolved.base_input_sha256s[row_index]:
                raise ValueError(
                    "Resolved row key does not bind the EEG base input at this position."
                )
    elif tuple(map(str, prevalidated_base_input_sha256s)) != tuple(
        resolved.base_input_sha256s
    ):
        raise ValueError("Prepared base-input proof differs from the resolved row order.")
    if resolved.impedance_by_channel.shape != (batch, 8) or (
        resolved.impedance_missing_by_channel.shape != (batch, 8)
    ):
        raise ValueError("Resolved impedance must have shape [batch,8].")

    values = torch.zeros((batch, channels), dtype=torch.float32, device=channel_ids.device)
    missing = torch.ones((batch, channels), dtype=torch.bool, device=channel_ids.device)
    active = channel_mask.to(device=channel_ids.device, dtype=torch.bool)
    for official_index, channel_id in enumerate(WEARABLE_CALIBRATION_CHANNEL_IDS):
        positions = channel_ids.eq(int(channel_id)) & active
        if not positions.sum(dim=1).eq(1).all():
            raise ValueError(
                "Every row must contain each official wearable channel exactly once."
            )
        donor_values = torch.tensor(
            resolved.impedance_by_channel[:, official_index],
            dtype=torch.float32,
            device=channel_ids.device,
        )
        donor_missing = torch.tensor(
            resolved.impedance_missing_by_channel[:, official_index],
            dtype=torch.bool,
            device=channel_ids.device,
        )
        values = torch.where(positions, donor_values.unsqueeze(1), values)
        missing = torch.where(positions, donor_missing.unsqueeze(1), missing)

    updated = dict(cond)
    updated["channel_impedance"] = values
    updated["channel_impedance_missing"] = missing
    updated["metadata_all_missing_rows"] = torch.tensor(
        [mode != "observed" for mode in resolved.metadata_modes],
        dtype=torch.bool,
        device=channel_ids.device,
    )
    return ResolvedMetadataModelInputs(
        condition=updated,
        electrode_types=resolved.electrode_types,
        metadata_modes=resolved.metadata_modes,
        resolved_context_values_sha256=resolved.resolved_context_values_sha256,
    )


def metadata_calibration_base_input_sha256(
    x: np.ndarray | torch.Tensor,
    channel_mask: np.ndarray | torch.Tensor,
) -> str:
    """Hash one exact model waveform and mask independently of row identity."""

    if torch.is_tensor(x):
        if x.dtype != torch.float32:
            raise ValueError("Base-input EEG tensors must use float32.")
        x_array = x.detach().contiguous().cpu().numpy()
    else:
        x_array = np.asarray(x)
        if x_array.dtype != np.float32:
            raise ValueError("Base-input EEG arrays must use float32.")
    if torch.is_tensor(channel_mask):
        if channel_mask.dtype != torch.bool:
            raise ValueError("Base-input channel masks must be boolean.")
        mask_array = channel_mask.detach().contiguous().cpu().numpy()
    else:
        mask_array = np.asarray(channel_mask)
        if mask_array.dtype != np.dtype(bool):
            raise ValueError("Base-input channel masks must be boolean.")
    if x_array.ndim != 2 or mask_array.shape != x_array.shape[:1]:
        raise ValueError("One base input must have [channel,time] EEG and [channel] mask.")
    active = x_array[mask_array]
    if not np.isfinite(active).all():
        raise ValueError("Active base-input EEG samples must be finite.")
    canonical_x = np.asarray(x_array, dtype="<f4", order="C")
    canonical_mask = np.asarray(mask_array, dtype=np.uint8, order="C")
    header = json.dumps(
        {
            "schema": "cfeg.metadata-calibration-base-input.v1",
            "x_shape": list(canonical_x.shape),
            "x_dtype": "<f4",
            "channel_mask_shape": list(canonical_mask.shape),
            "channel_mask_dtype": "uint8",
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    digest = hashlib.sha256()
    for value in (header, canonical_x.tobytes(order="C"), canonical_mask.tobytes(order="C")):
        digest.update(len(value).to_bytes(8, byteorder="big", signed=False))
        digest.update(value)
    return digest.hexdigest()


def resolved_context_usage_sha256(
    batches: Sequence[ResolvedMetadataInterventionBatch],
    *,
    expected_row_tokens: Sequence[str],
) -> str:
    """Hash a complete context-usage ledger independently of runtime batching."""

    if not batches:
        raise ValueError("A resolved context-usage ledger cannot be empty.")
    first = batches[0]
    expected_tokens = tuple(str(value) for value in expected_row_tokens)
    if len(set(expected_tokens)) != len(expected_tokens) or not all(
        _ROW_TOKEN.fullmatch(value) for value in expected_tokens
    ):
        raise ValueError("Expected context-usage rows must be unique opaque tokens.")
    records: dict[str, dict[str, object]] = {}
    for batch in batches:
        if batch.mapping_sha256 != first.mapping_sha256 or (
            batch.cell_execution_spec != first.cell_execution_spec
        ) or batch.phase != first.phase:
            raise ValueError("One context-usage ledger cannot mix mappings or cell dispatches.")
        if _resolved_context_batch_sha256(batch) != batch.resolved_context_values_sha256:
            raise ValueError("A resolved metadata batch drifted before ledger aggregation.")
        for row_index in range(len(batch.target_block_keys)):
            ledger_key = batch.row_tokens[row_index]
            record = _resolved_context_record(batch, row_index=row_index)
            if ledger_key in records:
                raise ValueError("A context-usage ledger cannot process one row token twice.")
            records[ledger_key] = record
    if set(records) != set(expected_tokens):
        raise ValueError("Resolved context usage does not cover the exact expected row tokens.")
    payload = {
        "schema": "cfeg.metadata-calibration-context-usage-ledger.v1",
        "mapping_sha256": first.mapping_sha256,
        "cell": dict(first.cell_execution_spec),
        "phase": first.phase,
        "rows": [records[key] for key in sorted(records)],
    }
    return _json_sha256(payload)


def _donor_block(
    *,
    subject_id: str,
    electrode_type: str,
    run_id: str,
    context: MetadataIntervention,
    shuffle_seed: int,
) -> tuple[str, str, str] | None:
    if context in {"exact_off", "all_missing"}:
        return None
    if context == "correct":
        return subject_id, electrode_type, run_id
    if context == "opposite_interface":
        opposite = "wet" if electrode_type == "dry" else "dry"
        return subject_id, opposite, run_id
    if context == "stale":
        block_number = _block_number(run_id)
        if block_number == 1:
            return None
        return subject_id, electrode_type, f"block{block_number - 1:02d}"
    if context == "block_shuffle":
        pool = _SUPPORT_RUNS if run_id in _SUPPORT_RUNS else _QUERY_RUNS
        position = pool.index(run_id)
        token = f"{shuffle_seed}:{subject_id}:{electrode_type}:{pool[0]}:{pool[-1]}"
        shift = 1 + int(hashlib.sha256(token.encode("utf-8")).hexdigest(), 16) % (len(pool) - 1)
        return subject_id, electrode_type, pool[(position + shift) % len(pool)]
    raise RuntimeError("Unreachable intervention context.")


def _validate_mapping_semantics(mapping: MetadataInterventionMapping) -> None:
    if not isinstance(mapping.shuffle_seed, int) or isinstance(mapping.shuffle_seed, bool):
        raise TypeError("Intervention mapping shuffle seed must be one integer.")
    if mapping.context == "block_shuffle" and mapping.shuffle_seed != 20260904:
        raise ValueError("Block-shuffle mapping must use the frozen seed 20260904.")
    for row in mapping.frame.itertuples(index=False):
        expected = _donor_block(
            subject_id=str(row.subject_id),
            electrode_type=str(row.electrode_type),
            run_id=str(row.run_id),
            context=mapping.context,
            shuffle_seed=mapping.shuffle_seed,
        )
        if expected is None:
            donor_values = (
                row.donor_subject_id,
                row.donor_electrode_type,
                row.donor_run_id,
            )
            observed = None if all(pd.isna(value) for value in donor_values) else donor_values
            expected_mode = "off" if mapping.context == "exact_off" else "all_missing"
        else:
            observed = (
                str(row.donor_subject_id),
                str(row.donor_electrode_type),
                str(row.donor_run_id),
            )
            expected_mode = "observed"
        if observed != expected or str(row.metadata_mode) != expected_mode:
            raise ValueError(
                "Intervention mapping donor semantics differ from the declared context rule."
            )


def _block_number(run_id: str) -> int:
    if run_id not in _ALL_RUNS:
        raise ValueError(f"Unknown wearable block {run_id!r}.")
    return int(run_id.removeprefix("block"))


def _resolved_context_batch_sha256(
    resolved: ResolvedMetadataInterventionBatch,
) -> str:
    n_rows = len(resolved.target_block_keys)
    if not (
        len(resolved.electrode_types)
        == len(resolved.metadata_modes)
        == len(resolved.row_tokens)
        == len(resolved.base_input_sha256s)
        == n_rows
        > 0
    ):
        raise ValueError("Resolved metadata row fields must have one nonempty common length.")
    if resolved.impedance_by_channel.shape != (n_rows, 8) or (
        resolved.impedance_missing_by_channel.shape != (n_rows, 8)
    ):
        raise ValueError("Resolved impedance values and masks must have shape [batch,8].")
    if resolved.impedance_by_channel.dtype != np.float32 or (
        resolved.impedance_missing_by_channel.dtype != np.dtype(bool)
    ):
        raise ValueError("Resolved impedance must use exact float32 values and boolean masks.")
    if resolved.batch_kind not in {"query", "support"}:
        raise ValueError("Resolved batch kind must be query or support.")
    if resolved.phase not in {"source_development", "held_participant_evaluation"}:
        raise ValueError("Resolved metadata phase is invalid.")
    if not _SHA256.fullmatch(resolved.mapping_sha256):
        raise ValueError("Resolved mapping binding must be one lowercase SHA-256.")
    available = ~resolved.impedance_missing_by_channel
    available_values = resolved.impedance_by_channel[available]
    if (
        not np.isfinite(available_values).all()
        or (available_values < 0.0).any()
        or (available_values > 2.0).any()
        or np.any(resolved.impedance_by_channel[~available] != 0.0)
    ):
        raise ValueError("Resolved impedance violates its normalized missing-value contract.")
    ablation = dict(resolved.cell_execution_spec).get(
        f"{resolved.batch_kind}_metadata_ablation"
    )
    if ablation not in {"full", "interface_only", "impedance_only"}:
        raise ValueError("Resolved metadata has an unknown feature ablation.")
    for row_index, (mode, interface) in enumerate(
        zip(resolved.metadata_modes, resolved.electrode_types)
    ):
        key = resolved.target_block_keys[row_index]
        if (
            len(key) != 3
            or not key[0]
            or key[1] not in {"dry", "wet"}
            or key[2] not in _ALL_RUNS
        ):
            raise ValueError("Resolved target block keys violate the wearable contract.")
        expected_runs = _QUERY_RUNS if resolved.batch_kind == "query" else _SUPPORT_RUNS
        if key[2] not in expected_runs:
            raise ValueError("Resolved row block is outside its declared support/query pool.")
        expected_prefix = "q_" if resolved.batch_kind == "query" else "s_"
        if (
            not _ROW_TOKEN.fullmatch(resolved.row_tokens[row_index])
            or not resolved.row_tokens[row_index].startswith(expected_prefix)
            or not _SHA256.fullmatch(resolved.base_input_sha256s[row_index])
        ):
            raise ValueError("Resolved row token or base-input binding is invalid.")
        if mode in {"off", "all_missing"}:
            if interface is not None or not resolved.impedance_missing_by_channel[row_index].all():
                raise ValueError("Off/all-missing metadata must null interface and impedance.")
        elif mode == "observed":
            if ablation == "full" and interface not in {"dry", "wet"}:
                raise ValueError("Full observed metadata must include a resolved interface.")
            if ablation == "interface_only" and (
                interface not in {"dry", "wet"}
                or not resolved.impedance_missing_by_channel[row_index].all()
            ):
                raise ValueError("Interface-only metadata must null every impedance value.")
            if ablation == "impedance_only" and interface is not None:
                raise ValueError("Impedance-only metadata must null the interface value.")
        else:
            raise ValueError(f"Unknown resolved metadata mode {mode!r}.")
    rows = [
        {"row_index": row_index, **_resolved_context_record(resolved, row_index=row_index)}
        for row_index in range(n_rows)
    ]
    payload = {
        "schema": "cfeg.metadata-calibration-resolved-context.v1",
        "mapping_sha256": resolved.mapping_sha256,
        "cell": dict(resolved.cell_execution_spec),
        "phase": resolved.phase,
        "batch_kind": resolved.batch_kind,
        "rows": rows,
    }
    return _json_sha256(payload)


def _resolved_context_record(
    resolved: ResolvedMetadataInterventionBatch,
    *,
    row_index: int,
) -> dict[str, object]:
    key = resolved.target_block_keys[row_index]
    return {
        "batch_kind": resolved.batch_kind,
        "target_subject_id": key[0],
        "target_electrode_type": key[1],
        "target_run_id": key[2],
        "row_token": resolved.row_tokens[row_index],
        "base_input_sha256": resolved.base_input_sha256s[row_index],
        "effective_metadata_mode": resolved.metadata_modes[row_index],
        "resolved_electrode_type": resolved.electrode_types[row_index],
        "impedance_by_channel": [
            float(value) for value in resolved.impedance_by_channel[row_index]
        ],
        "impedance_missing_by_channel": [
            bool(value) for value in resolved.impedance_missing_by_channel[row_index]
        ],
    }


def _json_sha256(value: object) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _validated_block_context_lookup(
    sealed_block_context: pd.DataFrame,
) -> dict[tuple[str, str, str], object]:
    required = {
        "dataset_id",
        "subject_id",
        "electrode_type",
        "run_id",
        "impedance_channel_ids",
        "impedance_kohm_by_channel",
    }
    missing = required - set(sealed_block_context.columns)
    extra = set(sealed_block_context.columns) - required
    if missing or extra:
        raise ValueError(
            "Sealed block-context view must use its exact label-free allowlist; "
            f"missing={sorted(missing)}, extra={sorted(extra)}."
        )
    frame = sealed_block_context.loc[:, sorted(required)].copy()
    for column in ("dataset_id", "subject_id", "electrode_type", "run_id"):
        frame[column] = frame[column].astype(str)
    if set(frame["dataset_id"]) != {"wearable"}:
        raise ValueError("Sealed block-context view is frozen to wearable_v3.")
    keys = ["subject_id", "electrode_type", "run_id"]
    if frame.duplicated(keys).any():
        raise ValueError("Sealed block-context view must contain one row per block.")
    lookup: dict[tuple[str, str, str], object] = {}
    for row in frame.itertuples(index=False):
        key = (str(row.subject_id), str(row.electrode_type), str(row.run_id))
        _validate_impedance_channel_ids(row.impedance_channel_ids)
        _model_impedance(row.impedance_kohm_by_channel)
        lookup[key] = row.impedance_kohm_by_channel
    participants = tuple(sorted(frame["subject_id"].unique().tolist()))
    if not participants:
        raise ValueError("Sealed block-context view cannot be empty.")
    for subject_id in participants:
        subject = frame.loc[frame["subject_id"].eq(subject_id)]
        if set(subject["electrode_type"]) != {"dry", "wet"}:
            raise ValueError(
                f"Subject {subject_id} must contain exact dry and wet interfaces."
            )
        for electrode_type in ("dry", "wet"):
            condition = subject.loc[subject["electrode_type"].eq(electrode_type)]
            if set(condition["run_id"]) != set(_ALL_RUNS):
                raise ValueError(
                    "Every sealed participant-interface must contain exact block01-block10."
                )
    return lookup


def _model_impedance(value: object) -> tuple[np.ndarray, np.ndarray]:
    if isinstance(value, np.ndarray):
        value = value.tolist()
    if not isinstance(value, (list, tuple)) or len(value) != 8:
        raise ValueError("Every block impedance vector must have eight channel values.")
    normalized = np.zeros(8, dtype=np.float32)
    missing = np.zeros(8, dtype=bool)
    for index, item in enumerate(value):
        if pd.isna(item):
            missing[index] = True
            continue
        numeric = float(item)
        if not np.isfinite(numeric) or numeric < 0.0:
            raise ValueError("Available block impedance values must be finite and nonnegative.")
        normalized[index] = np.float32(
            np.clip(np.log1p(numeric) / np.log1p(100.0), 0.0, 2.0)
        )
    return normalized, missing


def _validate_impedance_channel_ids(value: object) -> None:
    numeric = np.asarray(value, dtype=np.float64)
    expected = np.asarray(WEARABLE_CALIBRATION_CHANNEL_IDS, dtype=np.float64)
    if (
        numeric.shape != expected.shape
        or not np.isfinite(numeric).all()
        or not np.equal(numeric, np.floor(numeric)).all()
        or not np.array_equal(numeric, expected)
    ):
        raise ValueError(
            "Block impedance must declare the exact official eight-channel canonical order."
        )


def _canonical_metadata_value(value: object) -> str:
    if isinstance(value, np.ndarray):
        value = value.tolist()
    if isinstance(value, (list, tuple)):
        if len(value) != 8:
            raise ValueError("Every block impedance vector must have eight channel values.")
        normalized = [None if pd.isna(item) else float(item) for item in value]
    else:
        raise TypeError("Every block impedance value must be one eight-channel vector.")
    return json.dumps(normalized, separators=(",", ":"), allow_nan=False)
