from __future__ import annotations

import math
from collections.abc import Iterable, Mapping
from typing import Any

import numpy as np
import torch

from cfeg.constants import (
    CATEGORICAL_VOCABS,
    EXTERNAL_CONTINUOUS_SCHEMA_V1,
    METADATA_CONTRACT_LEGACY,
    METADATA_CONTRACT_V04_DEV,
    PROTOCOL_V04_EXTERNAL_CATEGORICAL_FIELDS,
    QUERY_QC_EXTRACTOR_V1,
    UNKNOWN_CATEGORY,
)
from cfeg.data.metadata_controls import (
    DEVELOPMENT_CONTROL_NONE,
    DONOR_CONTROLS,
    normalize_development_control,
)
from cfeg.data.schema import EEGSample


def metadata_collate_kwargs(cfg: Mapping[str, Any]) -> dict[str, str]:
    model_cfg = cfg.get("model", cfg)
    condition_cfg = model_cfg.get("condition_encoder", {})
    protocol_cfg = cfg.get("protocol", {})
    return {
        "metadata_contract_version": str(
            protocol_cfg.get("metadata_contract_version", METADATA_CONTRACT_LEGACY)
        ),
        "external_metadata_mode": str(condition_cfg.get("external_metadata_mode", "observed")),
        "query_qc_extractor_version": str(
            protocol_cfg.get("query_qc_extractor_version", QUERY_QC_EXTRACTOR_V1)
        ),
        "external_continuous_schema": str(
            protocol_cfg.get("external_continuous_schema", EXTERNAL_CONTINUOUS_SCHEMA_V1)
        ),
        "development_control": normalize_development_control(
            protocol_cfg.get("development_control", DEVELOPMENT_CONTROL_NONE)
        ),
    }


def build_vocabularies(
    samples: Iterable[EEGSample | Mapping[str, Any]] | None = None,
) -> dict[str, dict[str, int]]:
    """Build categorical vocabularies with a stable unknown ID.

    Passing samples builds a closed vocabulary from those samples only.  Calling
    without samples preserves the legacy global registry used as a fallback for
    checkpoints created before vocabularies were saved.
    """
    if samples is None:
        values_by_field = CATEGORICAL_VOCABS
    else:
        observed = {field: set() for field in CATEGORICAL_VOCABS}
        for sample in samples:
            for field, values in observed.items():
                value = _sample_field(sample, field)
                values.add(value)
        values_by_field = {
            field: [UNKNOWN_CATEGORY, *sorted(values - {UNKNOWN_CATEGORY})]
            for field, values in observed.items()
        }

    return {
        field: {
            value: index
            for index, value in enumerate(
                [UNKNOWN_CATEGORY, *(v for v in values if v != UNKNOWN_CATEGORY)]
            )
        }
        for field, values in values_by_field.items()
    }


def collate_eeg(
    batch: list[EEGSample],
    vocabularies: Mapping[str, Mapping[str, int]] | None = None,
    categorical_metadata_dropout_prob: float = 0.0,
    metadata_contract_version: str = METADATA_CONTRACT_LEGACY,
    external_metadata_mode: str = "observed",
    query_qc_extractor_version: str = QUERY_QC_EXTRACTOR_V1,
    external_continuous_schema: str = EXTERNAL_CONTINUOUS_SCHEMA_V1,
    development_control: str = DEVELOPMENT_CONTROL_NONE,
    external_metadata_overrides: Mapping[str, EEGSample | Mapping[str, Any]] | None = None,
):
    if not 0.0 <= categorical_metadata_dropout_prob <= 1.0:
        raise ValueError("categorical_metadata_dropout_prob must be between 0 and 1")
    if metadata_contract_version not in {
        METADATA_CONTRACT_LEGACY,
        METADATA_CONTRACT_V04_DEV,
    }:
        raise ValueError(f"Unknown metadata_contract_version={metadata_contract_version!r}.")
    if external_metadata_mode not in {"observed", "null"}:
        raise ValueError(
            f"external_metadata_mode must be 'observed' or 'null', got {external_metadata_mode!r}."
        )
    development_control = normalize_development_control(development_control)
    if development_control != DEVELOPMENT_CONTROL_NONE:
        if metadata_contract_version != METADATA_CONTRACT_V04_DEV:
            raise ValueError("Development controls require metadata_contract_version='0.4-dev'.")
        if external_metadata_mode != "observed":
            raise ValueError("Development controls require external_metadata_mode='observed'.")
    if development_control in DONOR_CONTROLS and external_metadata_overrides is None:
        raise ValueError(f"{development_control} requires a complete external donor mapping.")
    vocabs = build_vocabularies() if vocabularies is None else vocabularies
    # A supplied override map is a declared intervention, never a best-effort hint.
    # Falling back to the target row would silently turn shuffle-train into clean training.
    donor_mapping_required = (
        development_control in DONOR_CONTROLS or external_metadata_overrides is not None
    )
    external_sources = [
        _external_source(sample, external_metadata_overrides, required=donor_mapping_required)
        for sample in batch
    ]
    x = torch.tensor(np.stack([b.x for b in batch]), dtype=torch.float32)
    y = torch.tensor([b.y for b in batch], dtype=torch.long)
    channel_mask = torch.tensor(np.stack([b.channel_mask for b in batch]), dtype=torch.bool)
    channel_ids = torch.tensor(np.stack([b.canonical_channel_ids for b in batch]), dtype=torch.long)
    cont, missing = zip(*[_continuous_features(b, c_max=x.shape[1]) for b in batch])
    external_cont, external_missing = zip(
        *[_external_continuous_features(source) for source in external_sources]
    )
    query_qc, query_qc_missing = zip(*[_query_qc_features(b) for b in batch])
    channel_query_qc, channel_query_qc_missing = zip(
        *[
            _channel_feature(b.query_signal_std_by_channel, x.shape[1], _norm_signal_std)
            for b in batch
        ]
    )
    channel_impedance, channel_impedance_missing = zip(
        *[
            _channel_feature(
                _sample_value(source, "impedance_kohm_by_channel"),
                x.shape[1],
                _norm_impedance,
            )
            for source in external_sources
        ]
    )
    continuous_array = np.stack(cont)
    continuous_missing_array = np.stack(missing)
    external_array = np.stack(external_cont)
    external_missing_array = np.stack(external_missing)
    channel_impedance_array = np.stack(channel_impedance)
    channel_impedance_missing_array = np.stack(channel_impedance_missing)
    if metadata_contract_version == METADATA_CONTRACT_V04_DEV:
        # Keep the legacy tensor for backbone/augmentation compatibility, but
        # scrub every non-structural field so it cannot become a side channel.
        # Indices 0:2 are processed sample rate and active-channel count.
        continuous_array[:, 2:] = 0.0
        continuous_missing_array[:, 2:] = True
        if external_metadata_mode == "null":
            external_array.fill(0.0)
            external_missing_array.fill(True)
            channel_impedance_array.fill(0.0)
            channel_impedance_missing_array.fill(True)
    cond = {
        "channel_ids": channel_ids,
        "channel_mask": channel_mask,
        "continuous": torch.tensor(continuous_array, dtype=torch.float32),
        "continuous_missing": torch.tensor(continuous_missing_array, dtype=torch.bool),
        "sfreq_processed_float": torch.tensor([b.sfreq for b in batch], dtype=torch.float32),
        "external_continuous": torch.tensor(external_array, dtype=torch.float32),
        "external_continuous_missing": torch.tensor(external_missing_array, dtype=torch.bool),
        "query_qc": torch.tensor(np.stack(query_qc), dtype=torch.float32),
        "query_qc_missing": torch.tensor(np.stack(query_qc_missing), dtype=torch.bool),
        "channel_query_qc": torch.tensor(np.stack(channel_query_qc), dtype=torch.float32),
        "channel_query_qc_missing": torch.tensor(
            np.stack(channel_query_qc_missing), dtype=torch.bool
        ),
        "channel_impedance": torch.tensor(channel_impedance_array, dtype=torch.float32),
        "channel_impedance_missing": torch.tensor(
            channel_impedance_missing_array, dtype=torch.bool
        ),
        "metadata_contract_version": metadata_contract_version,
        "external_metadata_mode": external_metadata_mode,
        "query_qc_extractor_version": query_qc_extractor_version,
        "external_continuous_schema": external_continuous_schema,
        "development_control": development_control,
    }
    for field in CATEGORICAL_VOCABS:
        allowed_v04 = field in PROTOCOL_V04_EXTERNAL_CATEGORICAL_FIELDS
        hide_v04 = metadata_contract_version == METADATA_CONTRACT_V04_DEV and (
            external_metadata_mode == "null" or not allowed_v04
        )
        categorical_sources = external_sources if allowed_v04 else batch
        category_values = [_sample_field(source, field) for source in categorical_sources]
        category_ids = (
            torch.zeros(len(batch), dtype=torch.long)
            if hide_v04
            else torch.tensor(
                [
                    (
                        0
                        if value == UNKNOWN_CATEGORY
                        else _presence_category_id(vocabs[field])
                    )
                    for value in category_values
                ],
                dtype=torch.long,
            )
            if development_control == "missingness_only" and allowed_v04
            else torch.tensor(
                [_category_id(value, vocabs[field]) for value in category_values],
                dtype=torch.long,
            )
        )
        if categorical_metadata_dropout_prob > 0.0 and (
            metadata_contract_version == METADATA_CONTRACT_LEGACY or allowed_v04
        ):
            drop = torch.rand(category_ids.shape) < categorical_metadata_dropout_prob
            category_ids = category_ids.masked_fill(drop, 0)
        cond[field] = category_ids
    _apply_development_input_control(x, cond, development_control)
    return {
        "x": x,
        "y": y,
        "sample_id": [b.sample_id for b in batch],
        "split_meta": {
            "dataset_id_str": [b.dataset_id for b in batch],
            "subject_id": [b.subject_id for b in batch],
            "session_id": [b.session_id for b in batch],
        },
        "cond": cond,
    }


def _sample_field(sample: EEGSample | Mapping[str, Any], field: str) -> str:
    value = _sample_value(sample, field)
    if field == "dataset_id":
        return UNKNOWN_CATEGORY if _is_missing(value) else str(value)
    if field == "reattach_flag":
        if _is_missing(value):
            return UNKNOWN_CATEGORY
        if isinstance(value, str):
            normalized = value.strip().lower()
            if normalized in {"true", "1"}:
                return "true"
            if normalized in {"false", "0"}:
                return "false"
        return "true" if bool(value) else "false"
    return UNKNOWN_CATEGORY if _is_missing(value) else str(value)


def _is_missing(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, str):
        return value.strip().lower() in {"", "nan", "<na>", "nat"}
    try:
        return bool(np.isnan(value))
    except (TypeError, ValueError):
        return str(value).strip().lower() in {"nan", "<na>", "nat"}


def _category_id(value: str, vocab: Mapping[str, int]) -> int:
    return int(vocab.get(value, vocab.get(UNKNOWN_CATEGORY, 0)))


def _presence_category_id(vocab: Mapping[str, int]) -> int:
    known = sorted(int(value) for key, value in vocab.items() if key != UNKNOWN_CATEGORY)
    return known[0] if known else int(vocab.get(UNKNOWN_CATEGORY, 0))


def _continuous_features(sample: EEGSample, c_max: int) -> tuple[np.ndarray, np.ndarray]:
    values = [
        sample.sfreq / 250.0,
        math.log1p(sample.n_channels_used) / math.log1p(c_max),
        _norm_impedance(sample.impedance_mean_kohm),
        _norm_impedance(sample.impedance_max_kohm),
        _norm_time(sample.time_since_last_session_hours),
    ]
    missing = [
        False,
        False,
        sample.impedance_mean_kohm is None,
        sample.impedance_max_kohm is None,
        sample.time_since_last_session_hours is None,
    ]
    arr = np.asarray([0.0 if m else v for v, m in zip(values, missing)], dtype=np.float32)
    return arr, np.asarray(missing, dtype=bool)


def _external_continuous_features(
    sample: EEGSample | Mapping[str, Any],
) -> tuple[np.ndarray, np.ndarray]:
    impedance_mean = _sample_value(sample, "impedance_mean_kohm")
    impedance_max = _sample_value(sample, "impedance_max_kohm")
    impedance_mean = None if _is_missing(impedance_mean) else float(impedance_mean)
    impedance_max = None if _is_missing(impedance_max) else float(impedance_max)
    values = [
        _norm_impedance(impedance_mean),
        _norm_impedance(impedance_max),
    ]
    missing = [
        impedance_mean is None,
        impedance_max is None,
    ]
    array = np.asarray(
        [0.0 if is_missing else value for value, is_missing in zip(values, missing)],
        dtype=np.float32,
    )
    return array, np.asarray(missing, dtype=bool)


def overwrite_external_metadata(
    cond: dict[str, Any],
    sample_ids: list[str],
    external_metadata_overrides: Mapping[str, EEGSample | Mapping[str, Any]],
    vocabularies: Mapping[str, Mapping[str, int]],
) -> dict[str, Any]:
    """Exchange the v0.4 external bundle without loading or changing donor EEG."""
    sources = []
    for sample_id in sample_ids:
        if sample_id not in external_metadata_overrides:
            raise KeyError(f"External donor mapping has no row for sample_id={sample_id!r}.")
        sources.append(external_metadata_overrides[sample_id])
    out = {key: value.clone() if torch.is_tensor(value) else value for key, value in cond.items()}
    device = cond["external_continuous"].device
    external, missing = zip(*[_external_continuous_features(source) for source in sources])
    out["external_continuous"] = torch.as_tensor(
        np.stack(external), dtype=torch.float32, device=device
    )
    out["external_continuous_missing"] = torch.as_tensor(
        np.stack(missing), dtype=torch.bool, device=device
    )
    c_max = int(cond["channel_impedance"].shape[1])
    channel_values, channel_missing = zip(
        *[
            _channel_feature(
                _sample_value(source, "impedance_kohm_by_channel"),
                c_max,
                _norm_impedance,
            )
            for source in sources
        ]
    )
    out["channel_impedance"] = torch.as_tensor(
        np.stack(channel_values), dtype=torch.float32, device=device
    )
    out["channel_impedance_missing"] = torch.as_tensor(
        np.stack(channel_missing), dtype=torch.bool, device=device
    )
    for field in PROTOCOL_V04_EXTERNAL_CATEGORICAL_FIELDS:
        out[field] = torch.as_tensor(
            [_category_id(_sample_field(source, field), vocabularies[field]) for source in sources],
            dtype=torch.long,
            device=device,
        )
    return out


def _external_source(
    sample: EEGSample,
    overrides: Mapping[str, EEGSample | Mapping[str, Any]] | None,
    *,
    required: bool,
) -> EEGSample | Mapping[str, Any]:
    if overrides is None:
        return sample
    source = overrides.get(sample.sample_id)
    if source is None:
        if required:
            raise KeyError(f"External donor mapping has no row for sample_id={sample.sample_id!r}.")
        return sample
    return source


def _sample_value(sample: EEGSample | Mapping[str, Any], field: str) -> Any:
    return sample.get(field) if isinstance(sample, Mapping) else getattr(sample, field, None)


def _apply_development_input_control(
    x: torch.Tensor,
    cond: dict[str, Any],
    development_control: str,
) -> None:
    if development_control not in {"metadata_only", "missingness_only"}:
        return
    # Both shortcut classifiers receive no waveform, structural channel identity,
    # time-grid value, or query-derived QC. Only the v0.4 external bundle remains.
    x.zero_()
    cond["channel_ids"].zero_()
    cond["channel_mask"].zero_()
    cond["continuous"].zero_()
    cond["continuous_missing"].fill_(True)
    cond["sfreq_processed_float"].zero_()
    cond["query_qc"].zero_()
    cond["query_qc_missing"].fill_(True)
    cond["channel_query_qc"].zero_()
    cond["channel_query_qc_missing"].fill_(True)
    if development_control == "missingness_only":
        cond["external_continuous"].zero_()
        cond["channel_impedance"].zero_()


def _query_qc_features(sample: EEGSample) -> tuple[np.ndarray, np.ndarray]:
    missing = sample.query_signal_std is None
    value = 0.0 if missing else _norm_signal_std(sample.query_signal_std)
    return np.asarray([value], dtype=np.float32), np.asarray([missing], dtype=bool)


def _channel_feature(
    values: np.ndarray | None,
    c_max: int,
    normalize,
) -> tuple[np.ndarray, np.ndarray]:
    source = np.full((c_max,), np.nan, dtype=np.float32)
    if values is not None:
        array = np.asarray(values, dtype=np.float32).reshape(-1)
        source[: min(c_max, len(array))] = array[:c_max]
    missing = ~np.isfinite(source)
    normalized = np.asarray(
        [
            0.0 if is_missing else normalize(float(value))
            for value, is_missing in zip(source, missing)
        ],
        dtype=np.float32,
    )
    return normalized, missing


def _norm_impedance(value: float | None) -> float:
    if value is None:
        return 0.0
    return float(np.clip(np.log1p(value) / np.log1p(100.0), 0.0, 2.0))


def _norm_signal_std(value: float | None) -> float:
    if value is None:
        return 0.0
    return float(np.clip(np.log1p(max(float(value), 0.0)) / np.log1p(100.0), 0.0, 2.0))


def _norm_time(value: float | None) -> float:
    if value is None:
        return 0.0
    return float(np.clip(np.log1p(value) / np.log1p(24.0 * 30.0), 0.0, 2.0))
