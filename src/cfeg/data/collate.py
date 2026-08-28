from __future__ import annotations

import math
from collections.abc import Iterable, Mapping
from typing import Any

import numpy as np
import torch

from cfeg.constants import CATEGORICAL_VOCABS, UNKNOWN_CATEGORY
from cfeg.data.schema import EEGSample


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
                values.add(_sample_field(sample, field))
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
):
    if not 0.0 <= categorical_metadata_dropout_prob <= 1.0:
        raise ValueError("categorical_metadata_dropout_prob must be between 0 and 1")
    vocabs = build_vocabularies() if vocabularies is None else vocabularies
    x = torch.tensor(np.stack([b.x for b in batch]), dtype=torch.float32)
    y = torch.tensor([b.y for b in batch], dtype=torch.long)
    channel_mask = torch.tensor(np.stack([b.channel_mask for b in batch]), dtype=torch.bool)
    channel_ids = torch.tensor(np.stack([b.canonical_channel_ids for b in batch]), dtype=torch.long)
    cont, missing = zip(*[_continuous_features(b, c_max=x.shape[1]) for b in batch])
    cond = {
        "channel_ids": channel_ids,
        "channel_mask": channel_mask,
        "continuous": torch.tensor(np.stack(cont), dtype=torch.float32),
        "continuous_missing": torch.tensor(np.stack(missing), dtype=torch.bool),
        "sfreq_processed_float": torch.tensor([b.sfreq for b in batch], dtype=torch.float32),
    }
    for field in CATEGORICAL_VOCABS:
        category_ids = torch.tensor(
            [_category_id(_sample_field(b, field), vocabs[field]) for b in batch],
            dtype=torch.long,
        )
        if categorical_metadata_dropout_prob > 0.0:
            drop = torch.rand(category_ids.shape) < categorical_metadata_dropout_prob
            category_ids = category_ids.masked_fill(drop, 0)
        cond[field] = category_ids
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
    value = sample.get(field) if isinstance(sample, Mapping) else getattr(sample, field, None)
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


def _norm_impedance(value: float | None) -> float:
    if value is None:
        return 0.0
    return float(np.clip(np.log1p(value) / np.log1p(100.0), 0.0, 2.0))


def _norm_time(value: float | None) -> float:
    if value is None:
        return 0.0
    return float(np.clip(np.log1p(value) / np.log1p(24.0 * 30.0), 0.0, 2.0))
