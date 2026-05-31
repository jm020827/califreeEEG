from __future__ import annotations

import math
from typing import Mapping

import numpy as np
import torch

from cfeg.constants import CATEGORICAL_VOCABS
from cfeg.data.schema import EEGSample


def build_vocabularies(samples: list[EEGSample] | None = None) -> dict[str, dict[str, int]]:
    vocabs = {name: {v: i for i, v in enumerate(values)} for name, values in CATEGORICAL_VOCABS.items()}
    if samples:
        for sample in samples:
            for field in vocabs:
                value = _sample_field(sample, field)
                if value not in vocabs[field]:
                    vocabs[field][value] = len(vocabs[field])
    return vocabs


def collate_eeg(batch: list[EEGSample], vocabularies: Mapping[str, Mapping[str, int]] | None = None):
    vocabs = vocabularies or build_vocabularies()
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
        cond[field] = torch.tensor(
            [_category_id(_sample_field(b, field), vocabs[field]) for b in batch],
            dtype=torch.long,
        )
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


def _sample_field(sample: EEGSample, field: str) -> str:
    if field == "dataset_id":
        return sample.dataset_id or "unknown"
    if field == "reattach_flag":
        if sample.reattach_flag is None:
            return "unknown"
        return "true" if sample.reattach_flag else "false"
    value = getattr(sample, field, None)
    return str(value) if value not in {None, "", "nan"} else "unknown"


def _category_id(value: str, vocab: Mapping[str, int]) -> int:
    return int(vocab.get(value, vocab.get("unknown", 0)))


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

