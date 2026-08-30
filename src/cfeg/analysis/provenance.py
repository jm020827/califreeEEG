from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

PRIMARY_ANALYSIS_MANIFEST_SCHEMA = "cfeg.primary-analysis-manifest.v1"
PRIMARY_ANALYSIS_MANIFEST_COLUMNS = (
    "sample_id",
    "dataset_id",
    "subject_id",
    "session_id",
    "run_id",
    "trial_id",
    "label",
    "stimulus_frequency_hz",
    "stimulus_phase_rad",
    "sfreq_original",
    "sfreq_processed",
    "window_start_sec",
    "window_duration_sec",
    "reference",
    "hardware_id",
    "cap_type",
    "electrode_type",
    "n_channels_original",
    "n_channels_used",
    "impedance_mean_kohm",
    "impedance_max_kohm",
    "reattach_flag",
    "time_since_last_session_hours",
    "environment_note_code",
    "headband_order",
    "condition_period",
)


@dataclass(frozen=True)
class VerifiedPredictionBundle:
    frame: pd.DataFrame
    artifact: dict[str, object]
    path: Path
    provenance_path: Path | None = None
    provenance_sha256: str | None = None


def primary_analysis_manifest_sha256(manifest: pd.DataFrame, sample_ids: Iterable[str]) -> str:
    missing = [
        column for column in PRIMARY_ANALYSIS_MANIFEST_COLUMNS if column not in manifest.columns
    ]
    if missing:
        raise ValueError(f"Manifest missing primary analysis columns: {missing}")
    normalized_ids = [str(value) for value in sample_ids]
    if len(normalized_ids) != len(set(normalized_ids)):
        raise ValueError("Primary analysis sample IDs must be unique.")
    table = manifest.copy()
    table["sample_id"] = table["sample_id"].astype(str)
    if table["sample_id"].duplicated().any():
        raise ValueError("Manifest sample_id values must be unique for provenance hashing.")
    selected = table.loc[
        table["sample_id"].isin(normalized_ids), list(PRIMARY_ANALYSIS_MANIFEST_COLUMNS)
    ].copy()
    missing_ids = sorted(set(normalized_ids) - set(selected["sample_id"]))
    if missing_ids:
        raise ValueError(
            "Manifest is missing prediction samples required for provenance hashing: "
            f"{missing_ids[:5]}"
        )
    selected = selected.sort_values("sample_id")
    rows = [
        {column: _normalize_scalar(value) for column, value in zip(selected.columns, row)}
        for row in selected.itertuples(index=False, name=None)
    ]
    payload = {
        "schema": PRIMARY_ANALYSIS_MANIFEST_SCHEMA,
        "columns": list(PRIMARY_ANALYSIS_MANIFEST_COLUMNS),
        "rows": rows,
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode(
        "utf-8"
    )
    return hashlib.sha256(encoded).hexdigest()


def _normalize_scalar(value):
    if isinstance(value, np.generic):
        value = value.item()
    if value is None:
        return None
    try:
        if bool(pd.isna(value)):
            return None
    except (TypeError, ValueError):
        pass
    if isinstance(value, (bool, int, float, str)):
        return value
    return str(value)
