from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

REQUIRED_MANIFEST_COLUMNS = [
    "sample_id",
    "h5_index",
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
    "channel_names_original",
    "channel_names_used",
    "canonical_channel_ids",
    "impedance_mean_kohm",
    "impedance_max_kohm",
    "reattach_flag",
    "time_since_last_session_hours",
    "environment_note_code",
    "source_file",
]

# These columns preserve information needed by Protocol 0.4-dev without making
# older processed datasets unreadable.  They are intentionally not all model
# inputs: headband order and condition period are analysis covariates only.
OPTIONAL_MANIFEST_COLUMNS = [
    "query_signal_std",
    "query_signal_std_by_channel",
    "impedance_kohm_by_channel",
    "headband_order",
    "condition_period",
]


@dataclass
class EEGSample:
    x: np.ndarray
    y: int
    sample_id: str
    dataset_id: str
    subject_id: str
    session_id: str
    channel_mask: np.ndarray
    canonical_channel_ids: np.ndarray
    sfreq: float
    reference: str | None
    hardware_id: str | None
    electrode_type: str | None
    cap_type: str | None
    n_channels_used: int
    impedance_mean_kohm: float | None
    impedance_max_kohm: float | None
    reattach_flag: bool | None
    time_since_last_session_hours: float | None
    query_signal_std: float | None = None
    query_signal_std_by_channel: np.ndarray | None = None
    impedance_kohm_by_channel: np.ndarray | None = None
    headband_order: str | None = None
    condition_period: str | None = None


def ordered_manifest_columns(rows: list[dict[str, Any]]) -> list[str]:
    """Return a stable schema while retaining versioned optional metadata."""
    present = {key for row in rows for key in row}
    optional = [column for column in OPTIONAL_MANIFEST_COLUMNS if column in present]
    extras = sorted(present - set(REQUIRED_MANIFEST_COLUMNS) - set(optional))
    return [*REQUIRED_MANIFEST_COLUMNS, *optional, *extras]


def nullable_vector(values: np.ndarray | list[float]) -> list[float | None]:
    """Serialize aligned numeric vectors without emitting non-standard NaN JSON."""
    array = np.asarray(values, dtype=float).reshape(-1)
    return [float(value) if np.isfinite(value) else None for value in array]


def validate_manifest(manifest: pd.DataFrame) -> None:
    missing = [c for c in REQUIRED_MANIFEST_COLUMNS if c not in manifest.columns]
    if missing:
        raise ValueError(f"Manifest missing required columns: {missing}")
    if manifest["sample_id"].duplicated().any():
        raise ValueError("Manifest sample_id values must be unique")
    if not (manifest["h5_index"].astype(int).to_numpy() == np.arange(len(manifest))).all():
        raise ValueError("Manifest h5_index must be contiguous and match row order")


def load_manifest(processed_dir: str | Path) -> pd.DataFrame:
    root = Path(processed_dir)
    parquet = root / "manifest.parquet"
    jsonl = root / "manifest.jsonl"
    if parquet.exists():
        try:
            return pd.read_parquet(parquet)
        except Exception:
            if not jsonl.exists():
                raise
    if jsonl.exists():
        return pd.read_json(jsonl, lines=True)
    raise FileNotFoundError(f"No manifest.parquet or manifest.jsonl found under {root}")


def write_manifest(manifest: pd.DataFrame, processed_dir: str | Path) -> None:
    root = Path(processed_dir)
    root.mkdir(parents=True, exist_ok=True)
    jsonl = root / "manifest.jsonl"
    jsonl_tmp = root / "manifest.jsonl.tmp"
    manifest.to_json(jsonl_tmp, orient="records", lines=True)
    os.replace(jsonl_tmp, jsonl)
    parquet = root / "manifest.parquet"
    parquet_tmp = root / "manifest.parquet.tmp"
    try:
        manifest.to_parquet(parquet_tmp, index=False)
        os.replace(parquet_tmp, parquet)
    # Parquet support is optional and backend failures are not standardized.
    except Exception as exc:  # noqa: BLE001
        parquet_tmp.unlink(missing_ok=True)
        # Never leave an older Parquet manifest shadowing the new JSONL.
        parquet.unlink(missing_ok=True)
        print(f"Warning: failed to write manifest.parquet ({exc}); manifest.jsonl was written.")


def nullable_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        if np.isnan(value):
            return None
    except TypeError:
        pass
    return float(value)
