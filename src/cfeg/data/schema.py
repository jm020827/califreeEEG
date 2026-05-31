from __future__ import annotations

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
    manifest.to_json(root / "manifest.jsonl", orient="records", lines=True)
    try:
        manifest.to_parquet(root / "manifest.parquet", index=False)
    except Exception as exc:
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

