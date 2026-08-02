from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from torch.utils.data import Dataset

from cfeg.data.io_hdf5 import read_sample
from cfeg.data.schema import EEGSample, load_manifest


class EEGProcessedDataset(Dataset):
    def __init__(self, processed_dirs: list[str | Path], indices: list[int] | None = None):
        self.roots = [Path(p) for p in processed_dirs]
        self.entries: list[tuple[Path, int, dict]] = []
        for root in self.roots:
            manifest = load_manifest(root)
            _validate_manifest_class_map(root, manifest)
            for _, row in manifest.iterrows():
                self.entries.append((root, int(row["h5_index"]), row.to_dict()))
        if indices is not None:
            self.entries = [self.entries[i] for i in indices]
        self.class_map = self._load_first_class_map()

    def _load_first_class_map(self) -> dict:
        reference = None
        for root in self.roots:
            path = root / "class_map.json"
            if path.exists():
                with path.open("r", encoding="utf-8") as f:
                    current = json.load(f)
                if reference is None:
                    reference = current
                    continue
                for label in set(reference) & set(current):
                    left = float(reference[label]["stimulus_frequency_hz"])
                    right = float(current[label]["stimulus_frequency_hz"])
                    if abs(left - right) > 1e-4:
                        raise ValueError(
                            f"Class {label} has conflicting frequencies across processed datasets: "
                            f"{left:g} vs {right:g} Hz. Re-run preparation with canonical frequency "
                            "label alignment. Run `bash scripts/cfeg.sh migrate-labels --apply` "
                            "for legacy Wang/BETA artifacts."
                        )
                reference = {**reference, **current}
        return reference or {}

    def __len__(self) -> int:
        return len(self.entries)

    def __getitem__(self, index: int) -> EEGSample:
        root, h5_index, row = self.entries[index]
        x, mask, y = read_sample(root / "signals.h5", h5_index)
        ids = _to_int_array(row.get("canonical_channel_ids"), length=x.shape[0], mask=mask)
        return EEGSample(
            x=x,
            y=y,
            sample_id=str(row["sample_id"]),
            dataset_id=str(row.get("dataset_id", "unknown")),
            subject_id=str(row.get("subject_id", "unknown")),
            session_id=str(row.get("session_id", "unknown")),
            channel_mask=mask,
            canonical_channel_ids=ids,
            sfreq=float(row.get("sfreq_processed", 0.0)),
            reference=_none_if_nan(row.get("reference")),
            hardware_id=_none_if_nan(row.get("hardware_id")),
            electrode_type=_none_if_nan(row.get("electrode_type")),
            cap_type=_none_if_nan(row.get("cap_type")),
            n_channels_used=int(row.get("n_channels_used", int(mask.sum()))),
            impedance_mean_kohm=_float_or_none(row.get("impedance_mean_kohm")),
            impedance_max_kohm=_float_or_none(row.get("impedance_max_kohm")),
            reattach_flag=_bool_or_none(row.get("reattach_flag")),
            time_since_last_session_hours=_float_or_none(row.get("time_since_last_session_hours")),
        )


def _validate_manifest_class_map(root: Path, manifest) -> None:
    """Reject stale processed assets whose integer labels no longer mean the same frequency."""
    path = root / "class_map.json"
    if not path.exists():
        return
    with path.open("r", encoding="utf-8") as handle:
        class_map = json.load(handle)
    for label, rows in manifest.groupby("label"):
        entry = class_map.get(str(int(label)))
        if entry is None:
            raise ValueError(
                f"Processed dataset {root} uses label {int(label)} but class_map.json has no entry. "
                "Re-run dataset preparation."
            )
        expected = float(entry["stimulus_frequency_hz"])
        observed = rows["stimulus_frequency_hz"].astype(float).to_numpy()
        if not np.allclose(observed, expected, atol=1e-4, rtol=0.0):
            examples = sorted(set(float(value) for value in observed))[:5]
            raise ValueError(
                f"Processed dataset {root} maps label {int(label)} to {examples}, but "
                f"class_map.json declares {expected:g} Hz. Restore canonical frequency alignment "
                "with `bash scripts/cfeg.sh migrate-labels --apply` for legacy Wang/BETA "
                "artifacts or re-run dataset preparation."
            )


def _none_if_nan(value):
    if value is None:
        return None
    try:
        if np.isnan(value):
            return None
    except TypeError:
        pass
    return value


def _float_or_none(value) -> float | None:
    value = _none_if_nan(value)
    return None if value is None else float(value)


def _bool_or_none(value) -> bool | None:
    value = _none_if_nan(value)
    if value is None:
        return None
    if isinstance(value, str):
        if value.lower() in {"true", "1"}:
            return True
        if value.lower() in {"false", "0"}:
            return False
        return None
    return bool(value)


def _to_int_array(value, length: int, mask: np.ndarray | None = None) -> np.ndarray:
    if isinstance(value, np.ndarray):
        arr = value.astype(np.int64)
    elif isinstance(value, list):
        arr = np.asarray(value, dtype=np.int64)
    else:
        arr = np.zeros((0,), dtype=np.int64)
    out = np.zeros((length,), dtype=np.int64)
    n = min(length, len(arr))
    out[:n] = arr[:n]
    if mask is not None and (len(arr) != length or ((out > 0) != mask).any()):
        out = (np.arange(length, dtype=np.int64) + 1) * mask.astype(np.int64)
    return out
