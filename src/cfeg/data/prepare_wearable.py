from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from cfeg.assets.errors import MissingAssetError
from cfeg.data.io_hdf5 import write_processed_hdf5
from cfeg.data.label_mapping import write_class_map
from cfeg.data.prepare_mat import _load_arrays
from cfeg.data.preprocess import CanonicalChannelMap, PreprocessConfig, preprocess_trial
from cfeg.data.schema import REQUIRED_MANIFEST_COLUMNS, validate_manifest, write_manifest


def prepare(raw_dir: Path, out_dir: Path, cfg: dict) -> None:
    """Prepare the Zhu et al. 102-subject wet/dry wearable SSVEP dataset."""
    if not raw_dir.exists():
        raise MissingAssetError(
            f"Wearable SSVEP raw_dir does not exist: {raw_dir}\n"
            "Download Figshare 13560281 into EEG_DATA_ROOT/raw/wearable. Expected files "
            "include S001.mat ... S102.mat and Impedance.mat."
        )
    subject_files = sorted(
        path for path in raw_dir.rglob("*.mat") if re.fullmatch(r"S\d{3}", path.stem, re.IGNORECASE)
    )
    if not subject_files:
        raise MissingAssetError(
            f"No S###.mat files found under {raw_dir}. Preserve the original wearable dataset names."
        )

    pcfg = PreprocessConfig.from_dict(cfg.get("preprocess"))
    cmap = CanonicalChannelMap.from_yaml()
    channel_names = list(cfg["channel_names"])
    frequencies = [float(value) for value in cfg["class_frequencies"]]
    phases = [float(value) for value in cfg.get("class_phases", [0.0] * len(frequencies))]
    electrode_types = list(cfg.get("electrode_types", ["dry", "wet"]))
    impedance = _load_impedance(raw_dir)

    xs: list[np.ndarray] = []
    masks: list[np.ndarray] = []
    ys: list[int] = []
    rows: list[dict[str, Any]] = []
    for file in subject_files:
        subject_index = int(re.search(r"(\d+)", file.stem).group(1)) - 1
        data = _wearable_data(_load_arrays(file))
        for electrode_index, electrode_type in enumerate(electrode_types):
            for block_index in range(data.shape[3]):
                imp = _impedance_for(impedance, subject_index, electrode_index, block_index)
                for target_index, frequency in enumerate(frequencies):
                    raw = data[:, :, electrode_index, block_index, target_index]
                    placed, mask, _ids, sfreq_processed = preprocess_trial(
                        raw, channel_names, float(cfg.get("raw_sfreq", 250.0)), pcfg, cmap
                    )
                    h5_index = len(xs)
                    slot_ids = ((np.arange(pcfg.c_max) + 1) * mask.astype(np.int64)).tolist()
                    xs.append(placed)
                    masks.append(mask)
                    ys.append(target_index)
                    rows.append(
                        {
                            "sample_id": (
                                f"wearable_sub{subject_index + 1:03d}_{electrode_type}_"
                                f"block{block_index + 1:02d}_target{target_index:02d}"
                            ),
                            "h5_index": h5_index,
                            "dataset_id": "wearable",
                            "subject_id": f"sub{subject_index + 1:03d}",
                            "session_id": electrode_type,
                            "run_id": f"block{block_index + 1:02d}",
                            "trial_id": f"target{target_index:02d}",
                            "label": target_index,
                            "stimulus_frequency_hz": frequency,
                            "stimulus_phase_rad": phases[target_index],
                            "sfreq_original": float(cfg.get("raw_sfreq", 250.0)),
                            "sfreq_processed": sfreq_processed,
                            "window_start_sec": pcfg.window_start_sec,
                            "window_duration_sec": pcfg.window_duration_sec,
                            "reference": cfg.get("reference", "forehead"),
                            "hardware_id": cfg.get("hardware_id", "neuracle_neusenw"),
                            "cap_type": "wearable",
                            "electrode_type": electrode_type,
                            "n_channels_original": len(channel_names),
                            "n_channels_used": int(mask.sum()),
                            "channel_names_original": channel_names,
                            "channel_names_used": channel_names,
                            "canonical_channel_ids": slot_ids,
                            "impedance_mean_kohm": None if imp is None else float(np.mean(imp)),
                            "impedance_max_kohm": None if imp is None else float(np.max(imp)),
                            "reattach_flag": block_index == 0,
                            "time_since_last_session_hours": None,
                            "environment_note_code": "unknown",
                            "source_file": str(file),
                        }
                    )
        print(f"prepared {file} trials={data.shape[2] * data.shape[3] * data.shape[4]}")

    out_dir.mkdir(parents=True, exist_ok=True)
    write_processed_hdf5(out_dir, np.stack(xs), np.stack(masks), np.asarray(ys, dtype=np.int64))
    manifest = pd.DataFrame(rows, columns=REQUIRED_MANIFEST_COLUMNS)
    validate_manifest(manifest)
    write_manifest(manifest, out_dir)
    write_class_map(frequencies, out_dir)
    with (out_dir / "preprocess_config.yaml").open("w", encoding="utf-8") as handle:
        yaml.safe_dump(pcfg.__dict__, handle, sort_keys=False)
    with (out_dir / "asset_info.json").open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "dataset_id": "wearable",
                "raw_dir": str(raw_dir),
                "processed_dir": str(out_dir),
                "created_by": "scripts/prepare_dataset.py",
                "source": "Figshare 13560281",
                "schema": "[channel,time,electrode,block,target]",
                "electrode_types": electrode_types,
                "has_impedance": impedance is not None,
            },
            handle,
            indent=2,
        )


def _wearable_data(arrays: dict[str, Any]) -> np.ndarray:
    candidates = [
        np.asarray(value)
        for key, value in arrays.items()
        if key.rsplit(".", 1)[-1].lower() == "data" and np.asarray(value).ndim == 5
    ]
    if not candidates:
        candidates = [np.asarray(value) for value in arrays.values() if np.asarray(value).ndim == 5]
    if not candidates:
        shapes = sorted({tuple(np.asarray(value).shape) for value in arrays.values()})
        raise ValueError(f"Wearable MAT has no 5-D EEG data array. Available shapes: {shapes}")
    data = max(candidates, key=lambda value: value.size)
    axes = [_axis_for_size(data.shape, size) for size in (8, 710, 2, 10, 12)]
    if len(set(axes)) != 5:
        raise ValueError(f"Cannot identify wearable axes in shape {data.shape}")
    return np.moveaxis(data, axes, range(5)).astype(np.float32)


def _load_impedance(raw_dir: Path) -> np.ndarray | None:
    paths = [path for path in raw_dir.rglob("*.mat") if path.stem.lower() == "impedance"]
    if not paths:
        return None
    arrays = _load_arrays(paths[0])
    candidates = [np.asarray(value) for value in arrays.values() if np.asarray(value).ndim == 4]
    if not candidates:
        return None
    data = max(candidates, key=lambda value: value.size)
    axes = [_axis_for_size(data.shape, size) for size in (8, 10, 2, 102)]
    if len(set(axes)) != 4:
        return None
    return np.moveaxis(data, axes, range(4)).astype(np.float32)


def _axis_for_size(shape: tuple[int, ...], size: int) -> int:
    matches = [index for index, value in enumerate(shape) if value == size]
    if len(matches) != 1:
        raise ValueError(f"Expected exactly one axis of size {size} in {shape}")
    return matches[0]


def _impedance_for(
    impedance: np.ndarray | None, subject: int, electrode: int, block: int
) -> np.ndarray | None:
    if impedance is None or subject >= impedance.shape[3]:
        return None
    values = impedance[:, block, electrode, subject]
    values = values[np.isfinite(values)]
    return values if values.size else None
