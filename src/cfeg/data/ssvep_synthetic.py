from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from cfeg.data.io_hdf5 import write_processed_hdf5
from cfeg.data.label_mapping import write_class_map
from cfeg.data.preprocess import CanonicalChannelMap, PreprocessConfig, normalize_trial, place_on_canonical_channels
from cfeg.data.schema import REQUIRED_MANIFEST_COLUMNS, validate_manifest, write_manifest


DEFAULT_FREQS = [8.0, 10.0, 12.0, 15.0, 9.0, 11.0, 13.0, 14.0]
DEFAULT_CHANNELS = ["Pz", "PO3", "PO4", "POz", "PO7", "O1", "Oz", "O2"]


def generate_synthetic_processed(
    out_dir: str | Path,
    n_subjects: int = 8,
    n_trials_per_class: int = 20,
    n_classes: int = 4,
    target_sfreq: float = 200.0,
    duration_sec: float = 2.0,
    c_max: int = 64,
    seed: int = 42,
) -> dict[str, int | str]:
    rng = np.random.default_rng(seed)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    freqs = DEFAULT_FREQS[:n_classes]
    n_samples = int(round(target_sfreq * duration_sec))
    t = np.arange(n_samples, dtype=np.float32) / float(target_sfreq)
    cmap = CanonicalChannelMap.from_yaml()
    xs: list[np.ndarray] = []
    masks: list[np.ndarray] = []
    ys: list[int] = []
    rows: list[dict] = []
    for subject in range(n_subjects):
        subject_phase = rng.uniform(0, 2 * np.pi)
        subject_noise = rng.uniform(0.15, 0.45)
        subject_amp = rng.uniform(0.8, 1.4)
        for label, freq in enumerate(freqs):
            for trial in range(n_trials_per_class):
                session = trial % 2
                drift = rng.normal(0.0, 0.03)
                channel_rows = []
                for ch_i, _ch in enumerate(DEFAULT_CHANNELS):
                    occipital_gain = 1.0 + 0.2 * (ch_i >= 3)
                    phase = subject_phase + rng.normal(0.0, 0.15) + ch_i * 0.03
                    sig = (
                        np.sin(2 * np.pi * (freq + drift) * t + phase)
                        + 0.45 * np.sin(2 * np.pi * 2 * freq * t + phase / 2)
                        + 0.20 * np.sin(2 * np.pi * 3 * freq * t + phase / 3)
                    )
                    sig = subject_amp * occipital_gain * sig
                    sig += rng.normal(0.0, subject_noise, size=n_samples)
                    channel_rows.append(sig.astype(np.float32))
                raw = np.stack(channel_rows, axis=0)
                if rng.random() < 0.08:
                    raw[rng.integers(0, len(DEFAULT_CHANNELS))] = 0.0
                raw = normalize_trial(raw)
                placed, mask, _ids = place_on_canonical_channels(raw, DEFAULT_CHANNELS, cmap, c_max)
                slot_ids = ((np.arange(c_max) + 1) * mask.astype(np.int64)).tolist()
                h5_index = len(xs)
                sample_id = f"synthetic_sub{subject:03d}_ses{session:02d}_cls{label:02d}_tr{trial:03d}"
                xs.append(placed)
                masks.append(mask)
                ys.append(label)
                rows.append(
                    {
                        "sample_id": sample_id,
                        "h5_index": h5_index,
                        "dataset_id": "synthetic",
                        "subject_id": f"sub{subject:03d}",
                        "session_id": f"ses{session:02d}",
                        "run_id": "run00",
                        "trial_id": f"{trial:03d}",
                        "label": label,
                        "stimulus_frequency_hz": float(freq),
                        "stimulus_phase_rad": 0.0,
                        "sfreq_original": float(target_sfreq),
                        "sfreq_processed": float(target_sfreq),
                        "window_start_sec": 0.0,
                        "window_duration_sec": float(duration_sec),
                        "reference": "average",
                        "hardware_id": "public_unknown",
                        "cap_type": "wet_cap",
                        "electrode_type": "wet",
                        "n_channels_original": len(DEFAULT_CHANNELS),
                        "n_channels_used": int(mask.sum()),
                        "channel_names_original": DEFAULT_CHANNELS,
                        "channel_names_used": DEFAULT_CHANNELS,
                        "canonical_channel_ids": slot_ids,
                        "impedance_mean_kohm": None,
                        "impedance_max_kohm": None,
                        "reattach_flag": None,
                        "time_since_last_session_hours": None,
                        "environment_note_code": "synthetic",
                        "source_file": "generated",
                    }
                )

    x_arr = np.stack(xs, axis=0).astype(np.float32)
    mask_arr = np.stack(masks, axis=0).astype(bool)
    y_arr = np.asarray(ys, dtype=np.int64)
    write_processed_hdf5(out_dir, x_arr, mask_arr, y_arr)
    manifest = pd.DataFrame(rows, columns=REQUIRED_MANIFEST_COLUMNS)
    validate_manifest(manifest)
    write_manifest(manifest, out_dir)
    write_class_map(freqs, out_dir)
    cfg = PreprocessConfig(
        target_sfreq=target_sfreq,
        window_start_sec=0.0,
        window_duration_sec=duration_sec,
        bandpass_low_hz=None,
        bandpass_high_hz=None,
        notch_hz=None,
        normalize="per_trial_channel_zscore",
        c_max=c_max,
    )
    with (out_dir / "preprocess_config.yaml").open("w", encoding="utf-8") as f:
        yaml.safe_dump(cfg.__dict__, f, sort_keys=False)
    asset_info = {
        "dataset_id": "synthetic",
        "raw_dir": None,
        "processed_dir": str(out_dir),
        "created_by": "scripts/prepare_synthetic.py",
        "target_sfreq": float(target_sfreq),
        "source": "generated synthetic non-human signal",
        "notes": "No raw human EEG is stored in repository.",
    }
    with (out_dir / "asset_info.json").open("w", encoding="utf-8") as f:
        json.dump(asset_info, f, indent=2)
    return {"processed_dir": str(out_dir), "n_samples": len(rows), "n_classes": n_classes}
