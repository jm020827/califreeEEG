from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from cfeg.data.io_hdf5 import write_processed_hdf5
from cfeg.data.label_mapping import write_class_map
from cfeg.data.preprocess import CanonicalChannelMap, PreprocessConfig, preprocess_trial
from cfeg.data.schema import REQUIRED_MANIFEST_COLUMNS, validate_manifest, write_manifest


def prepare(raw_session_dir: Path, out_dir: Path, cfg: dict) -> None:
    eeg_csv = raw_session_dir / "eeg.csv"
    events_csv = raw_session_dir / "events.csv"
    meta_json = raw_session_dir / "session_meta.json"
    missing = [p.name for p in [eeg_csv, events_csv, meta_json] if not p.exists()]
    if missing:
        raise FileNotFoundError(
            f"OpenBCI session is missing {missing}. Expected eeg.csv, events.csv, session_meta.json."
        )
    with meta_json.open("r", encoding="utf-8") as f:
        meta = json.load(f)
    eeg = pd.read_csv(eeg_csv)
    events = pd.read_csv(events_csv)
    ch_names = list(meta.get("channel_names") or cfg.get("channel_names") or [])
    if not ch_names:
        ch_names = [c for c in eeg.columns if c.lower() not in {"time", "timestamp", "sample"}]
    x_all = eeg[ch_names].to_numpy(dtype=np.float32).T
    sfreq = float(meta.get("sfreq", cfg.get("raw_sfreq", 250.0)))
    pcfg = PreprocessConfig.from_dict(cfg.get("preprocess"))
    cmap = CanonicalChannelMap.from_yaml()
    xs, masks, ys, rows = [], [], [], []
    freqs = sorted(events["stimulus_frequency_hz"].astype(float).unique().tolist())
    freq_to_label = {f: i for i, f in enumerate(freqs)}
    for _, ev in events.iterrows():
        start = int(round(float(ev["onset_sec"]) * sfreq))
        duration = float(ev.get("duration_sec", pcfg.window_duration_sec))
        stop = start + int(round((pcfg.window_start_sec + duration + 0.05) * sfreq))
        raw_trial = x_all[:, start:min(stop, x_all.shape[-1])]
        placed, mask, _ids, new_sfreq = preprocess_trial(raw_trial, ch_names, sfreq, pcfg, cmap)
        slot_ids = ((np.arange(pcfg.c_max) + 1) * mask.astype(np.int64)).tolist()
        label = int(ev.get("class_id", freq_to_label[float(ev["stimulus_frequency_hz"])]))
        h5_index = len(xs)
        xs.append(placed)
        masks.append(mask)
        ys.append(label)
        impedance = meta.get("impedance_kohm_by_channel") or {}
        impedance_values = [float(v) for v in impedance.values()] if impedance else []
        rows.append(
            {
                "sample_id": f"{meta.get('session_id', raw_session_dir.name)}_{str(ev['trial_id']).zfill(4)}",
                "h5_index": h5_index,
                "dataset_id": "openbci",
                "subject_id": meta.get("subject_id", "unknown"),
                "session_id": meta.get("session_id", raw_session_dir.name),
                "run_id": "run00",
                "trial_id": str(ev["trial_id"]),
                "label": label,
                "stimulus_frequency_hz": float(ev["stimulus_frequency_hz"]),
                "stimulus_phase_rad": float(ev.get("stimulus_phase_rad", 0.0)),
                "sfreq_original": sfreq,
                "sfreq_processed": new_sfreq,
                "window_start_sec": pcfg.window_start_sec,
                "window_duration_sec": pcfg.window_duration_sec,
                "reference": meta.get("reference", "openbci_default"),
                "hardware_id": meta.get("hardware_id", "openbci_cyton"),
                "cap_type": meta.get("cap_type", "unknown"),
                "electrode_type": meta.get("electrode_type", "unknown"),
                "n_channels_original": len(ch_names),
                "n_channels_used": int(mask.sum()),
                "channel_names_original": ch_names,
                "channel_names_used": ch_names,
                "canonical_channel_ids": slot_ids,
                "impedance_mean_kohm": float(np.mean(impedance_values)) if impedance_values else None,
                "impedance_max_kohm": float(np.max(impedance_values)) if impedance_values else None,
                "reattach_flag": meta.get("reattach_flag"),
                "time_since_last_session_hours": meta.get("time_since_last_session_hours"),
                "environment_note_code": meta.get("environment_note_code", "unknown"),
                "source_file": str(eeg_csv),
            }
        )
    out_dir.mkdir(parents=True, exist_ok=True)
    write_processed_hdf5(out_dir, np.stack(xs), np.stack(masks), np.asarray(ys))
    manifest = pd.DataFrame(rows, columns=REQUIRED_MANIFEST_COLUMNS)
    validate_manifest(manifest)
    write_manifest(manifest, out_dir)
    write_class_map(freqs, out_dir)
    with (out_dir / "preprocess_config.yaml").open("w", encoding="utf-8") as f:
        yaml.safe_dump(pcfg.__dict__, f, sort_keys=False)
    with (out_dir / "asset_info.json").open("w", encoding="utf-8") as f:
        json.dump({"dataset_id": "openbci", "raw_dir": str(raw_session_dir), "processed_dir": str(out_dir)}, f)
