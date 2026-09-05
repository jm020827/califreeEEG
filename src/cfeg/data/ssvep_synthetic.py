from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from cfeg.data.io_hdf5 import write_processed_hdf5
from cfeg.data.label_mapping import write_class_map
from cfeg.data.preprocess import (
    CanonicalChannelMap,
    PreprocessConfig,
    normalize_trial,
    place_channel_values,
    place_on_canonical_channels,
)
from cfeg.data.schema import (
    nullable_vector,
    ordered_manifest_columns,
    validate_manifest,
    write_manifest,
)

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
    n_samples = round(target_sfreq * duration_sec)
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
                query_std_native = np.std(raw, axis=-1).astype(np.float32)
                query_std_by_channel = place_channel_values(
                    query_std_native, DEFAULT_CHANNELS, cmap, c_max
                )
                query_signal_std = float(np.median(query_std_native))
                raw = normalize_trial(raw)
                placed, mask, slot_ids = place_on_canonical_channels(
                    raw, DEFAULT_CHANNELS, cmap, c_max
                )
                h5_index = len(xs)
                sample_id = (
                    f"synthetic_sub{subject:03d}_ses{session:02d}_cls{label:02d}_tr{trial:03d}"
                )
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
                        "query_signal_std": query_signal_std,
                        "query_signal_std_by_channel": nullable_vector(query_std_by_channel),
                    }
                )

    x_arr = np.stack(xs, axis=0).astype(np.float32)
    mask_arr = np.stack(masks, axis=0).astype(bool)
    y_arr = np.asarray(ys, dtype=np.int64)
    write_processed_hdf5(out_dir, x_arr, mask_arr, y_arr)
    manifest = pd.DataFrame(rows, columns=ordered_manifest_columns(rows))
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
        "query_qc_extractor_version": "filtered_cropped_pre_zscore_channel_std_median_v1",
        "external_continuous_schema": "impedance_mean_max_v1",
        "source": "generated synthetic non-human signal",
        "notes": "No raw human EEG is stored in repository.",
    }
    with (out_dir / "asset_info.json").open("w", encoding="utf-8") as f:
        json.dump(asset_info, f, indent=2)
    return {"processed_dir": str(out_dir), "n_samples": len(rows), "n_classes": n_classes}


def generate_quality_synthetic_processed(
    out_dir: str | Path,
    n_subjects: int = 8,
    n_blocks_per_interface: int = 2,
    n_repetitions_per_class_per_block: int = 2,
    n_classes: int = 4,
    target_sfreq: float = 200.0,
    duration_sec: float = 2.0,
    c_max: int = 64,
    seed: int = 42,
) -> dict[str, int | str]:
    """Atomically publish a fresh block-coherent acquisition-quality assay."""

    output = Path(out_dir)
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"Synthetic quality output already exists: {output}.")
    output.parent.mkdir(parents=True, exist_ok=True)
    reservation = output.parent / f".{output.name}.generation-reservation"
    try:
        reservation.mkdir(mode=0o700)
    except FileExistsError as exc:
        raise FileExistsError(
            f"Synthetic quality generation is already reserved: {reservation}."
        ) from exc
    staging: Path | None = None
    try:
        if output.exists() or output.is_symlink():
            raise FileExistsError(
                f"Synthetic quality output appeared before generation: {output}."
            )
        staging = Path(
            tempfile.mkdtemp(prefix=f".{output.name}.staging-", dir=output.parent)
        )
        result = _generate_quality_synthetic_processed_into(
            staging,
            n_subjects=n_subjects,
            n_blocks_per_interface=n_blocks_per_interface,
            n_repetitions_per_class_per_block=n_repetitions_per_class_per_block,
            n_classes=n_classes,
            target_sfreq=target_sfreq,
            duration_sec=duration_sec,
            c_max=c_max,
            seed=seed,
            recorded_out_dir=output,
        )
        entries = list(staging.iterdir())
        if not entries or any(path.is_symlink() or not path.is_file() for path in entries):
            raise ValueError("Synthetic quality staging tree contains an unsafe entry.")
        for path in entries:
            with path.open("rb") as handle:
                os.fsync(handle.fileno())
        _fsync_directory(staging)
        if output.exists() or output.is_symlink():
            raise FileExistsError(f"Synthetic quality output appeared during generation: {output}.")
        os.replace(staging, output)
        _fsync_directory(output.parent)
    finally:
        if staging is not None and staging.exists():
            shutil.rmtree(staging)
        if reservation.is_dir() and not reservation.is_symlink():
            reservation.rmdir()
            _fsync_directory(reservation.parent)
    result.update(
        {
            "processed_dir": str(output),
            "quality_truth": str(output / "quality_truth.jsonl"),
        }
    )
    return result


def _generate_quality_synthetic_processed_into(
    out_dir: str | Path,
    n_subjects: int = 8,
    n_blocks_per_interface: int = 2,
    n_repetitions_per_class_per_block: int = 2,
    n_classes: int = 4,
    target_sfreq: float = 200.0,
    duration_sec: float = 2.0,
    c_max: int = 64,
    seed: int = 42,
    *,
    recorded_out_dir: str | Path | None = None,
) -> dict[str, int | str]:
    """Generate a block-coherent acquisition-quality engineering assay.

    Wet/dry interface and per-channel impedance are balanced across every class
    and are sampled once per acquisition block. They influence injected noise,
    not the target label. Simulation-only realized-noise targets are written to
    a separate file that the normal dataset/collate path never reads.
    """

    if n_subjects < 3:
        raise ValueError("quality synthetic data requires at least three subjects.")
    if n_blocks_per_interface < 1 or n_repetitions_per_class_per_block < 1:
        raise ValueError("quality synthetic block/repetition counts must be positive.")
    if not 1 <= n_classes <= len(DEFAULT_FREQS):
        raise ValueError(f"n_classes must be between 1 and {len(DEFAULT_FREQS)}.")

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    freqs = DEFAULT_FREQS[:n_classes]
    n_samples = round(target_sfreq * duration_sec)
    time = np.arange(n_samples, dtype=np.float32) / float(target_sfreq)
    cmap = CanonicalChannelMap.from_yaml()
    canonical_ids = cmap.get_ids(DEFAULT_CHANNELS)
    xs: list[np.ndarray] = []
    masks: list[np.ndarray] = []
    ys: list[int] = []
    rows: list[dict] = []
    quality_rows: list[dict] = []

    for subject in range(n_subjects):
        subject_rng = _hierarchical_rng(seed, 0, subject)
        subject_phase = subject_rng.uniform(0, 2 * np.pi)
        subject_noise = subject_rng.uniform(0.03, 0.10)
        subject_amp = subject_rng.uniform(0.85, 1.25)
        subject_contact_bias = subject_rng.normal(0.0, 0.35)
        wet_first = bool(_hierarchical_rng(seed, 4, subject).integers(0, 2))
        interface_order = ["wet", "dry"] if wet_first else ["dry", "wet"]
        order_label = "wet_first" if wet_first else "dry_first"
        for period_index, electrode_type in enumerate(interface_order):
            interface_code = 0 if electrode_type == "wet" else 1
            condition_offset = -0.45 if electrode_type == "wet" else 0.45
            for block in range(n_blocks_per_interface):
                measurement_id = (
                    f"synthetic_q_sub{subject:03d}_{electrode_type}_block{block:02d}"
                )
                block_rng = _hierarchical_rng(seed, 1, subject, interface_code, block)
                latent_logit = (
                    subject_contact_bias
                    + condition_offset
                    + block_rng.normal(0.0, 0.30)
                    + block_rng.normal(0.0, 0.55, size=len(DEFAULT_CHANNELS))
                )
                latent_badness = 1.0 / (1.0 + np.exp(-latent_logit))
                impedance_rng = _hierarchical_rng(seed, 2, subject, interface_code, block)
                impedance = np.exp(
                    np.log(4.0)
                    + 2.7 * latent_badness
                    + impedance_rng.normal(0.0, 0.32, size=len(DEFAULT_CHANNELS))
                ).clip(1.0, 250.0).astype(np.float32)
                clean_gain = (1.0 - 0.50 * latent_badness).astype(np.float32)
                noise_std = (
                    subject_noise + 0.08 + 0.58 * latent_badness
                ).clip(0.08, 0.78).astype(np.float32)
                mains_amplitude = (0.01 + 0.22 * latent_badness).astype(np.float32)
                artifact_loading = (0.03 + 0.42 * latent_badness).astype(np.float32)
                dropout_probability = np.clip(
                    0.01 + 0.22 * np.square(latent_badness), 0.01, 0.25
                )
                artifact_frequency = float(block_rng.uniform(18.0, 42.0))
                impedance_by_channel = place_channel_values(
                    impedance, DEFAULT_CHANNELS, cmap, c_max
                )
                for repetition in range(n_repetitions_per_class_per_block):
                    order_rng = _hierarchical_rng(
                        seed, 5, subject, interface_code, block, repetition
                    )
                    label_order = order_rng.permutation(n_classes)
                    for trial_position, label in enumerate(label_order):
                        freq = freqs[int(label)]
                        trial_rng = _hierarchical_rng(
                            seed,
                            3,
                            subject,
                            interface_code,
                            block,
                            repetition,
                            int(label),
                        )
                        trial_phase_jitter = trial_rng.normal(0.0, 0.08)
                        shared_white = trial_rng.normal(0.0, 1.0, size=n_samples)
                        shared_colored = np.convolve(
                            shared_white, np.ones(9, dtype=np.float32) / 9.0, mode="same"
                        )
                        shared_colored /= np.std(shared_colored) + 1e-6
                        shared_artifact = (
                            0.65
                            * np.sin(
                                2 * np.pi * artifact_frequency * time
                                + trial_rng.uniform(0, 2 * np.pi)
                            )
                            + 0.35 * shared_colored
                        )
                        clean_rows = []
                        raw_rows = []
                        corruption_rows = []
                        dropout_fractions = []
                        for channel_index, _channel_name in enumerate(DEFAULT_CHANNELS):
                            occipital_gain = 1.0 + 0.18 * (channel_index >= 3)
                            phase = (
                                subject_phase
                                + trial_phase_jitter
                                + channel_index * 0.035
                                + trial_rng.normal(0.0, 0.04)
                            )
                            clean = clean_gain[channel_index] * subject_amp * occipital_gain * (
                                np.sin(2 * np.pi * freq * time + phase)
                                + 0.42 * np.sin(2 * np.pi * 2 * freq * time + phase / 2)
                                + 0.18 * np.sin(2 * np.pi * 3 * freq * time + phase / 3)
                            )
                            independent_noise = trial_rng.normal(
                                0.0,
                                float(noise_std[channel_index]),
                                size=n_samples,
                            )
                            mains_phase = trial_rng.uniform(0, 2 * np.pi)
                            mains_noise = mains_amplitude[channel_index] * np.sin(
                                2 * np.pi * 50.0 * time + mains_phase
                            )
                            corruption = (
                                independent_noise
                                + mains_noise
                                + artifact_loading[channel_index] * shared_artifact
                            )
                            dropout_mask = np.zeros(n_samples, dtype=bool)
                            if trial_rng.random() < dropout_probability[channel_index]:
                                length = int(
                                    trial_rng.integers(
                                        max(2, n_samples // 40), max(3, n_samples // 8)
                                    )
                                )
                                start = int(trial_rng.integers(0, n_samples - length + 1))
                                dropout_mask[start : start + length] = True
                            raw = clean + corruption
                            if dropout_mask.any():
                                raw = raw.copy()
                                raw[dropout_mask] = raw[start - 1] if start > 0 else 0.0
                            clean_rows.append(clean.astype(np.float32))
                            raw_rows.append(raw.astype(np.float32))
                            corruption_rows.append(corruption.astype(np.float32))
                            dropout_fractions.append(float(dropout_mask.mean()))

                        clean_native = np.stack(clean_rows, axis=0)
                        raw_native = np.stack(raw_rows, axis=0)
                        corruption_native = np.stack(corruption_rows, axis=0)
                        query_std_native = np.std(raw_native, axis=-1).astype(np.float32)
                        query_std_by_channel = place_channel_values(
                            query_std_native, DEFAULT_CHANNELS, cmap, c_max
                        )
                        normalized = normalize_trial(raw_native)
                        placed, mask, slot_ids = place_on_canonical_channels(
                            normalized, DEFAULT_CHANNELS, cmap, c_max
                        )
                        h5_index = len(xs)
                        sample_id = (
                            f"{measurement_id}_rep{repetition:02d}_cls{int(label):02d}"
                        )
                        xs.append(placed)
                        masks.append(mask)
                        ys.append(int(label))
                        rows.append(
                            {
                                "sample_id": sample_id,
                                "h5_index": h5_index,
                                "dataset_id": "synthetic_quality",
                                "subject_id": f"sub{subject:03d}",
                                "session_id": electrode_type,
                                "run_id": f"block{block:02d}",
                                "trial_id": f"rep{repetition:02d}_cls{int(label):02d}",
                                "label": int(label),
                                "stimulus_frequency_hz": float(freq),
                                "stimulus_phase_rad": 0.0,
                                "sfreq_original": float(target_sfreq),
                                "sfreq_processed": float(target_sfreq),
                                "window_start_sec": 0.0,
                                "window_duration_sec": float(duration_sec),
                                "reference": "average",
                                "hardware_id": "public_unknown",
                                "cap_type": "wearable",
                                "electrode_type": electrode_type,
                                "n_channels_original": len(DEFAULT_CHANNELS),
                                "n_channels_used": int(mask.sum()),
                                "channel_names_original": DEFAULT_CHANNELS,
                                "channel_names_used": DEFAULT_CHANNELS,
                                "canonical_channel_ids": slot_ids,
                                "impedance_mean_kohm": float(np.mean(impedance)),
                                "impedance_max_kohm": float(np.max(impedance)),
                                "reattach_flag": bool(block > 0),
                                "time_since_last_session_hours": float(period_index * 2),
                                "environment_note_code": "synthetic_quality_v1",
                                "source_file": "generated",
                                "query_signal_std": float(np.median(query_std_native)),
                                "query_signal_std_by_channel": nullable_vector(
                                    query_std_by_channel
                                ),
                                "impedance_kohm_by_channel": nullable_vector(
                                    impedance_by_channel
                                ),
                                "headband_order": order_label,
                                "condition_period": f"period_{period_index + 1}",
                                "acquisition_block_id": measurement_id,
                                "metadata_measurement_id": measurement_id,
                                "metadata_measurement_time_sec": float(
                                    (period_index * n_blocks_per_interface + block) * 3600
                                ),
                                "query_time_sec": float(
                                    (period_index * n_blocks_per_interface + block) * 3600
                                    + 60
                                    + repetition * n_classes
                                    + trial_position
                                ),
                                "impedance_unit": "kOhm",
                            }
                        )
                        realized_error = np.mean(np.square(raw_native - clean_native), axis=-1)
                        clean_energy = np.mean(np.square(clean_native), axis=-1)
                        corruption_energy = np.mean(np.square(corruption_native), axis=-1)
                        oracle_signal_fraction = clean_energy / (
                            clean_energy + corruption_energy + 1e-8
                        )
                        realized_signal_fraction = clean_energy / (
                            clean_energy + realized_error + 1e-8
                        )
                        for channel_index, canonical_id in enumerate(canonical_ids):
                            quality_rows.append(
                                {
                                    "sample_id": sample_id,
                                    "canonical_channel_id": int(canonical_id),
                                    "acquisition_block_id": measurement_id,
                                    "latent_badness": float(latent_badness[channel_index]),
                                    "clean_gain": float(clean_gain[channel_index]),
                                    "injected_noise_std": float(noise_std[channel_index]),
                                    "injected_mains_amplitude": float(
                                        mains_amplitude[channel_index]
                                    ),
                                    "dropout_fraction": dropout_fractions[channel_index],
                                    "artifact_loading": float(
                                        artifact_loading[channel_index]
                                    ),
                                    "oracle_signal_fraction": float(
                                        oracle_signal_fraction[channel_index]
                                    ),
                                    "clean_signal_power": float(
                                        clean_energy[channel_index]
                                    ),
                                    "realized_signal_fraction": float(
                                        realized_signal_fraction[channel_index]
                                    ),
                                    "realized_clean_signal_mse": float(
                                        realized_error[channel_index]
                                    ),
                                }
                            )

    x_array = np.stack(xs, axis=0).astype(np.float32)
    mask_array = np.stack(masks, axis=0).astype(bool)
    y_array = np.asarray(ys, dtype=np.int64)
    write_processed_hdf5(out_dir, x_array, mask_array, y_array)
    manifest = pd.DataFrame(rows, columns=ordered_manifest_columns(rows))
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
    with (out_dir / "preprocess_config.yaml").open("w", encoding="utf-8") as handle:
        yaml.safe_dump(cfg.__dict__, handle, sort_keys=False)

    quality_path = out_dir / "quality_truth.jsonl"
    quality_tmp = out_dir / "quality_truth.jsonl.tmp"
    pd.DataFrame(quality_rows).to_json(quality_tmp, orient="records", lines=True)
    os.replace(quality_tmp, quality_path)
    quality_spec = {
        "schema": "cfeg.synthetic-quality-truth.v1",
        "scope": "simulation_engineering_only",
        "quality_mechanism_label_effect": "none_balanced_by_complete_block",
        "trial_rng_key_includes_label": True,
        "model_input": False,
        "row_unit": "sample_x_canonical_channel",
        "targets": [
            "latent_badness",
            "clean_gain",
            "injected_noise_std",
            "injected_mains_amplitude",
            "dropout_fraction",
            "artifact_loading",
            "oracle_signal_fraction",
            "clean_signal_power",
            "realized_signal_fraction",
            "realized_clean_signal_mse",
        ],
        "quality_truth_sha256": _file_sha256(quality_path),
    }
    with (out_dir / "quality_target_spec.json").open("w", encoding="utf-8") as handle:
        json.dump(quality_spec, handle, indent=2)
    asset_info = {
        "dataset_id": "synthetic_quality",
        "dataset_revision": "synthetic_quality_v1",
        "raw_dir": None,
        "processed_dir": str(recorded_out_dir or out_dir),
        "created_by": "scripts/prepare_synthetic.py --quality-aware",
        "target_sfreq": float(target_sfreq),
        "duration_sec": float(duration_sec),
        "c_max": int(c_max),
        "query_qc_extractor_version": "filtered_cropped_pre_zscore_channel_std_median_v1",
        "external_continuous_schema": "impedance_mean_max_v1",
        "source": "generated synthetic non-human signal",
        "quality_target_spec": "quality_target_spec.json",
        "generator_schema": "cfeg.synthetic-quality-generator.v2",
        "seed": int(seed),
        "rng_schema": "numpy_seedsequence_hierarchical_v1",
        "processed_subject_count": int(n_subjects),
        "n_blocks_per_interface": int(n_blocks_per_interface),
        "n_repetitions_per_class_per_block": int(
            n_repetitions_per_class_per_block
        ),
        "n_classes": int(n_classes),
        "n_samples": len(rows),
        "n_quality_rows": len(quality_rows),
        "n_active_channels": len(DEFAULT_CHANNELS),
        "active_channel_names": list(DEFAULT_CHANNELS),
        "active_canonical_channel_ids": [int(value) for value in canonical_ids],
        "conditions": ["wet", "dry"],
        "balanced_complete_blocks": True,
        "metadata_label_independence": (
            "metadata RNG is keyed only by seed,subject,interface,block"
        ),
        "analysis_only_artifacts": {
            "quality_truth.jsonl": {
                "schema": "cfeg.synthetic-quality-truth.v1",
                "sha256": _file_sha256(quality_path),
                "model_input": False,
            },
            "quality_target_spec.json": {
                "schema": "cfeg.synthetic-quality-truth.v1",
                "sha256": _file_sha256(out_dir / "quality_target_spec.json"),
                "model_input": False,
            },
        },
        "notes": (
            "Balanced block-level metadata/corruption assay; simulation targets are never "
            "loaded by EEGProcessedDataset."
        ),
    }
    with (out_dir / "asset_info.json").open("w", encoding="utf-8") as handle:
        json.dump(asset_info, handle, indent=2)
    return {
        "processed_dir": str(out_dir),
        "n_samples": len(rows),
        "n_classes": n_classes,
        "n_quality_rows": len(quality_rows),
        "quality_truth": str(quality_path),
    }


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _hierarchical_rng(seed: int, *coordinates: int) -> np.random.Generator:
    return np.random.default_rng(np.random.SeedSequence([int(seed), *coordinates]))
