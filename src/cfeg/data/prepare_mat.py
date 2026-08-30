from __future__ import annotations

import json
import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import pandas as pd
import yaml
from scipy.io import loadmat

from cfeg.data.io_hdf5 import write_processed_hdf5
from cfeg.data.label_mapping import remap_source_label, write_class_map
from cfeg.data.preprocess import (
    CanonicalChannelMap,
    PreprocessConfig,
    preprocess_trial_with_qc,
)
from cfeg.data.schema import (
    nullable_vector,
    ordered_manifest_columns,
    validate_manifest,
    write_manifest,
)

LABEL_KEYS = {"label", "labels", "y", "target", "targets", "class", "classes", "class_id"}
CHANNEL_KEYS = {"channel", "channels", "channel_names", "chan", "chans", "chanlocs", "chaninfo"}
FREQUENCY_KEYS = {"freq", "freqs", "frequency", "frequencies", "stimulus_frequency_hz"}
PHASE_KEYS = {"phase", "phases", "stimulus_phase_rad"}
SFREQ_KEYS = {"srate", "sfreq", "sample_rate", "sampling_rate"}


def prepare_mat_directory(
    raw_dir: Path, out_dir: Path, cfg: dict[str, Any], dataset_id: str
) -> None:
    files, duplicate_files = _dedupe_files(
        sorted([*raw_dir.rglob("*.mat"), *raw_dir.rglob("*.npz")])
    )
    if not files:
        raise FileNotFoundError(
            f"No .mat or .npz files found under {raw_dir}. If this dataset is in another raw "
            "format, inspect it first and add a schema-specific adapter."
        )
    if duplicate_files:
        print(f"skipped {len(duplicate_files)} duplicate linked raw file(s)")
    pcfg = PreprocessConfig.from_dict(cfg.get("preprocess"))
    cmap = CanonicalChannelMap.from_yaml()
    expected_channels = int(cfg.get("expected", {}).get("n_channels") or pcfg.c_max)
    channel_names = cfg.get("channel_names") or [
        cmap.id_to_name.get(i + 1, f"CH{i + 1}") for i in range(expected_channels)
    ]
    _validate_configured_channels(channel_names, expected_channels, cmap, dataset_id)
    raw_sfreq = float(cfg.get("raw_sfreq", cfg.get("sfreq", pcfg.target_sfreq)))
    n_targets = int(cfg.get("expected", {}).get("n_targets") or cfg.get("n_targets") or 0)
    class_freqs = [
        float(v) for v in (cfg.get("class_frequencies") or _default_freqs(max(n_targets, 1)))
    ]
    canonical_freqs = [float(v) for v in (cfg.get("canonical_class_frequencies") or class_freqs)]
    class_phases = cfg.get("class_phases")
    drop_unknown_channels = bool(cfg.get("drop_unknown_channels", False))
    dropped_unknown_names: set[str] = set()

    xs, masks, ys, rows = [], [], [], []
    observed_source_schemas: set[tuple[int, ...]] = set()
    for file in files:
        arrays = _load_arrays(file)
        try:
            data_key, data = _select_data_array(arrays, expected_channels=expected_channels)
        except KeyError:
            print(f"skipped support MAT without an EEG array: {file}")
            continue
        source_shape = _validate_source_schema(data, cfg, dataset_id, file)
        observed_source_schemas.add(source_shape)
        trials = _to_trials_channels_time(data, expected_channels=expected_channels)
        labels = _extract_labels(
            arrays,
            len(trials),
            n_targets=n_targets,
            data=data,
            expected_channels=expected_channels,
        )
        _validate_known_subject_trials(
            labels,
            len(trials),
            n_targets=n_targets,
            n_blocks=cfg.get("expected", {}).get("n_blocks_per_subject"),
            dataset_id=dataset_id,
            file=file,
        )
        block_indices = _trial_block_indices(
            data,
            expected_channels=expected_channels,
            n_targets=n_targets,
        )
        file_channel_names = _extract_channel_names(arrays, expected_channels) or channel_names
        file_class_freqs = _extract_numeric_vector(arrays, FREQUENCY_KEYS, n_targets) or class_freqs
        file_class_phases = _extract_numeric_vector(arrays, PHASE_KEYS, n_targets) or class_phases
        file_raw_sfreq = _extract_scalar(arrays, SFREQ_KEYS) or raw_sfreq
        subject_id = _subject_from_path(file)
        for trial_i, trial in enumerate(trials):
            original_trial = trial
            original_ch_names = file_channel_names[: original_trial.shape[0]]
            ch_names = original_ch_names
            if drop_unknown_channels:
                trial, ch_names, dropped = _drop_unknown_channels(
                    original_trial, original_ch_names, cmap
                )
                new_dropped = [name for name in dropped if name not in dropped_unknown_names]
                if new_dropped:
                    print(f"dropped unknown channel(s) from {file.name}: {', '.join(new_dropped)}")
                    dropped_unknown_names.update(new_dropped)
            placed, mask, slot_ids, sfreq_processed, query_qc = preprocess_trial_with_qc(
                trial, ch_names, file_raw_sfreq, pcfg, cmap
            )
            source_label = int(labels[trial_i])
            label, freq = remap_source_label(source_label, file_class_freqs, canonical_freqs)
            h5_index = len(xs)
            xs.append(placed)
            masks.append(mask)
            ys.append(label)
            phase = (
                float(file_class_phases[source_label % len(file_class_phases)])
                if file_class_phases
                else None
            )
            rows.append(
                {
                    "sample_id": f"{dataset_id}_{file.stem}_{trial_i:05d}",
                    "h5_index": h5_index,
                    "dataset_id": dataset_id,
                    "subject_id": subject_id,
                    "session_id": "session01" if block_indices is not None else "unknown",
                    "run_id": (
                        f"block{int(block_indices[trial_i]) + 1:02d}"
                        if block_indices is not None
                        else "unknown"
                    ),
                    "trial_id": f"{trial_i:05d}",
                    "label": label,
                    "stimulus_frequency_hz": freq,
                    "stimulus_phase_rad": phase,
                    "sfreq_original": file_raw_sfreq,
                    "sfreq_processed": sfreq_processed,
                    "window_start_sec": pcfg.window_start_sec,
                    "window_duration_sec": pcfg.window_duration_sec,
                    "reference": cfg.get("reference", "unknown"),
                    "hardware_id": cfg.get("hardware_id", "public_unknown"),
                    "cap_type": cfg.get("cap_type", "unknown"),
                    "electrode_type": cfg.get("electrode_type", "unknown"),
                    "n_channels_original": int(original_trial.shape[0]),
                    "n_channels_used": int(mask.sum()),
                    "channel_names_original": original_ch_names,
                    "channel_names_used": ch_names,
                    "canonical_channel_ids": slot_ids,
                    "impedance_mean_kohm": None,
                    "impedance_max_kohm": None,
                    "reattach_flag": None,
                    "time_since_last_session_hours": None,
                    "environment_note_code": cfg.get("environment_note_code", "unknown"),
                    "source_file": str(file),
                    "query_signal_std": query_qc.signal_std,
                    "query_signal_std_by_channel": nullable_vector(query_qc.signal_std_by_channel),
                }
            )
        print(f"prepared {file} key={data_key} trials={len(trials)}")

    if not xs:
        raise RuntimeError(f"No trials could be prepared from {raw_dir}")
    out_dir.mkdir(parents=True, exist_ok=True)
    write_processed_hdf5(out_dir, np.stack(xs), np.stack(masks), np.asarray(ys, dtype=np.int64))
    manifest = pd.DataFrame(rows, columns=ordered_manifest_columns(rows))
    validate_manifest(manifest)
    write_manifest(manifest, out_dir)
    write_class_map(canonical_freqs, out_dir)
    with (out_dir / "preprocess_config.yaml").open("w", encoding="utf-8") as f:
        yaml.safe_dump(pcfg.__dict__, f, sort_keys=False)
    with (out_dir / "asset_info.json").open("w", encoding="utf-8") as f:
        json.dump(
            {
                "dataset_id": dataset_id,
                "dataset_revision": cfg.get("dataset_revision"),
                "raw_dir": str(raw_dir),
                "processed_dir": str(out_dir),
                "created_by": "scripts/prepare_dataset.py",
                "source": "manual .mat/.npz adapter",
                "class_alignment": "stimulus_frequency_hz",
                "canonical_class_frequencies": canonical_freqs,
                "observed_source_schemas": [
                    list(shape) for shape in sorted(observed_source_schemas)
                ],
                "distributed_variant": cfg.get("distributed_variant"),
                "acquisition_provenance": cfg.get("acquisition_provenance", {}),
                "query_qc_extractor_version": ("filtered_cropped_pre_zscore_channel_std_median_v1"),
                "external_continuous_schema": "impedance_mean_max_v1",
                "notes": "Raw files are not copied; source labels are remapped by frequency.",
            },
            f,
            indent=2,
        )


def _validate_configured_channels(
    channel_names: list[str],
    expected_channels: int,
    canonical_map: CanonicalChannelMap,
    dataset_id: str,
) -> None:
    if len(channel_names) != expected_channels:
        raise ValueError(
            f"{dataset_id} config has {len(channel_names)} channel names, expected "
            f"{expected_channels}."
        )
    channel_ids = canonical_map.get_ids(channel_names)
    unknown = [name for name, channel_id in zip(channel_names, channel_ids) if channel_id == 0]
    if unknown:
        raise ValueError(f"{dataset_id} config contains unknown channels: {unknown}")
    if len(set(channel_ids)) != len(channel_ids):
        raise ValueError(f"{dataset_id} config maps multiple raw channels to one canonical slot.")


def _validate_source_schema(
    data: np.ndarray,
    cfg: dict[str, Any],
    dataset_id: str,
    file: Path,
) -> tuple[int, ...]:
    shape = tuple(int(value) for value in np.squeeze(np.asarray(data)).shape)
    configured = cfg.get("source_schema_variants")
    if configured is None and cfg.get("source_schema") is not None:
        configured = [cfg["source_schema"]]
    if configured is not None:
        expected = {tuple(int(value) for value in variant) for variant in configured}
        if shape not in expected:
            raise ValueError(
                f"Unexpected {dataset_id} EEG schema in {file}: got {shape}, expected one of "
                f"{sorted(expected)}. Refusing heuristic axis/label inference."
            )
    if not np.isfinite(data).all():
        raise ValueError(f"{dataset_id} EEG array contains non-finite values: {file}")
    return shape


def _validate_known_subject_trials(
    labels: np.ndarray,
    n_trials: int,
    *,
    n_targets: int,
    n_blocks: int | None,
    dataset_id: str,
    file: Path,
) -> None:
    if not n_blocks:
        return
    expected_trials = int(n_targets) * int(n_blocks)
    if n_trials != expected_trials:
        raise ValueError(
            f"{dataset_id} {file.name} has {n_trials} trials, expected {n_targets} targets × "
            f"{n_blocks} blocks = {expected_trials}."
        )
    counts = np.bincount(np.asarray(labels, dtype=int), minlength=n_targets)
    if len(counts) != n_targets or not np.all(counts == int(n_blocks)):
        raise ValueError(
            f"{dataset_id} {file.name} label counts must be {n_blocks} per target; got "
            f"{counts.tolist()}."
        )


def _trial_block_indices(
    data: np.ndarray,
    *,
    expected_channels: int,
    n_targets: int,
) -> np.ndarray | None:
    shape = np.squeeze(np.asarray(data)).shape
    if len(shape) < 4 or n_targets <= 0:
        return None
    channel_axis = _channel_axis(shape, expected_channels)
    time_axis = _time_axis(shape, channel_axis)
    trial_axes = [axis for axis in range(len(shape)) if axis not in {channel_axis, time_axis}]
    class_axes = [axis for axis in trial_axes if shape[axis] == n_targets]
    if len(class_axes) != 1:
        return None
    block_axes = [axis for axis in trial_axes if axis != class_axes[0]]
    if len(block_axes) != 1:
        return None
    trial_shape = tuple(shape[axis] for axis in trial_axes)
    block_axis_in_trials = trial_axes.index(block_axes[0])
    return np.indices(trial_shape, dtype=np.int64)[block_axis_in_trials].reshape(-1)


def _dedupe_files(files: list[Path]) -> tuple[list[Path], list[Path]]:
    seen: set[tuple[int, int] | str] = set()
    unique: list[Path] = []
    duplicates: list[Path] = []
    for file in files:
        try:
            stat = file.stat()
            key: tuple[int, int] | str = (stat.st_dev, stat.st_ino)
        except OSError:
            key = str(file.resolve(strict=False))
        if key in seen:
            duplicates.append(file)
            continue
        seen.add(key)
        unique.append(file)
    return unique, duplicates


def _load_arrays(path: Path) -> dict[str, Any]:
    if path.suffix.lower() == ".npz":
        with np.load(path, allow_pickle=False) as data:
            return {k: np.asarray(v) for k, v in data.items()}
    try:
        mat = loadmat(path, simplify_cells=True)
        return _flatten_items({k: v for k, v in mat.items() if not k.startswith("__")})
    except NotImplementedError:
        arrays: dict[str, Any] = {}
        with h5py.File(path, "r") as h5:
            h5.visititems(
                lambda name, obj: (
                    arrays.setdefault(name, np.asarray(obj))
                    if isinstance(obj, h5py.Dataset)
                    else None
                )
            )
        return arrays


def _flatten_items(value: Any, prefix: str = "") -> dict[str, Any]:
    flat: dict[str, Any] = {}
    if isinstance(value, Mapping):
        for key, child in value.items():
            name = f"{prefix}.{key}" if prefix else str(key)
            flat.update(_flatten_items(child, name))
        return flat
    arr = np.asarray(value)
    if arr.dtype == object and arr.ndim > 0 and arr.size <= 512:
        flat[prefix or "array"] = value
        for index in np.ndindex(arr.shape):
            name = f"{prefix}[{','.join(str(i) for i in index)}]"
            flat.update(_flatten_items(arr[index], name))
        return flat
    flat[prefix or "array"] = value
    return flat


def _select_data_array(
    arrays: dict[str, Any], expected_channels: int | None = None
) -> tuple[str, np.ndarray]:
    candidates = []
    for key, value in arrays.items():
        arr = np.asarray(value)
        if key.lower() in LABEL_KEYS:
            continue
        if expected_channels and expected_channels not in arr.shape:
            continue
        if arr.ndim >= 2 and np.issubdtype(arr.dtype, np.number):
            candidates.append((arr.size, key, arr))
    if not candidates:
        raise KeyError(f"No numeric EEG data array found. Available keys: {sorted(arrays)}")
    _, key, arr = max(candidates, key=lambda item: item[0])
    return key, np.asarray(arr, dtype=np.float32)


def _to_trials_channels_time(data: np.ndarray, expected_channels: int) -> np.ndarray:
    arr = np.asarray(data, dtype=np.float32)
    arr = np.squeeze(arr)
    if arr.ndim == 2:
        ch_axis = _channel_axis(arr.shape, expected_channels)
        if ch_axis == 1:
            arr = arr.T
        return arr[None, :, :]
    if arr.ndim < 2:
        raise ValueError(f"EEG data array must have at least 2 dims, got shape {arr.shape}")
    ch_axis = _channel_axis(arr.shape, expected_channels)
    time_axis = _time_axis(arr.shape, ch_axis)
    other_axes = [i for i in range(arr.ndim) if i not in {ch_axis, time_axis}]
    transposed = np.moveaxis(arr, [*other_axes, ch_axis, time_axis], list(range(arr.ndim)))
    n_trials = int(np.prod([arr.shape[i] for i in other_axes])) if other_axes else 1
    return transposed.reshape(n_trials, arr.shape[ch_axis], arr.shape[time_axis])


def _channel_axis(shape: tuple[int, ...], expected_channels: int) -> int:
    for i, size in enumerate(shape):
        if size == expected_channels:
            return i
    plausible = [
        (abs(size - expected_channels), i) for i, size in enumerate(shape) if 2 <= size <= 128
    ]
    if plausible:
        return min(plausible)[1]
    return int(np.argmin(shape))


def _time_axis(shape: tuple[int, ...], channel_axis: int) -> int:
    candidates = [(size, i) for i, size in enumerate(shape) if i != channel_axis]
    return max(candidates)[1]


def _extract_labels(
    arrays: dict[str, Any],
    n_trials: int,
    n_targets: int,
    *,
    data: np.ndarray | None = None,
    expected_channels: int = 0,
) -> np.ndarray:
    for key, value in arrays.items():
        if _last_key(key) in LABEL_KEYS:
            labels = np.asarray(value).squeeze().astype(int).reshape(-1)
            if len(labels) >= n_trials:
                labels = labels[:n_trials]
                if labels.size and labels.min() == 1:
                    labels = labels - 1
                return labels
    if n_targets <= 0:
        return np.zeros(n_trials, dtype=np.int64)
    if data is not None:
        inferred = _labels_from_data_shape(data, expected_channels, n_targets)
        if inferred is not None and len(inferred) == n_trials:
            return inferred
    return np.arange(n_trials, dtype=np.int64) % n_targets


def _labels_from_data_shape(
    data: np.ndarray, expected_channels: int, n_targets: int
) -> np.ndarray | None:
    shape = np.squeeze(np.asarray(data)).shape
    if len(shape) < 3:
        return None
    channel_axis = _channel_axis(shape, expected_channels)
    time_axis = _time_axis(shape, channel_axis)
    trial_axes = [axis for axis in range(len(shape)) if axis not in {channel_axis, time_axis}]
    class_axes = [axis for axis in trial_axes if shape[axis] == n_targets]
    if len(class_axes) != 1:
        return None
    trial_shape = tuple(shape[axis] for axis in trial_axes)
    class_axis_in_trials = trial_axes.index(class_axes[0])
    return np.indices(trial_shape, dtype=np.int64)[class_axis_in_trials].reshape(-1)


def _extract_channel_names(arrays: dict[str, Any], expected_channels: int) -> list[str] | None:
    for key, value in arrays.items():
        if _last_key(key) not in CHANNEL_KEYS:
            continue
        names = _strings_from_value(value, expected_channels)
        if names:
            return names
    for value in arrays.values():
        names = _strings_from_value(value, expected_channels)
        if names:
            return names
    return None


def _strings_from_value(value: Any, expected_channels: int) -> list[str] | None:
    arr = np.asarray(value, dtype=object)
    if arr.ndim == 0:
        if isinstance(arr.item(), str):
            return [arr.item()] if expected_channels == 1 else None
        return None
    if arr.ndim == 1:
        strings = [str(x) for x in arr.tolist() if isinstance(x, str)]
        return (
            strings
            if len(strings) == expected_channels and _channel_name_score(strings) > 0
            else None
        )
    candidates: list[tuple[int, list[str]]] = []
    for axis in range(arr.ndim):
        if arr.shape[axis] != expected_channels:
            continue
        moved = np.moveaxis(arr, axis, 0).reshape(expected_channels, -1)
        for col in range(moved.shape[1]):
            strings = [str(x) for x in moved[:, col].tolist() if isinstance(x, str)]
            if len(strings) == expected_channels:
                score = _channel_name_score(strings)
                if score > 0:
                    candidates.append((score, strings))
    return max(candidates, key=lambda item: item[0])[1] if candidates else None


def _channel_name_score(names: list[str]) -> int:
    return sum(bool(re.search(r"[A-Za-z]", name)) and not _is_number_like(name) for name in names)


def _is_number_like(value: str) -> bool:
    try:
        float(value)
        return True
    except ValueError:
        return False


def _extract_numeric_vector(
    arrays: dict[str, Any], keys: set[str], expected_len: int
) -> list[float] | None:
    for key, value in arrays.items():
        if _last_key(key) not in keys:
            continue
        arr = np.asarray(value).squeeze()
        if arr.ndim != 1 or not np.issubdtype(arr.dtype, np.number):
            continue
        if expected_len > 0 and arr.size < expected_len:
            continue
        n = expected_len if expected_len > 0 else arr.size
        return [float(x) for x in arr[:n].tolist()]
    return None


def _extract_scalar(arrays: dict[str, Any], keys: set[str]) -> float | None:
    for key, value in arrays.items():
        if _last_key(key) not in keys:
            continue
        arr = np.asarray(value).squeeze()
        if arr.size != 1 or not np.issubdtype(arr.dtype, np.number):
            continue
        return float(arr.item())
    return None


def _drop_unknown_channels(
    trial: np.ndarray, ch_names: list[str], canonical_map: CanonicalChannelMap
) -> tuple[np.ndarray, list[str], list[str]]:
    keep_indices: list[int] = []
    dropped: list[str] = []
    for i, name in enumerate(ch_names):
        if canonical_map.get_id(name) > 0:
            keep_indices.append(i)
        else:
            dropped.append(name)
    if not keep_indices:
        raise ValueError(f"All channels are unknown: {ch_names}")
    return trial[keep_indices], [ch_names[i] for i in keep_indices], dropped


def _last_key(key: str) -> str:
    key = key.rsplit(".", maxsplit=1)[-1]
    return key.split("[", maxsplit=1)[0].lower()


def _subject_from_path(path: Path) -> str:
    match = re.search(r"(sub|subject|s)[_-]?(\d+)", path.stem, flags=re.IGNORECASE)
    if match:
        return f"sub{int(match.group(2)):03d}"
    return path.stem


def _default_freqs(n_targets: int) -> list[float]:
    base = [8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0]
    if n_targets <= len(base):
        return base[:n_targets]
    return [8.0 + 0.2 * i for i in range(n_targets)]
