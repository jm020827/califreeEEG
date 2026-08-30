from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from cfeg.assets.errors import MissingAssetError
from cfeg.constants import EXTERNAL_CONTINUOUS_SCHEMA_V1, QUERY_QC_EXTRACTOR_V1
from cfeg.data.io_hdf5 import write_processed_hdf5
from cfeg.data.label_mapping import write_class_map
from cfeg.data.prepare_mat import _load_arrays
from cfeg.data.preprocess import (
    CanonicalChannelMap,
    PreprocessConfig,
    place_channel_values,
    preprocess_trial_with_qc,
)
from cfeg.data.schema import (
    nullable_vector,
    ordered_manifest_columns,
    validate_manifest,
    write_manifest,
)

OFFICIAL_WEARABLE_CHANNELS = ["POz", "PO3", "PO4", "PO5", "PO6", "Oz", "O1", "O2"]
QUERY_QC_EXTRACTOR_VERSION = QUERY_QC_EXTRACTOR_V1
EXTERNAL_CONTINUOUS_SCHEMA = EXTERNAL_CONTINUOUS_SCHEMA_V1


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
    impedance_electrode_types = list(cfg.get("impedance_electrode_types", ["dry", "wet"]))
    _validate_wearable_config(
        cfg,
        cmap,
        channel_names,
        frequencies,
        phases,
        electrode_types,
        impedance_electrode_types,
        pcfg.c_max,
    )
    if len(impedance_electrode_types) != 2 or len(set(impedance_electrode_types)) != 2:
        raise ValueError(
            "impedance_electrode_types must contain exactly two unique condition labels."
        )
    missing_impedance_types = sorted(set(electrode_types) - set(impedance_electrode_types))
    if missing_impedance_types:
        raise ValueError(
            "impedance_electrode_types must map every EEG electrode type; missing "
            f"{missing_impedance_types}."
        )
    impedance = _load_impedance(raw_dir)
    _validate_required_impedance(impedance, cfg)
    impedance_signature = _validate_impedance_signature(impedance, impedance_electrode_types, cfg)
    require_impedance = bool(cfg.get("expected", {}).get("has_impedance", False))
    headband_orders = _load_headband_orders(raw_dir)
    _validate_headband_orders(headband_orders, cfg)
    require_headband_order = bool(cfg.get("expected", {}).get("has_headband_order", False))

    xs: list[np.ndarray] = []
    masks: list[np.ndarray] = []
    ys: list[int] = []
    rows: list[dict[str, Any]] = []
    for file in subject_files:
        subject_index = int(re.search(r"(\d+)", file.stem).group(1)) - 1
        headband_order = headband_orders.get(subject_index + 1)
        if require_headband_order and headband_order is None:
            raise ValueError(
                f"Required wearable headband order is missing for subject={subject_index + 1}."
            )
        data = _wearable_data(_load_arrays(file))
        for electrode_index, electrode_type in enumerate(electrode_types):
            impedance_electrode_index = _impedance_electrode_index(
                electrode_type, impedance_electrode_types
            )
            for block_index in range(data.shape[3]):
                imp = _impedance_for(
                    impedance,
                    subject_index,
                    impedance_electrode_index,
                    block_index,
                )
                if require_impedance and imp is None:
                    raise ValueError(
                        "Required wearable impedance is missing or non-finite for "
                        f"subject={subject_index + 1}, electrode={electrode_type}, "
                        f"block={block_index + 1}."
                    )
                finite_impedance = (
                    np.asarray([], dtype=np.float32) if imp is None else imp[np.isfinite(imp)]
                )
                impedance_by_channel = (
                    np.full((pcfg.c_max,), np.nan, dtype=np.float32)
                    if imp is None
                    else place_channel_values(imp, channel_names, cmap, pcfg.c_max)
                )
                for target_index, frequency in enumerate(frequencies):
                    raw = data[:, :, electrode_index, block_index, target_index]
                    placed, mask, slot_ids, sfreq_processed, query_qc = preprocess_trial_with_qc(
                        raw, channel_names, float(cfg.get("raw_sfreq", 250.0)), pcfg, cmap
                    )
                    h5_index = len(xs)
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
                            "impedance_mean_kohm": (
                                float(np.mean(finite_impedance)) if finite_impedance.size else None
                            ),
                            "impedance_max_kohm": (
                                float(np.max(finite_impedance)) if finite_impedance.size else None
                            ),
                            "reattach_flag": None,
                            "time_since_last_session_hours": None,
                            "environment_note_code": cfg.get("environment_note_code", "unknown"),
                            "source_file": str(file),
                            "query_signal_std": query_qc.signal_std,
                            "query_signal_std_by_channel": nullable_vector(
                                query_qc.signal_std_by_channel
                            ),
                            "impedance_kohm_by_channel": nullable_vector(impedance_by_channel),
                            "headband_order": headband_order,
                            "condition_period": _condition_period(headband_order, electrode_type),
                        }
                    )
        print(f"prepared {file} trials={data.shape[2] * data.shape[3] * data.shape[4]}")

    out_dir.mkdir(parents=True, exist_ok=True)
    write_processed_hdf5(out_dir, np.stack(xs), np.stack(masks), np.asarray(ys, dtype=np.int64))
    manifest = pd.DataFrame(rows, columns=ordered_manifest_columns(rows))
    validate_manifest(manifest)
    write_manifest(manifest, out_dir)
    write_class_map(frequencies, out_dir)
    with (out_dir / "preprocess_config.yaml").open("w", encoding="utf-8") as handle:
        yaml.safe_dump(pcfg.__dict__, handle, sort_keys=False)
    with (out_dir / "asset_info.json").open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "dataset_id": "wearable",
                "dataset_revision": cfg.get("dataset_revision", "unknown"),
                "raw_dir": str(raw_dir),
                "processed_dir": str(out_dir),
                "created_by": "scripts/prepare_dataset.py",
                "source": "Figshare 13560281",
                "source_version": 4,
                "schema": "[channel,time,electrode,block,target]",
                "electrode_types": electrode_types,
                "impedance_electrode_types": impedance_electrode_types,
                "impedance_numeric_signature": impedance_signature,
                "source_cohort_headband_order_counts": _headband_order_counts(headband_orders),
                "processed_subject_count": len(subject_files),
                "processed_subject_ids": [f"sub{int(path.stem[1:]):03d}" for path in subject_files],
                "has_impedance": impedance is not None,
                "per_channel_impedance_preserved": True,
                "query_qc_extractor_version": QUERY_QC_EXTRACTOR_VERSION,
                "external_continuous_schema": EXTERNAL_CONTINUOUS_SCHEMA,
                "has_headband_order": bool(headband_orders),
                "acquisition_provenance": cfg.get("acquisition_provenance", {}),
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
    if len(paths) != 1:
        raise ValueError(
            f"Expected exactly one Impedance.mat under {raw_dir}, found {len(paths)}: {paths}."
        )
    arrays = _load_arrays(paths[0])
    candidates = [np.asarray(value) for value in arrays.values() if np.asarray(value).ndim == 4]
    if not candidates:
        return None
    data = max(candidates, key=lambda value: value.size)
    axes = [_axis_for_size(data.shape, size) for size in (8, 10, 2, 102)]
    if len(set(axes)) != 4:
        return None
    return np.moveaxis(data, axes, range(4)).astype(np.float32)


def _validate_required_impedance(impedance: np.ndarray | None, cfg: dict) -> None:
    if bool(cfg.get("expected", {}).get("has_impedance", False)) and impedance is None:
        raise ValueError(
            "configs/data/wearable.yaml requires impedance metadata, but Impedance.mat "
            "is missing or does not contain the official 4-D [8,10,2,102] array. "
            "Set expected.has_impedance=false only for an explicitly preregistered "
            "metadata-exclusion run."
        )


def _validate_impedance_signature(
    impedance: np.ndarray | None,
    impedance_order: list[str],
    cfg: dict,
) -> dict[str, Any] | None:
    """Fail fast when a plausible-looking impedance axis is assigned backwards."""
    if impedance is None:
        return None
    axis_means: list[float] = []
    for axis in range(2):
        values = np.asarray(impedance[:, :, axis, :], dtype=np.float64)
        finite_values = values[np.isfinite(values)]
        axis_means.append(float(np.mean(finite_values)) if finite_values.size else float("nan"))
    expected_cfg = cfg.get("expected", {})
    expected = expected_cfg.get("impedance_mean_kohm", {})
    tolerance = float(cfg.get("expected", {}).get("impedance_mean_tolerance_kohm", 1.0))
    if bool(expected_cfg.get("has_impedance", False)):
        if set(expected) != set(impedance_order):
            raise ValueError(
                "Required wearable impedance signature must define exactly the configured "
                f"conditions {impedance_order}; got {sorted(expected)}."
            )
        if not np.isfinite(tolerance) or tolerance <= 0:
            raise ValueError("Wearable impedance signature tolerance must be finite and positive.")
        if any(not np.isfinite(float(value)) for value in expected.values()):
            raise ValueError("Wearable expected impedance means must all be finite.")
    for axis, electrode_type in enumerate(impedance_order):
        expected_mean = expected.get(electrode_type)
        if expected_mean is None:
            continue
        if not np.isfinite(axis_means[axis]):
            raise ValueError(
                "Wearable impedance numeric signature is not finite for "
                f"axis={axis}, configured={electrode_type!r}. The Figshare v4 "
                "Impedance.mat must contain finite measurements."
            )
        if abs(axis_means[axis] - float(expected_mean)) > tolerance:
            raise ValueError(
                "Wearable impedance numeric signature does not match the configured "
                f"condition axis: axis={axis}, configured={electrode_type!r}, "
                f"observed_mean={axis_means[axis]:.6g} kOhm, "
                f"expected_mean={float(expected_mean):.6g}±{tolerance:g} kOhm. "
                "The empirically verified numeric storage order is [dry, wet], "
                "based on the condition means reported with the dataset."
            )
    return {
        "axis_mean_kohm": axis_means,
        "axis_labels": list(impedance_order),
        "expected_mean_kohm": {str(key): float(value) for key, value in expected.items()},
        "tolerance_kohm": tolerance,
    }


def _load_headband_orders(raw_dir: Path) -> dict[int, str]:
    paths = [path for path in raw_dir.rglob("*.mat") if path.stem.lower() == "subjects_information"]
    if not paths:
        return {}
    if len(paths) != 1:
        raise ValueError(
            "Expected exactly one Subjects_Information.mat under "
            f"{raw_dir}, found {len(paths)}: {paths}."
        )
    arrays = _load_arrays(paths[0])
    tables = [
        np.asarray(value, dtype=object)
        for value in arrays.values()
        if np.asarray(value).ndim == 2
        and np.asarray(value).shape[0] >= 2
        and np.asarray(value).shape[1] >= 2
    ]
    if not tables:
        raise ValueError(
            "Subjects_Information.mat does not contain the expected 2-D subject table."
        )
    table = max(tables, key=lambda value: value.size)
    headers = [str(value).strip().lower() for value in table[0]]
    matching_columns = [
        index for index, header in enumerate(headers) if "headband to wear first" in header
    ]
    if len(matching_columns) != 1:
        raise ValueError(
            "Subjects_Information.mat must contain exactly one 'Headband to wear first' column."
        )
    order_column = matching_columns[0]
    result: dict[int, str] = {}
    for row in table[1:]:
        subject_match = re.search(r"(\d+)", str(row[0]))
        if subject_match is None:
            continue
        subject = int(subject_match.group(1))
        order = str(row[order_column]).strip().lower()
        if order not in {"dry", "wet"}:
            raise ValueError(
                f"Invalid headband order for subject {subject}: {row[order_column]!r}."
            )
        if subject in result:
            raise ValueError(f"Duplicate headband order for subject {subject}.")
        result[subject] = order
    return result


def _headband_order_counts(orders: dict[int, str]) -> dict[str, int]:
    return {
        electrode_type: sum(value == electrode_type for value in orders.values())
        for electrode_type in ("dry", "wet")
    }


def _validate_headband_orders(orders: dict[int, str], cfg: dict) -> None:
    expected = cfg.get("expected", {})
    if bool(expected.get("has_headband_order", False)) and not orders:
        raise ValueError(
            "configs/data/wearable.yaml requires Subjects_Information.mat and its "
            "headband-order column."
        )
    n_subjects = expected.get("n_subjects")
    if orders and n_subjects is not None:
        expected_subjects = set(range(1, int(n_subjects) + 1))
        observed_subjects = set(orders)
        if observed_subjects != expected_subjects:
            missing = sorted(expected_subjects - observed_subjects)
            extra = sorted(observed_subjects - expected_subjects)
            raise ValueError(
                "Wearable headband-order subject IDs do not match the official cohort: "
                f"missing={missing[:5]}, extra={extra[:5]}."
            )
    expected_counts = expected.get("headband_order_counts", {})
    observed_counts = _headband_order_counts(orders)
    for electrode_type, count in expected_counts.items():
        if observed_counts.get(str(electrode_type)) != int(count):
            raise ValueError(
                "Wearable headband-order signature mismatch: "
                f"{electrode_type} observed={observed_counts.get(str(electrode_type), 0)}, "
                f"expected={int(count)}."
            )


def _condition_period(headband_order: str | None, electrode_type: str) -> str | None:
    if headband_order not in {"dry", "wet"}:
        return None
    return "first" if headband_order == electrode_type else "second"


def _validate_wearable_config(
    cfg: dict,
    canonical_map: CanonicalChannelMap,
    channel_names: list[str],
    frequencies: list[float],
    phases: list[float],
    electrode_types: list[str],
    impedance_order: list[str],
    c_max: int,
) -> None:
    if cfg.get("dataset_revision") != "wearable_v3":
        raise ValueError("The corrected wearable adapter requires dataset_revision=wearable_v3.")
    if channel_names != OFFICIAL_WEARABLE_CHANNELS:
        raise ValueError(
            "wearable_v3 requires the distributed EEG channel order "
            f"{OFFICIAL_WEARABLE_CHANNELS}; got {channel_names}."
        )
    if electrode_types != ["dry", "wet"]:
        raise ValueError(
            "wearable_v3 EEG condition axis must be [dry, wet], as stated by the "
            "distributed dataset Readme."
        )
    if impedance_order != ["dry", "wet"]:
        raise ValueError(
            "wearable_v3 impedance numeric axis must use the empirically verified "
            "[dry, wet] mapping."
        )
    expected_targets = int(cfg.get("expected", {}).get("n_targets", 12))
    if len(frequencies) != expected_targets or len(phases) != expected_targets:
        raise ValueError(
            "wearable_v3 frequency/phase lengths must match expected.n_targets; "
            f"got frequencies={len(frequencies)}, phases={len(phases)}, "
            f"expected={expected_targets}."
        )
    channel_ids = canonical_map.get_ids(channel_names)
    if 0 in channel_ids or len(channel_ids) != len(set(channel_ids)):
        raise ValueError(
            f"wearable_v3 channels must map to unique known canonical IDs; got {channel_ids}."
        )
    if max(channel_ids) > c_max:
        raise ValueError(
            f"wearable_v3 c_max={c_max} cannot represent canonical channel ID {max(channel_ids)}."
        )


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
    return values if np.isfinite(values).any() else None


def _impedance_electrode_index(electrode_type: str, impedance_order: list[str]) -> int:
    try:
        return impedance_order.index(electrode_type)
    except ValueError as exc:
        raise ValueError(
            f"No impedance axis is configured for electrode_type={electrode_type!r}."
        ) from exc
