from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import yaml
from _bootstrap import add_src_to_path

add_src_to_path()

from cfeg.assets.verify import verify_processed_dir
from cfeg.data.prepare_mat import _load_arrays
from cfeg.data.prepare_wearable import (
    EXTERNAL_CONTINUOUS_SCHEMA,
    QUERY_QC_EXTRACTOR_VERSION,
    _condition_period,
    _impedance_electrode_index,
    _impedance_for,
    _load_headband_orders,
    _load_impedance,
    _validate_impedance_signature,
    _validate_wearable_config,
    _wearable_data,
)
from cfeg.data.preprocess import (
    CanonicalChannelMap,
    PreprocessConfig,
    place_channel_values,
    place_on_canonical_channels,
    preprocess_trial_with_qc,
)
from cfeg.data.schema import load_manifest

REQUIRED_V3_COLUMNS = [
    "query_signal_std",
    "query_signal_std_by_channel",
    "impedance_kohm_by_channel",
    "headband_order",
    "condition_period",
]


def audit_wearable_processed(
    root: Path,
    raw_dir: Path,
    expected_subjects: int,
    *,
    data_config: Path = Path("configs/data/wearable.yaml"),
    deep_signal_check: bool = True,
) -> dict:
    with data_config.open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    expected_rows = expected_subjects * 2 * 10 * 12
    verify_processed_dir(
        root,
        expected_dataset_id="wearable",
        expected_revision="wearable_v3",
        required_manifest_columns=REQUIRED_V3_COLUMNS,
        expected_counts={
            "n_samples": expected_rows,
            "n_subjects": expected_subjects,
            "n_targets": 12,
        },
    )
    _validate_manifest_serializations(root)
    manifest = load_manifest(root)
    if len(manifest) != expected_rows:
        raise ValueError(
            f"Wearable v3 row count mismatch: observed={len(manifest)}, expected={expected_rows}."
        )
    if manifest["subject_id"].nunique() != expected_subjects:
        raise ValueError(
            "Wearable v3 subject count mismatch: "
            f"observed={manifest['subject_id'].nunique()}, expected={expected_subjects}."
        )
    condition_counts = manifest.groupby("electrode_type").size()
    expected_per_condition = expected_subjects * 10 * 12
    if condition_counts.to_dict() != {
        "dry": expected_per_condition,
        "wet": expected_per_condition,
    }:
        raise ValueError(f"Wearable v3 dry/wet row imbalance: {condition_counts.to_dict()}.")
    target_counts = manifest.groupby(["subject_id", "electrode_type", "label"]).size()
    if not (target_counts == 10).all():
        raise ValueError("Every wearable subject×condition×target must contain exactly 10 blocks.")
    order_counts_per_subject = manifest.groupby("subject_id")["headband_order"].nunique()
    if not (order_counts_per_subject == 1).all():
        raise ValueError("Headband order must be constant within each subject.")
    expected_period = np.where(
        manifest["headband_order"].astype(str) == manifest["electrode_type"].astype(str),
        "first",
        "second",
    )
    if not np.array_equal(expected_period, manifest["condition_period"].astype(str).to_numpy()):
        raise ValueError("condition_period is inconsistent with headband order.")
    if manifest["query_signal_std"].isna().any():
        raise ValueError("Wearable v3 contains missing query_signal_std values.")

    vector_lengths = {len(value) for value in manifest["impedance_kohm_by_channel"]}
    finite_counts = {
        sum(item is not None and np.isfinite(item) for item in value)
        for value in manifest["impedance_kohm_by_channel"]
    }
    if vector_lengths != {64} or finite_counts != {8}:
        raise ValueError(
            "Wearable v3 impedance vectors must have 64 canonical slots and 8 finite "
            f"native values; lengths={vector_lengths}, finite_counts={finite_counts}."
        )

    impedance_means = manifest.groupby("electrode_type")["impedance_mean_kohm"].mean().to_dict()
    if not float(impedance_means["dry"]) > float(impedance_means["wet"]):
        raise ValueError(f"Wearable v3 impedance direction is reversed: {impedance_means}.")

    subject_orders = manifest[["subject_id", "headband_order"]].drop_duplicates()
    headband_counts = subject_orders.groupby("headband_order").size().to_dict()
    if expected_subjects == 102 and headband_counts != {"dry": 53, "wet": 49}:
        raise ValueError(f"Full wearable headband-order count mismatch: {headband_counts}.")

    raw_alignment = _audit_raw_alignment(
        root,
        raw_dir,
        manifest,
        config,
        expected_subjects=expected_subjects,
        deep_signal_check=deep_signal_check,
    )

    return {
        "status": "accepted" if deep_signal_check else "metadata_only_not_confirmatory",
        "processed_dir": str(root),
        "dataset_revision": "wearable_v3",
        "n_rows": len(manifest),
        "n_subjects": expected_subjects,
        "condition_rows": {str(key): int(value) for key, value in condition_counts.items()},
        "impedance_mean_kohm": {str(key): float(value) for key, value in impedance_means.items()},
        "headband_order_subjects": {str(key): int(value) for key, value in headband_counts.items()},
        "query_qc_missing": int(manifest["query_signal_std"].isna().sum()),
        "impedance_vector_length": 64,
        "impedance_finite_channels": 8,
        "raw_alignment": raw_alignment,
    }


def _validate_manifest_serializations(root: Path) -> None:
    parquet_path = root / "manifest.parquet"
    jsonl_path = root / "manifest.jsonl"
    if not (parquet_path.exists() and jsonl_path.exists()):
        return
    parquet = _normalize_object_nulls(pd.read_parquet(parquet_path))
    jsonl = _normalize_object_nulls(pd.read_json(jsonl_path, lines=True))
    try:
        pd.testing.assert_frame_equal(
            parquet,
            jsonl,
            check_dtype=False,
            check_exact=False,
            rtol=1e-7,
            atol=1e-7,
        )
    except AssertionError as exc:
        raise ValueError(
            "manifest.parquet and manifest.jsonl disagree; remove stale artifacts and "
            "re-run preparation."
        ) from exc


def _normalize_object_nulls(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    for column in out.columns:
        if out[column].isna().all():
            out[column] = pd.Series([None] * len(out), index=out.index, dtype=object)
        elif out[column].dtype == object:
            out[column] = out[column].map(_normalize_object_value)
    return out


def _normalize_object_value(value):
    if isinstance(value, np.ndarray):
        return [_normalize_object_value(item) for item in value.tolist()]
    if isinstance(value, list):
        return [_normalize_object_value(item) for item in value]
    if isinstance(value, dict):
        return {key: _normalize_object_value(item) for key, item in value.items()}
    if value is None:
        return None
    try:
        return None if bool(pd.isna(value)) else value
    except (TypeError, ValueError):
        return value


def _audit_raw_alignment(
    root: Path,
    raw_dir: Path,
    manifest: pd.DataFrame,
    config: dict,
    *,
    expected_subjects: int,
    deep_signal_check: bool,
) -> dict:
    if not raw_dir.exists():
        raise ValueError(f"Raw wearable directory does not exist: {raw_dir}.")
    pcfg = PreprocessConfig.from_dict(config.get("preprocess"))
    channel_map = CanonicalChannelMap.from_yaml()
    channel_names = list(config["channel_names"])
    frequencies = [float(value) for value in config["class_frequencies"]]
    phases = [float(value) for value in config["class_phases"]]
    electrode_types = list(config["electrode_types"])
    impedance_order = list(config["impedance_electrode_types"])
    _validate_wearable_config(
        config,
        channel_map,
        channel_names,
        frequencies,
        phases,
        electrode_types,
        impedance_order,
        pcfg.c_max,
    )
    _validate_fixed_manifest_contract(root, manifest, config, pcfg)

    subject_files = sorted(
        path for path in raw_dir.rglob("*.mat") if re.fullmatch(r"S\d{3}", path.stem, re.IGNORECASE)
    )
    raw_subject_ids = {f"sub{int(path.stem[1:]):03d}" for path in subject_files}
    manifest_subject_ids = set(manifest["subject_id"].astype(str))
    if len(subject_files) != expected_subjects or raw_subject_ids != manifest_subject_ids:
        raise ValueError(
            "Raw/manifest subject identity mismatch: "
            f"raw={sorted(raw_subject_ids)}, manifest={sorted(manifest_subject_ids)}."
        )
    source_basenames = manifest["source_file"].astype(str).map(lambda value: Path(value).name)
    expected_source = manifest["subject_id"].map(
        lambda value: f"S{int(str(value).removeprefix('sub')):03d}.mat"
    )
    if not source_basenames.equals(expected_source):
        raise ValueError("Manifest source_file does not match its subject_id.")

    provenance = _validate_figshare_raw_provenance(raw_dir, subject_files)
    impedance = _load_impedance(raw_dir)
    if impedance is None:
        raise ValueError("Raw Impedance.mat could not be loaded for cross-checking.")
    impedance_signature = _validate_impedance_signature(impedance, impedance_order, config)
    headband_orders = _load_headband_orders(raw_dir)

    with (root / "asset_info.json").open(encoding="utf-8") as handle:
        asset_info = json.load(handle)
    expected_asset_fields = {
        "dataset_id": "wearable",
        "dataset_revision": "wearable_v3",
        "source_version": 4,
        "query_qc_extractor_version": QUERY_QC_EXTRACTOR_VERSION,
        "external_continuous_schema": EXTERNAL_CONTINUOUS_SCHEMA,
        "processed_subject_count": expected_subjects,
        "schema": "[channel,time,electrode,block,target]",
        "electrode_types": electrode_types,
        "impedance_electrode_types": impedance_order,
        "has_impedance": True,
        "per_channel_impedance_preserved": True,
        "has_headband_order": True,
        "acquisition_provenance": config.get("acquisition_provenance", {}),
    }
    for key, expected in expected_asset_fields.items():
        if asset_info.get(key) != expected:
            raise ValueError(
                f"asset_info provenance mismatch for {key}: expected={expected!r}, "
                f"observed={asset_info.get(key, 'missing')!r}."
            )
    if set(asset_info.get("processed_subject_ids", [])) != raw_subject_ids:
        raise ValueError("asset_info processed_subject_ids does not match the raw cohort.")
    stored_signature = asset_info.get("impedance_numeric_signature", {})
    if stored_signature.get("axis_labels") != impedance_signature.get(
        "axis_labels"
    ) or not np.allclose(
        stored_signature.get("axis_mean_kohm", []),
        impedance_signature.get("axis_mean_kohm", []),
        rtol=0.0,
        atol=1e-9,
    ):
        raise ValueError("asset_info impedance numeric signature does not match the raw file.")
    source_order_counts = {
        str(key): int(value)
        for key, value in pd.Series(headband_orders).value_counts().to_dict().items()
    }
    if asset_info.get("source_cohort_headband_order_counts") != source_order_counts:
        raise ValueError(
            "asset_info source cohort headband-order counts do not match Subjects_Information.mat."
        )

    dummy = np.zeros((len(channel_names), 1), dtype=np.float32)
    _, expected_mask, expected_channel_ids = place_on_canonical_channels(
        dummy, channel_names, channel_map, pcfg.c_max
    )
    expected_channel_ids_array = np.asarray(expected_channel_ids, dtype=np.int64)

    row_lookup: dict[tuple[str, str, int, int], int] = {}
    for row_index, row in manifest.iterrows():
        subject_id = str(row["subject_id"])
        electrode_type = str(row["electrode_type"])
        block = _numeric_suffix(row["run_id"], "block")
        target = _numeric_suffix(row["trial_id"], "target")
        key = (subject_id, electrode_type, block, target)
        if key in row_lookup:
            raise ValueError(f"Duplicate wearable grid key: {key}.")
        if str(row["session_id"]) != electrode_type or int(row["label"]) != target:
            raise ValueError(f"Wearable session/label identity mismatch for {key}.")
        expected_sample_id = (
            f"wearable_{subject_id}_{electrode_type}_block{block:02d}_target{target:02d}"
        )
        if str(row["sample_id"]) != expected_sample_id:
            raise ValueError(
                f"Wearable sample_id mismatch: expected={expected_sample_id!r}, "
                f"observed={row['sample_id']!r}."
            )
        row_lookup[key] = int(row_index)

    expected_keys = {
        (subject_id, electrode_type, block, target)
        for subject_id in raw_subject_ids
        for electrode_type in electrode_types
        for block in range(1, 11)
        for target in range(12)
    }
    if set(row_lookup) != expected_keys:
        missing = sorted(expected_keys - set(row_lookup))[:5]
        extra = sorted(set(row_lookup) - expected_keys)[:5]
        raise ValueError(f"Wearable exact grid mismatch: missing={missing}, extra={extra}.")

    signal_max_abs_error = 0.0
    query_vector_max_abs_error = 0.0
    impedance_vector_max_abs_error = 0.0
    signal_content_digest = hashlib.sha256()
    # One logical trial spans 32 compressed x chunks in the full 102-subject file.
    # Keep those chunks resident while the next rows reuse them; h5py's 1 MiB
    # default cache otherwise re-reads and re-decompresses the same chunks hundreds
    # of times during a full-cohort audit.
    with h5py.File(
        root / "signals.h5",
        "r",
        rdcc_nbytes=64 * 1024 * 1024,
        rdcc_nslots=100_003,
    ) as h5:
        signal_dataset = h5["x"]
        mask_dataset = h5["channel_mask"]
        label_dataset = h5["y"]
        expected_signal_shape = (
            len(manifest),
            pcfg.c_max,
            int(pcfg.window_duration_sec * pcfg.target_sfreq),
        )
        if tuple(signal_dataset.shape) != expected_signal_shape:
            raise ValueError(
                f"Wearable HDF5 signal shape mismatch: observed={tuple(signal_dataset.shape)}, "
                f"expected={expected_signal_shape}."
            )
        for subject_file in subject_files:
            subject_number = int(subject_file.stem[1:])
            subject_id = f"sub{subject_number:03d}"
            subject_index = subject_number - 1
            data = _wearable_data(_load_arrays(subject_file)) if deep_signal_check else None
            headband_order = headband_orders.get(subject_number)
            for electrode_index, electrode_type in enumerate(electrode_types):
                impedance_index = _impedance_electrode_index(electrode_type, impedance_order)
                for block_index in range(10):
                    native_impedance = _impedance_for(
                        impedance, subject_index, impedance_index, block_index
                    )
                    if native_impedance is None:
                        raise ValueError(
                            f"Raw impedance missing for {subject_id}/{electrode_type}/"
                            f"block{block_index + 1:02d}."
                        )
                    expected_impedance = place_channel_values(
                        native_impedance, channel_names, channel_map, pcfg.c_max
                    )
                    finite_impedance = expected_impedance[np.isfinite(expected_impedance)]
                    for target_index in range(12):
                        row_index = row_lookup[
                            (subject_id, electrode_type, block_index + 1, target_index)
                        ]
                        row = manifest.iloc[row_index]
                        h5_index = int(row["h5_index"])
                        observed_mask = mask_dataset[h5_index].astype(bool)
                        observed_y = int(label_dataset[h5_index])
                        if (
                            not np.array_equal(observed_mask, expected_mask)
                            or observed_y != target_index
                        ):
                            raise ValueError(f"HDF5 mask/label mismatch at {row['sample_id']}.")
                        observed_ids = _manifest_vector(
                            row["canonical_channel_ids"], pcfg.c_max, missing=0, dtype=np.int64
                        )
                        if not np.array_equal(observed_ids, expected_channel_ids_array):
                            raise ValueError(
                                f"Canonical channel IDs mismatch at {row['sample_id']}."
                            )
                        observed_impedance = _manifest_vector(
                            row["impedance_kohm_by_channel"],
                            pcfg.c_max,
                            missing=np.nan,
                            dtype=np.float32,
                        )
                        if not np.array_equal(np.isfinite(observed_impedance), expected_mask):
                            raise ValueError(
                                f"Impedance/channel mask mismatch at {row['sample_id']}."
                            )
                        impedance_vector_max_abs_error = max(
                            impedance_vector_max_abs_error,
                            _max_finite_abs_error(observed_impedance, expected_impedance),
                        )
                        if not np.allclose(
                            observed_impedance,
                            expected_impedance,
                            rtol=1e-6,
                            atol=1e-5,
                            equal_nan=True,
                        ):
                            raise ValueError(
                                f"Raw/manifest impedance mismatch at {row['sample_id']}."
                            )
                        if not np.isclose(
                            float(row["impedance_mean_kohm"]),
                            float(np.mean(finite_impedance)),
                            rtol=1e-6,
                            atol=1e-5,
                        ) or not np.isclose(
                            float(row["impedance_max_kohm"]),
                            float(np.max(finite_impedance)),
                            rtol=1e-6,
                            atol=1e-5,
                        ):
                            raise ValueError(
                                f"Impedance vector/scalar mismatch at {row['sample_id']}."
                            )
                        if str(row["headband_order"]) != headband_order or str(
                            row["condition_period"]
                        ) != _condition_period(headband_order, electrode_type):
                            raise ValueError(
                                f"Raw headband order join mismatch at {row['sample_id']}."
                            )
                        if not np.isclose(
                            float(row["stimulus_frequency_hz"]), frequencies[target_index]
                        ) or not np.isclose(float(row["stimulus_phase_rad"]), phases[target_index]):
                            raise ValueError(
                                f"Label/frequency/phase mismatch at {row['sample_id']}."
                            )

                        observed_x = signal_dataset[h5_index].astype(np.float32)
                        signal_content_digest.update(observed_x.tobytes())
                        signal_content_digest.update(observed_mask.astype(np.uint8).tobytes())
                        signal_content_digest.update(
                            np.asarray([observed_y], dtype=np.int64).tobytes()
                        )
                        if not np.isfinite(observed_x).all():
                            raise ValueError(f"Non-finite HDF5 signal at {row['sample_id']}.")
                        if not np.allclose(observed_x[~expected_mask], 0.0, atol=0.0):
                            raise ValueError(
                                f"Inactive HDF5 channels are non-zero at {row['sample_id']}."
                            )
                        if deep_signal_check:
                            raw_trial = data[:, :, electrode_index, block_index, target_index]
                            expected_x, signal_mask, signal_ids, _, query_qc = (
                                preprocess_trial_with_qc(
                                    raw_trial,
                                    channel_names,
                                    float(config["raw_sfreq"]),
                                    pcfg,
                                    channel_map,
                                )
                            )
                            signal_max_abs_error = max(
                                signal_max_abs_error,
                                float(np.max(np.abs(observed_x - expected_x))),
                            )
                            if not np.allclose(observed_x, expected_x, rtol=0.0, atol=1e-6):
                                raise ValueError(f"Raw EEG/HDF5 mismatch at {row['sample_id']}.")
                            if not np.array_equal(signal_mask, expected_mask) or not np.array_equal(
                                np.asarray(signal_ids), expected_channel_ids_array
                            ):
                                raise ValueError(
                                    f"Raw preprocessing channel mapping mismatch at {row['sample_id']}."
                                )
                            observed_query = _manifest_vector(
                                row["query_signal_std_by_channel"],
                                pcfg.c_max,
                                missing=np.nan,
                                dtype=np.float32,
                            )
                            query_vector_max_abs_error = max(
                                query_vector_max_abs_error,
                                _max_finite_abs_error(
                                    observed_query, query_qc.signal_std_by_channel
                                ),
                            )
                            if not np.allclose(
                                observed_query,
                                query_qc.signal_std_by_channel,
                                rtol=1e-6,
                                atol=1e-6,
                                equal_nan=True,
                            ) or not np.isclose(
                                float(row["query_signal_std"]),
                                float(query_qc.signal_std),
                                rtol=1e-6,
                                atol=1e-6,
                            ):
                                raise ValueError(f"Raw/query-QC mismatch at {row['sample_id']}.")

    return {
        "status": "deep_verified_atol_1e-6" if deep_signal_check else "metadata_only",
        "raw_dir": str(raw_dir),
        "figshare": provenance,
        "impedance_numeric_signature": impedance_signature,
        "signal_max_abs_error": signal_max_abs_error if deep_signal_check else None,
        "query_vector_max_abs_error": (query_vector_max_abs_error if deep_signal_check else None),
        "impedance_vector_max_abs_error": impedance_vector_max_abs_error,
        "signals_content_sha256": signal_content_digest.hexdigest(),
        "signals_file_sha256": _file_digest(root / "signals.h5", "sha256"),
        "processed_metadata_sha256": {
            name: _file_digest(root / name, "sha256")
            for name in (
                "asset_info.json",
                "manifest.jsonl",
                "manifest.parquet",
                "preprocess_config.yaml",
                "class_map.json",
            )
            if (root / name).exists()
        },
    }


def _validate_fixed_manifest_contract(
    root: Path, manifest: pd.DataFrame, config: dict, pcfg: PreprocessConfig
) -> None:
    expected_scalars = {
        "dataset_id": "wearable",
        "reference": config["reference"],
        "hardware_id": config["hardware_id"],
        "cap_type": "wearable",
        "sfreq_original": float(config["raw_sfreq"]),
        "sfreq_processed": float(pcfg.target_sfreq),
        "window_start_sec": float(pcfg.window_start_sec),
        "window_duration_sec": float(pcfg.window_duration_sec),
        "n_channels_original": len(config["channel_names"]),
        "n_channels_used": len(config["channel_names"]),
    }
    for column, expected in expected_scalars.items():
        observed = manifest[column]
        if isinstance(expected, float):
            valid = np.allclose(observed.astype(float).to_numpy(), expected, rtol=0.0, atol=1e-9)
        else:
            valid = set(observed.astype(str)) == {str(expected)}
        if not valid:
            raise ValueError(
                f"Manifest fixed metadata mismatch for {column}: expected={expected!r}."
            )
    expected_names = list(config["channel_names"])
    for column in ("channel_names_original", "channel_names_used"):
        if any(list(value) != expected_names for value in manifest[column]):
            raise ValueError(f"Manifest {column} does not match the distributed channel order.")

    with (root / "preprocess_config.yaml").open(encoding="utf-8") as handle:
        observed_preprocess = yaml.safe_load(handle)
    expected_preprocess = pcfg.__dict__
    if observed_preprocess != expected_preprocess:
        raise ValueError(
            "preprocess_config.yaml does not match configs/data/wearable.yaml; re-prepare "
            "the processed asset."
        )


def _validate_figshare_raw_provenance(raw_dir: Path, subject_files: list[Path]) -> dict:
    metadata_paths = list(raw_dir.rglob("figshare_article.json"))
    if len(metadata_paths) != 1:
        raise ValueError(
            f"Expected exactly one figshare_article.json under {raw_dir}, found "
            f"{len(metadata_paths)}."
        )
    with metadata_paths[0].open(encoding="utf-8") as handle:
        metadata = json.load(handle)
    if int(metadata.get("id", -1)) != 13560281 or int(metadata.get("version", -1)) != 4:
        raise ValueError(
            "Raw Figshare provenance must be article 13560281 version 4; got "
            f"id={metadata.get('id')}, version={metadata.get('version')}."
        )
    records = {str(item["name"]): item for item in metadata.get("files", [])}
    required_paths = [
        *subject_files,
        *[
            path
            for path in raw_dir.rglob("*.mat")
            if path.stem.lower() in {"impedance", "subjects_information"}
        ],
    ]
    checked = 0
    for path in required_paths:
        record = records.get(path.name)
        if record is None:
            raise ValueError(f"Raw file {path.name} is absent from figshare_article.json.")
        if path.stat().st_size != int(record["size"]):
            raise ValueError(f"Raw file size mismatch for {path.name}.")
        observed_md5 = _file_digest(path, "md5")
        if observed_md5.lower() != str(record["computed_md5"]).lower():
            raise ValueError(f"Raw file MD5 mismatch for {path.name}.")
        checked += 1
    return {
        "article_id": 13560281,
        "version": 4,
        "verified_file_count": checked,
        "metadata_sha256": _file_digest(metadata_paths[0], "sha256"),
    }


def _numeric_suffix(value, prefix: str) -> int:
    match = re.fullmatch(rf"{re.escape(prefix)}(\d+)", str(value))
    if not match:
        raise ValueError(f"Expected {prefix}<integer>, got {value!r}.")
    return int(match.group(1))


def _manifest_vector(value, length: int, *, missing, dtype) -> np.ndarray:
    if not isinstance(value, (list, np.ndarray)) or len(value) != length:
        raise ValueError(f"Expected manifest vector of length {length}, got {type(value)}.")
    return np.asarray([missing if item is None else item for item in value], dtype=dtype)


def _max_finite_abs_error(left: np.ndarray, right: np.ndarray) -> float:
    finite = np.isfinite(left) & np.isfinite(right)
    return float(np.max(np.abs(left[finite] - right[finite]))) if finite.any() else 0.0


def _file_digest(path: Path, algorithm: str) -> str:
    digest = hashlib.new(algorithm)
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the publication-facing wearable_v3 acceptance audit."
    )
    parser.add_argument("--processed-dir", required=True)
    parser.add_argument("--raw-dir", required=True)
    parser.add_argument("--data-config", default="configs/data/wearable.yaml")
    parser.add_argument("--expected-subjects", type=int, default=102)
    parser.add_argument(
        "--receipt",
        default=None,
        help="Audit receipt path (default: <processed-dir>/wearable_v3_audit_receipt.json).",
    )
    parser.add_argument(
        "--skip-deep-signal-check",
        action="store_true",
        help="Skip raw EEG reprocessing; not acceptable for a confirmatory asset receipt.",
    )
    args = parser.parse_args()
    result = audit_wearable_processed(
        Path(args.processed_dir),
        Path(args.raw_dir),
        expected_subjects=args.expected_subjects,
        data_config=Path(args.data_config),
        deep_signal_check=not args.skip_deep_signal_check,
    )
    if result["status"] == "accepted":
        receipt_path = (
            Path(args.receipt)
            if args.receipt
            else (Path(args.processed_dir) / "wearable_v3_audit_receipt.json")
        )
        receipt_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = receipt_path.with_suffix(f"{receipt_path.suffix}.tmp")
        with temporary.open("w", encoding="utf-8") as handle:
            json.dump(result, handle, indent=2)
        os.replace(temporary, receipt_path)
        result["receipt_path"] = str(receipt_path)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
