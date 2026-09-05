from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from pathlib import Path

import h5py
import numpy as np
from torch.utils.data import Dataset

from cfeg.data.io_hdf5 import HDF5SampleReader
from cfeg.data.schema import EEGSample, load_manifest, validate_manifest

PROTOCOL_V04_REQUIRED_COLUMNS = ["query_signal_std", "query_signal_std_by_channel"]
REVISION_REQUIRED_COLUMNS = {
    "wearable_v3": [
        *PROTOCOL_V04_REQUIRED_COLUMNS,
        "impedance_kohm_by_channel",
        "headband_order",
        "condition_period",
    ],
    "synthetic_quality_v1": [
        *PROTOCOL_V04_REQUIRED_COLUMNS,
        "impedance_kohm_by_channel",
        "acquisition_block_id",
        "metadata_measurement_id",
        "metadata_measurement_time_sec",
        "query_time_sec",
        "impedance_unit",
    ],
}
WEARABLE_V3_CANONICAL_IDS = [54, 55, 56, 57, 58, 61, 62, 63]


class EEGProcessedDataset(Dataset):
    def __init__(
        self,
        processed_dirs: list[str | Path],
        indices: list[int] | None = None,
        expected_revisions: Mapping[str, str] | None = None,
        expected_protocol: Mapping[str, str] | None = None,
        expected_dataset_counts: Mapping[str, Mapping[str, int]] | None = None,
        require_audit_receipt: bool = False,
        allowed_subject_ids: set[str] | None = None,
        persistent_hdf5_handles: bool = False,
        preload_hdf5_to_memory: bool = False,
    ):
        self.roots = [Path(p) for p in processed_dirs]
        self.entries: list[tuple[Path, int, dict]] = []
        self.asset_infos: list[dict] = []
        self._sample_reader = HDF5SampleReader(persistent=persistent_hdf5_handles)
        observed_dataset_ids: set[str] = set()
        for root in self.roots:
            info = _validate_processed_revision(root, expected_revisions or {})
            manifest = load_manifest(root)
            validate_manifest(manifest)
            dataset_id = _validate_manifest_asset_identity(root, manifest, info)
            _validate_protocol_contract(root, manifest, info, expected_protocol or {})
            _validate_dataset_counts(
                root,
                manifest,
                (expected_dataset_counts or {}).get(dataset_id, {}),
            )
            if require_audit_receipt:
                _validate_audit_receipt(root, manifest, info)
            _validate_manifest_class_map(root, manifest)
            observed_dataset_ids.add(dataset_id)
            selected_manifest = manifest
            if allowed_subject_ids is not None:
                observed_subjects = set(manifest["subject_id"].astype(str))
                missing_subjects = sorted(set(allowed_subject_ids) - observed_subjects)
                if missing_subjects:
                    raise ValueError(
                        f"Allowed subject IDs are absent from {root}: {missing_subjects}."
                    )
                selected_manifest = manifest.loc[
                    manifest["subject_id"].astype(str).isin(allowed_subject_ids)
                ]
            self.asset_infos.append(info)
            for _, row in selected_manifest.iterrows():
                self.entries.append((root, int(row["h5_index"]), row.to_dict()))
        missing_expected = set(expected_revisions or {}) - observed_dataset_ids
        if missing_expected:
            raise ValueError(
                "No processed root was supplied for expected dataset revision(s): "
                f"{sorted(missing_expected)}."
            )
        if indices is not None:
            self.entries = [self.entries[i] for i in indices]
        self._preloaded_samples: dict[int, tuple[np.ndarray, np.ndarray, int]] = {}
        if preload_hdf5_to_memory:
            self.preload_indices(range(len(self.entries)))
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
        if index in self._preloaded_samples:
            x, mask, y = self._preloaded_samples[index]
        else:
            x, mask, y = self._sample_reader.read(root / "signals.h5", h5_index)
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
            query_signal_std=_float_or_none(row.get("query_signal_std")),
            query_signal_std_by_channel=_to_float_array(
                row.get("query_signal_std_by_channel"), length=x.shape[0]
            ),
            impedance_kohm_by_channel=_to_float_array(
                row.get("impedance_kohm_by_channel"), length=x.shape[0]
            ),
            headband_order=_none_if_nan(row.get("headband_order")),
            condition_period=_none_if_nan(row.get("condition_period")),
        )

    def preload_indices(self, indices) -> None:
        """Preload only authorized indices without touching sealed held-out signals."""

        requested = sorted({int(index) for index in indices})
        if any(index < 0 or index >= len(self.entries) for index in requested):
            raise IndexError("Preload indices fall outside the processed dataset.")
        missing = [index for index in requested if index not in self._preloaded_samples]
        if not missing:
            return
        self._preloaded_samples.update(self._read_selected_samples(missing))
        self._sample_reader.close()

    def _read_selected_samples(
        self, indices: list[int]
    ) -> dict[int, tuple[np.ndarray, np.ndarray, int]]:
        """Read contiguous HDF5 runs once to avoid shuffled gzip-chunk thrashing."""

        grouped: dict[Path, list[tuple[int, int]]] = {}
        for position in indices:
            root, h5_index, _ = self.entries[position]
            grouped.setdefault(root / "signals.h5", []).append((position, int(h5_index)))
        samples: dict[int, tuple[np.ndarray, np.ndarray, int]] = {}
        for path, positions in grouped.items():
            ordered = sorted(positions, key=lambda value: value[1])
            with h5py.File(path, "r") as handle:
                for run in _contiguous_index_runs(ordered):
                    start = run[0][1]
                    stop = run[-1][1] + 1
                    x = handle["x"][start:stop].astype("float32")
                    masks = handle["channel_mask"][start:stop].astype("bool")
                    labels = handle["y"][start:stop].astype("int64")
                    for position, h5_index in run:
                        offset = h5_index - start
                        samples[position] = (x[offset], masks[offset], int(labels[offset]))
        if set(samples) != set(indices):  # pragma: no cover - defensive invariant
            raise RuntimeError("Failed to preload every selected HDF5 sample.")
        return samples


def _contiguous_index_runs(
    positions: list[tuple[int, int]],
) -> list[list[tuple[int, int]]]:
    runs: list[list[tuple[int, int]]] = []
    for value in positions:
        if not runs or value[1] != runs[-1][-1][1] + 1:
            runs.append([value])
        else:
            runs[-1].append(value)
    return runs


def _validate_processed_revision(root: Path, expected_revisions: Mapping[str, str]) -> dict:
    path = root / "asset_info.json"
    if not path.exists():
        if expected_revisions:
            raise ValueError(
                f"Processed asset {root} has no asset_info.json; cannot verify revision."
            )
        return {}
    with path.open("r", encoding="utf-8") as handle:
        info = json.load(handle)
    if not expected_revisions:
        return info
    if not info.get("dataset_id"):
        raise ValueError(f"Processed asset {root} has no dataset_id in asset_info.json.")
    dataset_id = str(info["dataset_id"])
    expected = expected_revisions.get(dataset_id)
    if expected is None:
        raise ValueError(
            f"Processed asset {root} declares unexpected dataset_id={dataset_id!r}; "
            f"expected one of {sorted(expected_revisions)}."
        )
    observed = str(info.get("dataset_revision", "missing"))
    if observed != str(expected):
        raise ValueError(
            f"Processed {dataset_id} revision mismatch under {root}: "
            f"expected {expected!r}, observed {observed!r}. Re-run preparation into "
            "the revisioned output directory before training or evaluation."
        )
    return info


def _validate_manifest_asset_identity(root: Path, manifest, info: Mapping) -> str:
    manifest_ids = {str(value) for value in manifest["dataset_id"].dropna().unique()}
    if len(manifest_ids) != 1:
        raise ValueError(
            f"Processed manifest under {root} must contain exactly one dataset_id; "
            f"got {sorted(manifest_ids)}."
        )
    dataset_id = next(iter(manifest_ids))
    asset_dataset_id = info.get("dataset_id")
    if asset_dataset_id is not None and str(asset_dataset_id) != dataset_id:
        raise ValueError(
            f"Manifest/asset dataset ID mismatch under {root}: manifest={dataset_id!r}, "
            f"asset_info={asset_dataset_id!r}."
        )
    return dataset_id


def _validate_protocol_contract(
    root: Path,
    manifest,
    info: Mapping,
    expected_protocol: Mapping[str, str],
) -> None:
    revision = str(info.get("dataset_revision", ""))
    required = list(REVISION_REQUIRED_COLUMNS.get(revision, []))
    if str(expected_protocol.get("metadata_contract_version", "legacy")) == "0.4-dev":
        required.extend(PROTOCOL_V04_REQUIRED_COLUMNS)
        for key in ("query_qc_extractor_version", "external_continuous_schema"):
            expected = expected_protocol.get(key)
            if expected is not None and str(info.get(key, "missing")) != str(expected):
                raise ValueError(
                    f"Processed protocol provenance mismatch under {root}: {key} expected "
                    f"{expected!r}, observed {info.get(key, 'missing')!r}. Re-prepare the asset."
                )
    missing = sorted(set(required) - set(manifest.columns))
    if missing:
        raise ValueError(f"Processed asset {root} is missing protocol/revision columns: {missing}.")
    if required:
        _validate_aligned_protocol_vectors(root, manifest, revision=revision)


def _validate_aligned_protocol_vectors(root: Path, manifest, *, revision: str) -> None:
    query_scalar = manifest["query_signal_std"].astype(float).to_numpy()
    if not np.isfinite(query_scalar).all():
        raise ValueError(f"Processed asset {root} contains non-finite query_signal_std.")
    with h5py.File(root / "signals.h5", "r") as h5:
        masks = h5["channel_mask"][:].astype(bool)
    c_max = masks.shape[1]
    ids = _stack_manifest_vectors(manifest["canonical_channel_ids"], c_max, dtype=np.int64)
    query = _stack_manifest_vectors(
        manifest["query_signal_std_by_channel"], c_max, dtype=np.float32
    )
    positive = ids > 0
    slot_ids = np.arange(1, c_max + 1, dtype=np.int64)[None, :]
    if np.any(ids < 0) or np.any(positive & ~masks) or np.any(positive & (ids != slot_ids)):
        raise ValueError(
            f"Processed asset {root} has positive canonical IDs outside active masks or "
            "outside their canonical tensor slots."
        )
    if not np.array_equal(np.isfinite(query), masks):
        raise ValueError(
            f"Processed asset {root} query-QC vector positions do not match channel masks."
        )
    query_median = np.nanmedian(query, axis=1)
    if not np.allclose(query_scalar, query_median, rtol=1e-6, atol=1e-6):
        raise ValueError(
            f"Processed asset {root} query_signal_std is inconsistent with its channel vector."
        )
    if revision == "wearable_v3":
        if not np.array_equal(positive, masks):
            raise ValueError(
                f"wearable_v3 under {root} must have a known canonical ID for every "
                "active official channel."
            )
        expected_ids = np.zeros((c_max,), dtype=np.int64)
        expected_ids[np.asarray(WEARABLE_V3_CANONICAL_IDS) - 1] = WEARABLE_V3_CANONICAL_IDS
        if not np.all(ids == expected_ids[None, :]):
            raise ValueError(
                f"wearable_v3 under {root} must use exactly the official eight canonical "
                f"channels {WEARABLE_V3_CANONICAL_IDS}."
            )
    if revision in {"wearable_v3", "synthetic_quality_v1"}:
        impedance = _stack_manifest_vectors(
            manifest["impedance_kohm_by_channel"], c_max, dtype=np.float32
        )
        if not np.array_equal(np.isfinite(impedance), masks):
            raise ValueError(
                f"{revision} impedance vector positions do not match channel masks under {root}."
            )
        finite_values = np.where(masks, impedance, np.nan)
        means = np.nanmean(finite_values, axis=1)
        maxima = np.nanmax(finite_values, axis=1)
        scalar_means = manifest["impedance_mean_kohm"].astype(float).to_numpy()
        scalar_maxima = manifest["impedance_max_kohm"].astype(float).to_numpy()
        if (
            not np.isfinite(finite_values[masks]).all()
            or np.any(finite_values[masks] < 0)
            or not np.allclose(means, scalar_means, rtol=1e-6, atol=1e-5)
            or not np.allclose(maxima, scalar_maxima, rtol=1e-6, atol=1e-5)
        ):
            raise ValueError(
                f"{revision} impedance vectors/scalars are inconsistent under {root}."
            )
        if revision == "synthetic_quality_v1":
            units = set(manifest["impedance_unit"].astype(str))
            if units != {"kOhm"}:
                raise ValueError(
                    "synthetic_quality_v1 impedance_unit must be exactly 'kOhm'."
                )
            measurement_time = manifest["metadata_measurement_time_sec"].astype(float)
            query_time = manifest["query_time_sec"].astype(float)
            if not np.isfinite(measurement_time).all() or not np.all(
                query_time > measurement_time
            ):
                raise ValueError(
                    "synthetic_quality_v1 metadata measurements must be finite and pre-query."
                )


def _stack_manifest_vectors(series, length: int, *, dtype) -> np.ndarray:
    rows = []
    missing_value = 0 if np.issubdtype(np.dtype(dtype), np.integer) else np.nan
    for value in series:
        if not isinstance(value, (list, np.ndarray)) or len(value) != length:
            raise ValueError(f"Expected manifest vectors of length {length}, got {type(value)}.")
        rows.append([missing_value if item is None else item for item in value])
    return np.asarray(rows, dtype=dtype)


def _validate_dataset_counts(root: Path, manifest, expected: Mapping[str, int]) -> None:
    if not expected:
        return
    observed = {
        "n_samples": len(manifest),
        "n_subjects": int(manifest["subject_id"].nunique()),
        "n_targets": int(manifest["label"].nunique()),
    }
    for key, value in expected.items():
        if key in observed and observed[key] != int(value):
            raise ValueError(
                f"Processed cohort completeness mismatch under {root}: {key} expected "
                f"{int(value)}, observed {observed[key]}."
            )


def _validate_audit_receipt(root: Path, manifest, info: Mapping) -> None:
    path = root / "wearable_v3_audit_receipt.json"
    if not path.exists():
        raise ValueError(
            f"Processed asset {root} has no wearable_v3 audit receipt. Run "
            "scripts/audit_wearable_processed.py with the raw directory first."
        )
    with path.open(encoding="utf-8") as handle:
        receipt = json.load(handle)
    if (
        receipt.get("status") != "accepted"
        or receipt.get("dataset_revision") != info.get("dataset_revision")
        or int(receipt.get("n_rows", -1)) != len(manifest)
        or not str(receipt.get("raw_alignment", {}).get("status", "")).startswith("deep_verified")
    ):
        raise ValueError(f"Processed asset {root} has an invalid or partial audit receipt.")
    expected_hashes = receipt.get("raw_alignment", {}).get("processed_metadata_sha256", {})
    required_hashes = {
        "asset_info.json",
        "manifest.jsonl",
        "preprocess_config.yaml",
        "class_map.json",
    }
    signal_digest = str(receipt.get("raw_alignment", {}).get("signals_content_sha256", ""))
    expected_signal_file_hash = str(receipt.get("raw_alignment", {}).get("signals_file_sha256", ""))
    if not required_hashes.issubset(expected_hashes) or not (
        len(signal_digest) == 64
        and all(char in "0123456789abcdef" for char in signal_digest.lower())
        and len(expected_signal_file_hash) == 64
        and all(char in "0123456789abcdef" for char in expected_signal_file_hash.lower())
    ):
        raise ValueError(f"Processed asset {root} has an incomplete audit fingerprint receipt.")
    for name, expected in expected_hashes.items():
        artifact = root / name
        if not artifact.exists() or _sha256_file(artifact) != str(expected):
            raise ValueError(
                f"Processed artifact {name} changed after the audit receipt under {root}."
            )
    signals_path = root / "signals.h5"
    if not signals_path.exists() or _sha256_file(signals_path) != expected_signal_file_hash:
        raise ValueError(
            f"Processed artifact signals.h5 changed after the audit receipt under {root}."
        )
    artifact_paths = [
        root / name
        for name in (
            "signals.h5",
            "asset_info.json",
            "manifest.jsonl",
            "manifest.parquet",
            "preprocess_config.yaml",
            "class_map.json",
        )
        if (root / name).exists()
    ]
    if any(artifact.stat().st_mtime_ns > path.stat().st_mtime_ns for artifact in artifact_paths):
        raise ValueError(
            f"Processed asset {root} is newer than its audit receipt; rerun the deep audit."
        )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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
            examples = sorted({float(value) for value in observed})[:5]
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
    return out


def _to_float_array(value, length: int) -> np.ndarray | None:
    if value is None:
        return None
    if isinstance(value, np.ndarray):
        arr = value.astype(np.float32).reshape(-1)
    elif isinstance(value, list):
        arr = np.asarray(
            [np.nan if item is None else item for item in value], dtype=np.float32
        ).reshape(-1)
    else:
        return None
    out = np.full((length,), np.nan, dtype=np.float32)
    out[: min(length, len(arr))] = arr[:length]
    return out
