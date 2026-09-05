from __future__ import annotations

import csv
import hashlib
import json
import os
import re
import shutil
import tarfile
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import pandas as pd
import yaml

from cfeg.data.label_mapping import frequency_to_label, write_class_map
from cfeg.data.preprocess import (
    CanonicalChannelMap,
    PreprocessConfig,
    bandpass_and_resample,
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


@dataclass(frozen=True)
class ChoiPair:
    subject: int
    day: int
    band: str
    session: int
    cnt_path: Path
    mrk_path: Path


@dataclass(frozen=True)
class ChoiMarkers:
    sample_indices_1based: np.ndarray
    within_band_labels: np.ndarray
    source_event_indices: np.ndarray


PREPARED_FILENAMES = (
    "signals.h5",
    "signals.h5.tmp",
    "manifest.jsonl",
    "manifest.jsonl.tmp",
    "manifest.parquet",
    "manifest.parquet.tmp",
    "class_map.json",
    "preprocess_config.yaml",
    "questionnaire_normalized.csv",
    "asset_info.json",
)


def prepare(
    raw_dir: Path,
    out_dir: Path,
    cfg: dict[str, Any],
    *,
    subjects: list[int] | None = None,
) -> dict[str, Any]:
    """Prepare Choi2019 while retaining day/run provenance and published QC flags."""

    raw_dir = raw_dir.resolve()
    out_dir = out_dir.resolve()
    _verify_raw_archive(raw_dir, cfg)
    extracted_root = ensure_extracted(raw_dir, cfg)
    pairs = discover_pairs(extracted_root, cfg, subjects=subjects)
    existing_outputs = [name for name in PREPARED_FILENAMES if (out_dir / name).exists()]
    if existing_outputs:
        raise FileExistsError(
            f"Refusing to overwrite an existing/partial prepared Choi dataset at {out_dir}: "
            f"{existing_outputs}"
        )
    out_dir.mkdir(parents=True, exist_ok=True)

    pcfg = PreprocessConfig.from_dict(cfg.get("preprocess"))
    canonical_map = CanonicalChannelMap.from_yaml()
    channel_names = [str(value) for value in cfg["channel_names"]]
    channel_ids = canonical_map.get_ids(channel_names)
    if any(value == canonical_map.unknown_id for value in channel_ids):
        unknown = [name for name, value in zip(channel_names, channel_ids) if value == 0]
        raise ValueError(f"Choi config contains unknown canonical EEG channels: {unknown}")

    expected_trials_per_pair = int(cfg["expected"]["n_stimulus_events_per_session"])
    total_trials = len(pairs) * expected_trials_per_pair
    samples_per_trial = round(pcfg.window_duration_sec * pcfg.target_sfreq)
    rows: list[dict[str, Any]] = []
    signal_tmp = out_dir / "signals.h5.tmp"

    published_exclusions = set(cfg.get("paper_reported_aggregate_exclusions", []))
    canonical_frequencies = [float(value) for value in cfg["canonical_class_frequencies"]]
    write_index = 0
    with h5py.File(signal_tmp, "x") as destination:
        signal_ds = destination.create_dataset(
            "x",
            shape=(total_trials, pcfg.c_max, samples_per_trial),
            dtype="float32",
            chunks=(1, pcfg.c_max, samples_per_trial),
            compression="gzip",
            compression_opts=4,
        )
        mask_ds = destination.create_dataset(
            "channel_mask",
            shape=(total_trials, pcfg.c_max),
            dtype="bool",
            chunks=(min(256, total_trials), pcfg.c_max),
            compression="gzip",
        )
        label_ds = destination.create_dataset(
            "y",
            shape=(total_trials,),
            dtype="int64",
            chunks=(min(1024, total_trials),),
            compression="gzip",
        )

        for pair in pairs:
            eeg, raw_sfreq, raw_channel_count = read_cnt(pair.cnt_path, cfg)
            markers = read_markers(pair.mrk_path, cfg, n_continuous_samples=eeg.shape[-1])
            filtered, processed_sfreq = bandpass_and_resample(eeg, raw_sfreq, pcfg)
            marker_scale = processed_sfreq / raw_sfreq
            band_frequencies = [
                float(value) for value in cfg["stimulus_by_band"][pair.band]["frequencies_hz"]
            ]
            repetitions: defaultdict[int, int] = defaultdict(int)
            subject_day = f"S{pair.subject}/Day{pair.day}"

            for run_trial_index, (marker, within_label, source_event_index) in enumerate(
                zip(
                    markers.sample_indices_1based,
                    markers.within_band_labels,
                    markers.source_event_indices,
                ),
                start=1,
            ):
                repetitions[int(within_label)] += 1
                marker_zero = int(marker) - 1
                processed_marker_zero = round(marker_zero * marker_scale)
                start = processed_marker_zero + round(pcfg.window_start_sec * processed_sfreq)
                end = start + samples_per_trial
                if start < 0 or end > filtered.shape[-1]:
                    raise ValueError(
                        f"Choi crop outside continuous signal for {pair.cnt_path}: "
                        f"marker={marker}, crop=[{start},{end}), samples={filtered.shape[-1]}"
                    )
                native = filtered[:, start:end].astype(np.float32, copy=True)
                native_std = np.std(native, axis=-1).astype(np.float32)
                finite_std = native_std[np.isfinite(native_std)]
                query_signal_std = float(np.median(finite_std)) if finite_std.size else None
                aligned_std = place_channel_values(
                    native_std, channel_names, canonical_map, pcfg.c_max
                )
                if pcfg.normalize == "per_trial_channel_zscore":
                    native = normalize_trial(native)
                elif pcfg.normalize not in {"none", None}:
                    raise ValueError(f"Unknown normalization mode: {pcfg.normalize}")
                placed, channel_mask, slot_ids = place_on_canonical_channels(
                    native, channel_names, canonical_map, pcfg.c_max
                )

                frequency = band_frequencies[int(within_label)]
                label = frequency_to_label(frequency, canonical_frequencies)
                signal_ds[write_index] = placed
                mask_ds[write_index] = channel_mask
                label_ds[write_index] = label
                rows.append(
                    {
                        "sample_id": (
                            f"choi2019_sub{pair.subject:03d}_day{pair.day:02d}_"
                            f"{pair.band.lower()}_session{pair.session:02d}_"
                            f"trial{run_trial_index:03d}"
                        ),
                        "h5_index": write_index,
                        "dataset_id": "choi2019",
                        "subject_id": f"sub{pair.subject:03d}",
                        "session_id": f"day{pair.day:02d}",
                        "run_id": f"{pair.band.lower()}_session{pair.session:02d}",
                        "trial_id": f"trial{run_trial_index:03d}",
                        "label": label,
                        "stimulus_frequency_hz": frequency,
                        "stimulus_phase_rad": None,
                        "sfreq_original": raw_sfreq,
                        "sfreq_processed": processed_sfreq,
                        "window_start_sec": pcfg.window_start_sec,
                        "window_duration_sec": pcfg.window_duration_sec,
                        "reference": cfg["reference"],
                        "hardware_id": cfg["hardware_id"],
                        "cap_type": cfg["cap_type"],
                        "electrode_type": cfg["electrode_type"],
                        "n_channels_original": len(channel_names),
                        "n_channels_used": int(channel_mask.sum()),
                        "channel_names_original": channel_names,
                        "channel_names_used": channel_names,
                        "canonical_channel_ids": slot_ids,
                        "impedance_mean_kohm": None,
                        "impedance_max_kohm": None,
                        "reattach_flag": None,
                        "time_since_last_session_hours": None,
                        "environment_note_code": cfg["environment_note_code"],
                        "source_file": str(pair.cnt_path.relative_to(extracted_root)),
                        "query_signal_std": query_signal_std,
                        "query_signal_std_by_channel": nullable_vector(aligned_std),
                        "day_index": pair.day,
                        "frequency_band": pair.band.lower(),
                        "within_band_label": int(within_label),
                        "within_run_class_repetition": repetitions[int(within_label)],
                        "source_marker_sample_index_1based": int(marker),
                        "source_marker_event_index": int(source_event_index),
                        "source_channels_total": raw_channel_count,
                        "auxiliary_channels_available_but_excluded": True,
                        "paper_aggregate_exclusion_subject_day": subject_day
                        in published_exclusions,
                        "questionnaire_day_linkage": "unresolved_not_joined",
                    }
                )
                write_index += 1

    if write_index != total_trials:
        raise RuntimeError(f"Prepared {write_index} trials, expected {total_trials}")
    os.replace(signal_tmp, out_dir / "signals.h5")
    manifest = pd.DataFrame(rows, columns=ordered_manifest_columns(rows))
    validate_manifest(manifest)
    write_manifest(manifest, out_dir)
    write_class_map(canonical_frequencies, out_dir)
    _write_yaml_once(out_dir / "preprocess_config.yaml", pcfg.__dict__)
    questionnaire_receipt = normalize_questionnaire(
        raw_dir / "questionnaires_answers.csv", out_dir / "questionnaire_normalized.csv"
    )
    info = {
        "schema": "cfeg.choi2019-processed-asset.v1",
        "dataset_id": "choi2019",
        "dataset_revision": cfg["dataset_revision"],
        "created_by": "scripts/prepare_choi2019.py",
        "raw_archive_sha256": _archive_spec(cfg)["sha256"],
        "subjects": sorted({f"sub{pair.subject:03d}" for pair in pairs}),
        "n_pairs": len(pairs),
        "n_trials": total_trials,
        "n_paper_flagged_trials_retained": int(
            manifest["paper_aggregate_exclusion_subject_day"].sum()
        ),
        "class_alignment": "stimulus_frequency_hz_across_three_band_specific_4_class_runs",
        "marker_time_unit": cfg["marker_time_unit"],
        "query_qc_extractor_version": "continuous_filtered_cropped_pre_zscore_channel_std_median_v1",
        "external_continuous_schema": "impedance_unavailable",
        "questionnaire": questionnaire_receipt,
        "metadata_claim_boundary": cfg["metadata_claim_boundary"],
    }
    _write_json_once(out_dir / "asset_info.json", info)
    return info


def _archive_spec(cfg: dict[str, Any]) -> dict[str, Any]:
    name = str(cfg["archive"]["name"])
    matches = [entry for entry in cfg["files"] if entry["name"] == name]
    if len(matches) != 1:
        raise ValueError(f"Expected one frozen archive spec for {name}, found {len(matches)}")
    return matches[0]


def _verify_raw_archive(raw_dir: Path, cfg: dict[str, Any]) -> None:
    spec = _archive_spec(cfg)
    archive = raw_dir / str(spec["name"])
    if not archive.exists():
        raise FileNotFoundError(
            f"Missing frozen Choi archive: {archive}. Run scripts/fetch_choi2019.py first."
        )
    observed = _hash_file(archive)
    expected = {
        "bytes": int(spec["object_size_bytes"]),
        "md5": str(spec["md5"]),
        "sha256": str(spec["sha256"]),
    }
    if observed != expected:
        raise ValueError(f"Frozen Choi archive mismatch: expected={expected}, observed={observed}")


def _hash_file(path: Path) -> dict[str, Any]:
    md5 = hashlib.md5()
    sha256 = hashlib.sha256()
    size = 0
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            size += len(chunk)
            md5.update(chunk)
            sha256.update(chunk)
    return {"bytes": size, "md5": md5.hexdigest(), "sha256": sha256.hexdigest()}


def _expected_archive_keys(cfg: dict[str, Any]) -> set[tuple[int, int, str, str, int]]:
    expected = cfg["expected"]
    return {
        (subject, day, kind, band, session)
        for subject in range(1, int(expected["n_subjects"]) + 1)
        for day in range(1, int(expected["n_days"]) + 1)
        for kind in ("cnt", "mrk")
        for band in cfg["stimulus_by_band"]
        for session in range(1, int(expected["n_sessions_per_band_day"]) + 1)
    }


def audit_archive(archive_path: Path, cfg: dict[str, Any]) -> dict[str, int]:
    pattern = re.compile(str(cfg["archive"]["allowed_member_regex"]))
    directory_pattern = re.compile(r"^S([1-9]|[12][0-9]|30)(?:/Day([12]))?/?$")
    observed: list[tuple[int, int, str, str, int]] = []
    directory_count = 0
    unpacked_bytes = 0
    with tarfile.open(archive_path, "r:gz") as archive:
        for member in archive:
            if member.isdir():
                if not directory_pattern.fullmatch(member.name):
                    raise ValueError(f"Unexpected Choi archive directory: {member.name!r}")
                directory_count += 1
                continue
            if not member.isfile():
                raise ValueError(f"Unsafe/non-regular Choi archive member: {member.name!r}")
            match = pattern.fullmatch(member.name)
            if not match:
                raise ValueError(f"Unexpected Choi archive file: {member.name!r}")
            observed.append(
                (
                    int(match.group(1)),
                    int(match.group(2)),
                    match.group(3),
                    match.group(4),
                    int(match.group(5)),
                )
            )
            unpacked_bytes += int(member.size)
    duplicates = [key for key, count in Counter(observed).items() if count != 1]
    expected_keys = _expected_archive_keys(cfg)
    if set(observed) != expected_keys or duplicates:
        raise ValueError(
            "Choi archive inventory mismatch: "
            f"missing={len(expected_keys - set(observed))}, "
            f"extra={len(set(observed) - expected_keys)}, duplicates={len(duplicates)}"
        )
    expected_archive = cfg["archive"]
    receipt = {
        "regular_files": len(observed),
        "directories": directory_count,
        "uncompressed_regular_file_bytes": unpacked_bytes,
    }
    for field, observed_value in receipt.items():
        if observed_value != int(expected_archive[field]):
            raise ValueError(
                f"Choi archive {field} drift: expected={expected_archive[field]}, "
                f"observed={observed_value}"
            )
    return receipt


def ensure_extracted(raw_dir: Path, cfg: dict[str, Any]) -> Path:
    archive_path = raw_dir / str(cfg["archive"]["name"])
    audit_archive(archive_path, cfg)
    destination = raw_dir / str(cfg["archive"]["extraction_subdir"])
    present = list(destination.rglob("*.mat")) if destination.exists() else []
    if present:
        _validate_extracted_inventory(destination, cfg)
        return destination
    destination.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive_path, "r:gz") as archive:
        for member in archive:
            target = (destination / member.name).resolve()
            if destination not in target.parents and target != destination:
                raise ValueError(f"Archive path traversal attempt: {member.name!r}")
            if member.isdir():
                target.mkdir(parents=True, exist_ok=True)
                continue
            if not member.isfile():
                raise ValueError(f"Unsafe/non-regular Choi archive member: {member.name!r}")
            target.parent.mkdir(parents=True, exist_ok=True)
            source = archive.extractfile(member)
            if source is None:
                raise ValueError(f"Could not read archive member: {member.name!r}")
            with source, target.open("xb") as output:
                shutil.copyfileobj(source, output, length=8 * 1024 * 1024)
    _validate_extracted_inventory(destination, cfg)
    return destination


def _validate_extracted_inventory(root: Path, cfg: dict[str, Any]) -> None:
    pattern = re.compile(str(cfg["archive"]["allowed_member_regex"]))
    observed = []
    for path in root.rglob("*.mat"):
        relative = path.relative_to(root).as_posix()
        match = pattern.fullmatch(relative)
        if not match:
            raise ValueError(f"Unexpected extracted Choi MAT path: {relative!r}")
        observed.append(
            (
                int(match.group(1)),
                int(match.group(2)),
                match.group(3),
                match.group(4),
                int(match.group(5)),
            )
        )
    expected = _expected_archive_keys(cfg)
    if set(observed) != expected or len(observed) != len(expected):
        raise ValueError(
            "Extracted Choi inventory is incomplete or duplicated; preserved files were not changed."
        )


def discover_pairs(
    extracted_root: Path,
    cfg: dict[str, Any],
    *,
    subjects: list[int] | None = None,
) -> list[ChoiPair]:
    _validate_extracted_inventory(extracted_root, cfg)
    selected = set(subjects or range(1, int(cfg["expected"]["n_subjects"]) + 1))
    valid_subjects = set(range(1, int(cfg["expected"]["n_subjects"]) + 1))
    if not selected or not selected <= valid_subjects:
        raise ValueError(f"Invalid Choi subject selection: {sorted(selected)}")
    pairs = []
    for subject in sorted(selected):
        for day in range(1, int(cfg["expected"]["n_days"]) + 1):
            for band in cfg["stimulus_by_band"]:
                for session in range(1, int(cfg["expected"]["n_sessions_per_band_day"]) + 1):
                    stem = f"{band}({session}).mat"
                    pairs.append(
                        ChoiPair(
                            subject=subject,
                            day=day,
                            band=band,
                            session=session,
                            cnt_path=extracted_root / f"S{subject}" / f"Day{day}" / f"cnt_{stem}",
                            mrk_path=extracted_root / f"S{subject}" / f"Day{day}" / f"mrk_{stem}",
                        )
                    )
    missing = [
        path for pair in pairs for path in (pair.cnt_path, pair.mrk_path) if not path.is_file()
    ]
    if missing:
        raise FileNotFoundError(f"Missing Choi pair member(s): {missing[:3]}")
    return pairs


def audit_source_schema(
    extracted_root: Path,
    cfg: dict[str, Any],
    *,
    subjects: list[int] | None = None,
) -> dict[str, Any]:
    pairs = discover_pairs(extracted_root, cfg, subjects=subjects)
    continuous_lengths = []
    trials = 0
    for pair in pairs:
        eeg, sfreq, raw_channel_count = read_cnt(pair.cnt_path, cfg)
        markers = read_markers(
            pair.mrk_path,
            cfg,
            n_continuous_samples=eeg.shape[-1],
        )
        if sfreq != float(cfg["raw_sfreq"]):
            raise ValueError(f"Unexpected Choi sfreq after parsing {pair.cnt_path}: {sfreq}")
        if raw_channel_count != int(cfg["expected"]["n_channels_distributed"]):
            raise ValueError(f"Unexpected Choi raw channel count in {pair.cnt_path}")
        continuous_lengths.append(int(eeg.shape[-1]))
        trials += len(markers.sample_indices_1based)
    expected_trials = len(pairs) * int(cfg["expected"]["n_stimulus_events_per_session"])
    if trials != expected_trials:
        raise ValueError(f"Choi source audit found {trials} trials, expected {expected_trials}")
    return {
        "schema": "cfeg.choi2019-source-schema-audit.v1",
        "subjects": len({pair.subject for pair in pairs}),
        "cnt_mrk_pairs": len(pairs),
        "stimulus_trials": trials,
        "distributed_sfreq_hz": float(cfg["raw_sfreq"]),
        "eeg_channels": len(cfg["channel_names"]),
        "auxiliary_channels": len(cfg["auxiliary_channel_names"]),
        "continuous_samples_min": min(continuous_lengths),
        "continuous_samples_max": max(continuous_lengths),
    }


def _decode_matlab_char(handle: h5py.File, reference: h5py.Reference) -> str:
    values = np.asarray(handle[reference]).reshape(-1)
    return "".join(chr(int(value)) for value in values if int(value))


def _read_cell_strings(handle: h5py.File, path: str) -> list[str]:
    return [
        _decode_matlab_char(handle, reference) for reference in np.asarray(handle[path]).reshape(-1)
    ]


def read_cnt(path: Path, cfg: dict[str, Any]) -> tuple[np.ndarray, float, int]:
    expected_channels = [str(value) for value in cfg["raw_channel_names"]]
    eeg_names = [str(value) for value in cfg["channel_names"]]
    with h5py.File(path, "r") as handle:
        required = {"cnt/x", "cnt/clab", "cnt/fs", "cnt/T"}
        if not all(key in handle for key in required):
            raise ValueError(f"Missing Choi cnt field(s) in {path}")
        source_names = _read_cell_strings(handle, "cnt/clab")
        if source_names != expected_channels:
            raise ValueError(f"Choi channel-order drift in {path}: {source_names}")
        source = handle["cnt/x"]
        if source.ndim != 2 or source.shape[0] != len(source_names):
            raise ValueError(f"Unexpected Choi cnt/x schema in {path}: {source.shape}")
        sfreq = float(np.asarray(handle["cnt/fs"]).reshape(-1)[0])
        if sfreq != float(cfg["raw_sfreq"]):
            raise ValueError(f"Choi distributed sampling-rate drift in {path}: {sfreq}")
        duration_ms = float(np.asarray(handle["cnt/T"]).reshape(-1)[0])
        expected_duration_ms = source.shape[1] / sfreq * 1000.0
        if duration_ms != expected_duration_ms:
            raise ValueError(
                f"Choi cnt duration mismatch in {path}: T={duration_ms}, "
                f"samples/fs={expected_duration_ms}"
            )
        indices = [source_names.index(name) for name in eeg_names]
        eeg = source[indices, :].astype(np.float32)
    if not np.isfinite(eeg).all():
        raise ValueError(f"Non-finite Choi EEG values in {path}")
    return eeg, sfreq, len(source_names)


def read_markers(
    path: Path,
    cfg: dict[str, Any],
    *,
    n_continuous_samples: int,
) -> ChoiMarkers:
    with h5py.File(path, "r") as handle:
        required = {"mrk/y", "mrk/time", "mrk/className"}
        if not all(key in handle for key in required):
            raise ValueError(f"Missing Choi marker field(s) in {path}")
        class_names = _read_cell_strings(handle, "mrk/className")
        if class_names != [str(value) for value in cfg["marker_class_names"]]:
            raise ValueError(f"Choi marker-class drift in {path}: {class_names}")
        y = np.asarray(handle["mrk/y"])
        times = np.asarray(handle["mrk/time"]).reshape(-1)
    expected_events = int(cfg["expected"]["n_marker_events_per_session"])
    expected_classes = int(cfg["expected"]["n_marker_classes"])
    if y.shape != (expected_events, expected_classes) or times.shape != (expected_events,):
        raise ValueError(
            f"Unexpected Choi marker schema in {path}: y={y.shape}, time={times.shape}"
        )
    if not np.all(np.isin(y, [0.0, 1.0])) or not np.all(y.sum(axis=1) == 1.0):
        raise ValueError(f"Choi marker labels are not strict one-hot values in {path}")
    if not np.all(np.isfinite(times)) or not np.all(times == np.floor(times)):
        raise ValueError(f"Choi marker times are not finite integer sample indices in {path}")
    if not np.all(np.diff(times) > 0):
        raise ValueError(f"Choi marker times are not strictly increasing in {path}")
    classes = np.argmax(y, axis=1)
    fixation_class = len(class_names) - 1
    stimulus_mask = classes != fixation_class
    if not np.all(stimulus_mask[0::2]) or np.any(stimulus_mask[1::2]):
        raise ValueError(f"Choi stimulus/fixation marker alternation drift in {path}")
    within_labels = classes[stimulus_mask]
    expected_per_class = int(cfg["expected"]["n_trials_per_class_session"])
    if not np.all(np.bincount(within_labels, minlength=fixation_class) == expected_per_class):
        raise ValueError(f"Choi within-session class counts drift in {path}")
    stimulus_times = times[stimulus_mask].astype(np.int64)
    max_needed = (
        stimulus_times.max()
        - 1
        + round(
            (
                float(cfg["preprocess"]["window_start_sec"])
                + float(cfg["preprocess"]["window_duration_sec"])
            )
            * float(cfg["raw_sfreq"])
        )
    )
    if stimulus_times.min() < 1 or max_needed > n_continuous_samples:
        raise ValueError(f"Choi marker/crop bounds exceed cnt samples in {path}")
    return ChoiMarkers(
        sample_indices_1based=stimulus_times,
        within_band_labels=within_labels.astype(np.int64),
        source_event_indices=np.flatnonzero(stimulus_mask).astype(np.int64),
    )


def normalize_questionnaire(source: Path, destination: Path) -> dict[str, Any]:
    rows = []
    with source.open("r", encoding="utf-8-sig", newline="") as handle:
        for raw in csv.reader(handle):
            if len(raw) < 8:
                continue
            match = re.fullmatch(r"S(\d+)_(Before|After)", raw[1].strip())
            if not match:
                continue
            rows.append(
                {
                    "subject_id": f"sub{int(match.group(1)):03d}",
                    "timing": match.group(2).lower(),
                    "gender_as_published": raw[2].strip(),
                    "sleeping_hour_as_published": _optional_int(raw[3]),
                    "body_condition_as_published": _optional_int(raw[4]),
                    "drowsiness_as_published": _optional_int(raw[5]),
                    "concentration_as_published": _optional_int(raw[6]),
                    "eye_strain_as_published": _optional_int(raw[7]),
                    "day_linkage": "unresolved",
                    "eligible_as_pre_query_model_input": False,
                }
            )
    frame = pd.DataFrame(rows)
    if len(frame) != 60:
        raise ValueError(f"Expected 60 Choi questionnaire rows, observed {len(frame)}")
    counts = frame.groupby(["subject_id", "timing"]).size()
    expected_keys = {
        (f"sub{subject:03d}", timing) for subject in range(1, 31) for timing in ("before", "after")
    }
    if set(counts.index) != expected_keys or not (counts == 1).all():
        raise ValueError("Choi questionnaire subject/timing inventory drift")
    pivot = frame.pivot(index="subject_id", columns="timing", values="gender_as_published")
    mismatches = sorted(pivot.index[pivot["before"] != pivot["after"]].tolist())
    before_counts = Counter(frame.loc[frame["timing"] == "before", "gender_as_published"])
    if before_counts != Counter({"M": 21, "W": 9}):
        raise ValueError(f"Choi before-row gender aggregate drift: {dict(before_counts)}")
    if mismatches != ["sub007"]:
        raise ValueError(f"Unexpected Choi within-subject gender inconsistencies: {mismatches}")
    if destination.exists():
        raise FileExistsError(f"Refusing to overwrite questionnaire table: {destination}")
    frame.to_csv(destination, index=False)
    return {
        "rows": len(frame),
        "day_linkage": "unresolved",
        "joined_to_trial_manifest": False,
        "before_row_gender_counts": dict(sorted(before_counts.items())),
        "published_gender_mismatch_subjects": mismatches,
    }


def _optional_int(raw: str) -> int | None:
    value = raw.strip()
    return int(value) if value else None


def _write_json_once(path: Path, payload: Any) -> None:
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite prepared provenance: {path}")
    with path.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")


def _write_yaml_once(path: Path, payload: Any) -> None:
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite prepared config: {path}")
    with path.open("x", encoding="utf-8") as handle:
        yaml.safe_dump(payload, handle, sort_keys=False)
