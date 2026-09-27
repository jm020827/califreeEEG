"""Role-limited data preparation for the frozen joint-harmonic experiment."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from cfeg.analysis.joint_harmonic import (
    impedance_summary,
    make_context,
    sham_donors,
    spectral_features,
)

SOURCE_IDS = (4, 6, 8, 11, 14, 21, 22, 25, 28, 29, 30, 31, 32, 33, 37, 41, 42, 43,
              44, 46, 54, 55, 56, 61, 63, 65, 67, 73, 74, 77, 80, 82, 83, 84, 89, 92,
              97, 100, 102)
MANIFEST_COLUMNS = ("subject_id", "electrode_type", "run_id", "impedance_kohm_by_channel",
                    "headband_order", "condition_period")
ARRAY_KEYS = ("ids", "orders", "spectra", "q", "q2", "impedance", "query_crop")


@dataclass(frozen=True)
class RoleData:
    ids: tuple[int, ...]
    orders: np.ndarray
    spectra: np.ndarray
    q: np.ndarray
    q2: np.ndarray
    impedance: np.ndarray
    query_crop: np.ndarray


def select_role(cache: dict[str, np.ndarray], ids: Sequence[int]) -> RoleData:
    """Copy only the named participants, not a view retaining an entire cohort."""
    wanted = tuple(ids)
    present = cache["ids"].tolist()
    if not wanted or len(set(wanted)) != len(wanted) or len(set(present)) != len(present):
        raise ValueError("Role/cache participant IDs must be unique and nonempty.")
    if not set(wanted) <= set(present):
        raise ValueError("Role contains an unavailable participant.")
    indices = [present.index(s) for s in wanted]
    arrays = {key: np.array(cache[key][indices], copy=True) for key in ARRAY_KEYS if key != "ids"}
    for array in arrays.values():
        array.setflags(write=False)
    return RoleData(wanted, **arrays)


def prefix_contexts(role: RoleData, arm: str, *, sham_seed: int) -> tuple[np.ndarray, dict]:
    """No Q/Q2 from late query blocks or M from outside the paid prefix."""
    orders = {s: ("dry", "wet")[int(role.orders[i])] for i, s in enumerate(role.ids)}
    donors = (sham_donors(role.ids, headband_orders=orders, seed=sham_seed)
              if arm == "SHAM" else {s: s for s in role.ids})
    contexts = []
    for i, subject in enumerate(role.ids):
        interfaces = []
        for interface in range(2):
            prefixes = []
            for k in (1, 3, 5):
                q = role.q[i, interface, :k].mean(axis=(0, 1))
                kwargs = {}
                if arm == "Q2":
                    kwargs["q2"] = role.q2[i, interface, :k].mean(axis=(0, 1))
                elif arm in {"QM", "SHAM"}:
                    donor_index = role.ids.index(donors[subject])
                    values, flags = impedance_summary(role.impedance[donor_index, interface, :k])
                    kwargs.update(metadata=values, available=flags)
                common = np.array([interface, role.orders[i],
                                   int(interface != role.orders[i]), np.log(k)])
                prefixes.append(make_context(q, common=common, arm=arm, **kwargs))
            interfaces.append(prefixes)
        contexts.append(interfaces)
    return np.asarray(contexts), donors


def parse_metadata(frame: pd.DataFrame, config: dict) -> tuple[np.ndarray, np.ndarray]:
    ids = config["source_subject_ids"]
    expected_people = {f"sub{s:03d}" for s in ids}
    classes = len(config["frequencies"])
    if set(frame.columns) != set(MANIFEST_COLUMNS) or set(frame["subject_id"]) != expected_people:
        raise ValueError("Metadata projection has forbidden columns/participants.")
    if len(frame) != len(ids) * 2 * 5 * classes:
        raise ValueError("Metadata projection has incorrect repeated row count.")
    slots = np.asarray(config["canonical_channel_ids"], int) - 1
    impedance = np.empty((len(ids), 2, 5, len(slots)), float)
    orders = np.empty(len(ids), int)
    for i, subject in enumerate(ids):
        person = frame[frame.subject_id == f"sub{subject:03d}"]
        if person.headband_order.nunique(dropna=False) != 1:
            raise ValueError("Inconsistent headband order within participant.")
        order = person.headband_order.iloc[0]
        if order not in ("dry", "wet"):
            raise ValueError("Unknown headband order.")
        orders[i] = ("dry", "wet").index(order)
        for e, interface in enumerate(("dry", "wet")):
            for block in range(5):
                rows = person[(person.electrode_type == interface)
                              & (person.run_id == f"block{block + 1:02d}")]
                if len(rows) != classes:
                    raise ValueError("Missing/duplicated pre-query block metadata packet.")
                expected_period = "first" if order == interface else "second"
                if set(rows.condition_period) != {expected_period}:
                    raise ValueError("Condition period does not match common headband order.")
                vectors = [np.asarray(v, float) for v in rows.impedance_kohm_by_channel]
                if any(v.shape != (64,) for v in vectors):
                    raise ValueError("Canonical impedance vector must have 64 slots.")
                if any(not np.array_equal(v, vectors[0], equal_nan=True) for v in vectors[1:]):
                    raise ValueError("Repeated class rows disagree on acquisition metadata.")
                # Check all decoded canonical entries, then select the declared electrodes.
                impedance_summary(vectors[0][None])
                impedance[i, e, block] = vectors[0][slots]
    return orders, impedance


def extract_participant(raw: np.ndarray, config: dict) -> dict[str, np.ndarray]:
    if tuple(raw.shape) != tuple(config["raw_shape"]):
        raise ValueError("Raw EEG shape differs from the frozen schema.")
    begin, n = config["start_sample"], config["n_samples"]
    if begin < 0 or begin + n > raw.shape[1]:
        raise ValueError("Requested crop not available.")
    # [interface,block,class,channel,time], no class-dependent operation.
    crop = np.asarray(raw[:, begin:begin + n].transpose(2, 3, 4, 0, 1), float)
    result = spectral_features(crop, frequencies=config["frequencies"], sfreq=config["sfreq"],
                               n_harmonics=config["n_harmonics"])
    result["query_crop"] = crop[:, 5:10]
    return {k: v.astype(np.float32) for k, v in result.items()}


def allowed_raw_path(subject: int, config: dict) -> Path:
    root = Path("/home/whwovy/eeg-data/raw/wearable")
    if tuple(config["source_subject_ids"]) != SOURCE_IDS or Path(config["raw_root"]) != root:
        raise ValueError("Immutable source39/raw-root boundary differs.")
    if subject not in SOURCE_IDS:
        raise ValueError("Participant is outside exposed source39; held access forbidden.")
    path = root / f"S{subject:03d}.mat"
    if path.resolve().parent != root.resolve() or path.is_symlink():
        raise ValueError("Raw path must be a direct allowlisted file, not a symlink.")
    return path


def read_human_cache(config: dict, meter, hash_file: Callable) -> tuple[dict, dict]:
    """Only called after the runner's durable one-shot reservation and qualification."""
    from cfeg.data.prepare_mat import _load_arrays
    from cfeg.data.prepare_wearable import _wearable_data

    if tuple(config["source_subject_ids"]) != SOURCE_IDS:
        raise ValueError("Human read cannot expand beyond source39.")
    manifest = Path("/home/whwovy/eeg-data/processed/wearable_v3/manifest.parquet")
    if Path(config["manifest_path"]) != manifest:
        raise ValueError("Unexpected metadata container.")
    meter.consume("human_extraction_attempts")
    meter.consume("manifest_hashes")
    manifest_sha = hash_file(manifest)
    if manifest_sha != config["manifest_sha256"]:
        raise ValueError("Public metadata container hash changed.")
    meter.consume("manifest_filtered_reads")
    frame = pd.read_parquet(
        manifest, columns=list(MANIFEST_COLUMNS),
        filters=[("subject_id", "in", [f"sub{s:03d}" for s in SOURCE_IDS]),
                 ("run_id", "in", [f"block{b:02d}" for b in range(1, 6)])],
    )
    meter.consume("manifest_returned_rows", len(frame))
    orders, impedance = parse_metadata(frame, config)
    parts, raw_hashes = [], {}
    for subject in SOURCE_IDS:
        path = allowed_raw_path(subject, config)
        meter.consume("raw_hashes")
        raw_hashes[str(subject)] = hash_file(path)
        meter.consume("raw_loads")
        parts.append(extract_participant(_wearable_data(_load_arrays(path)), config))
        meter.check()
        meter.event("extracted", subject_id=subject)
    cache = {key: np.stack([part[key] for part in parts]) for key in parts[0]}
    cache.update(ids=np.asarray(SOURCE_IDS), orders=orders, impedance=impedance)
    return cache, {"manifest_sha256": manifest_sha, "raw_file_sha256": raw_hashes,
                   "returned_metadata_rows": len(frame), "returned_subjects": list(SOURCE_IDS),
                   "returned_metadata_blocks": [0, 1, 2, 3, 4],
                   "columns": list(MANIFEST_COLUMNS), "query_scores_computed": False}
