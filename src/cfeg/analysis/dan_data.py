"""One allowlisted raw extraction. Query samples are sealed until global freeze.

The MAT container is decoded once per permitted person, so late query samples
necessarily enter extraction memory. They are cached without feature extraction
or scoring; the learning API receives only SupportArrays, never this vault.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from cfeg.analysis.dan_roles import FreezeGate, SupportArrays
from cfeg.analysis.dan_runtime import save_npz, sha_file
from cfeg.analysis.dan_signal import filter_prefix
from cfeg.analysis.dan_teacher import support_quality
from cfeg.analysis.joint_harmonic_data import MANIFEST_COLUMNS, SOURCE_IDS, parse_metadata

RAW_ROOT = Path("/home/whwovy/eeg-data/raw/wearable")
MANIFEST = Path("/home/whwovy/eeg-data/processed/wearable_v3/manifest.parquet")
MANIFEST_SHA = "0355cc8d7d9d9e2a8af59b71f8ffdef800d08c6b05339c08487ff19e3e5de71e"


class QueryVault:
    def __init__(self, ids: tuple[int, ...], prefix: np.ndarray):
        self.ids = ids
        self.__prefix = prefix
        self.__taken = False

    def reveal(self, gate: FreezeGate) -> np.ndarray:
        if self.__taken:
            raise RuntimeError("Query vault already revealed.")
        gate.reveal_once()
        self.__taken = True
        return self.__prefix


def extract_human(config: dict, meter) -> tuple[dict, dict]:
    from cfeg.data.prepare_mat import _load_arrays
    from cfeg.data.prepare_wearable import _wearable_data

    if tuple(config["source_subject_ids"]) != SOURCE_IDS:
        raise ValueError("Immutable source39 allowlist required; held60 forbidden.")
    meter.consume("human_extraction_attempts")
    meter.consume("manifest_hashes")
    if sha_file(MANIFEST) != MANIFEST_SHA:
        raise ValueError("Metadata container SHA changed.")
    meter.consume("manifest_filtered_reads")
    frame = pd.read_parquet(MANIFEST, columns=list(MANIFEST_COLUMNS), filters=[
        ("subject_id", "in", [f"sub{s:03d}" for s in SOURCE_IDS]),
        ("run_id", "in", [f"block{b:02d}" for b in range(1, 6)])])
    meter.consume("manifest_returned_rows", len(frame))
    orders, impedance = parse_metadata(frame, config)
    bands, qs, q2s, query, hashes = [], [], [], [], {}
    for subject in SOURCE_IDS:
        path = RAW_ROOT / f"S{subject:03d}.mat"
        if path.is_symlink() or path.resolve().parent != RAW_ROOT.resolve():
            raise ValueError("Raw path must be a direct allowlisted file.")
        meter.consume("raw_hashes")
        hashes[str(subject)] = sha_file(path)
        meter.consume("raw_loads")
        raw = _wearable_data(_load_arrays(path))
        if tuple(raw.shape) != tuple(config["raw_shape"]):
            raise ValueError("Raw schema differs.")
        end = config["preprocessing"]["available_prefix_end_exclusive"]
        prefix = np.ascontiguousarray(raw[:, :end].transpose(2, 3, 4, 0, 1), dtype=np.float64)
        processed = filter_prefix(prefix[:, :6], config).transpose(1, 0, 2, 3, 4, 5).copy()
        quality = [[support_quality(processed[e, b, :5], frequencies=config["frequencies"],
                                    sfreq=config["sfreq"], n_harmonics=3)
                    for b in range(3)] for e in range(2)]
        bands.append(processed)
        qs.append(np.array([[p[0] for p in row] for row in quality]))
        q2s.append(np.array([[p[1] for p in row] for row in quality]))
        query.append(prefix[:, 6:10].copy())
        meter.event("raw_extracted", subject=subject)
    return {"ids": np.array(SOURCE_IDS), "orders": orders, "impedance": impedance,
            "bands": np.stack(bands), "q": np.stack(qs), "q2": np.stack(q2s),
            "query_prefix": np.stack(query)}, {
                "manifest_sha256": MANIFEST_SHA, "raw_sha256": hashes,
                "metadata_rows": len(frame), "metadata_blocks": [0, 1, 2, 3, 4],
                "query_preprocessed_before_freeze": False}


def save_cache(output: Path, arrays: dict) -> str:
    save_npz(output / "cache.npz", **arrays)
    return sha_file(output / "cache.npz")


def load_cache(output: Path, meter=None) -> tuple[SupportArrays, QueryVault]:
    if meter is not None:
        meter.consume("human_cache_loads")
    with np.load(output / "cache.npz", allow_pickle=False) as packet:
        ids = tuple(int(v) for v in packet["ids"])
        data = SupportArrays(ids, **{key: packet[key] for key in
                                     ("orders", "bands", "q", "q2", "impedance")})
        vault = QueryVault(ids, packet["query_prefix"])
    data.validate()
    return data, vault
