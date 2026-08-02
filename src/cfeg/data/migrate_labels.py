from __future__ import annotations

import json
import os
import shutil
import tempfile
import uuid
from datetime import datetime, timezone
from pathlib import Path

import h5py
import numpy as np

from cfeg.data.datasets import _validate_manifest_class_map
from cfeg.data.label_mapping import SSVEP_40_FREQUENCIES, frequency_to_label, write_class_map
from cfeg.data.schema import load_manifest, validate_manifest


_METADATA_FILES = ("manifest.jsonl", "manifest.parquet", "class_map.json")


def migrate_processed_label_alignment(
    processed_dir: str | Path, *, apply: bool = False
) -> dict[str, object]:
    """Remap an existing Wang/BETA artifact by frequency without rewriting EEG signals."""
    root = Path(processed_dir)
    manifest = load_manifest(root)
    validate_manifest(manifest)
    frequencies = manifest["stimulus_frequency_hz"].astype(float).to_numpy()
    new_labels = np.asarray(
        [frequency_to_label(value, SSVEP_40_FREQUENCIES) for value in frequencies],
        dtype=np.int64,
    )
    old_labels = manifest["label"].astype(int).to_numpy(dtype=np.int64)
    signal_path = root / "signals.h5"
    if not signal_path.exists():
        raise FileNotFoundError(f"Missing processed signal file: {signal_path}")
    with h5py.File(signal_path, "r") as h5:
        if "y" not in h5:
            raise ValueError(f"{signal_path} has no y dataset.")
        h5_labels = h5["y"][:].astype(np.int64)
    if len(h5_labels) != len(manifest) or not np.array_equal(h5_labels, old_labels):
        raise ValueError(
            f"Refusing label migration for {root}: manifest labels and signals.h5/y disagree."
        )

    changed = int(np.count_nonzero(old_labels != new_labels))
    result: dict[str, object] = {
        "processed_dir": str(root),
        "n_samples": int(len(manifest)),
        "changed_labels": changed,
        "status": "dry_run" if not apply else "pending",
    }
    if not apply:
        return result

    backup = root / (
        ".label-alignment-backup-"
        + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ-")
        + uuid.uuid4().hex[:8]
    )
    backup.mkdir(parents=False, exist_ok=False)
    existed: dict[str, bool] = {}
    for name in _METADATA_FILES:
        source = root / name
        existed[name] = source.exists()
        if source.exists():
            shutil.copy2(source, backup / name)
    np.save(backup / "y.npy", old_labels)
    (backup / "migration.json").write_text(
        json.dumps({**result, "canonical_frequencies_hz": list(SSVEP_40_FREQUENCIES)}, indent=2),
        encoding="utf-8",
    )

    stage = Path(tempfile.mkdtemp(prefix=".label-alignment-stage-", dir=root))
    new_manifest = manifest.copy()
    new_manifest["label"] = new_labels
    try:
        new_manifest.to_json(stage / "manifest.jsonl", orient="records", lines=True)
        new_manifest.to_parquet(stage / "manifest.parquet", index=False)
        write_class_map(SSVEP_40_FREQUENCIES, stage)
        with h5py.File(signal_path, "r+") as h5:
            h5["y"][:] = new_labels
            h5.flush()
        for name in _METADATA_FILES:
            os.replace(stage / name, root / name)
        _validate_manifest_class_map(root, load_manifest(root))
    except Exception:
        with h5py.File(signal_path, "r+") as h5:
            h5["y"][:] = old_labels
            h5.flush()
        for name in _METADATA_FILES:
            destination = root / name
            saved = backup / name
            if existed[name]:
                shutil.copy2(saved, destination)
            elif destination.exists():
                destination.unlink()
        raise
    finally:
        shutil.rmtree(stage, ignore_errors=True)

    result["status"] = "migrated"
    result["backup_dir"] = str(backup)
    return result
