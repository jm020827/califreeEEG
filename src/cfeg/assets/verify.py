from __future__ import annotations

import json
from pathlib import Path

import h5py
import numpy as np
from typing import Any

from cfeg.assets.errors import AssetVerificationError, MissingAssetError
from cfeg.assets.hf import assert_hf_snapshot_present
from cfeg.data.datasets import _validate_manifest_class_map
from cfeg.data.schema import REQUIRED_MANIFEST_COLUMNS, load_manifest, validate_manifest


def verify_processed_dir(processed_dir: str | Path) -> dict[str, Any]:
    root = Path(processed_dir)
    required = ["signals.h5", "class_map.json", "preprocess_config.yaml"]
    missing = [name for name in required if not (root / name).exists()]
    if not ((root / "manifest.parquet").exists() or (root / "manifest.jsonl").exists()):
        missing.append("manifest.parquet or manifest.jsonl")
    if missing:
        raise MissingAssetError(
            f"Processed dataset is incomplete at {root}. Missing: {', '.join(missing)}\n"
            "Create it with:\n"
            "  python scripts/prepare_synthetic.py --out_dir data/processed/synthetic\n"
            "or for public data:\n"
            "  python scripts/prepare_dataset.py --dataset <name> --config configs/data/<name>.yaml"
        )
    manifest = load_manifest(root)
    validate_manifest(manifest)
    _validate_manifest_class_map(root, manifest)
    with (root / "class_map.json").open("r", encoding="utf-8") as f:
        class_map = json.load(f)
    with h5py.File(root / "signals.h5", "r") as h5:
        missing_arrays = [name for name in ("x", "channel_mask", "y") if name not in h5]
        if missing_arrays:
            raise AssetVerificationError(
                f"Processed signals.h5 is missing arrays: {', '.join(missing_arrays)}"
            )
        lengths = {name: len(h5[name]) for name in ("x", "channel_mask", "y")}
        if set(lengths.values()) != {len(manifest)}:
            raise AssetVerificationError(
                f"Manifest/HDF5 sample counts disagree: manifest={len(manifest)}, arrays={lengths}"
            )
        if not np.array_equal(h5["y"][:].astype(int), manifest["label"].astype(int).to_numpy()):
            raise AssetVerificationError(
                "HDF5 labels disagree with the manifest. Re-run dataset preparation."
            )
    return {
        "processed_dir": str(root),
        "n_samples": int(len(manifest)),
        "n_classes": int(len(class_map)),
        "required_columns": REQUIRED_MANIFEST_COLUMNS,
    }


def verify_raw_dir(raw_dir: str | Path) -> dict[str, Any]:
    root = Path(raw_dir)
    if not root.exists():
        raise MissingAssetError(
            f"Raw dataset directory does not exist: {root}\n"
            "Fetch explicitly or place manually downloaded files there, then rerun verification."
        )
    files = [p for p in root.rglob("*") if p.is_file()]
    if not files:
        raise AssetVerificationError(f"Raw dataset directory exists but contains no files: {root}")
    return {"raw_dir": str(root), "file_count": len(files)}


def verify_reve_assets(model_id: str, positions_id: str, cache_dir: str | None = None) -> dict[str, Any]:
    positions_path = assert_hf_snapshot_present(positions_id, cache_dir)
    model_path = assert_hf_snapshot_present(model_id, cache_dir)
    return {
        "model_id": model_id,
        "model_path": str(model_path),
        "positions_id": positions_id,
        "positions_path": str(positions_path),
    }

