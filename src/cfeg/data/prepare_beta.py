from __future__ import annotations

from pathlib import Path

from cfeg.assets.errors import MissingAssetError
from cfeg.data.prepare_mat import prepare_mat_directory


def prepare(raw_dir: Path, out_dir: Path, cfg: dict) -> None:
    if not raw_dir.exists():
        raise MissingAssetError(
            f"BETA raw_dir does not exist: {raw_dir}\n"
            "Prepare it on the GPU/storage server with:\n"
            "  python scripts/fetch_dataset.py --dataset beta --probe-remote\n"
            "  python scripts/fetch_dataset.py --dataset beta\n"
            "or place manual S*.mat files under EEG_DATA_ROOT/raw/beta."
        )
    if not list(raw_dir.rglob("*.mat")) and (raw_dir / "hf_dataset_info.txt").exists():
        raise MissingAssetError(
            f"BETA raw_dir contains a Hugging Face dataset export but no S*.mat EEG files: {raw_dir}\n"
            "The Bingchuan/BETA Hugging Face entry exposes PDFs only in this environment. Fetch the "
            "actual Figshare .mat files instead:\n"
            "  python scripts/fetch_dataset.py --dataset beta --probe-remote\n"
            "  python scripts/fetch_dataset.py --dataset beta --raw-dir \"$EEG_DATA_ROOT/raw/beta\""
        )
    prepare_mat_directory(raw_dir, out_dir, cfg, dataset_id="beta")
