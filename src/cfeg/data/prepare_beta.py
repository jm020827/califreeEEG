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
            "then download/cache the dataset or place manual raw files under EEG_DATA_ROOT/raw/beta."
        )
    prepare_mat_directory(raw_dir, out_dir, cfg, dataset_id="beta")
