from __future__ import annotations

from pathlib import Path

from cfeg.assets.errors import MissingAssetError
from cfeg.data.prepare_mat import prepare_mat_directory


def prepare(raw_dir: Path, out_dir: Path, cfg: dict) -> None:
    if not raw_dir.exists():
        raise MissingAssetError(
            f"Dong2023 raw_dir does not exist: {raw_dir}\n"
            "Fetch the versioned Zenodo mirror with:\n"
            "  python scripts/fetch_dataset.py --dataset dong2023 --probe-remote\n"
            "  python scripts/fetch_dataset.py --dataset dong2023"
        )
    prepare_mat_directory(raw_dir, out_dir, cfg, dataset_id="dong2023")
