from __future__ import annotations

from pathlib import Path

from cfeg.assets.errors import MissingAssetError
from cfeg.data.prepare_mat import prepare_mat_directory


def prepare(raw_dir: Path, out_dir: Path, cfg: dict) -> None:
    if not raw_dir.exists():
        raise MissingAssetError(
            f"Wang2016 raw_dir does not exist: {raw_dir}\n"
            "Install MOABB and fetch on the GPU/storage server:\n"
            "  pip install -e '.[moabb]'\n"
            "  python scripts/fetch_dataset.py --dataset wang --method moabb\n"
            "or manually place Wang2016 files under EEG_DATA_ROOT/raw/wang."
        )
    prepare_mat_directory(raw_dir, out_dir, cfg, dataset_id="wang")
