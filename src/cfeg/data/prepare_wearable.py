from __future__ import annotations

from pathlib import Path

from cfeg.assets.errors import MissingAssetError
from cfeg.data.prepare_mat import prepare_mat_directory


def prepare(raw_dir: Path, out_dir: Path, cfg: dict) -> None:
    if not raw_dir.exists():
        raise MissingAssetError(
            f"Wearable SSVEP raw_dir does not exist: {raw_dir}\n"
            "Download from Figshare/manual source on the GPU/storage server and place files under "
            "EEG_DATA_ROOT/raw/wearable, preserving impedance and wet/dry metadata when available."
        )
    prepare_mat_directory(raw_dir, out_dir, cfg, dataset_id="wearable")
