from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np


def write_processed_hdf5(
    out_dir: str | Path,
    x: np.ndarray,
    channel_mask: np.ndarray,
    y: np.ndarray,
) -> Path:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "signals.h5"
    with h5py.File(path, "w") as h5:
        h5.create_dataset("x", data=x.astype("float32"), compression="gzip", compression_opts=4)
        h5.create_dataset("channel_mask", data=channel_mask.astype("bool"), compression="gzip")
        h5.create_dataset("y", data=y.astype("int64"), compression="gzip")
    return path


def read_sample(h5_path: str | Path, index: int) -> tuple[np.ndarray, np.ndarray, int]:
    with h5py.File(h5_path, "r") as h5:
        x = h5["x"][index].astype("float32")
        mask = h5["channel_mask"][index].astype("bool")
        y = int(h5["y"][index])
    return x, mask, y

