from __future__ import annotations

import atexit
import os
from contextlib import suppress
from pathlib import Path

import h5py
import numpy as np

_READ_HANDLES: dict[tuple[int, str], h5py.File] = {}


def _close_read_handles() -> None:
    for handle in _READ_HANDLES.values():
        with suppress(Exception):
            handle.close()
    _READ_HANDLES.clear()


atexit.register(_close_read_handles)


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
    resolved = str(Path(h5_path).resolve())
    key = (os.getpid(), resolved)
    h5 = _READ_HANDLES.get(key)
    if h5 is None or not h5.id.valid:
        h5 = h5py.File(resolved, "r")
        _READ_HANDLES[key] = h5
    x = h5["x"][index].astype("float32")
    mask = h5["channel_mask"][index].astype("bool")
    y = int(h5["y"][index])
    return x, mask, y
