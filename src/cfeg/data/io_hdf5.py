from __future__ import annotations

import os
from pathlib import Path
from typing import Any

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


def read_unlabeled_signal(
    h5_path: str | Path, index: int
) -> tuple[np.ndarray, np.ndarray]:
    """Read only signal and channel mask; never open the label dataset."""

    with h5py.File(h5_path, "r") as h5:
        x = h5["x"][index].astype("float32")
        mask = h5["channel_mask"][index].astype("bool")
    return x, mask


class HDF5SampleReader:
    """Lazily reuse read-only HDF5 handles within one process or DataLoader worker."""

    def __init__(self, *, persistent: bool = False):
        self.persistent = bool(persistent)
        self._handles: dict[Path, h5py.File] = {}
        self._pid = os.getpid()

    def read(self, h5_path: str | Path, index: int) -> tuple[np.ndarray, np.ndarray, int]:
        if not self.persistent:
            return read_sample(h5_path, index)
        if self._pid != os.getpid():
            # A forked worker must not use handles opened by its parent.
            self.close()
            self._pid = os.getpid()
        path = Path(h5_path).resolve()
        handle = self._handles.get(path)
        if handle is None or not handle.id.valid:
            handle = h5py.File(path, "r")
            self._handles[path] = handle
        x = handle["x"][index].astype("float32")
        mask = handle["channel_mask"][index].astype("bool")
        y = int(handle["y"][index])
        return x, mask, y

    def read_unlabeled(
        self, h5_path: str | Path, index: int
    ) -> tuple[np.ndarray, np.ndarray]:
        """Read query signal without evaluating or indexing the HDF5 `y` object."""

        if not self.persistent:
            return read_unlabeled_signal(h5_path, index)
        if self._pid != os.getpid():
            self.close()
            self._pid = os.getpid()
        path = Path(h5_path).resolve()
        handle = self._handles.get(path)
        if handle is None or not handle.id.valid:
            handle = h5py.File(path, "r")
            self._handles[path] = handle
        x = handle["x"][index].astype("float32")
        mask = handle["channel_mask"][index].astype("bool")
        return x, mask

    def close(self) -> None:
        for handle in self._handles.values():
            if handle.id.valid:
                handle.close()
        self._handles.clear()

    def __getstate__(self) -> dict[str, Any]:
        # h5py handles must never cross a multiprocessing spawn boundary.
        return {"persistent": self.persistent, "_handles": {}, "_pid": None}

    def __setstate__(self, state: dict[str, Any]) -> None:
        self.persistent = bool(state.get("persistent", False))
        self._handles = {}
        self._pid = os.getpid()

    def __del__(self) -> None:  # pragma: no cover - interpreter shutdown is nondeterministic
        try:
            self.close()
        except (AttributeError, OSError, RuntimeError, ValueError):
            pass
