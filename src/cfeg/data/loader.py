from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class LoaderSettings:
    train_batch_size: int
    eval_batch_size: int
    num_workers: int
    persistent_workers: bool
    pin_memory: bool
    persistent_hdf5_handles: bool
    preload_hdf5_to_memory: bool
    preload_train_indices_to_memory: bool

    def contract(self) -> dict[str, int | bool]:
        return asdict(self)

    def kwargs(self) -> dict[str, int | bool]:
        return {
            "num_workers": self.num_workers,
            "persistent_workers": self.persistent_workers,
            "pin_memory": self.pin_memory,
        }


def resolve_loader_settings(data_cfg: dict, *, device_type: str) -> LoaderSettings:
    train_batch = int(data_cfg.get("batch_size", 16))
    eval_batch = int(data_cfg.get("eval_batch_size", train_batch))
    workers = int(data_cfg.get("num_workers", 0))
    if train_batch < 1 or eval_batch < 1:
        raise ValueError("data.batch_size and data.eval_batch_size must be positive.")
    if workers < 0:
        raise ValueError("data.num_workers must be non-negative.")
    return LoaderSettings(
        train_batch_size=train_batch,
        eval_batch_size=eval_batch,
        num_workers=workers,
        persistent_workers=bool(workers > 0 and data_cfg.get("persistent_workers", False)),
        pin_memory=bool(data_cfg.get("pin_memory", device_type == "cuda")),
        persistent_hdf5_handles=bool(data_cfg.get("persistent_hdf5_handles", False)),
        preload_hdf5_to_memory=bool(data_cfg.get("preload_hdf5_to_memory", False)),
        preload_train_indices_to_memory=bool(
            data_cfg.get("preload_train_indices_to_memory", False)
        ),
    )
