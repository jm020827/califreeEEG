from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch


def save_checkpoint(
    path: str | Path,
    model,
    optimizer=None,
    scheduler=None,
    *,
    config: dict[str, Any],
    epoch: int,
    best_metric: float,
    vocabularies: dict | None = None,
    class_map: dict | None = None,
    asset_info: dict | None = None,
    train_z_mean=None,
    save_trainable_only: bool = False,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state": _checkpoint_state_dict(model, save_trainable_only=save_trainable_only),
            "save_trainable_only": save_trainable_only,
            "optimizer_state": optimizer.state_dict() if optimizer else None,
            "scheduler_state": scheduler.state_dict() if scheduler else None,
            "config": config,
            "epoch": epoch,
            "best_metric": best_metric,
            "vocabularies": vocabularies,
            "class_map": class_map,
            "asset_info": asset_info,
            "train_z_mean": train_z_mean,
        },
        path,
    )


def load_checkpoint(path: str | Path, map_location="cpu") -> dict[str, Any]:
    return torch.load(path, map_location=map_location)


def save_json(path: str | Path, payload: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def _checkpoint_state_dict(model, *, save_trainable_only: bool) -> dict:
    state = model.state_dict()
    if not save_trainable_only:
        return state
    trainable_names = {name for name, p in model.named_parameters() if p.requires_grad}
    return {name: value for name, value in state.items() if name in trainable_names}
