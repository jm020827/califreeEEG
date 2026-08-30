from __future__ import annotations

import hashlib
import json
import os
import random
import uuid
from pathlib import Path
from typing import Any

import numpy as np
import torch


def save_checkpoint(
    path: str | Path,
    model,
    optimizer=None,
    scheduler=None,
    *,
    config: dict[str, Any],
    epoch: int,
    best_metric: float | None,
    vocabularies: dict | None = None,
    class_map: dict | None = None,
    asset_info: dict | None = None,
    train_z_mean=None,
    save_trainable_only: bool = False,
    checkpoint_role: str | None = None,
    selection_split: str | None = None,
    selection_metric: str | None = None,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    _atomic_torch_save(
        path,
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
            "checkpoint_role": checkpoint_role,
            "selection_split": selection_split,
            "selection_metric": selection_metric,
        },
    )


def load_checkpoint(path: str | Path, map_location="cpu") -> dict[str, Any]:
    return torch.load(path, map_location=map_location, weights_only=False)


def load_checkpoint_model_state(model, checkpoint: dict[str, Any]) -> None:
    state = checkpoint.get("model_state")
    if not isinstance(state, dict):
        raise TypeError("Checkpoint has no model_state mapping.")
    if not bool(checkpoint.get("save_trainable_only", False)):
        model.load_state_dict(state, strict=True)
        return
    materialize = getattr(model, "materialize_checkpoint_modules", None)
    if callable(materialize):
        materialize(state)
    trainable = {name for name, parameter in model.named_parameters() if parameter.requires_grad}
    stored = set(state)
    missing_trainable = sorted(trainable - stored)
    unexpected = sorted(stored - set(model.state_dict()))
    nontrainable_stored = sorted(stored - trainable)
    if missing_trainable or unexpected or nontrainable_stored:
        raise ValueError(
            "Trainable-only checkpoint state contract mismatch: "
            f"missing_trainable={missing_trainable[:5]}, "
            f"unexpected={unexpected[:5]}, nontrainable_stored={nontrainable_stored[:5]}."
        )
    incompatible = model.load_state_dict(state, strict=False)
    expected_missing = set(model.state_dict()) - trainable
    if set(incompatible.missing_keys) != expected_missing or incompatible.unexpected_keys:
        raise ValueError(
            "Trainable-only checkpoint load produced unexpected missing/state keys: "
            f"missing={incompatible.missing_keys[:5]}, "
            f"unexpected={incompatible.unexpected_keys[:5]}."
        )


def save_json(path: str | Path, payload: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)
    _fsync_directory(path.parent)


def capture_rng_state(train_generator: torch.Generator) -> dict[str, Any]:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda_all": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        "train_loader_generator": train_generator.get_state(),
    }


def restore_rng_state(state: dict[str, Any], train_generator: torch.Generator) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"])
    if state.get("torch_cuda_all") is not None:
        if not torch.cuda.is_available():
            raise RuntimeError("Resume state contains CUDA RNG but CUDA is unavailable.")
        torch.cuda.set_rng_state_all(state["torch_cuda_all"])
    train_generator.set_state(state["train_loader_generator"])


def save_resume_state(output_dir: str | Path, payload: dict[str, Any]) -> dict[str, Any]:
    root = Path(output_dir) / ".resume"
    root.mkdir(parents=True, exist_ok=True)
    commit_path = root / "commit.json"
    previous = None
    if commit_path.exists():
        previous = json.loads(commit_path.read_text(encoding="utf-8"))
    generation = int((previous or {}).get("generation", -1)) + 1
    slot = generation % 2
    state = dict(payload)
    state["schema"] = "cfeg.exact-resume.v1"
    state["generation"] = generation
    state_path = root / f"state.{slot}.pt"
    _atomic_torch_save(state_path, state)
    commit = {
        "schema": "cfeg.exact-resume-commit.v1",
        "generation": generation,
        "slot": slot,
        "state_path": state_path.name,
        "state_sha256": _sha256_file(state_path),
    }
    save_json(commit_path, commit)
    return commit


def load_resume_state(
    output_dir: str | Path, *, map_location: torch.device | str = "cpu"
) -> tuple[dict[str, Any], dict[str, Any]]:
    root = Path(output_dir) / ".resume"
    commit_path = root / "commit.json"
    if not commit_path.is_file():
        raise FileNotFoundError(f"Exact-resume commit is missing: {commit_path}.")
    commit = json.loads(commit_path.read_text(encoding="utf-8"))
    if commit.get("schema") != "cfeg.exact-resume-commit.v1":
        raise ValueError("Unknown exact-resume commit schema.")
    state_path = root / str(commit.get("state_path", ""))
    if not state_path.is_file() or _sha256_file(state_path) != commit.get("state_sha256"):
        raise ValueError("Committed exact-resume state is missing or corrupted.")
    # The resume journal is locally produced and contains Python/NumPy RNG state,
    # which is intentionally outside PyTorch's weights-only allowlist.
    state = torch.load(state_path, map_location=map_location, weights_only=False)
    if (
        state.get("schema") != "cfeg.exact-resume.v1"
        or int(state.get("generation", -1)) != int(commit.get("generation", -2))
    ):
        raise ValueError("Exact-resume state and commit generations disagree.")
    return state, commit


def _checkpoint_state_dict(model, *, save_trainable_only: bool) -> dict:
    state = model.state_dict()
    if not save_trainable_only:
        return state
    trainable_names = {name for name, p in model.named_parameters() if p.requires_grad}
    return {name: value for name, value in state.items() if name in trainable_names}


def _atomic_torch_save(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("wb") as handle:
            torch.save(payload, handle)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        _fsync_directory(path.parent)
    finally:
        if temporary.exists():
            temporary.unlink()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _fsync_directory(path: Path) -> None:
    try:
        descriptor = os.open(path, os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
