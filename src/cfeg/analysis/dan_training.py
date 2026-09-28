"""Two-stage DAN waveform learning, with no query or human-data IO entry.

Science parameters are passed from the frozen config. A caller must provide
budget callbacks and enforce participant-role boundaries before constructing arrays.
"""

from __future__ import annotations

import hashlib
import math
import time
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import torch

from cfeg.models.dan_alignment import DanAlignment, alignment_loss, weighted_teacher


def state_digest(state: dict) -> str:
    digest = hashlib.sha256()
    for key, value in sorted(state.items()):
        array = value.detach().cpu().contiguous().numpy()
        digest.update(key.encode())
        digest.update(str((array.dtype.str, array.shape)).encode())
        digest.update(array.tobytes())
    return digest.hexdigest()


def _snapshot(model: DanAlignment) -> dict:
    return {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}


@dataclass(frozen=True)
class AlignmentFit:
    states: tuple[dict, ...]  # pretrain + five fine-tuned states
    records: tuple[dict, ...]
    transformed: np.ndarray  # [5*6,class,channel,time], never target-query EEG
    teacher: np.ndarray
    initial_sha256: str
    optimizer_updates: int
    source_validation_outputs: int
    wall_seconds: float


def _fit(model: DanAlignment, train: torch.Tensor, valid: torch.Tensor,
         teacher: torch.Tensor, *, epochs: int, batch_size: int, seed: int,
         learning_rate: float, gradient_clip: float, consume: Callable,
         check: Callable) -> tuple[dict, dict]:
    if epochs < 1 or batch_size < 2 or len(train) < 2 or len(valid) < 2:
        raise ValueError("Positive epochs and nonempty source train/validation required.")
    classes = len(teacher)
    if len(train) % classes or len(valid) % classes:
        raise ValueError("Source arrays must contain complete class-major blocks.")
    if len(train) % batch_size == 1:
        raise ValueError("A singleton final training batch cannot update BatchNorm.")
    train_labels = torch.arange(len(train), device=train.device) % classes
    valid_labels = torch.arange(len(valid), device=valid.device) % classes
    generator = torch.Generator(device="cpu").manual_seed(seed)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=0)
    consume("optimizer_fits", 1)
    initial_sha = state_digest(model.state_dict())
    best_loss, best_epoch, best_state = math.inf, -1, None
    losses, schedule = [], hashlib.sha256()
    updates = 0
    for epoch in range(epochs):
        check()
        model.train()
        order = torch.randperm(len(train), generator=generator)
        schedule.update(order.numpy().tobytes())
        for start in range(0, len(train), batch_size):
            consume("optimizer_updates", 1)
            index = order[start:start + batch_size].to(train.device)
            optimizer.zero_grad(set_to_none=True)
            loss = alignment_loss(model, train[index], train_labels[index], teacher)
            if not torch.isfinite(loss):
                raise ValueError("Nonfinite source training loss.")
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
            if not torch.isfinite(norm):
                raise ValueError("Nonfinite alignment gradient.")
            optimizer.step()
            updates += 1
        model.eval()
        consume("source_validation_outputs", 1)
        with torch.inference_mode():
            value = float(alignment_loss(model, valid, valid_labels, teacher))
        if not math.isfinite(value):
            raise ValueError("Nonfinite source validation loss.")
        losses.append(value)
        if value < best_loss:  # Earlier epoch wins exact ties.
            best_loss, best_epoch, best_state = value, epoch, _snapshot(model)
    if best_state is None:
        raise RuntimeError("No source-validation checkpoint selected.")
    model.load_state_dict(best_state)
    model.eval()
    return best_state, {
        "initial_sha256": initial_sha, "selected_sha256": state_digest(best_state),
        "batch_order_sha256": schedule.hexdigest(), "best_epoch_zero_based": best_epoch,
        "validation_mse": losses, "optimizer_updates": updates,
        "train_rows": len(train), "validation_rows": len(valid),
    }


def align_sources(source: np.ndarray, support: np.ndarray, weights: np.ndarray,
                  training: dict, *, seed: int, device: str,
                  consume: Callable, check: Callable) -> AlignmentFit:
    """5source x6blocks xclass xchannel xtime; paid support k xclass xchannel xtime.

    Source indices0:4 fit pretrain, index4 validates. Each fine stage trains
    blocks0:4 and validates4:6. No array containing target query is accepted.
    """
    start = time.monotonic()
    source = np.ascontiguousarray(source, dtype=np.float32)
    support = np.ascontiguousarray(support, dtype=np.float32)
    if (source.ndim != 5 or source.shape[:2] != (5, 6) or support.ndim != 4
            or source.shape[2:] != support.shape[1:] or len(support) not in (2, 3, 5)
            or not np.isfinite(source).all() or not np.isfinite(support).all()):
        raise ValueError("Invalid source5/blocks6 and paid support geometry.")
    if training.get("weight_decay", 0) != 0 or training.get("optimizer", "Adam") != "Adam":
        raise ValueError("The frozen optimizer is Adam without weight decay.")
    x = torch.from_numpy(source).to(device)
    target = torch.from_numpy(support).to(device)
    teacher = weighted_teacher(target, torch.as_tensor(weights, dtype=torch.float64, device=device))
    classes, channels, samples = source.shape[2:]
    torch.manual_seed(seed)
    model = DanAlignment(channels, samples).to(device)
    initial = state_digest(model.state_dict())
    common = {"batch_size": training["batch_size"], "learning_rate": training["learning_rate"],
              "gradient_clip": training["gradient_clip"], "consume": consume, "check": check}
    pre_state, pre_record = _fit(
        model, x[:4].reshape(-1, channels, samples), x[4].reshape(-1, channels, samples),
        teacher, epochs=training["pretrain_epochs"], seed=seed, **common)
    states, records, transformed = [pre_state], [pre_record], []
    for person in range(5):
        model.load_state_dict(pre_state)
        state, record = _fit(
            model, x[person, :4].reshape(-1, channels, samples),
            x[person, 4:].reshape(-1, channels, samples), teacher,
            epochs=training["fine_epochs"], seed=seed + 1000 + person, **common)
        with torch.inference_mode():
            wave = model.transform_frozen(x[person].reshape(-1, channels, samples))
        transformed.append(wave.cpu().numpy().reshape(6, classes, channels, samples))
        states.append(state)
        records.append(record)
    check()
    return AlignmentFit(
        tuple(states), tuple(records), np.concatenate(transformed), teacher.cpu().numpy(),
        initial, sum(r["optimizer_updates"] for r in records),
        sum(len(r["validation_mse"]) for r in records), time.monotonic() - start)
