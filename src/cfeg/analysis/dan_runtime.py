"""Durable artifacts and finite-budget accounting for the single DAN programme."""

from __future__ import annotations

import hashlib
import json
import os
import resource
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path: Path, value: dict, *, replace: bool = False) -> None:
    target = path.with_suffix(path.suffix + ".next") if replace else path
    with target.open("xb") as stream:
        stream.write((json.dumps(value, ensure_ascii=False, allow_nan=False, indent=2) + "\n").encode())
        stream.flush()
        os.fsync(stream.fileno())
    if replace:
        os.replace(target, path)


def save_npz(path: Path, **arrays) -> None:
    with path.open("xb") as stream:
        np.savez(stream, **arrays)
        stream.flush()
        os.fsync(stream.fileno())


def save_torch(path: Path, value: dict) -> None:
    with path.open("xb") as stream:
        torch.save(value, stream)
        stream.flush()
        os.fsync(stream.fileno())


class Meter:
    def __init__(self, output: Path, limits: dict, *, generated: bool):
        self.output, self.limits, self.generated = output, limits, generated
        self.start = time.monotonic()
        self.last_check = self.start - 10
        self.counts: dict[str, int] = {}
        self.training_seconds = 0.0
        self.active_training_since: float | None = None
        self.sequence = 0

    def consume(self, key: str, n: int = 1) -> None:
        value = self.counts.get(key, 0) + n
        if n < 0 or key not in self.limits or value > self.limits[key]:
            raise RuntimeError(f"DAN budget exceeded: {key}={value}")
        self.counts[key] = value

    def snapshot(self) -> dict:
        active = 0 if self.active_training_since is None else time.monotonic() - self.active_training_since
        return {"at": now(), "generated": self.generated, "counts": self.counts.copy(),
                "wall_seconds": time.monotonic() - self.start,
                "training_seconds": self.training_seconds + active,
                "peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
                "peak_torch_allocated_bytes": torch.cuda.max_memory_allocated()
                if torch.cuda.is_initialized() else 0}

    def check(self, *, force: bool = False) -> None:
        if not force and time.monotonic() - self.last_check < 5:
            return
        self.last_check = time.monotonic()
        snapshot = self.snapshot()
        checks = {"human_wall_seconds": snapshot["wall_seconds"],
                  "gpu_training_seconds": snapshot["training_seconds"],
                  "ram_bytes": snapshot["peak_rss_bytes"],
                  "gpu_allocated_bytes": snapshot["peak_torch_allocated_bytes"],
                  "new_output_bytes": sum(p.stat().st_size for p in self.output.rglob("*")
                                          if p.is_file())}
        for key, value in checks.items():
            if value > self.limits[key]:
                raise RuntimeError(f"DAN resource ceiling exceeded: {key}={value}")
        write_json(self.output / "meter.json", snapshot, replace=True)

    def event(self, name: str, **fields) -> None:
        self.sequence += 1
        row = {"event": name, "sequence": self.sequence, "at": now(), **fields}
        with (self.output / "journal.jsonl").open("a", encoding="utf8") as stream:
            stream.write(json.dumps(row, allow_nan=False) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        if not self.generated:
            print(json.dumps(row), flush=True)
        self.check(force=True)
