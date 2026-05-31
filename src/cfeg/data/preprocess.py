from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from scipy import signal


@dataclass
class PreprocessConfig:
    target_sfreq: float = 200.0
    window_start_sec: float = 0.14
    window_duration_sec: float = 2.0
    bandpass_low_hz: float | None = 6.0
    bandpass_high_hz: float | None = 90.0
    notch_hz: float | None = None
    normalize: str = "per_trial_channel_zscore"
    c_max: int = 64

    @classmethod
    def from_dict(cls, cfg: dict[str, Any] | None) -> "PreprocessConfig":
        return cls(**(cfg or {}))


class CanonicalChannelMap:
    def __init__(self, unknown_id: int, name_to_id: dict[str, int], id_to_name: dict[int, str]):
        self.unknown_id = unknown_id
        self.name_to_id = name_to_id
        self.id_to_name = id_to_name

    @classmethod
    def from_yaml(cls, path: str | Path = "configs/canonical_channels.yaml") -> "CanonicalChannelMap":
        with Path(path).open("r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        name_to_id: dict[str, int] = {}
        id_to_name: dict[int, str] = {}
        for row in cfg["channels"]:
            cid = int(row["id"])
            name = str(row["name"])
            id_to_name[cid] = name
            for alias in [name, *row.get("aliases", [])]:
                name_to_id[canonicalize_channel_name(alias)] = cid
        return cls(int(cfg.get("unknown_id", 0)), name_to_id, id_to_name)

    def get_id(self, name: str) -> int:
        return self.name_to_id.get(canonicalize_channel_name(name), self.unknown_id)

    def get_ids(self, names: list[str]) -> list[int]:
        return [self.get_id(name) for name in names]


def canonicalize_channel_name(name: str) -> str:
    return str(name).strip().replace(" ", "").upper()


def bandpass_and_resample(
    x: np.ndarray, sfreq: float, cfg: PreprocessConfig
) -> tuple[np.ndarray, float]:
    y = np.asarray(x, dtype=np.float32)
    if cfg.notch_hz:
        b, a = signal.iirnotch(w0=cfg.notch_hz, Q=30.0, fs=sfreq)
        y = signal.filtfilt(b, a, y, axis=-1).astype(np.float32)
    if cfg.bandpass_low_hz is not None and cfg.bandpass_high_hz is not None:
        high = min(float(cfg.bandpass_high_hz), sfreq / 2.0 - 1e-3)
        low = float(cfg.bandpass_low_hz)
        if 0 < low < high:
            sos = signal.butter(4, [low, high], btype="bandpass", fs=sfreq, output="sos")
            y = signal.sosfiltfilt(sos, y, axis=-1).astype(np.float32)
    if abs(sfreq - cfg.target_sfreq) > 1e-6:
        gcd = np.gcd(int(round(sfreq)), int(round(cfg.target_sfreq)))
        up = int(round(cfg.target_sfreq)) // gcd
        down = int(round(sfreq)) // gcd
        y = signal.resample_poly(y, up=up, down=down, axis=-1).astype(np.float32)
        sfreq = float(cfg.target_sfreq)
    return y, float(sfreq)


def crop_window(x: np.ndarray, sfreq: float, t_start: float, duration: float) -> np.ndarray:
    start = int(round(t_start * sfreq))
    n = int(round(duration * sfreq))
    end = start + n
    if start < 0 or end > x.shape[-1]:
        raise ValueError(
            f"Crop [{t_start}, {t_start + duration}] sec is outside signal with "
            f"{x.shape[-1]} samples at {sfreq} Hz"
        )
    return x[..., start:end].astype(np.float32)


def normalize_trial(x: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    mean = x.mean(axis=-1, keepdims=True)
    std = x.std(axis=-1, keepdims=True)
    return ((x - mean) / (std + eps)).astype(np.float32)


def place_on_canonical_channels(
    x: np.ndarray,
    ch_names: list[str],
    canonical_map: CanonicalChannelMap,
    c_max: int,
) -> tuple[np.ndarray, np.ndarray, list[int]]:
    out = np.zeros((c_max, x.shape[-1]), dtype=np.float32)
    mask = np.zeros((c_max,), dtype=bool)
    ids: list[int] = []
    unknown_cursor = 0
    for row, name in zip(x, ch_names):
        cid = canonical_map.get_id(name)
        ids.append(cid)
        if cid > 0 and cid <= c_max:
            slot = cid - 1
        else:
            while unknown_cursor < c_max and mask[unknown_cursor]:
                unknown_cursor += 1
            if unknown_cursor >= c_max:
                continue
            slot = unknown_cursor
        out[slot] = row
        mask[slot] = True
    return out, mask, ids


def preprocess_trial(
    x: np.ndarray,
    ch_names: list[str],
    sfreq: float,
    cfg: PreprocessConfig,
    canonical_map: CanonicalChannelMap,
) -> tuple[np.ndarray, np.ndarray, list[int], float]:
    y, new_sfreq = bandpass_and_resample(x, sfreq, cfg)
    y = crop_window(y, new_sfreq, cfg.window_start_sec, cfg.window_duration_sec)
    if cfg.normalize == "per_trial_channel_zscore":
        y = normalize_trial(y)
    elif cfg.normalize not in {"none", None}:
        raise ValueError(f"Unknown normalization mode: {cfg.normalize}")
    placed, mask, ids = place_on_canonical_channels(y, ch_names, canonical_map, cfg.c_max)
    return placed, mask, ids, new_sfreq

