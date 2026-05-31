from __future__ import annotations

from pathlib import Path
from typing import Any

from cfeg.utils.config import load_config


class AssetRegistry:
    def __init__(self, cfg: dict[str, Any]):
        self.cfg = cfg

    @classmethod
    def from_yaml(cls, path: str | Path = "configs/assets.yaml", *, strict_env: bool = False) -> "AssetRegistry":
        return cls(load_config(path, strict_env=strict_env))

    def dataset(self, name: str) -> dict[str, Any]:
        try:
            return self.cfg["datasets"][name]
        except KeyError as exc:
            raise KeyError(f"Unknown dataset '{name}'. Known: {sorted(self.cfg.get('datasets', {}))}") from exc

    def model(self, name: str) -> dict[str, Any]:
        try:
            return self.cfg["models"][name]
        except KeyError as exc:
            raise KeyError(f"Unknown model '{name}'. Known: {sorted(self.cfg.get('models', {}))}") from exc

