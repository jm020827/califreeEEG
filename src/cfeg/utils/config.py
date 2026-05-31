from __future__ import annotations

import ast
import copy
import os
import re
from pathlib import Path
from typing import Any

import yaml


_ENV_PATTERN = re.compile(r"\$\{env:([A-Za-z_][A-Za-z0-9_]*)\}")
_PATH_PATTERN = re.compile(r"\$\{paths\.([A-Za-z_][A-Za-z0-9_]*)\}")


def load_config(path: str | Path, *, strict_env: bool = False) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    return resolve_interpolations(cfg, strict_env=strict_env)


def resolve_interpolations(cfg: Any, *, strict_env: bool = False, root: dict[str, Any] | None = None) -> Any:
    root = cfg if root is None and isinstance(cfg, dict) else root
    if isinstance(cfg, dict):
        return {k: resolve_interpolations(v, strict_env=strict_env, root=root) for k, v in cfg.items()}
    if isinstance(cfg, list):
        return [resolve_interpolations(v, strict_env=strict_env, root=root) for v in cfg]
    if not isinstance(cfg, str):
        return cfg

    def replace_env(match: re.Match[str]) -> str:
        name = match.group(1)
        if name in os.environ:
            return os.environ[name]
        if strict_env:
            raise RuntimeError(
                f"Environment variable {name} is not set.\n"
                "Set it before using external assets, for example:\n"
                f"  export {name}=/path/to/value"
            )
        return f"${{env:{name}}}"

    value = _ENV_PATTERN.sub(replace_env, cfg)
    if root and "paths" in root:
        paths = root["paths"]

        def replace_path(match: re.Match[str]) -> str:
            key = match.group(1)
            return str(paths.get(key, match.group(0)))

        value = _PATH_PATTERN.sub(replace_path, value)
        value = _ENV_PATTERN.sub(replace_env, value)
    return value


def merge_overrides(cfg: dict[str, Any], overrides: list[str]) -> dict[str, Any]:
    out = copy.deepcopy(cfg)
    for item in overrides:
        if "=" not in item:
            raise ValueError(f"Override must be key=value, got: {item}")
        key, raw_value = item.split("=", 1)
        value = _parse_value(raw_value)
        cursor = out
        parts = key.split(".")
        for part in parts[:-1]:
            cursor = cursor.setdefault(part, {})
        cursor[parts[-1]] = value
    return out


def _parse_value(raw: str) -> Any:
    if raw.lower() == "true":
        return True
    if raw.lower() == "false":
        return False
    if raw.lower() in {"null", "none"}:
        return None
    try:
        return ast.literal_eval(raw)
    except Exception:
        return raw


def save_config(cfg: dict[str, Any], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)

