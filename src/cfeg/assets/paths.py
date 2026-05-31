from __future__ import annotations

import os
from pathlib import Path

from cfeg.assets.errors import MissingAssetError


def env_path(name: str, *, required: bool = True) -> Path | None:
    value = os.environ.get(name)
    if value:
        return Path(value).expanduser()
    if required:
        raise MissingAssetError(
            f"Environment variable {name} is not set.\n"
            "Set external asset roots before fetching or preparing public datasets:\n"
            "  cd $HOME/work/jm020827/califreeEEG\n"
            "  source scripts/setup_gpu_pod.sh"
        )
    return None


def ensure_outside_repo(path: Path, repo_root: Path) -> None:
    path = path.resolve()
    repo_root = repo_root.resolve()
    try:
        path.relative_to(repo_root)
    except ValueError:
        return
    allowed = repo_root / "data" / "processed" / "synthetic"
    if not str(path).startswith(str(allowed.resolve())):
        raise MissingAssetError(
            f"Refusing to place large external asset under repository path: {path}\n"
            "Use scripts/setup_gpu_pod.sh so large assets go under .local/ and stay ignored by Git."
        )
