from __future__ import annotations

import os
from pathlib import Path

from cfeg.assets.errors import MissingAssetError


def resolve_hf_hub_cache(cache_dir: str | Path | None = None) -> Path:
    """Resolve the directory containing models--*/datasets--* cache repositories."""
    if cache_dir:
        return Path(cache_dir).expanduser()
    if os.environ.get("HF_HUB_CACHE"):
        return Path(os.environ["HF_HUB_CACHE"]).expanduser()
    if os.environ.get("HF_HOME"):
        return Path(os.environ["HF_HOME"]).expanduser() / "hub"
    return Path.home() / ".cache" / "huggingface" / "hub"


def hf_cache_hint(
    repo_id: str, cache_dir: str | Path | None = None, repo_type: str = "model"
) -> str:
    resolved = resolve_hf_hub_cache(cache_dir)
    return (
        f"Missing Hugging Face {repo_type} asset: {repo_id} under cache_dir={resolved}.\n"
        "Run:\n"
        "  huggingface-cli login\n"
        "  python scripts/fetch_reve.py --model brain-bzh/reve-base "
        "--positions brain-bzh/reve-positions"
        f" --cache-dir {resolved}"
        "\nor use:\n"
        "  model.backbone.name=tiny_transformer"
    )


def assert_hf_snapshot_present(
    repo_id: str, cache_dir: str | Path | None = None
) -> Path:
    try:
        from huggingface_hub import try_to_load_from_cache
    except Exception as exc:
        raise MissingAssetError(
            "huggingface_hub is required to verify Hugging Face assets. Install requirements.txt."
        ) from exc
    resolved = resolve_hf_hub_cache(cache_dir)
    marker = try_to_load_from_cache(
        repo_id=repo_id, filename="config.json", cache_dir=str(resolved)
    )
    if not marker or not isinstance(marker, str):
        raise MissingAssetError(hf_cache_hint(repo_id, resolved))
    return Path(marker).parent
