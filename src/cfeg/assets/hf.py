from __future__ import annotations

from pathlib import Path

from cfeg.assets.errors import MissingAssetError


def hf_cache_hint(repo_id: str, cache_dir: str | None = None, repo_type: str = "model") -> str:
    cache_msg = f" under cache_dir={cache_dir}" if cache_dir else " in the Hugging Face cache"
    return (
        f"Missing Hugging Face {repo_type} asset: {repo_id}{cache_msg}.\n"
        "Run:\n"
        "  huggingface-cli login\n"
        f"  python scripts/fetch_reve.py --model brain-bzh/reve-base --positions brain-bzh/reve-positions"
        + (f" --cache-dir {cache_dir}" if cache_dir else "")
        + "\nor use:\n"
        "  model.backbone.name=tiny_transformer"
    )


def assert_hf_snapshot_present(repo_id: str, cache_dir: str | None = None) -> Path:
    try:
        from huggingface_hub import try_to_load_from_cache
    except Exception as exc:
        raise MissingAssetError(
            "huggingface_hub is required to verify Hugging Face assets. Install requirements.txt."
        ) from exc
    marker = try_to_load_from_cache(repo_id=repo_id, filename="config.json", cache_dir=cache_dir)
    if not marker or not isinstance(marker, str):
        raise MissingAssetError(hf_cache_hint(repo_id, cache_dir))
    return Path(marker).parent

