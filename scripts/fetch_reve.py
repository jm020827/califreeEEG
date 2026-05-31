#!/usr/bin/env python
from __future__ import annotations

import argparse
import os

from _bootstrap import add_src_to_path

add_src_to_path()

from cfeg.assets.hf import assert_hf_snapshot_present


def fetch_reve(model_id: str, positions_id: str, cache_dir: str | None, dry_run: bool = False, probe_remote: bool = False) -> None:
    if probe_remote:
        try:
            from huggingface_hub import HfApi
        except Exception as exc:
            raise SystemExit(
                "huggingface_hub is not installed. Install requirements first:\n"
                "  python -m pip install -r requirements.txt"
            ) from exc

        api = HfApi(token=os.environ.get("HF_TOKEN"))
        for repo_id in [positions_id, model_id]:
            info = api.model_info(repo_id)
            print(f"remote ok: {repo_id} sha={info.sha}")
    if dry_run:
        for repo_id in [positions_id, model_id]:
            try:
                path = assert_hf_snapshot_present(repo_id, cache_dir)
                print(f"cache ok: {repo_id} -> {path}")
            except Exception as exc:
                print(f"cache missing: {repo_id}: {exc}")
        return
    from huggingface_hub import snapshot_download

    for repo_id in [positions_id, model_id]:
        path = snapshot_download(repo_id=repo_id, cache_dir=cache_dir, repo_type="model")
        print(f"downloaded: {repo_id} -> {path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="brain-bzh/reve-base")
    parser.add_argument("--positions", default="brain-bzh/reve-positions")
    parser.add_argument("--cache-dir", default=os.environ.get("HF_HOME"))
    parser.add_argument("--dry-run", action="store_true", help="Only check local cache; do not download.")
    parser.add_argument("--probe-remote", action="store_true", help="Check that HF repos are reachable without downloading snapshots.")
    args = parser.parse_args()
    try:
        fetch_reve(args.model, args.positions, args.cache_dir, args.dry_run, args.probe_remote)
    except Exception as exc:
        token_hint = "" if os.environ.get("HF_TOKEN") else "\nIf access is gated, run: huggingface-cli login"
        raise SystemExit(f"{exc}{token_hint}")


if __name__ == "__main__":
    main()
