#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path

from _bootstrap import add_src_to_path

add_src_to_path()

from cfeg.assets.errors import AssetVerificationError, MissingAssetError
from cfeg.assets.registry import AssetRegistry
from cfeg.assets.verify import verify_processed_dir, verify_raw_dir, verify_reve_assets


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--assets-config", default="configs/assets.yaml")
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--dataset")
    parser.add_argument("--stage", choices=["raw", "processed"], default="processed")
    parser.add_argument("--model")
    parser.add_argument("--raw-dir", default=None)
    parser.add_argument("--processed-dir", default=None)
    args = parser.parse_args()
    registry = AssetRegistry.from_yaml(args.assets_config, strict_env=False)
    checks = []
    if args.all:
        checks.extend(("dataset", name) for name in registry.cfg.get("datasets", {}))
        checks.extend(("model", name) for name in registry.cfg.get("models", {}) if name != "tiny_eeg_transformer")
    elif args.dataset:
        checks.append(("dataset", args.dataset))
    elif args.model:
        checks.append(("model", args.model))
    else:
        parser.error("Use --all, --dataset, or --model")

    for kind, name in checks:
        try:
            if kind == "dataset":
                ds = registry.dataset(name)
                override = args.raw_dir if args.stage == "raw" else args.processed_dir
                path = override or ds.get(f"{args.stage}_dir") or ds.get("processed_dir")
                if not path:
                    raise MissingAssetError(f"No {args.stage}_dir configured for dataset {name}")
                result = verify_raw_dir(path) if args.stage == "raw" else verify_processed_dir(Path(path))
            else:
                model = registry.model(name)
                if model.get("type") == "local_code":
                    result = {"model": name, "status": "local_code"}
                else:
                    result = verify_reve_assets(
                        model["repo_id"], model["positions_repo_id"], model.get("cache_dir")
                    )
            print(json.dumps({"ok": True, "kind": kind, "name": name, "result": result}, indent=2))
        except (MissingAssetError, AssetVerificationError, FileNotFoundError, ValueError) as exc:
            print(json.dumps({"ok": False, "kind": kind, "name": name, "error": str(exc)}, indent=2))
            if not args.all:
                raise SystemExit(1)


if __name__ == "__main__":
    main()
