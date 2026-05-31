#!/usr/bin/env python
from __future__ import annotations

import argparse
import os

from _bootstrap import add_src_to_path

add_src_to_path()

from cfeg.assets.registry import AssetRegistry


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, choices=["beta", "wang", "wearable", "openbci"])
    parser.add_argument("--assets-config", default="configs/assets.yaml")
    parser.add_argument("--method", default=None)
    parser.add_argument("--dry-run", action="store_true", help="Print planned action only.")
    parser.add_argument("--probe-remote", action="store_true", help="Probe public source metadata without downloading data.")
    args = parser.parse_args()
    registry = AssetRegistry.from_yaml(args.assets_config, strict_env=False)
    cfg = registry.dataset(args.dataset)
    if args.dataset == "openbci":
        print("openbci is local private data. Use scripts/openbci_convert.py on a local session folder.")
        return
    if args.dataset == "wearable":
        print(
            "Wearable SSVEP is treated as manual/Figshare source. Download on storage server from:\n"
            "  https://figshare.com/articles/dataset/An_Open_Dataset_for_Wearable_SSVEP-Based_Brain-Computer_Interfaces/13560281\n"
            "Then place files under EEG_DATA_ROOT/raw/wearable and run verify/prepare."
        )
        return
    if args.probe_remote and args.dataset == "beta":
        try:
            from huggingface_hub import HfApi
        except Exception as exc:
            raise SystemExit(
                "huggingface_hub is not installed. Install requirements first:\n"
                "  python -m pip install -r requirements.txt"
            ) from exc

        repo_id = cfg.get("repo_id", "Bingchuan/beta")
        info = HfApi(token=os.environ.get("HF_TOKEN")).dataset_info(repo_id)
        print(f"remote ok: dataset {repo_id} sha={info.sha}")
        return
    if args.dataset == "beta":
        if args.dry_run:
            print("Would run datasets.load_dataset('Bingchuan/beta') on the GPU/storage server.")
            return
        from datasets import load_dataset

        ds = load_dataset(cfg.get("repo_id", "Bingchuan/beta"))
        print(ds)
        return
    if args.dataset == "wang":
        if args.dry_run:
            print("Would use MOABB Wang2016 loader if installed, otherwise manual raw_dir.")
            return
        try:
            from moabb.datasets import Wang2016
        except Exception:
            raise SystemExit(
                "MOABB is not installed. Install with:\n"
                "  pip install -e '.[moabb]'\n"
                "or manually place Wang2016 raw files under $EEG_DATA_ROOT/raw/wang."
            )
        dataset = Wang2016()
        print(dataset)


if __name__ == "__main__":
    main()
