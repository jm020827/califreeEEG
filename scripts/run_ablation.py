#!/usr/bin/env python
from __future__ import annotations

import argparse
import os
import traceback
from pathlib import Path

import pandas as pd

from _bootstrap import add_src_to_path

add_src_to_path()

from cfeg.train_loop import run_training
from cfeg.utils.config import load_config, merge_overrides


def main() -> None:
    parser = argparse.ArgumentParser(description="Execute A0-A5 ablation training variants.")
    parser.add_argument("--config", default="configs/train/ablation.yaml")
    parser.add_argument("--only", default=None, help="Comma-separated variant names.")
    parser.add_argument("--include-optional", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--continue-on-error", action="store_true")
    args = parser.parse_args()

    suite = load_config(args.config, strict_env=False)
    base_config = os.environ.get("CFEG_ABLATION_BASE_CONFIG", suite["base_config"])
    base = load_config(base_config, strict_env=False)
    _apply_runtime_environment(base)
    selected = set(args.only.split(",")) if args.only else None
    optional = set(suite.get("optional_variants", []))
    rows = []
    failed = False
    for name, overrides in suite.get("variants", {}).items():
        if selected is not None and name not in selected:
            continue
        if name in optional and not args.include_optional:
            rows.append({"variant": name, "status": "skipped_optional"})
            continue
        cfg = merge_overrides(base, [f"{key}={value!r}" for key, value in overrides.items()])
        cfg["run_name"] = name
        cfg["output_dir"] = str(Path("outputs/ablation") / name)
        wandb_cfg = cfg.setdefault("tracking", {}).setdefault("wandb", {})
        tags = list(wandb_cfg.get("tags") or [])
        wandb_cfg["tags"] = sorted(set(tags + ["ablation", name]))
        try:
            result = run_training(cfg, dry_run=args.dry_run)
            rows.append(
                {
                    "variant": name,
                    "status": "dry_run" if args.dry_run else "completed",
                    "best_accuracy": result.get("best_accuracy"),
                    "test_accuracy": (result.get("test") or {}).get("accuracy"),
                    "output_dir": result.get("output_dir", cfg["output_dir"]),
                }
            )
        except Exception as exc:
            failed = True
            rows.append({"variant": name, "status": "failed", "error": str(exc)})
            traceback.print_exc()
            if not args.continue_on_error:
                break

    out = Path("outputs/ablation/summary.csv")
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out, index=False)
    print(out)
    if failed:
        raise SystemExit(1)


def _apply_runtime_environment(cfg: dict) -> None:
    backbone = os.environ.get("CFEG_BACKBONE")
    if backbone:
        backbone_cfg = cfg.setdefault("model", {}).setdefault("backbone", {})
        backbone_cfg["name"] = backbone
        if backbone == "reve":
            backbone_cfg.update(
                {
                    "hf_model": "brain-bzh/reve-base",
                    "hf_positions": "brain-bzh/reve-positions",
                    "cache_dir": os.environ.get("HF_HUB_CACHE"),
                    "local_files_only": True,
                    "freeze": True,
                }
            )

    mode = os.environ.get("WANDB_MODE")
    if not mode:
        mode = "online" if os.environ.get("WANDB_API_KEY") else "disabled"
    wandb_cfg = cfg.setdefault("tracking", {}).setdefault("wandb", {})
    wandb_cfg["enabled"] = mode != "disabled"
    wandb_cfg["mode"] = mode
    wandb_cfg["project"] = os.environ.get(
        "WANDB_PROJECT", wandb_cfg.get("project", "calibration-free-eeg")
    )
    if os.environ.get("WANDB_ENTITY"):
        wandb_cfg["entity"] = os.environ["WANDB_ENTITY"]


if __name__ == "__main__":
    main()
