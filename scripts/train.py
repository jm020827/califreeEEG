#!/usr/bin/env python
from __future__ import annotations

import argparse

from _bootstrap import add_src_to_path

add_src_to_path()

from cfeg.train_loop import run_training
from cfeg.utils.config import load_config, merge_overrides


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--dry-run", action="store_true", help="Build data/model and run one forward pass only.")
    parser.add_argument("overrides", nargs="*")
    args = parser.parse_args()
    cfg = merge_overrides(load_config(args.config, strict_env=False), args.overrides)
    print(run_training(cfg, dry_run=args.dry_run))


if __name__ == "__main__":
    main()
