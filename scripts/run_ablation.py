#!/usr/bin/env python
from __future__ import annotations

import argparse

from _bootstrap import add_src_to_path

add_src_to_path()

from cfeg.utils.config import load_config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/train/ablation.yaml")
    parser.add_argument("--only", default=None)
    args = parser.parse_args()
    cfg = load_config(args.config, strict_env=False)
    for name, overrides in cfg.get("variants", {}).items():
        print(f"{name}: ready overrides={overrides}")
    print("Ablation runner scaffold is ready; execute training variants on GPU server.")


if __name__ == "__main__":
    main()

