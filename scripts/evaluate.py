#!/usr/bin/env python
from __future__ import annotations

import argparse

from _bootstrap import add_src_to_path

add_src_to_path()

from cfeg.eval_loop import run_channel_stress_eval
from cfeg.utils.config import load_config, merge_overrides


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("overrides", nargs="*")
    args = parser.parse_args()
    cfg = merge_overrides(load_config(args.config, strict_env=False), args.overrides)
    print(run_channel_stress_eval(cfg, args.ckpt))


if __name__ == "__main__":
    main()
