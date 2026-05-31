#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path

from _bootstrap import add_src_to_path

add_src_to_path()

from cfeg.data.prepare_beta import prepare as prepare_beta
from cfeg.data.prepare_openbci import prepare as prepare_openbci
from cfeg.data.prepare_wang import prepare as prepare_wang
from cfeg.data.prepare_wearable import prepare as prepare_wearable
from cfeg.utils.config import load_config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, choices=["beta", "wang", "wearable", "openbci"])
    parser.add_argument("--raw_dir", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    cfg = load_config(args.config, strict_env=False)
    funcs = {
        "beta": prepare_beta,
        "wang": prepare_wang,
        "wearable": prepare_wearable,
        "openbci": prepare_openbci,
    }
    funcs[args.dataset](Path(args.raw_dir), Path(args.out_dir), cfg)


if __name__ == "__main__":
    main()

