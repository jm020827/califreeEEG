#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path

from _bootstrap import add_src_to_path

add_src_to_path()

from cfeg.data.prepare_openbci import prepare
from cfeg.utils.config import load_config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw_session_dir", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--config", default="configs/data/openbci.yaml")
    parser.add_argument("--target_sfreq", type=float, default=None)
    args = parser.parse_args()
    cfg = load_config(args.config, strict_env=False)
    if args.target_sfreq is not None:
        cfg.setdefault("preprocess", {})["target_sfreq"] = args.target_sfreq
    prepare(Path(args.raw_session_dir), Path(args.out_dir), cfg)


if __name__ == "__main__":
    main()

