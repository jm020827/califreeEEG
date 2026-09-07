#!/usr/bin/env python3
"""Consume the single frozen metadata-free AQ development/evaluation study."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

for name in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS"):
    os.environ[name] = "1"
sys.dont_write_bytecode = True

from _bootstrap import add_src_to_path

add_src_to_path()

from cfeg.analysis.metadata_calibration_v4_aq_dev import run_study


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    print(json.dumps(run_study(args.plan, args.output, args.workers), sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
