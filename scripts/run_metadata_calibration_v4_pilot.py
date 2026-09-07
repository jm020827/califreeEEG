#!/usr/bin/env python3
"""One-time CPU entry point for the independent V4 engineering pilot."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

# Numerical libraries must see these settings on their first import, including children.
for name in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS"):
    os.environ[name] = "1"
sys.dont_write_bytecode = True

from _bootstrap import add_src_to_path

add_src_to_path()

from cfeg.analysis.metadata_calibration_v4_pilot import run_pilot


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    print(json.dumps(run_pilot(args.plan, args.output, args.workers), sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
