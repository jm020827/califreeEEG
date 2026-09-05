#!/usr/bin/env python
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

from _bootstrap import add_src_to_path

add_src_to_path()

from cfeg.analysis.synthetic_reliability_stage0 import (
    _reserve_stage0_execution,
    run_stage0,
    write_stage0_outputs,
)
from cfeg.utils.config import load_config

REPO = Path(__file__).resolve().parents[1]
CANONICAL_PROCESSED_DIR = REPO / "data/processed/synthetic_quality_v1"
CANONICAL_PLAN = REPO / "configs/analysis/synthetic_reliability_stage0.yaml"
CANONICAL_OUTPUT_DIR = REPO / "outputs/engineering/reliability-spatial-v1/stage0"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the participant-disjoint synthetic reliability Stage-0 assay."
    )
    parser.add_argument("--processed-dir", default="data/processed/synthetic_quality_v1")
    parser.add_argument(
        "--plan", default="configs/analysis/synthetic_reliability_stage0.yaml"
    )
    parser.add_argument(
        "--output-dir", default="outputs/engineering/reliability-spatial-v1/stage0"
    )
    args = parser.parse_args()

    observed = {
        "processed-dir": Path(args.processed_dir).resolve(),
        "plan": Path(args.plan).resolve(),
        "output-dir": Path(args.output_dir).resolve(),
    }
    expected = {
        "processed-dir": CANONICAL_PROCESSED_DIR.resolve(),
        "plan": CANONICAL_PLAN.resolve(),
        "output-dir": CANONICAL_OUTPUT_DIR.resolve(),
    }
    drifted = [name for name in expected if observed[name] != expected[name]]
    if drifted:
        raise ValueError(
            "Reliability Stage-0 is a one-shot canonical execution; noncanonical "
            f"arguments are forbidden: {drifted}."
        )

    reservation = _reserve_stage0_execution(Path(args.output_dir))
    result = run_stage0(
        args.processed_dir,
        load_config(args.plan, strict_env=False),
        plan_path=args.plan,
    )
    receipt = write_stage0_outputs(
        result,
        args.output_dir,
        _execution_reservation=reservation,
    )
    digest = hashlib.sha256(receipt.read_bytes()).hexdigest()
    print(f"receipt={receipt}")
    print(f"receipt_sha256={digest}")
    print(f"status={result['receipt']['status']}")
    if result["receipt"]["status"] != "passed":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
