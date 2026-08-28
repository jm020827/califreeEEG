#!/usr/bin/env python
from __future__ import annotations

import argparse

from _bootstrap import add_src_to_path

add_src_to_path()

from cfeg.prediction import run_prediction


def main() -> None:
    parser = argparse.ArgumentParser(description="Run calibration-free inference on processed EEG.")
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--processed-dir", required=True)
    parser.add_argument("--out", default="outputs/predictions.csv")
    parser.add_argument(
        "--split",
        choices=["test", "all"],
        default="test",
        help=(
            "Prediction selection. Default 'test' requires checkpoint-adjacent split.csv; "
            "use 'all' only for explicit whole-manifest inference."
        ),
    )
    parser.add_argument(
        "--scenario",
        default="clean",
        help="Scenario identifier recorded in the prediction artifact.",
    )
    args = parser.parse_args()

    result = run_prediction(
        args.ckpt,
        args.processed_dir,
        args.out,
        split=args.split,
        scenario=args.scenario,
    )
    print(result["output_csv"])
    print(result["provenance_json"])


if __name__ == "__main__":
    main()
