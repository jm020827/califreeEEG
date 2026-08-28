#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml
from _bootstrap import add_src_to_path

add_src_to_path()

from cfeg.baselines.evaluate import evaluate_frequency_baseline


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate training-free CCA or FBCCA.")
    parser.add_argument("--processed-dir", required=True)
    parser.add_argument("--method", choices=["cca", "fbcca"], required=True)
    parser.add_argument(
        "--selection-csv",
        default=None,
        help=(
            "Optional split.csv or CSV containing explicit sample_id values. "
            "Use this for held-out comparison with learned models."
        ),
    )
    parser.add_argument(
        "--selection-split",
        default="test",
        help="Split value selected when --selection-csv contains a split column.",
    )
    parser.add_argument(
        "--channel-set",
        required=True,
        help="Named set from configs/channel_sets.yaml, or 'all'.",
    )
    parser.add_argument(
        "--allow-channel-intersection",
        action="store_true",
        help="Allow missing named channels instead of failing (not recommended for comparisons).",
    )
    parser.add_argument("--max-subjects", type=int, default=None)
    parser.add_argument("--n-harmonics", type=int, default=3)
    parser.add_argument("--regularization", type=float, default=1e-8)
    parser.add_argument(
        "--filterbank-config",
        default=None,
        help="Optional YAML mapping accepted by FilterBankConfig.",
    )
    parser.add_argument("--trial-time-sec", type=float, default=2.0)
    parser.add_argument("--out-prefix", required=True)
    args = parser.parse_args()

    filterbank = None
    if args.filterbank_config:
        with Path(args.filterbank_config).open(encoding="utf-8") as handle:
            filterbank = yaml.safe_load(handle) or {}
    predictions, subjects, summary = evaluate_frequency_baseline(
        args.processed_dir,
        method=args.method,
        channel_set=args.channel_set,
        selection_csv=args.selection_csv,
        selection_split=args.selection_split,
        strict_channel_set=not args.allow_channel_intersection,
        max_subjects=args.max_subjects,
        n_harmonics=args.n_harmonics,
        regularization=args.regularization,
        filterbank=filterbank,
        trial_time_sec=args.trial_time_sec,
    )
    prefix = Path(args.out_prefix)
    prefix.parent.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(prefix.with_name(f"{prefix.name}_predictions.csv"), index=False)
    subjects.to_csv(prefix.with_name(f"{prefix.name}_subjects.csv"), index=False)
    summary_path = prefix.with_name(f"{prefix.name}_summary.json")
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(summary_path)


if __name__ == "__main__":
    main()
