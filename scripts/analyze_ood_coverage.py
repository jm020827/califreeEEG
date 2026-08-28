#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
from _bootstrap import add_src_to_path

add_src_to_path()

from cfeg.analysis.ood_coverage import compare_ood_coverage
from cfeg.data.schema import load_manifest


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare paired A0/treatment predictions by subject and OOD cell."
    )
    parser.add_argument("--baseline", required=True, help="A0 prediction CSV.")
    parser.add_argument("--candidate", required=True, help="A2/A3 prediction CSV.")
    parser.add_argument("--processed-dir", required=True, help="Directory containing manifest.*.")
    parser.add_argument(
        "--cell-columns",
        nargs="+",
        default=["subject_id", "dataset_id", "electrode_type", "window_duration_sec"],
    )
    parser.add_argument(
        "--tag",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Constant cell dimension such as scenario=channel_drop_50 (repeatable).",
    )
    parser.add_argument("--success-threshold", type=float, required=True)
    parser.add_argument(
        "--expected-n-labels",
        type=int,
        default=None,
        help="Require every subject/cell to contain this many classes.",
    )
    parser.add_argument("--out-prefix", required=True)
    args = parser.parse_args()

    tags = _parse_tags(args.tag)
    cells, subjects, summary = compare_ood_coverage(
        pd.read_csv(args.baseline),
        pd.read_csv(args.candidate),
        load_manifest(args.processed_dir),
        cell_columns=args.cell_columns,
        success_threshold=args.success_threshold,
        expected_n_labels=args.expected_n_labels,
        tags=tags,
    )
    prefix = Path(args.out_prefix)
    prefix.parent.mkdir(parents=True, exist_ok=True)
    cells.to_csv(prefix.with_name(f"{prefix.name}_cells.csv"), index=False)
    subjects.to_csv(prefix.with_name(f"{prefix.name}_subjects.csv"), index=False)
    summary_path = prefix.with_name(f"{prefix.name}_summary.json")
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(summary_path)


def _parse_tags(values: list[str]) -> dict[str, str]:
    tags: dict[str, str] = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"Tag must use KEY=VALUE syntax, got {value!r}.")
        key, item = value.split("=", 1)
        if not key or not item:
            raise ValueError(f"Tag must use non-empty KEY=VALUE syntax, got {value!r}.")
        if key in tags:
            raise ValueError(f"Duplicate tag key: {key}")
        tags[key] = item
    return tags


if __name__ == "__main__":
    main()
