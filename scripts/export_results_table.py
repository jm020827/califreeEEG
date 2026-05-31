#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", nargs="+", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    rows = []
    for run in args.runs:
        for csv in Path(run).rglob("*.csv"):
            df = pd.read_csv(csv)
            df["source_csv"] = str(csv)
            rows.append(df)
    if not rows:
        raise SystemExit("No CSV files found.")
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.concat(rows, ignore_index=True).to_csv(out, index=False)
    print(out)


if __name__ == "__main__":
    main()

