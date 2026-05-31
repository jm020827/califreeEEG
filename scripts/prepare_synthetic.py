#!/usr/bin/env python
from __future__ import annotations

import argparse

from _bootstrap import add_src_to_path

add_src_to_path()

from cfeg.data.ssvep_synthetic import generate_synthetic_processed


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir", default="data/processed/synthetic")
    parser.add_argument("--n_subjects", type=int, default=8)
    parser.add_argument("--n_trials_per_class", type=int, default=20)
    parser.add_argument("--n_classes", type=int, default=4)
    parser.add_argument("--target_sfreq", type=float, default=200.0)
    parser.add_argument("--duration_sec", type=float, default=2.0)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    info = generate_synthetic_processed(**vars(args))
    print(info)


if __name__ == "__main__":
    main()

