#!/usr/bin/env python
from __future__ import annotations

import argparse

from _bootstrap import add_src_to_path

add_src_to_path()

from cfeg.data.ssvep_synthetic import (
    generate_quality_synthetic_processed,
    generate_synthetic_processed,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir", default=None)
    parser.add_argument("--n_subjects", type=int, default=8)
    parser.add_argument("--n_trials_per_class", type=int, default=20)
    parser.add_argument("--n_blocks_per_interface", type=int, default=2)
    parser.add_argument("--n_repetitions_per_class_per_block", type=int, default=2)
    parser.add_argument("--n_classes", type=int, default=4)
    parser.add_argument("--target_sfreq", type=float, default=200.0)
    parser.add_argument("--duration_sec", type=float, default=2.0)
    parser.add_argument("--c_max", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--quality-aware", action="store_true")
    args = parser.parse_args()
    if args.quality_aware:
        info = generate_quality_synthetic_processed(
            out_dir=args.out_dir or "data/processed/synthetic_quality_v1",
            n_subjects=args.n_subjects,
            n_blocks_per_interface=args.n_blocks_per_interface,
            n_repetitions_per_class_per_block=args.n_repetitions_per_class_per_block,
            n_classes=args.n_classes,
            target_sfreq=args.target_sfreq,
            duration_sec=args.duration_sec,
            c_max=args.c_max,
            seed=args.seed,
        )
    else:
        info = generate_synthetic_processed(
            out_dir=args.out_dir or "data/processed/synthetic",
            n_subjects=args.n_subjects,
            n_trials_per_class=args.n_trials_per_class,
            n_classes=args.n_classes,
            target_sfreq=args.target_sfreq,
            duration_sec=args.duration_sec,
            c_max=args.c_max,
            seed=args.seed,
        )
    print(info)


if __name__ == "__main__":
    main()
