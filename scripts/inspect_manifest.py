#!/usr/bin/env python
from __future__ import annotations

import argparse

from _bootstrap import add_src_to_path

add_src_to_path()

from cfeg.data.schema import load_manifest, validate_manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--processed_dir", required=True)
    args = parser.parse_args()
    manifest = load_manifest(args.processed_dir)
    validate_manifest(manifest)
    print(f"processed_dir={args.processed_dir}")
    print(f"n_samples={len(manifest)}")
    print(f"datasets={sorted(manifest['dataset_id'].astype(str).unique().tolist())}")
    print(f"subjects={manifest['subject_id'].nunique()}")
    print(f"classes={sorted(manifest['label'].astype(int).unique().tolist())}")
    print(f"sfreq_processed={sorted(manifest['sfreq_processed'].astype(float).unique().tolist())}")


if __name__ == "__main__":
    main()

