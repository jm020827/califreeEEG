#!/usr/bin/env python
from __future__ import annotations

import argparse
import json

from _bootstrap import add_src_to_path

add_src_to_path()

from cfeg.data.migrate_labels import migrate_processed_label_alignment


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Remap existing Wang/BETA labels by stimulus frequency without rewriting EEG."
    )
    parser.add_argument("--processed-dir", action="append", required=True)
    parser.add_argument("--apply", action="store_true", help="Write changes; default is dry-run.")
    args = parser.parse_args()
    for processed_dir in args.processed_dir:
        result = migrate_processed_label_alignment(processed_dir, apply=args.apply)
        print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
