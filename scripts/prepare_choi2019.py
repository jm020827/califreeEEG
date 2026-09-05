#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path

from _bootstrap import add_src_to_path

add_src_to_path()

from cfeg.data.prepare_choi2019 import (
    _verify_raw_archive,
    audit_source_schema,
    ensure_extracted,
    prepare,
)
from cfeg.utils.config import load_config

DEFAULT_RAW_DIR = "/home/whwovy/eeg-data/raw/choi2019_gigadb"
DEFAULT_OUT_DIR = "/home/whwovy/eeg-data/processed/choi2019_v1"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/data/choi2019.yaml")
    parser.add_argument("--raw-dir", default=DEFAULT_RAW_DIR)
    parser.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    parser.add_argument(
        "--subjects",
        default=None,
        help="Optional comma-separated subject numbers for an explicitly labeled smoke asset.",
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--extract-only", action="store_true")
    mode.add_argument("--audit-only", action="store_true")
    args = parser.parse_args()

    cfg = load_config(args.config, strict_env=False)
    raw_dir = Path(args.raw_dir)
    subjects = _parse_subjects(args.subjects)
    if args.extract_only or args.audit_only:
        _verify_raw_archive(raw_dir.resolve(), cfg)
        extracted = ensure_extracted(raw_dir.resolve(), cfg)
    if args.extract_only:
        print(extracted)
        return
    if args.audit_only:
        print(json.dumps(audit_source_schema(extracted, cfg, subjects=subjects), indent=2))
        return
    receipt = prepare(raw_dir, Path(args.out_dir), cfg, subjects=subjects)
    print(json.dumps(receipt, indent=2))


def _parse_subjects(raw: str | None) -> list[int] | None:
    if not raw:
        return None
    values = [int(value.strip()) for value in raw.split(",") if value.strip()]
    if len(values) != len(set(values)):
        raise ValueError("Duplicate Choi subject numbers are not allowed")
    return values


if __name__ == "__main__":
    main()
