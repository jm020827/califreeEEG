#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from _bootstrap import add_src_to_path

add_src_to_path()

from cfeg.analysis.metadata_calibration_v2_synthetic import (
    DEFAULT_SYNTHETIC_PLAN_PATH,
    build_lockbox_authorization_template,
    build_preparation_receipt,
    load_json_object,
    run_development_to_path,
    run_lockbox_to_path,
    validate_synthetic_contract,
    write_json_exclusive,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Execute the byte-frozen V2 synthetic mechanism checks. Reserved seeds are "
            "selected only by their named phase; the lockbox requires a separate exact "
            "one-time authorization receipt."
        )
    )
    parser.add_argument("--plan", type=Path, default=DEFAULT_SYNTHETIC_PLAN_PATH)
    subparsers = parser.add_subparsers(dest="phase", required=True)

    prepare = subparsers.add_parser("prepare", help="validate bindings without running a seed")
    prepare.add_argument("--output", type=Path, required=True)
    prepare.add_argument(
        "--lockbox-output",
        type=Path,
        help="optionally include a non-authorizing template for this future output path",
    )
    prepare.add_argument(
        "--development-result",
        type=Path,
        help="record the frozen engineering result in the optional lockbox template",
    )

    development = subparsers.add_parser(
        "development", help="run the frozen development seed as engineering evidence"
    )
    development.add_argument("--output", type=Path, required=True)

    lockbox = subparsers.add_parser(
        "lockbox", help="consume one exact authorization and one never-used output path"
    )
    lockbox.add_argument("--authorization", type=Path, required=True)
    lockbox.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    contract = validate_synthetic_contract(args.plan)
    if args.phase == "prepare":
        receipt = build_preparation_receipt(contract, output_path=args.output)
        if args.lockbox_output is not None:
            receipt.pop("completion_receipt_sha256")
            receipt["lockbox_authorization_template"] = build_lockbox_authorization_template(
                contract,
                output_path=args.lockbox_output,
                development_result_path=args.development_result,
            )
            from cfeg.analysis.metadata_calibration_v2_synthetic import _canonical_json_sha256

            receipt["completion_receipt_sha256"] = _canonical_json_sha256(receipt)
        written = write_json_exclusive(args.output, receipt)
        status = receipt["status"]
        digest = receipt["completion_receipt_sha256"]
    elif args.phase == "development":
        written = run_development_to_path(contract, output_path=args.output)
        result = load_json_object(written, name="synthetic development result")
        status = result["status"]
        digest = result["result_sha256"]
    elif args.phase == "lockbox":
        authorization = load_json_object(args.authorization, name="synthetic lockbox authorization")
        written = run_lockbox_to_path(
            contract,
            authorization=authorization,
            output_path=args.output,
        )
        result = load_json_object(written, name="synthetic lockbox result")
        status = result["status"]
        digest = result["result_sha256"]
    else:  # pragma: no cover - argparse enforces this.
        raise AssertionError(f"Unhandled phase {args.phase!r}.")
    print(
        json.dumps(
            {
                "schema": "cfeg.metadata-calibration-v2-synthetic-cli-result.v1",
                "phase": args.phase,
                "status": status,
                "output": str(written),
                "receipt_sha256": digest,
            },
            sort_keys=True,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
