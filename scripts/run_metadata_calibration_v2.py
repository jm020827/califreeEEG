#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pandas as pd
from _bootstrap import add_src_to_path

add_src_to_path()

from cfeg.analysis.metadata_calibration_v2 import (
    DEFAULT_V2_ALLOCATION_PATH,
    DEFAULT_V2_PLAN_PATH,
    V2_DEFAULT_RESAMPLES,
    build_v2_preparation_receipt,
    evaluate_v2_independent_aq_gate,
    reject_forbidden_v2_data_path,
    score_v2_external_request,
    select_v2_development_candidate,
    validate_v2_plan_and_allocation,
    write_v2_completion_receipt_exclusive,
)


def _positive_integer(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("value must be a positive integer")
    return parsed


def _guarded_regular_file(value: str | Path, *, name: str) -> Path:
    path = reject_forbidden_v2_data_path(value)
    if not path.exists() or path.is_symlink() or not path.is_file():
        raise ValueError(f"{name} must be one existing, nonsymlink regular file: {path}")
    return path


def _load_json_object(value: str | Path, *, name: str) -> dict[str, Any]:
    path = _guarded_regular_file(value, name=name)
    decoded = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(decoded, Mapping):
        raise TypeError(f"{name} must decode to one JSON object.")
    return dict(decoded)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Fail-closed V2 external preparation, label-free scoring, development "
            "selection, and independent A_Q gating. This CLI has no held-data phase."
        )
    )
    parser.add_argument("--plan", type=Path, default=DEFAULT_V2_PLAN_PATH)
    parser.add_argument("--allocation", type=Path, default=DEFAULT_V2_ALLOCATION_PATH)
    subparsers = parser.add_subparsers(dest="phase", required=True)

    prepare = subparsers.add_parser(
        "prepare", help="validate the frozen plan and replay the sealed external allocation"
    )
    prepare.add_argument("--output", type=Path, required=True)

    score = subparsers.add_parser(
        "score", help="consume one exact query-outcome-free operator request"
    )
    score.add_argument("--request", type=Path, required=True)
    score.add_argument("--output", type=Path, required=True)

    select = subparsers.add_parser(
        "select", help="select and freeze one candidate from the complete development grid"
    )
    select.add_argument("--participant-deltas", type=Path, required=True)
    select.add_argument("--output", type=Path, required=True)
    select.add_argument("--resamples", type=_positive_integer, default=V2_DEFAULT_RESAMPLES)

    gate = subparsers.add_parser(
        "gate", help="evaluate the selected candidate on the independent external cohort"
    )
    gate.add_argument("--participant-deltas", type=Path, required=True)
    gate.add_argument("--selection-receipt", type=Path, required=True)
    gate.add_argument("--output", type=Path, required=True)
    gate.add_argument("--resamples", type=_positive_integer, default=V2_DEFAULT_RESAMPLES)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    # Guard every user-controlled path before reading it. The canonical V2 plan and
    # allocation paths pass; wearable/source39/held60 aliases and symlinks do not.
    plan_path = reject_forbidden_v2_data_path(args.plan)
    allocation_path = reject_forbidden_v2_data_path(args.allocation)
    contract = validate_v2_plan_and_allocation(plan_path, allocation_path)
    output = reject_forbidden_v2_data_path(args.output)

    if args.phase == "prepare":
        receipt = build_v2_preparation_receipt(contract)
    elif args.phase == "score":
        request = _load_json_object(args.request, name="V2 score request")
        receipt = score_v2_external_request(request, contract=contract)
    elif args.phase == "select":
        delta_path = _guarded_regular_file(
            args.participant_deltas, name="V2 development participant-delta CSV"
        )
        receipt = select_v2_development_candidate(
            pd.read_csv(delta_path),
            contract=contract,
            n_resamples=args.resamples,
        )
    elif args.phase == "gate":
        delta_path = _guarded_regular_file(
            args.participant_deltas, name="V2 gate participant-delta CSV"
        )
        selection = _load_json_object(
            args.selection_receipt, name="V2 development-selection receipt"
        )
        receipt = evaluate_v2_independent_aq_gate(
            pd.read_csv(delta_path),
            development_selection_receipt=selection,
            contract=contract,
            n_resamples=args.resamples,
        )
    else:  # pragma: no cover - argparse enforces a known phase.
        raise AssertionError(f"Unhandled V2 phase: {args.phase}")

    written = write_v2_completion_receipt_exclusive(output, receipt)
    print(
        json.dumps(
            {
                "schema": "cfeg.metadata-calibration-v2-cli-result.v1",
                "phase": args.phase,
                "status": receipt["status"],
                "output": str(written),
                "completion_receipt_sha256": receipt["completion_receipt_sha256"],
                "held_access_authorized": False,
            },
            sort_keys=True,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
