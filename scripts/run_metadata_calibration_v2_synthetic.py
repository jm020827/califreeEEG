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
    canonical_execution_path,
    fetch_nist_beacon_receipt,
    load_json_object,
    run_development_to_path,
    run_full_suite_test_evidence,
    run_lockbox_to_path,
    seal_authorization_record,
    validate_lockbox_authorization,
    validate_synthetic_contract,
    write_json_exclusive,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run the v8 synthetic governance phases at their exact canonical paths. "
            "The beacon phase refuses network access before the frozen target time."
        )
    )
    parser.add_argument("--plan", type=Path, default=DEFAULT_SYNTHETIC_PLAN_PATH)
    phases = parser.add_subparsers(dest="phase", required=True)
    phases.add_parser("prepare")
    phases.add_parser("development")
    phases.add_parser("test-evidence")
    phases.add_parser("fetch-beacon")
    phases.add_parser("auth-template")
    authorization = phases.add_parser("authorize")
    authorization.add_argument("--authorized-by", required=True)
    authorization.add_argument("--authorization-basis", required=True)
    authorization.add_argument("--one-time-nonce-sha256", required=True)
    phases.add_parser("lockbox")
    return parser


def _receipt_digest(payload: dict) -> str:
    for name in (
        "completion_receipt_sha256",
        "result_sha256",
        "test_evidence_sha256",
        "beacon_receipt_sha256",
        "authorization_receipt_sha256",
    ):
        if name in payload:
            return str(payload[name])
    raise ValueError("Governed phase did not produce a recognized receipt digest.")


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    contract = validate_synthetic_contract(args.plan)
    phase_to_artifact = {
        "prepare": "preparation_receipt",
        "development": "development_result",
        "test-evidence": "full_suite_test_evidence",
        "fetch-beacon": "beacon_receipt",
        "auth-template": "lockbox_authorization",
        "authorize": "lockbox_authorization",
        "lockbox": "lockbox_result",
    }
    output = canonical_execution_path(contract, phase_to_artifact[args.phase])
    output.parent.mkdir(parents=True, exist_ok=True)
    if args.phase == "lockbox":
        canonical_execution_path(contract, "seed_global_lockbox_claim").parent.mkdir(
            parents=True, exist_ok=True
        )

    if args.phase == "prepare":
        payload = build_preparation_receipt(contract, output_path=output)
        written = write_json_exclusive(output, payload)
    elif args.phase == "development":
        written = run_development_to_path(contract, output_path=output)
        payload = load_json_object(written, name="synthetic development result")
    elif args.phase == "test-evidence":
        written = run_full_suite_test_evidence(contract, output_path=output)
        payload = load_json_object(written, name="synthetic test evidence")
    elif args.phase == "fetch-beacon":
        written = fetch_nist_beacon_receipt(contract, output_path=output)
        payload = load_json_object(written, name="synthetic beacon receipt")
    elif args.phase == "auth-template":
        payload = build_lockbox_authorization_template(
            contract,
            output_path=canonical_execution_path(contract, "lockbox_result"),
        )
        print(json.dumps(payload, sort_keys=True, indent=2))
        return 0
    elif args.phase == "authorize":
        payload = build_lockbox_authorization_template(
            contract,
            output_path=canonical_execution_path(contract, "lockbox_result"),
        )
        payload.pop("authorization_receipt_sha256")
        payload.update(
            {
                "status": "authorized_for_one_time_execution",
                "authorized": True,
                "authorized_by": args.authorized_by,
                "authorization_basis": args.authorization_basis,
                "one_time_nonce_sha256": args.one_time_nonce_sha256,
            }
        )
        payload = seal_authorization_record(payload)
        validate_lockbox_authorization(
            payload,
            contract=contract,
            output_path=canonical_execution_path(contract, "lockbox_result"),
        )
        written = write_json_exclusive(output, payload)
    elif args.phase == "lockbox":
        written = run_lockbox_to_path(
            contract,
            authorization_path=canonical_execution_path(contract, "lockbox_authorization"),
            output_path=output,
        )
        payload = load_json_object(written, name="synthetic lockbox terminal result")
    else:  # pragma: no cover - argparse enforces the choices.
        raise AssertionError(f"Unhandled phase {args.phase!r}.")

    print(
        json.dumps(
            {
                "schema": "cfeg.metadata-calibration-v2-synthetic-cli-result.v2",
                "phase": args.phase,
                "status": payload["status"],
                "output": str(written),
                "receipt_sha256": _receipt_digest(payload),
            },
            sort_keys=True,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
