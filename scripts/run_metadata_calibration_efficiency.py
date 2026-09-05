#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

# Freeze a one-device view before importing torch or any CUDA-aware cfeg module.
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

import torch

REPO = Path(__file__).resolve().parents[1]
if str(REPO / "src") not in sys.path:
    sys.path.insert(0, str(REPO / "src"))

from cfeg.metadata_calibration_authority import read_json_object
from cfeg.metadata_calibration_contract import validate_metadata_calibration_plan
from cfeg.utils.config import load_config


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fail-closed entrypoint for metadata-calibration experiment preparation."
    )
    parser.add_argument("action", choices=["preflight", "execute", "recover"])
    parser.add_argument(
        "--plan",
        type=Path,
        default=REPO / "configs/analysis/metadata_calibration_efficiency_v1.yaml",
    )
    parser.add_argument("--run-root", type=Path)
    parser.add_argument("--source-tag")
    parser.add_argument(
        "--processed-asset-root",
        type=Path,
        default=Path("/home/whwovy/eeg-data/processed/wearable_v3"),
    )
    parser.add_argument("--decision-date", default="2026-09-05")
    args = parser.parse_args()
    if args.action == "recover":
        if args.run_root is None:
            raise SystemExit("Recover requires --run-root.")
        from cfeg.metadata_calibration_lifecycle import (
            recover_metadata_calibration_lifecycle_completion,
        )

        signed_hash = recover_metadata_calibration_lifecycle_completion(args.run_root)
        print(
            json.dumps(
                {
                    "schema": "cfeg.metadata-calibration-lifecycle-recovery.v1",
                    "run_root": str(args.run_root.absolute()),
                    "status": "canonical_completion_recovered",
                    "lifecycle_signed_record_sha256": signed_hash,
                },
                sort_keys=True,
                indent=2,
            )
        )
        return
    binding = validate_metadata_calibration_plan(args.plan)
    plan = load_config(args.plan, strict_env=False)
    report = {
        "schema": "cfeg.metadata-calibration-preflight.v1",
        **binding,
        "cuda_available": torch.cuda.is_available(),
        "cuda_device_count": torch.cuda.device_count(),
        "execution_freeze_blockers": plan["execution_freeze_blockers"],
    }
    if torch.cuda.is_available():
        report["cuda_devices"] = [
            torch.cuda.get_device_name(index) for index in range(torch.cuda.device_count())
        ]
    print(json.dumps(report, sort_keys=True, indent=2))
    if args.action == "execute":
        canonical_plan = (
            REPO / "configs/analysis/metadata_calibration_efficiency_v1.yaml"
        ).absolute()
        if args.plan.absolute() != canonical_plan:
            raise SystemExit("Execution requires the canonical frozen plan path.")
        if args.run_root is None or not args.source_tag:
            raise SystemExit("Execute requires --run-root and --source-tag.")
        if args.processed_asset_root.absolute() != Path(
            "/home/whwovy/eeg-data/processed/wearable_v3"
        ):
            raise SystemExit("Execution requires the canonical wearable_v3 asset root.")
        if args.decision_date != str(plan["recorded_date"]):
            raise SystemExit("Execution decision date must equal the frozen plan date.")
        from cfeg.metadata_calibration_lifecycle import (
            run_metadata_calibration_lifecycle,
        )

        result = run_metadata_calibration_lifecycle(
            run_root=args.run_root,
            processed_asset_root=args.processed_asset_root,
            source_tag=args.source_tag,
            decision_date=args.decision_date,
            authorization_basis=str(binding["authorization_basis"]),
            python_executable=Path(sys.executable),
        )
        completion = read_json_object(result.lifecycle_receipt_path)
        print(
            json.dumps(
                {
                    "schema": "cfeg.metadata-calibration-execution-finished.v1",
                    "run_root": str(result.run_root),
                    "lifecycle_receipt_path": str(result.lifecycle_receipt_path),
                    "lifecycle_signed_record_sha256": completion[
                        "signed_record_sha256"
                    ],
                    "source_gate_passed": result.source_gate_passed,
                    "source_result_bundle": str(result.source.result_bundle_root),
                    "source_result_bundle_sha256": result.source.result_bundle_sha256,
                    "held_phase_opened": result.held is not None,
                    "held_result_bundle": (
                        None if result.held is None else str(result.held.result_bundle_root)
                    ),
                    "held_result_bundle_sha256": (
                        None if result.held is None else result.held.result_bundle_sha256
                    ),
                    "generated_private_keys_destroyed": (
                        result.generated_private_keys_destroyed
                    ),
                },
                sort_keys=True,
                indent=2,
            )
        )


if __name__ == "__main__":
    main()
