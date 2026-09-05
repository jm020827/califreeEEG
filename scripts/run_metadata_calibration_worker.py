#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO / "src") not in sys.path:
    sys.path.insert(0, str(REPO / "src"))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run one exact signed metadata-calibration manifest job."
    )
    parser.add_argument("--trusted-signing-public-key", type=Path, required=True)
    parser.add_argument("--job-capability", type=Path, required=True)
    parser.add_argument("--checkpoint-group", required=True)
    parser.add_argument("--job-id", required=True)
    args = parser.parse_args()

    # This process intentionally receives no label-decryption key or password.
    if any("FINALIZER" in name and "PUBLIC" not in name for name in os.environ):
        raise RuntimeError("Worker environment unexpectedly contains finalizer secret material.")

    from cfeg.metadata_calibration_job import (
        run_projected_metadata_calibration_job,
    )

    completed, isolation = run_projected_metadata_calibration_job(
        job_capability_path=args.job_capability,
        trusted_signing_public_key_path=args.trusted_signing_public_key,
        repository_root=REPO,
        checkpoint_group=args.checkpoint_group,
        job_id=args.job_id,
    )
    print(
        json.dumps(
            {
                "job_id": completed.job_id,
                "producer_kind": completed.producer_kind,
                "canonical_job_root": str(completed.canonical_job_root),
                "producer_receipt_file_sha256": (
                    completed.producer_receipt_file_sha256
                ),
                "isolation_attestation": isolation,
            },
            sort_keys=True,
            separators=(",", ":"),
        )
    )


if __name__ == "__main__":
    main()
