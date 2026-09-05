#!/usr/bin/env python3
from __future__ import annotations


def main() -> None:
    raise SystemExit(
        "Disabled: standalone sealing bypasses the encrypted-key, lease, worker-isolation, "
        "one-shot-claim, and atomic lifecycle. Use run_metadata_calibration_efficiency.py."
    )


if __name__ == "__main__":
    main()
