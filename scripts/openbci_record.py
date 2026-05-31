#!/usr/bin/env python
from __future__ import annotations


def main() -> None:
    try:
        import brainflow  # noqa: F401
    except Exception:
        raise SystemExit(
            "BrainFlow is not installed. Install brainflow or export CSV from OpenBCI GUI and use "
            "scripts/openbci_convert.py."
        )
    raise SystemExit("OpenBCI live recording scaffold is ready; implement board-specific parameters before use.")


if __name__ == "__main__":
    main()

