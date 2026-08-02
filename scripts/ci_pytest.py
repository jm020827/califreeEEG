#!/usr/bin/env python3
"""Run pytest and expose failures as a GitHub Actions check annotation."""

from __future__ import annotations

import subprocess
import sys


def _escape_annotation(value: str) -> str:
    return value.replace("%", "%25").replace("\r", "%0D").replace("\n", "%0A")


def main() -> int:
    completed = subprocess.run(
        [sys.executable, "-m", "pytest", "-q"],
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    print(completed.stdout, end="")
    if completed.returncode:
        excerpt = completed.stdout[-50_000:]
        print(f"::error title=pytest failed::{_escape_annotation(excerpt)}")
    return completed.returncode


if __name__ == "__main__":
    raise SystemExit(main())
