"""Run only the fixed artificial v1 suite; no data path or configurable scientific grid."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import scipy

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from cfeg.analysis.metadata_trca_prior_synthetic import run_suite

PLAN_SHA = "3dd7c5fa38d2b5416a2e1bd166b5b50fcaec7c00af527691d8bd4aa3b81a9da2"
PLAN = ROOT / "configs/analysis/metadata_trca_prior_synthetic_v1.json"
CODE = (
    "src/cfeg/analysis/metadata_trca_prior.py",
    "src/cfeg/analysis/metadata_trca_prior_synthetic.py",
    "scripts/run_metadata_trca_prior_synthetic.py",
)


def guard(event, args):
    """Reviewed Python defense in depth; not an OS sandbox or exhaustive read trace."""
    if event in ("socket.connect", "socket.getaddrinfo", "subprocess.Popen"):
        raise RuntimeError("No network/process execution during artificial suite")
    if (
        event == "open"
        and isinstance(args[0], (str, bytes, os.PathLike))
        and Path(os.fsdecode(args[0])).suffix.lower()
        in (
            ".mat",
            ".npz",
            ".npy",
            ".parquet",
            ".pkl",
            ".pt",
            ".h5",
            ".hdf5",
        )
    ):
        raise RuntimeError("Data-file loading is not part of this artificial suite")


def publish(path: Path, value: dict):
    data = (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o400)
    with os.fdopen(fd, "wb") as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())
    return hashlib.sha256(data).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if hashlib.sha256(PLAN.read_bytes()).hexdigest() != PLAN_SHA:
        raise RuntimeError("Frozen plan changed")
    if subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True):
        raise RuntimeError("Commit reviewed implementation before suite exposure")
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    hashes = {p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest() for p in CODE}
    output = args.output.absolute()
    output.mkdir(exist_ok=False)  # never overwrite or silently retry a prior attempt
    start = {
        "study_id": "metadata-trca-prior-synthetic-v1",
        "source_revision": revision,
        "source_hashes": hashes,
        "plan_sha256": PLAN_SHA,
        "started_at": datetime.now(timezone.utc).isoformat(),
        "python": sys.version.split()[0],
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "scope": "artificial only; no human efficacy or automatic promotion",
    }
    publish(output / "start.json", start)
    sys.addaudithook(guard)
    before = time.perf_counter()
    result = run_suite()
    result["provenance"] = start
    result["elapsed_seconds"] = time.perf_counter() - before
    result["finished_at"] = datetime.now(timezone.utc).isoformat()
    digest = publish(output / "result.json", result)
    print(
        json.dumps(
            {
                "output": str(output),
                "sha256": digest,
                "screen": result["screen"],
                "rows": len(result["rows"]),
                "elapsed_seconds": result["elapsed_seconds"],
            }
        )
    )


if __name__ == "__main__":
    main()
