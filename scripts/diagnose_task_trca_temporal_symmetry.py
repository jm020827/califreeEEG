"""One frozen postfailure S/C-only replay; no query, numeric M, training or repair."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import signal
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from cfeg.analysis import task_trca_shape_archive as pinned
from cfeg.analysis import task_trca_shape_operator as operator

ATTEMPT = Path("/home/whwovy/task-trca-temporal-source39-v1-primary1-20260909")
FAILURE_SHA = "4b57bc9bad8cd53eda24f49cbee063e32eba3729ce72d10b5d3b94de2f167c7e"
KEY = (63, 1, 125, 3)


def require(value, message):
    if not value:
        raise ValueError(message)


def source_matrices(item, key=KEY):
    """Hash all bytes for integrity but numerically decode ONLY keys/S/C."""
    with (
        pinned._PinnedFile(item["path"], item["sha256"], item["bytes"]) as source,
        os.fdopen(os.dup(source.fd), "rb") as stream,
        np.load(stream, allow_pickle=False) as arrays,
    ):
        keys = arrays["keys"]
        require(keys.dtype.kind in "iu" and keys.ndim == 2 and keys.shape[1] == 4, "source keys")
        positions = np.flatnonzero(np.all(keys == np.asarray(key), axis=1))
        require(len(positions) == 1, "exact single diagnostic key")
        result = {name: arrays[name][positions[0]].copy() for name in ("s", "c")}
    for value in result.values():
        require(value.shape == (5, 12, 8, 8) and np.isfinite(value).all(), "fixed finite S/C")
    return result


def symmetry(value):
    norm = torch.linalg.matrix_norm(value, ord="fro", dim=(-2, -1))
    error = torch.linalg.matrix_norm(value - value.mT, ord="fro", dim=(-2, -1))
    ratio = error / (1e-12 * torch.clamp_min(norm, 1.0))
    return {
        "norm": norm.tolist(),
        "antisymmetric_error": error.tolist(),
        "error_to_frozen_tolerance_ratio": ratio.tolist(),
        "maximum_ratio": float(ratio.max()),
        "failed_band_class_indices_zero_based": torch.nonzero(ratio > 1).tolist(),
    }


def measure(s, c):
    s, c = (torch.tensor(v, dtype=torch.float64) for v in (s, c))
    original = {"s": symmetry(s), "c": symmetry(c)}
    s, c = operator._symmetric(s, "s", constant=True), operator._symmetric(c, "c", constant=True)
    eigen = torch.linalg.eigvalsh(c)
    require(bool(torch.all(eigen[..., 0] > 0)), "Strict C SPD remains mandatory")
    result = {
        "original_symmetry": original,
        "c_min_eigenvalue": eigen[..., 0].tolist(),
        "c_condition_number": (eigen[..., -1] / eigen[..., 0]).tolist(),
        "operators": {},
    }
    for name in ("C1_ISO", "C2_UNIFORM_NECESSARY_CONDITION"):
        b = c
        if name == "C1_ISO":
            tau = operator.ETA * eigen[..., 0] / operator.R_MAX
            b = c + torch.diag_embed(tau[..., None] * torch.ones_like(c[..., 0, :]))
        lower = torch.linalg.cholesky(b)
        left = torch.linalg.solve_triangular(lower, s, upper=False)
        h = torch.linalg.solve_triangular(lower, left.mT, upper=False).mT
        row = symmetry(h)
        try:
            operator.leading_projector(h)
            row["strict_projector_status"] = "PASS"
            row["strict_projector_error"] = None
        except ValueError as error:
            row["strict_projector_status"] = "REJECTED"
            row["strict_projector_error"] = str(error)
        result["operators"][name] = row
    return result


def run(output):
    output = Path(output).absolute()
    require(
        output.resolve() == output and not output.exists() and output.parent.is_dir(),
        "New unaliased diagnostic receipt required",
    )
    require(
        all(
            os.environ.get(name) == "1"
            for name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS")
        ),
        "CPU1 only",
    )
    torch.set_num_threads(1)
    began = time.perf_counter()

    def expired(*unused):
        raise TimeoutError("Diagnostic60second budget")

    signal.signal(signal.SIGALRM, expired)
    signal.alarm(60)
    try:
        with pinned._PinnedFile(ATTEMPT / "failure.json", FAILURE_SHA) as source:
            failure = json.loads(pinned.pread_exact(source.fd, source.before.st_size, 0))
        require(failure["status"] == "VALIDITY_FAILURE", "Failure-first scope")
        items = [failure["artifacts"][name] for name in ("source1", "source2")]
        for index, item in enumerate(items, 1):
            require(item["path"] == str(ATTEMPT / f"source{index}.npz"), "Exact saved source path")
        left, right = (source_matrices(item) for item in items)
        require(
            all(np.array_equal(left[name], right[name]) for name in ("s", "c")),
            "Two source snapshots differ",
        )
        result = {
            "status": "BOUNDED_S_C_DIAGNOSIS_COMPLETE",
            "code_sha256": {
                name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
                for name in (
                    "scripts/diagnose_task_trca_temporal_symmetry.py",
                    "src/cfeg/analysis/task_trca_shape_operator.py",
                    "src/cfeg/analysis/task_trca_shape_archive.py",
                )
            },
            "failure_sha256": FAILURE_SHA,
            "key": list(KEY),
            "source_descriptors": items,
            "two_saved_snapshots_exact": True,
            "numeric_fields_read": ["keys", "s", "c"],
            "human_optimizer_updates": 0,
            "new_query_reads_or_scores": 0,
            "numeric_metadata_decoded": False,
            "held60_access": False,
            "numerical_policy_changed": False,
            "measurement": measure(left["s"], left["c"]),
            "scope": "Single saved-prefix operator replay; failed evaluation local matrix not separately captured. No repaired score, model refit, efficacy or whole-C2 stability claim.",
            "elapsed_seconds": time.perf_counter() - began,
        }
        encoded = json.dumps(result, indent=2, allow_nan=False) + "\n"
        require(len(encoded.encode()) < 1024**2, "Diagnostic1MiB output budget")
        with output.open("x") as stream:
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
        output.chmod(0o400)
        return result
    finally:
        signal.alarm(0)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = run(args.output)
    print(
        json.dumps(
            {
                "status": result["status"],
                "operators": {
                    name: {
                        key: value[key]
                        for key in (
                            "maximum_ratio",
                            "failed_band_class_indices_zero_based",
                            "strict_projector_status",
                            "strict_projector_error",
                        )
                    }
                    for name, value in result["measurement"]["operators"].items()
                },
            },
            indent=2,
        )
    )
