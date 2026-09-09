"""Frozen generated-only C2 condition ladder; no human reader or numerical rescue."""

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
from cfeg.analysis import metadata_trca_prior as native
from cfeg.analysis import task_trca_pair_s_features as features
from cfeg.analysis import task_trca_pair_s_operator as pair
from cfeg.analysis import task_trca_shape_operator as primitive

SEED = 20260909
CONDITIONS = (1, 1000, 100000, 120000)
DESIGN_SHA = "1561541f541d3cbf9a3b2b5a35e78a409edc4099cff3bddf6b343c0d37109ee3"
PROGRAM_SHA = "78ca14d5c155841b727ef8b35b958188639dd7677d1e41ca9e8ffa9bc3bc3a10"


def require(value, message):
    if not value:
        raise ValueError(message)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    with path.open("x") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    path.chmod(0o400)


def generated_support():
    """Exactly one frozen draw order; no conditions, parameters or input paths."""
    rng = np.random.default_rng(SEED)
    shared = rng.normal(size=(12, 5, 8, 125))
    noise = rng.normal(size=(5, 12, 5, 8, 125))
    return shared + noise


def transformed_case(support, k, condition):
    """Same native prefix with trace-preserving non-diagonal C congruence.

    Each class/band transformation is independent. This changes coordinates,
    not the exact-arithmetic generalized spectrum; it is not biological data.
    """
    require(k in (3, 5) and condition in CONDITIONS, "Frozen condition and prefix only")
    x = np.asarray(support, dtype=np.float64)
    require(x.shape == (5, 12, 5, 8, 125), "Frozen generated support shape")
    changed = x[:k].copy()
    s0, c = np.empty((5, 12, 8, 8)), np.empty((5, 12, 8, 8))
    for label in range(12):
        for band in range(5):
            _, old_c = native.trca_matrices(x[:k, label, band])
            eigenvalues, vectors = np.linalg.eigh(old_c)
            require(bool(np.all(eigenvalues > 0)), "Generated original C must be SPD")
            spectrum = np.power(float(condition), np.arange(8) / 7)
            desired = spectrum * np.trace(old_c) / spectrum.sum()
            transform = (vectors * np.sqrt(desired / eigenvalues)) @ vectors.T
            changed[:, label, band] = transform @ x[:k, label, band]
            s0[band, label], c[band, label] = native.trca_matrices(changed[:, label, band])
    return changed, s0, c, features.pair_cross_products(changed)


def raw_h(s, c):
    """Original two solves and validation order, for diagnostics without rescue."""
    s, c = primitive._symmetric(s, "s"), primitive._symmetric(c, "c", constant=True)
    lower = torch.linalg.cholesky(c)
    left = torch.linalg.solve_triangular(lower, s, upper=False)
    return torch.linalg.solve_triangular(lower, left.mT, upper=False).mT


def diagnostics(h, c):
    error = torch.linalg.matrix_norm(h - h.mT, ord="fro", dim=(-2, -1))
    norm = torch.linalg.matrix_norm(h, ord="fro", dim=(-2, -1))
    ratios = error / (1e-12 * torch.clamp_min(norm, 1.0))
    # Symmetrized roots are diagnostic only. This H is NEVER passed to c_projectors.
    roots = torch.linalg.eigvalsh((h + h.mT) / 2)
    c_roots = torch.linalg.eigvalsh(c)
    gaps = roots[..., -1] - roots[..., -2]
    limits = 1e-10 * torch.clamp_min(roots.abs().amax(-1), 1.0)
    return {
        "maximum_symmetry_ratio": float(ratios.max()),
        "symmetry_ratios": ratios.tolist(),
        "rejected_symmetry_indices_zero_based": torch.nonzero(ratios > 1).tolist(),
        "minimum_top_gap_ratio": float((gaps / limits).min()),
        "top_gap_ratios": (gaps / limits).tolist(),
        "rejected_top_gap_indices_zero_based": torch.nonzero(gaps <= limits).tolist(),
        "c_condition_min": float((c_roots[..., -1] / c_roots[..., 0]).min()),
        "c_condition_max": float((c_roots[..., -1] / c_roots[..., 0]).max()),
        "c_minimum_eigenvalue": float(c_roots[..., 0].min()),
        "h_frobenius_norm": norm.tolist(),
        "h_antisymmetric_frobenius_norm": error.tolist(),
    }


def measure(s0, c, cross, logits):
    s0, c, cross, logits = (
        torch.tensor(value, dtype=torch.float64) for value in (s0, c, cross, logits)
    )
    s = pair.weighted_numerator(s0, cross, logits)
    h = raw_h(s, c)
    row = {"batch": diagnostics(h, c), "scalar_failures": []}
    arrays = {"s": s.numpy(), "h_batch": h.numpy()}
    uniform = bool(torch.all(logits == logits[..., :1]))
    row["uniform_s0_exact"] = bool(torch.equal(s, s0)) if uniform else None
    require(not uniform or row["uniform_s0_exact"], "Uniform S0 recovery failed")
    batch_f = None
    try:
        batch_f = pair.pair_projectors(s0, c, cross, logits)
        row["batch"]["status"] = "PASS"
        arrays["f_batch"] = batch_f.numpy()
    except ValueError as error:
        row["batch"].update(status="REJECTED", error=str(error))
    scalar_h = torch.empty_like(h)
    scalar_f = torch.empty_like(s)
    for band in range(5):
        for label in range(12):
            scalar_h[band, label] = raw_h(s[band, label], c[band, label])
            try:
                scalar_f[band, label] = pair.c_projectors(s[band, label], c[band, label])
            except ValueError as error:
                row["scalar_failures"].append({"band": band, "class": label, "error": str(error)})
    row["scalar"] = diagnostics(scalar_h, c)
    row["scalar"]["status"] = "REJECTED" if row["scalar_failures"] else "PASS"
    arrays["h_scalar"] = scalar_h.numpy()
    if not row["scalar_failures"]:
        arrays["f_scalar"] = scalar_f.numpy()
        if batch_f is not None:
            row["scalar_batch_f_max_error"] = float((batch_f - scalar_f).abs().max())
            row["scalar_batch_f_close"] = bool(
                torch.allclose(batch_f, scalar_f, rtol=1e-8, atol=1e-10)
            )
    row["status"] = (
        "PASS"
        if row["batch"]["status"] == row["scalar"]["status"] == "PASS"
        and row.get("scalar_batch_f_close", False)
        else "REJECTED"
    )
    row["first_failure"] = None
    if row["batch"]["status"] == "REJECTED":
        indices = row["batch"]["rejected_symmetry_indices_zero_based"]
        if not indices:
            indices = row["batch"]["rejected_top_gap_indices_zero_based"]
        row["first_failure"] = {
            "backend": "batch_projector",
            "band": indices[0][0] if indices else None,
            "class": indices[0][1] if indices else None,
            "reason": row["batch"]["error"],
            "location_scope": "First diagnostic rejecting matrix in batch; scalar execution order is separately recorded",
        }
    elif row["scalar_failures"]:
        row["first_failure"] = {"backend": "scalar_projector", **row["scalar_failures"][0]}
    elif row["status"] != "PASS":
        position = np.unravel_index(int((batch_f - scalar_f).abs().argmax()), s.shape)
        row["first_failure"] = {
            "backend": "scalar_batch_comparison",
            "band": int(position[0]),
            "class": int(position[1]),
            "reason": "Projector comparison tolerance exceeded",
        }
    return row, arrays


def exit_code(result):
    return 0 if result["status"] == "GENERATED_CONDITIONING_PASS" else 2


def run(output):
    output = Path(output).absolute()
    require(
        output.resolve() == output and not output.exists() and output.parent.is_dir(),
        "Fresh unaliased output required",
    )
    require(
        all(
            os.environ.get(name) == "1"
            for name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS")
        ),
        "CPU1 only",
    )
    torch.set_num_threads(1)
    design = ROOT / "configs/analysis/task_trca_pair_s_v1_design.json"
    program = ROOT / "configs/analysis/metadata_learning_program_v1.json"
    require(
        sha(design) == DESIGN_SHA and sha(program) == PROGRAM_SHA, "Frozen science/program pins"
    )
    files = {Path(__file__).resolve(), design, program, ROOT / "docs/task_trca_pair_s_v1_build.md"}
    files.update(
        Path(module.__file__).resolve()
        for name, module in sys.modules.items()
        if name.startswith("cfeg") and getattr(module, "__file__", None)
    )
    pins = {str(path.relative_to(ROOT)): sha(path) for path in sorted(files)}
    started = time.perf_counter()
    output.mkdir()
    write_json(
        output / "start.json",
        {
            "schema": "cfeg.pair_s.conditioning.v1",
            "seed": SEED,
            "conditions": CONDITIONS,
            "code_config_pins": pins,
            "cpu_threads": 1,
            "seconds_budget": 60,
            "bytes_budget": 67108864,
            "human_reads": False,
        },
    )

    def expired(*unused):
        raise TimeoutError("Frozen conditioning gate60 second budget")

    signal.signal(signal.SIGALRM, expired)
    signal.alarm(60)
    try:
        support = generated_support()
        rows, arrays = [], {"generated_base_support": support}
        for k in (3, 5):
            for condition in CONDITIONS:
                changed, s0, c, cross = transformed_case(support, k, condition)
                case_key = f"k{k}_cond{condition}"
                arrays.update(
                    {
                        f"{case_key}_{name}": value
                        for name, value in (
                            ("support", changed),
                            ("s0", s0),
                            ("c", c),
                            ("cross", cross),
                        )
                    }
                )
                n = k * (k - 1) // 2
                for arm in ("UNIFORM_C2", "FIXED_BOUNDED_PAIR_LOGITS"):
                    logits = (
                        np.zeros((5, n))
                        if arm == "UNIFORM_C2"
                        else np.broadcast_to(
                            np.linspace(-primitive.LOGIT_BOUND, primitive.LOGIT_BOUND, n), (5, n)
                        ).copy()
                    )
                    row, values = measure(s0, c, cross, logits)
                    row.update(k=k, target_condition=condition, arm=arm)
                    rows.append(row)
                    key = f"{case_key}_{arm}"
                    arrays[f"{key}_logits"] = logits
                    arrays.update({f"{key}_{name}": value for name, value in values.items()})
        array_path = output / "generated_arrays.npz"
        with array_path.open("xb") as stream:
            np.savez(stream, **arrays)
            stream.flush()
            os.fsync(stream.fileno())
        array_path.chmod(0o400)
        failed = [i for i, row in enumerate(rows) if row["status"] != "PASS"]
        result = {
            "schema": "cfeg.pair_s.conditioning.v1",
            "status": "GENERATED_CONDITIONING_PASS"
            if not failed
            else "GENERATED_CONDITIONING_VALIDITY_FAILURE",
            "start_sha256": sha(output / "start.json"),
            "generated_arrays": {
                "path": str(array_path),
                "sha256": sha(array_path),
                "bytes": array_path.stat().st_size,
            },
            "rows": rows,
            "rejected_row_indices": failed,
            "first_rejected_row": failed[0] if failed else None,
            "first_failure": {"row": failed[0], **rows[failed[0]]["first_failure"]}
            if failed
            else None,
            "elapsed_seconds": time.perf_counter() - started,
            "optimizer_updates": 0,
            "human_reads": False,
            "query_reads_or_scores": 0,
            "gpu_used": False,
            "held60_access": False,
            "numerical_policy_changed": False,
            "interpretation": "Coordinate-conditioning generated gate, not biological observations, learning or human efficacy; no selected passing subset.",
            "batch_scope": "Scalar/batch projector comparison shares the previously computed S; not independent numerator or learner batching parity.",
            "failure_precedence": "If failure.json exists it overrides every result status, including PASS.",
        }
        require(
            sum(p.stat().st_size for p in output.iterdir())
            + len((json.dumps(result, indent=2, allow_nan=False) + "\n").encode())
            <= 67108864,
            "Conditioning output budget",
        )
        require(time.perf_counter() - started < 60, "Conditioning elapsed budget")
        write_json(output / "result.json", result)
        print(
            json.dumps(
                {
                    "status": result["status"],
                    "result_sha256": sha(output / "result.json"),
                    "elapsed_seconds": result["elapsed_seconds"],
                    "rejected_rows": len(failed),
                },
                allow_nan=False,
            )
        )
        return result
    except Exception as error:
        write_json(
            output / "failure.json",
            {
                "status": "ENGINEERING_EXECUTION_FAILURE",
                "error": str(error),
                "elapsed_seconds": time.perf_counter() - started,
                "human_reads": False,
            },
        )
        raise
    finally:
        signal.alarm(0)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    raise SystemExit(exit_code(run(parser.parse_args().output)))
