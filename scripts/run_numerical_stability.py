"""One prospectively frozen, generated-only numerical-methods measurement run."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import platform
import signal
import stat
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import mpmath
import numpy as np
import scipy
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
DESIGN = ROOT / "configs/analysis/numerical_stability_v1.json"
DESIGN_SHA = "04d0968d478c795e8a0860c561fcfb881b9d510b5777c1a7ff45e65764b6fbe8"
LEGACY = Path("/home/whwovy/task-trca-pair-s-conditioning-Vqli4o/primary1")
METHODS = {"n1": "N1_SYM_CHOLESKY", "n2": "N2_GVD_IMPLICIT"}
EXCLUSIONS = {
    "human_reads": False,
    "query_reads_or_scores": 0,
    "optimizer_updates": 0,
    "gpu_used": False,
    "held60": False,
    "external_outreach": False,
    "paid_resources": False,
}
PIN_FILES = (
    "configs/analysis/numerical_stability_v1.json",
    "docs/numerical_stability_v1_design.md",
    "docs/numerical_stability_v1_implementation.md",
    "scripts/run_numerical_stability.py",
    "scripts/audit_numerical_stability.py",
    "src/cfeg/analysis/numerical_stability_operator.py",
    "src/cfeg/analysis/task_trca_shape_operator.py",
    "src/cfeg/__init__.py",
    "src/cfeg/analysis/__init__.py",
)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def regular_readonly(path):
    path = Path(path)
    require(path.is_absolute() and path.resolve() == path, "Unaliased absolute input required")
    info = path.lstat()
    require(stat.S_ISREG(info.st_mode), "Regular input required")
    require(info.st_nlink == 1, "Hardlinked input is not unaliased")
    require(stat.S_IMODE(info.st_mode) == 0o400, "Input must be immutable mode0400")
    return path


def bound_bytes(path, expected_hash):
    path = regular_readonly(path)
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
    with os.fdopen(fd, "rb") as stream:
        info = os.fstat(stream.fileno())
        require(
            stat.S_ISREG(info.st_mode) and info.st_nlink == 1, "Unaliased regular input required"
        )
        require(stat.S_IMODE(info.st_mode) == 0o400, "Bound input must be mode0400")
        value = stream.read()
        require(len(value) == info.st_size, "Bound input size changed")
    require(hashlib.sha256(value).hexdigest() == expected_hash, "Legacy input hash mismatch")
    return value


def sym(x):
    return (x + x.T) / 2


def probe(raw):
    value = sym(raw)
    return value / np.linalg.norm(value, "fro")


def canonical_qr(raw):
    q, r = np.linalg.qr(raw)
    return q * np.where(np.diag(r) < 0, -1.0, 1.0)


def directions(b, c, w, es, eb, ec):
    lb, lc = np.linalg.cholesky(b), np.linalg.cholesky(c)
    return tuple(sym((l @ p) @ l.T) for l, p in ((lc, w), (lb, es), (lb, eb), (lc, ec)))


def new_cases(seed, group):
    """Literal frozen grid; toy tests must use a nonregistered seed."""
    rng = np.random.default_rng(seed)
    u, v = (canonical_qr(rng.normal(size=(8, 8))) for _ in range(2))
    w, es, eb, ec = (probe(rng.normal(size=(8, 8))) for _ in range(4))
    logits = np.linspace(-np.log(2) / 2, np.log(2) / 2, 8)
    mass = np.exp(logits - logits.max())
    shape = 8 * mass / mass.sum()
    for condition in (1, 1000, 120000, 1000000):
        for scale in (2.0**-20, 1.0, 2.0**20):
            c = scale * (
                np.eye(8)
                if condition == 1
                else sym((u @ np.diag(condition ** (-np.arange(8) / 7))) @ u.T)
            )
            for spectrum in ("distinct", "repeated"):
                for gap in (0.25, 0.001):
                    roots = (
                        [-2, -1.5, -1, -0.5, 0, 0.5, 1 - gap, 1]
                        if spectrum == "distinct"
                        else [-1, -1, -1, -1, -1, -1, 1 - gap, 1]
                    )
                    for denominator in ("B_EQUALS_C", "BOUNDED_DIAGONAL"):
                        b = c.copy()
                        if denominator == "BOUNDED_DIAGONAL":
                            b = sym(
                                c + (0.1 * np.linalg.eigvalsh(c)[0] / (16 / 9)) * np.diag(shape)
                            )
                        lb = np.linalg.cholesky(b)
                        h = (v @ np.diag(roots)) @ v.T
                        s = sym((lb @ h) @ lb.T)
                        g, ds, db, dc = directions(b, c, w, es, eb, ec)
                        meta = {
                            "group": group,
                            "seed": seed,
                            "condition": condition,
                            "scale": scale,
                            "spectrum": spectrum,
                            "gap": gap,
                            "denominator": denominator,
                        }
                        yield (s, b, c, g, ds, db, dc), meta


def load_known(config):
    spec = config["known_generated_regression"]
    result_path = LEGACY / "result.json"
    arrays_path = LEGACY / "generated_arrays.npz"
    require(spec["result_path"] == str(result_path), "Unauthorized legacy result path")
    require(spec["arrays_path"] == str(arrays_path), "Unauthorized legacy array path")
    result_bytes = bound_bytes(result_path, spec["result_sha256"])
    array_bytes = bound_bytes(arrays_path, spec["arrays_sha256"])
    require(len(array_bytes) == spec["arrays_bytes"], "Legacy input size mismatch")
    receipt = json.loads(result_bytes)
    require(
        receipt["status"] == "GENERATED_CONDITIONING_VALIDITY_FAILURE", "Legacy failure required"
    )
    require(receipt["human_reads"] is False, "Only generated legacy inputs allowed")
    require(
        receipt["generated_arrays"]["sha256"] == spec["arrays_sha256"], "Legacy descriptor mismatch"
    )
    rng = np.random.default_rng(20260912)
    w, es, eb, ec = (probe(rng.normal(size=(8, 8))) for _ in range(4))
    with np.load(io.BytesIO(array_bytes), allow_pickle=False) as archive:
        row = 0
        for k in (3, 5):
            for condition in (1, 1000, 100000, 120000):
                prefix = f"k{k}_cond{condition}"
                c_array = archive[f"{prefix}_c"]
                require(
                    c_array.shape == (5, 12, 8, 8) and c_array.dtype == np.float64, "Legacy C shape"
                )
                for arm in ("UNIFORM_C2", "FIXED_BOUNDED_PAIR_LOGITS"):
                    s_array = archive[f"{prefix}_{arm}_s"]
                    require(
                        s_array.shape == c_array.shape and s_array.dtype == np.float64,
                        "Legacy S shape",
                    )
                    for band in range(5):
                        for label in range(12):
                            s, c = sym(s_array[band, label]), sym(c_array[band, label])
                            b = c.copy()
                            g, ds, db, dc = directions(b, c, w, es, eb, ec)
                            meta = {
                                "group": "known_regression",
                                "row": row,
                                "k": k,
                                "condition": condition,
                                "arm": arm,
                                "band": band,
                                "class": label,
                            }
                            yield (s, b, c, g, ds, db, dc), meta
                    row += 1


def negative_cases():
    s0 = np.diag([-2, -1.5, -1, -0.5, 0, 0.5, 0.75, 1.0])
    values = []
    for target in (0, 1, 2):
        case = [s0.copy(), np.eye(8), np.eye(8)]
        case[target][0, 1] += 0.001
        values.append(case)
    for target, index, value in (
        (0, 0, np.nan),
        (1, 0, 0),
        (1, 0, -1),
        (2, 0, 0),
        (0, 6, 1),
        (0, 6, 1 - 5e-12),
    ):
        case = [s0.copy(), np.eye(8), np.eye(8)]
        case[target][index, index] = value
        values.append(case)
    return tuple(np.stack([v[j] for v in values]) for j in range(3))


def make_inputs(config):
    cases = list(load_known(config))
    for seed, group in zip(config["new_fixtures"]["seeds"], config["new_fixtures"]["seed_roles"]):
        cases.extend(new_cases(seed, group))
    require(len(cases) == 1152, "Exact fixed fixture count required")
    inputs = {
        key: np.stack([c[0][j] for c in cases])
        for j, key in enumerate(("s", "b", "c", "g", "ds", "db", "dc"))
    }
    metadata = [dict(id=i, **case[1]) for i, case in enumerate(cases)]
    mask = np.zeros(1152, dtype=bool)
    mask[::60][:16] = True
    mask[851] = True
    mask[960:] = True
    require(int(mask.sum()) == 209, "Exact frozen gradient selection required")
    inputs["gradient_mask"] = mask
    for key, value in zip(("negative_s", "negative_b", "negative_c"), negative_cases()):
        inputs[key] = value
    return inputs, metadata


def versions():
    return {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "torch": torch.__version__,
        "mpmath": mpmath.__version__,
    }


def write_json(path, value):
    with path.open("x") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    path.chmod(0o400)


def write_arrays(path, values):
    with path.open("xb") as stream:
        np.savez(stream, **values)
        stream.flush()
        os.fsync(stream.fileno())
    path.chmod(0o400)


def descriptor(path):
    return {"path": str(path), "sha256": sha(path), "bytes": path.stat().st_size}


def measure(inputs):
    from cfeg.analysis.numerical_stability_operator import diagnostics, projector

    arrays, failures, diags, negative_errors = {}, [], {}, {}
    size = len(inputs["s"])
    for prefix, method in METHODS.items():
        for key in ("f", "gs", "gb", "gc", "batch_f"):
            arrays[f"{prefix}_{key}"] = np.full((size, 8, 8), np.nan)
        for key in ("ok", "gradient_ok"):
            arrays[f"{prefix}_{key}"] = np.zeros(size, dtype=bool)
        arrays[f"{prefix}_negative_rejected"] = np.zeros(9, dtype=bool)
        diags[prefix], negative_errors[prefix] = [], []
        for i in range(size):
            tensors = [torch.tensor(inputs[key][i], dtype=torch.float64) for key in ("s", "b", "c")]
            try:
                f = projector(*tensors, method)
                diagnostic = {key: float(value) for key, value in diagnostics(*tensors, f).items()}
                require(all(np.isfinite(v) for v in diagnostic.values()), "Nonfinite diagnostics")
                arrays[f"{prefix}_f"][i] = f.detach().numpy()
                arrays[f"{prefix}_ok"][i] = True
                diags[prefix].append(diagnostic)
            except (ValueError, RuntimeError) as error:
                failures.append(
                    {"method": prefix, "case": i, "stage": "scalar", "error": str(error)}
                )
                diags[prefix].append(None)
                continue
            if inputs["gradient_mask"][i]:
                try:
                    leaves = [x.requires_grad_() for x in tensors]
                    f = projector(*leaves, method)
                    loss = (torch.tensor(inputs["g"][i]) * f).sum()
                    gradients = torch.autograd.grad(loss, leaves)
                    require(all(torch.isfinite(g).all() for g in gradients), "Nonfinite VJP")
                    for name, value in zip(("gs", "gb", "gc"), gradients):
                        arrays[f"{prefix}_{name}"][i] = value.detach().numpy()
                    arrays[f"{prefix}_gradient_ok"][i] = True
                except (ValueError, RuntimeError) as error:
                    failures.append(
                        {"method": prefix, "case": i, "stage": "gradient", "error": str(error)}
                    )
            if (i + 1) % 192 == 0:
                print(f"{prefix}: {i + 1}/{size} scalar cases measured", flush=True)
        try:
            batch = [torch.tensor(inputs[key], dtype=torch.float64) for key in ("s", "b", "c")]
            arrays[f"{prefix}_batch_f"][:] = projector(*batch, method).detach().numpy()
        except (ValueError, RuntimeError) as error:
            failures.append({"method": prefix, "case": None, "stage": "batch", "error": str(error)})
        for i in range(9):
            try:
                bad = [
                    torch.tensor(inputs[key][i])
                    for key in ("negative_s", "negative_b", "negative_c")
                ]
                projector(*bad, method)
            except (ValueError, RuntimeError) as error:
                arrays[f"{prefix}_negative_rejected"][i] = True
                negative_errors[prefix].append(str(error))
            else:
                negative_errors[prefix].append(None)
                failures.append(
                    {
                        "method": prefix,
                        "case": i,
                        "stage": "negative",
                        "error": "NEGATIVE_CONTROL_ACCEPTED",
                    }
                )
    return arrays, failures, diags, negative_errors


def run(output):
    output = Path(output).absolute()
    require(
        output.resolve() == output and not output.exists() and output.parent.is_dir(),
        "Fresh unaliased output directory required",
    )
    require(sha(DESIGN) == DESIGN_SHA, "Frozen design changed")
    config = json.loads(DESIGN.read_text())
    require(
        all(
            os.environ.get(key) == "1"
            for key in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS")
        ),
        "Launch with all BLAS/OpenMP thread environment limits set to1 before imports",
    )
    for args in (("diff", "--quiet"), ("diff", "--cached", "--quiet")):
        subprocess.run(["git", *args], cwd=ROOT, check=True)
    pins = {name: sha(ROOT / name) for name in PIN_FILES}
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    output.mkdir(mode=0o700)
    started, cpu_started = time.monotonic(), time.process_time()
    start = {
        "schema": "cfeg.numerical_stability.start.v1",
        "design_path": str(DESIGN),
        "design_sha256": DESIGN_SHA,
        "code_pins": pins,
        "source_revision": revision,
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "versions": versions(),
        "budget": config["budget"],
        "exclusions": EXCLUSIONS,
    }
    write_json(output / "start.json", start)

    def expired(*_):
        raise TimeoutError("Frozen primary numerical walltime cap900s reached")

    previous = signal.signal(signal.SIGALRM, expired)
    signal.alarm(900)
    try:
        inputs, metadata = make_inputs(config)
        write_arrays(output / "inputs.npz", inputs)
        arrays, failures, diagnostics, errors = measure(inputs)
        write_arrays(output / "measurements.npz", arrays)
        bytes_before = sum(p.stat().st_size for p in output.iterdir())
        require(bytes_before < 128 * 1024**2, "Numerical output reservation exceeded")
        result = {
            "schema": "cfeg.numerical_stability.result.v1",
            "status": "MEASUREMENT_FAILURE" if failures else "MEASUREMENTS_COMPLETE",
            "start_sha256": sha(output / "start.json"),
            "inputs": descriptor(output / "inputs.npz"),
            "measurements": descriptor(output / "measurements.npz"),
            "case_metadata": metadata,
            "method_failures": failures,
            "diagnostics": diagnostics,
            "negative_errors": errors,
            "numeric_wall_seconds": time.monotonic() - started,
            "numeric_cpu_seconds": time.process_time() - cpu_started,
            "output_bytes_before_result": bytes_before,
            "versions": versions(),
            "exclusions": EXCLUSIONS,
        }
        signal.alarm(0)
        write_json(output / "result.json", result)
        return result
    except BaseException as error:
        signal.alarm(0)
        write_json(
            output / "failure.json",
            {
                "status": "NUMERICAL_MEASUREMENT_INTERRUPTED",
                "error_type": type(error).__name__,
                "error": str(error),
                "elapsed_seconds": time.monotonic() - started,
                "cpu_seconds": time.process_time() - cpu_started,
                "start_sha256": sha(output / "start.json"),
                "exclusions": EXCLUSIONS,
            },
        )
        raise
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, previous)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    result = run(args.output)
    print(
        json.dumps(
            {
                "status": result["status"],
                "method_failures": len(result["method_failures"]),
                "elapsed_seconds": result["numeric_wall_seconds"],
            }
        )
    )
    return 0 if result["status"] == "MEASUREMENTS_COMPLETE" else 2


if __name__ == "__main__":
    raise SystemExit(main())
