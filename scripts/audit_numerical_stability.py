"""Independent generated-only numerical study audit; no producer/cfeg imports.

Registered fixtures and legacy arrays are accessed only by the explicit CLI.
Reference arithmetic solves the actual saved binary64 problem at 80 digits.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import os
import signal
import stat
import time
import zipfile
from datetime import datetime
from pathlib import Path

import mpmath as mp
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DESIGN_PATH = "configs/analysis/numerical_stability_v1.json"
DESIGN_SHA = "04d0968d478c795e8a0860c561fcfb881b9d510b5777c1a7ff45e65764b6fbe8"
METHODS = {"n1": "N1_SYM_CHOLESKY", "n2": "N2_GVD_IMPLICIT"}
COUNT = 1152
OLD_COUNT = 960
GRADIENT_INDICES = tuple(sorted((*range(0, 960, 60), 851, *range(960, 1152))))
RECHECK_INDICES = (851, 960, 1055, 1056, 1151)
MAX_BYTES = 128 * 1024**2
MAX_SECONDS = 7200
OLD_RESULT = Path("/home/whwovy/task-trca-pair-s-conditioning-Vqli4o/primary1/result.json")
OLD_RESULT_SHA = "98ea5607b72d47174c2f0e3c1569bf9a872bca7230c160f29554b8b9cd841ee2"
OLD_ARRAYS = OLD_RESULT.parent / "generated_arrays.npz"
OLD_ARRAYS_SHA = "01ad4233bba861f32f0a4ca5f68e1f1af883ffc5d51c55fa1b04cbb423421211"
OLD_ARRAYS_BYTES = 22286792
DIRECTIONS = ("S", "B", "C", "joint")
PIN_FILES = (
    DESIGN_PATH,
    "docs/numerical_stability_v1_design.md",
    "docs/numerical_stability_v1_implementation.md",
    "scripts/run_numerical_stability.py",
    "scripts/audit_numerical_stability.py",
    "src/cfeg/analysis/numerical_stability_operator.py",
    "src/cfeg/analysis/task_trca_shape_operator.py",
    "src/cfeg/__init__.py",
    "src/cfeg/analysis/__init__.py",
)
EXCLUSIONS = {
    "human_reads": False,
    "query_reads_or_scores": 0,
    "optimizer_updates": 0,
    "gpu_used": False,
    "held60": False,
    "external_outreach": False,
    "paid_resources": False,
}
INPUT_NAMES = ("s", "b", "c", "g", "ds", "db", "dc")
DIAGNOSTIC_KEYS = {"lambda", "residual", "normalization_error", "symmetry_error", "raw_h_skew"}


class ReferenceFailure(RuntimeError):
    """An unavailable/inconsistent oracle cannot promote or reject a method."""


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(data):
    return hashlib.sha256(data).hexdigest()


def valid_pin(value):
    return (
        isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value)
    )


def file_identity(info):
    return (
        info.st_dev,
        info.st_ino,
        info.st_mode,
        info.st_nlink,
        info.st_size,
        info.st_mtime_ns,
        info.st_ctime_ns,
    )


def read_pinned(path, pin, *, immutable=True, size=None):
    path = Path(path)
    require(path.is_absolute() and path.resolve() == path, "Unaliased absolute input required")
    require(valid_pin(pin), "Invalid external hash")
    with os.fdopen(os.open(path, os.O_RDONLY | os.O_NOFOLLOW), "rb") as stream:
        info = os.fstat(stream.fileno())
        require(
            stat.S_ISREG(info.st_mode) and info.st_nlink == 1, "Regular nonalias input required"
        )
        require(not immutable or stat.S_IMODE(info.st_mode) == 0o400, "Input mode must be 0400")
        require(
            info.st_size <= MAX_BYTES and (size is None or info.st_size == size), "Input byte bound"
        )
        data = stream.read(MAX_BYTES + 1)
        final = os.fstat(stream.fileno())
        require(
            file_identity(info) == file_identity(final) and len(data) == info.st_size,
            "Input changed while reading",
        )
    require(
        path.resolve() == path and file_identity(path.stat()) == file_identity(info),
        "Input alias changed while reading",
    )
    require(digest(data) == pin, f"Hash mismatch: {path.name}")
    return data


def decode_json(data):
    def pairs(items):
        result = {}
        for key, value in items:
            require(key not in result, "Duplicate JSON key")
            result[key] = value
        return result

    def invalid(value):
        raise ValueError(f"Nonfinite JSON constant: {value}")

    value = json.loads(data, object_pairs_hook=pairs, parse_constant=invalid)
    require(isinstance(value, dict), "JSON object required")
    return value


def same_json(actual, expected, message):
    require(
        json.dumps(actual, sort_keys=True, allow_nan=False)
        == json.dumps(expected, sort_keys=True, allow_nan=False),
        message,
    )


def decode_npz(data, shapes, *, selected=False):
    """Validate headers/inventory before decoding only the named numeric members."""
    with zipfile.ZipFile(io.BytesIO(data)) as archive:
        items = archive.infolist()
        names = [item.filename for item in items]
        require(len(names) == len(set(names)), "Duplicate NPZ member")
        expected = {name + ".npy" for name in shapes}
        require(expected.issubset(names) if selected else expected == set(names), "NPZ inventory")
        require(sum(item.file_size for item in items) <= MAX_BYTES, "Expanded NPZ byte bound")
        for item in items:
            require(not item.is_dir() and not item.flag_bits & 1, "Unsupported NPZ member")
            if item.filename not in expected:
                continue
            shape, dtype = shapes[item.filename[:-4]]
            with archive.open(item) as stream:
                require(np.lib.format.read_magic(stream) == (1, 0), "NPY header version")
                actual, fortran, actual_dtype = np.lib.format.read_array_header_1_0(stream)
                require(actual == shape and actual_dtype == np.dtype(dtype), "NPY shape/dtype")
                require(not fortran, "NPY Fortran order is not canonical")
                require(
                    item.file_size - stream.tell() == math.prod(shape) * actual_dtype.itemsize,
                    "NPY size/trailing payload",
                )
    with np.load(io.BytesIO(data), allow_pickle=False) as archive:
        return {name: archive[name] for name in shapes}


def sym(value):
    return (value + value.T) / 2


def canonical_qr(raw):
    q, r = np.linalg.qr(raw)
    return q * np.where(np.diag(r) < 0, -1.0, 1.0)


def normalized(raw):
    value = sym(raw)
    return value / np.linalg.norm(value, "fro")


def probes(s, b, c, draws):
    w, es, eb, ec = draws
    lb, lc = np.linalg.cholesky(b), np.linalg.cholesky(c)
    return {
        "s": s,
        "b": b,
        "c": c,
        "g": sym(lc @ w @ lc.T),
        "ds": sym(lb @ es @ lb.T),
        "db": sym(lb @ eb @ lb.T),
        "dc": sym(lc @ ec @ lc.T),
    }


def new_cases(seed):
    """Literal recipe; called on registered seeds only during the actual audit."""
    rng = np.random.default_rng(seed)
    u, v = (canonical_qr(rng.normal(size=(8, 8))) for _ in range(2))
    draws = tuple(normalized(rng.normal(size=(8, 8))) for _ in range(4))
    logits = np.linspace(-math.log(2) / 2, math.log(2) / 2, 8)
    exp = np.exp(logits - logits.max())
    r = 8 * exp / exp.sum()
    for condition in (1, 1000, 120000, 1000000):
        for scale in (2.0**-20, 1.0, 2.0**20):
            c = scale * (
                np.eye(8)
                if condition == 1
                else sym((u @ np.diag(condition ** (-np.arange(8) / 7))) @ u.T)
            )
            for spectrum in ("distinct", "repeated"):
                for gap in (0.25, 0.001):
                    roots = np.array(
                        ([-2, -1.5, -1, -0.5, 0, 0.5] if spectrum == "distinct" else [-1] * 6)
                        + [1 - gap, 1]
                    )
                    for denominator in ("B_EQUALS_C", "BOUNDED_DIAGONAL"):
                        b = (
                            c.copy()
                            if denominator == "B_EQUALS_C"
                            else sym(c + (0.1 * np.linalg.eigvalsh(c)[0] / (16 / 9)) * np.diag(r))
                        )
                        lb = np.linalg.cholesky(b)
                        h = (v @ np.diag(roots)) @ v.T
                        s = sym((lb @ h) @ lb.T)
                        yield (condition, scale, spectrum, gap, denominator), probes(s, b, c, draws)


def negative_cases():
    s = np.diag([-2, -1.5, -1, -0.5, 0, 0.5, 0.75, 1.0])
    arrays = {name: np.tile(s if name == "s" else np.eye(8), (9, 1, 1)) for name in ("s", "b", "c")}
    for index, name in enumerate(("s", "b", "c")):
        arrays[name][index, 0, 1] += 0.001
    arrays["s"][3, 0, 0] = np.nan
    arrays["b"][4, 0, 0] = 0
    arrays["b"][5, 0, 0] = -1
    arrays["c"][6, 0, 0] = 0
    arrays["s"][7, -2, -2] = 1
    arrays["s"][8, -2, -2] = 1 - 5e-12
    return {"negative_" + name: value for name, value in arrays.items()}


def mp_matrix(value):
    array = np.asarray(value)
    require(array.shape == (8, 8) and np.isfinite(array).all(), "Reference matrix shape/finite")
    return mp.matrix(
        [
            [mp.mpf(int(x.as_integer_ratio()[0])) / int(x.as_integer_ratio()[1]) for x in row]
            for row in array.astype(float)
        ]
    )


def fro(value):
    return mp.sqrt(sum(x * x for x in value))


def trace(value):
    return sum(value[i, i] for i in range(value.rows))


def solve_reference(s, b, c):
    """Spectral B whitening, independent of both producer formulations."""
    values, vectors = mp.eigsy(b)
    if values[0] <= 0:
        raise ReferenceFailure("Reference denominator is not SPD")
    whitening = vectors * mp.diag([1 / mp.sqrt(x) for x in values]) * vectors.T
    h = whitening * s * whitening
    roots, basis = mp.eigsy((h + h.T) / 2)
    if roots[7] - roots[6] <= mp.mpf("1e-10") * max(1, max(abs(x) for x in roots)):
        raise ReferenceFailure("Reference top root is not admissible")
    z = whitening * basis[:, 7]
    norm = (z.T * c * z)[0]
    if norm <= 0:
        raise ReferenceFailure("Reference C normalization is not positive")
    return roots[7], (z * z.T) / norm


def reference(value, *, precision=80):
    with mp.workdps(precision):
        mp.cholesky(mp_matrix(value["c"]))
        return solve_reference(*(mp_matrix(value[name]) for name in ("s", "b", "c")))


def directional_references(value):
    with mp.workdps(80):
        s, b, c, g, ds, db, dc = (
            mp_matrix(value[name]) for name in ("s", "b", "c", "g", "ds", "db", "dc")
        )
        result = {}
        for direction in DIRECTIONS:
            answers = []
            for step in (mp.mpf("1e-20"), mp.mpf("1e-25")):
                losses = []
                for sign in (-1, 1):
                    delta = sign * step
                    matrices = (
                        s + delta * ds if direction in ("S", "joint") else s,
                        b + delta * db if direction in ("B", "joint") else b,
                        c + delta * dc if direction in ("C", "joint") else c,
                    )
                    _, f = solve_reference(*matrices)
                    losses.append(trace(g.T * f))
                answers.append((losses[1] - losses[0]) / (2 * step))
            error = abs(answers[0] - answers[1])
            limit = mp.mpf("1e-20") * max(1, abs(answers[1]))
            if error > limit:
                raise ReferenceFailure("High-precision finite differences inconsistent")
            result[direction] = {
                "value": answers[1],
                "fd_difference": error,
                "h1": answers[0],
                "h2": answers[1],
            }
        return result


def forward_metrics(value, f_array, ref):
    with mp.workdps(80):
        s, b, c, f = (mp_matrix(x) for x in (value["s"], value["b"], value["c"], f_array))
        denominator = trace(b * f)
        require(denominator > 0 and fro(f) > 0, "Invalid zero/nonpositive projector")
        eigenvalue = trace(s * f) / denominator
        lc = mp.cholesky(c)
        scale = (fro(s) + abs(eigenvalue) * fro(b)) * fro(f)
        require(scale > 0, "Invalid residual denominator")
        metrics = {
            "residual": fro(s * f - eigenvalue * b * f) / scale,
            "normalization_error": abs(trace(c * f) - 1),
            "symmetry_error": fro(f - f.T) / max(fro(f), mp.mpf(np.finfo(float).tiny)),
            "top_eigenvalue_error": abs(eigenvalue - ref[0]) / max(1, abs(ref[0])),
            "projector_error": fro(lc.T * (f - ref[1]) * lc),
        }
        limits = {
            "residual": 1e-12,
            "normalization_error": 1e-10,
            "symmetry_error": 1e-12,
            "top_eigenvalue_error": 1e-8,
            "projector_error": 1e-6,
        }
        failures = [key for key, number in metrics.items() if number > limits[key]]
        return {key: float(number) for key, number in metrics.items()}, failures


def gradient_metrics(value, gradients, refs):
    with mp.workdps(80):
        products = {
            name: trace(
                mp_matrix(gradients["g" + name.lower()]).T * mp_matrix(value["d" + name.lower()])
            )
            for name in ("S", "B", "C")
        }
        products["joint"] = sum(products.values())
        result, failures = {}, []
        for direction, actual in products.items():
            expected = refs[direction]["value"]
            error = abs(actual - expected)
            limit = mp.mpf("1e-7") + mp.mpf("1e-4") * max(abs(actual), abs(expected))
            result[direction] = {
                "actual": float(actual),
                "reference": float(expected),
                "absolute_error": float(error),
                "limit": float(limit),
                "fd_difference": str(refs[direction]["fd_difference"]),
                "reference_h1": str(refs[direction]["h1"]),
                "reference_h2": str(refs[direction]["h2"]),
            }
            if error > limit:
                failures.append(direction)
        return result, failures


def exact_keys(value, keys, message):
    require(isinstance(value, dict) and set(value) == set(keys), message)


def finite_number(value):
    return type(value) in (int, float) and math.isfinite(value)


def load_envelope(result_path, external_pin):
    """Bind every path, mode, hash and schema before any numeric member is decoded."""
    result_path = Path(result_path)
    require(
        result_path.is_absolute() and result_path.resolve() == result_path,
        "Unaliased absolute result required",
    )
    require(result_path.name == "result.json", "Canonical result filename required")
    parent = result_path.parent
    require(
        not (parent / "failure.json").exists() and not (parent / "failure.json").is_symlink(),
        "failure.json dominates result",
    )
    inventory = {"start.json", "inputs.npz", "measurements.npz", "result.json"}
    require({p.name for p in parent.iterdir()} == inventory, "Generated archive inventory")
    for name in inventory:
        path = parent / name
        info = path.lstat()
        require(
            path.resolve() == path
            and stat.S_ISREG(info.st_mode)
            and stat.S_IMODE(info.st_mode) == 0o400,
            "Archive must be unaliased and 0400",
        )
    result_data = read_pinned(result_path, external_pin)
    result = decode_json(result_data)
    exact_keys(
        result,
        (
            "schema",
            "status",
            "start_sha256",
            "inputs",
            "measurements",
            "case_metadata",
            "method_failures",
            "diagnostics",
            "negative_errors",
            "numeric_wall_seconds",
            "numeric_cpu_seconds",
            "output_bytes_before_result",
            "versions",
            "exclusions",
        ),
        "Result schema keys",
    )
    require(result["schema"] == "cfeg.numerical_stability.result.v1", "Result schema")
    for name in ("inputs", "measurements"):
        desc = result[name]
        exact_keys(desc, ("path", "sha256", "bytes"), "Descriptor keys")
        require(desc["path"] == str(parent / (name + ".npz")), "Unauthorized descriptor path")
        require(
            valid_pin(desc["sha256"])
            and type(desc["bytes"]) is int
            and 0 < desc["bytes"] <= MAX_BYTES,
            "Descriptor hash/bytes",
        )
    start_data = read_pinned(parent / "start.json", result["start_sha256"])
    start = decode_json(start_data)
    exact_keys(
        start,
        (
            "schema",
            "design_path",
            "design_sha256",
            "code_pins",
            "source_revision",
            "started_utc",
            "versions",
            "budget",
            "exclusions",
        ),
        "Start schema keys",
    )
    require(start["schema"] == "cfeg.numerical_stability.start.v1", "Start schema")
    require(
        start["design_path"] == str(ROOT / DESIGN_PATH) and start["design_sha256"] == DESIGN_SHA,
        "Frozen design binding",
    )
    config = decode_json(read_pinned(ROOT / DESIGN_PATH, DESIGN_SHA, immutable=False))
    same_json(start["budget"], config["budget"], "Frozen budget mismatch")
    exact_keys(start["code_pins"], PIN_FILES, "Exact code pin inventory")
    for name, pin in start["code_pins"].items():
        read_pinned(ROOT / name, pin, immutable=False)
    require(start["code_pins"][DESIGN_PATH] == DESIGN_SHA, "Design code pin mismatch")
    require(
        isinstance(start["source_revision"], str)
        and len(start["source_revision"]) == 40
        and all(c in "0123456789abcdef" for c in start["source_revision"]),
        "Source revision",
    )
    require(isinstance(start["started_utc"], str), "Start time type")
    require(
        datetime.fromisoformat(start["started_utc"]).utcoffset().total_seconds() == 0,
        "UTC start time required",
    )
    exact_keys(start["versions"], ("python", "numpy", "scipy", "torch", "mpmath"), "Version keys")
    require(all(isinstance(v, str) and v for v in start["versions"].values()), "Version values")
    require(start["versions"]["mpmath"] == mp.__version__ == "1.3.0", "Reference version pin")
    require(start["versions"]["numpy"] == np.__version__, "RNG/reconstruction NumPy version")
    same_json(result["versions"], start["versions"], "Version envelope mismatch")
    same_json(start["exclusions"], EXCLUSIONS, "Start exclusions")
    same_json(result["exclusions"], EXCLUSIONS, "Result exclusions")
    for name in ("numeric_wall_seconds", "numeric_cpu_seconds"):
        require(finite_number(result[name]) and 0 <= result[name] < MAX_SECONDS, "Producer time")
    require(type(result["output_bytes_before_result"]) is int, "Archive byte count type")
    archive_bytes = sum((parent / name).stat().st_size for name in inventory)
    require(archive_bytes < MAX_BYTES, "Archive output reservation")
    require(
        result["output_bytes_before_result"] == archive_bytes - len(result_data),
        "Before-result byte count mismatch",
    )
    payloads = {
        name: read_pinned(
            parent / (name + ".npz"), result[name]["sha256"], size=result[name]["bytes"]
        )
        for name in ("inputs", "measurements")
    }
    return result, start, payloads, archive_bytes


def archive_shapes():
    inputs = {name: ((COUNT, 8, 8), "float64") for name in INPUT_NAMES}
    inputs["gradient_mask"] = ((COUNT,), "bool")
    inputs.update({"negative_" + name: ((9, 8, 8), "float64") for name in ("s", "b", "c")})
    measurements = {}
    for method in METHODS:
        measurements.update(
            {
                method + "_" + name: ((COUNT, 8, 8), "float64")
                for name in ("f", "gs", "gb", "gc", "batch_f")
            }
        )
        measurements.update(
            {method + "_" + name: ((COUNT,), "bool") for name in ("ok", "gradient_ok")}
        )
        measurements[method + "_negative_rejected"] = ((9,), "bool")
    return inputs, measurements


def expected_inputs():
    """Registered reconstruction: never invoked by toy tests or at import time."""
    read_pinned(OLD_RESULT, OLD_RESULT_SHA)
    shapes = {}
    for k in (3, 5):
        for condition in (1, 1000, 100000, 120000):
            stem = f"k{k}_cond{condition}"
            for suffix in ("c", "UNIFORM_C2_s", "FIXED_BOUNDED_PAIR_LOGITS_s"):
                shapes[stem + "_" + suffix] = ((5, 12, 8, 8), "float64")
    old = decode_npz(
        read_pinned(OLD_ARRAYS, OLD_ARRAYS_SHA, size=OLD_ARRAYS_BYTES), shapes, selected=True
    )
    rng = np.random.default_rng(20260912)
    draws = tuple(normalized(rng.normal(size=(8, 8))) for _ in range(4))
    values, metadata = [], []
    row = 0
    for k in (3, 5):
        for condition in (1, 1000, 100000, 120000):
            stem = f"k{k}_cond{condition}"
            for arm in ("UNIFORM_C2", "FIXED_BOUNDED_PAIR_LOGITS"):
                for band in range(5):
                    for klass in range(12):
                        c = sym(old[stem + "_c"][band, klass])
                        s = sym(old[stem + "_" + arm + "_s"][band, klass])
                        metadata.append(
                            dict(
                                id=len(values),
                                group="known_regression",
                                row=row,
                                k=k,
                                condition=condition,
                                arm=arm,
                                band=band,
                                **{"class": klass},
                            )
                        )
                        values.append(probes(s, c.copy(), c, draws))
                row += 1
    for seed, group in (
        (20260910, "registered_development"),
        (20260911, "registered_unexposed_validation"),
    ):
        for fields, value in new_cases(seed):
            condition, scale, spectrum, gap, denominator = fields
            metadata.append(
                {
                    "id": len(values),
                    "group": group,
                    "seed": seed,
                    "condition": condition,
                    "scale": scale,
                    "spectrum": spectrum,
                    "gap": gap,
                    "denominator": denominator,
                }
            )
            values.append(value)
    require(len(values) == COUNT, "Independent reconstruction case count")
    result = {name: np.stack([value[name] for value in values]) for name in INPUT_NAMES}
    mask = np.zeros(COUNT, dtype=bool)
    mask[list(GRADIENT_INDICES)] = True
    result["gradient_mask"] = mask
    result.update(negative_cases())
    return result, metadata


def validate_reconstruction(inputs, metadata, expected, expected_metadata):
    same_json(metadata, expected_metadata, "Exact fixture metadata/order")
    require(
        np.array_equal(inputs["gradient_mask"], expected["gradient_mask"]),
        "Exact 209-case gradient mask",
    )
    for name in INPUT_NAMES:
        actual, target = inputs[name], expected[name]
        require(np.isfinite(actual).all() and np.isfinite(target).all(), "Finite fixture required")
        scales = np.maximum(np.linalg.norm(target, axis=(-2, -1)), np.finfo(float).tiny)
        errors = np.linalg.norm(actual - target, axis=(-2, -1)) / scales
        require(np.all(errors <= 1e-13), f"Fixture reconstruction mismatch: {name}")
        require(np.array_equal(actual, actual.swapaxes(-1, -2)), "Stored exact symmetry")
    for name in ("negative_s", "negative_b", "negative_c"):
        require(
            np.array_equal(inputs[name], expected[name], equal_nan=True), "Negative fixture recipe"
        )


def validate_measurements(result, inputs, arrays):
    """Failed methods are valid observations; contradictory archives are not."""
    failures = result["method_failures"]
    require(isinstance(failures, list), "Failure list")
    require(
        result["status"] == ("MEASUREMENT_FAILURE" if failures else "MEASUREMENTS_COMPLETE"),
        "Failure/status coherence",
    )
    recorded = set()
    for item in failures:
        exact_keys(item, ("method", "case", "stage", "error"), "Failure entry schema")
        method, case, stage = item["method"], item["case"], item["stage"]
        require(
            method in METHODS and stage in ("scalar", "gradient", "batch", "negative"),
            "Failure method/stage",
        )
        require(
            (
                case is None
                if stage == "batch"
                else type(case) is int and 0 <= case < (9 if stage == "negative" else COUNT)
            ),
            "Failure case",
        )
        require(isinstance(item["error"], str) and item["error"], "Failure error string")
        require(
            stage != "negative" or item["error"] == "NEGATIVE_CONTROL_ACCEPTED",
            "Negative failure message",
        )
        key = (method, stage, case)
        require(key not in recorded, "Duplicate failure")
        recorded.add(key)
    exact_keys(result["diagnostics"], METHODS, "Diagnostic methods")
    exact_keys(result["negative_errors"], METHODS, "Negative-error methods")
    expected = set()
    for method in METHODS:
        ok, grad_ok = arrays[method + "_ok"], arrays[method + "_gradient_ok"]
        require(
            not np.any(grad_ok & (~ok | ~inputs["gradient_mask"])), "Invalid gradient success mask"
        )
        diags, errors = result["diagnostics"][method], result["negative_errors"][method]
        require(isinstance(diags, list) and len(diags) == COUNT, "Diagnostic case coverage")
        require(isinstance(errors, list) and len(errors) == 9, "Negative-error coverage")
        for i in range(COUNT):
            f = arrays[method + "_f"][i]
            if ok[i]:
                require(np.isfinite(f).all(), "Successful F must be finite")
                exact_keys(diags[i], DIAGNOSTIC_KEYS, "Diagnostic fields")
                require(
                    all(type(x) is float and math.isfinite(x) for x in diags[i].values()),
                    "Diagnostic finite float values",
                )
                if inputs["gradient_mask"][i] and not grad_ok[i]:
                    expected.add((method, "gradient", i))
            else:
                expected.add((method, "scalar", i))
                require(np.isnan(f).all() and diags[i] is None, "Failed scalar sentinel")
            for name in ("gs", "gb", "gc"):
                value = arrays[method + "_" + name][i]
                require(
                    np.isfinite(value).all() if grad_ok[i] else np.isnan(value).all(),
                    "Gradient finite/NaN sentinel",
                )
        batch = arrays[method + "_batch_f"]
        if np.isnan(batch).all():
            expected.add((method, "batch", None))
        else:
            require(np.isfinite(batch).all(), "Whole batch only; no partial failed rows")
        for i, rejected in enumerate(arrays[method + "_negative_rejected"]):
            require(
                isinstance(errors[i], str) and bool(errors[i]) if rejected else errors[i] is None,
                "Negative error/mask coherence",
            )
            if not rejected:
                expected.add((method, "negative", i))
    require(recorded == expected, "Failure logs disagree with exact masks/sentinels")


def precision_recheck(value, ref):
    higher = reference(value, precision=120)
    with mp.workdps(120):
        errors = {
            "eigenvalue": abs(ref[0] - higher[0]) / max(1, abs(higher[0])),
            "projector": fro(ref[1] - higher[1]) / max(1, fro(higher[1])),
        }
        if any(error > mp.mpf("1e-50") for error in errors.values()):
            raise ReferenceFailure("80/120 digit reference disagreement")
        return {key: str(error) for key, error in errors.items()}


def inspect_numerics(inputs, arrays, result):
    reports = {
        method: {
            "method": name,
            "eligible": True,
            "failures": [],
            "cases": [],
            "gradients": [],
            "forward_maxima": {},
            "gradient_maxima": {},
        }
        for method, name in METHODS.items()
    }
    for item in result["method_failures"]:
        reports[item["method"]]["failures"].append({"gate": "producer_failure", **item})
    rechecks, coverage, fd_max = {}, {}, mp.mpf(0)
    for i in range(COUNT):
        value = {name: inputs[name][i] for name in INPUT_NAMES}
        try:
            ref = reference(value)
            if i in RECHECK_INDICES:
                rechecks[str(i)] = precision_recheck(value, ref)
            directions = directional_references(value) if inputs["gradient_mask"][i] else None
        except (ValueError, ZeroDivisionError, ArithmeticError) as error:
            raise ReferenceFailure(f"Reference unavailable at case {i}: {error}") from error
        if directions is not None:
            fd_max = max(fd_max, *(item["fd_difference"] for item in directions.values()))
            metadata = result["case_metadata"][i]
            if metadata["group"] != "known_regression":
                key = f"{metadata['seed']}:{metadata['denominator']}"
                maxima = coverage.setdefault(key, {"S": 0.0, "B": 0.0})
                for direction in ("S", "B"):
                    maxima[direction] = max(
                        maxima[direction], float(abs(directions[direction]["value"]))
                    )
        for method, report in reports.items():
            row = {"id": i, "scalar": None, "batch": None, "scalar_batch_error": None}
            for kind, name in (("scalar", "f"), ("batch", "batch_f")):
                f = arrays[method + "_" + name][i]
                if np.isnan(f).all():
                    continue
                try:
                    metrics, failures = forward_metrics(value, f, ref)
                except (ValueError, ArithmeticError) as error:
                    report["failures"].append(
                        {"case": i, "gate": kind + "_invalid_F", "error": str(error)}
                    )
                    continue
                row[kind] = metrics
                for gate, number in metrics.items():
                    key = kind + ":" + gate
                    report["forward_maxima"][key] = max(
                        report["forward_maxima"].get(key, 0), number
                    )
                report["failures"].extend(
                    {"case": i, "gate": kind + ":" + gate} for gate in failures
                )
            if row["scalar"] is not None and row["batch"] is not None:
                scalar, batch = arrays[method + "_f"][i], arrays[method + "_batch_f"][i]
                error = np.linalg.norm(scalar - batch, "fro") / max(
                    np.linalg.norm(scalar, "fro"), np.finfo(float).tiny
                )
                row["scalar_batch_error"] = float(error)
                report["forward_maxima"]["scalar_batch_error"] = max(
                    report["forward_maxima"].get("scalar_batch_error", 0), float(error)
                )
                if not math.isfinite(error) or error > 1e-8:
                    report["failures"].append({"case": i, "gate": "scalar_batch_error"})
            report["cases"].append(row)
            if directions is not None:
                grad = {
                    "id": i,
                    "directions": None,
                    "reference": {
                        d: {k: mp.nstr(v, 80) for k, v in item.items()}
                        for d, item in directions.items()
                    },
                }
                if arrays[method + "_gradient_ok"][i]:
                    gradients = {
                        name: arrays[method + "_" + name][i] for name in ("gs", "gb", "gc")
                    }
                    metrics, failures = gradient_metrics(value, gradients, directions)
                    grad["directions"] = metrics
                    for direction, metric in metrics.items():
                        report["gradient_maxima"][direction] = max(
                            report["gradient_maxima"].get(direction, 0), metric["absolute_error"]
                        )
                    report["failures"].extend(
                        {"case": i, "gate": "gradient:" + d} for d in failures
                    )
                report["gradients"].append(grad)
        if (i + 1) % 96 == 0:
            print(f"Independent audit: {i + 1}/{COUNT} cases", flush=True)
    expected_coverage = {
        f"{seed}:{denominator}"
        for seed in (20260910, 20260911)
        for denominator in ("B_EQUALS_C", "BOUNDED_DIAGONAL")
    }
    if set(coverage) != expected_coverage or any(
        value <= 1e-4 for maxima in coverage.values() for value in maxima.values()
    ):
        raise ReferenceFailure("Insufficient nonzero S/B reference-gradient coverage")
    require(set(rechecks) == {str(i) for i in RECHECK_INDICES}, "Precision recheck coverage")
    for report in reports.values():
        report["eligible"] = not report["failures"]
        report["verdict"] = "ELIGIBLE" if report["eligible"] else "REJECTED_FIXED_STUDY"
        require(
            len(report["cases"]) == COUNT and len(report["gradients"]) == len(GRADIENT_INDICES),
            "Complete metric coverage",
        )
    recommended = next((METHODS[m] for m in METHODS if reports[m]["eligible"]), None)
    return {
        "methods": reports,
        "recommended_method": recommended,
        "reference": {
            "backend": "mpmath spectral whitening/eigsy",
            "digits": 80,
            "forward_cases": COUNT,
            "gradient_cases": len(GRADIENT_INDICES),
            "directions": list(DIRECTIONS),
            "fd_steps": ["1e-20", "1e-25"],
            "fd_max_absolute_difference": mp.nstr(fd_max, 80),
            "precision_recheck_digits": 120,
            "precision_rechecks": rechecks,
            "nonzero_reference_coverage": coverage,
        },
    }


def audit(result_path, external_pin):
    require(mp.__version__ == "1.3.0", "mpmath 1.3.0 required")
    result, start, payloads, archive_bytes = load_envelope(result_path, external_pin)
    input_shapes, measurement_shapes = archive_shapes()
    inputs = decode_npz(payloads["inputs"], input_shapes)
    measurements = decode_npz(payloads["measurements"], measurement_shapes)
    expected, metadata = expected_inputs()
    validate_reconstruction(inputs, result["case_metadata"], expected, metadata)
    validate_measurements(result, inputs, measurements)
    output = inspect_numerics(inputs, measurements, result)
    output.update(
        status="NUMERICAL_STUDY_VERIFIED",
        source_result_sha256=external_pin,
        source_result_path=str(result_path),
        start_sha256=result["start_sha256"],
        inputs=result["inputs"],
        measurements=result["measurements"],
        code_pins=start["code_pins"],
        source_revision=start["source_revision"],
        design_sha256=DESIGN_SHA,
        archive_bytes=archive_bytes,
        producer_numeric_wall_seconds=result["numeric_wall_seconds"],
        producer_numeric_cpu_seconds=result["numeric_cpu_seconds"],
        producer_versions=result["versions"],
        producer_diagnostics=result["diagnostics"],
        negative_rejections={m: measurements[m + "_negative_rejected"].tolist() for m in METHODS},
        negative_errors=result["negative_errors"],
    )
    return output


def write_receipt(path, receipt):
    receipt["receipt_bytes"] = 0
    for _ in range(10):
        data = (json.dumps(receipt, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()
        if receipt["receipt_bytes"] == len(data):
            break
        receipt["receipt_bytes"] = len(data)
    require(len(data) + receipt.get("archive_bytes", 0) < MAX_BYTES, "Combined output reservation")
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o400)
    with os.fdopen(fd, "wb") as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result", required=True, type=Path)
    parser.add_argument("--sha256", required=True)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    output = args.output
    require(
        output.is_absolute()
        and output.resolve() == output
        and output.parent.is_dir()
        and not output.exists()
        and not output.is_symlink(),
        "Fresh unaliased absolute receipt",
    )
    require(output.parent != args.result.parent, "Receipt must not mutate source archive inventory")
    started, cpu_started = time.monotonic(), time.process_time()
    receipt = {"recommended_method": None}

    def expired(*_):
        raise ReferenceFailure("Remaining numerical walltime budget exhausted")

    previous = signal.signal(signal.SIGALRM, expired)
    signal.alarm(MAX_SECONDS)
    try:
        require(
            all(
                os.environ.get(name) == "1"
                for name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS")
            ),
            "Set OPENBLAS/OMP/MKL_NUM_THREADS=1 before process imports",
        )
        result, _, _, _ = load_envelope(args.result, args.sha256)
        remaining = MAX_SECONDS - result["numeric_wall_seconds"] - (time.monotonic() - started)
        require(remaining > 0, "No numerical budget remains")
        signal.setitimer(signal.ITIMER_REAL, remaining)
        receipt = audit(args.result, args.sha256)
    except Exception as error:  # noqa: BLE001 -- unexpected audit failures must leave a null receipt
        receipt.update(
            status="INCONCLUSIVE_REFERENCE"
            if isinstance(error, ReferenceFailure)
            else "INTEGRITY_OR_AUDIT_FAILURE",
            error_type=type(error).__name__,
            error=str(error),
            recommended_method=None,
            source_result_path=str(args.result),
            source_result_sha256=args.sha256,
        )
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous)
    receipt.update(
        schema="cfeg.numerical_stability.audit.v1",
        exclusions=EXCLUSIONS,
        audit_wall_seconds=time.monotonic() - started,
        audit_cpu_seconds=time.process_time() - cpu_started,
        audit_versions={"numpy": np.__version__, "mpmath": mp.__version__},
        limitations=[
            "Generated-only numerical evidence; no human efficacy or calibration finding.",
            "Known 960 regressions are not unseen; only 192 fixtures have registered seed roles.",
            "Normwise residual is not a certified minimal structured backward error.",
            "Producer raw-H diagnostics are retained, not independently backend-parity certified.",
            "First-order scalar gradients only; N2 is CPU-only, not CUDA-ready.",
            "Success does not reopen old candidates or authorize a human run.",
        ],
    )
    write_receipt(output, receipt)
    print(
        json.dumps(
            {
                "status": receipt["status"],
                "recommended_method": receipt["recommended_method"],
                "output": str(output),
            }
        )
    )
    return 0 if receipt["status"] == "NUMERICAL_STUDY_VERIFIED" else 2


if __name__ == "__main__":
    raise SystemExit(main())
