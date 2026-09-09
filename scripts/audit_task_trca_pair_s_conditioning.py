"""Independent generated C2 conditioning terminal audit; never a new experiment.

No checker, feature, operator or cfeg module is imported. NumPy/SciPy formula
agreement and original-backend Torch raw-H replay are separate evidence scopes.
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
from pathlib import Path

import numpy as np
import scipy
import torch
from scipy import linalg

ROOT = Path(__file__).resolve().parents[1]
SCHEMA = "cfeg.pair_s.conditioning.v1"
SEED = 20260909
CONDITIONS = (1, 1000, 100000, 120000)
ARMS = ("UNIFORM_C2", "FIXED_BOUNDED_PAIR_LOGITS")
LIMIT = math.log(2) / 2
BYTES_BUDGET = 67108864
SECONDS_BUDGET = 60
DESIGN_SHA = "1561541f541d3cbf9a3b2b5a35e78a409edc4099cff3bddf6b343c0d37109ee3"
PROGRAM_SHA = "78ca14d5c155841b727ef8b35b958188639dd7677d1e41ca9e8ffa9bc3bc3a10"
CODE_PATHS = (
    "configs/analysis/metadata_learning_program_v1.json",
    "configs/analysis/task_trca_pair_s_v1_design.json",
    "docs/task_trca_pair_s_v1_build.md",
    "scripts/check_task_trca_pair_s_conditioning.py",
    "src/cfeg/__init__.py",
    "src/cfeg/analysis/__init__.py",
    "src/cfeg/analysis/metadata_prior_source.py",
    "src/cfeg/analysis/metadata_prior_validation.py",
    "src/cfeg/analysis/metadata_trca_prior.py",
    "src/cfeg/analysis/ood_coverage.py",
    "src/cfeg/analysis/primary_aggregate.py",
    "src/cfeg/analysis/primary_inference.py",
    "src/cfeg/analysis/provenance.py",
    "src/cfeg/analysis/task_trca_pair_s_features.py",
    "src/cfeg/analysis/task_trca_pair_s_operator.py",
    "src/cfeg/analysis/task_trca_shape_features.py",
    "src/cfeg/analysis/task_trca_shape_operator.py",
    "src/cfeg/metrics.py",
)
TOLERANCES = {
    "support": {"atol": 2e-10, "rtol": 2e-10},
    "native_matrices": {"atol": 2e-8, "rtol": 2e-10},
    "numpy_weighted_s": {"atol": 2e-8, "rtol": 2e-10},
    "scipy_h": {"atol": 2e-9, "rtol": 2e-9},
    "scipy_f": {"atol": 1e-10, "rtol": 1e-8},
    "numpy_diagnostics": {"atol": 2e-8, "rtol": 2e-8},
    "torch_replay": {"atol": 0.0, "rtol": 0.0},
}
LOCATION_SCOPE = (
    "First diagnostic rejecting matrix in batch; scalar execution order is separately recorded"
)
INTERPRETATION = (
    "Coordinate-conditioning generated gate, not biological observations, learning or human "
    "efficacy; no selected passing subset."
)
BATCH_SCOPE = (
    "Scalar/batch projector comparison shares the previously computed S; not independent "
    "numerator or learner batching parity."
)
FAILURE_PRECEDENCE = "If failure.json exists it overrides every result status, including PASS."


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(data):
    return hashlib.sha256(data).hexdigest()


def _pin(value):
    require(
        isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value),
        "Invalid SHA256 pin",
    )
    return value


def regular(path, *, immutable=True):
    path = Path(path)
    require(path.is_absolute() and path.resolve() == path, "Absolute unaliased input required")
    info = path.lstat()
    require(stat.S_ISREG(info.st_mode), "Input must be a regular non-symlink file")
    require(not immutable or not info.st_mode & 0o222, "Saved input must not be writable")
    require(info.st_size <= BYTES_BUDGET, "Input exceeds bytes budget")
    return path


def read_pinned(path, pin, *, immutable=True):
    path = regular(path, immutable=immutable)
    data = path.read_bytes()
    require(digest(data) == _pin(pin), f"Hash mismatch: {path.name}")
    return data


def _object(pairs):
    result = {}
    for key, value in pairs:
        require(key not in result, "Duplicate JSON key")
        result[key] = value
    return result


def decode_json(data):
    def invalid(value):
        raise ValueError(f"Nonfinite JSON value: {value}")

    value = json.loads(data, object_pairs_hook=_object, parse_constant=invalid)
    require(isinstance(value, dict), "JSON object required")
    return value


def _keys(value, expected, name):
    require(isinstance(value, dict) and set(value) == set(expected), f"{name} inventory mismatch")


def _same_json(actual, expected, name):
    # JSON booleans and integers must not compare equal by Python coercion.
    require(
        json.dumps(actual, sort_keys=True, allow_nan=False)
        == json.dumps(expected, sort_keys=True, allow_nan=False),
        f"{name} mismatch",
    )


def provenance(result_path, external_pin):
    result_path = Path(result_path)
    require(result_path.name == "result.json", "Exact result.json required")
    directory = result_path.parent
    require(not os.path.lexists(directory / "failure.json"), "failure.json overrides result")
    require(
        {p.name for p in directory.iterdir()}
        == {"result.json", "start.json", "generated_arrays.npz"},
        "Gate directory inventory mismatch",
    )
    result_bytes = read_pinned(result_path, external_pin)
    result = decode_json(result_bytes)
    _keys(
        result,
        (
            "schema",
            "status",
            "start_sha256",
            "generated_arrays",
            "rows",
            "rejected_row_indices",
            "first_rejected_row",
            "first_failure",
            "elapsed_seconds",
            "optimizer_updates",
            "human_reads",
            "query_reads_or_scores",
            "gpu_used",
            "held60_access",
            "numerical_policy_changed",
            "interpretation",
            "batch_scope",
            "failure_precedence",
        ),
        "Result",
    )
    require(result["schema"] == SCHEMA, "Wrong generated schema")
    require(
        result["status"]
        in ("GENERATED_CONDITIONING_PASS", "GENERATED_CONDITIONING_VALIDITY_FAILURE"),
        "Wrong terminal status",
    )
    for key in ("human_reads", "gpu_used", "held60_access", "numerical_policy_changed"):
        require(result[key] is False, f"Forbidden claim: {key}")
    for key in ("optimizer_updates", "query_reads_or_scores"):
        require(type(result[key]) is int and result[key] == 0, f"Forbidden budget: {key}")
    elapsed = result["elapsed_seconds"]
    require(
        type(elapsed) in (int, float) and math.isfinite(elapsed) and 0 <= elapsed < SECONDS_BUDGET,
        "Elapsed budget mismatch",
    )
    for key, expected in (
        ("interpretation", INTERPRETATION),
        ("batch_scope", BATCH_SCOPE),
        ("failure_precedence", FAILURE_PRECEDENCE),
    ):
        require(result[key] == expected, f"Wrong evidence scope: {key}")
    start_bytes = read_pinned(directory / "start.json", result["start_sha256"])
    start = decode_json(start_bytes)
    _keys(
        start,
        (
            "schema",
            "seed",
            "conditions",
            "code_config_pins",
            "cpu_threads",
            "seconds_budget",
            "bytes_budget",
            "human_reads",
        ),
        "Start",
    )
    _same_json(
        {key: value for key, value in start.items() if key != "code_config_pins"},
        {
            "schema": SCHEMA,
            "seed": SEED,
            "conditions": list(CONDITIONS),
            "cpu_threads": 1,
            "seconds_budget": SECONDS_BUDGET,
            "bytes_budget": BYTES_BUDGET,
            "human_reads": False,
        },
        "Frozen start recipe/budgets",
    )
    pins = start["code_config_pins"]
    _keys(pins, CODE_PATHS, "Code/config pins")
    require(
        pins[CODE_PATHS[0]] == PROGRAM_SHA and pins[CODE_PATHS[1]] == DESIGN_SHA,
        "Frozen science pins mismatch",
    )
    for relative, pin in pins.items():
        read_pinned(ROOT / relative, pin, immutable=False)
    descriptor = result["generated_arrays"]
    _keys(descriptor, ("path", "sha256", "bytes"), "Generated arrays descriptor")
    array_path = directory / "generated_arrays.npz"
    require(descriptor["path"] == str(array_path), "Exact generated NPZ path required")
    require(type(descriptor["bytes"]) is int and descriptor["bytes"] > 0, "Bad NPZ byte count")
    array_bytes = read_pinned(array_path, descriptor["sha256"])
    require(len(array_bytes) == descriptor["bytes"], "NPZ byte count mismatch")
    require(
        len(result_bytes) + len(start_bytes) + len(array_bytes) <= BYTES_BUDGET,
        "Output bytes budget exceeded",
    )
    expected_rows = [
        (k, condition, arm) for k in (3, 5) for condition in CONDITIONS for arm in ARMS
    ]
    require(
        isinstance(result["rows"], list) and len(result["rows"]) == 16, "Exactly 16 rows required"
    )
    for row, (k, condition, arm) in zip(result["rows"], expected_rows, strict=True):
        require(isinstance(row, dict), "Row object required")
        _same_json(
            [row.get("k"), row.get("target_condition"), row.get("arm")],
            [k, condition, arm],
            "Frozen row order",
        )
        for backend in ("batch", "scalar"):
            require(
                isinstance(row.get(backend), dict)
                and row[backend].get("status") in ("PASS", "REJECTED"),
                "Invalid backend status",
            )
    return result, start, array_bytes


def expected_arrays(rows):
    shapes = {"generated_base_support": (5, 12, 5, 8, 125)}
    for row in rows:
        k, condition, arm = row["k"], row["target_condition"], row["arm"]
        n = k * (k - 1) // 2
        case = f"k{k}_cond{condition}"
        shapes.update(
            {
                f"{case}_support": (k, 12, 5, 8, 125),
                f"{case}_s0": (5, 12, 8, 8),
                f"{case}_c": (5, 12, 8, 8),
                f"{case}_cross": (5, n, 12, 8, 8),
            }
        )
        key = f"{case}_{arm}"
        shapes[f"{key}_logits"] = (5, n)
        for name in ("s", "h_batch", "h_scalar"):
            shapes[f"{key}_{name}"] = (5, 12, 8, 8)
        for backend in ("batch", "scalar"):
            if row[backend]["status"] == "PASS":
                shapes[f"{key}_f_{backend}"] = (5, 12, 8, 8)
    return shapes


def decode_arrays(data, rows):
    shapes = expected_arrays(rows)
    # Validate names, duplicate entries, expanded sizes and dtype/shape headers
    # before any numeric array is decoded; no pickle or extra payload is allowed.
    with zipfile.ZipFile(io.BytesIO(data)) as archive:
        infos = archive.infolist()
        require(len(infos) == len(shapes), "NPZ inventory count mismatch")
        require(
            {item.filename for item in infos} == {key + ".npy" for key in shapes},
            "NPZ inventory mismatch",
        )
        require(
            sum(item.file_size for item in infos) <= BYTES_BUDGET, "Expanded NPZ budget exceeded"
        )
        for info in infos:
            require(not info.is_dir() and not info.flag_bits & 1, "Unexpected NPZ member")
            with archive.open(info) as stream:
                version = np.lib.format.read_magic(stream)
                require(version == (1, 0), "Unexpected NPY header version")
                shape, _, dtype = np.lib.format.read_array_header_1_0(stream)
                require(
                    shape == shapes[info.filename[:-4]] and dtype == np.dtype("float64"),
                    "NPZ array shape/dtype mismatch",
                )
                require(
                    info.file_size - stream.tell() == math.prod(shape) * 8,
                    "NPY trailing payload or size mismatch",
                )
    with np.load(io.BytesIO(data), allow_pickle=False) as archive:
        result = {name: archive[name] for name in shapes}
    require(
        all(np.isfinite(value).all() for value in result.values()), "Nonfinite saved numeric arrays"
    )
    return result


def base_support():
    """Literal fixed Gaussian draw, called only by the real audited path."""
    rng = np.random.default_rng(SEED)
    shared = rng.normal(size=(12, 5, 8, 125))
    noise = rng.normal(size=(5, 12, 5, 8, 125))
    return shared + noise


def native_matrices(trials):
    """Independent native floating construction; individual trials uncentered."""
    total = np.zeros_like(trials[0])
    for trial in trials:
        total = total + trial
    joined = np.concatenate([trial.T for trial in trials], axis=0)
    s = total @ total.T - joined.T @ joined
    centered = joined - joined.mean(axis=0)
    c = centered.T @ centered
    energy = float(np.trace(c))
    require(
        np.isfinite(s).all()
        and np.isfinite(c).all()
        and math.isfinite(energy)
        and energy > np.finfo(np.float64).tiny,
        "Native arithmetic/energy failure",
    )
    return s, c


def reconstruct_case(base, k, condition):
    support = base[:k].copy()
    s0, c = np.empty((5, 12, 8, 8)), np.empty((5, 12, 8, 8))
    for label in range(12):
        for band in range(5):
            _, initial = native_matrices(base[:k, label, band])
            values, vectors = np.linalg.eigh(initial)
            require(np.all(values > 0), "Base C must be SPD")
            spectrum = float(condition) ** (np.arange(8) / 7)
            desired = spectrum * np.trace(initial) / spectrum.sum()
            transform = (vectors * np.sqrt(desired / values)) @ vectors.T
            support[:, label, band] = transform @ base[:k, label, band]
            s0[band, label], c[band, label] = native_matrices(support[:, label, band])
    pairs = [(i, j) for i in range(k) for j in range(i + 1, k)]
    cross = np.empty((5, len(pairs), 12, 8, 8))
    for band in range(5):
        for e, (i, j) in enumerate(pairs):
            for label in range(12):
                xi, xj = support[i, label, band], support[j, label, band]
                cross[band, e, label] = xi @ xj.T + xj @ xi.T
    return support, s0, c, cross


def compare(actual, expected, scope, maxima):
    actual, expected = np.asarray(actual), np.asarray(expected)
    require(
        actual.shape == expected.shape
        and np.isfinite(actual).all()
        and np.isfinite(expected).all(),
        f"Invalid {scope} arrays",
    )
    error = float(np.max(np.abs(actual - expected))) if actual.size else 0.0
    maxima[scope] = max(maxima.get(scope, 0.0), error)
    require(
        np.allclose(actual, expected, **TOLERANCES[scope]),
        f"{scope} comparison failed (max absolute error {error})",
    )


def symmetric(value, name):
    scale = torch.linalg.matrix_norm(value, ord="fro", dim=(-2, -1))
    error = torch.linalg.matrix_norm(value - value.mT, ord="fro", dim=(-2, -1))
    require(
        bool(
            torch.isfinite(value).all()
            and torch.isfinite(scale).all()
            and torch.isfinite(error).all()
        ),
        f"Nonfinite {name}",
    )
    require(
        not bool((error > 1e-12 * torch.clamp_min(scale, 1.0)).any()),
        f"{name} must be symmetric within the frozen tolerance",
    )
    return (value + value.mT) / 2


def torch_raw_h(s, c):
    s, c = symmetric(s, "s"), symmetric(c, "c")
    require(bool((torch.linalg.eigvalsh(c)[..., 0] > 0).all()), "C must be strictly SPD")
    lower = torch.linalg.cholesky(c)
    left = torch.linalg.solve_triangular(lower, s, upper=False)
    return torch.linalg.solve_triangular(lower, left.mT, upper=False).mT


def torch_diagnostics(h, c):
    error = torch.linalg.matrix_norm(h - h.mT, ord="fro", dim=(-2, -1))
    norm = torch.linalg.matrix_norm(h, ord="fro", dim=(-2, -1))
    ratios = error / (1e-12 * torch.clamp_min(norm, 1.0))
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


def torch_projector(h, c):
    h = symmetric(h, "h")  # Original guard first; averaging only after acceptance.
    values, vectors = torch.linalg.eigh(h)
    gaps = values[..., -1] - values[..., -2]
    tolerance = 1e-10 * torch.clamp_min(values.abs().amax(-1), 1.0)
    require(
        not bool((gaps <= tolerance).any()), "leading eigenspace is degenerate or nearly degenerate"
    )
    top = vectors[..., -1]
    p = top[..., :, None] * top[..., None, :]
    lower = torch.linalg.cholesky(symmetric(c, "c"))
    left = torch.linalg.solve_triangular(lower.mT, p, upper=True)
    k = torch.linalg.solve_triangular(lower.mT, left.mT, upper=True).mT
    k = (k + k.mT) / 2
    norm = torch.einsum("...ij,...ji->...", symmetric(c, "c"), k)
    require(bool(torch.isfinite(norm).all() and (norm > 0).all()), "Invalid projector C trace")
    f = k / norm[..., None, None]
    require(bool(torch.isfinite(f).all()), "Nonfinite projector")
    return f


def replay(s, c):
    """Independent original-backend replay, separate from SciPy formula checks."""
    s, c = (torch.tensor(value, dtype=torch.float64) for value in (s, c))
    h = torch_raw_h(s, c)
    row = {"batch": torch_diagnostics(h, c), "scalar_failures": []}
    arrays = {"h_batch": h.numpy()}
    try:
        batch_f = torch_projector(h, c)
        row["batch"]["status"] = "PASS"
        arrays["f_batch"] = batch_f.numpy()
    except ValueError as error:
        row["batch"].update(status="REJECTED", error=str(error))
    scalar_h, scalar_f = torch.empty_like(h), torch.empty_like(s)
    for band in range(5):
        for label in range(12):
            scalar_h[band, label] = torch_raw_h(s[band, label], c[band, label])
            try:
                scalar_f[band, label] = torch_projector(scalar_h[band, label], c[band, label])
            except ValueError as error:
                row["scalar_failures"].append({"band": band, "class": label, "error": str(error)})
    row["scalar"] = {
        **torch_diagnostics(scalar_h, c),
        "status": "REJECTED" if row["scalar_failures"] else "PASS",
    }
    arrays["h_scalar"] = scalar_h.numpy()
    if not row["scalar_failures"]:
        arrays["f_scalar"] = scalar_f.numpy()
        if "f_batch" in arrays:
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
        indices = (
            row["batch"]["rejected_symmetry_indices_zero_based"]
            or row["batch"]["rejected_top_gap_indices_zero_based"]
        )
        row["first_failure"] = {
            "backend": "batch_projector",
            "band": indices[0][0] if indices else None,
            "class": indices[0][1] if indices else None,
            "reason": row["batch"]["error"],
            "location_scope": LOCATION_SCOPE,
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


def audit_arrays(result, arrays):
    maxima = {}
    base = base_support()
    require(
        arrays["generated_base_support"].shape == base.shape
        and arrays["generated_base_support"].tobytes() == base.tobytes(),
        "Base Gaussian draw is not exact",
    )
    checked_rows, distributions = [], []
    for k in (3, 5):
        n = k * (k - 1) // 2
        for condition in CONDITIONS:
            case = f"k{k}_cond{condition}"
            expected_case = reconstruct_case(base, k, condition)
            for name, expected in zip(("support", "s0", "c", "cross"), expected_case, strict=True):
                compare(
                    arrays[f"{case}_{name}"],
                    expected,
                    "support" if name == "support" else "native_matrices",
                    maxima,
                )
            s0, c, cross = (arrays[f"{case}_{name}"] for name in ("s0", "c", "cross"))
            for arm in ARMS:
                key = f"{case}_{arm}"
                logits = arrays[f"{key}_logits"]
                expected_logits = (
                    np.zeros((5, n))
                    if arm == ARMS[0]
                    else np.broadcast_to(np.linspace(-LIMIT, LIMIT, n), (5, n))
                )
                require(np.array_equal(logits, expected_logits), "Frozen logits mismatch")
                u = np.exp(logits - logits.max(-1, keepdims=True))
                p, d = (
                    u / u.sum(-1, keepdims=True),
                    (u - u.mean(-1, keepdims=True)) / u.mean(-1, keepdims=True),
                )
                require(
                    np.all(p > 0) and np.allclose(p.sum(-1), 1, atol=1e-14, rtol=0),
                    "Pair mass normalization mismatch",
                )
                require(
                    np.all(p.max(-1) / p.min(-1) <= 2 + 1e-12)
                    and np.all(p >= 1 / (2 * n - 1) - 1e-12)
                    and np.all(p <= 2 / (n + 1) + 1e-12),
                    "Pair mass bound mismatch",
                )
                require(
                    np.allclose(d, n * p - 1, atol=1e-14, rtol=0),
                    "Pair deviation identity mismatch",
                )
                distributions.append(
                    {
                        "k": k,
                        "condition": condition,
                        "arm": arm,
                        "p_min": float(p.min()),
                        "p_max": float(p.max()),
                        "mass_max_error": float(np.max(np.abs(p.sum(-1) - 1))),
                        "effective_pair_number": (1 / (p**2).sum(-1)).tolist(),
                    }
                )
                s = arrays[f"{key}_s"]
                compare(
                    s, s0 + (d[..., None, None, None] * cross).sum(-4), "numpy_weighted_s", maxima
                )
                z = torch.tensor(logits, dtype=torch.float64)
                tu = torch.exp(z - z.amax(-1, keepdim=True))
                td = (tu - tu.mean(-1, keepdim=True)) / tu.mean(-1, keepdim=True)
                ts = torch.tensor(s0) + (td[..., None, None, None] * torch.tensor(cross)).sum(-4)
                require(np.array_equal(s, ts.numpy()), "Torch numerator replay is not exact")
                uniform = arm == ARMS[0]
                require(not uniform or s.tobytes() == s0.tobytes(), "Uniform S0 must be bit-exact")
                row, expected_values = replay(s, c)
                row.update(
                    k=k,
                    target_condition=condition,
                    arm=arm,
                    uniform_s0_exact=True if uniform else None,
                )
                reported = result["rows"][len(checked_rows)]
                _same_json(reported, row, "Reported row/diagnostics/first failure")
                for name, expected in expected_values.items():
                    compare(arrays[f"{key}_{name}"], expected, "torch_replay", maxima)
                for backend in ("batch", "scalar"):
                    h = arrays[f"{key}_h_{backend}"]
                    antisymmetric = np.linalg.norm(h - h.swapaxes(-1, -2), axis=(-2, -1))
                    ratios = antisymmetric / (
                        1e-12 * np.maximum(np.linalg.norm(h, axis=(-2, -1)), 1)
                    )
                    compare(ratios, row[backend]["symmetry_ratios"], "numpy_diagnostics", maxima)
                    _same_json(
                        np.argwhere(ratios > 1).tolist(),
                        row[backend]["rejected_symmetry_indices_zero_based"],
                        "Independent symmetry rejection locations",
                    )
                for band in range(5):
                    for label in range(12):
                        ss = (s[band, label] + s[band, label].T) / 2
                        cc = (c[band, label] + c[band, label].T) / 2
                        lower = linalg.cholesky(cc, lower=True)
                        left = linalg.solve_triangular(lower, ss, lower=True)
                        sh = linalg.solve_triangular(lower, left.T, lower=True).T
                        for backend in ("batch", "scalar"):
                            compare(
                                arrays[f"{key}_h_{backend}"][band, label], sh, "scipy_h", maxima
                            )
                        if any(row[backend]["status"] == "PASS" for backend in ("batch", "scalar")):
                            _, vectors = linalg.eigh(ss, cc)
                            w = vectors[:, -1]
                            f = np.outer(w, w) / (w @ cc @ w)
                            for backend in ("batch", "scalar"):
                                if row[backend]["status"] == "PASS":
                                    saved_f = arrays[f"{key}_f_{backend}"][band, label]
                                    compare(saved_f, f, "scipy_f", maxima)
                                    require(
                                        abs(np.trace(cc @ saved_f) - 1) <= 1e-10,
                                        "Saved F C trace diagnostic mismatch",
                                    )
                checked_rows.append(row)
    terminal = terminal_summary(checked_rows)
    for name in ("status", "rejected_row_indices", "first_rejected_row", "first_failure"):
        _same_json(result[name], terminal[name], f"Terminal {name}")
    return {
        **terminal,
        "rows_verified": 16,
        "native_cases_verified": 8,
        "maximum_absolute_errors": maxima,
        "pair_distributions": distributions,
    }


def terminal_summary(rows):
    rejected = [i for i, row in enumerate(rows) if row["status"] != "PASS"]
    return {
        "status": "GENERATED_CONDITIONING_VALIDITY_FAILURE"
        if rejected
        else "GENERATED_CONDITIONING_PASS",
        "rejected_row_indices": rejected,
        "first_rejected_row": rejected[0] if rejected else None,
        "first_failure": {"row": rejected[0], **rows[rejected[0]]["first_failure"]}
        if rejected
        else None,
        "pass_rows": len(rows) - len(rejected),
        "rejected_rows": len(rejected),
    }


def write_receipt(path, value):
    data = (json.dumps(value, indent=2, allow_nan=False) + "\n").encode()
    require(len(data) <= BYTES_BUDGET, "Audit receipt exceeds output budget")
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o400)
    with os.fdopen(descriptor, "wb") as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())


def run(result, sha256, output):
    result, output = Path(result), Path(output)
    require(
        output.is_absolute()
        and output.resolve() == output
        and not os.path.lexists(output)
        and output.parent.is_dir(),
        "Fresh absolute unaliased audit output required",
    )
    require(output.parent != result.parent, "Audit output must be outside exact gate inventory")
    require(
        all(
            os.environ.get(name) == "1"
            for name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS")
        ),
        "CPU1 environment required",
    )
    torch.set_num_threads(1)
    started = time.perf_counter()
    audit_paths = (
        Path(__file__).resolve(),
        ROOT / "docs/task_trca_pair_s_conditioning_audit_contract.md",
    )
    audit_pins = {
        str(path.relative_to(ROOT)): digest(regular(path, immutable=False).read_bytes())
        for path in audit_paths
    }

    def expired(*unused):
        raise TimeoutError("Audit 60 second budget exceeded")

    previous = signal.signal(signal.SIGALRM, expired)
    signal.alarm(SECONDS_BUDGET)
    try:
        body, start, data = provenance(result, sha256)
        arrays = decode_arrays(data, body["rows"])
        verification = audit_arrays(body, arrays)
        # Rebind all inputs and code, and apply failure precedence again before publication.
        provenance(result, sha256)
        for relative, pin in audit_pins.items():
            read_pinned(ROOT / relative, pin, immutable=False)
        receipt = {
            "schema": "cfeg.pair_s.conditioning.terminal_audit.v1",
            "status": "GENERATED_TERMINAL_VALIDITY_FAILURE_VERIFIED"
            if verification["rejected_rows"]
            else "GENERATED_TERMINAL_PASS_VERIFIED",
            "result_path": str(result),
            "result_sha256": sha256,
            "start_sha256": body["start_sha256"],
            "generated_arrays": body["generated_arrays"],
            "code_config_pins": start["code_config_pins"],
            "audit_code_pins": audit_pins,
            "verification": verification,
            "tolerances": TOLERANCES,
            "versions": {
                "numpy": np.__version__,
                "scipy": scipy.__version__,
                "torch": torch.__version__,
            },
            "elapsed_seconds": time.perf_counter() - started,
            "cpu_threads": 1,
            "human_reads": False,
            "query_reads_or_scores": 0,
            "optimizer_updates": 0,
            "gpu_used": False,
            "scientific_efficacy": "NOT_EVALUATED",
            "scope": [
                "Exact frozen base draw and independent NumPy native/congruence/pair/S reconstruction",
                "Independent SciPy generalized F and formula H agreement",
                "Exact saved-S/C Torch raw-H/projector/diagnostic replay with unchanged acceptance guards",
            ],
            "limitations": [
                "Torch replay shares the original numerical backend; SciPy need not reject at the same rounding boundary",
                "Generated schema and pinned code support zero human/query/fit/GPU claims, not an OS sandbox attestation",
                "No learner, nested selection, CUDA parity, human efficacy or new experiment is evaluated",
                "A verified validity failure is not a candidate or scientific PASS",
            ],
        }
        require(receipt["elapsed_seconds"] < SECONDS_BUDGET, "Audit elapsed budget exceeded")
        write_receipt(output, receipt)
        return receipt
    except Exception as error:
        if not os.path.lexists(output):
            write_receipt(
                output,
                {
                    "schema": "cfeg.pair_s.conditioning.terminal_audit.v1",
                    "status": "GENERATED_TERMINAL_AUDIT_FAILURE",
                    "result_path": str(result),
                    "result_sha256": sha256,
                    "audit_code_pins": audit_pins,
                    "error": str(error),
                    "elapsed_seconds": time.perf_counter() - started,
                    "scientific_efficacy": "NOT_EVALUATED",
                },
            )
        raise
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, previous)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result", required=True)
    parser.add_argument("--sha256", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    try:
        receipt = run(args.result, args.sha256, args.output)
    except Exception as error:  # noqa: BLE001 -- CLI must return nonzero on every audit mismatch.
        print(json.dumps({"status": "GENERATED_TERMINAL_AUDIT_FAILURE", "error": str(error)}))
        return 2
    print(json.dumps({"status": receipt["status"], "output": args.output}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
