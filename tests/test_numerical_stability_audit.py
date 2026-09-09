"""Toy-only independent-auditor tests. Never execute registered fixture seeds."""

from __future__ import annotations

import ast
import copy
import importlib.util
import io
import json
import os
import zipfile
from pathlib import Path
from types import SimpleNamespace

import mpmath as mp
import numpy as np
import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/audit_numerical_stability.py"
SPEC = importlib.util.spec_from_file_location("independent_numerical_audit", SCRIPT)
audit = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(audit)


@pytest.fixture(autouse=True)
def prohibit_registered_draws(monkeypatch):
    for name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
        monkeypatch.setenv(name, "1")
    original = np.random.default_rng

    def guarded(seed=None):
        assert seed not in (20260910, 20260911, 20260912), "Registered execution forbidden in toys"
        return original(seed)

    monkeypatch.setattr(np.random, "default_rng", guarded)


def toy():
    s = np.diag([-2.0, -1.5, -1, -0.5, 0, 0.5, 0.75, 1])
    value = {"s": s, "b": np.eye(8), "c": np.eye(8)}
    for name in ("g", "ds", "db", "dc"):
        value[name] = np.zeros((8, 8))
    value["g"][7, 7] = 2
    value["g"][0, 7] = value["g"][7, 0] = 0.5
    for name in ("ds", "db"):
        value[name][0, 7] = value[name][7, 0] = 1
    value["dc"][7, 7] = 1
    f = np.zeros((8, 8))
    f[7, 7] = 1
    gradients = {name: np.zeros((8, 8)) for name in ("gs", "gb", "gc")}
    gradients["gs"][0, 7] = gradients["gs"][7, 0] = 1 / 6
    gradients["gb"] = -gradients["gs"]
    gradients["gc"][7, 7] = -2
    return value, f, gradients


def immutable_bytes(path, data):
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o400)
    with os.fdopen(fd, "wb") as stream:
        stream.write(data)


def json_bytes(value):
    return json.dumps(value, sort_keys=True, allow_nan=False).encode()


def npz_bytes(values):
    stream = io.BytesIO()
    np.savez(stream, **values)
    return stream.getvalue()


@pytest.fixture
def envelope(tmp_path, monkeypatch):
    """Four closed-form toys exercise a complete archive without registered draws."""
    root, parent = tmp_path / "repo", tmp_path / "archive"
    root.mkdir()
    parent.mkdir()
    design = (audit.ROOT / audit.DESIGN_PATH).read_bytes()
    pins = {}
    for name in audit.PIN_FILES:
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        data = design if name == audit.DESIGN_PATH else b"independent toy code pin\n"
        immutable_bytes(path, data)
        pins[name] = audit.digest(data)
    monkeypatch.setattr(audit, "ROOT", root)
    monkeypatch.setattr(audit, "COUNT", 4)
    monkeypatch.setattr(audit, "GRADIENT_INDICES", (0, 1, 2, 3))
    monkeypatch.setattr(audit, "RECHECK_INDICES", (0,))
    value, f, gradients = toy()
    inputs = {name: np.tile(array, (4, 1, 1)) for name, array in value.items()}
    inputs["gradient_mask"] = np.ones(4, dtype=bool)
    inputs.update(audit.negative_cases())
    metadata = [
        {
            "id": i,
            "group": "registered_development" if i < 2 else "registered_unexposed_validation",
            "seed": 20260910 + i // 2,
            "condition": 1,
            "scale": 1.0,
            "spectrum": "distinct",
            "gap": 0.25,
            "denominator": "B_EQUALS_C" if i % 2 == 0 else "BOUNDED_DIAGONAL",
        }
        for i in range(4)
    ]
    monkeypatch.setattr(audit, "expected_inputs", lambda: (inputs, metadata))
    measurements = {}
    diagnostics = {}
    for method in audit.METHODS:
        for name, array in {"f": f, "batch_f": f, **gradients}.items():
            measurements[method + "_" + name] = np.tile(array, (4, 1, 1))
        for name in ("ok", "gradient_ok"):
            measurements[method + "_" + name] = np.ones(4, dtype=bool)
        measurements[method + "_negative_rejected"] = np.ones(9, dtype=bool)
        diagnostics[method] = [
            dict(
                **{"lambda": 1.0},
                residual=0.0,
                normalization_error=0.0,
                symmetry_error=0.0,
                raw_h_skew=0.0,
            )
            for _ in range(4)
        ]
    versions = {
        "python": "3.toy",
        "numpy": np.__version__,
        "scipy": "toy",
        "torch": "toy",
        "mpmath": mp.__version__,
    }
    start = {
        "schema": "cfeg.numerical_stability.start.v1",
        "design_path": str(root / audit.DESIGN_PATH),
        "design_sha256": audit.DESIGN_SHA,
        "code_pins": pins,
        "source_revision": "a" * 40,
        "started_utc": "2026-09-09T00:00:00+00:00",
        "versions": versions,
        "budget": json.loads(design)["budget"],
        "exclusions": audit.EXCLUSIONS,
    }
    start_data, input_data, measurement_data = (
        json_bytes(start),
        npz_bytes(inputs),
        npz_bytes(measurements),
    )
    result = {
        "schema": "cfeg.numerical_stability.result.v1",
        "status": "MEASUREMENTS_COMPLETE",
        "start_sha256": audit.digest(start_data),
        "case_metadata": metadata,
        "method_failures": [],
        "diagnostics": diagnostics,
        "negative_errors": {m: ["expected rejection"] * 9 for m in audit.METHODS},
        "numeric_wall_seconds": 0.01,
        "numeric_cpu_seconds": 0.01,
        "output_bytes_before_result": len(start_data) + len(input_data) + len(measurement_data),
        "versions": versions,
        "exclusions": audit.EXCLUSIONS,
    }
    for name, data in (("inputs", input_data), ("measurements", measurement_data)):
        result[name] = {
            "path": str(parent / (name + ".npz")),
            "sha256": audit.digest(data),
            "bytes": len(data),
        }
    for name, data in (
        ("start.json", start_data),
        ("inputs.npz", input_data),
        ("measurements.npz", measurement_data),
        ("result.json", json_bytes(result)),
    ):
        immutable_bytes(parent / name, data)
    return {
        "parent": parent,
        "root": root,
        "result": result,
        "start": start,
        "inputs": inputs,
        "measurements": measurements,
        "pin": audit.digest(json_bytes(result)),
    }


def test_independent_imports_and_exact_float_conversion():
    tree = ast.parse(SCRIPT.read_text())
    names = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.extend(alias.name for alias in node.names)
        if isinstance(node, ast.ImportFrom):
            names.append(node.module or "")
    assert not any("cfeg" in name or "run_numerical_stability" in name for name in names)
    value = np.eye(8) * 0.1
    with mp.workdps(80):
        actual = audit.mp_matrix(value)[0, 0]
        numerator, denominator = (0.1).as_integer_ratio()
        assert actual == mp.mpf(numerator) / denominator
        assert actual != mp.mpf("0.1")


def test_literal_nonregistered_grid_and_probes():
    cases = list(audit.new_cases(73))
    assert len(cases) == 96
    assert cases[0][0] == (1, 2**-20, "distinct", 0.25, "B_EQUALS_C")
    assert cases[-1][0] == (1000000, 2**20, "repeated", 0.001, "BOUNDED_DIAGONAL")
    for fields, value in cases:
        for matrix in value.values():
            assert np.array_equal(matrix, matrix.T)
        if fields[0] == 1:
            assert np.array_equal(value["c"], fields[1] * np.eye(8))
        if fields[-1] == "B_EQUALS_C":
            assert np.array_equal(value["b"], value["c"])


@pytest.mark.parametrize("lower_repeated", [False, True])
def test_reference_repeated_lower_roots_and_scale(lower_repeated):
    value, f, _ = toy()
    if lower_repeated:
        value["s"][np.arange(6), np.arange(6)] = -1
    for scale in (2**-20, 1, 2**20):
        scaled = {name: matrix * scale for name, matrix in value.items()}
        ref = audit.reference(scaled)
        assert abs(float(ref[0]) - 1) < 1e-14
        metrics, failures = audit.forward_metrics(scaled, f / scale, ref)
        assert not failures
        assert metrics["projector_error"] < 1e-60
        assert audit.precision_recheck(scaled, ref)


@pytest.mark.parametrize("kind", ["lower", "scaled", "zero", "asymmetric"])
def test_forward_mutants(kind):
    value, f, _ = toy()
    if kind == "lower":
        f[7, 7], f[0, 0] = 0, 1
    elif kind == "scaled":
        f *= 2
    elif kind == "zero":
        f *= 0
    else:
        f[0, 7] = 0.01
    if kind == "zero":
        with pytest.raises(ValueError, match="zero"):
            audit.forward_metrics(value, f, audit.reference(value))
    else:
        metrics, failures = audit.forward_metrics(value, f, audit.reference(value))
        assert failures
        if kind == "lower":
            assert metrics["residual"] == 0
            assert "top_eigenvalue_error" in failures


def test_high_precision_four_direction_gradients_and_detachment():
    value, _, gradients = toy()
    refs = audit.directional_references(value)
    expected = {"S": 1 / 3, "B": -1 / 3, "C": -2, "joint": -2}
    for direction, number in expected.items():
        assert abs(float(refs[direction]["value"]) - number) < 1e-14
        assert refs[direction]["fd_difference"] < mp.mpf("1e-20")
    _, failures = audit.gradient_metrics(value, gradients, refs)
    assert not failures
    detached = {name: np.zeros((8, 8)) for name in gradients}
    _, failures = audit.gradient_metrics(value, detached, refs)
    assert set(failures) == set(audit.DIRECTIONS)


def test_reference_top_tie_and_spd_failures():
    value, _, _ = toy()
    value["s"][6, 6] = 1
    with pytest.raises(audit.ReferenceFailure, match="top root"):
        audit.reference(value)
    value, _, _ = toy()
    value["b"][0, 0] = 0
    with pytest.raises(audit.ReferenceFailure, match="SPD"):
        audit.reference(value)


def test_full_toy_audit_and_cli_receipt(envelope, tmp_path):
    output = tmp_path / "receipt.json"
    assert (
        audit.main(
            [
                "--result",
                str(envelope["parent"] / "result.json"),
                "--sha256",
                envelope["pin"],
                "--output",
                str(output),
            ]
        )
        == 0
    )
    receipt = json.loads(output.read_text())
    assert receipt["status"] == "NUMERICAL_STUDY_VERIFIED"
    assert receipt["recommended_method"] == "N1_SYM_CHOLESKY"
    assert receipt["receipt_bytes"] == output.stat().st_size
    assert output.stat().st_mode & 0o777 == 0o400
    assert all(len(report["gradients"]) == 4 for report in receipt["methods"].values())


@pytest.mark.parametrize(
    "mutation",
    [
        "missing_case",
        "wrong_metadata",
        "wrong_mask",
        "wrong_array",
        "negative_nan",
        "extra_failure",
        "missing_gradient",
        "partial_batch",
        "negative_acceptance",
        "bad_diagnostic",
    ],
)
def test_in_memory_corruptions(envelope, mutation):
    inputs = copy.deepcopy(envelope["inputs"])
    arrays = copy.deepcopy(envelope["measurements"])
    result = copy.deepcopy(envelope["result"])
    if mutation == "missing_case":
        result["case_metadata"].pop()
    elif mutation == "wrong_metadata":
        result["case_metadata"][0]["gap"] = 0.2
    elif mutation == "wrong_mask":
        inputs["gradient_mask"][0] = False
    elif mutation == "wrong_array":
        inputs["s"][0, 0, 0] += 0.1
    elif mutation == "negative_nan":
        inputs["negative_s"][3, 0, 0] = 0
    elif mutation == "extra_failure":
        result["method_failures"].append(
            {"method": "n1", "case": 0, "stage": "scalar", "error": "forged"}
        )
        result["status"] = "MEASUREMENT_FAILURE"
    elif mutation == "missing_gradient":
        arrays["n1_gradient_ok"][0] = False
    elif mutation == "partial_batch":
        arrays["n1_batch_f"][0] = np.nan
    elif mutation == "negative_acceptance":
        arrays["n1_negative_rejected"][0] = False
    else:
        result["diagnostics"]["n1"][0]["lambda"] = None
    with pytest.raises(ValueError):
        audit.validate_reconstruction(
            inputs, result["case_metadata"], envelope["inputs"], envelope["result"]["case_metadata"]
        )
        audit.validate_measurements(result, inputs, arrays)


def test_honest_method_failures_do_not_make_archive_inconclusive(envelope):
    arrays = copy.deepcopy(envelope["measurements"])
    result = copy.deepcopy(envelope["result"])
    for method in audit.METHODS:
        arrays[method + "_batch_f"][:] = np.nan
        result["method_failures"].append(
            {"method": method, "case": None, "stage": "batch", "error": "toy fail"}
        )
    result["status"] = "MEASUREMENT_FAILURE"
    audit.validate_measurements(result, envelope["inputs"], arrays)
    numerical = audit.inspect_numerics(envelope["inputs"], arrays, result)
    assert numerical["recommended_method"] is None
    assert all(not report["eligible"] for report in numerical["methods"].values())


def test_failure_dominates_before_archive_read(envelope, monkeypatch):
    immutable_bytes(envelope["parent"] / "failure.json", b"{}")
    monkeypatch.setattr(audit, "read_pinned", lambda *_args, **_kwargs: pytest.fail("Input opened"))
    with pytest.raises(ValueError, match="dominates"):
        audit.load_envelope(envelope["parent"] / "result.json", envelope["pin"])


@pytest.mark.parametrize(
    "mutation", ["pin", "descriptor", "symlink", "hardlink", "writable", "extra_file", "code_pin"]
)
def test_envelope_corruption_before_numeric_decode(envelope, monkeypatch, mutation):
    monkeypatch.setattr(
        audit, "decode_npz", lambda *_args, **_kwargs: pytest.fail("Numeric decoded")
    )
    path = envelope["parent"] / "result.json"
    pin = envelope["pin"]
    if mutation == "pin":
        pin = "0" * 64
    elif mutation == "descriptor":
        result = copy.deepcopy(envelope["result"])
        result["inputs"]["path"] = "/forbidden/human/inputs.npz"
        os.chmod(path, 0o600)
        path.unlink()
        immutable_bytes(path, json_bytes(result))
        pin = audit.digest(json_bytes(result))
    elif mutation == "symlink":
        target = envelope["parent"] / "inputs.npz"
        saved = target.parent.parent / "aliased-inputs.npz"
        target.rename(saved)
        target.symlink_to(saved)
    elif mutation == "hardlink":
        os.link(envelope["parent"] / "inputs.npz", envelope["parent"].parent / "hardlink.npz")
    elif mutation == "writable":
        os.chmod(envelope["parent"] / "inputs.npz", 0o600)
    elif mutation == "extra_file":
        immutable_bytes(envelope["parent"] / "unexpected", b"x")
    else:
        path = envelope["root"] / audit.PIN_FILES[-1]
        os.chmod(path, 0o600)
        path.unlink()
        immutable_bytes(path, b"mutant")
    with pytest.raises(ValueError):
        audit.load_envelope(envelope["parent"] / "result.json", pin)


def test_cli_integrity_failure_receipt_and_no_overwrite(envelope, tmp_path):
    output = tmp_path / "failure-receipt.json"
    argv = [
        "--result",
        str(envelope["parent"] / "result.json"),
        "--sha256",
        "0" * 64,
        "--output",
        str(output),
    ]
    assert audit.main(argv) == 2
    receipt = json.loads(output.read_text())
    assert receipt["status"] == "INTEGRITY_OR_AUDIT_FAILURE"
    assert receipt["recommended_method"] is None
    with pytest.raises(ValueError, match="Fresh"):
        audit.main(argv)


@pytest.mark.parametrize("kind", ["missing", "extra", "object", "shape", "duplicate", "trailing"])
def test_npz_inventory_header_and_payload_corruption(kind):
    values = {"x": np.eye(8)}
    if kind == "missing":
        values = {}
    elif kind == "extra":
        values["y"] = np.eye(8)
    elif kind == "object":
        values["x"] = np.array([object()], dtype=object)
    elif kind == "shape":
        values["x"] = np.ones((7, 8))
    data = npz_bytes(values)
    if kind in ("duplicate", "trailing"):
        with zipfile.ZipFile(io.BytesIO(data)) as source:
            item = source.read("x.npy")
        stream = io.BytesIO()
        with zipfile.ZipFile(stream, "w") as target:
            target.writestr("x.npy", item + (b"x" if kind == "trailing" else b""))
            if kind == "duplicate":
                with pytest.warns(UserWarning, match="Duplicate"):
                    target.writestr("x.npy", item)
        data = stream.getvalue()
    with pytest.raises(ValueError):
        audit.decode_npz(data, {"x": ((8, 8), "float64")})


def test_json_duplicates_nonfinite_and_boolean_integer_distinction():
    for data in (b'{"x": 1, "x": 2}', b'{"x": NaN}', b"[]"):
        with pytest.raises(ValueError):
            audit.decode_json(data)
    with pytest.raises(ValueError):
        audit.same_json({"x": False}, {"x": 0}, "Strict JSON")


def test_dense_noncommuting_congruence_forward_and_four_gradients():
    value, f, gradients = toy()
    q = audit.canonical_qr(np.random.default_rng(81).normal(size=(8, 8)))
    transform = (q @ np.diag(np.linspace(1, 2, 8))) @ q.T
    inverse = np.linalg.inv(transform)
    moved = {name: audit.sym((transform @ matrix) @ transform.T) for name, matrix in value.items()}
    target = audit.sym((inverse.T @ f) @ inverse)
    ref = audit.reference(moved)
    metrics, failures = audit.forward_metrics(moved, target, ref)
    assert not failures
    assert metrics["projector_error"] < 1e-12
    refs = audit.directional_references(moved)
    transformed_gradients = {
        name: audit.sym((inverse.T @ matrix) @ inverse) for name, matrix in gradients.items()
    }
    _, failures = audit.gradient_metrics(moved, transformed_gradients, refs)
    assert not failures
    assert abs(float(refs["S"]["value"]) - 1 / 3) < 1e-12
    assert abs(float(refs["B"]["value"]) + 1 / 3) < 1e-12
    assert abs(float(refs["C"]["value"]) + 2) < 1e-12


def test_high_precision_fd_inconsistency_cannot_be_method_rejection(monkeypatch):
    original = audit.solve_reference
    calls = 0

    def corrupted(*args):
        nonlocal calls
        calls += 1
        eigenvalue, f = original(*args)
        if calls == 4:
            f[7, 7] += mp.mpf("1e-40")
        return eigenvalue, f

    monkeypatch.setattr(audit, "solve_reference", corrupted)
    with pytest.raises(audit.ReferenceFailure, match="differences inconsistent"):
        audit.directional_references(toy()[0])


def test_precision_recheck_inconsistency(monkeypatch):
    value = toy()[0]
    ref = audit.reference(value)
    original = audit.reference

    def corrupt(value, *, precision=80):
        eigenvalue, f = original(value, precision=precision)
        with mp.workdps(120):
            return eigenvalue + mp.mpf("1e-30"), f

    monkeypatch.setattr(audit, "reference", corrupt)
    with pytest.raises(audit.ReferenceFailure, match="80/120"):
        audit.precision_recheck(value, ref)


def test_insufficient_reference_coverage_is_inconclusive(envelope, monkeypatch):
    original = audit.directional_references

    def zero_probe(value):
        result = original(value)
        for direction in ("S", "B"):
            result[direction]["value"] = mp.mpf(0)
        return result

    monkeypatch.setattr(audit, "directional_references", zero_probe)
    with pytest.raises(audit.ReferenceFailure, match="coverage"):
        audit.inspect_numerics(envelope["inputs"], envelope["measurements"], envelope["result"])


def test_scalar_batch_disagreement_rejects_only_bad_method(envelope):
    arrays = copy.deepcopy(envelope["measurements"])
    arrays["n1_batch_f"][0] *= 1.1
    numerical = audit.inspect_numerics(envelope["inputs"], arrays, envelope["result"])
    assert numerical["recommended_method"] == "N2_GVD_IMPLICIT"
    assert not numerical["methods"]["n1"]["eligible"]
    assert any(
        item["gate"] == "scalar_batch_error" for item in numerical["methods"]["n1"]["failures"]
    )


def test_legitimate_atime_change_does_not_invalidate_content(tmp_path, monkeypatch):
    path = tmp_path / "immutable.bin"
    data = b"toy content"
    immutable_bytes(path, data)
    original = os.fstat
    reads = 0

    def atime_changes(fd):
        nonlocal reads
        reads += 1
        info = original(fd)
        fields = {
            name: getattr(info, name)
            for name in (
                "st_dev",
                "st_ino",
                "st_mode",
                "st_nlink",
                "st_size",
                "st_mtime_ns",
                "st_ctime_ns",
            )
        }
        return SimpleNamespace(**fields, st_atime=info.st_atime + reads)

    monkeypatch.setattr(os, "fstat", atime_changes)
    assert audit.read_pinned(path, audit.digest(data)) == data


def test_reference_failure_cli_is_null_and_exit_two(envelope, tmp_path, monkeypatch):
    def unavailable(*_args):
        raise audit.ReferenceFailure("toy oracle unavailable")

    monkeypatch.setattr(audit, "inspect_numerics", unavailable)
    output = tmp_path / "inconclusive.json"
    assert (
        audit.main(
            [
                "--result",
                str(envelope["parent"] / "result.json"),
                "--sha256",
                envelope["pin"],
                "--output",
                str(output),
            ]
        )
        == 2
    )
    receipt = json.loads(output.read_text())
    assert receipt["status"] == "INCONCLUSIVE_REFERENCE"
    assert receipt["recommended_method"] is None
