"""Toy algebra and hostile receipts only; never draw or load the actual ladder.

Protocol fixtures use the audit replay to populate verbose wire diagnostics;
separate hand-derived matrix tests below are their numerical oracle. No producer
module is imported and the actual seed draw is replaced before every full audit.
"""

import ast
import copy
import importlib.util
import io
import json
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/audit_task_trca_pair_s_conditioning.py"
spec = importlib.util.spec_from_file_location("pair_terminal_audit_test", SCRIPT)
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)


def toy_base(tied=False):
    """Sparse exact zero-mean orthogonal waves, not Gaussian seed-search data."""
    base = np.zeros((5, 12, 5, 8, 125))
    for block in range(5):
        for label in range(12):
            for band in range(5):
                for channel in range(8):
                    amplitude = (1 + (block + 1) * (1 if tied else channel + 1) / 16) * (
                        1 + label / 32 + band / 16
                    )
                    base[block, label, band, channel, 2 * channel] = amplitude
                    base[block, label, band, channel, 2 * channel + 1] = -amplitude
    return base


def toy_case(base, k, condition):
    """Analytic diagonal congruence oracle; no auditor native/rebuild calls."""
    changed = base[:k].copy()
    s0, c = np.zeros((5, 12, 8, 8)), np.zeros((5, 12, 8, 8))
    cross = np.zeros((5, k * (k - 1) // 2, 12, 8, 8))
    for label in range(12):
        for band in range(5):
            amplitudes = np.array(
                [base[:k, label, band, channel, 2 * channel] for channel in range(8)]
            ).T
            energies = 2 * (amplitudes**2).sum(0)
            ordering = np.argsort(energies, kind="stable")
            spectrum = float(condition) ** (np.arange(8) / 7)
            desired = spectrum * energies.sum() / spectrum.sum()
            multipliers = np.empty(8)
            multipliers[ordering] = np.sqrt(desired / energies[ordering])
            changed[:, label, band] *= multipliers[None, :, None]
            amplitudes *= multipliers[None, :]
            # The sums are analytic, independent of concatenated matrix products.
            c[band, label] = np.diag(2 * (amplitudes**2).sum(0))
            s0[band, label] = np.diag(2 * amplitudes.sum(0) ** 2 - 2 * (amplitudes**2).sum(0))
            pairs = [(i, j) for i in range(k) for j in range(i + 1, k)]
            for e, (i, j) in enumerate(pairs):
                cross[band, e, label] = np.diag(4 * amplitudes[i] * amplitudes[j])
    return changed, s0, c, cross


def protocol_fixture(tied=False):
    base = toy_base(tied)
    arrays = {"generated_base_support": base}
    rows = []
    for k in (3, 5):
        n = k * (k - 1) // 2
        for condition in audit.CONDITIONS:
            case = f"k{k}_cond{condition}"
            values = toy_case(base, k, condition)
            for name, value in zip(("support", "s0", "c", "cross"), values, strict=True):
                arrays[f"{case}_{name}"] = value
            _, s0, c, cross = values
            for arm in audit.ARMS:
                key = f"{case}_{arm}"
                logits = (
                    np.zeros((5, n))
                    if arm == audit.ARMS[0]
                    else np.broadcast_to(np.linspace(-audit.LIMIT, audit.LIMIT, n), (5, n)).copy()
                )
                # Exact Torch wire arithmetic, tested against NumPy by the auditor.
                z = torch.tensor(logits)
                u = torch.exp(z - z.max(-1, keepdim=True).values)
                d = (u - u.mean(-1, keepdim=True)) / u.mean(-1, keepdim=True)
                s = (
                    torch.tensor(s0) + (d[..., None, None, None] * torch.tensor(cross)).sum(-4)
                ).numpy()
                row, replayed = audit.replay(s, c)
                row.update(
                    k=k,
                    target_condition=condition,
                    arm=arm,
                    uniform_s0_exact=True if arm == audit.ARMS[0] else None,
                )
                arrays.update({f"{key}_logits": logits, f"{key}_s": s})
                arrays.update({f"{key}_{name}": value for name, value in replayed.items()})
                rows.append(row)
    return base, rows, arrays


@pytest.fixture(scope="module")
def passing():
    return protocol_fixture()


@pytest.fixture(scope="module")
def rejected():
    return protocol_fixture(tied=True)


def pack(arrays):
    stream = io.BytesIO()
    np.savez_compressed(stream, **arrays)
    return stream.getvalue()


def save_json(path, value):
    if path.exists():
        path.chmod(0o600)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    path.chmod(0o400)


def wire(tmp_path, fixture):
    base, rows, arrays = fixture
    directory = tmp_path / "gate"
    directory.mkdir()
    array_path = directory / "generated_arrays.npz"
    data = pack(arrays)
    array_path.write_bytes(data)
    array_path.chmod(0o400)
    start = {
        "schema": audit.SCHEMA,
        "seed": audit.SEED,
        "conditions": list(audit.CONDITIONS),
        "code_config_pins": {
            name: audit.digest((ROOT / name).read_bytes()) for name in audit.CODE_PATHS
        },
        "cpu_threads": 1,
        "seconds_budget": 60,
        "bytes_budget": 67108864,
        "human_reads": False,
    }
    save_json(directory / "start.json", start)
    terminal = audit.terminal_summary(rows)
    result = {
        key: value for key, value in terminal.items() if key not in ("pass_rows", "rejected_rows")
    }
    result.update(
        schema=audit.SCHEMA,
        rows=copy.deepcopy(rows),
        start_sha256=audit.digest((directory / "start.json").read_bytes()),
        generated_arrays={
            "path": str(array_path),
            "sha256": audit.digest(data),
            "bytes": len(data),
        },
        elapsed_seconds=0.5,
        optimizer_updates=0,
        human_reads=False,
        query_reads_or_scores=0,
        gpu_used=False,
        held60_access=False,
        numerical_policy_changed=False,
        interpretation=audit.INTERPRETATION,
        batch_scope=audit.BATCH_SCOPE,
        failure_precedence=audit.FAILURE_PRECEDENCE,
    )
    path = directory / "result.json"
    save_json(path, result)
    return base, path, result, start


@pytest.fixture
def wired(tmp_path, passing):
    return wire(tmp_path, passing)


def result_pin(path):
    return audit.digest(path.read_bytes())


def repin_start(path, result, start):
    save_json(path.parent / "start.json", start)
    result["start_sha256"] = result_pin(path.parent / "start.json")
    save_json(path, result)


@pytest.fixture(autouse=True)
def cpu1(monkeypatch):
    for name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
        monkeypatch.setenv(name, "1")
    torch.set_num_threads(1)


def test_no_producer_imports_or_calls_and_cli_exact():
    tree = ast.parse(SCRIPT.read_text())
    imports = [node.module or "" for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)]
    imports += [
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    ]
    assert not any(name.startswith("cfeg") or "check_task_trca" in name for name in imports)
    with pytest.raises(SystemExit) as error:
        audit.main([])
    assert error.value.code == 2
    assert len(audit.CODE_PATHS) == 18


def test_literal_native_and_analytic_congruence_without_random_draw():
    base = toy_base()
    actual = audit.native_matrices(base[:3, 0, 0])
    amplitude = np.array([base[:3, 0, 0, channel, 2 * channel] for channel in range(8)]).T
    expected_c = np.diag(2 * (amplitude**2).sum(0))
    expected_s = np.diag(2 * amplitude.sum(0) ** 2 - 2 * (amplitude**2).sum(0))
    np.testing.assert_array_equal(actual[0], expected_s)
    np.testing.assert_array_equal(actual[1], expected_c)
    # One algebraic toy condition, not the actual Gaussian seed ladder.
    for actual, expected in zip(
        audit.reconstruct_case(base, 3, 1000), toy_case(base, 3, 1000), strict=True
    ):
        np.testing.assert_allclose(actual, expected, atol=2e-12, rtol=2e-12)


def test_native_energy_uses_original_machine_tiny_not_feature_threshold():
    _, c = audit.native_matrices(toy_base()[:3, 0, 0] * 1e-14)
    assert np.finfo(np.float64).tiny < np.trace(c) < 1e-24


def test_raw_h_diagnostic_and_projector_hand_derived():
    c = torch.diag(torch.arange(1, 9, dtype=torch.float64)).expand(5, 12, 8, 8).clone()
    s = torch.diag(torch.arange(1, 9, dtype=torch.float64) ** 2).expand_as(c).clone()
    h = audit.torch_raw_h(s, c)
    np.testing.assert_allclose(
        h, np.broadcast_to(np.diag(np.arange(1, 9)), h.shape), atol=2e-15, rtol=0
    )
    f = audit.torch_projector(h, c)
    expected = np.zeros(h.shape)
    expected[..., 7, 7] = 1 / 8
    np.testing.assert_allclose(f, expected, atol=1e-16, rtol=0)
    diagnostic = audit.torch_diagnostics(h, c)
    assert diagnostic["maximum_symmetry_ratio"] == 0
    assert diagnostic["rejected_top_gap_indices_zero_based"] == []
    assert diagnostic["minimum_top_gap_ratio"] == pytest.approx(1 / (8e-10))


def test_raw_h_failure_not_symmetrized_away():
    h = torch.diag(torch.arange(8, dtype=torch.float64)).expand(5, 12, 8, 8).clone()
    h[0, 11, 0, 1] = 1e-8
    c = torch.eye(8, dtype=torch.float64).expand_as(h)
    diagnostic = audit.torch_diagnostics(h, c)
    assert diagnostic["rejected_symmetry_indices_zero_based"] == [[0, 11]]
    with pytest.raises(ValueError, match="h must be symmetric"):
        audit.torch_projector(h, c)


@pytest.mark.parametrize("minimum", [0, -1])
def test_original_spd_guard(minimum):
    c = torch.eye(8, dtype=torch.float64)
    c[0, 0] = minimum
    with pytest.raises(ValueError, match="SPD"):
        audit.torch_raw_h(torch.eye(8, dtype=torch.float64), c)


def test_original_top_gap_guard_and_rejected_f_absence():
    s = np.broadcast_to(np.eye(8), (5, 12, 8, 8)).copy()
    row, arrays = audit.replay(s, s)
    assert row["status"] == "REJECTED" and len(row["scalar_failures"]) == 60
    assert row["first_failure"]["band"] == row["first_failure"]["class"] == 0
    assert set(arrays) == {"h_batch", "h_scalar"}


@pytest.mark.parametrize("fixture_name", ["passing", "rejected"])
def test_full_toy_terminal_and_immutable_separate_receipt(
    request, fixture_name, tmp_path, monkeypatch
):
    base, path, body, _ = wire(tmp_path, request.getfixturevalue(fixture_name))
    monkeypatch.setattr(audit, "base_support", lambda: base.copy())
    output = tmp_path / "terminal.json"
    receipt = audit.run(path, result_pin(path), output)
    expected = (
        "GENERATED_TERMINAL_PASS_VERIFIED"
        if fixture_name == "passing"
        else "GENERATED_TERMINAL_VALIDITY_FAILURE_VERIFIED"
    )
    assert receipt["status"] == expected
    assert receipt["scientific_efficacy"] == "NOT_EVALUATED"
    assert receipt["verification"]["rows_verified"] == 16
    assert receipt["verification"]["native_cases_verified"] == 8
    assert receipt["verification"]["rejected_row_indices"] == body["rejected_row_indices"]
    assert output.stat().st_mode & 0o777 == 0o400
    assert set(receipt["audit_code_pins"]) == {
        "scripts/audit_task_trca_pair_s_conditioning.py",
        "docs/task_trca_pair_s_conditioning_audit_contract.md",
    }
    assert {p.name for p in path.parent.iterdir()} == {
        "start.json",
        "result.json",
        "generated_arrays.npz",
    }
    before = output.read_bytes()
    with pytest.raises(ValueError, match="Fresh"):
        audit.run(path, result_pin(path), output)
    assert output.read_bytes() == before


@pytest.mark.parametrize(
    "field,value",
    [
        ("schema", "wrong"),
        ("status", "PASS"),
        ("optimizer_updates", 1),
        ("optimizer_updates", False),
        ("query_reads_or_scores", 1),
        ("gpu_used", True),
        ("human_reads", True),
        ("held60_access", True),
        ("numerical_policy_changed", True),
        ("elapsed_seconds", 60),
        ("elapsed_seconds", -1),
        ("elapsed_seconds", True),
        ("batch_scope", "independent learner parity"),
        ("failure_precedence", "ignore failures"),
    ],
)
def test_provenance_rejects_wrong_schema_budget_and_claims(wired, field, value):
    _, path, result, _ = wired
    result[field] = value
    save_json(path, result)
    with pytest.raises(ValueError):
        audit.provenance(path, result_pin(path))


@pytest.mark.parametrize(
    "mutation",
    [
        "missing_code",
        "extra_code",
        "wrong_code",
        "wrong_design",
        "wrong_program",
        "wrong_seed",
        "wrong_order",
        "threads",
        "byte_budget",
    ],
)
def test_start_code_recipe_pins_before_numeric_decode(wired, mutation, monkeypatch):
    _, path, result, start = wired
    if mutation == "missing_code":
        start["code_config_pins"].pop("src/cfeg/metrics.py")
    elif mutation == "extra_code":
        start["code_config_pins"]["extra.py"] = "0" * 64
    elif mutation == "wrong_code":
        start["code_config_pins"]["src/cfeg/metrics.py"] = "0" * 64
    elif mutation == "wrong_design":
        start["code_config_pins"][audit.CODE_PATHS[1]] = "0" * 64
    elif mutation == "wrong_program":
        start["code_config_pins"][audit.CODE_PATHS[0]] = "0" * 64
    elif mutation == "wrong_seed":
        start["seed"] += 1
    elif mutation == "wrong_order":
        start["conditions"].reverse()
    elif mutation == "threads":
        start["cpu_threads"] = True
    else:
        start["bytes_budget"] += 1
    repin_start(path, result, start)
    monkeypatch.setattr(
        audit, "decode_arrays", lambda *args: pytest.fail("Numerical decode before pin rejection")
    )
    with pytest.raises(ValueError):
        audit.run(path, result_pin(path), path.parent.parent / "audit.json")


@pytest.mark.parametrize(
    "mutation",
    [
        "result_hash",
        "start_hash",
        "npz_hash",
        "npz_bytes",
        "npz_path",
        "row_order",
        "row_count",
        "extra_field",
    ],
)
def test_external_chain_descriptor_and_exact_order(wired, mutation):
    _, path, result, _ = wired
    pin = result_pin(path)
    if mutation == "start_hash":
        result["start_sha256"] = "0" * 64
    elif mutation == "npz_hash":
        result["generated_arrays"]["sha256"] = "0" * 64
    elif mutation == "npz_bytes":
        result["generated_arrays"]["bytes"] += 1
    elif mutation == "npz_path":
        result["generated_arrays"]["path"] = str(path.parent.parent / "other.npz")
    elif mutation == "row_order":
        result["rows"][0], result["rows"][1] = result["rows"][1], result["rows"][0]
    elif mutation == "row_count":
        result["rows"].pop()
    elif mutation == "extra_field":
        result["extra"] = True
    save_json(path, result)
    with pytest.raises(ValueError):
        audit.provenance(path, "0" * 64 if mutation == "result_hash" else result_pin(path))
    assert len(pin) == 64


@pytest.mark.parametrize("name", ["result.json", "start.json", "generated_arrays.npz"])
def test_reject_writable_and_symlink_saved_inputs(wired, name):
    _, path, _, _ = wired
    target = path.parent / name
    target.chmod(0o600)
    with pytest.raises(ValueError, match="writable"):
        audit.provenance(path, result_pin(path))
    target.chmod(0o400)
    moved = path.parent.parent / name
    target.rename(moved)
    target.symlink_to(moved)
    with pytest.raises(ValueError, match="unaliased"):
        audit.provenance(path, result_pin(path))


@pytest.mark.parametrize("name", ["failure.json", "unexpected.json"])
def test_failure_precedence_and_exact_source_directory(wired, name, monkeypatch):
    _, path, _, _ = wired
    (path.parent / name).write_text("not even decoded")
    monkeypatch.setattr(
        audit, "decode_arrays", lambda *args: pytest.fail("Forbidden numeric decode")
    )
    output = path.parent.parent / "audit.json"
    assert (
        audit.main(["--result", str(path), "--sha256", result_pin(path), "--output", str(output)])
        == 2
    )
    assert json.loads(output.read_text())["status"] == "GENERATED_TERMINAL_AUDIT_FAILURE"


@pytest.mark.parametrize(
    "mutation", ["extra", "missing_f", "dtype", "shape", "object", "nonfinite"]
)
def test_npz_inventory_dtype_shape_and_values(passing, mutation):
    _, rows, originals = passing
    arrays = dict(originals)
    key = "k3_cond1_UNIFORM_C2_f_batch"
    if mutation == "extra":
        arrays["query_values"] = np.zeros(2)
    elif mutation == "missing_f":
        arrays.pop(key)
    elif mutation == "dtype":
        arrays[key] = arrays[key].astype(np.float32)
    elif mutation == "shape":
        arrays[key] = arrays[key][0]
    elif mutation == "object":
        arrays[key] = arrays[key].astype(object)
    else:
        arrays[key] = arrays[key].copy()
        arrays[key][0, 0, 0, 0] = np.nan
    with pytest.raises(ValueError):
        audit.decode_arrays(pack(arrays), rows)


def test_rejected_row_cannot_have_f_artifact(rejected):
    _, rows, originals = rejected
    arrays = dict(originals)
    arrays["k3_cond1_UNIFORM_C2_f_batch"] = np.zeros((5, 12, 8, 8))
    with pytest.raises(ValueError, match="inventory"):
        audit.decode_arrays(pack(arrays), rows)


@pytest.mark.parametrize(
    "suffix",
    [
        "support",
        "s0",
        "c",
        "cross",
        "UNIFORM_C2_logits",
        "UNIFORM_C2_s",
        "UNIFORM_C2_h_batch",
        "UNIFORM_C2_h_scalar",
        "UNIFORM_C2_f_batch",
    ],
)
def test_numerical_tamper_detected_without_changing_seed(passing, suffix, monkeypatch):
    base, rows, originals = passing
    monkeypatch.setattr(audit, "base_support", lambda: base.copy())
    arrays = dict(originals)
    key = "k3_cond1_" + suffix
    arrays[key] = arrays[key].copy()
    arrays[key].flat[0] += 0.1
    result = {**audit.terminal_summary(rows), "rows": rows}
    with pytest.raises(ValueError):
        audit.audit_arrays(result, arrays)


@pytest.mark.parametrize(
    "mutation",
    ["status", "diagnostic", "first_failure", "unknown", "terminal_index", "terminal_count"],
)
def test_reported_status_diagnostics_and_terminal_not_trusted(passing, mutation, monkeypatch):
    base, original_rows, arrays = passing
    monkeypatch.setattr(audit, "base_support", lambda: base.copy())
    rows = copy.deepcopy(original_rows)
    result = {**audit.terminal_summary(rows), "rows": rows}
    if mutation == "status":
        rows[0]["status"] = "REJECTED"
    elif mutation == "diagnostic":
        rows[0]["batch"]["maximum_symmetry_ratio"] = 2.0
    elif mutation == "first_failure":
        rows[0]["first_failure"] = {"band": 0}
    elif mutation == "unknown":
        rows[0]["query_read"] = 1
    elif mutation == "terminal_index":
        result["first_rejected_row"] = 14
    else:
        result["rejected_row_indices"] = [14]
    with pytest.raises(ValueError):
        audit.audit_arrays(result, arrays)


def test_exact_base_uniform_and_fixed_bounds(passing, monkeypatch):
    base, rows, arrays = passing
    changed = base.copy()
    changed.flat[0] += 1e-15
    monkeypatch.setattr(audit, "base_support", lambda: changed)
    with pytest.raises(ValueError, match="Base Gaussian"):
        audit.audit_arrays({"rows": rows}, arrays)
    p = np.exp(np.linspace(-audit.LIMIT, audit.LIMIT, 10))
    p /= p.sum()
    assert p.max() / p.min() == pytest.approx(2)


def test_duplicate_json_and_nonfinite_rejected():
    for data in (b'{"a":1,"a":2}', b'{"a":NaN}', b"[]"):
        with pytest.raises(ValueError):
            audit.decode_json(data)


def test_output_and_cpu_boundaries_before_numeric_decode(wired, monkeypatch):
    _, path, _, _ = wired
    with pytest.raises(ValueError, match="outside"):
        audit.run(path, result_pin(path), path.parent / "audit.json")
    monkeypatch.setenv("OMP_NUM_THREADS", "2")
    output = path.parent.parent / "audit.json"
    with pytest.raises(ValueError, match="CPU1"):
        audit.run(path, result_pin(path), output)
    assert not output.exists()


def test_terminal_failure_scope_does_not_mean_scientific_pass():
    rows = [{"status": "PASS", "first_failure": None} for _ in range(16)]
    failure = {
        "backend": "batch_projector",
        "band": 0,
        "class": 11,
        "reason": "h must be symmetric within the frozen tolerance",
        "location_scope": audit.LOCATION_SCOPE,
    }
    rows[14] = {"status": "REJECTED", "first_failure": failure}
    result = audit.terminal_summary(rows)
    assert result["status"] == "GENERATED_CONDITIONING_VALIDITY_FAILURE"
    assert result["first_failure"] == {"row": 14, **failure}
    assert result["pass_rows"] == 15 and result["rejected_rows"] == 1


def test_publication_rechecks_late_failure_receipt(wired, passing, monkeypatch):
    base, path, _, _ = wired
    monkeypatch.setattr(audit, "base_support", lambda: base.copy())
    original = audit.audit_arrays

    def with_late_failure(body, arrays):
        answer = original(body, arrays)
        (path.parent / "failure.json").write_text("late failure")
        return answer

    monkeypatch.setattr(audit, "audit_arrays", with_late_failure)
    output = path.parent.parent / "audit.json"
    with pytest.raises(ValueError, match="failure.json"):
        audit.run(path, result_pin(path), output)
    assert json.loads(output.read_text())["status"] == "GENERATED_TERMINAL_AUDIT_FAILURE"


def test_rejected_first_location_must_match_replay(rejected, monkeypatch):
    base, original_rows, arrays = rejected
    monkeypatch.setattr(audit, "base_support", lambda: base.copy())
    rows = copy.deepcopy(original_rows)
    result = {**audit.terminal_summary(rows), "rows": rows}
    rows[0]["first_failure"]["class"] = 7
    with pytest.raises(ValueError, match="first failure"):
        audit.audit_arrays(result, arrays)


def test_elapsed_failure_is_immutable_and_cli_nonzero(wired, monkeypatch):
    base, path, _, _ = wired
    monkeypatch.setattr(audit, "base_support", lambda: base.copy())
    calls = iter((0.0, 61.0, 62.0))
    monkeypatch.setattr(audit.time, "perf_counter", lambda: next(calls))
    output = path.parent.parent / "audit.json"
    code = audit.main(
        ["--result", str(path), "--sha256", result_pin(path), "--output", str(output)]
    )
    assert code == 2
    receipt = json.loads(output.read_text())
    assert receipt["status"] == "GENERATED_TERMINAL_AUDIT_FAILURE"
    assert "elapsed budget" in receipt["error"]
    assert output.stat().st_mode & 0o777 == 0o400
