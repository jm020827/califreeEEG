"""Seed73 toys only: these tests never execute the registered resource seed.

The engine toy uses real unchanged N1 construction/packing/four-head800 updates
on eight cases. Only the caller's envelope/grid is narrowed; inherited module
globals and optimizer implementations are never patched. A separate subprocess
checks one first Adam step under the actual irreversible resource I/O guard.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import subprocess
import sys
import types
import weakref
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/check_task_trca_n1_source39_resources.py"
spec = importlib.util.spec_from_file_location("source39_resource_toy", SCRIPT)
resource_check = importlib.util.module_from_spec(spec)
spec.loader.exec_module(resource_check)


@pytest.fixture(scope="module")
def configuration():
    # Tracked numerical constants only; no referenced external path is followed.
    return resource_check.read_json(ROOT / resource_check.NATIVE_PATH, resource_check.NATIVE_SHA)


def location(tmp_path, monkeypatch, suffix="toy"):
    monkeypatch.setattr(resource_check, "OUTPUT_PARENT_ROOT", tmp_path)
    parent = tmp_path / ("task-trca-n1-source39-" + suffix)
    parent.mkdir()
    return parent / "resource1"


@pytest.fixture(scope="module")
def executed(tmp_path_factory, configuration):
    """One actual800-update toy execution, never a production-envelope PASS."""
    mp = pytest.MonkeyPatch()
    path = location(tmp_path_factory.mktemp("actual_engine"), mp)
    original_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    mp.setattr(resource_check, "SEED", 73)
    mp.setattr(resource_check, "FIT_IDS", (6, 8))
    mp.setattr(resource_check, "SAMPLES", (125, 500))
    mp.setattr(resource_check, "INTERFACES", (0,))
    mp.setattr(resource_check, "CASE_COUNT", 8)
    # Process-global guard and full repository closure are tested separately.
    mp.setattr(resource_check, "_guard", lambda *args: None)
    checks, saved, packed_refs = [], {}, []
    mp.setattr(resource_check, "_check_code", lambda pins: checks.append(pins))
    actual_builder = resource_check.build_cases
    actual_fit = resource_check.learning.fit_pipeline
    actual_pack = resource_check.evaluation.pack_cases

    def build(*args, **kwargs):
        assert kwargs["seed"] == 73
        cases = actual_builder(*args, **kwargs)
        saved["cases"] = cases
        return cases

    def pack(cases):
        result = actual_pack(cases)
        packed_refs.extend(weakref.ref(value) for value in result.values())
        saved["packed_bytes"] = sum(value.nbytes for value in result.values())
        return result

    def fit(cases, lam, **kwargs):
        assert kwargs == {"backend": "batch"} and lam == 0.001
        assert packed_refs and all(ref() is not None for ref in packed_refs)
        result = actual_fit(cases, lam, **kwargs)
        assert all(ref() is not None for ref in packed_refs)
        saved["pipeline"] = result
        return result

    # Facades alter only this new caller's references, never old module globals.
    mp.setattr(
        resource_check,
        "learning",
        types.SimpleNamespace(**{**vars(resource_check.learning), "fit_pipeline": fit}),
    )
    mp.setattr(
        resource_check,
        "evaluation",
        types.SimpleNamespace(**{**vars(resource_check.evaluation), "pack_cases": pack}),
    )
    mp.setattr(resource_check, "build_cases", build)
    try:
        saved["receipt"] = resource_check._execute(path, configuration, "toy-revision", {})
        saved["output"] = path
        saved["checks"] = checks
    finally:
        mp.undo()
        torch.set_num_threads(original_threads)
    return saved


def test_frozen_design_and_grid_are_literal():
    design = resource_check.read_json(ROOT / resource_check.DESIGN_PATH, resource_check.DESIGN_SHA)
    profile = design["human_profile"]
    assert resource_check.SEED == 20260915  # Inspect constant, do not execute it.
    assert resource_check.FIT_IDS == tuple(
        pid for rank, pid in enumerate(profile["source_ids"]) if rank % 3 != 0
    )
    assert len(resource_check.FIT_IDS) == 26
    assert resource_check.SAMPLES == tuple(profile["samples"]) == (125, 188, 250, 500)
    assert resource_check.BUDGETS == (3, 5) and resource_check.INTERFACES == (0, 1)
    assert len(resource_check.FIT_IDS) * 2 * 4 * 2 == resource_check.CASE_COUNT == 416
    assert resource_check.UPDATES == 800 and resource_check.learning.STEPS == 200
    assert resource_check.SECONDS_MAX == design["resource_rehearsal"]["seconds_max"] == 1200
    assert resource_check.HUMAN_SECONDS_MAX == design["budget"]["human_seconds_max"] == 21600
    assert len(resource_check.CODE_PATHS) == len(set(resource_check.CODE_PATHS)) == 36
    assert resource_check.DESIGN_PATH in resource_check.CODE_PATHS
    assert "scripts/audit_task_trca_n1_source39.py" in resource_check.CODE_PATHS
    assert "src/cfeg/analysis/task_trca_n1_source39_archive.py" in resource_check.CODE_PATHS


@pytest.mark.parametrize(
    "name",
    ["OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "PYTHONDONTWRITEBYTECODE"],
)
def test_preflight_rejects_wrong_environment_before_any_file_or_generation(monkeypatch, name):
    monkeypatch.setenv(name, "0")
    with pytest.raises(ValueError, match="CPU1/no-bytecode"):
        resource_check._preflight()


def test_real_engine_persists_exact_updates_and_compact_immutable_receipts(executed):
    receipt, output = executed["receipt"], executed["output"]
    assert receipt["status"] == "GENERATED_RESOURCE_PASS"
    assert receipt["study_id"] == "task-trca-n1-source39-v1"
    assert receipt["algorithm_schema"] == "task-trca-n1-integration-v1"
    assert receipt["seed"] == 73 and receipt["cases"] == 8
    assert receipt["updates_completed"] == 800
    assert receipt["state"]["optimizer_updates_completed"] == 800
    assert receipt["state"]["optimizer_updates_charged"] == 800
    assert receipt["arrays_generated"] is True
    assert receipt["human_reads"] is False and receipt["gpu_used"] is False
    assert receipt["cpu_threads"] == 1 and receipt["device"] == "cpu"
    assert receipt["resident_packed_bytes"] == executed["packed_bytes"] > 0
    assert receipt["projected_seconds"] == (
        36 * receipt["fit_seconds"] + 3 * receipt["preparation_seconds"] + 600
    )
    assert receipt["elapsed_seconds"] >= receipt["fit_seconds"] + receipt["preparation_seconds"]
    assert len(executed["checks"]) == 2
    assert {path.name for path in output.iterdir()} == {
        "start.json",
        "pipeline.json",
        "resource.json",
    }
    for path in output.iterdir():
        assert path.stat().st_mode & 0o777 == 0o400
    for name in ("start", "pipeline"):
        assert receipt[name] == resource_check.descriptor(output / (name + ".json"))
    assert sum(path.stat().st_size for path in output.iterdir()) < 1024**2
    pipeline = executed["pipeline"]
    for head in (pipeline.q, *pipeline.residuals.values()):
        assert len(head.trace) == 200
        assert [row["step"] for row in head.trace] == list(range(1, 201))
        assert np.isfinite([row["gradient_norm"] for row in head.trace]).all()


def test_literal_seed73_recipe_and_genuine_prefixes(executed, configuration):
    rng = np.random.default_rng(73)
    prototype = rng.normal(size=(12, 5, 8, 500))
    blocks = prototype[None] + rng.normal(size=(6, 12, 5, 8, 500))
    blocks += rng.normal(size=(6, 12, 5, 8, 1))
    packet = rng.uniform(0, 40, (5, 8))
    packet[0, 1], packet[0, 3], packet[1, 0], packet[3, 2] = 0, np.nan, np.nan, np.nan
    cases = [case for case in executed["cases"] if case.participant_id == 6]
    assert [case.key for case in cases] == [(6, 0, n, k) for n in (125, 500) for k in (3, 5)]
    for case in cases:
        assert case.order == 1 and case.role == "fit"
        assert case.samples == case.statistics.samples
        assert case.s.shape == case.c.shape == (5, 12, 8, 8)
        assert case.s.device.type == "cpu" and case.s.dtype == torch.float64
        np.testing.assert_array_equal(case.packet, packet[: case.k])
        expected = resource_check.learning.make_task_case(
            6,
            0,
            1,
            blocks[: case.k, ..., : case.samples],
            packet[: case.k],
            configuration["frequencies"],
            blocks[5, ..., : case.samples],
            weights=configuration["native_weights"]["ETRCA"]["dry"],
            device="cpu",
        )
        for name in ("s", "c", "anchors", "q", "m", "available", "mask"):
            np.testing.assert_array_equal(getattr(case, name), getattr(expected, name))
        for name in resource_check.evaluation.STAT_NAMES:
            np.testing.assert_array_equal(
                getattr(case.statistics, name), getattr(expected.statistics, name)
            )
    assert all(case.order == 0 for case in executed["cases"] if case.participant_id == 8)
    assert not np.array_equal(cases[0].s, cases[2].s)


def test_pipeline_validation_and_restore(executed, monkeypatch):
    model = executed["pipeline"]
    monkeypatch.setattr(resource_check, "FIT_IDS", model.fit_ids)
    record = resource_check.validate_pipeline(model)
    restored = resource_check.evaluation.pipeline_from_record(record)
    assert resource_check.validate_pipeline(restored) == record
    with pytest.raises(ValueError, match="Actual unchanged"):
        resource_check.validate_pipeline(object())
    with pytest.raises(ValueError, match="lambda"):
        resource_check.validate_pipeline(replace(model, regularization=0.01))


@pytest.mark.parametrize("defect", ["short", "order", "nan_loss", "nan_gradient"])
def test_pipeline_rejects_incomplete_or_nonfinite_trace(executed, monkeypatch, defect):
    model = executed["pipeline"]
    monkeypatch.setattr(resource_check, "FIT_IDS", model.fit_ids)
    rows = [dict(row) for row in model.q.trace]
    if defect == "short":
        rows.pop()
    elif defect == "order":
        rows[0]["step"] = 200
    elif defect == "nan_gradient":
        rows[0]["gradient_norm"] = float("nan")
    head = replace(
        model.q,
        trace=tuple(rows),
        initial_loss=float("nan") if defect == "nan_loss" else model.q.initial_loss,
    )
    with pytest.raises(ValueError):
        resource_check.validate_pipeline(replace(model, q=head))


def test_projection_exact_boundary_and_one_step_over():
    result = resource_check.projection(400, 550, 4 * 1024**3)
    assert result == {
        "projected_seconds": 21600,
        "projected_peak_rss_bytes": 16 * 1024**3,
        "runtime_ok": True,
        "memory_ok": True,
    }
    assert not resource_check.projection(400, 550.0001, 1)["runtime_ok"]
    assert not resource_check.projection(400, 550, 4 * 1024**3 + 1)["memory_ok"]


@pytest.mark.parametrize(
    "args",
    [
        (-1, 0, 1),
        (0, -1, 1),
        (np.nan, 0, 1),
        (0, np.inf, 1),
        (True, 0, 1),
        (0, 0, 0),
        (0, 0, True),
        (0, 0, 1.0),
    ],
)
def test_projection_denies_invalid_measurements(args):
    with pytest.raises(ValueError):
        resource_check.projection(*args)


def test_json_exclusive_immutable_and_descriptor_binding(tmp_path):
    pinned = resource_check.write_json(tmp_path, "start.json", {"toy": True})
    path = tmp_path / "start.json"
    assert pinned["sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
    assert resource_check.read_json(path, pinned["sha256"]) == {"toy": True}
    with pytest.raises(FileExistsError):
        resource_check.write_json(tmp_path, "start.json", {"overwrite": True})
    with pytest.raises(ValueError):
        resource_check.write_json(tmp_path, "source.npz", {})
    with pytest.raises(ValueError):
        resource_check.write_json(tmp_path, "pipeline.json", {"x": np.nan})
    with pytest.raises(ValueError):
        resource_check.read_json(path, "0" * 64)
    alias = tmp_path / "alias.json"
    alias.symlink_to(path)
    with pytest.raises((ValueError, OSError)):
        resource_check.sha(alias)
    hard = tmp_path / "hard.json"
    os.link(path, hard)
    with pytest.raises(ValueError):
        resource_check.sha(hard)


def test_output_budget_reserves_terminal_failure(tmp_path, monkeypatch):
    monkeypatch.setattr(resource_check, "OUTPUT_BYTES_MAX", 1024**2 + 100)
    with pytest.raises(ValueError, match="output budget"):
        resource_check.write_json(tmp_path, "pipeline.json", {"large": "x" * 200})
    assert not (tmp_path / "pipeline.json").exists()
    resource_check.write_json(tmp_path, "failure.json", {"status": "FAIL"})


def test_exact_code_closure_and_tamper(tmp_path, monkeypatch):
    path = tmp_path / "start.json"
    resource_check.write_json(tmp_path, path.name, {"toy": True})
    monkeypatch.setattr(resource_check, "ROOT", tmp_path)
    monkeypatch.setattr(resource_check, "CODE_PATHS", (path.name,))
    resource_check._check_code({path.name: resource_check.sha(path)})
    for pins in (
        {},
        {path.name: "0" * 64},
        {path.name: resource_check.sha(path), "other": "0" * 64},
    ):
        with pytest.raises(ValueError):
            resource_check._check_code(pins)


def test_output_location_and_alias_are_rejected_before_preflight(tmp_path, monkeypatch):
    output = location(tmp_path, monkeypatch)
    calls = []
    monkeypatch.setattr(resource_check, "_preflight", lambda: calls.append(True))
    for candidate in (output.parent / "resource2", tmp_path / "resource1"):
        with pytest.raises(ValueError):
            resource_check.run(candidate)
    output.mkdir()
    with pytest.raises(ValueError):
        resource_check.run(output)
    alias = output.parent / "alias"
    alias.symlink_to(output, target_is_directory=True)
    with pytest.raises(ValueError):
        resource_check.run(alias)
    assert calls == []


def test_setup_failure_is_preserved_without_fake_start_or_model(tmp_path, monkeypatch):
    output = location(tmp_path, monkeypatch)
    monkeypatch.setattr(resource_check, "SEED", 73)

    def fail_before_generation():
        raise ValueError("toy setup failure")

    monkeypatch.setattr(resource_check, "_preflight", fail_before_generation)
    with pytest.raises(ValueError, match="toy setup failure"):
        resource_check.run(output)
    assert {path.name for path in output.iterdir()} == {"failure.json"}
    path = output / "failure.json"
    receipt = json.loads(path.read_text())
    assert receipt["status"] == "RESOURCE_FAILURE" and receipt["stage"] == "PREFLIGHT"
    assert receipt["updates_completed"] == receipt["updates_charged"] == 0
    assert receipt["arrays_generated"] is False
    assert receipt["human_reads"] is receipt["gpu_used"] is False
    assert "start" not in receipt and "pipeline" not in receipt and "code_pins" not in receipt
    assert path.stat().st_mode & 0o777 == 0o400
    before = path.read_bytes()
    with pytest.raises(ValueError, match="New unaliased"):
        resource_check.run(output)
    assert path.read_bytes() == before


@pytest.mark.parametrize("defect", ["pin", "preparation", "time", "rss"])
def test_first_failure_preserves_zero_updates_and_cannot_restart(tmp_path, monkeypatch, defect):
    output = location(tmp_path, monkeypatch)
    monkeypatch.setattr(resource_check, "SEED", 73)
    monkeypatch.setattr(resource_check, "_guard", lambda *args: None)
    monkeypatch.setattr(resource_check, "_check_code", lambda pins: None)

    def poison(*args, **kwargs):
        raise ValueError("toy failure before any array generation")

    monkeypatch.setattr(resource_check, "build_cases", poison)
    if defect == "pin":
        monkeypatch.setattr(resource_check, "_check_code", poison)
    elif defect in ("time", "rss"):
        # Invoke the exact installed limit callback without any RNG or fit.
        def over_limit(*args, **kwargs):
            import signal

            signal.getsignal(signal.SIGALRM)()

        monkeypatch.setattr(resource_check, "build_cases", over_limit)
        if defect == "time":
            monkeypatch.setattr(resource_check, "SECONDS_MAX", -1)
        else:
            monkeypatch.setattr(resource_check, "RSS_BYTES_MAX", 1)
    with pytest.raises((ValueError, TimeoutError, MemoryError)):
        resource_check._execute(output, {}, "toy-revision", {})
    assert {path.name for path in output.iterdir()} == {"start.json", "failure.json"}
    failure = json.loads((output / "failure.json").read_text())
    assert failure["updates_completed"] == 0
    assert failure["state"]["optimizer_updates_charged"] == 0
    assert failure["human_reads"] is False and failure["gpu_used"] is False
    for path in output.iterdir():
        assert path.stat().st_mode & 0o777 == 0o400
    with pytest.raises(ValueError, match="New unaliased"):
        resource_check._execute(output, {}, "toy-revision", {})


def test_guard_allows_actual_first_adam_step_and_denies_external_io(tmp_path):
    program = """
import importlib.util, json, pathlib, sys, torch
spec = importlib.util.spec_from_file_location('resource_guard_toy', sys.argv[1])
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
output = pathlib.Path(sys.argv[2])
torch.set_num_threads(1)
torch.set_num_interop_threads(1)
p = torch.nn.Parameter(torch.ones(1, dtype=torch.float64))
optimizer = torch.optim.Adam([p], lr=.01)
m._guard([], output)
p.square().sum().backward()
optimizer.step()
assert p.item() < 1
denied = 0
for path, mode in ((output.parent/'forbidden.json', 'rb'),
                   (output.parent/'forbidden.txt', 'wb')):
    try:
        path.open(mode)
    except PermissionError:
        denied += 1
assert denied == 2
m.write_json(output, 'start.json', {'toy_updates': 1, 'denied': denied})
print(json.dumps({'toy_updates': 1, 'denied': denied}))
"""
    environment = {
        **os.environ,
        "PYTHONDONTWRITEBYTECODE": "1",
        "OMP_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "PYTHONPATH": str(ROOT / "src"),
    }
    run = subprocess.run(
        [sys.executable, "-c", program, str(SCRIPT), str(tmp_path)],
        text=True,
        capture_output=True,
        timeout=20,
        env=environment,
        check=False,
    )
    assert run.returncode == 0, run.stderr
    assert json.loads(run.stdout) == {"toy_updates": 1, "denied": 2}


def test_cli_nonzero_existing_output_has_no_generation(tmp_path):
    run = subprocess.run(
        [sys.executable, str(SCRIPT), "--output", str(tmp_path)],
        text=True,
        capture_output=True,
        timeout=20,
        check=False,
    )
    assert run.returncode != 0 and "New unaliased" in run.stderr
    assert list(tmp_path.iterdir()) == []
