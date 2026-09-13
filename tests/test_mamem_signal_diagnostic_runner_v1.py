"""Operator tests use temporary generated receipts and mocked IO, never real MAT."""

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from cfeg import mamem_signal_diagnostic_v1 as core

SPEC = importlib.util.spec_from_file_location(
    "signal_runner", Path(__file__).resolve().parents[1] / "scripts/analysis/diagnose_mamem_signal_v1.py")
runner = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(runner)


@pytest.fixture
def temporary_run(tmp_path, monkeypatch):
    monkeypatch.setattr(runner, "RUN", tmp_path)
    monkeypatch.setattr(runner, "remaining", lambda: 120)
    return tmp_path


def test_json_exclusive_size_and_nonfinite(temporary_run):
    path = temporary_run / "receipt.json"
    runner.save(path, {"ok": True})
    assert runner.read_json(path) == {"ok": True}
    with pytest.raises(FileExistsError):
        runner.save(path, {"ok": False})
    with pytest.raises(ValueError):
        runner.save(temporary_run / "nonfinite.json", {"x": float("nan")})
    with pytest.raises(ValueError, match="output_cap"):
        runner.save(temporary_run / "big.json", {"x": "x" * runner.CAP})


def fixed_manifest():
    pins = {p: runner.UNCHANGED.get(p, "0" * 64) for p in runner.PINS}
    return dict(runner.FIXED, created_utc="generated", code_sha256=pins)


@pytest.mark.parametrize("mutation", ["missing_pin", "extra_pin", "core", "scope", "budget", "key"])
def test_manifest_rejects_changed_contract_before_io(mutation, monkeypatch):
    cfg = fixed_manifest()
    monkeypatch.setattr(runner, "sha", lambda p: cfg["code_sha256"].get(str(Path(p).relative_to(runner.ROOT))))
    if mutation == "missing_pin":
        cfg["code_sha256"].pop(runner.PINS[0])
    elif mutation == "extra_pin":
        cfg["code_sha256"]["unrelated"] = "f" * 64
    elif mutation == "core":
        cfg["code_sha256"][next(iter(runner.UNCHANGED))] = "f" * 64
    elif mutation == "scope":
        cfg["subject"] = "S002"
    elif mutation == "budget":
        cfg["attempt_budget"] = 2
    else:
        cfg["extra"] = True
    with pytest.raises(ValueError):
        runner.validate_manifest(cfg)


def test_code_hash_change_and_valid_manifest(monkeypatch):
    cfg = fixed_manifest()
    monkeypatch.setattr(runner, "sha", lambda p: cfg["code_sha256"][str(Path(p).relative_to(runner.ROOT))])
    assert runner.validate_manifest(cfg) is cfg
    monkeypatch.setattr(runner, "sha", lambda _: "changed")
    with pytest.raises(ValueError, match="code_hash"):
        runner.validate_manifest(cfg)


@pytest.mark.parametrize("rate_type", ["double", "uint16"])
def test_inspect_exact_three_variable_request_and_header(rate_type, monkeypatch):
    monkeypatch.setattr(runner, "manifest", dict)
    monkeypatch.setattr(runner, "validate_input", lambda: None)
    monkeypatch.setattr(runner, "remaining", lambda: 120)
    monkeypatch.setattr(core, "diagnose", lambda *a, **kw: {"generated": True})
    monkeypatch.setattr(core, "validate_report", lambda _: True)
    calls = []
    def loader(path, **kwargs):
        calls.append(kwargs)
        return {"eeg": SimpleNamespace(shape=(257, 117917)), "DIN_1": np.empty((4, 1966), object),
                "samplingRate": np.array([[250]], dtype=np.float64)}
    header = lambda p: [("eeg", (257, 117917), "double"), ("DIN_1", (4, 1966), "cell"),
                        ("samplingRate", (1, 1), rate_type)]
    access = []
    result = runner.inspect(loader, header, access)
    assert result["status"] == "DIAGNOSTIC_COMPLETE_NOT_EFFICACY"
    assert calls == [{"variable_names": ["eeg", "DIN_1", "samplingRate"], "squeeze_me": False,
                      "struct_as_record": True, "verify_compressed_data_integrity": True}]
    assert access[-1] == "15_main_windows_diagnosed"


def test_input_failure_and_bad_header_never_load(monkeypatch):
    monkeypatch.setattr(runner, "manifest", lambda: runner.require(False, "pin"))
    def forbidden(*a, **kw):
        raise AssertionError("IO reached")
    with pytest.raises(ValueError, match="pin"):
        runner.inspect(forbidden, forbidden)
    monkeypatch.setattr(runner, "manifest", dict)
    monkeypatch.setattr(runner, "validate_input", lambda: None)
    with pytest.raises(ValueError, match="MAT_header"):
        runner.inspect(forbidden, lambda _: [])


@pytest.mark.parametrize("case", ["parent", "terminal", "duplicate_claim"])
def test_worker_cannot_reach_input_without_exclusive_active_claim(case, temporary_run, monkeypatch):
    monkeypatch.setattr(runner, "manifest", dict)
    monkeypatch.setattr(runner, "sha", lambda _: "f" * 64)
    monkeypatch.setattr(runner.resource, "setrlimit", lambda *args: None)
    monkeypatch.setattr(runner, "WALL_END", None)
    def forbidden(*args, **kwargs):
        raise AssertionError("actual input reached")
    monkeypatch.setattr(runner, "inspect", forbidden)
    started = {"status": "STARTED", "attempt": 1, "parent_pid": runner.os.getppid(),
               "manifest_sha256": "f" * 64}
    if case == "parent":
        started["parent_pid"] = -1
    elif case == "terminal":
        runner.save(temporary_run / "terminal.json", {"status": "COMPLETE"})
    else:
        runner.save(temporary_run / "worker_claim.json", {"pid": 123})
    runner.save(temporary_run / "started.json", started)
    with pytest.raises((ValueError, FileExistsError)):
        runner.worker()


@pytest.mark.parametrize("worker_code,good_report", [(0, True), (1, True), (0, False)])
def test_execute_terminal_exit_and_no_retry(worker_code, good_report, temporary_run, monkeypatch):
    monkeypatch.setattr(runner, "manifest", dict)
    monkeypatch.setattr(runner, "sha", lambda _: "f" * 64)
    monkeypatch.setattr(runner.shutil, "disk_usage", lambda _: SimpleNamespace(free=9*1024**3))
    monkeypatch.setattr(core, "validate_report", lambda _: True)
    calls = []
    class Process:
        returncode = worker_code
        def wait(self, timeout=None):
            return self.returncode
        def poll(self):
            return self.returncode
    def start(*args, **kwargs):
        calls.append(kwargs)
        runner.save(temporary_run / "diagnostic.json", {
            "status": "DIAGNOSTIC_COMPLETE_NOT_EFFICACY" if good_report else "NOT_COMPLETE",
            "mat_sha256": runner.MAT_SHA, "stored_samplingRate_hz": 250})
        return Process()
    monkeypatch.setattr(runner.subprocess, "Popen", start)
    if worker_code == 0 and good_report:
        runner.execute()
        assert runner.read_json(temporary_run / "terminal.json")["status"] == "COMPLETE"
    else:
        with pytest.raises(SystemExit) as error:
            runner.execute()
        assert error.value.code == 1
        assert runner.read_json(temporary_run / "terminal.json")["status"] == "STOPPED_NO_RETRY"
    with pytest.raises(FileExistsError):
        runner.execute()
    assert len(calls) == 1
    assert calls[0]["env"]["OPENBLAS_NUM_THREADS"] == "1"


def test_real_input_paths_are_pinned_without_reading_them():
    assert str(runner.MAT) == runner.FIXED["mat_path"]
    assert str(runner.ROLE) == "/home/whwovy/data/mamem_i_v1_20260913/development_role.json"
    assert runner.FIXED["fits"] == 0
    assert runner.FIXED["attempt_budget"] == 1
    assert len(runner.PINS) == len(set(runner.PINS)) == 6
