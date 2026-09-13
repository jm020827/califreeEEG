"""Generated and mocked I/O only; never read actual EEG, DIN or stored outcomes."""

import copy
import importlib.util
from pathlib import Path

import numpy as np
import pytest

from cfeg import mamem_events_v2 as events
from cfeg.mamem_reference_probe_v1 import NOMINAL_FREQUENCIES, reference_bank

SPEC = importlib.util.spec_from_file_location(
    "reference_runner", Path(__file__).resolve().parents[1]
    / "scripts/analysis/run_mamem_reference_probe_v1.py")
runner = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(runner)


class Poison:
    def __float__(self):
        raise AssertionError("descriptor access")

    def __int__(self):
        raise AssertionError("descriptor access")


def generated_din():
    periods = [75.5, 66.5, 58.5, 52.5, 43.5]
    labels = [0, 1, 2, 3, 4, 0, 1, 2] + [v for v in (3, 0, 4, 1, 2) for _ in range(3)]
    groups = []
    for index, label in enumerate(labels):
        relative = np.arange(int(5000 / periods[label]) + 1) * periods[label]
        group = np.empty((4, len(relative)), dtype=object)
        group[0, :] = Poison()
        group[2, :] = Poison()
        group[1, :] = 1000 + index * 10000 + relative
        group[3, :] = 101 + index * 2000 + np.rint(relative / 4).astype(int)
        groups.append(group)
    return np.concatenate(groups, axis=1), groups


@pytest.fixture(autouse=True)
def no_actual_deadline(monkeypatch):
    monkeypatch.setattr(runner, "remaining", lambda: 60)


def windows_and_signal(din):
    records = events.parse_main_trials(din, 117917, include_metadata=False)
    traces = {}
    time = np.arange(1, 501) / 250
    for record in records:
        key = (record["start0"], record["end0"])
        traces[key] = np.sin(2 * np.pi * NOMINAL_FREQUENCIES[record["label"]] * time + 0.4)

    class OnlyAllowedWindows:
        def __getitem__(self, key):
            channel, window = key
            assert channel == 125 and window.step is None
            return traces[(window.start, window.stop)].copy()

    return records, OnlyAllowedWindows()


def test_first_support_selection_bounds_no_metadata(monkeypatch):
    din, _ = generated_din()
    monkeypatch.setattr(events, "marker_features", lambda *_: pytest.fail("M extraction"))
    support, frequencies, banks = runner.support_bank(din, 117917)
    assert [r["label"] for r in support] == list(range(5))
    assert {r["group_index"] for r in support} == {8, 11, 14, 17, 20}
    for row in support:
        assert len(row["event_samples"]) >= 4
        assert all(row["start0"] <= sample - 1 < row["end0"] for sample in row["event_samples"])
    assert np.all(np.diff(frequencies["SAMPLE_SUPPORT"]) > 0)
    assert all(not bank.flags.writeable for bank in banks.values())


def test_unselected_repeats_cannot_change_bank():
    din, groups = generated_din()
    original = runner.support_bank(din, 117917)
    for group_index in (9, 10, 12, 13, 15, 16, 18, 19, 21, 22):
        groups[group_index][3, :] = groups[group_index][3, :] + 1
    changed = runner.support_bank(np.concatenate(groups, axis=1), 117917)
    assert original[:2] == changed[:2]
    for arm in runner.ARM_NAMES:
        np.testing.assert_array_equal(original[2][arm], changed[2][arm])


def test_nonmonotone_support_bank_stops_without_resort(monkeypatch):
    din, _ = generated_din()
    from cfeg import mamem_reference_probe_v1 as core

    def fail(_):
        raise ValueError("frequency_range_order")

    monkeypatch.setattr(core, "sample_clock_frequencies", fail)
    with pytest.raises(ValueError, match="frequency_range_order"):
        runner.support_bank(din, 117917)


def test_full_generated_inspect_access_order_and_cost(monkeypatch):
    din, _ = generated_din()
    records, eeg = windows_and_signal(din)
    access = []
    seen = []

    def loader(role):
        seen.append(role)
        if role == "a":
            return {"DIN_1": din, "samplingRate": np.array([[250]])}, 117917
        assert access == ["a_loaded", "support_bank_frozen"]
        return {"DIN_1": din, "eeg": eeg, "samplingRate": np.array([[250]])}, 117917

    original = runner.evaluate

    def evaluation(decoded, labels):
        assert access[-1] == "all_scores_frozen"
        assert all("label" not in row for row in decoded)
        return original(decoded, labels)

    monkeypatch.setattr(runner, "evaluate", evaluation)
    result = runner.inspect(access, loader)
    assert seen == ["a", "b"]
    assert access == ["a_loaded", "support_bank_frozen", "b_loaded", "all_scores_frozen",
                      "evaluation_done"]
    assert result["summary"]["NOMINAL"]["correct"] == 15
    assert result["summary"]["decision"] == "REFERENCE_DIAGNOSTIC_COMPLETE"
    assert len(result["queries"]) == 15 and result["scope"] == runner.SCOPE
    assert result["cost"]["selected_stimulus_seconds"] == 25
    assert result["cost"]["analyzed_support_seconds"] == 10
    assert result["cost"]["full_support_record_seconds"] == 471.668
    assert result["cost"]["selected_support_elapsed_prefix_seconds"] == (
        max(r["trial_end0"] for r in records if r["group_index"] in (8, 11, 14, 17, 20)) / 250)
    assert result["cost"]["query_ready_elapsed"] == result["cost"]["setup_seconds"] == "UNKNOWN"


def test_query_labels_cannot_enter_decoder_and_only_change_evaluation():
    din, _ = generated_din()
    records, eeg = windows_and_signal(din)
    windows = [{k: r[k] for k in ("group_index", "start0", "end0")} for r in records]
    banks = {arm: reference_bank(NOMINAL_FREQUENCIES) for arm in runner.ARM_NAMES}
    decoded = runner.decode_windows(eeg, windows, banks)
    before = copy.deepcopy(decoded)
    truths = [r["label"] for r in records]
    first, first_summary = runner.evaluate(decoded, truths)
    second, second_summary = runner.evaluate(decoded, [(label + 1) % 5 for label in truths])
    assert decoded == before
    assert first_summary["NOMINAL"]["correct"] == 15
    assert second_summary["NOMINAL"]["correct"] == 0
    assert second_summary["decision"] == "NO_ARM_PASSES_STOP"
    for left, right in zip(first, second):
        for arm in runner.ARM_NAMES:
            assert left["arms"][arm]["scores"] == right["arms"][arm]["scores"]
            assert left["arms"][arm]["prediction"] == right["arms"][arm]["prediction"]
    windows[0]["label"] = 3
    with pytest.raises(ValueError, match="window_no_truth"):
        runner.decode_windows(eeg, windows, banks)


@pytest.mark.parametrize("role", ["a", "b"])
def test_load_whitelist_header_and_exact_one_load(role, monkeypatch):
    monkeypatch.setattr(runner, "validate_input", lambda value: None)
    sample_count = 117917 if role == "a" else 4000
    din_count = 1966 if role == "a" else 12
    calls = []
    data = {"DIN_1": np.ones((4, din_count), dtype=object), "samplingRate": np.array([[250]])}
    if role == "b":
        data["eeg"] = np.zeros((257, sample_count), dtype=np.float64)

    def loader(path, **kwargs):
        calls.append(kwargs)
        return data

    def header(path):
        return [("eeg", (257, sample_count), "double"), ("DIN_1", (4, din_count), "cell"),
                ("samplingRate", (1, 1), "uint16")]

    result, total = runner.load_role(role, loader, header)
    assert result is data and total == sample_count
    assert len(calls) == 1 and calls[0]["variable_names"] == runner.VARIABLES[role]
    assert calls[0]["verify_compressed_data_integrity"] is True


@pytest.mark.parametrize("wrong", ["extra_eeg", "rate", "header"])
def test_A_loader_rejects_scope_mismatches(wrong, monkeypatch):
    monkeypatch.setattr(runner, "validate_input", lambda value: None)
    data = {"DIN_1": np.ones((4, 1966), dtype=object), "samplingRate": np.array([[250]])}
    if wrong == "extra_eeg":
        data["eeg"] = Poison()
    elif wrong == "rate":
        data["samplingRate"] = np.array([[251]])
    header = [("eeg", (257, 117916 if wrong == "header" else 117917), "double"),
              ("DIN_1", (4, 1966), "cell"), ("samplingRate", (1, 1), "double")]
    with pytest.raises(ValueError):
        runner.load_role("a", lambda *a, **k: data, lambda _: header)


@pytest.mark.parametrize("mutation", ["budget", "bool", "role", "core", "extra", "missing"])
def test_manifest_poison(mutation, monkeypatch):
    cfg = dict(runner.FIXED, created_utc="generated", code_sha256={
        p: runner.UNCHANGED.get(p, "a" * 64) for p in runner.PINS})
    cfg = copy.deepcopy(cfg)
    if mutation == "budget":
        cfg["attempt_budget"] = 2
    elif mutation == "bool":
        cfg["fits"] = False
    elif mutation == "role":
        cfg["role_sha256"]["a"] = "f" * 64
    elif mutation == "core":
        cfg["code_sha256"][next(iter(runner.UNCHANGED))] = "f" * 64
    elif mutation == "extra":
        cfg["extra"] = 1
    else:
        cfg["code_sha256"].pop(runner.PINS[0])
    monkeypatch.setattr(runner, "sha", lambda _: "a" * 64)
    with pytest.raises(ValueError):
        runner.validate_manifest(cfg)


def test_manifest_valid_and_code_hash_mismatch(monkeypatch):
    cfg = dict(runner.FIXED, created_utc="generated", code_sha256={
        p: runner.UNCHANGED.get(p, "a" * 64) for p in runner.PINS})
    monkeypatch.setattr(runner, "sha", lambda p: cfg["code_sha256"][str(p.relative_to(runner.ROOT))])
    assert runner.validate_manifest(cfg) is cfg
    monkeypatch.setattr(runner, "sha", lambda _: "f" * 64)
    with pytest.raises(ValueError, match="code_hash"):
        runner.validate_manifest(cfg)


def test_exclusive_creation_and_aggregate_budget(tmp_path, monkeypatch):
    monkeypatch.setattr(runner, "RUN", tmp_path)
    runner.save(tmp_path / "one.json", {"ok": True})
    with pytest.raises(FileExistsError):
        runner.save(tmp_path / "one.json", {"ok": True})
    with pytest.raises(ValueError, match="output_cap"):
        runner.save(tmp_path / "too_big.json", {"value": "x" * runner.CAP})
    monkeypatch.setattr(runner, "TOTAL_CAP", 16)
    with pytest.raises(ValueError, match="total_output_cap"):
        runner.save(tmp_path / "two.json", {"ok": True})


def test_execute_repeat_cannot_spawn(tmp_path, monkeypatch):
    monkeypatch.setattr(runner, "RUN", tmp_path)
    monkeypatch.setattr(runner, "manifest", dict)
    monkeypatch.setattr(runner, "sha", lambda _: "a" * 64)
    monkeypatch.setattr(runner.subprocess, "Popen", lambda *a, **kw: pytest.fail("repeated worker"))
    runner.save(tmp_path / "started.json", {"status": "STARTED"})
    with pytest.raises(FileExistsError):
        runner.execute()


def test_deadline_is_checked_before_launch(tmp_path, monkeypatch):
    monkeypatch.setattr(runner, "RUN", tmp_path)
    monkeypatch.setattr(runner, "manifest", dict)
    monkeypatch.setattr(runner, "sha", lambda _: "a" * 64)
    monkeypatch.setattr(runner, "remaining", lambda: runner.require(False, "deadline"))
    monkeypatch.setattr(runner.subprocess, "Popen", lambda *a, **kw: pytest.fail("expired launch"))
    with pytest.raises(SystemExit):
        runner.execute()
    terminal = runner.read_json(tmp_path / "terminal.json")
    assert terminal["status"] == "STOPPED_NO_RETRY" and terminal["reason"] == "deadline"


def test_post_started_hash_failure_persists_terminal(tmp_path, monkeypatch):
    import hashlib

    monkeypatch.setattr(runner, "RUN", tmp_path)
    monkeypatch.setattr(runner, "manifest", dict)

    def hash_once(path):
        if path.name == "started.json":
            raise ValueError("deadline")
        return "a" * 64

    monkeypatch.setattr(runner, "sha", hash_once)
    monkeypatch.setattr(runner.subprocess, "Popen", lambda *a, **kw: pytest.fail("expired launch"))
    with pytest.raises(SystemExit):
        runner.execute()
    terminal = runner.read_json(tmp_path / "terminal.json")
    assert terminal["status"] == "STOPPED_NO_RETRY"
    assert terminal["started_sha256"] == hashlib.sha256((tmp_path / "started.json").read_bytes()).hexdigest()
    assert terminal["worker_claim_sha256"] is None


@pytest.mark.parametrize("wrong", ["parent", "manifest", "terminal", "claimed"])
def test_worker_refuses_unbound_or_repeated_parent(tmp_path, monkeypatch, wrong):
    import os

    monkeypatch.setattr(runner, "RUN", tmp_path)
    monkeypatch.setattr(runner, "manifest", dict)
    monkeypatch.setattr(runner, "sha", lambda _: "a" * 64)
    monkeypatch.setattr(runner.resource, "setrlimit", lambda *args: None)
    monkeypatch.setattr(runner, "inspect", lambda *_: pytest.fail("invalid worker data read"))
    started = {"status": "STARTED", "attempt": 1, "parent_pid": os.getppid(),
               "manifest_sha256": "a" * 64}
    if wrong == "parent":
        started["parent_pid"] += 1
    elif wrong == "manifest":
        started["manifest_sha256"] = "f" * 64
    runner.save(tmp_path / "started.json", started)
    if wrong == "terminal":
        runner.save(tmp_path / "terminal.json", {"status": "COMPLETE"})
    if wrong == "claimed":
        runner.save(tmp_path / "worker_claim.json", {"attempt": 1})
    with pytest.raises(FileExistsError if wrong == "claimed" else ValueError):
        runner.worker()
