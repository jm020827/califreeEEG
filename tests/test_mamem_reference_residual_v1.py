"""Synthetic-only algebra, poisoned EEG, and mocked lifecycle integration."""

import copy
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

from cfeg import mamem_events_v2 as events
from cfeg.mamem_reference_residual_v1 import SUBJECTS, common_logs, extract, summarize

ROOT = Path(__file__).resolve().parents[1]


def module(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts/analysis" / (name+".py"))
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


runner = module("run_mamem_reference_residual_v1")
auditor = module("audit_mamem_reference_residual_v1")


class Poison:
    def __float__(self):
        raise AssertionError("forbidden descriptor or EEG conversion")

    def __int__(self):
        raise AssertionError("forbidden descriptor or EEG conversion")


def din(order=(3, 0, 4, 1, 2)):
    periods = [75.5, 66.5, 58.5, 52.5, 43.5]
    labels = [0, 1, 2, 3, 4, 0, 1, 2]+[c for c in order for _ in range(3)]
    groups = []
    for i, label in enumerate(labels):
        t = np.arange(int(5000/periods[label])+1)*periods[label]
        group = np.empty((4, len(t)), dtype=object)
        group[0, :] = Poison()
        group[2, :] = Poison()
        group[1, :] = 1000+i*10000+t
        group[3, :] = 101+i*2000+np.rint(t/4).astype(int)
        groups.append(group)
    return np.concatenate(groups, axis=1)


def world(kind="positive"):
    base = np.array([6.5, 7.3, 8.3, 9.7, 11.6])[None, None, :, None]
    people = np.linspace(-.012, .012, 10)[:, None, None, None]
    repeat = np.array([0, -.0001, .0001])[None, None, None, :]
    data = base*np.exp(people+repeat+np.array([0, .005])[None, :, None, None])
    if kind == "reversed":
        data[:, 1] = data[::-1, 1]
    elif kind == "constant":
        data = np.broadcast_to(base, (10, 2, 5, 3)).copy()
    return data


@pytest.fixture(autouse=True)
def no_deadline(monkeypatch):
    monkeypatch.setattr(runner, "remaining", lambda: 60)


@pytest.mark.parametrize("kind", ["positive", "reversed", "constant"])
def test_independent_scalar_summary(kind):
    data = world(kind)
    result = summarize(data)
    assert auditor.compare(result, auditor.independent_summary(data.tolist())) < 1e-9
    if kind == "positive":
        assert result["decision"] == "ELIGIBLE_CLASSWISE_DIRECT_TRANSFER_ONLY"
        assert result["improved_participants"] == 10
    else:
        assert result["decision"] == "RETIRE_FULL_RESIDUAL_ROUTE"
    if kind == "constant":
        assert result["common_error_floor"] and result["factor_descriptive"]["zero_residual_energy"]


def test_both_target_runs_excluded_from_common():
    values = world()
    original = common_logs(values, 3)
    values[3] *= 1.1
    np.testing.assert_array_equal(original, common_logs(values, 3))


def test_only_first_a_is_predictor_and_b_is_diagnostic():
    values = world()
    before = summarize(values)["participants"][3]
    values[3, 0, :, 1:] *= 1.01
    values[3, 1] *= 1.02
    after = summarize(values)["participants"][3]
    assert before["direct_b_hz"] == after["direct_b_hz"]
    assert before["target_b_hz"] != after["target_b_hz"]


@pytest.mark.parametrize("bad", [np.zeros((10, 2, 5, 3)), np.ones((9, 2, 5, 3)),
                                 np.full((10, 2, 5, 3), np.nan),
                                 np.broadcast_to(np.arange(5, 0, -1)[None, None, :, None],
                                                 (10, 2, 5, 3)), world().astype(complex)])
def test_bad_frequencies_stop(bad):
    with pytest.raises(ValueError):
        summarize(bad)


def test_extract_order_bounds_and_no_m2(monkeypatch):
    monkeypatch.setattr(events, "marker_features", lambda *_: pytest.fail("M2 access"))
    for order in ((3, 0, 4, 1, 2), (4, 3, 2, 1, 0)):
        records = extract(din(order), 50000)
        assert [r["label"] for r in records] == [c for c in order for _ in range(3)]
        assert [r["repeat"] for r in records] == [0, 1, 2]*5
        for row in records:
            assert all(row["start0"] <= v-1 < row["end0"] for v in row["event_samples"])
            f = 250*(len(row["event_samples"])-1)/(2*(row["event_samples"][-1]-row["event_samples"][0]))
            assert row["frequency_hz"] == f


def test_din_whitelist_and_header_only_eeg(monkeypatch):
    monkeypatch.setattr(runner, "validate_input", lambda _: None)
    value = din()
    calls = []

    def loader(path, **kwargs):
        calls.append(kwargs)
        assert kwargs == {"variable_names": ["DIN_1", "samplingRate"], "squeeze_me": False,
                          "struct_as_record": True, "verify_compressed_data_integrity": True}
        return {"DIN_1": value, "samplingRate": np.array([[250.]])}

    header = lambda _: [("eeg", (257, 50000), "double"), ("DIN_1", value.shape, "cell"),
                        ("samplingRate", (1, 1), "double")]
    result, n = runner.read_din({"path": "FAKE"}, loader, header)
    assert result is value and n == 50000 and len(calls) == 1
    with pytest.raises(ValueError, match="loaded_variable_whitelist"):
        runner.read_din({"path": "FAKE"}, lambda *a, **kw: {**loader(*a, **kw), "eeg": Poison()}, header)


@pytest.mark.parametrize("mutation", ["duplicate", "channels", "rate"])
def test_bad_headers_before_load(monkeypatch, mutation):
    monkeypatch.setattr(runner, "validate_input", lambda _: None)
    header = [("eeg", (257, 50000), "double"), ("DIN_1", (4, 1000), "cell"),
              ("samplingRate", (1, 1), "double")]
    if mutation == "duplicate":
        header.append(header[0])
    elif mutation == "channels":
        header[0] = ("eeg", (256, 50000), "double")
    else:
        header[2] = ("samplingRate", (1, 2), "double")
    with pytest.raises(ValueError):
        runner.read_din({"path": "FAKE"}, lambda *a, **kw: pytest.fail("premature load"), lambda _: header)


@pytest.mark.parametrize("payload", ['{"x":1,"x":2}', '{"x":NaN}', '{"x":Infinity}'])
def test_invalid_json(tmp_path, payload):
    path = tmp_path/"bad.json"
    path.write_text(payload)
    with pytest.raises(ValueError):
        runner.read_json(path)


def test_output_cap_reserves_terminal_and_no_overwrite(tmp_path, monkeypatch):
    monkeypatch.setattr(runner, "RUN", tmp_path)
    monkeypatch.setattr(runner, "CAP", 100)
    monkeypatch.setattr(runner, "TOTAL_CAP", 500)
    runner.save(tmp_path/"one.json", {"s": "a"*80})
    runner.save(tmp_path/"two.json", {"s": "a"*80})
    with pytest.raises(ValueError, match="run_total_cap"):
        runner.save(tmp_path/"overflow.json", {"s": "a"*40})
    runner.save(tmp_path/"terminal.json", {"status": "STOPPED_NO_RETRY"})
    with pytest.raises(FileExistsError):
        runner.save(tmp_path/"terminal.json", {})


def fake_workspace(tmp_path, monkeypatch):
    root, run = tmp_path/"repo", tmp_path/"run"
    root.mkdir()
    run.mkdir()
    monkeypatch.setattr(runner, "ROOT", root)
    monkeypatch.setattr(runner, "RUN", run)
    rows = []
    for s in SUBJECTS:
        for r in ("a", "b"):
            rows.append({"subject": s, "run": r, "sha256": "a"*64, "receipt_sha256": "b"*64})
    for path in runner.PINS:
        file = root/path
        file.parent.mkdir(parents=True, exist_ok=True)
        file.write_text("fake pin\n")
    (root/runner.SPEC).write_text(json.dumps({"schema": "cfeg.mamem-reference-residual-v1.inputs",
                                           "inputs": rows}))
    monkeypatch.setattr(runner, "specification", lambda: rows)
    state = {"calls": []}

    def read(row):
        if row["run"] == "b":
            assert (run/"a_frozen.json").exists()
        state["calls"].append(row["subject"]+row["run"])
        return din((3, 0, 4, 1, 2) if row["run"] == "a" else (4, 2, 0, 3, 1)), 50000

    monkeypatch.setattr(runner, "read_din", read)
    runner.save(run/"manifest.json", {"schema": "cfeg.mamem-reference-residual-v1.manifest",
         "created_utc": runner.now(), "deadline_utc": runner.DEADLINE,
         "attempts": 1, "scope": runner.SCOPE, "pins": {p: runner.sha(root/p) for p in runner.PINS}})
    runner.save(run/"started.json", {"status": "STARTED", "attempt": 1, "parent_pid": 77,
         "started_utc": runner.now(), "manifest_sha256": runner.sha(run/"manifest.json")})
    runner.save(run/"worker_claim.json", {"attempt": 1, "parent_pid": 77, "started_utc": runner.now(),
         "manifest_sha256": runner.sha(run/"manifest.json")})
    return root, run, state


def test_generated_full_reader_to_independent_audit(tmp_path, monkeypatch):
    root, run, state = fake_workspace(tmp_path, monkeypatch)
    result = runner.inspect()
    runner.save(run/"result.json", result)
    for name in ("stdout.txt", "stderr.txt"):
        (run/name).touch()
    start = runner.read_json(run/"started.json")
    runner.save(run/"terminal.json", {"attempts": 1, "fits": 0, "status": "COMPLETE",
         "started_utc": start["started_utc"], "ended_utc": runner.now(),
         "manifest_sha256": runner.sha(run/"manifest.json"),
         "started_sha256": runner.sha(run/"started.json"),
         "worker_claim_sha256": runner.sha(run/"worker_claim.json"),
         "result_sha256": runner.sha(run/"result.json")})
    receipt = auditor.audit(root, run)
    assert receipt["status"] == "PASS_SAVED_EVENTS_AND_RECORDED_SCOPE"
    assert state["calls"] == auditor.NAMES
    assert result["scope"]["eeg_arrays_decoded"] == 0
    tampered = copy.deepcopy(result["summary"])
    tampered["participants"][0]["direct_b_hz"][0] += .1
    with pytest.raises(ValueError, match="numeric_scalar"):
        auditor.compare(tampered, result["summary"])


def test_partial_failure_does_not_read_remaining_files(tmp_path, monkeypatch):
    _, run, state = fake_workspace(tmp_path, monkeypatch)
    original = runner.read_din

    def fail(row):
        if row["subject"] == "S004":
            raise ValueError("synthetic_stop")
        return original(row)

    monkeypatch.setattr(runner, "read_din", fail)
    with pytest.raises(ValueError, match="synthetic_stop"):
        runner.inspect()
    assert state["calls"] == ["S002a", "S003a"]
    assert (run/"record_S003a.json").exists() and (run/"attempt_S004a.json").exists()
    assert not (run/"a_frozen.json").exists() and not (run/"result.json").exists()


def test_monotonic_deadline_stops(monkeypatch):
    fresh = module("run_mamem_reference_residual_v1")
    fresh.WALL_END = -1
    with pytest.raises(ValueError, match="deadline"):
        fresh.remaining()
