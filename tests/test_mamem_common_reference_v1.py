"""Generated-only common-bank and reader canaries; actual EEG access forbidden."""

import importlib.util
import json
import subprocess
from pathlib import Path

import numpy as np
import pytest

from cfeg import mamem_common_reference_v1 as core
from cfeg.mamem_reference_probe_v1 import reference_bank

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "common_runner", ROOT/"scripts/analysis/run_mamem_common_reference_v1.py")
runner = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(runner)


@pytest.fixture(autouse=True)
def avoid_actual_deadline(monkeypatch):
    monkeypatch.setattr(runner, "remaining", lambda: 60)


def source():
    return np.array([6.5, 7.3, 8.3, 9.6, 11.6])[None, :, None]*np.exp(
        np.linspace(-.005, .005, 10)[:, None, None]+np.array([-.0001, 0, .0001])[None, None, :])


def test_scalar_source_reference_and_target_exclusion():
    data = source()
    original = core.common_frequencies(data)
    for p, s in enumerate(core.SUBJECTS):
        expected = [np.exp(sum(np.log(data[o, c, j]) for o in range(10) if o != p
                               for j in range(3))/27) for c in range(5)]
        np.testing.assert_allclose(original[s], expected, atol=1e-12, rtol=0)
    data[2] *= 1.02
    updated = core.common_frequencies(data)
    assert updated[core.SUBJECTS[2]] == original[core.SUBJECTS[2]]
    assert updated[core.SUBJECTS[0]] != original[core.SUBJECTS[0]]


@pytest.mark.parametrize("bad", [np.ones((9, 5, 3)), np.zeros((10, 5, 3)),
                                 np.full((10, 5, 3), np.nan), source().astype(complex),
                                 source()[:, ::-1, :]])
def test_invalid_common_sources_stop(bad):
    with pytest.raises(ValueError):
        core.common_frequencies(bad)


def generated_windows():
    return [{"group_index": i+8, "start0": i*1000+250, "end0": i*1000+750}
            for i in range(15)]


def test_only_allowed_eeg_windows_and_no_labels_to_scorer(monkeypatch):
    windows = generated_windows()
    frequencies = [6.66, 7.5, 8.57, 10., 12.]
    t = np.arange(1, 501)/250
    calls = []

    class EEG:
        def __getitem__(self, key):
            row, window = key
            assert row == 125 and window.step is None
            index = next(i for i, w in enumerate(windows)
                         if (window.start, window.stop) == (w["start0"], w["end0"]))
            calls.append(index)
            return np.sin(2*np.pi*frequencies[index//3]*t+.3)

    original = core.scores
    scorer_calls = []

    def scorer(query, bank):
        scorer_calls.append(query.shape)
        return original(query, bank)

    monkeypatch.setattr(core, "scores", scorer)
    banks = {arm: reference_bank(frequencies) for arm in core.ARMS}
    result = core.decode_windows(EEG(), windows, banks)
    assert calls == list(range(15)) and scorer_calls == [(500,)]*30
    assert all("label" not in row for row in result)
    for i, row in enumerate(result):
        for value in row["arms"].values():
            assert value["prediction"] == i//3
            np.testing.assert_allclose(np.sum(np.asarray(value["projection"])**2, axis=1),
                                       value["scores"], atol=1e-12, rtol=0)


def fake_scores(nominal_correct, common_correct):
    labels = {s: [c for c in range(5) for _ in range(3)] for s in core.SUBJECTS}
    scored = []
    for p, s in enumerate(core.SUBJECTS):
        counts = {"NOMINAL": nominal_correct[p], "COMMON": common_correct[p]}
        queries = []
        # First12 labels cover all classes: use chronological order but choose errors atrepeat2.
        order = [i for i in range(15) if i % 3 != 2]+[i for i in range(15) if i % 3 == 2]
        for i, window in enumerate(generated_windows()):
            arms = {arm: {"prediction": labels[s][i] if order.index(i) < counts[arm]
                          else (labels[s][i]+1) % 5} for arm in core.ARMS}
            queries.append(dict(window, arms=arms))
        scored.append({"subject": s, "queries": queries})
    return scored, labels


@pytest.mark.parametrize("n,c,decision", [([15]*10, [15]*10, "BOTH_ZERO_SUPPORT_READY_DEV"),
                                         ([3]*10, [15]*10, "COMMON_ONLY_READY_DEV"),
                                         ([15]*10, [3]*10, "NOMINAL_ONLY_READY_DEV"),
                                         ([3]*10, [3]*10, "NEITHER_BASELINE_READY_STOP")])
def test_all_four_fixed_decisions(n, c, decision):
    scored, labels = fake_scores(n, c)
    participants, summary = core.evaluate(scored, labels)
    assert summary["decision"] == decision and len(participants) == 10
    assert summary["NOMINAL"]["correct"] == sum(n)
    assert summary["COMMON"]["correct"] == sum(c)
    assert sum(summary["paired"].values()) == 150
    assert sum(summary["subject_changes"].values()) == 10


def test_pooled_ready_also_requires_seven_people():
    scores, labels = fake_scores([15]*10, [15]*6+[10]*4)
    _, summary = core.evaluate(scores, labels)
    assert summary["COMMON"]["correct"] == 130
    assert summary["COMMON"]["subjects_ready"] == 6
    assert not summary["COMMON"]["ready"]


def test_exact_ready_boundary():
    scored, labels = fake_scores([12]*10, [12]*10)
    participants, summary = core.evaluate(scored, labels)
    assert all(p["summary"]["COMMON"]["ready"] for p in participants)
    assert summary["COMMON"]["ready"] and summary["COMMON"]["correct"] == 120


def test_labels_cannot_enter_windows():
    windows = generated_windows()
    windows[0]["label"] = 1
    with pytest.raises(ValueError, match="label_free_windows"):
        core.decode_windows(None, windows, dict.fromkeys(core.ARMS))


def test_reader_eeg_only_and_no_whole_array_qc(monkeypatch):
    monkeypatch.setattr(runner, "validate_input", lambda _: None)
    eeg = np.full((257, 1800), np.nan)
    eeg[125] = np.arange(1800)
    header = [("eeg", (257, 1800), "double"), ("DIN_1", (4, 1000), "cell"),
              ("samplingRate", (1, 1), "double")]
    kwargs_seen = []

    def loader(_, **kwargs):
        kwargs_seen.append(kwargs)
        assert kwargs == {"variable_names": ["eeg"], "squeeze_me": False,
                          "struct_as_record": True, "verify_compressed_data_integrity": True}
        return {"eeg": eeg}

    result = runner.read_eeg({"path": "FAKE"}, {"total_samples": 1800}, loader, lambda _: header)
    assert result is eeg and len(kwargs_seen) == 1
    with pytest.raises(ValueError, match="eeg_only"):
        runner.read_eeg({"path": "FAKE"}, {"total_samples": 1800},
                        lambda *a, **kw: {**loader(*a, **kw), "DIN_1": object()}, lambda _: header)


@pytest.mark.parametrize("mode", ["duplicate", "wrong_shape", "rate_shape"])
def test_bad_header_stops_before_decode(monkeypatch, mode):
    monkeypatch.setattr(runner, "validate_input", lambda _: None)
    header = [("eeg", (257, 1800), "double"), ("DIN_1", (4, 1000), "cell"),
              ("samplingRate", (1, 1), "double")]
    if mode == "duplicate":
        header.append(header[0])
    elif mode == "wrong_shape":
        header[0] = ("eeg", (256, 1800), "double")
    else:
        header[2] = ("samplingRate", (2, 1), "double")
    with pytest.raises(ValueError):
        runner.read_eeg({"path": "FAKE"}, {"total_samples": 1800},
                        lambda *a, **kw: pytest.fail("unexpected load"), lambda _: header)


@pytest.mark.parametrize("text", ['{"x":1,"x":2}', '{"x":NaN}', '{"x":Infinity}'])
def test_json_guards(tmp_path, text):
    path = tmp_path/"fixture.json"
    path.write_text(text)
    with pytest.raises(ValueError):
        runner.read(path)


def test_output_cap_preserves_terminal(tmp_path, monkeypatch):
    monkeypatch.setattr(runner, "RUN", tmp_path)
    monkeypatch.setattr(runner, "CAP", 100)
    monkeypatch.setattr(runner, "TOTAL_CAP", 500)
    runner.save(tmp_path/"first.json", {"x": "x"*80})
    runner.save(tmp_path/"second.json", {"x": "x"*80})
    with pytest.raises(ValueError, match="total_cap"):
        runner.save(tmp_path/"extra.json", {"x": "x"*40})
    runner.save(tmp_path/"terminal.json", {"status": "STOPPED_NO_RETRY"})
    with pytest.raises(FileExistsError):
        runner.save(tmp_path/"terminal.json", {})


@pytest.mark.parametrize("timeout", [False, True])
def test_worker_failure_preserved_and_no_retry(tmp_path, monkeypatch, timeout):
    monkeypatch.setattr(runner, "RUN", tmp_path)
    monkeypatch.setattr(runner, "manifest", dict)
    (tmp_path/"manifest.json").write_text("{}")
    calls = []

    class Child:
        returncode = 1
        killed = False

        def wait(self, timeout=None):
            if timeout is not None and not self.killed and calls[0]:
                raise subprocess.TimeoutExpired("FAKE", timeout)
            return self.returncode

        def poll(self):
            return None if calls[0] and not self.killed else self.returncode

        def kill(self):
            self.killed = True

    def create(*args, **kwargs):
        calls.append(timeout)
        return Child()

    monkeypatch.setattr(runner.subprocess, "Popen", create)
    with pytest.raises(SystemExit):
        runner.execute()
    result = json.loads((tmp_path/"terminal.json").read_text())
    assert result["status"] == "STOPPED_NO_RETRY" and "result_sha256" not in result
    with pytest.raises(FileExistsError):
        runner.execute()
    assert len(calls) == 1
