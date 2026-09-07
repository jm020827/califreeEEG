"""Artificial numerical and serialized lifecycle checks; no human artifact access."""

import copy
import hashlib
import json
import os
import subprocess
from pathlib import Path

import numpy as np
import pytest
from test_native_subset_known_zero_source import load, synthetic_parent

H = load("native_subset_headroom")


def fixture_predictions():
    truth = np.tile(np.arange(12), 5)
    full = np.where(truth < 6, truth, (truth + 1) % 12)
    pred = np.tile(full, (4, 1))
    pred[1, np.isin(truth, [6, 7])] = truth[np.isin(truth, [6, 7])]
    pred[2, np.isin(truth, [8, 9])] = truth[np.isin(truth, [8, 9])]
    pred[3, truth < 2] = (truth[truth < 2] + 1) % 12
    pred[3, truth == 10] = 10
    pred[3, truth == 11] = 1
    q = np.where(truth == 6, 1, np.where(truth == 0, 3, 0))
    qm = np.where(np.isin(truth, [6, 7]), 1, np.where(truth == 8, 2, 0))
    return pred, truth, q, qm


def test_scalar_oracle_not_best_expert_and_disagreement_not_recovery():
    pred, truth, q, qm = fixture_predictions()
    before = [x.copy() for x in (pred, truth, q, qm)]
    r = H.cell_metrics(pred, truth, q, qm)
    scalar = sum(any(int(pred[c, j]) == int(truth[j]) for c in range(4)) for j in range(60))
    assert scalar == r["oracle_correct"] == 55
    assert (pred == truth).sum(axis=1).tolist() == [30, 40, 40, 25]
    assert r["disagreement"] == 40 and r["disagreement_all_wrong"] == 5
    assert r["q_correct"] == 30 and r["qm_correct"] == 45
    assert r["recoverable_q"] == 25 and r["recoverable_qm"] == 10
    assert r["qm_repairs_q"] == 15 and r["qm_damages_q"] == 0
    for x, original in zip((pred, truth, q, qm), before):
        np.testing.assert_array_equal(x, original)


@pytest.mark.parametrize("correct", [True, False])
def test_unanimous_prediction(correct):
    truth = np.tile(np.arange(12), 5)
    pred = np.tile(truth if correct else (truth + 1) % 12, (4, 1))
    r = H.cell_metrics(pred, truth, np.zeros(60, int), np.ones(60, int))
    assert r["oracle_correct"] == 60 * correct
    assert r["recoverable_q"] == r["disagreement"] == r["qm_q_prediction_changes"] == 0


def test_permutation_duplication_extension_invariants():
    pred, truth, q, qm = fixture_predictions()
    base = H.cell_metrics(pred, truth, q, qm)
    permutation = np.array([0, 2, 3, 1])
    inverse = np.argsort(permutation)
    shuffled = H.cell_metrics(pred[permutation], truth, inverse[q], inverse[qm])
    assert shuffled == base
    duplicate = H.cell_metrics(np.concatenate([pred, pred[1:2]]), truth, q, qm)
    assert duplicate["oracle_correct"] == base["oracle_correct"]
    extended = H.cell_metrics(np.concatenate([pred, truth[None, :]]), truth, q, qm)
    assert extended["oracle_correct"] >= base["oracle_correct"]
    reversed_queries = H.cell_metrics(pred[:, ::-1], truth[::-1], q[::-1], qm[::-1])
    assert reversed_queries == base
    relabeled = H.cell_metrics((pred + 5) % 12, (truth + 5) % 12, q, qm)
    assert relabeled == base


@pytest.mark.parametrize(
    "which,bad",
    [
        (2, [False] + [0] * 59),
        (2, [4] * 60),
        (3, [-1] * 60),
        (2, [0.0] * 60),
        (3, [0] * 59),
        (1, [12] * 60),
        (1, [True] * 60),
        (0, np.zeros((4, 60), float)),
        (0, np.full((4, 60), -1)),
    ],
)
def test_invalid_arrays(which, bad):
    args = list(fixture_predictions())
    args[which] = bad
    with pytest.raises(ValueError):
        H.cell_metrics(*args)


@pytest.mark.parametrize(
    "counts,expected", [((48, 0, 0), 0), ((47, 48, 47), 3), ((47, 47, 48), 5), ((47, 47, 47), None)]
)
def test_first_observed_not_monotonic(counts, expected):
    assert H.first80(dict(zip((0, 3, 5), counts))) == expected


@pytest.fixture(scope="module")
def artificial():
    obj = synthetic_parent(fitted=False)
    return (
        obj["cache"],
        obj["parent_result"],
        obj["science"],
        obj["core"],
        load("audit_native_subset_m_source"),
    )


def test_full_artificial_grid_and_participant_aggregation(artificial):
    result = H.evaluate(*artificial)
    assert result["comparator_replay_verified"]
    assert result["replayed_ba_rows"] == 2184
    for key, count in (("cells", 624), ("summary", 18), ("cost", 9), ("attainment", 1248)):
        assert len(result[key]) == count
    for row in result["summary"]:
        assert row["metrics"]["oracle_ba"]["mean"] == 0.25
        assert row["metrics"]["oracle_ba"]["n_participants"] == 39
    assert all(r["oracle3_minus_fixed_q5"]["mean"] == pytest.approx(1 / 6) for r in result["cost"])
    text = json.dumps(result, allow_nan=False)
    assert '"actions"' not in text and '"truth"' not in text


@pytest.mark.parametrize("corruption", ["last_ba", "duplicate", "missing", "bad_action", "padding"])
def test_all_replay_precedes_any_oracle(artificial, monkeypatch, corruption):
    cache, saved, science, core, auditor = artificial
    saved = copy.deepcopy(saved)
    if corruption == "last_ba":
        next(r for r in reversed(saved["rows"]) if r["method"] == "QM")["ba"] += 0.1
    elif corruption == "duplicate":
        saved["diagnostics"][-1] = saved["diagnostics"][0]
    elif corruption == "missing":
        saved["rows"].pop()
    elif corruption == "bad_action":
        saved["diagnostics"][-1]["actions"]["QM"][-1] = True
    else:
        cache = {key: value.copy() for key, value in cache.items()}
        cache["expert_r"][0, 0, 0, 0, 5, 0, 0, 0, 0] = 0.1

    def forbidden(*_):
        pytest.fail("Oracle ran before complete replay")

    monkeypatch.setattr(H, "cell_metrics", forbidden)
    with pytest.raises(ValueError):
        H.evaluate(cache, saved, science, core, auditor)


def test_interval_uses_people_and_does_not_clip():
    values = np.arange(39) / 39
    result = H.interval(values)
    assert result["mean"] == pytest.approx(19 / 39)
    assert result["ci95"][0] < result["mean"] < result["ci95"][1]
    with pytest.raises(ValueError):
        H.interval(np.tile(values, 8))


def test_guard_and_exclusive_publication(tmp_path):
    output = tmp_path / "out"
    output.mkdir()
    plan = {
        "inputs": {"cache": {"path": str(tmp_path / "in.npz")}},
        "output_root": str(output),
        "artifacts": ["start.json", "result.json"],
    }
    guard = H.guard_for(plan)
    guard("open", (str(tmp_path / "in.npz"), "r", os.O_RDONLY))
    for event, args in [
        ("open", (str(tmp_path / "metadata.json"), "r", os.O_RDONLY)),
        ("open", (str(output / "result.json"), "w", os.O_WRONLY)),
        ("subprocess.Popen", ()),
        ("socket.connect", ()),
        ("os.remove", (str(output / "start.json"),)),
        ("os.truncate", (str(output / "start.json"), 0)),
        ("os.utime", (str(output / "start.json"),)),
        ("os.chown", (str(output / "start.json"),)),
        ("os.setxattr", (str(output / "start.json"),)),
        ("os.removexattr", (str(output / "start.json"),)),
    ]:
        with pytest.raises(ValueError):
            guard(event, args)
    path = output / "start.json"
    H.publish(path, {"test": True}, 10000)
    assert path.stat().st_mode & 0o777 == 0o400
    assert H.artifact_bytes(path, 10000)
    with pytest.raises(FileExistsError):
        H.publish(path, {}, 10000)


@pytest.mark.parametrize("failure", [None, "initial_hash", "final_hash"])
def test_serialized_execute_start_first_and_no_retry(artificial, tmp_path, failure):
    cache, saved, science, _, _ = artificial
    plan = json.loads((H.ROOT / H.PLAN_PATH).read_text())
    paths = {"cache": tmp_path / "synthetic.npz", "result": tmp_path / "synthetic.json"}
    np.savez_compressed(paths["cache"], **cache)
    saved = {
        **saved,
        **{key: plan["saved_result"][key] for key in ("study_id", "source_commit")},
        "scientific_plan_sha256": plan["science"]["sha256"],
        "status": "DEVELOPMENT_ASSESSMENT_COMPLETE",
    }
    paths["result"].write_text(json.dumps(saved))
    for key, path in paths.items():
        path.chmod(0o400)
        plan["inputs"][key] = {
            "path": str(path),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
    plan["output_root"] = str(tmp_path / "out")
    if failure == "initial_hash":
        plan["inputs"]["cache"]["sha256"] = "0" * 64
    fixture = tmp_path / "fixture.json"
    fixture.write_text(json.dumps({"plan": plan, "science": science, "failure": failure}))
    completed = subprocess.run(
        [str(H.ROOT / ".venv/bin/python"), str(Path(__file__).resolve()), str(fixture)],
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
        env={
            **os.environ,
            "OPENBLAS_NUM_THREADS": "1",
            "OMP_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
        },
    )
    output = Path(plan["output_root"])
    if failure is None:
        assert completed.returncode == 0, completed.stdout + completed.stderr
        assert sorted(p.name for p in output.iterdir()) == ["result.json", "start.json"]
        result = json.loads((output / "result.json").read_text())
        assert result["status"] == "DIAGNOSTIC_COMPLETE" and len(result["cells"]) == 624
    else:
        assert completed.returncode != 0
        assert sorted(p.name for p in output.iterdir()) == ["start.json"]
        message = "Input hash" if failure == "initial_hash" else "Input changed during diagnostic"
        assert message in completed.stderr
    second = subprocess.run(
        [str(H.ROOT / ".venv/bin/python"), str(Path(__file__).resolve()), str(fixture)],
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    assert second.returncode != 0


@pytest.mark.parametrize(
    "failure", [None, "plan", "helper", "dirty", "branch", "source", "runtime"]
)
def test_preflight_pins_git_and_runtime_without_human_reads(monkeypatch, failure):
    import sys

    monkeypatch.setattr(sys, "argv", ["diagnostic"])
    exists = Path.exists
    output = Path(json.loads((H.ROOT / H.PLAN_PATH).read_text())["output_root"])
    monkeypatch.setattr(Path, "exists", lambda p: False if p == output else exists(p))
    for key in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
        monkeypatch.setenv(key, "1")
    if failure == "plan":
        monkeypatch.setattr(H, "PLAN_SHA", "0" * 64)
    if failure == "runtime":
        monkeypatch.setenv("OPENBLAS_NUM_THREADS", "2")
    if failure == "helper":
        original_digest = H.digest
        helper = (H.ROOT / "scripts/native_subset_m_core.py").read_bytes()
        monkeypatch.setattr(
            H, "digest", lambda raw: "0" * 64 if raw == helper else original_digest(raw)
        )

    def git(command):
        args = command[3:]
        if args[0] == "branch":
            return b"wrong" if failure == "branch" else b"main"
        if args[0] == "status":
            return b"modified" if failure == "dirty" else b""
        if args[0] == "rev-parse":
            return b"synthetic-commit"
        relative = args[1].split(":", 1)[1]
        raw = (H.ROOT / relative).read_bytes()
        return raw + b"drift" if failure == "source" else raw

    monkeypatch.setattr(H.subprocess, "check_output", git)
    if failure is None:
        assert H.preflight()[3] == "synthetic-commit"
    else:
        with pytest.raises(ValueError):
            H.preflight()


if __name__ == "__main__":
    import sys

    payload = json.loads(Path(sys.argv[1]).read_text())
    modules = {
        name: load(name) for name in ("native_subset_m_core", "audit_native_subset_m_source")
    }
    H.t.ppf(0.975, 38)
    H.preflight = lambda: (
        payload["plan"],
        payload["science"],
        modules,
        "synthetic",
        "synthetic",
        {},
    )
    original_read = H.artifact_bytes
    reads = 0

    def assert_started(path, limit):
        global reads

        assert (Path(payload["plan"]["output_root"]) / "start.json").exists()
        reads += 1
        raw = original_read(path, limit)
        return raw + b"changed" if payload["failure"] == "final_hash" and reads > 2 else raw

    H.artifact_bytes = assert_started
    H.execute()
