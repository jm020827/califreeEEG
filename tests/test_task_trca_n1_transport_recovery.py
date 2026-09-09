"""Generated process tests only; no human data, models, or registered seeds."""

import importlib.util
import json
import os
import signal
import subprocess
import sys
import time
import types
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/launch_task_trca_n1_transport_recovery.py"
RUNTIME = ROOT / "scripts/run_task_trca_n1_source39.py"
SPEC = importlib.util.spec_from_file_location("recovery_toy", SCRIPT)
recovery = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(recovery)


def command(code):
    return [sys.executable, "-c", code]


def wait_file(path, timeout=12):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if path.exists():
            return
        time.sleep(0.05)
    pytest.fail("bounded generated child did not publish " + str(path))


def test_normal_exit_files_rusage_and_exclusive(tmp_path):
    result = recovery.run_child(command("print('generated done')"), tmp_path, "child", 5)
    assert result["exit_code"] == 0 and result["watchdog_reason"] is None
    assert result["peak_rss_kib"] > 0
    assert (tmp_path / "child.stdout.log").read_text() == "generated done\n"
    assert (tmp_path / "child.exit.json").stat().st_mode & 0o777 == 0o400
    with pytest.raises(FileExistsError):
        recovery.run_child(command("raise Exception('must not run')"), tmp_path, "child", 5)


@pytest.mark.parametrize(
    "exit_code,failure,result,watchdog,expected",
    [
        (0, False, True, None, "complete"),
        (1, True, False, None, "failure"),
        (0, True, True, None, "failure"),
        (0, False, False, None, None),
        (1, False, True, None, None),
        (0, False, True, "TIME_LIMIT", None),
        (1, True, False, "TIME_LIMIT", "failure"),
    ],
)
def test_audit_selection(tmp_path, exit_code, failure, result, watchdog, expected):
    if failure:
        (tmp_path / "failure.json").write_text("{}")
    if result:
        (tmp_path / "result.json").write_text("{}")
    assert (
        recovery.audit_mode(tmp_path, {"exit_code": exit_code, "watchdog_reason": watchdog})
        == expected
    )


def test_nonzero_without_receipt_stays_inconclusive(tmp_path):
    result = recovery.run_child(command("raise SystemExit(7)"), tmp_path, "child", 5)
    assert result["exit_code"] == 7
    assert recovery.audit_mode(tmp_path, result) is None


def test_exclusive_publish_symlink_and_fsync_failure(tmp_path, monkeypatch):
    path = tmp_path / "value.json"
    recovery.publish(path, {"generated": True})
    with pytest.raises(FileExistsError):
        recovery.publish(path, {})
    alias = tmp_path / "alias.json"
    alias.symlink_to(path)
    with pytest.raises(ValueError, match="unaliased"):
        recovery.descriptor(alias)
    os.link(path, tmp_path / "hard.json")
    with pytest.raises(ValueError, match="single-link"):
        recovery.descriptor(path)

    def broken(_):
        raise OSError("generated fsync failure")

    monkeypatch.setattr(os, "fsync", broken)
    with pytest.raises(OSError, match="fsync failure"):
        recovery.publish(tmp_path / "failed.json", {})


def test_child_cleaned_if_start_receipt_write_fails(tmp_path, monkeypatch):
    spawned = []
    original = recovery.detach

    def capture(*args):
        proc = original(*args)
        spawned.append(proc)
        return proc

    def broken(*args):
        raise OSError("generated disk error")

    monkeypatch.setattr(recovery, "detach", capture)
    monkeypatch.setattr(recovery, "publish", broken)
    with pytest.raises(OSError, match="disk error"):
        recovery.run_child(command("import time; time.sleep(20)"), tmp_path, "child", 5)
    assert spawned[0].returncode == -signal.SIGKILL


def test_watchdog_grace_preserves_generated_failure(tmp_path):
    code = (
        "import signal,time,pathlib; "
        f"p=pathlib.Path({str(tmp_path / 'failure.json')!r}); "
        "signal.signal(signal.SIGINT,lambda *_:(p.write_text('{}'),exit(3))); "
        "time.sleep(20)"
    )
    result = recovery.run_child(command(code), tmp_path, "child", 0.5, grace=0.3)
    assert result["watchdog_reason"] == "TIME_LIMIT" and result["exit_code"] == 3
    assert recovery.audit_mode(tmp_path, result) == "failure"


def test_watchdog_kills_owned_group_only(tmp_path):
    sentinel = subprocess.Popen(command("import time; time.sleep(20)"), start_new_session=True)
    try:
        grandchild = (
            "import signal,time; signal.signal(signal.SIGINT,signal.SIG_IGN); time.sleep(20)"
        )
        code = (
            "import signal,subprocess,sys,time,pathlib; "
            "signal.signal(signal.SIGINT,signal.SIG_IGN); "
            f"p=subprocess.Popen([sys.executable,'-c',{grandchild!r}]); "
            f"pathlib.Path({str(tmp_path / 'grandchild.pid')!r}).write_text(str(p.pid)); "
            "time.sleep(20)"
        )
        result = recovery.run_child(command(code), tmp_path, "child", 0.5, grace=0.3)
        assert result["exit_code"] == -signal.SIGKILL
        assert sentinel.poll() is None
        pid = int((tmp_path / "grandchild.pid").read_text())
        status = Path(f"/proc/{pid}/stat")
        if status.exists():
            assert status.read_text().split()[2] == "Z"  # exited, awaiting init reaper
    finally:
        sentinel.kill()
        sentinel.wait()


def journal_code(path):
    return (
        "import importlib.util,types,time,pathlib; "
        f"spec=importlib.util.spec_from_file_location('original_journal',{str(RUNTIME)!r}); "
        "m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m); "
        f"p=pathlib.Path({str(path)!r}); "
        "j=m.EventJournal(p,types.SimpleNamespace(seq=0),time.perf_counter(),1048576); "
        "j({'event':'generated_before'}); time.sleep(0.2); j({'event':'generated_done'})"
    )


def test_actual_original_journal_detects_closed_pipe_control(tmp_path):
    proc = subprocess.Popen(
        command(journal_code(tmp_path / "control.jsonl")),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=recovery.ENV,
    )
    proc.stdout.close()
    stderr = proc.stderr.read()
    assert proc.wait(timeout=12) != 0
    assert b"BrokenPipeError" in stderr
    events = (tmp_path / "control.jsonl").read_text().splitlines()
    assert len(events) == 1  # file fsync precedes failing console print


def test_actual_journal_survives_closed_launcher_pipe_and_parent_exit(tmp_path):
    code = journal_code(tmp_path / "events.jsonl")
    worker = (
        "import importlib.util,pathlib; "
        f"s=importlib.util.spec_from_file_location('r',{str(SCRIPT)!r}); "
        "m=importlib.util.module_from_spec(s); s.loader.exec_module(m); "
        f"m.run_child({command(code)!r},pathlib.Path({str(tmp_path)!r}),'worker',12)"
    )
    launcher = (
        "import importlib.util,pathlib; "
        f"s=importlib.util.spec_from_file_location('r',{str(SCRIPT)!r}); "
        "m=importlib.util.module_from_spec(s); s.loader.exec_module(m); "
        f"m.detach({command(worker)!r},pathlib.Path({str(tmp_path)!r}),'detached'); "
        "print('consumer deliberately gone',flush=True)"
    )
    parent = subprocess.Popen(
        command(launcher), stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, env=recovery.ENV
    )
    parent.stdout.close()
    assert parent.wait(timeout=5) != 0  # actual launcher console failure
    wait_file(tmp_path / "worker.exit.json")
    receipt = json.loads((tmp_path / "worker.exit.json").read_text())
    assert receipt["exit_code"] == 0 and receipt["watchdog_reason"] is None
    events = [json.loads(x) for x in (tmp_path / "events.jsonl").read_text().splitlines()]
    assert [x["event"] for x in events] == ["generated_before", "generated_done"]
    assert [x["seq"] for x in events] == [0, 1]
    assert len((tmp_path / "worker.stdout.log").read_text().splitlines()) == 2


def test_descriptor_tamper_rejected(tmp_path):
    path = tmp_path / "value.json"
    item = recovery.publish(path, {"generated": True})
    recovery.bound(item)
    path.chmod(0o600)
    path.write_text("{}")
    path.chmod(0o400)
    with pytest.raises(ValueError, match="mismatch"):
        recovery.bound(item)


def toy_authority(tmp_path):
    folder = tmp_path / "supervisor"
    folder.mkdir()
    manifest = {"output_root": str(tmp_path / "primary"), "generated": True}
    recovery.publish(tmp_path / "human_manifest.json", manifest)
    receipt = recovery.publish(folder / "recovery.json", {"generated_toy_only": True})
    config = {"budget": {"human_seconds": 5, "audit_seconds": 5}}
    return config, tmp_path, folder, receipt, manifest


def test_single_launch_claim_and_no_console_dependency(tmp_path, monkeypatch):
    values = toy_authority(tmp_path)
    calls = []
    monkeypatch.setattr(recovery, "validated_recovery", lambda _: values)
    monkeypatch.setattr(
        recovery, "detach", lambda *args: (calls.append(args), types.SimpleNamespace(pid=123))[1]
    )
    recovery.launch("generated-toy-sha")
    with pytest.raises(FileExistsError):
        recovery.launch("generated-toy-sha")
    assert len(calls) == 1
    assert calls[0][0][-2:] == ["--receipt-sha256", "generated-toy-sha"]
    assert json.loads((values[2] / "launch_claim.json").read_text())["attempt"] == 1


@pytest.mark.parametrize(
    "mode", ["complete", "failure", "missing", "audit_failed", "audit_timeout"]
)
def test_supervisor_terminal_paths_once(tmp_path, monkeypatch, mode):
    values = toy_authority(tmp_path)
    _config, _parent, folder, receipt, manifest = values
    recovery.publish(folder / "launch_claim.json", {"recovery": receipt, "attempt": 1})
    monkeypatch.setattr(recovery, "validated_recovery", lambda _: values)
    calls = []

    def fake_child(argv, target, label, *args, **kwargs):
        calls.append(label)
        output = Path(manifest["output_root"])
        if label == "primary":
            output.mkdir()
            if mode != "missing":
                recovery.publish(
                    output / ("failure.json" if mode == "failure" else "result.json"), {}
                )
        elif mode != "audit_failed":
            recovery.publish(
                output / ("terminal_audit.json" if mode == "failure" else "cold_audit.json"),
                {
                    "status": "FIRST_FAILURE_PROVENANCE_AUDIT_PASS"
                    if mode == "failure"
                    else "COLD_INDEPENDENT_AUDIT_PASS",
                    "manifest_sha256": recovery.descriptor(tmp_path / "human_manifest.json")[
                        "sha256"
                    ],
                },
            )
        return {
            "exit_code": 1
            if (label == "primary" and mode == "failure")
            or (label == "audit" and mode == "audit_failed")
            else 0,
            "watchdog_reason": "TIME_LIMIT"
            if label == "audit" and mode == "audit_timeout"
            else None,
        }

    # Audit-failure case needs successful primary before nonzero audit exit.
    def wrapped(*args, **kwargs):
        result = fake_child(*args, **kwargs)
        if mode == "audit_failed" and args[2] == "primary":
            result["exit_code"] = 0
        return result

    monkeypatch.setattr(recovery, "run_child", wrapped)
    recovery.supervise("generated-toy-sha")
    result = json.loads((folder / "completion.json").read_text())
    assert calls == (["primary"] if mode == "missing" else ["primary", "audit"])
    assert result["audit_mode"] == (
        None if mode == "missing" else "failure" if mode == "failure" else "complete"
    )
    assert not (Path(manifest["output_root"]) / "launch_claim.json").exists()
    assert result["status"] == (
        "TERMINAL_AUDIT_AVAILABLE"
        if mode in ("complete", "failure")
        else "INFRASTRUCTURE_INCONCLUSIVE"
    )
    with pytest.raises(FileExistsError):
        recovery.supervise("generated-toy-sha")
    assert calls.count("primary") == 1


def test_manifest_only_fresh_output_may_change(tmp_path):
    old = {"output_root": "/generated/old", "profile": {"generated": True}, "code_pins": {"x": "a"}}
    expected = {**old, "output_root": str(tmp_path / "task-trca-n1-source39-primary1")}
    path = tmp_path / "human_manifest.json"
    recovery.publish(path, expected)
    recovery.validate_new_manifest({}, tmp_path, old)
    path.chmod(0o600)
    path.write_text(json.dumps({**expected, "profile": {"generated": False}}))
    path.chmod(0o400)
    with pytest.raises(ValueError, match="same-science"):
        recovery.validate_new_manifest({}, tmp_path, old)


def test_preparation_failure_prevents_same_manifest_launch(tmp_path):
    old = {"output_root": "/generated/old"}
    recovery.publish(
        tmp_path / "human_manifest.json",
        {"output_root": str(tmp_path / "task-trca-n1-source39-primary1")},
    )
    recovery.publish(tmp_path / "prepare_failure.json", {"generated": True})
    with pytest.raises(ValueError, match="failure precedence"):
        recovery.validate_new_manifest({}, tmp_path, old)


def test_final_byte_limit_checked_after_fast_exit(tmp_path, monkeypatch):
    monkeypatch.setattr(recovery, "size", lambda _: 129 * 1024**2)
    real_wait4 = os.wait4
    # Force the first observation to be the terminal branch. The old loop's
    # pre-limit break would report no breach here, so this is a sensitive test.
    monkeypatch.setattr(os, "wait4", lambda pid, _flags: real_wait4(pid, 0))
    result = recovery.run_child(command("pass"), tmp_path, "child", 5, grace=0.2)
    assert result["watchdog_reason"] == "LOG_BYTE_LIMIT"


def test_exited_leader_remaining_descendant_is_closed(tmp_path):
    grandchild = "import signal,time; signal.signal(signal.SIGINT,signal.SIG_IGN); time.sleep(20)"
    code = (
        "import subprocess,sys,pathlib; "
        f"p=subprocess.Popen([sys.executable,'-c',{grandchild!r}]); "
        f"pathlib.Path({str(tmp_path / 'remaining.pid')!r}).write_text(str(p.pid))"
    )
    result = recovery.run_child(command(code), tmp_path, "child", 5, grace=0.2)
    assert result["exit_code"] == 0
    assert result["watchdog_reason"] == "DESCENDANTS_AFTER_LEADER_EXIT"
    child_status = Path("/proc") / (tmp_path / "remaining.pid").read_text() / "stat"
    deadline = time.monotonic() + 1
    while (
        child_status.exists()
        and child_status.read_text().split()[2] != "Z"
        and time.monotonic() < deadline
    ):
        time.sleep(0.02)
    assert not child_status.exists() or child_status.read_text().split()[2] == "Z"
