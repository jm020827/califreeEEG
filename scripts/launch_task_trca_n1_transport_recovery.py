"""One explicitly authorized, file-backed, detached same-science recovery.

No model/data imports. The original36 scientific/runtime pins remain unchanged.
Only this module owns child lifetimes; it never fabricates producer receipts.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import signal
import stat
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/analysis/task_trca_n1_transport_recovery_r1.json"
CONFIG_SHA = "29e3fb1e7896c61d74c2299ba75c34a3c2b39d0b810d8f3dcaab1adcec5082a6"
PYTHON = ROOT / ".venv/bin/python"
SCRIPT = ROOT / "scripts/launch_task_trca_n1_transport_recovery.py"
TEST = ROOT / "tests/test_task_trca_n1_transport_recovery.py"
ENV = {
    **os.environ,
    **dict.fromkeys(
        ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "PYTHONDONTWRITEBYTECODE"),
        "1",
    ),
}


def now():
    return datetime.now(timezone.utc).isoformat()


def require(ok, message):
    if not ok:
        raise ValueError(message)


def canonical(path):
    path = Path(path)
    require(
        path.is_absolute() and path.resolve() == path and not path.is_symlink(),
        "canonical unaliased path required",
    )
    return path


def descriptor(path, *, immutable=False):
    path = canonical(path)
    with os.fdopen(os.open(path, os.O_RDONLY | os.O_NOFOLLOW), "rb") as stream:
        before = os.fstat(stream.fileno())
        require(stat.S_ISREG(before.st_mode) and before.st_nlink == 1, "regular single-link file")
        require(not immutable or stat.S_IMODE(before.st_mode) == 0o400, "immutable0400 required")
        digest = hashlib.sha256()
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)

        def identity(info):
            return (
                info.st_dev,
                info.st_ino,
                info.st_size,
                info.st_mtime_ns,
                info.st_ctime_ns,
                info.st_nlink,
            )

        require(
            identity(os.fstat(stream.fileno())) == identity(before)
            and identity(path.stat()) == identity(before),
            "file changed",
        )
    return {"path": str(path), "sha256": digest.hexdigest(), "bytes": before.st_size}


def bound(item, *, immutable=True):
    actual = descriptor(item["path"], immutable=immutable)
    require(all(actual[k] == v for k, v in item.items()), "bound descriptor mismatch")
    return actual


def read_json(path):
    return json.loads(Path(path).read_text())


def publish(path, value):
    path = canonical(path)
    with path.open("x") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    path.chmod(0o400)
    return descriptor(path, immutable=True)


def detach(argv, folder, label):
    """No child descriptor references the caller's console pipe or terminal."""
    folder = canonical(folder)
    require(folder.is_dir(), "existing exact log directory")
    with (
        (folder / (label + ".stdout.log")).open("xb") as stdout,
        (folder / (label + ".stderr.log")).open("xb") as stderr,
    ):
        return subprocess.Popen(
            argv,
            cwd=ROOT,
            env=ENV,
            stdin=subprocess.DEVNULL,
            stdout=stdout,
            stderr=stderr,
            close_fds=True,
            start_new_session=True,
        )


def size(path):
    return sum(p.lstat().st_size for p in path.rglob("*") if p.is_file())


def group_exists(pgid):
    try:
        os.killpg(pgid, 0)
        return True
    except ProcessLookupError:
        return False


def run_child(argv, folder, label, seconds, *, grace=10, output_parent=None):
    """Only wait4 reaps this child; watchdog targets its freshly created PGID."""
    began = time.monotonic()
    proc = detach(argv, folder, label)
    interrupted_at = None
    reason = None
    usage = None
    status = None
    try:
        publish(
            folder / (label + ".start.json"),
            {
                "pid": proc.pid,
                "pgid": proc.pid,
                "time": now(),
                "argv": list(argv),
                "limit_seconds": seconds,
                "grace_seconds": grace,
            },
        )
        while True:
            if status is None:
                pid, candidate_status, candidate_usage = os.wait4(proc.pid, os.WNOHANG)
                if pid:
                    status, usage = candidate_status, candidate_usage
                    proc.returncode = os.waitstatus_to_exitcode(status)
            elapsed = time.monotonic() - began
            if interrupted_at is None:
                if elapsed >= seconds:
                    reason = "TIME_LIMIT"
                elif size(folder) > 128 * 1024**2:
                    reason = "LOG_BYTE_LIMIT"
                elif output_parent is not None and size(output_parent) > 16 * 1024**3:
                    reason = "TOTAL_BYTE_LIMIT"
                if status is not None:
                    if not group_exists(proc.pid):
                        break  # Bounds above also apply to fast/final exits.
                    reason = reason or "DESCENDANTS_AFTER_LEADER_EXIT"
                if reason:
                    interrupted_at = time.monotonic()
                    try:
                        os.killpg(proc.pid, signal.SIGINT)
                    except ProcessLookupError:
                        pass
            elif time.monotonic() - interrupted_at >= grace:
                # Also terminates descendants ignoring SIGINT after leader exit.
                try:
                    os.killpg(proc.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                if status is None:
                    _, status, usage = os.wait4(proc.pid, 0)
                    proc.returncode = os.waitstatus_to_exitcode(status)
                break
            time.sleep(0.2)
    except BaseException:
        # Only the child group created by this function is in scope.
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        if status is None:
            _, status, usage = os.wait4(proc.pid, 0)
            proc.returncode = os.waitstatus_to_exitcode(status)
        raise
    finally:
        for suffix in ("stdout.log", "stderr.log"):
            path = folder / (label + "." + suffix)
            with path.open("rb") as stream:
                os.fsync(stream.fileno())
            path.chmod(0o400)
    result = {
        "pid": proc.pid,
        "exit_code": proc.returncode,
        "watchdog_reason": reason,
        "elapsed_seconds": time.monotonic() - began,
        "peak_rss_kib": usage.ru_maxrss if usage is not None else None,
        "rss_source": "Linux wait4 child rusage; not supervisor-wide sampled RSS",
        "user_seconds": usage.ru_utime if usage is not None else None,
        "system_seconds": usage.ru_stime if usage is not None else None,
        "time": now(),
        "stdout": descriptor(folder / (label + ".stdout.log")),
        "stderr": descriptor(folder / (label + ".stderr.log")),
    }
    publish(folder / (label + ".exit.json"), result)
    return result


def audit_mode(output, primary):
    if (output / "failure.json").exists():
        return "failure"
    if (
        primary["exit_code"] == 0
        and primary["watchdog_reason"] is None
        and (output / "result.json").exists()
    ):
        return "complete"
    return None


def configuration():
    require(descriptor(CONFIG)["sha256"] == CONFIG_SHA, "frozen recovery authority changed")
    config = read_json(CONFIG)
    parent = canonical(config["parent"])
    require(
        parent.parent == Path("/home/whwovy")
        and parent.name.startswith("task-trca-n1-source39-recovery-r1-"),
        "exact scope",
    )
    for name in ("old_failure", "old_terminal", "old_manifest"):
        bound(config[name])
    old = read_json(config["old_manifest"]["path"])
    for name, pin in old["code_pins"].items():
        require(descriptor(ROOT / name)["sha256"] == pin, "original science/runtime pin: " + name)
    return config, parent, old


def validate_new_manifest(config, parent, old):
    path = parent / "human_manifest.json"
    desc = descriptor(path, immutable=True)
    manifest = read_json(path)
    # Exact old manifest apart from this fresh canonical output. This binds all
    # profile, input, code, budget and prior generated/resource proof fields.
    expected = {**old, "output_root": str(parent / "task-trca-n1-source39-primary1")}
    require(manifest == expected, "same-science manifest except fresh output path")
    require(
        not any((parent / name).exists() for name in ("prepare_failure.json", "prep_failure.json")),
        "preparation failure precedence",
    )
    return desc, manifest


def freeze(test_report):
    config, parent, old = configuration()
    for cmd in (["git", "diff", "--quiet"], ["git", "diff", "--cached", "--quiet"]):
        subprocess.run(cmd, cwd=ROOT, check=True)
    manifest_desc, manifest = validate_new_manifest(config, parent, old)
    require(not Path(manifest["output_root"]).exists(), "no previous recovery output")
    report_desc = descriptor(test_report, immutable=True)
    report = read_json(test_report)
    require(
        report["status"] == "TRANSPORT_TESTS_PASS"
        and report["seconds"] <= 300
        and report["allocated_bytes"] <= 128 * 1024**2,
        "bounded passing transport proof",
    )
    bound(report["junit"])
    require(
        report["script_sha256"] == descriptor(SCRIPT)["sha256"]
        and report["tests_sha256"] == descriptor(TEST)["sha256"],
        "tested exact new code",
    )
    folder = parent / "supervisor"
    folder.mkdir()  # Exclusive: a second freeze is never a recovery retry.
    publish(
        folder / "recovery.json",
        {
            "recovery_id": config["recovery_id"],
            "time": now(),
            "config": descriptor(CONFIG),
            "script": descriptor(SCRIPT),
            "tests": descriptor(TEST),
            "test_report": report_desc,
            "manifest": manifest_desc,
            "old_failure": config["old_failure"],
            "old_terminal": config["old_terminal"],
            "old_manifest": config["old_manifest"],
            "original_code_pins": old["code_pins"],
            "human_attempts_authorized": 1,
            "terminal_audits_max": 1,
            "new_human_numeric_reads": False,
            "source_revision": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
            ).strip(),
        },
    )


def validated_recovery(expected_sha):
    config, parent, old = configuration()
    manifest_desc, manifest = validate_new_manifest(config, parent, old)
    folder = parent / "supervisor"
    receipt_desc = descriptor(folder / "recovery.json", immutable=True)
    require(receipt_desc["sha256"] == expected_sha, "exact frozen recovery SHA")
    receipt = read_json(receipt_desc["path"])
    for name in ("script", "tests", "config"):
        bound(receipt[name], immutable=False)
    for name in ("test_report", "old_failure", "old_terminal", "old_manifest", "manifest"):
        bound(receipt[name])
    require(
        receipt["manifest"] == manifest_desc
        and receipt["original_code_pins"] == old["code_pins"]
        and receipt["recovery_id"] == config["recovery_id"],
        "recovery binding",
    )
    return config, parent, folder, receipt_desc, manifest


def launch(expected_sha):
    _, parent, folder, receipt_desc, manifest = validated_recovery(expected_sha)
    require(not Path(manifest["output_root"]).exists(), "fresh output required")
    publish(folder / "launch_claim.json", {"recovery": receipt_desc, "time": now(), "attempt": 1})
    proc = detach(
        [str(PYTHON), str(SCRIPT), "supervise", "--receipt-sha256", expected_sha],
        folder,
        "supervisor",
    )
    publish(
        folder / "launcher.json",
        {
            "pid": proc.pid,
            "pgid": proc.pid,
            "time": now(),
            "parent": str(parent),
            "recovery": receipt_desc,
        },
    )
    # Deliberately no console print: a closed caller pipe is immaterial.


def supervise(expected_sha):
    config, parent, folder, receipt_desc, manifest = validated_recovery(expected_sha)
    claim = read_json(folder / "launch_claim.json")
    require(claim["recovery"] == receipt_desc and claim["attempt"] == 1, "one launch claim")
    publish(
        folder / "supervisor_start.json",
        {
            "pid": os.getpid(),
            "pgid": os.getpgrp(),
            "time": now(),
            "recovery": receipt_desc,
        },
    )
    output = Path(manifest["output_root"])
    manifest_path = parent / "human_manifest.json"
    manifest_sha = descriptor(manifest_path)["sha256"]
    try:
        require(not output.exists(), "one fresh primary")
        primary = run_child(
            [
                str(PYTHON),
                str(ROOT / "scripts/run_task_trca_n1_source39.py"),
                "--manifest",
                str(manifest_path),
                "--manifest-sha256",
                manifest_sha,
            ],
            folder,
            "primary",
            config["budget"]["human_seconds"],
            output_parent=parent,
        )
        mode = audit_mode(output, primary)
        cold = None
        terminal = None
        if mode is not None:
            publish(
                folder / "audit_claim.json",
                {"mode": mode, "time": now(), "attempt": 1, "recovery": receipt_desc},
            )
            cold = run_child(
                [
                    str(PYTHON),
                    str(ROOT / "scripts/audit_task_trca_n1_source39.py"),
                    "--output",
                    str(output),
                    "--manifest",
                    str(manifest_path),
                    "--manifest-sha256",
                    manifest_sha,
                    *(["--failure-only"] if mode == "failure" else []),
                ],
                folder,
                "audit",
                config["budget"]["audit_seconds"],
                output_parent=parent,
            )
            terminal_path = output / (
                "terminal_audit.json" if mode == "failure" else "cold_audit.json"
            )
            if terminal_path.exists():
                terminal = descriptor(terminal_path, immutable=True)
                scientific_receipt = read_json(terminal_path)
                require(
                    scientific_receipt["manifest_sha256"] == manifest_sha
                    and scientific_receipt["status"]
                    == (
                        "FIRST_FAILURE_PROVENANCE_AUDIT_PASS"
                        if mode == "failure"
                        else "COLD_INDEPENDENT_AUDIT_PASS"
                    ),
                    "bound scientific audit identity",
                )
        # Rebind unchanged source and recovery authority after execution.
        validated_recovery(expected_sha)
        status = (
            "TERMINAL_AUDIT_AVAILABLE"
            if terminal and cold["exit_code"] == 0 and cold["watchdog_reason"] is None
            else "INFRASTRUCTURE_INCONCLUSIVE"
        )
        publish(
            folder / "completion.json",
            {
                "status": status,
                "time": now(),
                "recovery": receipt_desc,
                "primary": primary,
                "audit_mode": mode,
                "audit": cold,
                "terminal": terminal,
                "efficacy": "READ_VERIFIED_COMPLETE_COLD"
                if status == "TERMINAL_AUDIT_AVAILABLE" and mode == "complete"
                else "NOT_EVALUATED",
                "no_retries": True,
                "old_failure_preserved": True,
                "limitations": "Transport receipt; scientific result resides in bound independent cold. No host/cgroup survival guarantee.",
            },
        )
    except BaseException as exc:
        publish(
            folder / "supervisor_failure.json",
            {
                "status": "INFRASTRUCTURE_INCONCLUSIVE",
                "error_type": type(exc).__name__,
                "error": str(exc),
                "time": now(),
                "recovery": receipt_desc,
                "efficacy": "NOT_EVALUATED",
                "no_retry": True,
            },
        )
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("freeze", "launch", "supervise"))
    parser.add_argument("--test-report", type=Path)
    parser.add_argument("--receipt-sha256")
    args = parser.parse_args()
    if args.mode == "freeze":
        require(args.test_report is not None, "test report required")
        freeze(args.test_report)
    else:
        require(args.receipt_sha256 is not None, "receipt SHA required")
        (launch if args.mode == "launch" else supervise)(args.receipt_sha256)


if __name__ == "__main__":
    main()
