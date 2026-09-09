"""File-backed, code-pinned, single-attempt generated efficacy lifecycle."""

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
CONFIG = ROOT / "configs/n1_metadata_generated_efficacy_v1.json"
CONFIG_SHA = "2c279918e841663976c4b935ea42cb7455cda0a577fcea4e9e309a85f348e890"
PYTHON = ROOT / ".venv/bin/python"
SCRIPT = ROOT / "scripts/launch_n1_metadata_generated_efficacy.py"
ENV = {
    **os.environ,
    "PYTHONPATH": str(ROOT / "src"),
    "PYTHONDONTWRITEBYTECODE": "1",
    "OMP_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "CUDA_VISIBLE_DEVICES": "",
}


def utc():
    return datetime.now(timezone.utc).isoformat()


def canonical(path):
    path = Path(path)
    if not path.is_absolute() or path.resolve() != path or path.is_symlink():
        raise ValueError("exact unaliased absolute path required")
    return path


def digest(path):
    path = canonical(path)
    with os.fdopen(os.open(path, os.O_RDONLY | os.O_NOFOLLOW), "rb") as stream:
        info = os.fstat(stream.fileno())
        if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
            raise ValueError("regular single-link file required")
        sha = hashlib.sha256()
        for block in iter(lambda: stream.read(1024**2), b""):
            sha.update(block)
        after = os.fstat(stream.fileno())
        if (info.st_size, info.st_mtime_ns, info.st_ctime_ns) != (
            after.st_size,
            after.st_mtime_ns,
            after.st_ctime_ns,
        ):
            raise ValueError("file changed while hashing")
    return {"sha256": sha.hexdigest(), "bytes": info.st_size}


def publish(path, data):
    with canonical(path).open("x") as stream:
        json.dump(data, stream, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    path.chmod(0o400)


def git(*args):
    return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()


def runtime_versions():
    code = (
        "import importlib.metadata as m,json,platform; "
        "print(json.dumps({'python':platform.python_version(),"
        "'packages':{k:m.version(k) for k in ('numpy','scipy','torch')}}))"
    )
    return json.loads(
        subprocess.check_output([str(PYTHON), "-c", code], cwd=ROOT, env=ENV, text=True)
    )


def prepare(parent, test_report):
    parent = canonical(parent)
    if parent.parent != ROOT.parent or not parent.name.startswith("n1-generated-efficacy-v1-"):
        raise ValueError("dedicated generated-study parent required")
    if list(parent.iterdir()):
        raise ValueError("new empty parent required")
    if digest(CONFIG)["sha256"] != CONFIG_SHA:
        raise ValueError("prospective configuration changed")
    if git("diff", "--name-only") or git("diff", "--cached", "--name-only"):
        raise ValueError("tracked code must be committed before registration")
    test_report = canonical(test_report)
    report = json.loads(test_report.read_text())
    if report.get("status") != "PASS" or report.get("code_commit") != git("rev-parse", "HEAD"):
        raise ValueError("passing test report must bind current code commit")
    names = git("ls-files", "src/**/*.py", "scripts/*.py", "tests/*.py").splitlines()
    names += [str(CONFIG.relative_to(ROOT)), "docs/n1_metadata_generated_efficacy_v1_design.md"]
    if not all(str(p.relative_to(ROOT)) in names for p in (SCRIPT, CONFIG)):
        raise ValueError("execution files must be tracked")
    code_pins = {name: digest(ROOT / name) for name in sorted(set(names))}
    manifest = {
        "schema": "n1-metadata-generated-efficacy-registration-v1",
        "utc": utc(),
        "parent": str(parent),
        "repository": str(ROOT),
        "code_commit": git("rev-parse", "HEAD"),
        "config": json.loads(CONFIG.read_text()),
        "config_sha256": CONFIG_SHA,
        "code_pins": code_pins,
        "runtime_versions": runtime_versions(),
        "test_report": {"path": str(test_report), **digest(test_report)},
        "primary_attempts": 1,
        "audit_attempts": 1,
        "retries": 0,
        "environment": {
            k: ENV[k]
            for k in (
                "PYTHONPATH",
                "PYTHONDONTWRITEBYTECODE",
                "OMP_NUM_THREADS",
                "OPENBLAS_NUM_THREADS",
                "MKL_NUM_THREADS",
                "CUDA_VISIBLE_DEVICES",
            )
        },
        "scientific_scope": "generated-only constructive capacity control; no human utility claim",
    }
    publish(parent / "registration.json", manifest)
    (parent / "data").mkdir()
    (parent / "logs").mkdir()
    return digest(parent / "registration.json")["sha256"]


def validate(parent, expected_sha):
    parent = canonical(parent)
    if digest(parent / "registration.json")["sha256"] != expected_sha:
        raise ValueError("registration identity mismatch")
    manifest = json.loads((parent / "registration.json").read_text())
    if manifest["parent"] != str(parent) or manifest["repository"] != str(ROOT):
        raise ValueError("wrong exact parent/repository")
    if manifest["config_sha256"] != CONFIG_SHA or digest(CONFIG)["sha256"] != CONFIG_SHA:
        raise ValueError("config changed")
    if runtime_versions() != manifest["runtime_versions"]:
        raise ValueError("runtime package versions changed")
    for name, pin in manifest["code_pins"].items():
        if Path(name).is_absolute() or ".." in Path(name).parts or digest(ROOT / name) != pin:
            raise ValueError("code pin changed: " + name)
    report = manifest["test_report"]
    if digest(Path(report["path"])) != {k: report[k] for k in ("sha256", "bytes")}:
        raise ValueError("test report changed")
    return manifest


def detach(argv, logs, label):
    with (
        (logs / (label + ".stdout.log")).open("xb") as stdout,
        (logs / (label + ".stderr.log")).open("xb") as stderr,
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


def group_exists(pid):
    try:
        os.killpg(pid, 0)
        return True
    except ProcessLookupError:
        return False


def run_child(argv, parent, label, seconds, config, *, grace=10):
    logs = parent / "logs"
    started = time.monotonic()
    process = detach(argv, logs, label)
    reason, interrupted, status, usage = None, None, None, None
    try:
        publish(
            logs / (label + ".start.json"),
            {
                "pid": process.pid,
                "pgid": process.pid,
                "utc": utc(),
                "argv": list(argv),
                "limit_seconds": seconds,
                "grace_seconds": grace,
            },
        )
        while True:
            if status is None:
                pid, s, u = os.wait4(process.pid, os.WNOHANG)
                if pid:
                    status, usage = s, u
                    process.returncode = os.waitstatus_to_exitcode(s)
            if interrupted is None:
                if time.monotonic() - started >= seconds:
                    reason = "TIME_LIMIT"
                elif size(parent) > config["output_limit_bytes"]:
                    reason = "OUTPUT_LIMIT"
                elif size(logs) > 128 * 1024**2:
                    reason = "LOG_LIMIT"
                try:
                    rss = next(
                        int(row.split()[1]) * 1024
                        for row in Path(f"/proc/{process.pid}/status").read_text().splitlines()
                        if row.startswith("VmRSS:")
                    )
                    if rss > config["rss_limit_bytes"]:
                        reason = "RSS_LIMIT"
                except (FileNotFoundError, ProcessLookupError, StopIteration):
                    pass
                if usage is not None and usage.ru_maxrss * 1024 > config["rss_limit_bytes"]:
                    reason = "RSS_LIMIT"
                if status is not None:
                    if not group_exists(process.pid):
                        break
                    reason = reason or "DESCENDANTS_AFTER_LEADER_EXIT"
                if reason:
                    interrupted = time.monotonic()
                    try:
                        os.killpg(process.pid, signal.SIGINT)
                    except ProcessLookupError:
                        pass
            elif time.monotonic() - interrupted >= grace:
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                if status is None:
                    _, status, usage = os.wait4(process.pid, 0)
                    process.returncode = os.waitstatus_to_exitcode(status)
                break
            time.sleep(0.2)
    except BaseException:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        if status is None:
            _, status, usage = os.wait4(process.pid, 0)
            process.returncode = os.waitstatus_to_exitcode(status)
        raise
    finally:
        for suffix in ("stdout.log", "stderr.log"):
            (logs / (label + "." + suffix)).chmod(0o400)
    receipt = {
        "pid": process.pid,
        "exit_code": process.returncode,
        "watchdog_reason": reason,
        "elapsed_seconds": time.monotonic() - started,
        "peak_rss_kib": usage.ru_maxrss if usage else None,
        "user_seconds": usage.ru_utime if usage else None,
        "system_seconds": usage.ru_stime if usage else None,
        "process_group_remaining": group_exists(process.pid),
        "utc": utc(),
    }
    publish(logs / (label + ".exit.json"), receipt)
    return receipt


def await_launch_receipt(parent, sha, *, seconds=5):
    """No primary spawn before launcher has durably sealed our exact PID receipt."""
    deadline = time.monotonic() + seconds
    path = parent / "supervisor.start.json"
    while time.monotonic() < deadline:
        if path.exists():
            info = path.lstat()
            if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
                raise ValueError("launch receipt must be a regular single-link file")
            if stat.S_IMODE(info.st_mode) == 0o400:
                digest(path)
                receipt = json.loads(path.read_text())
                claim = json.loads((parent / "launch.claim.json").read_text())
                if (
                    receipt["pid"] != os.getpid()
                    or receipt["pgid"] != os.getpid()
                    or claim["registration_sha256"] != sha
                ):
                    raise ValueError("launch receipt/claim identity mismatch")
                return
        time.sleep(0.02)
    raise TimeoutError("no durable launch receipt; primary not started")


def supervise(parent, sha):
    manifest = validate(parent, sha)
    publish(
        parent / "supervisor.claim.json",
        {"utc": utc(), "pid": os.getpid(), "registration_sha256": sha},
    )
    config = manifest["config"]
    args = [str(PYTHON), "-m", "cfeg.analysis.n1_metadata_generated_efficacy_run"]
    suffix = ["--folder", str(parent / "data"), "--config", str(CONFIG)]
    primary, audited = None, None
    terminal, error = "EFFICACY_NOT_EVALUATED", None
    try:
        await_launch_receipt(parent, sha)
        primary = run_child(
            [*args, "primary", *suffix], parent, "primary", config["primary_seconds"], config
        )
        validate(parent, sha)
        if (
            primary["exit_code"] == 0
            and primary["watchdog_reason"] is None
            and not primary["process_group_remaining"]
        ):
            audited = run_child(
                [*args, "audit", *suffix], parent, "audit", config["audit_seconds"], config
            )
            validate(parent, sha)
            if (
                audited["exit_code"] == 0
                and audited["watchdog_reason"] is None
                and not audited["process_group_remaining"]
            ):
                ar = json.loads((parent / "data/audit.json").read_text())
                if ar["status"] == "PASS":
                    terminal = json.loads((parent / "data/result.json").read_text())["terminal"]
    except BaseException as exc:  # noqa: BLE001 -- preserve terminal receipt even on SIGINT
        error = {"type": type(exc).__name__, "message": str(exc)}
    completion = {
        "schema": "n1-metadata-generated-efficacy-completion-v1",
        "terminal": terminal,
        "primary": primary,
        "audit": audited,
        "error": error,
        "utc": utc(),
        "registration_sha256": sha,
        "output_bytes": size(parent),
        "artifacts": {
            str(p.relative_to(parent)): digest(p)
            for p in sorted((parent / "data").iterdir())
            if p.is_file()
        },
    }
    publish(parent / "completion.json", completion)


def launch(parent, sha):
    validate(parent, sha)
    publish(parent / "launch.claim.json", {"utc": utc(), "registration_sha256": sha})
    process = detach(
        [str(PYTHON), str(SCRIPT), "supervise", "--parent", str(parent), "--sha", sha],
        parent / "logs",
        "supervisor",
    )
    try:
        publish(
            parent / "supervisor.start.json",
            {"pid": process.pid, "pgid": process.pid, "utc": utc()},
        )
    except BaseException:
        # Before the sealed handshake the supervisor cannot spawn a primary.
        # SIGINT also invokes its owned-child cleanup if publication just completed.
        try:
            os.killpg(process.pid, signal.SIGINT)
        except ProcessLookupError:
            pass
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            process.wait()
        raise
    return process.pid


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("prepare", "launch", "supervise"))
    parser.add_argument("--parent", type=Path, required=True)
    parser.add_argument("--test-report", type=Path)
    parser.add_argument("--sha")
    args = parser.parse_args()
    if args.mode == "prepare":
        if args.test_report is None:
            parser.error("prepare requires --test-report")
        print(prepare(args.parent, args.test_report))
    elif args.sha is None:
        parser.error("launch/supervise require --sha")
    elif args.mode == "launch":
        print(launch(args.parent, args.sha))
    else:
        supervise(args.parent, args.sha)


if __name__ == "__main__":
    main()
