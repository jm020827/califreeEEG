"""Register and run exactly one approved source39-only channel-margin probe."""

from __future__ import annotations

import argparse
import json
import os
import resource
import signal
import subprocess
from pathlib import Path

import launch_n1_metadata_generated_efficacy as lifecycle

from cfeg.analysis import source_channel_margin_runtime as runtime

ROOT = runtime.ROOT
SCRIPT = ROOT / "scripts/launch_source_channel_margin_probe.py"
PYTHON = ROOT / ".venv/bin/python"
LIMITS = {"rss_limit_bytes": 4 * 1024**3, "output_limit_bytes": 1024**3}


def prepare(parent, tests):
    parent, tests = runtime.canonical(parent), runtime.canonical(tests)
    if parent.parent != ROOT.parent or not parent.name.startswith("channel-margin-probe-v1-"):
        raise ValueError("dedicated source probe output parent required")
    if list(parent.iterdir()):
        raise ValueError("fresh empty output required")
    if lifecycle.git("diff", "--name-only") or lifecycle.git("diff", "--cached", "--name-only"):
        raise ValueError("tracked code must be committed before registration")
    report = json.loads(tests.read_text())
    commit = lifecycle.git("rev-parse", "HEAD")
    if report.get("status") != "PASS" or report.get("code_commit") != commit:
        raise ValueError("passing current-code test report required")
    if runtime.digest(ROOT / runtime.CONFIG)["sha256"] != runtime.CONFIG_SHA:
        raise ValueError("frozen science changed")
    names = runtime.code_names()
    if str(SCRIPT.relative_to(ROOT)) not in names:
        raise ValueError("launcher not tracked")
    registration = {
        "schema": runtime.SCHEMA,
        "utc": runtime.utc(),
        "human_execution_authorized": True,
        "authority": "2026-09-10 KST user 응 시작, accepting exact preceding source39-only120fit/oneaudit proposal",
        "parent": str(parent),
        "repository": str(ROOT),
        "code_commit": commit,
        "code_pins": {name: runtime.digest(ROOT / name) for name in sorted(set(names))},
        "runtime_versions": lifecycle.runtime_versions(),
        "source_ids": list(runtime.archive.SOURCE_IDS),
        "config_sha256": runtime.CONFIG_SHA,
        "inputs": runtime.input_envelope(),
        "tests": {"path": str(tests), **runtime.digest(tests)},
        "attempts": {"primary": 1, "audit": 1, "retries": 0},
        "limits": {**LIMITS, "primary_seconds": 1800, "audit_seconds": 900},
        "numeric_scope": "support0-2/block5/N250,M0-2 only; one source extraction, no query/FULL/A0/held60",
    }
    runtime.publish(parent / "registration.json", registration)
    (parent / "data").mkdir()
    (parent / "logs").mkdir()
    return runtime.digest(parent / "registration.json")["sha256"]


def validate(parent, sha):
    result = runtime.validate_registration(parent, sha)
    if result["runtime_versions"] != lifecycle.runtime_versions():
        raise ValueError("runtime versions changed")
    if result["limits"] != {**LIMITS, "primary_seconds": 1800, "audit_seconds": 900}:
        raise ValueError("runtime budget changed")
    return result


def supervise(parent, sha):
    validate(parent, sha)
    runtime.publish(
        parent / "supervisor.claim.json",
        {"utc": runtime.utc(), "pid": os.getpid(), "registration_sha256": sha},
    )
    primary = audit = None
    terminal, error = "EXECUTION_FAILURE", None
    try:
        lifecycle.await_launch_receipt(parent, sha)
        args = [str(PYTHON), str(SCRIPT)]
        suffix = ["--parent", str(parent), "--sha", sha]
        primary = lifecycle.run_child([*args, "primary", *suffix], parent, "primary", 1800, LIMITS)
        validate(parent, sha)
        failure_path = parent / "data/failure.json"
        if primary["exit_code"] != 0 and failure_path.exists():
            failure = json.loads(failure_path.read_text())
            if failure.get("terminal") == "INVALID_INPUT":
                terminal = "INVALID_INPUT"
        if (
            primary["exit_code"] == 0
            and primary["watchdog_reason"] is None
            and not primary["process_group_remaining"]
        ):
            terminal = "AUDIT_FAILURE"
            audit = lifecycle.run_child([*args, "audit", *suffix], parent, "audit", 900, LIMITS)
            validate(parent, sha)
            if (
                audit["exit_code"] == 0
                and audit["watchdog_reason"] is None
                and not audit["process_group_remaining"]
            ):
                a = json.loads((parent / "data/audit.json").read_text())
                if a["status"] == "PASS":
                    terminal = json.loads((parent / "data/result.json").read_text())["terminal"]
    except BaseException as exc:  # noqa: BLE001 -- preserve terminal receipt after child cleanup
        error = {"type": type(exc).__name__, "message": str(exc)}
    runtime.publish(
        parent / "completion.json",
        {
            "schema": "cfeg.source39-channel-margin-probe.completion.v1",
            "utc": runtime.utc(),
            "registration_sha256": sha,
            "terminal": terminal,
            "primary": primary,
            "audit": audit,
            "error": error,
            "output_bytes": lifecycle.size(parent),
            "artifacts": {
                str(p.relative_to(parent)): runtime.digest(p)
                for p in sorted((parent / "data").iterdir())
                if p.is_file()
            },
        },
    )


def launch(parent, sha):
    validate(parent, sha)
    runtime.publish(
        parent / "launch.claim.json", {"utc": runtime.utc(), "registration_sha256": sha}
    )
    process = lifecycle.detach(
        [str(PYTHON), str(SCRIPT), "supervise", "--parent", str(parent), "--sha", sha],
        parent / "logs",
        "supervisor",
    )
    try:
        runtime.publish(
            parent / "supervisor.start.json",
            {"pid": process.pid, "pgid": process.pid, "utc": runtime.utc()},
        )
    except BaseException:
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
    parser.add_argument("mode", choices=("prepare", "launch", "supervise", "primary", "audit"))
    parser.add_argument("--parent", required=True, type=Path)
    parser.add_argument("--sha")
    parser.add_argument("--tests", type=Path)
    args = parser.parse_args()
    if args.mode == "prepare":
        if args.tests is None:
            parser.error("prepare requires --tests")
        print(prepare(args.parent, args.tests))
    elif not args.sha:
        parser.error("execution modes require --sha")
    elif args.mode == "launch":
        print(launch(args.parent, args.sha))
    elif args.mode == "supervise":
        supervise(args.parent, args.sha)
    else:
        validate(args.parent, args.sha)
        resource.setrlimit(resource.RLIMIT_AS, (8 * 1024**3, 8 * 1024**3))
        resource.setrlimit(resource.RLIMIT_FSIZE, (1024**3, 1024**3))
        (runtime.run_primary if args.mode == "primary" else runtime.run_audit)(
            args.parent, args.sha
        )


if __name__ == "__main__":
    main()
