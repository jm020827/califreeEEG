"""One development DIN-only observation; no label assignment or EEG loading."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import resource
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RUN = ROOT / "docs/reports/mamem_event_label_diagnostic_v1_run"
MAT = Path("/home/whwovy/data/mamem_i_v1_20260913/development_first.mat")
ROLE = MAT.with_name("development_role.json")
MAT_SHA = "57a72c3fde0ff3bc9aaae10721cd7cb450bda63eb5299a96f41704ea696ad10a"
ROLE_SHA = "dc26f85099ed667208ec7b992430d257d01f2a4aa727b7b98f8a5195f6d9af18"
DEADLINE = "2026-09-13T13:00:00+00:00"
CAP = 65536
PINS = [
    "scripts/analysis/diagnose_mamem_event_labels_v1.py",
    "src/cfeg/mamem_events_v1.py",
    "docs/mamem_event_label_diagnostic_v1_plan.md",
]


def require(condition, reason):
    if not condition:
        raise ValueError(reason)


def now():
    return datetime.now(timezone.utc).isoformat()


def remaining():
    value = datetime.fromisoformat(DEADLINE).timestamp() - time.time()
    require(value > 0, "overall_deadline")
    return min(value, 90)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024**2), b""):
            remaining()
            digest.update(chunk)
    return digest.hexdigest()


def save(path, value):
    raw = json.dumps(value, indent=2, allow_nan=False).encode() + b"\n"
    require(len(raw) <= CAP, "output_cap")
    with Path(path).open("xb") as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())


def freeze():
    remaining()
    require(shutil.disk_usage(MAT.parent).free >= 8 * 1024**3, "free_reserve")
    RUN.mkdir()
    save(
        RUN / "manifest.json",
        {
            "deadline": DEADLINE,
            "mat_path": str(MAT),
            "mat_sha256": MAT_SHA,
            "role_sha256": ROLE_SHA,
            "code_sha256": {p: sha(ROOT / p) for p in PINS},
            "created_utc": now(),
            "fits": 0,
            "EEG_variables_requested": False,
            "attempt_budget": 1,
        },
    )


def manifest():
    remaining()
    cfg = json.loads((RUN / "manifest.json").read_bytes())
    require(
        cfg["deadline"] == DEADLINE
        and cfg["mat_path"] == str(MAT)
        and cfg["mat_sha256"] == MAT_SHA
        and cfg["role_sha256"] == ROLE_SHA,
        "manifest_scope",
    )
    require(all(sha(ROOT / p) == value for p, value in cfg["code_sha256"].items()), "code_hash")
    return cfg


def diagnose(din, total_samples):
    """Uniform summaries, including failed membership; never emit assigned labels."""
    import numpy as np

    from cfeg.mamem_events_v1 import FREQUENCIES, _scalar

    require(
        isinstance(din, np.ndarray)
        and din.dtype == object
        and din.ndim == 2
        and din.shape[0] == 4
        and 0 < din.shape[1] <= 10000,
        "DIN_shape",
    )
    stamps = np.array([float(_scalar(v)) for v in din[1]], dtype=np.float64)
    samples = []
    for v in din[3]:
        scalar = _scalar(v)
        require(scalar == int(scalar) and 1 <= scalar <= total_samples, "sample_range_integer")
        samples.append(int(scalar))
    samples = np.array(samples, dtype=np.int64)
    delta = np.diff(stamps)
    require(np.isfinite(delta).all() and (delta > 0).all(), "timestamp_order")
    require((np.diff(samples) > 0).all(), "sample_order")
    bounds = np.r_[0, np.flatnonzero(delta > 2000) + 1, len(stamps)]
    require(len(bounds) == 24, "expected23groups")
    spacing = np.abs(FREQUENCIES[:, None] - FREQUENCIES[None, :])
    np.fill_diagonal(spacing, np.inf)
    widths = spacing.min(axis=1) / 4
    rows = []
    for i in range(23):
        left, right = bounds[i : i + 2]
        ts, ss = stamps[left:right], samples[left:right]
        dt = np.diff(ts)
        freq = 1000 / (2 * dt.mean()) if len(dt) else None
        match = (
            []
            if freq is None
            else FREQUENCIES[
                (freq >= FREQUENCIES - widths) & (freq <= FREQUENCIES + widths)
            ].tolist()
        )
        start0 = int(ss[0]) - 1 + 250
        end0 = start0 + 500
        trial_end0 = int(ss[0]) - 1 + 1250
        rows.append(
            {
                "group_index": i,
                "role": "adaptation" if i < 8 else "main",
                "events": len(ts),
                "first_sample1": int(ss[0]),
                "last_sample1": int(ss[-1]),
                "first_relative_timestamp_ms": float(ts[0] - stamps[0]),
                "last_relative_timestamp_ms": float(ts[-1] - stamps[0]),
                "duration_ms": float(ts[-1] - ts[0]),
                "interval_ms": None
                if not len(dt)
                else {
                    "mean": float(dt.mean()),
                    "median": float(np.median(dt)),
                    "min": float(dt.min()),
                    "max": float(dt.max()),
                },
                "continuous_frequency_hz": None if freq is None else float(freq),
                "old_guardband_matching_nominal_hz": match,
                "old_guardband_unique": len(match) == 1,
                "fixed_window_contained": end0 <= int(ss[-1]) and end0 <= total_samples,
                "full_trial_cost_contained": trial_end0 <= total_samples,
                "analysis_start0": start0,
                "analysis_end0": end0,
                "analysis_event_count": int((((ss - 1) >= start0) & ((ss - 1) < end0)).sum()),
            }
        )
    failures = [r["group_index"] for r in rows if not r["old_guardband_unique"]]
    return {
        "groups": rows,
        "group_count": 23,
        "first_guardband_failure_group": min(failures) if failures else None,
        "guardband_failure_groups": failures,
        "assigned_labels": False,
        "threshold_search": False,
        "EEG_values_loaded": False,
        "M_features_computed": False,
        "fits": 0,
    }


def inspect(loader=None, header_loader=None):
    manifest()
    require(
        MAT.is_file()
        and not MAT.is_symlink()
        and MAT.stat().st_size == 137357437
        and sha(MAT) == MAT_SHA,
        "MAT_pin",
    )
    require(ROLE.is_file() and not ROLE.is_symlink() and sha(ROLE) == ROLE_SHA, "role_pin")
    role = json.loads(ROLE.read_bytes())
    require(
        role["subject"] == "S001"
        and role["subject_role"] == "development_only_all_records"
        and role["mat_path"] == str(MAT)
        and role["mat_sha256"] == MAT_SHA,
        "development_role",
    )
    if loader is None or header_loader is None:
        from scipy.io import loadmat, whosmat

        loader, header_loader = loader or loadmat, header_loader or whosmat
    headers = header_loader(MAT)
    require(
        [(shape, kind) for name, shape, kind in headers if name == "eeg"]
        == [((257, 117917), "double")],
        "EEG_header",
    )
    require(
        [(shape, kind) for name, shape, kind in headers if name == "DIN_1"]
        == [((4, 1966), "cell")],
        "DIN_header",
    )
    loaded = loader(
        MAT,
        variable_names=["DIN_1"],
        squeeze_me=False,
        struct_as_record=True,
        verify_compressed_data_integrity=True,
    )
    require(
        "DIN_1" in loaded and set(loaded) <= {"DIN_1", "__header__", "__version__", "__globals__"},
        "loaded_key_whitelist",
    )
    require(loaded["DIN_1"].shape == (4, 1966), "loaded_DIN_shape")
    result = diagnose(loaded["DIN_1"], 117917)
    return dict(
        result,
        status="DIAGNOSTIC_COMPLETE_NOT_LABEL_VALIDATION",
        mat_sha256=MAT_SHA,
        subject="S001",
        DIN_variable_fully_decoded=True,
        descriptor_interpretation=False,
        completed_utc=now(),
    )


def worker():
    resource.setrlimit(resource.RLIMIT_AS, (2 * 1024**3, 2 * 1024**3))
    resource.setrlimit(resource.RLIMIT_CPU, (60, 60))
    resource.setrlimit(resource.RLIMIT_FSIZE, (CAP, CAP))
    manifest()
    started = json.loads((RUN / "started.json").read_bytes())
    require(
        started["parent_pid"] == os.getppid() and not (RUN / "terminal.json").exists(),
        "active_parent",
    )
    save(RUN / "worker_claim.json", {"pid": os.getpid(), "started_utc": now(), "attempt": 1})
    result = inspect()
    save(RUN / "diagnostic.json", result)


def execute():
    manifest()
    require(shutil.disk_usage(MAT.parent).free >= 8 * 1024**3, "free_reserve")
    save(RUN / "started.json", {"parent_pid": os.getpid(), "started_utc": now(), "attempt": 1})
    report = {"started_utc": now(), "attempts": 1, "fits": 0}
    try:
        env = dict(
            os.environ,
            OPENBLAS_NUM_THREADS="1",
            OMP_NUM_THREADS="1",
            MKL_NUM_THREADS="1",
            PYTHONPATH=str(ROOT / "src"),
        )
        with (RUN / "stdout.txt").open("xb") as out, (RUN / "stderr.txt").open("xb") as err:
            timeout = remaining()
            proc = subprocess.Popen(
                [sys.executable, str(Path(__file__).resolve()), "worker"],
                stdout=out,
                stderr=err,
                env=env,
            )
            try:
                proc.wait(timeout=timeout)
                require(proc.returncode == 0, "worker_failure")
            finally:
                if proc.poll() is None:
                    proc.kill()
                proc.wait()
            require(out.tell() <= CAP and err.tell() <= CAP, "capture_cap")
        remaining()
        report.update(status="COMPLETE", diagnostic_sha256=sha(RUN / "diagnostic.json"))
    except Exception as exc:  # noqa: BLE001 -- preserve every terminal failure, never retry
        report.update(
            status="STOPPED_NO_RETRY",
            error_type=type(exc).__name__,
            reason=str(exc) if isinstance(exc, ValueError) else "bounded_worker_log",
        )
    report["ended_utc"] = now()
    save(RUN / "terminal.json", report)
    print(json.dumps(report))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("freeze", "execute", "worker"))
    action = parser.parse_args().mode
    {"freeze": freeze, "execute": execute, "worker": worker}[action]()
