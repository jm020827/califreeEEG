"""Pinned, once-only S001 reference diagnostic; never a learned-M efficacy run."""

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
RUN = ROOT / "docs/reports/mamem_reference_probe_development_v1_run"
DEADLINE = "2026-09-13T15:00:00+00:00"
CAP = 128 * 1024
TOTAL_CAP = 1024 * 1024
WALL_END = None
ARM_NAMES = ("NOMINAL", "SAMPLE_SUPPORT")
MAT = {
    "a": Path("/home/whwovy/data/mamem_i_v1_20260913/development_first.mat"),
    "b": Path("/home/whwovy/data/mamem_recorded_event_source_v2/S001b.mat"),
}
MAT_BYTES = {"a": 137357437, "b": 137631703}
MAT_SHA = {
    "a": "57a72c3fde0ff3bc9aaae10721cd7cb450bda63eb5299a96f41704ea696ad10a",
    "b": "3beaafa44f7717c690915e7ed85dfa720d96e7da89c7af59113305686c7c9a03",
}
ROLE = {"a": MAT["a"].with_name("development_role.json"),
        "b": MAT["b"].with_name("S001b_input.json")}
ROLE_SHA = {
    "a": "dc26f85099ed667208ec7b992430d257d01f2a4aa727b7b98f8a5195f6d9af18",
    "b": "4d9f3ab103dd5fba24dd6bb3653c4bb2aaa6f7bd4f791e1a5b01bfa5ee380b59",
}
UNCHANGED = {
    "src/cfeg/mamem_reference_probe_v1.py":
        "a147f8ac18016d521832f33dc9f45f956ac0f8163aa56df5a5da9f38f32eb7e3",
    "src/cfeg/mamem_events_v1.py":
        "1c6762783fed206201a9c0539de5aa4c94d45564190bb08954ec4b0c242c096a",
    "src/cfeg/mamem_events_v2.py":
        "e25565ebbf3b0af8bdd469c3847f4114040268dfe2cff19a708d66a13914c666",
}
PINS = ["docs/mamem_reference_probe_development_v1_contract.md",
        "scripts/analysis/run_mamem_reference_probe_v1.py",
        "scripts/analysis/audit_mamem_reference_probe_v1.py", *UNCHANGED]
VARIABLES = {"a": ["DIN_1", "samplingRate"], "b": ["eeg", "DIN_1", "samplingRate"]}
FIXED = {"schema": "cfeg.mamem-reference-probe-dev-v1.manifest", "deadline_utc": DEADLINE,
         "subject": "S001", "attempt_budget": 1, "fits": 0, "queries": 15,
         "predictions": 30, "mat_sha256": MAT_SHA, "role_sha256": ROLE_SHA}
SCOPE = {"a_loaded_variables": VARIABLES["a"], "b_loaded_variables": VARIABLES["b"],
         "a_eeg_numeric_windows": 0, "b_eeg_numeric_windows": 15, "b_eeg_numeric_channels": 1,
         "query_metadata_extractions": 0, "gate_fits": 0, "source_cohort_reads": 0,
         "held60_openings": 0}


def require(condition, reason):
    if not condition:
        raise ValueError(reason)


def now():
    return datetime.now(timezone.utc).isoformat()


def remaining():
    seconds = datetime.fromisoformat(DEADLINE).timestamp() - time.time()
    if WALL_END is not None:
        seconds = min(seconds, WALL_END - time.monotonic())
    require(seconds > 0, "deadline")
    return min(seconds, 120)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024**2), b""):
            remaining()
            digest.update(chunk)
    return digest.hexdigest()


def fsync_dir(path):
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def budget(extra=0):
    files = list(RUN.iterdir())
    require(all(p.is_file() and not p.is_symlink() and p.stat().st_size <= CAP for p in files),
            "bounded_run_files")
    require(sum(p.stat().st_size for p in files) + extra <= TOTAL_CAP, "total_output_cap")


def save(path, value):
    raw = json.dumps(value, allow_nan=False, separators=(",", ":")).encode() + b"\n"
    require(len(raw) <= CAP, "output_cap")
    budget(len(raw))
    with Path(path).open("xb") as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())
    fsync_dir(Path(path).parent)


def read_json(path):
    path = Path(path)
    require(path.is_file() and not path.is_symlink() and path.stat().st_size <= CAP,
            "bounded_regular_json")
    return json.loads(path.read_bytes(), parse_constant=lambda _: require(False, "json_nonfinite"))


def validate_manifest(cfg):
    require(set(cfg) == set(FIXED) | {"created_utc", "code_sha256"}, "manifest_keyset")
    require(all(type(cfg[k]) is type(v) and cfg[k] == v for k, v in FIXED.items()),
            "manifest_scope")
    require(set(cfg["code_sha256"]) == set(PINS), "code_pin_keyset")
    require(all(cfg["code_sha256"][p] == h for p, h in UNCHANGED.items()), "unchanged_core")
    require(all(sha(ROOT / p) == h for p, h in cfg["code_sha256"].items()), "code_hash")
    return cfg


def manifest():
    remaining()
    return validate_manifest(read_json(RUN / "manifest.json"))


def freeze():
    remaining()
    require(shutil.disk_usage(ROOT).free >= 8 * 1024**3, "free_reserve")
    cfg = dict(FIXED, created_utc=now(), code_sha256={p: sha(ROOT / p) for p in PINS})
    validate_manifest(cfg)
    RUN.mkdir()
    fsync_dir(RUN.parent)
    save(RUN / "manifest.json", cfg)


def validate_input(role):
    path = MAT[role]
    require(path.is_file() and not path.is_symlink() and path.resolve() == path
            and path.stat().st_size == MAT_BYTES[role] and sha(path) == MAT_SHA[role], "MAT_pin")
    require(ROLE[role].resolve() == ROLE[role] and sha(ROLE[role]) == ROLE_SHA[role], "role_pin")
    receipt = read_json(ROLE[role])
    path_key, hash_key = ("mat_path", "mat_sha256") if role == "a" else ("path", "sha256")
    require(receipt["subject"] == "S001" and receipt["member"] == f"EEG-SSVEP-Part1/S001{role}.mat"
            and receipt["bytes"] == MAT_BYTES[role] and receipt[path_key] == str(path)
            and receipt[hash_key] == MAT_SHA[role], "role_binding")
    require(receipt.get("subject_role") == "development_only_all_records" if role == "a"
            else receipt.get("run") == "b", "development_role")


def load_role(role, loader=None, header_loader=None):
    import numpy as np
    from scipy.io import loadmat, whosmat

    from cfeg.mamem_events_v1 import _scalar

    validate_input(role)
    loader, header_loader = loader or loadmat, header_loader or whosmat
    headers = header_loader(MAT[role])
    chosen = {}
    for name in ("eeg", "DIN_1", "samplingRate"):
        entries = [(shape, kind) for n, shape, kind in headers if n == name]
        require(len(entries) == 1, "unique_header")
        chosen[name] = entries[0]
    shape, kind = chosen["eeg"]
    require(len(shape) == 2 and shape[0] == 257 and 0 < shape[1] <= 200000 and kind == "double",
            "EEG_header")
    total_samples = shape[1]
    shape, kind = chosen["DIN_1"]
    require(len(shape) == 2 and shape[0] == 4 and 0 < shape[1] <= 10000 and kind == "cell",
            "DIN_header")
    if role == "a":
        require(chosen["eeg"][0] == (257, 117917) and shape == (4, 1966), "A_header_pin")
    shape, kind = chosen["samplingRate"]
    numeric = {"double", "single", "int8", "uint8", "int16", "uint16", "int32", "uint32",
               "int64", "uint64"}
    require(shape == (1, 1) and kind in numeric, "samplingRate_header")
    remaining()
    data = loader(MAT[role], variable_names=VARIABLES[role], squeeze_me=False,
                  struct_as_record=True, verify_compressed_data_integrity=True)
    require(set(VARIABLES[role]) <= set(data)
            <= set(VARIABLES[role]) | {"__header__", "__version__", "__globals__"},
            "variable_whitelist")
    require(float(_scalar(data["samplingRate"])) == 250, "samplingRate")
    require(isinstance(data["DIN_1"], np.ndarray) and data["DIN_1"].shape == chosen["DIN_1"][0]
            and data["DIN_1"].dtype == object, "DIN_loaded_shape")
    if role == "b":
        require(isinstance(data["eeg"], np.ndarray) and data["eeg"].shape == chosen["eeg"][0]
                and data["eeg"].dtype == np.float64, "EEG_loaded_shape")
    return data, total_samples


def support_bank(din, total_samples):
    import numpy as np

    from cfeg.mamem_events_v1 import _scalar
    from cfeg.mamem_events_v2 import parse_main_trials
    from cfeg.mamem_reference_probe_v1 import (
        NOMINAL_FREQUENCIES,
        reference_bank,
        sample_clock_frequencies,
    )

    records = parse_main_trials(din, total_samples, include_metadata=False)
    samples = np.array([int(_scalar(cell)) for cell in din[3, :]], dtype=np.int64)
    selected = []
    for label in range(5):
        first = min((row for row in records if row["label"] == label), key=lambda r: r["start0"])
        inside = (samples - 1 >= first["start0"]) & (samples - 1 < first["end0"])
        selected.append({**{key: first[key] for key in
                            ("label", "group_index", "start0", "end0", "trial_end0")},
                         "event_samples": samples[inside].tolist()})
    frequencies = {"NOMINAL": list(NOMINAL_FREQUENCIES), "SAMPLE_SUPPORT":
                   sample_clock_frequencies([r["event_samples"] for r in selected]).tolist()}
    banks = {arm: reference_bank(frequencies[arm]) for arm in ARM_NAMES}
    for bank in banks.values():
        bank.setflags(write=False)
    return selected, frequencies, banks


def decode_windows(eeg, windows, banks):
    """Windows carry only indices; truth and DIN never enter this function."""
    import numpy as np

    from cfeg.mamem_reference_probe_v1 import scores

    require(set(banks) == set(ARM_NAMES) and len(windows) == 15, "decode_scope")
    decoded = []
    for window in windows:
        remaining()
        require(set(window) == {"group_index", "start0", "end0"}, "window_no_truth")
        require(window["end0"] - window["start0"] == 500, "window_length")
        query = eeg[125, window["start0"]:window["end0"]]
        arms = {}
        for arm in ARM_NAMES:
            result = scores(query, banks[arm])
            centered = query / np.max(np.abs(query))
            centered = centered - centered.mean()
            normalized = centered / np.linalg.norm(centered)
            projection = banks[arm].transpose(0, 2, 1) @ normalized
            require(np.max(np.abs(np.sum(projection**2, axis=1) - result)) <= 1e-12,
                    "projection_score_consistency")
            arms[arm] = {"projection": projection.tolist(), "scores": result.tolist(),
                         "prediction": int(np.argmax(result))}
        decoded.append(dict(window, arms=arms))
    return decoded


def evaluate(decoded, labels):
    require(len(decoded) == len(labels) == 15, "evaluation_scope")
    queries = []
    for row, label in zip(decoded, labels):
        require(type(label) is int and 0 <= label < 5, "evaluation_label")
        arms = {arm: dict(value, correct=value["prediction"] == label)
                for arm, value in row["arms"].items()}
        queries.append(dict(row, label=label, arms=arms))
    summary = {}
    for arm in ARM_NAMES:
        counts = [sum(r["arms"][arm]["correct"] for r in queries if r["label"] == label)
                  for label in range(5)]
        summary[arm] = {"correct": sum(counts), "total": 15, "per_class_correct": counts,
                        "signal_present_dev": sum(counts) >= 12 and min(counts) >= 1}
    paired = dict.fromkeys(("both_correct", "nominal_only", "sample_only", "both_wrong"), 0)
    for row in queries:
        a, b = (row["arms"][arm]["correct"] for arm in ARM_NAMES)
        key = "both_correct" if a and b else "nominal_only" if a else "sample_only" if b else "both_wrong"
        paired[key] += 1
    summary["paired"] = paired
    summary["decision"] = "REFERENCE_DIAGNOSTIC_COMPLETE" if any(
        summary[arm]["signal_present_dev"] for arm in ARM_NAMES) else "NO_ARM_PASSES_STOP"
    return queries, summary


def inspect(access=None, role_loader=None):
    from cfeg.mamem_events_v2 import parse_main_trials

    access = [] if access is None else access
    role_loader = role_loader or load_role
    a, a_samples = role_loader("a")
    access.append("a_loaded")
    support, frequencies, banks = support_bank(a["DIN_1"], a_samples)
    access.append("support_bank_frozen")
    del a
    b, b_samples = role_loader("b")
    access.append("b_loaded")
    records = parse_main_trials(b["DIN_1"], b_samples, include_metadata=False)
    windows = [{k: r[k] for k in ("group_index", "start0", "end0")} for r in records]
    decoded = decode_windows(b["eeg"], windows, banks)
    access.append("all_scores_frozen")
    queries, summary = evaluate(decoded, [r["label"] for r in records])
    access.append("evaluation_done")
    cost = {"selected_trials": 5, "selected_stimulus_seconds": 25, "analyzed_support_seconds": 10,
            "selected_support_elapsed_prefix_seconds": max(r["trial_end0"] for r in support) / 250,
            "full_support_record_seconds": a_samples / 250, "query_ready_elapsed": "UNKNOWN",
            "setup_seconds": "UNKNOWN"}
    remaining()
    return {"schema": "cfeg.mamem-reference-probe-dev-v1.result", "subject": "S001",
            "status": "DEVELOPMENT_DIAGNOSTIC_NOT_EFFICACY", "mat_sha256": MAT_SHA,
            "role_sha256": ROLE_SHA, "completed_utc": now(), "sampling_rate_hz": 250,
            "channel_index": 125, "window_samples": 500, "fits": 0, "frequencies_hz": frequencies,
            "support": support, "queries": queries, "summary": summary, "cost": cost,
            "scope": SCOPE, "access_stages": access}


def worker():
    global WALL_END
    WALL_END = time.monotonic() + 120
    for limit, cap in ((resource.RLIMIT_AS, 2 * 1024**3), (resource.RLIMIT_CPU, 90),
                       (resource.RLIMIT_FSIZE, CAP)):
        resource.setrlimit(limit, (cap, cap))
    manifest()
    started = read_json(RUN / "started.json")
    require(started["status"] == "STARTED" and started["attempt"] == 1
            and started["parent_pid"] == os.getppid()
            and started["manifest_sha256"] == sha(RUN / "manifest.json")
            and not (RUN / "terminal.json").exists(), "active_parent")
    save(RUN / "worker_claim.json", {"attempt": 1, "pid": os.getpid(), "parent_pid": os.getppid(),
         "started_utc": now(), "manifest_sha256": sha(RUN / "manifest.json")})
    access = []
    try:
        result = inspect(access)
        remaining()
        save(RUN / "result.json", result)
    except Exception as error:
        save(RUN / "worker_failure.json", {"access_stages": access, "error_type": type(error).__name__,
             "reason": str(error) if isinstance(error, ValueError) else "see_bounded_stderr"})
        raise


def execute():
    global WALL_END
    WALL_END = time.monotonic() + 120
    manifest()
    require(shutil.disk_usage(ROOT).free >= 8 * 1024**3, "free_reserve")
    started = {"status": "STARTED", "attempt": 1, "parent_pid": os.getpid(), "started_utc": now(),
               "manifest_sha256": sha(RUN / "manifest.json")}
    save(RUN / "started.json", started)
    terminal = {"attempts": 1, "fits": 0, "started_utc": started["started_utc"],
                "manifest_sha256": started["manifest_sha256"],
                "started_sha256": None, "worker_claim_sha256": None}
    try:
        terminal["started_sha256"] = sha(RUN / "started.json")
        env = dict(os.environ, OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1",
                   NUMEXPR_NUM_THREADS="1", BLIS_NUM_THREADS="1", PYTHONPATH=str(ROOT / "src"),
                   PYTHONDONTWRITEBYTECODE="1")
        with (RUN / "stdout.txt").open("xb") as out, (RUN / "stderr.txt").open("xb") as err:
            timeout = remaining()
            process = subprocess.Popen([sys.executable, __file__, "worker"], stdout=out, stderr=err,
                                       env=env)
            try:
                process.wait(timeout=timeout)
                require(process.returncode == 0, "worker_failed")
            finally:
                if process.poll() is None:
                    process.kill()
                process.wait()
        remaining()
        budget()
        result = read_json(RUN / "result.json")
        require(result["status"] == "DEVELOPMENT_DIAGNOSTIC_NOT_EFFICACY"
                and result["mat_sha256"] == MAT_SHA and result["role_sha256"] == ROLE_SHA
                and len(result["queries"]) == 15 and result["scope"] == SCOPE, "result_binding")
        manifest()
        terminal.update(status="COMPLETE", predictions=30, result_sha256=sha(RUN / "result.json"),
                        worker_claim_sha256=sha(RUN / "worker_claim.json"))
        remaining()
    except Exception as error:  # noqa: BLE001 -- preserve failure, never retry
        terminal.update(status="STOPPED_NO_RETRY", error_type=type(error).__name__,
                        reason=str(error) if isinstance(error, ValueError) else "see_bounded_log")
        if terminal["started_sha256"] is None:
            # save() has already persisted precisely this JSON serialization.
            encoded = json.dumps(started, allow_nan=False, separators=(",", ":")).encode() + b"\n"
            terminal["started_sha256"] = hashlib.sha256(encoded).hexdigest()
        if (RUN / "worker_claim.json").is_file():
            # Completion hash needed even when runtime deadline has expired.
            terminal["worker_claim_sha256"] = hashlib.sha256(
                (RUN / "worker_claim.json").read_bytes()).hexdigest()
    terminal["ended_utc"] = now()
    save(RUN / "terminal.json", terminal)
    print(json.dumps(terminal))
    if terminal["status"] != "COMPLETE":
        raise SystemExit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("freeze", "execute", "worker"))
    {"freeze": freeze, "execute": execute, "worker": worker}[parser.parse_args().mode]()
