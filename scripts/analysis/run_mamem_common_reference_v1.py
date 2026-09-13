"""One target-support-free common-reference EEG diagnostic, no prior run restart."""

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
RUN = ROOT / "docs/reports/mamem_common_reference_v1_run"
PREVIOUS = "docs/reports/mamem_reference_residual_v1_run"
SPEC = "configs/governance/mamem_reference_residual_v1_inputs.json"
DATA = Path("/home/whwovy/data/mamem_recorded_event_source_v2")
SUBJECTS = tuple(f"S{p:03d}" for p in range(2, 12))
DEADLINE = "2026-09-13T16:00:00+00:00"
CAP, TOTAL_CAP = 1024**2, 8*1024**2
WALL_END = None
KNOWN_SHA = {
    SPEC: "99a3adfae3765457bb23bf0305519775268007546742b6bff512d33fb3d7135b",
    PREVIOUS+"/result.json": "50cef4d3482975efe69d4fc5c145a95d2e4ab3822db30203576ce63dad906389",
    PREVIOUS+"/terminal.json": "2b2eeb8e1e8b6c61447d6fcc932515962da081cc7b54ded917be4c5df89a0ff4",
    "docs/reports/mamem_reference_residual_v1_audit.json":
        "f54841c75840ae0f2b0b0d911b34c0489501eb846f315a7cff4344a90ff2c1c4",
    "src/cfeg/mamem_reference_probe_v1.py":
        "a147f8ac18016d521832f33dc9f45f956ac0f8163aa56df5a5da9f38f32eb7e3",
}
PINS = [*KNOWN_SHA, "docs/mamem_common_reference_v1_contract.md",
        "src/cfeg/mamem_common_reference_v1.py", "scripts/analysis/run_mamem_common_reference_v1.py",
        "scripts/analysis/audit_mamem_common_reference_v1.py",
        *[PREVIOUS+"/record_"+s+"b.json" for s in SUBJECTS]]
SCOPE = {"mat_byte_hashes": 10, "mat_headers": 10, "mat_eeg_loads": 10, "din_decodes": 0,
         "rate_decodes": 0, "eeg_numeric_windows": 150, "eeg_numeric_channels": 1,
         "predictions": 300, "class_scores": 1500, "projection_scalars": 6000, "fits": 0,
         "target_support_trials": 0, "held60_openings": 0}
COST = {"target_labeled_support_trials": 0, "target_support_stimulus_seconds": 0,
        "source_subjects_per_fold": 9, "source_event_windows_per_fold": 135,
        "source_collection": "PREVIOUSLY_ACQUIRED_NOT_TARGET_CALIBRATION",
        "query_ready_elapsed": "UNKNOWN", "setup_seconds": "UNKNOWN"}


def require(ok, why):
    if not ok:
        raise ValueError(why)


def now():
    return datetime.now(timezone.utc).isoformat()


def remaining():
    left = datetime.fromisoformat(DEADLINE).timestamp()-time.time()
    if WALL_END is not None:
        left = min(left, WALL_END-time.monotonic())
    require(left > 0, "deadline")
    return min(left, 180)


def sha(path, timed=True):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for part in iter(lambda: stream.read(1024**2), b""):
            if timed:
                remaining()
            digest.update(part)
    return digest.hexdigest()


def unique(items):
    result = {}
    for k, v in items:
        require(k not in result, "duplicate_json_key")
        result[k] = v
    return result


def read(path):
    path = Path(path)
    require(path.is_file() and not path.is_symlink() and path.stat().st_size <= CAP, "bounded_json")
    return json.loads(path.read_bytes(), object_pairs_hook=unique,
                      parse_constant=lambda _: require(False, "nonfinite_json"))


def fsync_dir(path):
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def save(path, payload):
    raw = json.dumps(payload, allow_nan=False, separators=(",", ":")).encode()+b"\n"
    files = list(RUN.iterdir())
    require(all(p.is_file() and not p.is_symlink() and p.stat().st_size <= CAP for p in files),
            "file_cap")
    reserve = 0 if Path(path).name == "terminal.json" else 3*CAP
    require(len(raw) <= CAP and len(raw)+sum(p.stat().st_size for p in files) <= TOTAL_CAP-reserve,
            "total_cap")
    with Path(path).open("xb") as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())
    fsync_dir(Path(path).parent)


def cached_inputs():
    cfg = read(ROOT/SPEC)
    require(cfg["schema"] == "cfeg.mamem-reference-residual-v1.inputs", "input_schema")
    require([(r["subject"], r["run"]) for r in cfg["inputs"]] ==
            [(s, r) for s in SUBJECTS for r in ("a", "b")], "source_roles")
    rows = [r for r in cfg["inputs"] if r["run"] == "b"]
    previous = read(ROOT/PREVIOUS/"result.json")
    records = {}
    for row in rows:
        s = row["subject"]
        require(row["path"] == str(DATA/(s+"b.mat")) and type(row["bytes"]) is int
                and 0 < row["bytes"] <= 512*1024**2, "input_path_size")
        path = ROOT/PREVIOUS/("record_"+s+"b.json")
        require(sha(path) == previous["record_sha256"][s+"b"], "cached_record_hash")
        record = read(path)
        require(record["subject"] == s and record["run"] == "b"
                and record["mat_sha256"] == row["sha256"]
                and record["receipt_sha256"] == row["receipt_sha256"]
                and record["loaded_variables"] == ["DIN_1", "samplingRate"], "cached_record_role")
        n = record["total_samples"]
        require(type(n) is int and 0 < n <= 200000, "cached_total_samples")
        trials = record["records"]
        require(len(trials) == 15 and [r["group_index"] for r in trials] == list(range(8, 23)),
                "cached_groups")
        labels = [r["label"] for r in trials]
        require(all(type(v) is int and 0 <= v < 5 for v in labels)
                and all(labels.count(c) == 3 for c in range(5))
                and all(len(set(labels[j:j+3])) == 1 for j in range(0, 15, 3)), "cached_labels")
        end = 0
        for trial in trials:
            require(all(type(trial[k]) is int for k in ("group_index", "start0", "end0", "trial_end0"))
                    and end <= trial["start0"] and trial["end0"]-trial["start0"] == 500
                    and trial["trial_end0"]-trial["start0"] == 1000
                    and trial["trial_end0"] <= n, "cached_windows")
            end = trial["end0"]
        records[s] = record
    return rows, records


def validate_manifest(cfg):
    require(set(cfg) == {"schema", "created_utc", "deadline_utc", "attempts", "scope", "pins",
                         "frequencies_hz"} and cfg["schema"] == "cfeg.mamem-common-reference-v1.manifest"
            and cfg["deadline_utc"] == DEADLINE and type(cfg["attempts"]) is int
            and cfg["attempts"] == 1 and cfg["scope"] == SCOPE and set(cfg["pins"]) == set(PINS),
            "manifest_scope")
    require(all(cfg["pins"][p] == h for p, h in KNOWN_SHA.items()), "known_pins")
    require(all(sha(ROOT/p) == h for p, h in cfg["pins"].items()), "code_source_pins")
    frequencies = cfg["frequencies_hz"]
    require(set(frequencies) == {"NOMINAL", "COMMON"}
            and frequencies["NOMINAL"] == [6.66, 7.5, 8.57, 10., 12.]
            and set(frequencies["COMMON"]) == set(SUBJECTS), "bank_scope")
    for bank in frequencies["COMMON"].values():
        require(isinstance(bank, list) and len(bank) == 5 and
                all(type(v) in (int, float) and 0 < v < 62.5 for v in bank)
                and all(bank[j] < bank[j+1] for j in range(4)), "bank_range")
    return cfg


def manifest():
    remaining()
    return validate_manifest(read(RUN/"manifest.json"))


def freeze():
    from cfeg.mamem_common_reference_v1 import common_frequencies

    remaining()
    require(shutil.disk_usage(ROOT).free >= 8*1024**3, "disk_reserve")
    pins = {p: sha(ROOT/p) for p in PINS}
    require(all(pins[p] == h for p, h in KNOWN_SHA.items()), "known_pins")
    cached_inputs()
    previous = read(ROOT/PREVIOUS/"result.json")
    frequencies = {"NOMINAL": [6.66, 7.5, 8.57, 10., 12.],
                   "COMMON": common_frequencies([p[1] for p in previous["frequencies_hz"]])}
    cfg = {"schema": "cfeg.mamem-common-reference-v1.manifest", "created_utc": now(),
           "deadline_utc": DEADLINE, "attempts": 1, "scope": SCOPE, "pins": pins,
           "frequencies_hz": frequencies}
    validate_manifest(cfg)
    RUN.mkdir()
    fsync_dir(RUN.parent)
    save(RUN/"manifest.json", cfg)


def validate_input(row):
    path = Path(row["path"])
    receipt = path.with_name(row["subject"]+"b_input.json")
    require(path.is_file() and not path.is_symlink() and path.resolve() == path
            and path.stat().st_size == row["bytes"], "mat_path_size")
    require(receipt.is_file() and not receipt.is_symlink() and receipt.resolve() == receipt
            and receipt.stat().st_size <= CAP and sha(receipt) == row["receipt_sha256"], "receipt_pin")
    require(read(receipt) == {k: v for k, v in row.items() if k != "receipt_sha256"}, "receipt_role")
    require(sha(path) == row["sha256"], "mat_pin")


def read_eeg(row, record, loader=None, header_loader=None):
    import numpy as np
    from scipy.io import loadmat, whosmat

    validate_input(row)
    header = (header_loader or whosmat)(row["path"])
    chosen = {}
    for name in ("eeg", "DIN_1", "samplingRate"):
        entries = [(shape, kind) for n, shape, kind in header if n == name]
        require(len(entries) == 1, "unique_header")
        chosen[name] = entries[0]
    require(chosen["eeg"] == ((257, record["total_samples"]), "double"), "eeg_header")
    shape, kind = chosen["DIN_1"]
    require(len(shape) == 2 and shape[0] == 4 and 0 < shape[1] <= 10000 and kind == "cell",
            "din_header")
    shape, kind = chosen["samplingRate"]
    require(shape == (1, 1) and kind in {"double", "single", "int8", "uint8", "int16", "uint16",
                                       "int32", "uint32", "int64", "uint64"}, "rate_header")
    remaining()
    data = (loader or loadmat)(row["path"], variable_names=["eeg"], squeeze_me=False,
                              struct_as_record=True, verify_compressed_data_integrity=True)
    require({"eeg"} <= set(data) <= {"eeg", "__header__", "__version__", "__globals__"}, "eeg_only")
    eeg = data["eeg"]
    require(isinstance(eeg, np.ndarray) and eeg.dtype == np.float64
            and eeg.shape == (257, record["total_samples"]), "loaded_eeg_shape")
    return eeg


def inspect(cfg):
    from cfeg.mamem_common_reference_v1 import decode_windows, evaluate
    from cfeg.mamem_reference_probe_v1 import reference_bank

    rows, records = cached_inputs()
    all_banks = {s: {arm: reference_bank(cfg["frequencies_hz"][arm] if arm == "NOMINAL"
                                        else cfg["frequencies_hz"][arm][s])
                     for arm in ("NOMINAL", "COMMON")} for s in SUBJECTS}
    for banks in all_banks.values():
        for bank in banks.values():
            bank.flags.writeable = False
    scored, hashes = [], {}
    for row in rows:
        s = row["subject"]
        save(RUN/("attempt_"+s+".json"), {"subject": s, "started_utc": now(),
             "mat_sha256": row["sha256"], "loaded_variables": ["eeg"]})
        eeg = read_eeg(row, records[s])
        windows = [{k: r[k] for k in ("group_index", "start0", "end0")} for r in records[s]["records"]]
        queries = decode_windows(eeg, windows, all_banks[s])
        del eeg
        result = {"subject": s, "mat_sha256": row["sha256"], "completed_utc": now(),
                  "source_record_sha256": cfg["pins"][PREVIOUS+"/record_"+s+"b.json"], "queries": queries}
        save(RUN/("scores_"+s+".json"), result)
        hashes[s] = sha(RUN/("scores_"+s+".json"))
        scored.append(result)
        remaining()
    save(RUN/"scores_frozen.json", {"created_utc": now(), "predictions": 300, "scores_sha256": hashes})
    participants, summary = evaluate(scored, {s: [r["label"] for r in records[s]["records"]]
                                             for s in SUBJECTS})
    return {"schema": "cfeg.mamem-common-reference-v1.result", "completed_utc": now(), "scope": SCOPE,
            "scores_frozen_sha256": sha(RUN/"scores_frozen.json"), "participants": participants,
            "summary": summary, "cost": COST}


def worker():
    global WALL_END
    WALL_END = time.monotonic()+180
    for key, value in ((resource.RLIMIT_AS, 2*1024**3), (resource.RLIMIT_CPU, 120),
                       (resource.RLIMIT_FSIZE, CAP)):
        resource.setrlimit(key, (value, value))
    cfg = manifest()
    start = read(RUN/"started.json")
    require(start["status"] == "STARTED" and start["attempt"] == 1
            and start["parent_pid"] == os.getppid() and not (RUN/"terminal.json").exists()
            and start["manifest_sha256"] == sha(RUN/"manifest.json"), "active_parent")
    save(RUN/"worker_claim.json", {"attempt": 1, "pid": os.getpid(), "parent_pid": os.getppid(),
         "started_utc": now(), "manifest_sha256": start["manifest_sha256"]})
    result = inspect(cfg)
    remaining()
    save(RUN/"result.json", result)


def execute():
    global WALL_END
    WALL_END = time.monotonic()+180
    manifest()
    require(shutil.disk_usage(ROOT).free >= 8*1024**3, "disk_reserve")
    start = {"status": "STARTED", "attempt": 1, "parent_pid": os.getpid(), "started_utc": now(),
             "manifest_sha256": sha(RUN/"manifest.json")}
    save(RUN/"started.json", start)
    terminal = {"attempts": 1, "fits": 0, "started_utc": start["started_utc"],
                "manifest_sha256": start["manifest_sha256"]}
    try:
        env = dict(os.environ, OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1",
                   BLIS_NUM_THREADS="1", NUMEXPR_NUM_THREADS="1", PYTHONPATH=str(ROOT/"src"),
                   PYTHONDONTWRITEBYTECODE="1")
        with (RUN/"stdout.txt").open("xb") as out, (RUN/"stderr.txt").open("xb") as err:
            timeout = remaining()
            child = subprocess.Popen([sys.executable, __file__, "worker"], env=env, stdout=out, stderr=err)
            try:
                child.wait(timeout=timeout)
                require(child.returncode == 0, "worker_failed")
            finally:
                if child.poll() is None:
                    child.kill()
                child.wait()
        result = read(RUN/"result.json")
        require(result["schema"] == "cfeg.mamem-common-reference-v1.result" and result["scope"] == SCOPE
                and [p["subject"] for p in result["participants"]] == list(SUBJECTS), "result_scope")
        manifest()
        terminal.update(status="COMPLETE", result_sha256=sha(RUN/"result.json"))
        remaining()
    except Exception as error:  # noqa: BLE001 -- retain failures without retry
        terminal.pop("result_sha256", None)
        terminal.update(status="STOPPED_NO_RETRY", error_type=type(error).__name__,
                        reason=str(error) if isinstance(error, ValueError) else "see_bounded_stderr")
    terminal.update(ended_utc=now(), started_sha256=sha(RUN/"started.json", timed=False),
                    worker_claim_sha256=sha(RUN/"worker_claim.json", timed=False)
                    if (RUN/"worker_claim.json").is_file() else None)
    save(RUN/"terminal.json", terminal)
    print(json.dumps(terminal))
    if terminal["status"] != "COMPLETE":
        raise SystemExit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("freeze", "execute", "worker"))
    {"freeze": freeze, "execute": execute, "worker": worker}[parser.parse_args().mode]()
