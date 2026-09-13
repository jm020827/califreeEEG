"""One bounded DIN-only residual screen. All actual reads are in a claimed worker."""

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
RUN = ROOT / "docs/reports/mamem_reference_residual_v1_run"
SPEC = "configs/governance/mamem_reference_residual_v1_inputs.json"
DEADLINE = "2026-09-13T15:30:00+00:00"
CAP, TOTAL_CAP = 1024**2, 8 * 1024**2
WALL_END = None
SUBJECTS = tuple(f"S{p:03d}" for p in range(2, 12))
DATA = Path("/home/whwovy/data/mamem_recorded_event_source_v2")
VARIABLES = ["DIN_1", "samplingRate"]
UNCHANGED = {
    "src/cfeg/mamem_events_v1.py":
        "1c6762783fed206201a9c0539de5aa4c94d45564190bb08954ec4b0c242c096a",
    "src/cfeg/mamem_events_v2.py":
        "e25565ebbf3b0af8bdd469c3847f4114040268dfe2cff19a708d66a13914c666",
    "src/cfeg/mamem_reference_probe_v1.py":
        "a147f8ac18016d521832f33dc9f45f956ac0f8163aa56df5a5da9f38f32eb7e3",
}
PINS = [SPEC, "docs/mamem_reference_residual_v1_contract.md",
        "src/cfeg/mamem_reference_residual_v1.py", "scripts/analysis/run_mamem_reference_residual_v1.py",
        "scripts/analysis/audit_mamem_reference_residual_v1.py", *UNCHANGED]
SCOPE = {"mat_byte_hashes": 20, "mat_headers": 20, "mat_din_loads": 20,
         "a_diagnostic_event_windows": 150, "b_diagnostic_event_windows": 150,
         "b_predictor_metadata_used": 0, "eeg_arrays_decoded": 0, "eeg_numeric_windows": 0,
         "fits": 0, "eeg_predictions": 0, "held60_openings": 0}


def require(condition, reason):
    if not condition:
        raise ValueError(reason)


def now():
    return datetime.now(timezone.utc).isoformat()


def remaining():
    left = datetime.fromisoformat(DEADLINE).timestamp() - time.time()
    if WALL_END is not None:
        left = min(left, WALL_END - time.monotonic())
    require(left > 0, "deadline")
    return min(180, left)


def sha(path, timed=True):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024**2), b""):
            if timed:
                remaining()
            result.update(block)
    return result.hexdigest()


def unique(pairs):
    result = {}
    for key, value in pairs:
        require(key not in result, "duplicate_json_key")
        result[key] = value
    return result


def read_json(path):
    path = Path(path)
    require(path.is_file() and not path.is_symlink() and path.stat().st_size <= CAP,
            "bounded_regular_json")
    return json.loads(path.read_bytes(), object_pairs_hook=unique,
                      parse_constant=lambda _: require(False, "json_nonfinite"))


def save(path, payload):
    raw = json.dumps(payload, allow_nan=False, separators=(",", ":")).encode()+b"\n"
    files = list(RUN.iterdir())
    require(all(p.is_file() and not p.is_symlink() and p.stat().st_size <= CAP for p in files),
            "run_file_cap")
    reserve = 0 if Path(path).name == "terminal.json" else 3*CAP
    require(len(raw) <= CAP and sum(p.stat().st_size for p in files)+len(raw) <= TOTAL_CAP-reserve,
            "run_total_cap")
    with Path(path).open("xb") as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())


def specification():
    cfg = read_json(ROOT / SPEC)
    require(set(cfg) == {"schema", "inputs"} and cfg["schema"] ==
            "cfeg.mamem-reference-residual-v1.inputs", "input_spec_schema")
    rows = cfg["inputs"]
    require([(r["subject"], r["run"]) for r in rows] ==
            [(s, r) for s in SUBJECTS for r in ("a", "b")], "input_order_roles")
    keys = {"subject", "run", "archive", "member", "bytes", "crc32", "path", "sha256",
            "receipt_sha256"}
    for row in rows:
        name = row["subject"] + row["run"]
        part = 1 if int(row["subject"][1:]) <= 6 else 2
        require(set(row) == keys and row["path"] == str(DATA / (name+".mat"))
                and row["archive"] ==
                f"/home/whwovy/data/mamem_i_v1_20260913/EEG-SSVEP-Part{part}.rar"
                and row["member"] == (f"EEG-SSVEP-Part1/{name}.mat" if part == 1
                                      else name+".mat")
                and type(row["bytes"]) is int and 0 < row["bytes"] <= 512*1024**2,
                "input_binding")
        for key in ("sha256", "receipt_sha256"):
            require(type(row[key]) is str and len(row[key]) == 64
                    and all(c in "0123456789abcdef" for c in row[key]), "hash_format")
        require(type(row["crc32"]) is str and len(row["crc32"]) == 8
                and all(c in "0123456789abcdef" for c in row["crc32"]), "crc_format")
    require(sum(row["bytes"] for row in rows) == 2760155793, "input_total_bytes")
    return rows


def manifest():
    remaining()
    cfg = read_json(RUN / "manifest.json")
    require(set(cfg) == {"schema", "created_utc", "deadline_utc", "attempts", "scope", "pins"}
            and cfg["schema"] == "cfeg.mamem-reference-residual-v1.manifest"
            and cfg["deadline_utc"] == DEADLINE and type(cfg["attempts"]) is int
            and cfg["attempts"] == 1 and cfg["scope"] == SCOPE
            and set(cfg["pins"]) == set(PINS), "manifest_scope")
    require(all(cfg["pins"][p] == h for p, h in UNCHANGED.items()), "immutable_core")
    require(all(sha(ROOT / p) == h for p, h in cfg["pins"].items()), "code_pins")
    return cfg


def freeze():
    remaining()
    require(shutil.disk_usage(ROOT).free >= 8*1024**3, "disk_reserve")
    specification()  # receipts only, no MAT read
    pins = {p: sha(ROOT / p) for p in PINS}
    require(all(pins[p] == h for p, h in UNCHANGED.items()), "immutable_core")
    RUN.mkdir()
    save(RUN / "manifest.json", {"schema": "cfeg.mamem-reference-residual-v1.manifest",
         "created_utc": now(), "deadline_utc": DEADLINE, "attempts": 1, "scope": SCOPE,
         "pins": pins})


def validate_input(row):
    path = Path(row["path"])
    receipt = path.with_name(row["subject"]+row["run"]+"_input.json")
    require(path.resolve() == path and path.is_file() and not path.is_symlink()
            and path.stat().st_size == row["bytes"], "mat_path_size")
    require(receipt.resolve() == receipt and receipt.is_file() and not receipt.is_symlink()
            and receipt.stat().st_size <= CAP
            and sha(receipt) == row["receipt_sha256"], "receipt_pin")
    require(read_json(receipt) == {k: v for k, v in row.items() if k != "receipt_sha256"},
            "receipt_binding")
    require(sha(path) == row["sha256"], "mat_pin")


def read_din(row, loader=None, header_loader=None):
    import numpy as np
    from scipy.io import loadmat, whosmat

    from cfeg.mamem_events_v1 import _scalar

    validate_input(row)
    loader, header_loader = loader or loadmat, header_loader or whosmat
    header = header_loader(row["path"])
    chosen = {}
    for name in ("eeg", "DIN_1", "samplingRate"):
        entries = [(shape, kind) for n, shape, kind in header if n == name]
        require(len(entries) == 1, "unique_header")
        chosen[name] = entries[0]
    (channels, samples), kind = chosen["eeg"]
    require(channels == 257 and 0 < samples <= 200000 and kind == "double", "eeg_header")
    shape, kind = chosen["DIN_1"]
    require(len(shape) == 2 and shape[0] == 4 and 0 < shape[1] <= 10000
            and kind == "cell", "din_header")
    rate_shape, rate_kind = chosen["samplingRate"]
    require(rate_shape == (1, 1) and rate_kind in
            {"double", "single", "int8", "uint8", "int16", "uint16", "int32", "uint32",
             "int64", "uint64"}, "rate_header")
    remaining()
    data = loader(row["path"], variable_names=VARIABLES, squeeze_me=False, struct_as_record=True,
                  verify_compressed_data_integrity=True)
    require(set(VARIABLES) <= set(data) <= set(VARIABLES)|{"__header__", "__globals__", "__version__"},
            "loaded_variable_whitelist")
    require(float(_scalar(data["samplingRate"])) == 250, "rate_value")
    require(isinstance(data["DIN_1"], np.ndarray) and data["DIN_1"].dtype == object
            and data["DIN_1"].shape == shape, "loaded_din_shape")
    return data["DIN_1"], samples


def inspect():
    import numpy as np

    from cfeg.mamem_reference_residual_v1 import extract, summarize

    values = np.empty((10, 2, 5, 3))
    hashes, costs = {}, {}
    for r, run in enumerate(("a", "b")):
        for p, subject in enumerate(SUBJECTS):
            name = subject+run
            row = next(row for row in specification() if row["subject"] == subject
                       and row["run"] == run)
            save(RUN / ("attempt_"+name+".json"), {"name": name, "started_utc": now(),
                 "mat_sha256": row["sha256"], "variable_names": VARIABLES})
            din, samples = read_din(row)
            records = extract(din, samples)
            for trial in records:
                values[p, r, trial["label"], trial["repeat"]] = trial["frequency_hz"]
            record = {"subject": subject, "run": run, "mat_sha256": row["sha256"],
                      "receipt_sha256": row["receipt_sha256"], "total_samples": samples,
                      "records": records, "completed_utc": now(), "loaded_variables": VARIABLES}
            save(RUN / ("record_"+name+".json"), record)
            hashes[name] = sha(RUN / ("record_"+name+".json"))
            if run == "a":
                first = [row for row in records if row["repeat"] == 0]
                costs[subject] = {"selected_trials": 5, "selected_stimulus_seconds": 25,
                                  "analyzed_din_seconds": 10,
                                  "selected_elapsed_prefix_seconds":
                                  max(row["trial_end0"] for row in first)/250,
                                  "full_a_record_seconds": samples/250,
                                  "query_ready_elapsed": "UNKNOWN", "setup_seconds": "UNKNOWN"}
            del din
        if run == "a":
            save(RUN / "a_frozen.json", {"created_utc": now(), "event_windows": 150,
                 "record_sha256": dict(hashes), "frequencies_hz": values[:, 0].tolist()})
    return {"schema": "cfeg.mamem-reference-residual-v1.result", "completed_utc": now(),
            "scope": SCOPE, "record_sha256": hashes, "a_frozen_sha256": sha(RUN / "a_frozen.json"),
            "frequencies_hz": values.tolist(), "summary": summarize(values), "costs": costs}


def worker():
    global WALL_END
    WALL_END = time.monotonic()+180
    for limit, cap in ((resource.RLIMIT_AS, 2*1024**3), (resource.RLIMIT_CPU, 120),
                       (resource.RLIMIT_FSIZE, CAP)):
        resource.setrlimit(limit, (cap, cap))
    manifest()
    start = read_json(RUN / "started.json")
    require(start["parent_pid"] == os.getppid() and start["attempt"] == 1
            and start["status"] == "STARTED" and start["manifest_sha256"] == sha(RUN / "manifest.json")
            and not (RUN / "terminal.json").exists(), "active_parent")
    save(RUN / "worker_claim.json", {"pid": os.getpid(), "parent_pid": os.getppid(), "attempt": 1,
         "started_utc": now(), "manifest_sha256": start["manifest_sha256"]})
    result = inspect()
    remaining()
    save(RUN / "result.json", result)


def execute():
    global WALL_END
    WALL_END = time.monotonic()+180
    manifest()
    require(shutil.disk_usage(ROOT).free >= 8*1024**3, "disk_reserve")
    start = {"status": "STARTED", "attempt": 1, "parent_pid": os.getpid(), "started_utc": now(),
             "manifest_sha256": sha(RUN / "manifest.json")}
    save(RUN / "started.json", start)
    terminal = {"started_utc": start["started_utc"], "attempts": 1, "fits": 0,
                "manifest_sha256": start["manifest_sha256"]}
    try:
        env = dict(os.environ, OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1",
                   NUMEXPR_NUM_THREADS="1", BLIS_NUM_THREADS="1", PYTHONPATH=str(ROOT / "src"),
                   PYTHONDONTWRITEBYTECODE="1")
        with (RUN / "stdout.txt").open("xb") as out, (RUN / "stderr.txt").open("xb") as err:
            timeout = remaining()
            child = subprocess.Popen([sys.executable, __file__, "worker"], stdout=out, stderr=err,
                                     env=env)
            try:
                child.wait(timeout=timeout)
                require(child.returncode == 0, "worker_failed")
            finally:
                if child.poll() is None:
                    child.kill()
                child.wait()
        result = read_json(RUN / "result.json")
        require(result["schema"] == "cfeg.mamem-reference-residual-v1.result"
                and result["scope"] == SCOPE and len(result["record_sha256"]) == 20,
                "result_scope")
        manifest()
        terminal.update(status="COMPLETE", result_sha256=sha(RUN / "result.json"))
        remaining()
    except Exception as error:  # noqa: BLE001 -- no retry, retain partial records
        terminal.pop("result_sha256", None)
        terminal.update(status="STOPPED_NO_RETRY", error_type=type(error).__name__,
                        reason=str(error) if isinstance(error, ValueError) else "see_bounded_stderr")
    terminal.update(ended_utc=now(), started_sha256=sha(RUN / "started.json", timed=False),
                    worker_claim_sha256=sha(RUN / "worker_claim.json", timed=False)
                    if (RUN / "worker_claim.json").is_file() else None)
    save(RUN / "terminal.json", terminal)
    print(json.dumps(terminal))
    if terminal["status"] != "COMPLETE":
        raise SystemExit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("freeze", "execute", "worker"))
    {"freeze": freeze, "execute": execute, "worker": worker}[parser.parse_args().mode]()
