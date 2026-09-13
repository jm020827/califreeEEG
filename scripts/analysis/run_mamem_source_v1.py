"""One bounded frozen MAMEM source candidate: generated, development, real.

No downloads, held60, replacement participants, retries, or tuning. Heavy arrays
and full model audit stay in the explicitly selected outside-Git cache directory.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import resource
import shutil
import subprocess
import sys
import time
import zlib
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SOURCE = Path("/home/whwovy/data/mamem_i_v1_20260913")
CACHE = Path("/home/whwovy/data/mamem_recorded_event_source_v1")
RUN = ROOT / "docs/reports/mamem_recorded_event_source_v1_run"
DEADLINE = "2026-09-13T13:20:00+00:00"
INVENTORY_SHA = "67791a601d253d7b6d3edf080a1781329991e41e944a3bea5b759cfaed34fb41"
ARCHIVE_SHA = {
    "EEG-SSVEP-Part1.rar": "09fd628b903a23d4fab82dbb69664a22e3063f70c51fb34268718f5de66b4995",
    "EEG-SSVEP-Part2.rar": "2fde94585ae96e7b2076f6adab38b7e2885aff3c26963ffe9aa7d2f5c783edfa",
}
DEV_SHA = "57a72c3fde0ff3bc9aaae10721cd7cb450bda63eb5299a96f41704ea696ad10a"
CODE = [
    "scripts/analysis/run_mamem_source_v1.py",
    "scripts/analysis/prepare_mamem_i_probe_v1.py",
    "scripts/analysis/probe_mamem_i_din_v1.py",
    "src/cfeg/mamem_events_v1.py",
    "src/cfeg/mamem_signal_v1.py",
    "src/cfeg/mamem_shrinkage_v1.py",
    "docs/mamem_recorded_event_source_v1_contract.md",
]


def require(value, reason):
    if not value:
        raise ValueError(reason)


def now():
    return datetime.now(timezone.utc).isoformat()


def remaining(cap):
    delta = datetime.fromisoformat(DEADLINE).timestamp() - time.time()
    require(delta > 0, "round_deadline")
    return min(cap, delta)


def sha(path, operation_deadline=None):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024**2), b""):
            remaining(1)
            require(
                operation_deadline is None or time.monotonic() < operation_deadline,
                "hash_operation_deadline",
            )
            digest.update(chunk)
    return digest.hexdigest()


def serial(value):
    if hasattr(value, "tolist"):
        return value.tolist()
    raise TypeError("unsupported_json")


def save(path, value, cap=2 * 1024**2):
    raw = json.dumps(value, default=serial, allow_nan=False, separators=(",", ":")).encode() + b"\n"
    require(len(raw) <= cap, "json_cap")
    with Path(path).open("xb") as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())


def append(phase, kind, payload):
    remaining(1)
    path = RUN / (phase + "_ledger.jsonl")
    require(not path.exists() or path.stat().st_size < 256 * 1024, "ledger_cap")
    raw = json.dumps(dict(kind=kind, utc=now(), **payload), allow_nan=False).encode() + b"\n"
    with path.open("ab") as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())


def budget():
    require(shutil.disk_usage(CACHE.parent).free >= 14 * 1024**3, "free_reserve")
    if CACHE.exists():
        files = [p for p in CACHE.iterdir() if p.is_file()]
        require(
            sum(p.stat().st_size for p in files if p.suffix == ".mat") <= 4 * 1024**3,
            "extraction_cap",
        )
        require(sum(p.stat().st_size for p in files if p.suffix != ".mat") <= 1024**3, "cache_cap")


def freeze():
    remaining(1)
    budget()
    require(not RUN.exists() and not CACHE.exists(), "fresh_round_paths_required")
    require(sha(SOURCE / "inventory.json") == INVENTORY_SHA, "inventory_hash")
    inventory = json.loads((SOURCE / "inventory.json").read_bytes())
    selected = []
    for listing in inventory["listings"]:
        archive = listing["archive"]["Path"]
        require(
            str(Path(archive).parent) == str(SOURCE) and Path(archive).name in ARCHIVE_SHA,
            "archive_scope",
        )
        for member in listing["members"]:
            match = re.fullmatch(r"(?:EEG-SSVEP-Part[12]/)?(S\d{3})([ab])\.mat", member["Path"])
            if match:
                size = int(member["Size"])
                require(
                    0 < size <= 512 * 1024**2 and member["Folder"] == "-", "member_size_or_kind"
                )
                selected.append(
                    {
                        "subject": match[1],
                        "run": match[2],
                        "archive": archive,
                        "member": member["Path"],
                        "bytes": size,
                        "crc32": member["CRC"].lower(),
                    }
                )
    selected.sort(key=lambda r: (r["subject"], r["run"]))
    require(
        [(r["subject"], r["run"]) for r in selected]
        == [(f"S{n:03}", r) for n in range(1, 12) for r in "ab"],
        "exact_22_members",
    )
    require(sum(r["bytes"] for r in selected[1:]) <= 4 * 1024**3, "planned_extraction_cap")
    RUN.mkdir()
    CACHE.mkdir()
    save(
        RUN / "manifest.json",
        {
            "schema": "cfeg.mamem-source-v1",
            "created_utc": now(),
            "deadline_utc": DEADLINE,
            "selected": selected,
            "archives": ARCHIVE_SHA,
            "inventory_sha256": INVENTORY_SHA,
            "code_sha256": {name: sha(ROOT / name) for name in CODE},
            "development_subject": "S001",
            "max_real_fits": 80,
            "max_generated_fits": 32,
            "query_ready_elapsed": "UNKNOWN",
        },
    )


def manifest():
    remaining(1)
    cfg = json.loads((RUN / "manifest.json").read_bytes())
    require(cfg["deadline_utc"] == DEADLINE, "frozen_deadline")
    require(
        all(sha(ROOT / name) == expected for name, expected in cfg["code_sha256"].items()),
        "frozen_code_hash",
    )
    return cfg


def child_limits(kind):
    memory, cpu = (2, 90) if kind == "io" else (4, 600)
    resource.setrlimit(resource.RLIMIT_AS, (memory * 1024**3, memory * 1024**3))
    resource.setrlimit(resource.RLIMIT_CPU, (cpu, cpu))
    resource.setrlimit(resource.RLIMIT_FSIZE, (512 * 1024**2, 512 * 1024**2))


def subprocess_child(kind, args):
    env = dict(
        os.environ,
        OPENBLAS_NUM_THREADS="1",
        OMP_NUM_THREADS="1",
        MKL_NUM_THREADS="1",
        NUMEXPR_NUM_THREADS="1",
        PYTHONPATH=str(ROOT / "src"),
    )
    # stdout is a small status object; stderr has a fixed parent file-size bound.
    tag = "_".join(args)
    phase = (
        args[1]
        if args[0] == "fit-child"
        else ("development" if args[1].startswith("S001") else "real")
    )
    save(
        RUN / (tag + "_authorization.json"),
        {
            "phase": phase,
            "kind": kind,
            "args": args,
            "parent_pid": os.getpid(),
            "script_sha256": sha(__file__),
        },
    )
    with (
        (RUN / (tag + "_stdout.txt")).open("xb") as out,
        (RUN / (tag + "_stderr.txt")).open("xb") as err,
    ):
        deadline = time.monotonic() + remaining(120 if kind == "io" else 600)
        proc = subprocess.Popen(
            [sys.executable, str(Path(__file__).resolve()), *args], stdout=out, stderr=err, env=env
        )
        try:
            while proc.poll() is None:
                require(time.monotonic() < deadline, "child_wall_timeout")
                require(out.tell() <= 65536 and err.tell() <= 65536, "child_capture_cap")
                time.sleep(0.05)
            require(proc.returncode == 0, "child_nonzero_exit")
            require(out.tell() <= 65536 and err.tell() <= 65536, "child_capture_cap")
        finally:
            if proc.poll() is None:
                proc.kill()
            proc.wait()


def extract(row, phase):
    operation_deadline = time.monotonic() + remaining(120)
    parent_cpu_started = time.process_time()
    name = row["subject"] + row["run"]
    append(phase, "EXTRACTION_STARTED", {"file": name, "member": row["member"]})
    if name == "S001a":
        path = SOURCE / "development_first.mat"
        require(
            path.stat().st_size == row["bytes"] and sha(path, operation_deadline) == DEV_SHA,
            "pinned_existing_development_mat",
        )
    else:
        from prepare_mamem_i_probe_v1 import bounded_run

        path = CACHE / (name + ".mat")
        budget()
        with path.open("xb") as stream:
            code, _, _ = bounded_run(
                [
                    "/usr/bin/unar",
                    "-q",
                    "-nr",
                    "-k",
                    "skip",
                    "-o",
                    "-",
                    row["archive"],
                    row["member"],
                ],
                row["bytes"],
                min(remaining(120), max(0.001, operation_deadline - time.monotonic())),
                stream,
                disk_guard=True,
            )
            stream.flush()
            os.fsync(stream.fileno())
        require(code == 0 and path.stat().st_size == row["bytes"], "extraction_failed_or_size")
    crc = 0
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024**2), b""):
            remaining(1)
            require(time.monotonic() < operation_deadline, "extraction_operation_deadline")
            crc = zlib.crc32(chunk, crc)
    require(f"{crc:08x}" == row["crc32"], "mat_crc")
    receipt = dict(row, path=str(path), sha256=sha(path, operation_deadline))
    require(
        time.monotonic() < operation_deadline and time.process_time() - parent_cpu_started < 90,
        "extraction_operation_limit",
    )
    save(CACHE / (name + "_input.json"), receipt)
    append(
        phase,
        "EXTRACTION_COMPLETE",
        {"file": name, "bytes": row["bytes"], "sha256": receipt["sha256"]},
    )


def io_child(name):
    cfg = manifest()
    import numpy as np
    from scipy.io import loadmat, whosmat

    from cfeg.mamem_events_v1 import parse_main_trials
    from cfeg.mamem_signal_v1 import analyze_window

    row = json.loads((CACHE / (name + "_input.json")).read_bytes())
    expected = [r for r in cfg["selected"] if r["subject"] + r["run"] == name]
    require(
        len(expected) == 1 and all(row.get(k) == v for k, v in expected[0].items()),
        "input_selection_binding",
    )
    path = Path(row["path"])
    exact_path = SOURCE / "development_first.mat" if name == "S001a" else CACHE / (name + ".mat")
    require(
        path == exact_path and path.is_file() and path.stat().st_size == row["bytes"],
        "exact_input_path_and_size",
    )
    require(not path.is_symlink() and sha(path) == row["sha256"], "mat_changed")
    headers = whosmat(path)
    for variable in ("eeg", "DIN_1", "samplingRate"):
        require(sum(v[0] == variable for v in headers) == 1, "missing_or_duplicate_variable")
    shapes = {v[0]: v[1] for v in headers}
    classes = {v[0]: v[2] for v in headers}
    require(classes["eeg"] == "double" and classes["DIN_1"] == "cell", "variable_classes")
    require(
        shapes["samplingRate"] == (1, 1)
        and classes["samplingRate"]
        in (
            "double",
            "single",
            "uint8",
            "int8",
            "uint16",
            "int16",
            "uint32",
            "int32",
            "uint64",
            "int64",
        ),
        "sampling_scalar_header",
    )
    require(
        len(shapes["eeg"]) == 2 and shapes["eeg"][0] == 257 and shapes["eeg"][1] < 500000,
        "eeg_header",
    )
    require(
        len(shapes["DIN_1"]) == 2 and shapes["DIN_1"][0] == 4 and shapes["DIN_1"][1] <= 10000,
        "din_header",
    )
    loaded = loadmat(
        path,
        variable_names=["eeg", "DIN_1", "samplingRate"],
        squeeze_me=False,
        struct_as_record=True,
        verify_compressed_data_integrity=True,
    )
    require(
        set(loaded) <= {"__header__", "__version__", "__globals__", "eeg", "DIN_1", "samplingRate"}
        and all(k in loaded for k in ("eeg", "DIN_1", "samplingRate")),
        "loaded_key_whitelist",
    )
    rate = loaded["samplingRate"]
    require(
        rate.size == 1 and rate.dtype.kind in "iuf" and float(rate.item()) == 250.0, "sampling_rate"
    )
    records = parse_main_trials(
        loaded["DIN_1"], shapes["eeg"][1], include_metadata=row["run"] == "a"
    )
    arrays = {key: [] for key in ("covariance", "factors", "q", "q2", "zero_shot")}
    for record in records:
        result = analyze_window(loaded["eeg"], record["start0"], record["end0"])
        for key, values in arrays.items():
            values.append(result[key])
    feature_path = CACHE / (name + "_features.npz")
    with feature_path.open("xb") as stream:
        np.savez(stream, **{key: np.stack(value) for key, value in arrays.items()})
    require(feature_path.stat().st_size <= 16 * 1024**2, "feature_file_cap")
    save(
        CACHE / (name + "_trials.json"),
        {
            "subject": row["subject"],
            "run": row["run"],
            "records": records,
            "feature_path": str(feature_path),
            "feature_sha256": sha(feature_path),
            "mat_sha256": row["sha256"],
            "fully_decoded_variables": ["eeg", "DIN_1", "samplingRate"],
            "numerical_eeg_scope": "15 fixed windows, rows[0:256], 500 samples each",
            "physical_units": "UNVERIFIED",
            "labels": "continuous DIN inference; not independent ground truth",
            "metadata_extracted": row["run"] == "a",
        },
    )
    print(
        json.dumps(
            {"status": "FEATURES_COMPLETE", "file": name, "windows": len(records), "fits": 0}
        )
    )


def load_data(subjects):
    import numpy as np

    data = {}
    receipts = [json.loads(line) for line in (RUN / "real_ledger.jsonl").read_text().splitlines()]
    completed = {
        r["file"]: r["trial_receipt_sha256"] for r in receipts if r["kind"] == "IO_COMPLETE"
    }
    for subject in subjects:
        data[subject] = {}
        for run in "ab":
            name = subject + run
            trial_path = CACHE / (name + "_trials.json")
            require(sha(trial_path) == completed.get(name), "trial_receipt_binding")
            receipt = json.loads(trial_path.read_bytes())
            require(
                receipt["subject"] == subject
                and receipt["run"] == run
                and receipt["feature_path"] == str(CACHE / (name + "_features.npz")),
                "exact_feature_path_and_role",
            )
            require(sha(receipt["feature_path"]) == receipt["feature_sha256"], "feature_changed")
            records = receipt["records"]
            with np.load(receipt["feature_path"], allow_pickle=False) as arrays:
                require(
                    set(arrays.files) == {"covariance", "factors", "q", "q2", "zero_shot"},
                    "cache_keys",
                )
                for n, record in enumerate(records):
                    record["metadata"] = (
                        None if record["metadata"] is None else np.array(record["metadata"])
                    )
                    record.update({key: arrays[key][n].copy() for key in arrays.files})
            data[subject][run] = records
    return data


def generated_data():
    import numpy as np

    from cfeg.mamem_events_v1 import FREQUENCIES, parse_main_trials
    from cfeg.mamem_signal_v1 import analyze_window

    data = {}
    rng = np.random.default_rng(20260913)
    labels = [0, 1, 2, 3, 4, 0, 1, 2] + [j for j in (3, 0, 4, 1, 2) for _ in range(3)]
    total = 101 + 2500 * 22 + 1750
    for person in range(4):
        subject = f"FAKE{person}"
        data[subject] = {}
        for run in "ab":
            groups = []
            for g, label in enumerate(labels):
                period = 1000 / (2 * FREQUENCIES[label])
                rel = np.arange(int(5000 / period) + 1) * period
                # Same samples/counts but variable tiny timestamp residuals: a
                # generated input/integration canary, not physical timing evidence.
                times = (
                    1000
                    + 10000 * g
                    + rel
                    + (0.04 + person * 0.03) * np.sin(np.arange(len(rel)) * 1.7)
                )
                samples = 101 + 2500 * g + np.rint(rel / 4).astype(int)
                group = np.empty((4, len(rel)), object)
                group[0] = None
                group[2] = None
                group[1] = times
                group[3] = samples
                groups.append(group)
            records = parse_main_trials(
                np.concatenate(groups, axis=1), total, include_metadata=run == "a"
            )
            eeg = np.zeros((257, total), dtype=np.float64)
            for row in records:
                t = np.arange(500) / 250
                waveform = np.sin(2 * np.pi * FREQUENCIES[row["label"]] * t)
                spatial = rng.normal(size=(256, 1))
                eeg[:256, row["start0"] : row["end0"]] = (
                    spatial * waveform + rng.normal(size=(256, 500)) * 0.5
                )
                row.update(analyze_window(eeg, row["start0"], row["end0"]))
            data[subject][run] = records
    return data


def fit_child(phase):
    manifest()
    from cfeg.mamem_shrinkage_v1 import preflight, prepare, run_folds, summarize

    data = (
        generated_data() if phase == "generated" else load_data([f"S{n:03}" for n in range(2, 12)])
    )
    prepared = prepare(data)
    folds = preflight(prepared)
    expected_ids = (
        [f"FAKE{n}" for n in range(4)]
        if phase == "generated"
        else [f"S{n:03}" for n in range(2, 12)]
    )
    require(sorted(data) == expected_ids, "exact_subject_coverage")
    require(
        [(f["target"], f["k"]) for f in folds] == [(s, k) for s in expected_ids for k in (1, 2)],
        "exact_fold_coverage",
    )
    # Persist ALL fold eligibility before the first fit or efficacy output.
    eligibility = [
        {
            key: f[key]
            for key in ("target", "k", "source", "sham", "prior_audit", "degenerate_oracles")
        }
        for f in folds
    ]
    save(
        RUN / (phase + "_preflight.json"),
        {
            "folds": eligibility,
            "total_folds": len(folds),
            "total_scalar_oracles": sum(f["y"].size for f in folds),
            "metadata_extracted_from_query": False,
        },
    )
    result = run_folds(prepared, folds, lambda kind, value: append(phase, kind, value))
    require(result["fits"] == (32 if phase == "generated" else 80), "fit_count")
    artifact = CACHE / (phase + "_full_result.json")
    save(artifact, result, cap=16 * 1024**2)
    summary = {
        "phase": phase,
        "fits": result["fits"],
        "status": "COMPLETE",
        "result_path": str(artifact),
        "result_sha256": sha(artifact),
        "summary": summarize(result),
        "completed_utc": now(),
    }
    if phase == "generated":
        summary["summary"]["decision"] = "GENERATED_INTEGRATION_ONLY_NOT_EFFICACY"
    save(RUN / (phase + "_summary.json"), summary)
    print(json.dumps({"phase": phase, "status": "COMPLETE", "fits": result["fits"]}))


def execute(phase):
    cfg = manifest()
    require(phase in ("generated", "development", "real"), "phase")
    if phase != "generated":
        require(
            json.loads((RUN / "generated_terminal.json").read_bytes())["status"] == "COMPLETE",
            "generated_required",
        )
    if phase == "real":
        require(
            json.loads((RUN / "development_terminal.json").read_bytes())["status"] == "COMPLETE",
            "development_required",
        )
    save(
        RUN / (phase + "_attempt.json"),
        {"phase": phase, "started_utc": now(), "attempt": 1, "parent_pid": os.getpid()},
    )
    terminal = {"phase": phase, "started_utc": now(), "attempts": 1}
    try:
        budget()
        if phase == "generated":
            subprocess_child("fit", ["fit-child", phase])
        else:
            if phase == "development":
                # Rehash both pinned archives before ANY newly selected extraction.
                for name, expected in ARCHIVE_SHA.items():
                    require(sha(SOURCE / name) == expected, "archive_hash")
                save(RUN / "archive_verification.json", {"archives": ARCHIVE_SHA, "utc": now()})
            for row in cfg["selected"]:
                if (row["subject"] == "S001") != (phase == "development"):
                    continue
                extract(row, phase)
                name = row["subject"] + row["run"]
                append(phase, "IO_STARTED", {"file": name})
                subprocess_child("io", ["io-child", name])
                append(
                    phase,
                    "IO_COMPLETE",
                    {"file": name, "trial_receipt_sha256": sha(CACHE / (name + "_trials.json"))},
                )
            if phase == "real":
                subprocess_child("fit", ["fit-child", phase])
        terminal["status"] = "COMPLETE"
    except Exception as exc:  # noqa: BLE001 -- terminal records must preserve every failure without retry
        terminal.update(
            status="STOPPED_NO_RETRY",
            error_type=type(exc).__name__,
            reason=str(exc)
            if isinstance(exc, ValueError)
            or (type(exc).__name__ == "Stop" and type(exc).__module__ == "probe_mamem_i_din_v1")
            else "details_in_bounded_child_logs",
        )
    terminal["ended_utc"] = now()
    save(RUN / (phase + "_terminal.json"), terminal)
    print(json.dumps(terminal))


def claim_child(mode, argument):
    """Exclusive worker claim; cannot invoke child paths outside the active parent."""
    require(mode in ("fit-child", "io-child"), "child_mode")
    if mode == "fit-child":
        require(argument in ("generated", "real"), "fit_phase")
        phase, kind = argument, "fit"
    else:
        require(re.fullmatch(r"S(?:00[1-9]|01[01])[ab]", argument or "") is not None, "io_name")
        phase, kind = ("development" if argument.startswith("S001") else "real"), "io"
    child_limits(kind)  # self-enforced before any NumPy/SciPy/raw/cache loading
    tag = mode + "_" + argument
    auth = json.loads((RUN / (tag + "_authorization.json")).read_bytes())
    attempt = json.loads((RUN / (phase + "_attempt.json")).read_bytes())
    require(
        auth["args"] == [mode, argument] and auth["phase"] == phase and auth["kind"] == kind,
        "child_authorization_scope",
    )
    require(auth["parent_pid"] == attempt["parent_pid"] == os.getppid(), "active_parent_required")
    require(
        auth["script_sha256"] == sha(__file__) and not (RUN / (phase + "_terminal.json")).exists(),
        "active_frozen_attempt",
    )
    if phase != "generated":
        require(
            json.loads((RUN / "generated_terminal.json").read_bytes())["status"] == "COMPLETE",
            "child_generated_gate",
        )
    if phase == "real":
        require(
            json.loads((RUN / "development_terminal.json").read_bytes())["status"] == "COMPLETE",
            "child_development_gate",
        )
    save(
        RUN / (tag + "_claim.json"),
        {"claimed_utc": now(), "pid": os.getpid(), "phase": phase, "one_attempt": True},
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["freeze", "execute", "io-child", "fit-child"])
    parser.add_argument("argument", nargs="?")
    args = parser.parse_args()
    if args.mode == "freeze":
        freeze()
    elif args.mode == "execute":
        execute(args.argument)
    elif args.mode == "io-child":
        claim_child(args.mode, args.argument)
        io_child(args.argument)
    else:
        claim_child(args.mode, args.argument)
        fit_child(args.argument)


if __name__ == "__main__":
    main()
