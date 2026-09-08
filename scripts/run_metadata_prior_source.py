"""Single frozen source39 fit/evaluation; never reads raw or held EEG."""

from __future__ import annotations

import hashlib
import json
import os
import signal
import subprocess
import sys
import time
import zipfile  # noqa: F401 -- warm archive support before runtime guard
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import scipy

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from export_metadata_prior_source import open_regular, publish, validate_cache

from cfeg.analysis import metadata_prior_source as core

CANONICAL = Path("/home/whwovy/califreeEEG")
PLAN = CANONICAL / "configs/analysis/metadata_prior_source39_v1.json"
PLAN_SHA = "ed67cc1f361ed89c37b6f7c5df1e9170b78484e428a5105020485c8c03e3934b"
OUTPUT = Path("/home/whwovy/metadata-prior-source39-v1/analysis")
ORIGINAL_NATIVE = Path("/home/whwovy/metadata-prior-source39-v1/native")
RECOVERED_NATIVE = Path("/home/whwovy/metadata-prior-source39-v1/native-cold-r1")
RECOVERY = {
    "original_root": str(ORIGINAL_NATIVE),
    "start_sha256": "0af0f7d02e7a5b31494133fb1a99924bbcc55c7713df49028479552ac15df1c7",
    "preserved_artifact_count": 39,
    "reason": "infrastructure_terminal_source_rehash_allowlist",
    "unchanged_plan_sha256": PLAN_SHA,
    "actual_native_root": str(RECOVERED_NATIVE),
}


def native_root(plan):
    declared = Path(plan["execution"]["native_root"])
    return RECOVERED_NATIVE if declared == ORIGINAL_NATIVE else declared


def digest(path):
    with open_regular(path) as stream:
        h = hashlib.sha256()
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
        return h.hexdigest()


def read_json(path, expected=None):
    with open_regular(path) as stream:
        encoded = stream.read()
    actual = hashlib.sha256(encoded).hexdigest()
    if expected is not None and actual != expected:
        raise ValueError("Input SHA mismatch: " + str(path))
    return json.loads(encoded), actual


def projection(plan):
    spec = plan["source_projection"]
    value, _ = read_json(Path(spec["path"]), spec["sha256"])
    fields = {
        "manifest_sha256",
        "packets",
        "returned_rows",
        "returned_packets",
        "returned_subject_ids",
        "columns",
    }
    if set(value) != fields | set(spec["envelope_provenance"]):
        raise ValueError("Projection requires the exact11-key envelope")
    if any(value[k] != v for k, v in spec["envelope_provenance"].items()):
        raise ValueError("Projection provenance differs")
    ids = plan["source_subject_ids"]
    if (
        value["returned_subject_ids"] != ids
        or value["returned_packets"] != len(ids) * 20
        or value["returned_rows"] != spec["returned_rows"]
        or value["manifest_sha256"] != spec["manifest_sha256"]
        or value["columns"] != spec["columns"]
    ):
        raise ValueError("Projection content receipt differs")
    z, order = np.full((len(ids), 2, 10, 8), np.nan), np.full(len(ids), -1, dtype=int)
    seen = set()
    for packet in value["packets"]:
        if set(packet) != {
            "subject_id",
            "interface",
            "block_id",
            "impedance_kohm",
            "headband_order",
            "condition_period",
        }:
            raise ValueError("Unexpected projection packet fields")
        if (
            packet["subject_id"] not in ids
            or packet["interface"] not in plan["interfaces"]
            or type(packet["block_id"]) is not int
            or packet["block_id"] not in range(10)
        ):
            raise ValueError("Projection packet outside source boundary")
        p, i, b = (
            ids.index(packet["subject_id"]),
            plan["interfaces"].index(packet["interface"]),
            packet["block_id"],
        )
        if (p, i, b) in seen or packet["headband_order"] not in plan["interfaces"]:
            raise ValueError("Duplicate packet or unknown order")
        first = plan["interfaces"].index(packet["headband_order"])
        if order[p] not in (-1, first) or packet["condition_period"] != (
            "first" if first == i else "second"
        ):
            raise ValueError("Inconsistent acquisition order")
        values = np.asarray(packet["impedance_kohm"], dtype=float)
        core.m_features(values[None])
        z[p, i, b], order[p] = values, first
        seen.add((p, i, b))
    if len(seen) != len(ids) * 20:
        raise ValueError("Incomplete source-only projection")
    return z, order


def native_manifest(plan):
    root = native_root(plan)
    manifest, sha = read_json(root / "result.json")
    if root == RECOVERED_NATIVE and (
        manifest.get("recovery") != RECOVERY
        or digest(ORIGINAL_NATIVE / "start.json") != RECOVERY["start_sha256"]
    ):
        raise ValueError("Exact infrastructure recovery receipt required")
    if (
        manifest["status"] != "COMPLETE"
        or manifest["study_id"] != plan["study_id"]
        or manifest["plan_sha256"] != PLAN_SHA
        or manifest["source_subject_ids"] != plan["source_subject_ids"]
    ):
        raise ValueError("Native manifest identity differs")
    if [r["subject"] for r in manifest["files"]] != plan["source_subject_ids"]:
        raise ValueError("Native participant manifest differs")
    if digest(root / "start.json") != manifest["start_sha256"]:
        raise ValueError("Native start hash mismatch")
    for row in manifest["files"]:
        if row["filename"] != f"S{row['subject']:03d}.npz":
            raise ValueError("Native path escape or identity mismatch")
    return manifest, sha


def load_native(row, plan):
    path = native_root(plan) / row["filename"]
    if digest(path) != row["sha256"] or path.stat().st_size != row["bytes"]:
        raise ValueError("Native array hash/size mismatch")
    with open_regular(path) as stream, np.load(stream, allow_pickle=False) as archive:
        arrays = {key: archive[key] for key in archive.files}
    validate_cache(arrays, plan)
    if digest(path) != row["sha256"]:
        raise ValueError("Native cache changed while loading")
    return arrays


def guard_for(plan, output):
    native = native_root(plan)
    readable = {
        native / "start.json",
        native / "result.json",
        Path(plan["source_projection"]["path"]),
        PLAN,
        ORIGINAL_NATIVE / "start.json",
    }
    readable |= {native / f"S{p:03d}.npz" for p in plan["source_subject_ids"]}
    writable = {
        output / name for name in plan["analysis_cache"]["artifacts"] if name != "audit.json"
    }

    def guard(event, args):
        if event in ("subprocess.Popen", "socket.connect", "socket.getaddrinfo"):
            raise RuntimeError("No processes/network during source analysis")
        if event != "open" or not isinstance(args[0], (str, bytes, os.PathLike)):
            return
        path = Path(os.fsdecode(args[0]))
        flags = args[2] if len(args) > 2 and isinstance(args[2], int) else 0
        if flags & (os.O_CREAT | os.O_WRONLY | os.O_RDWR | os.O_TRUNC) and path not in writable:
            raise RuntimeError("Only new analysis artifacts may be written")
        if (
            path.suffix.lower()
            in (
                ".json",
                ".jsonl",
                ".mat",
                ".npz",
                ".npy",
                ".h5",
                ".hdf5",
                ".parquet",
                ".pkl",
                ".db",
                ".sqlite",
            )
            and path not in readable | writable
        ):
            raise RuntimeError("Unapproved raw/held/old data input")

    return guard


def run():
    if ROOT != CANONICAL or OUTPUT.resolve() != OUTPUT or OUTPUT.exists():
        raise ValueError("Require canonical main and new analysis attempt")
    plan, _ = read_json(PLAN, PLAN_SHA)
    if subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True):
        raise ValueError("Commit clean source before execution")
    if (
        subprocess.check_output(["git", "branch", "--show-current"], cwd=ROOT, text=True).strip()
        != "main"
    ):
        raise ValueError("Only integrated main may execute")
    if (
        Path(sys.executable) != Path(plan["execution"]["analysis_python"])
        or np.__version__ != "1.26.4"
        or scipy.__version__ != "1.15.3"
    ):
        raise ValueError("Use existing pinned project environment")
    if any(
        os.environ.get(k) != "1"
        for k in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS")
    ):
        raise ValueError("Require BLAS1")
    code = [
        "src/cfeg/analysis/metadata_prior_source.py",
        "scripts/run_metadata_prior_source.py",
        "scripts/export_metadata_prior_source.py",
        plan["operator"]["path"],
        plan["q_helper"]["path"],
    ]
    hashes = {p: digest(ROOT / p) for p in code}
    for name in ("operator", "q_helper"):
        if hashes[plan[name]["path"]] != plan[name]["sha256"]:
            raise ValueError("Frozen helper changed")
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    started, began = datetime.now(timezone.utc).isoformat(), time.monotonic()
    common = {
        "plan_sha256": PLAN_SHA,
        "source_commit": commit,
        "source_hashes": hashes,
        "python": sys.version.split()[0],
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "native_root": str(native_root(plan)),
        "native_recovery": RECOVERY if native_root(plan) == RECOVERED_NATIVE else None,
    }
    OUTPUT.mkdir(mode=0o700)
    quota = plan["execution"]["budget_bytes"]
    start_receipt = publish(
        OUTPUT / "start.json",
        {
            "schema": "cfeg.metadata-prior-source.start.v1",
            "study_id": plan["study_id"],
            "started_at": started,
            **common,
        },
        quota,
    )
    sys.dont_write_bytecode = True
    sys.addaudithook(guard_for(plan, OUTPUT))

    def timeout(_signum, _frame):
        raise TimeoutError("Analysis stage exceeded fixed runtime ceiling")

    signal.signal(signal.SIGALRM, timeout)
    signal.alarm(plan["execution"]["max_seconds_per_stage"])
    manifest, manifest_sha = native_manifest(plan)
    z, order = projection(plan)
    q = np.empty(plan["analysis_cache"]["features"]["q"])
    target = np.empty(plan["analysis_cache"]["features"]["target"])
    for p, row in enumerate(manifest["files"]):
        arrays = load_native(row, plan)
        q[p], target[p] = core.subject_features(arrays, z[p], order[p], plan)
        del arrays
        print(json.dumps({"stage": "features", "completed": p + 1}), flush=True)
    features = {"q": q, "target": target, "z": z, "order": order}
    feature_receipt = publish(OUTPUT / "features.npz", features, quota, npz=True)
    freezes = core.fit_all(features, plan, progress=lambda msg: print(msg, flush=True))
    freezes["plan_sha256"] = PLAN_SHA
    freeze_receipt = publish(OUTPUT / "fold-freezes.json", freezes, quota)
    # Every learned coefficient and choice is published before outer query scoring.
    scores = np.empty(plan["analysis_cache"]["scores"]["scores"])
    anchors = np.empty(plan["analysis_cache"]["scores"]["a0_scores"])
    rows, max_error = [], 0.0
    for p, entry in enumerate(manifest["files"]):
        arrays = load_native(entry, plan)
        new, scores[p], anchors[p], error = core.evaluate_subject(
            arrays, features, p, freezes, plan
        )
        rows.extend(new)
        max_error = max(max_error, error)
        del arrays
        print(json.dumps({"stage": "evaluation", "completed": p + 1}), flush=True)
    score_receipt = publish(
        OUTPUT / "scores.npz", {"scores": scores, "a0_scores": anchors}, quota, npz=True
    )
    reports = core.summarize(rows, features, freezes, plan)
    for path, expected in hashes.items():
        if digest(ROOT / path) != expected:
            raise ValueError("Source changed during analysis")
    if (
        digest(Path(plan["source_projection"]["path"])) != plan["source_projection"]["sha256"]
        or digest(native_root(plan) / "result.json") != manifest_sha
    ):
        raise ValueError("Input manifest/projection changed during analysis")
    result = {
        "schema": "cfeg.metadata-prior-source.result.v1",
        "study_id": plan["study_id"],
        "provenance": {
            **common,
            "start_sha256": start_receipt["sha256"],
            "native_manifest_sha256": manifest_sha,
            "projection_sha256": plan["source_projection"]["sha256"],
            "features_sha256": feature_receipt["sha256"],
            "freezes_sha256": freeze_receipt["sha256"],
            "scores_sha256": score_receipt["sha256"],
        },
        "rows": rows,
        **reports,
        "compatibility": {
            "max_native_correlation_error": max_error,
            "native_prediction_mismatches": 0,
        },
        "finished_at": datetime.now(timezone.utc).isoformat(),
        "elapsed_seconds": time.monotonic() - began,
    }
    publish(OUTPUT / "result.json", result, quota)
    signal.alarm(0)
    print(json.dumps(result["verdict"]), flush=True)
    return result


if __name__ == "__main__":
    run()
