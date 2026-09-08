"""One source39 native export; closed runners and metadata are never executed/read.

Pure compute functions accept artificial arrays. The human CLI requires the
canonical clean main checkout and frozen contract before creating its attempt.
The Python audit hook is defense in depth, not an OS security sandbox.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import importlib.util
import json
import os
import signal
import stat
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from scipy import io

SOURCE_ROOT = Path("/home/whwovy/califreeEEG")
PLAN_PATH = SOURCE_ROOT / "configs/analysis/metadata_prior_source39_v1.json"
PLAN_SHA256 = "ed67cc1f361ed89c37b6f7c5df1e9170b78484e428a5105020485c8c03e3934b"
RAW_ROOT = Path("/home/whwovy/eeg-data/raw/wearable")
OUTPUT_ROOT = Path("/home/whwovy/metadata-prior-source39-v1/native")
RECOVERY_ROOT = Path("/home/whwovy/metadata-prior-source39-v1/native-cold-r1")
FAILED_START_SHA256 = "0af0f7d02e7a5b31494133fb1a99924bbcc55c7713df49028479552ac15df1c7"
UPSTREAM_ROOT = Path("/home/whwovy/ssvep-author-compatibility-20260907/upstream")
PYTHON_PATH = Path("/home/whwovy/ssvep-author-compatibility-20260907/.venv/bin/python")
SOURCE_IDS = (
    4,
    6,
    8,
    11,
    14,
    21,
    22,
    25,
    28,
    29,
    30,
    31,
    32,
    33,
    37,
    41,
    42,
    43,
    44,
    46,
    54,
    55,
    56,
    61,
    63,
    65,
    67,
    73,
    74,
    77,
    80,
    82,
    83,
    84,
    89,
    92,
    97,
    100,
    102,
)
SAMPLE_COUNTS = (125, 188, 250, 500)
VERSIONS = {
    "numpy": "1.23.4",
    "scipy": "1.13.0",
    "joblib": "1.4.2",
    "scikit-learn": "1.3.0",
    "mat73": "0.63",
    "h5py": "3.11.0",
    "threadpoolctl": "3.5.0",
}


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def exact_path(path):
    path = Path(path)
    if not path.is_absolute() or ".." in path.parts or path.resolve() != path:
        raise ValueError("Require an absolute exact path without symlink/parent traversal")
    if path.is_symlink():
        raise ValueError("Symlink paths are forbidden")
    return path


def open_regular(path):
    path = exact_path(path)
    descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
    stream = os.fdopen(descriptor, "rb")
    info = os.fstat(stream.fileno())
    if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
        stream.close()
        raise ValueError("Require a regular single-link file")
    return stream


def stream_sha256(stream):
    digest = hashlib.sha256()
    for chunk in iter(lambda: stream.read(1024 * 1024), b""):
        digest.update(chunk)
    return digest.hexdigest()


def file_sha256(path):
    with open_regular(path) as stream:
        return stream_sha256(stream)


def read_plan(path):
    with open_regular(path) as stream:
        encoded = stream.read()
    if hashlib.sha256(encoded).hexdigest() != PLAN_SHA256:
        raise ValueError("Frozen source39 plan hash mismatch")
    plan = json.loads(encoded)
    if (
        tuple(plan["source_subject_ids"]) != SOURCE_IDS
        or plan["raw_root"] != str(RAW_ROOT)
        or plan["execution"]["native_root"] != str(OUTPUT_ROOT)
        or plan["upstream"]["root"] != str(UPSTREAM_ROOT)
        or tuple(plan["sample_counts"]) != SAMPLE_COUNTS
        or plan["budgets"] != [3, 5]
        or plan["proxy_block"] != 5
        or plan["query_blocks"] != [6, 7, 8, 9]
        or any(
            plan["authority"][key]
            for key in (
                "held_access",
                "retired_access",
                "full_manifest_or_impedance_access",
                "old_runner_execution",
                "external_requests",
            )
        )
    ):
        raise ValueError("Immutable authority differs")
    return plan


def git(root, *args):
    return subprocess.check_output(["git", "-C", str(root), *args], text=True).strip()


def recovery_record(output, plan):
    """Bind the one declared infrastructure recovery; never read old NPZ bytes."""
    if output == OUTPUT_ROOT:
        return None, 0
    if output != RECOVERY_ROOT:
        raise ValueError("Undeclared native recovery path")
    original = exact_path(OUTPUT_ROOT)
    expected = {"start.json", *(f"S{subject:03d}.npz" for subject in SOURCE_IDS)}
    if {entry.name for entry in original.iterdir()} != expected:
        raise ValueError("Failed native attempt inventory differs; require start plus39NPZ")
    preserved_bytes = 0
    for name in sorted(expected):
        path = exact_path(original / name)
        info = os.stat(path, follow_symlinks=False)
        if (
            not stat.S_ISREG(info.st_mode)
            or info.st_nlink != 1
            or stat.S_IMODE(info.st_mode) != 0o400
        ):
            raise ValueError("Failed native artifacts must remain regular single-link0400")
        preserved_bytes += info.st_size
    with open_regular(original / "start.json") as stream:
        encoded = stream.read()
    if hashlib.sha256(encoded).hexdigest() != FAILED_START_SHA256:
        raise ValueError("Failed native start hash differs")
    start = json.loads(encoded)
    if (
        start["plan_sha256"] != PLAN_SHA256
        or start["source_subject_ids"] != list(SOURCE_IDS)
        or start["status"] != "STARTED"
        or start["study_id"] != plan["study_id"]
        or start["output_root"] != str(OUTPUT_ROOT)
    ):
        raise ValueError("Failed native start authority differs")
    return {
        "original_root": str(OUTPUT_ROOT),
        "start_sha256": FAILED_START_SHA256,
        "preserved_artifact_count": len(SOURCE_IDS),
        "reason": "infrastructure_terminal_source_rehash_allowlist",
        "unchanged_plan_sha256": PLAN_SHA256,
        "actual_native_root": str(RECOVERY_ROOT),
    }, preserved_bytes


def preflight(plan_path, output):
    if exact_path(plan_path) != PLAN_PATH or exact_path(output) not in (OUTPUT_ROOT, RECOVERY_ROOT):
        raise ValueError("Only canonical plan/output paths are permitted")
    if Path(__file__).resolve().parents[1] != SOURCE_ROOT:
        raise RuntimeError("Human export requires the integrated main checkout")
    if output.exists():
        raise FileExistsError("Attempt already exists; no overwrite, resume or silent retry")
    plan = read_plan(plan_path)
    if (
        git(SOURCE_ROOT, "status", "--porcelain")
        or git(SOURCE_ROOT, "branch", "--show-current") != "main"
    ):
        raise RuntimeError("Human export requires clean committed main")
    if (
        exact_path(UPSTREAM_ROOT) != UPSTREAM_ROOT
        or git(UPSTREAM_ROOT, "rev-parse", "HEAD") != plan["upstream"]["revision"]
        or git(UPSTREAM_ROOT, "status", "--porcelain")
    ):
        raise RuntimeError("Pinned upstream identity differs")
    pins = dict(plan["upstream"]["pins"])
    for relative, expected in pins.items():
        if file_sha256(UPSTREAM_ROOT / relative) != expected:
            raise RuntimeError("Upstream numerical hash differs: " + relative)
    if Path(sys.executable) != PYTHON_PATH or sys.version.split()[0] != "3.9.21":
        raise RuntimeError("Use the frozen native Python3.9.21 interpreter")
    for name, version in VERSIONS.items():
        if importlib.metadata.version(name) != version:
            raise RuntimeError("Dependency mismatch: " + name)
    for name in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS"):
        if os.environ.get(name) != "1":
            raise RuntimeError("Set " + name + "=1 before Python startup")
    source_hashes = {}
    for key in ("native_helper", "operator", "q_helper"):
        spec = plan[key]
        actual = file_sha256(SOURCE_ROOT / spec["path"])
        if actual != spec["sha256"]:
            raise RuntimeError("Pinned project helper hash differs: " + key)
        source_hashes[spec["path"]] = actual
    source_hashes["scripts/export_metadata_prior_source.py"] = file_sha256(Path(__file__).resolve())
    source_hashes[str(PLAN_PATH.relative_to(SOURCE_ROOT))] = PLAN_SHA256
    recovery, preserved_bytes = recovery_record(output, plan)
    provenance = {
        "plan_sha256": PLAN_SHA256,
        "source_commit": git(SOURCE_ROOT, "rev-parse", "HEAD"),
        "source_tree": git(SOURCE_ROOT, "rev-parse", "HEAD^{tree}"),
        "source_hashes": source_hashes,
        "helper_sha256": plan["native_helper"]["sha256"],
        "upstream_revision": plan["upstream"]["revision"],
        "upstream_pins": pins,
        "python": sys.version.split()[0],
        "dependencies": dict(VERSIONS),
        "workers": 1,
        "blas_threads": 1,
    }
    if recovery is not None:
        provenance.update(recovery=recovery, preserved_native_bytes=preserved_bytes)
    return plan, provenance


def import_helper(plan):
    path = SOURCE_ROOT / plan["native_helper"]["path"]
    if file_sha256(path) != plan["native_helper"]["sha256"]:
        raise ValueError("Native helper changed after preflight")
    spec = importlib.util.spec_from_file_location("metadata_prior_native_helpers", path)
    helper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helper)
    return helper


def finite_array(value, shape):
    if (
        not isinstance(value, np.ndarray)
        or value.shape != tuple(shape)
        or value.dtype.kind not in "fiu"
        or not np.isfinite(value).all()
    ):
        raise ValueError("Expected exact finite real array geometry")


def load_raw(subject, plan):
    if type(subject) is not int or subject not in SOURCE_IDS:
        raise ValueError("Raw participant outside source39")
    if tuple(plan["source_subject_ids"]) != SOURCE_IDS or plan["raw_root"] != str(RAW_ROOT):
        raise ValueError("Source-only raw authority differs")
    path = exact_path(RAW_ROOT) / f"S{subject:03d}.mat"
    with open_regular(path) as stream:
        digest = stream_sha256(stream)
        raw_bytes = os.fstat(stream.fileno()).st_size
        stream.seek(0)
        payload = io.loadmat(
            stream,
            variable_names=["data"],
            squeeze_me=False,
            mat_dtype=False,
            verify_compressed_data_integrity=True,
        )
        raw = payload["data"]
        finite_array(raw, (8, 710, 2, 10, 12))
        stored_dtype = str(raw.dtype)
        stream.seek(0)
        if stream_sha256(stream) != digest:
            raise ValueError("Raw bytes changed during loading")
    return np.asarray(raw, dtype=np.float64), {
        "subject": subject,
        "path": str(path),
        "sha256": digest,
        "bytes": raw_bytes,
        "stored_dtype": stored_dtype,
        "shape": list(raw.shape),
    }


def compute_participant(raw, plan, helper, native):
    """Pure native pipeline, all four windows; no file access or metadata."""
    finite_array(raw, (8, 710, 2, 10, 12))
    result = {}
    for n in plan["sample_counts"]:
        x = np.empty((2, 10, 12, 5, 8, n), dtype=np.float64)
        full = np.empty((2, 2, 4, 12, 5, 12), dtype=np.float64)
        a0 = np.empty((2, 4, 12, 5, 12), dtype=np.float64)
        references = helper.references_for_window(n, plan, native.gen_ref_sin)
        for interface_index, interface in enumerate(plan["interfaces"]):
            for block in range(10):
                for label in range(12):
                    x[interface_index, block, label] = helper.preprocess_trial(
                        raw[:, :, interface_index, block, label], n, native
                    )
            wa = plan["native_weights"]["A0_author"][interface]
            we = plan["native_weights"]["ETRCA"][interface]
            for weights, method in ((wa, "cca"), (we, "trca")):
                if not np.array_equal(
                    weights, native.suggested_weights_filterbank(5, interface, method)
                ):
                    raise ValueError("Native method/interface weights differ")
            query = list(x[interface_index, plan["query_blocks"]].reshape(48, 5, 8, n))
            anchor = native.SCCA(
                n_component=1,
                n_jobs=None,
                weights_filterbank=wa,
                force_output_UV=False,
                update_UV=True,
            )
            anchor.fit(ref_sig=references)
            a0[interface_index] = helper.predict_correlations(anchor, query, wa).reshape(
                4, 12, 5, 12
            )
            if anchor.model["U"] is not None or anchor.model["V"] is not None:
                raise ValueError("Native anchor cached query-specific filters")
            for budget_index, k in enumerate(plan["budgets"]):
                model = native.ETRCA(n_jobs=None, weights_filterbank=we)
                model.fit(
                    X=list(x[interface_index, :k].reshape(12 * k, 5, 8, n)), Y=list(range(12)) * k
                )
                full[interface_index, budget_index] = helper.predict_correlations(
                    model, query, we
                ).reshape(4, 12, 5, 12)
        result.update({f"x_{n}": x, f"full_{n}": full, f"a0_{n}": a0})
    validate_cache(result, plan)
    return result


def validate_cache(cache, plan):
    expected = {
        f"{kind}_{n}": tuple(n if dim == "N" else dim for dim in shape)
        for n in plan["sample_counts"]
        for key, shape in plan["native_cache"]["keys_per_N"].items()
        for kind in [key.removesuffix("_N")]
    }
    if set(cache) != set(expected):
        raise ValueError("Native cache keys differ")
    for key, shape in expected.items():
        finite_array(cache[key], shape)
        if cache[key].dtype != np.float64:
            raise ValueError("Native cache requires float64")
        if not key.startswith("x_") and np.max(np.abs(cache[key])) > 1 + 1e-10:
            raise ValueError("Native correlation outside numerical domain")


def fsync_directory(path):
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def publish(path, payload, quota, *, npz=False):
    """Create immutable bytes exclusively; any failed partial file is preserved."""
    path = exact_path(path)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o400)
    with os.fdopen(descriptor, "wb") as stream:
        if npz:
            np.savez(stream, **payload)
        else:
            stream.write(
                (json.dumps(payload, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()
            )
        stream.flush()
        os.fsync(stream.fileno())
        info = os.fstat(stream.fileno())
        if info.st_size > quota or info.st_nlink != 1 or stat.S_IMODE(info.st_mode) != 0o400:
            raise ValueError("Publication byte/mode/link quota failed")
    fsync_directory(path.parent)
    return {"filename": path.name, "sha256": file_sha256(path), "bytes": info.st_size}


def runtime_guard(output, *, library_roots=()):
    """Return an audit-hook closure; only exact raw files and outputs are data."""
    raw_paths = {RAW_ROOT / f"S{subject:03d}.mat" for subject in SOURCE_IDS}
    output_paths = {output / name for name in ("start.json", "result.json")}
    output_paths.update(output / f"S{subject:03d}.npz" for subject in SOURCE_IDS)
    # These exact immutable helpers are hashed before and after export. They
    # are never imported/executed by the native exporter; no src tree wildcard.
    pinned_source_paths = {
        SOURCE_ROOT / "src/cfeg/analysis/metadata_trca_prior.py",
        SOURCE_ROOT / "src/cfeg/analysis/metadata_prior_validation.py",
    }
    roots = tuple(Path(path).resolve() for path in library_roots)

    def guard(event, args):
        if event in ("socket.connect", "socket.getaddrinfo", "subprocess.Popen"):
            raise RuntimeError("Network/subprocess access forbidden during source export")
        if event == "import" and str(args[0]).startswith(
            ("cfeg", "SSVEPAnalysisToolbox.datasets", "SSVEPAnalysisToolbox.evaluator")
        ):
            raise RuntimeError("Closed project/dataset/evaluator imports forbidden")
        if event != "open" or not isinstance(args[0], (str, bytes, os.PathLike)):
            return
        path = Path(os.fsdecode(args[0]))
        flags = args[2] if len(args) > 2 and isinstance(args[2], int) else 0
        writing = bool(flags & (os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC))
        if writing and path not in output_paths:
            raise RuntimeError("Only new native artifacts may be written")
        if (
            path in raw_paths
            or path in output_paths
            or path in pinned_source_paths
            or path == output
            or path == PLAN_PATH
        ):
            exact_path(path)
            return
        # Shared data containers never become authorized by a library root.
        if path.suffix.lower() in (
            ".mat",
            ".npz",
            ".npy",
            ".h5",
            ".hdf5",
            ".parquet",
            ".json",
            ".jsonl",
            ".sqlite",
            ".db",
            ".pkl",
        ):
            raise RuntimeError("Source-only export forbids metadata/old/held containers")
        if not writing and path.is_absolute() and ".." not in path.parts:
            resolved = path.resolve()
            if any(resolved == root or root in resolved.parents for root in roots):
                return
        raise RuntimeError("Unlisted runtime path: " + str(path))

    return guard


def run(plan_path=PLAN_PATH, output=OUTPUT_ROOT):
    plan_path, output = Path(plan_path), Path(output)
    plan, provenance = preflight(plan_path, output)
    began = time.monotonic()
    limit = plan["execution"]["max_seconds_per_stage"]
    budget = plan["execution"]["budget_bytes"] - provenance.get("preserved_native_bytes", 0)
    if budget <= 0:
        raise ValueError("Preserved failed artifacts consume the total8GiB budget")
    exact_path(output.parent)
    output.parent.mkdir(mode=0o700, exist_ok=True)
    output.mkdir(mode=0o700)
    fsync_directory(output.parent)
    start = dict(
        provenance,
        schema="cfeg.metadata-prior-source.native-start.v1",
        study_id=plan["study_id"],
        status="STARTED",
        created_at=utc_now(),
        source_subject_ids=list(SOURCE_IDS),
        sample_counts=list(SAMPLE_COUNTS),
        raw_root=str(RAW_ROOT),
        output_root=str(output),
        max_seconds=limit,
        budget_bytes=budget,
        metadata_access=False,
        held_access=False,
    )
    start_receipt = publish(output / "start.json", start, budget)
    used = start_receipt["bytes"]
    sys.dont_write_bytecode = True
    helper = import_helper(plan)
    imported_native_hashes = helper.import_native(UPSTREAM_ROOT)
    native = helper._NATIVE
    # Warm the narrow preprocessing/scoring imports before installing the hook;
    # importing these modules does not execute a dataset constructor or runner.
    roots = (
        SOURCE_ROOT / "scripts",
        UPSTREAM_ROOT,
        Path(sys.prefix),
        Path(sys.base_prefix),
        Path("/usr/lib"),
        Path("/usr/share"),
        Path("/etc"),
    )
    sys.addaudithook(runtime_guard(output, library_roots=roots))
    previous_alarm = signal.getsignal(signal.SIGALRM)

    def timeout_handler(_signum, _frame):
        raise TimeoutError("Frozen source export stage runtime ceiling exceeded")

    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(max(1, int(limit - (time.monotonic() - began))))
    files = []
    try:
        for subject in SOURCE_IDS:
            if time.monotonic() - began >= limit:
                raise TimeoutError("Frozen source export runtime ceiling exceeded")
            raw, raw_receipt = load_raw(subject, plan)
            cache = compute_participant(raw, plan, helper, native)
            validate_cache(cache, plan)
            # The uncompressed archive adds only bounded ZIP/NPY headers, but
            # reject inadequate remaining space before streaming waveform bytes.
            if sum(value.nbytes for value in cache.values()) + 65536 > budget - used:
                raise ValueError("Insufficient remaining native publication byte budget")
            info = publish(output / f"S{subject:03d}.npz", cache, budget - used, npz=True)
            used += info["bytes"]
            files.append(dict(info, subject=subject, raw_receipt=raw_receipt))
            del raw, cache
            print(
                json.dumps(
                    {
                        "subject": subject,
                        "completed": len(files),
                        "elapsed_seconds": round(time.monotonic() - began, 3),
                    }
                ),
                flush=True,
            )
        for entry in files:
            if file_sha256(output / entry["filename"]) != entry["sha256"]:
                raise ValueError("Published native bytes changed before completion")
        if file_sha256(output / "start.json") != start_receipt["sha256"]:
            raise ValueError("Start receipt changed before completion")
        for relative, expected in imported_native_hashes.items():
            if file_sha256(UPSTREAM_ROOT / relative) != expected:
                raise ValueError("Imported native source changed during export")
        for relative, expected in provenance["source_hashes"].items():
            if file_sha256(SOURCE_ROOT / relative) != expected:
                raise ValueError("Pinned project source changed during export")
        result = dict(
            provenance,
            schema="cfeg.metadata-prior-source.native-result.v1",
            study_id=plan["study_id"],
            status="COMPLETE",
            created_at=utc_now(),
            start_sha256=start_receipt["sha256"],
            imported_native_hashes=imported_native_hashes,
            source_subject_ids=list(SOURCE_IDS),
            sample_counts=list(SAMPLE_COUNTS),
            files=files,
            total_bytes=sum(entry["bytes"] for entry in files),
            elapsed_seconds=time.monotonic() - began,
        )
        publish(output / "result.json", result, budget - used)
        return result
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, previous_alarm)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, default=PLAN_PATH)
    parser.add_argument("--output", type=Path, default=OUTPUT_ROOT)
    args = parser.parse_args()
    run(args.plan, args.output)


if __name__ == "__main__":
    main()
