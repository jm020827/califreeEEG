"""Single fixed support-only geometry diagnostic; no classification execution."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import platform
import signal
import stat
import subprocess
import sys
import time
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import scipy

from cfeg.analysis import metadata_trca_prior, native_support_prefix, trca_support_geometry
from cfeg.analysis.native_support_prefix import fd_sha256, regular_fd, require, verified_archive
from cfeg.analysis.trca_support_geometry import matrix_geometry, summarize, support_geometry

ROOT = Path("/home/whwovy/califreeEEG")
PLAN = ROOT / "configs/analysis/trca_support_geometry_v1.json"
PLAN_SHA = "e9bbf0d132b25355e2ae189b081b5ed8c40f366435b3c079566b3faec7119b86"
SOURCE_PATHS = (
    "scripts/run_trca_support_geometry.py",
    "src/cfeg/analysis/native_support_prefix.py",
    "src/cfeg/analysis/trca_support_geometry.py",
    "src/cfeg/analysis/metadata_trca_prior.py",
    "src/cfeg/analysis/__init__.py",
    "configs/analysis/trca_support_geometry_v1.json",
)


def utcnow():
    return datetime.now(timezone.utc).isoformat()


def sha(path):
    fd = regular_fd(path)
    try:
        return fd_sha256(fd, os.fstat(fd).st_size)
    finally:
        os.close(fd)


def read_json(path, expected_sha):
    fd = regular_fd(path)
    try:
        with os.fdopen(os.dup(fd), "rb") as stream:
            raw = stream.read()
        require(hashlib.sha256(raw).hexdigest() == expected_sha, "Pinned JSON SHA mismatch")
        return json.loads(raw)
    finally:
        os.close(fd)


def publish(path, content, remaining):
    raw = (
        (
            json.dumps(content, ensure_ascii=False, allow_nan=False, separators=(",", ":")) + "\n"
        ).encode()
        if isinstance(content, dict)
        else content
    )
    require(len(raw) <= remaining, "Incremental publication byte budget exceeded")
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o400)
    try:
        with os.fdopen(fd, "wb", closefd=False) as stream:
            stream.write(raw)
            stream.flush()
            os.fsync(stream.fileno())
    finally:
        os.close(fd)
    sync_directory(Path(path).parent)
    return {
        "filename": Path(path).name,
        "sha256": hashlib.sha256(raw).hexdigest(),
        "bytes": len(raw),
    }


def sync_directory(path):
    fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def guard_for(reads, output):
    reads = {str(Path(p).absolute()) for p in reads}
    reads.add(str(output))
    writes = {
        str(output / name) for name in ("start.json", "geometry.npz", "result.json", "failure.json")
    }

    def guard(event, args):
        if event == "open":
            path, _mode, flags = args
            if isinstance(path, int):
                return
            name = os.path.abspath(os.fsdecode(path))
            writing = bool(
                flags & (os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC | os.O_APPEND)
            )
            require(
                name in writes if writing else name in reads | writes,
                f"Out-of-scope {'write' if writing else 'read'} denied: {name}",
            )
        elif event.startswith(("socket.", "subprocess.")) or event in (
            "os.system",
            "os.remove",
            "os.rename",
            "os.rmdir",
            "os.mkdir",
            "os.link",
            "os.symlink",
        ):
            raise ValueError(f"Out-of-scope runtime action: {event}")

    return guard


def compute(plan, output, provenance, *, install_guard=True):
    """Reusable lifecycle for artificial tests; production pins checked by main."""
    started = time.monotonic()
    output = Path(output)
    require(
        not output.exists() and output.parent.resolve() == output.parent, "Fresh output required"
    )
    output.mkdir(mode=0o700)
    sync_directory(output.parent)
    budget = plan["resources"]["max_incremental_bytes"]
    start = publish(
        output / "start.json",
        {
            "schema": "cfeg.trca-support-geometry.start.v1",
            "study_id": plan["study_id"],
            "started_at": utcnow(),
            "plan_sha256": provenance["plan_sha256"],
            "source_commit": provenance["source_commit"],
            "source_hashes": provenance["source_hashes"],
            "analysis_type": plan["analysis_type"],
            "native_manifest_sha256": plan["native_manifest_sha256"],
            "decoded_inputs": plan["decoded_inputs"],
            "integrity_allowance": plan["integrity_allowance"],
            "pins": plan["pins"],
            "query_scoring": False,
            "numeric_metadata": False,
        },
        budget,
    )
    budget -= start["bytes"]
    native = Path(plan["native_directory"])
    source_paths = {Path(path) for path in provenance["source_hashes"]}
    archives = {native / f"S{p:03d}.npz" for p in plan["source_subject_ids"]}
    if install_guard:
        sys.addaudithook(guard_for(source_paths | archives | {native / "result.json"}, output))
    try:
        manifest = read_json(native / "result.json", plan["native_manifest_sha256"])
        require(
            manifest["schema"] == "cfeg.metadata-prior-source.native-result.v1"
            and manifest["status"] == "COMPLETE"
            and manifest["plan_sha256"] == plan["native_plan_sha256"]
            and manifest["source_subject_ids"] == plan["source_subject_ids"]
            and manifest["sample_counts"] == plan["sample_counts"],
            "Native manifest contract",
        )
        entries = manifest["files"]
        require(
            [e["subject"] for e in entries] == plan["source_subject_ids"],
            "Exact native participant order",
        )
        require(
            all(e["filename"] == f"S{e['subject']:03d}.npz" for e in entries),
            "Exact native filenames",
        )
        all_rows, all_matrices, all_full, all_iso = [], [], [], []
        receipts = []
        for entry in entries:
            path = native / entry["filename"]
            info = path.stat(follow_symlinks=False)
            require(stat.S_IMODE(info.st_mode) == 0o400, "Immutable native artifact mode required")
            with verified_archive(
                path, entry["sha256"], entry["bytes"], plan["sample_counts"]
            ) as load:
                for n in plan["sample_counts"]:
                    x = load(n)
                    for ei, interface in enumerate(plan["interfaces"]):
                        for k in plan["budgets"]:
                            rows, matrices, full, iso = support_geometry(
                                x[ei, :k],
                                subject=entry["subject"],
                                interface=interface,
                                samples=n,
                                k=k,
                            )
                            all_rows.extend(rows)
                            all_matrices.append(matrices)
                            all_full.append(full)
                            all_iso.append(iso)
                    del x
            receipts.append({key: entry[key] for key in ("subject", "filename", "sha256", "bytes")})
            require(
                time.monotonic() - started <= plan["resources"]["max_seconds"],
                "Time budget exceeded",
            )
            print(
                json.dumps(
                    {
                        "completed_participants": len(receipts),
                        "total": len(entries),
                        "geometry_records": len(all_rows),
                    }
                ),
                flush=True,
            )
        require(len(all_rows) == plan["records_expected"], "Record count mismatch")
        summaries, bands = summarize(all_rows, plan)
        matrices = np.concatenate(all_matrices)
        buffer = io.BytesIO()
        np.savez_compressed(
            buffer,
            S=matrices[:, 0],
            C=matrices[:, 1],
            full_unit=np.concatenate(all_full),
            iso_unit=np.concatenate(all_iso),
        )
        geometry = publish(output / "geometry.npz", buffer.getvalue(), budget)
        budget -= geometry["bytes"]
        require(sha(native / "result.json") == plan["native_manifest_sha256"], "Manifest changed")
        for path, expected in provenance["source_hashes"].items():
            require(sha(path) == expected, "Diagnostic source changed")
        result = {
            "schema": "cfeg.trca-support-geometry.result.v1",
            "study_id": plan["study_id"],
            "status": "GEOMETRY_DIAGNOSED",
            "analysis_type": plan["analysis_type"],
            "completed_at": utcnow(),
            "elapsed_seconds": time.monotonic() - started,
            "plan_sha256": provenance["plan_sha256"],
            "source_commit": provenance["source_commit"],
            "source_hashes": provenance["source_hashes"],
            "start": start,
            "geometry": geometry,
            "native_manifest_sha256": plan["native_manifest_sha256"],
            "archives": receipts,
            "support_blocks_decoded": [0, 1, 2, 3, 4],
            "whole_archive_hash_before_after": True,
            "query_values_decoded": False,
            "proxy_values_decoded": False,
            "numeric_metadata_read": False,
            "classification_executed": False,
            "record_count": len(all_rows),
            "rows": all_rows,
            "band_groups": bands,
            "summaries": summaries,
            "limitations": [
                "Post-outcome, repeatedly exposed development cohort",
                "No causal attribution of prior accuracy loss",
                "No alternate gamma or metadata efficacy tested",
                "Repeated class/band records are not independent participants",
            ],
        }
        receipt = publish(output / "result.json", result, budget)
        print(
            json.dumps(
                {
                    "status": result["status"],
                    "result": receipt,
                    "elapsed_seconds": result["elapsed_seconds"],
                }
            ),
            flush=True,
        )
        return result
    except BaseException as error:
        publish(
            output / "failure.json",
            {
                "status": "VALIDITY_FAILURE",
                "at": utcnow(),
                "exception": type(error).__name__,
                "message": str(error),
                "start": start,
            },
            budget,
        )
        raise


def warm_artificial_runtime():
    matrix_geometry(np.diag([2.0, 1.0]), np.eye(2))
    buffer = io.BytesIO()
    np.savez_compressed(buffer, artificial=np.ones(1))
    buffer.seek(0)
    with zipfile.ZipFile(buffer) as archive:
        require(archive.namelist() == ["artificial.npy"], "Artificial ZIP warmup")


def verify_origins():
    require(Path(__file__).resolve() == ROOT / SOURCE_PATHS[0], "Exact runner origin required")
    for module in (native_support_prefix, trca_support_geometry, metadata_trca_prior):
        require(
            Path(module.__file__).resolve()
            == ROOT / "src/cfeg/analysis" / f"{module.__name__.split('.')[-1]}.py",
            "Exact imported analysis module origin required",
        )


def main():
    argparse.ArgumentParser(description=__doc__).parse_args()
    plan = read_json(PLAN, PLAN_SHA)
    require(Path.cwd() == ROOT, "Run from fixed main repository")
    verify_origins()
    require(
        platform.python_version() == plan["pins"]["python"]
        and np.__version__ == plan["pins"]["numpy"]
        and scipy.__version__ == plan["pins"]["scipy"],
        "Pinned existing project environment required",
    )
    require(
        all(
            os.environ.get(key) == "1"
            for key in (
                "PYTHONDONTWRITEBYTECODE",
                "OPENBLAS_NUM_THREADS",
                "OMP_NUM_THREADS",
                "MKL_NUM_THREADS",
            )
        ),
        "One-thread/no-bytecode environment required",
    )
    run_git = lambda *args: subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()
    require(
        run_git("branch", "--show-current") == "main" and not run_git("status", "--porcelain"),
        "Clean main commit required before diagnostic",
    )
    provenance = {
        "plan_sha256": PLAN_SHA,
        "source_commit": run_git("rev-parse", "HEAD"),
        "source_hashes": {str(ROOT / p): sha(ROOT / p) for p in SOURCE_PATHS},
    }
    require(
        provenance["source_hashes"][str(ROOT / "src/cfeg/analysis/metadata_trca_prior.py")]
        == plan["operator_sha256"],
        "Closed original operator must be unchanged",
    )
    # Warm only artificial linear algebra/compression before the strict runtime read allowlist.
    warm_artificial_runtime()
    signal.signal(signal.SIGALRM, lambda *_: (_ for _ in ()).throw(TimeoutError("600s limit")))
    signal.alarm(plan["resources"]["max_seconds"])
    compute(plan, Path(plan["output_directory"]), provenance)
    signal.alarm(0)


if __name__ == "__main__":
    main()
