"""Explicit envelope-only execution amendment; original scientific code is pinned."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import importlib.metadata
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path("/home/whwovy/califreeEEG")
EXECUTION_PATH = ROOT / "configs/analysis/native_subset_m_source39_envelope_r1.json"
EXECUTION_SHA256 = "93108bd8d0cce8b0572e6cbc200208c3990be725ffdbdf2fdcc0bc8daec82115"
OUTPUT_ROOT = Path("/home/whwovy/native-subset-m-artifacts/source39-v1-envelope-r1")
ARTIFACTS = (
    "start.json",
    "source-projection.json",
    "features.npz",
    "fold-freezes.json",
    "result.json",
)
SCIENCE_SHA256 = "ee9b758f755a18f7540c978e9faf18a5711e652948b94df595377bbfb83cb80a"


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_execution(path):
    encoded = Path(path).read_bytes()
    if hashlib.sha256(encoded).hexdigest() != EXECUTION_SHA256:
        raise ValueError("Execution amendment hash mismatch")
    value = json.loads(encoded)
    if (
        value["scientific_plan"]["sha256"] != SCIENCE_SHA256
        or value["study_id"] != value["randomization_namespace"]
        or value["study_id"] != "native-subset-m-source39-v1"
        or value["scientific_changes"]
    ):
        raise ValueError("Scientific identity must remain unchanged")
    return value


def load_helpers(amendment):
    for relative, expected in amendment["source_helpers"].items():
        if sha(ROOT / relative) != expected:
            raise ValueError("Pinned helper changed: " + relative)
    producer = importlib.import_module("run_native_subset_m_source")
    core = importlib.import_module("native_subset_m_core")
    for module in (producer, core):
        if Path(module.__file__).resolve() != ROOT / "scripts" / (module.__name__ + ".py"):
            raise ValueError("Unexpected imported helper location")
    return producer, core


def content_projection(envelope, amendment):
    content_keys = set(amendment["content_keys"])
    provenance = amendment["envelope_provenance"]
    if set(envelope) != content_keys | set(provenance):
        raise ValueError("Expected exact original eleven-field envelope")
    for key, expected in provenance.items():
        if type(envelope[key]) is not str or envelope[key] != expected:
            raise ValueError("Original projection provenance mismatch: " + key)
    return {key: envelope[key] for key in amendment["content_keys"]}


def preflight(execution_path):
    if (
        Path(execution_path) != EXECUTION_PATH
        or Path(execution_path).resolve() != EXECUTION_PATH
        or Path(__file__).resolve().parents[1] != ROOT
    ):
        raise ValueError("Execution requires canonical integrated main paths")
    amendment = load_execution(execution_path)
    if OUTPUT_ROOT != Path(amendment["output_root"]) or OUTPUT_ROOT.resolve() != OUTPUT_ROOT:
        raise ValueError("Execution output destination drift")
    if OUTPUT_ROOT.exists() or OUTPUT_ROOT.is_symlink():
        raise FileExistsError("Attempt already exists: no overwrite, resume, or retry")
    producer, core = load_helpers(amendment)
    plan = producer.read_plan(amendment["scientific_plan"]["path"])
    if plan["study_id"] != amendment["study_id"]:
        raise ValueError("SHAM randomization namespace changed")
    if (
        producer.git(ROOT, "status", "--porcelain")
        or producer.git(ROOT, "branch", "--show-current") != "main"
    ):
        raise RuntimeError("Integrated main must be clean and committed")
    upstream = producer.UPSTREAM_ROOT
    if producer.git(upstream, "rev-parse", "HEAD") != producer.REVISION or producer.git(
        upstream, "status", "--porcelain"
    ):
        raise RuntimeError("Native upstream revision/cleanliness mismatch")
    for relative, expected in plan["upstream"]["pins"].items():
        if sha(upstream / relative) != expected:
            raise RuntimeError("Native source pin mismatch")
    if Path(sys.executable) != producer.PYTHON_PATH or sys.version.split()[0] != "3.9.21":
        raise RuntimeError("Require frozen external Python3.9.21")
    for name, expected in producer.VERSIONS.items():
        if importlib.metadata.version(name) != expected:
            raise RuntimeError("Frozen dependency differs: " + name)
    for name in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS"):
        if os.environ.get(name) != "1":
            raise RuntimeError("Require " + name + "=1")
    provenance = {
        "plan_sha256": SCIENCE_SHA256,
        "execution_plan_sha256": EXECUTION_SHA256,
        "attempt_id": amendment["attempt_id"],
        "source_commit": producer.git(ROOT, "rev-parse", "HEAD"),
        "source_tree": producer.git(ROOT, "rev-parse", "HEAD^{tree}"),
        "upstream_revision": producer.REVISION,
        "python": sys.version.split()[0],
        "dependencies": dict(producer.VERSIONS),
    }
    return amendment, plan, provenance, producer, core


def guard_for(amendment, plan):
    allowed_raw = {Path(plan["raw_root"]) / f"S{s:03d}.mat" for s in plan["source_subject_ids"]}
    destination = Path(amendment["output_root"])
    writable = {destination / name for name in ARTIFACTS}
    json_reads = writable | {Path(plan["source_projection"]["path"])}
    npz_reads = {destination / "features.npz", Path(plan["baseline_reference"]["path"])}

    def guard(event, args):
        if event == "import" and str(args[0]).startswith(
            ("cfeg", "SSVEPAnalysisToolbox.datasets", "SSVEPAnalysisToolbox.evaluator")
        ):
            raise RuntimeError("Forbidden dataset/evaluator import")
        if event in ("socket.connect", "socket.getaddrinfo", "subprocess.Popen"):
            raise RuntimeError("Network/subprocess forbidden after preflight")
        if event != "open" or not isinstance(args[0], (str, bytes, os.PathLike)):
            return
        path = Path(os.fsdecode(args[0]))
        suffix = path.suffix.lower()
        if suffix == ".mat" and path not in allowed_raw:
            raise RuntimeError("Only exact source39 raw paths allowed")
        if suffix in (".h5", ".hdf5", ".npy", ".parquet", ".pkl"):
            raise RuntimeError("Processed/shared data containers forbidden")
        if suffix == ".npz" and path not in npz_reads:
            raise RuntimeError("Unapproved cache")
        if suffix == ".json" and path not in json_reads:
            raise RuntimeError("Unapproved JSON/manifest/attempt")
        flags = args[2] if len(args) > 2 and isinstance(args[2], int) else 0
        if flags & (os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC) and path not in writable:
            raise RuntimeError("Only five new artifact files may be written")

    return guard


def execute(execution_path):
    clock_start = time.monotonic()
    amendment, plan, provenance, producer, core = preflight(execution_path)
    sys.dont_write_bytecode = True
    imported = producer.import_native(producer.UPSTREAM_ROOT)
    # Import-only library initialization precedes the data/write guard.
    sys.addaudithook(guard_for(amendment, plan))
    common = {
        key: provenance[key]
        for key in (
            "plan_sha256",
            "execution_plan_sha256",
            "attempt_id",
            "source_commit",
            "source_tree",
            "upstream_revision",
        )
    }
    common.update(study_id=plan["study_id"], started_at=producer.utc_now())
    destination = Path(amendment["output_root"])
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.mkdir(mode=0o700)
    quota = amendment["execution"]["resource_budget_bytes"]
    start = {
        "schema": "cfeg.native-subset-m.start.v1",
        **provenance,
        **common,
        "source_subject_ids": plan["source_subject_ids"],
        "raw_root": plan["raw_root"],
        "output_root": str(destination),
        "imported_source_hashes": imported,
        "core_sha256": amendment["source_helpers"]["scripts/native_subset_m_core.py"],
        "source_helper_hashes": amendment["source_helpers"],
        "metadata_access": True,
        "source_projection_reuse": True,
        "baseline_cache_read": True,
        "held_access": False,
        "retired_access": False,
        "manifest_or_full_impedance_access": False,
    }
    producer.publish(destination / "start.json", start, quota=quota)
    original = producer.load_projection(plan)
    content = content_projection(original, amendment)
    _, order = core.projection_arrays(content, plan)
    projection_hash = producer.publish(
        destination / "source-projection.json",
        {
            "schema": "cfeg.native-subset-m.projection.v1",
            "input_path": plan["source_projection"]["path"],
            "input_sha256": plan["source_projection"]["sha256"],
            "projection": original,
        },
        quota=quota,
    )
    print(
        "Original eleven-field envelope verified and preserved; starting source39 EEG extraction",
        flush=True,
    )
    cache, raw_files = producer.gather_participants(plan, order)
    producer.validate_cache(cache, plan)
    agreement = producer.check_baseline(cache, plan)
    features_hash = producer.publish(
        destination / "features.npz", cache, compressed=True, quota=quota
    )
    freeze = core.fit_all(cache, content, plan)
    if freeze["plan_sha256"] != SCIENCE_SHA256 or len(freeze["folds"]) != 3:
        raise ValueError("Scientific freeze drift")
    freeze.update(common, source_projection_sha256=projection_hash, features_sha256=features_hash)
    freeze_hash = producer.publish(destination / "fold-freezes.json", freeze, quota=quota)
    evaluated = core.evaluate_all(cache, content, freeze, plan)
    if time.monotonic() - clock_start > amendment["execution"]["expected_runtime_ceiling_seconds"]:
        raise TimeoutError("Execution runtime budget exceeded")
    result = {
        **evaluated,
        **common,
        "schema": plan["reporting"]["schema"],
        "status": "DEVELOPMENT_ASSESSMENT_COMPLETE",
        "completed_at": producer.utc_now(),
        "source_projection_sha256": projection_hash,
        "features_sha256": features_hash,
        "fold_freezes_sha256": freeze_hash,
        "raw_files": raw_files,
        "baseline_agreement": agreement,
    }
    producer.publish(destination / "result.json", result, quota=quota)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execution-plan", type=Path, default=EXECUTION_PATH)
    args = parser.parse_args()
    try:
        result = execute(args.execution_plan)
    except Exception as error:
        print(
            json.dumps(
                {
                    "status": "infrastructure_or_data_inconclusive",
                    "error": str(error),
                    "retry_allowed": False,
                    "attempt": "envelope-r1",
                }
            ),
            file=sys.stderr,
        )
        raise
    print(
        json.dumps(
            {
                "status": result["status"],
                "rows": len(result["rows"]),
                "attempt_id": result["attempt_id"],
            }
        )
    )


if __name__ == "__main__":
    main()
