"""One fixed generated full-shape CPU resource screen; never a human reader.

The registered seed/run belongs to the root execution stage. This script keeps
one packed source grid resident during the real unchanged four-head N1 fit.
Only compact JSON evidence is persisted, not generated EEG or packed arrays.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import resource
import signal
import subprocess
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import scipy
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from cfeg.analysis import task_trca_n1_evaluation as evaluation
from cfeg.analysis import task_trca_n1_learning as learning
from cfeg.analysis import task_trca_shape_archive as byte_helpers

STUDY_ID = "task-trca-n1-source39-v1"
SCHEMA = "cfeg.task_trca_n1_source39.resource.v1"
DESIGN_PATH = "configs/analysis/task_trca_n1_source39_v1.json"
DESIGN_SHA = "bb90cc361a28c3e0a53861564ec007f38602cf5d16554e0923a6550eb1be7c95"
NATIVE_PATH = "configs/analysis/metadata_prior_source39_v1.json"
NATIVE_SHA = "ed67cc1f361ed89c37b6f7c5df1e9170b78484e428a5105020485c8c03e3934b"
SEED = 20260915
SAMPLES = (125, 188, 250, 500)
INTERFACES = (0, 1)
BUDGETS = (3, 5)
FIT_IDS = tuple(pid for rank, pid in enumerate(byte_helpers.SOURCE_IDS) if rank % 3 != 0)
CASE_COUNT = 416
UPDATES = 800
SECONDS_MAX = 1200
OUTPUT_BYTES_MAX = 128 * 1024**2
RSS_BYTES_MAX = 4 * 1024**3
HUMAN_SECONDS_MAX = 21600
HUMAN_RSS_BYTES_MAX = 16 * 1024**3
OUTPUT_PARENT_ROOT = Path("/home/whwovy")
CODE_PATHS = (
    "configs/analysis/task_trca_n1_integration_v1.json",
    "configs/analysis/metadata_prior_source39_v1.json",
    "src/cfeg/__init__.py",
    "src/cfeg/metrics.py",
    "src/cfeg/analysis/__init__.py",
    "src/cfeg/analysis/ood_coverage.py",
    "src/cfeg/analysis/primary_aggregate.py",
    "src/cfeg/analysis/primary_inference.py",
    "src/cfeg/analysis/provenance.py",
    "src/cfeg/analysis/metadata_prior_validation.py",
    "src/cfeg/analysis/metadata_prior_source.py",
    "src/cfeg/analysis/metadata_trca_prior.py",
    "src/cfeg/analysis/native_support_prefix.py",
    "src/cfeg/analysis/task_trca_shape_inputs.py",
    "src/cfeg/analysis/task_trca_shape_archive.py",
    "src/cfeg/analysis/task_trca_shape_features.py",
    "src/cfeg/analysis/task_trca_shape_operator.py",
    "src/cfeg/analysis/task_trca_shape_signfree.py",
    "src/cfeg/analysis/task_trca_shape_audit.py",
    "src/cfeg/analysis/numerical_stability_operator.py",
    "src/cfeg/analysis/task_trca_n1_signfree.py",
    "src/cfeg/analysis/task_trca_n1_learning.py",
    "src/cfeg/analysis/task_trca_n1_batch.py",
    "src/cfeg/analysis/task_trca_n1_evaluation.py",
    "src/cfeg/analysis/task_trca_n1_audit.py",
    "configs/analysis/task_trca_n1_source39_v1.json",
    "configs/analysis/task_trca_temporal_v1_design.json",
    "docs/task_trca_n1_source39_v1_build.md",
    "scripts/prepare_task_trca_n1_source39.py",
    "scripts/run_task_trca_n1_source39.py",
    "scripts/audit_task_trca_n1_source39.py",
    "scripts/check_task_trca_n1_source39_resources.py",
    "src/cfeg/analysis/task_trca_n1_source39_archive.py",
    "src/cfeg/analysis/task_trca_n1_source39_audit.py",
    "src/cfeg/analysis/task_trca_n1_source39_artifact_audit.py",
    "src/cfeg/analysis/task_trca_temporal_audit.py",
)


def require(value, message):
    if not value:
        raise ValueError(message)


def sha(path):
    fd = byte_helpers.regular_fd(path)
    try:
        return byte_helpers.fd_sha256(fd, os.fstat(fd).st_size)
    finally:
        os.close(fd)


def read_json(path, expected):
    with byte_helpers._PinnedFile(path, expected, maximum_bytes=1024**2) as pinned:
        return json.loads(
            byte_helpers.pread_exact(pinned.fd, pinned.before.st_size, 0),
            object_pairs_hook=byte_helpers._unique_pairs,
        )


def descriptor(path):
    return {"path": str(path), "sha256": sha(path), "bytes": path.stat().st_size}


def write_json(output, name, value):
    require(
        name in ("start.json", "pipeline.json", "resource.json", "failure.json"),
        "Only fixed resource artifacts",
    )
    encoded = (json.dumps(value, indent=2, allow_nan=False) + "\n").encode()
    used = sum(path.stat().st_size for path in output.iterdir() if path.is_file())
    # Keep enough space for a terminal failure even if a pipeline is oversized.
    reserve = 0 if name == "failure.json" else 1024**2
    require(used + len(encoded) + reserve <= OUTPUT_BYTES_MAX, "Resource output budget")
    path = output / name
    require(output.resolve() == output and path.resolve() == path, "Unaliased output required")
    with path.open("xb") as stream:
        stream.write(encoded)
        stream.flush()
        os.fsync(stream.fileno())
    path.chmod(0o400)
    return descriptor(path)


def _check_code(pins):
    require(bool(CODE_PATHS) and set(pins) == set(CODE_PATHS), "Exact frozen code closure")
    for name, expected in pins.items():
        require(sha(ROOT / name) == expected, "Code pin changed: " + name)


def _preflight():
    require(
        all(
            os.environ.get(name) == "1"
            for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")
        )
        and os.environ.get("PYTHONDONTWRITEBYTECODE") == "1",
        "Launch CPU1/no-bytecode before imports",
    )
    require(
        Path(sys.executable) == Path("/home/whwovy/califreeEEG/.venv/bin/python")
        and sys.version_info[:2] == (3, 10)
        and np.__version__ == "1.26.4"
        and scipy.__version__ == "1.15.3"
        and torch.__version__ == "2.2.2+cu121",
        "Pinned root CPU environment required",
    )
    for args in (["git", "diff", "--quiet"], ["git", "diff", "--cached", "--quiet"]):
        subprocess.run(args, cwd=ROOT, check=True)
    require(bool(CODE_PATHS), "Code closure must be frozen before generation")
    pins = {name: sha(ROOT / name) for name in CODE_PATHS}
    design = read_json(ROOT / DESIGN_PATH, DESIGN_SHA)
    require(design["study_id"] == STUDY_ID, "New experiment identity")
    require(design["resource_rehearsal"]["seed"] == SEED, "Frozen resource seed")
    require(design["resource_rehearsal"]["cases"] == CASE_COUNT, "Frozen case count")
    require(design["resource_rehearsal"]["updates"] == UPDATES, "Frozen update count")
    configuration = read_json(ROOT / NATIVE_PATH, NATIVE_SHA)
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    # Constructor loads Adam dependencies without taking any optimizer step.
    unused = torch.optim.Adam([torch.nn.Parameter(torch.zeros(1, dtype=torch.float64))], lr=0.01)
    del unused
    return configuration, revision, pins


def build_cases(
    configuration, *, seed, fit_ids, samples=SAMPLES, interfaces=INTERFACES, budgets=BUDGETS
):
    """Pure generated recipe; explicit arguments enable nonregistered toy tests.

    Production run supplies only the frozen constants. No old fixture, saved
    human matrix or external numeric input is opened here.
    """
    rng = np.random.default_rng(seed)
    cases = []
    for pid in fit_ids:
        order = byte_helpers.SOURCE_IDS.index(pid) % 2
        for interface in interfaces:
            prototype = rng.normal(size=(12, 5, 8, 500))
            blocks = prototype[None] + rng.normal(size=(6, 12, 5, 8, 500))
            blocks += rng.normal(size=(6, 12, 5, 8, 1))
            packet = rng.uniform(0, 40, size=(5, 8))
            packet[0, 1] = 0
            packet[0, 3] = np.nan
            packet[1, 0] = np.nan
            packet[3, 2] = np.nan
            weights = configuration["native_weights"]["ETRCA"][("dry", "wet")[interface]]
            for n in samples:
                for k in budgets:
                    cases.append(
                        learning.make_task_case(
                            pid,
                            interface,
                            order,
                            blocks[:k, ..., :n],
                            packet[:k],
                            configuration["frequencies"],
                            blocks[5, ..., :n],
                            weights=weights,
                            device="cpu",
                        )
                    )
    return tuple(cases)


def projection(preparation_seconds, fit_seconds, peak_rss_bytes):
    require(
        all(
            isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v) and v >= 0
            for v in (preparation_seconds, fit_seconds)
        ),
        "Finite nonnegative timings",
    )
    require(type(peak_rss_bytes) is int and peak_rss_bytes > 0, "Positive peak RSS")
    projected_seconds = 36 * fit_seconds + 3 * preparation_seconds + 600
    projected_rss = 4 * peak_rss_bytes
    return {
        "projected_seconds": projected_seconds,
        "projected_peak_rss_bytes": projected_rss,
        "runtime_ok": projected_seconds <= HUMAN_SECONDS_MAX,
        "memory_ok": peak_rss_bytes <= RSS_BYTES_MAX and projected_rss <= HUMAN_RSS_BYTES_MAX,
    }


def validate_pipeline(pipeline):
    require(isinstance(pipeline, learning.Pipeline), "Actual unchanged N1 Pipeline required")
    require(
        pipeline.fit_ids == FIT_IDS and pipeline.regularization == 0.001,
        "Exact generated fit IDs and lambda",
    )
    record = pipeline.record()
    require(record["schema"] == learning.SCHEMA, "Unchanged algorithm schema")
    heads = {"Q": record["Q"], **record["residuals"]}
    require(set(heads) == {"Q", "Q2", "QM", "SHAM_REFIT"}, "Exactly four actual heads")
    for name, head in heads.items():
        require(head["steps"] == 200 and len(head["trace"]) == 200, "Actual200 steps/head")
        require([row["step"] for row in head["trace"]] == list(range(1, 201)), "Ordered steps")
        values = [head["initial_loss"], head["final_loss"], *head["coefficients"]]
        values += [
            value
            for row in head["trace"]
            for value in (row["loss_before_step"], row["ce_before_step"], row["gradient_norm"])
        ]
        require(all(math.isfinite(value) for value in values), "Finite head record: " + name)
    return record


def _peak_rss_bytes():
    return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) * 1024


def _guard(readable, output):
    allowed = {Path(path).absolute() for path in readable}
    libraries = (Path(sys.executable).parent.parent, Path("/usr/local/lib"), Path("/usr/lib"))
    data_suffixes = {
        ".json",
        ".jsonl",
        ".mat",
        ".npy",
        ".npz",
        ".h5",
        ".hdf5",
        ".pkl",
        ".parquet",
        ".db",
        ".sqlite",
        ".sqlite3",
    }

    def guard(event, args):
        if event in ("subprocess.Popen", "socket.connect", "socket.getaddrinfo"):
            raise PermissionError("No processes/network during resource generation or fit")
        if event != "open" or not isinstance(args[0], (str, bytes, os.PathLike)):
            return
        path = Path(os.fsdecode(args[0])).absolute()
        own = path.parent == output and path.resolve() == path
        flags = args[2] if len(args) > 2 and isinstance(args[2], int) else 0
        if flags & (os.O_CREAT | os.O_WRONLY | os.O_RDWR | os.O_TRUNC) and not own:
            raise PermissionError("Resource writes restricted to new output")
        if (
            path.suffix.lower() in data_suffixes
            and path not in allowed
            and not own
            and not any(path.is_relative_to(root) for root in libraries)
        ):
            raise PermissionError("No external data inputs to resource rehearsal")

    sys.addaudithook(guard)


def _new_output(output):
    output = Path(output).absolute()
    require(
        output.resolve() == output and not output.exists() and output.parent.is_dir(),
        "New unaliased output directory required",
    )
    require(
        output.name == "resource1"
        and output.parent.parent == OUTPUT_PARENT_ROOT
        and output.parent.name.startswith("task-trca-n1-source39-")
        and output.parent.name != "task-trca-n1-source39-",
        "Frozen resource1 output location required",
    )
    return output


def _execute(output, configuration, revision, pins):
    output = _new_output(output)
    output.mkdir()
    started = time.perf_counter()
    state = {"stage": "START", "optimizer_updates_completed": 0, "optimizer_updates_charged": 0}
    common = {
        "schema": SCHEMA,
        "study_id": STUDY_ID,
        "algorithm_schema": learning.SCHEMA,
        "design_sha256": DESIGN_SHA,
        "source_revision": revision,
        "code_pins": pins,
        "seed": SEED,
        "fit_ids": list(FIT_IDS),
        "cases": CASE_COUNT,
        "steps_per_head": 200,
        "arrays_generated": True,
        "human_reads": False,
        "gpu_used": False,
        "device": "cpu",
        "cpu_threads": 1,
        "precision": "float64",
        "backend": "batch",
        "versions": {
            "python": sys.version.split()[0],
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "torch": torch.__version__,
        },
    }
    artifacts, measured = {}, {}
    previous = signal.getsignal(signal.SIGALRM)

    def limit(*unused):
        if time.perf_counter() - started > SECONDS_MAX:
            raise TimeoutError("Frozen1200 second resource limit")
        if _peak_rss_bytes() > RSS_BYTES_MAX:
            raise MemoryError("Frozen4GiB resource RSS limit")

    signal.signal(signal.SIGALRM, limit)
    signal.setitimer(signal.ITIMER_REAL, 1, 1)
    try:
        artifacts["start"] = write_json(
            output,
            "start.json",
            {
                **common,
                "status": "RESOURCE_STARTED",
                "time": datetime.now(timezone.utc).isoformat(),
                "updates_completed": 0,
            },
        )
        _check_code(pins)
        _guard([ROOT / name for name in pins], output)
        state["stage"] = "GENERATED_PREPARATION"
        prep_started = time.perf_counter()
        cases = build_cases(
            configuration,
            seed=SEED,
            fit_ids=FIT_IDS,
            samples=SAMPLES,
            interfaces=INTERFACES,
            budgets=BUDGETS,
        )
        require(len(cases) == CASE_COUNT, "Exact416 actual generated cases")
        actual, ids = learning._validate_cases(cases, fitting=True)
        require(
            ids == FIT_IDS
            and {c.condition for c in actual}
            == {(i, n, k) for i in INTERFACES for n in SAMPLES for k in BUDGETS},
            "Full generated resource condition grid",
        )
        packed = evaluation.pack_cases(cases)
        packed_bytes = sum(value.nbytes for value in packed.values())
        measured["preparation_seconds"] = time.perf_counter() - prep_started
        measured["resident_packed_bytes"] = packed_bytes
        limit()
        state.update(stage="ACTUAL_FOUR_HEAD_FIT", optimizer_updates_charged=UPDATES)
        fit_started = time.perf_counter()
        pipeline = learning.fit_pipeline(cases, 0.001, backend="batch")
        measured["fit_seconds"] = time.perf_counter() - fit_started
        # An actual pack stays live across the fit; it is not serialized under128MiB.
        require(
            sum(value.nbytes for value in packed.values()) == packed_bytes,
            "Resident pack was not retained",
        )
        record = validate_pipeline(pipeline)
        state["optimizer_updates_completed"] = UPDATES
        state["stage"] = "RESOURCE_ASSESSMENT"
        artifacts["pipeline"] = write_json(output, "pipeline.json", record)
        measured["peak_rss_bytes"] = _peak_rss_bytes()
        measured.update(
            projection(
                measured["preparation_seconds"], measured["fit_seconds"], measured["peak_rss_bytes"]
            )
        )
        require(measured["runtime_ok"] and measured["memory_ok"], "Fixed resource screen failed")
        _check_code(pins)
        require(not (output / "failure.json").exists(), "Failure receipt takes precedence")
        limit()
        result = {
            **common,
            **measured,
            "status": "GENERATED_RESOURCE_PASS",
            "updates_completed": UPDATES,
            "elapsed_seconds": time.perf_counter() - started,
            "state": state,
            "start": artifacts["start"],
            "pipeline": artifacts["pipeline"],
            "interpretation": "Generated CPU resource screening estimate, not a certified runtime or human-validity guarantee",
        }
        signal.setitimer(signal.ITIMER_REAL, 0)
        write_json(output, "resource.json", result)
        return result
    except BaseException as error:
        signal.setitimer(signal.ITIMER_REAL, 0)
        for path in output.iterdir():
            if path.is_file() and not path.is_symlink():
                path.chmod(0o400)
        write_json(
            output,
            "failure.json",
            {
                **common,
                **measured,
                "status": "RESOURCE_PREFLIGHT_FAILURE",
                "state": state,
                "updates_completed": state["optimizer_updates_completed"],
                "artifacts": artifacts,
                "elapsed_seconds": time.perf_counter() - started,
                "peak_rss_bytes": _peak_rss_bytes(),
                "error_type": type(error).__name__,
                "error": str(error),
                "traceback": traceback.format_exc(),
            },
        )
        raise
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous)


def run(output):
    output = _new_output(output)
    preflight_started = time.perf_counter()
    try:
        configuration, revision, pins = _preflight()
    except BaseException as error:
        # A rejected path never reaches this block. A setup failure has no fake
        # start/model or completed code binding, but still closes this attempt.
        output = _new_output(output)
        output.mkdir()
        write_json(
            output,
            "failure.json",
            {
                "schema": SCHEMA,
                "study_id": STUDY_ID,
                "design_sha256": DESIGN_SHA,
                "seed": SEED,
                "status": "RESOURCE_FAILURE",
                "stage": "PREFLIGHT",
                "updates_completed": 0,
                "updates_charged": 0,
                "arrays_generated": False,
                "human_reads": False,
                "gpu_used": False,
                "elapsed_seconds": time.perf_counter() - preflight_started,
                "peak_rss_bytes": _peak_rss_bytes(),
                "error_type": type(error).__name__,
                "error": str(error),
                "traceback": traceback.format_exc(),
            },
        )
        raise
    return _execute(output, configuration, revision, pins)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.output), allow_nan=False))
