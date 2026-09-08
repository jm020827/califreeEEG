"""One new temporal source39 attempt; same core path for fixed GENERATED rehearsal."""

from __future__ import annotations

import argparse
import gc
import importlib
import json
import os
import resource
import signal
import subprocess
import sys
import time
import traceback
from datetime import datetime, timezone
from functools import partial
from pathlib import Path

import numpy as np
import scipy
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from cfeg.analysis import task_trca_shape_archive as byte_helpers
from cfeg.analysis import task_trca_shape_features as features
from cfeg.analysis import task_trca_temporal_archive as archive
from cfeg.analysis import task_trca_temporal_evaluation as evaluation
from cfeg.analysis import task_trca_temporal_learning as learning
from cfeg.analysis.task_trca_shape_inputs import RolePartition

DESIGN_SHA = "e456c3bcfc066e6ba3cb95a9c76798a006416bee11d064d083d9bd98ce7b97da"
NATIVE_MANIFEST_SHA = "15c088048f3435a500d3758138c3cd6a1616e869d568bdf68d10bcc5e978d83a"
OLD_PLAN_SHA = "ed67cc1f361ed89c37b6f7c5df1e9170b78484e428a5105020485c8c03e3934b"
CODE_PATHS = (
    "scripts/audit_task_trca_temporal_source39.py",
    "scripts/run_task_trca_temporal_source39.py",
    "src/cfeg/__init__.py",
    "src/cfeg/metrics.py",
    *(
        "src/cfeg/analysis/" + name + ".py"
        for name in (
            "__init__",
            "ood_coverage",
            "primary_aggregate",
            "primary_inference",
            "provenance",
            "metadata_prior_validation",
            "metadata_prior_source",
            "metadata_trca_prior",
            "native_support_prefix",
            "task_trca_shape_inputs",
            "task_trca_shape_archive",
            "task_trca_shape_features",
            "task_trca_shape_operator",
            "task_trca_shape_signfree",
            "task_trca_shape_audit",
            "task_trca_temporal_learning",
            "task_trca_temporal_batch",
            "task_trca_temporal_evaluation",
            "task_trca_temporal_audit",
            "task_trca_temporal_archive",
            "task_trca_temporal_artifact_audit",
        )
    ),
)
QUERY_KINDS = frozenset(("query", "full_k3", "full_k5", "a0"))


def require(value, message):
    if not value:
        raise ValueError(message)


def now():
    return datetime.now(timezone.utc).isoformat()


def sha(path):
    fd = byte_helpers.regular_fd(path)
    try:
        return byte_helpers.fd_sha256(fd, os.fstat(fd).st_size)
    finally:
        os.close(fd)


def read_json(path, expected):
    with byte_helpers._PinnedFile(path, expected, maximum_bytes=64 * 1024**2) as source:
        return json.loads(
            byte_helpers.pread_exact(source.fd, source.before.st_size, 0),
            object_pairs_hook=byte_helpers._unique_pairs,
        )


def descriptor(path):
    path = Path(path).absolute()
    return {"path": str(path), "sha256": sha(path), "bytes": path.stat().st_size}


def _space(path, required_bytes, output, budget_bytes):
    path = Path(path).absolute()
    if output is not None:
        require(path.parent == output and path.resolve() == path, "Exact output-only publication")
        used = sum(p.stat().st_size for p in output.iterdir() if p.is_file())
        require(used + required_bytes + 1024**2 <= budget_bytes, "Output budget reserve exhausted")
    return path


def write_json(path, value, *, output=None, budget_bytes=None):
    encoded = (json.dumps(value, indent=2, allow_nan=False) + "\n").encode()
    path = _space(path, len(encoded), output, budget_bytes)
    with path.open("xb") as stream:
        stream.write(encoded)
        stream.flush()
        os.fsync(stream.fileno())
    path.chmod(0o400)
    return descriptor(path)


def write_npz(path, arrays, *, output=None, budget_bytes=None):
    require(
        all(isinstance(v, np.ndarray) and not v.dtype.hasobject for v in arrays.values()),
        "Flat non-object NPZ arrays required",
    )
    reserve = sum(v.nbytes + 8192 for v in arrays.values())
    path = _space(path, reserve, output, budget_bytes)
    with path.open("xb") as stream:
        np.savez(stream, **arrays)
        stream.flush()
        os.fsync(stream.fileno())
    path.chmod(0o400)
    return descriptor(path)


def read_saved_arrays(artifact):
    require(set(artifact) == {"path", "sha256", "bytes"}, "Exact saved-array descriptor")
    with (
        byte_helpers._PinnedFile(artifact["path"], artifact["sha256"], artifact["bytes"]) as pinned,
        os.fdopen(os.dup(pinned.fd), "rb") as stream,
        np.load(stream, allow_pickle=False) as saved,
    ):
        result = {name: saved[name] for name in saved.files}
    require(all(not value.dtype.hasobject for value in result.values()), "Object arrays forbidden")
    return result


class AccessJournal:
    """A row is a durable attempted decode, not an inferred successful read."""

    def __init__(self, path, state, manifest_sha256):
        self.path, self.state, self.manifest_sha256 = Path(path), state, manifest_sha256
        self.seq, self.freeze_sha256 = 0, None
        with self.path.open("xb") as stream:
            stream.flush()
            os.fsync(stream.fileno())

    def _append(self, row):
        with self.path.open("a") as stream:
            stream.write(json.dumps({"seq": self.seq, **row}, allow_nan=False) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        self.seq += 1

    def __call__(self, item):
        require(item.get("phase") == "before_decode", "Only before_decode access events")
        require(
            type(self.state["outer_fold"]) is int and self.state["outer_fold"] in range(3),
            "Explicit outer fold required for each decode",
        )
        if item["kind"] in QUERY_KINDS:
            require(
                self.freeze_sha256 is not None and item.get("freeze_sha256") == self.freeze_sha256,
                "No query decode before durable all-model barrier",
            )
        self._append({"outer_fold": self.state["outer_fold"], **item})
        if item["kind"] in QUERY_KINDS:
            self.state["query_access_count"] += 1

    def barrier(self, freeze_sha256):
        byte_helpers._sha(freeze_sha256)
        require(
            self.freeze_sha256 is None and self.state["query_access_count"] == 0,
            "Exactly one pre-query all-model barrier",
        )
        self._append(
            {
                "kind": "all_models_frozen",
                "phase": "barrier",
                "freeze_sha256": freeze_sha256,
                "manifest_sha256": self.manifest_sha256,
            }
        )
        self.freeze_sha256 = freeze_sha256


class EventJournal:
    """Durable progress sequence linked to the number of persisted access rows."""

    def __init__(self, path, access, started, budget_bytes):
        self.path, self.access, self.started = Path(path), access, started
        self.seq, self.budget_bytes = 0, budget_bytes
        with self.path.open("xb") as stream:
            stream.flush()
            os.fsync(stream.fileno())

    def __call__(self, value):
        require(
            not set(value).intersection(("seq", "access_seq", "time", "elapsed_seconds")),
            "Journal sequence/timing fields cannot be supplied by callers",
        )
        record = {
            **value,
            "seq": self.seq,
            "access_seq": self.access.seq,
            "time": now(),
            "elapsed_seconds": time.perf_counter() - self.started,
        }
        with self.path.open("a") as stream:
            stream.write(json.dumps(record, allow_nan=False) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        self.seq += 1
        print(json.dumps(record, allow_nan=False), flush=True)
        size = sum(p.stat().st_size for p in self.path.parent.iterdir() if p.is_file())
        require(size <= self.budget_bytes, "Frozen output budget exceeded")


def seal_journals(output):
    result = {}
    for name in ("access", "events"):
        path = output / (name + ".jsonl")
        if path.exists():
            path.chmod(0o400)
            result[name] = descriptor(path)
    return result


def failure_context(error):
    """Record diagnostic position without refitting or touching a new input."""
    result = {}
    tb = error.__traceback__
    while tb is not None:
        local, name = tb.tb_frame.f_locals, tb.tb_frame.f_code.co_name
        if name == "_fit_head":
            result["head_step_zero_based"] = local.get("step")
            result["head_trace_rows"] = len(local.get("trace", ()))
            result["head_dimensions"] = local.get("dimensions")
        if name == "fit_pipeline":
            result["head"] = local.get("arm", "Q")
            result["completed_residual_heads"] = sorted(local.get("residuals", {}))
            result["q_completed"] = "qfit" in local
        tb = tb.tb_next
    return result


def generated_summary(scores, a0, profile):
    require(profile == archive.GENERATED_PROFILE, "Generated summary only")
    require(
        scores.shape == (9, 2, 1, 2, 10, 48, 12)
        and a0.shape == (9, 2, 1, 48, 12)
        and np.isfinite(scores).all()
        and np.isfinite(a0).all(),
        "Exact finite generated score grid",
    )
    truth = np.tile(np.arange(12), 4)
    return {
        "status": "METADATA_NOT_EVALUATED",
        "terminal": "GENERATED_NOT_EVALUATED",
        "generated": True,
        "profile": profile.record(),
        "arms": list(evaluation.ARMS),
        "correct_counts": (scores.argmax(-1) == truth).sum(-1).tolist(),
        "a0_correct_counts": (a0.argmax(-1) == truth).sum(-1).tolist(),
        "query_count_per_cell": 48,
        "interpretation": "Generated completed-path rehearsal; not human metadata efficacy",
    }


def _audit_module():
    # Lazy only to allow leaf contract tests before the independent lane merges.
    return importlib.import_module("cfeg.analysis.task_trca_temporal_artifact_audit")


def validate_manifest(manifest, profile):
    require(
        profile in (archive.HUMAN_PROFILE, archive.GENERATED_PROFILE), "Frozen profile required"
    )
    expected_schema = (
        "cfeg.task_trca_temporal.source39_generated.v1"
        if profile.generated
        else "cfeg.task_trca_temporal.source39_execution.v1"
    )
    require(
        manifest.get("schema") == expected_schema
        and manifest.get("status")
        == ("GENERATED_FROZEN" if profile.generated else "EXECUTION_FROZEN"),
        "Wrong execution schema/status; old and generated manifests cannot run as human",
    )
    require(
        manifest.get("profile") == profile.record()
        and json.dumps(manifest["profile"], sort_keys=True)
        == json.dumps(profile.record(), sort_keys=True)
        and manifest.get("source_ids") == list(profile.source_ids)
        and all(type(pid) is int for pid in manifest["source_ids"])
        and manifest.get("generated") is profile.generated,
        "Exact fixed manifest profile required",
    )
    for key in ("held60_authorized", "retired1_3_authorized", "old_attempts_reopened"):
        require(manifest.get(key) is False, "Explicit source-only authority required: " + key)
    require(manifest.get("study_id") == archive.SCHEMA, "Only C1 temporal scientific study")
    require(
        manifest.get("score_schema") == archive.SCORE_SCHEMA,
        "Only fixed temporal scientific scorer",
    )
    require(manifest["design"]["sha256"] == DESIGN_SHA, "Unchanged C1 scientific design pin")
    byte_helpers._sha(manifest["program_sha256"])
    require(
        manifest.get("backend") == "batch"
        and manifest.get("precision") == "float64"
        and type(manifest.get("cpu_threads")) is int
        and manifest["cpu_threads"] == 1,
        "Pinned float64 full-batch CPU1 runtime",
    )
    require(
        manifest.get("device") in (("cpu", "cuda") if profile.generated else ("cuda",)),
        "Frozen device required; no silent runtime fallback",
    )
    require(
        type(manifest.get("max_seconds")) is int and 0 < manifest["max_seconds"] <= 7200,
        "Bounded7200second attempt required",
    )
    require(
        type(manifest.get("optimizer_updates_max")) is int
        and manifest["optimizer_updates_max"] == 24000,
        "Exactly24000 update primary cap required",
    )
    maximum = 4 * 1024**3 if profile.generated else 12 * 1024**3
    require(
        type(manifest.get("output_budget_bytes")) is int
        and 0 < manifest["output_budget_bytes"] <= maximum,
        "Frozen output budget required",
    )
    require(
        type(manifest.get("gpu_memory_fraction")) in (float, int)
        and 0 < manifest["gpu_memory_fraction"] <= 0.16,
        "GPU memory fraction cap",
    )
    require(
        isinstance(manifest.get("code_pins"), dict)
        and set(manifest["code_pins"]) == set(CODE_PATHS),
        "Exact25 dependency code pins required",
    )
    for digest in manifest["code_pins"].values():
        byte_helpers._sha(digest)
    for key in ("program_path", "native_root", "output_root"):
        require(
            isinstance(manifest.get(key), str) and Path(manifest[key]).is_absolute(),
            "Absolute manifest path: " + key,
        )
    for item in (manifest["design"], manifest["native_plan"]):
        require(
            isinstance(item["path"], str) and Path(item["path"]).is_absolute(),
            "Absolute input descriptor path",
        )
        byte_helpers._sha(item["sha256"])
    require(
        manifest["program_path"] == str(ROOT / "configs/analysis/metadata_learning_program_v1.json")
        and manifest["design"]["path"]
        == str(ROOT / "configs/analysis/task_trca_temporal_v1_design.json"),
        "Canonical tracked program/design paths required before parsing",
    )
    require(
        manifest["native_plan"]["path"]
        == str(
            Path(manifest["native_root"]) / "native_plan.json"
            if profile.generated
            else ROOT / "configs/analysis/metadata_prior_source39_v1.json"
        ),
        "Exact approved native configuration path required before parsing",
    )
    byte_helpers._sha(manifest["native_manifest_sha256"])
    if profile.generated:
        require(
            manifest.get("preflight") is None, "Generated rehearsal cannot inherit human preflight"
        )
    else:
        require(
            manifest["native_plan"]["sha256"] == OLD_PLAN_SHA
            and manifest["native_manifest_sha256"] == NATIVE_MANIFEST_SHA,
            "Pinned source39 native input provenance",
        )
        preflight = manifest.get("preflight")
        require(
            isinstance(preflight, dict) and set(preflight) == {"path", "sha256", "bytes"},
            "Completed generated preflight descriptor required before human input",
        )
    return manifest


def validate_program(program, manifest):
    require(
        program["schema"] == "cfeg.metadata_learning_program.v1"
        and program["status"] == "FAMILY_AND_BUDGET_FROZEN"
        and program["program_id"] == "metadata-learning-program-v1",
        "Program authority invalid",
    )
    require(
        program["authority"]["source39_development_authorized_after_candidate_preflight"] is True
        and program["authority"]["held60_authorized"] is False
        and program["authority"]["retired_s1_s3_authorized"] is False,
        "Program source-only authority",
    )
    require(
        len(program["candidates"]) == 2
        and program["candidates"][0]["slot"] == "C1"
        and program["candidates"][0]["id"] == archive.SCHEMA
        and program["candidates"][0]["design_path"]
        == "configs/analysis/task_trca_temporal_v1_design.json"
        and program["candidates"][0]["design_sha256"] == DESIGN_SHA,
        "Program must bind exact C1 scientific design",
    )
    candidate = program["candidates"][1]
    require(
        candidate["slot"] == "C2"
        and candidate["id"] == "task-trca-pair-s-v1"
        and candidate["design_path"] == "configs/analysis/task_trca_pair_s_v1_design.json",
        "C2 mathematical specification must be frozen before C1 input",
    )
    byte_helpers._sha(candidate["design_sha256"])
    design = read_json(ROOT / candidate["design_path"], candidate["design_sha256"])
    require(
        design["schema"] == "cfeg.task_trca_pair_s.design.v1"
        and design["study_id"] == candidate["id"]
        and design["slot"] == "C2"
        and design["program_id"] == program["program_id"]
        and design["status"] == "SCIENTIFIC_SPEC_FROZEN_BEFORE_C1_HUMAN_FITTING"
        and design["inherit_design_sha256"] == manifest["design"]["sha256"]
        and design["score_schema"] == archive.SCORE_SCHEMA,
        "Pinned C2 mathematical freeze does not match this C1 program",
    )


def _no_failure_receipts(output):
    require(
        all(
            not os.path.lexists(output / name)
            for name in ("failure.json", "cold_audit_failure.json")
        ),
        "Failure takes precedence over success receipts",
    )


def validate_preflight(manifest):
    """Resolve the complete generated receipt chain before human data access."""
    proof = manifest["preflight"]
    require(set(proof) == {"path", "sha256", "bytes"}, "Exact preflight descriptor")
    path = Path(proof["path"])
    require(
        path.is_absolute() and path.name == "cold_audit.json" and path.resolve() == path,
        "Exact unaliased cold_audit.json preflight path",
    )
    _no_failure_receipts(path.parent)
    require(
        type(proof["bytes"]) is int and path.stat().st_size == proof["bytes"],
        "Preflight byte count",
    )
    receipt = read_json(path, proof["sha256"])
    require(
        receipt.get("status") == "GENERATED_COLD_INDEPENDENT_AUDIT_PASS"
        and receipt.get("generated") is True
        and receipt.get("profile") == archive.GENERATED_PROFILE.record()
        and receipt.get("code_pins") == manifest["code_pins"]
        and receipt.get("program_sha256") == manifest["program_sha256"],
        "Same-code same-program completed generated cold preflight required",
    )
    generated_path = Path(receipt["manifest_path"])
    require(
        generated_path.is_absolute() and generated_path.resolve() == generated_path,
        "Unaliased generated manifest path required",
    )
    generated = validate_manifest(
        read_json(generated_path, receipt["manifest_sha256"]), archive.GENERATED_PROFILE
    )
    require(
        generated["output_root"] == str(path.parent)
        and generated["code_pins"] == manifest["code_pins"]
        and generated["program_sha256"] == manifest["program_sha256"],
        "Generated preflight manifest/code/program/output binding",
    )
    result = read_json(path.parent / "result.json", receipt["result_sha256"])
    require(
        result.get("status") == "GENERATED_COMPLETE"
        and result.get("cold_audit_status") == "COLD_AUDIT_PENDING"
        and result.get("generated") is True
        and result.get("profile") == generated["profile"]
        and result.get("program_sha256") == generated["program_sha256"]
        and result.get("manifest_sha256") == receipt["manifest_sha256"]
        and result.get("summary", {}).get("terminal") == "GENERATED_NOT_EVALUATED",
        "Generated preflight result must bind the completed same-program manifest",
    )
    _no_failure_receipts(path.parent)
    return receipt


def install_guard(readable, output):
    readable = {Path(v).absolute() for v in readable}
    suffixes = {
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
        ".sqlite3",
    }
    environments = (Path(sys.executable).parent.parent, Path("/usr/local/lib"), Path("/usr/lib"))

    def guard(event, args):
        if event in ("subprocess.Popen", "socket.connect", "socket.getaddrinfo"):
            raise PermissionError("No external processes/network during study")
        if event != "open" or not isinstance(args[0], (str, bytes, os.PathLike)):
            return
        path = Path(os.fsdecode(args[0])).absolute()
        flags = args[2] if len(args) > 2 and isinstance(args[2], int) else 0
        own = path.parent == output and path.resolve() == path
        if flags & (os.O_CREAT | os.O_WRONLY | os.O_RDWR | os.O_TRUNC) and not own:
            raise PermissionError("Write outside new artifacts denied: " + str(path))
        if (
            path.suffix.lower() in suffixes
            and path not in readable
            and not own
            and not any(path.is_relative_to(p) for p in environments)
        ):
            raise PermissionError("Unapproved data/old outcome/held input denied: " + str(path))

    sys.addaudithook(guard)


def _run(manifest_path, manifest_sha, profile):
    manifest_path = Path(manifest_path).absolute()
    manifest = validate_manifest(read_json(manifest_path, manifest_sha), profile)
    program = read_json(manifest["program_path"], manifest["program_sha256"])
    validate_program(program, manifest)
    read_json(manifest["design"]["path"], DESIGN_SHA)
    old = read_json(manifest["native_plan"]["path"], manifest["native_plan"]["sha256"])
    if profile.generated:
        require(
            old.get("generated") is True and old.get("seed") == profile.seed,
            "Generated native plan required",
        )
    else:
        require(ROOT == Path("/home/whwovy/califreeEEG"), "Only integrated canonical human runner")
    weights = {
        key: [old["native_weights"][key][name] for name in old["interfaces"]]
        for key in ("ETRCA", "A0_author")
    }
    require(
        old["interfaces"] == list(archive.INTERFACES) and manifest.get("native_weights") == weights,
        "Explicit native weight tables must match pinned plan",
    )
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    require(
        not subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True),
        "Committed clean integration required",
    )
    if not profile.generated:
        require(
            subprocess.check_output(
                ["git", "branch", "--show-current"], cwd=ROOT, text=True
            ).strip()
            == "main",
            "Human execution only on main",
        )
    for path, digest in manifest["code_pins"].items():
        require(sha(ROOT / path) == digest, "Implementation pin differs: " + path)
    if not profile.generated:
        validate_preflight(manifest)
    require(
        sys.version_info[:2] == (3, 10)
        and np.__version__ == "1.26.4"
        and scipy.__version__ == "1.15.3"
        and torch.__version__ == "2.2.2+cu121",
        "Pinned existing environment required",
    )
    require(
        Path(sys.executable) == Path("/home/whwovy/califreeEEG/.venv/bin/python"),
        "Pinned root executable required",
    )
    require(
        all(
            os.environ.get(v) == "1"
            for v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS")
        )
        and os.environ.get("PYTHONDONTWRITEBYTECODE") == "1",
        "BLAS1 and no-bytecode required",
    )
    torch.set_num_threads(1)
    if manifest["device"] == "cuda":
        require(torch.cuda.is_available(), "Frozen CUDA unavailable; no CPU fallback")
        torch.cuda.set_per_process_memory_fraction(float(manifest["gpu_memory_fraction"]))
        torch.cuda.reset_peak_memory_stats()
    warm = torch.nn.Parameter(torch.zeros(1, dtype=torch.float64, device=manifest["device"]))
    optimizer = torch.optim.Adam([warm], lr=0.01)
    warm.square().sum().backward()
    optimizer.step()
    del warm, optimizer
    return _execute(manifest_path, manifest_sha, manifest, old, profile, revision)


def run(manifest_path, manifest_sha):
    return _run(manifest_path, manifest_sha, archive.HUMAN_PROFILE)


def run_generated(manifest_path, manifest_sha, profile=archive.GENERATED_PROFILE):
    require(profile == archive.GENERATED_PROFILE, "Only fixed generated rehearsal profile")
    return _run(manifest_path, manifest_sha, profile)


def _execute(manifest_path, manifest_sha, manifest, old, profile, revision):
    audit = _audit_module()
    output = Path(manifest["output_root"])
    require(
        output.is_absolute() and output.resolve() == output and not output.exists(),
        "New exact output directory required",
    )
    require(
        (profile.generated or output.parent == Path("/home/whwovy"))
        and output.name.startswith(
            "task-trca-temporal-source39-generated-"
            if profile.generated
            else "task-trca-temporal-source39-v1-"
        ),
        "Bounded new output path",
    )
    require(
        os.statvfs(output.parent).f_bavail * os.statvfs(output.parent).f_frsize >= 20 * 1024**3,
        "Free disk reserve20GiB required",
    )
    output.mkdir()
    started = time.perf_counter()
    state = {
        "stage": "START",
        "outer_fold": None,
        "query_access_count": 0,
        "models_frozen": False,
        "optimizer_updates_completed": 0,
        "optimizer_updates_budgeted": 0,
        "optimizer_updates_charged": 0,
    }
    profile_name = "generated" if profile.generated else "human"
    artifacts = {}
    publish_json = partial(write_json, output=output, budget_bytes=manifest["output_budget_bytes"])
    publish_npz = partial(write_npz, output=output, budget_bytes=manifest["output_budget_bytes"])
    journal = AccessJournal(output / "access.jsonl", state, manifest_sha)
    event = EventJournal(output / "events.jsonl", journal, started, manifest["output_budget_bytes"])

    def progress(row, fold):
        if row["event"] in ("inner_start", "final_refit_start"):
            require(
                state["optimizer_updates_budgeted"] + 800 <= manifest["optimizer_updates_max"],
                "Frozen optimizer update budget exceeded before pipeline fit",
            )
            state["optimizer_updates_budgeted"] += 800
            # On failure charge the full reserved current pipeline: trace rows
            # are appended before optimizer.step and are not exact completions.
            state["optimizer_updates_charged"] = state["optimizer_updates_budgeted"]
        if row["event"] == "inner_complete":
            state["optimizer_updates_completed"] += 800
        event({"outer_fold": fold, **row})

    def alarm(*unused):
        raise TimeoutError("Frozen whole-attempt time limit exceeded")

    signal.signal(signal.SIGALRM, alarm)
    signal.alarm(manifest["max_seconds"])
    try:
        artifacts["start"] = publish_json(
            output / "start.json",
            {
                "status": "STARTED",
                "time": now(),
                "manifest_sha256": manifest_sha,
                "manifest": manifest,
                "source_revision": revision,
                "held60_access": False,
                "old_attempts_reopened": False,
                "device_name": torch.cuda.get_device_name()
                if manifest["device"] == "cuda"
                else "CPU",
                "generated": profile.generated,
                "profile": profile.record(),
            },
        )
        native_root = Path(manifest["native_root"])
        readable = [
            manifest_path,
            manifest["program_path"],
            manifest["design"]["path"],
            manifest["native_plan"]["path"],
            native_root / "result.json",
            native_root / "start.json",
            old["source_projection"]["path"],
        ]
        readable += [ROOT / path for path in manifest["code_pins"]]
        readable += [native_root / f"S{pid:03d}.npz" for pid in profile.source_ids]
        install_guard(readable, output)
        native = read_json(native_root / "result.json", manifest["native_manifest_sha256"])
        require(
            native["status"] == ("GENERATED_COMPLETE" if profile.generated else "COMPLETE")
            and native["plan_sha256"] == manifest["native_plan"]["sha256"]
            and native["source_subject_ids"] == list(profile.source_ids),
            "Native manifest identity",
        )
        native_start = read_json(native_root / "start.json", native["start_sha256"])
        if profile.generated:
            require(
                native.get("generated") is True
                and native.get("seed") == profile.seed
                and native_start.get("generated") is True
                and native_start.get("seed") == profile.seed,
                "Generated native provenance required",
            )
        require(
            [r["subject"] for r in native["files"]] == list(profile.source_ids),
            "Exact native source39 files",
        )
        specs = {}
        for row in native["files"]:
            require(row["filename"] == f"S{row['subject']:03d}.npz", "Native filename identity")
            specs[row["subject"]] = archive.ArchiveSpec(
                native_root / row["filename"], row["subject"], row["sha256"], row["bytes"]
            )
        spec = old["source_projection"]
        envelope = {
            **spec["envelope_provenance"],
            "manifest_sha256": spec["manifest_sha256"],
            "returned_rows": spec["returned_rows"],
            "returned_packets": spec["packets"],
            "returned_subject_ids": list(archive.SOURCE_IDS),
            "columns": spec["columns"],
        }
        weights = [old["native_weights"]["ETRCA"][name] for name in old["interfaces"]]
        a0_weights = np.array(
            [old["native_weights"]["A0_author"][name] for name in old["interfaces"]]
        )
        frozen = []
        for fold in range(3):
            state.update(stage="SOURCE_PREPARATION", outer_fold=fold)
            evaluation_ids = profile.source_ids[fold::3]
            fit_ids = tuple(pid for pid in profile.source_ids if pid not in evaluation_ids)
            partition = RolePartition(fit_ids, (), evaluation_ids)
            cases = []
            with archive.SupportMetadata(
                spec["path"],
                spec["sha256"],
                envelope,
                partition,
                event_sink=journal,
                profile=profile,
            ) as metadata:
                for pid in fit_ids:
                    with archive.NativeArchive(
                        specs[pid], partition, event_sink=journal, profile=profile
                    ) as source:
                        for interface in profile.interfaces:
                            packets = {
                                k: metadata.support(pid, interface, k) for k in profile.budgets
                            }
                            for n in profile.samples:
                                supervision = source.supervision(interface, n)
                                for k in profile.budgets:
                                    packet, order = packets[k]
                                    support = source.support(interface, n, k)
                                    cases.append(
                                        learning.make_task_case(
                                            pid,
                                            interface,
                                            order,
                                            support,
                                            packet,
                                            old["frequencies"],
                                            supervision,
                                            weights=weights[interface],
                                            device=manifest["device"],
                                        )
                                    )
                    event(
                        {"event": "source_participant_ready", "fold": fold, "participant_id": pid}
                    )
            cases = tuple(sorted(cases, key=lambda c: c.key))
            packed = evaluation.pack_cases(cases)
            artifacts[f"source{fold}"] = publish_npz(output / f"source{fold}.npz", packed)
            packed = read_saved_arrays(artifacts[f"source{fold}"])
            state["stage"] = "NESTED_TRAINING"
            pipeline, selection = learning.nested_fit(
                cases,
                outer_evaluation_ids=evaluation_ids,
                backend="batch",
                progress=lambda row, fold=fold: progress(row, fold),
            )
            state["optimizer_updates_completed"] += 800  # Final refit completed.
            model = {
                "pipeline": pipeline.record(),
                "selection": selection,
                "source_artifact": artifacts[f"source{fold}"],
                "manifest_sha256": manifest_sha,
            }
            state["stage"] = "SOURCE_AUDIT"
            receipt = audit.audit_training(
                packed, model, fit_ids, evaluation_ids, profile=profile_name
            )
            model["independent_source_audit"] = receipt
            artifact = publish_json(output / f"model{fold}.json", model)
            artifacts[f"model{fold}"] = artifact
            frozen.append(
                {
                    "fold_id": fold,
                    "fit_ids": list(fit_ids),
                    "evaluation_ids": list(evaluation_ids),
                    **artifact,
                }
            )
            event({"event": "outer_model_frozen", "fold": fold, "source_audit": receipt})
            del cases, packed, pipeline, model
            gc.collect()
            if manifest["device"] == "cuda":
                torch.cuda.empty_cache()
        require(state["query_access_count"] == 0, "Query already accessed before global freeze")
        freeze_doc = {
            "schema": archive.FREEZE_SCHEMA,
            "status": "ALL_MODELS_FROZEN",
            "source_ids": list(profile.source_ids),
            "manifest_sha256": manifest_sha,
            "query_access_count": 0,
            "models": frozen,
        }
        artifacts["freeze"] = publish_json(output / "globalfreeze.json", freeze_doc)
        token = archive.verify_freeze(
            output / "globalfreeze.json",
            artifacts["freeze"]["sha256"],
            manifest_sha,
            profile=profile,
        )
        journal.barrier(artifacts["freeze"]["sha256"])
        state.update(stage="FINAL_QUERY_EVALUATION", models_frozen=True)
        event({"event": "all_models_frozen", "freeze_sha256": artifacts["freeze"]["sha256"]})
        all_scores = np.full(
            (len(profile.source_ids), 2, len(profile.samples), 2, 10, 48, 12), np.nan
        )
        all_a0 = np.full((len(profile.source_ids), 2, len(profile.samples), 48, 12), np.nan)
        coverage_m = np.zeros((len(profile.source_ids), 2, 8, 2))
        coverage_donor = np.zeros_like(coverage_m)
        coverage_available = np.zeros((len(profile.source_ids), 2, 8), dtype=bool)
        r_changes = np.zeros((len(profile.source_ids), 2, len(profile.samples), 2))
        projector_changes = np.zeros_like(r_changes)
        j_changes = np.zeros_like(r_changes)
        qm_coefficients = []
        for fold, frozen_model in enumerate(frozen):
            state["outer_fold"] = fold
            state["stage"] = "FINAL_QUERY_EVALUATION"
            model = read_json(frozen_model["path"], frozen_model["sha256"])
            pipeline = evaluation.pipeline_from_record(model["pipeline"])
            qm_coefficients.append(pipeline.residuals["QM"].coefficients.tolist())
            fit_ids, evaluation_ids = (
                tuple(frozen_model["fit_ids"]),
                tuple(frozen_model["evaluation_ids"]),
            )
            partition = RolePartition(fit_ids, (), evaluation_ids)
            states = {}
            with archive.SupportMetadata(
                spec["path"],
                spec["sha256"],
                envelope,
                partition,
                event_sink=journal,
                profile=profile,
            ) as metadata:
                for pid in evaluation_ids:
                    with archive.NativeArchive(
                        specs[pid], partition, event_sink=journal, profile=profile
                    ) as source:
                        for interface in profile.interfaces:
                            packets = {
                                k: metadata.support(pid, interface, k) for k in profile.budgets
                            }
                            for n in profile.samples:
                                for k in profile.budgets:
                                    packet, order = packets[k]
                                    states[(pid, interface, n, k)] = evaluation.support_state(
                                        pid,
                                        interface,
                                        order,
                                        source.support(interface, n, k),
                                        packet,
                                        old["frequencies"],
                                        weights[interface],
                                    )
            donors = {}
            for interface in profile.interfaces:
                for n in profile.samples:
                    for k in profile.budgets:
                        rows = [states[(pid, interface, n, k)] for pid in evaluation_ids]
                        mapping = features.donor_map(
                            list(evaluation_ids),
                            np.stack([c.mask for c in rows]),
                            [c.order for c in rows],
                            interface,
                        )
                        donors.update(
                            {c.key: (mapping[c.participant_id], *c.condition) for c in rows}
                        )
            evaluation_partition = evaluation.EvaluationPartition(
                tuple(states[key] for key in sorted(states)),
                tuple(evaluation_ids),
                profile.conditions,
            )
            fold_records = []
            for pid in evaluation_ids:
                p = profile.source_ids.index(pid)
                records = []
                with archive.NativeArchive(
                    specs[pid], partition, event_sink=journal, profile=profile
                ) as source:
                    for interface in profile.interfaces:
                        for ni, n in enumerate(profile.samples):
                            query = source.query(interface, n, token).reshape(48, 5, 8, n)
                            a0_corr = source.a0_correlations(interface, n, token).reshape(48, 5, 12)
                            all_a0[p, interface, ni] = np.einsum(
                                "nbc,b->nc", a0_corr, a0_weights[interface]
                            )
                            for ki, k in enumerate(profile.budgets):
                                current = states[(pid, interface, n, k)]
                                donor = states[donors[current.key]]
                                result = evaluation.evaluate(
                                    pipeline,
                                    current,
                                    query,
                                    partition=evaluation_partition,
                                    donor=donor,
                                )
                                result.update(
                                    {
                                        "keys": np.array(current.key),
                                        "orders": current.order,
                                        "q": current.q,
                                        "m": current.m,
                                        "available": current.available,
                                        "packet5": np.pad(
                                            current.packet,
                                            ((0, 5 - k), (0, 0)),
                                            constant_values=np.nan,
                                        ),
                                        "s": current.s.numpy(),
                                        "c": current.c.numpy(),
                                        "anchors": current.anchors.numpy(),
                                        "weights": current.weights.numpy(),
                                        "donor_id": donor.participant_id,
                                        "a0_correlations": a0_corr,
                                        "cached_full_correlations": source.full_correlations(
                                            interface, n, k, token
                                        ).reshape(48, 5, 12),
                                    }
                                )
                                all_scores[p, interface, ni, ki] = result["scores"]
                                r_changes[p, interface, ni, ki] = np.max(
                                    np.abs(result["r"][3] - result["r"][1])
                                )
                                projector_changes[p, interface, ni, ki] = np.max(
                                    np.abs(result["projectors"][3] - result["projectors"][1])
                                )
                                j_changes[p, interface, ni, ki] = np.max(
                                    np.abs(
                                        result["projectors"][3].sum(1)
                                        - result["projectors"][1].sum(1)
                                    )
                                )
                                if k == 3:
                                    coverage_m[p, interface] = current.m
                                    coverage_donor[p, interface] = donor.m
                                    coverage_available[p, interface] = current.available
                                records.append(result)
                packed_eval = audit.pack_evaluation(records)
                artifact = publish_npz(output / f"evaluation_S{pid:03d}.npz", packed_eval)
                artifacts[f"evaluation{pid}"] = artifact
                # Independent donor scoring needs the entire outer partition;
                # raw per-participant files are audited together below.
                fold_records.append(read_saved_arrays(artifact))
                event(
                    {"event": "evaluation_participant_saved", "fold": fold, "participant_id": pid}
                )
            packed_fold = audit.combine_evaluation(fold_records)
            state["stage"] = "EVALUATION_AUDIT"
            receipt = audit.audit_evaluation(
                packed_fold, model["pipeline"], evaluation_ids, profile=profile_name
            )
            artifacts[f"evaluation_audit{fold}"] = publish_json(
                output / f"evaluation_audit{fold}.json", receipt
            )
            event({"event": "evaluation_fold_audit_pass", "fold": fold, "receipt": receipt})
            del states, packed_fold, fold_records, packed_eval
            gc.collect()
        state["stage"] = "TERMINAL_ASSESSMENT"
        require(
            np.isfinite(all_scores).all() and np.isfinite(all_a0).all(), "Incomplete score grid"
        )
        aggregates = {
            "scores": all_scores,
            "a0": all_a0,
            "coverage_m": coverage_m,
            "coverage_donor_m": coverage_donor,
            "coverage_available": coverage_available,
            "r_changes": r_changes,
            "projector_changes": projector_changes,
            "j_changes": j_changes,
            "qm_coefficients": np.array(qm_coefficients),
            "weights": np.array(weights),
            "a0_weights": a0_weights,
        }
        artifacts["scores"] = publish_npz(output / "scores.npz", aggregates)
        from cfeg.analysis.task_trca_temporal_audit import summarize

        summary = (
            generated_summary(all_scores, all_a0, profile)
            if profile.generated
            else summarize(
                all_scores,
                all_a0,
                {"m": coverage_m, "donor_m": coverage_donor, "available": coverage_available},
                {
                    "band_weights": weights,
                    "a0_band_weights": a0_weights,
                    "r_max_abs_qm_minus_q": r_changes,
                    "projector_max_abs_qm_minus_q": projector_changes,
                    "j_max_abs_qm_minus_q": j_changes,
                    "qm_coefficients": qm_coefficients,
                },
            )
        )
        result = {
            "status": "GENERATED_COMPLETE" if profile.generated else "COMPLETE",
            "cold_audit_status": "COLD_AUDIT_PENDING",
            "generated": profile.generated,
            "profile": profile.record(),
            "program_sha256": manifest["program_sha256"],
            "summary": summary,
            "artifacts": artifacts,
            "manifest_sha256": manifest_sha,
            "source_revision": revision,
            "time": now(),
            "elapsed_seconds": time.perf_counter() - started,
            "peak_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated()
            if manifest["device"] == "cuda"
            else 0,
            "state": state,
            "held60_access": False,
            "query_metadata_decoded": False,
            "evidence_scope": "GENERATED rehearsal; human metadata NOT_EVALUATED"
            if profile.generated
            else "repeatedly exposed source39 development; no independent confirmation",
        }
        require(
            state["query_access_count"]
            == len(profile.source_ids) * len(profile.interfaces) * len(profile.samples) * 4,
            "Exact query-bearing decode attempt total required",
        )
        event(
            {
                "event": "attempt_verified_before_publication",
                "result_path": str(output / "result.json"),
                "terminal": summary.get("terminal"),
            }
        )
        require(
            sum(p.stat().st_size for p in output.iterdir() if p.is_file())
            + len(json.dumps(result)) * 2
            + 1024**2
            < manifest["output_budget_bytes"],
            "Final receipt budget reserve",
        )
        require(
            state["optimizer_updates_completed"] == 24000,
            "Exactly30 pipelines/120heads/24000 task updates required",
        )
        _no_failure_receipts(output)
        artifacts.update(seal_journals(output))
        result["artifacts"] = artifacts
        signal.alarm(0)
        publish_json(output / "result.json", result)
        return result
    except BaseException as exc:
        signal.alarm(0)
        state["failure_context"] = failure_context(exc)
        # Preserve even an incompletely written new artifact, read-only. Its
        # presence never becomes success: the exclusive failure receipt wins.
        for path in output.iterdir():
            if path.is_file() and not path.is_symlink():
                path.chmod(0o400)
        artifacts.update(seal_journals(output))
        write_json(
            output / "failure.json",
            {
                "status": "VALIDITY_FAILURE",
                "error_type": type(exc).__name__,
                "error": str(exc),
                "traceback": traceback.format_exc(),
                "state": state,
                "artifacts": artifacts,
                "time": now(),
                "elapsed_seconds": time.perf_counter() - started,
                "held60_access": False,
                "manifest_sha256": manifest_sha,
                "interpretation": "Candidate attempt stopped; no scientific tuning, sample exclusion, or efficacy interpretation of partial results",
            },
        )
        raise
    finally:
        signal.alarm(0)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--manifest-sha256", required=True)
    args = parser.parse_args()
    run(args.manifest, args.manifest_sha256)
