"""Canonical one-shot artifact reuse/publication; no raw data or policy fitting."""

import hashlib
import io
import json
import os
import stat
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from types import ModuleType

import numpy as np
import scipy

ROOT = Path("/home/whwovy/califreeEEG")
PLAN_PATH = ROOT / "configs/analysis/native_subset_known_zero_source39_v1.json"
PLAN_SHA256 = "7b3e478892574d3edb78f7b6659935a2215cf369c619c7bd1da5fd6b68105258"


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def read_source(path):
    path = Path(path)
    require(path.resolve() == path.absolute() and path.is_file(), "Noncanonical source path")
    return path.read_bytes()


def load_plan(path=PLAN_PATH):
    require(Path(path) == PLAN_PATH, "Only canonical execution plan")
    raw = read_source(path)
    require(digest(raw) == PLAN_SHA256, "Execution plan hash")
    return json.loads(raw)


def compile_helper(relative, expected):
    path = ROOT / relative
    raw = read_source(path)
    require(digest(raw) == expected, "Changed pinned helper: " + relative)
    module = ModuleType("known_zero_pinned_" + path.stem)
    module.__file__ = str(path)
    exec(compile(raw, str(path), "exec"), module.__dict__)  # noqa: S102
    return module


def git(*args):
    return subprocess.check_output(["git", "-C", str(ROOT), *args])


def preflight(path=PLAN_PATH):
    require(Path(__file__).resolve().parents[1] == ROOT, "Canonical integrated repository")
    plan = load_plan(path)
    destination = Path(plan["output_root"])
    require(destination.resolve() == destination, "Canonical output path")
    require(not destination.exists() and not destination.is_symlink(), "No retry/resume/overwrite")
    require(destination.parent.is_dir(), "Existing explicit output parent required")
    require(not git("status", "--porcelain"), "Clean committed main required")
    require(git("branch", "--show-current").decode().strip() == "main", "Main only")
    require(Path(sys.executable).absolute() == ROOT / ".venv/bin/python", "Existing main venv only")
    for actual, key in (
        (sys.version.split()[0], "python"),
        (np.__version__, "numpy"),
        (scipy.__version__, "scipy"),
    ):
        require(actual == plan["runtime"][key], "Runtime version: " + key)
    for name in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS"):
        require(os.environ.get(name) == "1", "BLAS1 required: " + name)
    require(os.environ.get("PYTHONDONTWRITEBYTECODE") == "1", "No bytecode writes")
    commit = git("rev-parse", "HEAD").decode().strip()
    tree = git("rev-parse", "HEAD^{tree}").decode().strip()
    expected = {
        **plan["pinned_helpers"],
        str(PLAN_PATH.relative_to(ROOT)): PLAN_SHA256,
        plan["science"]["path"]: plan["science"]["sha256"],
        plan["mechanism"]["path"]: plan["mechanism"]["sha256"],
        plan["parent"]["execution_plan_path"]: plan["parent"]["execution_plan_sha256"],
    }
    expected.update({rel: digest(read_source(ROOT / rel)) for rel in plan["code_paths"]})
    for rel, pin in expected.items():
        require(digest(read_source(ROOT / rel)) == pin, "Local source pin: " + rel)
        require(digest(git("show", commit + ":" + rel)) == pin, "Committed source: " + rel)
    parent_commit = plan["parent"]["source_commit"]
    require(
        git("rev-parse", parent_commit + "^{tree}").decode().strip()
        == plan["parent"]["source_tree"],
        "Parent committed tree",
    )
    amendment = json.loads(read_source(ROOT / plan["parent"]["execution_plan_path"]))
    for rel, pin in {
        plan["science"]["path"]: plan["science"]["sha256"],
        plan["parent"]["execution_plan_path"]: plan["parent"]["execution_plan_sha256"],
        **amendment["source_helpers"],
    }.items():
        require(
            digest(git("show", parent_commit + ":" + rel)) == pin,
            "Parent committed scientific source: " + rel,
        )
    modules = {
        Path(rel).stem: compile_helper(rel, pin) for rel, pin in plan["pinned_helpers"].items()
    }
    science = json.loads(read_source(ROOT / plan["science"]["path"]))
    require(science["study_id"] == plan["randomization_namespace"], "Unchanged namespace")
    # Initialize statistical code before the closed read/write runtime boundary.
    modules["native_subset_m_core"].interval([0.0, 0.1, 0.2])
    return plan, science, amendment, modules, commit, tree, expected


def guard_for(plan):
    parent = Path(plan["parent"]["root"])
    destination = Path(plan["output_root"])
    inputs = {parent / name for name in plan["parent"]["artifacts"]}
    outputs = {destination / name for name in plan["artifacts"]}
    sources = {ROOT / p for p in (*plan["code_paths"], *plan["pinned_helpers"])}
    libraries = [ROOT / ".venv/lib", Path("/usr/lib/python3.10"), Path("/usr/local/lib/python3.10")]

    def guard(event, args):
        if event in (
            "subprocess.Popen",
            "os.system",
            "socket.connect",
            "socket.bind",
            "os.remove",
            "os.rename",
            "os.rmdir",
            "os.link",
            "os.symlink",
        ):
            raise RuntimeError("Forbidden runtime event: " + event)
        if event == "os.mkdir":
            require(Path(os.fsdecode(args[0])).absolute() == destination, "Only new attempt mkdir")
        if event == "os.listdir":
            require(
                Path(os.fsdecode(args[0])).absolute() in (parent, destination),
                "Only exact artifact inventories",
            )
        if event == "os.chmod" and not isinstance(args[0], int):
            require(Path(os.fsdecode(args[0])).absolute() in outputs, "Only new artifact chmod")
        if event != "open" or not args or isinstance(args[0], int):
            return
        p = Path(os.fsdecode(args[0])).absolute()
        require(".." not in p.parts, "No lexical path traversal")
        flags = args[2] if len(args) > 2 and isinstance(args[2], int) else 0
        mode = args[1] if len(args) > 1 and isinstance(args[1], str) else ""
        write = flags & (os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC) or any(
            c in mode for c in "wax+"
        )
        if write:
            require(p in outputs, "Only two new artifact writes")
        else:
            require(
                p in inputs | outputs | sources | {destination}
                or any(p.is_relative_to(r) for r in libraries),
                "Unapproved runtime read: " + str(p),
            )
        require(p.resolve() == p, "No symlink path traversal")

    return guard


def artifact_bytes(path, max_bytes=134217728):
    path = Path(path)
    require(path.resolve() == path.absolute(), "Artifact path canonical")
    with path.open("rb") as stream:
        info = os.fstat(stream.fileno())
        require(
            stat.S_ISREG(info.st_mode)
            and stat.S_IMODE(info.st_mode) == 0o400
            and info.st_nlink == 1,
            "Artifact regular0400/nlink1",
        )
        require(info.st_size <= max_bytes, "Artifact byte bound before read")
        raw = stream.read(max_bytes + 1)
        require(len(raw) == info.st_size and len(raw) <= max_bytes, "Artifact size changed")
        return raw, info


def load_parent(plan, science, amendment, modules, new_started_at):
    parent = Path(plan["parent"]["root"])
    require(parent.is_dir() and parent.resolve() == parent, "Canonical parent directory")
    require(
        {p.name for p in parent.iterdir()} == set(plan["parent"]["artifacts"]),
        "Exact five parent artifacts",
    )
    blobs, timestamps, total = {}, [], 0
    for name, pin in plan["parent"]["artifacts"].items():
        require(
            total + (parent / name).lstat().st_size <= plan["runtime"]["parent_budget_bytes"],
            "Parent byte budget before reading",
        )
        raw, info = artifact_bytes(parent / name)
        require(digest(raw) == pin, "Parent artifact hash: " + name)
        blobs[name] = raw
        timestamps.append(info.st_mtime_ns)
        total += info.st_size
    require(
        total == plan["parent"]["total_bytes"] and total <= plan["runtime"]["parent_budget_bytes"],
        "Parent byte budget",
    )
    require(timestamps == sorted(timestamps), "Parent publication order")
    payloads = {name: json.loads(raw) for name, raw in blobs.items() if name.endswith(".json")}
    start, freeze, result = (
        payloads[name] for name in ("start.json", "fold-freezes.json", "result.json")
    )
    for receipt in (start, freeze, result):
        for key in ("source_commit", "source_tree", "attempt_id", "study_id"):
            require(receipt[key] == plan["parent"][key], "Parent receipt " + key)
        require(
            receipt["execution_plan_sha256"] == plan["parent"]["execution_plan_sha256"],
            "Parent execution pin",
        )
        require(receipt["plan_sha256"] == plan["science"]["sha256"], "Parent science pin")
        require(receipt["started_at"] == start["started_at"], "Parent receipt timestamp")
    require(
        start["schema"] == "cfeg.native-subset-m.start.v1"
        and freeze["schema"] == "cfeg.native-subset-m.freezes.v1"
        and result["schema"] == science["reporting"]["schema"]
        and result["status"] == "DEVELOPMENT_ASSESSMENT_COMPLETE",
        "Completed parent schemas",
    )
    require(
        start["output_root"] == str(parent)
        and start["source_subject_ids"] == science["source_subject_ids"],
        "Parent source identities/output",
    )
    require(start["source_helper_hashes"] == amendment["source_helpers"], "Parent helper pins")
    require(
        start["core_sha256"] == amendment["source_helpers"]["scripts/native_subset_m_core.py"],
        "Parent core pin",
    )
    for receipt in (freeze, result):
        require(
            receipt["features_sha256"] == plan["parent"]["artifacts"]["features.npz"],
            "Parent cache binding",
        )
        require(
            receipt["source_projection_sha256"]
            == plan["parent"]["artifacts"]["source-projection.json"],
            "Parent projection binding",
        )
    require(
        result["fold_freezes_sha256"] == plan["parent"]["artifacts"]["fold-freezes.json"],
        "Parent freeze binding",
    )
    before, completed, now = [
        datetime.fromisoformat(v)
        for v in (start["started_at"], result["completed_at"], new_started_at)
    ]
    require(
        all(v.tzinfo is not None for v in (before, completed, now)) and before <= completed <= now,
        "Parent completion precedes new attempt",
    )
    wrapper = payloads["source-projection.json"]
    require(
        set(wrapper) == {"schema", "input_path", "input_sha256", "projection"},
        "Projection wrapper keys",
    )
    require(
        wrapper["schema"] == "cfeg.native-subset-m.projection.v1"
        and wrapper["input_path"] == science["source_projection"]["path"]
        and wrapper["input_sha256"] == science["source_projection"]["sha256"],
        "Projection wrapper bindings",
    )
    content = modules["audit_native_subset_m_envelope_r1"].content_projection(
        wrapper["projection"], amendment
    )
    _, order = modules["native_subset_m_core"].projection_arrays(content, science)
    with np.load(io.BytesIO(blobs["features.npz"]), allow_pickle=False) as stored:
        require(len(stored.files) == len(set(stored.files)), "Unique cache members")
        cache = {key: stored[key] for key in stored.files}
    modules["native_subset_m_core"].validate_cache(cache, science)
    numeric_audit = modules["audit_native_subset_m_source"]
    parent_scores = numeric_audit.cache_scores(cache, science)
    numeric_audit.verify_q(cache, parent_scores, order, science)
    for array in cache.values():
        array.flags.writeable = False
    return cache, content, freeze, result


def publish(path, payload, budget):
    raw = (
        json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n"
    ).encode()
    used = sum(p.stat().st_size for p in path.parent.iterdir())
    require(used + len(raw) <= budget, "New output byte budget")
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o400)
    with os.fdopen(fd, "wb") as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())
    directory = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)
    return digest(raw)


def execute(evaluator, execution_path=PLAN_PATH):
    began = time.monotonic()
    plan, science, amendment, modules, commit, tree, hashes = preflight(execution_path)
    common = {
        "attempt_id": plan["attempt_id"],
        "study_id": plan["study_id"],
        "randomization_namespace": plan["randomization_namespace"],
        "execution_plan_sha256": PLAN_SHA256,
        "mechanism_plan_sha256": plan["mechanism"]["sha256"],
        "scientific_plan_sha256": plan["science"]["sha256"],
        "source_commit": commit,
        "source_tree": tree,
        "started_at": datetime.now(timezone.utc).isoformat(),
        "parent_root": plan["parent"]["root"],
        "parent_artifact_sha256": plan["parent"]["artifacts"],
        "source_hashes": hashes,
        "runtime": plan["runtime"],
        "output_root": plan["output_root"],
        "policy_refit": False,
        "held_access": False,
        "retired_access": False,
        "raw_eeg_access": False,
    }
    require(set(common) == set(plan["common_fields"]), "Common receipt fields")
    sys.dont_write_bytecode = True
    sys.addaudithook(guard_for(plan))
    output = Path(plan["output_root"])
    output.mkdir(mode=0o700)
    start_hash = publish(
        output / "start.json",
        {"schema": plan["schemas"]["start"], **common},
        plan["runtime"]["output_budget_bytes"],
    )
    print("New start published; verifying exact five parent artifacts", flush=True)
    cache, content, freezes, parent_result = load_parent(
        plan, science, amendment, modules, common["started_at"]
    )
    frozen_before = json.dumps(freezes, sort_keys=True, allow_nan=False)
    evaluated = evaluator(
        cache,
        content,
        freezes,
        parent_result,
        science,
        modules["native_subset_m_core"],
        modules["native_subset_known_zero"],
    )
    require(
        json.dumps(freezes, sort_keys=True, allow_nan=False) == frozen_before, "Frozen fit mutated"
    )
    require(
        evaluated["parent_replay_verified"] is True, "Complete parent replay must precede revision"
    )
    for key in ("rows", "summary", "contrasts", "diagnostics", "attainment"):
        require(
            len(evaluated[key]) == plan["evaluation"][key], "Complete new reporting grid: " + key
        )
    require(len(evaluated["revision_diagnostics"]) == 624, "Complete revision diagnostics")
    for name, pin in plan["parent"]["artifacts"].items():
        raw, _ = artifact_bytes(Path(plan["parent"]["root"]) / name)
        require(digest(raw) == pin, "Parent input changed during execution")
    require(time.monotonic() - began <= plan["runtime"]["ceiling_seconds"], "Runtime ceiling")
    result = {
        **evaluated,
        **common,
        "schema": plan["schemas"]["result"],
        "status": "DEVELOPMENT_ASSESSMENT_COMPLETE",
        "start_sha256": start_hash,
        "completed_at": datetime.now(timezone.utc).isoformat(),
    }
    require(
        set(result) == {"schema", *plan["common_fields"], *plan["result_extra_fields"]},
        "Exact new result fields",
    )
    publish(output / "result.json", result, plan["runtime"]["output_budget_bytes"])
    print(
        json.dumps(
            {
                "status": result["status"],
                "rows": len(result["rows"]),
                "attempt_id": result["attempt_id"],
            }
        ),
        flush=True,
    )
    return result
