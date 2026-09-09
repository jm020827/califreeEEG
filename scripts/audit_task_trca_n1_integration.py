"""Independent cold audit of a manifest-bound completed temporal attempt.

Reads generated source archives/projection only for byte hashes, then saved
statistics for numerical replay. No producer, optimizer, N1 or source decoder imports.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import signal
import stat
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from cfeg.analysis import task_trca_n1_artifact_audit as audit
from cfeg.analysis import task_trca_n1_audit as independent

CODE_PATHS = (
    "configs/analysis/task_trca_n1_integration_v1.json",
    "configs/analysis/metadata_prior_source39_v1.json",
    "docs/task_trca_n1_integration_v1_build.md",
    "scripts/prepare_task_trca_n1_integration.py",
    "scripts/audit_task_trca_n1_integration.py",
    "scripts/run_task_trca_n1_integration.py",
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
            "numerical_stability_operator",
            "task_trca_n1_signfree",
            "task_trca_n1_learning",
            "task_trca_n1_batch",
            "task_trca_n1_evaluation",
            "task_trca_n1_audit",
            "task_trca_n1_archive",
            "task_trca_n1_artifact_audit",
        )
    ),
)
DESIGN_SHA = "15d5ee385bc0f30d5dff780ae582034597db06793d34c9c8029c44f42adf25cd"
DESIGN_PATH = "configs/analysis/task_trca_n1_integration_v1.json"
INPUT_LIMIT = 2 * 1024**3
AUDIT_SECONDS = 1800
SHA_PATTERN = re.compile(r"[0-9a-f]{64}\Z")
QUERY_KINDS = ("query", "a0", "full_k3", "full_k5")


def _identity(info):
    return (
        info.st_dev,
        info.st_ino,
        info.st_size,
        info.st_mtime_ns,
        info.st_ctime_ns,
        info.st_nlink,
    )


def _pairs(pairs):
    result = {}
    for key, value in pairs:
        audit.require(key not in result, "duplicate JSON key")
        result[key] = value
    return result


def _json(raw):
    return json.loads(
        raw,
        object_pairs_hook=_pairs,
        parse_constant=lambda value: audit.require(False, "nonfinite JSON " + value),
    )


def _hash_stream(stream):
    stream.seek(0)
    digest = hashlib.sha256()
    for block in iter(lambda: stream.read(1024 * 1024), b""):
        digest.update(block)
    return digest.hexdigest()


def read_pinned(path, expected_sha=None, expected_bytes=None, *, kind="json", immutable=True):
    """One regular single-link descriptor, hash before/after parsing and path identity."""
    path = Path(path).absolute()
    audit.require(path.resolve() == path and not path.is_symlink(), "unaliased pinned path")
    if expected_sha is not None:
        audit.require(
            isinstance(expected_sha, str) and SHA_PATTERN.fullmatch(expected_sha), "expected SHA256"
        )
    if expected_bytes is not None:
        audit.require(type(expected_bytes) is int and expected_bytes > 0, "positive byte binding")
    with os.fdopen(os.open(path, os.O_RDONLY | os.O_NOFOLLOW), "rb") as stream:
        before = os.fstat(stream.fileno())
        audit.require(
            stat.S_ISREG(before.st_mode) and before.st_nlink == 1, "single-link regular pinned file"
        )
        audit.require(not immutable or stat.S_IMODE(before.st_mode) == 0o400, "immutable0400 input")
        audit.require(0 < before.st_size <= INPUT_LIMIT, "bounded input bytes")
        audit.require(expected_bytes is None or before.st_size == expected_bytes, "pinned size")
        digest = _hash_stream(stream)
        audit.require(expected_sha is None or digest == expected_sha, "pinned hash: " + path.name)
        stream.seek(0)
        if kind == "npz":
            with np.load(stream, allow_pickle=False) as arrays:
                audit.require(len(set(arrays.files)) == len(arrays.files), "unique NPZ names")
                value = {name: arrays[name] for name in arrays.files}
                audit.require(
                    all(a.dtype.kind in "biufU" for a in value.values()),
                    "no object/complex/byte-string NPZ fields",
                )
        elif kind == "json":
            value = _json(stream.read())
        elif kind == "jsonl":
            value = [_json(line) for line in stream.read().splitlines()]
        elif kind == "hash":
            value = None
        else:
            raise ValueError("unknown pinned read kind")
        audit.require(
            _hash_stream(stream) == digest
            and _identity(os.fstat(stream.fileno())) == _identity(before),
            "pinned descriptor changed during read",
        )
        audit.require(
            path.resolve() == path
            and _identity(os.stat(path, follow_symlinks=False)) == _identity(before),
            "pinned path replaced during read",
        )
    return value, {"path": str(path), "sha256": digest, "bytes": before.st_size}


def failure_precedence(output):
    for name in ("failure.json", "cold_audit_failure.json"):
        audit.require(not os.path.lexists(output / name), name + " has precedence")
    for name in ("prep_failure.json", "prepare_failure.json"):
        audit.require(not os.path.lexists(output.parent / name), name + " has precedence")


def strict_equal(actual, expected, message):
    audit.require(
        json.dumps(actual, sort_keys=True, allow_nan=False)
        == json.dumps(expected, sort_keys=True, allow_nan=False),
        message,
    )


def descriptor(item, path):
    audit.require(
        isinstance(item, dict)
        and set(item) == {"path", "sha256", "bytes"}
        and item["path"] == str(path)
        and isinstance(item["sha256"], str)
        and SHA_PATTERN.fullmatch(item["sha256"])
        and type(item["bytes"]) is int
        and 0 < item["bytes"] <= INPUT_LIMIT,
        "exact descriptor/path binding",
    )
    return item


def fixture_binding(manifest):
    """Hash generated source members only; never decode source EEG or projection numbers."""
    native_root = Path(manifest["native_root"])
    fixture_desc = descriptor(manifest["fixture"], native_root / "fixture.json")
    expected_names = {
        "native_plan.json",
        "start.json",
        "result.json",
        "fixture.json",
        "projection.json",
        *(f"S{pid:03d}.npz" for pid in audit.profile_record()["source_ids"]),
    }
    audit.require(
        {p.name for p in native_root.iterdir()} == expected_names, "exact14 generated input files"
    )
    audit.require(
        sum(p.lstat().st_size for p in native_root.iterdir()) + manifest["output_budget_bytes"]
        <= INPUT_LIMIT,
        "combined generated input/run byte budget",
    )
    fixture, _ = read_pinned(fixture_desc["path"], fixture_desc["sha256"], fixture_desc["bytes"])
    audit.require(
        fixture.get("status") == "GENERATED_INPUTS_ONLY"
        and type(fixture.get("seed")) is int
        and fixture["seed"] == 20260913
        and fixture.get("human_artifact_reads") is False
        and fixture.get("metadata_effect") == "NOT_EVALUATED",
        "generated fixture origin",
    )
    strict_equal(fixture["profile"], audit.profile_record(), "fixture exact profile")
    pins = manifest["code_pins"]
    for field, name in (
        ("generator_sha256", "scripts/prepare_task_trca_n1_integration.py"),
        ("native_function_sha256", "src/cfeg/analysis/metadata_trca_prior.py"),
        ("native_configuration_sha256", "configs/analysis/metadata_prior_source39_v1.json"),
    ):
        audit.require(fixture[field] == pins[name], "fixture generation-code binding")
    for key, basename in (
        ("result", "result.json"),
        ("plan", "native_plan.json"),
        ("projection", "projection.json"),
    ):
        item = descriptor(fixture[key], native_root / basename)
        read_pinned(item["path"], item["sha256"], item["bytes"], kind="hash")
    audit.require(
        fixture["result"]["sha256"] == manifest["native_manifest_sha256"]
        and fixture["plan"]["sha256"] == manifest["native_plan"]["sha256"],
        "fixture/native manifest and plan bindings",
    )
    plan, _ = read_pinned(
        fixture["plan"]["path"], fixture["plan"]["sha256"], fixture["plan"]["bytes"]
    )
    audit.require(
        plan.get("generated") is True
        and type(plan.get("seed")) is int
        and plan["seed"] == 20260913
        and plan["interfaces"] == ["dry", "wet"],
        "generated native plan",
    )
    source = plan["source_projection"]
    audit.require(
        source["path"] == str(native_root / "projection.json")
        and source["sha256"] == fixture["projection"]["sha256"],
        "generated projection exact child binding",
    )
    origin = source["envelope_provenance"]
    audit.require(
        "GENERATED" in origin["schema"]
        and "GENERATED" in origin["study_id"]
        and independent.SCHEMA in origin["study_id"],
        "new generated projection identity",
    )
    audit.require(
        source["packets"] == 780 and source["returned_rows"] == 9360,
        "complete generated routing projection",
    )
    configuration, _ = read_pinned(
        ROOT / "configs/analysis/metadata_prior_source39_v1.json",
        pins["configs/analysis/metadata_prior_source39_v1.json"],
        immutable=False,
    )
    strict_equal(plan["frequencies"], configuration["frequencies"], "native frequencies unchanged")
    strict_equal(
        plan["native_weights"], configuration["native_weights"], "native weights unchanged"
    )
    strict_equal(
        source["columns"], configuration["source_projection"]["columns"], "projection columns"
    )
    for name in ("ETRCA", "A0_author"):
        strict_equal(
            manifest["native_weights"][name],
            [plan["native_weights"][name][i] for i in ("dry", "wet")],
            "manifest native weights",
        )
    native, _ = read_pinned(
        fixture["result"]["path"], fixture["result"]["sha256"], fixture["result"]["bytes"]
    )
    audit.require(
        native["status"] == "GENERATED_COMPLETE"
        and native.get("generated") is True
        and type(native.get("seed")) is int
        and native["seed"] == 20260913
        and native["plan_sha256"] == fixture["plan"]["sha256"],
        "generated native receipt",
    )
    strict_equal(native["source_subject_ids"], audit.profile_record()["source_ids"], "nine EEG IDs")
    start, _ = read_pinned(native_root / "start.json", native["start_sha256"])
    audit.require(
        start.get("generated") is True
        and type(start.get("seed")) is int
        and start["seed"] == 20260913,
        "generated native start",
    )
    strict_equal(start["profile"], audit.profile_record(), "native start profile")
    audit.require(
        isinstance(native["files"], list) and len(native["files"]) == 9,
        "nine native source descriptors",
    )
    for pid, item in zip(audit.profile_record()["source_ids"], native["files"]):
        audit.require(
            set(item) == {"subject", "filename", "sha256", "bytes"}
            and type(item["subject"]) is int
            and item["subject"] == pid
            and item["filename"] == f"S{pid:03d}.npz",
            "native descriptor identity",
        )
        bound = descriptor(
            {
                "path": str(native_root / item["filename"]),
                "sha256": item["sha256"],
                "bytes": item["bytes"],
            },
            native_root / f"S{pid:03d}.npz",
        )
        read_pinned(bound["path"], bound["sha256"], bound["bytes"], kind="hash")
    return fixture


def validate_manifest(manifest, expected_sha):
    audit.require(
        isinstance(expected_sha, str) and SHA_PATTERN.fullmatch(expected_sha),
        "manifest trust-root hash",
    )
    keys = {
        "schema",
        "study_id",
        "score_schema",
        "attempt_id",
        "status",
        "generated",
        "seed",
        "profile",
        "source_ids",
        "held60_authorized",
        "retired1_3_authorized",
        "old_attempts_reopened",
        "design",
        "native_plan",
        "native_root",
        "native_manifest_sha256",
        "native_weights",
        "code_pins",
        "device",
        "backend",
        "precision",
        "cpu_threads",
        "output_root",
        "output_budget_bytes",
        "max_seconds",
        "optimizer_updates_max",
        "preflight",
        "fixture",
    }
    audit.require(
        isinstance(manifest, dict) and set(manifest) == keys, "exact generated manifest fields"
    )
    audit.require(
        manifest["schema"] == "cfeg.task_trca_n1.generated_execution.v1"
        and manifest["status"] == "GENERATED_FROZEN"
        and manifest["generated"] is True,
        "generated-only executable schema/status",
    )
    contract = audit.profile_record()
    strict_equal(manifest["profile"], contract, "exact externally frozen generated profile")
    strict_equal(manifest["source_ids"], contract["source_ids"], "nine generated source IDs")
    audit.require(
        type(manifest["seed"]) is int and manifest["seed"] == 20260913,
        "fixed registered generated seed",
    )
    audit.require(
        manifest["study_id"] == independent.SCHEMA
        and manifest["score_schema"] == independent.SCORE_SCHEMA
        and manifest["device"] == "cpu"
        and manifest["backend"] == "batch"
        and manifest["precision"] == "float64"
        and manifest["preflight"] is None,
        "fixed N1 CPU runtime, no human preflight",
    )
    for key in ("held60_authorized", "retired1_3_authorized", "old_attempts_reopened"):
        audit.require(manifest[key] is False, "no human/held/old-attempt authority")
    output, native_root = Path(manifest["output_root"]), Path(manifest["native_root"])
    parent = output.parent
    audit.require(
        output.is_absolute()
        and output.resolve() == output
        and parent.name.startswith("task-trca-n1-integration-")
        and output.name == "task-trca-n1-integration-primary1"
        and native_root == parent / "inputs"
        and native_root.resolve() == native_root
        and native_root.is_dir()
        and manifest["attempt_id"] == output.name,
        "exact unaliased generated sibling paths",
    )
    failure_precedence(output)
    strict_equal(
        manifest["design"],
        {"path": str(ROOT / DESIGN_PATH), "sha256": DESIGN_SHA},
        "new frozen design binding",
    )
    audit.require(
        isinstance(manifest["native_plan"], dict)
        and set(manifest["native_plan"]) == {"path", "sha256"}
        and manifest["native_plan"]["path"] == str(native_root / "native_plan.json"),
        "generated native plan before reads",
    )
    pins = manifest["code_pins"]
    audit.require(isinstance(pins, dict) and set(pins) == set(CODE_PATHS), "exact31 code-pin set")
    for name in CODE_PATHS:
        read_pinned(ROOT / name, pins[name], kind="hash", immutable=False)
    audit.require(pins[DESIGN_PATH] == DESIGN_SHA, "frozen science code pin")
    design, _ = read_pinned(ROOT / DESIGN_PATH, DESIGN_SHA, immutable=False)
    strict_equal(design["generated_profile"], contract, "design profile")
    audit.require(
        design["study_id"] == independent.SCHEMA
        and design["numerical_prerequisite"]["selected_method"] == "N1_SYM_CHOLESKY",
        "new design/numerical method",
    )
    for key, minimum, maximum in (
        ("cpu_threads", 1, 1),
        ("max_seconds", 1, 7200),
        ("output_budget_bytes", 1, INPUT_LIMIT),
        ("optimizer_updates_max", 24000, 24000),
    ):
        audit.require(
            type(manifest[key]) is int and minimum <= manifest[key] <= maximum,
            "fixed runtime budget: " + key,
        )
    fixture_binding(manifest)
    return "generated", contract


def artifact_names(ids):
    return {
        "start": "start.json",
        "freeze": "globalfreeze.json",
        "scores": "scores.npz",
        "access": "access.jsonl",
        "events": "events.jsonl",
        **{f"source{fold}": f"source{fold}.npz" for fold in range(3)},
        **{f"model{fold}": f"model{fold}.json" for fold in range(3)},
        **{f"evaluation_audit{fold}": f"evaluation_audit{fold}.json" for fold in range(3)},
        **{f"evaluation{pid}": f"evaluation_S{pid:03d}.npz" for pid in ids},
    }


def bind_artifacts(output, artifacts, ids):
    names = artifact_names(ids)
    audit.require(
        isinstance(artifacts, dict) and set(artifacts) == set(names), "exact artifact inventory"
    )
    for key, name in names.items():
        item = artifacts[key]
        audit.require(
            isinstance(item, dict)
            and set(item) == {"path", "sha256", "bytes"}
            and item["path"] == str(output / name),
            "fixed artifact basename: " + key,
        )
        read_pinned(item["path"], item["sha256"], item["bytes"], kind="hash")
    return names


def load_artifact(artifacts, key, kind="json"):
    item = artifacts[key]
    return read_pinned(item["path"], item["sha256"], item["bytes"], kind=kind)[0]


def check_access(rows, contract, freeze_sha, manifest_sha):
    """Exact full request multiset plus durable sequence/global-freeze barrier."""
    ids, windows = tuple(contract["source_ids"]), contract["samples"]
    expected, observed = Counter(), Counter()

    def event_key(fold, role, pid, interface, samples, kind, blocks, freeze=None):
        return fold, role, pid, interface, samples, kind, tuple(blocks), freeze

    for fold in range(3):
        for pid in ids:
            role = "evaluation" if pid in ids[fold::3] else "fit"
            for interface in (0, 1):
                for k in (3, 5):
                    expected[
                        event_key(fold, role, pid, interface, None, "metadata_support", range(k))
                    ] += 1
                for samples in windows:
                    for k in (3, 5):
                        expected[
                            event_key(fold, role, pid, interface, samples, "support", range(k))
                        ] += 1
                    if role == "fit":
                        expected[
                            event_key(
                                fold, role, pid, interface, samples, "source_supervision", (5,)
                            )
                        ] += 1
                    else:
                        for kind in QUERY_KINDS:
                            expected[
                                event_key(
                                    fold,
                                    role,
                                    pid,
                                    interface,
                                    samples,
                                    kind,
                                    range(6, 10),
                                    freeze_sha,
                                )
                            ] += 1
    barrier = False
    kind_counts = Counter()
    for sequence, row in enumerate(rows):
        audit.require(
            type(row.get("seq")) is int and row["seq"] == sequence, "durable event sequence"
        )
        if row.get("kind") == "all_models_frozen":
            audit.require(
                set(row) == {"seq", "kind", "phase", "freeze_sha256", "manifest_sha256"}
                and not barrier
                and row["phase"] == "barrier"
                and row["freeze_sha256"] == freeze_sha
                and row["manifest_sha256"] == manifest_sha,
                "unique bound global-freeze barrier",
            )
            audit.require(
                all(observed[key] == value for key, value in expected.items() if key[1] == "fit"),
                "all source reads precede global freeze",
            )
            barrier = True
            continue
        kind = row.get("kind")
        required = {
            "seq",
            "outer_fold",
            "phase",
            "participant_id",
            "role",
            "kind",
            "interface",
            "blocks",
        }
        if kind != "metadata_support":
            required.add("samples")
        if kind in QUERY_KINDS:
            required.add("freeze_sha256")
        audit.require(
            set(row) == required and row["phase"] == "before_decode", "exact read-event fields"
        )
        audit.require(
            all(type(row[name]) is int for name in ("outer_fold", "participant_id", "interface"))
            and type(row["blocks"]) is list
            and all(type(v) is int for v in row["blocks"]),
            "integer read-event routing",
        )
        if kind != "metadata_support":
            audit.require(type(row["samples"]) is int, "integer read window")
        audit.require(row["role"] != "fit" or not barrier, "source read after freeze")
        audit.require(row["role"] != "evaluation" or barrier, "evaluation phase before freeze")
        key = event_key(
            row["outer_fold"],
            row["role"],
            row["participant_id"],
            row["interface"],
            row.get("samples"),
            kind,
            row["blocks"],
            row.get("freeze_sha256"),
        )
        observed[key] += 1
        kind_counts[(row["role"], kind)] += 1
        audit.require(
            key in expected and observed[key] <= expected[key], "unexpected/duplicate read cell"
        )
    audit.require(barrier and observed == expected, "complete exact read-event Cartesian multiset")
    return {f"{role}:{kind}": value for (role, kind), value in sorted(kind_counts.items())}


def check_events(rows, access, contract, artifacts, models, result, output):
    """Cross-journal source freeze, evaluation completion and publication order."""
    ids = tuple(contract["source_ids"])
    barrier_index = next(j for j, row in enumerate(access) if row["kind"] == "all_models_frozen")
    frozen, saved, audited = [], [], []
    barrier = False
    previous_access, previous_elapsed = 0, 0.0
    for sequence, row in enumerate(rows):
        audit.require(
            type(row.get("seq")) is int
            and row["seq"] == sequence
            and type(row.get("access_seq")) is int
            and previous_access <= row["access_seq"] <= len(access),
            "ordered event/access sequence binding",
        )
        elapsed = row.get("elapsed_seconds")
        audit.require(
            type(elapsed) in (int, float) and np.isfinite(elapsed) and elapsed >= previous_elapsed,
            "monotone finite event elapsed time",
        )
        previous_access, previous_elapsed = row["access_seq"], elapsed
        name = row.get("event")
        if name == "outer_model_frozen":
            fold = row["fold"]
            audit.require(
                not barrier
                and type(fold) is int
                and fold == len(frozen)
                and fold in (0, 1, 2)
                and row["access_seq"] <= barrier_index
                and row["source_audit"] == models[fold]["independent_source_audit"],
                "three ordered source-audited model freeze events",
            )
            frozen.append(fold)
        elif name == "all_models_frozen":
            audit.require(
                not barrier
                and frozen == [0, 1, 2]
                and row["access_seq"] == barrier_index + 1
                and row["freeze_sha256"] == artifacts["freeze"]["sha256"],
                "events/access exact global-freeze barrier",
            )
            barrier = True
        elif name == "evaluation_participant_saved":
            fold, pid = row["fold"], row["participant_id"]
            audit.require(
                barrier
                and type(fold) is int
                and fold in (0, 1, 2)
                and type(pid) is int
                and pid in ids[fold::3]
                and (fold, pid) not in saved,
                "exact evaluation completion event",
            )
            last_read = max(
                j
                for j, event in enumerate(access)
                if event.get("role") == "evaluation"
                and event.get("participant_id") == pid
                and event.get("outer_fold") == fold
            )
            audit.require(
                row["access_seq"] > last_read, "participant saved after all permitted reads"
            )
            saved.append((fold, pid))
        elif name == "evaluation_fold_audit_pass":
            fold = row["fold"]
            audit.require(
                barrier
                and type(fold) is int
                and fold == len(audited)
                and fold in (0, 1, 2)
                and all((fold, pid) in saved for pid in ids[fold::3])
                and row["receipt"] == load_artifact(artifacts, f"evaluation_audit{fold}"),
                "ordered completed evaluation fold audits",
            )
            audited.append(fold)
        elif name == "attempt_verified_before_publication":
            audit.require(
                sequence == len(rows) - 1
                and barrier
                and audited == [0, 1, 2]
                and saved == [(fold, pid) for fold in range(3) for pid in ids[fold::3]]
                and row["access_seq"] == len(access)
                and row["result_path"] == str(output / "result.json")
                and row["terminal"] == result["summary"]["terminal"],
                "last complete publication barrier",
            )
        else:
            audit.require(
                not barrier
                and name
                in {
                    "source_participant_ready",
                    "inner_start",
                    "inner_complete",
                    "final_refit_start",
                },
                "only source progress before global freeze",
            )
    audit.require(
        bool(rows) and rows[-1].get("event") == "attempt_verified_before_publication",
        "required final publication event",
    )
    return {
        "event_count": len(rows),
        "access_count": len(access),
        "all_models_frozen_access_seq": barrier_index + 1,
        "publication_barrier_verified": True,
    }


def check_freeze(freeze, artifacts, models, manifest_sha, contract):
    audit.require(
        set(freeze)
        == {"schema", "status", "source_ids", "manifest_sha256", "query_access_count", "models"},
        "exact freeze envelope",
    )
    audit.require(
        freeze["schema"] == "cfeg.task_trca_n1.all_models_frozen.v1"
        and freeze["status"] == "ALL_MODELS_FROZEN"
        and freeze["source_ids"] == contract["source_ids"]
        and freeze["manifest_sha256"] == manifest_sha
        and type(freeze["query_access_count"]) is int
        and freeze["query_access_count"] == 0,
        "global freeze authority",
    )
    ids = tuple(contract["source_ids"])
    expected = []
    for fold, model in enumerate(models):
        evaluation = ids[fold::3]
        fitting = tuple(pid for pid in ids if pid not in evaluation)
        expected.append(
            {
                "fold_id": fold,
                "fit_ids": list(fitting),
                "evaluation_ids": list(evaluation),
                **artifacts[f"model{fold}"],
            }
        )
        audit.require(
            model["source_artifact"] == artifacts[f"source{fold}"]
            and model["manifest_sha256"] == manifest_sha,
            "model/source/manifest byte binding",
        )
        independent._schema(model["pipeline"])
        independent._schema(model["selection"])
        audit.require(
            model["pipeline"]["fit_ids"] == list(fitting)
            and model["selection"]["outer_evaluation_ids"] == list(evaluation),
            "frozen model roles",
        )
        audit.require(
            model["independent_source_audit"]["status"] == "TEMPORAL_SOURCE_SELECTION_AUDIT_PASS",
            "model completed source audit",
        )
    audit.require(
        freeze["models"] == expected, "freeze models exactly match saved evaluation model bytes"
    )


def generated_summary(aggregate):
    truth = np.tile(np.arange(12), 4)
    counts = (aggregate["scores"].argmax(-1) == truth).sum(-1, dtype=np.int64)
    a0_counts = (aggregate["a0"].argmax(-1) == truth).sum(-1, dtype=np.int64)
    return {
        "status": "METADATA_NOT_EVALUATED",
        "metadata_effect": "NOT_EVALUATED",
        "terminal": "GENERATED_NOT_EVALUATED",
        "generated": True,
        "profile": audit.profile_record("generated"),
        "arms": list(audit.ARMS),
        "correct_counts": counts.tolist(),
        "a0_correct_counts": a0_counts.tolist(),
        "query_count_per_cell": 48,
        "interpretation": "Generated completed-path rehearsal; not human metadata efficacy",
    }


def summarize(aggregate, profile="generated"):
    audit.require(profile == "generated", "no human summary authority")
    return generated_summary(aggregate)


def run(output, manifest_path, manifest_sha):
    began = time.perf_counter()
    output = Path(output).absolute()
    audit.require(output.resolve() == output and output.is_dir(), "unaliased output directory")
    failure_precedence(output)
    audit.require(not os.path.lexists(output / "cold_audit.json"), "cold audit append-only once")
    audit.require(
        output.name == "task-trca-n1-integration-primary1"
        and output.parent.name.startswith("task-trca-n1-integration-")
        and Path(manifest_path) == output.parent / "manifest.json",
        "canonical generated output/manifest before reads",
    )
    manifest, _ = read_pinned(manifest_path, manifest_sha)
    profile, contract = validate_manifest(manifest, manifest_sha)
    audit.require(manifest["output_root"] == str(output), "manifest exact output binding")
    result, result_binding = read_pinned(output / "result.json")
    ids, windows = tuple(contract["source_ids"]), tuple(contract["samples"])
    audit.require(
        result["status"] == "GENERATED_COMPLETE"
        and result["cold_audit_status"] == "COLD_AUDIT_PENDING"
        and result["manifest_sha256"] == manifest_sha
        and result["design_sha256"] == DESIGN_SHA
        and result["profile"] == contract
        and result["generated"] is contract["generated"],
        "pending completion provenance",
    )
    audit.require(
        result["held60_access"] is False and result["query_metadata_decoded"] is False,
        "source-only completion declaration",
    )
    artifacts = result["artifacts"]
    audit.require(
        sum(p.lstat().st_size for p in output.iterdir()) <= manifest["output_budget_bytes"],
        "completed output byte budget",
    )
    audit.require(
        {p.name for p in output.iterdir()} == set(artifact_names(ids).values()) | {"result.json"},
        "exact completed output files before numeric decode",
    )
    bind_artifacts(output, artifacts, ids)
    start = load_artifact(artifacts, "start")
    audit.require(
        start["status"] == "STARTED"
        and start["manifest"] == manifest
        and start["manifest_sha256"] == manifest_sha
        and start["source_revision"] == result["source_revision"]
        and start["profile"] == contract
        and start["generated"] is contract["generated"]
        and start["held60_access"] is False
        and start["old_attempts_reopened"] is False,
        "externally pinned manifest/start/result identity",
    )
    models = [load_artifact(artifacts, f"model{fold}") for fold in range(3)]
    freeze = load_artifact(artifacts, "freeze")
    check_freeze(freeze, artifacts, models, manifest_sha, contract)
    access = load_artifact(artifacts, "access", "jsonl")
    counts = check_access(
        access,
        contract,
        artifacts["freeze"]["sha256"],
        manifest_sha,
    )
    event_receipt = check_events(
        load_artifact(artifacts, "events", "jsonl"),
        access,
        contract,
        artifacts,
        models,
        result,
        output,
    )
    expected_queries = len(ids) * 2 * len(windows) * 4
    audit.require(
        result["state"]["query_access_count"] == expected_queries
        and result["state"]["models_frozen"] is True
        and all(
            type(result["state"][key]) is int and result["state"][key] == 24000
            for key in (
                "optimizer_updates_completed",
                "optimizer_updates_budgeted",
                "optimizer_updates_charged",
            )
        )
        and result["peak_cuda_allocated_bytes"] == 0,
        "completed runtime state",
    )
    aggregate = load_artifact(artifacts, "scores", "npz")
    expected_shapes = {
        "scores": (len(ids), 2, len(windows), 2, 10, 48, 12),
        "a0": (len(ids), 2, len(windows), 48, 12),
        "coverage_m": (len(ids), 2, 8, 2),
        "coverage_donor_m": (len(ids), 2, 8, 2),
        "coverage_available": (len(ids), 2, 8),
        "r_changes": (len(ids), 2, len(windows), 2),
        "projector_changes": (len(ids), 2, len(windows), 2),
        "j_changes": (len(ids), 2, len(windows), 2),
        "qm_coefficients": (3, 3),
        "weights": (2, 5),
        "a0_weights": (2, 5),
    }
    audit.require(set(aggregate) == set(expected_shapes), "exact aggregate arrays")
    for name, shape in expected_shapes.items():
        audit.require(
            aggregate[name].shape == shape and np.isfinite(aggregate[name]).all(),
            "finite complete aggregate geometry: " + name,
        )
    audit.require(aggregate["coverage_available"].dtype.kind == "b", "boolean coverage")
    np.testing.assert_array_equal(aggregate["weights"], manifest["native_weights"]["ETRCA"])
    np.testing.assert_array_equal(aggregate["a0_weights"], manifest["native_weights"]["A0_author"])
    training_receipts, evaluation_receipts = [], []
    for fold, model in enumerate(models):
        evaluation_ids = ids[fold::3]
        fitting = tuple(pid for pid in ids if pid not in evaluation_ids)
        source = load_artifact(artifacts, f"source{fold}", "npz")
        np.testing.assert_array_equal(source["weights"], aggregate["weights"][source["keys"][:, 1]])
        training_receipts.append(
            audit.audit_training(source, model, fitting, evaluation_ids, profile=profile)
        )
        audit.require(
            training_receipts[-1] == model["independent_source_audit"],
            "warm/cold source receipt equality",
        )
        del source
        parts = [load_artifact(artifacts, f"evaluation{pid}", "npz") for pid in evaluation_ids]
        for pid, part in zip(evaluation_ids, parts):
            audit._grid(part, (pid,), profile)
        data = audit.combine_evaluation(parts)
        del parts
        np.testing.assert_array_equal(data["weights"], aggregate["weights"][data["keys"][:, 1]])
        receipt = audit.audit_evaluation(data, model["pipeline"], evaluation_ids, profile=profile)
        evaluation_receipts.append(receipt)
        audit.require(
            receipt == load_artifact(artifacts, f"evaluation_audit{fold}"),
            "warm/cold evaluation receipt equality",
        )
        donors = audit.donor_positions(data, range(len(data["keys"])))
        for j, (pid, interface, n, k) in enumerate(data["keys"]):
            p, ni, ki = ids.index(int(pid)), windows.index(int(n)), (3, 5).index(int(k))
            np.testing.assert_array_equal(
                aggregate["scores"][p, interface, ni, ki], data["scores"][j]
            )
            a0 = np.einsum(
                "nbc,b->nc", data["a0_correlations"][j], aggregate["a0_weights"][interface]
            )
            np.testing.assert_array_equal(aggregate["a0"][p, interface, ni], a0)
            r, f = data["r"][j], data["projectors"][j]
            for name, value in (
                ("r_changes", np.max(np.abs(r[3] - r[1]))),
                ("projector_changes", np.max(np.abs(f[3] - f[1]))),
                ("j_changes", np.max(np.abs(f[3].sum(1) - f[1].sum(1)))),
            ):
                audit.require(
                    aggregate[name][p, interface, ni, ki] == value, "actuation binding: " + name
                )
            if k == 3:
                for name, value in (
                    ("coverage_m", data["m"][j]),
                    ("coverage_donor_m", data["m"][donors[j]]),
                    ("coverage_available", data["available"][j]),
                ):
                    np.testing.assert_array_equal(aggregate[name][p, interface], value)
        np.testing.assert_array_equal(
            aggregate["qm_coefficients"][fold], model["pipeline"]["residuals"]["QM"]["coefficients"]
        )
        del data
    summary = summarize(aggregate, profile)
    audit.require(
        summary == result["summary"], "exact independently reconstructed counts/summary/terminal"
    )
    failure_precedence(output)
    coverage = independent.learning_path_coverage(models)
    # Recheck every input binding after computation, including the unsigned result.
    read_pinned(
        result_binding["path"], result_binding["sha256"], result_binding["bytes"], kind="hash"
    )
    bind_artifacts(output, artifacts, ids)
    read_pinned(manifest_path, manifest_sha, kind="hash")
    for name in CODE_PATHS:
        read_pinned(ROOT / name, manifest["code_pins"][name], kind="hash", immutable=False)
    fixture_binding(manifest)
    return {
        "status": "GENERATED_COLD_INDEPENDENT_AUDIT_PASS",
        "terminal": summary["terminal"],
        "result_sha256": result_binding["sha256"],
        "manifest_sha256": manifest_sha,
        "manifest_path": str(Path(manifest_path).absolute()),
        "design_sha256": DESIGN_SHA,
        "fixture": manifest["fixture"],
        "learning_path_coverage": coverage,
        "metadata_effect": "NOT_EVALUATED",
        "code_pins": manifest["code_pins"],
        "profile": contract,
        "generated": contract["generated"],
        "source_audits": training_receipts,
        "evaluation_audits": evaluation_receipts,
        "exact_access_counts": counts,
        "event_barriers": event_receipt,
        "elapsed_seconds": time.perf_counter() - began,
        "gpu_used": False,
        "human_inputs": False,
        "held60_access": False,
        "raw_input_access": False,
        "generated_input_byte_hashes_verified": True,
        "source_numeric_decode": False,
        "limitations": [
            "saved Q/S/C evidence, not independent raw preprocessing/Q15/native fit/Adam replay",
            "workflow provenance, not an OS sandbox",
            "development, not independent confirmation",
        ],
    }


def publish(path, value):
    with path.open("x") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    path.chmod(0o400)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--manifest-sha256", required=True)
    args = parser.parse_args(argv)
    output = args.output.absolute()
    already_cold = os.path.lexists(output / "cold_audit.json")

    def expired(*_):
        raise TimeoutError("Frozen independent cold audit1800s budget exhausted")

    previous = signal.signal(signal.SIGALRM, expired)
    signal.alarm(AUDIT_SECONDS)
    try:
        audit.require(
            all(
                os.environ.get(name) == "1"
                for name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS")
            )
            and os.environ.get("PYTHONDONTWRITEBYTECODE") == "1",
            "CPU1/no-bytecode launch required",
        )
        receipt = run(output, args.manifest, args.manifest_sha256)
        failure_precedence(output)
        publish(output / "cold_audit.json", receipt)
    except Exception as exc:
        # Never overwrite earlier evidence or create a second contradictory failure.
        if (
            output.resolve() == output
            and output.is_dir()
            and not os.path.lexists(output / "cold_audit_failure.json")
            and not already_cold
        ):
            publish(
                output / "cold_audit_failure.json",
                {
                    "status": "VALIDITY_FAILURE",
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                    "manifest_sha256": args.manifest_sha256,
                    "efficacy": "NOT_EVALUATED",
                },
            )
        raise
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, previous)
    print(json.dumps(receipt, indent=2), flush=True)


if __name__ == "__main__":
    main()
