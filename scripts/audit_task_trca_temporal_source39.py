"""Independent cold audit of a manifest-bound completed temporal attempt.

Reads saved artifacts/configuration/code only: no source archive, projection,
producer, optimizer, raw EEG or human metadata decoder imports.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import stat
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from cfeg.analysis import task_trca_temporal_artifact_audit as audit
from cfeg.analysis import task_trca_temporal_audit as independent

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
DESIGN_SHA = "e456c3bcfc066e6ba3cb95a9c76798a006416bee11d064d083d9bd98ce7b97da"
OLD_PLAN_SHA = "ed67cc1f361ed89c37b6f7c5df1e9170b78484e428a5105020485c8c03e3934b"
NATIVE_MANIFEST_SHA = "15c088048f3435a500d3758138c3cd6a1616e869d568bdf68d10bcc5e978d83a"
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


def read_pinned(path, expected_sha=None, expected_bytes=None, *, kind="json"):
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


def validate_manifest(manifest, expected_sha):
    schema = manifest.get("schema")
    generated = schema == "cfeg.task_trca_temporal.source39_generated.v1"
    profile = "generated" if generated else "human"
    audit.require(
        schema
        == (
            "cfeg.task_trca_temporal.source39_generated.v1"
            if generated
            else "cfeg.task_trca_temporal.source39_execution.v1"
        ),
        "new temporal executable schema",
    )
    audit.require(
        manifest.get("status") == ("GENERATED_FROZEN" if generated else "EXECUTION_FROZEN"),
        "frozen manifest status",
    )
    contract = audit.profile_record(profile)
    audit.require(
        manifest.get("profile") == contract
        and manifest.get("source_ids") == contract["source_ids"],
        "exact externally frozen profile/source IDs",
    )
    audit.require(
        manifest.get("study_id") == "task-trca-temporal-v1"
        and manifest.get("score_schema") == independent.SCORE_SCHEMA
        and manifest.get("generated", False) is generated
        and manifest.get("backend") == "batch"
        and manifest.get("precision") == "float64",
        "fixed candidate/scorer/runtime",
    )
    audit.require(
        manifest.get("held60_authorized") is False
        and manifest.get("retired1_3_authorized") is False
        and manifest.get("old_attempts_reopened") is False,
        "no held/old-attempt authority",
    )
    audit.require(manifest["design"]["sha256"] == DESIGN_SHA, "C1 scientific design pin")
    audit.require(
        manifest["design"]["path"]
        == str(ROOT / "configs/analysis/task_trca_temporal_v1_design.json")
        and manifest["program_path"]
        == str(ROOT / "configs/analysis/metadata_learning_program_v1.json"),
        "canonical tracked program/design configuration paths",
    )
    read_pinned(manifest["design"]["path"], DESIGN_SHA)
    if generated:
        audit.require(
            manifest["native_plan"]["path"]
            == str(Path(manifest["native_root"]) / "native_plan.json"),
            "generated native-plan basename/root binding",
        )
    else:
        audit.require(
            manifest["native_plan"]["sha256"] == OLD_PLAN_SHA
            and manifest["native_plan"]["path"]
            == str(ROOT / "configs/analysis/metadata_prior_source39_v1.json"),
            "canonical native compatibility pin before decoding",
        )
        audit.require(
            manifest["native_manifest_sha256"] == NATIVE_MANIFEST_SHA,
            "fixed historical native input-manifest pin",
        )
    native_plan, _ = read_pinned(manifest["native_plan"]["path"], manifest["native_plan"]["sha256"])
    if generated:
        audit.require(
            native_plan.get("generated") is True
            and native_plan.get("seed") == 20260909
            and "GENERATED" in native_plan["source_projection"]["envelope_provenance"]["study_id"],
            "explicit generated native/projection origin",
        )
        audit.require(
            manifest.get("preflight") is None, "generated preflight is not human authority"
        )
    audit.require(native_plan["interfaces"] == ["dry", "wet"], "native interface order")
    for name in ("ETRCA", "A0_author"):
        expected_weights = [
            native_plan["native_weights"][name][interface] for interface in ("dry", "wet")
        ]
        np.testing.assert_array_equal(manifest["native_weights"][name], expected_weights)
    program, _ = read_pinned(manifest["program_path"], manifest["program_sha256"])
    audit.require(
        program["schema"] == "cfeg.metadata_learning_program.v1"
        and program["program_id"] == "metadata-learning-program-v1",
        "fixed program",
    )
    authority = program["authority"]
    audit.require(
        authority["source39_development_authorized_after_candidate_preflight"] is True
        and authority["implementation_and_generated_validation_authorized"] is True
        and authority["held60_authorized"] is False
        and authority["retired_s1_s3_authorized"] is False
        and authority["old_terminal_candidates_reopened"] is False,
        "program authority",
    )
    c1 = [candidate for candidate in program["candidates"] if candidate["slot"] == "C1"]
    audit.require(
        len(c1) == 1
        and c1[0]["id"] == "task-trca-temporal-v1"
        and c1[0]["design_sha256"] == DESIGN_SHA,
        "program C1 design binding",
    )
    c2 = [candidate for candidate in program["candidates"] if candidate["slot"] == "C2"]
    audit.require(
        len(c2) == 1
        and c2[0]["id"] == "task-trca-pair-s-v1"
        and isinstance(c2[0].get("design_path"), str)
        and not Path(c2[0]["design_path"]).is_absolute()
        and ".." not in Path(c2[0]["design_path"]).parts,
        "C2 specification fixed before C1",
    )
    read_pinned(ROOT / c2[0]["design_path"], c2[0]["design_sha256"])
    budget = program["budget"]
    audit.require(
        manifest["cpu_threads"] == budget["cpu_threads"] == 1
        and 0 < manifest["max_seconds"] <= budget["human_seconds_per_attempt_max"]
        and 0 < manifest["output_budget_bytes"] <= budget["output_bytes_per_attempt_max"]
        and manifest["optimizer_updates_max"]
        == budget["optimizer_updates_per_primary_max"]
        == 24000
        and 0 < manifest["gpu_memory_fraction"] <= budget["gpu_memory_fraction_max"] <= 0.16,
        "fixed program runtime bounds",
    )
    pins = manifest["code_pins"]
    audit.require(
        isinstance(pins, dict) and set(pins) == set(CODE_PATHS),
        "exact mandatory imported code-pin set",
    )
    for name in CODE_PATHS:
        read_pinned(ROOT / name, pins[name], kind="hash")
    if not generated:
        preflight = manifest["preflight"]
        audit.require(
            set(preflight) == {"path", "sha256", "bytes"}, "generated preflight descriptor"
        )
        preflight_root = Path(preflight["path"]).absolute().parent
        audit.require(
            Path(preflight["path"]).name == "cold_audit.json", "canonical generated cold receipt"
        )
        failure_precedence(preflight_root)
        receipt, _ = read_pinned(preflight["path"], preflight["sha256"], preflight["bytes"])
        audit.require(
            receipt["status"] == "GENERATED_COLD_INDEPENDENT_AUDIT_PASS"
            and receipt["generated"] is True
            and receipt["profile"] == audit.profile_record("generated")
            and receipt["code_pins"] == pins
            and receipt["program_sha256"] == manifest["program_sha256"],
            "same-code generated completed-path prerequisite",
        )
        generated_manifest, _ = read_pinned(receipt["manifest_path"], receipt["manifest_sha256"])
        generated_result, _ = read_pinned(preflight_root / "result.json", receipt["result_sha256"])
        audit.require(
            generated_manifest["schema"] == "cfeg.task_trca_temporal.source39_generated.v1"
            and generated_manifest["status"] == "GENERATED_FROZEN"
            and generated_manifest["profile"] == audit.profile_record("generated")
            and generated_manifest["program_sha256"] == manifest["program_sha256"]
            and generated_manifest["code_pins"] == pins
            and generated_manifest["output_root"] == str(preflight_root)
            and generated_result["status"] == "GENERATED_COMPLETE"
            and generated_result["manifest_sha256"] == receipt["manifest_sha256"]
            and generated_result["program_sha256"] == manifest["program_sha256"]
            and generated_result["profile"] == audit.profile_record("generated")
            and generated_result["generated"] is True,
            "generated preflight manifest/result/program binding",
        )
        failure_precedence(preflight_root)
    audit.require(
        isinstance(expected_sha, str) and SHA_PATTERN.fullmatch(expected_sha),
        "manifest trust-root hash",
    )
    return profile, contract


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
        freeze["schema"] == "cfeg.task_trca_temporal.all_models_frozen.v1"
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
        "terminal": "GENERATED_NOT_EVALUATED",
        "generated": True,
        "profile": audit.profile_record("generated"),
        "arms": list(audit.ARMS),
        "correct_counts": counts.tolist(),
        "a0_correct_counts": a0_counts.tolist(),
        "query_count_per_cell": 48,
        "interpretation": "Generated completed-path rehearsal; not human metadata efficacy",
    }


def summarize(aggregate, profile):
    if profile == "generated":
        return generated_summary(aggregate)
    return independent.summarize(
        aggregate["scores"],
        aggregate["a0"],
        {
            "m": aggregate["coverage_m"],
            "donor_m": aggregate["coverage_donor_m"],
            "available": aggregate["coverage_available"],
        },
        {
            "band_weights": aggregate["weights"],
            "a0_band_weights": aggregate["a0_weights"],
            "qm_coefficients": aggregate["qm_coefficients"],
            "r_max_abs_qm_minus_q": aggregate["r_changes"],
            "projector_max_abs_qm_minus_q": aggregate["projector_changes"],
            "j_max_abs_qm_minus_q": aggregate["j_changes"],
        },
    )


def run(output, manifest_path, manifest_sha):
    began = time.perf_counter()
    output = Path(output).absolute()
    audit.require(output.resolve() == output and output.is_dir(), "unaliased output directory")
    failure_precedence(output)
    audit.require(not os.path.lexists(output / "cold_audit.json"), "cold audit append-only once")
    manifest, _ = read_pinned(manifest_path, manifest_sha)
    profile, contract = validate_manifest(manifest, manifest_sha)
    audit.require(manifest["output_root"] == str(output), "manifest exact output binding")
    result, result_binding = read_pinned(output / "result.json")
    ids, windows = tuple(contract["source_ids"]), tuple(contract["samples"])
    audit.require(
        result["status"] == ("GENERATED_COMPLETE" if profile == "generated" else "COMPLETE")
        and result["cold_audit_status"] == "COLD_AUDIT_PENDING"
        and result["manifest_sha256"] == manifest_sha
        and result["program_sha256"] == manifest["program_sha256"]
        and result["profile"] == contract
        and result["generated"] is contract["generated"],
        "pending completion provenance",
    )
    audit.require(
        result["held60_access"] is False and result["query_metadata_decoded"] is False,
        "source-only completion declaration",
    )
    artifacts = result["artifacts"]
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
        and result["state"]["models_frozen"] is True,
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
    # Recheck every input binding after computation, including the unsigned result.
    read_pinned(
        result_binding["path"], result_binding["sha256"], result_binding["bytes"], kind="hash"
    )
    bind_artifacts(output, artifacts, ids)
    read_pinned(manifest_path, manifest_sha, kind="hash")
    return {
        "status": "GENERATED_COLD_INDEPENDENT_AUDIT_PASS"
        if contract["generated"]
        else "COLD_INDEPENDENT_AUDIT_PASS",
        "terminal": summary["terminal"],
        "result_sha256": result_binding["sha256"],
        "manifest_sha256": manifest_sha,
        "manifest_path": str(Path(manifest_path).absolute()),
        "program_sha256": manifest["program_sha256"],
        "code_pins": manifest["code_pins"],
        "profile": contract,
        "generated": contract["generated"],
        "source_audits": training_receipts,
        "evaluation_audits": evaluation_receipts,
        "exact_access_counts": counts,
        "event_barriers": event_receipt,
        "elapsed_seconds": time.perf_counter() - began,
        "held60_access": False,
        "raw_input_access": False,
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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--manifest-sha256", required=True)
    args = parser.parse_args()
    output = args.output.absolute()
    already_cold = os.path.lexists(output / "cold_audit.json")
    try:
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
    print(json.dumps(receipt, indent=2), flush=True)


if __name__ == "__main__":
    main()
