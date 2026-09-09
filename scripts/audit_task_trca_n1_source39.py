"""Independent terminal audit of one new manifest-bound N1 source39 attempt.

Reads authorized source archives/projection only for byte hashes, then completed
saved statistics for numerical replay. No producer, optimizer or source decoder imports.
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
from cfeg.analysis import task_trca_n1_audit as independent
from cfeg.analysis import task_trca_n1_source39_artifact_audit as audit
from cfeg.analysis import task_trca_n1_source39_audit as endpoint

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
DESIGN_SHA = "bb90cc361a28c3e0a53861564ec007f38602cf5d16554e0923a6550eb1be7c95"
DESIGN_PATH = "configs/analysis/task_trca_n1_source39_v1.json"
INPUT_LIMIT = 12 * 1024**3
GENERATED_LIMIT = 2 * 1024**3
STUDY_ID = endpoint.STUDY_ID
AUDIT_SECONDS = 1800
SHA_PATTERN = re.compile(r"[0-9a-f]{64}\Z")
QUERY_KINDS = ("query", "a0", "full_k3", "full_k5")
OUTPUT_PARENT_ROOT = Path("/home/whwovy")


def scoped_output(path):
    return (
        path.is_absolute()
        and path.resolve() == path
        and path.parent.parent == OUTPUT_PARENT_ROOT
        and path.parent.name.startswith("task-trca-n1-source39-")
        and path.parent.name != "task-trca-n1-source39-"
        and path.name in ("task-trca-n1-source39-primary1", "task-trca-n1-source39-generated1")
    )


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


def read_pinned(
    path, expected_sha=None, expected_bytes=None, *, kind="json", immutable=True, allow_empty=False
):
    """One regular single-link descriptor, hash before/after parsing and path identity."""
    path = Path(path).absolute()
    audit.require(path.resolve() == path and not path.is_symlink(), "unaliased pinned path")
    if expected_sha is not None:
        audit.require(
            isinstance(expected_sha, str) and SHA_PATTERN.fullmatch(expected_sha), "expected SHA256"
        )
    if expected_bytes is not None:
        audit.require(
            type(expected_bytes) is int and expected_bytes >= (0 if allow_empty else 1),
            "positive byte binding",
        )
    with os.fdopen(os.open(path, os.O_RDONLY | os.O_NOFOLLOW), "rb") as stream:
        before = os.fstat(stream.fileno())
        audit.require(
            stat.S_ISREG(before.st_mode) and before.st_nlink == 1, "single-link regular pinned file"
        )
        audit.require(not immutable or stat.S_IMODE(before.st_mode) == 0o400, "immutable0400 input")
        audit.require(
            (0 if allow_empty else 1) <= before.st_size <= INPUT_LIMIT, "bounded input bytes"
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
    for name in (
        "failure.json",
        "cold_audit_failure.json",
        "terminal_audit.json",
        "terminal_audit_failure.json",
    ):
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
        *(f"S{pid:03d}.npz" for pid in audit.profile_record("generated")["source_ids"]),
    }
    audit.require(
        {p.name for p in native_root.iterdir()} == expected_names, "exact14 generated input files"
    )
    audit.require(
        sum(p.lstat().st_size for p in native_root.iterdir()) + manifest["output_budget_bytes"]
        <= GENERATED_LIMIT,
        "combined generated input/run byte budget",
    )
    fixture, _ = read_pinned(fixture_desc["path"], fixture_desc["sha256"], fixture_desc["bytes"])
    audit.require(
        fixture.get("status") == "GENERATED_INPUTS_ONLY"
        and type(fixture.get("seed")) is int
        and fixture["seed"] == 20260914
        and fixture.get("human_artifact_reads") is False
        and fixture.get("metadata_effect") == "NOT_EVALUATED",
        "generated fixture origin",
    )
    strict_equal(fixture["profile"], audit.profile_record("generated"), "fixture exact profile")
    pins = manifest["code_pins"]
    for field, name in (
        ("generator_sha256", "scripts/prepare_task_trca_n1_source39.py"),
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
        and plan["seed"] == 20260914
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
        and endpoint.STUDY_ID in origin["study_id"],
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
        and native["seed"] == 20260914
        and native["plan_sha256"] == fixture["plan"]["sha256"],
        "generated native receipt",
    )
    strict_equal(
        native["source_subject_ids"],
        audit.profile_record("generated")["source_ids"],
        "nine EEG IDs",
    )
    start, _ = read_pinned(native_root / "start.json", native["start_sha256"])
    audit.require(
        start.get("generated") is True
        and type(start.get("seed")) is int
        and start["seed"] == 20260914,
        "generated native start",
    )
    strict_equal(start["profile"], audit.profile_record("generated"), "native start profile")
    audit.require(
        isinstance(native["files"], list) and len(native["files"]) == 9,
        "nine native source descriptors",
    )
    for pid, item in zip(audit.profile_record("generated")["source_ids"], native["files"]):
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


def _bound_read(item, path, *, kind="json"):
    descriptor(item, path)
    return read_pinned(item["path"], item["sha256"], item["bytes"], kind=kind)[0]


def prior_binding(design, pins):
    """Previous generated N1 proof is immutable evidence, never execution authority."""
    item = design["prerequisite"]
    path = Path(item["path"])
    failure_precedence(path.parent)
    receipt, _ = read_pinned(path, item["sha256"])
    audit.require(
        receipt["status"] == "GENERATED_COLD_INDEPENDENT_AUDIT_PASS"
        and receipt["generated"] is True,
        "previous N1 proof",
    )
    for name in set(receipt["code_pins"]) & set(pins):
        audit.require(receipt["code_pins"][name] == pins[name], "previous shared N1 code unchanged")
    previous_path = path.parent.parent / "manifest.json"
    audit.require(receipt["manifest_path"] == str(previous_path), "previous manifest path")
    previous, _ = read_pinned(previous_path, receipt["manifest_sha256"])
    result, _ = read_pinned(path.parent / "result.json", receipt["result_sha256"])
    audit.require(
        previous["study_id"] == independent.SCHEMA
        and previous["output_root"] == str(path.parent)
        and previous["code_pins"] == receipt["code_pins"]
        and result["status"] == "GENERATED_COMPLETE"
        and result["manifest_sha256"] == receipt["manifest_sha256"],
        "previous generated proof binding",
    )
    failure_precedence(path.parent)


def resource_binding(manifest):
    item = manifest["resource_preflight"]
    audit.require(isinstance(item, dict) and "path" in item, "resource descriptor required")
    path = Path(item["path"])
    root = path.parent
    audit.require(
        path.is_absolute()
        and path.resolve() == path
        and path.name == "resource.json"
        and root.name == "resource1"
        and root.parent.parent == OUTPUT_PARENT_ROOT
        and root.parent.name.startswith("task-trca-n1-source39-"),
        "canonical resource prerequisite",
    )
    failure_precedence(root)
    receipt = _bound_read(item, path)
    audit.require(
        receipt["status"] == "GENERATED_RESOURCE_PASS"
        and receipt["study_id"] == STUDY_ID
        and receipt["design_sha256"] == DESIGN_SHA
        and receipt["code_pins"] == manifest["code_pins"]
        and type(receipt["seed"]) is int
        and receipt["seed"] == 20260915
        and type(receipt["cases"]) is int
        and receipt["cases"] == 416
        and type(receipt["updates_completed"]) is int
        and receipt["updates_completed"] == 800
        and receipt["arrays_generated"] is True
        and receipt["human_reads"] is False
        and receipt["gpu_used"] is False,
        "same-code generated resource prerequisite",
    )
    fit_ids = [p for j, p in enumerate(audit.SOURCE_IDS) if j % 3]
    strict_equal(receipt["fit_ids"], fit_ids, "resource exact26 generated fit IDs")
    for name in ("preparation_seconds", "fit_seconds", "elapsed_seconds", "projected_seconds"):
        audit.require(
            type(receipt[name]) in (int, float)
            and np.isfinite(receipt[name])
            and receipt[name] >= 0,
            "finite resource timing",
        )
    expected = 36 * receipt["fit_seconds"] + 3 * receipt["preparation_seconds"] + 600
    audit.require(
        receipt["projected_seconds"] == expected <= 21600
        and receipt["elapsed_seconds"] <= 1200
        and type(receipt["peak_rss_bytes"]) is int
        and 0 < receipt["peak_rss_bytes"] <= 4 * 1024**3,
        "frozen resource screening inequalities",
    )
    start = _bound_read(receipt["start"], root / "start.json")
    pipeline = _bound_read(receipt["pipeline"], root / "pipeline.json")
    independent._schema(pipeline)
    audit.require(
        pipeline["fit_ids"] == fit_ids
        and pipeline["lambda"] == 0.001
        and set(pipeline["residuals"]) == {"Q2", "QM", "SHAM_REFIT"},
        "resource pipeline identity",
    )
    for name, head in (("Q", pipeline["Q"]), *pipeline["residuals"].items()):
        coefficient = np.asarray(head["coefficients"], dtype=np.float64)
        trace = head["trace"]
        audit.require(
            coefficient.shape == (16 if name == "Q" else 3,)
            and np.isfinite(coefficient).all()
            and type(head["steps"]) is int
            and head["steps"] == 200
            and len(trace) == 200
            and all(type(r["step"]) is int for r in trace)
            and [r["step"] for r in trace] == list(range(1, 201)),
            "resource completed head trace",
        )
        values = np.asarray(
            [
                [r[k] for k in ("loss_before_step", "ce_before_step", "gradient_norm")]
                for r in trace
            ],
            dtype=np.float64,
        )
        audit.require(
            np.isfinite(values).all() and np.all(values[:, 2] >= 0), "resource finite trace"
        )
    audit.require(
        start["study_id"] == STUDY_ID
        and start["design_sha256"] == DESIGN_SHA
        and start["code_pins"] == manifest["code_pins"]
        and start["seed"] == 20260915
        and start["source_revision"] == receipt["source_revision"],
        "resource start binding",
    )
    audit.require(
        {p.name for p in root.iterdir()} == {"start.json", "pipeline.json", "resource.json"},
        "exact successful resource files",
    )
    failure_precedence(root)
    return receipt


def generated_preflight_binding(manifest):
    item = manifest["preflight"]
    audit.require(isinstance(item, dict) and "path" in item, "new generated preflight required")
    path = Path(item["path"])
    output = path.parent
    audit.require(
        path.is_absolute()
        and path.resolve() == path
        and path.name == "cold_audit.json"
        and output.name == "task-trca-n1-source39-generated1"
        and output.parent.parent == OUTPUT_PARENT_ROOT
        and output.parent.name.startswith("task-trca-n1-source39-"),
        "new generated cold exact path",
    )
    failure_precedence(output)
    receipt = _bound_read(item, path)
    profile = audit.profile_record("generated")
    audit.require(
        receipt["status"] == "GENERATED_COLD_INDEPENDENT_AUDIT_PASS"
        and receipt["study_id"] == STUDY_ID
        and receipt["algorithm_schema"] == independent.SCHEMA
        and receipt["design_sha256"] == DESIGN_SHA
        and receipt["code_pins"] == manifest["code_pins"]
        and receipt["generated"] is True
        and receipt["profile"] == profile
        and receipt["learning_path_coverage"]["exercised"] is True,
        "same-code new generated cold prerequisite",
    )
    generated_path = output.parent / "generated_manifest.json"
    audit.require(receipt["manifest_path"] == str(generated_path), "generated manifest exact path")
    generated, _ = read_pinned(generated_path, receipt["manifest_sha256"])
    audit.require(
        generated["schema"] == "cfeg.task_trca_n1_source39.generated.v1"
        and generated["study_id"] == STUDY_ID
        and generated["status"] == "GENERATED_FROZEN"
        and generated["algorithm_schema"] == independent.SCHEMA
        and generated["design"] == manifest["design"]
        and generated["profile"] == profile
        and generated["generated"] is True
        and generated["seed"] == 20260914
        and generated["output_root"] == str(output)
        and generated["code_pins"] == manifest["code_pins"]
        and generated["resource_preflight"] == manifest["resource_preflight"],
        "generated manifest/design/resource/code binding",
    )
    result, _ = read_pinned(output / "result.json", receipt["result_sha256"])
    audit.require(
        result["status"] == "GENERATED_COMPLETE"
        and result["study_id"] == STUDY_ID
        and result["algorithm_schema"] == independent.SCHEMA
        and result["manifest_sha256"] == receipt["manifest_sha256"]
        and result["design_sha256"] == DESIGN_SHA
        and result["profile"] == profile
        and result["generated"] is True
        and result["summary"]["terminal"] == "GENERATED_NOT_EVALUATED",
        "generated result binding",
    )
    failure_precedence(output)
    return receipt


def input_binding(manifest):
    if manifest["generated"]:
        return fixture_binding(manifest)
    design, _ = read_pinned(ROOT / DESIGN_PATH, DESIGN_SHA, immutable=False)
    inputs = design["human_inputs"]
    root = Path(inputs["native_root"])
    native, _ = read_pinned(root / "result.json", inputs["native_manifest_sha256"])
    audit.require(
        native["status"] == "COMPLETE"
        and native["plan_sha256"] == inputs["native_plan_sha256"]
        and native["source_subject_ids"] == list(audit.SOURCE_IDS)
        and native["sample_counts"] == list(audit.SAMPLES)
        and isinstance(native["files"], list)
        and len(native["files"]) == 39,
        "fixed human native result",
    )
    for pid, row in zip(audit.SOURCE_IDS, native["files"]):
        audit.require(
            set(row) == {"subject", "filename", "sha256", "bytes", "raw_receipt"}
            and type(row["subject"]) is int
            and row["subject"] == pid
            and row["filename"] == f"S{pid:03d}.npz",
            "native descriptor identity",
        )
        raw = row["raw_receipt"]
        audit.require(
            set(raw) == {"subject", "path", "sha256", "bytes", "stored_dtype", "shape"}
            and type(raw["subject"]) is int
            and raw["subject"] == pid
            and isinstance(raw["path"], str)
            and Path(raw["path"]).is_absolute()
            and Path(raw["path"]).name == f"S{pid:03d}.mat"
            and isinstance(raw["sha256"], str)
            and SHA_PATTERN.fullmatch(raw["sha256"])
            and type(raw["bytes"]) is int
            and raw["bytes"] > 0
            and isinstance(raw["stored_dtype"], str)
            and bool(raw["stored_dtype"])
            and raw["shape"] == [8, 710, 2, 10, 12],
            "pinned raw lineage; raw path never opened",
        )
        item = {"path": str(root / row["filename"]), "sha256": row["sha256"], "bytes": row["bytes"]}
        _bound_read(item, root / f"S{pid:03d}.npz", kind="hash")
    read_pinned(inputs["projection_path"], inputs["projection_sha256"], kind="hash")
    plan, _ = read_pinned(
        ROOT / "configs/analysis/metadata_prior_source39_v1.json",
        inputs["native_plan_sha256"],
        immutable=False,
    )
    audit.require(
        plan["source_projection"]["path"] == inputs["projection_path"]
        and plan["source_projection"]["sha256"] == inputs["projection_sha256"],
        "source projection plan binding",
    )
    for name in ("ETRCA", "A0_author"):
        strict_equal(
            manifest["native_weights"][name],
            [plan["native_weights"][name][i] for i in ("dry", "wet")],
            "genuine frozen native weights",
        )
    return native


def validate_manifest(manifest, expected_sha, *, failure_only=False, bootstrap=False):
    audit.require(not bootstrap or failure_only, "bootstrap validation is failure-only")
    audit.require(
        isinstance(expected_sha, str) and SHA_PATTERN.fullmatch(expected_sha),
        "manifest trust-root hash",
    )
    keys = {
        "schema",
        "study_id",
        "algorithm_schema",
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
        "resource_preflight",
        "fixture",
    }
    audit.require(isinstance(manifest, dict) and set(manifest) == keys, "exact manifest fields")
    generated = manifest["schema"] == "cfeg.task_trca_n1_source39.generated.v1"
    profile = "generated" if generated else "human"
    contract = audit.profile_record(profile)
    audit.require(
        manifest["schema"]
        == (
            "cfeg.task_trca_n1_source39.generated.v1"
            if generated
            else "cfeg.task_trca_n1_source39.execution.v1"
        )
        and manifest["status"] == ("GENERATED_FROZEN" if generated else "EXECUTION_FROZEN")
        and manifest["generated"] is generated
        and manifest["study_id"] == STUDY_ID
        and manifest["algorithm_schema"] == independent.SCHEMA
        and manifest["score_schema"] == independent.SCORE_SCHEMA
        and manifest["device"] == "cpu"
        and manifest["backend"] == "batch"
        and manifest["precision"] == "float64",
        "new experiment/unchanged algorithm/runtime",
    )
    strict_equal(manifest["profile"], contract, "exact frozen profile")
    strict_equal(manifest["source_ids"], contract["source_ids"], "exact source IDs")
    strict_equal(manifest["seed"], contract["seed"], "fixed profile seed")
    for key in ("held60_authorized", "retired1_3_authorized", "old_attempts_reopened"):
        audit.require(manifest[key] is False, "no held/retired/old-attempt authority")
    output, native_root = Path(manifest["output_root"]), Path(manifest["native_root"])
    basename = "task-trca-n1-source39-generated1" if generated else "task-trca-n1-source39-primary1"
    audit.require(
        scoped_output(output) and output.name == basename and manifest["attempt_id"] == basename,
        "canonical new output path",
    )
    if not failure_only:
        failure_precedence(output)
    strict_equal(
        manifest["design"],
        {"path": str(ROOT / DESIGN_PATH), "sha256": DESIGN_SHA},
        "new frozen design binding",
    )
    pins = manifest["code_pins"]
    audit.require(isinstance(pins, dict) and set(pins) == set(CODE_PATHS), "exact36 code-pin set")
    for name in CODE_PATHS:
        read_pinned(ROOT / name, pins[name], kind="hash", immutable=False)
    audit.require(pins[DESIGN_PATH] == DESIGN_SHA, "frozen design code pin")
    design, _ = read_pinned(ROOT / DESIGN_PATH, DESIGN_SHA, immutable=False)
    audit.require(
        design["study_id"] == STUDY_ID
        and design["candidate"]["algorithm_schema"] == independent.SCHEMA,
        "new design authority",
    )
    strict_equal(design[profile + "_profile"], contract, "design profile")
    authority = design["authority"]
    audit.require(
        authority["implementation"] is True
        and authority["generated_validation"] is True
        and authority["source39_after_prerequisites"] is True
        and all(
            authority[k] is False
            for k in (
                "old_candidates_reopened",
                "held60",
                "retired1_3",
                "raw_mat_or_full_metadata",
                "outreach",
                "paid",
                "gpu",
            )
        ),
        "new bounded authority",
    )
    for key, low, high in (
        ("cpu_threads", 1, 1),
        ("max_seconds", 1, 1800 if generated else 21600),
        ("output_budget_bytes", 1, GENERATED_LIMIT if generated else INPUT_LIMIT),
        ("optimizer_updates_max", 24000, 24000),
    ):
        audit.require(
            type(manifest[key]) is int and low <= manifest[key] <= high,
            "fixed runtime budget: " + key,
        )
    if generated:
        audit.require(
            native_root == output.parent / "inputs"
            and native_root.resolve() == native_root
            and native_root.is_dir()
            and manifest["preflight"] is None,
            "generated sibling input paths",
        )
        audit.require(
            set(manifest["native_plan"]) == {"path", "sha256"}
            and manifest["native_plan"]["path"] == str(native_root / "native_plan.json"),
            "generated native plan before reads",
        )
    else:
        inputs = design["human_inputs"]
        strict_equal(
            manifest["native_plan"],
            {"path": inputs["native_plan_path"], "sha256": inputs["native_plan_sha256"]},
            "canonical human native plan before reads",
        )
        audit.require(
            manifest["native_root"] == inputs["native_root"]
            and manifest["native_manifest_sha256"] == inputs["native_manifest_sha256"]
            and manifest["fixture"] is None,
            "human native provenance before reads",
        )
    if bootstrap:
        # This verifies only scoped declaration/code authority. The terminal
        # auditor separately records prerequisite failures without opening data.
        return profile, contract
    # Every prerequisite precedes human provenance/source reads.
    prior_binding(design, pins)
    resource_binding(manifest)
    if not generated:
        generated_preflight_binding(manifest)
    input_binding(manifest)
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


def check_access(rows, contract, freeze_sha, manifest_sha, *, partial=False):
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
    if not partial:
        audit.require(
            barrier and observed == expected, "complete exact read-event Cartesian multiset"
        )
    return {f"{role}:{kind}": value for (role, kind), value in sorted(kind_counts.items())}


def check_events(rows, access, contract, artifacts, models, result, output, *, partial=False):
    """Cross-journal source freeze, evaluation completion and publication order."""
    ids = tuple(contract["source_ids"])
    barrier_index = next(
        (j for j, row in enumerate(access) if row["kind"] == "all_models_frozen"), len(access)
    )
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
        partial or (bool(rows) and rows[-1].get("event") == "attempt_verified_before_publication"),
        "required final publication event",
    )
    return {
        "event_count": len(rows),
        "access_count": len(access),
        "all_models_frozen_access_seq": barrier_index + 1,
        "publication_barrier_verified": bool(rows)
        and rows[-1].get("event") == "attempt_verified_before_publication",
    }


def check_freeze(freeze, artifacts, models, manifest_sha, contract):
    audit.require(
        set(freeze)
        == {"schema", "status", "source_ids", "manifest_sha256", "query_access_count", "models"},
        "exact freeze envelope",
    )
    audit.require(
        freeze["schema"] == "cfeg.task_trca_n1_source39.all_models_frozen.v1"
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


def summarize(aggregate, profile):
    if profile == "generated":
        return generated_summary(aggregate)
    audit.require(profile == "human", "fixed endpoint profile")
    return endpoint.summarize(
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
    audit.require(
        scoped_output(output)
        and Path(manifest_path)
        == output.parent
        / (
            "generated_manifest.json"
            if output.name.endswith("generated1")
            else "human_manifest.json"
        ),
        "canonical generated or human output/manifest before reads",
    )
    manifest, _ = read_pinned(manifest_path, manifest_sha)
    profile, contract = validate_manifest(manifest, manifest_sha)
    audit.require(manifest["output_root"] == str(output), "manifest exact output binding")
    result, result_binding = read_pinned(output / "result.json")
    ids, windows = tuple(contract["source_ids"]), tuple(contract["samples"])
    audit.require(
        result["status"] == ("GENERATED_COMPLETE" if contract["generated"] else "COMPLETE")
        and result["cold_audit_status"] == "COLD_AUDIT_PENDING"
        and result["manifest_sha256"] == manifest_sha
        and result["design_sha256"] == DESIGN_SHA
        and result["study_id"] == STUDY_ID
        and result["algorithm_schema"] == independent.SCHEMA
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
        len(access) == (325 if contract["generated"] else 4213)
        and event_receipt["event_count"] == (92 if contract["generated"] else 182),
        "complete frozen access/event row counts",
    )
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
    audit.require(
        type(result["elapsed_seconds"]) in (int, float)
        and np.isfinite(result["elapsed_seconds"])
        and 0 <= result["elapsed_seconds"] <= manifest["max_seconds"]
        and type(result["peak_rss_kib"]) is int
        and 0 < result["peak_rss_kib"] * 1024 <= 16 * 1024**3,
        "completed measured runtime/RSS bounds",
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
    coverage = endpoint.learning_path_coverage(models, profile=profile)
    # Recheck every input binding after computation, including the unsigned result.
    read_pinned(
        result_binding["path"], result_binding["sha256"], result_binding["bytes"], kind="hash"
    )
    bind_artifacts(output, artifacts, ids)
    read_pinned(manifest_path, manifest_sha, kind="hash")
    for name in CODE_PATHS:
        read_pinned(ROOT / name, manifest["code_pins"][name], kind="hash", immutable=False)
    input_binding(manifest)
    return {
        "status": "GENERATED_COLD_INDEPENDENT_AUDIT_PASS"
        if contract["generated"]
        else "COLD_INDEPENDENT_AUDIT_PASS",
        "terminal": summary["terminal"],
        "result_sha256": result_binding["sha256"],
        "manifest_sha256": manifest_sha,
        "manifest_path": str(Path(manifest_path).absolute()),
        "design_sha256": DESIGN_SHA,
        "fixture": manifest["fixture"],
        "learning_path_coverage": coverage,
        "metadata_effect": "NOT_EVALUATED" if contract["generated"] else summary["terminal"],
        "study_id": endpoint.STUDY_ID,
        "algorithm_schema": independent.SCHEMA,
        "code_pins": manifest["code_pins"],
        "profile": contract,
        "generated": contract["generated"],
        "source_audits": training_receipts,
        "evaluation_audits": evaluation_receipts,
        "exact_access_counts": counts,
        "event_barriers": event_receipt,
        "elapsed_seconds": time.perf_counter() - began,
        "gpu_used": False,
        "human_inputs": not contract["generated"],
        "held60_access": False,
        "raw_input_access": False,
        "source_input_byte_hashes_verified": True,
        "source_numeric_decode": False,
        "limitations": [
            "saved Q/S/C evidence, not independent raw preprocessing/Q15/native fit/Adam replay",
            "workflow provenance, not an OS sandbox",
            "development, not independent confirmation",
        ],
    }


def audit_bootstrap_failure(
    output, manifest_path, manifest_sha, manifest, contract, failure, failure_descriptor, began
):
    """Empty bootstrap provenance, explicitly not satisfied execution prerequisites."""
    state = failure["state"]
    audit.require(
        failure["partial_files"] == {}
        and failure["artifacts"] == {}
        and {p.name for p in output.iterdir()} == {"failure.json"}
        and failure["human_numeric_reads"] is False
        and state["models_frozen"] is False
        and all(
            type(state[name]) is int and state[name] == 0
            for name in (
                "optimizer_updates_completed",
                "optimizer_updates_charged",
                "optimizer_updates_budgeted",
                "query_access_count",
            )
        ),
        "bootstrap requires no start/journals/updates/query or human numeric reads",
    )
    design, _ = read_pinned(ROOT / DESIGN_PATH, DESIGN_SHA, immutable=False)
    checks = {}
    functions = [
        ("prior", lambda: prior_binding(design, manifest["code_pins"])),
        ("resource", lambda: resource_binding(manifest)),
    ]
    if not contract["generated"]:
        functions.append(("new_generated", lambda: generated_preflight_binding(manifest)))
    for name, check in functions:
        try:
            check()
            checks[name] = {"status": "VERIFIED"}
        except TimeoutError:
            raise  # A failed prerequisite must not swallow the whole-audit deadline.
        except (ValueError, KeyError, TypeError, OSError) as exc:
            checks[name] = {
                "status": "NOT_VERIFIED",
                "error_type": type(exc).__name__,
                "error": str(exc),
            }
    read_pinned(output / "failure.json", failure_descriptor["sha256"], failure_descriptor["bytes"])
    read_pinned(manifest_path, manifest_sha, kind="hash")
    for name in CODE_PATHS:
        read_pinned(ROOT / name, manifest["code_pins"][name], kind="hash", immutable=False)
    return {
        "status": "FIRST_FAILURE_PROVENANCE_AUDIT_PASS",
        "terminal": "VALIDITY_FAILURE",
        "study_id": STUDY_ID,
        "algorithm_schema": independent.SCHEMA,
        "design_sha256": DESIGN_SHA,
        "manifest_sha256": manifest_sha,
        "manifest_path": str(Path(manifest_path).absolute()),
        "failure": failure_descriptor,
        "profile": contract,
        "generated": contract["generated"],
        "code_pins": manifest["code_pins"],
        "scope": "BOOTSTRAP_AUTHORITY",
        "prerequisite_checks": checks,
        "execution_prerequisites_satisfied": False,
        "efficacy": "NOT_EVALUATED",
        "calibration": "NOT_EVALUATED",
        "numeric_failure_replayed": False,
        "source_numeric_decode": False,
        "human_numeric_reads": False,
        "partial_files": {},
        "state": state,
        "elapsed_seconds": time.perf_counter() - began,
        "held60_access": False,
        "raw_input_access": False,
        "limitations": [
            "bootstrap provenance only; does not certify satisfied execution prerequisites",
            "no start, journals, source arrays or failed numerical computation reconstructed",
            "zero numeric reads are the sealed producer declaration, not OS sandbox proof",
        ],
    }


def run_failure(output, manifest_path, manifest_sha):
    """Audit a first failed attempt without decoding any numeric artifact."""
    began = time.perf_counter()
    output = Path(output).absolute()
    audit.require(output.resolve() == output and output.is_dir(), "unaliased output directory")
    for name in (
        "cold_audit.json",
        "cold_audit_failure.json",
        "terminal_audit.json",
        "terminal_audit_failure.json",
    ):
        audit.require(not os.path.lexists(output / name), "one terminal audit mode, append-only")
    for name in ("prep_failure.json", "prepare_failure.json"):
        audit.require(not os.path.lexists(output.parent / name), "preparation failure precedence")
    audit.require(os.path.lexists(output / "failure.json"), "failure-only requires first failure")
    expected_name = (
        "generated_manifest.json"
        if output.name == "task-trca-n1-source39-generated1"
        else "human_manifest.json"
    )
    audit.require(
        scoped_output(output) and Path(manifest_path) == output.parent / expected_name,
        "failure output/manifest exact path",
    )
    manifest, _ = read_pinned(manifest_path, manifest_sha)
    failure, failure_descriptor = read_pinned(output / "failure.json")
    bootstrap = failure.get("state", {}).get("stage") == "BOOTSTRAP_AUTHORITY"
    _profile, contract = validate_manifest(
        manifest, manifest_sha, failure_only=True, bootstrap=bootstrap
    )
    audit.require(manifest["output_root"] == str(output), "failure output binding")
    audit.require(
        failure["status"] == "VALIDITY_FAILURE"
        and failure["study_id"] == STUDY_ID
        and failure["algorithm_schema"] == independent.SCHEMA
        and failure["design_sha256"] == DESIGN_SHA
        and failure["manifest_sha256"] == manifest_sha
        and failure["profile"] == contract
        and failure["generated"] is contract["generated"]
        and failure["held60_access"] is False
        and isinstance(failure["error_type"], str)
        and bool(failure["error_type"])
        and isinstance(failure["error"], str),
        "new experiment first-failure identity",
    )
    if bootstrap:
        return audit_bootstrap_failure(
            output,
            manifest_path,
            manifest_sha,
            manifest,
            contract,
            failure,
            failure_descriptor,
            began,
        )
    inventory = failure["partial_files"]
    names = artifact_names(contract["source_ids"])
    allowed = set(names.values()) | {"result.json"}
    audit.require(
        isinstance(inventory, dict)
        and set(inventory) <= allowed
        and {p.name for p in output.iterdir()} == set(inventory) | {"failure.json"},
        "exact actual partial file inventory",
    )

    def partial_read(name, kind="hash"):
        item = inventory[name]
        audit.require(
            isinstance(item, dict)
            and set(item) == {"path", "sha256", "bytes"}
            and item["path"] == str(output / name)
            and type(item["bytes"]) is int
            and item["bytes"] >= 0,
            "exact partial descriptor including empty files",
        )
        return read_pinned(
            item["path"], item["sha256"], item["bytes"], kind=kind, allow_empty=True
        )[0]

    for name in inventory:
        partial_read(name)
    artifacts = failure["artifacts"]
    audit.require(
        isinstance(artifacts, dict) and set(artifacts) <= set(names),
        "known completed partial artifacts",
    )
    for key, item in artifacts.items():
        audit.require(
            names[key] in inventory and item == inventory[names[key]],
            "completed/partial inventory binding",
        )
    audit.require("start" in artifacts, "bound start required for failure provenance")
    start = partial_read("start.json", "json")
    audit.require(
        start["status"] == "STARTED"
        and start["manifest"] == manifest
        and start["manifest_sha256"] == manifest_sha
        and start["profile"] == contract
        and start["generated"] is contract["generated"]
        and start["held60_access"] is False
        and start["old_attempts_reopened"] is False,
        "failure start identity",
    )
    state = failure["state"]
    for key in (
        "optimizer_updates_completed",
        "optimizer_updates_charged",
        "optimizer_updates_budgeted",
        "query_access_count",
    ):
        audit.require(type(state[key]) is int and state[key] >= 0, "integer failure counter")
    audit.require(
        state["optimizer_updates_completed"]
        <= state["optimizer_updates_charged"]
        == state["optimizer_updates_budgeted"]
        <= 24000
        and type(state["models_frozen"]) is bool
        and isinstance(state["stage"], str)
        and bool(state["stage"]),
        "bounded failure counters/stage",
    )
    models = [None, None, None]
    completed = []
    for fold in range(3):
        key = f"model{fold}"
        if key not in artifacts:
            continue
        audit.require(
            fold == len(completed) and f"source{fold}" in artifacts,
            "ordered completed source/model artifacts",
        )
        model = partial_read(names[key], "json")
        fitting = [p for p in contract["source_ids"] if p not in contract["source_ids"][fold::3]]
        independent._schema(model["pipeline"])
        independent._schema(model["selection"])
        audit.require(
            model["source_artifact"] == artifacts[f"source{fold}"]
            and model["manifest_sha256"] == manifest_sha
            and model["pipeline"]["fit_ids"] == fitting
            and model["selection"]["outer_evaluation_ids"] == contract["source_ids"][fold::3]
            and model["independent_source_audit"]["status"]
            == "TEMPORAL_SOURCE_SELECTION_AUDIT_PASS",
            "completed model source/role/manifest binding",
        )
        models[fold] = model
        completed.append(fold)
    freeze_sha = None
    if "freeze" in artifacts:
        audit.require(completed == [0, 1, 2], "freeze requires all completed models")
        freeze = partial_read(names["freeze"], "json")
        check_freeze(freeze, artifacts, models, manifest_sha, contract)
        freeze_sha = artifacts["freeze"]["sha256"]
    access = partial_read("access.jsonl", "jsonl") if "access.jsonl" in inventory else []
    events = partial_read("events.jsonl", "jsonl") if "events.jsonl" in inventory else []
    counts = check_access(access, contract, freeze_sha, manifest_sha, partial=True)
    queries = sum(counts.get("evaluation:" + kind, 0) for kind in QUERY_KINDS)
    barrier = any(row["kind"] == "all_models_frozen" for row in access)
    audit.require(
        queries == state["query_access_count"]
        and state["models_frozen"] is barrier
        and (not barrier or freeze_sha is not None),
        "partial read/query/freeze counter binding",
    )
    fake_result = {"summary": {"terminal": events[-1].get("terminal") if events else None}}
    event_receipt = check_events(
        events, access, contract, artifacts, models, fake_result, output, partial=True
    )
    for name in inventory:
        partial_read(name)
    read_pinned(output / "failure.json", failure_descriptor["sha256"], failure_descriptor["bytes"])
    read_pinned(manifest_path, manifest_sha, kind="hash")
    for name in CODE_PATHS:
        read_pinned(ROOT / name, manifest["code_pins"][name], kind="hash", immutable=False)
    return {
        "status": "FIRST_FAILURE_PROVENANCE_AUDIT_PASS",
        "terminal": "VALIDITY_FAILURE",
        "study_id": STUDY_ID,
        "algorithm_schema": independent.SCHEMA,
        "design_sha256": DESIGN_SHA,
        "manifest_sha256": manifest_sha,
        "manifest_path": str(Path(manifest_path).absolute()),
        "failure": failure_descriptor,
        "profile": contract,
        "generated": contract["generated"],
        "code_pins": manifest["code_pins"],
        "efficacy": "NOT_EVALUATED",
        "calibration": "NOT_EVALUATED",
        "numeric_failure_replayed": False,
        "source_numeric_decode": False,
        "partial_files": inventory,
        "completed_model_folds": completed,
        "exact_partial_access_counts": counts,
        "partial_events": event_receipt,
        "state": state,
        "elapsed_seconds": time.perf_counter() - began,
        "held60_access": False,
        "raw_input_access": False,
        "gpu_used": False,
        "limitations": [
            "failure/provenance audit only, not numerical cause validation",
            "no incomplete numeric decoding, raw preprocessing or Adam replay",
            "partial query requests do not establish completed scores or efficacy",
            "workflow provenance, not an OS sandbox",
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
    parser.add_argument("--failure-only", action="store_true")
    args = parser.parse_args(argv)
    output = args.output.absolute()
    success_name = "terminal_audit.json" if args.failure_only else "cold_audit.json"
    failure_name = "terminal_audit_failure.json" if args.failure_only else "cold_audit_failure.json"
    already_audited = any(
        os.path.lexists(output / name)
        for name in (
            "terminal_audit.json",
            "terminal_audit_failure.json",
            "cold_audit.json",
            "cold_audit_failure.json",
        )
    )

    def expired(*_):
        raise TimeoutError("Frozen independent terminal audit1800s budget exhausted")

    previous = signal.signal(signal.SIGALRM, expired)
    signal.alarm(AUDIT_SECONDS)
    try:
        audit.require(not already_audited, "one terminal audit mode, append-only")
        audit.require(scoped_output(output), "bounded terminal output path")
        audit.require(
            all(
                os.environ.get(name) == "1"
                for name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS")
            )
            and os.environ.get("PYTHONDONTWRITEBYTECODE") == "1",
            "CPU1/no-bytecode launch required",
        )
        receipt = (run_failure if args.failure_only else run)(
            output, args.manifest, args.manifest_sha256
        )
        if not args.failure_only:
            failure_precedence(output)
        publish(output / success_name, receipt)
    except Exception as exc:
        if (
            scoped_output(output)
            and output.is_dir()
            and not already_audited
            and not os.path.lexists(output / failure_name)
        ):
            publish(
                output / failure_name,
                {
                    "status": "VALIDITY_FAILURE",
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                    "manifest_sha256": args.manifest_sha256,
                    "efficacy": "NOT_EVALUATED",
                    "calibration": "NOT_EVALUATED",
                    "numeric_failure_replayed": False,
                },
            )
        raise
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, previous)
    print(json.dumps(receipt, indent=2), flush=True)


if __name__ == "__main__":
    main()
