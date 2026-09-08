"""Generated-only cold provenance contracts; full real-fit rehearsal is root-owned."""

import ast
import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path

import numpy as np
import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/audit_task_trca_temporal_source39.py"
spec = importlib.util.spec_from_file_location("temporal_cold_test", SCRIPT)
cold = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cold)


def save(path, value):
    with path.open("x") as stream:
        json.dump(value, stream)
    return {
        "path": str(path),
        "bytes": path.stat().st_size,
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }


def journal(profile="generated"):
    contract = cold.audit.profile_record(profile)
    ids = tuple(contract["source_ids"])
    rows = []

    def add(fold, pid, kind, interface, blocks, samples=None):
        row = {
            "seq": len(rows),
            "phase": "before_decode",
            "outer_fold": fold,
            "participant_id": pid,
            "role": "evaluation" if pid in ids[fold::3] else "fit",
            "kind": kind,
            "interface": interface,
            "blocks": list(blocks),
        }
        if samples is not None:
            row["samples"] = samples
        if kind in cold.QUERY_KINDS:
            row["freeze_sha256"] = "a" * 64
        rows.append(row)

    for role in ("fit", "evaluation"):
        if role == "evaluation":
            rows.append(
                {
                    "seq": len(rows),
                    "kind": "all_models_frozen",
                    "phase": "barrier",
                    "freeze_sha256": "a" * 64,
                    "manifest_sha256": "b" * 64,
                }
            )
        for fold in range(3):
            for pid in ids:
                if (pid in ids[fold::3]) != (role == "evaluation"):
                    continue
                for interface in (0, 1):
                    for k in (3, 5):
                        add(fold, pid, "metadata_support", interface, range(k))
                    for n in contract["samples"]:
                        for k in (3, 5):
                            add(fold, pid, "support", interface, range(k), n)
                        if role == "fit":
                            add(fold, pid, "source_supervision", interface, (5,), n)
                        else:
                            for kind in cold.QUERY_KINDS:
                                add(fold, pid, kind, interface, range(6, 10), n)
    return rows, contract


@pytest.mark.parametrize("profile,events,queries", [("generated", 325, 72), ("human", 4213, 1248)])
def test_exact_generated_and_human_access_profiles(profile, events, queries):
    rows, contract = journal(profile)
    counts = cold.check_access(rows, contract, "a" * 64, "b" * 64)
    assert len(rows) == events
    assert sum(counts["evaluation:" + kind] for kind in cold.QUERY_KINDS) == queries


@pytest.mark.parametrize(
    "defect",
    [
        "duplicate_cell",
        "missing",
        "wrong_window",
        "wrong_interface",
        "pre_freeze",
        "no_barrier",
        "duplicate_barrier",
        "wrong_freeze",
        "wrong_manifest",
        "late_source",
        "future_metadata",
        "bad_sequence",
        "wrong_role",
        "extra_field",
        "boolean_routing",
    ],
)
def test_access_tampering_rejected_even_when_kind_counts_match(defect):
    rows, contract = journal()
    barrier = next(j for j, row in enumerate(rows) if row["kind"] == "all_models_frozen")
    if defect == "duplicate_cell":
        rows[1] = {**rows[0], "seq": 1}
    elif defect == "missing":
        rows.pop()
    elif defect == "wrong_window":
        next(row for row in rows if "samples" in row)["samples"] = 23
    elif defect == "wrong_interface":
        rows[0]["interface"] = 1
    elif defect == "pre_freeze":
        rows[0], rows[barrier + 1] = rows[barrier + 1], rows[0]
        for j, row in enumerate(rows):
            row["seq"] = j
    elif defect == "no_barrier":
        rows.pop(barrier)
    elif defect == "duplicate_barrier":
        rows.insert(barrier, dict(rows[barrier]))
        for j, row in enumerate(rows):
            row["seq"] = j
    elif defect == "wrong_freeze":
        next(row for row in rows if row["kind"] == "query")["freeze_sha256"] = "c" * 64
    elif defect == "wrong_manifest":
        rows[barrier]["manifest_sha256"] = "c" * 64
    elif defect == "late_source":
        rows.append({**rows[0], "seq": len(rows)})
    elif defect == "future_metadata":
        rows[0]["blocks"] = [0, 1, 5]
    elif defect == "bad_sequence":
        rows[0]["seq"] = 1
    elif defect == "wrong_role":
        rows[0]["role"] = "evaluation"
    elif defect == "extra_field":
        rows[0]["samples"] = 17
    else:
        rows[0]["interface"] = False
    with pytest.raises(ValueError):
        cold.check_access(rows, contract, "a" * 64, "b" * 64)


@pytest.mark.parametrize("name", ["failure.json", "cold_audit_failure.json"])
def test_failure_precedes_manifest_and_arrays(tmp_path, monkeypatch, name):
    save(tmp_path / name, {"status": "VALIDITY_FAILURE"})
    monkeypatch.setattr(cold, "read_pinned", lambda *a, **kw: pytest.fail("no reads after failure"))
    with pytest.raises(ValueError, match="precedence"):
        cold.run(tmp_path, tmp_path / "not-read.json", "a" * 64)


def test_dangling_failure_and_existing_cold_receipt_precedence(tmp_path, monkeypatch):
    (tmp_path / "failure.json").symlink_to(tmp_path / "nonexistent")
    with pytest.raises(ValueError, match="precedence"):
        cold.failure_precedence(tmp_path)
    other = tmp_path / "separate"
    other.mkdir()
    save(other / "cold_audit.json", {})
    monkeypatch.setattr(cold, "read_pinned", lambda *a, **kw: pytest.fail("already cold"))
    with pytest.raises(ValueError, match="append-only"):
        cold.run(other, other / "not-read", "a" * 64)


@pytest.mark.parametrize(
    "defect", ["hash", "size", "symlink", "hardlink", "duplicate_json", "mutation"]
)
def test_pinned_file_guards(tmp_path, monkeypatch, defect):
    path = tmp_path / "generated.json"
    binding = save(path, {"generated": True})
    if defect == "hash":
        binding["sha256"] = "a" * 64
    elif defect == "size":
        binding["bytes"] += 1
    elif defect == "symlink":
        alias = tmp_path / "alias.json"
        alias.symlink_to(path)
        binding["path"] = str(alias)
    elif defect == "hardlink":
        os.link(path, tmp_path / "link.json")
    elif defect == "duplicate_json":
        path.write_text('{"generated":true,"generated":false}')
        binding["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
        binding["bytes"] = path.stat().st_size
    else:
        original = cold._json

        def mutate(raw):
            path.write_text('{"generated":false}')
            return original(raw)

        monkeypatch.setattr(cold, "_json", mutate)
    with pytest.raises(ValueError):
        cold.read_pinned(binding["path"], binding["sha256"], binding["bytes"])


def test_numpy_without_pickle_and_cold_publication(tmp_path):
    path = tmp_path / "generated.npz"
    with path.open("xb") as stream:
        np.savez(stream, example=np.array([1, 2]))
    values, binding = cold.read_pinned(path, kind="npz")
    assert values["example"].tolist() == [1, 2]
    assert binding["bytes"] == path.stat().st_size
    output = tmp_path / "receipt.json"
    cold.publish(output, {"status": "GENERATED_ONLY"})
    assert output.stat().st_mode & 0o777 == 0o400
    with pytest.raises(FileExistsError):
        cold.publish(output, {})


def freeze_fixture(tmp_path):
    contract = cold.audit.profile_record("generated")
    ids = tuple(contract["source_ids"])
    artifacts, models, frozen = {}, [], []
    for fold in range(3):
        fitting = [pid for pid in ids if pid not in ids[fold::3]]
        artifacts[f"source{fold}"] = {
            "path": str(tmp_path / f"source{fold}.npz"),
            "sha256": "a" * 64,
            "bytes": 10,
        }
        artifacts[f"model{fold}"] = {
            "path": str(tmp_path / f"model{fold}.json"),
            "sha256": "c" * 64,
            "bytes": 20,
        }
        schema = {"schema": cold.independent.SCHEMA, "score_schema": cold.independent.SCORE_SCHEMA}
        models.append(
            {
                "pipeline": {**schema, "fit_ids": fitting},
                "selection": {**schema, "outer_evaluation_ids": list(ids[fold::3])},
                "source_artifact": artifacts[f"source{fold}"],
                "manifest_sha256": "b" * 64,
                "independent_source_audit": {"status": "TEMPORAL_SOURCE_SELECTION_AUDIT_PASS"},
            }
        )
        frozen.append(
            {
                "fold_id": fold,
                "fit_ids": fitting,
                "evaluation_ids": list(ids[fold::3]),
                **artifacts[f"model{fold}"],
            }
        )
    freeze = {
        "schema": "cfeg.task_trca_temporal.all_models_frozen.v1",
        "status": "ALL_MODELS_FROZEN",
        "source_ids": list(ids),
        "manifest_sha256": "b" * 64,
        "query_access_count": 0,
        "models": frozen,
    }
    return freeze, artifacts, models, contract


@pytest.mark.parametrize(
    "defect",
    [None, "model_hash", "model_path", "source", "fit", "schema", "audit", "query", "duplicate"],
)
def test_freeze_binds_exact_model_and_source_artifacts(tmp_path, defect):
    freeze, artifacts, models, contract = freeze_fixture(tmp_path)
    if defect == "model_hash":
        freeze["models"][0]["sha256"] = "d" * 64
    elif defect == "model_path":
        freeze["models"][0]["path"] = str(tmp_path / "different.json")
    elif defect == "source":
        models[0]["source_artifact"] = artifacts["source1"]
    elif defect == "fit":
        models[0]["pipeline"]["fit_ids"] = models[1]["pipeline"]["fit_ids"]
    elif defect == "schema":
        models[0]["pipeline"]["schema"] = "task-trca-shape-v1"
    elif defect == "audit":
        models[0]["independent_source_audit"]["status"] = "PENDING"
    elif defect == "query":
        freeze["query_access_count"] = 1
    elif defect == "duplicate":
        freeze["models"][1] = freeze["models"][0]
    if defect is None:
        cold.check_freeze(freeze, artifacts, models, "b" * 64, contract)
    else:
        with pytest.raises(ValueError):
            cold.check_freeze(freeze, artifacts, models, "b" * 64, contract)


def test_artifact_exact_inventory_and_names(tmp_path):
    ids = cold.audit.profile_record("generated")["source_ids"]
    artifacts = {
        key: save(tmp_path / name, {"generated": True})
        for key, name in cold.artifact_names(ids).items()
    }
    cold.bind_artifacts(tmp_path, artifacts, ids)
    bad = {key: value for key, value in artifacts.items() if key != "source1"}
    with pytest.raises(ValueError, match="inventory"):
        cold.bind_artifacts(tmp_path, bad, ids)
    bad = {**artifacts, "model0": artifacts["model1"]}
    with pytest.raises(ValueError, match="basename"):
        cold.bind_artifacts(tmp_path, bad, ids)


def test_generated_summary_is_not_human_efficacy():
    scores = np.zeros((9, 2, 1, 2, 10, 48, 12))
    a0 = np.zeros((9, 2, 1, 48, 12))
    result = cold.generated_summary({"scores": scores, "a0": a0})
    assert result["terminal"] == "GENERATED_NOT_EVALUATED"
    assert result["status"] == "METADATA_NOT_EVALUATED"
    assert np.asarray(result["correct_counts"]).shape == (9, 2, 1, 2, 10)
    assert np.asarray(result["correct_counts"]).min() == 4


def test_mandatory_code_pin_closure_and_no_producer_imports():
    assert len(cold.CODE_PATHS) == len(set(cold.CODE_PATHS)) == 25
    assert "src/cfeg/analysis/metadata_prior_validation.py" in cold.CODE_PATHS
    tree = ast.parse(SCRIPT.read_text())
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module == "cfeg.analysis":
            assert {value.name for value in node.names} <= {
                "task_trca_temporal_artifact_audit",
                "task_trca_temporal_audit",
            }


def manifest_fixture():
    weights = {
        name: {interface: [1.0] * 5 for interface in ("dry", "wet")}
        for name in ("ETRCA", "A0_author")
    }
    native = {
        "generated": True,
        "seed": 20260909,
        "interfaces": ["dry", "wet"],
        "native_weights": weights,
        "source_projection": {"envelope_provenance": {"study_id": "GENERATED-test"}},
    }
    program = {
        "schema": "cfeg.metadata_learning_program.v1",
        "program_id": "metadata-learning-program-v1",
        "authority": {
            "source39_development_authorized_after_candidate_preflight": True,
            "implementation_and_generated_validation_authorized": True,
            "held60_authorized": False,
            "retired_s1_s3_authorized": False,
            "old_terminal_candidates_reopened": False,
        },
        "candidates": [
            {"slot": "C1", "id": "task-trca-temporal-v1", "design_sha256": cold.DESIGN_SHA},
            {
                "slot": "C2",
                "id": "task-trca-pair-s-v1",
                "design_path": "configs/c2.json",
                "design_sha256": "d" * 64,
            },
        ],
        "budget": {
            "cpu_threads": 1,
            "human_seconds_per_attempt_max": 7200,
            "output_bytes_per_attempt_max": 12 * 1024**3,
            "optimizer_updates_per_primary_max": 24000,
            "gpu_memory_fraction_max": 0.16,
        },
    }
    manifest = {
        "schema": "cfeg.task_trca_temporal.source39_generated.v1",
        "status": "GENERATED_FROZEN",
        "study_id": "task-trca-temporal-v1",
        "score_schema": cold.independent.SCORE_SCHEMA,
        "generated": True,
        "backend": "batch",
        "precision": "float64",
        "profile": cold.audit.profile_record("generated"),
        "source_ids": cold.audit.profile_record("generated")["source_ids"],
        "held60_authorized": False,
        "retired1_3_authorized": False,
        "old_attempts_reopened": False,
        "design": {
            "path": str(cold.ROOT / "configs/analysis/task_trca_temporal_v1_design.json"),
            "sha256": cold.DESIGN_SHA,
        },
        "native_root": "/tmp/GENERATED-fixture",
        "native_plan": {"path": "/tmp/GENERATED-fixture/native_plan.json", "sha256": "a" * 64},
        "program_path": str(cold.ROOT / "configs/analysis/metadata_learning_program_v1.json"),
        "program_sha256": "b" * 64,
        "preflight": None,
        "native_weights": {name: [[1.0] * 5] * 2 for name in weights},
        "cpu_threads": 1,
        "max_seconds": 7200,
        "output_budget_bytes": 4 * 1024**3,
        "optimizer_updates_max": 24000,
        "gpu_memory_fraction": 0.16,
        "code_pins": {path: "c" * 64 for path in cold.CODE_PATHS},
    }
    return manifest, program, native


@pytest.mark.parametrize(
    "defect",
    [
        None,
        "empty_pins",
        "missing_pin",
        "extra_pin",
        "profile",
        "native_origin",
        "held_authority",
        "candidate",
        "c2_missing",
        "steps",
        "seconds",
        "weights",
        "scorer",
    ],
)
def test_manifest_program_code_grid_and_budget_guards(monkeypatch, defect):
    manifest, program, native = copy.deepcopy(manifest_fixture())
    reads = []

    def read(path, *unused, **kwargs):
        reads.append(str(path))
        assert kwargs.get("kind") != "npz"
        return {manifest["program_path"]: program, manifest["native_plan"]["path"]: native}.get(
            str(path), {}
        ), {}

    monkeypatch.setattr(cold, "read_pinned", read)
    if defect == "empty_pins":
        manifest["code_pins"] = {}
    elif defect == "missing_pin":
        manifest["code_pins"].pop(cold.CODE_PATHS[-1])
    elif defect == "extra_pin":
        manifest["code_pins"]["../wrong.py"] = "c" * 64
    elif defect == "profile":
        manifest["profile"]["samples"] = [23]
    elif defect == "native_origin":
        native["generated"] = False
    elif defect == "held_authority":
        program["authority"]["held60_authorized"] = True
    elif defect == "candidate":
        program["candidates"][0]["design_sha256"] = "d" * 64
    elif defect == "c2_missing":
        program["candidates"].pop()
    elif defect == "steps":
        manifest["optimizer_updates_max"] += 1
    elif defect == "seconds":
        manifest["max_seconds"] += 1
    elif defect == "weights":
        manifest["native_weights"]["ETRCA"][0][0] = 2.0
    elif defect == "scorer":
        manifest["score_schema"] = "global-score"
    if defect is None:
        profile, contract = cold.validate_manifest(manifest, "e" * 64)
        assert profile == "generated" and contract == cold.audit.profile_record("generated")
        assert {str(cold.ROOT / name) for name in cold.CODE_PATHS} <= set(reads)
    else:
        with pytest.raises((ValueError, AssertionError)):
            cold.validate_manifest(manifest, "e" * 64)


def events_fixture(tmp_path):
    access, contract = journal()
    _freeze, artifacts, models, _ = freeze_fixture(tmp_path)
    artifacts["freeze"] = {"sha256": "a" * 64}
    barrier = next(j for j, row in enumerate(access) if row["kind"] == "all_models_frozen")
    result = {"summary": {"terminal": "GENERATED_NOT_EVALUATED"}}
    rows = []

    def add(name, access_seq, **fields):
        rows.append(
            {
                "seq": len(rows),
                "access_seq": access_seq,
                "elapsed_seconds": float(len(rows)),
                "event": name,
                **fields,
            }
        )

    for fold in range(3):
        add(
            "outer_model_frozen",
            (fold + 1) * barrier // 3,
            fold=fold,
            source_audit=models[fold]["independent_source_audit"],
        )
    add("all_models_frozen", barrier + 1, freeze_sha256="a" * 64)
    for fold in range(3):
        for pid in contract["source_ids"][fold::3]:
            end = 1 + max(
                j
                for j, row in enumerate(access)
                if row.get("role") == "evaluation" and row.get("participant_id") == pid
            )
            add("evaluation_participant_saved", end, fold=fold, participant_id=pid)
        add("evaluation_fold_audit_pass", end, fold=fold, receipt={"status": "PASS", "fold": fold})
    add(
        "attempt_verified_before_publication",
        len(access),
        result_path=str(tmp_path / "result.json"),
        terminal=result["summary"]["terminal"],
    )
    return rows, access, contract, artifacts, models, result


@pytest.mark.parametrize(
    "defect",
    [
        None,
        "missing",
        "wrong_access_seq",
        "duplicate_freeze",
        "wrong_model",
        "wrong_participant",
        "early_save",
        "wrong_terminal",
        "late_progress",
    ],
)
def test_event_access_freeze_and_publication_binding(tmp_path, monkeypatch, defect):
    rows, access, contract, artifacts, models, result = events_fixture(tmp_path)
    monkeypatch.setattr(
        cold, "load_artifact", lambda _, key: {"status": "PASS", "fold": int(key[-1])}
    )
    if defect == "missing":
        rows.pop()
    elif defect == "wrong_access_seq":
        rows[3]["access_seq"] -= 1
    elif defect == "duplicate_freeze":
        rows[4] = {**rows[3], "seq": 4}
    elif defect == "wrong_model":
        rows[0]["fold"] = 1
    elif defect == "wrong_participant":
        rows[4]["participant_id"] = 102
    elif defect == "early_save":
        rows[4]["access_seq"] = rows[3]["access_seq"]
    elif defect == "wrong_terminal":
        rows[-1]["terminal"] = "DEVELOPMENT_CALIBRATION_BENEFIT_CANDIDATE"
    elif defect == "late_progress":
        rows[-1]["event"] = "inner_start"
    if defect is None:
        assert cold.check_events(rows, access, contract, artifacts, models, result, tmp_path)[
            "publication_barrier_verified"
        ]
    else:
        with pytest.raises(ValueError):
            cold.check_events(rows, access, contract, artifacts, models, result, tmp_path)


@pytest.mark.parametrize(
    "defect",
    [
        None,
        "failure",
        "cold_failure",
        "program",
        "result",
        "native_pin",
        "retired",
        "program_retired",
    ],
)
def test_human_preflight_chain_and_authority_before_data(tmp_path, monkeypatch, defect):
    manifest, program, native = manifest_fixture()
    manifest.update(
        schema="cfeg.task_trca_temporal.source39_execution.v1",
        status="EXECUTION_FROZEN",
        generated=False,
        profile=cold.audit.profile_record("human"),
        source_ids=cold.audit.profile_record("human")["source_ids"],
        native_plan={
            "path": str(cold.ROOT / "configs/analysis/metadata_prior_source39_v1.json"),
            "sha256": cold.OLD_PLAN_SHA,
        },
        native_manifest_sha256=cold.NATIVE_MANIFEST_SHA,
        preflight={"path": str(tmp_path / "cold_audit.json"), "sha256": "d" * 64, "bytes": 10},
    )
    receipt = {
        "status": "GENERATED_COLD_INDEPENDENT_AUDIT_PASS",
        "generated": True,
        "profile": cold.audit.profile_record("generated"),
        "code_pins": manifest["code_pins"],
        "program_sha256": manifest["program_sha256"],
        "manifest_path": str(tmp_path / "generated_manifest.json"),
        "manifest_sha256": "e" * 64,
        "result_sha256": "f" * 64,
    }
    generated_manifest = {
        "schema": "cfeg.task_trca_temporal.source39_generated.v1",
        "status": "GENERATED_FROZEN",
        "profile": receipt["profile"],
        "program_sha256": manifest["program_sha256"],
        "code_pins": manifest["code_pins"],
        "output_root": str(tmp_path),
    }
    generated_result = {
        "status": "GENERATED_COMPLETE",
        "manifest_sha256": "e" * 64,
        "program_sha256": manifest["program_sha256"],
        "profile": receipt["profile"],
        "generated": True,
    }
    reads = []

    def read(path, *unused, **kwargs):
        reads.append(str(path))
        assert kwargs.get("kind") != "npz"
        return {
            manifest["program_path"]: program,
            manifest["native_plan"]["path"]: native,
            manifest["preflight"]["path"]: receipt,
            receipt["manifest_path"]: generated_manifest,
            str(tmp_path / "result.json"): generated_result,
        }.get(str(path), {}), {}

    monkeypatch.setattr(cold, "read_pinned", read)
    if defect in ("failure", "cold_failure"):
        save(tmp_path / ("failure.json" if defect == "failure" else "cold_audit_failure.json"), {})
    elif defect == "program":
        receipt["program_sha256"] = "f" * 64
    elif defect == "result":
        generated_result["manifest_sha256"] = "f" * 64
    elif defect == "native_pin":
        manifest["native_plan"]["sha256"] = "a" * 64
    elif defect == "retired":
        manifest["retired1_3_authorized"] = True
    elif defect == "program_retired":
        program["authority"]["retired_s1_s3_authorized"] = True
    if defect is None:
        assert cold.validate_manifest(manifest, "a" * 64)[0] == "human"
    else:
        with pytest.raises(ValueError):
            cold.validate_manifest(manifest, "a" * 64)
        if defect == "native_pin":
            assert manifest["native_plan"]["path"] not in reads
        if defect in ("failure", "cold_failure"):
            assert manifest["preflight"]["path"] not in reads
