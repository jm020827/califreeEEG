"""Pure generated manifest/journal/publication tests, not a completed study run."""

import importlib.util
import json
import os
from pathlib import Path

import numpy as np
import pytest

from cfeg.analysis import task_trca_temporal_archive as archive

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/run_task_trca_temporal_source39.py"
SPEC = importlib.util.spec_from_file_location("temporal_runtime_leaf_test", SCRIPT)
runtime = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(runtime)


def manifest(profile=archive.GENERATED_PROFILE):
    return {
        "schema": "cfeg.task_trca_temporal.source39_generated.v1"
        if profile.generated
        else "cfeg.task_trca_temporal.source39_execution.v1",
        "status": "GENERATED_FROZEN" if profile.generated else "EXECUTION_FROZEN",
        "study_id": archive.SCHEMA,
        "score_schema": archive.SCORE_SCHEMA,
        "generated": profile.generated,
        "profile": profile.record(),
        "source_ids": list(profile.source_ids),
        "held60_authorized": False,
        "retired1_3_authorized": False,
        "old_attempts_reopened": False,
        "program_path": str(runtime.ROOT / "configs/analysis/metadata_learning_program_v1.json"),
        "program_sha256": "a" * 64,
        "design": {
            "path": str(runtime.ROOT / "configs/analysis/task_trca_temporal_v1_design.json"),
            "sha256": runtime.DESIGN_SHA,
        },
        "native_plan": {
            "path": "/explicit/generated/native/native_plan.json"
            if profile.generated
            else str(runtime.ROOT / "configs/analysis/metadata_prior_source39_v1.json"),
            "sha256": "b" * 64 if profile.generated else runtime.OLD_PLAN_SHA,
        },
        "native_root": "/explicit/generated/native",
        "native_manifest_sha256": "c" * 64 if profile.generated else runtime.NATIVE_MANIFEST_SHA,
        "device": "cpu" if profile.generated else "cuda",
        "backend": "batch",
        "precision": "float64",
        "cpu_threads": 1,
        "gpu_memory_fraction": 0.16,
        "output_root": "/explicit/generated/task-trca-temporal-source39-generated-test",
        "output_budget_bytes": 4 * 1024**3,
        "max_seconds": 7200,
        "optimizer_updates_max": 24000,
        "code_pins": dict.fromkeys(runtime.CODE_PATHS, "d" * 64),
        "preflight": None
        if profile.generated
        else {"path": "/explicit/generated/cold.json", "sha256": "e" * 64, "bytes": 100},
    }


def test_profiles_have_exact_25_code_pins_and_immutable_grids():
    assert len(runtime.CODE_PATHS) == len(set(runtime.CODE_PATHS)) == 25
    for profile in (archive.HUMAN_PROFILE, archive.GENERATED_PROFILE):
        doc = manifest(profile)
        assert runtime.validate_manifest(doc, profile) is doc
    assert len(archive.HUMAN_PROFILE.conditions) == 16
    assert len(archive.GENERATED_PROFILE.conditions) == 4


@pytest.mark.parametrize(
    "field,value",
    [
        ("schema", "cfeg.task_trca_shape.source39_execution.v1"),
        ("status", "DESIGN_ONLY"),
        ("score_schema", "native-global"),
        ("study_id", "task-trca-pair-s-v1"),
        ("held60_authorized", True),
        ("retired1_3_authorized", True),
        ("old_attempts_reopened", True),
        ("device", "automatic"),
        ("backend", "scalar"),
        ("precision", "float32"),
        ("cpu_threads", 2),
        ("cpu_threads", True),
        ("max_seconds", 7201),
        ("max_seconds", True),
        ("optimizer_updates_max", 24001),
        ("output_budget_bytes", 4 * 1024**3 + 1),
        ("gpu_memory_fraction", 0.17),
        ("gpu_memory_fraction", True),
        ("native_root", "relative/root"),
        ("program_sha256", "bad"),
        ("generated", False),
    ],
)
def test_manifest_rejects_scientific_scope_and_budget_drift(field, value):
    doc = manifest()
    doc[field] = value
    with pytest.raises((ValueError, KeyError)):
        runtime.validate_manifest(doc, archive.GENERATED_PROFILE)


@pytest.mark.parametrize(
    "field,value",
    [
        ("source_ids", [4, 6, 8]),
        ("interfaces", [0]),
        ("interfaces", [False, True]),
        ("samples", [13]),
        ("budgets", [1, 3, 5]),
        ("seed", 20260910),
        ("generated", False),
    ],
)
def test_generated_grid_cannot_be_arbitrarily_overridden(field, value):
    doc = manifest()
    doc["profile"][field] = value
    with pytest.raises(ValueError, match="profile"):
        runtime.validate_manifest(doc, archive.GENERATED_PROFILE)


@pytest.mark.parametrize("case", ["missing_pin", "extra_pin", "bad_pin", "design", "preflight"])
def test_pin_contract_rejects_missing_or_switched_pins(case):
    doc = manifest()
    if case == "missing_pin":
        doc["code_pins"].pop(runtime.CODE_PATHS[0])
    elif case == "extra_pin":
        doc["code_pins"]["arbitrary.py"] = "e" * 64
    elif case == "bad_pin":
        doc["code_pins"][runtime.CODE_PATHS[0]] = "wrong"
    elif case == "design":
        doc["design"]["sha256"] = "0" * 64
    else:
        doc["preflight"] = {"previous": "human"}
    with pytest.raises(ValueError):
        runtime.validate_manifest(doc, archive.GENERATED_PROFILE)


def test_human_and_generated_manifests_never_coerce_each_other():
    with pytest.raises(ValueError, match="schema/status"):
        runtime.validate_manifest(manifest(), archive.HUMAN_PROFILE)
    with pytest.raises(ValueError, match="schema/status"):
        runtime.validate_manifest(manifest(archive.HUMAN_PROFILE), archive.GENERATED_PROFILE)
    for key in ("device", "native_plan", "native_manifest_sha256", "preflight"):
        doc = manifest(archive.HUMAN_PROFILE)
        doc[key] = manifest()[key]
        with pytest.raises(ValueError):
            runtime.validate_manifest(doc, archive.HUMAN_PROFILE)


def test_generated_entry_rejects_profile_before_reading_manifest():
    with pytest.raises(ValueError, match="fixed generated"):
        runtime.run_generated("/never-opened", "a" * 64, profile=archive.HUMAN_PROFILE)


def test_rejected_human_manifest_never_reaches_input_loader(tmp_path, monkeypatch):
    path = tmp_path / "generated-manifest.json"
    pin = runtime.write_json(path, manifest())
    called = []
    original = runtime.read_json

    def read(path, expected):
        called.append(str(path))
        return original(path, expected)

    monkeypatch.setattr(runtime, "read_json", read)
    with pytest.raises(ValueError, match="schema/status"):
        runtime.run(path, pin["sha256"])
    assert called == [str(path)]


def access(kind="support", *, freeze=None):
    value = {
        "phase": "before_decode",
        "participant_id": 4,
        "role": "evaluation",
        "kind": kind,
        "interface": 0,
        "samples": 17,
        "blocks": [6, 7, 8, 9] if kind in runtime.QUERY_KINDS else [0, 1, 2],
    }
    if freeze is not None:
        value["freeze_sha256"] = freeze
    return value


def test_access_journal_fsync_and_ordered_barrier_before_query(tmp_path, monkeypatch):
    path = tmp_path / "access.jsonl"
    state = {"outer_fold": 0, "query_access_count": 0}
    syncs = []
    original = os.fsync

    def sync(fd):
        syncs.append(fd)
        return original(fd)

    monkeypatch.setattr(os, "fsync", sync)
    journal = runtime.AccessJournal(path, state, "a" * 64)
    journal(access())
    assert len(syncs) == 2  # Empty journal creation + durable predecode row.
    with pytest.raises(ValueError, match="barrier"):
        journal(access("query", freeze="b" * 64))
    assert state["query_access_count"] == 0
    journal.barrier("b" * 64)
    journal(access("query", freeze="b" * 64))
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    assert [row["seq"] for row in rows] == [0, 1, 2]
    assert rows[1] == {
        "seq": 1,
        "kind": "all_models_frozen",
        "phase": "barrier",
        "freeze_sha256": "b" * 64,
        "manifest_sha256": "a" * 64,
    }
    assert state["query_access_count"] == 1 and len(syncs) == 4


def test_journal_failure_cannot_advance_decode_state(tmp_path, monkeypatch):
    state = {"outer_fold": 0, "query_access_count": 0}
    journal = runtime.AccessJournal(tmp_path / "access.jsonl", state, "a" * 64)
    journal.barrier("b" * 64)

    def fail(_):
        raise OSError("fsync failed")

    monkeypatch.setattr(os, "fsync", fail)
    with pytest.raises(OSError):
        journal(access("query", freeze="b" * 64))
    assert journal.seq == 1 and state["query_access_count"] == 0


def test_duplicate_barrier_wrong_hash_and_missing_fold_rejected(tmp_path):
    state = {"outer_fold": None, "query_access_count": 0}
    journal = runtime.AccessJournal(tmp_path / "access.jsonl", state, "a" * 64)
    with pytest.raises(ValueError, match="outer fold"):
        journal(access())
    journal.barrier("b" * 64)
    with pytest.raises(ValueError, match="Exactly one"):
        journal.barrier("b" * 64)
    state["outer_fold"] = 0
    with pytest.raises(ValueError, match="barrier"):
        journal(access("a0", freeze="c" * 64))


def test_event_journal_links_durable_access_sequence(tmp_path, monkeypatch):
    state = {"outer_fold": 0, "query_access_count": 0}
    access_log = runtime.AccessJournal(tmp_path / "access.jsonl", state, "a" * 64)
    event_log = runtime.EventJournal(
        tmp_path / "events.jsonl", access_log, runtime.time.perf_counter(), 1024**2
    )
    event_log({"event": "outer_model_frozen", "fold": 0})
    access_log(access())
    access_log.barrier("b" * 64)
    event_log({"event": "all_models_frozen", "freeze_sha256": "b" * 64})
    access_log(access("query", freeze="b" * 64))
    event_log(
        {"event": "attempt_verified_before_publication", "terminal": "GENERATED_NOT_EVALUATED"}
    )
    rows = [json.loads(row) for row in event_log.path.read_text().splitlines()]
    assert [row["seq"] for row in rows] == [0, 1, 2]
    assert [row["access_seq"] for row in rows] == [0, 2, 3]
    with pytest.raises(ValueError, match="cannot be supplied"):
        event_log({"event": "forged", "seq": 0})

    def fail(_):
        raise OSError("event fsync failure")

    monkeypatch.setattr(os, "fsync", fail)
    with pytest.raises(OSError, match="event fsync"):
        event_log({"event": "durability_failure"})
    assert event_log.seq == 3


def preflight_fixture(tmp_path, mutation=None):
    output = tmp_path / "task-trca-temporal-source39-generated-preflight"
    output.mkdir()
    human = manifest(archive.HUMAN_PROFILE)
    generated = manifest()
    generated["output_root"] = str(output)
    if mutation == "generated_code":
        generated["code_pins"][runtime.CODE_PATHS[0]] = "f" * 64
    elif mutation == "generated_output":
        generated["output_root"] = str(tmp_path / "unrelated-output")
    generated_pin = runtime.write_json(tmp_path / "generated-manifest.json", generated)
    result = {
        "status": "GENERATED_COMPLETE",
        "cold_audit_status": "COLD_AUDIT_PENDING",
        "generated": True,
        "profile": archive.GENERATED_PROFILE.record(),
        "program_sha256": human["program_sha256"],
        "manifest_sha256": generated_pin["sha256"],
        "summary": {"terminal": "GENERATED_NOT_EVALUATED"},
    }
    if mutation == "result_status":
        result["status"] = "COMPLETE"
    elif mutation == "result_program":
        result["program_sha256"] = "f" * 64
    elif mutation == "result_manifest":
        result["manifest_sha256"] = "f" * 64
    result_pin = runtime.write_json(output / "result.json", result)
    receipt = {
        "status": "GENERATED_COLD_INDEPENDENT_AUDIT_PASS",
        "generated": True,
        "profile": archive.GENERATED_PROFILE.record(),
        "code_pins": human["code_pins"],
        "program_sha256": human["program_sha256"],
        "manifest_path": generated_pin["path"],
        "manifest_sha256": generated_pin["sha256"],
        "result_sha256": result_pin["sha256"],
    }
    if mutation in ("receipt_program", "receipt_manifest", "receipt_result"):
        receipt[mutation.removeprefix("receipt_") + "_sha256"] = "f" * 64
    elif mutation == "receipt_status":
        receipt["status"] = "COLD_INDEPENDENT_AUDIT_PASS"
    human["preflight"] = runtime.write_json(output / "cold_audit.json", receipt)
    if mutation in ("failure.json", "cold_audit_failure.json"):
        runtime.write_json(output / mutation, {"status": "VALIDITY_FAILURE"})
    elif mutation == "bytes":
        human["preflight"]["bytes"] += 1
    elif mutation == "alias":
        alias = tmp_path / "alias"
        alias.symlink_to(output, target_is_directory=True)
        human["preflight"]["path"] = str(alias / "cold_audit.json")
    return human, receipt


def test_preflight_requires_complete_same_code_program_generated_chain(tmp_path):
    document, receipt = preflight_fixture(tmp_path)
    assert runtime.validate_preflight(document) == receipt


@pytest.mark.parametrize(
    "mutation",
    [
        "generated_code",
        "generated_output",
        "result_status",
        "result_program",
        "result_manifest",
        "receipt_program",
        "receipt_manifest",
        "receipt_result",
        "receipt_status",
        "failure.json",
        "cold_audit_failure.json",
        "bytes",
        "alias",
    ],
)
def test_preflight_denies_broken_or_failed_generated_chain(tmp_path, mutation):
    document, _ = preflight_fixture(tmp_path, mutation)
    with pytest.raises((ValueError, PermissionError)):
        runtime.validate_preflight(document)


def test_preflight_failure_is_checked_before_any_receipt_parse(tmp_path, monkeypatch):
    document, _ = preflight_fixture(tmp_path, "failure.json")

    def denied(*_):
        raise AssertionError("Must not parse a success receipt after a failure exists")

    monkeypatch.setattr(runtime, "read_json", denied)
    with pytest.raises(ValueError, match="Failure takes precedence"):
        runtime.validate_preflight(document)


def test_program_requires_c2_math_pin_before_any_native_input(tmp_path, monkeypatch):
    monkeypatch.setattr(runtime, "ROOT", tmp_path)
    design_dir = tmp_path / "configs/analysis"
    design_dir.mkdir(parents=True)
    design = {
        "schema": "cfeg.task_trca_pair_s.design.v1",
        "study_id": "task-trca-pair-s-v1",
        "slot": "C2",
        "program_id": "metadata-learning-program-v1",
        "status": "SCIENTIFIC_SPEC_FROZEN_BEFORE_C1_HUMAN_FITTING",
        "inherit_design_sha256": runtime.DESIGN_SHA,
        "score_schema": archive.SCORE_SCHEMA,
    }
    pin = runtime.write_json(design_dir / "task_trca_pair_s_v1_design.json", design)
    program = {
        "schema": "cfeg.metadata_learning_program.v1",
        "status": "FAMILY_AND_BUDGET_FROZEN",
        "program_id": "metadata-learning-program-v1",
        "authority": {
            "source39_development_authorized_after_candidate_preflight": True,
            "held60_authorized": False,
            "retired_s1_s3_authorized": False,
        },
        "candidates": [
            {
                "slot": "C1",
                "id": archive.SCHEMA,
                "design_path": "configs/analysis/task_trca_temporal_v1_design.json",
                "design_sha256": runtime.DESIGN_SHA,
            },
            {
                "slot": "C2",
                "id": "task-trca-pair-s-v1",
                "design_path": "configs/analysis/task_trca_pair_s_v1_design.json",
                "design_sha256": pin["sha256"],
            },
        ],
    }
    runtime.validate_program(program, manifest())
    program["candidates"][1]["design_sha256"] = "f" * 64
    with pytest.raises(ValueError, match="SHA mismatch"):
        runtime.validate_program(program, manifest())


def test_publication_is_exclusive_readonly_and_pinned(tmp_path):
    output = tmp_path / "output"
    output.mkdir()
    json_path, npz_path = output / "model.json", output / "source.npz"
    payload = {"schema": archive.SCHEMA, "value": [1, 2, 3]}
    pin = runtime.write_json(json_path, payload, output=output, budget_bytes=4 * 1024**2)
    assert json_path.stat().st_mode & 0o777 == 0o400
    assert runtime.read_json(json_path, pin["sha256"]) == payload
    with pytest.raises(FileExistsError):
        runtime.write_json(json_path, payload)
    data = {"schema": np.array(archive.SCHEMA), "keys": np.arange(8).reshape(2, 4)}
    artifact = runtime.write_npz(npz_path, data, output=output, budget_bytes=4 * 1024**2)
    loaded = runtime.read_saved_arrays(artifact)
    assert npz_path.stat().st_mode & 0o777 == 0o400
    for key, value in data.items():
        np.testing.assert_array_equal(loaded[key], value)


def test_publication_rejects_objects_outside_paths_and_budget(tmp_path):
    output = tmp_path / "output"
    output.mkdir()
    with pytest.raises(ValueError, match="non-object"):
        runtime.write_npz(output / "object.npz", {"x": np.array({"key": 1}, dtype=object)})
    with pytest.raises(ValueError, match="output-only"):
        runtime.write_json(tmp_path / "outside.json", {}, output=output, budget_bytes=4 * 1024**2)
    with pytest.raises(ValueError, match="budget"):
        runtime.write_json(output / "small.json", {}, output=output, budget_bytes=100)
    assert not list(output.iterdir())


def test_read_saved_pin_and_alias_validation(tmp_path):
    path = tmp_path / "source.npz"
    pin = runtime.write_npz(path, {"x": np.zeros(3)})
    with pytest.raises(ValueError, match="SHA"):
        runtime.read_saved_arrays({**pin, "sha256": "0" * 64})
    alias = tmp_path / "alias.npz"
    alias.symlink_to(path)
    with pytest.raises(ValueError, match="alias"):
        runtime.read_saved_arrays({**pin, "path": str(alias)})


def test_generated_summary_is_not_human_efficiency_inference():
    scores = np.zeros((9, 2, 1, 2, 10, 48, 12))
    a0 = np.zeros((9, 2, 1, 48, 12))
    value = runtime.generated_summary(scores, a0, archive.GENERATED_PROFILE)
    assert (
        value["status"] == "METADATA_NOT_EVALUATED"
        and value["terminal"] == "GENERATED_NOT_EVALUATED"
    )
    assert set(value) == {
        "status",
        "terminal",
        "generated",
        "profile",
        "arms",
        "correct_counts",
        "a0_correct_counts",
        "query_count_per_cell",
        "interpretation",
    }
    assert np.asarray(value["correct_counts"]).shape == (9, 2, 1, 2, 10)
    assert np.all(np.asarray(value["correct_counts"]) == 4)
    assert np.all(np.asarray(value["a0_correct_counts"]) == 4)
    with pytest.raises(ValueError):
        runtime.generated_summary(scores, a0, archive.HUMAN_PROFILE)
    with pytest.raises(ValueError):
        runtime.generated_summary(scores[:1], a0[:1], archive.GENERATED_PROFILE)


def test_source_and_model_schema_fields_are_not_legacy_in_new_runner():
    source = SCRIPT.read_text()
    assert "run_task_trca_shape_source39" not in source
    assert "task_trca_shape_learning" not in source
    assert "task_trca_shape_evaluation" not in source
    assert "tracked(" not in source
    assert "audit.pack_evaluation(records)" in source
    assert "audit.combine_evaluation(fold_records)" in source
    assert '"cold_audit_status": "COLD_AUDIT_PENDING"' in source


def test_generated_input_failure_preserves_readonly_journals_and_budget(tmp_path, monkeypatch):
    """Generated orchestration failure only: no learner or human-input execution."""
    output = tmp_path / "task-trca-temporal-source39-generated-input-failure"
    document = manifest()
    document["output_root"] = str(output)
    document["native_root"] = str(tmp_path / "GENERATED-unopened-native")
    reads = []
    monkeypatch.setattr(runtime, "_audit_module", lambda: object())
    monkeypatch.setattr(runtime, "install_guard", lambda *_: None)

    def fail_input(path, _):
        reads.append(path)
        raise ValueError("GENERATED deliberately unavailable native transport")

    monkeypatch.setattr(runtime, "read_json", fail_input)
    with pytest.raises(ValueError, match="GENERATED deliberately"):
        runtime._execute(
            tmp_path / "GENERATED-manifest.json",
            "a" * 64,
            document,
            {"source_projection": {"path": str(tmp_path / "GENERATED-unopened-projection.json")}},
            archive.GENERATED_PROFILE,
            "GENERATED-unit-only",
        )
    failure = json.loads((output / "failure.json").read_text())
    assert failure["status"] == "VALIDITY_FAILURE"
    assert failure["state"]["query_access_count"] == 0
    assert failure["state"]["optimizer_updates_completed"] == 0
    assert failure["state"]["optimizer_updates_charged"] == 0
    assert set(failure["artifacts"]) == {"start", "events", "access"}
    assert not (output / "result.json").exists()
    assert reads == [Path(document["native_root"]) / "result.json"]
    assert all(path.stat().st_mode & 0o777 == 0o400 for path in output.iterdir())
