"""Generated-only archive boundary fixtures; no trained-model efficacy assertion."""

import hashlib
import importlib.util
import json
import os
import subprocess
import sys
import zipfile
from pathlib import Path

import numpy as np
import pytest

from cfeg.analysis import task_trca_n1_source39_archive as archive
from cfeg.analysis import task_trca_shape_archive as old

PROFILE = archive.GENERATED_PROFILE


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def partition(fold=0):
    evaluation = PROFILE.source_ids[fold::3]
    return archive.RolePartition(
        tuple(pid for pid in PROFILE.source_ids if pid not in evaluation), (), evaluation
    )


def make_archive(path, *, compressed=False, poison=False):
    x = np.empty((2, 10, 12, 5, 8, 17))
    for interface in (0, 1):
        for block in range(10):
            x[interface, block] = 10 * interface + block
    if poison:
        x[:, 5:] = np.nan
    arrays = {
        "x_17": x,
        "full_17": np.full((2, 2, 4, 12, 5, 12), 0.25),
        "a0_17": np.full((2, 4, 12, 5, 12), 0.5),
    }
    with zipfile.ZipFile(
        path, "w", compression=zipfile.ZIP_DEFLATED if compressed else zipfile.ZIP_STORED
    ) as target:
        for name, value in arrays.items():
            with target.open(name + ".npy", "w", force_zip64=True) as stream:
                np.lib.format.write_array(stream, value, allow_pickle=False)
    return archive.ArchiveSpec(path, 4, sha(path), path.stat().st_size)


@pytest.fixture
def source(tmp_path):
    return make_archive(tmp_path / "S004.npz")


def write_json(path, data):
    path.write_text(json.dumps(data))
    return {"path": str(path), "sha256": sha(path), "bytes": path.stat().st_size}


@pytest.fixture
def frozen(tmp_path):
    root = tmp_path / "synthetic-freeze-format-only"
    root.mkdir()
    models = []
    for fold in range(3):
        roles = partition(fold)
        source = root / f"source{fold}.npz"
        source.write_bytes(b"GENERATED opaque source pin fixture; not a trained model")
        source_artifact = {
            "path": str(source),
            "sha256": sha(source),
            "bytes": source.stat().st_size,
        }
        model = {
            "pipeline": {
                "schema": archive.SCHEMA,
                "score_schema": archive.SCORE_SCHEMA,
                "fit_ids": list(roles.fit_ids),
            },
            "selection": {
                "schema": archive.SCHEMA,
                "score_schema": archive.SCORE_SCHEMA,
                "outer_evaluation_ids": list(roles.evaluation_ids),
            },
            "source_artifact": source_artifact,
            "manifest_sha256": "a" * 64,
            "independent_source_audit": {"status": "TEMPORAL_SOURCE_SELECTION_AUDIT_PASS"},
        }
        descriptor = write_json(root / f"model{fold}.json", model)
        models.append(
            {
                "fold_id": fold,
                "fit_ids": list(roles.fit_ids),
                "evaluation_ids": list(roles.evaluation_ids),
                **descriptor,
            }
        )
    value = {
        "schema": archive.FREEZE_SCHEMA,
        "status": "ALL_MODELS_FROZEN",
        "source_ids": list(PROFILE.source_ids),
        "manifest_sha256": "a" * 64,
        "query_access_count": 0,
        "models": models,
    }
    path = root / "globalfreeze.json"
    write_json(path, value)
    return path, value


def token(frozen):
    path, _ = frozen
    return archive.verify_freeze(path, sha(path), "a" * 64, profile=PROFILE)


def test_exact_human_and_generated_profiles(source):
    assert archive.HUMAN_PROFILE == archive.RuntimeProfile()
    assert archive.HUMAN_PROFILE.source_ids == old.SOURCE_IDS
    assert archive.HUMAN_PROFILE.samples == (125, 188, 250, 500)
    assert PROFILE.seed == 20260914 and PROFILE.samples == (17,)
    assert archive.SCHEMA == "task-trca-n1-integration-v1"
    assert archive.STUDY_ID == "task-trca-n1-source39-v1"
    for kwargs in (
        {"samples": (125,)},
        {"interfaces": (0,)},
        {"seed": 73},
        {"source_ids": PROFILE.source_ids},
    ):
        with pytest.raises(ValueError):
            archive.RuntimeProfile(**kwargs)
    with archive.NativeArchive(
        source, partition(), event_sink=lambda _: None, profile=PROFILE
    ) as reader:
        assert reader.support(0, 17, 3).shape == (3, 12, 5, 8, 17)


def test_mandatory_sink_and_role_rejection_before_open(source, monkeypatch):
    with pytest.raises(ValueError, match="sink"):
        archive.NativeArchive(source, partition(), event_sink=None, profile=PROFILE)
    monkeypatch.setattr(old, "regular_fd", lambda *a: pytest.fail("Role-denied input opened"))
    with pytest.raises(PermissionError):
        archive.NativeArchive(
            source, archive.RolePartition((6,)), event_sink=lambda _: None, profile=PROFILE
        )


@pytest.mark.parametrize("k", [3, 5])
def test_support_event_is_before_any_decode(source, monkeypatch, k):
    events = []
    original = old._decode_float64

    def decode(*args):
        assert events and events[-1]["phase"] == "before_decode"
        assert events[-1]["kind"] == "support" and events[-1]["blocks"] == list(range(k))
        return original(*args)

    monkeypatch.setattr(old, "_decode_float64", decode)
    with archive.NativeArchive(
        source, partition(), event_sink=events.append, profile=PROFILE
    ) as reader:
        result = reader.support(1, 17, k)
        assert result.shape == (k, 12, 5, 8, 17)
        assert not result.flags.writeable
        assert np.array_equal(result[:, 0, 0, 0, 0], 10 + np.arange(k))
        assert reader.access_log == events


def test_sink_failure_prevents_even_authorized_decode(source, monkeypatch):
    def broken_sink(_):
        raise OSError("durable journal unavailable")

    monkeypatch.setattr(
        old, "_decode_float64", lambda *a: pytest.fail("Decoded without durable event")
    )
    with archive.NativeArchive(
        source, partition(), event_sink=broken_sink, profile=PROFILE
    ) as reader:
        with pytest.raises(OSError, match="journal"):
            reader.support(0, 17, 3)
        assert reader.access_log == []


@pytest.mark.parametrize(
    "method,args",
    [
        ("query", (0, 17)),
        ("full_correlations", (0, 17, 3)),
        ("a0_correlations", (0, 17)),
        ("supervision", (0, 17)),
    ],
)
def test_denied_requests_emit_no_decode_event(source, monkeypatch, method, args):
    events = []
    monkeypatch.setattr(old, "_npy_data", lambda *a: pytest.fail("Denied NPY header read"))
    with (
        archive.NativeArchive(
            source, partition(), event_sink=events.append, profile=PROFILE
        ) as reader,
        pytest.raises(PermissionError),
    ):
        getattr(reader, method)(*args)
    assert not events


def test_query_new_freeze_and_exact_ranges(source, frozen):
    freeze, events = token(frozen), []
    with archive.NativeArchive(
        source, partition(), event_sink=events.append, profile=PROFILE
    ) as reader:
        assert np.array_equal(reader.query(1, 17, freeze)[:, 0, 0, 0, 0], np.arange(16, 20))
        assert np.all(reader.full_correlations(0, 17, 3, freeze) == 0.25)
        assert np.all(reader.full_correlations(1, 17, 5, freeze) == 0.25)
        assert np.all(reader.a0_correlations(0, 17, freeze) == 0.5)
        assert len(events) == 4
        assert all(
            row["blocks"] == [6, 7, 8, 9] and row["freeze_sha256"] == freeze.receipt_sha256
            for row in events
        )


def test_legacy_token_is_not_temporal_authority(source):
    # Deliberately uninitialized OLD type suffices: reject before its attributes.
    legacy = object.__new__(old.AllModelsFrozen)
    with (
        archive.NativeArchive(
            source, partition(), event_sink=lambda _: pytest.fail("Legacy event"), profile=PROFILE
        ) as reader,
        pytest.raises(PermissionError, match="new verified"),
    ):
        reader.query(0, 17, legacy)


def test_nonfinite_authorized_decode_retains_attempt_event(tmp_path):
    events = []
    source = make_archive(tmp_path / "poison.npz", poison=True)
    with archive.NativeArchive(
        source, partition(1), event_sink=events.append, profile=PROFILE
    ) as reader:
        with pytest.raises(ValueError, match="Nonfinite"):
            reader.supervision(0, 17)
        assert events[0]["kind"] == "source_supervision"


@pytest.mark.parametrize(
    "case",
    [
        "old_schema",
        "manifest",
        "source_ids",
        "query_count",
        "model_filename",
        "model_parent",
        "pipeline_schema",
        "score_schema",
        "fit_ids",
        "model_manifest",
        "source_filename",
        "source_hash",
        "selection_schema",
        "selection_roles",
        "audit_failure",
    ],
)
def test_freeze_rejects_wrong_binding(frozen, case):
    path, value = frozen
    model_path = path.parent / "model0.json"
    model = json.loads(model_path.read_text())
    if case == "old_schema":
        value["schema"] = "cfeg.task_trca_shape.all_models_frozen.v1"
    elif case == "manifest":
        value["manifest_sha256"] = "b" * 64
    elif case == "source_ids":
        value["source_ids"] = list(old.SOURCE_IDS)
    elif case == "query_count":
        value["query_access_count"] = 1
    elif case == "model_filename":
        value["models"][0]["path"] = str(path.parent / "renamed.json")
    elif case == "model_parent":
        value["models"][0]["path"] = str(path.parent.parent / "model0.json")
    else:
        if case == "pipeline_schema":
            model["pipeline"]["schema"] = "legacy"
        elif case == "score_schema":
            model["pipeline"]["score_schema"] = "legacy-global"
        elif case == "fit_ids":
            model["pipeline"]["fit_ids"] = list(partition().evaluation_ids)
        elif case == "model_manifest":
            model["manifest_sha256"] = "b" * 64
        elif case == "source_filename":
            model["source_artifact"]["path"] = str(path.parent / "source1.npz")
        elif case == "source_hash":
            model["source_artifact"]["sha256"] = "0" * 64
        elif case == "selection_schema":
            model["selection"]["schema"] = "legacy"
        elif case == "selection_roles":
            model["selection"]["outer_evaluation_ids"] = []
        else:
            model["independent_source_audit"]["status"] = "FAIL"
        descriptor = write_json(model_path, model)
        value["models"][0].update(descriptor)
    write_json(path, value)
    with pytest.raises(ValueError):
        token((path, value))


@pytest.mark.parametrize("name", ["source0.npz", "model1.json", "globalfreeze.json"])
def test_mutated_frozen_source_model_or_receipt_denies_query(source, frozen, name):
    freeze = token(frozen)
    changed = frozen[0].parent / name
    changed.write_bytes(changed.read_bytes() + b" ")
    with (
        archive.NativeArchive(
            source, partition(), event_sink=lambda _: pytest.fail("Mutation event"), profile=PROFILE
        ) as reader,
        pytest.raises(ValueError, match="changed"),
    ):
        reader.query(0, 17, freeze)


@pytest.mark.parametrize("case", ["compressed", "sha", "symlink", "hardlink"])
def test_native_identity_guards(tmp_path, case):
    path = tmp_path / "S004.npz"
    spec = make_archive(path, compressed=case == "compressed")
    if case == "sha":
        spec = archive.ArchiveSpec(path, 4, "0" * 64, spec.bytes)
    elif case == "symlink":
        alias = tmp_path / "alias.npz"
        alias.symlink_to(path)
        spec = archive.ArchiveSpec(alias, 4, spec.sha256, spec.bytes)
    elif case == "hardlink":
        os.link(path, tmp_path / "hard.npz")
    with (
        pytest.raises(ValueError),
        archive.NativeArchive(spec, partition(), event_sink=lambda _: None, profile=PROFILE),
    ):
        pass


@pytest.fixture
def metadata(tmp_path):
    envelope = {
        "schema": "cfeg.GENERATED.projection.v1",
        "study_id": "GENERATED-prefix-test",
        "plan_sha256": "0" * 64,
        "source_commit": "GENERATED",
        "start_sha256": "1" * 64,
        "manifest_sha256": "2" * 64,
        "returned_rows": 9360,
        "returned_packets": 780,
        "returned_subject_ids": list(old.SOURCE_IDS),
        "columns": ["six", "routing", "columns", "with", "fixed", "names"],
    }
    packets = []
    for pid in old.SOURCE_IDS:
        for interface in ("dry", "wet"):
            for block in range(10):
                packets.append(
                    {
                        "subject_id": pid,
                        "interface": interface,
                        "block_id": block,
                        "impedance_kohm": [None, 0, 1, 2, 3, 4, 5, block]
                        if block < 5
                        else ["FUTURE_EXPONENT"] * 8,
                        "headband_order": "dry",
                        "condition_period": "first" if interface == "dry" else "second",
                    }
                )
    path = tmp_path / "generated-projection.json"
    path.write_text(
        json.dumps({**envelope, "packets": packets}).replace('"FUTURE_EXPONENT"', "1e999")
    )
    return path, envelope


def test_metadata_event_before_numeric_prefix_conversion(metadata, monkeypatch):
    path, envelope = metadata
    events, decoded = [], []
    original = old._JSONSpans.decode

    def decode(self, node):
        if node.kind == "array" and len(node.children) == 8:
            assert events and events[-1]["kind"] == "metadata_support"
            assert events[-1]["phase"] == "before_decode"
            assert b"1e999" not in self.raw[node.start : node.end]
            decoded.append(node)
        return original(self, node)

    monkeypatch.setattr(old._JSONSpans, "decode", decode)
    with archive.SupportMetadata(
        path, sha(path), envelope, partition(), event_sink=events.append, profile=PROFILE
    ) as reader:
        assert not decoded and not events
        values, order = reader.support(4, 0, 3)
        assert len(decoded) == 3 and len(events) == 1 and order == 0
        assert np.isnan(values[:, 0]).all() and np.all(values[:, 1] == 0)
        assert not values.flags.writeable


def test_metadata_sink_failure_prevents_conversion(metadata, monkeypatch):
    path, envelope = metadata

    def sink(_):
        raise OSError("journal failed")

    with archive.SupportMetadata(
        path, sha(path), envelope, partition(), event_sink=sink, profile=PROFILE
    ) as reader:
        monkeypatch.setattr(
            old._JSONSpans, "decode", lambda *a: pytest.fail("Unjournaled numeric M")
        )
        with pytest.raises(OSError):
            reader.support(4, 0, 3)


def test_generated_profile_cannot_read_unmarked_metadata(metadata):
    path, envelope = metadata
    with pytest.raises(ValueError, match="Generated metadata envelope"):
        archive.SupportMetadata(
            path,
            sha(path),
            {**envelope, "study_id": "human"},
            partition(),
            event_sink=lambda _: None,
            profile=PROFILE,
        )


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/run_task_trca_n1_source39.py"
SPEC = importlib.util.spec_from_file_location("n1_runtime_leaf_test", SCRIPT)
runtime = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(runtime)


def manifest():
    parent = Path("/home/whwovy/task-trca-n1-source39-TOY")
    return {
        "schema": "cfeg.task_trca_n1_source39.generated.v1",
        "status": "GENERATED_FROZEN",
        "study_id": archive.STUDY_ID,
        "algorithm_schema": archive.SCHEMA,
        "attempt_id": "task-trca-n1-source39-generated1",
        "score_schema": archive.SCORE_SCHEMA,
        "generated": True,
        "seed": 20260914,
        "profile": PROFILE.record(),
        "source_ids": list(PROFILE.source_ids),
        "held60_authorized": False,
        "retired1_3_authorized": False,
        "old_attempts_reopened": False,
        "design": {
            "path": str(runtime.ROOT / "configs/analysis/task_trca_n1_source39_v1.json"),
            "sha256": runtime.DESIGN_SHA,
        },
        "native_root": str(parent / "inputs"),
        "native_plan": {"path": str(parent / "inputs/native_plan.json"), "sha256": "a" * 64},
        "fixture": {"path": str(parent / "inputs/fixture.json"), "sha256": "b" * 64, "bytes": 100},
        "output_root": str(parent / "task-trca-n1-source39-generated1"),
        "code_pins": dict.fromkeys(runtime.CODE_PATHS, "d" * 64),
        "device": "cpu",
        "backend": "batch",
        "precision": "float64",
        "cpu_threads": 1,
        "max_seconds": 1800,
        "optimizer_updates_max": 24000,
        "output_budget_bytes": 2 * 1024**3,
        "preflight": None,
        "resource_preflight": {"path": "/generated/resource.json", "sha256": "e" * 64, "bytes": 1},
    }


def test_manifest_scope_pins_and_profile(monkeypatch):
    monkeypatch.setattr(runtime, "sha", lambda _: "d" * 64)
    value = manifest()
    assert len(runtime.CODE_PATHS) == len(set(runtime.CODE_PATHS)) == 36
    assert runtime.validate_manifest(value, PROFILE) is value


@pytest.mark.parametrize(
    "field,value",
    [
        ("schema", "cfeg.task_trca_temporal.source39_execution.v1"),
        ("study_id", "task-trca-temporal-v1"),
        ("generated", False),
        ("seed", 73),
        ("held60_authorized", True),
        ("old_attempts_reopened", True),
        ("device", "cuda"),
        ("device", "automatic"),
        ("backend", "scalar"),
        ("precision", "float32"),
        ("cpu_threads", True),
        ("cpu_threads", 2),
        ("program_path", "/never/human.json"),
        ("program_sha256", "a" * 64),
        ("gpu_memory_fraction", 0.16),
        ("preflight", {"old": "authority"}),
        ("max_seconds", 1801),
        ("max_seconds", True),
        ("optimizer_updates_max", 24001),
        ("output_budget_bytes", 2 * 1024**3 + 1),
        ("native_root", "/never/human"),
        ("output_root", "/never/old-output"),
        ("source_ids", list(old.SOURCE_IDS)),
    ],
)
def test_scope_drift_rejected(field, value, monkeypatch):
    monkeypatch.setattr(runtime, "sha", lambda _: "d" * 64)
    document = manifest()
    document[field] = value
    with pytest.raises((ValueError, KeyError)):
        runtime.validate_manifest(document, PROFILE)


@pytest.mark.parametrize("change", ["missing", "extra", "wrong", "design", "fixture"])
def test_missing_wrong_or_aliased_pins_rejected(change, monkeypatch):
    monkeypatch.setattr(runtime, "sha", lambda _: "d" * 64)
    document = manifest()
    if change == "missing":
        document["code_pins"].pop(runtime.CODE_PATHS[0])
    elif change == "extra":
        document["code_pins"]["unexpected.py"] = "d" * 64
    elif change == "wrong":
        document["code_pins"][runtime.CODE_PATHS[0]] = "e" * 64
    elif change == "design":
        document["design"]["path"] = "/never/human.json"
    else:
        document["fixture"]["path"] = "/never/human.npz"
    with pytest.raises(ValueError):
        runtime.validate_manifest(document, PROFILE)


def test_rejected_human_manifest_never_reaches_fixture(tmp_path, monkeypatch):
    document = manifest()
    document["generated"] = False
    path = tmp_path / "manifest.json"
    pin = runtime.write_json(path, document)
    monkeypatch.setattr(runtime, "_cold_module", lambda: pytest.fail("Rejected fixture reached"))
    with pytest.raises(ValueError):
        runtime.run_generated(path, pin["sha256"])


def test_no_optimizer_warmup_step():
    # Static supplement to the live registered24k counter, not an Adam correctness proof.
    import ast

    tree = ast.parse(SCRIPT.read_text())
    function = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "_run")
    assert not any(
        isinstance(n, ast.Call)
        and isinstance(n.func, ast.Attribute)
        and n.func.attr in ("step", "backward")
        for n in ast.walk(function)
    )


def test_first_cpu_adam_update_under_actual_file_guard(tmp_path):
    code = """
import importlib.util, sys
from pathlib import Path
import torch
spec = importlib.util.spec_from_file_location('guarded_n1_toy', sys.argv[1])
runtime = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runtime)
torch.set_num_threads(1)
parameter = torch.nn.Parameter(torch.tensor([1.0], dtype=torch.float64))
optimizer = torch.optim.Adam([parameter], lr=.01)
output = Path(sys.argv[2])
runtime.install_guard([], output)
parameter.square().sum().backward()
optimizer.step()
assert 0 < float(parameter.detach()) < 1
try:
    open('/explicit/forbidden-input.npy', 'rb')
except PermissionError:
    pass
else:
    raise AssertionError('Forbidden input was allowed')
assert not torch.cuda.is_initialized()
print('ONE_TOY_UPDATE_GUARDED_NO_CUDA')
"""
    result = subprocess.run(
        [sys.executable, "-c", code, str(SCRIPT), str(tmp_path)],
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
        env={
            **os.environ,
            "OPENBLAS_NUM_THREADS": "1",
            "OMP_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "PYTHONDONTWRITEBYTECODE": "1",
        },
    )
    assert "ONE_TOY_UPDATE_GUARDED_NO_CUDA" in result.stdout


def test_old_temporal_token_rejected(source):
    from cfeg.analysis import task_trca_temporal_archive as previous

    obsolete = object.__new__(previous.AllModelsFrozen)
    with (
        archive.NativeArchive(
            source, partition(), event_sink=lambda _: None, profile=PROFILE
        ) as reader,
        pytest.raises(PermissionError),
    ):
        reader.query(0, 17, obsolete)


def test_failure_precedence(tmp_path):
    runtime._no_failure_receipts(tmp_path)
    runtime.write_json(tmp_path / "failure.json", {"failed": True})
    with pytest.raises(ValueError, match="preced"):
        runtime._no_failure_receipts(tmp_path)


def test_generated_summary_never_claims_efficacy():
    scores = np.zeros((9, 2, 1, 2, 10, 48, 12))
    a0 = np.zeros((9, 2, 1, 48, 12))
    summary = runtime.generated_summary(scores, a0, PROFILE)
    assert summary["metadata_effect"] == "NOT_EVALUATED"
    assert summary["terminal"] == "GENERATED_NOT_EVALUATED"


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


def test_old_n1_integration_token_is_not_new_experiment_authority(source):
    from cfeg.analysis import task_trca_n1_archive as previous

    obsolete = object.__new__(previous.AllModelsFrozen)
    with (
        archive.NativeArchive(
            source, partition(), event_sink=lambda _: None, profile=PROFILE
        ) as reader,
        pytest.raises(PermissionError),
    ):
        reader.query(0, 17, obsolete)


def test_exact_human_manifest_is_distinct_from_generated(monkeypatch):
    monkeypatch.setattr(runtime, "sha", lambda _: "d" * 64)
    value = manifest()
    value.update(
        schema="cfeg.task_trca_n1_source39.execution.v1",
        status="EXECUTION_FROZEN",
        generated=False,
        seed=None,
        profile=archive.HUMAN_PROFILE.record(),
        source_ids=list(archive.HUMAN_PROFILE.source_ids),
        fixture=None,
        preflight={"path": "/generated/cold_audit.json", "sha256": "e" * 64, "bytes": 1},
        native_root="/home/whwovy/metadata-prior-source39-v1/native-cold-r1",
        native_plan={
            "path": str(runtime.ROOT / "configs/analysis/metadata_prior_source39_v1.json"),
            "sha256": runtime.OLD_PLAN_SHA,
        },
        native_manifest_sha256=runtime.NATIVE_MANIFEST_SHA,
        output_root="/home/whwovy/task-trca-n1-source39-TOY/task-trca-n1-source39-primary1",
        attempt_id="task-trca-n1-source39-primary1",
        max_seconds=21600,
        output_budget_bytes=12 * 1024**3,
    )
    assert runtime.validate_manifest(value, archive.HUMAN_PROFILE) is value
    with pytest.raises(ValueError):
        runtime.validate_manifest(value, PROFILE)
    value["source_ids"] = value["source_ids"][:-1] + [103]
    with pytest.raises(ValueError):
        runtime.validate_manifest(value, archive.HUMAN_PROFILE)


def test_zero_byte_partial_failure_file_can_be_hashed(tmp_path):
    empty = tmp_path / "partial.npz"
    empty.touch()
    item = runtime.descriptor(empty)
    assert item["bytes"] == 0 and item["sha256"] == hashlib.sha256(b"").hexdigest()
