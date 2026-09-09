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

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/audit_task_trca_n1_integration.py"
spec = importlib.util.spec_from_file_location("temporal_cold_test", SCRIPT)
cold = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cold)


@pytest.fixture(autouse=True)
def prohibit_registered_generation(monkeypatch):
    original = np.random.default_rng

    def guarded(seed=None):
        assert seed == 73, "Only nonregistered toy seed73 is permitted"
        return original(seed)

    monkeypatch.setattr(np.random, "default_rng", guarded)


def save(path, value):
    with path.open("x") as stream:
        json.dump(value, stream)
    path.chmod(0o400)
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


@pytest.mark.parametrize("profile,events,queries", [("generated", 325, 72)])
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
        path.chmod(0o600)
        path.write_text('{"generated":true,"generated":false}')
        path.chmod(0o400)
        binding["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
        binding["bytes"] = path.stat().st_size
    else:
        original = cold._json

        def mutate(raw):
            path.chmod(0o600)
            path.write_text('{"generated":false}')
            path.chmod(0o400)
            return original(raw)

        monkeypatch.setattr(cold, "_json", mutate)
    with pytest.raises(ValueError):
        cold.read_pinned(binding["path"], binding["sha256"], binding["bytes"])


def test_numpy_without_pickle_and_cold_publication(tmp_path):
    path = tmp_path / "generated.npz"
    with path.open("xb") as stream:
        np.savez(stream, example=np.array([1, 2]))
    path.chmod(0o400)
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
        "schema": "cfeg.task_trca_n1.all_models_frozen.v1",
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
    assert len(cold.CODE_PATHS) == len(set(cold.CODE_PATHS)) == 31
    assert "src/cfeg/analysis/metadata_prior_validation.py" in cold.CODE_PATHS
    tree = ast.parse(SCRIPT.read_text())
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module == "cfeg.analysis":
            assert {value.name for value in node.names} <= {
                "task_trca_n1_artifact_audit",
                "task_trca_n1_audit",
            }


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


@pytest.fixture
def manifest_bundle(tmp_path, monkeypatch):
    """Real tiny pinned envelope; source NPZ contents are never numeric-decoded."""
    repo = tmp_path / "repo"
    repo.mkdir()
    original_root = cold.ROOT
    configuration_path = "configs/analysis/metadata_prior_source39_v1.json"
    config = json.loads((original_root / configuration_path).read_text())
    pins = {}
    for name in cold.CODE_PATHS:
        path = repo / name
        path.parent.mkdir(parents=True, exist_ok=True)
        value = (
            (original_root / name).read_bytes()
            if name in (cold.DESIGN_PATH, configuration_path)
            else b"pinned toy code\n"
        )
        with path.open("xb") as stream:
            stream.write(value)
        pins[name] = hashlib.sha256(value).hexdigest()
    monkeypatch.setattr(cold, "ROOT", repo)
    parent = tmp_path / "task-trca-n1-integration-toy"
    parent.mkdir()
    inputs = parent / "inputs"
    inputs.mkdir()
    output = parent / "task-trca-n1-integration-primary1"
    output.mkdir()
    profile = cold.audit.profile_record()
    origin = {
        "schema": "cfeg.GENERATED.n1.projection.v1",
        "study_id": "GENERATED-task-trca-n1-integration-v1",
    }
    projection = save(inputs / "projection.json", {**origin, "packets": "not numerically decoded"})
    source = {
        "path": projection["path"],
        "sha256": projection["sha256"],
        "packets": 780,
        "returned_rows": 9360,
        "columns": config["source_projection"]["columns"],
        "envelope_provenance": origin,
    }
    plan = {
        "generated": True,
        "seed": 20260913,
        "interfaces": ["dry", "wet"],
        "frequencies": config["frequencies"],
        "native_weights": config["native_weights"],
        "source_projection": source,
    }
    plan_desc = save(inputs / "native_plan.json", plan)
    start = save(inputs / "start.json", {"generated": True, "seed": 20260913, "profile": profile})
    files = []
    for pid in profile["source_ids"]:
        path = inputs / f"S{pid:03d}.npz"
        value = f"non-numeric toy archive {pid}".encode()
        with path.open("xb") as stream:
            stream.write(value)
        path.chmod(0o400)
        files.append(
            {
                "subject": pid,
                "filename": path.name,
                "sha256": hashlib.sha256(value).hexdigest(),
                "bytes": len(value),
            }
        )
    native = {
        "status": "GENERATED_COMPLETE",
        "generated": True,
        "seed": 20260913,
        "plan_sha256": plan_desc["sha256"],
        "source_subject_ids": profile["source_ids"],
        "start_sha256": start["sha256"],
        "files": files,
    }
    native_desc = save(inputs / "result.json", native)
    fixture = {
        "status": "GENERATED_INPUTS_ONLY",
        "seed": 20260913,
        "profile": profile,
        "generator_sha256": pins["scripts/prepare_task_trca_n1_integration.py"],
        "native_function_sha256": pins["src/cfeg/analysis/metadata_trca_prior.py"],
        "native_configuration_sha256": pins[configuration_path],
        "result": native_desc,
        "plan": plan_desc,
        "projection": projection,
        "human_artifact_reads": False,
        "metadata_effect": "NOT_EVALUATED",
    }
    fixture_desc = save(inputs / "fixture.json", fixture)
    manifest = {
        "schema": "cfeg.task_trca_n1.generated_execution.v1",
        "study_id": cold.independent.SCHEMA,
        "score_schema": cold.independent.SCORE_SCHEMA,
        "attempt_id": output.name,
        "status": "GENERATED_FROZEN",
        "generated": True,
        "seed": 20260913,
        "profile": profile,
        "source_ids": profile["source_ids"],
        "held60_authorized": False,
        "retired1_3_authorized": False,
        "old_attempts_reopened": False,
        "design": {"path": str(repo / cold.DESIGN_PATH), "sha256": cold.DESIGN_SHA},
        "native_plan": {key: plan_desc[key] for key in ("path", "sha256")},
        "native_root": str(inputs),
        "native_manifest_sha256": native_desc["sha256"],
        "native_weights": {
            name: [config["native_weights"][name][i] for i in ("dry", "wet")]
            for name in ("ETRCA", "A0_author")
        },
        "code_pins": pins,
        "device": "cpu",
        "backend": "batch",
        "precision": "float64",
        "cpu_threads": 1,
        "output_root": str(output),
        "output_budget_bytes": cold.INPUT_LIMIT
        - sum(p.stat().st_size for p in inputs.iterdir())
        - 2 * 1024**2,
        "max_seconds": 7200,
        "optimizer_updates_max": 24000,
        "preflight": None,
        "fixture": fixture_desc,
    }
    bound = save(parent / "manifest.json", manifest)
    return {
        "manifest": manifest,
        "binding": bound,
        "fixture": fixture,
        "native": native,
        "parent": parent,
        "inputs": inputs,
        "output": output,
    }


def test_complete_generated_manifest_and_fixture_hash_only(manifest_bundle, monkeypatch):
    monkeypatch.setattr(np, "load", lambda *_args, **_kwargs: pytest.fail("Source numeric decode"))
    profile, contract = cold.validate_manifest(
        manifest_bundle["manifest"], manifest_bundle["binding"]["sha256"]
    )
    assert profile == "generated" and contract["seed"] == 20260913


@pytest.mark.parametrize(
    "defect",
    [
        "human_schema",
        "human_profile",
        "old_schema",
        "device",
        "program",
        "native_path",
        "plan_path",
        "fixture_path",
        "extra_pin",
        "missing_pin",
        "bad_pin",
        "seed",
        "weights",
        "budget",
        "bool_budget",
        "preflight",
    ],
)
def test_manifest_authority_and_binding_mutations(manifest_bundle, monkeypatch, defect):
    value = copy.deepcopy(manifest_bundle["manifest"])
    if defect == "human_schema":
        value["schema"] = "cfeg.task_trca_temporal.source39_execution.v1"
    elif defect == "human_profile":
        value["profile"]["generated"] = False
    elif defect == "old_schema":
        value["study_id"] = "task-trca-temporal-v1"
    elif defect == "device":
        value["device"] = "cuda"
    elif defect == "program":
        value["program_path"] = "/forbidden/human/program.json"
    elif defect == "native_path":
        value["native_root"] = "/forbidden/human"
    elif defect == "plan_path":
        value["native_plan"]["path"] = "/forbidden/human/native_plan.json"
    elif defect == "fixture_path":
        value["fixture"]["path"] = "/forbidden/human/fixture.json"
    elif defect == "extra_pin":
        value["code_pins"]["arbitrary.py"] = "f" * 64
    elif defect == "missing_pin":
        value["code_pins"].pop(next(iter(value["code_pins"])))
    elif defect == "bad_pin":
        value["code_pins"]["scripts/prepare_task_trca_n1_integration.py"] = "0" * 64
    elif defect == "seed":
        value["seed"] = 73
    elif defect == "weights":
        value["native_weights"]["ETRCA"][0][0] += 1
    elif defect == "budget":
        value["optimizer_updates_max"] = 24001
    elif defect == "bool_budget":
        value["cpu_threads"] = True
    else:
        value["preflight"] = {"old_human_authority": True}
    original = cold.read_pinned

    def guarded(path, *args, **kwargs):
        assert not str(path).startswith("/forbidden/"), "Unauthorized path opened"
        return original(path, *args, **kwargs)

    monkeypatch.setattr(cold, "read_pinned", guarded)
    with pytest.raises((ValueError, AssertionError)):
        cold.validate_manifest(value, manifest_bundle["binding"]["sha256"])


@pytest.mark.parametrize(
    "defect", ["missing", "extra", "writable", "hardlink", "symlink", "prep_failure", "tamper"]
)
def test_fixture_inventory_and_byte_integrity_before_source_decode(
    manifest_bundle, monkeypatch, defect
):
    path = manifest_bundle["inputs"] / "S004.npz"
    if defect == "missing":
        path.rename(path.parent.parent / "removed-toy.npz")
    elif defect == "extra":
        save(path.parent / "unexpected.json", {})
    elif defect == "writable":
        path.chmod(0o600)
    elif defect == "hardlink":
        os.link(path, path.parent.parent / "hardlink.npz")
    elif defect == "symlink":
        target = path.parent.parent / "moved.npz"
        path.rename(target)
        path.symlink_to(target)
    elif defect == "prep_failure":
        save(path.parent.parent / "prep_failure.json", {})
    else:
        path.chmod(0o600)
        with path.open("wb") as stream:
            stream.write(b"corruption")
        path.chmod(0o400)
    monkeypatch.setattr(np, "load", lambda *_args, **_kwargs: pytest.fail("Source numeric decode"))
    with pytest.raises(ValueError):
        cold.validate_manifest(manifest_bundle["manifest"], manifest_bundle["binding"]["sha256"])
