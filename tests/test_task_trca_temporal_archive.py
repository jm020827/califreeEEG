"""Generated-only archive boundary fixtures; no trained-model efficacy assertion."""

import hashlib
import json
import os
import zipfile

import numpy as np
import pytest

from cfeg.analysis import task_trca_shape_archive as old
from cfeg.analysis import task_trca_temporal_archive as archive

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


def test_exact_profiles_and_separate_generated_authority(source):
    assert archive.HUMAN_PROFILE.source_ids == old.SOURCE_IDS
    assert archive.HUMAN_PROFILE.samples == (125, 188, 250, 500)
    assert PROFILE.record() == {
        "source_ids": [4, 6, 8, 11, 14, 21, 22, 25, 28],
        "interfaces": [0, 1],
        "samples": [17],
        "budgets": [3, 5],
        "generated": True,
        "seed": 20260909,
    }
    with (
        pytest.raises(ValueError, match="archive members"),
        archive.NativeArchive(source, partition(), event_sink=lambda _: None),
    ):
        pass
    for kwargs in (
        {"samples": (17,)},
        {"source_ids": PROFILE.source_ids},
        {"generated": True},
        {"interfaces": (0,)},
    ):
        with pytest.raises(ValueError):
            archive.RuntimeProfile(**kwargs)


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
