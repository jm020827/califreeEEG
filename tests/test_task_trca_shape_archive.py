"""Artificial-only fixtures for role, byte-range, freeze, and metadata boundaries."""

import hashlib
import json
import os
import zipfile

import numpy as np
import pytest

from cfeg.analysis import task_trca_shape_archive as core
from cfeg.analysis.task_trca_shape_inputs import RolePartition


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def outer(fold=0):
    return RolePartition(
        tuple(v for rank, v in enumerate(core.SOURCE_IDS) if rank % 3 != fold),
        (),
        tuple(v for rank, v in enumerate(core.SOURCE_IDS) if rank % 3 == fold),
    )


@pytest.fixture
def archive_builder(tmp_path, monkeypatch):
    # Preserve the real channel/class/block/band layout; reduce only N and window
    # multiplicity for bounded artificial fixture disk/CPU use.
    monkeypatch.setattr(core, "SAMPLE_COUNTS", (13,))
    counter = [0]

    def build(
        *,
        dtype="<f8",
        fortran=False,
        compressed=False,
        extra=False,
        poison_support=False,
        poison_future=False,
        invalid_score_header=False,
        pid=4,
    ):
        counter[0] += 1
        path = tmp_path / f"artificial{counter[0]}.npz"
        x = np.empty((2, 10, 12, 5, 8, 13), dtype=dtype)
        for interface in (0, 1):
            for block in range(10):
                x[interface, block] = interface * 10 + block
        if poison_support:
            x[:, 2] = np.nan
        if poison_future:
            x[:, 5:] = np.nan
        if fortran:
            x = np.asfortranarray(x)
        full = np.empty((2, 2, 4, 12, 5, 12))
        for interface in (0, 1):
            for budget in (0, 1):
                full[interface, budget] = 0.1 * (interface * 2 + budget)
        a0 = np.empty((2, 4, 12, 5, 12))
        a0[0], a0[1] = 0.7, 0.8
        compression = zipfile.ZIP_DEFLATED if compressed else zipfile.ZIP_STORED
        with zipfile.ZipFile(path, "w", compression=compression) as archive:
            for name, array in (("x_13", x), ("full_13", full), ("a0_13", a0)):
                with archive.open(name + ".npy", "w", force_zip64=True) as stream:
                    if invalid_score_header and name != "x_13":
                        stream.write(b"denied score bytes must not be decoded")
                    else:
                        np.lib.format.write_array(stream, array, allow_pickle=False)
            if extra:
                archive.writestr("extra.npy", b"unapproved")
        return core.ArchiveSpec(path, pid, digest(path), path.stat().st_size)

    return build


@pytest.fixture
def freeze_builder(tmp_path):
    counter = [0]

    def build(change=None):
        counter[0] += 1
        root = tmp_path / f"freeze{counter[0]}"
        root.mkdir()
        models = []
        for fold in range(3):
            model = root / f"model{fold}.json"
            model.write_text(json.dumps({"pipeline": {"artificial": True}, "fold": fold}))
            partition = outer(fold)
            models.append(
                {
                    "fold_id": fold,
                    "fit_ids": list(partition.fit_ids),
                    "evaluation_ids": list(partition.evaluation_ids),
                    "path": str(model),
                    "sha256": digest(model),
                    "bytes": model.stat().st_size,
                }
            )
        document = {
            "schema": "cfeg.task_trca_shape.all_models_frozen.v1",
            "status": "ALL_MODELS_FROZEN",
            "source_ids": list(core.SOURCE_IDS),
            "manifest_sha256": "a" * 64,
            "query_access_count": 0,
            "models": models,
        }
        if change:
            change(document)
        path = root / "all-models-frozen.json"
        path.write_text(json.dumps(document))
        return path, document

    return build


def freeze_token(builder):
    path, document = builder()
    return core.verify_freeze(path, digest(path), "a" * 64), path, document


def test_fixed_source_and_sample_allowlists():
    assert core.SAMPLE_COUNTS == (125, 188, 250, 500)
    assert len(core.SOURCE_IDS) == 39
    assert not {1, 2, 3, 5, 7, 9, 10} & set(core.SOURCE_IDS)


@pytest.mark.parametrize("role", ["fit", "validation", "evaluation"])
@pytest.mark.parametrize("k", [3, 5])
def test_support_exact_prefix_no_future_or_score_decode(archive_builder, monkeypatch, role, k):
    spec = archive_builder(poison_future=True, invalid_score_header=True)
    groups = {"fit_ids": (), "validation_ids": (), "evaluation_ids": ()}
    groups[role + "_ids"] = (4,)
    partition = RolePartition(**groups)
    decoded = []
    original = core._decode_float64

    def record(fd, count, offset, shape):
        decoded.append(shape)
        return original(fd, count, offset, shape)

    monkeypatch.setattr(core, "_decode_float64", record)
    monkeypatch.setattr(np, "load", lambda *args, **kw: pytest.fail("Whole-array load forbidden"))
    with core.NativeArchive(spec, partition) as reader:
        for interface in (0, 1):
            result = reader.support(interface, 13, k)
            assert result.shape == (k, 12, 5, 8, 13)
            for block in range(k):
                assert np.all(result[block] == interface * 10 + block)
            assert not result.flags.writeable
        assert reader.access_log == [
            {
                "participant_id": 4,
                "role": role,
                "kind": "support",
                "interface": interface,
                "samples": 13,
                "blocks": list(range(k)),
            }
            for interface in (0, 1)
        ]
    assert decoded == [(k, 12, 5, 8, 13)] * 2


@pytest.mark.parametrize("role", ["fit", "validation"])
def test_supervision_block5_only(archive_builder, role):
    spec = archive_builder()
    groups = {"fit_ids": (), "validation_ids": (), "evaluation_ids": ()}
    groups[role + "_ids"] = (4,)
    with core.NativeArchive(spec, RolePartition(**groups)) as reader:
        assert np.all(reader.supervision(1, 13) == 15)
        assert reader.access_log[0]["blocks"] == [5]


@pytest.mark.parametrize(
    "method,args",
    [
        ("supervision", (0, 13)),
        ("query", (0, 13)),
        ("full_correlations", (0, 13, 3)),
        ("a0_correlations", (0, 13)),
    ],
)
def test_denied_before_header_or_array_decode(archive_builder, monkeypatch, method, args):
    spec = archive_builder(poison_future=True, invalid_score_header=True)
    with core.NativeArchive(spec, outer()) as reader:
        monkeypatch.setattr(
            core, "_npy_data", lambda *a, **k: pytest.fail("Denied NPY header touched")
        )
        monkeypatch.setattr(
            core, "_decode_float64", lambda *a, **k: pytest.fail("Denied values decoded")
        )
        with pytest.raises(PermissionError):
            getattr(reader, method)(*args)
        assert not reader.access_log


def test_query_and_native_correlations_only_after_all_model_freeze(archive_builder, freeze_builder):
    spec = archive_builder()
    token, _, _ = freeze_token(freeze_builder)
    with core.NativeArchive(spec, outer()) as reader:
        for interface in (0, 1):
            query = reader.query(interface, 13, token)
            for block in range(4):
                assert np.all(query[block] == interface * 10 + block + 6)
            for k in (3, 5):
                expected = 0.1 * (interface * 2 + (k == 5))
                assert np.all(reader.full_correlations(interface, 13, k, token) == expected)
            assert np.all(reader.a0_correlations(interface, 13, token) == (0.7, 0.8)[interface])
        assert len(reader.access_log) == 8
        assert all(
            v["blocks"] == [6, 7, 8, 9] and v["freeze_sha256"] == token.receipt_sha256
            for v in reader.access_log
        )
        with pytest.raises(PermissionError):
            reader.supervision(0, 13)


def test_query_requires_exact_outer_roles(archive_builder, freeze_builder, monkeypatch):
    spec = archive_builder()
    token, _, _ = freeze_token(freeze_builder)
    for partition in (RolePartition((4,)), RolePartition((), (4,)), RolePartition((), (), (4,))):
        with core.NativeArchive(spec, partition) as reader:
            monkeypatch.setattr(
                core, "_decode_float64", lambda *a, **k: pytest.fail("Denied decode")
            )
            with pytest.raises(PermissionError):
                reader.query(0, 13, token)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"dtype": "<f4"},
        {"fortran": True},
        {"compressed": True},
        {"extra": True},
        {"poison_support": True},
    ],
)
def test_archive_rejects_layout_or_nonfinite_support(archive_builder, kwargs):
    spec = archive_builder(**kwargs)
    with pytest.raises(ValueError), core.NativeArchive(spec, RolePartition((4,))) as reader:
        reader.support(0, 13, 3)


@pytest.mark.parametrize("case", ["sha", "size", "symlink", "hardlink", "parent_alias"])
def test_archive_pins_and_aliases(archive_builder, tmp_path, case):
    spec = archive_builder()
    path, sha, size = spec.path, spec.sha256, spec.bytes
    if case == "sha":
        sha = "0" * 64
    elif case == "size":
        size += 1
    elif case == "symlink":
        path = tmp_path / "alias.npz"
        path.symlink_to(spec.path)
    elif case == "parent_alias":
        alias = tmp_path / "parent"
        alias.symlink_to(tmp_path, target_is_directory=True)
        path = alias / spec.path.name
    else:
        os.link(spec.path, tmp_path / "hard.npz")
    with (
        pytest.raises(ValueError),
        core.NativeArchive(core.ArchiveSpec(path, 4, sha, size), RolePartition((4,))),
    ):
        pass


@pytest.mark.parametrize("case", ["mutate", "replace", "link_during"])
def test_archive_post_work_mutation_guards(archive_builder, tmp_path, case):
    spec = archive_builder()
    with (
        pytest.raises(ValueError, match="changed|replaced"),
        core.NativeArchive(spec, RolePartition((4,))) as reader,
    ):
        reader.support(0, 13, 3)
        if case == "mutate":
            with spec.path.open("r+b") as stream:
                stream.seek(-1, 2)
                stream.write(b"x")
        elif case == "replace":
            replacement = tmp_path / "replacement"
            replacement.write_bytes(spec.path.read_bytes())
            os.replace(replacement, spec.path)
        else:
            os.link(spec.path, tmp_path / "created-hardlink")


@pytest.mark.parametrize("pid", [1, 2, 3, 5, True, 4.0, "4"])
def test_outside_source_denied_before_open(pid, monkeypatch):
    monkeypatch.setattr(core, "regular_fd", lambda *a: pytest.fail("Disallowed path opened"))
    with pytest.raises(ValueError):
        core.ArchiveSpec("does-not-exist", pid, "a" * 64, 8)


def test_role_exclusion_denied_before_open(archive_builder, monkeypatch):
    spec = archive_builder()
    monkeypatch.setattr(core, "regular_fd", lambda *a: pytest.fail("Disallowed path opened"))
    with pytest.raises(PermissionError):
        core.NativeArchive(spec, RolePartition((6,)))
    with pytest.raises(ValueError):
        core.NativeArchive(spec, RolePartition((4,), (), (5,)))


@pytest.mark.parametrize(
    "method,args",
    [
        ("support", (0, 13, 4)),
        ("support", (0, 13, True)),
        ("support", (True, 13, 3)),
        ("support", (0, 14, 3)),
    ],
)
def test_invalid_support_requests_before_decode(archive_builder, monkeypatch, method, args):
    with core.NativeArchive(archive_builder(), RolePartition((4,))) as reader:
        monkeypatch.setattr(core, "_decode_float64", lambda *a, **k: pytest.fail("Invalid decode"))
        with pytest.raises(ValueError):
            getattr(reader, method)(*args)


@pytest.mark.parametrize(
    "case",
    [
        "status",
        "queries",
        "query_bool",
        "source",
        "missing_model",
        "fold",
        "partition",
        "model_sha",
        "model_size",
        "manifest",
        "extra",
        "duplicate_model_path",
    ],
)
def test_invalid_freeze_receipts(freeze_builder, case):
    def change(doc):
        if case == "status":
            doc["status"] = "FITTING"
        elif case == "queries":
            doc["query_access_count"] = 1
        elif case == "query_bool":
            doc["query_access_count"] = False
        elif case == "source":
            doc["source_ids"][0] = 5
        elif case == "missing_model":
            doc["models"].pop()
        elif case == "fold":
            doc["models"][1]["fold_id"] = 0
        elif case == "partition":
            doc["models"][0]["evaluation_ids"].pop()
        elif case == "model_sha":
            doc["models"][0]["sha256"] = "0" * 64
        elif case == "model_size":
            doc["models"][0]["bytes"] += 1
        elif case == "manifest":
            doc["manifest_sha256"] = "b" * 64
        elif case == "extra":
            doc["override"] = True
        elif case == "duplicate_model_path":
            for key in ("path", "sha256", "bytes"):
                doc["models"][1][key] = doc["models"][0][key]

    path, _ = freeze_builder(change)
    with pytest.raises(ValueError):
        core.verify_freeze(path, digest(path), "a" * 64)


@pytest.mark.parametrize("target", ["receipt", "model"])
def test_freeze_file_mutation_denies_query(archive_builder, freeze_builder, monkeypatch, target):
    token, path, document = freeze_token(freeze_builder)
    changed = path if target == "receipt" else core.Path(document["models"][2]["path"])
    changed.write_bytes(changed.read_bytes() + b" ")
    with core.NativeArchive(archive_builder(), outer()) as reader:
        monkeypatch.setattr(core, "_npy_data", lambda *a: pytest.fail("Query header touched"))
        with pytest.raises(ValueError, match="changed|replaced"):
            reader.query(0, 13, token)


def test_token_cannot_be_directly_constructed():
    with pytest.raises(ValueError, match="verify_freeze"):
        core.AllModelsFrozen(None, "a" * 64, "b" * 64, [], [])


@pytest.fixture
def metadata_builder(tmp_path):
    counter = [0]

    def build(change=None, *, poison_future=True):
        counter[0] += 1
        path = tmp_path / f"artificial-metadata{counter[0]}.json"
        envelope = {
            "schema": "cfeg.context-template-source.projection.v1",
            "study_id": "artificial-only",
            "plan_sha256": "a" * 64,
            "source_commit": "b" * 40,
            "start_sha256": "c" * 64,
            "manifest_sha256": "d" * 64,
            "returned_rows": 9360,
            "returned_packets": 780,
            "returned_subject_ids": list(core.SOURCE_IDS),
            "columns": [
                "subject_id",
                "electrode_type",
                "run_id",
                "impedance_kohm_by_channel",
                "headband_order",
                "condition_period",
            ],
        }
        packets = []
        for pid in core.SOURCE_IDS:
            first = "dry" if pid % 2 == 0 else "wet"
            for interface in core.INTERFACES:
                for block in range(10):
                    values = [None, 0, 1, 2, 3, 4, 5, block]
                    if poison_future and block >= 5:
                        values = ["FUTURE_EXPONENT"] * 8
                    packets.append(
                        {
                            "subject_id": pid,
                            "interface": interface,
                            "block_id": block,
                            "impedance_kohm": values,
                            "headband_order": first,
                            "condition_period": "first" if first == interface else "second",
                        }
                    )
        document = {**envelope, "packets": packets}
        if change:
            change(document)
        raw = json.dumps(document, separators=(",", ":")).replace('"FUTURE_EXPONENT"', "1e999")
        path.write_text(raw)
        return path, envelope

    return build


@pytest.mark.parametrize("k", [3, 5])
@pytest.mark.parametrize("role", ["fit", "validation", "evaluation"])
def test_metadata_decodes_only_requested_prefix(metadata_builder, monkeypatch, k, role):
    path, envelope = metadata_builder()
    groups = {"fit_ids": (), "validation_ids": (), "evaluation_ids": ()}
    groups[role + "_ids"] = (4,)
    converted = []
    original = core._JSONSpans.decode

    def track(self, node):
        if node.kind == "array" and len(node.children) == 8:
            raw = self.raw[node.start : node.end]
            assert b"1e999" not in raw, "Future impedance converted"
            converted.append(raw)
        return original(self, node)

    monkeypatch.setattr(core._JSONSpans, "decode", track)
    with core.SupportMetadata(path, digest(path), envelope, RolePartition(**groups)) as reader:
        assert not converted
        result, order = reader.support(4, 1, k)
        assert len(converted) == k
        assert result.shape == (k, 8) and order == 0
        assert np.isnan(result[:, 0]).all() and np.all(result[:, 1] == 0)
        assert np.array_equal(result[:, -1], np.arange(k))
        assert not result.flags.writeable
        assert reader.access_log == [
            {
                "participant_id": 4,
                "role": role,
                "kind": "metadata_support",
                "interface": 1,
                "blocks": list(range(k)),
            }
        ]


@pytest.mark.parametrize(
    "args,exception",
    [
        ((6, 0, 3), PermissionError),
        ((5, 0, 3), PermissionError),
        ((4, True, 3), ValueError),
        ((4, 0, 6), ValueError),
        ((4, 0, True), ValueError),
    ],
)
def test_metadata_denied_before_conversion(metadata_builder, monkeypatch, args, exception):
    path, envelope = metadata_builder()
    with core.SupportMetadata(path, digest(path), envelope, RolePartition((4,))) as reader:
        monkeypatch.setattr(core._JSONSpans, "decode", lambda *a: pytest.fail("Denied M converted"))
        with pytest.raises(exception):
            reader.support(*args)


@pytest.mark.parametrize(
    "case",
    [
        "source",
        "duplicate",
        "missing",
        "block",
        "pid_bool",
        "period",
        "order",
        "packet_extra",
        "top_extra",
        "envelope",
        "impedance_string",
        "impedance_length",
    ],
)
def test_metadata_routing_and_envelope_rejected_without_values(metadata_builder, case):
    def change(doc):
        packet = doc["packets"][0]
        if case == "source":
            packet["subject_id"] = 5
        elif case == "duplicate":
            doc["packets"][1] = packet
        elif case == "missing":
            doc["packets"].pop()
        elif case == "block":
            packet["block_id"] = 10
        elif case == "pid_bool":
            packet["subject_id"] = True
        elif case == "period":
            packet["condition_period"] = "second"
        elif case == "order":
            packet["headband_order"] = "other"
        elif case == "packet_extra":
            packet["unapproved"] = 1
        elif case == "top_extra":
            doc["unapproved"] = 1
        elif case == "envelope":
            doc["returned_rows"] = 9361
        elif case == "impedance_string":
            packet["impedance_kohm"][0] = "not-numeric"
        elif case == "impedance_length":
            packet["impedance_kohm"].pop()

    path, envelope = metadata_builder(change)
    with (
        pytest.raises(ValueError),
        core.SupportMetadata(path, digest(path), envelope, RolePartition((4,))),
    ):
        pass


@pytest.mark.parametrize("bad", [-1, "SUPPORT_EXPONENT"])
def test_nonfinite_or_negative_authorized_m_fails(metadata_builder, bad):
    path, envelope = metadata_builder(
        lambda doc: doc["packets"][0]["impedance_kohm"].__setitem__(1, bad)
    )
    if bad == "SUPPORT_EXPONENT":
        path.write_text(path.read_text().replace('"SUPPORT_EXPONENT"', "1e999"))
    with (
        core.SupportMetadata(path, digest(path), envelope, RolePartition((4,))) as reader,
        pytest.raises(ValueError, match="nonnegative finite"),
    ):
        reader.support(4, 0, 3)


def test_metadata_file_mutation_guard(metadata_builder):
    path, envelope = metadata_builder()
    with (
        pytest.raises(ValueError, match="changed"),
        core.SupportMetadata(path, digest(path), envelope, RolePartition((4,))) as reader,
    ):
        reader.support(4, 0, 3)
        path.write_bytes(path.read_bytes() + b" ")


@pytest.mark.parametrize(
    "raw",
    [
        b'{"a":1,"a":2}',
        b'{"a":01}',
        b'{"a":NaN}',
        b'{"a":Infinity}',
        b'{"a":1,}',
        b"[1,]",
        b'{"a":"bad\\q"}',
        b'{"a":1} x',
        b"[[[[[[[[[[[[[[0]]]]]]]]]]]]]]",
        b'{"a":1e}',
        b'{"a":truefalse}',
    ],
)
def test_structural_json_scan_rejects_invalid(raw):
    with pytest.raises((ValueError, json.JSONDecodeError)):
        core._JSONSpans(raw)


def test_json_scan_leaves_numeric_lexemes_unconverted():
    parser = core._JSONSpans(b'{"x":[1e999,-1e999,null,0,1.5]}')
    values = parser.root.children["x"].children
    assert [node.kind for node in values] == ["number", "number", "null", "number", "number"]
    assert parser.raw[values[0].start : values[0].end] == b"1e999"


def test_reader_methods_fail_after_context_closed(archive_builder, metadata_builder):
    with core.NativeArchive(archive_builder(), RolePartition((4,))) as archive:
        pass
    with pytest.raises(ValueError, match="not open"):
        archive.support(0, 13, 3)
    path, envelope = metadata_builder()
    with core.SupportMetadata(path, digest(path), envelope, RolePartition((4,))) as metadata:
        pass
    with pytest.raises(ValueError, match="not open"):
        metadata.support(4, 0, 3)
