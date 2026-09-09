"""Generated-only selective-reader boundaries and durable publication tests."""

import hashlib
import json
import zipfile

import numpy as np
import pytest
from test_task_trca_shape_archive import metadata_builder  # noqa: F401

from cfeg.analysis import source_channel_margin_runtime as r
from cfeg.analysis import task_trca_shape_archive as low


@pytest.fixture(scope="module")
def selective_archive(tmp_path_factory):
    folder = tmp_path_factory.mktemp("selective")
    path = folder / "S004.npz"
    x = np.full((2, 10, 12, 5, 8, 250), np.nan)
    for interface in (0, 1):
        for block in (0, 1, 2, 5):
            x[interface, block] = interface * 10 + block
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_STORED) as output:
        for n in (125, 188, 250, 500):
            for prefix in ("x", "full", "a0"):
                name = f"{prefix}_{n}.npy"
                if name == "x_250.npy":
                    with output.open(name, "w", force_zip64=True) as stream:
                        np.lib.format.write_array(stream, x, allow_pickle=False)
                else:
                    output.writestr(name, b"forbidden NPY header/values must not be parsed")
    return r.archive.ArchiveSpec(path, 4, r.digest(path)["sha256"], path.stat().st_size)


def test_only_explicit_support_and_source_spans(selective_archive, monkeypatch):
    events, counts = [], []
    original = low._decode_float64

    def tracked(fd, count, offset, shape):
        counts.append((count, shape, events[-1]["kind"]))
        return original(fd, count, offset, shape)

    monkeypatch.setattr(low, "_decode_float64", tracked)
    with r.SourceOnlyArchive(selective_archive, event_sink=events.append) as reader:
        for i in (0, 1):
            assert reader.support3(i).shape == (3, 12, 5, 8, 250)
            assert np.all(reader.source_block5(i) == i * 10 + 5)
        assert not hasattr(reader, "query")
        assert not hasattr(reader, "full_correlations")
        assert not hasattr(reader, "a0_correlations")
        with pytest.raises(TypeError):
            reader.support3(0, 5)
        with pytest.raises(ValueError):
            reader.support3(True)
    assert len(counts) == len(events) == 4
    assert [e["blocks"] for e in events] == [[0, 1, 2], [5], [0, 1, 2], [5]]
    assert all(e["role"] == "source_extraction" for e in events)


def test_journal_error_denies_numeric_decode(selective_archive, monkeypatch):
    def fail(event):
        raise OSError("journal broken")

    monkeypatch.setattr(
        low, "_decode_float64", lambda *a: pytest.fail("decoded after journal failure")
    )
    with r.SourceOnlyArchive(selective_archive, event_sink=fail) as reader:
        with pytest.raises(OSError, match="journal broken"):
            reader.support3(0)
        with pytest.raises(OSError, match="journal broken"):
            reader.source_block5(0)


def test_metadata_prefix_and_zero_missing_semantics(metadata_builder):  # noqa: F811
    path, envelope = metadata_builder()
    events = []
    with r.SourceOnlyMetadata(
        path, r.digest(path)["sha256"], envelope, event_sink=events.append
    ) as metadata:
        packet, order = metadata.metadata3(4, 0)
        assert packet.shape == (3, 8) and order == 0
        assert np.isnan(packet[:, 0]).all() and np.all(packet[:, 1] == 0)
        assert events[0]["blocks"] == [0, 1, 2]
        with pytest.raises(PermissionError):
            metadata.metadata3(5, 0)


def test_metadata_journal_failure_prevents_numeric_packet_conversion(metadata_builder, monkeypatch):  # noqa: F811
    path, envelope = metadata_builder()

    def fail(event):
        raise OSError("journal failed")

    with r.SourceOnlyMetadata(
        path, r.digest(path)["sha256"], envelope, event_sink=fail
    ) as metadata:
        monkeypatch.setattr(low._JSONSpans, "decode", lambda *a: pytest.fail("packet decoded"))
        with pytest.raises(OSError):
            metadata.metadata3(4, 0)


def test_journal_exclusive_sealed_and_sequence(tmp_path):
    p = tmp_path / "events.jsonl"
    j = r.Journal(p, "a" * 64)
    j({"kind": "first"})
    j({"kind": "second"})
    j.close()
    rows = [json.loads(line) for line in p.read_text().splitlines()]
    assert [row["seq"] for row in rows] == [0, 1]
    assert p.stat().st_mode & 0o777 == 0o400
    with pytest.raises(FileExistsError):
        r.Journal(p, "a" * 64)


def test_pinned_json_and_npz(tmp_path):
    p = tmp_path / "record.json"
    r.publish(p, {"ok": True})
    assert r.read_json(p, hashlib.sha256(p.read_bytes()).hexdigest()) == {"ok": True}
    with pytest.raises(FileExistsError):
        r.publish(p, {})
    with pytest.raises(ValueError):
        r.read_json(p, "0" * 64)
    n = tmp_path / "source.npz"
    r.save_npz(n, example=np.arange(4))
    with np.load(n, allow_pickle=False) as data:
        np.testing.assert_array_equal(data["example"], np.arange(4))
    assert n.stat().st_mode & 0o777 == 0o400


@pytest.mark.parametrize("change", ["schema", "authority", "ids", "config", "attempts"])
def test_wrong_authority_denied_before_input_envelope(tmp_path, monkeypatch, change):
    registration = {
        "schema": r.SCHEMA,
        "human_execution_authorized": True,
        "parent": str(tmp_path),
        "repository": str(r.ROOT),
        "source_ids": list(r.archive.SOURCE_IDS),
        "config_sha256": r.CONFIG_SHA,
        "attempts": {"primary": 1, "audit": 1, "retries": 0},
    }
    key, bad = {
        "schema": ("schema", "old-generated"),
        "authority": ("human_execution_authorized", False),
        "ids": ("source_ids", [4]),
        "config": ("config_sha256", "0" * 64),
        "attempts": ("attempts", {}),
    }[change]
    registration[key] = bad
    path = tmp_path / "registration.json"
    r.publish(path, registration)
    monkeypatch.setattr(r, "input_envelope", lambda: pytest.fail("input read before authority"))
    with pytest.raises(PermissionError):
        r.validate_registration(tmp_path, r.digest(path)["sha256"])
