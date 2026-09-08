import hashlib
import io
import os
import struct
import zipfile

import numpy as np
import pytest

from cfeg.analysis.native_support_prefix import pread_exact, verified_archive


def make_archive(path, *, dtype="<f8", fortran=False, compressed=False, extra_key=False):
    x = np.full((2, 10, 12, 5, 8, 3), np.nan, dtype=dtype)
    x[0, :5] = 17
    x[1, :5] = 23
    if fortran:
        x = np.asfortranarray(x)
    kwargs = {"x_3": x, "full_3": np.array([np.nan]), "a0_3": np.array([np.nan])}
    if extra_key:
        kwargs["unapproved"] = np.ones(1)
    # Exercise explicit ZIP64 sentinels independently of stdlib FileHeader policy.
    compression = zipfile.ZIP_DEFLATED if compressed else zipfile.ZIP_STORED
    with zipfile.ZipFile(path, "w", compression=compression) as archive:
        for name, array in kwargs.items():
            with archive.open(name + ".npy", "w", force_zip64=True) as stream:
                np.lib.format.write_array(stream, array, allow_pickle=False)
    with zipfile.ZipFile(path) as archive:
        offsets = [entry.header_offset for entry in archive.infolist()]
    raw = bytearray(path.read_bytes())
    for offset in offsets:
        struct.pack_into("<II", raw, offset + 18, 0xFFFFFFFF, 0xFFFFFFFF)
    path.write_bytes(raw)
    return hashlib.sha256(path.read_bytes()).hexdigest(), path.stat().st_size


def test_only_support_is_decoded_and_zip64_offsets(tmp_path, monkeypatch):
    path = tmp_path / "artificial.npz"
    digest, size = make_archive(path)
    # This fixture deliberately includes ZIP64 size sentinels and extra fields.
    assert struct.unpack_from("<I", path.read_bytes(), 18)[0] == 0xFFFFFFFF
    reads = []
    original = np.frombuffer

    def record(buffer, *args, **kwargs):
        reads.append(len(buffer))
        return original(buffer, *args, **kwargs)

    monkeypatch.setattr(np, "frombuffer", record)
    monkeypatch.setattr(np, "load", lambda *a, **k: pytest.fail("np.load forbidden"))
    with verified_archive(path, digest, size, [3]) as load:
        x = load(3)
        assert x.shape == (2, 5, 12, 5, 8, 3)
        assert (x[0] == 17).all() and (x[1] == 23).all()
    assert reads == [5 * 12 * 5 * 8 * 3 * 8] * 2
    assert hashlib.sha256(path.read_bytes()).hexdigest() == digest


@pytest.mark.parametrize(
    "kwargs", [{"dtype": "<f4"}, {"fortran": True}, {"compressed": True}, {"extra_key": True}]
)
def test_reject_unfrozen_layout(tmp_path, kwargs):
    path = tmp_path / "artificial.npz"
    digest, size = make_archive(path, **kwargs)
    with pytest.raises(ValueError), verified_archive(path, digest, size, [3]) as load:
        load(3)


@pytest.mark.parametrize("case", ["hash", "size", "symlink", "hardlink"])
def test_reject_identity_and_path_errors(tmp_path, case):
    path = tmp_path / "artificial.npz"
    digest, size = make_archive(path)
    if case == "hash":
        digest = "0" * 64
    elif case == "size":
        size += 1
    elif case == "symlink":
        alias = tmp_path / "alias.npz"
        alias.symlink_to(path)
        path = alias
    else:
        os.link(path, tmp_path / "alias.npz")
    with pytest.raises(ValueError), verified_archive(path, digest, size, [3]):
        pass


def test_reject_mutation_during_work(tmp_path):
    path = tmp_path / "artificial.npz"
    digest, size = make_archive(path)
    with (
        pytest.raises(ValueError, match="changed|SHA mismatch"),
        verified_archive(path, digest, size, [3]) as load,
    ):
        load(3)
        with path.open("r+b") as stream:
            stream.seek(size - 1)
            stream.write(b"x")


def test_short_pread_loops_and_truncation(tmp_path, monkeypatch):
    path = tmp_path / "bytes"
    path.write_bytes(b"abcdefghij")
    original = os.pread
    monkeypatch.setattr(os, "pread", lambda fd, count, offset: original(fd, min(count, 2), offset))
    with path.open("rb") as stream:
        assert pread_exact(stream.fileno(), 10, 0) == b"abcdefghij"
        with pytest.raises(ValueError, match="Truncated"):
            pread_exact(stream.fileno(), 11, 0)


@pytest.mark.parametrize(
    "case",
    [
        "duplicate",
        "wrong_shape",
        "nonfinite_support",
        "version",
        "npy_keys",
        "npy_header",
        "local_name",
        "zip64",
    ],
)
def test_reject_poisoned_archive_structure(tmp_path, case):
    path = tmp_path / "artificial.npz"
    make_archive(path)
    if case == "duplicate":
        with zipfile.ZipFile(path, "a") as archive, pytest.warns(UserWarning):
            archive.writestr("x_3.npy", b"duplicate")
    else:
        raw = bytearray(path.read_bytes())
        npy = raw.index(b"\x93NUMPY")
        if case == "wrong_shape":
            start = raw.index(b"(2, 10, 12, 5, 8, 3)")
            raw[start : start + 2] = b"(3"
        elif case == "nonfinite_support":
            start = npy + 10 + struct.unpack_from("<H", raw, npy + 8)[0]
            raw[start : start + 8] = struct.pack("<d", float("nan"))
        elif case == "version":
            raw[npy + 6] = 2
        elif case == "npy_keys":
            start = raw.index(b"'descr'")
            raw[start : start + 7] = b"'other'"
        elif case == "npy_header":
            raw[npy + 8 : npy + 10] = struct.pack("<H", 5000)
        elif case == "local_name":
            raw[30] = ord("y")
        elif case == "zip64":
            name_len = struct.unpack_from("<H", raw, 26)[0]
            raw[30 + name_len + 4] ^= 1
        path.write_bytes(raw)
    digest, size = hashlib.sha256(path.read_bytes()).hexdigest(), path.stat().st_size
    with (
        pytest.raises((ValueError, SyntaxError)),
        verified_archive(path, digest, size, [3]) as load,
    ):
        load(3)


def test_no_other_array_headers_or_values_decoded(tmp_path):
    path = tmp_path / "artificial.npz"
    x = np.ones((2, 10, 12, 5, 8, 3))
    buffer = io.BytesIO()
    np.save(buffer, x)
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("x_3.npy", buffer.getvalue())
        archive.writestr("full_3.npy", b"not even a valid NPY header")
        archive.writestr("a0_3.npy", b"not a valid score array")
    with verified_archive(
        path, hashlib.sha256(path.read_bytes()).hexdigest(), path.stat().st_size, [3]
    ) as load:
        assert np.all(load(3) == 1)
