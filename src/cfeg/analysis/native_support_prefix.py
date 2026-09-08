"""Read only native-cache support values; whole-file hashing is allowed.

ZIP64-aware direct offsets avoid ZipExtFile.seek or loading query arrays.
Archive SHA, not a partial member CRC, establishes the pinned byte identity.
"""

from __future__ import annotations

import ast
import hashlib
import os
import stat
import struct
import zipfile
from contextlib import contextmanager
from itertools import pairwise
from pathlib import Path

import numpy as np


def require(condition, message):
    if not condition:
        raise ValueError(message)


def pread_exact(fd, count, offset):
    parts = []
    while count:
        part = os.pread(fd, count, offset)
        require(bool(part), "Truncated direct archive read")
        parts.append(part)
        offset += len(part)
        count -= len(part)
    return b"".join(parts)


def fd_sha256(fd, size):
    h = hashlib.sha256()
    for offset in range(0, size, 1024 * 1024):
        h.update(pread_exact(fd, min(1024 * 1024, size - offset), offset))
    return h.hexdigest()


def regular_fd(path):
    """Caller owns descriptor; reject symlinks including parent path aliases."""
    p = Path(path).absolute()
    require(p.resolve() == p and not p.is_symlink(), "Symlink/path alias denied")
    fd = os.open(p, os.O_RDONLY | os.O_NOFOLLOW)
    try:
        info = os.fstat(fd)
        require(
            stat.S_ISREG(info.st_mode) and info.st_nlink == 1, "Single-link regular file required"
        )
    except BaseException:
        os.close(fd)
        raise
    return fd


def member_spans(fd, sample_counts, size):
    """Validate all ZIP directory/local envelopes, but decode no NPY values."""
    with os.fdopen(os.dup(fd), "rb") as stream, zipfile.ZipFile(stream) as archive:
        infos = archive.infolist()
        central_start = archive.start_dir
    expected = {f"{prefix}_{n}.npy" for n in sample_counts for prefix in ("x", "full", "a0")}
    require(
        len(infos) == len(expected) and {v.filename for v in infos} == expected,
        "Exact unique archive members required",
    )
    spans = {}
    ranges = []
    for info in infos:
        require(
            info.compress_type == zipfile.ZIP_STORED
            and info.flag_bits == 0
            and info.file_size == info.compress_size,
            "Stored unencrypted members required",
        )
        require(0 <= info.header_offset < central_start < size, "Archive bounds")
        header = struct.unpack("<4s5H3I2H", pread_exact(fd, 30, info.header_offset))
        magic, _, flags, method, _, _, crc, compressed32, uncompressed32, name_len, extra_len = (
            header
        )
        require(
            magic == b"PK\x03\x04" and flags == info.flag_bits and method == info.compress_type,
            "Local/central ZIP header mismatch",
        )
        require(crc == info.CRC, "Local/central CRC mismatch")
        require(
            compressed32 in (info.compress_size, 0xFFFFFFFF)
            and uncompressed32 in (info.file_size, 0xFFFFFFFF),
            "Local ZIP sizes disagree",
        )
        name = pread_exact(fd, name_len, info.header_offset + 30)
        require(name == info.filename.encode("ascii"), "Local/central member name mismatch")
        extra = pread_exact(fd, extra_len, info.header_offset + 30 + name_len)
        cursor, zip64 = 0, None
        while cursor < len(extra):
            require(cursor + 4 <= len(extra), "Malformed ZIP extra field")
            kind, length = struct.unpack_from("<HH", extra, cursor)
            cursor += 4
            require(cursor + length <= len(extra), "Malformed ZIP extra length")
            if kind == 1:
                require(zip64 is None, "Duplicate ZIP64 extra field")
                zip64 = extra[cursor : cursor + length]
            cursor += length
        cursor = 0
        for local_size, actual in (
            (uncompressed32, info.file_size),
            (compressed32, info.compress_size),
        ):
            if local_size == 0xFFFFFFFF:
                require(zip64 is not None and cursor + 8 <= len(zip64), "Missing ZIP64 size")
                require(
                    struct.unpack_from("<Q", zip64, cursor)[0] == actual, "ZIP64 size disagreement"
                )
                cursor += 8
        payload = info.header_offset + 30 + name_len + extra_len
        end = payload + info.file_size
        require(end <= central_start, "Member extends into ZIP directory")
        spans[info.filename] = (payload, info.file_size)
        ranges.append((info.header_offset, end))
    ordered = sorted(ranges)
    require(all(a[1] <= b[0] for a, b in pairwise(ordered)), "Overlapping ZIP members")
    return spans


def support_prefix(fd, span, samples):
    """Decode two C-order spans containing first five blocks, never block5..9."""
    payload, size = span
    head = pread_exact(fd, 10, payload)
    require(head[:8] == b"\x93NUMPY\x01\x00", "Only pinned NPY v1.0 is allowed")
    header_len = struct.unpack("<H", head[8:])[0]
    require(1 <= header_len <= 4096 and 10 + header_len <= size, "Bounded NPY header required")
    raw_header = pread_exact(fd, header_len, payload + 10)
    require(raw_header.endswith(b"\n"), "NPY header newline missing")
    header = ast.literal_eval(raw_header.decode("ascii"))
    require(
        isinstance(header, dict) and set(header) == {"descr", "fortran_order", "shape"},
        "Exact NPY header keys required",
    )
    require(
        header["descr"] == "<f8"
        and header["fortran_order"] is False
        and header["shape"] == (2, 10, 12, 5, 8, samples),
        "Frozen native NPY layout required",
    )
    block_bytes = 12 * 5 * 8 * samples * 8
    require(size == 10 + header_len + 20 * block_bytes, "Exact NPY data size required")
    data = payload + 10 + header_len
    arrays = [
        np.frombuffer(
            pread_exact(fd, 5 * block_bytes, data + i * 10 * block_bytes), dtype="<f8"
        ).reshape(5, 12, 5, 8, samples)
        for i in range(2)
    ]
    result = np.stack(arrays)
    require(np.isfinite(result).all(), "Nonfinite support; no block substitution")
    return result


@contextmanager
def verified_archive(path, expected_sha256, expected_bytes, sample_counts):
    fd = regular_fd(path)
    try:
        before = os.fstat(fd)
        require(before.st_size == expected_bytes, "Archive byte count mismatch")
        require(fd_sha256(fd, before.st_size) == expected_sha256, "Archive SHA mismatch")
        spans = member_spans(fd, sample_counts, before.st_size)
        yield lambda samples: support_prefix(fd, spans[f"x_{samples}.npy"], samples)
        after = os.fstat(fd)
        require(
            (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns, before.st_ctime_ns)
            == (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns, after.st_ctime_ns),
            "Archive changed during diagnostic",
        )
        require(fd_sha256(fd, after.st_size) == expected_sha256, "Archive post-work SHA mismatch")
        require(
            os.stat(path, follow_symlinks=False).st_ino == before.st_ino,
            "Archive path replaced during diagnostic",
        )
    finally:
        os.close(fd)
