"""Role-bound source39 archive and support-only metadata byte readers.

Integrity hashing and JSON lexical indexing touch bytes, not decoded EEG or
impedance values. No whole native NPY member or future impedance is decoded.
These are auditable workflow boundaries, not a hostile-process Python sandbox.
"""

from __future__ import annotations

import ast
import json
import os
import re
import struct
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .native_support_prefix import fd_sha256, member_spans, pread_exact, regular_fd, require
from .task_trca_shape_inputs import RolePartition

SOURCE_IDS = (
    4,
    6,
    8,
    11,
    14,
    21,
    22,
    25,
    28,
    29,
    30,
    31,
    32,
    33,
    37,
    41,
    42,
    43,
    44,
    46,
    54,
    55,
    56,
    61,
    63,
    65,
    67,
    73,
    74,
    77,
    80,
    82,
    83,
    84,
    89,
    92,
    97,
    100,
    102,
)
SAMPLE_COUNTS = (125, 188, 250, 500)
INTERFACES = ("dry", "wet")
_SHA = re.compile(r"[0-9a-f]{64}\Z")
_NUMBER = re.compile(rb"-?(?:0|[1-9][0-9]*)(?:\.[0-9]+)?(?:[eE][+-]?[0-9]+)?")
_FREEZE_KEY = object()


def _sha(value):
    require(isinstance(value, str) and _SHA.fullmatch(value), "Pinned SHA256 required")
    return value


def _partition(partition):
    require(isinstance(partition, RolePartition), "Explicit RolePartition required")
    ids = partition.fit_ids + partition.validation_ids + partition.evaluation_ids
    require(bool(ids) and set(ids) <= set(SOURCE_IDS), "Only pinned source39 IDs allowed")
    return partition


def _identity(info):
    return (
        info.st_dev,
        info.st_ino,
        info.st_size,
        info.st_mtime_ns,
        info.st_ctime_ns,
        info.st_nlink,
    )


class _PinnedFile:
    def __init__(self, path, sha256, size=None, maximum_bytes=None):
        self.path, self.sha256 = Path(path).absolute(), _sha(sha256)
        require(size is None or type(size) is int and size > 0, "Positive byte size required")
        self.size, self.maximum_bytes, self.fd = size, maximum_bytes, None

    def __enter__(self):
        require(self.fd is None, "Pinned reader already entered")
        self.fd = regular_fd(self.path)
        try:
            self.before = os.fstat(self.fd)
            require(
                self.size is None or self.before.st_size == self.size, "Pinned byte count mismatch"
            )
            require(
                self.maximum_bytes is None or self.before.st_size <= self.maximum_bytes,
                "Pinned file exceeds byte limit",
            )
            require(fd_sha256(self.fd, self.before.st_size) == self.sha256, "Pinned SHA mismatch")
        except BaseException:
            os.close(self.fd)
            self.fd = None
            raise
        return self

    def check(self, *, hash_bytes=False):
        require(self.fd is not None, "Pinned reader is not open")
        after = os.fstat(self.fd)
        require(_identity(self.before) == _identity(after), "Pinned file changed during access")
        # resolve() catches a replaced ancestor becoming a symlink as well.
        require(self.path.resolve() == self.path, "Pinned path became an alias")
        require(
            _identity(os.stat(self.path, follow_symlinks=False)) == _identity(self.before),
            "Pinned path replaced during access",
        )
        if hash_bytes:
            require(
                fd_sha256(self.fd, after.st_size) == self.sha256, "Pinned post-work SHA mismatch"
            )

    def __exit__(self, *exc):
        try:
            self.check(hash_bytes=True)
        finally:
            os.close(self.fd)
            self.fd = None


@dataclass(frozen=True)
class ArchiveSpec:
    path: str | Path
    participant_id: int
    sha256: str
    bytes: int

    def __post_init__(self):
        require(
            type(self.participant_id) is int and self.participant_id in SOURCE_IDS,
            "Archive participant outside pinned source39",
        )
        _sha(self.sha256)
        require(type(self.bytes) is int and self.bytes > 0, "Positive archive byte count required")


class AllModelsFrozen:
    """Token created only by verifying a pinned receipt and all three model files."""

    __slots__ = ("_files", "_key", "_partitions", "manifest_sha256", "receipt_sha256")

    def __init__(self, key, receipt_sha256, manifest_sha256, files, partitions):
        require(key is _FREEZE_KEY, "Use verify_freeze to obtain query authorization")
        self._key = key
        self.receipt_sha256, self.manifest_sha256 = receipt_sha256, manifest_sha256
        self._files, self._partitions = tuple(files), tuple(partitions)

    def authorize(self, participant_id, partition):
        require(self._key is _FREEZE_KEY, "Unverified freeze token")
        allowed = next((p for p in self._partitions if participant_id in p.evaluation_ids), None)
        if allowed is None or partition != allowed:
            raise PermissionError("Query requires exact frozen outer26/13 role partition")
        for path, identity in self._files:
            require(
                path.resolve() == path
                and _identity(os.stat(path, follow_symlinks=False)) == identity,
                "Frozen receipt/model file changed or replaced",
            )


def verify_freeze(path, expected_sha256, expected_manifest_sha256):
    """Verify immutable query authority; receipt/model paths are absolute and pinned.

    Receipt exact keys: schema, status, source_ids, manifest_sha256,
    query_access_count, models. Model exact keys: fold_id, fit_ids,
    evaluation_ids, path, sha256, bytes. Model JSON content is deliberately opaque.
    """
    _sha(expected_manifest_sha256)
    files, partitions = [], []
    with _PinnedFile(path, expected_sha256, maximum_bytes=1024 * 1024) as receipt:
        document = json.loads(
            pread_exact(receipt.fd, receipt.before.st_size, 0), object_pairs_hook=_unique_pairs
        )
        require(
            isinstance(document, dict)
            and set(document)
            == {
                "schema",
                "status",
                "source_ids",
                "manifest_sha256",
                "query_access_count",
                "models",
            },
            "Exact all-model freeze envelope required",
        )
        require(
            document["schema"] == "cfeg.task_trca_shape.all_models_frozen.v1"
            and document["status"] == "ALL_MODELS_FROZEN",
            "All models must be frozen",
        )
        require(
            document["source_ids"] == list(SOURCE_IDS)
            and all(type(v) is int for v in document["source_ids"]),
            "Freeze source39 mismatch",
        )
        require(
            document["manifest_sha256"] == expected_manifest_sha256,
            "Freeze execution manifest mismatch",
        )
        require(
            type(document["query_access_count"]) is int and document["query_access_count"] == 0,
            "Query access must be zero before freeze",
        )
        models = document["models"]
        require(
            isinstance(models, list) and len(models) == 3, "Exactly three outer models required"
        )
        for fold, model in enumerate(models):
            require(
                isinstance(model, dict)
                and set(model)
                == {"fold_id", "fit_ids", "evaluation_ids", "path", "sha256", "bytes"},
                "Exact frozen model envelope required",
            )
            fitting = tuple(v for rank, v in enumerate(SOURCE_IDS) if rank % 3 != fold)
            evaluation = tuple(v for rank, v in enumerate(SOURCE_IDS) if rank % 3 == fold)
            require(
                type(model["fold_id"]) is int
                and model["fold_id"] == fold
                and model["fit_ids"] == list(fitting)
                and model["evaluation_ids"] == list(evaluation)
                and all(type(v) is int for v in model["fit_ids"] + model["evaluation_ids"]),
                "Frozen outer fold partition mismatch",
            )
            require(
                isinstance(model["path"], str) and Path(model["path"]).is_absolute(),
                "Absolute frozen model path required",
            )
            with _PinnedFile(model["path"], model["sha256"], model["bytes"]) as pinned:
                files.append((pinned.path, _identity(pinned.before)))
            partitions.append(RolePartition(fitting, (), evaluation))
        files.append((receipt.path, _identity(receipt.before)))
    require(len({str(path) for path, _ in files}) == 4, "Distinct receipt and model paths required")
    return AllModelsFrozen(
        _FREEZE_KEY, expected_sha256, expected_manifest_sha256, files, partitions
    )


def _npy_data(fd, span, shape):
    payload, size = span
    head = pread_exact(fd, 10, payload)
    require(head[:8] == b"\x93NUMPY\x01\x00", "Only pinned NPY v1.0 allowed")
    length = struct.unpack("<H", head[8:])[0]
    require(1 <= length <= 4096 and 10 + length <= size, "Bounded NPY header required")
    raw = pread_exact(fd, length, payload + 10)
    require(raw.endswith(b"\n"), "NPY header newline missing")
    header = ast.literal_eval(raw.decode("ascii"))
    require(
        isinstance(header, dict) and set(header) == {"descr", "fortran_order", "shape"},
        "Exact NPY header keys required",
    )
    require(
        header["descr"] == "<f8"
        and header["fortran_order"] is False
        and header["shape"] == shape
        and all(type(v) is int for v in header["shape"]),
        "Frozen native NPY geometry required",
    )
    require(size == 10 + length + int(np.prod(shape)) * 8, "Exact NPY byte size required")
    return payload + 10 + length


def _decode_float64(fd, count, offset, shape):
    result = np.frombuffer(pread_exact(fd, count * 8, offset), dtype="<f8").reshape(shape)
    require(np.isfinite(result).all(), "Nonfinite authorized EEG/correlation values")
    return result


class NativeArchive:
    """One participant, one explicit role partition, exact contiguous block reads."""

    def __init__(self, spec, partition):
        require(isinstance(spec, ArchiveSpec), "ArchiveSpec required")
        self.spec, self.partition = spec, _partition(partition)
        self.role = self.partition.role(spec.participant_id)  # Deny before opening a path.
        self._file = _PinnedFile(spec.path, spec.sha256, spec.bytes)
        self.access_log = []

    def __enter__(self):
        self._file.__enter__()
        try:
            self._spans = member_spans(self._file.fd, SAMPLE_COUNTS, self.spec.bytes)
        except BaseException:
            self._file.__exit__(None, None, None)
            raise
        return self

    def __exit__(self, *exc):
        return self._file.__exit__(*exc)

    def _context(self, interface, samples):
        require(type(interface) is int and interface in (0, 1), "Interface must be integer0/1")
        require(
            type(samples) is int and samples in SAMPLE_COUNTS, "Only fixed native windows allowed"
        )
        self._file.check()

    def _query_authority(self, freeze):
        if self.role != "evaluation" or not isinstance(freeze, AllModelsFrozen):
            raise PermissionError("Query denied before verified all-outer-model freeze")
        freeze.authorize(self.spec.participant_id, self.partition)

    def _event(self, kind, interface, samples, blocks, freeze=None):
        event = {
            "participant_id": self.spec.participant_id,
            "role": self.role,
            "kind": kind,
            "interface": interface,
            "samples": samples,
            "blocks": list(blocks),
        }
        if freeze is not None:
            event["freeze_sha256"] = freeze.receipt_sha256
        self.access_log.append(event)

    def _eeg(self, interface, samples, first, count):
        shape = (2, 10, 12, 5, 8, samples)
        data = _npy_data(self._file.fd, self._spans[f"x_{samples}.npy"], shape)
        block = 12 * 5 * 8 * samples
        return _decode_float64(
            self._file.fd,
            count * block,
            data + (interface * 10 + first) * block * 8,
            (count, 12, 5, 8, samples),
        )

    def support(self, interface, samples, k):
        require(type(k) is int and k in (3, 5), "Only support budgets3/5 allowed")
        self._context(interface, samples)
        result = self._eeg(interface, samples, 0, k)
        self._event("support", interface, samples, range(k))
        return result

    def supervision(self, interface, samples):
        if self.role not in ("fit", "validation"):
            raise PermissionError("Evaluation block5 denied before decoding")
        self._context(interface, samples)
        result = self._eeg(interface, samples, 5, 1)[0]
        self._event("source_supervision", interface, samples, (5,))
        return result

    def query(self, interface, samples, freeze=None):
        self._query_authority(freeze)
        self._context(interface, samples)
        result = self._eeg(interface, samples, 6, 4)
        self._event("query", interface, samples, range(6, 10), freeze)
        return result

    def full_correlations(self, interface, samples, k, freeze=None):
        self._query_authority(freeze)
        require(type(k) is int and k in (3, 5), "Only full budgets3/5 allowed")
        self._context(interface, samples)
        data = _npy_data(self._file.fd, self._spans[f"full_{samples}.npy"], (2, 2, 4, 12, 5, 12))
        count = 4 * 12 * 5 * 12
        result = _decode_float64(
            self._file.fd, count, data + (interface * 2 + (k == 5)) * count * 8, (4, 12, 5, 12)
        )
        self._event(f"full_k{k}", interface, samples, range(6, 10), freeze)
        return result

    def a0_correlations(self, interface, samples, freeze=None):
        self._query_authority(freeze)
        self._context(interface, samples)
        data = _npy_data(self._file.fd, self._spans[f"a0_{samples}.npy"], (2, 4, 12, 5, 12))
        count = 4 * 12 * 5 * 12
        result = _decode_float64(self._file.fd, count, data + interface * count * 8, (4, 12, 5, 12))
        self._event("a0", interface, samples, range(6, 10), freeze)
        return result


def _unique_pairs(pairs):
    result = {}
    for key, value in pairs:
        require(key not in result, "Duplicate JSON key")
        result[key] = value
    return result


@dataclass(frozen=True)
class _Node:
    start: int
    end: int
    kind: str
    children: object = None


class _JSONSpans:
    """Strict bounded JSON grammar; number tokens remain byte spans, not floats."""

    def __init__(self, raw):
        self.raw, self.pos = raw, 0
        self.root = self.value(0)
        self.space()
        require(self.pos == len(raw), "Trailing projection JSON bytes")

    def space(self):
        while self.pos < len(self.raw) and self.raw[self.pos] in b" \t\r\n":
            self.pos += 1

    def value(self, depth):
        require(depth <= 12, "Projection JSON nesting limit")
        self.space()
        start = self.pos
        require(start < len(self.raw), "Truncated projection JSON")
        char = self.raw[start]
        if char == 34:
            self.pos += 1
            while self.pos < len(self.raw):
                current = self.raw[self.pos]
                self.pos += 1
                if current == 92:
                    require(self.pos < len(self.raw), "Truncated JSON string escape")
                    self.pos += 1
                elif current == 34:
                    # Strings are routing/structural values; never impedance.
                    json.loads(self.raw[start : self.pos])
                    return _Node(start, self.pos, "string")
            raise ValueError("Unterminated projection JSON string")
        if char in (91, 123):
            is_object = char == 123
            end = 125 if is_object else 93
            self.pos += 1
            children = {} if is_object else []
            self.space()
            if self.pos < len(self.raw) and self.raw[self.pos] == end:
                self.pos += 1
                return _Node(start, self.pos, "object" if is_object else "array", children)
            while True:
                if is_object:
                    key = self.value(depth + 1)
                    require(key.kind == "string", "JSON object key must be string")
                    decoded = self.decode(key)
                    require(decoded not in children, "Duplicate JSON key")
                    self.space()
                    require(
                        self.pos < len(self.raw) and self.raw[self.pos] == 58, "JSON colon missing"
                    )
                    self.pos += 1
                    children[decoded] = self.value(depth + 1)
                else:
                    children.append(self.value(depth + 1))
                self.space()
                require(self.pos < len(self.raw), "Truncated projection JSON container")
                separator = self.raw[self.pos]
                self.pos += 1
                if separator == end:
                    return _Node(start, self.pos, "object" if is_object else "array", children)
                require(separator == 44, "JSON comma missing")
        for literal in (b"null", b"true", b"false"):
            if self.raw.startswith(literal, start):
                self.pos += len(literal)
                return _Node(start, self.pos, literal.decode("ascii"))
        match = _NUMBER.match(self.raw, start)
        require(
            match is not None and match.end() - start <= 128, "Invalid/big projection number token"
        )
        self.pos = match.end()
        return _Node(start, self.pos, "number")

    def decode(self, node):
        return json.loads(self.raw[node.start : node.end], object_pairs_hook=_unique_pairs)


class SupportMetadata:
    """Selective support impedance reader for the pinned source-only JSON projection.

    ``expected_envelope`` is the exact ten non-``packets`` top-level fields:
    the old plan's five ``envelope_provenance`` fields plus manifest_sha256,
    returned_rows, returned_packets, returned_subject_ids, columns. It must not
    contain ``packets``. All 780 routing records are indexed, but only a requested
    participant/interface's first k numeric packet spans are converted.
    """

    def __init__(self, path, sha256, expected_envelope, partition):
        self.partition = _partition(partition)
        fields = {
            "schema",
            "study_id",
            "plan_sha256",
            "source_commit",
            "start_sha256",
            "manifest_sha256",
            "returned_rows",
            "returned_packets",
            "returned_subject_ids",
            "columns",
        }
        require(
            isinstance(expected_envelope, dict) and set(expected_envelope) == fields,
            "Exact ten-field expected metadata envelope required",
        )
        require(
            expected_envelope["returned_subject_ids"] == list(SOURCE_IDS)
            and all(type(v) is int for v in expected_envelope["returned_subject_ids"])
            and type(expected_envelope["returned_packets"]) is int
            and expected_envelope["returned_packets"] == 780
            and type(expected_envelope["returned_rows"]) is int
            and expected_envelope["returned_rows"] == 9360,
            "Expected metadata source-only boundary mismatch",
        )
        self.expected = json.loads(json.dumps(expected_envelope))
        self._file = _PinnedFile(path, sha256, maximum_bytes=16 * 1024 * 1024)
        self.access_log = []

    def __enter__(self):
        self._file.__enter__()
        try:
            self._index()
        except BaseException:
            self._file.__exit__(None, None, None)
            raise
        return self

    def __exit__(self, *exc):
        try:
            return self._file.__exit__(*exc)
        finally:
            self._json, self._packets = None, None

    def _index(self):
        parser = _JSONSpans(pread_exact(self._file.fd, self._file.before.st_size, 0))
        require(
            parser.root.kind == "object"
            and set(parser.root.children) == set(self.expected) | {"packets"},
            "Exact eleven-field metadata envelope required",
        )
        for key, expected in self.expected.items():
            actual = parser.decode(parser.root.children[key])
            require(
                actual == expected and type(actual) is type(expected),
                "Metadata envelope mismatch: " + key,
            )
        packet_array = parser.root.children["packets"]
        require(
            packet_array.kind == "array" and len(packet_array.children) == 780,
            "Exactly780 source-only metadata packets required",
        )
        packets, orders = {}, {}
        for packet in packet_array.children:
            require(
                packet.kind == "object"
                and set(packet.children)
                == {
                    "subject_id",
                    "interface",
                    "block_id",
                    "impedance_kohm",
                    "headband_order",
                    "condition_period",
                },
                "Exact metadata packet keys required",
            )
            routing = {
                key: parser.decode(value)
                for key, value in packet.children.items()
                if key != "impedance_kohm"
            }
            pid, interface, block = routing["subject_id"], routing["interface"], routing["block_id"]
            require(
                type(pid) is int
                and pid in SOURCE_IDS
                and interface in INTERFACES
                and type(block) is int
                and 0 <= block < 10,
                "Metadata routing outside source boundary",
            )
            first = routing["headband_order"]
            require(
                first in INTERFACES
                and orders.get(pid, first) == first
                and routing["condition_period"] == ("first" if first == interface else "second"),
                "Inconsistent acquisition order",
            )
            key = (pid, INTERFACES.index(interface), block)
            require(key not in packets, "Duplicate metadata packet")
            numeric = packet.children["impedance_kohm"]
            require(
                numeric.kind == "array"
                and len(numeric.children) == 8
                and all(v.kind in ("number", "null") for v in numeric.children),
                "Metadata packet requires eight numeric/null lexemes",
            )
            packets[key], orders[pid] = numeric, first
        expected_keys = {
            (pid, interface, block)
            for pid in SOURCE_IDS
            for interface in (0, 1)
            for block in range(10)
        }
        require(set(packets) == expected_keys, "Incomplete source-only metadata projection")
        self._json, self._packets = parser, packets
        self._orders = {pid: INTERFACES.index(first) for pid, first in orders.items()}

    def support(self, participant_id, interface, k):
        role = self.partition.role(participant_id)
        require(type(interface) is int and interface in (0, 1), "Interface must be integer0/1")
        require(type(k) is int and k in (3, 5), "Only metadata support budgets3/5 allowed")
        self._file.check()
        rows = [
            self._json.decode(self._packets[participant_id, interface, block]) for block in range(k)
        ]
        result = np.asarray(rows, dtype=np.float64)
        require(
            result.shape == (k, 8)
            and np.all(np.isnan(result) | (np.isfinite(result) & (result >= 0))),
            "Authorized impedance must be nonnegative finite or missing",
        )
        result = np.frombuffer(result.tobytes(), dtype=np.float64).reshape(k, 8)
        self.access_log.append(
            {
                "participant_id": participant_id,
                "role": role,
                "kind": "metadata_support",
                "interface": interface,
                "blocks": list(range(k)),
            }
        )
        return result, self._orders[participant_id]
