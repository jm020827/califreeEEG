"""Scoped source-only extraction and file-backed channel-margin probe lifecycle.

Workflow restrictions, not a hostile-process security sandbox. No old query,
FULL or A0 API is exposed by the new facade. Only a new pinned registration may
invoke human extraction; pure fitting and tests do not grant that authority.
"""

from __future__ import annotations

import hashlib
import json
import os
import stat
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from cfeg.analysis import task_trca_n1_source39_archive as archive
from cfeg.analysis import task_trca_shape_archive as byteio
from cfeg.analysis.source_channel_margin import source_channel_margin
from cfeg.analysis.task_trca_shape_features import support_q_mask
from cfeg.analysis.task_trca_shape_inputs import RolePartition

ROOT = Path(__file__).resolve().parents[3]
CONFIG = "configs/analysis/source39_channel_margin_probe_v1.json"
CONFIG_SHA = "236f15e8bc08eae604364e5752be323a86233158604f3df8b406e22f551577f3"
NATIVE_PLAN = "configs/analysis/metadata_prior_source39_v1.json"
NATIVE_PLAN_SHA = "ed67cc1f361ed89c37b6f7c5df1e9170b78484e428a5105020485c8c03e3934b"
NATIVE_ROOT = Path("/home/whwovy/metadata-prior-source39-v1/native-cold-r1")
NATIVE_SHA = "15c088048f3435a500d3758138c3cd6a1616e869d568bdf68d10bcc5e978d83a"
SCHEMA = "cfeg.source39-channel-margin-probe.execution.v1"
DATA_KEYS = {"ids", "orders", "support", "source_block", "packet", "q", "target"}


def code_names():
    names = subprocess.check_output(
        ["git", "ls-files", "src/**/*.py", "scripts/*.py", "tests/*.py"], cwd=ROOT, text=True
    ).splitlines()
    names += [
        CONFIG,
        NATIVE_PLAN,
        "docs/source39_channel_margin_probe_v1_design.md",
        "docs/source39_channel_margin_probe_v1_execution.md",
    ]
    required = {
        "src/cfeg/analysis/source_channel_margin_runtime.py",
        "src/cfeg/analysis/source_channel_margin_probe.py",
        "src/cfeg/analysis/source_channel_margin_audit.py",
        "scripts/launch_source_channel_margin_probe.py",
    }
    if not required <= set(names):
        raise ValueError("all execution modules must be tracked before registration")
    return sorted(set(names))


def utc():
    return datetime.now(timezone.utc).isoformat()


def canonical(path):
    p = Path(path)
    if not p.is_absolute() or p.resolve() != p or p.is_symlink():
        raise ValueError("exact unaliased absolute path required")
    return p


def digest(path):
    path = canonical(path)
    with os.fdopen(os.open(path, os.O_RDONLY | os.O_NOFOLLOW), "rb") as stream:
        before = os.fstat(stream.fileno())
        if not stat.S_ISREG(before.st_mode) or before.st_nlink != 1:
            raise ValueError("regular single-link input required")
        h = hashlib.sha256()
        for chunk in iter(lambda: stream.read(1024**2), b""):
            h.update(chunk)
        if byteio._identity(before) != byteio._identity(os.fstat(stream.fileno())):
            raise ValueError("input changed during hash")
    return {"sha256": h.hexdigest(), "bytes": before.st_size}


def read_json(path, sha):
    with byteio._PinnedFile(canonical(path), sha, maximum_bytes=32 * 1024**2) as pinned:
        return json.loads(
            byteio.pread_exact(pinned.fd, pinned.before.st_size, 0),
            object_pairs_hook=byteio._unique_pairs,
        )


def publish(path, value):
    path = canonical(path)
    with path.open("x") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    path.chmod(0o400)


def save_npz(path, **arrays):
    path = canonical(path)
    if any(np.asarray(v).dtype.kind not in "fiu" for v in arrays.values()):
        raise ValueError("only numeric real arrays may be saved")
    with path.open("xb") as stream:
        np.savez(stream, **arrays)
        stream.flush()
        os.fsync(stream.fileno())
    path.chmod(0o400)


class Journal:
    def __init__(self, path, registration_sha):
        self.path = canonical(path)
        self.stream = self.path.open("x")
        self.seq = 0
        self.sha = registration_sha

    def __call__(self, value):
        row = {"seq": self.seq, "utc": utc(), "registration_sha256": self.sha, "event": value}
        self.stream.write(json.dumps(row, allow_nan=False) + "\n")
        self.stream.flush()
        os.fsync(self.stream.fileno())
        self.seq += 1

    def close(self):
        self.stream.close()
        self.path.chmod(0o400)


def _source_sink(sink):
    if not callable(sink):
        raise TypeError("durable event sink required")

    def wrapped(event):
        sink({**event, "role": "source_extraction", "underlying_role": "fit"})

    return wrapped


class SourceOnlyArchive:
    """Composition: fixed k3/N250 and source block5; no query/FULL/A0 methods."""

    def __init__(self, spec, *, event_sink):
        self._reader = archive.NativeArchive(
            spec, RolePartition(archive.SOURCE_IDS), event_sink=_source_sink(event_sink)
        )

    def __enter__(self):
        self._reader.__enter__()
        return self

    def __exit__(self, *exc):
        return self._reader.__exit__(*exc)

    def support3(self, interface):
        return self._reader.support(interface, 250, 3)

    def source_block5(self, interface):
        return self._reader.supervision(interface, 250)


class SourceOnlyMetadata:
    def __init__(self, path, sha, envelope, *, event_sink):
        self._reader = archive.SupportMetadata(
            path,
            sha,
            envelope,
            RolePartition(archive.SOURCE_IDS),
            event_sink=_source_sink(event_sink),
        )

    def __enter__(self):
        self._reader.__enter__()
        return self

    def __exit__(self, *exc):
        return self._reader.__exit__(*exc)

    def metadata3(self, pid, interface):
        return self._reader.support(pid, interface, 3)


def input_envelope():
    """Read only pinned old plan/native manifest lineage, never numeric arrays."""
    plan = read_json(ROOT / NATIVE_PLAN, NATIVE_PLAN_SHA)
    native = read_json(NATIVE_ROOT / "result.json", NATIVE_SHA)
    if (
        native["status"] != "COMPLETE"
        or native["plan_sha256"] != NATIVE_PLAN_SHA
        or native["source_subject_ids"] != list(archive.SOURCE_IDS)
        or [row["subject"] for row in native["files"]] != list(archive.SOURCE_IDS)
    ):
        raise ValueError("native source39 manifest mismatch")
    read_json(NATIVE_ROOT / "start.json", native["start_sha256"])
    files = []
    for row in native["files"]:
        if row["filename"] != f"S{row['subject']:03d}.npz":
            raise ValueError("wrong native filename")
        files.append(
            {
                "path": str(NATIVE_ROOT / row["filename"]),
                **{k: row[k] for k in ("subject", "sha256", "bytes")},
            }
        )
    projection = plan["source_projection"]
    envelope = {
        **projection["envelope_provenance"],
        "manifest_sha256": projection["manifest_sha256"],
        "returned_rows": projection["returned_rows"],
        "returned_packets": projection["packets"],
        "returned_subject_ids": list(archive.SOURCE_IDS),
        "columns": projection["columns"],
    }
    return {
        "archives": files,
        "metadata": {
            "path": projection["path"],
            "sha256": projection["sha256"],
            "envelope": envelope,
        },
        "frequencies": plan["frequencies"],
        "native_manifest": {"path": str(NATIVE_ROOT / "result.json"), "sha256": NATIVE_SHA},
        "native_start": {"path": str(NATIVE_ROOT / "start.json"), "sha256": native["start_sha256"]},
    }


def validate_registration(parent, sha):
    parent = canonical(parent)
    r = read_json(parent / "registration.json", sha)
    if (
        r.get("schema") != SCHEMA
        or r.get("human_execution_authorized") is not True
        or r.get("parent") != str(parent)
        or r.get("repository") != str(ROOT)
        or r.get("source_ids") != list(archive.SOURCE_IDS)
        or r.get("config_sha256") != CONFIG_SHA
        or r.get("attempts") != {"primary": 1, "audit": 1, "retries": 0}
    ):
        raise PermissionError("new exact source-only execution authority required")
    if digest(ROOT / CONFIG)["sha256"] != CONFIG_SHA:
        raise ValueError("science config changed")
    if set(r["code_pins"]) != set(code_names()):
        raise ValueError("incomplete or extra execution code pins")
    for name, pin in r["code_pins"].items():
        if Path(name).is_absolute() or ".." in Path(name).parts or digest(ROOT / name) != pin:
            raise ValueError("code pin mismatch: " + name)
    if r["inputs"] != input_envelope():
        raise ValueError("registered source input envelope changed")
    test = r["tests"]
    if digest(Path(test["path"])) != {k: test[k] for k in ("sha256", "bytes")}:
        raise ValueError("test receipt changed")
    return r


def extract(inputs, journal):
    """Called only after registration validation by primary; all reads journaled."""
    ids = np.array(archive.SOURCE_IDS, dtype=np.int64)
    data = {
        "ids": ids,
        "orders": np.empty(39, dtype=np.int64),
        "support": np.empty((39, 2, 3, 12, 5, 8, 250)),
        "source_block": np.empty((39, 2, 12, 5, 8, 250)),
        "packet": np.empty((39, 2, 3, 8)),
        "q": np.empty((39, 2, 5, 8, 15)),
        "target": np.empty((39, 2, 5, 8)),
    }
    m = inputs["metadata"]
    with SourceOnlyMetadata(m["path"], m["sha256"], m["envelope"], event_sink=journal) as metadata:
        for rank, spec in enumerate(inputs["archives"]):
            pid = int(ids[rank])
            if spec["subject"] != pid:
                raise ValueError("source extraction order mismatch")
            descriptor = archive.ArchiveSpec(spec["path"], pid, spec["sha256"], spec["bytes"])
            with SourceOnlyArchive(descriptor, event_sink=journal) as reader:
                for interface in (0, 1):
                    packet, order = metadata.metadata3(pid, interface)
                    if interface and order != data["orders"][rank]:
                        raise ValueError("paired order mismatch")
                    data["orders"][rank] = order
                    support = reader.support3(interface)
                    source = reader.source_block5(interface)
                    data["support"][rank, interface] = support
                    data["source_block"][rank, interface] = source
                    data["packet"][rank, interface] = packet
                    data["q"][rank, interface] = support_q_mask(
                        support, np.isfinite(packet), interface, order, inputs["frequencies"]
                    )
                    data["target"][rank, interface] = source_channel_margin(
                        support, source, np.arange(12), np.arange(12)
                    ).channel_shape
            journal({"kind": "participant_extracted", "participant_id": pid})
    return data


def run_primary(parent, sha):
    from cfeg.analysis.source_channel_margin_probe import evaluate_probe, fit_probe

    registration = validate_registration(parent, sha)
    folder = parent / "data"
    publish(folder / "primary.claim.json", {"utc": utc(), "registration_sha256": sha})
    journal = Journal(folder / "events.jsonl", sha)
    stage = "extraction"
    try:
        journal({"kind": "extraction_start"})
        data = extract(registration["inputs"], journal)
        save_npz(folder / "source.npz", **data)
        journal({"kind": "extraction_complete", "source": digest(folder / "source.npz")})
        stage = "fit"
        models = fit_probe(data, event_sink=journal)
        publish(folder / "models.json", models)
        freeze = {
            "utc": utc(),
            "registration_sha256": sha,
            "models": digest(folder / "models.json"),
            "source": digest(folder / "source.npz"),
            "event_count_before_freeze": journal.seq,
        }
        publish(folder / "globalfreeze.json", freeze)
        journal({"kind": "all_models_frozen", "freeze": digest(folder / "globalfreeze.json")})
        predictions, result = evaluate_probe(data, models)
        save_npz(folder / "predictions.npz", predictions=predictions)
        publish(folder / "result.json", result)
        journal({"kind": "evaluation_complete", "terminal": result["terminal"]})
    except BaseException as exc:
        publish(
            folder / "failure.json",
            {
                "utc": utc(),
                "type": type(exc).__name__,
                "message": str(exc),
                "stage": stage,
                "terminal": "INVALID_INPUT"
                if stage == "extraction" and isinstance(exc, ValueError)
                else "EXECUTION_FAILURE",
                "events": journal.seq,
                "registration_sha256": sha,
            },
        )
        raise
    finally:
        journal.close()


def run_audit(parent, sha):
    from cfeg.analysis.source_channel_margin_audit import audit_probe

    validate_registration(parent, sha)
    folder = parent / "data"
    publish(folder / "audit.claim.json", {"utc": utc(), "registration_sha256": sha})
    # Primary is already terminated. Bind these exact sealed artifacts at entry/exit.
    names = (
        "source.npz",
        "models.json",
        "globalfreeze.json",
        "predictions.npz",
        "result.json",
        "events.jsonl",
    )
    audit = None
    try:
        pins = {n: digest(folder / n) for n in names}
        freeze = read_json(folder / "globalfreeze.json", pins["globalfreeze.json"]["sha256"])
        if (
            freeze["models"] != pins["models.json"]
            or freeze["source"] != pins["source.npz"]
            or freeze["registration_sha256"] != sha
        ):
            raise ValueError("source/model freeze mismatch")
        events = [json.loads(line) for line in (folder / "events.jsonl").read_text().splitlines()]
        if [r["seq"] for r in events] != list(range(len(events))) or any(
            r["registration_sha256"] != sha for r in events
        ):
            raise ValueError("journal identity/sequence mismatch")
        decodes = [r["event"] for r in events if r["event"].get("phase") == "before_decode"]
        expected = [
            (p, i, k, b)
            for p in archive.SOURCE_IDS
            for i in (0, 1)
            for k, b in (
                ("metadata_support", [0, 1, 2]),
                ("support", [0, 1, 2]),
                ("source_supervision", [5]),
            )
        ]
        observed = [(r["participant_id"], r["interface"], r["kind"], r["blocks"]) for r in decodes]
        if observed != expected or any(
            r["role"] != "source_extraction" or r["underlying_role"] != "fit" for r in decodes
        ):
            raise ValueError("not the exact allowed234 decodes")
        if any(r.get("samples") != 250 for r in decodes if r["kind"] != "metadata_support"):
            raise ValueError("wrong window read")
        frozen_rows = [r for r in events if r["event"].get("kind") == "all_models_frozen"]
        evaluated_rows = [r for r in events if r["event"].get("kind") == "evaluation_complete"]
        if (
            len(frozen_rows) != 1
            or len(evaluated_rows) != 1
            or frozen_rows[0]["seq"] != freeze["event_count_before_freeze"]
            or frozen_rows[0]["seq"] >= evaluated_rows[0]["seq"]
        ):
            raise ValueError("invalid publication chronology")
        fit_events = [row for row in events if row["event"].get("event") == "ridge_fit_completed"]
        if [row["event"]["fit_index"] for row in fit_events] != list(range(1, 121)):
            raise ValueError("exact120 primary fit events required")
        if (
            max(row["seq"] for row in events if row["event"].get("phase") == "before_decode")
            >= fit_events[0]["seq"]
            or fit_events[-1]["seq"] >= frozen_rows[0]["seq"]
            or frozen_rows[0]["event"]["freeze"] != pins["globalfreeze.json"]
        ):
            raise ValueError("extraction/fit/freeze chronology mismatch")
        with np.load(folder / "source.npz", allow_pickle=False) as stored:
            if set(stored.files) != DATA_KEYS:
                raise ValueError("wrong permitted source array keys")
            data = {name: stored[name] for name in stored.files}
        if not np.array_equal(data["ids"], archive.SOURCE_IDS) or data["support"].shape != (
            39,
            2,
            3,
            12,
            5,
            8,
            250,
        ):
            raise ValueError("wrong human source geometry")
        with np.load(folder / "predictions.npz", allow_pickle=False) as stored:
            if stored.files != ["predictions"]:
                raise ValueError("unexpected prediction keys")
            predictions = stored["predictions"]
        result = read_json(folder / "result.json", pins["result.json"]["sha256"])
        extracted_rows = [
            row for row in events if row["event"].get("kind") == "extraction_complete"
        ]
        if (
            len(extracted_rows) != 1
            or extracted_rows[0]["event"]["source"] != pins["source.npz"]
            or extracted_rows[0]["seq"] >= fit_events[0]["seq"]
            or evaluated_rows[0]["event"]["terminal"] != result["terminal"]
        ):
            raise ValueError("extraction/evaluation journal binding mismatch")
        models = read_json(folder / "models.json", pins["models.json"]["sha256"])
        audit = audit_probe(data, models, predictions, result)
        if audit["status"] != "PASS":
            raise ValueError("independent numerical audit failed")
        if pins != {n: digest(folder / n) for n in names}:
            raise ValueError("artifacts changed during audit")
        publish(
            folder / "audit.json",
            {
                **audit,
                "artifacts": pins,
                "allowed_decodes": len(decodes),
                "registration_sha256": sha,
                "utc": utc(),
                "reader_limitation": "Source extraction is journaled and byte selective; raw archive preprocessing is inherited, not independently reprocessed. Q helper shared; workflow not OS sandbox.",
            },
        )
    except BaseException as exc:
        publish(
            folder / "audit_failure.json",
            {
                "utc": utc(),
                "type": type(exc).__name__,
                "message": str(exc),
                "independent_audit": audit,
                "registration_sha256": sha,
            },
        )
        raise
