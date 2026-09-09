"""Generated-only registration/failure routing and true file producer/audit path."""

import importlib.util
import json
import sys

import numpy as np
import pytest

from cfeg.analysis import source_channel_margin_runtime as r


def test_incomplete_code_pin_closure_denied(tmp_path, monkeypatch):
    registration = {
        "schema": r.SCHEMA,
        "human_execution_authorized": True,
        "parent": str(tmp_path),
        "repository": str(r.ROOT),
        "source_ids": list(r.archive.SOURCE_IDS),
        "config_sha256": r.CONFIG_SHA,
        "attempts": {"primary": 1, "audit": 1, "retries": 0},
        "code_pins": {},
    }
    r.publish(tmp_path / "registration.json", registration)
    monkeypatch.setattr(r, "code_names", lambda: ["required.py"])
    monkeypatch.setattr(r, "input_envelope", lambda: pytest.fail("input read before closure"))
    with pytest.raises(ValueError, match="code pins"):
        r.validate_registration(tmp_path, r.digest(tmp_path / "registration.json")["sha256"])


@pytest.fixture
def launcher(monkeypatch):
    path = r.ROOT / "scripts/launch_n1_metadata_generated_efficacy.py"
    spec = importlib.util.spec_from_file_location("launch_n1_metadata_generated_efficacy", path)
    legacy = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(legacy)
    monkeypatch.setitem(sys.modules, spec.name, legacy)
    spec = importlib.util.spec_from_file_location(
        "margin_launcher", r.ROOT / "scripts/launch_source_channel_margin_probe.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize(
    "case,terminal,calls",
    [
        ("ok", "NO_PROMISING_SIGNAL_IN_THIS_PROBE", 2),
        ("input", "INVALID_INPUT", 1),
        ("primary", "EXECUTION_FAILURE", 1),
        ("audit", "AUDIT_FAILURE", 2),
    ],
)
def test_supervisor_terminal_routing_no_retry(
    tmp_path, monkeypatch, launcher, case, terminal, calls
):
    (tmp_path / "data").mkdir()
    (tmp_path / "logs").mkdir()
    seen = []
    monkeypatch.setattr(launcher, "validate", lambda *_: {})
    monkeypatch.setattr(launcher.lifecycle, "await_launch_receipt", lambda *_: None)

    def child(argv, parent, label, seconds, limits):
        seen.append(label)
        assert seconds == (1800 if label == "primary" else 900)
        if label == "primary":
            if case == "input":
                r.publish(parent / "data/failure.json", {"terminal": "INVALID_INPUT"})
            else:
                r.publish(
                    parent / "data/result.json", {"terminal": "NO_PROMISING_SIGNAL_IN_THIS_PROBE"}
                )
        elif case != "audit":
            r.publish(parent / "data/audit.json", {"status": "PASS"})
        failed = (label == "primary" and case in ("input", "primary")) or (
            label == "audit" and case == "audit"
        )
        return {"exit_code": int(failed), "watchdog_reason": None, "process_group_remaining": False}

    monkeypatch.setattr(launcher.lifecycle, "run_child", child)
    launcher.supervise(tmp_path, "a" * 64)
    result = json.loads((tmp_path / "completion.json").read_text())
    assert result["terminal"] == terminal
    assert len(seen) == calls
    assert result["error"] is None


def test_audit_missing_files_records_failure(tmp_path, monkeypatch):
    (tmp_path / "data").mkdir()
    monkeypatch.setattr(r, "validate_registration", lambda *_: {})
    with pytest.raises(FileNotFoundError):
        r.run_audit(tmp_path, "a" * 64)
    failure = json.loads((tmp_path / "data/audit_failure.json").read_text())
    assert failure["type"] == "FileNotFoundError"


def test_primary_bad_extraction_is_invalid_input(tmp_path, monkeypatch):
    (tmp_path / "data").mkdir()
    monkeypatch.setattr(r, "validate_registration", lambda *_: {"inputs": {}})

    def bad(*_):
        raise ValueError("undefined centered correlation")

    monkeypatch.setattr(r, "extract", bad)
    with pytest.raises(ValueError):
        r.run_primary(tmp_path, "a" * 64)
    failure = json.loads((tmp_path / "data/failure.json").read_text())
    assert failure["terminal"] == "INVALID_INPUT"
    assert not (tmp_path / "data/models.json").exists()


@pytest.fixture(scope="module")
def generated_output(tmp_path_factory):
    """Human-sized generated arrays, injected extraction; no human file reads."""
    root = tmp_path_factory.mktemp("file-producer-auditor")
    (root / "data").mkdir()
    original_validate, original_extract = r.validate_registration, r.extract
    frequencies = [9.25, 11.25, 13.25, 9.75, 11.75, 13.75, 10.25, 12.25, 14.25, 10.75, 12.75, 14.75]

    def generated(_inputs, sink):
        rng = np.random.default_rng(404)
        support = rng.normal(0, 0.5, size=(39, 2, 3, 12, 5, 8, 250))
        source = rng.normal(0, 0.5, size=(39, 2, 12, 5, 8, 250))
        wave = np.sin(2 * np.pi * np.asarray(frequencies)[:, None] * np.arange(250)[None, :] / 250)
        support += wave[None, None, None, :, None, None, :]
        source += wave[None, None, :, None, None, :]
        data = {
            "ids": np.asarray(r.archive.SOURCE_IDS),
            "orders": np.arange(39) % 2,
            "support": support,
            "source_block": source,
            "packet": rng.uniform(0, 50, size=(39, 2, 3, 8)),
            "q": np.empty((39, 2, 5, 8, 15)),
            "target": np.empty((39, 2, 5, 8)),
        }
        for p, pid in enumerate(data["ids"]):
            for interface in (0, 1):
                for kind, blocks in (
                    ("metadata_support", [0, 1, 2]),
                    ("support", [0, 1, 2]),
                    ("source_supervision", [5]),
                ):
                    event = {
                        "phase": "before_decode",
                        "participant_id": int(pid),
                        "interface": interface,
                        "kind": kind,
                        "blocks": blocks,
                        "role": "source_extraction",
                        "underlying_role": "fit",
                    }
                    if kind != "metadata_support":
                        event["samples"] = 250
                    sink(event)
                s, x, m = support[p, interface], source[p, interface], data["packet"][p, interface]
                data["q"][p, interface] = r.support_q_mask(
                    s, np.isfinite(m), interface, int(data["orders"][p]), frequencies
                )
                data["target"][p, interface] = r.source_channel_margin(
                    s, x, np.arange(12), np.arange(12)
                ).channel_shape
        return data

    r.validate_registration = lambda *_: {"inputs": {}}
    r.extract = generated
    try:
        r.run_primary(root, "a" * 64)
    finally:
        r.validate_registration, r.extract = original_validate, original_extract
    return root


def test_complete_generated_file_path(generated_output, monkeypatch):
    monkeypatch.setattr(r, "validate_registration", lambda *_: {})
    r.run_audit(generated_output, "a" * 64)
    audit = json.loads((generated_output / "data/audit.json").read_text())
    assert audit["status"] == "PASS"
    assert audit["allowed_decodes"] == 234
    result = json.loads((generated_output / "data/result.json").read_text())
    assert result["ridge_fit_count"] == 120
    with pytest.raises(FileExistsError):
        r.run_primary(generated_output, "a" * 64)


def test_changed_freeze_binding_rejected(tmp_path, generated_output, monkeypatch):
    import cfeg.analysis.source_channel_margin_audit as auditor

    folder = tmp_path / "data"
    folder.mkdir()
    real_digest, real_read = r.digest, r.read_json
    source = generated_output / "data"
    names = {
        "source.npz",
        "models.json",
        "globalfreeze.json",
        "predictions.npz",
        "result.json",
        "events.jsonl",
    }

    def digest(path):
        return (
            real_digest(source / path.name)
            if path.parent == folder and path.name in names
            else real_digest(path)
        )

    def read(path, sha):
        if path.parent == folder and path.name == "globalfreeze.json":
            result = dict(real_read(source / path.name, sha))
            result["models"] = {"sha256": "0" * 64, "bytes": 1}
            return result
        return real_read(path, sha)

    monkeypatch.setattr(r, "validate_registration", lambda *_: {})
    monkeypatch.setattr(r, "digest", digest)
    monkeypatch.setattr(r, "read_json", read)
    monkeypatch.setattr(
        auditor, "audit_probe", lambda *_: pytest.fail("math reached before freeze check")
    )
    with pytest.raises(ValueError, match="freeze mismatch"):
        r.run_audit(tmp_path, "a" * 64)
    assert (folder / "audit_failure.json").exists()
