"""Reproduce the terminal attempt's schema failure using artificial data only.

This is a historical failure regression, NOT a repaired runner or retry permit.
Unlike the original lifecycle fixture, it connects the real loader and core.
"""

import copy
import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def load(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / (name + ".py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_original_writer_envelope_rejected_before_eeg_by_terminal_attempt(tmp_path, monkeypatch):
    producer, core = load("run_native_subset_m_source"), load("native_subset_m_core")
    plan = json.loads((ROOT / "configs/analysis/native_subset_m_source39_v1.json").read_text())
    packets = []
    for subject in plan["source_subject_ids"]:
        for interface in ("dry", "wet"):
            for block in range(10):
                packets.append(
                    {
                        "subject_id": subject,
                        "interface": interface,
                        "block_id": block,
                        "impedance_kohm": [1.0] * 8,
                        "headband_order": "dry",
                        "condition_period": "first" if interface == "dry" else "second",
                    }
                )
    content = {
        "manifest_sha256": plan["source_projection"]["manifest_sha256"],
        "packets": packets,
        "returned_rows": 9360,
        "returned_packets": 780,
        "returned_subject_ids": plan["source_subject_ids"],
        "columns": plan["source_projection"]["columns"],
    }
    # Mirrors the original writer's execute_phases envelope, without its data.
    envelope = {
        **content,
        "schema": "cfeg.context-template-source.projection.v1",
        "study_id": "context-template-source39-v1",
        "plan_sha256": "1" * 64,
        "source_commit": "2" * 40,
        "start_sha256": "3" * 64,
    }
    encoded = producer.json_bytes(envelope)
    path = tmp_path / "projection.json"
    path.write_bytes(encoded)
    fake_hash = hashlib.sha256(encoded).hexdigest()
    monkeypatch.setattr(producer, "PROJECTION_PATH", path)
    monkeypatch.setattr(producer, "PROJECTION_SHA256", fake_hash)
    plan = copy.deepcopy(plan)
    plan["source_projection"].update(path=str(path), sha256=fake_hash)
    assert producer.load_projection(plan) == envelope
    assert core.projection_arrays(content, plan)[0].shape == (39, 2, 10, 8)

    provenance = {
        "plan_sha256": producer.PLAN_SHA256,
        "source_commit": "4" * 40,
        "source_tree": "5" * 40,
        "upstream_revision": producer.REVISION,
    }
    monkeypatch.setattr(producer, "preflight", lambda *args: (plan, provenance))
    monkeypatch.setattr(producer, "import_native", lambda path: {})
    monkeypatch.setattr(producer, "import_core", lambda: core)
    monkeypatch.setattr(producer.sys, "addaudithook", lambda hook: None)

    def forbidden_eeg(*args):
        raise AssertionError("EEG gathering must not occur after projection rejection")

    monkeypatch.setattr(producer, "gather_participants", forbidden_eeg)
    output = tmp_path / "attempt"
    with pytest.raises(ValueError, match="Projection does not match source-only contract"):
        producer.run(tmp_path / "plan.json", output, tmp_path / "upstream")
    assert [p.name for p in output.iterdir()] == ["start.json"]
    assert (output / "start.json").stat().st_mode & 0o777 == 0o400
