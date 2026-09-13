"""Generated operator checks only; never launch workers, open MAT or fit gates."""

import importlib.util
import json
from pathlib import Path

import pytest

SPEC = importlib.util.spec_from_file_location(
    "mamem_source_runner",
    Path(__file__).resolve().parents[1] / "scripts/analysis/run_mamem_source_v1.py",
)
runner = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(runner)


def test_json_is_exclusive_and_bounded(tmp_path):
    path = tmp_path / "receipt.json"
    runner.save(path, {"attempt": 1})
    with pytest.raises(FileExistsError):
        runner.save(path, {"attempt": 2})
    with pytest.raises(ValueError, match="json_cap"):
        runner.save(tmp_path / "large.json", {"value": "x" * 100}, cap=10)
    assert json.loads(path.read_text()) == {"attempt": 1}


@pytest.mark.parametrize("phase", ["development", "anything", "", None])
def test_fit_child_phase_allowlist(phase):
    with pytest.raises(ValueError, match="fit_phase"):
        runner.claim_child("fit-child", phase)


@pytest.mark.parametrize("name", ["S001c", "S000a", "S012a", "../S001a", "", None])
def test_io_child_name_allowlist(name):
    with pytest.raises(ValueError, match="io_name"):
        runner.claim_child("io-child", name)


def test_child_requires_authorization_before_data(tmp_path, monkeypatch):
    monkeypatch.setattr(runner, "RUN", tmp_path)
    monkeypatch.setattr(runner, "child_limits", lambda kind: None)
    with pytest.raises(FileNotFoundError):
        runner.claim_child("fit-child", "real")
