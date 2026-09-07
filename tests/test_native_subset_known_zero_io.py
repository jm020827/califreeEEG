"""Artificial publication and runtime-boundary checks; no human inputs."""

import copy
import importlib.util
import json
import os
import stat
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "kz_io", ROOT / "scripts/native_subset_known_zero_io.py"
)
IO = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(IO)


@pytest.fixture
def plan():
    return IO.load_plan()


def test_plan_pin_and_no_old_policy_changes(plan, tmp_path, monkeypatch):
    assert plan["authority"]["policy_refit"] is False
    assert plan["authority"]["human_outcome_execution"] is True
    assert plan["randomization_namespace"] == "native-subset-m-source39-v1"
    path = tmp_path / "plan.json"
    path.write_bytes(IO.PLAN_PATH.read_bytes() + b" ")
    monkeypatch.setattr(IO, "PLAN_PATH", path)
    with pytest.raises(ValueError, match="plan hash"):
        IO.load_plan(path)


@pytest.mark.parametrize("filename", ["start.json", "result.json"])
def test_exclusive_immutable_publication(tmp_path, filename):
    path = tmp_path / filename
    value = {"synthetic": True}
    digest = IO.publish(path, value, 1024)
    raw, info = IO.artifact_bytes(path)
    assert json.loads(raw) == value and IO.digest(raw) == digest
    assert stat.S_IMODE(info.st_mode) == 0o400 and info.st_nlink == 1
    with pytest.raises(FileExistsError):
        IO.publish(path, value, 1024)
    assert path.read_bytes() == raw


def test_quota_nan_and_symlink_reject_before_write(tmp_path):
    path = tmp_path / "result.json"
    with pytest.raises(ValueError, match="budget"):
        IO.publish(path, {"synthetic": True}, 1)
    with pytest.raises(ValueError):
        IO.publish(path, {"x": float("nan")}, 1024)
    assert not path.exists()
    target = tmp_path / "existing.json"
    IO.publish(target, {}, 1024)
    path.symlink_to(target)
    with pytest.raises((FileExistsError, OSError)):
        IO.publish(path, {}, 1024)
    with pytest.raises(ValueError, match="canonical"):
        IO.artifact_bytes(path)


@pytest.mark.parametrize("damage", ["writable", "hardlink"])
def test_artifact_permission_and_link_contract(tmp_path, damage):
    path = tmp_path / "start.json"
    IO.publish(path, {}, 1024)
    if damage == "writable":
        path.chmod(0o600)
    else:
        os.link(path, tmp_path / "linked.json")
    with pytest.raises(ValueError, match="regular0400"):
        IO.artifact_bytes(path)


def test_guard_allows_only_bound_parent_and_new_outputs(plan):
    guard = IO.guard_for(plan)
    for name in plan["parent"]["artifacts"]:
        guard("open", (plan["parent"]["root"] + "/" + name, "rb", os.O_RDONLY))
    for name in plan["artifacts"]:
        guard("open", (plan["output_root"] + "/" + name, "wb", os.O_WRONLY | os.O_CREAT))
    guard("os.mkdir", (plan["output_root"], 0o700, -1))
    guard("os.listdir", (plan["parent"]["root"],))


@pytest.mark.parametrize(
    "path",
    [
        "/home/whwovy/eeg-data/raw/wearable/S004.mat",
        "/home/whwovy/eeg-data/raw/wearable/S005.mat",
        "/home/whwovy/eeg-data/raw/wearable/Impedance.mat",
        "/home/whwovy/native-subset-m-artifacts/source39-v1/start.json",
        "/home/whwovy/author-etrca-artifacts/source39-v1/correlations.npz",
        "/tmp/unapproved.json",
        "/tmp/unapproved.npy",
        "/tmp/unapproved.txt",
    ],
)
def test_guard_rejects_all_unapproved_reads(plan, path):
    with pytest.raises(ValueError, match="Unapproved runtime read"):
        IO.guard_for(plan)("open", (path, "rb", os.O_RDONLY))


@pytest.mark.parametrize(
    "event,args",
    [
        ("subprocess.Popen", ("git",)),
        ("socket.connect", ()),
        ("os.system", ("true",)),
        ("os.remove", ("/tmp/keep",)),
        ("os.rename", ("/tmp/a", "/tmp/b")),
        ("os.mkdir", ("/tmp/unapproved", 0o700, -1)),
        ("os.listdir", ("/home/whwovy",)),
        ("os.chmod", ("/tmp/keep", 0o600, -1)),
    ],
)
def test_guard_rejects_other_runtime_operations(plan, event, args):
    with pytest.raises((ValueError, RuntimeError)):
        IO.guard_for(plan)(event, args)


def test_guard_rejects_parent_or_unapproved_writes(plan):
    guard = IO.guard_for(plan)
    for path in (plan["parent"]["root"] + "/result.json", "/tmp/new.txt"):
        with pytest.raises(ValueError, match="artifact writes"):
            guard("open", (path, "wb", os.O_WRONLY | os.O_TRUNC))


def test_bad_parent_hash_fails_before_json_parse(plan, tmp_path):
    changed = copy.deepcopy(plan)
    changed["parent"]["root"] = str(tmp_path)
    for name in changed["parent"]["artifacts"]:
        IO.publish(tmp_path / name, {"synthetic": True}, 4096)
    # Exact file names and permissions, but unauthenticated bytes: no parsing.
    with pytest.raises(ValueError, match="Parent artifact hash"):
        IO.load_parent(changed, {}, {}, {}, "2026-09-08T00:00:00+00:00")


def test_guard_rejects_lexical_library_escape_without_open(plan):
    path = str(ROOT / ".venv/lib/../../../eeg-data/raw/wearable/S001.mat")
    with pytest.raises(ValueError, match="lexical path traversal"):
        IO.guard_for(plan)("open", (path, "rb", os.O_RDONLY))


def test_guard_rejects_library_symlink_escape(plan, tmp_path, monkeypatch):
    library = tmp_path / ".venv/lib"
    library.mkdir(parents=True)
    target = tmp_path / "outside.txt"
    target.write_text("artificial")
    path = library / "link.txt"
    path.symlink_to(target)
    monkeypatch.setattr(IO, "ROOT", tmp_path)
    with pytest.raises(ValueError, match="symlink path traversal"):
        IO.guard_for(plan)("open", (str(path), "rb", os.O_RDONLY))


def test_read_byte_bound_precedes_payload_read(tmp_path):
    path = tmp_path / "bounded.json"
    IO.publish(path, {"artificial": 12345}, 1024)
    with pytest.raises(ValueError, match="byte bound before read"):
        IO.artifact_bytes(path, max_bytes=1)
