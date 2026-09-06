from __future__ import annotations

import builtins
import os
import stat
import subprocess
from pathlib import Path

import numpy as np
import pytest

import cfeg.metadata_calibration_v3_governance as gov


def test_pure_parser_uses_only_supplied_immutable_bytes(monkeypatch: pytest.MonkeyPatch) -> None:
    path = "/declared/artifact.json"
    value = gov.seal_payload({"schema": "example.artifact.v1", "nested": {"value": 7}})
    data = gov.artifact_bytes(value)
    specifications = {
        path: gov.ArtifactSpec(
            schema="example.artifact.v1",
            exact_fields=frozenset({"schema", "nested", "payload_sha256"}),
        )
    }

    def forbidden(*args, **kwargs):
        raise AssertionError("pure parser attempted hidden I/O or RNG access")

    monkeypatch.setattr(builtins, "open", forbidden)
    monkeypatch.setattr(os, "open", forbidden)
    monkeypatch.setattr(os, "getenv", forbidden)
    monkeypatch.setattr(subprocess, "run", forbidden)
    monkeypatch.setattr(Path, "read_bytes", forbidden)
    monkeypatch.setattr(np.random, "default_rng", forbidden)
    monkeypatch.setattr(np.random, "SeedSequence", forbidden)

    parsed = gov.parse_exact_artifacts({path: data}, specifications)
    assert parsed[path]["nested"]["value"] == 7
    with pytest.raises(TypeError):
        parsed[path]["nested"]["value"] = 8
    with pytest.raises(TypeError):
        parsed[path] = value


def test_exact_loader_rejects_undeclared_symlink_and_hardlink_paths(tmp_path: Path) -> None:
    regular = tmp_path / "regular.json"
    regular.write_bytes(b"regular")
    regular.chmod(0o400)
    loaded = gov.load_exact_artifact_bytes([regular], declared_paths=[regular])
    assert loaded[str(regular)] == b"regular"

    with pytest.raises(gov.ValidationError, match="declared path set"):
        gov.load_exact_artifact_bytes([regular], declared_paths=[regular, tmp_path / "missing"])

    symlink = tmp_path / "alias.json"
    symlink.symlink_to(regular)
    with pytest.raises(OSError):
        gov.load_exact_artifact_bytes([symlink], declared_paths=[symlink])

    hardlink = tmp_path / "hardlink.json"
    os.link(regular, hardlink)
    with pytest.raises(gov.ValidationError, match="singly linked"):
        gov.load_exact_artifact_bytes([regular], declared_paths=[regular])


def test_exact_loader_rejects_normalized_aliases_and_weak_file_modes(tmp_path: Path) -> None:
    artifact = tmp_path / "artifact.json"
    artifact.write_bytes(b"payload")
    artifact.chmod(0o400)
    alias = f"{tmp_path}//artifact.json"
    with pytest.raises(gov.ValidationError, match="canonical POSIX"):
        gov.load_exact_artifact_bytes([alias], declared_paths=[alias])

    artifact.chmod(0o600)
    with pytest.raises(gov.ValidationError, match="mode-0400"):
        gov.load_exact_artifact_bytes([artifact], declared_paths=[artifact])

    artifact.chmod(0o400)
    tmp_path.chmod(0o755)
    with pytest.raises(gov.PublicationError, match="mode must be 0700"):
        gov.load_exact_artifact_bytes([artifact], declared_paths=[artifact])


def test_write_once_publication_uses_private_modes_and_candidate_wide_path(tmp_path: Path) -> None:
    root = tmp_path / "v3-artifacts"
    first = gov.seal_payload(
        {
            "schema": gov.GLOBAL_CLAIM_SCHEMA,
            "candidate_id": gov.CANDIDATE_ID,
            "attempt_id": "sha256-" + "1" * 64,
        }
    )
    first_bytes = gov.artifact_bytes(first)
    published = gov._publish_write_once_under_test_root(
        root,
        "scientific/global-claim.json",
        first_bytes,
        create_root=True,
    )
    assert published.path == root / "scientific/global-claim.json"
    assert published.path.read_bytes() == first_bytes
    assert stat.S_IMODE(root.stat().st_mode) == 0o700
    assert stat.S_IMODE((root / "scientific").stat().st_mode) == 0o700
    assert stat.S_IMODE(published.path.stat().st_mode) == 0o400
    reopened = gov.load_exact_artifact_bytes(
        [published.path],
        declared_paths=[published.path],
    )
    assert reopened[os.fspath(published.path)] == first_bytes
    assert gov.parse_artifact_bytes(
        reopened[os.fspath(published.path)],
        gov.ArtifactSpec(
            schema=gov.GLOBAL_CLAIM_SCHEMA,
            exact_fields=frozenset(first),
        ),
    ) == first

    second = gov.seal_payload(
        {
            "schema": gov.GLOBAL_CLAIM_SCHEMA,
            "candidate_id": gov.CANDIDATE_ID,
            "attempt_id": "sha256-" + "2" * 64,
        }
    )
    with pytest.raises(gov.PublicationError, match="already consumed"):
        gov._publish_write_once_under_test_root(
            root,
            "scientific/global-claim.json",
            gov.artifact_bytes(second),
        )
    assert published.path.read_bytes() == first_bytes


def test_symlinked_parent_and_nonprivate_root_fail_closed(tmp_path: Path) -> None:
    outside = tmp_path / "outside"
    outside.mkdir(mode=0o700)
    root = tmp_path / "root"
    root.mkdir(mode=0o700)
    (root / "scientific").symlink_to(outside, target_is_directory=True)
    with pytest.raises(OSError):
        gov._publish_write_once_under_test_root(
            root, "scientific/global-claim.json", b"payload"
        )
    assert not (outside / "global-claim.json").exists()

    weak_root = tmp_path / "weak-root"
    weak_root.mkdir(mode=0o755)
    with pytest.raises(gov.PublicationError, match="mode must be 0700"):
        gov._publish_write_once_under_test_root(weak_root, "artifact.json", b"payload")

    with pytest.raises(gov.PublicationError, match="V2 artifact root"):
        gov._publish_write_once_under_test_root(
            "/home/whwovy/v2-artifacts",
            "forbidden.json",
            b"payload",
        )


def test_partial_write_consumes_path_and_cannot_be_retried(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "partial-root"
    root.mkdir(mode=0o700)
    real_write = os.write

    def no_progress(descriptor: int, data: bytes) -> int:
        del descriptor, data
        return 0

    monkeypatch.setattr(os, "write", no_progress)
    with pytest.raises(gov.PublicationError, match="short write"):
        gov._publish_write_once_under_test_root(
            root, "result.json", b"complete-payload"
        )
    monkeypatch.setattr(os, "write", real_write)

    consumed = root / "result.json"
    assert consumed.exists()
    assert consumed.stat().st_size == 0
    with pytest.raises(gov.PublicationError, match="already consumed"):
        gov._publish_write_once_under_test_root(root, "result.json", b"retry-forbidden")


def test_low_level_publication_cannot_consume_production_or_alias_paths(
    tmp_path: Path,
) -> None:
    assert not hasattr(gov, "publish_write_once")
    with pytest.raises(gov.AuthorityError, match="production namespace"):
        gov._publish_write_once_under_test_root(
            gov.V3_CANONICAL_ROOT,
            "scientific/global-claim.json",
            b"hostile",
            create_root=True,
        )
    with pytest.raises(gov.AuthorityError, match="validated publisher"):
        gov._publish_write_once(
            tmp_path,
            "artifact.json",
            b"hostile",
            create_root=True,
            _publisher=object(),
        )
    for alias in ("a//b.json", "a/./b.json", "a/b.json/"):
        with pytest.raises(gov.ValidationError, match="canonical nonempty relative"):
            gov._publish_write_once_under_test_root(
                tmp_path / "aliases",
                alias,
                b"hostile",
                create_root=True,
            )
    assert not (tmp_path / "aliases").exists()
