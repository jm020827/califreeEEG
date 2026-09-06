from __future__ import annotations

import builtins
import os
import stat
import subprocess
import tempfile
from collections.abc import Iterator
from pathlib import Path

import numpy as np
import pytest

import cfeg.metadata_calibration_v3_governance as gov


@pytest.fixture
def private_test_path() -> Iterator[Path]:
    with tempfile.TemporaryDirectory(prefix="cfeg-v3-test-", dir="/tmp") as raw:
        path = Path(raw)
        path.chmod(0o700)
        yield path


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


def test_write_once_publication_uses_private_modes_and_candidate_wide_path(
    private_test_path: Path,
) -> None:
    root = private_test_path / "v3-artifacts"
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
    assert (
        gov.parse_artifact_bytes(
            reopened[os.fspath(published.path)],
            gov.ArtifactSpec(
                schema=gov.GLOBAL_CLAIM_SCHEMA,
                exact_fields=frozenset(first),
            ),
        )
        == first
    )

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


def test_symlinked_parent_and_nonprivate_root_fail_closed(
    private_test_path: Path,
) -> None:
    outside = private_test_path / "outside"
    outside.mkdir(mode=0o700)
    root = private_test_path / "root"
    root.mkdir(mode=0o700)
    (root / "scientific").symlink_to(outside, target_is_directory=True)
    with pytest.raises(OSError):
        gov._publish_write_once_under_test_root(root, "scientific/global-claim.json", b"payload")
    assert not (outside / "global-claim.json").exists()

    weak_root = private_test_path / "weak-root"
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
    private_test_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = private_test_path / "partial-root"
    root.mkdir(mode=0o700)
    real_write = os.write

    def no_progress(descriptor: int, data: bytes) -> int:
        del descriptor, data
        return 0

    monkeypatch.setattr(os, "write", no_progress)
    with pytest.raises(gov.PublicationError, match="short write"):
        gov._publish_write_once_under_test_root(root, "result.json", b"complete-payload")
    monkeypatch.setattr(os, "write", real_write)

    consumed = root / "result.json"
    assert consumed.exists()
    assert consumed.stat().st_size == 0
    with pytest.raises(gov.PublicationError, match="already consumed"):
        gov._publish_write_once_under_test_root(root, "result.json", b"retry-forbidden")


def test_low_level_publication_cannot_consume_production_or_alias_paths(
    private_test_path: Path,
    monkeypatch: pytest.MonkeyPatch,
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
            private_test_path,
            "artifact.json",
            b"hostile",
            create_root=True,
            _publisher=object(),
        )
    guarded = private_test_path / "guarded"
    monkeypatch.setattr(gov, "_ACTIVE_GOVERNED_PROCESS_CAPABILITY", None)
    with pytest.raises(gov.AuthorityError, match="GovernedProcessCapability"):
        gov._publish_write_once(
            guarded,
            "artifact.json",
            b"hostile",
            create_root=True,
            _publisher=gov._PUBLICATION_ISSUER,
        )
    assert not guarded.exists()
    for alias in ("a//b.json", "a/./b.json", "a/b.json/"):
        with pytest.raises(gov.ValidationError, match="canonical nonempty relative"):
            gov._publish_write_once_under_test_root(
                private_test_path / "aliases",
                alias,
                b"hostile",
                create_root=True,
            )
    assert not (private_test_path / "aliases").exists()
    with pytest.raises(gov.AuthorityError, match="exact canonical path"):
        gov._validate_context_reference_at_path(
            private_test_path / "context-reference.json",
            development_bundle=object(),  # type: ignore[arg-type]
            rng_authority=object(),
            core_capability=object(),
            scope="canonical",
        )
    with pytest.raises(gov.AuthorityError, match="scope is not recognized"):
        gov._validate_development_bundle_at_path(
            private_test_path / "development-bundle.json",
            snapshot=object(),  # type: ignore[arg-type]
            frozen_file_bytes={},
            focused_test_run=object(),  # type: ignore[arg-type]
            scope="caller-selected",
        )


def test_ordinary_python_cannot_enter_production_authority_gateways(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(gov, "_ACTIVE_GOVERNED_PROCESS_CAPABILITY", None)
    probe = object()
    calls = (
        lambda: gov.build_development_bundle(
            snapshot=probe,  # type: ignore[arg-type]
            frozen_file_bytes={},
            created_at_UTC="2026-09-06T12:00:00Z",
            focused_test_run=probe,  # type: ignore[arg-type]
        ),
        lambda: gov.publish_development_bundle(
            snapshot=probe,  # type: ignore[arg-type]
            frozen_file_bytes={},
            created_at_UTC="2026-09-06T12:00:00Z",
            focused_test_run=probe,  # type: ignore[arg-type]
        ),
        lambda: gov.validate_development_bundle(
            snapshot=probe,  # type: ignore[arg-type]
            frozen_file_bytes={},
            focused_test_run=probe,  # type: ignore[arg-type]
        ),
        gov.reopen_development_bundle,
        lambda: gov.validate_context_reference(
            development_bundle=probe,  # type: ignore[arg-type]
            rng_authority=probe,
            core_capability=probe,
        ),
        lambda: gov.publish_development_start(
            development_bundle=probe,  # type: ignore[arg-type]
            context_reference=probe,  # type: ignore[arg-type]
            started_at_UTC="2026-09-07T00:00:00Z",
        ),
        lambda: gov.reopen_development_start_receipt(
            development_bundle=probe,  # type: ignore[arg-type]
            context_reference=probe,  # type: ignore[arg-type]
        ),
        lambda: gov.validate_development_result(
            development_bundle=probe,  # type: ignore[arg-type]
            context_reference=probe,  # type: ignore[arg-type]
            development_start=probe,  # type: ignore[arg-type]
            development_rng_authority=probe,
            validated_context_reference=probe,
            core_capability=probe,
        ),
        lambda: gov.observe_canonical_development_result_for_a_recovery(
            development_bundle=probe,  # type: ignore[arg-type]
            context_reference=probe,  # type: ignore[arg-type]
        ),
        lambda: gov.validate_bundle_selected_delta_recovery(
            development_bundle=probe,  # type: ignore[arg-type]
            snapshot=probe,  # type: ignore[arg-type]
        ),
        lambda: gov.run_observed_test(
            "focused_v3",
            snapshot=probe,  # type: ignore[arg-type]
        ),
        lambda: gov.publish_test_evidence(
            {},
            selected_method_freeze=probe,  # type: ignore[arg-type]
            observed_runs=(),
            snapshot=probe,  # type: ignore[arg-type]
        ),
        lambda: gov.publish_canary_authorization(
            {},
            selected_method_freeze=probe,  # type: ignore[arg-type]
            test_evidence=probe,  # type: ignore[arg-type]
        ),
        gov.run_fresh_canary_audit_in_exec_subprocess,
    )
    for call in calls:
        with pytest.raises(gov.AuthorityError, match="GovernedProcessCapability"):
            call()


def test_test_root_ancestor_and_resolved_destination_cannot_escape_to_production(
    private_test_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    reached_writer = False

    def forbidden_writer(*_args: object, **_kwargs: object) -> None:
        nonlocal reached_writer
        reached_writer = True

    monkeypatch.setattr(gov, "_publish_write_once", forbidden_writer)
    attacks = (
        (
            Path("/home/whwovy"),
            ("v3-artifacts/metadata-calibration-efficiency-v3/governance-canary-v1/result.json"),
        ),
        (
            Path("/home/whwovy"),
            "califreeEEG/src/cfeg/metadata_calibration_v3_governance.py",
        ),
    )
    for root, relative in attacks:
        with pytest.raises(gov.AuthorityError, match="production namespace"):
            gov._publish_write_once_under_test_root(
                root,
                relative,
                b"must not reach writer",
                create_root=True,
            )
    assert not reached_writer

    resolved_alias = private_test_path / "resolved-home-alias"
    resolved_alias.symlink_to("/home/whwovy", target_is_directory=True)
    for attempt in ("development-v1", "development-v2", "development-v3", "development-v4"):
        with pytest.raises(gov.AuthorityError, match="production namespace"):
            gov._publish_write_once_under_test_root(
                resolved_alias,
                f"v3-artifacts/metadata-calibration-efficiency-v3/{attempt}/"
                "development-result.json",
                b"must not reach writer",
            )
    assert not reached_writer

    with pytest.raises(gov.AuthorityError, match="owner-private namespace below /tmp"):
        gov._publish_write_once_under_test_root(
            "/var/tmp/cfeg-v3-governance-test",
            "artifact.json",
            b"must not reach writer",
            create_root=True,
        )
    assert not reached_writer
