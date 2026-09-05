from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from cfeg.metadata_calibration_authority import generate_authority_keypair
from cfeg.metadata_calibration_lifecycle import (
    _publish_signed_pending_noreplace,
    _validate_capability_path_disjointness,
    _write_failure_receipt,
    validate_metadata_calibration_lifecycle_failure,
)


def _authority(root: Path) -> tuple[Path, bytes, str]:
    keys = root / "keys"
    private = keys / "authority-signing-private.pem"
    public = keys / "authority-signing-public.pem"
    password = b"test-lifecycle-authority-password"
    key_id = generate_authority_keypair(
        private_key_path=private,
        public_key_path=public,
        password=password,
    )
    return private, password, key_id


@pytest.mark.parametrize(
    "claimed_phases,expected_status,redacted,retry_permitted",
    [
        (
            (),
            "failed_before_outcome_access_new_exact_run_permitted",
            False,
            True,
        ),
        (
            ("source_development",),
            "manual_audit_required_no_new_same_cohort_run",
            True,
            False,
        ),
    ],
)
def test_signed_lifecycle_failure_enforces_claim_aware_policy(
    tmp_path: Path,
    claimed_phases: tuple[str, ...],
    expected_status: str,
    redacted: bool,
    retry_permitted: bool,
) -> None:
    root = tmp_path / ("claimed" if claimed_phases else "preclaim")
    root.mkdir(mode=0o700)
    private, password, key_id = _authority(root)
    _write_failure_receipt(
        root,
        error=RuntimeError("sensitive failure detail"),
        decision_date="2026-09-05",
        authorization_basis="owner-approved exact experiment",
        authority_key_id=key_id,
        authority_private_key=private,
        authority_password=password,
        current_run_outcome_access_started=bool(claimed_phases),
        failed_phase="source_development",
        current_run_claimed_phases=claimed_phases,
        registry_integrity_failure=False,
        blocked_by_prior_claim=False,
        blocking_claim_path=None,
        blocking_scope_sha256=None,
    )

    receipt_path = root / "lifecycle_failure.json"
    assert validate_metadata_calibration_lifecycle_failure(receipt_path)
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    assert receipt["status"] == expected_status
    assert receipt["post_outcome_error_detail_redacted"] is redacted
    assert receipt["same_cohort_automatic_retry_permitted"] is retry_permitted
    assert (receipt["error_message"] is None) is redacted
    assert (receipt["error_detail_sha256"] is None) is redacted


def test_capability_paths_include_registry_lock_and_staging_siblings(
    tmp_path: Path,
) -> None:
    repository = tmp_path / "repository"
    asset = tmp_path / "asset"
    run = tmp_path / "run"
    registry = tmp_path / "registry"
    _validate_capability_path_disjointness(
        repository=repository,
        processed_asset_root=asset,
        run_root=run,
        outcome_access_registry_root=registry,
    )

    for colliding_run in (
        tmp_path / "registry-locks" / "run",
        tmp_path / "registry-staging" / "run",
    ):
        with pytest.raises(ValueError, match="Capability-bearing paths"):
            _validate_capability_path_disjointness(
                repository=repository,
                processed_asset_root=asset,
                run_root=colliding_run,
                outcome_access_registry_root=registry,
            )


def test_pending_completion_publication_is_noreplace_and_key_gated(
    tmp_path: Path,
) -> None:
    pending = tmp_path / "lifecycle_completion.pending.json"
    canonical = tmp_path / "lifecycle_completion.json"
    pending.write_text('{"signed":true}\n', encoding="utf-8")
    os.chmod(pending, 0o400)
    private_paths = (tmp_path / "authority-private.pem", tmp_path / "finalizer.pem")

    private_paths[0].write_text("still present", encoding="utf-8")
    with pytest.raises(PermissionError, match="private key"):
        _publish_signed_pending_noreplace(
            pending,
            canonical,
            private_key_paths=private_paths,
        )
    private_paths[0].unlink()

    _publish_signed_pending_noreplace(
        pending,
        canonical,
        private_key_paths=private_paths,
    )
    assert not pending.exists()
    assert canonical.read_text(encoding="utf-8") == '{"signed":true}\n'

    replacement = tmp_path / "replacement.pending.json"
    replacement.write_text("replacement\n", encoding="utf-8")
    os.chmod(replacement, 0o400)
    with pytest.raises(FileExistsError, match="never replaces"):
        _publish_signed_pending_noreplace(
            replacement,
            canonical,
            private_key_paths=private_paths,
        )
