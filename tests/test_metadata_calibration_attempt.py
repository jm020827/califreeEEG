from __future__ import annotations

from pathlib import Path

import pytest

from cfeg.metadata_calibration_attempt import (
    OutcomeAccessAlreadyClaimed,
    build_outcome_access_scope,
    claim_outcome_access,
    outcome_access_started_for_run,
)
from cfeg.metadata_calibration_authority import (
    canonical_json_bytes,
    generate_authority_keypair,
    issue_signed_record,
)


def _manifest(run_root: Path, *, source_commit: str = "1" * 40) -> dict:
    subjects = [
        "sub004", "sub006", "sub008", "sub011", "sub014", "sub021", "sub022",
        "sub025", "sub028", "sub029", "sub030", "sub031", "sub032", "sub033",
        "sub037", "sub041", "sub042", "sub043", "sub044", "sub046", "sub054",
        "sub055", "sub056", "sub061", "sub063", "sub065", "sub067", "sub073",
        "sub074", "sub077", "sub080", "sub082", "sub083", "sub084", "sub089",
        "sub092", "sub097", "sub100", "sub102",
    ]
    return {
        "candidate_id": "metadata-calibration-efficiency-v1",
        "phase": "source_development",
        "phase_manifest_sha256": "2" * 64,
        "execution_contract_sha256": "3" * 64,
        "execution_output_root": str(run_root / "source" / "execution"),
        "source_commit": source_commit,
        "source_tree_sha256": "4" * 64,
        "source_tag": "metadata-calibration-freeze-test",
        "execution_contract": {
            "candidate_id": "metadata-calibration-efficiency-v1",
            "phase": "source_development",
            "phase_spec": {
                "n_classes": 12,
                "conditions": ["dry", "wet"],
                "query_identity_bundle_sha256": (
                    "224d8239aef5d0f9dd04c388f8ed8d2f961fc150c1cf72741f5ab22b756f80de"
                ),
                "target_subject_ids_by_checkpoint_group": {
                    "fold0": subjects[:13],
                    "fold1": subjects[13:26],
                    "fold2": subjects[26:],
                },
            },
        },
    }


def _authority_for_run(run_root: Path, *, password: bytes) -> tuple[Path, Path, str]:
    private_key = run_root / "keys" / "authority-signing-private.pem"
    public_key = run_root / "keys" / "authority-signing-public.pem"
    key_id = generate_authority_keypair(
        private_key_path=private_key,
        public_key_path=public_key,
        password=password,
    )
    return private_key, public_key, key_id


def _private_seal(
    run_root: Path,
    *,
    private_key: Path,
    key_id: str,
    password: bytes,
) -> tuple[dict, Path]:
    record = {
        "schema": "cfeg.metadata-calibration-filesystem-private-seal.v1",
        "candidate_id": "metadata-calibration-efficiency-v1",
        "phase": "source_development",
        "execution_output_root": str(run_root / "source" / "execution"),
        "filesystem_private_tree_sha256": "6" * 64,
        "outcome_decision_signed_record_sha256": "7" * 64,
        "execution_contract_sha256": "3" * 64,
        "source_commit": "1" * 40,
        "source_tree_sha256": "4" * 64,
        "source_tag": "metadata-calibration-freeze-test",
        "signing_key_id": key_id,
        "signer_role": "delegated_automation_authority",
        "authorization_basis": "owner-delegated-test",
        "signature_algorithm": "ed25519",
    }
    signed = issue_signed_record(
        record,
        private_key_path=private_key,
        private_key_password=password,
        hash_field="private_seal_payload_sha256",
    )
    path = run_root / "source" / "private" / "filesystem-private-seal.json"
    path.parent.mkdir(parents=True)
    path.write_bytes(canonical_json_bytes(signed) + b"\n")
    path.chmod(0o400)
    return signed, path


def test_scope_is_stable_across_run_source_and_fold_order_changes(tmp_path: Path) -> None:
    first = _manifest(tmp_path / "run-a")
    second = _manifest(tmp_path / "run-b", source_commit="9" * 40)
    original = first["execution_contract"]["phase_spec"][
        "target_subject_ids_by_checkpoint_group"
    ]
    second["execution_contract"]["phase_spec"][
        "target_subject_ids_by_checkpoint_group"
    ] = {key: list(reversed(value)) for key, value in reversed(original.items())}

    assert build_outcome_access_scope(first) == build_outcome_access_scope(second)


def test_second_same_cohort_claim_is_denied_across_run_roots(tmp_path: Path) -> None:
    password = b"p" * 48
    registry = tmp_path / "registry"
    first_run = tmp_path / "run-a"
    second_run = tmp_path / "run-b"
    first_key, first_public, first_key_id = _authority_for_run(
        first_run, password=password
    )
    first_seal, first_private_path = _private_seal(
        first_run, private_key=first_key, key_id=first_key_id, password=password
    )
    first_decision = {
        "signed_record_sha256": first_seal[
            "outcome_decision_signed_record_sha256"
        ],
        "decision_id": "DEC-test",
        "decision_date": "2026-09-05",
        "signing_key_id": first_key_id,
        "signer_role": "delegated_automation_authority",
    }

    first = claim_outcome_access(
        manifest=_manifest(first_run),
        decision=first_decision,
        signed_private_seal=first_seal,
        private_seal_path=first_private_path,
        registry_root=registry,
        trusted_signing_public_key_path=first_public,
        authority_signing_private_key_path=first_key,
        authority_signing_private_key_password=password,
        authorization_basis="owner-delegated-test",
    )
    assert first.claim_path.is_file()
    assert outcome_access_started_for_run(first_run, registry_root=registry)
    assert not outcome_access_started_for_run(second_run, registry_root=registry)

    second_key, second_public, second_key_id = _authority_for_run(
        second_run, password=password
    )
    second_seal, second_private_path = _private_seal(
        second_run, private_key=second_key, key_id=second_key_id, password=password
    )
    second_decision = {**first_decision, "signing_key_id": second_key_id}

    with pytest.raises(OutcomeAccessAlreadyClaimed) as blocked:
        claim_outcome_access(
            manifest=_manifest(second_run, source_commit="9" * 40),
            decision=second_decision,
            signed_private_seal=second_seal,
            private_seal_path=second_private_path,
            registry_root=registry,
            trusted_signing_public_key_path=second_public,
            authority_signing_private_key_path=second_key,
            authority_signing_private_key_password=password,
            authorization_basis="owner-delegated-test",
        )
    assert blocked.value.claim_path == first.claim_path
