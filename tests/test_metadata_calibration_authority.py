from __future__ import annotations

import copy
import os

import pytest

from cfeg.metadata_calibration_authority import (
    decrypt_json_for_finalizer,
    encrypt_json_for_finalizer,
    generate_authority_keypair,
    generate_finalizer_keypair,
    issue_signed_record,
    verify_signed_record,
)


def test_signed_record_rejects_payload_and_key_drift(tmp_path) -> None:
    authority_dir = tmp_path / "authority"
    private_path = authority_dir / "owner-private.pem"
    public_path = authority_dir / "owner-public.pem"
    key_id = generate_authority_keypair(
        private_key_path=private_path,
        public_key_path=public_path,
    )
    assert os.stat(private_path).st_mode & 0o777 == 0o400
    record = {
        "schema": "example.v1",
        "signing_key_id": key_id,
        "signature_algorithm": "ed25519",
        "signer_role": "delegated_automation_authority",
        "authorization_basis": "test fixture",
        "decision": "approved",
    }
    signed = issue_signed_record(
        record,
        private_key_path=private_path,
        hash_field="decision_payload_sha256",
    )
    assert verify_signed_record(
        signed,
        public_key_path=public_path,
        hash_field="decision_payload_sha256",
        expected_schema="example.v1",
    ) == signed["signed_record_sha256"]

    tampered = copy.deepcopy(signed)
    tampered["decision"] = "denied"
    with pytest.raises(ValueError, match="reference hash"):
        verify_signed_record(
            tampered,
            public_key_path=public_path,
            hash_field="decision_payload_sha256",
        )


def test_finalizer_encryption_roundtrip_and_aad_tamper(tmp_path) -> None:
    key_dir = tmp_path / "finalizer"
    private_path = key_dir / "label-private.pem"
    public_path = key_dir / "label-public.pem"
    generate_finalizer_keypair(
        private_key_path=private_path,
        public_key_path=public_path,
    )
    aad = {"phase": "source_development", "seal": "a" * 64}
    secret = {"rows": [{"query_token": "q_" + "b" * 64, "label": 2}]}
    envelope = encrypt_json_for_finalizer(
        secret,
        public_key_path=public_path,
        associated_data=aad,
    )
    assert decrypt_json_for_finalizer(
        envelope,
        private_key_path=private_path,
        expected_associated_data=aad,
    ) == secret

    tampered = copy.deepcopy(envelope)
    tampered["ciphertext_base64"] = "A" + tampered["ciphertext_base64"][1:]
    with pytest.raises(ValueError, match="self-hash"):
        decrypt_json_for_finalizer(
            tampered,
            private_key_path=private_path,
            expected_associated_data=aad,
        )
    with pytest.raises(ValueError, match="associated-data"):
        decrypt_json_for_finalizer(
            envelope,
            private_key_path=private_path,
            expected_associated_data={"phase": "held_participant_evaluation"},
        )


def test_encrypted_finalizer_private_key_requires_in_memory_password(tmp_path) -> None:
    key_dir = tmp_path / "encrypted-finalizer"
    private_path = key_dir / "label-private.pem"
    public_path = key_dir / "label-public.pem"
    password = os.urandom(48)
    generate_finalizer_keypair(
        private_key_path=private_path,
        public_key_path=public_path,
        password=password,
    )
    aad = {"phase": "source_development", "seal": "c" * 64}
    envelope = encrypt_json_for_finalizer(
        {"rows": [{"query_token": "q_" + "d" * 64, "label": 1}]},
        public_key_path=public_path,
        associated_data=aad,
    )
    with pytest.raises((TypeError, ValueError)):
        decrypt_json_for_finalizer(
            envelope,
            private_key_path=private_path,
            expected_associated_data=aad,
        )
    assert decrypt_json_for_finalizer(
        envelope,
        private_key_path=private_path,
        private_key_password=password,
        expected_associated_data=aad,
    )["rows"][0]["label"] == 1


def test_key_generation_refuses_overwrite(tmp_path) -> None:
    private_path = tmp_path / "keys" / "private.pem"
    public_path = tmp_path / "keys" / "public.pem"
    generate_authority_keypair(
        private_key_path=private_path,
        public_key_path=public_path,
    )
    with pytest.raises(FileExistsError):
        generate_authority_keypair(
            private_key_path=private_path,
            public_key_path=public_path,
        )
