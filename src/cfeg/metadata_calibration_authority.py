from __future__ import annotations

import base64
import hashlib
import json
import os
import stat
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import ed25519, padding, rsa
from cryptography.hazmat.primitives.ciphers.aead import AESGCM

ED25519_ALGORITHM = "ed25519"
LABEL_ENCRYPTION_ALGORITHM = "RSA-OAEP-SHA256+AES-256-GCM"
ENCRYPTED_JSON_SCHEMA = "cfeg.encrypted-json.v1"


def canonical_json_bytes(value: Mapping[str, Any]) -> bytes:
    """Encode a JSON object without platform- or insertion-order ambiguity."""

    try:
        return json.dumps(
            dict(value),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise ValueError("Authority records must be finite canonical JSON objects.") from error


def canonical_json_sha256(value: Mapping[str, Any]) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def public_key_fingerprint_sha256(public_key: object) -> str:
    if not hasattr(public_key, "public_bytes"):
        raise TypeError("Expected a cryptography public-key object.")
    encoded = public_key.public_bytes(  # type: ignore[union-attr]
        encoding=serialization.Encoding.DER,
        format=serialization.PublicFormat.SubjectPublicKeyInfo,
    )
    return hashlib.sha256(encoded).hexdigest()


def generate_authority_keypair(
    *,
    private_key_path: str | Path,
    public_key_path: str | Path,
    password: bytes | None = None,
) -> str:
    """Create a research-specific Ed25519 keypair without overwriting a path."""

    private_path = Path(private_key_path)
    public_path = Path(public_key_path)
    _require_new_key_paths(private_path, public_path)
    private_key = ed25519.Ed25519PrivateKey.generate()
    if password is not None and len(password) < 32:
        raise ValueError("Authority private-key passwords must contain at least 32 bytes.")
    _write_new_private_key(
        private_path,
        private_key.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.PKCS8,
            encryption_algorithm=(
                serialization.NoEncryption()
                if password is None
                else serialization.BestAvailableEncryption(password)
            ),
        ),
    )
    _write_new_public_key(
        public_path,
        private_key.public_key().public_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PublicFormat.SubjectPublicKeyInfo,
        ),
    )
    return public_key_fingerprint_sha256(private_key.public_key())


def generate_finalizer_keypair(
    *,
    private_key_path: str | Path,
    public_key_path: str | Path,
    key_size: int = 3072,
    password: bytes | None = None,
) -> str:
    """Create an RSA key used only to encrypt finalizer-side label records."""

    if key_size < 3072:
        raise ValueError("Finalizer RSA keys must be at least 3072 bits.")
    private_path = Path(private_key_path)
    public_path = Path(public_key_path)
    _require_new_key_paths(private_path, public_path)
    if password is not None and len(password) < 32:
        raise ValueError("Finalizer private-key passwords must contain at least 32 bytes.")
    private_key = rsa.generate_private_key(public_exponent=65537, key_size=key_size)
    _write_new_private_key(
        private_path,
        private_key.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.PKCS8,
            encryption_algorithm=(
                serialization.NoEncryption()
                if password is None
                else serialization.BestAvailableEncryption(password)
            ),
        ),
    )
    _write_new_public_key(
        public_path,
        private_key.public_key().public_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PublicFormat.SubjectPublicKeyInfo,
        ),
    )
    return public_key_fingerprint_sha256(private_key.public_key())


def issue_signed_record(
    record: Mapping[str, Any],
    *,
    private_key_path: str | Path,
    private_key_password: bytes | None = None,
    hash_field: str,
    signature_field: str = "signature",
    signed_record_hash_field: str = "signed_record_sha256",
) -> dict[str, Any]:
    """Return an Ed25519-signed copy with a deterministic self-hash."""

    signed = dict(record)
    signed.pop(hash_field, None)
    signed.pop(signature_field, None)
    signed.pop(signed_record_hash_field, None)
    if signed.get("signature_algorithm") != ED25519_ALGORITHM:
        raise ValueError("Signed records must declare signature_algorithm='ed25519'.")
    if signed.get("signer_role") not in {
        "human_owner",
        "delegated_automation_authority",
    }:
        raise ValueError("Signed records must declare an explicit trusted signer_role.")
    private_key = _load_ed25519_private_key(
        private_key_path,
        password=private_key_password,
    )
    fingerprint = public_key_fingerprint_sha256(private_key.public_key())
    if signed.get("signing_key_id") != fingerprint:
        raise ValueError("Record signing-key ID differs from the supplied private key.")
    if not isinstance(signed.get("authorization_basis"), str) or not str(
        signed["authorization_basis"]
    ).strip():
        raise ValueError("Signed records must state their authorization_basis.")
    signed[hash_field] = canonical_json_sha256(signed)
    message = canonical_json_bytes(signed)
    signed[signature_field] = base64.b64encode(private_key.sign(message)).decode("ascii")
    signed[signed_record_hash_field] = canonical_json_sha256(signed)
    return signed


def verify_signed_record(
    record: Mapping[str, Any],
    *,
    public_key_path: str | Path,
    hash_field: str,
    signature_field: str = "signature",
    signed_record_hash_field: str = "signed_record_sha256",
    expected_schema: str | None = None,
) -> str:
    """Verify schema, fingerprint, deterministic hash, and Ed25519 signature."""

    observed = dict(record)
    signed_record_hash = observed.pop(signed_record_hash_field, None)
    if not isinstance(signed_record_hash, str):
        raise TypeError("Signed record reference hash must be a string.")
    if signed_record_hash != canonical_json_sha256(observed):
        raise ValueError("Signed-record reference hash mismatch.")
    if expected_schema is not None and observed.get("schema") != expected_schema:
        raise ValueError("Signed record has an unexpected schema.")
    if observed.get("signature_algorithm") != ED25519_ALGORITHM:
        raise ValueError("Signed record has an unsupported signature algorithm.")
    if observed.get("signer_role") not in {
        "human_owner",
        "delegated_automation_authority",
    }:
        raise ValueError("Signed record has an unsupported signer role.")
    signature_text = observed.pop(signature_field, None)
    claimed_hash = observed.pop(hash_field, None)
    if not isinstance(signature_text, str) or not isinstance(claimed_hash, str):
        raise TypeError("Signed record hash and signature must both be strings.")
    expected_hash = canonical_json_sha256(observed)
    if claimed_hash != expected_hash:
        raise ValueError("Signed record self-hash mismatch.")
    public_key = _load_ed25519_public_key(public_key_path)
    fingerprint = public_key_fingerprint_sha256(public_key)
    if observed.get("signing_key_id") != fingerprint:
        raise ValueError("Signed record key ID is not trusted by this public key.")
    if not isinstance(observed.get("authorization_basis"), str) or not str(
        observed["authorization_basis"]
    ).strip():
        raise ValueError("Signed record does not state an authorization basis.")
    message_record = dict(observed)
    message_record[hash_field] = claimed_hash
    try:
        signature = base64.b64decode(signature_text, validate=True)
        public_key.verify(signature, canonical_json_bytes(message_record))
    except Exception as error:
        raise ValueError("Signed record signature verification failed.") from error
    return signed_record_hash


def encrypt_json_for_finalizer(
    value: Mapping[str, Any],
    *,
    public_key_path: str | Path,
    associated_data: Mapping[str, Any],
) -> dict[str, Any]:
    """Hybrid-encrypt one JSON object for the label-capable finalizer."""

    public_key = _load_rsa_public_key(public_key_path)
    plaintext = canonical_json_bytes(value)
    aad = canonical_json_bytes(associated_data)
    data_key = AESGCM.generate_key(bit_length=256)
    nonce = os.urandom(12)
    ciphertext = AESGCM(data_key).encrypt(nonce, plaintext, aad)
    wrapped_key = public_key.encrypt(
        data_key,
        padding.OAEP(
            mgf=padding.MGF1(algorithm=hashes.SHA256()),
            algorithm=hashes.SHA256(),
            label=None,
        ),
    )
    envelope: dict[str, Any] = {
        "schema": ENCRYPTED_JSON_SCHEMA,
        "algorithm": LABEL_ENCRYPTION_ALGORITHM,
        "public_key_fingerprint_sha256": public_key_fingerprint_sha256(public_key),
        "associated_data": dict(associated_data),
        "associated_data_sha256": hashlib.sha256(aad).hexdigest(),
        "wrapped_key_base64": base64.b64encode(wrapped_key).decode("ascii"),
        "nonce_base64": base64.b64encode(nonce).decode("ascii"),
        "ciphertext_base64": base64.b64encode(ciphertext).decode("ascii"),
    }
    envelope["envelope_payload_sha256"] = canonical_json_sha256(envelope)
    return envelope


def decrypt_json_for_finalizer(
    envelope: Mapping[str, Any],
    *,
    private_key_path: str | Path,
    expected_associated_data: Mapping[str, Any],
    private_key_password: bytes | None = None,
) -> dict[str, Any]:
    """Decrypt and authenticate one finalizer-only JSON object."""

    observed = dict(envelope)
    claimed_envelope_hash = observed.pop("envelope_payload_sha256", None)
    if claimed_envelope_hash != canonical_json_sha256(observed):
        raise ValueError("Encrypted JSON envelope self-hash mismatch.")
    if observed.get("schema") != ENCRYPTED_JSON_SCHEMA:
        raise ValueError("Encrypted JSON envelope has an unexpected schema.")
    if observed.get("algorithm") != LABEL_ENCRYPTION_ALGORITHM:
        raise ValueError("Encrypted JSON envelope has an unsupported algorithm.")
    expected_aad = canonical_json_bytes(expected_associated_data)
    if observed.get("associated_data") != dict(expected_associated_data) or observed.get(
        "associated_data_sha256"
    ) != hashlib.sha256(expected_aad).hexdigest():
        raise ValueError("Encrypted JSON associated-data binding mismatch.")
    private_key = _load_rsa_private_key(
        private_key_path,
        password=private_key_password,
    )
    if observed.get("public_key_fingerprint_sha256") != public_key_fingerprint_sha256(
        private_key.public_key()
    ):
        raise ValueError("Encrypted JSON was not addressed to the supplied finalizer key.")
    try:
        data_key = private_key.decrypt(
            base64.b64decode(str(observed["wrapped_key_base64"]), validate=True),
            padding.OAEP(
                mgf=padding.MGF1(algorithm=hashes.SHA256()),
                algorithm=hashes.SHA256(),
                label=None,
            ),
        )
        plaintext = AESGCM(data_key).decrypt(
            base64.b64decode(str(observed["nonce_base64"]), validate=True),
            base64.b64decode(str(observed["ciphertext_base64"]), validate=True),
            expected_aad,
        )
    except Exception as error:
        raise ValueError("Encrypted JSON authentication or decryption failed.") from error
    try:
        value = json.loads(plaintext)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("Decrypted payload is not valid JSON.") from error
    if not isinstance(value, dict) or canonical_json_bytes(value) != plaintext:
        raise ValueError("Decrypted payload is not a canonical JSON object.")
    return value


def read_json_object(path: str | Path) -> dict[str, Any]:
    data = _read_regular_file(Path(path), private=False)
    try:
        value = json.loads(data)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"Invalid JSON object at {path}.") from error
    if not isinstance(value, dict):
        raise TypeError(f"Expected a JSON object at {path}.")
    return value


def _require_new_key_paths(private_path: Path, public_path: Path) -> None:
    if private_path == public_path or private_path.exists() or public_path.exists():
        raise FileExistsError("Key generation never overwrites an existing path.")
    if private_path.parent != public_path.parent:
        raise ValueError("Private and public research keys must share one protected directory.")
    _make_protected_directory(private_path.parent)


def _write_new_private_key(path: Path, data: bytes) -> None:
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o400)
    try:
        with os.fdopen(descriptor, "wb", closefd=False) as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
    finally:
        os.close(descriptor)
    _fsync_directory(path.parent)


def _write_new_public_key(path: Path, data: bytes) -> None:
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
    try:
        with os.fdopen(descriptor, "wb", closefd=False) as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
    finally:
        os.close(descriptor)
    _fsync_directory(path.parent)


def _read_regular_file(path: Path, *, private: bool) -> bytes:
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags)
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode) or before.st_nlink != 1:
            raise ValueError(f"Refusing a non-regular or hard-linked file: {path}.")
        if private and stat.S_IMODE(before.st_mode) & 0o077:
            raise PermissionError(
                f"Private-key file is accessible by group or others: {path}."
            )
        chunks: list[bytes] = []
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
        after = os.fstat(descriptor)
        if (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns) != (
            after.st_dev,
            after.st_ino,
            after.st_size,
            after.st_mtime_ns,
        ):
            raise RuntimeError(f"Key file changed while it was being read: {path}.")
        return b"".join(chunks)
    finally:
        os.close(descriptor)


def _make_protected_directory(path: Path) -> None:
    absolute = path.absolute()
    component = Path(absolute.anchor)
    for name in absolute.parts[1:]:
        component /= name
        try:
            observed = component.lstat()
        except FileNotFoundError:
            os.mkdir(component, mode=0o700)
            observed = component.lstat()
        if stat.S_ISLNK(observed.st_mode) or not stat.S_ISDIR(observed.st_mode):
            raise ValueError(f"Key directory path contains a non-directory link: {component}.")
    descriptor = os.open(
        absolute,
        os.O_RDONLY | os.O_DIRECTORY | getattr(os, "O_NOFOLLOW", 0),
    )
    try:
        os.fchmod(descriptor, 0o700)
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(
        path,
        os.O_RDONLY | os.O_DIRECTORY | getattr(os, "O_NOFOLLOW", 0),
    )
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _load_ed25519_private_key(
    path: str | Path,
    *,
    password: bytes | None = None,
) -> ed25519.Ed25519PrivateKey:
    key = serialization.load_pem_private_key(
        _read_regular_file(Path(path), private=True), password=password
    )
    if not isinstance(key, ed25519.Ed25519PrivateKey):
        raise TypeError("Authority private key is not Ed25519.")
    return key


def _load_ed25519_public_key(path: str | Path) -> ed25519.Ed25519PublicKey:
    key = serialization.load_pem_public_key(_read_regular_file(Path(path), private=False))
    if not isinstance(key, ed25519.Ed25519PublicKey):
        raise TypeError("Authority public key is not Ed25519.")
    return key


def _load_rsa_private_key(
    path: str | Path,
    *,
    password: bytes | None = None,
) -> rsa.RSAPrivateKey:
    key = serialization.load_pem_private_key(
        _read_regular_file(Path(path), private=True), password=password
    )
    if not isinstance(key, rsa.RSAPrivateKey):
        raise TypeError("Finalizer private key is not RSA.")
    if key.key_size < 3072:
        raise ValueError("Finalizer RSA key must contain at least 3072 bits.")
    return key


def _load_rsa_public_key(path: str | Path) -> rsa.RSAPublicKey:
    key = serialization.load_pem_public_key(_read_regular_file(Path(path), private=False))
    if not isinstance(key, rsa.RSAPublicKey):
        raise TypeError("Finalizer public key is not RSA.")
    if key.key_size < 3072:
        raise ValueError("Finalizer RSA key must contain at least 3072 bits.")
    return key
