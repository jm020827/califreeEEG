from __future__ import annotations

import hashlib
from collections.abc import Iterable


def canonical_identity_sha256(values: Iterable[str]) -> str:
    """Hash sorted identities with unambiguous length prefixes."""

    digest = hashlib.sha256()
    for value in sorted(str(item) for item in values):
        encoded = value.encode("utf-8")
        digest.update(len(encoded).to_bytes(8, byteorder="big", signed=False))
        digest.update(encoded)
    return digest.hexdigest()
