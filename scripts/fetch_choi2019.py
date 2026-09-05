#!/usr/bin/env python
"""Fetch the frozen Choi2019 GigaDB asset without relying on mutable mirrors."""

from __future__ import annotations

import argparse
import base64
import binascii
import hashlib
import json
import os
import urllib.request
from collections.abc import Callable
from pathlib import Path
from typing import Any, BinaryIO

import yaml

DEFAULT_CONFIG = "configs/data/choi2019.yaml"
DEFAULT_RAW_DIR = "/home/whwovy/eeg-data/raw/choi2019_gigadb"
CHUNK_BYTES = 8 * 1024 * 1024


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--raw-dir", default=DEFAULT_RAW_DIR)
    parser.add_argument(
        "--only",
        default=None,
        help="Optional comma-separated frozen file names. Remote inventory is still fully checked.",
    )
    parser.add_argument(
        "--probe-remote",
        action="store_true",
        help="Validate official APIs and object metadata without writing or downloading.",
    )
    args = parser.parse_args()

    cfg = _load_config(Path(args.config))
    selected = _selected_files(cfg, args.only)
    snapshots = fetch_and_validate_official_metadata(cfg)
    print("Validated GigaDB dataset 100660 exact three-file inventory and DataCite CC0-1.0.")
    for spec in selected:
        headers = head_and_validate_object(spec)
        print(
            f"HEAD {spec['name']}: {headers['content_length']} bytes, "
            f"etag={headers.get('etag', 'missing')}"
        )
    if args.probe_remote:
        return

    raw_dir = Path(args.raw_dir).expanduser().resolve()
    raw_dir.mkdir(parents=True, exist_ok=True)
    _write_json_once(raw_dir / "gigadb_dataset_api.json", snapshots["gigadb_dataset"])
    _write_json_once(raw_dir / "gigadb_files_api.json", snapshots["gigadb_files"])
    _write_json_once(raw_dir / "datacite_api.json", snapshots["datacite"])

    receipts = []
    for spec in selected:
        destination = raw_dir / str(spec["name"])
        status = download_verified_file(spec, destination)
        receipts.append(status)
        print(
            f"{status['status']} {destination}: {status['bytes']} bytes, sha256={status['sha256']}"
        )
    receipt = {
        "schema": "cfeg.choi2019-download-receipt.v1",
        "dataset_revision": cfg["dataset_revision"],
        "dataset_doi": cfg["official_source"]["dataset_doi"],
        "license_spdx": cfg["official_source"]["license_spdx"],
        "files": receipts,
    }
    _write_json_once(raw_dir / "download_receipt.json", receipt)


def _load_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle)
    if cfg.get("schema") != "cfeg.dataset.choi2019.v1":
        raise ValueError(f"Unexpected Choi config schema in {path}: {cfg.get('schema')!r}")
    return cfg


def _selected_files(cfg: dict[str, Any], raw: str | None) -> list[dict[str, Any]]:
    files = list(cfg["files"])
    if not raw:
        return files
    names = {part.strip() for part in raw.split(",") if part.strip()}
    known = {str(spec["name"]) for spec in files}
    unknown = sorted(names - known)
    if unknown:
        raise ValueError(
            f"Unknown --only Choi file(s): {unknown}; frozen names are {sorted(known)}"
        )
    return [spec for spec in files if spec["name"] in names]


def _fetch_json(
    url: str,
    *,
    opener: Callable[..., BinaryIO] = urllib.request.urlopen,
) -> dict[str, Any]:
    request = urllib.request.Request(url, headers={"User-Agent": "califreeEEG-data-audit/1"})
    with opener(request, timeout=60) as response:
        return json.loads(response.read().decode("utf-8"))


def fetch_and_validate_official_metadata(
    cfg: dict[str, Any],
    *,
    fetch_json: Callable[[str], dict[str, Any]] = _fetch_json,
) -> dict[str, dict[str, Any]]:
    source = cfg["official_source"]
    snapshots = {
        "gigadb_dataset": fetch_json(str(source["gigadb_dataset_api"])),
        "gigadb_files": fetch_json(str(source["gigadb_files_api"])),
        "datacite": fetch_json(str(source["datacite_api"])),
    }
    validate_official_metadata(cfg, **snapshots)
    return snapshots


def _gigadb_data(payload: dict[str, Any]) -> Any:
    try:
        return payload["data"]["data"]
    except (KeyError, TypeError) as exc:
        raise ValueError("Unexpected GigaDB API envelope; expected data.data") from exc


def validate_official_metadata(
    cfg: dict[str, Any],
    *,
    gigadb_dataset: dict[str, Any],
    gigadb_files: dict[str, Any],
    datacite: dict[str, Any],
) -> None:
    source = cfg["official_source"]
    dataset = _gigadb_data(gigadb_dataset)
    if str(dataset.get("dataset_id")) != "100660":
        raise ValueError(f"GigaDB dataset-id drift: {dataset.get('dataset_id')!r}")
    if str(dataset.get("doi", "")).lower() != str(source["dataset_doi"]).lower():
        raise ValueError(f"GigaDB DOI drift: {dataset.get('doi')!r}")

    remote_files = _gigadb_data(gigadb_files)
    if not isinstance(remote_files, list):
        raise TypeError("Unexpected GigaDB file-list payload")
    remote_by_name = {str(item.get("file_name")): item for item in remote_files}
    frozen_by_name = {str(item["name"]): item for item in cfg["files"]}
    if set(remote_by_name) != set(frozen_by_name):
        raise ValueError(
            "GigaDB file inventory drift: "
            f"expected={sorted(frozen_by_name)}, observed={sorted(remote_by_name)}"
        )
    if len(remote_by_name) != len(remote_files):
        raise ValueError("GigaDB file inventory contains duplicate names")
    for name, frozen in frozen_by_name.items():
        remote = remote_by_name[name]
        checks = {
            "file id": (int(remote.get("id")), int(frozen["gigadb_file_id"])),
            "catalog size": (int(remote.get("file_size")), int(frozen["catalog_size_bytes"])),
            "URL": (str(remote.get("url")), str(frozen["url"])),
        }
        for field, (observed, expected) in checks.items():
            if observed != expected:
                raise ValueError(
                    f"GigaDB {field} drift for {name}: expected={expected!r}, observed={observed!r}"
                )

    try:
        attributes = datacite["data"]["attributes"]
    except (KeyError, TypeError) as exc:
        raise ValueError("Unexpected DataCite API envelope") from exc
    if str(attributes.get("doi", "")).lower() != str(source["dataset_doi"]).lower():
        raise ValueError(f"DataCite DOI drift: {attributes.get('doi')!r}")
    rights = {
        str(item.get("rightsIdentifier", "")).lower() for item in attributes.get("rightsList", [])
    }
    if str(source["license_spdx"]).lower() not in rights:
        raise ValueError(
            f"DataCite license drift: expected {source['license_spdx']}, observed {sorted(rights)}"
        )


def _clean_etag(value: str | None) -> str | None:
    if value is None:
        return None
    return str(value).strip().strip('"')


def _metadata_md5(headers: dict[str, str]) -> str | None:
    encoded = headers.get("x-amz-meta-md5chksum")
    if not encoded:
        return None
    try:
        decoded = base64.b64decode(encoded, validate=True)
    except (ValueError, binascii.Error) as exc:
        raise ValueError(f"Malformed x-amz-meta-md5chksum: {encoded!r}") from exc
    if len(decoded) != 16:
        raise ValueError("x-amz-meta-md5chksum did not decode to a 16-byte MD5")
    return decoded.hex()


def head_and_validate_object(
    spec: dict[str, Any],
    *,
    opener: Callable[..., BinaryIO] = urllib.request.urlopen,
) -> dict[str, Any]:
    request = urllib.request.Request(
        str(spec["url"]), method="HEAD", headers={"User-Agent": "califreeEEG-data-audit/1"}
    )
    with opener(request, timeout=60) as response:
        headers = {str(key).lower(): str(value) for key, value in response.headers.items()}
        status = int(getattr(response, "status", response.getcode()))
    if status != 200:
        raise ValueError(f"HEAD {spec['name']} returned HTTP {status}")
    observed_size = int(headers.get("content-length", -1))
    expected_size = int(spec["object_size_bytes"])
    if observed_size != expected_size:
        raise ValueError(
            f"Object-size drift for {spec['name']}: expected={expected_size}, "
            f"observed={observed_size}"
        )
    etag = _clean_etag(headers.get("etag"))
    if etag != str(spec["etag"]):
        raise ValueError(
            f"ETag drift for {spec['name']}: expected={spec['etag']!r}, observed={etag!r}"
        )
    if bool(spec["etag_is_md5"]):
        observed_md5 = etag
    else:
        observed_md5 = _metadata_md5(headers)
    if observed_md5 != str(spec["md5"]):
        raise ValueError(
            f"Object metadata MD5 drift for {spec['name']}: expected={spec['md5']}, "
            f"observed={observed_md5}"
        )
    return {
        "status": status,
        "content_length": observed_size,
        "etag": etag,
        "metadata_md5": observed_md5,
    }


def file_hashes(path: Path) -> dict[str, Any]:
    md5 = hashlib.md5()
    sha256 = hashlib.sha256()
    size = 0
    with path.open("rb") as handle:
        while chunk := handle.read(CHUNK_BYTES):
            size += len(chunk)
            md5.update(chunk)
            sha256.update(chunk)
    return {"bytes": size, "md5": md5.hexdigest(), "sha256": sha256.hexdigest()}


def verify_local_file(spec: dict[str, Any], path: Path) -> dict[str, Any]:
    observed = file_hashes(path)
    expected = {
        "bytes": int(spec["object_size_bytes"]),
        "md5": str(spec["md5"]),
        "sha256": str(spec["sha256"]),
    }
    if observed != expected:
        raise ValueError(
            f"Checksum mismatch for {path}: expected={expected}, observed={observed}. "
            "The file was preserved for audit."
        )
    return observed


def _open_download(
    url: str,
    offset: int,
    *,
    opener: Callable[..., BinaryIO] = urllib.request.urlopen,
) -> BinaryIO:
    headers = {"User-Agent": "califreeEEG-data-audit/1"}
    if offset:
        headers["Range"] = f"bytes={offset}-"
    request = urllib.request.Request(url, headers=headers)
    return opener(request, timeout=120)


def download_verified_file(
    spec: dict[str, Any],
    destination: Path,
    *,
    open_download: Callable[[str, int], BinaryIO] = _open_download,
) -> dict[str, Any]:
    if destination.exists():
        hashes = verify_local_file(spec, destination)
        return {"name": destination.name, "status": "verified-existing", **hashes}

    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_name(f"{destination.name}.part")
    offset = partial.stat().st_size if partial.exists() else 0
    expected_size = int(spec["object_size_bytes"])
    if offset > expected_size:
        raise ValueError(
            f"Partial file is larger than frozen object ({offset}>{expected_size}): {partial}"
        )
    if offset == expected_size:
        hashes = verify_local_file(spec, partial)
        os.replace(partial, destination)
        return {"name": destination.name, "status": "promoted-verified-partial", **hashes}

    with open_download(str(spec["url"]), offset) as response:
        status = int(getattr(response, "status", response.getcode()))
        if offset and status != 206:
            raise RuntimeError(
                f"Server did not honor Range resume for {partial} (HTTP {status}); "
                "the partial file was preserved."
            )
        if not offset and status != 200:
            raise RuntimeError(f"Download for {destination.name} returned HTTP {status}")
        mode = "ab" if offset else "wb"
        with partial.open(mode) as handle:
            while chunk := response.read(CHUNK_BYTES):
                handle.write(chunk)

    hashes = verify_local_file(spec, partial)
    os.replace(partial, destination)
    status_text = "downloaded-resumed" if offset else "downloaded"
    return {"name": destination.name, "status": status_text, **hashes}


def _write_json_once(path: Path, payload: Any) -> None:
    if path.exists():
        with path.open("r", encoding="utf-8") as handle:
            existing = json.load(handle)
        if existing != payload and not _same_download_identity(existing, payload):
            raise RuntimeError(f"Refusing to overwrite changed provenance snapshot: {path}")
        return
    temporary = path.with_name(f"{path.name}.tmp")
    with temporary.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(temporary, path)


def _same_download_identity(existing: Any, proposed: Any) -> bool:
    """Ignore only the first-run transport status in an immutable receipt.

    A fresh download records ``downloaded`` while an idempotent rerun observes
    ``verified-existing``. Those executions identify the same bytes and must
    not force a provenance overwrite; every other receipt field remains exact.
    """

    if not isinstance(existing, dict) or not isinstance(proposed, dict):
        return False
    schema = "cfeg.choi2019-download-receipt.v1"
    if existing.get("schema") != schema or proposed.get("schema") != schema:
        return False

    def without_status(payload: dict[str, Any]) -> dict[str, Any]:
        normalized = dict(payload)
        normalized["files"] = [
            {key: value for key, value in dict(row).items() if key != "status"}
            for row in payload.get("files", [])
        ]
        return normalized

    return without_status(existing) == without_status(proposed)


if __name__ == "__main__":
    main()
