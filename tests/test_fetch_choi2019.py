from __future__ import annotations

import base64
import copy
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import pytest
import yaml


def _load_fetch_module():
    scripts_dir = Path(__file__).resolve().parents[1] / "scripts"
    sys.path.insert(0, str(scripts_dir))
    spec = importlib.util.spec_from_file_location(
        "cfeg_fetch_choi2019", scripts_dir / "fetch_choi2019.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


fetch = _load_fetch_module()


def _config() -> dict:
    path = Path(__file__).resolve().parents[1] / "configs" / "data" / "choi2019.yaml"
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def _official_payloads(cfg: dict) -> tuple[dict, dict, dict]:
    dataset = {
        "data": {"data": {"dataset_id": "100660", "doi": cfg["official_source"]["dataset_doi"]}}
    }
    files = {
        "data": {
            "data": [
                {
                    "id": spec["gigadb_file_id"],
                    "file_name": spec["name"],
                    "file_size": spec["catalog_size_bytes"],
                    "url": spec["url"],
                }
                for spec in cfg["files"]
            ]
        }
    }
    datacite = {
        "data": {
            "attributes": {
                "doi": cfg["official_source"]["dataset_doi"],
                "rightsList": [{"rightsIdentifier": "cc0-1.0"}],
            }
        }
    }
    return dataset, files, datacite


def test_frozen_official_inventory_accepts_known_readme_size_drift() -> None:
    cfg = _config()
    dataset, files, datacite = _official_payloads(cfg)

    fetch.validate_official_metadata(
        cfg,
        gigadb_dataset=dataset,
        gigadb_files=files,
        datacite=datacite,
    )

    readme = next(item for item in cfg["files"] if item["name"] == "readme_100660.txt")
    assert readme["catalog_size_bytes"] == 3187
    assert readme["object_size_bytes"] == 3267


def test_frozen_official_inventory_rejects_added_file() -> None:
    cfg = _config()
    dataset, files, datacite = _official_payloads(cfg)
    files["data"]["data"].append(
        {"id": 1, "file_name": "unexpected.mat", "file_size": 1, "url": "https://example"}
    )

    with pytest.raises(ValueError, match="file inventory drift"):
        fetch.validate_official_metadata(
            cfg,
            gigadb_dataset=dataset,
            gigadb_files=files,
            datacite=datacite,
        )


def test_metadata_md5_decodes_archive_header() -> None:
    expected = "0c7156a80ede01ee1a62e1f489f2677f"
    encoded = base64.b64encode(bytes.fromhex(expected)).decode("ascii")

    assert fetch._metadata_md5({"x-amz-meta-md5chksum": encoded}) == expected


def test_verify_local_file_requires_both_md5_and_sha256(tmp_path: Path) -> None:
    payload = b"frozen-object"
    path = tmp_path / "asset.bin"
    path.write_bytes(payload)
    spec = {
        "object_size_bytes": len(payload),
        "md5": hashlib.md5(payload).hexdigest(),
        "sha256": hashlib.sha256(payload).hexdigest(),
    }

    assert fetch.verify_local_file(spec, path)["bytes"] == len(payload)

    changed = copy.deepcopy(spec)
    changed["sha256"] = "0" * 64
    with pytest.raises(ValueError, match="Checksum mismatch"):
        fetch.verify_local_file(changed, path)


def test_bad_existing_file_is_preserved(tmp_path: Path) -> None:
    path = tmp_path / "asset.bin"
    path.write_bytes(b"do-not-overwrite")
    spec = {
        "name": path.name,
        "url": "https://example.invalid/asset.bin",
        "object_size_bytes": 1,
        "md5": "0" * 32,
        "sha256": "0" * 64,
    }

    with pytest.raises(ValueError, match="Checksum mismatch"):
        fetch.download_verified_file(spec, path)

    assert path.read_bytes() == b"do-not-overwrite"


def test_download_receipt_is_idempotent_across_transport_status(tmp_path: Path) -> None:
    path = tmp_path / "download_receipt.json"
    base = {
        "schema": "cfeg.choi2019-download-receipt.v1",
        "dataset_revision": "frozen",
        "files": [{"name": "asset.bin", "bytes": 3, "sha256": "abc"}],
    }
    first = copy.deepcopy(base)
    first["files"][0]["status"] = "downloaded"
    rerun = copy.deepcopy(base)
    rerun["files"][0]["status"] = "verified-existing"

    fetch._write_json_once(path, first)
    fetch._write_json_once(path, rerun)

    assert json.loads(path.read_text(encoding="utf-8")) == first

    changed = copy.deepcopy(rerun)
    changed["files"][0]["bytes"] = 4
    with pytest.raises(RuntimeError, match="Refusing to overwrite"):
        fetch._write_json_once(path, changed)


def test_partial_fetch_can_never_claim_the_canonical_full_receipt(tmp_path: Path) -> None:
    cfg = _config()
    selected = fetch._selected_files(cfg, "questionnaires_answers.csv")
    partial = fetch._download_receipt_path(tmp_path, cfg, selected)

    assert partial.parent == tmp_path
    assert partial.name.startswith("download_receipt.partial-")
    assert partial.name.endswith(".json")
    assert partial != tmp_path / "download_receipt.json"
    assert fetch._download_receipt_path(tmp_path, cfg, list(cfg["files"])) == (
        tmp_path / "download_receipt.json"
    )
    reordered = list(reversed(cfg["files"]))
    assert fetch._download_receipt_path(tmp_path, cfg, reordered) != (
        tmp_path / "download_receipt.json"
    )


def test_empty_explicit_file_selection_is_rejected() -> None:
    with pytest.raises(ValueError, match="selects no files"):
        fetch._selected_files(_config(), ", ,")
