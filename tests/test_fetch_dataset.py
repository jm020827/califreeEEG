from __future__ import annotations

import hashlib
import importlib.util
import sys
from pathlib import Path

import pytest


def _load_fetch_module():
    scripts_dir = Path(__file__).resolve().parents[1] / "scripts"
    sys.path.insert(0, str(scripts_dir))
    spec = importlib.util.spec_from_file_location(
        "cfeg_fetch_dataset", scripts_dir / "fetch_dataset.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


fetch_dataset = _load_fetch_module()


def _wang_record(n_subjects: int = 35) -> dict:
    files = [
        {"key": f"S{subject}.mat", "size": subject, "links": {"self": "https://example"}}
        for subject in range(1, n_subjects + 1)
    ]
    files.extend(
        {"key": name, "size": 1, "links": {"self": "https://example"}}
        for name in ("64-channels.loc", "Readme.txt", "Sub_info.txt")
    )
    return {"files": files}


def _beta_article(n_subjects: int = 70) -> dict:
    files = [
        {"name": f"S{subject}.mat", "size": subject, "download_url": "https://example"}
        for subject in range(1, n_subjects + 1)
    ]
    files.extend(
        {"name": name, "size": 1, "download_url": "https://example"}
        for name in ("Description.pdf", "note.txt")
    )
    return {"files": files}


def _wearable_article(n_subjects: int = 102) -> dict:
    files = [
        {"name": f"S{subject:03d}.mat", "size": subject, "download_url": "https://example"}
        for subject in range(1, n_subjects + 1)
    ]
    files.extend(
        {"name": name, "size": 1, "download_url": "https://example"}
        for name in (
            "Impedance.mat",
            "Readme.pdf",
            "Subjects_Information.mat",
            "stimulation_information.pdf",
        )
    )
    return {"files": files}


def _dong_record(n_subjects: int = 59) -> dict:
    files = [
        {"key": f"S{subject}.mat", "size": subject, "links": {"self": "https://example"}}
        for subject in range(1, n_subjects + 1)
    ]
    files.extend(
        {"key": name, "size": 1, "links": {"self": "https://example"}}
        for name in ("8-channels.mat", "59-subject.mat")
    )
    return {"files": files}


def test_select_beta_files_includes_full_70_subject_inventory_and_support() -> None:
    files = fetch_dataset._select_beta_files(_beta_article(), subjects=None)

    assert len(files) == 72
    assert [file["subject"] for file in files[:70]] == list(range(1, 71))
    assert {file["name"] for file in files[70:]} == {"Description.pdf", "note.txt"}


def test_select_beta_files_rejects_silent_full_cohort_drift() -> None:
    with pytest.raises(SystemExit, match=r"missing=\[70\]"):
        fetch_dataset._select_beta_files(_beta_article(n_subjects=69), subjects=None)


def test_select_beta_files_rejects_duplicate_subject() -> None:
    article = _beta_article()
    article["files"].append({"name": "S1.mat", "size": 1, "download_url": "https://duplicate"})

    with pytest.raises(SystemExit, match=r"duplicates=\[1\]"):
        fetch_dataset._select_beta_files(article, subjects=None)


def test_select_beta_files_rejects_missing_support_file() -> None:
    article = _beta_article()
    article["files"] = [file for file in article["files"] if file["name"] != "note.txt"]

    with pytest.raises(SystemExit, match=r"support-file inventory mismatch"):
        fetch_dataset._select_beta_files(article, subjects=None)


def test_select_wearable_files_requires_full_inventory_and_support() -> None:
    files = fetch_dataset._select_wearable_files(_wearable_article(), subjects=None)

    assert len(files) == 106
    assert [file["subject"] for file in files[:102]] == list(range(1, 103))


def test_select_wearable_files_rejects_silent_full_cohort_drift() -> None:
    with pytest.raises(SystemExit, match=r"missing=\[102\]"):
        fetch_dataset._select_wearable_files(_wearable_article(n_subjects=101), subjects=None)


def test_select_dong_files_requires_full_inventory_and_support() -> None:
    files = fetch_dataset._select_dong2023_files(_dong_record(), subjects=None)

    assert len(files) == 61
    assert [file["subject"] for file in files[:59]] == list(range(1, 60))


def test_select_dong_files_rejects_silent_full_cohort_drift() -> None:
    with pytest.raises(SystemExit, match=r"missing=\[59\]"):
        fetch_dataset._select_dong2023_files(_dong_record(n_subjects=58), subjects=None)


def test_select_wang_files_requires_and_returns_full_35_subject_inventory() -> None:
    files = fetch_dataset._select_wang_files(_wang_record(), subjects=None)

    assert len(files) == 38
    assert [file["subject"] for file in files[:35]] == list(range(1, 36))
    assert {file["key"] for file in files[35:]} == {
        "64-channels.loc",
        "Readme.txt",
        "Sub_info.txt",
    }


def test_select_wang_files_rejects_silent_full_cohort_drift() -> None:
    with pytest.raises(SystemExit, match=r"missing=\[35\]"):
        fetch_dataset._select_wang_files(_wang_record(n_subjects=34), subjects=None)


def test_select_wang_files_allows_explicit_subject_subset() -> None:
    files = fetch_dataset._select_wang_files(_wang_record(), subjects=[1, 35])

    assert [file["subject"] for file in files[:2]] == [1, 35]
    assert len(files) == 5


def test_force_update_preserves_verified_existing_file_when_download_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    existing = b"verified-existing"
    destination = tmp_path / "S1.mat"
    destination.write_bytes(existing)
    file = {
        "name": "S1.mat",
        "size": len(existing),
        "computed_md5": hashlib.md5(existing).hexdigest(),
        "download_url": "https://example.invalid/S1.mat",
    }

    def fail_download(_url: str):
        raise OSError("network failure")

    monkeypatch.setattr(fetch_dataset.urllib.request, "urlopen", fail_download)

    with pytest.raises(OSError, match="network failure"):
        fetch_dataset._download_figshare_file(file, destination, force_update=True)

    assert destination.read_bytes() == existing
