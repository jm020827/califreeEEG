from __future__ import annotations

import copy
import hashlib
import io
import os
import tarfile
from pathlib import Path

import h5py
import numpy as np
import pytest
import yaml

from cfeg.data.prepare_choi2019 import (
    PREPARED_FILENAMES,
    _member_manifest_sha256,
    _prepare_directory_atomically,
    _validate_extracted_inventory,
    audit_archive,
    normalize_questionnaire,
    read_cnt,
    read_markers,
    verify_raw_files,
)


def _config() -> dict:
    path = Path(__file__).resolve().parents[1] / "configs" / "data" / "choi2019.yaml"
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def _write_cell_strings(handle: h5py.File, path: str, values: list[str]) -> None:
    references = handle.require_group("#refs#")
    dataset = handle.create_dataset(path, shape=(len(values), 1), dtype=h5py.ref_dtype)
    for index, value in enumerate(values):
        chars = np.asarray([ord(char) for char in value], dtype=np.uint16).reshape(-1, 1)
        char_dataset = references.create_dataset(f"string_{len(references)}", data=chars)
        dataset[index, 0] = char_dataset.ref


def _write_cnt(path: Path, cfg: dict, n_samples: int = 10000) -> None:
    with h5py.File(path, "w") as handle:
        handle.create_dataset("cnt/x", data=np.zeros((39, n_samples), dtype=np.float64))
        handle.create_dataset("cnt/fs", data=np.asarray([[200.0]]))
        handle.create_dataset("cnt/T", data=np.asarray([[n_samples / 200.0 * 1000.0]]))
        _write_cell_strings(handle, "cnt/clab", cfg["raw_channel_names"])


def _write_markers(path: Path, cfg: dict, *, wrong_class_count: bool = False) -> None:
    labels = np.tile(np.arange(4), 10)
    if wrong_class_count:
        labels[0] = 1
    y = np.zeros((80, 5), dtype=np.float64)
    times = np.zeros(80, dtype=np.float64)
    for index, label in enumerate(labels):
        times[2 * index] = 101 + index * 220
        times[2 * index + 1] = 201 + index * 220
        y[2 * index, label] = 1
        y[2 * index + 1, 4] = 1
    with h5py.File(path, "w") as handle:
        handle.create_dataset("mrk/y", data=y)
        handle.create_dataset("mrk/time", data=times.reshape(-1, 1))
        _write_cell_strings(handle, "mrk/className", cfg["marker_class_names"])


def _fixture_member_records(cfg: dict, payload: bytes = b"x") -> list[tuple[str, int, str]]:
    digest = hashlib.sha256(payload).hexdigest()
    return [
        (
            f"S{subject}/Day{day}/{kind}_{band}({session}).mat",
            len(payload),
            digest,
        )
        for subject in range(1, 31)
        for day in (1, 2)
        for kind in ("cnt", "mrk")
        for band in ("LOW", "MID", "HIGH")
        for session in (1, 2)
    ]


def _write_exact_extracted_fixture(root: Path, cfg: dict) -> dict:
    records = _fixture_member_records(cfg)
    for relative, _, _ in records:
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"x")
    cfg["archive"]["uncompressed_regular_file_bytes"] = len(records)
    cfg["archive"]["member_content_manifest_sha256"] = _member_manifest_sha256(records)
    return cfg


def test_cnt_and_marker_reader_freeze_v73_orientation_and_one_based_times(tmp_path: Path) -> None:
    cfg = _config()
    cnt_path = tmp_path / "cnt_LOW(1).mat"
    mrk_path = tmp_path / "mrk_LOW(1).mat"
    _write_cnt(cnt_path, cfg)
    _write_markers(mrk_path, cfg)

    eeg, sfreq, raw_channels = read_cnt(cnt_path, cfg)
    markers = read_markers(mrk_path, cfg, n_continuous_samples=eeg.shape[-1])

    assert eeg.shape == (33, 10000)
    assert sfreq == 200.0
    assert raw_channels == 39
    assert markers.sample_indices_1based[0] == 101
    assert markers.source_event_indices.tolist() == list(range(0, 80, 2))
    np.testing.assert_array_equal(np.bincount(markers.within_band_labels), [10, 10, 10, 10])


def test_marker_reader_rejects_class_count_drift(tmp_path: Path) -> None:
    cfg = _config()
    path = tmp_path / "mrk_LOW(1).mat"
    _write_markers(path, cfg, wrong_class_count=True)

    with pytest.raises(ValueError, match="class counts drift"):
        read_markers(path, cfg, n_continuous_samples=10000)


def test_questionnaire_is_normalized_but_not_eligible_for_trial_join(tmp_path: Path) -> None:
    source = tmp_path / "questionnaires_answers.csv"
    lines = [",,,Survey_Before", "NO.,Subject,Gender,Sleep,Body,Drowsy,Focus,Eye"]
    row_number = 1
    for subject in range(1, 31):
        gender = "W" if subject in {2, 3, 6, 9, 11, 13, 18, 26, 29} else "M"
        lines.append(f"{row_number},S{subject}_Before,{gender},7,5,5,5,5")
        row_number += 1
        after_gender = "W" if subject == 7 else gender
        lines.append(f"{row_number},S{subject}_After,{after_gender},7,5,5,5,5")
        row_number += 1
    source.write_text("\n".join(lines) + "\n", encoding="utf-8")
    destination = tmp_path / "normalized.csv"

    receipt = normalize_questionnaire(source, destination)

    assert receipt["rows"] == 60
    assert receipt["joined_to_trial_manifest"] is False
    assert receipt["published_gender_mismatch_subjects"] == ["sub007"]
    assert "eligible_as_pre_query_model_input" in destination.read_text(encoding="utf-8")


def test_archive_audit_requires_exact_720_member_grid(tmp_path: Path) -> None:
    cfg = copy.deepcopy(_config())
    cfg["archive"]["uncompressed_regular_file_bytes"] = 720
    cfg["archive"]["member_content_manifest_sha256"] = _member_manifest_sha256(
        _fixture_member_records(cfg)
    )
    archive_path = tmp_path / "fixture.tar.gz"
    with tarfile.open(archive_path, "w:gz") as archive:
        for subject in range(1, 31):
            for directory in (f"S{subject}/", f"S{subject}/Day1/", f"S{subject}/Day2/"):
                info = tarfile.TarInfo(directory)
                info.type = tarfile.DIRTYPE
                archive.addfile(info)
            for day in (1, 2):
                for kind in ("cnt", "mrk"):
                    for band in ("LOW", "MID", "HIGH"):
                        for session in (1, 2):
                            info = tarfile.TarInfo(
                                f"S{subject}/Day{day}/{kind}_{band}({session}).mat"
                            )
                            info.size = 1
                            archive.addfile(info, io.BytesIO(b"x"))

    assert audit_archive(archive_path, cfg) == {
        "regular_files": 720,
        "directories": 90,
        "uncompressed_regular_file_bytes": 720,
        "member_content_manifest_sha256": cfg["archive"]["member_content_manifest_sha256"],
    }


def test_prepared_output_contract_includes_partial_and_companion_files() -> None:
    assert {
        "signals.h5",
        "signals.h5.tmp",
        "manifest.jsonl",
        "manifest.parquet",
        "class_map.json",
        "preprocess_config.yaml",
        "questionnaire_normalized.csv",
        "asset_info.json",
    } <= set(PREPARED_FILENAMES)


def test_reused_extracted_tree_hashes_every_member_and_rejects_tampering(tmp_path: Path) -> None:
    cfg = _write_exact_extracted_fixture(tmp_path / "extracted", copy.deepcopy(_config()))
    receipt = _validate_extracted_inventory(tmp_path / "extracted", cfg)
    assert receipt["regular_files"] == 720

    (tmp_path / "extracted/S1/Day1/cnt_LOW(1).mat").write_bytes(b"y")
    with pytest.raises(ValueError, match="member-content manifest mismatch"):
        _validate_extracted_inventory(tmp_path / "extracted", cfg)


def test_reused_extracted_tree_rejects_symlink_extra_and_nonregular_entries(
    tmp_path: Path,
) -> None:
    for case in ("symlink", "extra", "fifo"):
        root = tmp_path / case / "extracted"
        cfg = _write_exact_extracted_fixture(root, copy.deepcopy(_config()))
        if case == "symlink":
            target = root / "S1/Day1/cnt_LOW(1).mat"
            target.unlink()
            target.symlink_to(root / "S1/Day1/mrk_LOW(1).mat")
            match = "Symlink is forbidden"
        elif case == "extra":
            (root / "unexpected.txt").write_text("unexpected", encoding="utf-8")
            match = "Unexpected file"
        else:
            os.mkfifo(root / "S1/Day1/unexpected-pipe")
            match = "Non-regular entry"
        with pytest.raises(ValueError, match=match):
            _validate_extracted_inventory(root, cfg)


def test_all_three_raw_files_are_verified_before_use(tmp_path: Path) -> None:
    cfg = copy.deepcopy(_config())
    for index, spec in enumerate(cfg["files"]):
        payload = f"raw-{index}".encode()
        (tmp_path / spec["name"]).write_bytes(payload)
        spec["object_size_bytes"] = len(payload)
        spec["md5"] = hashlib.md5(payload).hexdigest()
        spec["sha256"] = hashlib.sha256(payload).hexdigest()
    assert set(verify_raw_files(tmp_path, cfg)) == {
        "mrk-and-cnt_datasets.tar.gz",
        "readme_100660.txt",
        "questionnaires_answers.csv",
    }

    (tmp_path / "questionnaires_answers.csv").write_bytes(b"tampered")
    with pytest.raises(ValueError, match="questionnaires_answers.csv"):
        verify_raw_files(tmp_path, cfg)


def test_preparation_publishes_atomically_and_preserves_failed_staging(tmp_path: Path) -> None:
    final = tmp_path / "choi2019_v1"

    def fail(staging: Path) -> dict:
        (staging / "partial.txt").write_text("preserve", encoding="utf-8")
        raise RuntimeError("injected failure")

    with pytest.raises(RuntimeError, match="injected failure"):
        _prepare_directory_atomically(final, fail)
    assert not final.exists()
    failed = list(tmp_path.glob(".choi2019_v1.staging-*"))
    assert len(failed) == 1
    assert (failed[0] / "partial.txt").read_text(encoding="utf-8") == "preserve"

    second_final = tmp_path / "choi2019_v1_complete"

    def succeed(staging: Path) -> dict:
        for name in PREPARED_FILENAMES:
            if not name.endswith(".tmp"):
                (staging / name).write_text(name, encoding="utf-8")
        return {"status": "ok"}

    assert _prepare_directory_atomically(second_final, succeed) == {"status": "ok"}
    assert second_final.is_dir()
    assert not list(tmp_path.glob(".choi2019_v1_complete.staging-*"))


@pytest.mark.parametrize("unsafe_kind", ["extra", "symlink", "directory"])
def test_preparation_rejects_unsafe_or_extra_staging_entries(
    tmp_path: Path, unsafe_kind: str
) -> None:
    final = tmp_path / f"choi2019_{unsafe_kind}"

    def build(staging: Path) -> dict:
        for name in PREPARED_FILENAMES:
            if not name.endswith(".tmp"):
                (staging / name).write_text(name, encoding="utf-8")
        unsafe = staging / "unexpected"
        if unsafe_kind == "extra":
            unsafe.write_text("extra", encoding="utf-8")
        elif unsafe_kind == "symlink":
            (staging / "signals.h5").unlink()
            (staging / "signals.h5").symlink_to(staging / "manifest.jsonl")
        else:
            unsafe.mkdir()
        return {"status": "must-not-publish"}

    match = "unsafe" if unsafe_kind == "symlink" else "unexpected"
    with pytest.raises(RuntimeError, match=match):
        _prepare_directory_atomically(final, build)
    assert not final.exists()
    assert len(list(tmp_path.glob(f".{final.name}.staging-*"))) == 1
