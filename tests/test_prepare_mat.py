from __future__ import annotations

from pathlib import Path

from cfeg.data.prepare_mat import _dedupe_files


def test_dedupe_files_skips_symlink_duplicate(tmp_path: Path):
    raw = tmp_path / "raw"
    raw.mkdir()
    original = raw / "S1.mat"
    original.write_bytes(b"placeholder")
    link_dir = raw / "moabb_links"
    link_dir.mkdir()
    link = link_dir / "S1.mat"
    link.symlink_to(original)

    unique, duplicates = _dedupe_files(sorted([original, link]))

    assert unique == [original]
    assert duplicates == [link]
