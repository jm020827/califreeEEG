from __future__ import annotations

import pandas as pd

from cfeg.constants import LEAKAGE_FIELDS
from cfeg.data.schema import REQUIRED_MANIFEST_COLUMNS, write_manifest


def test_manifest_columns_include_split_and_eval_metadata():
    assert "subject_id" in REQUIRED_MANIFEST_COLUMNS
    assert "stimulus_frequency_hz" in REQUIRED_MANIFEST_COLUMNS


def test_leakage_fields_declared():
    assert {"label", "stimulus_frequency_hz", "subject_id", "trial_id"} <= LEAKAGE_FIELDS


def test_manifest_writer_removes_stale_parquet_when_parquet_write_fails(tmp_path, monkeypatch):
    stale = tmp_path / "manifest.parquet"
    stale.write_text("stale", encoding="utf-8")

    def fail_parquet(*args, **kwargs):
        raise RuntimeError("no parquet engine")

    monkeypatch.setattr(pd.DataFrame, "to_parquet", fail_parquet)
    write_manifest(pd.DataFrame({"sample_id": ["new"]}), tmp_path)

    assert (tmp_path / "manifest.jsonl").exists()
    assert not stale.exists()
