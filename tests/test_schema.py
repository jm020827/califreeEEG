from __future__ import annotations

from cfeg.constants import LEAKAGE_FIELDS
from cfeg.data.schema import REQUIRED_MANIFEST_COLUMNS


def test_manifest_columns_include_split_and_eval_metadata():
    assert "subject_id" in REQUIRED_MANIFEST_COLUMNS
    assert "stimulus_frequency_hz" in REQUIRED_MANIFEST_COLUMNS


def test_leakage_fields_declared():
    assert {"label", "stimulus_frequency_hz", "subject_id", "trial_id"} <= LEAKAGE_FIELDS

