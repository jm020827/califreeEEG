from __future__ import annotations

import json

import pandas as pd
import pytest

from cfeg.data.datasets import _validate_manifest_class_map


def test_manifest_class_map_rejects_stale_label_alignment(tmp_path):
    class_map = {"0": {"label": 0, "stimulus_frequency_hz": 8.0}}
    (tmp_path / "class_map.json").write_text(json.dumps(class_map), encoding="utf-8")
    manifest = pd.DataFrame({"label": [0, 0], "stimulus_frequency_hz": [8.6, 8.6]})

    with pytest.raises(ValueError, match="canonical frequency alignment"):
        _validate_manifest_class_map(tmp_path, manifest)


def test_manifest_class_map_accepts_canonical_alignment(tmp_path):
    class_map = {"0": {"label": 0, "stimulus_frequency_hz": 8.0}}
    (tmp_path / "class_map.json").write_text(json.dumps(class_map), encoding="utf-8")
    manifest = pd.DataFrame({"label": [0, 0], "stimulus_frequency_hz": [8.0, 8.0]})

    _validate_manifest_class_map(tmp_path, manifest)
