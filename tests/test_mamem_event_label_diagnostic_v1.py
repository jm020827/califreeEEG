"""Generated diagnostic tests only; no actual MAT or fitting."""

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from cfeg.mamem_events_v1 import FREQUENCIES

SPEC = importlib.util.spec_from_file_location(
    "diagnostic",
    Path(__file__).resolve().parents[1] / "scripts/analysis/diagnose_mamem_event_labels_v1.py",
)
diagnostic = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(diagnostic)


class Poison:
    def __float__(self):
        raise AssertionError("descriptor touched")


def fixture():
    groups = []
    for i, j in enumerate([0, 1, 2, 3, 4, 0, 1, 2] + [j for j in range(5) for _ in range(3)]):
        rel = np.arange(int(5000 * 2 * FREQUENCIES[j] / 1000) + 1) * 1000 / (2 * FREQUENCIES[j])
        group = np.empty((4, len(rel)), object)
        group[0] = Poison()
        group[2] = Poison()
        group[1] = 1000 + i * 10000 + rel
        group[3] = 101 + i * 2500 + np.rint(rel / 4).astype(int)
        groups.append(group)
    return groups


def test_all_groups_uniform_and_no_label_assignment():
    result = diagnostic.diagnose(np.concatenate(fixture(), axis=1), 60000)
    assert result["group_count"] == 23 and len(result["groups"]) == 23
    assert result["guardband_failure_groups"] == []
    assert result["assigned_labels"] is False
    assert result["EEG_values_loaded"] is False and result["M_features_computed"] is False
    assert all(r["fixed_window_contained"] for r in result["groups"])
    assert [r["role"] for r in result["groups"]] == ["adaptation"] * 8 + ["main"] * 15


@pytest.mark.parametrize("bad_group", [0, 7, 8, 22])
def test_outside_guardband_is_observed_not_repaired_or_dropped(bad_group):
    groups = fixture()
    groups[bad_group][1] = (
        1000 + bad_group * 10000 + np.arange(groups[bad_group].shape[1]) * 1000 / (2 * 7.05)
    )
    result = diagnostic.diagnose(np.concatenate(groups, axis=1), 60000)
    assert result["guardband_failure_groups"] == [bad_group]
    assert result["first_guardband_failure_group"] == bad_group
    assert result["groups"][bad_group]["old_guardband_matching_nominal_hz"] == []
    assert len(result["groups"]) == 23


def test_invalid_numeric_rows_fail():
    groups = fixture()
    groups[0][3, 2] = groups[0][3, 1]
    with pytest.raises(ValueError, match="sample_order"):
        diagnostic.diagnose(np.concatenate(groups, axis=1), 60000)


def test_exclusive_bounded_output(tmp_path):
    output = tmp_path / "saved.json"
    diagnostic.save(output, {"ok": True})
    with pytest.raises(FileExistsError):
        diagnostic.save(output, {"ok": False})
    with pytest.raises(ValueError, match="output_cap"):
        diagnostic.save(tmp_path / "large.json", {"x": "x" * 65536})
    assert json.loads(output.read_text()) == {"ok": True}


def test_inspect_requests_only_din_after_header_checks(tmp_path, monkeypatch):
    class FakeMat:
        def is_file(self):
            return True

        def is_symlink(self):
            return False

        def stat(self):
            return SimpleNamespace(st_size=137357437)

        def __str__(self):
            return "/generated/no-actual-mat"

    mat = FakeMat()
    role = tmp_path / "role.json"
    role.write_text(
        json.dumps(
            {
                "subject": "S001",
                "subject_role": "development_only_all_records",
                "mat_path": str(mat),
                "mat_sha256": diagnostic.MAT_SHA,
            }
        )
    )
    monkeypatch.setattr(diagnostic, "MAT", mat)
    monkeypatch.setattr(diagnostic, "ROLE", role)
    monkeypatch.setattr(diagnostic, "manifest", dict)
    monkeypatch.setattr(
        diagnostic, "sha", lambda path: diagnostic.MAT_SHA if path is mat else diagnostic.ROLE_SHA
    )
    monkeypatch.setattr(diagnostic, "diagnose", lambda din, total: {"diagnostic_fake": True})
    calls = []

    def load(path, **kwargs):
        calls.append(kwargs)
        return {"DIN_1": np.empty((4, 1966), object)}

    headers = lambda path: [("eeg", (257, 117917), "double"), ("DIN_1", (4, 1966), "cell")]
    assert diagnostic.inspect(load, headers)["diagnostic_fake"] is True
    assert calls == [
        {
            "variable_names": ["DIN_1"],
            "squeeze_me": False,
            "struct_as_record": True,
            "verify_compressed_data_integrity": True,
        }
    ]
