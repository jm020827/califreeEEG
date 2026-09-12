"""Generated MAT/cell fixtures only; no human data or network."""
import importlib.util
from pathlib import Path

import numpy as np
import pytest
from scipy.io import savemat

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/analysis/probe_mamem_i_din_v1.py"
spec = importlib.util.spec_from_file_location("mamem_probe", SCRIPT)
p = importlib.util.module_from_spec(spec)
spec.loader.exec_module(p)


def cells(times, samples=None):
    x = np.empty((4, len(times)), dtype=object)
    x[:] = "unread"
    for i, t in enumerate(times):
        x[1, i] = np.array([[t]])
        x[3, i] = np.array([[samples[i] if samples is not None else 1 + t / 4]])
    return x


def test_boundary_does_not_read_boundary_sample_or_later_values():
    x = cells([0, 40, 84, 2200, np.nan])
    x[3, 3] = "forbidden boundary sample"
    r = p.summarize_prefix(x)
    assert r["events_in_first_group_prefix"] == 3 and r["group_complete"]
    assert r["interval_summary"]["max_ms"] == 44
    assert r["interval_summary"]["timestamp_minus_4sample_max_abs_ms"] == 0
    assert not r["interpretation"]["physical_jitter_verified"]


def test_200_cap_does_not_read_201():
    x = cells([i * 40 for i in range(201)])
    x[1, 200] = "forbidden"
    r = p.summarize_prefix(x)
    assert r["events_in_first_group_prefix"] == 200
    assert r["censored_without_boundary"]


def test_exact_2000_is_not_boundary():
    assert not p.summarize_prefix(cells([0, 2000]))["boundary_observed"]


@pytest.mark.parametrize("times", [[], [0]])
def test_insufficient_does_not_replace_group(times):
    r = p.summarize_prefix(cells(times))
    assert r["status"] == "INSUFFICIENT_FIRST_PREFIX"
    assert r["interval_summary"] is None


@pytest.mark.parametrize("times,samples", [([0, np.nan], [1, 11]),
                                          ([40, 0], [11, 1]),
                                          ([0, 40], [1, 1]),
                                          ([0, 40], [1, 10.5])])
def test_invalid_selected_values_stop(times, samples):
    with pytest.raises(p.Stop):
        p.summarize_prefix(cells(times, samples))


def test_clock_mismatch_not_auto_rescaled():
    r = p.summarize_prefix(cells([0, 0.04], [1, 11]))
    assert r["status"] == "CLOCK_MAPPING_UNRESOLVED"


def test_wrong_shape_no_transpose():
    with pytest.raises(p.Stop):
        p.summarize_prefix(np.zeros((3, 10)))


def test_allocation_cap(monkeypatch):
    monkeypatch.setattr(p, "CAP", 8)
    with pytest.raises(p.Stop, match="memory_cap"):
        p.decoded_bytes(cells([0]))


def test_only_DIN_requested_and_schema_reads_no_arrays(tmp_path):
    path = tmp_path / "S001a.mat"
    savemat(path, {"DIN_1": cells([0, 40, 80]), "eeg": np.ones((3, 21))}, do_compression=True)
    role = {"mat_path": str(path.resolve()), "mat_sha256": p.hash_file(path), "subject": "S001",
            "subject_role": "development_only_all_records",
            "member_selection": "lexicographic_first_MAT_across_both_archives"}
    calls = []
    from scipy.io import loadmat

    def load_din(path, **kwargs):
        calls.append(kwargs)
        assert kwargs["variable_names"] == ["DIN_1"]
        return loadmat(path, **kwargs)

    r = p.inspect(path, "schema", role, loader=load_din)
    assert r["status"] == "HEADER_ONLY" and calls == []
    r = p.inspect(path, "din", role, loader=load_din)
    assert len(calls) == 1 and r["status"] == "DIN_PREFIX_SCHEMA_OBSERVED"
    assert r["DIN_variable_fully_decoded"]


def test_role_rejected_before_header_load(tmp_path):
    path = tmp_path / "generated.mat"
    path.write_bytes(b"generated")
    calls = []
    with pytest.raises(p.Stop, match="path_mismatch"):
        p.inspect(path, "din", {"mat_path": "elsewhere"}, header_loader=lambda _: calls.append(1))
    assert not calls


def test_selected_sample_beyond_EEG_header_stops():
    with pytest.raises(p.Stop, match="sample_outside_EEG_header"):
        p.summarize_prefix(cells([0, 40, 80]), sample_limit=20)


def test_boundary_sample_beyond_EEG_header_is_not_read():
    r = p.summarize_prefix(cells([0, 40, 80, 3000]), sample_limit=21)
    assert r["group_complete"] and r["sample_scalars_inspected"] == 3
