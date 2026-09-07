"""Independent arithmetic fixtures; no human raw files or native toolbox imports."""

import copy
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "author_source_audit", ROOT / "scripts/audit_author_etrca_source.py"
)
AUDIT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(AUDIT)
PLAN_PATH = ROOT / "configs/analysis/author_etrca_source39_v1.json"


@pytest.fixture(scope="module")
def plan():
    assert AUDIT.digest(PLAN_PATH) == AUDIT.PLAN_SHA256
    return json.loads(PLAN_PATH.read_text())


@pytest.fixture(scope="module")
def cache(plan):
    result = {}
    for name in ("a0_r", "etrca_lobo_r", "etrca_chrono_r"):
        spec = plan["cache"][name]
        array = np.zeros(spec["shape"], dtype=np.float64)
        for label in range(12):
            candidate = (label + 1) % 12 if name == "a0_r" else label
            array[..., label, :, candidate] = 1
        result[name] = array
    result["a0_r"][0] = np.roll(result["a0_r"][0], -1, axis=-1)
    result["etrca_chrono_r"][1] = np.roll(result["etrca_chrono_r"][1], 1, axis=-1)
    return result


@pytest.fixture(scope="module")
def reconstructed(cache, plan):
    return AUDIT.reconstruct(cache, plan)


def test_complete_grid(reconstructed):
    rows, summary = reconstructed
    assert len(rows) == 2028 and len(summary) == 52
    assert len({tuple(row[k] for k in ("participant", *AUDIT.KEYS)) for row in rows}) == 2028
    for row in rows:
        assert row["query_count"] == (120 if row["view"] == "all10" else 60)
        assert row["label_count"] == 12 * row["k"]
        assert row["k"] in (0, 3, 5, 9)
    assert all(row["n_participants"] == 39 for row in summary)


def test_known_accuracy_and_rotations(reconstructed):
    for row in reconstructed[1]:
        native = row["stage"] == "native_lobo_2s"
        if row["method"] == "A0_author":
            expected = 1 / 39
        elif row["method"] == "ETRCA":
            expected = 1 if native else 38 / 39
        else:
            expected = 0 if native else 1 / (39 * 11)
        assert row["mean_ba"] == pytest.approx(expected, abs=1e-14)


def test_diagnostics_and_attainment(reconstructed):
    report = AUDIT.diagnostics(reconstructed[0])
    assert len(report["contrasts"]) == 50
    assert len(report["harm"]) == 624
    assert len(report["attainment"]) == 312
    for row in report["attainment"]:
        expected = (
            0
            if row["participant"] == AUDIT.IDS[0]
            else (None if row["participant"] == AUDIT.IDS[1] else 3)
        )
        assert row["first_observed_k"] == expected
    for row in report["overall_harm"]:
        assert row["help"] == 37 and row["tie"] == 2 and row["harm"] == 0


def test_interval():
    result = AUDIT.interval(np.ones(39))
    assert result == {"mean": 1.0, "ci95_low": 1.0, "ci95_high": 1.0, "n_participants": 39}
    result = AUDIT.interval(np.arange(39))
    assert result["mean"] == 19 and result["ci95_low"] < 19 < result["ci95_high"]


@pytest.mark.parametrize("damage", ["missing", "extra", "shape", "dtype", "nan", "range"])
def test_cache_corruption_rejected(cache, plan, damage):
    candidate = dict(cache)
    if damage == "missing":
        candidate.pop("a0_r")
    elif damage == "extra":
        candidate["extra"] = candidate["a0_r"]
    elif damage == "shape":
        candidate["a0_r"] = candidate["a0_r"][:1]
    elif damage == "dtype":
        candidate["a0_r"] = candidate["a0_r"].astype(np.float32)
    else:
        candidate["a0_r"] = candidate["a0_r"].copy()
        candidate["a0_r"].flat[0] = np.nan if damage == "nan" else 1.1
    with pytest.raises(ValueError):
        AUDIT.validate_cache(candidate, plan)


@pytest.mark.parametrize("damage", ["value", "nan", "missing", "extra", "order", "duplicate"])
def test_records_corruption_rejected(reconstructed, damage):
    expected = reconstructed[0]
    candidate = copy.deepcopy(expected)
    if damage == "value":
        candidate[0]["ba"] += 0.01
    elif damage == "nan":
        candidate[0]["ba"] = np.nan
    elif damage == "missing":
        candidate[0].pop("k")
    elif damage == "extra":
        candidate[0]["extra"] = 1
    elif damage == "order":
        candidate[0], candidate[1] = candidate[1], candidate[0]
    else:
        candidate.append(candidate[0])
    with pytest.raises(ValueError):
        AUDIT.compare_records(candidate, expected, "rows")


def test_tiny_roundoff_tolerated(reconstructed):
    candidate = copy.deepcopy(reconstructed[0])
    candidate[0]["ba"] += 1e-15
    AUDIT.compare_records(candidate, reconstructed[0], "rows")


def test_wrong_plan_fails_before_output_access(tmp_path):
    wrong = tmp_path / "wrong.json"
    wrong.write_text("{}")
    with pytest.raises(ValueError, match="frozen plan"):
        AUDIT.audit(wrong, tmp_path / "absent")


@pytest.fixture(scope="module")
def publication(tmp_path_factory, cache, reconstructed, plan):
    output = tmp_path_factory.mktemp("synthetic_publication")
    np.savez_compressed(output / "correlations.npz", **cache)
    binding = {
        "study_id": plan["study_id"],
        "plan_sha256": AUDIT.PLAN_SHA256,
        "source_commit": "1" * 40,
        "source_tree": "2" * 40,
        "upstream_revision": plan["upstream"]["revision"],
        "started_at": "2026-09-07T10:00:00+00:00",
    }
    start = {
        **binding,
        "schema": "cfeg.author-etrca-source.start.v1",
        "source_subject_ids": list(AUDIT.IDS),
        "metadata_access": False,
        "held_access": False,
        "retired_access": False,
        "raw_root": plan["raw_root"],
        "output_root": plan["execution"]["output_root"],
        "python": "3.9.21",
        "dependencies": {
            "numpy": "1.23.4",
            "scipy": "1.13.0",
            "joblib": "1.4.2",
            "scikit-learn": "1.3.0",
            "mat73": "0.63",
            "h5py": "3.11.0",
            "threadpoolctl": "3.5.0",
        },
        "imported_source_hashes": plan["upstream"]["pins"],
    }
    result = {
        **binding,
        "schema": plan["reporting"]["schema"],
        "status": "COMPATIBILITY_ASSESSMENT_COMPLETE",
        "completed_at": "2026-09-07T10:05:00+00:00",
        "cache_sha256": AUDIT.digest(output / "correlations.npz"),
        "raw_files": [
            {
                "subject": p,
                "path": f"{plan['raw_root']}/S{p:03d}.mat",
                "sha256": "3" * 64,
                "stored_dtype": "float64",
                "shape": plan["raw_shape"],
            }
            for p in AUDIT.IDS
        ],
        "rows": reconstructed[0],
        "summary": reconstructed[1],
    }
    for name, payload in (("start.json", start), ("result.json", result)):
        (output / name).write_text(json.dumps(payload))
    for path in output.iterdir():
        path.chmod(0o400)
    return output


def test_full_synthetic_publication(publication):
    report = AUDIT.audit(PLAN_PATH, publication)
    assert report["status"] == "AUDIT_PASS" and report["rows"] == 2028


@pytest.mark.parametrize(
    "damage",
    ["commit", "authority", "cohort", "output", "time", "runtime", "import", "cache", "raw"],
)
def test_publication_binding_corruption(publication, monkeypatch, damage):
    original_loads = json.loads

    def corrupted(encoded, *args, **kwargs):
        value = original_loads(encoded, *args, **kwargs)
        if value.get("schema") == "cfeg.author-etrca-source.start.v1":
            if damage == "commit":
                value["source_commit"] = "4" * 40
            elif damage == "authority":
                value["held_access"] = True
            elif damage == "cohort":
                value["source_subject_ids"][0] = 1
            elif damage == "output":
                value["output_root"] = "/tmp/other"
            elif damage == "runtime":
                value["dependencies"]["numpy"] = "1.26.4"
            elif damage == "import":
                value["imported_source_hashes"] = {}
        if value.get("schema") == "cfeg.author-etrca-source.result.v1":
            if damage == "time":
                value["completed_at"] = "2026-09-07T09:00:00+00:00"
            elif damage == "cache":
                value["cache_sha256"] = "0" * 64
            elif damage == "raw":
                value["raw_files"][0]["path"] = "/tmp/other.mat"
        return value

    monkeypatch.setattr(AUDIT.json, "loads", corrupted)
    with pytest.raises(ValueError):
        AUDIT.audit(PLAN_PATH, publication)
