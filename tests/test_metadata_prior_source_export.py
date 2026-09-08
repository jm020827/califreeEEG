"""Artificial-only export tests; never read raw human or saved metadata/outcomes."""

import copy
import hashlib
import importlib.util
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from scipy import io

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/export_metadata_prior_source.py"
SPEC = importlib.util.spec_from_file_location("prior_source_export_test", SCRIPT)
EXPORT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(EXPORT)
PLAN_FILE = ROOT / "configs/analysis/metadata_prior_source39_v1.json"


@pytest.fixture
def plan():
    return EXPORT.read_plan(PLAN_FILE)


def fake_pipeline(plan):
    events = []

    class Model:
        def __init__(self, **kwargs):
            self.anchor = "n_component" in kwargs
            self.model = {"U": None, "V": None}

        def fit(self, **kwargs):
            if not self.anchor:
                assert kwargs["Y"] == list(range(12)) * (len(kwargs["X"]) // 12)
                self.k = len(kwargs["X"]) // 12
                events.append(("support", self.k, [x[0, 0, 0] for x in kwargs["X"]]))
            else:
                assert len(kwargs["ref_sig"]) == 12

    def preprocess(raw, n, native):
        return np.full((5, 8, n), raw[0, 0], dtype=np.float64)

    def predict(model, query, weights):
        events.append(("query", [x[0, 0, 0] for x in query]))
        return np.full((48, 5, 12), 0.125 if model.anchor else model.k / 10)

    def weights(_n, interface, method):
        return plan["native_weights"]["A0_author" if method == "cca" else "ETRCA"][interface]

    helper = SimpleNamespace(
        preprocess_trial=preprocess,
        references_for_window=lambda n, p, f: [np.zeros((10, n)) for _ in range(12)],
        predict_correlations=predict,
    )
    native = SimpleNamespace(
        SCCA=Model, ETRCA=Model, gen_ref_sin=None, suggested_weights_filterbank=weights
    )
    return helper, native, events


def test_frozen_plan_and_no_human_paths_opened(plan):
    assert len(plan["source_subject_ids"]) == 39
    assert hashlib.sha256(PLAN_FILE.read_bytes()).hexdigest() == EXPORT.PLAN_SHA256
    assert plan["query_blocks"] == [6, 7, 8, 9]


def test_bad_plan_rejected_before_json(tmp_path):
    path = tmp_path / "plan.json"
    path.write_text("{}")
    with pytest.raises(ValueError, match="hash"):
        EXPORT.read_plan(path)


def test_pure_native_complete_geometry_prefix_and_query(plan):
    raw = np.zeros((8, 710, 2, 10, 12), dtype=np.float32)
    for interface in range(2):
        for block in range(10):
            for label in range(12):
                raw[:, :, interface, block, label] = 100 * interface + 10 * block + label
    helper, native, events = fake_pipeline(plan)
    result = EXPORT.compute_participant(raw, plan, helper, native)
    assert set(result) == {
        f"{name}_{n}" for name in ("x", "a0", "full") for n in EXPORT.SAMPLE_COUNTS
    }
    for n in EXPORT.SAMPLE_COUNTS:
        assert result[f"x_{n}"].shape == (2, 10, 12, 5, 8, n)
        assert result[f"full_{n}"].shape == (2, 2, 4, 12, 5, 12)
        assert result[f"a0_{n}"].shape == (2, 4, 12, 5, 12)
        np.testing.assert_array_equal(result[f"x_{n}"][1, 7, 3], np.full((5, 8, n), 173))
        assert np.all(result[f"full_{n}"][:, 0] == 0.3)
        assert np.all(result[f"full_{n}"][:, 1] == 0.5)
        assert np.all(result[f"a0_{n}"] == 0.125)
    for event in events:
        if event[0] == "support":
            assert len(event[2]) == 12 * event[1]
            assert max((value % 100) // 10 for value in event[2]) < event[1] + 1
        else:
            assert len(event[1]) == 48
            # Values retain block/label order. Label10/11 overlaps decimal
            # representation, so verify the exact ordered sequence directly.
            offset = 100 if event[1][0] == 160 else 0
            assert event[1] == [offset + 10 * b + c for b in (6, 7, 8, 9) for c in range(12)]
    assert raw.dtype == np.float32


@pytest.mark.parametrize("kind", ["shape", "complex", "nan", "dtype", "keys", "correlation"])
def test_cache_rejects_invalid(plan, kind):
    local = copy.deepcopy(plan)
    local["sample_counts"] = [125]
    cache = {
        "x_125": np.zeros((2, 10, 12, 5, 8, 125)),
        "full_125": np.zeros((2, 2, 4, 12, 5, 12)),
        "a0_125": np.zeros((2, 4, 12, 5, 12)),
    }
    if kind == "shape":
        cache["x_125"] = cache["x_125"][:, :9]
    elif kind == "complex":
        cache["x_125"] = cache["x_125"].astype(complex)
    elif kind == "nan":
        cache["x_125"].flat[0] = np.nan
    elif kind == "dtype":
        cache["x_125"] = cache["x_125"].astype(np.float32)
    elif kind == "keys":
        cache["extra"] = np.array([1.0])
    else:
        cache["a0_125"].flat[0] = 2.0
    with pytest.raises(ValueError):
        EXPORT.validate_cache(cache, local)


def test_mismatched_native_preset_fails(plan):
    local = copy.deepcopy(plan)
    local["sample_counts"] = [125]
    helper, native, _ = fake_pipeline(plan)
    native.suggested_weights_filterbank = lambda *a: [0] * 5
    with pytest.raises(ValueError, match="weights"):
        EXPORT.compute_participant(np.zeros((8, 710, 2, 10, 12)), local, helper, native)


def test_narrow_raw_loader_exact_variable_dtype_hash_and_no_mutation(plan, tmp_path, monkeypatch):
    monkeypatch.setattr(EXPORT, "RAW_ROOT", tmp_path)
    local = copy.deepcopy(plan)
    local["raw_root"] = str(tmp_path)
    raw = np.arange(8 * 710 * 2 * 10 * 12, dtype=np.float32).reshape(8, 710, 2, 10, 12)
    path = tmp_path / "S004.mat"
    io.savemat(path, {"data": raw, "forbidden_other_variable": np.array([123])})
    actual_loadmat = EXPORT.io.loadmat
    calls = []

    def checked(*args, **kwargs):
        calls.append(kwargs)
        return actual_loadmat(*args, **kwargs)

    monkeypatch.setattr(EXPORT.io, "loadmat", checked)
    actual, receipt = EXPORT.load_raw(4, local)
    np.testing.assert_array_equal(actual, raw)
    assert actual.dtype == np.float64
    assert receipt["stored_dtype"] == "float32"
    assert receipt["sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
    assert calls == [
        {
            "variable_names": ["data"],
            "squeeze_me": False,
            "mat_dtype": False,
            "verify_compressed_data_integrity": True,
        }
    ]


@pytest.mark.parametrize("subject", [1, 3, 5, 103, True, "4"])
def test_loader_forbidden_subject_before_file_open(plan, monkeypatch, subject):
    monkeypatch.setattr(EXPORT, "open_regular", lambda *a: pytest.fail("No data access permitted"))
    with pytest.raises(ValueError, match="source39"):
        EXPORT.load_raw(subject, plan)


def test_raw_change_during_load_fails(plan, tmp_path, monkeypatch):
    monkeypatch.setattr(EXPORT, "RAW_ROOT", tmp_path)
    local = copy.deepcopy(plan)
    local["raw_root"] = str(tmp_path)
    path = tmp_path / "S004.mat"
    path.write_bytes(b"fixture")

    def change(*args, **kwargs):
        path.write_bytes(b"changed")
        return {"data": np.zeros((8, 710, 2, 10, 12))}

    monkeypatch.setattr(EXPORT.io, "loadmat", change)
    with pytest.raises(ValueError, match="changed"):
        EXPORT.load_raw(4, local)


def test_path_rejects_symlink_parent_traversal_and_hardlink(tmp_path):
    target = tmp_path / "target"
    target.write_bytes(b"fixture")
    alias = tmp_path / "alias"
    alias.symlink_to(target)
    with pytest.raises(ValueError):
        EXPORT.open_regular(alias)
    with pytest.raises(ValueError):
        EXPORT.exact_path(tmp_path / ".." / tmp_path.name / "target")
    hard = tmp_path / "hard"
    os.link(target, hard)
    with pytest.raises(ValueError, match="single-link"):
        EXPORT.open_regular(target)


def test_exclusive_publication_and_failed_quota_are_preserved(tmp_path):
    path = tmp_path / "start.json"
    receipt = EXPORT.publish(path, {"status": "STARTED"}, 4096)
    assert path.stat().st_mode & 0o777 == 0o400
    assert receipt["sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
    with pytest.raises(FileExistsError):
        EXPORT.publish(path, {}, 4096)
    failed = tmp_path / "failed.json"
    with pytest.raises(ValueError, match="quota"):
        EXPORT.publish(failed, {"too_large": 1}, 1)
    assert failed.exists() and failed.stat().st_mode & 0o777 == 0o400


def test_npz_publication_is_nonpickle_and_exact(tmp_path):
    path = tmp_path / "S004.npz"
    cache = {"x_125": np.arange(32, dtype=np.float64)}
    EXPORT.publish(path, cache, 4096, npz=True)
    with np.load(path, allow_pickle=False) as archive:
        assert archive.files == ["x_125"]
        np.testing.assert_array_equal(archive["x_125"], cache["x_125"])


@pytest.mark.parametrize(
    "path",
    [
        "/home/whwovy/eeg-data/raw/wearable/S001.mat",
        "/home/whwovy/eeg-data/raw/wearable/S005.mat",
        "/home/whwovy/eeg-data/raw/wearable/Impedance.mat",
        "/home/whwovy/context-template-artifacts/source39-v1/source-projection.json",
        "/home/whwovy/author-etrca-artifacts/source39-v1/correlations.npz",
        "/home/whwovy/califreeEEG/secret.sqlite",
    ],
)
def test_guard_denies_all_metadata_held_and_old_containers(tmp_path, path):
    guard = EXPORT.runtime_guard(tmp_path, library_roots=("/home/whwovy",))
    with pytest.raises(RuntimeError):
        guard("open", (path, "r", os.O_RDONLY))


def test_guard_denies_network_mutation_escape_and_relative_paths(tmp_path):
    guard = EXPORT.runtime_guard(tmp_path, library_roots=(str(tmp_path),))
    for event in ("socket.connect", "socket.getaddrinfo", "subprocess.Popen"):
        with pytest.raises(RuntimeError):
            guard(event, ())
    for path in (str(tmp_path / "unlisted.py"), str(tmp_path / ".." / "escape.py"), "relative.py"):
        with pytest.raises(RuntimeError):
            guard("open", (path, "w", os.O_WRONLY | os.O_CREAT))
    guard("open", (str(tmp_path / "start.json"), "w", os.O_WRONLY | os.O_CREAT))
    with pytest.raises(RuntimeError):
        guard("import", ("SSVEPAnalysisToolbox.datasets",))


@pytest.mark.parametrize("fail", [False, True])
def test_cold_lifecycle_start_before_inputs_manifest_and_no_retry(tmp_path, fail):
    # Only temp paths and artificial arrays. Run the real lifecycle/publication
    # and audit hook in a subprocess because hooks cannot be removed.
    program = r"""
import copy, hashlib, importlib.util, json, pathlib, sys
from types import SimpleNamespace
import numpy as np
spec = importlib.util.spec_from_file_location("export_fixture", sys.argv[1])
module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
plan = json.loads(pathlib.Path(sys.argv[2]).read_text())
root = pathlib.Path(sys.argv[3]); output = root / "native"
module.SOURCE_IDS = (4,)
plan["source_subject_ids"] = [4]
plan["sample_counts"] = [125]
plan["execution"]["max_seconds_per_stage"] = 30
plan["execution"]["budget_bytes"] = 16777216
provenance = {"plan_sha256": module.PLAN_SHA256, "source_commit": "a" * 40,
              "source_tree": "b" * 40, "source_hashes": {},
              "helper_sha256": "d" * 64, "upstream_revision": "e" * 40,
              "upstream_pins": {}}
def preflight(*args):
    if output.exists(): raise FileExistsError("consumed")
    return plan, provenance
module.preflight = preflight
helper = SimpleNamespace(_NATIVE=None, import_native=lambda p: {})
def import_helper(p):
    assert (output / "start.json").exists()
    return helper
module.import_helper = import_helper
def load_raw(subject, p):
    assert (output / "start.json").exists()
    if sys.argv[4] == "True": raise ValueError("declared artificial failure")
    return np.zeros((1,)), {"subject": subject, "sha256": "f" * 64, "stored_dtype": "float64"}
module.load_raw = load_raw
module.compute_participant = lambda *args: {
    "x_125": np.zeros((2,10,12,5,8,125)),
    "full_125": np.zeros((2,2,4,12,5,12)),
    "a0_125": np.zeros((2,4,12,5,12)),
}
failed = False
try:
    result = module.run(pathlib.Path(sys.argv[2]), output)
    assert result["status"] == "COMPLETE"
    assert result["files"][0]["subject"] == 4
    assert result["files"][0]["raw_receipt"]["stored_dtype"] == "float64"
except ValueError:
    failed = True
assert failed == (sys.argv[4] == "True")
assert (output / "start.json").stat().st_mode & 0o777 == 0o400
assert (output / "result.json").exists() == (not failed)
try: module.run(pathlib.Path(sys.argv[2]), output)
except FileExistsError: pass
else: raise AssertionError("Retry was permitted")
print("COLD PASS")
"""
    result = subprocess.run(
        [sys.executable, "-c", program, str(SCRIPT), str(PLAN_FILE), str(tmp_path), str(fail)],
        capture_output=True,
        text=True,
        timeout=45,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "COLD PASS" in result.stdout


def test_human_cli_in_worktree_refuses_before_data_or_attempt(tmp_path):
    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--output", str(tmp_path / "not-canonical")],
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert result.returncode != 0
    assert "canonical" in result.stderr
    assert not (tmp_path / "not-canonical").exists()
