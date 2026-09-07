"""Artificial-only producer tests. No author environment or human files required."""

import copy
import importlib.util
import io
import json
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from scipy import io as scipy_io

SPEC = importlib.util.spec_from_file_location(
    "author_source_producer",
    Path(__file__).resolve().parents[1] / "scripts/run_author_etrca_source.py",
)
PRODUCER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(PRODUCER)
PLAN_FILE = Path(__file__).resolve().parents[1] / "configs/analysis/author_etrca_source39_v1.json"


@pytest.fixture
def plan():
    return PRODUCER.read_plan(PLAN_FILE)


def artificial_cache(plan):
    result = {}
    for name in PRODUCER.CACHE_KEYS:
        shape = (len(plan["source_subject_ids"]), *plan["cache"][name]["shape"][1:])
        # Deliberate exact ties for some queries check argmax after score roll.
        arr = np.zeros(shape, dtype=np.float64)
        for label in range(12):
            arr[..., label, :, label] = 0.8
        arr[..., 0, :, :] = 0.1
        result[name] = arr
    return result


@pytest.mark.parametrize("n_samples", [125, 188, 250, 500])
def test_native_slice_order_copy_and_float64(n_samples):
    raw = np.arange(8 * 710, dtype=np.float32).reshape(8, 710)
    original = raw.copy()
    seen = []

    def preprocess(descriptor, values):
        assert descriptor.srate == 250
        seen.append(values.copy())
        values += 1
        return values

    def bank(descriptor, values):
        return np.stack([values * (band + 1) for band in range(5)])

    native = SimpleNamespace(preprocess=preprocess, filterbank=bank)
    output = PRODUCER.preprocess_trial(raw, n_samples, native)
    np.testing.assert_array_equal(seen[0], raw[:, 125 : 160 + n_samples])
    expected = np.stack([(raw[:, 160 : 160 + n_samples] + 1) * (b + 1) for b in range(5)])
    np.testing.assert_array_equal(output, expected)
    np.testing.assert_array_equal(raw, original)
    assert output.dtype == np.float64
    changed = raw.copy()
    changed[:, :125] = 123456
    changed[:, 160 + n_samples :] = -123456
    np.testing.assert_array_equal(PRODUCER.preprocess_trial(changed, n_samples, native), output)


@pytest.mark.parametrize(
    "raw,n_samples",
    [
        (np.zeros((7, 710)), 125),
        (np.zeros((8, 710)), 499),
        (np.zeros((8, 710), dtype=complex), 125),
        (np.full((8, 710), np.nan), 125),
    ],
)
def test_invalid_trials_fail_without_native_call(raw, n_samples):
    with pytest.raises(ValueError):
        PRODUCER.preprocess_trial(raw, n_samples, None)


def test_reference_argument_order_no_latency_shift(plan):
    calls = []

    def generate(*args):
        calls.append(args)
        return np.zeros((10, args[2]))

    references = PRODUCER.references_for_window(188, plan, generate)
    assert len(references) == 12
    assert calls == [
        (f, 250, 188, 5, phase * np.pi) for f, phase in zip(plan["frequencies"], plan["phases_pi"])
    ]


def test_complete_rows_summary_counts_rotations_and_aliases(plan):
    cache = artificial_cache(plan)
    rows = PRODUCER.rows_from_cache(cache, plan)
    assert len(rows) == 2028
    assert rows == sorted(
        rows, key=lambda r: tuple(r[k] for k in ("participant", *PRODUCER.ROW_KEY))
    )
    assert {r["method"] for r in rows} == {"A0_author", "ETRCA", "ETRCA_rotated"}
    for row in rows:
        assert row["query_count"] == (120 if row["view"] == "all10" else 60)
        assert row["label_count"] == 12 * row["k"]
        assert row["k"] != 1
        if row["method"] != "ETRCA_rotated":
            assert row["ba"] == 1
        else:
            assert row["ba"] == pytest.approx(1 / 12)
    summaries = PRODUCER.summarize_rows(rows)
    assert len(summaries) == 52
    assert all(s["n_participants"] == 39 for s in summaries)
    for summary in summaries:
        assert summary["ci95_low"] == summary["mean_ba"]
        assert summary["ci95_high"] == summary["mean_ba"]
    lookup = {
        (r["participant"], r["interface"], r["stage"], r["view"], r["method"], r["k"]): r
        for r in rows
        if r["n_samples"] == 500
    }
    for s in plan["source_subject_ids"]:
        for interface in plan["interfaces"]:
            a = lookup[s, interface, "native_lobo_2s", "last5", "A0_author", 0]
            b = lookup[s, interface, "bridge_chronological_2s", "last5", "A0_author", 0]
            assert a["ba"] == b["ba"]


def test_native_matmul_and_rotated_argmax_ties():
    r = np.zeros((1, 12, 5, 12), dtype=np.float64)
    weights = [0.7, 0.5, 0.3, 0.2, 0.1]
    scores = PRODUCER.scores_from_correlations(r, weights)
    expected = np.asarray([np.array(weights)[None] @ x for x in r.reshape(-1, 5, 12)]).reshape(
        1, 12, 12
    )
    np.testing.assert_array_equal(scores, expected)
    # All ties stay at class zero after rotating scores, not prediction+shift.
    assert PRODUCER.balanced_accuracy(r, weights, rotated=True) == pytest.approx(1 / 12)


def test_summary_uses_people_not_trials_and_rejects_duplicates():
    rows = [
        {
            "participant": p,
            "stage": "s",
            "view": "v",
            "interface": "dry",
            "n_samples": 125,
            "method": "ETRCA",
            "k": 3,
            "ba": ba,
            "query_count": count,
            "label_count": 36,
        }
        for p, ba, count in [(4, 0.0, 60), (6, 1.0, 120), (8, 0.5, 60)]
    ]
    summary = PRODUCER.summarize_rows(rows)[0]
    assert summary["mean_ba"] == 0.5
    assert summary["n_participants"] == 3
    assert summary["ci95_low"] < 0.5 < summary["ci95_high"]
    with pytest.raises(ValueError, match="Repeated"):
        PRODUCER.summarize_rows(rows + rows[:1])


@pytest.mark.parametrize("mutation", ["missing", "extra", "shape", "nan", "range", "float32"])
def test_cache_schema_and_numeric_fail_closed(plan, mutation):
    plan["source_subject_ids"] = [4]
    cache = artificial_cache(plan)
    if mutation == "missing":
        cache.pop("a0_r")
    elif mutation == "extra":
        cache["extra"] = np.zeros(1)
    elif mutation == "shape":
        cache["a0_r"] = cache["a0_r"][:, :, :-1]
    elif mutation == "nan":
        cache["a0_r"].flat[0] = np.nan
    elif mutation == "range":
        cache["a0_r"].flat[0] = 1.1
    else:
        cache["a0_r"] = cache["a0_r"].astype(np.float32)
    with pytest.raises(ValueError):
        PRODUCER.rows_from_cache(cache, plan)


def test_public_api_wiring_all_splits_and_query_label_independence(plan):
    raw = np.zeros((8, 710, 2, 10, 12), dtype=np.float64)
    for i in range(2):
        for b in range(10):
            for c in range(12):
                raw[:, :, i, b, c] = i * 120 + b * 12 + c
    fits, predictions = [], []

    class Anchor:
        def __init__(self, **kwargs):
            assert kwargs["n_jobs"] is None
            assert kwargs["n_component"] == 1
            assert kwargs["update_UV"] is True
            assert kwargs["force_output_UV"] is False
            self.model = {"U": None, "V": None}

        def fit(self, *, ref_sig):
            assert len(ref_sig) == 12

        def predict(self, trials):
            r = np.zeros((len(trials), 5, 12))
            for q, trial in enumerate(trials):
                r[q, :, int(trial[0, 0, 0]) % 12] = 0.8
            return r[:, 0].argmax(axis=-1).tolist(), r

    class Ensemble(Anchor):
        def __init__(self, **kwargs):
            assert kwargs["n_jobs"] is None

        def fit(self, *, X, Y):
            self.ids = [int(x[0, 0, 0]) for x in X]
            assert Y == [x % 12 for x in self.ids]
            assert len(X) in (36, 60, 108)
            fits.append((X[0].shape[-1], self.ids.copy()))

        def predict(self, trials):
            ids = [int(x[0, 0, 0]) for x in trials]
            assert not set(ids) & set(self.ids)
            predictions.append((self.ids.copy(), ids))
            return super().predict(trials)

    native = SimpleNamespace(
        SCCA=Anchor,
        ETRCA=Ensemble,
        preprocess=lambda _, x: x,
        filterbank=lambda _, x: np.stack([x] * 5),
        gen_ref_sin=lambda f, sf, n, h, phase: np.zeros((2 * h, n)),
        suggested_weights_filterbank=lambda _, interface, method: plan["methods"]["weights"][
            "A0_author" if method == "cca" else "ETRCA"
        ][interface],
    )
    arrays = PRODUCER.compute_participant(raw, plan, native)
    assert len(fits) == 36  # 16 chronological fits plus20 LOBO fits.
    assert sum(len(ids) == 108 for _, ids in fits) == 20
    for train_ids, query_ids in predictions:
        if len(train_ids) != 108:
            assert [x % 120 for x in train_ids] == list(range(len(train_ids)))
            assert [x % 120 for x in query_ids] == list(range(60, 120))
        else:
            query_block = query_ids[0] % 120 // 12
            assert all(x % 120 // 12 != query_block for x in train_ids)
    for key, array in arrays.items():
        assert array.shape == tuple(plan["cache"][key]["shape"][1:])
        assert array.dtype == np.float64


def test_loader_data_only_native_shape_and_no_float32_roundtrip(plan, monkeypatch, tmp_path):
    # Synthetic MAT fixture only; no raw EEG directory is even inspected.
    raw = np.full((8, 710, 2, 10, 12), 1.00000000001, dtype=np.float64)
    buffer = io.BytesIO()
    scipy_io.savemat(buffer, {"data": raw, "forbidden_extra": np.arange(3)})
    path = tmp_path / "S004.mat"
    path.write_bytes(buffer.getvalue())
    monkeypatch.setattr(PRODUCER, "RAW_ROOT", tmp_path)
    plan["raw_root"] = str(tmp_path)
    original = scipy_io.loadmat
    seen = []

    def load(stream, **kwargs):
        assert kwargs == {
            "variable_names": ["data"],
            "squeeze_me": False,
            "mat_dtype": False,
            "verify_compressed_data_integrity": True,
        }
        result = original(stream, **kwargs)
        assert "forbidden_extra" not in result
        return result

    monkeypatch.setattr(PRODUCER.io, "loadmat", load)
    monkeypatch.setattr(
        PRODUCER, "compute_participant", lambda data, p, n: seen.append(data.copy()) or {}
    )
    subject, arrays, record = PRODUCER.load_participant(4, plan)
    assert subject == 4 and arrays == {}
    assert record["stored_dtype"] == "float64"
    assert record["shape"] == [8, 710, 2, 10, 12]
    assert record["sha256"] == PRODUCER.file_sha256(path)
    np.testing.assert_array_equal(seen[0], raw)
    assert seen[0].flat[0] != np.float64(np.float32(raw.flat[0]))


@pytest.mark.parametrize("subject", [1, 3, 5, 103])
def test_loader_rejects_non_source_before_file_access(subject, plan, monkeypatch):
    monkeypatch.setattr(
        PRODUCER.os, "open", lambda *a, **k: pytest.fail("No file access permitted")
    )
    with pytest.raises(ValueError):
        PRODUCER.load_participant(subject, plan)


def test_loader_rejects_symlink_before_loading(plan, monkeypatch, tmp_path):
    monkeypatch.setattr(PRODUCER, "RAW_ROOT", tmp_path)
    plan["raw_root"] = str(tmp_path)
    (tmp_path / "S004.mat").symlink_to(tmp_path / "absent.mat")
    monkeypatch.setattr(PRODUCER.io, "loadmat", lambda *a, **k: pytest.fail("No loading"))
    with pytest.raises(ValueError, match="Symlink"):
        PRODUCER.load_participant(4, plan)


def test_tampered_plan_hash_and_allowlist_fail_closed(plan, monkeypatch, tmp_path):
    path = tmp_path / "plan.json"
    path.write_text(json.dumps(plan))
    with pytest.raises(ValueError, match="hash"):
        PRODUCER.read_plan(path)
    plan["source_subject_ids"][0] = 1
    path.write_text(json.dumps(plan))
    monkeypatch.setattr(PRODUCER, "PLAN_SHA256", PRODUCER.file_sha256(path))
    with pytest.raises(ValueError, match="authority"):
        PRODUCER.read_plan(path)


def test_exclusive_publication_roundtrip_and_permissions(tmp_path):
    path = tmp_path / "fixture.npz"
    arrays = {"only": np.array([1.0, 2.0])}
    digest = PRODUCER.publish(path, arrays, compressed=True)
    assert digest == PRODUCER.file_sha256(path)
    assert path.stat().st_mode & 0o777 == 0o400
    with np.load(path, allow_pickle=False) as data:
        np.testing.assert_array_equal(data["only"], arrays["only"])
    with pytest.raises(FileExistsError):
        PRODUCER.publish(path, arrays, compressed=True)


def test_start_precedes_raw_and_failure_preserves_consumed_attempt(plan, monkeypatch, tmp_path):
    output = tmp_path / "attempt"
    provenance = {
        "plan_sha256": PRODUCER.PLAN_SHA256,
        "source_commit": "a" * 40,
        "source_tree": "b" * 40,
        "upstream_revision": PRODUCER.REVISION,
    }
    monkeypatch.setattr(PRODUCER, "preflight", lambda *a: (plan, provenance))
    monkeypatch.setattr(PRODUCER.sys, "addaudithook", lambda hook: None)
    monkeypatch.setattr(PRODUCER, "import_native", lambda _: {})
    calls = []

    def gather(_):
        calls.append("raw")
        assert (output / "start.json").exists()
        assert json.loads((output / "start.json").read_text())["metadata_access"] is False
        raise ValueError("Artificial numeric failure")

    monkeypatch.setattr(PRODUCER, "gather_participants", gather)
    with pytest.raises(ValueError, match="Artificial"):
        PRODUCER.run(PLAN_FILE, output, tmp_path)
    assert {p.name for p in output.iterdir()} == {"start.json"}
    with pytest.raises(FileExistsError):
        PRODUCER.run(PLAN_FILE, output, tmp_path)
    assert calls == ["raw"]


def test_full_artifact_lifecycle_from_artificial_cache(plan, monkeypatch, tmp_path):
    output = tmp_path / "attempt"
    provenance = {
        "plan_sha256": PRODUCER.PLAN_SHA256,
        "source_commit": "a" * 40,
        "source_tree": "b" * 40,
        "upstream_revision": PRODUCER.REVISION,
    }
    monkeypatch.setattr(PRODUCER, "preflight", lambda *a: (plan, provenance))
    monkeypatch.setattr(PRODUCER.sys, "addaudithook", lambda hook: None)
    monkeypatch.setattr(PRODUCER, "import_native", lambda _: {"fixture.py": "c" * 64})
    monkeypatch.setattr(PRODUCER, "gather_participants", lambda _: (artificial_cache(plan), []))
    result = PRODUCER.run(PLAN_FILE, output, tmp_path)
    start = json.loads((output / "start.json").read_text())
    assert result["started_at"] == start["started_at"]
    assert result["cache_sha256"] == PRODUCER.file_sha256(output / "correlations.npz")
    assert len(result["rows"]) == 2028 and len(result["summary"]) == 52
    assert result["status"] == "COMPATIBILITY_ASSESSMENT_COMPLETE"
    assert {p.name for p in output.iterdir()} == {"start.json", "correlations.npz", "result.json"}


@pytest.mark.parametrize(
    "event,args",
    [
        ("import", ("cfeg.data",)),
        ("import", ("SSVEPAnalysisToolbox.datasets",)),
        ("socket.connect", (None,)),
        ("socket.getaddrinfo", (None,)),
        ("subprocess.Popen", (None,)),
        ("open", ("/unknown/S001.mat", "rb", 0)),
        ("open", ("/unknown/Impedance.mat", "rb", 0)),
        ("open", ("/unknown/manifest.parquet", "rb", 0)),
        ("open", ("/unknown/result.json", "wb", os.O_WRONLY)),
    ],
)
def test_runtime_guard_negative_events_without_real_access(event, args):
    with pytest.raises(RuntimeError):
        PRODUCER.runtime_guard(event, args)


def test_preflight_rejects_arbitrary_paths_before_runtime_checks(tmp_path):
    with pytest.raises(ValueError, match="exact"):
        PRODUCER.preflight(tmp_path / "plan.json", tmp_path / "output", tmp_path)


def test_predict_rejects_invalid_native_results():
    model = SimpleNamespace(predict=lambda _: ([1], np.zeros((1, 5, 12))))
    with pytest.raises(ValueError, match="disagree"):
        PRODUCER.predict_correlations(model, [None], [1] * 5)


@pytest.mark.parametrize(
    "failure",
    [
        None,
        "source_dirty",
        "branch",
        "upstream_revision",
        "upstream_dirty",
        "pin",
        "python_path",
        "python_version",
        "dependency",
        "blas",
        "existing",
    ],
)
def test_preflight_binds_checkout_runtime_and_existing_attempt(
    plan, monkeypatch, tmp_path, failure
):
    # All paths, version readers and git responses are artificial; no live study
    # output directory, environment mutation or raw directory checks occur here.
    root = tmp_path / "source"
    root.mkdir()
    upstream = tmp_path / "upstream"
    upstream.mkdir()
    plan_path = root / "plan.json"
    plan_path.write_text("fixture")
    output = tmp_path / "out"
    monkeypatch.setattr(PRODUCER, "SOURCE_ROOT", root)
    monkeypatch.setattr(PRODUCER, "PLAN_PATH", plan_path)
    monkeypatch.setattr(PRODUCER, "OUTPUT_ROOT", output)
    monkeypatch.setattr(PRODUCER, "UPSTREAM_ROOT", upstream)
    monkeypatch.setattr(PRODUCER, "__file__", str(root / "scripts/runner.py"))
    monkeypatch.setattr(PRODUCER, "read_plan", lambda _: plan)
    monkeypatch.setattr(PRODUCER, "PYTHON_PATH", tmp_path / "python")
    monkeypatch.setattr(PRODUCER.sys, "executable", str(tmp_path / "python"))
    monkeypatch.setattr(PRODUCER.sys, "version", "3.9.21 fixture")
    monkeypatch.setattr(PRODUCER.importlib.metadata, "version", lambda key: PRODUCER.VERSIONS[key])
    monkeypatch.setattr(
        PRODUCER,
        "file_sha256",
        lambda path: plan["upstream"]["pins"][str(path.relative_to(upstream))],
    )
    for env in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS"):
        monkeypatch.setenv(env, "1")

    def git(path, *args):
        if args == ("status", "--porcelain"):
            return (
                " M fixture"
                if failure == ("source_dirty" if path == root else "upstream_dirty")
                else ""
            )
        if args == ("rev-parse", "--abbrev-ref", "HEAD"):
            return "wrong" if failure == "branch" else "main"
        if path == upstream:
            return "wrong" if failure == "upstream_revision" else PRODUCER.REVISION
        return "a" * 40

    monkeypatch.setattr(PRODUCER, "git", git)
    if failure == "pin":
        monkeypatch.setattr(PRODUCER, "file_sha256", lambda path: "wrong")
    elif failure == "python_path":
        monkeypatch.setattr(PRODUCER.sys, "executable", "/fixture/wrong/python")
    elif failure == "python_version":
        monkeypatch.setattr(PRODUCER.sys, "version", "3.10.1 fixture")
    elif failure == "dependency":
        monkeypatch.setattr(PRODUCER.importlib.metadata, "version", lambda _: "wrong")
    elif failure == "blas":
        monkeypatch.setenv("OPENBLAS_NUM_THREADS", "2")
    elif failure == "existing":
        output.mkdir()
    if failure is None:
        loaded, provenance = PRODUCER.preflight(plan_path, output, upstream)
        assert loaded is plan
        assert provenance["source_commit"] == "a" * 40
        assert provenance["source_tree"] == "a" * 40
        assert provenance["dependencies"] == PRODUCER.VERSIONS
    else:
        with pytest.raises((RuntimeError, FileExistsError)):
            PRODUCER.preflight(plan_path, output, upstream)


def test_worker_failure_terminates_only_owned_pool(plan, monkeypatch):
    calls = []

    class FakeProcess:
        def terminate(self):
            calls.append("terminate")

    class FakePool:
        def __init__(self, **kwargs):
            assert kwargs["max_workers"] == 4
            self._processes = {n: FakeProcess() for n in range(4)}

        def submit(self, fn, subject, given_plan):
            assert fn is PRODUCER.load_participant
            assert given_plan is plan
            return subject

        def shutdown(self, **kwargs):
            assert kwargs == {"wait": True, "cancel_futures": True}
            calls.append("shutdown")

    def failed(*args, **kwargs):
        raise TimeoutError("artificial deadline")

    monkeypatch.setattr(PRODUCER, "ProcessPoolExecutor", FakePool)
    monkeypatch.setattr(PRODUCER, "as_completed", failed)
    with pytest.raises(TimeoutError):
        PRODUCER.gather_participants(plan)
    assert calls == ["terminate"] * 4 + ["shutdown"]


def test_mutated_public_preset_fails_before_classifier_fit(plan):
    plan = copy.deepcopy(plan)
    plan["methods"]["weights"]["A0_author"]["dry"][0] += 0.1
    native = SimpleNamespace(
        preprocess=lambda _, x: x,
        filterbank=lambda _, x: np.stack([x] * 5),
        gen_ref_sin=lambda *args: np.zeros((10, args[2])),
        suggested_weights_filterbank=lambda *args: [1] * 5,
    )
    with pytest.raises(ValueError, match="weights"):
        PRODUCER.compute_participant(np.zeros((8, 710, 2, 10, 12)), plan, native)
