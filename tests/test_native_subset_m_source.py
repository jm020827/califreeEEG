"""Artificial-only producer contracts; never open the study's data paths."""

from __future__ import annotations

import copy
import hashlib
import importlib.util
import inspect
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from scipy import io, linalg

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "subset_producer_fixture", ROOT / "scripts/run_native_subset_m_source.py"
)
m = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(m)
PLAN = json.loads((ROOT / "configs/analysis/native_subset_m_source39_v1.json").read_bytes())


def small_cache():
    plan = copy.deepcopy(PLAN)
    plan["source_subject_ids"] = [4]
    return plan, {
        key: np.zeros((1, *entry["shape"][1:]), dtype=np.float64)
        for key, entry in plan["cache"].items()
    }


def q_fixture():
    rng = np.random.default_rng(914026)
    return (
        rng.normal(size=(10, 12, 5, 8, 23)),
        rng.uniform(-1, 1, size=(10, 12, 5, 12)),
        rng.uniform(-1, 1, size=(6, 5, 12, 5, 12)),
    )


def q_call(trials, a0, expert, k=3, order=0):
    return m.make_q_features(
        trials,
        a0,
        expert,
        k,
        0,
        125,
        order,
        PLAN["native_weights"]["A0_author"]["dry"],
        PLAN["native_weights"]["ETRCA"]["dry"],
    )


def scalar_scores(r, weights):
    return np.sum(r * np.asarray(weights)[:, None], axis=-2) / np.sum(weights)


def scalar_log(x):
    x = x - x.mean(axis=1, keepdims=True)
    covariance = x @ x.T / (x.shape[1] - 1)
    mean_variance = np.trace(covariance) / 8
    return linalg.logm(
        0.8 * covariance + (0.2 * mean_variance + max(mean_variance, 1e-12) * 1e-8) * np.eye(8)
    ).real


def test_plan_and_import_boundaries():
    assert m.read_plan(ROOT / "configs/analysis/native_subset_m_source39_v1.json") == PLAN
    assert len(m.SOURCE_IDS) == 39 and 1 not in m.SOURCE_IDS
    assert m._NATIVE is None
    assert "query_labels" not in inspect.signature(m.make_q_features).parameters
    assert "impedance" not in inspect.signature(m.compute_participant).parameters
    assert "native_subset_m_core" not in inspect.getsource(m).split("def import_core")[0]


@pytest.mark.parametrize("n", [125, 188, 250, 500])
def test_preprocess_exact_crop_float64_and_locality(n):
    raw = np.arange(8 * 710, dtype=np.float32).reshape(8, 710)
    original = raw.copy()
    received = []

    def preprocess(descriptor, x):
        assert descriptor.srate == 250 and x.dtype == np.float64
        received.append(x.copy())
        x += 7
        return x

    native = SimpleNamespace(
        preprocess=preprocess,
        filterbank=lambda descriptor, x: np.stack([x + f for f in range(5)]),
    )
    actual = m.preprocess_trial(raw, n, native)
    np.testing.assert_array_equal(received[0], original[:, 125 : 160 + n])
    np.testing.assert_array_equal(
        actual, np.stack([original[:, 160 : 160 + n] + 7 + f for f in range(5)])
    )
    np.testing.assert_array_equal(raw, original)
    changed = raw.copy()
    changed[:, :125] = -123
    changed[:, 160 + n :] = -456
    np.testing.assert_array_equal(m.preprocess_trial(changed, n, native), actual)


def test_all_33_q_columns_against_independent_scalar_oracle():
    trials, a0, expert = q_fixture()
    before = trials.copy()
    actual = q_call(trials, a0, expert)
    wa = PLAN["native_weights"]["A0_author"]["dry"]
    we = PLAN["native_weights"]["ETRCA"]["dry"]
    logs = [[scalar_log(trials[b, c, 0]) for c in range(12)] for b in range(3)]
    support_logs = np.mean(logs, axis=1)
    support_scores = scalar_scores(a0[:3], wa)
    true_corr = np.array([np.mean([support_scores[b, c, c] for c in range(12)]) for b in range(3)])
    true_margin = np.array(
        [
            np.mean(
                [
                    support_scores[b, c, c] - max(np.delete(support_scores[b, c], c))
                    for c in range(12)
                ]
            )
            for b in range(3)
        ]
    )
    for omitted in range(3):
        retained = [b for b in range(3) if b != omitted]
        consistency = []
        for c in range(12):
            rhos = [
                np.corrcoef(
                    trials[omitted, c, f].ravel(), trials[retained, c, f].mean(axis=0).ravel()
                )[0, 1]
                for f in range(5)
            ]
            consistency.append(np.dot(rhos, we) / sum(we))
        for qb, c in [(0, 0), (2, 7), (4, 11)]:
            q = qb + 5
            full = scalar_scores(expert[0, qb, c], we)
            drop = scalar_scores(expert[omitted + 1, qb, c], we)
            anchor = scalar_scores(a0[q, c], wa)
            margin = lambda x: np.sort(x)[-1] - np.sort(x)[-2]
            md = margin(drop) - margin(full)
            querylog = scalar_log(trials[q, c, 0])
            distance = np.array(
                [np.log1p(np.linalg.norm(querylog - x, "fro") ** 2 / 8) for x in support_logs]
            )
            dd = distance[omitted] - distance[retained].mean()
            expected = [
                0,
                0.25,
                0.6,
                q / 9,
                omitted / 4,
                (q - omitted) / 9,
                0,
                0,
                max(full),
                margin(full),
                full.std(),
                max(anchor),
                margin(anchor),
                full.argmax() == anchor.argmax(),
                max(drop),
                margin(drop),
                np.sqrt(np.mean((drop - full) ** 2)),
                drop.argmax() == full.argmax(),
                drop.argmax() == anchor.argmax(),
                max(drop) - max(full),
                md,
                true_corr[omitted],
                true_corr[retained].mean(),
                true_margin[omitted],
                true_margin[retained].mean(),
                distance[omitted],
                distance[retained].mean(),
                dd,
                np.mean(consistency),
                min(consistency),
                md**2,
                dd * np.mean(consistency),
                (true_margin[omitted] - true_margin[retained].mean()) * md,
            ]
            np.testing.assert_allclose(actual[omitted, qb, c], expected, atol=2e-13, rtol=1e-12)
    np.testing.assert_array_equal(trials, before)


def test_q_future_support_and_query_locality_and_shared_order():
    trials, a0, experts = q_fixture()
    expected = q_call(trials, a0, experts)
    changed_t, changed_a, changed_e = trials.copy(), a0.copy(), experts.copy()
    changed_t[3:5] *= 100
    changed_a[3:5] *= -1
    changed_e[4:] *= -1
    np.testing.assert_array_equal(q_call(changed_t, changed_a, changed_e), expected)
    changed_t[7, 8] *= 3
    changed_a[7, 8] *= -1
    changed_e[:, 2, 8] *= -1
    changed = q_call(changed_t, changed_a, changed_e)
    mask = np.ones((5, 12), dtype=bool)
    mask[2, 8] = False
    np.testing.assert_array_equal(changed[:, mask], expected[:, mask])
    assert np.any(changed[:, 2, 8] != expected[:, 2, 8])
    other_order = q_call(trials, a0, experts, order=1)
    keep = [j for j in range(33) if j not in (6, 7)]
    np.testing.assert_array_equal(other_order[..., keep], expected[..., keep])
    assert np.all(other_order[..., 6:8] == 1)


def test_covariance_zero_and_support_consistency_zero():
    trials = np.zeros((10, 12, 5, 8, 9))
    np.testing.assert_allclose(
        m.covariance_logs(trials[:, :, 0]),
        np.broadcast_to(np.eye(8) * np.log(1e-20), (10, 12, 8, 8)),
    )
    assert m.support_consistency(trials, 0, [1, 2], [1] * 5) == (0.0, 0.0)


def test_native_score_order_and_prediction_guard():
    rng = np.random.default_rng(714)
    r = rng.uniform(-1, 1, size=(2, 3, 5, 12))
    weights = PLAN["native_weights"]["ETRCA"]["wet"]
    expected = np.stack([np.asarray(weights)[None] @ x for x in r.reshape(-1, 5, 12)]).reshape(
        2, 3, 12
    )
    np.testing.assert_array_equal(m.scores_from_correlations(r, weights), expected)
    model = SimpleNamespace(
        predict=lambda x: (expected.reshape(-1, 12).argmax(-1), r.reshape(-1, 5, 12))
    )
    np.testing.assert_array_equal(
        m.predict_correlations(model, [None] * 6, weights), r.reshape(-1, 5, 12)
    )
    model.predict = lambda x: ([100] * 6, r.reshape(-1, 5, 12))
    with pytest.raises(ValueError, match="predictions"):
        m.predict_correlations(model, [None] * 6, weights)


def test_compute_pool_order_padding_and_no_mutation(monkeypatch):
    fits = []
    query_seen = []
    references = []
    weights = PLAN["native_weights"]

    class Base:
        def __init__(self, **kwargs):
            self.weights = kwargs["weights_filterbank"]
            self.model = {"U": None, "V": None}
            self.offset = 0

        def fit(self, **kwargs):
            if "X" in kwargs:
                codes = [round(float(x[0, 0, 0])) for x in kwargs["X"]]
                fits.append((codes, kwargs["Y"]))
                self.offset = sum(codes) * 1e-8

        def predict(self, x):
            query_seen.append([round(float(v[0, 0, 0])) for v in x])
            r = np.broadcast_to(np.linspace(-0.3, 0.3, 12), (len(x), 5, 12)).copy() + self.offset
            return m.scores_from_correlations(r, self.weights).argmax(-1), r

    def reference(*args):
        references.append(args)
        return np.zeros((10, args[2]))

    native = SimpleNamespace(
        SCCA=Base,
        ETRCA=Base,
        gen_ref_sin=reference,
        suggested_weights_filterbank=lambda f, interface, method: weights[
            "A0_author" if method == "cca" else "ETRCA"
        ][interface],
    )
    raw = np.zeros((8, 710, 2, 10, 12))
    for i in range(2):
        for b in range(10):
            for c in range(12):
                raw[:, :, i, b, c] = 1000 * i + 100 * b + c
    before = raw.copy()
    monkeypatch.setattr(
        m, "preprocess_trial", lambda x, n, native: np.broadcast_to(x[0, 0], (5, 8, n)).copy()
    )
    result = m.compute_participant(raw, PLAN, native, 1)
    assert len(fits) == 80
    assert references == [
        (f, 250, n, 5, phase * np.pi)
        for n in PLAN["sample_counts"]
        for f, phase in zip(PLAN["frequencies"], PLAN["phases_pi"])
    ]
    for j, (codes, labels) in enumerate(fits):
        local = j % 10
        budget = 3 if local < 4 else 5
        expert = local if local < 4 else local - 4
        blocks = (
            list(range(budget)) if expert == 0 else [b for b in range(budget) if b != expert - 1]
        )
        interface = (j // 10) % 2
        assert codes == [1000 * interface + 100 * b + c for b in blocks for c in range(12)]
        assert labels == list(range(12)) * len(blocks)
    for codes in query_seen:
        if len(codes) == 60:
            assert all((v % 1000) // 100 >= 5 for v in codes)
    np.testing.assert_array_equal(raw, before)
    assert not np.any(result["expert_r"][:, :, 0, 4:])
    assert not np.any(result["q_features"][:, :, 0, 3:])
    plan, _ = small_cache()
    m.validate_cache({key: value[None] for key, value in result.items()}, plan)


def test_cache_shapes_domain_finite_padding():
    plan, cache = small_cache()
    m.validate_cache(cache, plan)
    cache["expert_r"][0, 0, 0, 0, 4, 0, 0, 0, 0] = 1
    with pytest.raises(ValueError, match="Padded"):
        m.validate_cache(cache, plan)
    cache["expert_r"].fill(0)
    cache["a0_r"].flat[0] = 2
    with pytest.raises(ValueError, match="domain"):
        m.validate_cache(cache, plan)
    cache["a0_r"].flat[0] = np.nan
    with pytest.raises(ValueError, match="finite"):
        m.validate_cache(cache, plan)


def baseline_fixture(cache):
    old = {
        "a0_r": np.zeros((1, 2, 4, 10, 12, 5, 12)),
        "etrca_chrono_r": cache["expert_r"][:, :, :, :, 0].copy(),
    }
    old["a0_r"][:, :, :, 5:] = cache["a0_r"]
    return old


def test_baseline_last5_and_exact_argmax_even_below_tolerance():
    plan, cache = small_cache()
    old = baseline_fixture(cache)
    old["a0_r"][:, :, :, :5] = 0.9
    assert m.compare_baseline(cache, old, plan)["a0_r"]["argmax_exact"]
    old["etrca_chrono_r"].flat[0] = 2e-12
    with pytest.raises(ValueError, match="correlation mismatch"):
        m.compare_baseline(cache, old, plan)
    old["etrca_chrono_r"].flat[0] = 0
    old["etrca_chrono_r"][0, 0, 0, 0, 0, 0, 0, 1] = 1e-14
    with pytest.raises(ValueError, match="argmax mismatch"):
        m.compare_baseline(cache, old, plan)


def test_baseline_pinned_read_only_decodes_allowed_arrays(tmp_path, monkeypatch):
    plan, cache = small_cache()
    old = baseline_fixture(cache)
    path = tmp_path / "artificial-baseline.npz"
    # Object LOBO data would fail allow_pickle=False if it were decoded.
    with path.open("wb") as stream:
        np.savez_compressed(stream, **old, etrca_lobo_r=np.array([{}], dtype=object))
    digest = m.file_sha256(path)
    monkeypatch.setattr(m, "BASELINE_PATH", path)
    monkeypatch.setattr(m, "BASELINE_SHA256", digest)
    plan["baseline_reference"].update(path=str(path), sha256=digest)
    report = m.check_baseline(cache, plan)
    assert report["input_sha256"] == digest and report["full_expert_r"]["max_abs_error"] == 0
    monkeypatch.setattr(m, "BASELINE_SHA256", "0" * 64)
    plan["baseline_reference"]["sha256"] = "0" * 64
    with pytest.raises(ValueError, match="hash mismatch"):
        m.check_baseline(cache, plan)


def test_projection_receipt_hash_and_allowlist(tmp_path, monkeypatch):
    path = tmp_path / "synthetic.json"
    payload = {
        "manifest_sha256": PLAN["source_projection"]["manifest_sha256"],
        "returned_subject_ids": list(m.SOURCE_IDS),
        "returned_packets": 780,
        "returned_rows": 9360,
        "columns": PLAN["source_projection"]["columns"],
        "packets": [{}] * 780,
    }
    encoded = m.json_bytes(payload)
    path.write_bytes(encoded)
    digest = hashlib.sha256(encoded).hexdigest()
    monkeypatch.setattr(m, "PROJECTION_PATH", path)
    monkeypatch.setattr(m, "PROJECTION_SHA256", digest)
    plan = copy.deepcopy(PLAN)
    plan["source_projection"].update(path=str(path), sha256=digest)
    assert m.load_projection(plan) == payload
    path.write_bytes(encoded + b" ")
    with pytest.raises(ValueError, match="hash mismatch"):
        m.load_projection(plan)
    path.write_bytes(encoded)
    plan["source_projection"]["returned_rows"] = -1
    with pytest.raises(ValueError, match="receipt"):
        m.load_projection(plan)


def test_data_only_loadmat_preserves_stored_dtype_and_float64(tmp_path, monkeypatch):
    raw = (
        np.arange(8 * 710 * 2 * 10 * 12, dtype=np.float64).reshape(8, 710, 2, 10, 12) + 0.123456789
    )
    path = tmp_path / "S004.mat"
    io.savemat(path, {"data": raw, "unrelated": np.ones(3)})
    monkeypatch.setattr(m, "RAW_ROOT", tmp_path)
    plan = copy.deepcopy(PLAN)
    plan["raw_root"] = str(tmp_path)
    original_loader = io.loadmat
    calls = []

    def loadmat(stream, **kwargs):
        calls.append(kwargs)
        return original_loader(stream, **kwargs)

    monkeypatch.setattr(m.io, "loadmat", loadmat)
    monkeypatch.setattr(
        m,
        "compute_participant",
        lambda x, p, n, order: {"same": np.array_equal(x, raw), "order": order},
    )
    subject, arrays, record = m.load_participant(4, plan, 1)
    assert subject == 4 and arrays == {"same": True, "order": 1}
    assert record["stored_dtype"] == "float64" and record["sha256"] == m.file_sha256(path)
    assert calls == [
        {
            "variable_names": ["data"],
            "squeeze_me": False,
            "mat_dtype": False,
            "verify_compressed_data_integrity": True,
        }
    ]
    with pytest.raises(ValueError, match="outside source39"):
        m.load_participant(1, plan, 0)


def test_regular_files_exclusive_readonly_publication_and_quota(tmp_path):
    path = tmp_path / "fixture.json"
    digest = m.publish(path, {"fixture": 1})
    assert digest == m.file_sha256(path) and path.stat().st_mode & 0o777 == 0o400
    with pytest.raises(FileExistsError):
        m.publish(path, {"fixture": 2})
    with pytest.raises(ValueError, match="quota"):
        m.publish(tmp_path / "second.json", {"fixture": 1}, quota=1)
    link = tmp_path / "link.json"
    link.symlink_to(path)
    with pytest.raises(ValueError, match="non-symlink"):
        m.open_regular(link)


@pytest.mark.parametrize(
    "path", ["/tmp/private.mat", "/tmp/private.npz", "/tmp/private.json", "/tmp/private.parquet"]
)
def test_runtime_guard_forbids_unapproved_inputs(path):
    with pytest.raises(RuntimeError):
        m.runtime_guard("open", (path, "r", 0))


def test_runtime_guard_network_import_and_write():
    for event, args in [
        ("socket.connect", ()),
        ("subprocess.Popen", ()),
        ("import", ("SSVEPAnalysisToolbox.datasets.wearable",)),
        ("import", ("cfeg.analysis.retired",)),
        ("open", ("/tmp/forbidden.txt", "w", 1)),
    ]:
        with pytest.raises(RuntimeError):
            m.runtime_guard(event, args)
    m.runtime_guard("open", (str(m.PROJECTION_PATH), "r", 0))
    m.runtime_guard("open", (str(m.RAW_ROOT / "S004.mat"), "r", 0))


def lifecycle_fixture(tmp_path, monkeypatch, fail_fit=False):
    output = tmp_path / "attempt"
    plan = copy.deepcopy(PLAN)
    events = []
    core_path = tmp_path / "core_fixture.py"
    core_path.write_text("# artificial core only\n")

    def preflight(*args):
        if output.exists():
            raise FileExistsError("Consumed attempt")
        return plan, {
            "plan_sha256": m.PLAN_SHA256,
            "source_commit": "fixturecommit",
            "source_tree": "fixturetree",
            "upstream_revision": m.REVISION,
        }

    def projection(p):
        assert (output / "start.json").is_file()
        events.append("projection")
        return {"fixture": True}

    def gather(p, order):
        assert (output / "source-projection.json").is_file()
        events.append("raw")
        return {"fixture": np.ones(2)}, []

    def fit(cache, projection, p):
        assert (output / "features.npz").is_file()
        assert not (output / "result.json").exists()
        events.append("fit_all")
        if fail_fit:
            raise ValueError("Synthetic fit failure")
        return {
            "schema": "cfeg.native-subset-m.freezes.v1",
            "plan_sha256": m.PLAN_SHA256,
            "folds": [{"fold": i} for i in range(3)],
        }

    def evaluate(cache, projection, freeze, p):
        assert (output / "fold-freezes.json").is_file()
        assert json.loads((output / "fold-freezes.json").read_bytes()) == freeze
        events.append("evaluate")
        return {
            "rows": [{}] * 4680,
            "summary": [{}] * 120,
            "contrasts": {},
            "diagnostics": {},
            "attainment": [],
        }

    core = SimpleNamespace(
        __file__=str(core_path),
        projection_arrays=lambda p, plan: (None, np.zeros(39)),
        fit_all=fit,
        evaluate_all=evaluate,
    )
    monkeypatch.setattr(m, "preflight", preflight)
    monkeypatch.setattr(m, "import_native", lambda path: {})
    monkeypatch.setattr(m, "import_core", lambda: core)
    monkeypatch.setattr(m.sys, "addaudithook", lambda hook: None)
    monkeypatch.setattr(m, "load_projection", projection)
    monkeypatch.setattr(m, "gather_participants", gather)
    monkeypatch.setattr(m, "validate_cache", lambda c, p: None)
    monkeypatch.setattr(m, "check_baseline", lambda c, p: {"fixture": True})
    return output, events


def test_lifecycle_all_freezes_precede_evaluation_and_hash_bindings(tmp_path, monkeypatch):
    output, events = lifecycle_fixture(tmp_path, monkeypatch)
    result = m.run(Path("fixture-plan"), output, Path("fixture-upstream"))
    assert events == ["projection", "raw", "fit_all", "evaluate"]
    assert {path.name for path in output.iterdir()} == set(m.ARTIFACTS)
    assert result["features_sha256"] == m.file_sha256(output / "features.npz")
    assert result["source_projection_sha256"] == m.file_sha256(output / "source-projection.json")
    assert result["fold_freezes_sha256"] == m.file_sha256(output / "fold-freezes.json")
    start = json.loads((output / "start.json").read_bytes())
    assert result["started_at"] == start["started_at"]
    assert start["metadata_access"] and not start["held_access"]
    with pytest.raises(FileExistsError):
        m.run(Path("fixture-plan"), output, Path("fixture-upstream"))


def test_partial_failure_is_preserved_and_no_evaluation_or_retry(tmp_path, monkeypatch):
    output, events = lifecycle_fixture(tmp_path, monkeypatch, fail_fit=True)
    with pytest.raises(ValueError, match="Synthetic fit failure"):
        m.run(Path("fixture-plan"), output, Path("fixture-upstream"))
    assert events == ["projection", "raw", "fit_all"]
    assert (output / "start.json").exists() and (output / "features.npz").exists()
    assert not (output / "fold-freezes.json").exists()
    with pytest.raises(FileExistsError):
        m.run(Path("fixture-plan"), output, Path("fixture-upstream"))


def test_preflight_rejects_wrong_paths_and_existing_attempt_without_data(tmp_path, monkeypatch):
    monkeypatch.setattr(m, "PLAN_PATH", tmp_path / "plan.json")
    monkeypatch.setattr(m, "OUTPUT_ROOT", tmp_path / "attempt")
    monkeypatch.setattr(m, "UPSTREAM_ROOT", tmp_path / "upstream")
    with pytest.raises(ValueError, match="exact"):
        m.preflight(tmp_path / "other.json", m.OUTPUT_ROOT, m.UPSTREAM_ROOT)
    m.OUTPUT_ROOT.mkdir()
    with pytest.raises(FileExistsError, match="consumes"):
        m.preflight(m.PLAN_PATH, m.OUTPUT_ROOT, m.UPSTREAM_ROOT)


def test_preflight_clean_pins_runtime_and_dependencies(tmp_path, monkeypatch):
    root, upstream = tmp_path / "project", tmp_path / "upstream"
    monkeypatch.setattr(m, "SOURCE_ROOT", root)
    monkeypatch.setattr(m, "__file__", str(root / "scripts/runner.py"))
    monkeypatch.setattr(m, "PLAN_PATH", root / "plan.json")
    monkeypatch.setattr(m, "OUTPUT_ROOT", tmp_path / "attempt")
    monkeypatch.setattr(m, "UPSTREAM_ROOT", upstream)
    monkeypatch.setattr(m.sys, "executable", str(m.PYTHON_PATH))
    monkeypatch.setattr(m.sys, "version", "3.9.21 fixture")
    monkeypatch.setattr(m.importlib.metadata, "version", lambda name: m.VERSIONS[name])
    monkeypatch.setattr(m, "read_plan", lambda path: PLAN)
    monkeypatch.setattr(
        m, "file_sha256", lambda path: PLAN["upstream"]["pins"][str(path.relative_to(upstream))]
    )
    state = {"dirty": False, "revision": m.REVISION}

    def git(path, *args):
        if args == ("status", "--porcelain"):
            return " M changed" if state["dirty"] else ""
        if args == ("rev-parse", "--abbrev-ref", "HEAD"):
            return "main"
        if path == upstream:
            return state["revision"]
        return "fixture-tree" if args[-1] == "HEAD^{tree}" else "fixture-commit"

    monkeypatch.setattr(m, "git", git)
    for name in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS"):
        monkeypatch.setenv(name, "1")
    _, provenance = m.preflight(m.PLAN_PATH, m.OUTPUT_ROOT, upstream)
    assert provenance["source_commit"] == "fixture-commit"
    state["dirty"] = True
    with pytest.raises(RuntimeError, match="clean"):
        m.preflight(m.PLAN_PATH, m.OUTPUT_ROOT, upstream)
    state["dirty"] = False
    state["revision"] = "wrong-revision"
    with pytest.raises(RuntimeError, match="pinned"):
        m.preflight(m.PLAN_PATH, m.OUTPUT_ROOT, upstream)
    state["revision"] = m.REVISION
    monkeypatch.setenv("OMP_NUM_THREADS", "2")
    with pytest.raises(RuntimeError, match="OMP_NUM_THREADS"):
        m.preflight(m.PLAN_PATH, m.OUTPUT_ROOT, upstream)
