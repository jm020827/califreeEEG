"""Artificial-only source-prior mathematics and real cold analysis lifecycle."""

import copy
import hashlib
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from cfeg.analysis import metadata_prior_source as core
from cfeg.analysis import metadata_trca_prior as op

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
SPEC = importlib.util.spec_from_file_location(
    "prior_source_runner_test", ROOT / "scripts/run_metadata_prior_source.py"
)
RUNNER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(RUNNER)


@pytest.fixture
def plan():
    return json.loads((ROOT / "configs/analysis/metadata_prior_source39_v1.json").read_text())


def test_impedance_zero_missing_and_relative_geometry():
    packet = np.array([[0, 1, 3, 7, 15, 31, 63, np.nan]] * 3, dtype=float)
    features, mask = core.m_features(packet)
    assert mask.tolist() == [True] * 7 + [False]
    np.testing.assert_allclose(features[:7, 0], np.arange(-3, 4) * np.log(2), atol=1e-14)
    np.testing.assert_array_equal(features[:, 1], 0)
    np.testing.assert_array_equal(features[7], 0)
    empty, observed = core.m_features(np.full((3, 8), np.nan))
    np.testing.assert_array_equal(empty, 0)
    assert not observed.any()
    shuffled = packet.copy()
    shuffled[1, :7] *= 2
    np.testing.assert_allclose(core.m_features(shuffled)[0], core.m_features(shuffled[::-1])[0])


@pytest.mark.parametrize("bad", [-1.0, np.inf, -np.inf])
def test_impedance_invalid_rejected(bad):
    packet = np.ones((3, 8))
    packet[0, 0] = bad
    with pytest.raises(ValueError):
        core.m_features(packet)


def test_feature_boundaries_and_numeric_M_exclusion(plan):
    plan["sample_counts"] = [32]
    rng = np.random.default_rng(908)
    arrays = {"x_32": rng.normal(size=(2, 10, 12, 5, 8, 32))}
    z = rng.uniform(0, 100, (2, 10, 8))
    q, target = core.subject_features(arrays, z, 0, plan)
    assert q.shape == (2, 10, 8, 15)
    changed = {"x_32": arrays["x_32"].copy()}
    changed["x_32"][:, 6:] *= 100
    q2, target2 = core.subject_features(changed, z * 9, 0, plan)
    np.testing.assert_array_equal(q, q2)
    np.testing.assert_array_equal(target, target2)
    changed["x_32"][:, 5] *= 3
    q2, target2 = core.subject_features(changed, z, 0, plan)
    np.testing.assert_array_equal(q, q2)
    assert not np.array_equal(target, target2)
    missing = z.copy()
    missing[0, 0, 0] = np.nan
    q2, _ = core.subject_features(arrays, missing, 0, plan)
    np.testing.assert_array_equal(q[..., :4], q2[..., :4])
    assert q2[0, 0, 0, 4] == 2 / 3
    np.testing.assert_array_equal(q[0, 0, :, 7:], np.eye(8))


def test_proxy_target_formula_and_scale_invariance():
    rng = np.random.default_rng(77)
    x = rng.normal(size=(3, 12, 5, 8, 32))
    future = rng.normal(size=x.shape[1:])
    expected = np.log(
        np.mean((future - x.mean(0)) ** 2, axis=(0, 3)) / np.mean(x**2, axis=(0, 1, 3, 4))[:, None]
    )
    np.testing.assert_allclose(core.proxy_target(x, future), expected)
    np.testing.assert_allclose(core.proxy_target(x * 10, future * 10), expected)


def test_donors_stay_in_partition_order_and_whole_mask():
    z = np.ones((9, 2, 10, 8))
    order = np.zeros(9, dtype=int)
    order[3] = 1
    z[4, 0, 0, 0] = np.nan
    indices = [0, 2, 3, 4]
    maps = core.donor_maps(z, order, indices, list(range(9)), [3, 5])
    assert maps[0, 0].tolist() == [2, 0, 3, 4]
    assert maps[1, 0].tolist() == [2, 4, 3, 0]
    assert set(maps.ravel()) <= set(indices)


def artificial_summary_input(plan):
    plan = copy.deepcopy(plan)
    plan["source_subject_ids"] = [4, 6, 8]
    plan["sample_counts"] = [125]
    rows = []
    for p in plan["source_subject_ids"]:
        for i in plan["interfaces"]:
            for k, arms in ((0, ["A0"]), (3, core.ARMS), (5, core.ARMS)):
                for arm in arms:
                    rows.append(
                        {
                            "participant": p,
                            "interface": i,
                            "n_samples": 125,
                            "k": k,
                            "arm": arm,
                            "accuracy": 0.5,
                            "proxy_components": None,
                            "prediction_changes_vs_Q": None if k == 0 else 0,
                        }
                    )
    features = {"z": np.ones((3, 2, 10, 8))}
    freezes = {
        "folds": [
            {
                "eval_ids": plan["source_subject_ids"],
                "eval_maps": np.broadcast_to(np.arange(3), (2, 2, 3)).tolist(),
            }
        ]
    }
    return rows, features, freezes, plan


def test_summary_no_reach_does_not_invent_cost(plan):
    report = core.summarize(*artificial_summary_input(plan))
    assert report["calibration"]["neither_reached"] == 6
    assert report["calibration"]["mean_label_delta_both"] is None
    assert report["verdict"]["status"] == "METADATA_INCREMENT_NOT_ESTABLISHED"
    assert all(row["labels"] is None for row in report["attainment"])


def test_summary_pooled_both_cost_new_lost_and_anchor(plan):
    rows, features, freezes, small = artificial_summary_input(plan)
    for row in rows:
        p, k, arm = row["participant"], row["k"], row["arm"]
        # p4 saves24labels; p6 newlyreaches; p8 losesreach. No cost to missing.
        if (
            (p == 4 and (k == 5 or (k == 3 and arm == "QM")))
            or (p == 6 and k == 3 and arm == "QM")
            or (p == 8 and k == 3 and arm == "Q")
        ):
            row["accuracy"] = 39 / 48
    report = core.summarize(rows, features, freezes, small)
    cal = report["calibration"]
    assert [cal[k] for k in ("both_reached", "new_reach", "lost_reach", "neither_reached")] == [
        2,
        2,
        2,
        0,
    ]
    assert cal["mean_label_delta_both"] == -24
    assert cal["transitions"] == {"5->3": 2, "None->3": 2, "3->None": 2}
    rows[0]["accuracy"] = 0.9
    report = core.summarize(rows, features, freezes, small)
    assert (
        next(
            r
            for r in report["attainment"]
            if r["participant"] == 4 and r["interface"] == "dry" and r["arm"] == "QM"
        )["labels"]
        == 0
    )


def test_summary_rejects_duplicate_grid(plan):
    rows, features, freezes, small = artificial_summary_input(plan)
    rows[-1] = rows[0]
    with pytest.raises(ValueError, match="grid"):
        core.summarize(rows, features, freezes, small)


@pytest.mark.parametrize(
    "path",
    [
        "/home/whwovy/eeg-data/raw/wearable/S004.mat",
        "/home/whwovy/eeg-data/raw/wearable/S005.mat",
        "/home/whwovy/eeg-data/raw/wearable/Impedance.mat",
        "/tmp/old-outcome.json",
    ],
)
def test_analysis_guard_denies_raw_held_and_old(plan, tmp_path, path):
    with pytest.raises(RuntimeError):
        RUNNER.guard_for(plan, tmp_path)("open", (path, "r", os.O_RDONLY))


def test_projection_exact_envelope_and_zero_values(plan, tmp_path):
    plan["source_subject_ids"] = [4]
    packets = [
        {
            "subject_id": 4,
            "interface": i,
            "block_id": b,
            "impedance_kohm": [0] * 8,
            "headband_order": "wet",
            "condition_period": "first" if i == "wet" else "second",
        }
        for i in plan["interfaces"]
        for b in range(10)
    ]
    value = {
        **plan["source_projection"]["envelope_provenance"],
        "manifest_sha256": plan["source_projection"]["manifest_sha256"],
        "packets": packets,
        "returned_packets": 20,
        "returned_rows": 240,
        "returned_subject_ids": [4],
        "columns": plan["source_projection"]["columns"],
    }
    path = tmp_path / "projection.json"
    path.write_text(json.dumps(value))
    plan["source_projection"].update(
        path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest(), returned_rows=240
    )
    z, order = RUNNER.projection(plan)
    np.testing.assert_array_equal(z, 0)
    assert order.tolist() == [1]
    value["packets"][-1] = value["packets"][0]
    path.write_text(json.dumps(value))
    plan["source_projection"]["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    with pytest.raises(ValueError, match="Duplicate"):
        RUNNER.projection(plan)


def test_actual_math_cold_analysis_on_artificial_nine_people(plan, tmp_path):
    """Real extraction/fit/scoring/guard/publication; artificial native anchor uses op.

    This proves lifecycle compatibility, not independent native EEG equivalence.
    """
    plan["source_subject_ids"] = list(range(11, 20))
    plan["sample_counts"] = [32]
    plan["analysis_cache"]["features"].update(q=[9, 2, 10, 8, 15], target=[9, 2, 10, 8])
    plan["analysis_cache"]["scores"].update(
        scores=[9, 2, 1, 2, 9, 48, 12], a0_scores=[9, 2, 1, 48, 12]
    )
    native = tmp_path / "native"
    native.mkdir()
    plan["execution"].update(
        native_root=str(native), analysis_python=sys.executable, max_seconds_per_stage=120
    )
    rng = np.random.default_rng(9082026)
    files, packets = [], []
    for p in plan["source_subject_ids"]:
        x = rng.normal(size=(2, 10, 12, 5, 8, 32))
        full = np.empty((2, 2, 4, 12, 5, 12))
        for i, interface in enumerate(plan["interfaces"]):
            for bi, k in enumerate(plan["budgets"]):
                model = op.fit_trca(x[i, :k], np.ones((5, 8)), 0)
                _, corr = op.score_trca(
                    model,
                    x[i, 6:10].reshape(48, 5, 8, 32),
                    np.array(plan["native_weights"]["ETRCA"][interface]),
                )
                full[i, bi] = corr.reshape(4, 12, 5, 12)
            for b in range(10):
                packets.append(
                    {
                        "subject_id": p,
                        "interface": interface,
                        "block_id": b,
                        "impedance_kohm": rng.uniform(0, 300, 8).tolist(),
                        "headband_order": "dry",
                        "condition_period": "first" if i == 0 else "second",
                    }
                )
        path = native / f"S{p:03d}.npz"
        np.savez(path, x_32=x, full_32=full, a0_32=np.zeros((2, 4, 12, 5, 12)))
        files.append(
            {
                "subject": p,
                "filename": path.name,
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "bytes": path.stat().st_size,
            }
        )
    spec = plan["source_projection"]
    value = {
        **spec["envelope_provenance"],
        "manifest_sha256": spec["manifest_sha256"],
        "packets": packets,
        "returned_packets": 180,
        "returned_rows": 2160,
        "returned_subject_ids": plan["source_subject_ids"],
        "columns": spec["columns"],
    }
    projection = tmp_path / "projection.json"
    projection.write_text(json.dumps(value))
    spec.update(
        path=str(projection),
        sha256=hashlib.sha256(projection.read_bytes()).hexdigest(),
        returned_rows=2160,
    )
    plan_path = tmp_path / "plan.json"
    plan_path.write_text(json.dumps(plan))
    plan_sha = hashlib.sha256(plan_path.read_bytes()).hexdigest()
    (native / "start.json").write_text("{}")
    manifest = {
        "status": "COMPLETE",
        "study_id": plan["study_id"],
        "plan_sha256": plan_sha,
        "source_subject_ids": plan["source_subject_ids"],
        "start_sha256": hashlib.sha256(b"{}").hexdigest(),
        "files": files,
    }
    (native / "result.json").write_text(json.dumps(manifest))
    program = r"""
import hashlib, importlib.util, json, pathlib, subprocess, sys
root=pathlib.Path(sys.argv[1]); fixture=pathlib.Path(sys.argv[2])
sys.path.insert(0,str(root / "scripts"))
spec=importlib.util.spec_from_file_location("cold_prior",root / "scripts/run_metadata_prior_source.py")
m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
m.PLAN=fixture / "plan.json"; m.PLAN_SHA=hashlib.sha256(m.PLAN.read_bytes()).hexdigest()
m.OUTPUT=fixture / "analysis"
def fake_git(command, **kwargs):
    if command[1:3] == ["status", "--porcelain"]: return ""
    if command[1:3] == ["branch", "--show-current"]: return "main\n"
    return "a" * 40 + "\n"
m.subprocess.check_output=fake_git
r=m.run()
assert len(r["rows"])==9*2*19
assert r["compatibility"]["max_native_correlation_error"]==0
assert r["provenance"]["start_sha256"]==m.digest(m.OUTPUT / "start.json")
for name in ("start.json","features.npz","fold-freezes.json","scores.npz","result.json"):
    assert (m.OUTPUT/name).stat().st_mode & 0o777 == 0o400
f=json.loads((m.OUTPUT/"fold-freezes.json").read_text())
for fold in f["folds"]:
    assert set(fold["fit_ids"]).isdisjoint(fold["eval_ids"])
    for cell in fold["cells"]:
        for inner in cell["q_receipt"]["oof_folds"]:
            assert set(inner["fit_ids"]).isdisjoint(inner["oof_ids"])
try: m.run()
except ValueError: pass
else: raise AssertionError("Existing attempt overwritten")
print("COLD REAL MATH PASS")
"""
    completed = subprocess.run(
        [sys.executable, "-c", program, str(ROOT), str(tmp_path)],
        env={
            **os.environ,
            "PYTHONDONTWRITEBYTECODE": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "OMP_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
        },
        capture_output=True,
        text=True,
        timeout=150,
        check=False,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert "COLD REAL MATH PASS" in completed.stdout
    # Cross-lane check: independent auditor consumes the actual artificial
    # producer bytes, not a separately handwritten result fixture.
    spec = importlib.util.spec_from_file_location(
        "prior_cross_lane_audit", ROOT / "scripts/audit_metadata_prior_source.py"
    )
    auditor = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(auditor)
    auditor.PLAN_SHA256 = plan_sha
    output = tmp_path / "analysis"
    with np.load(output / "features.npz", allow_pickle=False) as archive:
        features = dict(archive)
    with np.load(output / "scores.npz", allow_pickle=False) as archive:
        scores = dict(archive)
    freezes = json.loads((output / "fold-freezes.json").read_text())
    result = json.loads((output / "result.json").read_text())
    checked = auditor.verify_results(scores, features, freezes, result, plan)
    assert checked["rows_checked"] == 342
    assert checked["selected_fit_replays"] == 210
