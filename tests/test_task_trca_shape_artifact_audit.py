"""Persisted artificial evaluation replay and adversarial source-graph checks."""

import copy
import hashlib
import json

import numpy as np
import pytest
import torch

from cfeg.analysis import task_trca_shape_artifact_audit as audit
from cfeg.analysis import task_trca_shape_evaluation as ev
from cfeg.analysis import task_trca_shape_learning as learn


def test_saved_artificial_evaluation_replay_and_corruption(tmp_path, monkeypatch):
    torch.set_num_threads(1)
    monkeypatch.setattr(audit, "SAMPLES", (17,))
    rng = np.random.default_rng(20260911)
    source = rng.normal(size=(5, 12, 5, 8, 17)) + 0.3
    task = rng.normal(size=(12, 5, 8, 17)) + 0.3
    packet = rng.uniform(0, 20, (5, 8))
    packet[:, 3] = np.nan
    weights = np.array([1.25, 0.67, 0.5, 0.43, 0.38])
    freq = np.linspace(9, 14.5, 12)
    case = learn.make_task_case(11001, 0, 0, source[:3], packet[:3], freq, task, weights=weights)
    model = learn.fit_pipeline((case,), 0.001, backend="batch")
    states = {}
    for pid in (11002, 11003):
        for interface in (0, 1):
            for k in (3, 5):
                states[pid, interface, 17, k] = ev.support_state(
                    pid, interface, 0, source[:k], packet[:k] * (pid - 11000), freq, weights
                )
    records = []
    for key, state in states.items():
        donor = states[(22005 - key[0], *key[1:])]
        result = ev.evaluate(model, state, np.tile(task, (4, 1, 1, 1)), donor=donor)
        result.update(
            keys=np.array(key),
            orders=state.order,
            q=state.q,
            m=state.m,
            available=state.available,
            packet5=np.pad(state.packet, ((0, 5 - state.k), (0, 0)), constant_values=np.nan),
            s=state.s.numpy(),
            c=state.c.numpy(),
            anchors=state.anchors.numpy(),
            weights=state.weights.numpy(),
            donor_id=donor.participant_id,
            cached_full_correlations=result["native_full_correlations"].copy(),
        )
        records.append(result)
    packed = {name: np.stack([r[name] for r in records]) for name in records[0]}
    path = tmp_path / "evaluation.npz"
    with path.open("xb") as stream:
        np.savez(stream, **packed)
    model_path = tmp_path / "model.json"
    model_path.write_text(json.dumps(model.record()))
    saved = json.loads(model_path.read_text())
    with np.load(path, allow_pickle=False) as stored:
        reloaded = {key: stored[key] for key in stored.files}
    receipt = audit.audit_evaluation(reloaded, saved)
    assert receipt["status"] == "EVALUATION_AUDIT_PASS"
    assert receipt["max_abs_errors"]["scores"] < 1e-10
    for field in ("m", "scores", "r", "donor_id", "keys"):
        corrupted = {k: v.copy() for k, v in reloaded.items()}
        corrupted[field].flat[0] += 1
        with pytest.raises((ValueError, AssertionError)):
            audit.audit_evaluation(corrupted, saved)


def source_fixture():
    ids = tuple(range(12001, 12007))
    keys = np.array(
        [[p, i, n, k] for p in ids for i in (0, 1) for n in audit.SAMPLES for k in (3, 5)]
    )
    packet = np.ones((len(keys), 5, 8))
    packet[keys[:, 3] == 3, 3:] = np.nan
    q = np.zeros((len(keys), 5, 8, 15))
    q[..., 4] = 1
    data = {
        "keys": keys,
        "packet5": packet,
        "q": q,
        "m": np.zeros((len(keys), 8, 2)),
        "available": np.ones((len(keys), 8), dtype=bool),
        "orders": np.zeros(len(keys), dtype=int),
        "weights": np.ones((len(keys), 5)),
        "labels": np.tile(np.arange(12), (len(keys), 1)),
    }

    def record(fit_ids, lam):
        positions = np.flatnonzero(np.isin(keys[:, 0], fit_ids))

        def scaler(dim):
            return {
                "mean": ([0] * 4 + [1] + [0] * 10) if dim == 15 else [0, 0],
                "scale": [1] * dim,
                "fit_ids": list(fit_ids),
            }

        def head(dim):
            return {
                "coefficients": [0.0] * dim,
                "steps": 200,
                "initial_loss": float(np.log(12)),
                "final_loss": float(np.log(12)),
                "trace": [
                    {
                        "step": j,
                        "loss_before_step": float(np.log(12)),
                        "ce_before_step": float(np.log(12)),
                        "gradient_norm": 0.0,
                    }
                    for j in range(1, 201)
                ],
            }

        digest = hashlib.sha256()
        qs = scaler(15)
        for values in (qs["mean"], qs["scale"], [0] * 16):
            digest.update(np.array(values, dtype="<f8").tobytes())
        return {
            "fit_ids": list(fit_ids),
            "lambda": lam,
            "q_scaler": qs,
            "m_scaler": scaler(2),
            "q2_scaler": scaler(2),
            "q_hash": digest.hexdigest(),
            "Q": head(16),
            "residuals": {arm: head(3) for arm in learn.RESIDUAL_ARMS},
            "donors": [
                {"case": keys[j].tolist(), "donor_id": int(keys[d, 0])}
                for j, d in audit.donor_positions(data, positions).items()
            ],
        }

    rows = []
    for fold in (0, 1, 2):
        val = ids[fold::3]
        fit = tuple(p for p in ids if p not in val)
        for lam in audit.LAMBDAS:
            rows.append(
                {
                    "inner_fold": fold,
                    "lambda": lam,
                    "fit_ids": list(fit),
                    "validation_ids": list(val),
                    "pipeline": record(fit, lam),
                    "validation_ce": {
                        arm: float(np.log(12)) for arm in ("Q", *learn.RESIDUAL_ARMS)
                    },
                }
            )
    model = {
        "pipeline": record(ids, 0.01),
        "selection": {"inner": rows, "selected_lambda": 0.01, "outer_evaluation_ids": [13001]},
    }
    return data, model, ids


def test_source_graph_replay_selection_and_mutations(monkeypatch):
    data, model, ids = source_fixture()
    monkeypatch.setattr(audit, "scores", lambda *args: (np.zeros((12, 12)), None, None))
    result = audit.audit_training(data, model, ids, (13001,))
    assert result["selected_lambda"] == 0.01 and result["validation_ce_comparisons"] == 36
    for field in ("keys", "labels", "m"):
        bad = {k: v.copy() for k, v in data.items()}
        bad[field].flat[0] += 1
        with pytest.raises((ValueError, AssertionError)):
            audit.audit_training(bad, model, ids, (13001,))
    bad = copy.deepcopy(model)
    bad["selection"]["selected_lambda"] = 0.001
    with pytest.raises(ValueError, match="selection"):
        audit.audit_training(data, bad, ids, (13001,))


def test_cold_runner_declines_historical_design_before_human_access(tmp_path):
    import importlib.util
    from pathlib import Path

    spec = importlib.util.spec_from_file_location(
        "task_cold_runner",
        Path(__file__).resolve().parents[1] / "scripts/run_task_trca_shape_source39.py",
    )
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)
    path = tmp_path / "design.json"
    path.write_text(
        json.dumps({"schema": "cfeg.task_aligned_trca_shape.design.v1", "status": "DESIGN_ONLY"})
    )
    with pytest.raises(ValueError, match="Executable frozen manifest"):
        runner.run(path, hashlib.sha256(path.read_bytes()).hexdigest())
