"""Nonregistered toy arrays and explicitly mocked fit records; no Adam calls.

The mock model coefficients/traces are not evidence of successful training. The
unchanged N1 evaluator supplies component-level comparison scores for testing the
independent audit, not the other way round. Registered seed20260916 is never run.
"""

import hashlib
import json
from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest
import torch
from scipy import stats

from cfeg.analysis import n1_metadata_generated_efficacy_audit as cold
from cfeg.analysis import task_trca_n1_evaluation as ev
from cfeg.analysis import task_trca_n1_learning as learn
from cfeg.analysis import task_trca_n1_signfree as temporal
from cfeg.analysis import task_trca_shape_features as features
from cfeg.analysis import task_trca_shape_operator as shape


def config():
    path = Path(__file__).resolve().parents[1] / "configs/n1_metadata_generated_efficacy_v1.json"
    value = json.loads(path.read_text())
    value.update(seed=73, fit_groups=2, evaluation_groups=2)
    return value


def save_json(path, value):
    if path.exists():
        path.chmod(0o600)
    path.write_text(json.dumps(value, allow_nan=False))
    path.chmod(0o400)


def save_npz(path, value):
    if path.exists():
        path.chmod(0o600)
    np.savez(path, **value)
    path.chmod(0o400)


def pin(path):
    return {"sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "bytes": path.stat().st_size}


def toy_group(cfg, split, group):
    """Independent test-only DGP construction, never import the fixture module."""
    assert cfg["seed"] in (73, 74)

    def rng(purpose, *extra):
        return np.random.Generator(
            np.random.PCG64(np.random.SeedSequence([cfg["seed"], split, group, purpose, *extra]))
        )

    n = 256
    bad, null = np.zeros(8, bool), np.zeros(8, bool)
    bad[rng(1).choice(8, 4, replace=False)] = True
    null[rng(2).choice(8, 4, replace=False)] = True
    phases = rng(3).uniform(0, 2 * np.pi, (5, 12))
    signal = np.empty((5, 12, n))
    columns = [np.ones(n) / 16]
    harmonics = sorted({h * m for h in range(1, 6) for m in range(6, 18)})
    for frequency in harmonics:
        columns.extend(
            [
                np.sin(2 * np.pi * frequency * np.arange(n) / n) / np.sqrt(n / 2),
                np.cos(2 * np.pi * frequency * np.arange(n) / n) / np.sqrt(n / 2),
            ]
        )
    basis = np.column_stack(columns)
    support = np.empty((3, 12, 5, 8, n))
    for band in range(5):
        matrix = rng(4, band).standard_normal((n, 24))
        for _ in range(2):
            matrix -= basis @ (basis.T @ matrix)
        q, r = np.linalg.qr(matrix, mode="reduced")
        q *= np.sign(np.diag(r))[None]
        noise = (16 * q.T).reshape(3, 8, n)
        for label in range(12):
            signal[band, label] = np.sqrt(2) * np.sin(
                2 * np.pi * (label + 6) * np.arange(n) / n + phases[band, label]
            )
            support[:, label, band] = signal[band, label] + 0.2 * noise
    source_delta = rng(5).uniform(-0.01, 0.01, 12)
    query_delta = rng(6).uniform(-0.01, 0.01, (4, 12))
    result = {
        "support": support,
        "bad": np.stack((bad, ~bad)),
        "null_mask": null,
        "phases": phases,
        "source_delta": source_delta,
        "query_delta": query_delta,
    }
    for key, delta in (("source", source_delta), ("query", query_delta.reshape(48))):
        wave = np.empty((2, len(delta), 5, 8, n))
        for member in range(2):
            for trial, difference in enumerate(delta):
                for channel in range(8):
                    corrupted = result["bad"][member, channel]
                    label = (trial + int(corrupted)) % 12
                    wave[member, trial, :, channel] = (
                        1 + (-1 if corrupted else 1) * difference
                    ) * signal[:, label]
        result[key] = wave
    result["packet_coupled"] = np.repeat(
        np.expm1(1 + 2 * result["bad"].astype(float))[:, None], 3, axis=1
    )
    result["packet_null"] = np.broadcast_to(np.expm1(1 + 2 * null.astype(float)), (2, 3, 8)).copy()
    return result


def toy_split(cfg, split):
    groups = cfg["fit_groups" if split == 0 else "evaluation_groups"]
    base = cfg["fit_id_base" if split == 0 else "evaluation_id_base"]
    draws = [toy_group(cfg, split, group) for group in range(groups)]
    rows = [(member, group) for member in range(2) for group in range(groups)]
    result = {
        "ids": np.arange(base + 1, base + 2 * groups + 1),
        "groups": np.tile(np.arange(groups), 2),
        "members": np.repeat(np.arange(2), groups),
    }
    result["support"] = np.stack([draws[g]["support"] for _, g in rows])
    for key in ("bad", "packet_coupled", "packet_null"):
        result[key] = np.stack([draws[g][key][m] for m, g in rows])
    for key in ("null_mask", "phases"):
        result[key] = np.stack([draw[key] for draw in draws])
    if split == 0:
        result["source"] = np.stack([draws[g]["source"][m] for m, g in rows])
        result["source_delta"] = np.stack([draw["source_delta"] for draw in draws])
        return result
    query = {key: result[key].copy() for key in ("ids", "groups", "members")}
    query["query"] = np.stack([draws[g]["query"][m] for m, g in rows])
    query["query_delta"] = np.stack([draw["query_delta"] for draw in draws])
    return result, query


def mock_pipeline(states, strength):
    """Hand-authored coefficients plus MOCK traces, never a fitted learner."""
    ids = tuple(state.participant_id for state in states)
    q, m = np.stack([s.q for s in states]), np.stack([s.m for s in states])
    qs = features.fit_scaler(q, ids, ids)
    ms = features.fit_scaler(m, ids, ids)
    q2s = features.fit_scaler(q[..., 1:3], ids, ids)

    def head(coefficients):
        mock_trace = tuple(
            {"step": i, "loss_before_step": 1.0, "ce_before_step": 1.0, "gradient_norm": 0.01}
            for i in range(1, 201)
        )
        return learn.HeadFit(np.asarray(coefficients, dtype=float), 1.0, 1.0, mock_trace)

    coefficients = np.zeros(16)
    coefficients[7:15] = np.linspace(-0.02, 0.02, 8)
    residuals = {
        "QM": head([strength, 0, 0]),
        "Q2": head([-0.1, 0.02, 0]),
        "SHAM_REFIT": head([-0.3, 0, 0.01]),
    }
    donors = {state.key: ids[(i + 1) % len(ids)] for i, state in enumerate(states)}
    return learn.Pipeline(ids, 0.001, qs, ms, q2s, head(coefficients), residuals, donors)


def build_mock_artifacts(folder, cfg):
    """Real unchanged forward components, mocked fit only; no producer imports."""
    torch.set_num_threads(1)
    fit = toy_split(cfg, 0)
    support, query = toy_split(cfg, 1)
    for name, arrays in (("fit.npz", fit), ("support.npz", support), ("query.npz", query)):
        save_npz(folder / name, arrays)
    fixture = {
        "schema": "n1-metadata-generated-efficacy-fixture-v1",
        "config": cfg,
        "counts": {key: cfg[key] for key in ("fit_groups", "evaluation_groups")},
        "artifacts": {name: pin(folder / name) for name in ("fit.npz", "support.npz", "query.npz")},
    }
    save_json(folder / "fixture.json", fixture)
    frequencies, weights = np.arange(6, 18) * 250 / 256, np.ones(5)

    def states(data, regime):
        return tuple(
            ev.support_state(
                int(pid),
                0,
                0,
                data["support"][i],
                data["packet_" + regime][i],
                frequencies,
                weights,
            )
            for i, pid in enumerate(data["ids"])
        )

    pipelines = {}
    feature_rows = {key: [] for key in ("q", "m", "s", "c")}
    geometry = 0.0
    for regime in cold.REGIMES:
        source_states = states(fit, regime)
        pipelines[regime] = mock_pipeline(source_states, 0.4 if regime == "coupled" else 0.0)
        for key, values in feature_rows.items():
            values.append(np.stack([getattr(s, key) for s in source_states]))
        for state in source_states:
            geometry = max(
                geometry,
                float(np.max(np.abs(state.s.numpy() - 6 * 256))),
                float(
                    np.max(np.abs(state.c.numpy() - 3 * 256 * (np.ones((8, 8)) + 0.04 * np.eye(8))))
                ),
            )
    save_npz(
        folder / "fit_features.npz", {key: np.stack(values) for key, values in feature_rows.items()}
    )
    model_record = {
        "schema": "n1-metadata-generated-efficacy-models-v1",
        "pipelines": {key: value.record() for key, value in pipelines.items()},
        "oracle": {"coefficients": [2, 0, 0], "trained": False},
    }
    save_json(folder / "models.json", model_record)
    timestamp = "2026-09-10T01:00:00+00:00"
    freeze = {
        "schema": "n1-metadata-generated-efficacy-freeze-v1",
        "utc": timestamp,
        "models_sha256": pin(folder / "models.json")["sha256"],
        "query_sha256": pin(folder / "query.npz")["sha256"],
        "completed_updates": 1600,
        "event": "both_models_frozen_before_query_decode",
    }
    save_json(folder / "freeze.json", freeze)
    arrays = {
        key: []
        for key in ("scores", "r", "projectors", "oracle_scores", "oracle_r", "oracle_projectors")
    }
    actuation = {}
    for regime in cold.REGIMES:
        p, current = pipelines[regime], states(support, regime)
        partition = ev.EvaluationPartition(
            current, tuple(int(i) for i in support["ids"]), ((0, 256, 3),)
        )
        rows = {key: [] for key in arrays}
        values = {
            "max_abs_R": 0.0,
            "max_abs_F": 0.0,
            "max_abs_J": 0.0,
            "max_abs_score": 0.0,
            "argmax_changed": 0,
            "gradient_max": 0.01,
            "coefficient_max_abs": float(np.abs(p.residuals["QM"].coefficients).max()),
        }
        for index, state in enumerate(current):
            value = ev.evaluate(p, state, query["query"][index], partition=partition)
            for key in ("scores", "r", "projectors"):
                rows[key].append(value[key])
            with torch.no_grad():
                qlog = learn._head(
                    ev._tensor(p.q_scaler.transform(state.q)), ev._tensor(p.q.coefficients), 0.8
                )
                mlog = learn._head(
                    ev._tensor(p.m_scaler.transform(state.m)), ev._tensor([2, 0, 0]), 0.2
                )
                prior = shape.shape_prior(qlog + mlog)
                projector = temporal.bounded_projectors(state.s, state.c, prior[:, None])
                statistics = temporal.temporal_statistics(
                    ev._tensor(state.model.templates), ev._tensor(query["query"][index])
                )
                scores, _ = temporal.score_temporal_gram(projector, statistics, state.weights)
            rows["oracle_r"].append(prior.numpy())
            rows["oracle_projectors"].append(projector.numpy())
            rows["oracle_scores"].append(scores.numpy())
            for key, number in value["actuation_QM_minus_Q"].items():
                values[key] = (
                    values[key] + number if key == "argmax_changed" else max(values[key], number)
                )
        actuation[regime] = values
        for key in arrays:
            arrays[key].append(np.stack(rows[key]))
    arrays = {key: np.stack(value) for key, value in arrays.items()}
    save_npz(folder / "evaluation.npz", arrays)
    # Gate logic is tested separately against hand-constructed counts below;
    # this fixture's scope is the raw-to-unchanged-forward-component bridge.
    summary = cold._summary(
        arrays["scores"],
        arrays["oracle_scores"],
        cfg,
        {
            "finite_arrays": True,
            "common_q_hashes": True,
            "constant_support_geometry": True,
            "donors_cross_group": True,
        },
        actuation,
    )
    start = {
        "schema": "n1-metadata-generated-efficacy-primary-start-v1",
        "utc": timestamp,
        "config": cfg,
        "registered_attempt": 1,
        "charged_updates": 1600,
        "prohibited": ["human_data", "held60", "old_models", "outreach", "paid", "gpu"],
    }
    save_json(folder / "primary_start.json", start)
    events = [
        ("generation_started", {}),
        ("generation_completed", {"artifacts": fixture["artifacts"]}),
    ]
    for i, regime in enumerate(cold.REGIMES):
        events += [
            (
                "fit_started",
                {"regime": regime, "charged_updates": 800, "completed_updates": i * 800},
            ),
            (
                "fit_completed",
                {
                    "regime": regime,
                    "completed_updates": (i + 1) * 800,
                    "q_hash": pipelines[regime].q_hash,
                },
            ),
        ]
    events += [
        (
            "both_models_frozen_before_query_decode",
            {"models_sha256": freeze["models_sha256"], "completed_updates": 1600},
        ),
        ("query_decode_started", {}),
        ("regime_evaluation_completed", {"regime": "coupled"}),
        ("regime_evaluation_completed", {"regime": "null"}),
        ("evaluation_completed", {"terminal": summary["terminal"], "completed_updates": 1600}),
    ]
    (folder / "events.jsonl").write_text(
        "\n".join(
            json.dumps(dict(sequence=i + 1, event=name, utc=timestamp, **fields))
            for i, (name, fields) in enumerate(events)
        )
        + "\n"
    )
    (folder / "events.jsonl").chmod(0o400)
    result = dict(
        schema="n1-metadata-generated-efficacy-result-v1",
        config=cfg,
        arms=list(cold.ARMS),
        **summary,
        actuation=actuation,
        pipeline_q_hashes={key: p.q_hash for key, p in pipelines.items()},
        updates_completed=1600,
        geometry_max_abs=geometry,
        artifacts={name: pin(folder / name) for name in cold.ARTIFACTS},
    )
    save_json(folder / "result.json", result)
    return result


@pytest.fixture(scope="module")
def completed(tmp_path_factory):
    folder = tmp_path_factory.mktemp("mock_completed")
    cfg = config()
    result = build_mock_artifacts(folder, cfg)
    return folder, cfg, result


def test_completed_mock_forward_replay_never_calls_learner(completed, monkeypatch):
    folder, cfg, result = completed

    def forbidden(*args, **kwargs):
        pytest.fail("audit must not call producer evaluator, fixture or optimizer")

    monkeypatch.setattr(learn, "fit_pipeline", forbidden)
    monkeypatch.setattr(ev, "evaluate", forbidden)
    monkeypatch.setattr(ev, "frozen_prior", forbidden)
    receipt = cold.audit(folder, cfg)
    assert receipt["status"] == "PASS", receipt
    assert receipt["recomputed_summary"]["terminal"] == result["terminal"]
    assert receipt["updates_completed_from_records"] == 1600
    assert receipt["result_sha256"] == pin(folder / "result.json")["sha256"]
    assert max(receipt["max_differences"].values()) <= 1e-8
    assert len(receipt["limitations"]) >= 5
    assert not (folder / "audit.json").exists()


def test_independent_second_toy_raw_recipe_and_noise_guards():
    cfg = config()
    cfg["seed"] = 74
    fit = toy_split(cfg, 0)
    checks = cold._Checks(1e-8)
    cold._raw_split(fit, None, cfg, 0, checks)
    fit["support"][0, 1, 0, 0, 0] += 0.001
    with pytest.raises(ValueError, match="support_formula"):
        cold._raw_split(fit, None, cfg, 0, cold._Checks(1e-8))


def test_mutated_waveform_and_packet_rejected(completed):
    folder, cfg, _ = completed
    data = cold._arrays(folder / "fit.npz", cold._shapes(cfg)["fit.npz"])
    original = data["source"][0, 0, 0, 0, 0]
    data["source"][0, 0, 0, 0, 0] += 0.01
    with pytest.raises(ValueError, match="waveform"):
        cold._raw_split(data, None, cfg, 0, cold._Checks(1e-8))
    data["source"][0, 0, 0, 0, 0] = original
    data["packet_null"][0, 0, 0] += 0.01
    with pytest.raises(ValueError, match="packet_null"):
        cold._raw_split(data, None, cfg, 0, cold._Checks(1e-8))


@pytest.mark.parametrize(
    "mutation", ["hash", "failure", "writable", "extra", "symlink", "hardlink"]
)
def test_integrity_precedes_array_decode(completed, tmp_path, monkeypatch, mutation):
    folder, cfg, _ = completed
    # Link-free tiny placeholders are sufficient: all errors must occur before
    # numeric decoding. No repeated 40MB copies are needed for negative cases.
    for name in cold.ARTIFACTS | {"result.json"}:
        target = tmp_path / name
        target.write_bytes(
            (folder / name).read_bytes() if name.endswith((".json", ".jsonl")) else b"{}"
        )
        target.chmod(0o400)
    if mutation == "failure":
        save_json(tmp_path / "failure.json", {"terminal": "EFFICACY_NOT_EVALUATED"})
    elif mutation == "writable":
        (tmp_path / "fit.npz").chmod(0o600)
    elif mutation == "extra":
        (tmp_path / "unexpected").mkdir()
    elif mutation == "symlink":
        (tmp_path / "fit.npz").unlink()
        (tmp_path / "fit.npz").symlink_to(folder / "fit.npz")
    elif mutation == "hardlink":
        (tmp_path / "outside").hardlink_to(tmp_path / "fit.npz")
        # Exact inventory also catches this; direct descriptor verifies link guard.
        with pytest.raises(ValueError, match="single-link"):
            cold._descriptor(tmp_path / "fit.npz")

    def forbidden(*args, **kwargs):
        pytest.fail("numeric decode before integrity rejection")

    monkeypatch.setattr(cold, "_arrays", forbidden)
    receipt = cold.audit(tmp_path, cfg)
    assert receipt["status"] == "FAIL"
    assert receipt["terminal_override"] == "EFFICACY_NOT_EVALUATED"
    assert receipt["errors"]
    if mutation == "hash":
        assert "hash/byte inventory" in receipt["errors"][0]


@pytest.mark.parametrize(
    "field", ["q_scaler", "q2_scaler", "m_scaler", "donors", "q_hash", "trace", "coefficient"]
)
def test_model_scaler_donor_and_trace_tampering(completed, field):
    folder, cfg, _ = completed
    data = cold._arrays(folder / "fit.npz", cold._shapes(cfg)["fit.npz"])
    arrays = cold._arrays(folder / "fit_features.npz", cold._shapes(cfg)["fit_features.npz"])
    record = json.loads((folder / "models.json").read_text())["pipelines"]["coupled"]
    if field.endswith("scaler"):
        record[field]["mean"][0] += 0.1
    elif field == "donors":
        record["donors"][0]["donor_id"] = record["fit_ids"][0]
    elif field == "q_hash":
        record["q_hash"] = "0" * 64
    elif field == "trace":
        record["residuals"]["QM"]["trace"][1]["step"] = 1
    else:
        record["Q"]["coefficients"][0] = float("nan")
    with pytest.raises(ValueError):
        cold._pipeline(record, data["ids"], arrays["q"][0], arrays["m"][0], cfg, cold._Checks(1e-8))


@pytest.mark.parametrize(
    "mutation",
    [
        "query_before_freeze",
        "double_query",
        "sequence",
        "elapsed",
        "model_hash",
        "count",
        "terminal",
    ],
)
def test_event_freeze_and_single_query_binding(completed, tmp_path, mutation):
    folder, cfg, result = completed
    fixture = json.loads((folder / "fixture.json").read_text())
    models = json.loads((folder / "models.json").read_text())["pipelines"]
    freeze = json.loads((folder / "freeze.json").read_text())
    rows = [json.loads(row) for row in (folder / "events.jsonl").read_text().splitlines()]
    (tmp_path / "primary_start.json").write_bytes((folder / "primary_start.json").read_bytes())
    if mutation == "query_before_freeze":
        rows[6], rows[7] = rows[7], rows[6]
    elif mutation == "double_query":
        rows.insert(8, deepcopy(rows[7]))
    elif mutation == "sequence":
        rows[3]["sequence"] = 1
    elif mutation == "elapsed":
        rows[-1]["utc"] = "2026-09-10T01:15:01+00:00"
    elif mutation == "model_hash":
        freeze["models_sha256"] = "0" * 64
    elif mutation == "count":
        rows[5]["completed_updates"] = 1599
    else:
        rows[-1]["terminal"] = "MADE_UP_PASS"
    (tmp_path / "events.jsonl").write_text("\n".join(map(json.dumps, rows)))
    with pytest.raises(ValueError):
        cold._chronology(
            tmp_path,
            result,
            fixture,
            models,
            freeze,
            {name: pin(folder / name) for name in cold.ARTIFACTS},
            cfg,
            cold._Checks(1e-8),
        )


@pytest.mark.parametrize("mutation", ["object", "shape", "extra", "float32", "fortran"])
def test_npz_headers_reject_before_numeric_coercion(tmp_path, mutation):
    value = np.ones((2, 3), dtype=float)
    if mutation == "object":
        value = value.astype(object)
    elif mutation == "shape":
        value = np.ones((100, 2))
    elif mutation == "float32":
        value = value.astype(np.float32)
    elif mutation == "fortran":
        value = np.asfortranarray(value)
    arrays = {"values": value}
    if mutation == "extra":
        arrays["unexpected"] = value
    save_npz(tmp_path / "toy.npz", arrays)
    with pytest.raises(ValueError):
        cold._arrays(tmp_path / "toy.npz", {"values": (2, 3)})


def synthetic_scores(groups=4):
    """Small score-only fixtures for decision gates, not generated efficacy data."""
    scores = np.zeros((2, 2 * groups, 10, 48, 12))
    oracle = np.zeros((2, 2 * groups, 48, 12))
    for regime in range(2):
        for person in range(2 * groups):
            for arm in range(10):
                successes = 24 if regime or arm != 5 else 30
                prediction = (np.arange(48) + (np.arange(48) >= successes)) % 12
                scores[regime, person, arm, np.arange(48), prediction] = 1
            prediction = (np.arange(48) + (np.arange(48) >= 36)) % 12
            oracle[regime, person, np.arange(48), prediction] = 1
    return scores, oracle


def test_summary_manual_counts_interaction_and_negative_result_precedence():
    cfg = config()
    cfg["evaluation_groups"] = 4
    scores, oracle = synthetic_scores()
    validity = {
        "finite_arrays": True,
        "common_q_hashes": True,
        "constant_support_geometry": True,
        "donors_cross_group": True,
    }
    actuation = {
        "coupled": {
            key: 1
            for key in (
                "coefficient_max_abs",
                "gradient_max",
                "max_abs_R",
                "max_abs_score",
                "argmax_changed",
            )
        }
    }
    summary = cold._summary(scores, oracle, cfg, validity, actuation)
    assert summary["terminal"] == "GENERATED_M_CAPACITY_PASS"
    assert summary["contrasts"]["coupled_QM_minus_Q"] == {
        "mean_pp": 12.5,
        "mcse_pp": 0,
        "low_pp": 12.5,
        "high_pp": 12.5,
        "n_groups": 4,
    }
    assert summary["contrasts"]["null_QM_minus_Q"]["mean_pp"] == 0
    assert summary["contrasts"]["interaction"]["mean_pp"] == 12.5
    validity["finite_arrays"] = False
    assert (
        cold._summary(scores, oracle, cfg, validity, actuation)["terminal"]
        == "EFFICACY_NOT_EVALUATED"
    )
    validity["finite_arrays"] = True
    assert (
        cold._summary(scores, scores[:, :, 3], cfg, validity, actuation)["terminal"]
        == "POSITIVE_CONTROL_NOT_QUALIFIED"
    )
    negative = scores.copy()
    negative[1, :, 5] = scores[0, :, 5]
    assert (
        cold._summary(negative, oracle, cfg, validity, actuation)["terminal"]
        == "NEGATIVE_CONTROL_NOT_QUALIFIED"
    )
    actuation["coupled"]["gradient_max"] = 0
    assert (
        cold._summary(scores, oracle, cfg, validity, actuation)["terminal"]
        == "LEARNER_CAPACITY_NOT_ESTABLISHED"
    )


def test_group_interval_uses_both_members_and_sample_sd():
    cfg = config()
    cfg["evaluation_groups"] = 4
    scores, oracle = synthetic_scores()
    # Only member1/group0 loses one success: the group contrast changes by 100/96.
    scores[0, 4, 5, 0] = 0
    scores[0, 4, 5, 0, 1] = 1
    actuation = {
        "coupled": {
            key: 1
            for key in (
                "coefficient_max_abs",
                "gradient_max",
                "max_abs_R",
                "max_abs_score",
                "argmax_changed",
            )
        }
    }
    result = cold._summary(scores, oracle, cfg, {"finite_arrays": True}, actuation)
    actual = result["contrasts"]["coupled_QM_minus_Q"]
    values = np.array([12.5 - 100 / 96, 12.5, 12.5, 12.5])
    se = values.std(ddof=1) / 2
    assert actual["mean_pp"] == pytest.approx(values.mean())
    assert actual["mcse_pp"] == pytest.approx(se)
    assert actual["low_pp"] == pytest.approx(values.mean() - stats.t.ppf(0.975, 3) * se)


def test_n1_original_tau_c_normalization_and_guard_failures():
    s = np.broadcast_to(1536 * np.ones((8, 8)), (5, 12, 8, 8)).copy()
    c = np.broadcast_to(768 * (np.ones((8, 8)) + 0.04 * np.eye(8)), s.shape).copy()
    r = np.broadcast_to(np.array([0.9] * 4 + [1.1] * 4), (5, 8))
    f = cold._projectors(s, c, r)
    metric = c[0, 0]
    denominator = metric + np.diag(0.1 * np.linalg.eigvalsh(metric)[0] / (16 / 9) * r[0])
    vector = np.linalg.solve(denominator, np.ones(8))
    expected = np.outer(vector, vector) / (vector @ metric @ vector)
    np.testing.assert_allclose(f, np.broadcast_to(expected, f.shape), atol=1e-15, rtol=0)
    s[0, 0, 0, 1] += 0.1
    with pytest.raises(ValueError, match="symmetry"):
        cold._projectors(s, c, r)
    s[0, 0, 0, 1] -= 0.1
    c[0, 0] = 0
    with pytest.raises(ValueError, match="SPD"):
        cold._projectors(s, c, r)
    with pytest.raises(ValueError, match="degenerate"):
        cold._projectors(np.zeros_like(s), np.broadcast_to(np.eye(8), s.shape).copy(), r)


def test_no_producer_or_torch_imports_in_audit_module():
    import ast

    tree = ast.parse(Path(cold.__file__).read_text())
    imports = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            imports.extend((node.module or "") + "." + alias.name for alias in node.names)
    assert not any(
        any(
            word in name
            for word in (
                "torch",
                "efficacy_fixture",
                "efficacy_run",
                "n1_learning",
                "n1_evaluation",
            )
        )
        for name in imports
    )


def test_registered_seed_cannot_use_reduced_grid():
    cfg = config()
    cfg["seed"] = 20260916  # Validate config only: absolutely no draw or run.
    with pytest.raises(ValueError, match="complete frozen group grid"):
        cold._config(cfg)


@pytest.fixture(scope="module")
def replay_inputs(completed):
    folder, cfg, _ = completed
    shapes = cold._shapes(cfg)
    data = cold._arrays(folder / "support.npz", shapes["support.npz"])
    query = cold._arrays(folder / "query.npz", shapes["query.npz"])
    q, m, s, c, _ = cold._case_features(data, cfg, cold._Checks(1e-8), "toy")
    models = json.loads((folder / "models.json").read_text())["pipelines"]
    saved = cold._arrays(folder / "evaluation.npz", shapes["evaluation.npz"])
    return data, query, q, m, s, c, models, saved, cfg


@pytest.mark.parametrize(
    "field", ["r", "projectors", "scores", "oracle_r", "oracle_projectors", "oracle_scores"]
)
def test_saved_all_arm_and_oracle_mutation_is_detected(replay_inputs, field):
    data, query, q, m, s, c, models, original, cfg = replay_inputs
    saved = dict(original)
    saved[field] = original[field].copy()
    saved[field].flat[0] += 1e-4
    with pytest.raises(ValueError, match="values differ"):
        cold._replay(data, query, q, m, s, c, models, saved, cfg, cold._Checks(1e-8))


def test_argmax_is_checked_separately_from_absolute_score_tolerance(replay_inputs):
    data, query, q, m, s, c, models, original, cfg = replay_inputs
    saved = dict(original)
    saved["scores"] = original["scores"].copy()
    # Isolate the discrete argmax contract: with a permissive comparison object,
    # allclose is insufficient and the exact decision assertion still rejects.
    saved["scores"][0, 0, 0, 0] = -100
    saved["scores"][0, 0, 0, 0, (int(original["scores"][0, 0, 0, 0].argmax()) + 1) % 12] = 100
    with pytest.raises(ValueError, match="all_arm_argmax"):
        cold._replay(data, query, q, m, s, c, models, saved, cfg, cold._Checks(1000))


def test_no_human_config_or_nonfinite_tolerance_override():
    for key, value in (
        ("numeric_atol", 1e-4),
        ("sampling_rate", 100),
        ("oracle_coefficients", [4, 0, 0]),
    ):
        cfg = config()
        cfg[key] = value
        with pytest.raises(ValueError, match="configuration mismatch"):
            cold._config(cfg)


def test_json_duplicate_keys_and_nonfinite_constants(tmp_path):
    path = tmp_path / "toy.json"
    for value in ('{"x": 1, "x": 2}', '{"x": NaN}'):
        path.write_text(value)
        with pytest.raises(ValueError):
            cold._json(path)
