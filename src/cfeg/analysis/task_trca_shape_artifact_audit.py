"""Independent audit of persisted source/evaluation artifacts.

NumPy/SciPy and the independent audit implementation only. No training/operator
producer imports and no raw EEG/archive access. This verifies downstream of the
recorded sufficient statistics, not native preprocessing or full Adam replay.
"""
from __future__ import annotations
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.special import logsumexp

from cfeg.analysis import task_trca_shape_audit as independent

ARMS = ("FULL", "ISO", "Q", "Q2", "QM", "SHAM_REFIT", "PERMUTED", "STALE", "MISSING")
STATS = ("query_gram", "template_gram", "cross_gram", "query_mean", "template_mean")
LAMBDAS = (.0001, .001, .01)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def require(value, message):
    if not value:
        raise ValueError("AUDIT_FAILURE: " + message)


def donor_positions(data, positions):
    groups = {}
    for j in positions:
        pid, interface, samples, k = data["keys"][j]
        mask = np.isfinite(data["packet5"][j, :k])
        key = int(interface), int(samples), int(k), int(data["orders"][j]), mask.tobytes()
        groups.setdefault(key, []).append(j)
    donors = {}
    for group in groups.values():
        group = sorted(group, key=lambda j: data["keys"][j, 0])
        donors.update({j: group[(i + 1) % len(group)] for i, j in enumerate(group)})
    return donors


def stale_features(packet):
    first = packet[0]
    available = np.isfinite(first)
    require(not np.isinf(first).any() and (first[available] >= 0).all(), "invalid stale M")
    m = np.zeros((8, 2))
    if available.any():
        logs = np.log1p(first[available])
        m[available, 0] = logs - logs.mean()
    return m, available


def prior(data, j, pipeline, arm, donors):
    donor_m = data["m"][donors[j]] if arm in ("SHAM_REFIT", "PERMUTED") else None
    stale_m, stale_available = stale_features(data["packet5"][j]) if arm == "STALE" else (None, None)
    return independent.independent_prior(
        data["q"][j], data["m"][j], data["available"][j], pipeline, arm,
        donor_m=donor_m, stale_m=stale_m, stale_available=stale_available,
    )


def scores(data, j, pipeline, arm, donors):
    r = prior(data, j, pipeline, arm, donors)
    w = independent.independent_filters(data["s"][j], data["c"][j], data["anchors"][j], r)
    stats = {name: data[name][j] for name in STATS}
    stats["samples"] = int(data["keys"][j, 2])
    return independent.independent_scores(w, stats, data["weights"][j]), r, w


def ce(values, weights):
    scaled = values / np.sum(weights) / .1
    truth = np.tile(np.arange(12), len(values) // 12)
    return float(np.mean(logsumexp(scaled, axis=-1) - scaled[np.arange(len(truth)), truth]))


def check_record(data, pipeline, ids):
    require(pipeline["fit_ids"] == list(ids), "pipeline fit IDs")
    positions = np.flatnonzero(np.isin(data["keys"][:, 0], ids))
    availability = data["available"][positions]
    q = data["q"][positions]
    for name, values, mask in (
        ("q_scaler", q, np.ones(q.shape[:-1], dtype=bool)),
        ("m_scaler", data["m"][positions], availability),
        ("q2_scaler", q[..., 1:3], np.broadcast_to(availability[:, None], q.shape[:-1])),
    ):
        rows = values[mask]
        mean = rows.mean(0) if len(rows) else np.zeros(values.shape[-1])
        scale = rows.std(0, ddof=0) if len(rows) else np.ones(values.shape[-1])
        scale[scale < 1e-12] = 1
        require(pipeline[name]["fit_ids"] == list(ids), "scaler fit IDs")
        np.testing.assert_allclose(pipeline[name]["mean"], mean, atol=1e-12, rtol=1e-12)
        np.testing.assert_allclose(pipeline[name]["scale"], scale, atol=1e-12, rtol=1e-12)
    actual = {tuple(d["case"]): d["donor_id"] for d in pipeline["donors"]}
    expected = {tuple(data["keys"][j]): int(data["keys"][donor, 0]) for j, donor in donor_positions(data, positions).items()}
    require(actual == expected, "fit donor partition/map")
    for head in (pipeline["Q"], *pipeline["residuals"].values()):
        require(head["steps"] == 200 and len(head["trace"]) == 200, "fixed optimizer length")
        require([r["step"] for r in head["trace"]] == list(range(1, 201)), "optimizer step indices")
        require(np.isfinite(head["coefficients"]).all(), "coefficients")
        require(np.isfinite([[r["loss_before_step"], r["ce_before_step"], r["gradient_norm"]] for r in head["trace"]]).all(), "trace finiteness")
    return positions


def audit_training(data, model, outer_ids, evaluation_ids):
    require(sorted(set(data["keys"][:, 0].tolist())) == list(outer_ids), "source participant roles")
    require(not set(outer_ids) & set(evaluation_ids), "outer role overlap")
    require(len(data["keys"]) == len(outer_ids) * 16, "complete source16case grid")
    selection = model["selection"]
    require(selection["outer_evaluation_ids"] == list(evaluation_ids), "selection evaluation IDs")
    require(len(selection["inner"]) == 9, "nine inner fits")
    means, errors = {}, []
    for row in selection["inner"]:
        inner = int(row["inner_fold"])
        require(inner in (0, 1, 2) and row["lambda"] in LAMBDAS, "inner grid")
        validation_ids = list(outer_ids[inner::3])
        training_ids = [p for p in outer_ids if p not in validation_ids]
        require(row["validation_ids"] == validation_ids and row["fit_ids"] == training_ids, "inner role split")
        pipeline = row["pipeline"]
        require(pipeline["lambda"] == row["lambda"], "lambda fit identity")
        check_record(data, pipeline, training_ids)
        positions = np.flatnonzero(np.isin(data["keys"][:, 0], validation_ids))
        donors = donor_positions(data, positions)
        losses = {}
        for arm in ("Q", "Q2", "QM", "SHAM_REFIT"):
            loss = float(np.mean([ce(scores(data, j, pipeline, arm, donors)[0], data["weights"][j]) for j in positions]))
            error = abs(loss - row["validation_ce"][arm])
            require(error <= 1e-8, "independent validation CE differs")
            errors.append(error)
            losses[arm] = loss
        means.setdefault(row["lambda"], []).append((inner, losses["Q"], len(validation_ids)))
    weighted = {}
    for value in LAMBDAS:
        rows = means[value]
        require(sorted(v[0] for v in rows) == [0, 1, 2], "unique inner lambda grid")
        weighted[value] = float(np.average([v[1] for v in rows], weights=[v[2] for v in rows]))
    selected = max(value for value, loss in weighted.items() if loss <= min(weighted.values()) + 1e-12)
    require(selected == selection["selected_lambda"] == model["pipeline"]["lambda"], "Q-only lambda selection")
    check_record(data, model["pipeline"], outer_ids)
    return {"status": "SOURCE_SELECTION_AUDIT_PASS", "selected_lambda": selected,
            "max_validation_ce_abs_error": max(errors), "validation_ce_comparisons": len(errors),
            "scope": "scalers/donors/200-step traces and independent inner-validation scores; not full independent Adam replay"}


def audit_evaluation(data, pipeline):
    positions = range(len(data["keys"]))
    donors = donor_positions(data, positions)
    errors = {"r": 0., "filters": 0., "scores": 0.}
    for j in positions:
        for index, arm in enumerate(ARMS[1:], start=1):
            actual, r, w = scores(data, j, pipeline, arm, donors)
            for name, left, right, tolerance in (
                ("r", r, data["r"][j, index - 1], 1e-10),
                ("filters", w, data["filters"][j, index], 1e-8),
                ("scores", actual, data["scores"][j, index], 1e-8),
            ):
                error = float(np.max(np.abs(left - right)))
                require(error <= tolerance, "independent evaluation " + name)
                errors[name] = max(errors[name], error)
            np.testing.assert_array_equal(actual.argmax(-1), data["scores"][j, index].argmax(-1))
        full = np.einsum("nbc,b->nc", data["native_full_correlations"][j], data["weights"][j])
        np.testing.assert_array_equal(full, data["scores"][j, 0])
        np.testing.assert_allclose(data["native_full_correlations"][j], data["cached_full_correlations"][j], atol=1e-9, rtol=0)
        np.testing.assert_array_equal(data["scores"][j, 2], data["scores"][j, 8])
    return {"status": "EVALUATION_AUDIT_PASS", "cases": len(data["keys"]), "max_abs_errors": errors,
            "positive_argmax_exact": True, "missing_exact_q": True, "full_native_compatibility": True,
            "scope": "independent downstream sufficient-statistic reconstruction; support feature/raw preprocessing covered by separate construction tests"}
