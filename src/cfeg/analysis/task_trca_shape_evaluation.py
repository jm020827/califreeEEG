"""Label-free evaluation of frozen task-shape models from role-limited arrays.

This module has no data reader. Evaluation never fabricates a source block5 or
passes an evaluation participant to the source-task builder.
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import torch

from cfeg.analysis import metadata_trca_prior as native
from cfeg.analysis import task_trca_shape_features as features
from cfeg.analysis import task_trca_shape_learning as learn
from cfeg.analysis import task_trca_shape_operator as op

ARMS = ("FULL", "ISO", "Q", "Q2", "QM", "SHAM_REFIT", "PERMUTED", "STALE", "MISSING")
STAT_NAMES = ("query_gram", "template_gram", "cross_gram", "query_mean", "template_mean")


@dataclass(frozen=True)
class SupportState:
    participant_id: int
    interface: int
    order: int
    k: int
    samples: int
    q: np.ndarray
    m: np.ndarray
    available: np.ndarray
    mask: np.ndarray
    packet: np.ndarray
    s: torch.Tensor
    c: torch.Tensor
    anchors: torch.Tensor
    weights: torch.Tensor
    model: object

    @property
    def condition(self):
        return self.interface, self.samples, self.k

    @property
    def key(self):
        return self.participant_id, *self.condition


def support_state(pid, interface, order, support, packet, frequencies, weights):
    # Q boundary only sees the exact k-prefix boolean mask, not numeric M.
    mask = np.isfinite(packet)
    q = features.support_q_mask(support, mask, interface, order, frequencies)
    m, available = features.metadata_features(packet)
    model = native.fit_trca(support, np.ones((5, 8)), 0.0)
    pairs = [[native.trca_matrices(support[:, label, band]) for label in range(12)] for band in range(5)]
    s = np.stack([[p[0] for p in band] for band in pairs])
    c = np.stack([[p[1] for p in band] for band in pairs])
    tensor = lambda v: torch.tensor(np.array(v, copy=True), dtype=torch.float64)
    weights = np.asarray(weights)
    if weights.shape != (5,) or not np.isfinite(weights).all() or (weights <= 0).any():
        raise ValueError("Explicit native positive weights required")
    return SupportState(pid, interface, order, len(support), support.shape[-1], q, m, available,
                        mask, packet, tensor(s), tensor(c), tensor(model.filters.transpose(0, 2, 1)),
                        tensor(weights), model)


def frozen_prior(pipeline, state, arm, *, donor=None):
    if arm not in ARMS[1:]:
        raise ValueError("Positive-mass arm required")
    if state.participant_id in pipeline.fit_ids:
        raise PermissionError("Evaluation participant overlaps frozen fit")
    tensor = lambda v: torch.tensor(np.array(v, copy=True), dtype=torch.float64)
    qx = pipeline.q_scaler.transform(state.q)
    qlog = learn._head(tensor(qx), tensor(pipeline.q.coefficients), .8)
    if arm == "ISO":
        qlog = torch.zeros_like(qlog)
    rlog = torch.zeros_like(qlog)
    if arm not in ("ISO", "Q", "MISSING"):
        available = state.available
        if arm == "Q2":
            values = pipeline.q2_scaler.transform(state.q[..., 1:3], np.broadcast_to(available, (5, 8)))
            coef = pipeline.residuals[arm].coefficients
        else:
            source = state
            if arm in ("SHAM_REFIT", "PERMUTED"):
                if donor is None or donor.condition != state.condition or donor.order != state.order or not np.array_equal(donor.mask, state.mask):
                    raise ValueError("Partition-local matched donor required")
                source = donor
            raw = source.m
            if arm == "STALE":
                raw, available = features.metadata_features(np.repeat(state.packet[:1], state.k, 0))
            values = pipeline.m_scaler.transform(raw, available)
            coef = pipeline.residuals["SHAM_REFIT" if arm == "SHAM_REFIT" else "QM"].coefficients
        rlog = learn._head(tensor(values), tensor(coef), .2) * tensor(available)
    return op.shape_prior(qlog + rlog)


def evaluate(pipeline, state, query, *, donor=None):
    if state.participant_id in pipeline.fit_ids:
        raise PermissionError("Evaluation participant overlaps frozen fit")
    if np.asarray(query).shape != (48, 5, 8, state.samples):
        raise ValueError("Exactly four twelve-class query blocks required")
    tensor = lambda v: torch.tensor(np.array(v, copy=True), dtype=torch.float64)
    stats = op.gram_statistics(tensor(state.model.templates), tensor(query))
    scores, correlations = native.score_trca(state.model, query, state.weights.numpy())
    all_scores, priors, filters = [scores], [], [state.model.filters]
    with torch.no_grad():
        for arm in ARMS[1:]:
            r = frozen_prior(pipeline, state, arm, donor=donor)
            w = op.bounded_filters(state.s, state.c, state.anchors, r[:, None, :]).transpose(-1, -2)
            scored, _ = op.score_gram(w, stats, state.weights)
            priors.append(r.numpy())
            filters.append(w.numpy())
            all_scores.append(scored.numpy())
    np.testing.assert_array_equal(all_scores[ARMS.index("Q")], all_scores[ARMS.index("MISSING")])
    return {
        "scores": np.stack(all_scores), "r": np.stack(priors), "filters": np.stack(filters),
        "native_full_correlations": correlations,
        **{name: getattr(stats, name).numpy() for name in STAT_NAMES},
    }


def pipeline_from_record(record):
    """Restore an already frozen record; never fits a scaler or head."""
    def scaler(name):
        value = record[name]
        return features.FeatureScaler(learn._readonly(value["mean"]), learn._readonly(value["scale"]), tuple(value["fit_ids"]))

    def head(value):
        return learn.HeadFit(learn._readonly(value["coefficients"]), value["initial_loss"], value["final_loss"], tuple(value["trace"]))

    result = learn.Pipeline(tuple(record["fit_ids"]), record["lambda"], scaler("q_scaler"), scaler("m_scaler"), scaler("q2_scaler"),
                            head(record["Q"]), {a: head(v) for a, v in record["residuals"].items()},
                            {tuple(d["case"]): d["donor_id"] for d in record["donors"]})
    if result.q_hash != record["q_hash"]:
        raise ValueError("Frozen Q hash mismatch")
    return result


def pack_cases(cases):
    """Portable sufficient statistics for independent fit/validation audit."""
    result = {"keys": np.array([c.key for c in cases], dtype=np.int64),
              "orders": np.array([c.order for c in cases], dtype=np.int64)}
    for name in ("q", "m", "available", "s", "c", "anchors", "weights", "labels"):
        result[name] = np.stack([getattr(c, name).detach().cpu().numpy() if isinstance(getattr(c, name), torch.Tensor) else getattr(c, name) for c in cases])
    for name in STAT_NAMES:
        result[name] = np.stack([getattr(c.statistics, name).detach().cpu().numpy() for c in cases])
    result["packet5"] = np.stack([np.pad(c.packet, ((0, 5 - c.k), (0, 0)), constant_values=np.nan) for c in cases])
    return result
