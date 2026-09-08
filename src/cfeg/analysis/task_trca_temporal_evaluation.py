"""Label-free in-memory evaluation for the NEW temporal-score study schema.

No archive reader, execution authorization, query labels or source block5 input.
Native FULL and centered FULL use identical native support filters/templates.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np
import torch

from cfeg.analysis import metadata_trca_prior as native
from cfeg.analysis import task_trca_shape_features as features
from cfeg.analysis import task_trca_shape_operator as old_operator
from cfeg.analysis import task_trca_shape_signfree as temporal
from cfeg.analysis import task_trca_temporal_learning as learn

SCHEMA = "task-trca-temporal-v1"
SCORE_SCHEMA = "component-time-centered-ensemble-pearson-v1"
ARMS = (
    "FULL_NATIVE",
    "FULL_CENTERED",
    "ISO",
    "Q",
    "Q2",
    "QM",
    "SHAM_REFIT",
    "PERMUTED",
    "STALE",
    "MISSING",
)
POSITIVE_ARMS = ARMS[2:]
STAT_NAMES = ("query_gram", "template_gram", "cross_gram")


def _tensor(value):
    return torch.tensor(np.array(value, copy=True), dtype=torch.float64)


def _numeric(value, name, *, finite=True):
    array = np.asarray(value)
    if array.dtype.kind not in "iuf":
        raise ValueError(f"{name} must be real numeric")
    array = np.asarray(array, dtype=np.float64)
    if finite and not np.isfinite(array).all():
        raise ValueError(f"{name} must be finite")
    return array


def _pipeline(pipeline):
    if not isinstance(pipeline, learn.Pipeline):
        raise TypeError("Temporal Pipeline required; legacy trained objectives cannot be reused")
    record = pipeline.record()
    if record.get("schema") != SCHEMA or record.get("score_schema") != SCORE_SCHEMA:
        raise ValueError("Wrong temporal pipeline schema/scorer")


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
    """Build only from already permitted support arrays; never from query labels."""
    if type(pid) is not int or pid <= 0:
        raise ValueError("Participant ID must be a positive integer")
    if type(interface) is not int or interface not in (0, 1):
        raise ValueError("Interface must be0/1")
    if type(order) is not int or order not in (0, 1):
        raise ValueError("Order must be0/1")
    x, packet = _numeric(support, "support"), _numeric(packet, "packet", finite=False)
    mask = np.isfinite(packet)
    q = features.support_q_mask(x, mask, interface, order, frequencies)
    m, available = features.metadata_features(packet)
    weights = _numeric(weights, "weights")
    if weights.shape != (5,) or np.any(weights <= 0):
        raise ValueError("Positive native weights[5] required")
    model = native.fit_trca(x, np.ones((5, 8)), 0)
    pairs = [[native.trca_matrices(x[:, cls, b]) for cls in range(12)] for b in range(5)]
    s = np.array([[p[0] for p in band] for band in pairs])
    c = np.array([[p[1] for p in band] for band in pairs])
    return SupportState(
        pid,
        interface,
        order,
        len(x),
        x.shape[-1],
        learn._readonly(q),
        learn._readonly(m),
        learn._readonly(available, bool),
        learn._readonly(mask, bool),
        learn._readonly(packet),
        _tensor(s),
        _tensor(c),
        _tensor(model.filters.transpose(0, 2, 1)),
        _tensor(weights),
        model,
    )


@dataclass(frozen=True)
class EvaluationPartition:
    """States bound to externally frozen IDs/grid, not inferred from a subset.

    The caller must obtain these expected IDs/conditions from its frozen design;
    this in-memory contract does not authenticate external execution authority.
    """

    states: tuple[SupportState, ...]
    evaluation_ids: tuple[int, ...]
    conditions: tuple[tuple[int, int, int], ...]


def _partition(pipeline, state, partition):
    """Validate the explicit common evaluation grid and deterministic donor map.

    This does not authorize dataset access: it validates caller-provided states.
    Object identity binds the supplied target/donor to this exact partition.
    """
    _pipeline(pipeline)
    if not isinstance(partition, EvaluationPartition):
        raise TypeError("Explicit frozen EvaluationPartition required")
    states = tuple(partition.states)
    if not states or any(not isinstance(s, SupportState) for s in states):
        raise ValueError("Nonempty evaluation SupportState partition required")
    lookup = {s.key: s for s in states}
    if len(lookup) != len(states):
        raise ValueError("Duplicate evaluation case")
    if not isinstance(state, SupportState) or lookup.get(state.key) is not state:
        raise ValueError("Target is not the supplied evaluation partition state")
    ids = sorted({s.participant_id for s in states})
    expected_ids = partition.evaluation_ids
    expected_conditions = partition.conditions
    if (
        not expected_ids
        or tuple(sorted(set(expected_ids))) != expected_ids
        or any(type(v) is not int or v <= 0 for v in expected_ids)
        or not expected_conditions
        or tuple(sorted(set(expected_conditions))) != expected_conditions
    ):
        raise ValueError("Frozen evaluation IDs/conditions must be nonempty sorted unique tuples")
    expected_keys = {(pid, *condition) for pid in expected_ids for condition in expected_conditions}
    if set(lookup) != expected_keys:
        raise PermissionError("Evaluation states differ from frozen expected IDs/condition grid")
    if set(ids) & set(pipeline.fit_ids):
        raise PermissionError("Evaluation partition overlaps frozen fit IDs")
    grids = [{s.condition for s in states if s.participant_id == pid} for pid in ids]
    if any(g != grids[0] for g in grids):
        raise ValueError("Evaluation partition requires a complete common grid")
    identities = {}
    for s in states:
        key = s.participant_id, s.interface, s.k
        if key in identities:
            previous = identities[key]
            if s.order != previous.order or not np.array_equal(
                s.packet, previous.packet, equal_nan=True
            ):
                raise ValueError("Evaluation metadata differs across windows")
        identities[key] = s
    rows = sorted((s for s in states if s.condition == state.condition), key=lambda s: s.key)
    mapping = features.donor_map(
        [s.participant_id for s in rows],
        np.stack([s.mask for s in rows]),
        [s.order for s in rows],
        state.interface,
    )
    expected = lookup[(mapping[state.participant_id], *state.condition)]
    return expected


def frozen_prior(pipeline, state, arm, *, partition, donor=None):
    if arm not in POSITIVE_ARMS:
        raise ValueError("Positive-mass temporal arm required")
    expected = _partition(pipeline, state, partition)
    if donor is not None and donor is not expected:
        raise PermissionError("Donor is not the deterministic evaluation-partition donor")
    qx = pipeline.q_scaler.transform(state.q)
    qlog = learn._head(_tensor(qx), _tensor(pipeline.q.coefficients), 0.8)
    if arm == "ISO":
        qlog = torch.zeros_like(qlog)
    rlog = torch.zeros_like(qlog)
    if arm not in ("ISO", "Q", "MISSING"):
        available = state.available
        if arm == "Q2":
            values = pipeline.q2_scaler.transform(
                state.q[..., 1:3], np.broadcast_to(available, (5, 8))
            )
            coefficients = pipeline.residuals["Q2"].coefficients
        else:
            source = expected if arm in ("SHAM_REFIT", "PERMUTED") else state
            raw = source.m
            if arm == "STALE":
                raw, available = features.metadata_features(np.repeat(state.packet[:1], state.k, 0))
            values = pipeline.m_scaler.transform(raw, available)
            coefficients = pipeline.residuals[
                "SHAM_REFIT" if arm == "SHAM_REFIT" else "QM"
            ].coefficients
        rlog = learn._head(_tensor(values), _tensor(coefficients), 0.2) * _tensor(available)
    return old_operator.shape_prior(qlog + rlog)


def evaluate(pipeline, state, query, *, partition, donor=None):
    """Return10 arms, no query labels; validate roles before coercing query."""
    expected = _partition(pipeline, state, partition)
    if donor is not None and donor is not expected:
        raise PermissionError("Donor is not the deterministic evaluation-partition donor")
    query = _numeric(query, "query")
    if query.shape != (48, 5, 8, state.samples):
        raise ValueError("Exactly four twelve-class query blocks required")
    templates, x = _tensor(state.model.templates), _tensor(query)
    temporal_stats = temporal.temporal_statistics(templates, x)
    native_stats = old_operator.gram_statistics(templates, x)
    native_scores, native_corr = native.score_trca(state.model, query, state.weights.numpy())
    native_w = _tensor(state.model.filters.transpose(0, 2, 1))
    full_f = native_w.unsqueeze(-1) * native_w.unsqueeze(-2)
    centered_scores, _ = temporal.score_temporal_gram(full_f, temporal_stats, state.weights)
    scores, priors, projectors = [native_scores, centered_scores.numpy()], [], []
    with torch.no_grad():
        for arm in POSITIVE_ARMS:
            r = frozen_prior(pipeline, state, arm, partition=partition, donor=expected)
            f = temporal.bounded_projectors(state.s, state.c, r[:, None, :])
            value, _ = temporal.score_temporal_gram(f, temporal_stats, state.weights)
            priors.append(r.numpy())
            projectors.append(f.numpy())
            scores.append(value.numpy())
    np.testing.assert_array_equal(scores[ARMS.index("Q")], scores[ARMS.index("MISSING")])
    f, r = np.stack(projectors), np.stack(priors)
    qidx, midx = POSITIVE_ARMS.index("Q"), POSITIVE_ARMS.index("QM")
    dq = scores[ARMS.index("QM")] - scores[ARMS.index("Q")]
    return {
        "schema": SCHEMA,
        "score_schema": SCORE_SCHEMA,
        "arms": ARMS,
        "scores": np.stack(scores),
        "r": r,
        "projectors": f,
        "native_filters": state.model.filters,
        "native_full_correlations": native_corr,
        "statistics": {
            **{n: getattr(temporal_stats, n).numpy() for n in STAT_NAMES},
            "samples": state.samples,
        },
        "native_statistics": {
            **{
                n: getattr(native_stats, n).numpy()
                for n in (*STAT_NAMES, "query_mean", "template_mean")
            },
            "samples": state.samples,
        },
        "actuation_QM_minus_Q": {
            "max_abs_R": float(np.max(np.abs(r[midx] - r[qidx]))),
            "max_abs_F": float(np.max(np.abs(f[midx] - f[qidx]))),
            "max_abs_J": float(np.max(np.abs(f[midx].sum(1) - f[qidx].sum(1)))),
            "max_abs_score": float(np.max(np.abs(dq))),
            "argmax_changed": int(
                np.count_nonzero(
                    scores[ARMS.index("QM")].argmax(-1) != scores[ARMS.index("Q")].argmax(-1)
                )
            ),
        },
    }


def pipeline_from_record(record):
    """Restore only new-schema frozen records; no fitting or old objective coercion."""
    if not isinstance(record, Mapping):
        raise TypeError("Temporal pipeline record required")
    if record.get("schema") != SCHEMA or record.get("score_schema") != SCORE_SCHEMA:
        raise ValueError("Legacy/wrong temporal pipeline schema")
    ids = record.get("fit_ids")
    if not isinstance(ids, list) or not ids or any(type(v) is not int or v <= 0 for v in ids):
        raise ValueError("Sorted unique positive fit IDs required")
    if ids != sorted(set(ids)) or record.get("lambda") not in (0.0001, 0.001, 0.01):
        raise ValueError("Invalid fit IDs/lambda")

    def scaler(name, dimensions):
        value = record[name]
        mean, scale = _numeric(value["mean"], "mean"), _numeric(value["scale"], "scale")
        if mean.shape != (dimensions,) or scale.shape != (dimensions,) or (scale <= 0).any():
            raise ValueError("Invalid scaler dimensions/scale")
        if value["fit_ids"] != ids:
            raise ValueError("Scaler fit IDs differ")
        return features.FeatureScaler(learn._readonly(mean), learn._readonly(scale), tuple(ids))

    def head(value, dimensions):
        coefficient = _numeric(value["coefficients"], "coefficients")
        trace = value["trace"]
        if coefficient.shape != (dimensions + 1,) or value.get("steps") != 200 or len(trace) != 200:
            raise ValueError("Invalid fixed head size/steps")
        if [v["step"] for v in trace] != list(range(1, 201)):
            raise ValueError("Invalid optimizer step sequence")
        _numeric(
            [
                [v[n] for n in ("loss_before_step", "ce_before_step", "gradient_norm")]
                for v in trace
            ],
            "trace",
        )
        _numeric([value["initial_loss"], value["final_loss"]], "loss")
        return learn.HeadFit(
            learn._readonly(coefficient),
            value["initial_loss"],
            value["final_loss"],
            tuple(dict(v) for v in trace),
        )

    if set(record["residuals"]) != {"Q2", "QM", "SHAM_REFIT"}:
        raise ValueError("Exact three residual heads required")
    donors = {tuple(d["case"]): d["donor_id"] for d in record["donors"]}
    if len(donors) != len(record["donors"]):
        raise ValueError("Duplicate donor case")
    if not donors or any(len(k) != 4 or k[0] not in ids or v not in ids for k, v in donors.items()):
        raise ValueError("Fit donor must remain in fitting partition")
    result = learn.Pipeline(
        tuple(ids),
        record["lambda"],
        scaler("q_scaler", 15),
        scaler("m_scaler", 2),
        scaler("q2_scaler", 2),
        head(record["Q"], 15),
        {name: head(value, 2) for name, value in record["residuals"].items()},
        donors,
    )
    if result.q_hash != record.get("q_hash"):
        raise ValueError("Frozen Q hash mismatch")
    return result


def pack_cases(cases):
    """Detached temporal sufficient statistics for independent generated-file audit."""
    cases = tuple(sorted(cases, key=lambda c: c.key))
    if not cases or any(not isinstance(c, learn.TaskCase) for c in cases):
        raise ValueError("New temporal TaskCase sequence required")
    if any(not isinstance(c.statistics, temporal.TemporalGramStatistics) for c in cases):
        raise ValueError("Global statistics cannot be packed as temporal")
    result = {
        "schema": np.array(SCHEMA),
        "score_schema": np.array(SCORE_SCHEMA),
        "keys": np.array([c.key for c in cases], dtype=np.int64),
        "orders": np.array([c.order for c in cases], dtype=np.int64),
    }
    for name in ("q", "m", "available", "s", "c", "anchors", "weights", "labels"):
        result[name] = np.stack(
            [
                getattr(c, name).detach().cpu().numpy()
                if isinstance(getattr(c, name), torch.Tensor)
                else getattr(c, name)
                for c in cases
            ]
        )
    for name in STAT_NAMES:
        result[name] = np.stack([getattr(c.statistics, name).detach().cpu().numpy() for c in cases])
    result["packet5"] = np.stack(
        [np.pad(c.packet, ((0, 5 - c.k), (0, 0)), constant_values=np.nan) for c in cases]
    )
    return result
