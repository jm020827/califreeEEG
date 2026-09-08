"""Temporal-score frozen-Q residual learner; generated/in-memory arrays only.

Distinct model/case schemas prevent legacy globally centered scores from being
silently reused. Native anchors are stored for controls, never used in the
positive-mass score or loss. This module grants no human-data execution access.
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass, replace

import numpy as np
import torch

from cfeg.analysis import metadata_trca_prior as native
from cfeg.analysis import task_trca_shape_features as features
from cfeg.analysis import task_trca_shape_operator as operator
from cfeg.analysis import task_trca_shape_signfree as signfree

SCHEMA = "task-trca-temporal-v1"
SCORE_SCHEMA = "component-time-centered-ensemble-pearson-v1"
A = math.log(2) / 2
LAMBDAS = (0.0001, 0.001, 0.01)
STEPS = 200
RESIDUAL_ARMS = ("Q2", "QM", "SHAM_REFIT")


def _readonly(value, dtype=np.float64):
    result = np.array(value, dtype=dtype, copy=True)
    return np.frombuffer(result.tobytes(), dtype=result.dtype).reshape(result.shape)


@dataclass(frozen=True)
class TaskCase:
    participant_id: int
    interface: int
    order: int
    k: int
    samples: int
    role: str
    q: np.ndarray
    m: np.ndarray
    available: np.ndarray
    mask: np.ndarray
    packet: np.ndarray
    s: torch.Tensor
    c: torch.Tensor
    anchors: torch.Tensor
    statistics: signfree.TemporalGramStatistics
    labels: torch.Tensor
    weights: torch.Tensor

    def __post_init__(self):
        if self.role not in ("fit", "validation"):
            raise PermissionError("Task supervision only belongs to source fit/validation")
        if not isinstance(self.statistics, signfree.TemporalGramStatistics):
            raise TypeError("Temporal TaskCase rejects legacy/global statistics")
        if (
            type(self.samples) is not int
            or self.samples < 2
            or self.samples != self.statistics.samples
        ):
            raise ValueError("Case and temporal Gram sample counts must agree")

    @property
    def condition(self):
        return self.interface, self.samples, self.k

    @property
    def key(self):
        return (self.participant_id, *self.condition)


def make_task_case(
    participant_id,
    interface,
    order,
    support,
    packet,
    frequencies,
    source_block5,
    *,
    role="fit",
    labels=None,
    weights,
    device="cpu",
):
    """Construct one source/validation case from already role-limited arrays.

    Evaluation block5 is rejected before any conversion of supplied arrays.
    Actual archive-role enforcement remains the responsibility of a future reader.
    """
    if role not in ("fit", "validation"):
        raise PermissionError("Task supervision only belongs to source fit/validation")
    if type(participant_id) is not int or participant_id <= 0:
        raise ValueError("Participant ID must be a positive integer")
    if np.asarray(support).dtype.kind not in "iuf" or np.asarray(packet).dtype.kind not in "iuf":
        raise ValueError("Support and metadata must be real numeric arrays")
    x = np.asarray(support, dtype=np.float64)
    packet = np.asarray(packet, dtype=np.float64)
    mask = np.isfinite(packet)
    q = features.support_q_mask(x, mask, interface, order, frequencies)
    m, available = features.metadata_features(packet)
    if np.asarray(source_block5).dtype.kind not in "iuf":
        raise ValueError("Source block5 must be a real numeric array")
    y = np.asarray(source_block5, dtype=np.float64)
    if y.shape != x.shape[1:] or not np.isfinite(y).all():
        raise ValueError("Expected separate source block5[12,5,8,N]")
    label_values = np.arange(12) if labels is None else np.asarray(labels)
    if label_values.dtype.kind not in "iu" or sorted(label_values.tolist()) != list(range(12)):
        raise ValueError("Source block5 requires exactly one label per class")
    weight_values = np.asarray(weights)
    if weight_values.dtype.kind not in "iuf":
        raise ValueError("Explicit real native band weights are required")
    weight_values = np.asarray(weight_values, dtype=np.float64)
    if weight_values.shape != (5,) or not np.isfinite(weight_values).all():
        raise ValueError("Expected finite native weights[5]")
    if np.any(weight_values <= 0):
        raise ValueError("Task loss requires positive native band weights")
    base = native.fit_trca(x, np.ones((5, 8)), 0.0)
    pairs = [[native.trca_matrices(x[:, label, band]) for label in range(12)] for band in range(5)]
    s = np.stack([[p[0] for p in band] for band in pairs])
    c = np.stack([[p[1] for p in band] for band in pairs])

    def tensor(v):
        return torch.tensor(np.array(v, copy=True), dtype=torch.float64, device=device)

    return TaskCase(
        participant_id,
        interface,
        order,
        len(x),
        x.shape[-1],
        role,
        _readonly(q),
        _readonly(m),
        _readonly(available, bool),
        _readonly(mask, bool),
        _readonly(packet),
        tensor(s),
        tensor(c),
        tensor(base.filters.transpose(0, 2, 1)),
        signfree.temporal_statistics(tensor(base.templates), tensor(y)),
        torch.tensor(label_values.copy(), dtype=torch.long, device=device),
        tensor(weight_values),
    )


def _validate_cases(cases, *, fitting=False):
    cases = tuple(cases)
    if not cases or any(not isinstance(c, TaskCase) for c in cases):
        raise ValueError("Expected nonempty TaskCase sequence")
    if any(c.role not in (("fit",) if fitting else ("fit", "validation")) for c in cases):
        raise PermissionError("Role is not allowed for this operation")
    if len({c.role for c in cases}) != 1:
        raise PermissionError("Mixed roles would merge separate donor partitions")
    if len({c.key for c in cases}) != len(cases):
        raise ValueError("Duplicate participant/condition case")
    ids = tuple(sorted({c.participant_id for c in cases}))
    grids = [{c.condition for c in cases if c.participant_id == pid} for pid in ids]
    if any(grid != grids[0] for grid in grids):
        raise ValueError("Equal participant/condition weighting requires a complete common grid")
    for c in cases:
        _validate_case_geometry(c)
    devices = {c.s.device for c in cases}
    if len(devices) != 1:
        raise ValueError("Cases must use one device")
    # Multiple windows share the same support metadata, not separately chosen packets.
    metadata = {}
    orders = {}
    for c in cases:
        if c.participant_id in orders and orders[c.participant_id] != c.order:
            raise ValueError("Participant acquisition order differs across conditions")
        orders[c.participant_id] = c.order
        key = c.participant_id, c.interface, c.k
        if key in metadata:
            other = metadata[key]
            if c.order != other.order or not np.array_equal(c.packet, other.packet, equal_nan=True):
                raise ValueError("Metadata identity differs across windows")
        metadata[key] = c
        for tensor in (c.s, c.c, c.anchors, c.weights, c.labels):
            if tensor.requires_grad:
                raise ValueError("Support, labels and weights are detached constants")
    for (pid, interface, k), case in metadata.items():
        longer = metadata.get((pid, interface, 5))
        if (
            k == 3
            and longer is not None
            and not np.array_equal(case.packet, longer.packet[:3], equal_nan=True)
        ):
            raise ValueError("Support budgets must share the exact nested metadata prefix")
    return tuple(sorted(cases, key=lambda c: c.key)), ids


def _validate_case_geometry(case):
    """Reject forged/legacy shapes before fitting or batch stacking, not a reader."""
    if type(case.participant_id) is not int or case.participant_id <= 0:
        raise ValueError("Participant ID must be a positive integer")
    if any(type(value) is not int or value not in (0, 1) for value in (case.interface, case.order)):
        raise ValueError("Interface and order must be integer0/1")
    if type(case.k) is not int or case.k not in (3, 5):
        raise ValueError("Only support budgets3/5 are allowed")
    for name, shape in (("q", (5, 8, 15)), ("m", (8, 2)), ("packet", (case.k, 8))):
        value = getattr(case, name)
        if not isinstance(value, np.ndarray) or value.dtype != np.float64 or value.shape != shape:
            raise ValueError("Invalid float64 temporal case " + name)
        if name != "packet" and not np.isfinite(value).all():
            raise ValueError("Nonfinite temporal case " + name)
    for name, shape in (("mask", (case.k, 8)), ("available", (8,))):
        value = getattr(case, name)
        if not isinstance(value, np.ndarray) or value.dtype != bool or value.shape != shape:
            raise ValueError("Invalid boolean temporal case " + name)
    raw_m, raw_available = features.metadata_features(case.packet)
    if (
        not np.array_equal(case.mask, np.isfinite(case.packet))
        or not np.array_equal(case.available, raw_available)
        or not np.array_equal(case.m, raw_m)
    ):
        raise ValueError("Metadata/mask/available must match the exact support packet")
    tensors = (
        (case.s, (5, 12, 8, 8)),
        (case.c, (5, 12, 8, 8)),
        (case.anchors, (5, 12, 8)),
        (case.weights, (5,)),
        (case.statistics.query_gram, (12, 5, 8, 8)),
        (case.statistics.template_gram, (12, 5, 8, 8)),
        (case.statistics.cross_gram, (12, 12, 5, 8, 8)),
    )
    for value, shape in tensors:
        if (
            not isinstance(value, torch.Tensor)
            or value.dtype != torch.float64
            or value.shape != shape
            or value.requires_grad
            or value.device != case.s.device
            or not torch.isfinite(value).all()
        ):
            raise ValueError("Invalid detached float64 temporal case tensor")
    if (case.weights <= 0).any():
        raise ValueError("Explicit positive native band weights are required")
    if (
        not isinstance(case.labels, torch.Tensor)
        or case.labels.dtype != torch.long
        or case.labels.shape != (12,)
        or case.labels.requires_grad
        or case.labels.device != case.s.device
        or not torch.equal(case.labels.sort().values, torch.arange(12, device=case.labels.device))
    ):
        raise ValueError("Exactly one source label per class is required")


def _donors(cases):
    mapping = {}
    for condition in sorted({c.condition for c in cases}):
        rows = [c for c in cases if c.condition == condition]
        donors = features.donor_map(
            [c.participant_id for c in rows],
            np.stack([c.mask for c in rows]),
            [c.order for c in rows],
            condition[0],
        )
        mapping.update({c.key: donors[c.participant_id] for c in rows})
    return mapping


@dataclass(frozen=True)
class HeadFit:
    coefficients: np.ndarray
    initial_loss: float
    final_loss: float
    trace: tuple[dict, ...]

    def record(self):
        return {
            "coefficients": self.coefficients.tolist(),
            "initial_loss": self.initial_loss,
            "final_loss": self.final_loss,
            "steps": len(self.trace),
            "trace": list(self.trace),
        }


@dataclass(frozen=True)
class Pipeline:
    fit_ids: tuple[int, ...]
    regularization: float
    q_scaler: object
    m_scaler: object
    q2_scaler: object
    q: HeadFit
    residuals: dict[str, HeadFit]
    donor_ids: dict[tuple, int]

    @property
    def q_hash(self):
        digest = hashlib.sha256()
        for value in (self.q_scaler.mean, self.q_scaler.scale, self.q.coefficients):
            digest.update(np.asarray(value, dtype="<f8").tobytes())
        return digest.hexdigest()

    def __post_init__(self):
        if (
            not isinstance(self.fit_ids, tuple)
            or not self.fit_ids
            or any(type(pid) is not int or pid <= 0 for pid in self.fit_ids)
            or tuple(sorted(set(self.fit_ids))) != self.fit_ids
        ):
            raise ValueError("Pipeline requires sorted unique positive fit IDs")
        if self.regularization not in LAMBDAS:
            raise ValueError("Pipeline lambda outside fixed grid")
        for scaler, dimensions in ((self.q_scaler, 15), (self.m_scaler, 2), (self.q2_scaler, 2)):
            if (
                not isinstance(scaler, features.FeatureScaler)
                or scaler.fit_ids != self.fit_ids
                or scaler.mean.shape != (dimensions,)
            ):
                raise ValueError("Pipeline scaler dimensions/fit IDs mismatch")
        if (
            not isinstance(self.q, HeadFit)
            or self.q.coefficients.shape != (16,)
            or not np.isfinite(self.q.coefficients).all()
        ):
            raise ValueError("Pipeline Q requires sixteen finite coefficients")
        if set(self.residuals) != set(RESIDUAL_ARMS) or any(
            not isinstance(head, HeadFit)
            or head.coefficients.shape != (3,)
            or not np.isfinite(head.coefficients).all()
            for head in self.residuals.values()
        ):
            raise ValueError("Pipeline requires three finite three-coefficient residuals")

    @property
    def schema(self):
        return SCHEMA

    @property
    def score_schema(self):
        return SCORE_SCHEMA

    def record(self):
        def scaler(s):
            return {"mean": s.mean.tolist(), "scale": s.scale.tolist(), "fit_ids": list(s.fit_ids)}

        return {
            "schema": SCHEMA,
            "score_schema": SCORE_SCHEMA,
            "fit_ids": list(self.fit_ids),
            "lambda": self.regularization,
            "q_hash": self.q_hash,
            "q_scaler": scaler(self.q_scaler),
            "m_scaler": scaler(self.m_scaler),
            "q2_scaler": scaler(self.q2_scaler),
            "Q": self.q.record(),
            "residuals": {k: v.record() for k, v in self.residuals.items()},
            "donors": [{"case": list(k), "donor_id": v} for k, v in sorted(self.donor_ids.items())],
        }


def _as_tensor(value, device):
    return torch.tensor(np.array(value, copy=True), dtype=torch.float64, device=device)


def _head(values, coefficients, fraction):
    return fraction * A * torch.tanh(values @ coefficients[:-1] + coefficients[-1])


def _prior_scores(case, q_logits, residual_logits=None):
    logits = q_logits if residual_logits is None else q_logits + residual_logits
    r = operator.shape_prior(logits)
    if not isinstance(case, TaskCase):
        raise TypeError("Temporal scorer requires temporal TaskCase")
    projectors = signfree.bounded_projectors(case.s, case.c, r[:, None, :])
    return signfree.score_temporal_gram(projectors, case.statistics, case.weights)[0]


def _ce(scores, case):
    return torch.nn.functional.cross_entropy(scores / case.weights.sum() / 0.1, case.labels)


def _fit_head(dimensions, loss_function, regularization, device):
    coefficient = torch.nn.Parameter(
        torch.zeros(dimensions + 1, dtype=torch.float64, device=device)
    )
    optimizer = torch.optim.Adam([coefficient], lr=0.01, betas=(0.9, 0.999), eps=1e-8)
    trace = []
    for step in range(STEPS):
        optimizer.zero_grad(set_to_none=True)
        ce = loss_function(coefficient)
        loss = ce + regularization * coefficient.square().mean()
        if not torch.isfinite(loss):
            raise ValueError("VALIDITY_FAILURE: nonfinite training loss")
        loss.backward()
        if coefficient.grad is None or not torch.isfinite(coefficient.grad).all():
            raise ValueError("VALIDITY_FAILURE: nonfinite training gradient")
        trace.append(
            {
                "step": step + 1,
                "loss_before_step": float(loss.detach()),
                "ce_before_step": float(ce.detach()),
                "gradient_norm": float(torch.linalg.vector_norm(coefficient.grad)),
            }
        )
        optimizer.step()
        if not torch.isfinite(coefficient).all():
            raise ValueError("VALIDITY_FAILURE: nonfinite coefficient update")
    with torch.no_grad():
        final = loss_function(coefficient) + regularization * coefficient.square().mean()
    if not torch.isfinite(final):
        raise ValueError("VALIDITY_FAILURE: nonfinite final loss")
    return HeadFit(
        _readonly(coefficient.detach().cpu().numpy()),
        trace[0]["loss_before_step"],
        float(final),
        tuple(trace),
    )


def fit_pipeline(cases, regularization, *, backend="scalar"):
    """Fit Q then three matched residuals at one predeclared lambda, 200steps/head."""
    if regularization not in LAMBDAS:
        raise ValueError("Lambda is outside the frozen selection grid")
    cases, ids = _validate_cases(cases, fitting=True)
    if backend not in ("scalar", "batch"):
        raise ValueError("Unknown task loss backend")
    batch = None
    if backend == "batch":
        from cfeg.analysis.task_trca_temporal_batch import TaskBatch

        batch = TaskBatch(cases)
    pids = [c.participant_id for c in cases]
    q = np.stack([c.q for c in cases])
    m = np.stack([c.m for c in cases])
    available = np.stack([c.available for c in cases])
    available_bands = np.broadcast_to(available[:, None], q.shape[:-1])
    q_scaler = features.fit_scaler(q, pids, ids)
    m_scaler = features.fit_scaler(m, pids, ids, available)
    q2_scaler = features.fit_scaler(q[..., 1:3], pids, ids, available_bands)
    device = cases[0].s.device
    qx = _as_tensor(q_scaler.transform(q), device)
    mx = _as_tensor(m_scaler.transform(m, available), device)
    q2x = _as_tensor(q2_scaler.transform(q[..., 1:3], available_bands), device)
    observed = _as_tensor(available, device)

    def q_loss(coefficient):
        logits = _head(qx, coefficient, 0.8)
        if batch is not None:
            return batch.loss(logits)
        return torch.stack(
            [_ce(_prior_scores(c, logits[j]), c) for j, c in enumerate(cases)]
        ).mean()

    qfit = _fit_head(15, q_loss, regularization, device)
    frozen_q = _head(qx, _as_tensor(qfit.coefficients, device), 0.8).detach()
    donors = _donors(cases)
    lookup = {c.key: j for j, c in enumerate(cases)}
    donor_positions = [lookup[(donors[c.key], *c.condition)] for c in cases]
    residuals = {}
    for arm in RESIDUAL_ARMS:
        values = q2x if arm == "Q2" else (mx if arm == "QM" else mx[donor_positions])
        if arm != "Q2":
            values = values[:, None].expand(-1, 5, -1, -1)

        def residual_loss(coefficient, values=values):
            logits = _head(values, coefficient, 0.2) * observed[:, None, :]
            if batch is not None:
                return batch.loss(frozen_q + logits)
            return torch.stack(
                [_ce(_prior_scores(c, frozen_q[j], logits[j]), c) for j, c in enumerate(cases)]
            ).mean()

        residuals[arm] = _fit_head(2, residual_loss, regularization, device)
    return Pipeline(
        ids, float(regularization), q_scaler, m_scaler, q2_scaler, qfit, residuals, donors
    )


def predict(pipeline, cases, arm="Q"):
    """In-memory source/validation scores. No final-query reader is invoked."""
    if not isinstance(pipeline, Pipeline):
        raise TypeError("Temporal prediction rejects legacy/global Pipeline")
    if arm not in ("ISO", "Q", "Q2", "QM", "SHAM_REFIT", "PERMUTED", "STALE", "MISSING"):
        raise ValueError("Unknown positive-mass arm")
    cases, ids = _validate_cases(cases)
    if cases[0].role == "validation" and set(ids) & set(pipeline.fit_ids):
        raise PermissionError("Validation IDs overlap fitted participants")
    if cases[0].role == "fit" and set(ids) != set(pipeline.fit_ids):
        raise PermissionError("Fit diagnostics must preserve the complete fitted donor partition")
    lookup = {c.key: c for c in cases}
    donors = _donors(cases) if arm in ("SHAM_REFIT", "PERMUTED") else None
    result = {}
    with torch.no_grad():
        for c in cases:
            device = c.s.device
            qx = _as_tensor(pipeline.q_scaler.transform(c.q), device)
            qlog = _head(qx, _as_tensor(pipeline.q.coefficients, device), 0.8)
            if arm == "ISO":
                qlog = torch.zeros_like(qlog)
            rlog = None
            if arm not in ("ISO", "Q", "MISSING"):
                source = lookup[(donors[c.key], *c.condition)] if donors is not None else c
                available = c.available
                if arm == "Q2":
                    mask = np.broadcast_to(available, (5, 8))
                    values = pipeline.q2_scaler.transform(c.q[..., 1:3], mask)
                    coefficients = pipeline.residuals[arm].coefficients
                else:
                    raw = source.m
                    if arm == "STALE":
                        raw, available = features.metadata_features(np.repeat(c.packet[:1], c.k, 0))
                    values = pipeline.m_scaler.transform(raw, available)
                    coefficients = pipeline.residuals[
                        "SHAM_REFIT" if arm == "SHAM_REFIT" else "QM"
                    ].coefficients
                rlog = _head(_as_tensor(values, device), _as_tensor(coefficients, device), 0.2)
                rlog = rlog * _as_tensor(available, device)
            result[c.key] = _prior_scores(c, qlog, rlog).cpu().numpy()
    return result


def validation_ce(pipeline, cases, arm):
    cases, _ = _validate_cases(cases)
    scores = predict(pipeline, cases, arm)
    return float(np.mean([float(_ce(_as_tensor(scores[c.key], c.s.device), c)) for c in cases]))


def choose_lambda(rows):
    """Only Q validation CE selects common lambda; residual values are not consulted."""
    rows = tuple(rows)
    if len(rows) != 9 or any(r["lambda"] not in LAMBDAS for r in rows):
        raise ValueError("Need exactly nine rows on the frozen lambda grid")
    means = {}
    for value in LAMBDAS:
        selected = [r for r in rows if r["lambda"] == value]
        if len(selected) != 3 or {r["inner_fold"] for r in selected} != {0, 1, 2}:
            raise ValueError("Need exactly three inner validation folds per lambda")
        counts = [len(r["validation_ids"]) for r in selected]
        if any(n <= 0 for n in counts):
            raise ValueError("Empty validation fold")
        means[value] = float(
            np.average([r["validation_ce"]["Q"] for r in selected], weights=counts)
        )
    if not all(math.isfinite(v) for v in means.values()):
        raise ValueError("Nonfinite Q validation loss")
    best = min(means.values())
    return max(value for value, loss in means.items() if loss <= best + 1e-12)


def nested_fit(outer_training_cases, *, outer_evaluation_ids=(), backend="scalar", progress=None):
    """Rebuild entire pipeline for each inner fold/lambda; final outer refit."""
    cases, ids = _validate_cases(outer_training_cases, fitting=True)
    outer_evaluation_ids = tuple(outer_evaluation_ids)
    if any(type(pid) is not int or pid <= 0 for pid in outer_evaluation_ids) or len(
        set(outer_evaluation_ids)
    ) != len(outer_evaluation_ids):
        raise ValueError("Outer evaluation IDs must be unique positive integers")
    if set(ids) & set(outer_evaluation_ids):
        raise PermissionError("Outer evaluation participant leaked into fitting")
    if len(ids) < 6:
        raise ValueError("Need at least6 participants for this three-fold engineering path")
    rows = []
    for fold in range(3):
        validation_ids = ids[fold::3]
        training_ids = tuple(pid for pid in ids if pid not in validation_ids)
        train = tuple(c for c in cases if c.participant_id in training_ids)
        validation = tuple(
            replace(c, role="validation") for c in cases if c.participant_id in validation_ids
        )
        for value in LAMBDAS:
            if progress is not None:
                progress({"event": "inner_start", "fold": fold, "lambda": value})
            fitted = fit_pipeline(
                train, value, **({"backend": backend} if backend != "scalar" else {})
            )
            rows.append(
                {
                    "inner_fold": fold,
                    "lambda": value,
                    "fit_ids": list(training_ids),
                    "validation_ids": list(validation_ids),
                    "pipeline": fitted.record(),
                    "validation_ce": {
                        arm: validation_ce(fitted, validation, arm) for arm in ("Q", *RESIDUAL_ARMS)
                    },
                }
            )
            if progress is not None:
                progress({"event": "inner_complete", "fold": fold, "lambda": value})
    selected = choose_lambda(rows)
    if progress is not None:
        progress({"event": "final_refit_start", "lambda": selected})
    final = fit_pipeline(cases, selected, **({"backend": backend} if backend != "scalar" else {}))
    return final, {
        "schema": SCHEMA,
        "score_schema": SCORE_SCHEMA,
        "selected_lambda": selected,
        "selection_signal": "Q CE only",
        "outer_evaluation_ids": list(outer_evaluation_ids),
        "inner": rows,
    }
