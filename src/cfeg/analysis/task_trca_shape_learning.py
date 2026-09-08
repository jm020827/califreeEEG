"""Task-aligned frozen-Q residual learning; pure in-memory, no dataset/file access.

All constants follow task_aligned_trca_shape_v1_design.json. This is an
engineering learner, not authorization to run a human study or open held data.
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
    statistics: object
    labels: torch.Tensor
    weights: torch.Tensor

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
        operator.gram_statistics(tensor(base.templates), tensor(y)),
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
    devices = {c.s.device for c in cases}
    if len(devices) != 1:
        raise ValueError("Cases must use one device")
    # Multiple windows share the same support metadata, not separately chosen packets.
    metadata = {}
    for c in cases:
        key = c.participant_id, c.interface, c.k
        if key in metadata:
            other = metadata[key]
            if c.order != other.order or not np.array_equal(c.packet, other.packet, equal_nan=True):
                raise ValueError("Metadata identity differs across windows")
        metadata[key] = c
        for tensor in (c.s, c.c, c.anchors, c.weights, c.labels):
            if tensor.requires_grad:
                raise ValueError("Support, labels and weights are detached constants")
    return tuple(sorted(cases, key=lambda c: c.key)), ids


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

    def record(self):
        def scaler(s):
            return {"mean": s.mean.tolist(), "scale": s.scale.tolist(), "fit_ids": list(s.fit_ids)}

        return {
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
    w = operator.bounded_filters(case.s, case.c, case.anchors, r[:, None, :])
    return operator.score_gram(w.transpose(-1, -2), case.statistics, case.weights)[0]


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


def fit_pipeline(cases, regularization):
    """Fit Q then three matched residuals at one predeclared lambda, 200steps/head."""
    if regularization not in LAMBDAS:
        raise ValueError("Lambda is outside the frozen selection grid")
    cases, ids = _validate_cases(cases, fitting=True)
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
            return torch.stack(
                [_ce(_prior_scores(c, frozen_q[j], logits[j]), c) for j, c in enumerate(cases)]
            ).mean()

        residuals[arm] = _fit_head(2, residual_loss, regularization, device)
    return Pipeline(
        ids, float(regularization), q_scaler, m_scaler, q2_scaler, qfit, residuals, donors
    )


def predict(pipeline, cases, arm="Q"):
    """In-memory source/validation scores. No final-query reader is invoked."""
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


def nested_fit(outer_training_cases, *, outer_evaluation_ids=()):
    """Rebuild entire pipeline for each inner fold/lambda; final outer refit."""
    cases, ids = _validate_cases(outer_training_cases, fitting=True)
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
            fitted = fit_pipeline(train, value)
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
    selected = choose_lambda(rows)
    final = fit_pipeline(cases, selected)
    return final, {
        "selected_lambda": selected,
        "selection_signal": "Q CE only",
        "outer_evaluation_ids": list(outer_evaluation_ids),
        "inner": rows,
    }
