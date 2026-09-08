"""M-blind engineering helpers, not an EEG adapter or an efficacy experiment.

Q inputs are precomputed support-only features. Callers must enforce their source
data rights; shapes and participant IDs cannot prove how a feature was obtained.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

ALPHAS = (1.0, 0.1, 0.01, 0.001, 0.0001, 0.0)
REPRESENTATIONS = ("local", "context")
FOLDS = 3
TIE_ATOL = 1e-12


def _finite(value, name):
    raw = np.asarray(value)
    if raw.dtype.kind not in "fiu" or not np.isfinite(raw).all():
        raise ValueError(f"{name} must contain finite real numbers")
    return np.asarray(raw, dtype=np.float64)


def q_design(q, representation):
    """Retain participant/budget/band/channel axes; context is within-channel-axis only."""
    x = _finite(q, "q")
    if x.ndim != 5 or min(x.shape) < 1 or x.shape[-2] < 2:
        raise ValueError("q must be [participant,budget,band,channel>=2,feature]")
    if representation == "local":
        return x.copy()
    if representation != "context":
        raise ValueError("Unknown Q representation")
    with np.errstate(over="ignore", invalid="ignore"):
        means = np.broadcast_to(x.mean(axis=-2, keepdims=True), x.shape)
    _finite(means, "derived Q context")
    return np.concatenate((x, means), axis=-1)


@dataclass(frozen=True)
class RidgeModel:
    mean: np.ndarray
    scale: np.ndarray
    coefficient: np.ndarray
    intercept: float
    alpha: float
    design_rank: int

    def predict(self, features):
        x = _finite(features, "features")
        if x.ndim < 1 or x.shape[-1] != len(self.mean):
            raise ValueError("Prediction feature count mismatch")
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            prediction = ((x - self.mean) / self.scale) @ self.coefficient + self.intercept
        return _finite(prediction, "ridge prediction")

    def record(self):
        return {
            "mean": self.mean.tolist(),
            "scale": self.scale.tolist(),
            "coefficient": self.coefficient.tolist(),
            "intercept": self.intercept,
            "alpha": self.alpha,
            "design_rank": self.design_rank,
        }


def fit_ridge(features, target, alpha):
    """Unpenalized intercept and mean-loss ridge; alpha0 uses minimum-norm least squares."""
    x, y, penalty = (
        _finite(features, "features"),
        _finite(target, "target"),
        _finite(alpha, "alpha"),
    )
    if x.ndim != 2 or min(x.shape) < 1 or y.shape != (len(x),):
        raise ValueError("Invalid ridge geometry")
    if penalty.ndim != 0 or penalty < 0:
        raise ValueError("alpha must be a nonnegative scalar")
    with np.errstate(over="ignore", invalid="ignore"):
        mean, scale = x.mean(axis=0), x.std(axis=0)
        intercept = float(y.mean())
    _finite(mean, "derived feature mean")
    _finite(scale, "derived feature scale")
    _finite(intercept, "derived intercept")
    scale = np.where(scale < 1e-12, 1.0, scale)
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        z = (x - mean) / scale
        response = (y - intercept) / np.sqrt(len(x))
    _finite(z, "standardized features")
    _finite(response, "centered response")
    normalized = z / np.sqrt(len(x))
    augmented = np.vstack((normalized, np.sqrt(penalty) * np.eye(x.shape[1])))
    augmented_y = np.concatenate((response, np.zeros(x.shape[1])))
    coefficient = np.linalg.lstsq(augmented, augmented_y, rcond=None)[0]
    if not np.isfinite(coefficient).all():
        raise ValueError("Nonfinite ridge solution")
    return RidgeModel(
        mean.copy(),
        scale.copy(),
        coefficient,
        intercept,
        float(penalty),
        int(np.linalg.matrix_rank(normalized)),
    )


def participant_folds(participant_ids):
    ids = np.asarray(participant_ids)
    if ids.ndim != 1 or ids.dtype.kind not in "iu" or len(ids) < FOLDS:
        raise ValueError("At least three integer participant IDs required")
    if len(np.unique(ids)) != len(ids) or np.any(ids < 0):
        raise ValueError("Participant IDs must be unique nonnegative integers")
    assignment = np.empty(len(ids), dtype=int)
    assignment[np.argsort(ids)] = np.arange(len(ids)) % FOLDS
    return assignment


@dataclass(frozen=True)
class SelectedQ:
    representation: str
    ridge: RidgeModel
    feature_count: int

    def predict(self, q):
        x = q_design(q, self.representation)
        if np.asarray(q).shape[-1] != self.feature_count:
            raise ValueError("Q feature count mismatch")
        return self.ridge.predict(x)

    def record(self):
        return {
            "representation": self.representation,
            "feature_count": self.feature_count,
            "ridge": self.ridge.record(),
        }


def _inputs(q, target, participant_ids):
    x, y = q_design(q, "local"), _finite(target, "target")
    ids = np.asarray(participant_ids)
    assignment = participant_folds(ids)
    if y.shape != x.shape[:-1] or len(ids) != len(x):
        raise ValueError("Q/target/participant geometry mismatch")
    return x, y, ids.copy(), assignment


def select_q(q, target, participant_ids):
    """Select on training participants only. Return fitted Q and full inner CV receipt."""
    x, y, ids, assignment = _inputs(q, target, participant_ids)
    candidate_rows, split_rows = [], []
    for fold in range(FOLDS):
        split_rows.append(
            {
                "fold": fold,
                "fit_ids": ids[assignment != fold].tolist(),
                "validation_ids": ids[assignment == fold].tolist(),
            }
        )
    for representation in REPRESENTATIONS:
        design = q_design(x, representation)
        for alpha in ALPHAS:
            losses = np.empty(len(x))
            for fold in range(FOLDS):
                train, valid = assignment != fold, assignment == fold
                model = fit_ridge(
                    design[train].reshape(-1, design.shape[-1]), y[train].ravel(), alpha
                )
                errors = (model.predict(design[valid]) - y[valid]) ** 2
                losses[valid] = errors.reshape(valid.sum(), -1).mean(axis=1)
            with np.errstate(over="ignore", invalid="ignore"):
                loss = float(losses.mean())
            if not np.isfinite(losses).all() or not np.isfinite(loss):
                raise ValueError("Nonfinite Q selection loss")
            candidate_rows.append(
                {
                    "representation": representation,
                    "alpha": alpha,
                    "participant_mse": losses.tolist(),
                    "mean_participant_mse": loss,
                }
            )
    minimum = min(row["mean_participant_mse"] for row in candidate_rows)
    # Compare every candidate to the global minimum, not a running incumbent:
    # tolerance-based ties are nontransitive and can otherwise skip priority.
    best = next(row for row in candidate_rows if row["mean_participant_mse"] <= minimum + TIE_ATOL)
    representation, alpha = best["representation"], best["alpha"]
    design = q_design(x, representation)
    selected = SelectedQ(
        representation,
        fit_ridge(design.reshape(-1, design.shape[-1]), y.ravel(), alpha),
        x.shape[-1],
    )
    return selected, {
        "participant_ids": ids.tolist(),
        "splits": split_rows,
        "candidates": candidate_rows,
        "selected": selected.record(),
        "loss_unit": "equal participant mean of budget/band/channel MSE",
    }


def crossfit_q(q, target, participant_ids):
    """Nested participant OOF nuisance fits plus one final all-training Q fit.

    No evaluation/query/M arguments. OOF predictions are not final efficacy
    evaluation; a future residual learner still needs untouched outer evaluation.
    """
    x, y, ids, assignment = _inputs(q, target, participant_ids)
    if len(x) < 6:
        raise ValueError("Nested cross-fitting requires at least six participants")
    oof = np.empty_like(y)
    receipts = []
    for fold in range(FOLDS):
        train, valid = assignment != fold, assignment == fold
        model, receipt = select_q(x[train], y[train], ids[train])
        oof[valid] = model.predict(x[valid])
        receipts.append(
            {
                "fold": fold,
                "fit_ids": ids[train].tolist(),
                "oof_ids": ids[valid].tolist(),
                "inner_selection": receipt,
            }
        )
    final, final_receipt = select_q(x, y, ids)
    return (
        final,
        oof,
        {
            "scope": "training nuisance cross-fit, not M or EEG efficacy",
            "oof_folds": receipts,
            "final_selection": final_receipt,
            "training_size_mismatch": {
                "oof_fit_participants": [len(r["fit_ids"]) for r in receipts],
                "final_fit_participants": len(ids),
            },
        },
    )


def proxy_error_components(prediction, target):
    """Per-last-axis MSE identity; level/shape are not physical noise components."""
    p, y = _finite(prediction, "prediction"), _finite(target, "target")
    if p.ndim < 1 or min(p.shape) < 1 or p.shape != y.shape:
        raise ValueError("Prediction and target must share a nonempty channel axis")
    error = p - y
    level = error.mean(axis=-1, keepdims=True)
    return {
        "raw_mse": np.mean(error**2, axis=-1),
        "level_mse": level[..., 0] ** 2,
        "shape_mse": np.mean((error - level) ** 2, axis=-1),
    }


def prior_log_shape(prior):
    p = _finite(prior, "prior")
    if p.ndim < 1 or min(p.shape) < 1 or np.any(p <= 0):
        raise ValueError("Prior must have a positive nonempty channel axis")
    logp = np.log(p)
    return logp - logp.mean(axis=-1, keepdims=True)


def score_sensitivity(baseline, alternative):
    """A strict m>2*epsilon certificate guarantees no argmax change; converse is false."""
    base, alt = _finite(baseline, "baseline"), _finite(alternative, "alternative")
    if base.ndim != 2 or base.shape[0] < 1 or base.shape[1] < 2 or alt.shape != base.shape:
        raise ValueError("Scores must be matching nonempty [query,class>=2] arrays")
    top_two = np.sort(base, axis=-1)[:, -2:]
    margin = top_two[:, 1] - top_two[:, 0]
    epsilon = np.max(np.abs(alt - base), axis=-1)
    return {
        "baseline_margin": margin,
        "score_max_abs_change": epsilon,
        "certified_unchanged": margin > 2 * epsilon,
        "prediction_changed": base.argmax(axis=-1) != alt.argmax(axis=-1),
    }
