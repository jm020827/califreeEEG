"""Pure, fixed source-channel probe; no reader, runtime authority, or I/O.

The caller freezes the complete ``fit_probe`` record before ``evaluate_probe``.
Raw source/support are optional for small fitting unit fixtures, but an actual
independent audit needs both to reconstruct the supplied Q and target. Shapes
cannot prove provenance. This probe predicts channel margins, not accuracy or
calibration cost. All four models are joint ridge fits, not N1 residual heads.
"""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np

from cfeg.analysis.metadata_prior_validation import RidgeModel, fit_ridge, participant_folds
from cfeg.analysis.source_channel_margin import participant_margin_mse
from cfeg.analysis.task_trca_shape_features import metadata_features

ARMS = ("Q", "Q2", "QM", "SHAM")
ALPHAS = (1.0, 0.01, 0.0001)
MODELS_SCHEMA = "cfeg.source-channel-margin-probe.models.v1"
RESULT_SCHEMA = "cfeg.source-channel-margin-probe.result.v1"
FIT_COUNT = 120
TIE_ATOL = 1e-12
HEADROOM = 1e-8
MIN_COVERED = 32


def _finite(value, name):
    x = np.asarray(value)
    if x.dtype.kind not in "fiu" or not np.isfinite(x).all():
        raise ValueError(f"{name} must be finite real numbers")
    return np.asarray(x, dtype=np.float64)


def _data(value):
    required = {"ids", "orders", "packet", "q", "target"}
    raw = {"support", "source_block"}
    if not isinstance(value, Mapping) or set(value) not in (required, required | raw):
        raise ValueError("data requires ids/orders/packet/q/target and optional paired raw arrays")
    ids, orders = np.asarray(value["ids"]), np.asarray(value["orders"])
    folds = participant_folds(ids)
    p = len(ids)
    if p < 9 or p % 3 or orders.shape != (p,) or orders.dtype.kind not in "iu":
        raise ValueError("need p>=9 divisible by3, with integer acquisition orders")
    if not np.isin(orders, (0, 1)).all():
        raise ValueError("orders must be first-interface bits")
    result = {"ids": ids, "orders": orders, "fold_assignment": folds}
    shapes = {"packet": (p, 2, 3, 8), "q": (p, 2, 5, 8, 15), "target": (p, 2, 5, 8)}
    for name, shape in shapes.items():
        x = np.asarray(value[name])
        if x.dtype != np.dtype("float64") or x.shape != shape:
            raise ValueError(f"{name} must be float64 with shape {shape}")
        if name == "packet":
            if np.isinf(x).any() or np.any(x[np.isfinite(x)] < 0):
                raise ValueError("packet permits only nonnegative observed values or NaN")
        else:
            _finite(x, name)
        result[name] = x
    if np.max(np.abs(result["target"].mean(axis=-1))) > TIE_ATOL:
        raise ValueError("target must be channel centered")
    if raw.issubset(value):
        support, source = np.asarray(value["support"]), np.asarray(value["source_block"])
        if (
            support.dtype != np.dtype("float64")
            or support.ndim != 7
            or support.shape[:6] != (p, 2, 3, 12, 5, 8)
            or support.shape[-1] < 24
            or source.dtype != np.dtype("float64")
            or source.shape != (p, 2, 12, 5, 8, support.shape[-1])
        ):
            raise ValueError("raw support/source geometry or float64 dtype mismatch")
        _finite(support, "support")
        _finite(source, "source_block")
    m, available = [], []
    for packet in result["packet"]:
        features = [metadata_features(interface) for interface in packet]
        m.append([item[0] for item in features])
        available.append([item[1] for item in features])
    result["m"] = np.asarray(m, dtype=np.float64)
    result["available"] = np.asarray(available, dtype=bool)
    return result


def _sorted_rows(data, selected):
    rows = np.flatnonzero(selected)
    return rows[np.argsort(data["ids"][rows])]


def _scaler(values, fit_ids, available=None):
    """Population statistics; supports the new nonnegative-ID unit API."""
    rows = values.reshape(-1, values.shape[-1])
    if available is not None:
        rows = rows[np.asarray(available).reshape(-1)]
    if len(rows):
        with np.errstate(over="ignore", invalid="ignore"):
            mean, scale = rows.mean(axis=0), rows.std(axis=0, ddof=0)
        _finite(mean, "scaler mean")
        _finite(scale, "scaler scale")
        scale = np.where(scale < 1e-12, 1.0, scale)
    else:
        mean, scale = np.zeros(values.shape[-1]), np.ones(values.shape[-1])
    return {"mean": mean.tolist(), "scale": scale.tolist(), "fit_ids": list(fit_ids)}


def _standardize(values, scaler, available=None):
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        output = (values - np.asarray(scaler["mean"])) / np.asarray(scaler["scale"])
    if available is not None:
        output = np.where(np.asarray(available)[..., None], output, 0.0)
    return _finite(output, "standardized features")


def _donor_rows(data, rows):
    """Paired-interface, exact full-mask/order strata; no values or labels."""
    groups = {}
    for row in rows:
        key = (int(data["orders"][row]), np.packbits(np.isfinite(data["packet"][row])).tobytes())
        groups.setdefault(key, []).append(int(row))
    donors = {}
    for group in groups.values():
        group.sort(key=lambda row: int(data["ids"][row]))
        for index, row in enumerate(group):
            donors[row] = group[(index + 1) % len(group)]
    return np.asarray([donors[int(row)] for row in rows], dtype=int)


def _design(data, rows, model, donor_rows=None):
    q = _standardize(data["q"][rows], model["q_scaler"])
    arm = model["arm"]
    if arm == "Q":
        design = q
    elif arm == "Q2":
        with np.errstate(over="ignore", invalid="ignore"):
            design = np.concatenate((q, q[..., [1, 2]] ** 2), axis=-1)
    elif arm in ("QM", "SHAM"):
        selected = rows if arm == "QM" else donor_rows
        if selected is None:
            selected = _donor_rows(data, rows)
        m = _standardize(data["m"][selected], model["m_scaler"], data["available"][selected])
        repeated = np.broadcast_to(m[:, :, None, :, :], (*q.shape[:-1], 2))
        design = np.concatenate((q, repeated), axis=-1)
    else:
        raise ValueError("Unknown arm")
    with np.errstate(over="ignore", invalid="ignore"):
        design = design - design.mean(axis=-2, keepdims=True)
    return _finite(design, "channel-centered design")


def _ridge(record):
    return RidgeModel(
        np.asarray(record["mean"]),
        np.asarray(record["scale"]),
        np.asarray(record["coefficient"]),
        record["intercept"],
        record["alpha"],
        record["design_rank"],
    )


def _coverage(data, rows, model):
    if model["arm"] != "SHAM":
        return None
    donor = _donor_rows(data, rows)
    own, other = data["packet"][rows], data["packet"][donor]
    # NaN positions agree by construction. A raw paired-packet change alone is
    # not the gate; both consumed interface designs must actually change.
    raw_changed = np.any(np.isfinite(own) & (own != other), axis=(1, 2, 3))
    own_design = _design(data, rows, model, rows)
    donor_design = _design(data, rows, model, donor)
    own_consumed = _standardize(own_design, model["ridge"])[..., -2:]
    donor_consumed = _standardize(donor_design, model["ridge"])[..., -2:]
    change = np.max(np.abs(own_consumed - donor_consumed), axis=(2, 3, 4)) > TIE_ATOL
    both = change.all(axis=1)
    return {
        "ids": data["ids"][rows].tolist(),
        "donor_ids": data["ids"][donor].tolist(),
        "raw_changed": raw_changed.tolist(),
        "raw_changed_fraction": float(raw_changed.mean()),
        "design_changed_by_interface": change.tolist(),
        "design_changed_both": both.tolist(),
        "design_changed_fraction": float(both.mean()),
    }


def _fit_model(data, rows, arm, alpha):
    ids = data["ids"][rows].tolist()
    model = {
        "arm": arm,
        "fit_ids": ids,
        "alpha": float(alpha),
        "q_scaler": _scaler(data["q"][rows], ids),
        "m_scaler": (
            _scaler(data["m"][rows], ids, data["available"][rows])
            if arm in ("QM", "SHAM")
            else None
        ),
        "nominal_coefficients": 16 if arm == "Q" else 18,
    }
    design = _design(data, rows, model)
    x = design.reshape(-1, design.shape[-1])
    fitted = fit_ridge(x, data["target"][rows].ravel(), alpha)
    model["ridge"] = fitted.record()
    model["inactive_columns"] = np.flatnonzero(x.std(axis=0, ddof=0) < 1e-12).tolist()
    model["fit_coverage"] = _coverage(data, rows, model)
    return model


def _predict(data, rows, model):
    prediction = _ridge(model["ridge"]).predict(_design(data, rows, model))
    return _finite(prediction - prediction.mean(axis=-1, keepdims=True), "centered prediction")


def _select_alpha(scores):
    minimum = min(row["mean_participant_mse"] for row in scores)
    return next(
        alpha
        for alpha in ALPHAS
        for row in scores
        if row["alpha"] == alpha and row["mean_participant_mse"] <= minimum + TIE_ATOL
    )


def fit_probe(data, event_sink=None):
    """Fit exactly120 ridges; never compute any outer-evaluation target loss.

    The event sink receives JSON-safe dicts, and any sink error aborts fitting.
    Recorded inner scores are source-training selection, not outer efficacy.
    """
    if event_sink is not None and not callable(event_sink):
        raise ValueError("event_sink must be callable")
    values = _data(data)
    record = {
        "schema": MODELS_SCHEMA,
        "arms": list(ARMS),
        "ids": values["ids"].tolist(),
        "fold_assignment": values["fold_assignment"].tolist(),
        "ridge_fit_count": 0,
        "outer": [],
    }

    def completed(outer, arm, stage, inner, alpha, fit, valid):
        record["ridge_fit_count"] += 1
        if record["ridge_fit_count"] > FIT_COUNT:
            raise RuntimeError("Ridge fit budget exceeded")
        if event_sink is not None:
            event_sink(
                {
                    "event": "ridge_fit_completed",
                    "fit_index": record["ridge_fit_count"],
                    "outer_fold": outer,
                    "arm": arm,
                    "stage": stage,
                    "inner_fold": inner,
                    "alpha": float(alpha),
                    "fit_ids": values["ids"][fit].tolist(),
                    "validation_ids": values["ids"][valid].tolist(),
                }
            )

    for fold in range(3):
        train = _sorted_rows(values, values["fold_assignment"] != fold)
        evaluation = _sorted_rows(values, values["fold_assignment"] == fold)
        inner_assignment = participant_folds(values["ids"][train])
        outer = {
            "fold": fold,
            "fit_ids": values["ids"][train].tolist(),
            "evaluation_ids": values["ids"][evaluation].tolist(),
            "arms": {},
        }
        for arm in ARMS:
            inner_records, scores = [], []
            for alpha in ALPHAS:
                losses = np.empty(len(train))
                for inner in range(3):
                    fit, valid = train[inner_assignment != inner], train[inner_assignment == inner]
                    model = _fit_model(values, fit, arm, alpha)
                    completed(fold, arm, "inner", inner, alpha, fit, valid)
                    loss = participant_margin_mse(
                        _predict(values, valid, model), values["target"][valid]
                    )
                    losses[inner_assignment == inner] = loss
                    inner_records.append(
                        {
                            "fold": inner,
                            "alpha": float(alpha),
                            "validation_ids": values["ids"][valid].tolist(),
                            "validation_participant_mse": loss.tolist(),
                            "validation_coverage": _coverage(values, valid, model),
                            "model": model,
                        }
                    )
                with np.errstate(over="ignore", invalid="ignore"):
                    mean_loss = float(losses.mean())
                _finite(mean_loss, "inner pooled participant MSE")
                scores.append(
                    {
                        "alpha": float(alpha),
                        "participant_ids": values["ids"][train].tolist(),
                        "participant_mse": losses.tolist(),
                        "mean_participant_mse": mean_loss,
                    }
                )
            selected = _select_alpha(scores)
            final = _fit_model(values, train, arm, selected)
            completed(fold, arm, "final", None, selected, train, evaluation)
            outer["arms"][arm] = {
                "selected_alpha": selected,
                "alpha_scores": scores,
                "inner": inner_records,
                "final": final,
                "evaluation_coverage": _coverage(values, evaluation, final),
            }
        record["outer"].append(outer)
    if record["ridge_fit_count"] != FIT_COUNT:
        raise RuntimeError("Incomplete fixed fit program")
    if event_sink is not None:
        event_sink({"event": "fit_probe_completed", "ridge_fit_count": FIT_COUNT})
    return record


def _validate_fit(record, arm, ids, alpha):
    if (
        record["arm"] != arm
        or record["fit_ids"] != ids
        or record["alpha"] != alpha
        or record["nominal_coefficients"] != (16 if arm == "Q" else 18)
    ):
        raise ValueError("Model identity/fit partition/alpha mismatch")
    for name, dimension in (("q_scaler", 15), ("m_scaler", 2)):
        scaler = record[name]
        if name == "m_scaler" and arm in ("Q", "Q2"):
            if scaler is not None:
                raise ValueError("M-blind arm must not have an M scaler")
            continue
        mean, scale = _finite(scaler["mean"], name), _finite(scaler["scale"], name)
        if (
            mean.shape != (dimension,)
            or scale.shape != mean.shape
            or np.any(scale <= 0)
            or scaler["fit_ids"] != ids
        ):
            raise ValueError("Scaler shape/fit IDs mismatch")
    ridge, dimension = record["ridge"], 15 if arm == "Q" else 17
    for name in ("mean", "scale", "coefficient"):
        array = _finite(ridge[name], f"ridge {name}")
        if array.shape != (dimension,) or (name == "scale" and np.any(array <= 0)):
            raise ValueError("Ridge vector shape/value mismatch")
    if (
        _finite(ridge["intercept"], "intercept").ndim != 0
        or ridge["alpha"] != alpha
        or type(ridge["design_rank"]) is not int
        or not 0 <= ridge["design_rank"] <= dimension
    ):
        raise ValueError("Invalid ridge scalar record")


def _validate_models(data, models):
    """Reject foreign schemas/partitions; independent audit verifies arithmetic."""
    try:
        if (
            models["schema"] != MODELS_SCHEMA
            or models["arms"] != list(ARMS)
            or models["ids"] != data["ids"].tolist()
            or models["fold_assignment"] != data["fold_assignment"].tolist()
            or models["ridge_fit_count"] != FIT_COUNT
            or len(models["outer"]) != 3
        ):
            raise ValueError("Incomplete or foreign probe models")
        for fold, outer in enumerate(models["outer"]):
            fit = _sorted_rows(data, data["fold_assignment"] != fold)
            valid = _sorted_rows(data, data["fold_assignment"] == fold)
            ids = data["ids"][fit].tolist()
            inner_assignment = participant_folds(data["ids"][fit])
            if (
                outer["fold"] != fold
                or outer["fit_ids"] != ids
                or outer["evaluation_ids"] != data["ids"][valid].tolist()
                or set(outer["arms"]) != set(ARMS)
            ):
                raise ValueError("Outer model partition mismatch")
            for arm in ARMS:
                selected = outer["arms"][arm]
                if (
                    len(selected["inner"]) != 9
                    or len(selected["alpha_scores"]) != 3
                    or [item["alpha"] for item in selected["alpha_scores"]] != list(ALPHAS)
                    or selected["selected_alpha"] != _select_alpha(selected["alpha_scores"])
                ):
                    raise ValueError("Incomplete fixed inner selection")
                for index, inner in enumerate(selected["inner"]):
                    alpha, inner_fold = ALPHAS[index // 3], index % 3
                    inner_ids = data["ids"][fit[inner_assignment != inner_fold]].tolist()
                    valid_ids = data["ids"][fit[inner_assignment == inner_fold]].tolist()
                    if (
                        inner["fold"] != inner_fold
                        or inner["alpha"] != alpha
                        or inner["validation_ids"] != valid_ids
                    ):
                        raise ValueError("Inner model partition mismatch")
                    _validate_fit(inner["model"], arm, inner_ids, alpha)
                _validate_fit(selected["final"], arm, ids, selected["selected_alpha"])
    except (KeyError, TypeError, IndexError, StopIteration) as error:
        raise ValueError("Malformed probe model record") from error


def _summarize(values, predictions, coverages):
    target = values["target"]
    losses = np.stack([participant_margin_mse(prediction, target) for prediction in predictions])
    with np.errstate(over="ignore", invalid="ignore"):
        mean = losses.mean(axis=1)
        fold_mean = np.stack(
            [losses[:, values["fold_assignment"] == fold].mean(axis=1) for fold in range(3)], axis=1
        )
        interface_mean = ((predictions - target[None]) ** 2).mean(axis=(1, 3, 4))
        energy = float(np.mean(target**2))
    _finite(mean, "pooled participant MSE")
    _finite(fold_mean, "outer-fold MSE")
    _finite(interface_mean, "interface MSE")
    _finite(energy, "target energy")
    comparisons = {}
    for index, arm in enumerate(ARMS):
        if arm == "QM":
            continue
        delta = fold_mean[index] - fold_mean[2]
        comparisons[arm] = {
            "relative_mse_gain": float((mean[index] - mean[2]) / mean[index])
            if mean[index] > HEADROOM
            else None,
            "absolute_mse_gain": float(mean[index] - mean[2]),
            "fold_absolute_mse_gain": delta.tolist(),
            "positive_outer_folds": int(np.count_nonzero(delta > TIE_ATOL)),
            "interface_absolute_mse_gain": (interface_mean[index] - interface_mean[2]).tolist(),
        }
    available = values["available"].sum(axis=-1)
    covered = np.all(available >= 2, axis=1)
    gates = {
        "target_variation": energy > HEADROOM,
        "metadata_coverage": int(covered.sum()) >= MIN_COVERED,
        "sham_coverage": all(row["design_changed_fraction"] >= 0.8 for row in coverages),
        "comparator_headroom": all(mean[ARMS.index(arm)] > HEADROOM for arm in comparisons),
        "relative_gain": all(
            row["relative_mse_gain"] is not None and row["relative_mse_gain"] >= 0.02
            for row in comparisons.values()
        ),
        "positive_outer_folds": all(
            row["positive_outer_folds"] >= 2 for row in comparisons.values()
        ),
        "positive_q_interfaces": bool(
            np.all(np.asarray(comparisons["Q"]["interface_absolute_mse_gain"]) > TIE_ATOL)
        ),
    }
    names = (
        "INSUFFICIENT_TARGET_VARIATION",
        "INSUFFICIENT_METADATA_COVERAGE",
        "INSUFFICIENT_SHAM_DESIGN_COVERAGE",
        "INSUFFICIENT_COMPARATOR_HEADROOM",
        "RELATIVE_GAIN_BELOW_THRESHOLD",
        "INSUFFICIENT_POSITIVE_OUTER_FOLDS",
        "NONPOSITIVE_Q_INTERFACE_GAIN",
    )
    failures = [name for name, passed in zip(names, gates.values()) if not passed]
    if not gates["target_variation"]:
        terminal = "INSUFFICIENT_TARGET_VARIATION"
    elif not gates["metadata_coverage"] or not gates["sham_coverage"]:
        terminal = "UNINFORMATIVE_CONTROL"
    elif not gates["comparator_headroom"]:
        terminal = "INSUFFICIENT_COMPARATOR_HEADROOM"
    elif all(gates.values()):
        terminal = "PROMISING_PROBE"
    else:
        terminal = "NO_PROMISING_SIGNAL_IN_THIS_PROBE"
    return {
        "schema": RESULT_SCHEMA,
        "arms": list(ARMS),
        "ids": values["ids"].tolist(),
        "fold_assignment": values["fold_assignment"].tolist(),
        "ridge_fit_count": FIT_COUNT,
        "terminal": terminal,
        "failure_reasons": failures,
        "target_energy": energy,
        "metadata": {
            "available_channel_counts": available.tolist(),
            "covered_participant_ids": values["ids"][covered].tolist(),
            "covered_participant_count": int(covered.sum()),
            "minimum_covered_participants": MIN_COVERED,
        },
        "sham_outer_coverage": coverages,
        "participant_mse": losses.tolist(),
        "mean_mse": mean.tolist(),
        "comparisons": comparisons,
        "fold_mean_mse": fold_mean.tolist(),
        "interface_mean_mse": interface_mean.tolist(),
        "participant_mse_range": np.stack(
            (losses.min(axis=1), losses.max(axis=1)), axis=1
        ).tolist(),
        "gates": {name: bool(value) for name, value in gates.items()},
    }


def evaluate_probe(data, models):
    """Produce one outer OOF prediction per arm/person after caller freeze.

    This pure function cannot establish a durable publication barrier or an
    audit PASS. The runner and independent auditor own those separate claims.
    """
    values = _data(data)
    _validate_models(values, models)
    predictions = np.empty((4, len(values["ids"]), 2, 5, 8), dtype=np.float64)
    coverage = []
    for fold, outer in enumerate(models["outer"]):
        rows = _sorted_rows(values, values["fold_assignment"] == fold)
        for arm_index, arm in enumerate(ARMS):
            model = outer["arms"][arm]["final"]
            predictions[arm_index, rows] = _predict(values, rows, model)
            if arm == "SHAM":
                observed = _coverage(values, rows, model)
                if observed != outer["arms"][arm]["evaluation_coverage"]:
                    raise ValueError("Frozen SHAM evaluation coverage changed")
                coverage.append(observed)
    return predictions, _summarize(values, predictions, coverage)
