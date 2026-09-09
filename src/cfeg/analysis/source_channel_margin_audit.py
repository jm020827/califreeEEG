"""Independent in-memory channel-margin probe replay; no file/access authority.

Only the unchanged support_q_mask implementation is shared. Target Pearson,
metadata aggregation, transforms, partition donors, ridge normal equations,
nested selection and terminal rules are reconstructed without producer imports.
The caller independently binds raw routing, allowed blocks, files and chronology.
"""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np

from cfeg.analysis.task_trca_shape_features import support_q_mask

ARMS = ("Q", "Q2", "QM", "SHAM")
ALPHAS = (1.0, 0.01, 0.0001)
FREQUENCIES = (9.25, 11.25, 13.25, 9.75, 11.75, 13.75, 10.25, 12.25, 14.25, 10.75, 12.75, 14.75)
ATOL = 1e-8
TIE = 1e-12
SCHEMA = "cfeg.source39-channel-margin-audit.v1"
LIMITATIONS = [
    "Q15 uses the shared frozen support_q_mask; it is not an independent Q implementation.",
    "This pure audit authenticates neither file hashes nor source-block/class routing, access rights, or freeze chronology.",
    "120 deterministic ridge reconstructions are audit replay, not an additional efficacy attempt.",
    "A verified probe result is developmental shared-slope channel-margin prediction, not human classification or calibration savings.",
]


def _require(condition, message):
    if not condition:
        raise ValueError(message)


class _Checks:
    def __init__(self):
        self.max_differences = {}
        self.checks = {}

    def equal(self, name, actual, expected, *, exact=False):
        a, b = np.asarray(actual), np.asarray(expected)
        _require(a.shape == b.shape, name + ": shape mismatch")
        _require(a.dtype.kind in "biuf" and b.dtype.kind in "biuf", name + ": numeric required")
        _require(np.isfinite(a).all() and np.isfinite(b).all(), name + ": nonfinite values")
        difference = float(np.max(np.abs(a.astype(float) - b.astype(float)), initial=0))
        self.max_differences[name] = max(difference, self.max_differences.get(name, 0.0))
        _require(
            np.array_equal(a, b) if exact else difference <= ATOL,
            name + ": mismatch (max_abs=" + repr(difference) + ")",
        )


def _validate(data):
    _require(isinstance(data, Mapping), "data mapping required")
    keys = {"ids", "orders", "support", "source_block", "packet", "q", "target"}
    _require(set(data) == keys, "exact source-only data keys required")
    ids, orders = np.asarray(data["ids"]), np.asarray(data["orders"])
    _require(
        ids.ndim == 1
        and ids.dtype.kind in "iu"
        and len(ids) >= 9
        and len(ids) % 3 == 0
        and len(set(ids.tolist())) == len(ids)
        and np.all(ids >= 0),
        "unique nonnegative participant IDs, count>=9 divisible3 required",
    )
    _require(
        orders.shape == ids.shape and orders.dtype.kind in "iu" and np.isin(orders, [0, 1]).all(),
        "integer participant acquisition orders0/1 required",
    )
    count = len(ids)
    support = np.asarray(data["support"])
    _require(
        support.ndim == 7
        and support.shape[:-1] == (count, 2, 3, 12, 5, 8)
        and support.shape[-1] >= 24,
        "complete source-only support geometry required",
    )
    n = support.shape[-1]
    shapes = {
        "support": support.shape,
        "source_block": (count, 2, 12, 5, 8, n),
        "packet": (count, 2, 3, 8),
        "q": (count, 2, 5, 8, 15),
        "target": (count, 2, 5, 8),
    }
    arrays = {"ids": ids, "orders": orders}
    for name, shape in shapes.items():
        value = np.asarray(data[name])
        _require(
            value.shape == shape and value.dtype.kind == "f" and value.dtype.itemsize == 8,
            name + ": exact float64 shape required",
        )
        if name == "packet":
            _require(
                not np.isinf(value).any() and np.all(value[np.isfinite(value)] >= 0),
                "nonnegative metadata, NaN alone missing",
            )
        else:
            _require(np.isfinite(value).all(), name + ": nonfinite input")
        arrays[name] = value
    _require(
        np.max(np.abs(arrays["target"].mean(axis=-1))) <= TIE,
        "target must already be channel centered",
    )
    return arrays


def _target(support, source):
    """Matrix Pearson reference, independently class-balanced and centered."""
    templates = support.mean(axis=0)
    target = np.empty((5, 8))
    for band in range(5):
        for channel in range(8):
            x = source[:, band, channel]
            t = templates[:, band, channel]
            x = x - x.mean(axis=1, keepdims=True)
            t = t - t.mean(axis=1, keepdims=True)
            xn, tn = np.sqrt(np.sum(x * x, axis=1)), np.sqrt(np.sum(t * t, axis=1))
            _require(
                np.isfinite(xn).all()
                and np.isfinite(tn).all()
                and np.all(xn > 1e-12)
                and np.all(tn > 1e-12),
                "undefined centered target correlation",
            )
            correlation = (x / xn[:, None]) @ (t / tn[:, None]).T
            _require(
                np.isfinite(correlation).all() and np.all(np.abs(correlation) <= 1 + 1e-12),
                "invalid target correlation",
            )
            correlation = np.clip(correlation, -1, 1)
            margins = [
                correlation[y, y] - max(correlation[y, j] for j in range(12) if j != y)
                for y in range(12)
            ]
            target[band, channel] = sum(margins) / 12
    return target - target.mean(axis=-1, keepdims=True)


def _metadata(packet):
    """NaN-only observed log1p moments; zero remains an observation."""
    _require(
        packet.shape == (3, 8)
        and not np.isinf(packet).any()
        and np.all(packet[np.isfinite(packet)] >= 0),
        "invalid metadata packet",
    )
    means, sd = np.zeros(8), np.zeros(8)
    available = np.isfinite(packet).any(axis=0)
    for channel in range(8):
        observed = packet[:, channel][np.isfinite(packet[:, channel])]
        if len(observed):
            logged = np.log1p(observed)
            means[channel] = logged.mean()
            sd[channel] = np.sqrt(np.mean((logged - means[channel]) ** 2))
    if available.any():
        means[available] -= means[available].mean()
    return np.column_stack((means, sd)), available


def _raw_bridge(data, checks):
    q, y, m, availability = [], [], [], []
    for index, _pid in enumerate(data["ids"]):
        q_rows, y_rows, m_rows, observed_rows = [], [], [], []
        for interface in range(2):
            packet = data["packet"][index, interface]
            support = data["support"][index, interface]
            q_rows.append(
                support_q_mask(
                    support, np.isfinite(packet), interface, int(data["orders"][index]), FREQUENCIES
                )
            )
            y_rows.append(_target(support, data["source_block"][index, interface]))
            feature, observed = _metadata(packet)
            m_rows.append(feature)
            observed_rows.append(observed)
        q.append(q_rows)
        y.append(y_rows)
        m.append(m_rows)
        availability.append(observed_rows)
    q, y, m, availability = map(np.asarray, (q, y, m, availability))
    checks.equal("raw_Q15", data["q"], q)
    checks.equal("raw_target", data["target"], y)
    checks.checks["raw_target_Q_and_independent_metadata"] = True
    return q, y, m, availability


def _folds(ids):
    result = np.empty(len(ids), dtype=np.int64)
    result[np.argsort(ids)] = np.arange(len(ids)) % 3
    return result


def _donors(ids, orders, packet, indices):
    groups = {}
    for index in indices:
        # One acquisition-order bit jointly determines both interface periods.
        key = int(orders[index]), tuple(np.isfinite(packet[index]).ravel())
        groups.setdefault(key, []).append(int(index))
    result = {}
    for group in groups.values():
        group.sort(key=lambda index: int(ids[index]))
        for position, index in enumerate(group):
            result[index] = group[(position + 1) % len(group)]
    return np.asarray([result[int(index)] for index in indices], dtype=int)


def _scale(values, available=None):
    rows = values.reshape(-1, values.shape[-1])
    if available is not None:
        rows = rows[np.asarray(available, dtype=bool).reshape(-1)]
    if not len(rows):
        return np.zeros(values.shape[-1]), np.ones(values.shape[-1])
    mean, scale = rows.mean(axis=0), rows.std(axis=0, ddof=0)
    return mean, np.where(scale < 1e-12, 1.0, scale)


def _design(q, m, availability, q_scaler, m_scaler, arm):
    standardized_q = (q - q_scaler[0]) / q_scaler[1]
    if arm == "Q":
        design = standardized_q
    elif arm == "Q2":
        design = np.concatenate((standardized_q, standardized_q[..., 1:3] ** 2), axis=-1)
    else:
        standardized_m = (m - m_scaler[0]) / m_scaler[1]
        standardized_m = np.where(availability[..., None], standardized_m, 0)
        repeated = np.broadcast_to(standardized_m[:, :, None], (*q.shape[:-1], 2))
        design = np.concatenate((standardized_q, repeated), axis=-1)
    return design - design.mean(axis=-2, keepdims=True)


def _ridge(design, target, alpha):
    """SPD normal equations, distinct from producer augmented least squares."""
    x = design.reshape(-1, design.shape[-1])
    y = target.reshape(-1)
    mean, scale = _scale(x)
    standardized = (x - mean) / scale
    intercept = float(y.mean())
    lhs = standardized.T @ standardized / len(y) + alpha * np.eye(x.shape[1])
    rhs = standardized.T @ (y - intercept) / len(y)
    coefficient = np.linalg.solve(lhs, rhs)
    _require(np.isfinite(coefficient).all(), "nonfinite independent ridge coefficient")
    return {
        "mean": mean,
        "scale": scale,
        "coefficient": coefficient,
        "intercept": intercept,
        "alpha": alpha,
        "design_rank": int(np.linalg.matrix_rank(standardized / np.sqrt(len(y)))),
    }


def _predict(design, ridge):
    values = ((design - ridge["mean"]) / ridge["scale"]) @ ridge["coefficient"] + ridge["intercept"]
    return values - values.mean(axis=-1, keepdims=True)


def _loss(prediction, target):
    prediction = prediction - prediction.mean(axis=-1, keepdims=True)
    return ((prediction - target) ** 2).mean(axis=(1, 2, 3))


def _scaler_record(scaler, ids):
    return {"mean": scaler[0].tolist(), "scale": scaler[1].tolist(), "fit_ids": ids.tolist()}


def _saved_scaler(record):
    return np.asarray(record["mean"]), np.asarray(record["scale"])


def _model_design(data, q, m, available, indices, record, *, sham):
    source = _donors(data["ids"], data["orders"], data["packet"], indices) if sham else indices
    m_scaler = None if record["m_scaler"] is None else _saved_scaler(record["m_scaler"])
    return _design(
        q[indices],
        m[source],
        available[source],
        _saved_scaler(record["q_scaler"]),
        m_scaler,
        record["arm"],
    )


def _coverage(data, q, m, available, indices, record):
    donors = _donors(data["ids"], data["orders"], data["packet"], indices)
    actual = _model_design(data, q, m, available, indices, record, sham=False)
    displaced = _model_design(data, q, m, available, indices, record, sham=True)
    ridge = record["ridge"]
    first = (actual - ridge["mean"]) / ridge["scale"]
    second = (displaced - ridge["mean"]) / ridge["scale"]
    changed = np.max(np.abs(first[..., -2:] - second[..., -2:]), axis=(2, 3, 4)) > TIE
    both = changed.all(axis=1)
    raw = [
        not np.array_equal(data["packet"][p], data["packet"][d], equal_nan=True)
        for p, d in zip(indices, donors, strict=True)
    ]
    return {
        "ids": data["ids"][indices].tolist(),
        "donor_ids": data["ids"][donors].tolist(),
        "raw_changed": raw,
        "raw_changed_fraction": float(np.mean(raw)),
        "design_changed_by_interface": changed.tolist(),
        "design_changed_both": both.tolist(),
        "design_changed_fraction": float(both.mean()),
    }


def _fit(data, q, y, m, available, indices, arm, alpha):
    fit_ids = data["ids"][indices]
    q_scaler = _scale(q[indices])
    m_scaler = _scale(m[indices], available[indices]) if arm in ("QM", "SHAM") else None
    record = {
        "arm": arm,
        "fit_ids": fit_ids.tolist(),
        "alpha": alpha,
        "q_scaler": _scaler_record(q_scaler, fit_ids),
        "m_scaler": None if m_scaler is None else _scaler_record(m_scaler, fit_ids),
    }
    design = _model_design(data, q, m, available, indices, record, sham=arm == "SHAM")
    ridge = _ridge(design, y[indices], alpha)
    record["ridge"] = {
        name: value.tolist() if isinstance(value, np.ndarray) else value
        for name, value in ridge.items()
    }
    raw_scale = design.reshape(-1, design.shape[-1]).std(axis=0, ddof=0)
    record["inactive_columns"] = np.flatnonzero(raw_scale < 1e-12).tolist()
    record["nominal_coefficients"] = 16 if arm == "Q" else 18
    record["fit_coverage"] = (
        _coverage(data, q, m, available, indices, record) if arm == "SHAM" else None
    )
    return record


def _model_prediction(data, q, m, available, indices, record):
    design = _model_design(data, q, m, available, indices, record, sham=record["arm"] == "SHAM")
    return _predict(design, record["ridge"])


def _nested(data, q, y, m, available):
    """Exactly3outer*4arms*(3alpha*3inner+1final) independent fits."""
    ids = data["ids"]
    assignment = _folds(ids)
    outer_rows = []
    predictions = np.empty((4, len(ids), 2, 5, 8))
    total_fits = 0
    for outer_fold in range(3):
        train = np.asarray(
            sorted(np.flatnonzero(assignment != outer_fold), key=lambda i: int(ids[i]))
        )
        evaluate = np.asarray(
            sorted(np.flatnonzero(assignment == outer_fold), key=lambda i: int(ids[i]))
        )
        inner_assignment = _folds(ids[train])
        arms = {}
        for arm_index, arm in enumerate(ARMS):
            inner, alpha_scores = [], []
            for alpha in ALPHAS:
                oof_losses = np.empty(len(train))
                for inner_fold in range(3):
                    fit_indices = train[inner_assignment != inner_fold]
                    validation = train[inner_assignment == inner_fold]
                    model = _fit(data, q, y, m, available, fit_indices, arm, alpha)
                    total_fits += 1
                    predicted = _model_prediction(data, q, m, available, validation, model)
                    losses = _loss(predicted, y[validation])
                    oof_losses[inner_assignment == inner_fold] = losses
                    coverage = (
                        _coverage(data, q, m, available, validation, model)
                        if arm == "SHAM"
                        else None
                    )
                    inner.append(
                        {
                            "fold": inner_fold,
                            "alpha": alpha,
                            "validation_ids": ids[validation].tolist(),
                            "validation_participant_mse": losses.tolist(),
                            "validation_coverage": coverage,
                            "model": model,
                        }
                    )
                alpha_scores.append(
                    {
                        "alpha": alpha,
                        "participant_ids": ids[train].tolist(),
                        "participant_mse": oof_losses.tolist(),
                        "mean_participant_mse": float(oof_losses.mean()),
                    }
                )
            minimum = min(value["mean_participant_mse"] for value in alpha_scores)
            chosen = next(
                value["alpha"]
                for value in alpha_scores
                if value["mean_participant_mse"] <= minimum + TIE
            )
            model = _fit(data, q, y, m, available, train, arm, chosen)
            total_fits += 1
            predictions[arm_index, evaluate] = _model_prediction(
                data, q, m, available, evaluate, model
            )
            arms[arm] = {
                "selected_alpha": chosen,
                "alpha_scores": alpha_scores,
                "inner": inner,
                "final": model,
                "evaluation_coverage": _coverage(data, q, m, available, evaluate, model)
                if arm == "SHAM"
                else None,
            }
        outer_rows.append(
            {
                "fold": outer_fold,
                "fit_ids": ids[train].tolist(),
                "evaluation_ids": ids[evaluate].tolist(),
                "arms": arms,
            }
        )
    _require(total_fits == 120, "independent reconstruction must contain exactly120ridge fits")
    models = {
        "schema": "cfeg.source-channel-margin-probe.models.v1",
        "arms": list(ARMS),
        "ids": ids.tolist(),
        "fold_assignment": assignment.tolist(),
        "ridge_fit_count": total_fits,
        "outer": outer_rows,
    }
    return models, predictions


def _result(data, models, prediction, target, available):
    ids = data["ids"]
    assignment = _folds(ids)
    error = (prediction - prediction.mean(axis=-1, keepdims=True) - target[None]) ** 2
    losses = error.mean(axis=(2, 3, 4))
    means = losses.mean(axis=1)
    fold_means = np.stack([losses[:, assignment == fold].mean(axis=1) for fold in range(3)], axis=1)
    interface_means = error.mean(axis=(1, 3, 4))
    covered = (available.sum(axis=-1) >= 2).all(axis=-1)
    coverages = [outer["arms"]["SHAM"]["evaluation_coverage"] for outer in models["outer"]]
    comparisons = {}
    for name in ("Q", "Q2", "SHAM"):
        index = ARMS.index(name)
        difference = float(means[index] - means[2])
        fold_difference = fold_means[index] - fold_means[2]
        comparisons[name] = {
            "relative_mse_gain": difference / float(means[index]) if means[index] > 1e-8 else None,
            "absolute_mse_gain": difference,
            "fold_absolute_mse_gain": fold_difference.tolist(),
            "positive_outer_folds": int(np.count_nonzero(fold_difference > TIE)),
            "interface_absolute_mse_gain": (interface_means[index] - interface_means[2]).tolist(),
        }
    energy = float(np.mean(target**2))
    gates = {
        "target_variation": energy > 1e-8,
        "metadata_coverage": int(covered.sum()) >= 32,
        "sham_coverage": all(value["design_changed_fraction"] >= 0.8 for value in coverages),
        "comparator_headroom": all(means[ARMS.index(name)] > 1e-8 for name in comparisons),
        "relative_gain": all(
            value["relative_mse_gain"] is not None and value["relative_mse_gain"] >= 0.02
            for value in comparisons.values()
        ),
        "positive_outer_folds": all(
            value["positive_outer_folds"] >= 2 for value in comparisons.values()
        ),
        "positive_q_interfaces": bool(np.all(interface_means[0] - interface_means[2] > TIE)),
    }
    gates = {key: bool(value) for key, value in gates.items()}
    reasons = [
        reason
        for key, reason in (
            ("target_variation", "INSUFFICIENT_TARGET_VARIATION"),
            ("metadata_coverage", "INSUFFICIENT_METADATA_COVERAGE"),
            ("sham_coverage", "INSUFFICIENT_SHAM_DESIGN_COVERAGE"),
            ("comparator_headroom", "INSUFFICIENT_COMPARATOR_HEADROOM"),
            ("relative_gain", "RELATIVE_GAIN_BELOW_THRESHOLD"),
            ("positive_outer_folds", "INSUFFICIENT_POSITIVE_OUTER_FOLDS"),
            ("positive_q_interfaces", "NONPOSITIVE_Q_INTERFACE_GAIN"),
        )
        if not gates[key]
    ]
    if not gates["target_variation"]:
        terminal = "INSUFFICIENT_TARGET_VARIATION"
    elif not (gates["metadata_coverage"] and gates["sham_coverage"]):
        terminal = "UNINFORMATIVE_CONTROL"
    elif not gates["comparator_headroom"]:
        terminal = "INSUFFICIENT_COMPARATOR_HEADROOM"
    elif all(gates.values()):
        terminal = "PROMISING_PROBE"
    else:
        terminal = "NO_PROMISING_SIGNAL_IN_THIS_PROBE"
    return {
        "schema": "cfeg.source-channel-margin-probe.result.v1",
        "arms": list(ARMS),
        "ids": ids.tolist(),
        "fold_assignment": assignment.tolist(),
        "ridge_fit_count": 120,
        "terminal": terminal,
        "failure_reasons": reasons,
        "target_energy": energy,
        "metadata": {
            "available_channel_counts": available.sum(axis=-1).tolist(),
            "covered_participant_ids": ids[covered].tolist(),
            "covered_participant_count": int(covered.sum()),
            "minimum_covered_participants": 32,
        },
        "sham_outer_coverage": coverages,
        "participant_mse": losses.tolist(),
        "mean_mse": means.tolist(),
        "comparisons": comparisons,
        "fold_mean_mse": fold_means.tolist(),
        "interface_mean_mse": interface_means.tolist(),
        "participant_mse_range": np.stack(
            (losses.min(axis=1), losses.max(axis=1)), axis=-1
        ).tolist(),
        "gates": gates,
    }


def _compare(checks, actual, expected, path):
    """Exact schema/discrete identity with fixed absolute numerical tolerance."""
    if isinstance(expected, dict):
        _require(
            isinstance(actual, dict) and set(actual) == set(expected),
            path + ": exact object fields",
        )
        for key, value in expected.items():
            _compare(checks, actual[key], value, path + "." + key)
    elif isinstance(expected, list):
        _require(
            isinstance(actual, list) and len(actual) == len(expected), path + ": list inventory"
        )
        for index, value in enumerate(expected):
            _compare(checks, actual[index], value, path + "." + str(index))
    elif isinstance(expected, (bool, int, str)) or expected is None:
        _require(
            type(actual) is type(expected) and actual == expected,
            path + ": discrete value mismatch",
        )
    else:
        _require(
            isinstance(actual, (int, float)) and not isinstance(actual, bool),
            path + ": numerical scalar required",
        )
        # Selected candidate/recorded alpha is a discrete scientific choice.
        checks.equal(path, actual, expected, exact=path.endswith((".alpha", ".selected_alpha")))


def audit_probe(data, models, predictions, result):
    """Return a read-only replay receipt; caller separately enforces provenance.

    Raw arrays are mandatory even though the producer's pure-math tests may omit
    them. A scientifically negative result may audit PASS. Any mismatch returns
    FAIL/AUDIT_FAILURE; no revised candidate or prediction is published in place.
    """
    checks = _Checks()
    receipt = {
        "schema": SCHEMA,
        "status": "FAIL",
        "errors": [],
        "checks": {},
        "max_differences": {},
        "ridge_fits_replayed": 0,
        "recomputed_result": None,
        "limitations": list(LIMITATIONS),
    }
    try:
        data = _validate(data)
        q, target, m, available = _raw_bridge(data, checks)
        expected_models, expected_predictions = _nested(data, q, target, m, available)
        receipt["ridge_fits_replayed"] = 120
        _compare(checks, models, expected_models, "models")
        checks.checks["all120_independent_ridges_scalers_donors_ranks_and_nested_selection"] = True
        predictions = np.asarray(predictions)
        _require(
            predictions.dtype.kind == "f" and predictions.dtype.itemsize == 8,
            "predictions must be float64",
        )
        checks.equal("complete_OOF_predictions", predictions, expected_predictions)
        checks.checks["complete_participant_separated_OOF_prediction_grid"] = True
        expected_result = _result(data, expected_models, expected_predictions, target, available)
        receipt["recomputed_result"] = expected_result
        _compare(checks, result, expected_result, "result")
        checks.checks["participant_losses_coverage_and_terminal_precedence"] = True
        receipt["status"] = "PASS"
    except (ValueError, TypeError, KeyError, IndexError, ArithmeticError) as error:
        receipt["errors"].append(type(error).__name__ + ": " + str(error))
        receipt["terminal_override"] = "AUDIT_FAILURE"
    receipt["checks"] = checks.checks
    receipt["max_differences"] = checks.max_differences
    return receipt
