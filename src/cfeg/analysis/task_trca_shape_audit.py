"""Independent, read-only NumPy/SciPy checks for frozen task-shape v1.

No producer modules are imported.  These pure functions audit recorded feature,
support and score sufficient statistics; they do not independently establish raw
preprocessing, role-bound access, optimizer gradients or execution provenance.
Malformed or nonfinite evidence raises ValueError: callers must preserve a
VALIDITY_FAILURE receipt, never replace a failed cell with missing metadata.
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping

import numpy as np
from scipy import linalg, special, stats

ARMS = ("FULL", "ISO", "Q", "Q2", "QM", "SHAM_REFIT", "PERMUTED", "STALE", "MISSING")
IDS = (
    4,
    6,
    8,
    11,
    14,
    21,
    22,
    25,
    28,
    29,
    30,
    31,
    32,
    33,
    37,
    41,
    42,
    43,
    44,
    46,
    54,
    55,
    56,
    61,
    63,
    65,
    67,
    73,
    74,
    77,
    80,
    82,
    83,
    84,
    89,
    92,
    97,
    100,
    102,
)
WINDOWS = (125, 188, 250, 500)
SCORE_SHAPE = (39, 2, 4, 2, 9, 48, 12)


def _array(value, name, shape=None, *, finite=True):
    raw = np.asarray(value)
    if raw.dtype.kind not in "iuf":
        raise ValueError(f"{name} must be real numeric, not bool/complex/object")
    result = np.asarray(raw, dtype=np.float64)
    if shape is not None and result.shape != shape:
        raise ValueError(f"{name} must have shape {shape}")
    if finite and not np.isfinite(result).all():
        raise ValueError(f"{name} must be finite")
    return result


def _mask(value, shape, name="available"):
    raw = np.asarray(value)
    if raw.dtype.kind != "b" or raw.shape != shape:
        raise ValueError(f"{name} must be boolean with shape {shape}")
    return raw


def _scaler(record, dimensions, fit_ids):
    if not isinstance(record, Mapping) or record.get("fit_ids") != fit_ids:
        raise ValueError("Scaler fitting IDs must exactly match pipeline fitting IDs")
    mean = _array(record.get("mean"), "scaler mean", (dimensions,))
    scale = _array(record.get("scale"), "scaler scale", (dimensions,))
    if np.any(scale <= 0):
        raise ValueError("Scaler scales must be positive")
    return mean, scale


def _coefficients(record, dimensions):
    if not isinstance(record, Mapping):
        raise ValueError("Missing head record")  # noqa: TRY004 -- uniform evidence-validation API
    return _array(record.get("coefficients"), "coefficients", (dimensions + 1,))


def independent_prior(
    q, m, available, pipeline_record, arm, donor_m=None, stale_m=None, stale_available=None
):
    """Reconstruct R[5,8] from a producer Pipeline.record() and raw features.

    Q/MISSING do not inspect numerical m; Q2 likewise never reads m.  STALE
    requires independently prepared stale features/availability, retaining q.
    Donor partition/mask identity must be checked separately with donor records.
    """
    if arm not in ARMS[1:]:
        raise ValueError("Expected a positive-mass arm; FULL is not ISO")
    if arm == "ISO":
        return np.ones((5, 8), dtype=np.float64)
    q = _array(q, "q", (5, 8, 15))
    if not isinstance(pipeline_record, Mapping):
        raise ValueError("Expected pipeline record")  # noqa: TRY004 -- uniform evidence-validation API
    ids = pipeline_record.get("fit_ids")
    if (
        not isinstance(ids, list)
        or not ids
        or any(type(pid) is not int or pid <= 0 for pid in ids)
        or ids != sorted(set(ids))
    ):
        raise ValueError("Pipeline fit_ids must be sorted unique positive integers")
    qm, qs = _scaler(pipeline_record.get("q_scaler"), 15, ids)
    qc = _coefficients(pipeline_record.get("Q"), 15)
    digest = hashlib.sha256()
    for value in (qm, qs, qc):
        digest.update(np.asarray(value, dtype="<f8").tobytes())
    if pipeline_record.get("q_hash") != digest.hexdigest():
        raise ValueError("Frozen Q hash does not match its scaler/coefficients")
    with np.errstate(over="raise", invalid="raise", divide="raise"):
        try:
            logits = 0.8 * np.log(2) / 2 * np.tanh(((q - qm) / qs) @ qc[:-1] + qc[-1])
            if arm not in ("Q", "MISSING"):
                observed = _mask(available, (8,))
                if arm == "Q2":
                    raw = q[..., 1:3]
                    mean, scale = _scaler(pipeline_record.get("q2_scaler"), 2, ids)
                    coefficient = _coefficients(pipeline_record["residuals"].get("Q2"), 2)
                    mask = np.broadcast_to(observed, (5, 8))
                else:
                    source = donor_m if arm in ("SHAM_REFIT", "PERMUTED") else m
                    if arm == "STALE":
                        source = stale_m
                        stale = _mask(stale_available, (8,), "stale_available")
                        if np.any(stale & ~observed):
                            raise ValueError("First-packet stale availability cannot add channels")
                        observed = stale
                    raw = _array(source, "residual metadata", (8, 2))
                    mean, scale = _scaler(pipeline_record.get("m_scaler"), 2, ids)
                    head = "SHAM_REFIT" if arm == "SHAM_REFIT" else "QM"
                    coefficient = _coefficients(pipeline_record["residuals"].get(head), 2)
                    mask = observed
                standardized = np.where(mask[..., None], (raw - mean) / scale, 0.0)
                residual = (
                    0.2 * np.log(2) / 2 * np.tanh(standardized @ coefficient[:-1] + coefficient[-1])
                )
                logits = logits + residual * mask
            exponential = np.exp(logits - logits.max(axis=-1, keepdims=True))
            result = 8 * exponential / exponential.sum(axis=-1, keepdims=True)
        except (FloatingPointError, KeyError, TypeError) as error:
            raise ValueError("Invalid frozen prior arithmetic/record") from error
    _check_prior(result)
    return result


def _check_prior(r):
    r = _array(r, "r", (5, 8))
    if (
        np.any(r <= 0)
        or np.any(np.abs(r.sum(-1) - 8) > 8e-12)
        or np.any(r.max(-1) / r.min(-1) > 2 + 1e-12)
        or np.any(r > 16 / 9 + 1e-12)
    ):
        raise ValueError("R violates positive trace-eight/ratio-two/16/9 bounds")
    return r


def _symmetric(value, name, shape):
    value = _array(value, name, shape)
    error = np.linalg.norm(value - value.swapaxes(-1, -2), axis=(-2, -1))
    scale = np.linalg.norm(value, axis=(-2, -1))
    if (
        not np.isfinite(error).all()
        or not np.isfinite(scale).all()
        or np.any(error > 1e-12 * np.maximum(1, scale))
    ):
        raise ValueError(f"{name} must be symmetric within 1e-12 relative tolerance")
    return (value + value.swapaxes(-1, -2)) / 2


def independent_filters(s, c, anchors, r):
    """Solve 60 SciPy generalized eigensystems, independently of torch projector.

    s,c[5,12,8,8], native FULL anchors[5,12,8], r[5,8] -> w[5,8,12].
    Repeated lower roots are legal; a degenerate top, nonpositive C or unstable
    native sign anchor fails the entire call, with no numerical rescue.
    """
    s = _symmetric(s, "s", (5, 12, 8, 8))
    c = _symmetric(c, "c", (5, 12, 8, 8))
    anchor = _array(anchors, "anchors", (5, 12, 8))
    r = _check_prior(r)
    output = np.empty((5, 8, 12), dtype=np.float64)
    for band in range(5):
        for label in range(12):
            metric = c[band, label]
            try:
                minimum = float(linalg.eigvalsh(metric, check_finite=True)[0])
            except linalg.LinAlgError as error:
                raise ValueError("C eigensolver failed") from error
            if not np.isfinite(minimum) or minimum <= 0:
                raise ValueError("C must be positive definite")
            tau = 0.1 * minimum / (16 / 9)
            if not np.isfinite(tau) or tau <= 0:
                raise ValueError("Positive penalty mass is not representable")
            denominator = metric + tau * np.diag(r[band])
            try:
                values, vectors = linalg.eigh(
                    s[band, label], denominator, type=1, driver="gvd", check_finite=True
                )
            except linalg.LinAlgError as error:
                raise ValueError("Generalized eigensolver failed; no jitter") from error
            if not np.isfinite(values).all() or values[-1] - values[-2] <= 1e-10 * max(
                1.0, np.max(np.abs(values))
            ):
                raise ValueError("Top generalized eigenspace is degenerate")
            w = vectors[:, -1]
            a = anchor[band, label]
            norm2, anchor_norm2 = float(w @ metric @ w), float(a @ metric @ a)
            if not np.isfinite([norm2, anchor_norm2]).all() or norm2 <= 0 or anchor_norm2 <= 0:
                raise ValueError("Filter/native anchor C norm must be positive finite")
            inner = float(w @ metric @ a)
            cosine = abs(inner) / np.sqrt(norm2) / np.sqrt(anchor_norm2)
            if not np.isfinite(cosine) or cosine <= 1e-6:
                raise ValueError("Native C sign anchor is nearly orthogonal")
            output[band, :, label] = w / np.sqrt(norm2) * (1 if inner > 0 else -1)
    return _array(output, "independent filters", (5, 8, 12))


def independent_scores(w, statistics, weights):
    """Native signed linear-band ensemble Pearson from saved Gram statistics.

    Computes projected covariance traces and projected global-mean correction
    independently of the producer's channel-space J/H implementation.
    """
    w = _array(w, "filters", (5, 8, 12))
    weights = _array(weights, "weights", (5,))
    if np.any(weights <= 0):
        raise ValueError("Explicit native band weights must be positive")
    if not isinstance(statistics, Mapping):
        raise ValueError("statistics must be a mapping")  # noqa: TRY004 -- evidence-validation API
    n = np.asarray(statistics.get("query_mean")).shape
    if len(n) != 3 or n[0] < 1:
        raise ValueError("query_mean must have shape [n,5,8]")
    n = n[0]
    samples = statistics.get("samples")
    if type(samples) is not int or samples < 2:
        raise ValueError("samples must be integer >=2")
    mx = _array(statistics.get("query_mean"), "query_mean", (n, 5, 8))
    mt = _array(statistics.get("template_mean"), "template_mean", (12, 5, 8))
    qg = _symmetric(statistics.get("query_gram"), "query_gram", (n, 5, 8, 8))
    tg = _symmetric(statistics.get("template_gram"), "template_gram", (12, 5, 8, 8))
    cross = _array(statistics.get("cross_gram"), "cross_gram", (n, 12, 5, 8, 8))
    result = np.zeros((n, 12), dtype=np.float64)
    with np.errstate(over="raise", invalid="raise", divide="raise"):
        try:
            for band in range(5):
                filters = w[band]
                query_means = mx[:, band] @ filters
                template_means = mt[:, band] @ filters
                query_means -= query_means.mean(-1, keepdims=True)
                template_means -= template_means.mean(-1, keepdims=True)
                query_var = np.einsum("ik,nij,jk->n", filters, qg[:, band], filters)
                query_var += samples * np.square(query_means).sum(-1)
                template_var = np.einsum("ik,cij,jk->c", filters, tg[:, band], filters)
                template_var += samples * np.square(template_means).sum(-1)
                if np.any(query_var <= 0) or np.any(template_var <= 0):
                    raise ValueError("Nonpositive projected variance; no floor")
                dot = np.einsum("ik,ncij,jk->nc", filters, cross[:, :, band], filters)
                dot += samples * query_means @ template_means.T
                correlation = dot / np.sqrt(query_var[:, None]) / np.sqrt(template_var[None])
                result += weights[band] * np.clip(correlation, -1, 1)
        except FloatingPointError as error:
            raise ValueError("Invalid Gram scoring arithmetic") from error
    return _array(result, "independent scores", (n, 12))


def independent_metadata(packet):
    """Independent support-prefix M construction; 0 observed, only NaN absent."""
    packet = _array(packet, "packet", finite=False)
    if packet.shape not in ((3, 8), (5, 8)):
        raise ValueError("metadata must be an exact 3/5-block prefix")
    if np.isinf(packet).any() or np.any(packet[np.isfinite(packet)] < 0):
        raise ValueError("metadata must be nonnegative or NaN")
    available = np.isfinite(packet).any(0)
    result = np.zeros((8, 2), dtype=np.float64)
    for channel in np.flatnonzero(available):
        logged = np.log1p(packet[np.isfinite(packet[:, channel]), channel])
        result[channel] = logged.mean(), logged.std(ddof=0)
    if available.any():
        result[available, 0] -= result[available, 0].mean()
    return result, available


def independent_donors(ids, masks, orders, interface):
    """Sorted-ID cyclic donors within exact prefix-mask/order partitions."""
    ids = list(ids)
    if not ids or any(type(pid) is not int or pid <= 0 for pid in ids) or len(set(ids)) != len(ids):
        raise ValueError("IDs must be nonempty unique positive integers")
    masks = np.asarray(masks)
    if masks.dtype.kind != "b" or masks.shape not in ((len(ids), 3, 8), (len(ids), 5, 8)):
        raise ValueError("Donor masks must be boolean exact 3/5-prefix masks")
    order = np.asarray(orders)
    if (
        order.dtype.kind not in "iu"
        or order.shape != (len(ids),)
        or not np.isin(order, [0, 1]).all()
        or type(interface) is not int
        or interface not in (0, 1)
    ):
        raise ValueError("Orders/interface must be binary integers")
    groups = {}
    for index, pid in enumerate(ids):
        key = int(order[index]), tuple(masks[index].ravel().tolist())
        groups.setdefault(key, []).append(pid)
    donors = {}
    for group in groups.values():
        ordered = sorted(group)
        donors.update(
            {pid: ordered[(index + 1) % len(ordered)] for index, pid in enumerate(ordered)}
        )
    return donors


def _paired_counts(participant_counts):
    """Integer net correctness across 8x48 queries precedes floating inference."""
    value = np.asarray(participant_counts)
    if value.shape != (39,) or value.dtype.kind not in "iu":
        raise ValueError("Paired endpoint requires 39 integer participant differences")
    pp = value.astype(np.float64) * (100 / (8 * 48))
    mean = float(pp.mean())
    half = float(stats.t.ppf(0.975, 38) * pp.std(ddof=1) / np.sqrt(39))
    return {
        "mean_pp": mean,
        "ci_low_pp": mean - half,
        "ci_high_pp": mean + half,
        "df": 38,
        "interval_kind": "descriptive paired t, repeatedly exposed development",
        "participant_integer_net_correct": value.tolist(),
        "helped": int((value > 0).sum()),
        "tied": int((value == 0).sum()),
        "harmed": int((value < 0).sum()),
        "worst_participant_change_pp": float(pp.min()),
    }


def _weights(value, name):
    result = _array(value, name)
    if result.shape == (5,):
        result = np.broadcast_to(result, (2, 5))
    if result.shape != (2, 5) or np.any(result <= 0) or not np.isfinite(result.sum(-1)).all():
        raise ValueError(f"{name} must be positive explicit [2,5] interface weights")
    return result


def _nullable(array):
    return np.where(array < 0, None, array).tolist()


def summarize(scores, a0, coverage, actuation):
    """Independently recompute endpoints and the frozen terminal branch.

    coverage contains m, donor_m[39,2,8,2] and available[39,2,8], always k3.
    actuation requires band_weights and a0_band_weights (each [2,5] or shared
    [5]), qm_coefficients[3,3], and nonnegative r_max_abs_qm_minus_q and
    filter_max_abs_qm_minus_q[39,2,4,2] (flattened also accepted).  Maxima are
    descriptions, not a substitute for independent per-case R/filter checks.
    Optional validity_errors is a list of failures supplied by the artifact
    auditor; nonempty gives VALIDITY_FAILURE precedence. Optional
    all_budget_available[39,2,2,8] certifies absence across k3 AND k5; k3-only
    absence does not establish global dormancy. Malformed inputs raise.
    """
    scores = _array(scores, "scores", SCORE_SHAPE)
    a0 = _array(a0, "a0", (39, 2, 4, 48, 12))
    if not isinstance(coverage, Mapping) or not isinstance(actuation, Mapping):
        raise ValueError("Coverage and actuation must contain raw evidence")  # noqa: TRY004 -- validation API
    original_m = _array(coverage.get("m"), "coverage m", (39, 2, 8, 2))
    donor_m = _array(coverage.get("donor_m"), "coverage donor_m", (39, 2, 8, 2))
    available = _mask(coverage.get("available"), (39, 2, 8))
    if np.any(original_m[~available] != 0) or np.any(donor_m[~available] != 0):
        raise ValueError("Unavailable metadata features must be zero and donor-mask matched")
    changed = np.any(np.abs(original_m - donor_m) > 1e-12, axis=(2, 3))
    coverage_fraction = float(changed.sum() / 78)
    weights = _weights(actuation.get("band_weights"), "band_weights")
    a0_weights = _weights(actuation.get("a0_band_weights"), "a0_band_weights")
    coefficients = _array(actuation.get("qm_coefficients"), "qm_coefficients", (3, 3))
    maxima = {}
    for field in ("r_max_abs_qm_minus_q", "filter_max_abs_qm_minus_q"):
        value = _array(actuation.get(field), field)
        if value.shape not in ((39, 2, 4, 2), (39 * 2 * 4 * 2,)) or np.any(value < 0):
            raise ValueError(f"{field} requires 624 nonnegative case maxima")
        maxima[field] = value.reshape(39, 2, 4, 2)
    errors = actuation.get("validity_errors", [])
    if not isinstance(errors, list) or any(
        not isinstance(error, str) or not error for error in errors
    ):
        raise ValueError("validity_errors must be a list of nonempty failure strings")

    truth = np.tile(np.arange(12), 4)
    predicted, predicted_a0 = scores.argmax(-1), a0.argmax(-1)
    correct = (predicted == truth).sum(-1, dtype=np.int64)
    correct_a0 = (predicted_a0 == truth).sum(-1, dtype=np.int64)
    q, qm, q2, sham, full = (ARMS.index(arm) for arm in ("Q", "QM", "Q2", "SHAM_REFIT", "FULL"))
    if not np.array_equal(scores[..., q, :, :], scores[..., ARMS.index("MISSING"), :, :]):
        raise ValueError("MISSING must equal frozen Q scores exactly")

    def comparison(left, right):
        return _paired_counts((left - right).sum(axis=(1, 2), dtype=np.int64))

    qm3 = correct[..., 0, qm]
    comparisons = {
        "QM3_minus_Q3": comparison(qm3, correct[..., 0, q]),
        "QM3_minus_Q2_3": comparison(qm3, correct[..., 0, q2]),
        "QM3_minus_SHAM3": comparison(qm3, correct[..., 0, sham]),
        "QM3_minus_Q5": comparison(qm3, correct[..., 1, q]),
        "QM3_minus_FULL3": comparison(qm3, correct[..., 0, full]),
        "QM3_minus_A0": comparison(qm3, correct_a0),
        "QM5_minus_Q5": comparison(correct[..., 1, qm], correct[..., 1, q]),
    }
    primary = comparisons["QM3_minus_Q3"]
    costs = {}
    for arm, position in (("Q", q), ("QM", qm)):
        # -1 means unobserved attainment, never an assigned 60/84-label cost.
        cost = np.full((39, 2, 4), -1, dtype=np.int64)
        cost[correct[..., 1, position] >= 39] = 60
        cost[correct[..., 0, position] >= 39] = 36
        cost[correct_a0 >= 39] = 0
        costs[arm] = cost
    both = (costs["Q"] >= 0) & (costs["QM"] >= 0)
    new = (costs["Q"] < 0) & (costs["QM"] >= 0)
    lost = (costs["Q"] >= 0) & (costs["QM"] < 0)
    neither = (costs["Q"] < 0) & (costs["QM"] < 0)
    savings = costs["Q"][both] - costs["QM"][both]
    grid = {
        "cells": 312,
        "threshold_correct_of_48": 39,
        "both_attain": int(both.sum()),
        "new_attain": int(new.sum()),
        "lost_attainment": int(lost.sum()),
        "neither_attain": int(neither.sum()),
        "pooled_label_savings_total": int(savings.sum()) if len(savings) else None,
        "pooled_label_savings_mean": float(savings.mean()) if len(savings) else None,
        "q_cost_mean_among_both": float(costs["Q"][both].mean()) if len(savings) else None,
        "qm_cost_mean_among_both": float(costs["QM"][both].mean()) if len(savings) else None,
        "q_labels": _nullable(costs["Q"]),
        "qm_labels": _nullable(costs["QM"]),
    }
    metadata_checks = {
        "primary_mean_at_least_1pp": primary["mean_pp"] >= 1,
        "primary_ci_low_gt_0": primary["ci_low_pp"] > 0,
        "qm3_minus_q2_ci_low_gt_0": comparisons["QM3_minus_Q2_3"]["ci_low_pp"] > 0,
        "qm3_minus_sham_ci_low_gt_0": comparisons["QM3_minus_SHAM3"]["ci_low_pp"] > 0,
        "sham_coverage_at_least_half": coverage_fraction >= 0.5,
    }
    calibration_checks = {
        "qm3_minus_q5_ci_low_gt_minus_1pp": comparisons["QM3_minus_Q5"]["ci_low_pp"] > -1,
        "qm3_mean_accuracy_at_least_point8": bool(qm3.mean() / 48 >= 0.8),
        "qm3_minus_full3_ci_low_gt_minus_1pp": comparisons["QM3_minus_FULL3"]["ci_low_pp"] > -1,
        "qm3_minus_a0_ci_low_gt_minus_1pp": comparisons["QM3_minus_A0"]["ci_low_pp"] > -1,
        "harmed_not_more_than_helped": primary["harmed"] <= primary["helped"],
        "positive_observed_label_savings": bool(len(savings) and savings.sum() > 0),
        "new_attainment_not_less_than_lost": int(new.sum()) >= int(lost.sum()),
    }
    metadata_passed, calibration_passed = (
        all(metadata_checks.values()),
        all(calibration_checks.values()),
    )
    # Sufficient certificates only, not reverse inference from no score changes.
    zero_coefficients = bool(np.all(coefficients == 0))
    absent_metadata = False
    if "all_budget_available" in actuation:
        all_available = _mask(
            actuation["all_budget_available"], (39, 2, 2, 8), "all_budget_available"
        )
        if not np.array_equal(all_available[..., 0, :], available):
            raise ValueError("All-budget and coverage k3 availability disagree")
        if np.any(all_available[..., 0, :] & ~all_available[..., 1, :]):
            raise ValueError("Prefix availability must be monotone from k3 to k5")
        absent_metadata = not bool(all_available.any())
    structural = zero_coefficients or absent_metadata
    if errors:
        terminal = "VALIDITY_FAILURE"
    elif structural:
        terminal = "STRUCTURAL_NO_ACTUATION"
    elif not metadata_passed:
        terminal = "METADATA_INCREMENT_NOT_ESTABLISHED"
    elif not calibration_passed:
        terminal = "CLASSIFICATION_INCREMENT_ONLY"
    else:
        terminal = "DEVELOPMENT_CALIBRATION_BENEFIT_CANDIDATE"

    nll = np.empty((39, 2, 4, 2, 9), dtype=np.float64)
    a0_nll = np.empty((39, 2, 4), dtype=np.float64)
    for interface in range(2):
        logits = scores[:, interface] / weights[interface].sum() / 0.1
        true_scores = np.take_along_axis(
            logits, np.broadcast_to(truth, logits.shape[:-1])[..., None], -1
        )[..., 0]
        nll[:, interface] = (special.logsumexp(logits, axis=-1) - true_scores).mean(-1)
        logits0 = a0[:, interface] / a0_weights[interface].sum() / 0.1
        true0 = np.take_along_axis(
            logits0, np.broadcast_to(truth, logits0.shape[:-1])[..., None], -1
        )[..., 0]
        a0_nll[:, interface] = (special.logsumexp(logits0, axis=-1) - true0).mean(-1)
    if not np.isfinite(nll).all() or not np.isfinite(a0_nll).all():
        raise ValueError("Nonfinite NLL arithmetic")
    conditions = []
    for interface in range(2):
        for window, samples in enumerate(WINDOWS):
            conditions.append(
                {
                    "interface": ("dry", "wet")[interface],
                    "samples": samples,
                    "accuracy": (correct[:, interface, window].mean(0) / 48).tolist(),
                    "nll": nll[:, interface, window].mean(0).tolist(),
                    "a0_accuracy": float(correct_a0[:, interface, window].mean() / 48),
                    "a0_nll": float(a0_nll[:, interface, window].mean()),
                }
            )
    actuation_report = {
        "structural_no_actuation": structural,
        "structural_certificate": (
            "all outer QM residual coefficients exactly zero"
            if zero_coefficients
            else "no k3 or k5 metadata available"
            if absent_metadata
            else "none; no argmax change does not establish dormancy"
        ),
        "r_filter_maxima_are_descriptive_not_independent_proof": True,
        "by_k": {},
    }
    for budget, k in enumerate((3, 5)):
        qscore, mscore = scores[..., budget, q, :, :], scores[..., budget, qm, :, :]
        qp, mp = predicted[..., budget, q, :], predicted[..., budget, qm, :]
        flipped = qp != mp
        change = correct[..., budget, qm] - correct[..., budget, q]
        delta_score = np.abs(mscore - qscore)
        sorted_q = np.sort(qscore, axis=-1)
        margin = sorted_q[..., -1] - sorted_q[..., -2]
        case = {
            "score_max_abs": float(delta_score.max()),
            "score_changed_cells_exact": int(np.any(delta_score != 0, axis=(-2, -1)).sum()),
            "score_changed_cells_above_1e_12": int(
                np.any(delta_score > 1e-12, axis=(-2, -1)).sum()
            ),
            "argmax_changed_queries": int(flipped.sum()),
            "changed_to_correct": int((flipped & (mp == truth)).sum()),
            "changed_from_correct": int((flipped & (qp == truth)).sum()),
            "changed_wrong_to_different_wrong": int(
                (flipped & (qp != truth) & (mp != truth)).sum()
            ),
            "helped_cells": int((change > 0).sum()),
            "tied_cells": int((change == 0).sum()),
            "harmed_cells": int((change < 0).sum()),
            "worst_cell_integer_change": int(change.min()),
            "cell_integer_change": change.tolist(),
            "q_top_two_native_margin_min": float(margin.min()),
            "q_top_two_native_margin_median": float(np.median(margin)),
            "q_top_two_native_margins_at_most_1e_12": int((margin <= 1e-12).sum()),
        }
        for name, value in maxima.items():
            selected = value[..., budget]
            case[name] = {
                "maximum": float(selected.max()),
                "changed_cells_exact": int((selected != 0).sum()),
                "changed_cells_above_1e_12": int((selected > 1e-12).sum()),
            }
        actuation_report["by_k"][str(k)] = case
    return {
        "schema": "cfeg.task_trca_shape.independent_summary.v1",
        "terminal": terminal,
        "evidence_scope": "source39 repeatedly exposed development; no held60 or confirmatory claim",
        "ids": list(IDS),
        "arms": list(ARMS),
        "budgets": [3, 5],
        "windows": list(WINDOWS),
        "correct_counts": correct.tolist(),
        "a0_correct_counts": correct_a0.tolist(),
        "accuracy_by_budget_arm": (correct.mean(axis=(0, 1, 2)) / 48).tolist(),
        "nll_by_budget_arm": nll.mean(axis=(0, 1, 2)).tolist(),
        "a0_accuracy": float(correct_a0.mean() / 48),
        "a0_nll": float(a0_nll.mean()),
        "nll_note": "native score / its explicit interface band-weight sum / fixed .1; not calibrated probabilities",
        "comparisons": comparisons,
        "conditions": conditions,
        "grid_attainment": grid,
        "coverage": {
            "changed_units": int(changed.sum()),
            "total_units": 78,
            "fraction": coverage_fraction,
            "changed_by_participant_interface": changed.tolist(),
        },
        "metadata_increment_checks": metadata_checks,
        "metadata_increment_passed": metadata_passed,
        "calibration_additional_checks": calibration_checks,
        "calibration_additional_passed": calibration_passed,
        "actuation": actuation_report,
        "validity_errors": errors,
        "all_terminal_branches_close_candidate": True,
        "automatic_held60_promotion": False,
    }
