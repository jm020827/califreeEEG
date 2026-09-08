"""Independent saved-array verification for one frozen source39 development study.

No producer, decoder, preprocessing, or project learning module is imported.
Pure helpers accept artificial arrays in tests; the CLI is bound to exact receipts.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import stat
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from scipy import stats

PLAN_SHA256 = "ed67cc1f361ed89c37b6f7c5df1e9170b78484e428a5105020485c8c03e3934b"
ARMS = ("FULL", "ISO", "Q", "Q2", "QM", "SHAM_REFIT", "PERMUTED", "STALE", "MISSING")
ORIGINAL_NATIVE = Path("/home/whwovy/metadata-prior-source39-v1/native")
RECOVERED_NATIVE = Path("/home/whwovy/metadata-prior-source39-v1/native-cold-r1")
RECOVERY = {
    "original_root": str(ORIGINAL_NATIVE),
    "start_sha256": "0af0f7d02e7a5b31494133fb1a99924bbcc55c7713df49028479552ac15df1c7",
    "preserved_artifact_count": 39,
    "reason": "infrastructure_terminal_source_rehash_allowlist",
    "unchanged_plan_sha256": PLAN_SHA256,
    "actual_native_root": str(RECOVERED_NATIVE),
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def finite(value, name):
    raw = np.asarray(value)
    require(raw.dtype.kind in "fiu", f"Real numeric {name}")
    with np.errstate(over="ignore", invalid="ignore"):
        out = np.asarray(raw, dtype=np.float64)
    require(np.isfinite(out).all(), f"Finite {name}")
    return out


def close(actual, expected, name, *, atol=1e-12, rtol=1e-12):
    a, b = finite(actual, name), finite(expected, name)
    require(a.shape == b.shape, f"Shape {name}")
    require(np.allclose(a, b, rtol=rtol, atol=atol), f"Mismatch {name}")


def paired_stats(values):
    """Participant-level descriptive two-sided Student t interval, not an iid claim."""
    x = finite(values, "participant differences")
    require(x.ndim == 1 and len(x) >= 2, "At least two participant differences")
    mean = float(np.mean(x))
    standard_error = float(np.std(x, ddof=1) / np.sqrt(len(x)))
    radius = float(stats.t.ppf(0.975, len(x) - 1) * standard_error)
    require(np.isfinite([mean, standard_error, radius]).all(), "Finite paired statistics")
    return {
        "n": len(x),
        "mean": mean,
        "lower": mean - radius,
        "upper": mean + radius,
        "help": int(np.count_nonzero(x > 0)),
        "tie": int(np.count_nonzero(x == 0)),
        "harm": int(np.count_nonzero(x < 0)),
    }


def accuracy_arrays(scores, anchors, plan):
    """Independently classify the declared class-balanced, chronological query grid."""
    s, a = finite(scores, "scores"), finite(anchors, "a0_scores")
    subjects = len(plan["source_subject_ids"])
    interfaces, windows = len(plan["interfaces"]), len(plan["sample_counts"])
    queries, classes = plan["endpoints"]["query_count"], len(plan["frequencies"])
    require(tuple(plan["arms"]) == ARMS, "Frozen arm order")
    require(plan["budgets"] == [3, 5], "Frozen support budgets")
    require(queries == classes * len(plan["query_blocks"]), "Balanced query geometry")
    require(
        s.shape == (subjects, interfaces, windows, 2, len(ARMS), queries, classes),
        "Saved score shape",
    )
    require(a.shape == (subjects, interfaces, windows, queries, classes), "Saved A0 shape")
    truth = np.tile(np.arange(classes), len(plan["query_blocks"]))
    predictions, anchor_predictions = s.argmax(-1), a.argmax(-1)
    return (
        np.mean(predictions == truth, axis=-1),
        np.mean(anchor_predictions == truth, axis=-1),
        predictions,
        anchor_predictions,
    )


def cost_arrays(accuracy, anchor_accuracy, plan):
    """First observed attainment; failure stays unavailable, never assigned a fake cost."""
    acc, a0 = finite(accuracy, "accuracy"), finite(anchor_accuracy, "anchor accuracy")
    require(acc.shape[:-2] == a0.shape and acc.shape[-2:] == (2, len(ARMS)), "Cost geometry")
    threshold = plan["endpoints"]["attainment_threshold"]
    require(plan["endpoints"]["attainment_grid"] == [0, 3, 5], "Attainment grid")
    output = np.full(acc.shape[:-2] + (len(ARMS),), np.nan)
    for arm in range(len(ARMS)):
        for budget_index, budget in reversed(list(enumerate(plan["budgets"]))):
            output[..., arm] = np.where(
                acc[..., budget_index, arm] >= threshold, budget, output[..., arm]
            )
        output[..., arm] = np.where(a0 >= threshold, 0, output[..., arm])
    costs = np.full_like(output, np.nan)
    for budget in plan["endpoints"]["attainment_grid"]:
        costs[output == budget] = plan["endpoints"]["label_costs"][str(budget)]
    return output, costs


def channel_design(q, representation):
    x = finite(q, "Q features")
    require(x.ndim >= 3 and x.shape[-2] >= 2, "Q channel geometry")
    require(representation in ("local", "context"), "Q representation")
    if representation == "local":
        return x.copy()
    means = np.broadcast_to(x.mean(axis=-2, keepdims=True), x.shape)
    return np.concatenate((x, means), axis=-1)


def independent_ridge(features, targets, alpha):
    """Independent standardized mean-loss ridge using Gram solve or pseudoinverse."""
    x, y = finite(features, "ridge X"), finite(targets, "ridge y")
    require(x.ndim == 2 and min(x.shape) > 0 and y.shape == (len(x),), "Ridge geometry")
    require(np.isfinite(alpha) and alpha >= 0, "Ridge alpha")
    mean, scale = x.mean(0), x.std(0)
    scale = np.where(scale < 1e-12, 1.0, scale)
    z = (x - mean) / scale
    intercept = float(y.mean())
    centered = y - intercept
    if alpha:
        coefficient = np.linalg.solve(
            z.T @ z / len(x) + alpha * np.eye(x.shape[-1]), z.T @ centered / len(x)
        )
    else:
        cutoff = max(len(x) + x.shape[-1], x.shape[-1]) * np.finfo(float).eps
        coefficient = np.linalg.pinv(z, rcond=cutoff) @ centered
    require(np.isfinite(coefficient).all(), "Finite independent ridge")
    return {
        "mean": mean,
        "scale": scale,
        "coefficient": coefficient,
        "intercept": intercept,
        "alpha": float(alpha),
        "design_rank": int(np.linalg.matrix_rank(z / np.sqrt(len(x)))),
    }


def predict_record(record, features):
    x = finite(features, "prediction inputs")
    mean, scale, coefficient = (
        finite(record[key], f"ridge {key}") for key in ("mean", "scale", "coefficient")
    )
    require(mean.ndim == 1 and mean.shape == scale.shape == coefficient.shape, "Ridge vectors")
    require(x.shape[-1] == len(mean) and np.all(scale > 0), "Ridge prediction geometry")
    answer = ((x - mean) / scale) @ coefficient + record["intercept"]
    return finite(answer, "ridge prediction")


def verify_ridge(record, features, targets, *, alpha=None):
    if alpha is not None:
        close(record["alpha"], alpha, "Required ridge alpha", atol=0, rtol=0)
    calculated = independent_ridge(features, targets, record["alpha"])
    for key in ("mean", "scale", "coefficient", "intercept"):
        close(record[key], calculated[key], f"Selected ridge {key}", atol=1e-8, rtol=1e-8)
    require(record["design_rank"] == calculated["design_rank"], "Selected design rank")
    return calculated


def verify_features(features, plan):
    require(set(features) == {"q", "target", "z", "order"}, "Exact feature inventory")
    p = len(plan["source_subject_ids"])
    cells = len(plan["interfaces"]) * len(plan["sample_counts"]) * 5
    q, y = finite(features["q"], "q"), finite(features["target"], "target")
    z, order = np.asarray(features["z"]), np.asarray(features["order"])
    require(q.shape == (p, 2, cells, 8, 15), "Q shape")
    require(y.shape == q.shape[:-1], "Target shape")
    require(z.shape == (p, 2, 10, 8) and z.dtype.kind == "f", "M packet shape/dtype")
    require(not np.isinf(z).any() and np.all(z[np.isfinite(z)] >= 0), "Raw M nonnegative/NaN")
    require(order.shape == (p,) and np.isin(order, [0, 1]).all(), "Headband order bits")
    for budget_index, k in enumerate(plan["budgets"]):
        close(q[:, budget_index, :, :, 3], np.full((p, cells, 8), np.log(k)), "Support log budget")
        for cell in range(cells):
            interface = cell // (len(plan["sample_counts"]) * 5)
            expected_mask = np.isfinite(z[:, interface, :k]).mean(axis=1)
            close(q[:, budget_index, cell, :, 4], expected_mask, "Common Q M availability")
            close(
                q[:, budget_index, cell, :, 5], np.broadcast_to(order[:, None], (p, 8)), "Q order"
            )
            close(
                q[:, budget_index, cell, :, 6],
                np.broadcast_to((order != interface).astype(int)[:, None], (p, 8)),
                "Q period",
            )
            close(
                q[:, budget_index, cell, :, 7:],
                np.broadcast_to(np.eye(8), (p, 8, 8)),
                "Q channel IDs",
            )
    return q, y, z, order


def metadata_features(packets):
    raw = np.asarray(packets, dtype=float)
    require(raw.ndim == 3 and raw.shape[-1] == 8, "M [participant,prefix,channel] geometry")
    require(not np.isinf(raw).any() and np.all(raw[np.isfinite(raw)] >= 0), "Valid M packets")
    valid = np.isfinite(raw)
    counts = valid.sum(axis=1)
    logged = np.log1p(np.where(valid, raw, 0))
    mean = logged.sum(axis=1) / np.maximum(counts, 1)
    available = counts > 0
    center = np.sum(mean * available, axis=-1, keepdims=True) / np.maximum(
        available.sum(axis=-1, keepdims=True), 1
    )
    variance = np.sum(np.where(valid, (logged - mean[:, None]) ** 2, 0), axis=1) / np.maximum(
        counts, 1
    )
    values = np.stack((mean - center, np.sqrt(variance)), axis=-1)
    return np.where(available[..., None], values, 0), available


def donor_map(z, order, indices, interface, k, subject_ids):
    groups = {}
    for index in sorted(map(int, indices), key=lambda i: subject_ids[i]):
        key = (int(order[index]), tuple(np.isfinite(z[index, interface, :k]).ravel()))
        groups.setdefault(key, []).append(index)
    mapping = {}
    for members in groups.values():
        mapping.update({p: members[(i + 1) % len(members)] for i, p in enumerate(members)})
    return np.array([mapping[int(i)] for i in indices], dtype=int)


def verify_selection(receipt, features, targets, subject_ids):
    """Check split/selection receipts and replay the selected fit, not every CV candidate."""
    ids = np.asarray(subject_ids)
    require(receipt["participant_ids"] == ids.tolist(), "Selection participant inventory")
    assignment = np.empty(len(ids), dtype=int)
    assignment[np.argsort(ids)] = np.arange(len(ids)) % 3
    require(len(receipt["splits"]) == 3, "Three inner splits")
    for fold, split in enumerate(receipt["splits"]):
        require(split["fold"] == fold, "Inner fold order")
        require(split["fit_ids"] == ids[assignment != fold].tolist(), "Inner train IDs")
        require(split["validation_ids"] == ids[assignment == fold].tolist(), "Inner held IDs")
    expected_order = [
        (rep, alpha) for rep in ("local", "context") for alpha in (1, 0.1, 0.01, 0.001, 0.0001, 0)
    ]
    candidates = receipt["candidates"]
    require(
        [(r["representation"], r["alpha"]) for r in candidates] == expected_order,
        "All Q candidates/order",
    )
    for row in candidates:
        losses = finite(row["participant_mse"], "Candidate participant losses")
        require(losses.shape == (len(ids),) and np.all(losses >= 0), "Candidate losses")
        close(row["mean_participant_mse"], losses.mean(), "Equal participant loss")
    minimum = min(r["mean_participant_mse"] for r in candidates)
    selected_row = next(r for r in candidates if r["mean_participant_mse"] <= minimum + 1e-12)
    selected = receipt["selected"]
    require(selected["representation"] == selected_row["representation"], "Chosen Q representation")
    require(selected["ridge"]["alpha"] == selected_row["alpha"], "Chosen Q alpha")
    require(selected["feature_count"] == features.shape[-1], "Q feature count")
    design = channel_design(features, selected["representation"])
    verify_ridge(selected["ridge"], design.reshape(-1, design.shape[-1]), targets.ravel())
    return selected


def verify_q_receipt(receipt, q, target, ids):
    assignment = np.empty(len(ids), dtype=int)
    assignment[np.argsort(ids)] = np.arange(len(ids)) % 3
    oof = np.empty_like(target)
    require(len(receipt["oof_folds"]) == 3, "Three nuisance OOF folds")
    sizes = []
    for fold, row in enumerate(receipt["oof_folds"]):
        fit, held = assignment != fold, assignment == fold
        require(row["fold"] == fold, "OOF fold order")
        require(row["fit_ids"] == ids[fit].tolist(), "OOF fit IDs")
        require(row["oof_ids"] == ids[held].tolist(), "OOF held IDs")
        model = verify_selection(row["inner_selection"], q[fit], target[fit], ids[fit])
        oof[held] = predict_record(model["ridge"], channel_design(q[held], model["representation"]))
        sizes.append(int(fit.sum()))
    require(
        receipt["training_size_mismatch"]
        == {"oof_fit_participants": sizes, "final_fit_participants": len(ids)},
        "OOF/final training-size receipt",
    )
    final = verify_selection(receipt["final_selection"], q, target, ids)
    return final, oof


def normalize_prior(raw):
    x = finite(raw, "Positive prior")
    require(np.all(x > 0), "Strictly positive prior")
    return x / x.mean(axis=-1, keepdims=True)


def verify_freezes(features, freeze, plan):
    q, targets, z, order = verify_features(features, plan)
    ids = np.asarray(plan["source_subject_ids"])
    require(freeze["plan_sha256"] == PLAN_SHA256, "Freeze plan binding")
    require(freeze["study_id"] == plan["study_id"], "Freeze study binding")
    require(freeze["schema"] == "cfeg.metadata-prior-source.freezes.v1", "Freeze schema")
    require([f["fold"] for f in freeze["folds"]] == [0, 1, 2], "Outer fold inventory")
    output_shape = q.shape[:-1] + (len(ARMS),)
    prior = np.empty(output_shape)
    proxy = np.full(output_shape, np.nan)
    cells = q.shape[2]
    changed, denominator, selected_fits = 0, 0, 0
    for fold in freeze["folds"]:
        f = fold["fold"]
        fit, ev = (
            np.flatnonzero(np.arange(len(ids)) % 3 != f),
            np.flatnonzero(np.arange(len(ids)) % 3 == f),
        )
        require(
            fold["fit_ids"] == ids[fit].tolist() and fold["eval_ids"] == ids[ev].tolist(),
            "Outer identities",
        )
        train_maps, eval_maps = np.asarray(fold["train_maps"]), np.asarray(fold["eval_maps"])
        require(
            train_maps.dtype.kind in "iu" and eval_maps.dtype.kind in "iu", "Integer donor indices"
        )
        require(
            train_maps.shape == (2, 2, len(fit)) and eval_maps.shape == (2, 2, len(ev)),
            "Donor map shape",
        )
        for interface in range(2):
            for bi, k in enumerate(plan["budgets"]):
                expected_train = donor_map(z, order, fit, interface, k, ids)
                expected_eval = donor_map(z, order, ev, interface, k, ids)
                require(
                    np.array_equal(train_maps[interface, bi], expected_train),
                    "Training donor strata/cycle",
                )
                require(
                    np.array_equal(eval_maps[interface, bi], expected_eval),
                    "Evaluation donor strata/cycle",
                )
                if k == 3:
                    real = metadata_features(z[ev, interface, :k])[0]
                    perm = metadata_features(z[expected_eval, interface, :k])[0]
                    changed += int(np.any(abs(real - perm) > 1e-12, axis=(1, 2)).sum())
                    denominator += len(ev)
        require([c["cell"] for c in fold["cells"]] == list(range(cells)), "Fitted cell inventory")
        for cell in fold["cells"]:
            ci = cell["cell"]
            interface = ci // (len(plan["sample_counts"]) * 5)
            qfit, yfit = q[fit, :, ci : ci + 1], targets[fit, :, ci : ci + 1]
            selected, oof = verify_q_receipt(cell["q_receipt"], qfit, yfit, ids[fit])
            require(cell["Q"] == selected, "Final Q record matches nuisance receipt")
            close(cell["oof_prediction"], oof, "OOF predictions", atol=1e-8, rtol=1e-8)
            residual = yfit - oof
            q2x = channel_design(qfit, selected["representation"])
            mfit = np.stack(
                [metadata_features(z[fit, interface, :k])[0] for k in plan["budgets"]], axis=1
            )[:, :, None]
            shamfit = np.stack(
                [
                    metadata_features(z[train_maps[interface, bi], interface, :k])[0]
                    for bi, k in enumerate(plan["budgets"])
                ],
                axis=1,
            )[:, :, None]
            require(
                set(cell["residuals"]) == {"Q2", "QM", "SHAM_REFIT"}, "Residual model inventory"
            )
            for name, design in (("Q2", q2x), ("QM", mfit), ("SHAM_REFIT", shamfit)):
                verify_ridge(
                    cell["residuals"][name],
                    design.reshape(-1, design.shape[-1]),
                    residual.ravel(),
                    alpha=0.1,
                )
            selected_fits += 7
            qeval = q[ev, :, ci : ci + 1]
            base_prediction = predict_record(
                selected["ridge"], channel_design(qeval, selected["representation"])
            )[:, :, 0]
            base_prior = normalize_prior(np.exp(np.clip(base_prediction, -3, 3)))
            for bi, k in enumerate(plan["budgets"]):
                real, available = metadata_features(z[ev, interface, :k])
                perm, perm_available = metadata_features(z[eval_maps[interface, bi], interface, :k])
                stale, stale_available = metadata_features(
                    np.repeat(z[ev, interface, :1], k, axis=1)
                )
                residual_inputs = {
                    "Q2": (
                        predict_record(
                            cell["residuals"]["Q2"],
                            channel_design(qeval, selected["representation"])[:, bi, 0],
                        ),
                        np.ones_like(available),
                    ),
                    "QM": (predict_record(cell["residuals"]["QM"], real), available),
                    "SHAM_REFIT": (
                        predict_record(cell["residuals"]["SHAM_REFIT"], perm),
                        available & perm_available,
                    ),
                    "PERMUTED": (
                        predict_record(cell["residuals"]["QM"], perm),
                        available & perm_available,
                    ),
                    "STALE": (
                        predict_record(cell["residuals"]["QM"], stale),
                        available & stale_available,
                    ),
                    "MISSING": (np.zeros_like(base_prediction[:, bi]), np.zeros_like(available)),
                }
                for arm, name in enumerate(ARMS):
                    if name in ("FULL", "ISO"):
                        computed = np.ones_like(base_prior[:, bi])
                    elif name == "Q":
                        computed = base_prior[:, bi]
                        proxy[ev, bi, ci, :, arm] = base_prediction[:, bi]
                    else:
                        delta, mask = residual_inputs[name]
                        clipped = np.where(mask, np.clip(delta, -0.2, 0.2), 0)
                        computed = normalize_prior(base_prior[:, bi] * np.exp(clipped))
                        computed = np.where(
                            ~mask.any(axis=-1, keepdims=True), base_prior[:, bi], computed
                        )
                        proxy[ev, bi, ci, :, arm] = base_prediction[:, bi] + clipped
                    prior[ev, bi, ci, :, arm] = computed
    require(denominator == len(ids) * 2, "Coverage primary denominator")
    return {
        "priors": prior,
        "proxy": proxy,
        "coverage_changed_fraction": changed / denominator,
        "selected_fit_replays": selected_fits,
    }


def expected_statistics(accuracy, anchor_accuracy, predictions, anchor_predictions, plan, coverage):
    """Reconstruct participant contrasts and observed-grid costs without producer functions."""
    ids, interfaces, windows = plan["source_subject_ids"], plan["interfaces"], plan["sample_counts"]
    arm_index = {name: index for index, name in enumerate(ARMS)}
    q_index, qm_index = arm_index["Q"], arm_index["QM"]
    summaries = []
    for ei, interface in enumerate(interfaces):
        for wi, n_samples in enumerate(windows):
            summaries.append(
                {
                    "interface": interface,
                    "n_samples": n_samples,
                    "k": 0,
                    "arm": "A0",
                    "accuracy": float(anchor_accuracy[:, ei, wi].mean()),
                    "help_vs_Q": None,
                    "tie_vs_Q": None,
                    "harm_vs_Q": None,
                    "prediction_changes_vs_Q": None,
                }
            )
            for bi, k in enumerate(plan["budgets"]):
                for ai, arm in enumerate(ARMS):
                    differences = accuracy[:, ei, wi, bi, ai] - accuracy[:, ei, wi, bi, q_index]
                    summaries.append(
                        {
                            "interface": interface,
                            "n_samples": n_samples,
                            "k": k,
                            "arm": arm,
                            "accuracy": float(accuracy[:, ei, wi, bi, ai].mean()),
                            "help_vs_Q": int((differences > 0).sum()),
                            "tie_vs_Q": int((differences == 0).sum()),
                            "harm_vs_Q": int((differences < 0).sum()),
                            "prediction_changes_vs_Q": int(
                                np.count_nonzero(
                                    predictions[:, ei, wi, bi, ai]
                                    != predictions[:, ei, wi, bi, q_index]
                                )
                            ),
                        }
                    )
    contrasts = []
    scopes = [("ALL", None, None)] + [
        (f"{e}_{n}", ei, wi) for ei, e in enumerate(interfaces) for wi, n in enumerate(windows)
    ]
    for scope, ei, wi in scopes:
        for bi, k in enumerate(plan["budgets"]):
            actual = accuracy[..., bi, qm_index]
            for comparator in ("Q", "Q2", "SHAM_REFIT", "FULL", "A0") + (("Q5",) if k == 3 else ()):
                if comparator == "A0":
                    baseline = anchor_accuracy
                elif comparator == "Q5":
                    baseline = accuracy[..., 1, q_index]
                else:
                    baseline = accuracy[..., bi, arm_index[comparator]]
                delta = actual - baseline
                values = delta.mean(axis=(1, 2)) if scope == "ALL" else delta[:, ei, wi]
                stat = paired_stats(values)
                contrasts.append(
                    {
                        "scope": scope,
                        "k": k,
                        "comparator": comparator,
                        "mean_delta": stat["mean"],
                        "ci_low": stat["lower"],
                        "ci_high": stat["upper"],
                        "help": stat["help"],
                        "tie": stat["tie"],
                        "harm": stat["harm"],
                        "n": stat["n"],
                    }
                )
    first, costs = cost_arrays(accuracy, anchor_accuracy, plan)
    attainment = []
    transitions = {}
    both, new, lost, neither, cost_deltas = 0, 0, 0, 0, []
    for pi, subject in enumerate(ids):
        for ei, interface in enumerate(interfaces):
            for wi, n_samples in enumerate(windows):
                for ai, arm in enumerate(ARMS):
                    finite_cost = np.isfinite(first[pi, ei, wi, ai])
                    attainment.append(
                        {
                            "participant": subject,
                            "interface": interface,
                            "n_samples": n_samples,
                            "arm": arm,
                            "first_k": int(first[pi, ei, wi, ai]) if finite_cost else None,
                            "labels": int(costs[pi, ei, wi, ai]) if finite_cost else None,
                        }
                    )
                qk, mk = first[pi, ei, wi, q_index], first[pi, ei, wi, qm_index]
                qr, mr = bool(np.isfinite(qk)), bool(np.isfinite(mk))
                key = f"{int(qk) if qr else None}->{int(mk) if mr else None}"
                transitions[key] = transitions.get(key, 0) + 1
                both += int(qr and mr)
                new += int(not qr and mr)
                lost += int(qr and not mr)
                neither += int(not qr and not mr)
                if qr and mr:
                    cost_deltas.append(
                        float(costs[pi, ei, wi, qm_index] - costs[pi, ei, wi, q_index])
                    )
    calibration = {
        "both_reached": both,
        "new_reach": new,
        "lost_reach": lost,
        "neither_reached": neither,
        "mean_label_delta_both": float(np.mean(cost_deltas)) if cost_deltas else None,
        "transitions": transitions,
    }
    lookup = {(r["scope"], r["k"], r["comparator"]): r for r in contrasts}
    primary = lookup["ALL", 3, "Q"]
    mean_qm = float(accuracy[..., 0, qm_index].mean())
    metadata_increment = bool(
        primary["mean_delta"] >= plan["endpoints"]["practical_delta"]
        and primary["ci_low"] > 0
        and lookup["ALL", 3, "Q2"]["ci_low"] > 0
        and lookup["ALL", 3, "SHAM_REFIT"]["ci_low"] > 0
        and coverage >= 0.5
    )
    calibration_benefit = bool(
        metadata_increment
        and lookup["ALL", 3, "Q5"]["ci_low"] > -plan["endpoints"]["ni_margin"]
        and mean_qm >= plan["endpoints"]["attainment_threshold"]
        and calibration["mean_label_delta_both"] is not None
        and calibration["mean_label_delta_both"] < 0
        and new >= lost
    )
    status = (
        "DEVELOPMENT_CALIBRATION_BENEFIT_CANDIDATE"
        if calibration_benefit
        else "CLASSIFICATION_INCREMENT_ONLY"
        if metadata_increment
        else "METADATA_INCREMENT_NOT_ESTABLISHED"
    )
    verdict = {
        "status": status,
        "metadata_increment": metadata_increment,
        "calibration_benefit": calibration_benefit,
        "primary_mean_QM3": mean_qm,
        "coverage_changed_fraction": coverage,
    }
    return summaries, contrasts, attainment, calibration, verdict


def verify_mapping(actual, expected, name, *, atol=1e-12):
    for key, value in expected.items():
        require(key in actual, f"Missing {name}.{key}")
        if isinstance(value, dict):
            require(set(actual[key]) == set(value), f"Exact mapping {name}.{key}")
            verify_mapping(actual[key], value, f"{name}.{key}", atol=atol)
        elif isinstance(value, list):
            require(actual[key] == value, f"Mismatch {name}.{key}")
        elif value is None or isinstance(value, (str, bool, int)):
            require(type(actual[key]) is type(value), f"Type {name}.{key}")
            require(actual[key] == value, f"Mismatch {name}.{key}")
        else:
            close(actual[key], value, f"{name}.{key}", atol=atol)


def indexed_rows(rows, fields, name):
    lookup = {tuple(row[field] for field in fields): row for row in rows}
    require(len(lookup) == len(rows), f"Unique {name}")
    return lookup


def verify_results(scores_cache, features, freeze, result, plan):
    require(set(scores_cache) == {"scores", "a0_scores"}, "Exact score inventory")
    accuracy, a0_accuracy, predictions, a0_predictions = accuracy_arrays(
        scores_cache["scores"], scores_cache["a0_scores"], plan
    )
    replay = verify_freezes(features, freeze, plan)
    require(result["study_id"] == plan["study_id"], "Result study")
    require(result["schema"] == "cfeg.metadata-prior-source.result.v1", "Result schema")
    compatibility = result["compatibility"]
    require(compatibility["native_prediction_mismatches"] == 0, "Native prediction compatibility")
    require(
        0
        <= compatibility["max_native_correlation_error"]
        <= plan["operator"]["native_correlation_atol"],
        "Native correlation receipt",
    )
    row_lookup = indexed_rows(
        result["rows"], ("participant", "interface", "n_samples", "k", "arm"), "result rows"
    )
    expected_rows = (
        len(plan["source_subject_ids"])
        * len(plan["interfaces"])
        * len(plan["sample_counts"])
        * (2 * len(ARMS) + 1)
    )
    require(len(row_lookup) == expected_rows, "Full result row inventory")
    row_proxy = {}
    for pi, subject in enumerate(plan["source_subject_ids"]):
        for ei, interface in enumerate(plan["interfaces"]):
            for wi, n_samples in enumerate(plan["sample_counts"]):
                prefix = (subject, interface, n_samples)
                anchor_row = row_lookup[prefix + (0, "A0")]
                close(
                    anchor_row["accuracy"], a0_accuracy[pi, ei, wi], "A0 accuracy", atol=0, rtol=0
                )
                require(
                    anchor_row["prior"] is None
                    and anchor_row["proxy_components"] is None
                    and anchor_row["prediction_changes_vs_Q"] is None,
                    "A0 no artificial prior or proxy",
                )
                first_cell = (ei * len(plan["sample_counts"]) + wi) * 5
                which = slice(first_cell, first_cell + 5)
                for bi, k in enumerate(plan["budgets"]):
                    q_predictions = predictions[pi, ei, wi, bi, ARMS.index("Q")]
                    for ai, arm in enumerate(ARMS):
                        key = prefix + (k, arm)
                        row = row_lookup[key]
                        close(
                            row["accuracy"],
                            accuracy[pi, ei, wi, bi, ai],
                            "Row accuracy",
                            atol=0,
                            rtol=0,
                        )
                        require(
                            row["prediction_changes_vs_Q"]
                            == int(
                                np.count_nonzero(predictions[pi, ei, wi, bi, ai] != q_predictions)
                            ),
                            "Row prediction changes",
                        )
                        close(
                            row["prior"],
                            replay["priors"][pi, bi, which, :, ai],
                            "Evaluated prior replay",
                            atol=1e-8,
                            rtol=1e-8,
                        )
                        stored_prior = finite(row["prior"], "Stored prior")
                        require(np.all(stored_prior > 0), "Strictly positive stored prior")
                        close(
                            stored_prior.sum(-1),
                            np.full(5, 8),
                            "Stored prior trace",
                            atol=1e-10,
                            rtol=0,
                        )
                        diagnostics = row["diagnostics"]
                        actual_scores = scores_cache["scores"][pi, ei, wi, bi, ai]
                        q_scores = scores_cache["scores"][pi, ei, wi, bi, ARMS.index("Q")]
                        q_top = np.sort(q_scores, axis=-1)[:, -2:]
                        epsilon = np.max(abs(actual_scores - q_scores), axis=-1)
                        computed_diagnostics = {
                            "max_score_change": float(epsilon.max()),
                            "certified_unchanged": int(
                                np.count_nonzero(q_top[:, 1] - q_top[:, 0] > 2 * epsilon)
                            ),
                        }
                        q_prior = np.asarray(row_lookup[prefix + (k, "Q")]["prior"])
                        log_prior, log_q = np.log(row["prior"]), np.log(q_prior)
                        computed_diagnostics["max_prior_clr_change"] = float(
                            np.max(
                                abs(
                                    (log_prior - log_prior.mean(-1, keepdims=True))
                                    - (log_q - log_q.mean(-1, keepdims=True))
                                )
                            )
                        )
                        for field, expected in computed_diagnostics.items():
                            if field in diagnostics:
                                verify_mapping(
                                    diagnostics, {field: expected}, "Saved row diagnostic"
                                )
                        for field in ("min_denominator_eigenvalue", "min_eigen_gap"):
                            if field in diagnostics:
                                require(
                                    np.isfinite(diagnostics[field]) and diagnostics[field] > 0,
                                    "Positive recorded eigensystem diagnostic",
                                )
                        if arm in ("FULL", "ISO"):
                            require(row["proxy_components"] is None, "Unmodeled proxy stays null")
                            row_proxy[key] = None
                        else:
                            errors = (
                                replay["proxy"][pi, bi, which, :, ai]
                                - features["target"][pi, bi, which]
                            )
                            level = errors.mean(axis=-1, keepdims=True)
                            parts = {
                                "raw_mse": float(np.mean(errors**2)),
                                "level_mse": float(np.mean(level**2)),
                                "shape_mse": float(np.mean((errors - level) ** 2)),
                            }
                            verify_mapping(row["proxy_components"], parts, "Row proxy", atol=1e-8)
                            row_proxy[key] = parts
                        if arm == "MISSING":
                            q_row = row_lookup[prefix + (k, "Q")]
                            require(
                                np.array_equal(row["prior"], q_row["prior"]),
                                "Exact missing Q prior",
                            )
                            require(
                                np.array_equal(predictions[pi, ei, wi, bi, ai], q_predictions),
                                "Exact missing Q predictions",
                            )
                            require(
                                np.array_equal(
                                    scores_cache["scores"][pi, ei, wi, bi, ai],
                                    scores_cache["scores"][pi, ei, wi, bi, ARMS.index("Q")],
                                ),
                                "Exact missing Q scores",
                            )
    summary, contrasts, attainment, calibration, verdict = expected_statistics(
        accuracy,
        a0_accuracy,
        predictions,
        a0_predictions,
        plan,
        replay["coverage_changed_fraction"],
    )
    for group in summary:
        for part in ("raw", "level", "shape"):
            values = [
                row_proxy.get(
                    (subject, group["interface"], group["n_samples"], group["k"], group["arm"])
                )
                for subject in plan["source_subject_ids"]
            ]
            group[f"proxy_{part}_mse"] = (
                None
                if group["arm"] in ("A0", "FULL", "ISO")
                else float(np.mean([v[f"{part}_mse"] for v in values]))
            )
    for collection, expected, keys in (
        ("summary", summary, ("interface", "n_samples", "k", "arm")),
        ("contrasts", contrasts, ("scope", "k", "comparator")),
        ("attainment", attainment, ("participant", "interface", "n_samples", "arm")),
    ):
        found = indexed_rows(result[collection], keys, collection)
        require(len(found) == len(expected), f"Full {collection} inventory")
        for row in expected:
            key = tuple(row[field] for field in keys)
            require(key in found, f"Missing {collection} key")
            verify_mapping(
                found[key], row, collection, atol=1e-8 if collection == "summary" else 1e-12
            )
    verify_mapping(result["calibration"], calibration, "Calibration")
    verify_mapping(result["verdict"], verdict, "Verdict")
    return {
        "rows_checked": expected_rows,
        "summary_checked": len(summary),
        "contrasts_checked": len(contrasts),
        "attainment_checked": len(attainment),
        "selected_fit_replays": replay["selected_fit_replays"],
        "coverage_changed_fraction": replay["coverage_changed_fraction"],
        "verdict": verdict,
        "numeric_tolerances": {
            "statistics_atol": 1e-12,
            "selected_ridge_atol_rtol": 1e-8,
            "prior_atol_rtol": 1e-8,
            "proxy_atol": 1e-8,
            "missing_scores_and_prior": "exact",
        },
        "limits": [
            "No raw EEG/preprocessing/Q feature or target extraction replay",
            "No full candidate CV-loss refit; recorded losses/splits/selection and every selected regression are checked",
            "Native compatibility diagnostics are checked as receipts, not independently recomputed",
            "No full TRCA eigensystem or waveform-to-score replay",
            "Development cohort results do not establish independent confirmation",
        ],
    }


def projection_arrays(value, plan):
    spec, ids = plan["source_projection"], plan["source_subject_ids"]
    fields = {
        "manifest_sha256",
        "packets",
        "returned_rows",
        "returned_packets",
        "returned_subject_ids",
        "columns",
    }
    require(
        set(value) == fields | set(spec["envelope_provenance"]), "Exact source projection envelope"
    )
    verify_mapping(value, spec["envelope_provenance"], "Projection provenance")
    verify_mapping(
        value,
        {
            "manifest_sha256": spec["manifest_sha256"],
            "returned_rows": spec["returned_rows"],
            "returned_packets": len(ids) * 20,
            "returned_subject_ids": ids,
            "columns": spec["columns"],
        },
        "Projection inventory",
    )
    require(len(value["packets"]) == len(ids) * 20, "Source packet count")
    z, order = np.full((len(ids), 2, 10, 8), np.nan), np.full(len(ids), -1, dtype=int)
    seen = set()
    for row in value["packets"]:
        require(
            set(row)
            == {
                "subject_id",
                "interface",
                "block_id",
                "impedance_kohm",
                "headband_order",
                "condition_period",
            },
            "Packet fields",
        )
        subject, interface, block = row["subject_id"], row["interface"], row["block_id"]
        require(type(subject) is int and subject in ids, "Source-only packet participant")
        require(
            interface in plan["interfaces"] and type(block) is int and 0 <= block < 10,
            "Packet condition/block",
        )
        require((subject, interface, block) not in seen, "Unique source packet")
        seen.add((subject, interface, block))
        require(row["headband_order"] in plan["interfaces"], "Acquisition order")
        p, i = ids.index(subject), plan["interfaces"].index(interface)
        first = plan["interfaces"].index(row["headband_order"])
        require(order[p] in (-1, first), "Consistent participant order")
        require(
            row["condition_period"] == ("first" if i == first else "second"), "Acquisition period"
        )
        raw = row["impedance_kohm"]
        require(isinstance(raw, list) and len(raw) == 8, "Eight native M channels")
        require(
            all(x is None or (type(x) in (int, float) and np.isfinite(x) and x >= 0) for x in raw),
            "Numeric M or explicit missing",
        )
        z[p, i, block], order[p] = np.asarray(raw, dtype=float), first
    require(len(seen) == len(ids) * 20, "Complete projection grid")
    return z, order


def regular_bytes(path):
    path = Path(path).absolute()
    require(path.resolve() == path, "No symlink input path")
    descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    with os.fdopen(descriptor, "rb") as stream:
        require(stat.S_ISREG(os.fstat(stream.fileno()).st_mode), "Regular input file")
        return stream.read()


def read_document(path, expected=None):
    raw = regular_bytes(path)
    digest = hashlib.sha256(raw).hexdigest()
    require(expected is None or digest == expected, f"Exact input hash {Path(path).name}")
    return json.loads(raw), digest


def verify_provenance(start, result, freeze, native_start, native_manifest, hashes, plan, root):
    provenance = result["provenance"]
    require(
        start["schema"] == "cfeg.metadata-prior-source.start.v1"
        and start["study_id"] == plan["study_id"],
        "Analysis start identity",
    )
    require(
        provenance["plan_sha256"] == start["plan_sha256"] == freeze["plan_sha256"] == PLAN_SHA256,
        "Analysis plan chain",
    )
    require(
        provenance["source_commit"] == start["source_commit"]
        and len(start["source_commit"]) == 40
        and all(c in "0123456789abcdef" for c in start["source_commit"]),
        "Analysis source commit receipt",
    )
    require(provenance["source_hashes"] == start["source_hashes"], "Analysis source hash chain")
    expected_sources = {
        "src/cfeg/analysis/metadata_prior_source.py",
        "scripts/run_metadata_prior_source.py",
        "scripts/export_metadata_prior_source.py",
        plan["operator"]["path"],
        plan["q_helper"]["path"],
    }
    require(set(provenance["source_hashes"]) == expected_sources, "Exact analysis source inventory")
    for relative, expected in provenance["source_hashes"].items():
        require(
            hashlib.sha256(regular_bytes(root / relative)).hexdigest() == expected,
            "Analysis source bytes " + relative,
        )
    for role in ("operator", "q_helper"):
        require(
            provenance["source_hashes"][plan[role]["path"]] == plan[role]["sha256"],
            "Pinned numerical helper",
        )
    for field in ("python", "numpy", "scipy"):
        require(provenance[field] == start[field], "Analysis dependency chain")
    expected_inputs = {
        "start_sha256": hashes["start.json"],
        "features_sha256": hashes["features.npz"],
        "freezes_sha256": hashes["fold-freezes.json"],
        "scores_sha256": hashes["scores.npz"],
        "native_manifest_sha256": hashes["native/result.json"],
        "projection_sha256": hashes["source_projection"],
    }
    verify_mapping(provenance, expected_inputs, "Analysis input binding")
    require(
        provenance["projection_sha256"] == plan["source_projection"]["sha256"], "Pinned projection"
    )
    require(
        native_manifest["schema"] == "cfeg.metadata-prior-source.native-result.v1"
        and native_manifest["status"] == "COMPLETE",
        "Complete native manifest",
    )
    require(
        native_start["schema"] == "cfeg.metadata-prior-source.native-start.v1",
        "Native start schema",
    )
    for document in (native_start, native_manifest):
        require(
            document["study_id"] == plan["study_id"] and document["plan_sha256"] == PLAN_SHA256,
            "Native plan/study",
        )
        require(
            document["source_subject_ids"] == plan["source_subject_ids"]
            and document["sample_counts"] == plan["sample_counts"],
            "Native source-only identity",
        )
    require(native_manifest["start_sha256"] == hashes["native/start.json"], "Native start hash")
    require(
        native_manifest["source_hashes"] == native_start["source_hashes"]
        and native_manifest["source_commit"] == native_start["source_commit"],
        "Native source provenance chain",
    )
    require(
        [entry["subject"] for entry in native_manifest["files"]] == plan["source_subject_ids"],
        "Native manifest participant inventory",
    )
    for entry in native_manifest["files"]:
        require(entry["filename"] == f"S{entry['subject']:03d}.npz", "Exact native cache name")
        require(type(entry["bytes"]) is int and entry["bytes"] > 0, "Native artifact bytes")
        require(
            len(entry["sha256"]) == 64 and all(c in "0123456789abcdef" for c in entry["sha256"]),
            "Native cache hash receipt",
        )
    require(
        native_manifest["total_bytes"] == sum(e["bytes"] for e in native_manifest["files"]),
        "Native manifest total bytes",
    )


def publish_exclusive(path, payload):
    path = Path(path).absolute()
    require(
        path.parent.resolve() == path.parent and not path.is_symlink(), "Exact audit output path"
    )
    raw = (json.dumps(payload, indent=2, allow_nan=False) + "\n").encode()
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o400)
    with os.fdopen(descriptor, "wb") as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())
    return hashlib.sha256(raw).hexdigest()


def verify_recovery(start, result, native_start, native_manifest, original_start_sha):
    require(original_start_sha == RECOVERY["start_sha256"], "Original failed start pin")
    for document in (start, result["provenance"]):
        require(document["native_root"] == str(RECOVERED_NATIVE), "Explicit recovered native path")
        require(document["native_recovery"] == RECOVERY, "Exact analysis recovery receipt")
    for document in (native_start, native_manifest):
        require(document["recovery"] == RECOVERY, "Exact native recovery receipt")


def run(analysis_root=None, plan_path=None):
    root = Path(__file__).resolve().parents[1]
    plan_path = (
        root / "configs/analysis/metadata_prior_source39_v1.json"
        if plan_path is None
        else Path(plan_path)
    )
    plan, _ = read_document(plan_path, PLAN_SHA256)
    declared = Path(plan["execution"]["analysis_root"])
    analysis = declared if analysis_root is None else Path(analysis_root).absolute()
    require(
        analysis == declared and analysis.resolve() == analysis, "Exact declared analysis directory"
    )
    output = analysis / "audit.json"
    require(not output.exists() and not output.is_symlink(), "Audit output already exists")
    names = ("start.json", "features.npz", "fold-freezes.json", "scores.npz", "result.json")
    paths = {name: analysis / name for name in names}
    require(Path(plan["execution"]["native_root"]) == ORIGINAL_NATIVE, "Original native plan path")
    native = RECOVERED_NATIVE
    paths.update(
        {
            "native/start.json": native / "start.json",
            "native/result.json": native / "result.json",
            "source_projection": Path(plan["source_projection"]["path"]),
            "original_native/start.json": ORIGINAL_NATIVE / "start.json",
        }
    )
    payloads, hashes = {}, {}
    for name, path in paths.items():
        if name.endswith(".npz"):
            raw = regular_bytes(path)
            hashes[name] = hashlib.sha256(raw).hexdigest()
            with np.load(io.BytesIO(raw), allow_pickle=False) as archive:
                payloads[name] = {key: archive[key] for key in archive.files}
        else:
            payloads[name], hashes[name] = read_document(path)
    verify_recovery(
        payloads["start.json"],
        payloads["result.json"],
        payloads["native/start.json"],
        payloads["native/result.json"],
        hashes["original_native/start.json"],
    )
    verify_provenance(
        payloads["start.json"],
        payloads["result.json"],
        payloads["fold-freezes.json"],
        payloads["native/start.json"],
        payloads["native/result.json"],
        hashes,
        plan,
        root,
    )
    z, order = projection_arrays(payloads["source_projection"], plan)
    require(
        np.array_equal(z, payloads["features.npz"]["z"], equal_nan=True)
        and np.array_equal(order, payloads["features.npz"]["order"]),
        "M/order exact source projection replay",
    )
    audit = verify_results(
        payloads["scores.npz"],
        payloads["features.npz"],
        payloads["fold-freezes.json"],
        payloads["result.json"],
        plan,
    )
    for name, path in paths.items():
        require(
            hashlib.sha256(regular_bytes(path)).hexdigest() == hashes[name],
            "Input bytes changed during audit",
        )
    result = {
        "schema": "cfeg.metadata-prior-source.audit.v1",
        "study_id": plan["study_id"],
        "status": "PASS",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "plan_sha256": PLAN_SHA256,
        "inputs": hashes,
        "auditor_sha256": hashlib.sha256(regular_bytes(Path(__file__))).hexdigest(),
        "metadata_projection_exact": True,
        **audit,
    }
    result["limits"].append(
        "Native manifest hashes bind declared waveform artifacts; this audit does not reopen or rehash those waveform files"
    )
    publish_exclusive(output, result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis-root", type=Path)
    parser.add_argument("--plan", type=Path)
    args = parser.parse_args()
    result = run(args.analysis_root, args.plan)
    print(
        json.dumps(
            {
                "status": result["status"],
                "rows_checked": result["rows_checked"],
                "verdict": result["verdict"],
            }
        )
    )


if __name__ == "__main__":
    main()
