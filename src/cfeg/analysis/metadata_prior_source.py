"""Frozen source39 prior experiment mathematics; no file or dataset access."""

from __future__ import annotations

import numpy as np
from scipy.stats import t

from cfeg.analysis import metadata_prior_validation as learning
from cfeg.analysis import metadata_trca_prior as op

ARMS = ("FULL", "ISO", "Q", "Q2", "QM", "SHAM_REFIT", "PERMUTED", "STALE", "MISSING")


def m_features(packet):
    raw = np.asarray(packet, dtype=float)
    if (
        raw.ndim != 2
        or raw.shape[1] != 8
        or np.isinf(raw).any()
        or np.any(raw[np.isfinite(raw)] < 0)
    ):
        raise ValueError("Expected nonnegative k-by8 impedance; NaN alone is missing")
    mask = np.isfinite(raw)
    count = mask.sum(0)
    available = count > 0
    logged = np.log1p(np.where(mask, raw, 0))
    mean = logged.sum(0) / np.maximum(count, 1)
    center = np.sum(mean * available) / max(int(available.sum()), 1)
    variance = np.sum(np.where(mask, (logged - mean) ** 2, 0), axis=0) / np.maximum(count, 1)
    features = np.stack((mean - center, np.sqrt(variance)), axis=-1)
    return np.where(available[:, None], features, 0), available


def support_q(support, packet, interface, order, frequencies):
    x = np.asarray(support, dtype=float)
    if (
        x.ndim != 5
        or x.shape[0] not in (3, 5)
        or x.shape[1:4] != (12, 5, 8)
        or not np.isfinite(x).all()
    ):
        raise ValueError("Invalid native support geometry")
    m_features(packet)
    if np.asarray(packet).shape != (len(x), 8) or interface not in (0, 1) or order not in (0, 1):
        raise ValueError("Invalid support-only context")
    power = np.mean(x**2, axis=(0, 1, 4))
    global_power = np.maximum(power.mean(-1, keepdims=True), 1e-12)
    off = np.zeros((5, 8))
    time = np.arange(x.shape[-1]) / 250
    for label, frequency in enumerate(frequencies):
        reference = np.stack(
            [fn(2 * np.pi * frequency * h * time) for h in range(1, 6) for fn in (np.sin, np.cos)],
            axis=1,
        )
        reference -= reference.mean(0)
        basis = np.linalg.qr(reference)[0]
        residual = x[:, label] - (x[:, label] @ basis) @ basis.T
        off += np.mean(residual**2, axis=(0, 3)) / 12
    centered = x - x.mean(-1, keepdims=True)
    correlations = []
    for a in range(len(x)):
        for b in range(a + 1, len(x)):
            left, right = centered[a], centered[b]
            norm = np.sqrt(np.sum(left**2, -1) * np.sum(right**2, -1))
            if np.any(norm <= 1e-24):
                raise ValueError("Zero centered support variance")
            correlations.append(np.clip(np.sum(left * right, -1) / norm, -1, 1))
    result = np.stack(
        (
            np.log(np.maximum(power / global_power, 1e-12)),
            np.log(np.clip(off / np.maximum(power, 1e-12), 1e-6, 1)),
            1 - np.mean(correlations, axis=(0, 1)),
            np.full((5, 8), np.log(len(x))),
            np.broadcast_to(np.isfinite(packet).mean(0), (5, 8)),
            np.full((5, 8), order),
            np.full((5, 8), interface != order),
        ),
        axis=-1,
    )
    return np.concatenate((result, np.broadcast_to(np.eye(8), (5, 8, 8))), axis=-1)


def proxy_target(support, later):
    x, future = np.asarray(support), np.asarray(later)
    if x.ndim != 5 or future.shape != x.shape[1:] or not np.isfinite(future).all():
        raise ValueError("Invalid separate repeat proxy")
    scale = np.maximum(np.mean(x * x, axis=(0, 1, 3, 4)), 1e-12)[:, None]
    error = np.mean((future - x.mean(0)) ** 2, axis=(0, 3))
    return np.log(np.maximum(error / scale, 1e-12))


def subject_features(arrays, z, order, plan):
    count = 2 * len(plan["sample_counts"]) * 5
    q = np.empty((2, count, 8, 15))
    target = np.empty((2, count, 8))
    for i in range(2):
        for wi, n in enumerate(plan["sample_counts"]):
            x = arrays[f"x_{n}"][i]
            cell = (i * len(plan["sample_counts"]) + wi) * 5
            for bi, k in enumerate(plan["budgets"]):
                q[bi, cell : cell + 5] = support_q(
                    x[:k], z[i, :k], i, int(order), plan["frequencies"]
                )
                target[bi, cell : cell + 5] = proxy_target(x[:k], x[plan["proxy_block"]])
    return q, target


def donor_maps(z, order, indices, ids, budgets):
    indices = np.asarray(indices, dtype=int)
    result = np.empty((2, 2, len(indices)), dtype=int)
    for i in range(2):
        for bi, k in enumerate(budgets):
            groups = {}
            for local, p in enumerate(indices):
                key = (int(order[p]), tuple(np.isfinite(z[p, i, :k]).ravel()))
                groups.setdefault(key, []).append((ids[p], local, int(p)))
            for group in groups.values():
                group.sort()
                for index, (_, local, _) in enumerate(group):
                    result[i, bi, local] = group[(index + 1) % len(group)][2]
    return result


def ridge_from_record(r):
    return learning.RidgeModel(
        np.array(r["mean"]),
        np.array(r["scale"]),
        np.array(r["coefficient"]),
        r["intercept"],
        r["alpha"],
        r["design_rank"],
    )


def q_from_record(r):
    return learning.SelectedQ(
        r["representation"], ridge_from_record(r["ridge"]), r["feature_count"]
    )


def metadata_rows(z, indices, interface, budgets, maps=None):
    return np.stack(
        [
            np.stack(
                [
                    m_features(z[p if maps is None else maps[interface, bi, j], interface, :k])[0]
                    for bi, k in enumerate(budgets)
                ]
            )
            for j, p in enumerate(indices)
        ]
    )[:, :, None]


def fit_all(features, plan, progress=None):
    q, y, z, order = (features[k] for k in ("q", "target", "z", "order"))
    ids = np.asarray(plan["source_subject_ids"])
    assignment = learning.participant_folds(ids)
    folds = []
    for fold in range(3):
        train, evaluate = np.flatnonzero(assignment != fold), np.flatnonzero(assignment == fold)
        train_maps = donor_maps(z, order, train, ids, plan["budgets"])
        eval_maps = donor_maps(z, order, evaluate, ids, plan["budgets"])
        cells = []
        for cell in range(q.shape[2]):
            interface = cell // (len(plan["sample_counts"]) * 5)
            x, target = q[train, :, cell : cell + 1], y[train, :, cell : cell + 1]
            final_q, oof, receipt = learning.crossfit_q(x, target, ids[train])
            residual = (target - oof).ravel()
            q2x = learning.q_design(x, final_q.representation)
            mx = metadata_rows(z, train, interface, plan["budgets"])
            sx = metadata_rows(z, train, interface, plan["budgets"], train_maps)
            residuals = {
                name: learning.fit_ridge(
                    values.reshape(-1, values.shape[-1]), residual, 0.1
                ).record()
                for name, values in (("Q2", q2x), ("QM", mx), ("SHAM_REFIT", sx))
            }
            cells.append(
                {
                    "cell": cell,
                    "Q": final_q.record(),
                    "oof_prediction": oof.tolist(),
                    "q_receipt": receipt,
                    "residuals": residuals,
                }
            )
            if progress and cell % 5 == 4:
                progress(f"fit fold={fold} cells={cell + 1}/{q.shape[2]}")
        folds.append(
            {
                "fold": fold,
                "fit_ids": ids[train].tolist(),
                "eval_ids": ids[evaluate].tolist(),
                "train_maps": train_maps.tolist(),
                "eval_maps": eval_maps.tolist(),
                "cells": cells,
            }
        )
    return {
        "schema": "cfeg.metadata-prior-source.freezes.v1",
        "study_id": plan["study_id"],
        "folds": folds,
    }


def subject_priors(features, p, i, wi, bi, fold, plan):
    z, q = features["z"], features["q"]
    k = plan["budgets"][bi]
    pid = plan["source_subject_ids"][p]
    local = fold["eval_ids"].index(pid)
    donor = fold["eval_maps"][i][bi][local]
    mf, observed = m_features(z[p, i, :k])
    perm, perm_mask = m_features(z[donor, i, :k])
    stale, stale_mask = m_features(np.repeat(z[p, i, :1], k, axis=0))
    base = np.empty((5, 8))
    deltas = {
        arm: np.empty((5, 8)) for arm in ("Q2", "QM", "SHAM_REFIT", "PERMUTED", "STALE", "MISSING")
    }
    for band in range(5):
        cell = (i * len(plan["sample_counts"]) + wi) * 5 + band
        frozen = fold["cells"][cell]
        qmodel = q_from_record(frozen["Q"])
        x = q[p : p + 1, bi : bi + 1, cell : cell + 1]
        base[band] = qmodel.predict(x).reshape(8)
        residual = {name: ridge_from_record(r) for name, r in frozen["residuals"].items()}
        deltas["Q2"][band] = (
            residual["Q2"].predict(learning.q_design(x, qmodel.representation)).reshape(8)
        )
        for name, values, model in (
            ("QM", mf, "QM"),
            ("SHAM_REFIT", perm, "SHAM_REFIT"),
            ("PERMUTED", perm, "QM"),
            ("STALE", stale, "QM"),
        ):
            deltas[name][band] = residual[model].predict(values)
        deltas["MISSING"][band] = 0
    qprior = op.trace_normalize(np.exp(np.clip(base, -3, 3)))
    priors = {"FULL": np.ones((5, 8)), "ISO": np.ones((5, 8)), "Q": qprior}
    proxy = {"FULL": None, "ISO": None, "Q": base}
    masks = {
        "Q2": np.ones(8, dtype=bool),
        "QM": observed,
        "SHAM_REFIT": observed & perm_mask,
        "PERMUTED": observed & perm_mask,
        "STALE": observed & stale_mask,
        "MISSING": np.zeros(8, dtype=bool),
    }
    for name, delta in deltas.items():
        mask = np.broadcast_to(masks[name], (5, 8))
        priors[name] = op.residual_prior(qprior, delta, mask, bound=0.2)
        proxy[name] = base + np.where(mask, np.clip(delta, -0.2, 0.2), 0)
    return priors, proxy


def native_scores(correlations, weights):
    r = np.asarray(correlations)
    return np.array([np.asarray(weights)[None] @ row for row in r.reshape(-1, 5, 12)]).reshape(
        -1, 12
    )


def evaluate_subject(arrays, features, p, freezes, plan):
    pid = plan["source_subject_ids"][p]
    fold = next(f for f in freezes["folds"] if pid in f["eval_ids"])
    windows = len(plan["sample_counts"])
    scores = np.empty((2, windows, 2, len(ARMS), 48, 12))
    anchors = np.empty((2, windows, 48, 12))
    rows, max_compat = [], 0.0
    labels = np.tile(np.arange(12), 4)
    for i, interface in enumerate(plan["interfaces"]):
        for wi, n in enumerate(plan["sample_counts"]):
            x = arrays[f"x_{n}"][i]
            anchors[i, wi] = native_scores(
                arrays[f"a0_{n}"][i], plan["native_weights"]["A0_author"][interface]
            )
            common = {"participant": pid, "interface": interface, "n_samples": n}
            rows.append(
                {
                    **common,
                    "k": 0,
                    "arm": "A0",
                    "accuracy": float(np.mean(anchors[i, wi].argmax(-1) == labels)),
                    "proxy_components": None,
                    "prior": None,
                    "prediction_changes_vs_Q": None,
                    "diagnostics": {},
                }
            )
            for bi, k in enumerate(plan["budgets"]):
                priors, proxy = subject_priors(features, p, i, wi, bi, fold, plan)
                models = {}
                for ai, arm in enumerate(ARMS):
                    model = op.fit_trca(x[:k], priors[arm], 0 if arm == "FULL" else 0.1)
                    score, corr = op.score_trca(
                        model,
                        x[6:10].reshape(48, 5, 8, n),
                        np.array(plan["native_weights"]["ETRCA"][interface]),
                    )
                    models[arm] = model
                    scores[i, wi, bi, ai] = score
                    if arm == "FULL":
                        reference = arrays[f"full_{n}"][i, bi].reshape(48, 5, 12)
                        error = float(np.max(abs(corr - reference)))
                        max_compat = max(max_compat, error)
                        if error > plan["operator"][
                            "native_correlation_atol"
                        ] or not np.array_equal(
                            score.argmax(-1),
                            native_scores(
                                reference, plan["native_weights"]["ETRCA"][interface]
                            ).argmax(-1),
                        ):
                            raise ValueError("Fresh native FULL compatibility failed")
                qscore = scores[i, wi, bi, ARMS.index("Q")]
                qfilter = models["Q"].filters
                qunit = qfilter / np.linalg.norm(qfilter, axis=1, keepdims=True)
                cell = (i * windows + wi) * 5
                target = features["target"][p, bi, cell : cell + 5]
                for ai, arm in enumerate(ARMS):
                    score = scores[i, wi, bi, ai]
                    model = models[arm]
                    unit = model.filters / np.linalg.norm(model.filters, axis=1, keepdims=True)
                    # Stable sign-invariant sin(angle); identical filters give exact0.
                    sine = 0.5 * np.linalg.norm(unit - qunit, axis=1) * np.linalg.norm(
                        unit + qunit, axis=1
                    )
                    sensitivity = learning.score_sensitivity(qscore, score)
                    pieces = (
                        None
                        if proxy[arm] is None
                        else {
                            key: float(value.mean())
                            for key, value in learning.proxy_error_components(
                                proxy[arm], target
                            ).items()
                        }
                    )
                    diagnostics = {
                        "max_score_change": float(np.max(sensitivity["score_max_abs_change"])),
                        "certified_unchanged": int(sensitivity["certified_unchanged"].sum()),
                        "max_filter_sine": float(sine.max()),
                        "max_prior_clr_change": float(
                            np.max(
                                abs(
                                    learning.prior_log_shape(priors[arm])
                                    - learning.prior_log_shape(priors["Q"])
                                )
                            )
                        ),
                        "min_denominator_eigenvalue": float(
                            np.min(model.diagnostics["denominator_min_eigenvalues"])
                        ),
                        "min_eigen_gap": float(np.min(model.diagnostics["top_eigenvalue_gaps"])),
                        "max_norm_error": float(
                            np.max(model.diagnostics["denominator_norm_errors"])
                        ),
                        "max_penalty_trace_error_vs_Q": float(
                            np.max(
                                abs(
                                    np.array(model.diagnostics["penalty_trace"])
                                    - models["Q"].diagnostics["penalty_trace"]
                                )
                            )
                        )
                        if arm != "FULL"
                        else None,
                    }
                    rows.append(
                        {
                            **common,
                            "k": k,
                            "arm": arm,
                            "accuracy": float(np.mean(score.argmax(-1) == labels)),
                            "proxy_components": pieces,
                            "prior": priors[arm].tolist(),
                            "prediction_changes_vs_Q": int(sensitivity["prediction_changed"].sum()),
                            "diagnostics": diagnostics,
                        }
                    )
                if not np.array_equal(priors["MISSING"], priors["Q"]) or not np.array_equal(
                    scores[i, wi, bi, ARMS.index("MISSING")], qscore
                ):
                    raise ValueError("Missing M did not reproduce fitted Q exactly")
    return rows, scores, anchors, max_compat


def paired_stats(values):
    d = np.asarray(values, dtype=float)
    mean = float(d.mean())
    half = float(t.ppf(0.975, len(d) - 1) * d.std(ddof=1) / np.sqrt(len(d)))
    return {
        "mean_delta": mean,
        "ci_low": mean - half,
        "ci_high": mean + half,
        "help": int((d > 0).sum()),
        "tie": int((d == 0).sum()),
        "harm": int((d < 0).sum()),
        "n": len(d),
    }


def summarize(rows, features, freezes, plan):
    ids, interfaces, windows = plan["source_subject_ids"], plan["interfaces"], plan["sample_counts"]
    lookup = {(r["participant"], r["interface"], r["n_samples"], r["k"], r["arm"]): r for r in rows}
    expected = len(ids) * len(interfaces) * len(windows) * (1 + 2 * len(ARMS))
    if len(lookup) != expected or len(rows) != expected:
        raise ValueError("Incomplete or duplicated outcome grid")
    summary, contrasts, attainment = [], [], []
    for i in interfaces:
        for n in windows:
            for k, arms in ((0, ("A0",)), (3, ARMS), (5, ARMS)):
                for arm in arms:
                    group = [lookup[p, i, n, k, arm] for p in ids]
                    diff = (
                        None
                        if k == 0
                        else np.array(
                            [
                                r["accuracy"] - lookup[r["participant"], i, n, k, "Q"]["accuracy"]
                                for r in group
                            ]
                        )
                    )
                    pieces = group[0]["proxy_components"]
                    summary.append(
                        {
                            "interface": i,
                            "n_samples": n,
                            "k": k,
                            "arm": arm,
                            "accuracy": float(np.mean([r["accuracy"] for r in group])),
                            **{
                                f"proxy_{part}_mse": None
                                if pieces is None
                                else float(
                                    np.mean([r["proxy_components"][part + "_mse"] for r in group])
                                )
                                for part in ("raw", "level", "shape")
                            },
                            "help_vs_Q": None if diff is None else int((diff > 0).sum()),
                            "tie_vs_Q": None if diff is None else int((diff == 0).sum()),
                            "harm_vs_Q": None if diff is None else int((diff < 0).sum()),
                            "prediction_changes_vs_Q": None
                            if k == 0
                            else sum(r["prediction_changes_vs_Q"] for r in group),
                        }
                    )
    scopes = [("ALL", [(i, n) for i in interfaces for n in windows])] + [
        (f"{i}_{n}", [(i, n)]) for i in interfaces for n in windows
    ]
    for scope, cells in scopes:
        for k in plan["budgets"]:
            for comparator in ("Q", "Q2", "SHAM_REFIT", "FULL", "A0") + (("Q5",) if k == 3 else ()):
                bk = 0 if comparator == "A0" else 5 if comparator == "Q5" else k
                ba = "Q" if comparator == "Q5" else comparator
                d = [
                    np.mean(
                        [
                            lookup[p, i, n, k, "QM"]["accuracy"]
                            - lookup[p, i, n, bk, ba]["accuracy"]
                            for i, n in cells
                        ]
                    )
                    for p in ids
                ]
                contrasts.append(
                    {"scope": scope, "k": k, "comparator": comparator, **paired_stats(d)}
                )
    for p in ids:
        for i in interfaces:
            for n in windows:
                for arm in ARMS:
                    first = next(
                        (
                            k
                            for k in (0, 3, 5)
                            if lookup[p, i, n, k, "A0" if k == 0 else arm]["accuracy"] >= 0.8
                        ),
                        None,
                    )
                    attainment.append(
                        {
                            "participant": p,
                            "interface": i,
                            "n_samples": n,
                            "arm": arm,
                            "first_k": first,
                            "labels": None if first is None else first * 12,
                        }
                    )
    reach = {(r["participant"], r["interface"], r["n_samples"], r["arm"]): r for r in attainment}
    calibration = {
        "both_reached": 0,
        "new_reach": 0,
        "lost_reach": 0,
        "neither_reached": 0,
        "transitions": {},
    }
    label_diffs = []
    for p in ids:
        for i in interfaces:
            for n in windows:
                left, right = reach[p, i, n, "Q"], reach[p, i, n, "QM"]
                key = f"{left['first_k']}->{right['first_k']}"
                calibration["transitions"][key] = calibration["transitions"].get(key, 0) + 1
                a, b = left["labels"], right["labels"]
                category = (
                    "both_reached"
                    if a is not None and b is not None
                    else "new_reach"
                    if b is not None
                    else "lost_reach"
                    if a is not None
                    else "neither_reached"
                )
                calibration[category] += 1
                if category == "both_reached":
                    label_diffs.append(b - a)
    calibration["mean_label_delta_both"] = float(np.mean(label_diffs)) if label_diffs else None
    changed = 0
    for fold in freezes["folds"]:
        for local, pid in enumerate(fold["eval_ids"]):
            p = ids.index(pid)
            for i in range(2):
                donor = fold["eval_maps"][i][0][local]
                changed += int(
                    np.any(
                        abs(
                            m_features(features["z"][p, i, :3])[0]
                            - m_features(features["z"][donor, i, :3])[0]
                        )
                        > 1e-12
                    )
                )
    coverage = changed / (len(ids) * 2)
    totals = {(c["k"], c["comparator"]): c for c in contrasts if c["scope"] == "ALL"}
    primary = totals[3, "Q"]
    increment = (
        primary["mean_delta"] >= 0.01
        and primary["ci_low"] > 0
        and all(totals[3, c]["ci_low"] > 0 for c in ("Q2", "SHAM_REFIT"))
        and coverage >= 0.5
    )
    mean_qm3 = float(
        np.mean(
            [lookup[p, i, n, 3, "QM"]["accuracy"] for p in ids for i in interfaces for n in windows]
        )
    )
    saving = bool(
        increment
        and totals[3, "Q5"]["ci_low"] > -0.01
        and mean_qm3 >= 0.8
        and label_diffs
        and np.mean(label_diffs) < 0
        and calibration["new_reach"] >= calibration["lost_reach"]
    )
    status = (
        "DEVELOPMENT_CALIBRATION_BENEFIT_CANDIDATE"
        if saving
        else "CLASSIFICATION_INCREMENT_ONLY"
        if increment
        else "METADATA_INCREMENT_NOT_ESTABLISHED"
    )
    return {
        "summary": summary,
        "contrasts": contrasts,
        "attainment": attainment,
        "calibration": calibration,
        "verdict": {
            "status": status,
            "metadata_increment": bool(increment),
            "calibration_benefit": saving,
            "primary_mean_QM3": mean_qm3,
            "coverage_changed_fraction": coverage,
        },
    }
