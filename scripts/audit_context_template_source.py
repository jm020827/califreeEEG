#!/usr/bin/env python3
"""Read-only independent Gram/score audit; never import the producer or open EEG."""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import stat
import subprocess
from pathlib import Path

import numpy as np
import yaml
from scipy import stats


def require(condition, message):
    if not condition:
        raise ValueError(message)


def close(actual, expected, label):
    require(np.allclose(actual, expected, rtol=0, atol=1e-10), f"Mismatch: {label}")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def softmax(scores, temperature=1.0):
    scaled = np.asarray(scores) / temperature
    exponent = np.exp(scaled - scaled.max(axis=-1, keepdims=True))
    return exponent / exponent.sum(axis=-1, keepdims=True)


def paired(values):
    values = np.asarray(values, dtype=float)
    count = len(values)
    mean = float(values.mean())
    se = float(values.std(ddof=1) / np.sqrt(count))
    t90, t95 = stats.t.ppf([0.95, 0.975], count - 1)
    return {
        "n": count,
        "mean": mean,
        "standard_error": se,
        "one_sided_95_LCB": float(mean - t90 * se),
        "two_sided_95_CI": [float(mean - t95 * se), float(mean + t95 * se)],
        "two_sided_90_CI": [float(mean - t90 * se), float(mean + t90 * se)],
        "participant_values": values.tolist(),
    }


def verify_features(cache, plan):
    p, l = len(plan["source_subject_ids"]), len(plan["sample_counts"]) * 2
    f, q, c, b, h = len(cache["filter_weights"]), 60, 12, 5, 8
    shapes = {
        "subject_ids": (p,),
        "cell_ids": (l,),
        "sample_counts": (l,),
        "interface_indices": (l,),
        "query_labels": (q,),
        "query_block_ids": (q,),
        "support_block_ids": (b,),
        "gram_cross": (p, l, f, q, c, b),
        "gram_support": (p, l, f, c, b, b),
        "query_norm2": (p, l, f, q),
        "a0_scores": (p, l, q, c),
        "etrca_scores": (p, l, 2, q, c),
        "q_full_support": (p, l, b, h, h),
        "q_full_query": (p, l, q, h, h),
        "q_diag_support": (p, l, b, h),
        "q_diag_query": (p, l, q, h),
        "reliability": (p, l, b),
        "impedance": (p, 2, 10, h),
        "raw_file_sha256": (p,),
        "crop_sha256": (p, l),
    }
    require(str(cache["schema"]) == "cfeg.context-template-source.features.v1", "Cache schema")
    for name, shape in shapes.items():
        require(cache[name].shape == shape, f"Cache shape {name}")
        if cache[name].dtype.kind == "f" and name != "impedance":
            require(np.isfinite(cache[name]).all(), f"Finite cache {name}")
    require(cache["subject_ids"].tolist() == plan["source_subject_ids"], "Subject allowlist")
    expected_cells = [f"{e}-n{n}" for n in plan["sample_counts"] for e in plan["interfaces"]]
    require(cache["cell_ids"].tolist() == expected_cells, "Cell order")
    require(
        cache["sample_counts"].tolist() == np.repeat(plan["sample_counts"], 2).tolist(), "Windows"
    )
    require(cache["interface_indices"].tolist() == [0, 1] * (l // 2), "Interfaces")
    require(np.array_equal(cache["query_labels"], np.tile(np.arange(c), 5)), "Fixed query labels")
    require(
        np.array_equal(cache["query_block_ids"], np.repeat(np.arange(5, 10), c)), "Query blocks"
    )
    require(np.array_equal(cache["support_block_ids"], np.arange(b)), "Support blocks")
    require(f == 7 and np.all(cache["filter_weights"] > 0), "Fixed filterbank")
    require(np.all(cache["query_norm2"] >= 0), "Query squared norms")
    close(cache["gram_support"], cache["gram_support"].swapaxes(-1, -2), "Symmetric Gram")
    require(np.linalg.eigvalsh(cache["gram_support"]).min() >= -1e-5, "PSD Gram tolerance")
    for name in ("q_full_query", "q_full_support"):
        close(cache[name], cache[name].swapaxes(-1, -2), f"Symmetric {name}")
    require(
        np.all((cache["reliability"] >= 0) & (cache["reliability"] <= 1 + 1e-10)), "Reliability"
    )
    z = cache["impedance"]
    require(not np.isinf(z).any() and np.all(z[np.isfinite(z)] >= 0), "Impedance validity")
    for name in ("raw_file_sha256", "crop_sha256"):
        require(
            all(
                len(str(v)) == 64 and all(x in "0123456789abcdef" for x in str(v))
                for v in cache[name].flat
            ),
            f"Hash format {name}",
        )


def qweights(cache, p, l, candidate, budget):
    queries = len(cache["query_labels"])
    if budget == 0:
        return np.empty((queries, 0))
    if budget == 1 or candidate["kind"] == "uniform":
        return np.full((queries, budget), 1 / budget)
    logits = np.broadcast_to(
        np.log(0.05 + cache["reliability"][p, l, :budget]), (queries, budget)
    ).copy()
    if candidate["kind"] != "reliability":
        kind = candidate["kind"]
        delta = (
            cache[f"q_{kind}_query"][p, l, :, None]
            - cache[f"q_{kind}_support"][p, l, None, :budget]
        )
        axes = (-2, -1) if kind == "full" else (-1,)
        distance = np.sum(delta**2, axis=axes) / 8
        logits -= distance / (2 * candidate["tau"] ** 2)
    return softmax(logits)


def metadata_scale(impedance):
    scale = np.full((2, 8), 0.1)
    available = np.zeros((2, 8), dtype=bool)
    for e in range(2):
        for h in range(8):
            z = impedance[:, e, :, h].ravel()
            finite = z[np.isfinite(z)]
            if finite.size:
                available[e, h] = True
                low, high = np.quantile(np.log1p(finite), [0.25, 0.75])
                scale[e, h] = max(0.1, high - low)
    return scale, available


def mweights(cache, p, l, qw, scale, available, beta, permutation=None, stale=False):
    budget = qw.shape[1]
    if budget < 2 or beta == 0:
        return qw.copy()
    e = int(cache["interface_indices"][l])
    packets = cache["impedance"][p, e]
    support = packets[:budget]
    if permutation is not None:
        support = support[list(permutation)]
    query = packets[cache["query_block_ids"]]
    if stale:
        query = np.broadcast_to(packets[budget - 1], query.shape)
    common = np.isfinite(query[:, None]) & np.isfinite(support[None]) & available[e]
    delta = np.where(common, np.log1p(query[:, None]) - np.log1p(support[None]), 0)
    squared = np.sum((delta / scale[e]) ** 2, axis=-1) / 8
    affinity = np.clip(np.exp(-beta * squared / 2), 0.25, 1)
    weighted = qw * affinity
    return weighted / weighted.sum(axis=-1, keepdims=True)


def order_weights(cache, qw, beta):
    budget = qw.shape[1]
    if budget < 2 or beta == 0:
        return qw.copy()
    difference = (cache["query_block_ids"][:, None] - np.arange(budget)) / 10
    affinity = np.clip(np.exp(-beta * difference**2 / 2), 0.25, 1)
    weighted = qw * affinity
    return weighted / weighted.sum(axis=-1, keepdims=True)


def template_scores(cache, p, l, weights):
    k = weights.shape[1]
    cross = cache["gram_cross"][p, l, ..., :k]
    gram = cache["gram_support"][p, l, ..., :k, :k]
    numerator = np.einsum("qb,fqcb->fqc", weights, cross)
    template_squared = np.einsum("qb,qk,fcbk->fqc", weights, weights, gram)
    denominator = np.sqrt(cache["query_norm2"][p, l, :, :, None] * np.maximum(template_squared, 0))
    corr = np.divide(numerator, denominator, out=np.zeros_like(numerator), where=denominator > 0)
    corr = np.clip(corr, -1, 1)
    return (
        np.einsum("f,fqc->qc", cache["filter_weights"], corr * np.abs(corr))
        / cache["filter_weights"].sum()
    )


def nll(probability, labels):
    return float(
        -np.log(
            np.maximum(np.take_along_axis(probability, labels[..., None], -1)[..., 0], 1e-300)
        ).mean()
    )


def verify_fit_record(record, objectives, grid, label):
    close(record["objective_values"], objectives, f"{label} objectives")
    # Independent summation may reorder machine-level ties. Producer must take its own first exact min.
    selected = int(np.argmin(record["objective_values"]))
    require(record["selected_index"] == selected, f"{label} exact first minimum")
    close(record["temperature"], grid[selected], f"{label} selected temperature")


def verify_folds(cache, document, plan):
    grid = np.logspace(-4, 1, 31)
    candidates = plan["q"]["candidates"]
    folds = document["folds"]
    require([f["fold_id"] for f in folds] == [0, 1, 2], "Exact fold inventory")
    total_objectives = 0
    labels = cache["query_labels"]
    for fold in folds:
        index = fold["fold_id"]
        fitting = [p for p in range(len(cache["subject_ids"])) if p % 3 != index]
        evaluation = [p for p in range(len(cache["subject_ids"])) if p % 3 == index]
        require(
            fold["fit_subject_ids"] == cache["subject_ids"][fitting].tolist(), "Fold fit partition"
        )
        require(
            fold["evaluation_subject_ids"] == cache["subject_ids"][evaluation].tolist(),
            "Fold eval partition",
        )
        close(fold["temperature_grid"], grid, "Temperature grid")
        a0_scores = cache["a0_scores"][fitting].reshape(-1, 12)
        truth = np.tile(labels, len(fitting) * len(cache["cell_ids"]))
        objectives = [nll(softmax(a0_scores, t), truth) for t in grid]
        verify_fit_record(fold["a0_temperature"], objectives, grid, "A0")
        total_objectives += len(grid)
        a0 = softmax(a0_scores, fold["a0_temperature"]["temperature"])
        candidate_objectives = []
        weights = {}
        for candidate in candidates:
            values = []
            for k in (1, 3, 5):
                weighted = [
                    qweights(cache, p, l, candidate, k)
                    for p in fitting
                    for l in range(len(cache["cell_ids"]))
                ]
                weights[candidate["id"], k] = weighted
                scores = np.concatenate(
                    [
                        template_scores(cache, p, l, w)
                        for (p, l), w in zip(
                            itertools.product(fitting, range(len(cache["cell_ids"]))),
                            weighted,
                            strict=True,
                        )
                    ]
                )
                objectives = [nll(0.5 * a0 + 0.5 * softmax(scores, t), truth) for t in grid]
                record = fold["support_temperatures"][candidate["id"]][str(k)]
                verify_fit_record(record, objectives, grid, f"{candidate['id']}/k{k}")
                total_objectives += len(grid)
                if k in (3, 5):
                    values.append(min(objectives))
            candidate_objectives.append(float(np.mean(values)))
        for key, selected_candidates in (
            ("q_selection", candidates),
            ("diag_selection", [c for c in candidates if c["kind"] == "diag"]),
            ("full_selection", [c for c in candidates if c["kind"] == "full"]),
        ):
            indices = [candidates.index(c) for c in selected_candidates]
            expected = [candidate_objectives[i] for i in indices]
            record = fold[key]
            close(record["candidate_objectives"], expected, key)
            require(
                record["candidate_id"]
                == selected_candidates[int(np.argmin(record["candidate_objectives"]))]["id"],
                f"{key} exact first minimum",
            )
        scale, available = metadata_scale(cache["impedance"][fitting])
        close(fold["metadata_scale"], scale, "Fit-only metadata IQR")
        require(
            np.array_equal(fold["metadata_scale_available"], available), "Fit metadata availability"
        )
        selected_q = fold["q_selection"]["candidate_id"]
        for key, beta_grid in (
            ("m_selection", plan["m"]["beta_grid"]),
            ("order_selection", plan["m"]["order_beta_grid"]),
        ):
            close(fold[key]["beta_grid"], beta_grid, f"{key} grid")
            values = []
            for beta in beta_grid:
                budgets = []
                for k in (3, 5):
                    transformed = []
                    for (p, l), qw in zip(
                        itertools.product(fitting, range(len(cache["cell_ids"]))),
                        weights[selected_q, k],
                        strict=True,
                    ):
                        w = (
                            mweights(cache, p, l, qw, scale, available, beta)
                            if key == "m_selection"
                            else order_weights(cache, qw, beta)
                        )
                        transformed.append(template_scores(cache, p, l, w))
                    t = fold["support_temperatures"][selected_q][str(k)]["temperature"]
                    budgets.append(
                        nll(0.5 * a0 + 0.5 * softmax(np.concatenate(transformed), t), truth)
                    )
                values.append(float(np.mean(budgets)))
            close(fold[key]["objective_values"], values, f"{key} objectives")
            best = int(np.argmin(fold[key]["objective_values"]))
            require(
                fold[key]["selected_index"] == best and fold[key]["beta"] == beta_grid[best],
                f"{key} first minimum",
            )
            total_objectives += len(beta_grid)
        for ki, k in enumerate((3, 5)):
            scores = cache["etrca_scores"][fitting, :, ki].reshape(-1, 12)
            objectives = [nll(softmax(scores, t), truth) for t in grid]
            verify_fit_record(fold["eTRCA_temperatures"][str(k)], objectives, grid, f"eTRCA{k}")
            total_objectives += len(grid)
    return total_objectives


def metrics(prob, labels, a0, aq):
    prediction = prob.argmax(-1)
    count = np.bincount(labels, minlength=prob.shape[1])
    correct = np.bincount(labels, weights=prediction == labels, minlength=prob.shape[1])
    truth = prob[np.arange(len(labels)), labels]
    others = prob.copy()
    others[np.arange(len(labels)), labels] = -np.inf
    return {
        "balanced_accuracy": float(np.mean(correct / count)),
        "correct_log_probability": float(np.log(np.maximum(truth, 1e-300)).mean()),
        "true_class_margin": float(np.mean(truth - others.max(-1))),
        "prediction_flips_vs_A0": float(np.mean(prediction != a0.argmax(-1))),
        "prediction_flips_vs_AQ": float(np.mean(prediction != aq.argmax(-1))),
        "correct_count": float(correct.sum()),
        "query_count": len(labels),
        "class_correct_counts": correct.tolist(),
        "class_query_counts": count.tolist(),
    }


def verify_rows(cache, freeze, result, plan):
    rows = result["rows"]
    by_key = {(r["subject_id"], r["cell_id"], r["method"], r["budget"]): r for r in rows}
    expected_count = len(cache["subject_ids"]) * len(cache["cell_ids"]) * 42
    require(len(by_key) == len(rows) == expected_count, "Row inventory and uniqueness")
    candidates = {c["id"]: c for c in plan["q"]["candidates"]}
    checked = 0
    for p, subject in enumerate(cache["subject_ids"]):
        fold = freeze["folds"][p % 3]
        selected = fold["q_selection"]["candidate_id"]
        scale, available = (
            np.asarray(fold["metadata_scale"]),
            np.asarray(fold["metadata_scale_available"]),
        )
        for l, cell in enumerate(cache["cell_ids"]):
            a0 = softmax(cache["a0_scores"][p, l], fold["a0_temperature"]["temperature"])
            for k in plan["budgets"]:
                qw = qweights(cache, p, l, candidates[selected], k)
                temperature = (
                    fold["support_temperatures"][selected][str(k)]["temperature"] if k else None
                )

                def probability(w, temp=temperature, a0=a0, k=k, p=p, l=l):
                    return (
                        a0
                        if k == 0
                        else 0.5 * a0 + 0.5 * softmax(template_scores(cache, p, l, w), temp)
                    )

                aq = probability(qw)
                for method in plan["decoder"]["methods"] + (["FB_eTRCA"] if k in (3, 5) else []):
                    key = (int(subject), str(cell), method, k)
                    require(key in by_key, f"Missing row {key}")
                    row = by_key[key]
                    require(
                        row["fold_id"] == p % 3
                        and row["n_samples"] == int(cache["sample_counts"][l])
                        and row["interface"]
                        == plan["interfaces"][int(cache["interface_indices"][l])],
                        "Row identity",
                    )
                    permutations = 0 if method == "M_shuffle" else 1
                    if method in ("A0", "FB_eTRCA"):
                        if method == "A0":
                            probabilities = [a0]
                        else:
                            scores = cache["etrca_scores"][p, l, [3, 5].index(k)]
                            probabilities = [
                                softmax(scores, fold["eTRCA_temperatures"][str(k)]["temperature"])
                            ]
                        all_weights = None
                    elif method in ("pooled_fusion", "diagQ_fusion", "fullQ_fusion"):
                        cid = (
                            "uniform"
                            if method == "pooled_fusion"
                            else fold[
                                "diag_selection" if method == "diagQ_fusion" else "full_selection"
                            ]["candidate_id"]
                        )
                        w = qweights(cache, p, l, candidates[cid], k)
                        t = fold["support_temperatures"][cid][str(k)]["temperature"] if k else None
                        all_weights, probabilities = [w], [probability(w, t)]
                    else:
                        if method == "M_shuffle" and k in (3, 5):
                            perms = [
                                perm
                                for perm in itertools.permutations(range(k))
                                if all(i != j for i, j in enumerate(perm))
                            ]
                            permutations = len(perms)
                            all_weights = [
                                mweights(
                                    cache,
                                    p,
                                    l,
                                    qw,
                                    scale,
                                    available,
                                    fold["m_selection"]["beta"],
                                    perm,
                                )
                                for perm in perms
                            ]
                        elif method in ("AQM", "M_stale"):
                            all_weights = [
                                mweights(
                                    cache,
                                    p,
                                    l,
                                    qw,
                                    scale,
                                    available,
                                    fold["m_selection"]["beta"],
                                    stale=method == "M_stale",
                                )
                            ]
                        elif method == "order_control":
                            all_weights = [
                                order_weights(cache, qw, fold["order_selection"]["beta"])
                            ]
                        else:
                            all_weights = [qw]
                        probabilities = [probability(w) for w in all_weights]
                    require(row["permutations"] == permutations, "Exact derangement inventory")
                    calculated = [
                        metrics(prob, cache["query_labels"], a0, aq) for prob in probabilities
                    ]
                    for name in calculated[0]:
                        close(
                            row[name],
                            np.mean([m[name] for m in calculated], axis=0),
                            f"{key}/{name}",
                        )
                    if all_weights is None or k == 0:
                        require(
                            row["weight_turnover_vs_AQ"] is None
                            and row["effective_blocks"] is None,
                            "No inapplicable weights",
                        )
                    else:
                        turnover = np.mean(
                            [0.5 * np.abs(w - qw).sum(-1).mean() for w in all_weights]
                        )
                        effective = (
                            0.0
                            if k == 0
                            else np.mean([(1 / (w**2).sum(-1)).mean() for w in all_weights])
                        )
                        close(row["weight_turnover_vs_AQ"], turnover, "Weight turnover")
                        close(row["effective_blocks"], effective, "Effective blocks")
                    if method == "M_shuffle":
                        require(
                            row["posterior_sha256"] is None, "Shuffle has no averaged predictor"
                        )
                    elif (
                        method == "M_missing"
                        or k == 0
                        or (method in ("AQM", "M_stale", "order_control") and k in (0, 1))
                        or (method in ("AQM", "M_stale") and fold["m_selection"]["beta"] == 0)
                    ):
                        require(
                            row["posterior_sha256"]
                            == by_key[int(subject), str(cell), "AQ", k]["posterior_sha256"],
                            "Exact null posterior alias",
                        )
                    checked += 1
    return checked


def verify_summary(result, plan):
    rows, summary = result["rows"], result["summary"]
    ids = plan["source_subject_ids"]
    cells = [f"{e}-n{n}" for n in plan["sample_counts"] for e in plan["interfaces"]]
    by_key = {(r["subject_id"], r["cell_id"], r["method"], r["budget"]): r for r in rows}

    def value(method, budget, selected_cells=cells):
        if budget == "eauc":
            return (
                value(method, 0, selected_cells) / 6
                + value(method, 1, selected_cells) / 2
                + value(method, 3, selected_cells) / 3
            )
        return np.array(
            [
                [by_key[s, c, method, budget]["balanced_accuracy"] for s in ids]
                for c in selected_cells
            ]
        ).mean(axis=0)

    cost_cells = [f"{e}-n{plan['reporting']['cost_window_samples']}" for e in plan["interfaces"]]
    vectors = {
        "AQ_eauc_gain": value("AQ", "eauc") - value("A0", "eauc"),
        "M_eauc_gain": value("AQM", "eauc") - value("AQ", "eauc"),
        "M3_minus_shuffle3": value("AQM", 3) - value("M_shuffle", 3),
        "M3_minus_order3": value("AQM", 3) - value("order_control", 3),
        "M3_minus_stale3": value("AQM", 3) - value("M_stale", 3),
        "M3_minus_Q5": value("AQM", 3, cost_cells) - value("AQ", 5, cost_cells),
        "Q5_minus_Q3": value("AQ", 5, cost_cells) - value("AQ", 3, cost_cells),
        "M3_minus_eTRCA3": value("AQM", 3, cost_cells) - value("FB_eTRCA", 3, cost_cells),
        "one_second_M3_BA": value("AQM", 3, cost_cells),
        "one_second_Q5_BA": value("AQ", 5, cost_cells),
    }
    comparisons = {key: paired(vector) for key, vector in vectors.items()}
    require(set(summary["comparisons"]) == set(comparisons), "Comparison inventory")
    for key, expected in comparisons.items():
        require(set(summary["comparisons"][key]) == set(expected), "Paired field inventory")
        for field, number in expected.items():
            close(summary["comparisons"][key][field], number, f"{key}/{field}")

    def positive(key, floor=0, inclusive=False):
        mean = comparisons[key]["mean"]
        return bool(
            (mean >= floor if inclusive else mean > floor)
            and comparisons[key]["one_sided_95_LCB"] > 0
        )

    screen = {
        "AQ_utility": positive("AQ_eauc_gain", 0.01, True),
        "M_increment": positive("M_eauc_gain", 0.005, True),
        "shuffle_specificity": positive("M3_minus_shuffle3"),
        "order_specificity": positive("M3_minus_order3"),
        "stale_specificity": positive("M3_minus_stale3"),
        "Q_learning": positive("Q5_minus_Q3", 0.01, True),
        "cost_noninferiority": comparisons["M3_minus_Q5"]["one_sided_95_LCB"] > -1 / 60,
        "cost_target": comparisons["one_second_M3_BA"]["mean"] >= 0.8
        and comparisons["one_second_Q5_BA"]["mean"] >= 0.8,
        "classical_noninferiority": comparisons["M3_minus_eTRCA3"]["mean"] >= 0
        and comparisons["M3_minus_eTRCA3"]["one_sided_95_LCB"] > -1 / 60,
    }
    require(summary["screen_components"] == screen, "All prespecified screens")
    if not screen["AQ_utility"]:
        status = "AQ_NOT_ESTABLISHED"
    elif not all(
        screen[k]
        for k in ("M_increment", "shuffle_specificity", "order_specificity", "stale_specificity")
    ):
        status = "M_INCREMENT_NOT_ESTABLISHED"
    elif not all(screen.values()):
        status = "CALIBRATION_SAVING_NOT_ESTABLISHED"
    else:
        status = "READY_FOR_CONFIRMATORY_PROTOCOL"
    require(
        summary["status"] == status and summary["human_held_unlock"] is False,
        "Terminal status and held boundary",
    )
    methods = plan["decoder"]["methods"] + ["FB_eTRCA"]
    aggregate = {(r["cell_id"], r["method"], r["budget"]): r for r in summary["aggregate_rows"]}
    require(
        len(aggregate) == len(summary["aggregate_rows"]) == len(cells) * 42, "Aggregate inventory"
    )
    harm = {(r["subject_id"], r["method"], r["budget"]): r for r in summary["participant_harm"]}
    require(len(harm) == len(summary["participant_harm"]) == len(ids) * 42, "Harm inventory")
    attainment = {(r["cell_id"], r["method"]): r for r in summary["calibration_attainment"]}
    require(
        len(attainment) == len(summary["calibration_attainment"]) == len(cells) * len(methods),
        "Attainment inventory",
    )
    fields = [
        "balanced_accuracy",
        "correct_log_probability",
        "true_class_margin",
        "prediction_flips_vs_A0",
        "prediction_flips_vs_AQ",
        "correct_count",
        "query_count",
        "weight_turnover_vs_AQ",
        "effective_blocks",
    ]
    for method in methods:
        budgets = [3, 5] if method == "FB_eTRCA" else plan["budgets"]
        for k in budgets:
            for i, subject in enumerate(ids):
                record = harm[subject, method, k]
                close(record["equal_cell_mean_BA"], value(method, k)[i], "Harm BA")
                close(record["delta_vs_A0"], value(method, k)[i] - value("A0", k)[i], "Harm vs A0")
                close(record["delta_vs_AQ"], value(method, k)[i] - value("AQ", k)[i], "Harm vs AQ")
        for cell in cells:
            n = int(cell.split("-n")[1])
            curve = {}
            for k in budgets:
                record = aggregate[cell, method, k]
                require(
                    record["interface"] == cell.split("-n")[0] and record["n_samples"] == n,
                    "Aggregate identity",
                )
                group = [by_key[s, cell, method, k] for s in ids]
                require(set(record["mean_metrics"]) == set(fields), "Aggregate metric fields")
                for field in fields:
                    numbers = [r[field] for r in group]
                    if numbers[0] is None:
                        require(
                            all(v is None for v in numbers)
                            and record["mean_metrics"][field] is None,
                            "Inapplicable aggregate",
                        )
                    else:
                        close(record["mean_metrics"][field], np.mean(numbers), "Aggregate metric")
                curve[str(k)] = [r["balanced_accuracy"] for r in group]
                for reference in ("A0", "AQ"):
                    expected = np.mean(
                        [
                            r["balanced_accuracy"]
                            < by_key[s, cell, reference, k]["balanced_accuracy"]
                            for s, r in zip(ids, group, strict=True)
                        ]
                    )
                    close(record[f"fraction_harmed_vs_{reference}"], expected, "Per-cell harms")
            record = attainment[cell, method]
            require(
                record["subject_ids"] == ids
                and set(record["participant_accuracy_curve"]) == set(curve),
                "Attainment identity/budgets",
            )
            for k, numbers in curve.items():
                close(record["participant_accuracy_curve"][k], numbers, "Observed attainment curve")
            first = [
                next((k for k in budgets if curve[str(k)][i] >= 0.8), ">5") for i in range(len(ids))
            ]
            require(record["first_observed_budget"] == first, "First observed budget")
            require(
                record["labeled_trials"] == [12 * k if isinstance(k, int) else None for k in first],
                "Label costs with censoring",
            )
            for actual, k in zip(record["EEG_seconds"], first, strict=True):
                if isinstance(k, int):
                    close(actual, 12 * k * n / 250, "Analyzed EEG cost")
                else:
                    require(actual is None, "Unattained cost not imputed")
            expected = (
                "expected_permutation_metric_not_deployable_predictor"
                if method == "M_shuffle"
                else "observed_budget_only"
            )
            require(record["interpretation"] == expected, "Attainment interpretation")
    return {
        "status": status,
        "aggregate_rows_checked": len(aggregate),
        "participant_harm_checked": len(harm),
        "attainment_rows_checked": len(attainment),
    }


def verify_projection(cache, projection, plan):
    require(
        projection["returned_subject_ids"] == plan["source_subject_ids"], "Projected source IDs"
    )
    require(projection["columns"] == plan["manifest_columns"], "Acquisition-only columns")
    require(
        projection["returned_rows"] == plan["expected_metadata_packets"] * 12
        and projection["returned_packets"] == plan["expected_metadata_packets"],
        "Projection row counts",
    )
    packets = {(r["subject_id"], r["interface"], r["block_id"]): r for r in projection["packets"]}
    require(
        len(packets) == len(projection["packets"]) == plan["expected_metadata_packets"],
        "Unique packets",
    )
    for p, subject in enumerate(plan["source_subject_ids"]):
        orders = set()
        for e, interface in enumerate(plan["interfaces"]):
            for b in range(10):
                record = packets[subject, interface, b]
                orders.add(record["headband_order"])
                require(record["headband_order"] in plan["interfaces"], "Headband order")
                require(
                    record["condition_period"]
                    == ("first" if record["headband_order"] == interface else "second"),
                    "Nuisance order consistency",
                )
                values = np.asarray(record["impedance_kohm"], dtype=float)
                require(
                    values.shape == (8,)
                    and np.array_equal(values, cache["impedance"][p, e, b], equal_nan=True),
                    "Projected impedance/cache identity",
                )
        require(len(orders) == 1, "Participant headband order consistent")


def verify_provenance(root, plan_path, plan, start, projection, freeze, result, repo):
    require(
        {p.name for p in root.iterdir()} == set(plan["execution"]["artifacts"]),
        "Exact artifact inventory",
    )
    previous_mtime = 0
    for name in plan["execution"]["artifacts"]:
        info = (root / name).lstat()
        require(
            stat.S_ISREG(info.st_mode)
            and stat.S_IMODE(info.st_mode) == 0o400
            and info.st_nlink == 1,
            f"Exclusive readonly file {name}",
        )
        require(info.st_mtime_ns >= previous_mtime, "Publication timestamps")
        previous_mtime = info.st_mtime_ns
    require(
        sum(p.stat().st_size for p in root.iterdir()) <= plan["execution"]["resource_budget_bytes"],
        "Artifact quota",
    )
    for name, document in (
        ("start", start),
        ("projection", projection),
        ("fold-freezes", freeze),
        ("result", result),
    ):
        require(document["schema"] == f"cfeg.context-template-source.{name}.v1", f"Schema {name}")
        require(
            document["study_id"] == plan["study_id"] and document["plan_sha256"] == sha(plan_path),
            "Plan identity",
        )
        require(document["source_commit"] == start["source_commit"], "Source commit")
    for document in (projection, freeze, result):
        require(document["start_sha256"] == sha(root / "start.json"), "Start binding")
    for document in (freeze, result):
        require(
            document["source_projection_sha256"] == sha(root / "source-projection.json"),
            "Projection binding",
        )
        require(document["features_sha256"] == sha(root / "features.npz"), "Feature binding")
    require(result["fold_freezes_sha256"] == sha(root / "fold-freezes.json"), "Freeze binding")
    require(projection["manifest_sha256"] == plan["manifest_sha256"], "Manifest identity receipt")
    require(result["evidence_role"] == plan["evidence_role"], "Exposed-development evidence role")
    require(
        start["human_held_access"] is False and result["summary"]["human_held_unlock"] is False,
        "Held boundary",
    )
    require(start["source_subject_ids"] == plan["source_subject_ids"], "Start allowlist")
    require(start["workers"] == plan["execution"]["workers"], "Workers")
    require(
        start["runtime"]["blas_threads"]
        == {k: "1" for k in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS")},
        "BLAS threads",
    )

    def git(*args):
        return subprocess.check_output(["git", "-C", str(repo), *args])

    commit = start["source_commit"]
    require(
        git("rev-parse", f"{commit}^{{tree}}").decode().strip() == start["source_tree"],
        "Historical tree",
    )
    require(
        git("show", f"{commit}:configs/analysis/context_template_source39_v1.json")
        == plan_path.read_bytes(),
        "Historical plan",
    )
    require(start["pinned_files"] == plan["pinned_files"], "Pinned helper inventory")
    for filename, expected in plan["pinned_files"].items():
        require(
            hashlib.sha256(git("show", f"{commit}:{filename}")).hexdigest() == expected,
            "Historical helper bytes",
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    plan = json.loads(args.plan.read_text())
    start, projection, freeze, result = [
        json.loads((args.output / name).read_text())
        for name in ("start.json", "source-projection.json", "fold-freezes.json", "result.json")
    ]
    verify_provenance(
        args.output,
        args.plan,
        plan,
        start,
        projection,
        freeze,
        result,
        Path(__file__).resolve().parents[1],
    )
    with np.load(args.output / "features.npz", allow_pickle=False) as archive:
        cache = {name: archive[name] for name in archive.files}
    verify_features(cache, plan)
    bank = yaml.safe_load(
        (Path(__file__).resolve().parents[1] / plan["filterbank_config"]).read_text()
    )
    close(cache["filter_weights"], bank["weights"], "Pinned filterbank weights")
    verify_projection(cache, projection, plan)
    fits = verify_folds(cache, freeze, plan)
    count = verify_rows(cache, freeze, result, plan)
    report = verify_summary(result, plan)
    print(
        json.dumps(
            {
                "audit": "PASS",
                "fit_objectives_checked": fits,
                "metric_rows_checked": count,
                **report,
                "result_sha256": sha(args.output / "result.json"),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
