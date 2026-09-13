"""Source-only ridge gating of normalized quadrature PSD templates.

Pure computation: callers own immutable attempt ledgers and data provenance.
Query prediction accepts covariance/factors only, never a labeled trial record.
"""

from __future__ import annotations

import itertools

import numpy as np

ARMS = ("Q", "Q2", "QM", "SHAM")


def normalized_psd(covariance, factors):
    h = factors @ np.swapaxes(factors, -1, -2)
    traces = np.trace(h, axis1=-2, axis2=-1)
    small = traces <= 1e-12 * np.trace(covariance)
    out = h / np.maximum(traces, np.finfo(float).tiny)[..., None, None]
    out[small] = np.eye(h.shape[-1]) / h.shape[-1]
    return out, int(np.sum(small))


def oracle_lambda(target, prior, repeat):
    delta = prior - target
    denom = np.sum(delta * delta, axis=(-2, -1))
    numer = np.sum((repeat - target) * delta, axis=(-2, -1))
    ratio = np.divide(numer, denom, out=np.zeros_like(numer), where=denom > 1e-24)
    return np.clip(ratio, 0, 1), int(np.sum(denom <= 1e-24))


def fit_ridge(x, y):
    if (
        x.ndim != 2
        or y.shape != (len(x), 2)
        or not np.isfinite(x).all()
        or not np.isfinite(y).all()
    ):
        raise ValueError("ridge_input")
    mean, scale = x.mean(axis=0), x.std(axis=0)
    scale[scale < 1e-9] = 1
    z = (x - mean) / scale
    center_y = y.mean(axis=0)
    coef = np.linalg.solve(z.T @ z + 0.1 * np.eye(z.shape[1]), z.T @ (y - center_y))
    if not np.isfinite(coef).all():
        raise ValueError("ridge_nonfinite")
    return {"mean": mean, "scale": scale, "coef": coef, "intercept": center_y}


def predict_ridge(model, x):
    return np.clip(
        ((x - model["mean"]) / model["scale"]) @ model["coef"] + model["intercept"], 0, 1
    )


def score_query(covariance, factors, metric):
    """All classes are scored without access to the query label or DIN."""
    if metric.shape != (5, 2, covariance.shape[0], covariance.shape[0]):
        raise ValueError("metric_shape")
    numerator = np.einsum("jhcr,jhcd,jhdr->jh", factors, metric, factors, optimize=True)
    denominator = np.einsum("jhcd,dc->jh", metric, covariance, optimize=True)
    scores = (numerator / np.maximum(denominator, 1e-12)) @ np.array([1.0, 0.5])
    if not np.isfinite(scores).all():
        raise ValueError("nonfinite_scores")
    return scores


def sham_cycle(rows, m):
    """Cycle joint M vectors inside exact common-info strata, no external donors."""
    strata = {}
    for i, row in enumerate(rows):
        strata.setdefault((row["label"], row["k"], row["event_count"]), []).append(i)
    donors = np.arange(len(rows))
    singleton = 0
    for indices in strata.values():
        ordered = sorted(indices, key=lambda i: rows[i]["subject"])
        singleton += len(ordered) == 1
        for n, i in enumerate(ordered):
            donors[i] = ordered[(n + 1) % len(ordered)]
    exact_changed = np.any(m != m[donors], axis=1)
    changed = np.any(np.abs(m - m[donors]) > 1e-9, axis=1)
    return m[donors].copy(), {
        "donor_indices": donors.tolist(),
        "changed_fraction": float(changed.mean()),
        "exact_changed_fraction": float(exact_changed.mean()),
        "singleton_rows": singleton,
        "strata": len(strata),
    }


def prepare(data):
    """Validate/cache per-trial PSD and person means; no fitting or outcomes."""
    if "S001" in data:
        raise ValueError("development_subject_in_efficacy")
    if len(data) < 4:
        raise ValueError("insufficient_subjects")
    psd, means, fallback = {}, {}, 0
    for subject in sorted(data):
        for run in ("a", "b"):
            records = data[subject][run]
            if (
                len(records) != 15
                or [sum(t["label"] == j for t in records) for j in range(5)] != [3] * 5
            ):
                raise ValueError("trial_coverage")
            if any(a["start0"] >= b["start0"] for a, b in itertools.pairwise(records)):
                raise ValueError("trial_sample_order")
            if run == "b" and any(t["metadata"] is not None for t in records):
                raise ValueError("query_metadata_present")
            for n, trial in enumerate(records):
                # Only the labeled source/support PSD is cached. A target's b
                # PSD is used solely when that person is a SOURCE in another fold.
                value, count = normalized_psd(trial["covariance"], trial["factors"][trial["label"]])
                psd[subject, run, n] = value
                fallback += count
        means[subject] = np.stack(
            [
                np.mean(
                    [
                        psd[subject, run, n]
                        for run in ("a", "b")
                        for n, t in enumerate(data[subject][run])
                        if t["label"] == j
                    ],
                    axis=0,
                )
                for j in range(5)
            ]
        )
    return {"data": data, "psd": psd, "means": means, "fallback_count": fallback}


def support_case(prepared, subject, label, k):
    trials = prepared["data"][subject]["a"]
    indices = sorted(
        [n for n, t in enumerate(trials) if t["label"] == label], key=lambda n: trials[n]["start0"]
    )[:k]
    selected = [trials[n] for n in indices]
    if len(selected) != k or any(t["metadata"] is None for t in selected):
        raise ValueError("support_missing")
    q = np.mean([t["q"][label].reshape(8) for t in selected], axis=0)
    count = float(np.mean([t["event_count"] for t in selected]))
    return {
        "subject": subject,
        "label": label,
        "k": k,
        "event_count": count,
        "base": np.concatenate([q, np.eye(5)[label], [k, count, 2.0]]),
        "q2": np.mean([t["q2"][label] for t in selected], axis=0),
        "m": np.mean([t["metadata"] for t in selected], axis=0),
        "template": np.mean([prepared["psd"][subject, "a", n] for n in indices], axis=0),
        "prefix_samples": max(t["trial_end0"] for t in selected),
    }


def build_fold(prepared, target, k):
    source = sorted(set(prepared["data"]) - {target})
    if target in source or "S001" in source:
        raise ValueError("source_exclusion")
    rows, labels, prior_audit, degenerate = [], [], [], 0
    for pseudo in source:
        contributors = [s for s in source if s != pseudo]
        if target in contributors or pseudo in contributors or len(contributors) != len(source) - 1:
            raise ValueError("pseudo_prior_exclusion")
        prior = np.mean([prepared["means"][s] for s in contributors], axis=0)
        prior_audit.append(
            {"target": target, "pseudo_target": pseudo, "contributors": contributors}
        )
        for j in range(5):
            row = support_case(prepared, pseudo, j, k)
            repeat = np.mean(
                [
                    prepared["psd"][pseudo, "b", n]
                    for n, t in enumerate(prepared["data"][pseudo]["b"])
                    if t["label"] == j
                ],
                axis=0,
            )
            value, count = oracle_lambda(row["template"], prior[j], repeat)
            labels.append(value)
            degenerate += count
            rows.append(row)
    eval_rows = [support_case(prepared, target, j, k) for j in range(5)]
    train_m = np.stack([r["m"] for r in rows])
    class_m = np.stack(
        [np.mean([r["m"] for r in rows if r["label"] == j], axis=0) for j in range(5)]
    )
    centered = train_m - class_m[[r["label"] for r in rows]]
    sham, sham_audit = sham_cycle(rows, centered)
    eval_m = np.stack([r["m"] for r in eval_rows]) - class_m
    base = np.stack([r["base"] for r in rows])
    eval_base = np.stack([r["base"] for r in eval_rows])
    inputs = {
        "Q": base,
        "Q2": np.column_stack([base, np.stack([r["q2"] for r in rows])]),
        "QM": np.column_stack([base, centered]),
        "SHAM": np.column_stack([base, sham]),
    }
    eval_inputs = {
        "Q": eval_base,
        "Q2": np.column_stack([eval_base, np.stack([r["q2"] for r in eval_rows])]),
        "QM": np.column_stack([eval_base, eval_m]),
        "SHAM": np.column_stack([eval_base, eval_m]),
    }
    if not all(
        np.isfinite(x).all() for x in [*inputs.values(), *eval_inputs.values(), np.array(labels)]
    ):
        raise ValueError("fold_nonfinite")
    return {
        "target": target,
        "k": k,
        "source": source,
        "x": inputs,
        "eval_x": eval_inputs,
        "y": np.array(labels),
        "class_mean_m": class_m,
        "template": np.stack([r["template"] for r in eval_rows]),
        "prior": np.mean([prepared["means"][s] for s in source], axis=0),
        "prefix_samples": max(r["prefix_samples"] for r in eval_rows),
        "rows": [{key: r[key] for key in ("subject", "label", "k", "event_count")} for r in rows],
        "sham": sham_audit,
        "prior_audit": prior_audit,
        "degenerate_oracles": degenerate,
    }


def preflight(prepared):
    folds = [build_fold(prepared, target, k) for target in sorted(prepared["data"]) for k in (1, 2)]
    if not any(f["sham"]["changed_fraction"] > 0 for f in folds):
        raise ValueError("no_conditional_m_or_sham_change")
    return folds


def run_folds(prepared, folds, ledger):
    """Caller provides append-only durable ledger(kind, payload), called before fits."""
    results, fit_count = [], 0
    for fold in folds:
        target, k = fold["target"], fold["k"]
        metrics, models, lambdas = {}, {}, {}
        for arm in ARMS:
            identity = {"target": target, "k": k, "arm": arm, "fit_index": fit_count + 1}
            ledger("FIT_STARTED", identity)
            fit_count += 1
            model = fit_ridge(fold["x"][arm], fold["y"])
            lam = predict_ridge(model, fold["eval_x"][arm])
            metrics[arm] = (1 - lam[..., None, None]) * fold["template"] + lam[
                ..., None, None
            ] * fold["prior"]
            models[arm], lambdas[arm] = model, lam
            ledger("FIT_COMPLETE", identity)
        metrics["TARGET_ONLY"], metrics["SOURCE_ONLY"] = fold["template"], fold["prior"]
        scores = {arm: [] for arm in (*metrics, "ZERO_SHOT")}
        truths = []
        for query in prepared["data"][target]["b"]:
            for arm, metric in metrics.items():
                scores[arm].append(score_query(query["covariance"], query["factors"], metric))
            scores["ZERO_SHOT"].append(query["zero_shot"])
            # Labels are collected only AFTER every arm has made all-candidate scores.
            truths.append(query["label"])
        predictions = {a: np.argmax(np.array(v), axis=1) for a, v in scores.items()}
        accuracy = {a: float(np.mean(v == truths)) for a, v in predictions.items()}
        results.append(
            {
                "target": target,
                "k": k,
                "fit_count": 4,
                "models": models,
                "lambdas": lambdas,
                "train_x": fold["x"],
                "eval_x": fold["eval_x"],
                "oracle": fold["y"],
                "class_mean_m": fold["class_mean_m"],
                "rows": fold["rows"],
                "sham": fold["sham"],
                "prior_audit": fold["prior_audit"],
                "degenerate_oracles": fold["degenerate_oracles"],
                "truth": truths,
                "scores": scores,
                "predictions": predictions,
                "accuracy": accuracy,
                "support_prefix_samples": fold["prefix_samples"],
                "support_prefix_seconds": fold["prefix_samples"] / 250,
                "query_ready_elapsed_seconds": None,
            }
        )
    return {"fits": fit_count, "folds": results, "psd_fallback_count": prepared["fallback_count"]}


def summarize(result):
    rows = {(r["target"], r["k"]): r for r in result["folds"]}
    people = sorted({s for s, _ in rows})
    rng = np.random.default_rng(20260913)
    draws = rng.integers(0, len(people), (10000, len(people)))
    comparisons = {}
    for label, comparator, k in (
        ("QM1-Q1", "Q", 1),
        ("QM1-Q2_1", "Q2", 1),
        ("QM1-SHAM1", "SHAM", 1),
        ("QM1-Q_k2", "Q", 2),
    ):
        delta = np.array(
            [rows[s, 1]["accuracy"]["QM"] - rows[s, k]["accuracy"][comparator] for s in people]
        )
        comparisons[label] = {
            "mean": float(delta.mean()),
            "ci95": np.quantile(delta[draws].mean(axis=1), [0.025, 0.975]).tolist(),
            "participant_deltas": delta.tolist(),
            "harm_over_5pp": int(np.sum(delta < -0.05)),
        }
    cost = np.array(
        [
            rows[s, 2]["support_prefix_seconds"] - rows[s, 1]["support_prefix_seconds"]
            for s in people
        ]
    )
    benefit = all(
        comparisons[l]["mean"] >= 0.01 and comparisons[l]["ci95"][0] > 0
        for l in ("QM1-Q1", "QM1-Q2_1", "QM1-SHAM1")
    )
    reduction = comparisons["QM1-Q_k2"]["ci95"][0] >= -0.02 and bool(np.all(cost > 0))
    means = {
        str(k): {
            a: float(np.mean([rows[s, k]["accuracy"][a] for s in people]))
            for a in (*ARMS, "TARGET_ONLY", "SOURCE_ONLY", "ZERO_SHOT")
        }
        for k in (1, 2)
    }
    return {
        "subjects": people,
        "comparisons": comparisons,
        "mean_accuracy": means,
        "support_prefix_difference_seconds": cost.tolist(),
        "break_even_added_setup_seconds": cost.tolist(),
        "actual_query_ready_time_savings": None,
        "zero_shot_at_least_80_percent": [
            s for s in people if rows[s, 1]["accuracy"]["ZERO_SHOT"] >= 0.8
        ],
        "decision": "PROMISING_EXPLORATORY_ONLY"
        if benefit and reduction
        else "RETIRE_UNDER_FROZEN_PROTOCOL",
        "limits": [
            "10-subject exploratory screen, multiple comparisons",
            "inferred DIN labels",
            "offline nominal run order",
            "support-prefix is not actual query-ready elapsed",
            "no held60 or independent confirmation",
        ],
    }
