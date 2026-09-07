"""Frozen, metadata-free spatial calibration development on exposed source39.

Pure array kernels never read inputs. Only the guarded single-attempt runner
opens allowlisted EEG files, after durable start publication. No study RNG.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import platform
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import scipy
import yaml
from scipy.linalg import eigh

from cfeg.analysis import metadata_calibration_v4_pilot as numeric
from cfeg.baselines.calibration import predict_filterbank_ensemble_trca
from cfeg.baselines.fbcca import apply_filterbank
from cfeg.data.prepare_mat import _load_arrays
from cfeg.data.prepare_wearable import _wearable_data

PLAN_SHA256 = "8198c28a36f43465fb2f7c62dfa1e0be2f6415aa11740c1dde906ab0cd157552"
SOURCE_IDS = (
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
RAW_ROOT = Path("/home/whwovy/eeg-data/raw/wearable")
SCORE_SCHEMA = "cfeg.spatial-calibration-source.scores.v1"
BUDGETS = (1, 3, 5)
TRCA_BUDGETS = (3, 5)
SCORE_KEYS = ("a0_scores", "itcca_scores", "etrca_unit_scores", "legacy_etrca_scores")
STANDALONE_KEYS = {
    "ITCCA": "itcca_scores",
    "FB_eTRCA_unit": "etrca_unit_scores",
    "legacy_FB_eTRCA": "legacy_etrca_scores",
}
METRIC_KEYS = (
    "balanced_accuracy",
    "correct_log_probability",
    "true_class_margin",
    "prediction_flips_vs_A0",
    "correct_count",
    "query_count",
    "class_correct_counts",
    "class_query_counts",
)
_WORKER = None


def cells(plan: dict) -> list[dict]:
    return [
        {
            "cell_id": f"{interface}-n{n}",
            "interface": interface,
            "interface_index": i,
            "n_samples": n,
        }
        for n in plan["sample_counts"]
        for i, interface in enumerate(plan["interfaces"])
    ]


def file_sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def array_sha(value: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def _trials(value: np.ndarray) -> np.ndarray:
    value = np.ascontiguousarray(value, dtype=np.float64)
    if (
        value.ndim != 3
        or min(value.shape) < 1
        or value.shape[-1] < 2
        or not np.isfinite(value).all()
    ):
        raise ValueError("Expected finite nonempty [trial,channel,time>=2] arrays.")
    return value


def whiten_trials(value: np.ndarray, ridge: float) -> np.ndarray:
    """Whiten separately per trial, with public cca_score's numerical rules."""
    value = _trials(value)
    if not math.isfinite(ridge) or ridge < 0:
        raise ValueError("CCA ridge must be finite and nonnegative.")
    centered = value - value.mean(axis=-1, keepdims=True)
    samples, channels = value.shape[-1], value.shape[-2]
    covariance = centered @ centered.swapaxes(-1, -2) / (samples - 1)
    scales = np.trace(covariance, axis1=-2, axis2=-1) / channels
    covariance += ridge * scales[:, None, None] * np.eye(channels)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    cutoff = np.maximum(np.finfo(np.float64).eps, np.maximum(eigenvalues[:, -1], 0) * 1e-12)
    roots = np.zeros_like(eigenvalues)
    np.divide(
        1, np.sqrt(np.maximum(eigenvalues, 0)), out=roots, where=eigenvalues > cutoff[:, None]
    )
    inverse = (eigenvectors * roots[:, None, :]) @ eigenvectors.swapaxes(-1, -2)
    whitened = inverse @ centered / np.sqrt(samples - 1)
    floor = np.finfo(np.float64).eps * np.sqrt(samples) * 10
    valid = (np.linalg.norm(centered, axis=(-2, -1)) > floor) & (scales > 0)
    return np.where(valid[:, None, None], whitened, 0.0)


def whitened_pairwise_cca(query: np.ndarray, templates: np.ndarray) -> np.ndarray:
    cross = np.einsum("qht,cjt->qchj", query, templates)
    return np.clip(np.linalg.svd(cross, compute_uv=False)[..., 0], 0, 1)


def pairwise_regularized_cca(
    query: np.ndarray, templates: np.ndarray, ridge: float = 0.01
) -> np.ndarray:
    query, templates = _trials(query), _trials(templates)
    if query.shape[-1] != templates.shape[-1]:
        raise ValueError("Query/template time windows must match.")
    return whitened_pairwise_cca(whiten_trials(query, ridge), whiten_trials(templates, ridge))


def unit_trca_filters(support: np.ndarray, ridge: float = 0.001) -> tuple[np.ndarray, np.ndarray]:
    """Explicit unit-filter TRCA variant, not an author-bitwise reproduction."""
    support = np.ascontiguousarray(support, dtype=np.float64)
    if (
        support.ndim != 4
        or support.shape[0] < 2
        or min(support.shape[1:]) < 1
        or support.shape[-1] < 2
        or not np.isfinite(support).all()
        or not math.isfinite(ridge)
        or ridge <= 0
    ):
        raise ValueError("TRCA needs >=2 finite trials per class and a positive ridge.")
    centered = support - support.mean(axis=-1, keepdims=True)
    templates = centered.mean(axis=0)
    filters = []
    for label in range(support.shape[1]):
        x = centered[:, label]
        q = np.einsum("bht,bjt->hj", x, x)
        summed = x.sum(axis=0)
        s = summed @ summed.T - q
        q, s = (q + q.T) / 2, (s + s.T) / 2
        regularizer = ridge * max(float(np.trace(q)) / len(q), 1e-12)
        values, vectors = eigh(s, q + regularizer * np.eye(len(q)), check_finite=True)
        vector = vectors[:, int(np.argmax(values))]
        vector = vector / np.linalg.norm(vector)
        if vector[np.argmax(np.abs(vector))] < 0:
            vector = -vector
        filters.append(vector)
    return templates, np.stack(filters, axis=1)


def unit_trca_scores(query: np.ndarray, templates: np.ndarray, filters: np.ndarray) -> np.ndarray:
    query, templates = _trials(query), _trials(templates)
    filters = np.asarray(filters, dtype=np.float64)
    if (
        query.shape[1:] != templates.shape[1:]
        or filters.shape != (query.shape[1], len(templates))
        or not np.isfinite(filters).all()
    ):
        raise ValueError("TRCA query, templates and ensemble filters must align.")
    query = query - query.mean(axis=-1, keepdims=True)
    templates = templates - templates.mean(axis=-1, keepdims=True)
    q = np.einsum("hc,qht->qct", filters, query)
    t = np.einsum("hc,lht->lct", filters, templates)
    q = (q - q.mean(axis=-1, keepdims=True)).reshape(len(q), -1)
    t = (t - t.mean(axis=-1, keepdims=True)).reshape(len(t), -1)
    denominator = np.linalg.norm(q, axis=1)[:, None] * np.linalg.norm(t, axis=1)[None, :]
    result = np.zeros((len(q), len(t)), dtype=np.float64)
    np.divide(q @ t.T, denominator, out=result, where=denominator > 0)
    return np.clip(result, -1, 1)


def build_workspace(plan: dict, filterbank: dict, samples: int) -> dict:
    return {
        "plan": plan,
        "filterbank": filterbank,
        "cca_cache": numeric.reference_cache(
            np.asarray(plan["frequencies"]),
            plan["sfreq"],
            samples,
            filterbank["n_harmonics"],
            filterbank["regularization"],
        ),
    }


def extract_cell(crop: np.ndarray, workspace: dict) -> dict:
    """Native [10,class,channel,time] crop; support labels implicit in class axis."""
    crop = np.ascontiguousarray(crop, dtype=np.float64)
    plan, bank = workspace["plan"], workspace["filterbank"]
    if crop.ndim != 4 or crop.shape[:2] != (10, plan["n_classes"]) or not np.isfinite(crop).all():
        raise ValueError("Crop violates the finite chronological trial grid.")
    _, classes, channels, samples = crop.shape
    query = crop[5:].reshape(-1, channels, samples)
    filtered, parameters = apply_filterbank(
        crop.reshape(-1, channels, samples), sfreq=plan["sfreq"], filterbank=bank
    )
    weights = np.asarray(parameters["weights"], dtype=np.float64)
    f = filtered.reshape(len(weights), 10, classes, channels, samples)
    fq = f[:, 5:].reshape(len(weights), -1, channels, samples)
    a0 = np.stack(
        [
            numeric.cached_cca_scores(band, workspace["cca_cache"], parameters["regularization"])
            for band in fq
        ]
    )
    itcca, trca = [], []
    whitened_queries = [whiten_trials(band, plan["decoder"]["ITCCA_ridge"]) for band in fq]
    for k in BUDGETS:
        correlations = np.stack(
            [
                whitened_pairwise_cca(
                    whitened_queries[b],
                    whiten_trials(f[b, :k].mean(axis=0), plan["decoder"]["ITCCA_ridge"]),
                )
                for b in range(len(f))
            ]
        )
        itcca.append(np.einsum("f,fqc->qc", weights, correlations**2) / weights.sum())
        if k in TRCA_BUDGETS:
            correlations = np.stack(
                [
                    unit_trca_scores(
                        fq[b], *unit_trca_filters(f[b, :k], plan["decoder"]["FB_eTRCA_unit_ridge"])
                    )
                    for b in range(len(f))
                ]
            )
            trca.append(np.einsum("f,fqc->qc", weights, correlations) / weights.sum())
    legacy = np.stack(
        [
            predict_filterbank_ensemble_trca(
                query,
                crop[:k].reshape(-1, channels, samples),
                np.tile(np.arange(classes), k),
                n_classes=classes,
                sfreq=plan["sfreq"],
                filterbank=bank,
                ridge_ratio=1e-6,
            ).scores
            for k in TRCA_BUDGETS
        ]
    )
    return {
        "a0_scores": np.einsum("f,fqc->qc", weights, a0**2) / weights.sum(),
        "itcca_scores": np.stack(itcca),
        "etrca_unit_scores": np.stack(trca),
        "legacy_etrca_scores": legacy,
    }


def extract_scores(raw_array: np.ndarray, plan: dict, workspaces: dict) -> dict:
    if raw_array.shape != tuple(plan["raw_shape"]) or not np.isfinite(raw_array).all():
        raise ValueError("Raw shape or finite-value contract differs.")
    records, hashes = [], []
    for cell in cells(plan):
        begin, n = plan["start_sample"], cell["n_samples"]
        crop = np.ascontiguousarray(
            raw_array[:, begin : begin + n, cell["interface_index"]].transpose(2, 3, 0, 1),
            dtype=np.float64,
        )
        if crop.shape[-1] != n:
            raise ValueError("Raw crop exceeds recorded samples.")
        hashes.append(array_sha(crop))
        records.append(extract_cell(crop, workspaces[n]))
    return {
        **{key: np.stack([r[key] for r in records]) for key in SCORE_KEYS},
        "crop_sha256": np.asarray(hashes),
    }


def initialize_worker(plan: dict, filterbank: dict) -> None:
    global _WORKER
    _WORKER = (plan, {n: build_workspace(plan, filterbank, n) for n in plan["sample_counts"]})


def allowed_raw_path(subject: int, plan: dict) -> Path:
    if (
        type(subject) is not int
        or subject not in SOURCE_IDS
        or tuple(plan["source_subject_ids"]) != SOURCE_IDS
        or Path(plan["raw_root"]) != RAW_ROOT
    ):
        raise ValueError("Immutable exposed source39 input boundary differs.")
    path = RAW_ROOT / f"S{subject:03d}.mat"
    if path.is_symlink() or path.resolve() != path:
        raise ValueError("Raw paths must be regular non-symlink allowlisted paths.")
    return path


def load_participant(subject: int) -> dict:
    if _WORKER is None:
        raise RuntimeError("Workers must be initialized after durable study start.")
    plan, workspaces = _WORKER
    path = allowed_raw_path(subject, plan)
    digest = file_sha(path)
    raw = _wearable_data(_load_arrays(path))
    return {
        "subject_id": subject,
        "raw_file_sha256": digest,
        **extract_scores(raw, plan, workspaces),
    }


def collect_scores(plan: dict, filterbank: dict, workers: int) -> dict:
    results = []
    with ProcessPoolExecutor(
        max_workers=workers, initializer=initialize_worker, initargs=(plan, filterbank)
    ) as executor:
        pending = [executor.submit(load_participant, s) for s in plan["source_subject_ids"]]
        for future in as_completed(pending):
            results.append(future.result())
            print(
                json.dumps(
                    {
                        "phase": "score_extraction",
                        "completed": len(results),
                        "total": len(SOURCE_IDS),
                    }
                ),
                flush=True,
            )
    results.sort(key=lambda r: r["subject_id"])
    cs = cells(plan)
    return {
        **{key: np.stack([r[key] for r in results]) for key in (*SCORE_KEYS, "crop_sha256")},
        "schema": np.asarray(SCORE_SCHEMA),
        "subject_ids": np.asarray([r["subject_id"] for r in results]),
        "cell_ids": np.asarray([c["cell_id"] for c in cs]),
        "sample_counts": np.asarray([c["n_samples"] for c in cs]),
        "interface_indices": np.asarray([c["interface_index"] for c in cs]),
        "query_labels": np.tile(np.arange(plan["n_classes"]), 5),
        "query_block_ids": np.repeat(plan["query_blocks"], plan["n_classes"]),
        "support_block_ids": np.asarray(plan["support_blocks"]),
        "calibration_budgets": np.asarray(BUDGETS),
        "trca_budgets": np.asarray(TRCA_BUDGETS),
        "raw_file_sha256": np.asarray([r["raw_file_sha256"] for r in results]),
        "filter_weights": np.asarray(filterbank["weights"], dtype=np.float64),
    }


def nll(probability: np.ndarray, labels: np.ndarray) -> float:
    truth = np.take_along_axis(
        probability, np.broadcast_to(labels, probability.shape[:-1])[..., None], axis=-1
    )[..., 0]
    return float(np.mean(-np.log(np.maximum(truth, 1e-300))))


def temperature_fit(scores: np.ndarray, labels: np.ndarray) -> dict:
    temperatures = np.logspace(-4, 1, 31)
    objectives = [nll(numeric.softmax(scores, t), labels) for t in temperatures]
    index = int(np.argmin(objectives))
    return {
        "temperatures": temperatures.tolist(),
        "objectives": objectives,
        "selected_index": index,
        "temperature": float(temperatures[index]),
        "nll": objectives[index],
    }


def fused_probability(
    p0: np.ndarray, scores: np.ndarray, temperature: float | None, strength: float
) -> np.ndarray:
    if strength == 0:
        return p0
    if not 0 < strength <= 0.75 or temperature is None or temperature <= 0:
        raise ValueError("Invalid frozen fusion parameters.")
    return (1 - strength) * p0 + strength * numeric.softmax(scores, temperature)


def fusion_fit(p0: np.ndarray, scores: np.ndarray, labels: np.ndarray, plan: dict) -> dict:
    candidates = [{"lambda": 0.0, "temperature": None, "nll": nll(p0, labels)}]
    for strength in plan["fitting"]["lambda_grid"][1:]:
        for t in np.logspace(-4, 1, 31):
            candidates.append(
                {
                    "lambda": strength,
                    "temperature": float(t),
                    "nll": nll(fused_probability(p0, scores, t, strength), labels),
                }
            )
    index = int(np.argmin([r["nll"] for r in candidates]))
    return {"candidates": candidates, "selected_index": index, **candidates[index]}


def fit_fold(cache: dict, fit_indices: np.ndarray, plan: dict, *, fold_id: int) -> dict:
    positions = np.arange(len(cache["subject_ids"]))
    expected, evaluation = (
        np.flatnonzero(positions % 3 != fold_id),
        np.flatnonzero(positions % 3 == fold_id),
    )
    if fold_id not in (0, 1, 2) or not np.array_equal(np.asarray(fit_indices), expected):
        raise ValueError("Fit identities must match the deterministic outer split.")
    windows = {}
    for n in plan["sample_counts"]:
        indices = np.flatnonzero(cache["sample_counts"] == n)
        a0 = cache["a0_scores"][expected][:, indices]
        a0fit = temperature_fit(a0, cache["query_labels"])
        p0 = numeric.softmax(a0, a0fit["temperature"])
        budgets = {}
        for k in BUDGETS:
            scores = cache["itcca_scores"][expected][:, indices, BUDGETS.index(k)]
            standalone = {}
            for method, key in STANDALONE_KEYS.items():
                available = BUDGETS if method == "ITCCA" else TRCA_BUDGETS
                if k in available:
                    standalone[method] = temperature_fit(
                        cache[key][expected][:, indices, available.index(k)], cache["query_labels"]
                    )
            budgets[str(k)] = {
                "fusion_fit": fusion_fit(p0, scores, cache["query_labels"], plan),
                "standalone_temperatures": standalone,
            }
        windows[str(n)] = {"a0_temperature": a0fit, "budgets": budgets}
    return {
        "fold_id": fold_id,
        "fit_subject_ids": cache["subject_ids"][expected].tolist(),
        "evaluation_subject_ids": cache["subject_ids"][evaluation].tolist(),
        "windows": windows,
    }


def posterior_metrics(probability: np.ndarray, labels: np.ndarray, a0: np.ndarray) -> dict:
    metrics = numeric.posterior_metrics(probability, labels, a0, a0)
    return {key: metrics[key] for key in METRIC_KEYS}


def mean_metrics(records: list[dict]) -> dict:
    return {key: np.mean([r[key] for r in records], axis=0).tolist() for key in METRIC_KEYS}


def evaluate_fold(cache: dict, eval_indices: np.ndarray, freeze: dict, plan: dict) -> list[dict]:
    expected = np.flatnonzero(np.arange(len(cache["subject_ids"])) % 3 == freeze["fold_id"])
    if (
        not np.array_equal(np.asarray(eval_indices), expected)
        or cache["subject_ids"][expected].tolist() != freeze["evaluation_subject_ids"]
    ):
        raise ValueError("Evaluation identities differ from the frozen outer split.")
    labels, rows = cache["query_labels"], []
    for p in expected:
        for c, cell in enumerate(cells(plan)):
            window = freeze["windows"][str(cell["n_samples"])]
            p0 = numeric.softmax(cache["a0_scores"][p, c], window["a0_temperature"]["temperature"])
            for k in plan["budgets"]:
                probabilities = {"A0": p0}
                permutations = []
                if k == 0:
                    for method in ("AQ_ITCCA", "uniform_shrinkage", "wrong_label_support"):
                        probabilities[method] = p0
                else:
                    fits = window["budgets"][str(k)]
                    selected = fits["fusion_fit"]
                    scores = cache["itcca_scores"][p, c, BUDGETS.index(k)]
                    strength, t = selected["lambda"], selected["temperature"]
                    probabilities["AQ_ITCCA"] = fused_probability(p0, scores, t, strength)
                    uniform = (
                        p0 if strength == 0 else (1 - strength) * p0 + strength / plan["n_classes"]
                    )
                    if not np.array_equal(uniform.argmax(axis=-1), p0.argmax(axis=-1)):
                        raise RuntimeError("Uniform shrinkage changed A0 decisions.")
                    probabilities["uniform_shrinkage"] = uniform
                    for shift in range(1, plan["n_classes"]):
                        probability = fused_probability(
                            p0, np.roll(scores, shift, axis=-1), t, strength
                        )
                        permutations.append(
                            {"shift": shift, **posterior_metrics(probability, labels, p0)}
                        )
                    for method, key in STANDALONE_KEYS.items():
                        available = BUDGETS if method == "ITCCA" else TRCA_BUDGETS
                        if k in available:
                            probabilities[method] = numeric.softmax(
                                cache[key][p, c, available.index(k)],
                                fits["standalone_temperatures"][method]["temperature"],
                            )
                for method, available in plan["decoder"]["method_budgets"].items():
                    if k not in available:
                        continue
                    if method == "wrong_label_support" and k > 0:
                        metrics, digest, count = mean_metrics(permutations), None, len(permutations)
                    else:
                        posterior = probabilities[method]
                        metrics, digest = (
                            posterior_metrics(posterior, labels, p0),
                            array_sha(posterior),
                        )
                        count = 0 if method == "wrong_label_support" else 1
                    row = {
                        "subject_id": int(cache["subject_ids"][p]),
                        "fold_id": freeze["fold_id"],
                        **{key: value for key, value in cell.items() if key != "interface_index"},
                        "method": method,
                        "budget": k,
                        "permutations": count,
                        "posterior_sha256": digest,
                        **metrics,
                    }
                    if method == "wrong_label_support":
                        row["permutation_metrics"] = permutations
                    rows.append(row)
        print(
            json.dumps(
                {
                    "phase": "outer_evaluation",
                    "fold": freeze["fold_id"],
                    "subject_completed": int(cache["subject_ids"][p]),
                }
            ),
            flush=True,
        )
    return rows


def summarize_results(rows: list[dict], plan: dict) -> dict:
    ids, cs = sorted({r["subject_id"] for r in rows}), cells(plan)
    lookup = {(r["subject_id"], r["cell_id"], r["method"], r["budget"]): r for r in rows}
    if len(lookup) != len(rows):
        raise ValueError("Duplicate participant/cell/method/budget rows.")

    def values(cid, method, budget):
        if budget == "eauc":
            return sum(w * values(cid, method, k) for k, w in ((0, 1 / 6), (1, 0.5), (3, 1 / 3)))
        return np.asarray([lookup[s, cid, method, budget]["balanced_accuracy"] for s in ids])

    def average(method, budget, interface=None):
        return np.mean(
            [
                values(c["cell_id"], method, budget)
                for c in cs
                if interface is None or c["interface"] == interface
            ],
            axis=0,
        )

    comparisons = {
        "AQ_eauc_gain": numeric.paired_summary(average("AQ_ITCCA", "eauc") - average("A0", "eauc")),
        "supervised_eauc_gain": numeric.paired_summary(
            average("AQ_ITCCA", "eauc") - average("wrong_label_support", "eauc")
        ),
        "AQ_k1_gain": numeric.paired_summary(average("AQ_ITCCA", 1) - average("A0", 1)),
        **{
            f"{interface}_eauc_gain": numeric.paired_summary(
                average("AQ_ITCCA", "eauc", interface) - average("A0", "eauc", interface)
            )
            for interface in plan["interfaces"]
        },
        "AQ5_minus_AQ3": numeric.paired_summary(average("AQ_ITCCA", 5) - average("AQ_ITCCA", 3)),
    }
    screen = {
        "AQ_utility": comparisons["AQ_eauc_gain"]["mean"] >= 0.01
        and comparisons["AQ_eauc_gain"]["one_sided_95_LCB"] > 0,
        "supervised_specificity": comparisons["supervised_eauc_gain"]["mean"] >= 0.01
        and comparisons["supervised_eauc_gain"]["one_sided_95_LCB"] > 0,
        "k1_safety": comparisons["AQ_k1_gain"]["mean"] >= 0,
        **{
            f"{interface}_safety": comparisons[f"{interface}_eauc_gain"]["mean"] >= -0.005
            and comparisons[f"{interface}_eauc_gain"]["one_sided_95_LCB"] > -1 / 60
            for interface in plan["interfaces"]
        },
    }
    aggregates, harms, attainment = [], [], []
    for cell in cs:
        cid = cell["cell_id"]
        for method, budgets in plan["decoder"]["method_budgets"].items():
            curves = {str(k): values(cid, method, k).tolist() for k in budgets}
            for k in budgets:
                aggregates.append(
                    {
                        **{key: value for key, value in cell.items() if key != "interface_index"},
                        "method": method,
                        "budget": k,
                        "mean_metrics": mean_metrics([lookup[s, cid, method, k] for s in ids]),
                        "fraction_harmed_vs_A0": float(
                            np.mean(values(cid, method, k) < values(cid, "A0", k))
                        ),
                    }
                )
            first = [
                next((k for k in budgets if curves[str(k)][i] >= 0.8), ">5")
                for i in range(len(ids))
            ]
            attainment.append(
                {
                    "cell_id": cid,
                    "method": method,
                    "subject_ids": ids,
                    "participant_accuracy_curve": curves,
                    "first_observed_budget": first,
                    "labeled_trials": [
                        plan["n_classes"] * k if isinstance(k, int) else None for k in first
                    ],
                    "EEG_seconds": [
                        plan["n_classes"] * k * cell["n_samples"] / plan["sfreq"]
                        if isinstance(k, int)
                        else None
                        for k in first
                    ],
                    "interpretation": "expected_permutation_metric_not_deployable_predictor"
                    if method == "wrong_label_support"
                    else "observed_budget_only",
                }
            )
    for method, budgets in plan["decoder"]["method_budgets"].items():
        for k in budgets:
            value, a0 = average(method, k), average("A0", k)
            harms.extend(
                {
                    "subject_id": subject,
                    "method": method,
                    "budget": k,
                    "equal_cell_mean_BA": float(value[i]),
                    "delta_vs_A0": float(value[i] - a0[i]),
                }
                for i, subject in enumerate(ids)
            )
    return {
        "status": "USEFUL_EEG_CALIBRATION_DEVELOPMENT"
        if all(screen.values())
        else "AQ_NOT_ESTABLISHED",
        "human_held_unlock": False,
        "comparisons": comparisons,
        "screen_components": screen,
        "aggregate_rows": aggregates,
        "participant_harm": harms,
        "calibration_attainment": attainment,
    }


def check_quota(output: Path, addition: int, quota: int) -> None:
    current = sum(p.stat().st_size for p in output.iterdir() if p.is_file())
    if current + addition > quota:
        raise RuntimeError("Study artifact quota would be exceeded.")


def write_json(path: Path, payload: dict, quota: int) -> str:
    size = len((json.dumps(payload, sort_keys=True, indent=2, allow_nan=False) + "\n").encode())
    check_quota(path.parent, size, quota)
    return numeric.write_json_exclusive(path, payload)


def write_scores(path: Path, cache: dict, quota: int) -> str:
    check_quota(path.parent, sum(v.nbytes for v in cache.values()) + 65536, quota)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o400)
    with os.fdopen(descriptor, "wb") as handle:
        np.savez_compressed(handle, **cache)
        handle.flush()
        os.fsync(handle.fileno())
    directory = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)
    return file_sha(path)


def execute_phases(
    plan: dict, filterbank: dict, output: Path, start: dict, start_sha: str, workers: int
) -> dict:
    quota = plan["execution"]["resource_budget_bytes"]
    cache = collect_scores(plan, filterbank, workers)
    scores_sha = write_scores(output / "scores.npz", cache, quota)
    provenance = {
        "study_id": plan["study_id"],
        "source_commit": start["source_commit"],
        "plan_sha256": start["plan_sha256"],
        "start_sha256": start_sha,
        "scores_sha256": scores_sha,
    }
    folds = []
    for fold in range(3):
        folds.append(
            fit_fold(
                cache,
                np.flatnonzero(np.arange(len(cache["subject_ids"])) % 3 != fold),
                plan,
                fold_id=fold,
            )
        )
        print(json.dumps({"phase": "fit", "fold_completed": fold}), flush=True)
    freezes = {
        "schema": "cfeg.spatial-calibration-source.fold-freezes.v1",
        **provenance,
        "folds": folds,
    }
    freeze_sha = write_json(output / "fold-freezes.json", freezes, quota)
    rows = []
    for freeze in folds:
        rows.extend(
            evaluate_fold(
                cache,
                np.flatnonzero(np.arange(len(cache["subject_ids"])) % 3 == freeze["fold_id"]),
                freeze,
                plan,
            )
        )
    rows.sort(key=lambda r: (r["subject_id"], r["cell_id"], r["budget"], r["method"]))
    if len(rows) != plan["reporting"]["expected_rows"]:
        raise RuntimeError("Frozen metric row count differs.")
    summary = summarize_results(rows, plan)
    result = {
        "schema": "cfeg.spatial-calibration-source.result.v1",
        **provenance,
        "evidence_role": plan["evidence_role"],
        "fold_freezes_sha256": freeze_sha,
        "rows": rows,
        "summary": summary,
    }
    result_sha = write_json(output / "result.json", result, quota)
    return {
        "status": summary["status"],
        "rows": len(rows),
        "result_sha256": result_sha,
        "human_held_unlock": False,
    }


def run_study(plan_path: Path, output: Path, workers: int) -> dict:
    root = Path(__file__).resolve().parents[3]
    if (
        not plan_path.is_absolute()
        or not output.is_absolute()
        or plan_path.resolve() != root / "configs/analysis/spatial_calibration_source39_v1.json"
        or file_sha(plan_path) != PLAN_SHA256
    ):
        raise ValueError("Exact absolute frozen spatial-source plan required.")
    plan = json.loads(plan_path.read_text())
    if tuple(plan["source_subject_ids"]) != SOURCE_IDS or Path(plan["raw_root"]) != RAW_ROOT:
        raise ValueError("Immutable exposed source39 boundary differs.")
    if (
        output.resolve() != output
        or str(output) != plan["execution"]["output_root"]
        or workers != plan["execution"]["workers"]
    ):
        raise ValueError("Exact output and worker contract required.")
    if (output / "start.json").exists():
        raise FileExistsError("Existing start consumes the attempt; no retry or resume.")

    def git(*args):
        return subprocess.check_output(["git", "-C", str(root), *args], text=True).strip()

    if git("status", "--porcelain", "--untracked-files=all"):
        raise RuntimeError("Clean committed source required before human input access.")
    for filename, expected in plan["pinned_files"].items():
        if file_sha(root / filename) != expected:
            raise ValueError(f"Pinned source changed: {filename}")
    thread_keys = ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS")
    if any(os.environ.get(key) != "1" for key in thread_keys):
        raise RuntimeError("All BLAS thread variables must be 1 before numerical imports.")
    filterbank = yaml.safe_load((root / plan["filterbank_config"]).read_text())
    output.mkdir(parents=True, exist_ok=True)
    if any(output.iterdir()):
        raise FileExistsError("Study output must be a new empty directory.")
    start = {
        "schema": "cfeg.spatial-calibration-source.start.v1",
        "study_id": plan["study_id"],
        "source_commit": git("rev-parse", "HEAD"),
        "source_tree": git("rev-parse", "HEAD^{tree}"),
        "plan_sha256": PLAN_SHA256,
        "pinned_files": plan["pinned_files"],
        "source_subject_ids": list(SOURCE_IDS),
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "pid": os.getpid(),
        "workers": workers,
        "human_held_access": False,
        "metadata_access": False,
        "runtime": {
            "python": sys.version,
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "platform": platform.platform(),
            "blas_threads": {key: os.environ[key] for key in thread_keys},
        },
    }
    start_sha = write_json(output / "start.json", start, plan["execution"]["resource_budget_bytes"])
    return execute_phases(plan, filterbank, output, start, start_sha, workers)
