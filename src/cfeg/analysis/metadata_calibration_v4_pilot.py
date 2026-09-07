"""Small, independent V4 waveform pilot; no V3 authority or data dependencies.

Only ``run_pilot`` opens the study RNG namespace, after exclusive start publication.
Pure numerical functions accept explicit arrays; fixture generators require explicit seeds.
"""

from __future__ import annotations

import hashlib
import itertools
import json
import math
import os
import platform
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import scipy
import yaml
from scipy import signal, stats

from cfeg.baselines.calibration import predict_supervised_template_correlation
from cfeg.baselines.fbcca import apply_filterbank, make_reference_signals

PLAN_SHA256 = "50607e9b8abe7058565b45cdb8a98dcd1a78f516c393af911f0e5f3ab0484124"
FILTERBANK_SHA256 = "b8c1ce4477d980b40f359bc3dc97f3125313191d24401f29a7e2dc473d0d1380"
COMPONENTS = {
    name: index + 1
    for index, name in enumerate(
        (
            "common_topography",
            "class_topography",
            "class_phase",
            "trial_jitter",
            "channel_signs",
            "block_states",
            "innovation",
            "measurement_noise",
            "null_states",
            "null_measurement_noise",
            "trial_order",
        )
    )
}
METHODS = ("A0", "AQ", "AQM_block", "AQM_scalar", "missing", "pooled")
_WORKSPACE: dict[str, Any] | None = None


def component_rng(root_seed: int, participant: int, component: str) -> np.random.Generator:
    if root_seed < 0 or participant < 0 or component not in COMPONENTS:
        raise ValueError("Invalid independent pilot RNG coordinate.")
    return np.random.Generator(
        np.random.PCG64DXSM(np.random.SeedSequence([root_seed, participant, COMPONENTS[component]]))
    )


def generate_participant(plan: dict, participant: int, *, root_seed: int) -> dict:
    """Explicit-seed array generator; callers must not use study seeds in fixtures."""
    dgp = plan["dgp"]
    classes, channels = plan["n_classes"], plan["n_channels"]
    blocks = plan["support_blocks"] + plan["query_blocks"]
    samples, harmonics = plan["n_samples"], dgp["harmonics"]
    rng = lambda component: component_rng(root_seed, participant, component)
    common = rng("common_topography").uniform(*dgp["common_topography_uniform"], channels)
    spatial = common + rng("class_topography").normal(
        0, dgp["class_topography_perturbation_sd"], (classes, channels)
    )
    spatial /= np.linalg.norm(spatial, axis=1, keepdims=True)
    phase = rng("class_phase").uniform(0, 2 * np.pi, (classes, harmonics))
    jitter = rng("trial_jitter").normal(
        0, dgp["trial_harmonic_phase_jitter_sd"], (blocks, classes, harmonics)
    )
    frequency = dgp["frequency_start"] + dgp["frequency_step"] * np.arange(classes)
    time = np.arange(samples, dtype=np.float64) / plan["sfreq"]
    h = np.arange(1, harmonics + 1)
    angle = (
        2
        * np.pi
        * frequency[None, :, None, None]
        * h[None, None, :, None]
        * time[None, None, None, :]
        + phase[None, :, :, None]
        + jitter[..., None]
    )
    waveform = np.sum(np.sin(angle) / h[None, None, :, None], axis=2)
    clean = spatial[None, :, :, None] * waveform[:, :, None, :]
    innovations = rng("innovation").standard_normal((blocks, classes, channels, samples))
    innovations *= dgp["ar_innovation_sd"]
    innovations[..., 0] /= math.sqrt(1 - dgp["ar_rho"] ** 2)
    noise = signal.lfilter([1.0], [1.0, -dgp["ar_rho"]], innovations, axis=-1)
    signs = rng("channel_signs").permutation(np.repeat([-1.0, 1.0], channels // 2))
    state = rng("block_states").choice([-1.0, 1.0], blocks)
    null_state = rng("null_states").choice([-1.0, 1.0], blocks)
    state_channel = state[:, None] * signs[None, :]
    drift = (
        clean * np.exp(dgp["signal_log_gain"] * state_channel)[:, None, :, None]
        + noise * np.exp(dgp["noise_log_gain"] * state_channel)[:, None, :, None]
    )
    order_rng = rng("trial_order")
    orders = np.stack([order_rng.permutation(classes) for _ in range(blocks)])
    stable = np.take_along_axis(clean + noise, orders[..., None, None], axis=1)
    drift = np.take_along_axis(drift, orders[..., None, None], axis=1)
    measurement_noise = rng("measurement_noise").normal(
        0, dgp["metadata_log_noise_sd"], (blocks, channels)
    )
    null_noise = rng("null_measurement_noise").normal(
        0, dgp["metadata_log_noise_sd"], (blocks, channels)
    )
    log_center = math.log(50.0)
    return {
        "labels": orders.astype(np.int64),
        "eeg": {"stable": stable, "acquisition_drift": drift, "metadata_null": drift},
        "metadata": {
            "stable": np.exp(log_center + measurement_noise),
            "acquisition_drift": np.exp(
                log_center + dgp["metadata_log_state_gain"] * state_channel + measurement_noise
            ),
            "metadata_null": np.exp(
                log_center
                + dgp["metadata_log_state_gain"] * null_state[:, None] * signs[None, :]
                + null_noise
            ),
        },
    }


def _inverse_root(matrix: np.ndarray) -> np.ndarray:
    eigenvalues, eigenvectors = np.linalg.eigh(matrix)
    tolerance = max(np.finfo(np.float64).eps, max(float(eigenvalues[-1]), 0) * 1e-12)
    inverse = np.zeros_like(eigenvalues)
    keep = eigenvalues > tolerance
    inverse[keep] = 1 / np.sqrt(eigenvalues[keep])
    return (eigenvectors * inverse) @ eigenvectors.T


def reference_cache(
    frequencies: np.ndarray, sfreq: float, samples: int, harmonics: int, regularization: float
) -> tuple[np.ndarray, np.ndarray]:
    references = make_reference_signals(frequencies, sfreq, samples, harmonics).astype(np.float64)
    references -= references.mean(axis=-1, keepdims=True)
    whitenings = []
    for reference in references:
        covariance = reference @ reference.T / (samples - 1)
        scale = float(np.trace(covariance)) / len(covariance)
        whitenings.append(
            _inverse_root(covariance + regularization * scale * np.eye(len(covariance)))
        )
    return references, np.stack(whitenings)


def cached_cca_scores(
    eeg: np.ndarray, cache: tuple[np.ndarray, np.ndarray], regularization: float
) -> np.ndarray:
    """Public CCA algebra, caching only reference and per-trial whitening."""
    values = np.asarray(eeg, dtype=np.float64)
    if values.ndim != 3 or not np.isfinite(values).all():
        raise ValueError("EEG must be finite [trial,channel,time].")
    references, reference_whitening = cache
    if values.shape[-1] != references.shape[-1]:
        raise ValueError("Reference and EEG samples differ.")
    output = np.zeros((len(values), len(references)))
    for index, trial in enumerate(values):
        centered = trial - trial.mean(axis=-1, keepdims=True)
        if np.linalg.norm(centered) <= np.finfo(np.float64).eps * math.sqrt(trial.shape[-1]) * 10:
            continue
        denominator = trial.shape[-1] - 1
        covariance = centered @ centered.T / denominator
        scale = float(np.trace(covariance)) / len(covariance)
        if scale <= 0:
            continue
        whitening = _inverse_root(covariance + regularization * scale * np.eye(len(covariance)))
        cross = centered[None] @ references.transpose(0, 2, 1) / denominator
        whitened = whitening[None] @ cross @ reference_whitening
        output[index] = np.clip(np.linalg.svd(whitened, compute_uv=False)[:, 0], 0, 1)
    return output


def projection_basis(rows: np.ndarray, cutoff: float) -> np.ndarray:
    values = np.asarray(rows, dtype=np.float64)
    values = values - values.mean(axis=-1, keepdims=True)
    _, singular, right = np.linalg.svd(values, full_matrices=False)
    keep = singular > (singular[0] * cutoff if len(singular) else 0)
    return right[keep]


def eeg_quality(eeg: np.ndarray, all_class_basis: np.ndarray, variance_floor: float) -> np.ndarray:
    """Current trial only; no labels, context, or other query trials."""
    eeg, all_class_basis = np.asarray(eeg), np.asarray(all_class_basis)
    if (
        eeg.ndim != 3
        or all_class_basis.ndim != 2
        or eeg.shape[-1] != all_class_basis.shape[-1]
        or not np.isfinite(eeg).all()
        or not np.isfinite(all_class_basis).all()
        or not math.isfinite(variance_floor)
        or variance_floor <= 0
    ):
        raise ValueError("Quality requires finite aligned EEG/basis and a positive variance floor.")
    centered = eeg - eeg.mean(axis=-1, keepdims=True)
    residual = centered - (centered @ all_class_basis.T) @ all_class_basis
    return np.log(np.maximum(np.mean(residual**2, axis=-1), variance_floor))


def block_reliability(
    support: np.ndarray,
    labels: np.ndarray,
    class_bases: tuple[np.ndarray, ...],
    variance_floor: float,
) -> float:
    ratios = []
    for trial, label in zip(support, labels):
        centered = trial - trial.mean(axis=-1, keepdims=True)
        power = float(np.sum((centered @ class_bases[int(label)].T) ** 2))
        total = float(np.sum(centered**2))
        ratios.append(np.clip(power / max(total, variance_floor), 0, 1))
    return float(np.mean(ratios))


def eeg_block_weights(
    query_q: np.ndarray,
    support_q: np.ndarray,
    reliability: np.ndarray,
    bandwidth: float,
    floor: float,
) -> np.ndarray:
    log_weights = np.log(floor + reliability)[None, :] - np.mean(
        (query_q[:, None] - support_q[None]) ** 2, axis=-1
    ) / (2 * bandwidth**2)
    log_weights -= log_weights.max(axis=1, keepdims=True)
    weights = np.exp(log_weights)
    return weights / weights.sum(axis=1, keepdims=True)


def metadata_affinity(
    query_metadata: np.ndarray, support_metadata: np.ndarray, bandwidth: float
) -> np.ndarray:
    if (
        not np.isfinite(query_metadata).all()
        or not np.isfinite(support_metadata).all()
        or np.any(query_metadata <= 0)
        or np.any(support_metadata <= 0)
    ):
        raise ValueError("Observed metadata must be finite and positive.")
    difference = np.log(query_metadata)[:, None] - np.log(support_metadata)[None]
    return np.exp(-np.mean(difference**2, axis=-1) / (2 * bandwidth**2))


def softmax(scores: np.ndarray, temperature: float) -> np.ndarray:
    values = scores / temperature
    values = values - values.max(axis=-1, keepdims=True)
    exp = np.exp(values)
    return exp / exp.sum(axis=-1, keepdims=True)


def fuse_posteriors(
    p0: np.ndarray,
    block_posteriors: np.ndarray,
    weights: np.ndarray,
    *,
    strength: float,
    affinities: np.ndarray | None = None,
) -> dict[str, np.ndarray]:
    """EEG posteriors and precomputed affinity only; no labels or raw metadata."""
    if (
        p0.ndim != 2
        or block_posteriors.ndim != 3
        or weights.ndim != 2
        or block_posteriors.shape != (*weights.shape, p0.shape[1])
        or weights.shape[0] != p0.shape[0]
        or not math.isfinite(strength)
        or not 0 <= strength <= 1
    ):
        raise ValueError("Fusion shapes or strength are invalid.")
    for probability in (p0, block_posteriors):
        if (
            not np.isfinite(probability).all()
            or np.any(probability < 0)
            or not np.allclose(probability.sum(axis=-1), 1, rtol=0, atol=1e-12)
        ):
            raise ValueError("Fusion inputs must be probability vectors.")
    if (
        not np.isfinite(weights).all()
        or np.any(weights < 0)
        or (weights.shape[1] and not np.allclose(weights.sum(axis=1), 1, rtol=0, atol=1e-12))
    ):
        raise ValueError("EEG block weights must be finite normalized nonnegative values.")
    if affinities is not None and (
        affinities.shape != weights.shape
        or not np.isfinite(affinities).all()
        or np.any(affinities < 0)
        or np.any(affinities > 1)
    ):
        raise ValueError("Affinities must be aligned finite values in [0,1].")
    if block_posteriors.shape[1] == 0:
        return {name: p0.copy() for name in ("AQ", "AQM_block", "AQM_scalar", "missing")}
    residuals = block_posteriors - p0[:, None]
    update = np.sum(weights[..., None] * residuals, axis=1)
    aq = p0 + strength * update
    if affinities is None or np.all(affinities == 1):
        block = scalar = aq
    else:
        g = np.sum(weights * affinities, axis=1)
        scalar = p0 + strength * g[:, None] * update
        if block_posteriors.shape[1] == 1 or np.all(affinities == affinities[:, :1]):
            block = scalar
        else:
            block = p0 + strength * np.sum((weights * affinities)[..., None] * residuals, axis=1)
    return {"AQ": aq, "AQM_block": block, "AQM_scalar": scalar, "missing": aq.copy()}


def posterior_metrics(
    probability: np.ndarray, labels: np.ndarray, a0: np.ndarray, aq: np.ndarray
) -> dict:
    prediction = probability.argmax(axis=1)
    classes = probability.shape[1]
    correct = prediction == labels
    counts = np.bincount(labels, minlength=classes)
    correct_counts = np.bincount(labels, weights=correct, minlength=classes)
    truth = probability[np.arange(len(labels)), labels]
    other = probability.copy()
    other[np.arange(len(labels)), labels] = -np.inf
    return {
        "balanced_accuracy": float(np.mean(correct_counts / counts)),
        "correct_log_probability": float(np.mean(np.log(np.maximum(truth, 1e-300)))),
        "true_class_margin": float(np.mean(truth - other.max(axis=1))),
        "prediction_flips_vs_A0": float(np.mean(prediction != a0.argmax(axis=1))),
        "prediction_flips_vs_AQ": float(np.mean(prediction != aq.argmax(axis=1))),
        "correct_count": float(correct.sum()),
        "query_count": len(labels),
        "class_correct_counts": correct_counts.tolist(),
        "class_query_counts": counts.tolist(),
    }


def affinity_metrics(weights: np.ndarray, affinity: np.ndarray) -> dict:
    if weights.shape[1] == 0:
        return {"metadata_total_trust": 0.0, "weight_turnover": 0.0, "affinity_variation": 0.0}
    weighted = weights * affinity
    g = weighted.sum(axis=1)
    normalized = np.divide(weighted, g[:, None], out=weights.copy(), where=g[:, None] > 0)
    return {
        "metadata_total_trust": float(g.mean()),
        "weight_turnover": float((0.5 * np.abs(normalized - weights).sum(axis=1)).mean()),
        "affinity_variation": float(affinity.std(axis=1).mean()),
    }


def applied_affinity_metrics(method: str, weights: np.ndarray, affinity: np.ndarray) -> dict:
    if method == "A0":
        return {"metadata_total_trust": 0.0, "weight_turnover": 0.0, "affinity_variation": 0.0}
    metric = affinity_metrics(
        weights, affinity if method.startswith("AQM") else np.ones_like(affinity)
    )
    if method in ("AQM_scalar", "AQM_scalar_deranged", "pooled"):
        metric["weight_turnover"] = 0.0
    return metric


def derangements(depth: int) -> tuple[tuple[int, ...], ...]:
    return tuple(
        p
        for p in itertools.permutations(range(depth))
        if all(i != value for i, value in enumerate(p))
    )


def mean_metric_records(records: list[dict]) -> dict:
    """Average each completed permutation's metrics, never its posteriors."""
    return {
        key: (
            np.mean([record[key] for record in records], axis=0).tolist()
            if isinstance(records[0][key], list)
            else float(np.mean([record[key] for record in records]))
        )
        for key in records[0]
    }


def initialize_workspace(plan: dict, filterbank: dict) -> None:
    global _WORKSPACE
    dgp = plan["dgp"]
    frequencies = dgp["frequency_start"] + dgp["frequency_step"] * np.arange(plan["n_classes"])
    references = make_reference_signals(
        frequencies, plan["sfreq"], plan["n_samples"], dgp["harmonics"]
    ).astype(np.float64)
    cutoff = plan["operator"]["projection_svd_relative_cutoff"]
    _WORKSPACE = {
        "plan": plan,
        "filterbank": filterbank,
        "cca_cache": reference_cache(
            frequencies,
            plan["sfreq"],
            plan["n_samples"],
            filterbank["n_harmonics"],
            filterbank["regularization"],
        ),
        "basis": projection_basis(references.reshape(-1, plan["n_samples"]), cutoff),
        "class_bases": tuple(projection_basis(r, cutoff) for r in references),
    }


def _waveform_scores(
    query: np.ndarray, support: np.ndarray, labels: np.ndarray, weights: np.ndarray, classes: int
) -> np.ndarray:
    subband = np.stack(
        [
            predict_supervised_template_correlation(
                query[f], support[f], labels, n_classes=classes
            ).scores
            for f in range(len(weights))
        ]
    )
    return np.einsum("f,fqc->qc", weights, np.sign(subband) * subband**2) / weights.sum()


def _decode_eeg(eeg: np.ndarray, labels: np.ndarray, workspace: dict) -> dict:
    plan, op = workspace["plan"], workspace["plan"]["operator"]
    blocks, classes, channels, samples = eeg.shape
    support_blocks = plan["support_blocks"]
    centered = eeg - eeg.mean(axis=-1, keepdims=True)
    flat = centered.reshape(-1, channels, samples)
    filtered, parameters = apply_filterbank(
        flat, sfreq=plan["sfreq"], filterbank=workspace["filterbank"]
    )
    filtered = filtered.reshape(len(parameters["weights"]), blocks, classes, channels, samples)
    weights = np.asarray(parameters["weights"])
    query = filtered[:, support_blocks:].reshape(len(weights), -1, channels, samples)
    # A0 receives the original trial, matching public predict_fbcca including filtering.
    original_query, _ = apply_filterbank(
        eeg[support_blocks:].reshape(-1, channels, samples),
        sfreq=plan["sfreq"],
        filterbank=workspace["filterbank"],
    )
    cca = np.stack(
        [
            cached_cca_scores(x, workspace["cca_cache"], parameters["regularization"])
            for x in original_query
        ]
    )
    a0_scores = np.einsum("f,fqc->qc", weights, cca**2) / weights.sum()
    p0 = softmax(a0_scores, op["probability_temperature"])
    block_scores = np.stack(
        [
            _waveform_scores(query, filtered[:, b], labels[b], weights, classes)
            for b in range(support_blocks)
        ],
        axis=1,
    )
    pb = softmax(block_scores, op["probability_temperature"])
    quality = eeg_quality(flat, workspace["basis"], op["variance_floor"])
    quality = quality.reshape(blocks, classes, channels)
    reliability = np.asarray(
        [
            block_reliability(eeg[b], labels[b], workspace["class_bases"], op["variance_floor"])
            for b in range(support_blocks)
        ]
    )
    return {
        "p0": p0,
        "pb": pb,
        "support_q": quality[:support_blocks].mean(axis=1),
        "query_q": quality[support_blocks:].reshape(-1, channels),
        "reliability": reliability,
        "filtered": filtered,
        "filtered_query": query,
        "filter_weights": weights,
    }


def evaluate_participant(participant: int, root_seed: int) -> dict:
    if _WORKSPACE is None:
        raise RuntimeError("Numerical workspace must be initialized.")
    workspace = _WORKSPACE
    plan, op = workspace["plan"], workspace["plan"]["operator"]
    data = generate_participant(plan, participant, root_seed=root_seed)
    labels, support_blocks = data["labels"], plan["support_blocks"]
    query_labels = labels[support_blocks:].reshape(-1)
    decoded = {
        family: _decode_eeg(data["eeg"][family], labels, workspace)
        for family in ("stable", "acquisition_drift")
    }
    decoded["metadata_null"] = decoded["acquisition_drift"]
    rows, control_counts, prediction_evidence, derangement_evidence = [], {}, [], []
    for family in plan["families"]:
        current = decoded[family]
        p0, pb = current["p0"], current["pb"]
        metadata = data["metadata"][family]
        query_m = np.repeat(metadata[support_blocks:], plan["n_classes"], axis=0)
        for budget in plan["budgets"]:
            if budget:
                pi = eeg_block_weights(
                    current["query_q"],
                    current["support_q"][:budget],
                    current["reliability"][:budget],
                    op["q_affinity_bandwidth"],
                    op["support_reliability_floor"],
                )
                a = metadata_affinity(query_m, metadata[:budget], op["m_affinity_bandwidth"])
                f = current["filtered"]
                pooled_score = _waveform_scores(
                    current["filtered_query"],
                    f[:, :budget].reshape(len(f), -1, *f.shape[-2:]),
                    labels[:budget].reshape(-1),
                    current["filter_weights"],
                    plan["n_classes"],
                )
                pooled = softmax(pooled_score, op["probability_temperature"])
            else:
                pi = a = np.empty((len(p0), 0))
                pooled = p0.copy()
            predictions = {
                "A0": p0,
                **fuse_posteriors(p0, pb[:, :budget], pi, strength=op["lambda"], affinities=a),
                "pooled": pooled,
            }
            for method in METHODS:
                row = posterior_metrics(predictions[method], query_labels, p0, predictions["AQ"])
                row.update(applied_affinity_metrics(method, pi, a))
                rows.append(
                    {
                        "participant": participant,
                        "family": family,
                        "method": method,
                        "budget": budget,
                        "permutations": 1,
                        **row,
                    }
                )
                posterior = predictions[method]
                true_probability = posterior[np.arange(len(query_labels)), query_labels]
                other = posterior.copy()
                other[np.arange(len(query_labels)), query_labels] = -np.inf
                prediction_evidence.append(
                    {
                        "family": family,
                        "method": method,
                        "budget": budget,
                        "predictions": posterior.argmax(axis=1).tolist(),
                        "posterior_sha256": hashlib.sha256(
                            np.ascontiguousarray(posterior).tobytes()
                        ).hexdigest(),
                        "true_class_probability": true_probability.tolist(),
                        "true_class_margin": (true_probability - other.max(axis=1)).tolist(),
                    }
                )
            if budget in (3, 5):
                controls = {"AQM_block_deranged": [], "AQM_scalar_deranged": []}
                permutations = derangements(budget)
                control_counts[str(budget)] = len(permutations)
                for permutation in permutations:
                    shuffled_a = a[:, permutation]
                    shuffled = fuse_posteriors(
                        p0, pb[:, :budget], pi, strength=op["lambda"], affinities=shuffled_a
                    )
                    for control, control_metrics in controls.items():
                        method = control.removesuffix("_deranged")
                        metric = posterior_metrics(
                            shuffled[method], query_labels, p0, predictions["AQ"]
                        )
                        metric.update(applied_affinity_metrics(method, pi, shuffled_a))
                        control_metrics.append(metric)
                for method, metrics in controls.items():
                    rows.append(
                        {
                            "participant": participant,
                            "family": family,
                            "method": method,
                            "budget": budget,
                            "permutations": len(permutations),
                            **mean_metric_records(metrics),
                        }
                    )
                    derangement_evidence.append(
                        {
                            "family": family,
                            "method": method,
                            "budget": budget,
                            "permutations": [list(p) for p in permutations],
                            "balanced_accuracy": [m["balanced_accuracy"] for m in metrics],
                        }
                    )
    hashes = {
        family: hashlib.sha256(np.ascontiguousarray(eeg).tobytes()).hexdigest()
        for family, eeg in data["eeg"].items()
    }
    return {
        "participant": participant,
        "rows": rows,
        "eeg_sha256": hashes,
        "query_labels": query_labels.tolist(),
        "prediction_evidence": prediction_evidence,
        "derangement_evidence": derangement_evidence,
        "label_order_sha256": hashlib.sha256(labels.tobytes()).hexdigest(),
        "support_indices": list(range(support_blocks)),
        "query_indices": list(range(support_blocks, support_blocks + plan["query_blocks"])),
        "derangement_counts": control_counts,
    }


def paired_summary(values: list[float] | np.ndarray) -> dict:
    values = np.asarray(values, dtype=np.float64)
    mean = float(values.mean())
    standard_error = float(values.std(ddof=1) / math.sqrt(len(values)))
    t95, t975 = stats.t.ppf([0.95, 0.975], len(values) - 1)
    return {
        "n": len(values),
        "mean": mean,
        "standard_error": standard_error,
        "one_sided_95_LCB": float(mean - t95 * standard_error),
        "two_sided_95_CI": [
            float(mean - t975 * standard_error),
            float(mean + t975 * standard_error),
        ],
        "two_sided_90_CI": [float(mean - t95 * standard_error), float(mean + t95 * standard_error)],
        "participant_values": values.tolist(),
    }


def summarize_results(participants: list[dict], plan: dict) -> dict:
    rows = [row for participant in participants for row in participant["rows"]]
    lookup = {(r["participant"], r["family"], r["method"], r["budget"]): r for r in rows}
    n = len(participants)

    def values(family: str, method: str, budget: int | str) -> np.ndarray:
        if budget == "eauc":
            return sum(
                weight * values(family, method, k)
                for k, weight in ((0, 1 / 6), (1, 1 / 2), (3, 1 / 3))
            )
        return np.asarray(
            [lookup[p, family, method, budget]["balanced_accuracy"] for p in range(n)]
        )

    comparisons = {}
    for name, family, first, second, budget in (
        ("stable_AQ_minus_A0_eauc", "stable", "AQ", "A0", "eauc"),
        ("stable_AQ_minus_A0_k1", "stable", "AQ", "A0", 1),
        ("drift_block_minus_AQ_eauc", "acquisition_drift", "AQM_block", "AQ", "eauc"),
        (
            "drift_correct_minus_deranged_k3",
            "acquisition_drift",
            "AQM_block",
            "AQM_block_deranged",
            3,
        ),
        ("drift_block_minus_scalar_k3", "acquisition_drift", "AQM_block", "AQM_scalar", 3),
        ("null_block_minus_AQ_k3", "metadata_null", "AQM_block", "AQ", 3),
        ("drift_block_minus_A0_k3", "acquisition_drift", "AQM_block", "A0", 3),
    ):
        comparisons[name] = paired_summary(
            values(family, first, budget) - values(family, second, budget)
        )
    threshold = plan["reporting"]["effect_threshold"]
    passes = lambda r: r["mean"] >= threshold and r["one_sided_95_LCB"] > 0
    aq_ok = (
        passes(comparisons["stable_AQ_minus_A0_eauc"])
        and comparisons["stable_AQ_minus_A0_k1"]["mean"] >= 0
    )
    scalar = comparisons["drift_block_minus_scalar_k3"]
    null_ci = comparisons["null_block_minus_AQ_k3"]["two_sided_90_CI"]
    margin = plan["reporting"]["null_equivalence_margin"]
    components = {
        "AQ_stable_utility": aq_ok,
        "metadata_eauc": passes(comparisons["drift_block_minus_AQ_eauc"]),
        "metadata_pairing": passes(comparisons["drift_correct_minus_deranged_k3"]),
        "block_over_scalar": scalar["mean"] > 0 and scalar["one_sided_95_LCB"] > 0,
        "null_equivalence": null_ci[0] >= -margin and null_ci[1] <= margin,
        "drift_nonnegative_vs_A0": comparisons["drift_block_minus_A0_k3"]["mean"] >= 0,
    }
    status = (
        "AQ_NOT_ESTABLISHED"
        if not aq_ok
        else "METADATA_NOT_ESTABLISHED"
        if not all(components.values())
        else "READY_FOR_INDEPENDENT_SYNTHETIC_REPLICATION"
    )
    aggregate, attainment, exploratory = [], [], []
    for family in plan["families"]:
        for method in (*METHODS, "AQM_block_deranged", "AQM_scalar_deranged"):
            for budget in (3, 5) if method.endswith("_deranged") else plan["budgets"]:
                records = [lookup[p, family, method, budget] for p in range(n)]
                aggregate.append(
                    {
                        "family": family,
                        "method": method,
                        "budget": budget,
                        "mean_metrics": mean_metric_records(
                            [
                                {
                                    k: r[k]
                                    for k in r
                                    if k
                                    not in {
                                        "participant",
                                        "family",
                                        "method",
                                        "budget",
                                        "permutations",
                                    }
                                }
                                for r in records
                            ]
                        ),
                        "fraction_harmed_vs_A0": float(
                            np.mean(values(family, method, budget) < values(family, "A0", budget))
                        ),
                    }
                )
            if method.endswith("_deranged"):
                continue
            curve = {str(k): values(family, method, k).tolist() for k in plan["budgets"]}
            attainment.append(
                {
                    "family": family,
                    "method": method,
                    "participant_budgets": [
                        next(
                            (
                                k
                                for k in plan["budgets"]
                                if curve[str(k)][p] >= plan["reporting"]["calibration_target"]
                            ),
                            ">5",
                        )
                        for p in range(n)
                    ],
                    "participant_accuracy_curve": curve,
                    "participant_eauc": values(family, method, "eauc").tolist(),
                }
            )
            exploratory.append(
                {
                    "family": family,
                    "method": method,
                    "contrast": "k3_minus_k1",
                    **paired_summary(values(family, method, 3) - values(family, method, 1)),
                }
            )
        exploratory.append(
            {
                "family": family,
                "contrast": "AQM_block_k1_minus_AQ_k3",
                **paired_summary(values(family, "AQM_block", 1) - values(family, "AQ", 3)),
            }
        )
    return {
        "status": status,
        "human_unlock": False,
        "screen_components": components,
        "comparisons": comparisons,
        "aggregate_rows": aggregate,
        "calibration_attainment": attainment,
        "exploratory_budget_contrasts": exploratory,
    }


def _file_sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json_exclusive(path: Path, payload: dict) -> str:
    content = (json.dumps(payload, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o400)
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(content)
        handle.flush()
        os.fsync(handle.fileno())
    parent = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(parent)
    finally:
        os.close(parent)
    return hashlib.sha256(content).hexdigest()


def run_pilot(plan_path: Path, output: Path, workers: int) -> dict:
    """Consume exactly one pilot attempt. An existing start always refuses rerun."""
    if not plan_path.is_absolute() or not output.is_absolute():
        raise ValueError("--plan and --output must be absolute paths.")
    plan_path, output = plan_path.resolve(), output.resolve()
    root = Path(__file__).resolve().parents[3]
    expected_plan = root / "configs/analysis/metadata_calibration_v4_pilot.json"
    if plan_path != expected_plan or _file_sha(plan_path) != PLAN_SHA256:
        raise ValueError("The exact committed V4 pilot plan is required.")
    plan = json.loads(plan_path.read_text())
    if (
        str(output) != plan["execution"]["new_output_root"]
        or workers != plan["execution"]["workers"]
    ):
        raise ValueError("Output root and worker count must match the fixed plan.")
    if (output / "start.json").exists():
        raise FileExistsError("Existing start consumes this attempt; rerun/resume is forbidden.")

    def git(*args: str) -> str:
        return subprocess.check_output(["git", "-C", str(root), *args], text=True).strip()

    if git("status", "--porcelain", "--untracked-files=all"):
        raise RuntimeError("V4 pilot requires clean committed source.")
    for name in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS"):
        if os.environ.get(name) != "1":
            raise RuntimeError(f"{name}=1 is required before NumPy import.")
    filterbank_path = root / plan["filterbank_config"]
    if _file_sha(filterbank_path) != FILTERBANK_SHA256:
        raise ValueError("Frozen shared filterbank hash differs.")
    filterbank = yaml.safe_load(filterbank_path.read_text())
    output.mkdir(parents=True, exist_ok=True)
    if any(output.iterdir()):
        raise FileExistsError("New output root must be empty.")
    start = {
        "schema": "cfeg.metadata-calibration-efficiency-v4.pilot-start.v1",
        "candidate_id": plan["candidate_id"],
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "pid": os.getpid(),
        "source_commit": git("rev-parse", "HEAD"),
        "source_tree": git("rev-parse", "HEAD^{tree}"),
        "plan_sha256": PLAN_SHA256,
        "filterbank_sha256": FILTERBANK_SHA256,
        "seed_namespace": plan["seed_namespace"],
        "rng_components": COMPONENTS,
        "workers": workers,
        "runtime": {
            "python": sys.version,
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "platform": platform.platform(),
            "machine": platform.machine(),
            "blas_threads": {
                key: os.environ[key]
                for key in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS")
            },
        },
        "human_data_access": False,
        "evidence_role": plan["evidence_role"],
    }
    start_sha = write_json_exclusive(output / "start.json", start)
    # This is the first entry to the study RNG namespace; exclusive start is durable.
    root_seed = int.from_bytes(hashlib.sha256(plan["seed_namespace"].encode()).digest()[:8], "big")
    results = []
    with ProcessPoolExecutor(
        max_workers=workers, initializer=initialize_workspace, initargs=(plan, filterbank)
    ) as executor:
        futures = {
            executor.submit(evaluate_participant, p, root_seed): p
            for p in range(plan["participants"])
        }
        for future in as_completed(futures):
            result = future.result()
            results.append(result)
            print(
                json.dumps(
                    {
                        "participant_completed": result["participant"],
                        "completed": len(results),
                        "total": plan["participants"],
                    }
                ),
                flush=True,
            )
    results.sort(key=lambda r: r["participant"])
    summary = summarize_results(results, plan)
    result = {
        "schema": "cfeg.metadata-calibration-efficiency-v4.pilot-result.v1",
        "candidate_id": plan["candidate_id"],
        "evidence_role": plan["evidence_role"],
        "start_file_sha256": start_sha,
        "root_seed": root_seed,
        "source_commit": start["source_commit"],
        "plan_sha256": PLAN_SHA256,
        "filterbank_sha256": FILTERBANK_SHA256,
        "participant_results": results,
        "summary": summary,
    }
    result_sha = write_json_exclusive(output / "result.json", result)
    return {
        "status": summary["status"],
        "participants": len(results),
        "metric_rows": sum(len(r["rows"]) for r in results),
        "result_sha256": result_sha,
        "human_unlock": False,
    }
