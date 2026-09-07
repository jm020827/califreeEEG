"""One bounded cross-fitted development study on the already exposed source39.

Pure functions operate on explicit arrays. Only the one-time runner opens human
files, after a durable start and an immutable participant allowlist check.
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
import pandas as pd
import scipy
import yaml

from cfeg.analysis import metadata_calibration_v4_pilot as numeric
from cfeg.baselines.calibration import predict_filterbank_ensemble_trca
from cfeg.baselines.fbcca import apply_filterbank, make_reference_signals
from cfeg.data.prepare_mat import _load_arrays
from cfeg.data.prepare_wearable import _wearable_data

PLAN_SHA256 = "a76f1df0e0d2989f6217013e610d074e50d9efb52b0cf636bbcdf38a905898d3"
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
MANIFEST_PATH = Path("/home/whwovy/eeg-data/processed/wearable_v3/manifest.parquet")
MANIFEST_COLUMNS = (
    "subject_id",
    "electrode_type",
    "run_id",
    "impedance_kohm_by_channel",
    "headband_order",
    "condition_period",
)
FEATURE_SCHEMA = "cfeg.context-template-source.features.v1"
BUDGETS = (1, 3, 5)
FEATURE_KEYS = (
    "gram_cross",
    "gram_support",
    "query_norm2",
    "a0_scores",
    "etrca_scores",
    "q_full_support",
    "q_full_query",
    "q_diag_support",
    "q_diag_query",
    "reliability",
)
_WORKER = None


def cells(plan: dict) -> list[dict]:
    return [
        {
            "cell_id": f"{interface}-n{n}",
            "interface": interface,
            "n_samples": n,
            "interface_index": e,
        }
        for n in plan["sample_counts"]
        for e, interface in enumerate(plan["interfaces"])
    ]


def file_sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def array_sha(value: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def validate_impedance(values: np.ndarray) -> np.ndarray:
    result = np.asarray(values, dtype=np.float64)
    if np.isinf(result).any() or np.any(result[np.isfinite(result)] < 0):
        raise ValueError("Impedance must be nonnegative finite values or NaN for missing.")
    return result


def metadata_projection(frame: pd.DataFrame, plan: dict) -> dict:
    """Validate an already predicate-filtered, column-projected source table."""
    expected = {f"sub{s:03d}" for s in plan["source_subject_ids"]}
    if set(frame.columns) != set(MANIFEST_COLUMNS) or set(frame["subject_id"]) != expected:
        raise ValueError("Projection returned forbidden columns or a non-source participant.")
    repetitions = plan["metadata_repeated_rows_per_packet"]
    if len(frame) != plan["expected_metadata_packets"] * repetitions:
        raise ValueError("Source metadata row count differs from the contract.")
    slots = np.asarray(plan["canonical_channel_ids"], dtype=int) - 1
    packets = []
    for subject in plan["source_subject_ids"]:
        person = frame[frame["subject_id"] == f"sub{subject:03d}"]
        if person["headband_order"].nunique(dropna=False) != 1:
            raise ValueError("Headband order must be constant within each participant.")
        for interface in plan["interfaces"]:
            for block in range(10):
                selected = frame[
                    (frame["subject_id"] == f"sub{subject:03d}")
                    & (frame["electrode_type"] == interface)
                    & (frame["run_id"] == f"block{block + 1:02d}")
                ]
                if len(selected) != repetitions:
                    raise ValueError("Missing, duplicated, or unknown source block packet.")
                vectors = [
                    validate_impedance(value) for value in selected["impedance_kohm_by_channel"]
                ]
                if any(v.shape != (64,) for v in vectors):
                    raise ValueError("Impedance vectors must have 64 canonical slots.")
                if any(not np.array_equal(v, vectors[0], equal_nan=True) for v in vectors[1:]):
                    raise ValueError("Class-repeated acquisition metadata disagree within block.")
                for name in ("headband_order", "condition_period"):
                    if selected[name].nunique(dropna=False) != 1:
                        raise ValueError("Acquisition nuisance fields disagree within block.")
                if selected["headband_order"].iloc[0] not in ("dry", "wet") or selected[
                    "condition_period"
                ].iloc[0] not in ("first", "second"):
                    raise ValueError("Unknown headband order or condition period.")
                expected_period = (
                    "first" if selected["headband_order"].iloc[0] == interface else "second"
                )
                if selected["condition_period"].iloc[0] != expected_period:
                    raise ValueError("Condition period contradicts participant headband order.")
                z = vectors[0][slots]
                packets.append(
                    {
                        "subject_id": subject,
                        "interface": interface,
                        "block_id": block,
                        "impedance_kohm": [float(v) if np.isfinite(v) else None for v in z],
                        "headband_order": selected["headband_order"].iloc[0],
                        "condition_period": selected["condition_period"].iloc[0],
                    }
                )
    if len(packets) != plan["expected_metadata_packets"]:
        raise ValueError("Metadata packet count differs.")
    return {
        "packets": packets,
        "returned_rows": len(frame),
        "returned_packets": len(packets),
        "returned_subject_ids": plan["source_subject_ids"],
        "columns": list(MANIFEST_COLUMNS),
    }


def project_source_metadata(plan: dict) -> dict:
    if (
        tuple(plan["source_subject_ids"]) != SOURCE_IDS
        or Path(plan["manifest_path"]) != MANIFEST_PATH
        or tuple(plan["manifest_columns"]) != MANIFEST_COLUMNS
    ):
        raise ValueError("Immutable source-only metadata boundary differs.")
    if file_sha(MANIFEST_PATH) != plan["manifest_sha256"]:
        raise ValueError("Shared public manifest hash differs.")
    frame = pd.read_parquet(
        MANIFEST_PATH,
        columns=list(MANIFEST_COLUMNS),
        filters=[("subject_id", "in", [f"sub{s:03d}" for s in SOURCE_IDS])],
    )
    return {"manifest_sha256": plan["manifest_sha256"], **metadata_projection(frame, plan)}


def covariance_descriptors(trials: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    x = np.asarray(trials, dtype=np.float64)
    if x.ndim != 3 or x.shape[-1] < 2 or not np.isfinite(x).all():
        raise ValueError("Covariance Q requires finite [trial,channel,time] arrays.")
    x = x - x.mean(axis=-1, keepdims=True)
    covariance = x @ x.transpose(0, 2, 1) / (x.shape[-1] - 1)
    h = x.shape[1]
    scale = np.trace(covariance, axis1=1, axis2=2) / h
    identity = np.eye(h)[None]
    shrunk = (
        0.8 * covariance + (0.2 * scale + np.maximum(scale, 1e-12) * 1e-8)[:, None, None] * identity
    )
    eigenvalues, eigenvectors = np.linalg.eigh(shrunk)
    if np.any(eigenvalues <= 0):
        raise ValueError("Shrinkage covariance was not positive definite.")
    full = (eigenvectors * np.log(eigenvalues)[:, None, :]) @ eigenvectors.transpose(0, 2, 1)
    diagonal = np.log(np.diagonal(shrunk, axis1=1, axis2=2))
    return full, diagonal


def gram_statistics(query: np.ndarray, support: np.ndarray) -> dict:
    """Input [band,query,channel,time] and [band,block,class,channel,time]."""
    q = np.array(query, dtype=np.float64, copy=True).reshape(*query.shape[:2], -1)
    s = np.array(support, dtype=np.float64, copy=True).reshape(*support.shape[:3], -1)
    q -= q.mean(axis=-1, keepdims=True)
    s -= s.mean(axis=-1, keepdims=True)
    return {
        "gram_cross": np.einsum("fqv,fbcv->fqcb", q, s),
        "gram_support": np.einsum("fbcv,fdcv->fcbd", s, s),
        "query_norm2": np.sum(q**2, axis=-1),
    }


def gram_template_scores(
    cross: np.ndarray,
    support_gram: np.ndarray,
    query_norm2: np.ndarray,
    query_weights: np.ndarray,
    filter_weights: np.ndarray,
) -> np.ndarray:
    w = np.asarray(query_weights, dtype=np.float64)
    k = w.shape[-1]
    if (
        k < 1
        or not np.isfinite(w).all()
        or np.any(w < 0)
        or not np.allclose(w.sum(axis=-1), 1, rtol=0, atol=1e-12)
    ):
        raise ValueError("Template weights must be finite normalized prefix weights.")
    numerator = np.einsum("...fqcb,...qb->...fqc", cross[..., :k], w)
    variance = np.einsum("...qb,...fcbd,...qd->...fqc", w, support_gram[..., :k, :k], w)
    denominator = np.sqrt(np.maximum(query_norm2[..., None] * np.maximum(variance, 0), 0))
    correlation = np.divide(
        numerator,
        denominator,
        out=np.zeros_like(numerator),
        where=denominator > 0,
    )
    correlation = np.clip(correlation, -1, 1)
    return (
        np.einsum("f,...fqc->...qc", filter_weights, np.sign(correlation) * correlation**2)
        / filter_weights.sum()
    )


def build_workspace(plan: dict, filterbank: dict, samples: int) -> dict:
    refs = make_reference_signals(
        np.asarray(plan["frequencies"]), plan["sfreq"], samples, filterbank["n_harmonics"]
    ).astype(np.float64)
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
        "class_bases": tuple(
            numeric.projection_basis(r, plan["q"]["projection_relative_cutoff"]) for r in refs
        ),
    }


def extract_cell(crop: np.ndarray, workspace: dict) -> dict:
    """Explicit [block,class,channel,time] cropped arrays; no M or query labels."""
    crop = np.ascontiguousarray(crop, dtype=np.float64)
    plan, bank = workspace["plan"], workspace["filterbank"]
    blocks, classes, channels, samples = crop.shape
    if blocks != 10 or classes != plan["n_classes"] or not np.isfinite(crop).all():
        raise ValueError("Cropped participant array violates the trial grid.")
    centered = crop - crop.mean(axis=-1, keepdims=True)
    filtered, parameters = apply_filterbank(
        centered.reshape(-1, channels, samples), sfreq=plan["sfreq"], filterbank=bank
    )
    filter_weights = np.asarray(parameters["weights"])
    f = filtered.reshape(len(filter_weights), blocks, classes, channels, samples)
    support, query = f[:, :5], f[:, 5:].reshape(len(f), -1, channels, samples)
    result = gram_statistics(query, support)
    full, diag = covariance_descriptors(filtered[0])
    full = full.reshape(blocks, classes, channels, channels)
    diag = diag.reshape(blocks, classes, channels)
    result.update(
        q_full_support=full[:5].mean(axis=1),
        q_full_query=full[5:].reshape(-1, channels, channels),
        q_diag_support=diag[:5].mean(axis=1),
        q_diag_query=diag[5:].reshape(-1, channels),
    )
    original_query = crop[5:].reshape(-1, channels, samples)
    raw_filtered, _ = apply_filterbank(original_query, sfreq=plan["sfreq"], filterbank=bank)
    cca = np.stack(
        [
            numeric.cached_cca_scores(band, workspace["cca_cache"], parameters["regularization"])
            for band in raw_filtered
        ]
    )
    result["a0_scores"] = np.einsum("f,fqc->qc", filter_weights, cca**2) / filter_weights.sum()
    result["etrca_scores"] = np.stack(
        [
            predict_filterbank_ensemble_trca(
                original_query,
                crop[:k].reshape(-1, channels, samples),
                np.tile(np.arange(classes), k),
                n_classes=classes,
                sfreq=plan["sfreq"],
                filterbank=bank,
                ridge_ratio=1e-6,
            ).scores
            for k in (3, 5)
        ]
    )
    result["reliability"] = np.asarray(
        [
            numeric.block_reliability(
                crop[b], np.arange(classes), workspace["class_bases"], plan["q"]["variance_floor"]
            )
            for b in range(5)
        ]
    )
    return result


def extract_features(raw_array: np.ndarray, plan: dict, workspaces: dict) -> dict:
    if raw_array.shape != tuple(plan["raw_shape"]) or not np.isfinite(raw_array).all():
        raise ValueError("Raw shape or finite-value contract differs.")
    records, hashes = [], []
    for cell in cells(plan):
        begin, n = plan["start_sample"], cell["n_samples"]
        crop = (
            raw_array[:, begin : begin + n, cell["interface_index"]]
            .transpose(2, 3, 0, 1)
            .astype(np.float64)
        )
        if crop.shape[-1] != n:
            raise ValueError("Declared raw crop exceeds recorded samples.")
        hashes.append(array_sha(crop))
        records.append(extract_cell(crop, workspaces[n]))
    return {
        **{key: np.stack([r[key] for r in records]) for key in FEATURE_KEYS},
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
    ):
        raise ValueError("Subject is outside the immutable exposed source39 allowlist.")
    if Path(plan["raw_root"]) != RAW_ROOT:
        raise ValueError("Raw root differs from the public source contract.")
    path = RAW_ROOT / f"S{subject:03d}.mat"
    if path.is_symlink() or path.resolve() != path:
        raise ValueError("Raw source paths must be regular, non-symlink allowlisted paths.")
    return path


def load_participant(subject: int) -> dict:
    if _WORKER is None:
        raise RuntimeError("Worker must be initialized after study start.")
    plan, workspaces = _WORKER
    path = allowed_raw_path(subject, plan)
    digest = file_sha(path)
    raw = _wearable_data(_load_arrays(path))
    return {
        "subject_id": subject,
        "raw_file_sha256": digest,
        **extract_features(raw, plan, workspaces),
    }


def collect_features(plan: dict, filterbank: dict, projection: dict, workers: int) -> dict:
    results = []
    with ProcessPoolExecutor(
        max_workers=workers, initializer=initialize_worker, initargs=(plan, filterbank)
    ) as executor:
        pending = [
            executor.submit(load_participant, subject) for subject in plan["source_subject_ids"]
        ]
        for future in as_completed(pending):
            results.append(future.result())
            print(
                json.dumps(
                    {
                        "phase": "feature_extraction",
                        "completed": len(results),
                        "total": len(SOURCE_IDS),
                    }
                ),
                flush=True,
            )
    results.sort(key=lambda r: r["subject_id"])
    metadata = {
        (p["subject_id"], p["interface"], p["block_id"]): p["impedance_kohm"]
        for p in projection["packets"]
    }
    cs = cells(plan)
    return {
        **{key: np.stack([r[key] for r in results]) for key in (*FEATURE_KEYS, "crop_sha256")},
        "schema": np.asarray(FEATURE_SCHEMA),
        "subject_ids": np.asarray([r["subject_id"] for r in results]),
        "cell_ids": np.asarray([c["cell_id"] for c in cs]),
        "sample_counts": np.asarray([c["n_samples"] for c in cs]),
        "interface_indices": np.asarray([c["interface_index"] for c in cs]),
        "query_labels": np.tile(np.arange(plan["n_classes"]), 5),
        "query_block_ids": np.repeat(plan["query_blocks"], plan["n_classes"]),
        "support_block_ids": np.asarray(plan["support_blocks"]),
        "raw_file_sha256": np.asarray([r["raw_file_sha256"] for r in results]),
        "filter_weights": np.asarray(filterbank["weights"], dtype=np.float64),
        "impedance": np.asarray(
            [
                [[metadata[s, e, b] for b in range(10)] for e in plan["interfaces"]]
                for s in plan["source_subject_ids"]
            ],
            dtype=np.float64,
        ),
    }


def q_weights(features: dict, candidate: dict, budget: int, plan: dict) -> np.ndarray:
    rel = features["reliability"][..., :budget]
    queries = features["a0_scores"].shape[-2]
    shape = (*rel.shape[:-1], queries, budget)
    if budget < 1:
        raise ValueError("Positive support budget required for Q weights.")
    if candidate["kind"] == "uniform" or budget == 1:
        return np.full(shape, 1 / budget, dtype=np.float64)
    logs = np.broadcast_to(np.log(plan["q"]["reliability_floor"] + rel)[..., None, :], shape).copy()
    kind = candidate["kind"]
    if kind in ("diag", "full"):
        support = (
            features[f"q_{kind}_support"][..., :budget, :]
            if kind == "diag"
            else features[f"q_{kind}_support"][..., :budget, :, :]
        )
        query = features[f"q_{kind}_query"]
        if kind == "diag":
            distance = np.mean((query[..., :, None, :] - support[..., None, :, :]) ** 2, axis=-1)
        else:
            distance = (
                np.sum(
                    (query[..., :, None, :, :] - support[..., None, :, :, :]) ** 2, axis=(-1, -2)
                )
                / query.shape[-1]
            )
        logs -= distance / (2 * candidate["tau"] ** 2)
    elif kind != "reliability":
        raise ValueError("Unknown EEG-only candidate kind.")
    return numeric.softmax(logs, 1.0)


def fit_metadata_scale(impedance_fit: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    values = validate_impedance(impedance_fit)
    e, h = values.shape[1], values.shape[-1]
    scale, available = np.full((e, h), 0.1), np.zeros((e, h), dtype=bool)
    for interface in range(e):
        for channel in range(h):
            current = values[:, interface, :, channel].reshape(-1)
            current = current[np.isfinite(current)]
            if len(current):
                available[interface, channel] = True
                lower, upper = np.percentile(np.log1p(current), [25, 75], method="linear")
                scale[interface, channel] = max(float(upper - lower), 0.1)
    return scale, available


def metadata_weights(
    qw: np.ndarray,
    support_m: np.ndarray,
    query_m: np.ndarray,
    scale: np.ndarray,
    available: np.ndarray,
    beta: float,
) -> np.ndarray:
    support, query = validate_impedance(support_m), validate_impedance(query_m)
    if beta == 0 or qw.shape[-1] <= 1:
        return qw
    if not math.isfinite(beta) or beta < 0 or not np.isfinite(scale).all() or np.any(scale <= 0):
        raise ValueError("Invalid metadata scale or beta.")
    common = (
        np.isfinite(query)[..., :, None, :]
        & np.isfinite(support)[..., None, :, :]
        & available[..., None, None, :]
    )
    difference = (np.log1p(query)[..., :, None, :] - np.log1p(support)[..., None, :, :]) / scale[
        ..., None, None, :
    ]
    distance = np.sum(np.where(common, difference**2, 0), axis=-1) / query.shape[-1]
    affinity = np.clip(np.exp(-beta * distance / 2), 0.25, 1)
    if np.all(affinity == 1):
        return qw
    changed = qw * affinity
    normalized = changed / changed.sum(axis=-1, keepdims=True)
    return np.where(np.all(affinity == 1, axis=-1)[..., None], qw, normalized)


def order_weights(
    qw: np.ndarray, query_blocks: np.ndarray, support_blocks: np.ndarray, beta: float
) -> np.ndarray:
    if beta == 0 or qw.shape[-1] <= 1:
        return qw
    distance = ((query_blocks[:, None] - support_blocks[None, : qw.shape[-1]]) / 10.0) ** 2
    changed = qw * np.clip(np.exp(-beta * distance / 2), 0.25, 1.0)
    return changed / changed.sum(axis=-1, keepdims=True)


def score_weights(features: dict, weights: np.ndarray) -> np.ndarray:
    return gram_template_scores(
        features["gram_cross"],
        features["gram_support"],
        features["query_norm2"],
        weights,
        features["filter_weights"],
    )


def subset_features(features: dict, indices: np.ndarray) -> dict:
    return {
        key: (
            value[indices]
            if key in (*FEATURE_KEYS, "impedance", "subject_ids", "raw_file_sha256", "crop_sha256")
            else value
        )
        for key, value in features.items()
    }


def fused_probability(p0: np.ndarray, scores: np.ndarray, temperature: float) -> np.ndarray:
    return 0.5 * p0 + 0.5 * numeric.softmax(scores, temperature)


def nll(probability: np.ndarray, labels: np.ndarray) -> float:
    truth = np.take_along_axis(
        probability, np.broadcast_to(labels[..., None], (*probability.shape[:-1], 1)), axis=-1
    )
    return float(-np.mean(np.log(np.maximum(truth, 1e-300))))


def temperature_fit(
    scores: np.ndarray, labels: np.ndarray, grid: np.ndarray, p0: np.ndarray | None = None
) -> dict:
    losses = [
        nll(numeric.softmax(scores, t) if p0 is None else fused_probability(p0, scores, t), labels)
        for t in grid
    ]
    chosen = int(np.argmin(losses))
    return {
        "selected_index": chosen,
        "temperature": float(grid[chosen]),
        "objective_values": losses,
    }


def select_candidate(candidates: list[dict], fits: dict) -> dict:
    values = [
        float(np.mean([min(fits[c["id"]][str(k)]["objective_values"]) for k in (3, 5)]))
        for c in candidates
    ]
    return {
        "candidate_id": candidates[int(np.argmin(values))]["id"],
        "candidate_objectives": values,
        "candidate_ids": [c["id"] for c in candidates],
    }


def cell_metadata(features: dict, plan: dict, budget: int) -> tuple[np.ndarray, np.ndarray]:
    values = features["impedance"][:, features["interface_indices"]]
    support = values[:, :, :budget]
    query = np.repeat(values[:, :, 5:], plan["n_classes"], axis=2)
    return support, query


def fit_fold(features: dict, fit_indices: np.ndarray, plan: dict, *, fold_id: int) -> dict:
    expected = np.flatnonzero(np.arange(len(features["subject_ids"])) % 3 != fold_id)
    if not np.array_equal(np.asarray(fit_indices), expected):
        raise ValueError("Fold fit indices must be the exact disjoint modulo-three complement.")
    fit = subset_features(features, expected)
    grid = np.logspace(-4, 1, 31, dtype=np.float64)
    labels = features["query_labels"]
    a0 = temperature_fit(fit["a0_scores"], labels, grid)
    p0 = numeric.softmax(fit["a0_scores"], a0["temperature"])
    candidates = plan["q"]["candidates"]
    temperatures, score_cache, weight_cache = {}, {}, {}
    for candidate in candidates:
        name = candidate["id"]
        temperatures[name] = {}
        for budget in BUDGETS:
            if budget == 1 and name != "uniform":
                temperatures[name]["1"] = temperatures["uniform"]["1"]
                score_cache[name, 1] = score_cache["uniform", 1]
                weight_cache[name, 1] = weight_cache["uniform", 1]
                continue
            weights = q_weights(fit, candidate, budget, plan)
            score = score_weights(fit, weights)
            weight_cache[name, budget], score_cache[name, budget] = weights, score
            temperatures[name][str(budget)] = temperature_fit(score, labels, grid, p0)
    chosen = select_candidate(candidates, temperatures)
    name = chosen["candidate_id"]
    scale, available = fit_metadata_scale(fit["impedance"])
    cell_scale, cell_available = (
        scale[fit["interface_indices"]],
        available[fit["interface_indices"]],
    )
    # Broadcasting P,L requires singleton participant prefix on interface/channel scales.
    cell_scale, cell_available = cell_scale[None], cell_available[None]

    def select_beta(beta_grid, kind):
        losses = []
        for beta in beta_grid:
            values = []
            for budget in (3, 5):
                qw = weight_cache[name, budget]
                if kind == "metadata":
                    support, query = cell_metadata(fit, plan, budget)
                    changed = metadata_weights(qw, support, query, cell_scale, cell_available, beta)
                else:
                    changed = order_weights(
                        qw, fit["query_block_ids"], fit["support_block_ids"], beta
                    )
                score = score_cache[name, budget] if changed is qw else score_weights(fit, changed)
                values.append(
                    nll(
                        fused_probability(
                            p0, score, temperatures[name][str(budget)]["temperature"]
                        ),
                        labels,
                    )
                )
            losses.append(float(np.mean(values)))
        index = int(np.argmin(losses))
        return {
            "beta_grid": beta_grid,
            "objective_values": losses,
            "selected_index": index,
            "beta": beta_grid[index],
        }

    return {
        "fold_id": fold_id,
        "fit_subject_ids": fit["subject_ids"].tolist(),
        "evaluation_subject_ids": features["subject_ids"][
            np.arange(len(features["subject_ids"])) % 3 == fold_id
        ].tolist(),
        "temperature_grid": grid.tolist(),
        "a0_temperature": a0,
        "support_temperatures": temperatures,
        "q_selection": chosen,
        "diag_selection": select_candidate(
            [c for c in candidates if c["kind"] == "diag"], temperatures
        ),
        "full_selection": select_candidate(
            [c for c in candidates if c["kind"] == "full"], temperatures
        ),
        "metadata_scale": scale.tolist(),
        "metadata_scale_available": available.tolist(),
        "m_selection": select_beta(plan["m"]["beta_grid"], "metadata"),
        "order_selection": select_beta(plan["m"]["order_beta_grid"], "order"),
        "eTRCA_temperatures": {
            str(k): temperature_fit(fit["etrca_scores"][:, :, i], labels, grid)
            for i, k in enumerate((3, 5))
        },
    }


def one_cell(features: dict, p: int, c: int) -> dict:
    return {
        **{key: features[key][p, c] for key in FEATURE_KEYS},
        "filter_weights": features["filter_weights"],
    }


def weighted_diagnostics(weights: np.ndarray | None, aq_weights: np.ndarray | None) -> dict:
    if weights is None or weights.shape[-1] == 0:
        return {"weight_turnover_vs_AQ": None, "effective_blocks": None}
    return {
        "weight_turnover_vs_AQ": float(np.mean(0.5 * np.abs(weights - aq_weights).sum(axis=-1))),
        "effective_blocks": float(np.mean(1 / np.sum(weights**2, axis=-1))),
    }


def evaluate_fold(features: dict, eval_indices: np.ndarray, freeze: dict, plan: dict) -> list[dict]:
    expected = np.flatnonzero(np.arange(len(features["subject_ids"])) % 3 == freeze["fold_id"])
    if (
        not np.array_equal(np.asarray(eval_indices), expected)
        or features["subject_ids"][expected].tolist() != freeze["evaluation_subject_ids"]
    ):
        raise ValueError("Evaluation fold does not match the frozen held development identities.")
    candidates = {c["id"]: c for c in plan["q"]["candidates"]}
    aq_name = freeze["q_selection"]["candidate_id"]
    selected = {
        "pooled_fusion": "uniform",
        "diagQ_fusion": freeze["diag_selection"]["candidate_id"],
        "fullQ_fusion": freeze["full_selection"]["candidate_id"],
        "AQ": aq_name,
    }
    labels = features["query_labels"]
    rows = []
    for p in expected:
        for c, cell in enumerate(cells(plan)):
            data = one_cell(features, p, c)
            p0 = numeric.softmax(data["a0_scores"], freeze["a0_temperature"]["temperature"])
            interface = cell["interface_index"]
            scale = np.asarray(freeze["metadata_scale"])[interface]
            available = np.asarray(freeze["metadata_scale_available"])[interface]
            metadata = features["impedance"][p, interface]
            for budget in plan["budgets"]:
                probabilities, weights = {"A0": p0}, {"A0": None}
                for method, name in selected.items():
                    if budget == 0:
                        probabilities[method], weights[method] = p0.copy(), None
                    else:
                        w = q_weights(data, candidates[name], budget, plan)
                        weights[method] = w
                        probabilities[method] = fused_probability(
                            p0,
                            score_weights(data, w),
                            freeze["support_temperatures"][name][str(budget)]["temperature"],
                        )
                aq, qw = probabilities["AQ"], weights["AQ"]
                probabilities["M_missing"], weights["M_missing"] = aq, qw

                def apply_changed(changed, qw=qw, aq=aq, p0=p0, data=data, budget=budget):
                    if changed is qw or np.array_equal(changed, qw):
                        return aq
                    probability = fused_probability(
                        p0,
                        score_weights(data, changed),
                        freeze["support_temperatures"][aq_name][str(budget)]["temperature"],
                    )
                    return np.where(np.all(changed == qw, axis=-1)[..., None], aq, probability)

                if budget <= 1:
                    for method in ("AQM", "M_stale", "order_control"):
                        probabilities[method], weights[method] = aq, qw
                    shuffled_records = []
                else:
                    support = metadata[:budget]
                    query = np.repeat(metadata[5:], plan["n_classes"], axis=0)
                    beta = freeze["m_selection"]["beta"]
                    changed = {
                        "AQM": metadata_weights(qw, support, query, scale, available, beta),
                        "M_stale": metadata_weights(
                            qw,
                            support,
                            np.broadcast_to(support[-1], query.shape),
                            scale,
                            available,
                            beta,
                        ),
                        "order_control": order_weights(
                            qw,
                            features["query_block_ids"],
                            features["support_block_ids"],
                            freeze["order_selection"]["beta"],
                        ),
                    }
                    for method, w in changed.items():
                        probabilities[method], weights[method] = apply_changed(w), w
                    shuffled_records = []
                    for permutation in numeric.derangements(budget):
                        w = metadata_weights(
                            qw, support[list(permutation)], query, scale, available, beta
                        )
                        shuffled_records.append(
                            {
                                **numeric.posterior_metrics(apply_changed(w), labels, p0, aq),
                                **weighted_diagnostics(w, qw),
                            }
                        )
                for method in plan["decoder"]["methods"]:
                    if method == "M_shuffle":
                        metric = (
                            numeric.mean_metric_records(shuffled_records)
                            if shuffled_records
                            else {
                                **numeric.posterior_metrics(aq, labels, p0, aq),
                                **weighted_diagnostics(qw, qw),
                            }
                        )
                        digest, permutations = None, len(shuffled_records)
                    else:
                        posterior = probabilities[method]
                        metric = {
                            **numeric.posterior_metrics(posterior, labels, p0, aq),
                            **weighted_diagnostics(weights[method], qw),
                        }
                        digest, permutations = array_sha(posterior), 1
                    rows.append(
                        {
                            "subject_id": int(features["subject_ids"][p]),
                            "fold_id": freeze["fold_id"],
                            **{k: v for k, v in cell.items() if k != "interface_index"},
                            "method": method,
                            "budget": budget,
                            "permutations": permutations,
                            **metric,
                            "posterior_sha256": digest,
                        }
                    )
                if budget in (3, 5):
                    posterior = numeric.softmax(
                        data["etrca_scores"][(3, 5).index(budget)],
                        freeze["eTRCA_temperatures"][str(budget)]["temperature"],
                    )
                    rows.append(
                        {
                            "subject_id": int(features["subject_ids"][p]),
                            "fold_id": freeze["fold_id"],
                            **{k: v for k, v in cell.items() if k != "interface_index"},
                            "method": "FB_eTRCA",
                            "budget": budget,
                            "permutations": 1,
                            **numeric.posterior_metrics(posterior, labels, p0, aq),
                            **weighted_diagnostics(None, qw),
                            "posterior_sha256": array_sha(posterior),
                        }
                    )
        print(
            json.dumps(
                {
                    "phase": "outer_evaluation",
                    "fold": freeze["fold_id"],
                    "subject_completed": int(features["subject_ids"][p]),
                }
            ),
            flush=True,
        )
    return rows


def summarize_results(rows: list[dict], plan: dict) -> dict:
    ids = sorted({r["subject_id"] for r in rows})
    cs = cells(plan)
    lookup = {(r["subject_id"], r["cell_id"], r["method"], r["budget"]): r for r in rows}

    def values(cid, method, budget, metric="balanced_accuracy"):
        if budget == "eauc":
            return sum(
                w * values(cid, method, k, metric) for k, w in ((0, 1 / 6), (1, 0.5), (3, 1 / 3))
            )
        return np.asarray([lookup[s, cid, method, budget][metric] for s in ids])

    def average(method, budget, sample_count=None):
        return np.mean(
            [
                values(c["cell_id"], method, budget)
                for c in cs
                if sample_count is None or c["n_samples"] == sample_count
            ],
            axis=0,
        )

    n = plan["reporting"]["cost_window_samples"]
    contrasts = [
        ("AQ_eauc_gain", "AQ", "A0", "eauc", "eauc", None),
        ("M_eauc_gain", "AQM", "AQ", "eauc", "eauc", None),
        ("M3_minus_shuffle3", "AQM", "M_shuffle", 3, 3, None),
        ("M3_minus_order3", "AQM", "order_control", 3, 3, None),
        ("M3_minus_stale3", "AQM", "M_stale", 3, 3, None),
        ("M3_minus_Q5", "AQM", "AQ", 3, 5, n),
        ("Q5_minus_Q3", "AQ", "AQ", 5, 3, n),
        ("M3_minus_eTRCA3", "AQM", "FB_eTRCA", 3, 3, n),
    ]
    comparisons = {
        name: numeric.paired_summary(average(a, ka, ns) - average(b, kb, ns))
        for name, a, b, ka, kb, ns in contrasts
    }
    comparisons["one_second_M3_BA"] = numeric.paired_summary(average("AQM", 3, n))
    comparisons["one_second_Q5_BA"] = numeric.paired_summary(average("AQ", 5, n))

    def positive(name, threshold=0.0, inclusive=False):
        record = comparisons[name]
        return (
            record["mean"] >= threshold if inclusive else record["mean"] > threshold
        ) and record["one_sided_95_LCB"] > 0

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
    aggregates, harms, attainment = [], [], []
    scalars = (
        "balanced_accuracy",
        "correct_log_probability",
        "true_class_margin",
        "prediction_flips_vs_A0",
        "prediction_flips_vs_AQ",
        "correct_count",
        "query_count",
        "weight_turnover_vs_AQ",
        "effective_blocks",
    )
    methods = [*plan["decoder"]["methods"], "FB_eTRCA"]
    for cell in cs:
        cid = cell["cell_id"]
        for method in methods:
            budgets = (3, 5) if method == "FB_eTRCA" else plan["budgets"]
            curves = {str(k): values(cid, method, k).tolist() for k in budgets}
            for k in budgets:
                mean = {}
                for key in scalars:
                    observed = [lookup[s, cid, method, k][key] for s in ids]
                    mean[key] = None if observed[0] is None else float(np.mean(observed))
                aggregates.append(
                    {
                        **{k: v for k, v in cell.items() if k != "interface_index"},
                        "method": method,
                        "budget": k,
                        "mean_metrics": mean,
                        "fraction_harmed_vs_A0": float(
                            np.mean(values(cid, method, k) < values(cid, "A0", k))
                        ),
                        "fraction_harmed_vs_AQ": float(
                            np.mean(values(cid, method, k) < values(cid, "AQ", k))
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
                    if method == "M_shuffle"
                    else "observed_budget_only",
                }
            )
    for method in methods:
        for k in (3, 5) if method == "FB_eTRCA" else plan["budgets"]:
            value, a0, aq = average(method, k), average("A0", k), average("AQ", k)
            for i, subject in enumerate(ids):
                harms.append(
                    {
                        "subject_id": subject,
                        "method": method,
                        "budget": k,
                        "equal_cell_mean_BA": float(value[i]),
                        "delta_vs_A0": float(value[i] - a0[i]),
                        "delta_vs_AQ": float(value[i] - aq[i]),
                    }
                )
    return {
        "status": status,
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


def write_features(path: Path, features: dict, quota: int) -> str:
    check_quota(path.parent, sum(v.nbytes for v in features.values()) + 65536, quota)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o400)
    with os.fdopen(descriptor, "wb") as handle:
        np.savez_compressed(handle, **features)
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
    projection = {
        "schema": "cfeg.context-template-source.projection.v1",
        "study_id": plan["study_id"],
        "source_commit": start["source_commit"],
        "plan_sha256": start["plan_sha256"],
        "start_sha256": start_sha,
        **project_source_metadata(plan),
    }
    projection_sha = write_json(output / "source-projection.json", projection, quota)
    features = collect_features(plan, filterbank, projection, workers)
    features_sha = write_features(output / "features.npz", features, quota)
    folds = []
    for fold in range(3):
        folds.append(
            fit_fold(
                features,
                np.flatnonzero(np.arange(len(features["subject_ids"])) % 3 != fold),
                plan,
                fold_id=fold,
            )
        )
        print(json.dumps({"phase": "fit", "fold_completed": fold}), flush=True)
    freezes = {
        "schema": "cfeg.context-template-source.fold-freezes.v1",
        "study_id": plan["study_id"],
        "source_commit": start["source_commit"],
        "plan_sha256": start["plan_sha256"],
        "start_sha256": start_sha,
        "source_projection_sha256": projection_sha,
        "features_sha256": features_sha,
        "folds": folds,
    }
    freeze_sha = write_json(output / "fold-freezes.json", freezes, quota)
    rows = []
    for freeze in folds:
        rows.extend(
            evaluate_fold(
                features,
                np.flatnonzero(np.arange(len(features["subject_ids"])) % 3 == freeze["fold_id"]),
                freeze,
                plan,
            )
        )
    rows.sort(key=lambda r: (r["subject_id"], r["cell_id"], r["budget"], r["method"]))
    summary = summarize_results(rows, plan)
    result = {
        "schema": "cfeg.context-template-source.result.v1",
        "study_id": plan["study_id"],
        "evidence_role": plan["evidence_role"],
        "source_commit": start["source_commit"],
        "plan_sha256": start["plan_sha256"],
        "start_sha256": start_sha,
        "source_projection_sha256": projection_sha,
        "features_sha256": features_sha,
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
        or plan_path.resolve() != root / "configs/analysis/context_template_source39_v1.json"
        or file_sha(plan_path) != PLAN_SHA256
    ):
        raise ValueError("Exact absolute frozen context-source plan required.")
    plan = json.loads(plan_path.read_text())
    if (
        tuple(plan["source_subject_ids"]) != SOURCE_IDS
        or Path(plan["raw_root"]) != RAW_ROOT
        or Path(plan["manifest_path"]) != MANIFEST_PATH
        or tuple(plan["manifest_columns"]) != MANIFEST_COLUMNS
    ):
        raise ValueError("Immutable source39 input boundary differs.")
    if (
        output.resolve() != output
        or str(output) != plan["execution"]["output_root"]
        or workers != plan["execution"]["workers"]
    ):
        raise ValueError("Exact output and worker contract required.")
    if (output / "start.json").exists():
        raise FileExistsError("Existing start consumes the attempt; no automatic retry or resume.")

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
        "schema": "cfeg.context-template-source.start.v1",
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
        "runtime": {
            "python": sys.version,
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "pandas": pd.__version__,
            "platform": platform.platform(),
            "blas_threads": {key: os.environ[key] for key in thread_keys},
        },
    }
    start_sha = write_json(output / "start.json", start, plan["execution"]["resource_budget_bytes"])
    return execute_phases(plan, filterbank, output, start, start_sha, workers)
