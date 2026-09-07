"""Frozen paired reference-guided calibration diagnostic on exposed source39.

Only the guarded one-attempt runner opens allowlisted EEG. Numerical kernels
use explicit arrays; no metadata, study RNG or closed-study execution is used.
"""

from __future__ import annotations

import copy
import json
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
from scipy import signal

from cfeg.analysis import metadata_calibration_v4_pilot as numeric
from cfeg.analysis import spatial_calibration_source as spatial
from cfeg.baselines.fbcca import apply_filterbank, make_reference_signals
from cfeg.data.prepare_mat import _load_arrays
from cfeg.data.prepare_wearable import _wearable_data

PLAN_SHA256 = "d7f89ef223d0ad0637f449bc021711167f9f1d5b8dabeba068b6bf8c11131e21"
FEATURE_SCHEMA = "cfeg.reference-calibration-source.features.v1"
SOURCE_IDS = spatial.SOURCE_IDS
RAW_ROOT = spatial.RAW_ROOT
BUDGETS = (1, 3, 5)
PIPELINES = ("no_notch", "causal_notch50")
FEATURE_KEYS = (
    "ecca_components",
    "wrong_ecca_scores",
    "itcca_band_rho",
    "aref_band_rho",
    "support_correlations",
    "reference_projection_fraction",
    "line50_fraction",
)
_WORKER = None


def covariance_statistics(values: np.ndarray, ridge: float) -> dict:
    """Per-trial centered covariance and whitening in original channel coordinates."""
    values = np.ascontiguousarray(values, dtype=np.float64)
    if (
        values.ndim != 3
        or min(values.shape) < 1
        or values.shape[-1] < 2
        or not np.isfinite(values).all()
        or not np.isfinite(ridge)
        or ridge < 0
    ):
        raise ValueError("CCA needs finite [trial,channel,time>=2] arrays and nonnegative ridge.")
    centered = values - values.mean(axis=-1, keepdims=True)
    samples, channels = values.shape[-1], values.shape[-2]
    covariance = centered @ centered.swapaxes(-1, -2) / (samples - 1)
    scales = np.trace(covariance, axis1=-2, axis2=-1) / channels
    regularized = covariance + ridge * scales[:, None, None] * np.eye(channels)
    eigenvalues, eigenvectors = np.linalg.eigh(regularized)
    cutoff = np.maximum(np.finfo(np.float64).eps, np.maximum(eigenvalues[:, -1], 0) * 1e-12)
    roots = np.zeros_like(eigenvalues)
    np.divide(
        1, np.sqrt(np.maximum(eigenvalues, 0)), out=roots, where=eigenvalues > cutoff[:, None]
    )
    inverse = (eigenvectors * roots[:, None, :]) @ eigenvectors.swapaxes(-1, -2)
    floor = np.finfo(np.float64).eps * np.sqrt(samples) * 10
    valid = (np.linalg.norm(centered, axis=(-2, -1)) > floor) & (scales > 0)
    return {"centered": centered, "covariance": covariance, "inverse": inverse, "valid": valid}


def safe_correlation(
    cross: np.ndarray, first_variance: np.ndarray, second_variance: np.ndarray
) -> np.ndarray:
    denominator = np.sqrt(np.maximum(first_variance, 0) * np.maximum(second_variance, 0))
    result = np.zeros(np.broadcast_shapes(cross.shape, denominator.shape), dtype=np.float64)
    np.divide(cross, denominator, out=result, where=denominator > 0)
    return np.clip(result, -1, 1)


def canonical_pairwise(first: dict, second: dict) -> dict:
    """All trial pairs; return spatial vectors and actual projected Pearson."""
    x, y = first["centered"], second["centered"]
    if x.shape[-1] != y.shape[-1]:
        raise ValueError("CCA time windows must match.")
    cross = np.einsum("qht,cjt->qchj", x, y) / (x.shape[-1] - 1)
    whitened = first["inverse"][:, None] @ cross @ second["inverse"][None]
    left, singular, right = np.linalg.svd(whitened, full_matrices=False)
    u = np.einsum("qhi,qci->qch", first["inverse"], left[..., 0])
    v = np.einsum("chi,qci->qch", second["inverse"], right[..., 0, :])
    valid = first["valid"][:, None] & second["valid"][None]
    u, v = np.where(valid[..., None], u, 0.0), np.where(valid[..., None], v, 0.0)
    pearson = safe_correlation(
        np.einsum("qch,qchj,qcj->qc", u, cross, v),
        np.einsum("qch,qhj,qcj->qc", u, first["covariance"], u),
        np.einsum("qch,chj,qcj->qc", v, second["covariance"], v),
    )
    # Couple the pair's signs explicitly. Shared-filter terms use one vector twice.
    v = np.where((pearson < 0)[..., None], -v, v)
    return {
        "u": u,
        "v": v,
        "pearson": np.abs(pearson),
        "singular": np.where(valid, np.clip(singular[..., 0], 0, 1), 0.0),
        "cross": cross,
    }


def prepare_query(query: np.ndarray, workspace: dict) -> dict:
    stats = covariance_statistics(query, workspace["plan"]["decoder"]["ECCA_ridge"])
    return {
        "statistics": stats,
        "reference_pair": canonical_pairwise(stats, workspace["reference_statistics"]),
        "itcca_whitened": spatial.whiten_trials(query, workspace["plan"]["decoder"]["ITCCA_ridge"]),
    }


def fit_ecca_templates(support: np.ndarray, workspace: dict) -> dict:
    """Support [prefix,class,channel,time]; all template/reference pairings cached."""
    support = np.ascontiguousarray(support, dtype=np.float64)
    if (
        support.ndim != 4
        or min(support.shape) < 1
        or support.shape[-1] < 2
        or not np.isfinite(support).all()
    ):
        raise ValueError("Expected finite nonempty support prefix grid.")
    templates = support.mean(axis=0)
    if len(templates) != len(workspace["references"]):
        raise ValueError("Every candidate requires one support template.")
    stats = covariance_statistics(templates, workspace["plan"]["decoder"]["ECCA_ridge"])
    return {
        "statistics": stats,
        "reference_pair": canonical_pairwise(stats, workspace["reference_statistics"]),
        "itcca_whitened": spatial.whiten_trials(
            templates, workspace["plan"]["decoder"]["ITCCA_ridge"]
        ),
    }


def ecca_query_features(query: dict, model: dict) -> dict:
    """Four Pearson terms; candidate references stay fixed under template shifts."""
    x, t = query["statistics"], model["statistics"]
    if x["centered"].shape[1:] != t["centered"].shape[1:]:
        raise ValueError("Shared-filter eCCA needs equal EEG channel/time dimensions.")
    xt = canonical_pairwise(x, t)
    u1, u2, u3 = query["reference_pair"]["u"], xt["u"], model["reference_pair"]["u"]
    cross, cxx, ctt = xt["cross"], x["covariance"], t["covariance"]
    r1 = query["reference_pair"]["pearson"]
    r2 = safe_correlation(
        np.einsum("qjh,qjhd,qjd->qj", u2, cross, u2),
        np.einsum("qjh,qhd,qjd->qj", u2, cxx, u2),
        np.einsum("qjh,jhd,qjd->qj", u2, ctt, u2),
    )
    r3 = safe_correlation(
        np.einsum("qch,qjhd,qcd->qjc", u1, cross, u1),
        np.einsum("qch,qhd,qcd->qc", u1, cxx, u1)[:, None],
        np.einsum("qch,jhd,qcd->qjc", u1, ctt, u1),
    )
    r4 = safe_correlation(
        np.einsum("jch,qjhd,jcd->qjc", u3, cross, u3),
        np.einsum("jch,qhd,jcd->qjc", u3, cxx, u3),
        np.einsum("jch,jhd,jcd->jc", u3, ctt, u3)[None],
    )
    classes = np.arange(r1.shape[-1])
    wrong = []
    for shift in range(len(classes)):
        j = (classes - shift) % len(classes)
        coefficients = np.stack((r1, r2[:, j], r3[:, j, classes], r4[:, j, classes]), axis=-1)
        if shift == 0:
            components = coefficients
        else:
            wrong.append(np.sum(np.sign(coefficients) * coefficients**2, axis=-1))
    return {
        "components": components,
        "wrong_band_scores": np.stack(wrong),
        "itcca_rho": spatial.whitened_pairwise_cca(
            query["itcca_whitened"], model["itcca_whitened"]
        ),
    }


def rowspace_basis(reference: np.ndarray) -> np.ndarray:
    centered = np.asarray(reference, dtype=np.float64)
    centered = centered - centered.mean(axis=-1, keepdims=True)
    _, singular, right = np.linalg.svd(centered, full_matrices=False)
    cutoff = max(np.finfo(np.float64).eps, float(singular[0]) * 1e-12)
    return right[singular > cutoff]


def projection_fraction(values: np.ndarray, bases: tuple[np.ndarray, ...]) -> np.ndarray:
    """All-reference descriptive energy fractions; never used by a decoder."""
    centered = np.asarray(values, dtype=np.float64)
    centered = centered - centered.mean(axis=-1, keepdims=True)
    energy = np.sum(centered**2, axis=(-2, -1))
    fractions = []
    for basis in bases:
        projected = np.sum((centered @ basis.T) ** 2, axis=(-2, -1))
        ratio = np.zeros_like(energy)
        np.divide(projected, energy, out=ratio, where=energy > 0)
        fractions.append(np.clip(ratio, 0, 1))
    return np.stack(fractions, axis=-1)


def signal_diagnostics(crop: np.ndarray, filtered_grid: np.ndarray, workspace: dict) -> dict:
    centered = filtered_grid - filtered_grid.mean(axis=-1, keepdims=True)
    support = centered[:, :5].transpose(0, 2, 1, 3, 4)
    support = support.reshape(*support.shape[:3], -1)
    norms = np.sum(support**2, axis=-1)
    correlations = safe_correlation(
        support @ support.swapaxes(-1, -2), norms[..., :, None], norms[..., None, :]
    )
    return {
        "support_correlations": correlations,
        "reference_projection_fraction": projection_fraction(
            filtered_grid, workspace["reference_bases"]
        ),
        "line50_fraction": projection_fraction(crop, (workspace["line50_basis"],))[..., 0],
    }


def build_workspace(plan: dict, filterbank: dict, samples: int) -> dict:
    references = make_reference_signals(
        plan["frequencies"], plan["sfreq"], samples, filterbank["n_harmonics"]
    ).astype(np.float64)
    line = make_reference_signals([50.0], plan["sfreq"], samples, 1)[0].astype(np.float64)
    return {
        "plan": plan,
        "filterbank": filterbank,
        "references": references,
        "reference_statistics": covariance_statistics(references, plan["decoder"]["ECCA_ridge"]),
        "reference_bases": tuple(rowspace_basis(r) for r in references),
        "line50_basis": rowspace_basis(line),
        "legacy_cache": numeric.reference_cache(
            np.asarray(plan["frequencies"]),
            plan["sfreq"],
            samples,
            filterbank["n_harmonics"],
            filterbank["regularization"],
        ),
    }


def prepare_pipeline_crop(
    raw_array: np.ndarray, cell: dict, plan: dict, pipeline: str
) -> np.ndarray:
    if pipeline not in PIPELINES:
        raise ValueError("Unknown frozen preprocessing pipeline.")
    begin, n = plan["start_sample"], cell["n_samples"]
    prefix = np.ascontiguousarray(
        raw_array[:, : begin + n, cell["interface_index"]].transpose(2, 3, 0, 1),
        dtype=np.float64,
    )
    if prefix.shape[-1] != begin + n or not np.isfinite(prefix).all():
        raise ValueError("Current-trial prefix is unavailable or nonfinite.")
    if pipeline == "causal_notch50":
        b, a = signal.iirnotch(
            plan["preprocessing"]["notch_frequency_hz"],
            plan["preprocessing"]["notch_quality_factor"],
            fs=plan["sfreq"],
        )
        prefix = signal.lfilter(b, a, prefix, axis=-1)
    return np.ascontiguousarray(prefix[..., begin : begin + n])


def extract_cell(crop: np.ndarray, workspace: dict, *, include_legacy: bool = False) -> dict:
    """All-class scoring on a retained [10,class,channel,time] grid."""
    crop = np.ascontiguousarray(crop, dtype=np.float64)
    plan, bank = workspace["plan"], workspace["filterbank"]
    if crop.ndim != 4 or crop.shape[:2] != (10, plan["n_classes"]) or not np.isfinite(crop).all():
        raise ValueError("Crop violates the finite chronological trial grid.")
    _, classes, channels, samples = crop.shape
    filtered, parameters = apply_filterbank(
        crop.reshape(-1, channels, samples), sfreq=plan["sfreq"], filterbank=bank
    )
    weights = np.asarray(parameters["weights"], dtype=np.float64)
    grid = filtered.reshape(len(weights), 10, classes, channels, samples)
    components, wrong, itcca, aref = [], [], [], []
    for band in grid:
        query = prepare_query(band[5:].reshape(-1, channels, samples), workspace)
        records = [
            ecca_query_features(query, fit_ecca_templates(band[:k], workspace)) for k in BUDGETS
        ]
        components.append(np.stack([r["components"] for r in records]))
        wrong.append(np.stack([r["wrong_band_scores"] for r in records]))
        itcca.append(np.stack([r["itcca_rho"] for r in records]))
        aref.append(query["reference_pair"]["pearson"])
    result = {
        "ecca_components": np.stack(components, axis=1),
        "wrong_ecca_scores": np.einsum("f,fksqc->ksqc", weights, np.stack(wrong)) / weights.sum(),
        "itcca_band_rho": np.stack(itcca, axis=1),
        "aref_band_rho": np.stack(aref),
        **signal_diagnostics(crop, grid, workspace),
    }
    if include_legacy:
        rho = np.stack(
            [
                numeric.cached_cca_scores(
                    band[5:].reshape(-1, channels, samples),
                    workspace["legacy_cache"],
                    parameters["regularization"],
                )
                for band in grid
            ]
        )
        result["legacy_a0_scores"] = np.einsum("f,fqc->qc", weights, rho**2) / weights.sum()
    return result


def extract_features(raw_array: np.ndarray, plan: dict, workspaces: dict) -> dict:
    if raw_array.shape != tuple(plan["raw_shape"]):
        raise ValueError("Raw shape differs from the native axis contract.")
    records, hashes = [], []
    for pipeline in PIPELINES:
        pipeline_records, pipeline_hashes = [], []
        for cell in spatial.cells(plan):
            crop = prepare_pipeline_crop(raw_array, cell, plan, pipeline)
            pipeline_hashes.append(spatial.array_sha(crop))
            pipeline_records.append(
                extract_cell(
                    crop,
                    workspaces[cell["n_samples"]],
                    include_legacy=pipeline == "no_notch",
                )
            )
        records.append(pipeline_records)
        hashes.append(pipeline_hashes)
    return {
        **{
            key: np.stack([np.stack([r[key] for r in arm]) for arm in records])
            for key in FEATURE_KEYS
        },
        "legacy_a0_scores": np.stack([r["legacy_a0_scores"] for r in records[0]]),
        "crop_sha256": np.asarray(hashes),
    }


def initialize_worker(plan: dict, filterbank: dict) -> None:
    global _WORKER
    _WORKER = (plan, {n: build_workspace(plan, filterbank, n) for n in plan["sample_counts"]})


def allowed_raw_path(subject: int, plan: dict) -> Path:
    return spatial.allowed_raw_path(subject, plan)


def load_participant(subject: int) -> dict:
    if _WORKER is None:
        raise RuntimeError("Worker input access requires initialization after durable start.")
    plan, workspaces = _WORKER
    path = allowed_raw_path(subject, plan)
    digest = spatial.file_sha(path)
    raw = _wearable_data(_load_arrays(path))
    return {
        "subject_id": subject,
        "raw_file_sha256": digest,
        **extract_features(raw, plan, workspaces),
    }


def collect_features(plan: dict, filterbank: dict, workers: int) -> dict:
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
                        "phase": "feature_extraction",
                        "completed": len(results),
                        "total": len(SOURCE_IDS),
                    }
                ),
                flush=True,
            )
    results.sort(key=lambda r: r["subject_id"])
    cells = spatial.cells(plan)
    return {
        **{
            key: np.stack([r[key] for r in results])
            for key in (*FEATURE_KEYS, "legacy_a0_scores", "crop_sha256")
        },
        "schema": np.asarray(FEATURE_SCHEMA),
        "subject_ids": np.asarray([r["subject_id"] for r in results]),
        "pipeline_ids": np.asarray(PIPELINES),
        "cell_ids": np.asarray([c["cell_id"] for c in cells]),
        "sample_counts": np.asarray([c["n_samples"] for c in cells]),
        "interface_indices": np.asarray([c["interface_index"] for c in cells]),
        "query_labels": np.tile(np.arange(plan["n_classes"]), 5),
        "query_block_ids": np.repeat(plan["query_blocks"], plan["n_classes"]),
        "support_block_ids": np.asarray(plan["support_blocks"]),
        "calibration_budgets": np.asarray(BUDGETS),
        "wrong_label_shifts": np.arange(1, plan["n_classes"]),
        "filter_weights": np.asarray(filterbank["weights"], dtype=np.float64),
        "raw_file_sha256": np.asarray([r["raw_file_sha256"] for r in results]),
    }


def band_average(values: np.ndarray, weights: np.ndarray) -> np.ndarray:
    return np.einsum("f,...fqc->...qc", weights, values) / weights.sum()


def decoder_scores(cache: dict) -> dict:
    coefficients, weights = cache["ecca_components"], cache["filter_weights"]
    return {
        "A0_reference": band_average(cache["aref_band_rho"] ** 2, weights),
        "ECCA": band_average(np.sum(np.sign(coefficients) * coefficients**2, axis=-1), weights),
        "ITCCA": band_average(cache["itcca_band_rho"] ** 2, weights),
    }


def fit_fold(cache: dict, fit_indices: np.ndarray, plan: dict, *, fold_id: int) -> dict:
    positions = np.arange(len(cache["subject_ids"]))
    expected = np.flatnonzero(positions % 3 != fold_id)
    evaluation = np.flatnonzero(positions % 3 == fold_id)
    if fold_id not in (0, 1, 2) or not np.array_equal(np.asarray(fit_indices), expected):
        raise ValueError("Fit identities must match the deterministic outer split.")
    scores, pipelines, legacy = decoder_scores(cache), {}, {}
    labels = cache["query_labels"]
    for arm, pipeline in enumerate(PIPELINES):
        windows = {}
        for n in plan["sample_counts"]:
            indices = np.flatnonzero(cache["sample_counts"] == n)
            a0fit = spatial.temperature_fit(
                scores["A0_reference"][expected, arm][:, indices], labels
            )
            budgets = {
                str(k): {
                    "ecca_temperature": spatial.temperature_fit(
                        scores["ECCA"][expected, arm][:, indices, ki], labels
                    ),
                    "itcca_temperature": spatial.temperature_fit(
                        scores["ITCCA"][expected, arm][:, indices, ki], labels
                    ),
                }
                for ki, k in enumerate(BUDGETS)
            }
            windows[str(n)] = {"aref_temperature": a0fit, "budgets": budgets}
        pipelines[pipeline] = {"windows": windows}
    for n in plan["sample_counts"]:
        indices = np.flatnonzero(cache["sample_counts"] == n)
        legacy[str(n)] = spatial.temperature_fit(
            cache["legacy_a0_scores"][expected][:, indices], labels
        )
    return {
        "fold_id": fold_id,
        "fit_subject_ids": cache["subject_ids"][expected].tolist(),
        "evaluation_subject_ids": cache["subject_ids"][evaluation].tolist(),
        "pipelines": pipelines,
        "legacy_a0_temperatures": legacy,
    }


def evaluate_fold(cache: dict, eval_indices: np.ndarray, freeze: dict, plan: dict) -> list[dict]:
    expected = np.flatnonzero(np.arange(len(cache["subject_ids"])) % 3 == freeze["fold_id"])
    if (
        not np.array_equal(np.asarray(eval_indices), expected)
        or cache["subject_ids"][expected].tolist() != freeze["evaluation_subject_ids"]
    ):
        raise ValueError("Evaluation identities differ from the frozen outer split.")
    scores, labels, rows = decoder_scores(cache), cache["query_labels"], []
    for p in expected:
        for arm, pipeline in enumerate(PIPELINES):
            for c, cell in enumerate(spatial.cells(plan)):
                window = freeze["pipelines"][pipeline]["windows"][str(cell["n_samples"])]
                p0 = numeric.softmax(
                    scores["A0_reference"][p, arm, c], window["aref_temperature"]["temperature"]
                )
                for k in plan["budgets"]:
                    probabilities, permutations = {"A0_reference": p0}, []
                    if k == 0:
                        probabilities["ECCA"] = probabilities["wrong_label_support"] = p0
                    else:
                        ki = BUDGETS.index(k)
                        fits = window["budgets"][str(k)]
                        t = fits["ecca_temperature"]["temperature"]
                        probabilities["ECCA"] = numeric.softmax(scores["ECCA"][p, arm, c, ki], t)
                        probabilities["ITCCA"] = numeric.softmax(
                            scores["ITCCA"][p, arm, c, ki], fits["itcca_temperature"]["temperature"]
                        )
                        for shift in range(1, plan["n_classes"]):
                            posterior = numeric.softmax(
                                cache["wrong_ecca_scores"][p, arm, c, ki, shift - 1], t
                            )
                            permutations.append(
                                {"shift": shift, **spatial.posterior_metrics(posterior, labels, p0)}
                            )
                    available_methods = dict(plan["decoder"]["method_budgets"])
                    if pipeline == "no_notch":
                        probabilities["legacy_A0"] = numeric.softmax(
                            cache["legacy_a0_scores"][p, c],
                            freeze["legacy_a0_temperatures"][str(cell["n_samples"])]["temperature"],
                        )
                        available_methods["legacy_A0"] = plan["decoder"]["legacy_budgets"]
                    for method, available in available_methods.items():
                        if k not in available:
                            continue
                        if method == "wrong_label_support" and k > 0:
                            metrics, digest, count = (
                                spatial.mean_metrics(permutations),
                                None,
                                len(permutations),
                            )
                        else:
                            posterior = probabilities[method]
                            metrics, digest = (
                                spatial.posterior_metrics(posterior, labels, p0),
                                spatial.array_sha(posterior),
                            )
                            count = 0 if method == "wrong_label_support" else 1
                        row = {
                            "subject_id": int(cache["subject_ids"][p]),
                            "fold_id": freeze["fold_id"],
                            "pipeline": pipeline,
                            **{
                                key: value
                                for key, value in cell.items()
                                if key != "interface_index"
                            },
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
    """Same descriptive screens per arm; paired diagnostics never select a winner."""
    to_prior = {"A0_reference": "A0", "ECCA": "AQ_ITCCA"}
    from_prior = {value: key for key, value in to_prior.items()}
    nested = {}
    for pipeline in PIPELINES:
        pipeline_rows = []
        for row in rows:
            if row["pipeline"] == pipeline:
                record = {key: value for key, value in row.items() if key != "pipeline"}
                record["method"] = to_prior.get(record["method"], record["method"])
                pipeline_rows.append(record)
        pipeline_plan = copy.deepcopy(plan)
        methods = {
            to_prior.get(method, method): budgets
            for method, budgets in plan["decoder"]["method_budgets"].items()
        }
        if pipeline == "no_notch":
            methods["legacy_A0"] = plan["decoder"]["legacy_budgets"]
        pipeline_plan["decoder"]["method_budgets"] = methods
        summary = spatial.summarize_results(pipeline_rows, pipeline_plan)
        for key in ("aggregate_rows", "participant_harm", "calibration_attainment"):
            for row in summary[key]:
                row["method"] = from_prior.get(row["method"], row["method"])
        for record in summary["calibration_attainment"]:
            record["history_seconds"] = [
                plan["n_classes"] * k * plan["preprocessing"]["history_samples"] / plan["sfreq"]
                if isinstance(k, int)
                else None
                for k in record["first_observed_budget"]
            ]
        nested[pipeline] = summary
    ids = sorted({r["subject_id"] for r in rows})
    lookup = {
        (r["subject_id"], r["pipeline"], r["cell_id"], r["method"], r["budget"]): r for r in rows
    }

    def average(pipeline, method, budget):
        if budget == "eauc":
            return sum(
                w * average(pipeline, method, k) for k, w in ((0, 1 / 6), (1, 0.5), (3, 1 / 3))
            )
        return np.mean(
            [
                [
                    lookup[s, pipeline, cell["cell_id"], method, budget]["balanced_accuracy"]
                    for s in ids
                ]
                for cell in spatial.cells(plan)
            ],
            axis=0,
        )

    contrasts = {
        "calibration_gain_interaction": numeric.paired_summary(
            np.asarray(
                nested["causal_notch50"]["comparisons"]["AQ_eauc_gain"]["participant_values"]
            )
            - np.asarray(nested["no_notch"]["comparisons"]["AQ_eauc_gain"]["participant_values"])
        ),
        **{
            f"{method}_eauc": numeric.paired_summary(
                average("causal_notch50", method, "eauc") - average("no_notch", method, "eauc")
            )
            for method in ("ECCA", "A0_reference", "wrong_label_support")
        },
        **{
            f"ITCCA_k{k}": numeric.paired_summary(
                average("causal_notch50", "ITCCA", k) - average("no_notch", "ITCCA", k)
            )
            for k in BUDGETS
        },
    }
    return {
        "status": "DIAGNOSTIC_COMPLETE",
        "human_held_unlock": False,
        "pipeline_summaries": nested,
        "paired_pipeline_diagnostics": contrasts,
    }


def execute_phases(
    plan: dict, filterbank: dict, output: Path, start: dict, start_sha: str, workers: int
) -> dict:
    quota = plan["execution"]["resource_budget_bytes"]
    features = collect_features(plan, filterbank, workers)
    features_sha = spatial.write_scores(output / "features.npz", features, quota)
    provenance = {
        "study_id": plan["study_id"],
        "source_commit": start["source_commit"],
        "plan_sha256": start["plan_sha256"],
        "start_sha256": start_sha,
        "features_sha256": features_sha,
    }
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
        "schema": "cfeg.reference-calibration-source.fold-freezes.v1",
        **provenance,
        "folds": folds,
    }
    freeze_sha = spatial.write_json(output / "fold-freezes.json", freezes, quota)
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
    rows.sort(
        key=lambda r: (r["subject_id"], r["pipeline"], r["cell_id"], r["budget"], r["method"])
    )
    if len(rows) != plan["reporting"]["expected_rows"]:
        raise RuntimeError("Frozen metric row count differs.")
    summary = summarize_results(rows, plan)
    result = {
        "schema": "cfeg.reference-calibration-source.result.v1",
        **provenance,
        "evidence_role": plan["evidence_role"],
        "fold_freezes_sha256": freeze_sha,
        "rows": rows,
        "summary": summary,
    }
    result_sha = spatial.write_json(output / "result.json", result, quota)
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
        or plan_path.resolve() != root / "configs/analysis/reference_calibration_source39_v1.json"
        or spatial.file_sha(plan_path) != PLAN_SHA256
    ):
        raise ValueError("Exact absolute frozen reference-source plan required.")
    plan = json.loads(plan_path.read_text())
    if (
        tuple(plan["source_subject_ids"]) != SOURCE_IDS
        or Path(plan["raw_root"]) != RAW_ROOT
        or tuple(plan["pipeline_ids"]) != PIPELINES
    ):
        raise ValueError("Immutable source39 input or pipeline boundary differs.")
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
        if spatial.file_sha(root / filename) != expected:
            raise ValueError(f"Pinned source changed: {filename}")
    thread_keys = ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS")
    if any(os.environ.get(key) != "1" for key in thread_keys):
        raise RuntimeError("All BLAS thread variables must be 1 before numerical imports.")
    bank = yaml.safe_load((root / plan["filterbank_config"]).read_text())
    output.mkdir(parents=True, exist_ok=True)
    if any(output.iterdir()):
        raise FileExistsError("Study output must be a new empty directory.")
    start = {
        "schema": "cfeg.reference-calibration-source.start.v1",
        "study_id": plan["study_id"],
        "source_commit": git("rev-parse", "HEAD"),
        "source_tree": git("rev-parse", "HEAD^{tree}"),
        "plan_sha256": PLAN_SHA256,
        "pinned_files": plan["pinned_files"],
        "source_subject_ids": list(SOURCE_IDS),
        "pipeline_ids": list(PIPELINES),
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
    start_sha = spatial.write_json(
        output / "start.json", start, plan["execution"]["resource_budget_bytes"]
    )
    return execute_phases(plan, bank, output, start, start_sha, workers)
