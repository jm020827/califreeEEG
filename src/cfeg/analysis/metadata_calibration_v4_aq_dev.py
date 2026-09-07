"""Metadata-free, source-calibrated AQ engineering study with independent evaluation.

Study namespaces are opened only by ``run_study`` after durable publication gates.
Array-level functions take explicit fixture seeds or scores, never evaluation labels.
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
from scipy import signal

from cfeg.analysis import metadata_calibration_v4_pilot as numeric
from cfeg.baselines.fbcca import apply_filterbank, make_reference_signals

PLAN_SHA256 = "428385a7b676212a236ae80a76c3f516fade9c241b5d5f6a2f00b22ce0052b3e"
CACHE_SCHEMA = "cfeg.v4-aq-study.scores.v1"
COMPONENTS = {
    name: index + 1
    for index, name in enumerate(
        ("common", "spatial", "phase", "jitter", "innovation", "signs", "state", "order")
    )
}
METHODS = (
    "A0_raw",
    "A0_cal",
    "pooled_raw",
    "pooled_cal",
    "block_raw",
    "block_cal",
    "fusion_pooled_raw",
    "fusion_pooled_cal",
    "fusion_block_raw",
    "fusion_block_cal",
)
POSITIVE_BUDGETS = (1, 3, 5)
_WORKER = None


def component_rng(root_seed: int, participant: int, component: str) -> np.random.Generator:
    if root_seed < 0 or participant < 0 or component not in COMPONENTS:
        raise ValueError("Invalid explicit RNG coordinate.")
    return np.random.Generator(
        np.random.PCG64DXSM(np.random.SeedSequence([root_seed, participant, COMPONENTS[component]]))
    )


def array_sha(array: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def cells(plan: dict) -> list[dict]:
    return [
        {
            "cell_id": f"{family}-n{samples:03d}-m{multiplier:g}",
            "family": family,
            "n_samples": samples,
            "noise_multiplier": multiplier,
        }
        for samples in plan["sample_counts"]
        for multiplier in plan["noise_multipliers"]
        for family in plan["families"]
    ]


def generate_base(plan: dict, participant: int, *, root_seed: int) -> dict:
    """Independent maximum-length draws, without metadata or study-seed derivation."""
    dgp = plan["dgp"]
    classes, channels = plan["n_classes"], plan["n_channels"]
    blocks = plan["support_blocks"] + plan["query_blocks"]
    samples, harmonics = plan["max_samples"], dgp["harmonics"]
    if channels % 2:
        raise ValueError("The frozen balanced channel-state DGP requires an even channel count.")
    rng = lambda name: component_rng(root_seed, participant, name)
    common = rng("common").uniform(*dgp["common_topography_uniform"], channels)
    spatial = common + rng("spatial").normal(
        0, dgp["class_topography_perturbation_sd"], (classes, channels)
    )
    spatial /= np.linalg.norm(spatial, axis=1, keepdims=True)
    phase = rng("phase").uniform(0, 2 * np.pi, (classes, harmonics))
    jitter = rng("jitter").normal(
        0, dgp["trial_harmonic_phase_jitter_sd"], (blocks, classes, harmonics)
    )
    frequencies = dgp["frequency_start"] + dgp["frequency_step"] * np.arange(classes)
    h = np.arange(1, harmonics + 1)
    time = np.arange(samples) / plan["sfreq"]
    angle = (
        2
        * np.pi
        * frequencies[None, :, None, None]
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
    signs = rng("signs").permutation(np.repeat([-1.0, 1.0], channels // 2))
    states = rng("state").choice([-1.0, 1.0], blocks)
    order_rng = rng("order")
    labels = np.stack([order_rng.permutation(classes) for _ in range(blocks)]).astype(np.int64)
    result = {"clean": clean, "noise": noise, "signs": signs, "states": states, "labels": labels}
    digest = hashlib.sha256()
    for key in ("clean", "noise", "signs", "states", "labels"):
        digest.update(np.ascontiguousarray(result[key]).tobytes())
    result["base_sha256"] = digest.hexdigest()
    return result


def construct_cell(base: dict, cell: dict, dgp: dict) -> np.ndarray:
    samples, multiplier = cell["n_samples"], cell["noise_multiplier"]
    clean, noise = base["clean"][..., :samples], base["noise"][..., :samples]
    if cell["family"] == "stable":
        eeg = clean + multiplier * noise
    elif cell["family"] == "drift":
        state = base["states"][:, None] * base["signs"][None, :]
        eeg = (
            clean * np.exp(dgp["signal_log_gain"] * state)[:, None, :, None]
            + multiplier * noise * np.exp(dgp["noise_log_gain"] * state)[:, None, :, None]
        )
    else:
        raise ValueError("Unknown metadata-free family.")
    return np.take_along_axis(eeg, base["labels"][..., None, None], axis=1)


def build_workspace(plan: dict, filterbank: dict, samples: int) -> dict:
    dgp, op = plan["dgp"], plan["operator"]
    frequencies = dgp["frequency_start"] + dgp["frequency_step"] * np.arange(plan["n_classes"])
    references = make_reference_signals(
        frequencies, plan["sfreq"], samples, dgp["harmonics"]
    ).astype(np.float64)
    return {
        "plan": plan,
        "filterbank": filterbank,
        "samples": samples,
        "cca_cache": numeric.reference_cache(
            frequencies,
            plan["sfreq"],
            samples,
            filterbank["n_harmonics"],
            filterbank["regularization"],
        ),
        "class_bases": tuple(
            numeric.projection_basis(r, op["projection_svd_relative_cutoff"]) for r in references
        ),
    }


def decode_scores(eeg: np.ndarray, support_labels: np.ndarray, workspace: dict) -> dict:
    """No query labels or metadata enter decoding; crops precede every filter call."""
    plan, op = workspace["plan"], workspace["plan"]["operator"]
    blocks, classes, channels, samples = eeg.shape
    support_blocks = plan["support_blocks"]
    if (
        support_labels.shape != (support_blocks, classes)
        or samples != workspace["samples"]
        or blocks != support_blocks + plan["query_blocks"]
        or not np.isfinite(eeg).all()
    ):
        raise ValueError("Finite cropped EEG and support labels must match the workspace.")
    centered = eeg - eeg.mean(axis=-1, keepdims=True)
    filtered, parameters = apply_filterbank(
        centered.reshape(-1, channels, samples),
        sfreq=plan["sfreq"],
        filterbank=workspace["filterbank"],
    )
    weights = np.asarray(parameters["weights"])
    filtered = filtered.reshape(len(weights), blocks, classes, channels, samples)
    query = filtered[:, support_blocks:].reshape(len(weights), -1, channels, samples)
    original_query, _ = apply_filterbank(
        eeg[support_blocks:].reshape(-1, channels, samples),
        sfreq=plan["sfreq"],
        filterbank=workspace["filterbank"],
    )
    cca = np.stack(
        [
            numeric.cached_cca_scores(x, workspace["cca_cache"], parameters["regularization"])
            for x in original_query
        ]
    )
    a0 = np.einsum("f,fqc->qc", weights, cca**2) / weights.sum()
    block = np.stack(
        [
            numeric._waveform_scores(query, filtered[:, b], support_labels[b], weights, classes)
            for b in range(support_blocks)
        ],
        axis=1,
    )
    pooled = [block[:, 0].copy()]
    for budget in POSITIVE_BUDGETS[1:]:
        pooled.append(
            numeric._waveform_scores(
                query,
                filtered[:, :budget].reshape(len(weights), -1, channels, samples),
                support_labels[:budget].reshape(-1),
                weights,
                classes,
            )
        )
    reliability = np.asarray(
        [
            numeric.block_reliability(
                eeg[b], support_labels[b], workspace["class_bases"], op["variance_floor"]
            )
            for b in range(support_blocks)
        ]
    )
    return {
        "a0_scores": a0,
        "block_scores": block,
        "pooled_scores": np.stack(pooled),
        "block_reliability": reliability,
    }


def initialize_worker(plan: dict, filterbank: dict) -> None:
    global _WORKER
    _WORKER = (plan, {n: build_workspace(plan, filterbank, n) for n in plan["sample_counts"]})


def score_participant(participant: int, root_seed: int) -> dict:
    if _WORKER is None:
        raise RuntimeError("Study worker not initialized.")
    plan, workspaces = _WORKER
    base = generate_base(plan, participant, root_seed=root_seed)
    scored, hashes = [], []
    for cell in cells(plan):
        eeg = construct_cell(base, cell, plan["dgp"])
        hashes.append(array_sha(eeg))
        scored.append(
            decode_scores(
                eeg, base["labels"][: plan["support_blocks"]], workspaces[cell["n_samples"]]
            )
        )
    result = {key: np.stack([record[key] for record in scored]) for key in scored[0]}
    result.update(
        participant_id=participant,
        labels=base["labels"],
        eeg_sha256=np.asarray(hashes),
        base_sha256=base["base_sha256"],
    )
    return result


def collect_scores(plan: dict, filterbank: dict, split: str, root_seed: int, workers: int) -> dict:
    if split not in ("source", "evaluation"):
        raise ValueError("Only source and evaluation partitions exist.")
    count = plan[f"{split}_participants"]
    results = []
    with ProcessPoolExecutor(
        max_workers=workers, initializer=initialize_worker, initargs=(plan, filterbank)
    ) as executor:
        pending = [executor.submit(score_participant, p, root_seed) for p in range(count)]
        for future in as_completed(pending):
            results.append(future.result())
            print(
                json.dumps({"phase": split, "completed": len(results), "total": count}), flush=True
            )
    results.sort(key=lambda r: r["participant_id"])
    cache = {
        key: np.stack([r[key] for r in results])
        for key in (
            "a0_scores",
            "block_scores",
            "pooled_scores",
            "block_reliability",
            "labels",
            "eeg_sha256",
            "base_sha256",
        )
    }
    cache.update(
        schema=np.asarray(CACHE_SCHEMA),
        split=np.asarray(split),
        participant_ids=np.asarray([r["participant_id"] for r in results], dtype=np.int64),
        cell_ids=np.asarray([c["cell_id"] for c in cells(plan)]),
    )
    return cache


def reliability_weights(reliability: np.ndarray, budget: int, floor: float) -> np.ndarray:
    current = floor + np.asarray(reliability)[..., :budget]
    if budget < 1 or not math.isfinite(floor) or floor <= 0 or not np.isfinite(current).all():
        raise ValueError("Reliability requires a positive floor and budget.")
    if np.any(current <= 0):
        raise ValueError("Reliability weights must be positive.")
    return current / current.sum(axis=-1, keepdims=True)


def block_posterior(scores: np.ndarray, weights: np.ndarray, temperature: float) -> np.ndarray:
    if (
        not math.isfinite(temperature)
        or temperature <= 0
        or not np.isfinite(scores).all()
        or not np.isfinite(weights).all()
        or np.any(weights < 0)
        or not np.allclose(weights.sum(axis=-1), 1, rtol=0, atol=1e-12)
    ):
        raise ValueError(
            "Block posterior requires finite scores, positive temperature and weights."
        )
    return np.sum(numeric.softmax(scores, temperature) * weights[..., None, :, None], axis=-2)


def mean_nll(probability: np.ndarray, query_labels: np.ndarray) -> float:
    labels = np.broadcast_to(query_labels[:, None, :, None], (*probability.shape[:-1], 1))
    truth = np.take_along_axis(probability, labels, axis=-1)[..., 0]
    return float(-np.mean(np.log(np.maximum(truth, 1e-300))))


def fit_temperatures(source: dict, plan: dict) -> dict:
    """Fit source-only actual decoder NLL, equally weighting people/cells/queries."""
    if str(source["split"]) != "source":
        raise ValueError("Temperature fitting accepts source scores only.")
    grid = np.logspace(-4, 1, 31, dtype=np.float64)
    labels = source["labels"][:, plan["support_blocks"] :].reshape(len(source["labels"]), -1)
    fits = {}
    for parameter in plan["temperature_fit"]["parameters"]:
        objective = []
        for temperature in grid:
            if parameter == "A0":
                probability = numeric.softmax(source["a0_scores"], temperature)
            else:
                budget = int(parameter.rsplit("k", 1)[1])
                if parameter.startswith("pooled"):
                    scores = source["pooled_scores"][:, :, POSITIVE_BUDGETS.index(budget)]
                    probability = numeric.softmax(scores, temperature)
                else:
                    pi = reliability_weights(
                        source["block_reliability"],
                        budget,
                        plan["operator"]["support_reliability_floor"],
                    )
                    probability = block_posterior(
                        source["block_scores"][..., :budget, :], pi, temperature
                    )
            objective.append(mean_nll(probability, labels))
        selected = int(np.argmin(objective))
        fits[parameter] = {
            "selected_index": selected,
            "temperature": float(grid[selected]),
            "objective_values": objective,
        }
    return {"temperature_grid": grid.tolist(), "fits": fits}


def source_difficulty(source: dict, plan: dict) -> dict:
    labels = source["labels"][:, plan["support_blocks"] :].reshape(len(source["labels"]), -1)
    accuracy = np.mean(source["a0_scores"].argmax(axis=-1) == labels[:, None], axis=-1)
    criterion = plan["source_difficulty"]
    low, high = criterion["mean_BA_range"]
    rows, selected = [], []
    for index, cell in enumerate(cells(plan)):
        values = accuracy[:, index]
        fraction = float(np.mean(values < criterion["target_BA"]))
        eligible = (
            cell["family"] == criterion["family"]
            and low <= values.mean() <= high
            and fraction >= criterion["minimum_fraction_below_target"]
        )
        rows.append(
            {
                "cell_id": cell["cell_id"],
                "family": cell["family"],
                "participant_BA": values.tolist(),
                "mean_BA": float(values.mean()),
                "fraction_below_target": fraction,
                "selected": bool(eligible),
            }
        )
        if eligible:
            selected.append(cell["cell_id"])
    return {"source_difficulty_rows": rows, "selected_stable_cells": selected}


def predict_methods(scores: dict, fits: dict, budget: int, plan: dict) -> dict:
    op = plan["operator"]
    p0 = {
        "raw": numeric.softmax(scores["a0_scores"], op["raw_temperature"]),
        "cal": numeric.softmax(scores["a0_scores"], fits["A0"]["temperature"]),
    }
    result = {"A0_raw": p0["raw"], "A0_cal": p0["cal"]}
    for suffix in ("raw", "cal"):
        if budget == 0:
            pooled = block = p0[suffix]
        else:
            index = POSITIVE_BUDGETS.index(budget)
            tp = (
                op["raw_temperature"]
                if suffix == "raw"
                else fits[f"pooled_k{budget}"]["temperature"]
            )
            pooled = numeric.softmax(scores["pooled_scores"][index], tp)
            if budget == 1:
                block = pooled
            else:
                tb = (
                    op["raw_temperature"]
                    if suffix == "raw"
                    else fits[f"block_k{budget}"]["temperature"]
                )
                pi = reliability_weights(
                    scores["block_reliability"], budget, op["support_reliability_floor"]
                )
                block = block_posterior(scores["block_scores"][:, :budget], pi, tb)
        result[f"pooled_{suffix}"] = pooled
        result[f"block_{suffix}"] = block
        for name, support in (("pooled", pooled), ("block", block)):
            result[f"fusion_{name}_{suffix}"] = (
                p0[suffix].copy()
                if budget == 0
                else (1 - op["fusion_strength"]) * p0[suffix] + op["fusion_strength"] * support
            )
    return result


def evaluate_rows(cache: dict, freeze: dict, plan: dict) -> list[dict]:
    rows = []
    for p, participant in enumerate(cache["participant_ids"]):
        labels = cache["labels"][p, plan["support_blocks"] :].reshape(-1)
        for c, cell in enumerate(cells(plan)):
            scores = {
                key: cache[key][p, c]
                for key in ("a0_scores", "block_scores", "pooled_scores", "block_reliability")
            }
            for budget in plan["budgets"]:
                predictions = predict_methods(scores, freeze["fits"], budget, plan)
                for method in METHODS:
                    posterior = predictions[method]
                    baseline = predictions["A0_cal" if method.endswith("_cal") else "A0_raw"]
                    metric = numeric.posterior_metrics(posterior, labels, baseline, baseline)
                    metric.pop("prediction_flips_vs_AQ")
                    rows.append(
                        {
                            "participant": int(participant),
                            **cell,
                            "method": method,
                            "budget": budget,
                            **metric,
                            "posterior_sha256": array_sha(posterior),
                        }
                    )
    return rows


def summarize_results(rows: list[dict], freeze: dict, plan: dict) -> dict:
    participants = sorted({r["participant"] for r in rows})
    lookup = {(r["participant"], r["cell_id"], r["method"], r["budget"]): r for r in rows}

    def values(cell: str, method: str, budget: int | str, metric: str = "balanced_accuracy"):
        if budget == "eauc":
            return sum(
                w * values(cell, method, k, metric) for k, w in ((0, 1 / 6), (1, 0.5), (3, 1 / 3))
            )
        return np.asarray([lookup[p, cell, method, budget][metric] for p in participants])

    selected = freeze["selected_stable_cells"]
    primary = {
        "selected_stable_cells": selected,
        "eauc_gain": None,
        "k1_gain": None,
        "difficulty_fraction_below_target": None,
        "screen_components": {},
    }
    if selected:
        gains = {
            budget: np.mean(
                [
                    values(cell, "fusion_pooled_cal", budget) - values(cell, "A0_cal", budget)
                    for cell in selected
                ],
                axis=0,
            )
            for budget in ("eauc", 1)
        }
        primary["eauc_gain"] = numeric.paired_summary(gains["eauc"])
        primary["k1_gain"] = numeric.paired_summary(gains[1])
        fraction = float(np.mean([values(cell, "A0_cal", 0) < 0.8 for cell in selected]))
        primary["difficulty_fraction_below_target"] = fraction
        screen = {
            "difficulty_replicated": fraction >= 0.5,
            "eauc_mean": float(gains["eauc"].mean()) >= plan["reporting"]["effect_threshold"],
            "eauc_LCB": primary["eauc_gain"]["one_sided_95_LCB"] > 0,
            "k1_nonnegative": float(gains[1].mean()) >= 0,
        }
        primary["screen_components"] = screen
        status = (
            "DIFFICULTY_NOT_CONFIRMED"
            if not screen["difficulty_replicated"]
            else "AQ_READY_FOR_CONTEXT_COMPARATOR_STUDY"
            if all(screen.values())
            else "AQ_NOT_ESTABLISHED"
        )
    else:
        status = "NO_SOURCE_INFORMATIVE_CELLS"
    aggregates, attainment, diagnostics = [], [], []
    scalar_metrics = (
        "balanced_accuracy",
        "correct_log_probability",
        "true_class_margin",
        "prediction_flips_vs_A0",
        "correct_count",
        "query_count",
    )
    for cell in cells(plan):
        cid = cell["cell_id"]
        for method in METHODS:
            baseline = "A0_cal" if method.endswith("_cal") else "A0_raw"
            curve = {str(k): values(cid, method, k).tolist() for k in plan["budgets"]}
            for budget in plan["budgets"]:
                aggregates.append(
                    {
                        **cell,
                        "method": method,
                        "budget": budget,
                        "mean_metrics": {
                            key: float(values(cid, method, budget, key).mean())
                            for key in scalar_metrics
                        },
                        "fraction_harmed_vs_A0": float(
                            np.mean(values(cid, method, budget) < values(cid, baseline, budget))
                        ),
                    }
                )
            first = [
                next((k for k in plan["budgets"] if curve[str(k)][i] >= 0.8), ">5")
                for i in range(len(participants))
            ]
            attainment.append(
                {
                    **cell,
                    "method": method,
                    "participant_ids": participants,
                    "participant_accuracy_curve": curve,
                    "first_observed_budget": first,
                    "labeled_trials": [
                        plan["n_classes"] * k if isinstance(k, int) else None for k in first
                    ],
                    "analyzed_EEG_seconds": [
                        plan["n_classes"] * k * cell["n_samples"] / plan["sfreq"]
                        if isinstance(k, int)
                        else None
                        for k in first
                    ],
                }
            )
            diagnostics.append(
                {
                    "cell_id": cid,
                    "method": method,
                    "contrast": "k3_minus_k1",
                    **numeric.paired_summary(values(cid, method, 3) - values(cid, method, 1)),
                }
            )
        for name in ("pooled", "block", "fusion_pooled", "fusion_block", "A0"):
            for budget in POSITIVE_BUDGETS:
                for metric in ("balanced_accuracy", "correct_log_probability"):
                    diagnostics.append(
                        {
                            "cell_id": cid,
                            "method": name,
                            "budget": budget,
                            "contrast": f"cal_minus_raw_{metric}",
                            **numeric.paired_summary(
                                values(cid, f"{name}_cal", budget, metric)
                                - values(cid, f"{name}_raw", budget, metric)
                            ),
                        }
                    )
    return {
        "status": status,
        "human_unlock": False,
        "selected_stable_cells": selected,
        "primary": primary,
        "aggregate_rows": aggregates,
        "calibration_attainment": attainment,
        "diagnostics": diagnostics,
    }


def file_sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_cache_exclusive(path: Path, cache: dict) -> str:
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


def derive_study_seed(namespace: str, split: str) -> int:
    if split not in ("source", "evaluation"):
        raise ValueError("Unknown RNG partition.")
    return int.from_bytes(hashlib.sha256(f"{namespace}/{split}".encode()).digest()[:8], "big")


def execute_phases(
    plan: dict, filterbank: dict, output: Path, start: dict, start_sha: str, workers: int
) -> dict:
    """Called only after durable start; source-freeze is durable before eval seed derivation."""
    source_seed = derive_study_seed(plan["seed_namespace"], "source")
    source = collect_scores(plan, filterbank, "source", source_seed, workers)
    source_sha = write_cache_exclusive(output / "source-scores.npz", source)
    freeze = {
        "schema": "cfeg.v4-aq-study.source-freeze.v1",
        "study_id": plan["study_id"],
        "source_commit": start["source_commit"],
        "plan_sha256": start["plan_sha256"],
        "start_sha256": start_sha,
        "source_scores_sha256": source_sha,
        "source_root_seed": source_seed,
        **fit_temperatures(source, plan),
        **source_difficulty(source, plan),
    }
    freeze_sha = numeric.write_json_exclusive(output / "source-freeze.json", freeze)
    evaluation_start = {
        "schema": "cfeg.v4-aq-study.evaluation-start.v1",
        "study_id": plan["study_id"],
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source_commit": start["source_commit"],
        "plan_sha256": start["plan_sha256"],
        "source_freeze_sha256": freeze_sha,
        "source_scores_sha256": source_sha,
    }
    evaluation_start_sha = numeric.write_json_exclusive(
        output / "evaluation-start.json", evaluation_start
    )
    evaluation_seed = derive_study_seed(plan["seed_namespace"], "evaluation")
    evaluation = collect_scores(plan, filterbank, "evaluation", evaluation_seed, workers)
    evaluation_sha = write_cache_exclusive(output / "evaluation-scores.npz", evaluation)
    rows = evaluate_rows(evaluation, freeze, plan)
    summary = summarize_results(rows, freeze, plan)
    result = {
        "schema": "cfeg.v4-aq-study.result.v1",
        "study_id": plan["study_id"],
        "evidence_role": plan["evidence_role"],
        "source_commit": start["source_commit"],
        "plan_sha256": start["plan_sha256"],
        "start_sha256": start_sha,
        "source_scores_sha256": source_sha,
        "source_freeze_sha256": freeze_sha,
        "evaluation_start_sha256": evaluation_start_sha,
        "evaluation_scores_sha256": evaluation_sha,
        "evaluation_root_seed": evaluation_seed,
        "rows": rows,
        "summary": summary,
    }
    result_sha = numeric.write_json_exclusive(output / "result.json", result)
    return {
        "status": summary["status"],
        "metric_rows": len(rows),
        "result_sha256": result_sha,
        "source_selected_stable_cells": freeze["selected_stable_cells"],
        "human_unlock": False,
    }


def run_study(plan_path: Path, output: Path, workers: int) -> dict:
    if not plan_path.is_absolute() or not output.is_absolute():
        raise ValueError("Plan and output paths must be absolute.")
    plan_path, output = plan_path.resolve(), output.resolve()
    root = Path(__file__).resolve().parents[3]
    if (
        plan_path != root / "configs/analysis/metadata_calibration_v4_aq_study.json"
        or file_sha(plan_path) != PLAN_SHA256
    ):
        raise ValueError("Exact frozen AQ study plan required.")
    plan = json.loads(plan_path.read_text())
    if str(output) != plan["execution"]["output_root"] or workers != plan["execution"]["workers"]:
        raise ValueError("Output root and worker count must match the frozen plan.")
    if (output / "start.json").exists():
        raise FileExistsError("Existing start consumes this attempt; no resume or retry.")

    def git(*args):
        return subprocess.check_output(["git", "-C", str(root), *args], text=True).strip()

    if git("status", "--porcelain", "--untracked-files=all"):
        raise RuntimeError("Clean committed source is required.")
    thread_keys = ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS")
    if any(os.environ.get(key) != "1" for key in thread_keys):
        raise RuntimeError("All declared BLAS thread environment variables must be 1.")
    filterbank_path = root / plan["filterbank_config"]
    if file_sha(filterbank_path) != plan["filterbank_sha256"]:
        raise ValueError("Frozen filterbank hash differs.")
    if file_sha(Path(numeric.__file__)) != plan["pure_helper_module_sha256"]:
        raise ValueError("Closed V4 pure-helper module hash differs.")
    filterbank = yaml.safe_load(filterbank_path.read_text())
    output.mkdir(parents=True, exist_ok=True)
    if any(output.iterdir()):
        raise FileExistsError("Output directory must be empty.")
    start = {
        "schema": "cfeg.v4-aq-study.start.v1",
        "study_id": plan["study_id"],
        "evidence_role": plan["evidence_role"],
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "pid": os.getpid(),
        "source_commit": git("rev-parse", "HEAD"),
        "source_tree": git("rev-parse", "HEAD^{tree}"),
        "plan_sha256": PLAN_SHA256,
        "filterbank_sha256": plan["filterbank_sha256"],
        "pure_helper_module_sha256": plan["pure_helper_module_sha256"],
        "rng_components": COMPONENTS,
        "seed_namespace": plan["seed_namespace"],
        "workers": workers,
        "human_data_access": False,
        "runtime": {
            "python": sys.version,
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "platform": platform.platform(),
            "machine": platform.machine(),
            "blas_threads": {key: os.environ[key] for key in thread_keys},
        },
    }
    start_sha = numeric.write_json_exclusive(output / "start.json", start)
    return execute_phases(plan, filterbank, output, start, start_sha, workers)
