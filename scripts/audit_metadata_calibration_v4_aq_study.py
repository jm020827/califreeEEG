#!/usr/bin/env python3
"""Independently audit saved AQ scores; no DGP imports, EEG generation or fitting writes."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import stat
import subprocess
from pathlib import Path

import numpy as np
from scipy import stats


def require(condition, message):
    if not condition:
        raise ValueError(message)


def close(actual, expected, label):
    require(np.allclose(actual, expected, rtol=0, atol=1e-11), f"Mismatch: {label}")


def softmax(scores, temperature):
    scaled = scores / temperature
    exp = np.exp(scaled - np.max(scaled, axis=-1, keepdims=True))
    return exp / exp.sum(axis=-1, keepdims=True)


def query_labels(cache, plan):
    return cache["labels"][:, plan["support_blocks"] :].reshape(len(cache["labels"]), -1)


def posterior(cache, kind, budget, temperature, plan):
    if kind == "A0":
        return softmax(cache["a0_scores"], temperature)
    if kind == "pooled":
        index = [k for k in plan["budgets"] if k].index(budget)
        return softmax(cache["pooled_scores"][:, :, index], temperature)
    weights = (
        cache["block_reliability"][:, :, :budget] + plan["operator"]["support_reliability_floor"]
    )
    weights = weights / weights.sum(axis=-1, keepdims=True)
    blocks = softmax(cache["block_scores"][:, :, :, :budget], temperature)
    return np.einsum("plb,plqbc->plqc", weights, blocks)


def nll(probability, labels):
    truth = np.take_along_axis(probability, labels[:, None, :, None], axis=-1)[..., 0]
    return float(-np.log(np.maximum(truth, 1e-300)).mean())


def paired(vector):
    vector = np.asarray(vector)
    n = len(vector)
    mean = vector.mean()
    se = vector.std(ddof=1) / np.sqrt(n)
    c90, c95 = stats.t.ppf([0.95, 0.975], n - 1)
    return {
        "n": n,
        "mean": mean,
        "standard_error": se,
        "one_sided_95_LCB": mean - c90 * se,
        "two_sided_95_CI": [mean - c95 * se, mean + c95 * se],
        "two_sided_90_CI": [mean - c90 * se, mean + c90 * se],
        "participant_values": vector.tolist(),
    }


def cell_records(plan):
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


def verify_cache(cache, plan, split):
    people = plan[f"{split}_participants"]
    cells = len(cell_records(plan))
    classes, support = plan["n_classes"], plan["support_blocks"]
    queries = plan["query_blocks"] * classes
    shapes = {
        "a0_scores": (people, cells, queries, classes),
        "block_scores": (people, cells, queries, support, classes),
        "pooled_scores": (people, cells, len(plan["budgets"]) - 1, queries, classes),
        "block_reliability": (people, cells, support),
        "labels": (people, support + plan["query_blocks"], classes),
        "cell_ids": (cells,),
        "participant_ids": (people,),
        "eeg_sha256": (people, cells),
        "base_sha256": (people,),
    }
    for name, shape in shapes.items():
        require(cache[name].shape == shape, f"Cache shape {split}/{name}")
        if name.endswith("scores") or name == "block_reliability":
            require(np.isfinite(cache[name]).all(), f"Finite {name}")
    require(cache["participant_ids"].tolist() == list(range(people)), "Participant order")
    require(cache["cell_ids"].tolist() == [c["cell_id"] for c in cell_records(plan)], "Cell order")
    require(str(cache["split"]) == split, "Cache split")
    require(str(cache["schema"]) == "cfeg.v4-aq-study.scores.v1", "Cache schema")
    for name in ("eeg_sha256", "base_sha256"):
        require(
            all(re.fullmatch(r"[0-9a-f]{64}", str(v)) for v in cache[name].flat), "Cache SHA format"
        )
    require(len(set(cache["base_sha256"])) == people, "Unique cohort participants")
    require(
        np.array_equal(
            np.sort(cache["labels"], axis=-1),
            np.broadcast_to(np.arange(classes), cache["labels"].shape),
        ),
        "Label balance",
    )
    require(
        np.all((cache["block_reliability"] >= 0) & (cache["block_reliability"] <= 1)), "Reliability"
    )
    close(cache["pooled_scores"][:, :, 0], cache["block_scores"][:, :, :, 0], "k1 score identity")


def verify_source(cache, freeze, plan):
    verify_cache(cache, plan, "source")
    grid = np.logspace(-4, 1, 31)
    close(freeze["temperature_grid"], grid, "Temperature grid")
    labels = query_labels(cache, plan)
    fits = freeze["fits"]
    require(
        set(fits) == set(plan["temperature_fit"]["parameters"]), "Temperature parameter inventory"
    )
    for name, fit in fits.items():
        if name == "A0":
            kind, budget = "A0", 0
        else:
            kind, k = name.split("_k")
            budget = int(k)
        objectives = [nll(posterior(cache, kind, budget, t, plan), labels) for t in grid]
        close(fit["objective_values"], objectives, f"NLL grid {name}")
        # Exact tie policy is relative to the producer's serialized float64 reduction.
        # Independent einsum can differ by an ulp from multiply-then-sum.
        index = int(np.argmin(fit["objective_values"]))
        require(fit["selected_index"] == index, f"Source-only minimum {name}")
        require(objectives[index] <= min(objectives) + 1e-11, f"Independent NLL minimum {name}")
        close(fit["temperature"], grid[index], f"Temperature {name}")
    source_rows = {r["cell_id"]: r for r in freeze["source_difficulty_rows"]}
    require(len(source_rows) == len(freeze["source_difficulty_rows"]), "Duplicate difficulty row")
    require(set(source_rows) == set(cache["cell_ids"].tolist()), "Difficulty row inventory")
    selected = []
    for index, cell in enumerate(cell_records(plan)):
        row = source_rows[cell["cell_id"]]
        require(row["family"] == cell["family"], "Source family descriptor")
        predictions = cache["a0_scores"][:, index].argmax(axis=-1)
        ba = np.mean(predictions == labels, axis=-1)  # exact class balance independently checked
        close(row["participant_BA"], ba, "Source A0-only BA")
        close(row["mean_BA"], ba.mean(), "Source mean BA")
        fraction = np.mean(ba < plan["source_difficulty"]["target_BA"])
        close(row["fraction_below_target"], fraction, "Source target fraction")
        low, high = plan["source_difficulty"]["mean_BA_range"]
        choose = (
            cell["family"] == "stable"
            and low <= ba.mean() <= high
            and fraction >= plan["source_difficulty"]["minimum_fraction_below_target"]
        )
        require(row["selected"] == choose, "A0-only condition selection")
        if choose:
            selected.append(cell["cell_id"])
    require(freeze["selected_stable_cells"] == selected, "Frozen source selection")


def probabilities(cache, freeze, plan, budget):
    raw_t = plan["operator"]["raw_temperature"]
    fits = freeze["fits"]
    output = {}
    for suffix, t0 in (("raw", raw_t), ("cal", fits["A0"]["temperature"])):
        a0 = posterior(cache, "A0", 0, t0, plan)
        output[f"A0_{suffix}"] = a0
        for kind in ("pooled", "block"):
            if budget:
                parameter = "pooled_k1" if kind == "block" and budget == 1 else f"{kind}_k{budget}"
                temperature = raw_t if suffix == "raw" else fits[parameter]["temperature"]
                support = posterior(cache, kind, budget, temperature, plan)
                strength = plan["operator"]["fusion_strength"]
                fusion = (1 - strength) * a0 + strength * support
            else:
                support = fusion = a0
            output[f"{kind}_{suffix}"] = support
            output[f"fusion_{kind}_{suffix}"] = fusion
    return output


def verify_evaluation(cache, freeze, result, plan):
    verify_cache(cache, plan, "evaluation")
    people = plan["evaluation_participants"]
    cells, methods, budgets = cell_records(plan), plan["operator"]["methods"], plan["budgets"]
    labels = query_labels(cache, plan)
    lookup = {(r["participant"], r["cell_id"], r["method"], r["budget"]): r for r in result["rows"]}
    expected = {
        (p, c["cell_id"], m, k)
        for p in range(people)
        for c in cells
        for m in methods
        for k in budgets
    }
    require(
        len(lookup) == len(result["rows"]) and set(lookup) == expected, "Evaluation row inventory"
    )
    accuracy = {}
    for budget in budgets:
        all_probs = probabilities(cache, freeze, plan, budget)
        require(
            np.array_equal(all_probs["A0_raw"].argmax(-1), all_probs["A0_cal"].argmax(-1)),
            "A0 positive-temperature class order",
        )
        if budget:
            require(
                np.array_equal(
                    all_probs["pooled_raw"].argmax(-1), all_probs["pooled_cal"].argmax(-1)
                ),
                "Pooled positive-temperature class order",
            )
        for method, probability in all_probs.items():
            anchor = all_probs["A0_cal" if method.endswith("_cal") else "A0_raw"]
            for p in range(people):
                for i, cell in enumerate(cells):
                    key = p, cell["cell_id"], method, budget
                    row = lookup[key]
                    require(
                        re.fullmatch(r"[0-9a-f]{64}", row["posterior_sha256"]),
                        "Posterior SHA format",
                    )
                    require(all(row[k] == v for k, v in cell.items()), "Metric cell descriptor")
                    if budget == 0:
                        anchor_method = "A0_cal" if method.endswith("_cal") else "A0_raw"
                        require(
                            row["posterior_sha256"]
                            == lookup[p, cell["cell_id"], anchor_method, budget][
                                "posterior_sha256"
                            ],
                            "k0 posterior exact alias",
                        )
                    if budget == 1 and "block" in method:
                        require(
                            row["posterior_sha256"]
                            == lookup[
                                p, cell["cell_id"], method.replace("block", "pooled"), budget
                            ]["posterior_sha256"],
                            "k1 posterior exact alias",
                        )
                    observed, truth_labels = probability[p, i], labels[p]
                    prediction = observed.argmax(axis=-1)
                    counts = np.bincount(truth_labels, minlength=plan["n_classes"])
                    correct = np.bincount(
                        truth_labels,
                        weights=(prediction == truth_labels),
                        minlength=plan["n_classes"],
                    )
                    truth = observed[np.arange(len(truth_labels)), truth_labels]
                    other = observed.copy()
                    other[np.arange(len(truth_labels)), truth_labels] = -np.inf
                    metrics = {
                        "balanced_accuracy": np.mean(correct / counts),
                        "correct_log_probability": np.log(np.maximum(truth, 1e-300)).mean(),
                        "true_class_margin": (truth - other.max(axis=-1)).mean(),
                        "prediction_flips_vs_A0": np.mean(
                            prediction != anchor[p, i].argmax(axis=-1)
                        ),
                        "correct_count": correct.sum(),
                        "query_count": len(truth_labels),
                        "class_correct_counts": correct,
                        "class_query_counts": counts,
                    }
                    for metric, value in metrics.items():
                        close(row[metric], value, f"{key}/{metric}")
                    accuracy[key] = metrics["balanced_accuracy"]
    summary = result["summary"]
    selected = freeze["selected_stable_cells"]
    require(summary["selected_stable_cells"] == selected, "Evaluation cannot reselect cells")
    aggregate_keys = [(r["cell_id"], r["method"], r["budget"]) for r in summary["aggregate_rows"]]
    expected_aggregates = {(c["cell_id"], m, k) for c in cells for m in methods for k in budgets}
    require(
        len(aggregate_keys) == len(set(aggregate_keys))
        and set(aggregate_keys) == expected_aggregates,
        "Aggregate inventory",
    )
    for row in summary["aggregate_rows"]:
        cell, method, budget = row["cell_id"], row["method"], row["budget"]
        require(
            all(
                row[key] == value
                for key, value in next(c for c in cells if c["cell_id"] == cell).items()
            ),
            "Aggregate cell descriptor",
        )
        require(
            set(row["mean_metrics"])
            == {
                "balanced_accuracy",
                "correct_log_probability",
                "true_class_margin",
                "prediction_flips_vs_A0",
                "correct_count",
                "query_count",
            },
            "Aggregate metric inventory",
        )
        group = [lookup[p, cell, method, budget] for p in range(people)]
        for metric, value in row["mean_metrics"].items():
            close(value, np.mean([r[metric] for r in group], axis=0), f"Aggregate {metric}")
        anchor = "A0_cal" if method.endswith("_cal") else "A0_raw"
        harm = np.mean(
            [
                accuracy[p, cell, method, budget] < accuracy[p, cell, anchor, budget]
                for p in range(people)
            ]
        )
        close(row["fraction_harmed_vs_A0"], harm, "Participant harm")
    verify_attainment_and_diagnostics(summary, lookup, plan)
    primary = summary["primary"]
    require(primary["selected_stable_cells"] == selected, "Primary selection")
    if not selected:
        require(
            all(
                primary[k] is None
                for k in ("eauc_gain", "k1_gain", "difficulty_fraction_below_target")
            )
            and primary["screen_components"] == {},
            "Empty-source primary must be absent",
        )
        status = "NO_SOURCE_INFORMATIVE_CELLS"
    else:
        first, second = plan["operator"]["primary_method"], plan["operator"]["primary_comparator"]
        delta = lambda p, c, k: accuracy[p, c, first, k] - accuracy[p, c, second, k]
        eauc = np.asarray(
            [
                np.mean(
                    [
                        (delta(p, c, 0) + 3 * delta(p, c, 1) + 2 * delta(p, c, 3)) / 6
                        for c in selected
                    ]
                )
                for p in range(people)
            ]
        )
        k1 = np.asarray([np.mean([delta(p, c, 1) for c in selected]) for p in range(people)])
        for name, vector in (("eauc_gain", eauc), ("k1_gain", k1)):
            for field, value in paired(vector).items():
                close(primary[name][field], value, f"Primary {name}/{field}")
        fraction = np.mean(
            [accuracy[p, c, second, 0] < 0.8 for p in range(people) for c in selected]
        )
        close(primary["difficulty_fraction_below_target"], fraction, "Difficulty replication")
        utility = (
            eauc.mean() >= plan["reporting"]["effect_threshold"]
            and paired(eauc)["one_sided_95_LCB"] > 0
            and k1.mean() >= 0
        )
        require(
            primary["screen_components"]
            == {
                "difficulty_replicated": bool(fraction >= 0.5),
                "eauc_mean": bool(eauc.mean() >= plan["reporting"]["effect_threshold"]),
                "eauc_LCB": bool(paired(eauc)["one_sided_95_LCB"] > 0),
                "k1_nonnegative": bool(k1.mean() >= 0),
            },
            "Primary screen components",
        )
        status = (
            "DIFFICULTY_NOT_CONFIRMED"
            if fraction < 0.5
            else "AQ_NOT_ESTABLISHED"
            if not utility
            else "AQ_READY_FOR_CONTEXT_COMPARATOR_STUDY"
        )
    require(summary["status"] == status and summary["human_unlock"] is False, "Status")
    return {
        "audit": "PASS",
        "status": status,
        "source_participants": plan["source_participants"],
        "evaluation_participants": people,
        "cells": len(cells),
        "metric_rows_checked": len(lookup),
        "temperature_objectives_checked": 6 * 31,
        "attainment_rows_checked": len(summary["calibration_attainment"]),
        "diagnostics_checked": len(summary["diagnostics"]),
        "selected_stable_cells": selected,
        "human_unlock": False,
    }


def verify_attainment_and_diagnostics(summary, lookup, plan):
    people = list(range(plan["evaluation_participants"]))
    cells, methods, budgets = cell_records(plan), plan["operator"]["methods"], plan["budgets"]

    def vector(cell, method, budget, metric="balanced_accuracy"):
        return np.asarray([lookup[p, cell, method, budget][metric] for p in people])

    attainment = {(r["cell_id"], r["method"]): r for r in summary["calibration_attainment"]}
    expected = {(c["cell_id"], m) for c in cells for m in methods}
    require(
        len(attainment) == len(summary["calibration_attainment"]) and set(attainment) == expected,
        "Attainment inventory",
    )
    for cell in cells:
        for method in methods:
            row = attainment[cell["cell_id"], method]
            require(
                all(row[key] == value for key, value in cell.items()), "Attainment cell descriptor"
            )
            require(row["participant_ids"] == people, "Attainment participant order")
            curve = row["participant_accuracy_curve"]
            require(set(curve) == {str(k) for k in budgets}, "Attainment curve budgets")
            for k in budgets:
                close(curve[str(k)], vector(cell["cell_id"], method, k), "Attainment curve")
            first = [next((k for k in budgets if curve[str(k)][p] >= 0.8), ">5") for p in people]
            require(row["first_observed_budget"] == first, "First observed target attainment")
            labels = [plan["n_classes"] * k if isinstance(k, int) else None for k in first]
            seconds = [
                n * cell["n_samples"] / plan["sfreq"] if n is not None else None for n in labels
            ]
            require(row["labeled_trials"] == labels, "No imputation of unattained labeled trials")
            require(row["analyzed_EEG_seconds"] == seconds, "Analyzed EEG seconds")

    expected_diagnostics = {}
    for cell in cells:
        cid = cell["cell_id"]
        for method in methods:
            expected_diagnostics[cid, method, "k3_minus_k1", None] = vector(
                cid, method, 3
            ) - vector(cid, method, 1)
        for method in ("A0", "pooled", "block", "fusion_pooled", "fusion_block"):
            for k in budgets[1:]:
                for metric in ("balanced_accuracy", "correct_log_probability"):
                    expected_diagnostics[cid, method, f"cal_minus_raw_{metric}", k] = vector(
                        cid, f"{method}_cal", k, metric
                    ) - vector(cid, f"{method}_raw", k, metric)
    diagnostics = {
        (r["cell_id"], r["method"], r["contrast"], r.get("budget")): r
        for r in summary["diagnostics"]
    }
    require(
        len(diagnostics) == len(summary["diagnostics"])
        and set(diagnostics) == set(expected_diagnostics),
        "Diagnostic inventory",
    )
    for key, values in expected_diagnostics.items():
        for field, value in paired(values).items():
            close(diagnostics[key][field], value, f"Diagnostic {key}/{field}")


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_cache(path):
    with np.load(path, allow_pickle=False) as archive:
        return {key: archive[key] for key in archive.files}


def verify_provenance(root, plan_path, plan, start, freeze, evaluation_start, result, repo):
    require(
        {p.name for p in root.iterdir()} == set(plan["execution"]["artifacts"]),
        "Artifact inventory",
    )
    for name in plan["execution"]["artifacts"]:
        path = root / name
        info = path.lstat()
        require(
            stat.S_ISREG(info.st_mode)
            and stat.S_IMODE(info.st_mode) == 0o400
            and info.st_nlink == 1,
            f"Artifact publication mode {name}",
        )
    documents = (
        ("start", start),
        ("source-freeze", freeze),
        ("evaluation-start", evaluation_start),
        ("result", result),
    )
    for name, document in documents:
        require(document["schema"] == f"cfeg.v4-aq-study.{name}.v1", f"Schema {name}")
        require(document["study_id"] == plan["study_id"], f"Study ID {name}")
        require(document["plan_sha256"] == sha(plan_path), f"Plan binding {name}")
        require(document["source_commit"] == start["source_commit"], f"Source commit {name}")
    for document in (freeze, result):
        require(document["start_sha256"] == sha(root / "start.json"), "Start binding")
    for document in (freeze, evaluation_start, result):
        require(
            document["source_scores_sha256"] == sha(root / "source-scores.npz"),
            "Source scores binding",
        )
    for document in (evaluation_start, result):
        require(
            document["source_freeze_sha256"] == sha(root / "source-freeze.json"),
            "Source freeze binding",
        )
    require(
        result["evaluation_scores_sha256"] == sha(root / "evaluation-scores.npz"),
        "Evaluation scores binding",
    )
    require(
        result["evaluation_start_sha256"] == sha(root / "evaluation-start.json"),
        "Evaluation start binding",
    )
    require(
        start["evidence_role"] == result["evidence_role"] == plan["evidence_role"], "Evidence role"
    )
    require(
        start["human_data_access"] is False and result["summary"]["human_unlock"] is False,
        "Human boundary",
    )
    require(start["seed_namespace"] == plan["seed_namespace"], "Namespace binding")
    # Pure hash checks after publication; no RNG creation or EEG replay.
    for split, document in (("source", freeze), ("evaluation", result)):
        expected = int.from_bytes(
            hashlib.sha256(f"{plan['seed_namespace']}/{split}".encode()).digest()[:8], "big"
        )
        require(document[f"{split}_root_seed"] == expected, f"Declared seed derivation {split}")
    require(
        freeze["source_root_seed"] != result["evaluation_root_seed"], "Distinct partition seeds"
    )
    require(start["workers"] == plan["execution"]["workers"], "Worker count")
    require(
        start["runtime"]["blas_threads"]
        == {k: "1" for k in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS")},
        "BLAS runtime",
    )

    def git(*args):
        return subprocess.check_output(["git", "-C", str(repo), *args])

    commit = start["source_commit"]
    require(
        git("rev-parse", f"{commit}^{{tree}}").decode().strip() == start["source_tree"],
        "Historical tree",
    )
    require(
        git("show", f"{commit}:configs/analysis/metadata_calibration_v4_aq_study.json")
        == plan_path.read_bytes(),
        "Historical plan",
    )
    for path, key in (
        (plan["filterbank_config"], "filterbank_sha256"),
        ("src/cfeg/analysis/metadata_calibration_v4_pilot.py", "pure_helper_module_sha256"),
    ):
        require(
            hashlib.sha256(git("show", f"{commit}:{path}")).hexdigest() == start[key] == plan[key],
            f"Historical helper {key}",
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = args.output
    plan = json.loads(args.plan.read_text())
    start = json.loads((root / "start.json").read_text())
    freeze = json.loads((root / "source-freeze.json").read_text())
    evaluation_start = json.loads((root / "evaluation-start.json").read_text())
    result = json.loads((root / "result.json").read_text())
    repo = Path(__file__).resolve().parents[1]
    verify_provenance(root, args.plan, plan, start, freeze, evaluation_start, result, repo)
    source, evaluation = (
        load_cache(root / "source-scores.npz"),
        load_cache(root / "evaluation-scores.npz"),
    )
    require(
        not set(source["base_sha256"]) & set(evaluation["base_sha256"]),
        "Cohort base identity separation",
    )
    require(
        not set(source["eeg_sha256"].flat) & set(evaluation["eeg_sha256"].flat),
        "Cohort EEG separation",
    )
    verify_source(source, freeze, plan)
    report = verify_evaluation(evaluation, freeze, result, plan)
    report["result_sha256"] = sha(root / "result.json")
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
