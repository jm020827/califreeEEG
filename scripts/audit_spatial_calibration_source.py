#!/usr/bin/env python3
"""Read-only independent score arithmetic/provenance audit; never open EEG."""

from __future__ import annotations

import argparse
import hashlib
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
    require(np.shape(actual) == np.shape(expected), f"Shape mismatch: {label}")
    require(np.allclose(actual, expected, rtol=0, atol=1e-10), f"Mismatch: {label}")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def softmax(scores, temperature):
    z = np.asarray(scores) / temperature
    e = np.exp(z - z.max(axis=-1, keepdims=True))
    return e / e.sum(axis=-1, keepdims=True)


def nll(probability, labels):
    return float(
        -np.log(np.maximum(probability[..., np.arange(len(labels)), labels], 1e-300)).mean()
    )


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


def verify_scores(cache, plan):
    p, l = len(plan["source_subject_ids"]), 2 * len(plan["sample_counts"])
    shapes = {
        "schema": (),
        "subject_ids": (p,),
        "cell_ids": (l,),
        "sample_counts": (l,),
        "interface_indices": (l,),
        "query_labels": (60,),
        "query_block_ids": (60,),
        "support_block_ids": (5,),
        "calibration_budgets": (3,),
        "trca_budgets": (2,),
        "a0_scores": (p, l, 60, 12),
        "itcca_scores": (p, l, 3, 60, 12),
        "etrca_unit_scores": (p, l, 2, 60, 12),
        "legacy_etrca_scores": (p, l, 2, 60, 12),
        "raw_file_sha256": (p,),
        "crop_sha256": (p, l),
        "filter_weights": (7,),
    }
    require(set(cache) == set(shapes), "Exact score cache inventory; no hidden descriptors")
    require(str(cache["schema"]) == "cfeg.spatial-calibration-source.scores.v1", "Score schema")
    for name, shape in shapes.items():
        require(cache[name].shape == shape, f"Cache shape {name}")
        if cache[name].dtype.kind == "f":
            require(np.isfinite(cache[name]).all(), f"Finite scores {name}")
    require(cache["subject_ids"].tolist() == plan["source_subject_ids"], "Subject allowlist")
    cells = [f"{e}-n{n}" for n in plan["sample_counts"] for e in plan["interfaces"]]
    require(cache["cell_ids"].tolist() == cells, "Cell order")
    require(
        cache["sample_counts"].tolist() == np.repeat(plan["sample_counts"], 2).tolist(), "Windows"
    )
    require(cache["interface_indices"].tolist() == [0, 1] * (l // 2), "Interfaces")
    require(np.array_equal(cache["query_labels"], np.tile(np.arange(12), 5)), "Query labels")
    require(
        np.array_equal(cache["query_block_ids"], np.repeat(np.arange(5, 10), 12)), "Query blocks"
    )
    require(np.array_equal(cache["support_block_ids"], np.arange(5)), "Support prefix")
    require(cache["calibration_budgets"].tolist() == [1, 3, 5], "Calibration budgets")
    require(cache["trca_budgets"].tolist() == [3, 5], "TRCA budgets")
    require(np.all(cache["filter_weights"] > 0), "Positive filter weights")
    for name in ("a0_scores", "itcca_scores"):
        require(
            np.all((cache[name] >= -1e-12) & (cache[name] <= 1 + 1e-12)), f"Score bounds {name}"
        )
    require(np.max(np.abs(cache["etrca_unit_scores"])) <= 1 + 1e-12, "Unit TRCA bounds")
    require(
        np.max(np.abs(cache["legacy_etrca_scores"])) <= cache["filter_weights"].sum() + 1e-12,
        "Legacy bounds",
    )
    for name in ("raw_file_sha256", "crop_sha256"):
        require(
            all(
                len(str(v)) == 64 and all(c in "0123456789abcdef" for c in str(v))
                for v in cache[name].flat
            ),
            f"Hash format {name}",
        )


def score_for(cache, method, k):
    key, budgets = {
        "ITCCA": ("itcca_scores", [1, 3, 5]),
        "FB_eTRCA_unit": ("etrca_unit_scores", [3, 5]),
        "legacy_FB_eTRCA": ("legacy_etrca_scores", [3, 5]),
    }[method]
    return cache[key][:, :, budgets.index(k)]


def verify_temperature(record, scores, labels, label):
    grid = np.logspace(-4, 1, 31)
    objectives = [nll(softmax(scores, t), labels) for t in grid]
    close(record["temperatures"], grid, f"{label} grid")
    close(record["objectives"], objectives, f"{label} objectives")
    # Independent arithmetic can order ULP-scale ties differently. Stored exact first minimum is binding.
    index = int(np.argmin(record["objectives"]))
    require(record["selected_index"] == index, f"{label} first minimum")
    close(record["temperature"], grid[index], f"{label} temperature")
    close(record["nll"], record["objectives"][index], f"{label} selected NLL")
    return 31


def mixture(a0, scores, weight, temperature):
    return a0 if weight == 0 else (1 - weight) * a0 + weight * softmax(scores, temperature)


def verify_folds(cache, document, plan):
    folds = document["folds"]
    require([f["fold_id"] for f in folds] == [0, 1, 2], "Fold inventory")
    total = 0
    labels = cache["query_labels"]
    for fold in folds:
        index = fold["fold_id"]
        fitting = np.flatnonzero(np.arange(len(cache["subject_ids"])) % 3 != index)
        evaluation = np.flatnonzero(np.arange(len(cache["subject_ids"])) % 3 == index)
        require(fold["fit_subject_ids"] == cache["subject_ids"][fitting].tolist(), "Fit partition")
        require(
            fold["evaluation_subject_ids"] == cache["subject_ids"][evaluation].tolist(),
            "Evaluation partition",
        )
        require(
            set(fold["windows"]) == {str(n) for n in plan["sample_counts"]}, "Window fit inventory"
        )
        for n in plan["sample_counts"]:
            window = fold["windows"][str(n)]
            which = np.flatnonzero(cache["sample_counts"] == n)
            a0scores = cache["a0_scores"][fitting][:, which]
            total += verify_temperature(window["a0_temperature"], a0scores, labels, "A0")
            a0 = softmax(a0scores, window["a0_temperature"]["temperature"])
            require(set(window["budgets"]) == {"1", "3", "5"}, "Fusion budget inventory")
            for k in (1, 3, 5):
                budget = window["budgets"][str(k)]
                scores = score_for(cache, "ITCCA", k)[fitting][:, which]
                fit = budget["fusion_fit"]
                grid = [(0, None)] + [
                    (w, float(t)) for w in (0.25, 0.5, 0.75) for t in np.logspace(-4, 1, 31)
                ]
                require(len(fit["candidates"]) == 94, "Fusion candidate count")
                for record, (weight, temperature) in zip(fit["candidates"], grid, strict=True):
                    require(
                        record["lambda"] == weight and record["temperature"] == temperature,
                        "Fusion grid order",
                    )
                    close(
                        record["nll"],
                        nll(mixture(a0, scores, weight, temperature), labels),
                        "Fusion fit objective",
                    )
                idx = int(np.argmin([v["nll"] for v in fit["candidates"]]))
                require(fit["selected_index"] == idx, "Fusion exact first minimum")
                for name in ("lambda", "temperature", "nll"):
                    require(fit[name] == fit["candidates"][idx][name], f"Fusion selected {name}")
                total += 94
                methods = ["ITCCA"] + (["FB_eTRCA_unit", "legacy_FB_eTRCA"] if k in (3, 5) else [])
                require(
                    set(budget["standalone_temperatures"]) == set(methods),
                    "Standalone fit inventory",
                )
                for method in methods:
                    total += verify_temperature(
                        budget["standalone_temperatures"][method],
                        score_for(cache, method, k)[fitting][:, which],
                        labels,
                        method,
                    )
    return total


def metrics(prob, labels, a0):
    prediction = prob.argmax(-1)
    count = np.bincount(labels, minlength=prob.shape[-1])
    correct = np.bincount(labels, weights=prediction == labels, minlength=prob.shape[-1])
    truth = prob[np.arange(len(labels)), labels]
    other = prob.copy()
    other[np.arange(len(labels)), labels] = -np.inf
    return {
        "balanced_accuracy": float(np.mean(correct / count)),
        "correct_log_probability": float(np.log(np.maximum(truth, 1e-300)).mean()),
        "true_class_margin": float(np.mean(truth - other.max(-1))),
        "prediction_flips_vs_A0": float(np.mean(prediction != a0.argmax(-1))),
        "correct_count": float(correct.sum()),
        "query_count": len(labels),
        "class_correct_counts": correct.tolist(),
        "class_query_counts": count.tolist(),
    }


def verify_rows(cache, freeze, result, plan):
    rows = result["rows"]
    lookup = {(r["subject_id"], r["cell_id"], r["method"], r["budget"]): r for r in rows}
    methods = plan["decoder"]["method_budgets"]
    expected = len(cache["subject_ids"]) * len(cache["cell_ids"]) * sum(map(len, methods.values()))
    require(len(lookup) == len(rows) == expected, "Row inventory/uniqueness")
    for p, subject in enumerate(cache["subject_ids"]):
        fold = freeze["folds"][p % 3]
        for l, cell in enumerate(cache["cell_ids"]):
            n = int(cache["sample_counts"][l])
            window = fold["windows"][str(n)]
            a0 = softmax(cache["a0_scores"][p, l], window["a0_temperature"]["temperature"])
            for method, budgets in methods.items():
                for k in budgets:
                    key = (int(subject), str(cell), method, k)
                    require(key in lookup, f"Missing row {key}")
                    row = lookup[key]
                    require(
                        row["fold_id"] == p % 3
                        and row["n_samples"] == n
                        and row["interface"]
                        == plan["interfaces"][int(cache["interface_indices"][l])],
                        "Row identity",
                    )
                    permutations = 0 if method == "wrong_label_support" and k == 0 else 1
                    if method == "A0" or k == 0:
                        probs = [a0]
                    else:
                        budget = window["budgets"][str(k)]
                        fit = budget["fusion_fit"]
                        scores = score_for(cache, "ITCCA", k)[p, l]
                        if method == "AQ_ITCCA":
                            probs = [mixture(a0, scores, fit["lambda"], fit["temperature"])]
                        elif method == "uniform_shrinkage":
                            probs = [
                                a0
                                if fit["lambda"] == 0
                                else (1 - fit["lambda"]) * a0 + fit["lambda"] / 12
                            ]
                            require(
                                np.array_equal(probs[0].argmax(-1), a0.argmax(-1)),
                                "Uniform A0 argmax invariant",
                            )
                        elif method == "wrong_label_support":
                            permutations = 11
                            probs = [
                                mixture(
                                    a0,
                                    np.roll(scores, shift, -1),
                                    fit["lambda"],
                                    fit["temperature"],
                                )
                                for shift in range(1, 12)
                            ]
                        else:
                            probs = [
                                softmax(
                                    score_for(cache, method, k)[p, l],
                                    budget["standalone_temperatures"][method]["temperature"],
                                )
                            ]
                    require(row["permutations"] == permutations, "Permutation count")
                    calculated = [metrics(prob, cache["query_labels"], a0) for prob in probs]
                    for name in calculated[0]:
                        close(
                            row[name],
                            np.mean([m[name] for m in calculated], axis=0),
                            f"{key}/{name}",
                        )
                    if method == "wrong_label_support":
                        records = row["permutation_metrics"]
                        require(len(records) == (11 if k else 0), "Wrong-label record count")
                        for shift, (record, calculated_shift) in enumerate(
                            zip(records, calculated, strict=False), 1
                        ):
                            require(record["shift"] == shift, "Wrong-label shift order")
                            for name, value in calculated_shift.items():
                                close(record[name], value, f"Wrong-label shift{shift}/{name}")
                    if method == "wrong_label_support" and k:
                        require(
                            row["posterior_sha256"] is None, "No metric-averaged predictor hash"
                        )
                    else:
                        digest = row["posterior_sha256"]
                        require(
                            isinstance(digest, str) and len(digest) == 64, "Posterior digest format"
                        )
                        if (
                            k == 0
                            or method == "A0"
                            or (
                                method in ("AQ_ITCCA", "uniform_shrinkage")
                                and window["budgets"][str(k)]["fusion_fit"]["lambda"] == 0
                            )
                        ):
                            require(
                                digest
                                == lookup[int(subject), str(cell), "A0", 0]["posterior_sha256"],
                                "Exact A0 posterior alias",
                            )
    return expected


def verify_summary(result, plan):
    rows, summary = result["rows"], result["summary"]
    ids = plan["source_subject_ids"]
    cells = [f"{e}-n{n}" for n in plan["sample_counts"] for e in plan["interfaces"]]
    methods = plan["decoder"]["method_budgets"]
    lookup = {(r["subject_id"], r["cell_id"], r["method"], r["budget"]): r for r in rows}

    def value(method, k, selected=cells):
        if k == "eauc":
            return (
                value(method, 0, selected) / 6
                + value(method, 1, selected) / 2
                + value(method, 3, selected) / 3
            )
        return np.array(
            [[lookup[s, c, method, k]["balanced_accuracy"] for s in ids] for c in selected]
        ).mean(axis=0)

    comparisons = {
        "AQ_eauc_gain": paired(value("AQ_ITCCA", "eauc") - value("A0", "eauc")),
        "supervised_eauc_gain": paired(
            value("AQ_ITCCA", "eauc") - value("wrong_label_support", "eauc")
        ),
        "AQ_k1_gain": paired(value("AQ_ITCCA", 1) - value("A0", 1)),
        "AQ5_minus_AQ3": paired(value("AQ_ITCCA", 5) - value("AQ_ITCCA", 3)),
    }
    for interface in ("dry", "wet"):
        selected = [c for c in cells if c.startswith(interface + "-")]
        comparisons[f"{interface}_eauc_gain"] = paired(
            value("AQ_ITCCA", "eauc", selected) - value("A0", "eauc", selected)
        )
    require(set(summary["comparisons"]) == set(comparisons), "Comparison inventory")
    for key, record in comparisons.items():
        require(set(summary["comparisons"][key]) == set(record), "Paired fields")
        for name, number in record.items():
            close(summary["comparisons"][key][name], number, f"{key}/{name}")

    def positive(key):
        return comparisons[key]["mean"] >= 0.01 and comparisons[key]["one_sided_95_LCB"] > 0

    screen = {
        "AQ_utility": positive("AQ_eauc_gain"),
        "supervised_specificity": positive("supervised_eauc_gain"),
        "k1_safety": comparisons["AQ_k1_gain"]["mean"] >= 0,
    }
    for interface in ("dry", "wet"):
        record = comparisons[f"{interface}_eauc_gain"]
        screen[f"{interface}_safety"] = (
            record["mean"] >= -0.005 and record["one_sided_95_LCB"] > -1 / 60
        )
    require(summary["screen_components"] == screen, "Prespecified gates")
    status = "USEFUL_EEG_CALIBRATION_DEVELOPMENT" if all(screen.values()) else "AQ_NOT_ESTABLISHED"
    require(
        summary["status"] == status and summary["human_held_unlock"] is False,
        "Status and held boundary",
    )
    aggregates = {(r["cell_id"], r["method"], r["budget"]): r for r in summary["aggregate_rows"]}
    harms = {(r["subject_id"], r["method"], r["budget"]): r for r in summary["participant_harm"]}
    attainments = {(r["cell_id"], r["method"]): r for r in summary["calibration_attainment"]}
    width = sum(map(len, methods.values()))
    require(
        len(aggregates) == len(summary["aggregate_rows"]) == len(cells) * width,
        "Aggregate inventory",
    )
    require(len(harms) == len(summary["participant_harm"]) == len(ids) * width, "Harm inventory")
    require(
        len(attainments) == len(summary["calibration_attainment"]) == len(cells) * len(methods),
        "Attainment inventory",
    )
    fields = {
        "balanced_accuracy",
        "correct_log_probability",
        "true_class_margin",
        "prediction_flips_vs_A0",
        "correct_count",
        "query_count",
        "class_correct_counts",
        "class_query_counts",
    }
    for method, budgets in methods.items():
        for k in budgets:
            for i, subject in enumerate(ids):
                record = harms[subject, method, k]
                close(record["equal_cell_mean_BA"], value(method, k)[i], "Harm BA")
                close(record["delta_vs_A0"], value(method, k)[i] - value("A0", k)[i], "Harm vs A0")
        for cell in cells:
            curve = {}
            n = int(cell.split("-n")[1])
            for k in budgets:
                record = aggregates[cell, method, k]
                require(
                    record["interface"] == cell.split("-n")[0] and record["n_samples"] == n,
                    "Aggregate identity",
                )
                group = [lookup[s, cell, method, k] for s in ids]
                require(set(record["mean_metrics"]) == fields, "Mean metric inventory")
                for field in fields:
                    close(
                        record["mean_metrics"][field],
                        np.mean([r[field] for r in group], axis=0),
                        f"Mean {field}",
                    )
                curve[str(k)] = [r["balanced_accuracy"] for r in group]
                close(
                    record["fraction_harmed_vs_A0"],
                    np.mean(
                        [
                            r["balanced_accuracy"] < lookup[s, cell, "A0", k]["balanced_accuracy"]
                            for s, r in zip(ids, group, strict=True)
                        ]
                    ),
                    "Cell harm",
                )
            record = attainments[cell, method]
            require(
                record["subject_ids"] == ids
                and set(record["participant_accuracy_curve"]) == set(curve),
                "Attainment identity/budgets",
            )
            for k, numbers in curve.items():
                close(record["participant_accuracy_curve"][k], numbers, "Attainment curve")
            first = [
                next((k for k in budgets if curve[str(k)][i] >= 0.8), ">5") for i in range(len(ids))
            ]
            require(record["first_observed_budget"] == first, "First observed budget")
            require(
                record["labeled_trials"] == [12 * k if isinstance(k, int) else None for k in first],
                "Label cost/censoring",
            )
            for actual, k in zip(record["EEG_seconds"], first, strict=True):
                if isinstance(k, int):
                    close(actual, 12 * k * n / 250, "Analyzed EEG cost")
                else:
                    require(actual is None, "Censored cost not imputed")
            interpretation = (
                "expected_permutation_metric_not_deployable_predictor"
                if method == "wrong_label_support"
                else "observed_budget_only"
            )
            require(record["interpretation"] == interpretation, "Attainment interpretation")
    return {
        "status": status,
        "aggregate_rows_checked": len(aggregates),
        "participant_harm_checked": len(harms),
        "attainment_rows_checked": len(attainments),
    }


def verify_provenance(root, plan_path, plan, start, freeze, result, repo):
    require(
        {p.name for p in root.iterdir()} == set(plan["execution"]["artifacts"]),
        "Artifact inventory",
    )
    for name in plan["execution"]["artifacts"]:
        info = (root / name).lstat()
        require(
            stat.S_ISREG(info.st_mode)
            and stat.S_IMODE(info.st_mode) == 0o400
            and info.st_nlink == 1,
            "Immutable regular artifact",
        )
    require(
        sum((root / n).stat().st_size for n in plan["execution"]["artifacts"])
        <= plan["execution"]["resource_budget_bytes"],
        "Artifact quota",
    )
    for doc, kind in ((start, "start"), (freeze, "fold-freezes"), (result, "result")):
        require(doc["schema"] == f"cfeg.spatial-calibration-source.{kind}.v1", "Document schema")
        require(
            doc["study_id"] == plan["study_id"] and doc["plan_sha256"] == sha(plan_path),
            "Study/plan binding",
        )
        if kind != "start":
            require(doc["start_sha256"] == sha(root / "start.json"), "Start binding")
            require(doc["scores_sha256"] == sha(root / "scores.npz"), "Score binding")
            require(doc["source_commit"] == start["source_commit"], "Source commit binding")
    require(result["fold_freezes_sha256"] == sha(root / "fold-freezes.json"), "Freeze binding")
    require(result["evidence_role"] == plan["evidence_role"], "Exposed-development evidence role")
    require(
        start["human_held_access"] is False and result["summary"]["human_held_unlock"] is False,
        "Held boundary",
    )
    require(start["metadata_access"] is False, "Metadata-free boundary")
    require(start["source_subject_ids"] == plan["source_subject_ids"], "Start allowlist")
    require(start["workers"] == plan["execution"]["workers"], "Workers")
    require(
        start["runtime"]["blas_threads"]
        == {k: "1" for k in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS")},
        "BLAS contract",
    )

    def git(*args):
        return subprocess.check_output(["git", "-C", str(repo), *args])

    commit = start["source_commit"]
    require(
        git("rev-parse", f"{commit}^{{tree}}").decode().strip() == start["source_tree"],
        "Historical tree",
    )
    require(
        git("show", f"{commit}:configs/analysis/spatial_calibration_source39_v1.json")
        == plan_path.read_bytes(),
        "Historical plan",
    )
    require(start["pinned_files"] == plan["pinned_files"], "Helper inventory")
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
    start, freeze, result = [
        json.loads((args.output / name).read_text())
        for name in ("start.json", "fold-freezes.json", "result.json")
    ]
    repo = Path(__file__).resolve().parents[1]
    verify_provenance(args.output, args.plan, plan, start, freeze, result, repo)
    with np.load(args.output / "scores.npz", allow_pickle=False) as archive:
        cache = {name: archive[name] for name in archive.files}
    verify_scores(cache, plan)
    close(
        cache["filter_weights"],
        yaml.safe_load((repo / plan["filterbank_config"]).read_text())["weights"],
        "Pinned weights",
    )
    fits = verify_folds(cache, freeze, plan)
    count = verify_rows(cache, freeze, result, plan)
    require(
        fits == plan["fitting"]["expected_objective_count"]
        and count == plan["reporting"]["expected_rows"],
        "Study count contract",
    )
    print(
        json.dumps(
            {
                "audit": "PASS",
                "fit_objectives_checked": fits,
                "metric_rows_checked": count,
                **verify_summary(result, plan),
                "result_sha256": sha(args.output / "result.json"),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
