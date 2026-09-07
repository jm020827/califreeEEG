#!/usr/bin/env python3
"""Post-outcome, read-only diagnostics of frozen scores; no refitting or EEG access."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def diagnose(cache, freezes, result):
    labels = cache["query_labels"]
    n_classes = cache["a0_scores"].shape[-1]
    rows = []
    for l, cell in enumerate(cache["cell_ids"]):
        for b, budget in enumerate(cache["calibration_budgets"]):
            scores = cache["itcca_scores"][:, l, b]
            truth = scores[:, np.arange(len(labels)), labels]
            wrong = scores.copy()
            wrong[:, np.arange(len(labels)), labels] = -np.inf
            prediction = scores.argmax(-1)
            accuracy = (prediction == labels).mean(-1)  # Fixed balanced class counts.
            histograms = np.stack([np.bincount(p, minlength=n_classes) for p in prediction])
            rows.append(
                {
                    "cell_id": str(cell),
                    "budget": int(budget),
                    "mean_correct_template_score": float(truth.mean()),
                    "mean_wrong_template_score": float(
                        (scores.sum(-1) - truth).mean() / (n_classes - 1)
                    ),
                    "mean_correct_minus_max_wrong_score": float((truth - wrong.max(-1)).mean()),
                    "median_participant_BA": float(np.median(accuracy)),
                    "maximum_participant_BA": float(accuracy.max()),
                    "pooled_predicted_class_counts": histograms.sum(0).tolist(),
                    "maximum_participant_modal_fraction": float(histograms.max() / len(labels)),
                }
            )
    fits = [
        budget["fusion_fit"]
        for fold in freezes["folds"]
        for window in fold["windows"].values()
        for budget in window["budgets"].values()
    ]
    means = {}
    for record in result["summary"]["aggregate_rows"]:
        key = (record["method"], record["budget"])
        means.setdefault(key, []).append(record["mean_metrics"])
    curves = [
        {
            "method": method,
            "budget": budget,
            "BA": float(np.mean([m["balanced_accuracy"] for m in records])),
            "correct_log_probability": float(
                np.mean([m["correct_log_probability"] for m in records])
            ),
        }
        for (method, budget), records in sorted(means.items())
    ]
    return {
        "schema": "cfeg.spatial-calibration-source.posthoc-diagnostic.v1",
        "evidence_role": "post-outcome descriptive diagnosis; no model/temperature/cell selection",
        "limitation": "summed scores cannot identify noise source, phase/latency, filter-edge effects, or metadata utility",
        "fusion_fits": len(fits),
        "zero_lambda_fits": sum(f["lambda"] == 0 for f in fits),
        "ITCCA_score_diagnostics": rows,
        "global_observed_curves": curves,
    }


def lineage(cache, previous):
    pairs = [
        (k, k) for k in ("subject_ids", "cell_ids", "raw_file_sha256", "crop_sha256", "a0_scores")
    ]
    pairs.append(("etrca_scores", "legacy_etrca_scores"))
    return {old: bool(np.array_equal(previous[old], cache[new])) for old, new in pairs}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path, required=True, help="Existing study artifact directory, read only"
    )
    parser.add_argument(
        "--previous-features",
        type=Path,
        help="Optional prior exposed-source cached statistics; no raw EEG",
    )
    args = parser.parse_args()
    with np.load(args.output / "scores.npz", allow_pickle=False) as archive:
        cache = {key: archive[key] for key in archive.files}
    freezes, result = [
        json.loads((args.output / name).read_text())
        for name in ("fold-freezes.json", "result.json")
    ]
    report = diagnose(cache, freezes, result)
    report["input_sha256"] = {
        name: hashlib.sha256((args.output / name).read_bytes()).hexdigest()
        for name in ("scores.npz", "fold-freezes.json", "result.json")
    }
    if args.previous_features is not None:
        with np.load(args.previous_features, allow_pickle=False) as previous:
            report["previous_cache_lineage"] = lineage(cache, previous)
        report["previous_features_sha256"] = hashlib.sha256(
            args.previous_features.read_bytes()
        ).hexdigest()
    print(json.dumps(report, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
