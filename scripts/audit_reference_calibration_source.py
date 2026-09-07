#!/usr/bin/env python3
"""Independent read-only reference-cache audit and all-band diagnostics; no EEG loader."""

from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.util
import json
import stat
import subprocess
from pathlib import Path

import numpy as np
import yaml

# This is a previous independent auditor, not a producer, decoder, or human-data module.
_spec = importlib.util.spec_from_file_location(
    "_independent_spatial_audit", Path(__file__).with_name("audit_spatial_calibration_source.py")
)
_audit = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_audit)
require, close, sha = _audit.require, _audit.close, _audit.sha
softmax, metrics, paired = _audit.softmax, _audit.metrics, _audit.paired


def verify_features(cache, plan):
    p, a, l = len(plan["source_subject_ids"]), 2, 2 * len(plan["sample_counts"])
    shapes = {
        "schema": (),
        "subject_ids": (p,),
        "pipeline_ids": (a,),
        "cell_ids": (l,),
        "sample_counts": (l,),
        "interface_indices": (l,),
        "query_labels": (60,),
        "query_block_ids": (60,),
        "support_block_ids": (5,),
        "calibration_budgets": (3,),
        "wrong_label_shifts": (11,),
        "filter_weights": (7,),
        "ecca_components": (p, a, l, 3, 7, 60, 12, 4),
        "wrong_ecca_scores": (p, a, l, 3, 11, 60, 12),
        "itcca_band_rho": (p, a, l, 3, 7, 60, 12),
        "aref_band_rho": (p, a, l, 7, 60, 12),
        "legacy_a0_scores": (p, l, 60, 12),
        "support_correlations": (p, a, l, 7, 12, 5, 5),
        "reference_projection_fraction": (p, a, l, 7, 10, 12, 12),
        "line50_fraction": (p, a, l, 10, 12),
        "raw_file_sha256": (p,),
        "crop_sha256": (p, a, l),
    }
    require(set(cache) == set(shapes), "Exact feature inventory, no hidden descriptors")
    require(str(cache["schema"]) == "cfeg.reference-calibration-source.features.v1", "Schema")
    for key, shape in shapes.items():
        require(cache[key].shape == shape, f"Feature shape {key}")
        if cache[key].dtype.kind == "f":
            require(np.isfinite(cache[key]).all(), f"Finite feature {key}")
    require(cache["subject_ids"].tolist() == plan["source_subject_ids"], "Subject allowlist")
    require(cache["pipeline_ids"].tolist() == plan["pipeline_ids"], "Pipeline order")
    require(
        cache["cell_ids"].tolist()
        == [f"{e}-n{n}" for n in plan["sample_counts"] for e in plan["interfaces"]],
        "Cell order",
    )
    require(cache["sample_counts"].tolist() == np.repeat(plan["sample_counts"], 2).tolist(), "N")
    require(cache["interface_indices"].tolist() == [0, 1] * (l // 2), "Interfaces")
    for key, expected in {
        "query_labels": np.tile(np.arange(12), 5),
        "query_block_ids": np.repeat(np.arange(5, 10), 12),
        "support_block_ids": np.arange(5),
        "calibration_budgets": [1, 3, 5],
        "wrong_label_shifts": np.arange(1, 12),
    }.items():
        require(np.array_equal(cache[key], expected), f"Index {key}")
    require(np.all(cache["filter_weights"] > 0), "Positive weights")
    for key in ("ecca_components", "support_correlations"):
        require(np.max(np.abs(cache[key])) <= 1 + 1e-10, f"Pearson bounds {key}")
    for key in (
        "itcca_band_rho",
        "aref_band_rho",
        "legacy_a0_scores",
        "reference_projection_fraction",
        "line50_fraction",
    ):
        require(
            np.min(cache[key]) >= -1e-10 and np.max(cache[key]) <= 1 + 1e-10,
            f"Fraction/rho bounds {key}",
        )
    require(np.max(np.abs(cache["wrong_ecca_scores"])) <= 4 + 1e-10, "Signed score bounds")
    for k in range(3):
        close(
            cache["ecca_components"][:, :, :, k, ..., 0],
            cache["aref_band_rho"],
            "Exact same projected r1 for every budget",
        )
    close(
        cache["support_correlations"],
        cache["support_correlations"].swapaxes(-1, -2),
        "Support correlation symmetry",
    )
    for key in ("raw_file_sha256", "crop_sha256"):
        require(
            all(
                len(str(v)) == 64 and all(c in "0123456789abcdef" for c in str(v))
                for v in cache[key].flat
            ),
            f"Hash format {key}",
        )


def reconstructed_scores(cache):
    w = cache["filter_weights"] / cache["filter_weights"].sum()
    r = cache["ecca_components"]
    return {
        "A0_reference": np.einsum("f,palfqc->palqc", w, cache["aref_band_rho"] ** 2),
        "ECCA": np.einsum("f,palkfqc->palkqc", w, (np.sign(r) * r**2).sum(-1)),
        "ITCCA": np.einsum("f,palkfqc->palkqc", w, cache["itcca_band_rho"] ** 2),
        "legacy_A0": cache["legacy_a0_scores"],
    }


def verify_folds(cache, freeze, plan, scores=None):
    scores = reconstructed_scores(cache) if scores is None else scores
    require([f["fold_id"] for f in freeze["folds"]] == [0, 1, 2], "Fold inventory")
    total = 0
    for fold in freeze["folds"]:
        fit = np.flatnonzero(np.arange(len(cache["subject_ids"])) % 3 != fold["fold_id"])
        ev = np.flatnonzero(np.arange(len(cache["subject_ids"])) % 3 == fold["fold_id"])
        require(fold["fit_subject_ids"] == cache["subject_ids"][fit].tolist(), "Fit identities")
        require(
            fold["evaluation_subject_ids"] == cache["subject_ids"][ev].tolist(), "Eval identities"
        )
        require(set(fold["pipelines"]) == set(plan["pipeline_ids"]), "Fit pipelines")
        require(
            set(fold["legacy_a0_temperatures"]) == {str(n) for n in plan["sample_counts"]},
            "Legacy window fits",
        )
        for a, pipeline in enumerate(plan["pipeline_ids"]):
            windows = fold["pipelines"][pipeline]["windows"]
            require(set(windows) == {str(n) for n in plan["sample_counts"]}, "Window fits")
            for n in plan["sample_counts"]:
                which = np.flatnonzero(cache["sample_counts"] == n)
                window = windows[str(n)]
                total += _audit.verify_temperature(
                    window["aref_temperature"],
                    scores["A0_reference"][fit, a][:, which],
                    cache["query_labels"],
                    "Aref",
                )
                require(set(window["budgets"]) == {"1", "3", "5"}, "Fit budget inventory")
                for j, k in enumerate((1, 3, 5)):
                    require(
                        set(window["budgets"][str(k)]) == {"ecca_temperature", "itcca_temperature"},
                        "No extra fitting or selected mixtures",
                    )
                    for method, key in (
                        ("ECCA", "ecca_temperature"),
                        ("ITCCA", "itcca_temperature"),
                    ):
                        total += _audit.verify_temperature(
                            window["budgets"][str(k)][key],
                            scores[method][fit, a][:, which, j],
                            cache["query_labels"],
                            method,
                        )
        for n in plan["sample_counts"]:
            which = np.flatnonzero(cache["sample_counts"] == n)
            total += _audit.verify_temperature(
                fold["legacy_a0_temperatures"][str(n)],
                scores["legacy_A0"][fit][:, which],
                cache["query_labels"],
                "Legacy",
            )
    return total


def methods_for(plan, pipeline):
    methods = copy.deepcopy(plan["decoder"]["method_budgets"])
    if pipeline == "no_notch":
        methods["legacy_A0"] = plan["decoder"]["legacy_budgets"]
    return methods


def verify_rows(cache, freeze, result, plan, scores=None):
    scores = reconstructed_scores(cache) if scores is None else scores
    lookup = {
        (r["subject_id"], r["pipeline"], r["cell_id"], r["method"], r["budget"]): r
        for r in result["rows"]
    }
    expected = (
        len(cache["subject_ids"])
        * len(cache["cell_ids"])
        * sum(sum(map(len, methods_for(plan, a).values())) for a in plan["pipeline_ids"])
    )
    require(len(lookup) == len(result["rows"]) == expected, "Full row inventory/uniqueness")
    for p, s in enumerate(cache["subject_ids"]):
        fold = freeze["folds"][p % 3]
        for a, pipeline in enumerate(plan["pipeline_ids"]):
            for l, cell in enumerate(cache["cell_ids"]):
                n = str(cache["sample_counts"][l])
                win = fold["pipelines"][pipeline]["windows"][n]
                a0 = softmax(
                    scores["A0_reference"][p, a, l], win["aref_temperature"]["temperature"]
                )
                anchor = lookup[int(s), pipeline, str(cell), "A0_reference", 0]["posterior_sha256"]
                for method, budgets in methods_for(plan, pipeline).items():
                    for k in budgets:
                        key = (int(s), pipeline, str(cell), method, k)
                        require(key in lookup, f"Missing {key}")
                        row = lookup[key]
                        require(
                            row["fold_id"] == p % 3
                            and row["n_samples"] == int(n)
                            and row["interface"]
                            == plan["interfaces"][cache["interface_indices"][l]],
                            "Row identity",
                        )
                        if method == "legacy_A0":
                            probs = [
                                softmax(
                                    scores[method][p, l],
                                    fold["legacy_a0_temperatures"][n]["temperature"],
                                )
                            ]
                        elif method == "A0_reference" or k == 0:
                            probs = [a0]
                        else:
                            j = (1, 3, 5).index(k)
                            fits = win["budgets"][str(k)]
                            if method == "wrong_label_support":
                                probs = [
                                    softmax(raw, fits["ecca_temperature"]["temperature"])
                                    for raw in cache["wrong_ecca_scores"][p, a, l, j]
                                ]
                            else:
                                probs = [
                                    softmax(
                                        scores[method][p, a, l, j],
                                        fits[
                                            "ecca_temperature"
                                            if method == "ECCA"
                                            else "itcca_temperature"
                                        ]["temperature"],
                                    )
                                ]
                        calculated = [metrics(prob, cache["query_labels"], a0) for prob in probs]
                        permutations = (11 if k else 0) if method == "wrong_label_support" else 1
                        require(row["permutations"] == permutations, "Permutation count")
                        for field in calculated[0]:
                            close(
                                row[field],
                                np.mean([r[field] for r in calculated], axis=0),
                                f"{key}/{field}",
                            )
                        if method == "wrong_label_support":
                            records = row["permutation_metrics"]
                            require(len(records) == (11 if k else 0), "Wrong-label metrics count")
                            for shift, (r, truth) in enumerate(
                                zip(records, calculated, strict=False), 1
                            ):
                                require(r["shift"] == shift, "Wrong-label shift order")
                                for field in truth:
                                    close(r[field], truth[field], f"Wrong shift{shift}/{field}")
                        if method == "wrong_label_support" and k:
                            require(
                                row["posterior_sha256"] is None,
                                "Not a deployable averaged predictor",
                            )
                        else:
                            digest = row["posterior_sha256"]
                            require(isinstance(digest, str) and len(digest) == 64, "Posterior hash")
                            if method == "A0_reference" or (k == 0 and method != "legacy_A0"):
                                require(digest == anchor, "Exact Aref posterior alias")
                            if method == "legacy_A0":
                                require(
                                    digest
                                    == lookup[int(s), pipeline, str(cell), method, 0][
                                        "posterior_sha256"
                                    ],
                                    "Legacy budget alias",
                                )
    return expected


def _translated(value):
    if isinstance(value, list):
        return [_translated(v) for v in value]
    if isinstance(value, dict):
        return {
            k: (
                {"A0_reference": "A0", "ECCA": "AQ_ITCCA"}.get(v, v)
                if k == "method"
                else _translated(v)
            )
            for k, v in value.items()
        }
    return value


def verify_summary(result, plan):
    summary = result["summary"]
    require(
        summary["status"] == "DIAGNOSTIC_COMPLETE" and summary["human_held_unlock"] is False,
        "Completion is not efficacy, held closed",
    )
    require(
        set(summary["pipeline_summaries"]) == set(plan["pipeline_ids"]), "Both pipelines reported"
    )
    counts = {}
    for pipeline in plan["pipeline_ids"]:
        subplan = copy.deepcopy(plan)
        subplan["decoder"]["method_budgets"] = {
            {"A0_reference": "A0", "ECCA": "AQ_ITCCA"}.get(m, m): b
            for m, b in methods_for(plan, pipeline).items()
        }
        rows = [r for r in result["rows"] if r["pipeline"] == pipeline]
        subsummary = summary["pipeline_summaries"][pipeline]
        counts[pipeline] = _audit.verify_summary(
            {"rows": _translated(rows), "summary": _translated(subsummary)}, subplan
        )
        for record in subsummary["calibration_attainment"]:
            require(
                len(record["history_seconds"]) == len(plan["source_subject_ids"]), "History count"
            )
            for actual, k in zip(
                record["history_seconds"], record["first_observed_budget"], strict=True
            ):
                if isinstance(k, int):
                    close(
                        actual,
                        12 * k * 160 / 250,
                        "Available history cost (not total acquisition time)",
                    )
                else:
                    require(actual is None, "Censored history")
    lookup = {
        (r["pipeline"], r["subject_id"], r["cell_id"], r["method"], r["budget"]): r
        for r in result["rows"]
    }
    cells = [f"{e}-n{n}" for n in plan["sample_counts"] for e in plan["interfaces"]]

    def value(pipeline, method, k):
        if k == "eauc":
            return (
                value(pipeline, method, 0) / 6
                + value(pipeline, method, 1) / 2
                + value(pipeline, method, 3) / 3
            )
        return np.array(
            [
                [
                    lookup[pipeline, s, c, method, k]["balanced_accuracy"]
                    for s in plan["source_subject_ids"]
                ]
                for c in cells
            ]
        ).mean(0)

    def delta(method, k):
        return value("causal_notch50", method, k) - value("no_notch", method, k)

    comparisons = {
        "calibration_gain_interaction": paired(
            delta("ECCA", "eauc") - delta("A0_reference", "eauc")
        ),
        **{
            f"{m}_eauc": paired(delta(m, "eauc"))
            for m in ("ECCA", "A0_reference", "wrong_label_support")
        },
        **{f"ITCCA_k{k}": paired(delta("ITCCA", k)) for k in (1, 3, 5)},
    }
    require(
        set(summary["paired_pipeline_diagnostics"]) == set(comparisons), "Paired contrast inventory"
    )
    for name, record in comparisons.items():
        require(set(summary["paired_pipeline_diagnostics"][name]) == set(record), "Contrast fields")
        for field, value in record.items():
            close(summary["paired_pipeline_diagnostics"][name][field], value, f"{name}/{field}")
    return counts


def diagnostic_report(cache):
    """Every cell/band retained. Offline true labels are evaluation labels, never decoder input."""
    output = {
        "role": "descriptive_saved_features_not_raw_replay_or_causal_attribution",
        "score_rows": [],
        "signal_rows": [],
        "line50_rows": [],
    }
    labels = cache["query_labels"]
    correct = np.eye(12, dtype=bool)[labels]
    offdiag = ~np.eye(5, dtype=bool)

    def report(values):
        values = np.asarray(values, dtype=float)
        return {"participant_values": values.tolist(), "mean": float(values.mean())}

    for a, pipeline in enumerate(cache["pipeline_ids"]):
        for l, cell in enumerate(cache["cell_ids"]):
            identity = {"pipeline": str(pipeline), "cell_id": str(cell)}
            line = cache["line50_fraction"][:, a, l]
            output["line50_rows"].append(
                {
                    **identity,
                    "support": report(line[:, :5].mean((1, 2))),
                    "query": report(line[:, 5:].mean((1, 2))),
                }
            )
            for f in range(7):
                base = {**identity, "band_index": f}
                cor = cache["support_correlations"][:, a, l, f]
                projection = cache["reference_projection_fraction"][:, a, l, f]
                row = {
                    **base,
                    "support_offdiagonal_correlation": report(cor[..., offdiag].mean((1, 2))),
                }
                for group, blocks in (("support", slice(0, 5)), ("query", slice(5, 10))):
                    fraction = projection[:, blocks]
                    true = np.diagonal(fraction, axis1=-2, axis2=-1)
                    wrong = (fraction.sum(-1) - true) / 11
                    row[f"{group}_matched_reference_fraction"] = report(true.mean((1, 2)))
                    row[f"{group}_meanwrong_reference_fraction"] = report(wrong.mean((1, 2)))
                    row[f"{group}_reference_margin"] = report((true - wrong).mean((1, 2)))
                output["signal_rows"].append(row)
                groups = [("A0_reference", 0, cache["aref_band_rho"][:, a, l, f] ** 2)]
                for j, k in enumerate((1, 3, 5)):
                    r = cache["ecca_components"][:, a, l, j, f]
                    groups.extend(
                        [
                            ("ECCA", k, (np.sign(r) * r**2).sum(-1)),
                            ("ITCCA", k, cache["itcca_band_rho"][:, a, l, j, f] ** 2),
                        ]
                    )
                for method, k, scores in groups:
                    truth = scores[:, np.arange(60), labels]
                    meanwrong = (scores.sum(-1) - truth) / 11
                    maxwrong = np.where(correct[None], -np.inf, scores).max(-1)
                    output["score_rows"].append(
                        {
                            **base,
                            "method": method,
                            "budget": k,
                            "balanced_accuracy": report((scores.argmax(-1) == labels).mean(-1)),
                            "true_score": report(truth.mean(-1)),
                            "meanwrong_score": report(meanwrong.mean(-1)),
                            "margin_vs_meanwrong": report((truth - meanwrong).mean(-1)),
                            "margin_vs_maxwrong": report((truth - maxwrong).mean(-1)),
                        }
                    )
    return output


def verify_provenance(root, plan_path, plan, start, freeze, result, repo):
    require(
        {p.name for p in root.iterdir()} == set(plan["execution"]["artifacts"]), "Exact artifacts"
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
    hashes = {n: sha(root / n) for n in plan["execution"]["artifacts"]}
    for doc, kind in ((start, "start"), (freeze, "fold-freezes"), (result, "result")):
        require(doc["schema"] == f"cfeg.reference-calibration-source.{kind}.v1", "Document schema")
        require(
            doc["study_id"] == plan["study_id"] and doc["plan_sha256"] == sha(plan_path),
            "Study/plan binding",
        )
        if kind != "start":
            require(
                doc["start_sha256"] == hashes["start.json"]
                and doc["features_sha256"] == hashes["features.npz"]
                and doc["source_commit"] == start["source_commit"],
                "Artifact/source linkage",
            )
    require(result["fold_freezes_sha256"] == hashes["fold-freezes.json"], "Fold freeze binding")
    require(result["evidence_role"] == plan["evidence_role"], "Exposed development role")
    require(
        start["human_held_access"] is False
        and start["metadata_access"] is False
        and result["summary"]["human_held_unlock"] is False,
        "Human/metadata boundary",
    )
    require(
        start["source_subject_ids"] == plan["source_subject_ids"]
        and start["workers"] == plan["execution"]["workers"],
        "Start scope/workers",
    )
    require(
        start["runtime"]["blas_threads"]
        == {k: "1" for k in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS")},
        "BLAS boundary",
    )

    def git(*args):
        return subprocess.check_output(["git", "-C", str(repo), *args])

    commit = start["source_commit"]
    require(
        git("rev-parse", f"{commit}^{{tree}}").decode().strip() == start["source_tree"],
        "Historical tree",
    )
    require(
        git("show", f"{commit}:configs/analysis/reference_calibration_source39_v1.json")
        == plan_path.read_bytes(),
        "Historical plan",
    )
    require(start["pinned_files"] == plan["pinned_files"], "Pin inventory")
    require(start["pipeline_ids"] == plan["pipeline_ids"], "Start pipeline binding")
    for filename, expected in plan["pinned_files"].items():
        require(
            hashlib.sha256(git("show", f"{commit}:{filename}")).hexdigest() == expected,
            "Historical pin",
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--diagnostics", action="store_true", help="Print every band/cell diagnostic after audit"
    )
    args = parser.parse_args()
    plan = json.loads(args.plan.read_text())
    start, freeze, result = [
        json.loads((args.output / name).read_text())
        for name in ("start.json", "fold-freezes.json", "result.json")
    ]
    repo = Path(__file__).resolve().parents[1]
    verify_provenance(args.output, args.plan, plan, start, freeze, result, repo)
    with np.load(args.output / "features.npz", allow_pickle=False) as archive:
        cache = {name: archive[name] for name in archive.files}
    verify_features(cache, plan)
    close(
        cache["filter_weights"],
        yaml.safe_load((repo / plan["filterbank_config"]).read_text())["weights"],
        "Weights",
    )
    scores = reconstructed_scores(cache)
    fits = verify_folds(cache, freeze, plan, scores)
    rows = verify_rows(cache, freeze, result, plan, scores)
    require(
        fits == plan["fitting"]["expected_objective_count"]
        and rows == plan["reporting"]["expected_rows"],
        "Expected study counts",
    )
    output = {
        "audit": "PASS",
        "fit_objectives_checked": fits,
        "metric_rows_checked": rows,
        "pipelines": verify_summary(result, plan),
        "result_sha256": sha(args.output / "result.json"),
    }
    if args.diagnostics:
        output["diagnostics"] = diagnostic_report(cache)
    print(json.dumps(output, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
