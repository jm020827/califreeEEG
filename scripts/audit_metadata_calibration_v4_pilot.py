#!/usr/bin/env python3
"""Read-only independent arithmetic audit; never imports or executes the pilot DGP."""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import subprocess
from pathlib import Path

import numpy as np
from scipy import stats

BASE_METHODS = ("A0", "AQ", "AQM_block", "AQM_scalar", "missing", "pooled")
CONTROLS = ("AQM_block_deranged", "AQM_scalar_deranged")


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def close(actual, expected, name: str) -> None:
    require(
        np.allclose(actual, expected, rtol=0, atol=1e-12, equal_nan=False),
        f"Arithmetic mismatch: {name}",
    )


def paired(values: np.ndarray) -> dict:
    n = len(values)
    mean = float(np.mean(values))
    se = float(np.std(values, ddof=1) / np.sqrt(n))
    critical = stats.t.ppf([0.95, 0.975], n - 1)
    return {
        "n": n,
        "mean": mean,
        "standard_error": se,
        "one_sided_95_LCB": mean - critical[0] * se,
        "two_sided_95_CI": [mean - critical[1] * se, mean + critical[1] * se],
        "two_sided_90_CI": [mean - critical[0] * se, mean + critical[0] * se],
        "participant_values": values.tolist(),
    }


def audit_result(plan: dict, result: dict) -> dict:
    """Check paired estimates and exact control evidence without regenerating EEG."""
    summary = result.get("summary", result)
    participants = result.get("participant_results", result.get("participant_evidence"))
    require(isinstance(participants, list), "Missing participant evidence")
    n = plan["participants"]
    require([p["participant"] for p in participants] == list(range(n)), "Participant inventory")
    rows = summary.get("participant_metric_rows")
    if rows is None:
        rows = [row for participant in participants for row in participant["rows"]]
    families, budgets = plan["families"], plan["budgets"]
    expected = {
        (p, f, m, k)
        for p in range(n)
        for f in families
        for m in (*BASE_METHODS, *CONTROLS)
        for k in ((3, 5) if m in CONTROLS else budgets)
    }
    key = lambda r: (r["participant"], r["family"], r["method"], r["budget"])
    lookup = {key(row): row for row in rows}
    require(len(rows) == len(lookup) and set(lookup) == expected, "Metric row inventory")
    query_count = plan["query_blocks"] * plan["n_classes"]
    prediction_checks = 0
    permutation_checks = 0
    for participant in participants:
        p = participant["participant"]
        require(participant["support_indices"] == list(range(plan["support_blocks"])), "Support")
        require(
            participant["query_indices"]
            == list(range(plan["support_blocks"], plan["support_blocks"] + plan["query_blocks"])),
            "Disjoint fixed query",
        )
        require(
            participant["eeg_sha256"]["acquisition_drift"]
            == participant["eeg_sha256"]["metadata_null"],
            "Null must share exact drift EEG",
        )
        require(participant["derangement_counts"] == {"3": 2, "5": 44}, "Derangement counts")
        labels = np.asarray(participant["query_labels"])
        require(labels.shape == (query_count,), "Query label count")
        counts = np.bincount(labels, minlength=plan["n_classes"])
        require(np.all(counts == plan["query_blocks"]), "Class balanced fixed query")
        evidence = participant["prediction_evidence"]
        ev = {(e["family"], e["method"], e["budget"]): e for e in evidence}
        ev_expected = {(f, m, k) for f in families for m in BASE_METHODS for k in budgets}
        require(len(evidence) == len(ev) and set(ev) == ev_expected, "Prediction inventory")
        for (family, method, budget), item in ev.items():
            row = lookup[p, family, method, budget]
            predictions = np.asarray(item["predictions"])
            require(predictions.shape == labels.shape, "Prediction shape")
            require(np.all((predictions >= 0) & (predictions < len(counts))), "Prediction domain")
            correct = np.bincount(labels, weights=(predictions == labels), minlength=len(counts))
            close(row["class_query_counts"], counts, "class counts")
            close(row["class_correct_counts"], correct, "class correct counts")
            close(row["correct_count"], correct.sum(), "correct total")
            close(row["query_count"], query_count, "query total")
            close(row["balanced_accuracy"], np.mean(correct / counts), "balanced accuracy")
            probability = np.asarray(item["true_class_probability"])
            margin = np.asarray(item["true_class_margin"])
            require(
                probability.shape == labels.shape and margin.shape == labels.shape, "Evidence shape"
            )
            require(np.all((probability > 0) & (probability <= 1)), "Probability range")
            require(np.all((margin >= -1) & (margin <= 1)), "Margin range")
            close(row["correct_log_probability"], np.log(probability).mean(), "log probability")
            close(row["true_class_margin"], margin.mean(), "margin")
            for anchor in ("A0", "AQ"):
                anchor_pred = np.asarray(ev[family, anchor, budget]["predictions"])
                close(
                    row[f"prediction_flips_vs_{anchor}"],
                    np.mean(predictions != anchor_pred),
                    "flips",
                )
            peer = None
            if budget == 0 or method == "A0":
                peer = ev[family, "A0", 0]
            elif method == "missing":
                peer = ev[family, "AQ", budget]
            elif method == "AQM_scalar" and budget == 1:
                peer = ev[family, "AQM_block", budget]
            if peer is not None:
                require(
                    item["posterior_sha256"] == peer["posterior_sha256"], "Exact posterior control"
                )
                require(item["predictions"] == peer["predictions"], "Exact prediction control")
            if family == "metadata_null" and method in ("A0", "AQ", "missing", "pooled"):
                peer = ev["acquisition_drift", method, budget]
                require(
                    item["posterior_sha256"] == peer["posterior_sha256"], "M-free null identity"
                )
            if method == "AQM_scalar":
                close(row["weight_turnover"], 0, "Scalar applied turnover")
                close(
                    row["metadata_total_trust"],
                    lookup[p, family, "AQM_block", budget]["metadata_total_trust"],
                    "Equal total metadata trust",
                )
            prediction_checks += 1
        deranged = participant["derangement_evidence"]
        require(len(deranged) == len(families) * len(CONTROLS) * 2, "Control evidence inventory")
        seen = set()
        for item in deranged:
            family, method, budget = item["family"], item["method"], item["budget"]
            require((family, method, budget) not in seen, "Duplicate control evidence")
            seen.add((family, method, budget))
            permutations = [
                list(v)
                for v in itertools.permutations(range(budget))
                if all(i != v[i] for i in range(budget))
            ]
            require(item["permutations"] == permutations, "Complete derangement enumeration")
            values = np.asarray(item["balanced_accuracy"])
            require(values.shape == (len(permutations),), "Per-permutation metric count")
            row = lookup[p, family, method, budget]
            close(row["balanced_accuracy"], values.mean(), "Mean of permutation metrics")
            close(row["permutations"], len(permutations), "Control row permutation count")
            if method == "AQM_scalar_deranged":
                close(row["weight_turnover"], 0, "Deranged scalar applied turnover")
            permutation_checks += len(permutations)

    def values(family: str, method: str, budget: int | str) -> np.ndarray:
        if budget == "eauc":
            return (
                values(family, method, 0)
                + 3 * values(family, method, 1)
                + 2 * values(family, method, 3)
            ) / 6
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
        comparisons[name] = paired(values(family, first, budget) - values(family, second, budget))
        for field, value in comparisons[name].items():
            close(summary["comparisons"][name][field], value, f"{name}/{field}")
    threshold = plan["reporting"]["effect_threshold"]
    passes = lambda r: r["mean"] >= threshold and r["one_sided_95_LCB"] > 0
    scalar = comparisons["drift_block_minus_scalar_k3"]
    null = comparisons["null_block_minus_AQ_k3"]["two_sided_90_CI"]
    margin = plan["reporting"]["null_equivalence_margin"]
    components = {
        "AQ_stable_utility": passes(comparisons["stable_AQ_minus_A0_eauc"])
        and comparisons["stable_AQ_minus_A0_k1"]["mean"] >= 0,
        "metadata_eauc": passes(comparisons["drift_block_minus_AQ_eauc"]),
        "metadata_pairing": passes(comparisons["drift_correct_minus_deranged_k3"]),
        "block_over_scalar": scalar["mean"] > 0 and scalar["one_sided_95_LCB"] > 0,
        "null_equivalence": null[0] >= -margin and null[1] <= margin,
        "drift_nonnegative_vs_A0": comparisons["drift_block_minus_A0_k3"]["mean"] >= 0,
    }
    status = (
        "AQ_NOT_ESTABLISHED"
        if not components["AQ_stable_utility"]
        else "METADATA_NOT_ESTABLISHED"
        if not all(components.values())
        else "READY_FOR_INDEPENDENT_SYNTHETIC_REPLICATION"
    )
    require(summary["screen_components"] == components, "Screen component mismatch")
    require(summary["status"] == status and summary["human_unlock"] is False, "Status mismatch")
    for row in summary["aggregate_rows"]:
        records = [lookup[p, row["family"], row["method"], row["budget"]] for p in range(n)]
        for metric, value in row["mean_metrics"].items():
            close(value, np.mean([r[metric] for r in records], axis=0), f"Aggregate {metric}")
        harm = np.mean(
            values(row["family"], row["method"], row["budget"])
            < values(row["family"], "A0", row["budget"])
        )
        close(row["fraction_harmed_vs_A0"], harm, "Harmed participant fraction")
    for item in summary["calibration_attainment"]:
        family, method = item["family"], item["method"]
        target = plan["reporting"]["calibration_target"]
        expected_budgets = [
            next((k for k in budgets if values(family, method, k)[p] >= target), ">5")
            for p in range(n)
        ]
        require(item["participant_budgets"] == expected_budgets, "Calibration crossing")
        close(item["participant_eauc"], values(family, method, "eauc"), "Participant eAUC")
    return {
        "audit": "PASS",
        "status": status,
        "participants": n,
        "metric_rows": len(rows),
        "prediction_records_checked": prediction_checks,
        "permutation_metric_records_checked": permutation_checks,
        "human_unlock": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--result", type=Path, required=True)
    args = parser.parse_args()
    result_bytes = args.result.read_bytes()
    result = json.loads(result_bytes)
    start_bytes = args.result.with_name("start.json").read_bytes()
    start = json.loads(start_bytes)
    plan_bytes = args.plan.read_bytes()
    require(hashlib.sha256(start_bytes).hexdigest() == result["start_file_sha256"], "Start binding")
    require(
        hashlib.sha256(plan_bytes).hexdigest() == result["plan_sha256"] == start["plan_sha256"],
        "Plan binding",
    )
    require(result["source_commit"] == start["source_commit"], "Source binding")
    plan = json.loads(plan_bytes)
    root = Path(__file__).resolve().parents[1]
    historical_plan = subprocess.check_output(
        [
            "git",
            "-C",
            str(root),
            "show",
            f"{start['source_commit']}:configs/analysis/metadata_calibration_v4_pilot.json",
        ]
    )
    require(historical_plan == plan_bytes, "Historical committed plan")
    historical_filterbank = subprocess.check_output(
        [
            "git",
            "-C",
            str(root),
            "show",
            f"{start['source_commit']}:{plan['filterbank_config']}",
        ]
    )
    require(
        hashlib.sha256(historical_filterbank).hexdigest()
        == result["filterbank_sha256"]
        == start["filterbank_sha256"],
        "Filterbank binding",
    )
    historical_tree = subprocess.check_output(
        [
            "git",
            "-C",
            str(root),
            "rev-parse",
            f"{start['source_commit']}^{{tree}}",
        ],
        text=True,
    ).strip()
    require(historical_tree == start["source_tree"], "Historical source tree")
    require(
        result["candidate_id"] == start["candidate_id"] == plan["candidate_id"], "Candidate binding"
    )
    require(
        result["evidence_role"] == start["evidence_role"] == plan["evidence_role"], "Evidence role"
    )
    expected_seed = int.from_bytes(
        hashlib.sha256(plan["seed_namespace"].encode()).digest()[:8], "big"
    )
    require(result["root_seed"] == expected_seed, "Seed derivation (no draws)")
    output = audit_result(plan, result)
    output["result_sha256"] = hashlib.sha256(result_bytes).hexdigest()
    print(json.dumps(output, sort_keys=True))


if __name__ == "__main__":
    main()
