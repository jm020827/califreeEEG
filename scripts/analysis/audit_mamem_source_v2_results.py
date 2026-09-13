"""Independent saved-model arithmetic audit; zero fits, zero MAT/feature reads.

Does not import the production engine or solve a regression. It verifies recorded
roles, ridge normal equations, predictions, matched controls and the frozen screen.
It cannot reconstruct original EEG-to-PSD/oracle/query scores from these scalars.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

ARMS = ("Q", "Q2", "QM", "SHAM")
ALL_ARMS = (*ARMS, "TARGET_ONLY", "SOURCE_ONLY", "ZERO_SHOT")
PEOPLE = [f"S{n:03}" for n in range(2, 12)]


def require(value, reason):
    if not value:
        raise ValueError(reason)


def close(actual, expected, reason, atol=1e-9):
    require(
        np.shape(actual) == np.shape(expected)
        and np.allclose(actual, expected, rtol=1e-9, atol=atol),
        reason,
    )


def audit_model(x, y, eval_x, model, saved_lambda):
    x, y, eval_x = map(np.asarray, (x, y, eval_x))
    model = {k: np.asarray(v) for k, v in model.items()}
    require(set(model) == {"mean", "scale", "coef", "intercept"}, "model_keys")
    require(
        np.isfinite(x).all() and np.isfinite(y).all() and np.isfinite(eval_x).all(),
        "finite_model_inputs",
    )
    scale = x.std(axis=0)
    scale[scale < 1e-9] = 1
    close(model["mean"], x.mean(axis=0), "source_only_mean")
    close(model["scale"], scale, "source_only_scale")
    close(model["intercept"], y.mean(axis=0), "source_only_intercept")
    z = (x - model["mean"]) / model["scale"]
    residual = (z.T @ z + 0.1 * np.eye(x.shape[1])) @ model["coef"] - z.T @ (y - model["intercept"])
    require(np.max(np.abs(residual)) < 1e-7, "ridge_normal_equation")
    prediction = np.clip(
        ((eval_x - model["mean"]) / model["scale"]) @ model["coef"] + model["intercept"], 0, 1
    )
    close(prediction, saved_lambda, "lambda_replay")
    return float(np.max(np.abs(residual))), float(np.max(np.abs(prediction - saved_lambda)))


def audit(result, summary, ledger):
    expected = [(s, k) for s in PEOPLE for k in (1, 2)]
    require(
        result["fits"] == 80 and [(r["target"], r["k"]) for r in result["folds"]] == expected,
        "exact20fold80fit",
    )
    expected_fits = [(s, k, a) for s, k in expected for a in ARMS]
    require(len(ledger) >= 160, "ledger_length")
    events = [r for r in ledger if r["kind"] in ("FIT_STARTED", "FIT_COMPLETE")]
    require(len(events) == 160, "exact160fit_events")
    for index, key in enumerate(expected_fits):
        for offset, kind in enumerate(("FIT_STARTED", "FIT_COMPLETE")):
            row = events[2 * index + offset]
            require(
                row["kind"] == kind
                and (row["target"], row["k"], row["arm"]) == key
                and row["fit_index"] == index + 1,
                "unique_ordered_fit_identity",
            )
    largest_residual = largest_lambda_error = 0.0
    rows = {(r["target"], r["k"]): r for r in result["folds"]}
    for r in result["folds"]:
        target, k = r["target"], r["k"]
        source = [s for s in PEOPLE if s != target]
        require(
            len(r["rows"]) == 45 and sorted({v["subject"] for v in r["rows"]}) == source,
            "source_rows_exclusion",
        )
        require(
            [(v["subject"], v["label"], v["k"]) for v in r["rows"]]
            == [(s, j, k) for s in source for j in range(5)],
            "source_row_identity",
        )
        require(len(r["prior_audit"]) == 9, "prior_count")
        for p, s in zip(r["prior_audit"], source, strict=True):
            require(
                p
                == {
                    "target": target,
                    "pseudo_target": s,
                    "contributors": [x for x in source if x != s],
                },
                "prior_exclusions",
            )
        train = {a: np.asarray(r["train_x"][a]) for a in ARMS}
        eval_x = {a: np.asarray(r["eval_x"][a]) for a in ARMS}
        y = np.asarray(r["oracle"])
        require(y.shape == (45, 2) and ((y >= 0) & (y <= 1)).all(), "oracle_shape_range")
        for a in ARMS:
            columns = 16 if a == "Q" else 18
            require(
                train[a].shape == (45, columns) and eval_x[a].shape == (5, columns), "input_shape"
            )
            close(train[a][:, :16], train["Q"], "common_train_input")
            close(eval_x[a][:, :16], eval_x["Q"], "common_eval_input")
            error, pred_error = audit_model(
                train[a], y, eval_x[a], r["models"][a], np.asarray(r["lambdas"][a])
            )
            largest_residual, largest_lambda_error = (
                max(largest_residual, error),
                max(largest_lambda_error, pred_error),
            )
        close(eval_x["QM"], eval_x["SHAM"], "same_true_eval_metadata")
        centered = train["QM"][:, -2:]
        donor = np.asarray(r["sham"]["donor_indices"])
        require(
            donor.shape == (45,)
            and np.issubdtype(donor.dtype, np.integer)
            and ((donor >= 0) & (donor < 45)).all(),
            "donor_scope",
        )
        close(train["SHAM"][:, -2:], centered[donor], "joint_sham_vector")
        strata = {}
        for i, item in enumerate(r["rows"]):
            strata.setdefault((item["label"], item["k"], item["event_count"]), []).append(i)
        for indices in strata.values():
            indices.sort(key=lambda i: r["rows"][i]["subject"])
            require(
                [int(donor[i]) for i in indices] == indices[1:] + indices[:1],
                "exact_conditional_cycle",
            )
        for j in range(5):
            close(
                centered[[i for i, item in enumerate(r["rows"]) if item["label"] == j]].mean(
                    axis=0
                ),
                np.zeros(2),
                "source_class_centering",
            )
        changed = np.any(np.abs(centered - centered[donor]) > 1e-9, axis=1).mean()
        close(changed, r["sham"]["changed_fraction"], "meaningful_sham_fraction")
        truth = np.asarray(r["truth"])
        require(
            truth.shape == (15,)
            and sorted(truth.tolist()) == [j for j in range(5) for _ in range(3)],
            "query_truth_coverage",
        )
        for a in ALL_ARMS:
            scores = np.asarray(r["scores"][a])
            require(scores.shape == (15, 5) and np.isfinite(scores).all(), "saved_score_shape")
            prediction = np.argmax(scores, axis=1)
            close(prediction, r["predictions"][a], "saved_argmax")
            close(np.mean(prediction == truth), r["accuracy"][a], "saved_accuracy")
        close(r["support_prefix_seconds"], r["support_prefix_samples"] / 250, "prefix_units")
        require(r["query_ready_elapsed_seconds"] is None, "offline_time_honesty")
    for s in PEOPLE:
        close(rows[s, 1]["truth"], rows[s, 2]["truth"], "same_query_labels_both_k")
        close(
            rows[s, 1]["scores"]["ZERO_SHOT"],
            rows[s, 2]["scores"]["ZERO_SHOT"],
            "same_zero_shot_both_k",
        )
    draws = np.random.default_rng(20260913).integers(0, 10, (10000, 10))
    comparisons = {}
    for name, arm, k in [
        ("QM1-Q1", "Q", 1),
        ("QM1-Q2_1", "Q2", 1),
        ("QM1-SHAM1", "SHAM", 1),
        ("QM1-Q_k2", "Q", 2),
    ]:
        delta = np.array(
            [rows[s, 1]["accuracy"]["QM"] - rows[s, k]["accuracy"][arm] for s in PEOPLE]
        )
        ci = np.quantile(delta[draws].mean(axis=1), [0.025, 0.975])
        got = summary["comparisons"][name]
        close(got["mean"], delta.mean(), "paired_mean")
        close(got["ci95"], ci, "participant_bootstrap")
        close(got["participant_deltas"], delta, "paired_deltas")
        require(got["harm_over_5pp"] == int(np.sum(delta < -0.05)), "harm_count")
        comparisons[name] = (float(delta.mean()), ci)
    cost = [
        rows[s, 2]["support_prefix_seconds"] - rows[s, 1]["support_prefix_seconds"] for s in PEOPLE
    ]
    close(summary["support_prefix_difference_seconds"], cost, "cost_difference")
    close(summary["break_even_added_setup_seconds"], cost, "conditional_setup_budget")
    require(summary["actual_query_ready_time_savings"] is None, "no_unmeasured_online_savings")
    require(
        summary["subjects"] == PEOPLE
        and summary["zero_shot_at_least_80_percent"]
        == [s for s in PEOPLE if rows[s, 1]["accuracy"]["ZERO_SHOT"] >= 0.8],
        "zero_shot_attainment",
    )
    benefit = all(
        comparisons[n][0] >= 0.01 and comparisons[n][1][0] > 0
        for n in ("QM1-Q1", "QM1-Q2_1", "QM1-SHAM1")
    )
    reduction = comparisons["QM1-Q_k2"][1][0] >= -0.02 and all(v > 0 for v in cost)
    require(
        summary["decision"]
        == (
            "PROMISING_EXPLORATORY_ONLY"
            if benefit and reduction
            else "RETIRE_UNDER_FROZEN_PROTOCOL"
        ),
        "frozen_decision",
    )
    for k in (1, 2):
        for a in ALL_ARMS:
            close(
                summary["mean_accuracy"][str(k)][a],
                np.mean([rows[s, k]["accuracy"][a] for s in PEOPLE]),
                "summary_accuracy",
            )
    return {
        "status": "PASS_SAVED_ARITHMETIC_AND_ROLES",
        "models": 80,
        "lambda_scalars": 800,
        "query_argmax_predictions": 2100,
        "sham_rows": 900,
        "fit_events": 160,
        "max_normal_equation_residual": largest_residual,
        "max_lambda_error": largest_lambda_error,
        "refits": 0,
        "MAT_or_feature_cache_reads": 0,
        "limitations": "Does not reconstruct original EEG PSD, oracle or query scores; labels remain inferred.",
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("run_directory", type=Path)
    args = parser.parse_args()
    run = args.run_directory
    require(
        run.resolve()
        == Path("/home/whwovy/califreeEEG/docs/reports/mamem_recorded_event_source_v2_run"),
        "run_scope",
    )
    terminal = json.loads((run / "real_terminal.json").read_bytes())
    require(terminal["status"] == "COMPLETE", "complete_real_attempt_required")
    summary = json.loads((run / "real_summary.json").read_bytes())
    result_path = Path(summary["result_path"])
    require(
        result_path
        == Path("/home/whwovy/data/mamem_recorded_event_source_v2/real_full_result.json")
        and result_path.is_file()
        and not result_path.is_symlink()
        and result_path.stat().st_size <= 16 * 1024**2,
        "result_scope",
    )
    raw = result_path.read_bytes()
    require(hashlib.sha256(raw).hexdigest() == summary["result_sha256"], "result_hash")
    ledger = [json.loads(line) for line in (run / "real_ledger.jsonl").read_text().splitlines()]
    output = audit(json.loads(raw), summary["summary"], ledger)
    output["result_sha256"] = summary["result_sha256"]
    print(json.dumps(output, allow_nan=False))


if __name__ == "__main__":
    main()
