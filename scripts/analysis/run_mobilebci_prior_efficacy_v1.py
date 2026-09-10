"""Finite nested participant experiment. All source choices precede outer outcomes."""

import argparse
import hashlib
import json
import os
import signal
import time
from pathlib import Path

import numpy as np
import torch

from cfeg.analysis.mobilebci_prior_learning import (
    balanced_accuracy,
    cca_scores,
    donor_indices,
    fit_head,
    prior,
    projection_scores,
    require,
    tensor,
)

ARMS = ["Q", "Q2", "QM", "SHAM"]
KS = [1, 2, 3, 5]
RIDGES = [0.01, 0.1, 1.0]


class Dataset:
    def __init__(self, arrays, runs):
        self.arrays = arrays
        self.subjects = [row["subject"] for row in runs]
        self.speeds = [row["speed"] for row in runs]

    def indices(self, subjects):
        return [i for i, subject in enumerate(self.subjects) if subject in subjects]

    def data(self, indices, ki):
        a = self.arrays
        return (
            tensor(a["support_cov"][indices, ki]),
            tensor(a["support_cross"][indices, ki]),
            tensor(a["query_cov"][indices]),
            tensor(a["query_cross"][indices]),
            tensor(a["reference_gram"][indices]),
            torch.as_tensor(a["query_labels"][indices], dtype=torch.long),
        )

    def inputs(self, indices, source, ki, arm):
        a = self.arrays
        base = np.concatenate((a["q"][indices, ki], a["common"][indices, ki]), axis=-1)
        audit = {}
        if arm == "Q":
            return tensor(base), audit
        aux = a["q2"][indices, ki] if arm == "Q2" else a["m"][indices, ki]
        if arm == "SHAM":
            donors, distances = donor_indices(
                source, indices, self.subjects, self.speeds, a["common"][:, ki]
            )
            aux = a["m"][donors, ki]
            differences = np.max(np.abs(aux - a["m"][indices, ki]), axis=-1)
            require(bool((differences > 0).all()), "sham_unchanged")
            audit = {
                "donors": [self.subjects[i] for i in donors],
                "distances": distances,
                "actual_change_min": float(differences.min()),
                "unique_donor_subjects": len({self.subjects[i] for i in donors}),
            }
        return tensor(np.concatenate((base, aux), axis=-1)), audit


def evaluate(dataset, indices, ki, ridge, diagonal):
    data = dataset.data(indices, ki)
    with torch.no_grad():
        scores, floor_count = projection_scores(*data[:5], diagonal, ridge)
    predictions = scores.argmax(-1).numpy()
    labels = data[5].numpy()
    accuracies = balanced_accuracy(predictions, labels)
    return [
        {
            "subject": dataset.subjects[i],
            "speed": dataset.speeds[i],
            "balanced_accuracy": float(accuracies[j]),
            "predictions": predictions[j].tolist(),
            "labels": labels[j].tolist(),
        }
        for j, i in enumerate(indices)
    ], floor_count


def pipeline(dataset, source, validation, ki, ridge, context, state, log):
    train_data = dataset.data(source, ki)
    heads, outputs = {}, {}
    qtrain = qvalid = None
    for arm in ARMS:
        state["current_fit"] = context | {"arm": arm, "k": KS[ki], "ridge": ridge}
        require(state["fits_attempted"] < 336, "fit_budget")
        state["fits_attempted"] += 1
        log({"event": "fit_started", **state["current_fit"], "fit_number": state["fits_attempted"]})
        training_inputs, training_sham = dataset.inputs(source, source, ki, arm)
        valid_inputs, valid_sham = dataset.inputs(validation, source, ki, arm)

        def step():
            state["optimizer_steps"] += 1
            require(state["optimizer_steps"] <= 33600, "step_budget")

        head, diagnostics = fit_head(training_inputs, train_data, ridge, qtrain, on_step=step)
        heads[arm] = head
        train_diagonal = prior(head, training_inputs, qtrain)
        valid_diagonal = prior(head, valid_inputs, qvalid)
        if arm == "QM":
            swapped_inputs, swap_audit = dataset.inputs(source, source, ki, "SHAM")
            with torch.no_grad():
                swapped_diagonal = prior(head, swapped_inputs, qtrain)
                actual_scores, _ = projection_scores(*train_data[:5], train_diagonal, ridge)
                swapped_scores, _ = projection_scores(*train_data[:5], swapped_diagonal, ridge)
            diagnostics["metadata_only_intervention"] = {
                "source_prior_max_abs_change": float(
                    (swapped_diagonal - train_diagonal).abs().max()
                ),
                "source_score_max_abs_change": float((swapped_scores - actual_scores).abs().max()),
                "donor_audit": swap_audit,
            }
        if arm == "Q":
            qtrain, qvalid = train_diagonal.detach(), valid_diagonal.detach()
        predictions, floors = evaluate(dataset, validation, ki, ridge, valid_diagonal)
        outputs[arm] = predictions
        record = {
            "event": "fit_complete",
            **state["current_fit"],
            "diagnostics": diagnostics,
            "source_subjects": sorted({dataset.subjects[i] for i in source}),
            "validation_subjects": sorted({dataset.subjects[i] for i in validation}),
            "training_sham": training_sham,
            "validation_sham": valid_sham,
            "validation_energy_floor_count": floors,
            "predictions": predictions,
        }
        state["fits_completed"] += 1
        log(record)
        print(
            json.dumps(
                {
                    "completed": state["fits_completed"],
                    **state["current_fit"],
                    "mean_bacc": float(np.mean([row["balanced_accuracy"] for row in predictions])),
                }
            ),
            flush=True,
        )
    return outputs, heads


def choose_source(inner):
    choices = {}
    for outer in range(3):
        selected, means = {}, {arm: {} for arm in ARMS}
        for k in KS:
            rows = [row for row in inner if row["outer"] == outer and row["k"] == k]
            candidates = [
                (
                    float(
                        np.mean(
                            [
                                v["balanced_accuracy"]
                                for row in rows
                                if row["ridge"] == ridge
                                for v in row["outputs"]["Q"]
                            ]
                        )
                    ),
                    ridge,
                )
                for ridge in RIDGES
            ]
            _, selected[str(k)] = max(candidates)
            for arm in ARMS:
                means[arm][str(k)] = float(
                    np.mean(
                        [
                            v["balanced_accuracy"]
                            for row in rows
                            if row["ridge"] == selected[str(k)]
                            for v in row["outputs"][arm]
                        ]
                    )
                )
        policy = {}
        for arm in ARMS:
            qualified = [k for k in KS if means[arm][str(k)] >= 0.8]
            policy[arm] = {
                "k": min(qualified) if qualified else 5,
                "source_target_unmet": not qualified,
            }
        choices[str(outer)] = {"ridge_by_k": selected, "inner_oof_means": means, "policy": policy}
    return choices


def summarize(outer_rows, choices, dataset, subject_folds):
    subjects = sorted(set(dataset.subjects))
    rng = np.random.default_rng(20260911)
    resamples = rng.integers(0, len(subjects), size=(2000, len(subjects)))

    def ci(values):
        return np.quantile(np.asarray(values)[resamples].mean(1), [0.025, 0.975]).tolist()

    def subject_values(arm, budgets):
        return np.array(
            [
                np.mean(
                    [
                        row["balanced_accuracy"]
                        for row in outer_rows
                        if row["subject"] == subject and row["arm"] == arm and row["k"] in budgets
                    ]
                )
                for subject in subjects
            ]
        )

    low = {arm: subject_values(arm, [1, 2]) for arm in ARMS + ["IDENTITY"]}
    high = {arm: subject_values(arm, [3, 5]) for arm in ARMS}
    contrasts = {
        arm: {
            "mean_difference": float((low["QM"] - low[arm]).mean()),
            "ci95": ci(low["QM"] - low[arm]),
        }
        for arm in ["Q", "Q2", "SHAM", "IDENTITY"]
    }
    policy, policy_subject = {}, {}
    for arm in ARMS:
        accuracy, cost, ready, reach = [], [], [], []
        for subject in subjects:
            k = choices[str(subject_folds[subject])]["policy"][arm]["k"]
            rows = [
                row
                for row in outer_rows
                if row["subject"] == subject and row["arm"] == arm and row["k"] == k
            ]
            indices = dataset.indices([subject])
            accuracy.append(np.mean([row["balanced_accuracy"] for row in rows]))
            reach.append(np.mean([row["balanced_accuracy"] >= 0.8 for row in rows]))
            cost.append(dataset.arrays["common"][indices, KS.index(k), 3].mean())
            ready.append(dataset.arrays["common"][indices, KS.index(k), 4].mean())
        policy_subject[arm] = {
            "accuracy": np.asarray(accuracy),
            "cost": np.asarray(cost),
            "reach": np.asarray(reach),
        }
        policy[arm] = {
            "mean_balanced_accuracy": float(np.mean(accuracy)),
            "mean_acquired_prefix_trials": float(np.mean(cost)),
            "mean_recording_ready_seconds": float(np.mean(ready)),
            "run_target_reach_rate": float(np.mean(reach)),
            "source_unmet_folds": [
                fold
                for fold, entry in choices.items()
                if entry["policy"][arm]["source_target_unmet"]
            ],
        }
    pdelta = policy_subject["QM"]["accuracy"] - policy_subject["Q"]["accuracy"]
    cost_reduction = 1 - policy_subject["QM"]["cost"].mean() / policy_subject["Q"]["cost"].mean()
    identity_at_qm = np.array(
        [
            np.mean(
                [
                    row["balanced_accuracy"]
                    for row in outer_rows
                    if row["subject"] == subject
                    and row["arm"] == "IDENTITY"
                    and row["k"] == choices[str(subject_folds[subject])]["policy"]["QM"]["k"]
                ]
            )
            for subject in subjects
        ]
    )
    cca = float(
        np.mean([row["balanced_accuracy"] for row in outer_rows if row["arm"] == "ZERO_CCA"])
    )
    criteria = {
        "low_k_qm_minus_q_at_least_2pp": contrasts["Q"]["mean_difference"] >= 0.02,
        "low_k_qm_beats_q2": contrasts["Q2"]["mean_difference"] > 0,
        "low_k_qm_beats_sham": contrasts["SHAM"]["mean_difference"] > 0,
        "low_k_harm_ci_lower_above_minus2pp": contrasts["Q"]["ci95"][0] > -0.02,
        "policy_qm_accuracy_at_least_80pct": policy["QM"]["mean_balanced_accuracy"] >= 0.8,
        "policy_harm_ci_lower_above_minus2pp": ci(pdelta)[0] > -0.02,
        "policy_cost_reduction_at_least_10pct": cost_reduction >= 0.1,
        "policy_reach_not_lower_than_q": policy["QM"]["run_target_reach_rate"]
        >= policy["Q"]["run_target_reach_rate"],
        "high_k_mean_harm_at_most_1pp": float((high["QM"] - high["Q"]).mean()) >= -0.01,
        "low_k_large_harm_at_most_three_subjects": int(((low["QM"] - low["Q"]) < -0.05).sum()) <= 3,
        "qm_policy_beats_zero_calibration_cca": policy["QM"]["mean_balanced_accuracy"] > cca,
    }
    criteria = {key: bool(value) for key, value in criteria.items()}
    return {
        "low_k_means": {arm: float(value.mean()) for arm, value in low.items()},
        "low_k_qm_contrasts": contrasts,
        "high_k_means": {arm: float(value.mean()) for arm, value in high.items()},
        "source_chosen_policy": policy,
        "policy_qm_minus_q": {
            "mean_difference": float(pdelta.mean()),
            "ci95": ci(pdelta),
            "prefix_cost_reduction_fraction": float(cost_reduction),
        },
        "zero_calibration_cca_mean_bacc": cca,
        "identity_at_qm_policy_budget_mean_bacc": float(identity_at_qm.mean()),
        "qm_policy_beats_same_budget_identity": bool(
            (policy_subject["QM"]["accuracy"] - identity_at_qm).mean() > 0
        ),
        "criteria": criteria,
        "provisional_statistical_retention": all(criteria.values()),
        "interpretation": "Exploratory fixed cross-fitted replay; final retention additionally requires nonzero source actuation and valid SHAM. Not total setup-time savings.",
    }


def run(config):
    cache = Path(config["cache_path"])
    report_bytes = Path(config["extraction_report_path"]).read_bytes()
    require(
        hashlib.sha256(report_bytes).hexdigest() == config["extraction_report_sha256"],
        "extraction_report_pin",
    )
    extraction = json.loads(report_bytes)
    require(extraction["status"] == "COMPLETE_FEATURE_EXTRACTION", "extraction_incomplete")
    require(
        hashlib.sha256(cache.read_bytes()).hexdigest()
        == config["cache_sha256"]
        == extraction["cache_sha256"],
        "cache_pin",
    )
    paths = {
        key: Path(config[key])
        for key in ["report_path", "journal_path", "choices_path", "models_path"]
    }
    require(all(not path.exists() for path in paths.values()), "no_restart")
    with np.load(cache, allow_pickle=False) as loaded:
        arrays = {key: loaded[key] for key in loaded.files}
    require(all(np.isfinite(a).all() for a in arrays.values()), "nonfinite_cache")
    dataset = Dataset(arrays, extraction["runs"])
    subjects = sorted(set(dataset.subjects))
    require(len(subjects) == 16 and len(dataset.subjects) == 48, "cohort_size")
    subject_folds = {subject: i % 3 for i, subject in enumerate(subjects)}
    state = {
        "schema": "cfeg.mobilebci-prior-efficacy.v1",
        "status": "RUNNING",
        "fits_attempted": 0,
        "fits_completed": 0,
        "optimizer_steps": 0,
        "cache_sha256": config["cache_sha256"],
        "subject_outer_folds": subject_folds,
        "outer_outcomes": [],
        "models": {},
    }
    started = time.monotonic()
    signal.setitimer(signal.ITIMER_REAL, 3600)
    try:
        with paths["journal_path"].open("x") as stream:

            def log(row):
                rendered = json.dumps(row, allow_nan=False) + "\n"
                require(
                    stream.tell() + len(rendered.encode()) <= 512 * 1024 * 1024,
                    "journal_output_budget",
                )
                stream.write(rendered)
                stream.flush()
                os.fsync(stream.fileno())

            inner_rows = []
            for outer in range(3):
                source_subjects = [s for s in subjects if subject_folds[s] != outer]
                for inner in range(2):
                    source = dataset.indices(
                        [s for i, s in enumerate(source_subjects) if i % 2 != inner]
                    )
                    validation = dataset.indices(
                        [s for i, s in enumerate(source_subjects) if i % 2 == inner]
                    )
                    for ki, k in enumerate(KS):
                        for ridge in RIDGES:
                            outputs, _ = pipeline(
                                dataset,
                                source,
                                validation,
                                ki,
                                ridge,
                                {"phase": "inner", "outer": outer, "inner": inner},
                                state,
                                log,
                            )
                            inner_rows.append(
                                {
                                    "outer": outer,
                                    "inner": inner,
                                    "k": k,
                                    "ridge": ridge,
                                    "outputs": outputs,
                                }
                            )
            choices = choose_source(inner_rows)
            with paths["choices_path"].open("x") as stream_choices:
                json.dump(
                    {
                        "choices": choices,
                        "locked_at_fits_completed": state["fits_completed"],
                        "outer_outcomes_evaluated": 0,
                    },
                    stream_choices,
                    indent=2,
                )
                stream_choices.flush()
                os.fsync(stream_choices.fileno())
            state["choices"] = choices
            log(
                {
                    "event": "all_source_choices_locked",
                    "fits_completed": state["fits_completed"],
                    "choices": choices,
                }
            )
            for outer in range(3):
                source = dataset.indices([s for s in subjects if subject_folds[s] != outer])
                test = dataset.indices([s for s in subjects if subject_folds[s] == outer])
                for ki, k in enumerate(KS):
                    ridge = choices[str(outer)]["ridge_by_k"][str(k)]
                    outputs, heads = pipeline(
                        dataset,
                        source,
                        test,
                        ki,
                        ridge,
                        {"phase": "outer", "outer": outer},
                        state,
                        log,
                    )
                    state["models"][f"outer{outer}_k{k}"] = {
                        arm: head.serialize() for arm, head in heads.items()
                    }
                    identity, _ = evaluate(
                        dataset, test, ki, ridge, torch.ones((len(test), 9), dtype=torch.float64)
                    )
                    outputs["IDENTITY"] = identity
                    for arm, rows in outputs.items():
                        state["outer_outcomes"].extend(
                            [
                                row | {"arm": arm, "k": k, "outer": outer, "ridge": ridge}
                                for row in rows
                            ]
                        )
                data = dataset.data(test, 0)
                with torch.no_grad():
                    predictions = cca_scores(data[2], data[3], data[4]).argmax(-1).numpy()
                accuracies = balanced_accuracy(predictions, data[5].numpy())
                state["outer_outcomes"].extend(
                    [
                        {
                            "subject": dataset.subjects[i],
                            "speed": dataset.speeds[i],
                            "arm": "ZERO_CCA",
                            "k": 0,
                            "outer": outer,
                            "balanced_accuracy": float(accuracies[j]),
                            "predictions": predictions[j].tolist(),
                            "labels": data[5][j].tolist(),
                        }
                        for j, i in enumerate(test)
                    ]
                )
            state["summary"] = summarize(state["outer_outcomes"], choices, dataset, subject_folds)
            require(
                state["fits_completed"] == 336 and state["optimizer_steps"] == 33600,
                "execution_counts",
            )
            state["status"] = "COMPLETE_FROZEN_EFFICACY_EXPERIMENT"
    except (
        ValueError,
        OSError,
        RuntimeError,
        MemoryError,
        KeyError,
        TypeError,
        IndexError,
        OverflowError,
    ) as error:
        state.update(
            status="STOPPED_NO_RETRY", error_type=type(error).__name__, error=str(error)[:512]
        )
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
    state["seconds"] = round(time.monotonic() - started, 6)
    models = state.pop("models")
    rendered_models = json.dumps(models, allow_nan=False)
    rendered_state = json.dumps(state, indent=2, allow_nan=False)
    require(
        len(rendered_models.encode()) + len(rendered_state.encode()) <= 512 * 1024 * 1024,
        "final_output_budget",
    )
    with paths["models_path"].open("x") as stream:
        stream.write(rendered_models)
        stream.flush()
        os.fsync(stream.fileno())
    with paths["report_path"].open("x") as stream:
        stream.write(rendered_state)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    print(
        json.dumps(
            {
                key: state.get(key)
                for key in [
                    "status",
                    "fits_completed",
                    "optimizer_steps",
                    "seconds",
                    "summary",
                    "error",
                ]
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    config = json.loads(parser.parse_args().config.read_text())
    torch.set_num_threads(1)
    torch.manual_seed(20260911)
    torch.use_deterministic_algorithms(True)

    def deadline(_sig, _frame):
        raise TimeoutError("training_deadline")

    signal.signal(signal.SIGALRM, deadline)
    run(config)
