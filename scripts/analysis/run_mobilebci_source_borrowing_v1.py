"""One bounded human replay; generated integration calls execute() without this reader."""

import argparse
import hashlib
import json
import os
import signal
import tempfile
import time
from pathlib import Path

import numpy as np
import torch
from run_mobilebci_prior_efficacy_v1 import summarize

from cfeg.analysis.mobilebci_prior_learning import require
from cfeg.analysis.mobilebci_source_borrowing import (
    ARMS,
    KS,
    Dataset,
    actuation,
    choose_policy,
    folds,
    outcome_rows,
    predict,
    train_router,
)
from cfeg.analysis.source_expert_borrowing import mix_probabilities


def write_json(path, value):
    path = Path(path)
    rendered = json.dumps(value, indent=2, allow_nan=False) + "\n"
    fd, partial = tempfile.mkstemp(prefix=path.name + ".partial-", dir=path.parent)
    with os.fdopen(fd, "w") as stream:
        stream.write(rendered)
        stream.flush()
        os.fsync(stream.fileno())
    # Atomic no-overwrite publication. An interrupted partial is preserved, not a report.
    os.link(partial, path)
    Path(partial).unlink()


def validate_anchor(dataset, zero_cca):
    require(len(zero_cca) == len(dataset.runs), "cca_anchor_run_count")
    keys = [(r["subject"], r["speed"]) for r in zero_cca]
    require(len(set(keys)) == len(keys) and set(keys) == set(dataset.lookup), "cca_anchor_run_set")
    for row in zero_cca:
        index = dataset.lookup[row["subject"], row["speed"]]
        require(
            row["labels"] == dataset.arrays["query_labels"][index].tolist(),
            "cca_anchor_label_mismatch",
        )


def execute(dataset, state, log, lock_choices, zero_cca, steps=100):
    validate_anchor(dataset, zero_cca)
    subjects = sorted(set(dataset.subjects))
    subject_folds, plans = folds(subjects)
    state["subject_outer_folds"] = subject_folds
    state["models"] = {}
    state["outer_outcomes"] = []
    inner_rows, m_diagnostics = [], []
    choices = None

    def count_step():
        state["optimizer_steps"] += 1
        require(state["optimizer_steps"] <= 144 * steps, "step_budget")

    for plan in plans:
        require(
            not set(plan["source"]).intersection(plan["evaluation"]), "source_evaluation_overlap"
        )
        if plan["phase"] == "outer" and choices is None:
            require(
                state["fits_completed"] == 96 and state["optimizer_steps"] == 96 * steps,
                "inner_count_at_lock",
            )
            choices = choose_policy(inner_rows)
            lock = {
                "choices": choices,
                "locked_at_fits_completed": 96,
                "locked_at_optimizer_steps": state["optimizer_steps"],
                "outer_outcomes_evaluated": 0,
            }
            lock_hash = lock_choices(lock)
            state["choices_sha256"] = lock_hash
            state["choices"] = choices
            log({"event": "all_source_choices_locked", **lock, "choices_sha256": lock_hash})
        for ki, k in enumerate(KS):
            context = {
                key: value for key, value in plan.items() if key not in ["source", "evaluation"]
            } | {"k": k}
            source_indices = dataset.indices(plan["source"])
            evaluation_indices = dataset.indices(plan["evaluation"])
            training = dataset.batch(source_indices, plan["source"], ki)
            routers = {}
            for arm in ARMS:
                current = context | {"arm": arm}
                state["current_fit"] = current
                require(state["fits_attempted"] < 144, "fit_budget")
                state["fits_attempted"] += 1
                log({"event": "fit_started", **current, "fit_number": state["fits_attempted"]})
                router, training_log = train_router(
                    training["inputs"][arm],
                    training["probabilities"],
                    training["labels"],
                    steps=steps,
                    on_step=count_step,
                )
                routers[arm] = router
                model_key = f"{plan['phase']}-{plan['outer']}-{plan.get('inner', 'all')}-{k}-{arm}"
                state["models"][model_key] = current | {
                    "source_subjects": plan["source"],
                    "router": router.state(),
                }
                if arm == "QM":
                    diag = actuation(training, router)
                    training_log["m_only_actuation"] = diag
                    m_diagnostics.append(current | diag)
                state["fits_completed"] += 1
                log(
                    {
                        "event": "fit_complete",
                        **current,
                        "fits_completed": state["fits_completed"],
                        "optimizer_steps": state["optimizer_steps"],
                        "source_subjects": plan["source"],
                        "evaluation_subjects": plan["evaluation"],
                        "training": training_log,
                        "source_roles": training["records"],
                        "source_energy_floor_count": training["energy_floors"],
                        "model_key": model_key,
                    }
                )
            # All four heads in this context are frozen before evaluation expert scores or predictions.
            require(
                all(not p.requires_grad for r in routers.values() for p in r.layer.parameters()),
                "head_not_frozen",
            )
            evaluation = dataset.batch(evaluation_indices, plan["source"], ki)
            for arm in ARMS:
                p = predict(evaluation, routers[arm], arm)
                rows = [row | context | {"arm": arm} for row in outcome_rows(evaluation, p)]
                diagnostics = actuation(evaluation, routers[arm]) if arm == "QM" else {}
                log(
                    {
                        "event": "evaluation_complete",
                        **context,
                        "arm": arm,
                        "choices_sha256": state.get("choices_sha256"),
                        "evaluation_roles": evaluation["records"],
                        "energy_floor_count": evaluation["energy_floors"],
                        "gate": routers[arm].weights(evaluation["inputs"][arm]).tolist(),
                        "m_only_actuation": diagnostics,
                        "predictions": rows,
                    }
                )
                if plan["phase"] == "inner":
                    inner_rows.extend(rows)
                else:
                    state["outer_outcomes"].extend(rows)
            if plan["phase"] == "outer":
                p = evaluation["probabilities"]
                uniform = mix_probabilities(
                    p, torch.full((len(evaluation_indices), 13), 1 / 13, dtype=torch.float64)
                )
                for arm, value in [("IDENTITY", p[:, 0]), ("UNIFORM", uniform)]:
                    state["outer_outcomes"].extend(
                        [row | context | {"arm": arm} for row in outcome_rows(evaluation, value)]
                    )
    require(
        state["fits_completed"] == 144 and state["optimizer_steps"] == 144 * steps,
        "final_fit_counts",
    )
    state["outer_outcomes"].extend(zero_cca)
    state["summary"] = summarize(state["outer_outcomes"], choices, dataset, subject_folds)
    state["summary"]["uniform_by_k"] = {
        str(k): float(
            np.mean(
                [
                    r["balanced_accuracy"]
                    for r in state["outer_outcomes"]
                    if r["arm"] == "UNIFORM" and r["k"] == k
                ]
            )
        )
        for k in KS
    }
    state["source_m_actuation"] = m_diagnostics
    actuated = all(
        d["gate_max_change"] > 1e-12 and d["class_margin_max_change"] > 1e-12 for d in m_diagnostics
    )
    state["valid_sham"] = True
    state["all_qm_source_actuated"] = actuated
    state["status"] = (
        "RETAIN_FOR_INDEPENDENT_VALIDATION"
        if (state["summary"]["provisional_statistical_retention"] and actuated)
        else "CLOSED_NEGATIVE"
    )
    state["actuation_decision"] = "DEMONSTRATED" if actuated else "NO_DEMONSTRATED_M_ACTUATION"


def run(config):
    keys = [
        "report_path",
        "journal_path",
        "choices_path",
        "models_path",
        "audit_arrays_path",
        "start_path",
    ]
    paths = {key: Path(config[key]) for key in keys}
    require(all(not path.exists() for path in paths.values()), "no_restart")
    write_json(
        paths["start_path"],
        {"schema": "cfeg.source-borrowing-start.v1", "attempt": 1, "config": config},
    )
    state = {
        "schema": "cfeg.mobilebci-source-borrowing.v1",
        "status": "RUNNING",
        "fits_attempted": 0,
        "fits_completed": 0,
        "optimizer_steps": 0,
        "ridge_solves_attempted": 0,
        "ridge_solves_completed": 0,
        "cache_checksum_passes": 0,
        "cache_loads": 0,
        "raw_decodes": 0,
        "cache_sha256": config["cache_sha256"],
    }
    started = time.monotonic()
    signal.setitimer(signal.ITIMER_REAL, 600)
    try:
        root = Path(__file__).resolve().parents[2]
        required = {
            str(root / name)
            for name in [
                "scripts/analysis/run_mobilebci_source_borrowing_v1.py",
                "scripts/analysis/run_mobilebci_prior_efficacy_v1.py",
                "src/cfeg/analysis/mobilebci_source_borrowing.py",
                "src/cfeg/analysis/source_expert_borrowing.py",
                "src/cfeg/analysis/mobilebci_prior_learning.py",
                "docs/mobilebci_source_borrowing_human_v1_contract.md",
            ]
        }
        required.update(
            config[key]
            for key in ["extraction_report_path", "cca_result_path", "unit_receipt_path"]
        )
        require(required <= set(config["pinned_files"]), "missing_required_pin")
        for name, digest in config["pinned_files"].items():
            require(
                hashlib.sha256(Path(name).read_bytes()).hexdigest() == digest,
                "pin_mismatch:" + name,
            )
        cache = Path(config["cache_path"])
        state["cache_checksum_passes"] += 1
        require(
            hashlib.sha256(cache.read_bytes()).hexdigest() == config["cache_sha256"], "cache_pin"
        )
        extraction = json.loads(Path(config["extraction_report_path"]).read_text())
        require(
            extraction["status"] == "COMPLETE_FEATURE_EXTRACTION"
            and extraction["cache_sha256"] == config["cache_sha256"],
            "extraction_pin",
        )
        state["cache_loads"] += 1
        with np.load(cache, allow_pickle=False) as loaded:
            arrays = {key: loaded[key] for key in loaded.files}
        require(
            {key: list(a.shape) for key, a in arrays.items()} == extraction["cache_shapes"],
            "extraction_shapes",
        )
        dataset = Dataset(arrays, extraction["runs"])
        require(
            sorted(set(dataset.subjects)) == ["s01"] + [f"s{i:02d}" for i in range(3, 18)],
            "cohort_members",
        )
        require(
            len(dataset.runs) == 48 and arrays["query_labels"].shape == (48, 20),
            "cohort_dimensions",
        )
        state["runs"] = [
            {"subject": s, "speed": v} for s, v in zip(dataset.subjects, dataset.speeds)
        ]

        def solve_count(completed):
            key = "ridge_solves_completed" if completed else "ridge_solves_attempted"
            state[key] += 1
            require(state[key] <= 576, "ridge_budget")

        dataset.fit_experts(solve_count)
        with paths["audit_arrays_path"].open("xb") as stream:
            np.savez_compressed(stream, **arrays, bank_weights=dataset.weights.numpy())
        state["audit_arrays_sha256"] = hashlib.sha256(
            paths["audit_arrays_path"].read_bytes()
        ).hexdigest()
        old = json.loads(Path(config["cca_result_path"]).read_text())
        require(old["cache_sha256"] == config["cache_sha256"], "cca_cache_identity")
        zero_cca = [r for r in old["outer_outcomes"] if r["arm"] == "ZERO_CCA"]
        with paths["journal_path"].open("x") as journal:

            def log(row):
                rendered = json.dumps(row, allow_nan=False) + "\n"
                require(
                    journal.tell() + len(rendered.encode()) < 64 * 1024 * 1024, "journal_budget"
                )
                journal.write(rendered)
                journal.flush()
                os.fsync(journal.fileno())

            def lock(value):
                write_json(paths["choices_path"], value)
                return hashlib.sha256(paths["choices_path"].read_bytes()).hexdigest()

            execute(dataset, state, log, lock, zero_cca)
        models = state["models"]
        write_json(paths["models_path"], models)
        state["models_sha256"] = hashlib.sha256(paths["models_path"].read_bytes()).hexdigest()
        del state["models"]
        state["compute_seconds"] = time.monotonic() - started
        state["interpretation"] = (
            "New cross-fitted development replay of already exposed16people. Not independent confirmation or total setup-time savings. Previous336-fit negative unchanged."
        )
        rendered = json.dumps(state, indent=2, allow_nan=False) + "\n"
        total = sum(path.stat().st_size for path in paths.values() if path.exists()) + len(
            rendered.encode()
        )
        require(total <= 128 * 1024 * 1024, "artifact_budget")
        write_json(paths["report_path"], state)
    except (
        ValueError,
        RuntimeError,
        OSError,
        MemoryError,
        KeyError,
        IndexError,
        TypeError,
    ) as error:
        signal.setitimer(signal.ITIMER_REAL, 0)
        state.update(
            status="VALIDITY_INCONCLUSIVE"
            if "sham_aux_unchanged" in str(error)
            else "STOPPED_NO_RETRY",
            error_type=type(error).__name__,
            error=str(error)[:1000],
            compute_seconds=time.monotonic() - started,
        )
        if not paths["report_path"].exists():
            write_json(paths["report_path"], state)
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
    try:
        print(
            json.dumps(
                {
                    key: state.get(key)
                    for key in [
                        "status",
                        "fits_completed",
                        "optimizer_steps",
                        "ridge_solves_completed",
                        "compute_seconds",
                        "summary",
                        "error",
                    ]
                }
            ),
            flush=True,
        )
    except BrokenPipeError:
        pass
    return state


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.manual_seed(20260911)
    torch.use_deterministic_algorithms(True)

    def deadline(_signal, _frame):
        raise TimeoutError("source_borrowing_600s_deadline")

    signal.signal(signal.SIGALRM, deadline)
    run(json.loads(args.config.read_text()))
