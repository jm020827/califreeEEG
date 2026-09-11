"""Single bounded SAFE replay, with a pre-score immutable role manifest."""

import argparse
import hashlib
import io
import json
import os
import signal
import time
from pathlib import Path

import numpy as np
import torch
from run_mobilebci_prior_efficacy_v1 import summarize
from run_mobilebci_source_borrowing_v1 import validate_anchor, write_json

from cfeg.analysis.block_scaled_router import actuation, centered, train
from cfeg.analysis.mobilebci_prior_learning import require
from cfeg.analysis.mobilebci_source_borrowing import (
    ARMS,
    KS,
    Dataset,
    choose_policy,
    folds,
    outcome_rows,
    predict,
)
from cfeg.analysis.source_expert_borrowing import mix_probabilities

OUTPUTS = ("start", "roles", "choices", "journal", "models", "audit_arrays", "report")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def fresh_state():
    return {
        "schema": "cfeg.block-scaled-router-human.v1",
        "status": "RUNNING",
        "fits_attempted": 0,
        "fits_completed": 0,
        "proposals_attempted": 0,
        "proposals_completed": 0,
        "candidate_checks": 0,
        "candidate_loss_evaluations": 0,
        "completed_fit_ordinary_loss_evaluations": 0,
        "accepted": 0,
        "rejected": 0,
        "ridge_solves_attempted": 0,
        "ridge_solves_completed": 0,
        "cache_checksum_passes": 0,
        "cache_loads": 0,
        "raw_decodes": 0,
    }


def context_key(plan, k):
    return f"{plan['phase']}-{plan['outer']}-{plan.get('inner', 'all')}-{k}"


def role_manifest(dataset):
    """No features, EEG query scores or labels are used to assign roles."""
    subject_folds, plans = folds(sorted(set(dataset.subjects)))
    entries = {}
    for plan in plans:
        require(not set(plan["source"]) & set(plan["evaluation"]), "plan_overlap")
        for k in KS:
            for role, people in [("training", plan["source"]), ("evaluation", plan["evaluation"])]:
                rows = []
                for i in dataset.indices(people):
                    source, donor = dataset.bank(i, plan["source"])
                    require(len(source) == len(donor) == 12, "bank_size")
                    require(
                        all(
                            dataset.subjects[j] in plan["source"]
                            and dataset.subjects[j] != dataset.subjects[i]
                            for j in source + donor
                        ),
                        "bank_person_leak",
                    )
                    rows.append(
                        {
                            "run": i,
                            "subject": dataset.subjects[i],
                            "speed": dataset.speeds[i],
                            "source_runs": source,
                            "donor_runs": donor,
                        }
                    )
                entries[f"{context_key(plan, k)}-{role}"] = rows
    require(len(entries) == 72, "role_context_count")
    require(sum(map(len, entries.values())) == 9 * 4 * len(dataset.runs), "role_row_count")
    return {
        "schema": "cfeg.block-scaled-human-roles.v1",
        "entries": entries,
        "plans": plans,
        "subject_outer_folds": subject_folds,
        "source_k": 5,
        "recipient_k": list(KS),
        "created_before_query_scoring": True,
    }


def check_roles(records, expected):
    keys = ("run", "subject", "speed", "source_runs", "donor_runs")
    require([{k: r[k] for k in keys} for r in records] == expected, "role_manifest_mismatch")


def check_outcomes(rows, manifest, phase, arms):
    expected = {
        (plan["outer"], plan.get("inner", "all"), k, arm, record["run"])
        for plan in manifest["plans"]
        if plan["phase"] == phase
        for k in KS
        for arm in arms
        for record in manifest["entries"][context_key(plan, k) + "-evaluation"]
    }
    actual = [(r["outer"], r.get("inner", "all"), r["k"], r["arm"], r["run"]) for r in rows]
    require(len(actual) == len(expected) and set(actual) == expected, "outcome_coverage:" + phase)


def m_diagnostic(batch, router):
    actual, sham = batch["inputs"]["QM"], batch["inputs"]["SHAM"]
    with torch.no_grad():
        require(torch.equal(actual[..., :123], sham[..., :123]), "m_intervention_base_changed")
        require(
            bool((actual[:, 0, -2:] == 0).all() and (sham[:, 0, -2:] == 0).all()), "m_self_changed"
        )
        result = actuation(actual, sham, batch["probabilities"], router)
        return {
            "aux_max_change": float((actual - sham).abs().max()),
            "centered_logit_max_change": float(
                (centered(router.logits(actual)) - centered(router.logits(sham))).abs().max()
            ),
            "gate_max_change": result["gate_delta"],
            "class_margin_max_change": result["margin_delta"],
            "self_mass_mean": float(router.weights(actual)[:, 0].mean()),
            "source_mass_mean": float(router.weights(actual)[:, 1:].sum(-1).mean()),
        }


def execute(dataset, state, log, lock_choices, zero_cca, manifest, roles_sha256, steps=100):
    validate_anchor(dataset, zero_cca)
    require(manifest == role_manifest(dataset), "invalid_role_manifest")
    require(len(roles_sha256) == 64, "roles_not_published")
    state.update(
        models={},
        outer_outcomes=[],
        roles_sha256=roles_sha256,
        subject_outer_folds=manifest["subject_outer_folds"],
    )
    observed_roles, inner_rows, diagnostics = set(), [], []
    choices = None

    def attempt():
        require(state["proposals_attempted"] < 144 * steps, "proposal_budget")
        state["proposals_attempted"] += 1

    def trial(kind):
        key = "candidate_checks" if kind == "candidate" else "candidate_loss_evaluations"
        require(state[key] < 144 * steps * 8, "candidate_budget")
        state[key] += 1

    def complete(row):
        state["proposals_completed"] += 1
        state["accepted" if row["accepted"] else "rejected"] += 1

    for plan in manifest["plans"]:
        if plan["phase"] == "outer" and choices is None:
            require(
                state["fits_completed"] == 96
                and state["proposals_completed"] == 96 * steps
                and not state["outer_outcomes"],
                "inner_count_at_lock",
            )
            check_outcomes(inner_rows, manifest, "inner", ARMS)
            choices = choose_policy(inner_rows)
            lock = {
                "choices": choices,
                "locked_at_fits_completed": 96,
                "locked_at_proposals_completed": state["proposals_completed"],
                "outer_outcomes_evaluated": 0,
                "roles_sha256": roles_sha256,
            }
            state["choices_sha256"] = lock_choices(lock)
            state["choices"] = choices
            log(
                dict(
                    event="all_source_choices_locked",
                    **lock,
                    choices_sha256=state["choices_sha256"],
                )
            )
        for ki, k in enumerate(KS):
            ctx = {
                key: value for key, value in plan.items() if key not in ("source", "evaluation")
            } | {"k": k}
            key = context_key(plan, k)
            training = dataset.batch(dataset.indices(plan["source"]), plan["source"], ki)
            check_roles(training["records"], manifest["entries"][key + "-training"])
            observed_roles.add(key + "-training")
            routers = {}
            for arm in ARMS:
                current = ctx | {"arm": arm}
                state["current_fit"] = current
                require(state["fits_attempted"] < 144, "fit_budget")
                state["fits_attempted"] += 1
                log(dict(event="fit_started", **current, fit_number=state["fits_attempted"]))
                router, receipt = train(
                    training["inputs"][arm],
                    training["probabilities"],
                    training["labels"],
                    "SAFE",
                    steps=steps,
                    on_attempt=attempt,
                    on_trial=trial,
                    on_step=complete,
                )
                routers[arm] = router
                model_key = key + "-" + arm
                state["models"][model_key] = current | {
                    "source_subjects": plan["source"],
                    "router": router.state(),
                }
                if arm == "QM":
                    receipt["m_only_actuation"] = m_diagnostic(training, router)
                    diagnostics.append(current | receipt["m_only_actuation"])
                state["completed_fit_ordinary_loss_evaluations"] += receipt[
                    "ordinary_loss_evaluations"
                ]
                state["fits_completed"] += 1
                log(
                    dict(
                        event="fit_complete",
                        **current,
                        model_key=model_key,
                        source_role_key=key + "-training",
                        training=receipt,
                        fits_completed=state["fits_completed"],
                        proposals_completed=state["proposals_completed"],
                    )
                )
            require(
                all(not p.requires_grad for r in routers.values() for p in r.layer.parameters()),
                "head_not_frozen",
            )
            if plan["phase"] == "outer":
                require(
                    choices is not None and len(state["choices_sha256"]) == 64, "policy_not_locked"
                )
            log(
                dict(
                    event="four_heads_frozen_before_evaluation",
                    **ctx,
                    roles_sha256=roles_sha256,
                    choices_sha256=state.get("choices_sha256"),
                )
            )
            evaluation = dataset.batch(dataset.indices(plan["evaluation"]), plan["source"], ki)
            check_roles(evaluation["records"], manifest["entries"][key + "-evaluation"])
            observed_roles.add(key + "-evaluation")
            for arm in ARMS:
                p = predict(evaluation, routers[arm], arm)
                rows = [r | ctx | {"arm": arm} for r in outcome_rows(evaluation, p)]
                log(
                    dict(
                        event="evaluation_complete",
                        **ctx,
                        arm=arm,
                        evaluation_role_key=key + "-evaluation",
                        choices_sha256=state.get("choices_sha256"),
                        probabilities=p.tolist(),
                        predictions=rows,
                        gate=routers[arm].weights(evaluation["inputs"][arm]).tolist(),
                        m_only_actuation=m_diagnostic(evaluation, routers[arm])
                        if arm == "QM"
                        else {},
                    )
                )
                (inner_rows if plan["phase"] == "inner" else state["outer_outcomes"]).extend(rows)
            if plan["phase"] == "outer":
                p = evaluation["probabilities"]
                uniform = mix_probabilities(p, torch.full(p.shape[:2], 1 / 13, dtype=torch.float64))
                for arm, value in [("IDENTITY", p[:, 0]), ("UNIFORM", uniform)]:
                    state["outer_outcomes"].extend(
                        [r | ctx | {"arm": arm} for r in outcome_rows(evaluation, value)]
                    )
    require(observed_roles == set(manifest["entries"]), "incomplete_role_coverage")
    require(
        state["fits_completed"] == state["fits_attempted"] == len(state["models"]) == 144,
        "final_fit_count",
    )
    require(
        state["proposals_attempted"] == state["proposals_completed"] == 144 * steps,
        "final_proposal_count",
    )
    require(state["accepted"] + state["rejected"] == 144 * steps, "acceptance_count")
    require(len(state["outer_outcomes"]) == len(dataset.runs) * 24, "outer_count")
    check_outcomes(state["outer_outcomes"], manifest, "outer", (*ARMS, "IDENTITY", "UNIFORM"))
    state["outer_outcomes"].extend(zero_cca)
    state["role_contexts_verified"] = len(observed_roles)
    state["sham_feature_records_verified"] = sum(len(rows) for rows in manifest["entries"].values())
    state["summary"] = summarize(
        state["outer_outcomes"], choices, dataset, manifest["subject_outer_folds"]
    )
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
    state["source_m_actuation"] = diagnostics
    state["all_qm_source_actuated"] = all(
        d["gate_max_change"] > 1e-12 and d["class_margin_max_change"] > 1e-12 for d in diagnostics
    )
    require(len(diagnostics) == 36, "actuation_count")
    state["scientific_decision"] = (
        "RETAIN_FOR_INDEPENDENT_VALIDATION"
        if state["summary"]["provisional_statistical_retention"] and state["all_qm_source_actuated"]
        else "CLOSE_GYRO_SOURCE_ROUTING_NO_MORE_OPTIMIZER_RESCUE"
    )
    # A success status is published only by run(), after all artifacts validate.
    state["status"] = "NUMERIC_COMPLETE_AWAITING_PUBLICATION"


def validate_inputs(config):
    root = Path(__file__).resolve().parents[2]
    required = {
        str(root / p)
        for p in [
            "scripts/analysis/run_block_scaled_router_human_v1.py",
            "scripts/analysis/run_mobilebci_source_borrowing_v1.py",
            "scripts/analysis/run_mobilebci_prior_efficacy_v1.py",
            "src/cfeg/analysis/block_scaled_router.py",
            "src/cfeg/analysis/mobilebci_source_borrowing.py",
            "src/cfeg/analysis/source_expert_borrowing.py",
            "src/cfeg/analysis/mobilebci_prior_learning.py",
            "docs/block_scaled_router_human_v1_contract.md",
        ]
    }
    required.update(
        config[key] for key in ("extraction_report_path", "cca_result_path", "unit_receipt_path")
    )
    require(required <= set(config["pinned_files"]), "missing_required_pin")
    for name, digest in config["pinned_files"].items():
        require(sha(name) == digest, "pin_mismatch:" + name)


def run(config):
    paths = {key: Path(config[key + "_path"]) for key in OUTPUTS}
    require(len(set(paths.values())) == len(paths), "output_path_collision")
    partial_journal = paths["journal"].with_suffix(".partial.jsonl")
    require(all(not p.exists() for p in [*paths.values(), partial_journal]), "no_restart")
    write_json(
        paths["start"],
        {"schema": "cfeg.block-scaled-human-start.v1", "attempt": 1, "config": config},
    )
    state = fresh_state()
    state["cache_sha256"] = config["cache_sha256"]
    started = time.monotonic()
    previous_handler = signal.getsignal(signal.SIGALRM)

    def deadline(_signal, _frame):
        raise TimeoutError("safe_human_600s_deadline")

    signal.signal(signal.SIGALRM, deadline)
    signal.setitimer(signal.ITIMER_REAL, 600)
    try:
        validate_inputs(config)
        state["cache_checksum_passes"] += 1
        cache_bytes = Path(config["cache_path"]).read_bytes()
        require(hashlib.sha256(cache_bytes).hexdigest() == config["cache_sha256"], "cache_pin")
        extraction = json.loads(Path(config["extraction_report_path"]).read_text())
        require(
            extraction["status"] == "COMPLETE_FEATURE_EXTRACTION"
            and extraction["cache_sha256"] == config["cache_sha256"],
            "extraction_pin",
        )
        state["cache_loads"] += 1
        with np.load(io.BytesIO(cache_bytes), allow_pickle=False) as loaded:
            arrays = {key: loaded[key] for key in loaded.files}
        del cache_bytes
        require(
            {k: list(a.shape) for k, a in arrays.items()} == extraction["cache_shapes"],
            "cache_shapes",
        )
        dataset = Dataset(arrays, extraction["runs"])
        require(
            sorted(set(dataset.subjects)) == ["s01"] + [f"s{i:02d}" for i in range(3, 18)]
            and len(dataset.runs) == 48
            and arrays["query_labels"].shape == (48, 20),
            "human_cohort",
        )
        old = json.loads(Path(config["cca_result_path"]).read_text())
        require(old["cache_sha256"] == config["cache_sha256"], "cca_cache_identity")
        zero_cca = [r for r in old["outer_outcomes"] if r["arm"] == "ZERO_CCA"]
        validate_anchor(dataset, zero_cca)
        manifest = role_manifest(dataset)
        write_json(paths["roles"], manifest)
        roles_hash = sha(paths["roles"])
        state["runs"] = [
            {"subject": s, "speed": v} for s, v in zip(dataset.subjects, dataset.speeds)
        ]

        def solve(completed):
            key = "ridge_solves_completed" if completed else "ridge_solves_attempted"
            require(state[key] < 576, "ridge_budget")
            state[key] += 1

        dataset.fit_experts(solve)
        partial_arrays = paths["audit_arrays"].with_suffix(".partial.npz")
        with partial_arrays.open("xb") as stream:
            np.savez_compressed(stream, **arrays, bank_weights=dataset.weights.numpy())
            stream.flush()
            os.fsync(stream.fileno())
        os.link(partial_arrays, paths["audit_arrays"])
        partial_arrays.unlink()
        with partial_journal.open("x") as journal:

            def log(row):
                rendered = json.dumps(row, allow_nan=False) + "\n"
                require(journal.tell() + len(rendered.encode()) <= 96 * 1024**2, "journal_budget")
                journal.write(rendered)
                journal.flush()
                os.fsync(journal.fileno())

            def lock(value):
                write_json(paths["choices"], value)
                return sha(paths["choices"])

            execute(dataset, state, log, lock, zero_cca, manifest, roles_hash)
        require(
            state["ridge_solves_attempted"] == state["ridge_solves_completed"] == 576, "ridge_count"
        )
        require(
            sha(paths["roles"]) == roles_hash and sha(paths["choices"]) == state["choices_sha256"],
            "immutable_ledger_changed",
        )
        write_json(paths["models"], state.pop("models"))
        # Publish complete journal atomically; interrupted .partial remains visibly incomplete.
        os.link(partial_journal, paths["journal"])
        partial_journal.unlink()
        state["artifacts"] = {
            key: {"path": str(path), "sha256": sha(path), "bytes": path.stat().st_size}
            for key, path in paths.items()
            if key != "report"
        }
        state["compute_seconds_before_report"] = time.monotonic() - started
        state["interpretation"] = (
            "Exposed16people development replay, not independent validation. Gyro M is not display/brain latency. "
            "Previous negatives preserved; source-only decisions; acquired-prefix cost not total setup time."
        )
        state["status"] = "COMPLETE"
        rendered = json.dumps(state, indent=2, allow_nan=False) + "\n"
        require(
            sum(p.stat().st_size for p in paths.values() if p.exists()) + len(rendered.encode())
            <= 128 * 1024**2,
            "artifact_budget",
        )
        write_json(paths["report"], state)
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
            status="STOPPED_NO_RETRY",
            error_type=type(error).__name__,
            error=str(error)[:1000],
            compute_seconds_before_report=time.monotonic() - started,
        )
        if not paths["report"].exists():
            write_json(paths["report"], state)
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous_handler)
    print(
        json.dumps(
            {
                k: state.get(k)
                for k in (
                    "status",
                    "scientific_decision",
                    "fits_completed",
                    "proposals_completed",
                    "candidate_checks",
                    "accepted",
                    "rejected",
                    "compute_seconds_before_report",
                    "summary",
                    "error",
                )
            }
        ),
        flush=True,
    )
    return state


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.manual_seed(20260911)
    torch.use_deterministic_algorithms(True)
    run(json.loads(args.config.read_text()))
