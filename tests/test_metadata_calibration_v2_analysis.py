from __future__ import annotations

import json
import stat
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from cfeg.analysis import metadata_calibration_v2 as analysis_module
from cfeg.analysis.metadata_calibration_v2 import (
    DEFAULT_V2_ALLOCATION_PATH,
    DEFAULT_V2_PLAN_PATH,
    V2_BUDGETS,
    V2_CANDIDATE_ID,
    V2_EAUC_WEIGHTS,
    V2_SCORE_REQUEST_SCHEMA,
    V2CandidateSpec,
    V2ContractBinding,
    build_v2_preparation_receipt,
    derive_v2_participant_eauc_deltas,
    evaluate_v2_independent_aq_gate,
    one_sided_paired_sensitivity,
    one_sided_t_lower_bound,
    reduce_v2_participant_balanced_accuracy,
    reject_forbidden_v2_data_path,
    score_v2_external_request,
    select_v2_development_candidate,
    v2_candidate_grid,
    validate_v2_external_allocation_receipt,
    validate_v2_plan_and_allocation,
    write_v2_completion_receipt_exclusive,
)

REPO = Path(__file__).resolve().parents[1]


@pytest.fixture(autouse=True)
def _exercise_retired_algorithms_without_operational_authority(monkeypatch) -> None:
    monkeypatch.setattr(
        analysis_module,
        "deny_v2_terminal_operational_action",
        lambda _action: None,
    )


@pytest.fixture(scope="module")
def contract() -> V2ContractBinding:
    return validate_v2_plan_and_allocation()


def _participant_delta_frame(
    contract: V2ContractBinding,
    *,
    cohort: str,
    candidates: tuple[V2CandidateSpec, ...],
    gains: Callable[[V2CandidateSpec, str, str], tuple[float, float]],
) -> pd.DataFrame:
    records: list[dict[str, object]] = []
    for candidate in candidates:
        for dataset_id in ("beta_v1", "dong2023_v1"):
            asset = contract.asset_binding(dataset_id)
            for subject_id in contract.subject_ids(dataset_id, cohort):  # type: ignore[arg-type]
                delta_k1, delta_k3 = gains(candidate, dataset_id, subject_id)
                a0 = {budget: 0.50 for budget in V2_BUDGETS}
                aq = {0: 0.50, 1: 0.50 + delta_k1, 3: 0.50 + delta_k3}
                a0_eauc = sum(V2_EAUC_WEIGHTS[k] * a0[k] for k in V2_BUDGETS)
                aq_eauc = sum(V2_EAUC_WEIGHTS[k] * aq[k] for k in V2_BUDGETS)
                records.append(
                    {
                        "candidate_id": V2_CANDIDATE_ID,
                        "plan_sha256": contract.plan_sha256,
                        "allocation_sha256": contract.allocation_sha256,
                        **asset,
                        "dataset_id": dataset_id,
                        "cohort": cohort,
                        **candidate.as_dict(),
                        "subject_id": subject_id,
                        "a0_k0": a0[0],
                        "aq_k0": aq[0],
                        "delta_k0": aq[0] - a0[0],
                        "a0_k1": a0[1],
                        "aq_k1": aq[1],
                        "delta_k1": aq[1] - a0[1],
                        "a0_k3": a0[3],
                        "aq_k3": aq[3],
                        "delta_k3": aq[3] - a0[3],
                        "a0_eauc": a0_eauc,
                        "aq_eauc": aq_eauc,
                        "eauc_delta": aq_eauc - a0_eauc,
                    }
                )
    return pd.DataFrame.from_records(records)


def _prediction_frame(
    contract: V2ContractBinding,
    *,
    candidate: V2CandidateSpec,
    cohort: str,
) -> pd.DataFrame:
    records: list[dict[str, object]] = []
    for dataset_id in ("beta_v1", "dong2023_v1"):
        asset = contract.asset_binding(dataset_id)
        for subject_id in contract.subject_ids(dataset_id, cohort):  # type: ignore[arg-type]
            for role in ("A0", "A_Q"):
                for budget in V2_BUDGETS:
                    for label in range(40):
                        prediction = label
                        if role == "A_Q" and budget == 1 and label == 0:
                            prediction = 1
                        records.append(
                            {
                                "candidate_id": V2_CANDIDATE_ID,
                                "plan_sha256": contract.plan_sha256,
                                "allocation_sha256": contract.allocation_sha256,
                                **asset,
                                "dataset_id": dataset_id,
                                "cohort": cohort,
                                **candidate.as_dict(),
                                "subject_id": subject_id,
                                "role": role,
                                "budget": budget,
                                "query_token": f"{subject_id}:class-{label:02d}",
                                "label": label,
                                "prediction": prediction,
                            }
                        )
    return pd.DataFrame.from_records(records)


def _score_request(
    contract: V2ContractBinding,
    *,
    candidate: V2CandidateSpec,
    budget: int,
    gate_enabled: bool | None,
    variant: str = "A_Q",
    cohort: str = "development",
    development_selection_receipt: dict[str, object] | None = None,
) -> dict[str, object]:
    support_count = budget * 40 if gate_enabled is True else 0
    return {
        "schema": V2_SCORE_REQUEST_SCHEMA,
        "candidate_id": V2_CANDIDATE_ID,
        "plan_sha256": contract.plan_sha256,
        "allocation_sha256": contract.allocation_sha256,
        **contract.asset_binding("beta_v1"),
        "dataset_id": "beta_v1",
        "cohort": cohort,
        "candidate": candidate.as_dict(),
        "development_selection_receipt": development_selection_receipt,
        "variant": variant,
        "subject_id": contract.subject_ids("beta_v1", cohort)[0],
        "budget": budget,
        "query_tokens": ["opaque-query-0001", "opaque-query-0002"],
        "query_fbcca_scores": [list(range(40)), list(reversed(range(40)))],
        "support_fbcca_scores": (
            None
            if support_count == 0
            else [
                [float(class_index == column) for column in range(40)]
                for class_index in range(40)
                for _ in range(budget)
            ]
        ),
        "support_labels": (
            None
            if support_count == 0
            else [class_index for class_index in range(40) for _ in range(budget)]
        ),
        "base_probabilities": None,
        "gate_enabled": gate_enabled,
        "template_query_class_scores": None,
        "template_query_class_probabilities": None,
        "template_score_provenance": None,
        "template_query_subbands": None,
        "template_support_subbands": None,
        "template_subband_weights": None,
        "query_interfaces": None,
        "support_interfaces": None,
        "query_impedance_kohm": None,
        "support_impedance_kohm": None,
        "relative_context_pairing_sha256": None,
    }


def test_r6_contract_and_allocation_replay_are_exact(contract: V2ContractBinding) -> None:
    assert contract.plan["method_revision"] == "r6_pre_outcome_explicit_filterbank_weights"
    assert contract.allocation_sha256 == (
        "3722449446183a7d5a4b7006b6a28c38cbcf536bbeac7bc74a07317143077cf4"
    )
    assert [
        len(contract.subject_ids(name, "development")) for name in ("beta_v1", "dong2023_v1")
    ] == [23, 19]
    assert [
        len(contract.subject_ids(name, "independent_gate")) for name in ("beta_v1", "dong2023_v1")
    ] == [45, 39]
    assert len(v2_candidate_grid(contract)) == 12

    tampered = json.loads(DEFAULT_V2_ALLOCATION_PATH.read_text(encoding="utf-8"))
    tampered["allocation"]["beta_v1"]["development"][0] = "sub070"
    with pytest.raises(ValueError):
        validate_v2_external_allocation_receipt(
            tampered, expected_sha256=contract.allocation_sha256
        )


@pytest.mark.parametrize(
    ("field_path", "replacement"),
    [
        (("method_revision",), "r4_pre_outcome_executable_formula_and_fail_closed_gate"),
        (
            ("support_operators", "fusion", "A_QM_lambda"),
            "lambda_max_times_budget_factor",
        ),
        (
            ("support_operators", "score_prototype_shrinkage", "prototype_formula"),
            "unbound_formula",
        ),
        (
            (
                "support_operators",
                "filterbank_target_template_residual",
                "filterbank_config_sha256",
            ),
            "0" * 64,
        ),
        (("target_local_gate", "authorization"), "implicit_true"),
        (("target_local_gate", "prequential_operator", "allowed_support_depths"), [1, 3]),
        (("relative_context", "model_effect"), "support_weight_only"),
        (("relative_context", "A_Q_context_policy"), "allow_context"),
    ],
)
def test_plan_validator_rejects_pre_r6_or_formula_drift(
    tmp_path: Path,
    field_path: tuple[str, ...],
    replacement: object,
) -> None:
    plan = yaml.safe_load(DEFAULT_V2_PLAN_PATH.read_text(encoding="utf-8"))
    cursor = plan
    for key in field_path[:-1]:
        cursor = cursor[key]
    cursor[field_path[-1]] = replacement
    plan_path = tmp_path / "plan.yaml"
    plan_path.write_text(yaml.safe_dump(plan, sort_keys=False), encoding="utf-8")
    with pytest.raises(ValueError, match="drifted|contract|formula"):
        validate_v2_plan_and_allocation(plan_path, DEFAULT_V2_ALLOCATION_PATH)


def test_one_sided_inference_is_deterministic_and_sign_flip_is_exact() -> None:
    values = np.array([0.10, 0.20, 0.30])
    primary = one_sided_t_lower_bound(values)
    first = one_sided_paired_sensitivity(values, null_margin=0.0, seed=19, n_resamples=128)
    second = one_sided_paired_sensitivity(values, null_margin=0.0, seed=19, n_resamples=128)
    assert primary["mean"] == pytest.approx(0.20)
    assert primary["confidence_interval_low"] > 0.0
    assert first == second
    assert first["sign_flip_exact"] is True
    assert first["sign_flip_draws"] == 8
    assert first["paired_sign_flip_p_value"] == pytest.approx(0.125)
    assert first["sensitivity_decision_role"] == "sensitivity_only"


def test_prediction_reduction_and_eauc_are_participant_paired(
    contract: V2ContractBinding,
) -> None:
    candidate = v2_candidate_grid(contract)[0]
    predictions = _prediction_frame(contract, candidate=candidate, cohort="development")
    participant_ba = reduce_v2_participant_balanced_accuracy(
        predictions,
        contract=contract,
        cohort="development",
        expected_candidate_keys=(candidate.candidate_key,),
    )
    deltas = derive_v2_participant_eauc_deltas(
        participant_ba,
        contract=contract,
        cohort="development",
        expected_candidate_keys=(candidate.candidate_key,),
    )
    assert len(participant_ba) == 42 * 2 * 3
    assert len(deltas) == 42
    assert np.array_equal(deltas["a0_k0"].to_numpy(), deltas["aq_k0"].to_numpy())
    assert np.allclose(deltas["delta_k1"], -1.0 / 40.0)
    assert np.allclose(deltas["eauc_delta"], -1.0 / 80.0)


def test_development_selection_applies_all_tie_breaks_and_stays_closed(
    contract: V2ContractBinding,
) -> None:
    candidates = v2_candidate_grid(contract)
    frame = _participant_delta_frame(
        contract,
        cohort="development",
        candidates=candidates,
        gains=lambda _candidate, _dataset, _subject: (0.02, 0.04),
    )
    receipt = select_v2_development_candidate(frame, contract=contract, n_resamples=128)
    assert receipt["status"] == "selected_candidate_frozen"
    assert receipt["selected_candidate"] == {
        "candidate_key": "score-prototype-pc1-lambda0p10",
        "operator": "score_prototype_shrinkage",
        "lambda_max": 0.10,
        "prototype_prior_pseudocount": 1.0,
    }
    assert receipt["independent_gate_authorized"] is True
    assert receipt["held_access_authorized"] is False


def test_development_selection_terminates_when_every_candidate_harms_one_dataset(
    contract: V2ContractBinding,
) -> None:
    frame = _participant_delta_frame(
        contract,
        cohort="development",
        candidates=v2_candidate_grid(contract),
        gains=lambda _candidate, dataset, _subject: (
            (-0.05, 0.02) if dataset == "dong2023_v1" else (0.01, 0.02)
        ),
    )
    receipt = select_v2_development_candidate(frame, contract=contract, n_resamples=64)
    assert receipt["status"] == "no_eligible_candidate_terminate"
    assert receipt["selected_candidate"] is None
    assert receipt["independent_gate_authorized"] is False
    assert receipt["held_access_authorized"] is False


def test_independent_gate_passes_all_requirements_but_never_authorizes_held(
    contract: V2ContractBinding,
) -> None:
    candidates = v2_candidate_grid(contract)
    development = _participant_delta_frame(
        contract,
        cohort="development",
        candidates=candidates,
        gains=lambda _candidate, _dataset, _subject: (0.02, 0.04),
    )
    selection = select_v2_development_candidate(development, contract=contract, n_resamples=64)
    selected_key = selection["selected_candidate"]["candidate_key"]
    selected = tuple(value for value in candidates if value.candidate_key == selected_key)
    gate_frame = _participant_delta_frame(
        contract,
        cohort="independent_gate",
        candidates=selected,
        gains=lambda _candidate, _dataset, _subject: (0.03, 0.03),
    )
    receipt = evaluate_v2_independent_aq_gate(
        gate_frame,
        development_selection_receipt=selection,
        contract=contract,
        n_resamples=128,
    )
    assert receipt["status"] == "pass"
    assert receipt["all_requirements_passed"] is True
    assert all(receipt["requirements"].values())
    assert receipt["held_access_authorized"] is False
    assert receipt["next_required"].startswith("choi_replication_clean_tag")


def test_independent_gate_failure_terminates_before_held(
    contract: V2ContractBinding,
) -> None:
    candidates = v2_candidate_grid(contract)
    development = _participant_delta_frame(
        contract,
        cohort="development",
        candidates=candidates,
        gains=lambda _candidate, _dataset, _subject: (0.02, 0.04),
    )
    selection = select_v2_development_candidate(development, contract=contract, n_resamples=32)
    selected_key = selection["selected_candidate"]["candidate_key"]
    selected = tuple(value for value in candidates if value.candidate_key == selected_key)
    gate_frame = _participant_delta_frame(
        contract,
        cohort="independent_gate",
        candidates=selected,
        gains=lambda _candidate, dataset, _subject: (
            (-0.01, 0.06) if dataset == "dong2023_v1" else (0.02, 0.06)
        ),
    )
    receipt = evaluate_v2_independent_aq_gate(
        gate_frame,
        development_selection_receipt=selection,
        contract=contract,
        n_resamples=64,
    )
    assert receipt["status"] == "failed_terminate_before_held60"
    assert receipt["requirements"]["each_dataset_k1_mean_ge_0"] is False
    assert receipt["held_access_authorized"] is False
    assert receipt["next_required"] == "terminate_candidate_without_wearable_held60_access"


@pytest.mark.parametrize(
    "path",
    [
        "/tmp/wearable_v3/request.json",
        "/tmp/source39/results.csv",
        "/tmp/held60/receipt.json",
        "/tmp/eeg-results/exposed.csv",
    ],
)
def test_data_path_guard_rejects_forbidden_names(path: str) -> None:
    with pytest.raises(PermissionError):
        reject_forbidden_v2_data_path(path)


def test_data_path_guard_rejects_symlink_alias(tmp_path: Path) -> None:
    real = tmp_path / "real.json"
    real.write_text("{}\n", encoding="utf-8")
    alias = tmp_path / "alias.json"
    alias.symlink_to(real)
    with pytest.raises(PermissionError, match="symlinks"):
        reject_forbidden_v2_data_path(alias)


def test_completion_receipt_is_read_only_and_never_overwritten(
    contract: V2ContractBinding, tmp_path: Path
) -> None:
    receipt = build_v2_preparation_receipt(contract)
    target = tmp_path / "completion.json"
    written = write_v2_completion_receipt_exclusive(target, receipt)
    assert written == target.absolute()
    assert stat.S_IMODE(target.stat().st_mode) == 0o400
    assert json.loads(target.read_text(encoding="utf-8")) == receipt
    with pytest.raises(FileExistsError):
        write_v2_completion_receipt_exclusive(target, receipt)


def test_score_request_rejects_query_outcomes_before_operator_import(
    contract: V2ContractBinding,
) -> None:
    candidate = v2_candidate_grid(contract)[0]
    request = _score_request(contract, candidate=candidate, budget=0, gate_enabled=False)
    request["query_labels"] = [0]
    with pytest.raises(ValueError, match="forbidden query fields"):
        score_v2_external_request(request, contract=contract)


def test_score_adapter_enforces_explicit_gate_before_support_access(
    contract: V2ContractBinding,
) -> None:
    candidate = v2_candidate_grid(contract)[0]
    missing = _score_request(contract, candidate=candidate, budget=1, gate_enabled=None)
    receipt = score_v2_external_request(missing, contract=contract)
    assert receipt["exact_fallback"] is True
    assert receipt["fallback_reason"] == "missing_gate_authorization"
    assert receipt["support_labels_loaded"] is False
    assert receipt["base_probabilities"] == receipt["fused_probabilities"]

    forbidden = dict(missing)
    forbidden["support_labels"] = list(range(40))
    with pytest.raises(PermissionError, match="omit support"):
        score_v2_external_request(forbidden, contract=contract)


def test_score_adapter_runs_active_aq_and_binds_output(contract: V2ContractBinding) -> None:
    candidate = v2_candidate_grid(contract)[0]
    request = _score_request(contract, candidate=candidate, budget=1, gate_enabled=True)
    receipt = score_v2_external_request(request, contract=contract)
    assert receipt["variant"] == "A_Q"
    assert receipt["exact_fallback"] is False
    assert receipt["fallback_reason"] is None
    assert receipt["support_labels_loaded"] is True
    assert receipt["final_budget"] == 1
    assert receipt["support_depth"] == 1
    assert receipt["prequential_fold_provenance"] is None
    assert receipt["relative_context_pairing_sha256"] is None
    assert receipt["comparable_context_pair_count"] == 0
    assert np.asarray(receipt["lambdas"]).min() > 0.0


def test_independent_scoring_requires_exact_selected_candidate_receipt(
    contract: V2ContractBinding,
) -> None:
    candidates = v2_candidate_grid(contract)
    development = _participant_delta_frame(
        contract,
        cohort="development",
        candidates=candidates,
        gains=lambda _candidate, _dataset, _subject: (0.02, 0.04),
    )
    selection = select_v2_development_candidate(development, contract=contract, n_resamples=32)
    selected = next(
        candidate
        for candidate in candidates
        if candidate.candidate_key == selection["selected_candidate"]["candidate_key"]
    )
    unauthorized = _score_request(
        contract,
        candidate=selected,
        budget=1,
        gate_enabled=True,
        cohort="independent_gate",
    )
    with pytest.raises(PermissionError, match="selection receipt"):
        score_v2_external_request(unauthorized, contract=contract)

    authorized = _score_request(
        contract,
        candidate=selected,
        budget=1,
        gate_enabled=True,
        cohort="independent_gate",
        development_selection_receipt=selection,
    )
    receipt = score_v2_external_request(authorized, contract=contract)
    assert (
        receipt["development_selection_receipt_sha256"] == (selection["completion_receipt_sha256"])
    )
    assert receipt["variant"] == "A_Q"

    authorized["variant"] = "A_QM"
    with pytest.raises(PermissionError, match="permits A_Q only"):
        score_v2_external_request(authorized, contract=contract)


def test_score_adapter_binds_aqm_pairing_and_context_count(
    contract: V2ContractBinding,
) -> None:
    candidate = v2_candidate_grid(contract)[0]
    request = _score_request(
        contract,
        candidate=candidate,
        budget=1,
        gate_enabled=True,
        variant="A_QM",
    )
    request["query_interfaces"] = ["wet", "wet"]
    request["support_interfaces"] = ["dry"] * 40
    request["relative_context_pairing_sha256"] = "a" * 64
    receipt = score_v2_external_request(request, contract=contract)
    assert receipt["variant"] == "A_QM"
    assert receipt["relative_context_pairing_sha256"] == "a" * 64
    assert receipt["comparable_context_pair_count"] == 80
    assert receipt["held_access_authorized"] is False


def test_score_adapter_requires_typed_p2_cache_provenance(
    contract: V2ContractBinding,
) -> None:
    candidate = next(
        value
        for value in v2_candidate_grid(contract)
        if value.candidate_key == "template-residual-lambda0p10"
    )
    request = _score_request(contract, candidate=candidate, budget=1, gate_enabled=True)
    request["template_query_class_scores"] = [list(range(40)), list(reversed(range(40)))]
    request["template_score_provenance"] = {
        "schema": "cfeg.metadata-calibration-v2-template-provenance.v1",
        "filterbank_sha256": ("b8c1ce4477d980b40f359bc3dc97f3125313191d24401f29a7e2dc473d0d1380"),
        "preprocessing_sha256": "b" * 64,
        "query_partition_sha256": "c" * 64,
        "support_partition_sha256": "d" * 64,
    }
    receipt = score_v2_external_request(request, contract=contract)
    assert receipt["template_score_provenance"] == request["template_score_provenance"]

    request["template_score_provenance"] = {
        **request["template_score_provenance"],
        "filterbank_sha256": "e" * 64,
    }
    with pytest.raises(ValueError, match="filterbank hash"):
        score_v2_external_request(request, contract=contract)


def test_prepare_cli_emits_exact_receipt(contract: V2ContractBinding, tmp_path: Path) -> None:
    output = tmp_path / "prepare.json"
    completed = subprocess.run(
        [
            sys.executable,
            str(REPO / "scripts" / "run_metadata_calibration_v2.py"),
            "--plan",
            str(DEFAULT_V2_PLAN_PATH),
            "--allocation",
            str(DEFAULT_V2_ALLOCATION_PATH),
            "prepare",
            "--output",
            str(output),
        ],
        cwd=REPO,
        check=True,
        capture_output=True,
        text=True,
    )
    report = json.loads(completed.stdout)
    receipt = json.loads(output.read_text(encoding="utf-8"))
    assert report["completion_receipt_sha256"] == receipt["completion_receipt_sha256"]
    assert report["held_access_authorized"] is False
    assert receipt["plan_sha256"] == contract.plan_sha256
