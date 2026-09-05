from __future__ import annotations

import copy
import hashlib
import json

import pytest

from cfeg.metadata_calibration_manifest import (
    EXECUTION_CONTRACT_SCHEMA,
    build_metadata_calibration_execution_contract,
    validate_metadata_calibration_execution_contract,
)


def _payload_sha256(value: dict, field: str) -> str:
    payload = copy.deepcopy(value)
    payload.pop(field)
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


@pytest.fixture(scope="module")
def source_contract() -> dict:
    return build_metadata_calibration_execution_contract(phase="source_development")


@pytest.fixture(scope="module")
def held_contract() -> dict:
    return build_metadata_calibration_execution_contract(
        phase="held_participant_evaluation"
    )


def test_source_execution_contract_has_exact_complete_job_grid(
    source_contract: dict,
) -> None:
    assert source_contract["schema"] == EXECUTION_CONTRACT_SCHEMA
    assert source_contract["execution_authorized"] is False
    assert source_contract["held_prerequisites"] is None
    assert source_contract["expected_candidate_jobs"] == 9
    assert source_contract["expected_baseline_jobs"] == 15
    assert source_contract["expected_job_count"] == 24

    phase_spec = source_contract["phase_spec"]
    assert phase_spec["checkpoint_groups"] == ["fold0", "fold1", "fold2"]
    assert len(phase_spec["cells"]) == 12
    assert all(
        len(phase_spec["target_subject_ids_by_checkpoint_group"][group]) == 13
        for group in phase_spec["checkpoint_groups"]
    )
    assert all(
        len(phase_spec["source_pool_subject_ids_by_checkpoint_group"][group]) == 26
        for group in phase_spec["checkpoint_groups"]
    )

    candidate = [
        job for job in source_contract["jobs"] if job["producer_kind"] == "candidate"
    ]
    baseline = [
        job for job in source_contract["jobs"] if job["producer_kind"] == "baseline"
    ]
    assert {
        (job["checkpoint_group"], job["seed"])
        for job in candidate
    } == {
        (group, seed)
        for group in ("fold0", "fold1", "fold2")
        for seed in (42, 43, 44)
    }
    assert {
        (job["checkpoint_group"], job["adapter_id"])
        for job in baseline
    } == {
        (group, adapter)
        for group in ("fold0", "fold1", "fold2")
        for adapter in (
            "strict_FBCCA",
            "target_template_correlation",
            "target_filterbank_eTRCA",
            "same3_filterbank_eTRCA",
            "chiang2021_LST_filterbank_eTRCA",
        )
    }
    assert all(job["seed"] is None for job in baseline)
    assert all(job["budgets"] == [0, 1, 3, 5] for job in candidate)
    assert all(len(job["cells"]) == 12 for job in candidate)
    assert all(
        job["outer_fold"] == int(job["checkpoint_group"].removeprefix("fold"))
        for job in source_contract["jobs"]
    )


def test_held_execution_contract_is_separate_and_requires_gate_receipt_slots(
    held_contract: dict,
) -> None:
    assert held_contract["execution_authorized"] is False
    assert held_contract["expected_candidate_jobs"] == 3
    assert held_contract["expected_baseline_jobs"] == 5
    assert held_contract["expected_job_count"] == 8
    assert held_contract["phase_spec"]["checkpoint_groups"] == ["held"]
    assert len(held_contract["phase_spec"]["cells"]) == 8
    assert len(
        held_contract["phase_spec"]["target_subject_ids_by_checkpoint_group"]["held"]
    ) == 60
    assert len(
        held_contract["phase_spec"]["source_pool_subject_ids_by_checkpoint_group"][
            "held"
        ]
    ) == 39
    assert held_contract["held_prerequisites"] == {
        "source_development_gate_receipt_sha256": "required_in_execution_manifest",
        "source_development_private_seal_sha256": "required_in_execution_manifest",
        "source_development_status": "passed",
        "held_selected_epochs_by_seed_sha256": "required_in_execution_manifest",
    }
    candidate = [
        job for job in held_contract["jobs"] if job["producer_kind"] == "candidate"
    ]
    baseline = [
        job for job in held_contract["jobs"] if job["producer_kind"] == "baseline"
    ]
    assert {(job["checkpoint_group"], job["seed"]) for job in candidate} == {
        ("held", 42),
        ("held", 43),
        ("held", 44),
    }
    assert {(job["checkpoint_group"], job["adapter_id"]) for job in baseline} == {
        ("held", "strict_FBCCA"),
        ("held", "target_template_correlation"),
        ("held", "target_filterbank_eTRCA"),
        ("held", "same3_filterbank_eTRCA"),
        ("held", "chiang2021_LST_filterbank_eTRCA"),
    }
    assert all(job["outer_fold"] is None for job in held_contract["jobs"])
    assert all(
        job["held_selected_epochs"] == "source_gate_bound_at_phase_manifest"
        for job in candidate
    )


def test_held_contract_binds_exact_source_gate_epoch_selection() -> None:
    selected = {
        "42": {"stage1_epochs": 12, "stage2_epochs": 7},
        "43": {"stage1_epochs": 14, "stage2_epochs": 9},
        "44": {"stage1_epochs": 16, "stage2_epochs": 11},
    }
    contract = build_metadata_calibration_execution_contract(
        phase="held_participant_evaluation",
        held_selected_epochs_by_seed=selected,
    )
    validate_metadata_calibration_execution_contract(contract)
    jobs = {
        int(job["seed"]): job
        for job in contract["jobs"]
        if job["producer_kind"] == "candidate"
    }
    assert jobs[42]["held_selected_epochs"] == selected["42"]
    assert jobs[43]["held_selected_epochs"] == selected["43"]
    assert jobs[44]["held_selected_epochs"] == selected["44"]


@pytest.mark.parametrize("fixture_name", ["source_contract", "held_contract"])
def test_job_id_output_and_receipt_paths_are_unique(
    fixture_name: str,
    request: pytest.FixtureRequest,
) -> None:
    jobs = request.getfixturevalue(fixture_name)["jobs"]
    for field in ("job_id", "required_output", "required_receipt"):
        values = [job[field] for job in jobs]
        assert len(values) == len(set(values))


@pytest.mark.parametrize("fixture_name", ["source_contract", "held_contract"])
def test_contract_and_every_job_have_canonical_self_hashes(
    fixture_name: str,
    request: pytest.FixtureRequest,
) -> None:
    contract = request.getfixturevalue(fixture_name)
    assert contract["execution_contract_sha256"] == _payload_sha256(
        contract,
        "execution_contract_sha256",
    )
    for job in contract["jobs"]:
        assert job["job_contract_sha256"] == _payload_sha256(
            job,
            "job_contract_sha256",
        )


def test_validator_accepts_only_phase_derived_contract(source_contract: dict) -> None:
    binding = validate_metadata_calibration_execution_contract(source_contract)
    assert binding.phase == "source_development"
    assert binding.expected_candidate_jobs == 9
    assert binding.expected_baseline_jobs == 15
    assert binding.execution_authorized is False

    tampered = copy.deepcopy(source_contract)
    tampered["jobs"][0]["budgets"] = [0, 1, 3]
    tampered["jobs"][0]["job_contract_sha256"] = _payload_sha256(
        tampered["jobs"][0],
        "job_contract_sha256",
    )
    tampered["execution_contract_sha256"] = _payload_sha256(
        tampered,
        "execution_contract_sha256",
    )
    with pytest.raises(ValueError, match="canonical phase-derived"):
        validate_metadata_calibration_execution_contract(tampered)

    type_tampered = copy.deepcopy(source_contract)
    type_tampered["execution_authorized"] = 0
    with pytest.raises(ValueError, match="canonical phase-derived"):
        validate_metadata_calibration_execution_contract(type_tampered)


def test_invalid_phase_is_rejected() -> None:
    with pytest.raises(ValueError, match="Unknown"):
        build_metadata_calibration_execution_contract(phase="invalid")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="invalid phase"):
        validate_metadata_calibration_execution_contract({"phase": "invalid"})
