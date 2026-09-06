from __future__ import annotations

import inspect

import numpy as np
import pytest

from cfeg.analysis import metadata_calibration_v3_synthetic as synthetic
from cfeg.models import metadata_calibration_v3 as v3

_UNIT_ROOT = 424242
@pytest.fixture(scope="module")
def contract() -> synthetic.SyntheticV3Contract:
    return synthetic.validate_synthetic_contract()


@pytest.fixture(scope="module")
def unit_rng():
    return synthetic._issue_unit_test_rng_authority(_UNIT_ROOT)


@pytest.fixture(scope="module")
def reference(unit_rng) -> v3.ContextReference:
    return synthetic._fit_synthetic_context_reference_for_test(unit_rng)


def _install_fake_scorer(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_score(_contract, signals):
        scores = np.empty((signals.shape[0], 12, 12), dtype=np.float64)
        for block in range(signals.shape[0]):
            scores[block] = np.eye(12) + 0.01 * block
        scores.setflags(write=False)
        return scores

    monkeypatch.setattr(synthetic, "_score_strict_fbcca_blocks", fake_score)


def test_contract_hashes_counts_and_no_public_raw_seed_runner(
    contract: synthetic.SyntheticV3Contract,
) -> None:
    assert contract.plan_sha256 == synthetic.EXPECTED_SYNTHETIC_PLAN_SHA256
    assert contract.master_plan_sha256 == synthetic.EXPECTED_MASTER_PLAN_SHA256
    assert contract.filterbank_sha256 == synthetic.EXPECTED_FILTERBANK_SHA256
    assert synthetic.validate_development_contract(contract) == {
        "condition_level_cells_per_parameter_pair": 172,
        "B4_composite_cells_per_parameter_pair": 34,
        "participant_summary_cells_per_parameter_pair": 206,
        "development_participant_rows_per_parameter_pair": 9_888,
        "development_participant_metric_rows": 88_992,
        "development_invariant_rows": 90,
        "total_development_rows": 89_082,
    }
    assert "run_development" not in vars(synthetic)
    assert "root_seed" not in inspect.signature(
        synthetic.generate_synthetic_participant
    ).parameters


def test_exact_identity_blueprint_has_89082_unique_nonoutcome_rows() -> None:
    metric = list(synthetic._expected_metric_identities())
    invariant = list(synthetic._expected_invariant_identities())
    assert len(metric) == len(set(metric)) == 88_992
    assert len(invariant) == len(set(invariant)) == 90
    assert len(metric) + len(invariant) == 89_082
    with pytest.raises(ValueError, match="requires 89082"):
        synthetic.validate_complete_development_rows(
            [], expected_stress_participant_indices=(0, 1, 2, 3, 4)
        )


def test_method_grid_matches_every_declared_family_cell_count() -> None:
    expected = {
        "B1_participant_class_confusion": 11,
        "B2_participant_phase_spatial_shift": 11,
        "B3_impedance_linked_transfer_shift": 25,
        "B4_interface_calibrated_impedance_shift": 68,
        "N1_clean_anchor": 16,
        "N2_random_support_labels": 11,
        "N3_nonstationary_calibration": 11,
        "N4_context_null": 19,
    }
    assert {
        family: len(synthetic.method_specs_for_family(family))
        for family in synthetic.FAMILY_NAMES
    } == expected
    b4 = synthetic.method_specs_for_family("B4_interface_calibrated_impedance_shift")
    assert sum(spec.condition == "wet" for spec in b4) == 34
    assert sum(spec.condition == "dry" for spec in b4) == 34


def test_covariate_only_reference_is_exact_ordered_and_roundtrips(
    reference: v3.ContextReference,
    unit_rng,
) -> None:
    packets = synthetic._generate_covariate_reference_packets_for_test(unit_rng)
    assert len(packets) == 3 * 256
    assert [packets[index]["interface"] for index in (0, 256, 512)] == [
        "neutral",
        "wet",
        "dry",
    ]
    assert set(packets[0]) == {
        "interface",
        "impedance_kohm_by_channel",
        "channel_availability",
    }
    assert [table.interface_lookup_key for table in reference.interface_tables] == [
        "neutral",
        "wet",
        "dry",
    ]
    assert reference.canonical_channels == tuple(f"ch{index:02d}" for index in range(8))
    assert min(min(table.observed_counts) for table in reference.interface_tables) >= 128
    payload = v3.context_reference_payload(reference)
    assert v3.validate_context_reference_payload(payload) == reference


def test_generation_is_key_order_invariant_and_B3_context_tracks_signal_state(
    contract: synthetic.SyntheticV3Contract,
    unit_rng,
) -> None:
    first = synthetic._generate_synthetic_participant_for_test(
        contract,
        family="B3_impedance_linked_transfer_shift",
        participant_index=7,
        rng_authority=unit_rng,
    )
    synthetic._generate_synthetic_participant_for_test(
        contract,
        family="B1_participant_class_confusion",
        participant_index=2,
        rng_authority=unit_rng,
    )
    second = synthetic._generate_synthetic_participant_for_test(
        contract,
        family="B3_impedance_linked_transfer_shift",
        participant_index=7,
        rng_authority=unit_rng,
    )
    assert np.array_equal(first.signals, second.signals)
    assert first.partition_sha256s == second.partition_sha256s
    assert sorted(first.signal_states.tolist()) == [-1] * 5 + [1] * 5
    assert first.context_states_by_condition[0][1] == tuple(first.signal_states)
    packets = first.context_packets("neutral")
    low = np.mean(
        [
            np.mean(packet.impedance_kohm_by_channel)
            for packet, state in zip(packets, first.signal_states, strict=True)
            if state == -1
        ]
    )
    high = np.mean(
        [
            np.mean(packet.impedance_kohm_by_channel)
            for packet, state in zip(packets, first.signal_states, strict=True)
            if state == 1
        ]
    )
    assert high > low
    assert not first.signals.flags.writeable


def test_family_specific_waveform_and_label_contracts(
    contract: synthetic.SyntheticV3Contract,
    unit_rng,
) -> None:
    b1 = synthetic._generate_synthetic_participant_for_test(
        contract,
        family="B1_participant_class_confusion",
        participant_index=1,
        rng_authority=unit_rng,
    )
    assert np.array_equal(
        b1.generated_confuser_classes[0],
        (np.arange(12) + 2) % 12,
    )
    n2 = synthetic._generate_synthetic_participant_for_test(
        contract,
        family="N2_random_support_labels",
        participant_index=1,
        rng_authority=unit_rng,
    )
    assert all(
        not np.array_equal(n2.recorded_support_labels[block], n2.true_labels[block])
        for block in range(5)
    )
    assert np.array_equal(n2.recorded_support_labels[5:], n2.true_labels[5:])
    n3 = synthetic._generate_synthetic_participant_for_test(
        contract,
        family="N3_nonstationary_calibration",
        participant_index=1,
        rng_authority=unit_rng,
    )
    assert np.array_equal(n3.generated_target_classes[0], (np.arange(12) + 1) % 12)
    assert np.array_equal(n3.generated_target_classes[5], np.arange(12))


def test_B4_source_stress_keeps_EEG_and_forces_every_query_to_OOD(
    contract: synthetic.SyntheticV3Contract,
    reference: v3.ContextReference,
    unit_rng,
) -> None:
    participant = synthetic._generate_synthetic_participant_for_test(
        contract,
        family="B4_interface_calibrated_impedance_shift",
        participant_index=0,
        rng_authority=unit_rng,
    )
    stressed = synthetic._apply_source_range_stress_for_test(
        participant,
        reference,
    )
    assert stressed.signals is participant.signals
    assert stressed.partition_sha256s == participant.partition_sha256s
    assert stressed.source_range_stress
    for condition in ("wet", "dry"):
        packets = stressed.context_packets(condition)
        for query_block in synthetic.QUERY_BLOCKS:
            query = packets[query_block - 1]
            support = packets[0]
            pairing = v3.context_pairing_sha256(
                query_key=query.packet_key,
                ordered_support_block_keys=(support.packet_key,),
                query_packet_key=query.packet_key,
                ordered_support_packet_keys=(support.packet_key,),
            )
            preflight = v3._preflight_context_unit(
                reference=reference,
                query_key=query.packet_key,
                ordered_support_block_keys=(support.packet_key,),
                query_packet=query,
                support_packets=(support,),
                pairing_sha256=pairing,
            )
            assert preflight.decision_reason == "query_context_OOD_exact_A0"


def test_rejected_synthetic_preflight_makes_zero_support_producer_calls(
    contract: synthetic.SyntheticV3Contract,
    reference: v3.ContextReference,
    unit_rng,
) -> None:
    participant = synthetic._generate_synthetic_participant_for_test(
        contract,
        family="B4_interface_calibrated_impedance_shift",
        participant_index=0,
        rng_authority=unit_rng,
    )
    participant = synthetic._apply_source_range_stress_for_test(
        participant,
        reference,
    )
    calls = {"waveform": 0, "label": 0, "score": 0}

    def forbidden_product(
        condition: str,
        budget: int,
        query_block: int,
    ) -> v3.V3SupportProduct:
        del condition, budget, query_block
        calls["waveform"] += 1
        calls["label"] += 1
        calls["score"] += 1
        raise AssertionError("support producer ran after rejecting context preflight")

    def forbidden_gate(budget: int) -> v3.PrequentialGateDecision | None:
        del budget
        calls["label"] += 1
        calls["score"] += 1
        raise AssertionError("support gate ran after rejecting context preflight")

    output = synthetic._evaluate_aqm_assignment(
        participant=participant,
        condition="wet",
        budget=1,
        query_block=6,
        query_scores=np.eye(12, dtype=np.float64),
        reference=reference,
        validated_reference=None,
        config=v3.V3OperatorConfig(v3.canonical_operator_grid()[0]),
        control="correct",
        assignment=(0,),
        product_for=forbidden_product,
        gate_for=forbidden_gate,
        gate_mode="deployed",
    )
    assert calls == {"waveform": 0, "label": 0, "score": 0}
    assert output.exact_a0
    assert output.reliability_sha256 is None


def test_invariant_suite_executes_all_ten_cases_for_each_cell(
    reference: v3.ContextReference,
) -> None:
    rows = synthetic._build_invariant_rows_for_test(
        reference,
        v3.canonical_operator_grid()[0],
    )
    assert len(rows) == 10
    assert all(row["passed"] for row in rows)
    assert {row["family"] for row in rows} == {
        "N5_invalid_context",
        "N6_support_reliability_invariants",
    }


def test_participant_evaluator_emits_exact_B3_and_B4_counts(
    monkeypatch: pytest.MonkeyPatch,
    contract: synthetic.SyntheticV3Contract,
    reference: v3.ContextReference,
    unit_rng,
) -> None:
    _install_fake_scorer(monkeypatch)
    cell = v3.canonical_operator_grid()[0]
    b3 = synthetic._generate_synthetic_participant_for_test(
        contract,
        family="B3_impedance_linked_transfer_shift",
        participant_index=0,
        rng_authority=unit_rng,
    )
    invalid_reference_calls = 0

    def forbidden_unvalidated_loader(
        budget: int,
    ) -> synthetic.StrictFBCCASupportProduct:
        nonlocal invalid_reference_calls
        del budget
        invalid_reference_calls += 1
        raise AssertionError("support loaded before reference authorization")

    with pytest.raises(TypeError, match="exact DevelopmentRNGAuthority"):
        synthetic.evaluate_synthetic_participant(
            contract,
            b3,
            synthetic.produce_strict_fbcca(contract, b3),
            reference,
            cell,
            rng_authority=unit_rng,
            validated_reference=reference,
            support_loader=forbidden_unvalidated_loader,
        )
    assert invalid_reference_calls == 0
    b3_rows = synthetic._evaluate_synthetic_participant_for_test(
        contract,
        b3,
        synthetic.produce_strict_fbcca(contract, b3),
        reference,
        cell,
        support_loader=lambda budget: synthetic.produce_strict_fbcca_support(
            contract,
            b3,
            budget=budget,
        ),
    )
    assert len(b3_rows) == 25
    assert sum(row["control"] == "within_prefix_packet_shuffle" for row in b3_rows) == 2
    k3 = next(
        row
        for row in b3_rows
        if row["control"] == "within_prefix_packet_shuffle" and row["budget"] == 3
    )
    assert k3["pairing_packet_binding_changed_fraction"] == 1.0
    assert len(k3["pairing_mean_derangement_abs_g_M_change_by_query_block"]) == 5

    b4 = synthetic._generate_synthetic_participant_for_test(
        contract,
        family="B4_interface_calibrated_impedance_shift",
        participant_index=0,
        rng_authority=unit_rng,
    )
    b4_rows = synthetic._evaluate_synthetic_participant_for_test(
        contract,
        b4,
        synthetic.produce_strict_fbcca(contract, b4),
        reference,
        cell,
        support_loader=lambda budget: synthetic.produce_strict_fbcca_support(
            contract,
            b4,
            budget=budget,
        ),
    )
    assert len(b4_rows) == 102
    assert sum(row["condition"] == "equal_condition_composite" for row in b4_rows) == 34


def test_strict_fbcca_producer_binds_frozen_shape_without_persisting_subbands(
    monkeypatch: pytest.MonkeyPatch,
    contract: synthetic.SyntheticV3Contract,
    unit_rng,
) -> None:
    participant = synthetic._generate_synthetic_participant_for_test(
        contract,
        family="N1_clean_anchor",
        participant_index=0,
        rng_authority=unit_rng,
    )
    weights = contract.filterbank["weights"]
    seen_shapes: list[tuple[int, ...]] = []

    def fake_filterbank(values, *, sfreq, filterbank):
        seen_shapes.append(values.shape)
        assert values.shape in {(60, 8, 500), (36, 8, 500)}
        assert sfreq == 250.0
        assert filterbank is contract.filterbank
        return np.broadcast_to(values, (7, *values.shape)), {"weights": weights}

    monkeypatch.setattr(synthetic, "apply_filterbank", fake_filterbank)
    monkeypatch.setattr(synthetic, "cca_score", lambda value, reference, regularization: 0.25)
    product = synthetic.produce_strict_fbcca(contract, participant)
    assert product.scores.shape == (5, 12, 12)
    assert product.query_eeg_sha256s == tuple(
        synthetic._array_sha256(participant.signals[block - 1])
        for block in synthetic.QUERY_BLOCKS
    )
    assert np.isfinite(product.scores).all()
    assert not product.scores.flags.writeable
    support = synthetic.produce_strict_fbcca_support(
        contract,
        participant,
        budget=3,
    )
    assert seen_shapes == [(60, 8, 500), (36, 8, 500)]
    assert support.scores.shape == (3, 12, 12)
    assert np.array_equal(
        support.recorded_support_labels,
        participant.recorded_support_labels[:3],
    )
    assert not support.recorded_support_labels.flags.writeable


def _gate_report(cell: v3.V3GridCell, *, eligible: bool, mean: float, lcb: float):
    components = {
        "invariant_suite": eligible,
        "pairing_potency": eligible,
        "A_Q_viability_B1_or_B2": eligible,
        "B3_metadata_efficacy": eligible,
        "B4_metadata_efficacy": eligible,
        "B4_in_reference_metadata_efficacy": eligible,
        "B4_source_range_stress_safety": eligible,
        "B4_interface_scale_value": eligible,
        "k3_pairing_mechanism": eligible,
        "B3_B4_deployment_viability": eligible,
        "N1_null_noninferiority": eligible,
        "N4_null_equivalence": eligible,
        "severe_harm": eligible,
        "adversarial_abstention": eligible,
        "helpful_use_anti_triviality": eligible,
    }
    return {
        "schema": "cfeg.metadata-calibration-efficiency-v3.grid-gate-report.v1",
        "grid_cell_index": cell.index,
        "grid_cell_id": cell.grid_cell_id,
        "operator_instance_sha256": cell.operator_instance_sha256,
        "prototype_prior_pseudocount": cell.prototype_prior_pseudocount,
        "lambda_max": cell.lambda_max,
        "components": components,
        "A_Q_viability": {},
        "metadata_efficacy": {},
        "B4_in_reference_metadata_efficacy": {},
        "B4_source_range_stress_safety": {},
        "B4_interface_scale_value": {},
        "pairing_mechanism": {},
        "deployment_viability": {},
        "pairing_potency": {},
        "N1_null_noninferiority": {},
        "N4_null_equivalence": {},
        "severe_harm": {},
        "adversarial_abstention": {},
        "helpful_use_anti_triviality": {},
        "selection_rank_tuple": {
            "minimum_mandatory_observed_gain": mean,
            "minimum_corresponding_one_sided_LCB": lcb,
        },
        "eligible": eligible,
    }


def test_selector_uses_gain_lcb_lambda_nu_then_index_without_rounding() -> None:
    cells = v3.canonical_operator_grid()
    reports = [_gate_report(cell, eligible=False, mean=0.0, lcb=0.0) for cell in cells]
    reports[0] = _gate_report(cells[0], eligible=True, mean=0.02, lcb=0.01)
    reports[3] = _gate_report(cells[3], eligible=True, mean=0.02, lcb=0.01)
    reports[6] = _gate_report(cells[6], eligible=True, mean=0.02, lcb=0.01)
    assert synthetic.select_grid_cell(reports)["grid_cell_id"] == cells[6].grid_cell_id
    reports[8] = _gate_report(cells[8], eligible=True, mean=0.0200000000001, lcb=0.009)
    assert synthetic.select_grid_cell(reports)["grid_cell_id"] == cells[8].grid_cell_id
    assert synthetic.select_grid_cell(
        [_gate_report(cell, eligible=False, mean=0.0, lcb=0.0) for cell in cells]
    ) is None


@pytest.mark.parametrize(
    ("capability_type", "require_capability"),
    (
        (
            synthetic.ValidatedDevelopmentResult,
            synthetic.require_validated_development_result,
        ),
        (synthetic.SelectedMethodProposal, synthetic.require_selected_method_proposal),
    ),
)
def test_publication_capabilities_reject_constructor_mapping_subclass_and_manual_new(
    capability_type,
    require_capability,
) -> None:
    with pytest.raises(TypeError, match="issued only"):
        capability_type()
    with pytest.raises(TypeError, match="exact"):
        require_capability({"payload_sha256": "a" * 64})
    with pytest.raises(TypeError, match="not issued"):
        require_capability(object.__new__(capability_type))

    forged_subclass = type("ForgedCapability", (capability_type,), {})
    with pytest.raises(TypeError, match="exact"):
        require_capability(object.__new__(forged_subclass))
