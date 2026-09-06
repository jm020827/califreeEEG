from __future__ import annotations

import inspect
import sys
from copy import deepcopy
from types import ModuleType, SimpleNamespace

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


def _bundle_rng_authorities(monkeypatch: pytest.MonkeyPatch):
    bundle = SimpleNamespace(
        payload_sha256="1" * 64,
        file_sha256="2" * 64,
        source_bundle_sha256="3" * 64,
        numerical_runtime_fingerprint_sha256="4" * 64,
        clean_commit="5" * 40,
        clean_tree="6" * 40,
        tracked_source_files=(
            SimpleNamespace(
                path=v3.MASTER_PLAN_REPOSITORY_PATH,
                file_sha256=v3.MASTER_PLAN_SHA256,
            ),
            SimpleNamespace(
                path=v3.SYNTHETIC_PLAN_REPOSITORY_PATH,
                file_sha256=v3.SYNTHETIC_PLAN_SHA256,
            ),
            SimpleNamespace(
                path=v3.MODEL_MODULE_REPOSITORY_PATH,
                file_sha256=v3._IMPORTED_MODEL_MODULE_SHA256,
            ),
            SimpleNamespace(
                path=synthetic.SYNTHETIC_MODULE_REPOSITORY_PATH,
                file_sha256=synthetic._IMPORTED_SYNTHETIC_MODULE_SHA256,
            ),
            SimpleNamespace(
                path=synthetic.FBCCA_MODULE_REPOSITORY_PATH,
                file_sha256=synthetic._IMPORTED_FBCCA_MODULE_SHA256,
            ),
            SimpleNamespace(
                path=synthetic.FILTERBANK_REPOSITORY_PATH,
                file_sha256=synthetic.EXPECTED_FILTERBANK_SHA256,
            ),
        ),
    )
    context = SimpleNamespace(
        schema=v3.CONTEXT_REFERENCE_SCHEMA,
        payload_sha256="7" * 64,
        file_sha256="8" * 64,
    )
    governance = ModuleType("cfeg.metadata_calibration_v3_governance")

    def require_bundle(value, *, expected_commit, expected_tree):
        assert value is bundle
        assert expected_commit == bundle.clean_commit
        assert expected_tree == bundle.clean_tree
        return bundle

    def require_development_prerequisites(
        value, context_value, *, expected_commit, expected_tree
    ):
        assert context_value is context
        return require_bundle(
            value,
            expected_commit=expected_commit,
            expected_tree=expected_tree,
        ), context

    governance.require_context_reference_rng_bundle_capability = require_bundle
    governance.require_development_rng_prerequisites = require_development_prerequisites
    monkeypatch.setitem(
        sys.modules,
        "cfeg.metadata_calibration_v3_governance",
        governance,
    )
    reference = v3.issue_bundle_bound_context_reference_rng_authority(
        bundle,
        expected_commit=bundle.clean_commit,
        expected_tree=bundle.clean_tree,
    )
    development = v3.issue_bundle_bound_development_rng_authority(
        bundle,
        context,
        expected_commit=bundle.clean_commit,
        expected_tree=bundle.clean_tree,
    )
    return bundle, reference, development


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


def test_contract_is_nominal_deeply_immutable_and_revalidated_before_scoring(
    monkeypatch: pytest.MonkeyPatch,
    unit_rng,
) -> None:
    contract = synthetic.validate_synthetic_contract()
    with pytest.raises(TypeError, match="issued only"):
        synthetic.SyntheticV3Contract()
    with pytest.raises(TypeError):
        contract.plan["waveform"] = {}
    with pytest.raises(TypeError):
        contract.filterbank["weights"] = ()
    with pytest.raises(TypeError):
        contract.filterbank["weights"][0] = 99.0

    participant = synthetic._generate_synthetic_participant_for_test(
        contract,
        family="N1_clean_anchor",
        participant_index=0,
        rng_authority=unit_rng,
    )
    mutated_filterbank = synthetic._deep_thaw_json(contract.filterbank)
    mutated_filterbank["weights"][0] = 99.0
    object.__setattr__(contract, "filterbank", mutated_filterbank)
    scorer_calls = 0

    def forbidden_scorer(*args, **kwargs):
        nonlocal scorer_calls
        del args, kwargs
        scorer_calls += 1
        raise AssertionError("scorer ran before contract mutation rejection")

    monkeypatch.setattr(synthetic, "_score_strict_fbcca_blocks", forbidden_scorer)
    with pytest.raises(TypeError, match="deeply immutable"):
        synthetic.produce_strict_fbcca(contract, participant)
    assert scorer_calls == 0


@pytest.mark.parametrize(
    "root_seed",
    (
        v3.DEVELOPMENT_ROOT_SEED,
        v3.CONTEXT_REFERENCE_ROOT_SEED,
        8812983586834372979543294859684702645563544465387352627235733726918051280063,
    ),
)
def test_unit_rng_rejects_every_governed_root_before_rng_construction(
    monkeypatch: pytest.MonkeyPatch,
    root_seed: int,
) -> None:
    calls = {"seed_sequence": 0, "bit_generator": 0, "generator": 0}

    def forbidden(name):
        def fail(*args, **kwargs):
            del args, kwargs
            calls[name] += 1
            raise AssertionError("RNG construction occurred before seed rejection")

        return fail

    monkeypatch.setattr(np.random, "SeedSequence", forbidden("seed_sequence"))
    monkeypatch.setattr(np.random, "PCG64DXSM", forbidden("bit_generator"))
    monkeypatch.setattr(np.random, "Generator", forbidden("generator"))
    with pytest.raises(ValueError, match="reserved"):
        synthetic._issue_unit_test_rng_authority(root_seed)
    assert calls == {"seed_sequence": 0, "bit_generator": 0, "generator": 0}


def test_production_generators_reject_unit_or_callable_authority_before_rng_use(
    monkeypatch: pytest.MonkeyPatch,
    contract: synthetic.SyntheticV3Contract,
    reference: v3.ContextReference,
    unit_rng,
) -> None:
    calls = 0

    def forbidden_seed_sequence(*args, **kwargs):
        nonlocal calls
        del args, kwargs
        calls += 1
        raise AssertionError("production rejection happened after RNG construction")

    monkeypatch.setattr(np.random, "SeedSequence", forbidden_seed_sequence)
    with pytest.raises(TypeError, match="ContextReferenceRNGAuthority"):
        synthetic.generate_covariate_reference_packets(rng_authority=unit_rng)
    with pytest.raises(TypeError, match="ContextReferenceRNGAuthority"):
        synthetic.generate_covariate_reference_packets(rng_authority=lambda *_: None)
    with pytest.raises(TypeError, match="DevelopmentRNGAuthority"):
        synthetic.generate_synthetic_participant(
            contract,
            family="N1_clean_anchor",
            participant_index=0,
            rng_authority=unit_rng,
        )
    with pytest.raises(TypeError, match="DevelopmentRNGAuthority"):
        synthetic.sensitivity_resampling_report(
            {"B1_AQ_minus_A0_eAUC": np.ones(4)},
            grid_cell_index=1,
            rng_authority=unit_rng,
        )
    with pytest.raises(TypeError, match="DevelopmentRNGAuthority"):
        synthetic.execute_complete_development(
            contract,
            reference,
            rng_authority=unit_rng,
            validated_reference=reference,
            context_reference_file_sha256="a" * 64,
        )
    assert calls == 0


def test_loaded_source_and_runtime_drift_fail_before_production_rng_use(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bundle, reference_authority, development_authority = _bundle_rng_authorities(
        monkeypatch
    )
    calls = 0

    def forbidden_seed_sequence(*args, **kwargs):
        nonlocal calls
        del args, kwargs
        calls += 1
        raise AssertionError("RNG constructed before provenance rejection")

    monkeypatch.setattr(np.random, "SeedSequence", forbidden_seed_sequence)
    monkeypatch.setattr(
        synthetic,
        "_IMPORTED_SYNTHETIC_MODULE_SHA256",
        "0" * 64,
    )
    with pytest.raises(RuntimeError, match="loaded V3 synthetic code"):
        synthetic.generate_covariate_reference_packets(
            rng_authority=reference_authority
        )
    assert calls == 0

    monkeypatch.setattr(
        synthetic,
        "_IMPORTED_SYNTHETIC_MODULE_SHA256",
        next(
            entry.file_sha256
            for entry in bundle.tracked_source_files
            if entry.path == synthetic.SYNTHETIC_MODULE_REPOSITORY_PATH
        ),
    )
    original_apply_filterbank = synthetic.apply_filterbank
    monkeypatch.setattr(synthetic, "apply_filterbank", lambda *args, **kwargs: None)
    with pytest.raises(RuntimeError, match="strict-FBCCA"):
        synthetic.generate_covariate_reference_packets(
            rng_authority=reference_authority
        )
    assert calls == 0
    monkeypatch.setattr(synthetic, "apply_filterbank", original_apply_filterbank)

    bundle.numerical_runtime_fingerprint_sha256 = "f" * 64
    with pytest.raises(ValueError, match="revalidated development bundle"):
        v3.require_development_rng_authority(development_authority)
    assert calls == 0


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


def test_dgp_applies_channel_gain_only_to_target_not_confuser(
    monkeypatch: pytest.MonkeyPatch,
    contract: synthetic.SyntheticV3Contract,
    unit_rng,
) -> None:
    monkeypatch.setattr(
        synthetic,
        "_oscillation",
        lambda frequency_hz, phase, drift, timeline: np.full(
            timeline.shape,
            frequency_hz,
            dtype=np.float64,
        ),
    )
    monkeypatch.setattr(
        synthetic,
        "_stationary_ar1",
        lambda standard_normal, *, innovation_sd, rho: np.zeros_like(
            standard_normal,
            dtype=np.float64,
        ),
    )
    participant = synthetic._generate_synthetic_participant_for_test(
        contract,
        family="B1_participant_class_confusion",
        participant_index=0,
        rng_authority=unit_rng,
    )
    factory = synthetic._unit_test_rng_factory(unit_rng)
    spatial = synthetic._rng(
        factory,
        "B1_participant_class_confusion",
        0,
        0,
        0,
        "spatial_signature",
    ).standard_normal(8)
    spatial /= np.linalg.norm(spatial)
    gains = synthetic._rng(
        factory,
        "B1_participant_class_confusion",
        0,
        1,
        0,
        "channel_gain",
    ).lognormal(mean=0.0, sigma=0.12, size=8)
    target = spatial[:, None] * 8.0
    confuser = 0.90 * spatial[:, None] * 8.4
    expected = gains[:, None] * target + confuser
    old_wrong_equation = gains[:, None] * (target + confuser)
    assert np.array_equal(participant.signals[0, 0], np.broadcast_to(expected, (8, 500)))
    assert not np.allclose(participant.signals[0, 0], old_wrong_equation)


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


def test_B4_source_stress_uses_exact_pooled_fallback_for_sparse_interface_channel(
    contract: synthetic.SyntheticV3Contract,
    reference: v3.ContextReference,
    unit_rng,
) -> None:
    payload = v3.context_reference_payload(reference)
    wet = payload["interface_tables"][1]
    assert wet["interface_lookup_key"] == "wet"
    wet["centers"][0] = None
    wet["scales"][0] = None
    wet["observed_counts"][0] = 127
    payload["payload_sha256"] = v3.canonical_payload_sha256(payload)
    sparse_wet = v3.validate_context_reference_payload(payload)
    participant = synthetic._generate_synthetic_participant_for_test(
        contract,
        family="B4_interface_calibrated_impedance_shift",
        participant_index=0,
        rng_authority=unit_rng,
    )
    stressed = synthetic._apply_source_range_stress_for_test(participant, sparse_wet)
    pooled_center = sparse_wet.pooled_table.centers[0]
    pooled_scale = sparse_wet.pooled_table.scales[0]
    assert pooled_center is not None and pooled_scale is not None
    expected = float(np.expm1(pooled_center + 10.0 * pooled_scale))
    for block in synthetic.QUERY_BLOCKS:
        packet = stressed.context_packets("wet")[block - 1]
        assert packet.channel_availability[0] is True
        assert packet.impedance_kohm_by_channel[0] == expected


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


def test_incremental_support_cache_never_reads_beyond_requested_budget(
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
    seen: list[np.ndarray] = []

    def fake_score(_contract, signals):
        seen.append(signals)
        scores = np.broadcast_to(
            np.eye(12, dtype=np.float64),
            (signals.shape[0], 12, 12),
        ).copy()
        scores.setflags(write=False)
        return scores

    monkeypatch.setattr(synthetic, "_score_strict_fbcca_blocks", fake_score)
    loader = (
        synthetic._incremental_strict_fbcca_support_loader_after_authorized_boundary(
            contract,
            participant,
            _issuer=synthetic._STRICT_FBCCA_COMPUTATION_ISSUER,
        )
    )
    assert loader(1).budget == 1
    assert len(seen) == 1
    assert np.shares_memory(seen[0], participant.signals)
    assert np.array_equal(seen[0], participant.signals[:1])
    assert loader(1).budget == 1
    assert len(seen) == 1
    assert loader(3).budget == 3
    assert np.array_equal(seen[1], participant.signals[1:3])
    assert loader(5).budget == 5
    assert np.array_equal(seen[2], participant.signals[3:5])
    assert [values.shape[0] for values in seen] == [1, 2, 2]


def test_incremental_support_cache_runs_real_fbcca_for_one_then_two_block_chunks(
    contract: synthetic.SyntheticV3Contract,
    unit_rng,
) -> None:
    participant = synthetic._generate_synthetic_participant_for_test(
        contract,
        family="N1_clean_anchor",
        participant_index=0,
        rng_authority=unit_rng,
    )
    loader = (
        synthetic._incremental_strict_fbcca_support_loader_after_authorized_boundary(
            contract,
            participant,
            _issuer=synthetic._STRICT_FBCCA_COMPUTATION_ISSUER,
        )
    )
    k1 = loader(1)
    k3 = loader(3)
    k5 = loader(5)
    assert (k1.scores.shape, k3.scores.shape, k5.scores.shape) == (
        (1, 12, 12),
        (3, 12, 12),
        (5, 12, 12),
    )
    assert np.array_equal(k3.scores[:1], k1.scores)
    assert np.array_equal(k5.scores[:3], k3.scores)


def test_strict_products_are_nominal_and_query_mutation_fails_before_support(
    monkeypatch: pytest.MonkeyPatch,
    contract: synthetic.SyntheticV3Contract,
    reference: v3.ContextReference,
    unit_rng,
) -> None:
    _install_fake_scorer(monkeypatch)
    participant = synthetic._generate_synthetic_participant_for_test(
        contract,
        family="N1_clean_anchor",
        participant_index=0,
        rng_authority=unit_rng,
    )
    query = synthetic.produce_strict_fbcca(contract, participant)
    with pytest.raises(TypeError, match="no public constructor"):
        synthetic.StrictFBCCAProduct()
    with pytest.raises(TypeError, match="no public constructor"):
        synthetic.StrictFBCCASupportProduct()
    with pytest.raises(TypeError, match="exact StrictFBCCAProduct"):
        synthetic.require_strict_fbcca_product({"producer_sha256": query.producer_sha256})

    query.scores.setflags(write=True)
    query.scores[0, 0, 0] += 0.5
    support_calls = 0

    def forbidden_support(_budget: int) -> synthetic.StrictFBCCASupportProduct:
        nonlocal support_calls
        support_calls += 1
        raise AssertionError("support loaded before query-product mutation rejection")

    with pytest.raises(ValueError, match="immutable|score bytes"):
        synthetic._evaluate_synthetic_participant_for_test(
            contract,
            participant,
            query,
            reference,
            v3.canonical_operator_grid()[0],
            support_loader=forbidden_support,
        )
    assert support_calls == 0


def test_low_level_synthetic_issuers_require_private_tokens() -> None:
    with pytest.raises(TypeError, match="clean-A recovery"):
        synthetic.AuditedDevelopmentResult()
    with pytest.raises(TypeError, match="exact AuditedDevelopmentResult"):
        synthetic.require_audited_development_result({})
    with pytest.raises(TypeError, match="issuer token"):
        synthetic._issue_synthetic_participant({}, _issuer=object())
    with pytest.raises(TypeError, match="issuer token"):
        synthetic._issue_strict_fbcca_product({}, _issuer=object())
    with pytest.raises(TypeError, match="issuer token"):
        synthetic._issue_strict_fbcca_support_product({}, _issuer=object())
    with pytest.raises(TypeError, match="issuer token"):
        synthetic._issue_validated_development_result(
            {},
            rng_authority=object(),
            validated_reference=object(),
            _issuer=object(),
        )
    with pytest.raises(TypeError, match="issuer token"):
        synthetic._issue_selected_method_proposal(
            {},
            validated_development_result=object(),
            _issuer=object(),
        )
    with pytest.raises(TypeError, match="issuer token"):
        synthetic._issue_audited_development_result({}, object(), _issuer=object())
    with pytest.raises(TypeError, match="issuer token"):
        synthetic._assemble_selected_method_freeze_payload(
            {},
            development_result_file_sha256="a" * 64,
            clean_commit="b" * 40,
            clean_tree="c" * 40,
            _issuer=object(),
        )
    with pytest.raises(TypeError):
        synthetic._issue_strict_fbcca_product({})


def _sensitivity_report_fixture() -> dict:
    primary = {
        endpoint: [float(index) / 1000.0 for index in range(48)]
        for endpoint in synthetic.SENSITIVITY_ENDPOINT_IDS
    }
    uniform = {
        endpoint: [float(index - 1) / 1000.0 for index in range(48)]
        for endpoint in synthetic.SENSITIVITY_ENDPOINT_IDS
    }
    vectors = {
        "schema": (
            "cfeg.metadata-calibration-efficiency-v3."
            "sensitivity-participant-vectors.v1"
        ),
        "participant_count": 48,
        "endpoint_order": list(synthetic.SENSITIVITY_ENDPOINT_IDS),
        "primary_values_by_endpoint": primary,
        "uniform_values_by_endpoint": uniform,
        "primary_values_sha256": synthetic._sensitivity_vector_values_sha256(
            "primary", primary
        ),
        "uniform_values_sha256": synthetic._sensitivity_vector_values_sha256(
            "uniform", uniform
        ),
    }
    uniform_summaries = {}
    resampling_summaries = {}
    for endpoint in synthetic.SENSITIVITY_ENDPOINT_IDS:
        primary_array = np.asarray(primary[endpoint], dtype=np.float64)
        uniform_array = np.asarray(uniform[endpoint], dtype=np.float64)
        uniform_summaries[endpoint] = {
            "metric": (
                "participant_eAUC"
                if endpoint.endswith("eAUC")
                else "participant_k3_balanced_accuracy"
            ),
            "n": 48,
            "reliability_weighted_observed_mean": float(np.mean(primary_array)),
            "uniform_weight_observed_mean": float(np.mean(uniform_array)),
            "paired_weighted_minus_uniform_mean": float(
                np.mean(primary_array - uniform_array)
            ),
        }
        extreme = 17
        family = synthetic._sensitivity_family(endpoint)
        code = synthetic._SENSITIVITY_ENDPOINT_CODES[endpoint]
        resampling_summaries[endpoint] = {
            "n": 48,
            "observed_mean": float(np.mean(primary_array)),
            "sign_flip_draws": synthetic.SENSITIVITY_SIGN_FLIP_DRAWS,
            "sign_flip_extreme_count": extreme,
            "sign_flip_one_sided_p_value": (extreme + 1.0)
            / (synthetic.SENSITIVITY_SIGN_FLIP_DRAWS + 1.0),
            "participant_bootstrap_draws": (
                synthetic.SENSITIVITY_PARTICIPANT_BOOTSTRAP_DRAWS
            ),
            "participant_bootstrap_95_CI_lower": 0.0,
            "participant_bootstrap_95_CI_upper": 1.0,
            "sign_flip_rng_key": synthetic._sensitivity_rng_key(
                family, 1, code, 0
            ),
            "participant_bootstrap_rng_key": synthetic._sensitivity_rng_key(
                family, 1, code, 1
            ),
        }
    return {
        "grid_cell_index": 1,
        "sensitivity_participant_vectors": vectors,
        "uniform_block_weight_sensitivity": {
            "schema": (
                "cfeg.metadata-calibration-efficiency-v3."
                "uniform-block-weight-sensitivity.v1"
            ),
            "promotion_or_selection_use": False,
            "common_weight_rule": "both_p_support_and_g_M_use_exact_one_over_k",
            "endpoints": uniform_summaries,
        },
        "resampling_sensitivity": {
            "schema": (
                "cfeg.metadata-calibration-efficiency-v3."
                "sensitivity-resampling-report.v1"
            ),
            "component": "sensitivity_resampling",
            "promotion_or_selection_use": False,
            "endpoints": resampling_summaries,
        },
    }


def test_sensitivity_vectors_hashes_means_and_p_formula_are_exact() -> None:
    report = _sensitivity_report_fixture()
    primary, uniform = synthetic._validate_gate_report_sensitivities(report)
    assert tuple(primary) == synthetic.SENSITIVITY_ENDPOINT_IDS
    assert tuple(uniform) == synthetic.SENSITIVITY_ENDPOINT_IDS

    bad_mean = deepcopy(report)
    bad_mean["uniform_block_weight_sensitivity"]["endpoints"][
        synthetic.SENSITIVITY_ENDPOINT_IDS[0]
    ]["uniform_weight_observed_mean"] += 1e-12
    with pytest.raises(ValueError, match="participant vectors"):
        synthetic._validate_gate_report_sensitivities(bad_mean)

    bad_p = deepcopy(report)
    bad_p["resampling_sensitivity"]["endpoints"][
        synthetic.SENSITIVITY_ENDPOINT_IDS[0]
    ]["sign_flip_one_sided_p_value"] += 1e-12
    with pytest.raises(ValueError, match="exact formulas"):
        synthetic._validate_gate_report_sensitivities(bad_p)

    bad_type = deepcopy(report)
    bad_type["sensitivity_participant_vectors"][
        "primary_values_by_endpoint"
    ][synthetic.SENSITIVITY_ENDPOINT_IDS[0]][0] = 0
    with pytest.raises(TypeError, match="exact finite floats"):
        synthetic._validate_gate_report_sensitivities(bad_type)


def test_audit_sensitivity_replay_compares_whole_report_without_exposing_rng(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    report = _sensitivity_report_fixture()
    expected_resampling = deepcopy(report["resampling_sensitivity"])
    calls: list[tuple[int, int, int]] = []

    def fake_replay(
        paired_vectors,
        *,
        grid_cell_index,
        rng_factory,
        sign_flip_draws,
        participant_bootstrap_draws,
        require_complete,
        _issuer,
    ):
        assert tuple(paired_vectors) == synthetic.SENSITIVITY_ENDPOINT_IDS
        assert rng_factory is synthetic._frozen_sensitivity_audit_rng_factory
        assert require_complete is True
        assert _issuer is synthetic._SYNTHETIC_RNG_PRODUCER_ISSUER
        calls.append(
            (grid_cell_index, sign_flip_draws, participant_bootstrap_draws)
        )
        return deepcopy(expected_resampling)

    monkeypatch.setattr(
        synthetic,
        "_sensitivity_resampling_with_rng_factory",
        fake_replay,
    )
    synthetic._validate_frozen_sensitivity_replay_for_audit(
        {"complete_grid_gate_report": [report]}
    )
    assert calls == [
        (
            1,
            synthetic.SENSITIVITY_SIGN_FLIP_DRAWS,
            synthetic.SENSITIVITY_PARTICIPANT_BOOTSTRAP_DRAWS,
        )
    ]
    report["resampling_sensitivity"]["endpoints"][
        synthetic.SENSITIVITY_ENDPOINT_IDS[0]
    ]["sign_flip_extreme_count"] += 1
    with pytest.raises(ValueError, match="component-11 replay"):
        synthetic._validate_frozen_sensitivity_replay_for_audit(
            {"complete_grid_gate_report": [report]}
        )


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


def test_holm_step_down_is_deterministic_and_stops_after_first_nonrejection() -> None:
    corrected = synthetic.holm_step_down(
        {"B4": 0.04, "B3": 0.01, "other": 0.001},
        familywise_alpha=0.05,
    )
    assert corrected["other"] == {
        "raw_p_value": 0.001,
        "holm_rank": 1,
        "holm_threshold": pytest.approx(0.05 / 3.0),
        "holm_adjusted_p_value": pytest.approx(0.003),
        "holm_reject": True,
    }
    assert corrected["B3"]["holm_rank"] == 2
    assert corrected["B3"]["holm_reject"] is True
    assert corrected["B4"]["holm_rank"] == 3
    assert corrected["B4"]["holm_reject"] is True

    stopped = synthetic.holm_step_down({"a": 0.03, "b": 0.031})
    assert stopped["a"]["holm_reject"] is False
    assert stopped["b"]["holm_reject"] is False


def test_sensitivity_resampling_uses_exact_keys_counts_and_is_deterministic(unit_rng) -> None:
    vector = np.asarray([0.04, -0.01, 0.02, 0.03, -0.02, 0.01], dtype=np.float64)
    kwargs = {
        "paired_vectors": {"B3_A_QM_minus_A_Q_eAUC": vector},
        "grid_cell_index": 2,
        "rng_authority": unit_rng,
        "sign_flip_draws": 257,
        "participant_bootstrap_draws": 113,
    }
    first = synthetic._sensitivity_resampling_report_for_test(**kwargs)
    second = synthetic._sensitivity_resampling_report_for_test(**kwargs)
    assert first == second
    assert first["promotion_or_selection_use"] is False
    endpoint = first["endpoints"]["B3_A_QM_minus_A_Q_eAUC"]
    assert endpoint["sign_flip_draws"] == 257
    assert endpoint["participant_bootstrap_draws"] == 113
    assert endpoint["sign_flip_rng_key"]["component"] == "sensitivity_resampling"
    assert (
        endpoint["participant_bootstrap_rng_key"]["component"]
        == "sensitivity_resampling"
    )
    assert endpoint["sign_flip_rng_key"] != endpoint["participant_bootstrap_rng_key"]

    other_cell = synthetic._sensitivity_resampling_report_for_test(
        **{**kwargs, "grid_cell_index": 3}
    )
    assert (
        endpoint["sign_flip_rng_key"]
        != other_cell["endpoints"]["B3_A_QM_minus_A_Q_eAUC"]["sign_flip_rng_key"]
    )
    assert synthetic.SENSITIVITY_SIGN_FLIP_DRAWS == 100_000
    assert synthetic.SENSITIVITY_PARTICIPANT_BOOTSTRAP_DRAWS == 10_000


def test_unit_orchestration_smoke_runs_same_scoring_grid_and_serialization_path(
    monkeypatch: pytest.MonkeyPatch,
    contract: synthetic.SyntheticV3Contract,
    reference: v3.ContextReference,
    unit_rng,
) -> None:
    _install_fake_scorer(monkeypatch)
    kwargs = {
        "rng_authority": unit_rng,
        "families": ("N1_clean_anchor",),
        "participant_count": 1,
        "grid_cell_indices": (1,),
    }
    first = synthetic._execute_development_smoke_for_test(contract, reference, **kwargs)
    second = synthetic._execute_development_smoke_for_test(contract, reference, **kwargs)
    assert first == second
    assert first["primary_row_count"] == 16
    assert first["uniform_sensitivity_row_count"] == 16
    assert first["selection_status"] == "FORBIDDEN_IN_UNIT_SMOKE"
    assert first["nominal_development_proof_issued"] is False
    assert first["root_seed"] == _UNIT_ROOT


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
