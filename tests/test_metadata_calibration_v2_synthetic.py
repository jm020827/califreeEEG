from __future__ import annotations

import inspect
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from cfeg.analysis import metadata_calibration_v2_synthetic as synthetic
from cfeg.baselines.fbcca import predict_fbcca, resolve_filterbank_parameters
from cfeg.models import metadata_calibration_v2 as v2_operator

NON_RESERVED_SEED = 17012026


def _fake_product(_contract, signals: np.ndarray) -> synthetic.StrictFBCCAProduct:
    n_blocks, n_classes = signals.shape[:2]
    scores = np.full((n_blocks, n_classes, n_classes), -0.2, dtype=np.float64)
    for block in range(n_blocks):
        for class_index in range(n_classes):
            scores[block, class_index, class_index] = 1.0 + 0.01 * block
            scores[block, class_index, (class_index + 1) % n_classes] = 0.8
    subbands = np.zeros((7, n_blocks, n_classes, 8, 500), dtype=np.float64)
    return synthetic.StrictFBCCAProduct(
        scores=scores,
        subbands=subbands,
        subband_weights=synthetic.P2_SUBBAND_WEIGHTS.copy(),
        producer_sha256="a" * 64,
    )


@pytest.fixture(scope="module")
def contract() -> synthetic.SyntheticContract:
    return synthetic.validate_synthetic_contract()


def test_frozen_r2_plan_filterbank_and_operator_weights_are_bitwise_identical(contract) -> None:
    assert contract.plan_sha256 == synthetic.EXPECTED_SYNTHETIC_PLAN_SHA256
    assert contract.filterbank_sha256 == synthetic.EXPECTED_FILTERBANK_SHA256
    resolved = resolve_filterbank_parameters(contract.filterbank, sfreq=250.0)
    runtime = np.asarray(resolved["weights"], dtype=np.float64)
    plan = np.asarray(contract.plan["waveform"]["exact_subband_weights"], dtype=np.float64)

    assert np.array_equal(runtime, plan)
    assert np.array_equal(runtime, synthetic.P2_SUBBAND_WEIGHTS)
    assert np.array_equal(runtime, v2_operator._P2_SUBBAND_WEIGHTS)
    assert runtime[4].hex() == plan[4].hex() == v2_operator._P2_SUBBAND_WEIGHTS[4].hex()
    assert contract.plan["waveform"]["random_variable_axes"] == {
        "spatial_signature": "participant_by_intended_class_by_channel_shared_across_blocks",
        "class_phase": "participant_by_frequency_class_shared_across_blocks",
        "block_phase_drift": "participant_by_block_shared_across_classes_and_channels",
        "channel_gain": "participant_by_block_by_channel_shared_across_all_class_trials",
        "cross_frequency_spatial_signature": "intended_class_spatial_signature",
        "impedance_noise": "participant_by_block_by_channel_shared_across_all_class_trials",
    }
    assert contract.plan["waveform"]["component_phase_formulas"] == {
        "target_harmonic_h": (
            "h_times_parenthesized_target_frequency_class_phase_plus_block_phase_drift"
        ),
        "cross_frequency_harmonic_h": (
            "h_times_parenthesized_cross_frequency_class_phase_plus_cross_extra_phase_plus_"
            "block_phase_drift"
        ),
    }


def test_plan_byte_drift_and_symlink_are_rejected(tmp_path: Path, contract) -> None:
    drifted = tmp_path / "plan.yaml"
    drifted.write_bytes(contract.plan_path.read_bytes() + b"\n")
    with pytest.raises(ValueError, match="byte hash drifted"):
        synthetic.validate_synthetic_contract(drifted)

    linked = tmp_path / "linked.yaml"
    linked.symlink_to(contract.plan_path)
    with pytest.raises(ValueError, match="nonsymlink"):
        synthetic.validate_synthetic_contract(linked)


def test_keyed_pcg64dxsm_streams_are_loop_order_independent(contract) -> None:
    key = {
        "root_seed": NON_RESERVED_SEED,
        "family": synthetic.FAMILY_NAMES[0],
        "participant_index": 3,
        "block": 4,
        "class_index": 5,
        "component": "innovation_noise",
    }
    first = synthetic.keyed_rng(contract, **key).standard_normal(20)
    synthetic.keyed_rng(
        contract,
        root_seed=NON_RESERVED_SEED,
        family=synthetic.FAMILY_NAMES[-1],
        participant_index=9,
        block=2,
        class_index=1,
        component="metadata_noise",
    ).standard_normal(10_000)
    second = synthetic.keyed_rng(contract, **key).standard_normal(20)

    assert np.array_equal(first, second)
    changed = synthetic.keyed_rng(contract, **{**key, "class_index": 6}).standard_normal(20)
    assert not np.array_equal(first, changed)


@pytest.mark.parametrize("family", synthetic.FAMILY_NAMES)
def test_all_seven_generator_families_follow_the_frozen_formulas(contract, family) -> None:
    participant = synthetic.generate_synthetic_participant(
        contract,
        root_seed=NON_RESERVED_SEED,
        family=family,
        participant_index=2,
    )
    classes = np.arange(12)
    assert participant.signals.shape == (10, 12, 8, 500)
    assert participant.signals.dtype == np.float64
    assert np.isfinite(participant.signals).all()
    assert len(set(participant.partition_sha256s)) == 10

    if family in {
        "B1_participant_specific_class_confusion",
        "N2_random_fit_support_labels",
    }:
        expected_cross = (classes + 1 + 2 % 3) % 12
        assert np.array_equal(participant.generated_cross_classes, np.tile(expected_cross, (10, 1)))
        assert np.array_equal(participant.innovation_sds, np.full((10, 12), 0.85))
    elif family == "B2_participant_phase_spatial_shift":
        assert np.array_equal(participant.generated_cross_classes, np.full((10, 12), -1))
        assert np.array_equal(participant.innovation_sds, np.full((10, 12), 1.10))
    elif family in {
        "B3_context_dependent_support_query_shift",
        "N4_context_independent_of_signal",
    }:
        for block in range(10):
            state = int(participant.signal_states[block])
            offset = 1 if state == -1 else 4
            expected_cross = (classes + offset + 2 % 3) % 12
            expected_sd = 0.85 if state == -1 else 1.30
            expected_phase = 0.0 if state == -1 else np.pi / 2.0
            assert np.array_equal(participant.generated_cross_classes[block], expected_cross)
            assert np.array_equal(participant.innovation_sds[block], np.full(12, expected_sd))
            assert np.array_equal(
                participant.cross_extra_phases[block], np.full(12, expected_phase)
            )
        assert np.isfinite(participant.impedance_kohm).all()
        assert np.all(participant.impedance_kohm > 0.0)
        for block in range(10):
            assert np.array_equal(
                participant.impedance_kohm[block],
                np.broadcast_to(participant.impedance_kohm[block, 0], (12, 8)),
            )
    elif family == "N1_perfect_clean_anchor":
        assert np.array_equal(participant.generated_target_classes, np.tile(classes, (10, 1)))
        assert np.array_equal(participant.innovation_sds, np.full((10, 12), 0.05))
    elif family == "N3_nonstationary_corrupted_calibration":
        for block in range(1, 6):
            assert np.array_equal(
                participant.generated_target_classes[block - 1], (classes + block) % 12
            )
        assert np.array_equal(participant.generated_target_classes[5:], np.tile(classes, (5, 1)))

    if family == "B3_context_dependent_support_query_shift":
        assert np.array_equal(participant.signal_states, [-1, 1] * 5)
        assert np.array_equal(participant.context_states, participant.signal_states)
    if family == "N4_context_independent_of_signal":
        # The two state vectors use disjoint complete keys; this frozen seed witnesses that fact.
        assert not np.array_equal(participant.signal_states, participant.context_states)
    if family == "N2_random_fit_support_labels":
        for block in range(5):
            difference = (
                participant.recorded_support_labels[block] - participant.true_labels[block]
            ) % 12
            assert np.all(difference == difference[0])
            assert int(difference[0]) in range(1, 12)
        assert np.array_equal(participant.recorded_support_labels[5:], participant.true_labels[5:])


def test_generation_is_repeatable_and_arrays_are_immutable(contract) -> None:
    first = synthetic.generate_synthetic_participant(
        contract,
        root_seed=NON_RESERVED_SEED,
        family=synthetic.FAMILY_NAMES[0],
        participant_index=0,
    )
    second = synthetic.generate_synthetic_participant(
        contract,
        root_seed=NON_RESERVED_SEED,
        family=synthetic.FAMILY_NAMES[0],
        participant_index=0,
    )
    assert np.array_equal(first.signals, second.signals)
    assert first.partition_sha256s == second.partition_sha256s
    with pytest.raises(ValueError):
        first.true_labels[0, 0] = 9


def test_strict_fbcca_matches_public_frozen_producer_without_labels(contract) -> None:
    timeline = np.arange(500, dtype=np.float64) / 250.0
    signals = np.empty((1, 12, 8, 500), dtype=np.float64)
    mixing = np.linspace(0.5, 1.2, 8)[:, None]
    for class_index, frequency in enumerate(8.0 + 0.4 * np.arange(12)):
        signals[0, class_index] = mixing * np.sin(2 * np.pi * frequency * timeline)[None, :]
    product = synthetic.produce_strict_fbcca(contract, signals)
    _, expected = predict_fbcca(
        signals[0, 4], 8.0 + 0.4 * np.arange(12), 250.0, contract.filterbank
    )

    assert product.scores.shape == (1, 12, 12)
    assert product.subbands.shape == (7, 1, 12, 8, 500)
    assert np.array_equal(product.subband_weights, synthetic.P2_SUBBAND_WEIGHTS)
    assert np.array_equal(product.scores[0, 4], expected)
    assert "label" not in inspect.signature(synthetic.produce_strict_fbcca).parameters


def test_hard_invariants_cover_nulls_order_equivariance_and_provenance(contract) -> None:
    participant = synthetic.generate_synthetic_participant(
        contract,
        root_seed=NON_RESERVED_SEED,
        family=synthetic.FAMILY_NAMES[0],
        participant_index=0,
    )
    assertions = synthetic.validate_hard_assertions(
        participant, _fake_product(contract, participant.signals), contract
    )
    assert assertions == {name: True for name in contract.plan["hard_assertions_before_efficacy"]}


def test_operator_boundary_has_no_query_label_or_identity_capability(contract, monkeypatch) -> None:
    participant = synthetic.generate_synthetic_participant(
        contract,
        root_seed=NON_RESERVED_SEED,
        family=synthetic.FAMILY_NAMES[0],
        participant_index=0,
    )
    product = _fake_product(contract, participant.signals)
    original = v2_operator.apply_v2_safe_operator
    calls = []

    def guarded(*args, **kwargs):
        assert not ({"query_labels", "query_identity", "participant_index", "row_id"} & set(kwargs))
        calls.append(set(kwargs))
        return original(*args, **kwargs)

    monkeypatch.setattr(v2_operator, "apply_v2_safe_operator", guarded)
    rows = synthetic.evaluate_synthetic_participant(participant, product, contract)
    assert rows
    assert calls
    b3_methods = {
        item[0]
        for item in synthetic._method_specs_for_family("B3_context_dependent_support_query_shift")
    }
    assert {
        "P1_A_QM_correct",
        "P1_A_QM_pair_shuffled",
        "P1_A_QM_stale",
        "P1_A_QM_opposite_interface",
        "P1_A_QM_all_missing",
    } <= b3_methods


def test_query_label_poison_is_not_touched_by_k1_operator_and_k5_prefix_chain_is_exact(
    contract,
) -> None:
    participant = synthetic.generate_synthetic_participant(
        contract,
        root_seed=NON_RESERVED_SEED,
        family=synthetic.FAMILY_NAMES[0],
        participant_index=1,
    )
    product = _fake_product(contract, participant.signals)

    class PoisonLabels:
        def __getitem__(self, key):
            raise AssertionError("query labels were touched by the k=1 operator path")

    poisoned = replace(participant, true_labels=PoisonLabels())
    output, gate = synthetic._apply_final_query_operator(
        poisoned,
        product,
        contract,
        budget=1,
        operator="score_prototype_shrinkage",
        variant="A_Q",
    )
    assert gate is None
    assert output.support_depth == 1

    decision = synthetic._prequential_gate(
        participant,
        product,
        contract,
        budget=5,
        operator="score_prototype_shrinkage",
        variant="A_Q",
        context_control="correct",
    )
    assert decision.evaluated_blocks == (2, 3, 4, 5)
    assert [len(item.fit_blocks) for item in decision.fold_provenance] == [1, 2, 3, 4]
    for previous, current in zip(decision.fold_provenance, decision.fold_provenance[1:]):
        assert current.fit_block_partition_sha256s == (
            *previous.fit_block_partition_sha256s,
            previous.evaluation_partition_sha256,
        )


def test_nonreserved_small_diagnostic_streams_and_reserved_seeds_have_no_alternate_path(
    contract,
) -> None:
    for reserved in synthetic.RESERVED_ROOT_SEEDS:
        with pytest.raises(ValueError, match="Reserved"):
            synthetic.run_synthetic_diagnostic(
                root_seed=reserved,
                participants_per_family=2,
                families=(synthetic.FAMILY_NAMES[0],),
                fbcca_producer=_fake_product,
            )
        with pytest.raises(ValueError, match="no alternate"):
            synthetic._execute_synthetic(
                contract,
                phase="diagnostic",
                root_seed=reserved,
                participants_per_family=2,
                families=(synthetic.FAMILY_NAMES[0],),
                sign_flip_draws=8,
                bootstrap_draws=8,
                fbcca_producer=_fake_product,
            )
    result = synthetic.run_synthetic_diagnostic(
        root_seed=NON_RESERVED_SEED,
        participants_per_family=2,
        families=(synthetic.FAMILY_NAMES[0],),
        sign_flip_draws=8,
        bootstrap_draws=8,
        fbcca_producer=_fake_product,
    )
    assert result["phase"] == "diagnostic"
    assert result["status"] == "engineering_only"
    assert result["hard_assertions_completed_before_metrics"] is True
    assert result["promotion"]["evaluated"] is False
    assert result["root_seed"] == NON_RESERVED_SEED


def _passing_metric_rows() -> list[dict]:
    rows = []
    method_by_family = {
        "B1_participant_specific_class_confusion": ("P1_A_Q",),
        "B2_participant_phase_spatial_shift": ("P2_A_Q",),
        "B3_context_dependent_support_query_shift": (
            "P1_A_Q",
            "P1_A_QM_correct",
            "P1_A_QM_pair_shuffled",
            "P1_A_QM_stale",
            "P1_A_QM_opposite_interface",
            "P1_A_QM_all_missing",
        ),
        "N1_perfect_clean_anchor": ("P1_A_Q", "P2_A_Q"),
        "N2_random_fit_support_labels": ("P1_A_Q",),
        "N3_nonstationary_corrupted_calibration": ("P1_A_Q",),
        "N4_context_independent_of_signal": (
            "P1_A_Q",
            "P1_A_QM_correct",
            "P1_A_QM_pair_shuffled",
            "P1_A_QM_all_missing",
        ),
    }
    for family, methods in method_by_family.items():
        for participant in range(64):
            for budget in synthetic.SUPPORT_BUDGETS:
                rows.append(
                    {
                        "family": family,
                        "participant_index": participant,
                        "method": "A0",
                        "budget": budget,
                        "balanced_accuracy": 0.50,
                        "gate_enabled": None,
                    }
                )
                for method in methods:
                    value = 0.50
                    if family == "B1_participant_specific_class_confusion" and method == "P1_A_Q":
                        value += {0: 0.0, 1: 0.03, 3: 0.06, 5: 0.06}[budget]
                    if family == "B2_participant_phase_spatial_shift" and method == "P2_A_Q":
                        value += {0: 0.0, 1: 0.03, 3: 0.06, 5: 0.06}[budget]
                    if (
                        family == "B3_context_dependent_support_query_shift"
                        and method == "P1_A_QM_correct"
                    ):
                        value += {0: 0.0, 1: 0.03, 3: 0.06, 5: 0.06}[budget]
                    gate = (
                        False
                        if family
                        in {
                            "N2_random_fit_support_labels",
                            "N3_nonstationary_corrupted_calibration",
                        }
                        and budget in {3, 5}
                        else None
                    )
                    rows.append(
                        {
                            "family": family,
                            "participant_index": participant,
                            "method": method,
                            "budget": budget,
                            "balanced_accuracy": value,
                            "gate_enabled": gate,
                        }
                    )
    return rows


def test_frozen_t_cp_helpful_null_pairing_and_adversarial_decisions(contract) -> None:
    summaries, promotion = synthetic.summarize_synthetic_metrics(
        _passing_metric_rows(),
        contract=contract,
        root_seed=NON_RESERVED_SEED,
        families=synthetic.FAMILY_NAMES,
        sign_flip_draws=8,
        bootstrap_draws=8,
    )
    assert summaries
    assert promotion["all_requirements_passed"] is True
    assert promotion["helpful_families"]["passed_count"] == 3
    assert promotion["B3_pairing_mechanism"]["passed"] is True
    assert promotion["null_noninferiority"]["passed"] is True
    assert promotion["adversarial_prequential_abstention"]["rate"] == 1.0
    assert promotion["severe_harm"]["passed"] is True
    assert synthetic._clopper_pearson_upper(0, 64) < 0.10


def test_participant_cache_is_bound_exclusive_and_pickle_free(tmp_path: Path, contract) -> None:
    participant = synthetic.generate_synthetic_participant(
        contract,
        root_seed=NON_RESERVED_SEED,
        family=synthetic.FAMILY_NAMES[0],
        participant_index=0,
    )
    product = _fake_product(contract, participant.signals)
    path = tmp_path / "participant-cache.npz"
    synthetic.write_strict_fbcca_cache_exclusive(
        path, product, contract=contract, participant=participant
    )
    loaded = synthetic.load_strict_fbcca_cache(path, contract=contract, participant=participant)
    assert np.array_equal(loaded.scores, product.scores)
    assert np.array_equal(loaded.subband_weights, synthetic.P2_SUBBAND_WEIGHTS)
    with pytest.raises(FileExistsError):
        synthetic.write_strict_fbcca_cache_exclusive(
            path, product, contract=contract, participant=participant
        )

    different = synthetic.generate_synthetic_participant(
        contract,
        root_seed=NON_RESERVED_SEED + 1,
        family=synthetic.FAMILY_NAMES[0],
        participant_index=0,
    )
    with pytest.raises(ValueError, match="provenance"):
        synthetic.load_strict_fbcca_cache(path, contract=contract, participant=different)


def _authorized_record(
    contract: synthetic.SyntheticContract,
    output: Path,
    identity: synthetic.GitIdentity,
    development_result: Path,
) -> dict:
    record = synthetic.build_lockbox_authorization_template(
        contract,
        output_path=output,
        development_result_path=development_result,
        identity=identity,
    )
    record.pop("authorization_receipt_sha256")
    record.update(
        {
            "status": "authorized_for_one_time_execution",
            "authorized": True,
            "authorized_by": "independent-test-authority",
            "authorization_basis": "unit-test-only-no-reserved-seed-execution",
            "one_time_nonce_sha256": "9" * 64,
        }
    )
    record["test_evidence"] = {
        **record["test_evidence"],
        "status": "passed",
        "exit_code": 0,
    }
    return synthetic.seal_authorization_record(record)


def _write_development_result(
    contract: synthetic.SyntheticContract,
    path: Path,
    identity: synthetic.GitIdentity,
) -> None:
    payload = {
        "schema": synthetic.SYNTHETIC_RESULT_SCHEMA,
        "phase": "development",
        "status": "engineering_only",
        "plan_sha256": contract.plan_sha256,
        "filterbank_sha256": contract.filterbank_sha256,
        "diagnostic_operator_configs_sha256": contract.diagnostic_operator_configs_sha256,
        "root_seed": synthetic.DEVELOPMENT_ROOT_SEED,
        "participants_per_family": 64,
        "families": list(synthetic.FAMILY_NAMES),
        "generator_commit": identity.commit,
        "generator_tree": identity.tree,
        "generator_source_bundle_sha256": identity.source_bundle_sha256,
    }
    payload["result_sha256"] = synthetic._canonical_json_sha256(payload)
    synthetic.write_json_exclusive(path, payload)


def test_lockbox_authorization_binds_clean_commit_tree_tests_configs_and_unused_path(
    tmp_path: Path, contract
) -> None:
    identity = synthetic.GitIdentity(
        commit="1" * 40,
        tree="2" * 40,
        source_bundle_sha256="3" * 64,
        clean=True,
    )
    output = tmp_path / "lockbox.json"
    development = tmp_path / "development.json"
    _write_development_result(contract, development, identity)
    record = _authorized_record(contract, output, identity, development)
    observed = synthetic.validate_lockbox_authorization(
        record, contract=contract, output_path=output, identity=identity
    )
    assert observed == record["authorization_receipt_sha256"]

    tampered = {**record, "generator_tree": "4" * 40}
    with pytest.raises(ValueError, match="self-hash"):
        synthetic.validate_lockbox_authorization(
            tampered, contract=contract, output_path=output, identity=identity
        )

    dirty = synthetic.GitIdentity(**{**identity.__dict__, "clean": False})
    with pytest.raises(ValueError, match="clean git"):
        synthetic.validate_lockbox_authorization(
            record, contract=contract, output_path=output, identity=dirty
        )

    synthetic.write_json_exclusive(output, {"used": True})
    with pytest.raises(FileExistsError):
        synthetic.validate_lockbox_authorization(
            record, contract=contract, output_path=output, identity=identity
        )
    with pytest.raises(FileExistsError):
        synthetic.write_json_exclusive(output, {"overwrite": True})
    broken_link = tmp_path / "linked-output.json"
    broken_link.symlink_to(tmp_path / "missing-target.json")
    with pytest.raises((ValueError, FileExistsError)):
        synthetic.write_json_exclusive(broken_link, {"forbidden": True})


def test_invalid_lockbox_receipt_fails_before_claim_or_reserved_execution(
    tmp_path: Path, contract, monkeypatch
) -> None:
    output = tmp_path / "never-created.json"
    called = False

    def forbidden_execute(*args, **kwargs):
        nonlocal called
        called = True
        raise AssertionError("reserved lockbox execution must not start")

    monkeypatch.setattr(synthetic, "_execute_synthetic", forbidden_execute)
    with pytest.raises(ValueError, match="exact frozen fields"):
        synthetic.run_lockbox_to_path(contract, authorization={}, output_path=output)
    assert called is False
    assert not output.exists()
    assert not Path(str(output) + ".lockbox-claim.json").exists()


def test_prepare_cli_never_runs_a_reserved_seed(tmp_path: Path, monkeypatch) -> None:
    import importlib.util

    script = Path("scripts/run_metadata_calibration_v2_synthetic.py").resolve()
    monkeypatch.syspath_prepend(str(script.parent))
    spec = importlib.util.spec_from_file_location("synthetic_cli", script)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    output = tmp_path / "prepare.json"

    def forbidden(*args, **kwargs):
        raise AssertionError("prepare must not execute any synthetic seed")

    monkeypatch.setattr(module, "run_development_to_path", forbidden)
    monkeypatch.setattr(module, "run_lockbox_to_path", forbidden)
    assert module.main(["prepare", "--output", str(output)]) == 0
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["status"] == "validated_no_outcome_execution"
    assert payload["reserved_seed_executed"] is False
