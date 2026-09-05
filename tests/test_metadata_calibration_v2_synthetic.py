from __future__ import annotations

import hashlib
import importlib.util
import inspect
import json
import os
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from cfeg.analysis import metadata_calibration_v2_synthetic as synthetic
from cfeg.baselines.fbcca import predict_fbcca, resolve_filterbank_parameters
from cfeg.models import metadata_calibration_v2 as v2_operator

NON_RESERVED_SEED = 17012026


@pytest.fixture(scope="module")
def contract() -> synthetic.SyntheticContract:
    return synthetic.validate_synthetic_contract()


def _fake_product(_contract, signals: np.ndarray) -> synthetic.StrictFBCCAProduct:
    n_blocks, n_classes = signals.shape[:2]
    scores = np.full((n_blocks, n_classes, n_classes), -0.2, dtype=np.float64)
    for block in range(n_blocks):
        for class_index in range(n_classes):
            scores[block, class_index, class_index] = 1.0 + 0.01 * block
            scores[block, class_index, (class_index + 1) % n_classes] = 0.8
    return synthetic.StrictFBCCAProduct(
        scores=scores,
        subbands=np.zeros((7, n_blocks, n_classes, 8, 500), dtype=np.float64),
        subband_weights=synthetic.P2_SUBBAND_WEIGHTS.copy(),
        producer_sha256="a" * 64,
    )


def _contract_with_paths(
    tmp_path: Path, contract: synthetic.SyntheticContract
) -> synthetic.SyntheticContract:
    plan = deepcopy(contract.plan)
    root = tmp_path / "artifacts"
    synthetic_dir = root / "synthetic"
    claims = root / "claims"
    synthetic_dir.mkdir(mode=0o700, parents=True)
    claims.mkdir(mode=0o700, parents=True)
    synthetic_dir.chmod(0o700)
    claims.chmod(0o700)
    paths = plan["freeze_and_stopping"]["canonical_execution_paths"]
    paths.update(
        {
            "preparation_receipt": str(synthetic_dir / "preparation.json"),
            "development_result": str(synthetic_dir / "development.json"),
            "full_suite_test_evidence": str(synthetic_dir / "tests.json"),
            "beacon_receipt": str(synthetic_dir / "beacon.json"),
            "lockbox_authorization": str(synthetic_dir / "authorization.json"),
            "lockbox_result": str(synthetic_dir / "lockbox.json"),
            "lockbox_terminal_receipt": str(synthetic_dir / "lockbox.terminal.json"),
            "seed_global_lockbox_claim": str(claims / "global.json"),
        }
    )
    return replace(contract, plan=plan)


def _retarget_beacon(
    contract: synthetic.SyntheticContract, target: datetime
) -> synthetic.SyntheticContract:
    plan = deepcopy(contract.plan)
    target = target.astimezone(timezone.utc)
    timestamp = target.isoformat(timespec="milliseconds").replace("+00:00", "Z")
    milliseconds = int(target.timestamp() * 1000)
    statement = f"cfeg.synthetic-unit-test|{timestamp}"
    beacon = plan["rng"]["independent_lockbox_root_seed"]
    beacon.update(
        {
            "target_timestamp_utc": timestamp,
            "target_timestamp_unix_milliseconds": milliseconds,
            "exact_endpoint": (
                f"https://beacon.nist.gov/beacon/2.0/pulse/time/{milliseconds}"
            ),
            "required_exact_pulse_timestamp": timestamp,
            "statement_utf8": statement,
            "statement_sha256": hashlib.sha256(statement.encode()).hexdigest(),
        }
    )
    return replace(contract, plan=plan)


class _FrozenDateTime(datetime):
    observed_now = datetime.now(timezone.utc)

    @classmethod
    def now(cls, tz=None):
        value = cls.observed_now
        return value if tz is None else value.astimezone(tz)


def _identity() -> synthetic.GitIdentity:
    return synthetic.GitIdentity(
        commit="1" * 40,
        tree="2" * 40,
        source_bundle_sha256="3" * 64,
        clean=True,
        commit_timestamp_utc="2026-09-05T18:00:00Z",
    )


def test_v11_contract_filterbank_and_operator_weights_are_bitwise_exact(contract) -> None:
    assert contract.plan_sha256 == synthetic.EXPECTED_SYNTHETIC_PLAN_SHA256
    assert (
        contract.plan["generator_revision"] == "v11_terminal_receipt_precedence"
    )
    resolved = np.asarray(
        resolve_filterbank_parameters(contract.filterbank, sfreq=250.0)["weights"],
        dtype=np.float64,
    )
    plan = np.asarray(contract.plan["waveform"]["exact_subband_weights"], dtype=np.float64)
    assert np.array_equal(resolved, plan)
    assert np.array_equal(resolved, synthetic.P2_SUBBAND_WEIGHTS)
    assert np.array_equal(resolved, v2_operator._P2_SUBBAND_WEIGHTS)
    assert resolved[4].hex() == plan[4].hex() == v2_operator._P2_SUBBAND_WEIGHTS[4].hex()


def test_plan_byte_drift_and_symlink_are_rejected(tmp_path: Path, contract) -> None:
    drifted = tmp_path / "plan.yaml"
    drifted.write_bytes(contract.plan_path.read_bytes() + b"\n")
    with pytest.raises(ValueError, match="byte hash drifted"):
        synthetic.validate_synthetic_contract(drifted)
    linked = tmp_path / "linked.yaml"
    linked.symlink_to(contract.plan_path)
    with pytest.raises(ValueError, match="symlink"):
        synthetic.validate_synthetic_contract(linked)


def test_keyed_streams_are_order_independent_and_all_static_reserved_apis_reject(
    contract,
) -> None:
    key = {
        "root_seed": NON_RESERVED_SEED,
        "family": synthetic.FAMILY_NAMES[0],
        "participant_index": 3,
        "block": 4,
        "class_index": 5,
        "component": "innovation_noise",
    }
    first = synthetic.keyed_rng(contract, **key).standard_normal(20)
    synthetic.keyed_rng(contract, **{**key, "family": synthetic.FAMILY_NAMES[-1]}).normal(size=100)
    assert np.array_equal(first, synthetic.keyed_rng(contract, **key).standard_normal(20))

    reserved = {
        int(contract.plan["rng"]["development_root_seed"]),
        int(contract.plan["rng"]["superseded_unexecuted_plaintext_lockbox_root_seed"]),
    }
    for seed in reserved:
        with pytest.raises(ValueError, match="internal governed"):
            synthetic.keyed_rng(contract, **{**key, "root_seed": seed})
        with pytest.raises(ValueError, match="internal governed"):
            synthetic.generate_synthetic_participant(
                contract,
                root_seed=seed,
                family=synthetic.FAMILY_NAMES[0],
                participant_index=0,
            )
        with pytest.raises(ValueError, match="internal governed"):
            synthetic.summarize_synthetic_metrics(
                [],
                contract=contract,
                root_seed=seed,
                families=(),
                sign_flip_draws=1,
                bootstrap_draws=1,
            )
        poisoned = replace(
            synthetic.generate_synthetic_participant(
                contract,
                root_seed=NON_RESERVED_SEED,
                family=synthetic.FAMILY_NAMES[0],
                participant_index=0,
            ),
            root_seed=seed,
        )
        product = _fake_product(contract, poisoned.signals)
        with pytest.raises(ValueError, match="internal governed"):
            synthetic.evaluate_synthetic_participant(poisoned, product, contract)
        with pytest.raises(ValueError, match="internal governed"):
            synthetic.validate_hard_assertions(poisoned, product, contract)
        with pytest.raises(ValueError, match="internal governed"):
            synthetic.write_strict_fbcca_cache_exclusive(
                Path("unreachable-reserved-cache.npz"),
                product,
                contract=contract,
                participant=poisoned,
            )
        with pytest.raises(ValueError, match="Reserved"):
            synthetic.run_synthetic_diagnostic(
                root_seed=seed,
                participants_per_family=2,
                families=(synthetic.FAMILY_NAMES[0],),
                fbcca_producer=_fake_product,
            )
    with pytest.raises(ValueError, match="internal governed"):
        synthetic.run_development(contract)


@pytest.mark.parametrize("family", synthetic.FAMILY_NAMES)
def test_all_seven_families_follow_frozen_axes_and_state_formulas(contract, family) -> None:
    participant = synthetic.generate_synthetic_participant(
        contract, root_seed=NON_RESERVED_SEED, family=family, participant_index=2
    )
    classes = np.arange(12)
    assert participant.signals.shape == (10, 12, 8, 500)
    assert np.isfinite(participant.signals).all()
    assert len(set(participant.partition_sha256s)) == 10
    if family in {synthetic.FAMILY_NAMES[0], synthetic.FAMILY_NAMES[4]}:
        expected = (classes + 3) % 12
        assert np.array_equal(participant.generated_cross_classes, np.tile(expected, (10, 1)))
    elif family == synthetic.FAMILY_NAMES[1]:
        assert np.array_equal(participant.innovation_sds, np.full((10, 12), 1.10))
    elif family in {synthetic.FAMILY_NAMES[2], synthetic.FAMILY_NAMES[6]}:
        for block in range(10):
            state = int(participant.signal_states[block])
            expected = (classes + (3 if state == -1 else 6)) % 12
            assert np.array_equal(participant.generated_cross_classes[block], expected)
            assert np.array_equal(
                participant.impedance_kohm[block],
                np.broadcast_to(participant.impedance_kohm[block, 0], (12, 8)),
            )
    elif family == synthetic.FAMILY_NAMES[3]:
        assert np.array_equal(participant.innovation_sds, np.full((10, 12), 0.05))
    elif family == synthetic.FAMILY_NAMES[5]:
        for block in range(1, 6):
            assert np.array_equal(
                participant.generated_target_classes[block - 1], (classes + block) % 12
            )
        assert np.array_equal(participant.generated_target_classes[5:], np.tile(classes, (5, 1)))
    if family == synthetic.FAMILY_NAMES[2]:
        assert np.array_equal(participant.signal_states, [-1, 1] * 5)
        assert np.array_equal(participant.context_states, participant.signal_states)
    if family == synthetic.FAMILY_NAMES[4]:
        for block in range(5):
            delta = (
                participant.recorded_support_labels[block] - participant.true_labels[block]
            ) % 12
            assert np.all(delta == delta[0]) and int(delta[0]) in range(1, 12)


def test_exact_waveform_reconstruction_ar1_gain_placement_and_golden_hashes(contract) -> None:
    family = synthetic.FAMILY_NAMES[0]
    participant_index = 2
    participant = synthetic.generate_synthetic_participant(
        contract,
        root_seed=NON_RESERVED_SEED,
        family=family,
        participant_index=participant_index,
    )
    class_index = 3
    n_channels = 8
    timeline = np.arange(500, dtype=np.float64) / 250.0
    spatial = synthetic.keyed_rng(
        contract,
        root_seed=NON_RESERVED_SEED,
        family=family,
        participant_index=participant_index,
        block=0,
        class_index=class_index,
        component="spatial_signature",
    ).standard_normal(n_channels)
    spatial /= np.linalg.norm(spatial)
    phases = []
    for frequency_class in range(12):
        phases.append(
            synthetic.keyed_rng(
                contract,
                root_seed=NON_RESERVED_SEED,
                family=family,
                participant_index=participant_index,
                block=0,
                class_index=frequency_class,
                component="class_phase",
            ).uniform(-np.pi, np.pi)
        )
    drift = synthetic.keyed_rng(
        contract,
        root_seed=NON_RESERVED_SEED,
        family=family,
        participant_index=participant_index,
        block=1,
        class_index=0,
        component="block_phase_drift",
    ).normal(0.0, 0.08)
    gains = synthetic.keyed_rng(
        contract,
        root_seed=NON_RESERVED_SEED,
        family=family,
        participant_index=participant_index,
        block=1,
        class_index=0,
        component="channel_gain",
    ).lognormal(0.0, 0.12, n_channels)
    cross_class = (class_index + 1 + participant_index % 3) % 12

    def oscillation(frequency_class: int, phase: float) -> np.ndarray:
        fundamental = 2 * np.pi * (8.0 + 0.4 * frequency_class) * timeline
        return np.sin(fundamental + phase + drift) + 0.35 * np.sin(
            2 * (fundamental + phase + drift)
        )

    signal = spatial[:, None] * oscillation(class_index, phases[class_index])
    signal += 0.90 * spatial[:, None] * oscillation(cross_class, phases[cross_class])
    z = synthetic.keyed_rng(
        contract,
        root_seed=NON_RESERVED_SEED,
        family=family,
        participant_index=participant_index,
        block=1,
        class_index=class_index,
        component="innovation_noise",
    ).standard_normal((8, 500))
    noise = np.empty_like(z)
    noise[:, 0] = 0.85 * z[:, 0] / np.sqrt(1 - 0.55**2)
    for sample in range(1, 500):
        noise[:, sample] = 0.55 * noise[:, sample - 1] + 0.85 * z[:, sample]
    expected = gains[:, None] * signal + noise
    assert np.array_equal(participant.signals[0, class_index], expected)
    assert synthetic._array_sha256(participant.signals) == (
        "b7bb38da05dc991fa5d52fead2b87a62c80f0ee227feee72926057b186bf8845"
    )
    assert participant.partition_sha256s[0] == (
        "a7187309efa9e28458eb42009c8a50d693a8ac28b462bf515bfbbe5cf7aee4a5"
    )

    n3 = synthetic.generate_synthetic_participant(
        contract,
        root_seed=NON_RESERVED_SEED,
        family=synthetic.FAMILY_NAMES[5],
        participant_index=participant_index,
    )
    assert synthetic._array_sha256(n3.signals) == (
        "677dc1bd8ccfe0eebc06b89ab97b63248a2d6b36bf1640cb6028bc571c6f9102"
    )


def test_strict_fbcca_is_the_exact_seven_band_label_free_producer(contract) -> None:
    timeline = np.arange(500, dtype=np.float64) / 250.0
    signals = np.empty((1, 12, 8, 500), dtype=np.float64)
    mixing = np.linspace(0.5, 1.2, 8)[:, None]
    for class_index, frequency in enumerate(8.0 + 0.4 * np.arange(12)):
        signals[0, class_index] = mixing * np.sin(2 * np.pi * frequency * timeline)[None, :]
    product = synthetic.produce_strict_fbcca(contract, signals)
    _, expected = predict_fbcca(
        signals[0, 4], 8.0 + 0.4 * np.arange(12), 250.0, contract.filterbank
    )
    assert product.subbands.shape == (7, 1, 12, 8, 500)
    assert np.array_equal(product.scores[0, 4], expected)
    assert np.array_equal(product.subband_weights, synthetic.P2_SUBBAND_WEIGHTS)
    assert "label" not in inspect.signature(synthetic.produce_strict_fbcca).parameters


def test_all_prefix_depth_context_donors_are_exactly_support_local(contract) -> None:
    participant = synthetic.generate_synthetic_participant(
        contract,
        root_seed=NON_RESERVED_SEED,
        family=synthetic.FAMILY_NAMES[2],
        participant_index=4,
    )
    global_order = synthetic.keyed_rng(
        contract,
        root_seed=NON_RESERVED_SEED,
        family=participant.family,
        participant_index=participant.participant_index,
        block=0,
        class_index=0,
        component="metadata_shuffle",
    ).permutation(np.arange(1, 6))
    for depth in range(1, 6):
        prefix = tuple(range(1, depth + 1))
        expected_shuffle = tuple(int(value) for value in global_order if value in prefix)
        for control in synthetic.CONTEXT_CONTROLS:
            donors = synthetic._metadata_source_blocks(participant, prefix, control, contract)
            assert set(donors) == set(prefix)
            assert len(donors) == len(prefix)
        assert (
            synthetic._metadata_source_blocks(participant, prefix, "pair_shuffled", contract)
            == expected_shuffle
        )
        assert (
            synthetic._metadata_source_blocks(participant, prefix, "stale", contract)
            == prefix[-1:] + prefix[:-1]
        )
    assert synthetic._metadata_source_blocks(participant, (1,), "stale", contract) == (1,)
    assert synthetic._metadata_source_blocks(participant, (1,), "pair_shuffled", contract) == (1,)
    for bad in ((1, 3), (2,), (1, 2, 3, 4, 5, 6)):
        with pytest.raises(ValueError, match="exact current prefix"):
            synthetic._metadata_source_blocks(participant, bad, "stale", contract)


def test_hard_invariants_and_prequential_chain_cover_the_frozen_contract(contract) -> None:
    participant = synthetic.generate_synthetic_participant(
        contract,
        root_seed=NON_RESERVED_SEED,
        family=synthetic.FAMILY_NAMES[0],
        participant_index=1,
    )
    product = _fake_product(contract, participant.signals)
    assertions = synthetic.validate_hard_assertions(participant, product, contract)
    assert assertions == {name: True for name in contract.plan["hard_assertions_before_efficacy"]}
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


def test_query_labels_are_not_an_operator_capability_or_touched_at_k1(
    contract, monkeypatch
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
            raise AssertionError(f"query labels were touched at {key!r}")

    original = v2_operator.apply_v2_safe_operator
    calls = []

    def guarded(*args, **kwargs):
        assert not {
            "query_label",
            "query_labels",
            "query_identity",
            "participant_index",
            "sample_id",
            "row_id",
        } & set(kwargs)
        calls.append(kwargs)
        return original(*args, **kwargs)

    monkeypatch.setattr(v2_operator, "apply_v2_safe_operator", guarded)
    output, gate = synthetic._apply_final_query_operator(
        replace(participant, true_labels=PoisonLabels()),
        product,
        contract,
        budget=1,
        operator="score_prototype_shrinkage",
        variant="A_Q",
    )
    assert gate is None and output.support_depth == 1 and calls


def test_nonreserved_small_diagnostic_streams_one_participant_at_a_time(contract) -> None:
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
    assert result["promotion"]["evaluated"] is False


def _passing_metric_rows() -> list[dict]:
    rows = []
    for family in synthetic.FAMILY_NAMES:
        methods = ["A0", *(item[0] for item in synthetic._method_specs_for_family(family))]
        for participant in range(64):
            for budget in synthetic.SUPPORT_BUDGETS:
                for method in methods:
                    value = 0.50
                    if family == synthetic.FAMILY_NAMES[0] and method == "P1_A_Q":
                        value += {0: 0.0, 1: 0.03, 3: 0.06, 5: 0.06}[budget]
                    if family == synthetic.FAMILY_NAMES[1] and method == "P2_A_Q":
                        value += {0: 0.0, 1: 0.03, 3: 0.06, 5: 0.06}[budget]
                    if family == synthetic.FAMILY_NAMES[2] and method == "P1_A_QM_correct":
                        value += {0: 0.0, 1: 0.03, 3: 0.06, 5: 0.06}[budget]
                    rows.append(
                        {
                            "family": family,
                            "participant_index": participant,
                            "method": method,
                            "budget": budget,
                            "balanced_accuracy": value,
                            "gate_enabled": (
                                False
                                if family in {synthetic.FAMILY_NAMES[4], synthetic.FAMILY_NAMES[5]}
                                and method == "P1_A_Q"
                                and budget in {3, 5}
                                else None
                            ),
                        }
                    )
    return rows


def test_frozen_decisions_and_exact_eight_severe_harm_estimands(contract) -> None:
    summaries, promotion = synthetic.summarize_synthetic_metrics(
        _passing_metric_rows(),
        contract=contract,
        root_seed=NON_RESERVED_SEED,
        families=synthetic.FAMILY_NAMES,
        sign_flip_draws=8,
        bootstrap_draws=8,
    )
    applicable = {item["contrast"] for item in summaries if item["severe_harm"]["applicable"]}
    assert applicable == set(contract.plan["promotion_requirements"]["severe_harm"]["contrast_ids"])
    assert "B3:P1_A_QM_correct-P1_A_QM_pair_shuffled:eAUC" not in applicable
    assert "N4:P1_A_QM_correct-P1_A_QM_pair_shuffled:eAUC" not in applicable
    assert promotion["all_requirements_passed"] is True


def test_cache_is_exclusive_pickle_free_and_seed_bound(tmp_path: Path, contract) -> None:
    participant = synthetic.generate_synthetic_participant(
        contract,
        root_seed=NON_RESERVED_SEED,
        family=synthetic.FAMILY_NAMES[0],
        participant_index=0,
    )
    product = _fake_product(contract, participant.signals)
    path = tmp_path / "cache.npz"
    synthetic.write_strict_fbcca_cache_exclusive(
        path, product, contract=contract, participant=participant
    )
    loaded = synthetic.load_strict_fbcca_cache(path, contract=contract, participant=participant)
    assert np.array_equal(loaded.scores, product.scores)
    with pytest.raises(FileExistsError):
        synthetic.write_strict_fbcca_cache_exclusive(
            path, product, contract=contract, participant=participant
        )


def test_governed_writer_rejects_ancestor_symlink_and_has_one_race_winner(
    tmp_path: Path,
) -> None:
    real_parent = tmp_path / "real"
    real_parent.mkdir(mode=0o700)
    alias = tmp_path / "alias"
    alias.symlink_to(real_parent, target_is_directory=True)
    with pytest.raises(ValueError, match="symlinked parent"):
        synthetic.write_json_exclusive(alias / "forbidden.json", {"value": 1})

    target = real_parent / "race.json"
    barrier = threading.Barrier(2)

    def contender(value: int):
        barrier.wait()
        try:
            return synthetic.write_json_exclusive(target, {"value": value})
        except FileExistsError:
            return None

    with ThreadPoolExecutor(max_workers=2) as pool:
        outcomes = list(pool.map(contender, (1, 2)))
    assert sum(outcome == target for outcome in outcomes) == 1
    assert sum(outcome is None for outcome in outcomes) == 1
    assert synthetic.load_json_object(target, name="race winner")["value"] in {1, 2}


def _mock_beacon_pulse(contract: synthetic.SyntheticContract) -> dict:
    beacon = contract.plan["rng"]["independent_lockbox_root_seed"]
    pulse = {
        "uri": "https://beacon.nist.gov/beacon/2.0/chain/7/pulse/11",
        "version": "2.0",
        "cipherSuite": 0,
        "period": 60000,
        "certificateId": "a" * 128,
        "chainIndex": 7,
        "pulseIndex": 11,
        "timeStamp": beacon["required_exact_pulse_timestamp"],
        "localRandomValue": "B" * 128,
        "external": {
            "sourceId": "0" * 128,
            "statusCode": 0,
            "value": "1" * 128,
        },
        "listValues": [
            {
                "uri": f"https://beacon.nist.gov/beacon/2.0/chain/7/pulse/{index}",
                "type": name,
                "value": character * 128,
            }
            for index, (name, character) in enumerate(
                zip(("previous", "hour", "day", "month", "year"), "23456"),
                start=1,
            )
        ],
        "precommitmentValue": "C" * 128,
        "statusCode": 0,
        "signatureValue": "AB" * 512,
        "outputValue": "0" * 128,
    }
    signed = synthetic._serialize_deployed_nist_signed_message(pulse)
    pulse["outputValue"] = hashlib.sha512(
        signed + bytes.fromhex(pulse["signatureValue"])
    ).hexdigest().upper()
    return pulse


def _valid_beacon_receipt(
    contract: synthetic.SyntheticContract,
    identity: synthetic.GitIdentity,
    environment: dict,
    *,
    claim_sha256: str,
) -> tuple[dict, int]:
    beacon = contract.plan["rng"]["independent_lockbox_root_seed"]
    pulse = _mock_beacon_pulse(contract)
    response = json.dumps({"pulse": pulse}, sort_keys=True, separators=(",", ":"))
    root_seed, digest, raw_sha = synthetic._derive_seed_material(
        contract,
        pulse_timestamp=pulse["timeStamp"],
        output_value=pulse["outputValue"],
    )
    receipt = {
        "schema": synthetic.SYNTHETIC_BEACON_RECEIPT_SCHEMA,
        "candidate_id": "metadata-calibration-efficiency-v2",
        "phase": "lockbox-beacon-after-claim",
        "status": "exact_future_pulse_recorded",
        "plan_sha256": contract.plan_sha256,
        "provider": beacon["provider"],
        "endpoint": beacon["exact_endpoint"],
        "target_timestamp_utc": beacon["target_timestamp_utc"],
        "fetched_at_utc": (
            synthetic._target_datetime(contract) + timedelta(seconds=2)
        ).isoformat(timespec="milliseconds").replace("+00:00", "Z"),
        "statement_utf8": beacon["statement_utf8"],
        "statement_sha256": beacon["statement_sha256"],
        "pulse_timestamp": pulse["timeStamp"],
        "pulse_uri": pulse["uri"],
        "pulse_version": pulse["version"],
        "pulse_period_milliseconds": pulse["period"],
        "pulse_cipher_suite": pulse["cipherSuite"],
        "certificate_id": pulse["certificateId"],
        "chain_index": pulse["chainIndex"],
        "pulse_index": pulse["pulseIndex"],
        "local_random_value": pulse["localRandomValue"],
        "precommitment_value": pulse["precommitmentValue"],
        "pulse_status_code": pulse["statusCode"],
        "signature_value": pulse["signatureValue"],
        "output_value": pulse["outputValue"],
        "deployed_json_serialization_revision": beacon[
            "deployed_json_serialization_revision"
        ],
        "signed_message_sha512": hashlib.sha512(
            synthetic._serialize_deployed_nist_signed_message(pulse)
        ).hexdigest(),
        "output_value_recomputed": pulse["outputValue"],
        "output_value_protocol_verified": True,
        "transport_authentication": beacon["transport_authentication"],
        "output_value_raw_sha256": raw_sha,
        "full_response_utf8": response,
        "full_response_sha256": hashlib.sha256(response.encode()).hexdigest(),
        "derived_seed_sha256": digest,
        "root_seed_formula": beacon["root_seed_formula"],
        "generator_commit": identity.commit,
        "generator_tree": identity.tree,
        "generator_source_bundle_sha256": identity.source_bundle_sha256,
        "git_worktree_clean": True,
        "generator_commit_timestamp_utc": identity.commit_timestamp_utc,
        "environment": environment,
        "preparation_receipt_sha256": "4" * 64,
        "development_result_sha256": "5" * 64,
        "test_evidence_sha256": "6" * 64,
        "one_time_claim_sha256": claim_sha256,
    }
    receipt["beacon_receipt_sha256"] = synthetic._canonical_json_sha256(receipt)
    return receipt, root_seed


def test_future_beacon_seed_uses_full_digest_and_fresh_process_guard(
    tmp_path: Path, contract, monkeypatch
) -> None:
    isolated = _retarget_beacon(
        _contract_with_paths(tmp_path, contract), datetime.now(timezone.utc) + timedelta(hours=1)
    )
    environment = synthetic.current_environment_receipt()
    identity = _identity()
    auth_path = synthetic.canonical_execution_path(isolated, "lockbox_authorization")
    auth = {
        "authorization_created_at_utc": (
            synthetic._target_datetime(isolated) - timedelta(minutes=10)
        ).isoformat(timespec="milliseconds").replace("+00:00", "Z"),
        "authorization_receipt_sha256": "8" * 64,
    }
    synthetic.write_json_exclusive(auth_path, auth)
    claim_path = synthetic.canonical_execution_path(isolated, "seed_global_lockbox_claim")
    claim = {
        "schema": synthetic.SYNTHETIC_LOCKBOX_CLAIM_SCHEMA,
        "candidate_id": "metadata-calibration-efficiency-v2",
        "phase": "lockbox-pre-network-claim",
        "status": "one_time_global_claimed_before_beacon_fetch_seed_or_rng",
        "plan_sha256": isolated.plan_sha256,
        "claim_created_at_utc": (
            synthetic._target_datetime(isolated) + timedelta(seconds=1)
        ).isoformat(timespec="milliseconds").replace("+00:00", "Z"),
        "future_beacon_target_utc": isolated.plan["rng"][
            "independent_lockbox_root_seed"
        ]["target_timestamp_utc"],
        "future_beacon_statement_sha256": isolated.plan["rng"][
            "independent_lockbox_root_seed"
        ]["statement_sha256"],
        "network_access_before_claim": False,
        "authorization_path": str(auth_path),
        "authorization_file_sha256": synthetic._sha256_file(auth_path),
        "authorization_receipt_sha256": auth["authorization_receipt_sha256"],
        "authorization_created_at_utc": auth["authorization_created_at_utc"],
        "generator_commit": identity.commit,
        "generator_tree": identity.tree,
        "generator_source_bundle_sha256": identity.source_bundle_sha256,
        "git_worktree_clean": True,
        "generator_commit_timestamp_utc": identity.commit_timestamp_utc,
        "environment_sha256": environment["environment_sha256"],
        "beacon_receipt_path": str(
            synthetic.canonical_execution_path(isolated, "beacon_receipt")
        ),
        "output_path": str(synthetic.canonical_execution_path(isolated, "lockbox_result")),
        "terminal_receipt_path": str(
            synthetic.canonical_execution_path(isolated, "lockbox_terminal_receipt")
        ),
    }
    claim["claim_sha256"] = synthetic._canonical_json_sha256(claim)
    synthetic.write_json_exclusive(claim_path, claim)
    monkeypatch.setattr(
        synthetic,
        "_validate_consumed_authorization_binding",
        lambda *args, **kwargs: auth["authorization_receipt_sha256"],
    )
    receipt, root_seed = _valid_beacon_receipt(
        isolated, identity, environment, claim_sha256=claim["claim_sha256"]
    )
    path = synthetic.canonical_execution_path(isolated, "beacon_receipt")
    synthetic.write_json_exclusive(path, receipt)
    beacon = isolated.plan["rng"]["independent_lockbox_root_seed"]
    expected = int.from_bytes(
        hashlib.sha256(
            beacon["statement_utf8"].encode() + b"\n" + bytes.fromhex(receipt["output_value"])
        ).digest(),
        "big",
        signed=False,
    )
    assert root_seed == expected
    synthetic._DYNAMIC_RESERVED_SEEDS.clear()
    synthetic._VALIDATED_BEACON_SEED_CACHE.clear()
    assert synthetic._derive_lockbox_seed_from_beacon_receipt(isolated, path) == expected
    key = {
        "root_seed": expected,
        "family": synthetic.FAMILY_NAMES[0],
        "participant_index": 0,
        "block": 0,
        "class_index": 0,
        "component": "spatial_signature",
    }
    with pytest.raises(ValueError, match="internal governed"):
        synthetic.keyed_rng(isolated, **key)
    synthetic._DYNAMIC_RESERVED_SEEDS.clear()
    synthetic._VALIDATED_BEACON_SEED_CACHE.clear()
    with pytest.raises(ValueError, match="internal governed"):
        synthetic.generate_synthetic_participant(
            isolated,
            root_seed=expected,
            family=synthetic.FAMILY_NAMES[0],
            participant_index=0,
        )
    synthetic._DYNAMIC_RESERVED_SEEDS.discard(expected)
    synthetic._VALIDATED_BEACON_SEED_CACHE.clear()


def test_standalone_beacon_fetch_is_always_forbidden(tmp_path: Path, contract) -> None:
    isolated = _contract_with_paths(tmp_path, contract)
    with pytest.raises(RuntimeError, match="forbids standalone"):
        synthetic.fetch_nist_beacon_receipt(
            isolated,
            output_path=synthetic.canonical_execution_path(isolated, "beacon_receipt"),
        )
    assert set(inspect.signature(synthetic.fetch_nist_beacon_receipt).parameters) == {
        "contract",
        "output_path",
    }


def test_internal_beacon_fetch_requires_durable_claim_before_network(
    tmp_path: Path, contract, monkeypatch
) -> None:
    target_time = datetime.now(timezone.utc) + timedelta(hours=1)
    isolated = _retarget_beacon(_contract_with_paths(tmp_path, contract), target_time)
    identity = _identity()
    environment = synthetic.current_environment_receipt()
    references = {
        name: {
            "path": str(tmp_path / f"{name}.json"),
            "file_sha256": character * 64,
            "receipt_sha256": character * 64,
        }
        for name, character in zip(
            ("preparation_evidence", "development_evidence", "test_evidence"),
            "456",
        )
    }
    monkeypatch.setattr(synthetic, "current_git_identity", lambda: identity)
    monkeypatch.setattr(synthetic, "current_environment_receipt", lambda: environment)
    monkeypatch.setattr(
        synthetic, "_prelockbox_authorization_evidence", lambda *args, **kwargs: references
    )
    _FrozenDateTime.observed_now = target_time - timedelta(minutes=10)
    monkeypatch.setattr(synthetic, "datetime", _FrozenDateTime)
    authorization = synthetic.build_lockbox_authorization_template(
        isolated,
        output_path=synthetic.canonical_execution_path(isolated, "lockbox_result"),
    )
    authorization.update(
        {
            "status": "authorized_for_one_time_execution",
            "authorized": True,
            "authorized_by": "unit test",
            "authorization_basis": "unit test missing-claim boundary",
            "one_time_nonce_sha256": "a" * 64,
        }
    )
    authorization = synthetic.seal_authorization_record(authorization)
    synthetic.write_json_exclusive(
        synthetic.canonical_execution_path(isolated, "lockbox_authorization"),
        authorization,
    )
    built = False
    contacted = False

    class ForbiddenOpener:
        def open(self, *args, **kwargs):
            nonlocal contacted
            contacted = True
            raise AssertionError("network must not be reached without a durable claim")

    def forbidden_opener_factory(*args, **kwargs):
        nonlocal built
        built = True
        return ForbiddenOpener()

    _FrozenDateTime.observed_now = target_time + timedelta(seconds=1)
    monkeypatch.setattr(synthetic.urllib.request, "build_opener", forbidden_opener_factory)
    with pytest.raises(ValueError, match="existing nonsymlink"):
        synthetic._fetch_nist_beacon_after_global_claim(
            isolated,
            identity=identity,
            environment=environment,
            preparation_receipt_sha256="4" * 64,
            development_result_sha256="5" * 64,
            test_evidence_sha256="6" * 64,
            one_time_claim_sha256="7" * 64,
        )
    assert built is False
    assert contacted is False
    assert "_opener" not in inspect.signature(
        synthetic._fetch_nist_beacon_after_global_claim
    ).parameters


def test_nist_transport_rejects_redirect_before_location_request() -> None:
    handler = synthetic._RejectHTTPRedirect()
    request = synthetic.urllib.request.Request(
        "https://beacon.nist.gov/beacon/2.0/pulse/time/1"
    )
    with pytest.raises(synthetic.urllib.error.HTTPError, match="Redirect forbidden"):
        handler.redirect_request(
            request,
            None,
            302,
            "Found",
            {"Location": "https://example.invalid/forbidden"},
            "https://example.invalid/forbidden",
        )


def test_beacon_pulse_core_fields_are_fail_closed(contract) -> None:
    valid = _mock_beacon_pulse(contract)
    assert synthetic._extract_beacon_pulse({"pulse": valid}) == valid
    signed_hash, recomputed = synthetic._verify_deployed_nist_output_value(valid)
    assert len(signed_hash) == 128 and recomputed == valid["outputValue"]
    for field, wrong in (
        ("version", "1.0"),
        ("period", 1),
        ("statusCode", 1),
        ("uri", "https://example.invalid"),
        ("certificateId", "A" * 128),
        ("signatureValue", "xyz"),
    ):
        with pytest.raises(ValueError):
            synthetic._extract_beacon_pulse({"pulse": {**valid, field: wrong}})
    tampered = {**valid, "localRandomValue": "E" * 128}
    with pytest.raises(ValueError, match="outputValue"):
        synthetic._verify_deployed_nist_output_value(tampered)


def test_deployed_nist_serializer_matches_frozen_historical_oracle() -> None:
    fixture_path = Path("tests/fixtures/nist_beacon_v2_20260905T180000Z.json")
    oracle = json.loads(fixture_path.read_text(encoding="utf-8"))
    assert oracle["source_endpoint"] == (
        "https://beacon.nist.gov/beacon/2.0/pulse/time/1788631200000"
    )
    assert synthetic._canonical_json_sha256(oracle["response"]) == oracle[
        "parsed_response_canonical_sha256"
    ]
    pulse = synthetic._extract_beacon_pulse(oracle["response"])
    signed_message = synthetic._serialize_deployed_nist_signed_message(pulse)
    assert len(signed_message) == oracle["signed_message_length_bytes"] == 807
    assert hashlib.sha512(signed_message).hexdigest() == oracle["signed_message_sha512"]
    _, recomputed = synthetic._verify_deployed_nist_output_value(pulse)
    assert recomputed == pulse["outputValue"]
    assert recomputed == (
        "BADC50F6F9E38477950DA01DBF00C0F637FEB4A2F5A1BC9232B0CCF084DE2612"
        "5539608A6138566AB5B904B816229D3D9B322B7F0B894C35EDAD1C4A057BEB03"
    )


def test_historical_nist_oracle_binds_each_signed_field_and_signature() -> None:
    oracle = json.loads(
        Path("tests/fixtures/nist_beacon_v2_20260905T180000Z.json").read_text(
            encoding="utf-8"
        )
    )
    pulse = oracle["response"]["pulse"]

    def flip_hex(value: str) -> str:
        return ("0" if value[0] != "0" else "1") + value[1:]

    mutations = []
    for field, value in (
        ("uri", pulse["uri"] + "/changed"),
        ("version", "2.1"),
        ("cipherSuite", 1),
        ("period", 60001),
        ("certificateId", flip_hex(pulse["certificateId"])),
        ("chainIndex", pulse["chainIndex"] + 1),
        ("pulseIndex", pulse["pulseIndex"] + 1),
        ("timeStamp", "2026-09-05T18:00:01.000Z"),
        ("localRandomValue", flip_hex(pulse["localRandomValue"])),
        ("precommitmentValue", flip_hex(pulse["precommitmentValue"])),
        ("statusCode", 1),
        ("signatureValue", flip_hex(pulse["signatureValue"])),
    ):
        changed = deepcopy(pulse)
        changed[field] = value
        mutations.append(changed)
    for field in ("sourceId", "statusCode", "value"):
        changed = deepcopy(pulse)
        changed["external"][field] = (
            1 if field == "statusCode" else flip_hex(changed["external"][field])
        )
        mutations.append(changed)
    for index in range(5):
        changed = deepcopy(pulse)
        changed["listValues"][index]["value"] = flip_hex(
            changed["listValues"][index]["value"]
        )
        mutations.append(changed)
    assert len(mutations) == 20
    for changed in mutations:
        with pytest.raises(ValueError, match="outputValue"):
            synthetic._verify_deployed_nist_output_value(changed)


def test_preparation_binds_pre_target_creation_and_head_commit(
    tmp_path: Path, contract, monkeypatch
) -> None:
    observed_now = datetime.now(timezone.utc)
    isolated = _retarget_beacon(
        _contract_with_paths(tmp_path, contract), observed_now + timedelta(hours=1)
    )
    environment = synthetic.current_environment_receipt()
    monkeypatch.setattr(synthetic, "current_git_identity", lambda: _identity())
    monkeypatch.setattr(synthetic, "current_environment_receipt", lambda: environment)
    _FrozenDateTime.observed_now = observed_now
    monkeypatch.setattr(synthetic, "datetime", _FrozenDateTime)
    output = synthetic.canonical_execution_path(isolated, "preparation_receipt")
    receipt = synthetic.build_preparation_receipt(isolated, output_path=output)
    assert receipt["created_at_utc"] == observed_now.isoformat(
        timespec="milliseconds"
    ).replace("+00:00", "Z")
    assert receipt["generator_commit_timestamp_utc"] == "2026-09-05T18:00:00Z"
    synthetic.write_json_exclusive(output, receipt)
    assert (output.stat().st_mode & 0o777) == 0o400
    validated, _ = synthetic._validate_preparation_receipt(
        isolated, identity=_identity(), environment=environment
    )
    assert validated == receipt

    late = _retarget_beacon(
        _contract_with_paths(tmp_path / "late", contract), observed_now + timedelta(hours=1)
    )
    _FrozenDateTime.observed_now = synthetic._target_datetime(late)
    with pytest.raises(ValueError, match="predate"):
        synthetic.build_preparation_receipt(
            late,
            output_path=synthetic.canonical_execution_path(late, "preparation_receipt"),
        )
    assert set(inspect.signature(synthetic.build_preparation_receipt).parameters) == {
        "contract",
        "output_path",
    }


def test_consumed_authorization_revalidates_exact_sealed_binding(
    tmp_path: Path, contract, monkeypatch
) -> None:
    observed_now = datetime.now(timezone.utc)
    isolated = _retarget_beacon(
        _contract_with_paths(tmp_path, contract), observed_now + timedelta(hours=1)
    )
    identity = _identity()
    environment = synthetic.current_environment_receipt()
    references = {
        name: {
            "path": str(tmp_path / f"{name}.json"),
            "file_sha256": character * 64,
            "receipt_sha256": character.upper().lower() * 64,
        }
        for name, character in zip(
            ("preparation_evidence", "development_evidence", "test_evidence"),
            "456",
        )
    }
    monkeypatch.setattr(synthetic, "current_git_identity", lambda: identity)
    monkeypatch.setattr(synthetic, "current_environment_receipt", lambda: environment)
    monkeypatch.setattr(
        synthetic, "_prelockbox_authorization_evidence", lambda *args, **kwargs: references
    )
    _FrozenDateTime.observed_now = observed_now
    monkeypatch.setattr(synthetic, "datetime", _FrozenDateTime)
    record = synthetic.build_lockbox_authorization_template(
        isolated,
        output_path=synthetic.canonical_execution_path(isolated, "lockbox_result"),
    )
    record.update(
        {
            "status": "authorized_for_one_time_execution",
            "authorized": True,
            "authorized_by": "unit test",
            "authorization_basis": "unit test only",
            "one_time_nonce_sha256": "a" * 64,
        }
    )
    record = synthetic.seal_authorization_record(record)
    assert synthetic._validate_consumed_authorization_binding(
        record,
        contract=isolated,
        identity=identity,
        environment=environment,
    ) == record["authorization_receipt_sha256"]

    forged = {**record, "generator_tree": "f" * 40}
    forged = synthetic.seal_authorization_record(forged)
    with pytest.raises(ValueError, match="source, scope, or time"):
        synthetic._validate_consumed_authorization_binding(
            forged,
            contract=isolated,
            identity=identity,
            environment=environment,
        )


def test_minimal_and_forged_development_results_are_rejected(contract) -> None:
    environment = synthetic.current_environment_receipt()
    with pytest.raises(ValueError, match="exact frozen fields"):
        synthetic.validate_development_result(
            {},
            contract=contract,
            identity=_identity(),
            environment=environment,
            preparation_receipt_sha256="4" * 64,
        )
    rows = _passing_metric_rows()
    assert len(rows) == 5888
    with pytest.raises(ValueError, match="exact frozen fields"):
        synthetic._validate_complete_metric_grid(rows)


def test_full_suite_evidence_uses_exact_current_python_and_captures_transcripts(
    tmp_path: Path, contract, monkeypatch
) -> None:
    isolated = _contract_with_paths(tmp_path, contract)
    identity = _identity()
    environment = synthetic.current_environment_receipt()
    prep = {"completion_receipt_sha256": "4" * 64}
    dev = {"result_sha256": "5" * 64}
    monkeypatch.setattr(synthetic, "current_git_identity", lambda: identity)
    monkeypatch.setattr(synthetic, "current_environment_receipt", lambda: environment)
    monkeypatch.setattr(
        synthetic,
        "_validate_preparation_receipt",
        lambda *args, **kwargs: (
            prep,
            synthetic.canonical_execution_path(isolated, "preparation_receipt"),
        ),
    )
    monkeypatch.setattr(synthetic, "_development_evidence", lambda *args, **kwargs: dev)
    observed = {}

    def runner(argv, **kwargs):
        observed.update({"argv": argv, **kwargs})
        return SimpleNamespace(returncode=0, stdout=b"all passed\n", stderr=b"")

    output = synthetic.canonical_execution_path(isolated, "full_suite_test_evidence")
    monkeypatch.setattr(synthetic.subprocess, "run", runner)
    synthetic.run_full_suite_test_evidence(isolated, output_path=output)
    receipt = synthetic.load_json_object(output, name="test receipt")
    assert observed["argv"] == [os.path.abspath(sys.executable), "-m", "pytest", "-q"]
    assert observed["cwd"] == synthetic._REPOSITORY
    assert receipt["scope"] == "complete_repository_test_suite"
    assert receipt["stdout_sha256"] == hashlib.sha256(b"all passed\n").hexdigest()
    stdout_path, stderr_path = synthetic._test_transcript_paths(output)
    assert stdout_path.read_bytes() == b"all passed\n"
    assert stderr_path.read_bytes() == b""
    assert (output.stat().st_mode & 0o777) == 0o400
    assert set(inspect.signature(synthetic.run_full_suite_test_evidence).parameters) == {
        "contract",
        "output_path",
    }


def test_forged_full_suite_receipt_is_rejected(tmp_path: Path, contract) -> None:
    isolated = _contract_with_paths(tmp_path, contract)
    identity = _identity()
    environment = synthetic.current_environment_receipt()
    output = synthetic.canonical_execution_path(isolated, "full_suite_test_evidence")
    stdout_path, stderr_path = synthetic._test_transcript_paths(output)
    synthetic._write_bytes_exclusive(stdout_path, b"ok\n")
    synthetic._write_bytes_exclusive(stderr_path, b"")
    receipt = {
        "schema": synthetic.SYNTHETIC_TEST_EVIDENCE_SCHEMA,
        "candidate_id": "metadata-calibration-efficiency-v2",
        "phase": "test-evidence",
        "status": "passed",
        "plan_sha256": isolated.plan_sha256,
        "argv": [os.path.abspath(sys.executable), "-m", "pytest", "-q"],
        "working_directory": str(synthetic._REPOSITORY),
        "scope": "complete_repository_test_suite",
        "exit_code": 0,
        "stdout_path": str(stdout_path),
        "stdout_sha256": "0" * 64,
        "stderr_path": str(stderr_path),
        "stderr_sha256": hashlib.sha256(b"").hexdigest(),
        "generator_commit": identity.commit,
        "generator_tree": identity.tree,
        "generator_source_bundle_sha256": identity.source_bundle_sha256,
        "git_worktree_clean": True,
        "generator_commit_timestamp_utc": identity.commit_timestamp_utc,
        "environment": environment,
        "preparation_receipt_sha256": "4" * 64,
        "development_result_sha256": "5" * 64,
    }
    receipt["test_evidence_sha256"] = synthetic._canonical_json_sha256(receipt)
    synthetic.write_json_exclusive(output, receipt)
    with pytest.raises(ValueError, match="forged"):
        synthetic._validate_test_evidence(
            isolated,
            identity=identity,
            environment=environment,
            preparation_receipt_sha256="4" * 64,
            development_result_sha256="5" * 64,
        )


@pytest.mark.parametrize(
    ("postclaim_error", "expected_status"),
    (
        (MemoryError("unit-test simulated OOM"), "consumed_inconclusive_infrastructure_error"),
        (synthetic.HardInvariantFailure("unit-test invariant"), "terminal_invariant_fail"),
    ),
)
def test_global_claim_consumes_seed_and_alternate_output_cannot_retry(
    tmp_path: Path, contract, monkeypatch, postclaim_error, expected_status
) -> None:
    target_time = datetime.now(timezone.utc) + timedelta(hours=1)
    isolated = _retarget_beacon(_contract_with_paths(tmp_path, contract), target_time)
    identity = _identity()
    environment = synthetic.current_environment_receipt()
    auth_path = synthetic.canonical_execution_path(isolated, "lockbox_authorization")
    output = synthetic.canonical_execution_path(isolated, "lockbox_result")
    authorization = {
        "authorization_created_at_utc": (
            target_time - timedelta(minutes=10)
        ).isoformat(timespec="milliseconds").replace("+00:00", "Z"),
        "authorization_receipt_sha256": "8" * 64,
        "preparation_evidence": {"receipt_sha256": "4" * 64},
        "development_evidence": {"receipt_sha256": "5" * 64},
        "test_evidence": {"receipt_sha256": "6" * 64},
        "future_beacon_commitment": synthetic._future_beacon_commitment(isolated),
    }
    synthetic.write_json_exclusive(auth_path, authorization)
    monkeypatch.setattr(synthetic, "current_git_identity", lambda: identity)
    monkeypatch.setattr(synthetic, "current_environment_receipt", lambda: environment)
    monkeypatch.setattr(
        synthetic, "validate_lockbox_authorization", lambda *args, **kwargs: "8" * 64
    )
    monkeypatch.setattr(
        synthetic,
        "_validate_consumed_authorization_binding",
        lambda *args, **kwargs: "8" * 64,
    )
    _FrozenDateTime.observed_now = target_time + timedelta(seconds=1)
    monkeypatch.setattr(synthetic, "datetime", _FrozenDateTime)

    def fetch_after_claim(*args, **kwargs):
        claim_path = synthetic.canonical_execution_path(
            isolated, "seed_global_lockbox_claim"
        )
        assert claim_path.exists()
        return (
            synthetic.canonical_execution_path(isolated, "beacon_receipt"),
            {"beacon_receipt_sha256": "7" * 64},
            NON_RESERVED_SEED + 99,
        )

    monkeypatch.setattr(
        synthetic, "_fetch_nist_beacon_after_global_claim", fetch_after_claim
    )
    monkeypatch.setattr(
        synthetic,
        "_validate_beacon_evidence",
        lambda *args, **kwargs: (
            {"beacon_receipt_sha256": "7" * 64, "derived_seed_sha256": "9" * 64},
            NON_RESERVED_SEED + 99,
        ),
    )

    def fail_after_claim(*args, **kwargs):
        raise postclaim_error

    monkeypatch.setattr(synthetic, "_execute_synthetic", fail_after_claim)
    written = synthetic.run_lockbox_to_path(
        isolated, authorization_path=auth_path, output_path=output
    )
    assert written == synthetic.canonical_execution_path(
        isolated, "lockbox_terminal_receipt"
    )
    terminal = synthetic.load_json_object(written, name="terminal receipt")
    assert terminal["status"] == expected_status
    assert synthetic.validate_lockbox_terminal_artifact(
        isolated, artifact_path=written
    )["status"] == expected_status
    claim = synthetic.canonical_execution_path(isolated, "seed_global_lockbox_claim")
    assert claim.exists() and (claim.stat().st_mode & 0o777) == 0o400
    with pytest.raises(FileExistsError):
        synthetic.run_lockbox_to_path(isolated, authorization_path=auth_path, output_path=output)
    alternate = tmp_path / "alternate.json"
    with pytest.raises(ValueError, match="exact canonical path"):
        synthetic.run_lockbox_to_path(isolated, authorization_path=auth_path, output_path=alternate)


@pytest.mark.parametrize(
    ("failure_point", "expected_status"),
    (
        ("claim", "consumed_inconclusive_claim_write_error"),
        ("result", "consumed_inconclusive_infrastructure_error"),
    ),
)
def test_partial_claim_or_result_write_still_gets_separate_terminal_receipt(
    tmp_path: Path, contract, monkeypatch, failure_point, expected_status
) -> None:
    target_time = datetime.now(timezone.utc) + timedelta(hours=1)
    isolated = _retarget_beacon(_contract_with_paths(tmp_path, contract), target_time)
    identity = _identity()
    environment = synthetic.current_environment_receipt()
    auth_path = synthetic.canonical_execution_path(isolated, "lockbox_authorization")
    result_path = synthetic.canonical_execution_path(isolated, "lockbox_result")
    claim_path = synthetic.canonical_execution_path(isolated, "seed_global_lockbox_claim")
    terminal_path = synthetic.canonical_execution_path(
        isolated, "lockbox_terminal_receipt"
    )
    authorization = {
        "authorization_created_at_utc": (
            target_time - timedelta(minutes=10)
        ).isoformat(timespec="milliseconds").replace("+00:00", "Z"),
        "authorization_receipt_sha256": "8" * 64,
        "preparation_evidence": {"receipt_sha256": "4" * 64},
        "development_evidence": {"receipt_sha256": "5" * 64},
        "test_evidence": {"receipt_sha256": "6" * 64},
        "future_beacon_commitment": synthetic._future_beacon_commitment(isolated),
    }
    synthetic.write_json_exclusive(auth_path, authorization)
    monkeypatch.setattr(synthetic, "current_git_identity", lambda: identity)
    monkeypatch.setattr(synthetic, "current_environment_receipt", lambda: environment)
    monkeypatch.setattr(
        synthetic, "validate_lockbox_authorization", lambda *args, **kwargs: "8" * 64
    )
    monkeypatch.setattr(
        synthetic,
        "_validate_consumed_authorization_binding",
        lambda *args, **kwargs: "8" * 64,
    )
    _FrozenDateTime.observed_now = target_time + timedelta(seconds=1)
    monkeypatch.setattr(synthetic, "datetime", _FrozenDateTime)

    def fetched(*args, **kwargs):
        assert claim_path.exists()
        return result_path.with_name("beacon.json"), {"beacon_receipt_sha256": "7" * 64}, 91

    monkeypatch.setattr(synthetic, "_fetch_nist_beacon_after_global_claim", fetched)
    monkeypatch.setattr(
        synthetic,
        "_validate_beacon_evidence",
        lambda *args, **kwargs: ({"beacon_receipt_sha256": "7" * 64}, 91),
    )
    monkeypatch.setattr(
        synthetic,
        "_execute_synthetic",
        lambda *args, **kwargs: {"status": "terminal_pass", "result_sha256": "0" * 64},
    )
    original_write = synthetic._write_bytes_exclusive
    failed_path = claim_path if failure_point == "claim" else result_path

    def partial_then_fail(path, payload):
        if Path(path) == failed_path:
            original_write(path, b"{")
            raise OSError(f"simulated partial {failure_point} write")
        return original_write(path, payload)

    monkeypatch.setattr(synthetic, "_write_bytes_exclusive", partial_then_fail)
    written = synthetic.run_lockbox_to_path(
        isolated, authorization_path=auth_path, output_path=result_path
    )
    assert written == terminal_path
    terminal = synthetic.load_json_object(terminal_path, name="terminal receipt")
    assert terminal["status"] == expected_status
    assert synthetic.validate_lockbox_terminal_artifact(
        isolated, artifact_path=terminal_path
    )["status"] == expected_status
    assert failed_path.exists()
    with pytest.raises(FileExistsError):
        synthetic.run_lockbox_to_path(
            isolated, authorization_path=auth_path, output_path=result_path
        )


def test_terminal_receipt_has_strict_precedence_over_scientific_result(
    tmp_path: Path, contract
) -> None:
    isolated = _contract_with_paths(tmp_path, contract)
    result_path = synthetic.canonical_execution_path(isolated, "lockbox_result")
    terminal_path = synthetic.canonical_execution_path(
        isolated, "lockbox_terminal_receipt"
    )
    synthetic.write_json_exclusive(result_path, {"otherwise": "valid-result-fixture"})
    synthetic.write_json_exclusive(terminal_path, {"status": "consumed"})
    with pytest.raises(ValueError, match="strict precedence"):
        synthetic._validate_lockbox_scientific_result(isolated, result_path)


def test_cli_exposes_every_canonical_governance_phase() -> None:
    script = Path("scripts/run_metadata_calibration_v2_synthetic.py").resolve()
    sys.path.insert(0, str(script.parent))
    try:
        spec = importlib.util.spec_from_file_location("synthetic_cli_v11", script)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    finally:
        sys.path.remove(str(script.parent))
    parser = module._parser()
    for phase in (
        "prepare",
        "development",
        "test-evidence",
        "auth-template",
        "authorize",
        "lockbox",
    ):
        arguments = [phase]
        if phase == "authorize":
            arguments += [
                "--authorized-by",
                "unit test",
                "--authorization-basis",
                "unit test only",
                "--one-time-nonce-sha256",
                "a" * 64,
            ]
        assert parser.parse_args(arguments).phase == phase
