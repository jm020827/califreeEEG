from __future__ import annotations

import inspect

import numpy as np
import pytest

from cfeg.models import metadata_calibration_v3 as v3

_HASH_A = "a" * 64


def _config(index: int = 1) -> v3.V3OperatorConfig:
    return v3.V3OperatorConfig(v3.canonical_operator_grid()[index - 1])


def _score_fixture(blocks: int = 3) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    query = np.asarray(
        [[0.90, 0.20, 0.10], [0.15, 0.80, 0.25], [0.40, 0.30, 0.60]],
        dtype=np.float64,
    )
    support = []
    labels = []
    for block in range(blocks):
        support.append(
            np.asarray(
                [
                    [0.55 + 0.02 * block, 0.50, 0.10],
                    [0.10, 0.56 + 0.02 * block, 0.51],
                    [0.52, 0.10, 0.57 + 0.02 * block],
                ]
            )
        )
        labels.append([0, 1, 2])
    return query, np.asarray(support), np.asarray(labels)


def _support_product(blocks: int = 3, *, query_key: str = "query") -> v3.V3SupportProduct:
    query, support, labels = _score_fixture(blocks)
    return v3.blockwise_p3_support(
        query,
        support,
        labels,
        query_key=query_key,
        ordered_support_block_keys=tuple(f"block{index:02d}" for index in range(1, blocks + 1)),
        support_eeg_label_manifest_sha256=_HASH_A,
        config=_config(),
    )


def _reference() -> v3.ContextReference:
    packets = []
    for interface, base in (("neutral", 2.0), ("wet", 3.0), ("dry", 5.0)):
        for packet_index in range(256):
            state = -0.5 if packet_index < 128 else 0.5
            logs = base + state + 0.01 * (np.arange(8) - 3.5)
            packets.append(
                {
                    "interface": interface,
                    "impedance_kohm_by_channel": np.expm1(logs).tolist(),
                    "channel_availability": [True] * 8,
                }
            )
    return v3.fit_context_reference(packets, domain="unit-test-domain")


def _raw_packet(key: str, *, z: float, reference: v3.ContextReference) -> dict[str, object]:
    table = reference.table_by_interface["neutral"]
    values = [
        float(np.expm1(center + z * scale))
        for center, scale in zip(table.centers, table.scales, strict=True)
    ]
    return {
        "packet_key": key,
        "interface_lookup_key": "neutral",
        "impedance_kohm_by_channel": values,
        "channel_availability": [True] * 8,
    }


def _preflight(
    reference: v3.ContextReference,
    *,
    query_z: float,
    support_z: tuple[float, ...],
    assignment: tuple[int, ...] | None = None,
) -> v3.ContextPreflightCapability:
    slots = tuple(f"block{index:02d}" for index in range(1, len(support_z) + 1))
    packets = tuple(
        _raw_packet(f"packet{index:02d}", z=value, reference=reference)
        for index, value in enumerate(support_z, start=1)
    )
    if assignment is not None:
        packets = tuple(packets[index] for index in assignment)
    query = _raw_packet("query", z=query_z, reference=reference)
    pairing = v3.context_pairing_sha256(
        query_key="query",
        ordered_support_block_keys=slots,
        query_packet_key="query",
        ordered_support_packet_keys=tuple(packet["packet_key"] for packet in packets),
    )
    return v3.preflight_context(
        reference=reference,
        query_key="query",
        ordered_support_block_keys=slots,
        query_packet=query,
        support_packets=packets,
        pairing_sha256=pairing,
    )


def test_exact_grid_digests_are_recomputed_not_trusted() -> None:
    cells = v3.canonical_operator_grid()
    assert [cell.index for cell in cells] == list(range(1, 10))
    assert len({cell.operator_instance_sha256 for cell in cells}) == 9
    for cell in cells:
        assert cell.operator_instance_sha256 == v3.operator_instance_sha256(
            grid_cell_id=cell.grid_cell_id,
            nu_token=cell.nu_token,
            lambda_token=cell.lambda_token,
        )
    with pytest.raises(ValueError, match="digest"):
        v3.V3GridCell(1, cells[0].grid_cell_id, "1", "0p10", 1.0, 0.1, _HASH_A)


def test_score_normalization_is_overflow_safe_row_local_and_argmax_preserving() -> None:
    scores = np.asarray([[0.0, 1.0e308, -1.0e308], [0.5, 0.5, 0.5]])
    normalized, probability = v3.normalize_fbcca_scores(scores)
    assert np.isfinite(normalized).all()
    assert np.isfinite(probability).all()
    assert np.array_equal(np.argmax(probability, axis=1), np.argmax(scores, axis=1))
    assert np.array_equal(probability[1], np.full(3, 1.0 / 3.0))
    for index in range(2):
        singleton = v3.normalize_fbcca_scores(scores[index : index + 1])[1]
        assert np.array_equal(singleton, probability[index : index + 1])
    assert not probability.flags.writeable


def test_p3_reliability_is_relative_M_free_and_shared() -> None:
    query, support, labels = _score_fixture(3)
    product = v3.blockwise_p3_support(
        query,
        support,
        labels,
        query_key="query",
        ordered_support_block_keys=("block01", "block02", "block03"),
        support_eeg_label_manifest_sha256=_HASH_A,
        config=_config(),
    )
    expected = []
    for block in range(3):
        probability = v3.normalize_fbcca_scores(support[block])[1]
        expected.append(float(np.mean(probability[np.arange(3), labels[block]])))
    assert np.array_equal(product.block_reliabilities, np.asarray(expected))
    assert np.array_equal(
        product.normalized_block_weights,
        product.block_reliabilities / product.block_reliabilities.sum(),
    )
    assert np.array_equal(
        product.support_probabilities,
        np.sum(
            product.normalized_block_weights[:, None, None]
            * product.per_block_support_probabilities,
            axis=0,
        ),
    )
    assert tuple(product.block_reliabilities) == (
        product.reliability_capability.block_reliabilities
    )
    assert not product.support_probabilities.flags.writeable

    wrong_support = -support
    wrong = v3.blockwise_p3_support(
        query,
        wrong_support,
        labels,
        query_key="query",
        ordered_support_block_keys=("block01", "block02", "block03"),
        support_eeg_label_manifest_sha256=_HASH_A,
        config=_config(),
    )
    assert np.all(wrong.block_reliabilities > 0.0)
    assert np.isclose(wrong.normalized_block_weights.sum(), 1.0)


def test_k1_weight_is_one_and_uniform_sensitivity_changes_corrupt_block_mix() -> None:
    one = _support_product(1)
    assert np.array_equal(one.normalized_block_weights, np.asarray([1.0]))
    assert np.array_equal(one.support_probabilities, one.per_block_support_probabilities[0])

    query, support, labels = _score_fixture(3)
    support[2] = -10.0 * support[2]
    product = v3.blockwise_p3_support(
        query,
        support,
        labels,
        query_key="query",
        ordered_support_block_keys=("block01", "block02", "block03"),
        support_eeg_label_manifest_sha256=_HASH_A,
        config=_config(),
    )
    uniform = v3.uniform_weight_support_probability(product)
    assert not np.array_equal(uniform, product.support_probabilities)


def test_context_reference_uses_pairwise_interface_then_pooled_fallback() -> None:
    reference = _reference()
    payload = v3.context_reference_payload(reference)
    assert v3.validate_context_reference_payload(payload) == reference
    for table in (*reference.interface_tables, reference.pooled_table):
        assert all(count >= 128 for count in table.observed_counts)
        assert all(scale is not None and scale >= 0.05 for scale in table.scales)

    sparse = [
        {
            "interface": "only",
            "impedance_kohm_by_channel": [1.0] * 8,
            "channel_availability": [True] * 8,
        }
        for _ in range(127)
    ]
    sparse_reference = v3.fit_context_reference(sparse, domain="unit-sparse")
    assert sparse_reference.interface_tables[0].centers == (None,) * 8
    assert sparse_reference.pooled_table.centers == (None,) * 8


def test_context_publication_capability_is_nominal_exact_and_hash_bound() -> None:
    payload = v3.context_reference_payload(_reference())
    proof = v3.validate_context_reference_for_publication(payload)
    assert type(proof) is v3.ValidatedContextReference
    assert proof.payload_sha256 == payload["payload_sha256"]
    assert proof.interface_lookup_keys == ("neutral", "wet", "dry")
    assert (
        v3.require_validated_context_reference(
            proof,
            expected_payload_sha256=payload["payload_sha256"],
        )
        is proof
    )

    with pytest.raises(TypeError, match="issued only"):
        v3.ValidatedContextReference()
    with pytest.raises(TypeError, match="exact ValidatedContextReference"):
        v3.require_validated_context_reference({"payload_sha256": payload["payload_sha256"]})
    with pytest.raises(TypeError, match="not issued"):
        v3.require_validated_context_reference(
            object.__new__(v3.ValidatedContextReference)
        )
    with pytest.raises(ValueError, match="different payload"):
        v3.require_validated_context_reference(
            proof,
            expected_payload_sha256="b" * 64,
        )

    class ForgedContextProof(v3.ValidatedContextReference):
        pass

    with pytest.raises(TypeError, match="exact ValidatedContextReference"):
        v3.require_validated_context_reference(object.__new__(ForgedContextProof))

    tampered = v3.validate_context_reference_for_publication(payload)
    object.__setattr__(tampered, "domain", "tampered")
    with pytest.raises(ValueError, match="semantic binding"):
        v3.require_validated_context_reference(tampered)


def test_preflight_pairing_changes_gM_but_not_M_free_reliability() -> None:
    reference = _reference()
    correct = _preflight(reference, query_z=-0.5, support_z=(-0.5, 0.5, 2.0))
    shuffled = _preflight(
        reference,
        query_z=-0.5,
        support_z=(-0.5, 0.5, 2.0),
        assignment=(1, 2, 0),
    )
    product = _support_product(3)
    correct_trust = v3.finalize_context_trust(correct, product.reliability_capability)
    shuffled_trust = v3.finalize_context_trust(shuffled, product.reliability_capability)
    assert correct.pairing_sha256 != shuffled.pairing_sha256
    assert correct_trust.support_reliability_capability_sha256 == (
        shuffled_trust.support_reliability_capability_sha256
    )
    assert correct_trust.g_M_by_query != shuffled_trust.g_M_by_query


def test_missing_metadata_is_exact_AQ_and_interface_only_has_no_effect() -> None:
    reference = _reference()
    product = _support_product(1)
    missing = {
        "packet_key": "query",
        "interface_lookup_key": "wet",
        "impedance_kohm_by_channel": [None] * 8,
        "channel_availability": [False] * 8,
    }
    support = {**missing, "packet_key": "packet01"}
    pairing = v3.context_pairing_sha256(
        query_key="query",
        ordered_support_block_keys=("block01",),
        query_packet_key="query",
        ordered_support_packet_keys=("packet01",),
    )
    preflight = v3.preflight_context(
        reference=reference,
        query_key="query",
        ordered_support_block_keys=("block01",),
        query_packet=missing,
        support_packets=(support,),
        pairing_sha256=pairing,
    )
    assert preflight.decision_reason == "all_missing_exact_A_Q"
    trust = v3.finalize_context_trust(preflight, product.reliability_capability)
    query = _score_fixture(1)[0]
    aq = v3.apply_v3_operator(
        query,
        query_key="query",
        budget=1,
        variant="A_Q",
        config=_config(),
        support_product=product,
    )
    aqm = v3.apply_v3_operator(
        query,
        query_key="query",
        budget=1,
        variant="A_QM",
        config=_config(),
        support_product=product,
        context_trust=trust,
    )
    assert trust.g_M_by_query == 1.0
    assert np.array_equal(aqm.fused_probabilities, aq.fused_probabilities)
    assert aqm.support_probabilities is product.support_probabilities


@pytest.mark.parametrize("failure", ["pairing", "ood", "wrong_count"])
def test_rejected_preflight_returns_A0_without_calling_support_loader(failure: str) -> None:
    reference = _reference()
    query = _raw_packet("query", z=10.0 if failure == "ood" else 0.0, reference=reference)
    support = _raw_packet("packet01", z=0.0, reference=reference)
    slots = ("block01",)
    pairing = v3.context_pairing_sha256(
        query_key="query",
        ordered_support_block_keys=slots,
        query_packet_key="query",
        ordered_support_packet_keys=("packet01",),
    )
    if failure == "pairing":
        pairing = _HASH_A
    elif failure == "wrong_count":
        support["impedance_kohm_by_channel"] = [1.0] * 7
    preflight = v3.preflight_context(
        reference=reference,
        query_key="query",
        ordered_support_block_keys=slots,
        query_packet=query,
        support_packets=(support,),
        pairing_sha256=pairing,
    )
    assert preflight.rejects_before_support_access
    calls = 0

    def forbidden_loader() -> v3.V3SupportProduct:
        nonlocal calls
        calls += 1
        raise AssertionError("support was accessed after rejecting preflight")

    scores = _score_fixture(1)[0]
    output = v3.apply_v3_after_preflight(
        scores,
        query_key="query",
        budget=1,
        config=_config(),
        preflight=preflight,
        support_loader=forbidden_loader,
    )
    assert calls == 0
    assert output.exact_fallback
    assert output.fused_probabilities is output.base_probabilities


def test_scientific_operator_has_no_raw_M_parameters_and_single_bounded_insertion() -> None:
    forbidden = {
        "interface",
        "interface_lookup_key",
        "impedance_kohm_by_channel",
        "channel_availability",
    }
    assert forbidden.isdisjoint(inspect.signature(v3.apply_v3_operator).parameters)
    product = _support_product(1)
    reference = _reference()
    preflight = _preflight(reference, query_z=-0.5, support_z=(0.5,))
    trust = v3.finalize_context_trust(preflight, product.reliability_capability)
    query = _score_fixture(1)[0]
    aq = v3.apply_v3_operator(
        query,
        query_key="query",
        budget=1,
        variant="A_Q",
        config=_config(1),
        support_product=product,
    )
    aqm = v3.apply_v3_operator(
        query,
        query_key="query",
        budget=1,
        variant="A_QM",
        config=_config(1),
        support_product=product,
        context_trust=trust,
    )
    assert aqm.g_M < 1.0
    assert np.array_equal(aq.support_probabilities, aqm.support_probabilities)
    assert np.allclose(aqm.effective_lambdas, aq.lambdas_Q * aqm.g_M)
    displacement = np.abs(aqm.fused_probabilities - aqm.base_probabilities).sum(axis=1)
    assert np.all(displacement <= 2.0 * aqm.lambdas_Q + 1.0e-12)


def test_prequential_gate_is_chronological_M_free_and_binding_checked() -> None:
    _, support, labels = _score_fixture(5)
    partitions = tuple(f"{index:x}" * 64 for index in range(1, 6))
    decision = v3.prequential_gate_decision(
        support,
        labels,
        budget=5,
        block_partition_sha256s=partitions,
        config=_config(),
    )
    assert decision.evaluated_blocks == (2, 3, 4, 5)
    assert decision.fit_block_prefixes == ((1,), (1, 2), (1, 2, 3), (1, 2, 3, 4))
    assert "metadata" not in inspect.signature(v3.prequential_gate_decision).parameters
    assert decision.enabled == (
        all(value >= 0.0 for value in decision.block_balanced_accuracy_deltas)
        and all(value > 0.0 for value in decision.block_log_probability_deltas)
        and any(value > 0.0 for value in decision.block_balanced_accuracy_deltas)
    )


def test_exact_derangement_counts_and_nonidentity() -> None:
    for depth, expected in ((3, 2), (5, 44)):
        values = v3.exhaustive_derangements(depth)
        assert len(values) == expected
        assert all(all(index != target for index, target in enumerate(value)) for value in values)
        assert tuple(sorted(values)) == values
