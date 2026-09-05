from __future__ import annotations

import inspect

import numpy as np
import pytest

from cfeg.models.metadata_calibration_v2 import (
    PrequentialFoldProvenance,
    TemplateScoreProvenance,
    V2OperatorConfig,
    apply_v2_prequential_operator,
    apply_v2_safe_operator,
    fuse_probabilities,
    normalize_fbcca_scores,
    normalized_entropy,
    prequential_gate_decision,
    relative_context_affinity,
    score_prototype_distribution,
    template_residual_scores_from_subbands,
)

_PAIRING_SHA256 = "a" * 64
_HASH_B = "b" * 64
_HASH_C = "c" * 64
_HASH_D = "d" * 64
_HASH_E = "e" * 64
_P2_WEIGHTS = np.asarray(
    [
        1.25,
        0.6704482076268572,
        0.5032785618838642,
        0.42677669529663687,
        0.38374806099528436,
        0.35649051737437876,
        0.33782687899303776,
    ]
)


def _template_provenance() -> TemplateScoreProvenance:
    return TemplateScoreProvenance(
        filterbank_sha256=_HASH_B,
        preprocessing_sha256=_HASH_C,
        query_partition_sha256=_HASH_D,
        support_partition_sha256=_HASH_E,
    )


def _fold_provenance(budget: int) -> tuple[PrequentialFoldProvenance, ...]:
    block_hashes = {block: f"{block:x}" * 64 for block in range(1, budget + 1)}
    return tuple(
        PrequentialFoldProvenance(
            evaluation_block=block,
            fit_blocks=tuple(range(1, block)),
            fit_block_partition_sha256s=tuple(
                block_hashes[fit_block] for fit_block in range(1, block)
            ),
            evaluation_partition_sha256=block_hashes[block],
        )
        for block in range(2, budget + 1)
    )


def _scores(budget: int = 3) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    query = np.asarray(
        [
            [0.90, 0.40, 0.10],
            [0.20, 0.75, 0.50],
            [0.45, 0.30, 0.70],
            [0.34, 0.33, 0.32],
        ],
        dtype=np.float64,
    )
    support_rows = []
    labels = []
    for block in range(budget):
        jitter = 0.01 * block
        support_rows.extend(
            (
                [0.70 + jitter, 0.62, 0.10],
                [0.15, 0.66 + jitter, 0.55],
                [0.52, 0.20, 0.68 + jitter],
            )
        )
        labels.extend((0, 1, 2))
    return query, np.asarray(support_rows), np.asarray(labels)


def _operator(
    budget: int = 3,
    **kwargs: object,
):
    query, support, labels = _scores(budget)
    parameters: dict[str, object] = {"variant": "A_Q", "gate_enabled": True}
    parameters.update(kwargs)
    return apply_v2_safe_operator(
        query,
        budget=budget,
        operator="score_prototype_shrinkage",
        support_fbcca_scores=support,
        support_labels=labels,
        **parameters,
    )


def test_score_normalization_preserves_argmax_and_is_query_batch_independent() -> None:
    scores, _, _ = _scores()
    normalized, probabilities = normalize_fbcca_scores(scores)
    assert np.array_equal(np.argmax(probabilities, axis=1), np.argmax(scores, axis=1))
    assert np.allclose(normalized.mean(axis=1), 0.0, atol=1.0e-15)
    assert np.allclose(np.mean(np.square(normalized), axis=1), 1.0)
    assert np.allclose(probabilities.sum(axis=1), 1.0)

    for row in range(len(scores)):
        singleton = normalize_fbcca_scores(scores[row : row + 1])[1]
        assert np.array_equal(singleton, probabilities[row : row + 1])

    constant = np.full((2, 3), 0.5)
    constant_normalized, constant_probability = normalize_fbcca_scores(constant)
    assert np.array_equal(constant_normalized, np.zeros_like(constant))
    assert np.array_equal(constant_probability, np.full((2, 3), 1.0 / 3.0))

    extreme = np.asarray([[0.0, 1.0e308, -1.0e308]])
    extreme_normalized, extreme_probability = normalize_fbcca_scores(extreme)
    assert np.isfinite(extreme_normalized).all()
    assert np.argmax(extreme_probability, axis=1).item() == 1


@pytest.mark.parametrize(
    ("budget", "enabled", "gate_enabled", "reason"),
    [
        (0, True, True, "k0_exact_strict_fbcca"),
        (3, False, True, "operator_disabled"),
        (3, True, False, "validation_gate_abstained"),
    ],
)
def test_k0_off_and_gate_short_circuit_return_exact_input_objects(
    budget: int,
    enabled: bool,
    gate_enabled: bool,
    reason: str,
) -> None:
    query, _, _ = _scores()
    base_probability = normalize_fbcca_scores(query)[1]
    output = apply_v2_safe_operator(
        query,
        budget=budget,
        operator="score_prototype_shrinkage",
        variant="A_Q",
        support_fbcca_scores=object(),  # type: ignore[arg-type]
        support_labels=object(),  # type: ignore[arg-type]
        base_probabilities=base_probability,
        enabled=enabled,
        gate_enabled=gate_enabled,
    )
    assert output.base_scores is query
    assert output.base_probabilities is base_probability
    assert output.support_probabilities is base_probability
    assert output.fused_probabilities is base_probability
    assert output.exact_fallback
    assert output.fallback_reason == reason
    assert np.array_equal(output.predictions, np.argmax(query, axis=1))
    assert output.affinities.shape == (len(query), 0)


def test_zero_lambda_is_an_exact_null_candidate_before_support_access() -> None:
    query, _, _ = _scores()
    base_probability = normalize_fbcca_scores(query)[1]
    output = apply_v2_safe_operator(
        query,
        budget=5,
        operator="score_prototype_shrinkage",
        variant="A_Q",
        config=V2OperatorConfig(lambda_max=0.0),
        support_fbcca_scores=object(),  # type: ignore[arg-type]
        support_labels=object(),  # type: ignore[arg-type]
        base_probabilities=base_probability,
        gate_enabled=True,
    )
    assert output.exact_fallback
    assert output.fallback_reason == "exact_null_candidate"
    assert output.fused_probabilities is base_probability


def test_missing_gate_authorization_fails_closed_before_support_access() -> None:
    query, _, _ = _scores()
    output = apply_v2_safe_operator(
        query,
        budget=3,
        operator="score_prototype_shrinkage",
        variant="A_Q",
        support_fbcca_scores=object(),  # type: ignore[arg-type]
        support_labels=object(),  # type: ignore[arg-type]
    )
    assert output.exact_fallback
    assert output.fallback_reason == "missing_gate_authorization"
    assert output.gate_enabled is None

    aqm = apply_v2_safe_operator(
        query,
        budget=3,
        operator="score_prototype_shrinkage",
        variant="A_QM",
        support_fbcca_scores=object(),  # type: ignore[arg-type]
        support_labels=object(),  # type: ignore[arg-type]
        query_interfaces=object(),  # type: ignore[arg-type]
        relative_context_pairing_sha256="not-inspected",
    )
    assert aqm.exact_fallback
    assert aqm.relative_context_pairing_sha256 is None


def test_k0_aqm_does_not_parse_context_or_pairing_capability() -> None:
    query, _, _ = _scores()
    output = apply_v2_safe_operator(
        query,
        budget=0,
        operator="score_prototype_shrinkage",
        variant="A_QM",
        query_interfaces=object(),  # type: ignore[arg-type]
        support_interfaces=object(),  # type: ignore[arg-type]
        relative_context_pairing_sha256="not-inspected",
    )
    assert output.exact_fallback
    assert output.fallback_reason == "k0_exact_strict_fbcca"
    assert output.relative_context_pairing_sha256 is None


def test_score_prototype_fusion_is_bounded_and_uses_uncertainty_scaling() -> None:
    output = _operator(config=V2OperatorConfig(lambda_max=0.30))
    expected = 0.30 * (3.0 / 4.0) * normalized_entropy(output.base_probabilities)
    assert np.allclose(output.lambdas, expected, rtol=0.0, atol=1.0e-15)
    assert output.support_probabilities.shape == output.base_probabilities.shape
    assert np.allclose(output.fused_probabilities.sum(axis=1), 1.0)
    displacement = np.abs(output.fused_probabilities - output.base_probabilities).sum(axis=1)
    assert np.all(displacement <= 2.0 * output.lambdas + 1.0e-12)
    assert output.lambdas[-1] > output.lambdas[0]
    assert not output.exact_fallback


def test_convex_fusion_zero_is_identity_and_rejects_nonconvex_weights() -> None:
    base = np.asarray([[0.7, 0.2, 0.1], [0.2, 0.5, 0.3]])
    support = np.asarray([[0.1, 0.2, 0.7], [0.6, 0.3, 0.1]])
    assert fuse_probabilities(base, support, 0.0) is base
    mixed = fuse_probabilities(base, support, np.asarray([0.1, 0.3]))
    assert np.allclose(mixed[0], 0.9 * base[0] + 0.1 * support[0])
    assert np.allclose(mixed[1], 0.7 * base[1] + 0.3 * support[1])
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        fuse_probabilities(base, support, np.asarray([0.2, 1.1]))


def test_relative_context_affinity_formula_and_all_missing_neutrality() -> None:
    observed = relative_context_affinity(
        n_query=2,
        n_support=2,
        query_interfaces=np.asarray(["dry", "wet"], dtype=object),
        support_interfaces=np.asarray(["dry", "wet"], dtype=object),
        query_impedance_kohm=np.asarray([[1.0, np.nan], [3.0, 7.0]]),
        support_impedance_kohm=np.asarray([[3.0, 99.0], [7.0, 3.0]]),
    )
    # dry/dry and one comparable channel: |log2((1+1)/(1+3))| = 1.
    assert observed[0, 0] == pytest.approx(0.5)
    # dry/wet mismatch (0.25) and |log2(2/8)|=2 (0.25).
    assert observed[0, 1] == pytest.approx(0.0625)
    assert np.all((observed > 0.0) & (observed <= 1.0))

    missing = relative_context_affinity(
        n_query=2,
        n_support=3,
        query_interfaces=[None, "unknown"],
        support_interfaces=[None, "n/a", np.nan],
        query_impedance_kohm=np.full((2, 4), np.nan),
        support_impedance_kohm=np.full((3, 4), np.nan),
    )
    assert np.array_equal(missing, np.ones((2, 3)))


def test_variant_capabilities_are_validated_and_audited() -> None:
    query, support, labels = _scores(1)
    with pytest.raises(ValueError, match="A_Q forbids"):
        apply_v2_safe_operator(
            query,
            budget=1,
            operator="score_prototype_shrinkage",
            variant="A_Q",
            gate_enabled=True,
            support_fbcca_scores=support,
            support_labels=labels,
            query_interfaces=["dry"] * len(query),
        )
    with pytest.raises((TypeError, ValueError), match="SHA-256"):
        apply_v2_safe_operator(
            query,
            budget=1,
            operator="score_prototype_shrinkage",
            variant="A_QM",
            gate_enabled=True,
            support_fbcca_scores=support,
            support_labels=labels,
            relative_context_pairing_sha256="not-a-digest",
        )

    aqm = apply_v2_safe_operator(
        query,
        budget=1,
        operator="score_prototype_shrinkage",
        variant="A_QM",
        gate_enabled=True,
        support_fbcca_scores=support,
        support_labels=labels,
        query_interfaces=["dry"] * len(query),
        support_interfaces=["wet"] * len(support),
        relative_context_pairing_sha256=_PAIRING_SHA256,
    )
    assert aqm.relative_context_pairing_sha256 == _PAIRING_SHA256
    assert aqm.comparable_context_pair_count == len(query) * len(support)


def test_p1_context_changes_prototypes_and_aggregate_effective_mass() -> None:
    query, support, labels = _scores(1)
    aq = apply_v2_safe_operator(
        query,
        budget=1,
        operator="score_prototype_shrinkage",
        variant="A_Q",
        gate_enabled=True,
        support_fbcca_scores=support,
        support_labels=labels,
    )
    aqm = apply_v2_safe_operator(
        query,
        budget=1,
        operator="score_prototype_shrinkage",
        variant="A_QM",
        gate_enabled=True,
        support_fbcca_scores=support,
        support_labels=labels,
        query_interfaces=["dry"] * len(query),
        support_interfaces=["wet"] * len(support),
        relative_context_pairing_sha256=_PAIRING_SHA256,
    )
    assert not np.array_equal(aq.support_probabilities, aqm.support_probabilities)
    assert np.allclose(aqm.lambdas, 0.25 * aq.lambdas, rtol=0.0, atol=1.0e-15)


def test_all_missing_aqm_is_exact_aq_and_support_order_is_exactly_invariant() -> None:
    query, support, labels = _scores(3)
    aq = apply_v2_safe_operator(
        query,
        budget=3,
        operator="score_prototype_shrinkage",
        variant="A_Q",
        gate_enabled=True,
        support_fbcca_scores=support,
        support_labels=labels,
    )
    aqm_missing = apply_v2_safe_operator(
        query,
        budget=3,
        operator="score_prototype_shrinkage",
        variant="A_QM",
        gate_enabled=True,
        support_fbcca_scores=support,
        support_labels=labels,
        query_interfaces=[None] * len(query),
        support_interfaces=[None] * len(support),
        query_impedance_kohm=np.full((len(query), 4), np.nan),
        support_impedance_kohm=np.full((len(support), 4), np.nan),
        relative_context_pairing_sha256=_PAIRING_SHA256,
    )
    for field in (
        "base_probabilities",
        "support_probabilities",
        "fused_probabilities",
        "lambdas",
        "affinities",
        "predictions",
    ):
        assert np.array_equal(getattr(aq, field), getattr(aqm_missing, field))

    order = np.asarray([7, 1, 8, 4, 0, 6, 5, 2, 3])
    reordered = apply_v2_safe_operator(
        query,
        budget=3,
        operator="score_prototype_shrinkage",
        variant="A_Q",
        gate_enabled=True,
        support_fbcca_scores=support[order],
        support_labels=labels[order],
    )
    assert np.array_equal(aq.support_probabilities, reordered.support_probabilities)
    assert np.array_equal(aq.fused_probabilities, reordered.fused_probabilities)


def test_score_prototype_is_class_permutation_equivariant() -> None:
    query_scores, support_scores, labels = _scores(3)
    query_probability = normalize_fbcca_scores(query_scores)[1]
    support_probability = normalize_fbcca_scores(support_scores)[1]
    original = score_prototype_distribution(
        query_probability,
        support_probability,
        labels,
    )
    permutation = np.asarray([2, 0, 1])
    inverse = np.argsort(permutation)
    permuted = score_prototype_distribution(
        query_probability[:, permutation],
        support_probability[:, permutation],
        inverse[labels],
    )
    assert np.allclose(permuted, original[:, permutation], rtol=0.0, atol=1.0e-15)


def _template_subbands(
    budget: int = 3,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    timeline = np.linspace(0.0, 1.0, 40, endpoint=False)
    labels = np.tile(np.arange(3), budget)
    support = np.empty((7, len(labels), 2, len(timeline)), dtype=np.float64)
    query = np.empty((7, 3, 2, len(timeline)), dtype=np.float64)
    for band in range(7):
        for row, label in enumerate(labels):
            base = np.sin(2.0 * np.pi * (label + 2) * timeline + 0.05 * row)
            support[band, row, 0] = base
            support[band, row, 1] = (0.5 + 0.1 * band) * base
        for label in range(3):
            base = np.sin(2.0 * np.pi * (label + 2) * timeline + 0.05)
            query[band, label, 0] = base
            query[band, label, 1] = (0.5 + 0.1 * band) * base
    return query, support, labels, _P2_WEIGHTS.copy()


def test_template_residual_supports_context_weighting_and_high_level_p2() -> None:
    query_subbands, support_subbands, labels, weights = _template_subbands(3)
    neutral = template_residual_scores_from_subbands(
        query_subbands,
        support_subbands,
        labels,
        weights,
    )
    assert np.array_equal(np.argmax(neutral, axis=1), np.arange(3))
    assert np.isfinite(neutral).all()

    query_scores, support_scores, support_labels = _scores(3)
    output = apply_v2_safe_operator(
        query_scores[:3],
        budget=3,
        operator="filterbank_target_template_residual",
        variant="A_QM",
        gate_enabled=True,
        support_fbcca_scores=support_scores,
        support_labels=support_labels,
        template_query_subbands=query_subbands,
        template_support_subbands=support_subbands,
        template_subband_weights=weights,
        query_interfaces=["dry", "dry", "dry"],
        support_interfaces=np.where(np.arange(len(support_labels)) % 2, "wet", "dry"),
        relative_context_pairing_sha256=_PAIRING_SHA256,
    )
    assert output.support_probabilities.shape == (3, 3)
    assert np.allclose(output.fused_probabilities.sum(axis=1), 1.0)
    assert not np.array_equal(output.affinities, np.ones_like(output.affinities))

    with pytest.raises(ValueError, match="Unweighted precomputed"):
        apply_v2_safe_operator(
            query_scores[:3],
            budget=3,
            operator="filterbank_target_template_residual",
            variant="A_QM",
            gate_enabled=True,
            support_fbcca_scores=support_scores,
            support_labels=support_labels,
            template_query_class_scores=neutral,
            template_score_provenance=_template_provenance(),
            query_interfaces=["dry", "dry", "dry"],
            support_interfaces=np.where(np.arange(len(support_labels)) % 2, "wet", "dry"),
            relative_context_pairing_sha256=_PAIRING_SHA256,
        )


def test_p2_k1_effective_mass_and_all_missing_exact_aq() -> None:
    query_subbands, support_subbands, _labels, weights = _template_subbands(1)
    query_scores, support_scores, support_labels = _scores(1)
    common = {
        "budget": 1,
        "operator": "filterbank_target_template_residual",
        "gate_enabled": True,
        "support_fbcca_scores": support_scores,
        "support_labels": support_labels,
        "template_query_subbands": query_subbands,
        "template_support_subbands": support_subbands,
        "template_subband_weights": weights,
    }
    aq = apply_v2_safe_operator(query_scores[:3], variant="A_Q", **common)
    aqm_missing = apply_v2_safe_operator(
        query_scores[:3],
        variant="A_QM",
        relative_context_pairing_sha256=_PAIRING_SHA256,
        **common,
    )
    for field in ("support_probabilities", "fused_probabilities", "lambdas", "predictions"):
        assert np.array_equal(getattr(aq, field), getattr(aqm_missing, field))
    assert aqm_missing.comparable_context_pair_count == 0

    aqm_mismatch = apply_v2_safe_operator(
        query_scores[:3],
        variant="A_QM",
        query_interfaces=["dry"] * 3,
        support_interfaces=["wet"] * 3,
        relative_context_pairing_sha256=_PAIRING_SHA256,
        **common,
    )
    # At k=1 the within-class scalar cancels from each normalized template;
    # r3 retains metadata through the aggregate effective-mass multiplier.
    assert np.array_equal(aq.support_probabilities, aqm_mismatch.support_probabilities)
    assert np.allclose(aqm_mismatch.lambdas, 0.25 * aq.lambdas, rtol=0.0, atol=1.0e-15)


def test_p2_query_support_and_class_permutations_are_equivariant() -> None:
    query_subbands, support_subbands, _labels, weights = _template_subbands(3)
    query_scores, support_scores, support_labels = _scores(3)
    original = apply_v2_safe_operator(
        query_scores[:3],
        budget=3,
        operator="filterbank_target_template_residual",
        variant="A_Q",
        gate_enabled=True,
        support_fbcca_scores=support_scores,
        support_labels=support_labels,
        template_query_subbands=query_subbands,
        template_support_subbands=support_subbands,
        template_subband_weights=weights,
    )
    query_order = np.asarray([2, 0, 1])
    support_order = np.asarray([7, 1, 8, 4, 0, 6, 5, 2, 3])
    reordered = apply_v2_safe_operator(
        query_scores[query_order],
        budget=3,
        operator="filterbank_target_template_residual",
        variant="A_Q",
        gate_enabled=True,
        support_fbcca_scores=support_scores[support_order],
        support_labels=support_labels[support_order],
        template_query_subbands=query_subbands[:, query_order],
        template_support_subbands=support_subbands[:, support_order],
        template_subband_weights=weights,
    )
    assert np.array_equal(
        original.support_probabilities[query_order], reordered.support_probabilities
    )
    assert np.array_equal(original.fused_probabilities[query_order], reordered.fused_probabilities)

    permutation = np.asarray([2, 0, 1])
    inverse = np.argsort(permutation)
    class_permuted = apply_v2_safe_operator(
        query_scores[:3, permutation],
        budget=3,
        operator="filterbank_target_template_residual",
        variant="A_Q",
        gate_enabled=True,
        support_fbcca_scores=support_scores[:, permutation],
        support_labels=inverse[support_labels],
        template_query_subbands=query_subbands,
        template_support_subbands=support_subbands,
        template_subband_weights=weights,
    )
    assert np.allclose(
        class_permuted.support_probabilities,
        original.support_probabilities[:, permutation],
        rtol=0.0,
        atol=1.0e-15,
    )


def test_precomputed_p2_requires_and_echoes_typed_provenance() -> None:
    query_subbands, support_subbands, labels, weights = _template_subbands(3)
    scores = template_residual_scores_from_subbands(
        query_subbands, support_subbands, labels, weights
    )
    query_scores, support_scores, support_labels = _scores(3)
    common = {
        "budget": 3,
        "operator": "filterbank_target_template_residual",
        "variant": "A_Q",
        "gate_enabled": True,
        "support_fbcca_scores": support_scores,
        "support_labels": support_labels,
        "template_query_class_scores": scores,
    }
    with pytest.raises(ValueError, match="typed producer provenance"):
        apply_v2_safe_operator(query_scores[:3], **common)
    with pytest.raises(TypeError, match="TemplateScoreProvenance"):
        apply_v2_safe_operator(
            query_scores[:3],
            template_score_provenance=object(),
            **common,  # type: ignore[arg-type]
        )
    provenance = _template_provenance()
    output = apply_v2_safe_operator(
        query_scores[:3], template_score_provenance=provenance, **common
    )
    assert output.template_score_provenance is provenance

    with pytest.raises(ValueError, match="exact frozen seven subband weights"):
        apply_v2_safe_operator(
            query_scores[:3],
            budget=3,
            operator="filterbank_target_template_residual",
            variant="A_Q",
            gate_enabled=True,
            support_fbcca_scores=support_scores,
            support_labels=support_labels,
            template_query_subbands=query_subbands,
            template_support_subbands=support_subbands,
            template_subband_weights=np.ones(7),
        )


def _gate_rows(helpful: bool = True) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    labels = np.tile(np.arange(3), 2)
    blocks = np.repeat(np.asarray([2, 3]), 3)
    base = np.full((6, 3), 0.25)
    base[np.arange(6), (labels + 1) % 3] = 0.50
    if helpful:
        candidate = np.full((6, 3), 0.15)
        candidate[np.arange(6), labels] = 0.70
    else:
        candidate = np.full((6, 3), 0.15)
        candidate[np.arange(6), (labels + 2) % 3] = 0.70
    return base, candidate, labels, blocks


def test_prequential_gate_uses_only_complete_later_calibration_blocks() -> None:
    base, helpful, labels, blocks = _gate_rows(helpful=True)
    passed = prequential_gate_decision(
        base,
        helpful,
        labels,
        blocks,
        budget=3,
        fold_provenance=_fold_provenance(3),
    )
    assert passed.enabled
    assert passed.evaluated_blocks == (2, 3)
    assert passed.mean_balanced_accuracy_delta == 1.0
    assert passed.mean_log_probability_delta > 0.0

    _, harmful, _, _ = _gate_rows(helpful=False)
    failed = prequential_gate_decision(
        base,
        harmful,
        labels,
        blocks,
        budget=3,
        fold_provenance=_fold_provenance(3),
    )
    assert not failed.enabled
    assert failed.reason == "prequential_gate_abstained"

    # No target-local k=1 evidence exists, so even poison inputs are untouched.
    k1 = prequential_gate_decision(
        object(),  # type: ignore[arg-type]
        object(),  # type: ignore[arg-type]
        object(),  # type: ignore[arg-type]
        object(),  # type: ignore[arg-type]
        budget=1,
    )
    assert not k1.enabled
    assert k1.evaluated_blocks == ()


def test_prequential_gate_rejects_missing_or_nonprior_fold_provenance() -> None:
    base, candidate, labels, blocks = _gate_rows(helpful=True)
    with pytest.raises(ValueError, match="per-fold provenance"):
        prequential_gate_decision(base, candidate, labels, blocks, budget=3)
    with pytest.raises(ValueError, match="exact prior fit_blocks"):
        PrequentialFoldProvenance(
            evaluation_block=3,
            fit_blocks=(1,),
            fit_block_partition_sha256s=(_HASH_B,),
            evaluation_partition_sha256=_HASH_C,
        )
    with pytest.raises(ValueError, match="ordered by the exact evaluation blocks"):
        prequential_gate_decision(
            base,
            candidate,
            labels,
            blocks,
            budget=3,
            fold_provenance=tuple(reversed(_fold_provenance(3))),
        )

    first, second = _fold_provenance(3)
    broken_chain = PrequentialFoldProvenance(
        evaluation_block=3,
        fit_blocks=(1, 2),
        fit_block_partition_sha256s=(first.fit_block_partition_sha256s[0], _HASH_E),
        evaluation_partition_sha256=second.evaluation_partition_sha256,
    )
    with pytest.raises(ValueError, match="chronological prefix chain"):
        prequential_gate_decision(
            base,
            candidate,
            labels,
            blocks,
            budget=3,
            fold_provenance=(first, broken_chain),
        )


def test_dedicated_prequential_operator_supports_intermediate_depth_two() -> None:
    query, support, labels = _scores(2)
    provenance = _fold_provenance(3)[1]
    output = apply_v2_prequential_operator(
        query[:3],
        final_budget=3,
        fold_provenance=provenance,
        operator="score_prototype_shrinkage",
        variant="A_Q",
        support_fbcca_scores=support,
        support_labels=labels,
    )
    assert output.final_budget == 3
    assert output.support_depth == 2
    assert output.prequential_fold_provenance is provenance
    assert output.gate_enabled is True
    assert not output.exact_fallback
    expected = 0.10 * (2.0 / 3.0) * normalized_entropy(output.base_probabilities)
    assert np.allclose(output.lambdas, expected, rtol=0.0, atol=1.0e-15)


def test_final_query_rejects_intermediate_budget_and_prequential_is_receipt_gated() -> None:
    query, support, labels = _scores(2)
    with pytest.raises(ValueError, match="one of"):
        apply_v2_safe_operator(
            query,
            budget=2,
            operator="score_prototype_shrinkage",
            variant="A_Q",
            gate_enabled=True,
            support_fbcca_scores=support,
            support_labels=labels,
        )
    with pytest.raises(TypeError, match="PrequentialFoldProvenance"):
        apply_v2_prequential_operator(
            query,
            final_budget=3,
            fold_provenance=object(),  # type: ignore[arg-type]
            operator="score_prototype_shrinkage",
            variant="A_Q",
            support_fbcca_scores=support,
            support_labels=labels,
        )
    with pytest.raises(ValueError, match="must not exceed"):
        apply_v2_prequential_operator(
            query,
            final_budget=3,
            fold_provenance=_fold_provenance(5)[2],
            operator="score_prototype_shrinkage",
            variant="A_Q",
            support_fbcca_scores=support,
            support_labels=labels,
        )
    evaluation_one = PrequentialFoldProvenance(
        evaluation_block=1,
        fit_blocks=(),
        fit_block_partition_sha256s=(),
        evaluation_partition_sha256=_HASH_B,
    )
    with pytest.raises(ValueError, match="at least two"):
        apply_v2_prequential_operator(
            query,
            final_budget=3,
            fold_provenance=evaluation_one,
            operator="score_prototype_shrinkage",
            variant="A_Q",
            support_fbcca_scores=support,
            support_labels=labels,
        )


def test_prequential_p2_provenance_binds_evaluation_and_fit_partitions() -> None:
    query, support, labels = _scores(2)
    provenance = _fold_provenance(3)[1]
    scores = np.asarray(
        [[0.9, 0.2, 0.1], [0.1, 0.8, 0.2], [0.2, 0.1, 0.7]],
        dtype=np.float64,
    )
    matching = TemplateScoreProvenance(
        filterbank_sha256=_HASH_B,
        preprocessing_sha256=_HASH_C,
        query_partition_sha256=provenance.evaluation_partition_sha256,
        support_partition_sha256=provenance.fit_partition_sha256,
    )
    with pytest.raises(TypeError, match="TemplateScoreProvenance"):
        apply_v2_prequential_operator(
            query[:3],
            final_budget=3,
            fold_provenance=provenance,
            operator="filterbank_target_template_residual",
            variant="A_Q",
            support_fbcca_scores=support,
            support_labels=labels,
            template_query_class_scores=scores,
            template_score_provenance=object(),  # type: ignore[arg-type]
        )
    output = apply_v2_prequential_operator(
        query[:3],
        final_budget=3,
        fold_provenance=provenance,
        operator="filterbank_target_template_residual",
        variant="A_Q",
        support_fbcca_scores=support,
        support_labels=labels,
        template_query_class_scores=scores,
        template_score_provenance=matching,
    )
    assert output.template_score_provenance is matching

    mismatched = TemplateScoreProvenance(
        filterbank_sha256=_HASH_B,
        preprocessing_sha256=_HASH_C,
        query_partition_sha256=_HASH_E,
        support_partition_sha256=provenance.fit_partition_sha256,
    )
    with pytest.raises(ValueError, match="must match the prequential"):
        apply_v2_prequential_operator(
            query[:3],
            final_budget=3,
            fold_provenance=provenance,
            operator="filterbank_target_template_residual",
            variant="A_Q",
            support_fbcca_scores=support,
            support_labels=labels,
            template_query_class_scores=scores,
            template_score_provenance=mismatched,
        )


def test_high_level_rejects_partial_support_and_has_no_query_label_or_id_api() -> None:
    query, support, labels = _scores(3)
    with pytest.raises(ValueError, match="exactly 3 support"):
        apply_v2_safe_operator(
            query,
            budget=3,
            operator="score_prototype_shrinkage",
            variant="A_Q",
            gate_enabled=True,
            support_fbcca_scores=support[:-1],
            support_labels=labels[:-1],
        )
    forbidden = {
        "query_labels",
        "query_y",
        "subject_id",
        "sample_id",
        "file_name",
        "row_id",
    }
    for function in (apply_v2_safe_operator, apply_v2_prequential_operator):
        parameters = set(inspect.signature(function).parameters)
        assert not parameters.intersection(forbidden)
