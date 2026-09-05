from __future__ import annotations

import hashlib
import math
from collections.abc import Sequence
from dataclasses import dataclass
from numbers import Integral, Real
from typing import Literal

import numpy as np

V2SupportOperator = Literal[
    "score_prototype_shrinkage",
    "filterbank_target_template_residual",
]
V2Variant = Literal["A_Q", "A_QM"]

_SCHEMA = "cfeg.metadata-calibration-v2-safe-operator.v2"
_TEMPLATE_PROVENANCE_SCHEMA = "cfeg.metadata-calibration-v2-template-provenance.v1"
_BUDGETS = (0, 1, 3, 5)
_P2_SUBBAND_WEIGHTS = np.asarray(
    [
        1.25,
        0.6704482076268572,
        0.5032785618838642,
        0.42677669529663687,
        0.38374806099528436,
        0.35649051737437876,
        0.33782687899303776,
    ],
    dtype=np.float64,
)


@dataclass(frozen=True)
class V2OperatorConfig:
    """Frozen numerical controls for the label-free-query V2 support operator."""

    score_epsilon: float = 1.0e-12
    ideal_prototype_smoothing: float = 0.05
    prototype_prior_pseudocount: float = 4.0
    lambda_max: float = 0.10
    different_interface_affinity: float = 0.25
    entropy_scaling: bool = True

    def __post_init__(self) -> None:
        _positive_finite(self.score_epsilon, "score_epsilon")
        smoothing = _finite_float(self.ideal_prototype_smoothing, "ideal_prototype_smoothing")
        if not 0.0 <= smoothing < 1.0:
            raise ValueError("ideal_prototype_smoothing must be in [0, 1).")
        _positive_finite(
            self.prototype_prior_pseudocount,
            "prototype_prior_pseudocount",
        )
        maximum = _finite_float(self.lambda_max, "lambda_max")
        if not 0.0 <= maximum <= 1.0:
            raise ValueError("lambda_max must be in [0, 1].")
        interface = _finite_float(
            self.different_interface_affinity,
            "different_interface_affinity",
        )
        if not 0.0 < interface <= 1.0:
            raise ValueError("different_interface_affinity must be in (0, 1].")
        if type(self.entropy_scaling) is not bool:
            raise TypeError("entropy_scaling must be a bool.")


@dataclass(frozen=True)
class TemplateScoreProvenance:
    """Capability binding cached P2 scores to governed signal producers."""

    filterbank_sha256: str
    preprocessing_sha256: str
    query_partition_sha256: str
    support_partition_sha256: str
    schema: str = _TEMPLATE_PROVENANCE_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != _TEMPLATE_PROVENANCE_SCHEMA:
            raise ValueError(f"template provenance schema must be {_TEMPLATE_PROVENANCE_SCHEMA!r}.")
        for name in (
            "filterbank_sha256",
            "preprocessing_sha256",
            "query_partition_sha256",
            "support_partition_sha256",
        ):
            _sha256(getattr(self, name), name)


@dataclass(frozen=True)
class PrequentialFoldProvenance:
    """Declares the exact earlier calibration blocks used for one fold."""

    evaluation_block: int
    fit_blocks: tuple[int, ...]

    def __post_init__(self) -> None:
        evaluation = _positive_integer(self.evaluation_block, "evaluation_block")
        if not isinstance(self.fit_blocks, tuple):
            raise TypeError("fit_blocks must be a tuple of exact integers.")
        resolved = tuple(_positive_integer(value, "fit_blocks") for value in self.fit_blocks)
        expected = tuple(range(1, evaluation))
        if resolved != expected:
            raise ValueError(
                f"evaluation block {evaluation} requires exact prior fit_blocks {expected}."
            )


@dataclass(frozen=True)
class V2OperatorOutput:
    """Auditable output of :func:`apply_v2_safe_operator`.

    ``base_scores`` is the exact object supplied by the caller when it is a
    NumPy array.  On an exact fallback, ``support_probabilities`` and
    ``fused_probabilities`` are the very same object as ``base_probabilities``;
    the implementation deliberately avoids a nominal ``(1 - 0) * p + 0 * u``
    calculation.
    """

    base_scores: np.ndarray
    base_probabilities: np.ndarray
    support_probabilities: np.ndarray
    fused_probabilities: np.ndarray
    predictions: np.ndarray
    lambdas: np.ndarray
    affinities: np.ndarray
    gate_enabled: bool | None
    exact_fallback: bool
    operator: V2SupportOperator
    variant: V2Variant
    relative_context_pairing_sha256: str | None
    comparable_context_pair_count: int
    template_score_provenance: TemplateScoreProvenance | None
    schema: str = _SCHEMA
    fallback_reason: str | None = None


@dataclass(frozen=True)
class PrequentialGateDecision:
    enabled: bool
    budget: int
    evaluated_blocks: tuple[int, ...]
    block_balanced_accuracy_deltas: tuple[float, ...]
    block_log_probability_deltas: tuple[float, ...]
    mean_balanced_accuracy_delta: float
    mean_log_probability_delta: float
    reason: str
    fold_provenance: tuple[PrequentialFoldProvenance, ...]


def normalize_fbcca_scores(
    scores: np.ndarray,
    *,
    epsilon: float = 1.0e-12,
) -> tuple[np.ndarray, np.ndarray]:
    """Row-center, RMS-normalize and softmax full class-score vectors.

    The transformation is per query and uses no query-batch statistic.  It
    preserves the raw score argmax for every non-constant row.  Constant rows
    become a uniform distribution with the canonical lowest-index tie break.
    """

    values = _finite_matrix(scores, "scores")
    eps = _positive_finite(epsilon, "epsilon")
    as_float = values.astype(np.float64, copy=False)
    # Scaling before centering preserves the row-center/RMS formula while
    # avoiding overflow in the mean, subtraction, and square for any finite
    # float64 input.  eps is rescaled to retain its original score units.
    scale = np.max(np.abs(as_float), axis=1, keepdims=True)
    safe_scale = np.where(scale > 0.0, scale, 1.0)
    scaled = as_float / safe_scale
    centered = scaled - scaled.mean(axis=1, keepdims=True)
    rms = np.sqrt(np.mean(np.square(centered), axis=1, keepdims=True))
    normalized = centered / np.maximum(rms, eps / safe_scale)
    normalized -= normalized.mean(axis=1, keepdims=True)
    probabilities = _softmax(normalized)
    return normalized, probabilities


def normalized_entropy(probabilities: np.ndarray) -> np.ndarray:
    """Return row-wise entropy divided by ``log(n_classes)`` in ``[0, 1]``."""

    values = _probability_matrix(probabilities, "probabilities")
    n_classes = values.shape[1]
    if n_classes <= 1:
        raise ValueError("Entropy scaling requires at least two classes.")
    positive = values > 0.0
    terms = np.zeros_like(values, dtype=np.float64)
    terms[positive] = values[positive] * np.log(values[positive])
    entropy = -terms.sum(axis=1) / math.log(n_classes)
    return np.clip(entropy, 0.0, 1.0)


def relative_context_affinity(
    *,
    n_query: int,
    n_support: int,
    query_interfaces: Sequence[object] | np.ndarray | None = None,
    support_interfaces: Sequence[object] | np.ndarray | None = None,
    query_impedance_kohm: np.ndarray | None = None,
    support_impedance_kohm: np.ndarray | None = None,
    different_interface_affinity: float = 0.25,
) -> np.ndarray:
    """Compute bounded query-to-support acquisition-context affinity.

    Interface mismatch contributes a fixed factor.  Impedance contributes
    ``2**(-median(abs(log2((1 + query) / (1 + support)))))`` over channels
    observed on both rows.  Missing comparisons are neutral.  Therefore a
    completely missing context returns an exact matrix of ones, making A_QM's
    support calculation numerically identical to A_Q.
    """

    affinities, _ = _relative_context_affinity_and_count(
        n_query=n_query,
        n_support=n_support,
        query_interfaces=query_interfaces,
        support_interfaces=support_interfaces,
        query_impedance_kohm=query_impedance_kohm,
        support_impedance_kohm=support_impedance_kohm,
        different_interface_affinity=different_interface_affinity,
    )
    return affinities


def _relative_context_affinity_and_count(
    *,
    n_query: int,
    n_support: int,
    query_interfaces: Sequence[object] | np.ndarray | None = None,
    support_interfaces: Sequence[object] | np.ndarray | None = None,
    query_impedance_kohm: np.ndarray | None = None,
    support_impedance_kohm: np.ndarray | None = None,
    different_interface_affinity: float = 0.25,
) -> tuple[np.ndarray, int]:
    q_count = _nonnegative_integer(n_query, "n_query")
    s_count = _nonnegative_integer(n_support, "n_support")
    mismatch_factor = _finite_float(different_interface_affinity, "different_interface_affinity")
    if not 0.0 < mismatch_factor <= 1.0:
        raise ValueError("different_interface_affinity must be in (0, 1].")
    affinities = np.ones((q_count, s_count), dtype=np.float64)
    comparable_pairs = np.zeros((q_count, s_count), dtype=bool)

    query_interface = _optional_interfaces(query_interfaces, q_count, "query_interfaces")
    support_interface = _optional_interfaces(support_interfaces, s_count, "support_interfaces")
    if query_interface is not None and support_interface is not None:
        for query_index, query_value in enumerate(query_interface):
            normalized_query = _normalized_interface(query_value)
            if normalized_query is None:
                continue
            for support_index, support_value in enumerate(support_interface):
                normalized_support = _normalized_interface(support_value)
                if normalized_support is None:
                    continue
                comparable_pairs[query_index, support_index] = True
                if normalized_query != normalized_support:
                    affinities[query_index, support_index] *= mismatch_factor

    query_impedance = _optional_impedance(query_impedance_kohm, q_count, "query_impedance_kohm")
    support_impedance = _optional_impedance(
        support_impedance_kohm, s_count, "support_impedance_kohm"
    )
    if query_impedance is not None and support_impedance is not None:
        if query_impedance.shape[1] != support_impedance.shape[1]:
            raise ValueError("Query and support impedance channel counts must match.")
        for query_index in range(q_count):
            for support_index in range(s_count):
                query_row = query_impedance[query_index]
                support_row = support_impedance[support_index]
                comparable = np.isfinite(query_row) & np.isfinite(support_row)
                if not comparable.any():
                    continue
                comparable_pairs[query_index, support_index] = True
                log_ratio = np.abs(
                    np.log2((1.0 + query_row[comparable]) / (1.0 + support_row[comparable]))
                )
                distance = float(np.median(log_ratio))
                affinities[query_index, support_index] *= 2.0 ** (-distance)

    if not np.isfinite(affinities).all() or np.any(affinities <= 0.0):
        raise RuntimeError("Context affinity must stay positive and finite.")
    if np.any(affinities > 1.0):
        raise RuntimeError("Context affinity must not upweight support above one.")
    return affinities, int(comparable_pairs.sum())


def score_prototype_distribution(
    query_probabilities: np.ndarray,
    support_probabilities: np.ndarray,
    support_labels: np.ndarray,
    *,
    affinities: np.ndarray | None = None,
    ideal_prototype_smoothing: float = 0.05,
    prior_pseudocount: float = 4.0,
    epsilon: float = 1.0e-12,
) -> np.ndarray:
    """Build per-query, classwise shrunk FBCCA-score prototypes.

    The target statistic is a distribution over the complete FBCCA codebook,
    not a query label.  Context, when present, changes only support weights.
    """

    query = _probability_matrix(query_probabilities, "query_probabilities")
    support = _probability_matrix(support_probabilities, "support_probabilities")
    if query.shape[1] != support.shape[1]:
        raise ValueError("Query and support probabilities must share class width.")
    n_query, n_classes = query.shape
    labels = _labels(support_labels, len(support), n_classes=n_classes)
    _require_balanced_labels(labels, n_classes=n_classes)
    smoothing = _finite_float(ideal_prototype_smoothing, "ideal_prototype_smoothing")
    if not 0.0 <= smoothing < 1.0:
        raise ValueError("ideal_prototype_smoothing must be in [0, 1).")
    prior = _positive_finite(prior_pseudocount, "prior_pseudocount")
    eps = _positive_finite(epsilon, "epsilon")
    weights = _affinity_matrix(affinities, n_query, len(support))

    ideal = np.full((n_classes, n_classes), smoothing / n_classes, dtype=np.float64)
    ideal[np.arange(n_classes), np.arange(n_classes)] += 1.0 - smoothing
    prototypes = np.empty((n_query, n_classes, n_classes), dtype=np.float64)
    for query_index in range(n_query):
        for class_index in range(n_classes):
            class_rows = np.flatnonzero(labels == class_index)
            class_weights = weights[query_index, class_rows]
            order = _canonical_weighted_order(support[class_rows], class_weights)
            ordered_rows = support[class_rows][order].astype(np.float64, copy=False)
            ordered_weights = class_weights[order].astype(np.float64, copy=False)
            total_weight = float(ordered_weights.sum(dtype=np.float64))
            weighted_sum = np.sum(ordered_rows * ordered_weights[:, None], axis=0, dtype=np.float64)
            prototypes[query_index, class_index] = (prior * ideal[class_index] + weighted_sum) / (
                prior + total_weight
            )

    query_by_class = np.broadcast_to(query[:, None, :], prototypes.shape)
    midpoint = 0.5 * (query_by_class + prototypes)
    divergence = 0.5 * (
        _rowwise_kl(query_by_class, midpoint, eps) + _rowwise_kl(prototypes, midpoint, eps)
    )
    _, probabilities = normalize_fbcca_scores(-divergence, epsilon=eps)
    return probabilities


def template_residual_scores_from_subbands(
    query_subbands: np.ndarray,
    support_subbands: np.ndarray,
    support_labels: np.ndarray,
    subband_weights: np.ndarray,
    *,
    affinities: np.ndarray | None = None,
) -> np.ndarray:
    """Return weighted signed-square target-template scores.

    Filtering remains the runner's responsibility.  Inputs are shaped
    ``[subband, trial, channel, time]``.  Query-specific context affinity is
    applied only while averaging labeled support waveforms into class
    templates.
    """

    query = _finite_rank4(query_subbands, "query_subbands")
    support = _finite_rank4(support_subbands, "support_subbands")
    if query.shape[0] != support.shape[0] or query.shape[2:] != support.shape[2:]:
        raise ValueError("Query and support subbands must share band/channel/time shape.")
    raw_weights = np.asarray(subband_weights)
    if raw_weights.shape != (query.shape[0],):
        raise ValueError("subband_weights must contain one value per subband.")
    if not np.issubdtype(raw_weights.dtype, np.number):
        raise TypeError("subband_weights must be numeric.")
    band_weights = raw_weights.astype(np.float64, copy=False)
    if (
        not np.isfinite(band_weights).all()
        or np.any(band_weights < 0.0)
        or not np.any(band_weights > 0.0)
    ):
        raise ValueError("subband_weights must be finite, non-negative, and not all zero.")
    labels = _labels(support_labels, support.shape[1])
    n_classes = int(labels.max()) + 1
    _require_balanced_labels(labels, n_classes=n_classes)
    affinity = _affinity_matrix(affinities, query.shape[1], support.shape[1])
    scores = np.zeros((query.shape[1], n_classes), dtype=np.float64)

    for query_index in range(query.shape[1]):
        for class_index in range(n_classes):
            class_rows = np.flatnonzero(labels == class_index)
            class_weights = affinity[query_index, class_rows]
            # Canonical order prevents the input row order becoming a hidden
            # floating-point feature of the support statistic.
            support_view = np.moveaxis(support[:, class_rows], 1, 0)
            order = _canonical_weighted_order(support_view, class_weights)
            ordered_support = support_view[order]
            ordered_weights = class_weights[order]
            denominator = float(ordered_weights.sum(dtype=np.float64))
            template = (
                np.sum(
                    ordered_support * ordered_weights[:, None, None, None],
                    axis=0,
                    dtype=np.float64,
                )
                / denominator
            )
            for band_index, band_weight in enumerate(band_weights):
                correlation = _pearson_flat(query[band_index, query_index], template[band_index])
                scores[query_index, class_index] += (
                    band_weight * np.sign(correlation) * correlation**2
                )
    return scores


def template_residual_distribution(
    template_scores: np.ndarray,
    *,
    epsilon: float = 1.0e-12,
) -> np.ndarray:
    """Normalize precomputed target-template residual scores."""

    _, probabilities = normalize_fbcca_scores(template_scores, epsilon=epsilon)
    return probabilities


def fuse_probabilities(
    base_probabilities: np.ndarray,
    support_probabilities: np.ndarray,
    lambdas: float | np.ndarray,
    *,
    tolerance: float = 1.0e-12,
) -> np.ndarray:
    """Convexly fuse two distributions and enforce the ``2 * lambda`` L1 bound."""

    base = _probability_matrix(base_probabilities, "base_probabilities")
    support = _probability_matrix(support_probabilities, "support_probabilities")
    if base.shape != support.shape:
        raise ValueError("Base and support probability matrices must have identical shape.")
    mix = np.asarray(lambdas)
    if mix.ndim == 0:
        mix = np.full(len(base), float(mix), dtype=np.float64)
    if mix.shape != (len(base),) or not np.issubdtype(mix.dtype, np.number):
        raise ValueError("lambdas must be a scalar or one numeric value per query.")
    mix = mix.astype(np.float64, copy=False)
    if not np.isfinite(mix).all() or np.any(mix < 0.0) or np.any(mix > 1.0):
        raise ValueError("lambdas must be finite and in [0, 1].")
    tol = _positive_finite(tolerance, "tolerance")
    if np.all(mix == 0.0):
        return base
    fused = (1.0 - mix[:, None]) * base + mix[:, None] * support
    _probability_matrix(fused, "fused_probabilities")
    displacement = np.abs(fused - base).sum(axis=1)
    if np.any(displacement > 2.0 * mix + tol):
        raise RuntimeError("Convex fusion exceeded its 2 * lambda L1 bound.")
    zero_rows = mix == 0.0
    if zero_rows.any():
        fused[zero_rows] = base[zero_rows]
    return fused


def prequential_gate_decision(
    base_probabilities: np.ndarray | None,
    support_probabilities: np.ndarray | None,
    labels: np.ndarray | None,
    block_numbers: np.ndarray | None,
    *,
    budget: int,
    fold_provenance: Sequence[PrequentialFoldProvenance] | None = None,
    probability_floor: float = 1.0e-12,
) -> PrequentialGateDecision:
    """Decide a target-local gate from calibration blocks only.

    For k=3, rows must be the prequential predictions for blocks 2 and 3;
    for k=5, blocks 2 through 5.  A caller must have produced each support
    prediction using only earlier blocks.  k=0 and k=1 cannot provide local
    validation and short-circuit without inspecting the optional arrays.
    """

    resolved_budget = _budget(budget)
    if resolved_budget in {0, 1}:
        return PrequentialGateDecision(
            enabled=False,
            budget=resolved_budget,
            evaluated_blocks=(),
            block_balanced_accuracy_deltas=(),
            block_log_probability_deltas=(),
            mean_balanced_accuracy_delta=0.0,
            mean_log_probability_delta=0.0,
            reason="no_target_local_validation_at_k0_or_k1",
            fold_provenance=(),
        )
    if any(
        value is None
        for value in (base_probabilities, support_probabilities, labels, block_numbers)
    ):
        raise ValueError("k=3/5 prequential gating requires probabilities, labels and blocks.")
    base = _probability_matrix(base_probabilities, "base_probabilities")
    candidate = _probability_matrix(support_probabilities, "support_probabilities")
    if base.shape != candidate.shape:
        raise ValueError("Prequential base and support probabilities must align.")
    n_classes = base.shape[1]
    observed_labels = _labels(labels, len(base), n_classes=n_classes)
    blocks_raw = np.asarray(block_numbers)
    if blocks_raw.shape != (len(base),):
        raise ValueError("block_numbers must contain one value per prequential row.")
    if not np.issubdtype(blocks_raw.dtype, np.integer):
        if not np.issubdtype(blocks_raw.dtype, np.number):
            raise TypeError("block_numbers must be exact integers.")
        as_float = blocks_raw.astype(np.float64)
        if not np.isfinite(as_float).all() or not np.equal(as_float, np.floor(as_float)).all():
            raise ValueError("block_numbers must be exact integers.")
    blocks = blocks_raw.astype(np.int64)
    expected_blocks = tuple(range(2, resolved_budget + 1))
    if tuple(sorted(set(blocks.tolist()))) != expected_blocks:
        raise ValueError(
            f"k={resolved_budget} prequential rows must contain exact blocks {expected_blocks}."
        )
    if fold_provenance is None:
        raise ValueError("k=3/5 prequential gating requires per-fold provenance.")
    provenance = tuple(fold_provenance)
    if len(provenance) != len(expected_blocks) or not all(
        isinstance(item, PrequentialFoldProvenance) for item in provenance
    ):
        raise ValueError("fold_provenance must contain one typed receipt per evaluation block.")
    if tuple(item.evaluation_block for item in provenance) != expected_blocks:
        raise ValueError("fold_provenance must be ordered by the exact evaluation blocks.")
    floor = _positive_finite(probability_floor, "probability_floor")
    if floor >= 1.0:
        raise ValueError("probability_floor must be below one.")

    accuracy_deltas: list[float] = []
    log_deltas: list[float] = []
    for block in expected_blocks:
        indices = np.flatnonzero(blocks == block)
        block_labels = observed_labels[indices]
        counts = np.bincount(block_labels, minlength=n_classes)
        if not np.array_equal(counts, np.ones(n_classes, dtype=np.int64)):
            raise ValueError("Every prequential block must contain exactly one row per class.")
        base_prediction = np.argmax(base[indices], axis=1)
        candidate_prediction = np.argmax(candidate[indices], axis=1)
        base_accuracy = float(np.mean(base_prediction == block_labels))
        candidate_accuracy = float(np.mean(candidate_prediction == block_labels))
        accuracy_deltas.append(candidate_accuracy - base_accuracy)
        row_indices = np.arange(len(indices))
        base_correct = np.clip(base[indices][row_indices, block_labels], floor, 1.0)
        candidate_correct = np.clip(candidate[indices][row_indices, block_labels], floor, 1.0)
        log_deltas.append(float(np.mean(np.log(candidate_correct) - np.log(base_correct))))
    mean_accuracy = float(np.mean(accuracy_deltas))
    mean_log = float(np.mean(log_deltas))
    enabled = mean_accuracy >= 0.0 and mean_log >= 0.0
    return PrequentialGateDecision(
        enabled=enabled,
        budget=resolved_budget,
        evaluated_blocks=expected_blocks,
        block_balanced_accuracy_deltas=tuple(accuracy_deltas),
        block_log_probability_deltas=tuple(log_deltas),
        mean_balanced_accuracy_delta=mean_accuracy,
        mean_log_probability_delta=mean_log,
        reason="prequential_means_nonnegative" if enabled else "prequential_gate_abstained",
        fold_provenance=provenance,
    )


def apply_v2_safe_operator(
    query_fbcca_scores: np.ndarray,
    *,
    budget: int,
    operator: V2SupportOperator,
    variant: V2Variant,
    config: V2OperatorConfig | None = None,
    support_fbcca_scores: np.ndarray | None = None,
    support_labels: np.ndarray | None = None,
    base_probabilities: np.ndarray | None = None,
    enabled: bool = True,
    gate_enabled: bool | None = None,
    template_query_class_scores: np.ndarray | None = None,
    template_query_class_probabilities: np.ndarray | None = None,
    template_score_provenance: TemplateScoreProvenance | None = None,
    template_query_subbands: np.ndarray | None = None,
    template_support_subbands: np.ndarray | None = None,
    template_subband_weights: np.ndarray | None = None,
    query_interfaces: Sequence[object] | np.ndarray | None = None,
    support_interfaces: Sequence[object] | np.ndarray | None = None,
    query_impedance_kohm: np.ndarray | None = None,
    support_impedance_kohm: np.ndarray | None = None,
    relative_context_pairing_sha256: str | None = None,
) -> V2OperatorOutput:
    """Apply a frozen V2 support residual without access to query labels.

    The function consumes full strict-FBCCA class-score vectors.  It never
    calls FBCCA itself, which keeps filtering/codebook provenance in the
    governed producer.  Optional template subbands are already filtered; their
    preprocessing, partition binding, and production remain the governed
    runner's responsibility.
    """

    resolved_config = V2OperatorConfig() if config is None else config
    if not isinstance(resolved_config, V2OperatorConfig):
        raise TypeError("config must be a V2OperatorConfig.")
    resolved_budget = _budget(budget)
    if operator not in {
        "score_prototype_shrinkage",
        "filterbank_target_template_residual",
    }:
        raise ValueError(f"Unknown V2 support operator {operator!r}.")
    if variant not in {"A_Q", "A_QM"}:
        raise ValueError("variant must be 'A_Q' or 'A_QM'.")
    context_values = (
        query_interfaces,
        support_interfaces,
        query_impedance_kohm,
        support_impedance_kohm,
    )
    if variant == "A_Q":
        if any(value is not None for value in context_values) or (
            relative_context_pairing_sha256 is not None
        ):
            raise ValueError("A_Q forbids acquisition context and a context pairing capability.")
        pairing_sha256 = None
    else:
        # A_QM capabilities and context are deliberately not parsed on any
        # exact fallback path, including k=0 and a missing gate authorization.
        pairing_sha256 = None
    if type(enabled) is not bool:
        raise TypeError("enabled must be a bool value.")
    if gate_enabled is not None and type(gate_enabled) is not bool:
        raise TypeError("gate_enabled must be bool or None.")

    base_scores = _finite_matrix(query_fbcca_scores, "query_fbcca_scores")
    _, computed_base = normalize_fbcca_scores(base_scores, epsilon=resolved_config.score_epsilon)
    if base_probabilities is None:
        base = computed_base
    else:
        base = _probability_matrix(
            base_probabilities,
            "base_probabilities",
            expected_shape=base_scores.shape,
        )
        if not np.array_equal(base, computed_base):
            raise ValueError(
                "base_probabilities must exactly match the frozen normalized FBCCA formula."
            )

    if resolved_budget == 0:
        return _fallback_output(
            base_scores,
            base,
            operator=operator,
            variant=variant,
            pairing_sha256=pairing_sha256,
            gate_enabled=gate_enabled,
            reason="k0_exact_strict_fbcca",
        )
    if not enabled:
        return _fallback_output(
            base_scores,
            base,
            operator=operator,
            variant=variant,
            pairing_sha256=pairing_sha256,
            gate_enabled=gate_enabled,
            reason="operator_disabled",
        )
    if gate_enabled is None:
        return _fallback_output(
            base_scores,
            base,
            operator=operator,
            variant=variant,
            pairing_sha256=pairing_sha256,
            gate_enabled=None,
            reason="missing_gate_authorization",
        )
    if not gate_enabled:
        return _fallback_output(
            base_scores,
            base,
            operator=operator,
            variant=variant,
            pairing_sha256=pairing_sha256,
            gate_enabled=False,
            reason="validation_gate_abstained",
        )
    if resolved_config.lambda_max == 0.0:
        return _fallback_output(
            base_scores,
            base,
            operator=operator,
            variant=variant,
            pairing_sha256=pairing_sha256,
            gate_enabled=True,
            reason="exact_null_candidate",
        )

    if variant == "A_QM":
        pairing_sha256 = _sha256(
            relative_context_pairing_sha256,
            "relative_context_pairing_sha256",
        )

    if support_fbcca_scores is None or support_labels is None:
        raise ValueError("An enabled k>0 V2 operator requires labeled support FBCCA scores.")
    support_scores = _finite_matrix(support_fbcca_scores, "support_fbcca_scores")
    if support_scores.shape[1] != base_scores.shape[1]:
        raise ValueError("Query and support FBCCA scores must share class width.")
    labels = _labels(
        support_labels,
        len(support_scores),
        n_classes=base_scores.shape[1],
    )
    _require_exact_budget(labels, n_classes=base_scores.shape[1], budget=resolved_budget)
    _, support_base_probabilities = normalize_fbcca_scores(
        support_scores, epsilon=resolved_config.score_epsilon
    )
    if variant == "A_Q":
        affinities = np.ones((len(base_scores), len(support_scores)), dtype=np.float64)
        comparable_pair_count = 0
    else:
        affinities, comparable_pair_count = _relative_context_affinity_and_count(
            n_query=len(base_scores),
            n_support=len(support_scores),
            query_interfaces=query_interfaces,
            support_interfaces=support_interfaces,
            query_impedance_kohm=query_impedance_kohm,
            support_impedance_kohm=support_impedance_kohm,
            different_interface_affinity=resolved_config.different_interface_affinity,
        )

    if operator == "score_prototype_shrinkage":
        if any(
            value is not None
            for value in (
                template_query_class_scores,
                template_query_class_probabilities,
                template_score_provenance,
                template_query_subbands,
                template_support_subbands,
                template_subband_weights,
            )
        ):
            raise ValueError("P1 forbids all P2 template inputs and provenance.")
        support_distribution = score_prototype_distribution(
            base,
            support_base_probabilities,
            labels,
            affinities=affinities,
            ideal_prototype_smoothing=resolved_config.ideal_prototype_smoothing,
            prior_pseudocount=resolved_config.prototype_prior_pseudocount,
            epsilon=resolved_config.score_epsilon,
        )
    else:
        supplied = sum(
            value is not None
            for value in (template_query_class_scores, template_query_class_probabilities)
        )
        subband_values = (
            template_query_subbands,
            template_support_subbands,
            template_subband_weights,
        )
        has_any_subband = any(value is not None for value in subband_values)
        has_all_subband = all(value is not None for value in subband_values)
        if has_any_subband and not has_all_subband:
            raise ValueError("Template subband inputs must be supplied together.")
        if has_all_subband and not np.array_equal(
            np.asarray(template_subband_weights), _P2_SUBBAND_WEIGHTS
        ):
            raise ValueError("High-level P2 requires the exact frozen seven subband weights.")
        if supplied + int(has_all_subband) != 1:
            raise ValueError(
                "Template residual requires exactly one of precomputed scores, "
                "precomputed probabilities, or the complete filtered-subband bundle."
            )
        if supplied and template_score_provenance is None:
            raise ValueError("Precomputed template scores require typed producer provenance.")
        if supplied and not isinstance(template_score_provenance, TemplateScoreProvenance):
            raise TypeError("template_score_provenance must be TemplateScoreProvenance.")
        if not supplied and template_score_provenance is not None:
            raise ValueError("Filtered-subband P2 must not claim precomputed-score provenance.")
        if template_query_class_scores is not None:
            if not np.array_equal(affinities, np.ones_like(affinities)):
                raise ValueError(
                    "Unweighted precomputed template scores cannot be used with non-neutral context."
                )
            support_distribution = template_residual_distribution(
                template_query_class_scores,
                epsilon=resolved_config.score_epsilon,
            )
        elif template_query_class_probabilities is not None:
            if not np.array_equal(affinities, np.ones_like(affinities)):
                raise ValueError(
                    "Unweighted precomputed template probabilities cannot be used with "
                    "non-neutral context."
                )
            support_distribution = _probability_matrix(
                template_query_class_probabilities,
                "template_query_class_probabilities",
                expected_shape=base.shape,
            )
        else:
            template_scores = template_residual_scores_from_subbands(
                template_query_subbands,
                template_support_subbands,
                labels,
                template_subband_weights,
                affinities=affinities,
            )
            support_distribution = template_residual_distribution(
                template_scores,
                epsilon=resolved_config.score_epsilon,
            )
        if support_distribution.shape != base.shape:
            raise ValueError("Template support distribution must align with query FBCCA scores.")

    budget_factor = resolved_budget / (resolved_budget + 1.0)
    lambdas = np.full(len(base), resolved_config.lambda_max * budget_factor, dtype=np.float64)
    if resolved_config.entropy_scaling:
        # High-entropy (uncertain) FBCCA rows receive more of the bounded
        # support residual; confident rows stay close to the immutable anchor.
        lambdas *= normalized_entropy(base)
    # Affinity is a support weight, never a query-score or anchor transform.
    # Its mean classwise mass also bounds the influence of P2 at k=1, where a
    # normalized one-example template average would otherwise cancel its only
    # scalar weight.  All-missing context is an exact matrix of ones and hence
    # remains bitwise A_Q.
    lambdas *= _mean_class_affinity(affinities, labels, base.shape[1])
    fused = fuse_probabilities(
        base,
        support_distribution,
        lambdas,
        tolerance=resolved_config.score_epsilon,
    )
    return V2OperatorOutput(
        base_scores=base_scores,
        base_probabilities=base,
        support_probabilities=support_distribution,
        fused_probabilities=fused,
        predictions=np.argmax(fused, axis=1),
        lambdas=lambdas,
        affinities=affinities,
        gate_enabled=True,
        exact_fallback=False,
        operator=operator,
        variant=variant,
        relative_context_pairing_sha256=pairing_sha256,
        comparable_context_pair_count=comparable_pair_count,
        template_score_provenance=template_score_provenance,
    )


def _fallback_output(
    base_scores: np.ndarray,
    base_probabilities: np.ndarray,
    *,
    operator: V2SupportOperator,
    variant: V2Variant,
    pairing_sha256: str | None,
    gate_enabled: bool | None,
    reason: str,
) -> V2OperatorOutput:
    return V2OperatorOutput(
        base_scores=base_scores,
        base_probabilities=base_probabilities,
        support_probabilities=base_probabilities,
        fused_probabilities=base_probabilities,
        predictions=np.argmax(base_scores, axis=1),
        lambdas=np.zeros(len(base_scores), dtype=np.float64),
        affinities=np.empty((len(base_scores), 0), dtype=np.float64),
        gate_enabled=gate_enabled,
        exact_fallback=True,
        operator=operator,
        variant=variant,
        relative_context_pairing_sha256=pairing_sha256,
        comparable_context_pair_count=0,
        template_score_provenance=None,
        fallback_reason=reason,
    )


def _softmax(values: np.ndarray) -> np.ndarray:
    shifted = values - values.max(axis=1, keepdims=True)
    exponential = np.exp(shifted)
    return exponential / exponential.sum(axis=1, keepdims=True)


def _rowwise_kl(first: np.ndarray, second: np.ndarray, epsilon: float) -> np.ndarray:
    safe_first = np.clip(first, epsilon, 1.0)
    safe_second = np.clip(second, epsilon, 1.0)
    return np.sum(safe_first * (np.log(safe_first) - np.log(safe_second)), axis=-1)


def _pearson_flat(first: np.ndarray, second: np.ndarray) -> float:
    first_flat = np.asarray(first, dtype=np.float64).reshape(-1)
    second_flat = np.asarray(second, dtype=np.float64).reshape(-1)
    first_centered = first_flat - first_flat.mean()
    second_centered = second_flat - second_flat.mean()
    denominator = float(np.linalg.norm(first_centered) * np.linalg.norm(second_centered))
    if denominator <= np.finfo(np.float64).eps:
        return 0.0
    return float(np.clip(first_centered @ second_centered / denominator, -1.0, 1.0))


def _canonical_weighted_order(values: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Canonicalize floating-point reduction order without accepting row IDs."""

    rows = np.asarray(values)
    resolved_weights = np.asarray(weights, dtype=np.float64)
    if len(rows) != len(resolved_weights):
        raise ValueError("Canonical support rows and weights must align.")
    keys = []
    for row, weight in zip(rows, resolved_weights):
        contiguous = np.ascontiguousarray(row)
        payload = (
            contiguous.dtype.str.encode("ascii")
            + np.asarray(contiguous.shape, dtype=np.int64).tobytes()
        )
        payload += contiguous.tobytes() + np.float64(weight).tobytes()
        keys.append(hashlib.sha256(payload).digest())
    return np.asarray(sorted(range(len(keys)), key=keys.__getitem__), dtype=np.int64)


def _finite_matrix(value: np.ndarray, name: str) -> np.ndarray:
    array = np.asarray(value)
    if array.ndim != 2 or array.shape[0] == 0 or array.shape[1] <= 1:
        raise ValueError(f"{name} must be nonempty [rows,classes] with at least two classes.")
    if not np.issubdtype(array.dtype, np.number):
        raise TypeError(f"{name} must be numeric.")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} must be finite.")
    return array


def _finite_rank4(value: np.ndarray, name: str) -> np.ndarray:
    array = np.asarray(value)
    if array.ndim != 4 or any(size <= 0 for size in array.shape):
        raise ValueError(f"{name} must be nonempty [subband,trial,channel,time].")
    if not np.issubdtype(array.dtype, np.number) or not np.isfinite(array).all():
        raise ValueError(f"{name} must contain only finite numeric values.")
    return array.astype(np.float64, copy=False)


def _probability_matrix(
    value: np.ndarray,
    name: str,
    *,
    expected_shape: tuple[int, int] | None = None,
) -> np.ndarray:
    array = np.asarray(value)
    if array.ndim != 2 or array.shape[0] == 0 or array.shape[1] <= 1:
        raise ValueError(f"{name} must be nonempty [rows,classes].")
    if expected_shape is not None and array.shape != expected_shape:
        raise ValueError(f"{name} must have shape {expected_shape}.")
    if not np.issubdtype(array.dtype, np.number):
        raise TypeError(f"{name} must be numeric.")
    if not np.isfinite(array).all() or np.any(array < 0.0) or np.any(array > 1.0):
        raise ValueError(f"{name} must contain finite probabilities in [0, 1].")
    if not np.allclose(array.sum(axis=1), 1.0, rtol=0.0, atol=1.0e-10):
        raise ValueError(f"{name} rows must sum to one.")
    return array


def _labels(
    value: np.ndarray,
    expected_rows: int,
    *,
    n_classes: int | None = None,
) -> np.ndarray:
    raw = np.asarray(value)
    if raw.shape != (expected_rows,):
        raise ValueError("support_labels must contain one value per support row.")
    if not np.issubdtype(raw.dtype, np.number):
        raise TypeError("support_labels must be exact integers.")
    as_float = raw.astype(np.float64)
    if not np.isfinite(as_float).all() or not np.equal(as_float, np.floor(as_float)).all():
        raise ValueError("support_labels must be exact integers.")
    labels = raw.astype(np.int64)
    if len(labels) == 0 or labels.min() < 0:
        raise ValueError("support_labels must be nonempty and non-negative.")
    if n_classes is not None and labels.max() >= n_classes:
        raise ValueError("support_labels fall outside the class vocabulary.")
    return labels


def _require_balanced_labels(labels: np.ndarray, *, n_classes: int) -> None:
    counts = np.bincount(labels, minlength=n_classes)
    if len(counts) != n_classes or counts[0] <= 0 or not np.all(counts == counts[0]):
        raise ValueError("Support must contain the same positive number of rows per class.")


def _require_exact_budget(labels: np.ndarray, *, n_classes: int, budget: int) -> None:
    counts = np.bincount(labels, minlength=n_classes)
    if len(counts) != n_classes or not np.array_equal(
        counts, np.full(n_classes, budget, dtype=np.int64)
    ):
        raise ValueError(f"k={budget} requires exactly {budget} support row(s) per class.")


def _affinity_matrix(value: np.ndarray | None, n_query: int, n_support: int) -> np.ndarray:
    if value is None:
        return np.ones((n_query, n_support), dtype=np.float64)
    array = np.asarray(value)
    if array.shape != (n_query, n_support) or not np.issubdtype(array.dtype, np.number):
        raise ValueError("affinities must have shape [query,support].")
    result = array.astype(np.float64, copy=False)
    if not np.isfinite(result).all() or np.any(result <= 0.0) or np.any(result > 1.0):
        raise ValueError("affinities must be finite in (0, 1].")
    return result


def _mean_class_affinity(
    affinities: np.ndarray,
    labels: np.ndarray,
    n_classes: int,
) -> np.ndarray:
    class_means = np.stack(
        [affinities[:, labels == class_index].mean(axis=1) for class_index in range(n_classes)],
        axis=1,
    )
    result = class_means.mean(axis=1)
    if not np.isfinite(result).all() or np.any(result <= 0.0) or np.any(result > 1.0):
        raise RuntimeError("Mean class affinity must remain in (0, 1].")
    return result


def _optional_interfaces(
    value: Sequence[object] | np.ndarray | None,
    expected_rows: int,
    name: str,
) -> np.ndarray | None:
    if value is None:
        return None
    array = np.asarray(value, dtype=object)
    if array.shape != (expected_rows,):
        raise ValueError(f"{name} must contain one value per row.")
    return array


def _normalized_interface(value: object) -> str | None:
    if value is None:
        return None
    if isinstance(value, Real) and not isinstance(value, bool) and not np.isfinite(float(value)):
        return None
    normalized = str(value).strip().casefold()
    if normalized in {"", "n/a", "na", "nan", "none", "null", "unknown"}:
        return None
    return normalized


def _optional_impedance(
    value: np.ndarray | None,
    expected_rows: int,
    name: str,
) -> np.ndarray | None:
    if value is None:
        return None
    raw = np.asarray(value)
    if raw.ndim != 2 or raw.shape[0] != expected_rows or raw.shape[1] == 0:
        raise ValueError(f"{name} must have shape [rows,channels].")
    if not np.issubdtype(raw.dtype, np.number):
        raise TypeError(f"{name} must be numeric with NaN for missing values.")
    array = raw.astype(np.float64, copy=False)
    if np.isinf(array).any() or np.any(array[np.isfinite(array)] < 0.0):
        raise ValueError(f"{name} must contain non-negative values or NaN.")
    return array


def _budget(value: int) -> int:
    if not isinstance(value, Integral) or isinstance(value, bool):
        raise TypeError("budget must be an exact integer.")
    resolved = int(value)
    if resolved not in _BUDGETS:
        raise ValueError(f"budget must be one of {_BUDGETS}.")
    return resolved


def _nonnegative_integer(value: int, name: str) -> int:
    if not isinstance(value, Integral) or isinstance(value, bool) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer.")
    return int(value)


def _finite_float(value: float, name: str) -> float:
    if not isinstance(value, Real) or isinstance(value, bool):
        raise TypeError(f"{name} must be a finite number.")
    result = float(value)
    if not np.isfinite(result):
        raise ValueError(f"{name} must be finite.")
    return result


def _positive_finite(value: float, name: str) -> float:
    result = _finite_float(value, name)
    if result <= 0.0:
        raise ValueError(f"{name} must be positive.")
    return result


def _positive_integer(value: int, name: str) -> int:
    if not isinstance(value, Integral) or isinstance(value, bool):
        raise TypeError(f"{name} must be an exact integer.")
    resolved = int(value)
    if resolved <= 0:
        raise ValueError(f"{name} must be positive.")
    return resolved


def _sha256(value: str | None, name: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a lowercase 64-character SHA-256 hex digest.")
    if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
        raise ValueError(f"{name} must be a lowercase 64-character SHA-256 hex digest.")
    return value
