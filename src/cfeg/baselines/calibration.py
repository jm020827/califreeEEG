from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.linalg import eigh

from cfeg.baselines.fbcca import apply_filterbank


@dataclass(frozen=True)
class CalibrationBaselineOutput:
    prediction: np.ndarray
    scores: np.ndarray
    probabilities: np.ndarray
    score_schema: str


def predict_supervised_template_correlation(
    query: np.ndarray,
    support: np.ndarray,
    support_labels: np.ndarray,
    *,
    n_classes: int,
    probability_temperature: float = 1.0,
) -> CalibrationBaselineOutput:
    """Classify with participant-specific multichannel mean templates."""

    query, support, labels = _validate_inputs(
        query, support, support_labels, n_classes=n_classes, minimum_per_class=1
    )
    templates = np.stack([support[labels == label].mean(axis=0) for label in range(n_classes)])
    scores = _rowwise_template_correlation(query, templates)
    return _output(
        scores,
        probability_temperature=probability_temperature,
        schema="multichannel_mean_template_pearson_v1",
    )


def predict_ensemble_trca(
    query: np.ndarray,
    support: np.ndarray,
    support_labels: np.ndarray,
    *,
    n_classes: int,
    ridge_ratio: float = 1e-6,
    probability_temperature: float = 1.0,
) -> CalibrationBaselineOutput:
    """Fit standard ensemble TRCA filters and return one full score vector per query.

    At least two trials per class are required; k=1 is deliberately rejected
    rather than silently inventing a single-trial TRCA variant.
    """

    if ridge_ratio <= 0.0:
        raise ValueError("ridge_ratio must be positive.")
    query, support, labels = _validate_inputs(
        query, support, support_labels, n_classes=n_classes, minimum_per_class=2
    )
    filters = []
    templates = []
    for label in range(n_classes):
        trials = support[labels == label]
        centered = trials - trials.mean(axis=-1, keepdims=True)
        within = np.einsum("nct,ndt->cd", centered, centered)
        summed = centered.sum(axis=0)
        between = summed @ summed.T - within
        within = 0.5 * (within + within.T)
        between = 0.5 * (between + between.T)
        scale = float(np.trace(within)) / within.shape[0]
        regularized = within + max(scale, 1.0) * ridge_ratio * np.eye(within.shape[0])
        eigenvalues, eigenvectors = eigh(between, regularized, check_finite=True)
        spatial_filter = eigenvectors[:, int(np.argmax(eigenvalues))]
        filters.append(spatial_filter)
        templates.append(trials.mean(axis=0))
    ensemble = np.stack(filters, axis=1)
    templates_array = np.stack(templates)
    projected_query = np.einsum("ck,bct->bkt", ensemble, query)
    projected_templates = np.einsum("ck,lct->lkt", ensemble, templates_array)
    scores = _flattened_pairwise_correlation(projected_query, projected_templates)
    return _output(
        scores,
        probability_temperature=probability_temperature,
        schema="ensemble_trca_generalized_eigen_pearson_v1",
    )


def predict_filterbank_ensemble_trca(
    query: np.ndarray,
    support: np.ndarray,
    support_labels: np.ndarray,
    *,
    n_classes: int,
    sfreq: float,
    filterbank: object,
    ridge_ratio: float = 1e-6,
) -> CalibrationBaselineOutput:
    """Run target-only ensemble TRCA with one shared frozen filter bank."""

    query, support, labels = _validate_inputs(
        query, support, support_labels, n_classes=n_classes, minimum_per_class=2
    )
    filtered_query, parameters = apply_filterbank(
        query, sfreq=sfreq, filterbank=filterbank
    )
    filtered_support, support_parameters = apply_filterbank(
        support, sfreq=sfreq, filterbank=filterbank
    )
    if parameters != support_parameters:
        raise RuntimeError("Query and support filter-bank parameters drifted.")
    weights = np.asarray(parameters["weights"], dtype=np.float64)
    subband_scores = []
    for band_index in range(len(weights)):
        output = predict_ensemble_trca(
            filtered_query[band_index],
            filtered_support[band_index],
            labels,
            n_classes=n_classes,
            ridge_ratio=ridge_ratio,
        )
        subband_scores.append(output.scores)
    scores = _weighted_signed_square_scores(np.stack(subband_scores), weights)
    return _output(
        scores,
        probability_temperature=1.0,
        schema="filterbank_ensemble_trca_weighted_signed_square_v1",
    )


def augment_same_trials(
    support: np.ndarray,
    support_labels: np.ndarray,
    *,
    frequencies_hz: np.ndarray,
    phases_rad: np.ndarray,
    sfreq: float,
    n_classes: int,
    n_harmonics: int = 5,
    n_augmentations: int = 3,
    noise_scale: float = 0.05,
    ridge_ratio: float = 1e-6,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Create deterministic SAME-style harmonic reconstructions at k=1.

    This is an independently implemented, common-protocol component comparator.
    It is not a reproduction of the full CSDuDoFN/OS-SSVEP pipeline.
    """

    if not np.isfinite(sfreq) or sfreq <= 0.0:
        raise ValueError("SAME sampling rate must be positive and finite.")
    if n_harmonics <= 0 or n_augmentations <= 0:
        raise ValueError("SAME harmonic and augmentation counts must be positive.")
    if not np.isfinite(noise_scale) or noise_scale < 0.0 or ridge_ratio <= 0.0:
        raise ValueError("SAME noise and ridge parameters are invalid.")
    _, support, labels = _validate_inputs(
        support,
        support,
        support_labels,
        n_classes=n_classes,
        minimum_per_class=1,
    )
    if not np.all(np.bincount(labels, minlength=n_classes) == 1):
        raise ValueError("The frozen SAME comparator requires exactly one real trial per class.")
    frequencies = np.asarray(frequencies_hz, dtype=np.float64)
    phases = np.asarray(phases_rad, dtype=np.float64)
    if frequencies.shape != (n_classes,) or phases.shape != (n_classes,):
        raise ValueError("SAME requires one frozen frequency and phase per class.")
    if not np.isfinite(frequencies).all() or not np.isfinite(phases).all():
        raise ValueError("SAME frequency/phase codebook must be finite.")

    rng = np.random.default_rng(int(seed))
    time = np.arange(support.shape[-1], dtype=np.float64) / float(sfreq)
    augmented: list[np.ndarray] = []
    augmented_labels: list[int] = []
    for label in range(n_classes):
        trial = support[labels == label][0]
        reference_rows: list[np.ndarray] = []
        for harmonic in range(1, n_harmonics + 1):
            angle = (
                2.0 * np.pi * harmonic * frequencies[label] * time
                + harmonic * phases[label]
            )
            reference_rows.extend((np.sin(angle), np.cos(angle)))
        reference = np.stack(reference_rows)
        gram = reference @ reference.T
        scale = max(float(np.trace(gram)) / len(gram), np.finfo(np.float64).eps)
        reconstruction = (
            trial
            @ reference.T
            @ np.linalg.pinv(
                gram + ridge_ratio * scale * np.eye(len(gram)),
                hermitian=True,
            )
            @ reference
        )
        channel_variance = np.var(reconstruction, axis=-1, ddof=1)
        channel_scale = np.sqrt(np.maximum(channel_variance, 0.0))[:, None]
        augmented.append(trial)
        augmented_labels.append(label)
        for _ in range(n_augmentations):
            noise = rng.standard_normal(reconstruction.shape) * channel_scale
            augmented.append(reconstruction + noise_scale * noise)
            augmented_labels.append(label)
    augmented_array = np.stack(augmented)
    labels_array = np.asarray(augmented_labels, dtype=np.int64)
    if not np.isfinite(augmented_array).all():
        raise ValueError("SAME augmentation produced non-finite EEG.")
    return augmented_array, labels_array


def predict_same_filterbank_ensemble_trca(
    query: np.ndarray,
    target_support: np.ndarray,
    target_labels: np.ndarray,
    *,
    frequencies_hz: np.ndarray,
    phases_rad: np.ndarray,
    n_classes: int,
    sfreq: float,
    filterbank: object,
    seed: int,
    trca_ridge_ratio: float = 1e-6,
) -> CalibrationBaselineOutput:
    """Run SAME(3)+filter-bank eTRCA as an honest one-shot component baseline."""

    augmented, augmented_labels = augment_same_trials(
        target_support,
        target_labels,
        frequencies_hz=frequencies_hz,
        phases_rad=phases_rad,
        sfreq=sfreq,
        n_classes=n_classes,
        seed=seed,
    )
    output = predict_filterbank_ensemble_trca(
        query,
        augmented,
        augmented_labels,
        n_classes=n_classes,
        sfreq=sfreq,
        filterbank=filterbank,
        ridge_ratio=trca_ridge_ratio,
    )
    return CalibrationBaselineOutput(
        prediction=output.prediction,
        scores=output.scores,
        probabilities=output.probabilities,
        score_schema="same3_filterbank_ensemble_trca_common_protocol_v1",
    )


def least_squares_transfer_trials(
    source: np.ndarray,
    source_labels: np.ndarray,
    target_support: np.ndarray,
    target_labels: np.ndarray,
    *,
    n_classes: int,
    ridge_ratio: float = 1e-6,
    chunk_size: int = 512,
) -> np.ndarray:
    """Map each source trial to its target-class mean by the LST paper equation."""

    if ridge_ratio <= 0.0:
        raise ValueError("LST ridge_ratio must be positive.")
    source, _, source_labels = _validate_inputs(
        source,
        source,
        source_labels,
        n_classes=n_classes,
        minimum_per_class=1,
    )
    _, target_support, target_labels = _validate_inputs(
        target_support,
        target_support,
        target_labels,
        n_classes=n_classes,
        minimum_per_class=1,
    )
    if source.shape[1:] != target_support.shape[1:]:
        raise ValueError("LST source and target support must share channel/time shape.")
    target_templates = np.stack(
        [target_support[target_labels == label].mean(axis=0) for label in range(n_classes)]
    )
    if type(chunk_size) is not int or chunk_size <= 0:
        raise ValueError("LST chunk_size must be a positive integer.")
    target_centered = target_templates - target_templates.mean(axis=-1, keepdims=True)
    transformed = np.empty_like(source)
    identity = np.eye(source.shape[1], dtype=np.float64)
    for start in range(0, len(source), chunk_size):
        stop = min(start + chunk_size, len(source))
        source_centered = source[start:stop] - source[start:stop].mean(
            axis=-1, keepdims=True
        )
        targets = target_centered[source_labels[start:stop]]
        gram = np.einsum("nct,ndt->ncd", source_centered, source_centered)
        scale = np.maximum(
            np.trace(gram, axis1=1, axis2=2) / gram.shape[1],
            np.finfo(np.float64).eps,
        )
        regularized = gram + ridge_ratio * scale[:, None, None] * identity
        cross = np.einsum("nct,ndt->ncd", targets, source_centered)
        transform = np.einsum(
            "ncd,nde->nce",
            cross,
            np.linalg.pinv(regularized, hermitian=True),
        )
        transformed[start:stop] = np.einsum(
            "ncd,ndt->nct",
            transform,
            source_centered,
        )
    if not np.isfinite(transformed).all():
        raise ValueError("LST produced non-finite transformed source trials.")
    return transformed


def predict_lst_filterbank_ensemble_trca(
    query: np.ndarray,
    target_support: np.ndarray,
    target_labels: np.ndarray,
    source: np.ndarray,
    source_labels: np.ndarray,
    *,
    n_classes: int,
    sfreq: float,
    filterbank: object,
    lst_ridge_ratio: float = 1e-6,
    trca_ridge_ratio: float = 1e-6,
    filtered_source_cache: tuple[np.ndarray, dict[str, object]] | None = None,
) -> CalibrationBaselineOutput:
    """Protocol-adapted Chiang-2021 LST followed by filter-bank ensemble TRCA."""

    query, target_support, target_labels = _validate_inputs(
        query,
        target_support,
        target_labels,
        n_classes=n_classes,
        minimum_per_class=1,
    )
    source, _, source_labels = _validate_inputs(
        source,
        source,
        source_labels,
        n_classes=n_classes,
        minimum_per_class=1,
    )
    if source.shape[1:] != query.shape[1:]:
        raise ValueError("LST source, target support and query shapes must align.")
    filtered_query, parameters = apply_filterbank(
        query, sfreq=sfreq, filterbank=filterbank
    )
    filtered_target, target_parameters = apply_filterbank(
        target_support, sfreq=sfreq, filterbank=filterbank
    )
    if filtered_source_cache is None:
        filtered_source, source_parameters = apply_filterbank(
            source, sfreq=sfreq, filterbank=filterbank
        )
    else:
        filtered_source, source_parameters = filtered_source_cache
        filtered_source = np.asarray(filtered_source, dtype=np.float64)
        if filtered_source.ndim != 4 or filtered_source.shape[1:] != source.shape:
            raise ValueError("Cached LST source filter bank has an invalid shape.")
    if not (parameters == target_parameters == source_parameters):
        raise RuntimeError("LST query/support/source filter-bank parameters drifted.")
    weights = np.asarray(parameters["weights"], dtype=np.float64)
    subband_scores = []
    source_labels = source_labels.astype(np.int64)
    for band_index in range(len(weights)):
        transferred = least_squares_transfer_trials(
            filtered_source[band_index],
            source_labels,
            filtered_target[band_index],
            target_labels,
            n_classes=n_classes,
            ridge_ratio=lst_ridge_ratio,
        )
        augmented = np.concatenate([filtered_target[band_index], transferred], axis=0)
        augmented_labels = np.concatenate([target_labels, source_labels], axis=0)
        output = predict_ensemble_trca(
            filtered_query[band_index],
            augmented,
            augmented_labels,
            n_classes=n_classes,
            ridge_ratio=trca_ridge_ratio,
        )
        subband_scores.append(output.scores)
    scores = _weighted_signed_square_scores(np.stack(subband_scores), weights)
    return _output(
        scores,
        probability_temperature=1.0,
        schema="chiang2021_lst_filterbank_ensemble_trca_cleanroom_v1",
    )


def _validate_inputs(
    query: np.ndarray,
    support: np.ndarray,
    support_labels: np.ndarray,
    *,
    n_classes: int,
    minimum_per_class: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    query = np.asarray(query, dtype=np.float64)
    support = np.asarray(support, dtype=np.float64)
    labels_raw = np.asarray(support_labels)
    if query.ndim != 3 or support.ndim != 3 or query.shape[1:] != support.shape[1:]:
        raise ValueError("Query and support must have aligned [trial,channel,time] shapes.")
    if query.shape[0] == 0 or support.shape[0] == 0 or query.shape[1] == 0 or query.shape[2] < 2:
        raise ValueError("Calibration baseline inputs must be nonempty.")
    if not np.isfinite(query).all() or not np.isfinite(support).all():
        raise ValueError("Calibration baseline EEG must be finite.")
    if labels_raw.ndim != 1 or len(labels_raw) != len(support):
        raise ValueError("support_labels must contain one value per support trial.")
    if not np.issubdtype(labels_raw.dtype, np.integer) and (
        not np.isfinite(labels_raw.astype(float)).all()
        or not np.equal(
            labels_raw.astype(float), np.floor(labels_raw.astype(float))
        ).all()
    ):
        raise ValueError("support_labels must be exact integers.")
    labels = labels_raw.astype(np.int64)
    if n_classes <= 1 or labels.min() < 0 or labels.max() >= n_classes:
        raise ValueError("Class vocabulary or support label is invalid.")
    counts = np.bincount(labels, minlength=n_classes)
    if not np.all(counts == counts[0]) or counts[0] < minimum_per_class:
        raise ValueError(
            f"Every class must have the same count and at least {minimum_per_class} trial(s)."
        )
    return query, support, labels


def _rowwise_template_correlation(query: np.ndarray, templates: np.ndarray) -> np.ndarray:
    return _flattened_pairwise_correlation(
        query.reshape(query.shape[0], 1, -1),
        templates.reshape(templates.shape[0], 1, -1),
    )


def _flattened_pairwise_correlation(query: np.ndarray, templates: np.ndarray) -> np.ndarray:
    query_flat = query.reshape(query.shape[0], -1)
    template_flat = templates.reshape(templates.shape[0], -1)
    query_centered = query_flat - query_flat.mean(axis=1, keepdims=True)
    template_centered = template_flat - template_flat.mean(axis=1, keepdims=True)
    numerator = query_centered @ template_centered.T
    denominator = np.linalg.norm(query_centered, axis=1, keepdims=True) * np.linalg.norm(
        template_centered, axis=1, keepdims=True
    ).T
    return np.divide(
        numerator,
        denominator,
        out=np.zeros_like(numerator),
        where=denominator > np.finfo(np.float64).eps,
    )


def _weighted_signed_square_scores(
    subband_scores: np.ndarray, weights: np.ndarray
) -> np.ndarray:
    if subband_scores.ndim != 3 or weights.shape != (subband_scores.shape[0],):
        raise ValueError("Subband scores and weights have incompatible shapes.")
    if not np.isfinite(subband_scores).all() or not np.isfinite(weights).all():
        raise ValueError("Subband scores and weights must be finite.")
    return np.einsum(
        "f,fqc->qc",
        weights,
        np.sign(subband_scores) * np.square(subband_scores),
    )


def _output(
    scores: np.ndarray,
    *,
    probability_temperature: float,
    schema: str,
) -> CalibrationBaselineOutput:
    if probability_temperature <= 0.0:
        raise ValueError("probability_temperature must be positive.")
    scaled = scores / probability_temperature
    scaled -= scaled.max(axis=1, keepdims=True)
    exponential = np.exp(scaled)
    probabilities = exponential / exponential.sum(axis=1, keepdims=True)
    return CalibrationBaselineOutput(
        prediction=np.argmax(scores, axis=1),
        scores=scores,
        probabilities=probabilities,
        score_schema=schema,
    )
