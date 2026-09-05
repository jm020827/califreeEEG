from __future__ import annotations

import hashlib
import inspect
import json
from pathlib import Path

import numpy as np
import pytest

from cfeg.baselines.calibration import (
    augment_same_trials,
    least_squares_transfer_trials,
    predict_ensemble_trca,
    predict_filterbank_ensemble_trca,
    predict_lst_filterbank_ensemble_trca,
    predict_same_filterbank_ensemble_trca,
    predict_supervised_template_correlation,
)
from cfeg.baselines.metadata_calibration_adapter import (
    BaselineSignalBatch,
    GovernedCalibrationBaselineAdapter,
)

FILTERBANK = {
    "bands": [[6.0, 35.0], [10.0, 45.0]],
    "order": 2,
    "weights": [1.0, 0.5],
    "filter_family": "butterworth",
    "n_harmonics": 3,
    "regularization": 1e-8,
    "reproduction_contract": "unit_test",
}


def _bound_filterbank(tmp_path: Path) -> tuple[Path, str]:
    path = tmp_path / "filterbank.json"
    path.write_text(json.dumps(FILTERBANK, sort_keys=True), encoding="utf-8")
    return path, hashlib.sha256(path.read_bytes()).hexdigest()


def _signals(k: int = 3):
    rng = np.random.default_rng(41)
    sfreq, time = 200.0, 400
    timeline = np.arange(time) / sfreq
    frequencies = (10.0, 12.0, 15.0)
    spatial = np.asarray(
        [[1.0, 0.5, 0.2], [0.4, 1.0, 0.3], [0.2, 0.3, 1.0], [0.6, 0.2, 0.5]]
    )
    support = []
    labels = []
    query = []
    for label, frequency in enumerate(frequencies):
        source = np.sin(2 * np.pi * frequency * timeline)
        harmonic = 0.3 * np.sin(2 * np.pi * 2 * frequency * timeline + 0.2)
        template = spatial[:, label, None] * (source + harmonic)
        for _ in range(k):
            support.append(template + 0.10 * rng.standard_normal(template.shape))
            labels.append(label)
        query.append(template + 0.10 * rng.standard_normal(template.shape))
    return np.asarray(query), np.asarray(support), np.asarray(labels)


def test_template_and_ensemble_trca_return_correct_full_vectors() -> None:
    query, support, labels = _signals(k=3)
    template = predict_supervised_template_correlation(
        query, support, labels, n_classes=3
    )
    trca = predict_ensemble_trca(query, support, labels, n_classes=3)
    for output in (template, trca):
        assert np.array_equal(output.prediction, np.arange(3))
        assert output.scores.shape == output.probabilities.shape == (3, 3)
        assert np.allclose(output.probabilities.sum(axis=1), 1.0)
        assert np.isfinite(output.probabilities).all()


def test_one_shot_template_is_defined_but_single_trial_trca_is_rejected() -> None:
    query, support, labels = _signals(k=1)
    output = predict_supervised_template_correlation(query, support, labels, n_classes=3)
    assert np.array_equal(output.prediction, np.arange(3))
    with pytest.raises(ValueError, match="at least 2"):
        predict_ensemble_trca(query, support, labels, n_classes=3)


def test_support_order_is_irrelevant_and_no_metadata_api_exists() -> None:
    query, support, labels = _signals(k=3)
    rng = np.random.default_rng(8)
    order = rng.permutation(len(support))
    first = predict_ensemble_trca(query, support, labels, n_classes=3)
    second = predict_ensemble_trca(query, support[order], labels[order], n_classes=3)
    assert np.allclose(first.scores, second.scores, rtol=1e-10, atol=1e-12)
    for predictor in (predict_supervised_template_correlation, predict_ensemble_trca):
        parameters = set(inspect.signature(predictor).parameters)
        assert not parameters.intersection(
            {"metadata", "electrode_type", "impedance", "subject_id", "sample_id"}
        )


def test_baselines_reject_unbalanced_support_classes() -> None:
    query, support, labels = _signals(k=3)
    with pytest.raises(ValueError, match="same count"):
        predict_supervised_template_correlation(
            query, support[:-1], labels[:-1], n_classes=3
        )


def test_filterbank_etrac_uses_same_decoder_and_rejects_k1() -> None:
    query, support, labels = _signals(k=3)
    output = predict_filterbank_ensemble_trca(
        query,
        support,
        labels,
        n_classes=3,
        sfreq=200.0,
        filterbank=FILTERBANK,
    )
    assert np.array_equal(output.prediction, np.arange(3))
    assert output.scores.shape == (3, 3)
    one_query, one_support, one_labels = _signals(k=1)
    with pytest.raises(ValueError, match="at least 2"):
        predict_filterbank_ensemble_trca(
            one_query,
            one_support,
            one_labels,
            n_classes=3,
            sfreq=200.0,
            filterbank=FILTERBANK,
        )


def test_same3_makes_one_shot_etrac_defined_and_deterministic() -> None:
    query, support, labels = _signals(k=1)
    frequencies = np.asarray([10.0, 12.0, 15.0])
    phases = np.asarray([0.0, 0.5, 1.0])
    augmented, augmented_labels = augment_same_trials(
        support,
        labels,
        frequencies_hz=frequencies,
        phases_rad=phases,
        sfreq=200.0,
        n_classes=3,
        seed=77,
    )
    repeated, repeated_labels = augment_same_trials(
        support,
        labels,
        frequencies_hz=frequencies,
        phases_rad=phases,
        sfreq=200.0,
        n_classes=3,
        seed=77,
    )
    assert augmented.shape == (12, 4, 400)
    assert np.array_equal(np.bincount(augmented_labels), np.full(3, 4))
    assert np.array_equal(augmented, repeated)
    assert np.array_equal(augmented_labels, repeated_labels)
    output = predict_same_filterbank_ensemble_trca(
        query,
        support,
        labels,
        frequencies_hz=frequencies,
        phases_rad=phases,
        n_classes=3,
        sfreq=200.0,
        filterbank=FILTERBANK,
        seed=77,
    )
    assert output.scores.shape == (3, 3)
    assert np.isfinite(output.scores).all()
    assert output.score_schema.startswith("same3_")


def test_lst_identity_rank_deficiency_and_protocol_adapted_k1() -> None:
    query, target_support, target_labels = _signals(k=1)
    _, source, source_labels = _signals(k=3)
    mixing = np.asarray(
        [[1.0, 0.2, 0.0, 0.1], [0.1, 0.9, 0.2, 0.0], [0.0, 0.2, 1.1, 0.1], [0.2, 0.0, 0.1, 0.8]]
    )
    shifted_source = np.einsum("cd,ndt->nct", mixing, source)
    transferred = least_squares_transfer_trials(
        shifted_source,
        source_labels,
        target_support,
        target_labels,
        n_classes=3,
    )
    assert transferred.shape == shifted_source.shape
    assert np.isfinite(transferred).all()

    rank_deficient = shifted_source.copy()
    rank_deficient[:, 1] = rank_deficient[:, 0]
    stable = least_squares_transfer_trials(
        rank_deficient,
        source_labels,
        target_support,
        target_labels,
        n_classes=3,
    )
    assert np.isfinite(stable).all()

    output = predict_lst_filterbank_ensemble_trca(
        query,
        target_support,
        target_labels,
        shifted_source,
        source_labels,
        n_classes=3,
        sfreq=200.0,
        filterbank=FILTERBANK,
    )
    assert output.score_schema.startswith("chiang2021_lst")
    assert output.scores.shape == (3, 3)
    assert np.isfinite(output.scores).all()


def test_vectorized_lst_matches_frozen_scalar_equation() -> None:
    _, target_support, target_labels = _signals(k=1)
    _, source, source_labels = _signals(k=3)
    observed = least_squares_transfer_trials(
        source,
        source_labels,
        target_support,
        target_labels,
        n_classes=3,
        chunk_size=4,
    )
    templates = np.stack(
        [target_support[target_labels == label].mean(axis=0) for label in range(3)]
    )
    expected = np.empty_like(source)
    for index, (trial, label) in enumerate(zip(source, source_labels)):
        source_centered = trial - trial.mean(axis=-1, keepdims=True)
        target_centered = templates[label] - templates[label].mean(
            axis=-1, keepdims=True
        )
        gram = source_centered @ source_centered.T
        scale = max(float(np.trace(gram)) / gram.shape[0], np.finfo(float).eps)
        regularized = gram + 1e-6 * scale * np.eye(gram.shape[0])
        transform = target_centered @ source_centered.T @ np.linalg.pinv(
            regularized,
            hermitian=True,
        )
        expected[index] = transform @ source_centered
    assert np.allclose(observed, expected, rtol=1e-10, atol=1e-11)


def test_lst_source_and_support_order_are_irrelevant_and_api_has_no_metadata() -> None:
    query, target_support, target_labels = _signals(k=1)
    _, source, source_labels = _signals(k=3)
    first = predict_lst_filterbank_ensemble_trca(
        query,
        target_support,
        target_labels,
        source,
        source_labels,
        n_classes=3,
        sfreq=200.0,
        filterbank=FILTERBANK,
    )
    source_order = np.random.default_rng(9).permutation(len(source))
    support_order = np.random.default_rng(10).permutation(len(target_support))
    second = predict_lst_filterbank_ensemble_trca(
        query,
        target_support[support_order],
        target_labels[support_order],
        source[source_order],
        source_labels[source_order],
        n_classes=3,
        sfreq=200.0,
        filterbank=FILTERBANK,
    )
    assert np.allclose(first.scores, second.scores, rtol=1e-9, atol=1e-11)
    for predictor in (
        predict_filterbank_ensemble_trca,
        predict_lst_filterbank_ensemble_trca,
    ):
        parameters = set(inspect.signature(predictor).parameters)
        assert not parameters.intersection(
            {"metadata", "electrode_type", "impedance", "subject_id", "sample_id"}
        )


def test_atomic_baseline_adapter_enforces_information_rights_and_budgets(
    tmp_path: Path,
) -> None:
    query, support, labels = _signals(k=3)
    filterbank_path, bound_hash = _bound_filterbank(tmp_path)
    fbcca = GovernedCalibrationBaselineAdapter(
        "strict_FBCCA",
        n_classes=3,
        filterbank=FILTERBANK,
        filterbank_config_path=filterbank_path,
        filterbank_config_sha256=bound_hash,
    )
    fbcca_batch = BaselineSignalBatch(
        query_x=query,
        target_support_x=None,
        target_support_y=None,
        source_x=None,
        source_y=None,
        frequencies_hz=np.asarray([10.0, 12.0, 15.0]),
        sfreq=200.0,
    )
    first = fbcca.predict(fbcca_batch, budget=0)
    second = fbcca.predict(fbcca_batch, budget=0)
    assert np.array_equal(first.scores, second.scores)
    assert first.score_interpretation == "uncalibrated_raw_class_score_argmax_only"
    with pytest.raises(ValueError, match="not applicable"):
        fbcca.predict(fbcca_batch, budget=1)
    with pytest.raises(ValueError, match="must not receive labeled EEG"):
        fbcca.predict(
            BaselineSignalBatch(
                **{**fbcca_batch.__dict__, "target_support_x": support, "target_support_y": labels}
            ),
            budget=0,
        )

    with pytest.raises(ValueError, match="frozen SHA-256"):
        GovernedCalibrationBaselineAdapter(
            "strict_FBCCA",
            n_classes=3,
            filterbank=FILTERBANK,
            filterbank_config_path=filterbank_path,
            filterbank_config_sha256="a" * 64,
        )


def test_atomic_lst_adapter_is_query_batch_independent(tmp_path: Path) -> None:
    query, support, labels = _signals(k=3)
    _, target_one, target_labels = _signals(k=1)
    filterbank_path, bound_hash = _bound_filterbank(tmp_path)
    adapter = GovernedCalibrationBaselineAdapter(
        "chiang2021_LST_filterbank_eTRCA",
        n_classes=3,
        filterbank=FILTERBANK,
        filterbank_config_path=filterbank_path,
        filterbank_config_sha256=bound_hash,
    )

    def batch(values: np.ndarray) -> BaselineSignalBatch:
        return BaselineSignalBatch(
            query_x=values,
            target_support_x=target_one,
            target_support_y=target_labels,
            source_x=support,
            source_y=labels,
            frequencies_hz=None,
            sfreq=200.0,
        )

    whole = adapter.predict(batch(query), budget=1).scores
    split = np.concatenate(
        [adapter.predict(batch(query[index : index + 1]), budget=1).scores for index in range(3)]
    )
    order = np.asarray([2, 0, 1])
    permuted = adapter.predict(batch(query[order]), budget=1).scores
    assert np.allclose(whole, split, rtol=1e-9, atol=1e-11)
    assert np.allclose(permuted, whole[order], rtol=1e-9, atol=1e-11)
    assert not set(BaselineSignalBatch.__dataclass_fields__).intersection(
        {"metadata", "impedance", "electrode_type", "subject_id", "sample_id", "query_label"}
    )


def test_atomic_same_adapter_uses_only_one_shot_and_frozen_codebook(
    tmp_path: Path,
) -> None:
    query, support, labels = _signals(k=1)
    filterbank_path, bound_hash = _bound_filterbank(tmp_path)
    adapter = GovernedCalibrationBaselineAdapter(
        "same3_filterbank_eTRCA",
        n_classes=3,
        filterbank=FILTERBANK,
        filterbank_config_path=filterbank_path,
        filterbank_config_sha256=bound_hash,
        same_seed=91,
    )
    batch = BaselineSignalBatch(
        query_x=query,
        target_support_x=support,
        target_support_y=labels,
        source_x=None,
        source_y=None,
        frequencies_hz=np.asarray([10.0, 12.0, 15.0]),
        phases_rad=np.asarray([0.0, 0.5, 1.0]),
        sfreq=200.0,
    )
    first = adapter.predict(batch, budget=1).scores
    second = adapter.predict(batch, budget=1).scores
    assert np.array_equal(first, second)
    assert first.shape == (3, 3)
    with pytest.raises(ValueError, match="not applicable"):
        adapter.predict(batch, budget=3)
