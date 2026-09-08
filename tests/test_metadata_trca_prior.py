"""Artificial-array checks only; no data files or declared suite seeds are used."""

from dataclasses import replace

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal
from scipy import linalg

from cfeg.analysis import metadata_trca_prior as prior_module
from cfeg.analysis.metadata_trca_prior import (
    fit_trca,
    residual_prior,
    score_trca,
    trace_normalize,
    trca_matrices,
)


def artificial_arrays(*, centered=False):
    rng = np.random.default_rng(8192)
    prototype = rng.normal(size=(4, 2, 3, 31))
    support = prototype[None] + rng.normal(scale=0.7, size=(3, 4, 2, 3, 31))
    query = prototype[[0, 1, 2, 3, 0]] + rng.normal(scale=0.7, size=(5, 2, 3, 31))
    if centered:
        support -= support.mean(axis=-1, keepdims=True)
        query -= query.mean(axis=-1, keepdims=True)
    return support, query


def native_reference(support, query, diagonal, gamma):
    """Independent transcription of the pinned helper and ETRCA scoring equations."""
    _, classes, bands, channels, _ = support.shape
    filters = np.zeros((bands, channels, classes))
    templates = support.mean(axis=0)
    for label in range(classes):
        for band in range(bands):
            trials = support[:, label, band]
            x1 = sum(trials)
            x2 = np.concatenate([trial.T for trial in trials], axis=0)
            s = x1 @ x1.T - x2.T @ x2
            x2 = x2 - x2.mean(axis=0)
            c = x2.T @ x2
            denominator = c + gamma * np.trace(c) / channels * np.diag(diagonal[band])
            values, vectors = linalg.eig(s, denominator)
            vectors = vectors[:, np.argsort(values)[::-1]]
            vectors = vectors / np.sqrt(np.diag(vectors.T @ denominator @ vectors))
            filters[band, :, label] = vectors[:, 0]
    correlations = np.empty((len(query), bands, classes))
    for index, trial in enumerate(query):
        for band in range(bands):
            for label in range(classes):
                a = (filters[band].T @ trial[band]).reshape(-1)
                b = (filters[band].T @ templates[label, band]).reshape(-1)
                correlations[index, band, label] = np.corrcoef(a, b)[0, 1]
    return filters, correlations


def test_trca_matrices_match_uncentered_native_equations():
    trials = artificial_arrays()[0][:, 0, 0]
    s, c = trca_matrices(trials)
    pair_sum = np.zeros_like(s)
    for i in range(len(trials)):
        for j in range(i + 1, len(trials)):
            pair_sum += trials[i] @ trials[j].T + trials[j] @ trials[i].T
    flattened = np.concatenate(trials, axis=1)
    centered = flattened - flattened.mean(axis=1, keepdims=True)
    assert_allclose(s, pair_sum, rtol=1e-13, atol=1e-12)
    assert_allclose(c, centered @ centered.T, rtol=1e-13, atol=1e-12)
    separately_centered = trials - trials.mean(axis=-1, keepdims=True)
    assert not np.allclose(s, trca_matrices(separately_centered)[0])


@pytest.mark.parametrize("gamma", [0.0, 0.1])
@pytest.mark.parametrize("diagonal", [np.ones((2, 3)), [[0.5, 1.0, 1.5], [1.3, 0.9, 0.8]]])
def test_native_reference_compatibility(gamma, diagonal):
    support, query = artificial_arrays()
    diagonal = np.asarray(diagonal)
    model = fit_trca(support, diagonal, gamma)
    weights = np.array([0.75, 1.25])
    scores, correlations = score_trca(model, query, weights)
    expected_filters, expected_corr = native_reference(support, query, diagonal, gamma)
    assert_allclose(model.filters, expected_filters, rtol=2e-13, atol=1e-14)
    assert_allclose(correlations, expected_corr, rtol=2e-13, atol=1e-14)
    assert_allclose(scores, np.einsum("nbc,b->nc", expected_corr, weights), atol=1e-13)
    assert_array_equal(model.templates, support.mean(axis=0))
    assert model.filters.shape == (2, 3, 4)
    assert scores.shape == (5, 4)
    assert correlations.shape == (5, 2, 4)


def test_denominator_normalization_and_trace_matching():
    support, _ = artificial_arrays()
    q = trace_normalize([[0.5, 1, 4], [3, 1, 2]])
    qm = residual_prior(q, [[0.2, -0.2, 0.1], [-0.2, 0.2, 0]], np.ones((2, 3), bool))
    first = fit_trca(support, q, 0.1)
    second = fit_trca(support, qm, 0.1)
    assert_allclose(first.diagnostics["penalty_trace"], second.diagnostics["penalty_trace"])
    assert_allclose(first.diagnostics["prior_trace"], [3, 3])
    assert np.max(first.diagnostics["denominator_norm_errors"]) < 1e-12
    assert np.min(first.diagnostics["denominator_min_eigenvalues"]) > 0
    for label in range(4):
        for band in range(2):
            _, c = trca_matrices(support[:, label, band])
            denominator = c + 0.1 * np.trace(c) / 3 * np.diag(q[band])
            w = first.filters[band, :, label]
            assert w @ denominator @ w == pytest.approx(1, abs=1e-13)


def test_prior_really_changes_filters_and_signed_scores():
    support, query = artificial_arrays()
    first = fit_trca(support, np.ones((2, 3)), 0.1)
    second = fit_trca(support, np.tile([0.5, 1, 1.5], (2, 1)), 0.1)
    scores, corr = score_trca(first, query, [1, 0.5])
    changed, _ = score_trca(second, query, [1, 0.5])
    assert np.max(np.abs(scores - changed)) > 1e-5
    negative, negative_corr = score_trca(first, -query, [1, 0.5])
    assert_allclose(negative_corr, -corr, atol=1e-14)
    assert_allclose(negative, -scores, atol=1e-14)
    # Zero weights are not silently replaced with a uniform or normalized blend.
    assert_array_equal(score_trca(first, query, [0, 0])[0], np.zeros((5, 4)))


def test_trace_normalization_leading_axes_and_extreme_common_units():
    values = np.arange(1, 25, dtype=float).reshape(2, 4, 3)
    expected = values / values.mean(axis=-1, keepdims=True)
    for amplitude in [1.0, 1e300, 1e-300]:
        actual = trace_normalize(values * amplitude)
        assert actual.dtype == np.float64
        assert_allclose(actual, expected, rtol=1e-14)
        assert_allclose(actual.sum(axis=-1), 3)
        assert np.all(actual > 0)


def test_residual_mask_fallback_and_bounds():
    q = trace_normalize([[0.4, 1, 5], [3, 5, 2], [1, 2, 3]])
    delta = np.array([[100.0, -100, np.nan], [np.nan, np.nan, np.nan], [0.1, 0.2, -0.1]])
    mask = np.isfinite(delta)
    actual = residual_prior(q, delta, mask)
    expected_first = trace_normalize(q[0] * np.exp([0.2, -0.2, 0]))
    assert_allclose(actual[0], expected_first, atol=1e-15)
    assert_array_equal(actual[1], q[1])
    assert actual[0, 2] != q[0, 2]  # Shared normalization affects missing channels.
    assert_allclose(actual.sum(axis=-1), 3)
    ratio = actual / q
    assert np.all(ratio >= np.exp(-0.4) - 1e-14)
    assert np.all(ratio <= np.exp(0.4) + 1e-14)
    assert_array_equal(residual_prior(q, np.full(q.shape, np.nan), np.zeros(q.shape, bool)), q)
    assert_allclose(residual_prior(q, np.zeros_like(q), np.ones(q.shape, bool), 0), q)


def test_no_input_mutation_or_returned_template_alias():
    support, query = artificial_arrays()
    q = trace_normalize([[0.5, 1, 2], [1, 3, 4]])
    delta = np.full((2, 3), 0.1)
    mask = np.ones((2, 3), dtype=bool)
    weights = np.array([0.5, 1.0])
    arrays = [support, query, q, delta, mask, weights]
    copies = [a.copy() for a in arrays]
    for array in arrays:
        array.flags.writeable = False
    adjusted = residual_prior(q, delta, mask)
    model = fit_trca(support, adjusted, 0.1)
    model_copies = (model.filters.copy(), model.templates.copy())
    score_trca(model, query, weights)
    for array, before in zip(arrays, copies):
        assert_array_equal(array, before)
    assert_array_equal(model.filters, model_copies[0])
    assert_array_equal(model.templates, model_copies[1])
    assert not np.shares_memory(model.templates, support)
    assert not np.shares_memory(adjusted, q)


@pytest.mark.parametrize("gamma", [0.0, 0.1])
def test_centered_channel_class_and_uniform_unit_invariance(gamma):
    support, query = artificial_arrays(centered=True)
    diagonal = np.array([[0.5, 1, 1.5], [1.3, 0.9, 0.8]])
    original = score_trca(fit_trca(support, diagonal, gamma), query, [1, 0.5])[0]
    channel_order = [2, 0, 1]
    class_order = [3, 0, 2, 1]
    shifted = fit_trca(
        support[:, class_order][:, :, :, channel_order], diagonal[:, channel_order], gamma
    )
    scores, _ = score_trca(shifted, query[:, :, channel_order], [1, 0.5])
    assert_allclose(scores, original[:, class_order], atol=1e-12)
    for amplitude in [1e-80, 1e80]:
        scaled = fit_trca(support * amplitude, diagonal, gamma)
        assert_allclose(score_trca(scaled, query * amplitude, [1, 0.5])[0], original, atol=2e-12)


def test_common_channel_gain_is_not_new_unregularized_method():
    support, query = artificial_arrays(centered=True)
    gains = np.array([0.5, 1.5, 3.0])[:, None]
    first = fit_trca(support, np.ones((2, 3)), 0)
    transformed = fit_trca(support * gains, np.ones((2, 3)), 0)
    assert_allclose(
        score_trca(first, query, [1, 0.5])[0],
        score_trca(transformed, query * gains, [1, 0.5])[0],
        atol=1e-12,
    )


@pytest.mark.parametrize(
    "values",
    [
        [],
        1,
        [0, 1],
        [-1, 2],
        [np.nan, 1],
        [np.inf, 1],
        [1 + 0j, 2],
        [True, False],
        ["1", "2"],
        [np.nextafter(0.0, 1.0), 1e300],
    ],
)
def test_trace_normalization_rejects_invalid_values(values):
    with pytest.raises(ValueError):
        trace_normalize(values)


@pytest.mark.parametrize(
    "damage",
    [
        "shape",
        "mask_shape",
        "mask_integer",
        "available_nan",
        "infinite",
        "negative_q",
        "trace",
        "bound",
        "complex",
    ],
)
def test_residual_prior_rejects_invalid_inputs(damage):
    q = np.ones((2, 3))
    delta = np.zeros((2, 3))
    mask = np.ones((2, 3), bool)
    bound = 0.2
    if damage == "shape":
        delta = delta[0]
    elif damage == "mask_shape":
        mask = mask[0]
    elif damage == "mask_integer":
        mask = mask.astype(int)
    elif damage == "available_nan":
        delta[0, 0] = np.nan
    elif damage == "infinite":
        delta[0, 0] = np.inf
        mask[0, 0] = False
    elif damage == "negative_q":
        q[0, 0] = -1
    elif damage == "trace":
        q *= 2
    elif damage == "bound":
        bound = -1
    else:
        delta = delta.astype(complex)
    with pytest.raises(ValueError):
        residual_prior(q, delta, mask, bound)


@pytest.mark.parametrize(
    "damage",
    [
        "k1",
        "empty_class",
        "one_channel",
        "one_sample",
        "ndim",
        "nan",
        "complex",
        "zero",
        "constant",
        "underflow",
        "overflow",
        "prior_shape",
        "prior_zero",
        "prior_trace",
        "gamma_negative",
        "gamma_nan",
        "gamma_array",
    ],
)
def test_fit_rejects_invalid_and_numerically_empty_inputs(damage):
    support, _ = artificial_arrays()
    diagonal = np.ones((2, 3))
    gamma = 0.1
    if damage == "k1":
        support = support[:1]
    elif damage == "empty_class":
        support = support[:, :0]
    elif damage == "one_channel":
        support = support[:, :, :, :1]
    elif damage == "one_sample":
        support = support[..., :1]
    elif damage == "ndim":
        support = support[0]
    elif damage == "nan":
        support[0, 0, 0, 0, 0] = np.nan
    elif damage == "complex":
        support = support.astype(complex)
    elif damage == "zero":
        support[:] = 0
    elif damage == "constant":
        support[:] = 1
    elif damage == "underflow":
        support *= 1e-200
    elif damage == "overflow":
        support *= 1e200
    elif damage == "prior_shape":
        diagonal = diagonal[0]
    elif damage == "prior_zero":
        diagonal[0, 0] = 0
    elif damage == "prior_trace":
        diagonal *= 2
    elif damage == "gamma_negative":
        gamma = -0.1
    elif damage == "gamma_nan":
        gamma = np.nan
    else:
        gamma = [0.1]
    with pytest.raises(ValueError):
        fit_trca(support, diagonal, gamma)


def test_singular_denominator_and_tied_eigenspace_are_rejected():
    support, _ = artificial_arrays()
    support[:, :, :, 2] = 0
    with pytest.raises(ValueError, match="positive definite"):
        fit_trca(support, np.ones((2, 3)), 0)
    # Two orthogonal, equal-energy, exactly repeated channels give a tied top root.
    basis = np.array([[1.0, -1, 1, -1], [1, 1, -1, -1]])
    degenerate = np.tile(basis, (3, 1, 1, 1, 1))
    with pytest.raises(ValueError, match="degenerate"):
        fit_trca(degenerate, np.ones((1, 2)), 0.1)


@pytest.mark.parametrize("values", [[1, 1 - 1e-11, 0], [1 + 1j, 0, -1], [np.inf, 1, 0]])
def test_near_tie_complex_and_nonfinite_eigensystems_are_rejected(monkeypatch, values):
    monkeypatch.setattr(prior_module.linalg, "eig", lambda *args: (np.asarray(values), np.eye(3)))
    with pytest.raises(ValueError):
        fit_trca(artificial_arrays()[0], np.ones((2, 3)), 0.1)


@pytest.mark.parametrize(
    "damage",
    [
        "empty",
        "shape",
        "nan",
        "complex",
        "zero",
        "weights_shape",
        "weights_nan",
        "template_zero",
        "filter_shape",
        "model",
    ],
)
def test_score_rejects_invalid_inputs_and_zero_projected_variance(damage):
    support, query = artificial_arrays()
    model = fit_trca(support, np.ones((2, 3)), 0.1)
    weights = np.ones(2)
    if damage == "empty":
        query = query[:0]
    elif damage == "shape":
        query = query[..., :-1]
    elif damage == "nan":
        query[0, 0, 0, 0] = np.nan
    elif damage == "complex":
        query = query.astype(complex)
    elif damage == "zero":
        query[:] = 0
    elif damage == "weights_shape":
        weights = weights[None]
    elif damage == "weights_nan":
        weights[0] = np.nan
    elif damage == "template_zero":
        model = replace(model, templates=np.zeros_like(model.templates))
    elif damage == "filter_shape":
        model = replace(model, filters=model.filters[..., :1])
    else:
        model = None
    with pytest.raises(TypeError if damage == "model" else ValueError):
        score_trca(model, query, weights)
