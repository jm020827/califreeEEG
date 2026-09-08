"""Artificial arrays only: no raw EEG, metadata packets, or study artifacts."""

from dataclasses import replace

import numpy as np
import pytest
import torch
from numpy.testing import assert_allclose, assert_array_equal
from scipy import linalg

from cfeg.analysis.metadata_trca_prior import TrcaModel, fit_trca, score_trca, trca_matrices
from cfeg.analysis.task_trca_shape_operator import (
    ETA,
    LOGIT_BOUND,
    R_MAX,
    bounded_filters,
    gram_statistics,
    leading_projector,
    native_zero_scores,
    score_flattened,
    score_gram,
    shape_prior,
)

DTYPE = torch.float64


def synthetic_matrices(batch=()):
    generator = torch.Generator().manual_seed(48271)
    basis = torch.linalg.qr(torch.randn(*batch, 8, 8, generator=generator, dtype=DTYPE)).Q
    c_values = torch.tensor([1, 1, 2, 4, 8, 16, 32, 1000], dtype=DTYPE)
    c = (basis * c_values) @ basis.transpose(-1, -2)
    lower = torch.linalg.cholesky(c)
    eigenbasis = torch.linalg.qr(torch.randn(*batch, 8, 8, generator=generator, dtype=DTYPE)).Q
    roots = torch.tensor([-1, -1, 0, 0, 1, 1, 2, 4], dtype=DTYPE)
    s = lower @ ((eigenbasis * roots) @ eigenbasis.transpose(-1, -2)) @ lower.transpose(-1, -2)
    s = (s + s.transpose(-1, -2)) / 2
    anchors = torch.linalg.solve_triangular(
        lower.transpose(-1, -2), eigenbasis[..., -1:], upper=True
    ).squeeze(-1)
    return s, c, anchors


def synthetic_scores():
    rng = np.random.default_rng(6151)
    templates = rng.normal(size=(12, 5, 8, 31)) + rng.normal(scale=3, size=(12, 5, 8, 1))
    query = rng.normal(size=(7, 5, 8, 31)) + rng.normal(scale=3, size=(7, 5, 8, 1))
    filters = rng.normal(size=(5, 8, 12))
    weights = np.array([1.4, 0.8, 0.5, 0.4, 0.3])
    return tuple(torch.tensor(item, dtype=DTYPE) for item in (filters, templates, query, weights))


@pytest.mark.parametrize("repeated_lower", [False, True])
def test_projector_gradcheck_unique_top_including_repeated_lower(repeated_lower):
    roots = torch.tensor([1] * 7 + [3] if repeated_lower else list(range(8)), dtype=DTYPE)
    h = torch.diag(roots).requires_grad_()
    assert torch.autograd.gradcheck(
        lambda value: leading_projector((value + value.T) / 2),
        (h,),
        eps=1e-6,
        atol=1e-5,
        rtol=1e-3,
    )
    expected = torch.zeros((8, 8), dtype=DTYPE)
    expected[-1, -1] = 1
    torch.testing.assert_close(leading_projector(h), expected, rtol=0, atol=0)


def test_projector_backward_directional_step_halved_check():
    generator = torch.Generator().manual_seed(929)
    q = torch.linalg.qr(torch.randn(8, 8, generator=generator, dtype=DTYPE)).Q
    h = (q @ torch.diag(torch.tensor([0] * 7 + [2.0], dtype=DTYPE)) @ q.T).requires_grad_()
    raw_direction = torch.randn(8, 8, generator=generator, dtype=DTYPE)
    direction = (raw_direction + raw_direction.T) / 2
    upstream = torch.randn(8, 8, generator=generator, dtype=DTYPE)
    value = (leading_projector(h) * upstream).sum()
    gradient = torch.autograd.grad(value, h)[0]
    analytic = float((gradient * direction).sum())
    errors = []
    for step in (1e-4, 5e-5, 2.5e-5):
        plus = (leading_projector(h.detach() + step * direction) * upstream).sum()
        minus = (leading_projector(h.detach() - step * direction) * upstream).sum()
        errors.append(abs(float((plus - minus) / (2 * step)) - analytic))
    assert max(errors) < 1e-7
    assert errors[-1] < errors[0]


@pytest.mark.parametrize("gap", [0.0, 1e-12, 9e-11])
def test_projector_rejects_top_tie_or_near_tie(gap):
    h = torch.diag(torch.tensor([0] * 6 + [1.0, 1.0 + gap], dtype=DTYPE))
    with pytest.raises(ValueError, match="degenerate"):
        leading_projector(h)


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), -float("inf")])
def test_projector_rejects_nonfinite_input_and_gradient(bad):
    h = torch.diag(torch.arange(8, dtype=DTYPE)).requires_grad_()
    corrupted = h.detach().clone()
    corrupted[0, 0] = bad
    with pytest.raises(ValueError, match="finite"):
        leading_projector(corrupted)
    with pytest.raises(ValueError, match="upstream gradient"):
        leading_projector(h).backward(torch.full((8, 8), bad, dtype=DTYPE))


def test_projector_rejects_asymmetry_and_non_double():
    bad = torch.eye(8, dtype=DTYPE)
    bad[0, 1] = 1e-5
    with pytest.raises(ValueError, match="symmetric"):
        leading_projector(bad)
    with pytest.raises(ValueError, match="float64"):
        leading_projector(bad.float())


def test_shape_prior_extreme_bound_trace_and_derivative():
    logits = torch.tensor(
        [[LOGIT_BOUND] + [-LOGIT_BOUND] * 7, [-LOGIT_BOUND] + [LOGIT_BOUND] * 7], dtype=DTYPE
    )
    r = shape_prior(logits)
    torch.testing.assert_close(
        r.sum(dim=-1), torch.full((2,), 8.0, dtype=DTYPE), atol=1e-14, rtol=0
    )
    assert bool((r > 0).all())
    assert float((r.amax(dim=-1) / r.amin(dim=-1)).max()) == pytest.approx(2)
    assert float(r.max()) == pytest.approx(R_MAX)
    torch.testing.assert_close(
        shape_prior(torch.zeros(8, dtype=DTYPE)), torch.ones(8, dtype=DTYPE), rtol=0, atol=0
    )
    with pytest.raises(ValueError, match="bound"):
        shape_prior(torch.full((8,), LOGIT_BOUND + 1e-6, dtype=DTYPE))


def test_positive_mass_bound_and_common_trace_with_imbalanced_spd():
    s, c, anchor = synthetic_matrices((2, 3))
    c_min = torch.linalg.eigvalsh(c)[..., 0]
    tau = ETA * c_min / R_MAX
    shapes = [
        torch.ones((2, 1, 8), dtype=DTYPE),
        shape_prior(torch.tensor([[LOGIT_BOUND] + [-LOGIT_BOUND] * 7], dtype=DTYPE)).expand(
            2, 1, 8
        ),
    ]
    previous_trace = None
    for shape in shapes:
        w = bounded_filters(s, c, anchor, shape)
        norm2 = torch.einsum("...i,...ij,...j->...", w, c, w)
        torch.testing.assert_close(norm2, torch.ones((2, 3), dtype=DTYPE), atol=1e-10, rtol=0)
        assert bool((torch.einsum("...i,...ij,...j->...", w, c, anchor) > 0).all())
        p = torch.diag_embed(tau.unsqueeze(-1) * shape)
        lo = torch.linalg.cholesky(c)
        left = torch.linalg.solve_triangular(lo, p, upper=False)
        relative = torch.linalg.solve_triangular(lo, left.transpose(-1, -2), upper=False).transpose(
            -1, -2
        )
        rho = torch.linalg.eigvalsh((relative + relative.transpose(-1, -2)) / 2)
        assert float(rho.min()) >= -1e-12
        assert float(rho.max()) <= ETA + 1e-10
        trace = p.diagonal(dim1=-2, dim2=-1).sum(dim=-1)
        torch.testing.assert_close(trace, 8 * tau, atol=1e-12, rtol=0)
        if previous_trace is not None:
            torch.testing.assert_close(trace, previous_trace, atol=1e-12, rtol=0)
        previous_trace = trace


def test_bounded_filter_gradcheck_only_shape_is_trainable():
    s, c, anchor = synthetic_matrices()
    logits = torch.zeros(8, dtype=DTYPE, requires_grad=True)
    assert torch.autograd.gradcheck(
        lambda value: bounded_filters(s, c, anchor, shape_prior(value)),
        (logits,),
        eps=1e-6,
        atol=1e-5,
        rtol=1e-3,
    )
    for index in range(3):
        inputs = [s.clone(), c.clone(), anchor.clone()]
        inputs[index].requires_grad_()
        with pytest.raises(ValueError, match="detached"):
            bounded_filters(*inputs, torch.ones(8, dtype=DTYPE))


def test_bounded_filters_preserve_inputs_and_eigensolver_sign_gauge(monkeypatch):
    s, c, anchor = synthetic_matrices((2, 3))
    logits = torch.zeros((2, 1, 8), dtype=DTYPE, requires_grad=True)
    snapshots = [value.detach().clone() for value in (s, c, anchor, logits)]
    first = bounded_filters(s, c, anchor, shape_prior(logits))
    first_gradient = torch.autograd.grad(first.square().sum(), logits)[0]
    original = torch.linalg.eigh

    def flipped(value):
        vals, vecs = original(value)
        signs = torch.tensor([-1, 1, -1, 1, -1, 1, -1, -1], dtype=DTYPE, device=value.device)
        return vals, vecs * signs

    monkeypatch.setattr(torch.linalg, "eigh", flipped)
    second = bounded_filters(s, c, anchor, shape_prior(logits))
    second_gradient = torch.autograd.grad(second.square().sum(), logits)[0]
    torch.testing.assert_close(first, second, rtol=0, atol=0)
    torch.testing.assert_close(first_gradient, second_gradient, rtol=0, atol=0)
    for value, snapshot in zip((s, c, anchor, logits), snapshots):
        torch.testing.assert_close(value, snapshot, rtol=0, atol=0)


def test_joint_channel_permutation_with_fixed_native_anchor():
    s, c, anchor = synthetic_matrices((5, 12))
    logits = torch.linspace(-LOGIT_BOUND / 2, LOGIT_BOUND / 2, 8, dtype=DTYPE).expand(5, 1, 8)
    r = shape_prior(logits)
    original = bounded_filters(s, c, anchor, r)
    permutation = torch.tensor([6, 1, 7, 3, 0, 5, 2, 4])
    changed = bounded_filters(
        s[..., permutation, :][..., permutation],
        c[..., permutation, :][..., permutation],
        anchor[..., permutation],
        r[..., permutation],
    )
    torch.testing.assert_close(changed, original[..., permutation], atol=1e-10, rtol=0)


@pytest.mark.parametrize("kind", ["singular", "indefinite", "asymmetric", "nan", "inf"])
def test_bounded_filters_reject_bad_c_without_rescue(kind):
    s = torch.diag(torch.arange(8, dtype=DTYPE))
    c = torch.eye(8, dtype=DTYPE)
    anchor = torch.eye(8, dtype=DTYPE)[-1]
    if kind == "singular":
        c[0, 0] = 0
    elif kind == "indefinite":
        c[0, 0] = -1
    elif kind == "asymmetric":
        c[0, 1] = 1e-3
    else:
        c[0, 0] = float(kind)
    snapshot = c.clone()
    with pytest.raises(ValueError):
        bounded_filters(s, c, anchor, torch.ones(8, dtype=DTYPE))
    torch.testing.assert_close(c, snapshot, rtol=0, atol=0, equal_nan=True)


@pytest.mark.parametrize("overlap", [0.0, 1e-8, 1e-6])
def test_bounded_filters_reject_nearly_orthogonal_anchor(overlap):
    s = torch.diag(torch.arange(8, dtype=DTYPE))
    c = torch.eye(8, dtype=DTYPE)
    anchor = torch.zeros(8, dtype=DTYPE)
    anchor[0], anchor[-1] = 1, overlap
    with pytest.raises(ValueError, match="anchor"):
        bounded_filters(s, c, anchor, torch.ones(8, dtype=DTYPE))


@pytest.mark.parametrize(
    "r",
    [
        torch.full((8,), 2.0, dtype=DTYPE),
        torch.tensor([0.0] + [8 / 7] * 7, dtype=DTYPE),
        torch.tensor([2.4] + [0.8] * 7, dtype=DTYPE),
    ],
)
def test_bounded_filters_reject_invalid_shapes(r):
    with pytest.raises(ValueError):
        bounded_filters(*synthetic_matrices(), r)


@pytest.mark.parametrize("eta", [0, 0.2, -1, True, float("nan")])
def test_no_eta_retuning_or_fake_native_zero_path(eta):
    with pytest.raises(ValueError, match="eta"):
        bounded_filters(*synthetic_matrices(), torch.ones(8, dtype=DTYPE), eta=eta)


def test_gram_and_flattened_match_unchanged_native_global_pearson_and_gradient():
    filters, templates, query, weights = synthetic_scores()
    stats = gram_statistics(templates, query)
    filters.requires_grad_()
    gram_scores, gram_corr = score_gram(filters, stats, weights)
    flat_scores, flat_corr = score_flattened(filters, templates, query, weights)
    model = TrcaModel(filters.detach().numpy(), templates.numpy(), {})
    native_scores, native_corr = score_trca(model, query.numpy(), weights.numpy())
    assert_allclose(gram_corr.detach().numpy(), native_corr, atol=1e-10, rtol=0)
    assert_allclose(gram_scores.detach().numpy(), native_scores, atol=1e-10, rtol=0)
    torch.testing.assert_close(gram_corr, flat_corr, atol=1e-10, rtol=0)
    torch.testing.assert_close(gram_scores, flat_scores, atol=1e-10, rtol=0)
    gram_grad = torch.autograd.grad(gram_scores.square().sum(), filters)[0]
    flat_grad = torch.autograd.grad(flat_scores.square().sum(), filters)[0]
    torch.testing.assert_close(gram_grad, flat_grad, atol=1e-10, rtol=1e-10)
    assert_array_equal(gram_scores.detach().numpy().argmax(axis=-1), native_scores.argmax(axis=-1))


def test_gram_gradient_matches_uncentered_original_formula():
    filters, templates, query, weights = synthetic_scores()
    filters.requires_grad_()
    score, _ = score_gram(filters, gram_statistics(templates, query), weights)
    j = filters @ filters.transpose(-1, -2)
    u = filters.sum(dim=-1)
    m = filters.shape[-1] * query.shape[-1]
    sx = torch.einsum("bi,nbi->nb", u, query.sum(dim=-1))
    st = torch.einsum("bi,cbi->bc", u, templates.sum(dim=-1))
    cross = torch.einsum("nbit,cbjt->ncbij", query, templates)
    dot = torch.einsum("bij,ncbij->nbc", j, cross) - sx.unsqueeze(-1) * st.unsqueeze(0) / m
    xx = torch.einsum("bij,nbij->nb", j, query @ query.transpose(-1, -2)) - sx.square() / m
    tt = torch.einsum("bij,cbij->bc", j, templates @ templates.transpose(-1, -2)) - st.square() / m
    raw_corr = torch.clamp(dot / xx.sqrt().unsqueeze(-1) / tt.sqrt().unsqueeze(0), -1, 1)
    raw_score = torch.einsum("nbc,b->nc", raw_corr, weights)
    torch.testing.assert_close(score, raw_score, atol=1e-10, rtol=0)
    first = torch.autograd.grad(score.square().sum(), filters)[0]
    second = torch.autograd.grad(raw_score.square().sum(), filters)[0]
    torch.testing.assert_close(first, second, atol=1e-10, rtol=1e-10)


def test_gram_filter_gradient_check():
    filters, templates, query, weights = synthetic_scores()
    filters = filters[:1, :, :2].clone().requires_grad_()
    templates, query, weights = templates[:2, :1], query[:1, :1], weights[:1]
    stats = gram_statistics(templates, query)
    assert torch.autograd.gradcheck(
        lambda value: score_gram(value, stats, weights)[0],
        (filters,),
        eps=1e-6,
        atol=1e-5,
        rtol=1e-3,
    )


def test_nonzero_mean_class_sign_flip_is_not_silently_invariant():
    filters, templates, query, weights = synthetic_scores()
    stats = gram_statistics(templates, query)
    original = score_gram(filters, stats, weights)[0]
    changed = filters.clone()
    changed[..., 0] *= -1
    signed = score_gram(changed, stats, weights)[0]
    assert float((original - signed).abs().max()) > 1e-3
    centered_stats = gram_statistics(
        templates - templates.mean(dim=-1, keepdim=True), query - query.mean(dim=-1, keepdim=True)
    )
    torch.testing.assert_close(
        score_gram(filters, centered_stats, weights)[0],
        score_gram(changed, centered_stats, weights)[0],
        atol=1e-12,
        rtol=0,
    )
    assert float((original - score_gram(filters, centered_stats, weights)[0]).abs().max()) > 1e-3


def test_gram_joint_channel_permutation_and_negative_query():
    filters, templates, query, weights = synthetic_scores()
    original, corr = score_gram(filters, gram_statistics(templates, query), weights)
    p = torch.tensor([6, 1, 7, 3, 0, 5, 2, 4])
    changed, _ = score_gram(
        filters[:, p], gram_statistics(templates[:, :, p], query[:, :, p]), weights
    )
    torch.testing.assert_close(original, changed, atol=1e-12, rtol=0)
    negative, negative_corr = score_gram(filters, gram_statistics(templates, -query), weights)
    torch.testing.assert_close(negative, -original, atol=1e-12, rtol=0)
    torch.testing.assert_close(negative_corr, -corr, atol=1e-12, rtol=0)
    zero, _ = score_gram(filters, gram_statistics(templates, query), torch.zeros_like(weights))
    torch.testing.assert_close(zero, torch.zeros_like(zero), atol=0, rtol=0)


def test_statistics_are_detached_unaliased_and_inputs_unmodified():
    inputs = synthetic_scores()
    snapshots = [item.clone() for item in inputs]
    filters, templates, query, weights = inputs
    stats = gram_statistics(templates, query)
    score_gram(filters, stats, weights)
    score_flattened(filters, templates, query, weights)
    for value, snapshot in zip(inputs, snapshots):
        torch.testing.assert_close(value, snapshot, rtol=0, atol=0)
    mean_copy = stats.query_mean.clone()
    query.zero_()
    torch.testing.assert_close(stats.query_mean, mean_copy, rtol=0, atol=0)
    with pytest.raises(ValueError, match="detached"):
        gram_statistics(templates.requires_grad_(), snapshots[2])


@pytest.mark.parametrize("scale", [1e200, 1e-200])
def test_gram_statistics_stable_under_extreme_common_positive_units(scale):
    filters, templates, query, weights = synthetic_scores()
    original = score_gram(filters, gram_statistics(templates, query), weights)[0]
    changed = score_gram(filters, gram_statistics(templates * scale, query * scale), weights)[0]
    torch.testing.assert_close(original, changed, atol=1e-12, rtol=0)


@pytest.mark.parametrize("scorer", ["gram", "flattened"])
def test_zero_projected_variance_is_failure_not_floor(scorer):
    filters = torch.ones((2, 8, 3), dtype=DTYPE)
    templates = torch.ones((3, 2, 8, 5), dtype=DTYPE)
    query = torch.ones((1, 2, 8, 5), dtype=DTYPE)
    weights = torch.ones(2, dtype=DTYPE)
    with pytest.raises(ValueError, match="variance"):
        if scorer == "gram":
            score_gram(filters, gram_statistics(templates, query), weights)
        else:
            score_flattened(filters, templates, query, weights)


def test_identical_trial_correlations_finite_and_clipped():
    filters, templates, _, weights = synthetic_scores()
    scores, corr = score_gram(filters, gram_statistics(templates, templates), weights)
    assert bool(torch.isfinite(scores).all())
    assert float(corr.max()) <= 1
    assert float(corr.min()) >= -1
    torch.testing.assert_close(
        corr[torch.arange(12), :, torch.arange(12)],
        torch.ones((12, 5), dtype=DTYPE),
        atol=1e-14,
        rtol=0,
    )


@pytest.mark.parametrize("fault", ["nonfinite", "shape", "gradient", "samples"])
def test_invalid_or_learned_gram_statistics_rejected(fault):
    filters, templates, query, weights = synthetic_scores()
    stats = gram_statistics(templates, query)
    if fault == "nonfinite":
        stats = replace(stats, query_mean=torch.full_like(stats.query_mean, float("nan")))
    elif fault == "shape":
        stats = replace(stats, cross_gram=stats.cross_gram[:, :1])
    elif fault == "gradient":
        stats = replace(stats, query_mean=stats.query_mean.clone().requires_grad_())
    else:
        stats = replace(stats, samples=True)
    with pytest.raises(ValueError):
        score_gram(filters, stats, weights)


def test_native_zero_path_exact_project_delegate_and_independent_native_equations():
    rng = np.random.default_rng(10617)
    prototypes = rng.normal(size=(12, 5, 8, 31))
    support = prototypes[None] + rng.normal(scale=0.7, size=(3, 12, 5, 8, 31))
    query = prototypes[:7] + rng.normal(scale=0.7, size=(7, 5, 8, 31))
    weights = np.array([1.4, 0.8, 0.5, 0.4, 0.3])
    expected_model = fit_trca(support, np.ones((5, 8)), gamma=0)
    expected = score_trca(expected_model, query, weights)
    actual = native_zero_scores(support, query, weights)
    for observed, reference in zip(actual, expected):
        assert_array_equal(observed, reference)
    filters = np.empty((5, 8, 12))
    for label in range(12):
        for band in range(5):
            trials = support[:, label, band]
            total = sum(trials)
            concat = np.concatenate([trial.T for trial in trials], axis=0)
            s = total @ total.T - concat.T @ concat
            centered = concat - concat.mean(axis=0)
            c = centered.T @ centered
            values, vectors = linalg.eig(s, c)
            vectors = vectors[:, np.argsort(values)[::-1]]
            vectors = vectors / np.sqrt(np.diag(vectors.T @ c @ vectors))
            filters[band, :, label] = vectors[:, 0]
    independent_corr = np.empty_like(actual[1])
    templates = support.mean(axis=0)
    for n, trial in enumerate(query):
        for band in range(5):
            projected_query = (filters[band].T @ trial[band]).reshape(-1)
            for label in range(12):
                projected_template = (filters[band].T @ templates[label, band]).reshape(-1)
                independent_corr[n, band, label] = np.corrcoef(projected_query, projected_template)[
                    0, 1
                ]
    assert_allclose(actual[1], independent_corr, atol=1e-9, rtol=0)
    assert_array_equal(
        actual[0].argmax(axis=-1), np.einsum("nbc,b->nc", independent_corr, weights).argmax(axis=-1)
    )
    # Positive ISO has a new bounded denominator and is not a gamma-zero alias.
    s = torch.empty((5, 12, 8, 8), dtype=DTYPE)
    c = torch.empty_like(s)
    for band in range(5):
        for label in range(12):
            s[band, label], c[band, label] = (
                torch.tensor(x, dtype=DTYPE) for x in trca_matrices(support[:, label, band])
            )
    anchors = torch.tensor(expected_model.filters.transpose(0, 2, 1), dtype=DTYPE)
    iso_filters = bounded_filters(s, c, anchors, torch.ones((5, 1, 8), dtype=DTYPE)).transpose(
        -1, -2
    )
    iso = score_gram(
        iso_filters,
        gram_statistics(torch.tensor(templates), torch.tensor(query)),
        torch.tensor(weights),
    )[0]
    assert np.max(np.abs(iso.numpy() - actual[0])) > 1e-6
