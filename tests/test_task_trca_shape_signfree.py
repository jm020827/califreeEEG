"""Generated matrices only; never open diagnostic/human artifacts in pytest."""

from dataclasses import replace

import numpy as np
import pytest
import torch
from scipy import linalg

from cfeg.analysis.task_trca_shape_operator import (
    bounded_filters,
    gram_statistics,
    score_flattened,
    shape_prior,
)
from cfeg.analysis.task_trca_shape_signfree import (
    bounded_projectors,
    score_temporal_gram,
    temporal_statistics,
)

D = torch.float64


def matrices(batch=()):
    g = torch.Generator().manual_seed(20260908)
    a = torch.randn(*batch, 8, 8, generator=g, dtype=D)
    c = a @ a.mT + torch.eye(8, dtype=D)
    l = torch.linalg.cholesky(c)
    q = torch.linalg.qr(torch.randn(*batch, 8, 8, generator=g, dtype=D)).Q
    h = (q * torch.tensor([-2, -1, 0, 0, 0, 1, 2, 4], dtype=D)) @ q.mT
    s = l @ h @ l.mT
    anchor = torch.linalg.solve_triangular(l.mT, q[..., -1:], upper=True).squeeze(-1)
    return (s + s.mT) / 2, c, anchor


def score_arrays():
    g = torch.Generator().manual_seed(20260908)
    w = torch.randn(2, 8, 3, generator=g, dtype=D)
    t = torch.randn(3, 2, 8, 17, generator=g, dtype=D)
    x = torch.randn(4, 2, 8, 17, generator=g, dtype=D)
    t += torch.randn(3, 2, 8, 1, generator=g, dtype=D) * 3
    x += torch.randn(4, 2, 8, 1, generator=g, dtype=D) * 3
    return w, t, x, torch.tensor([1.4, 0.7], dtype=D)


def outer(w):
    v = w.transpose(1, 2)
    return v.unsqueeze(-1) * v.unsqueeze(-2)


def literal(w, t, x, weights):
    # Independent literal projection: no producer statistics or output helper.
    ux = torch.einsum("bik,nbit->nbkt", w, x)
    ut = torch.einsum("bik,cbit->cbkt", w, t)
    ux = (ux - ux.mean(-1, keepdim=True)).flatten(2)
    ut = (ut - ut.mean(-1, keepdim=True)).flatten(2)
    ux = ux / torch.linalg.vector_norm(ux, dim=-1, keepdim=True)
    ut = ut / torch.linalg.vector_norm(ut, dim=-1, keepdim=True)
    corr = torch.einsum("nbt,cbt->nbc", ux, ut)
    return torch.einsum("nbc,b->nc", corr, weights), corr


@pytest.mark.parametrize("batch", [(), (3,), (2, 3)])
def test_scipy_generalized_eigenprojector_and_c_trace(batch):
    s, c, anchors = matrices(batch)
    r = shape_prior(torch.linspace(-0.2, 0.2, 8, dtype=D))
    actual = bounded_projectors(s, c, r)
    expected = []
    for ss, cc in zip(s.reshape(-1, 8, 8).numpy(), c.reshape(-1, 8, 8).numpy()):
        tau = 0.1 * linalg.eigvalsh(cc)[0] / (16 / 9)
        _, vectors = linalg.eigh(ss, cc + tau * np.diag(r.numpy()))
        v = vectors[:, -1]
        expected.append(np.outer(v, v) / (v @ cc @ v))
    np.testing.assert_allclose(actual, np.array(expected).reshape(actual.shape), atol=1e-12)
    torch.testing.assert_close(
        torch.einsum("...ij,...ij->...", c, actual), torch.ones(batch, dtype=D), atol=2e-13, rtol=0
    )
    roots = torch.linalg.eigvalsh(actual)
    assert roots[..., -1].min() > 0
    assert roots[..., :-1].abs().max() < 1e-12
    w = bounded_filters(s, c, anchors, r)
    torch.testing.assert_close(actual, w[..., None] * w[..., None, :], atol=2e-12, rtol=0)


def test_real_native_top_switch_has_no_anchor_in_new_api():
    c = torch.eye(8, dtype=D)
    s = torch.diag(torch.tensor([1.001, 1.0, 0, 0, 0, 0, 0, 0], dtype=D))
    native_anchor = c[0]
    r = torch.tensor([1.5] + [6.5 / 7] * 7, dtype=D)
    with pytest.raises(ValueError, match="anchor"):
        bounded_filters(s, c, native_anchor, r)
    expected = torch.zeros(8, 8, dtype=D)
    expected[1, 1] = 1
    torch.testing.assert_close(bounded_projectors(s, c, r), expected, atol=0, rtol=0)


def test_r_gradient_including_trace_normalization_and_score():
    s, c, _ = matrices((2, 3))
    logits = torch.linspace(-0.17, 0.17, 16, dtype=D).reshape(2, 8).requires_grad_()
    _, t, x, weights = score_arrays()
    stats = temporal_statistics(t, x)
    assert torch.autograd.gradcheck(
        lambda z: bounded_projectors(s, c, shape_prior(z)[:, None, :]),
        (logits,),
        atol=2e-6,
        rtol=2e-4,
    )
    assert torch.autograd.gradcheck(
        lambda z: score_temporal_gram(
            bounded_projectors(s, c, shape_prior(z)[:, None, :]), stats, weights
        )[0],
        (logits,),
        atol=2e-6,
        rtol=2e-4,
    )


def test_repeated_lower_roots_allowed_and_top_ties_rejected():
    c = torch.eye(8, dtype=D)
    r = torch.ones(8, dtype=D)
    good = torch.diag(torch.tensor([1] * 7 + [3], dtype=D))
    assert torch.isfinite(bounded_projectors(good, c, r)).all()
    for gap in (0, 1e-12, 9e-11):
        bad = torch.diag(torch.tensor([0] * 6 + [1, 1 + gap], dtype=D))
        with pytest.raises(ValueError, match="degenerate"):
            bounded_projectors(bad, c, r)


def test_literal_scores_and_gradients_match_new_gram():
    w, t, x, weights = score_arrays()
    w.requires_grad_()
    actual = score_temporal_gram(outer(w), temporal_statistics(t, x), weights)
    expected = literal(w, t, x, weights)
    for a, b in zip(actual, expected):
        torch.testing.assert_close(a, b, atol=2e-14, rtol=0)
    ga = torch.autograd.grad(actual[0].square().sum(), w)[0]
    gb = torch.autograd.grad(expected[0].square().sum(), w)[0]
    torch.testing.assert_close(ga, gb, atol=2e-14, rtol=0)
    assert ga.abs().max() > 1e-7  # not a disconnected/constant scorer


def test_sign_component_permutation_and_fixed_filter_offset_invariance():
    w, t, x, weights = score_arrays()
    a = score_temporal_gram(outer(w), temporal_statistics(t, x), weights)[0]
    w2 = w[..., [2, 0, 1]] * torch.tensor([-1, 1, -1], dtype=D)
    b = literal(w2, t, x, weights)[0]
    c = score_temporal_gram(outer(w2), temporal_statistics(t + 13, x - 9), weights)[0]
    torch.testing.assert_close(a, b, atol=2e-14, rtol=0)
    torch.testing.assert_close(a, c, atol=2e-14, rtol=0)


def test_centered_inputs_equal_old_scorer_for_same_filters():
    w, t, x, weights = score_arrays()
    t, x = t - t.mean(-1, keepdim=True), x - x.mean(-1, keepdim=True)
    actual = score_temporal_gram(outer(w), temporal_statistics(t, x), weights)
    expected = score_flattened(w, t, x, weights)
    for a, b in zip(actual, expected):
        torch.testing.assert_close(a, b, atol=2e-14, rtol=0)


def test_exact_offset_counterexample_native_sign_changes_but_new_does_not():
    # Six identical copies of each component embed the example in 12-filter eTRCA.
    w = torch.zeros(1, 8, 12, dtype=D)
    w[0, 0, :6], w[0, 1, 6:] = 1, 1
    x = torch.zeros(1, 1, 8, 2, dtype=D)
    t = torch.zeros(12, 1, 8, 2, dtype=D)
    x[0, 0, :2] = torch.tensor([[1, 3], [3, 5]], dtype=D)
    t[:, 0, :2] = torch.tensor([[1, 3], [-3, -1]], dtype=D)
    weights = torch.ones(1, dtype=D)
    native = score_flattened(w, t, x, weights)[0]
    new = score_temporal_gram(outer(w), temporal_statistics(t, x), weights)[0]
    flipped = w.clone()
    flipped[..., 6:] *= -1
    native_flip = score_flattened(flipped, t, x, weights)[0]
    new_flip = score_temporal_gram(outer(flipped), temporal_statistics(t, x), weights)[0]
    torch.testing.assert_close(native, torch.full_like(native, -1 / np.sqrt(10)))
    torch.testing.assert_close(native_flip, -native, atol=2e-15, rtol=0)
    torch.testing.assert_close(new, torch.ones_like(new), atol=2e-15, rtol=0)
    torch.testing.assert_close(new_flip, new, atol=0, rtol=0)


def test_new_zero_temporal_variance_is_rejected_even_if_native_defined():
    w, t, x, weights = score_arrays()
    x = x[..., :1].expand_as(x).clone()
    # Use exact channel constants avoiding residual cancellation from mean rounding.
    x[:] = torch.arange(8, dtype=D)[None, None, :, None]
    assert torch.isfinite(score_flattened(w, t, x, weights)[0]).all()
    with pytest.raises(ValueError, match="variance"):
        score_temporal_gram(outer(w), temporal_statistics(t, x), weights)


def test_common_channel_permutation_equivariance():
    s, c, _ = matrices((2, 3))
    r = shape_prior(torch.linspace(-0.2, 0.2, 8, dtype=D))
    p = torch.tensor([3, 6, 0, 7, 1, 5, 2, 4])
    f = bounded_projectors(s, c, r)
    fp = bounded_projectors(s[..., p, :][..., p], c[..., p, :][..., p], r[p])
    torch.testing.assert_close(fp, f[..., p, :][..., p], atol=2e-13, rtol=0)
    _, t, x, weights = score_arrays()
    a = score_temporal_gram(f, temporal_statistics(t, x), weights)[0]
    b = score_temporal_gram(fp, temporal_statistics(t[:, :, p], x[:, :, p]), weights)[0]
    torch.testing.assert_close(a, b, atol=2e-13, rtol=0)


@pytest.mark.parametrize("which", ["s", "c", "r"])
@pytest.mark.parametrize("bad", [float("nan"), float("inf")])
def test_reject_nonfinite_matrix_prior(which, bad):
    s, c, _ = matrices()
    inputs = {"s": s, "c": c, "r": torch.ones(8, dtype=D)}
    inputs[which] = inputs[which].clone()
    inputs[which].reshape(-1)[0] = bad
    with pytest.raises(ValueError, match="finite"):
        bounded_projectors(**inputs)


@pytest.mark.parametrize("which", ["s", "c"])
def test_reject_grad_on_support(which):
    s, c, _ = matrices()
    inputs = {"s": s, "c": c, "r": torch.ones(8, dtype=D)}
    inputs[which].requires_grad_()
    with pytest.raises(ValueError, match="detached"):
        bounded_projectors(**inputs)


@pytest.mark.parametrize("bad", [0.0, -1.0])
def test_reject_non_spd_c(bad):
    s, c, _ = matrices()
    c = torch.eye(8, dtype=D)
    c[0, 0] = bad
    with pytest.raises(ValueError, match="positive definite"):
        bounded_projectors(s, c, torch.ones(8, dtype=D))


@pytest.mark.parametrize(
    "bad_r",
    [torch.zeros(8, dtype=D), torch.ones(7, dtype=D), torch.ones(8), torch.ones(2, 8, dtype=D)],
)
def test_reject_invalid_prior(bad_r):
    s, c, _ = matrices()
    with pytest.raises(ValueError):
        bounded_projectors(s, c, bad_r)


def test_no_implicit_global_statistics_coercion_and_validation_guards():
    w, t, x, weights = score_arrays()
    f = outer(w)
    with pytest.raises(TypeError, match="TemporalGramStatistics"):
        score_temporal_gram(f, gram_statistics(t, x), weights)
    stats = temporal_statistics(t, x)
    for bad in (True, 1, 17.0):
        with pytest.raises(ValueError, match="sample"):
            score_temporal_gram(f, replace(stats, samples=bad), weights)
    for bad in (f.float(), -f, torch.zeros_like(f), f[:, :1]):
        with pytest.raises(ValueError):
            score_temporal_gram(bad, stats, weights)
    with pytest.raises(ValueError, match="detached"):
        score_temporal_gram(
            f, replace(stats, query_gram=stats.query_gram.requires_grad_()), weights
        )


def test_rotating_complete_frame_does_not_prove_score_actuation():
    # Every rank-one component changes but J=I is unchanged: geometry != efficacy.
    w = torch.eye(8, dtype=D).unsqueeze(0)
    q = torch.eye(8, dtype=D)
    q[:2, :2] = torch.tensor([[0.6, -0.8], [0.8, 0.6]], dtype=D)
    assert (outer(w) - outer(q.unsqueeze(0))).abs().max() > 0.1
    torch.testing.assert_close(outer(w).sum(1), outer(q.unsqueeze(0)).sum(1))
