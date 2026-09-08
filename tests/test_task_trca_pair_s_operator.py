"""Generated algebra only; no artifacts, readers, human data or root gate."""

import numpy as np
import pytest
import torch
from scipy import linalg

import cfeg.analysis.task_trca_pair_s_operator as operator
from cfeg.analysis.task_trca_pair_s_operator import (
    c_projectors,
    pair_distribution,
    pair_projectors,
    weighted_numerator,
)
from cfeg.analysis.task_trca_shape_operator import LOGIT_BOUND, leading_projector
from cfeg.analysis.task_trca_shape_signfree import score_temporal_gram, temporal_statistics

D = torch.float64


def matrices(batch=()):
    generator = torch.Generator().manual_seed(608)
    a = torch.randn(*batch, 8, 8, generator=generator, dtype=D)
    c = a @ a.mT + 2 * torch.eye(8, dtype=D)
    lower = torch.linalg.cholesky(c)
    q = torch.linalg.qr(torch.randn(*batch, 8, 8, generator=generator, dtype=D)).Q
    roots = torch.tensor([-2, -1, 0, 0.2, 0.6, 1, 2, 4], dtype=D)
    s = lower @ ((q * roots) @ q.mT) @ lower.mT
    return (s + s.mT) / 2, c


def pair_case(n=3, batch=(), bands=2):
    s0, c = matrices((*batch, bands, 12))
    generator = torch.Generator().manual_seed(609)
    a = torch.randn(*batch, bands, n, 12, 8, 8, generator=generator, dtype=D)
    a = (a + a.mT) / 20
    z = torch.linspace(-0.17, 0.19, n, dtype=D).expand(*batch, bands, n).clone()
    return s0, c, a, z


def score_arrays(bands=2):
    generator = torch.Generator().manual_seed(610)
    templates = torch.randn(12, bands, 8, 17, generator=generator, dtype=D) + 3
    query = torch.randn(3, bands, 8, 17, generator=generator, dtype=D) - 7
    weights = torch.linspace(1.4, 0.7, bands, dtype=D)
    return templates, query, weights


def literal_score(filters, templates, query, weights):
    """Project, time-center each component, flatten, then signed Pearson."""
    x = torch.einsum("bik,nbit->nbkt", filters, query)
    t = torch.einsum("bik,cbit->cbkt", filters, templates)
    x = (x - x.mean(-1, keepdim=True)).flatten(2)
    t = (t - t.mean(-1, keepdim=True)).flatten(2)
    x = x / torch.linalg.vector_norm(x, dim=-1, keepdim=True)
    t = t / torch.linalg.vector_norm(t, dim=-1, keepdim=True)
    corr = torch.einsum("nbt,cbt->nbc", x, t)
    return torch.einsum("nbc,b->nc", corr, weights), corr


def literal_filters(s, c):
    """Independent ordinary-eigh derivative oracle, with distinct roots."""
    lower = torch.linalg.cholesky(c)
    left = torch.linalg.solve_triangular(lower, s, upper=False)
    h = torch.linalg.solve_triangular(lower, left.mT, upper=False).mT
    _, vectors = torch.linalg.eigh((h + h.mT) / 2)
    w = torch.linalg.solve_triangular(lower.mT, vectors[..., -1:], upper=True).squeeze(-1)
    w = w / torch.einsum("...i,...ij,...j->...", w, c, w).sqrt()[..., None]
    return w.transpose(-1, -2)


@pytest.mark.parametrize("n", [3, 10])
def test_distribution_literal_bounds_and_shift_invariance(n):
    z = torch.linspace(-LOGIT_BOUND, LOGIT_BOUND, n, dtype=D)
    cases = torch.stack((z, -z, torch.full_like(z, LOGIT_BOUND), torch.zeros_like(z)))
    p, d = pair_distribution(cases)
    u = np.exp(cases.numpy() - cases.numpy().max(axis=-1, keepdims=True))
    np.testing.assert_allclose(p, u / u.sum(axis=-1, keepdims=True), atol=2e-16, rtol=0)
    np.testing.assert_allclose(d, (u - u.mean(-1, keepdims=True)) / u.mean(-1, keepdims=True))
    assert (p > 0).all()
    assert p.min() >= 1 / (2 * n - 1) - 1e-15
    assert p.max() <= 2 / (n + 1) + 1e-15
    assert (p.amax(-1) / p.amin(-1)).max() <= 2 + 1e-15
    torch.testing.assert_close(p.sum(-1), torch.ones(4, dtype=D), atol=2e-16, rtol=0)
    for sign, expected in ((1, 2 / (n + 1)), (-1, 1 / (2 * n - 1))):
        edge = torch.full((n,), -sign * LOGIT_BOUND, dtype=D)
        edge[0] = sign * LOGIT_BOUND
        assert pair_distribution(edge)[0][0].item() == pytest.approx(expected, abs=1e-16)
    for first, second in zip(pair_distribution(z / 2), pair_distribution(z / 2 + 0.1)):
        torch.testing.assert_close(first, second, atol=3e-16, rtol=0)


@pytest.mark.parametrize("n", [3, 10])
@pytest.mark.parametrize("batch", [(), (2,), (2, 1)])
def test_weighted_literal_axes_and_common_positive_scaling(n, batch):
    s0, c, a, z = pair_case(n, batch)
    result = weighted_numerator(s0, a, z)
    u = np.exp(z.numpy() - z.numpy().max(-1, keepdims=True))
    d = (u - u.mean(-1, keepdims=True)) / u.mean(-1, keepdims=True)
    expected = s0.numpy() + (d[..., None, None, None] * a.numpy()).sum(axis=-4)
    np.testing.assert_allclose(result, expected, atol=2e-14, rtol=0)
    assert result.shape == (*batch, 2, 12, 8, 8)
    actual = pair_projectors(s0, c, a, z)
    torch.testing.assert_close(actual, c_projectors(result, c), atol=0, rtol=0)
    torch.testing.assert_close(
        pair_projectors(3.25 * s0, c, 3.25 * a, z), actual, atol=3e-14, rtol=0
    )


@pytest.mark.parametrize("n", [3, 10])
@pytest.mark.parametrize("value", [-LOGIT_BOUND, 0.0, LOGIT_BOUND])
def test_uniform_is_exact_original_s0_with_live_contrast_gradient(n, value):
    s0, _, a, _ = pair_case(n)
    # Accepted native roundoff is validated but must not be rewritten here.
    s0[0, 0, 0, 1] += 1e-14
    a[0, 0, 0, 0, 1] += 1e-14
    z = torch.full((2, n), value, dtype=D, requires_grad=True)
    p, d = pair_distribution(z)
    assert torch.equal(d, torch.zeros_like(d))
    assert torch.equal(weighted_numerator(s0, a, z), s0)
    torch.testing.assert_close(p, torch.full_like(p, 1 / n), atol=0, rtol=0)
    gradient = torch.autograd.grad(weighted_numerator(s0, a, z)[..., 0, 1].sum(), z)[0]
    assert gradient.abs().max() > 1e-3
    torch.testing.assert_close(gradient.sum(-1), torch.zeros(2, dtype=D), atol=1e-14, rtol=0)


@pytest.mark.parametrize("batch", [(), (3,), (2, 3)])
def test_scipy_generalized_eigensystem_rank_one_and_c_trace(batch):
    s, c = matrices(batch)
    actual = c_projectors(s, c)
    expected = []
    for ss, cc in zip(s.reshape(-1, 8, 8).numpy(), c.reshape(-1, 8, 8).numpy(), strict=True):
        _, vectors = linalg.eigh(ss, cc)
        w = vectors[:, -1]
        expected.append(np.outer(w, w) / (w @ cc @ w))
    np.testing.assert_allclose(actual, np.array(expected).reshape(actual.shape), atol=2e-14, rtol=0)
    torch.testing.assert_close(
        torch.einsum("...ij,...ji->...", c, actual), torch.ones(batch, dtype=D), atol=1e-10, rtol=0
    )
    roots = torch.linalg.eigvalsh(actual)
    assert (roots[..., -1] > 0).all()
    assert roots[..., :-1].abs().max() < 1e-14


@pytest.mark.parametrize("n", [3, 10])
def test_literal_scipy_temporal_scores(n):
    s0, c, a, z = pair_case(n)
    templates, query, weights = score_arrays()
    actual = score_temporal_gram(
        pair_projectors(s0, c, a, z), temporal_statistics(templates, query), weights
    )
    # Build S independently; this oracle does not call the producer numerator.
    u = np.exp(z.numpy())
    deviations = (u - u.mean(-1, keepdims=True)) / u.mean(-1, keepdims=True)
    s = s0.numpy() + (deviations[..., None, None, None] * a.numpy()).sum(-4)
    filters = np.empty((2, 8, 12))
    for band in range(2):
        for label in range(12):
            _, v = linalg.eigh(s[band, label], c[band, label].numpy())
            w = v[:, -1]
            filters[band, :, label] = w / np.sqrt(w @ c[band, label].numpy() @ w)
    expected = literal_score(torch.from_numpy(filters), templates, query, weights)
    for aa, ee in zip(actual, expected, strict=True):
        torch.testing.assert_close(aa, ee, atol=2e-14, rtol=0)


@pytest.mark.parametrize("n", [3, 10])
@pytest.mark.parametrize("uniform", [False, True])
def test_pair_projector_and_temporal_score_gradcheck(n, uniform):
    s0, c, a, z = pair_case(n, bands=1)
    if uniform:
        z.zero_()
    z.requires_grad_()
    templates, query, weights = score_arrays(bands=1)
    stats = temporal_statistics(templates, query)
    assert torch.autograd.gradcheck(pair_distribution, (z,), fast_mode=True)
    assert torch.autograd.gradcheck(lambda zz: weighted_numerator(s0, a, zz), (z,), fast_mode=True)
    assert torch.autograd.gradcheck(lambda zz: pair_projectors(s0, c, a, zz), (z,), fast_mode=True)
    assert torch.autograd.gradcheck(
        lambda zz: score_temporal_gram(pair_projectors(s0, c, a, zz), stats, weights)[0],
        (z,),
        fast_mode=True,
    )
    actual = score_temporal_gram(pair_projectors(s0, c, a, z), stats, weights)[0]
    literal = literal_score(
        literal_filters(weighted_numerator(s0, a, z), c), templates, query, weights
    )[0]
    ga = torch.autograd.grad(actual.square().sum(), z)[0]
    gl = torch.autograd.grad(literal.square().sum(), z)[0]
    torch.testing.assert_close(ga, gl, atol=2e-15, rtol=2e-10)
    assert ga.abs().max() > 1e-8
    torch.testing.assert_close(ga.sum(-1), torch.zeros(1, dtype=D), atol=1e-14, rtol=0)


def test_direct_s_gradient_and_first_derivatives_only():
    s, c = matrices()
    s.requires_grad_()
    assert torch.autograd.gradcheck(
        lambda ss: c_projectors((ss + ss.mT) / 2, c), (s,), fast_mode=True
    )
    gradient = torch.autograd.grad(c_projectors(s, c)[0, 1], s, create_graph=True)[0]
    assert gradient.abs().max() > 1e-5
    with pytest.raises(RuntimeError):
        torch.autograd.grad(gradient.sum(), s)


@pytest.mark.parametrize("n", [3, 10])
def test_uniform_ce_contrast_derivative_and_nonuniform_actuation(n):
    s0, c, a, z = pair_case(n)
    originals = tuple(value.clone() for value in (s0, c, a))
    templates, query, weights = score_arrays()
    stats = temporal_statistics(templates, query)
    labels = torch.tensor([0, 4, 9])

    def loss(logits):
        scores = score_temporal_gram(pair_projectors(s0, c, a, logits), stats, weights)[0]
        return torch.nn.functional.cross_entropy(scores / weights.sum() / 0.1, labels)

    uniform = torch.zeros_like(z, requires_grad=True)
    gradient = torch.autograd.grad(loss(uniform), uniform)[0]
    assert gradient.abs().max() > 1e-8
    direction = torch.linspace(-1, 1, n, dtype=D).expand_as(z)
    step = 1e-5
    numerical = (
        loss(uniform.detach() + step * direction) - loss(uniform.detach() - step * direction)
    ) / (2 * step)
    torch.testing.assert_close((gradient * direction).sum(), numerical, atol=1e-9, rtol=1e-5)
    weighted = weighted_numerator(s0, a, z)
    normalize = lambda value: value / torch.linalg.matrix_norm(value, dim=(-2, -1))[..., None, None]
    assert (normalize(weighted) - normalize(s0)).abs().max() > 1e-8
    f0, f1 = pair_projectors(s0, c, a, uniform), pair_projectors(s0, c, a, z)
    assert (f1 - f0).abs().max() > 1e-8
    assert (f1.sum(1) - f0.sum(1)).abs().max() > 1e-8
    scores0, scores1 = (score_temporal_gram(f, stats, weights)[0] for f in (f0, f1))
    assert (scores1 - scores0).abs().max() > 1e-8
    for value, original in zip((s0, c, a), originals, strict=True):
        assert torch.equal(value, original)


def test_scalar_and_grouped_batch_projectors_agree():
    s, c = matrices((2, 12))
    grouped = c_projectors(s, c)
    scalar = torch.stack(
        [c_projectors(ss, cc) for ss, cc in zip(s.flatten(0, 1), c.flatten(0, 1), strict=True)]
    ).reshape_as(grouped)
    torch.testing.assert_close(grouped, scalar, atol=2e-14, rtol=0)


@pytest.mark.parametrize("n", [3, 10])
def test_channel_permutation_equivariance_including_temporal_score(n):
    s0, c, a, z = pair_case(n, batch=(2,))
    permutation = torch.tensor([7, 2, 0, 5, 1, 6, 4, 3])

    def permute(matrix):
        return matrix[..., permutation, :][..., :, permutation]

    f = pair_projectors(s0, c, a, z)
    changed = pair_projectors(permute(s0), permute(c), permute(a), z)
    torch.testing.assert_close(changed, permute(f), atol=2e-14, rtol=0)
    templates, query, weights = score_arrays()
    score = score_temporal_gram(f[0], temporal_statistics(templates, query), weights)[0]
    changed_score = score_temporal_gram(
        changed[0],
        temporal_statistics(templates[:, :, permutation], query[:, :, permutation]),
        weights,
    )[0]
    torch.testing.assert_close(score, changed_score, atol=2e-14, rtol=0)


@pytest.mark.parametrize(
    "bad", [None, [0.0] * 3, torch.zeros(3), torch.zeros(3, dtype=torch.int64)]
)
def test_distribution_rejects_non_float64_tensor(bad):
    with pytest.raises(ValueError, match="float64"):
        pair_distribution(bad)


@pytest.mark.parametrize("shape", [(), (1,), (2,), (5,), (8,), (11,), (0, 3), (2, 0)])
def test_distribution_rejects_wrong_or_empty_pair_axes(shape):
    with pytest.raises(ValueError):
        pair_distribution(torch.zeros(shape, dtype=D))


@pytest.mark.parametrize(
    "value", [float("nan"), float("inf"), -float("inf"), LOGIT_BOUND + 1e-10, -LOGIT_BOUND - 1e-10]
)
def test_distribution_rejects_nonfinite_and_out_of_envelope(value):
    with pytest.raises(ValueError):
        pair_distribution(torch.tensor([0, value, 0], dtype=D))


@pytest.mark.parametrize("which", ["s0", "a", "c"])
def test_support_matrices_must_be_constants(which):
    s0, c, a, z = pair_case()
    {"s0": s0, "a": a, "c": c}[which].requires_grad_()
    with pytest.raises(ValueError, match="constant"):
        pair_projectors(s0, c, a, z)


@pytest.mark.parametrize("which", ["s0", "a", "c"])
@pytest.mark.parametrize("bad", ["asymmetric", "nan", "float32"])
def test_support_validation(which, bad):
    s0, c, a, z = pair_case()
    values = {"s0": s0, "a": a, "c": c}
    if bad == "float32":
        values[which] = values[which].float()
    elif bad == "asymmetric":
        values[which][..., 0, 1] += 0.01
    else:
        values[which][..., 0, 0] = float("nan")
    with pytest.raises(ValueError):
        pair_projectors(values["s0"], values["c"], values["a"], z)


@pytest.mark.parametrize(
    "case",
    [
        "classes",
        "bands",
        "pair_width",
        "missing_band",
        "batch_broadcast",
        "c_broadcast",
        "channels",
    ],
)
def test_exact_shapes_no_implicit_broadcast(case):
    s0, c, a, z = pair_case()
    if case == "classes":
        s0, c, a = s0[:, :11], c[:, :11], a[:, :, :11]
    elif case == "bands":
        z = z[:1]
    elif case == "pair_width":
        a = a[:, :2]
    elif case == "missing_band":
        s0, c, a, z = s0[0], c[0], a[0], z[0]
    elif case == "batch_broadcast":
        s0, c = s0.unsqueeze(0), c.unsqueeze(0)
    elif case == "c_broadcast":
        c = c[:1]
    else:
        s0 = s0[..., :7, :7]
    with pytest.raises(ValueError):
        pair_projectors(s0, c, a, z)


@pytest.mark.parametrize("minimum", [0.0, -0.01])
def test_non_spd_c_is_rejected(minimum):
    s = torch.diag(torch.arange(8, dtype=D))
    c = torch.eye(8, dtype=D)
    c[0, 0] = minimum
    with pytest.raises(ValueError, match="positive definite"):
        c_projectors(s, c)


@pytest.mark.parametrize("gap", [0.0, 1e-12, 9e-11])
@pytest.mark.parametrize("scale", [1.0, 100.0])
def test_original_top_gap_guard(gap, scale):
    s = scale * torch.diag(torch.tensor([0] * 6 + [1, 1 + gap], dtype=D))
    with pytest.raises(ValueError, match="degenerate"):
        c_projectors(s, torch.eye(8, dtype=D))


def test_repeated_lower_roots_are_allowed():
    c = torch.eye(8, dtype=D)
    direction = torch.arange(1, 9, dtype=D)
    direction /= direction.norm()
    s = c + 2 * direction[:, None] * direction[None, :]
    s.requires_grad_()
    actual = c_projectors(s, c)
    torch.testing.assert_close(actual, direction[:, None] * direction[None, :], atol=1e-14, rtol=0)
    assert torch.autograd.grad(actual[0, 1], s)[0].abs().max() > 1e-3


def test_raw_h_reaches_original_symmetry_guard_without_rescue(monkeypatch):
    assert operator.leading_projector is leading_projector
    solve = torch.linalg.solve_triangular
    calls = []

    def perturbed_second_solve(*args, **kwargs):
        result = solve(*args, **kwargs)
        calls.append(kwargs["upper"])
        if len(calls) == 2:
            result = result.clone()
            result[0, 1] += 1e-7
        return result

    monkeypatch.setattr(torch.linalg, "solve_triangular", perturbed_second_solve)
    with pytest.raises(ValueError, match="h must be symmetric within the frozen tolerance"):
        c_projectors(torch.diag(torch.arange(8, dtype=D)), torch.eye(8, dtype=D))
    assert calls == [False, False]


@pytest.mark.parametrize("scale", [0.0, -1.0, float("inf")])
def test_nonpositive_nonfinite_c_trace_is_rejected(monkeypatch, scale):
    def invalid_projector(h):
        return leading_projector(h) * scale

    monkeypatch.setattr(operator, "leading_projector", invalid_projector)
    with pytest.raises(ValueError, match="projector C trace"):
        c_projectors(torch.diag(torch.arange(8, dtype=D)), torch.eye(8, dtype=D))


def test_c_trace_near_one_is_diagnostic_not_extra_runtime_threshold(monkeypatch):
    s, c = matrices()
    expected = c_projectors(s, c)
    monkeypatch.setattr(operator, "leading_projector", lambda h: 1.25 * leading_projector(h))
    torch.testing.assert_close(c_projectors(s, c), expected, atol=2e-15, rtol=0)


@pytest.mark.parametrize("stage", ["eigvalsh", "cholesky"])
def test_solver_failures_propagate_without_backend_or_jitter(monkeypatch, stage):
    def fail(*args, **kwargs):
        raise RuntimeError("injected solver failure")

    monkeypatch.setattr(torch.linalg, stage, fail)
    with pytest.raises(ValueError, match="eigensolver failed|Cholesky failed"):
        c_projectors(torch.diag(torch.arange(8, dtype=D)), torch.eye(8, dtype=D))


@pytest.mark.parametrize("constant", ["query", "template"])
def test_temporal_zero_variance_remains_terminal(constant):
    s0, c, a, z = pair_case()
    templates, query, weights = score_arrays()
    if constant == "query":
        query[0].fill_(5)
    else:
        templates[0].fill_(5)
    with pytest.raises(ValueError, match="variance"):
        score_temporal_gram(
            pair_projectors(s0, c, a, z), temporal_statistics(templates, query), weights
        )
