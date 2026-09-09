"""Deterministic toy algebra only: no RNG seeds, artifacts or registered run."""

import math

import pytest
import torch

import cfeg.analysis.numerical_stability_operator as operator
from cfeg.analysis.numerical_stability_operator import METHODS, diagnostics, projector

D = torch.float64


def sym(value):
    return (value + value.mT) / 2


def rotation():
    q = torch.eye(8, dtype=D)
    for i, j, angle in ((0, 7, 0.31), (2, 5, -0.43), (1, 6, 0.27), (3, 7, 0.19), (0, 4, -0.23)):
        turn = torch.eye(8, dtype=D)
        turn[i, i] = turn[j, j] = math.cos(angle)
        turn[i, j], turn[j, i] = -math.sin(angle), math.sin(angle)
        q = q @ turn
    return q


def case(batch=()):
    lower = torch.tril(torch.arange(64, dtype=D).reshape(8, 8) / 200)
    lower += torch.diag(torch.linspace(1, 2, 8, dtype=D))
    b = lower @ lower.mT
    q = rotation()
    roots = torch.tensor([-2, -1.5, -1, -0.5, 0, 0.5, 1, 3], dtype=D)
    s = sym(lower @ ((q * roots) @ q.mT) @ lower.mT)
    c = sym((q.mT * torch.linspace(0.7, 2.1, 8, dtype=D)) @ q)
    g = sym(torch.cos(torch.arange(64, dtype=D).reshape(8, 8)))
    assert torch.linalg.matrix_norm(g @ s - s @ g) > 0.1
    if batch:
        index = torch.arange(math.prod(batch), dtype=D).reshape(batch)
        factors = (1 + index / 11)[..., None, None]
        s, b, c, g = s * factors, b / factors, c * (2 - 1 / factors), g * factors
    return s, b, c, g


def grads(s, b, c, g, method):
    inputs = tuple(value.clone().requires_grad_() for value in (s, b, c))
    f = projector(*inputs, method)
    return f.detach(), torch.autograd.grad((f * g).sum(), inputs)


@pytest.mark.parametrize("method", METHODS)
def test_closed_form_rotated_2d_projector_and_angle_gradient(method):
    angle = torch.tensor(0.37, dtype=D, requires_grad=True)
    top = torch.cat((torch.stack((angle.cos(), angle.sin())), torch.zeros(6, dtype=D)))
    other = torch.cat((torch.stack((-angle.sin(), angle.cos())), torch.zeros(6, dtype=D)))
    s = 3 * top[:, None] * top[None, :] + other[:, None] * other[None, :]
    s = s + torch.diag(torch.tensor([0, 0, -2, -3, -4, -5, -6, -7], dtype=D))
    b = torch.eye(8, dtype=D)
    c = torch.diag(torch.linspace(0.7, 1.4, 8, dtype=D))
    g = case()[3]
    expected = top[:, None] * top[None, :] / (top @ c @ top)
    actual = projector(s, b, c, method)
    torch.testing.assert_close(actual, expected, atol=2e-15, rtol=0)
    ga = torch.autograd.grad((actual * g).sum(), angle, retain_graph=True)[0]
    ge = torch.autograd.grad((expected * g).sum(), angle)[0]
    torch.testing.assert_close(ga, ge, atol=2e-14, rtol=0)
    assert ga.abs() > 0.1


@pytest.mark.parametrize("method", METHODS)
def test_independent_s_b_c_gradcheck_and_dense_probe(method):
    s, b, c, g = case()
    inputs = tuple(value.clone().requires_grad_() for value in (s, b, c))
    assert torch.autograd.gradcheck(
        lambda ss, bb, cc: projector(sym(ss), sym(bb), sym(cc), method),
        inputs,
        fast_mode=True,
        atol=2e-6,
        rtol=2e-4,
    )
    _, gradients = grads(s, b, c, g, method)
    assert all(value.abs().max() > 1e-4 for value in gradients)
    assert all(torch.allclose(value, value.mT, rtol=0, atol=1e-15) for value in gradients)


@pytest.mark.parametrize("method", METHODS)
def test_tied_b_equals_c_accumulates_both_gradients(method):
    s, b, _, g = case()
    expected_f, (gs, gb, gc) = grads(s, b, b, g, method)
    ss, common = s.clone().requires_grad_(), b.clone().requires_grad_()
    actual = projector(ss, common, common, method)
    actual_s, actual_common = torch.autograd.grad((actual * g).sum(), (ss, common))
    torch.testing.assert_close(actual, expected_f, atol=0, rtol=0)
    torch.testing.assert_close(actual_s, gs, atol=1e-14, rtol=0)
    torch.testing.assert_close(actual_common, gb + gc, atol=1e-14, rtol=0)
    assert gc.abs().max() > 1e-4  # Dropping normalization's C path cannot pass.
    assert torch.autograd.gradcheck(
        lambda bb: projector(s, sym(bb), sym(bb), method),
        (common,),
        fast_mode=True,
        atol=2e-6,
        rtol=2e-4,
    )


@pytest.mark.parametrize("method", METHODS)
def test_exact_repeated_lower_roots_with_live_s_and_b_gradients(method):
    # Exact rational matrix: seven eigenvalues -1, one eigenvalue 3.
    top = torch.tensor([0.5] * 4 + [0.0] * 4, dtype=D)
    b = torch.eye(8, dtype=D)
    s = -b + 4 * top[:, None] * top[None, :]
    c, g = case()[2:]
    expected = top[:, None] * top[None, :] / (top @ c @ top)
    actual, gradient = grads(s, b, c, g, method)
    torch.testing.assert_close(actual, expected, atol=2e-15, rtol=0)
    assert gradient[0].abs().max() > 1e-3 and gradient[1].abs().max() > 1e-3
    assert torch.autograd.gradcheck(
        lambda ss, bb: projector(sym(ss), sym(bb), c, method),
        (s.clone().requires_grad_(), b.clone().requires_grad_()),
        fast_mode=True,
        atol=2e-6,
        rtol=2e-4,
    )


@pytest.mark.parametrize("method", METHODS)
def test_exact_repeated_b_roots_are_not_an_eigenbasis_gradient_failure(method):
    b = torch.diag(torch.tensor([1, 1, 4, 4, 16, 16, 64, 64], dtype=D))
    lower = torch.diag(b.diag().sqrt())
    q = rotation()
    h = (q * torch.tensor([-2, -1.5, -1, -0.5, 0, 0.5, 1, 3], dtype=D)) @ q.mT
    s = sym(lower @ h @ lower.mT)
    c = torch.eye(8, dtype=D)
    assert torch.autograd.gradcheck(
        lambda bb: projector(s, sym(bb), c, method),
        (b.requires_grad_(),),
        fast_mode=True,
        atol=2e-6,
        rtol=2e-4,
    )


@pytest.mark.parametrize("method", METHODS)
@pytest.mark.parametrize("batch", [(3,), (2, 2)])
def test_scalar_batch_forward_and_vjp_parity(method, batch):
    s, b, c, g = case(batch)
    batch_f, batch_grad = grads(s, b, c, g, method)
    scalar_f, scalar_grad = [], [[], [], []]
    for values in zip(*(value.reshape(-1, 8, 8) for value in (s, b, c, g)), strict=True):
        f, gradient = grads(*values, method)
        scalar_f.append(f)
        for storage, value in zip(scalar_grad, gradient, strict=True):
            storage.append(value)
    torch.testing.assert_close(
        batch_f, torch.stack(scalar_f).reshape_as(batch_f), atol=2e-14, rtol=0
    )
    for batched, individual in zip(batch_grad, scalar_grad, strict=True):
        torch.testing.assert_close(
            batched, torch.stack(individual).reshape_as(batched), atol=2e-14, rtol=0
        )


@pytest.mark.parametrize("method", METHODS)
@pytest.mark.parametrize("scale", [2.0**-12, 2.0**12])
def test_joint_power_of_two_scaling_for_f_and_gradients(method, scale):
    s, b, c, g = case()
    base_f, base_grad = grads(s, b, c, g, method)
    scaled_f, scaled_grad = grads(scale * s, scale * b, scale * c, g, method)
    torch.testing.assert_close(scale * scaled_f, base_f, atol=2e-14, rtol=0)
    for scaled, base in zip(scaled_grad, base_grad, strict=True):
        torch.testing.assert_close(scale**2 * scaled, base, atol=2e-14, rtol=0)


def test_two_methods_forward_and_full_vjp_agree_on_toy():
    s, b, c, g = case((2,))
    first_f, first_grad = grads(s, b, c, g, METHODS[0])
    second_f, second_grad = grads(s, b, c, g, METHODS[1])
    torch.testing.assert_close(first_f, second_f, atol=2e-14, rtol=0)
    for a, e in zip(first_grad, second_grad, strict=True):
        torch.testing.assert_close(a, e, atol=2e-14, rtol=0)


@pytest.mark.parametrize("method", METHODS)
def test_diagnostics_are_detached_exact_shapes_and_original_pair_metrics(method):
    s, b, c, _ = case((2, 3))
    s.requires_grad_()
    f = projector(s, b, c, method)
    actual = diagnostics(s, b, c, f)
    assert set(actual) == {
        "lambda",
        "residual",
        "normalization_error",
        "symmetry_error",
        "raw_h_skew",
    }
    assert all(value.shape == (2, 3) and not value.requires_grad for value in actual.values())
    eigenvalue = (s @ f).diagonal(dim1=-2, dim2=-1).sum(-1) / (b @ f).diagonal(
        dim1=-2, dim2=-1
    ).sum(-1)
    denominator = (
        torch.linalg.matrix_norm(s) + eigenvalue.abs() * torch.linalg.matrix_norm(b)
    ) * torch.linalg.matrix_norm(f)
    expected = torch.linalg.matrix_norm(s @ f - eigenvalue[..., None, None] * (b @ f)) / denominator
    torch.testing.assert_close(actual["lambda"], eigenvalue.detach(), atol=2e-14, rtol=0)
    torch.testing.assert_close(actual["residual"], expected.detach(), atol=1e-15, rtol=0)
    assert actual["residual"].max() <= 1e-12
    assert actual["normalization_error"].max() < 1e-14


def test_diagnostics_do_not_claim_small_residual_certifies_topness():
    s = torch.diag(torch.arange(1, 9, dtype=D))
    b = torch.eye(8, dtype=D)
    lower = torch.zeros_like(s)
    lower[0, 0] = 1
    answer = diagnostics(s, b, b, lower)
    assert answer["residual"] == 0 and answer["lambda"] == 1
    assert answer["normalization_error"] == 0
    scaled = diagnostics(s, b, b, 2 * lower)
    assert scaled["normalization_error"] == 1


@pytest.mark.parametrize("method", METHODS)
@pytest.mark.parametrize("which", [0, 1, 2])
@pytest.mark.parametrize("bad", ["asymmetric", "nan", "float32", "empty", "bad_shape", "meta"])
def test_bad_inputs_rejected_before_solver(method, which, bad, monkeypatch):
    inputs = list(case()[:3])
    if bad == "asymmetric":
        inputs[which][0, 1] += 1e-3
    elif bad == "nan":
        inputs[which][0, 0] = float("nan")
    elif bad == "float32":
        inputs[which] = inputs[which].float()
    elif bad == "empty":
        inputs[which] = torch.empty(0, 8, 8, dtype=D)
    elif bad == "bad_shape":
        inputs[which] = inputs[which][:7, :7]
    else:
        inputs[which] = torch.empty(8, 8, dtype=D, device="meta")
    monkeypatch.setattr(operator, "_n1", lambda *args: pytest.fail("N1 entered before validation"))
    monkeypatch.setattr(
        operator.linalg, "eigh", lambda *args, **kwargs: pytest.fail("N2 entered before validation")
    )
    with pytest.raises(ValueError):
        projector(*inputs, method)


@pytest.mark.parametrize("method", METHODS)
@pytest.mark.parametrize("which", [1, 2])
@pytest.mark.parametrize("minimum", [0.0, -1.0])
def test_non_spd_b_or_c_is_terminal(method, which, minimum):
    inputs = [torch.diag(torch.arange(8, dtype=D)), torch.eye(8, dtype=D), torch.eye(8, dtype=D)]
    inputs[which][0, 0] = minimum
    with pytest.raises(ValueError, match="positive definite"):
        projector(*inputs, method)


@pytest.mark.parametrize("method", METHODS)
@pytest.mark.parametrize("gap", [0.0, 5e-12, 9e-11, 1.1e-10, 1.9e-10, 2.1e-10])
def test_top_gap_both_threshold_sides(method, gap):
    s = torch.diag(torch.tensor([-2, -1.5, -1, -0.5, 0, 0.5, 1 - gap, 1], dtype=D))
    b = torch.eye(8, dtype=D)
    # max(abs(roots))=2, so the exact frozen threshold is 2e-10.
    if gap <= 2e-10:
        with pytest.raises(ValueError, match="degenerate"):
            projector(s, b, b, method)
    else:
        assert torch.isfinite(projector(s, b, b, method)).all()


@pytest.mark.parametrize("method", METHODS)
def test_relative_input_symmetry_not_old_absolute_floor(method):
    s, b, c, _ = case()
    s, b, c = s * 1e-9, b * 1e-9, c * 1e-9
    s[0, 1] += 1e-15  # Small absolute skew, excessive relative skew.
    with pytest.raises(ValueError, match="symmetry"):
        projector(s, b, c, method)


@pytest.mark.parametrize("method", METHODS)
def test_no_implicit_broadcast_and_exact_method_names(method):
    s, b, c, _ = case()
    with pytest.raises(ValueError, match="without broadcasting"):
        projector(s[None], b, c, method)
    for unknown in ("N1", "n2", None, True, 1):
        with pytest.raises(ValueError, match="method"):
            projector(s, b, c, unknown)


@pytest.mark.parametrize("method", METHODS)
def test_solver_failure_never_falls_back(method, monkeypatch):
    def fail(*args, **kwargs):
        raise RuntimeError("injected N1 failure")

    def fail_gvd(*args, **kwargs):
        assert kwargs["driver"] == "gvd" and kwargs["type"] == 1 and kwargs["lower"] is True
        raise operator.linalg.LinAlgError("injected GVD failure")

    monkeypatch.setattr(operator, "_n1", fail)
    monkeypatch.setattr(operator.linalg, "eigh", fail_gvd)
    with pytest.raises(ValueError, match="no fallback"):
        projector(*case()[:3], method)


def test_n1_new_symmetrization_keeps_raw_h_diagnostic(monkeypatch):
    s = torch.diag(torch.arange(1, 9, dtype=D))
    b = torch.eye(8, dtype=D)
    original = operator._raw_h

    def skewed(ss, bb):
        lower, raw = original(ss, bb)
        raw = raw.clone()
        raw[0, 1] += 1e-11
        raw[1, 0] -= 1e-11
        return lower, raw

    monkeypatch.setattr(operator, "_raw_h", skewed)
    f = projector(s, b, b, METHODS[0])
    assert diagnostics(s, b, b, f)["raw_h_skew"] > 1e-12
    assert f[7, 7] == 1


def test_original_pair_residual_rejects_bad_new_symmetry_result(monkeypatch):
    s = torch.diag(torch.arange(1, 9, dtype=D))
    b = torch.eye(8, dtype=D)

    def wrong(*args):
        w = torch.ones(8, dtype=D) / math.sqrt(8)
        return w[:, None] * w[None, :]

    monkeypatch.setattr(operator, "_n1", wrong)
    with pytest.raises(ValueError, match="Original-pair residual"):
        projector(s, b, b, METHODS[0])


@pytest.mark.parametrize("method", METHODS)
def test_first_order_s_gradient_not_a_second_derivative_contract(method):
    s, b, c, g = case()
    s.requires_grad_()
    gradient = torch.autograd.grad((projector(s, b, c, method) * g).sum(), s, create_graph=True)[0]
    assert gradient.abs().max() > 1e-4
    with pytest.raises(RuntimeError):
        torch.autograd.grad(gradient.sum(), s)


def test_diagnostics_uses_original_not_only_averaged_inputs():
    s = torch.diag(torch.arange(1, 9, dtype=D))
    b = torch.eye(8, dtype=D)
    s[0, 7] = 1e-12
    f = torch.zeros_like(s)
    f[7, 7] = 1
    raw = diagnostics(s, b, b, f)["residual"]
    averaged = diagnostics(sym(s), b, b, f)["residual"]
    assert raw > averaged > 0


@pytest.mark.parametrize("method", METHODS)
def test_no_input_mutation_and_cpu1_toy_process(method):
    assert torch.get_num_threads() == 1
    inputs = tuple(value.clone().requires_grad_() for value in case()[:3])
    snapshots = tuple(value.detach().clone() for value in inputs)
    f = projector(*inputs, method)
    torch.autograd.grad(f.square().sum(), inputs)
    for value, expected in zip(inputs, snapshots, strict=True):
        assert torch.equal(value.detach(), expected)
