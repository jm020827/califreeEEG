"""Batch/scalar equality on generated arrays, including the fixed optimizer."""

from dataclasses import replace

import numpy as np
import pytest
import torch

from cfeg.analysis import task_trca_shape_learning as learning
from cfeg.analysis.task_trca_shape_batch import TaskBatch


def fixture_cases(device="cpu", people=2):
    rng = np.random.default_rng(20260909)
    cases = []
    for pid in range(11001, 11001 + people):
        packet5 = rng.uniform(0, 30, (5, 8))
        packet5[:, 2] = np.nan
        for samples in (17, 23):
            proto = rng.normal(size=(12, 5, 8, samples)) + 0.7
            x = proto[None] + rng.normal(size=(5, 12, 5, 8, samples))
            y = proto + rng.normal(size=proto.shape)
            for k in (3, 5):
                cases.append(
                    learning.make_task_case(
                        pid,
                        0,
                        0,
                        x[:k],
                        packet5[:k],
                        np.linspace(9, 14.5, 12),
                        y,
                        weights=np.array([1.25, 0.67, 0.50, 0.43, 0.38]),
                        device=device,
                    )
                )
    return tuple(cases)


def test_batched_scores_loss_gradient_mixed_samples():
    torch.set_num_threads(1)
    cases = fixture_cases()
    logits = (
        torch.linspace(-0.2, 0.2, len(cases) * 40, dtype=torch.float64)
        .reshape(-1, 5, 8)
        .requires_grad_()
    )
    batch = TaskBatch(cases)
    scores = batch.scores(logits)
    expected = torch.stack([learning._prior_scores(c, logits[j]) for j, c in enumerate(cases)])
    torch.testing.assert_close(scores, expected, atol=1e-10, rtol=0)
    actual_loss = batch.loss(logits)
    expected_loss = torch.stack([learning._ce(expected[j], c) for j, c in enumerate(cases)]).mean()
    (actual_grad,) = torch.autograd.grad(actual_loss, logits, retain_graph=True)
    (expected_grad,) = torch.autograd.grad(expected_loss, logits)
    torch.testing.assert_close(actual_loss, expected_loss, atol=1e-12, rtol=0)
    torch.testing.assert_close(actual_grad, expected_grad, atol=1e-10, rtol=1e-7)


def test_all_four_heads_200_steps_equal_reference():
    torch.set_num_threads(1)
    cases = fixture_cases()
    scalar = learning.fit_pipeline(cases, 0.001)
    batch = learning.fit_pipeline(cases, 0.001, backend="batch")
    for name, a, b in [("Q", scalar.q, batch.q)] + [
        (arm, scalar.residuals[arm], batch.residuals[arm]) for arm in learning.RESIDUAL_ARMS
    ]:
        np.testing.assert_allclose(
            a.coefficients, b.coefficients, atol=1e-8, rtol=1e-6, err_msg=name
        )
        np.testing.assert_allclose(a.final_loss, b.final_loss, atol=1e-10, rtol=0)
        assert len(b.trace) == 200
    for arm in ("Q", *learning.RESIDUAL_ARMS):
        a, b = learning.predict(scalar, cases, arm), learning.predict(batch, cases, arm)
        for key in a:
            np.testing.assert_allclose(a[key], b[key], atol=1e-8, rtol=0)
            np.testing.assert_array_equal(a[key].argmax(-1), b[key].argmax(-1))


def test_unknown_backend():
    with pytest.raises(ValueError, match="backend"):
        learning.fit_pipeline(fixture_cases(people=1), 0.001, backend="automatic")


def test_mismatched_case_gram_samples_rejected():
    cases = fixture_cases(people=1)
    with pytest.raises(ValueError, match="sample counts"):
        TaskBatch((replace(cases[0], samples=99),))
