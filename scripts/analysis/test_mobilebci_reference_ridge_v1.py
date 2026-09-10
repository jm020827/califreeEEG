"""Generated-only checks for one-shot ridge actuation and its expected invariances."""

import math
import unittest

import torch

from cfeg.analysis.mobilebci_reference_ridge import (
    center,
    fit_reference_ridge,
    q_diagonal,
    reference_projection_score,
    references,
    residual_diagonal,
)

torch.set_num_threads(1)


class ReferenceRidgeTests(unittest.TestCase):
    def setUp(self):
        generator = torch.Generator().manual_seed(20260911)
        self.refs = [references(f, 128, 128) for f in (5.45, 8.57)]
        a, b = self.refs[0][0], self.refs[1][0]
        self.supports = [
            torch.stack((a + 0.8 * b, 0.6 * a + b))[None],
            torch.stack((0.7 * a + b, a + 0.4 * b))[None],
        ]
        self.supports = [
            x + 0.1 * torch.randn(x.shape, generator=generator, dtype=torch.float64)
            for x in self.supports
        ]
        self.query = torch.stack((a + 0.7 * b, 0.5 * a + b)) + 0.1 * torch.randn(
            (2, 128), generator=generator, dtype=torch.float64
        )

    def scores(self, diagonal):
        prior = torch.diag(diagonal)
        return torch.stack(
            [
                reference_projection_score(fit_reference_ridge(x, y, prior, 0.3), self.query, y)
                for x, y in zip(self.supports, self.refs)
            ]
        )

    def test_k1_solve_and_units(self):
        prior = torch.eye(2, dtype=torch.float64)
        w = fit_reference_ridge(self.supports[0], self.refs[0], prior, 0.3)
        scaled = fit_reference_ridge(self.supports[0] * 100, self.refs[0], prior, 0.3)
        torch.testing.assert_close(scaled * 100, w, atol=1e-11, rtol=1e-11)
        self.assertTrue(torch.isfinite(w).all())

    def test_anisotropic_margin_actuation(self):
        base = torch.ones(2, dtype=torch.float64)
        changed = residual_diagonal(base, torch.tensor([2.0, -2.0], dtype=torch.float64))
        delta = torch.diff(self.scores(changed)) - torch.diff(self.scores(base))
        self.assertGreater(abs(float(delta)), 1e-8)

    def test_metadata_gradient_and_frozen_q(self):
        q_logits = torch.zeros(2, dtype=torch.float64, requires_grad=True)
        metadata = torch.tensor(0.5, dtype=torch.float64, requires_grad=True)
        logits = metadata * torch.tensor([1.0, -1.0], dtype=torch.float64)
        diagonal = residual_diagonal(q_diagonal(q_logits), logits)
        torch.diff(self.scores(diagonal)).sum().backward()
        self.assertIsNone(q_logits.grad)
        self.assertGreater(abs(float(metadata.grad)), 1e-8)

    def test_gradcheck(self):
        logits = torch.tensor([0.4, -0.2], dtype=torch.float64, requires_grad=True)

        def margin(value):
            return torch.diff(
                self.scores(residual_diagonal(torch.ones(2, dtype=torch.float64), value))
            )

        self.assertTrue(torch.autograd.gradcheck(margin, (logits,), eps=1e-6, atol=1e-6, rtol=1e-4))

    def test_trace_spd_and_constant_noop(self):
        q = q_diagonal(torch.tensor([-100.0, 0.0, 100.0], dtype=torch.float64))
        d = residual_diagonal(q, torch.tensor([100.0, -100.0, -100.0], dtype=torch.float64))
        torch.testing.assert_close(d.sum(), q.sum())
        self.assertGreaterEqual(float(d.min()), 0.15)
        torch.testing.assert_close(residual_diagonal(q, torch.ones(3, dtype=torch.float64)), q)

    def test_filter_scale_noop(self):
        w = fit_reference_ridge(
            self.supports[0], self.refs[0], torch.eye(2, dtype=torch.float64), 0.3
        )
        torch.testing.assert_close(
            reference_projection_score(w, self.query, self.refs[0]),
            reference_projection_score(-3 * w, self.query, self.refs[0]),
            atol=1e-12,
            rtol=1e-12,
        )

    def test_phase_rotation(self):
        theta = 0.7
        pair = torch.tensor(
            [[math.cos(theta), -math.sin(theta)], [math.sin(theta), math.cos(theta)]],
            dtype=torch.float64,
        )
        rotation = torch.block_diag(pair, pair, pair)
        y = self.refs[0]
        rotated = rotation @ y
        prior = torch.eye(2, dtype=torch.float64)
        w = fit_reference_ridge(self.supports[0], y, prior, 0.3)
        wr = fit_reference_ridge(self.supports[0], rotated, prior, 0.3)
        torch.testing.assert_close(
            reference_projection_score(w, self.query, y),
            reference_projection_score(wr, self.query, rotated),
            atol=1e-12,
            rtol=1e-12,
        )

    def test_projector_range_and_zero(self):
        y = self.refs[0]
        w = torch.eye(6, dtype=torch.float64)
        torch.testing.assert_close(
            reference_projection_score(w, y, y), torch.tensor(1.0, dtype=torch.float64)
        )
        self.assertEqual(float(reference_projection_score(w, torch.zeros_like(y), y)), 0)

    def test_singular_reference_and_bad_prior(self):
        with self.assertRaisesRegex(ValueError, "reference_gram_not_spd"):
            reference_projection_score(
                torch.eye(2, dtype=torch.float64),
                self.query,
                torch.ones((2, 128), dtype=torch.float64),
            )
        with self.assertRaisesRegex(ValueError, "prior_not_spd"):
            fit_reference_ridge(
                self.supports[0], self.refs[0], -torch.eye(2, dtype=torch.float64), 0.3
            )
        self.assertTrue(
            torch.allclose(
                center(self.query).mean(-1), torch.zeros(2, dtype=torch.float64), atol=1e-15
            )
        )


if __name__ == "__main__":
    unittest.main()
