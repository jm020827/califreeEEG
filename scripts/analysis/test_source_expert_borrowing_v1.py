"""Tiny generated unit checks, not the registered constructive-capacity experiment."""

import unittest

import numpy as np
import torch

from cfeg.analysis.mobilebci_features import query_statistics, references, support_statistics
from cfeg.analysis.mobilebci_reference_ridge import reference_projection_score
from cfeg.analysis.source_expert_borrowing import (
    Router,
    expert_scores,
    fit_router,
    metadata_differences,
    mix_probabilities,
    support_compatibility,
    validate_source_subjects,
)

torch.set_num_threads(1)


class BorrowingTests(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.default_rng(19)
        self.y = references(128, 80)
        self.x = self.rng.normal(size=(3, 6, 80))
        q = query_statistics(self.x, self.y)
        self.cov = torch.tensor(q["query_cov"][None])
        self.cross = torch.tensor(q["query_cross"][None])
        self.gram = torch.tensor(q["reference_gram"])
        self.w = torch.tensor(self.rng.normal(size=(3, 3, 6, 6)))

    def test_bank_scores_equal_primitive(self):
        actual, _ = expert_scores(self.w, self.cov, self.cross, self.gram)
        for e in range(3):
            for c in range(3):
                expected = reference_projection_score(
                    self.w[e, c], torch.tensor(self.x[0]), torch.tensor(self.y[c])
                )
                torch.testing.assert_close(actual[0, e, 0, c], expected, atol=1e-11, rtol=1e-11)

    def test_support_compatibility_correct_scaling(self):
        stats, _ = support_statistics(self.x, [0, 1, 2], np.ones((3, 3, 128)), self.y)
        actual = support_compatibility(
            self.w,
            torch.tensor(stats["support_cov"]),
            torch.tensor(stats["support_cross"]),
            self.gram,
            80,
        )
        for e in range(3):
            for c in range(3):
                expected = reference_projection_score(
                    self.w[e, c], torch.tensor(self.x[c]), torch.tensor(self.y[c])
                )
                torch.testing.assert_close(actual[e, c], expected, atol=1e-11, rtol=1e-11)

    def test_probability_and_exact_no_transfer(self):
        scores, _ = expert_scores(self.w, self.cov, self.cross, self.gram)
        p = torch.softmax(10 * scores, -1)
        w = torch.tensor([[0.2, 0.3, 0.5]], dtype=torch.float64)
        actual = mix_probabilities(p, w)
        torch.testing.assert_close(actual.sum(-1), torch.ones((1, 3), dtype=torch.float64))
        self.assertTrue(torch.equal(mix_probabilities(p, w, no_transfer=True), p[:, 0]))
        with self.assertRaisesRegex(ValueError, "negative_mixture"):
            mix_probabilities(p, torch.tensor([[-1.0, 1.0, 1.0]], dtype=torch.float64))
        with self.assertRaisesRegex(ValueError, "nonfinite_mixture"):
            mix_probabilities(p, w * float("nan"))
        with self.assertRaisesRegex(ValueError, "router_must_not_have_query_axis"):
            mix_probabilities(p, w[:, :, None])

    def test_source_order_and_orthogonal_invariance(self):
        base, _ = expert_scores(self.w, self.cov, self.cross, self.gram)
        order = [0, 2, 1]
        changed, _ = expert_scores(self.w[order], self.cov, self.cross, self.gram)
        torch.testing.assert_close(changed, base[:, order])
        orthogonal = torch.linalg.qr(torch.tensor(self.rng.normal(size=(6, 6)))).Q
        changed, _ = expert_scores(self.w @ orthogonal, self.cov, self.cross, self.gram)
        scaled, _ = expert_scores(-3 * self.w, self.cov, self.cross, self.gram)
        torch.testing.assert_close(changed, base, atol=1e-10, rtol=1e-10)
        torch.testing.assert_close(scaled, base, atol=1e-10, rtol=1e-10)

    def test_self_and_heldout_sources_rejected(self):
        validate_source_subjects(["a", "b"], ["a", "b"], ["target"])
        with self.assertRaisesRegex(ValueError, "recipient_in_source_bank"):
            validate_source_subjects(["a", "b"], ["a", "b"], ["a"])
        with self.assertRaisesRegex(ValueError, "source_outside_training_pool"):
            validate_source_subjects(["a", "held"], ["a", "b"], ["target"])

    def test_pairwise_metadata_avoids_common_logit_cancellation(self):
        base = torch.tensor([[0.4, 0.2, -0.3]], dtype=torch.float64)
        torch.testing.assert_close(torch.softmax(base, -1), torch.softmax(base + 100, -1))
        m = metadata_differences(
            torch.tensor([[-1.0, 0.0], [1.0, 0.0]], dtype=torch.float64),
            torch.tensor([[-1.0, 0.0], [1.0, 0.0]], dtype=torch.float64),
        )
        w = torch.softmax(-m[:, :, 0], -1)
        self.assertGreater(float((w[0] - w[1]).abs().max()), 0.1)
        self.assertTrue(torch.equal(m[:, 0], torch.zeros((2, 2), dtype=torch.float64)))

    def test_router_training_scaler_and_no_query_feature(self):
        inputs = torch.tensor(self.rng.normal(size=(4, 3, 5)))
        probabilities = torch.softmax(torch.tensor(self.rng.normal(size=(4, 3, 3, 3))), -1)
        labels = torch.tensor([[0, 1, 2]] * 4)
        router, log = fit_router(inputs, probabilities, labels, steps=2)
        self.assertEqual(log["steps"], 2)
        torch.testing.assert_close(router.mean, inputs.mean((0, 1)))
        weights = router.weights(inputs)
        labels[:] = 2
        torch.testing.assert_close(router.weights(inputs), weights)
        order = [0, 2, 1]
        torch.testing.assert_close(router.weights(inputs[:, order]), weights[:, order])
        self.assertTrue(all(not p.requires_grad for p in router.layer.parameters()))

    def test_zero_energy_uniform_and_nonfinite(self):
        score, count = expert_scores(torch.zeros_like(self.w), self.cov, self.cross, self.gram)
        self.assertEqual(count, 27)
        torch.testing.assert_close(torch.softmax(10 * score, -1), torch.full_like(score, 1 / 3))
        with self.assertRaisesRegex(ValueError, "nonfinite_expert_input"):
            expert_scores(self.w * float("nan"), self.cov, self.cross, self.gram)
        with self.assertRaisesRegex(ValueError, "router_input"):
            Router(torch.full((2, 3, 1), float("nan"), dtype=torch.float64))

    def test_step_counter_preserves_partial_failure(self):
        inputs = torch.tensor(self.rng.normal(size=(4, 3, 5)))
        probabilities = torch.softmax(torch.tensor(self.rng.normal(size=(4, 3, 3, 3))), -1)
        labels = torch.tensor([[0, 1, 2]] * 4)
        completed = []

        def count_step():
            completed.append(1)
            if len(completed) == 2:
                raise RuntimeError("injected_after_second_update")

        with self.assertRaisesRegex(RuntimeError, "injected_after_second_update"):
            fit_router(inputs, probabilities, labels, steps=4, on_step=count_step)
        self.assertEqual(len(completed), 2)


if __name__ == "__main__":
    unittest.main()
