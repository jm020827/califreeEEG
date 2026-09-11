"""Nontraining boundary tests. No optimizer updates, files, or human data."""

import unittest

import torch

from cfeg.analysis.block_scaled_router import BlockRouter, accept_proposal
from cfeg.analysis.mobilebci_source_borrowing import balanced_probability_loss


class BoundaryTests(unittest.TestCase):
    def test_after_scaler_block_sizes_and_roundtrip(self):
        x = torch.arange(2 * 13 * 125, dtype=torch.float64).reshape(2, 13, 125)
        model = BlockRouter(x)
        with torch.no_grad():
            model.layer.weight.fill_(0.1)
        expected = ((x - model.mean) / model.std * model.scale).sum(-1) * 0.1
        torch.testing.assert_close(model.logits(x), expected)
        self.assertEqual(model.scale[0], 1 / 5)
        self.assertEqual(model.scale[64], 1 / 59)
        self.assertEqual(model.scale[124], 1 / 2)
        torch.testing.assert_close(
            model.weights(x), BlockRouter.from_state(model.state()).weights(x)
        )

    def proposal(self, value):
        layer = torch.nn.Linear(1, 1, bias=False, dtype=torch.float64)
        with torch.no_grad():
            layer.weight.zero_()

        def evaluate():
            w = layer.weight.squeeze()
            return (w - 1).square(), torch.stack((w, -w))

        old = [layer.weight.detach().clone()]
        receipt = accept_proposal(
            layer,
            old,
            [torch.full_like(layer.weight, value)],
            evaluate,
            1.0,
            torch.zeros(2, dtype=torch.float64),
        )
        return layer, receipt

    def test_full_accept(self):
        layer, r = self.proposal(0.5)
        self.assertEqual(r["factor"], 1)
        self.assertEqual(float(layer.weight), 0.5)

    def test_shrink_accept(self):
        layer, r = self.proposal(4)
        self.assertEqual(r["factor"], 0.25)
        self.assertEqual(float(layer.weight), 1)

    def test_reject_restore(self):
        layer, r = self.proposal(-1)
        self.assertFalse(r["accepted"])
        self.assertEqual(len(r["trials"]), 8)
        self.assertEqual(float(layer.weight), 0)

    def test_nonfinite_restore(self):
        layer, r = self.proposal(float("inf"))
        self.assertFalse(r["accepted"])
        self.assertTrue(all(t["loss"] is None for t in r["trials"]))
        self.assertEqual(float(layer.weight), 0)

    def test_input_and_class_guards(self):
        with self.assertRaisesRegex(ValueError, "expert_shape"):
            BlockRouter(torch.zeros((2, 12, 125), dtype=torch.float64))
        with self.assertRaisesRegex(ValueError, "feature_shape"):
            BlockRouter(torch.zeros((2, 13, 124), dtype=torch.float64))
        with self.assertRaisesRegex(ValueError, "query_class_missing"):
            balanced_probability_loss(
                torch.full((1, 2, 3), 1 / 3), torch.zeros((1, 2), dtype=torch.long)
            )


if __name__ == "__main__":
    torch.set_num_threads(1)
    unittest.main()
