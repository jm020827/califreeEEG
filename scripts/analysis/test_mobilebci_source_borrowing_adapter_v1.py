"""Small generated sufficient-statistic fixture and role/math tests; no human files."""

import unittest

import numpy as np
import torch

from cfeg.analysis.mobilebci_features import query_statistics, references
from cfeg.analysis.mobilebci_prior_learning import tensor
from cfeg.analysis.mobilebci_reference_ridge import fit_reference_ridge, reference_projection_score
from cfeg.analysis.mobilebci_source_borrowing import (
    KS,
    SPEEDS,
    Dataset,
    balanced_probability_loss,
    folds,
    select_sources,
)
from cfeg.analysis.source_expert_borrowing import support_compatibility

torch.set_num_threads(1)


def fixture():
    rng = np.random.default_rng(20260912)
    runs = [{"subject": f"g{i:02d}", "speed": v} for i in range(16) for v in SPEEDS]
    n, samples = len(runs), 32
    y = references(128, samples)
    arrays = {
        "support_cov": np.empty((n, 4, 3, 9, 9)),
        "support_cross": np.empty((n, 4, 3, 9, 6)),
        "q": rng.normal(size=(n, 4, 54)),
        "q2": rng.normal(size=(n, 4, 2)),
        "m": rng.normal(size=(n, 4, 2)),
        "common": np.empty((n, 4, 5)),
    }
    query = []
    for i, run in enumerate(runs):
        for ki, k in enumerate(KS):
            arrays["common"][i, ki] = [float(run["speed"]), i % 3, k, 3 * k + 1, 9 * (3 * k + 1)]
            for c in range(3):
                x = rng.normal(size=(9, samples))
                x[c] += (1 + ki * 0.2) * y[c, 0]
                x -= x.mean(-1, keepdims=True)
                arrays["support_cov"][i, ki, c] = x @ x.T / samples
                arrays["support_cross"][i, ki, c] = x @ y[c].T / samples
        x = rng.normal(size=(6, 9, samples))
        query.append(query_statistics(x, y))
    for key in ["query_cov", "query_cross", "reference_gram"]:
        arrays[key] = np.stack([row[key] for row in query])
    arrays["query_labels"] = np.tile([0, 0, 0, 1, 1, 2], (n, 1))
    if (
        sum(a.size for a in arrays.values()) > 200000
        or sum(a.nbytes for a in arrays.values()) > 2 * 1024 * 1024
    ):
        raise ValueError("generated_fixture_budget")
    return arrays, runs, samples


class AdapterTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.arrays, cls.runs, cls.samples = fixture()
        cls.dataset = Dataset(cls.arrays, cls.runs, cls.samples)
        cls.dataset.fit_experts()

    def test_all_folds_roles_bank_size(self):
        _, plans = folds(set(self.dataset.subjects))
        for plan in plans:
            self.assertFalse(set(plan["source"]) & set(plan["evaluation"]))
            for subject in plan["source"] + plan["evaluation"]:
                selected = select_sources(subject, plan["source"])
                self.assertEqual(len(selected), 4)
                self.assertNotIn(subject, selected)
                self.assertTrue(set(selected) <= set(plan["source"]))

    def test_feature_dimensions_and_self_real_compatibility(self):
        d = self.dataset
        inputs, bank, record = d.features(0, sorted(set(d.subjects)), 0)
        self.assertEqual(inputs["Q"].shape, (13, 123))
        for arm in ["Q2", "QM", "SHAM"]:
            self.assertEqual(inputs[arm].shape, (13, 125))
            self.assertTrue(torch.equal(inputs[arm][0, -2:], torch.zeros(2, dtype=torch.float64)))
        torch.testing.assert_close(bank[0], d.weights[0, 0])
        torch.testing.assert_close(bank[1:], d.weights[record["source_runs"], 3])
        compat = support_compatibility(
            bank,
            tensor(d.arrays["support_cov"][0, 0]),
            tensor(d.arrays["support_cross"][0, 0]),
            tensor(d.arrays["reference_gram"][0]),
            self.samples,
        )
        self.assertGreater(float(compat[0].std()), 1e-8)
        torch.testing.assert_close(inputs["Q"][0, 2], compat[0].std(correction=0))
        torch.testing.assert_close(inputs["Q"][0, 3], compat[0].min())
        torch.testing.assert_close(
            inputs["Q"][0, 64:], tensor(np.r_[d.arrays["q"][0, 0], d.arrays["common"][0, 0]])
        )

    def test_sham_preserves_joint_vectors_and_bank_across_k(self):
        d, pool = self.dataset, sorted(set(self.dataset.subjects))
        a, _, r1 = d.features(0, pool, 0)
        _, _, r5 = d.features(0, pool, 3)
        self.assertEqual(r1["source_runs"], r5["source_runs"])
        self.assertEqual(len(r1["source_runs"]), 12)
        for speed in range(3):
            ix = list(range(1 + speed, 13, 3))
            self.assertEqual(
                sorted(map(tuple, a["QM"][ix, -2:].tolist())),
                sorted(map(tuple, a["SHAM"][ix, -2:].tolist())),
            )
        self.assertGreater(r1["sham_aux_max_change"], 0)
        for source, donor in zip(r1["source_runs"], r1["donor_runs"]):
            self.assertNotEqual(d.subjects[source], d.subjects[donor])
            self.assertEqual(d.speeds[source], d.speeds[donor])

    def test_own_query_labels_not_features(self):
        d, pool = self.dataset, sorted(set(self.dataset.subjects))
        before, _, _ = d.features(0, pool, 0)
        old = d.arrays["query_labels"][0].copy()
        d.arrays["query_labels"][0] = (old + 1) % 3
        try:
            after, _, _ = d.features(0, pool, 0)
            for arm in before:
                torch.testing.assert_close(before[arm], after[arm])
        finally:
            d.arrays["query_labels"][0] = old

    def test_balanced_probability_loss_unequal_counts(self):
        p = tensor([[[0.8, 0.1, 0.1], [0.6, 0.2, 0.2], [0.3, 0.5, 0.2], [0.2, 0.2, 0.6]]])
        labels = torch.tensor([[0, 0, 1, 2]])
        expected = -(np.log(0.8) / 2 + np.log(0.6) / 2 + np.log(0.5) + np.log(0.6)) / 3
        self.assertAlmostEqual(float(balanced_probability_loss(p, labels)), expected, places=12)

    def test_literal_ridge_and_query_projection_500samples(self):
        rng = np.random.default_rng(44)
        y = tensor(references(500, 500))
        x = tensor(rng.normal(size=(5, 9, 500)))
        x = x - x.mean(-1, keepdim=True)
        for k in [1, 5]:
            cov = torch.einsum("nct,ndt->cd", x[:k], x[:k]) / (k * 500)
            xy = torch.einsum("nct,ht->ch", x[:k], y[0]) / (k * 500)
            w = torch.linalg.solve(
                cov + 0.1 * cov.diagonal().mean() * torch.eye(9, dtype=torch.float64), xy
            )
            literal = fit_reference_ridge(x[:k], y[0], torch.eye(9, dtype=torch.float64), 0.1)
            torch.testing.assert_close(w, literal, atol=1e-11, rtol=1e-11)
            from cfeg.analysis.source_expert_borrowing import expert_scores

            qs = query_statistics(x[:1].numpy(), y.numpy())
            bank = w[None, None].repeat(13, 3, 1, 1)
            scores, _ = expert_scores(
                bank,
                tensor(qs["query_cov"][None]),
                tensor(qs["query_cross"][None]),
                tensor(qs["reference_gram"]),
            )
            torch.testing.assert_close(
                scores[0, 0, 0, 0],
                reference_projection_score(w, x[0], y[0]),
                atol=1e-11,
                rtol=1e-11,
            )

    def test_invalid_role_and_unchanged_sham(self):
        with self.assertRaisesRegex(ValueError, "fewer_than_four"):
            select_sources("a", ["a", "b", "c", "d"])
        d = self.dataset
        old = d.arrays["m"].copy()
        d.arrays["m"][:] = 0
        try:
            with self.assertRaisesRegex(ValueError, "sham_aux_unchanged"):
                d.features(0, sorted(set(d.subjects)), 0)
        finally:
            d.arrays["m"][:] = old


if __name__ == "__main__":
    unittest.main()
