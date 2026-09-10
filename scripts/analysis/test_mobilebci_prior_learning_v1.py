"""Generated-only batched learner and nested selection checks."""

import json
import unittest

import numpy as np
import torch
from run_mobilebci_prior_efficacy_v1 import Dataset, choose_source, pipeline, summarize

from cfeg.analysis.mobilebci_features import query_statistics, references, support_statistics
from cfeg.analysis.mobilebci_prior_learning import (
    balanced_loss,
    cca_scores,
    donor_indices,
    fit_head,
    new_head,
    prior,
    projection_scores,
    tensor,
)
from cfeg.analysis.mobilebci_reference_ridge import fit_reference_ridge, reference_projection_score

torch.set_num_threads(1)


class LearnerTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(20260911)
        self.reference = references(128, 80)
        self.x = rng.normal(size=(3, 9, 80))
        self.query = rng.normal(size=(6, 9, 80))
        for c in range(3):
            self.x[c] += self.reference[c][0]
            self.query[c] += self.reference[c][0]
            self.query[c + 3] += self.reference[c][0]
        self.motion = rng.normal(size=(3, 3, 128))
        self.stats, _ = support_statistics(self.x, [0, 1, 2], self.motion, self.reference)
        self.qstats = query_statistics(self.query, self.reference)
        self.data = tuple(
            tensor(value[None])
            for value in [
                self.stats["support_cov"],
                self.stats["support_cross"],
                self.qstats["query_cov"],
                self.qstats["query_cross"],
                self.qstats["reference_gram"],
            ]
        ) + (torch.tensor([[0, 1, 2, 0, 1, 2]]),)

    def test_batched_scores_equal_verified_primitive(self):
        diagonal = tensor([[0.3, 0.4, 0.5, 0.6, 0.7, 1.0, 1.5, 2.0, 2.0]])
        score, _ = projection_scores(*self.data[:5], diagonal, 0.1)
        for c in range(3):
            w = fit_reference_ridge(
                tensor(self.x[c : c + 1]), tensor(self.reference[c]), torch.diag(diagonal[0]), 0.1
            )
            for n in range(6):
                expected = reference_projection_score(
                    w, tensor(self.query[n]), tensor(self.reference[c])
                )
                torch.testing.assert_close(score[0, n, c], expected, atol=1e-11, rtol=1e-11)

    def test_batched_gradcheck(self):
        diagonal = torch.ones((1, 9), dtype=torch.float64, requires_grad=True)
        self.assertTrue(
            torch.autograd.gradcheck(
                lambda d: projection_scores(*self.data[:5], d, 0.1)[0], (diagonal,), atol=1e-5
            )
        )

    def test_balanced_loss_equals_manual(self):
        scores = torch.arange(18, dtype=torch.float64).reshape(1, 6, 3) / 20
        labels = torch.tensor([[0, 0, 0, 1, 2, 2]])
        loss = balanced_loss(scores, labels)
        ce = torch.nn.functional.cross_entropy(10 * scores[0], labels[0], reduction="none")
        torch.testing.assert_close(
            loss, torch.stack([ce[labels[0] == c].mean() for c in range(3)]).mean()
        )

    def test_training_and_frozen_q(self):
        inputs = tensor([[0.1, 0.2]])
        qhead, qlog = fit_head(inputs, self.data, 0.1, steps=3)
        qdiag = prior(qhead, inputs)
        old = qhead.layer.weight.clone()
        metadata_inputs = tensor([[0.1, 0.2, 0.7, -0.1]])
        mhead, mlog = fit_head(metadata_inputs, self.data, 0.1, qdiag, steps=3)
        torch.testing.assert_close(qhead.layer.weight, old)
        self.assertFalse(qdiag.requires_grad)
        self.assertEqual(qlog["steps"], 3)
        self.assertGreater(mlog["residual_max_abs"], 0)
        d = prior(mhead, metadata_inputs, qdiag)
        torch.testing.assert_close(d.sum(-1), torch.tensor([9.0], dtype=torch.float64))
        self.assertGreaterEqual(float(d.min()), 0.15)

    def test_source_only_standardizer_and_constant_feature(self):
        inputs = tensor([[1.0, 4.0], [3.0, 4.0]])
        head = new_head(inputs)
        torch.testing.assert_close(head.mean, tensor([2.0, 4.0]))
        torch.testing.assert_close(head.std, tensor([1.0, 1e-12]))
        self.assertEqual(head.constant_features, 1)
        head.logits(tensor([[10000.0, -1000.0]]))
        torch.testing.assert_close(head.mean, tensor([2.0, 4.0]))

    def test_donors_are_nonself_same_condition_source_only(self):
        subjects = ["a", "b", "c", "d", "a", "b", "c", "d"]
        speeds = ["0.0"] * 4 + ["0.8"] * 4
        common = np.array([[0, 0, 1, 6, 50]] * 8, dtype=float)
        source = [0, 1, 4, 5]
        donors, distances = donor_indices(source, list(range(8)), subjects, speeds, common)
        for recipient, donor in enumerate(donors):
            self.assertIn(donor, source)
            self.assertNotEqual(subjects[recipient], subjects[donor])
            self.assertEqual(speeds[recipient], speeds[donor])
        self.assertEqual(distances, [0.0] * 8)

    def test_cca_label_free_and_scale_invariant(self):
        scores = cca_scores(*self.data[2:5])
        scaled = cca_scores(self.data[2] * 100, self.data[3] * 10, self.data[4])
        torch.testing.assert_close(scores, scaled, atol=1e-11, rtol=1e-11)
        self.assertTrue(bool((scores >= 0).all() and (scores <= 1).all()))

    def test_source_selection_weighting_tie_and_no_target_hit(self):
        rows = []
        for outer in range(3):
            for inner in range(2):
                for k in [1, 2, 3, 5]:
                    for ridge in [0.01, 0.1, 1.0]:
                        # Unequal folds; pooled 3 observations must get equal weights.
                        score = 0.9 if inner == 0 else 0.6
                        outputs = {
                            arm: [{"balanced_accuracy": score if arm == "Q" else 0.5}]
                            * (1 if inner == 0 else 2)
                            for arm in ["Q", "Q2", "QM", "SHAM"]
                        }
                        rows.append(
                            {
                                "outer": outer,
                                "inner": inner,
                                "k": k,
                                "ridge": ridge,
                                "outputs": outputs,
                            }
                        )
        choices = choose_source(rows)
        for choice in choices.values():
            self.assertEqual(choice["ridge_by_k"]["1"], 1.0)
            self.assertAlmostEqual(choice["inner_oof_means"]["Q"]["1"], 0.7)
            self.assertEqual(choice["policy"]["Q"], {"k": 5, "source_target_unmet": True})

    def test_inputs_do_not_use_query_labels_or_target_m_for_sham(self):
        rng = np.random.default_rng(7)
        arrays = {
            "q": rng.normal(size=(3, 4, 54)),
            "common": np.array([[[0.0, 0.0, k, 6, 50] for k in [1, 2, 3, 5]]] * 3),
            "m": rng.normal(size=(3, 4, 2)),
            "q2": rng.normal(size=(3, 4, 2)),
            "query_labels": np.zeros((3, 6)),
        }
        dataset = Dataset(arrays, [{"subject": s, "speed": "0.0"} for s in ["a", "b", "test"]])
        actual, _ = dataset.inputs([2], [0, 1], 0, "SHAM")
        arrays["m"][2] += 1000
        arrays["query_labels"][:] = 2
        changed, _ = dataset.inputs([2], [0, 1], 0, "SHAM")
        torch.testing.assert_close(changed, actual)

    def test_generated_four_arm_pipeline(self):
        rng = np.random.default_rng(7)
        arrays = {
            key: np.tile(value[None, None], (3, 4) + (1,) * value.ndim)
            for key, value in self.stats.items()
        }
        arrays.update(
            {key: np.repeat(value[None], 3, axis=0) for key, value in self.qstats.items()}
        )
        arrays["common"] = np.array([[[0.0, 0.0, k, 6, 50] for k in [1, 2, 3, 5]]] * 3)
        arrays["m"] = rng.normal(size=(3, 4, 2))
        arrays["query_labels"] = np.tile([0, 1, 2, 0, 1, 2], (3, 1))
        dataset = Dataset(arrays, [{"subject": s, "speed": "0.0"} for s in ["a", "b", "test"]])
        state = {"fits_attempted": 0, "fits_completed": 0, "optimizer_steps": 0}
        logs = []
        outputs, _ = pipeline(
            dataset, [0, 1], [2], 0, 0.1, {"phase": "generated"}, state, logs.append
        )
        self.assertEqual(state["fits_completed"], 4)
        self.assertEqual(state["optimizer_steps"], 400)
        self.assertEqual(set(outputs), {"Q", "Q2", "QM", "SHAM"})
        qm = next(row for row in logs if row["event"] == "fit_complete" and row["arm"] == "QM")
        self.assertIn("metadata_only_intervention", qm["diagnostics"])
        json.dumps(logs, allow_nan=False)

    def test_summary_policy_cost_and_json_serialization(self):
        subjects = [f"s{i:02}" for i in range(16)]
        runs = [{"subject": s, "speed": v} for s in subjects for v in ["0.0", "0.8", "1.6"]]
        common = np.array([[[0.0, 0.0, k, 3 * k + 2, 10 * k] for k in [1, 2, 3, 5]]] * 48)
        dataset = Dataset({"common": common}, runs)
        folds = {s: i % 3 for i, s in enumerate(subjects)}
        choices = {
            str(f): {
                "policy": {
                    a: {"k": 1 if a == "QM" else 2, "source_target_unmet": False}
                    for a in ["Q", "Q2", "QM", "SHAM"]
                }
            }
            for f in range(3)
        }
        accuracies = {"Q": 0.75, "Q2": 0.76, "QM": 0.81, "SHAM": 0.77, "IDENTITY": 0.82}
        rows = [
            run | {"k": k, "arm": a, "balanced_accuracy": score}
            for run in runs
            for k in [1, 2, 3, 5]
            for a, score in accuracies.items()
        ]
        rows += [run | {"k": 0, "arm": "ZERO_CCA", "balanced_accuracy": 0.8} for run in runs]
        result = summarize(rows, choices, dataset, folds)
        self.assertAlmostEqual(
            result["policy_qm_minus_q"]["prefix_cost_reduction_fraction"], 1 - 5 / 8
        )
        self.assertAlmostEqual(result["identity_at_qm_policy_budget_mean_bacc"], 0.82)
        self.assertFalse(result["qm_policy_beats_same_budget_identity"])
        json.dumps(result, allow_nan=False)


if __name__ == "__main__":
    unittest.main()
