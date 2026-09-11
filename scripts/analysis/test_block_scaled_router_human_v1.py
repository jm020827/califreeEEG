"""Bounded generated full-flow and nontraining guards; never calls human run()."""

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import run_block_scaled_router_human_v1 as runner
import torch
from test_mobilebci_source_borrowing_adapter_v1 import fixture

from cfeg.analysis.block_scaled_router import BlockRouter, accept_proposal, train
from cfeg.analysis.mobilebci_source_borrowing import (
    Dataset,
    balanced_probability_loss,
    choose_policy,
)


class HumanRunnerTests(unittest.TestCase):
    def test_generated_full_flow(self):
        arrays, runs, samples = fixture()
        d = Dataset(arrays, runs, samples)
        state = runner.fresh_state()
        manifest = runner.role_manifest(d)
        self.assertEqual(sum(map(len, manifest["entries"].values())), 1344)
        self.assertIsNone(d.weights)
        d.fit_experts()
        logs, trained, locked = [], [], []
        original_batch = d.batch

        def batch(recipients, source, ki):
            evaluation = all(d.subjects[i] not in source for i in recipients)
            if evaluation:
                self.assertEqual(logs[-1]["event"], "four_heads_frozen_before_evaluation")
                self.assertEqual(state["fits_completed"] % 4, 0)
                self.assertTrue(
                    all(not p.requires_grad for r in trained[-4:] for p in r.layer.parameters())
                )
                if len(source) >= 10:
                    self.assertEqual(len(locked), 1)
            return original_batch(recipients, source, ki)

        def tracked_train(inputs, *args, **kwargs):
            model, info = train(inputs, *args, **kwargs)
            expected = inputs.flatten(0, 1)
            torch.testing.assert_close(model.mean, expected.mean(0), rtol=0, atol=0)
            torch.testing.assert_close(
                model.std, expected.std(0, correction=0).clamp_min(1e-12), rtol=0, atol=0
            )
            copy = BlockRouter.from_state(model.state())
            torch.testing.assert_close(model.weights(inputs), copy.weights(inputs), rtol=0, atol=0)
            trained.append(model)
            return model, info

        zero = [
            r
            | {
                "arm": "ZERO_CCA",
                "k": 0,
                "labels": arrays["query_labels"][i].tolist(),
                "predictions": [0] * 6,
                "balanced_accuracy": 1 / 3,
            }
            for i, r in enumerate(runs)
        ]
        with tempfile.TemporaryDirectory(prefix="safe-human-mock-") as directory:
            role_path = Path(directory) / "roles.json"
            choice_path = Path(directory) / "choices.json"
            runner.write_json(role_path, manifest)
            digest = runner.sha(role_path)

            def lock(value):
                self.assertEqual(state["fits_completed"], 96)
                self.assertEqual(state["proposals_completed"], 192)
                self.assertFalse(state["outer_outcomes"])
                runner.write_json(choice_path, value)
                locked.append(value)
                return runner.sha(choice_path)

            with (
                patch.object(d, "batch", side_effect=batch),
                patch.object(runner, "train", side_effect=tracked_train),
            ):
                runner.execute(d, state, logs.append, lock, zero, manifest, digest, steps=2)
            self.assertEqual(runner.sha(role_path), digest)
            self.assertEqual(json.loads(role_path.read_text()), manifest)
        self.assertEqual(state["fits_attempted"], 144)
        self.assertEqual(state["fits_completed"], 144)
        self.assertEqual(state["proposals_attempted"], 288)
        self.assertEqual(state["proposals_completed"], 288)
        self.assertEqual(state["accepted"] + state["rejected"], 288)
        self.assertLessEqual(state["candidate_checks"], 2304)
        self.assertEqual(state["candidate_checks"], state["candidate_loss_evaluations"])
        self.assertEqual(state["completed_fit_ordinary_loss_evaluations"], 576)
        self.assertEqual(state["role_contexts_verified"], 72)
        self.assertEqual(len(state["outer_outcomes"]), 1200)
        self.assertEqual(sum(r["event"] == "evaluation_complete" for r in logs), 144)
        self.assertEqual(sum(r["event"] == "four_heads_frozen_before_evaluation" for r in logs), 36)
        self.assertEqual(
            sum(len(r["training"]["trajectory"]) for r in logs if r["event"] == "fit_complete"), 288
        )
        self.assertEqual(state["status"], "NUMERIC_COMPLETE_AWAITING_PUBLICATION")
        # Generated reports are not human efficacy artifacts.
        self.assertLess(len(json.dumps(state, allow_nan=False)), 8 * 1024**2)

    def test_policy_and_duplicate_guard(self):
        rows = [
            {
                "outer": o,
                "k": k,
                "arm": a,
                "balanced_accuracy": 0.8 if a == "QM" and k >= 2 else 0.7,
            }
            for o in range(3)
            for k in (1, 2, 3, 5)
            for a in ("Q", "Q2", "QM", "SHAM")
        ]
        for value in choose_policy(rows).values():
            self.assertEqual(value["policy"]["QM"], {"k": 2, "source_target_unmet": False})
            self.assertEqual(value["policy"]["Q"], {"k": 5, "source_target_unmet": True})
        manifest = {
            "plans": [{"phase": "outer", "outer": 0}],
            "entries": {f"outer-0-all-{k}-evaluation": [{"run": 0}] for k in (1, 2, 3, 5)},
        }
        outcomes = [{"outer": 0, "k": k, "arm": "Q", "run": 0} for k in (1, 2, 3, 5)]
        runner.check_outcomes(outcomes, manifest, "outer", ["Q"])
        with self.assertRaisesRegex(ValueError, "outcome_coverage"):
            runner.check_outcomes(outcomes + outcomes[:1], manifest, "outer", ["Q"])

    def test_atomic_no_overwrite_and_failure(self):
        with tempfile.TemporaryDirectory(prefix="safe-human-atomic-") as directory:
            p = Path(directory) / "report.json"
            with (
                patch("run_mobilebci_source_borrowing_v1.os.link", side_effect=OSError("injected")),
                self.assertRaises(OSError),
            ):
                runner.write_json(p, {"status": "COMPLETE"})
            self.assertFalse(p.exists())
            runner.write_json(p, {"status": "STOPPED"})
            with self.assertRaises(FileExistsError):
                runner.write_json(p, {"status": "COMPLETE"})
            self.assertEqual(json.loads(p.read_text())["status"], "STOPPED")

    def test_missing_pin_stops_before_io(self):
        config = {
            key: "/not-human"
            for key in ("extraction_report_path", "cca_result_path", "unit_receipt_path")
        }
        config["pinned_files"] = {}
        with (
            patch.object(runner, "sha", side_effect=AssertionError("unexpected_io")),
            self.assertRaisesRegex(ValueError, "missing_required_pin"),
        ):
            runner.validate_inputs(config)

    def test_balanced_loss_and_anchor(self):
        labels = torch.tensor([[0, 0, 0, 1, 1, 2]])
        p = torch.tensor([[[0.6, 0.2, 0.2]] * 6], dtype=torch.float64)
        expected = -(np.log(0.6) + 2 * np.log(0.2)) / 3
        self.assertAlmostEqual(float(balanced_probability_loss(p, labels)), expected)
        d = type(
            "D",
            (),
            {
                "runs": [0],
                "lookup": {("g", "0.0"): 0},
                "arrays": {"query_labels": np.array([[0, 1, 2]])},
            },
        )()
        runner.validate_anchor(d, [{"subject": "g", "speed": "0.0", "labels": [0, 1, 2]}])
        with self.assertRaisesRegex(ValueError, "cca_anchor_label_mismatch"):
            runner.validate_anchor(d, [{"subject": "g", "speed": "0.0", "labels": [1, 0, 2]}])

    def test_rejected_proposal_attempts_restore(self):
        layer = torch.nn.Linear(1, 1, dtype=torch.float64)
        old = [p.detach().clone() for p in layer.parameters()]
        proposed = [p + 2 for p in old]
        calls = []
        receipt = accept_proposal(
            layer,
            old,
            proposed,
            lambda: (torch.tensor(2.0), torch.zeros(1)),
            1.0,
            torch.zeros(1),
            calls.append,
        )
        self.assertFalse(receipt["accepted"])
        self.assertEqual(calls.count("candidate"), 8)
        self.assertEqual(calls.count("loss"), 8)
        for p, original in zip(layer.parameters(), old):
            torch.testing.assert_close(p, original, rtol=0, atol=0)


if __name__ == "__main__":
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    unittest.main()
