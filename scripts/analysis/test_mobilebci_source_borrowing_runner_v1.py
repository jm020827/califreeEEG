"""Generated full-path integration at2updates/fit; no human reader invocation."""

import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from run_mobilebci_source_borrowing_v1 import execute, run, validate_anchor, write_json
from test_mobilebci_source_borrowing_adapter_v1 import fixture

from cfeg.analysis.mobilebci_source_borrowing import Dataset, choose_policy, train_router


class RunnerTests(unittest.TestCase):
    def test_atomic_report_failure_not_published(self):
        with tempfile.TemporaryDirectory(prefix="source-borrowing-atomic-") as directory:
            path = Path(directory) / "result.json"
            with (
                patch("run_mobilebci_source_borrowing_v1.os.link", side_effect=OSError("injected")),
                self.assertRaises(OSError),
            ):
                write_json(path, {"status": "RUNNING"})
            self.assertFalse(path.exists())
            self.assertEqual(len(list(Path(directory).iterdir())), 1)
            write_json(path, {"status": "STOPPED"})
            self.assertEqual(json.loads(path.read_text())["status"], "STOPPED")

    def test_anchor_unique_run_identity(self):
        arrays, runs, samples = fixture()
        dataset = Dataset(arrays, runs, samples)
        rows = [{**r, "labels": arrays["query_labels"][i].tolist()} for i, r in enumerate(runs)]
        validate_anchor(dataset, rows)
        rows[0] = rows[1]
        with self.assertRaisesRegex(ValueError, "cca_anchor_run_set"):
            validate_anchor(dataset, rows)

    def test_missing_pin_stops_before_cache_access(self):
        with tempfile.TemporaryDirectory(prefix="source-borrowing-pin-") as directory:
            config = {
                key: str(Path(directory) / key)
                for key in [
                    "report_path",
                    "journal_path",
                    "choices_path",
                    "models_path",
                    "audit_arrays_path",
                    "start_path",
                    "extraction_report_path",
                    "cca_result_path",
                    "unit_receipt_path",
                ]
            }
            config.update(cache_path="/not-a-human-input", cache_sha256="0" * 64, pinned_files={})
            with (
                patch(
                    "run_mobilebci_source_borrowing_v1.np.load",
                    side_effect=AssertionError("must_not_read"),
                ),
                patch("builtins.print"),
            ):
                state = run(config)
            self.assertEqual(state["status"], "STOPPED_NO_RETRY")
            self.assertEqual(state["error"], "missing_required_pin")
            self.assertEqual(state["cache_checksum_passes"], 0)
            with self.assertRaisesRegex(ValueError, "no_restart"):
                run(config)

    def test_generated_full_execution_lock_and_outputs(self):
        arrays, runs, samples = fixture()
        d = Dataset(arrays, runs, samples)
        d.fit_experts()
        state = {"fits_attempted": 0, "fits_completed": 0, "optimizer_steps": 0}
        logs, locked, trained = [], [], []
        original_batch = d.batch

        def batch(recipients, source, ki):
            is_evaluation = all(d.subjects[i] not in source for i in recipients)
            if is_evaluation:
                self.assertEqual(state["fits_completed"] % 4, 0)
                self.assertTrue(
                    all(not p.requires_grad for r in trained[-4:] for p in r.layer.parameters())
                )
                if len(source) >= 10:
                    self.assertTrue(locked)
            return original_batch(recipients, source, ki)

        def train(*args, **kwargs):
            router, info = train_router(*args, **kwargs)
            trained.append(router)
            return router, info

        zero = [
            {
                "subject": r["subject"],
                "speed": r["speed"],
                "arm": "ZERO_CCA",
                "k": 0,
                "labels": arrays["query_labels"][i].tolist(),
                "predictions": [0] * 6,
                "balanced_accuracy": 1 / 3,
            }
            for i, r in enumerate(runs)
        ]
        with tempfile.TemporaryDirectory(prefix="source-borrowing-test-") as temporary:
            path = Path(temporary) / "choices.json"

            def lock(value):
                self.assertEqual(state["fits_completed"], 96)
                self.assertEqual(state["optimizer_steps"], 192)
                self.assertFalse(state["outer_outcomes"])
                write_json(path, value)
                locked.append(value)
                return hashlib.sha256(path.read_bytes()).hexdigest()

            with (
                patch.object(d, "batch", side_effect=batch),
                patch("run_mobilebci_source_borrowing_v1.train_router", side_effect=train),
            ):
                execute(d, state, logs.append, lock, zero, steps=2)
            self.assertEqual(state["fits_completed"], 144)
            self.assertEqual(state["optimizer_steps"], 288)
            self.assertEqual(len(trained), 144)
            self.assertEqual(len(state["outer_outcomes"]), 48 * 4 * 6 + 48)
            self.assertEqual(len(state["source_m_actuation"]), 36)
            self.assertEqual(len(locked), 1)
            rendered = json.dumps(state, allow_nan=False)
            self.assertLess(len(rendered), 8 * 1024 * 1024)
            self.assertTrue(all(np.isfinite(list(state["summary"]["low_k_means"].values()))))
            self.assertEqual(sum(r["event"] == "fit_complete" for r in logs), 144)
            self.assertEqual(sum(r["event"] == "evaluation_complete" for r in logs), 144)

    def test_policy_exact_threshold_and_fallback(self):
        rows = [
            {
                "outer": o,
                "k": k,
                "arm": a,
                "balanced_accuracy": 0.8 if (a == "QM" and k >= 2) else 0.7,
            }
            for o in range(3)
            for k in [1, 2, 3, 5]
            for a in ["Q", "Q2", "QM", "SHAM"]
        ]
        choices = choose_policy(rows)
        for entry in choices.values():
            self.assertEqual(entry["policy"]["QM"], {"k": 2, "source_target_unmet": False})
            self.assertEqual(entry["policy"]["Q"], {"k": 5, "source_target_unmet": True})


if __name__ == "__main__":
    unittest.main()
