"""Tiny generated-only checks; no public human MAT paths are opened."""

import copy
import hashlib
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import read_mobilebci_cohort_metadata_v1 as reader
from scipy.io import savemat

LIMITS = {"visited_units": 8192, "string_bytes": 8192, "nesting_depth": 5}


def values(device="IMU"):
    labels = reader.GYRO + [f"unused{i}" for i in range(24)]
    if device == "scalp":
        labels = reader.EEG + [f"unused{i}" for i in range(27)]
    codes = np.tile([1, 2, 3], 20)
    codes[[1, 3]] = codes[[3, 1]]
    return {
        "raw_fs": 128 if device == "IMU" else 500,
        "raw_clab": labels,
        "event": {
            "className": ["5.45", "8.57", "12"],
            "decs": codes,
            "y": np.eye(3, dtype=np.uint8)[:, codes - 1],
            "time": 4001 + np.arange(60) * 9000.0,
        },
        "t": np.arange(1, 401) * 10,
    }


def spec(device="IMU", speed="0.0", subject="s01"):
    return {
        "id": 1 if device == "IMU" else 2,
        "name": f"{subject}_{device}_SSVEP_{speed}.mat",
        "path": "/generated-only-placeholder",
        "sha256": "0" * 64,
        "subject": subject,
        "speed": speed,
        "device": device,
        "raw_shape": [600 * (128 if device == "IMU" else 500), 27 if device == "IMU" else 36],
    }


class MetadataTests(unittest.TestCase):
    def fixture(self, folder):
        data = values()
        data["raw_clab"] = np.array(data["raw_clab"], dtype=object)
        data["event"]["className"] = np.array(data["event"]["className"], dtype=object)
        data["raw_x"] = np.zeros((2, 2))  # Sentinel, deliberately not the declared raw shape.
        path = Path(folder) / "tiny.mat"
        savemat(path, data, do_compression=True)
        self.assertLess(path.stat().st_size, 65536)
        item = spec() | {
            "path": str(path),
            "bytes": path.stat().st_size,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
        return item

    def test_selected_fields_and_observed_onehot(self):
        with tempfile.TemporaryDirectory(prefix="mobile-metadata-generated-") as folder:
            item = self.fixture(folder)
            calls = []

            def loader(path, **kwargs):
                calls.append(kwargs)
                return reader.loadmat(path, **kwargs)

            counters = {"checksum_passes": 0, "decode_calls": 0}
            row = reader.read_record(item, LIMITS, loader, counters)
            self.assertEqual(row["issues"], [])
            self.assertTrue(row["onehot_agrees"])
            self.assertEqual(np.asarray(row["event_onehot"]).shape, (3, 60))
            self.assertEqual(calls, [{"variable_names": reader.FIELDS, "simplify_cells": True}])
            self.assertNotIn("raw_x", row)
            self.assertEqual(counters, {"checksum_passes": 1, "decode_calls": 1})

    def test_checksum_fails_before_decode(self):
        with tempfile.TemporaryDirectory(prefix="mobile-metadata-generated-") as folder:
            item = self.fixture(folder) | {"sha256": "0" * 64}
            counters = {"checksum_passes": 0, "decode_calls": 0}
            with self.assertRaisesRegex(ValueError, "input_sha256"):
                reader.read_record(item, LIMITS, counters=counters)
            self.assertEqual(counters, {"checksum_passes": 1, "decode_calls": 0})

    def test_mutation_detected(self):
        with tempfile.TemporaryDirectory(prefix="mobile-metadata-generated-") as folder:
            item = self.fixture(folder)

            def loader(path, **kwargs):
                result = reader.loadmat(path, **kwargs)
                stat = path.stat()
                os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns + 1000000))
                return result

            with self.assertRaisesRegex(ValueError, "input_changed"):
                reader.read_record(item, LIMITS, loader)

    def test_scientific_mismatches_recorded(self):
        data = values()
        data["raw_fs"] = 127
        data["event"]["className"][0] = "other"
        data["event"]["y"][:] = 0
        data["event"]["time"][0] = 1.25
        row = reader.normalize(spec(), data)
        self.assertFalse(row["onehot_agrees"])
        for issue in [
            "raw_rate",
            "class_names",
            "onehot_agreement",
            "marker_sample_grid",
            "marker_window_bounds",
        ]:
            self.assertIn(issue, row["issues"])

    def test_exact_support_and_charged_prefix(self):
        records = [reader.normalize(spec(d), values(d)) for d in ["IMU", "scalp"]]
        pairs, eligible = reader.paired_roles(records)
        self.assertEqual(eligible, [])  # Only one of three conditions supplied.
        self.assertEqual(pairs[0]["issues"], [])
        one = pairs[0]["budgets"]["1"]
        self.assertEqual(one["support_indices"], [0, 2, 3])
        self.assertEqual(one["learner_trials"], 3)
        self.assertEqual(one["acquired_prefix_trials"], 4)
        self.assertEqual(pairs[0]["query_indices"], list(range(40, 60)))

    def test_three_conditions_and_paired_mismatch(self):
        records = [
            reader.normalize(spec(d, s), values(d))
            for s in ["0.0", "0.8", "1.6"]
            for d in ["IMU", "scalp"]
        ]
        self.assertEqual(reader.paired_roles(records)[1], ["s01"])
        records[-1]["event_codes"][0] = 2
        pairs, eligible = reader.paired_roles(records)
        self.assertEqual(eligible, [])
        self.assertIn("paired_event_classes", pairs[-1]["issues"])
        with self.assertRaisesRegex(ValueError, "duplicate_device_role"):
            reader.paired_roles(records + [records[0]])

    def test_missing_query_class_and_support_overlap(self):
        records = []
        for device in ["IMU", "scalp"]:
            data = values(device)
            codes = np.repeat([1, 2, 3], 20)
            data["event"]["decs"] = codes
            data["event"]["y"] = np.eye(3)[:, codes - 1]
            records.append(reader.normalize(spec(device), data))
        issues = reader.paired_roles(records)[0][0]["issues"]
        self.assertIn("query_class_missing", issues)
        self.assertIn("support_query_overlap_k1", issues)
        self.assertIn("support_query_time_overlap_k1", issues)

    def test_negative_gap_even_with_disjoint_indices(self):
        records = [reader.normalize(spec(d), values(d)) for d in ["IMU", "scalp"]]
        for row in records:
            row["event_times_ms_candidate"] = [4001 + 10 * i for i in range(60)]
        issues = reader.paired_roles(records)[0][0]["issues"]
        self.assertIn("support_query_time_overlap_k1", issues)
        self.assertNotIn("support_query_overlap_k1", issues)

    def test_bounded_output_fallback(self):
        row = reader.normalize(spec(), values())
        result = {"records": [row], "pairs": [], "journal_path": "/generated-journal"}
        rendered = reader.bounded_report(result, 1024)
        self.assertLessEqual(len(rendered.encode()), 1024)
        self.assertEqual(result["status"], "STOPPED_OUTPUT_LIMIT")
        self.assertFalse(result["proposed_design_ready"])
        self.assertNotIn("event_codes", result["records"][0])

    def test_eof_durable_stop_and_no_restart(self):
        with tempfile.TemporaryDirectory(prefix="mobile-metadata-generated-") as folder:
            root = Path(folder)
            rows = []
            for subject in range(1, 17):
                for speed in ["0.0", "0.8", "1.6"]:
                    for device in ["IMU", "scalp"]:
                        row = spec(device, speed, f"s{subject:02}")
                        row.update(id=len(rows), path=str(root / row["name"]), bytes=1)
                        rows.append(row)
            source_rows = copy.deepcopy(rows)
            for row in source_rows:
                row["top_level_schema"] = [{"name": "raw_x", "shape": row["raw_shape"]}]
            acquisition = root / "acquisition.json"
            acquisition.write_text(json.dumps({"records": source_rows, "reused": []}))
            original = root / "pair.json"
            original.write_text('{"records": []}')
            config = {
                "files": rows,
                "acquisition_path": str(acquisition),
                "acquisition_sha256": hashlib.sha256(acquisition.read_bytes()).hexdigest(),
                "original_pair_path": str(original),
                "original_pair_sha256": hashlib.sha256(original.read_bytes()).hexdigest(),
                "output_path": str(root / "result.json"),
                "journal_path": str(root / "journal.jsonl"),
                "limits": LIMITS
                | {"batch_seconds": 600, "file_seconds": 30, "output_bytes": 4194304},
            }
            with patch.object(
                reader, "read_record", side_effect=EOFError("generated EOF")
            ) as mocked:
                reader.main(config)
                self.assertEqual(mocked.call_count, 1)
            report = json.loads(Path(config["output_path"]).read_text())
            self.assertEqual(report["status"], "STOPPED_NO_RETRY")
            self.assertEqual(report["error_type"], "EOFError")
            self.assertEqual(report["attempted_files"], 1)
            with self.assertRaisesRegex(ValueError, "no_restart"):
                reader.main(config)


if __name__ == "__main__":
    unittest.main()
