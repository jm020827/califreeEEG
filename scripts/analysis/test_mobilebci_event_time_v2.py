"""Tiny generated-only regression coverage; never reads a human file."""

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
from read_mobilebci_event_time_v2 import Projector, inspect, render_bounded
from scipy.io import loadmat, savemat

LIMITS = {
    "visited_units": 8192,
    "string_bytes": 8192,
    "nesting_depth": 5,
    "seconds_per_file": 10,
    "output_bytes": 65536,
}


class ProjectionTests(unittest.TestCase):
    def test_numeric_shape(self):
        node = Projector(LIMITS).project(np.arange(6).reshape(2, 3), "event")
        self.assertEqual(node["shape"], [2, 3])
        self.assertEqual(node["values"], [[0, 1, 2], [3, 4, 5]])

    def test_cell_and_struct_list(self):
        cell = np.empty(2, dtype=object)
        cell[0], cell[1] = np.array([1, 2]), {"type": ["A", "B"]}
        node = Projector(LIMITS).project(cell, "event")
        self.assertEqual(node["kind"], "cell")
        self.assertEqual(node["items"][1]["fields"]["type"]["items"][0]["value"], "A")

    def test_scalar_object_array(self):
        value = np.empty((), dtype=object)
        value[()] = "head"
        p = Projector(LIMITS)
        node = p.project(value, "event")
        self.assertEqual(node["shape"], [])
        self.assertEqual(node["items"][0]["value"], "head")
        self.assertEqual(p.visited_units, 2)

    def test_empty_scalar_ragged(self):
        p = Projector(LIMITS)
        self.assertEqual(p.project(np.empty((0, 2)), "empty")["shape"], [0, 2])
        self.assertEqual(p.project(np.array(11), "code")["values"], 11)
        self.assertEqual(len(p.project(([1], [2, 3]), "ragged")["items"]), 2)

    def test_bytes_unicode(self):
        p = Projector(LIMITS)
        self.assertEqual(p.project(b"head", "label")["value"], "head")
        self.assertEqual(p.project(np.array([b"H", b"L"]), "labels")["values"], ["H", "L"])
        self.assertEqual(p.project("머리", "label")["value"], "머리")

    def test_bad_utf8(self):
        with self.assertRaises(UnicodeDecodeError):
            Projector(LIMITS).project(b"\xff", "label")

    def test_no_arbitrary_coercion(self):
        class Bad:
            def __array__(self):
                raise AssertionError("must_not_coerce")

        p = Projector(LIMITS)
        with self.assertRaisesRegex(ValueError, "unsupported_leaf_type"):
            p.project({"bad": Bad()}, "event")
        self.assertEqual(p.context["path"], "event.bad")

    def test_complex_nonfinite(self):
        for value in [np.array([1j]), complex(1, 2), float("inf"), np.array([np.nan])]:
            with self.subTest(value_type=type(value).__name__), self.assertRaises(ValueError):
                Projector(LIMITS).project(value, "event")

    def test_depth_and_cycle(self):
        cyclic = []
        cyclic.append(cyclic)
        with self.assertRaisesRegex(ValueError, "depth_limit"):
            Projector(LIMITS).project(cyclic, "event")

    def test_visited_and_string_caps(self):
        with self.assertRaisesRegex(ValueError, "visited_limit"):
            Projector({**LIMITS, "visited_units": 3}).project(np.arange(3), "event")
        with self.assertRaisesRegex(ValueError, "string_limit"):
            Projector({**LIMITS, "string_bytes": 2}).project("abc", "event")

    def test_field_limits(self):
        for value in [{str(i): i for i in range(65)}, {"x" * 129: 0}]:
            with self.assertRaises(ValueError):
                Projector(LIMITS).project(value, "event")

    def test_output_fallback(self):
        for content in ["x" * 2000, object()]:
            record = {
                "status": "COMPLETE_EVENT_TIME_METADATA_ONLY",
                "records": [{"path": "fixture", "sha256": "0" * 64, "metadata": content}],
            }
            result = json.loads(render_bounded(record, 512))
            self.assertEqual(result["status"], "STOPPED")
            self.assertTrue(result["values_omitted"])


class IntegrationTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix="mobilebci-event-v2-generated-")
        self.addCleanup(self.tmp.cleanup)
        self.files = []
        cell = np.empty(3, dtype=object)
        cell[:] = ["5.45", "8.57", "12"]
        for i in range(2):
            path = Path(self.tmp.name) / f"generated{i}.mat"
            savemat(
                path,
                {
                    "event": {
                        "type": cell,
                        "sample": np.array([1, 10, 20]),
                        "nested": [{"a": 1}, {"a": 2}],
                    },
                    "t": np.arange(4) / 100,
                    "raw_x": np.ones((2, 2)),
                },
            )
            payload = path.read_bytes()
            self.assertLessEqual(len(payload), 65536)
            self.files.append(
                {
                    "path": str(path),
                    "bytes": len(payload),
                    "sha256": hashlib.sha256(payload).hexdigest(),
                }
            )
        self.config = {"variables": ["event", "t"], "files": self.files, "limits": LIMITS}

    def test_selective_nested_roundtrip(self):
        calls = []

        def loader(path, **kwargs):
            self.assertEqual(kwargs, {"variable_names": ["event", "t"], "simplify_cells": True})
            values = loadmat(path, **kwargs)
            self.assertNotIn("raw_x", values)
            calls.append(str(path))
            return values

        output, ok = inspect(self.config, loader=loader)
        result = json.loads(output)
        self.assertTrue(ok)
        self.assertEqual(len(calls), 2)
        self.assertEqual(result["decode_calls"], 2)
        self.assertEqual(
            result["records"][0]["metadata"]["event"]["fields"]["type"]["items"][2]["value"], "12"
        )

    def test_checksum_stops_before_decode(self):
        self.files[0]["sha256"] = "0" * 64
        output, ok = inspect(self.config, loader=lambda *a, **k: self.fail("decoder_called"))
        self.assertFalse(ok)
        self.assertEqual(json.loads(output)["decode_calls"], 0)

    def test_first_bad_value_stops_pair_and_reports_path(self):
        calls = []

        def loader(*args, **kwargs):
            calls.append(1)
            return {"event": {"nested": np.array([1j])}, "t": np.array([0, 1])}

        output, ok = inspect(self.config, loader=loader)
        result = json.loads(output)
        self.assertFalse(ok)
        self.assertEqual(calls, [1])
        self.assertEqual(
            result["error_context"],
            {"path": "event.nested", "python_type": "ndarray", "dtype": "complex128", "shape": [1]},
        )

    def test_unexpected_waveform_key_stops(self):
        output, ok = inspect(
            self.config, loader=lambda *a, **k: {"event": {}, "t": [1], "raw_x": [1]}
        )
        self.assertFalse(ok)
        self.assertEqual(json.loads(output)["error"], "unexpected_fields")

    def test_post_stat_mutation(self):
        def loader(path, **kwargs):
            values = loadmat(path, **kwargs)
            # This is an isolated generated fixture, never a human file.
            with path.open("ab") as generated_fixture:
                generated_fixture.write(b"x")
            return values

        output, ok = inspect(self.config, loader=loader)
        result = json.loads(output)
        self.assertFalse(ok)
        self.assertEqual(result["decode_calls"], 1)
        self.assertEqual(result["error"], "input_changed")
        self.assertEqual(result["error_context"]["phase"], "post_stat")

    def test_minimum_output_config(self):
        self.config["limits"] = {**LIMITS, "output_bytes": 100}
        with self.assertRaisesRegex(ValueError, "output_limit_config"):
            inspect(self.config, loader=lambda *a, **k: self.fail("decoder_called"))


if __name__ == "__main__":
    unittest.main()
