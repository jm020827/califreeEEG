"""Generated-only extraction/statistic tests; at most three suite calls allowed."""

import unittest

import numpy as np
from scipy.signal import butter, sosfilt

from cfeg.analysis.mobilebci_features import (
    eeg_trials,
    gyro_trials,
    query_statistics,
    references,
    support_statistics,
)


class FeatureTests(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.default_rng(20260911)
        self.y = references()
        self.labels = np.array([0, 1, 2])
        self.x = self.rng.normal(size=(3, 9, 500))
        for i in range(3):
            self.x[i] += self.rng.normal(size=(9, 6)) @ self.y[i]
        self.motion = self.rng.normal(size=(3, 3, 128))

    def test_feature_shapes_and_finiteness(self):
        result, floors = support_statistics(self.x, self.labels, self.motion)
        self.assertEqual(
            {key: value.shape for key, value in result.items()},
            {
                "q": (54,),
                "q2": (2,),
                "m": (2,),
                "support_cov": (3, 9, 9),
                "support_cross": (3, 9, 6),
            },
        )
        self.assertTrue(all(np.isfinite(value).all() for value in result.values()))
        self.assertEqual(sum(floors.values()), 0)

    def test_support_stats_equal_direct_algebra(self):
        result, _ = support_statistics(self.x, self.labels, self.motion)
        x = self.x - self.x.mean(-1, keepdims=True)
        for c in range(3):
            np.testing.assert_allclose(result["support_cov"][c], x[c] @ x[c].T / 500)
            np.testing.assert_allclose(result["support_cross"][c], x[c] @ self.y[c].T / 500)
        q = query_statistics(x)
        np.testing.assert_allclose(q["query_cov"][0], x[0] @ x[0].T)
        np.testing.assert_allclose(q["query_cross"][0, 1], x[0] @ self.y[1].T)

    def test_trial_permutation_invariance(self):
        a, _ = support_statistics(self.x, self.labels, self.motion)
        order = [2, 0, 1]
        b, _ = support_statistics(self.x[order], self.labels[order], self.motion[order])
        for key in a:
            np.testing.assert_allclose(a[key], b[key], atol=1e-12)

    def test_motion_changes_no_eeg_feature(self):
        a, _ = support_statistics(self.x, self.labels, self.motion)
        b, _ = support_statistics(self.x, self.labels, self.motion * 5)
        for key in a:
            if key != "m":
                np.testing.assert_array_equal(a[key], b[key])
        np.testing.assert_allclose(b["m"] - a["m"], [np.log(5), 0], atol=1e-12)

    def test_degenerate_floors_and_invalid_support(self):
        result, floors = support_statistics(
            np.zeros_like(self.x), self.labels, np.zeros_like(self.motion)
        )
        self.assertGreater(sum(floors.values()), 0)
        self.assertTrue(all(np.isfinite(value).all() for value in result.values()))
        with self.assertRaisesRegex(ValueError, "support_class_balance"):
            support_statistics(self.x, [0, 0, 2], self.motion)
        bad = self.x.copy()
        bad[0, 0, 0] = np.nan
        with self.assertRaisesRegex(ValueError, "nonfinite_support"):
            support_statistics(bad, self.labels, self.motion)

    def test_causal_window_and_unconsumed_future(self):
        raw = self.rng.normal(size=(4000, 10))
        markers = [1000]
        actual = eeg_trials(raw, markers, [0], list(range(9)))
        sos = butter(4, [4, 40], btype="bandpass", fs=500, output="sos")
        expected = sosfilt(sos, raw[750:1750, :9], axis=0)[500:].T
        expected -= expected.mean(-1, keepdims=True)
        np.testing.assert_allclose(actual[0], expected)
        raw[1750:] = np.nan
        raw[:750] = np.nan
        raw[:, 9] = np.nan
        np.testing.assert_array_equal(eeg_trials(raw, markers, [0], list(range(9))), actual)
        with self.assertRaisesRegex(ValueError, "eeg_window_bounds"):
            eeg_trials(raw, [0], [0], list(range(9)))

    def test_gyro_only_selected_window_channels(self):
        raw = self.rng.normal(size=(1500, 6))
        actual = gyro_trials(raw, [256], [0], [0, 2, 4])
        np.testing.assert_array_equal(actual[0], raw[320:448, [0, 2, 4]].T)
        raw[:320] = np.nan
        raw[448:] = np.nan
        raw[:, [1, 3, 5]] = np.nan
        np.testing.assert_array_equal(gyro_trials(raw, [256], [0], [0, 2, 4]), actual)

    def test_query_nonfinite_and_overflow_stop(self):
        with self.assertRaisesRegex(ValueError, "nonfinite_query_input"):
            query_statistics(np.full_like(self.x, np.nan))
        with (
            np.errstate(over="ignore", invalid="ignore"),
            self.assertRaisesRegex(ValueError, "nonfinite_query_statistics"),
        ):
            query_statistics(self.x * 1e200)


if __name__ == "__main__":
    unittest.main()
