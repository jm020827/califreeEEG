"""Generated signal algebra only: no files, participant data or gate fitting."""

import numpy as np
import pytest

from cfeg.mamem_signal_v1 import analyze_window


def raw_window(window, start=37, tail=31):
    raw = np.zeros((257, start + 500 + tail), dtype=np.float64)
    raw[:256, start:start + 500] = window
    return raw, start, start + 500


def noise(seed=1):
    return np.random.default_rng(seed).standard_normal((256, 500))


def clean(frequency=10.0, channel=0):
    window = np.zeros((256, 500), dtype=np.float64)
    window[channel] = np.sin(2 * np.pi * frequency * np.arange(500) / 250)
    return window


def compare(a, b, tolerance=1e-9, factor_sign=1):
    assert a.keys() == b.keys()
    for key in a:
        expected = a[key] * factor_sign if key == "factors" else a[key]
        np.testing.assert_allclose(b[key], expected, atol=tolerance, rtol=tolerance)


def test_shapes_finite_and_population_covariance():
    raw, start, end = raw_window(noise())
    result = analyze_window(raw, start, end)
    shapes = {"covariance": (256, 256), "factors": (5, 2, 256, 2),
              "q": (5, 2, 4), "q2": (5, 2), "zero_shot": (5,)}
    assert set(result) == set(shapes)
    x = raw[:256, start:end].copy()
    x -= x.mean(axis=1, keepdims=True)
    x /= np.sqrt(np.mean(x ** 2))
    np.testing.assert_allclose(result["covariance"], x @ x.T / 500, atol=1e-12)
    for name, shape in shapes.items():
        assert result[name].shape == shape
        assert result[name].dtype == np.float64
        assert np.isfinite(result[name]).all()
        assert result[name].flags.owndata
        assert not np.shares_memory(result[name], raw)
    assert np.trace(result["covariance"]) == pytest.approx(256)
    assert np.all((result["zero_shot"] >= 0) & (result["zero_shot"] <= 1))


def test_excluded_row_and_times_poisoned_before_numeric_checks():
    raw, start, end = raw_window(noise())
    expected = analyze_window(raw, start, end)
    raw[256] = np.nan
    raw[:256, :start] = np.inf
    raw[:256, end:] = -np.inf
    raw.setflags(write=False)
    compare(expected, analyze_window(raw, start, end), tolerance=0)


def test_raw_unchanged_results_detached_and_channel_order_retained():
    raw, start, end = raw_window(noise(2))
    before = raw.copy()
    result = analyze_window(raw, start, end)
    np.testing.assert_array_equal(raw, before)
    permutation = np.arange(255, -1, -1)
    reordered = raw.copy()
    reordered[:256] = reordered[permutation]
    moved = analyze_window(reordered, start, end)
    np.testing.assert_allclose(moved["covariance"], result["covariance"][permutation][:, permutation], atol=1e-12)
    np.testing.assert_allclose(moved["factors"], result["factors"][:, :, permutation], atol=1e-12)
    for name in ("q", "q2", "zero_shot"):
        np.testing.assert_allclose(moved[name], result[name], atol=1e-9)
    snapshot = {name: value.copy() for name, value in result.items()}
    raw[:] = 12345
    compare(snapshot, result, tolerance=0)


def test_noncontiguous_input_window_and_numpy_integer_indices():
    raw, start, end = raw_window(noise(3))
    expanded = np.repeat(raw, 2, axis=1)
    noncontiguous = expanded[:, ::2]
    assert not noncontiguous.flags.c_contiguous
    compare(analyze_window(raw, start, end),
            analyze_window(noncontiguous, np.int64(start), np.int64(end)), tolerance=0)


@pytest.mark.parametrize("scale", [1e-200, 1e200, -3.0])
def test_global_scale_without_physical_amplitude_cutoff(scale):
    raw, start, end = raw_window(noise(4))
    reference = analyze_window(raw, start, end)
    transformed = analyze_window(raw * scale, start, end)
    compare(reference, transformed, factor_sign=np.sign(scale), tolerance=1e-9)


def test_per_channel_dc_offsets_are_removed():
    window = noise(5)
    reference = analyze_window(*raw_window(window))
    offset = np.linspace(-100, 100, 256)[:, None]
    transformed = analyze_window(*raw_window(window + offset))
    compare(reference, transformed, tolerance=1e-9)


def test_real_quadrature_psd_not_complex_outer_product():
    window = clean()
    window[1] = np.cos(2 * np.pi * 10 * np.arange(500) / 250)
    result = analyze_window(*raw_window(window))
    factor = result["factors"][3, 0]
    projected = factor @ factor.T
    assert not np.iscomplexobj(factor)
    np.testing.assert_allclose(projected[:2, :2], np.eye(2) * 128, atol=1e-10)
    assert np.max(np.abs(projected[2:])) == 0
    assert result["q"][3, 0, 0] == pytest.approx(1)
    assert result["q"][3, 0, 2] == pytest.approx(2 / 256, abs=1e-12)


def test_quadrature_gram_is_psd_and_bounded_by_covariance():
    result = analyze_window(*raw_window(noise(6)))
    covariance = result["covariance"]
    for factor in result["factors"].reshape(10, 256, 2):
        projected = factor @ factor.T
        assert np.linalg.eigvalsh(projected).min() >= -1e-10
        assert np.linalg.eigvalsh(covariance - projected).min() >= -1e-10
    assert np.all((result["q"][..., 0] >= 0) & (result["q"][..., 0] <= 1))


def test_zero_shot_matches_rank_one_regularized_cca_formula():
    result = analyze_window(*raw_window(clean()))
    expected = 256 / (256 + 1e-6)
    assert result["zero_shot"][3] == pytest.approx(expected, abs=1e-9)
    assert int(np.argmax(result["zero_shot"])) == 3


def test_strong_q_changes_with_signal_noise_and_spatial_rank():
    stable = analyze_window(*raw_window(clean()))
    noisy = analyze_window(*raw_window(clean() + 0.5 * noise(7)))
    assert stable["q"][3, 0, 0] > noisy["q"][3, 0, 0] + 0.5
    assert stable["q"][3, 0, 1] > noisy["q"][3, 0, 1] + 2
    assert noisy["q"][3, 0, 2] > stable["q"][3, 0, 2] + 0.1
    assert stable["q2"][3, 0] > noisy["q2"][3, 0] + 2


def test_neighbor_band_signal_changes_log_harmonic_ratio():
    signal = analyze_window(*raw_window(clean(10)))
    neighbor = analyze_window(*raw_window(clean(10.5)))
    assert signal["q"][3, 0, 1] > neighbor["q"][3, 0, 1] + 10


def test_split_half_q_detects_changing_spatial_template():
    stable = analyze_window(*raw_window(clean()))
    changed = clean()
    changed[1, 250:] = changed[0, 250:]
    changed[0, 250:] = 0
    shifted = analyze_window(*raw_window(changed))
    assert stable["q"][3, 0, 3] < 1e-10
    assert shifted["q"][3, 0, 3] == pytest.approx(np.sqrt(2), abs=1e-10)


def test_q2_lag_is_shifted_global_cosine_and_candidate_invariant():
    low = analyze_window(*raw_window(clean()))
    high = np.zeros((256, 500), dtype=np.float64)
    high[0] = (-1.0) ** np.arange(500)
    alternating = analyze_window(*raw_window(high))
    np.testing.assert_array_equal(low["q2"], np.tile(low["q2"][0], (5, 1)))
    assert low["q2"][0, 1] > 0.9
    assert alternating["q2"][0, 1] == pytest.approx(-1)


@pytest.mark.parametrize("bad", [np.zeros((256, 500)), np.zeros((258, 500)),
                                   np.zeros((257, 500), dtype=np.float32),
                                   np.zeros((257, 500), dtype=np.int64),
                                   np.zeros((257, 500), dtype=np.complex128),
                                   [[0.0]], np.zeros(257)])
def test_invalid_structure_is_rejected(bad):
    with pytest.raises(ValueError):
        analyze_window(bad, 0, 500)


@pytest.mark.parametrize("start,end", [(True, 501), (0, False), (0.0, 500),
                                         (-1, 499), (0, 499), (0, 501), (1, 501)])
def test_invalid_intervals_are_rejected(start, end):
    with pytest.raises(ValueError):
        analyze_window(np.zeros((257, 500)), start, end)


@pytest.mark.parametrize("value", [0.0, 17.0])
def test_exactly_zero_centered_energy_is_rejected(value):
    with pytest.raises(ValueError, match="zero_centered_energy"):
        analyze_window(np.full((257, 500), value, dtype=np.float64), 0, 500)


@pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf])
def test_selected_nonfinite_values_are_rejected(value):
    raw, start, end = raw_window(noise(8))
    raw[17, start + 17] = value
    with pytest.raises(ValueError, match="nonfinite_selected_window"):
        analyze_window(raw, start, end)
