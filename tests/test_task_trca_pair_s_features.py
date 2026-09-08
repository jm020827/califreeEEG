"""Generated-only literal checks of the frozen C2 pair feature contracts."""

import inspect
from dataclasses import FrozenInstanceError

import numpy as np
import pytest

from cfeg.analysis import task_trca_pair_s_features as features
from cfeg.analysis import task_trca_shape_features as old


@pytest.fixture
def support():
    rng = np.random.default_rng(20260909)
    x = rng.normal(size=(5, 12, 5, 8, 31))
    x *= np.linspace(0.5, 3.0, 8)[None, None, None, :, None]
    x += np.linspace(-2, 3, 5)[:, None, None, None, None]
    return x


@pytest.fixture
def frequencies():
    return np.linspace(8.0, 15.0, 12)


def literal_q(x, mask, interface, order, frequencies, *, center_residual_signal=False):
    """Separate band/channel/waveform calculation; no producer helper calls."""
    k, _, _, channels, samples = x.shape
    coordinates = [(i, j) for i in range(k) for j in range(i + 1, k)]
    bases = []
    time = np.arange(samples) / 250
    for frequency in frequencies:
        columns = []
        for harmonic in range(1, 6):
            angle = 2 * np.pi * frequency * harmonic * time
            columns.extend((np.sin(angle), np.cos(angle)))
        reference = np.column_stack(columns)
        bases.append(np.linalg.qr(reference - np.mean(reference, axis=0))[0])
    output = np.zeros((5, len(coordinates), 15))
    for band in range(5):
        power = np.array(
            [[np.mean(x[i, :, band, c] ** 2) for c in range(channels)] for i in range(k)]
        )
        log_power = np.log(np.maximum(power / max(np.mean(power), 1e-12), 1e-12))
        off = np.zeros_like(power)
        for i in range(k):
            for c in range(channels):
                for label, basis in enumerate(bases):
                    wave = x[i, label, band, c]
                    if center_residual_signal:
                        wave = wave - wave.mean()
                    residual = wave - basis @ (basis.T @ wave)
                    off[i, c] += np.dot(residual, residual) / samples / 12
        log_off = np.log(np.clip(off / np.maximum(power, 1e-12), 1e-6, 1))
        for e, (i, j) in enumerate(coordinates):
            disagreement = []
            for c in range(channels):
                class_d = []
                for label in range(12):
                    left, right = x[i, label, band, c], x[j, label, band, c]
                    left, right = left - left.mean(), right - right.mean()
                    corr = np.dot(left, right) / np.sqrt(np.dot(left, left) * np.dot(right, right))
                    class_d.append(1 - np.clip(corr, -1, 1))
                disagreement.append(np.mean(class_d))
            output[band, e] = [
                np.mean((log_power[i] + log_power[j]) / 2),
                np.sqrt(np.mean((log_power[i] - log_power[j]) ** 2)),
                np.mean((log_off[i] + log_off[j]) / 2),
                np.sqrt(np.mean((log_off[i] - log_off[j]) ** 2)),
                np.mean(disagreement),
                np.sqrt(np.mean((disagreement - np.mean(disagreement)) ** 2)),
                np.log(k),
                interface,
                order,
                float(interface != order),
                (i + j) / (2 * (k - 1)),
                (j - i) / (k - 1),
                np.count_nonzero(mask[i] & mask[j]) / channels,
                np.count_nonzero(mask[i] ^ mask[j]) / channels,
                abs(np.count_nonzero(mask[i]) - np.count_nonzero(mask[j])) / channels,
            ]
    return output


def literal_m(packet):
    k = len(packet)
    z = np.log1p(packet)
    means = [
        np.mean(z[np.isfinite(z[:, channel]), channel]) if np.isfinite(z[:, channel]).any() else 0.0
        for channel in range(8)
    ]
    pairs, available = [], []
    for i in range(k):
        for j in range(i + 1, k):
            common = [c for c in range(8) if np.isfinite(z[i, c]) and np.isfinite(z[j, c])]
            available.append(bool(common))
            if not common:
                pairs.append([0.0, 0.0])
            else:
                pairs.append(
                    [
                        np.mean([(z[i, c] + z[j, c]) / 2 - means[c] for c in common]),
                        np.sqrt(np.mean([(z[i, c] - z[j, c]) ** 2 for c in common])),
                    ]
                )
    return np.array(pairs), np.array(available)


@pytest.mark.parametrize("k", [3, 5, np.int64(3), np.int32(5)])
def test_pair_coordinates_are_literal_immutable_native_width(k):
    pairs = features.pair_coordinates(k)
    assert pairs.dtype == np.int64 and pairs.shape == (k * (k - 1) // 2, 2)
    np.testing.assert_array_equal(pairs, [(i, j) for i in range(k) for j in range(i + 1, k)])
    with pytest.raises(ValueError):
        pairs.setflags(write=True)
    with pytest.raises(ValueError):
        pairs[0, 0] = 1
    assert not np.shares_memory(pairs, features.pair_coordinates(k))


@pytest.mark.parametrize("bad", [0, 1, 2, 4, 6, True, np.bool_(False), 3.0, "3", None])
def test_pair_coordinates_reject_noninteger_or_nonbudget(bad):
    with pytest.raises(ValueError):
        features.pair_coordinates(bad)


@pytest.mark.parametrize("k", [3, 5])
@pytest.mark.parametrize("interface,order", [(0, 0), (0, 1), (1, 0), (1, 1)])
def test_q_matches_independent_literal_all_15_features(support, frequencies, k, interface, order):
    x = support[:k]
    mask = np.arange(k * 8).reshape(k, 8) % 3 != 0
    before, mask_before = x.copy(), mask.copy()
    actual = features.support_q_mask(x, mask, interface, order, frequencies)
    expected = literal_q(x, mask, interface, order, frequencies)
    np.testing.assert_allclose(actual, expected, atol=2e-14, rtol=2e-14)
    assert actual.shape == (5, k * (k - 1) // 2, 15) and actual.dtype == np.float64
    np.testing.assert_array_equal(x, before)
    np.testing.assert_array_equal(mask, mask_before)
    assert features.Q2_INDICES == (2, 4)


def test_q_residual_keeps_uncentered_signal_with_centered_reference(support, frequencies):
    x = support[:3]
    mask = np.ones((3, 8), bool)
    actual = features.support_q_mask(x, mask, 0, 1, frequencies)
    wrong = literal_q(x, mask, 0, 1, frequencies, center_residual_signal=True)
    assert np.max(np.abs(actual[..., 2:4] - wrong[..., 2:4])) > 0.1


def test_mask_only_api_cannot_read_numeric_m_or_old_q(support, frequencies, monkeypatch):
    def forbidden(*_, **__):
        raise AssertionError("Pair Q crossed a forbidden numeric-M/old-Q path")

    monkeypatch.setattr(features, "metadata_features", forbidden)
    monkeypatch.setattr(old, "metadata_features", forbidden)
    monkeypatch.setattr(old, "support_q_mask", forbidden)
    packet = np.arange(40, dtype=float).reshape(5, 8)
    packet[0, 4] = np.nan
    changed = 500 + 100 * packet
    actual = features.support_q_mask(support, np.isfinite(packet), 0, 0, frequencies)
    np.testing.assert_array_equal(
        actual, features.support_q_mask(support, np.isfinite(changed), 0, 0, frequencies)
    )
    assert tuple(inspect.signature(features.support_q_mask).parameters) == (
        "support",
        "mask",
        "interface",
        "order",
        "frequencies",
    )


def test_actual_k3_pairs_and_features_are_not_first_three_k5(support, frequencies):
    assert features.pair_coordinates(3).tolist() == [[0, 1], [0, 2], [1, 2]]
    assert features.pair_coordinates(5)[:3].tolist() == [[0, 1], [0, 2], [0, 3]]
    mask = np.ones((5, 8), bool)
    q3 = features.support_q_mask(support[:3], mask[:3], 0, 0, frequencies)
    q5 = features.support_q_mask(support, mask, 0, 0, frequencies)
    np.testing.assert_array_equal(q3[..., 6], np.log(3))
    np.testing.assert_array_equal(q5[..., 6], np.log(5))
    assert not np.allclose(q3[..., :6], q5[:, :3, :6])
    before = q3.copy()
    support[3:] = np.nan
    mask[3:] = False
    np.testing.assert_array_equal(
        before, features.support_q_mask(support[:3], mask[:3], 0, 0, frequencies)
    )
    with pytest.raises(ValueError, match="mask"):
        features.support_q_mask(support[:3], mask, 0, 0, frequencies)


@pytest.mark.parametrize("k", [3, 5])
def test_pair_cross_products_literal_uncentered_and_prefix_mapping(support, k):
    x = support[:k]
    before = x.copy()
    actual = features.pair_cross_products(x)
    expected = np.empty((5, k * (k - 1) // 2, 12, 8, 8))
    pairs = [(i, j) for i in range(k) for j in range(i + 1, k)]
    for band in range(5):
        for pair, (i, j) in enumerate(pairs):
            for label in range(12):
                left, right = x[i, label, band], x[j, label, band]
                expected[band, pair, label] = left @ right.T + right @ left.T
    np.testing.assert_array_equal(actual, expected)
    np.testing.assert_array_equal(actual, actual.swapaxes(-1, -2))
    np.testing.assert_array_equal(x, before)
    centered = features.pair_cross_products(x - x.mean(-1, keepdims=True))
    assert np.max(np.abs(actual - centered)) > 1
    if k == 5:
        np.testing.assert_array_equal(features.pair_cross_products(x[:3]), actual[:, [0, 1, 4]])


@pytest.mark.parametrize("k", [3, 5])
def test_channel_permutation_q_m_invariance_and_cross_equivariance(support, frequencies, k):
    permutation = np.array([5, 1, 7, 2, 0, 4, 6, 3])
    packet = np.expm1(np.arange(k * 8).reshape(k, 8) / 10)
    packet[0, 2], packet[1, 6] = np.nan, np.nan
    x, mask = support[:k], np.isfinite(packet)
    q = features.support_q_mask(x, mask, 1, 0, frequencies)
    q_permuted = features.support_q_mask(
        x[:, :, :, permutation], mask[:, permutation], 1, 0, frequencies
    )
    np.testing.assert_allclose(q, q_permuted, atol=2e-14, rtol=2e-14)
    m, available = features.metadata_features(packet)
    mp, ap = features.metadata_features(packet[:, permutation])
    np.testing.assert_allclose(m, mp, atol=2e-14, rtol=2e-14)
    np.testing.assert_array_equal(available, ap)
    cross = features.pair_cross_products(x)
    changed = features.pair_cross_products(x[:, :, :, permutation])
    np.testing.assert_array_equal(changed, cross[..., permutation, :][..., permutation])


@pytest.mark.parametrize("dtype", [int, float, object])
def test_q_rejects_numeric_masks(support, frequencies, dtype):
    with pytest.raises(ValueError, match="boolean"):
        features.support_q_mask(support, np.ones((5, 8), dtype), 0, 0, frequencies)


@pytest.mark.parametrize("k", [0, 1, 2, 4, 6])
def test_support_rejects_wrong_prefix_width(support, frequencies, k):
    x = np.resize(support, (k, 12, 5, 8, 31))
    with pytest.raises(ValueError, match="support"):
        features.support_q_mask(x, np.ones((k, 8), bool), 0, 0, frequencies)
    with pytest.raises(ValueError, match="support"):
        features.pair_cross_products(x)


@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf, 1e300])
def test_support_rejects_nonfinite_or_overflow(support, frequencies, bad):
    support[0, 0, 0, 0, 0] = bad
    with pytest.raises(ValueError):
        features.support_q_mask(support, np.ones((5, 8), bool), 0, 0, frequencies)
    if abs(bad) != 1e300:
        with pytest.raises(ValueError):
            features.pair_cross_products(support)
    else:
        support[1, 0, 0, 0, 0] = bad
        with pytest.raises(ValueError):
            features.pair_cross_products(support)


@pytest.mark.parametrize(
    "shape", [(3, 11, 5, 8, 31), (3, 12, 4, 8, 31), (3, 12, 5, 7, 31), (3, 12, 5, 8, 1)]
)
def test_support_rejects_wrong_geometry(shape, frequencies):
    with pytest.raises(ValueError):
        features.support_q_mask(np.ones(shape), np.ones((3, 8), bool), 0, 0, frequencies)
    with pytest.raises(ValueError):
        features.pair_cross_products(np.ones(shape))


@pytest.mark.parametrize("context", [(2, 0), (0, -1), (0.0, 1), (True, 0), (0, False)])
def test_q_rejects_invalid_context(support, frequencies, context):
    with pytest.raises(ValueError, match="integer"):
        features.support_q_mask(support, np.ones((5, 8), bool), *context, frequencies)


@pytest.mark.parametrize(
    "frequency", [np.ones(11), np.ones((12, 1)), np.zeros(12), [np.nan] * 12, [np.inf] * 12]
)
def test_q_rejects_invalid_frequency(support, frequency):
    with pytest.raises(ValueError, match="frequencies"):
        features.support_q_mask(support, np.ones((5, 8), bool), 0, 0, frequency)


def test_q_rejects_zero_centered_variance(support, frequencies):
    support[0, 0, 0, 0] = 5
    with pytest.raises(ValueError, match="variance"):
        features.support_q_mask(support, np.ones((5, 8), bool), 0, 0, frequencies)


@pytest.mark.parametrize("k", [3, 5])
def test_metadata_literal_prefix_center_and_common_channels(k):
    packet = np.expm1(np.arange(k * 8).reshape(k, 8) / 10)
    packet[0, 1], packet[1, 2], packet[:, 7] = np.nan, np.nan, np.nan
    before = packet.copy()
    expected, ea = literal_m(packet)
    actual, available = features.metadata_features(packet)
    np.testing.assert_allclose(actual, expected, atol=2e-15, rtol=2e-15)
    np.testing.assert_array_equal(available, ea)
    np.testing.assert_array_equal(packet, before)
    assert actual.shape == (k * (k - 1) // 2, 2) and actual.dtype == np.float64
    assert available.dtype == bool


def test_metadata_common_channels_can_be_empty_per_pair_and_zero_is_observed():
    packet = np.full((3, 8), np.nan)
    packet[0, 0], packet[1, 1], packet[2, 0] = 0, 0, 0
    m, available = features.metadata_features(packet)
    np.testing.assert_array_equal(m, 0)
    np.testing.assert_array_equal(available, [False, True, False])
    zeros, observed = features.metadata_features(np.zeros((3, 8)))
    missing, absent = features.metadata_features(np.full((3, 8), np.nan))
    np.testing.assert_array_equal(zeros, missing)
    assert observed.all() and not absent.any()


def test_metadata_center_is_current_prefix_not_only_pair_or_k5():
    packet = np.zeros((5, 8))
    packet[2] = np.expm1(3)
    packet[3:] = np.expm1(8)
    m3, _ = features.metadata_features(packet[:3])
    m5, _ = features.metadata_features(packet)
    np.testing.assert_allclose(m3[0], [-1, 0], atol=1e-15)
    assert m5[0, 0] != m3[0, 0]
    packet[3:] = np.nan
    np.testing.assert_array_equal(features.metadata_features(packet[:3])[0], m3)


@pytest.mark.parametrize("k", [3, 5])
def test_stale_preserves_original_mask_and_never_invents_block0_values(k):
    packet = np.arange(k * 8, dtype=float).reshape(k, 8)
    packet[0, 2], packet[1, 4], packet[-1, 5] = np.nan, np.nan, np.nan
    before = packet.copy()
    actual = features.stale_packet(packet)
    expected = np.repeat(packet[:1], k, axis=0)
    expected[~np.isfinite(packet)] = np.nan
    np.testing.assert_array_equal(actual, expected)
    np.testing.assert_array_equal(packet, before)
    assert not np.isfinite(actual[:, 2]).any()
    m, available = features.metadata_features(actual)
    np.testing.assert_allclose(m, 0, atol=5e-16)
    assert available.all()


def test_stale_availability_can_disappear_when_block0_missing():
    packet = np.full((3, 8), np.nan)
    packet[1:, 4] = [0.0, 1.0]
    _, original_available = features.metadata_features(packet)
    stale = features.stale_packet(packet)
    m, stale_available = features.metadata_features(stale)
    np.testing.assert_array_equal(original_available, [False, False, True])
    assert not np.isfinite(stale).any() and not stale_available.any()
    np.testing.assert_array_equal(m, 0)


@pytest.mark.parametrize("bad", [-1.0, np.inf, -np.inf])
@pytest.mark.parametrize("function", [features.metadata_features, features.stale_packet])
def test_metadata_stale_reject_invalid_observed_values(bad, function):
    packet = np.ones((3, 8))
    packet[0, 0] = bad
    with pytest.raises(ValueError, match="Impedance"):
        function(packet)


@pytest.mark.parametrize("shape", [(0, 8), (1, 8), (4, 8), (6, 8), (3, 7), (2, 3, 8)])
def test_metadata_stale_reject_nonprefix_shape(shape):
    for function in (features.metadata_features, features.stale_packet):
        with pytest.raises(ValueError, match="prefix"):
            function(np.ones(shape))


@pytest.mark.parametrize("dtype", [complex, object, bool])
def test_numeric_apis_reject_non_real_dtype(support, frequencies, dtype):
    for function in (features.metadata_features, features.stale_packet):
        with pytest.raises(ValueError, match="real numeric"):
            function(np.ones((3, 8), dtype))
    with pytest.raises(ValueError, match="real numeric"):
        features.pair_cross_products(support.astype(dtype))
    with pytest.raises(ValueError, match="real numeric"):
        features.support_q_mask(support.astype(dtype), np.ones((5, 8), bool), 0, 0, frequencies)


def test_weighted_scaler_global_population_and_equal_real_case_mass():
    values = np.r_[np.zeros((3, 2)), np.full((10, 2), 6.0)]
    ids = [4] * 3 + [6] * 10
    masses = np.r_[np.full(3, 1 / 3), np.full(10, 1 / 10)]
    scaler = features.fit_weighted_scaler(values, ids, [6, 4], masses)
    np.testing.assert_allclose(scaler.mean, [3, 3], atol=1e-15)
    np.testing.assert_allclose(scaler.scale, [3, 3], atol=1e-15)
    assert scaler.fit_ids == (4, 6)
    assert not np.allclose(scaler.mean, values.mean(0))
    available = np.r_[np.ones(3, bool), np.ones(1, bool), np.zeros(9, bool)]
    filtered = features.fit_weighted_scaler(values, ids, [4, 6], masses, available)
    expected_mean = 6 * 0.1 / 1.1
    np.testing.assert_allclose(filtered.mean, expected_mean, atol=1e-15)
    assert not np.allclose(filtered.mean, 3)  # No person-wise retained-mass renormalization.


def test_weighted_scaler_excludes_nonfit_and_unavailable_numeric_poison():
    values = np.array([[1, 8], [3, 4], [20, 40], [5, 2], [7, 0]], float)
    ids, masses = [4, 4, 6, 8, 8], np.array([0.2, 0.8, 10, 0.3, 0.7])
    available = np.array([True, False, True, True, True])
    expected = features.fit_weighted_scaler(values, ids, [4, 8], masses, available)
    values[1], values[2] = np.nan, np.inf
    actual = features.fit_weighted_scaler(values, ids, [4, 8], masses, available)
    np.testing.assert_array_equal(actual.mean, expected.mean)
    np.testing.assert_array_equal(actual.scale, expected.scale)
    chosen, weights = values[[0, 3, 4]], masses[[0, 3, 4]]
    weights /= weights.sum()
    mean = np.sum(chosen * weights[:, None], axis=0)
    variance = np.sum((chosen - mean) ** 2 * weights[:, None], axis=0)
    np.testing.assert_array_equal(actual.mean, mean)
    np.testing.assert_array_equal(actual.scale, np.sqrt(variance))


def test_weighted_scaler_empty_available_floor_and_immutable_export():
    assert features.FeatureScaler is old.FeatureScaler and features.donor_map is old.donor_map
    scaler = features.fit_weighted_scaler(
        np.full((3, 2), np.nan), [4, 4, 6], [4], [1 / 3] * 3, np.zeros(3, bool)
    )
    np.testing.assert_array_equal(scaler.mean, [0, 0])
    np.testing.assert_array_equal(scaler.scale, [1, 1])
    np.testing.assert_array_equal(scaler.transform(np.full((3, 2), np.nan), np.zeros(3, bool)), 0)
    with pytest.raises(ValueError):
        scaler.mean.setflags(write=True)
    with pytest.raises(FrozenInstanceError):
        scaler.fit_ids = (6,)
    tiny = features.fit_weighted_scaler([[2.0, 0.0], [2.0, 1e-13]], [4, 6], [4, 6], [0.5, 0.5])
    np.testing.assert_array_equal(tiny.scale, [1, 1])
    np.testing.assert_array_equal(tiny.transform([1e8, 1e8]), np.array([1e8, 1e8]) - tiny.mean)


@pytest.mark.parametrize(
    "masses", [[0, 1], [1, 0], [-1, 1], [np.nan, 1], [np.inf, 1], [[1], [1]], [1], [True, True]]
)
def test_weighted_scaler_rejects_invalid_or_padding_mass_globally(masses):
    with pytest.raises(ValueError):
        features.fit_weighted_scaler([[1], [np.nan]], [4, 6], [4], masses, [True, False])


@pytest.mark.parametrize("fit_ids", [[], [8], [4, 4], [True], [4.0]])
def test_weighted_scaler_rejects_missing_duplicate_or_invalid_fit_ids(fit_ids):
    with pytest.raises(ValueError):
        features.fit_weighted_scaler([[1], [2]], [4, 6], fit_ids, [0.5, 0.5])


@pytest.mark.parametrize("ids", [[4], [0, 6], [4.0, 6], [True, 6]])
def test_weighted_scaler_rejects_invalid_participant_axis(ids):
    with pytest.raises(ValueError):
        features.fit_weighted_scaler([[1], [2]], ids, [4], [0.5, 0.5])


@pytest.mark.parametrize(
    "values",
    [np.zeros((2, 3, 1)), np.ones(2), np.zeros((2, 0)), [[np.nan], [2]], [[1e300], [-1e300]]],
)
def test_weighted_scaler_rejects_bad_geometry_selected_numbers_or_overflow(values):
    with pytest.raises(ValueError):
        features.fit_weighted_scaler(values, [4, 6], [4, 6], [0.5, 0.5])


@pytest.mark.parametrize("availability", [[1, 0], [[True], [False]], [True]])
def test_weighted_scaler_rejects_nonboolean_or_wrong_availability(availability):
    with pytest.raises(ValueError):
        features.fit_weighted_scaler([[1], [2]], [4, 6], [4, 6], [0.5, 0.5], availability)


def test_pair_feature_module_does_not_reuse_old_scaler_or_m_formula(monkeypatch):
    def forbidden(*_, **__):
        raise AssertionError("C2 cannot reuse old unweighted scaler or M formula")

    monkeypatch.setattr(old, "fit_scaler", forbidden)
    monkeypatch.setattr(old, "metadata_features", forbidden)
    features.fit_weighted_scaler([[1], [2]], [4, 6], [4, 6], [0.5, 0.5])
    features.metadata_features(np.zeros((3, 8)))
