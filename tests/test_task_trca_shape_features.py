"""Artificial-only checks of metadata exclusion and fit-partition boundaries."""

import inspect
from dataclasses import FrozenInstanceError

import numpy as np
import pytest

from cfeg.analysis import metadata_prior_source as reference
from cfeg.analysis import task_trca_shape_features as features


@pytest.fixture
def artificial_support():
    return np.random.default_rng(908).normal(size=(5, 12, 5, 8, 32))


@pytest.fixture
def frequencies():
    return np.linspace(8.0, 15.0, 12)


@pytest.mark.parametrize("k", [3, 5])
@pytest.mark.parametrize("interface,order", [(0, 0), (0, 1), (1, 0), (1, 1)])
def test_mask_only_q_exact_artificial_reference(
    artificial_support, frequencies, k, interface, order
):
    mask = np.ones((k, 8), dtype=bool)
    mask[0, 0] = False
    mask[:, 7] = False
    support = artificial_support[:k]
    support_before, mask_before = support.copy(), mask.copy()
    expected = reference.support_q(
        support, np.where(mask, 0.0, np.nan), interface, order, frequencies
    )
    actual = features.support_q_mask(support, mask, interface, order, frequencies)
    np.testing.assert_array_equal(actual, expected)
    np.testing.assert_array_equal(support, support_before)
    np.testing.assert_array_equal(mask, mask_before)
    assert actual.shape == (5, 8, 15)
    assert actual.dtype == np.float64
    np.testing.assert_array_equal(actual[..., 4], np.broadcast_to(mask.mean(0), (5, 8)))
    np.testing.assert_array_equal(actual[..., 7:], np.broadcast_to(np.eye(8), (5, 8, 8)))
    assert features.Q2_INDICES == (1, 2)


def test_q_does_not_call_old_numeric_m_functions(artificial_support, frequencies, monkeypatch):
    def denied(*args, **kwargs):
        raise AssertionError("Q crossed into a numeric metadata function")

    monkeypatch.setattr(reference, "support_q", denied)
    monkeypatch.setattr(reference, "m_features", denied)
    monkeypatch.setattr(features, "m_features", denied)
    q = features.support_q_mask(artificial_support, np.ones((5, 8), bool), 0, 0, frequencies)
    assert np.isfinite(q).all()
    assert tuple(inspect.signature(features.support_q_mask).parameters) == (
        "support",
        "mask",
        "interface",
        "order",
        "frequencies",
    )


def test_numeric_packet_changes_do_not_change_q(artificial_support, frequencies):
    packet = np.arange(40, dtype=float).reshape(5, 8)
    packet[0, 2] = np.nan
    changed = packet * 100 + 500
    q = features.support_q_mask(artificial_support, np.isfinite(packet), 0, 1, frequencies)
    q_changed = features.support_q_mask(artificial_support, np.isfinite(changed), 0, 1, frequencies)
    np.testing.assert_array_equal(q, q_changed)
    assert not np.array_equal(
        features.metadata_features(packet)[0], features.metadata_features(changed)[0]
    )


def test_prefix_only_q_does_not_sum_future_mask(artificial_support, frequencies):
    mask = np.ones((5, 8), bool)
    q = features.support_q_mask(artificial_support[:3], mask[:3], 0, 0, frequencies)
    mask[3:] = False
    changed = artificial_support.copy()
    changed[3:] = np.nan
    np.testing.assert_array_equal(
        q, features.support_q_mask(changed[:3], mask[:3], 0, 0, frequencies)
    )
    with pytest.raises(ValueError, match="mask"):
        features.support_q_mask(changed[:3], mask, 0, 0, frequencies)


@pytest.mark.parametrize("dtype", [float, int, object])
def test_q_rejects_numeric_masks(artificial_support, frequencies, dtype):
    with pytest.raises(ValueError, match="boolean"):
        features.support_q_mask(artificial_support, np.ones((5, 8), dtype=dtype), 0, 0, frequencies)


@pytest.mark.parametrize("k", [0, 1, 2, 4, 6])
def test_q_rejects_nonbudget_prefix(artificial_support, frequencies, k):
    support = np.resize(artificial_support, (k, 12, 5, 8, 32))
    with pytest.raises(ValueError, match="support"):
        features.support_q_mask(support, np.ones((k, 8), bool), 0, 0, frequencies)


@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf, 1e300])
def test_q_rejects_nonfinite_or_overflow_support(artificial_support, frequencies, bad):
    artificial_support[0, 0, 0, 0, 0] = bad
    with pytest.raises(ValueError):
        features.support_q_mask(artificial_support, np.ones((5, 8), bool), 0, 0, frequencies)


def test_q_rejects_zero_centered_variance(artificial_support, frequencies):
    artificial_support[:, 0, 0, 0] = 5
    with pytest.raises(ValueError, match="variance"):
        features.support_q_mask(artificial_support, np.ones((5, 8), bool), 0, 0, frequencies)


@pytest.mark.parametrize("context", [(2, 0), (0, -1), (0.0, 1), (True, 0), (0, False)])
def test_q_rejects_invalid_context(artificial_support, frequencies, context):
    with pytest.raises(ValueError, match="integer"):
        features.support_q_mask(artificial_support, np.ones((5, 8), bool), *context, frequencies)


@pytest.mark.parametrize(
    "frequency", [np.ones(11), np.ones((12, 1)), np.zeros(12), [np.nan] * 12, [np.inf] * 12]
)
def test_q_rejects_invalid_frequencies(artificial_support, frequency):
    with pytest.raises(ValueError, match="frequencies"):
        features.support_q_mask(artificial_support, np.ones((5, 8), bool), 0, 0, frequency)


@pytest.mark.parametrize("k", [3, 5])
def test_metadata_zero_missing_and_population_sd(k):
    packet = np.array([[0, 1, 3, 7, 15, 31, 63, np.nan]] * k, dtype=float)
    before = packet.copy()
    actual, available = features.metadata_features(packet)
    np.testing.assert_array_equal(actual, reference.m_features(packet)[0])
    np.testing.assert_allclose(actual[:7, 0], np.arange(-3, 4) * np.log(2), atol=1e-14)
    np.testing.assert_array_equal(actual[:, 1], 0)
    assert available.tolist() == [True] * 7 + [False]
    np.testing.assert_array_equal(actual[7], 0)
    np.testing.assert_array_equal(packet, before)
    packet[:, 0] = np.expm1(np.arange(k))
    actual, _ = features.metadata_features(packet)
    assert actual[0, 1] == pytest.approx(np.std(np.arange(k), ddof=0))


def test_metadata_all_missing_is_not_observed_zero():
    missing, absent = features.metadata_features(np.full((3, 8), np.nan))
    zeros, observed = features.metadata_features(np.zeros((3, 8)))
    np.testing.assert_array_equal(missing, zeros)
    assert not absent.any()
    assert observed.all()


@pytest.mark.parametrize("bad", [-1, np.inf, -np.inf])
def test_metadata_rejects_invalid_numbers(bad):
    packet = np.ones((3, 8))
    packet[0, 0] = bad
    with pytest.raises(ValueError):
        features.metadata_features(packet)


@pytest.mark.parametrize("shape", [(0, 8), (1, 8), (4, 8), (6, 8), (3, 7), (2, 3, 8)])
def test_metadata_rejects_nonprefix_shape(shape):
    with pytest.raises(ValueError, match="prefix"):
        features.metadata_features(np.ones(shape))


@pytest.mark.parametrize("which", ["support", "metadata"])
def test_numeric_features_reject_complex_arrays(artificial_support, frequencies, which):
    with pytest.raises(ValueError, match="real numeric"):
        if which == "support":
            features.support_q_mask(
                artificial_support.astype(complex), np.ones((5, 8), bool), 0, 0, frequencies
            )
        else:
            features.metadata_features(np.ones((3, 8), complex))


def test_scaler_fit_participant_rows_and_population_statistics():
    values = np.arange(4 * 2 * 5 * 8 * 15, dtype=float).reshape(4, 2, 5, 8, 15)
    scaler = features.fit_scaler(values, [4, 6, 8, 4], [8, 4])
    selected = values[[0, 2, 3]].reshape(-1, 15)
    np.testing.assert_array_equal(scaler.mean, selected.mean(0))
    np.testing.assert_array_equal(scaler.scale, selected.std(0, ddof=0))
    assert scaler.fit_ids == (4, 8)
    standardized = scaler.transform(values)
    np.testing.assert_array_equal(standardized, (values - selected.mean(0)) / selected.std(0))


def test_scaler_excludes_outside_fit_and_unavailable_poison():
    values = np.array([[[1, 8], [3, 4]], [[20, 40], [60, 80]], [[5, 2], [7, 0]]], float)
    available = np.array([[True, False], [True, True], [True, True]])
    expected = features.fit_scaler(values, [4, 6, 8], [4, 8], available)
    values[1] = np.nan
    values[0, 1] = np.inf
    actual = features.fit_scaler(values, [4, 6, 8], [4, 8], available)
    np.testing.assert_array_equal(actual.mean, expected.mean)
    np.testing.assert_array_equal(actual.scale, expected.scale)
    transformed = actual.transform(values[[0, 2]], available[[0, 2]])
    np.testing.assert_array_equal(transformed[0, 1], 0)
    assert np.isfinite(transformed).all()
    with pytest.raises(ValueError, match="Available transform"):
        actual.transform(values, available)


def test_scaler_all_empty_defaults_and_exact_zero_unavailable():
    values = np.full((2, 5, 8, 2), np.nan)
    mask = np.zeros(values.shape[:-1], bool)
    scaler = features.fit_scaler(values, [4, 6], [4], mask)
    np.testing.assert_array_equal(scaler.mean, [0, 0])
    np.testing.assert_array_equal(scaler.scale, [1, 1])
    np.testing.assert_array_equal(scaler.transform(values, mask), np.zeros(values.shape))


def test_scaler_constant_small_sd_no_clip_and_vector_transform():
    values = np.array([[5.0, 0.0], [5.0, 1e-13]])
    scaler = features.fit_scaler(values, [4, 6], [4, 6])
    np.testing.assert_array_equal(scaler.scale, [1, 1])
    query = np.array([1e8, 1e8])
    np.testing.assert_array_equal(scaler.transform(query), query - scaler.mean)
    np.testing.assert_array_equal(scaler.transform([np.nan, np.inf], np.array(False)), [0, 0])


def test_scaler_copies_immutable_statistics_and_fit_ids():
    mean, scale, ids = np.array([1.0, 2.0]), np.array([2.0, 3.0]), [8, 4]
    scaler = features.FeatureScaler(mean, scale, ids)
    mean[:] = 0
    scale[:] = 99
    ids.append(6)
    np.testing.assert_array_equal(scaler.mean, [1, 2])
    np.testing.assert_array_equal(scaler.scale, [2, 3])
    assert scaler.fit_ids == (4, 8)
    with pytest.raises(ValueError):
        scaler.mean[0] = 100
    with pytest.raises(ValueError):
        scaler.scale.setflags(write=True)
    with pytest.raises(FrozenInstanceError):
        scaler.fit_ids = (6,)


@pytest.mark.parametrize(
    "ids,fit_ids",
    [
        ([4, 6], []),
        ([4, 6], [8]),
        ([4, 6], [4, 4]),
        ([4], [4]),
        ([4, 6.0], [4]),
        ([4, True], [4]),
        ([4, 0], [4]),
        ([4, 6], [False]),
    ],
)
def test_scaler_rejects_invalid_identity_boundary(ids, fit_ids):
    with pytest.raises(ValueError):
        features.fit_scaler(np.ones((2, 8, 2)), ids, fit_ids)


@pytest.mark.parametrize("mask", [np.ones((2, 8)), np.ones((2, 1), bool), np.ones(8, bool)])
def test_scaler_rejects_nonboolean_or_broadcast_mask(mask):
    with pytest.raises(ValueError, match="boolean"):
        features.fit_scaler(np.ones((2, 8, 2)), [4, 6], [4], mask)
    scaler = features.FeatureScaler(np.zeros(2), np.ones(2), (4,))
    with pytest.raises(ValueError, match="boolean"):
        scaler.transform(np.ones((2, 8, 2)), mask)


@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf, 1e308])
def test_scaler_rejects_invalid_fitting_arithmetic(bad):
    values = np.ones((2, 8, 2))
    values[0, 0, 0] = bad
    with pytest.raises(ValueError):
        features.fit_scaler(values, [4, 6], [4])


@pytest.mark.parametrize(
    "mean,scale",
    [([np.nan], [1]), ([0], [0]), ([0], [-1]), ([0], [np.inf]), ([0, 0], [1]), ([], [])],
)
def test_scaler_rejects_invalid_frozen_state(mean, scale):
    with pytest.raises(ValueError):
        features.FeatureScaler(mean, scale, (4,))


def test_q2_has_separate_available_only_scaler(artificial_support, frequencies):
    mask = np.ones((5, 8), bool)
    mask[:, 0] = False
    q = features.support_q_mask(artificial_support, mask, 0, 0, frequencies)
    rows = np.stack((q, q * 2))
    available = np.broadcast_to(mask.any(0), rows.shape[:-1])
    q_scaler = features.fit_scaler(rows, [4, 6], [4, 6])
    q2 = rows[..., features.Q2_INDICES]
    residual_scaler = features.fit_scaler(q2, [4, 6], [4, 6], available)
    assert residual_scaler.mean.shape == (2,)
    np.testing.assert_array_equal(residual_scaler.mean, q2[available].mean(0))
    assert not np.array_equal(residual_scaler.mean, q_scaler.mean[list(features.Q2_INDICES)])
    np.testing.assert_array_equal(residual_scaler.transform(q2, available)[:, :, 0], 0)


@pytest.mark.parametrize("k", [3, 5])
def test_donor_exact_mask_order_partition_and_deterministic_id_cycle(k):
    ids = [80, 4, 21, 6, 8, 11, 14]
    masks = np.ones((len(ids), k, 8), bool)
    orders = np.array([0, 0, 0, 1, 0, 0, 0])
    masks[4, 0, 0] = False
    masks[5, 1, 0] = False  # Same availability/count as id8, different whole pattern.
    masks[6, 0, 0] = False
    expected = {4: 21, 6: 6, 8: 14, 11: 11, 14: 8, 21: 80, 80: 4}
    actual = features.donor_map(ids, masks, orders, 0)
    assert actual == expected
    assert set(actual.values()) == set(ids)
    permutation = [3, 6, 0, 5, 1, 4, 2]
    assert (
        features.donor_map(
            [ids[index] for index in permutation], masks[permutation], orders[permutation], 0
        )
        == expected
    )
    # No donor may come from the other fit partition, even if its stratum matches.
    assert features.donor_map(ids[:2], masks[:2], orders[:2], 0) == {4: 80, 80: 4}


def test_donor_preserves_row_multiset_and_common_available_m_scaler():
    ids = [4, 6, 8, 11]
    packets = np.arange(4 * 3 * 8, dtype=float).reshape(4, 3, 8)
    packets[:2, 0, 0] = np.nan
    packets[2:, :, 7] = np.nan
    masks = np.isfinite(packets)
    mapping = features.donor_map(ids, masks, np.zeros(4, int), 1)
    indices = [ids.index(mapping[person]) for person in ids]
    m_rows, observed = zip(*(features.metadata_features(packet) for packet in packets))
    m_rows, observed = np.stack(m_rows), np.stack(observed)
    correct = features.fit_scaler(m_rows, ids, ids, observed)
    sham = features.fit_scaler(m_rows[indices], ids, ids, observed)
    np.testing.assert_allclose(correct.mean, sham.mean, atol=1e-15, rtol=1e-15)
    np.testing.assert_allclose(correct.scale, sham.scale, atol=1e-15, rtol=1e-15)
    np.testing.assert_array_equal(masks[indices], masks)
    np.testing.assert_array_equal(observed[indices], observed)
    # The actual implementation shares the correct scaler, avoiding re-fit rounding.
    assert correct.fit_ids == sham.fit_ids == tuple(ids)


def test_donor_keeps_singletons_and_numerically_unchanged_cases():
    masks = np.ones((3, 3, 8), bool)
    mapping = features.donor_map([4, 6, 8], masks, [0, 0, 1], 0)
    assert mapping == {4: 6, 6: 4, 8: 8}
    packet = np.ones((3, 8))
    values = {person: features.metadata_features(packet)[0] for person in mapping}
    assert sum(not np.array_equal(values[p], values[d]) for p, d in mapping.items()) == 0
    assert len(mapping) == 3
    assert tuple(inspect.signature(features.donor_map).parameters) == (
        "ids",
        "masks",
        "orders",
        "interface",
    )


def test_donor_strata_depend_on_boolean_values_not_storage_bytes():
    storage = np.ones((2, 3, 8), dtype=np.uint8)
    storage[1] = 255
    masks = storage.view(np.bool_)
    assert masks.all()
    assert masks[0].tobytes() != masks[1].tobytes()
    assert features.donor_map([4, 6], masks, [0, 0], 0) == {4: 6, 6: 4}


@pytest.mark.parametrize("ids", [[4, 4], [4, 0], [4, 6.0], [4, False]])
def test_donor_rejects_invalid_ids(ids):
    with pytest.raises(ValueError):
        features.donor_map(ids, np.ones((2, 3, 8), bool), [0, 0], 0)


@pytest.mark.parametrize("shape", [(2, 1, 8), (2, 4, 8), (2, 10, 8), (2, 3, 7), (3, 3, 8)])
def test_donor_rejects_future_or_invalid_mask_geometry(shape):
    with pytest.raises(ValueError, match="masks"):
        features.donor_map([4, 6], np.ones(shape, bool), [0, 0], 0)


@pytest.mark.parametrize("orders", [[0.0, 0.0], [True, False], [0, 2], [0], [[0, 0]]])
def test_donor_rejects_invalid_orders(orders):
    with pytest.raises(ValueError, match="orders"):
        features.donor_map([4, 6], np.ones((2, 3, 8), bool), orders, 0)


def test_donor_rejects_numeric_metadata_and_invalid_interface():
    with pytest.raises(ValueError, match="boolean"):
        features.donor_map([4, 6], np.ones((2, 3, 8)), [0, 0], 0)
    with pytest.raises(ValueError, match="interface"):
        features.donor_map([4, 6], np.ones((2, 3, 8), bool), [0, 0], 2)
