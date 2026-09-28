"""Generated participant/control boundaries; zero optimizer updates or data IO."""

import copy

import numpy as np
import pytest

from cfeg.analysis.dan_roles import FoldSupport, FreezeGate, SupportArrays, fold_roles, sham_mapping


def fixture():
    rng = np.random.default_rng(73)
    return SupportArrays(
        tuple(range(12)), np.zeros(12, dtype=int), rng.normal(size=(12, 2, 3, 6, 12, 2, 32)),
        rng.normal(size=(12, 2, 3, 5, 2)), rng.normal(size=(12, 2, 3, 5, 2)),
        rng.uniform(0, 100, size=(12, 2, 5, 2)))


def test_fold_roles_and_source_only_scalers():
    data = fixture()
    role = FoldSupport(data, 0, sham_seed=7)
    assert role.source_ids == (1, 2, 4, 5, 7)
    assert role.target_ids == (0, 3, 6, 9)
    assert role.fit_ids == (1, 2, 4, 5)
    altered = copy.deepcopy(data)
    for person in (0, 3, 6, 7, 9, 10, 11):
        altered.q[person] += 1000
        altered.q2[person] += 500
        altered.impedance[person] += 1e6
    other = FoldSupport(altered, 0, sham_seed=7)
    for key in role.scalers:
        assert role.scalers[key].fit_subject_ids == role.fit_ids
        np.testing.assert_array_equal(role.scalers[key].mean, other.scalers[key].mean)
        np.testing.assert_array_equal(role.scalers[key].scale, other.scalers[key].scale)
    with pytest.raises(ValueError):
        fold_roles((3, 2, 1), 0)


@pytest.mark.parametrize("arm", ["U", "Q", "Q2", "QM", "SHAM"])
def test_packets_contain_only_source_role_and_paid_target_prefix(arm):
    data = fixture()
    role = FoldSupport(data, 1, sham_seed=7)
    source, support, weights = role.packet(1, 0, 2, 2, arm)
    np.testing.assert_array_equal(source, data.bands[list(role.source_ids), 0, 2])
    np.testing.assert_array_equal(support, data.bands[1, 0, 2, :2])
    assert weights.shape == (2, 2)
    assert not np.shares_memory(source, data.bands)
    assert not np.shares_memory(support, data.bands)
    changed = copy.deepcopy(data)
    changed.bands[1, :, :, 2:] = 1000
    changed.q[1, :, :, 2:] = 1000
    changed.q2[1, :, :, 2:] = 1000
    changed.impedance[1, :, 2:] = 1000
    other = FoldSupport(changed, 1, sham_seed=7)
    for left, right in zip((source, support, weights), other.packet(1, 0, 2, 2, arm)):
        np.testing.assert_array_equal(left, right)
    for target, k in ((0, 2), (1, 1), (1, 6)):
        with pytest.raises(ValueError):
            role.packet(target, 0, 2, k, arm)


def test_q_is_invariant_to_metadata_but_sham_preserves_strata_and_is_stable():
    data = fixture()
    role = FoldSupport(data, 0, sham_seed=7)
    assert all(k != v for k, v in role.donors.items())
    assert set(role.donors.values()) == set(role.target_ids)
    altered = copy.deepcopy(data)
    altered.impedance[:] = np.nan
    other = FoldSupport(altered, 0, sham_seed=7)
    for k in (2, 3, 5):
        for left, right in zip(role.packet(0, 0, 0, k, "Q"), other.packet(0, 0, 0, k, "Q")):
            np.testing.assert_array_equal(left, right)
        np.testing.assert_array_equal(other.packet(0, 0, 0, k, "QM")[2],
                                      other.packet(0, 0, 0, k, "Q")[2])
    assert other.donors == role.donors
    orders = {0: 0, 3: 0, 6: 1, 9: 1}
    donors = sham_mapping((0, 3, 6, 9), orders, 7)
    assert all(orders[k] == orders[v] and k != v for k, v in donors.items())
    with pytest.raises(ValueError, match="at least two"):
        sham_mapping((0, 3, 6), {0: 0, 3: 0, 6: 1}, 7)


def test_freeze_requires_all_artifacts_before_single_query_batch():
    gate = FreezeGate({("Q", 0), ("QM", 0)})
    with pytest.raises(RuntimeError):
        gate.reveal_once()
    with pytest.raises(ValueError):
        gate.register(("Q", 0), "not-a-digest")
    gate.register(("Q", 0), "a" * 64)
    with pytest.raises(RuntimeError):
        gate.seal()
    with pytest.raises(RuntimeError):
        gate.register(("Q", 0), "b" * 64)
    gate.register(("QM", 0), "b" * 64)
    gate.seal()
    with pytest.raises(RuntimeError):
        gate.register(("QM", 0), "c" * 64)
    gate.reveal_once()
    with pytest.raises(RuntimeError):
        gate.reveal_once()
