"""Pure generated timing models, not human EEG evaluation."""
import importlib.util
from pathlib import Path
import numpy as np
import pytest

SCRIPT = Path(__file__).resolve().parents[1] / 'scripts/analysis/mamem_timing_identifiability_v1.py'
spec = importlib.util.spec_from_file_location('timing_generated', SCRIPT)
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


def test_fixed_three_worlds():
    r = m.report()
    assert r['scenario_count'] == 3 and r['random_draws'] == 0
    assert r['status'] == 'GENERATED_INVARIANTS_PASS'
    assert all(r['invariants'].values())


def test_equality_is_DIN_not_response():
    _, _, _, worlds = m.construct()
    a, b = worlds[1:]
    assert np.array_equal(a['z'], b['z']) and np.array_equal(a['s'], b['s'])
    assert not np.array_equal(a['response'], b['response'])


def test_negative_absolute_delay_rejected():
    with pytest.raises(ValueError, match='negative_absolute_delay'):
        m.observe(np.array([1000.]), np.array([-1.]))


def test_timestamp_only_postassignment_is_a_different_model():
    _, j, _, worlds = m.construct()
    normal = worlds[0]
    # Definition test only; no additional science scenario/fit.
    post_z = normal['z'] + j
    post_s = normal['s'].copy()
    assert np.array_equal(post_s, normal['s'])
    assert not np.array_equal(post_z, normal['z'])
    assert not np.array_equal(post_s, worlds[2]['s'])


def test_oracle_not_claimed_as_measured_correction():
    r = m.report()
    assert not r['scope']['quantized_DIN_to_true_j_recovery']
    assert not r['scope']['EEG_plus_DIN_nonidentifiability_proved']
    assert not r['scope']['learned_arms_instantiated']
    assert r['scope']['accuracy_evaluations'] == 0


def test_report_deterministic():
    assert m.report() == m.report()
