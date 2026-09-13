"""No gate fitting or full evaluate() execution; generated algebra only."""
import importlib.util
import json
from pathlib import Path
import sys
import numpy as np
import pytest

SCRIPT = Path(__file__).resolve().parents[1] / 'scripts/analysis/recorded_event_shrinkage_generated_v1.py'
sys.path.insert(0, str(SCRIPT.parent))
spec = importlib.util.spec_from_file_location('event_shrinkage', SCRIPT)
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


@pytest.mark.parametrize('state', [0, 1])
def test_extractor_schedule_and_origin(state):
    expected = [0, 0] if state == 0 else [2, -1]
    assert np.allclose(m.features(m.timestamps(state)), expected, atol=1e-9, rtol=0)
    assert np.allclose(m.features(m.timestamps(state, 1, 1e6)), expected, atol=1e-9, rtol=0)


@pytest.mark.parametrize('z', [[1, 2], [1, 2, 2, 3], [1, 2, np.nan, 4]])
def test_bad_timestamps(z):
    with pytest.raises(m.Stop): m.features(z)


@pytest.mark.parametrize('name', ['positive', 'null'])
def test_sham_joint_multiset_and_independence(name):
    target, features = m.world(name, 32)
    shuffled = features[m.donor_map(32)]
    table = np.zeros((2, 2), int)
    for y, row in zip(target.astype(int), shuffled): table[y, int(row[0] > 1)] += 1
    assert np.all(table == 8)
    assert sorted(map(tuple, features)) == sorted(map(tuple, shuffled))
    assert np.mean(np.any(features != shuffled, axis=1)) == .5


def test_schedule_constant_after_numeric_floor():
    _, features = m.world('schedule_only', 32)
    assert np.array_equal(features, np.zeros((32, 2)))


def test_psd_trace_oracle_and_balanced_loss():
    template, prior = np.diag([1., 0.]), np.eye(2) / 2
    for amount in [0., .3, 1.]:
        target = m.mix(template, prior, amount)
        assert abs(m.oracle(template, prior, target) - amount) < 1e-12
        assert np.linalg.eigvalsh(target).min() >= 0 and np.trace(target) == 1
    middle = m.mix(template, prior, .5)
    assert np.mean([np.linalg.norm(middle - t, 'fro')**2 for t in [template, prior]]) == pytest.approx(.125)
    assert m.oracle(prior, prior, template) == 0


def test_k1_metric_actuation_and_cancellation():
    template, prior = np.diag([1., 0.]), np.eye(2) / 2
    for amount in [0, .5, 1]:
        metric = m.mix(template, prior, amount)
        assert m.score(metric, template, np.eye(2)) == 1 - amount / 2
        assert m.score(metric, np.eye(2), np.eye(2)) == 1
        assert m.score(m.mix(prior, prior, amount), template, np.eye(2)) == .5


def test_zero_query_energy_and_bad_permutation_size():
    with pytest.raises(m.Stop): m.score(np.zeros((2, 2)), np.eye(2), np.eye(2))
    with pytest.raises(m.Stop): m.donor_map(7)


@pytest.mark.parametrize('bad', ['empty', 'loss', 'fits', 'scope'])
def test_parent_rejects_unverified_reports_without_fitting(bad):
    report = dict(status='GENERATED_ABILITY_PASS', scope=m.SCOPE, fits=12, additional_score_forwards=6,
                  invariants={k: True for k in m.invariant_names()}, worlds={})
    for name in ['positive', 'null', 'schedule_only']:
        report['worlds'][name] = {'template_frobenius_mse': {
            arm: 0. if name == 'positive' and arm == 'QM' else .125 for arm in ['Q', 'Q2', 'QM', 'SHAM']}}
    m.validate_report(report)
    if bad == 'empty': report['invariants'] = {}
    if bad == 'loss': report['worlds']['positive']['template_frobenius_mse']['QM'] = .5
    if bad == 'fits': report['fits'] = 13
    if bad == 'scope': report['scope'] = {}
    with pytest.raises(m.Stop): m.validate_report(report)


def test_manifest_budget_pin_without_fitting():
    cfg = json.loads((SCRIPT.parents[2] / 'configs/recorded_event_shrinkage_generated_v1.json').read_text())
    cfg['producer_sha256'] = m.sha(SCRIPT)
    m.check_manifest(cfg)
    cfg['fits'] = 16
    with pytest.raises(m.Stop): m.check_manifest(cfg)
