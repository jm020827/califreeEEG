"""Synthetic-only qualification; no datasets, paths, readers or network."""

from dataclasses import FrozenInstanceError, replace
from unittest.mock import patch

import numpy as np
import pytest

from cfeg.baselines.welch_source import WelchSpec, fit_source, welch_power

SPEC = WelchSpec(250, 500, 128, 64, 512)
CLASSES = (10, 20, 30, 40, 50)


@pytest.fixture(scope="module")
def synthetic():
    rng = np.random.default_rng(20260914)
    time = np.arange(500) / 250
    x, y, ids = [], [], []
    for p in range(5):
        for c, frequency in zip(CLASSES, (8, 13, 18, 23, 28)):
            for _ in range(2):
                phase = rng.uniform(0, 2*np.pi)
                x.append((1 + .1*p) * np.sin(2*np.pi*frequency*time + phase)
                         + .03*rng.normal(size=500) + p)
                y.append(c)
                ids.append(f"synthetic-{p}")
    return np.asarray(x), np.asarray(y), ids


@pytest.fixture(scope="module")
def training_trace():
    return []


@pytest.fixture(scope="module")
def fitted(synthetic, training_trace):
    pytest.importorskip("sklearn")
    from sklearn.svm import SVC
    x, y, ids = synthetic
    original_x, original_y, original_ids = x.copy(), y.copy(), ids.copy()
    original_fit = SVC.fit
    def traced_fit(self, z, binary):
        result = original_fit(self, z, binary)
        training_trace.append((z.copy(), binary.copy(), self.decision_function(z).copy()))
        return result
    with patch.object(SVC, "fit", traced_fit):
        model = fit_source(x[:30], y[:30], ids[:30], target_ids=ids[30::10],
                           classes=CLASSES, spec=SPEC)
    np.testing.assert_array_equal(x, original_x)
    np.testing.assert_array_equal(y, original_y)
    assert ids == original_ids
    return model


@pytest.mark.parametrize("nfft", [128, 129, 512])
def test_direct_dft_and_parseval(nfft):
    spec = WelchSpec(250, 500, 128, 64, nfft)
    x = np.random.default_rng(77).normal(size=(2, 500))
    observed = welch_power(x, spec)
    centered = x - x.mean(axis=1, keepdims=True)
    window = .54 - .46*np.cos(2*np.pi*np.arange(128)/127)
    dft = np.exp(-2j*np.pi*np.arange(nfft//2+1)[:, None]*np.arange(128)/nfft)
    segment_powers, time_energies = [], []
    for start in range(0, 500-128+1, 64):
        segment = centered[:, start:start+128]*window
        p = np.abs(segment @ dft.T)**2/(250*(window @ window))
        p[:, 1:(-1 if nfft % 2 == 0 else None)] *= 2
        segment_powers.append(p)
        time_energies.append(np.sum(segment**2, axis=1)/(window @ window))
    expected = np.mean(segment_powers, axis=0)
    np.testing.assert_allclose(observed, expected, atol=1e-14, rtol=1e-11)
    np.testing.assert_allclose(observed.sum(axis=1)*250/nfft,
                               np.mean(time_energies, axis=0), atol=1e-13)


def test_trial_independence_scale_and_input_unchanged(synthetic):
    x = synthetic[0][:2].copy()
    before = x.copy()
    power = welch_power(x, SPEC)
    np.testing.assert_array_equal(x, before)
    np.testing.assert_allclose(welch_power(x+100, SPEC), power, atol=1e-13)
    np.testing.assert_allclose(welch_power(x*3, SPEC), power*9, atol=1e-13)
    np.testing.assert_array_equal(welch_power(x[:1], SPEC), power[:1])


@pytest.mark.parametrize("args", [
    (0, 500, 128, 64, 512), (True, 500, 128, 64, 512),
    (250, 500.0, 128, 64, 512), (250, 100, 128, 64, 512),
    (250, 500, 128, 128, 512), (250, 500, 128, -1, 512),
    (250, 500, 128, 64, 64), (float("inf"), 500, 128, 64, 512),
])
def test_invalid_spec(args):
    with pytest.raises(ValueError):
        WelchSpec(*args)


@pytest.mark.parametrize("x", [np.zeros((1, 500)), np.ones((2, 500)),
    np.full((1, 500), .1), np.full((1, 500), -.1),
    np.ones((1, 499)), np.ones(500), np.ones((1, 500), dtype=complex),
    np.full((1, 500), np.nan), np.full((1, 500), np.inf),
    np.full((1, 500), 1e308), np.ones((1, 500), dtype=bool), np.empty((0, 500))])
def test_invalid_windows(x):
    with pytest.raises(ValueError):
        welch_power(x, SPEC)


def test_constant_rejected_before_welch(monkeypatch):
    from cfeg.baselines import welch_source
    def forbidden(*args, **kwargs):
        pytest.fail("constant input must fail before spectral calculation")
    monkeypatch.setattr(welch_source, "welch", forbidden)
    with pytest.raises(ValueError, match="constant_trial"):
        welch_power(np.full((1, 500), .1), SPEC)


def test_fit_sanity_scaler_and_immutable_export(fitted, synthetic):
    x, y, ids = synthetic
    before = repr(fitted)
    source_power = welch_power(x[:30], SPEC)
    np.testing.assert_allclose(fitted.mean, source_power.mean(axis=0), atol=1e-15)
    assert np.mean(fitted.predict(x[30:], ids[30:]) == y[30:]) >= .90
    scores = fitted.decision_function(x[30:], ids[30:])
    rows = np.concatenate([fitted.decision_function(x[i:i+1], ids[i:i+1])
                           for i in range(30, 50)])
    np.testing.assert_allclose(scores, rows, atol=1e-12)
    fitted.predict(x[30:]*4 + 12, ids[30:])
    assert repr(fitted) == before
    with pytest.raises(FrozenInstanceError):
        fitted.mean = ()


def test_source_permutation_and_positive_orientation(fitted, synthetic):
    x, y, ids = synthetic
    order = np.arange(30)[::-1]
    second = fit_source(x[order], y[order], [ids[i] for i in order],
                        target_ids=ids[30::10], classes=CLASSES, spec=SPEC)
    np.testing.assert_array_equal(second.predict(x[30:], ids[30:]),
                                  fitted.predict(x[30:], ids[30:]))
    scores = fitted.decision_function(x[30:], ids[30:])
    for i, c in enumerate(CLASSES):
        assert scores[y[30:] == c, i].mean() > 0
        assert scores[y[30:] != c, i].mean() < 0


def test_export_matches_actual_binary_solver(fitted, training_trace, synthetic):
    assert len(training_trace) == 5
    y = synthetic[1][:30]
    for i, c in enumerate(CLASSES):
        z, binary, margins = training_trace[i]
        assert z.shape == (30, 257)
        np.testing.assert_array_equal(binary, np.where(y == c, 1, -1))
        np.testing.assert_allclose(z @ fitted.coefficients[i] + fitted.intercepts[i],
                                   margins, atol=1e-11)


def test_direct_export_copies_mutable_inputs(fitted):
    mean, coef = list(fitted.mean), [list(row) for row in fitted.coefficients]
    copied = replace(fitted, mean=mean, coefficients=coef)
    mean[0], coef[0][0] = 987, 654
    assert copied == fitted


@pytest.mark.parametrize("change", [{"mean": ()}, {"scale": (0.,)*257},
    {"coefficients": ((1.,)*257,)}, {"intercepts": (float("nan"),)*5},
    {"classes": (20, 10, 30, 40, 50)}, {"target_ids": ("synthetic-0",)}])
def test_invalid_export_rejected(fitted, change):
    with pytest.raises(ValueError):
        replace(fitted, **change)


@pytest.mark.parametrize("role", ["synthetic-0", "unregistered", " synthetic-4", ""])
def test_query_role_rejected(fitted, synthetic, role):
    with pytest.raises(ValueError):
        fitted.predict(synthetic[0][30:31], [role])


@pytest.mark.parametrize("case", ["overlap", "single_source", "missing_class", "class_person",
                                  "float_label", "labels_length", "duplicate_target"])
def test_fit_validation_before_solver(synthetic, case):
    x, y, ids = synthetic
    x, y, ids = x[:30].copy(), y[:30].copy(), ids[:30].copy()
    target = ["synthetic-3", "synthetic-4"]
    if case == "overlap": target[0] = ids[0]
    if case == "single_source": ids = ["one"]*30
    if case == "missing_class": y[y == 50] = 40
    if case == "class_person":
        for i in np.flatnonzero(y == 10): ids[i] = "only_this_person"
    if case == "float_label": y = y.astype(float)
    if case == "labels_length": x = x[:29]
    if case == "duplicate_target": target = ["synthetic-3"]*2
    with pytest.raises(ValueError):
        fit_source(x, y, ids, target_ids=target, classes=CLASSES, spec=SPEC)


def test_no_query_refit(fitted, synthetic, monkeypatch):
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import SVC
    def forbidden(*args, **kwargs):
        pytest.fail("query must not fit")
    monkeypatch.setattr(StandardScaler, "fit", forbidden)
    monkeypatch.setattr(SVC, "fit", forbidden)
    x, _, ids = synthetic
    fitted.predict(x[30:], ids[30:])


def test_nonconvergence_is_failure(synthetic, monkeypatch):
    pytest.importorskip("sklearn")
    from sklearn.svm import SVC
    def not_converged(self, *args, **kwargs):
        self.fit_status_ = 1
        return self
    monkeypatch.setattr(SVC, "fit", not_converged)
    x, y, ids = synthetic
    with pytest.raises(ValueError, match="svm_not_converged"):
        fit_source(x[:30], y[:30], ids[:30], target_ids=ids[30::10],
                   classes=CLASSES, spec=SPEC)
