"""Generated inputs only; no MAT, NPZ, subject data or persisted outcomes."""

import inspect

import numpy as np
import pytest
from scipy.linalg import orth

from cfeg import mamem_reference_probe_v1 as probe


@pytest.fixture(scope="module")
def nominal():
    return probe.reference_bank(probe.NOMINAL_FREQUENCIES)


def trace(frequency, phase=0.0):
    time = np.arange(1, 501) / 250
    return np.cos(2 * np.pi * frequency * time + phase)


@pytest.mark.parametrize("label", range(5))
def test_nominal_positive_canary(label, nominal):
    query = trace(probe.NOMINAL_FREQUENCIES[label], 0.73)
    assert probe.predict(query, nominal) == label
    assert probe.scores(query, nominal)[label] == pytest.approx(1, abs=1e-12)


@pytest.mark.parametrize("label", range(5))
def test_shifted_bank_actuates_scores(label, nominal):
    frequencies = [6.49, 7.34, 8.31, 9.60, 11.61]
    shifted = probe.reference_bank(frequencies)
    query = trace(frequencies[label])
    assert probe.predict(query, shifted) == label
    result = probe.scores(query, shifted)
    assert result[label] == pytest.approx(1, abs=1e-12)
    assert np.max(np.abs(result - probe.scores(query, nominal))) > 1e-4


def test_independent_lstsq_projection(nominal):
    query = np.random.default_rng(901).normal(size=500)
    centered = query - query.mean()
    time = np.arange(1, 501) / 250
    expected = []
    for frequency in probe.NOMINAL_FREQUENCIES:
        raw = np.column_stack([function(2 * np.pi * h * frequency * time)
                               for h in (1, 2) for function in (np.sin, np.cos)])
        raw -= raw.mean(axis=0)
        fitted = raw @ np.linalg.lstsq(raw, centered, rcond=None)[0]
        expected.append(float(fitted @ fitted / (centered @ centered)))
    np.testing.assert_allclose(probe.scores(query, nominal), expected, atol=1e-10, rtol=0)


@pytest.mark.parametrize("scale,offset", [(1, 29), (-7, 0), (3, -4), (1e200, 0),
                                         (1e-200, 0)])
def test_scale_sign_offset_invariance(scale, offset, nominal):
    query = trace(8.57, 0.8) + 0.3 * trace(17.14, -0.2)
    expected = probe.scores(query, nominal)
    np.testing.assert_allclose(probe.scores(scale * query + offset, nominal),
                               expected, atol=1e-10, rtol=0)


def test_reference_phase_rotation_invariance(nominal):
    time = np.arange(1, 501) / 250
    bank = []
    for frequency in probe.NOMINAL_FREQUENCIES:
        columns = []
        for harmonic in (1, 2):
            phase = 0.42 * harmonic
            angle = 2 * np.pi * harmonic * frequency * time + phase
            columns.extend([np.sin(angle), np.cos(angle)])
        raw = np.column_stack(columns)
        bank.append(orth(raw - raw.mean(axis=0)))
    query = np.random.default_rng(77).normal(size=500)
    np.testing.assert_allclose(probe.scores(query, bank), probe.scores(query, nominal),
                               atol=1e-10, rtol=0)


def test_null_orthogonal_to_whole_bank(nominal):
    union = orth(np.column_stack([np.ones(500), *nominal]))
    query = np.random.default_rng(902).normal(size=500)
    query -= union @ (union.T @ query)
    assert np.max(probe.scores(query, nominal)) < 1e-20


def test_inputs_unchanged_and_label_free_signature(nominal):
    query = trace(10)
    query_copy, bank_copy = query.copy(), nominal.copy()
    probe.predict(query, nominal)
    np.testing.assert_array_equal(query, query_copy)
    np.testing.assert_array_equal(nominal, bank_copy)
    assert list(inspect.signature(probe.scores).parameters) == ["query", "bank"]
    assert list(inspect.signature(probe.predict).parameters) == ["query", "bank"]


def support_samples():
    return [1001 + np.arange(20) * step for step in (19, 17, 15, 13, 11)]


def test_sample_clock_formula_and_translation():
    samples = support_samples()
    saved = [value.copy() for value in samples]
    expected = 250 / (2 * np.array([19, 17, 15, 13, 11]))
    np.testing.assert_allclose(probe.sample_clock_frequencies(samples), expected,
                               atol=1e-12, rtol=0)
    np.testing.assert_array_equal(probe.sample_clock_frequencies(samples),
                                  probe.sample_clock_frequencies([v + 123 for v in samples]))
    for value, before in zip(samples, saved):
        np.testing.assert_array_equal(value, before)


@pytest.mark.parametrize("bad", [np.zeros(500), np.ones(500), np.full(500, np.nan),
                                 np.full(500, np.inf), np.ones((1, 500)),
                                 np.ones(499), np.ones(500, dtype=complex)])
def test_bad_query(bad, nominal):
    with pytest.raises(ValueError):
        probe.scores(bad, nominal)


@pytest.mark.parametrize("bad", [[6, 7, 8, 9], [6, 7, 7, 9, 11], [7, 6, 8, 9, 11],
                                 [0, 7, 8, 9, 11], [6, 7, 8, 9, 63],
                                 [6, 7, 8, 9, np.nan], [True] * 5])
def test_bad_frequencies(bad):
    with pytest.raises(ValueError):
        probe.reference_bank(bad)


@pytest.mark.parametrize("bad", [[1, 2, 3], [1, 2, 2, 3], [4, 3, 2, 1],
                                 [1, 2.5, 3, 4], [0, 2, 3, 4], [1, 2, 3, 501],
                                 [1, 2, 3, np.inf], [2**63] * 4, [True] * 4])
def test_bad_sample_sequence(bad):
    samples = support_samples()
    samples[0] = bad
    with pytest.raises(ValueError):
        probe.sample_clock_frequencies(samples)


@pytest.mark.parametrize("kind", ["shape", "nonorthogonal", "uncentered", "nan"])
def test_bad_bank(kind, nominal):
    bank = nominal.copy()
    if kind == "shape":
        bank = bank[:4]
    elif kind == "nonorthogonal":
        bank *= 2
    elif kind == "uncentered":
        bank[0, :, 0] = 1 / np.sqrt(500)
    else:
        bank[0, 0, 0] = np.nan
    with pytest.raises(ValueError):
        probe.scores(trace(10), bank)


def test_exact_tie_smallest_class(nominal):
    equal_bank = np.repeat(nominal[:1], 5, axis=0)
    assert probe.predict(trace(10), equal_bank) == 0


def test_missing_support_sequence():
    with pytest.raises(ValueError, match="five_support_sequences"):
        probe.sample_clock_frequencies(support_samples()[:4])
