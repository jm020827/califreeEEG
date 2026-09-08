import numpy as np
import pytest
from scipy import linalg

from cfeg.analysis.metadata_trca_prior import TrcaModel, fit_trca, score_trca, trca_matrices
from cfeg.analysis.trca_support_geometry import (
    distribution,
    matrix_geometry,
    summarize,
    support_geometry,
)


def test_isotropic_covariance_direction_and_normalization():
    s = np.diag([5.0, 2.0, 1.0])
    c = np.eye(3) * 7
    m, full, iso = matrix_geometry(s, c)
    assert m["direction_sine"] == 0
    np.testing.assert_allclose(full, iso)
    assert m["full_direction_rho"] == pytest.approx(0.1)
    assert m["iso_direction_rho"] == pytest.approx(0.1)
    assert m["iso_c_to_b_norm_factor"] == pytest.approx(1 / np.sqrt(1.1))
    assert m["effective_rank"] == pytest.approx(3)
    np.testing.assert_allclose(m["tau_over_covariance_eigenvalues"], 0.1)


def test_trace_fraction_is_not_direction_fraction():
    s = np.diag([0.5, 1.0, 3.0])
    c = np.diag([1e-4, 1.0, 8.0])
    m, _, _ = matrix_geometry(s, c)
    assert m["tau_over_covariance_eigenvalues"][0] > 3000
    assert m["full_direction_rho"] > 3000
    assert m["modes_tau_over_lambda_ge10"] == 1


def test_common_amplitude_scale_invariance():
    rng = np.random.default_rng(817)
    x = rng.normal(size=(5, 8, 70))
    s, c = trca_matrices(x)
    first, full0, iso0 = matrix_geometry(s, c)
    s2, c2 = trca_matrices(100 * x)
    second, full1, iso1 = matrix_geometry(s2, c2)
    for key in first:
        if key != "tau":
            np.testing.assert_allclose(first[key], second[key], rtol=1e-9, atol=1e-10)
    assert second["tau"] == pytest.approx(first["tau"] * 10000)
    assert abs(full0 @ full1) == pytest.approx(1)
    assert abs(iso0 @ iso1) == pytest.approx(1)


def test_matrices_independent_pair_sum_and_native_fit_match():
    rng = np.random.default_rng(911)
    support = rng.normal(size=(3, 12, 5, 8, 20))
    rows, matrices, full, iso = support_geometry(
        support, subject=999, interface="artificial", samples=20, k=3
    )
    for index, row in enumerate(rows):
        x = support[:, row["class"], row["band"]]
        independent_s = sum(x[a] @ x[b].T for a in range(3) for b in range(3) if a != b)
        concatenated = np.concatenate(x, axis=1)
        centered = concatenated - concatenated.mean(axis=1, keepdims=True)
        np.testing.assert_allclose(matrices[index, 0], independent_s, atol=1e-12)
        np.testing.assert_allclose(matrices[index, 1], centered @ centered.T, atol=1e-12)
    for gamma, units in ((0, full), (0.1, iso)):
        native_convention = fit_trca(support, np.ones((5, 8)), gamma)
        for index, row in enumerate(rows):
            vector = native_convention.filters[row["band"], :, row["class"]]
            assert abs(vector @ units[index]) / np.linalg.norm(vector) == pytest.approx(
                1, abs=1e-13
            )


def test_normalization_factor_holds_iso_direction_fixed():
    rng = np.random.default_rng(417)
    s, c = trca_matrices(rng.normal(size=(3, 8, 20)))
    m, _, iso = matrix_geometry(s, c)
    c_norm = iso / np.sqrt(iso @ c @ iso)
    b = c + m["tau"] * np.eye(8)
    b_norm = iso / np.sqrt(iso @ b @ iso)
    np.testing.assert_allclose(b_norm, c_norm * m["iso_c_to_b_norm_factor"])


def test_ensemble_common_scale_invariant_class_specific_scale_not_invariant():
    rng = np.random.default_rng(392)
    templates = rng.normal(size=(3, 1, 2, 20))
    query = rng.normal(size=(4, 1, 2, 20))
    filters = rng.normal(size=(1, 2, 3))
    model = lambda f: TrcaModel(f, templates, {})
    base = score_trca(model(filters), query, [1])[0]
    np.testing.assert_allclose(base, score_trca(model(filters / np.sqrt(1.1)), query, [1])[0])
    changed = score_trca(model(filters * [0.2, 0.7, 0.99]), query, [1])[0]
    assert np.max(np.abs(changed - base)) > 0.01


@pytest.mark.parametrize(
    "s,c",
    [
        (np.eye(2), np.eye(2)),  # gap failure
        (np.diag([2.0, 1.0]), np.diag([0.0, 1.0])),
        (np.diag([2.0, 1.0]), np.diag([-1.0, 1.0])),
        (np.diag([2.0, np.nan]), np.eye(2)),
        (np.diag([2.0, 1.0]), np.diag([1.0, np.inf])),
        (np.ones((2, 3)), np.eye(2)),
        (np.array([[2.0, 1.0], [0.0, 1.0]]), np.eye(2)),
        (np.eye(2, dtype=complex), np.eye(2)),
    ],
)
def test_no_numerical_rescue(s, c):
    with pytest.raises(ValueError):
        matrix_geometry(s, c)


def test_direction_sine_sign_invariant(monkeypatch):
    s, c = np.array([[2.0, 0.1], [0.1, 1.0]]), np.diag([0.5, 2.0])
    baseline = matrix_geometry(s, c)[0]
    original = linalg.eig
    count = 0

    def flipped(*args):
        nonlocal count
        values, vectors = original(*args)
        count += 1
        return values, -vectors if count == 2 else vectors

    monkeypatch.setattr(linalg, "eig", flipped)
    changed = matrix_geometry(s, c)[0]
    assert baseline["direction_sine"] == pytest.approx(changed["direction_sine"])


def test_distribution_and_complete_grid():
    assert distribution([0, 1, 2, 3])["p10"] == pytest.approx(0.3)
    rng = np.random.default_rng(611)
    x = rng.normal(size=(5, 12, 5, 8, 20))
    rows = []
    for k in (3, 5):
        rows.extend(support_geometry(x[:k], subject=999, interface="dry", samples=20, k=k)[0])
    plan = {"source_subject_ids": [999], "interfaces": ["dry"], "sample_counts": [20]}
    summary, bands = summarize(rows, plan)
    assert len(bands) == 10 and len(summary) == 4
    assert summary[0]["filter_records"] == 60
    assert summary[0]["participant_count"] == 1
    assert summary[0]["class_norm_factor_cv"]["n"] == 5
    assert summary[0]["participant_mean_descriptive"]["direction_sine"]["n"] == 1
    for bad in (rows[:-1], rows + rows[:1], rows[:-1] + rows[:1]):
        with pytest.raises(ValueError):
            summarize(bad, plan)


@pytest.mark.parametrize("values", [[], [np.nan], [[1, 2]]])
def test_bad_summary_input(values):
    with pytest.raises(ValueError):
        distribution(values)


@pytest.mark.parametrize("scale", [1e-200, 1e200])
def test_extreme_matrix_scaling_keeps_participation_rank(scale):
    s, c = np.diag([0.02, 0.9]), np.diag([0.01, 1.0])
    baseline = matrix_geometry(s, c)[0]
    scaled = matrix_geometry(s * scale, c * scale)[0]
    assert scaled["effective_rank"] == pytest.approx(baseline["effective_rank"])


def test_large_finite_summary_does_not_overflow():
    assert distribution([1e308, 1e308])["mean"] == 1e308
    assert distribution([-1e308, 1e308])["median"] == 0
