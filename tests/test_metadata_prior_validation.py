"""Independent small-array checks; never import the closed synthetic v1 suite."""

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from cfeg.analysis import metadata_prior_validation as m


def linear_fixture():
    x = (np.arange(9)[:, None] - 4 + np.array([[-1.0, 1.0]]))[:, None, None, :, None]
    y = np.broadcast_to(np.array([-1.0, 1.0]), (9, 1, 1, 2)).copy()
    return x, y, np.arange(100, 109)


def test_context_is_channel_local_permutation_equivariant_and_copied():
    x = np.arange(3 * 2 * 2 * 4 * 3.0).reshape(3, 2, 2, 4, 3)
    expected = x.copy()
    design = m.q_design(x, "context")
    np.testing.assert_array_equal(design[..., :3], x)
    np.testing.assert_allclose(design[..., 3:], np.broadcast_to(x.mean(-2, keepdims=True), x.shape))
    order = [2, 0, 3, 1]
    np.testing.assert_array_equal(m.q_design(x[..., order, :], "context"), design[..., order, :])
    x[1] += 1000
    np.testing.assert_array_equal(m.q_design(x, "context")[[0, 2]], design[[0, 2]])
    assert not np.shares_memory(m.q_design(expected, "local"), expected)


@pytest.mark.parametrize("alpha", m.ALPHAS)
def test_ridge_matches_independent_gram_or_pseudoinverse(alpha):
    rng = np.random.default_rng(7091)
    x = rng.normal(size=(16, 3))
    y = rng.normal(size=16)
    model = m.fit_ridge(x, y, alpha)
    z = (x - x.mean(0)) / x.std(0)
    expected = (
        np.linalg.solve(z.T @ z / len(x) + alpha * np.eye(3), z.T @ (y - y.mean()) / len(x))
        if alpha
        else np.linalg.pinv(z) @ (y - y.mean())
    )
    np.testing.assert_allclose(model.coefficient, expected, atol=1e-12)
    assert model.intercept == y.mean()


def test_ridge_residual_is_deshinkage_not_new_information():
    x = np.array([[1, 2], [2, 0], [4, 1], [0, -2], [-1, 1], [3, 4]], dtype=float)
    y = np.array([1, 0, 2, -3, 4, 1.0])
    q = m.fit_ridge(x, y, 0.1)
    residual = m.fit_ridge(x, y - q.predict(x), 0.1)
    z = (x - x.mean(0)) / x.std(0)
    a = z.T @ z / len(x)
    b = z.T @ (y - y.mean()) / len(x)
    inverse = np.linalg.inv(a + 0.1 * np.eye(2))
    np.testing.assert_allclose(residual.coefficient, 0.1 * inverse @ inverse @ b, atol=1e-12)
    h = z @ inverse @ z.T / len(x)
    np.testing.assert_allclose(
        q.predict(x) + residual.predict(x), y.mean() + (2 * h - h @ h) @ (y - y.mean()), atol=1e-12
    )


def test_rank_deficient_and_constant_columns_remain_finite():
    x = np.column_stack((np.arange(6.0), np.arange(6.0), np.ones(6)))
    model = m.fit_ridge(x, np.arange(6.0), 0)
    np.testing.assert_allclose(model.predict(x), np.arange(6.0), atol=1e-12)
    assert model.scale[-1] == 1
    assert model.design_rank == 1


def test_finite_but_unrepresentable_scale_and_prediction_are_rejected():
    with pytest.raises(ValueError, match="derived feature scale"):
        m.fit_ridge(np.array([[-2.0], [-1], [1], [2]]) * 1e200, [-2, -1, 1, 2], 0.1)
    model = m.RidgeModel(np.array([0.0]), np.array([1.0]), np.array([1e200]), 0.0, 0.1, 1)
    with pytest.raises(ValueError, match="ridge prediction"):
        model.predict([[1e200]])


def test_nested_crossfit_recovers_declared_context_identity_and_records_all_candidates():
    x, y, ids = linear_fixture()
    original = x.copy(), y.copy(), ids.copy()
    model, oof, receipt = m.crossfit_q(x, y, ids)
    assert model.representation == "context"
    assert model.ridge.alpha == 0
    np.testing.assert_allclose(oof, y, atol=1e-10)
    np.testing.assert_allclose(model.predict(x), y, atol=1e-10)
    assert receipt["training_size_mismatch"] == {
        "oof_fit_participants": [6, 6, 6],
        "final_fit_participants": 9,
    }
    json.dumps(receipt, allow_nan=False)
    for received, before in zip((x, y, ids), original):
        np.testing.assert_array_equal(received, before)
    all_held = []
    for outer in receipt["oof_folds"]:
        fit, held = set(outer["fit_ids"]), set(outer["oof_ids"])
        assert not fit & held and fit | held == set(ids)
        all_held.extend(held)
        inner = outer["inner_selection"]
        assert set(inner["participant_ids"]) == fit
        assert len(inner["candidates"]) == 12
        for split in inner["splits"]:
            a, b = set(split["fit_ids"]), set(split["validation_ids"])
            assert not a & b and a | b == fit
            assert not (a | b) & held
        for candidate in inner["candidates"]:
            assert candidate["mean_participant_mse"] == np.mean(candidate["participant_mse"])
    assert sorted(all_held) == list(ids)


def test_oof_subject_target_cannot_change_own_model_selection_scaler_or_prediction():
    x, y, ids = linear_fixture()
    _, old, before = m.crossfit_q(x, y, ids)
    y2 = y.copy()
    y2[0] += 1e6
    _, new, after = m.crossfit_q(x, y2, ids)
    np.testing.assert_array_equal(old[0], new[0])
    assert before["oof_folds"][0] == after["oof_folds"][0]


def test_oof_subject_features_cannot_change_own_training_or_selection_receipt():
    x, y, ids = linear_fixture()
    _, _, before = m.crossfit_q(x, y, ids)
    x2 = x.copy()
    x2[0] += 1000
    _, _, after = m.crossfit_q(x2, y, ids)
    assert before["oof_folds"][0] == after["oof_folds"][0]


def test_unequal_fold_sizes_have_equal_participant_mass():
    x, y, ids = linear_fixture()
    _, receipt = m.select_q(x[:7], y[:7], ids[:7])
    assert [len(f["validation_ids"]) for f in receipt["splits"]] == [3, 2, 2]
    for candidate in receipt["candidates"]:
        losses = np.array(candidate["participant_mse"])
        # Refit each declared candidate independently and reconstruct per-person losses.
        design = m.q_design(x[:7], candidate["representation"])
        for split in receipt["splits"]:
            train, valid = (
                np.isin(ids[:7], split["fit_ids"]),
                np.isin(ids[:7], split["validation_ids"]),
            )
            z = design[train].reshape(-1, design.shape[-1])
            mean, scale = z.mean(0), z.std(0)
            scale[scale < 1e-12] = 1
            z = (z - mean) / scale
            target = y[:7][train].ravel()
            alpha = candidate["alpha"]
            coef = np.linalg.pinv(z.T @ z / len(z) + alpha * np.eye(z.shape[-1])) @ (
                z.T @ (target - target.mean()) / len(z)
            )
            pred = ((design[valid] - mean) / scale) @ coef + target.mean()
            expected = ((pred - y[:7][valid]) ** 2).reshape(valid.sum(), -1).mean(1)
            np.testing.assert_allclose(losses[valid], expected, atol=1e-11)
        assert candidate["mean_participant_mse"] == losses.sum() / 7


def test_row_permutation_retains_group_partition_and_predictions():
    x, y, ids = linear_fixture()
    order = np.array([3, 7, 1, 0, 8, 2, 5, 6, 4])
    _, base, _ = m.crossfit_q(x, y, ids)
    _, perm, _ = m.crossfit_q(x[order], y[order], ids[order])
    np.testing.assert_allclose(base[order], perm, atol=1e-10)
    np.testing.assert_array_equal(m.participant_folds(ids)[order], m.participant_folds(ids[order]))


def test_ties_choose_local_then_larger_alpha():
    x, y, ids = linear_fixture()
    model, _ = m.select_q(x, np.zeros_like(y), ids)
    assert model.representation == "local" and model.ridge.alpha == 1.0


def test_tolerance_ties_compare_to_global_minimum(monkeypatch):
    x, y, ids = linear_fixture()
    losses = {1.0: 2e-12, 0.1: 1.25e-12, 0.01: 0.5e-12}

    class FakeRidge:
        def __init__(self, alpha):
            self.alpha = alpha

        def predict(self, features):
            return np.full(features.shape[:-1], np.sqrt(losses.get(self.alpha, 1.0)))

        def record(self):
            return {"alpha": self.alpha}

    monkeypatch.setattr(m, "fit_ridge", lambda features, target, alpha: FakeRidge(alpha))
    model, _ = m.select_q(x, np.zeros_like(y), ids)
    assert model.representation == "local" and model.ridge.alpha == 0.1


def test_multiple_budgets_bands_and_channels_remain_with_their_participant():
    rng = np.random.default_rng(8173)
    q = rng.normal(size=(8, 2, 2, 4, 3))
    y = q[..., 0] - q[..., 0].mean(axis=-1, keepdims=True)
    ids = np.arange(20, 28)
    _, oof, before = m.crossfit_q(q, y, ids)
    assert oof.shape == (8, 2, 2, 4)
    modified = y.copy()
    modified[0] += rng.normal(size=modified[0].shape) * 100
    _, other, after = m.crossfit_q(q, modified, ids)
    np.testing.assert_array_equal(oof[0], other[0])
    assert before["oof_folds"][0] == after["oof_folds"][0]


def test_proxy_level_shape_identity_and_common_shift():
    pred = np.array([[2.0, 4, 8], [1, 1, 1]])
    target = np.array([[1.0, 3, 9], [0, 0, 0]])
    parts = m.proxy_error_components(pred, target)
    np.testing.assert_allclose(
        parts["raw_mse"], parts["level_mse"] + parts["shape_mse"], atol=1e-12
    )
    shifted = m.proxy_error_components(pred + 3, target)
    np.testing.assert_allclose(parts["shape_mse"], shifted["shape_mse"], atol=1e-12)
    np.testing.assert_allclose(
        m.prior_log_shape(np.exp(pred)), m.prior_log_shape(np.exp(pred + 3)), atol=1e-12
    )
    # Clipping can make a formerly common offset alter prior shape.
    assert not np.allclose(
        m.prior_log_shape(np.exp(np.clip(pred, -3, 3))),
        m.prior_log_shape(np.exp(np.clip(pred + 3, -3, 3))),
    )


def test_score_stability_certificate_is_one_way_and_handles_ties():
    base = np.array([[1, 0], [0.51, 0.49], [1, 0], [1, 1]], dtype=float)
    alt = np.array([[0.9, 0.1], [0.49, 0.51], [11, 10], [1, 1]], dtype=float)
    result = m.score_sensitivity(base, alt)
    assert result["certified_unchanged"].tolist() == [True, False, False, False]
    assert result["prediction_changed"].tolist() == [False, True, False, False]
    assert not np.any(result["certified_unchanged"] & result["prediction_changed"])


@pytest.mark.parametrize("ids", [[1, 1, 2], [1.0, 2, 3], [-1, 0, 1], [1, 2], [[1, 2, 3]]])
def test_invalid_ids_rejected(ids):
    with pytest.raises(ValueError):
        m.participant_folds(ids)


@pytest.mark.parametrize("value", [np.nan, np.inf, 1j, "1", True])
def test_invalid_numeric_inputs_rejected(value):
    with pytest.raises(ValueError):
        m.fit_ridge([[value], [value]], [0, 1], 0.1)


def test_invalid_geometry_and_choices_rejected():
    x, y, ids = linear_fixture()
    bad_calls = [
        lambda: m.crossfit_q(x[:5], y[:5], ids[:5]),
        lambda: m.select_q(x, y, ids[:8]),
        lambda: m.select_q(x, y[..., :1], ids),
        lambda: m.q_design(x, "unknown"),
        lambda: m.q_design(x[..., :1, :], "local"),
        lambda: m.fit_ridge([[1]], [1], -1),
        lambda: m.fit_ridge([[1]], [1], [1]),
        lambda: m.prior_log_shape([0, 1]),
        lambda: m.proxy_error_components([1], [1, 2]),
        lambda: m.score_sensitivity([[1]], [[1]]),
    ]
    for call in bad_calls:
        with pytest.raises(ValueError):
            call()


def test_real_operator_engineering_harness_has_no_efficacy_labels():
    path = Path(__file__).resolve().parents[1] / "scripts/check_metadata_prior_validation.py"
    spec = importlib.util.spec_from_file_location("prior_engineering_check", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    result = module.run_checks()
    assert result["status"] == "ENGINEERING_CHECKS_PASSED"
    assert result["actuation"]["baseline_predictions"] == [1, 0]
    assert result["actuation"]["alternative_predictions"] == [0, 1]
    assert result["actuation"]["query_labels"] is None
    assert result["actuation"]["accuracy"] is None
    assert result["human_data_access"] is False
    assert result["metadata_fit"] is False
    np.testing.assert_allclose(result["actuation"]["S"], np.diag([5.82, 5.805]), atol=1e-12)
    np.testing.assert_allclose(result["actuation"]["baseline_scores"], [[0, 1], [1, 0]], atol=1e-12)
    np.testing.assert_allclose(
        result["actuation"]["alternative_scores"], [[1, 0], [0, 1]], atol=1e-12
    )
    json.dumps(result, allow_nan=False)


def test_fresh_cli_and_exclusive_receipt(tmp_path):
    root = Path(__file__).resolve().parents[1]
    output = tmp_path / "engineering.json"
    command = [
        sys.executable,
        str(root / "scripts/check_metadata_prior_validation.py"),
        "--output",
        str(output),
    ]
    completed = subprocess.run(
        command, cwd=root, text=True, capture_output=True, timeout=60, check=False
    )
    assert completed.returncode == 0, completed.stderr
    original = output.read_bytes()
    report = json.loads(original)
    assert report["status"] == "ENGINEERING_CHECKS_PASSED"
    assert len(report["provenance"]["source_sha256"]) == 4
    repeat = subprocess.run(
        command, cwd=root, text=True, capture_output=True, timeout=60, check=False
    )
    assert repeat.returncode != 0 and "FileExistsError" in repeat.stderr
    assert output.read_bytes() == original
