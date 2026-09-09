"""Generated unit fixtures only; no source files, study seeds, or efficacy run."""

import copy
import json

import numpy as np
import pytest

from cfeg.analysis import source_channel_margin_probe as probe
from cfeg.analysis.source_channel_margin import source_channel_margin
from cfeg.analysis.task_trca_shape_features import support_q_mask


def fixture(p=9, seed=431):
    """Arbitrary feature arrays are unit inputs, not an EEG efficacy generator."""
    rng = np.random.default_rng(seed)
    q = rng.normal(size=(p, 2, 5, 8, 15))
    q[..., 3] = np.log(3)
    q[..., 4] = 1.0
    q[..., 5] = 0.0
    q[..., 6] = np.arange(2)[None, :, None, None]
    q[..., 7:] = np.eye(8)[None, None, None]
    target = rng.normal(size=(p, 2, 5, 8))
    target -= target.mean(axis=-1, keepdims=True)
    return {
        "ids": np.arange(p, dtype=np.int64), "orders": np.zeros(p, dtype=np.int64),
        "packet": rng.uniform(0, 50, size=(p, 2, 3, 8)), "q": q, "target": target,
    }


@pytest.fixture(scope="module")
def fitted():
    data = fixture()
    events = []
    models = probe.fit_probe(data, events.append)
    predictions, result = probe.evaluate_probe(data, models)
    return data, models, predictions, result, events


def test_full120_fit_program_and_json_contract(fitted):
    data, models, predictions, result, events = fitted
    assert models["ridge_fit_count"] == 120
    assert predictions.shape == (4, 9, 2, 5, 8)
    assert np.max(np.abs(predictions.mean(axis=-1))) < 1e-12
    assert [event["fit_index"] for event in events[:-1]] == list(range(1, 121))
    assert events[-1] == {"event": "fit_probe_completed", "ridge_fit_count": 120}
    assert sum(event.get("stage") == "final" for event in events) == 12
    assert result["terminal"] == "UNINFORMATIVE_CONTROL"  # fixed32; never rescale for p9
    assert result["metadata"]["minimum_covered_participants"] == 32
    assert result["ids"] == list(range(9))
    json.dumps(models, allow_nan=False)
    json.dumps(result, allow_nan=False)
    assert "participant_mse" not in models
    for outer in models["outer"]:
        assert set(outer["fit_ids"]).isdisjoint(outer["evaluation_ids"])
        for arm, selection in outer["arms"].items():
            assert "evaluation_participant_mse" not in selection
            assert len(selection["inner"]) == 9
            for item in selection["inner"]:
                model = item["model"]
                assert set(model["fit_ids"]).isdisjoint(item["validation_ids"])
                assert set(model["fit_ids"]).isdisjoint(outer["evaluation_ids"])
                assert model["q_scaler"]["fit_ids"] == model["fit_ids"]
                assert model["ridge"]["alpha"] == item["alpha"]
                assert model["nominal_coefficients"] == (16 if arm == "Q" else 18)
                assert {3, 4, 5, 6}.issubset(model["inactive_columns"])
                if arm in ("QM", "SHAM"):
                    assert model["m_scaler"]["fit_ids"] == model["fit_ids"]
                else:
                    assert model["m_scaler"] is None
                    assert item["validation_coverage"] is None
            if arm == "SHAM":
                for coverage in [selection["final"]["fit_coverage"], selection["evaluation_coverage"]]:
                    assert set(coverage["donor_ids"]).issubset(coverage["ids"])
    np.testing.assert_array_equal(data["ids"], np.arange(9))


def test_outer_targets_and_covariates_do_not_affect_own_fit_or_selection(fitted):
    data, models, *_ = fitted
    changed = copy.deepcopy(data)
    held = np.arange(9) % 3 == 0
    changed["target"][held] *= -11
    changed["q"][held, ..., :3] *= 23
    changed["packet"][held] *= 17
    refit = probe.fit_probe(changed)
    for arm in probe.ARMS:
        before, after = models["outer"][0]["arms"][arm], refit["outer"][0]["arms"][arm]
        assert before["inner"] == after["inner"]
        assert before["alpha_scores"] == after["alpha_scores"]
        assert before["selected_alpha"] == after["selected_alpha"]
        assert before["final"] == after["final"]


def test_fit_never_computes_outer_loss(monkeypatch):
    original = probe.participant_margin_mse
    sizes = []

    def tracked(prediction, target):
        sizes.append(len(target))
        return original(prediction, target)

    monkeypatch.setattr(probe, "participant_margin_mse", tracked)
    probe.fit_probe(fixture())
    assert sizes == [2] * 108  # inner only; no p3 outer losses


def test_pooled_participant_selection_not_equal_fold_means():
    models = probe.fit_probe(fixture(15, seed=432))
    unequal = False
    for outer in models["outer"]:
        for selection in outer["arms"].values():
            for index, candidate in enumerate(selection["alpha_scores"]):
                inner = selection["inner"][index * 3:(index + 1) * 3]
                mapped = {pid: loss for row in inner for pid, loss in
                          zip(row["validation_ids"], row["validation_participant_mse"])}
                losses = [mapped[pid] for pid in candidate["participant_ids"]]
                np.testing.assert_array_equal(candidate["participant_mse"], losses)
                assert candidate["mean_participant_mse"] == np.mean(losses)
                equal_fold = np.mean([np.mean(row["validation_participant_mse"]) for row in inner])
                unequal |= abs(candidate["mean_participant_mse"] - equal_fold) > 1e-8
    assert unequal


def test_global_min_tie_rule_prefers_larger_alpha():
    scores = [{"alpha": alpha, "mean_participant_mse": loss} for alpha, loss in
              zip(probe.ALPHAS, (2e-12, 1e-12, 0.4e-12))]
    assert probe._select_alpha(scores) == 0.01
    for row in scores:
        row["mean_participant_mse"] = 1.0
    assert probe._select_alpha(scores) == 1.0


def test_q2_squares_initial_standardized_q_before_channel_centering():
    values = probe._data(fixture())
    rows = np.arange(6)
    model = probe._fit_model(values, rows, "Q2", 1.0)
    q = (values["q"][rows] - model["q_scaler"]["mean"]) / model["q_scaler"]["scale"]
    expected = np.concatenate((q, q[..., [1, 2]] ** 2), axis=-1)
    expected -= expected.mean(axis=-2, keepdims=True)
    np.testing.assert_array_equal(probe._design(values, rows, model), expected)
    centered_q = q - q.mean(axis=-2, keepdims=True)
    wrong = centered_q[..., [1, 2]] ** 2
    wrong -= wrong.mean(axis=-2, keepdims=True)
    assert np.max(np.abs(expected[..., -2:] - wrong)) > 0.1
    assert model["ridge"]["design_rank"] <= 17


@pytest.mark.parametrize("arm", probe.ARMS)
def test_joint_ridge_matches_mean_loss_normal_equations_and_rank(arm):
    values = probe._data(fixture())
    rows = np.arange(6)
    model = probe._fit_model(values, rows, arm, 0.01)
    design = probe._design(values, rows, model)
    x = design.reshape(-1, design.shape[-1])
    mean, scale = x.mean(axis=0), x.std(axis=0)
    scale = np.where(scale < 1e-12, 1.0, scale)
    z = (x - mean) / scale
    y = values["target"][rows].reshape(-1)
    coefficient = np.linalg.solve(z.T @ z / len(z) + 0.01 * np.eye(z.shape[1]),
                                  z.T @ (y - y.mean()) / len(z))
    np.testing.assert_allclose(model["ridge"]["coefficient"], coefficient, atol=1e-12, rtol=0)
    assert model["ridge"]["intercept"] == y.mean()
    assert model["ridge"]["design_rank"] == np.linalg.matrix_rank(z / np.sqrt(len(z)))
    assert model["inactive_columns"] == np.flatnonzero(x.std(axis=0) < 1e-12).tolist()


def test_initial_scalers_population_stats_and_missing_neutral():
    data = fixture()
    data["packet"][:, :, :, 0] = np.nan
    data["packet"][0, 0, 0, 1] = 0.0  # observed zero
    values = probe._data(data)
    fit = np.arange(6)
    model = probe._fit_model(values, fit, "QM", 0.01)
    q_rows = data["q"][fit].reshape(-1, 15)
    np.testing.assert_array_equal(model["q_scaler"]["mean"], q_rows.mean(axis=0))
    expected_sd = q_rows.std(axis=0)
    np.testing.assert_array_equal(model["q_scaler"]["scale"], np.where(expected_sd < 1e-12, 1, expected_sd))
    m_rows = values["m"][fit][values["available"][fit]]
    np.testing.assert_array_equal(model["m_scaler"]["mean"], m_rows.mean(axis=0))
    standardized = probe._standardize(values["m"][fit], model["m_scaler"], values["available"][fit])
    assert np.count_nonzero(standardized[:, :, 0]) == 0
    assert values["available"][0, 0, 1]


def test_q_and_q2_ignore_numeric_metadata_with_fixed_mask(fitted):
    data, models, predictions, *_ = fitted
    changed = copy.deepcopy(data)
    changed["packet"] = data["packet"] ** 2 + 1.0
    refit = probe.fit_probe(changed)
    predicted, _ = probe.evaluate_probe(changed, refit)
    np.testing.assert_array_equal(predictions[:2], predicted[:2])
    for before, after in zip(models["outer"], refit["outer"]):
        for arm in ("Q", "Q2"):
            assert before["arms"][arm] == after["arms"][arm]


def test_all_missing_m_has_zero_scaler_and_no_new_design():
    data = fixture()
    data["packet"].fill(np.nan)
    models = probe.fit_probe(data)
    prediction, result = probe.evaluate_probe(data, models)
    np.testing.assert_allclose(prediction[0], prediction[2], atol=1e-12, rtol=0)
    np.testing.assert_allclose(prediction[0], prediction[3], atol=1e-12, rtol=0)
    assert result["metadata"]["covered_participant_count"] == 0
    assert "INSUFFICIENT_METADATA_COVERAGE" in result["failure_reasons"]
    for outer in models["outer"]:
        model = outer["arms"]["QM"]["final"]
        assert model["m_scaler"]["mean"] == [0.0, 0.0]
        assert model["m_scaler"]["scale"] == [1.0, 1.0]
        assert {15, 16}.issubset(model["inactive_columns"])
        assert outer["arms"]["SHAM"]["evaluation_coverage"]["design_changed_fraction"] == 0.0


def test_donor_pairs_full_mask_order_partition_and_sorted_next():
    data = fixture()
    data["orders"][4:6] = 1
    data["packet"][6, 0, 0, 0] = np.nan
    data["packet"][7, 0, 1, 0] = np.nan  # same per-channel availability, different full mask
    data["packet"][8, 1, 0, 0] = np.nan  # other interface distinguishes paired stratum
    values = probe._data(data)
    rows = np.arange(9)
    np.testing.assert_array_equal(probe._donor_rows(values, rows), [1, 2, 3, 0, 5, 4, 6, 7, 8])
    np.testing.assert_array_equal(probe._donor_rows(values, np.array([3, 1])), [1, 3])
    data["packet"][np.isfinite(data["packet"])] *= 101
    data["target"] *= -30
    np.testing.assert_array_equal(probe._donor_rows(probe._data(data), rows), [1, 2, 3, 0, 5, 4, 6, 7, 8])


def test_raw_swaps_can_be_inactive_consumed_design():
    data = fixture()
    base = 1.0 + np.arange(8) / 4
    for person in range(9):
        data["packet"][person] = np.expm1(base[None, None, :] + person * 0.1)
    values = probe._data(data)
    rows = np.arange(6)
    model = probe._fit_model(values, rows, "SHAM", 0.01)
    coverage = probe._coverage(values, np.arange(6, 9), model)
    assert coverage["raw_changed_fraction"] == 1.0
    assert coverage["design_changed_fraction"] == 0.0
    assert not np.asarray(coverage["design_changed_by_interface"]).any()


def test_consumed_coverage_requires_both_interfaces():
    data = fixture()
    data["packet"][:, 1] = np.arange(24).reshape(3, 8)
    values = probe._data(data)
    model = probe._fit_model(values, np.arange(6), "SHAM", 0.01)
    coverage = probe._coverage(values, np.arange(6, 9), model)
    np.testing.assert_array_equal(coverage["design_changed_by_interface"], [[True, False]] * 3)
    assert coverage["raw_changed_fraction"] == 1.0
    assert coverage["design_changed_fraction"] == 0.0


def test_participant_input_permutation_keeps_sorted_fold_predictions(fitted):
    data, models, prediction, *_ = fitted
    order = np.array([8, 1, 4, 0, 7, 2, 6, 5, 3])
    permuted = {name: value[order] for name, value in data.items()}
    refit = probe.fit_probe(permuted)
    other, _ = probe.evaluate_probe(permuted, refit)
    assert refit["outer"] == models["outer"]
    np.testing.assert_array_equal(other, prediction[:, order])


def test_channel_axis_equivariance(fitted):
    data, _, prediction, *_ = fitted
    order = np.array([7, 0, 3, 1, 6, 5, 4, 2])
    changed = copy.deepcopy(data)
    changed["q"] = data["q"][..., order, :]
    changed["target"] = data["target"][..., order]
    changed["packet"] = data["packet"][..., order]
    other, _ = probe.evaluate_probe(changed, probe.fit_probe(changed))
    np.testing.assert_allclose(other, prediction[..., order], atol=2e-11, rtol=0)


def test_generated_actual_q_and_target_constructor_smoke_and_immutable():
    data = fixture()
    rng = np.random.default_rng(433)
    data["support"] = rng.normal(size=(9, 2, 3, 12, 5, 8, 24))
    data["source_block"] = rng.normal(size=(9, 2, 12, 5, 8, 24))
    frequency = np.array([9.25, 11.25, 13.25, 9.75, 11.75, 13.75, 10.25, 12.25, 14.25, 10.75, 12.75, 14.75])
    for p in range(9):
        for interface in range(2):
            data["q"][p, interface] = support_q_mask(
                data["support"][p, interface], np.isfinite(data["packet"][p, interface]),
                interface, 0, frequency,
            )
            data["target"][p, interface] = source_channel_margin(
                data["support"][p, interface], data["source_block"][p, interface],
                np.arange(12), np.arange(12),
            ).channel_shape
    copies = {name: value.copy() for name, value in data.items()}
    for value in data.values():
        value.setflags(write=False)
    model = probe.fit_probe(data)
    prediction, result = probe.evaluate_probe(data, model)
    assert prediction.shape == (4, 9, 2, 5, 8)
    assert result["ridge_fit_count"] == 120
    for name in data:
        np.testing.assert_array_equal(data[name], copies[name])


def summarize_case(p=39):
    values = probe._data(fixture(p, seed=434))
    # Direct decision unit input, not fitted/generated scientific efficacy.
    predictions = np.zeros((4, p, 2, 5, 8))
    predictions[2] = values["target"] * 0.5
    coverage = [{"design_changed_fraction": 1.0}] * 3
    return values, predictions, coverage


def test_promising_decision_requires_all_fixed_gates():
    values, predictions, coverage = summarize_case()
    result = probe._summarize(values, predictions, coverage)
    assert result["terminal"] == "PROMISING_PROBE"
    assert all(result["gates"].values())
    assert result["failure_reasons"] == []
    for comparison in result["comparisons"].values():
        assert comparison["relative_mse_gain"] == pytest.approx(0.75)
        assert comparison["positive_outer_folds"] == 3


@pytest.mark.parametrize("failure", ["target", "metadata", "sham", "headroom", "q2", "one_fold", "one_interface"])
def test_terminal_gates_and_precedence(failure):
    values, predictions, coverage = summarize_case()
    expected = "NO_PROMISING_SIGNAL_IN_THIS_PROBE"
    if failure == "target":
        values["target"].fill(0)
        values["available"].fill(False)
        expected = "INSUFFICIENT_TARGET_VARIATION"
    elif failure == "metadata":
        values["available"][:8].fill(False)  #31covered; literal32 fails
        expected = "UNINFORMATIVE_CONTROL"
    elif failure == "sham":
        coverage[0] = {"design_changed_fraction": 10 / 13}
        expected = "UNINFORMATIVE_CONTROL"
    elif failure == "headroom":
        predictions[0] = values["target"]
        expected = "INSUFFICIENT_COMPARATOR_HEADROOM"
    elif failure == "q2":
        predictions[1] = values["target"] * 0.8
    elif failure == "one_fold":
        mask = values["fold_assignment"] != 0
        predictions[2, mask] = -values["target"][mask] * 0.01
    elif failure == "one_interface":
        predictions[2, :, 1] = -values["target"][:, 1] * 0.01
    result = probe._summarize(values, predictions, coverage)
    assert result["terminal"] == expected
    if failure == "headroom":
        assert result["comparisons"]["Q"]["relative_mse_gain"] is None
    assert result["failure_reasons"]


def test_exact_coverage_and_positive_tolerance_boundaries():
    values, predictions, coverage = summarize_case()
    values["available"][:7].fill(False)
    coverage[0] = {"design_changed_fraction": 11 / 13}
    assert probe._summarize(values, predictions, coverage)["terminal"] == "PROMISING_PROBE"
    predictions[2] = values["target"] * 1e-14
    result = probe._summarize(values, predictions, coverage)
    assert result["comparisons"]["Q"]["positive_outer_folds"] == 0
    assert not result["gates"]["positive_q_interfaces"]


@pytest.mark.parametrize("mutation", [
    "duplicate_ids", "negative_ids", "float_ids", "bool_ids", "order2", "bool_orders",
    "p8", "p10", "packet_inf", "packet_negative", "q_nan", "target_nan", "not_centered",
    "q_float32", "packet_float32", "bad_q_shape", "extra", "missing", "support_only", "source_only",
])
def test_invalid_inputs_reject_before_fit(monkeypatch, mutation):
    data = fixture(10 if mutation == "p10" else (8 if mutation == "p8" else 9))
    if mutation == "duplicate_ids":
        data["ids"][0] = data["ids"][1]
    elif mutation == "negative_ids":
        data["ids"][0] = -1
    elif mutation == "float_ids":
        data["ids"] = data["ids"].astype(float)
    elif mutation == "bool_ids":
        data["ids"] = data["ids"].astype(bool)
    elif mutation == "order2":
        data["orders"][0] = 2
    elif mutation == "bool_orders":
        data["orders"] = data["orders"].astype(bool)
    elif mutation == "packet_inf":
        data["packet"][0, 0, 0, 0] = np.inf
    elif mutation == "packet_negative":
        data["packet"][0, 0, 0, 0] = -1
    elif mutation == "q_nan":
        data["q"][0, 0, 0, 0, 0] = np.nan
    elif mutation == "target_nan":
        data["target"][0, 0, 0, 0] = np.nan
    elif mutation == "not_centered":
        data["target"] += 0.1
    elif mutation == "q_float32":
        data["q"] = data["q"].astype(np.float32)
    elif mutation == "packet_float32":
        data["packet"] = data["packet"].astype(np.float32)
    elif mutation == "bad_q_shape":
        data["q"] = data["q"][..., :14]
    elif mutation == "extra":
        data["query"] = None
    elif mutation == "missing":
        del data["target"]
    elif mutation == "support_only":
        data["support"] = None
    elif mutation == "source_only":
        data["source_block"] = None

    def forbidden(*_):
        raise AssertionError("invalid input reached fitting")

    monkeypatch.setattr(probe, "fit_ridge", forbidden)
    with pytest.raises(ValueError):
        probe.fit_probe(data)


@pytest.mark.parametrize("mutation", ["short_samples", "wrong_source", "support_nan", "source_float32"])
def test_optional_raw_inputs_still_validate_geometry_and_finiteness(mutation):
    data = fixture()
    data["support"] = np.ones((9, 2, 3, 12, 5, 8, 24), dtype=np.float64)
    data["source_block"] = np.ones((9, 2, 12, 5, 8, 24), dtype=np.float64)
    if mutation == "short_samples":
        data["support"] = data["support"][..., :23]
        data["source_block"] = data["source_block"][..., :23]
    elif mutation == "wrong_source":
        data["source_block"] = data["source_block"][:, :1]
    elif mutation == "support_nan":
        data["support"][0, 0, 0, 0, 0, 0, 0] = np.nan
    else:
        data["source_block"] = data["source_block"].astype(np.float32)
    with pytest.raises(ValueError):
        probe.fit_probe(data)


@pytest.mark.parametrize("mutation", ["schema", "ids", "outer", "inner", "scaler_ids", "m_leak", "nan_coef", "scale0", "count"])
def test_frozen_models_reject_legacy_or_partition_mutation(fitted, mutation):
    data, original, *_ = fitted
    model = copy.deepcopy(original)
    final = model["outer"][0]["arms"]["Q"]["final"]
    if mutation == "schema":
        model["schema"] = "task-trca-n1-integration-v1"
    elif mutation == "ids":
        model["ids"].reverse()
    elif mutation == "outer":
        model["outer"][0]["evaluation_ids"][0] = 999
    elif mutation == "inner":
        model["outer"][0]["arms"]["Q"]["inner"].pop()
    elif mutation == "scaler_ids":
        final["q_scaler"]["fit_ids"].append(0)
    elif mutation == "m_leak":
        final["m_scaler"] = {"mean": [0, 0], "scale": [1, 1], "fit_ids": final["fit_ids"]}
    elif mutation == "nan_coef":
        final["ridge"]["coefficient"][0] = np.nan
    elif mutation == "scale0":
        final["ridge"]["scale"][0] = 0
    elif mutation == "count":
        model["ridge_fit_count"] = 119
    with pytest.raises(ValueError):
        probe.evaluate_probe(data, model)


def test_event_failure_propagates_and_stops_after_one_fit(monkeypatch):
    fitted = []
    original = probe.fit_ridge

    def tracked(*args):
        fitted.append(1)
        return original(*args)

    def broken_sink(event):
        assert event["fit_index"] == 1
        raise OSError("durable event sink failure")

    monkeypatch.setattr(probe, "fit_ridge", tracked)
    with pytest.raises(OSError, match="event sink"):
        probe.fit_probe(fixture(), broken_sink)
    assert len(fitted) == 1


def test_empty_or_foreign_model_and_sink_types(fitted):
    data, *_ = fitted
    with pytest.raises(ValueError):
        probe.fit_probe(data, event_sink=1)
    for model in ({}, None, {"schema": "legacy"}):
        with pytest.raises(ValueError):
            probe.evaluate_probe(data, model)
