"""Generated-only independent mathematics and perturbed-record checks.

The local nested fixture shares audit orchestration but substitutes the old
augmented-lstsq solver as its coefficient reference; it is not claimed as a
new-producer integration test. Root owns that separate end-to-end test. No human
data or old model artifacts, no files containing EEG, and no optimizer are used.
"""

import ast
from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest
from scipy.spatial.distance import cdist
from scipy.stats import pearsonr

from cfeg.analysis import source_channel_margin_audit as audit
from cfeg.analysis.metadata_prior_validation import fit_ridge
from cfeg.analysis.task_trca_shape_features import support_q_mask


def generated_data(seed=73, count=9, samples=24):
    rng = np.random.default_rng(seed)
    prototype = rng.normal(size=(count, 2, 12, 5, 8, samples))
    support = prototype[:, :, None] + 0.4 * rng.normal(size=(count, 2, 3, 12, 5, 8, samples))
    source = prototype + rng.normal(size=prototype.shape)
    packet = rng.uniform(0, 30, (count, 2, 3, 8))
    packet[:, :, 0, 0] = 0.0
    ids = np.arange(count, dtype=np.int64)[
        ::-1
    ]  # Nonnegative ID0 and input-row order are intentional.
    orders = np.zeros(count, dtype=np.int64)
    q = np.empty((count, 2, 5, 8, 15))
    target = np.empty((count, 2, 5, 8))
    for person in range(count):
        for interface in range(2):
            q[person, interface] = support_q_mask(
                support[person, interface],
                np.isfinite(packet[person, interface]),
                interface,
                0,
                audit.FREQUENCIES,
            )
            template = support[person, interface].mean(axis=0)
            for band in range(5):
                for channel in range(8):
                    correlation = 1 - cdist(
                        source[person, interface, :, band, channel],
                        template[:, band, channel],
                        metric="correlation",
                    )
                    wrong = np.where(np.eye(12, dtype=bool), -np.inf, correlation)
                    target[person, interface, band, channel] = np.mean(
                        correlation.diagonal() - wrong.max(axis=1)
                    )
    target -= target.mean(axis=-1, keepdims=True)
    return {
        "ids": ids,
        "orders": orders,
        "support": support,
        "source_block": source,
        "packet": packet,
        "q": q,
        "target": target,
    }


@pytest.fixture(scope="module")
def completed():
    data = generated_data()
    q, y, m, available = audit._raw_bridge(data, audit._Checks())
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(
            audit,
            "_ridge",
            lambda x, target, alpha: fit_ridge(
                x.reshape(-1, x.shape[-1]), target.ravel(), alpha
            ).record(),
        )
        models, predictions = audit._nested(data, q, y, m, available)
    result = audit._result(data, models, predictions, y, available)
    return data, models, predictions, result, (q, y, m, available)


def test_all120_nested_fit_replay_and_negative_terminal_can_pass(completed):
    data, models, predictions, result, _ = completed
    receipt = audit.audit_probe(data, models, predictions, result)
    assert receipt["status"] == "PASS", receipt["errors"]
    assert receipt["ridge_fits_replayed"] == 120
    assert result["terminal"] == "UNINFORMATIVE_CONTROL"  # p9 cannot satisfy literal32.
    assert receipt["recomputed_result"]["terminal"] == result["terminal"]
    assert len(receipt["checks"]) == 4
    assert max(receipt["max_differences"].values()) < 1e-8


def test_target_cdist_and_scalar_pearson_reference(completed):
    data = completed[0]
    support, source = data["support"][0, 0], data["source_block"][0, 0]
    actual = audit._target(support, source)
    np.testing.assert_allclose(actual, data["target"][0, 0], rtol=0, atol=1e-14)
    template = support.mean(axis=0)
    margins = []
    for channel in range(8):
        per_class = []
        for label in range(12):
            values = [
                pearsonr(source[label, 0, channel], template[j, 0, channel]).statistic
                for j in range(12)
            ]
            per_class.append(values[label] - max(v for j, v in enumerate(values) if j != label))
        margins.append(np.mean(per_class))
    np.testing.assert_allclose(
        actual[0], np.asarray(margins) - np.mean(margins), atol=1e-14, rtol=0
    )
    np.testing.assert_allclose(
        audit._target(3 * support + 9, 2 * source - 7), actual, atol=1e-14, rtol=0
    )


def test_metadata_zero_nan_population_sd_and_missing_centering():
    packet = np.full((3, 8), np.nan)
    packet[:, 0] = [0, 3, np.nan]
    packet[:, 1] = [8, 8, 8]
    m, observed = audit._metadata(packet)
    first = np.log(4) / 2
    second = np.log(9)
    expected = np.zeros((8, 2))
    expected[0] = [(first - second) / 2, first]
    expected[1, 0] = (second - first) / 2
    np.testing.assert_allclose(m, expected, atol=1e-15, rtol=0)
    np.testing.assert_array_equal(observed, [True, True, False, False, False, False, False, False])
    m, observed = audit._metadata(np.full((3, 8), np.nan))
    assert not m.any() and not observed.any()
    for bad in (-1, np.inf):
        broken = packet.copy()
        broken[0, 0] = bad
        with pytest.raises(ValueError):
            audit._metadata(broken)


@pytest.mark.parametrize("alpha", audit.ALPHAS)
def test_independent_normal_equations_match_augmented_lstsq_with_rank_deficiency(alpha):
    rng = np.random.default_rng(79)
    x = rng.normal(size=(4, 2, 5, 8, 17))
    x[..., 3] = 1
    x[..., 5] = x[..., 4]
    x -= x.mean(axis=-2, keepdims=True)
    y = rng.normal(size=x.shape[:-1])
    y -= y.mean(axis=-1, keepdims=True)
    actual = audit._ridge(x, y, alpha)
    expected = fit_ridge(x.reshape(-1, 17), y.ravel(), alpha).record()
    for key in ("mean", "scale", "coefficient", "intercept"):
        np.testing.assert_allclose(actual[key], expected[key], rtol=0, atol=1e-10)
    assert actual["design_rank"] == expected["design_rank"]


@pytest.mark.parametrize("field", ["support", "source_block", "q", "target"])
def test_raw_feature_and_target_bridge_mutation_rejected(completed, field):
    data, models, predictions, result, _ = completed
    changed = dict(data)
    changed[field] = data[field].copy()
    changed[field].flat[0] += 0.01
    receipt = audit.audit_probe(changed, models, predictions, result)
    assert receipt["status"] == "FAIL"
    assert receipt["ridge_fits_replayed"] == 0
    assert receipt["terminal_override"] == "AUDIT_FAILURE"


@pytest.mark.parametrize(
    "field",
    [
        "coefficient",
        "q_mean",
        "m_scale",
        "fit_ids",
        "alpha",
        "inactive",
        "rank",
        "donor",
        "inner_loss",
        "selection",
        "outer_ids",
        "count",
    ],
)
def test_model_and_nested_receipt_mutation_rejected(completed, field):
    data, original, predictions, result, _ = completed
    models = deepcopy(original)
    arm = models["outer"][0]["arms"]["SHAM"]
    model = arm["inner"][0]["model"]
    if field == "coefficient":
        model["ridge"]["coefficient"][0] += 0.001
    elif field == "q_mean":
        model["q_scaler"]["mean"][0] += 0.001
    elif field == "m_scale":
        model["m_scaler"]["scale"][0] += 0.001
    elif field == "fit_ids":
        model["fit_ids"][0] = 10000
    elif field == "alpha":
        model["alpha"] = 0.1
    elif field == "inactive":
        model["inactive_columns"] = []
    elif field == "rank":
        model["ridge"]["design_rank"] += 1
    elif field == "donor":
        model["fit_coverage"]["donor_ids"][0] = 10000
    elif field == "inner_loss":
        arm["alpha_scores"][0]["participant_mse"][0] += 0.001
    elif field == "selection":
        arm["selected_alpha"] = 0.1
    elif field == "outer_ids":
        models["outer"][0]["evaluation_ids"][0] = 10000
    else:
        models["ridge_fit_count"] = 121
    receipt = audit.audit_probe(data, models, predictions, result)
    assert receipt["status"] == "FAIL", field
    assert receipt["errors"]


@pytest.mark.parametrize(
    "field",
    [
        "predictions",
        "terminal",
        "mean",
        "coverage",
        "relative_gain",
        "reason",
        "metadata_count",
        "extra_field",
    ],
)
def test_prediction_result_and_terminal_mutation_rejected(completed, field):
    data, models, original_predictions, original_result, _ = completed
    predictions = original_predictions.copy()
    result = deepcopy(original_result)
    if field == "predictions":
        predictions[0, 0, 0, 0, 0] += 0.001
    elif field == "terminal":
        result["terminal"] = "PROMISING_PROBE"
    elif field == "mean":
        result["mean_mse"][0] += 0.001
    elif field == "coverage":
        result["sham_outer_coverage"][0]["design_changed_fraction"] = 0.123
    elif field == "relative_gain":
        result["comparisons"]["Q"]["relative_mse_gain"] += 0.01
    elif field == "reason":
        result["failure_reasons"] = []
    elif field == "metadata_count":
        result["metadata"]["covered_participant_count"] += 1
    else:
        result["posthoc_threshold"] = 0.01
    assert audit.audit_probe(data, models, predictions, result)["status"] == "FAIL", field


def test_paired_donor_keeps_both_interfaces_mask_and_order_together():
    ids = np.array([90, 20, 70, 10, 80, 30, 60, 40, 50])
    orders = np.array([0, 0, 0, 0, 1, 1, 1, 1, 0])
    packet = np.ones((9, 2, 3, 8))
    packet[8, 1, 0, 0] = np.nan  # Paired-mask singleton.
    indices = np.arange(9)
    donor = audit._donors(ids, orders, packet, indices)
    actual = dict(zip(ids, ids[donor], strict=True))
    assert actual == {10: 20, 20: 70, 70: 90, 90: 10, 30: 40, 40: 60, 60: 80, 80: 30, 50: 50}
    subset = np.array([0, 1, 2])
    assert set(audit._donors(ids, orders, packet, subset)) <= set(subset)


def test_raw_metadata_change_does_not_imply_consumed_design_change(completed):
    data = dict(completed[0])
    packet = np.empty_like(data["packet"])
    for p in range(len(packet)):
        packet[p] = np.expm1(0.2 * p + np.arange(1, 9)[None, None])
    data["packet"] = packet
    q, target, m, available = audit._raw_bridge(data, audit._Checks())
    indices = np.argsort(data["ids"])
    model = audit._fit(data, q, target, m, available, indices, "SHAM", 1)
    coverage = model["fit_coverage"]
    assert coverage["raw_changed_fraction"] == 1
    assert coverage["design_changed_fraction"] == 0


def test_fit_scaling_ignores_held_rows_and_q_is_numeric_m_blind(completed):
    data, _, _, _, (q, y, m, available) = completed
    indices = np.argsort(data["ids"])[3:]
    before = audit._fit(data, q, y, m, available, indices, "QM", 0.01)
    q2, m2, y2 = q.copy(), m.copy(), y.copy()
    held = np.setdiff1d(np.arange(len(q)), indices)
    q2[held] += 100
    m2[held] += 100
    y2[held] += 100
    after = audit._fit(data, q2, y2, m2, available, indices, "QM", 0.01)
    assert before == after
    actual_q = audit._fit(data, q, y, m, available, indices, "Q", 0.01)
    changed_q = audit._fit(data, q, y, m + 100, available, indices, "Q", 0.01)
    assert actual_q == changed_q


def test_alpha_selection_is_pooled_participant_oof_not_mean_of_folds():
    data = generated_data(seed=79, count=12)
    q, y, m, available = audit._raw_bridge(data, audit._Checks())
    models, _ = audit._nested(data, q, y, m, available)
    unequal_weighting_is_distinguishable = False
    for outer in models["outer"]:
        for arm in outer["arms"].values():
            for score in arm["alpha_scores"]:
                assert score["mean_participant_mse"] == pytest.approx(
                    np.mean(score["participant_mse"])
                )
                inner = [row for row in arm["inner"] if row["alpha"] == score["alpha"]]
                assert [len(row["validation_ids"]) for row in inner] == [3, 3, 2]
                equal_fold = np.mean([np.mean(row["validation_participant_mse"]) for row in inner])
                unequal_weighting_is_distinguishable |= (
                    abs(equal_fold - score["mean_participant_mse"]) > 1e-8
                )
            minimum = min(row["mean_participant_mse"] for row in arm["alpha_scores"])
            expected = next(
                row["alpha"]
                for row in arm["alpha_scores"]
                if row["mean_participant_mse"] <= minimum + 1e-12
            )
            assert arm["selected_alpha"] == expected
    assert unequal_weighting_is_distinguishable


def test_sham_gate_needs_both_consumed_interfaces_changed(completed):
    data = dict(completed[0])
    data["packet"] = data["packet"].copy()
    data["packet"][:, 1] = np.arange(24, dtype=float).reshape(3, 8)
    q, y, m, available = audit._raw_bridge(data, audit._Checks())
    model = audit._fit(data, q, y, m, available, np.argsort(data["ids"]), "SHAM", 0.01)
    coverage = model["fit_coverage"]
    assert coverage["raw_changed_fraction"] == 1
    assert all(pair == [True, False] for pair in coverage["design_changed_by_interface"])
    assert coverage["design_changed_fraction"] == 0


def test_decision_gates_literal_coverage_tie_and_failure_precedence():
    count = 39
    ids = np.arange(count)
    pattern = np.tile(np.array([-1.0, 1.0]), 4)
    target = np.broadcast_to(pattern, (count, 2, 5, 8)).copy()
    pred = np.zeros((4, count, 2, 5, 8))
    pred[2] = 0.1 * target
    data = {"ids": ids}
    available = np.ones((count, 2, 8), bool)
    coverage = {"design_changed_fraction": 1.0}
    models = {"outer": [{"arms": {"SHAM": {"evaluation_coverage": coverage}}} for _ in range(3)]}
    result = audit._result(data, models, pred, target, available)
    assert result["terminal"] == "PROMISING_PROBE"
    assert result["comparisons"]["Q"]["relative_mse_gain"] == pytest.approx(0.19)
    tied = np.zeros_like(pred)
    tied[2] = 1e-14 * target
    result = audit._result(data, models, tied, target, available)
    assert not result["gates"]["positive_outer_folds"]
    assert not result["gates"]["positive_q_interfaces"]
    missing = available.copy()
    missing[:8] = False
    assert audit._result(data, models, pred, target, missing)["terminal"] == "UNINFORMATIVE_CONTROL"
    assert (
        audit._result(data, models, pred * 0, target * 0, missing)["terminal"]
        == "INSUFFICIENT_TARGET_VARIATION"
    )
    perfect_q = pred.copy()
    perfect_q[0] = target
    result = audit._result(data, models, perfect_q, target, available)
    assert result["terminal"] == "INSUFFICIENT_COMPARATOR_HEADROOM"
    assert result["comparisons"]["Q"]["relative_mse_gain"] is None


@pytest.mark.parametrize(
    "field",
    [
        "missing_raw",
        "extra_query",
        "duplicate_ids",
        "float_ids",
        "wrong_order",
        "negative_m",
        "infinite_m",
        "nan_source",
    ],
)
def test_input_boundary_no_proxy_only_audit(completed, field):
    data = dict(completed[0])
    if field == "missing_raw":
        del data["support"]
    elif field == "extra_query":
        data["query"] = object()
    elif field == "duplicate_ids":
        data["ids"] = np.zeros(9, dtype=int)
    elif field == "float_ids":
        data["ids"] = data["ids"].astype(float)
    elif field == "wrong_order":
        data["orders"] = np.full(9, 2)
    elif field in ("negative_m", "infinite_m"):
        data["packet"] = data["packet"].copy()
        data["packet"].flat[0] = -1 if field == "negative_m" else np.inf
    else:
        data["source_block"] = data["source_block"].copy()
        data["source_block"].flat[0] = np.nan
    with pytest.raises(ValueError):
        audit._validate(data)


def test_zero_norm_target_rejects_instead_of_clipping_to_zero(completed):
    support = completed[0]["support"][0, 0].copy()
    source = completed[0]["source_block"][0, 0].copy()
    source[0, 0, 0] = 1
    with pytest.raises(ValueError, match="undefined"):
        audit._target(support, source)


def test_auditor_imports_no_producer_target_fitter_or_decision():
    tree = ast.parse(Path(audit.__file__).read_text())
    modules = [node.module or "" for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)]
    assert not any(
        name.endswith(
            (
                "source_channel_margin",
                "source_channel_margin_probe",
                "metadata_prior_validation",
                "metadata_prior_source",
            )
        )
        for name in modules
    )
