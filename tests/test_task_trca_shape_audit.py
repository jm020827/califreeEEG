"""Artificial-only independent audit tests; no actual cohort inputs accessed."""

import ast
import hashlib
import inspect

import numpy as np
import pytest
from scipy import linalg, stats

from cfeg.analysis import task_trca_shape_audit as audit


def pipeline_record():
    record = {
        "fit_ids": [1001, 1002],
        "q_scaler": {
            "mean": np.zeros(15).tolist(),
            "scale": np.ones(15).tolist(),
            "fit_ids": [1001, 1002],
        },
        "m_scaler": {"mean": [0, 0], "scale": [1, 1], "fit_ids": [1001, 1002]},
        "q2_scaler": {"mean": [0, 0], "scale": [1, 1], "fit_ids": [1001, 1002]},
        "Q": {"coefficients": np.linspace(-0.4, 0.3, 16).tolist()},
        "residuals": {
            arm: {"coefficients": [0.2, -0.3, 0.1]} for arm in ("Q2", "QM", "SHAM_REFIT")
        },
    }
    digest = hashlib.sha256()
    for x in (record["q_scaler"]["mean"], record["q_scaler"]["scale"], record["Q"]["coefficients"]):
        digest.update(np.asarray(x, dtype="<f8").tobytes())
    record["q_hash"] = digest.hexdigest()
    return record


def synthetic_geometry():
    rng = np.random.default_rng(45691)
    x = rng.normal(size=(5, 12, 8, 8))
    c = x @ x.swapaxes(-1, -2) + np.eye(8)
    x = rng.normal(size=(5, 12, 8, 8))
    s = x @ x.swapaxes(-1, -2)
    anchors = np.empty((5, 12, 8))
    for band in range(5):
        for label in range(12):
            anchors[band, label] = linalg.eigh(s[band, label], c[band, label])[1][:, -1]
    r = np.exp(rng.uniform(-0.2, 0.2, (5, 8)))
    r *= 8 / r.sum(-1, keepdims=True)
    return s, c, anchors, r


def raw_statistics():
    rng = np.random.default_rng(82218)
    template = rng.normal(size=(12, 5, 8, 19)) + rng.normal(size=(12, 5, 8, 1))
    query = rng.normal(size=(7, 5, 8, 19)) + rng.normal(size=(7, 5, 8, 1))
    mx, mt = query.mean(-1), template.mean(-1)
    xc, tc = query - mx[..., None], template - mt[..., None]
    statistics = {
        "query_gram": xc @ xc.swapaxes(-1, -2),
        "template_gram": tc @ tc.swapaxes(-1, -2),
        "cross_gram": np.einsum("nbit,cbjt->ncbij", xc, tc),
        "query_mean": mx,
        "template_mean": mt,
        "samples": 19,
    }
    return template, query, statistics


class Poison:
    def __array__(self, *args, **kwargs):
        raise AssertionError("Forbidden numeric input touched")


def test_module_imports_are_independent():
    tree = ast.parse(inspect.getsource(audit))
    imports = [node.module for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)]
    imports += [
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    ]
    assert all(not name.startswith(("torch", "cfeg")) for name in imports)


@pytest.mark.parametrize("arm", ["Q", "MISSING", "Q2"])
def test_no_numeric_metadata_read_in_q_and_q2(arm):
    r = audit.independent_prior(
        np.zeros((5, 8, 15)), Poison(), np.ones(8, bool), pipeline_record(), arm
    )
    np.testing.assert_array_equal(r, np.ones((5, 8)))


def test_iso_reads_no_features_or_pipeline():
    np.testing.assert_array_equal(
        audit.independent_prior(Poison(), Poison(), Poison(), Poison(), "ISO"), np.ones((5, 8))
    )


@pytest.mark.parametrize("arm", audit.ARMS[1:])
def test_prior_agrees_with_manual_bounded_head_and_masks(arm):
    rng = np.random.default_rng(602)
    q, m = rng.normal(size=(5, 8, 15)), rng.normal(size=(8, 2))
    donor, stale = 0.5 * m, 1.5 * m
    mask = np.array([True, True, False, True, True, True, False, True])
    oldmask = mask & (np.arange(8) % 2 == 0)
    record = pipeline_record()
    got = audit.independent_prior(
        q, m, mask, record, arm, donor_m=donor, stale_m=stale, stale_available=oldmask
    )
    a = np.log(2) / 2
    coeff = np.array(record["Q"]["coefficients"])
    logits = 0.8 * a * np.tanh(q @ coeff[:-1] + coeff[-1])
    if arm == "ISO":
        logits[:] = 0
    elif arm not in ("Q", "MISSING"):
        used = (
            q[..., 1:3]
            if arm == "Q2"
            else donor
            if arm in ("SHAM_REFIT", "PERMUTED")
            else stale
            if arm == "STALE"
            else m
        )
        observed = oldmask if arm == "STALE" else mask
        logits += 0.2 * a * np.tanh(used @ np.array([0.2, -0.3]) + 0.1) * observed
    expected = 8 * np.exp(logits) / np.exp(logits).sum(-1, keepdims=True)
    np.testing.assert_allclose(got, expected, atol=4e-16, rtol=4e-16)
    assert np.max(got) <= 16 / 9
    assert np.max(got.max(-1) / got.min(-1)) <= 2


@pytest.mark.parametrize("mutation", ["hash", "scale", "ids", "coefficient", "stale_mask"])
def test_prior_rejects_bad_records(mutation):
    record = pipeline_record()
    q, m, available = np.zeros((5, 8, 15)), np.zeros((8, 2)), np.ones(8, bool)
    stale = available.copy()
    if mutation == "hash":
        record["q_hash"] = "a" * 64
    elif mutation == "scale":
        record["q_scaler"]["scale"][0] = 0
    elif mutation == "ids":
        record["m_scaler"]["fit_ids"] = [999]
    elif mutation == "coefficient":
        record["residuals"]["QM"]["coefficients"][0] = float("nan")
    else:
        available[0] = False
    with pytest.raises(ValueError):
        audit.independent_prior(q, m, available, record, "STALE", stale_m=m, stale_available=stale)


def test_independent_filter_equation_c_normalization_bound_and_sign():
    s, c, anchors, r = synthetic_geometry()
    w = audit.independent_filters(s, c, anchors, r)
    for band in range(5):
        for label in range(12):
            v, metric = w[band, :, label], c[band, label]
            penalty = 0.1 * np.linalg.eigvalsh(metric)[0] / (16 / 9) * np.diag(r[band])
            denominator = metric + penalty
            value = (v @ s[band, label] @ v) / (v @ denominator @ v)
            np.testing.assert_allclose(
                s[band, label] @ v, value * denominator @ v, rtol=5e-12, atol=3e-13
            )
            assert v @ metric @ v == pytest.approx(1.0, abs=3e-15)
            assert v @ metric @ anchors[band, label] > 0
            assert np.linalg.eigvalsh(0.1 * metric - penalty)[0] >= -1e-13


def test_independent_filter_matches_torch_operator_artificial_only():
    import torch

    from cfeg.analysis import task_trca_shape_operator as operator

    s, c, anchors, r = synthetic_geometry()
    expected = (
        operator.bounded_filters(
            *(torch.from_numpy(x) for x in (s, c, anchors)), torch.from_numpy(r[:, None, :])
        )
        .numpy()
        .transpose(0, 2, 1)
    )
    np.testing.assert_allclose(
        audit.independent_filters(s, c, anchors, r), expected, atol=2e-14, rtol=2e-12
    )


@pytest.mark.parametrize("mutation", ["bad_c", "tie", "anchor", "asymmetric", "bad_r"])
def test_filter_failures_are_not_rescued(mutation):
    s, c, anchors, r = synthetic_geometry()
    if mutation == "bad_c":
        c[0, 0] = -np.eye(8)
    elif mutation == "tie":
        s[:] = 0
    elif mutation == "anchor":
        anchors[:] = 0
    elif mutation == "asymmetric":
        s[0, 0, 1, 2] += 0.2
    else:
        r[0, 0] += 1
    with pytest.raises(ValueError):
        audit.independent_filters(s, c, anchors, r)


def test_repeated_lower_roots_allowed_but_orthogonal_anchor_rejected():
    c = np.broadcast_to(np.eye(8), (5, 12, 8, 8)).copy()
    s = c.copy()
    s[..., -1, -1] = 3
    anchors = np.zeros((5, 12, 8))
    anchors[..., -1] = 1
    result = audit.independent_filters(s, c, anchors, np.ones((5, 8)))
    np.testing.assert_array_equal(result[:, -1], np.ones((5, 12)))
    anchors[..., -1] = 0
    anchors[..., 0] = 1
    with pytest.raises(ValueError, match="orthogonal"):
        audit.independent_filters(s, c, anchors, np.ones((5, 8)))


def test_independent_scores_equal_literal_global_pearson_with_nonzero_channel_means():
    template, query, statistics = raw_statistics()
    w = np.random.default_rng(282).normal(size=(5, 8, 12))
    weights = np.array(
        [1.25, 0.6704482076268572, 0.5032785618838642, 0.42677669529663687, 0.3837480609952844]
    )
    expected = np.zeros((7, 12))
    for sample in range(7):
        for label in range(12):
            for band in range(5):
                x = (w[band].T @ query[sample, band]).ravel()
                t = (w[band].T @ template[label, band]).ravel()
                expected[sample, label] += weights[band] * np.corrcoef(x, t)[0, 1]
    np.testing.assert_allclose(
        audit.independent_scores(w, statistics, weights), expected, atol=9e-16, rtol=2e-14
    )


@pytest.mark.parametrize("field", ["samples", "weights", "query_gram", "filters"])
def test_score_invalidity_raises(field):
    _, _, statistics = raw_statistics()
    w, weights = np.ones((5, 8, 12)), np.ones(5)
    if field == "samples":
        statistics["samples"] = True
    elif field == "weights":
        weights[0] = -1
    elif field == "query_gram":
        statistics["query_gram"][:] = 0
        statistics["query_mean"][:] = 0
    else:
        w = w.astype(complex)
    with pytest.raises(ValueError):
        audit.independent_scores(w, statistics, weights)


def test_metadata_zero_valid_missing_absent_and_population_sd():
    packet = np.full((3, 8), np.nan)
    packet[:, 0] = [0, 1, 3]
    packet[:2, 1] = [3, 3]
    m, available = audit.independent_metadata(packet)
    np.testing.assert_array_equal(available, [True, True, False, False, False, False, False, False])
    means = np.array([np.log(2), np.log(4)])
    np.testing.assert_allclose(m[:2, 0], means - means.mean())
    assert m[0, 1] == pytest.approx(np.std([0, np.log(2), np.log(4)], ddof=0))
    np.testing.assert_array_equal(m[2:], np.zeros((6, 2)))
    with pytest.raises(ValueError):
        audit.independent_metadata(np.full((3, 8), -1))


def test_donors_use_exact_prefix_masks_and_order_in_local_partition():
    ids = [9, 2, 7, 4, 6]
    masks = np.ones((5, 3, 8), bool)
    masks[3, 0, 0] = False
    masks[4, 1, 0] = False  # same aggregate availability, different prefix mask
    donors = audit.independent_donors(ids, masks, [0, 0, 1, 0, 0], 0)
    assert donors == {2: 9, 9: 2, 7: 7, 4: 4, 6: 6}


def summary_inputs(*, q3=38, qm3=40, q5=40, full3=40, a0_correct=0):
    scores = np.zeros(audit.SCORE_SHAPE)
    a0 = np.zeros((39, 2, 4, 48, 12))
    truth = np.tile(np.arange(12), 4)
    for index, arm in enumerate(audit.ARMS):
        for budget in range(2):
            correct = (
                (qm3 if arm == "QM" else full3 if arm == "FULL" else q3) if budget == 0 else q5
            )
            prediction = np.where(np.arange(48) < correct, truth, (truth + 1) % 12)
            scores[..., budget, index, np.arange(48), prediction] = 1
    prediction = np.where(np.arange(48) < a0_correct, truth, (truth + 1) % 12)
    a0[..., np.arange(48), prediction] = 1
    coverage = {
        "m": np.zeros((39, 2, 8, 2)),
        "donor_m": np.ones((39, 2, 8, 2)),
        "available": np.ones((39, 2, 8), bool),
    }
    actuation = {
        "band_weights": np.ones((2, 5)),
        "a0_band_weights": np.ones((2, 5)),
        "qm_coefficients": np.ones((3, 3)),
        "r_max_abs_qm_minus_q": np.full((39, 2, 4, 2), 0.01),
        "filter_max_abs_qm_minus_q": np.full((39, 2, 4, 2), 0.001),
    }
    return scores, a0, coverage, actuation


@pytest.mark.parametrize(
    "expected,kwargs",
    [
        ("DEVELOPMENT_CALIBRATION_BENEFIT_CANDIDATE", {}),
        ("CLASSIFICATION_INCREMENT_ONLY", {"q3": 20, "qm3": 22, "q5": 22, "full3": 22}),
        ("METADATA_INCREMENT_NOT_ESTABLISHED", {"qm3": 38}),
    ],
)
def test_fixed_terminal_branches_and_json_safe_summary(expected, kwargs):
    import json

    result = audit.summarize(*summary_inputs(**kwargs))
    assert result["terminal"] == expected
    assert result["grid_attainment"]["cells"] == 312
    assert result["coverage"]["changed_units"] == 78
    assert result["comparisons"]["QM3_minus_Q3"]["df"] == 38
    assert result["automatic_held60_promotion"] is False
    json.dumps(result, allow_nan=False)


def test_positive_fixture_has_exact_integer_gates_costs_and_effects():
    result = audit.summarize(*summary_inputs())
    primary = result["comparisons"]["QM3_minus_Q3"]
    assert primary["participant_integer_net_correct"] == [16] * 39
    assert primary["helped"] == 39 and primary["harmed"] == primary["tied"] == 0
    assert primary["mean_pp"] == pytest.approx(100 * 2 / 48)
    grid = result["grid_attainment"]
    assert grid["both_attain"] == 312
    assert grid["pooled_label_savings_mean"] == 24
    assert grid["pooled_label_savings_total"] == 312 * 24
    assert result["actuation"]["by_k"]["3"]["argmax_changed_queries"] == 312 * 2
    assert result["actuation"]["by_k"]["3"]["changed_to_correct"] == 624


def test_all_zero_a0_cost_is_shared_and_not_positive_saving():
    result = audit.summarize(*summary_inputs(a0_correct=40))
    assert result["grid_attainment"]["q_cost_mean_among_both"] == 0
    assert result["grid_attainment"]["qm_cost_mean_among_both"] == 0
    assert result["terminal"] == "CLASSIFICATION_INCREMENT_ONLY"


def test_unattained_costs_remain_null():
    result = audit.summarize(*summary_inputs(q3=20, qm3=22, q5=22, full3=22))
    grid = result["grid_attainment"]
    assert grid["both_attain"] == 0 and grid["neither_attain"] == 312
    assert grid["pooled_label_savings_total"] is None
    assert grid["pooled_label_savings_mean"] is None
    assert grid["q_labels"][0][0][0] is None


def test_no_argmax_change_is_not_structural_absence():
    scores, a0, coverage, actuation = summary_inputs(qm3=38)
    scores[..., audit.ARMS.index("QM"), :, :] += 1e-5
    result = audit.summarize(scores, a0, coverage, actuation)
    assert result["terminal"] == "METADATA_INCREMENT_NOT_ESTABLISHED"
    assert result["actuation"]["structural_no_actuation"] is False
    assert result["actuation"]["by_k"]["3"]["argmax_changed_queries"] == 0
    assert result["actuation"]["by_k"]["3"]["score_changed_cells_exact"] == 312


def test_validity_and_structural_precedence():
    scores, a0, coverage, actuation = summary_inputs(qm3=38)
    actuation["qm_coefficients"][:] = 0
    assert audit.summarize(scores, a0, coverage, actuation)["terminal"] == "STRUCTURAL_NO_ACTUATION"
    actuation["validity_errors"] = ["artificial independent saved-filter mismatch"]
    assert audit.summarize(scores, a0, coverage, actuation)["terminal"] == "VALIDITY_FAILURE"


def test_k3_absence_does_not_prove_k5_absence():
    scores, a0, coverage, actuation = summary_inputs(qm3=38)
    coverage["m"][:] = coverage["donor_m"][:] = 0
    coverage["available"][:] = False
    assert (
        audit.summarize(scores, a0, coverage, actuation)["terminal"]
        == "METADATA_INCREMENT_NOT_ESTABLISHED"
    )
    actuation["all_budget_available"] = np.zeros((39, 2, 2, 8), bool)
    assert audit.summarize(scores, a0, coverage, actuation)["terminal"] == "STRUCTURAL_NO_ACTUATION"


def test_half_coverage_is_inclusive_and_denominator_preserves_singletons():
    scores, a0, coverage, actuation = summary_inputs()
    coverage["donor_m"].reshape(78, 8, 2)[39:] = 0
    result = audit.summarize(scores, a0, coverage, actuation)
    assert result["coverage"]["fraction"] == 0.5
    assert result["terminal"] == "DEVELOPMENT_CALIBRATION_BENEFIT_CANDIDATE"
    coverage["donor_m"].reshape(78, 8, 2)[38] = 0
    assert (
        audit.summarize(scores, a0, coverage, actuation)["terminal"]
        == "METADATA_INCREMENT_NOT_ESTABLISHED"
    )


def test_paired_ci_uses_participants_not_312_independent_cells():
    scores, a0, coverage, actuation = summary_inputs()
    truth = np.tile(np.arange(12), 4)
    qm = audit.ARMS.index("QM")
    for participant in range(39):
        count = 37 + participant % 5
        prediction = np.where(np.arange(48) < count, truth, (truth + 1) % 12)
        scores[participant, ..., 0, qm, :, :] = 0
        scores[participant, ..., 0, qm, np.arange(48), prediction] = 1
    result = audit.summarize(scores, a0, coverage, actuation)
    pp = (np.array([37 + p % 5 for p in range(39)]) - 38) / 48 * 100
    half = stats.t.ppf(0.975, 38) * np.std(pp, ddof=1) / np.sqrt(39)
    primary = result["comparisons"]["QM3_minus_Q3"]
    assert primary["ci_low_pp"] == pytest.approx(pp.mean() - half)
    assert primary["helped"] == int((pp > 0).sum())
    assert primary["tied"] == int((pp == 0).sum())
    assert primary["harmed"] == int((pp < 0).sum())


@pytest.mark.parametrize(
    "field", ["scores", "a0", "weights", "coverage", "available", "actuation", "missing"]
)
def test_summary_rejects_invalid_or_incomplete_evidence(field):
    scores, a0, coverage, actuation = summary_inputs()
    if field == "scores":
        scores.flat[0] = np.nan
    elif field == "a0":
        a0 = a0[:38]
    elif field == "weights":
        actuation.pop("band_weights")
    elif field == "coverage":
        coverage = 0.9
    elif field == "available":
        coverage["available"] = coverage["available"].astype(float)
    elif field == "actuation":
        actuation["filter_max_abs_qm_minus_q"].flat[0] = -1
    else:
        scores[..., audit.ARMS.index("MISSING"), :, :] += 1e-15
    with pytest.raises(ValueError):
        audit.summarize(scores, a0, coverage, actuation)


def test_summary_nll_respects_interface_specific_native_weights():
    scores, a0, coverage, actuation = summary_inputs()
    actuation["band_weights"][1] *= 2
    result = audit.summarize(scores, a0, coverage, actuation)
    # Wet scores have the same argmax, but a different recorded native score scale.
    assert result["conditions"][0]["accuracy"] == result["conditions"][4]["accuracy"]
    assert result["conditions"][0]["nll"] != result["conditions"][4]["nll"]


def test_artificial_fitted_pipeline_all_positive_arms_match_independent_audit():
    import torch

    from cfeg.analysis import task_trca_shape_learning as learning

    previous_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        rng = np.random.default_rng(772002)
        cases = []
        weights = np.array([1.25, 0.67, 0.5, 0.426, 0.384])
        for pid in (1001, 1002):
            prototype = rng.normal(size=(12, 5, 8, 19))
            support = prototype[None] + rng.normal(size=(3, 12, 5, 8, 19))
            source = prototype + rng.normal(size=prototype.shape)
            packet = rng.uniform(0, 20, (3, 8))
            packet[0, 0] = np.nan
            packet[:, 2] = np.nan
            cases.append(
                learning.make_task_case(
                    pid, 0, 0, support, packet, np.linspace(9, 14.5, 12), source, weights=weights
                )
            )
        fitted = learning.fit_pipeline(cases, 0.001)
        record = fitted.record()
        for index, case in enumerate(cases):
            m, available = audit.independent_metadata(case.packet)
            np.testing.assert_allclose(m, case.m, atol=5e-16)
            np.testing.assert_array_equal(available, case.available)
            stale_m, stale_available = audit.independent_metadata(
                np.repeat(case.packet[:1], 3, axis=0)
            )
            statistics = {
                name: getattr(case.statistics, name).numpy()
                for name in (
                    "query_gram",
                    "template_gram",
                    "cross_gram",
                    "query_mean",
                    "template_mean",
                )
            }
            statistics["samples"] = case.samples
            for arm in audit.ARMS[1:]:
                prior = audit.independent_prior(
                    case.q,
                    m,
                    available,
                    record,
                    arm,
                    donor_m=cases[1 - index].m,
                    stale_m=stale_m,
                    stale_available=stale_available,
                )
                filters = audit.independent_filters(
                    case.s.numpy(), case.c.numpy(), case.anchors.numpy(), prior
                )
                independent = audit.independent_scores(filters, statistics, weights)
                producer = learning.predict(fitted, cases, arm)[case.key]
                np.testing.assert_allclose(independent, producer, atol=2e-12, rtol=2e-12)
                np.testing.assert_array_equal(independent.argmax(-1), producer.argmax(-1))
    finally:
        torch.set_num_threads(previous_threads)
