"""Generated-array math, synthetic records and adversarial ten-arm endpoints."""

import ast
import copy
import hashlib
import inspect
import json

import numpy as np
import pytest
from scipy import linalg, special

from cfeg.analysis import task_trca_shape_audit as old
from cfeg.analysis import task_trca_temporal_audit as audit


def record(data, ids, lam=0.01):
    positions = np.flatnonzero(np.isin(data["keys"][:, 0], ids))
    available, q = data["available"][positions], data["q"][positions]
    result = {
        "schema": audit.SCHEMA,
        "score_schema": audit.SCORE_SCHEMA,
        "fit_ids": list(ids),
        "lambda": lam,
    }
    for name, values, mask in (
        ("q_scaler", q, np.ones(q.shape[:-1], bool)),
        ("m_scaler", data["m"][positions], available),
        ("q2_scaler", q[..., 1:3], np.broadcast_to(available[:, None], q.shape[:-1])),
    ):
        rows = values[mask]
        mean = rows.mean(0) if len(rows) else np.zeros(values.shape[-1])
        scale = rows.std(0) if len(rows) else np.ones(values.shape[-1])
        scale[scale < 1e-12] = 1
        result[name] = {"mean": mean.tolist(), "scale": scale.tolist(), "fit_ids": list(ids)}

    def head(dim):
        return {
            "coefficients": [0.0] * dim,
            "initial_loss": 1.0,
            "final_loss": 1.0,
            "steps": 200,
            "trace": [
                {"step": step, "loss_before_step": 1.0, "ce_before_step": 1.0, "gradient_norm": 0.0}
                for step in range(1, 201)
            ],
        }

    result["Q"] = head(16)
    result["residuals"] = {arm: head(3) for arm in ("Q2", "QM", "SHAM_REFIT")}
    result["donors"] = [
        {"case": data["keys"][j].tolist(), "donor_id": int(data["keys"][d, 0])}
        for j, d in audit.donor_positions(data, positions).items()
    ]
    digest = hashlib.sha256()
    for values in (
        result["q_scaler"]["mean"],
        result["q_scaler"]["scale"],
        result["Q"]["coefficients"],
    ):
        digest.update(np.asarray(values, dtype="<f8").tobytes())
    result["q_hash"] = digest.hexdigest()
    return result


def fixture(*, conditions=((0, 17, 3),), missing=False, participants=6):
    rng = np.random.default_rng(20260909)
    ids = tuple(range(1001, 1001 + participants))
    keys = np.array([[pid, *condition] for pid in ids for condition in conditions], dtype=np.int64)
    arrays = {
        name: []
        for name in (
            "q",
            "m",
            "available",
            "packet5",
            "s",
            "c",
            "weights",
            "labels",
            "query_gram",
            "template_gram",
            "cross_gram",
        )
    }
    losses = []
    for index, (pid, interface, samples, k) in enumerate(keys):
        packet = np.full((5, 8), np.nan)
        if not missing:
            # Deterministic identity across windows and budgets; realistic mask strata.
            packet[:k] = np.arange(k * 8).reshape(k, 8) + (pid - 1000)
            packet[0, 0] = np.nan
            packet[:, 2] = np.nan
        m, available = old.independent_metadata(packet[:k])
        q = rng.normal(size=(5, 8, 15))
        q[..., 4] = np.isfinite(packet[:k]).mean(0)
        template = rng.normal(size=(12, 5, 8, samples)) + rng.normal(size=(12, 5, 8, 1))
        labels = np.roll(np.arange(12), index % 12)
        query = template[labels] + rng.normal(size=template.shape) * 0.2
        template -= template[..., :1]
        query -= query[..., :1]
        template -= template.mean(-1, keepdims=True)
        query -= query.mean(-1, keepdims=True)
        weight = np.array([1.25, 0.67, 0.5, 0.43, 0.38]) * (1 + interface)
        # All S/C have a unique e8 top direction for any tested zero head.
        s = np.broadcast_to(np.diag(np.arange(1, 9, dtype=np.float64)), (5, 12, 8, 8)).copy()
        c = np.broadcast_to(np.eye(8), s.shape).copy()
        corr = np.einsum("nbt,cbt->nbc", query[:, :, 7], template[:, :, 7])
        corr /= np.linalg.norm(query[:, :, 7], axis=-1)[:, :, None]
        corr /= np.linalg.norm(template[:, :, 7], axis=-1).T[None]
        values = np.einsum("nbc,b->nc", corr, weight)
        logits = values / weight.sum() / 0.1
        losses.append(
            float((special.logsumexp(logits, axis=-1) - logits[np.arange(12), labels]).mean())
        )
        item = {
            "q": q,
            "m": m,
            "available": available,
            "packet5": packet,
            "s": s,
            "c": c,
            "weights": weight,
            "labels": labels,
            "query_gram": query @ query.swapaxes(-1, -2),
            "template_gram": template @ template.swapaxes(-1, -2),
            "cross_gram": np.einsum("nbit,cbjt->ncbij", query, template),
        }
        for name, values in arrays.items():
            values.append(item[name])
    data = {name: np.stack(values) for name, values in arrays.items()}
    data.update(keys=keys, orders=np.zeros(len(keys), dtype=np.int64))
    rows = []
    for fold in range(3):
        val_ids = ids[fold::3]
        fit_ids = tuple(pid for pid in ids if pid not in val_ids)
        loss = float(np.mean([value for j, value in enumerate(losses) if keys[j, 0] in val_ids]))
        for lam in audit.LAMBDAS:
            rows.append(
                {
                    "inner_fold": fold,
                    "lambda": lam,
                    "fit_ids": list(fit_ids),
                    "validation_ids": list(val_ids),
                    "pipeline": record(data, fit_ids, lam),
                    "validation_ce": {arm: loss for arm in ("Q", "Q2", "QM", "SHAM_REFIT")},
                }
            )
    selection = {
        "schema": audit.SCHEMA,
        "score_schema": audit.SCORE_SCHEMA,
        "selection_signal": "Q CE only",
        "selected_lambda": 0.01,
        "outer_evaluation_ids": [2001, 2002],
        "inner": rows,
    }
    return data, {"pipeline": record(data, ids), "selection": selection}, ids, (2001, 2002)


@pytest.fixture(scope="module")
def generated():
    return fixture()


def test_no_producer_imports_in_independent_module():
    tree = ast.parse(inspect.getsource(audit))
    imports = [node.module for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)]
    imports += [
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    ]
    assert not any(
        "torch" in name
        or any(
            word in name for word in ("learning", "features", "operator", "signfree", "evaluation")
        )
        for name in imports
    )


@pytest.mark.parametrize("field", ["schema", "score_schema"])
def test_prior_rejects_old_model_even_for_iso(generated, field):
    data, model, _, _ = generated
    pipeline = copy.deepcopy(model["pipeline"])
    pipeline.pop(field)
    with pytest.raises(ValueError, match="schema"):
        audit.independent_prior(data["q"][0], None, None, pipeline, "ISO")


def test_prior_same_head_equations_without_reading_numeric_m(generated):
    class Poison:
        def __array__(self, *args, **kwargs):
            raise AssertionError("Numeric M accessed")

    data, model, _, _ = generated
    for arm in ("Q", "MISSING", "Q2"):
        got = audit.independent_prior(
            data["q"][0], Poison(), data["available"][0], model["pipeline"], arm
        )
        np.testing.assert_array_equal(got, np.ones((5, 8)))


@pytest.mark.parametrize("arm", audit.ARMS[2:])
def test_all_prior_controls_match_independent_legacy_head(generated, arm):
    data, model, _, _ = generated
    pipeline = copy.deepcopy(model["pipeline"])
    for head in pipeline["residuals"].values():
        head["coefficients"] = [0.2, -0.3, 0.1]
    stale, stale_available = old.independent_metadata(np.repeat(data["packet5"][0, :1], 3, axis=0))
    arguments = (data["q"][0], data["m"][0], data["available"][0], pipeline, arm)
    kwargs = {"donor_m": data["m"][1], "stale_m": stale, "stale_available": stale_available}
    np.testing.assert_array_equal(
        audit.independent_prior(*arguments, **kwargs), old.independent_prior(*arguments, **kwargs)
    )


def test_projector_is_c_normalized_psd_top_outer_product_and_anchor_free():
    rng = np.random.default_rng(20260909)
    a = rng.normal(size=(5, 12, 8, 8))
    c = a @ a.swapaxes(-1, -2) + np.eye(8)
    a = rng.normal(size=c.shape)
    s = a @ a.swapaxes(-1, -2)
    r = np.exp(rng.uniform(-0.2, 0.2, (5, 8)))
    r *= 8 / r.sum(-1, keepdims=True)
    f = audit.independent_projectors(s, c, r)
    np.testing.assert_allclose(np.einsum("bcij,bcij->bc", f, c), 1, atol=2e-15)
    np.testing.assert_allclose(f @ c @ f, f, atol=1e-15)
    for band in range(5):
        for label in range(12):
            penalty = 0.1 * np.linalg.eigvalsh(c[band, label])[0] / (16 / 9) * np.diag(r[band])
            roots = linalg.eigvalsh(s[band, label], c[band, label] + penalty)
            np.testing.assert_allclose(
                s[band, label] @ f[band, label],
                roots[-1] * (c[band, label] + penalty) @ f[band, label],
                atol=3e-14,
                rtol=2e-12,
            )


@pytest.mark.parametrize("defect", ["tie", "indefinite", "asymmetric", "r", "nonfinite"])
def test_projector_invalidity_is_not_rescued(generated, defect):
    data, _, _, _ = generated
    s, c, r = data["s"][0].copy(), data["c"][0].copy(), np.ones((5, 8))
    if defect == "tie":
        s[:] = np.eye(8)
    elif defect == "indefinite":
        c[0, 0, 0, 0] = -1
    elif defect == "asymmetric":
        s[0, 0, 0, 1] = 1
    elif defect == "r":
        r[0, 0] += 1
    else:
        s[0, 0, 0, 0] = np.nan
    with pytest.raises(ValueError):
        audit.independent_projectors(s, c, r)


def test_scores_match_literal_component_temporal_pearson_and_ignore_sign():
    rng = np.random.default_rng(20260909)
    w = rng.normal(size=(5, 8, 12))
    template = rng.normal(size=(12, 5, 8, 23)) + rng.normal(size=(12, 5, 8, 1)) * 3
    query = rng.normal(size=(7, 5, 8, 23)) + rng.normal(size=(7, 5, 8, 1)) * 4
    t, x = template - template.mean(-1, keepdims=True), query - query.mean(-1, keepdims=True)
    statistics = {
        "query_gram": x @ x.swapaxes(-1, -2),
        "template_gram": t @ t.swapaxes(-1, -2),
        "cross_gram": np.einsum("nbit,cbjt->ncbij", x, t),
        "samples": 23,
    }
    f = np.einsum("bik,bjk->bkij", w, w)
    weights = np.array([1.25, 0.67, 0.5, 0.43, 0.38])
    values, corr = audit.independent_scores(f, statistics, weights)
    expected = np.empty_like(corr)
    for n in range(7):
        for b in range(5):
            px = w[b].T @ query[n, b]
            px -= px.mean(-1, keepdims=True)
            for label in range(12):
                pt = w[b].T @ template[label, b]
                pt -= pt.mean(-1, keepdims=True)
                expected[n, b, label] = np.sum(px * pt) / np.linalg.norm(px) / np.linalg.norm(pt)
    np.testing.assert_allclose(corr, expected, atol=5e-16, rtol=1e-13)
    np.testing.assert_allclose(values, np.einsum("nbc,b->nc", expected, weights), atol=5e-16)
    flipped = w * rng.choice([-1, 1], (5, 1, 12))
    np.testing.assert_array_equal(f, np.einsum("bik,bjk->bkij", flipped, flipped))


@pytest.mark.parametrize("defect", ["means", "weight", "zero_variance", "sample", "projector"])
def test_temporal_score_guards(generated, defect):
    data, _, _, _ = generated
    statistics = {name: data[name][0].copy() for name in audit.STATS}
    statistics["samples"] = 17
    f = audit.independent_projectors(data["s"][0], data["c"][0], np.ones((5, 8)))
    weights = data["weights"][0].copy()
    if defect == "means":
        statistics["query_mean"] = np.zeros((12, 5, 8))
    elif defect == "weight":
        weights[0] = 0
    elif defect == "zero_variance":
        statistics["query_gram"][:] = 0
    elif defect == "sample":
        statistics["samples"] = True
    else:
        f *= -1
    with pytest.raises(ValueError):
        audit.independent_scores(f, statistics, weights)


def test_complete_one_condition_synthetic_nested_artifact(generated):
    result = audit.audit_training(*generated)
    assert result["status"] == "TEMPORAL_SOURCE_SELECTION_AUDIT_PASS"
    assert result["selected_lambda"] == 0.01
    assert result["cases"] == 6 and result["validation_ce_comparisons"] == 36
    assert result["max_validation_ce_abs_error"] < 1e-12
    json.dumps(result, allow_nan=False)


def test_complete_multicondition_grid_and_all_missing_metadata():
    result = audit.audit_training(*fixture(conditions=((0, 17, 3), (0, 23, 3)), missing=True))
    assert result["cases"] == 12 and result["selected_lambda"] == 0.01


def test_q_selection_weights_participants_not_unequal_validation_folds(monkeypatch):
    data, model, ids, evaluation = fixture(participants=7)
    losses = {0.0001: (0.1, 1.0, 1.0), 0.001: (0.65, 0.65, 0.65), 0.01: (1.2, 1.2, 1.2)}
    for row in model["selection"]["inner"]:
        row["validation_ce"] = {
            arm: losses[row["lambda"]][row["inner_fold"]] for arm in ("Q", "Q2", "QM", "SHAM_REFIT")
        }
    model["selection"]["selected_lambda"] = model["pipeline"]["lambda"] = 0.0001

    def fake_scores(values, j, pipeline, arm, donors):
        fold = ids.index(int(values["keys"][j, 0])) % 3
        ce = losses[pipeline["lambda"]][fold]
        target = np.log(11 / np.expm1(ce))
        result = np.zeros((12, 12))
        result[np.arange(12), values["labels"][j]] = target * values["weights"][j].sum() * 0.1
        return result

    # Focused selection-aggregation oracle, separate from the real score tests.
    monkeypatch.setattr(audit, "_case_scores", fake_scores)
    result = audit.audit_training(data, model, ids, evaluation)
    assert result["selected_lambda"] == 0.0001
    assert result["weighted_q_validation_ce"]["0.0001"] == pytest.approx((3 * 0.1 + 4) / 7)
    assert np.mean(losses[0.0001]) > np.mean(
        losses[0.001]
    )  # Wrong equal-fold rule reverses winner.


@pytest.mark.parametrize("defect", ["prefix", "order"])
def test_cross_budget_prefix_and_participant_order_identity(defect):
    data, model, ids, evaluation = fixture(conditions=((0, 17, 3), (0, 23, 5)))
    if defect == "prefix":
        data["packet5"][1, 1, 0] += 1
        data["m"][1], data["available"][1] = old.independent_metadata(data["packet5"][1])
    else:
        data["orders"][1] = 1
    with pytest.raises(ValueError, match="prefixes|acquisition order"):
        audit.audit_training(data, model, ids, evaluation)


@pytest.mark.parametrize(
    "defect", ["keys", "m", "labels", "future_m", "global_means", "orders", "availability"]
)
def test_saved_source_corruption_rejected(generated, defect):
    original, model, ids, evaluation = generated
    data = {key: value.copy() for key, value in original.items()}
    if defect == "keys":
        data["keys"][0, 2] += 1
    elif defect == "m":
        data["m"][0, 0, 0] += 0.01
    elif defect == "labels":
        data["labels"][0] = data["labels"][0, ::-1]
    elif defect == "future_m":
        data["packet5"][0, 4, 0] = 1
    elif defect == "global_means":
        data["query_mean"] = np.zeros((6, 12, 5, 8))
    elif defect == "orders":
        data["orders"][0] = 2
    else:
        data["available"] = data["available"].astype(float)
    with pytest.raises((ValueError, AssertionError)):
        audit.audit_training(data, model, ids, evaluation)


@pytest.mark.parametrize(
    "defect",
    [
        "schema",
        "selection_schema",
        "donor",
        "donor_duplicate",
        "scaler",
        "ce",
        "selected",
        "grid",
        "trace",
        "q_hash",
        "extra_arm",
        "overlap",
    ],
)
def test_nested_records_role_selection_and_schema_corruption(generated, defect):
    data, original, ids, evaluation = generated
    model = copy.deepcopy(original)
    row = model["selection"]["inner"][0]
    if defect == "schema":
        row["pipeline"]["score_schema"] = "native-global"
    elif defect == "selection_schema":
        model["selection"].pop("schema")
    elif defect == "donor":
        row["pipeline"]["donors"][0]["donor_id"] = 9999
    elif defect == "donor_duplicate":
        row["pipeline"]["donors"].append(row["pipeline"]["donors"][0])
    elif defect == "scaler":
        row["pipeline"]["q_scaler"]["mean"][0] += 0.1
    elif defect == "ce":
        row["validation_ce"]["QM"] += 0.1
    elif defect == "selected":
        model["selection"]["selected_lambda"] = 0.001
    elif defect == "grid":
        model["selection"]["inner"][-1] = copy.deepcopy(row)
    elif defect == "trace":
        row["pipeline"]["Q"]["trace"][0]["gradient_norm"] = -1
    elif defect == "q_hash":
        model["pipeline"]["q_hash"] = "0" * 64
    elif defect == "extra_arm":
        row["pipeline"]["residuals"]["OTHER"] = copy.deepcopy(row["pipeline"]["Q"])
    else:
        evaluation = (ids[0],)
    with pytest.raises((ValueError, AssertionError)):
        audit.audit_training(data, model, ids, evaluation)


def endpoint_fixture(*, native3=40, centered3=40, q3=38, qm3=40, q5=40):
    scores = np.zeros((39, 2, 4, 2, 10, 48, 12))
    a0 = np.zeros((39, 2, 4, 48, 12))
    truth = np.tile(np.arange(12), 4)
    for position, arm in enumerate(audit.ARMS):
        for budget in range(2):
            count = (
                (
                    qm3
                    if arm == "QM"
                    else native3
                    if arm == "FULL_NATIVE"
                    else centered3
                    if arm == "FULL_CENTERED"
                    else q3
                )
                if budget == 0
                else q5
            )
            prediction = np.where(np.arange(48) < count, truth, (truth + 1) % 12)
            scores[..., budget, position, np.arange(48), prediction] = 1
    a0[..., np.arange(48), (truth + 1) % 12] = 1
    coverage = {
        "m": np.zeros((39, 2, 8, 2)),
        "donor_m": np.ones((39, 2, 8, 2)),
        "available": np.ones((39, 2, 8), bool),
    }
    actuation = {
        "band_weights": np.ones((2, 5)),
        "a0_band_weights": np.ones((2, 5)),
        "qm_coefficients": np.ones((3, 3)),
        **{
            field: np.full((39, 2, 4, 2), 0.01)
            for field in (
                "r_max_abs_qm_minus_q",
                "projector_max_abs_qm_minus_q",
                "j_max_abs_qm_minus_q",
            )
        },
    }
    return scores, a0, coverage, actuation


def test_ten_arm_positive_endpoint_and_original_costs_preserved():
    result = audit.summarize(*endpoint_fixture())
    assert result["terminal"] == "DEVELOPMENT_CALIBRATION_BENEFIT_CANDIDATE"
    assert result["arms"] == list(audit.ARMS)
    assert np.asarray(result["correct_counts"]).shape == (39, 2, 4, 2, 10)
    assert np.asarray(result["nll_by_budget_arm"]).shape == (2, 10)
    assert np.asarray(result["conditions"][0]["accuracy"]).shape == (2, 10)
    assert result["grid_attainment"]["pooled_label_savings_mean"] == 24
    assert result["comparisons"]["QM3_minus_FULL_NATIVE3"]["mean_pp"] == 0
    assert result["comparisons"]["QM3_minus_FULL_CENTERED3"]["mean_pp"] == 0
    assert result["comparisons"]["FULL_CENTERED3_minus_FULL_NATIVE3"]["mean_pp"] == 0
    assert "projector_max_abs_qm_minus_q" in result["actuation"]["by_k"]["3"]
    assert "j_max_abs_qm_minus_q" in result["actuation"]["by_k"]["3"]
    assert "filter_max_abs_qm_minus_q" not in result["actuation"]["by_k"]["3"]
    json.dumps(result, allow_nan=False)


def test_added_centered_guard_only_demotes_in_correct_direction():
    result = audit.summarize(*endpoint_fixture(centered3=41))
    assert result["terminal_before_centered_guard"] == "DEVELOPMENT_CALIBRATION_BENEFIT_CANDIDATE"
    assert result["terminal"] == "CLASSIFICATION_INCREMENT_ONLY"
    assert result["comparisons"]["QM3_minus_FULL_CENTERED3"]["ci_low_pp"] < -1
    assert not result["calibration_additional_checks"][
        "qm3_minus_full_centered3_ci_low_gt_minus_1pp"
    ]


def test_strong_native_guard_cannot_be_replaced_by_easy_centered_control():
    result = audit.summarize(*endpoint_fixture(native3=42, centered3=0))
    assert result["terminal_before_centered_guard"] == "CLASSIFICATION_INCREMENT_ONLY"
    assert result["terminal"] == "CLASSIFICATION_INCREMENT_ONLY"
    assert not result["calibration_additional_checks"]["qm3_minus_full_native3_ci_low_gt_minus_1pp"]
    assert result["calibration_additional_checks"]["qm3_minus_full_centered3_ci_low_gt_minus_1pp"]


@pytest.mark.parametrize(
    "terminal",
    ["METADATA_INCREMENT_NOT_ESTABLISHED", "STRUCTURAL_NO_ACTUATION", "VALIDITY_FAILURE"],
)
def test_centered_control_never_promotes_old_failure_branches(terminal):
    scores, a0, coverage, actuation = endpoint_fixture(centered3=0, qm3=38)
    if terminal == "STRUCTURAL_NO_ACTUATION":
        actuation["qm_coefficients"][:] = 0
    elif terminal == "VALIDITY_FAILURE":
        actuation["validity_errors"] = ["artificial saved-artifact mismatch"]
    result = audit.summarize(scores, a0, coverage, actuation)
    assert result["terminal"] == terminal
    assert result["terminal_before_centered_guard"] == terminal


def test_nonattainment_costs_remain_null_and_argmax_static_not_structural():
    scores, a0, coverage, actuation = endpoint_fixture(q3=20, qm3=20, q5=20, centered3=20)
    scores[..., 5, :, :] += 0.0001
    actuation["j_max_abs_qm_minus_q"][:] = 0
    result = audit.summarize(scores, a0, coverage, actuation)
    assert result["grid_attainment"]["pooled_label_savings_mean"] is None
    assert result["grid_attainment"]["q_labels"][0][0][0] is None
    assert result["terminal"] == "METADATA_INCREMENT_NOT_ESTABLISHED"
    assert result["actuation"]["by_k"]["3"]["argmax_changed_queries"] == 0


@pytest.mark.parametrize("defect", ["old_shape", "missing_diff", "projector", "j", "native_nan"])
def test_summary_rejects_mixed_or_bad_evidence(defect):
    scores, a0, coverage, actuation = endpoint_fixture()
    if defect == "old_shape":
        scores = scores[..., 1:, :, :]
    elif defect == "missing_diff":
        scores[..., 9, :, :] += 1e-15
    elif defect == "projector":
        actuation["projector_max_abs_qm_minus_q"].flat[0] = -1
    elif defect == "j":
        actuation["j_max_abs_qm_minus_q"].flat[0] = np.nan
    else:
        scores[..., 0, :, :] = np.nan
    with pytest.raises((ValueError, AssertionError)):
        audit.summarize(scores, a0, coverage, actuation)
