"""Fixed artificial learning and selection isolation; no efficacy dataset."""

from dataclasses import replace

import numpy as np
import pytest
import torch

from cfeg.analysis import task_trca_shape_learning as learning


def make_case(pid=1001, role="fit", *, missing=False):
    rng = np.random.default_rng(20260908 + pid)
    prototype = rng.normal(size=(12, 5, 8, 17))
    support = prototype[None] + rng.normal(size=(3, 12, 5, 8, 17))
    source = prototype + rng.normal(size=prototype.shape)
    packet = np.full((3, 8), np.nan) if missing else rng.uniform(0, 20, (3, 8))
    return learning.make_task_case(
        pid, 0, 0, support, packet, np.linspace(9, 14.5, 12), source, role=role
    )


@pytest.fixture(scope="module")
def trained():
    original = torch.get_num_threads()
    torch.set_num_threads(1)
    cases = (make_case(), make_case(1002))
    model = learning.fit_pipeline(cases, 0.001)
    yield cases, model
    torch.set_num_threads(original)


def test_actual_fixed_optimizer_and_frozen_q(trained):
    cases, model = trained
    assert model.q.coefficients.shape == (16,)
    assert model.q.final_loss < model.q.initial_loss
    before = model.q_hash
    for arm, fitted in model.residuals.items():
        assert arm in learning.RESIDUAL_ARMS
        assert fitted.coefficients.shape == (3,)
        assert len(fitted.trace) == 200
        assert np.isfinite([r["gradient_norm"] for r in fitted.trace]).all()
    assert len(model.q.trace) == 200
    for arm in ("Q", *learning.RESIDUAL_ARMS, "PERMUTED", "STALE", "MISSING", "ISO"):
        scored = learning.predict(model, cases, arm)
        assert all(v.shape == (12, 12) and np.isfinite(v).all() for v in scored.values())
    assert model.q_hash == before
    assert model.fit_ids == (1001, 1002)
    assert model.record()["q_hash"] == before


def test_missing_and_zero_residual_exact_q(trained):
    cases, model = trained
    q = learning.predict(model, cases, "Q")
    absent = learning.predict(model, cases, "MISSING")
    for key in q:
        np.testing.assert_array_equal(q[key], absent[key])
    zero = replace(model.residuals["QM"], coefficients=np.zeros(3))
    zero_model = replace(model, residuals={**model.residuals, "QM": zero})
    for key, values in learning.predict(zero_model, cases, "QM").items():
        np.testing.assert_array_equal(values, q[key])


def test_training_denies_validation_and_outer_overlap(trained):
    cases, _ = trained
    with pytest.raises(PermissionError):
        learning.fit_pipeline((replace(cases[0], role="validation"),), 0.001)
    with pytest.raises(PermissionError):
        learning.nested_fit(cases, outer_evaluation_ids=(1001,))
    with pytest.raises(ValueError, match="grid"):
        learning.fit_pipeline(cases, 1.0)
    with pytest.raises(ValueError, match="Duplicate"):
        learning.fit_pipeline((cases[0], cases[0]), 0.001)


def test_eval_builder_rejects_before_array_conversion():
    class Poison:
        def __array__(self, *args, **kwargs):
            raise AssertionError("Evaluation block5 was touched")

    with pytest.raises(PermissionError):
        learning.make_task_case(1, 0, 0, Poison(), Poison(), Poison(), Poison(), role="evaluation")


def test_choose_lambda_ignores_all_metadata_validation_outcomes():
    rows = [
        {
            "lambda": value,
            "inner_fold": fold,
            "validation_ids": [fold],
            "validation_ce": {"Q": 1.0, "QM": float("nan")},
        }
        for value in learning.LAMBDAS
        for fold in range(3)
    ]
    assert learning.choose_lambda(rows) == 0.01
    for r in rows:
        r["validation_ce"]["Q"] = {0.0001: 0.5, 0.001: 0.7, 0.01: 0.9}[r["lambda"]]
        r["validation_ce"]["QM"] = -1e20 if r["lambda"] == 0.01 else 1e20
    assert learning.choose_lambda(rows) == 0.0001


def test_nested_entire_fit_graph_rebuilt_with_disjoint_people(monkeypatch, trained):
    cases, _ = trained
    # Structural graph test with a fit spy, not an actual nested efficacy run.
    expanded = tuple(replace(cases[0], participant_id=pid) for pid in range(1001, 1007))
    fits = []

    class Fitted:
        def __init__(self, ids, lam):
            self.ids, self.lam = ids, lam

        def record(self):
            return {"fit_ids": list(self.ids)}

    def fit(rows, lam):
        assert all(c.role == "fit" for c in rows)
        ids = tuple(sorted(c.participant_id for c in rows))
        fits.append((ids, lam))
        return Fitted(ids, lam)

    def val(model, rows, arm):
        assert all(c.role == "validation" for c in rows)
        assert not set(model.ids) & {c.participant_id for c in rows}
        return 1.0 if arm == "Q" else -model.lam

    monkeypatch.setattr(learning, "fit_pipeline", fit)
    monkeypatch.setattr(learning, "validation_ce", val)
    model, receipt = learning.nested_fit(expanded, outer_evaluation_ids=(2001,))
    assert len(fits) == 10 and len(receipt["inner"]) == 9
    assert receipt["selected_lambda"] == 0.01
    assert len(model.ids) == 6
    for row in receipt["inner"]:
        assert len(row["fit_ids"]) == 4 and len(row["validation_ids"]) == 2


def test_no_available_metadata_neutral_scalers_and_zero_residuals():
    original = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        case = make_case(missing=True)
        model = learning.fit_pipeline((case,), 0.001)
        for scaler in (model.m_scaler, model.q2_scaler):
            np.testing.assert_array_equal(scaler.mean, np.zeros(2))
            np.testing.assert_array_equal(scaler.scale, np.ones(2))
        q = learning.predict(model, (case,), "Q")[case.key]
        for arm in learning.RESIDUAL_ARMS:
            np.testing.assert_array_equal(model.residuals[arm].coefficients, np.zeros(3))
            assert len(model.residuals[arm].trace) == 200
            np.testing.assert_array_equal(learning.predict(model, (case,), arm)[case.key], q)
    finally:
        torch.set_num_threads(original)
