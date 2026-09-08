"""Generated arrays only; old/new schema, label-free and frozen partition checks."""

from copy import deepcopy
from dataclasses import replace

import numpy as np
import pytest
import torch

from cfeg.analysis import task_trca_temporal_evaluation as ev
from cfeg.analysis import task_trca_temporal_learning as learn


@pytest.fixture(scope="module")
def generated():
    torch.set_num_threads(1)
    rng = np.random.default_rng(20260909)
    frequency = np.linspace(9, 14.5, 12)
    weights = np.array([1.25, 0.67, 0.50, 0.43, 0.38])
    source, bundles = [], []
    for pid in range(11001, 11006):
        prototype = rng.normal(size=(12, 5, 8, 17))
        x = prototype[None] + rng.normal(size=(3, 12, 5, 8, 17))
        x += rng.normal(size=(3, 12, 5, 8, 1))
        y = prototype + rng.normal(size=prototype.shape)
        packet = rng.uniform(0, 40, (3, 8))
        bundles.append((pid, x, y, packet))
        if pid <= 11002:
            source.append(learn.make_task_case(pid, 0, 0, x, packet, frequency, y, weights=weights))
    model = learn.fit_pipeline(tuple(source), 0.001, backend="batch")
    states = tuple(
        ev.support_state(pid, 0, 0, x, packet, frequency, weights)
        for pid, x, y, packet in bundles[2:]
    )
    partition = ev.EvaluationPartition(states, (11003, 11004, 11005), ((0, 17, 3),))
    return model, states, partition, source, bundles, frequency, weights


def literal_centered(w, templates, query, weights):
    x = np.einsum("bik,nbit->nbkt", w, query)
    t = np.einsum("bik,cbit->cbkt", w, templates)
    x = (x - x.mean(-1, keepdims=True)).reshape(len(query), 5, -1)
    t = (t - t.mean(-1, keepdims=True)).reshape(12, 5, -1)
    x /= np.linalg.norm(x, axis=-1, keepdims=True)
    t /= np.linalg.norm(t, axis=-1, keepdims=True)
    return np.einsum("nbt,cbt,b->nc", x, t, weights)


def test_native_exact_centered_same_filters_and_validation_matching(generated):
    model, states, partition, _, bundles, frequencies, weights = generated
    restored = ev.pipeline_from_record(model.record())
    assert restored.record() == model.record()
    state = states[0]
    query = np.tile(bundles[2][2], (4, 1, 1, 1))
    out = ev.evaluate(restored, state, query, partition=partition)
    assert out["scores"].shape == (10, 48, 12)
    assert out["projectors"].shape == (8, 5, 12, 8, 8)
    assert not hasattr(state, "labels") and not hasattr(state, "statistics")
    expected, correlations = ev.native.score_trca(state.model, query, weights)
    np.testing.assert_array_equal(out["scores"][0], expected)
    np.testing.assert_array_equal(out["native_full_correlations"], correlations)
    np.testing.assert_array_equal(out["native_filters"], state.model.filters)
    centered = literal_centered(state.model.filters, state.model.templates, query, weights)
    np.testing.assert_allclose(out["scores"][1], centered, atol=1e-12, rtol=0)
    assert np.max(np.abs(out["scores"][0] - out["scores"][1])) > 1e-5
    validation = tuple(
        learn.make_task_case(
            pid, 0, 0, x, packet, frequencies, y, weights=weights, role="validation"
        )
        for pid, x, y, packet in bundles[2:]
    )
    for arm in ev.POSITIVE_ARMS:
        expected = learn.predict(model, validation, arm)[state.key]
        np.testing.assert_allclose(
            out["scores"][ev.ARMS.index(arm)], np.tile(expected, (4, 1)), atol=1e-10, rtol=0
        )
    np.testing.assert_array_equal(out["scores"][3], out["scores"][-1])
    assert set(out["actuation_QM_minus_Q"]) == {
        "max_abs_R",
        "max_abs_F",
        "max_abs_J",
        "max_abs_score",
        "argmax_changed",
    }
    assert "query_mean" not in out["statistics"] and "query_mean" in out["native_statistics"]


class ForbiddenArray:
    def __array__(self, *args, **kwargs):
        pytest.fail("query values must not be accessed before role/partition validation")


def test_frozen_partition_completeness_and_donor_before_query_access(generated):
    model, states, partition, _source, bundles, frequencies, weights = generated
    for incomplete in (states[:1], states[:-1]):
        with pytest.raises(PermissionError, match="frozen expected"):
            ev.evaluate(
                model, states[0], ForbiddenArray(), partition=replace(partition, states=incomplete)
            )
    with pytest.raises(PermissionError, match="deterministic"):
        ev.evaluate(model, states[0], ForbiddenArray(), partition=partition, donor=states[0])
    first = bundles[0]
    fitstate = ev.support_state(first[0], 0, 0, first[1], first[3], frequencies, weights)
    bad = ev.EvaluationPartition((fitstate,), (first[0],), ((0, 17, 3),))
    with pytest.raises(PermissionError, match="fit IDs"):
        ev.evaluate(model, fitstate, ForbiddenArray(), partition=bad)
    with pytest.raises(TypeError, match="EvaluationPartition"):
        ev.evaluate(model, states[0], ForbiddenArray(), partition=states)


def test_numeric_m_changes_do_not_change_q(generated):
    model, states, partition, _, bundles, frequencies, weights = generated
    pid, x, _y, packet = bundles[2]
    altered = ev.support_state(pid, 0, 0, x, packet * 3 + 7, frequencies, weights)
    changed = replace(partition, states=(altered, *states[1:]))
    np.testing.assert_array_equal(states[0].q, altered.q)
    a = ev.frozen_prior(model, states[0], "Q", partition=partition)
    b = ev.frozen_prior(model, altered, "Q", partition=changed)
    np.testing.assert_array_equal(a, b)
    with pytest.raises(ValueError, match="Positive-mass"):
        ev.frozen_prior(model, altered, "FULL_CENTERED", partition=changed)


@pytest.mark.parametrize("field", ["schema", "score_schema", "q_hash"])
def test_wrong_pipeline_identity_rejected(generated, field):
    record = deepcopy(generated[0].record())
    record[field] = "old-or-tampered"
    with pytest.raises(ValueError):
        ev.pipeline_from_record(record)


@pytest.mark.parametrize("field", ["schema", "score_schema"])
def test_legacy_untagged_record_rejected(generated, field):
    record = deepcopy(generated[0].record())
    del record[field]
    with pytest.raises(ValueError, match="schema"):
        ev.pipeline_from_record(record)


def test_record_budget_scaler_donor_and_coefficient_guards(generated):
    record = generated[0].record()
    bad = deepcopy(record)
    bad["Q"]["steps"] = 199
    with pytest.raises(ValueError, match="steps"):
        ev.pipeline_from_record(bad)
    bad = deepcopy(record)
    bad["m_scaler"]["fit_ids"] = [11003]
    with pytest.raises(ValueError, match="fit IDs"):
        ev.pipeline_from_record(bad)
    bad = deepcopy(record)
    bad["donors"][0]["donor_id"] = 11003
    with pytest.raises(ValueError, match="donor"):
        ev.pipeline_from_record(bad)
    bad = deepcopy(record)
    bad["residuals"]["QM"]["coefficients"][0] = float("nan")
    with pytest.raises(ValueError, match="finite"):
        ev.pipeline_from_record(bad)


def test_packed_source_schema_and_missing_padding(generated):
    packed = ev.pack_cases(generated[3])
    assert str(packed["schema"]) == ev.SCHEMA
    assert str(packed["score_schema"]) == ev.SCORE_SCHEMA
    assert packed["packet5"].shape == (2, 5, 8)
    assert np.isnan(packed["packet5"][:, 3:]).all()
    assert "query_mean" not in packed and "template_mean" not in packed


def test_native_model_and_returned_filters_are_immutable(generated):
    model, states, partition, _, bundles, _, _ = generated
    state = states[0]
    before = state.model.filters.copy()
    for array in (state.model.filters, state.model.templates):
        with pytest.raises(ValueError):
            array.flat[0] = 17
        with pytest.raises(ValueError):
            array.flags.writeable = True
    out = ev.evaluate(model, state, np.tile(bundles[2][2], (4, 1, 1, 1)), partition=partition)
    with pytest.raises(ValueError):
        out["native_filters"].flat[0] = 17
    np.testing.assert_array_equal(state.model.filters, before)


def test_pack_rejects_bad_role_duplicate_or_sample_count(generated):
    source = tuple(generated[3])
    with pytest.raises(PermissionError):
        ev.pack_cases((replace(source[0], role="evaluation"),))
    with pytest.raises(ValueError, match="Duplicate"):
        ev.pack_cases((source[0], source[0]))
    with pytest.raises(ValueError):
        ev.pack_cases((replace(source[0], samples=23),))


@pytest.mark.parametrize("pid", [True, 0, 1.5])
def test_invalid_participant_rejected_before_support_access(pid):
    with pytest.raises(ValueError, match="Participant"):
        ev.support_state(pid, 0, 0, ForbiddenArray(), ForbiddenArray(), None, None)
