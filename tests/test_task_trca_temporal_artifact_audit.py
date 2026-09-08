"""Generated CPU-only independent artifact replay and adversarial contracts."""

import ast
import copy
import inspect

import numpy as np
import pytest
import torch
from test_task_trca_temporal_audit import fixture as source_fixture

from cfeg.analysis import task_trca_temporal_artifact_audit as audit
from cfeg.analysis import task_trca_temporal_evaluation as ev


def generated_source():
    data, model, old_ids, _ = source_fixture(
        conditions=((0, 17, 3), (0, 17, 5), (1, 17, 3), (1, 17, 5))
    )
    for j, labels in enumerate(data["labels"]):
        permutation = np.argsort(labels)
        data["query_gram"][j] = data["query_gram"][j, permutation]
        data["cross_gram"][j] = data["cross_gram"][j, permutation]
        data["labels"][j] = np.arange(12)
    ids = tuple(audit.profile_record("generated")["source_ids"])
    evaluation_ids = ids[::3]
    fitting = tuple(pid for pid in ids if pid not in evaluation_ids)
    mapping = dict(zip(old_ids, fitting))
    for old, new in mapping.items():
        data["keys"][data["keys"][:, 0] == old, 0] = new

    def translate(value):
        if isinstance(value, dict):
            return {key: translate(item) for key, item in value.items()}
        if isinstance(value, list):
            return [translate(item) for item in value]
        return mapping.get(value, value) if type(value) is int else value

    model = translate(model)
    model["selection"]["outer_evaluation_ids"] = list(evaluation_ids)
    return data, model, fitting, evaluation_ids


@pytest.fixture(scope="module")
def generated():
    torch.set_num_threads(1)
    _source, record, _fitting, evaluation_ids = generated_source()
    pipeline = ev.pipeline_from_record(record["pipeline"])
    rng = np.random.default_rng(20260909)
    frequency = np.linspace(9, 14.5, 12)
    states, queries = [], {}
    for pid in evaluation_ids:
        for interface in (0, 1):
            prototype = rng.normal(size=(12, 5, 8, 17))
            x = prototype[None] + rng.normal(size=(5, 12, 5, 8, 17))
            packet = rng.uniform(0, 40, (5, 8))
            packet[:, 2] = np.nan
            weights = np.array([1.25, 0.67, 0.5, 0.43, 0.38]) * (interface + 1)
            query = np.tile(prototype + rng.normal(size=prototype.shape), (4, 1, 1, 1))
            queries[pid, interface] = query
            for k in (3, 5):
                states.append(
                    ev.support_state(pid, interface, 0, x[:k], packet[:k], frequency, weights)
                )
    partition = ev.EvaluationPartition(
        tuple(states), evaluation_ids, ((0, 17, 3), (0, 17, 5), (1, 17, 3), (1, 17, 5))
    )
    rows = []
    for state in states:
        result = ev.evaluate(
            pipeline, state, queries[state.participant_id, state.interface], partition=partition
        )
        result.update(
            keys=np.array(state.key),
            orders=state.order,
            q=state.q,
            m=state.m,
            available=state.available,
            packet5=np.pad(state.packet, ((0, 5 - state.k), (0, 0)), constant_values=np.nan),
            s=state.s.numpy(),
            c=state.c.numpy(),
            anchors=state.anchors.numpy(),
            weights=state.weights.numpy(),
            donor_id=ev._partition(pipeline, state, partition).participant_id,
            a0_correlations=result["native_full_correlations"].copy(),
            cached_full_correlations=result["native_full_correlations"].copy(),
        )
        rows.append(result)
    return audit.pack_evaluation(rows), record["pipeline"], evaluation_ids, rows


def test_fixed_profiles_and_human_grid_are_not_inferred():
    human = audit.profile_record()
    generated = audit.profile_record("generated")
    assert len(human["source_ids"]) == 39 and human["samples"] == [125, 188, 250, 500]
    assert generated["source_ids"] == [4, 6, 8, 11, 14, 21, 22, 25, 28]
    assert generated["seed"] == 20260909
    for bad in (None, {}, "synthetic", True):
        with pytest.raises(ValueError):
            audit.profile_record(bad)
    data = {
        "keys": np.array(
            [
                [p, i, n, k]
                for p in human["source_ids"]
                for i in (0, 1)
                for n in human["samples"]
                for k in (3, 5)
            ]
        )
    }
    audit._grid(data, tuple(human["source_ids"]))
    with pytest.raises(ValueError, match="Cartesian"):
        audit._grid({"keys": data["keys"][:-1]}, tuple(human["source_ids"]))


def test_training_fixed_generated_wrapper_and_human_rejection():
    data, model, fitting, evaluation_ids = generated_source()
    receipt = audit.audit_training(data, model, fitting, evaluation_ids, profile="generated")
    assert receipt["status"] == "TEMPORAL_SOURCE_SELECTION_AUDIT_PASS"
    assert receipt["validation_ce_comparisons"] == 36
    assert receipt["fixed_profile_grid_verified"] and not receipt["fixed_source39_grid_verified"]
    with pytest.raises(ValueError, match="partition"):
        audit.audit_training(data, model, fitting, evaluation_ids)
    bad = {**data, "labels": data["labels"].copy()}
    bad["labels"][0] = np.roll(bad["labels"][0], 1)
    with pytest.raises(AssertionError):
        audit.audit_training(bad, model, fitting, evaluation_ids, profile="generated")


def test_saved_generated_ten_arm_replay_and_flat_headers(generated, tmp_path):
    data, pipeline, ids, _ = generated
    path = tmp_path / "generated.npz"
    with path.open("xb") as stream:
        np.savez(stream, **data)
    with np.load(path, allow_pickle=False) as arrays:
        saved = {name: arrays[name] for name in arrays.files}
    result = audit.audit_evaluation(saved, pipeline, ids, profile="generated")
    assert result["cases"] == 12 and result["integer_count_comparisons"] == 132
    assert result["both_full_same_native_filters"] and result["ten_arm_argmax_exact"]
    assert saved["schema"].shape == () and saved["arms"].shape == (10,)
    parts = [
        {name: value if name in audit.HEADERS else value[j : j + 4] for name, value in data.items()}
        for j in (0, 4, 8)
    ]
    combined = audit.combine_evaluation(parts)
    for name in data:
        np.testing.assert_array_equal(combined[name], data[name])


@pytest.mark.parametrize(
    "field",
    [
        "scores",
        "r",
        "projectors",
        "native_filters",
        "native_full_correlations",
        "cached_full_correlations",
        "q",
        "m",
        "donor_id",
        "orders",
        "packet5",
        "weights",
        "anchors",
        "query_gram",
        "native_query_mean",
    ],
)
def test_corruption_rejected(generated, field):
    data, pipeline, ids, _ = generated
    bad = dict(data)
    bad[field] = data[field].copy()
    if field == "q":
        # EEG-derived Q0 is outside this audit's starting point, and this
        # fixture's Q coefficients are zero. Corrupt its independently known mask.
        bad[field][0, 0, 0, 4] += 1
    else:
        bad[field].flat[0] += 1
    with pytest.raises((ValueError, AssertionError)):
        audit.audit_evaluation(bad, pipeline, ids, profile="generated")


@pytest.mark.parametrize("defect", ["missing", "duplicate", "header", "arms", "global", "nan"])
def test_grid_headers_and_geometry_rejected(generated, defect):
    data, pipeline, ids, _ = generated
    bad = dict(data)
    if defect == "missing":
        bad = {name: value if name in audit.HEADERS else value[:-4] for name, value in data.items()}
    elif defect == "duplicate":
        bad["keys"] = data["keys"].copy()
        bad["keys"][-1] = bad["keys"][0]
    elif defect == "header":
        bad["schema"] = np.array([audit.SCHEMA] * len(data["keys"]))
    elif defect == "arms":
        bad["arms"] = data["arms"][::-1]
    elif defect == "global":
        bad["query_mean"] = data["native_query_mean"]
    else:
        bad["scores"] = data["scores"].copy()
        bad["scores"].flat[0] = np.nan
    with pytest.raises((ValueError, AssertionError)):
        audit.audit_evaluation(bad, pipeline, ids, profile="generated")


def test_pack_and_combine_reject_ambiguous_or_object_payloads(generated):
    data, _, _, rows = generated
    bad = copy.deepcopy(rows[:1])
    bad[0]["statistics"]["samples"] = 23
    with pytest.raises(ValueError, match="samples"):
        audit.pack_evaluation(bad)
    bad = {**data, "q": data["q"].astype(object)}
    with pytest.raises(ValueError, match="numeric"):
        audit.combine_evaluation([bad])
    with pytest.raises(ValueError):
        audit.combine_evaluation([])


def test_weight_cancelling_native_correlation_corruption_is_rejected(generated):
    data, pipeline, ids, _ = generated
    bad = dict(data)
    for name in ("native_full_correlations", "cached_full_correlations", "scores"):
        bad[name] = data[name].copy()
    corr, weights = bad["native_full_correlations"][0], bad["weights"][0]
    corr[0, 0, 0] += 1e-4
    corr[0, 1, 0] -= 1e-4 * weights[0] / weights[1]
    bad["cached_full_correlations"][0] = corr
    bad["scores"][0, 0] = np.einsum("nbc,b->nc", corr, weights)
    with pytest.raises(AssertionError):
        audit.audit_evaluation(bad, pipeline, ids, profile="generated")


def test_independent_module_has_no_producer_imports():
    tree = ast.parse(inspect.getsource(audit))
    allowed = {"__future__", "hashlib", "collections.abc", "pathlib", "numpy", "cfeg.analysis"}
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            assert node.module in allowed
            if node.module == "cfeg.analysis":
                assert {v.name for v in node.names} <= {
                    "task_trca_shape_audit",
                    "task_trca_temporal_audit",
                }
        if isinstance(node, ast.Import):
            assert {v.name for v in node.names} <= allowed
