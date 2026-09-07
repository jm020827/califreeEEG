"""Artificial-only sensitivity tests; no human cache, labels or output access."""

import copy
import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]


def load(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / (name + ".py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


RULE = load("native_subset_known_zero")
OLD = load("native_subset_m_core")


def toy():
    predictions = np.array([[0, 0, 0], [0, 0, 0], [1, 1, 1], [2, 2, 2]])
    gains = np.tile(np.array([0.3, 0.2, -0.1])[:, None], (1, 3))
    return gains, predictions


def test_closed_parent_pins_and_artificial_scope():
    path = ROOT / "configs/analysis/native_subset_known_zero_v1.json"
    assert hashlib.sha256(path.read_bytes()).hexdigest() == (
        "9f3ed2309b9dc2efff81460a91c49e131722595ef1ff8de4b24f9291f66c9e55"
    )
    plan = json.loads(path.read_text())
    for prefix in ("science", "core"):
        parent_path = ROOT / plan["parent"][prefix + "_path"]
        assert (
            hashlib.sha256(parent_path.read_bytes()).hexdigest()
            == plan["parent"][prefix + "_sha256"]
        )
    assert not plan["authority"]["human_outcome_execution"]
    assert plan["budget"]["additional_scientific_variants"] == 0


def test_exact_identity_for_all_twelve_possible_truths():
    _, pred = toy()
    for label in range(12):
        delta = (pred[1:] == label).astype(int) - (pred[0] == label).astype(int)
        assert np.all(delta[pred[1:] == pred[0]] == 0)


def test_old_linear_ridge_does_not_enforce_identity():
    # Same-answer rows have the CORRECT zero training target. Regularization
    # still leaves nonzero alias predictions; this is not a target-label bug.
    x = np.zeros((4, 69))
    x[:2, 17] = 1.0
    fit = OLD.fit_router(x, np.array([0.0, 0.0, 1.0, 1.0]), np.full(4, 0.25))
    gain = OLD.predict_gain(x, fit)
    assert np.all(gain[:2] > 0)
    np.testing.assert_allclose(gain[:2], 0.0454545455, atol=1e-12, rtol=0)


def test_sensitive_parent_failure_and_corrected_choice_can_help_or_harm():
    gains, pred = toy()
    old_action = OLD.choose(gains)
    action = RULE.choose_known_zero(gains, pred)
    assert old_action.tolist() == [1, 1, 1]  # Parent violates the new constraint.
    assert action.tolist() == [2, 2, 2]
    truth = np.array([1, 0, 2])
    before, after = pred[old_action, np.arange(3)], pred[action, np.arange(3)]
    change = (after == truth).astype(int) - (before == truth).astype(int)
    assert change.tolist() == [1, -1, 0]  # Repair, harm, wrong-to-wrong.


@pytest.mark.parametrize("k", [3, 5])
def test_randomized_branch_invariants_idempotence_and_query_locality(k):
    rng = np.random.default_rng(840 + k)
    pred = rng.integers(0, 12, (k + 1, 200))
    pred[1, ::2] = pred[0, ::2]
    gains = np.round(rng.normal(size=(k, 200)), 10)
    before_gains, before_pred = gains.copy(), pred.copy()
    projected = RULE.project_gains(gains, pred)
    alias = pred[1:] == pred[0]
    assert np.all(projected[alias] == 0)
    assert np.array_equal(projected[~alias], gains[~alias])
    assert np.array_equal(projected, RULE.project_gains(projected, pred))
    assert not np.shares_memory(projected, gains)
    old, new = OLD.choose(gains), RULE.choose_known_zero(gains, pred)
    for q in range(pred.shape[1]):
        if old[q] == 0 or pred[old[q], q] != pred[0, q]:
            assert new[q] == old[q]
        elif np.any((~alias[:, q]) & (gains[:, q] > 0)):
            assert new[q] > 0 and pred[new[q], q] != pred[0, q]
        else:
            assert new[q] == 0
        assert RULE.choose_known_zero(gains[:, q : q + 1], pred[:, q : q + 1])[0] == new[q]
    perm = rng.permutation(pred.shape[1])
    assert np.array_equal(RULE.choose_known_zero(gains[:, perm], pred[:, perm]), new[perm])
    assert np.array_equal(gains, before_gains) and np.array_equal(pred, before_pred)


def test_full_negative_and_positive_ties_and_preexisting_quantization():
    pred = np.array([[0] * 5, [0] * 5, [1] * 5, [2] * 5])
    gains = np.round(
        np.array(
            [
                [0.3, -0.3, 0.3, 0.3, 0.3],
                [0, -0.2, 0.2, 4e-11, 6e-11],
                [-0.1, -0.4, 0.2, -0.1, -0.1],
            ]
        ),
        10,
    )
    assert RULE.choose_known_zero(gains, pred).tolist() == [0, 0, 2, 0, 2]


def test_no_alias_identity_and_all_alias_full():
    gains, pred = toy()
    pred[1] = 3
    assert np.array_equal(RULE.choose_known_zero(gains, pred), OLD.choose(gains))
    pred[:] = 0
    assert not RULE.choose_known_zero(gains, pred).any()


def family():
    gains, pred = toy()
    by_mode = {mode: gains.copy() for mode in RULE.MODES}
    fallback = {mode: np.array([False, True, False]) for mode in RULE.M_MODES}
    # Without projection this candidate never beats the spurious alias; after
    # projection M routes3, while fallback must use the NEW Q route2 (not old1).
    for mode in RULE.M_MODES:
        by_mode[mode][2] = 0.25
    return (
        pred,
        by_mode,
        fallback,
        [by_mode["QM"].copy()] * 2,
        [np.array([True, False, False]), np.array([False, True, True])],
    )


def test_fair_family_each_shuffle_and_missing_uses_corrected_q():
    args = family()
    before = copy.deepcopy(args)
    output = RULE.route_family(*args)
    assert set(output) == {"actions", "shuffle_actions"}
    actions = output["actions"]
    assert actions["Q"].tolist() == [2, 2, 2]
    for mode in RULE.M_MODES:
        assert actions[mode].tolist() == [3, 2, 3]
    assert np.array_equal(actions["M_MISSING"], actions["Q"])
    assert not np.shares_memory(actions["M_MISSING"], actions["Q"])
    assert [v.tolist() for v in output["shuffle_actions"]] == [[2, 3, 3], [3, 2, 2]]
    np.testing.assert_array_equal(args[0], before[0])
    for j in (1, 2):
        for key in args[j]:
            np.testing.assert_array_equal(args[j][key], before[j][key])
    for j in (3, 4):
        for actual, expected in zip(args[j], before[j]):
            np.testing.assert_array_equal(actual, expected)


def test_unavailable_shuffle_keeps_one_identity_evaluation():
    pred, gains, fallback, _, _ = family()
    result = RULE.route_family(pred, gains, fallback, [gains["QM"]], [fallback["QM"]])
    assert len(result["shuffle_actions"]) == 1
    np.testing.assert_array_equal(result["shuffle_actions"][0], result["actions"]["QM"])


@pytest.mark.parametrize(
    "damage",
    [
        "nan_alias",
        "inf",
        "float32",
        "integer_gain",
        "list",
        "unrounded",
        "k2",
        "empty",
        "pred_float",
        "pred_bool",
        "pred_shape",
        "pred_negative",
        "pred_high",
    ],
)
def test_rejects_invalid_or_unrounded_inputs(damage):
    gains, pred = toy()
    if damage == "nan_alias":
        gains[0, 0] = np.nan
    elif damage == "inf":
        gains[1, 0] = np.inf
    elif damage == "float32":
        gains = gains.astype(np.float32)
    elif damage == "integer_gain":
        gains = gains.astype(int)
    elif damage == "list":
        gains = gains.tolist()
    elif damage == "unrounded":
        gains[1, 0] = 4e-11
    elif damage == "k2":
        gains, pred = gains[:2], pred[:3]
    elif damage == "empty":
        gains, pred = gains[:, :0], pred[:, :0]
    elif damage == "pred_float":
        pred = pred.astype(float)
    elif damage == "pred_bool":
        pred = pred.astype(bool)
    elif damage == "pred_shape":
        pred = pred[:-1]
    elif damage == "pred_negative":
        pred[0, 0] = -1
    elif damage == "pred_high":
        pred[0, 0] = 12
    with pytest.raises(ValueError):
        RULE.choose_known_zero(gains, pred)


@pytest.mark.parametrize(
    "damage",
    [
        "extra_gain",
        "missing_gain",
        "extra_mask",
        "mask_dtype",
        "mask_shape",
        "no_shuffle",
        "mismatch",
        "invalid_hidden",
    ],
)
def test_rejects_family_boundary_errors(damage):
    pred, gains, masks, shuffle, shuffle_masks = family()
    if damage == "extra_gain":
        gains["M_MISSING"] = gains["Q"]
    elif damage == "missing_gain":
        del gains["SHAM_REFIT"]
    elif damage == "extra_mask":
        masks["Q"] = np.zeros(3, dtype=bool)
    elif damage == "mask_dtype":
        masks["QM"] = masks["QM"].astype(int)
    elif damage == "mask_shape":
        masks["M_STALE"] = np.zeros(2, dtype=bool)
    elif damage == "no_shuffle":
        shuffle, shuffle_masks = [], []
    elif damage == "mismatch":
        shuffle_masks = shuffle_masks[:1]
    elif damage == "invalid_hidden":
        masks["QM"][:] = True
        gains["QM"][0, 0] = np.nan
    with pytest.raises(ValueError):
        RULE.route_family(pred, gains, masks, shuffle, shuffle_masks)
