"""Artificial-only scalar oracle checks; no production/core or outcome imports."""

import ast
import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "known_zero_scalar_oracle", ROOT / "scripts/audit_native_subset_known_zero.py"
)
ORACLE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(ORACLE)


def family_fixture(k=3, queries=5):
    predictions = np.tile(np.arange(k + 1, dtype=np.int64)[:, None], (1, queries))
    predictions[1] = predictions[0]
    gains = np.full((k, queries), 0.1)
    gains[0], gains[1] = 0.9, 0.5
    modes = {mode: gains.copy() for mode in ORACLE.MODES}
    fallbacks = {mode: np.zeros(queries, dtype=bool) for mode in ORACLE.FALLBACK_MODES}
    return predictions, modes, fallbacks, [gains.copy()], [np.zeros(queries, dtype=bool)]


def old_action(gains):
    return np.vstack([np.zeros(gains.shape[1]), gains]).argmax(axis=0)


def test_frozen_contract_hash_and_no_execution_authority():
    raw = (ROOT / "configs/analysis/native_subset_known_zero_v1.json").read_bytes()
    assert hashlib.sha256(raw).hexdigest() == ORACLE.PLAN_SHA256
    plan = json.loads(raw)
    assert plan["authority"]["human_outcome_execution"] is False
    assert plan["api"]["mode_keys"] == list(ORACLE.MODES)
    assert plan["api"]["fallback_keys"] == list(ORACLE.FALLBACK_MODES)


@pytest.mark.parametrize("full_class", range(12))
def test_alias_exact_zero_for_all_possible_truth_labels(full_class):
    # Exhaust all possible truth labels outside the oracle API. The identity is
    # algebraic, not an estimated quality statement or access to hidden labels.
    for truth in range(12):
        alias_gain = int(full_class == truth) - int(full_class == truth)
        assert alias_gain == 0
    predictions = np.full((4, 1), full_class, dtype=np.int64)
    gains = np.array([[0.99], [0.5], [-0.4]], dtype=np.float64)
    np.testing.assert_array_equal(ORACLE.choose_known_zero_scalar(gains, predictions), [0])


@pytest.mark.parametrize("k", [3, 5])
def test_strict_positive_first_index_ties_and_no_nonfull_deduplication(k):
    predictions = np.zeros((k + 1, 4), dtype=np.int64)
    predictions[2:, :] = 2
    gains = np.full((k, 4), -0.3)
    gains[0] = [0.9, -0.1, 0.9, 0.9]
    gains[1] = [0.5, -0.2, 0.0, 0.2]
    gains[2] = [0.5, -0.4, -0.1, 0.7]
    # Experts2 and3 predict the same NON-FULL class. They remain distinct, so
    # expert3 wins the last query and the lower index wins the first exact tie.
    np.testing.assert_array_equal(ORACLE.choose_known_zero_scalar(gains, predictions), [2, 0, 0, 3])


def test_repair_harm_and_wrong_to_wrong_are_all_possible():
    predictions = np.array([[1, 1, 1], [1, 1, 1], [2, 2, 2], [3, 3, 3]], dtype=np.int64)
    gains = np.array([[0.9, 0.9, 0.9], [0.5, 0.5, 0.5], [0.1, 0.1, 0.1]])
    before = predictions[old_action(gains), np.arange(3)]
    after = predictions[ORACLE.choose_known_zero_scalar(gains, predictions), np.arange(3)]
    synthetic_truth = np.array([2, 1, 7])
    np.testing.assert_array_equal(
        (after == synthetic_truth).astype(int) - (before == synthetic_truth), [1, -1, 0]
    )
    np.testing.assert_array_equal(before, [1, 1, 1])
    np.testing.assert_array_equal(after, [2, 2, 2])


@pytest.mark.parametrize("k", [3, 5])
def test_only_old_alias_winners_can_change_and_new_class_requires_positive_nonalias(k):
    rng = np.random.default_rng(581 + k)
    queries = 127
    predictions = rng.integers(0, 4, (k + 1, queries), dtype=np.int64)
    gains = np.round(rng.uniform(-0.9, 0.9, (k, queries)), 10)
    saved_gains, saved_predictions = gains.copy(), predictions.copy()
    original = old_action(gains)
    corrected = ORACLE.choose_known_zero_scalar(gains, predictions)
    for q in range(queries):
        old = int(original[q])
        old_alias = old > 0 and predictions[old, q] == predictions[0, q]
        if not old_alias:
            assert corrected[q] == old
        if predictions[corrected[q], q] != predictions[old, q]:
            assert old_alias
            assert corrected[q] > 0 and gains[corrected[q] - 1, q] > 0
            assert predictions[corrected[q], q] != predictions[0, q]
        choices = [(0.0, 0)] + [
            (float(gains[j - 1, q]), -j)
            for j in range(1, k + 1)
            if predictions[j, q] != predictions[0, q]
        ]
        assert corrected[q] == -max(choices)[1]
    np.testing.assert_array_equal(gains, saved_gains)
    np.testing.assert_array_equal(predictions, saved_predictions)
    aliases = predictions[1:] == predictions[0]
    projected = gains.copy()
    projected[aliases] = 0.0
    np.testing.assert_array_equal(
        ORACLE.choose_known_zero_scalar(projected, predictions), corrected
    )
    permutation = rng.permutation(queries)
    np.testing.assert_array_equal(
        ORACLE.choose_known_zero_scalar(gains[:, permutation], predictions[:, permutation]),
        corrected[permutation],
    )


@pytest.mark.parametrize("k", [3, 5])
def test_every_mode_and_shuffle_uses_rule_then_corrected_q_fallback(k):
    predictions, modes, fallbacks, shuffles, shuffle_fallbacks = family_fixture(k)
    # Corrected Q is expert2. Other arms prefer expert3 but keep a larger alias
    # gain that must be ignored; fallback must never reuse old Q expert1.
    for mode in ORACLE.FALLBACK_MODES:
        modes[mode][2] = 0.7
    fallbacks["QM"][0] = True
    fallbacks["SHAM_REFIT"][1] = True
    fallbacks["M_STALE"][2] = True
    shuffles[0][2] = 0.7
    shuffle_fallbacks[0][3] = True
    shuffles.append(modes["Q"].copy())
    shuffle_fallbacks.append(np.zeros(5, dtype=bool))
    snapshots = {mode: gains.copy() for mode, gains in modes.items()}
    original_fallbacks = {mode: mask.copy() for mode, mask in fallbacks.items()}
    outcome = ORACLE.route_family_scalar(predictions, modes, fallbacks, shuffles, shuffle_fallbacks)
    assert set(outcome) == {"actions", "shuffle_actions"}
    assert set(outcome["actions"]) == {*ORACLE.MODES, "M_MISSING"}
    np.testing.assert_array_equal(outcome["actions"]["Q"], [2] * 5)
    for mode, index in zip(ORACLE.FALLBACK_MODES, (0, 1, 2)):
        expected = np.full(5, 3)
        expected[index] = 2
        np.testing.assert_array_equal(outcome["actions"][mode], expected)
    np.testing.assert_array_equal(outcome["shuffle_actions"][0], [3, 3, 3, 2, 3])
    np.testing.assert_array_equal(outcome["shuffle_actions"][1], outcome["actions"]["Q"])
    np.testing.assert_array_equal(outcome["actions"]["M_MISSING"], outcome["actions"]["Q"])
    assert not np.shares_memory(outcome["actions"]["Q"], outcome["actions"]["M_MISSING"])
    for mode in ORACLE.MODES:
        np.testing.assert_array_equal(modes[mode], snapshots[mode])
    for mode in ORACLE.FALLBACK_MODES:
        np.testing.assert_array_equal(fallbacks[mode], original_fallbacks[mode])


def test_identical_modes_and_identity_shuffle_remain_fair_even_when_all_fallback():
    predictions, modes, fallbacks, shuffles, shuffle_fallbacks = family_fixture(3, 1)
    for mask in fallbacks.values():
        mask[:] = True
    shuffle_fallbacks[0][:] = True
    result = ORACLE.route_family_scalar(
        predictions, modes, fallbacks, tuple(shuffles), tuple(shuffle_fallbacks)
    )
    for values in [*result["actions"].values(), *result["shuffle_actions"]]:
        np.testing.assert_array_equal(values, [2])


@pytest.mark.parametrize(
    "bad_gains",
    [
        [[0.1], [0.2], [0.3]],
        np.zeros((3, 1), dtype=np.float32),
        np.zeros((3, 1), dtype=int),
        np.zeros((3, 1), dtype=complex),
        np.full((3, 1), np.nan),
        np.full((3, 1), np.inf),
        np.zeros((3,)),
        np.zeros((3, 0)),
        np.zeros((5, 1)),
        np.full((3, 1), 0.12345678901),
    ],
)
def test_invalid_gain_array_dtype_shape_finite_and_rounding_rejected(bad_gains):
    with pytest.raises(ValueError):
        ORACLE.choose_known_zero_scalar(bad_gains, np.zeros((4, 1), dtype=np.int64))


@pytest.mark.parametrize(
    "bad_predictions",
    [
        [[0], [1], [2], [3]],
        np.zeros((4, 1), dtype=bool),
        np.zeros((4, 1), dtype=float),
        np.zeros((4, 1), dtype=complex),
        np.zeros((4, 1), dtype=object),
        np.full((4, 1), -1),
        np.full((4, 1), 12),
        np.zeros((4,), dtype=int),
        np.zeros((3, 1), dtype=int),
        np.zeros((4, 0), dtype=int),
    ],
)
def test_invalid_prediction_dtype_shape_and_class_rejected(bad_predictions):
    with pytest.raises(ValueError):
        ORACLE.choose_known_zero_scalar(np.zeros((3, 1)), bad_predictions)


@pytest.mark.parametrize("dtype", [np.int8, np.uint8, np.int32, np.int64])
def test_valid_integer_prediction_types_and_strided_inputs(dtype):
    predictions, modes, _, _, _ = family_fixture(3, 6)
    np.testing.assert_array_equal(
        ORACLE.choose_known_zero_scalar(modes["Q"][:, ::2], predictions.astype(dtype)[:, ::2]),
        [2] * 3,
    )


@pytest.mark.parametrize(
    "damage",
    [
        "missing_mode",
        "extra_mode",
        "gains_not_dict",
        "missing_fallback",
        "extra_fallback",
        "fallback_not_dict",
        "fallback_float",
        "fallback_list",
        "fallback_shape",
        "empty_shuffles",
        "different_shuffle_count",
        "shuffle_array",
        "shuffle_fallback_array",
        "invalid_shuffle_gain",
        "invalid_shuffle_mask",
        "unrounded_mode",
    ],
)
def test_family_rejects_invalid_modes_masks_and_intervention_contract(damage):
    predictions, modes, fallbacks, shuffles, shuffle_fallbacks = family_fixture()
    if damage == "missing_mode":
        del modes["M_STALE"]
    elif damage == "extra_mode":
        modes["OTHER"] = modes["Q"]
    elif damage == "gains_not_dict":
        modes = list(modes.values())
    elif damage == "missing_fallback":
        del fallbacks["QM"]
    elif damage == "extra_fallback":
        fallbacks["Q"] = np.zeros(5, dtype=bool)
    elif damage == "fallback_not_dict":
        fallbacks = list(fallbacks.values())
    elif damage == "fallback_float":
        fallbacks["QM"] = np.zeros(5)
    elif damage == "fallback_list":
        fallbacks["QM"] = [False] * 5
    elif damage == "fallback_shape":
        fallbacks["QM"] = np.zeros((1, 5), dtype=bool)
    elif damage == "empty_shuffles":
        shuffles, shuffle_fallbacks = [], []
    elif damage == "different_shuffle_count":
        shuffle_fallbacks.append(np.zeros(5, dtype=bool))
    elif damage == "shuffle_array":
        shuffles = np.stack(shuffles)
    elif damage == "shuffle_fallback_array":
        shuffle_fallbacks = np.stack(shuffle_fallbacks)
    elif damage == "invalid_shuffle_gain":
        shuffles[0] = np.zeros((3, 5), dtype=np.float32)
    elif damage == "invalid_shuffle_mask":
        shuffle_fallbacks[0] = np.zeros(5, dtype=int)
    else:
        modes["QM"][0, 0] = 0.12345678901
    with pytest.raises(ValueError):
        ORACLE.route_family_scalar(predictions, modes, fallbacks, shuffles, shuffle_fallbacks)


def test_no_truth_fitting_production_import_or_runtime_entrypoint():
    import inspect

    assert list(inspect.signature(ORACLE.choose_known_zero_scalar).parameters) == [
        "rounded_gains",
        "expert_predictions",
    ]
    assert list(inspect.signature(ORACLE.route_family_scalar).parameters) == [
        "expert_predictions",
        "gains_by_mode",
        "fallback_by_mode",
        "shuffled_gains",
        "shuffled_fallbacks",
    ]
    tree = ast.parse(Path(ORACLE.__file__).read_text())
    assert not any(
        isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name in ("main", "run", "audit")
        for node in ast.walk(tree)
    )
    imports = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.extend(name.name for name in node.names)
        elif isinstance(node, ast.ImportFrom):
            imports.append(node.module)
    assert set(imports) == {"__future__", "numpy"}
