"""Artificial arrays only: metadata alignment, ridge, routing, and reporting."""

import copy
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "subset_core", ROOT / "scripts/native_subset_m_core.py"
)
CORE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(CORE)
PLAN = json.loads((ROOT / "configs/analysis/native_subset_m_source39_v1.json").read_text())


@pytest.fixture(scope="module")
def projection():
    packets = []
    for p, subject in enumerate(PLAN["source_subject_ids"]):
        for i, interface in enumerate(("dry", "wet")):
            for b in range(10):
                packets.append(
                    {
                        "subject_id": subject,
                        "interface": interface,
                        "block_id": b,
                        "impedance_kohm": [float(p + b + c) for c in range(8)],
                        "headband_order": ("dry", "wet")[p % 2],
                        "condition_period": "first" if p % 2 == i else "second",
                    }
                )
    return {
        "manifest_sha256": PLAN["source_projection"]["manifest_sha256"],
        "packets": packets,
        "returned_rows": 9360,
        "returned_packets": 780,
        "returned_subject_ids": PLAN["source_subject_ids"],
        "columns": PLAN["source_projection"]["columns"],
    }


def test_projection_preserves_zero_and_native_alignment(projection):
    z, order = CORE.projection_arrays(projection, PLAN)
    assert z.shape == (39, 2, 10, 8) and z[0, 0, 0, 0] == 0
    assert np.array_equal(order, np.arange(39) % 2)
    changed = copy.deepcopy(projection)
    changed["packets"][0]["impedance_kohm"][0] = None
    assert np.isnan(CORE.projection_arrays(changed, PLAN)[0][0, 0, 0, 0])


@pytest.mark.parametrize(
    "damage",
    ["extra", "held", "duplicate", "order", "period", "negative", "infinite", "shape", "hash"],
)
def test_projection_rejects_corruption(projection, damage):
    value = copy.deepcopy(projection)
    first = value["packets"][0]
    if damage == "extra":
        first["query_label"] = 1
    elif damage == "held":
        first["subject_id"] = 1
    elif damage == "duplicate":
        value["packets"][1] = first
    elif damage == "order":
        first["headband_order"] = "unknown"
    elif damage == "period":
        first["condition_period"] = "second"
    elif damage == "negative":
        first["impedance_kohm"][0] = -1
    elif damage == "infinite":
        first["impedance_kohm"][0] = np.inf
    elif damage == "shape":
        first["impedance_kohm"] = [0]
    else:
        value["manifest_sha256"] = "0" * 64
    with pytest.raises(ValueError):
        CORE.projection_arrays(value, PLAN)


def test_scaler_uses_only_unique_fit_packets(projection):
    z, _ = CORE.projection_arrays(projection, PLAN)
    fit = [p for p in range(39) if p % 3 != 0]
    before = CORE.fit_metadata_scaler(z, fit)
    changed = z.copy()
    changed[::3] = 1e100
    assert CORE.fit_metadata_scaler(changed, fit) == before
    expected = np.log1p(z[fit, 0, :, 0].ravel())
    assert before["median"][0][0] == np.median(expected)
    assert before["scale"][0][0] == max(
        np.percentile(expected, 75) - np.percentile(expected, 25), 0.1
    )


def test_scaler_missing_and_constant_floor():
    z = np.zeros((3, 2, 10, 8))
    z[..., 1] = np.nan
    result = CORE.fit_metadata_scaler(z, [0, 1])
    assert result["available"][0][0] and not result["available"][0][1]
    assert result["scale"][0] == [0.1] * 8


@pytest.mark.parametrize("k,expected", [(3, 2), (5, 4)])
def test_mask_group_rotation_family(k, expected):
    prefix = np.arange(k * 8, dtype=float).reshape(k, 8)
    maps = CORE.support_maps(prefix)
    assert len(maps) == expected
    assert all(sorted(m) == list(range(k)) and m != tuple(range(k)) for m in maps)
    assert CORE.sham_map(prefix, PLAN, 0, 4, "dry", k) in maps
    assert CORE.sham_map(prefix, PLAN, 0, 4, "dry", k) == CORE.sham_map(
        prefix, PLAN, 0, 4, "dry", k
    )


def test_cartesian_mask_groups_and_singletons():
    prefix = np.ones((5, 8))
    prefix[:2, 0] = np.nan
    maps = CORE.support_maps(prefix)
    assert len(maps) == 5  # 2*3 rotations less the global identity
    for mapping in maps:
        assert np.array_equal(np.isfinite(prefix), np.isfinite(prefix[list(mapping)]))
    singleton = np.eye(5, 8)
    singleton[singleton == 0] = np.nan
    assert CORE.support_maps(singleton) == []


def identity_scaler():
    return {"median": [[0.0] * 8] * 2, "scale": [[1.0] * 8] * 2, "available": [[True] * 8] * 2}


def test_metadata_denominators_masks_and_interactions():
    prefix = np.full((3, 8), np.nan)
    prefix[:, 0] = np.expm1([1.0, 2.0, 3.0])
    prefix[2, 0] = np.nan
    query = np.full((5, 8), np.nan)
    query[:, 0] = 0
    q = np.zeros((3, 60, 33))
    q[..., 20] = 2
    q[..., 28] = -3
    extra, fallback = CORE.metadata_features(prefix, query, q, identity_scaler(), 0)
    assert not fallback.any()
    # Omit0: squared distance1, retained (4+missing0)/2=2, delta=-1.
    assert extra[0, 0, 0] == pytest.approx(-1)
    assert extra[0, 0, 8] == pytest.approx(-1)  # valid retained mean z=2, NOT /2
    assert extra[0, 0, 27:30] == pytest.approx([1 / 8, 2 / 8, 1 / 8])
    assert extra[0, 0, 30:33] == pytest.approx([2 / 8, 4 / 8, 2 / 8])
    assert extra[0, 0, 33:36] == pytest.approx([-3 / 8, -6 / 8, -3 / 8])
    # Removing the sole valid matching block does not change prefix-level fallback.
    prefix[1:] = np.nan
    assert not CORE.metadata_features(prefix, query, q, identity_scaler(), 0)[1].any()
    query[:] = np.nan
    assert CORE.metadata_features(prefix, query, q, identity_scaler(), 0)[1].all()


def test_relative_z_does_not_require_query_channel():
    prefix = np.expm1(np.tile(np.arange(3.0)[:, None], (1, 8)))
    query = np.zeros((5, 8))
    query[:, 1] = np.nan
    extra, _ = CORE.metadata_features(prefix, query, np.zeros((3, 60, 33)), identity_scaler(), 0)
    assert extra[0, 0, 1] == 0  # masked squared distance
    assert extra[0, 0, 9] == pytest.approx(-1.5)  # omission-retained relative z stays valid


def test_ridge_matches_independent_augmented_least_squares():
    rng = np.random.default_rng(7191)
    x = rng.normal(size=(180, 69))
    x[:, 33:] = 0
    x[:, 0] = 1
    y = rng.integers(-1, 2, size=180).astype(float)
    weights = np.linspace(1, 3, 180)
    weights /= weights.sum()
    model = CORE.fit_router(x, y, weights)
    mean, scale = np.asarray(model["feature_mean"]), np.asarray(model["feature_scale"])
    a = np.column_stack([np.ones(len(x)), np.clip((x - mean) / scale, -10, 10)])
    regularizer = np.diag([0] + [np.sqrt(0.1)] * 69)
    expected = np.linalg.lstsq(
        np.vstack([a * np.sqrt(weights[:, None]), regularizer]),
        np.concatenate([y * np.sqrt(weights), np.zeros(70)]),
        rcond=None,
    )[0]
    assert np.allclose(model["coef"], expected, atol=1e-12, rtol=1e-12)
    assert np.max(np.abs(np.asarray(model["coef"])[34:])) < 1e-14
    assert model["effective_nonconstant_features"] == 32
    assert model["objective"] == pytest.approx(model["train_weighted_mse"] + model["ridge_penalty"])


@pytest.mark.parametrize("damage", ["nan", "weight", "mass", "lambda"])
def test_ridge_rejects_unfrozen_or_invalid_inputs(damage):
    x, y, weights = np.zeros((3, 69)), np.zeros(3), np.ones(3) / 3
    ridge = 0.1
    if damage == "nan":
        x[0, 0] = np.nan
    elif damage == "weight":
        weights[0] = 0
    elif damage == "mass":
        weights *= 2
    else:
        ridge = 1
    with pytest.raises(ValueError):
        CORE.fit_router(x, y, weights, ridge)


def test_routing_full_wins_nonpositive_and_omission_ties():
    gains = np.array([[0, -0.2, 0.3, 0.2], [0, -0.1, 0.3, 0.4], [-0.1, -0.4, 0.1, 0.4]])
    assert CORE.choose(gains).tolist() == [0, 0, 1, 2]


def test_sham_coverage_distinguishes_unavailable_and_numeric_noop():
    z = np.ones((1, 2, 10, 8))
    z[0, 0, :5] = np.where(np.eye(5, 8), 1.0, np.nan)
    report = CORE.fit_sham_diagnostics(z, [0], PLAN, 0)
    dry, wet = report["support_groups"][:2], report["support_groups"][2:]
    assert all(not x["control_available"] and x["packet_changes"] == 0 for x in dry)
    assert all(x["control_available"] and x["packet_changes"] == 0 for x in wet)
    a = np.zeros((3, 60, 69))
    b = a.copy()
    b[..., 0] = 100  # Only M columns count.
    b[0, 0, 33] = 1e-13
    assert CORE.feature_changes(a, b) == 0
    b[1, 5, 68] = 1e-5
    assert CORE.feature_changes(a, b) == 1


def test_training_mass_target_alignment_and_excluded_person_independence(monkeypatch):
    rng = np.random.default_rng(412)
    scores = {"expert_r": rng.normal(size=(3, 2, 4, 2, 6, 5, 12, 12))}
    z = np.ones((3, 2, 10, 8))

    def inputs(cache, z, plan, scaler, p, i, w, bi, mode, fold):
        x = np.zeros((PLAN["budgets"][bi], 60, 69))
        x[..., 0] = p
        return x, np.zeros(60, dtype=bool)

    monkeypatch.setattr(CORE, "inputs_for_cell", inputs)
    x, y, weights = CORE.training_examples({}, z, scores, PLAN, {}, [0, 1], "Q", 2)
    assert len(y) == 7680 and weights.sum() == pytest.approx(1)
    cursor = 0
    for p in (0, 1):
        assert weights[x[:, 0] == p].sum() == pytest.approx(0.5)
        for i in range(2):
            for w in range(4):
                for bi, k in enumerate((3, 5)):
                    pred = (
                        scores["expert_r"][p, i, w, bi, : k + 1].reshape(k + 1, 60, 12).argmax(-1)
                    )
                    correct = pred == np.tile(np.arange(12), 5)
                    sl = slice(cursor, cursor + k * 60)
                    assert np.array_equal(y[sl], (correct[1:].astype(float) - correct[0]).ravel())
                    assert weights[sl].sum() == pytest.approx(1 / 32)
                    cursor += k * 60
    scores["expert_r"][2] = np.nan
    changed = CORE.training_examples({}, z, scores, PLAN, {}, [0, 1], "Q", 2)
    assert all(np.array_equal(a, b) for a, b in zip((x, y, weights), changed))


def test_evaluation_keeps_unavailable_sham_and_exact_q_missing(monkeypatch, projection):
    projection = copy.deepcopy(projection)
    for packet in projection["packets"]:
        b = packet["block_id"]
        packet["impedance_kohm"] = (
            [1.0 if c == b else None for c in range(8)] if b < 5 else [None] * 8
        )
    monkeypatch.setattr(CORE, "validate_cache", lambda cache, plan: None)
    scores = {
        "a0_r": np.zeros((39, 2, 4, 5, 12, 12)),
        "expert_r": np.zeros((39, 2, 4, 2, 6, 5, 12, 12)),
    }
    monkeypatch.setattr(CORE, "native_scores", lambda cache, plan: scores)
    cache = {"q_features": np.zeros((39, 2, 4, 2, 5, 5, 12, 33))}
    router = {"feature_mean": [0.0] * 69, "feature_scale": [1.0] * 69, "coef": [0.0] * 70}
    freezes = {"plan_sha256": CORE.PLAN_SHA256, "folds": []}
    for fold in range(3):
        freezes["folds"].append(
            {
                "fold": fold,
                "fit_subject_ids": [
                    v for p, v in enumerate(PLAN["source_subject_ids"]) if p % 3 != fold
                ],
                "evaluation_subject_ids": [
                    v for p, v in enumerate(PLAN["source_subject_ids"]) if p % 3 == fold
                ],
                "metadata_scaler": identity_scaler(),
                "routers": {mode: router for mode in CORE.MODES},
            }
        )
    result = CORE.evaluate_all(cache, projection, freezes, PLAN)
    assert len(result["rows"]) == 4680
    assert all(
        not r["control_available"] and r["intervention_count"] == 0
        for r in result["rows"]
        if r["method"] in ("SHAM_REFIT", "M_SHUFFLE")
    )
    for d in result["diagnostics"]:
        assert d["qm_missing_fallback_queries"] == 60
        assert d["actions"]["QM"] == d["actions"]["Q"]
        assert d["mean_shuffle_feature_changes"] == d["mean_shuffle_prediction_changes"] == 0
        assert d["shuffle_interventions"][0]["mapping"] == list(range(d["k"]))


def test_reporting_complete_grid_first_hit_and_nonmonotonicity():
    rows = []
    for p in PLAN["source_subject_ids"]:
        for interface in ("dry", "wet"):
            for n in PLAN["sample_counts"]:
                rows.append(
                    {
                        "participant": p,
                        "interface": interface,
                        "n_samples": n,
                        "method": "A0_author",
                        "k": 0,
                        "ba": 0.2,
                    }
                )
                for method in PLAN["controls"]["methods"][1:]:
                    for k in (3, 5):
                        rows.append(
                            {
                                "participant": p,
                                "interface": interface,
                                "n_samples": n,
                                "method": method,
                                "k": k,
                                "ba": 0.8 if k == 3 else 0.5,
                            }
                        )
    summary, contrasts, attainment = CORE.reporting(rows, PLAN)
    assert (
        len(rows) == 4680
        and len(summary) == 120
        and len(contrasts) == 153
        and len(attainment) == 1248
    )
    assert all(x["first_observed_k"] == 3 and x["label_count"] == 36 for x in attainment)
    assert all(x["history_seconds"] == pytest.approx(5.04) for x in attainment)
    primary = next(
        x for x in contrasts if x["contrast"] == "QM3_minus_Q3" and x["interface"] == "all8"
    )
    assert primary["mean"] == 0 and primary["ci95_low"] == primary["ci95_high"] == 0
