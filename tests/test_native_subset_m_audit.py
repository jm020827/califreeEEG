"""Independent auditor tests use artificial arrays/metadata and temporary artifacts only."""

import copy
import hashlib
import importlib.util
import itertools
from pathlib import Path

import numpy as np
import pytest

SPEC = importlib.util.spec_from_file_location(
    "independent_subset_audit",
    Path(__file__).resolve().parents[1] / "scripts/audit_native_subset_m_source.py",
)
AUDIT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(AUDIT)
PLAN_PATH = (
    Path(__file__).resolve().parents[1] / "configs/analysis/native_subset_m_source39_v1.json"
)


@pytest.fixture
def plan():
    return AUDIT.load_plan(PLAN_PATH)


def small_plan(plan):
    result = copy.deepcopy(plan)
    result["source_subject_ids"] = [4, 6, 8]
    for spec in result["cache"].values():
        spec["shape"][0] = 3
    result["source_projection"]["packets"] = 60
    result["source_projection"]["returned_rows"] = 720
    return result


def artificial_projection(plan):
    packets = []
    for p, subject in enumerate(plan["source_subject_ids"]):
        first = ("dry", "wet")[p % 2]
        for interface, block in itertools.product(("dry", "wet"), range(10)):
            packets.append(
                {
                    "subject_id": subject,
                    "interface": interface,
                    "block_id": block,
                    "impedance_kohm": [0.0, None, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0],
                    "headband_order": first,
                    "condition_period": "first" if interface == first else "second",
                }
            )
    return {
        "schema": "cfeg.native-subset-m.projection.v1",
        "input_path": plan["source_projection"]["path"],
        "input_sha256": plan["source_projection"]["sha256"],
        "projection": {
            "manifest_sha256": plan["source_projection"]["manifest_sha256"],
            "returned_rows": plan["source_projection"]["returned_rows"],
            "returned_packets": len(packets),
            "returned_subject_ids": plan["source_subject_ids"],
            "columns": plan["source_projection"]["columns"],
            "packets": packets,
        },
    }


def artificial_cache(plan):
    cache = {key: np.zeros(spec["shape"], np.float64) for key, spec in plan["cache"].items()}
    order = np.arange(len(plan["source_subject_ids"])) % 2
    for p, i, w, bi in itertools.product(range(len(order)), range(2), range(4), range(2)):
        k = plan["budgets"][bi]
        q = cache["q_features"][p, i, w, bi, :k]
        q[..., 0], q[..., 1], q[..., 2] = i, plan["sample_counts"][w] / 500, k / 5
        for j, block in itertools.product(range(k), range(5, 10)):
            q[j, block - 5, :, 3] = block / 9
            q[j, block - 5, :, 4] = j / 4
            q[j, block - 5, :, 5] = (block - j) / 9
        q[..., 6], q[..., 7] = order[p], int(order[p] != i)
        q[..., 13], q[..., 17], q[..., 18] = 1, 1, 1
    return cache, order


def zero_router():
    return {"feature_mean": [0.0] * 69, "feature_scale": [1.0] * 69, "coef": [0.0] * 70}


def known_zero_freezes(cache, scores, z, plan):
    """Analytic zero-target ridge solution; no fitted algorithm is called here."""
    folds = []
    for fold in range(3):
        fit = [p for p in range(len(z)) if p % 3 != fold]
        scaler = AUDIT.impedance_scaler(z, fit)
        routers = {}
        groups = []
        for p, i, k in itertools.product(fit, range(2), (3, 5)):
            maps = AUDIT.permutation_family(z[p, i, :k])
            mapping = AUDIT.select_sham(
                z[p, i, :k],
                plan["study_id"],
                fold,
                plan["source_subject_ids"][p],
                plan["interfaces"][i],
            )
            groups.append(
                {
                    "participant": plan["source_subject_ids"][p],
                    "interface": plan["interfaces"][i],
                    "k": k,
                    "admissible_maps": len(maps),
                    "control_available": bool(maps),
                    "mapping": list(mapping),
                    "packet_changes": 0,
                }
            )
        for mode in AUDIT.ROUTERS:
            x, y, weights = AUDIT.training_data(cache, scores, z, plan, scaler, fold, mode)
            assert not np.any(y)
            mean = (weights[:, None] * x).sum(0)
            scale = np.sqrt((weights[:, None] * (x - mean) ** 2).sum(0))
            effective = int(np.count_nonzero(scale >= 1e-12))
            scale[scale < 1e-12] = 1
            routers[mode] = {
                "feature_mean": mean.tolist(),
                "feature_scale": scale.tolist(),
                "coef": [0.0] * 70,
                "train_weighted_mse": 0.0,
                "ridge_penalty": 0.0,
                "objective": 0.0,
                "train_rows": len(x),
                "effective_nonconstant_features": effective,
            }
        folds.append(
            {
                "fold": fold,
                "fit_subject_ids": [plan["source_subject_ids"][p] for p in fit],
                "evaluation_subject_ids": [
                    plan["source_subject_ids"][p] for p in range(len(z)) if p % 3 == fold
                ],
                "metadata_scaler": scaler,
                "routers": routers,
                "sham_diagnostics": {
                    "support_groups": groups,
                    "feature_change_tolerance": 1e-12,
                    "omission_query_rows": len(x),
                    "changed_feature_rows": 0,
                },
            }
        )
    return {
        "schema": "cfeg.native-subset-m.freezes.v1",
        "plan_sha256": AUDIT.PLAN_SHA256,
        "folds": folds,
    }


def test_plan_hash_sensitivity(tmp_path):
    path = tmp_path / "changed-plan.json"
    path.write_bytes(PLAN_PATH.read_bytes() + b" ")
    with pytest.raises(ValueError, match="Frozen plan hash"):
        AUDIT.load_plan(path)


def test_projection_zero_missing_and_complete_identity_grid(plan):
    wrapper = artificial_projection(plan)
    z, order = AUDIT.load_projection(wrapper, plan)
    assert z.shape == (39, 2, 10, 8)
    assert np.all(z[..., 0] == 0) and np.isnan(z[..., 1]).all()
    np.testing.assert_array_equal(order, np.arange(39) % 2)


@pytest.mark.parametrize(
    "damage", ["duplicate", "subject", "order", "negative", "infinite", "boolean", "hash"]
)
def test_projection_sensitivity(plan, damage):
    wrapper = artificial_projection(plan)
    packet = wrapper["projection"]["packets"][0]
    if damage == "duplicate":
        wrapper["projection"]["packets"][-1] = packet.copy()
    elif damage == "subject":
        packet["subject_id"] = 1
    elif damage == "order":
        packet["condition_period"] = "second"
    elif damage == "hash":
        wrapper["input_sha256"] = "0" * 64
    else:
        packet["impedance_kohm"][0] = {"negative": -1.0, "infinite": np.inf, "boolean": True}[
            damage
        ]
    with pytest.raises(ValueError):
        AUDIT.load_projection(wrapper, plan)


def test_fit_only_scaler_uses_unique_packets_and_unavailable_channels():
    z = np.zeros((3, 2, 10, 8))
    z[0] = np.expm1(np.arange(10))[:, None]
    z[1] = np.expm1(np.arange(10) + 1)[:, None]
    z[:2, :, :, 7] = np.nan
    first = AUDIT.impedance_scaler(z, [0, 1])
    z[2] = 1e100
    assert AUDIT.impedance_scaler(z, [0, 1]) == first
    assert first["median"][0][0] == 5
    assert first["scale"][0][0] == 4.5
    assert first["median"][0][7] == 0 and first["scale"][0][7] == 0.1
    assert first["available"][0][7] is False


def test_cartesian_mask_family_sham_determinism_and_packet_noops():
    prefix = np.ones((5, 8))
    prefix[2:, 0] = np.nan
    maps = AUDIT.permutation_family(prefix)
    assert len(maps) == 5
    assert maps == sorted(set(maps))
    for mapping in maps:
        np.testing.assert_array_equal(np.isfinite(prefix), np.isfinite(prefix[list(mapping)]))
        assert AUDIT.changed_packets(prefix, mapping) == 0
    key = "native-subset-m-source39-v1|2|54|dry|5"
    expected = maps[int(hashlib.sha256(key.encode()).hexdigest(), 16) % 5]
    assert AUDIT.select_sham(prefix, "native-subset-m-source39-v1", 2, 54, "dry") == expected
    singleton_masks = np.full((3, 8), np.nan)
    singleton_masks[np.arange(3), np.arange(3)] = 0
    assert AUDIT.permutation_family(singleton_masks) == []


@pytest.mark.parametrize("k", [3, 5])
def test_all_m_coordinates_against_scalar_channel_oracle(k):
    rng = np.random.default_rng(128 + k)
    prefix, query = np.expm1(rng.uniform(0, 5, (k, 8))), np.expm1(rng.uniform(0, 5, (5, 8)))
    prefix[0, 0], prefix[1, 1], query[0, 2] = np.nan, np.nan, np.nan
    q = rng.normal(size=(k, 60, 33))
    scaler = {
        "median": np.ones((2, 8)).tolist(),
        "scale": np.full((2, 8), 2.0).tolist(),
        "available": np.ones((2, 8), bool).tolist(),
    }
    scaler["available"][1][7] = False
    found, fallback = AUDIT.m_features(prefix, query, q, scaler, 1)
    assert not fallback.any()
    for j, b in itertools.product(range(k), range(5)):
        delta, relative, qz, od, rd = [], [], [], [], []
        qvalid = ovalid = rvalid = 0
        for c in range(8):
            active = c != 7
            transform = lambda value: (np.log1p(value) - 1) / 2
            support_values = [
                transform(prefix[r, c]) if active and np.isfinite(prefix[r, c]) else None
                for r in range(k)
            ]
            current = transform(query[b, c]) if active and np.isfinite(query[b, c]) else None
            omit = support_values[j]
            keep = [value for r, value in enumerate(support_values) if r != j]
            usable = [value for value in keep if value is not None]
            o_distance = (omit - current) ** 2 if omit is not None and current is not None else 0.0
            r_distance = (
                sum((value - current) ** 2 for value in usable) / (k - 1)
                if current is not None
                else 0.0
            )
            difference = omit - sum(usable) / len(usable) if omit is not None and usable else 0.0
            delta.append(o_distance - r_distance)
            relative.append(difference)
            qz.append(current if current is not None else 0.0)
            od.append(o_distance)
            rd.append(r_distance)
            qvalid += current is not None
            ovalid += omit is not None
            rvalid += len(usable)
        aggregate = [sum(od) / 8, sum(rd) / 8, sum(abs(value) for value in relative) / 8]
        base = delta + relative + qz + [qvalid / 8, ovalid / 8, rvalid / ((k - 1) * 8)] + aggregate
        for row in range(b * 12, (b + 1) * 12):
            expected = base + [q[j, row, 20] * value for value in aggregate]
            expected += [q[j, row, 28] * value for value in aggregate]
            np.testing.assert_allclose(found[j, row], expected, atol=1e-13, rtol=0)


def test_full_prefix_fallback_not_omission_local_and_zero_valid():
    prefix, query = np.full((3, 8), np.nan), np.full((5, 8), np.nan)
    prefix[0, 0] = query[0, 0] = 0
    q = np.zeros((3, 60, 33))
    scaler = {
        "median": np.zeros((2, 8)).tolist(),
        "scale": np.ones((2, 8)).tolist(),
        "available": np.ones((2, 8), bool).tolist(),
    }
    values, fallback = AUDIT.m_features(prefix, query, q, scaler, 0)
    assert not fallback[:12].any() and fallback[12:].all()
    assert values[0, 0, 24] == 1 / 8 and values[0, 0, 25] == 1 / 8


def test_ridge_vs_augmented_least_squares_and_constant_slots():
    rng = np.random.default_rng(73)
    x = rng.normal(size=(213, 69))
    x[:, 42:] = 0
    x[0, 0] = 1000
    y = rng.integers(-1, 2, len(x))
    weight = rng.uniform(0.2, 2, len(x))
    weight /= weight.sum()
    actual = AUDIT.ridge_replay(x, y, weight)
    mean = (x * weight[:, None]).sum(0)
    sd = np.sqrt(((x - mean) ** 2 * weight[:, None]).sum(0))
    sd[sd < 1e-12] = 1
    design = np.column_stack([np.ones(len(x)), np.clip((x - mean) / sd, -10, 10)])
    regularizer = np.diag(np.r_[0.0, np.full(69, np.sqrt(0.1))])
    augmented = np.vstack([design * np.sqrt(weight[:, None]), regularizer])
    targets = np.r_[y * np.sqrt(weight), np.zeros(70)]
    coef = np.linalg.lstsq(augmented, targets, rcond=None)[0]
    np.testing.assert_allclose(actual["coef"], coef, atol=2e-14, rtol=0)
    assert actual["effective_nonconstant_features"] == 42
    assert actual["train_rows"] == 213


def test_route_round_full_lowest_ties_and_exact_missing_fallback():
    model = zero_router()
    model["coef"][1] = 1
    x = np.zeros((3, 5, 69))
    x[:, :, 0] = [
        [0.0, 4e-11, 0.1, -0.1, 0.2],
        [0.0, 0.0, 0.1, -0.2, 0.4],
        [0.0, 0.0, 0.1, -0.3, 0.3],
    ]
    np.testing.assert_array_equal(AUDIT.route(x, model), [0, 0, 1, 0, 2])
    missing = np.array([True, False, False, False, True])
    np.testing.assert_array_equal(
        AUDIT.route(x, model, missing, np.array([3, 3, 3, 3, 3])), [3, 0, 1, 0, 3]
    )


def test_cache_q_checks_sensitive_to_order_confidence_interactions_and_padding(plan):
    plan = small_plan(plan)
    cache, order = artificial_cache(plan)
    scores = AUDIT.cache_scores(cache, plan)
    assert AUDIT.verify_q(cache, scores, order, plan) == 0
    for feature in (6, 8, 20, 27, 30, 31, 32):
        cache["q_features"][0, 0, 0, 0, 0, 0, 0, feature] += 0.1
        with pytest.raises(ValueError, match="Q "):
            AUDIT.verify_q(cache, scores, order, plan)
        cache["q_features"][0, 0, 0, 0, 0, 0, 0, feature] -= 0.1
    cache["expert_r"][0, 0, 0, 0, 4, 0, 0, 0, 0] = 0.1
    with pytest.raises(ValueError, match="Padded experts"):
        AUDIT.cache_scores(cache, plan)


def test_training_mass_equal_units_fit_exclusion_and_all_nine_analytic_models(plan):
    plan = small_plan(plan)
    cache, _ = artificial_cache(plan)
    scores = AUDIT.cache_scores(cache, plan)
    z, _ = AUDIT.load_projection(artificial_projection(plan), plan)
    freeze = known_zero_freezes(cache, scores, z, plan)
    assert AUDIT.verify_fits(cache, scores, z, plan, freeze) == 0
    scaler = freeze["folds"][0]["metadata_scaler"]
    x, y, mass = AUDIT.training_data(cache, scores, z, plan, scaler, 0, "Q")
    assert x.shape == (7680, 69)
    assert mass.sum() == pytest.approx(1)
    offset = 0
    for _p, _i, _w, k in itertools.product(range(2), range(2), range(4), (3, 5)):
        assert mass[offset : offset + k * 60].sum() == pytest.approx(1 / 32)
        offset += k * 60
    cache["q_features"][0] += 99
    unchanged, unchanged_y, unchanged_mass = AUDIT.training_data(
        cache, scores, z, plan, scaler, 0, "Q"
    )
    np.testing.assert_array_equal(x, unchanged)
    np.testing.assert_array_equal(y, unchanged_y)
    np.testing.assert_array_equal(mass, unchanged_mass)


@pytest.mark.parametrize("damage", ["scaler", "coefficient", "mean", "fit_id", "sham_coverage"])
def test_fit_replay_sensitivity(plan, damage):
    plan = small_plan(plan)
    cache, _ = artificial_cache(plan)
    scores = AUDIT.cache_scores(cache, plan)
    z, _ = AUDIT.load_projection(artificial_projection(plan), plan)
    freeze = known_zero_freezes(cache, scores, z, plan)
    fold = freeze["folds"][0]
    if damage == "scaler":
        fold["metadata_scaler"]["median"][0][0] += 0.2
    elif damage == "coefficient":
        fold["routers"]["Q"]["coef"][1] += 0.2
    elif damage == "mean":
        fold["routers"]["Q"]["feature_mean"][1] += 0.2
    elif damage == "fit_id":
        fold["fit_subject_ids"][0] = 1
    else:
        fold["sham_diagnostics"]["changed_feature_rows"] += 1
    with pytest.raises(ValueError):
        AUDIT.verify_fits(cache, scores, z, plan, freeze)


def test_routes_grid_controls_noops_missing_and_reporting(plan):
    plan = small_plan(plan)
    cache, _ = artificial_cache(plan)
    scores = AUDIT.cache_scores(cache, plan)
    z, _ = AUDIT.load_projection(artificial_projection(plan), plan)
    # Every support block has a different finite mask, so no legal map exists.
    for b in range(5):
        z[:, :, b] = np.nan
        z[:, :, b, b] = 0
    freeze = {
        "folds": [
            {
                "metadata_scaler": AUDIT.impedance_scaler(z, [1, 2]),
                "routers": {mode: zero_router() for mode in AUDIT.ROUTERS},
            }
        ]
        * 3
    }
    rows, diagnostics = AUDIT.replay_evaluation(cache, scores, z, plan, freeze)
    assert len(rows) == 360 and len(diagnostics) == 48
    assert all(row["ba"] == pytest.approx(1 / 12) for row in rows)
    for row in rows:
        if row["method"] in ("M_SHUFFLE", "SHAM_REFIT"):
            assert row["control_available"] is False and row["intervention_count"] == 0
    assert all(
        item["shuffle_interventions"][0]["control_available"] is False for item in diagnostics
    )
    assert all(item["mean_shuffle_feature_changes"] == 0 for item in diagnostics)
    summary, contrasts, attainment = AUDIT.replay_reporting(rows, plan)
    assert (len(summary), len(contrasts), len(attainment)) == (120, 153, 96)
    assert all(item["mean"] == 0 for item in contrasts)
    assert all(item["unreached_tested_grid"] for item in attainment)
    changed = copy.deepcopy(rows)
    changed[0]["ba"] += 1 / 60
    with pytest.raises(ValueError, match="rows"):
        AUDIT.equal(changed, rows, "rows")
    changed_summary = copy.deepcopy(summary)
    changed_summary[0]["ci95_low"] -= 0.01
    with pytest.raises(ValueError):
        AUDIT.equal(changed_summary, summary, "summary")


def test_first_hit_is_observed_grid_not_monotonic_or_minimum_claim(plan):
    plan = small_plan(plan)
    rows = []
    for p, interface, n in itertools.product(
        plan["source_subject_ids"], plan["interfaces"], plan["sample_counts"]
    ):
        for method, k in [("A0_author", 0)] + [
            (method, k) for method in plan["controls"]["methods"][1:] for k in (3, 5)
        ]:
            rows.append(
                {
                    "participant": p,
                    "interface": interface,
                    "n_samples": n,
                    "method": method,
                    "k": k,
                    "ba": 0.9 if k == 3 else 0.4,
                }
            )
    _, _, attainment = AUDIT.replay_reporting(rows, plan)
    assert all(item["first_observed_k"] == 3 and item["label_count"] == 36 for item in attainment)


def test_inventory_permission_and_binding_sensitivity(tmp_path, plan):
    root = tmp_path / "artifacts"
    root.mkdir()
    with pytest.raises(ValueError, match="five-artifact"):
        AUDIT.audit(root, PLAN_PATH)
    for name in plan["execution"]["artifacts"]:
        (root / name).write_text("{}")
    with pytest.raises(ValueError, match="Immutable regular"):
        AUDIT.audit(root, PLAN_PATH)


def provenance_fixture(root, plan):
    # Synthetic-only destination contract; no call to the pinned load_plan audit.
    plan["execution"]["output_root"] = str(root)
    common = {
        "plan_sha256": AUDIT.PLAN_SHA256,
        "study_id": plan["study_id"],
        "upstream_revision": plan["upstream"]["revision"],
        "source_commit": "a" * 40,
        "source_tree": "b" * 40,
        "started_at": "2026-09-07T12:00:00+00:00",
    }
    hashes = {
        name: hashlib.sha256(name.encode()).hexdigest() for name in plan["execution"]["artifacts"]
    }
    start = {
        **common,
        "schema": "cfeg.native-subset-m.start.v1",
        "core_sha256": hashlib.sha256(b"artificial core blob").hexdigest(),
        "imported_source_hashes": plan["upstream"]["pins"].copy(),
        "source_subject_ids": plan["source_subject_ids"],
        "raw_root": plan["raw_root"],
        "output_root": str(root),
        "python": plan["execution"]["python_version"],
        "dependencies": plan["execution"]["dependencies"].copy(),
        "metadata_access": True,
        "source_projection_reuse": True,
        "baseline_cache_read": True,
        "held_access": False,
        "retired_access": False,
        "manifest_or_full_impedance_access": False,
    }
    freeze = {
        **common,
        "source_projection_sha256": hashes["source-projection.json"],
        "features_sha256": hashes["features.npz"],
    }
    result = {
        **freeze,
        "schema": plan["reporting"]["schema"],
        "status": "DEVELOPMENT_ASSESSMENT_COMPLETE",
        "completed_at": "2026-09-07T12:00:01+00:00",
        "fold_freezes_sha256": hashes["fold-freezes.json"],
        "raw_files": [
            {
                "subject": subject,
                "path": f"{plan['raw_root']}/S{subject:03d}.mat",
                "sha256": "c" * 64,
                "stored_dtype": "float32",
                "shape": plan["raw_shape"],
            }
            for subject in plan["source_subject_ids"]
        ],
        "baseline_agreement": {
            "input_path": plan["baseline_reference"]["path"],
            "input_sha256": plan["baseline_reference"]["sha256"],
            "a0_r": {"max_abs_error": 0.0, "argmax_exact": True},
            "full_expert_r": {"max_abs_error": 0.0, "argmax_exact": True},
        },
    }
    return {"start.json": start, "fold-freezes.json": freeze, "result.json": result}, hashes


def git_object_stub(command, **kwargs):
    assert command[:3] == ["git", "-C", "/synthetic/repository"]
    if "rev-parse" in command:
        assert kwargs == {"text": True}
        return "b" * 40 + "\n"
    if command[-1].endswith(":configs/analysis/native_subset_m_source39_v1.json"):
        return PLAN_PATH.read_bytes()
    assert command[-1].endswith(":scripts/native_subset_m_core.py")
    return b"artificial core blob"


def test_provenance_matches_saved_git_objects_and_all_hash_bindings(monkeypatch, tmp_path, plan):
    payloads, hashes = provenance_fixture(tmp_path, plan)
    monkeypatch.setattr(AUDIT.subprocess, "check_output", git_object_stub)
    AUDIT.verify_provenance(tmp_path, plan, payloads, hashes, Path("/synthetic/repository"))


@pytest.mark.parametrize(
    "damage",
    [
        "cache_hash",
        "projection_hash",
        "freeze_hash",
        "git_tree",
        "core_blob",
        "native_pin",
        "protected",
        "raw_path",
        "raw_id",
        "timestamps",
        "baseline",
        "destination",
    ],
)
def test_provenance_binding_corruption_rejected(monkeypatch, tmp_path, plan, damage):
    payloads, hashes = provenance_fixture(tmp_path, plan)
    start, freeze, result = (
        payloads[name] for name in ("start.json", "fold-freezes.json", "result.json")
    )
    monkeypatch.setattr(AUDIT.subprocess, "check_output", git_object_stub)
    if damage == "cache_hash":
        result["features_sha256"] = "0" * 64
    elif damage == "projection_hash":
        freeze["source_projection_sha256"] = "0" * 64
    elif damage == "freeze_hash":
        result["fold_freezes_sha256"] = "0" * 64
    elif damage == "git_tree":
        for item in (start, freeze, result):
            item["source_tree"] = "d" * 40
    elif damage == "core_blob":
        start["core_sha256"] = "0" * 64
    elif damage == "native_pin":
        start["imported_source_hashes"][next(iter(plan["upstream"]["pins"]))] = "0" * 64
    elif damage == "protected":
        start["held_access"] = True
    elif damage == "raw_path":
        result["raw_files"][0]["path"] = "/forbidden/S001.mat"
    elif damage == "raw_id":
        result["raw_files"][0]["subject"] = 1
    elif damage == "timestamps":
        result["completed_at"] = "2026-09-07T11:59:59+00:00"
    elif damage == "destination":
        plan["execution"]["output_root"] = "/different/frozen/destination"
    else:
        result["baseline_agreement"]["a0_r"]["argmax_exact"] = False
    with pytest.raises(ValueError):
        AUDIT.verify_provenance(tmp_path, plan, payloads, hashes, Path("/synthetic/repository"))


@pytest.mark.parametrize(
    "actual,expected",
    [(True, 1), (1, True), ([1], [1, 2]), ({"x": 1, "y": 2}, {"x": 1}), (float("nan"), 0.0)],
)
def test_structural_comparison_catches_types_extras_and_nan(actual, expected):
    with pytest.raises(ValueError):
        AUDIT.equal(actual, expected, "damaged")


def test_implementation_imports_no_producer_core_toolbox_or_raw_loader():
    import ast

    tree = ast.parse(Path(AUDIT.__file__).read_text())
    imports = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.extend(item.name for item in node.names)
        elif isinstance(node, ast.ImportFrom):
            imports.append(node.module)
    assert not any(
        any(
            fragment in name
            for fragment in ("native_subset_m_core", "run_native", "SSVEP", "scipy.io", "cfeg")
        )
        for name in imports
    )
