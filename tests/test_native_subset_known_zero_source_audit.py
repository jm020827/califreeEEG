"""Synthetic-only independent revised-policy and artifact receipt checks."""

import ast
import copy
import hashlib
import importlib.util
import itertools
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "revised_source_auditor", ROOT / "scripts/audit_native_subset_known_zero_source.py"
)
AUDIT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(AUDIT)


@pytest.fixture
def plan():
    return AUDIT.load_plan(ROOT / AUDIT.PLAN_RELATIVE)


@pytest.fixture
def modules(plan):
    return [
        AUDIT.load_definition(ROOT, relative, plan["pinned_helpers"][relative])
        for relative in AUDIT.AUDIT_DEFINITIONS
    ]


def small_parent(plan, helper):
    """Three artificial people, all native experts tied; analytically zero-target fit."""
    science = json.loads((ROOT / plan["science"]["path"]).read_bytes())
    science["source_subject_ids"] = [4, 6, 8]
    for spec in science["cache"].values():
        spec["shape"][0] = 3
    science["source_projection"]["returned_rows"] = 720
    science["source_projection"]["packets"] = 60
    cache = {
        name: np.zeros(spec["shape"], dtype=np.float64) for name, spec in science["cache"].items()
    }
    packets = []
    for p, participant in enumerate(science["source_subject_ids"]):
        for i, interface in enumerate(science["interfaces"]):
            for block in range(10):
                packets.append(
                    {
                        "subject_id": participant,
                        "interface": interface,
                        "block_id": block,
                        "impedance_kohm": [0.0, None, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0],
                        "headband_order": science["interfaces"][p % 2],
                        "condition_period": "first" if p % 2 == i else "second",
                    }
                )
            for w, n in enumerate(science["sample_counts"]):
                for bi, k in enumerate(science["budgets"]):
                    q = cache["q_features"][p, i, w, bi, :k]
                    q[..., 0], q[..., 1], q[..., 2] = i, n / 500, k / 5
                    q[..., 6], q[..., 7] = p % 2, int(p % 2 != i)
                    q[..., 13], q[..., 17], q[..., 18] = 1, 1, 1
                    for j, block in itertools.product(range(k), range(5, 10)):
                        q[j, block - 5, :, 3] = block / 9
                        q[j, block - 5, :, 4] = j / 4
                        q[j, block - 5, :, 5] = (block - j) / 9
    content = {
        "manifest_sha256": science["source_projection"]["manifest_sha256"],
        "packets": packets,
        "returned_rows": 720,
        "returned_packets": 60,
        "returned_subject_ids": [4, 6, 8],
        "columns": science["source_projection"]["columns"].copy(),
    }
    z, _ = helper.load_projection(
        {
            "schema": "cfeg.native-subset-m.projection.v1",
            "input_path": science["source_projection"]["path"],
            "input_sha256": science["source_projection"]["sha256"],
            "projection": content,
        },
        science,
    )
    scores = helper.cache_scores(cache, science)
    freezes = {
        "schema": "cfeg.native-subset-m.freezes.v1",
        "plan_sha256": plan["science"]["sha256"],
        "folds": [],
    }
    for fold in range(3):
        fit = [p for p in range(3) if p != fold]
        scaler = helper.impedance_scaler(z, fit)
        routers, groups = {}, []
        for p, i, k in itertools.product(fit, range(2), (3, 5)):
            maps = helper.permutation_family(z[p, i, :k])
            mapping = helper.select_sham(
                z[p, i, :k], science["study_id"], fold, [4, 6, 8][p], science["interfaces"][i]
            )
            groups.append(
                {
                    "participant": [4, 6, 8][p],
                    "interface": science["interfaces"][i],
                    "k": k,
                    "admissible_maps": len(maps),
                    "control_available": bool(maps),
                    "mapping": list(mapping),
                    "packet_changes": 0,
                }
            )
        for mode in ("Q", "QM", "SHAM_REFIT"):
            x, y, mass = helper.training_data(cache, scores, z, science, scaler, fold, mode)
            assert not y.any()
            mean = np.sum(mass[:, None] * x, axis=0)
            sd = np.sqrt(np.sum(mass[:, None] * (x - mean) ** 2, axis=0))
            effective = int(np.count_nonzero(sd >= 1e-12))
            sd[sd < 1e-12] = 1
            routers[mode] = {
                "feature_mean": mean.tolist(),
                "feature_scale": sd.tolist(),
                "coef": [0.0] * 70,
                "train_weighted_mse": 0.0,
                "ridge_penalty": 0.0,
                "objective": 0.0,
                "train_rows": len(x),
                "effective_nonconstant_features": effective,
            }
        freezes["folds"].append(
            {
                "fold": fold,
                "fit_subject_ids": [[4, 6, 8][p] for p in fit],
                "evaluation_subject_ids": [[4, 6, 8][fold]],
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
    rows, diagnostics = helper.replay_evaluation(cache, scores, z, science, freezes)
    summary, contrasts, attainment = helper.replay_reporting(rows, science)
    parent = {
        "rows": rows,
        "diagnostics": diagnostics,
        "summary": summary,
        "contrasts": contrasts,
        "attainment": attainment,
        "evidence_scope": "adaptively exposed development, descriptive intervals, no automatic promotion",
    }
    return cache, content, freezes, parent, science


def test_frozen_plan_and_import_allowlist(plan, tmp_path):
    damaged = tmp_path / "plan.json"
    damaged.write_bytes((ROOT / AUDIT.PLAN_RELATIVE).read_bytes() + b" ")
    with pytest.raises(ValueError, match="execution plan hash"):
        AUDIT.load_plan(damaged)
    for forbidden in (
        "scripts/native_subset_m_core.py",
        "scripts/native_subset_known_zero.py",
        "scripts/native_subset_known_zero_io.py",
        "scripts/run_native_subset_known_zero_source.py",
    ):
        with pytest.raises(ValueError, match="Only three"):
            AUDIT.load_definition(ROOT, forbidden, "0" * 64)
    with pytest.raises(ValueError, match="definition hash"):
        AUDIT.load_definition(ROOT, AUDIT.AUDIT_DEFINITIONS[0], "0" * 64)


def test_gains_are_original_normalization_clipping_and_ten_decimal_output():
    x = np.zeros((3, 12, 69))
    x[0, :, 0] = 1000
    x[1, :, 0] = -1000
    x[2, :, 0] = 1.12345678901
    model = {
        "feature_mean": [0.0] * 69,
        "feature_scale": [1.0] * 69,
        "coef": [0.05, 0.1] + [0.0] * 68,
    }
    expected = np.round(np.array([[1.05] * 12, [-0.95] * 12, [0.162345678901] * 12]), 10)
    np.testing.assert_array_equal(AUDIT.rounded_gains(x, model), expected)


@pytest.mark.parametrize("outcome", ["help", "harm", "wrong_wrong"])
def test_revision_metrics_do_not_assume_monotonic_improvement(outcome):
    truth = np.tile(np.arange(12), 5)
    full = truth if outcome == "harm" else (truth + 1) % 12
    candidate = truth if outcome == "help" else (truth + 2) % 12
    predictions = np.stack([full, full, candidate, (truth + 3) % 12])
    gains = np.tile(np.array([[0.9], [0.5], [-0.2]]), (1, 60))
    old, new = np.ones(60, int), np.full(60, 2)
    actual = AUDIT.revision_metrics(predictions, gains, old, new, old, new)
    assert actual["old_alias_selected"] == 60 and actual["old_full_selected"] == 0
    assert actual["old_nonalias_selected"] == 0 and actual["projected_gain_entries_changed"] == 60
    assert actual["prefallback_action_changes"] == actual["action_changes"] == 60
    assert actual["prefallback_prediction_changes"] == actual["prediction_changes"] == 60
    assert actual["repaired_previous"] == (60 if outcome == "help" else 0)
    assert actual["damaged_previous"] == (60 if outcome == "harm" else 0)


def test_revision_metrics_count_postfallback_categories_separately_from_prefallback():
    truth = np.tile(np.arange(12), 5)
    predictions = np.stack([truth, truth, (truth + 1) % 12, (truth + 2) % 12])
    gains = np.tile(np.array([[0.9], [0.5], [-0.2]]), (1, 60))
    before, revised_before = np.ones(60, int), np.full(60, 2)
    old_final = np.repeat([0, 1, 2], 20)
    new_final = old_final.copy()
    actual = AUDIT.revision_metrics(
        predictions, gains, before, revised_before, old_final, new_final
    )
    assert (
        actual["old_full_selected"],
        actual["old_alias_selected"],
        actual["old_nonalias_selected"],
    ) == (20, 20, 20)
    assert actual["prefallback_action_changes"] == actual["prefallback_prediction_changes"] == 60
    assert actual["action_changes"] == actual["prediction_changes"] == 0
    assert actual["old_ba"] == actual["new_ba"] == pytest.approx(2 / 3)


def test_zero_gain_alias_not_counted_as_changed_but_negative_alias_is():
    predictions = np.zeros((4, 12), dtype=int)
    gains = np.tile(np.array([[0.0], [-0.2], [0.3]]), (1, 12))
    before, after = np.full(12, 3), np.zeros(12, int)
    actual = AUDIT.revision_metrics(predictions, gains, before, after, before, after)
    assert actual["projected_gain_entries_changed"] == 24
    assert actual["action_changes"] == 12 and actual["prediction_changes"] == 0


def test_complete_independent_analytic_parent_and_revision_replay(plan, modules):
    helper, _, oracle = modules
    fixture = small_parent(plan, helper)
    cache, _content, freezes, parent, science = fixture
    original_parent, original_freezes = copy.deepcopy(parent), copy.deepcopy(freezes)
    result = AUDIT.independent_replay(*fixture, helper, oracle)
    assert result["parent_replay_verified"] is True
    for key in AUDIT.SCIENTIFIC_FIELDS:
        helper.equal(result[key], parent[key], key)
    assert (
        len(result["rows"]),
        len(result["summary"]),
        len(result["contrasts"]),
        len(result["diagnostics"]),
        len(result["attainment"]),
        len(result["revision_diagnostics"]),
    ) == (360, 120, 153, 48, 96, 48)
    metric_fields = set(plan["revision_diagnostics"]["metric_fields"])
    for item in result["revision_diagnostics"]:
        assert set(item) == set(plan["revision_diagnostics"]["keys"])
        assert set(item["methods"]) == set(plan["revision_diagnostics"]["methods"])
        for metric in item["methods"].values():
            assert set(metric) == metric_fields
            assert metric["old_full_selected"] == 60 and metric["prediction_changes"] == 0
        for metric in item["shuffle"]:
            assert set(metric) == metric_fields | {"mapping"}
            assert metric["action_changes"] == 0
    helper.equal(parent, original_parent, "Parent input immutable")
    helper.equal(freezes, original_freezes, "Policy freezes immutable")
    assert cache["expert_r"].shape[0] == len(science["source_subject_ids"]) == 3


@pytest.mark.parametrize("field", ["rows", "summary", "contrasts", "diagnostics", "attainment"])
def test_parent_corruption_stops_before_any_revised_routing(plan, modules, field):
    helper, _, _ = modules
    fixture = small_parent(plan, helper)
    parent = fixture[3]
    parent[field] = parent[field][:-1]

    def forbidden_revision(*args, **kwargs):
        raise AssertionError("Revised selection must not run before verified parent")

    protocol = SimpleNamespace(route_family_scalar=forbidden_revision)
    with pytest.raises(ValueError, match="Unchanged parent"):
        AUDIT.independent_replay(*fixture, helper, protocol)


def test_fit_corruption_rejected_before_revision(plan, modules):
    helper, _, oracle = modules
    fixture = small_parent(plan, helper)
    fixture[2]["folds"][0]["routers"]["Q"]["coef"][1] = 0.01
    with pytest.raises(ValueError, match="Ridge"):
        AUDIT.independent_replay(*fixture, helper, oracle)


def write_artificial_bundle(root, values):
    root.mkdir()
    hashes = {}
    for name, raw in values.items():
        path = root / name
        path.write_bytes(raw)
        path.chmod(0o400)
        hashes[name] = hashlib.sha256(raw).hexdigest()
    return hashes


def test_bundle_exact_bytes_inventory_modes_and_hash_sensitivity(tmp_path):
    root = tmp_path / "artifacts"
    values = {"start.json": b"{}", "result.json": b'{"artificial":true}'}
    hashes = write_artificial_bundle(root, values)
    found, actual, size = AUDIT.read_bundle(root, list(values), 1000, hashes)
    assert found == values and actual == hashes and size == sum(map(len, values.values()))
    bad = {**hashes, "result.json": "0" * 64}
    with pytest.raises(ValueError, match="Pinned parent artifact hash"):
        AUDIT.read_bundle(root, list(values), 1000, bad)
    with pytest.raises(ValueError, match="byte budget"):
        AUDIT.read_bundle(root, list(values), 2, hashes)
    (root / "result.json").chmod(0o600)
    with pytest.raises(ValueError, match="Immutable"):
        AUDIT.read_bundle(root, list(values), 1000)
    (root / "result.json").chmod(0o400)
    (root / "extra.json").write_bytes(b"{}")
    with pytest.raises(ValueError, match="inventory"):
        AUDIT.read_bundle(root, list(values), 1000)


@pytest.mark.parametrize("link", ["symlink", "hardlink"])
def test_artifact_link_aliases_rejected(tmp_path, link):
    root = tmp_path / "artifact-links"
    root.mkdir()
    original = tmp_path / "original.bin"
    original.write_bytes(b"artificial")
    original.chmod(0o400)
    target = root / "start.json"
    if link == "symlink":
        target.symlink_to(original)
    else:
        target.hardlink_to(original)
    with pytest.raises(ValueError, match="Immutable"):
        AUDIT.read_bundle(root, ["start.json"], 1000)


def receipt_fixture(tmp_path, plan):
    repository = tmp_path / "synthetic-repository"
    repository.mkdir()
    source_hashes = {}
    for relative in {
        *plan["pinned_helpers"],
        *AUDIT.configuration_hashes(plan),
        *plan["code_paths"],
    }:
        path = repository / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        raw = (
            b"# Artificial new execution definition\n"
            if relative in plan["code_paths"]
            else (ROOT / relative).read_bytes()
        )
        path.write_bytes(raw)
        source_hashes[relative] = hashlib.sha256(raw).hexdigest()
    common = {
        "attempt_id": plan["attempt_id"],
        "study_id": plan["study_id"],
        "randomization_namespace": plan["randomization_namespace"],
        "execution_plan_sha256": AUDIT.PLAN_SHA256,
        "mechanism_plan_sha256": plan["mechanism"]["sha256"],
        "scientific_plan_sha256": plan["science"]["sha256"],
        "source_commit": "a" * 40,
        "source_tree": "b" * 40,
        "started_at": "2026-09-08T01:00:00+00:00",
        "parent_root": plan["parent"]["root"],
        "parent_artifact_sha256": plan["parent"]["artifacts"].copy(),
        "source_hashes": source_hashes,
        "runtime": plan["runtime"].copy(),
        "output_root": plan["output_root"],
        "policy_refit": False,
        "held_access": False,
        "retired_access": False,
        "raw_eeg_access": False,
    }
    start = {"schema": plan["schemas"]["start"], **common}
    start_hash = hashlib.sha256(json.dumps(start).encode()).hexdigest()
    result = {
        "schema": plan["schemas"]["result"],
        **copy.deepcopy(common),
        "completed_at": "2026-09-08T01:00:01+00:00",
        "start_sha256": start_hash,
        "status": "DEVELOPMENT_ASSESSMENT_COMPLETE",
        "parent_replay_verified": True,
        "rows": [],
        "summary": [],
        "contrasts": [],
        "diagnostics": [],
        "attainment": [],
        "revision_diagnostics": [],
        "evidence_scope": "artificial receipt only",
    }

    def fake_git(repo, *args):
        assert Path(repo) == repository
        if args[0] == "rev-parse":
            assert args[1] == "a" * 40 + "^{tree}"
            return ("b" * 40).encode()
        assert args[0] == "show" and args[1].startswith("a" * 40 + ":")
        return (repository / args[1].split(":", 1)[1]).read_bytes()

    return (
        repository,
        {"start.json": start, "result.json": result},
        {"start.json": start_hash},
        fake_git,
    )


def test_new_receipts_and_exact_twelve_source_bindings(plan, modules, tmp_path, monkeypatch):
    helper = modules[0]
    repository, payloads, hashes, fake_git = receipt_fixture(tmp_path, plan)
    assert len(payloads["start.json"]["source_hashes"]) == 12
    monkeypatch.setattr(AUDIT, "git_bytes", fake_git)
    AUDIT.verify_new_provenance(
        Path(plan["output_root"]), plan, payloads, hashes, repository, helper
    )


@pytest.mark.parametrize(
    "damage",
    [
        "extra",
        "missing",
        "namespace",
        "policy_refit",
        "held",
        "raw",
        "parenthash",
        "starthash",
        "runtime",
        "source_hash",
        "source_extra",
        "tree",
        "time",
        "destination",
        "notverified",
    ],
)
def test_new_receipt_provenance_tampering_rejected(plan, modules, tmp_path, monkeypatch, damage):
    helper = modules[0]
    repository, payloads, hashes, fake_git = receipt_fixture(tmp_path, plan)
    monkeypatch.setattr(AUDIT, "git_bytes", fake_git)
    start, result = payloads["start.json"], payloads["result.json"]
    output = Path(plan["output_root"])
    if damage == "extra":
        result["extra"] = 0
    elif damage == "missing":
        del start["runtime"]
    elif damage == "namespace":
        start["randomization_namespace"] = result["randomization_namespace"] = plan["study_id"]
    elif damage in ("policy_refit", "held", "raw"):
        field = {"policy_refit": "policy_refit", "held": "held_access", "raw": "raw_eeg_access"}[
            damage
        ]
        start[field] = result[field] = True
    elif damage == "parenthash":
        start["parent_artifact_sha256"]["result.json"] = result["parent_artifact_sha256"][
            "result.json"
        ] = "0" * 64
    elif damage == "starthash":
        result["start_sha256"] = "0" * 64
    elif damage == "runtime":
        start["runtime"]["cpu_workers"] = result["runtime"]["cpu_workers"] = 4
    elif damage == "source_hash":
        relative = next(iter(plan["pinned_helpers"]))
        start["source_hashes"][relative] = result["source_hashes"][relative] = "0" * 64
    elif damage == "source_extra":
        start["source_hashes"]["unexpected.py"] = result["source_hashes"]["unexpected.py"] = (
            "0" * 64
        )
    elif damage == "tree":
        start["source_tree"] = result["source_tree"] = "c" * 40
    elif damage == "time":
        result["completed_at"] = "2026-09-08T00:59:59+00:00"
    elif damage == "destination":
        output = Path("/synthetic/copied-artifact-destination")
        start["output_root"] = result["output_root"] = str(output)
    else:
        result["parent_replay_verified"] = False
    with pytest.raises(ValueError):
        AUDIT.verify_new_provenance(output, plan, payloads, hashes, repository, helper)


def test_wrong_root_cli_fails_before_parent_or_artifact_access(plan):
    with pytest.raises(ValueError, match="Exact revised artifact root"):
        AUDIT.audit("/synthetic/wrong-output", ROOT / AUDIT.PLAN_RELATIVE, ROOT)


def test_no_production_or_io_import_and_no_old_entrypoint_call():
    tree = ast.parse(Path(AUDIT.__file__).read_text())
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            assert node.func.attr not in ("evaluate_all", "fit_all", "run", "main", "audit")
        if isinstance(node, ast.ImportFrom):
            assert node.module not in (
                "native_subset_m_core",
                "native_subset_known_zero_io",
                "native_subset_known_zero",
            )
