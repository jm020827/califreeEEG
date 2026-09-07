"""Artificial cache→parent replay→revised evaluator integration, no human I/O."""

import copy
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]


def load(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / (name + ".py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


EVALUATOR = load("run_native_subset_known_zero_source")


def synthetic_parent(*, fitted=True):
    """Return evaluator kwargs; default genuinely fits nine artificial routers.

    Shapes/protocol/Q confidence and interaction fields pass the parent's cache
    auditor. These arrays are deliberately not an EEG extraction reproduction.
    fitted=False supplies transparent sensitive coefficients, not an audited fit.
    No metadata/data/artifact path is opened; only tracked Python/JSON code.
    """
    core, rule = load("native_subset_m_core"), load("native_subset_known_zero")
    science = json.loads((ROOT / "configs/analysis/native_subset_m_source39_v1.json").read_text())
    cache = {
        name: np.zeros(spec["shape"], dtype=np.float64) for name, spec in science["cache"].items()
    }
    # Same answer but differing native-score confidence for FULL and omit0.
    cache["a0_r"][..., 0] = 0.55
    for bi, k in enumerate((3, 5)):
        for expert in range(k + 1):
            prediction = 0 if expert < 2 else (1 if expert == 2 else 2)
            confidence = (0.6, 0.8, 0.7)[expert] if expert < 3 else 0.4
            cache["expert_r"][:, :, :, bi, expert, ..., prediction] = confidence
    packets = []
    for p, subject in enumerate(science["source_subject_ids"]):
        for i, interface in enumerate(science["interfaces"]):
            for b in range(10):
                values = [float(p + b + c) / 10 for c in range(8)]
                if p == 0 and b == 7:
                    values = [None] * 8
                if p == 1 and b < 5:
                    values = [1.0 if c == b else None for c in range(8)]
                packets.append(
                    {
                        "subject_id": subject,
                        "interface": interface,
                        "block_id": b,
                        "impedance_kohm": values,
                        "headband_order": science["interfaces"][p % 2],
                        "condition_period": "first" if i == p % 2 else "second",
                    }
                )
    projection = {
        "manifest_sha256": science["source_projection"]["manifest_sha256"],
        "packets": packets,
        "returned_rows": 9360,
        "returned_packets": 780,
        "returned_subject_ids": science["source_subject_ids"],
        "columns": science["source_projection"]["columns"],
    }
    scores = core.native_scores(cache, science)
    for p in range(39):
        for i, interface in enumerate(science["interfaces"]):
            for w, n in enumerate(science["sample_counts"]):
                a0 = scores["a0_r"][p, i, w].reshape(60, 12) / sum(
                    science["native_weights"]["A0_author"][interface]
                )
                a0sort = np.sort(a0, axis=-1)
                for bi, k in enumerate((3, 5)):
                    experts = scores["expert_r"][p, i, w, bi, : k + 1].reshape(k + 1, 60, 12) / sum(
                        science["native_weights"]["ETRCA"][interface]
                    )
                    full = experts[0]
                    fullsort = np.sort(full, axis=-1)
                    for j in range(k):
                        q = cache["q_features"][p, i, w, bi, j].reshape(60, 33)
                        candidate = experts[j + 1]
                        ordered = np.sort(candidate, axis=-1)
                        q[:, 0], q[:, 1], q[:, 2] = i, n / 500, k / 5
                        q[:, 3], q[:, 4] = np.repeat(np.arange(5, 10), 12) / 9, j / 4
                        q[:, 5], q[:, 6], q[:, 7] = (
                            (np.repeat(np.arange(5, 10), 12) - j) / 9,
                            p % 2,
                            i != p % 2,
                        )
                        q[:, 8], q[:, 9], q[:, 10] = (
                            fullsort[:, -1],
                            fullsort[:, -1] - fullsort[:, -2],
                            full.std(-1),
                        )
                        q[:, 11], q[:, 12] = a0sort[:, -1], a0sort[:, -1] - a0sort[:, -2]
                        q[:, 13] = full.argmax(-1) == a0.argmax(-1)
                        q[:, 14], q[:, 15] = ordered[:, -1], ordered[:, -1] - ordered[:, -2]
                        q[:, 16] = np.sqrt(np.mean((candidate - full) ** 2, axis=-1))
                        q[:, 17], q[:, 18] = (
                            candidate.argmax(-1) == full.argmax(-1),
                            candidate.argmax(-1) == a0.argmax(-1),
                        )
                        q[:, 19], q[:, 20] = q[:, 14] - q[:, 8], q[:, 15] - q[:, 9]
                        # Artificial signal-quality variation, not real class labels
                        # entering an inference API. This makes genuine fits nontrivial.
                        q[:, 25] = np.tile(np.arange(12), 5) / 12
                        q[:, 27] = q[:, 25]
                        q[:, 30] = q[:, 20] ** 2
    if fitted:
        freezes = core.fit_all(cache, projection, science)
    else:
        z, _ = core.projection_arrays(projection, science)
        freezes = {"plan_sha256": core.PLAN_SHA256, "folds": []}
        for fold in range(3):
            fit = [p for p in range(39) if p % 3 != fold]
            router = {"feature_mean": [0.0] * 69, "feature_scale": [1.0] * 69, "coef": [0.0] * 70}
            router["coef"][0], router["coef"][15] = -0.5, 1.0
            freezes["folds"].append(
                {
                    "fold": fold,
                    "fit_subject_ids": [science["source_subject_ids"][p] for p in fit],
                    "evaluation_subject_ids": [
                        s for p, s in enumerate(science["source_subject_ids"]) if p % 3 == fold
                    ],
                    "metadata_scaler": core.fit_metadata_scaler(z, fit),
                    "routers": {mode: copy.deepcopy(router) for mode in core.MODES},
                }
            )
    parent = core.evaluate_all(cache, projection, freezes, science)
    return {
        "cache": cache,
        "content_projection": projection,
        "freezes": freezes,
        "parent_result": parent,
        "science": science,
        "core": core,
        "rule": rule,
    }


@pytest.fixture(scope="module")
def sensitive():
    return synthetic_parent(fitted=False)


@pytest.fixture(scope="module")
def revised(sensitive):
    return EVALUATOR.evaluate(**sensitive)


def test_sensitive_real_helper_to_route_and_complete_reporting(sensitive, revised):
    assert revised["parent_replay_verified"]
    for key, count in {
        "rows": 4680,
        "summary": 120,
        "contrasts": 153,
        "diagnostics": 624,
        "attainment": 1248,
        "revision_diagnostics": 624,
    }.items():
        assert len(revised[key]) == count
    first_old = sensitive["parent_result"]["diagnostics"][0]
    first_new = revised["diagnostics"][0]
    assert first_old["actions"]["Q"] == [1] * 60
    assert first_new["actions"]["Q"] == [2] * 60
    for row in revised["revision_diagnostics"]:
        for mode in ("Q", "QM", "SHAM_REFIT", "M_STALE", "M_MISSING"):
            metric = row["methods"][mode]
            assert metric["old_alias_selected"] == 60
            assert metric["action_changes"] == metric["prediction_changes"] == 60
            assert metric["repaired_previous"] == metric["damaged_previous"] == 5
            assert metric["old_ba"] == metric["new_ba"] == 1 / 12
        for record in row["shuffle"]:
            assert record["prediction_changes"] == 60
    json.dumps(revised, allow_nan=False)


def test_schema_missing_alias_availability_and_unchanged_anchors(sensitive, revised):
    plan = json.loads(
        (ROOT / "configs/analysis/native_subset_known_zero_source39_v1.json").read_text()
    )
    revision_spec = plan["revision_diagnostics"]
    for row in revised["revision_diagnostics"]:
        assert set(row) == set(revision_spec["keys"])
        assert set(row["methods"]) == set(revision_spec["methods"])
        assert row["methods"]["M_MISSING"] == row["methods"]["Q"]
        for metric in row["methods"].values():
            assert set(metric) == set(revision_spec["metric_fields"])
            assert (
                metric["old_full_selected"]
                + metric["old_alias_selected"]
                + metric["old_nonalias_selected"]
                == 60
            )
        for metric in row["shuffle"]:
            assert set(metric) == {"mapping", *revision_spec["metric_fields"]}
    for old, new in zip(sensitive["parent_result"]["rows"], revised["rows"]):
        if old["method"] in ("A0_author", "FULL"):
            assert new == old
        assert old["control_available"] == new["control_available"]
        assert old["intervention_count"] == new["intervention_count"]
        assert old["label_count"] == new["label_count"]
    for row in revised["diagnostics"]:
        assert row["actions"]["QM"] == row["actions"]["Q"]
        if row["participant"] == 6:
            assert not row["admissible_shuffle_maps"]
            assert len(row["shuffle_interventions"]) == 1


@pytest.mark.parametrize("field", [*EVALUATOR.TABLES, "evidence_scope"])
def test_parent_mismatch_blocks_all_new_routing(field, sensitive):
    parent = copy.deepcopy(sensitive["parent_result"])
    parent[field] = "corrupted" if field == "evidence_scope" else parent[field][:-1]
    calls = []
    core = SimpleNamespace(evaluate_all=lambda *args: sensitive["parent_result"])
    rule = SimpleNamespace(
        route_family=lambda *args: calls.append("forbidden"),
        project_gains=lambda *args: calls.append("forbidden"),
    )
    kwargs = {**sensitive, "parent_result": parent, "core": core, "rule": rule}
    with pytest.raises(ValueError, match="Parent replay"):
        EVALUATOR.evaluate(**kwargs)
    assert not calls


def test_parent_comparison_checks_nested_values_and_tolerance():
    EVALUATOR.compare_parent({"a": [1, None, True, 0.1]}, {"a": [1, None, True, 0.1 + 5e-13]})
    for value in (
        {"a": [1, None, True, 0.1 + 2e-12]},
        {"a": [1, None, 1, 0.1]},
        {"a": [1, None, True, np.nan]},
        {"a": [1, None, True, 0.1], "extra": 0},
    ):
        with pytest.raises(ValueError, match="Parent replay"):
            EVALUATOR.compare_parent({"a": [1, None, True, 0.1]}, value)


def test_revision_metrics_distinguish_prefallback_and_actual_actions(sensitive):
    core, rule = sensitive["core"], sensitive["rule"]
    pred = np.tile(np.array([0, 0, 1, 2])[:, None], (1, 60))
    gain = np.tile(np.array([0.1, 0.2, 0.3])[:, None], (1, 60))
    # Own nonalias winner remains3 before fallback, but inherited Q changes1→2.
    metric = EVALUATOR.revision_metrics(
        pred, gain, np.ones(60, dtype=int), np.full(60, 2), core, rule
    )
    assert metric["old_alias_selected"] == 60
    assert metric["prefallback_action_changes"] == metric["prefallback_prediction_changes"] == 0
    assert metric["action_changes"] == metric["prediction_changes"] == 60
    assert metric["projected_gain_entries_changed"] == 60


def test_evaluator_does_not_fit_or_mutate_inputs(sensitive):
    core = sensitive["core"]
    before_freezes = copy.deepcopy(sensitive["freezes"])
    before_projection = copy.deepcopy(sensitive["content_projection"])
    before_parent = copy.deepcopy(sensitive["parent_result"])
    cache = sensitive["cache"]
    for array in cache.values():
        array.flags.writeable = False
    proxy = SimpleNamespace(
        **{
            name: getattr(core, name)
            for name in (
                "evaluate_all",
                "projection_arrays",
                "native_scores",
                "inputs_for_cell",
                "predict_gain",
                "support_maps",
                "choose",
                "packet_changes",
                "feature_changes",
                "reporting",
                "KEYS",
            )
        }
    )
    EVALUATOR.evaluate(**{**sensitive, "core": proxy})
    assert sensitive["freezes"] == before_freezes
    assert sensitive["content_projection"] == before_projection
    assert sensitive["parent_result"] == before_parent


def test_identity_rule_reproduces_every_original_table(sensitive):
    core = sensitive["core"]

    def original_family(pred, gains, masks, shuffled_gains, shuffled_masks):
        q = core.choose(gains["Q"])
        actions = {"Q": q, "M_MISSING": q.copy()}
        for mode in ("QM", "SHAM_REFIT", "M_STALE"):
            actions[mode] = np.where(masks[mode], q, core.choose(gains[mode]))
        return {
            "actions": actions,
            "shuffle_actions": [
                np.where(mask, q, core.choose(gain))
                for gain, mask in zip(shuffled_gains, shuffled_masks)
            ],
        }

    identity = SimpleNamespace(
        route_family=original_family, project_gains=lambda gains, pred: gains.copy()
    )
    result = EVALUATOR.evaluate(**{**sensitive, "rule": identity})
    for table in (*EVALUATOR.TABLES, "evidence_scope"):
        EVALUATOR.compare_parent(result[table], sensitive["parent_result"][table])
    for row in result["revision_diagnostics"]:
        for metric in [*row["methods"].values(), *row["shuffle"]]:
            assert metric["projected_gain_entries_changed"] == 0
            assert metric["action_changes"] == metric["prediction_changes"] == 0


def test_pure_main_imports_only_new_io(monkeypatch):
    import sys

    calls = []
    monkeypatch.setitem(
        sys.modules,
        "native_subset_known_zero_io",
        SimpleNamespace(execute=lambda function: calls.append(function)),
    )
    EVALUATOR.main()
    assert calls == [EVALUATOR.evaluate]
