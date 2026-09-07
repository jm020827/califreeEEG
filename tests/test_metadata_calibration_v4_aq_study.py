from __future__ import annotations

import copy
import inspect
import json
from pathlib import Path

import numpy as np
import pytest

from cfeg.analysis import metadata_calibration_v4_aq_dev as study
from cfeg.baselines.fbcca import predict_fbcca

FIXTURE_SEED = 93817  # Explicit unit namespace; never derive any declared study namespace.
FILTERBANK = {
    "bands": [[6, 45], [14, 45]],
    "weights": [1.0, 0.5],
    "order": 2,
    "n_harmonics": 3,
    "regularization": 1e-8,
    "filter_family": "butterworth",
}


@pytest.fixture(scope="module")
def plan():
    root = Path(__file__).resolve().parents[1]
    result = json.loads(
        (root / "configs/analysis/metadata_calibration_v4_aq_study.json").read_text()
    )
    result.update(
        source_participants=2,
        evaluation_participants=2,
        n_classes=3,
        n_channels=4,
        max_samples=96,
        sample_counts=[48, 96],
        noise_multipliers=[1.0, 2.0],
        query_blocks=2,
        seed_namespace="unit-fixture-do-not-use-study-namespace",
    )
    result["dgp"]["frequency_step"] = 2.0
    return result


@pytest.fixture(scope="module")
def source(plan):
    study.initialize_worker(plan, FILTERBANK)
    records = [study.score_participant(p, FIXTURE_SEED) for p in range(2)]
    cache = {
        key: np.stack([r[key] for r in records])
        for key in (
            "a0_scores",
            "block_scores",
            "pooled_scores",
            "block_reliability",
            "labels",
            "eeg_sha256",
            "base_sha256",
        )
    }
    cache.update(
        schema=np.asarray(study.CACHE_SCHEMA),
        split=np.asarray("source"),
        participant_ids=np.arange(2),
        cell_ids=np.asarray([c["cell_id"] for c in study.cells(plan)]),
    )
    return cache


@pytest.fixture(scope="module")
def freeze(source, plan):
    return {**study.fit_temperatures(source, plan), **study.source_difficulty(source, plan)}


def test_explicit_rng_components_and_independent_partitions(plan):
    one = study.generate_base(plan, 0, root_seed=FIXTURE_SEED)
    study.component_rng(FIXTURE_SEED, 0, "state").normal(size=100)
    same = study.generate_base(plan, 0, root_seed=FIXTURE_SEED)
    another_person = study.generate_base(plan, 1, root_seed=FIXTURE_SEED)
    other_partition = study.generate_base(plan, 0, root_seed=FIXTURE_SEED + 1)
    assert one["base_sha256"] == same["base_sha256"]
    assert len({x["base_sha256"] for x in (one, another_person, other_partition)}) == 3
    assert "metadata" not in one
    with pytest.raises(ValueError):
        study.component_rng(-1, 0, "state")
    # Hash only an explicitly unrelated fixture namespace; no study namespace is opened.
    assert study.derive_study_seed("fixture-only", "source") != study.derive_study_seed(
        "fixture-only", "evaluation"
    )


def test_grid_uses_same_cropped_base_noise_states_and_orders(plan):
    base = study.generate_base(plan, 0, root_seed=FIXTURE_SEED)
    small = {"family": "stable", "n_samples": 48, "noise_multiplier": 1.0}
    long = {**small, "n_samples": 96}
    low = study.construct_cell(base, small, plan["dgp"])
    assert np.array_equal(low, study.construct_cell(base, long, plan["dgp"])[..., :48])
    high = study.construct_cell(base, {**small, "noise_multiplier": 2.0}, plan["dgp"])
    clean = np.take_along_axis(base["clean"][..., :48], base["labels"][..., None, None], axis=1)
    assert np.allclose(high - clean, 2 * (low - clean), atol=1e-15)
    neutral = {**plan["dgp"], "signal_log_gain": 0.0, "noise_log_gain": 0.0}
    assert np.array_equal(low, study.construct_cell(base, {**small, "family": "drift"}, neutral))
    assert all(np.array_equal(np.sort(row), np.arange(3)) for row in base["labels"])
    assert not np.array_equal(low[:2], low[5:])


def test_decode_matches_public_fbcca_crops_and_query_label_free(plan):
    base = study.generate_base(plan, 0, root_seed=FIXTURE_SEED)
    cell = study.cells(plan)[0]
    eeg = study.construct_cell(base, cell, plan["dgp"])
    workspace = study.build_workspace(plan, FILTERBANK, cell["n_samples"])
    decoded = study.decode_scores(eeg, base["labels"][:5], workspace)
    query = eeg[5:].reshape(-1, 4, 48)
    frequencies = 8 + 2 * np.arange(3)
    expected = np.stack([predict_fbcca(q, frequencies, 250, FILTERBANK)[1] for q in query]) / 1.5
    assert np.allclose(decoded["a0_scores"], expected, atol=1e-10, rtol=0)
    assert np.array_equal(decoded["pooled_scores"][0], decoded["block_scores"][:, 0])
    assert set(inspect.signature(study.decode_scores).parameters) == {
        "eeg",
        "support_labels",
        "workspace",
    }
    for function in (study.decode_scores, study.predict_methods, study.reliability_weights):
        assert not set(inspect.signature(function).parameters) & {
            "metadata",
            "query_labels",
            "state",
            "participant_id",
            "query_context",
        }
    changed = base["labels"].copy()
    changed[5:] = (changed[5:] + 1) % 3
    second = study.decode_scores(eeg, changed[:5], workspace)
    assert all(np.array_equal(decoded[key], second[key]) for key in decoded)
    malformed = eeg.copy()
    malformed[0, 0, 0, 0] = np.nan
    with pytest.raises(ValueError, match="Finite"):
        study.decode_scores(malformed, base["labels"][:5], workspace)


def test_queries_cannot_influence_other_query_rows(plan):
    base = study.generate_base(plan, 0, root_seed=FIXTURE_SEED)
    cell = study.cells(plan)[0]
    eeg = study.construct_cell(base, cell, plan["dgp"])
    workspace = study.build_workspace(plan, FILTERBANK, 48)
    one = study.decode_scores(eeg, base["labels"][:5], workspace)
    changed = eeg.copy()
    changed[6] *= -3
    two = study.decode_scores(changed, base["labels"][:5], workspace)
    assert np.array_equal(one["a0_scores"][:3], two["a0_scores"][:3])
    assert np.array_equal(one["block_scores"][:3], two["block_scores"][:3])
    assert np.array_equal(one["pooled_scores"][:, :3], two["pooled_scores"][:, :3])
    assert np.array_equal(one["block_reliability"], two["block_reliability"])


def test_k0_k1_aliases_and_fixed_fusion(source, freeze, plan):
    scores = {
        k: source[k][0, 0]
        for k in ("a0_scores", "block_scores", "pooled_scores", "block_reliability")
    }
    for budget in plan["budgets"]:
        result = study.predict_methods(scores, freeze["fits"], budget, plan)
        assert set(result) == set(study.METHODS)
        assert np.array_equal(result["A0_raw"].argmax(axis=1), result["A0_cal"].argmax(axis=1))
        for suffix in ("raw", "cal"):
            if budget == 0:
                assert all(
                    np.array_equal(p, result[f"A0_{suffix}"])
                    for name, p in result.items()
                    if name.endswith(suffix)
                )
            else:
                assert np.array_equal(
                    result[f"fusion_pooled_{suffix}"],
                    0.5 * result[f"A0_{suffix}"] + 0.5 * result[f"pooled_{suffix}"],
                )
                if budget == 1:
                    assert np.array_equal(result[f"pooled_{suffix}"], result[f"block_{suffix}"])
        if budget:
            # This invariance is NOT asserted for a mixture of block posteriors.
            assert np.array_equal(
                result["pooled_raw"].argmax(axis=1), result["pooled_cal"].argmax(axis=1)
            )


def test_source_fit_objective_is_actual_posterior_mixture(source, freeze, plan):
    labels = source["labels"][:, 5:].reshape(2, -1)
    temperature = freeze["temperature_grid"][11]
    weights = study.reliability_weights(source["block_reliability"], 3, 0.05)
    block_scores = source["block_scores"][..., :3, :]
    probability = np.zeros_like(source["a0_scores"])
    for block in range(3):
        probability += (
            study.numeric.softmax(block_scores[..., block, :], temperature)
            * weights[..., block, None, None]
        )
    objective = study.mean_nll(probability, labels)
    assert objective == pytest.approx(freeze["fits"]["block_k3"]["objective_values"][11], abs=1e-14)
    incorrect = study.numeric.softmax(
        np.sum(block_scores * weights[..., None, :, None], axis=-2), temperature
    )
    assert not np.allclose(probability, incorrect)
    eval_scores = {**source, "split": np.asarray("evaluation")}
    with pytest.raises(ValueError, match="source"):
        study.fit_temperatures(eval_scores, plan)
    # Evaluation answers cannot change a function which receives only source arrays.
    unrelated_eval_labels = (source["labels"] + 1) % plan["n_classes"]
    assert not np.array_equal(unrelated_eval_labels, source["labels"])
    assert study.fit_temperatures(source, plan) == study.fit_temperatures(source, plan)


def test_deterministic_temperature_ties_and_bad_block_inputs(source, plan):
    tied = copy.deepcopy(source)
    for key in ("a0_scores", "block_scores", "pooled_scores"):
        tied[key].fill(0)
    fit = study.fit_temperatures(tied, plan)
    assert all(record["selected_index"] == 0 for record in fit["fits"].values())
    scores = np.zeros((2, 3, 3))
    for temperature in (0.0, -1.0, np.nan):
        with pytest.raises(ValueError, match="temperature"):
            study.block_posterior(scores, np.ones(3) / 3, temperature)
    with pytest.raises(ValueError, match="floor"):
        study.reliability_weights(np.ones(5), 0, 0.05)


def test_difficulty_selection_is_a0_only_and_full_cache_shape(source, plan):
    original = study.source_difficulty(source, plan)
    altered = copy.deepcopy(source)
    altered["block_scores"] *= -500
    altered["pooled_scores"] += 1000
    assert study.source_difficulty(altered, plan) == original
    assert len(original["source_difficulty_rows"]) == 8
    assert all(r["family"] == "stable" for r in original["source_difficulty_rows"] if r["selected"])
    assert source["a0_scores"].shape == (2, 8, 6, 3)
    assert source["block_scores"].shape == (2, 8, 6, 5, 3)
    assert source["pooled_scores"].shape == (2, 8, 3, 6, 3)
    assert source["labels"].shape == (2, 7, 3)


def test_complete_metrics_and_participant_level_primary(source, freeze, plan):
    selected = [c["cell_id"] for c in study.cells(plan) if c["family"] == "stable"][:2]
    chosen = {**freeze, "selected_stable_cells": selected}
    rows = study.evaluate_rows(source, chosen, plan)
    assert len(rows) == 2 * 8 * 4 * 10
    assert all("prediction_flips_vs_AQ" not in row for row in rows)
    summary = study.summarize_results(rows, chosen, plan)
    assert summary["primary"]["eauc_gain"]["n"] == 2  # Not people times selected cells.
    assert len(summary["aggregate_rows"]) == 8 * 4 * 10
    assert len(summary["calibration_attainment"]) == 8 * 10
    lookup = {
        (r["participant"], r["cell_id"], r["method"], r["budget"]): r["balanced_accuracy"]
        for r in rows
    }
    expected = []
    for participant in range(2):
        expected.append(
            np.mean(
                [
                    sum(
                        weight
                        * (
                            lookup[participant, cell, "fusion_pooled_cal", k]
                            - lookup[participant, cell, "A0_cal", k]
                        )
                        for k, weight in ((0, 1 / 6), (1, 0.5), (3, 1 / 3))
                    )
                    for cell in selected
                ]
            )
        )
    assert np.allclose(summary["primary"]["eauc_gain"]["participant_values"], expected, atol=1e-15)
    empty = study.summarize_results(rows, {**chosen, "selected_stable_cells": []}, plan)
    assert empty["status"] == "NO_SOURCE_INFORMATIVE_CELLS"
    assert empty["primary"]["eauc_gain"] is None
    assert len(empty["aggregate_rows"]) == len(summary["aggregate_rows"])


def test_exclusive_cache_publication_and_readback(source, tmp_path):
    path = tmp_path / "fixture.npz"
    digest = study.write_cache_exclusive(path, source)
    assert digest == study.file_sha(path)
    assert path.stat().st_mode & 0o777 == 0o400
    assert path.stat().st_nlink == 1
    with np.load(path, allow_pickle=False) as restored:
        assert set(restored.files) == set(source)
        assert all(np.array_equal(restored[key], source[key]) for key in source)
    with pytest.raises(FileExistsError):
        study.write_cache_exclusive(path, source)


def test_source_freeze_is_durable_before_evaluation_namespace(source, plan, monkeypatch, tmp_path):
    events = []

    def seed(namespace, split):
        assert namespace == "unit-fixture-do-not-use-study-namespace"
        if split == "evaluation":
            assert (tmp_path / "source-freeze.json").exists()
            assert (tmp_path / "evaluation-start.json").exists()
        events.append(split)
        return FIXTURE_SEED + (split == "evaluation")

    def collect(_plan, _filterbank, split, _seed, _workers):
        return {**source, "split": np.asarray(split)}

    monkeypatch.setattr(study, "derive_study_seed", seed)
    monkeypatch.setattr(study, "collect_scores", collect)
    start = {"source_commit": "fixture-only", "plan_sha256": "fixture-plan"}
    start_sha = study.numeric.write_json_exclusive(tmp_path / "start.json", start)
    result = study.execute_phases(plan, FILTERBANK, tmp_path, start, start_sha, 1)
    assert events == ["source", "evaluation"]
    assert {p.name for p in tmp_path.iterdir()} == set(plan["execution"]["artifacts"])
    assert result["human_unlock"] is False
    freeze = json.loads((tmp_path / "source-freeze.json").read_text())
    evaluation_start = json.loads((tmp_path / "evaluation-start.json").read_text())
    assert freeze["source_scores_sha256"] == study.file_sha(tmp_path / "source-scores.npz")
    assert evaluation_start["source_freeze_sha256"] == study.file_sha(
        tmp_path / "source-freeze.json"
    )


def test_runner_refuses_existing_start_before_any_study_rng(monkeypatch):
    root = Path(__file__).resolve().parents[1]
    path = root / "configs/analysis/metadata_calibration_v4_aq_study.json"
    fixed = json.loads(path.read_text())
    original_exists = Path.exists

    def exists(current):
        if current == Path(fixed["execution"]["output_root"]) / "start.json":
            return True
        return original_exists(current)

    monkeypatch.setattr(Path, "exists", exists)
    monkeypatch.setattr(
        study, "derive_study_seed", lambda *args: pytest.fail("Study RNG forbidden")
    )
    with pytest.raises(FileExistsError, match="consumes"):
        study.run_study(path, Path(fixed["execution"]["output_root"]), 4)


def test_support_row_and_class_equivariance(plan):
    base = study.generate_base(plan, 0, root_seed=FIXTURE_SEED)
    eeg = study.construct_cell(base, study.cells(plan)[0], plan["dgp"])
    workspace = study.build_workspace(plan, FILTERBANK, 48)
    labels = base["labels"][:5]
    original = study.decode_scores(eeg, labels, workspace)
    reordered = eeg.copy()
    reordered[:5] = reordered[:5, ::-1]
    changed = study.decode_scores(reordered, labels[:, ::-1], workspace)
    assert all(np.allclose(original[key], changed[key], atol=1e-14) for key in original)
    # For template-only class relabeling the expected score axis is permuted.
    relabeled = study.decode_scores(eeg, (labels + 1) % 3, workspace)
    assert np.allclose(np.roll(original["block_scores"], 1, axis=-1), relabeled["block_scores"])
    assert np.allclose(np.roll(original["pooled_scores"], 1, axis=-1), relabeled["pooled_scores"])
