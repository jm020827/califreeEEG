from __future__ import annotations

import copy
import inspect
import json
from pathlib import Path

import numpy as np
import pytest

from cfeg.analysis import metadata_calibration_v4_pilot as pilot
from cfeg.baselines.fbcca import apply_filterbank, cca_score, make_reference_signals, predict_fbcca

FIXTURE_SEED = 817  # Unit namespace only; never derive the declared study root here.
FILTERBANK = {
    "bands": [[6, 45], [14, 45]],
    "weights": [1.0, 0.5],
    "order": 2,
    "n_harmonics": 3,
    "regularization": 1e-8,
    "filter_family": "butterworth",
}


@pytest.fixture
def plan():
    path = (
        Path(__file__).resolve().parents[1] / "configs/analysis/metadata_calibration_v4_pilot.json"
    )
    result = json.loads(path.read_text())
    result.update(n_classes=3, n_channels=4, n_samples=96, query_blocks=2, participants=4)
    result["dgp"]["frequency_step"] = 2.0
    return result


def test_rng_components_are_independent_of_call_order():
    first = pilot.component_rng(FIXTURE_SEED, 2, "innovation").normal(size=10)
    pilot.component_rng(FIXTURE_SEED, 2, "trial_order").normal(size=300)
    second = pilot.component_rng(FIXTURE_SEED, 2, "innovation").normal(size=10)
    assert np.array_equal(first, second)
    assert not np.array_equal(
        first, pilot.component_rng(FIXTURE_SEED, 3, "innovation").normal(size=10)
    )
    assert not np.array_equal(
        first, pilot.component_rng(FIXTURE_SEED, 2, "trial_jitter").normal(size=10)
    )


def test_generator_drift_null_identity_and_reproducibility(plan):
    one = pilot.generate_participant(plan, 1, root_seed=FIXTURE_SEED)
    two = pilot.generate_participant(plan, 1, root_seed=FIXTURE_SEED)
    assert one["eeg"]["acquisition_drift"] is one["eeg"]["metadata_null"]
    assert np.array_equal(one["eeg"]["stable"], two["eeg"]["stable"])
    assert np.array_equal(one["labels"], two["labels"])
    assert all(np.array_equal(np.sort(row), np.arange(plan["n_classes"])) for row in one["labels"])
    assert not np.array_equal(
        one["metadata"]["acquisition_drift"], one["metadata"]["metadata_null"]
    )
    assert not np.array_equal(one["eeg"]["stable"][:2], one["eeg"]["stable"][5:])
    # Turn off state effects: the two EEG paths then expose identical underlying draws.
    neutral = copy.deepcopy(plan)
    neutral["dgp"]["signal_log_gain"] = neutral["dgp"]["noise_log_gain"] = 0
    shared = pilot.generate_participant(neutral, 1, root_seed=FIXTURE_SEED)
    assert np.array_equal(shared["eeg"]["stable"], shared["eeg"]["acquisition_drift"])


def test_cached_cca_matches_public_oracle_and_batch_filtering():
    rng = np.random.Generator(np.random.PCG64DXSM(FIXTURE_SEED))
    eeg = rng.normal(size=(3, 4, 96))
    eeg[1, 3] = eeg[1, 0]  # Rank deficiency is a meaningful whitening edge.
    frequencies = np.array([8.0, 10.0, 12.0])
    cache = pilot.reference_cache(frequencies, 250, 96, 3, 1e-8)
    actual = pilot.cached_cca_scores(eeg, cache, 1e-8)
    refs = make_reference_signals(frequencies, 250, 96, 3)
    expected = np.asarray([[cca_score(trial, ref) for ref in refs] for trial in eeg])
    assert np.allclose(actual, expected, rtol=0, atol=1e-10)
    filtered, parameters = apply_filterbank(eeg, sfreq=250, filterbank=FILTERBANK)
    cached = np.stack([pilot.cached_cca_scores(band, cache, 1e-8) for band in filtered])
    scores = np.einsum("f,fqc->qc", parameters["weights"], cached**2)
    expected_fbcca = np.stack(
        [predict_fbcca(trial, frequencies, 250, FILTERBANK)[1] for trial in eeg]
    )
    assert np.allclose(scores, expected_fbcca, rtol=0, atol=1e-10)
    for index, trial in enumerate(eeg):
        solo, _ = apply_filterbank(trial, sfreq=250, filterbank=FILTERBANK)
        assert np.array_equal(solo, filtered[:, index])
    assert not pilot.cached_cca_scores(np.zeros_like(eeg), cache, 1e-8).any()


def _posterior_fixture():
    p0 = np.array([[0.55, 0.3, 0.15], [0.15, 0.6, 0.25]])
    pb = np.array(
        [
            [[0.1, 0.8, 0.1], [0.9, 0.05, 0.05], [0.2, 0.3, 0.5]],
            [[0.6, 0.3, 0.1], [0.1, 0.1, 0.8], [0.05, 0.9, 0.05]],
        ]
    )
    pi = np.array([[0.2, 0.6, 0.2], [0.4, 0.2, 0.4]])
    return p0, pb, pi


def test_exact_k0_missing_k1_uniform_and_block_scalar_identity():
    p0, pb, pi = _posterior_fixture()
    zero = pilot.fuse_posteriors(p0, pb[:, :0], pi[:, :0], strength=0.5)
    assert all(np.array_equal(p0, value) for value in zero.values())
    missing = pilot.fuse_posteriors(p0, pb, pi, strength=0.5)
    assert all(np.array_equal(missing["AQ"], value) for value in missing.values())
    one = pilot.fuse_posteriors(
        p0, pb[:, :1], np.ones((2, 1)), strength=0.5, affinities=np.array([[0.2], [0.7]])
    )
    assert np.array_equal(one["AQM_block"], one["AQM_scalar"])
    uniform = np.array([[0.2] * 3, [0.7] * 3])
    same = pilot.fuse_posteriors(p0, pb, pi, strength=0.5, affinities=uniform)
    assert np.array_equal(same["AQM_block"], same["AQM_scalar"])
    a = np.array([[0.8, 0.2, 0.3], [0.1, 0.9, 0.4]])
    different = pilot.fuse_posteriors(p0, pb, pi, strength=0.5, affinities=a)
    g = np.sum(pi * a, axis=1)
    oracle = 0.5 * np.sum((pi * (a - g[:, None]))[..., None] * pb, axis=1)
    assert np.allclose(different["AQM_block"] - different["AQM_scalar"], oracle, atol=1e-15)
    assert np.array_equal(different["AQ"], same["AQ"])
    assert np.array_equal(different["missing"], same["missing"])
    for probability in different.values():
        assert (probability >= 0).all()
        assert np.allclose(probability.sum(axis=1), 1)
    assert pilot.affinity_metrics(np.ones((2, 1)), a[:, :1])["weight_turnover"] == 0
    assert pilot.affinity_metrics(pi, uniform)["weight_turnover"] < 1e-15
    assert pilot.applied_affinity_metrics("AQM_scalar", pi, a)["weight_turnover"] == 0
    assert pilot.applied_affinity_metrics("A0", pi, a)["metadata_total_trust"] == 0


def test_projection_quality_is_label_free_row_local_and_offset_invariant(plan):
    pilot.initialize_workspace(plan, FILTERBANK)
    workspace = pilot._WORKSPACE
    eeg = pilot.generate_participant(plan, 0, root_seed=FIXTURE_SEED)["eeg"]["stable"][0]
    basis = workspace["basis"]
    q = pilot.eeg_quality(eeg, basis, 1e-12)
    solo = pilot.eeg_quality(eeg[:1], basis, 1e-12)
    assert np.allclose(q[:1], solo, atol=1e-12, rtol=0)
    offset = np.arange(plan["n_channels"])[None, :, None] * 123
    assert np.allclose(q, pilot.eeg_quality(eeg + offset, basis, 1e-12), atol=1e-10, rtol=0)
    for function in (pilot.eeg_quality, pilot.eeg_block_weights):
        parameters = set(inspect.signature(function).parameters)
        assert not parameters & {"metadata", "query_labels", "labels", "subject_id", "state"}
    assert not set(inspect.signature(pilot.fuse_posteriors).parameters) & {
        "labels",
        "query_labels",
        "metadata",
    }


def test_signed_waveform_scoring_and_query_batch_isolation():
    timeline = np.arange(96) / 250
    wave = np.stack([np.sin(2 * np.pi * 10 * timeline), np.cos(2 * np.pi * 10 * timeline)])
    support = np.stack([wave, -wave])
    query = np.stack([wave, -wave, wave * 0.3])
    score = pilot._waveform_scores(query[None], support[None], np.array([0, 1]), np.ones(1), 2)
    assert np.array_equal(score.argmax(axis=1), [0, 1, 0])
    assert np.allclose(score[0], [1, -1], atol=1e-12)
    assert np.allclose(
        score[:1],
        pilot._waveform_scores(query[None, :1], support[None], np.array([0, 1]), np.ones(1), 2),
        atol=1e-15,
    )
    reordered = pilot._waveform_scores(
        query[None], support[None, ::-1], np.array([1, 0]), np.ones(1), 2
    )
    relabeled = pilot._waveform_scores(query[None], support[None], np.array([1, 0]), np.ones(1), 2)
    assert np.array_equal(score, reordered)
    assert np.array_equal(score[:, ::-1], relabeled)


def test_decoding_never_uses_query_labels(plan):
    pilot.initialize_workspace(plan, FILTERBANK)
    data = pilot.generate_participant(plan, 0, root_seed=FIXTURE_SEED)
    changed_labels = data["labels"].copy()
    changed_labels[plan["support_blocks"] :] = (
        changed_labels[plan["support_blocks"] :] + 1
    ) % plan["n_classes"]
    original = pilot._decode_eeg(data["eeg"]["stable"], data["labels"], pilot._WORKSPACE)
    changed = pilot._decode_eeg(data["eeg"]["stable"], changed_labels, pilot._WORKSPACE)
    for key in original:
        assert np.array_equal(original[key], changed[key])


def test_nonfinite_and_malformed_numerical_inputs_are_rejected():
    p0, pb, pi = _posterior_fixture()
    with pytest.raises(ValueError, match="shapes"):
        pilot.fuse_posteriors(p0, pb[:, :2], pi, strength=0.5)
    with pytest.raises(ValueError, match="Affinities"):
        pilot.fuse_posteriors(p0, pb, pi, strength=0.5, affinities=np.full_like(pi, np.nan))
    with pytest.raises(ValueError, match="probability"):
        pilot.fuse_posteriors(p0 * 2, pb, pi, strength=0.5)
    with pytest.raises(ValueError, match="Quality"):
        pilot.eeg_quality(np.full((2, 4, 96), np.nan), np.zeros((3, 96)), 1e-12)
    with pytest.raises(ValueError, match="positive"):
        pilot.metadata_affinity(np.zeros((2, 4)), np.ones((3, 4)), 0.6)


def test_derangements_average_metrics_not_posteriors():
    assert len(pilot.derangements(3)) == 2
    assert len(pilot.derangements(5)) == 44
    assert all(all(i != x for i, x in enumerate(p)) for p in pilot.derangements(5))
    labels = np.array([0, 1])
    p0 = np.array([[0.6, 0.4], [0.4, 0.6]])
    one, two = np.array([[0.51, 0.49], [0.49, 0.51]]), np.array([[0.01, 0.99], [0.99, 0.01]])
    records = [pilot.posterior_metrics(p, labels, p0, p0) for p in (one, two)]
    mean = pilot.mean_metric_records(records)
    posterior_mean = pilot.posterior_metrics((one + two) / 2, labels, p0, p0)
    assert mean["balanced_accuracy"] == 0.5
    assert posterior_mean["balanced_accuracy"] == 0.0
    assert mean["correct_log_probability"] != posterior_mean["correct_log_probability"]


def test_small_fixture_full_participant_evidence_and_exact_controls(plan):
    pilot.initialize_workspace(plan, FILTERBANK)
    result = pilot.evaluate_participant(0, FIXTURE_SEED)
    assert len(result["rows"]) == 84
    assert len(result["prediction_evidence"]) == 72
    assert len(result["derangement_evidence"]) == 12
    assert set(result["support_indices"]).isdisjoint(result["query_indices"])
    assert result["eeg_sha256"]["acquisition_drift"] == result["eeg_sha256"]["metadata_null"]
    lookup = {(r["family"], r["method"], r["budget"]): r for r in result["prediction_evidence"]}
    for family in plan["families"]:
        for budget in plan["budgets"]:
            assert (
                lookup[family, "AQ", budget]["posterior_sha256"]
                == lookup[family, "missing", budget]["posterior_sha256"]
            )
            if budget == 0:
                assert len({lookup[family, m, 0]["posterior_sha256"] for m in pilot.METHODS}) == 1
            if budget == 1:
                assert (
                    lookup[family, "AQM_block", 1]["posterior_sha256"]
                    == lookup[family, "AQM_scalar", 1]["posterior_sha256"]
                )
            for method in ("A0", "AQ", "pooled", "missing"):
                assert (
                    lookup["acquisition_drift", method, budget]["posterior_sha256"]
                    == lookup["metadata_null", method, budget]["posterior_sha256"]
                )
    metrics = {(r["family"], r["method"], r["budget"]): r for r in result["rows"]}
    for evidence in result["derangement_evidence"]:
        assert metrics[evidence["family"], evidence["method"], evidence["budget"]][
            "balanced_accuracy"
        ] == np.mean(evidence["balanced_accuracy"])
    # No support/query RNG draw is repeated to build this deterministic comparison.
    assert result == pilot.evaluate_participant(0, FIXTURE_SEED)


def test_paired_intervals_zero_difference_and_negative_screen(plan):
    interval = pilot.paired_summary([0, 0, 0, 0])
    assert interval["two_sided_95_CI"] == [0, 0]
    assert interval["one_sided_95_LCB"] == 0
    pilot.initialize_workspace(plan, FILTERBANK)
    fixture = pilot.evaluate_participant(0, FIXTURE_SEED)
    results = []
    for participant in range(4):
        entry = copy.deepcopy(fixture)
        entry["participant"] = participant
        for row in entry["rows"]:
            row["participant"] = participant
            row["balanced_accuracy"] = 0.5
        results.append(entry)
    summary = pilot.summarize_results(results, plan)
    assert summary["status"] == "AQ_NOT_ESTABLISHED"
    assert summary["human_unlock"] is False
    assert all(
        row["participant_budgets"] == [">5"] * 4 for row in summary["calibration_attainment"]
    )
    # Hand-defined separated contrasts satisfy every stated screen.
    for entry in results:
        for row in entry["rows"]:
            family, method, budget = row["family"], row["method"], row["budget"]
            row["balanced_accuracy"] = 0.4 if method == "A0" or budget == 0 else 0.5
            if family == "stable" and method == "AQ" and budget > 0:
                row["balanced_accuracy"] = 0.6
            if family == "acquisition_drift" and method == "AQM_block" and budget > 0:
                row["balanced_accuracy"] = 0.6
            if family == "acquisition_drift" and method == "AQM_scalar" and budget > 0:
                row["balanced_accuracy"] = 0.55
    positive = pilot.summarize_results(results, plan)
    assert positive["status"] == "READY_FOR_INDEPENDENT_SYNTHETIC_REPLICATION"
    assert positive["human_unlock"] is False


def test_exclusive_json_never_overwrites(tmp_path):
    path = tmp_path / "start.json"
    pilot.write_json_exclusive(path, {"fixture": True})
    original = path.read_bytes()
    with pytest.raises(FileExistsError):
        pilot.write_json_exclusive(path, {"fixture": False})
    assert path.read_bytes() == original
    assert path.stat().st_mode & 0o777 == 0o400
