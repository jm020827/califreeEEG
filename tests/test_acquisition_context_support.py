"""Integrated generated schedule/context checks, not EEG efficacy experiments."""

from dataclasses import replace

import numpy as np
import pytest

from cfeg.analysis.acquisition_context_support import bind_selected_contexts
from cfeg.analysis.calibration_collection_cost import CollectedTrial, balanced_prefix_cost
from cfeg.data.acquisition_context_boundary import (
    SignalSlice,
    TimingEvidence,
    extract_context,
    plan_context_window,
)
from cfeg.data.acquisition_context_contract import RunKey, SampleSpan

RUN = RunKey("generated", "p1", "day1", "low", "session1")
CLOCK = "generated-acquisition-200Hz"


def fixture(k=1, source=None, **cost_overrides):
    labels = (0, 0, 1, 2, 3, 0, 1, 2, 3, 1, 2, 3)
    trials = tuple(
        CollectedTrial(RUN, f"t{i}", label, 400 * (i + 1), 400 * (i + 1) + 100)
        for i, label in enumerate(labels)
    )
    args = {
        "run": RUN,
        "classes": (0, 1, 2, 3),
        "k": k,
        "sfreq": 200,
        "collection_start_sample": 0,
        "setup_seconds": None,
        "extra_baseline_seconds": 0,
        "clock_id": CLOCK,
    }
    cost = balanced_prefix_cost(trials, **(args | cost_overrides))
    if source is None:
        source = np.sin(np.arange(6000, dtype=float) * 0.03)[None, :]
    calls = []
    contexts = {}

    def fetch(run, span):
        calls.append((run, span))
        return SignalSlice(run, CLOCK, 200, span, source[:, span.start : span.stop].copy())

    for trial in trials:
        if cost.selected_trial_ids is None or trial.trial_id not in cost.selected_trial_ids:
            continue
        onset = trial.onset_sample
        timing = TimingEvidence(RUN, CLOCK, 200, 0, 0, "trial_onset", "generated-identity-export")
        p = plan_context_window(
            timing=timing,
            run=RUN,
            output=SampleSpan(onset - 20, onset - 5),
            cutoff_sample=onset,
            allowed=(SampleSpan(onset - 32, onset),),
            taps=(0.5, 0.3, 0.2),
        )
        contexts[trial.trial_id] = extract_context(p, fetch)
    return trials, cost, contexts, calls


def test_selected_contexts_follow_counted_prefix_not_discarded_or_future_trials():
    trials, cost, contexts, calls = fixture()
    bound = bind_selected_contexts(trials, cost, contexts)
    assert cost.selected_labels == 4 and cost.collected_labels == 5
    assert cost.elapsed_recording_seconds == 10.5 and cost.total_seconds is None
    assert [t.trial_id for t in bound.trials] == ["t0", "t2", "t3", "t4"]
    assert len(calls) == len(bound.contexts) == 4
    for trial, feature, (run, span) in zip(bound.trials, bound.contexts, calls, strict=True):
        assert run == RUN and feature.clock_id == cost.clock_id
        assert span.stop < trial.onset_sample
        assert feature.cutoff_sample == trial.onset_sample
    changed = np.full((1, 6000), np.nan)
    original = np.sin(np.arange(6000, dtype=float) * 0.03)[None, :]
    for _, span in calls:
        changed[:, span.start : span.stop] = original[:, span.start : span.stop]
    other_trials, other_cost, other_contexts, other_calls = fixture(source=changed)
    assert bind_selected_contexts(other_trials, other_cost, other_contexts) == bound
    assert other_calls == calls


def test_integrated_k0_k1_k3_and_unattained_k5_have_no_efficacy_claim():
    counts = []
    for k in (0, 1, 3):
        trials, cost, contexts, calls = fixture(k=k, setup_seconds=7, extra_baseline_seconds=2)
        bound = bind_selected_contexts(trials, cost, contexts)
        assert len(bound.trials) == len(bound.contexts) == 4 * k
        counts.append((cost.selected_labels, cost.collected_labels))
        if k == 0:
            assert calls == [] and cost.total_seconds == 9
    assert counts == [(0, 0), (4, 5), (12, 12)]
    trials, cost, contexts, calls = fixture(k=5)
    assert not cost.attained and cost.total_seconds is None
    assert cost.collected_labels == 12 and contexts == {} and calls == []
    with pytest.raises(ValueError, match="not collected"):
        bind_selected_contexts(trials, cost, contexts)


def test_exact_membership_and_same_run_clock_rate_onset_are_required():
    trials, cost, contexts, _ = fixture()
    identifier = cost.selected_trial_ids[0]
    feature = contexts[identifier]
    for change in (
        {"run": replace(RUN, session="session2")},
        {"clock_id": "other200Hz"},
        {"sfreq": 1000},
        {"cutoff_sample": feature.cutoff_sample + 1},
    ):
        with pytest.raises(ValueError):
            bind_selected_contexts(
                trials, cost, contexts | {identifier: replace(feature, **change)}
            )
    for altered in (
        {key: value for key, value in contexts.items() if key != identifier},
        contexts | {"t1": feature},
        contexts | {identifier: contexts["t2"]},
    ):
        with pytest.raises(ValueError):
            bind_selected_contexts(trials, cost, altered)
    with pytest.raises(ValueError, match="known collection clock"):
        bind_selected_contexts(trials, replace(cost, clock_id=None), contexts)
    # Self-consistent cost schema but a false elapsed claim must fail recomputation.
    forged_cost = replace(cost, elapsed_recording_seconds=9.0)
    with pytest.raises(ValueError, match="does not match"):
        bind_selected_contexts(trials, forged_cost, contexts)


def test_precollection_context_is_not_free_and_unknown_time_stays_unknown():
    # Onset400 needs exported history from378: 22/200 =0.11s before collection_start.
    for insufficient in (0.0, 0.109):
        trials, cost, contexts, _ = fixture(
            collection_start_sample=400, setup_seconds=0, extra_baseline_seconds=insufficient
        )
        with pytest.raises(ValueError, match="omits required"):
            bind_selected_contexts(trials, cost, contexts)
    for extra in (None, 0.11, 1.0):
        trials, cost, contexts, _ = fixture(
            collection_start_sample=400, setup_seconds=0, extra_baseline_seconds=extra
        )
        bound = bind_selected_contexts(trials, cost, contexts)
        assert bound.required_precollection_seconds == 0.11
        assert cost.elapsed_recording_seconds == 8.5
        assert cost.total_seconds == (None if extra is None else 8.5 + extra)
    # Collection starting early already includes the context; do not double-charge.
    trials, cost, contexts, _ = fixture(setup_seconds=0, extra_baseline_seconds=0)
    assert bind_selected_contexts(trials, cost, contexts).required_precollection_seconds == 0
