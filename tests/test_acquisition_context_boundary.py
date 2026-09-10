from dataclasses import replace

import numpy as np
import pytest

from cfeg.data.acquisition_context_boundary import (
    SignalSlice,
    TimingEvidence,
    extract_context,
    marker_1based_to_onset,
    plan_context_window,
    require_context_pairing,
    seconds_to_samples,
)
from cfeg.data.acquisition_context_contract import RunKey, SampleSpan

RUN = RunKey("generated", "p1", "day1", "low", "run1")


def timing(**changes):
    defaults = {
        "run": RUN,
        "clock_id": "generated-200Hz",
        "sfreq": 200,
        "export_history_samples": 2,
        "export_future_samples": 1,
        "marker_anchor": "trial_onset",
        "evidence_id": "generated-export-fir-known-support",
    }
    return TimingEvidence(**(defaults | changes))


def plan(**changes):
    defaults = {
        "timing": timing(),
        "run": RUN,
        "output": SampleSpan(10, 20),
        "cutoff_sample": 21,
        "allowed": (SampleSpan(5, 21),),
        "taps": (0.5, 0.3, 0.2),
    }
    return plan_context_window(**(defaults | changes))


def fetcher(x, calls, **changes):
    def fetch(run, span):
        calls.append((run, span))
        values = x[:, span.start : span.stop].copy()
        fields = {
            "run": run,
            "clock_id": "generated-200Hz",
            "sfreq": 200,
            "span": span,
            "values": values,
        }
        return SignalSlice(**(fields | changes))

    return fetch


def test_exact_grid_conversion_and_marker_origin():
    assert seconds_to_samples("0.14", 200) == 28
    assert seconds_to_samples("1/200", 200) == 1
    assert seconds_to_samples("0", 200) == 0
    assert marker_1based_to_onset(1) == 0
    for invalid in ("0.143", "-0.01", "nan", "inf", "1/0", 0.14, True):
        with pytest.raises(ValueError):
            seconds_to_samples(invalid, 200)
    for invalid in (0, -1, True, 2.0):
        with pytest.raises(ValueError):
            marker_1based_to_onset(invalid)
    with pytest.raises(ValueError):
        seconds_to_samples("1", True)


def test_run_and_span_types_are_strict():
    for field in ("dataset", "participant", "day", "band", "session"):
        with pytest.raises(ValueError):
            replace(RUN, **{field: " "})
        with pytest.raises(ValueError):
            replace(RUN, **{field: 1})
    for start, stop in ((-1, 3), (1, 1), (True, 4), (1.0, 3), (0, False)):
        with pytest.raises(ValueError):
            SampleSpan(start, stop)
    assert SampleSpan(0, 4).length == 4
    assert SampleSpan(0, 4).contains(SampleSpan(1, 4))


def test_composed_half_open_dependency_boundary_and_holes():
    p = plan()
    assert p.exported_read == SampleSpan(8, 20)
    assert p.dependency == SampleSpan(6, 21)
    assert plan(allowed=(SampleSpan(6, 11), SampleSpan(11, 21))) == replace(
        p, allowed=(SampleSpan(6, 11), SampleSpan(11, 21))
    )
    for changes in (
        {"cutoff_sample": 20},
        {"allowed": (SampleSpan(6, 10), SampleSpan(11, 21))},
        {"allowed": (SampleSpan(7, 21),)},
        {"output": SampleSpan(1, 3)},
        {"cutoff_sample": True},
        {"allowed": []},
        {"taps": ()},
        {"taps": (True,)},
        {"taps": (float("nan"),)},
    ):
        with pytest.raises(ValueError):
            plan(**changes)


def test_unknowns_and_all_identity_dimensions_fail_before_fetch():
    calls = []
    x = np.ones((1, 30))
    invalid_timings = [
        timing(export_history_samples=None),
        timing(export_future_samples=None),
        timing(marker_anchor=None),
        timing(marker_anchor="cue_onset"),
        timing(evidence_id=None),
        timing(sfreq=200.0),
        timing(clock_id=""),
        timing(export_future_samples=True),
    ]
    for invalid in invalid_timings:
        with pytest.raises(ValueError):
            extract_context(plan(timing=invalid), fetcher(x, calls))
    for field in ("dataset", "participant", "day", "band", "session"):
        with pytest.raises(ValueError):
            extract_context(plan(run=replace(RUN, **{field: "other"})), fetcher(x, calls))
    assert calls == []


def test_future_forbidden_gap_and_previous_call_invariance_with_sensitivity():
    p = plan()
    x = np.arange(80, dtype=float).reshape(2, 40)
    calls = []
    a = extract_context(p, fetcher(x, calls))
    # All requested values are equal; forbidden past/gap and future can be NaN.
    changed = x.copy()
    changed[:, :6] = np.nan
    changed[:, 21:] = 1e100
    b = extract_context(p, fetcher(changed, calls))
    assert a == b
    assert calls == [(RUN, SampleSpan(8, 20))] * 2
    p2 = plan(output=SampleSpan(25, 30), cutoff_sample=31, allowed=(SampleSpan(20, 31),))
    _ = extract_context(p2, fetcher(x, []))
    assert extract_context(p, fetcher(x, [])) == a  # No cross-call filter state.
    changed[:, 12] += 10
    assert extract_context(p, fetcher(changed, [])) != a


def test_fir_orientation_matches_independent_nested_sum_and_constant():
    p = plan()
    x = np.arange(40, dtype=float)[None, :]
    expected = np.array(
        [
            sum(float(p.taps[lag]) * x[0, t - lag] for lag in range(len(p.taps)))
            for t in range(p.output.start, p.output.stop)
        ]
    )
    got = extract_context(p, fetcher(x, []))
    np.testing.assert_allclose(got.mean, [expected.mean()], atol=1e-12, rtol=0)
    np.testing.assert_allclose(got.std, [expected.std()], atol=1e-12, rtol=0)
    np.testing.assert_allclose(got.mean_absolute_difference, [1.0], atol=1e-12, rtol=0)
    constant = extract_context(p, fetcher(np.ones((1, 40)), []))
    assert constant.mean == (1.0,)
    assert constant.std == constant.mean_absolute_difference == (0.0,)


def test_deliberately_leaky_reference_detects_future_perturbation():
    # Intentionally invalid full-run centering is a future-sensitive control.
    x = np.arange(40, dtype=float)[None, :]
    changed = x.copy()
    changed[:, 21:] += 1000
    leaky = lambda z: (z - z.mean(axis=1, keepdims=True))[:, 10:20].mean()
    assert leaky(x) != leaky(changed)
    assert extract_context(plan(), fetcher(x, [])) == extract_context(plan(), fetcher(changed, []))


def test_returned_clock_identity_span_shape_and_values_are_checked():
    x = np.ones((1, 40))
    for changes in (
        {"run": replace(RUN, participant="other")},
        {"clock_id": "unrelated-clock"},
        {"sfreq": 1000},
        {"sfreq": 200.0},
        {"span": SampleSpan(7, 20)},
        {"values": np.ones((12,))},
        {"values": np.ones((0, 12))},
        {"values": np.ones((1, 13))},
        {"values": np.full((1, 12), np.nan)},
        {"values": np.ones((1, 12), dtype=complex)},
        {"values": np.ones((1, 12), dtype=bool)},
        {"values": np.ones((1, 12), dtype=object)},
    ):
        with pytest.raises(ValueError):
            extract_context(plan(), fetcher(x, [], **changes))
    with pytest.raises(ValueError):
        extract_context(plan(), lambda run, span: x)


def test_single_sample_diff_and_numeric_overflow():
    p = plan(output=SampleSpan(10, 11))
    assert extract_context(p, fetcher(np.ones((1, 40)), [])).mean_absolute_difference == (0.0,)
    with pytest.raises(ValueError):
        extract_context(plan(taps=(1e308,)), fetcher(np.full((1, 40), 1e308), []))


def test_realized_toy_export_future_support_is_composed_before_fetch():
    # This is a known generated transform, NOT a claim about the Choi exporter.
    raw = np.arange(80, dtype=float).reshape(2, 40)
    requested_dependencies = []

    def exported_fetch(source):
        def fetch(run, span):
            dependency = SampleSpan(span.start - 2, span.stop + 1)
            requested_dependencies.append(dependency)
            values = (
                0.2 * source[:, span.start - 2 : span.stop - 2]
                + 0.5 * source[:, span.start : span.stop]
                + 0.3 * source[:, span.start + 1 : span.stop + 1]
            )
            return SignalSlice(run, "generated-200Hz", 200, span, values)

        return fetch

    original = extract_context(plan(), exported_fetch(raw))
    changed = raw.copy()
    changed[:, :6] = np.nan
    changed[:, 21:] = np.nan
    assert extract_context(plan(), exported_fetch(changed)) == original
    # The last permitted upstream value REALLY contributes via exported sample19.
    changed[:, 20] += 100
    assert extract_context(plan(), exported_fetch(changed)) != original
    assert requested_dependencies == [SampleSpan(6, 21)] * 3
    with pytest.raises(ValueError):
        extract_context(plan(cutoff_sample=20), exported_fetch(raw))
    assert len(requested_dependencies) == 3


def test_hole_rejection_does_not_fetch_and_impulse_does_not_use_later_values():
    calls = []
    source = np.zeros((1, 40))
    source[0, 19] = 1
    p = plan(taps=(1.0, 2.0, 3.0))
    actual = extract_context(p, fetcher(source, calls))
    assert actual.mean == (0.1,)  # Last output is1, no reversed/future FIR terms.
    with pytest.raises(ValueError):
        extract_context(
            plan(allowed=(SampleSpan(6, 10), SampleSpan(11, 21))), fetcher(source, calls)
        )
    assert calls == [(RUN, SampleSpan(8, 20))]


def test_nested_frozen_objects_are_revalidated_before_fetch():
    calls = []
    source = np.ones((1, 40))
    for field, invalid in (("start", -1), ("stop", float("inf")), ("stop", True)):
        forged = SampleSpan(6, 21)
        object.__setattr__(forged, field, invalid)
        with pytest.raises(ValueError):
            extract_context(plan(allowed=(forged,)), fetcher(source, calls))
    forged_run = replace(RUN)
    object.__setattr__(forged_run, "participant", "")
    with pytest.raises(ValueError):
        extract_context(plan(run=forged_run, timing=timing(run=forged_run)), fetcher(source, calls))
    p = plan()
    object.__setattr__(p.output, "stop", 100.5)
    with pytest.raises(ValueError):
        extract_context(p, fetcher(source, calls))
    assert calls == []


def test_feature_receipt_retains_clock_rate_and_cutoff():
    feature = extract_context(plan(), fetcher(np.ones((1, 40)), []))
    assert feature.clock_id == "generated-200Hz"
    assert feature.sfreq == 200
    assert feature.cutoff_sample == 21


def test_pairing_requires_known_same_clock_rate_identity_and_trial_onset():
    feature = extract_context(plan(), fetcher(np.ones((1, 40)), []))
    args = {"run": RUN, "clock_id": "generated-200Hz", "sfreq": 200, "cutoff_sample": 21}
    require_context_pairing(feature, **args)
    for change in (
        {"clock_id": None},
        {"clock_id": "other-200Hz"},
        {"sfreq": 1000},
        {"cutoff_sample": 22},
        {"run": replace(RUN, participant="other")},
    ):
        with pytest.raises(ValueError):
            require_context_pairing(feature, **(args | change))
    for change in (
        {"dependency": SampleSpan(0, 22)},
        {"std": (-1.0,)},
        {"mean": (float("nan"),)},
        {"mean_absolute_difference": (0.0, 1.0)},
        {"evidence_id": ""},
        {"sfreq": 200.0},
    ):
        with pytest.raises(ValueError):
            require_context_pairing(replace(feature, **change), **args)
