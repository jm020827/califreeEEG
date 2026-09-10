"""Generated schedules only: selected support is not actual collection burden."""

from dataclasses import FrozenInstanceError, replace

import numpy as np
import pytest

from cfeg.analysis.calibration_collection_cost import (
    CollectedTrial,
    CollectionCost,
    balanced_prefix_cost,
)
from cfeg.data.acquisition_context_contract import RunKey

RUN = RunKey("generated", "p1", "day1", "low", "session1")
CLASSES = (0, 1, 2, 3)


def schedule(labels=(0, 0, 1, 2, 3, 1, 2, 3), *, run=RUN):
    return tuple(
        CollectedTrial(run, f"t{i}", label, 20 * i + 10, 20 * i + 20)
        for i, label in enumerate(labels)
    )


def cost(trials=None, **kwargs):
    parameters = {"run": RUN, "classes": CLASSES, "k": 1, "sfreq": 10, "collection_start_sample": 0}
    parameters.update(kwargs)
    return balanced_prefix_cost(schedule() if trials is None else trials, **parameters)


def test_repeated_class_prefix_and_rest_cost_not_selected_label_count():
    result = cost(setup_seconds=3, extra_baseline_seconds=2, clock_id="export-grid-v1")
    assert result.attained
    assert result.selected_trial_ids == ("t0", "t2", "t3", "t4")
    assert result.selected_labels == 4
    assert result.collected_trial_ids == ("t0", "t1", "t2", "t3", "t4")
    assert result.collected_trials == result.collected_labels == 5
    assert result.elapsed_recording_seconds == 10.0
    assert result.total_seconds == 15.0
    assert (result.run, result.classes, result.k) == (RUN, CLASSES, 1)
    assert (result.sfreq, result.collection_start_sample, result.clock_id) == (
        10,
        0,
        "export-grid-v1",
    )


def test_changing_order_changes_collection_not_balanced_support_budget():
    balanced = cost(schedule((0, 1, 2, 3, 0)), setup_seconds=0, extra_baseline_seconds=0)
    repeated = cost(schedule((0, 0, 1, 2, 3)), setup_seconds=0, extra_baseline_seconds=0)
    assert balanced.selected_labels == repeated.selected_labels == 4
    assert balanced.collected_trials == 4 and repeated.collected_trials == 5
    assert balanced.total_seconds == 8 and repeated.total_seconds == 10


def test_increasing_k_has_nested_selection_and_prefix_and_first_k_rule():
    one, two = cost(), cost(k=2)
    assert set(one.selected_trial_ids) < set(two.selected_trial_ids)
    assert two.selected_trial_ids == tuple(f"t{i}" for i in range(8))
    assert two.collected_trial_ids[: one.collected_trials] == one.collected_trial_ids
    assert two.selected_labels == two.collected_labels == 8
    assert two.elapsed_recording_seconds == 16


def test_unattained_preserves_observed_schedule_but_not_attainment_cost():
    result = cost(k=3, setup_seconds=2, extra_baseline_seconds=4)
    assert not result.attained
    assert result.selected_trial_ids is result.selected_labels is result.total_seconds is None
    assert result.collected_trial_ids == tuple(f"t{i}" for i in range(8))
    assert result.collected_trials == result.collected_labels == 8
    assert result.elapsed_recording_seconds == 16
    assert result.setup_seconds == 2 and result.extra_baseline_seconds == 4
    missing_class = cost(schedule((0, 0, 1, 2)))
    assert not missing_class.attained and missing_class.collected_labels == 4


def test_empty_unattained_and_k_zero_have_distinct_semantics():
    missing = cost((), k=1, setup_seconds=0, extra_baseline_seconds=0)
    assert not missing.attained and missing.selected_labels is None
    assert missing.total_seconds is None and missing.elapsed_recording_seconds == 0
    assert missing.collected_trial_ids == ()
    anchor = cost(k=0, setup_seconds=3, extra_baseline_seconds=4)
    assert anchor.attained and anchor.selected_trial_ids == anchor.collected_trial_ids == ()
    assert anchor.selected_labels == anchor.collected_trials == anchor.collected_labels == 0
    assert anchor.elapsed_recording_seconds == 0 and anchor.total_seconds == 7
    assert cost((), k=0, setup_seconds=3, extra_baseline_seconds=4) == anchor


def test_unknown_setup_or_baseline_is_not_zero_including_k_zero():
    for k in (0, 1):
        for setup, baseline in ((None, None), (None, 0), (0, None)):
            result = cost(k=k, setup_seconds=setup, extra_baseline_seconds=baseline)
            assert result.total_seconds is None
            assert result.setup_seconds == setup and result.extra_baseline_seconds == baseline


def test_nonzero_collection_start_and_adjacent_half_open_trials():
    result = cost(collection_start_sample=10, setup_seconds=0, extra_baseline_seconds=0)
    assert result.elapsed_recording_seconds == 9
    touching = (
        CollectedTrial(RUN, "a", 7, 100, 110),
        CollectedTrial(RUN, "b", 11, 110, 130),
    )
    result = cost(touching, classes=(7, 11), collection_start_sample=100, sfreq=10)
    assert result.selected_trial_ids == ("a", "b") and result.elapsed_recording_seconds == 3


def test_invalid_scalar_arguments_are_rejected_without_coercion():
    for name, bad_values in (
        ("k", (-1, True, 1.0, np.int64(1), None)),
        ("sfreq", (0, -1, True, 10.0, np.int64(10), float("nan"))),
        ("collection_start_sample", (-1, True, 0.0, np.int64(0))),
        ("run", (None, "generated")),
        ("clock_id", ("", " padded", 0, True)),
    ):
        for value in bad_values:
            with pytest.raises(ValueError):
                cost(**{name: value})


def test_exact_class_contract_and_out_of_task_labels():
    for classes in ((), (0,), (0, 0), [0, 1, 2, 3], 4, (0, True), (0, 1.0), (0, -1)):
        with pytest.raises(ValueError):
            cost(classes=classes)
    with pytest.raises(ValueError, match="outside"):
        cost(classes=(0, 1, 2))
    assert cost(classes=(3, 1, 0, 2)).selected_trial_ids == cost().selected_trial_ids


def test_optional_time_values_reject_nonfinite_negative_and_nonreal_types():
    for name in ("setup_seconds", "extra_baseline_seconds"):
        for value in (
            -1,
            float("nan"),
            float("inf"),
            -float("inf"),
            True,
            "1",
            1j,
            np.float64(1),
            10**400,
        ):
            with pytest.raises(ValueError):
                cost(**{name: value})
    with pytest.raises(ValueError, match="finite|overflow"):
        cost(setup_seconds=1e308, extra_baseline_seconds=1e308)


def test_trial_dataclass_strict_types_time_and_frozen_values():
    original = schedule()[0]
    for field, invalid in (
        ("run", "run"),
        ("trial_id", ""),
        ("trial_id", " padded"),
        ("trial_id", 1),
        ("label", True),
        ("label", -1),
        ("label", np.int64(0)),
        ("onset_sample", 10.0),
        ("onset_sample", -1),
        ("onset_sample", True),
        ("end_sample", 10),
        ("end_sample", float("nan")),
        ("end_sample", 0),
    ):
        with pytest.raises(ValueError):
            replace(original, **{field: invalid})
    with pytest.raises(FrozenInstanceError):
        original.label = 1


def test_run_mismatch_for_every_identity_axis_and_invalid_record_containers():
    for axis in ("dataset", "participant", "day", "band", "session"):
        other = replace(RUN, **{axis: "other"})
        trials = (*schedule()[:-1], replace(schedule()[-1], run=other))
        with pytest.raises(ValueError, match="mixed run"):
            cost(trials)
    for trials in (iter(schedule()), "trials", np.array([1]), ("bad",)):
        with pytest.raises(ValueError):
            cost(trials)


def test_invalid_schedule_is_not_sorted_or_hidden_by_early_attainment():
    trials = schedule()
    invalids = (
        (trials[1], trials[0], *trials[2:]),
        (trials[0], replace(trials[1], onset_sample=trials[0].onset_sample), *trials[2:]),
        (trials[0], replace(trials[1], onset_sample=trials[0].end_sample - 1), *trials[2:]),
        (*trials[:-1], replace(trials[-1], trial_id=trials[0].trial_id)),
    )
    for candidate in invalids:
        for k in (0, 1, 3):
            with pytest.raises(ValueError):
                cost(candidate, k=k)
    with pytest.raises(ValueError, match="starts after"):
        cost(collection_start_sample=11)


def test_no_mutation_no_extra_collected_trials_and_finite_elapsed():
    trials = list(schedule())
    before = tuple(trials)
    result = cost(trials)
    assert tuple(trials) == before and len(trials) == 8
    assert result.collected_trial_ids[-1] == "t4"
    with pytest.raises(FrozenInstanceError):
        result.attained = False
    gigantic = (CollectedTrial(RUN, "a", 0, 0, 1), CollectedTrial(RUN, "b", 1, 2, 10**400))
    with pytest.raises(ValueError, match="overflow"):
        cost(gigantic, classes=(0, 1), sfreq=1)


def test_result_contract_rejects_forged_counts_missingness_and_identity():
    original = cost(setup_seconds=0, extra_baseline_seconds=0)
    assert isinstance(original, CollectionCost)
    corruptions = (
        {"run": None},
        {"classes": (0,)},
        {"k": True},
        {"attained": 1},
        {"sfreq": 0},
        {"collection_start_sample": -1},
        {"clock_id": " padded"},
        {"selected_labels": 5},
        {"selected_trial_ids": ("t4", "t2", "t3", "t0")},
        {"collected_trials": 4},
        {"collected_labels": 4},
        {"collected_trial_ids": ("t0", "t0", "t2", "t3", "t4")},
        {"elapsed_recording_seconds": 0},
        {"total_seconds": None},
        {"setup_seconds": None},
        {"total_seconds": float("inf")},
    )
    for corruption in corruptions:
        with pytest.raises(ValueError):
            replace(original, **corruption)
    missing = cost(k=3)
    for corruption in ({"selected_labels": 0}, {"selected_trial_ids": ()}, {"total_seconds": 0}):
        with pytest.raises(ValueError):
            replace(missing, **corruption)


def test_forged_nested_run_or_trial_fields_do_not_pass_schema_validation():
    forged_run = replace(RUN)
    object.__setattr__(forged_run, "day", "")
    with pytest.raises(ValueError):
        cost(run=forged_run)
    nested = replace(schedule()[0])
    object.__setattr__(nested, "run", forged_run)
    with pytest.raises(ValueError):
        cost((nested, *schedule()[1:]))
    for field, invalid in (("label", True), ("end_sample", 1), ("trial_id", "")):
        trial = replace(schedule()[0])
        object.__setattr__(trial, field, invalid)
        with pytest.raises(ValueError):
            cost((trial, *schedule()[1:]))
    with pytest.raises(ValueError):
        cost((object.__new__(CollectedTrial),))
