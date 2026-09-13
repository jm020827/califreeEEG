"""Generated DIN-cell algebra only: no raw data, EEG, gates, or file reads."""

import numpy as np
import pytest

from cfeg import mamem_events_v1 as v1
from cfeg import mamem_events_v2 as events

PERIODS = np.array([75.5, 66.5, 58.5, 52.5, 43.5])
KEYS = [6, 7, 8, 9, 11]


class DescriptorPoison:
    def __float__(self):
        raise AssertionError("descriptor conversion forbidden")

    def __int__(self):
        raise AssertionError("descriptor conversion forbidden")

    def __array__(self, *args, **kwargs):
        raise AssertionError("descriptor conversion forbidden")


def fixture(block_order=(3, 0, 4, 1, 2), *, cell_arrays=False):
    labels = [0, 1, 2, 3, 4, 0, 1, 2] + [label for label in block_order for _ in range(3)]
    groups = []
    for index, label in enumerate(labels):
        relative = np.arange(int(5000.0 / PERIODS[label]) + 1) * PERIODS[label]
        times = 1000.0 + 10000.0 * index + relative
        samples = 101 + 2000 * index + np.rint(relative / 4).astype(np.int64)
        group = np.empty((4, len(times)), dtype=object)
        for column, (timestamp, sample) in enumerate(zip(times, samples)):
            group[0, column] = DescriptorPoison()
            group[2, column] = DescriptorPoison()
            group[1, column] = np.array([[timestamp]]) if cell_arrays else timestamp
            group[3, column] = np.array([[sample]]) if cell_arrays else sample
        groups.append(group)
    return groups, 101 + 2000 * (len(groups) - 1) + 1750


def parsed(groups, total, *, include_metadata=True):
    return events.parse_main_trials(np.concatenate(groups, axis=1), total, include_metadata)


def replace_period(group, period):
    group[1] = float(group[1, 0]) + np.arange(group.shape[1]) * period


@pytest.mark.parametrize("cell_arrays", [False, True])
@pytest.mark.parametrize("order", [(3, 0, 4, 1, 2), (4, 2, 0, 3, 1)])
def test_exact_api_23_groups_three_per_class_and_no_order_labels(cell_arrays, order):
    groups, total = fixture(order, cell_arrays=cell_arrays)
    actual = parsed(groups, total)
    assert len(actual) == 15
    assert [row["group_index"] for row in actual] == list(range(8, 23))
    assert [row["label"] for row in actual] == [label for label in order for _ in range(3)]
    for row in actual:
        assert set(row) == {
            "group_index",
            "label",
            "start0",
            "end0",
            "event_count",
            "metadata",
            "trial_end0",
        }
        assert all(type(row[key]) is int for key in row if key != "metadata")
        assert row["end0"] - row["start0"] == 500
        assert row["trial_end0"] - row["start0"] == 1000
        assert row["metadata"].shape == (2,) and row["metadata"].dtype == np.float64
        assert np.isfinite(row["metadata"]).all()


@pytest.mark.parametrize("label,period", list(enumerate(PERIODS)))
def test_exact_documented_double_floor_keys_are_not_reference_frequencies(label, period):
    times = 1000.0 + np.arange(12) * period
    period_int = int(sum(np.diff(times)) // (times.size - 1))
    key = 1000 // (2 * period_int)
    assert key == KEYS[label]
    assert events._group_label(times) == label
    assert events.FREQUENCIES[label] == [6.66, 7.50, 8.57, 10.00, 12.00][label]


@pytest.mark.parametrize("period", [50.0, 1000.0 / 24.0, 41.6667])
@pytest.mark.parametrize("index", [0, 8, 22])
def test_exact_nominal_ten_twelve_hz_unsupported_no_band_or_order_repair(
    period, index, monkeypatch
):
    groups, total = fixture()
    replace_period(groups[index], period)
    monkeypatch.setattr(events, "marker_features", lambda _: pytest.fail("premature metadata"))
    with pytest.raises(ValueError, match="^unsupported_frequency_key$"):
        parsed(groups, total)


@pytest.mark.parametrize("label,period,simple_key", [(1, 71.75, 6), (2, 62.75, 7), (3, 55.75, 8)])
def test_floor_of_interval_before_reciprocal_is_not_floor_of_continuous_frequency(
    label, period, simple_key
):
    groups, total = fixture((0, 1, 2, 3, 4))
    index = 8 + 3 * label
    replace_period(groups[index], period)
    assert int(np.floor(1000.0 / (2 * period))) == simple_key
    assert simple_key != KEYS[label]
    assert parsed(groups, total)[index - 8]["label"] == label


def test_mean_interval_not_median_or_mean_of_integer_intervals():
    intervals = np.array([62.875, 62.875, 62.875, 64.875])
    assert np.mean(intervals) == 63.375
    assert np.median(intervals) == 62.875
    # Median would yield key 8/class 2, but the frozen mean gives key 7/class 1.
    assert events._group_label(np.r_[0, np.cumsum(intervals)]) == 1


@pytest.mark.parametrize("period", [0.5, 0.999])
def test_zero_integer_period_stops_before_division(period, monkeypatch):
    groups, total = fixture()
    replace_period(groups[0], period)
    monkeypatch.setattr(events, "marker_features", lambda _: pytest.fail("premature metadata"))
    with pytest.raises(ValueError, match="^nonpositive_integer_period$"):
        parsed(groups, total)


def test_one_based_half_open_selection_and_support_only_extractor(monkeypatch):
    groups, total = fixture()
    group = groups[8]
    first = int(group[3, 0])
    group[3] = first + np.arange(group.shape[1]) * 10
    captured = []

    def capture(times):
        captured.append(times.copy())
        return np.zeros(2, dtype=np.float64)

    monkeypatch.setattr(events, "marker_features", capture)
    actual = parsed(groups, total)
    assert len(captured) == 15
    assert actual[0]["start0"] == first - 1 + 250
    assert actual[0]["end0"] == first - 1 + 750
    assert actual[0]["event_count"] == 50
    np.testing.assert_array_equal(captured[0], np.array(list(group[1, 25:75]), dtype=float))
    for source, row, times in zip(groups[8:], actual, captured):
        samples0 = np.array(list(source[3]), dtype=np.int64) - 1
        inside = (samples0 >= row["start0"]) & (samples0 < row["end0"])
        assert row["event_count"] == int(inside.sum())
        np.testing.assert_array_equal(times, np.array(list(source[1]), dtype=float)[inside])


def test_final_inclusive_marker_containment_without_next_group():
    groups, total = fixture((0, 4, 1, 2, 3))
    group = groups[-1][:, :58].copy()
    group[3, -1] = int(group[3, 0]) - 1 + 750
    groups[-1] = group
    assert parsed(groups, total)[-1]["end0"] == group[3, -1]
    group[3, -1] -= 1
    with pytest.raises(ValueError, match="^incomplete_analysis_window$"):
        parsed(groups, total)


def test_query_never_calls_metadata_extractor(monkeypatch):
    groups, total = fixture()
    monkeypatch.setattr(events, "marker_features", lambda _: pytest.fail("query metadata"))
    actual = parsed(groups, total, include_metadata=False)
    assert len(actual) == 15 and all(row["metadata"] is None for row in actual)


@pytest.mark.parametrize("index", [0, 8, 22])
@pytest.mark.parametrize(
    "length,reason", [(1, "incomplete_group"), (8, "incomplete_analysis_window")]
)
def test_incomplete_group_stops_no_skipping_no_shift(index, length, reason, monkeypatch):
    groups, total = fixture()
    groups[index] = groups[index][:, :length]
    monkeypatch.setattr(events, "marker_features", lambda _: pytest.fail("premature metadata"))
    with pytest.raises(ValueError, match=f"^{reason}$"):
        parsed(groups, total)


@pytest.mark.parametrize("count", [22, 24])
def test_group_count_failure_precedes_metadata(count, monkeypatch):
    groups, total = fixture()
    if count == 22:
        groups = groups[:-1]
    else:
        extra = groups[-1].copy()
        extra[1] += 10000
        extra[3] += 2000
        groups.append(extra)
        total += 2000
    monkeypatch.setattr(events, "marker_features", lambda _: pytest.fail("premature metadata"))
    with pytest.raises(ValueError, match="^expected_23_groups$"):
        parsed(groups, total)


@pytest.mark.parametrize("gap,valid", [(2000.0, False), (2000.001, True)])
def test_strict_split_threshold(gap, valid):
    groups, total = fixture()
    groups[1][1] += float(groups[0][1, -1]) + gap - float(groups[1][1, 0])
    if valid:
        assert len(parsed(groups, total)) == 15
    else:
        with pytest.raises(ValueError, match="^expected_23_groups$"):
            parsed(groups, total)


def test_class_coverage_and_contiguous_blocks_before_metadata(monkeypatch):
    monkeypatch.setattr(events, "marker_features", lambda _: pytest.fail("premature metadata"))
    groups, total = fixture((0, 0, 2, 3, 4))
    with pytest.raises(ValueError, match="^main_class_coverage_mismatch$"):
        parsed(groups, total)
    groups, total = fixture()
    replace_period(groups[9], PERIODS[0])
    replace_period(groups[12], PERIODS[3])
    with pytest.raises(ValueError, match="^main_class_blocks_mismatch$"):
        parsed(groups, total)


@pytest.mark.parametrize("row", [1, 3])
@pytest.mark.parametrize("bad", [np.nan, np.inf, True, "7", 1 + 0j, np.array([1, 2])])
def test_only_finite_real_scalar_cells(row, bad):
    groups, total = fixture()
    groups[8][row, 10] = bad
    with pytest.raises(ValueError, match="^(invalid_numeric_cell|nonfinite_numeric_cell)$"):
        parsed(groups, total)


@pytest.mark.parametrize("bad", [0, -1, 101.5, 1000000])
def test_positive_integer_sample_bounds(bad):
    groups, total = fixture()
    groups[0][3, 0] = bad
    with pytest.raises(ValueError, match="^invalid_sample_index$"):
        parsed(groups, total)


@pytest.mark.parametrize(
    "row,reason", [(1, "nonincreasing_timestamps"), (3, "nonincreasing_samples")]
)
@pytest.mark.parametrize("backwards", [False, True])
def test_strict_event_monotonicity(row, reason, backwards):
    groups, total = fixture()
    groups[8][row, 10] = groups[8][row, 9] - int(backwards)
    with pytest.raises(ValueError, match=f"^{reason}$"):
        parsed(groups, total)


def test_full_trial_cost_window_must_fit():
    groups, total = fixture((0, 4, 1, 2, 3))
    groups[-1] = groups[-1][:, :81]
    total = int(groups[-1][3, 0]) - 1 + 1100
    with pytest.raises(ValueError, match="^incomplete_trial_cost_window$"):
        parsed(groups, total)


def test_at_least_four_selected_events_required():
    groups, total = fixture()
    group = groups[8]
    first = int(group[3, 0])
    group[3] = np.r_[
        first + np.arange(20),
        first + 250 + np.arange(3),
        first + 750 + np.arange(group.shape[1] - 23),
    ]
    with pytest.raises(ValueError, match="^insufficient_window_events$"):
        parsed(groups, total)


def test_schema_and_extra_descriptor_row_poison():
    groups, total = fixture()
    din = np.concatenate(groups, axis=1)
    extra = np.empty((1, din.shape[1]), dtype=object)
    for column in range(extra.shape[1]):
        extra[0, column] = DescriptorPoison()
    for bad in (
        np.concatenate([din, extra]),
        din[:3],
        din.reshape(-1),
        np.empty((4, 0), dtype=object),
        np.empty((4, 10001), dtype=object),
        np.zeros((4, 100)),
        [[1, 2]],
    ):
        with pytest.raises(ValueError, match="^invalid_din_shape_or_dtype$"):
            events.parse_main_trials(bad, total)


@pytest.mark.parametrize("total", [0, -1, 100.5, True, "100", np.iinfo(np.int64).max + 1])
def test_total_samples_type_bounds(total):
    with pytest.raises(ValueError, match="^invalid_total_samples$"):
        events.parse_main_trials(np.empty((4, 1), dtype=object), total)


@pytest.mark.parametrize("flag", [None, 1, "False"])
def test_boolean_metadata_flag(flag):
    with pytest.raises(ValueError, match="^invalid_metadata_flag$"):
        events.parse_main_trials(np.empty((4, 1), dtype=object), 100, flag)


def test_inputs_unmodified_and_descriptor_values_irrelevant():
    groups, total = fixture(cell_arrays=True)
    din = np.concatenate(groups, axis=1)
    numeric_before = [[cell.copy() for cell in din[row]] for row in (1, 3)]
    before = events.parse_main_trials(din, total)
    din[0] = "not a label"
    din[2] = np.nan
    after = events.parse_main_trials(din, total)
    for row, saved in zip((1, 3), numeric_before):
        for cell, original in zip(din[row], saved):
            np.testing.assert_array_equal(cell, original)
    for first, second in zip(before, after):
        for key in first:
            np.testing.assert_equal(first[key], second[key])


def test_reuses_frozen_v1_metadata_and_nominal_reference_api():
    assert events.marker_features is v1.marker_features
    assert events.FREQUENCIES is v1.FREQUENCIES
    assert events.FREQUENCIES.dtype == np.float64 and not events.FREQUENCIES.flags.writeable
    times = np.r_[0, np.cumsum(4 * (1000 / 60) + np.resize([-2.0, 2.0], 16))]
    np.testing.assert_allclose(events.marker_features(times), [2.0, -1.0], atol=1e-9, rtol=0)
