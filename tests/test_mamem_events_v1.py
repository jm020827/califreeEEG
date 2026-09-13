"""Generated event-cell fixtures only: no data reads, EEG, fitting, or network."""

import numpy as np
import pytest

from cfeg import mamem_events_v1 as events


class DescriptorPoison:
    def __float__(self):
        raise AssertionError("descriptor must not be converted")

    def __int__(self):
        raise AssertionError("descriptor must not be converted")

    def __array__(self, *args, **kwargs):
        raise AssertionError("descriptor must not be converted")


def fixture(block_order=(3, 0, 4, 1, 2), *, cell_arrays=False):
    labels = [0, 1, 2, 3, 4, 0, 1, 2] + [label for label in block_order for _ in range(3)]
    groups = []
    for index, label in enumerate(labels):
        period = 1000.0 / (2.0 * events.FREQUENCIES[label])
        relative = np.arange(int(5000.0 / period) + 1) * period
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


def parsed(groups, total_samples, include_metadata=True):
    return events.parse_main_trials(
        np.concatenate(groups, axis=1), total_samples, include_metadata=include_metadata
    )


@pytest.mark.parametrize("cell_arrays", [False, True])
def test_main_23_group_alignment_and_inferred_not_order_labels(cell_arrays):
    groups, total = fixture(cell_arrays=cell_arrays)
    actual = parsed(groups, total)
    assert len(actual) == 15
    assert [row["group_index"] for row in actual] == list(range(8, 23))
    assert [row["label"] for row in actual] == [3] * 3 + [0] * 3 + [4] * 3 + [1] * 3 + [2] * 3
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
        assert row["metadata"].shape == (2,)
        assert row["metadata"].dtype == np.float64
        assert np.isfinite(row["metadata"]).all()


def test_one_based_half_open_selection_and_support_only_m(monkeypatch):
    groups, total = fixture()
    observed = []

    def capture(times):
        observed.append(times.copy())
        return np.zeros(2, dtype=np.float64)

    monkeypatch.setattr(events, "marker_features", capture)
    actual = parsed(groups, total)
    assert len(observed) == 15  # No adaptation feature extraction.
    for group, row, times in zip(groups[8:], actual, observed):
        sample1 = np.array(list(group[3]), dtype=np.int64)
        expected_start = int(sample1[0]) - 1 + 250
        expected_end = expected_start + 500
        selected = ((sample1 - 1) >= expected_start) & ((sample1 - 1) < expected_end)
        assert row["start0"] == expected_start
        assert row["end0"] == expected_end
        assert row["event_count"] == int(selected.sum())
        np.testing.assert_array_equal(times, np.array(list(group[1]), dtype=float)[selected])
    # First main group is 10 Hz: markers exactly at both window boundaries.
    assert actual[0]["event_count"] == 40
    assert observed[0][0] == groups[8][1, 20]
    assert observed[0][-1] == groups[8][1, 59]


def test_inclusive_last_sample_containment_and_no_following_group_required():
    groups, total = fixture(block_order=(0, 4, 1, 2, 3))
    group = groups[-1][:, :61].copy()  # Final group is 10 Hz.
    first = int(group[3, 0])
    group[3, -1] = first - 1 + 750  # last_sample == end0 is sufficient.
    groups[-1] = group
    actual = parsed(groups, total)
    assert actual[-1]["end0"] == group[3, -1]
    assert actual[-1]["event_count"] == 41
    groups[-1][3, -1] -= 1
    with pytest.raises(ValueError, match="^incomplete_analysis_window$"):
        parsed(groups, total)


def test_query_never_calls_metadata_extractor(monkeypatch):
    groups, total = fixture()

    def forbidden(*args):
        raise AssertionError("query must not extract metadata")

    monkeypatch.setattr(events, "marker_features", forbidden)
    actual = parsed(groups, total, include_metadata=False)
    assert len(actual) == 15 and all(row["metadata"] is None for row in actual)


@pytest.mark.parametrize("index", [0, 8, 22])
def test_short_groups_stop_without_dropping_or_misalignment(index, monkeypatch):
    groups, total = fixture()
    groups[index] = groups[index][:, :8]
    monkeypatch.setattr(events, "marker_features", lambda _: pytest.fail("premature M call"))
    with pytest.raises(ValueError, match="^incomplete_analysis_window$"):
        parsed(groups, total)


@pytest.mark.parametrize("count", [22, 24])
def test_exact_group_count(count):
    groups, total = fixture()
    if count == 22:
        groups = groups[:-1]
    else:
        extra = groups[-1].copy()
        extra[1] += 10000
        extra[3] += 2000
        groups.append(extra)
        total += 2000
    with pytest.raises(ValueError, match="^expected_23_groups$"):
        parsed(groups, total)


@pytest.mark.parametrize("gap,valid", [(2000.0, False), (2000.001, True)])
def test_group_split_is_strictly_greater_than_2000(gap, valid):
    groups, total = fixture()
    groups[1][1] += float(groups[0][1, -1]) + gap - float(groups[1][1, 0])
    if valid:
        assert len(parsed(groups, total)) == 15
    else:
        with pytest.raises(ValueError, match="^expected_23_groups$"):
            parsed(groups, total)


def test_exactly_three_per_class_and_single_contiguous_block_required(monkeypatch):
    monkeypatch.setattr(events, "marker_features", lambda _: pytest.fail("premature M call"))
    groups, total = fixture(block_order=(0, 0, 2, 3, 4))
    with pytest.raises(ValueError, match="^main_class_coverage_mismatch$"):
        parsed(groups, total)
    groups, total = fixture()
    # Swap frequency periods, preserving chronology and three/class but breaking blocks.
    for left, right in [(9, 12)]:
        for index, other in [(left, right), (right, left)]:
            label = 0 if other == right else 3
            group = groups[index]
            group[1] = float(group[1, 0]) + np.arange(group.shape[1]) * (
                1000.0 / (2.0 * events.FREQUENCIES[label])
            )
    with pytest.raises(ValueError, match="^main_class_blocks_mismatch$"):
        parsed(groups, total)


@pytest.mark.parametrize("label", range(5))
def test_closed_guardbands_with_no_nearest_fallback(label):
    frequencies = events.FREQUENCIES
    neighbors = np.abs(frequencies - frequencies[label])
    width = neighbors[neighbors > 0].min() / 4
    lower, upper = frequencies[label] - width, frequencies[label] + width
    for frequency in (lower, frequencies[label], upper):
        assert events._frequency_label(frequency) == label
    for frequency in (np.nextafter(lower, -np.inf), np.nextafter(upper, np.inf)):
        with pytest.raises(ValueError, match="^frequency_outside_unique_guardband$"):
            events._frequency_label(frequency)


@pytest.mark.parametrize("frequency", [6.0, 9.0, 11.0, 9.2])
def test_legacy_integer_keys_never_repair_primary_decoder(frequency):
    groups, total = fixture()
    group = groups[0]
    group[1] = float(group[1, 0]) + np.arange(group.shape[1]) * 1000 / (2 * frequency)
    with pytest.raises(ValueError, match="^frequency_outside_unique_guardband$"):
        parsed(groups, total)


@pytest.mark.parametrize("index", [0, 8, 22])
def test_invalid_adaptation_or_main_frequency_stops_before_metadata(index, monkeypatch):
    groups, total = fixture()
    group = groups[index]
    group[1] = float(group[1, 0]) + np.arange(group.shape[1]) * 1000 / (2 * 9.2)
    monkeypatch.setattr(events, "marker_features", lambda _: pytest.fail("premature M call"))
    with pytest.raises(ValueError, match="^frequency_outside_unique_guardband$"):
        parsed(groups, total)


@pytest.mark.parametrize("row", [1, 3])
@pytest.mark.parametrize("bad", [np.nan, np.inf, True, "7", 1 + 0j, np.array([1, 2])])
def test_only_numeric_scalar_event_cells_accepted(row, bad):
    groups, total = fixture()
    groups[8][row, 10] = bad
    with pytest.raises(ValueError, match="^(invalid_numeric_cell|nonfinite_numeric_cell)$"):
        parsed(groups, total)


@pytest.mark.parametrize("bad", [0, -1, 101.5, 1000000])
def test_sample_indices_must_be_positive_integers_inside_file(bad):
    groups, total = fixture()
    groups[0][3, 0] = bad
    with pytest.raises(ValueError, match="^invalid_sample_index$"):
        parsed(groups, total)


@pytest.mark.parametrize(
    "row,reason", [(1, "nonincreasing_timestamps"), (3, "nonincreasing_samples")]
)
@pytest.mark.parametrize("backwards", [False, True])
def test_event_rows_strictly_increasing(row, reason, backwards):
    groups, total = fixture()
    groups[8][row, 10] = groups[8][row, 9] - int(backwards)
    with pytest.raises(ValueError, match=f"^{reason}$"):
        parsed(groups, total)


def test_full_five_second_cost_must_fit_even_when_analysis_window_fits():
    groups, total = fixture(block_order=(0, 4, 1, 2, 3))
    groups[-1] = groups[-1][:, :81]
    total = int(groups[-1][3, 0]) - 1 + 1100
    with pytest.raises(ValueError, match="^incomplete_trial_cost_window$"):
        parsed(groups, total)


def test_insufficient_window_events_fail_even_with_window_containment():
    groups, total = fixture()
    group = groups[8]
    # Time-based labels remain valid; generated sample locations are intentionally sparse.
    sample0 = int(group[3, 0])
    group[3] = np.r_[
        sample0 + np.arange(20),
        sample0 + 250 + np.arange(3),
        sample0 + 750 + np.arange(group.shape[1] - 23),
    ]
    with pytest.raises(ValueError, match="^insufficient_window_events$"):
        parsed(groups, total)


def test_extra_din_row_poison_and_wrong_shapes_rejected_without_descriptor_access():
    groups, total = fixture()
    din = np.concatenate(groups, axis=1)
    extra = np.empty((1, din.shape[1]), dtype=object)
    for column in range(extra.shape[1]):
        extra[0, column] = DescriptorPoison()
    for malformed in (
        np.concatenate([din, extra]),
        din[:3],
        din.reshape(-1),
        np.empty((4, 0), dtype=object),
        np.empty((4, 10001), dtype=object),
        np.zeros((4, 100)),
        [[1, 2]],
    ):
        with pytest.raises(ValueError, match="^invalid_din_shape_or_dtype$"):
            events.parse_main_trials(malformed, total)


@pytest.mark.parametrize("total", [0, -1, 100.5, True, "100", np.iinfo(np.int64).max + 1])
def test_bad_total_samples(total):
    with pytest.raises(ValueError, match="^invalid_total_samples$"):
        events.parse_main_trials(np.empty((4, 1), dtype=object), total)


@pytest.mark.parametrize("flag", [None, 1, "False"])
def test_metadata_flag_must_be_boolean(flag):
    with pytest.raises(ValueError, match="^invalid_metadata_flag$"):
        events.parse_main_trials(np.empty((4, 1), dtype=object), 100, include_metadata=flag)


def test_parser_does_not_mutate_numeric_cells_or_use_descriptors():
    groups, total = fixture(cell_arrays=True)
    din = np.concatenate(groups, axis=1)
    original = [[cell.copy() for cell in din[row]] for row in (1, 3)]
    first = events.parse_main_trials(din, total)
    din[0] = "not a label"
    din[2] = np.nan
    second = events.parse_main_trials(din, total)
    for row, saved in zip((1, 3), original):
        for cell, before in zip(din[row], saved):
            np.testing.assert_array_equal(cell, before)
    for before, after in zip(first, second):
        for key in before:
            np.testing.assert_equal(before[key], after[key])


@pytest.mark.parametrize("state", [0, 1])
@pytest.mark.parametrize("schedule", [[4, 4, 5, 4, 3], [2, 3]])
@pytest.mark.parametrize("offset", [0.0, 1e6])
def test_existing_frame_residual_definition_schedule_and_origin(state, schedule, offset):
    intervals = np.resize(schedule, 16) * (1000 / 60) + state * np.resize([-2.0, 2.0], 16)
    times = offset + np.r_[0, np.cumsum(intervals)]
    original = times.copy()
    result = events.marker_features(times)
    np.testing.assert_allclose(result, [2, -1] if state else [0, 0], atol=1e-9, rtol=0)
    np.testing.assert_array_equal(times, original)
    assert result.dtype == np.float64 and result.shape == (2,)


def test_marker_mad_and_pearson_match_independent_nonconstant_example():
    residuals = np.array([-2.0, 1.0, 3.0, -1.0, 2.0, -3.0])
    times = np.r_[0.0, np.cumsum(4 * (1000 / 60) + residuals)]
    expected_mad = np.median(np.abs(residuals - np.median(residuals)))
    a = residuals[:-1] - residuals[:-1].mean()
    b = residuals[1:] - residuals[1:].mean()
    expected_lag = np.dot(a, b) / np.sqrt(np.dot(a, a) * np.dot(b, b))
    np.testing.assert_allclose(
        events.marker_features(times), [expected_mad, expected_lag], atol=1e-12, rtol=0
    )


def test_marker_numerical_zero_and_constant_residual():
    for residuals in (np.array([0, 1e-10, -1e-10, 2e-10]), np.full(4, 2.0)):
        times = np.r_[0.0, np.cumsum(4 * (1000 / 60) + residuals)]
        np.testing.assert_allclose(events.marker_features(times), [0, 0], atol=1e-12, rtol=0)


@pytest.mark.parametrize(
    "bad",
    [
        [1, 2, 3],
        [1, 2, 2, 4],
        [1, 3, 2, 4],
        [1, 2, np.nan, 4],
        [1, 2, np.inf, 4],
        [[1, 2, 3, 4]],
        [True] * 4,
        ["1", "2", "3", "4"],
        [1j, 2j, 3j, 4j],
        [-1.7e308, 1.7e308, 1.75e308, 1.79e308],
    ],
)
def test_marker_bad_inputs_are_static_value_errors(bad):
    with pytest.raises(
        ValueError,
        match="^(invalid_timestamps|nonfinite_timestamps|"
        "nonincreasing_timestamps|invalid_marker_features)$",
    ):
        events.marker_features(bad)
