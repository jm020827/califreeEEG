from __future__ import annotations

import numpy as np

from cfeg.data.preprocess import (
    CanonicalChannelMap,
    PreprocessConfig,
    place_on_canonical_channels,
    preprocess_trial_with_qc,
)


def test_query_qc_is_measured_before_trial_zscore():
    channel_map = CanonicalChannelMap(
        unknown_id=0,
        name_to_id={"O1": 1, "O2": 2},
        id_to_name={1: "O1", 2: "O2"},
    )
    config = PreprocessConfig(
        target_sfreq=200.0,
        window_start_sec=0.0,
        window_duration_sec=1.0,
        bandpass_low_hz=None,
        bandpass_high_hz=None,
        normalize="per_trial_channel_zscore",
        c_max=2,
    )
    time = np.arange(200, dtype=np.float32) / 200.0
    signal = np.stack([np.sin(2 * np.pi * 10 * time), np.sin(2 * np.pi * 12 * time)]).astype(
        np.float32
    )

    first, _, _, _, first_qc = preprocess_trial_with_qc(
        signal, ["O1", "O2"], 200.0, config, channel_map
    )
    scaled, _, _, _, scaled_qc = preprocess_trial_with_qc(
        signal * 5.0, ["O1", "O2"], 200.0, config, channel_map
    )

    np.testing.assert_allclose(first, scaled, atol=2e-5)
    np.testing.assert_allclose(scaled_qc.signal_std, first_qc.signal_std * 5.0, rtol=1e-5)


def test_unknown_channel_identity_is_not_replaced_by_tensor_slot_id():
    channel_map = CanonicalChannelMap(
        unknown_id=0,
        name_to_id={"O2": 2},
        id_to_name={2: "O2"},
    )
    signal = np.ones((2, 8), dtype=np.float32)

    _, mask, slot_ids = place_on_canonical_channels(signal, ["mystery", "O2"], channel_map, c_max=4)

    assert mask.tolist() == [True, True, False, False]
    assert slot_ids == [0, 2, 0, 0]


def test_unknown_channel_does_not_occupy_a_later_known_channel_slot():
    channel_map = CanonicalChannelMap(
        unknown_id=0,
        name_to_id={"O1": 1},
        id_to_name={1: "O1"},
    )
    signal = np.stack(
        [np.full(8, 7.0, dtype=np.float32), np.full(8, 3.0, dtype=np.float32)]
    )

    placed, mask, slot_ids = place_on_canonical_channels(
        signal, ["mystery", "O1"], channel_map, c_max=3
    )

    assert mask.tolist() == [True, True, False]
    assert slot_ids == [1, 0, 0]
    np.testing.assert_array_equal(placed[0], signal[1])
    np.testing.assert_array_equal(placed[1], signal[0])


def test_duplicate_canonical_channel_aliases_fail_before_overwrite():
    channel_map = CanonicalChannelMap(
        unknown_id=0,
        name_to_id={"TP7": 1, "TP9": 1},
        id_to_name={1: "TP7"},
    )

    try:
        place_on_canonical_channels(
            np.ones((2, 8), dtype=np.float32),
            ["TP7", "TP9"],
            channel_map,
            c_max=3,
        )
    except ValueError as exc:
        assert "same canonical electrode ID" in str(exc)
    else:
        raise AssertionError("duplicate canonical aliases silently overwrote a channel")
