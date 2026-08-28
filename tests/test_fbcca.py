from __future__ import annotations

import numpy as np
import pytest

from cfeg.baselines import (
    FilterBankConfig,
    cca_score,
    make_reference_signals,
    predict_cca,
    predict_fbcca,
)


def _synthetic_ssvep(
    frequency: float,
    *,
    sfreq: float = 250.0,
    duration: float = 2.0,
) -> np.ndarray:
    time = np.arange(round(sfreq * duration), dtype=np.float64) / sfreq
    sources = np.vstack(
        [
            np.sin(2 * np.pi * frequency * time + 0.4),
            0.65 * np.sin(2 * np.pi * 2 * frequency * time + 1.1),
            0.35 * np.cos(2 * np.pi * 3 * frequency * time - 0.3),
        ]
    )
    spatial_mixing = np.asarray(
        [
            [1.0, 0.2, -0.1],
            [0.4, 0.9, 0.25],
            [-0.2, 0.5, 1.0],
            [0.8, -0.25, 0.2],
        ]
    )
    rng = np.random.default_rng(17)
    slow_drift = 0.3 * np.sin(2 * np.pi * 1.3 * time)
    return spatial_mixing @ sources + slow_drift + 0.06 * rng.standard_normal((4, len(time)))


def test_reference_signals_have_frequency_harmonic_component_shape():
    references = make_reference_signals([10.0, 12.0], 250.0, 500, n_harmonics=2)

    assert references.shape == (2, 4, 500)
    assert references.dtype == np.float32
    np.testing.assert_allclose(references[:, :, 0], [[0.0, 1.0, 0.0, 1.0]] * 2)


def test_plain_cca_selects_synthetic_ssvep_frequency_with_rank_deficient_channels():
    eeg = _synthetic_ssvep(12.0)
    eeg = np.vstack([eeg, eeg[0]])  # duplicated channel makes Cxx rank deficient

    predicted, scores = predict_cca(eeg, [10.0, 12.0, 14.0], 250.0)

    assert predicted == 1
    assert scores.shape == (3,)
    assert scores[1] > 0.95
    assert np.isfinite(scores).all()


def test_fbcca_selects_synthetic_ssvep_frequency_with_default_and_custom_filterbanks():
    eeg = _synthetic_ssvep(12.0)
    config = FilterBankConfig(
        bands=[(6.0, 45.0), (14.0, 45.0), (22.0, 45.0)],
        weights=[1.0, 0.6, 0.3],
        order=3,
        n_harmonics=3,
    )

    for filterbank in (None, config):
        predicted, scores = predict_fbcca(
            eeg, [10.0, 12.0, 14.0], 250.0, filterbank
        )

        assert predicted == 1
        assert scores.shape == (3,)
        assert scores[1] > scores[[0, 2]].max()
        assert np.isfinite(scores).all()


def test_constant_inputs_have_zero_finite_scores():
    eeg = np.ones((4, 500), dtype=np.float64)
    references = make_reference_signals([10.0], 250.0, 500)[0]

    assert cca_score(eeg, references) == 0.0
    for predictor in (predict_cca, predict_fbcca):
        predicted, scores = predictor(eeg, [10.0, 12.0], 250.0)
        assert predicted == 0
        np.testing.assert_array_equal(scores, np.zeros(2))
        assert np.isfinite(scores).all()


@pytest.mark.parametrize(
    ("call", "message"),
    [
        (lambda: predict_cca(np.ones(500), [10.0], 250.0), "x must have shape"),
        (
            lambda: make_reference_signals([50.0], 200.0, 400, n_harmonics=2),
            "highest requested harmonic",
        ),
        (
            lambda: predict_fbcca(
                np.ones((2, 500)),
                [10.0],
                250.0,
                {"bands": [(6.0, 125.0)]},
            ),
            "Nyquist",
        ),
        (
            lambda: predict_fbcca(
                np.ones((2, 500)),
                [10.0],
                250.0,
                {"bands": [(6.0, 45.0), (14.0, 45.0)], "weights": [1.0]},
            ),
            "one value per band",
        ),
    ],
)
def test_invalid_shapes_harmonics_and_filterbank_configs_raise_clear_errors(call, message):
    with pytest.raises(ValueError, match=message):
        call()
