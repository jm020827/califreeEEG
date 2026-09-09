"""Generated mathematics tests only; never access human EEG or metadata."""

import numpy as np
import pytest
from scipy.stats import pearsonr

from cfeg.analysis.source_channel_margin import participant_margin_mse, source_channel_margin


def waves():
    # Twelve orthogonal, centered Fourier signals; deterministic engineering data.
    t = np.arange(64) / 64
    wave = np.sin(2 * np.pi * np.arange(1, 13)[:, None] * t)
    source = np.broadcast_to(wave[:, None, None, :], (12, 5, 8, 64)).copy()
    return np.broadcast_to(source, (3, *source.shape)).copy(), source


def target(s, x):
    return source_channel_margin(s, x, np.arange(12), np.arange(12))


def test_orthogonal_templates_are_perfect_but_have_no_channel_shape():
    s, x = waves()
    out = target(s, x)
    np.testing.assert_allclose(out.class_margin, 1, atol=1e-14)
    np.testing.assert_allclose(out.level, 1, atol=1e-14)
    np.testing.assert_allclose(out.channel_shape, 0, atol=1e-14)


def test_wrong_class_channel_produces_negative_signed_margin_and_relative_shape():
    s, x = waves()
    x[:, :, 0] = np.roll(x[:, :, 0], 1, axis=0)
    out = target(s, x)
    np.testing.assert_allclose(out.channel_margin[:, 0], -1, atol=1e-14)
    np.testing.assert_allclose(out.channel_margin[:, 1:], 1, atol=1e-14)
    np.testing.assert_allclose(out.channel_shape[:, 0], -1.75, atol=1e-14)
    np.testing.assert_allclose(out.channel_shape[:, 1:], 0.25, atol=1e-14)


def test_independent_scalar_pearson_and_competitor_reference():
    rng = np.random.default_rng(101)
    s = rng.normal(size=(3, 12, 5, 8, 24))
    x = rng.normal(size=(12, 5, 8, 24))
    out = target(s, x)
    templates = s.mean(axis=0)
    for y in range(12):
        for b, c in ((0, 0), (2, 7), (4, 3)):
            r = np.array([pearsonr(x[y, b, c], templates[j, b, c]).statistic for j in range(12)])
            np.testing.assert_allclose(out.correlations[y, :, b, c], r, atol=1e-14)
            assert out.class_margin[y, b, c] == pytest.approx(r[y] - np.delete(r, y).max())


def test_positive_scaling_offsets_and_class_reordering():
    s, x = waves()
    x[:, :, 0] = np.roll(x[:, :, 0], 1, axis=0)
    baseline = target(s, x)
    perm = np.arange(12)[::-1]
    out = source_channel_margin(3 * s[:, perm] + 7, 2 * x[perm] - 11, perm, perm)
    np.testing.assert_allclose(out.channel_shape, baseline.channel_shape, atol=1e-13)
    np.testing.assert_allclose(out.class_margin, baseline.class_margin[perm], atol=1e-13)


def test_readonly_inputs_and_channel_permutation():
    s, x = waves()
    x[:, :, 0] = np.roll(x[:, :, 0], 1, axis=0)
    s.flags.writeable = x.flags.writeable = False
    baseline = target(s, x)
    out = target(s[:, :, :, ::-1], x[:, :, ::-1])
    np.testing.assert_allclose(out.channel_shape, baseline.channel_shape[:, ::-1], atol=1e-14)


@pytest.mark.parametrize("norm,valid", [(0.99e-12, False), (1.01e-12, True)])
def test_centered_norm_boundary(norm, valid):
    s, x = waves()
    wave = x[0, 0, 0]
    small = wave * (norm / np.linalg.norm(wave - wave.mean()))
    s[:, 0, 0, 0] = small
    x[0, 0, 0] = small
    if valid:
        assert np.isfinite(target(s, x).channel_shape).all()
    else:
        with pytest.raises(ValueError, match="undefined"):
            target(s, x)


def test_individually_valid_support_can_have_undefined_average_template():
    s, x = waves()
    s[2, 0, 0, 0] *= -2
    assert np.all(np.linalg.norm(s[:, 0, 0, 0], axis=-1) > 1)
    with pytest.raises(ValueError, match="template"):
        target(s, x)


@pytest.mark.parametrize("which", ["support", "source"])
@pytest.mark.parametrize("bad", [0.0, 1e-20, np.nan, np.inf])
def test_undefined_or_nonfinite_waveform_rejects_whole_target(which, bad):
    s, x = waves()
    if which == "support":
        s[:, 0, 0, 0] = bad
    else:
        x[0, 0, 0] = bad
    with pytest.raises(ValueError):
        target(s, x)


@pytest.mark.parametrize(
    "labels", [np.arange(12) + 1, np.zeros(12, dtype=int), np.arange(12.0), [True] * 12, [0, 1]]
)
def test_bad_labels_rejected(labels):
    s, x = waves()
    with pytest.raises(ValueError):
        source_channel_margin(s, x, labels, labels)


def test_label_order_mismatch_rejected():
    s, x = waves()
    with pytest.raises(ValueError, match="order"):
        source_channel_margin(s, x, np.arange(12), np.arange(12)[::-1])


@pytest.mark.parametrize(
    "change", ["k", "classes", "bands", "channels", "samples", "source_length", "bool", "complex"]
)
def test_bad_geometry_or_dtype_rejected(change):
    s, x = waves()
    if change == "k":
        s = s[:2]
    elif change == "classes":
        s = s[:, :11]
    elif change == "bands":
        s = s[:, :, :4]
    elif change == "channels":
        s = s[:, :, :, :7]
    elif change == "samples":
        s = s[..., :1]
    elif change == "source_length":
        x = x[..., :-1]
    elif change == "bool":
        s = s.astype(bool)
    elif change == "complex":
        x = x.astype(complex)
    with pytest.raises(ValueError):
        target(s, x)


def test_equal_person_loss_and_prediction_centering():
    y = np.zeros((2, 2, 5, 8))
    p = y.copy()
    p[0, :, :, 0] = 8
    np.testing.assert_allclose(participant_margin_mse(p, y), [7, 0])
    np.testing.assert_allclose(participant_margin_mse(p + 100, y), [7, 0])


@pytest.mark.parametrize("change", ["uncentered", "nan", "empty", "wrong_shape", "overflow"])
def test_invalid_losses_rejected(change):
    p = np.zeros((2, 2, 5, 8))
    y = p.copy()
    if change == "uncentered":
        y += 1
    elif change == "nan":
        p[0, 0, 0, 0] = np.nan
    elif change == "empty":
        p, y = p[:0], y[:0]
    elif change == "wrong_shape":
        p = p[:, :1]
    elif change == "overflow":
        p[0, 0, 0, 0] = 1e308
    with pytest.raises(ValueError):
        participant_margin_mse(p, y)
