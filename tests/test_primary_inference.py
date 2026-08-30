from __future__ import annotations

import pandas as pd
import pytest

from cfeg.analysis.primary_inference import primary_subject_inference


def _subjects(differences):
    return pd.DataFrame(
        {
            "subject_id": [f"sim{index:03d}" for index in range(len(differences))],
            "balanced_accuracy_delta": differences,
        }
    )


def test_exact_one_sided_sign_flip_has_n3_resolution_one_eighth() -> None:
    result = primary_subject_inference(
        _subjects([1.0, 1.0, 1.0]),
        alpha=0.05,
        alternative="greater",
        null_margin_ba=0.0,
        seed=7,
        n_resamples=100,
    )

    assert result["sign_flip_exact"] is True
    assert result["paired_sign_flip_p_value"] == 0.125
    assert result["reject_null"] is False


def test_minimum_effect_test_centers_on_sesoi_not_zero() -> None:
    subjects = _subjects([0.03] * 17)
    above_zero = primary_subject_inference(
        subjects,
        alpha=0.05,
        alternative="greater",
        null_margin_ba=0.0,
        seed=9,
        n_resamples=2_000,
    )
    above_sesoi = primary_subject_inference(
        subjects,
        alpha=0.05,
        alternative="greater",
        null_margin_ba=0.04,
        seed=9,
        n_resamples=2_000,
    )

    assert above_zero["reject_null"] is True
    assert above_sesoi["reject_null"] is False
    assert above_sesoi["confidence_interval_low"] == 0.03


def test_primary_inference_rejects_duplicate_subject_ids() -> None:
    subjects = _subjects([0.01, 0.02])
    subjects.loc[1, "subject_id"] = subjects.loc[0, "subject_id"]

    with pytest.raises(ValueError, match="must be unique"):
        primary_subject_inference(
            subjects,
            alpha=0.05,
            alternative="greater",
            null_margin_ba=0.0,
            seed=9,
            n_resamples=100,
        )


def test_primary_inference_requires_at_least_two_subjects() -> None:
    with pytest.raises(ValueError, match="at least two"):
        primary_subject_inference(
            _subjects([0.01]),
            alpha=0.05,
            alternative="greater",
            null_margin_ba=0.0,
            seed=9,
            n_resamples=100,
        )
