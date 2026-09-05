from __future__ import annotations

import numpy as np
import pytest

from cfeg.analysis.query_reliability_statistics import (
    analytic_one_sided_mean_t_power,
    evaluate_query_reliability_promotion_gates,
    one_sided_paired_mean_result,
)


def test_fixed_one_sided_mean_procedure_and_zero_variance_rule() -> None:
    result = one_sided_paired_mean_result(np.full(20, 0.03), null_margin=0.0)
    assert result.mean == pytest.approx(0.03)
    assert result.standard_deviation_ddof1 == 0.0
    assert result.lower_confidence_bound == pytest.approx(0.03)
    assert result.p_value_greater == 0.0

    at_null = one_sided_paired_mean_result(np.full(20, -0.01), null_margin=-0.01)
    assert at_null.p_value_greater == 1.0
    assert at_null.lower_confidence_bound == pytest.approx(-0.01)


def test_joint_promotion_requires_benefit_consistency_and_two_clean_safeties() -> None:
    passing = evaluate_query_reliability_promotion_gates(
        corrupted_q1_aug_minus_q0_aug=np.full(20, 0.03),
        clean_q1_aug_minus_q0_aug=np.zeros(20),
        clean_q1_aug_minus_q0_clean=np.zeros(20),
    )
    assert passing["promotion_pass"] is True
    assert passing["contract_status"] == "draft_provisional_not_frozen"
    assert passing["grants_outcome_execution_or_publication_authority"] is False
    assert passing["required_positive_participant_count"] == 12
    assert passing["positive_fraction_is_inferential_test"] is False

    concentrated = evaluate_query_reliability_promotion_gates(
        corrupted_q1_aug_minus_q0_aug=np.asarray([0.08] * 10 + [-0.01] * 10),
        clean_q1_aug_minus_q0_aug=np.zeros(20),
        clean_q1_aug_minus_q0_clean=np.zeros(20),
    )
    assert concentrated["gates"]["positive_participant_consistency"] is False
    assert concentrated["promotion_pass"] is False

    deployment_harm = evaluate_query_reliability_promotion_gates(
        corrupted_q1_aug_minus_q0_aug=np.full(20, 0.03),
        clean_q1_aug_minus_q0_aug=np.zeros(20),
        clean_q1_aug_minus_q0_clean=np.full(20, -0.02),
    )
    assert deployment_harm["gates"]["clean_simple_noninferiority"] is True
    assert deployment_harm["gates"]["clean_deployment_noninferiority"] is False
    assert deployment_harm["promotion_pass"] is False


def test_statistics_fail_closed_on_missing_participants_or_nonfinite_values() -> None:
    with pytest.raises(ValueError, match="exactly 20"):
        one_sided_paired_mean_result(np.zeros(19), null_margin=0.0)
    values = np.zeros(20)
    values[-1] = np.nan
    with pytest.raises(ValueError, match="finite"):
        one_sided_paired_mean_result(values, null_margin=0.0)
    with pytest.raises(ValueError, match="at least two"):
        one_sided_paired_mean_result([0.0], null_margin=0.0, expected_n=1)
    with pytest.raises(ValueError, match="finite"):
        one_sided_paired_mean_result(np.zeros(20), null_margin=np.nan)


def test_promotion_thresholds_and_positive_count_boundary_fail_closed() -> None:
    common = {
        "clean_q1_aug_minus_q0_aug": np.zeros(20),
        "clean_q1_aug_minus_q0_clean": np.zeros(20),
    }
    eleven = evaluate_query_reliability_promotion_gates(
        corrupted_q1_aug_minus_q0_aug=np.asarray([0.1] * 11 + [0.0] * 9),
        **common,
    )
    twelve = evaluate_query_reliability_promotion_gates(
        corrupted_q1_aug_minus_q0_aug=np.asarray([0.1] * 12 + [0.0] * 8),
        **common,
    )
    assert eleven["gates"]["positive_participant_consistency"] is False
    assert twelve["gates"]["positive_participant_consistency"] is True

    at_margin = evaluate_query_reliability_promotion_gates(
        corrupted_q1_aug_minus_q0_aug=np.full(20, 0.03),
        clean_q1_aug_minus_q0_aug=np.full(20, -0.01),
        clean_q1_aug_minus_q0_clean=np.zeros(20),
    )
    assert at_margin["gates"]["clean_simple_noninferiority"] is False

    with pytest.raises(ValueError, match="thresholds must all be finite"):
        evaluate_query_reliability_promotion_gates(
            corrupted_q1_aug_minus_q0_aug=np.full(20, 0.03),
            practical_effect_min=np.nan,
            **common,
        )
    with pytest.raises(ValueError, match="balanced-accuracy range"):
        evaluate_query_reliability_promotion_gates(
            corrupted_q1_aug_minus_q0_aug=np.full(20, 1.01),
            **common,
        )
    with pytest.raises(ValueError, match="practical_effect_min"):
        evaluate_query_reliability_promotion_gates(
            corrupted_q1_aug_minus_q0_aug=np.full(20, 0.03),
            practical_effect_min=-0.01,
            **common,
        )
    with pytest.raises(ValueError, match="clean_noninferiority_margin"):
        evaluate_query_reliability_promotion_gates(
            corrupted_q1_aug_minus_q0_aug=np.full(20, 0.03),
            clean_noninferiority_margin=0.0,
            **common,
        )


def test_outcome_free_analytic_power_exposes_n20_clean_ni_bottleneck() -> None:
    clean_ni = analytic_one_sided_mean_t_power(
        n_participants=20,
        true_mean=0.0,
        null_margin=-0.01,
        standard_deviation=0.04,
    )
    primary = analytic_one_sided_mean_t_power(
        n_participants=20,
        true_mean=0.03,
        null_margin=0.0,
        standard_deviation=0.04,
    )
    assert clean_ni == pytest.approx(0.2855, abs=1e-4)
    assert primary == pytest.approx(0.9437, abs=1e-4)
    with pytest.raises(ValueError, match="positive"):
        analytic_one_sided_mean_t_power(
            n_participants=20,
            true_mean=0.0,
            null_margin=-0.01,
            standard_deviation=0.0,
        )
