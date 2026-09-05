from __future__ import annotations

import pytest

from cfeg.analysis.metadata_calibration_power import (
    HELD_CORE_ENDPOINTS,
    MonteCarloSpec,
    one_sample_t_power,
    simulate_development_gate,
    simulate_held_core_hierarchy,
    simulate_held_mechanism_holm,
    simulate_marginal_claim,
)


def test_normal_theory_reference_powers_match_frozen_examples() -> None:
    primary = one_sample_t_power(
        n=60,
        true_mean=0.02,
        sd=0.06,
        null_margin=0.0,
        alpha=0.05,
    )
    noninferiority = one_sample_t_power(
        n=60,
        true_mean=0.0,
        sd=0.045,
        null_margin=-(1.0 / 60.0),
        alpha=0.025,
    )
    assert primary == pytest.approx(0.817887, abs=1e-6)
    assert noninferiority == pytest.approx(0.8056, abs=2e-4)


def test_normal_monte_carlo_matches_analytic_and_separates_mean_screen() -> None:
    simulation = MonteCarloSpec(draws=40_000, chunk_size=4_000, master_seed=7)
    result = simulate_marginal_claim(
        n=60,
        true_mean=0.02,
        sd=0.06,
        null_margin=0.0,
        alpha=0.05,
        practical_threshold=0.02,
        family="normal",
        simulation=simulation,
    )
    analytic = one_sample_t_power(
        n=60,
        true_mean=0.02,
        sd=0.06,
        null_margin=0.0,
        alpha=0.05,
    )
    mc_se = result["maximum_nominal_mc_standard_error"]
    assert abs(result["inferential_rejection_power"] - analytic) < 4 * mc_se
    assert result["full_claim_probability"] == pytest.approx(0.50, abs=0.02)
    assert result["realized_mean"] == pytest.approx(0.02, abs=5e-4)
    assert result["realized_sd"] == pytest.approx(0.06, abs=5e-4)


@pytest.mark.parametrize(
    "family",
    ["normal", "standardized_t5", "twenty_percent_harmed_mixture"],
)
def test_core_hierarchy_is_reproducible_and_models_shared_endpoint(family: str) -> None:
    simulation = MonteCarloSpec(draws=3_000, chunk_size=500, master_seed=18)
    kwargs = {
        "true_means": (0.03, 0.03, 0.0),
        "sds": (0.06, 0.06, 0.045),
        "correlation_matrix": (
            (1.0, 0.0, 0.3),
            (0.0, 1.0, -0.5),
            (0.3, -0.5, 1.0),
        ),
        "family": family,
        "simulation": simulation,
    }
    first = simulate_held_core_hierarchy(**kwargs)
    second = simulate_held_core_hierarchy(**kwargs)
    assert first == second
    assert first["endpoint_order"] == list(HELD_CORE_ENDPOINTS)
    assert first["calibration_saving_full_claim_probability"] <= first["H1_claim_probability"]
    assert first["same_budget_positive_identity_probability_among_H2"] == 1.0
    if family == "twenty_percent_harmed_mixture":
        assert first["realized_harmed_fraction"] == pytest.approx(0.20, abs=0.01)
    else:
        assert first["realized_harmed_fraction"] is None


def test_core_hierarchy_rejects_invalid_correlation_matrix() -> None:
    with pytest.raises(ValueError, match="positive semidefinite"):
        simulate_held_core_hierarchy(
            true_means=(0.03, 0.03, 0.0),
            sds=(0.06, 0.06, 0.045),
            correlation_matrix=((1.0, 0.9, 0.9), (0.9, 1.0, -0.9), (0.9, -0.9, 1.0)),
            family="normal",
            simulation=MonteCarloSpec(draws=100, chunk_size=50),
        )


def test_held_mechanism_holm_is_conditional_reproducible_sensitivity() -> None:
    simulation = MonteCarloSpec(draws=4_000, chunk_size=500, master_seed=27)
    kwargs = {
        "true_means": (0.02, 0.02),
        "sds": (0.04, 0.04),
        "endpoint_correlation": 0.5,
        "family": "normal",
        "simulation": simulation,
    }
    first = simulate_held_mechanism_holm(**kwargs)
    assert first == simulate_held_mechanism_holm(**kwargs)
    assert first["conditional_on_confirmatory_core_open"] is True
    assert first["not_a_joint_pipeline_probability"] is True
    assert first["both_mechanism_claim_probability"] <= first[
        "any_mechanism_claim_probability"
    ]
    assert set(first["individual_holm_claim_probability"]) == {
        "correct_context_over_block_shuffle",
        "correct_context_over_stale",
    }


def test_development_gate_reports_joint_promotion_not_only_marginal_power() -> None:
    simulation = MonteCarloSpec(draws=2_000, chunk_size=500, master_seed=19)
    viable = simulate_development_gate(
        contrast_means=(0.03, 0.025, 0.025, 0.0),
        contrast_sds=(0.03, 0.03, 0.03, 0.02),
        baseline_mean=0.60,
        baseline_sd=0.03,
        endpoint_correlation=0.5,
        family="normal",
        simulation=simulation,
    )
    null = simulate_development_gate(
        contrast_means=(0.0, 0.0, 0.0, 0.0),
        contrast_sds=(0.03, 0.03, 0.03, 0.02),
        baseline_mean=0.48,
        baseline_sd=0.03,
        endpoint_correlation=0.5,
        family="normal",
        simulation=simulation,
    )
    assert viable["hard_gate_promotion_probability"] > 0.70
    assert null["hard_gate_promotion_probability"] < 0.01
    assert viable["all_checks_pass_probability"] <= viable["hard_gate_promotion_probability"]
    assert sum(viable["hard_first_failure_probability"].values()) == pytest.approx(1.0)
