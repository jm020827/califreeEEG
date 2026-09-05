from __future__ import annotations

import numpy as np
import pytest

from cfeg.analysis.query_reliability_power import (
    analytic_power_grid,
    simulate_joint_promotion_probability,
    simulate_primary_promotion_probability,
)


def test_analytic_power_grid_reproduces_the_clean_sensitivity_bottleneck() -> None:
    rows = analytic_power_grid(
        true_means=(0.0,),
        standard_deviations=(0.04,),
        null_margin=-0.01,
    )
    assert rows[0]["power"] == pytest.approx(0.2855, abs=1e-4)


def test_primary_and_joint_simulations_are_deterministic_and_bounded() -> None:
    kwargs = {
        "true_mean": 0.03,
        "standard_deviation": 0.04,
        "draws": 2_000,
        "seed": 20260904,
        "chunk_size": 257,
    }
    first = simulate_primary_promotion_probability(**kwargs)
    second = simulate_primary_promotion_probability(**kwargs)
    assert first == second
    assert 0.0 <= first["probability"] <= 1.0
    assert first["draws"] == 2_000

    joint = simulate_joint_promotion_probability(
        primary_mean=0.03,
        clean_simple_mean=0.0,
        clean_deployment_mean=0.0,
        standard_deviation=0.04,
        correlation=np.eye(3),
        draws=2_000,
        seed=20260904,
        chunk_size=257,
    )
    assert 0.0 <= joint["probability"] <= 1.0
    with pytest.raises(ValueError, match="positive definite"):
        simulate_joint_promotion_probability(
            primary_mean=0.03,
            clean_simple_mean=0.0,
            clean_deployment_mean=0.0,
            standard_deviation=0.04,
            correlation=np.ones((3, 3)),
            draws=10,
            seed=1,
        )
