from __future__ import annotations

import pytest

from cfeg.analysis.inference_simulation import run_target_free_inference_simulation


def _config() -> dict:
    return {
        "schema": "cfeg.target-free-inference-simulation-plan.v1",
        "freeze_eligible": True,
        "seed": 17,
        "n_simulations": 80,
        "n_subjects": 17,
        "n_folds": 5,
        "n_optimization_seeds": 3,
        "alpha_grid": [0.05],
        "effect_above_sesoi_grid": [0.03, 0.2],
        "sign_flip_resamples": 400,
        "bootstrap_resamples": 300,
        "acceptance": {
            "max_null_rejection_rate": 0.3,
            "power_effect_above_sesoi": 0.2,
            "minimum_power": 0.8,
        },
        "scenarios": [
            {
                "name": "symmetric",
                "distribution": "normal",
                "subject_sd": 0.08,
                "seed_subject_sd": 0.01,
                "seed_global_sd": 0.0,
                "fold_seed_sd": 0.0,
            }
        ],
    }


def test_target_free_simulation_is_reproducible_and_declares_no_outcome_access() -> None:
    first = run_target_free_inference_simulation(_config())
    second = run_target_free_inference_simulation(_config())

    assert first == second
    assert first["status"] == "accepted"
    assert first["target_outcomes_accessed"] is False
    assert first["sesoi_translation_invariant"] is True


def test_target_free_simulation_rejects_unregistered_power_effect() -> None:
    config = _config()
    config["acceptance"]["power_effect_above_sesoi"] = 0.1

    with pytest.raises(ValueError, match="must be present"):
        run_target_free_inference_simulation(config)
