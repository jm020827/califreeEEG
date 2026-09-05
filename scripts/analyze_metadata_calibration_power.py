#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import sys
from pathlib import Path

import numpy as np
import scipy

REPO = Path(__file__).resolve().parents[1]
if str(REPO / "src") not in sys.path:
    sys.path.insert(0, str(REPO / "src"))

from cfeg.analysis.metadata_calibration_power import (
    MonteCarloSpec,
    one_sample_t_power,
    simulate_development_gate,
    simulate_held_core_hierarchy,
    simulate_held_mechanism_holm,
    simulate_marginal_claim,
)
from cfeg.metadata_calibration_contract import validate_metadata_calibration_plan
from cfeg.utils.config import load_config

POWER_MODULE = REPO / "src/cfeg/analysis/metadata_calibration_power.py"
ANALYSIS_MODULE = REPO / "src/cfeg/analysis/metadata_calibration_efficiency.py"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate outcome-free metadata-calibration power sensitivity receipt."
    )
    parser.add_argument(
        "--plan",
        type=Path,
        default=REPO / "configs/analysis/metadata_calibration_efficiency_v1.yaml",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=(
            REPO / "configs/governance/metadata_calibration_power_receipt.json"
        ),
    )
    parser.add_argument(
        "--draws",
        type=int,
        default=None,
        help="Override only for an explicitly marked quick diagnostic receipt.",
    )
    args = parser.parse_args()
    receipt = build_receipt(plan_path=args.plan, draws_override=args.draws)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(receipt, sort_keys=True, indent=2) + "\n"
    args.output.write_text(encoded, encoding="utf-8")
    print(args.output)
    print(receipt["receipt_payload_sha256"])


def build_receipt(*, plan_path: Path, draws_override: int | None = None) -> dict[str, object]:
    binding = validate_metadata_calibration_plan(plan_path, require_power_receipt=False)
    plan = load_config(plan_path, strict_env=False)
    power = plan["power_sensitivity"]
    configured_draws = int(power["monte_carlo_draws_per_cell"])
    draws = configured_draws if draws_override is None else int(draws_override)
    simulation = MonteCarloSpec(
        draws=draws,
        chunk_size=min(int(power["chunk_size"]), draws),
        master_seed=int(power["master_seed"]),
    )
    status = (
        "outcome_free_design_sensitivity_not_execution_authority"
        if draws == configured_draws
        else "quick_diagnostic_not_freeze_receipt"
    )

    primary_analytic = []
    for effect in power["effect_grid"]:
        for sd in power["participant_difference_sd_grid"]:
            primary_analytic.append(
                {
                    "n": 60,
                    "true_mean": float(effect),
                    "sd": float(sd),
                    "null_margin": 0.0,
                    "alpha": 0.05,
                    "inferential_rejection_power": one_sample_t_power(
                        n=60,
                        true_mean=float(effect),
                        sd=float(sd),
                        null_margin=0.0,
                        alpha=0.05,
                    ),
                }
            )
    noninferiority_analytic = []
    for true_mean in (-(1.0 / 30.0), -(1.0 / 60.0), -0.01, 0.0, 0.01):
        for sd in power["noninferiority_participant_difference_sd_grid"]:
            noninferiority_analytic.append(
                {
                    "n": 60,
                    "true_mean": true_mean,
                    "sd": float(sd),
                    "null_margin": -(1.0 / 60.0),
                    "alpha": 0.025,
                    "inferential_rejection_power": one_sample_t_power(
                        n=60,
                        true_mean=true_mean,
                        sd=float(sd),
                        null_margin=-(1.0 / 60.0),
                        alpha=0.025,
                    ),
                }
            )

    families = tuple(power["distribution_families"])
    marginal = []
    for family in families:
        marginal.append(
            simulate_marginal_claim(
                n=60,
                true_mean=0.02,
                sd=0.06,
                null_margin=0.0,
                alpha=0.05,
                practical_threshold=0.02,
                family=family,
                simulation=simulation,
            )
        )
        marginal.append(
            simulate_marginal_claim(
                n=60,
                true_mean=0.0,
                sd=0.045,
                null_margin=-(1.0 / 60.0),
                alpha=0.025,
                practical_threshold=None,
                family=family,
                simulation=simulation,
            )
        )

    planning_means = tuple(float(value) for value in power["core_planning_means"])
    planning_sds = tuple(float(value) for value in power["core_planning_sds"])
    matrices = {
        name: tuple(tuple(float(value) for value in row) for row in matrix)
        for name, matrix in power["core_contrast_correlation_scenarios"].items()
    }
    held_core = []
    for matrix_name, matrix in matrices.items():
        result = simulate_held_core_hierarchy(
            true_means=planning_means,
            sds=planning_sds,
            correlation_matrix=matrix,
            family="normal",
            simulation=simulation,
        )
        result["scenario"] = f"planning_{matrix_name}"
        held_core.append(result)
    for family in families:
        if family == "normal":
            continue
        result = simulate_held_core_hierarchy(
            true_means=planning_means,
            sds=planning_sds,
            correlation_matrix=matrices["shared_A_Q_k3"],
            family=family,
            simulation=simulation,
        )
        result["scenario"] = f"planning_shared_A_Q_k3_{family}"
        held_core.append(result)
    for scenario, means in (
        ("global_null", (0.0, 0.0, -(1.0 / 60.0))),
        ("calibration_plateau", (0.03, 0.0, 0.0)),
        ("harmful_low_budget", (0.03, 0.03, -0.03)),
        ("sesoi_boundary", (0.02, 0.02, 0.0)),
    ):
        result = simulate_held_core_hierarchy(
            true_means=means,
            sds=planning_sds,
            correlation_matrix=matrices["shared_A_Q_k3"],
            family="normal",
            simulation=simulation,
        )
        result["scenario"] = scenario
        held_core.append(result)
    for scenario, configured_means in power["core_partial_null_means"].items():
        result = simulate_held_core_hierarchy(
            true_means=tuple(float(value) for value in configured_means),
            sds=planning_sds,
            correlation_matrix=matrices["shared_A_Q_k3"],
            family="normal",
            simulation=simulation,
        )
        result["scenario"] = str(scenario)
        held_core.append(result)

    held_sample_size_sensitivity = []
    for n in power["core_planning_sample_size_sensitivity"]:
        result = simulate_held_core_hierarchy(
            true_means=planning_means,
            sds=planning_sds,
            correlation_matrix=matrices["shared_A_Q_k3"],
            family="normal",
            simulation=simulation,
            n=int(n),
        )
        result["scenario"] = f"planning_shared_A_Q_k3_n{int(n)}"
        held_sample_size_sensitivity.append(result)

    mechanism_power = plan["power_sensitivity"]["held_mechanism_power"]
    held_mechanism = []
    for correlation in mechanism_power["endpoint_correlations"]:
        result = simulate_held_mechanism_holm(
            true_means=tuple(float(value) for value in mechanism_power["planning_means"]),
            sds=tuple(float(value) for value in mechanism_power["planning_sds"]),
            endpoint_correlation=float(correlation),
            family="normal",
            simulation=simulation,
            practical_threshold=float(mechanism_power["practical_mean_min"]),
        )
        result["scenario"] = f"planning_correlation_{float(correlation):g}"
        held_mechanism.append(result)
    for family in families:
        if family == "normal":
            continue
        result = simulate_held_mechanism_holm(
            true_means=tuple(float(value) for value in mechanism_power["planning_means"]),
            sds=tuple(float(value) for value in mechanism_power["planning_sds"]),
            endpoint_correlation=0.5,
            family=family,
            simulation=simulation,
            practical_threshold=float(mechanism_power["practical_mean_min"]),
        )
        result["scenario"] = f"planning_correlation_0.5_{family}"
        held_mechanism.append(result)

    development = []
    for scenario, development_spec in power["development_gate_scenarios"].items():
        for family in development_spec["families"]:
            result = simulate_development_gate(
                contrast_means=tuple(
                    float(value) for value in development_spec["contrast_means"]
                ),
                contrast_sds=tuple(
                    float(value) for value in development_spec["contrast_sds"]
                ),
                baseline_mean=float(development_spec["baseline_mean"]),
                baseline_sd=float(development_spec["baseline_sd"]),
                endpoint_correlation=float(development_spec["endpoint_correlation"]),
                family=family,
                simulation=simulation,
            )
            result["scenario"] = str(scenario)
            development.append(result)

    receipt: dict[str, object] = {
        "schema": "cfeg.metadata-calibration-power-receipt.v1",
        "status": status,
        "candidate_id": binding["candidate_id"],
        "outcome_data_used": False,
        "data_dependencies": [],
        "n_development": 39,
        "n_held": 60,
        "master_seed": simulation.master_seed,
        "cell_seed_rule": "first_64_bits_sha256_of_master_seed_colon_canonical_cell_json",
        "monte_carlo_draws_per_cell": simulation.draws,
        "chunk_size": simulation.chunk_size,
        "maximum_nominal_mc_standard_error": 0.5 / np.sqrt(simulation.draws),
        "plan_sha256": binding["plan_sha256"],
        "power_config_sha256": _json_sha256(power),
        "power_script_sha256": _sha256_file(Path(__file__)),
        "power_module_sha256": _sha256_file(POWER_MODULE),
        "analysis_module_sha256": _sha256_file(ANALYSIS_MODULE),
        "source": _source_state(),
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "scipy": scipy.__version__,
        },
        "margin_semantic_unit": (
            "one_mean_additional_error_per_interface_sixty_queries_equivalently_"
            "two_total_errors_across_equal_dry_wet_participant_outcome"
        ),
        "decision_rules": {
            "primary": "one_sided_t_alpha_0p05_lower_gt_0_and_observed_mean_ge_0p020",
            "calibration_value": (
                "one_sided_t_alpha_0p05_lower_gt_0_and_observed_mean_ge_0p020"
            ),
            "confirmatory_savings": "A_QM_k1_noninferior_to_A_Q_k3_only",
            "mechanisms": "one_sided_t_alpha_0p05_lower_gt_0_and_observed_mean_ge_0p010",
            "noninferiority": "one_sided_t_alpha_0p025_lower_gt_minus_1_over_60",
            "wrong_context_tail": "required_descriptive_safety_report_not_sequence_gate",
            "metadata_only": (
                "deterministic_block_constant_M_plus_one_per_class_block_implies_chance"
            ),
            "confirmatory_hierarchy": (
                "H1_then_H2_intersection_of_calibration_value_and_k1_vs_k3_noninferiority"
            ),
            "mechanism_multiplicity": "Holm_alpha_0p05_after_confirmatory_core",
            "all_missing": "deterministic_unit_invariant_not_a_random_power_endpoint",
        },
        "distribution_definitions": power["distribution_definitions"],
        "sd_semantics": power["sd_semantics"],
        "analytic_primary": primary_analytic,
        "analytic_noninferiority": noninferiority_analytic,
        "marginal_monte_carlo": marginal,
        "held_core_hierarchy_monte_carlo": held_core,
        "held_core_sample_size_sensitivity": held_sample_size_sensitivity,
        "held_mechanism_holm_monte_carlo": held_mechanism,
        "development_gate_monte_carlo": development,
        "interpretation": {
            "test_power_vs_full_claim": (
                "Inferential rejection and practical mean-screen probabilities are separate."
            ),
            "utility_floor": (
                "Held core simulations are conditional on both endpoint BA means meeting 0.50 "
                "because contrast distributions do not identify absolute endpoint BA."
            ),
            "shared_endpoint_dependence": (
                "Calibration value C=A_Q(k3)-A_Q(k1) and savings S=A_QM(k1)-A_Q(k3) "
                "share A_Q(k3) with opposite signs; negative C-S correlations are included."
            ),
            "execution_authority": "none",
        },
    }
    receipt["receipt_payload_sha256"] = _json_sha256(receipt)
    return receipt


def _source_state() -> dict[str, object]:
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=REPO, check=True, capture_output=True, text=True
    ).stdout.strip()
    status = subprocess.run(
        ["git", "status", "--porcelain=v1", "--untracked-files=all"],
        cwd=REPO,
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    return {
        "git_commit": commit,
        "git_worktree_clean": not bool(status),
        "git_status_sha256": hashlib.sha256(status.encode()).hexdigest(),
    }


def _json_sha256(payload: object) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


if __name__ == "__main__":
    main()
