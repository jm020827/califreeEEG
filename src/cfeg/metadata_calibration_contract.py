from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from itertools import pairwise
from pathlib import Path

import numpy as np
import pandas as pd

from cfeg.identity import canonical_identity_sha256
from cfeg.utils.config import load_config

METADATA_CALIBRATION_PLAN = "configs/analysis/metadata_calibration_efficiency_v1.yaml"
METADATA_CALIBRATION_SOURCE_RECIPE = "configs/train/metadata_calibration_efficiency_v1.yaml"
METADATA_CALIBRATION_BASELINE_LEDGER = (
    "configs/analysis/metadata_calibration_baseline_ledger_v1.yaml"
)
METADATA_CALIBRATION_INPUT_ARTIFACT_CONTRACT = (
    "configs/analysis/metadata_calibration_input_artifacts_v1.yaml"
)
METADATA_CALIBRATION_INPUT_ARTIFACT_CONTRACT_SHA256 = (
    "6443956372afb318a21ec38776ea02e6470fad14310fe9c6b724632de6b840ca"
)
METADATA_CALIBRATION_POWER_RECEIPT = (
    "configs/governance/metadata_calibration_power_receipt.json"
)
METADATA_CALIBRATION_BUDGETS = (0, 1, 3, 5)
METADATA_CALIBRATION_SUPPORT_BLOCKS = (
    "block01",
    "block02",
    "block03",
    "block04",
    "block05",
)
METADATA_CALIBRATION_QUERY_BLOCKS = (
    "block06",
    "block07",
    "block08",
    "block09",
    "block10",
)


@dataclass(frozen=True)
class CompleteBlockCalibrationPartition:
    support_by_budget: dict[int, np.ndarray]
    query_indices: np.ndarray
    query_identity_sha256: str
    dataset_id: str
    subject_id: str
    electrode_type: str

    def contract(self) -> dict:
        return {
            "budgets": list(self.support_by_budget),
            "support_counts": {
                str(budget): len(indices) for budget, indices in self.support_by_budget.items()
            },
            "query_count": len(self.query_indices),
            "query_identity_sha256": self.query_identity_sha256,
            "dataset_id": self.dataset_id,
            "subject_id": self.subject_id,
            "electrode_type": self.electrode_type,
        }


@dataclass(frozen=True)
class CompleteBlockCohortBinding:
    role: str
    subject_ids: tuple[str, ...]
    participant_condition_groups: int
    query_count: int
    query_identity_bundle_sha256: str

    def contract(self) -> dict:
        return {
            "role": self.role,
            "subject_ids": list(self.subject_ids),
            "participant_condition_groups": self.participant_condition_groups,
            "query_count": self.query_count,
            "query_identity_bundle_sha256": self.query_identity_bundle_sha256,
        }


def validate_metadata_calibration_plan(
    path: str | Path | None = None,
    *,
    require_power_receipt: bool = True,
) -> dict:
    """Validate the frozen plan and its exact delegated execution authority."""

    plan_path = Path(path) if path is not None else _repository_root() / METADATA_CALIBRATION_PLAN
    plan = load_config(plan_path, strict_env=False)
    expected_header = {
        "schema": "cfeg.metadata-calibration-efficiency-plan.v1",
        "decision_id": "DEC-20260905-006",
        "candidate_id": "metadata-calibration-efficiency-v1",
        "status": "frozen_execution_ready_signed_phase_authority_required",
    }
    observed_header = {key: plan.get(key) for key in expected_header}
    if observed_header != expected_header:
        raise ValueError(f"Metadata-calibration plan header drifted: {observed_header}.")

    authority = plan.get("authority") or {}
    expected_authorization_basis = (
        "응 그렇게 해보자. 추가데이터 및 필요시 결정까지 다 너가 알아서 해서 실험 종료까지 시작 계속."
    )
    if authority.get("direction_approved") is not True:
        raise ValueError("The metadata-calibration research direction is not recorded.")
    if authority.get("approval_message") != expected_authorization_basis:
        raise ValueError("The delegated execution authorization basis drifted.")
    if (
        authority.get("delegation_evidence")
        != "chat_recorded_manual_owner_directive"
        or authority.get("cryptographic_root_of_trust")
        != "ephemeral_run_integrity_key_not_an_external_owner_signature"
    ):
        raise ValueError("The authority trust-model disclosure drifted.")
    if authority.get("exact_outcome_bundle_approved") is not True:
        raise ValueError("The frozen plan lacks exact outcome-bundle approval.")
    if (
        authority.get("current_authorized_stage") != "source_then_conditional_held"
        or authority.get("authorization_stages")
        != {
            "input_seal": (
                "delegated_signed_phase_decision_required_before_raw_manifest_signal_or_label_read"
            ),
            "outcome_execution": (
                "delegated_signed_phase_decision_required_after_and_bound_to_input_seal_receipt"
            ),
        }
    ):
        raise ValueError("The two-stage input-seal/outcome authority contract drifted.")
    if authority.get("allowed_now") != [
        "literature_public_docs_and_existing_artifact_receipt_audit_no_human_raw_path",
        "outcome_free_design_and_contract_implementation",
        "synthetic_and_unit_invariants",
        "cuda_forward_only_contract_probe",
        "statistical_bundle_and_power_simulation",
        "source_development_human_eeg_under_exact_signed_phase_chain",
        "held_participant_human_eeg_only_after_signed_passing_source_gate",
        "aggregate_only_finalization_and_required_research_record",
    ]:
        raise ValueError("The pre-authorization work allowlist drifted.")
    required_forbidden = {
        "execution_without_exact_signed_phase_chain",
        "held_access_before_passing_source_gate",
        "worker_access_to_raw_processed_labels_or_private_keys",
        "partial_grid_or_partial_reveal",
        "same_cohort_retry_or_post_outcome_retuning",
    }
    if not required_forbidden.issubset(set(authority.get("forbidden_now") or [])):
        raise ValueError("The frozen plan does not fail closed on unauthorized outcome paths.")

    predecessors = plan.get("predecessor_boundaries") or {}
    expected_predecessors = {
        "physical_hybrid_v1": "retired_no_go_do_not_reopen",
        "reliability_spatial_v1": "terminal_synthetic_stage0_failure_do_not_rerun",
        "query_reliability_spatial_v1": ("withdrawn_before_human_outcome_frozen_baseline_only"),
    }
    for candidate, disposition in expected_predecessors.items():
        if (predecessors.get(candidate) or {}).get("disposition") != disposition:
            raise ValueError(f"Predecessor boundary drifted for {candidate}.")

    data = plan.get("data_contract") or {}
    if (
        data.get("primary_dataset") != "wearable_v3"
        or data.get("expected_subjects") != 102
        or data.get("expected_rows") != 24480
        or data.get("broad_device_site_reference_generalization_claim_allowed") is not False
    ):
        raise ValueError("The narrow wearable data and claim contract drifted.")
    if set(data.get("identifiable_external_context") or []) != {
        "electrode_interface_wet_or_dry",
        "pre_block_per_channel_impedance_kohm",
        "impedance_availability",
    }:
        raise ValueError("External acquisition-context treatment fields drifted.")
    forbidden_predictors = set(data.get("forbidden_predictors") or [])
    if not {"dataset_id", "subject_id", "target_query_label", "target_query_outcome"}.issubset(
        forbidden_predictors
    ):
        raise ValueError("The forbidden metadata proxy contract is incomplete.")
    _validate_allocation_contract(data.get("proposed_unopened_allocation") or {})

    method = plan.get("method_contract") or {}
    if (
        method.get("method_family") != "nested_metadata_residual_target_calibration_prior"
        or method.get("implementation_status")
        != "frozen_implemented_signed_phase_execution_required"
    ):
        raise ValueError("The low-sample measurement-prior method family drifted.")
    roles = method.get("primary_roles") or {}
    if set(roles) != {"A_Q", "A_QM"}:
        raise ValueError("The primary pair must be exactly A_Q and A_QM.")
    fairness = method.get("fairness") or {}
    expected_fairness = {
        "same_parameter_graph": "required",
        "same_composite_source_checkpoint": "required",
        "common_backbone_and_Q_estimator_frozen_before_M_fit": "required",
        "same_source_data_and_support": "required",
        "same_task_codebook_and_adaptation_rule": "required",
        "primary_treatment_difference": ("A_QM_has_external_context_while_A_Q_has_exact_null"),
        "missing_metadata_fallback": "same_frozen_A_Q_path_exact_behavior",
        "separately_trained_A_Q_checkpoint_claim": "forbidden",
    }
    if (
        (roles.get("A_Q") or {}).get("external_context_mode") != "residual_exact_off"
        or (roles.get("A_QM") or {}).get("external_context_mode") != "observed_bounded_residual"
        or fairness != expected_fairness
    ):
        raise ValueError("The incremental metadata and exact-fallback contract drifted.")
    staged = method.get("staged_source_fit") or {}
    if staged != {
        "stage_1": "fit_and_freeze_common_backbone_and_Q_only_calibration_estimator",
        "stage_2": "fit_only_bounded_M_residual_with_common_path_frozen",
        "target_support_update": "closed_form_or_small_estimator_only_backbone_frozen",
        "common_path_metadata_influence": "forbidden",
    }:
        raise ValueError("The staged nested-metadata fit contract drifted.")
    if method.get("source_recipe") != METADATA_CALIBRATION_SOURCE_RECIPE:
        raise ValueError("The exact source-training recipe path drifted.")
    formula = method.get("exact_formula") or {}
    expected_formula = {
        "representation": "z_equals_l2_normalized_frozen_spectral_embedding",
        "source_anchor": "one_stage1_learned_anchor_per_class",
        "base_precision": "exp_clipped_base_log_precision",
        "q_precision": "lambda_Q_equals_clip_exp_base_plus_log4_tanh_gQ",
        "metadata_residual": "delta_M_equals_availability_times_log1p25_tanh_gM",
        "effective_precision": "lambda_QM_equals_clip_lambda_Q_times_exp_delta_M",
        "source_prior": "P0_equals_source_prior_strength_times_diag_base_precision",
        "posterior_precision": "Pc_equals_P0_plus_sum_support_diag_effective_precision",
        "posterior_mean": (
            "muc_equals_Pc_inverse_times_P0_mu_source_plus_sum_support_diag_precision_z"
        ),
        "predictive_variance": (
            "vcq_equals_inverse_query_precision_plus_inverse_posterior_precision"
        ),
        "score": "minus_half_sum_squared_distance_over_v_plus_log_v",
        "probability": "softmax_over_twelve_class_scores",
        "k0_behavior": (
            "empty_support_keeps_source_anchor_but_M_may_change_query_predictive_precision"
        ),
    }
    expected_bounds = {
        "q_precision_ratio": [0.25, 4.0],
        "metadata_precision_ratio": [0.8, 1.25],
        "absolute_precision": [0.05, 20.0],
        "source_prior_strength": 1.0,
        "metadata_last_layer_initialization": "exact_zero",
        "metadata_hidden_width": 32,
    }
    expected_runtime_modes = {
        "A_0": {"q_mode": "exact_off", "metadata_mode": "exact_off"},
        "A_M": {"q_mode": "exact_off", "metadata_mode": "observed"},
        "A_Q": {"q_mode": "observed", "metadata_mode": "exact_off"},
        "A_QM": {"q_mode": "observed", "metadata_mode": "observed"},
    }
    implementation = method.get("implementation") or {}
    expected_feature_implementation = {
        "q_builder": (
            "src/cfeg/data/metadata_calibration_features.py::build_wearable_q_features"
        ),
        "metadata_builder": (
            "src/cfeg/data/metadata_calibration_features.py::build_wearable_metadata_features"
        ),
        "composite_module": "src/cfeg/models/metadata_calibration_composite.py",
        "composite_class": "MetadataCalibrationComposite",
        "q_feature_schema": "wearable_q_8_channels_x_7_v1",
        "q_feature_width": 56,
        "metadata_feature_schema": "wearable_interface2_impedance8_availability8_v1",
        "metadata_feature_width": 18,
        "official_channel_order": ["POz", "PO3", "PO4", "PO5", "PO6", "Oz", "O1", "O2"],
        "official_canonical_id_order": [56, 55, 57, 54, 58, 62, 61, 63],
        "impedance_input_scale": "already_log1p_kohm_over_log101_clipped_0_2",
        "q_builder_reads_external_metadata": False,
        "metadata_builder_reads_waveform_or_Q": False,
        "generic_categorical_id_input": "forbidden",
        "backbone_condition_keys": ["channel_ids", "channel_mask"],
        "stage2_trainable_prefix": "prior.metadata_precision_encoder",
        "checkpoint_writer": (
            "src/cfeg/metadata_calibration_checkpoint.py::write_full_composite_checkpoint"
        ),
        "checkpoint_verifier": (
            "src/cfeg/metadata_calibration_checkpoint.py::verify_full_composite_checkpoint"
        ),
        "checkpoint_schema": "cfeg.metadata-calibration-composite-checkpoint.v1",
        "checkpoint_receipt_schema": "cfeg.metadata-calibration-composite-receipt.v1",
    }
    if (
        formula != expected_formula
        or (method.get("parameter_bounds") or {}) != expected_bounds
        or (method.get("runtime_modes") or {}) != expected_runtime_modes
        or implementation.get("module") != "src/cfeg/models/metadata_calibration_prior.py"
        or implementation.get("class") != "BoundedMetadataCalibrationPrior"
        or any(
            implementation.get(field) != expected
            for field, expected in expected_feature_implementation.items()
        )
        or any(
            implementation.get(field) != "forbidden"
            for field in (
                "query_waveform_transformation",
                "channel_mixing_by_metadata",
                "metadata_class_or_label_input",
            )
        )
    ):
        raise ValueError("The exact diagonal-Gaussian calibration formula drifted.")
    expected_implementation_gates = {
        "positive_definite_or_bounded_reliability_operator",
        "no_label_or_id_path_from_metadata",
        "exact_null_and_all_missing_fallback",
        "frozen_common_path_has_no_M_gradient_or_statistics",
        "correct_metadata_changes_only_the_declared_prior",
        "support_and_query_are_disjoint",
        "every_role_resets_from_the_same_source_checkpoint_at_every_budget",
    }
    if set(method.get("implementation_gate_before_human_outcome") or []) != (
        expected_implementation_gates
    ):
        raise ValueError("The pre-outcome method implementation gates drifted.")

    calibration = plan.get("calibration_contract") or {}
    if tuple(calibration.get("budgets") or []) != METADATA_CALIBRATION_BUDGETS:
        raise ValueError("Calibration budgets must be exactly k=0,1,3,5.")
    if tuple(calibration.get("support_pool_blocks") or []) != (METADATA_CALIBRATION_SUPPORT_BLOCKS):
        raise ValueError("Complete-block support pool drifted.")
    if tuple(calibration.get("immutable_query_blocks") or []) != (
        METADATA_CALIBRATION_QUERY_BLOCKS
    ):
        raise ValueError("Immutable query blocks drifted.")
    expected_support = {
        0: [],
        1: ["block01"],
        3: ["block01", "block02", "block03"],
        5: list(METADATA_CALIBRATION_SUPPORT_BLOCKS),
    }
    observed_support = {
        int(key): value for key, value in (calibration.get("nested_support") or {}).items()
    }
    if observed_support != expected_support:
        raise ValueError("Nested complete-block support contract drifted.")
    expected_labeled_trials = {0: 0, 1: 12, 3: 36, 5: 60}
    observed_labeled_trials = {
        int(key): int(value)
        for key, value in (calibration.get("labeled_trials_per_condition") or {}).items()
    }
    if observed_labeled_trials != expected_labeled_trials:
        raise ValueError("Calibration block-to-trial accounting drifted.")
    if calibration.get("classwise_mixed_block_sampling_allowed") is not False:
        raise ValueError("Classwise mixed-block calibration must remain forbidden.")
    if (
        calibration.get("k_definition")
        != "complete_labeled_blocks_per_participant_per_electrode_condition"
        or calibration.get("partition_mode") != "prospective_chronological_primary"
        or calibration.get("condition_handling")
        != "calibrate_and_score_wet_and_dry_separately_then_equal_average_within_participant"
        or calibration.get("reset_rule")
        != "independent_reset_from_frozen_source_checkpoint_for_each_role_budget_condition"
        or calibration.get("current_generic_calibration_utility_compatible") is not False
    ):
        raise ValueError("The calibration partition, condition, or reset contract drifted.")

    estimands = plan.get("estimands") or {}
    primary = estimands.get("primary") or {}
    if (
        primary.get("name") != "normalized_early_budget_auc_increment"
        or primary.get("budgets") != [0, 1, 3]
        or primary.get("formula") != "eAUC_equals_Y0_over_6_plus_Y1_over_2_plus_Y3_over_3"
        or primary.get("contrast") != "eAUC_A_QM_minus_eAUC_A_Q"
        or primary.get("alternative") != "greater_than_zero"
        or primary.get("alpha") != 0.05
        or primary.get("inference")
        != "one_sided_participant_paired_t_with_sign_flip_and_bootstrap_sensitivity"
        or primary.get("source_training_seeds") != [42, 43, 44]
        or primary.get("seed_reduction") != "average_class_probabilities_per_query_before_argmax"
        or primary.get("tie_break") != "lowest_canonical_class_id"
        or primary.get("any_missing_role_budget_condition_or_seed") != "invalidate_primary"
        or estimands.get("participant_unit") is not True
        or estimands.get("seed_class_block_or_support_draw_as_independent_n") is not False
    ):
        raise ValueError("The single participant-level eAUC primary estimand drifted.")
    savings = estimands.get("equivalent_calibration_savings") or {}
    if (
        savings.get("confirmatory_only_after_primary_pass") is not True
        or savings.get("confirmatory_contrast")
        != "A_QM_k1_noninferior_to_A_Q_k3"
        or savings.get("calibration_value_subgate")
        != {
            "contrast": "A_Q_k3_superior_to_A_Q_k1",
            "alternative": "greater_than_zero",
            "alpha": 0.05,
            "observed_mean_min": 0.020,
            "role": "inferential_claim_admissibility",
        }
        or savings.get("descriptive_contrasts")
        != [
            "A_QM_k0_noninferior_to_A_Q_k1",
            "A_QM_k3_noninferior_to_A_Q_k5",
        ]
        or savings.get("alpha") != 0.025
        or savings.get("noninferiority_margin") != 1.0 / 60.0
        or savings.get("margin_semantic_unit")
        != (
            "one_mean_additional_error_per_interface_sixty_queries_equivalently_"
            "two_total_errors_across_equal_dry_wet_participant_outcome"
        )
        or savings.get("margin_sign_convention")
        != (
            "pass_when_lower_CI_of_lower_budget_A_QM_minus_higher_budget_A_Q_"
            "exceeds_negative_margin"
        )
        or savings.get("operational_balanced_accuracy_floor") != 0.50
        or savings.get("margin_status")
        != "owner_delegated_agent_freeze_one_additional_error_per_sixty_queries"
        or savings.get("inverse_curve_or_interpolated_savings") != "descriptive_only"
        or estimands.get("promotion_sesoi") != 0.020
        or estimands.get("calibration_value_sesoi") != 0.020
        or estimands.get("mechanism_sesoi") != 0.010
        or estimands.get("sesoi_status")
        != "owner_delegated_agent_freeze_two_point_eAUC_increment"
        or estimands.get("held_wrong_context_tail_safety")
        != {
            "severe_harm_cutoff": -0.05,
            "severe_harm_participants_max": 6,
            "catastrophic_harm_floor": -0.10,
            "role": "required_safety_report_not_fixed_sequence_gate",
        }
        or estimands.get("confirmatory_core")
        != {
            "stage_H1": "primary_eAUC_superiority",
            "stage_H2_intersection": [
                "calibration_value_A_Q_k3_over_k1",
                "A_QM_k1_noninferior_to_A_Q_k3",
            ],
            "multiplicity": "serial_gatekeeping_then_intersection_union_no_alpha_split",
        }
        or estimands.get("key_mechanism_family")
        != {
            "endpoints": [
                "correct_context_over_block_shuffle",
                "correct_context_over_stale",
            ],
            "inference": "one_sided_participant_paired_t",
            "multiplicity": "holm_alpha_0p05_after_confirmatory_core",
            "stale_role": "supportive_sensitivity",
        }
        or estimands.get("deployment_safety")
        != {
            "wrong_context_mean": "one_sided_noninferiority_alpha_0p025",
            "wrong_context_tail": "required_participant_tail_report",
            "failure_effect": (
                "forbid_robust_or_safe_deployment_claim_without_reversing_"
                "correct_context_efficacy"
            ),
        }
    ):
        raise ValueError("The calibration-savings or unresolved SESOI contract drifted.")

    development_gate = plan.get("source_development_gate") or {}
    expected_components = {
        "nested_source_held_participant_eAUC_A_QM_minus_A_Q",
        "correct_context_minus_within_interface_block_shuffle",
        "all_missing_exact_fallback",
        "metadata_only_shortcut_screen",
    }
    expected_hard_gates = [
        "baseline_viability",
        "early_budget_mean",
        "early_budget_every_fold_positive",
        "correct_minus_shuffle_mean",
        "correct_minus_shuffle_no_negative_fold",
        "missing_exact_fallback",
        "metadata_only_shortcut",
    ]
    expected_reports = [
        "early_budget_median",
        "early_budget_positive_participants",
        "correct_minus_shuffle_positive_participants",
        "correct_minus_stale_mean",
        "correct_minus_stale_positive_participants",
        "correct_minus_stale_no_negative_fold",
        "wrong_context_mean_safety",
        "wrong_context_severe_harm_count",
        "wrong_context_no_catastrophic_harm",
        "correct_context_severe_harm_count",
        "correct_context_no_catastrophic_harm",
    ]
    expected_thresholds = {
        "baseline_A_Q_k5_mean_min": 0.50,
        "eAUC_increment_mean_min": 0.010,
        "eAUC_increment_median_strictly_positive": True,
        "eAUC_positive_participants_min": 24,
        "eAUC_every_outer_fold_strictly_positive": True,
        "correct_minus_shuffle_mean_min": 0.010,
        "correct_minus_stale_mean_min": 0.010,
        "mechanism_positive_participants_min": 24,
        "mechanism_outer_fold_mean_min": 0.0,
        "all_missing_probability_max_abs_difference": 0.0,
        "wrong_context_mean_harm_floor": -(1.0 / 60.0),
        "severe_harm_cutoff": -0.05,
        "severe_harm_participants_max": 4,
        "catastrophic_harm_floor": -0.10,
        "metadata_only_balanced_accuracy_exact": 1.0 / 12.0,
        "metadata_block_constant_within_complete_class_balanced_block": True,
    }
    if (
        development_gate.get("cohort") != "exact_39_subject_source_development_allocation"
        or development_gate.get("held_60_access_before_pass") != "forbidden"
        or set(development_gate.get("required_components") or []) != expected_components
        or development_gate.get("hard_promotion_gates") != expected_hard_gates
        or development_gate.get("required_diagnostic_or_deployment_reports")
        != expected_reports
        or (development_gate.get("numeric_thresholds") or {}) != expected_thresholds
        or development_gate.get("outer_gate_folds") != {"count": 3, "participants_per_fold": 13}
        or development_gate.get("metadata_only_screen")
        != {
            "method": "deterministic_information_structure_proof",
            "requirements": [
                "M_exactly_constant_within_each_block",
                "each_block_contains_exactly_one_trial_per_each_of_twelve_classes",
                "metadata_only_API_has_no_row_order_or_identity",
            ],
            "implied_balanced_accuracy": 1.0 / 12.0,
        }
        or development_gate.get("status")
        != (
            "minimal_nonredundant_promotion_gate_implemented_reports_preserved_"
            "atomic_runner_frozen"
        )
    ):
        raise ValueError("The source-development gate or its blocked status drifted.")

    atomic = plan.get("atomic_bundle_contract") or {}
    expected_held_cells = [
        {"role": "A_0", "context": "exact_off"},
        {"role": "A_M", "context": "correct"},
        {"role": "A_Q", "context": "exact_off"},
        {"role": "A_QM", "context": "correct"},
        {"role": "A_QM", "context": "all_missing"},
        {"role": "A_QM", "context": "block_shuffle"},
        {"role": "A_QM", "context": "stale"},
        {"role": "A_QM", "context": "opposite_interface"},
    ]
    expected_development_diagnostics = [
        {
            "role": "A_QM",
            "context": "support_metadata_only",
            "intervention": "correct",
            "query_metadata_mode": "all_missing",
            "support_metadata_mode": "observed",
            "feature_ablation": "full",
        },
        {
            "role": "A_QM",
            "context": "query_metadata_only",
            "intervention": "correct",
            "query_metadata_mode": "observed",
            "support_metadata_mode": "all_missing",
            "feature_ablation": "full",
        },
        {
            "role": "A_QM",
            "context": "interface_only",
            "intervention": "correct",
            "query_metadata_mode": "observed",
            "support_metadata_mode": "observed",
            "feature_ablation": "interface_only",
        },
        {
            "role": "A_QM",
            "context": "impedance_only",
            "intervention": "correct",
            "query_metadata_mode": "observed",
            "support_metadata_mode": "observed",
            "feature_ablation": "impedance_only",
        },
    ]
    if (
        atomic.get("held_cells") != expected_held_cells
        or atomic.get("development_additional_diagnostics")
        != expected_development_diagnostics
        or atomic.get("development_structural_checks")
        != {
            "metadata_only_shortcut": (
                "exact_chance_from_block_constant_M_and_one_trial_per_class_per_block"
            )
        }
        or atomic.get("probability_staging_contains_query_outcomes") is not False
        or atomic.get("staging_query_identifier") != "opaque_q_HMAC_SHA256_token"
        or atomic.get("raw_sample_id_in_staging_allowed") is not False
        or atomic.get("raw_h5_index_downstream_allowed") is not False
        or atomic.get("sealed_query_signal_artifact")
        != "token_order_permuted_query_signals_h5_with_x_and_channel_mask_only"
        or atomic.get("sealed_signal_index_semantics")
        != "local_index_into_sealed_query_signals_not_source_h5_index"
        or atomic.get("sealed_query_row_order")
        != "ascending_opaque_query_token_never_source_order"
        or atomic.get("query_token_secret_persisted_in_execution_manifest") is not False
        or atomic.get("query_token_secret_hash_bound") is not True
        or atomic.get("opaque_row_token_builder")
        != "src/cfeg/data/unlabeled_query.py::opaque_metadata_row_token"
        or atomic.get("token_secret_commitment_builder")
        != "src/cfeg/data/unlabeled_query.py::token_secret_commitment_sha256"
        or atomic.get("unlabeled_query_reader")
        != "src/cfeg/data/unlabeled_query.py::UnlabeledQueryReader"
        or atomic.get("unlabeled_query_collator")
        != (
            "src/cfeg/data/unlabeled_query.py::"
            "collate_metadata_calibration_queries"
        )
        or atomic.get("unlabeled_query_base_input_bytes_rehashed_on_read") is not True
        or atomic.get("label_free_partition_builder")
        != (
            "src/cfeg/metadata_calibration_contract.py::"
            "build_complete_block_calibration_partition"
        )
        or atomic.get("privileged_raw_manifest_auditor")
        != (
            "src/cfeg/metadata_calibration_contract.py::"
            "audit_complete_block_cohort_partition"
        )
        or atomic.get("input_seal_decision_template")
        != (
            "configs/governance/"
            "metadata_calibration_input_seal_decision.template.json"
        )
        or atomic.get("input_artifact_contract")
        != "configs/analysis/metadata_calibration_input_artifacts_v1.yaml"
        or atomic.get("input_seal_execution_status")
        != "authorized_pending_atomic_lifecycle"
        or atomic.get("input_seal_phase_scope")
        != "separate_source_development_then_held_after_passing_gate_receipts"
        or atomic.get("sealed_source_fit_and_support_signals")
        != "token_order_no_y_HDF5_required"
        or atomic.get("token_domain_separation")
        != "phase_checkpoint_group_and_purpose_bound_to_seal_decision"
        or atomic.get("finalizer_only_query_label_capability")
        != "selected_public_key_encryption_plus_bubblewrap_worker_namespace"
        or atomic.get("production_calls_privileged_raw_manifest_auditor") is not True
        or atomic.get("block_intervention_mapping_unit")
        != "one_row_per_participant_interface_block_no_sample_id"
        or atomic.get("cell_execution_dispatch")
        != (
            "src/cfeg/metadata_calibration_execution.py::"
            "resolve_cell_execution_spec"
        )
        or atomic.get("execution_contract_builder")
        != (
            "src/cfeg/metadata_calibration_manifest.py::"
            "build_metadata_calibration_execution_contract"
        )
        or atomic.get("execution_contract_validator")
        != (
            "src/cfeg/metadata_calibration_manifest.py::"
            "validate_metadata_calibration_execution_contract"
        )
        or atomic.get("execution_contract_grants_authority") is not False
        or atomic.get("candidate_job_unit")
        != "one_seed_checkpoint_group_covers_all_declared_cells_and_budgets"
        or atomic.get("baseline_job_unit")
        != (
            "one_adapter_checkpoint_group_covers_all_declared_budgets_with_seed_null"
        )
        or atomic.get("source_development_expected_jobs")
        != {"candidate": 9, "baseline": 15, "total": 24}
        or atomic.get("held_expected_jobs")
        != {"candidate": 3, "baseline": 5, "total": 8}
        or atomic.get("governed_candidate_model_entrypoint")
        != (
            "src/cfeg/models/metadata_calibration_composite.py::"
            "MetadataCalibrationComposite.forward_cell_from_precomputed"
        )
        or atomic.get("governed_candidate_cache_builder")
        != (
            "src/cfeg/models/metadata_calibration_composite.py::"
            "MetadataCalibrationComposite.prepare_evaluation_features"
        )
        or atomic.get("cache_scope")
        != "episode_query_once_and_support_once_per_exact_budget"
        or atomic.get("cached_paths") != ["spectral_embedding", "observed_Q"]
        or atomic.get("cached_condition_keys")
        != [
            "channel_ids",
            "channel_mask",
            "channel_query_qc",
            "channel_query_qc_missing",
            "sfreq_processed_float",
            "metadata_contract_version",
        ]
        or atomic.get("raw_external_context_M_in_cache") != "forbidden"
        or atomic.get("base_input_hash_cuda_transfer")
        != "one_batched_copy_per_prepared_query_or_support_batch"
        or atomic.get("noncached_paths")
        != ["resolved_M", "metadata_ablation", "posterior_update"]
        or atomic.get("probability_cpu_materialization")
        != "one_order_preserving_copy_per_role_context_budget_grid"
        or atomic.get("production_direct_low_level_composite_forward") != "forbidden"
        or atomic.get("producer_receipt_binds_cell_execution_contract_sha256")
        is not True
        or atomic.get("producer_receipt_binds_intervention_mapping_sha256") is not True
        or atomic.get("producer_receipt_binds_resolved_context_values_sha256")
        is not True
        or atomic.get("producer_receipt_binds_resolved_context_usage_sha256")
        is not True
        or atomic.get(
            "private_candidate_staging_binds_cell_dispatch_and_resolved_context_usage"
        )
        is not True
        or atomic.get("staging_column_policy") != "exact_allowlist_reject_extras"
        or atomic.get("all_missing_equality_validation") != "before_label_join"
        or atomic.get("expected_support_artifact")
        != "sample_level_rows_rehashed_by_finalizer"
        or atomic.get("producer_job_receipts_required") is not True
        or atomic.get("production_finalizer_api")
        != "execution_manifest_path_and_sealed_staging_path_only"
        or atomic.get("label_join") != "only_after_complete_private_grid_validation"
        or atomic.get("held_checkpoint_key") != ["seed"]
        or atomic.get("development_checkpoint_key") != ["seed", "outer_fold"]
        or atomic.get("full_composite_checkpoint_required") is not True
        or atomic.get("trainable_only_checkpoint_allowed") is not False
    ):
        raise ValueError("The outcome-private atomic bundle contract drifted.")

    controls = plan.get("mechanism_and_safety_controls") or {}
    if (
        controls.get("intervention_builder")
        != (
            "src/cfeg/data/metadata_calibration_interventions.py::"
            "build_metadata_intervention_mapping"
        )
        or controls.get("intervention_batch_resolver")
        != (
            "src/cfeg/data/metadata_calibration_interventions.py::"
            "resolve_metadata_intervention_batch"
        )
        or controls.get("intervention_condition_applier")
        != (
            "src/cfeg/data/metadata_calibration_interventions.py::"
            "apply_resolved_metadata_to_condition"
        )
        or controls.get("intervention_usage_hasher")
        != (
            "src/cfeg/data/metadata_calibration_interventions.py::"
            "resolved_context_usage_sha256"
        )
        or controls.get("resolver_row_binding")
        != "purpose_matched_opaque_token_plus_actual_EEG_base_input_sha256"
        or controls.get("resolver_exact_token_coverage_required") is not True
        or controls.get("resolver_phase_and_canonical_cell_bound") is not True
        or controls.get("block_context_impedance_channel_ids")
        != [56, 55, 57, 54, 58, 62, 61, 63]
        or controls.get("block_shuffle_seed") != 20260904
        or controls.get("block_shuffle_seed_bound_in_mapping_sha256") is not True
        or controls.get("block_shuffle_scope")
        != "within_participant_interface_and_within_support_or_query_pool"
        or controls.get("stale_definition")
        != "true_previous_block_with_block01_all_missing"
        or controls.get("opposite_interface_definition")
        != "same_participant_same_block_other_interface"
        or controls.get("opposite_interface_replaces_interface_and_impedance") is not True
        or controls.get("per_row_all_missing_masks_all_18_metadata_features") is not True
        or controls.get("same_checkpoint_interventions")
        != [
            "correct_context",
            "all_missing_context",
            "within_interface_block_shuffled_impedance",
            "stale_block_impedance",
            "opposite_interface_context",
            "support_metadata_only",
            "query_metadata_only",
        ]
        or controls.get("ablations") != ["interface_only", "impedance_only"]
        or controls.get("required_interpretation")
        != {
            "correct_not_better_than_shuffled": (
                "metadata_specific_mechanism_not_established"
            ),
            "missing_or_wrong_harms_A_Q": "unsafe_negative_transfer",
            "metadata_only_above_ceiling": "shortcut_or_confound",
        }
        or controls.get("synthetic_corruption_role")
        != "engineering_robustness_secondary_not_primary"
    ):
        raise ValueError("The block-level metadata intervention contract drifted.")

    power = plan.get("power_sensitivity") or {}
    if (
        power.get("held_participant_n") != 60
        or power.get("development_participant_n") != 39
        or power.get("master_seed") != 20260904
        or power.get("monte_carlo_draws_per_cell") != 200000
        or power.get("chunk_size") != 10000
        or power.get("sd_semantics")
        != {
            "participant_difference_sd_grid": (
                "participant_level_primary_eAUC_difference_sd"
            ),
            "noninferiority_participant_difference_sd_grid": (
                "participant_level_A_QM_k1_minus_A_Q_k3_"
                "balanced_accuracy_difference_sd"
            ),
            "core_planning_sds_by_endpoint": {
                "primary_eAUC_superiority": "participant_level_eAUC_difference_sd",
                "calibration_value_A_Q_k3_over_k1": (
                    "participant_level_balanced_accuracy_difference_sd"
                ),
                "A_QM_k1_noninferior_to_A_Q_k3": (
                    "participant_level_balanced_accuracy_difference_sd"
                ),
            },
        }
        or power.get("participant_difference_sd_grid") != [0.03, 0.04, 0.06, 0.08]
        or power.get("noninferiority_participant_difference_sd_grid")
        != [0.03, 0.04, 0.06, 0.08]
        or power.get("held_endpoint_order")
        != [
            "primary_eAUC_superiority",
            "calibration_value_A_Q_k3_over_k1",
            "A_QM_k1_noninferior_to_A_Q_k3",
        ]
        or power.get("core_planning_means") != [0.030, 0.030, 0.000]
        or power.get("core_planning_sds") != [0.060, 0.060, 0.045]
        or power.get("core_partial_null_means")
        != {
            "H2a_boundary_other_components_alternative": [0.030, 0.000, 0.030],
            "H2b_boundary_other_components_alternative": [
                0.030,
                0.030,
                -(1.0 / 60.0),
            ],
        }
        or power.get("core_planning_sample_size_sensitivity") != [60, 75, 80, 83, 90]
        or power.get("core_contrast_correlation_scenarios")
        != {
            "independent": [
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
            ],
            "shared_A_Q_k3": [
                [1.0, 0.0, 0.3],
                [0.0, 1.0, -0.5],
                [0.3, -0.5, 1.0],
            ],
            "adverse_shared_A_Q_k3": [
                [1.0, -0.2, 0.4],
                [-0.2, 1.0, -0.7],
                [0.4, -0.7, 1.0],
            ],
        }
        or power.get("core_endpoint_semantics")
        != [
            "eAUC_A_QM_minus_A_Q",
            "A_Q_k3_minus_A_Q_k1",
            "A_QM_k1_minus_A_Q_k3",
        ]
        or power.get("core_power_is_conditional_on_absolute_endpoint_BA_floor")
        is not True
        or power.get("held_mechanism_power")
        != {
            "endpoint_order": [
                "correct_context_over_block_shuffle",
                "correct_context_over_stale",
            ],
            "planning_means": [0.020, 0.020],
            "planning_sds": [0.040, 0.040],
            "endpoint_correlations": [0.0, 0.5, 0.8],
            "practical_mean_min": 0.010,
            "holm_family_alpha": 0.05,
            "conditional_on_confirmatory_core_open": True,
            "not_a_joint_pipeline_probability": True,
        }
        or power.get("development_gate_scenarios")
        != {
            "planning": {
                "contrast_means": [0.020, 0.020, 0.020, 0.000],
                "contrast_sds": [0.040, 0.040, 0.040, 0.030],
                "baseline_mean": 0.55,
                "baseline_sd": 0.040,
                "endpoint_correlation": 0.5,
                "families": [
                    "normal",
                    "standardized_t5",
                    "twenty_percent_harmed_mixture",
                ],
            },
            "metadata_null_viable_baseline": {
                "contrast_means": [0.000, 0.000, 0.000, 0.000],
                "contrast_sds": [0.040, 0.040, 0.040, 0.030],
                "baseline_mean": 0.55,
                "baseline_sd": 0.040,
                "endpoint_correlation": 0.5,
                "families": ["normal"],
            },
            "joint_bad_null": {
                "contrast_means": [0.000, 0.000, 0.000, 0.000],
                "contrast_sds": [0.040, 0.040, 0.040, 0.030],
                "baseline_mean": 0.48,
                "baseline_sd": 0.040,
                "endpoint_correlation": 0.5,
                "families": ["normal"],
            },
        }
        or power.get("distinguish_inferential_power_from_full_claim_probability") is not True
        or power.get("receipt_status") != "final_receipt_required_and_bound"
        or power.get("receipt_path") != METADATA_CALIBRATION_POWER_RECEIPT
    ):
        raise ValueError("The outcome-free power sensitivity contract drifted.")
    baseline = plan.get("baseline_contract") or {}
    if (
        baseline.get("information_rights_ledger") != METADATA_CALIBRATION_BASELINE_LEDGER
        or baseline.get("required_resource_matched")
        != [
            "strict_FBCCA_at_k0",
            "target_template_at_k1_k3_k5",
            "target_filterbank_eTRCA_at_k3_k5",
            "SAME3_filterbank_eTRCA_one_shot_component_at_k1",
            "source_transfer_without_metadata",
            "same_checkpoint_A_Q_at_k0_k1_k3_k5",
        ]
        or baseline.get("selected_direct_comparator")
        != "chiang2021_LST_filterbank_eTRCA_common_protocol_port"
        or baseline.get("selected_one_shot_component_comparator")
        != "SAME3_filterbank_eTRCA_common_protocol_port"
        or baseline.get("high_value_direct_comparators")
        != [
            "SSVEP_DAN",
            "CSDuDoFN_OS_SSVEP_one_shot_lineage",
            "TDCA_or_SAME_TDCA",
        ]
    ):
        raise ValueError("The baseline information-rights ledger path drifted.")
    blockers = plan.get("execution_freeze_blockers")
    resolved_controls = set(plan.get("resolved_execution_freeze_controls") or [])
    required_controls = {
        "privileged_input_sealer_and_receipt",
        "model_and_mandatory_baseline_producer_receipts_and_execution_manifest_binding",
        "high_value_direct_comparator_feasibility_and_information_rights_resolution",
        "production_finalizer_and_atomic_all_role_all_budget_runner",
        "clean_source_tag_and_hashes",
        "new_exact_owner_decision_receipt_and_authorization_envelope",
    }
    if blockers != [] or resolved_controls != required_controls:
        raise ValueError("The resolved execution-freeze controls drifted.")

    recipe_binding = validate_metadata_calibration_source_recipe(
        _repository_root() / METADATA_CALIBRATION_SOURCE_RECIPE,
        expected_source_subjects=tuple(
            data["proposed_unopened_allocation"]["source_development_subject_ids"]
        ),
    )
    baseline_binding = validate_metadata_calibration_baseline_ledger(
        _repository_root() / METADATA_CALIBRATION_BASELINE_LEDGER
    )
    if baseline_binding.get("outcome_execution_authorized") is not True:
        raise ValueError("The mandatory baseline ledger is not execution-ready.")
    input_artifact_binding = validate_metadata_calibration_input_artifact_contract(
        _repository_root() / METADATA_CALIBRATION_INPUT_ARTIFACT_CONTRACT
    )
    result = {
        **expected_header,
        "plan_sha256": _sha256_file(plan_path),
        "source_recipe_sha256": recipe_binding["source_recipe_sha256"],
        "baseline_ledger_sha256": baseline_binding["baseline_ledger_sha256"],
        "input_artifact_contract_sha256": input_artifact_binding[
            "input_artifact_contract_sha256"
        ],
        "outcome_execution_authorized": True,
        "authorization_basis": authority["approval_message"],
    }
    if require_power_receipt:
        power_binding = validate_metadata_calibration_power_receipt(
            _repository_root() / METADATA_CALIBRATION_POWER_RECEIPT,
            expected_plan_sha256=result["plan_sha256"],
            expected_power_config=power,
        )
        result["power_receipt_sha256"] = power_binding["power_receipt_sha256"]
    return result


def validate_metadata_calibration_input_artifact_contract(
    path: str | Path | None = None,
) -> dict[str, object]:
    """Validate the label-separating artifact schema without opening human data."""

    contract_path = (
        Path(path)
        if path is not None
        else _repository_root() / METADATA_CALIBRATION_INPUT_ARTIFACT_CONTRACT
    )
    contract = load_config(contract_path, strict_env=False)
    if _sha256_file(contract_path) != METADATA_CALIBRATION_INPUT_ARTIFACT_CONTRACT_SHA256:
        raise ValueError("Metadata-calibration input artifact contract bytes drifted.")
    phases = contract.get("phase_contract") or {}
    source = phases.get("source_development") or {}
    held = phases.get("held_participant_evaluation") or {}
    tokens = contract.get("token_contract") or {}
    signals = contract.get("signal_artifact_contract") or {}
    block_context = contract.get("block_context_contract") or {}
    publication = contract.get("publication_contract") or {}
    label_capability = contract.get("label_capability_contract") or {}
    covariates = contract.get("analysis_covariate_contract") or {}
    source_privacy = contract.get("source_development_outcome_privacy") or {}
    held_privacy = contract.get("held_participant_outcome_privacy") or {}
    receipt = contract.get("receipt_contract") or {}
    if (
        contract.get("schema") != "cfeg.metadata-calibration-input-artifacts.v1"
        or contract.get("candidate_id") != "metadata-calibration-efficiency-v1"
        or contract.get("status")
        != "frozen_schema_implemented_signed_runtime_authority_required"
        or source.get("participant_count") != 39
        or source.get("checkpoint_groups") != ["fold0", "fold1", "fold2"]
        or source.get("held_subject_access") != "forbidden"
        or source.get("target_subject_scope")
        != "exact_source_development_39_cross_fitted_outer_evaluation"
        or source.get("final_refit_source_participants")
        != "forbidden_in_source_development_phase"
        or source.get("source_fit_artifact_policy")
        != "phase_scoped_seal_exact_fold_specific_26"
        or (source.get("source_fit_groups") or {})
        != {
            "per_fold_inner_fit_participants": 20,
            "per_fold_inner_validation_participants": 6,
            "per_fold_outer_evaluation_participants": 13,
        }
        or held.get("participant_count") != 60
        or held.get("checkpoint_groups") != ["held"]
        or held.get("target_subject_scope") != "exact_held_60"
        or held.get("final_refit_source_participants") != 39
        or held.get("source_fit_artifact_policy")
        != "phase_scoped_reseal_exact_source_39_under_held_domain"
        or held.get("held_gate_receipts")
        != [
            "source_development_gate_receipt_sha256",
            "source_development_private_seal_sha256",
        ]
        or tokens.get("builder")
        != "src/cfeg/data/unlabeled_query.py::opaque_metadata_row_token"
        or tokens.get("secret_commitment")
        != "src/cfeg/data/unlabeled_query.py::token_secret_commitment_sha256"
        or tokens.get("master_secret_minimum_bytes") != 32
        or tokens.get("purpose_prefixes")
        != {"evaluation_query": "q_", "evaluation_support": "s_", "source_fit": "f_"}
        or tokens.get("raw_sample_id_or_token_key_persisted") is not False
        or signals.get("exact_hdf5_datasets") != ["channel_mask", "x"]
        or signals.get("forbidden_datasets") != ["y"]
        or signals.get("required_for")
        != ["evaluation_query", "evaluation_support", "source_fit"]
        or block_context.get("canonical_channel_ids")
        != [56, 55, 57, 54, 58, 62, 61, 63]
        or set(contract.get("forbidden_downstream_fields") or [])
        != {
            "raw_sample_id",
            "raw_h5_index",
            "source_row_position",
            "raw_trial_id",
            "target_frequency_or_phase",
            "source_file",
            "token_secret",
        }
        or publication.get("canonical_root_must_be_new") is not True
        or publication.get("publish_operation")
        != "single_directory_renameat2_RENAME_NOREPLACE_then_parent_fsync"
        or publication.get("query_label_capability_boundary")
        != "finalizer_public_key_encryption_plus_bubblewrap_worker_namespace"
        or label_capability.get("encryption_algorithm")
        != "RSA-OAEP-SHA256+AES-256-GCM"
        or label_capability.get("encrypted_sidecar_plaintext_hash_exposed") is not False
        or label_capability.get("selected_mode")
        != "finalizer_public_key_encryption_plus_bubblewrap_worker_namespace"
        or label_capability.get("private_key_files_password_encrypted") is not True
        or label_capability.get("private_key_passwords_parent_memory_only") is not True
        or label_capability.get("label_capable_parent_dumpable") is not False
        or label_capability.get("label_capable_parent_core_dump_limit_bytes") != 0
        or label_capability.get("worker_namespace")
        != "bubblewrap_unshare_all_cap_drop_all_no_new_privileges"
        or label_capability.get("worker_processed_asset_root_mounted") is not False
        or label_capability.get("worker_private_key_root_mounted") is not False
        or label_capability.get("worker_output_mount")
        != "fresh_per_job_spool_at_signed_execution_path"
        or label_capability.get("worker_prior_job_outputs_visible") is not False
        or label_capability.get("python_secret_memory_zeroization_guaranteed") is not False
        or covariates.get("encrypted_finalizer_only_fields")
        != ["query_token", "headband_order", "condition_period", "block_number"]
        or covariates.get("use") != "descriptive_balance_and_design_invariant_audit_only"
        or covariates.get("decoder_input") != "forbidden"
        or covariates.get("confirmatory_claim_adjustment") != "none_prespecified"
        or covariates.get("block_number_query_scope") != [6, 7, 8, 9, 10]
        or source_privacy.get("scope") != "checkpoint_group"
        or source_privacy.get("claim")
        != "scientific_job_isolation_not_phase_wide_cryptographic_blinding"
        or held_privacy.get("scope") != "whole_phase"
        or held_privacy.get("held_60_labels_appear_in_any_training_artifact") is not False
        or held_privacy.get("query_labels_encrypted_until_complete_private_grid_is_sealed")
        is not True
        or receipt.get("runner_visible_per_query_label_hash_forbidden") is not True
        or receipt.get("grants_training_scoring_or_outcome_authority") is not False
    ):
        raise ValueError("Metadata-calibration input artifact contract drifted.")
    return {
        "schema": contract["schema"],
        "candidate_id": contract["candidate_id"],
        "input_artifact_contract_sha256": _sha256_file(contract_path),
        "human_input_access_authorized": False,
    }


def validate_metadata_calibration_power_receipt(
    path: str | Path | None = None,
    *,
    expected_plan_sha256: str,
    expected_power_config: Mapping[str, object],
) -> dict[str, object]:
    receipt_path = (
        Path(path)
        if path is not None
        else _repository_root() / METADATA_CALIBRATION_POWER_RECEIPT
    )
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    claimed_payload_hash = receipt.pop("receipt_payload_sha256", None)
    observed_payload_hash = hashlib.sha256(
        json.dumps(receipt, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()
    if claimed_payload_hash != observed_payload_hash:
        raise ValueError("Metadata-calibration power receipt payload hash is invalid.")
    expected_files = {
        "power_script_sha256": _repository_root()
        / "scripts/analyze_metadata_calibration_power.py",
        "power_module_sha256": _repository_root()
        / "src/cfeg/analysis/metadata_calibration_power.py",
        "analysis_module_sha256": _repository_root()
        / "src/cfeg/analysis/metadata_calibration_efficiency.py",
    }
    expected_power_hash = hashlib.sha256(
        json.dumps(
            expected_power_config,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode()
    ).hexdigest()
    if (
        receipt.get("schema") != "cfeg.metadata-calibration-power-receipt.v1"
        or receipt.get("status") != "outcome_free_design_sensitivity_not_execution_authority"
        or receipt.get("candidate_id") != "metadata-calibration-efficiency-v1"
        or receipt.get("outcome_data_used") is not False
        or receipt.get("data_dependencies") != []
        or receipt.get("n_development") != 39
        or receipt.get("n_held") != 60
        or receipt.get("monte_carlo_draws_per_cell") != 200000
        or receipt.get("plan_sha256") != expected_plan_sha256
        or receipt.get("power_config_sha256") != expected_power_hash
        or not receipt.get("held_core_hierarchy_monte_carlo")
        or not receipt.get("held_mechanism_holm_monte_carlo")
        or "held_fixed_sequence_monte_carlo" in receipt
        or (receipt.get("decision_rules") or {}).get("confirmatory_savings")
        != "A_QM_k1_noninferior_to_A_Q_k3_only"
        or any(receipt.get(field) != _sha256_file(file_path) for field, file_path in expected_files.items())
    ):
        raise ValueError("Metadata-calibration final power receipt binding drifted.")
    expected_mc_se = 0.5 / np.sqrt(float(receipt["monte_carlo_draws_per_cell"]))
    if (
        receipt.get("master_seed") != expected_power_config.get("master_seed")
        or receipt.get("chunk_size") != expected_power_config.get("chunk_size")
        or not np.isclose(
            float(receipt.get("maximum_nominal_mc_standard_error", np.nan)),
            expected_mc_se,
            rtol=0.0,
            atol=1e-15,
        )
    ):
        raise ValueError("Metadata-calibration receipt simulation configuration drifted.")

    planning_means = expected_power_config["core_planning_means"]
    planning_sds = expected_power_config["core_planning_sds"]
    matrices = expected_power_config["core_contrast_correlation_scenarios"]
    expected_core: dict[str, dict[str, object]] = {}
    for matrix_name, matrix in matrices.items():
        expected_core[f"planning_{matrix_name}"] = {
            "n": 60,
            "true_means": planning_means,
            "sds": planning_sds,
            "correlation_matrix": matrix,
            "family": "normal",
            "utility_floor_assumed": True,
        }
    for family in expected_power_config["distribution_families"]:
        if family != "normal":
            expected_core[f"planning_shared_A_Q_k3_{family}"] = {
                "n": 60,
                "true_means": planning_means,
                "sds": planning_sds,
                "correlation_matrix": matrices["shared_A_Q_k3"],
                "family": family,
                "utility_floor_assumed": True,
            }
    for scenario, means in {
        "global_null": [0.0, 0.0, -(1.0 / 60.0)],
        "calibration_plateau": [0.03, 0.0, 0.0],
        "harmful_low_budget": [0.03, 0.03, -0.03],
        "sesoi_boundary": [0.02, 0.02, 0.0],
        **expected_power_config["core_partial_null_means"],
    }.items():
        expected_core[scenario] = {
            "n": 60,
            "true_means": means,
            "sds": planning_sds,
            "correlation_matrix": matrices["shared_A_Q_k3"],
            "family": "normal",
            "utility_floor_assumed": True,
        }
    _validate_power_cells(
        receipt["held_core_hierarchy_monte_carlo"],
        expected=expected_core,
        kind="held_core_hierarchy",
        master_seed=int(receipt["master_seed"]),
        draws=int(receipt["monte_carlo_draws_per_cell"]),
        expected_mc_se=expected_mc_se,
    )

    expected_sample_rows = {
        f"planning_shared_A_Q_k3_n{int(n)}": {
            "n": int(n),
            "true_means": planning_means,
            "sds": planning_sds,
            "correlation_matrix": matrices["shared_A_Q_k3"],
            "family": "normal",
            "utility_floor_assumed": True,
        }
        for n in expected_power_config["core_planning_sample_size_sensitivity"]
    }
    _validate_power_cells(
        receipt.get("held_core_sample_size_sensitivity") or [],
        expected=expected_sample_rows,
        kind="held_core_hierarchy",
        master_seed=int(receipt["master_seed"]),
        draws=int(receipt["monte_carlo_draws_per_cell"]),
        expected_mc_se=expected_mc_se,
    )

    mechanism = expected_power_config["held_mechanism_power"]
    expected_mechanism: dict[str, dict[str, object]] = {}
    for correlation in mechanism["endpoint_correlations"]:
        expected_mechanism[f"planning_correlation_{float(correlation):g}"] = {
            "n": 60,
            "true_means": mechanism["planning_means"],
            "sds": mechanism["planning_sds"],
            "endpoint_correlation": float(correlation),
            "family": "normal",
            "practical_threshold": mechanism["practical_mean_min"],
        }
    for family in expected_power_config["distribution_families"]:
        if family != "normal":
            expected_mechanism[f"planning_correlation_0.5_{family}"] = {
                "n": 60,
                "true_means": mechanism["planning_means"],
                "sds": mechanism["planning_sds"],
                "endpoint_correlation": 0.5,
                "family": family,
                "practical_threshold": mechanism["practical_mean_min"],
            }
    _validate_power_cells(
        receipt["held_mechanism_holm_monte_carlo"],
        expected=expected_mechanism,
        kind="held_mechanism_holm",
        master_seed=int(receipt["master_seed"]),
        draws=int(receipt["monte_carlo_draws_per_cell"]),
        expected_mc_se=expected_mc_se,
    )

    expected_development: dict[str, dict[str, object]] = {}
    for scenario, scenario_spec in expected_power_config[
        "development_gate_scenarios"
    ].items():
        for family in scenario_spec["families"]:
            key = f"{scenario}:{family}"
            expected_development[key] = {
                "n": 39,
                "contrast_means": scenario_spec["contrast_means"],
                "contrast_sds": scenario_spec["contrast_sds"],
                "baseline_mean": scenario_spec["baseline_mean"],
                "baseline_sd": scenario_spec["baseline_sd"],
                "endpoint_correlation": scenario_spec["endpoint_correlation"],
                "family": family,
            }
    _validate_power_cells(
        receipt.get("development_gate_monte_carlo") or [],
        expected=expected_development,
        kind="development_gate",
        master_seed=int(receipt["master_seed"]),
        draws=int(receipt["monte_carlo_draws_per_cell"]),
        expected_mc_se=expected_mc_se,
        scenario_key_with_family=True,
    )
    held_scenarios = {
        str(row.get("scenario")): row
        for row in receipt["held_core_hierarchy_monte_carlo"]
    }
    partial_nulls = {
        "H2a_boundary_other_components_alternative": (
            "calibration_value_A_Q_k3_over_k1",
            0.05,
        ),
        "H2b_boundary_other_components_alternative": (
            "A_QM_k1_noninferior_to_A_Q_k3",
            0.025,
        ),
    }
    for scenario, (endpoint, alpha) in partial_nulls.items():
        row = held_scenarios.get(scenario)
        if row is None:
            raise ValueError("Metadata-calibration power receipt omits an H2 partial null.")
        if not np.isclose(
            float(row["maximum_nominal_mc_standard_error"]),
            expected_mc_se,
            rtol=0.0,
            atol=1e-15,
        ):
            raise ValueError("Metadata-calibration power receipt has a false MC error bound.")
        local_probability = float(row["local_claim_probability"][endpoint])
        full_probability = float(row["calibration_saving_full_claim_probability"])
        if (
            local_probability > alpha + 4.0 * expected_mc_se
            or full_probability > alpha + 4.0 * expected_mc_se
        ):
            raise ValueError("Metadata-calibration H2 partial-null error control failed.")
    sample_rows = receipt.get("held_core_sample_size_sensitivity") or []
    if [int(row.get("n", -1)) for row in sample_rows] != [60, 75, 80, 83, 90]:
        raise ValueError("Metadata-calibration sample-size sensitivity grid drifted.")
    expected_hard_gates = [
        "baseline_viability",
        "early_budget_mean",
        "early_budget_every_fold_positive",
        "correct_minus_shuffle_mean",
        "correct_minus_shuffle_no_negative_fold",
        "missing_exact_fallback",
        "metadata_only_shortcut",
    ]
    expected_reports = [
        "early_budget_median",
        "early_budget_positive_participants",
        "correct_minus_shuffle_positive_participants",
        "correct_minus_stale_mean",
        "correct_minus_stale_positive_participants",
        "correct_minus_stale_no_negative_fold",
        "wrong_context_mean_safety",
        "wrong_context_severe_harm_count",
        "wrong_context_no_catastrophic_harm",
        "correct_context_severe_harm_count",
        "correct_context_no_catastrophic_harm",
    ]
    development_rows = receipt.get("development_gate_monte_carlo") or []
    if not development_rows or any(
        row.get("hard_promotion_gates") != expected_hard_gates
        or row.get("required_diagnostic_or_deployment_reports") != expected_reports
        for row in development_rows
    ):
        raise ValueError("Metadata-calibration development gate receipt drifted.")
    return {
        "schema": receipt["schema"],
        "power_receipt_sha256": _sha256_file(receipt_path),
        "receipt_payload_sha256": claimed_payload_hash,
        "outcome_execution_authorized": False,
    }


def validate_metadata_calibration_baseline_ledger(
    path: str | Path | None = None,
) -> dict[str, object]:
    ledger_path = (
        Path(path)
        if path is not None
        else _repository_root() / METADATA_CALIBRATION_BASELINE_LEDGER
    )
    ledger = load_config(ledger_path, strict_env=False)
    if {
        "schema": ledger.get("schema"),
        "candidate_id": ledger.get("candidate_id"),
        "status": ledger.get("status"),
        "human_execution_authorized": ledger.get("human_execution_authorized"),
    } != {
        "schema": "cfeg.metadata-calibration-baseline-ledger.v1",
        "candidate_id": "metadata-calibration-efficiency-v1",
        "status": (
            "mandatory_methods_and_label_free_query_adapters_frozen_"
            "signed_phase_authority_required"
        ),
        "human_execution_authorized": True,
    }:
        raise ValueError("Metadata-calibration baseline ledger header drifted.")
    common = ledger.get("common_protocol") or {}
    if (
        common.get("dataset_revision") != "wearable_v3"
        or common.get("support") != "exact_same_complete_nested_blocks_per_participant_interface"
        or common.get("query") != "exact_blocks06_through10"
        or common.get("external_context_access") != "forbidden_for_every_baseline"
        or common.get("participant_or_query_batch_transductive_statistics") != "forbidden"
        or common.get("deterministic_baseline_execution") != "once_with_seed_none"
        or common.get("score_interpretation")
        != "uncalibrated_argmax_only_no_cross_method_confidence_comparison"
    ):
        raise ValueError("Baseline common protocol or information rights drifted.")
    mandatory = {item.get("id"): item for item in ledger.get("mandatory_before_held_reveal") or []}
    if set(mandatory) != {
        "FBCCA_k0",
        "supervised_template_correlation",
        "target_filterbank_eTRCA",
        "SAME3_filterbank_eTRCA_one_shot_component",
        "proposed_common_A_Q",
        "chiang2021_LST_filterbank_eTRCA",
    }:
        raise ValueError("The mandatory baseline subset drifted.")
    if (
        mandatory["FBCCA_k0"].get("budgets") != [0]
        or mandatory["FBCCA_k0"].get("adapter_id") != "strict_FBCCA"
        or mandatory["FBCCA_k0"].get("producer_kind") != "baseline"
        or mandatory["supervised_template_correlation"].get("budgets") != [1, 3, 5]
        or mandatory["supervised_template_correlation"].get("adapter_id")
        != "target_template_correlation"
        or mandatory["supervised_template_correlation"].get("producer_kind")
        != "baseline"
        or mandatory["supervised_template_correlation"].get("implementation")
        != (
            "src/cfeg/baselines/metadata_calibration_adapter.py::"
            "GovernedCalibrationBaselineAdapter"
        )
        or mandatory["target_filterbank_eTRCA"].get("budgets") != [3, 5]
        or mandatory["target_filterbank_eTRCA"].get("adapter_id")
        != "target_filterbank_eTRCA"
        or mandatory["target_filterbank_eTRCA"].get("producer_kind") != "baseline"
        or mandatory["SAME3_filterbank_eTRCA_one_shot_component"].get("budgets")
        != [1]
        or mandatory["SAME3_filterbank_eTRCA_one_shot_component"].get("adapter_id")
        != "same3_filterbank_eTRCA"
        or mandatory["SAME3_filterbank_eTRCA_one_shot_component"].get("producer_kind")
        != "baseline"
        or mandatory["SAME3_filterbank_eTRCA_one_shot_component"].get(
            "interpretation"
        )
        != "cleanroom_one_shot_augmentation_component_not_full_OS_SSVEP_reproduction"
        or mandatory["chiang2021_LST_filterbank_eTRCA"].get("budgets") != [1, 3, 5]
        or mandatory["chiang2021_LST_filterbank_eTRCA"].get("adapter_id")
        != "chiang2021_LST_filterbank_eTRCA"
        or mandatory["chiang2021_LST_filterbank_eTRCA"].get("producer_kind")
        != "baseline"
        or mandatory["chiang2021_LST_filterbank_eTRCA"].get("core_implementation")
        != (
            "src/cfeg/baselines/calibration.py::"
            "predict_lst_filterbank_ensemble_trca"
        )
        or mandatory["chiang2021_LST_filterbank_eTRCA"].get("k1_interpretation")
        != "protocol_adapted_not_paper_reproduced"
        or mandatory["proposed_common_A_Q"].get("budgets") != [0, 1, 3, 5]
        or mandatory["proposed_common_A_Q"].get("adapter_id") is not None
        or mandatory["proposed_common_A_Q"].get("producer_kind") != "candidate"
        or mandatory["proposed_common_A_Q"].get("source_labeled_eeg")
        != "exact_nonouter_26_in_development_or_exact_39_in_held"
        or mandatory["chiang2021_LST_filterbank_eTRCA"].get("source_labeled_eeg")
        != "exact_nonouter_26_in_development_or_exact_39_in_held"
        or any(item.get("external_context_M") is not False for item in mandatory.values())
    ):
        raise ValueError("Baseline budget or external-context rights drifted.")
    expected_filterbanks = {
        "FBCCA_k0": "configs/baselines/fbcca_chen2015_m3.yaml",
        "target_filterbank_eTRCA": (
            "configs/baselines/fb_etrac_chiang2021_common_protocol.yaml"
        ),
        "SAME3_filterbank_eTRCA_one_shot_component": (
            "configs/baselines/fb_etrac_chiang2021_common_protocol.yaml"
        ),
        "chiang2021_LST_filterbank_eTRCA": (
            "configs/baselines/fb_etrac_chiang2021_common_protocol.yaml"
        ),
    }
    for baseline_id, config_path in expected_filterbanks.items():
        item = mandatory[baseline_id]
        if (
            item.get("filterbank_config") != config_path
            or item.get("filterbank_config_sha256")
            != _sha256_file(_repository_root() / config_path)
        ):
            raise ValueError("A filter-bank baseline is not bound to the strict config.")
    direct = {
        item.get("id"): item for item in ledger.get("direct_prior_art_comparators") or []
    }
    if (
        not {
            "chiang2021_LST_filterbank_eTRCA",
            "TDCA",
            "SAME3_filterbank_eTRCA_one_shot_component",
            "CSDuDoFN_OS_SSVEP",
            "SSVEP_DAN",
        }.issubset(direct)
        or direct["chiang2021_LST_filterbank_eTRCA"].get("priority")
        != "selected_mandatory"
        or direct["TDCA"].get("budgets") != [3, 5]
        or direct["SAME3_filterbank_eTRCA_one_shot_component"].get("priority")
        != "selected_mandatory_component_comparator"
        or direct["SAME3_filterbank_eTRCA_one_shot_component"].get("budgets") != [1]
        or direct["CSDuDoFN_OS_SSVEP"].get("budgets") != [1]
        or direct["CSDuDoFN_OS_SSVEP"].get("aliases")
        != ["CSDuDoFN_arXiv_v1", "OS_SSVEP_peer_reviewed_successor"]
        or direct["CSDuDoFN_OS_SSVEP"].get("original_target_access")
        != "exactly_one_labeled_trial_per_class"
        or direct["CSDuDoFN_OS_SSVEP"].get("priority")
        != "audited_not_run"
        or direct["CSDuDoFN_OS_SSVEP"].get("resolution") != "NOT_RUN"
        or direct["CSDuDoFN_OS_SSVEP"].get("status")
        != "audited_not_run_before_outcome_freeze"
        or direct["CSDuDoFN_OS_SSVEP"].get("official_repository_license")
        != "absent"
        or not str(direct["CSDuDoFN_OS_SSVEP"].get("not_run_reason", "")).strip()
        or direct["SSVEP_DAN"].get("budgets") != [3, 5]
        or direct["SSVEP_DAN"].get("original_minimum_labeled_trials_per_class") != 2
        or direct["SSVEP_DAN"].get("k1_status")
        != "incompatible_with_original_method_do_not_claim_or_run_as_paper_reproduction"
        or direct["SSVEP_DAN"].get("priority")
        != "audited_not_run"
        or direct["SSVEP_DAN"].get("resolution") != "NOT_RUN"
        or direct["SSVEP_DAN"].get("status")
        != "audited_not_run_before_outcome_freeze"
        or direct["SSVEP_DAN"].get("official_repository_license") != "absent"
        or not str(direct["SSVEP_DAN"].get("not_run_reason", "")).strip()
    ):
        raise ValueError("The direct prior-art comparator ledger is incomplete.")
    rules = ledger.get("ranking_rules") or {}
    if (
        rules.get("paper_reported_number_may_enter_numeric_table") is not False
        or rules.get("different_target_label_budget")
        != "separate_stratum_never_rank_as_equal_budget"
        or rules.get("target_only_filterbank_eTRCA_k1") != "forbidden"
        or rules.get("LST_filterbank_eTRCA_k1")
        != "protocol_adapted_not_paper_reproduced"
        or rules.get("deterministic_method_seed_replication") != "forbidden"
        or rules.get("unavailable_direct_comparator")
        != "report_as_not_run_with_reason_never_silently_omit"
        or rules.get("metadata_increment_primary_comparator")
        != "proposed_common_A_Q_same_checkpoint_exact_M_off"
    ):
        raise ValueError("Baseline ranking rules drifted.")
    if ledger.get("remaining_implementation") != []:
        raise ValueError("Baseline implementation ledger drifted.")
    return {
        "schema": ledger["schema"],
        "candidate_id": ledger["candidate_id"],
        "baseline_ledger_sha256": _sha256_file(ledger_path),
        "outcome_execution_authorized": True,
    }


def _validate_power_cells(
    rows: Sequence[object],
    *,
    expected: Mapping[str, Mapping[str, object]],
    kind: str,
    master_seed: int,
    draws: int,
    expected_mc_se: float,
    scenario_key_with_family: bool = False,
) -> None:
    observed: dict[str, Mapping[str, object]] = {}
    for value in rows:
        if not isinstance(value, Mapping):
            raise TypeError("Metadata-calibration power receipt contains a non-object cell.")
        scenario = str(value.get("scenario"))
        key = f"{scenario}:{value.get('family')}" if scenario_key_with_family else scenario
        if key in observed:
            raise ValueError("Metadata-calibration power receipt contains a duplicate scenario.")
        observed[key] = value
    if set(observed) != set(expected):
        raise ValueError("Metadata-calibration power receipt scenario grid drifted.")
    for key, expected_fields in expected.items():
        row = observed[key]
        if row.get("kind") != kind or any(
            row.get(field) != expected_value
            for field, expected_value in expected_fields.items()
        ):
            raise ValueError("Metadata-calibration power receipt cell DGP drifted.")
        cell = {"kind": kind, **expected_fields}
        if (
            row.get("draws") != draws
            or row.get("cell_seed") != _power_cell_seed(master_seed, cell)
            or not np.isclose(
                float(row.get("maximum_nominal_mc_standard_error", np.nan)),
                expected_mc_se,
                rtol=0.0,
                atol=1e-15,
            )
        ):
            raise ValueError("Metadata-calibration power receipt cell binding drifted.")


def _power_cell_seed(master_seed: int, cell: Mapping[str, object]) -> int:
    payload = json.dumps(cell, sort_keys=True, separators=(",", ":"), allow_nan=False)
    digest = hashlib.sha256(f"{master_seed}:{payload}".encode()).digest()
    return int.from_bytes(digest[:8], "big", signed=False)


def validate_metadata_calibration_source_recipe(
    path: str | Path | None = None,
    *,
    expected_source_subjects: Sequence[str] | None = None,
) -> dict[str, object]:
    """Validate nested source fitting without granting human-data execution."""

    recipe_path = (
        Path(path) if path is not None else _repository_root() / METADATA_CALIBRATION_SOURCE_RECIPE
    )
    recipe = load_config(recipe_path, strict_env=False)
    if {
        "schema": recipe.get("schema"),
        "candidate_id": recipe.get("candidate_id"),
        "status": recipe.get("status"),
        "human_execution_authorized": recipe.get("human_execution_authorized"),
        "plan": recipe.get("plan"),
    } != {
        "schema": "cfeg.metadata-calibration-source-recipe.v1",
        "candidate_id": "metadata-calibration-efficiency-v1",
        "status": "exact_recipe_frozen_signed_phase_authority_required",
        "human_execution_authorized": True,
        "plan": METADATA_CALIBRATION_PLAN,
    }:
        raise ValueError("Metadata-calibration source recipe header drifted.")

    runtime = recipe.get("runtime") or {}
    if (
        runtime.get("device") != "cuda"
        or runtime.get("hardware") != "single_rtx_4090"
        or runtime.get("source_training_seeds") != [42, 43, 44]
        or runtime.get("deterministic_algorithms") is not True
    ):
        raise ValueError("Source recipe runtime or seed contract drifted.")
    data = recipe.get("data") or {}
    source_subjects = tuple(str(value) for value in data.get("source_development_subjects") or [])
    if expected_source_subjects is not None and source_subjects != tuple(expected_source_subjects):
        raise ValueError("Source recipe subjects differ from the plan allocation.")
    if (
        len(source_subjects) != 39
        or len(set(source_subjects)) != 39
        or data.get("permanently_retired_subjects") != ["sub001", "sub002", "sub003"]
        or data.get("dataset_revision") != "wearable_v3"
        or data.get("metadata_contract_version") != "0.4-dev"
        or data.get("outer_fold_seed") != 20260904
    ):
        raise ValueError("Source recipe data contract drifted.")
    folds = data.get("folds") or {}
    if {int(key) for key in folds} != {0, 1, 2}:
        raise ValueError("Source recipe must contain exact outer folds 0,1,2.")
    outer_union: set[str] = set()
    source_set = set(source_subjects)
    for fold_index in range(3):
        fold = folds.get(fold_index) or folds.get(str(fold_index)) or {}
        outer = {str(value) for value in fold.get("outer_evaluation") or []}
        inner_validation = {str(value) for value in fold.get("inner_validation") or []}
        inner_fit = {str(value) for value in fold.get("inner_fit") or []}
        if (
            len(outer) != 13
            or len(inner_validation) != 6
            or len(inner_fit) != 20
            or outer & inner_validation
            or outer & inner_fit
            or inner_validation & inner_fit
            or outer | inner_validation | inner_fit != source_set
        ):
            raise ValueError(f"Source recipe fold {fold_index} is not an exact 13/6/20 split.")
        outer_union.update(outer)
    if outer_union != source_set:
        raise ValueError("Source recipe outer folds do not partition all 39 participants.")

    episode = recipe.get("episode") or {}
    if (
        episode.get("unit") != "participant_by_electrode_interface"
        or episode.get("conditions") != ["dry", "wet"]
        or episode.get("budgets") != [0, 1, 3, 5]
        or episode.get("query_blocks") != list(METADATA_CALIBRATION_QUERY_BLOCKS)
        or episode.get("all_four_budgets_per_participant_condition_per_epoch") is not True
        or episode.get("support_query_overlap") != "forbidden"
        or episode.get("target_participant_gradient_update") != "forbidden"
    ):
        raise ValueError("Source recipe episode contract drifted.")
    model = recipe.get("model") or {}
    prior = model.get("prior") or {}
    if (
        model.get("n_classes") != 12
        or model.get("c_max") != 64
        or model.get("t_len") != 400
        or model.get("target_sfreq") != 200.0
        or model.get("embedding_dim") != 256
        or (model.get("backbone") or {}).get("name") != "spectral_transformer"
        or prior.get("q_feature_dim") != 56
        or prior.get("metadata_feature_dim") != 18
        or prior.get("metadata_precision_ratio") != [0.8, 1.25]
    ):
        raise ValueError("Source recipe model contract drifted.")
    stage1 = recipe.get("stage_1_common_Q") or {}
    stage2 = recipe.get("stage_2_bounded_M") or {}
    if (
        stage1.get("metadata_mode") != "exact_off"
        or stage1.get("optimizer") != "AdamW"
        or stage1.get("learning_rate") != 0.0003
        or stage1.get("maximum_epochs") != 40
        or stage1.get("inner_selection", {}).get("tie_2") != "earliest_epoch"
        or stage2.get("common_path_mode") != "frozen_and_eval"
        or stage2.get("trainable_prefixes") != ["prior.metadata_precision_encoder"]
        or stage2.get("optimizer") != "AdamW"
        or stage2.get("learning_rate") != 0.001
        or stage2.get("maximum_epochs") != 25
        or stage2.get("wrong_or_shuffled_context_used_as_training_target") is not False
    ):
        raise ValueError("Source recipe staged optimization contract drifted.")
    target = recipe.get("target_calibration") or {}
    checkpoint = recipe.get("checkpoint") or {}
    if (
        target.get("backbone_gradient_update") != "forbidden"
        or target.get("prior_parameter_gradient_update") != "forbidden"
        or target.get("update")
        != "exact_closed_form_diagonal_gaussian_sufficient_statistics"
        or checkpoint.get("full_composite_state_required") is not True
        or checkpoint.get("save_trainable_only") is not False
    ):
        raise ValueError("Target update or composite checkpoint recipe drifted.")
    return {
        "schema": recipe["schema"],
        "candidate_id": recipe["candidate_id"],
        "source_recipe_sha256": _sha256_file(recipe_path),
        "outcome_execution_authorized": False,
    }


def reject_metadata_calibration_generic_target_access(
    data_config: Mapping[str, object], *, action: str
) -> None:
    """Keep wearable_v3 outcomes out of generic data-bearing entrypoints."""

    if not data_config_targets_wearable_v3(data_config):
        return
    binding = validate_metadata_calibration_plan()
    from cfeg.governance import GovernanceError

    raise GovernanceError(
        "wearable_v3 human outcome access through "
        f"{action!r} is blocked by {binding['decision_id']}. Authorization is confined to "
        "the signed, one-shot metadata-calibration lifecycle; generic training, prediction, "
        "calibration-label access, and metrics remain forbidden."
    )


def build_complete_block_calibration_partition(
    manifest: pd.DataFrame,
    indices: Sequence[int] | np.ndarray,
    *,
    budgets: Sequence[int] = METADATA_CALIBRATION_BUDGETS,
    support_blocks: Sequence[str] = METADATA_CALIBRATION_SUPPORT_BLOCKS,
    query_blocks: Sequence[str] = METADATA_CALIBRATION_QUERY_BLOCKS,
    expected_classes: int = 12,
) -> CompleteBlockCalibrationPartition:
    """Partition one label-free opaque-token participant/interface view.

    Production callers must pass only the five structural columns below.  Raw
    sample identities, labels, HDF5 row numbers and stimulus/order proxies are
    deliberately outside this API.  Class completeness is checked later for
    support labels and, after the private seal, for the query-label sidecar.
    """

    required_columns = {
        "row_token",
        "dataset_id",
        "subject_id",
        "electrode_type",
        "run_id",
    }
    missing = required_columns - set(manifest)
    extra = set(manifest) - required_columns
    if missing or extra:
        raise ValueError(
            "Label-free calibration view must use its exact allowlist; "
            f"missing={sorted(missing)}, extra={sorted(extra)}."
        )

    global_indices = np.asarray(indices, dtype=int)
    if global_indices.ndim != 1 or len(global_indices) == 0:
        raise ValueError("Calibration indices must be one nonempty one-dimensional collection.")
    if len(np.unique(global_indices)) != len(global_indices):
        raise ValueError("Calibration indices contain duplicates.")
    if global_indices.min() < 0 or global_indices.max() >= len(manifest):
        raise ValueError("Calibration indices fall outside the manifest.")

    expected_budgets = tuple(int(value) for value in budgets)
    if expected_budgets != METADATA_CALIBRATION_BUDGETS:
        raise ValueError("Complete-block primary budgets must be exactly (0, 1, 3, 5).")
    if expected_classes != 12:
        raise ValueError("The sealed complete-block contract requires exactly 12 rows per block.")
    support_blocks = tuple(str(value) for value in support_blocks)
    query_blocks = tuple(str(value) for value in query_blocks)
    if support_blocks != METADATA_CALIBRATION_SUPPORT_BLOCKS:
        raise ValueError("Complete-block primary support must be block01 through block05.")
    if query_blocks != METADATA_CALIBRATION_QUERY_BLOCKS:
        raise ValueError("Complete-block primary query must be block06 through block10.")
    if set(support_blocks) & set(query_blocks):
        raise ValueError("Calibration support and query blocks overlap.")

    selected = manifest.iloc[global_indices].copy()
    selected["_global_index"] = global_indices
    group_values: dict[str, str] = {}
    for column in ("dataset_id", "subject_id", "electrode_type"):
        values = sorted(selected[column].astype(str).unique().tolist())
        if len(values) != 1:
            raise ValueError(
                "Complete-block calibration requires one participant-condition group; "
                f"{column} has {values}."
            )
        group_values[column] = values[0]
    if group_values["dataset_id"] != "wearable":
        raise ValueError("The current complete-block primary contract is wearable-only.")

    expected_runs = set(support_blocks) | set(query_blocks)
    observed_runs = set(selected["run_id"].astype(str))
    if observed_runs != expected_runs:
        raise ValueError(
            "Participant-condition runs differ from the exact ten-block contract: "
            f"missing={sorted(expected_runs - observed_runs)}, "
            f"extra={sorted(observed_runs - expected_runs)}."
        )
    selected["row_token"] = selected["row_token"].astype(str)
    if selected["row_token"].duplicated().any() or not selected["row_token"].str.fullmatch(
        r"[qs]_[0-9a-f]{64}"
    ).all():
        raise ValueError("Calibration row tokens must be unique opaque q_/s_ HMAC tokens.")

    by_block: dict[str, np.ndarray] = {}
    for run_id in (*support_blocks, *query_blocks):
        block = selected.loc[selected["run_id"].astype(str).eq(run_id)].copy()
        if len(block) != expected_classes:
            raise ValueError(f"{run_id} does not contain exactly {expected_classes} sealed rows.")
        expected_prefix = "s_" if run_id in support_blocks else "q_"
        if not block["row_token"].str.startswith(expected_prefix).all():
            raise ValueError("Support rows must use s_ tokens and query rows q_ tokens.")
        block = block.sort_values("row_token", kind="stable")
        by_block[run_id] = block["_global_index"].to_numpy(dtype=int)

    support_by_budget: dict[int, np.ndarray] = {}
    for budget in expected_budgets:
        chosen = support_blocks[:budget]
        support_by_budget[budget] = (
            np.concatenate([by_block[run_id] for run_id in chosen])
            if chosen
            else np.asarray([], dtype=int)
        )
        if len(support_by_budget[budget]) != budget * expected_classes:
            raise RuntimeError("Complete-block support accounting failed.")

    for smaller, larger in pairwise(expected_budgets):
        if not set(support_by_budget[smaller]).issubset(set(support_by_budget[larger])):
            raise RuntimeError("Calibration supports are not nested.")

    query_indices = np.concatenate([by_block[run_id] for run_id in query_blocks])
    if any(set(indices_).intersection(query_indices) for indices_ in support_by_budget.values()):
        raise RuntimeError("Calibration support leaked into the immutable query.")
    query_ids = manifest.iloc[query_indices]["row_token"].astype(str).tolist()

    return CompleteBlockCalibrationPartition(
        support_by_budget=support_by_budget,
        query_indices=query_indices,
        query_identity_sha256=canonical_identity_sha256(query_ids),
        dataset_id=group_values["dataset_id"],
        subject_id=group_values["subject_id"],
        electrode_type=group_values["electrode_type"],
    )


def audit_complete_block_cohort_partition(
    manifest: pd.DataFrame,
    *,
    role: str,
) -> CompleteBlockCohortBinding:
    """Privileged historical raw-manifest audit; never a production runner API.

    This preserves the already-recorded raw sample-ID bundle digest.  It reads
    label-bearing asset columns and therefore must only be used by the input
    sealer/auditor, before producing a label-free opaque-token view.  Training,
    inference and final reduction must never call it.
    """

    plan_path = _repository_root() / METADATA_CALIBRATION_PLAN
    validate_metadata_calibration_plan(plan_path)
    plan = load_config(plan_path, strict_env=False)
    allocation = (plan.get("data_contract") or {}).get("proposed_unopened_allocation") or {}
    role_fields = {
        "source_development": (
            "source_development_subject_ids",
            "source_query_identity_bundle_sha256",
        ),
        "held_participant_evaluation": (
            "held_participant_evaluation_subject_ids",
            "held_query_identity_bundle_sha256",
        ),
    }
    if role not in role_fields:
        raise ValueError(
            "Complete-block cohort role must be source_development or held_participant_evaluation."
        )
    subject_field, digest_field = role_fields[role]
    expected_subjects = tuple(sorted(str(value) for value in allocation[subject_field]))
    expected_query_identity_bundle_sha256 = str(allocation[digest_field])
    required = {"dataset_id", "subject_id", "electrode_type"}
    missing = required - set(manifest)
    if missing:
        raise ValueError(f"Cohort manifest is missing columns: {sorted(missing)}.")

    positions = np.arange(len(manifest), dtype=int)
    selected_mask = manifest["dataset_id"].astype(str).eq("wearable") & manifest[
        "subject_id"
    ].astype(str).isin(expected_subjects)
    selected = manifest.loc[selected_mask, ["subject_id", "electrode_type"]].copy()
    observed_subjects = tuple(sorted(selected["subject_id"].astype(str).unique().tolist()))
    if observed_subjects != expected_subjects:
        raise ValueError(
            "Complete-block cohort membership drifted: "
            f"expected={expected_subjects}, observed={observed_subjects}."
        )
    selected["_position"] = positions[selected_mask.to_numpy()]

    records: list[tuple[str, str, str]] = []
    for subject_id in expected_subjects:
        conditions = tuple(
            sorted(
                selected.loc[selected["subject_id"].astype(str).eq(subject_id), "electrode_type"]
                .astype(str)
                .unique()
                .tolist()
            )
        )
        if conditions != ("dry", "wet"):
            raise ValueError(
                f"Subject {subject_id} must have exact dry/wet conditions, found {conditions}."
            )
        for condition in conditions:
            group_positions = selected.loc[
                selected["subject_id"].astype(str).eq(subject_id)
                & selected["electrode_type"].astype(str).eq(condition),
                "_position",
            ].to_numpy(dtype=int)
            query_identity = _audit_raw_complete_block_group(
                manifest,
                group_positions,
            )
            records.append((subject_id, condition, query_identity))

    query_bundle = _query_identity_bundle_sha256(records)
    if query_bundle != expected_query_identity_bundle_sha256:
        raise ValueError(
            "Complete-block cohort query identity drifted: "
            f"expected={expected_query_identity_bundle_sha256}, observed={query_bundle}."
        )
    return CompleteBlockCohortBinding(
        role=str(role),
        subject_ids=expected_subjects,
        participant_condition_groups=len(records),
        query_count=len(records) * len(METADATA_CALIBRATION_QUERY_BLOCKS) * 12,
        query_identity_bundle_sha256=query_bundle,
    )


def _audit_raw_complete_block_group(
    manifest: pd.DataFrame,
    indices: Sequence[int] | np.ndarray,
) -> str:
    """Validate the historical raw manifest and return its query-ID digest."""

    required = {
        "sample_id",
        "dataset_id",
        "subject_id",
        "electrode_type",
        "run_id",
        "label",
    }
    missing = required - set(manifest)
    if missing:
        raise ValueError(f"Historical calibration audit is missing columns: {sorted(missing)}.")
    positions = np.asarray(indices, dtype=int)
    selected = manifest.iloc[positions].loc[:, sorted(required)].copy()
    if selected["sample_id"].astype(str).duplicated().any():
        raise ValueError("Historical calibration sample IDs must be unique.")
    for column in ("dataset_id", "subject_id", "electrode_type"):
        if selected[column].astype(str).nunique() != 1:
            raise ValueError("Historical complete-block audit mixed participant conditions.")
    if set(selected["run_id"].astype(str)) != {
        *METADATA_CALIBRATION_SUPPORT_BLOCKS,
        *METADATA_CALIBRATION_QUERY_BLOCKS,
    }:
        raise ValueError("Historical complete-block audit does not contain exact block01-block10.")
    labels = list(range(12))
    for run_id, block in selected.groupby(selected["run_id"].astype(str), sort=False):
        if len(block) != 12 or sorted(block["label"].astype(int).tolist()) != labels:
            raise ValueError(f"Historical {run_id} is not one exact class-complete block.")
    query_ids = selected.loc[
        selected["run_id"].astype(str).isin(METADATA_CALIBRATION_QUERY_BLOCKS),
        "sample_id",
    ].astype(str)
    return canonical_identity_sha256(query_ids)


def _validate_allocation_contract(allocation: Mapping[str, object]) -> None:
    eligible = np.asarray([f"sub{index:03d}" for index in range(4, 103)])
    rng = np.random.default_rng(42)
    rng.shuffle(eligible)
    expected_source = sorted(str(value) for value in eligible[:39])
    expected_held = sorted(str(value) for value in eligible[39:])
    if (
        allocation.get("allocation_decision_id") != "DEC-20260830-002"
        or allocation.get("allocation_seed") != 42
        or allocation.get("allocation_algorithm")
        != "numpy_default_rng_shuffle_sorted_subject_ids_train_first_v1"
        or allocation.get("source_development_subjects") != 39
        or allocation.get("held_participant_evaluation_subjects") != 60
        or allocation.get("source_development_subject_ids") != expected_source
        or allocation.get("held_participant_evaluation_subject_ids") != expected_held
        or allocation.get("reuse_existing_outcome_blind_allocation") is not True
        or allocation.get("status") != "frozen_authorized_source_then_conditional_held"
    ):
        raise ValueError("The exact outcome-blind 39/60 allocation contract drifted.")
    legacy_payload = {
        "allocation_decision_id": allocation.get("allocation_decision_id"),
        "allocation_seed": allocation.get("allocation_seed"),
        "allocation_algorithm": allocation.get("allocation_algorithm"),
        "confirmatory_training_subject_ids": allocation.get("source_development_subject_ids"),
        "primary_included_subject_ids": allocation.get("held_participant_evaluation_subject_ids"),
    }
    if allocation.get("allocation_sha256") != _sha256_json(legacy_payload):
        raise ValueError("The outcome-blind 39/60 allocation digest drifted.")
    if (
        allocation.get("source_query_identity_bundle_sha256")
        != "224d8239aef5d0f9dd04c388f8ed8d2f961fc150c1cf72741f5ab22b756f80de"
        or allocation.get("held_query_identity_bundle_sha256")
        != "feffdcc005f2f4818bb496c61b683dc7afd735166c2f50cf7892262ef55d2a16"
    ):
        raise ValueError("The source or held complete-block query identity binding drifted.")


def data_config_targets_wearable_v3(data_config: Mapping[str, object]) -> bool:
    """Recognize wearable assets before a generic loader can expose their rows.

    The asset receipt is the canonical identity source.  The manifest probe is a
    defense-in-depth fallback for renamed or stale copies whose receipt was lost
    or became unreadable; it reads only ``dataset_id`` and never opens signals.h5.
    """

    expected = data_config.get("expected_revisions") or {}
    if isinstance(expected, Mapping) and str(expected.get("wearable", "")) == "wearable_v3":
        return True
    processed_dirs = data_config.get("processed_dirs") or []
    if isinstance(processed_dirs, (str, Path)):
        processed_dirs = [processed_dirs]
    for raw_path in processed_dirs:
        path = Path(os.path.expandvars(str(raw_path))).expanduser()
        if path.name == "wearable_v3":
            return True
        info_path = path / "asset_info.json"
        if info_path.is_file():
            try:
                info = json.loads(info_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                info = {}
            if (
                info.get("dataset_id") == "wearable"
                or info.get("dataset_revision") == "wearable_v3"
            ):
                return True
        if _manifest_declares_wearable(path):
            return True
    return False


def _manifest_declares_wearable(path: Path) -> bool:
    parquet_path = path / "manifest.parquet"
    if parquet_path.is_file():
        try:
            dataset_ids = pd.read_parquet(parquet_path, columns=["dataset_id"])["dataset_id"]
        except (ImportError, KeyError, OSError, ValueError):  # pragma: no cover - jsonl fallback
            dataset_ids = None
        if dataset_ids is not None and dataset_ids.astype(str).eq("wearable").any():
            return True

    jsonl_path = path / "manifest.jsonl"
    if not jsonl_path.is_file():
        return False
    try:
        with jsonl_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                row = json.loads(line)
                if isinstance(row, Mapping) and row.get("dataset_id") == "wearable":
                    return True
    except (OSError, json.JSONDecodeError):
        return False
    return False


def _query_identity_bundle_sha256(records: Sequence[tuple[str, str, str]]) -> str:
    payload = "\n".join("|".join(record) for record in sorted(records)).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _repository_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_json(value: object) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()
