from __future__ import annotations

import hashlib
import itertools
import json
import math
import os
import re
import stat
import uuid
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd
import yaml
from scipy.stats import beta as beta_distribution
from scipy.stats import t as student_t

V2_CANDIDATE_ID = "metadata-calibration-efficiency-v2"
V2_PLAN_SCHEMA = "cfeg.metadata-calibration-efficiency-plan.v2"
V2_ALLOCATION_SCHEMA = "cfeg.metadata-calibration-efficiency-v2.external-allocation.v1"
V2_PREPARATION_RECEIPT_SCHEMA = "cfeg.metadata-calibration-v2-preparation-completion.v1"
V2_SELECTION_RECEIPT_SCHEMA = "cfeg.metadata-calibration-v2-development-selection-completion.v1"
V2_GATE_RECEIPT_SCHEMA = "cfeg.metadata-calibration-v2-independent-aq-gate-completion.v1"
V2_SCORE_REQUEST_SCHEMA = "cfeg.metadata-calibration-v2-score-request.v1"
V2_SCORE_RECEIPT_SCHEMA = "cfeg.metadata-calibration-v2-score-completion.v1"

_REPOSITORY = Path(__file__).resolve().parents[3]
DEFAULT_V2_PLAN_PATH = _REPOSITORY / "configs/analysis/metadata_calibration_efficiency_v2.yaml"
DEFAULT_V2_ALLOCATION_PATH = (
    _REPOSITORY / "configs/governance/metadata_calibration_v2_external_allocation.json"
)

V2_EXTERNAL_DATASETS = ("beta_v1", "dong2023_v1")
V2_EXTERNAL_COHORTS = ("development", "independent_gate")
V2_BUDGETS = (0, 1, 3)
V2_EAUC_WEIGHTS = {0: 1.0 / 6.0, 1: 1.0 / 2.0, 3: 1.0 / 3.0}
V2_N_CLASSES = 40
V2_FBCCA_CONFIG_SHA256 = "494ca99cd54c7254e085e5b58a4f81acd501d3122bb8860339d1d803f5e125c5"
V2_DEVELOPMENT_SENSITIVITY_SEED = 20260906
V2_GATE_SENSITIVITY_SEED = 20260907
V2_DEFAULT_RESAMPLES = 20_000

_ALLOCATION_EXPECTATIONS = {
    "beta_v1": {
        "seed": 20260906,
        "n_subjects": 70,
        "excluded_exposed": ("sub001", "sub016"),
        "development_count": 23,
        "independent_gate_count": 45,
    },
    "dong2023_v1": {
        "seed": 20260907,
        "n_subjects": 59,
        "excluded_exposed": ("sub001",),
        "development_count": 19,
        "independent_gate_count": 39,
    },
}

_PLAN_CRITERION_ORDER = (
    "reject_any_candidate_failing_datasetwise_k1_mean_ge_0",
    "reject_any_candidate_failing_pooled_k1_lower_CI_gt_minus_0p025",
    "maximize_equal_dataset_mean_eAUC_A_Q_minus_A0",
    "minimize_lambda_max",
    "prefer_score_prototype_shrinkage",
    "minimize_prototype_prior_pseudocount",
)
_GATE_REQUIREMENT_NAMES = (
    "pooled_one_sided_95_percent_lower_bound_gt_0",
    "pooled_observed_mean_ge_0p01",
    "each_dataset_k1_mean_ge_0",
    "each_dataset_each_budget_lower_bound_gt_minus_0p025",
    "participant_delta_below_minus_0p10_upper_rate_lt_0p10",
)
_CANDIDATE_SPEC_FIELDS = {
    "candidate_key",
    "operator",
    "lambda_max",
    "prototype_prior_pseudocount",
}
_DEVELOPMENT_SUMMARY_FIELDS = {
    *_CANDIDATE_SPEC_FIELDS,
    "dataset_k1_means",
    "dataset_eauc_means",
    "equal_dataset_mean_eauc_delta",
    "pooled_k1_inference",
    "datasetwise_k1_mean_nonnegative",
    "pooled_k1_lower_bound_above_minus_0p025",
    "eligible",
    "rejection_reasons",
}

_PREDICTION_COLUMNS = {
    "candidate_id",
    "plan_sha256",
    "allocation_sha256",
    "asset_manifest_sha256",
    "asset_signals_sha256",
    "dataset_id",
    "cohort",
    "candidate_key",
    "operator",
    "lambda_max",
    "prototype_prior_pseudocount",
    "subject_id",
    "role",
    "budget",
    "query_token",
    "label",
    "prediction",
}
_PARTICIPANT_BA_COLUMNS = {
    *_PREDICTION_COLUMNS - {"query_token", "label", "prediction"},
    "balanced_accuracy",
}
_PARTICIPANT_DELTA_COLUMNS = {
    "candidate_id",
    "plan_sha256",
    "allocation_sha256",
    "asset_manifest_sha256",
    "asset_signals_sha256",
    "dataset_id",
    "cohort",
    "candidate_key",
    "operator",
    "lambda_max",
    "prototype_prior_pseudocount",
    "subject_id",
    "a0_k0",
    "aq_k0",
    "delta_k0",
    "a0_k1",
    "aq_k1",
    "delta_k1",
    "a0_k3",
    "aq_k3",
    "delta_k3",
    "a0_eauc",
    "aq_eauc",
    "eauc_delta",
}

_PREPARATION_RECEIPT_FIELDS = {
    "schema",
    "candidate_id",
    "phase",
    "status",
    "plan_path",
    "plan_sha256",
    "plan_content_sha256",
    "allocation_path",
    "allocation_file_sha256",
    "allocation_sha256",
    "development_subjects",
    "independent_gate_subjects",
    "outcomes_loaded",
    "v1_source39_accessed",
    "wearable_held60_accessed",
    "held_access_authorized",
    "completion_receipt_sha256",
}
_SELECTION_RECEIPT_FIELDS = {
    "schema",
    "candidate_id",
    "phase",
    "status",
    "plan_sha256",
    "plan_content_sha256",
    "allocation_sha256",
    "allocation_file_sha256",
    "candidate_grid_sha256",
    "participant_delta_sha256",
    "criterion_order",
    "candidate_summaries",
    "selected_candidate",
    "k1_global_enabled",
    "candidate_and_parameters_immutable",
    "n_candidates",
    "n_participants",
    "independent_gate_authorized",
    "held_access_authorized",
    "completion_receipt_sha256",
}
_GATE_RECEIPT_FIELDS = {
    "schema",
    "candidate_id",
    "phase",
    "status",
    "plan_sha256",
    "plan_content_sha256",
    "allocation_sha256",
    "allocation_file_sha256",
    "development_selection_receipt_sha256",
    "participant_delta_sha256",
    "selected_candidate",
    "pooled_eauc_inference",
    "dataset_k1_means",
    "dataset_budget_lower_bounds",
    "severe_harm",
    "requirements",
    "all_requirements_passed",
    "n_participants",
    "held_access_authorized",
    "next_required",
    "completion_receipt_sha256",
}
_SCORE_REQUEST_FIELDS = {
    "schema",
    "candidate_id",
    "plan_sha256",
    "allocation_sha256",
    "asset_manifest_sha256",
    "asset_signals_sha256",
    "dataset_id",
    "cohort",
    "candidate",
    "variant",
    "subject_id",
    "budget",
    "query_tokens",
    "query_fbcca_scores",
    "support_fbcca_scores",
    "support_labels",
    "base_probabilities",
    "gate_enabled",
    "template_query_class_scores",
    "template_query_class_probabilities",
    "template_score_provenance",
    "template_query_subbands",
    "template_support_subbands",
    "template_subband_weights",
    "query_interfaces",
    "support_interfaces",
    "query_impedance_kohm",
    "support_impedance_kohm",
    "relative_context_pairing_sha256",
}
_SCORE_RECEIPT_FIELDS = {
    "schema",
    "candidate_id",
    "phase",
    "status",
    "plan_sha256",
    "plan_content_sha256",
    "allocation_sha256",
    "allocation_file_sha256",
    "asset_manifest_sha256",
    "asset_signals_sha256",
    "dataset_id",
    "cohort",
    "candidate",
    "variant",
    "subject_id",
    "budget",
    "query_tokens",
    "base_predictions",
    "predictions",
    "base_probabilities",
    "fused_probabilities",
    "lambdas",
    "base_scores_sha256",
    "base_probabilities_sha256",
    "fused_probabilities_sha256",
    "affinities_sha256",
    "gate_enabled",
    "exact_fallback",
    "fallback_reason",
    "operator_schema",
    "relative_context_pairing_sha256",
    "comparable_context_pair_count",
    "template_score_provenance",
    "final_budget",
    "support_depth",
    "prequential_fold_provenance",
    "support_labels_loaded",
    "query_outcomes_loaded",
    "v1_source39_accessed",
    "wearable_held60_accessed",
    "held_access_authorized",
    "completion_receipt_sha256",
}

_SHA256_PATTERN = r"[0-9a-f]{64}"


@dataclass(frozen=True)
class V2CandidateSpec:
    candidate_key: str
    operator: Literal["score_prototype_shrinkage", "filterbank_target_template_residual"]
    lambda_max: float
    prototype_prior_pseudocount: float | None

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class V2ContractBinding:
    plan_path: Path
    allocation_path: Path
    plan_sha256: str
    plan_content_sha256: str
    allocation_file_sha256: str
    allocation_sha256: str
    plan: Mapping[str, Any]
    allocation_receipt: Mapping[str, Any]

    def subject_ids(
        self,
        dataset_id: str,
        cohort: Literal["development", "independent_gate"],
    ) -> tuple[str, ...]:
        if dataset_id not in V2_EXTERNAL_DATASETS or cohort not in V2_EXTERNAL_COHORTS:
            raise ValueError("V2 external allocation requested an unknown dataset or cohort.")
        values = self.allocation_receipt["allocation"][dataset_id][cohort]
        return tuple(str(value) for value in values)

    def asset_binding(self, dataset_id: str) -> dict[str, str]:
        if dataset_id not in V2_EXTERNAL_DATASETS:
            raise ValueError("V2 external asset binding requested an unknown dataset.")
        external = self.plan["external_allocation"][dataset_id]
        return {
            "asset_manifest_sha256": str(external["manifest_sha256"]),
            "asset_signals_sha256": str(external["signals_sha256"]),
        }


def validate_v2_plan_and_allocation(
    plan_path: str | Path = DEFAULT_V2_PLAN_PATH,
    allocation_path: str | Path | None = None,
) -> V2ContractBinding:
    """Validate the frozen V2 analysis contract without opening an EEG asset."""

    resolved_plan = _regular_file_path(plan_path, name="V2 plan")
    plan = yaml.safe_load(resolved_plan.read_text(encoding="utf-8")) or {}
    if not isinstance(plan, Mapping):
        raise TypeError("V2 plan must decode to one mapping.")
    if {
        "schema": plan.get("schema"),
        "candidate_id": plan.get("candidate_id"),
        "decision_id": plan.get("decision_id"),
        "status": plan.get("status"),
        "method_revision": plan.get("method_revision"),
    } != {
        "schema": V2_PLAN_SCHEMA,
        "candidate_id": V2_CANDIDATE_ID,
        "decision_id": "DEC-20260906-001",
        "status": "method_and_source_gate_frozen_held_forbidden_until_all_gates_pass",
        "method_revision": "r5_pre_outcome_prequential_prefix_execution",
    }:
        raise ValueError("V2 plan header differs from the frozen contract.")

    support_operators = plan.get("support_operators") or {}
    anchor = plan.get("anchor") or {}
    normalization = plan.get("score_normalization") or {}
    score_prototype = support_operators.get("score_prototype_shrinkage") or {}
    template_residual = support_operators.get("filterbank_target_template_residual") or {}
    fusion = support_operators.get("fusion") or {}
    relative_context = plan.get("relative_context") or {}
    target_gate = plan.get("target_local_gate") or {}
    if {
        "implementation": anchor.get("implementation"),
        "filterbank_config": anchor.get("filterbank_config"),
        "filterbank_config_sha256": anchor.get("filterbank_config_sha256"),
        "score_schema": anchor.get("score_schema"),
        "candidate_codebook_rule": anchor.get("candidate_codebook_rule"),
    } != {
        "implementation": "src/cfeg/baselines/fbcca.py::predict_fbcca",
        "filterbank_config": "configs/baselines/fbcca_chen2015_m3.yaml",
        "filterbank_config_sha256": V2_FBCCA_CONFIG_SHA256,
        "score_schema": "strict_fbcca_bound_filterbank_raw_scores_v1",
        "candidate_codebook_rule": "all_classes_same_order_for_every_row",
    }:
        raise ValueError("V2 r4 strict-FBCCA anchor binding drifted.")
    filterbank_path = _regular_file_path(
        _REPOSITORY / str(anchor["filterbank_config"]), name="V2 anchor filterbank config"
    )
    if _sha256_file(filterbank_path) != V2_FBCCA_CONFIG_SHA256:
        raise ValueError("V2 r4 anchor filterbank file differs from its frozen SHA-256.")
    if {
        "formula": normalization.get("formula"),
        "overflow_safe_row_scale": normalization.get("overflow_safe_row_scale"),
        "epsilon": normalization.get("epsilon"),
        "probability": normalization.get("probability"),
        "base_argmax": normalization.get("base_argmax"),
    } != {
        "formula": "overflow_safe_row_scale_then_center_then_divide_by_root_mean_square_with_epsilon",
        "overflow_safe_row_scale": (
            "divide_by_row_max_absolute_value_before_equivalent_center_RMS_standardization"
        ),
        "epsilon": 1.0e-12,
        "probability": "softmax_normalized_score_temperature_one",
        "base_argmax": "raw_FBCCA_argmax",
    }:
        raise ValueError("V2 r4 score-normalization formula drifted.")
    if score_prototype != {
        "base_distribution": "normalized_FBCCA_probability",
        "ideal_prototype_smoothing": 0.05,
        "ideal_prototype_formula": (
            "smoothing_divided_by_C_plus_one_minus_smoothing_on_true_class"
        ),
        "prototype_prior_pseudocount_grid": [1.0, 4.0, 16.0],
        "prototype_formula": (
            "prior_times_ideal_plus_sum_affinity_times_support_probability_divided_by "
            "prior_plus_sum_affinity_within_each_class"
        ),
        "similarity": "negative_Jensen_Shannon_divergence",
        "query_class_score": (
            "negative_Jensen_Shannon_between_query_probability_and_class_prototype"
        ),
        "support_probability": (
            "apply_frozen_score_normalization_and_softmax_to_query_class_scores"
        ),
    }:
        raise ValueError("V2 r4 score-prototype P1 formula drifted.")
    if template_residual != {
        "statistic": "weighted_signed_squared_multichannel_template_correlation",
        "source_EEG_access": False,
        "filterbank": "exact_anchor_filterbank",
        "filterbank_config_sha256": V2_FBCCA_CONFIG_SHA256,
        "subband_weights": [
            1.25,
            0.6704482076268572,
            0.5032785618838642,
            0.42677669529663687,
            0.38374806099528436,
            0.35649051737437876,
            0.33782687899303776,
        ],
        "class_template_formula": "affinity_weighted_mean_of_filtered_support_within_class",
        "correlation": "Pearson_after_flattening_channel_and_time_within_each_subband",
        "query_class_score": "sum_subband_weight_times_sign_rho_times_rho_squared",
        "support_probability": (
            "apply_frozen_score_normalization_and_softmax_to_query_class_scores"
        ),
        "governed_precomputed_path": (
            "requires_schema_preprocess_filterbank_query_partition_and_support_partition_SHA256"
        ),
    }:
        raise ValueError("V2 r4 template-residual P2 formula or filterbank binding drifted.")
    if {
        "query_quality_factor": fusion.get("query_quality_factor"),
        "A_Q_lambda": fusion.get("A_Q_lambda"),
        "A_QM_lambda": fusion.get("A_QM_lambda"),
        "fusion_formula": fusion.get("formula"),
        "budget_factor": fusion.get("budget_factor"),
        "exact_null_candidate": fusion.get("exact_null_candidate"),
        "direct_return_when_disabled": fusion.get("direct_return_without_arithmetic_when_disabled"),
        "l1_bound": fusion.get("l1_probability_displacement_bound"),
        "model_effect": relative_context.get("model_effect"),
        "support_mass_factor": relative_context.get("support_mass_factor"),
        "may_change_anchor_or_query_precision": relative_context.get(
            "may_change_anchor_or_query_precision"
        ),
        "per_query_class_prototype": relative_context.get("per_query_class_prototype"),
        "integrity_binding": relative_context.get("integrity_binding"),
        "A_Q_context_policy": relative_context.get("A_Q_context_policy"),
        "A_QM_context_policy": relative_context.get("A_QM_context_policy"),
        "target_gate_authorization": target_gate.get("authorization"),
        "missing_gate_authorization": target_gate.get("missing_authorization"),
        "prequential_probability": target_gate.get("prequential_probability"),
        "prequential_prefix_receipt": target_gate.get("prequential_prefix_receipt"),
        "final_query_operator_budgets": target_gate.get("final_query_operator_budgets"),
        "prequential_operator": target_gate.get("prequential_operator"),
    } != {
        "query_quality_factor": "normalized_entropy_of_p_FBCCA",
        "A_Q_lambda": "lambda_max_times_budget_factor_times_query_quality_factor",
        "A_QM_lambda": "A_Q_lambda_times_equal_class_mean_query_support_affinity",
        "fusion_formula": ("p_equals_one_minus_lambda_times_p_FBCCA_plus_lambda_times_p_support"),
        "budget_factor": "k_divided_by_k_plus_one",
        "exact_null_candidate": True,
        "direct_return_when_disabled": True,
        "l1_bound": "two_times_lambda",
        "model_effect": "support_weight_and_aggregate_effective_support_mass_only",
        "support_mass_factor": "equal_class_mean_query_support_affinity",
        "may_change_anchor_or_query_precision": False,
        "per_query_class_prototype": True,
        "integrity_binding": "relative_context_pairing_sha256",
        "A_Q_context_policy": "reject_every_context_or_pairing_input",
        "A_QM_context_policy": "require_pairing_SHA256_and_report_comparable_pair_count",
        "target_gate_authorization": (
            "explicit_required_for_every_enabled_k_greater_than_zero_call"
        ),
        "missing_gate_authorization": "exact_anchor_before_support_access",
        "prequential_probability": "final_fused_candidate_probability",
        "prequential_prefix_receipt": (
            "fit_blocks_must_equal_1_through_evaluation_block_minus_one"
        ),
        "final_query_operator_budgets": [0, 1, 3, 5],
        "prequential_operator": {
            "dedicated_API_required": True,
            "final_budget": [3, 5],
            "evaluation_blocks": [2, 3, 4, 5],
            "support_depth_formula": "evaluation_block_minus_one",
            "allowed_support_depths": [1, 2, 3, 4],
            "may_score_only_the_declared_evaluation_block": True,
            "unconditional_candidate_application_for_gate_estimation": True,
            "authorization": "typed_PrequentialFoldProvenance",
            "output_must_echo_fold_provenance_and_support_depth": True,
            "final_query_access": "forbidden",
        },
    }:
        raise ValueError("V2 r5 fusion, gate, provenance, or context policy drifted.")

    rights = plan.get("information_rights") or {}
    forbidden = set(rights.get("forbidden") or [])
    if not {
        "v1_source39_outcomes_for_hyperparameter_or_candidate_selection",
        "held60_input_or_outcome_access_before_all_gates_and_new_signed_authorization",
        "target_query_labels_or_outcomes_during_fit_selection_or_gating",
    }.issubset(forbidden):
        raise ValueError("V2 plan does not retain its outcome-access prohibitions.")
    calibration = plan.get("calibration_contract") or {}
    if (
        calibration.get("budgets") != list(V2_BUDGETS) + [5]
        or calibration.get("external_four_block_support")
        != {1: ["block01"], 3: ["block01", "block02", "block03"]}
        or calibration.get("external_four_block_query") != ["block04"]
        or calibration.get("support_query_disjoint") is not True
    ):
        raise ValueError("V2 external support/query calibration contract drifted.")
    selection = plan.get("development_selection") or {}
    independent = plan.get("independent_A_Q_gate") or {}
    if (
        selection.get("candidates_total") != 12
        or tuple(selection.get("criterion_order") or ()) != _PLAN_CRITERION_ORDER
        or tuple(independent.get("requirements") or ()) != _GATE_REQUIREMENT_NAMES
    ):
        raise ValueError("V2 selection or independent-gate contract drifted.")
    held = plan.get("held_boundary") or {}
    stopping = plan.get("stopping_rules") or {}
    if (
        held.get("any_failure") != "held_forbidden"
        or stopping.get("synthetic_or_independent_A_Q_gate_failure")
        != "terminate_without_held_access"
        or "new_signed_owner_authorization" not in set(held.get("prerequisites") or [])
    ):
        raise ValueError("V2 held-access boundary is not fail-closed.")

    external = plan.get("external_allocation") or {}
    declared_relative = external.get("receipt")
    if declared_relative != "configs/governance/metadata_calibration_v2_external_allocation.json":
        raise ValueError("V2 plan points to an unexpected allocation receipt.")
    resolved_allocation = _regular_file_path(
        DEFAULT_V2_ALLOCATION_PATH if allocation_path is None else allocation_path,
        name="V2 allocation receipt",
    )
    allocation = json.loads(resolved_allocation.read_text(encoding="utf-8"))
    if not isinstance(allocation, Mapping):
        raise TypeError("V2 allocation receipt must decode to one mapping.")
    allocation_sha256 = validate_v2_external_allocation_receipt(
        allocation,
        expected_sha256=str(external.get("allocation_sha256") or ""),
    )
    if external.get("schema") != V2_ALLOCATION_SCHEMA or external.get("method") != allocation.get(
        "method"
    ):
        raise ValueError("V2 plan and allocation receipt disagree on schema or method.")
    if external.get("seeds") != allocation.get("seeds"):
        raise ValueError("V2 plan and allocation receipt disagree on seeds.")
    for dataset_id in V2_EXTERNAL_DATASETS:
        declared = external.get(dataset_id) or {}
        observed = allocation["allocation"][dataset_id]
        if any(
            declared.get(field) != observed.get(field)
            for field in ("excluded_exposed", "development", "independent_gate")
        ):
            raise ValueError(f"V2 plan and allocation receipt disagree for {dataset_id}.")
        for hash_field in ("manifest_sha256", "signals_sha256"):
            if not _is_sha256(declared.get(hash_field)):
                raise ValueError(f"V2 {dataset_id} {hash_field} is not a SHA-256 binding.")

    return V2ContractBinding(
        plan_path=resolved_plan,
        allocation_path=resolved_allocation,
        plan_sha256=_sha256_file(resolved_plan),
        plan_content_sha256=_canonical_json_sha256(plan),
        allocation_file_sha256=_sha256_file(resolved_allocation),
        allocation_sha256=allocation_sha256,
        plan=dict(plan),
        allocation_receipt=dict(allocation),
    )


def validate_v2_external_allocation_receipt(
    receipt: Mapping[str, Any],
    *,
    expected_sha256: str,
) -> str:
    """Replay the frozen RNG allocation and return its canonical content hash."""

    if set(receipt) != {"schema", "method", "seeds", "allocation"}:
        raise ValueError("V2 external allocation receipt has a noncanonical schema.")
    if (
        receipt.get("schema") != V2_ALLOCATION_SCHEMA
        or receipt.get("method")
        != "dataset_stratified_numpy_default_rng_permutation_sorted_ids_dev_first"
    ):
        raise ValueError("V2 external allocation receipt header drifted.")
    seeds = receipt.get("seeds")
    allocation = receipt.get("allocation")
    if not isinstance(seeds, Mapping) or set(seeds) != set(V2_EXTERNAL_DATASETS):
        raise ValueError("V2 external allocation seeds have a wrong exact schema.")
    if not isinstance(allocation, Mapping) or set(allocation) != set(V2_EXTERNAL_DATASETS):
        raise ValueError("V2 external allocation datasets have a wrong exact schema.")

    for dataset_id, expected in _ALLOCATION_EXPECTATIONS.items():
        if type(seeds.get(dataset_id)) is not int or seeds[dataset_id] != expected["seed"]:
            raise ValueError(f"V2 {dataset_id} allocation seed drifted.")
        observed = allocation.get(dataset_id)
        if not isinstance(observed, Mapping) or set(observed) != {
            "excluded_exposed",
            "development",
            "independent_gate",
        }:
            raise ValueError(f"V2 {dataset_id} allocation row has a wrong exact schema.")
        excluded = _strict_subject_list(
            observed["excluded_exposed"], name=f"{dataset_id} excluded_exposed"
        )
        development = _strict_subject_list(
            observed["development"], name=f"{dataset_id} development"
        )
        gate = _strict_subject_list(
            observed["independent_gate"], name=f"{dataset_id} independent_gate"
        )
        if excluded != expected["excluded_exposed"]:
            raise ValueError(f"V2 {dataset_id} exposed-subject exclusion drifted.")
        if (
            len(development) != expected["development_count"]
            or len(gate) != expected["independent_gate_count"]
        ):
            raise ValueError(f"V2 {dataset_id} allocation count drifted.")
        universe = tuple(f"sub{index:03d}" for index in range(1, expected["n_subjects"] + 1))
        eligible = tuple(value for value in universe if value not in set(excluded))
        if set(development) & set(gate) or set(development) | set(gate) != set(eligible):
            raise ValueError(f"V2 {dataset_id} development/gate allocation is not a partition.")
        shuffled = np.asarray(eligible, dtype=str)
        np.random.default_rng(expected["seed"]).shuffle(shuffled)
        expected_development = tuple(sorted(map(str, shuffled[: len(development)])))
        expected_gate = tuple(sorted(map(str, shuffled[len(development) :])))
        if development != expected_development or gate != expected_gate:
            raise ValueError(f"V2 {dataset_id} allocation differs from the frozen RNG replay.")

    observed_sha256 = _canonical_json_sha256(receipt)
    if not _is_sha256(expected_sha256) or observed_sha256 != expected_sha256:
        raise ValueError("V2 external allocation canonical SHA-256 drifted.")
    return observed_sha256


def v2_candidate_grid(contract: V2ContractBinding) -> tuple[V2CandidateSpec, ...]:
    """Build the exact 9 score-prototype plus 3 template-residual candidates."""

    support = contract.plan.get("support_operators") or {}
    score = support.get("score_prototype_shrinkage") or {}
    fusion = support.get("fusion") or {}
    pseudocounts = tuple(
        float(value) for value in score.get("prototype_prior_pseudocount_grid") or ()
    )
    lambdas = tuple(float(value) for value in fusion.get("lambda_max_grid") or ())
    if pseudocounts != (1.0, 4.0, 16.0) or lambdas != (0.10, 0.20, 0.30):
        raise ValueError("V2 candidate grid differs from the frozen 3-by-3 plus 3 design.")
    candidates = [
        V2CandidateSpec(
            candidate_key=_candidate_key("score_prototype_shrinkage", value, pseudocount),
            operator="score_prototype_shrinkage",
            lambda_max=value,
            prototype_prior_pseudocount=pseudocount,
        )
        for value, pseudocount in itertools.product(lambdas, pseudocounts)
    ]
    candidates.extend(
        V2CandidateSpec(
            candidate_key=_candidate_key("filterbank_target_template_residual", value, None),
            operator="filterbank_target_template_residual",
            lambda_max=value,
            prototype_prior_pseudocount=None,
        )
        for value in lambdas
    )
    result = tuple(sorted(candidates, key=lambda value: value.candidate_key))
    if len(result) != 12 or len({item.candidate_key for item in result}) != 12:
        raise RuntimeError("V2 candidate-grid construction did not produce 12 unique candidates.")
    return result


def reduce_v2_participant_balanced_accuracy(
    predictions: pd.DataFrame,
    *,
    contract: V2ContractBinding,
    cohort: Literal["development", "independent_gate"],
    expected_candidate_keys: Sequence[str] | None = None,
    n_classes: int = V2_N_CLASSES,
) -> pd.DataFrame:
    """Reduce a complete external prediction grid to participant BA cells."""

    frame = _validate_prediction_grid(
        predictions,
        contract=contract,
        cohort=cohort,
        expected_candidate_keys=expected_candidate_keys,
        n_classes=n_classes,
    )
    group_columns = sorted(_PARTICIPANT_BA_COLUMNS - {"balanced_accuracy"})
    records: list[dict[str, object]] = []
    for key, group in frame.groupby(group_columns, sort=False, dropna=False):
        values = dict(zip(group_columns, key))
        label = group["label"].to_numpy(dtype=np.int64)
        prediction = group["prediction"].to_numpy(dtype=np.int64)
        recalls = [
            float(np.mean(prediction[label == value] == value)) for value in range(n_classes)
        ]
        records.append({**values, "balanced_accuracy": float(np.mean(recalls))})
    output = pd.DataFrame(records, columns=group_columns + ["balanced_accuracy"])
    return output.sort_values(
        ["candidate_key", "dataset_id", "subject_id", "role", "budget"],
        kind="mergesort",
    ).reset_index(drop=True)


def derive_v2_participant_eauc_deltas(
    participant_ba: pd.DataFrame,
    *,
    contract: V2ContractBinding,
    cohort: Literal["development", "independent_gate"],
    expected_candidate_keys: Sequence[str] | None = None,
) -> pd.DataFrame:
    """Derive k-specific and early-curve A_Q-minus-A0 effects per participant."""

    frame = _validate_participant_ba(
        participant_ba,
        contract=contract,
        cohort=cohort,
        expected_candidate_keys=expected_candidate_keys,
    )
    identity = sorted(_PARTICIPANT_BA_COLUMNS - {"role", "budget", "balanced_accuracy"})
    wide = frame.pivot(
        index=identity,
        columns=["role", "budget"],
        values="balanced_accuracy",
    )
    expected_columns = {(role, budget) for role in ("A0", "A_Q") for budget in V2_BUDGETS}
    if set(wide.columns) != expected_columns:
        raise ValueError("Participant BA table lost one required A0/A_Q budget cell.")
    result = wide.index.to_frame(index=False)
    for budget in V2_BUDGETS:
        result[f"a0_k{budget}"] = wide[("A0", budget)].to_numpy(dtype=float)
        result[f"aq_k{budget}"] = wide[("A_Q", budget)].to_numpy(dtype=float)
        result[f"delta_k{budget}"] = result[f"aq_k{budget}"] - result[f"a0_k{budget}"]
    if not np.array_equal(
        result["a0_k0"].to_numpy(dtype=float), result["aq_k0"].to_numpy(dtype=float)
    ):
        raise ValueError("V2 A_Q k=0 must be exactly equal to repeated strict FBCCA A0.")
    result["a0_eauc"] = sum(
        weight * result[f"a0_k{budget}"] for budget, weight in V2_EAUC_WEIGHTS.items()
    )
    result["aq_eauc"] = sum(
        weight * result[f"aq_k{budget}"] for budget, weight in V2_EAUC_WEIGHTS.items()
    )
    result["eauc_delta"] = result["aq_eauc"] - result["a0_eauc"]
    result = result.loc[:, sorted(_PARTICIPANT_DELTA_COLUMNS)]
    return result.sort_values(
        ["candidate_key", "dataset_id", "subject_id"], kind="mergesort"
    ).reset_index(drop=True)


def one_sided_t_lower_bound(
    differences: Sequence[float] | np.ndarray,
    *,
    alpha: float = 0.05,
) -> dict[str, float | int | str]:
    """Return the prespecified one-sided Student-t lower confidence bound."""

    values = _finite_vector(differences, name="paired differences")
    if not 0.0 < float(alpha) < 1.0:
        raise ValueError("alpha must lie strictly between zero and one.")
    mean = float(values.mean())
    sd = float(values.std(ddof=1))
    standard_error = float(sd / math.sqrt(len(values)))
    lower = (
        mean
        if standard_error <= np.finfo(float).eps
        else float(mean - student_t.ppf(1.0 - float(alpha), df=len(values) - 1) * standard_error)
    )
    return {
        "method": "one_sided_participant_paired_student_t_lower_bound",
        "n_participants": len(values),
        "mean": mean,
        "sd": sd,
        "standard_error": standard_error,
        "alpha": float(alpha),
        "confidence_level_one_sided": float(1.0 - alpha),
        "confidence_interval_low": lower,
    }


def one_sided_paired_sensitivity(
    differences: Sequence[float] | np.ndarray,
    *,
    null_margin: float,
    alpha: float = 0.05,
    seed: int,
    n_resamples: int = V2_DEFAULT_RESAMPLES,
) -> dict[str, object]:
    """Student-t primary summary with deterministic sign-flip/bootstrap sensitivity."""

    values = _finite_vector(differences, name="paired differences")
    if not np.isfinite(null_margin):
        raise ValueError("null_margin must be finite.")
    if type(seed) is not int or type(n_resamples) is not int or n_resamples < 1:
        raise ValueError("Sensitivity seed and resample count must be positive exact integers.")
    primary = one_sided_t_lower_bound(values, alpha=alpha)
    bootstrap_seed, sign_seed = np.random.SeedSequence(seed).spawn(2)
    bootstrap_rng = np.random.default_rng(bootstrap_seed)
    sign_rng = np.random.default_rng(sign_seed)

    bootstrap = bootstrap_rng.choice(
        values,
        size=(n_resamples, len(values)),
        replace=True,
    ).mean(axis=1)
    bootstrap_low = float(np.quantile(bootstrap, alpha, method="linear"))

    centered = values - float(null_margin)
    observed = float(centered.mean())
    exact = len(centered) <= 16
    if exact:
        signs = np.asarray(
            list(itertools.product((-1.0, 1.0), repeat=len(centered))),
            dtype=np.float64,
        )
    else:
        signs = sign_rng.choice((-1.0, 1.0), size=(n_resamples, len(centered)))
    null_means = (signs * centered).mean(axis=1)
    exceedances = int(np.count_nonzero(null_means >= observed))
    sign_flip_p = (
        float(exceedances / len(null_means))
        if exact
        else float((exceedances + 1) / (len(null_means) + 1))
    )
    return {
        "schema": "cfeg.metadata-calibration-v2-one-sided-paired-inference.v1",
        **primary,
        "null_margin": float(null_margin),
        "t_lower_bound_exceeds_null_margin": bool(
            float(primary["confidence_interval_low"]) > float(null_margin)
        ),
        "paired_sign_flip_p_value": sign_flip_p,
        "sign_flip_exact": exact,
        "sign_flip_draws": len(signs),
        "bootstrap_percentile_lower": bootstrap_low,
        "bootstrap_resamples": int(n_resamples),
        "sensitivity_seed": int(seed),
        "random_streams": "numpy_seedsequence_independent_bootstrap_and_sign_flip",
        "sensitivity_decision_role": "sensitivity_only",
    }


def select_v2_development_candidate(
    participant_deltas: pd.DataFrame,
    *,
    contract: V2ContractBinding,
    n_resamples: int = V2_DEFAULT_RESAMPLES,
) -> dict[str, Any]:
    """Apply the frozen six-level development selection rule to all 12 candidates."""

    candidates = v2_candidate_grid(contract)
    candidate_keys = tuple(item.candidate_key for item in candidates)
    frame = validate_v2_participant_deltas(
        participant_deltas,
        contract=contract,
        cohort="development",
        expected_candidate_keys=candidate_keys,
    )
    spec_by_key = {item.candidate_key: item for item in candidates}
    summaries: list[dict[str, Any]] = []
    for candidate_key in candidate_keys:
        candidate = frame.loc[frame["candidate_key"].eq(candidate_key)]
        dataset_k1_means = {
            dataset_id: float(
                candidate.loc[candidate["dataset_id"].eq(dataset_id), "delta_k1"].mean()
            )
            for dataset_id in V2_EXTERNAL_DATASETS
        }
        dataset_eauc_means = {
            dataset_id: float(
                candidate.loc[candidate["dataset_id"].eq(dataset_id), "eauc_delta"].mean()
            )
            for dataset_id in V2_EXTERNAL_DATASETS
        }
        pooled_k1 = one_sided_paired_sensitivity(
            candidate["delta_k1"].to_numpy(dtype=float),
            null_margin=-0.025,
            alpha=0.05,
            seed=_derived_seed(
                V2_DEVELOPMENT_SENSITIVITY_SEED,
                "development-k1",
                candidate_key,
            ),
            n_resamples=n_resamples,
        )
        datasetwise_pass = all(value >= 0.0 for value in dataset_k1_means.values())
        pooled_pass = float(pooled_k1["confidence_interval_low"]) > -0.025
        summary = {
            **spec_by_key[candidate_key].as_dict(),
            "dataset_k1_means": dataset_k1_means,
            "dataset_eauc_means": dataset_eauc_means,
            "equal_dataset_mean_eauc_delta": float(np.mean(list(dataset_eauc_means.values()))),
            "pooled_k1_inference": pooled_k1,
            "datasetwise_k1_mean_nonnegative": datasetwise_pass,
            "pooled_k1_lower_bound_above_minus_0p025": pooled_pass,
            "eligible": bool(datasetwise_pass and pooled_pass),
            "rejection_reasons": [
                reason
                for failed, reason in (
                    (
                        not datasetwise_pass,
                        "datasetwise_k1_mean_below_zero",
                    ),
                    (
                        not pooled_pass,
                        "pooled_k1_lower_bound_not_above_minus_0p025",
                    ),
                )
                if failed
            ],
        }
        summaries.append(summary)

    selected_summary = _select_summary_by_frozen_ties(summaries)

    selected_candidate = (
        None
        if selected_summary is None
        else {
            key: selected_summary[key]
            for key in (
                "candidate_key",
                "operator",
                "lambda_max",
                "prototype_prior_pseudocount",
            )
        }
    )
    participant_count = int(frame[["dataset_id", "subject_id"]].drop_duplicates().shape[0])
    payload: dict[str, Any] = {
        "schema": V2_SELECTION_RECEIPT_SCHEMA,
        "candidate_id": V2_CANDIDATE_ID,
        "phase": "development_selection",
        "status": (
            "selected_candidate_frozen"
            if selected_candidate is not None
            else "no_eligible_candidate_terminate"
        ),
        "plan_sha256": contract.plan_sha256,
        "plan_content_sha256": contract.plan_content_sha256,
        "allocation_sha256": contract.allocation_sha256,
        "allocation_file_sha256": contract.allocation_file_sha256,
        "candidate_grid_sha256": _canonical_json_sha256(
            {"candidates": [item.as_dict() for item in candidates]}
        ),
        "participant_delta_sha256": canonical_dataframe_sha256(frame),
        "criterion_order": list(_PLAN_CRITERION_ORDER),
        "candidate_summaries": summaries,
        "selected_candidate": selected_candidate,
        "k1_global_enabled": selected_candidate is not None,
        "candidate_and_parameters_immutable": selected_candidate is not None,
        "n_candidates": len(candidates),
        "n_participants": participant_count,
        "independent_gate_authorized": selected_candidate is not None,
        "held_access_authorized": False,
    }
    receipt = _with_completion_hash(payload)
    validate_v2_development_selection_receipt(receipt, contract=contract)
    return receipt


def evaluate_v2_independent_aq_gate(
    participant_deltas: pd.DataFrame,
    *,
    development_selection_receipt: Mapping[str, Any],
    contract: V2ContractBinding,
    n_resamples: int = V2_DEFAULT_RESAMPLES,
) -> dict[str, Any]:
    """Evaluate the independent A_Q gate while keeping held access forbidden."""

    selection = validate_v2_development_selection_receipt(
        development_selection_receipt,
        contract=contract,
    )
    selected = selection.get("selected_candidate")
    if selection.get("status") != "selected_candidate_frozen" or not isinstance(selected, Mapping):
        raise PermissionError("Independent gate requires one frozen development candidate.")
    candidate_key = str(selected["candidate_key"])
    frame = validate_v2_participant_deltas(
        participant_deltas,
        contract=contract,
        cohort="independent_gate",
        expected_candidate_keys=(candidate_key,),
    )
    observed_spec = _single_candidate_spec(frame)
    if observed_spec.as_dict() != dict(selected):
        raise ValueError("Independent-gate candidate differs from development selection.")

    pooled = one_sided_paired_sensitivity(
        frame["eauc_delta"].to_numpy(dtype=float),
        null_margin=0.0,
        alpha=0.05,
        seed=V2_GATE_SENSITIVITY_SEED,
        n_resamples=n_resamples,
    )
    dataset_k1_means = {
        dataset_id: float(frame.loc[frame["dataset_id"].eq(dataset_id), "delta_k1"].mean())
        for dataset_id in V2_EXTERNAL_DATASETS
    }
    dataset_budget_lower_bounds: dict[str, dict[str, dict[str, float | int | str]]] = {}
    for dataset_id in V2_EXTERNAL_DATASETS:
        dataset = frame.loc[frame["dataset_id"].eq(dataset_id)]
        dataset_budget_lower_bounds[dataset_id] = {
            str(budget): one_sided_t_lower_bound(
                dataset[f"delta_k{budget}"].to_numpy(dtype=float), alpha=0.05
            )
            for budget in V2_BUDGETS
        }

    severe = frame["eauc_delta"].to_numpy(dtype=float) < -0.10
    severe_harm = {
        "cutoff": -0.10,
        "count": int(np.count_nonzero(severe)),
        "n_participants": len(severe),
        "observed_rate": float(np.mean(severe)),
        "one_sided_95_percent_clopper_pearson_upper": _binomial_upper_bound(
            int(np.count_nonzero(severe)), len(severe), alpha=0.05
        ),
    }
    requirements = {
        "pooled_one_sided_95_percent_lower_bound_gt_0": bool(
            float(pooled["confidence_interval_low"]) > 0.0
        ),
        "pooled_observed_mean_ge_0p01": bool(float(pooled["mean"]) >= 0.01),
        "each_dataset_k1_mean_ge_0": all(value >= 0.0 for value in dataset_k1_means.values()),
        "each_dataset_each_budget_lower_bound_gt_minus_0p025": all(
            float(summary["confidence_interval_low"]) > -0.025
            for budgets in dataset_budget_lower_bounds.values()
            for summary in budgets.values()
        ),
        "participant_delta_below_minus_0p10_upper_rate_lt_0p10": bool(
            severe_harm["one_sided_95_percent_clopper_pearson_upper"] < 0.10
        ),
    }
    if tuple(requirements) != _GATE_REQUIREMENT_NAMES:
        raise RuntimeError("Independent-gate requirement order drifted.")
    passed = all(requirements.values())
    selection_hash = str(selection["completion_receipt_sha256"])
    payload: dict[str, Any] = {
        "schema": V2_GATE_RECEIPT_SCHEMA,
        "candidate_id": V2_CANDIDATE_ID,
        "phase": "independent_A_Q_gate",
        "status": "pass" if passed else "failed_terminate_before_held60",
        "plan_sha256": contract.plan_sha256,
        "plan_content_sha256": contract.plan_content_sha256,
        "allocation_sha256": contract.allocation_sha256,
        "allocation_file_sha256": contract.allocation_file_sha256,
        "development_selection_receipt_sha256": selection_hash,
        "participant_delta_sha256": canonical_dataframe_sha256(frame),
        "selected_candidate": dict(selected),
        "pooled_eauc_inference": pooled,
        "dataset_k1_means": dataset_k1_means,
        "dataset_budget_lower_bounds": dataset_budget_lower_bounds,
        "severe_harm": severe_harm,
        "requirements": requirements,
        "all_requirements_passed": passed,
        "n_participants": len(frame),
        # Passing this gate is necessary but deliberately not sufficient to open held60.
        "held_access_authorized": False,
        "next_required": (
            "choi_replication_clean_tag_power_receipt_and_new_signed_owner_authorization"
            if passed
            else "terminate_candidate_without_wearable_held60_access"
        ),
    }
    receipt = _with_completion_hash(payload)
    validate_v2_independent_gate_receipt(receipt, contract=contract)
    return receipt


def build_v2_preparation_receipt(contract: V2ContractBinding) -> dict[str, Any]:
    """Record allocation validation without claiming any outcome access."""

    payload = {
        "schema": V2_PREPARATION_RECEIPT_SCHEMA,
        "candidate_id": V2_CANDIDATE_ID,
        "phase": "prepare",
        "status": "frozen_plan_and_external_allocation_validated",
        "plan_path": str(contract.plan_path),
        "plan_sha256": contract.plan_sha256,
        "plan_content_sha256": contract.plan_content_sha256,
        "allocation_path": str(contract.allocation_path),
        "allocation_file_sha256": contract.allocation_file_sha256,
        "allocation_sha256": contract.allocation_sha256,
        "development_subjects": sum(
            len(contract.subject_ids(dataset_id, "development"))
            for dataset_id in V2_EXTERNAL_DATASETS
        ),
        "independent_gate_subjects": sum(
            len(contract.subject_ids(dataset_id, "independent_gate"))
            for dataset_id in V2_EXTERNAL_DATASETS
        ),
        "outcomes_loaded": False,
        "v1_source39_accessed": False,
        "wearable_held60_accessed": False,
        "held_access_authorized": False,
    }
    receipt = _with_completion_hash(payload)
    validate_v2_preparation_receipt(receipt, contract=contract)
    return receipt


def score_v2_external_request(
    request: Mapping[str, Any],
    *,
    contract: V2ContractBinding,
) -> dict[str, Any]:
    """Run one label-free-query external score request through the V2 operator.

    This entry point is intentionally limited to BETA/Dong development and
    independent-gate participants. Query outcomes are not part of its exact
    request schema. The operator import is lazy so plan/allocation preflight can
    run before the operator lane is integrated.
    """

    if set(request) != _SCORE_REQUEST_FIELDS:
        raise ValueError("V2 score request has a wrong exact schema or forbidden query fields.")
    if (
        request.get("schema") != V2_SCORE_REQUEST_SCHEMA
        or request.get("candidate_id") != V2_CANDIDATE_ID
        or request.get("plan_sha256") != contract.plan_sha256
        or request.get("allocation_sha256") != contract.allocation_sha256
    ):
        raise ValueError("V2 score request contract binding drifted.")
    dataset_id = str(request.get("dataset_id"))
    cohort = str(request.get("cohort"))
    subject_id = str(request.get("subject_id"))
    if dataset_id not in V2_EXTERNAL_DATASETS or cohort not in V2_EXTERNAL_COHORTS:
        raise PermissionError("V2 score requests are limited to external development/gate data.")
    if subject_id not in contract.subject_ids(dataset_id, cohort):  # type: ignore[arg-type]
        raise PermissionError("V2 score-request participant is outside the frozen allocation.")
    asset = contract.asset_binding(dataset_id)
    if any(request.get(key) != value for key, value in asset.items()):
        raise ValueError("V2 score request differs from the frozen external asset binding.")

    candidate_value = request.get("candidate")
    if not isinstance(candidate_value, Mapping) or set(candidate_value) != {
        "candidate_key",
        "operator",
        "lambda_max",
        "prototype_prior_pseudocount",
    }:
        raise ValueError("V2 score request must bind one exact candidate specification.")
    candidates = {item.candidate_key: item for item in v2_candidate_grid(contract)}
    candidate_key = str(candidate_value.get("candidate_key"))
    candidate = candidates.get(candidate_key)
    if candidate is None or candidate.as_dict() != dict(candidate_value):
        raise ValueError("V2 score request candidate is outside the frozen grid.")
    variant = request.get("variant")
    if variant not in {"A_Q", "A_QM"}:
        raise ValueError("V2 score request variant must be exactly A_Q or A_QM.")
    budget = request.get("budget")
    if type(budget) is not int or budget not in V2_BUDGETS:
        raise ValueError("V2 external score budget must be one of 0, 1, or 3.")
    gate_enabled = request.get("gate_enabled")
    if gate_enabled is not None and type(gate_enabled) is not bool:
        raise TypeError("V2 score gate_enabled must be an exact bool or null.")
    query_tokens = request.get("query_tokens")
    if (
        not isinstance(query_tokens, list)
        or not query_tokens
        or any(type(value) is not str or not value.strip() for value in query_tokens)
        or len(set(query_tokens)) != len(query_tokens)
    ):
        raise ValueError("V2 score query tokens must be nonempty unique strings.")
    query_scores = _optional_numeric_array(request["query_fbcca_scores"])
    if (
        query_scores is None
        or query_scores.ndim != 2
        or query_scores.shape != (len(query_tokens), V2_N_CLASSES)
    ):
        raise ValueError("V2 score query FBCCA matrix must have shape [query,40].")
    if not np.isfinite(query_scores).all():
        raise ValueError("V2 score query FBCCA matrix must be finite.")

    support_scores: np.ndarray | None = None
    support_labels: np.ndarray | None = None
    template_query_class_scores: np.ndarray | None = None
    template_query_class_probabilities: np.ndarray | None = None
    template_query_subbands: np.ndarray | None = None
    template_support_subbands: np.ndarray | None = None
    template_subband_weights: np.ndarray | None = None
    template_score_provenance = None
    query_interfaces = None
    support_interfaces = None
    query_impedance_kohm: np.ndarray | None = None
    support_impedance_kohm: np.ndarray | None = None
    pairing_sha256 = None
    support_fields = (
        "support_fbcca_scores",
        "support_labels",
        "template_query_class_scores",
        "template_query_class_probabilities",
        "template_score_provenance",
        "template_query_subbands",
        "template_support_subbands",
        "template_subband_weights",
        "query_interfaces",
        "support_interfaces",
        "query_impedance_kohm",
        "support_impedance_kohm",
        "relative_context_pairing_sha256",
    )
    support_authorized = budget > 0 and gate_enabled is True
    if not support_authorized and any(request[field] is not None for field in support_fields):
        raise PermissionError(
            "V2 k=0, missing-gate, and abstained requests must omit support payloads."
        )
    if support_authorized:
        support_scores = _optional_numeric_array(request["support_fbcca_scores"])
        raw_support_labels = _optional_numeric_array(request["support_labels"])
        if raw_support_labels is not None:
            if not np.equal(raw_support_labels, np.floor(raw_support_labels)).all():
                raise ValueError("V2 support labels must be finite exact integers.")
            support_labels = raw_support_labels.astype(np.int64)
        template_query_class_scores = _optional_numeric_array(
            request["template_query_class_scores"]
        )
        template_query_class_probabilities = _optional_numeric_array(
            request["template_query_class_probabilities"]
        )
        template_query_subbands = _optional_numeric_array(request["template_query_subbands"])
        template_support_subbands = _optional_numeric_array(request["template_support_subbands"])
        template_subband_weights = _optional_numeric_array(request["template_subband_weights"])
        if variant == "A_Q":
            if any(
                request[field] is not None
                for field in (
                    "query_interfaces",
                    "support_interfaces",
                    "query_impedance_kohm",
                    "support_impedance_kohm",
                    "relative_context_pairing_sha256",
                )
            ):
                raise ValueError("V2 A_Q requests reject every context and pairing field.")
        else:
            pairing_sha256 = request["relative_context_pairing_sha256"]
            if not _is_sha256(pairing_sha256):
                raise ValueError("Active V2 A_QM requires a lowercase SHA-256 pairing binding.")
            query_interfaces = request["query_interfaces"]
            support_interfaces = request["support_interfaces"]
            query_impedance_kohm = _optional_numeric_array(request["query_impedance_kohm"])
            support_impedance_kohm = _optional_numeric_array(request["support_impedance_kohm"])
    expected_support = budget * V2_N_CLASSES
    if support_authorized and (
        support_scores is None
        or support_scores.shape != (expected_support, V2_N_CLASSES)
        or support_labels is None
        or support_labels.shape != (expected_support,)
    ):
        raise ValueError("V2 k>0 score request lacks its exact class-complete support matrix.")
    elif support_authorized and (
        (support_labels < 0).any()
        or (support_labels >= V2_N_CLASSES).any()
        or not np.array_equal(
            np.bincount(support_labels, minlength=V2_N_CLASSES),
            np.full(V2_N_CLASSES, budget, dtype=np.int64),
        )
    ):
        raise ValueError("V2 k>0 support labels must contain exactly k rows per class.")

    base_probabilities = _optional_numeric_array(request["base_probabilities"])

    try:
        from cfeg.models.metadata_calibration_v2 import (
            TemplateScoreProvenance,
            V2OperatorConfig,
            apply_v2_safe_operator,
        )
    except ImportError as error:
        raise RuntimeError(
            "The frozen V2 operator implementation is not integrated; scoring is refused."
        ) from error
    raw_template_provenance = request["template_score_provenance"]
    if support_authorized and raw_template_provenance is not None:
        if not isinstance(raw_template_provenance, Mapping) or set(raw_template_provenance) != {
            "schema",
            "filterbank_sha256",
            "preprocessing_sha256",
            "query_partition_sha256",
            "support_partition_sha256",
        }:
            raise ValueError("V2 P2 template provenance has a wrong exact schema.")
        template_score_provenance = TemplateScoreProvenance(**raw_template_provenance)
        if template_score_provenance.filterbank_sha256 != V2_FBCCA_CONFIG_SHA256:
            raise ValueError("V2 P2 template provenance has the wrong frozen filterbank hash.")

    operator_plan = contract.plan["support_operators"]
    score_plan = operator_plan["score_prototype_shrinkage"]
    config = V2OperatorConfig(
        score_epsilon=float(contract.plan["score_normalization"]["epsilon"]),
        ideal_prototype_smoothing=float(score_plan["ideal_prototype_smoothing"]),
        prototype_prior_pseudocount=(
            4.0
            if candidate.prototype_prior_pseudocount is None
            else candidate.prototype_prior_pseudocount
        ),
        lambda_max=candidate.lambda_max,
        different_interface_affinity=float(
            contract.plan["relative_context"]["interface_affinity"]["different"]
        ),
        # Corrected frozen semantics: normalized entropy, so confident queries
        # receive less support weight and uncertain queries receive more.
        entropy_scaling=True,
    )
    output = apply_v2_safe_operator(
        query_scores,
        budget=budget,
        operator=candidate.operator,
        variant=variant,
        config=config,
        support_fbcca_scores=support_scores,
        support_labels=support_labels,
        base_probabilities=base_probabilities,
        gate_enabled=gate_enabled,
        template_query_class_scores=template_query_class_scores,
        template_query_class_probabilities=template_query_class_probabilities,
        template_score_provenance=template_score_provenance,
        template_query_subbands=template_query_subbands,
        template_support_subbands=template_support_subbands,
        template_subband_weights=template_subband_weights,
        query_interfaces=query_interfaces,
        support_interfaces=support_interfaces,
        query_impedance_kohm=query_impedance_kohm,
        support_impedance_kohm=support_impedance_kohm,
        relative_context_pairing_sha256=pairing_sha256,
    )
    base_scores = np.asarray(output.base_scores, dtype=np.float64)
    base = np.asarray(output.base_probabilities, dtype=np.float64)
    fused = np.asarray(output.fused_probabilities, dtype=np.float64)
    predictions = np.asarray(output.predictions, dtype=np.int64)
    lambdas = np.asarray(output.lambdas, dtype=np.float64)
    affinities = np.asarray(output.affinities, dtype=np.float64)
    if (
        base.shape != fused.shape
        or base.shape != query_scores.shape
        or predictions.shape != (len(query_tokens),)
    ):
        raise RuntimeError("V2 operator returned an invalid query-aligned output shape.")
    payload: dict[str, Any] = {
        "schema": V2_SCORE_RECEIPT_SCHEMA,
        "candidate_id": V2_CANDIDATE_ID,
        "phase": "external_score",
        "status": "complete_query_outcome_free",
        "plan_sha256": contract.plan_sha256,
        "plan_content_sha256": contract.plan_content_sha256,
        "allocation_sha256": contract.allocation_sha256,
        "allocation_file_sha256": contract.allocation_file_sha256,
        **asset,
        "dataset_id": dataset_id,
        "cohort": cohort,
        "candidate": candidate.as_dict(),
        "variant": str(output.variant),
        "subject_id": subject_id,
        "budget": budget,
        "query_tokens": list(query_tokens),
        "base_predictions": np.argmax(base_scores, axis=1).astype(int).tolist(),
        "predictions": predictions.astype(int).tolist(),
        "base_probabilities": base.tolist(),
        "fused_probabilities": fused.tolist(),
        "lambdas": lambdas.tolist(),
        "base_scores_sha256": _array_sha256(base_scores),
        "base_probabilities_sha256": _array_sha256(base),
        "fused_probabilities_sha256": _array_sha256(fused),
        "affinities_sha256": _array_sha256(affinities),
        "gate_enabled": output.gate_enabled,
        "exact_fallback": bool(output.exact_fallback),
        "fallback_reason": output.fallback_reason,
        "operator_schema": str(output.schema),
        "relative_context_pairing_sha256": output.relative_context_pairing_sha256,
        "comparable_context_pair_count": int(output.comparable_context_pair_count),
        "template_score_provenance": (
            None
            if output.template_score_provenance is None
            else asdict(output.template_score_provenance)
        ),
        "final_budget": int(output.final_budget),
        "support_depth": int(output.support_depth),
        "prequential_fold_provenance": (
            None
            if output.prequential_fold_provenance is None
            else asdict(output.prequential_fold_provenance)
        ),
        "support_labels_loaded": support_labels is not None,
        "query_outcomes_loaded": False,
        "v1_source39_accessed": False,
        "wearable_held60_accessed": False,
        "held_access_authorized": False,
    }
    receipt = _with_completion_hash(payload)
    validate_v2_score_receipt(receipt, contract=contract)
    return receipt


def validate_v2_preparation_receipt(
    receipt_or_path: Mapping[str, Any] | str | Path,
    *,
    contract: V2ContractBinding,
) -> dict[str, Any]:
    receipt = _load_receipt(receipt_or_path)
    _validate_completion_hash(receipt, expected_schema=V2_PREPARATION_RECEIPT_SCHEMA)
    if set(receipt) != _PREPARATION_RECEIPT_FIELDS:
        raise ValueError("V2 preparation completion receipt has a wrong exact schema.")
    if (
        receipt.get("candidate_id") != V2_CANDIDATE_ID
        or receipt.get("phase") != "prepare"
        or receipt.get("status") != "frozen_plan_and_external_allocation_validated"
        or receipt.get("plan_path") != str(contract.plan_path)
        or receipt.get("plan_sha256") != contract.plan_sha256
        or receipt.get("plan_content_sha256") != contract.plan_content_sha256
        or receipt.get("allocation_path") != str(contract.allocation_path)
        or receipt.get("allocation_file_sha256") != contract.allocation_file_sha256
        or receipt.get("allocation_sha256") != contract.allocation_sha256
        or receipt.get("development_subjects") != 42
        or receipt.get("independent_gate_subjects") != 84
        or receipt.get("outcomes_loaded") is not False
        or receipt.get("v1_source39_accessed") is not False
        or receipt.get("wearable_held60_accessed") is not False
        or receipt.get("held_access_authorized") is not False
    ):
        raise ValueError("V2 preparation completion receipt differs from the frozen contract.")
    return receipt


def validate_v2_score_receipt(
    receipt_or_path: Mapping[str, Any] | str | Path,
    *,
    contract: V2ContractBinding,
) -> dict[str, Any]:
    receipt = _load_receipt(receipt_or_path)
    _validate_completion_hash(receipt, expected_schema=V2_SCORE_RECEIPT_SCHEMA)
    if set(receipt) != _SCORE_RECEIPT_FIELDS:
        raise ValueError("V2 score completion receipt has a wrong exact schema.")
    dataset_id = str(receipt.get("dataset_id"))
    cohort = str(receipt.get("cohort"))
    subject_id = str(receipt.get("subject_id"))
    if (
        receipt.get("candidate_id") != V2_CANDIDATE_ID
        or receipt.get("phase") != "external_score"
        or receipt.get("status") != "complete_query_outcome_free"
        or receipt.get("plan_sha256") != contract.plan_sha256
        or receipt.get("plan_content_sha256") != contract.plan_content_sha256
        or receipt.get("allocation_sha256") != contract.allocation_sha256
        or receipt.get("allocation_file_sha256") != contract.allocation_file_sha256
        or dataset_id not in V2_EXTERNAL_DATASETS
        or cohort not in V2_EXTERNAL_COHORTS
        or subject_id not in contract.subject_ids(dataset_id, cohort)  # type: ignore[arg-type]
        or receipt.get("variant") not in {"A_Q", "A_QM"}
        or receipt.get("operator_schema") != "cfeg.metadata-calibration-v2-safe-operator.v2"
        or receipt.get("query_outcomes_loaded") is not False
        or receipt.get("v1_source39_accessed") is not False
        or receipt.get("wearable_held60_accessed") is not False
        or receipt.get("held_access_authorized") is not False
    ):
        raise ValueError("V2 score completion receipt binding drifted.")
    if any(receipt.get(key) != value for key, value in contract.asset_binding(dataset_id).items()):
        raise ValueError("V2 score completion receipt has a wrong external asset binding.")
    candidate = receipt.get("candidate")
    grid = {item.candidate_key: item.as_dict() for item in v2_candidate_grid(contract)}
    if not isinstance(candidate, Mapping) or grid.get(str(candidate.get("candidate_key"))) != dict(
        candidate
    ):
        raise ValueError("V2 score completion receipt has a non-frozen candidate.")
    tokens = receipt.get("query_tokens")
    predictions = receipt.get("predictions")
    base_predictions = receipt.get("base_predictions")
    base = np.asarray(receipt.get("base_probabilities"), dtype=float)
    fused = np.asarray(receipt.get("fused_probabilities"), dtype=float)
    lambdas = np.asarray(receipt.get("lambdas"), dtype=float)
    if (
        not isinstance(tokens, list)
        or not tokens
        or any(type(value) is not str or not value.strip() for value in tokens)
        or len(set(tokens)) != len(tokens)
        or not isinstance(predictions, list)
        or not isinstance(base_predictions, list)
        or len(predictions) != len(tokens)
        or len(base_predictions) != len(tokens)
        or any(type(value) is not int or not 0 <= value < V2_N_CLASSES for value in predictions)
        or any(
            type(value) is not int or not 0 <= value < V2_N_CLASSES for value in base_predictions
        )
        or base.shape != (len(tokens), V2_N_CLASSES)
        or fused.shape != base.shape
        or lambdas.shape != (len(tokens),)
        or not np.isfinite(base).all()
        or not np.isfinite(fused).all()
        or not np.isfinite(lambdas).all()
        or (base < 0.0).any()
        or (base > 1.0).any()
        or (fused < 0.0).any()
        or (fused > 1.0).any()
        or (lambdas < 0.0).any()
        or (lambdas > float(candidate["lambda_max"])).any()
        or not np.allclose(base.sum(axis=1), 1.0, rtol=0.0, atol=1e-12)
        or not np.allclose(fused.sum(axis=1), 1.0, rtol=0.0, atol=1e-12)
    ):
        raise ValueError("V2 score completion receipt contains invalid output arrays.")
    if not np.array_equal(
        np.asarray(base_predictions), np.argmax(base, axis=1)
    ) or not np.array_equal(np.asarray(predictions), np.argmax(fused, axis=1)):
        raise ValueError("V2 score completion predictions disagree with their probabilities.")
    budget = receipt.get("budget")
    gate_enabled = receipt.get("gate_enabled")
    exact_fallback = receipt.get("exact_fallback")
    if (
        type(budget) is not int
        or budget not in V2_BUDGETS
        or (gate_enabled is not None and type(gate_enabled) is not bool)
        or type(exact_fallback) is not bool
        or type(receipt.get("support_labels_loaded")) is not bool
        or type(receipt.get("comparable_context_pair_count")) is not int
        or receipt.get("comparable_context_pair_count") < 0
        or receipt.get("final_budget") != budget
        or receipt.get("support_depth") != budget
        or receipt.get("prequential_fold_provenance") is not None
        or (
            receipt.get("fallback_reason") is not None
            and type(receipt.get("fallback_reason")) is not str
        )
    ):
        raise ValueError("V2 score completion capability state is malformed.")
    expected_fallback_reason = (
        "k0_exact_strict_fbcca"
        if budget == 0
        else (
            "missing_gate_authorization"
            if gate_enabled is None
            else ("validation_gate_abstained" if gate_enabled is False else None)
        )
    )
    if expected_fallback_reason is not None:
        if (
            exact_fallback is not True
            or receipt.get("fallback_reason") != expected_fallback_reason
            or receipt.get("support_labels_loaded") is not False
            or not np.array_equal(base, fused)
            or not np.equal(lambdas, 0.0).all()
        ):
            raise ValueError("V2 score completion exact-fallback state is inconsistent.")
    elif exact_fallback or receipt.get("fallback_reason") is not None:
        raise ValueError("Active V2 score completion unexpectedly claimed exact fallback.")
    if receipt.get("support_labels_loaded") is not (budget > 0 and gate_enabled is True):
        raise ValueError("V2 score completion support-access state is inconsistent.")
    pairing = receipt.get("relative_context_pairing_sha256")
    comparable_count = receipt.get("comparable_context_pair_count")
    if receipt.get("variant") == "A_Q":
        if pairing is not None or comparable_count != 0:
            raise ValueError("V2 A_Q score completion must not claim context pairing.")
    elif expected_fallback_reason is not None:
        if pairing is not None or comparable_count != 0:
            raise ValueError("Fallback V2 A_QM must not inspect or report context pairing.")
    elif not _is_sha256(pairing):
        raise ValueError("Active V2 A_QM score completion lacks its pairing SHA-256.")
    template_provenance = receipt.get("template_score_provenance")
    if template_provenance is not None and (
        not isinstance(template_provenance, Mapping)
        or set(template_provenance)
        != {
            "schema",
            "filterbank_sha256",
            "preprocessing_sha256",
            "query_partition_sha256",
            "support_partition_sha256",
        }
        or template_provenance.get("schema")
        != "cfeg.metadata-calibration-v2-template-provenance.v1"
        or template_provenance.get("filterbank_sha256") != V2_FBCCA_CONFIG_SHA256
        or not all(
            _is_sha256(template_provenance.get(field))
            for field in (
                "preprocessing_sha256",
                "query_partition_sha256",
                "support_partition_sha256",
            )
        )
    ):
        raise ValueError("V2 score completion P2 provenance is invalid.")
    if candidate["operator"] == "score_prototype_shrinkage" and template_provenance is not None:
        raise ValueError("V2 P1 score completion must not claim P2 provenance.")
    for field in (
        "base_scores_sha256",
        "base_probabilities_sha256",
        "fused_probabilities_sha256",
        "affinities_sha256",
    ):
        if not _is_sha256(receipt.get(field)):
            raise ValueError(f"V2 score completion receipt has an invalid {field}.")
    if receipt.get("base_probabilities_sha256") != _array_sha256(base) or receipt.get(
        "fused_probabilities_sha256"
    ) != _array_sha256(fused):
        raise ValueError("V2 score completion receipt probability hash drifted.")
    return receipt


def validate_v2_development_selection_receipt(
    receipt_or_path: Mapping[str, Any] | str | Path,
    *,
    contract: V2ContractBinding,
) -> dict[str, Any]:
    receipt = _load_receipt(receipt_or_path)
    _validate_completion_hash(receipt, expected_schema=V2_SELECTION_RECEIPT_SCHEMA)
    if set(receipt) != _SELECTION_RECEIPT_FIELDS:
        raise ValueError("V2 development selection receipt has a wrong exact schema.")
    summaries = receipt.get("candidate_summaries")
    selected = receipt.get("selected_candidate")
    status = receipt.get("status")
    if (
        receipt.get("candidate_id") != V2_CANDIDATE_ID
        or receipt.get("phase") != "development_selection"
        or receipt.get("plan_sha256") != contract.plan_sha256
        or receipt.get("plan_content_sha256") != contract.plan_content_sha256
        or receipt.get("allocation_sha256") != contract.allocation_sha256
        or receipt.get("allocation_file_sha256") != contract.allocation_file_sha256
        or receipt.get("criterion_order") != list(_PLAN_CRITERION_ORDER)
        or not isinstance(summaries, list)
        or len(summaries) != 12
        or receipt.get("n_candidates") != 12
        or receipt.get("n_participants") != 42
        or receipt.get("held_access_authorized") is not False
    ):
        raise ValueError("V2 development selection receipt binding drifted.")
    expected_grid = v2_candidate_grid(contract)
    if receipt.get("candidate_grid_sha256") != _canonical_json_sha256(
        {"candidates": [item.as_dict() for item in expected_grid]}
    ):
        raise ValueError("V2 development selection candidate-grid hash drifted.")
    observed_keys = [
        str(item.get("candidate_key")) for item in summaries if isinstance(item, Mapping)
    ]
    if observed_keys != [item.candidate_key for item in expected_grid]:
        raise ValueError("V2 development selection summaries are incomplete or reordered.")
    expected_specs = {item.candidate_key: item.as_dict() for item in expected_grid}
    for summary in summaries:
        if not isinstance(summary, Mapping) or set(summary) != _DEVELOPMENT_SUMMARY_FIELDS:
            raise ValueError("V2 development candidate summary has a wrong exact schema.")
        candidate_key = str(summary["candidate_key"])
        if {key: summary[key] for key in _CANDIDATE_SPEC_FIELDS} != expected_specs[candidate_key]:
            raise ValueError("V2 development candidate summary specification drifted.")
        k1_means = summary["dataset_k1_means"]
        eauc_means = summary["dataset_eauc_means"]
        inference = summary["pooled_k1_inference"]
        if (
            not isinstance(k1_means, Mapping)
            or tuple(k1_means) != V2_EXTERNAL_DATASETS
            or not isinstance(eauc_means, Mapping)
            or tuple(eauc_means) != V2_EXTERNAL_DATASETS
            or not all(
                np.isfinite(float(value)) for value in (*k1_means.values(), *eauc_means.values())
            )
            or not isinstance(inference, Mapping)
            or inference.get("schema")
            != "cfeg.metadata-calibration-v2-one-sided-paired-inference.v1"
            or inference.get("n_participants") != 42
            or inference.get("null_margin") != -0.025
            or not np.isfinite(float(inference.get("confidence_interval_low", np.nan)))
        ):
            raise ValueError("V2 development candidate statistics are malformed.")
        datasetwise_pass = all(float(value) >= 0.0 for value in k1_means.values())
        pooled_pass = float(inference["confidence_interval_low"]) > -0.025
        expected_reasons = [
            reason
            for failed, reason in (
                (not datasetwise_pass, "datasetwise_k1_mean_below_zero"),
                (not pooled_pass, "pooled_k1_lower_bound_not_above_minus_0p025"),
            )
            if failed
        ]
        if (
            summary["equal_dataset_mean_eauc_delta"] != float(np.mean(list(eauc_means.values())))
            or summary["datasetwise_k1_mean_nonnegative"] is not datasetwise_pass
            or summary["pooled_k1_lower_bound_above_minus_0p025"] is not pooled_pass
            or summary["eligible"] is not (datasetwise_pass and pooled_pass)
            or summary["rejection_reasons"] != expected_reasons
        ):
            raise ValueError("V2 development candidate decision fields are inconsistent.")
    expected_selected_summary = _select_summary_by_frozen_ties(summaries)
    expected_selected = (
        None
        if expected_selected_summary is None
        else {key: expected_selected_summary[key] for key in _CANDIDATE_SPEC_FIELDS}
    )
    selected_state = status == "selected_candidate_frozen"
    if (
        selected_state != isinstance(selected, Mapping)
        or receipt.get("k1_global_enabled") is not selected_state
        or receipt.get("candidate_and_parameters_immutable") is not selected_state
        or receipt.get("independent_gate_authorized") is not selected_state
        or selected != expected_selected
    ):
        raise ValueError("V2 development selection state is internally inconsistent.")
    if status not in {"selected_candidate_frozen", "no_eligible_candidate_terminate"}:
        raise ValueError("V2 development selection status is unknown.")
    if selected_state:
        selected_specs = [
            item.as_dict()
            for item in expected_grid
            if item.candidate_key == selected["candidate_key"]
        ]
        if selected_specs != [dict(selected)]:
            raise ValueError("V2 selected candidate is outside the frozen grid.")
    if not _is_sha256(receipt.get("participant_delta_sha256")):
        raise ValueError("V2 development receipt lacks a participant-delta SHA-256.")
    return receipt


def validate_v2_independent_gate_receipt(
    receipt_or_path: Mapping[str, Any] | str | Path,
    *,
    contract: V2ContractBinding,
) -> dict[str, Any]:
    receipt = _load_receipt(receipt_or_path)
    _validate_completion_hash(receipt, expected_schema=V2_GATE_RECEIPT_SCHEMA)
    if set(receipt) != _GATE_RECEIPT_FIELDS:
        raise ValueError("V2 independent gate receipt has a wrong exact schema.")
    requirements = receipt.get("requirements")
    if (
        receipt.get("candidate_id") != V2_CANDIDATE_ID
        or receipt.get("phase") != "independent_A_Q_gate"
        or receipt.get("plan_sha256") != contract.plan_sha256
        or receipt.get("plan_content_sha256") != contract.plan_content_sha256
        or receipt.get("allocation_sha256") != contract.allocation_sha256
        or receipt.get("allocation_file_sha256") != contract.allocation_file_sha256
        or not isinstance(requirements, Mapping)
        or tuple(requirements) != _GATE_REQUIREMENT_NAMES
        or any(type(value) is not bool for value in requirements.values())
        or receipt.get("n_participants") != 84
        or receipt.get("held_access_authorized") is not False
        or not _is_sha256(receipt.get("development_selection_receipt_sha256"))
        or not _is_sha256(receipt.get("participant_delta_sha256"))
    ):
        raise ValueError("V2 independent gate receipt binding drifted.")
    selected = receipt.get("selected_candidate")
    frozen_grid = {item.candidate_key: item.as_dict() for item in v2_candidate_grid(contract)}
    if (
        not isinstance(selected, Mapping)
        or set(selected) != _CANDIDATE_SPEC_FIELDS
        or frozen_grid.get(str(selected.get("candidate_key"))) != dict(selected)
    ):
        raise ValueError("V2 independent gate candidate is outside the frozen grid.")
    pooled = receipt.get("pooled_eauc_inference")
    dataset_k1 = receipt.get("dataset_k1_means")
    budget_bounds = receipt.get("dataset_budget_lower_bounds")
    severe = receipt.get("severe_harm")
    if (
        not isinstance(pooled, Mapping)
        or pooled.get("schema") != "cfeg.metadata-calibration-v2-one-sided-paired-inference.v1"
        or pooled.get("n_participants") != 84
        or pooled.get("null_margin") != 0.0
        or not np.isfinite(float(pooled.get("mean", np.nan)))
        or not np.isfinite(float(pooled.get("confidence_interval_low", np.nan)))
        or not isinstance(dataset_k1, Mapping)
        or tuple(dataset_k1) != V2_EXTERNAL_DATASETS
        or not all(np.isfinite(float(value)) for value in dataset_k1.values())
        or not isinstance(budget_bounds, Mapping)
        or tuple(budget_bounds) != V2_EXTERNAL_DATASETS
        or not isinstance(severe, Mapping)
        or set(severe)
        != {
            "cutoff",
            "count",
            "n_participants",
            "observed_rate",
            "one_sided_95_percent_clopper_pearson_upper",
        }
    ):
        raise ValueError("V2 independent gate statistics are malformed.")
    dataset_counts = {"beta_v1": 45, "dong2023_v1": 39}
    for dataset_id, expected_count in dataset_counts.items():
        per_budget = budget_bounds[dataset_id]
        if not isinstance(per_budget, Mapping) or tuple(per_budget) != tuple(map(str, V2_BUDGETS)):
            raise ValueError("V2 independent gate budget bounds have a wrong exact schema.")
        for summary in per_budget.values():
            if (
                not isinstance(summary, Mapping)
                or summary.get("method") != "one_sided_participant_paired_student_t_lower_bound"
                or summary.get("n_participants") != expected_count
                or summary.get("alpha") != 0.05
                or not np.isfinite(float(summary.get("confidence_interval_low", np.nan)))
            ):
                raise ValueError("V2 independent gate dataset-budget inference is malformed.")
    severe_count = severe.get("count")
    if (
        severe.get("cutoff") != -0.10
        or type(severe_count) is not int
        or not 0 <= severe_count <= 84
        or severe.get("n_participants") != 84
        or severe.get("observed_rate") != severe_count / 84.0
        or severe.get("one_sided_95_percent_clopper_pearson_upper")
        != _binomial_upper_bound(severe_count, 84, alpha=0.05)
    ):
        raise ValueError("V2 independent gate severe-harm summary is inconsistent.")
    expected_requirements = {
        "pooled_one_sided_95_percent_lower_bound_gt_0": bool(
            float(pooled["confidence_interval_low"]) > 0.0
        ),
        "pooled_observed_mean_ge_0p01": bool(float(pooled["mean"]) >= 0.01),
        "each_dataset_k1_mean_ge_0": all(float(value) >= 0.0 for value in dataset_k1.values()),
        "each_dataset_each_budget_lower_bound_gt_minus_0p025": all(
            float(summary["confidence_interval_low"]) > -0.025
            for per_budget in budget_bounds.values()
            for summary in per_budget.values()
        ),
        "participant_delta_below_minus_0p10_upper_rate_lt_0p10": bool(
            float(severe["one_sided_95_percent_clopper_pearson_upper"]) < 0.10
        ),
    }
    if dict(requirements) != expected_requirements:
        raise ValueError("V2 independent gate requirement booleans disagree with statistics.")
    passed = all(requirements.values())
    if (
        receipt.get("all_requirements_passed") is not passed
        or receipt.get("status") != ("pass" if passed else "failed_terminate_before_held60")
        or receipt.get("next_required")
        != (
            "choi_replication_clean_tag_power_receipt_and_new_signed_owner_authorization"
            if passed
            else "terminate_candidate_without_wearable_held60_access"
        )
    ):
        raise ValueError("V2 independent gate decision is internally inconsistent.")
    return receipt


def validate_v2_participant_deltas(
    participant_deltas: pd.DataFrame,
    *,
    contract: V2ContractBinding,
    cohort: Literal["development", "independent_gate"],
    expected_candidate_keys: Sequence[str],
) -> pd.DataFrame:
    """Validate a complete, allocation-bound participant delta table."""

    if cohort not in V2_EXTERNAL_COHORTS:
        raise ValueError("V2 participant deltas requested an unknown cohort.")
    if set(participant_deltas.columns) != _PARTICIPANT_DELTA_COLUMNS:
        raise ValueError("V2 participant delta table has a wrong exact schema.")
    frame = participant_deltas.loc[:, sorted(_PARTICIPANT_DELTA_COLUMNS)].copy()
    _validate_common_bindings(frame, contract=contract, cohort=cohort)
    candidate_keys = tuple(str(value) for value in expected_candidate_keys)
    if not candidate_keys or len(set(candidate_keys)) != len(candidate_keys):
        raise ValueError("Expected V2 candidate keys must be nonempty and unique.")
    if set(frame["candidate_key"].astype(str)) != set(candidate_keys):
        raise ValueError("V2 participant deltas contain the wrong candidate set.")
    _validate_candidate_columns(frame, contract=contract, expected_candidate_keys=candidate_keys)
    numeric = frame.loc[
        :,
        [f"{prefix}_k{budget}" for prefix in ("a0", "aq", "delta") for budget in V2_BUDGETS]
        + ["a0_eauc", "aq_eauc", "eauc_delta"],
    ].apply(pd.to_numeric, errors="raise")
    if not np.isfinite(numeric.to_numpy(dtype=float)).all():
        raise ValueError("V2 participant deltas must be finite.")
    for prefix in ("a0", "aq"):
        values = numeric[[f"{prefix}_k{budget}" for budget in V2_BUDGETS]].to_numpy()
        if (values < 0.0).any() or (values > 1.0).any():
            raise ValueError("V2 participant BA values must lie in [0,1].")
    for budget in V2_BUDGETS:
        if not np.allclose(
            numeric[f"delta_k{budget}"].to_numpy(dtype=float),
            numeric[f"aq_k{budget}"].to_numpy(dtype=float)
            - numeric[f"a0_k{budget}"].to_numpy(dtype=float),
            rtol=0.0,
            atol=1e-15,
        ):
            raise ValueError(f"V2 participant delta_k{budget} is algebraically inconsistent.")
    if not np.array_equal(
        numeric["a0_k0"].to_numpy(dtype=float), numeric["aq_k0"].to_numpy(dtype=float)
    ):
        raise ValueError("V2 participant deltas violate exact k=0 A_Q/A0 equality.")
    expected_a0 = sum(
        weight * numeric[f"a0_k{budget}"] for budget, weight in V2_EAUC_WEIGHTS.items()
    )
    expected_aq = sum(
        weight * numeric[f"aq_k{budget}"] for budget, weight in V2_EAUC_WEIGHTS.items()
    )
    if (
        not np.allclose(numeric["a0_eauc"], expected_a0, rtol=0.0, atol=1e-15)
        or not np.allclose(numeric["aq_eauc"], expected_aq, rtol=0.0, atol=1e-15)
        or not np.allclose(numeric["eauc_delta"], expected_aq - expected_a0, rtol=0.0, atol=1e-15)
    ):
        raise ValueError("V2 participant eAUC values are algebraically inconsistent.")
    key = ["candidate_key", "dataset_id", "subject_id"]
    if frame.duplicated(key).any():
        raise ValueError("V2 participant delta table contains duplicate participant candidates.")
    expected_rows = {
        (candidate_key, dataset_id, subject_id)
        for candidate_key in candidate_keys
        for dataset_id in V2_EXTERNAL_DATASETS
        for subject_id in contract.subject_ids(dataset_id, cohort)
    }
    observed_rows = set(frame[key].astype(str).itertuples(index=False, name=None))
    if observed_rows != expected_rows:
        raise ValueError("V2 participant delta table is not the exact allocated candidate grid.")
    return frame.sort_values(key, kind="mergesort").reset_index(drop=True)


def canonical_dataframe_sha256(frame: pd.DataFrame) -> str:
    """Hash an exact finite dataframe independently of row and column order."""

    if frame.columns.duplicated().any():
        raise ValueError("Cannot hash a dataframe with duplicate columns.")
    ordered_columns = sorted(map(str, frame.columns))
    normalized = frame.loc[:, ordered_columns].copy()
    sort_columns = ordered_columns
    if sort_columns:
        normalized = normalized.sort_values(sort_columns, kind="mergesort", na_position="first")
    records: list[dict[str, object]] = []
    for raw in normalized.to_dict(orient="records"):
        record: dict[str, object] = {}
        for key in ordered_columns:
            value = raw[key]
            if value is None or (isinstance(value, float) and math.isnan(value)):
                record[key] = None
            elif isinstance(value, np.generic):
                record[key] = value.item()
            else:
                record[key] = value
        records.append(record)
    return _canonical_json_sha256({"columns": ordered_columns, "rows": records})


def write_v2_completion_receipt_exclusive(
    path: str | Path,
    receipt: Mapping[str, Any],
) -> Path:
    """Atomically publish one read-only completion receipt without replacement."""

    target = Path(path).expanduser().absolute()
    _reject_forbidden_external_path(target)
    schema = receipt.get("schema")
    _validate_completion_hash(receipt, expected_schema=str(schema))
    if schema not in {
        V2_PREPARATION_RECEIPT_SCHEMA,
        V2_SELECTION_RECEIPT_SCHEMA,
        V2_GATE_RECEIPT_SCHEMA,
        V2_SCORE_RECEIPT_SCHEMA,
    }:
        raise ValueError("Unknown V2 completion receipt schema.")
    if os.path.lexists(target):
        raise FileExistsError("V2 completion receipt path already exists.")
    target.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    if target.parent.is_symlink() or not target.parent.is_dir():
        raise ValueError("V2 completion receipt parent must be a real directory.")
    staging = target.parent / f".{target.name}.staging-{uuid.uuid4().hex}"
    payload = (
        json.dumps(dict(receipt), sort_keys=True, indent=2, ensure_ascii=False, allow_nan=False)
        + "\n"
    )
    descriptor = os.open(staging, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o400)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            descriptor = -1
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(staging, target)
        _fsync_directory(target.parent)
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        if os.path.lexists(staging):
            staging.unlink()
            _fsync_directory(target.parent)
    return target


def reject_forbidden_v2_data_path(path: str | Path) -> Path:
    """Public fail-closed boundary for development/gate data-bearing paths."""

    target = Path(path).expanduser().absolute()
    _reject_forbidden_external_path(target)
    return target


def _validate_prediction_grid(
    predictions: pd.DataFrame,
    *,
    contract: V2ContractBinding,
    cohort: str,
    expected_candidate_keys: Sequence[str] | None,
    n_classes: int,
) -> pd.DataFrame:
    if type(n_classes) is not int or n_classes < 2:
        raise ValueError("n_classes must be an exact integer of at least two.")
    if set(predictions.columns) != _PREDICTION_COLUMNS:
        raise ValueError("V2 external prediction table has a wrong exact schema.")
    frame = predictions.loc[:, sorted(_PREDICTION_COLUMNS)].copy()
    _validate_common_bindings(frame, contract=contract, cohort=cohort)
    candidate_keys = (
        tuple(item.candidate_key for item in v2_candidate_grid(contract))
        if expected_candidate_keys is None
        else tuple(map(str, expected_candidate_keys))
    )
    if set(frame["candidate_key"].astype(str)) != set(candidate_keys):
        raise ValueError("V2 prediction table contains the wrong candidate set.")
    _validate_candidate_columns(frame, contract=contract, expected_candidate_keys=candidate_keys)
    frame["budget"] = _strict_integer_series(frame["budget"], name="budget")
    frame["label"] = _strict_integer_series(frame["label"], name="label")
    frame["prediction"] = _strict_integer_series(frame["prediction"], name="prediction")
    if set(frame["budget"]) != set(V2_BUDGETS) or set(frame["role"].astype(str)) != {
        "A0",
        "A_Q",
    }:
        raise ValueError("V2 prediction roles or budgets differ from the external contract.")
    if (
        (frame["label"] < 0).any()
        or (frame["label"] >= n_classes).any()
        or (frame["prediction"] < 0).any()
        or (frame["prediction"] >= n_classes).any()
    ):
        raise ValueError("V2 labels or predictions lie outside the class range.")
    if frame["query_token"].astype(str).str.strip().eq("").any():
        raise ValueError("V2 query tokens must be nonempty opaque identifiers.")
    key = ["candidate_key", "dataset_id", "subject_id", "role", "budget", "query_token"]
    if frame.duplicated(key).any():
        raise ValueError("V2 prediction table contains duplicate atomic rows.")

    expected_subject_rows = {
        (dataset_id, subject_id)
        for dataset_id in V2_EXTERNAL_DATASETS
        for subject_id in contract.subject_ids(dataset_id, cohort)  # type: ignore[arg-type]
    }
    observed_subject_rows = set(
        frame[["dataset_id", "subject_id"]]
        .astype(str)
        .drop_duplicates()
        .itertuples(index=False, name=None)
    )
    if observed_subject_rows != expected_subject_rows:
        raise ValueError("V2 prediction table differs from the frozen cohort allocation.")
    groups = frame.groupby(
        ["candidate_key", "dataset_id", "subject_id", "role", "budget"],
        sort=False,
    )
    if not groups.size().eq(n_classes).all():
        raise ValueError(
            "Every V2 external participant cell must contain one complete class block."
        )
    if not groups["label"].agg(lambda values: set(map(int, values)) == set(range(n_classes))).all():
        raise ValueError("Every V2 external participant cell must cover every class exactly once.")
    query_identity = frame.groupby(["dataset_id", "subject_id", "query_token"], sort=False)[
        "label"
    ].nunique()
    if not query_identity.eq(1).all():
        raise ValueError("V2 query labels differ across candidates, roles, or budgets.")
    token_sets = groups["query_token"].agg(lambda values: tuple(sorted(map(str, values))))
    if not token_sets.groupby(level=[1, 2]).nunique().eq(1).all():
        raise ValueError("V2 query-token coverage differs across candidate cells.")

    anchor = frame.loc[frame["role"].eq("A0")]
    anchor_consistency = anchor.groupby(["dataset_id", "subject_id", "query_token"], sort=False)[
        "prediction"
    ].nunique()
    if not anchor_consistency.eq(1).all():
        raise ValueError("Repeated strict-FBCCA A0 predictions changed across candidate cells.")
    k0 = frame.loc[frame["budget"].eq(0)].pivot(
        index=["candidate_key", "dataset_id", "subject_id", "query_token"],
        columns="role",
        values="prediction",
    )
    if not np.array_equal(k0["A0"].to_numpy(), k0["A_Q"].to_numpy()):
        raise ValueError("V2 A_Q k=0 predictions are not exactly the strict-FBCCA anchor.")
    return frame


def _validate_participant_ba(
    participant_ba: pd.DataFrame,
    *,
    contract: V2ContractBinding,
    cohort: str,
    expected_candidate_keys: Sequence[str] | None,
) -> pd.DataFrame:
    if set(participant_ba.columns) != _PARTICIPANT_BA_COLUMNS:
        raise ValueError("V2 participant BA table has a wrong exact schema.")
    frame = participant_ba.loc[:, sorted(_PARTICIPANT_BA_COLUMNS)].copy()
    _validate_common_bindings(frame, contract=contract, cohort=cohort)
    candidate_keys = (
        tuple(item.candidate_key for item in v2_candidate_grid(contract))
        if expected_candidate_keys is None
        else tuple(map(str, expected_candidate_keys))
    )
    _validate_candidate_columns(frame, contract=contract, expected_candidate_keys=candidate_keys)
    frame["budget"] = _strict_integer_series(frame["budget"], name="budget")
    frame["balanced_accuracy"] = pd.to_numeric(frame["balanced_accuracy"], errors="raise")
    if (
        not np.isfinite(frame["balanced_accuracy"].to_numpy(dtype=float)).all()
        or (frame["balanced_accuracy"] < 0.0).any()
        or (frame["balanced_accuracy"] > 1.0).any()
    ):
        raise ValueError("V2 participant balanced accuracy must be finite and in [0,1].")
    key = ["candidate_key", "dataset_id", "subject_id", "role", "budget"]
    if frame.duplicated(key).any():
        raise ValueError("V2 participant BA table contains duplicate cells.")
    expected = {
        (candidate_key, dataset_id, subject_id, role, budget)
        for candidate_key in candidate_keys
        for dataset_id in V2_EXTERNAL_DATASETS
        for subject_id in contract.subject_ids(dataset_id, cohort)  # type: ignore[arg-type]
        for role in ("A0", "A_Q")
        for budget in V2_BUDGETS
    }
    observed = set(
        frame[key].assign(budget=frame["budget"].astype(int)).itertuples(index=False, name=None)
    )
    if observed != expected:
        raise ValueError("V2 participant BA table is not the exact allocated candidate grid.")
    return frame


def _validate_common_bindings(
    frame: pd.DataFrame,
    *,
    contract: V2ContractBinding,
    cohort: str,
) -> None:
    if cohort not in V2_EXTERNAL_COHORTS:
        raise ValueError("Unknown V2 external cohort.")
    expected_constant = {
        "candidate_id": V2_CANDIDATE_ID,
        "plan_sha256": contract.plan_sha256,
        "allocation_sha256": contract.allocation_sha256,
        "cohort": cohort,
    }
    for column, expected in expected_constant.items():
        if set(frame[column].astype(str)) != {str(expected)}:
            raise ValueError(f"V2 table has an invalid {column} binding.")
    if set(frame["dataset_id"].astype(str)) != set(V2_EXTERNAL_DATASETS):
        raise ValueError("V2 table must contain exact BETA and Dong datasets.")
    for dataset_id in V2_EXTERNAL_DATASETS:
        dataset = frame.loc[frame["dataset_id"].astype(str).eq(dataset_id)]
        expected_asset = contract.asset_binding(dataset_id)
        for column, expected in expected_asset.items():
            if set(dataset[column].astype(str)) != {expected}:
                raise ValueError(f"V2 {dataset_id} table has an invalid {column} binding.")


def _validate_candidate_columns(
    frame: pd.DataFrame,
    *,
    contract: V2ContractBinding,
    expected_candidate_keys: Sequence[str],
) -> None:
    grid = {item.candidate_key: item for item in v2_candidate_grid(contract)}
    expected_keys = tuple(map(str, expected_candidate_keys))
    if not expected_keys or not set(expected_keys).issubset(grid):
        raise ValueError("Expected candidates fall outside the frozen V2 grid.")
    if set(frame["candidate_key"].astype(str)) != set(expected_keys):
        raise ValueError("V2 table contains the wrong candidate keys.")
    for candidate_key in expected_keys:
        selected = frame.loc[frame["candidate_key"].astype(str).eq(candidate_key)]
        spec = grid[candidate_key]
        if set(selected["operator"].astype(str)) != {spec.operator}:
            raise ValueError(f"V2 candidate {candidate_key} operator binding drifted.")
        lambda_values = pd.to_numeric(selected["lambda_max"], errors="raise").to_numpy(float)
        if (
            not np.isfinite(lambda_values).all()
            or not np.equal(lambda_values, spec.lambda_max).all()
        ):
            raise ValueError(f"V2 candidate {candidate_key} lambda binding drifted.")
        pseudocount = pd.to_numeric(
            selected["prototype_prior_pseudocount"], errors="coerce"
        ).to_numpy(float)
        if spec.prototype_prior_pseudocount is None:
            if not np.isnan(pseudocount).all():
                raise ValueError(f"V2 template candidate {candidate_key} gained a pseudocount.")
        elif not np.equal(pseudocount, spec.prototype_prior_pseudocount).all():
            raise ValueError(f"V2 candidate {candidate_key} pseudocount binding drifted.")


def _single_candidate_spec(frame: pd.DataFrame) -> V2CandidateSpec:
    candidate_keys = tuple(frame["candidate_key"].astype(str).unique())
    if len(candidate_keys) != 1:
        raise ValueError("Expected exactly one frozen V2 candidate.")
    pseudocounts = frame["prototype_prior_pseudocount"]
    present = pd.to_numeric(pseudocounts, errors="coerce").dropna().unique()
    pseudocount = None if len(present) == 0 else float(present[0])
    return V2CandidateSpec(
        candidate_key=candidate_keys[0],
        operator=str(frame["operator"].iloc[0]),  # type: ignore[arg-type]
        lambda_max=float(frame["lambda_max"].iloc[0]),
        prototype_prior_pseudocount=pseudocount,
    )


def _select_summary_by_frozen_ties(
    summaries: Sequence[Mapping[str, Any]],
) -> Mapping[str, Any] | None:
    eligible = [item for item in summaries if item.get("eligible") is True]
    if not eligible:
        return None
    best_eauc = max(float(item["equal_dataset_mean_eauc_delta"]) for item in eligible)
    finalists = [
        item for item in eligible if float(item["equal_dataset_mean_eauc_delta"]) == best_eauc
    ]
    minimum_lambda = min(float(item["lambda_max"]) for item in finalists)
    finalists = [item for item in finalists if float(item["lambda_max"]) == minimum_lambda]
    if any(item["operator"] == "score_prototype_shrinkage" for item in finalists):
        finalists = [item for item in finalists if item["operator"] == "score_prototype_shrinkage"]
    minimum_pseudocount = min(
        math.inf
        if item["prototype_prior_pseudocount"] is None
        else float(item["prototype_prior_pseudocount"])
        for item in finalists
    )
    finalists = [
        item
        for item in finalists
        if (
            math.inf
            if item["prototype_prior_pseudocount"] is None
            else float(item["prototype_prior_pseudocount"])
        )
        == minimum_pseudocount
    ]
    if len(finalists) != 1:
        raise RuntimeError("Frozen V2 development tie rules did not identify one candidate.")
    return finalists[0]


def _candidate_key(operator: str, lambda_max: float, pseudocount: float | None) -> str:
    lambda_token = f"{lambda_max:.2f}".replace(".", "p")
    if operator == "score_prototype_shrinkage":
        assert pseudocount is not None
        pseudocount_token = str(int(pseudocount))
        return f"score-prototype-pc{pseudocount_token}-lambda{lambda_token}"
    return f"template-residual-lambda{lambda_token}"


def _strict_subject_list(value: object, *, name: str) -> tuple[str, ...]:
    if not isinstance(value, list) or any(type(item) is not str for item in value):
        raise TypeError(f"{name} must be a JSON string list.")
    subjects = tuple(value)
    if subjects != tuple(sorted(set(subjects))) or any(
        len(subject) != 6 or not subject.startswith("sub") or not subject[3:].isdigit()
        for subject in subjects
    ):
        raise ValueError(f"{name} must contain sorted unique subNNN identifiers.")
    return subjects


def _strict_integer_series(values: pd.Series, *, name: str) -> pd.Series:
    numeric = pd.to_numeric(values, errors="raise")
    array = numeric.to_numpy(dtype=float)
    if not np.isfinite(array).all() or not np.equal(array, np.floor(array)).all():
        raise ValueError(f"V2 {name} values must be finite exact integers.")
    return numeric.astype(np.int64)


def _finite_vector(values: Sequence[float] | np.ndarray, *, name: str) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or len(array) < 2 or not np.isfinite(array).all():
        raise ValueError(f"{name} must be a one-dimensional vector with at least two values.")
    return array


def _binomial_upper_bound(successes: int, total: int, *, alpha: float) -> float:
    if type(successes) is not int or type(total) is not int or not 0 <= successes <= total:
        raise ValueError("Binomial count must satisfy 0 <= successes <= total.")
    if total < 1 or not 0.0 < alpha < 1.0:
        raise ValueError("Binomial total and alpha are invalid.")
    if successes == total:
        return 1.0
    return float(beta_distribution.ppf(1.0 - alpha, successes + 1, total - successes))


def _derived_seed(master: int, *parts: str) -> int:
    payload = ":".join((str(master), *map(str, parts)))
    return int.from_bytes(hashlib.sha256(payload.encode("utf-8")).digest()[:8], "big")


def _with_completion_hash(payload: Mapping[str, Any]) -> dict[str, Any]:
    result = dict(payload)
    if "completion_receipt_sha256" in result:
        raise ValueError("Completion payload already contains its self-hash.")
    result["completion_receipt_sha256"] = _canonical_json_sha256(result)
    return result


def _validate_completion_hash(receipt: Mapping[str, Any], *, expected_schema: str) -> None:
    if receipt.get("schema") != expected_schema:
        raise ValueError("V2 completion receipt schema is unknown.")
    payload = dict(receipt)
    observed = payload.pop("completion_receipt_sha256", None)
    if not _is_sha256(observed) or observed != _canonical_json_sha256(payload):
        raise ValueError("V2 completion receipt self-hash is invalid.")


def _load_receipt(value: Mapping[str, Any] | str | Path) -> dict[str, Any]:
    if isinstance(value, Mapping):
        return dict(value)
    path = _regular_file_path(value, name="V2 completion receipt")
    result = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(result, dict):
        raise TypeError("V2 completion receipt must decode to one mapping.")
    return result


def _canonical_json_sha256(value: object) -> str:
    try:
        payload = json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise ValueError("V2 canonical JSON values must be finite and serializable.") from error
    return hashlib.sha256(payload).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _is_sha256(value: object) -> bool:
    return isinstance(value, str) and re.fullmatch(_SHA256_PATTERN, value) is not None


def _optional_numeric_array(
    value: object,
    *,
    dtype: type[np.float64 | np.int64] | np.dtype[Any] = np.float64,
) -> np.ndarray | None:
    if value is None:
        return None
    try:
        raw = np.asarray(value)
    except (TypeError, ValueError) as error:
        raise TypeError("V2 score request arrays must be rectangular numeric values.") from error
    if not np.issubdtype(raw.dtype, np.number) or np.issubdtype(raw.dtype, np.bool_):
        raise TypeError("V2 score request arrays must contain JSON numbers, not strings or bools.")
    array = raw.astype(dtype, copy=False)
    if not np.isfinite(array).all():
        raise ValueError("V2 score request numeric arrays must be finite.")
    return array


def _array_sha256(value: np.ndarray) -> str:
    array = np.asarray(value)
    if not np.isfinite(array).all():
        raise ValueError("Cannot hash a non-finite V2 array.")
    canonical = np.ascontiguousarray(array)
    header = json.dumps(
        {"dtype": canonical.dtype.str, "shape": list(canonical.shape)},
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    digest = hashlib.sha256()
    for part in (header, canonical.tobytes(order="C")):
        digest.update(len(part).to_bytes(8, "big"))
        digest.update(part)
    return digest.hexdigest()


def _regular_file_path(path: str | Path, *, name: str) -> Path:
    target = Path(path).expanduser().absolute()
    if not target.exists():
        raise FileNotFoundError(f"{name} does not exist: {target}.")
    observed = target.lstat()
    if stat.S_ISLNK(observed.st_mode) or not stat.S_ISREG(observed.st_mode):
        raise ValueError(f"{name} must be one real regular file.")
    return target


def _reject_forbidden_external_path(path: Path) -> None:
    existing = path
    while not os.path.lexists(existing):
        if existing.parent == existing:
            break
        existing = existing.parent
    if existing.is_symlink() or any(parent.is_symlink() for parent in existing.parents):
        raise PermissionError("V2 development/gate paths must not traverse symlinks.")
    resolved = path.resolve(strict=False)
    lowered = {str(candidate).lower() for candidate in (path, resolved)}
    components = {part.lower() for candidate in (path, resolved) for part in candidate.parts}
    if (
        any("wearable" in part for part in components)
        or any(
            "source39" in part or "source-39" in part or "source_39" in part for part in components
        )
        or any(
            part == "held" or "held60" in part or "held-60" in part or "held_60" in part
            for part in components
        )
        or any(value.endswith("/eeg-results") or "/eeg-results/" in value for value in lowered)
    ):
        raise PermissionError(
            "V2 development/gate paths must not reference wearable, source39, held60, "
            "or the outcome-results tree."
        )


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
