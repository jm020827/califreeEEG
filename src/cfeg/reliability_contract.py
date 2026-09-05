from __future__ import annotations

import copy
import hashlib
import json
from collections.abc import Mapping
from pathlib import Path

from cfeg.data.synthetic_quality import SYNTHETIC_QUALITY_ASSET_FILES
from cfeg.governance import (
    implementation_contract_sha256,
    validate_physical_candidate_retirement,
)
from cfeg.utils.config import load_config, merge_overrides

RELIABILITY_ROLES = ("A0", "A_M", "A_Q", "A_QM")
RELIABILITY_TREATMENTS = {
    "A0": ("null", "null"),
    "A_M": ("null", "observed"),
    "A_Q": ("observed", "null"),
    "A_QM": ("observed", "observed"),
}
RELIABILITY_CANDIDATE_PLAN = "configs/analysis/reliability_spatial_v1.yaml"
RELIABILITY_STAGE0_PLAN = "configs/analysis/synthetic_reliability_stage0.yaml"
RELIABILITY_BASE_CONFIG = "configs/train/synthetic_reliability_candidate.yaml"
RELIABILITY_SUITE_CONFIG = "configs/train/synthetic_reliability_2x2.yaml"
PHYSICAL_RETIREMENT_RECEIPT = "configs/governance/wearable_s1_s3_retirement.json"
STAGE0_ARTIFACT_FILES = {
    "predictions.csv",
    "subject_metrics.csv",
    "contrasts.csv",
    "fold_models.json",
}
_STAGE0_VERIFICATION_CACHE: set[tuple[str, str, str, str]] = set()


def validate_reliability_family(base: dict, suite: dict) -> str | None:
    """Validate the new 2x2 mechanism family without changing old A0/A2 contracts."""

    family = suite.get("reliability_family")
    if family is None:
        return None
    if family.get("schema") != "cfeg.reliability-mechanism-family.v1":
        raise ValueError("Unknown reliability mechanism-family schema.")
    candidate = validate_reliability_candidate_bindings(base)
    if tuple(family.get("roles") or ()) != RELIABILITY_ROLES:
        raise ValueError(f"Reliability family must declare exact roles {RELIABILITY_ROLES}.")
    if (
        family.get("primary_contrast") != "A_QM_minus_A_Q"
        or family.get("secondary_contrast") != "A_M_minus_A0"
        or list(family.get("treatment_axes") or [])
        != ["query_qc_access", "external_metadata_access"]
    ):
        raise ValueError("Reliability family has an invalid primary contrast/access-axis contract.")
    for field in (
        "population_inference_allowed",
        "confirmatory_execution_authorized",
        "wearable_s1_s3_reuse_allowed",
    ):
        if family.get(field) is not False:
            raise ValueError(f"Reliability engineering family requires {field}=false.")
    if family.get("evidence_scope") != "synthetic_engineering_only":
        raise ValueError("Reliability family is currently restricted to synthetic engineering.")
    if base.get("protocol", {}).get("execution_phase") != "synthetic_engineering":
        raise ValueError("Reliability family base must use execution_phase=synthetic_engineering.")
    if base.get("protocol", {}).get("candidate_id") != family.get("candidate_id"):
        raise ValueError("Reliability family/base candidate IDs differ.")
    if base.get("protocol", {}).get("development_control", "none") != "none":
        raise ValueError("Reliability family does not permit a development control treatment.")
    expected_revisions = base.get("data", {}).get("expected_revisions") or {}
    if expected_revisions != {"synthetic_quality": "synthetic_quality_v1"}:
        raise ValueError(
            "Synthetic reliability family may access only synthetic_quality_v1."
        )

    variants = suite.get("variants") or {}
    if set(variants) != set(RELIABILITY_ROLES):
        raise ValueError("Reliability family variants must be exactly A0/A_M/A_Q/A_QM.")
    signatures: dict[str, str] = {}
    observed_treatments: dict[str, tuple[str, str]] = {}
    for role in RELIABILITY_ROLES:
        cfg = merge_overrides(
            copy.deepcopy(base),
            [f"{key}={value!r}" for key, value in variants[role].items()],
        )
        _validate_reliability_model_contract(cfg)
        model = cfg["model"]
        treatment = (
            str(model["conditioning"]["spatial_reliability"]["query_qc_mode"]),
            str(model["condition_encoder"]["external_metadata_mode"]),
        )
        observed_treatments[role] = treatment
        if treatment != RELIABILITY_TREATMENTS[role]:
            raise ValueError(
                f"Reliability treatment mismatch for {role}: "
                f"expected {RELIABILITY_TREATMENTS[role]}, got {treatment}."
            )
        normalized = copy.deepcopy(cfg)
        normalized["model"]["conditioning"]["spatial_reliability"]["query_qc_mode"] = (
            "<query-access>"
        )
        normalized["model"]["condition_encoder"]["external_metadata_mode"] = (
            "<metadata-access>"
        )
        payload = json.dumps(normalized, sort_keys=True, separators=(",", ":"))
        signatures[role] = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    if len(set(signatures.values())) != 1:
        raise ValueError(
            "Reliability configs differ outside the two declared access axes: "
            f"{signatures}."
        )
    if _candidate_base_contract_sha256(base) != candidate["stage1"][
        "base_config_contract_sha256"
    ]:
        raise ValueError("Reliability base config differs from its frozen full contract.")
    canonical_suite = load_config(
        _repository_root() / RELIABILITY_SUITE_CONFIG, strict_env=False
    )
    if _sha256_json(suite) != _sha256_json(canonical_suite):
        raise ValueError("Reliability suite differs from its frozen canonical file.")
    payload = {
        "family": family,
        "normalized_config_sha256": signatures[RELIABILITY_ROLES[0]],
        "role_treatments": observed_treatments,
    }
    return _sha256_json(payload)


def validate_reliability_candidate_bindings(base: dict) -> dict:
    protocol = base.get("protocol", {})
    plan_path = _canonical_project_path(
        protocol.get("candidate_plan"), RELIABILITY_CANDIDATE_PLAN
    )
    retirement_path = _canonical_project_path(
        protocol.get("predecessor_retirement_receipt"),
        PHYSICAL_RETIREMENT_RECEIPT,
    )
    if protocol.get("candidate_plan_sha256") != _sha256_file(plan_path):
        raise ValueError("Reliability candidate-plan digest is missing or stale.")
    if protocol.get("predecessor_retirement_receipt_sha256") != _sha256_file(
        retirement_path
    ):
        raise ValueError("Physical predecessor-retirement digest is missing or stale.")
    retirement = _validate_physical_retirement_receipt(retirement_path)
    plan = load_config(plan_path, strict_env=False)
    canonical_base = load_config(
        _repository_root() / RELIABILITY_BASE_CONFIG, strict_env=False
    )
    canonical_suite_path = _repository_root() / RELIABILITY_SUITE_CONFIG
    expected_model = {
        "architecture": "reliability_spatial_v1",
        "backbone": "spectral_transformer",
        "placement": "pre_backbone_waveform",
        "operator_family": "diagonal_low_rank",
        "query_feature_schema": "query_window_reliability_features_v1",
        "max_operator_frobenius_norm": 0.25,
        "query_operator_frobenius_norm": 0.2,
        "metadata_operator_frobenius_norm": 0.05,
        "metadata_alpha_limit": 1.0,
        "metadata_role": "bounded_additive_residual",
        "normal_decoder_metadata_vector_route_allowed": False,
    }
    expected_stage1 = {
        "design": "exact_2x2_query_qc_by_external_metadata_access",
        "base_config": RELIABILITY_BASE_CONFIG,
        "base_config_contract_normalization": "candidate_plan_sha256_sentinel_v1",
        "base_config_contract_sha256": _candidate_base_contract_sha256(
            canonical_base
        ),
        "roles": list(RELIABILITY_ROLES),
        "treatment_axes": ["query_qc_access", "external_metadata_access"],
        "primary_contrast": "A_QM_minus_A_Q",
        "secondary_contrast": "A_M_minus_A0",
        "interaction": "(A_QM_minus_A_Q)_minus_(A_M_minus_A0)",
        "suite": RELIABILITY_SUITE_CONFIG,
        "suite_file_sha256": _sha256_file(canonical_suite_path),
        "output_root": "outputs/engineering/reliability-spatial-v1/stage1",
        "seed": 42,
        "split_seed": 42,
        "split": "cross_subject_train_val",
        "validation_subject_fraction": 0.25,
        "epochs": 8,
        "checkpoint_selection": "validation_best",
        "checkpoint_selection_metric": "accuracy",
        "all_roles_run_together": True,
        "cli_recipe_overrides_allowed": False,
        "partial_role_execution_allowed": False,
        "execution_transaction": {
            "orchestrator_authorization": (
                "in_process_private_token_not_config_or_cli"
            ),
            "concurrency_reservation": (
                "fixed_name_atomic_mkdir_before_any_outcome_execution"
            ),
            "publication": "hidden_staging_fsync_then_atomic_suite_root_rename",
            "final_input_revalidation": (
                "stage0_assets_checkpoints_splits_metrics_runtime_and_attempts_"
                "rehashed_before_publish"
            ),
            "failed_attempt": "retained_mode_0700_noncanonical_with_attempt_receipt",
            "unreceipted_io_crash": (
                "fixed_reservation_retained_for_owner_review"
            ),
            "retry": "new_owner_decision_required_after_attempt_review",
            "reservation_release": "only_after_complete_atomic_publication",
            "dry_run": "fresh_dedicated_noncanonical_prefix_only",
        },
        "analysis": {
            "schema": "cfeg.synthetic-reliability-stage1-analysis.v1",
            "score_scope": "checkpoint_selection_validation_set",
            "statistical_unit": "participant",
            "primary_metric": "subject_macro_balanced_accuracy",
            "substantive_effect_gate": "none_descriptive_only",
            "interventions": ["missing", "stale_block", "wrong_interface"],
            "external_metadata_bundle_fields": [
                "reference",
                "electrode_type",
                "cap_type",
                "impedance_mean_kohm",
                "impedance_max_kohm",
                "impedance_kohm_by_channel",
            ],
            "missing_intervention": "full_external_bundle_null",
            "stale_block_donor_scope": "same_subject_same_interface_other_run",
            "wrong_interface_donor_scope": "same_subject_same_run_opposite_interface",
            "donor_pairing_uses_label": False,
            "same_aqm_checkpoint_for_interventions": True,
            "missing_fallback_interpretation": (
                "same_aqm_checkpoint_with_metadata_operator_zero_not_separately_trained_a_q"
            ),
            "a_q_comparator": "descriptive_subject_level",
            "integrity_tolerance": 1e-6,
            "metadata_only_probe": {
                "method": "participant_disjoint_multiclass_ridge",
                "ridge_alphas": [1e-6, 0.0001, 0.01, 1.0, 10.0, 100.0],
                "active_canonical_channel_ids": [48, 55, 57, 56, 53, 61, 62, 63],
                "block_logit_invariance_tolerance": 1e-12,
                "balanced_accuracy_chance_tolerance": 1e-12,
            },
            "output_files": [
                "validation_predictions.csv",
                "validation_subject_metrics.csv",
                "validation_group_metrics.csv",
                "stage1_subject_contrasts.csv",
                "metadata_intervention_predictions.csv",
                "metadata_intervention_mapping.csv",
                "metadata_intervention_subject_metrics.csv",
                "stage1_metadata_safety.csv",
                "metadata_only_probe_predictions.csv",
                "metadata_only_probe_subject_metrics.csv",
                "metadata_only_probe_model.json",
                "stage1_receipt.json",
            ],
        },
    }
    if (
        plan.get("schema") != "cfeg.reliability-candidate-plan.v1"
        or plan.get("status") != "frozen_for_synthetic_engineering"
        or plan.get("decision_id") != "DEC-20260903-006"
        or plan.get("approved_by") != "active_workspace_owner"
        or plan.get("authority_basis")
        != "explicit_user_approval_in_active_codex_session"
        or not str(plan.get("approval_recorded_at_utc") or "").endswith("Z")
        or plan.get("candidate_id") != protocol.get("candidate_id")
        or (plan.get("predecessor") or {}).get("retirement_receipt")
        != PHYSICAL_RETIREMENT_RECEIPT
        or (plan.get("predecessor") or {}).get("retirement_receipt_sha256")
        != _sha256_file(retirement_path)
        or (plan.get("data_access") or {}).get("allowed_dataset_revisions")
        != {"synthetic_quality": "synthetic_quality_v1"}
        or any(
            (plan.get("data_access") or {}).get(field) is not False
            for field in ("wearable_v3_allowed", "s1_s3_allowed", "s4_s102_allowed")
        )
        or (plan.get("stage0") or {}).get("plan") != RELIABILITY_STAGE0_PLAN
        or (plan.get("stage0") or {}).get("plan_sha256")
        != _sha256_file(_repository_root() / RELIABILITY_STAGE0_PLAN)
        or (plan.get("stage0") or {}).get("required_status_before_stage1") != "passed"
        or (plan.get("stage0") or {}).get("simulation_truth_model_input") is not False
        or (plan.get("stage0") or {}).get("outcome_publication")
        != "one_shot_fresh_atomic_no_overwrite_or_supersede"
        or plan.get("stage1") != expected_stage1
        or plan.get("model_contract") != expected_model
        or any(
            (plan.get("claims") or {}).get(field) is not False
            for field in (
                "human_eeg_claim_allowed",
                "population_inference_allowed",
                "confirmatory_execution_authorized",
                "architecture_superiority_claim_allowed",
            )
        )
    ):
        raise ValueError("Reliability candidate plan is incomplete, unsafe, or stale.")
    stage1 = plan["stage1"]
    data = base.get("data", {})
    train = base.get("train", {})
    frozen_recipe = {
        "seed": base.get("seed"),
        "split_seed": data.get("split_seed"),
        "split": data.get("split"),
        "validation_subject_fraction": data.get("val_ratio"),
        "epochs": train.get("epochs"),
        "checkpoint_selection": train.get("checkpoint_selection"),
    }
    expected_recipe = {
        key: stage1[key]
        for key in (
            "seed",
            "split_seed",
            "split",
            "validation_subject_fraction",
            "epochs",
            "checkpoint_selection",
        )
    }
    if frozen_recipe != expected_recipe:
        raise ValueError("Reliability base recipe differs from the frozen candidate plan.")
    _canonical_project_path(stage1["base_config"], RELIABILITY_BASE_CONFIG)
    _canonical_project_path(stage1["suite"], RELIABILITY_SUITE_CONFIG)
    if retirement["status"] != "retired_no_go":  # pragma: no cover - defensive
        raise ValueError("Physical predecessor is not retired.")
    if protocol.get("reliability_query_feature_schema") != (
        "query_window_reliability_features_v1"
    ):
        raise ValueError("Reliability query-feature schema is missing or stale.")
    return plan


def validate_reliability_stage0_gate(base: dict, *, require_receipt: bool) -> str | None:
    """Require a source- and asset-bound passing Stage-0 receipt before outcome runs."""

    candidate = validate_reliability_candidate_bindings(base)
    protocol = base.get("protocol", {})
    receipt_value = protocol.get("stage0_receipt")
    if receipt_value != "outputs/engineering/reliability-spatial-v1/stage0/receipt.json":
        raise ValueError("Reliability Stage-0 receipt path is noncanonical.")
    if not require_receipt:
        # Outcome-free graph/contract dry-runs deliberately do not depend on, or read,
        # a prior Stage-0 outcome.  This also keeps them available after a failed gate.
        return None
    receipt_candidate = _repository_root() / str(receipt_value)
    if receipt_candidate.is_symlink() or (
        receipt_candidate.exists()
        and receipt_candidate.resolve() != receipt_candidate.absolute()
    ):
        raise ValueError("Reliability Stage-0 receipt cannot traverse a symlink.")
    receipt_path = receipt_candidate.resolve()
    if not receipt_path.is_file():
        if require_receipt:
            raise ValueError("A passing Stage-0 receipt is required before Stage-1 training.")
        return None
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    stage0_path = _repository_root() / RELIABILITY_STAGE0_PLAN
    stage0_plan = load_config(stage0_path, strict_env=False)
    required_false = (
        "population_inference_allowed",
        "confirmatory_execution_authorized",
        "wearable_s1_s3_reuse_allowed",
        "decoder_checkpoint_reuse_allowed",
    )
    if (
        receipt.get("schema") != "cfeg.synthetic-reliability-stage0-receipt.v1"
        or receipt.get("candidate_id") != protocol.get("candidate_id")
        or receipt.get("status") != "passed"
        or receipt.get("scope") != "synthetic_engineering_only"
        or receipt.get("primary_target") != "realized_signal_fraction"
        or receipt.get("plan_file_sha256") != _sha256_file(stage0_path)
        or receipt.get("plan_sha256") != _sha256_json(stage0_plan)
        or not (receipt.get("integrity") or {}).get("all_checks_passed")
        or any(receipt.get(field) is not False for field in required_false)
        or (candidate.get("stage0") or {}).get("required_status_before_stage1")
        != receipt.get("status")
    ):
        raise ValueError("Reliability Stage-0 receipt failed its frozen gate contract.")
    if receipt.get("implementation_contract_sha256") != implementation_contract_sha256():
        raise ValueError(
            "Reliability Stage-0 receipt was produced from another implementation contract."
        )
    processed_dirs = base.get("data", {}).get("processed_dirs") or []
    if len(processed_dirs) != 1:
        raise ValueError("Reliability Stage-1 requires exactly one processed asset root.")
    root = (_repository_root() / str(processed_dirs[0])).resolve()
    expected_assets = {
        name: _sha256_file(root / name)
        for name in SYNTHETIC_QUALITY_ASSET_FILES
    }
    if receipt.get("asset_fingerprints") != expected_assets:
        raise ValueError("Reliability Stage-0 receipt targets another processed asset.")
    artifacts = receipt.get("artifacts")
    if not isinstance(artifacts, Mapping) or set(artifacts) != STAGE0_ARTIFACT_FILES:
        raise ValueError("Reliability Stage-0 artifact set is incomplete or unexpected.")
    observed_artifact_fingerprints: dict[str, str] = {}
    for name, contract in artifacts.items():
        if Path(name).name != name:
            raise ValueError("Reliability Stage-0 artifact paths must be simple filenames.")
        artifact = receipt_path.parent / name
        if (
            not isinstance(contract, Mapping)
            or not artifact.is_file()
            or artifact.is_symlink()
            or artifact.resolve() != artifact.absolute()
            or contract.get("sha256") != _sha256_file(artifact)
            or contract.get("size_bytes") != artifact.stat().st_size
        ):
            raise ValueError("Reliability Stage-0 output artifact is missing or stale.")
        observed_artifact_fingerprints[name] = str(contract["sha256"])

    receipt_sha256 = _sha256_file(receipt_path)
    cache_key = (
        receipt_sha256,
        implementation_contract_sha256(),
        _sha256_json(expected_assets),
        _sha256_json(observed_artifact_fingerprints),
    )
    if cache_key not in _STAGE0_VERIFICATION_CACHE:
        from cfeg.analysis.synthetic_reliability_stage0 import (
            run_stage0,
            validate_stage0_output_tree,
        )

        recomputed = run_stage0(root, stage0_plan, plan_path=stage0_path)
        validate_stage0_output_tree(recomputed, receipt_path.parent)
        _STAGE0_VERIFICATION_CACHE.add(cache_key)
    return receipt_sha256


def validate_reliability_training_preflight(
    cfg: dict, *, dry_run: bool
) -> str | None:
    """Fail closed on direct or drifted reliability-spatial-v1 training routes."""

    architecture = (
        cfg.get("model", {}).get("conditioning", {}).get("architecture")
    )
    if architecture != "reliability_spatial_v1":
        return None

    _validate_reliability_model_contract(cfg)
    validate_reliability_candidate_bindings(cfg)
    protocol = cfg.get("protocol", {})
    role = protocol.get("reliability_family_role")
    if role not in RELIABILITY_ROLES:
        raise ValueError(
            "reliability_spatial_v1 training must run through the canonical four-arm "
            "orchestrator with an explicit family role."
        )
    observed_treatment = (
        str(
            cfg["model"]["conditioning"]["spatial_reliability"]["query_qc_mode"]
        ),
        str(cfg["model"]["condition_encoder"]["external_metadata_mode"]),
    )
    if observed_treatment != RELIABILITY_TREATMENTS[str(role)]:
        raise ValueError("Reliability runtime role and treatment access do not match.")

    canonical_base = load_config(
        _repository_root() / RELIABILITY_BASE_CONFIG, strict_env=False
    )
    canonical_suite = load_config(
        _repository_root() / RELIABILITY_SUITE_CONFIG, strict_env=False
    )
    expected_family_sha256 = validate_reliability_family(
        canonical_base, canonical_suite
    )
    if protocol.get("reliability_mechanism_family_sha256") != expected_family_sha256:
        raise ValueError("Reliability runtime lacks the canonical mechanism-family binding.")
    expected_cfg = merge_overrides(
        canonical_base,
        [
            f"{key}={value!r}"
            for key, value in canonical_suite["variants"][str(role)].items()
        ],
    )
    if _runtime_recipe_sha256(cfg) != _runtime_recipe_sha256(expected_cfg):
        raise ValueError(
            "Reliability runtime recipe differs from the frozen canonical arm recipe."
        )

    receipt_sha256 = validate_reliability_stage0_gate(
        cfg, require_receipt=not dry_run
    )
    if protocol.get("reliability_stage0_receipt_sha256") != receipt_sha256:
        raise ValueError("Reliability runtime Stage-0 receipt binding is missing or stale.")
    return receipt_sha256


def validate_reliability_runtime_family(
    rows: list[dict], family: Mapping | None
) -> str:
    if not family:
        return "not_configured"
    observed = {
        row.get("variant"): row
        for row in rows
        if row.get("variant") in RELIABILITY_ROLES
    }
    if set(observed) != set(RELIABILITY_ROLES):
        return "not_run_together"
    if any(
        observed[role].get("status") not in {"dry_run", "completed"}
        for role in RELIABILITY_ROLES
    ):
        return "incomplete"
    treatment_errors = {
        role: {
            "role": observed[role].get("reliability_family_role"),
            "treatment": (
                observed[role].get("query_qc_mode"),
                observed[role].get("external_metadata_mode"),
            ),
        }
        for role in RELIABILITY_ROLES
        if observed[role].get("reliability_family_role") != role
        or (
            observed[role].get("query_qc_mode"),
            observed[role].get("external_metadata_mode"),
        )
        != RELIABILITY_TREATMENTS[role]
    }
    if treatment_errors:
        raise ValueError(
            f"Reliability runtime treatment/role contract is invalid: {treatment_errors}."
        )
    if any(observed[role].get("status") == "completed" for role in RELIABILITY_ROLES) and any(
        not observed[role].get("reliability_stage0_receipt_sha256")
        for role in RELIABILITY_ROLES
    ):
        raise ValueError("Completed reliability runs require a non-null Stage-0 receipt hash.")
    equality_fields = (
        "reliability_mechanism_family_sha256",
        "candidate_id",
        "candidate_plan_sha256",
        "predecessor_retirement_receipt_sha256",
        "reliability_query_feature_schema",
        "reliability_stage0_receipt_sha256",
        "seed",
        "split_seed",
        "n_folds",
        "fold_index",
        "total_parameters",
        "trainable_parameters",
        "parameter_schema_sha256",
        "initial_trainable_state_sha256",
        "split_assignment_sha256",
        "vocabulary_sha256",
        "asset_provenance_sha256",
        "source_commit_sha",
        "source_dirty",
        "source_tree_sha256",
        "environment_sha256",
        "execution_phase",
        "analysis_plan_status",
        "cohort_role",
        "loader_settings_sha256",
        "development_control_sha256",
        "runtime_metrics_schema",
        "resolved_device",
    )
    nullable_equality_fields = {
        "analysis_plan_status",
        "n_folds",
        "fold_index",
        "reliability_stage0_receipt_sha256",
    }
    missing = {
        role: [
            field
            for field in equality_fields
            if observed[role].get(field) is None
            and field not in nullable_equality_fields
        ]
        for role in RELIABILITY_ROLES
    }
    missing = {role: fields for role, fields in missing.items() if fields}
    if missing:
        raise ValueError(f"Reliability runtime contract is incomplete: {missing}.")
    reference = RELIABILITY_ROLES[0]
    mismatched = [
        field
        for field in equality_fields
        if any(
            observed[role][field] != observed[reference][field]
            for role in RELIABILITY_ROLES[1:]
        )
    ]
    if mismatched:
        raise ValueError(
            "Reliability runtime contracts differ outside treatment access: "
            f"{mismatched}."
        )
    return "verified_equal"


def _validate_reliability_model_contract(cfg: dict) -> None:
    protocol = cfg.get("protocol", {})
    model = cfg.get("model", {})
    conditioning = model.get("conditioning", {})
    spatial = conditioning.get("spatial_reliability", {})
    encoder = model.get("condition_encoder", {})
    if protocol.get("metadata_contract_version") != "0.4-dev":
        raise ValueError("Reliability family requires metadata contract 0.4-dev.")
    if protocol.get("reliability_query_feature_schema") != (
        "query_window_reliability_features_v1"
    ):
        raise ValueError("Reliability family requires its exact Q feature schema.")
    expected = {
        "architecture": "reliability_spatial_v1",
        "backbone": "spectral_transformer",
        "placement": "pre_backbone_waveform",
        "operator_family": "diagonal_low_rank",
        "n_prompt_tokens": 0,
        "fields": ["electrode_type"],
        "include_continuous": True,
        "include_channels": False,
        "force_missing": False,
        "adapter_enabled": False,
        "latent_enabled": False,
        "metadata_residual_enabled": True,
    }
    observed = {
        "architecture": conditioning.get("architecture"),
        "backbone": model.get("backbone", {}).get("name"),
        "placement": spatial.get("placement"),
        "operator_family": spatial.get("operator_family"),
        "n_prompt_tokens": int(encoder.get("n_prompt_tokens", -1)),
        "fields": list(encoder.get("fields") or []),
        "include_continuous": bool(encoder.get("include_continuous", False)),
        "include_channels": bool(encoder.get("include_channels", True)),
        "force_missing": bool(encoder.get("force_missing", False)),
        "adapter_enabled": bool(model.get("adapter", {}).get("enabled", True)),
        "latent_enabled": bool(model.get("latent", {}).get("enabled", True)),
        "metadata_residual_enabled": bool(spatial.get("metadata_residual_enabled", False)),
    }
    if observed != expected:
        raise ValueError(
            f"Reliability model contract mismatch: expected={expected}, observed={observed}."
        )
    max_norm = float(spatial.get("max_operator_norm", -1.0))
    query_norm = float(spatial.get("query_operator_norm", -1.0))
    metadata_norm = float(spatial.get("metadata_operator_norm", -1.0))
    alpha_limit = float(spatial.get("metadata_alpha_limit", -1.0))
    if (
        not 0.0 < max_norm < 1.0
        or query_norm <= 0.0
        or metadata_norm <= 0.0
        or alpha_limit <= 0.0
        or query_norm + alpha_limit * metadata_norm > max_norm + 1e-12
        or int(spatial.get("operator_rank", 0)) < 1
    ):
        raise ValueError("Reliability operator norm/rank contract is invalid.")


def _validate_physical_retirement_receipt(path: Path) -> dict:
    canonical = (_repository_root() / PHYSICAL_RETIREMENT_RECEIPT).resolve()
    if path.resolve() != canonical:
        raise ValueError("Physical predecessor-retirement receipt is noncanonical.")
    return validate_physical_candidate_retirement()


def _runtime_recipe_sha256(cfg: dict) -> str:
    """Normalize only orchestrator bookkeeping, never scientific recipe fields."""

    normalized = copy.deepcopy(cfg)
    normalized.pop("runtime_contract", None)
    normalized.setdefault("augment", {}).pop("channel_subset_ids", None)
    normalized["run_name"] = "<orchestrated-role>"
    normalized["output_dir"] = "<orchestrated-output>"
    protocol = normalized.setdefault("protocol", {})
    for field in (
        "reliability_mechanism_family_sha256",
        "reliability_family_role",
        "reliability_stage0_receipt_sha256",
    ):
        protocol.pop(field, None)
    tags = normalized.setdefault("tracking", {}).setdefault("wandb", {}).get("tags")
    if isinstance(tags, list):
        normalized["tracking"]["wandb"]["tags"] = sorted(
            value
            for value in tags
            if value not in {"ablation", *RELIABILITY_ROLES}
        )
    return _sha256_json(normalized)


def _candidate_base_contract_sha256(cfg: dict) -> str:
    """Hash the complete base recipe without a candidate-plan self-reference."""

    normalized = copy.deepcopy(cfg)
    protocol = normalized.setdefault("protocol", {})
    protocol["candidate_plan_sha256"] = "<candidate-plan-sha256>"
    return _sha256_json(normalized)


def _repository_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _canonical_project_path(value: object, expected: str) -> Path:
    if value != expected:
        raise ValueError(f"Expected canonical project path {expected!r}, got {value!r}.")
    candidate = _repository_root() / expected
    path = candidate.resolve()
    if (
        not candidate.is_file()
        or candidate.is_symlink()
        or path != candidate.absolute()
    ):
        raise ValueError(f"Required reliability contract file is missing or unsafe: {expected}.")
    return path


def _sha256_json(value) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
