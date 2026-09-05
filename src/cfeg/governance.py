from __future__ import annotations

import copy
import datetime as dt
import hashlib
import json
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

from cfeg.data.splits import SplitIndices
from cfeg.metadata_calibration_contract import data_config_targets_wearable_v3


class GovernanceError(RuntimeError):
    """Raised before governed data can be opened under an invalid research policy."""


WEARABLE_V3_DEVELOPMENT_SUBJECTS = ("sub001", "sub002", "sub003")
WEARABLE_V3_DECISION_ID = "DEC-20260830-001"
WEARABLE_V3_ALLOCATION_DECISION_ID = "DEC-20260830-002"
WEARABLE_V3_ROLE_CONFIG = "configs/governance/wearable_cohort_roles.yaml"
WEARABLE_V3_ANALYSIS_PLAN = "configs/analysis/wearable_primary.yaml"
WEARABLE_V3_SIMULATION_PLAN = "configs/analysis/wearable_lockbox_inference_simulation.yaml"
WEARABLE_V3_EXECUTION_MANIFEST = "outputs/confirmatory-primary/execution_manifest.json"
WEARABLE_V3_EXECUTION_OUTPUT_ROOT = "outputs/confirmatory-primary/runs"
WEARABLE_V3_PHYSICAL_GRID_MANIFEST = (
    "outputs/development-loso/physical-mechanism-v2/grid_manifest.json"
)
WEARABLE_V3_PHYSICAL_GRID_ROOT = "outputs/development-loso/physical-mechanism-v2"
WEARABLE_V3_PHYSICAL_BASE_CONFIG = (
    "configs/train/wearable_physical_development_loso.yaml"
)
WEARABLE_V3_PHYSICAL_MECHANISM_CONFIG = "configs/train/wearable_physical_mechanism.yaml"
WEARABLE_V3_PHYSICAL_INTERVENTION_CONFIG = "configs/eval/wearable_physical_mechanism.yaml"
WEARABLE_V3_PHYSICAL_DECISION_RECEIPT = (
    "configs/governance/wearable_physical_reveal2_decision.json"
)
WEARABLE_V3_PHYSICAL_FREEZE_TAG = "physical-reveal2-freeze-20260901-r1"
WEARABLE_V3_DEVELOPMENT_REVEAL_LEDGER = "outputs/development-loso/reveal_ledger.json"
WEARABLE_V3_PHYSICAL_REVEALED_SUMMARY = (
    "outputs/development-loso/physical-mechanism-v2/reveal-bundle/analysis/"
    "revealed_summary.json"
)
WEARABLE_V3_PHYSICAL_FINAL_RECEIPT = (
    "outputs/development-loso/physical-mechanism-v2/reveal_receipt.json"
)
WEARABLE_V3_FIRST_REVEAL_LEDGER = "configs/governance/wearable_s1_s3_reveal_ledger.json"
WEARABLE_V3_PHYSICAL_RETIREMENT = "configs/governance/wearable_s1_s3_retirement.json"
PHYSICAL_MECHANISM_ROLES = (
    "A0_eeg_only",
    "A2_structured_condition_prompt",
    "M1_global_only",
    "M2_channel_only",
    "M3_full_shuffle_train",
    "M4_metadata_only",
)


@dataclass(frozen=True)
class ResearchAccess:
    governed: bool
    execution_phase: str
    cohort_role: str
    plan_path: str | None
    plan_status: str | None
    plan_sha256: str | None
    cohort_roles_path: str | None
    cohort_roles_sha256: str | None
    cohort_decision_id: str | None
    role_config: dict
    allow_outer_test_during_training: bool

    def contract(self) -> dict:
        value = asdict(self)
        value.pop("role_config", None)
        return value


@dataclass(frozen=True)
class CohortBinding:
    global_indices: np.ndarray
    selected_subject_ids: tuple[str, ...]
    excluded_subject_ids: tuple[str, ...]
    expected_n_subjects: int
    expected_n_samples: int
    cohort_sha256: str

    def contract(self) -> dict:
        return {
            "selected_subject_ids": list(self.selected_subject_ids),
            "excluded_subject_ids": list(self.excluded_subject_ids),
            "expected_n_subjects": self.expected_n_subjects,
            "expected_n_samples": self.expected_n_samples,
            "cohort_sha256": self.cohort_sha256,
        }


def resolve_research_access(cfg: dict) -> ResearchAccess:
    protocol = cfg.get("protocol", {})
    evaluation = cfg.get("evaluation", {})
    requires_governance = data_config_targets_wearable_v3(cfg.get("data", {}))
    governance_flag = protocol.get("governance_required")
    if requires_governance and governance_flag is not True:
        raise GovernanceError(
            "wearable_v3 is governed by cohort roles; protocol.governance_required=true "
            "cannot be omitted or overridden."
        )
    governed = governance_flag is True
    if not governed:
        return ResearchAccess(
            governed=False,
            execution_phase=str(protocol.get("execution_phase", "legacy")),
            cohort_role=str(cfg.get("data", {}).get("cohort_role", "all")),
            plan_path=None,
            plan_status=None,
            plan_sha256=None,
            cohort_roles_path=None,
            cohort_roles_sha256=None,
            cohort_decision_id=None,
            role_config={},
            allow_outer_test_during_training=bool(evaluation.get("open_test", True)),
        )

    phase = str(protocol.get("execution_phase", ""))
    if phase not in {"development", "confirmatory_training"}:
        raise GovernanceError(
            "Governed training requires protocol.execution_phase=development or "
            "confirmatory_training."
        )
    role = str(cfg.get("data", {}).get("cohort_role", ""))
    expected_role = "development" if phase == "development" else "confirmatory_primary"
    if role != expected_role:
        raise GovernanceError(
            f"Execution phase {phase!r} requires data.cohort_role={expected_role!r}, got {role!r}."
        )
    if bool(evaluation.get("open_test", False)):
        raise GovernanceError(
            "Governed training must keep evaluation.open_test=false. Outer-test prediction is a "
            "separate post-freeze action."
        )

    plan_path = _project_path(protocol.get("analysis_plan"), field="protocol.analysis_plan")
    canonical_plan_path = (
        Path(__file__).resolve().parents[2] / WEARABLE_V3_ANALYSIS_PLAN
    ).resolve()
    if plan_path != canonical_plan_path:
        raise GovernanceError(f"Governed wearable runs require {WEARABLE_V3_ANALYSIS_PLAN}.")
    plan = _load_yaml(plan_path)
    validate_analysis_plan_contract(plan)

    roles_path = _project_path(plan.get("cohort_roles_config"), field="plan.cohort_roles_config")
    roles = _load_yaml(roles_path)
    validate_cohort_roles_contract(roles)
    roles_sha256 = _sha256_file(roles_path)
    if plan.get("cohort_roles_sha256") != roles_sha256:
        raise GovernanceError("Analysis plan does not bind the current cohort-role manifest hash.")
    decision_id = str(roles.get("decision_id", ""))
    if not decision_id or plan.get("cohort_decision_id") != decision_id:
        raise GovernanceError("Analysis plan and cohort-role manifest decision IDs differ.")
    role_config = (roles.get("roles") or {}).get(role)
    if not isinstance(role_config, dict):
        raise GovernanceError(f"Cohort role {role!r} is missing from {roles_path}.")

    if phase == "development":
        planned = {str(value) for value in plan.get("development_subject_ids", [])}
        configured = {str(value) for value in role_config.get("include_subject_ids", [])}
        if not planned or planned != configured:
            raise GovernanceError("Development subjects differ between plan and role manifest.")
    else:
        planned = {str(value) for value in plan.get("primary_excluded_subject_ids", [])}
        configured = {str(value) for value in role_config.get("exclude_subject_ids", [])}
        if planned != configured:
            raise GovernanceError("Primary exclusions differ between plan and role manifest.")
        validate_frozen_analysis_plan(plan)
        _validate_confirmatory_training_config(cfg, plan)

    return ResearchAccess(
        governed=True,
        execution_phase=phase,
        cohort_role=role,
        plan_path=str(plan_path),
        plan_status=str(plan.get("status")),
        plan_sha256=_sha256_file(plan_path),
        cohort_roles_path=str(roles_path),
        cohort_roles_sha256=roles_sha256,
        cohort_decision_id=decision_id,
        role_config=role_config,
        allow_outer_test_during_training=False,
    )


def bind_cohort(manifest: pd.DataFrame, access: ResearchAccess) -> CohortBinding:
    if "subject_id" not in manifest or "sample_id" not in manifest:
        raise GovernanceError("Cohort binding requires subject_id and sample_id columns.")
    subjects = manifest["subject_id"].astype(str)
    all_subjects = set(subjects)
    if access.governed:
        if access.cohort_role == "development":
            included = {str(value) for value in access.role_config.get("include_subject_ids", [])}
            mask = subjects.isin(included).to_numpy()
        elif access.cohort_role == "confirmatory_primary":
            excluded = {str(value) for value in access.role_config.get("exclude_subject_ids", [])}
            mask = ~subjects.isin(excluded).to_numpy()
        else:  # pragma: no cover - resolve_research_access rejects this first
            raise GovernanceError(f"Unknown governed cohort role {access.cohort_role!r}.")
    else:
        mask = np.ones(len(manifest), dtype=bool)

    indices = np.flatnonzero(mask)
    selected = tuple(sorted(set(subjects.iloc[indices])))
    excluded = tuple(sorted(all_subjects - set(selected)))
    expected_subjects = int(access.role_config.get("expected_n_subjects", len(selected)))
    expected_samples = int(access.role_config.get("expected_n_samples", len(indices)))
    if len(selected) != expected_subjects or len(indices) != expected_samples:
        raise GovernanceError(
            f"Cohort role {access.cohort_role!r} expected {expected_subjects} subjects/"
            f"{expected_samples} samples, observed {len(selected)}/{len(indices)}."
        )
    if access.governed and access.cohort_role == "confirmatory_primary":
        plan = _load_yaml(Path(str(access.plan_path)))
        if int(plan.get("confirmatory_cohort_expected_n_subjects", -1)) != len(selected) or int(
            plan.get("confirmatory_cohort_expected_n_samples", -1)
        ) != len(indices):
            raise GovernanceError("Bound confirmatory cohort does not match the analysis plan.")

    payload = {
        "decision_id": access.cohort_decision_id,
        "role": access.cohort_role,
        "role_manifest_sha256": access.cohort_roles_sha256,
        "selected_subject_ids": list(selected),
        "excluded_subject_ids": list(excluded),
        "sample_ids": sorted(manifest.iloc[indices]["sample_id"].astype(str).tolist()),
    }
    return CohortBinding(
        global_indices=indices,
        selected_subject_ids=selected,
        excluded_subject_ids=excluded,
        expected_n_subjects=expected_subjects,
        expected_n_samples=expected_samples,
        cohort_sha256=_sha256_json(payload),
    )


def remap_split_to_global(split: SplitIndices, global_indices: np.ndarray) -> SplitIndices:
    return SplitIndices(
        train=global_indices[np.asarray(split.train, dtype=int)],
        val=global_indices[np.asarray(split.val, dtype=int)],
        test=global_indices[np.asarray(split.test, dtype=int)],
    )


def validate_checkpoint_evaluation_access(
    train_cfg: dict,
    eval_cfg: dict,
    *,
    checkpoint_path: str | Path | None = None,
    access_route: str = "generic_evaluation",
) -> ResearchAccess:
    access = resolve_research_access(train_cfg)
    if not access.governed:
        return access
    data_cfg = eval_cfg.get("data", {})
    split = str(data_cfg.get("split", ""))
    explicit_target = bool(
        eval_cfg.get("test_filter")
        or data_cfg.get("test_filter")
        or eval_cfg.get("test_datasets")
        or data_cfg.get("test_datasets")
    )
    if access.execution_phase == "development":
        if split != "val" or explicit_target:
            raise GovernanceError(
                "Development checkpoints may evaluate only their recorded validation split; "
                "test/all and explicit target filters are sealed."
            )
        # Every governed development checkpoint is outcome-gated.  Never trust a
        # mutable checkpoint/config boolean to decide whether the gate applies.
        _validate_development_grid_authorization(
            train_cfg,
            eval_cfg,
            checkpoint_path=checkpoint_path,
            access_route=access_route,
        )
    elif access.execution_phase == "confirmatory_training":
        if access_route != "prediction_bundle":
            raise GovernanceError(
                "Confirmatory checkpoints are unavailable to the generic evaluation route."
            )
        _validate_confirmatory_evaluation_shape(eval_cfg)
        if (
            split != "test"
            or explicit_target
            or eval_cfg.get("research_action") != "primary_prediction_lockbox_clean"
            or eval_cfg.get("scenario", "clean") != "clean"
        ):
            raise GovernanceError(
                "Confirmatory checkpoints may open only the registered clean outer-test "
                "prediction action after the analysis plan is frozen."
            )
        runtime = train_cfg.get("runtime_contract", {})
        if runtime.get("analysis_plan_sha256") != access.plan_sha256:
            raise GovernanceError("The frozen analysis plan changed after checkpoint creation.")
        if runtime.get("cohort_roles_sha256") != access.cohort_roles_sha256:
            raise GovernanceError("The cohort-role manifest changed after checkpoint creation.")
        current_source = current_source_revision_contract()
        for field in ("source_commit_sha", "source_dirty", "source_tree_sha256"):
            if runtime.get(field) != current_source.get(field):
                raise GovernanceError(
                    "Confirmatory evaluation source differs from the training checkpoint."
                )
        _validate_confirmatory_lockbox_authorization(
            train_cfg,
            eval_cfg,
            access=access,
            checkpoint_path=checkpoint_path,
        )
    return access


def _validate_development_grid_authorization(
    train_cfg: dict,
    eval_cfg: dict,
    *,
    checkpoint_path: str | Path | None,
    access_route: str,
) -> None:
    allowed_routes = {"prediction_bundle", "physical_mechanism_intervention"}
    if access_route not in allowed_routes:
        raise GovernanceError(
            "Outcome-gated physical development checkpoints are unavailable to generic "
            "evaluation; use the canonical grid orchestrator."
        )
    authorization = eval_cfg.get("development_authorization")
    if not isinstance(authorization, dict):
        raise GovernanceError("Physical development prediction requires grid authorization.")
    protocol = train_cfg.get("protocol", {})
    manifest_path, manifest, recorded = _load_physical_development_grid_manifest(
        authorization.get("grid_manifest")
    )
    expected_manifest = _project_path(
        protocol.get("development_grid_manifest"),
        field="protocol.development_grid_manifest",
    )
    if manifest_path != expected_manifest:
        raise GovernanceError("Physical development checkpoint references another grid manifest.")
    job_id = str(protocol.get("development_grid_job_id", ""))
    jobs = [job for job in manifest.get("jobs", []) if str(job.get("job_id")) == job_id]
    if len(jobs) != 1:
        raise GovernanceError("Physical development checkpoint has no unique grid job.")
    job = jobs[0]
    expected_checkpoint = (Path(str(job["output_dir"])) / "final.pt").resolve()
    observed_checkpoint = Path(str(checkpoint_path or "")).expanduser().resolve()
    if observed_checkpoint != expected_checkpoint:
        raise GovernanceError("Physical development checkpoint differs from its grid job.")
    if access_route == "prediction_bundle":
        expected_prediction = Path(str(job["prediction_staging_csv"])).resolve()
        observed_prediction = (
            Path(str(eval_cfg.get("prediction_output_path", ""))).expanduser().resolve()
        )
        if observed_prediction != expected_prediction:
            raise GovernanceError("Physical development output differs from its grid job.")
        if (
            eval_cfg.get("scenario") != "physical_mechanism_fixed_epoch_clean"
            or eval_cfg.get("data", {}).get("split") != "val"
        ):
            raise GovernanceError(
                "Physical development reveal permits only the registered clean val grid."
            )
    else:
        interventions = [
            item for item in manifest.get("interventions") or [] if item.get("job_id") == job_id
        ]
        if job.get("role") != "A2_structured_condition_prompt" or len(interventions) != 1:
            raise GovernanceError("Physical interventions require the registered Full A2 job.")
        intervention = interventions[0]
        normalized_eval = copy.deepcopy(eval_cfg)
        normalized_eval.pop("development_authorization", None)
        if (
            normalized_eval != intervention.get("config")
            or _sha256_json(normalized_eval) != intervention.get("resolved_config_sha256")
            or Path(str(eval_cfg.get("output_csv", ""))).resolve()
            != Path(str(intervention["staging_output_csv"])).resolve()
            or eval_cfg.get("mode") != "robustness"
            or eval_cfg.get("data", {}).get("split") != "val"
        ):
            raise GovernanceError(
                "Physical A2 intervention config differs from the registered scenario matrix."
            )

    _validate_physical_private_staging_authorization(
        manifest, recorded, authorization
    )
    if manifest.get("source_contract") != current_source_revision_contract():
        raise GovernanceError(
            "Physical development evaluation source differs from the grid source."
        )
    reject_retired_physical_candidate_action()


def validate_physical_development_training_authorization(cfg: dict) -> None:
    """Fail closed before an outcome-bearing physical S1--S3 training loader is built."""

    protocol = cfg.get("protocol", {})
    manifest_path, manifest, _ = _load_physical_development_grid_manifest(
        protocol.get("development_grid_manifest")
    )
    job_id = str(protocol.get("development_grid_job_id", ""))
    matches = [job for job in manifest["jobs"] if str(job.get("job_id")) == job_id]
    if len(matches) != 1:
        raise GovernanceError("Physical development config has no unique authorized grid job.")
    job = matches[0]
    planned = copy.deepcopy(cfg)
    planned.pop("runtime_contract", None)
    if planned != job["config"]:
        raise GovernanceError("Physical development training config differs from its grid job.")
    if Path(str(cfg.get("output_dir", ""))).resolve() != Path(str(job["output_dir"])).resolve():
        raise GovernanceError("Physical development output path differs from its grid job.")
    if manifest_path != _project_path(
        protocol.get("development_grid_manifest"),
        field="protocol.development_grid_manifest",
    ):
        raise GovernanceError("Physical development manifest binding is noncanonical.")
    if manifest.get("source_contract") != current_source_revision_contract():
        raise GovernanceError("Physical development training source differs from the grid source.")
    reject_retired_physical_candidate_action()


def _load_physical_freeze_decision(path: Path) -> dict:
    if not path.is_file():
        raise GovernanceError("Physical development decision receipt is missing.")
    try:
        decision = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise GovernanceError("Physical development decision receipt is invalid JSON.") from exc
    scope = decision.get("scope") or {}
    stopping = decision.get("stopping_rules") or {}
    controls = decision.get("control_values") or {}
    paths = decision.get("canonical_paths") or {}
    bound = decision.get("bound_artifacts") or {}
    repository = Path(__file__).resolve().parents[2]
    required_controls = {
        "chance_balanced_accuracy",
        "clean_a2_mean_minimum_delta",
        "clean_a2_subject_harm_margin",
        "shortcut_equivalence_margin",
        "pairing_mechanism_margin",
        "counterfactual_mechanism_margin",
        "wrong_metadata_safety_harm_margin",
        "minimum_bundle_changed_fraction",
        "minimum_condition_flip_fraction",
        "confirmatory_control_policy",
    }
    if (
        decision.get("schema") != "cfeg.physical-development-freeze-decision.v1"
        or decision.get("status") != "approved"
        or scope.get("grid_id") != "wearable-v3-s1-s3-physical-mechanism-v2"
        or int(scope.get("outcome_reveal_index", -1)) != 2
        or int(scope.get("training_job_count", -1)) != 18
        or int(scope.get("intervention_bundle_count", -1)) != 3
        or int(scope.get("fixed_epochs", -1)) != 10
        or tuple(scope.get("roles") or []) != PHYSICAL_MECHANISM_ROLES
        or set(controls) != required_controls
        or controls.get("confirmatory_control_policy") != "development_gate_only"
        or stopping.get("no_partial_reveal") is not True
        or stopping.get("any_gate_failure")
        != "block_confirmatory_and_forbid_s1_s3_retuning"
        or stopping.get("all_gates_pass")
        != "require_separate_owner_review_before_any_confirmatory_freeze"
        or (decision.get("source_freeze") or {}).get("annotated_tag")
        != WEARABLE_V3_PHYSICAL_FREEZE_TAG
        or paths.get("mechanism_config") != WEARABLE_V3_PHYSICAL_MECHANISM_CONFIG
        or paths.get("intervention_config") != WEARABLE_V3_PHYSICAL_INTERVENTION_CONFIG
        or paths.get("historical_reveal_ledger") != WEARABLE_V3_FIRST_REVEAL_LEDGER
        or paths.get("global_reveal_ledger") != WEARABLE_V3_DEVELOPMENT_REVEAL_LEDGER
        or paths.get("output_root") != WEARABLE_V3_PHYSICAL_GRID_ROOT
        or bound.get("mechanism_config_sha256")
        != _sha256_file(repository / WEARABLE_V3_PHYSICAL_MECHANISM_CONFIG)
        or bound.get("intervention_config_sha256")
        != _sha256_file(repository / WEARABLE_V3_PHYSICAL_INTERVENTION_CONFIG)
        or bound.get("historical_reveal_ledger_sha256")
        != _sha256_file(repository / WEARABLE_V3_FIRST_REVEAL_LEDGER)
    ):
        raise GovernanceError("Physical development decision receipt contract is invalid.")
    return decision


def _load_physical_development_grid_manifest(
    value: str | Path | None,
) -> tuple[Path, dict, str]:
    from cfeg.data.preprocess import CanonicalChannelMap
    from cfeg.execution_manifest import resolved_config_sha256, validate_primary_pair_and_hash
    from cfeg.utils.config import load_config, merge_overrides

    path = _project_path(value, field="protocol.development_grid_manifest")
    expected_path = _project_path(
        WEARABLE_V3_PHYSICAL_GRID_MANIFEST,
        field="canonical physical development grid manifest",
    )
    if path != expected_path or not path.is_file():
        raise GovernanceError("Physical development grid manifest path is missing or noncanonical.")
    try:
        stored = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise GovernanceError("Physical development grid manifest is not valid JSON.") from exc
    recorded = stored.pop("grid_content_sha256", None)
    if (
        stored.get("schema") != "cfeg.physical-mechanism-grid.v1"
        or not isinstance(recorded, str)
        or recorded != _sha256_json(stored)
        or stored.get("design_status") != "frozen"
    ):
        raise GovernanceError("Physical development grid manifest is invalid or not frozen.")

    repository = Path(__file__).resolve().parents[2]
    root = (repository / WEARABLE_V3_PHYSICAL_GRID_ROOT).resolve()
    base_path = (repository / WEARABLE_V3_PHYSICAL_BASE_CONFIG).resolve()
    mechanism_path = (repository / WEARABLE_V3_PHYSICAL_MECHANISM_CONFIG).resolve()
    intervention_path = (repository / WEARABLE_V3_PHYSICAL_INTERVENTION_CONFIG).resolve()
    decision_path = (repository / WEARABLE_V3_PHYSICAL_DECISION_RECEIPT).resolve()
    historical_path = (repository / WEARABLE_V3_FIRST_REVEAL_LEDGER).resolve()
    global_path = (repository / WEARABLE_V3_DEVELOPMENT_REVEAL_LEDGER).resolve()
    exact_paths = {
        "canonical_output_root": root,
        "historical_reveal_ledger": historical_path,
        "global_reveal_ledger": global_path,
    }
    if any(Path(str(stored.get(key, ""))).resolve() != expected for key, expected in exact_paths.items()):
        raise GovernanceError("Physical development manifest uses a noncanonical path.")
    bound_files = {
        "base_config": base_path,
        "mechanism_config": mechanism_path,
        "intervention_config": intervention_path,
        "decision_receipt": decision_path,
    }
    for field, file_path in bound_files.items():
        if (
            _project_path(stored.get(field), field=field) != file_path
            or stored.get(f"{field}_sha256") != _sha256_file(file_path)
        ):
            raise GovernanceError(f"Physical development {field} binding is stale.")
    if stored.get("historical_reveal_ledger_sha256") != _sha256_file(historical_path):
        raise GovernanceError("Physical development first-reveal ledger binding is stale.")

    base = load_config(base_path, strict_env=False)
    suite = load_config(mechanism_path, strict_env=False)
    design = base.get("development_grid") or {}
    authorization = design.get("freeze_authorization") or {}
    decision = _load_physical_freeze_decision(decision_path)
    control = decision.get("control_values") or {}
    margin_keys = (
        "clean_a2_mean_minimum_delta",
        "clean_a2_subject_harm_margin",
        "shortcut_equivalence_margin",
        "pairing_mechanism_margin",
        "counterfactual_mechanism_margin",
        "wrong_metadata_safety_harm_margin",
        "minimum_bundle_changed_fraction",
        "minimum_condition_flip_fraction",
        "confirmatory_control_policy",
    )
    if (
        design.get("status") != "frozen"
        or base.get("protocol", {}).get("development_outcome_gate_required") is not True
        or authorization.get("status") != "approved"
        or authorization.get("outcome_reveal_2_approved") is not True
        or any(
            not str(authorization.get(key) or "").strip()
            for key in (
                "decision_id",
                "approved_by",
                "approved_at_utc",
                "authorization_source",
                "decision_receipt",
                "decision_receipt_sha256",
            )
        )
        or _project_path(
            authorization.get("decision_receipt"), field="decision receipt"
        )
        != decision_path
        or authorization.get("decision_receipt_sha256") != _sha256_file(decision_path)
        or decision.get("decision_id") != authorization.get("decision_id")
        or decision.get("approved_by") != authorization.get("approved_by")
        or decision.get("approval_recorded_at_utc")
        != authorization.get("approved_at_utc")
        or decision.get("authority_basis") != authorization.get("authorization_source")
        or any(control.get(key) is None for key in margin_keys)
        or stored.get("freeze_authorization") != authorization
        or stored.get("decision_receipt_sha256") != _sha256_file(decision_path)
        or stored.get("development_control_values") != control
        or stored.get("source_freeze_tag") != WEARABLE_V3_PHYSICAL_FREEZE_TAG
        or stored.get("grid_id") != design.get("grid_id")
        or stored.get("interpretation") != design.get("interpretation")
    ):
        raise GovernanceError("Physical development owner approval/margin freeze is incomplete.")
    try:
        approved_at = dt.datetime.fromisoformat(str(authorization["approved_at_utc"]))
        bounded_margins = {
            key: float(control[key])
            for key in (
                "clean_a2_mean_minimum_delta",
                "clean_a2_subject_harm_margin",
                "shortcut_equivalence_margin",
                "pairing_mechanism_margin",
                "counterfactual_mechanism_margin",
                "wrong_metadata_safety_harm_margin",
            )
        }
        potency_margins = {
            key: float(control[key])
            for key in ("minimum_bundle_changed_fraction", "minimum_condition_flip_fraction")
        }
    except (KeyError, TypeError, ValueError) as exc:
        raise GovernanceError(
            "Physical development approval timestamp/margins are malformed."
        ) from exc
    if approved_at.tzinfo is None or approved_at.utcoffset() != dt.timedelta(0):
        raise GovernanceError("Physical development approval timestamp must be explicitly UTC.")
    for key, margin_value in bounded_margins.items():
        if not 0.0 <= margin_value <= 1.0:
            raise GovernanceError(f"Physical development {key} must be within [0, 1].")
    for key, margin_value in potency_margins.items():
        if not 0.0 < margin_value <= 1.0:
            raise GovernanceError(f"Physical development {key} must be within (0, 1].")
    if control["confirmatory_control_policy"] != "development_gate_only":
        raise GovernanceError(
            "Physical development currently supports only development_gate_only controls."
        )
    source_contract = stored.get("source_contract") or {}
    if (
        source_contract.get("source_dirty") is not False
        or not str(source_contract.get("source_commit_sha") or "")
    ):
        raise GovernanceError("Physical development source contract is not clean.")
    try:
        tag_type = subprocess.run(
            [
                "git",
                "cat-file",
                "-t",
                f"refs/tags/{WEARABLE_V3_PHYSICAL_FREEZE_TAG}",
            ],
            cwd=repository,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        tagged_commit = subprocess.run(
            [
                "git",
                "rev-parse",
                f"refs/tags/{WEARABLE_V3_PHYSICAL_FREEZE_TAG}^{{commit}}",
            ],
            cwd=repository,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except subprocess.CalledProcessError as exc:
        raise GovernanceError("Physical development freeze tag is missing or invalid.") from exc
    if tag_type != "tag" or tagged_commit != source_contract.get("source_commit_sha"):
        raise GovernanceError("Physical development tag does not bind its recorded commit.")
    roles = tuple(design.get("roles") or [])
    if roles != PHYSICAL_MECHANISM_ROLES:
        raise GovernanceError("Physical development design does not declare the exact six roles.")
    variants = suite.get("variants") or {}
    fairness = validate_primary_pair_and_hash(base, variants)
    if fairness != stored.get("primary_fairness_hash"):
        raise GovernanceError("Physical development primary fairness hash is invalid.")

    registry = _load_yaml(repository / "configs/channel_sets.yaml")
    canonical = CanonicalChannelMap.from_yaml(repository / "configs/canonical_channels.yaml")
    expected_jobs: list[dict] = []
    for seed in design.get("seeds") or []:
        for fold in design.get("folds") or []:
            for role in roles:
                cfg = merge_overrides(
                    copy.deepcopy(base),
                    [f"{key}={item!r}" for key, item in variants[role].items()],
                )
                cfg["seed"] = int(seed)
                cfg["data"]["fold_index"] = int(fold)
                cfg["run_name"] = f"physical-mechanism-fold{fold}-{role}"
                cfg["output_dir"] = str(root / "runs" / f"fold{fold}" / role)
                protocol = cfg.setdefault("protocol", {})
                protocol["primary_ablation_role"] = role
                protocol["primary_fairness_hash"] = fairness
                protocol["development_control_family_sha256"] = stored.get(
                    "development_control_family_sha256"
                )
                protocol["physical_mechanism_family_sha256"] = stored.get(
                    "physical_mechanism_family_sha256"
                )
                job_id = f"seed{seed}-fold{fold}-{role}"
                protocol["development_grid_job_id"] = job_id
                protocol["development_grid_manifest"] = str(path)
                names = cfg.setdefault("augment", {}).get("channel_sets") or []
                cfg["augment"]["channel_subset_ids"] = [
                    canonical.get_ids(list(registry[name])) for name in names
                ]
                expected_jobs.append(
                    {
                        "job_id": job_id,
                        "seed": int(seed),
                        "fold_index": int(fold),
                        "role": role,
                        "output_dir": cfg["output_dir"],
                        "prediction_csv": str(
                            root / "reveal-bundle" / "predictions" / f"{job_id}.csv"
                        ),
                        "prediction_staging_csv": str(
                            root / ".reveal-staging" / "predictions" / f"{job_id}.csv"
                        ),
                        "resolved_config_sha256": resolved_config_sha256(cfg),
                        "config": cfg,
                    }
                )
    expected_keys = {(42, fold, role) for fold in range(3) for role in PHYSICAL_MECHANISM_ROLES}
    observed_keys = {
        (int(job["seed"]), int(job["fold_index"]), str(job["role"]))
        for job in stored.get("jobs") or []
    }
    intervention_base = load_config(intervention_path, strict_env=False)
    expected_interventions: list[dict] = []
    for job in expected_jobs:
        if job["role"] != "A2_structured_condition_prompt":
            continue
        fold = int(job["fold_index"])
        evaluation = copy.deepcopy(intervention_base)
        evaluation["seed"] = int(job["seed"])
        evaluation["data"]["processed_dirs"] = list(job["config"]["data"]["processed_dirs"])
        evaluation["data"]["split"] = "val"
        evaluation["output_csv"] = str(
            root / ".reveal-staging" / "interventions" / f"fold{fold}-A2.csv"
        )
        evaluation["save_predictions"] = True
        expected_interventions.append(
            {
                "job_id": job["job_id"],
                "fold_index": fold,
                "checkpoint_path": str(Path(job["output_dir"]) / "final.pt"),
                "output_csv": str(
                    root / "reveal-bundle" / "interventions" / f"fold{fold}-A2.csv"
                ),
                "staging_output_csv": evaluation["output_csv"],
                "resolved_config_sha256": resolved_config_sha256(evaluation),
                "config": evaluation,
            }
        )
    if (
        len(expected_jobs) != 18
        or observed_keys != expected_keys
        or stored.get("expected_job_count") != 18
        or stored.get("jobs") != expected_jobs
        or stored.get("expected_intervention_count") != 3
        or stored.get("interventions") != expected_interventions
        or stored.get("outcome_reveal_index") != 2
    ):
        raise GovernanceError("Physical development manifest differs from the exact 18-job grid.")
    return path, stored, recorded


def _validate_physical_private_staging_authorization(
    manifest: dict,
    manifest_digest: str,
    authorization: dict,
) -> None:
    repository = Path(__file__).resolve().parents[2]
    receipt_path = (repository / WEARABLE_V3_PHYSICAL_DECISION_RECEIPT).resolve()
    if (
        authorization.get("authorization_phase") != "private_complete_bundle_staging"
        or Path(str(authorization.get("decision_receipt", ""))).resolve()
        != receipt_path
        or not receipt_path.is_file()
        or authorization.get("decision_receipt_sha256") != _sha256_file(receipt_path)
        or manifest.get("decision_receipt_sha256") != _sha256_file(receipt_path)
        or manifest.get("development_control_values")
        != (_load_physical_freeze_decision(receipt_path).get("control_values") or {})
        or int(authorization.get("reveal_index", -1))
        != int(manifest.get("outcome_reveal_index", -2))
    ):
        raise GovernanceError(
            "Physical development private-staging authorization is invalid or stale."
        )
    if not isinstance(manifest_digest, str) or len(manifest_digest) != 64:
        raise GovernanceError("Physical development grid digest is invalid.")


def validate_analysis_plan_contract(plan: dict) -> None:
    expected = {
        "schema": "cfeg.confirmatory-plan.v3",
        "dataset_id": "wearable",
        "dataset_revision": "wearable_v3",
        "cohort_roles_config": WEARABLE_V3_ROLE_CONFIG,
        "cohort_decision_id": WEARABLE_V3_DECISION_ID,
        "development_subject_ids": list(WEARABLE_V3_DEVELOPMENT_SUBJECTS),
        "primary_excluded_subject_ids": list(WEARABLE_V3_DEVELOPMENT_SUBJECTS),
        "prediction_schema_version": "cfeg.predictions.v3",
        "scenario": "clean",
        "split_seed": 42,
        "split_algorithm": "confirmatory_lockbox_v1",
        "inner_val_ratio": None,
        "allocation_decision_id": WEARABLE_V3_ALLOCATION_DECISION_ID,
        "allocation_seed": 42,
        "allocation_algorithm": "numpy_default_rng_shuffle_sorted_subject_ids_train_first_v1",
        "execution_manifest_path": WEARABLE_V3_EXECUTION_MANIFEST,
        "execution_output_root": WEARABLE_V3_EXECUTION_OUTPUT_ROOT,
        "fold_indices": [0],
        "optimization_seeds": [42, 43, 44],
        "asset_expected_n_samples": 24_480,
        "asset_expected_n_subjects": 102,
        "confirmatory_cohort_expected_n_samples": 23_760,
        "confirmatory_cohort_expected_n_subjects": 99,
        "training_expected_n_samples": 9_360,
        "training_expected_n_subjects": 39,
        "expected_n_samples": 14_400,
        "expected_n_subjects": 60,
        "expected_n_labels": 12,
        "condition_column": "electrode_type",
        "expected_conditions": ["dry", "wet"],
        "seed_reduction": "mean_subject_metric",
        "inference_method": "paired_subject_t_bound_sign_flip_with_bootstrap_sensitivity",
        "inference_seed": 20_260_829,
        "n_resamples": 10_000,
        "training_randomness_target": "conditional_fixed_seed_ensemble_42_43_44",
        "confirmatory_actions": ["primary_prediction_lockbox_clean"],
    }
    mismatched = {
        key: {"expected": value, "observed": plan.get(key)}
        for key, value in expected.items()
        if plan.get(key) != value
    }
    control = plan.get("development_control_protocol") or {}
    expected_control = {
        "schema": "cfeg.development-control-decision.v3",
        "control_seed": 42,
        "chance_balanced_accuracy": 1.0 / 12.0,
        "training_controls": [
            "metadata_only",
            "within_class_shuffle",
        ],
        "invalid_assays": ["missingness_only"],
        "evaluation_only_controls": [
            "within_class_shuffle",
            "counterfactual_wet_dry",
        ],
        "shuffle_scope": "within_split_dataset_block_derangement_label_aligned",
        "counterfactual_pair_key": ("dataset_subject_label_block_window_opposite_electrode"),
        "missingness_single_pattern_policy": "invalid_assay",
    }
    mismatched.update(
        {
            f"development_control_protocol.{key}": {
                "expected": value,
                "observed": control.get(key),
            }
            for key, value in expected_control.items()
            if control.get(key) != value
        }
    )
    expected_model_contract = {
        "conditioning_architecture": "physical_hybrid_v1",
        "backbone_name": "spectral_transformer",
        "spectral_frequency_range_hz": [6.0, 60.0],
        "conditioning_hidden_dim": 64,
        "common_query_path": "zero_init_bounded_residual_film",
        "external_global_fields": [
            "electrode_type",
            "impedance_mean_kohm",
            "impedance_max_kohm",
        ],
        "external_global_path": "zero_init_bounded_residual_film",
        "channel_quality_fields": [
            "impedance_kohm_by_channel",
            "impedance_availability_by_channel",
        ],
        "channel_quality_path": ("shared_nonmonotonic_bounded_gain_pre_channel_embedding"),
        "impedance_transform": "log1p_kohm_over_log1p_100_clip_0_2_v1",
        "null_semantics": "algebraic_identity_same_checkpoint",
        "max_film_scale_delta": 0.25,
        "max_film_shift": 0.25,
        "max_channel_gain_delta": 0.25,
        "latent_enabled": False,
        "excluded_constant_primary_fields": ["reference", "cap_type"],
        "legacy_prompt_adapter_role": "secondary_architecture_only",
    }
    if plan.get("primary_model_contract") != expected_model_contract:
        mismatched["primary_model_contract"] = {
            "expected": expected_model_contract,
            "observed": plan.get("primary_model_contract"),
        }
    eligible = np.asarray([f"sub{index:03d}" for index in range(4, 103)])
    allocation_rng = np.random.default_rng(42)
    allocation_rng.shuffle(eligible)
    expected_training = sorted(str(value) for value in eligible[:39])
    expected_lockbox = sorted(str(value) for value in eligible[39:])
    for key, expected_ids in (
        ("confirmatory_training_subject_ids", expected_training),
        ("primary_included_subject_ids", expected_lockbox),
    ):
        if plan.get(key) != expected_ids:
            mismatched[key] = {"expected": expected_ids, "observed": plan.get(key)}
    allocation_payload = {
        "allocation_decision_id": plan.get("allocation_decision_id"),
        "allocation_seed": plan.get("allocation_seed"),
        "allocation_algorithm": plan.get("allocation_algorithm"),
        "confirmatory_training_subject_ids": plan.get("confirmatory_training_subject_ids"),
        "primary_included_subject_ids": plan.get("primary_included_subject_ids"),
    }
    expected_allocation_sha256 = _sha256_json(allocation_payload)
    if plan.get("allocation_sha256") != expected_allocation_sha256:
        mismatched["allocation_sha256"] = {
            "expected": expected_allocation_sha256,
            "observed": plan.get("allocation_sha256"),
        }
    missing_outcomes = plan.get("missing_outcomes") or {}
    expected_missing_outcomes = {
        "primary_policy": "require_exact_complete_grid",
        "partial_seed_policy": "rerun_or_primary_invalid",
        "partial_subject_policy": "primary_invalid",
        "metadata_input_missingness": "retain_row_use_missing_token",
        "sensitivity_policy": "exploratory_bounds_only",
    }
    if missing_outcomes != expected_missing_outcomes:
        mismatched["missing_outcomes"] = {
            "expected": expected_missing_outcomes,
            "observed": missing_outcomes,
        }
    simulation = plan.get("target_free_simulation") or {}
    if simulation.get("plan") != WEARABLE_V3_SIMULATION_PLAN:
        mismatched["target_free_simulation.plan"] = {
            "expected": WEARABLE_V3_SIMULATION_PLAN,
            "observed": simulation.get("plan"),
        }
    expected_multiplicity = {
        "method": "single_primary_contrast_no_adjustment_conditions_descriptive",
        "family": ["overall_a2_minus_a0_balanced_accuracy"],
    }
    if plan.get("multiplicity") != expected_multiplicity:
        mismatched["multiplicity"] = {
            "expected": expected_multiplicity,
            "observed": plan.get("multiplicity"),
        }
    if mismatched:
        raise GovernanceError(f"Analysis plan violates the wearable protocol: {mismatched}.")
    if plan.get("status") not in {"dev_not_frozen", "frozen"}:
        raise GovernanceError("Analysis plan status must be dev_not_frozen or frozen.")
    roles_hash = str(plan.get("cohort_roles_sha256", ""))
    if len(roles_hash) != 64 or any(char not in "0123456789abcdef" for char in roles_hash):
        raise GovernanceError("Analysis plan has no valid cohort_roles_sha256 binding.")


def validate_cohort_roles_contract(roles: dict) -> None:
    expected_top = {
        "schema": "cfeg.cohort-roles.v1",
        "status": "frozen",
        "decision_id": WEARABLE_V3_DECISION_ID,
        "dataset_id": "wearable",
        "dataset_revision": "wearable_v3",
        "asset_contract": {"expected_n_subjects": 102, "expected_n_samples": 24_480},
    }
    mismatched = {
        key: {"expected": value, "observed": roles.get(key)}
        for key, value in expected_top.items()
        if roles.get(key) != value
    }
    role_map = roles.get("roles") or {}
    development = role_map.get("development") or {}
    primary = role_map.get("confirmatory_primary") or {}
    expected_roles = {
        "development.include_subject_ids": (
            development.get("include_subject_ids"),
            list(WEARABLE_V3_DEVELOPMENT_SUBJECTS),
        ),
        "development.expected_n_subjects": (development.get("expected_n_subjects"), 3),
        "development.expected_n_samples": (development.get("expected_n_samples"), 720),
        "development.performance_access": (development.get("performance_access"), "allowed"),
        "confirmatory_primary.exclude_subject_ids": (
            primary.get("exclude_subject_ids"),
            list(WEARABLE_V3_DEVELOPMENT_SUBJECTS),
        ),
        "confirmatory_primary.expected_n_subjects": (primary.get("expected_n_subjects"), 99),
        "confirmatory_primary.expected_n_samples": (primary.get("expected_n_samples"), 23_760),
        "confirmatory_primary.performance_access": (
            primary.get("performance_access"),
            "sealed_until_analysis_plan_frozen",
        ),
    }
    mismatched.update(
        {
            key: {"expected": expected, "observed": observed}
            for key, (observed, expected) in expected_roles.items()
            if observed != expected
        }
    )
    if mismatched:
        raise GovernanceError(f"Cohort-role manifest violates the frozen decision: {mismatched}.")


def validate_frozen_analysis_plan(plan: dict) -> None:
    validate_analysis_plan_contract(plan)
    if plan.get("status") != "frozen":
        raise GovernanceError(
            "Confirmatory training/evaluation is sealed until plan.status=frozen."
        )
    required_values = {
        "success_threshold": plan.get("success_threshold"),
        "sesoi_balanced_accuracy": plan.get("sesoi_balanced_accuracy"),
        "inference.alpha": (plan.get("inference") or {}).get("alpha"),
        "inference.alternative": (plan.get("inference") or {}).get("alternative"),
        "multiplicity.method": (plan.get("multiplicity") or {}).get("method"),
        "multiplicity.family": (plan.get("multiplicity") or {}).get("family"),
        "primary_fairness_hash": plan.get("primary_fairness_hash"),
        "source_lock.freeze_tag": (plan.get("source_lock") or {}).get("freeze_tag"),
        "source_lock.implementation_contract_sha256": (plan.get("source_lock") or {}).get(
            "implementation_contract_sha256"
        ),
        "target_free_simulation.plan_sha256": (plan.get("target_free_simulation") or {}).get(
            "plan_sha256"
        ),
        "target_free_simulation.accepted_receipt": (plan.get("target_free_simulation") or {}).get(
            "accepted_receipt"
        ),
        "target_free_simulation.accepted_receipt_sha256": (
            plan.get("target_free_simulation") or {}
        ).get("accepted_receipt_sha256"),
        "development_control_protocol.clean_a2_mean_minimum_delta": (
            plan.get("development_control_protocol") or {}
        ).get("clean_a2_mean_minimum_delta"),
        "development_control_protocol.clean_a2_subject_harm_margin": (
            plan.get("development_control_protocol") or {}
        ).get("clean_a2_subject_harm_margin"),
        "development_control_protocol.shortcut_equivalence_margin": (
            plan.get("development_control_protocol") or {}
        ).get("shortcut_equivalence_margin"),
        "development_control_protocol.pairing_mechanism_margin": (
            plan.get("development_control_protocol") or {}
        ).get("pairing_mechanism_margin"),
        "development_control_protocol.counterfactual_mechanism_margin": (
            plan.get("development_control_protocol") or {}
        ).get("counterfactual_mechanism_margin"),
        "development_control_protocol.wrong_metadata_safety_harm_margin": (
            plan.get("development_control_protocol") or {}
        ).get("wrong_metadata_safety_harm_margin"),
        "development_control_protocol.minimum_bundle_changed_fraction": (
            plan.get("development_control_protocol") or {}
        ).get("minimum_bundle_changed_fraction"),
        "development_control_protocol.minimum_condition_flip_fraction": (
            plan.get("development_control_protocol") or {}
        ).get("minimum_condition_flip_fraction"),
        "development_control_protocol.confirmatory_control_policy": (
            plan.get("development_control_protocol") or {}
        ).get("confirmatory_control_policy"),
    }
    missing = sorted(key for key, value in required_values.items() if value is None or value == [])
    if missing:
        raise GovernanceError(f"Frozen analysis plan is incomplete: {missing}.")
    if not 0.0 <= float(plan["success_threshold"]) <= 1.0:
        raise GovernanceError("success_threshold must be within [0, 1].")
    if not 0.0 < float(plan["sesoi_balanced_accuracy"]) <= 1.0:
        raise GovernanceError("sesoi_balanced_accuracy must be within (0, 1].")
    alpha = float(plan["inference"]["alpha"])
    if not 0.0 < alpha < 1.0:
        raise GovernanceError("inference.alpha must be within (0, 1).")
    if plan["inference"]["alternative"] not in {"greater", "two-sided"}:
        raise GovernanceError("inference.alternative must be greater or two-sided.")
    control = plan["development_control_protocol"]
    for key in (
        "clean_a2_mean_minimum_delta",
        "clean_a2_subject_harm_margin",
        "shortcut_equivalence_margin",
        "pairing_mechanism_margin",
        "counterfactual_mechanism_margin",
        "wrong_metadata_safety_harm_margin",
    ):
        if not 0.0 <= float(control[key]) <= 1.0:
            raise GovernanceError(f"development_control_protocol.{key} must be within [0, 1].")
    for key in ("minimum_bundle_changed_fraction", "minimum_condition_flip_fraction"):
        if not 0.0 < float(control[key]) <= 1.0:
            raise GovernanceError(f"development_control_protocol.{key} must be within (0, 1].")
    if control["confirmatory_control_policy"] not in {
        "development_gate_only",
        "confirmatory_once",
    }:
        raise GovernanceError(
            "development_control_protocol.confirmatory_control_policy must be "
            "development_gate_only or confirmatory_once."
        )
    if control["confirmatory_control_policy"] != "development_gate_only":
        raise GovernanceError(
            "confirmatory_once is not implemented by the sealed confirmatory runner; "
            "freeze with development_gate_only or implement and pre-register that grid."
        )
    _validate_source_lock(plan["source_lock"])
    _validate_target_free_simulation(plan)
    reject_retired_physical_candidate_action()


def validate_physical_candidate_retirement() -> dict:
    """Validate the superseding deny overlay without rewriting historical receipts."""

    repository = Path(__file__).resolve().parents[2]
    path = repository / WEARABLE_V3_PHYSICAL_RETIREMENT
    if not path.is_file() or path.is_symlink():
        raise GovernanceError("Physical candidate-retirement receipt is missing or unsafe.")
    try:
        receipt = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise GovernanceError("Physical candidate-retirement receipt is invalid JSON.") from exc
    retired = receipt.get("retired_candidate") or {}
    budget = receipt.get("outcome_reveal_budget") or {}
    evidence = receipt.get("decision_evidence") or {}
    bound = receipt.get("bound_artifacts") or {}
    expected_failed_gates = [
        "clean_a2_directional_mean",
        "counterfactual_reliance",
        "inference_pairing_shuffle",
        "training_pairing_shuffle",
    ]
    expected_paths = {
        "decision_receipt": WEARABLE_V3_PHYSICAL_DECISION_RECEIPT,
        "global_reveal_ledger": WEARABLE_V3_DEVELOPMENT_REVEAL_LEDGER,
        "revealed_summary": WEARABLE_V3_PHYSICAL_REVEALED_SUMMARY,
        "final_publication_receipt": WEARABLE_V3_PHYSICAL_FINAL_RECEIPT,
    }
    artifact_mismatch = False
    for name, relative in expected_paths.items():
        contract = bound.get(name) or {}
        artifact = repository / relative
        if (
            contract.get("path") != relative
            or not artifact.is_file()
            or artifact.is_symlink()
            or contract.get("sha256") != _sha256_file(artifact)
        ):
            artifact_mismatch = True
    if (
        receipt.get("schema") != "cfeg.candidate-retirement.v1"
        or receipt.get("decision_id") != "DEC-20260903-005"
        or receipt.get("status") != "retired_no_go"
        or receipt.get("approved_by") != "active_workspace_owner"
        or receipt.get("authority_basis")
        != "explicit_user_approval_in_active_codex_session"
        or retired.get("candidate_id") != "physical-hybrid-v1"
        or retired.get("conditioning_architecture") != "physical_hybrid_v1"
        or retired.get("development_cohort") != list(WEARABLE_V3_DEVELOPMENT_SUBJECTS)
        or retired.get("historical_roles_preserved")
        != ["A0_eeg_only", "A2_structured_condition_prompt"]
        or retired.get("confirmatory_training_blocked") is not True
        or retired.get("lockbox_prediction_blocked") is not True
        or budget.get("consumed_indices") != [1, 2]
        or budget.get("further_s1_s3_outcome_reveals_allowed") is not False
        or budget.get("post_outcome_s1_s3_tuning_allowed") is not False
        or evidence.get("assay_valid") is not True
        or evidence.get("diagnostic_status")
        != "diagnostic_no_go_one_or_more_substantive_gates_failed"
        or evidence.get("failed_substantive_gates") != expected_failed_gates
        or evidence.get("confirmatory_execution_authorized") is not False
        or evidence.get("population_inference_allowed") is not False
        or receipt.get("forbidden_actions")
        != [
            "s1_s3_training",
            "s1_s3_prediction",
            "s1_s3_new_metric_computation",
            "s1_s3_model_selection",
            "physical_hybrid_v1_confirmatory_training",
            "physical_hybrid_v1_lockbox_prediction",
        ]
        or receipt.get("allowed_actions")
        != [
            "read_only_historical_status",
            "artifact_integrity_audit",
            "outcome_free_contract_regression",
        ]
        or receipt.get("source_freeze")
        != {
            "commit": "7bb8afc64c02e45beda7e245370563eac0e021e2",
            "annotated_tag": WEARABLE_V3_PHYSICAL_FREEZE_TAG,
        }
        or set(bound)
        != {
            *expected_paths,
            "reveal_event_sha256",
            "bundle_content_sha256",
        }
        or artifact_mismatch
    ):
        raise GovernanceError("Physical candidate-retirement receipt is incomplete or stale.")

    ledger = json.loads(
        (repository / WEARABLE_V3_DEVELOPMENT_REVEAL_LEDGER).read_text(encoding="utf-8")
    )
    entries = ledger.get("entries") or []
    matching_entries = [entry for entry in entries if entry.get("reveal_index") == 2]
    summary = json.loads(
        (repository / WEARABLE_V3_PHYSICAL_REVEALED_SUMMARY).read_text(encoding="utf-8")
    )
    final = json.loads(
        (repository / WEARABLE_V3_PHYSICAL_FINAL_RECEIPT).read_text(encoding="utf-8")
    )
    gate = summary.get("predeclared_gate_evaluation") or {}
    reveal_event = bound.get("reveal_event_sha256")
    bundle_content = bound.get("bundle_content_sha256")
    if (
        ledger.get("schema") != "cfeg.development-reveal-ledger.v1"
        or len(matching_entries) != 1
        or matching_entries[0].get("reveal_event_sha256") != reveal_event
        or matching_entries[0].get("bundle_content_sha256") != bundle_content
        or matching_entries[0].get("decision_receipt_sha256")
        != bound["decision_receipt"]["sha256"]
        or final.get("schema") != "cfeg.physical-mechanism-reveal-receipt.v4"
        or final.get("reveal_index") != 2
        or final.get("reveal_event_sha256") != reveal_event
        or final.get("bundle_content_sha256") != bundle_content
        or final.get("final_publication_state") != "public_bundle_published"
        or final.get("confirmatory_claim_allowed") is not False
        or final.get("population_inference_allowed") is not False
        or gate.get("assay_valid") is not True
        or gate.get("status") != evidence["diagnostic_status"]
        or gate.get("failed_substantive_gates") != expected_failed_gates
        or gate.get("confirmatory_execution_authorized") is not False
        or gate.get("population_inference_allowed") is not False
    ):
        raise GovernanceError("Physical retirement evidence chain is inconsistent or stale.")
    try:
        tag_commit = subprocess.run(
            ["git", "rev-list", "-n", "1", WEARABLE_V3_PHYSICAL_FREEZE_TAG],
            cwd=repository,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        raise GovernanceError("Physical retirement source-freeze tag is unavailable.") from exc
    if tag_commit != receipt["source_freeze"]["commit"]:
        raise GovernanceError("Physical retirement source-freeze tag moved.")
    return receipt


def reject_retired_physical_candidate_action() -> None:
    receipt = validate_physical_candidate_retirement()
    raise GovernanceError(
        "physical_hybrid_v1 and further S1-S3 outcome actions are retired by "
        f"{receipt['decision_id']}; historical read-only status remains allowed."
    )


def current_source_revision_contract() -> dict[str, object]:
    repository = Path(__file__).resolve().parents[2]
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repository,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
        ).stdout.strip()
        diff = subprocess.run(
            ["git", "diff", "--binary", "HEAD", "--", "."],
            cwd=repository,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
        ).stdout
        untracked_output = subprocess.run(
            ["git", "ls-files", "--others", "--exclude-standard", "-z"],
            cwd=repository,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
        ).stdout
    except (OSError, subprocess.CalledProcessError):
        return {"source_commit_sha": None, "source_dirty": None, "source_tree_sha256": None}
    digest = hashlib.sha256()
    digest.update(commit.encode("ascii"))
    digest.update(diff)
    untracked = sorted(
        path
        for path in untracked_output.decode("utf-8", errors="surrogateescape").split("\0")
        if path
    )
    for relative in untracked:
        encoded = relative.encode("utf-8", errors="surrogateescape")
        digest.update(len(encoded).to_bytes(8, "big"))
        digest.update(encoded)
        path = repository / relative
        if path.is_file():
            with path.open("rb") as handle:
                for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                    digest.update(chunk)
    return {
        "source_commit_sha": commit,
        "source_dirty": bool(diff or untracked),
        "source_tree_sha256": digest.hexdigest(),
    }


def implementation_contract_sha256() -> str:
    repository = Path(__file__).resolve().parents[2]
    paths: list[Path] = []
    for directory in ("src", "scripts", "configs", "tests"):
        paths.extend(path for path in (repository / directory).rglob("*") if path.is_file())
    paths.extend(
        repository / name
        for name in ("pyproject.toml", "requirements.txt", "requirements-cuda121.txt")
    )
    excluded = (repository / "configs/analysis/wearable_primary.yaml").resolve()
    digest = hashlib.sha256()
    for path in sorted({path.resolve() for path in paths if path.exists()}):
        if path == excluded or "__pycache__" in path.parts or path.suffix == ".pyc":
            continue
        relative = path.relative_to(repository).as_posix().encode("utf-8")
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    return digest.hexdigest()


def _validate_confirmatory_training_config(cfg: dict, plan: dict) -> None:
    data = cfg.get("data", {})
    protocol = cfg.get("protocol", {})
    expected = {
        "seed": (cfg.get("seed"), plan["optimization_seeds"]),
        "data.split": (data.get("split"), "confirmatory_lockbox"),
        "data.split_seed": (data.get("split_seed"), plan["split_seed"]),
        "data.n_folds": (data.get("n_folds"), len(plan["fold_indices"])),
        "data.fold_index": (data.get("fold_index"), plan["fold_indices"]),
        "data.training_subject_ids": (
            data.get("training_subject_ids"),
            plan["confirmatory_training_subject_ids"],
        ),
        "data.lockbox_subject_ids": (
            data.get("lockbox_subject_ids"),
            plan["primary_included_subject_ids"],
        ),
        "protocol.primary_ablation_role": (
            protocol.get("primary_ablation_role"),
            ["A0_eeg_only", "A2_structured_condition_prompt"],
        ),
        "protocol.primary_fairness_hash": (
            protocol.get("primary_fairness_hash"),
            plan["primary_fairness_hash"],
        ),
    }
    invalid = {}
    for field, (observed, allowed) in expected.items():
        if field in {"data.training_subject_ids", "data.lockbox_subject_ids"}:
            valid = observed == allowed
        elif isinstance(allowed, list):
            valid = observed in allowed
        else:
            valid = observed == allowed
        if not valid:
            invalid[field] = {"expected": allowed, "observed": observed}
    role = protocol.get("primary_ablation_role")
    expected_job_id = f"seed{cfg.get('seed')}-fold{data.get('fold_index')}-{role}"
    if protocol.get("execution_job_id") != expected_job_id:
        invalid["protocol.execution_job_id"] = {
            "expected": expected_job_id,
            "observed": protocol.get("execution_job_id"),
        }
    execution_contract = protocol.get("execution_contract_sha256")
    if not isinstance(execution_contract, str) or len(execution_contract) != 64:
        invalid["protocol.execution_contract_sha256"] = {
            "expected": "64-character execution-contract SHA-256",
            "observed": execution_contract,
        }
    elif any(character not in "0123456789abcdef" for character in execution_contract):
        invalid["protocol.execution_contract_sha256"] = {
            "expected": "lowercase hexadecimal execution-contract SHA-256",
            "observed": execution_contract,
        }
    manifest_value = protocol.get("execution_manifest")
    if not manifest_value:
        invalid["protocol.execution_manifest"] = {
            "expected": "path to the frozen six-job execution manifest",
            "observed": manifest_value,
        }
    if invalid:
        raise GovernanceError(f"Confirmatory run config differs from the frozen plan: {invalid}.")
    _, manifest, _ = _load_bound_execution_manifest(manifest_value, plan=plan)
    job = _manifest_job_for_config(manifest, cfg)
    planned = copy.deepcopy(cfg)
    planned.pop("runtime_contract", None)
    from cfeg.execution_manifest import resolved_config_sha256

    if resolved_config_sha256(planned) != job.get("resolved_config_sha256"):
        raise GovernanceError(
            "Confirmatory training config is not the resolved config in its execution manifest."
        )


def _load_bound_execution_manifest(
    value: str | Path,
    *,
    plan: dict,
) -> tuple[Path, dict, str]:
    from cfeg.execution_manifest import sha256_json, validate_primary_execution_manifest

    path = _project_path(value, field="protocol.execution_manifest")
    expected_path = _project_path(
        plan.get("execution_manifest_path"), field="plan.execution_manifest_path"
    )
    if path != expected_path:
        raise GovernanceError("Confirmatory execution must use the plan's canonical manifest.")
    try:
        stored = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise GovernanceError("Confirmatory execution manifest is not valid JSON.") from exc
    recorded = stored.pop("manifest_content_sha256", None)
    if not isinstance(recorded, str) or recorded != sha256_json(stored):
        raise GovernanceError("Confirmatory execution manifest content digest is invalid.")
    plan_path = (Path(__file__).resolve().parents[2] / WEARABLE_V3_ANALYSIS_PLAN).resolve()
    try:
        validate_primary_execution_manifest(
            stored,
            plan=plan,
            analysis_plan_sha256=_sha256_file(plan_path),
            manifest_path=path,
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise GovernanceError(f"Confirmatory execution manifest is invalid: {exc}") from exc
    if stored.get("plan_status") != "frozen" or stored.get("execution_allowed") is not True:
        raise GovernanceError("Confirmatory execution manifest is not authorized for execution.")
    if stored.get("source_contract") != current_source_revision_contract():
        raise GovernanceError("Confirmatory execution manifest source contract is stale.")
    return path, stored, recorded


def _manifest_job_for_config(manifest: dict, cfg: dict) -> dict:
    protocol = cfg.get("protocol") or {}
    job_id = protocol.get("execution_job_id")
    matches = [job for job in manifest.get("jobs", []) if job.get("job_id") == job_id]
    if len(matches) != 1:
        raise GovernanceError("Confirmatory config does not resolve to one manifest job.")
    job = matches[0]
    if protocol.get("execution_contract_sha256") != manifest.get("execution_contract_sha256"):
        raise GovernanceError("Confirmatory config execution contract differs from the manifest.")
    return job


def _validate_confirmatory_lockbox_authorization(
    train_cfg: dict,
    eval_cfg: dict,
    *,
    access: ResearchAccess,
    checkpoint_path: str | Path | None,
) -> None:
    authorization = eval_cfg.get("confirmatory_authorization")
    if not isinstance(authorization, dict):
        raise GovernanceError(
            "Confirmatory lockbox access requires the six-job orchestrator authorization."
        )
    manifest_value = authorization.get("execution_manifest")
    receipt_value = authorization.get("lockbox_reveal_receipt")
    configured_manifest = (train_cfg.get("protocol") or {}).get("execution_manifest")
    if not manifest_value or not configured_manifest:
        raise GovernanceError("Confirmatory lockbox authorization has no execution manifest.")
    manifest_path, manifest, manifest_digest = _load_bound_execution_manifest(
        manifest_value,
        plan=_load_yaml(Path(str(access.plan_path))),
    )
    if manifest_path != _project_path(configured_manifest, field="protocol.execution_manifest"):
        raise GovernanceError("Checkpoint and authorization reference different manifests.")
    job = _manifest_job_for_config(manifest, train_cfg)
    if (
        checkpoint_path is None
        or Path(checkpoint_path).expanduser().resolve()
        != Path(str(job["checkpoint_path"])).expanduser().resolve()
    ):
        raise GovernanceError("Confirmatory checkpoint path differs from its manifest job.")

    output_value = eval_cfg.get("prediction_output_path")
    final_path = Path(str(job["prediction_csv"])).expanduser().resolve()
    final_dir = final_path.parent
    expected_output_path = (
        final_dir.with_name(f".{final_dir.name}.{manifest['execution_contract_sha256']}.staging")
        / final_path.name
    )
    if not output_value or Path(str(output_value)).expanduser().resolve() != expected_output_path:
        raise GovernanceError(
            "Confirmatory prediction output must be the orchestrator's manifest-bound "
            "hidden staging path."
        )

    expected_receipt_path = manifest_path.parent / "lockbox_reveal_receipt.json"
    if (
        not receipt_value
        or _project_path(receipt_value, field="confirmatory_authorization.lockbox_reveal_receipt")
        != expected_receipt_path.resolve()
    ):
        raise GovernanceError("Confirmatory reveal receipt path is not manifest-bound.")
    try:
        receipt = json.loads(expected_receipt_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise GovernanceError("Confirmatory reveal receipt is not valid JSON.") from exc
    expected_receipt = {
        "schema": "cfeg.confirmatory-lockbox-reveal.v1",
        "execution_contract_sha256": manifest["execution_contract_sha256"],
        "execution_manifest_content_sha256": manifest_digest,
        "scope": "all-six-seed-role-lockbox-predictions",
        "confirmatory_reveal_count": 1,
    }
    if any(receipt.get(key) != value for key, value in expected_receipt.items()):
        raise GovernanceError("Confirmatory reveal receipt is invalid or stale.")

    incomplete = []
    for planned_job in manifest["jobs"]:
        root = Path(str(planned_job["output_dir"]))
        completion_path = root / "training_completion.json"
        final_path = root / "final.pt"
        metrics_path = root / "metrics_train.csv"
        if (
            not completion_path.is_file()
            or not final_path.is_file()
            or not metrics_path.is_file()
            or not (root / "split.csv").is_file()
            or (root / "metrics_val.csv").exists()
        ):
            incomplete.append(planned_job["job_id"])
            continue
        try:
            completion = json.loads(completion_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise GovernanceError(
                f"Training completion receipt is invalid: {planned_job['job_id']}."
            ) from exc
        if (
            completion.get("schema") != "cfeg.training-completion.v1"
            or completion.get("status") != "completed"
            or completion.get("resolved_config_sha256") != planned_job["resolved_config_sha256"]
            or (completion.get("checkpoint_sha256") or {}).get("final.pt")
            != _sha256_file(final_path)
            or (completion.get("artifact_sha256") or {})
            != {
                "metrics_train.csv": _sha256_file(metrics_path),
                "split.csv": _sha256_file(root / "split.csv"),
            }
        ):
            raise GovernanceError(
                f"Training completion is stale or invalid: {planned_job['job_id']}."
            )
    if incomplete:
        raise GovernanceError(
            f"All six training jobs must complete before lockbox access: {incomplete}."
        )
    from cfeg.execution_manifest import (
        confirmatory_training_grid_sha256,
        validate_confirmatory_training_artifacts,
    )

    for planned_job in manifest["jobs"]:
        try:
            checkpoint = torch.load(
                Path(str(planned_job["checkpoint_path"])),
                map_location="cpu",
                weights_only=False,
            )
            validate_confirmatory_training_artifacts(
                planned_job,
                manifest,
                checkpoint,
            )
        except (OSError, TypeError, ValueError) as exc:
            raise GovernanceError(str(exc)) from exc

    try:
        grid_sha256 = confirmatory_training_grid_sha256(manifest)
    except ValueError as exc:
        raise GovernanceError(str(exc)) from exc
    if receipt.get("training_grid_sha256") != grid_sha256:
        raise GovernanceError("Confirmatory reveal receipt training-grid digest is stale.")


def _validate_confirmatory_evaluation_shape(eval_cfg: dict) -> None:
    """Reject every confirmatory inference surface except the registered predictor."""

    allowed_top_level = {
        "data",
        "research_action",
        "scenario",
        "confirmatory_authorization",
        "prediction_output_path",
    }
    unexpected = sorted(set(eval_cfg) - allowed_top_level)
    data = eval_cfg.get("data")
    authorization = eval_cfg.get("confirmatory_authorization")
    if unexpected:
        raise GovernanceError(f"Confirmatory evaluation has unregistered options: {unexpected}.")
    if not isinstance(data, dict) or set(data) != {"processed_dirs", "split"}:
        raise GovernanceError(
            "Confirmatory evaluation data must contain exactly processed_dirs and split."
        )
    if not isinstance(authorization, dict) or set(authorization) != {
        "execution_manifest",
        "lockbox_reveal_receipt",
    }:
        raise GovernanceError(
            "Confirmatory authorization must contain exactly the canonical manifest and "
            "reveal receipt paths."
        )


def _validate_target_free_simulation(plan: dict) -> None:
    simulation = plan["target_free_simulation"]
    plan_path = _project_path(simulation["plan"], field="target_free_simulation.plan")
    if _sha256_file(plan_path) != simulation["plan_sha256"]:
        raise GovernanceError("Target-free simulation plan hash does not match the frozen plan.")
    receipt_path = _project_path(
        simulation["accepted_receipt"], field="target_free_simulation.accepted_receipt"
    )
    if _sha256_file(receipt_path) != simulation["accepted_receipt_sha256"]:
        raise GovernanceError("Target-free simulation receipt hash does not match the plan.")
    try:
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise GovernanceError("Target-free simulation receipt is not valid JSON.") from exc
    expected = {
        "schema": "cfeg.target-free-inference-simulation.v1",
        "status": "accepted",
        "freeze_eligible": True,
        "target_outcomes_accessed": False,
        "n_subjects": int(plan["expected_n_subjects"]),
        "n_folds": len(plan["fold_indices"]),
        "n_optimization_seeds": len(plan["optimization_seeds"]),
        "simulation_plan_sha256": simulation["plan_sha256"],
        "acceptance_checks_passed": True,
        "n_simulations": 10_000,
        "sign_flip_resamples": 10_000,
        "bootstrap_resamples": 10_000,
    }
    mismatched = {
        key: {"expected": value, "observed": receipt.get(key)}
        for key, value in expected.items()
        if receipt.get(key) != value
    }
    if mismatched:
        raise GovernanceError(
            f"Target-free simulation receipt is not freeze-eligible: {mismatched}."
        )
    core = receipt.get("core_results") or {}
    expected_core = {
        "independent_symmetric",
        "heavy_tailed_symmetric",
        "skewed_bounded",
        "paired_discrete_balanced_accuracy",
    }
    if set(core) != expected_core:
        raise GovernanceError("Target-free simulation receipt has an incomplete core DGP set.")
    for name, result in core.items():
        if (
            float(result.get("type_i_wilson95_high", 1.0)) > 0.06
            or float(result.get("coverage_wilson95_low", 0.0)) < 0.94
            or float(result.get("power_at_sesoi_plus_0_03_wilson95_low", 0.0)) < 0.80
        ):
            raise GovernanceError(
                f"Target-free simulation core scenario {name!r} violates a freeze threshold."
            )


def _validate_source_lock(source_lock: dict) -> None:
    expected_hash = str(source_lock.get("implementation_contract_sha256", ""))
    if expected_hash != implementation_contract_sha256():
        raise GovernanceError("Current implementation differs from the frozen contract hash.")
    current = current_source_revision_contract()
    if current.get("source_dirty") is not False:
        raise GovernanceError("Confirmatory execution requires a clean Git worktree.")
    freeze_tag = str(source_lock.get("freeze_tag", ""))
    repository = Path(__file__).resolve().parents[2]
    try:
        tagged_commit = subprocess.run(
            ["git", "rev-list", "-n", "1", freeze_tag],
            cwd=repository,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        raise GovernanceError(f"Frozen source tag does not resolve: {freeze_tag!r}.") from exc
    if tagged_commit != current.get("source_commit_sha"):
        raise GovernanceError("Confirmatory execution must run at the frozen source tag commit.")


def _project_path(value, *, field: str) -> Path:
    if value is None or not str(value).strip():
        raise GovernanceError(f"Missing {field}.")
    path = Path(str(value)).expanduser()
    if not path.is_absolute():
        path = Path(__file__).resolve().parents[2] / path
    path = path.resolve()
    if not path.is_file():
        raise GovernanceError(f"{field} does not exist: {path}.")
    return path


def _load_yaml(path: Path) -> dict:
    value = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(value, dict):
        raise GovernanceError(f"Expected a mapping in {path}.")
    return value


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_json(value) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()
