from __future__ import annotations

import hashlib
import json
import os
import re
import stat
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from cfeg.governance import current_source_revision_contract
from cfeg.identity import canonical_identity_sha256
from cfeg.metadata_calibration_authority import (
    canonical_json_bytes,
    canonical_json_sha256,
    public_key_fingerprint_sha256,
    read_json_object,
    verify_signed_record,
)
from cfeg.metadata_calibration_contract import (
    METADATA_CALIBRATION_BASELINE_LEDGER,
    METADATA_CALIBRATION_INPUT_ARTIFACT_CONTRACT,
    METADATA_CALIBRATION_PLAN,
    validate_metadata_calibration_plan,
)
from cfeg.metadata_calibration_execution import (
    ExperimentPhase,
    cell_execution_contract_sha256,
    execution_cells_for_phase,
    resolve_cell_execution_spec,
)
from cfeg.metadata_calibration_sealer import (
    validate_input_seal_root,
    validate_source_gate_artifacts,
)
from cfeg.utils.config import load_config

EXECUTION_CONTRACT_SCHEMA = "cfeg.metadata-calibration-execution-contract.v1"
PHASE_MANIFEST_SCHEMA = "cfeg.metadata-calibration-phase-manifest.v1"
OUTCOME_DECISION_SCHEMA = "cfeg.metadata-calibration-outcome-decision.v1"
AUTHORIZATION_ENVELOPE_SCHEMA = "cfeg.metadata-calibration-authorization-envelope.v1"
_SHA256 = re.compile(r"[0-9a-f]{64}")


@dataclass(frozen=True)
class MetadataCalibrationExecutionContractBinding:
    phase: str
    execution_contract_sha256: str
    expected_candidate_jobs: int
    expected_baseline_jobs: int
    execution_authorized: bool = False


@dataclass(frozen=True)
class PhaseExecutionBinding:
    phase: str
    phase_manifest_sha256: str
    outcome_decision_signed_record_sha256: str
    input_seal_receipt_signed_record_sha256: str
    execution_output_root: Path
    expected_job_count: int


def build_metadata_calibration_execution_contract(
    *,
    phase: ExperimentPhase,
    held_selected_epochs_by_seed: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build the exact outcome-free job contract from canonical repository inputs."""

    if phase not in {"source_development", "held_participant_evaluation"}:
        raise ValueError(f"Unknown metadata-calibration phase {phase!r}.")
    root = _repository_root()
    plan_binding = validate_metadata_calibration_plan(root / METADATA_CALIBRATION_PLAN)
    plan = load_config(root / METADATA_CALIBRATION_PLAN, strict_env=False)
    recipe = load_config(
        root / "configs/train/metadata_calibration_efficiency_v1.yaml",
        strict_env=False,
    )
    ledger = load_config(root / METADATA_CALIBRATION_BASELINE_LEDGER, strict_env=False)
    allocation = (plan.get("data_contract") or {}).get("proposed_unopened_allocation") or {}
    seeds = tuple(int(value) for value in recipe["runtime"]["source_training_seeds"])
    held_epochs = _normalize_held_selected_epochs(
        held_selected_epochs_by_seed,
        required=False,
    )
    if phase == "source_development" and held_epochs is not None:
        raise ValueError("Source execution cannot carry held refit epoch selections.")
    budgets = tuple(int(value) for value in recipe["episode"]["budgets"])
    checkpoint_groups, target_subjects, source_pool_by_group = _phase_groups(
        phase=phase,
        recipe=recipe,
        allocation=allocation,
    )
    cells = [
        resolve_cell_execution_spec(role=role, context=context, phase=phase).as_dict()
        for role, context in execution_cells_for_phase(phase=phase)
    ]
    baseline_methods = _baseline_methods(ledger)
    query_identity_bundle_field = (
        "source_query_identity_bundle_sha256"
        if phase == "source_development"
        else "held_query_identity_bundle_sha256"
    )
    query_identity_bundle_sha256 = str(
        allocation.get(query_identity_bundle_field, "")
    )
    if not _SHA256.fullmatch(query_identity_bundle_sha256):
        raise ValueError("Frozen phase query-identity bundle is not one SHA-256 value.")

    jobs: list[dict[str, Any]] = []
    for checkpoint_group in checkpoint_groups:
        outer_fold = (
            int(checkpoint_group.removeprefix("fold"))
            if checkpoint_group.startswith("fold")
            else None
        )
        for seed in seeds:
            payload: dict[str, Any] = {
                "job_id": f"candidate-{checkpoint_group}-seed{seed}",
                "producer_kind": "candidate",
                "checkpoint_group": checkpoint_group,
                "outer_fold": outer_fold,
                "seed": seed,
                "adapter_id": None,
                "budgets": list(budgets),
                "cells": cells,
                "target_subject_ids_sha256": canonical_identity_sha256(
                    target_subjects[checkpoint_group]
                ),
                "source_pool_subject_ids_sha256": canonical_identity_sha256(
                    source_pool_by_group[checkpoint_group]
                ),
                "required_output": (
                    f"candidate/{checkpoint_group}/seed{seed}/probabilities.parquet"
                ),
                "required_receipt": (
                    f"candidate/{checkpoint_group}/seed{seed}/producer_receipt.json"
                ),
                "required_artifacts": [
                    {
                        "relative_path": (
                            f"candidate/{checkpoint_group}/seed{seed}/probabilities.parquet"
                        ),
                        "artifact_kind": "candidate_probability_grid",
                    },
                    {
                        "relative_path": (
                            f"candidate/{checkpoint_group}/seed{seed}/composite.safetensors"
                        ),
                        "artifact_kind": "full_composite_checkpoint",
                    },
                    {
                        "relative_path": (
                            f"candidate/{checkpoint_group}/seed{seed}/composite.receipt.json"
                        ),
                        "artifact_kind": "full_composite_checkpoint_receipt",
                    },
                    {
                        "relative_path": (
                            f"candidate/{checkpoint_group}/seed{seed}/training_selection.json"
                        ),
                        "artifact_kind": "epoch_selection_record",
                    },
                    {
                        "relative_path": (
                            f"candidate/{checkpoint_group}/seed{seed}/producer_receipt.json"
                        ),
                        "artifact_kind": "producer_receipt",
                    },
                ],
                "query_outcomes_loaded": False,
                "held_selected_epochs": (
                    None
                    if phase == "source_development"
                    else (
                        "source_gate_bound_at_phase_manifest"
                        if held_epochs is None
                        else held_epochs[str(seed)]
                    )
                ),
            }
            payload["job_contract_sha256"] = _json_sha256(payload)
            jobs.append(payload)
        for method in baseline_methods:
            adapter_id = str(method["adapter_id"])
            source_pool = (
                source_pool_by_group[checkpoint_group]
                if adapter_id == "chiang2021_LST_filterbank_eTRCA"
                else ()
            )
            payload = {
                "job_id": f"baseline-{adapter_id}-{checkpoint_group}",
                "producer_kind": "baseline",
                "checkpoint_group": checkpoint_group,
                "outer_fold": outer_fold,
                "seed": None,
                "adapter_id": adapter_id,
                "budgets": [int(value) for value in method["budgets"]],
                "cells": [],
                "target_subject_ids_sha256": canonical_identity_sha256(
                    target_subjects[checkpoint_group]
                ),
                "source_pool_subject_ids_sha256": canonical_identity_sha256(source_pool),
                "required_output": (
                    f"baseline/{adapter_id}/{checkpoint_group}/scores.parquet"
                ),
                "required_receipt": (
                    f"baseline/{adapter_id}/{checkpoint_group}/producer_receipt.json"
                ),
                "required_artifacts": [
                    {
                        "relative_path": (
                            f"baseline/{adapter_id}/{checkpoint_group}/scores.parquet"
                        ),
                        "artifact_kind": "baseline_raw_score_grid",
                    },
                    {
                        "relative_path": (
                            f"baseline/{adapter_id}/{checkpoint_group}/producer_receipt.json"
                        ),
                        "artifact_kind": "producer_receipt",
                    },
                ],
                "external_context_access": False,
                "query_batch_transduction": False,
                "query_outcomes_loaded": False,
            }
            payload["job_contract_sha256"] = _json_sha256(payload)
            jobs.append(payload)

    candidate_count = len(checkpoint_groups) * len(seeds)
    baseline_count = len(checkpoint_groups) * len(baseline_methods)
    contract: dict[str, Any] = {
        "schema": EXECUTION_CONTRACT_SCHEMA,
        "candidate_id": "metadata-calibration-efficiency-v1",
        "phase": phase,
        "status": "outcome_free_contract_not_execution_authority",
        "global_contracts": {
            "plan_sha256": plan_binding["plan_sha256"],
            "source_recipe_sha256": plan_binding["source_recipe_sha256"],
            "baseline_ledger_sha256": plan_binding["baseline_ledger_sha256"],
            "power_receipt_sha256": plan_binding["power_receipt_sha256"],
            "input_artifact_contract_sha256": plan_binding[
                "input_artifact_contract_sha256"
            ],
            "cell_execution_contract_sha256": cell_execution_contract_sha256(
                phase=phase
            ),
        },
        "phase_spec": {
            "n_classes": 12,
            "budgets": list(budgets),
            "seeds": list(seeds),
            "conditions": ["dry", "wet"],
            "query_identity_bundle_sha256": query_identity_bundle_sha256,
            "checkpoint_groups": list(checkpoint_groups),
            "target_subject_ids_by_checkpoint_group": {
                key: list(target_subjects[key]) for key in checkpoint_groups
            },
            "source_pool_subject_ids_by_checkpoint_group": {
                key: list(source_pool_by_group[key]) for key in checkpoint_groups
            },
            "cells": cells,
            "held_selected_epochs_by_seed": (
                None
                if phase == "source_development"
                else (
                    "source_gate_bound_at_phase_manifest"
                    if held_epochs is None
                    else held_epochs
                )
            ),
        },
        "input_requirements": {
            "artifact_contract": METADATA_CALIBRATION_INPUT_ARTIFACT_CONTRACT,
            "input_seal_receipt_required": True,
            "query_label_sidecar_available_to_producers": False,
            "intervention_mapping_sha256_required": True,
            "resolved_context_values_sha256_required": True,
            "resolved_context_usage_sha256_required": True,
            "exact_opaque_row_token_coverage_required": True,
        },
        "held_prerequisites": (
            None
            if phase == "source_development"
            else {
                "source_development_gate_receipt_sha256": "required_in_execution_manifest",
                "source_development_private_seal_sha256": "required_in_execution_manifest",
                "source_development_status": "passed",
                "held_selected_epochs_by_seed_sha256": (
                    "required_in_execution_manifest"
                    if held_epochs is None
                    else canonical_json_sha256(held_epochs)
                ),
            }
        ),
        "jobs": jobs,
        "expected_candidate_jobs": candidate_count,
        "expected_baseline_jobs": baseline_count,
        "expected_job_count": candidate_count + baseline_count,
        "execution_authorized": False,
    }
    final_binding = validate_metadata_calibration_plan(root / METADATA_CALIBRATION_PLAN)
    if final_binding != plan_binding:
        raise RuntimeError("Execution-contract dependencies drifted while being read.")
    contract["execution_contract_sha256"] = _json_sha256(contract)
    return contract


def validate_metadata_calibration_execution_contract(
    contract: Mapping[str, Any],
) -> MetadataCalibrationExecutionContractBinding:
    """Require byte-semantic equality with the phase-derived canonical contract."""

    phase = str(contract.get("phase", ""))
    if phase not in {"source_development", "held_participant_evaluation"}:
        raise ValueError("Execution contract has an invalid phase.")
    phase_spec = contract.get("phase_spec")
    observed_held_epochs = (
        phase_spec.get("held_selected_epochs_by_seed")
        if isinstance(phase_spec, Mapping)
        else None
    )
    held_epochs = (
        observed_held_epochs
        if isinstance(observed_held_epochs, Mapping)
        else None
    )
    expected = build_metadata_calibration_execution_contract(
        phase=phase,
        held_selected_epochs_by_seed=held_epochs,
    )
    try:
        observed_payload = _canonical_json(contract)
    except (TypeError, ValueError) as error:
        raise ValueError("Execution contract is not canonical JSON.") from error
    if observed_payload != _canonical_json(expected):
        raise ValueError("Execution contract differs from the canonical phase-derived contract.")
    return MetadataCalibrationExecutionContractBinding(
        phase=phase,
        execution_contract_sha256=str(expected["execution_contract_sha256"]),
        expected_candidate_jobs=int(expected["expected_candidate_jobs"]),
        expected_baseline_jobs=int(expected["expected_baseline_jobs"]),
    )


def build_phase_execution_manifest(
    *,
    phase: ExperimentPhase,
    input_seal_root: str | Path,
    execution_output_root: str | Path,
    trusted_signing_public_key_path: str | Path,
    source_tag: str,
    source_development_gate_receipt_path: str | Path | None = None,
    source_development_private_seal_path: str | Path | None = None,
) -> dict[str, Any]:
    """Bind an exact sealed input tree to one new, outcome-free execution root."""

    input_binding = validate_input_seal_root(
        input_seal_root,
        trusted_signing_public_key_path=trusted_signing_public_key_path,
    )
    if input_binding.phase != phase:
        raise ValueError("Input-seal phase differs from the requested execution phase.")
    receipt = read_json_object(input_binding.sealed_root / "input_seal_receipt.json")
    output_root = _absolute_no_symlink_path(Path(execution_output_root), must_exist=False)
    if os.path.lexists(output_root):
        raise FileExistsError("Execution output root must be completely new.")
    source = current_source_revision_contract()
    if source.get("source_dirty") is not False:
        raise RuntimeError("Phase manifests require a clean source tree.")
    if receipt.get("source_commit") != source.get("source_commit_sha") or receipt.get(
        "source_tree_sha256"
    ) != source.get("source_tree_sha256"):
        raise ValueError("Input seal and current source revision differ.")
    _verify_source_tag(source_tag, str(source["source_commit_sha"]))

    gate_binding: dict[str, Any] | None = None
    held_epochs: dict[str, dict[str, int]] | None = None
    if phase == "held_participant_evaluation":
        if source_development_gate_receipt_path is None or (
            source_development_private_seal_path is None
        ):
            raise PermissionError("Held manifest requires the source PASS gate and private seal.")
        source_gate = validate_source_gate_artifacts(
            source_development_gate_receipt_path,
            source_development_private_seal_path,
            trusted_signing_public_key_path=trusted_signing_public_key_path,
        )
        gate_hash = source_gate.gate_receipt_file_sha256
        seal_hash = source_gate.private_seal_file_sha256
        if receipt.get("input_seal_decision_signed_record_sha256") is None:
            raise ValueError("Held input receipt lacks its input decision binding.")
        input_decision = read_json_object(input_binding.sealed_root / "input_seal_decision.json")
        if input_decision.get("source_development_gate_receipt_sha256") != gate_hash or (
            input_decision.get("source_development_private_seal_sha256") != seal_hash
        ):
            raise ValueError("Held input authorization differs from source gate artifacts.")
        gate_binding = {
            "source_development_gate_receipt_file_sha256": gate_hash,
            "source_development_private_seal_file_sha256": seal_hash,
            "source_development_gate_signed_record_sha256": (
                source_gate.gate_signed_record_sha256
            ),
            "source_development_private_seal_signed_record_sha256": (
                source_gate.private_seal_signed_record_sha256
            ),
            "source_development_status": "pass",
            "held_selected_epochs_by_seed_sha256": (
                source_gate.held_selected_epochs_by_seed_sha256
            ),
        }
        held_epochs = source_gate.held_selected_epochs_by_seed
    elif any(
        value is not None
        for value in (
            source_development_gate_receipt_path,
            source_development_private_seal_path,
        )
    ):
        raise ValueError("Source-development manifest must not carry prior-phase artifacts.")

    contract = build_metadata_calibration_execution_contract(
        phase=phase,
        held_selected_epochs_by_seed=held_epochs,
    )
    contract_binding = validate_metadata_calibration_execution_contract(contract)

    manifest: dict[str, Any] = {
        "schema": PHASE_MANIFEST_SCHEMA,
        "candidate_id": "metadata-calibration-efficiency-v1",
        "phase": phase,
        "status": "outcome_free_exact_execution_manifest",
        "execution_contract": contract,
        "execution_contract_sha256": contract_binding.execution_contract_sha256,
        "input_seal_root": str(input_binding.sealed_root),
        "input_seal_receipt_signed_record_sha256": (
            input_binding.input_seal_receipt_signed_record_sha256
        ),
        "input_artifact_tree_sha256": input_binding.artifact_tree_sha256,
        "execution_output_root": str(output_root),
        "expected_job_count": int(contract["expected_job_count"]),
        "source_commit": source["source_commit_sha"],
        "source_tree_sha256": source["source_tree_sha256"],
        "source_tag": source_tag,
        "source_development_gate_binding": gate_binding,
        "query_outcomes_loaded": False,
        "execution_authorized": False,
    }
    manifest["phase_manifest_sha256"] = canonical_json_sha256(manifest)
    return manifest


def validate_phase_execution_manifest(
    manifest: Mapping[str, Any],
    *,
    trusted_signing_public_key_path: str | Path,
) -> dict[str, Any]:
    observed = dict(manifest)
    claimed_hash = observed.pop("phase_manifest_sha256", None)
    if not _is_sha256(claimed_hash) or claimed_hash != canonical_json_sha256(observed):
        raise ValueError("Phase manifest self-hash is invalid.")
    observed["phase_manifest_sha256"] = claimed_hash
    phase = str(observed.get("phase"))
    expected_fields = {
        "schema",
        "candidate_id",
        "phase",
        "status",
        "execution_contract",
        "execution_contract_sha256",
        "input_seal_root",
        "input_seal_receipt_signed_record_sha256",
        "input_artifact_tree_sha256",
        "execution_output_root",
        "expected_job_count",
        "source_commit",
        "source_tree_sha256",
        "source_tag",
        "source_development_gate_binding",
        "query_outcomes_loaded",
        "execution_authorized",
        "phase_manifest_sha256",
    }
    if set(observed) != expected_fields or observed.get("schema") != PHASE_MANIFEST_SCHEMA:
        raise ValueError("Phase manifest uses an unexpected exact schema.")
    if phase not in {"source_development", "held_participant_evaluation"}:
        raise ValueError("Phase manifest phase is invalid.")
    if type(observed.get("query_outcomes_loaded")) is not bool or observed.get(
        "query_outcomes_loaded"
    ) is not False:
        raise ValueError("Phase manifest must be outcome-free.")
    if type(observed.get("execution_authorized")) is not bool or observed.get(
        "execution_authorized"
    ) is not False:
        raise ValueError("A phase manifest itself cannot grant execution authority.")
    contract_binding = validate_metadata_calibration_execution_contract(
        observed["execution_contract"]
    )
    if observed.get("execution_contract_sha256") != (
        contract_binding.execution_contract_sha256
    ) or observed.get("expected_job_count") != (
        contract_binding.expected_candidate_jobs + contract_binding.expected_baseline_jobs
    ):
        raise ValueError("Phase manifest execution contract binding is invalid.")
    input_binding = validate_input_seal_root(
        str(observed["input_seal_root"]),
        trusted_signing_public_key_path=trusted_signing_public_key_path,
    )
    if (
        input_binding.phase != phase
        or observed.get("input_seal_receipt_signed_record_sha256")
        != input_binding.input_seal_receipt_signed_record_sha256
        or observed.get("input_artifact_tree_sha256") != input_binding.artifact_tree_sha256
    ):
        raise ValueError("Phase manifest differs from its input seal.")
    output_root = _absolute_no_symlink_path(
        Path(str(observed["execution_output_root"])), must_exist=False
    )
    source = current_source_revision_contract()
    if source.get("source_dirty") is not False or observed.get("source_commit") != source.get(
        "source_commit_sha"
    ) or observed.get("source_tree_sha256") != source.get("source_tree_sha256"):
        raise ValueError("Phase manifest source revision is no longer current and clean.")
    _verify_source_tag(str(observed["source_tag"]), str(source["source_commit_sha"]))
    if phase == "held_participant_evaluation":
        gate = observed.get("source_development_gate_binding")
        if not isinstance(gate, Mapping) or gate.get("source_development_status") != "pass" or (
            not _is_sha256(gate.get("source_development_gate_receipt_file_sha256"))
        ) or not _is_sha256(
            gate.get("source_development_private_seal_file_sha256")
        ) or not _is_sha256(
            gate.get("source_development_gate_signed_record_sha256")
        ) or not _is_sha256(
            gate.get("source_development_private_seal_signed_record_sha256")
        ) or not _is_sha256(
            gate.get("held_selected_epochs_by_seed_sha256")
        ):
            raise PermissionError("Held manifest lacks a bound passing source gate.")
        input_decision = read_json_object(
            input_binding.sealed_root / "input_seal_decision.json"
        )
        source_gate = validate_source_gate_artifacts(
            str(input_decision["source_development_gate_receipt_path"]),
            str(input_decision["source_development_private_seal_path"]),
            trusted_signing_public_key_path=trusted_signing_public_key_path,
        )
        expected_gate = {
            "source_development_gate_receipt_file_sha256": (
                source_gate.gate_receipt_file_sha256
            ),
            "source_development_private_seal_file_sha256": (
                source_gate.private_seal_file_sha256
            ),
            "source_development_gate_signed_record_sha256": (
                source_gate.gate_signed_record_sha256
            ),
            "source_development_private_seal_signed_record_sha256": (
                source_gate.private_seal_signed_record_sha256
            ),
            "source_development_status": "pass",
            "held_selected_epochs_by_seed_sha256": (
                source_gate.held_selected_epochs_by_seed_sha256
            ),
        }
        if dict(gate) != expected_gate:
            raise PermissionError("Held manifest differs from its actual signed source gate.")
        held_epochs = observed["execution_contract"]["phase_spec"][
            "held_selected_epochs_by_seed"
        ]
        if held_epochs != source_gate.held_selected_epochs_by_seed or (
            canonical_json_sha256(held_epochs)
            != source_gate.held_selected_epochs_by_seed_sha256
        ):
            raise PermissionError("Held execution epochs differ from the source gate.")
        for job in observed["execution_contract"]["jobs"]:
            if job["producer_kind"] == "candidate" and job[
                "held_selected_epochs"
            ] != held_epochs[str(job["seed"])]:
                raise PermissionError("Held candidate job epochs differ from the source gate.")
    elif observed.get("source_development_gate_binding") is not None:
        raise ValueError("Source manifest unexpectedly carries a prior-phase gate.")
    return {
        "phase": phase,
        "phase_manifest_sha256": claimed_hash,
        "input_binding": input_binding,
        "execution_output_root": output_root,
        "execution_contract": observed["execution_contract"],
    }


def build_outcome_execution_decision(
    manifest: Mapping[str, Any],
    *,
    trusted_signing_public_key_path: str | Path,
    signing_key_id: str,
    decision_id: str,
    decision_date: str,
    authorization_basis: str,
    signer_role: str = "delegated_automation_authority",
) -> dict[str, Any]:
    binding = validate_phase_execution_manifest(
        manifest,
        trusted_signing_public_key_path=trusted_signing_public_key_path,
    )
    if signing_key_id != public_key_fingerprint_sha256_from_path(
        trusted_signing_public_key_path
    ):
        raise ValueError("Outcome decision signing key ID differs from the trusted key.")
    return {
        "schema": OUTCOME_DECISION_SCHEMA,
        "candidate_id": "metadata-calibration-efficiency-v1",
        "decision_id": str(decision_id),
        "decision_date": str(decision_date),
        "phase": binding["phase"],
        "outcome_execution_approved": True,
        "approved_action": (
            "execute_exact_manifest_seal_complete_private_grid_then_finalize_once"
        ),
        "phase_manifest_sha256": binding["phase_manifest_sha256"],
        "input_seal_receipt_signed_record_sha256": (
            binding["input_binding"].input_seal_receipt_signed_record_sha256
        ),
        "input_artifact_tree_sha256": binding["input_binding"].artifact_tree_sha256,
        "execution_contract_sha256": binding["execution_contract"][
            "execution_contract_sha256"
        ],
        "execution_output_root": str(binding["execution_output_root"]),
        "expected_job_count": int(binding["execution_contract"]["expected_job_count"]),
        "source_commit": manifest["source_commit"],
        "source_tree_sha256": manifest["source_tree_sha256"],
        "source_tag": manifest["source_tag"],
        "source_development_gate_binding": manifest[
            "source_development_gate_binding"
        ],
        "query_label_access_timing": (
            "finalizer_only_after_complete_filesystem_private_seal"
        ),
        "partial_grid_reveal_forbidden": True,
        "signing_key_id": signing_key_id,
        "signer_role": signer_role,
        "authorization_basis": authorization_basis,
        "signature_algorithm": "ed25519",
    }


def validate_outcome_execution_decision(
    decision: Mapping[str, Any],
    *,
    manifest: Mapping[str, Any],
    trusted_signing_public_key_path: str | Path,
) -> str:
    signed_hash = verify_signed_record(
        decision,
        public_key_path=trusted_signing_public_key_path,
        hash_field="decision_payload_sha256",
        expected_schema=OUTCOME_DECISION_SCHEMA,
    )
    manifest_binding = validate_phase_execution_manifest(
        manifest,
        trusted_signing_public_key_path=trusted_signing_public_key_path,
    )
    expected = build_outcome_execution_decision(
        manifest,
        trusted_signing_public_key_path=trusted_signing_public_key_path,
        signing_key_id=str(decision.get("signing_key_id")),
        decision_id=str(decision.get("decision_id")),
        decision_date=str(decision.get("decision_date")),
        authorization_basis=str(decision.get("authorization_basis")),
        signer_role=str(decision.get("signer_role")),
    )
    unsigned = dict(decision)
    for field in ("decision_payload_sha256", "signature", "signed_record_sha256"):
        unsigned.pop(field, None)
    if canonical_json_bytes(unsigned) != canonical_json_bytes(expected):
        raise ValueError("Outcome decision differs from its exact phase manifest authority.")
    if type(decision.get("outcome_execution_approved")) is not bool or decision.get(
        "outcome_execution_approved"
    ) is not True:
        raise PermissionError("Outcome execution is not approved.")
    if decision.get("phase_manifest_sha256") != manifest_binding[
        "phase_manifest_sha256"
    ]:
        raise ValueError("Outcome decision phase-manifest binding drifted.")
    return signed_hash


def build_authorization_envelope(
    *,
    manifest: Mapping[str, Any],
    outcome_decision: Mapping[str, Any],
    trusted_signing_public_key_path: str | Path,
) -> dict[str, Any]:
    decision_hash = validate_outcome_execution_decision(
        outcome_decision,
        manifest=manifest,
        trusted_signing_public_key_path=trusted_signing_public_key_path,
    )
    envelope: dict[str, Any] = {
        "schema": AUTHORIZATION_ENVELOPE_SCHEMA,
        "candidate_id": "metadata-calibration-efficiency-v1",
        "phase": manifest["phase"],
        "phase_manifest_sha256": manifest["phase_manifest_sha256"],
        "outcome_decision_signed_record_sha256": decision_hash,
        "input_seal_receipt_signed_record_sha256": manifest[
            "input_seal_receipt_signed_record_sha256"
        ],
        "input_artifact_tree_sha256": manifest["input_artifact_tree_sha256"],
        "execution_contract_sha256": manifest["execution_contract_sha256"],
        "execution_output_root": manifest["execution_output_root"],
        "expected_job_count": manifest["expected_job_count"],
        "source_commit": manifest["source_commit"],
        "source_tree_sha256": manifest["source_tree_sha256"],
        "source_tag": manifest["source_tag"],
        "source_development_gate_binding": manifest[
            "source_development_gate_binding"
        ],
    }
    envelope["authorization_envelope_sha256"] = canonical_json_sha256(envelope)
    return envelope


def validate_authorization_envelope(
    envelope: Mapping[str, Any],
    *,
    manifest: Mapping[str, Any],
    outcome_decision: Mapping[str, Any],
    trusted_signing_public_key_path: str | Path,
) -> PhaseExecutionBinding:
    expected = build_authorization_envelope(
        manifest=manifest,
        outcome_decision=outcome_decision,
        trusted_signing_public_key_path=trusted_signing_public_key_path,
    )
    if canonical_json_bytes(dict(envelope)) != canonical_json_bytes(expected):
        raise ValueError("Authorization envelope differs from its exact signed dependencies.")
    return PhaseExecutionBinding(
        phase=str(manifest["phase"]),
        phase_manifest_sha256=str(manifest["phase_manifest_sha256"]),
        outcome_decision_signed_record_sha256=str(
            outcome_decision["signed_record_sha256"]
        ),
        input_seal_receipt_signed_record_sha256=str(
            manifest["input_seal_receipt_signed_record_sha256"]
        ),
        execution_output_root=Path(str(manifest["execution_output_root"])),
        expected_job_count=int(manifest["expected_job_count"]),
    )


def _phase_groups(
    *,
    phase: ExperimentPhase,
    recipe: Mapping[str, Any],
    allocation: Mapping[str, Any],
) -> tuple[
    tuple[str, ...],
    dict[str, tuple[str, ...]],
    dict[str, tuple[str, ...]],
]:
    if phase == "held_participant_evaluation":
        held = tuple(str(value) for value in allocation["held_participant_evaluation_subject_ids"])
        source = tuple(str(value) for value in allocation["source_development_subject_ids"])
        return ("held",), {"held": held}, {"held": source}

    checkpoints: list[str] = []
    targets: dict[str, tuple[str, ...]] = {}
    sources: dict[str, tuple[str, ...]] = {}
    for fold in (0, 1, 2):
        group = f"fold{fold}"
        fold_spec = recipe["data"]["folds"][fold]
        outer = tuple(str(value) for value in fold_spec["outer_evaluation"])
        source = tuple(
            str(value)
            for value in (*fold_spec["inner_fit"], *fold_spec["inner_validation"])
        )
        if len(outer) != 13 or len(source) != 26 or set(outer) & set(source):
            raise ValueError("Source-development fold membership drifted.")
        checkpoints.append(group)
        targets[group] = outer
        sources[group] = source
    expected = {str(value) for value in allocation["source_development_subject_ids"]}
    if set().union(*map(set, targets.values())) != expected or any(
        set(targets[group]) | set(sources[group]) != expected for group in checkpoints
    ):
        raise ValueError("Source-development execution groups differ from the allocation.")
    return tuple(checkpoints), targets, sources


def _normalize_held_selected_epochs(
    value: Mapping[str, Any] | None,
    *,
    required: bool,
) -> dict[str, dict[str, int]] | None:
    if value is None:
        if required:
            raise ValueError("Held execution requires source-gate refit epoch selections.")
        return None
    if set(map(str, value)) != {"42", "43", "44"}:
        raise ValueError("Held refit epoch selection must contain exact seeds 42,43,44.")
    normalized: dict[str, dict[str, int]] = {}
    for seed in (42, 43, 44):
        observed = value.get(str(seed))
        if not isinstance(observed, Mapping) or set(observed) != {
            "stage1_epochs",
            "stage2_epochs",
        }:
            raise ValueError("Held refit epoch selection has an unexpected schema.")
        stage1 = observed.get("stage1_epochs")
        stage2 = observed.get("stage2_epochs")
        if type(stage1) is not int or not 10 <= stage1 <= 40:
            raise ValueError("Held Stage-1 epoch count must lie in [10,40].")
        if type(stage2) is not int or not 5 <= stage2 <= 25:
            raise ValueError("Held Stage-2 epoch count must lie in [5,25].")
        normalized[str(seed)] = {
            "stage1_epochs": stage1,
            "stage2_epochs": stage2,
        }
    return normalized


def _baseline_methods(ledger: Mapping[str, Any]) -> tuple[dict[str, Any], ...]:
    methods = tuple(
        dict(value)
        for value in ledger.get("mandatory_before_held_reveal", ())
        if value.get("producer_kind") == "baseline"
    )
    adapters = tuple(str(value.get("adapter_id")) for value in methods)
    if adapters != (
        "strict_FBCCA",
        "target_template_correlation",
        "target_filterbank_eTRCA",
        "same3_filterbank_eTRCA",
        "chiang2021_LST_filterbank_eTRCA",
    ):
        raise ValueError("Execution contract baseline adapters differ from the ledger.")
    return methods


def _json_sha256(value: object) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _canonical_json(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def public_key_fingerprint_sha256_from_path(path: str | Path) -> str:
    from cryptography.hazmat.primitives import serialization

    target = Path(path).absolute()
    observed = target.lstat()
    if not stat.S_ISREG(observed.st_mode) or observed.st_nlink != 1:
        raise ValueError("Trusted signing public key must be a non-hard-linked regular file.")
    descriptor = os.open(target, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    try:
        current = os.fstat(descriptor)
        if (current.st_dev, current.st_ino) != (observed.st_dev, observed.st_ino):
            raise ValueError("Trusted signing public key changed while being opened.")
        payload = b""
        while True:
            chunk = os.read(descriptor, 65536)
            if not chunk:
                break
            payload += chunk
    finally:
        os.close(descriptor)
    key = serialization.load_pem_public_key(payload)
    return public_key_fingerprint_sha256(key)


def _absolute_no_symlink_path(path: Path, *, must_exist: bool) -> Path:
    absolute = path.absolute()
    component = Path(absolute.anchor)
    for name in absolute.parts[1:]:
        component /= name
        try:
            observed = component.lstat()
        except FileNotFoundError:
            if must_exist:
                raise FileNotFoundError(component) from None
            continue
        if stat.S_ISLNK(observed.st_mode):
            raise ValueError(f"Path contains a symlink component: {component}.")
    if must_exist and not absolute.exists():
        raise FileNotFoundError(absolute)
    return absolute


def _verify_source_tag(tag: str, source_commit: str) -> None:
    import subprocess

    if not tag.strip() or tag != tag.strip() or tag.startswith("refs/"):
        raise ValueError("Source freeze tag must be nonempty.")
    try:
        subprocess.run(
            ["git", "check-ref-format", f"refs/tags/{tag}"],
            cwd=_repository_root(),
            check=True,
            capture_output=True,
            text=True,
        )
        observed = subprocess.run(
            ["git", "rev-parse", "--verify", f"refs/tags/{tag}^{{commit}}"],
            cwd=_repository_root(),
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError) as error:
        raise ValueError("Source freeze tag does not exist.") from error
    if observed != source_commit:
        raise ValueError("Source freeze tag does not resolve to the current commit.")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _is_sha256(value: object) -> bool:
    return isinstance(value, str) and bool(_SHA256.fullmatch(value))


def _repository_root() -> Path:
    return Path(__file__).resolve().parents[2]
