from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from cfeg.analysis.provenance import (
    PRIMARY_ANALYSIS_MANIFEST_SCHEMA,
    VerifiedPredictionBundle,
    primary_analysis_manifest_sha256,
)
from cfeg.eval_loop import (
    _checkpoint_split_indices,
    _loader,
    _sample_id_identity_sha256,
    collect_predictions,
    load_evaluation_context,
)
from cfeg.execution_manifest import resolved_config_sha256, sha256_json
from cfeg.train_loop import _environment_contract, _sha256_json, _source_revision_contract
from cfeg.utils.checkpoint import save_json

PREDICTION_SCHEMA_VERSION = "cfeg.predictions.v3"
OOD_REQUIRED_COLUMNS = ("sample_id", "label", "prediction")
PRIMARY_CONTRACT_COLUMNS = (
    "execution_job_id",
    "execution_contract_sha256",
    "execution_manifest_content_sha256",
    "lockbox_reveal_receipt_sha256",
    "planned_config_sha256",
    "training_completion_sha256",
    "primary_ablation_role",
    "primary_fairness_hash",
    "metadata_contract_version",
    "conditioning_architecture",
    "external_metadata_mode",
    "optimization_seed",
    "split_seed",
    "n_folds",
    "fold_index",
    "parameter_schema_sha256",
    "initial_trainable_state_sha256",
    "split_assignment_sha256",
    "vocabulary_sha256",
    "asset_provenance_sha256",
    "execution_phase",
    "analysis_plan_sha256",
    "cohort_roles_sha256",
    "cohort_sha256",
    "outer_test_access_during_training",
    "source_commit_sha",
    "source_dirty",
    "source_tree_sha256",
    "environment_sha256",
    "inference_environment_sha256",
    "checkpoint_role",
    "checkpoint_selection_split",
    "checkpoint_selection_metric",
    "analysis_manifest_schema",
    "analysis_manifest_sha256",
    "split_manifest_sha256",
)


def run_prediction(
    ckpt_path: str | Path,
    processed_dir: str | Path,
    output_path: str | Path,
    *,
    split: str = "test",
    scenario: str = "clean",
    confirmatory_authorization: dict[str, str] | None = None,
    development_authorization: dict[str, str] | None = None,
    recorded_output_path: str | Path | None = None,
) -> dict[str, Any]:
    """Write an auditable prediction artifact for paired held-out analysis.

    The safe default selects the checkpoint-adjacent ``split.csv`` test rows.
    Whole-manifest inference remains available only through explicit
    ``split="all"`` selection.
    """

    if split not in {"val", "test", "all"}:
        raise ValueError(f"split must be 'val', 'test', or 'all', got {split!r}.")
    if not str(scenario).strip():
        raise ValueError("scenario must be non-empty.")

    checkpoint_path = Path(ckpt_path).expanduser().resolve()
    processed_path = Path(processed_dir).expanduser().resolve()
    output = Path(output_path).expanduser()
    eval_cfg = {
        "data": {"processed_dirs": [str(processed_path)], "split": split},
        "research_action": "primary_prediction_lockbox_clean",
        "scenario": str(scenario),
        "prediction_output_path": str(output.resolve()),
    }
    if confirmatory_authorization is not None:
        eval_cfg["confirmatory_authorization"] = dict(confirmatory_authorization)
    if development_authorization is not None:
        eval_cfg["development_authorization"] = dict(development_authorization)
    context = load_evaluation_context(eval_cfg, checkpoint_path, "prediction_bundle")
    _validate_primary_execution_source(context)
    indices = _prediction_indices(context, split=split)
    y_true, logits, sample_ids = collect_predictions(
        context["model"], _loader(context, indices), context["device"]
    )
    frame, artifact = _prediction_artifact(
        context=context,
        checkpoint_path=checkpoint_path,
        processed_path=processed_path,
        split=split,
        scenario=str(scenario),
        sample_ids=sample_ids,
        y_true=y_true,
        logits=logits,
    )

    output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output, index=False)
    recorded_output = (
        Path(recorded_output_path).expanduser().resolve()
        if recorded_output_path is not None
        else output.resolve()
    )
    artifact["prediction_csv"] = str(recorded_output)
    artifact["prediction_csv_sha256"] = _file_sha256(output)
    provenance_path = output.with_name(f"{output.stem}_provenance.json")
    save_json(provenance_path, artifact)
    return {
        "output_csv": str(output),
        "provenance_json": str(provenance_path),
        **artifact,
    }


def _prediction_indices(context: dict[str, Any], *, split: str) -> np.ndarray:
    if split in {"val", "test"}:
        return _checkpoint_split_indices(context, split=split)
    if split == "all":
        return np.arange(len(context["dataset"]), dtype=int)
    raise ValueError(f"split must be 'val', 'test', or 'all', got {split!r}.")


def _prediction_artifact(
    *,
    context: dict[str, Any],
    checkpoint_path: Path,
    processed_path: Path,
    split: str,
    scenario: str,
    sample_ids: list[str],
    y_true: np.ndarray,
    logits: np.ndarray,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    labels = np.asarray(y_true)
    scores = np.asarray(logits)
    if scores.ndim != 2:
        raise ValueError(f"logits must have shape [N,K], got {scores.shape}.")
    if len(sample_ids) != len(labels) or len(labels) != len(scores):
        raise ValueError(
            "Prediction arrays have inconsistent lengths: "
            f"sample_ids={len(sample_ids)}, labels={len(labels)}, logits={len(scores)}."
        )
    normalized_ids = [str(sample_id) for sample_id in sample_ids]
    if len(set(normalized_ids)) != len(normalized_ids):
        raise ValueError("Prediction sample_id values must be unique within one artifact.")

    probabilities = torch.softmax(torch.from_numpy(scores), dim=1).numpy()
    predicted = scores.argmax(axis=1)
    class_map = context["checkpoint"].get("class_map") or {}
    frequencies = [
        (class_map.get(str(int(label))) or {}).get("stimulus_frequency_hz") for label in predicted
    ]
    sample_hash = _sample_id_identity_sha256(normalized_ids)
    checkpoint_hash = _file_sha256(checkpoint_path)
    primary_contract = _primary_contract(context)
    is_primary = primary_contract.get("metadata_contract_version") == "0.4-dev" and (
        primary_contract.get("primary_ablation_role")
        in {"A0_eeg_only", "A2_structured_condition_prompt"}
        or (
            primary_contract.get("conditioning_architecture") == "physical_hybrid_v1"
            and primary_contract.get("execution_phase") == "development"
        )
    )
    primary_contract["analysis_manifest_schema"] = (
        PRIMARY_ANALYSIS_MANIFEST_SCHEMA if is_primary else None
    )
    primary_contract["analysis_manifest_sha256"] = (
        primary_analysis_manifest_sha256(context["manifest"], normalized_ids)
        if is_primary
        else None
    )
    split_path = checkpoint_path.parent / "split.csv" if split in {"val", "test"} else None
    split_manifest_hash = _file_sha256(split_path) if split_path is not None else None
    primary_contract["split_manifest_sha256"] = split_manifest_hash
    frame = pd.DataFrame(
        {
            "sample_id": normalized_ids,
            "label": labels.astype(int),
            "prediction": predicted.astype(int),
            "predicted_frequency_hz": frequencies,
            "confidence": probabilities.max(axis=1),
            "scenario": scenario,
            "selection_split": split,
            "sample_identity_sha256": sample_hash,
            "checkpoint_sha256": checkpoint_hash,
            "prediction_schema_version": PREDICTION_SCHEMA_VERSION,
            **primary_contract,
        }
    )
    artifact: dict[str, Any] = {
        "prediction_schema_version": PREDICTION_SCHEMA_VERSION,
        "created_by": "scripts/predict.py",
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_sha256": checkpoint_hash,
        "processed_dir": str(processed_path),
        "scenario": scenario,
        "selection_split": split,
        "split_manifest_path": str(split_path) if split_path is not None else None,
        "split_manifest_sha256": split_manifest_hash,
        "n_samples": len(frame),
        "sample_identity_sha256": sample_hash,
        "required_ood_columns": list(OOD_REQUIRED_COLUMNS),
        "primary_contract": primary_contract,
    }
    return frame, artifact


def _primary_contract(context: dict[str, Any]) -> dict[str, Any]:
    checkpoint = context.get("checkpoint", {})
    cfg = context.get("train_config") or checkpoint.get("config") or {}
    protocol = cfg.get("protocol", {})
    data = cfg.get("data", {})
    encoder = cfg.get("model", {}).get("condition_encoder", {})
    conditioning = cfg.get("model", {}).get("conditioning", {})
    runtime = cfg.get("runtime_contract", {})
    inference_environment = _environment_contract()
    planned = copy.deepcopy(cfg)
    planned.pop("runtime_contract", None)
    completion_sha256 = _validate_training_completion(context, planned)
    manifest_sha256, reveal_receipt_sha256 = _authorization_contract(context)
    return {
        "execution_job_id": (
            protocol.get("execution_job_id")
            or protocol.get("development_grid_job_id")
            or "none"
        ),
        "execution_contract_sha256": protocol.get("execution_contract_sha256") or "none",
        "execution_manifest_content_sha256": manifest_sha256,
        "lockbox_reveal_receipt_sha256": reveal_receipt_sha256,
        "planned_config_sha256": resolved_config_sha256(planned),
        "training_completion_sha256": completion_sha256 or "none",
        "primary_ablation_role": protocol.get("primary_ablation_role"),
        "primary_fairness_hash": protocol.get("primary_fairness_hash"),
        "metadata_contract_version": protocol.get("metadata_contract_version"),
        "conditioning_architecture": conditioning.get(
            "architecture", "prompt_adapter_v1"
        ),
        "external_metadata_mode": encoder.get("external_metadata_mode"),
        "optimization_seed": cfg.get("seed"),
        "split_seed": data.get("split_seed", 42),
        "n_folds": data.get("n_folds"),
        "fold_index": data.get("fold_index"),
        "parameter_schema_sha256": runtime.get("parameter_schema_sha256"),
        "initial_trainable_state_sha256": runtime.get("initial_trainable_state_sha256"),
        "split_assignment_sha256": runtime.get("split_assignment_sha256"),
        "vocabulary_sha256": runtime.get("vocabulary_sha256"),
        "asset_provenance_sha256": runtime.get("asset_provenance_sha256"),
        "execution_phase": runtime.get("execution_phase"),
        "analysis_plan_sha256": runtime.get("analysis_plan_sha256"),
        "cohort_roles_sha256": runtime.get("cohort_roles_sha256"),
        "cohort_sha256": runtime.get("cohort_sha256"),
        "outer_test_access_during_training": runtime.get(
            "outer_test_access_during_training"
        ),
        "source_commit_sha": runtime.get("source_commit_sha"),
        "source_dirty": runtime.get("source_dirty"),
        "source_tree_sha256": runtime.get("source_tree_sha256"),
        "environment_sha256": runtime.get("environment_sha256"),
        "inference_environment_sha256": _sha256_json(inference_environment),
        "checkpoint_role": checkpoint.get("checkpoint_role"),
        "checkpoint_selection_split": checkpoint.get("selection_split") or "none",
        "checkpoint_selection_metric": checkpoint.get("selection_metric") or "none",
    }


def _authorization_contract(context: dict[str, Any]) -> tuple[str, str]:
    """Bind governed predictions to their immutable outcome-access authorization."""

    eval_config = context.get("eval_config") or {}
    authorization = eval_config.get("confirmatory_authorization")
    manifest_field = "execution_manifest"
    receipt_field = "lockbox_reveal_receipt"
    digest_field = "manifest_content_sha256"
    if not isinstance(authorization, dict):
        authorization = eval_config.get("development_authorization")
        manifest_field = "grid_manifest"
        receipt_field = "decision_receipt"
        digest_field = "grid_content_sha256"
    if not isinstance(authorization, dict):
        return "none", "none"
    manifest_path = Path(str(authorization.get(manifest_field, ""))).expanduser()
    receipt_path = Path(str(authorization.get(receipt_field, ""))).expanduser()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    recorded = manifest.pop(digest_field, None)
    if not isinstance(recorded, str) or recorded != sha256_json(manifest):
        raise ValueError("Prediction authorization manifest digest is invalid or stale.")
    if not receipt_path.is_file():
        raise ValueError("Prediction authorization receipt is missing.")
    recorded_receipt = authorization.get(f"{receipt_field}_sha256")
    if recorded_receipt is not None and recorded_receipt != _file_sha256(receipt_path):
        raise ValueError("Prediction authorization receipt hash is stale.")
    return recorded, _file_sha256(receipt_path)


def _validate_training_completion(context: dict[str, Any], planned_cfg: dict) -> str | None:
    checkpoint = context.get("checkpoint", {})
    cfg = context.get("train_config") or checkpoint.get("config") or {}
    protocol = cfg.get("protocol", {})
    phase = cfg.get("runtime_contract", {}).get("execution_phase")
    confirmatory = (
        phase == "confirmatory_training"
        and protocol.get("primary_ablation_role")
        in {"A0_eeg_only", "A2_structured_condition_prompt"}
        and bool(protocol.get("execution_job_id"))
        and bool(protocol.get("execution_contract_sha256"))
    )
    physical_development = (
        phase == "development"
        and cfg.get("model", {}).get("conditioning", {}).get("architecture")
        == "physical_hybrid_v1"
        and bool(protocol.get("development_grid_job_id"))
        and bool(protocol.get("development_grid_manifest"))
    )
    if not confirmatory and not physical_development:
        return None
    checkpoint_path = Path(context["checkpoint_path"])
    completion_path = checkpoint_path.parent / "training_completion.json"
    if not completion_path.is_file():
        raise ValueError("Governed prediction requires a training-completion receipt.")
    receipt = json.loads(completion_path.read_text(encoding="utf-8"))
    expected_config = resolved_config_sha256(planned_cfg)
    expected_checkpoint = _file_sha256(checkpoint_path)
    last_checkpoint_path = checkpoint_path.parent / "last.pt"
    artifact_names = ["metrics_train.csv", "split.csv", "development_control.json"]
    if physical_development and (
        checkpoint_path.parent / "development_control_donors.csv"
    ).is_file():
        artifact_names.append("development_control_donors.csv")
    expected_artifacts = {
        name: _file_sha256(checkpoint_path.parent / name)
        for name in artifact_names
        if (checkpoint_path.parent / name).is_file()
    }
    expected_epochs = int(cfg.get("train", {}).get("epochs", -1))
    runtime_sha256 = sha256_json(cfg.get("runtime_contract") or {})
    resume_generation = receipt.get("resume_generation")
    resume_state_sha256 = str(receipt.get("resume_state_sha256") or "")
    valid_resume_state = len(resume_state_sha256) == 64 and all(
        character in "0123456789abcdef" for character in resume_state_sha256
    )
    if (
        receipt.get("schema") != "cfeg.training-completion.v1"
        or not last_checkpoint_path.is_file()
        or receipt.get("status") != "completed"
        or receipt.get("completed_epoch") != expected_epochs
        or receipt.get("stop_reason") != "max_epochs"
        or receipt.get("resolved_config_sha256") != expected_config
        or receipt.get("runtime_contract_sha256") != runtime_sha256
        or receipt.get("checkpoint_sha256")
        != {
            checkpoint_path.name: expected_checkpoint,
            "last.pt": (
                _file_sha256(last_checkpoint_path)
                if last_checkpoint_path.is_file()
                else "missing"
            ),
        }
        or (receipt.get("artifact_sha256") or {}) != expected_artifacts
        or set(expected_artifacts) != set(artifact_names)
        or (checkpoint_path.parent / "metrics_val.csv").exists()
        or not isinstance(resume_generation, int)
        or isinstance(resume_generation, bool)
        or resume_generation < 0
        or not valid_resume_state
        or checkpoint.get("epoch") != expected_epochs
        or checkpoint.get("checkpoint_role")
        != ("confirmatory_fixed_epoch" if confirmatory else "development_fixed_epoch")
        or checkpoint.get("selection_split") is not None
        or checkpoint.get("selection_metric") != "fixed_epoch"
    ):
        raise ValueError("Governed training-completion receipt is invalid or stale.")
    return _file_sha256(completion_path)


def _validate_primary_execution_source(context: dict[str, Any]) -> None:
    checkpoint = context.get("checkpoint", {})
    cfg = context.get("train_config") or checkpoint.get("config") or {}
    protocol = cfg.get("protocol", {})
    conditioning = cfg.get("model", {}).get("conditioning", {})
    governed_physical = (
        conditioning.get("architecture") == "physical_hybrid_v1"
        and bool(protocol.get("development_grid_job_id"))
    )
    if protocol.get("metadata_contract_version") != "0.4-dev" or (
        protocol.get("primary_ablation_role")
        not in {"A0_eeg_only", "A2_structured_condition_prompt"}
        and not governed_physical
    ):
        return
    _validate_current_source_contract(cfg.get("runtime_contract", {}))


def _validate_current_source_contract(contract: dict[str, Any]) -> None:
    current = _source_revision_contract()
    fields = ("source_commit_sha", "source_dirty", "source_tree_sha256")
    differs = any(current.get(field) != contract.get(field) for field in fields)
    confirmatory_dirty = (
        contract.get("execution_phase") == "confirmatory_training"
        and current.get("source_dirty") is not False
    )
    if differs or confirmatory_dirty:
        raise ValueError(
            "Primary inference/analysis must run from the same clean source tree as training."
        )


def _file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_verified_prediction_bundle(
    csv_path: str | Path,
    *,
    expected_recorded_csv_path: str | Path | None = None,
) -> VerifiedPredictionBundle:
    path = Path(csv_path).expanduser().resolve()
    provenance_path = path.with_name(f"{path.stem}_provenance.json")
    if not provenance_path.exists():
        raise FileNotFoundError(f"Prediction provenance sidecar is missing: {provenance_path}.")
    artifact = json.loads(provenance_path.read_text(encoding="utf-8"))
    if artifact.get("prediction_csv_sha256") != _file_sha256(path):
        raise ValueError(f"Prediction CSV changed after provenance was written: {path}.")
    recorded_path = Path(str(artifact.get("prediction_csv", ""))).expanduser().resolve()
    expected_recorded_path = (
        Path(expected_recorded_csv_path).expanduser().resolve()
        if expected_recorded_csv_path is not None
        else path
    )
    if recorded_path != expected_recorded_path:
        raise ValueError("Prediction sidecar points to a different CSV path.")
    frame = pd.read_csv(path, keep_default_na=False)
    if int(artifact.get("n_samples", -1)) != len(frame):
        raise ValueError("Prediction sidecar row count does not match the CSV.")
    expected_sample_hash = _sample_id_identity_sha256(frame["sample_id"].astype(str).tolist())
    if artifact.get("sample_identity_sha256") != expected_sample_hash:
        raise ValueError("Prediction sidecar sample identity does not match the CSV.")
    checkpoint_path = Path(str(artifact.get("checkpoint_path", ""))).expanduser()
    if not checkpoint_path.exists() or artifact.get("checkpoint_sha256") != _file_sha256(
        checkpoint_path
    ):
        raise ValueError("Prediction sidecar no longer resolves to the recorded checkpoint.")
    split_path_value = artifact.get("split_manifest_path")
    if split_path_value is not None:
        split_path = Path(str(split_path_value)).expanduser()
        if not split_path.exists() or artifact.get("split_manifest_sha256") != _file_sha256(
            split_path
        ):
            raise ValueError("Prediction sidecar split manifest fingerprint is invalid.")
    contract = artifact.get("primary_contract")
    if not isinstance(contract, dict):
        raise TypeError("Prediction sidecar has no primary contract mapping.")
    for column in PRIMARY_CONTRACT_COLUMNS:
        if column not in frame.columns or column not in contract:
            raise ValueError(f"Prediction bundle is missing primary contract field {column}.")
        values = frame[column]
        if values.isna().any() or values.nunique(dropna=False) != 1:
            raise ValueError(f"Prediction CSV contract field {column} is not constant.")
        if not _contract_values_equal(values.iloc[0], contract[column]):
            raise ValueError(
                f"Prediction CSV and sidecar disagree on primary contract field {column}."
            )
    _validate_current_source_contract(contract)
    return VerifiedPredictionBundle(
        frame=frame,
        artifact=artifact,
        path=path,
        provenance_path=provenance_path,
        provenance_sha256=_file_sha256(provenance_path),
    )


def _contract_values_equal(left: Any, right: Any) -> bool:
    if isinstance(left, np.generic):
        left = left.item()
    if isinstance(right, np.generic):
        right = right.item()
    if isinstance(left, bool) or isinstance(right, bool):
        return str(left).strip().lower() == str(right).strip().lower()
    return str(left) == str(right)
