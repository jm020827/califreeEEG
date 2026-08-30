from __future__ import annotations

import copy
import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from cfeg.utils.config import merge_overrides

PRIMARY_ROLES = ("A0_eeg_only", "A2_structured_condition_prompt")
PRIMARY_MODES = {
    "A0_eeg_only": "null",
    "A2_structured_condition_prompt": "observed",
}
PRIMARY_CONDITIONING_ARCHITECTURE = "physical_hybrid_v1"
PROJECT_ROOT = Path(__file__).resolve().parents[2]


def sha256_json(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def resolved_config_sha256(cfg: Mapping[str, Any]) -> str:
    """Bind every resolved per-run value, including seed, fold, paths, and role."""

    return sha256_json(cfg)


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resolve_plan_execution_path(value: str | Path) -> Path:
    """Resolve a plan-owned execution path relative to the repository, not the CWD."""

    path = Path(value).expanduser()
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path.resolve()


def validate_canonical_execution_paths(
    plan: Mapping[str, Any],
    *,
    manifest_path: str | Path,
    output_root: str | Path | None = None,
) -> tuple[Path, Path]:
    manifest_value = plan.get("execution_manifest_path")
    output_value = plan.get("execution_output_root")
    if not manifest_value or not output_value:
        raise ValueError("Analysis plan has no canonical execution manifest/output paths.")
    expected_manifest = resolve_plan_execution_path(str(manifest_value))
    expected_output = resolve_plan_execution_path(str(output_value))
    if resolve_plan_execution_path(manifest_path) != expected_manifest:
        raise ValueError("Execution manifest path differs from the canonical analysis plan path.")
    if output_root is not None and resolve_plan_execution_path(output_root) != expected_output:
        raise ValueError("Execution output root differs from the canonical analysis plan path.")
    return expected_manifest, expected_output


def confirmatory_training_grid_sha256(manifest: Mapping[str, Any]) -> str:
    """Fingerprint all six immutable inputs required before lockbox reveal."""

    rows: list[dict[str, str]] = []
    for job in sorted(manifest.get("jobs") or [], key=lambda value: str(value["job_id"])):
        root = Path(str(job["output_dir"]))
        paths = {
            "training_completion_sha256": root / "training_completion.json",
            "final_checkpoint_sha256": root / "final.pt",
            "last_checkpoint_sha256": root / "last.pt",
            "train_metrics_sha256": root / "metrics_train.csv",
            "split_manifest_sha256": root / "split.csv",
            "development_control_sha256": root / "development_control.json",
        }
        missing = [str(path) for path in paths.values() if not path.is_file()]
        if missing:
            raise ValueError(
                f"Confirmatory training grid is incomplete for {job['job_id']}: {missing}."
            )
        rows.append(
            {
                "job_id": str(job["job_id"]),
                **{name: file_sha256(path) for name, path in paths.items()},
            }
        )
    expected_count = int(manifest.get("expected_job_count", -1))
    if expected_count < 1 or len(rows) != expected_count:
        raise ValueError("Confirmatory training grid count differs from the manifest.")
    return sha256_json(
        {
            "schema": "cfeg.confirmatory-training-grid.v1",
            "jobs": rows,
        }
    )


def validate_confirmatory_training_artifacts(
    job: Mapping[str, Any],
    manifest: Mapping[str, Any],
    checkpoint: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate one fixed-epoch training completion before any lockbox access."""

    root = Path(str(job["output_dir"]))
    completion_path = root / "training_completion.json"
    final_path = root / "final.pt"
    last_path = root / "last.pt"
    metrics_path = root / "metrics_train.csv"
    split_path = root / "split.csv"
    control_path = root / "development_control.json"
    required = (
        completion_path,
        final_path,
        last_path,
        metrics_path,
        split_path,
        control_path,
    )
    missing = [str(path) for path in required if not path.is_file()]
    if missing or (root / "metrics_val.csv").exists():
        raise ValueError(
            f"Confirmatory training artifacts are incomplete for {job['job_id']}: {missing}."
        )
    try:
        completion = json.loads(completion_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"Confirmatory completion receipt is invalid for {job['job_id']}."
        ) from exc

    cfg = checkpoint.get("config") or {}
    runtime_contract = cfg.get("runtime_contract") or {}
    planned = copy.deepcopy(dict(cfg))
    planned.pop("runtime_contract", None)
    protocol = cfg.get("protocol") or {}
    expected_epochs = int((job.get("config") or {}).get("train", {}).get("epochs", -1))
    resume_generation = completion.get("resume_generation")
    resume_state_sha256 = str(completion.get("resume_state_sha256") or "")
    valid_resume_state = len(resume_state_sha256) == 64 and all(
        character in "0123456789abcdef" for character in resume_state_sha256
    )
    expected_completion = {
        "schema": "cfeg.training-completion.v1",
        "status": "completed",
        "completed_epoch": expected_epochs,
        "stop_reason": "max_epochs",
        "resolved_config_sha256": job["resolved_config_sha256"],
        "runtime_contract_sha256": sha256_json(runtime_contract),
        "checkpoint_sha256": {
            "final.pt": file_sha256(final_path),
            "last.pt": file_sha256(last_path),
        },
        "artifact_sha256": {
            "metrics_train.csv": file_sha256(metrics_path),
            "split.csv": file_sha256(split_path),
            "development_control.json": file_sha256(control_path),
        },
    }
    mismatched = {
        key: {"expected": value, "observed": completion.get(key)}
        for key, value in expected_completion.items()
        if completion.get(key) != value
    }
    if (
        not isinstance(resume_generation, int)
        or isinstance(resume_generation, bool)
        or resume_generation < 0
    ):
        mismatched["resume_generation"] = {
            "expected": "non-negative exact-resume generation",
            "observed": resume_generation,
        }
    if not valid_resume_state:
        mismatched["resume_state_sha256"] = {
            "expected": "lowercase 64-character SHA-256",
            "observed": completion.get("resume_state_sha256"),
        }
    checkpoint_expected = {
        "checkpoint_role": "confirmatory_fixed_epoch",
        "selection_split": None,
        "selection_metric": "fixed_epoch",
        "epoch": expected_epochs,
    }
    mismatched.update(
        {
            f"checkpoint.{key}": {"expected": value, "observed": checkpoint.get(key)}
            for key, value in checkpoint_expected.items()
            if checkpoint.get(key) != value
        }
    )
    if resolved_config_sha256(planned) != job["resolved_config_sha256"]:
        mismatched["checkpoint.config"] = {
            "expected": job["resolved_config_sha256"],
            "observed": resolved_config_sha256(planned),
        }
    if (
        protocol.get("execution_job_id") != job["job_id"]
        or protocol.get("execution_contract_sha256") != manifest["execution_contract_sha256"]
    ):
        mismatched["checkpoint.execution_contract"] = {
            "expected": {
                "job_id": job["job_id"],
                "execution_contract_sha256": manifest["execution_contract_sha256"],
            },
            "observed": {
                "job_id": protocol.get("execution_job_id"),
                "execution_contract_sha256": protocol.get("execution_contract_sha256"),
            },
        }
    if mismatched:
        raise ValueError(
            f"Confirmatory completion receipt failed for {job['job_id']}: {mismatched}."
        )
    return completion


def normalized_primary_fairness_config(cfg: Mapping[str, Any]) -> dict[str, Any]:
    """Remove execution coordinates and normalize the sole A0/A2 treatment axis.

    The resulting digest is shared by all six confirmatory runs. Each run still has
    a separate full resolved-config digest, so normalization cannot hide a changed
    seed, fold, output path, or tracking identity from provenance.
    """

    normalized = copy.deepcopy(dict(cfg))
    normalized["seed"] = "<optimization-seed>"
    normalized["run_name"] = "<run-name>"
    normalized["output_dir"] = "<output-dir>"
    normalized.pop("runtime_contract", None)

    data = normalized.setdefault("data", {})
    data["fold_index"] = "<outer-fold>"

    encoder = normalized.setdefault("model", {}).setdefault("condition_encoder", {})
    encoder["external_metadata_mode"] = "<treatment-access>"
    encoder["force_missing"] = bool(encoder.get("force_missing", False))

    protocol = normalized.setdefault("protocol", {})
    protocol["primary_ablation_role"] = "<primary-role>"
    protocol["primary_fairness_hash"] = "<primary-fairness-hash>"
    protocol["execution_job_id"] = "<execution-job>"
    protocol["execution_contract_sha256"] = "<execution-contract>"
    protocol.pop("execution_manifest", None)

    # Tracking labels and storage identity have no effect on the scientific recipe.
    # They remain covered by resolved_config_sha256 for each concrete run.
    normalized.pop("tracking", None)
    return normalized


def validate_primary_pair_and_hash(
    base: Mapping[str, Any],
    variants: Mapping[str, Mapping[str, Any]],
    *,
    enforce_frozen_architecture: bool = True,
) -> str | None:
    if any(name not in variants for name in PRIMARY_ROLES):
        return None
    configs = {
        name: merge_overrides(
            copy.deepcopy(dict(base)),
            [f"{key}={value!r}" for key, value in variants[name].items()],
        )
        for name in PRIMARY_ROLES
    }
    modes = {
        name: configs[name]
        .get("model", {})
        .get("condition_encoder", {})
        .get("external_metadata_mode", "observed")
        for name in PRIMARY_ROLES
    }
    if modes != PRIMARY_MODES:
        raise ValueError(
            f"Primary A0/A2 must differ by external_metadata_mode=null|observed; got {modes}."
        )
    versions = {
        name: str(configs[name].get("protocol", {}).get("metadata_contract_version", "legacy"))
        for name in PRIMARY_ROLES
    }
    if set(versions.values()) != {"0.4-dev"}:
        raise ValueError(
            "Primary A0/A2 must both use protocol.metadata_contract_version=0.4-dev; "
            f"got {versions}."
        )
    architecture_contracts = {
        role: _primary_architecture_contract(cfg) for role, cfg in configs.items()
    }
    expected_architecture = _expected_primary_architecture_contract()
    if enforce_frozen_architecture and any(
        contract != expected_architecture for contract in architecture_contracts.values()
    ):
        raise ValueError(
            "Primary A0/A2 must use the frozen physical_hybrid_v1 neutral-fusion "
            f"contract; got {architecture_contracts}."
        )
    signatures = {
        name: sha256_json(normalized_primary_fairness_config(cfg)) for name, cfg in configs.items()
    }
    if len(set(signatures.values())) != 1:
        raise ValueError(
            "Primary A0/A2 configs differ outside external_metadata_mode. "
            f"Fairness signatures: {signatures}."
        )
    return signatures[PRIMARY_ROLES[0]]


def _expected_primary_architecture_contract() -> dict[str, Any]:
    return {
        "architecture": PRIMARY_CONDITIONING_ARCHITECTURE,
        "backbone_name": "spectral_transformer",
        "spectral_min_frequency_hz": 6.0,
        "spectral_max_frequency_hz": 60.0,
        "conditioning_hidden_dim": 64,
        "fields": ["electrode_type"],
        "n_prompt_tokens": 0,
        "include_continuous": True,
        "include_channels": False,
        "force_missing": False,
        "adapter_enabled": False,
        "latent_enabled": False,
        "query_film_enabled": True,
        "query_max_scale_delta": 0.25,
        "query_max_shift": 0.25,
        "external_film_enabled": True,
        "external_max_scale_delta": 0.25,
        "external_max_shift": 0.25,
        "channel_quality_enabled": True,
        "channel_quality_source": "channel_impedance",
        "channel_quality_placement": "pre_channel_embedding",
        "channel_max_gain_delta": 0.25,
    }


def _primary_architecture_contract(cfg: Mapping[str, Any]) -> dict[str, Any]:
    model = cfg.get("model", {})
    encoder = model.get("condition_encoder", {})
    conditioning = model.get("conditioning", {})
    query_film = conditioning.get("common_query_film", {})
    external_film = conditioning.get("external_global_film", {})
    channel_quality = conditioning.get("channel_quality", {})
    backbone = model.get("backbone", {})
    return {
        "architecture": conditioning.get("architecture"),
        "backbone_name": backbone.get("name"),
        "spectral_min_frequency_hz": float(backbone.get("min_frequency_hz", -1.0)),
        "spectral_max_frequency_hz": float(backbone.get("max_frequency_hz", -1.0)),
        "conditioning_hidden_dim": int(conditioning.get("hidden_dim", -1)),
        "fields": list(encoder.get("fields") or []),
        "n_prompt_tokens": int(encoder.get("n_prompt_tokens", -1)),
        "include_continuous": bool(encoder.get("include_continuous", False)),
        "include_channels": bool(encoder.get("include_channels", True)),
        "force_missing": bool(encoder.get("force_missing", False)),
        "adapter_enabled": bool(model.get("adapter", {}).get("enabled", True)),
        "latent_enabled": bool(model.get("latent", {}).get("enabled", True)),
        "query_film_enabled": bool(query_film.get("enabled", False)),
        "query_max_scale_delta": float(query_film.get("max_scale_delta", -1.0)),
        "query_max_shift": float(query_film.get("max_shift", -1.0)),
        "external_film_enabled": bool(external_film.get("enabled", False)),
        "external_max_scale_delta": float(external_film.get("max_scale_delta", -1.0)),
        "external_max_shift": float(external_film.get("max_shift", -1.0)),
        "channel_quality_enabled": bool(channel_quality.get("enabled", False)),
        "channel_quality_source": channel_quality.get("source"),
        "channel_quality_placement": channel_quality.get("placement"),
        "channel_max_gain_delta": float(channel_quality.get("max_gain_delta", -1.0)),
    }


def build_primary_execution_manifest(
    *,
    base: Mapping[str, Any],
    variants: Mapping[str, Mapping[str, Any]],
    plan: Mapping[str, Any],
    output_root: str | Path,
    manifest_path: str | Path | None = None,
    plan_sha256: str,
    source_contract: Mapping[str, Any],
) -> dict[str, Any]:
    validate_canonical_execution_paths(
        plan,
        manifest_path=(manifest_path or str(plan.get("execution_manifest_path", ""))),
        output_root=output_root,
    )
    fairness_hash = validate_primary_pair_and_hash(base, variants)
    if fairness_hash is None:  # pragma: no cover - guarded by normal suite validation
        raise ValueError("The ablation suite has no complete A0/A2 primary pair.")
    execution_contract_sha256 = sha256_json(
        {
            "schema": "cfeg.primary-execution-contract.v2",
            "analysis_plan_sha256": plan_sha256,
            "plan_status": plan.get("status"),
            "primary_fairness_hash": fairness_hash,
            "source_contract": dict(source_contract),
            "optimization_seeds": list(plan["optimization_seeds"]),
            "fold_indices": list(plan["fold_indices"]),
            "roles": list(PRIMARY_ROLES),
            "allocation_decision_id": plan["allocation_decision_id"],
            "allocation_sha256": plan["allocation_sha256"],
            "execution_manifest_path": str(plan["execution_manifest_path"]),
            "execution_output_root": str(plan["execution_output_root"]),
        }
    )
    jobs: list[dict[str, Any]] = []
    output_root = Path(str(plan["execution_output_root"]))
    manifest_path = Path(str(plan["execution_manifest_path"]))
    for seed in plan["optimization_seeds"]:
        for fold in plan["fold_indices"]:
            pair_id = f"seed{int(seed)}-fold{int(fold)}"
            for role in PRIMARY_ROLES:
                cfg = merge_overrides(
                    copy.deepcopy(dict(base)),
                    [f"{key}={value!r}" for key, value in variants[role].items()],
                )
                cfg["seed"] = int(seed)
                cfg.setdefault("data", {})["fold_index"] = int(fold)
                cfg["run_name"] = f"primary-{pair_id}-{role}"
                cfg["output_dir"] = str(output_root / pair_id / role)
                protocol = cfg.setdefault("protocol", {})
                protocol["primary_ablation_role"] = role
                protocol["primary_fairness_hash"] = fairness_hash
                job_id = f"{pair_id}-{role}"
                protocol["execution_job_id"] = job_id
                protocol["execution_contract_sha256"] = execution_contract_sha256
                protocol["execution_manifest"] = str(manifest_path)
                tags = list(
                    cfg.setdefault("tracking", {}).setdefault("wandb", {}).get("tags") or []
                )
                cfg["tracking"]["wandb"]["tags"] = sorted(
                    set(tags + ["confirmatory-primary", pair_id, role])
                )
                jobs.append(
                    {
                        "job_id": job_id,
                        "pair_id": pair_id,
                        "seed": int(seed),
                        "fold_index": int(fold),
                        "role": role,
                        "output_dir": cfg["output_dir"],
                        "checkpoint_path": str(Path(cfg["output_dir"]) / "final.pt"),
                        "prediction_csv": str(output_root.parent / "predictions" / f"{job_id}.csv"),
                        "resolved_config_sha256": resolved_config_sha256(cfg),
                        "normalized_fairness_sha256": sha256_json(
                            normalized_primary_fairness_config(cfg)
                        ),
                        "config": cfg,
                    }
                )
    manifest = {
        "schema": "cfeg.primary-execution-manifest.v1",
        "plan_status": plan.get("status"),
        "execution_allowed": plan.get("status") == "frozen",
        "analysis_plan_sha256": plan_sha256,
        "primary_fairness_hash": fairness_hash,
        "execution_contract_sha256": execution_contract_sha256,
        "allocation_decision_id": plan["allocation_decision_id"],
        "allocation_sha256": plan["allocation_sha256"],
        "execution_manifest_path": str(manifest_path),
        "execution_output_root": str(output_root),
        "source_contract": dict(source_contract),
        "expected_job_count": len(plan["optimization_seeds"])
        * len(plan["fold_indices"])
        * len(PRIMARY_ROLES),
        "jobs": jobs,
    }
    validate_primary_execution_manifest(manifest, plan=plan)
    manifest["manifest_content_sha256"] = sha256_json(manifest)
    return manifest


def validate_primary_execution_manifest(
    manifest: Mapping[str, Any],
    *,
    plan: Mapping[str, Any],
    analysis_plan_sha256: str | None = None,
    manifest_path: str | Path | None = None,
) -> None:
    if manifest.get("schema") != "cfeg.primary-execution-manifest.v1":
        raise ValueError("Unknown primary execution-manifest schema.")
    if manifest.get("plan_status") != plan.get("status") or manifest.get("execution_allowed") != (
        plan.get("status") == "frozen"
    ):
        raise ValueError("Execution manifest status does not match the analysis plan.")
    if (
        analysis_plan_sha256 is not None
        and manifest.get("analysis_plan_sha256") != analysis_plan_sha256
    ):
        raise ValueError("Execution manifest analysis-plan digest is stale.")
    if manifest.get("allocation_decision_id") != plan.get("allocation_decision_id") or manifest.get(
        "allocation_sha256"
    ) != plan.get("allocation_sha256"):
        raise ValueError("Execution manifest allocation binding differs from the plan.")
    expected_path_fields = {
        "execution_manifest_path": str(plan.get("execution_manifest_path", "")),
        "execution_output_root": str(plan.get("execution_output_root", "")),
    }
    if any(manifest.get(key) != value for key, value in expected_path_fields.items()):
        raise ValueError("Execution manifest paths differ from the canonical analysis plan paths.")
    validate_canonical_execution_paths(
        plan,
        manifest_path=(manifest_path or str(manifest.get("execution_manifest_path", ""))),
        output_root=str(manifest.get("execution_output_root", "")),
    )
    expected = {
        (int(seed), int(fold), role)
        for seed in plan["optimization_seeds"]
        for fold in plan["fold_indices"]
        for role in PRIMARY_ROLES
    }
    jobs = list(manifest.get("jobs") or [])
    observed = {(int(job["seed"]), int(job["fold_index"]), str(job["role"])) for job in jobs}
    if len(jobs) != len(expected) or observed != expected:
        raise ValueError(
            "Execution manifest is not the exact seed x fold x A0/A2 Cartesian product."
        )
    if manifest.get("expected_job_count") != len(expected):
        raise ValueError("Execution manifest expected-job count is inconsistent.")
    job_ids = [str(job.get("job_id")) for job in jobs]
    outputs = [str(job.get("output_dir")) for job in jobs]
    if len(set(job_ids)) != len(jobs) or len(set(outputs)) != len(jobs):
        raise ValueError("Execution manifest job IDs and output directories must be unique.")
    fairness = str(manifest.get("primary_fairness_hash", ""))
    if len(fairness) != 64:
        raise ValueError("Execution manifest primary fairness hash is invalid.")
    if plan.get("status") == "frozen" and plan.get("primary_fairness_hash") != fairness:
        raise ValueError("Frozen analysis plan does not bind the execution fairness hash.")
    expected_execution_contract = sha256_json(
        {
            "schema": "cfeg.primary-execution-contract.v2",
            "analysis_plan_sha256": manifest.get("analysis_plan_sha256"),
            "plan_status": plan.get("status"),
            "primary_fairness_hash": fairness,
            "source_contract": dict(manifest.get("source_contract") or {}),
            "optimization_seeds": list(plan["optimization_seeds"]),
            "fold_indices": list(plan["fold_indices"]),
            "roles": list(PRIMARY_ROLES),
            "allocation_decision_id": plan["allocation_decision_id"],
            "allocation_sha256": plan["allocation_sha256"],
            "execution_manifest_path": plan["execution_manifest_path"],
            "execution_output_root": plan["execution_output_root"],
        }
    )
    if manifest.get("execution_contract_sha256") != expected_execution_contract:
        raise ValueError("Execution manifest contract digest does not recompute.")
    for job in jobs:
        expected_pair_id = f"seed{int(job['seed'])}-fold{int(job['fold_index'])}"
        expected_job_id = f"{expected_pair_id}-{job['role']}"
        if job.get("pair_id") != expected_pair_id or job.get("job_id") != expected_job_id:
            raise ValueError("Execution manifest pair/job identity does not match its coordinates.")
        if resolved_config_sha256(job["config"]) != job.get("resolved_config_sha256"):
            raise ValueError(f"Resolved config digest mismatch for {job.get('job_id')}.")
        cfg = job["config"]
        recomputed_fairness = sha256_json(normalized_primary_fairness_config(cfg))
        if (
            job.get("normalized_fairness_sha256") != recomputed_fairness
            or recomputed_fairness != fairness
        ):
            raise ValueError(f"Primary fairness digest does not recompute for {job.get('job_id')}.")
        role = str(job["role"])
        mode = (
            cfg.get("model", {})
            .get("condition_encoder", {})
            .get("external_metadata_mode", "observed")
        )
        if mode != PRIMARY_MODES[role]:
            raise ValueError(
                f"Primary treatment mode mismatch for {job.get('job_id')}: "
                f"expected {PRIMARY_MODES[role]!r}, got {mode!r}."
            )
        architecture = _primary_architecture_contract(cfg)
        if architecture != _expected_primary_architecture_contract():
            raise ValueError(
                f"Primary architecture contract mismatch for {job.get('job_id')}: {architecture}."
            )
        if int(cfg.get("seed")) != int(job["seed"]) or int(
            cfg.get("data", {}).get("fold_index")
        ) != int(job["fold_index"]):
            raise ValueError(f"Execution coordinates mismatch for {job.get('job_id')}.")
        protocol = cfg.get("protocol", {})
        if (
            cfg.get("output_dir") != job.get("output_dir")
            or cfg.get("run_name") != f"primary-{expected_job_id}"
            or protocol.get("primary_ablation_role") != job["role"]
            or protocol.get("primary_fairness_hash") != fairness
            or protocol.get("execution_job_id") != job["job_id"]
            or protocol.get("execution_contract_sha256")
            != manifest.get("execution_contract_sha256")
            or protocol.get("execution_manifest") != manifest.get("execution_manifest_path")
        ):
            raise ValueError(f"Primary role/fairness binding mismatch for {job.get('job_id')}.")
        expected_output = (
            Path(str(plan["execution_output_root"])) / str(job["pair_id"]) / str(job["role"])
        )
        expected_checkpoint = str(expected_output / "final.pt")
        expected_prediction = str(
            Path(str(plan["execution_manifest_path"])).parent
            / "predictions"
            / f"{job['job_id']}.csv"
        )
        if (
            job.get("output_dir") != str(expected_output)
            or job.get("checkpoint_path") != expected_checkpoint
            or job.get("prediction_csv") != expected_prediction
        ):
            raise ValueError(f"Execution artifact paths mismatch for {job.get('job_id')}.")
