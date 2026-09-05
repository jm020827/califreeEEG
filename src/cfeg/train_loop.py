from __future__ import annotations

import copy
import hashlib
import json
import os
import platform
import time
from functools import partial
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader, Subset

from cfeg.data.collate import build_vocabularies, collate_eeg, metadata_collate_kwargs
from cfeg.data.datasets import EEGProcessedDataset
from cfeg.data.loader import resolve_loader_settings
from cfeg.data.metadata_controls import (
    DEVELOPMENT_CONTROL_NONE,
    DevelopmentControlPlan,
    build_development_control_plan,
    normalize_development_control,
)
from cfeg.data.preprocess import CanonicalChannelMap
from cfeg.data.query_reliability_corruption import (
    apply_query_reliability_training_mixture,
)
from cfeg.data.schema import load_manifest
from cfeg.data.splits import (
    make_confirmatory_lockbox_split,
    make_cross_condition_split,
    make_cross_dataset_split,
    make_cross_subject_fold_split,
    make_cross_subject_split,
    make_cross_subject_train_val_split,
    make_development_subject_fold_split,
    make_fixed_subject_partition,
    make_joint_subject_condition_split,
)
from cfeg.data.transforms import make_two_views
from cfeg.execution_manifest import resolved_config_sha256
from cfeg.governance import (
    CohortBinding,
    GovernanceError,
    ResearchAccess,
    bind_cohort,
    current_source_revision_contract,
    resolve_research_access,
    validate_physical_development_training_authorization,
)
from cfeg.losses import kl_normal, representation_consistency_loss, symmetric_kl_logits
from cfeg.metrics import classification_metrics, itr_bits_per_min
from cfeg.models.full_model import ConditionedEEGDecoder
from cfeg.query_reliability_contract import (
    query_reliability_contract_marker_present,
    reject_query_reliability_beta_lockbox_access,
    validate_query_reliability_training_preflight,
)
from cfeg.reliability_contract import validate_reliability_training_preflight
from cfeg.runtime import RuntimeMeasurement, resolve_device
from cfeg.seed import seed_everything
from cfeg.utils.checkpoint import (
    capture_rng_state,
    load_checkpoint_model_state,
    load_resume_state,
    restore_rng_state,
    save_checkpoint,
    save_json,
    save_resume_state,
)
from cfeg.utils.config import save_config
from cfeg.utils.params import count_parameters

_RELIABILITY_ORCHESTRATOR_AUTHORIZATION = object()


def run_training(
    cfg: dict,
    *,
    dry_run: bool = False,
    resume_exact: bool = False,
    _test_crash_after_epoch_commit: int | None = None,
    _artifact_output_dir: str | Path | None = None,
    _reliability_orchestrator_authorization: object | None = None,
) -> dict:
    architecture = cfg.get("model", {}).get("conditioning", {}).get("architecture")
    if (
        architecture == "reliability_spatial_v1"
        and _reliability_orchestrator_authorization is not _RELIABILITY_ORCHESTRATOR_AUTHORIZATION
    ):
        raise GovernanceError(
            "reliability_spatial_v1 training is authorized only inside the exact "
            "four-arm Stage-1 orchestrator; direct train.py/API execution is forbidden."
        )
    if query_reliability_contract_marker_present(cfg) and not dry_run:
        raise GovernanceError(
            "query_reliability_spatial_v1 outcome training is blocked while its exact "
            "human-data bundle remains an implementation draft. Only outcome-free "
            "dry-run graph validation is currently permitted."
        )
    if dry_run and resume_exact:
        raise ValueError("dry_run and resume_exact cannot be combined.")
    if resume_exact:
        _apply_exact_determinism()
    run_started = time.perf_counter()
    measurement_holder: dict[str, RuntimeMeasurement] = {}
    try:
        result = _run_training_impl(
            cfg,
            dry_run=dry_run,
            resume_exact=resume_exact,
            test_crash_after_epoch_commit=_test_crash_after_epoch_commit,
            artifact_output_dir=_artifact_output_dir,
            run_started=run_started,
            measurement_holder=measurement_holder,
        )
    except Exception as exc:
        measurement = measurement_holder.get("measurement")
        if measurement is not None:
            attempt_metrics = measurement.finish(status="failed", error=exc)
            output_dir = Path(
                _artifact_output_dir
                if _artifact_output_dir is not None
                else cfg.get("output_dir", "outputs/debug")
            )
            if not dry_run and output_dir.is_dir():
                _record_runtime_attempt(output_dir, attempt_metrics)
        raise
    measurement = measurement_holder["measurement"]
    attempt_metrics = measurement.finish(
        status="dry_run" if dry_run else "completed",
    )
    result["runtime_attempt_metrics"] = attempt_metrics
    result["total_elapsed_time_sec"] = float(time.perf_counter() - run_started)
    if dry_run:
        runtime_metrics = attempt_metrics
    else:
        output_dir = Path(result["output_dir"])
        runtime_metrics, runtime_path = _record_runtime_attempt(output_dir, attempt_metrics)
        result["runtime_metrics_path"] = str(runtime_path)
        result["runtime_metrics_sha256"] = _sha256_file(runtime_path)
    result["runtime_metrics"] = runtime_metrics
    result["elapsed_time_sec"] = runtime_metrics["elapsed_time_sec"]
    return result


def _run_training_impl(
    cfg: dict,
    *,
    dry_run: bool,
    resume_exact: bool,
    test_crash_after_epoch_commit: int | None,
    artifact_output_dir: str | Path | None,
    run_started: float,
    measurement_holder: dict[str, RuntimeMeasurement],
) -> dict:
    validate_reliability_training_preflight(cfg, dry_run=dry_run)
    validate_query_reliability_training_preflight(cfg, dry_run=dry_run, verify_assets=True)
    access = resolve_research_access(cfg)
    development_control, training_external_control = _validate_development_control_access(
        cfg, access
    )
    _resolve_augmentation_channel_sets(cfg)
    if not dry_run and access.governed and access.execution_phase == "development":
        validate_physical_development_training_authorization(cfg)
    output_dir = Path(
        artifact_output_dir
        if artifact_output_dir is not None
        else cfg.get("output_dir", "outputs/debug")
    )
    resume_config_sha256 = _resume_config_sha256(cfg)
    resume_state = None
    resume_loaded_commit = None
    if resume_exact:
        _validate_exact_resume_options(cfg)
        if output_dir.exists() and any(output_dir.iterdir()):
            resume_state, resume_loaded_commit = load_resume_state(output_dir, map_location="cpu")
            _validate_resume_preflight(
                resume_state,
                resolved_config_digest=resume_config_sha256,
                current_source=current_source_revision_contract(),
            )
    elif access.governed and not dry_run:
        _validate_fresh_governed_output(output_dir)
    device = resolve_device(cfg)
    measurement_holder["measurement"] = RuntimeMeasurement.start(
        cfg,
        device,
        run_mode="dry_run_forward_only" if dry_run else "training_and_validation",
    )
    measurement = measurement_holder["measurement"]
    measurement.start_phase("setup")
    seed_everything(int(cfg.get("seed", 42)))
    processed_roots = [Path(value) for value in cfg["data"]["processed_dirs"]]
    asset_manifest = pd.concat([load_manifest(root) for root in processed_roots], ignore_index=True)
    cohort = bind_cohort(asset_manifest, access)
    loader_settings = resolve_loader_settings(cfg["data"], device_type=device.type)
    if loader_settings.preload_hdf5_to_memory:
        preload_manifest = asset_manifest
        if access.governed:
            preload_manifest = asset_manifest.loc[
                asset_manifest["subject_id"].astype(str).isin(cohort.selected_subject_ids)
            ]
        reject_query_reliability_beta_lockbox_access(
            preload_manifest,
            action="whole_cohort_preload_before_split",
        )
    full_ds = EEGProcessedDataset(
        cfg["data"]["processed_dirs"],
        expected_revisions=cfg["data"].get("expected_revisions"),
        expected_protocol=cfg.get("protocol"),
        expected_dataset_counts=cfg["data"].get("expected_dataset_counts"),
        require_audit_receipt=bool(cfg["data"].get("require_audit_receipt", False)),
        allowed_subject_ids=(set(cohort.selected_subject_ids) if access.governed else None),
        persistent_hdf5_handles=loader_settings.persistent_hdf5_handles,
        preload_hdf5_to_memory=loader_settings.preload_hdf5_to_memory,
    )
    manifest = pd.DataFrame([entry[2] for entry in full_ds.entries])
    if tuple(sorted(manifest["subject_id"].astype(str).unique())) != cohort.selected_subject_ids:
        raise GovernanceError("Dataset cohort view differs from the frozen cohort binding.")
    cohort_manifest = manifest.reset_index(drop=True)
    split_seed = _resolve_split_seed(cfg)
    split_name = cfg["data"].get("split", "cross_subject")
    if split_name == "cross_subject":
        split = make_cross_subject_split(
            cohort_manifest,
            seed=split_seed,
            val_ratio=float(cfg["data"].get("val_ratio", 0.2)),
            test_ratio=float(cfg["data"].get("test_ratio", 0.2)),
        )
    elif split_name == "cross_subject_train_val":
        split = make_cross_subject_train_val_split(
            cohort_manifest,
            seed=split_seed,
            val_ratio=float(cfg["data"].get("val_ratio", 0.34)),
        )
    elif split_name == "cross_subject_fold":
        split = make_cross_subject_fold_split(
            cohort_manifest,
            seed=split_seed,
            n_folds=int(cfg["data"].get("n_folds", 5)),
            fold_index=int(cfg["data"].get("fold_index", 0)),
            val_ratio=float(cfg["data"].get("val_ratio", 0.2)),
        )
    elif split_name == "development_subject_fold":
        split = make_development_subject_fold_split(
            cohort_manifest,
            seed=split_seed,
            n_folds=int(cfg["data"].get("n_folds", 3)),
            fold_index=int(cfg["data"].get("fold_index", 0)),
        )
    elif split_name == "confirmatory_lockbox":
        split = make_confirmatory_lockbox_split(
            cohort_manifest,
            training_subject_ids=list(cfg["data"]["training_subject_ids"]),
            lockbox_subject_ids=list(cfg["data"]["lockbox_subject_ids"]),
        )
    elif split_name == "fixed_subject_partition":
        split = make_fixed_subject_partition(
            cohort_manifest,
            training_subject_ids=list(cfg["data"]["training_subject_ids"]),
            validation_subject_ids=list(cfg["data"]["validation_subject_ids"]),
            test_subject_ids=list(cfg["data"]["test_subject_ids"]),
            excluded_subject_ids=list(cfg["data"].get("excluded_subject_ids") or []),
            allow_empty_validation=bool(cfg["data"].get("allow_empty_validation", False)),
        )
    elif split_name == "cross_dataset":
        split = make_cross_dataset_split(
            cohort_manifest,
            train_datasets=list(cfg["data"]["train_datasets"]),
            test_datasets=list(cfg["data"]["test_datasets"]),
            seed=split_seed,
            val_ratio=float(cfg["data"].get("val_ratio", 0.2)),
        )
    elif split_name == "cross_condition":
        split = make_cross_condition_split(
            cohort_manifest,
            train_filter=dict(cfg["data"]["train_filter"]),
            test_filter=dict(cfg["data"]["test_filter"]),
            seed=split_seed,
            val_ratio=float(cfg["data"].get("val_ratio", 0.2)),
        )
    elif split_name == "joint_subject_condition":
        split = make_joint_subject_condition_split(
            cohort_manifest,
            train_filter=dict(cfg["data"]["train_filter"]),
            test_filter=dict(cfg["data"]["test_filter"]),
            seed=split_seed,
            val_ratio=float(cfg["data"].get("val_ratio", 0.2)),
            test_ratio=float(cfg["data"].get("test_ratio", 0.2)),
        )
    else:
        raise ValueError(
            f"Unknown data.split={split_name!r}; use cross_subject, cross_subject_train_val, "
            "development_subject_fold, confirmatory_lockbox, cross_subject_fold, "
            "fixed_subject_partition, cross_dataset, cross_condition, "
            "or joint_subject_condition."
        )
    _validate_primary_split_contract(cfg, manifest, split, access=access, cohort=cohort)
    train_validation_indices = np.concatenate([split.train, split.val])
    reject_query_reliability_beta_lockbox_access(
        manifest.iloc[train_validation_indices],
        action="training_or_checkpoint_selection",
    )
    if access.allow_outer_test_during_training:
        reject_query_reliability_beta_lockbox_access(
            manifest.iloc[np.asarray(split.test, dtype=int)],
            action="outer_test_during_training",
        )
    if bool(cfg["data"].get("preload_train_indices_to_memory", False)):
        if bool(cfg["data"].get("preload_hdf5_to_memory", False)):
            raise ValueError("Choose either whole-cohort preload or train-only preload, not both.")
        full_ds.preload_indices(split.train)
    control_indices = (
        {"train": split.train}
        if training_external_control != DEVELOPMENT_CONTROL_NONE
        else {"train": split.train, "val": split.val, "test": split.test}
    )
    effective_control = (
        training_external_control
        if training_external_control != DEVELOPMENT_CONTROL_NONE
        else development_control
    )
    control_plan = build_development_control_plan(
        manifest,
        control_indices,
        control=effective_control,
        seed=int(cfg.get("protocol", {}).get("development_control_seed", 42)),
    )
    if training_external_control != DEVELOPMENT_CONTROL_NONE:
        control_plan.contract.update(
            {
                "application_splits": ["train"],
                "validation_external_metadata": "correct",
                "model_development_control": development_control,
            }
        )

    n_classes = int(cfg.get("model", {}).get("n_classes", 0))
    # Model construction/training governance may inspect source train+validation
    # labels, but not the final held-out target labels before checkpoint freeze.
    selected_indices = train_validation_indices
    selected_labels = manifest.iloc[selected_indices]["label"].astype(int)
    if len(selected_labels) and (selected_labels.min() < 0 or selected_labels.max() >= n_classes):
        raise ValueError(
            f"Training labels span {selected_labels.min()}..{selected_labels.max()} but "
            f"model.n_classes={n_classes}. Re-run preparation with the correct class map."
        )

    train_rows = (manifest.iloc[int(index)].to_dict() for index in split.train)
    vocab = build_vocabularies(train_rows)
    collate_kwargs = metadata_collate_kwargs(cfg)
    train_collate = partial(
        collate_eeg,
        vocabularies=vocab,
        **collate_kwargs,
        external_metadata_overrides=control_plan.sources_for("train"),
        categorical_metadata_dropout_prob=float(
            cfg.get("augment", {}).get("categorical_metadata_dropout_prob", 0.0)
        ),
    )
    eval_collate = partial(
        collate_eeg,
        vocabularies=vocab,
        **collate_kwargs,
        external_metadata_overrides=(
            None
            if training_external_control != DEVELOPMENT_CONTROL_NONE
            else control_plan.sources_for("val")
        ),
        categorical_metadata_dropout_prob=0.0,
    )
    loader_kwargs = loader_settings.kwargs()
    train_generator = torch.Generator()
    train_generator.manual_seed(int(cfg.get("seed", 42)))
    train_loader = DataLoader(
        Subset(full_ds, split.train.tolist()),
        batch_size=loader_settings.train_batch_size,
        shuffle=True,
        collate_fn=train_collate,
        generator=train_generator,
        **loader_kwargs,
    )
    val_loader = DataLoader(
        Subset(full_ds, split.val.tolist()),
        batch_size=loader_settings.eval_batch_size,
        shuffle=False,
        collate_fn=eval_collate,
        **loader_kwargs,
    )
    test_loader = None
    if access.allow_outer_test_during_training:
        test_collate = partial(
            collate_eeg,
            vocabularies=vocab,
            **collate_kwargs,
            external_metadata_overrides=(
                None
                if training_external_control != DEVELOPMENT_CONTROL_NONE
                else control_plan.sources_for("test")
            ),
            categorical_metadata_dropout_prob=0.0,
        )
        test_loader = DataLoader(
            Subset(full_ds, split.test.tolist()),
            batch_size=loader_settings.eval_batch_size,
            shuffle=False,
            collate_fn=test_collate,
            **loader_kwargs,
        )

    vocab_sizes = {field: len(values) for field, values in vocab.items()}
    model = ConditionedEEGDecoder(cfg, vocab_sizes=vocab_sizes).to(device)

    batch = next(iter(train_loader))
    batch = _to_device(batch, device)
    with torch.no_grad():
        out = model(batch["x"], batch["cond"])
    params = count_parameters(model)
    asset_provenance = _processed_asset_provenance(full_ds.roots)
    runtime_contract = _runtime_contract(
        model,
        manifest,
        split,
        vocab,
        asset_provenance,
        access=access,
        cohort=cohort,
        control_plan=control_plan,
        loader_settings=loader_settings.contract(),
    )
    cfg["runtime_contract"] = runtime_contract
    dry_result = {
        "n_samples": len(full_ds),
        "logits_shape": tuple(out.logits.shape),
        "params": params,
        "runtime_contract": runtime_contract,
        "research_access": access.contract(),
        "cohort": cohort.contract(),
        "development_control": control_plan.contract,
        "elapsed_time_sec": float(time.perf_counter() - run_started),
    }
    initial_diagnostics = _summarize_reliability_aux([out.aux])
    if initial_diagnostics:
        dry_result["reliability_diagnostics"] = initial_diagnostics
    measurement.finish_phase()
    if dry_run:
        return dry_result

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=float(cfg["train"].get("lr", 3e-4)),
        weight_decay=float(cfg["train"].get("weight_decay", 0.01)),
    )
    amp_enabled = bool(cfg["train"].get("amp", True)) and device.type == "cuda"
    eval_amp_enabled = bool(cfg["train"].get("eval_amp", False)) and device.type == "cuda"
    if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
        scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
    else:
        scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled)

    best_acc = -1.0
    patience = int(cfg["train"].get("early_stop_patience", 10))
    save_trainable_only = bool(cfg.get("checkpoint", {}).get("save_trainable_only", True))
    max_epochs = int(cfg["train"].get("epochs", 20))
    selection_policy, validation_epochs = _resolve_training_selection(cfg, max_epochs)
    if selection_policy == "fixed_epoch_blind":
        if access.execution_phase != "development":
            raise GovernanceError(
                "fixed_epoch_blind is restricted to the governed development cohort."
            )
        if access.allow_outer_test_during_training:
            raise GovernanceError(
                "fixed_epoch_blind cannot be combined with outer-test access during training."
            )
    if selection_policy == "fixed_epoch_sealed":
        if access.execution_phase != "confirmatory_training":
            raise GovernanceError(
                "fixed_epoch_sealed is restricted to governed confirmatory training."
            )
        if len(split.val) or access.allow_outer_test_during_training:
            raise GovernanceError(
                "fixed_epoch_sealed requires an empty validation split and sealed outer test."
            )
    stale = 0
    metrics_rows: list[dict] = []
    global_step = 0
    completed_epoch = 0
    best_epoch: int | None = None
    best_model_state = None
    stop_reason: str | None = None
    runtime_contract_sha256 = _sha256_json(runtime_contract)

    output_dir.mkdir(parents=True, exist_ok=True)
    if resume_state is None:
        save_config(cfg, output_dir / "config.yaml")
        _save_split_manifest(manifest, split, output_dir / "split.csv")
        save_json(output_dir / "research_access.json", {**access.contract(), **cohort.contract()})
        save_json(output_dir / "development_control.json", control_plan.contract)
        if not control_plan.donor_mapping.empty:
            control_plan.donor_mapping.to_csv(
                output_dir / "development_control_donors.csv", index=False
            )
        save_json(output_dir / "vocab.json", vocab)
        save_json(output_dir / "params.json", params)
        save_json(output_dir / "class_map.json", full_ds.class_map)
    wandb_run = _init_wandb(
        cfg,
        output_dir,
        params,
        len(split.train),
        len(split.val),
        len(train_loader),
        len(val_loader),
    )
    if wandb_run is not None and cfg.get("tracking", {}).get("wandb", {}).get("watch_model", False):
        _watch_wandb(model)

    resume_contract = {
        "resolved_config_sha256": resume_config_sha256,
        "runtime_contract_sha256": runtime_contract_sha256,
        "source_contract": current_source_revision_contract(),
        "selection_policy": selection_policy,
        "validation_epochs": sorted(validation_epochs),
        "max_epochs": max_epochs,
        "determinism": _exact_determinism_contract() if resume_exact else None,
    }
    last_resume_commit = resume_loaded_commit
    if resume_state is not None:
        _validate_resume_runtime(resume_state, resume_contract)
        model.load_state_dict(resume_state["training_state"]["model_state"], strict=True)
        optimizer.load_state_dict(resume_state["training_state"]["optimizer_state"])
        scaler.load_state_dict(resume_state["training_state"]["scaler_state"])
        progress = resume_state["progress"]
        completed_epoch = int(progress["completed_epoch"])
        best_acc = float(progress["best_metric"])
        stale = int(progress["stale"])
        global_step = int(progress["global_step"])
        metrics_rows = list(progress["metrics_rows"])
        best_epoch = progress.get("best_epoch")
        best_epoch = int(best_epoch) if best_epoch is not None else None
        best_model_state = resume_state.get("best_model_state")
        stop_reason = progress.get("stop_reason")
        if stop_reason not in {None, "early_stop", "max_epochs"}:
            raise ValueError(f"Exact-resume state has an unknown stop reason: {stop_reason!r}.")
        _write_training_metrics(output_dir, selection_policy, metrics_rows)
        if best_model_state is not None and best_epoch is not None:
            current_model_state = {
                name: value.detach().clone() for name, value in model.state_dict().items()
            }
            model.load_state_dict(best_model_state, strict=True)
            save_checkpoint(
                output_dir / "best.pt",
                model,
                optimizer,
                config=cfg,
                epoch=best_epoch,
                best_metric=best_acc,
                vocabularies=vocab,
                class_map=full_ds.class_map,
                asset_info=asset_provenance,
                save_trainable_only=save_trainable_only,
                checkpoint_role=(
                    "source_validation_best"
                    if selection_policy == "validation_best"
                    else "fixed_final_epoch"
                ),
                selection_split="val",
                selection_metric="accuracy",
            )
            model.load_state_dict(current_model_state, strict=True)
        # Restore stochastic state only after every setup/re-materialization action.
        restore_rng_state(resume_state["rng"], train_generator)
    elif resume_exact:
        last_resume_commit = save_resume_state(
            output_dir,
            _exact_resume_payload(
                contract=resume_contract,
                model=model,
                optimizer=optimizer,
                scaler=scaler,
                train_generator=train_generator,
                completed_epoch=0,
                best_metric=best_acc,
                stale=stale,
                global_step=global_step,
                metrics_rows=metrics_rows,
                best_epoch=best_epoch,
                best_model_state=best_model_state,
                stop_reason=None,
            ),
        )

    epoch_iterator = range(completed_epoch + 1, max_epochs + 1) if stop_reason is None else ()
    for epoch in epoch_iterator:
        measurement.start_phase("train", epoch=epoch)
        try:
            train_loss, global_step = _train_epoch(
                model,
                train_loader,
                optimizer,
                scaler,
                cfg,
                device,
                amp_enabled,
                wandb_run=wandb_run,
                epoch=epoch,
                global_step=global_step,
            )
        except Exception as exc:
            measurement.finish_phase(status="failed", error=exc)
            raise
        measurement.finish_phase()
        completed_epoch = epoch
        should_validate = selection_policy == "validation_best" or epoch in validation_epochs
        if should_validate:
            measurement.start_phase("validation", epoch=epoch)
            try:
                val = evaluate_loader(
                    model,
                    val_loader,
                    device,
                    amp_enabled=eval_amp_enabled,
                )
            except Exception as exc:
                measurement.finish_phase(status="failed", error=exc)
                raise
            measurement.finish_phase()
        else:
            val = None
        row = {"epoch": epoch, "train_loss": train_loss}
        if val is not None:
            row.update({f"val_{key}": value for key, value in val.items()})
        metrics_rows.append(row)
        _write_training_metrics(output_dir, selection_policy, metrics_rows)
        log_payload = {
            "epoch": epoch,
            "train/loss": train_loss,
            "lr": optimizer.param_groups[0]["lr"],
        }
        if val is not None:
            log_payload.update(
                {
                    "val/accuracy": val["accuracy"],
                    "val/nll": val["nll"],
                }
            )
        improved = False
        if selection_policy == "validation_best" and val is not None:
            improved = val["accuracy"] > best_acc
            if improved:
                best_acc = val["accuracy"]
                stale = 0
            else:
                stale += 1
            log_payload["best/accuracy"] = best_acc
        elif selection_policy == "fixed_final" and epoch == max_epochs:
            if val is None:  # pragma: no cover - policy resolver always includes final epoch
                raise RuntimeError("fixed_final policy did not evaluate its final epoch.")
            best_acc = val["accuracy"]
            improved = True
            log_payload["fixed_final/accuracy"] = best_acc
        if improved:
            best_epoch = epoch
            best_model_state = {
                name: value.detach().clone() for name, value in model.state_dict().items()
            }
        _log_wandb(
            wandb_run,
            log_payload,
            step=global_step,
        )
        save_checkpoint(
            output_dir / "last.pt",
            model,
            optimizer,
            config=cfg,
            epoch=epoch,
            best_metric=best_acc,
            vocabularies=vocab,
            class_map=full_ds.class_map,
            asset_info=asset_provenance,
            save_trainable_only=save_trainable_only,
            checkpoint_role="last_epoch",
            selection_split="val" if val is not None else None,
            selection_metric="accuracy" if val is not None else None,
        )
        if selection_policy in {"fixed_epoch_blind", "fixed_epoch_sealed"} and epoch == max_epochs:
            save_checkpoint(
                output_dir / "final.pt",
                model,
                optimizer,
                config=cfg,
                epoch=epoch,
                best_metric=None,
                vocabularies=vocab,
                class_map=full_ds.class_map,
                asset_info=asset_provenance,
                save_trainable_only=save_trainable_only,
                checkpoint_role=(
                    "development_fixed_epoch"
                    if selection_policy == "fixed_epoch_blind"
                    else "confirmatory_fixed_epoch"
                ),
                selection_split=None,
                selection_metric="fixed_epoch",
            )
            _log_checkpoint_artifact(wandb_run, output_dir / "final.pt", cfg, epoch)
        elif improved:
            save_checkpoint(
                output_dir / "best.pt",
                model,
                optimizer,
                config=cfg,
                epoch=epoch,
                best_metric=best_acc,
                vocabularies=vocab,
                class_map=full_ds.class_map,
                asset_info=asset_provenance,
                save_trainable_only=save_trainable_only,
                checkpoint_role=(
                    "source_validation_best"
                    if selection_policy == "validation_best"
                    else "fixed_final_epoch"
                ),
                selection_split="val",
                selection_metric="accuracy",
            )
            _log_checkpoint_artifact(wandb_run, output_dir / "best.pt", cfg, epoch)
        stop_reason = (
            "early_stop"
            if selection_policy == "validation_best" and not improved and stale >= patience
            else "max_epochs"
            if epoch == max_epochs
            else None
        )
        if resume_exact:
            last_resume_commit = save_resume_state(
                output_dir,
                _exact_resume_payload(
                    contract=resume_contract,
                    model=model,
                    optimizer=optimizer,
                    scaler=scaler,
                    train_generator=train_generator,
                    completed_epoch=completed_epoch,
                    best_metric=best_acc,
                    stale=stale,
                    global_step=global_step,
                    metrics_rows=metrics_rows,
                    best_epoch=best_epoch,
                    best_model_state=best_model_state,
                    stop_reason=stop_reason,
                ),
            )
            if test_crash_after_epoch_commit == epoch:
                raise RuntimeError(f"Injected exact-resume crash after committed epoch {epoch}.")
        if stop_reason == "early_stop":
            break
    if not access.allow_outer_test_during_training:
        _finish_wandb(wandb_run)
        if resume_exact:
            save_json(
                output_dir / "training_completion.json",
                _training_completion_receipt(
                    output_dir,
                    resume_contract=resume_contract,
                    completed_epoch=completed_epoch,
                    resume_commit=last_resume_commit,
                    stop_reason=stop_reason,
                ),
            )
        return {
            "best_validation_accuracy": (
                None
                if selection_policy in {"fixed_epoch_blind", "fixed_epoch_sealed"}
                else best_acc
            ),
            "output_dir": str(output_dir),
            "params": params,
            "runtime_contract": runtime_contract,
            "research_access": access.contract(),
            "cohort": cohort.contract(),
            "development_control": control_plan.contract,
            "checkpoint_selection_policy": selection_policy,
            "completed_epochs": completed_epoch,
            "resume_exact": resume_exact,
            "resumed_from_epoch": (
                int(resume_state["progress"]["completed_epoch"])
                if resume_state is not None
                else None
            ),
            "elapsed_time_sec": float(time.perf_counter() - run_started),
        }

    best_checkpoint = torch.load(output_dir / "best.pt", map_location=device, weights_only=False)
    load_checkpoint_model_state(model, best_checkpoint)
    trial_time_sec = float(cfg.get("evaluation", {}).get("trial_time_sec", 2.0))
    if test_loader is None:  # pragma: no cover - policy and construction are coupled above
        raise GovernanceError("Outer-test access was granted without a test loader.")
    measurement.start_phase("outer_test")
    try:
        test_metrics = evaluate_loader(
            model,
            test_loader,
            device,
            trial_time_sec=trial_time_sec,
            amp_enabled=eval_amp_enabled,
        )
    except Exception as exc:
        measurement.finish_phase(status="failed", error=exc)
        raise
    measurement.finish_phase()
    reference_itr = itr_bits_per_min(n_classes, best_acc, trial_time_sec)
    test_metrics.update(
        {
            "reference_validation_accuracy": best_acc,
            "accuracy_drop": float(best_acc - test_metrics["accuracy"]),
            "accuracy_drop_rate": _relative_drop(best_acc, test_metrics["accuracy"]),
            "reference_validation_itr_bits_per_min": reference_itr,
            "itr_drop": float(reference_itr - test_metrics.get("itr_bits_per_min", 0.0)),
            "itr_drop_rate": _relative_drop(
                reference_itr, test_metrics.get("itr_bits_per_min", 0.0)
            ),
        }
    )
    save_json(output_dir / "metrics_test.json", test_metrics)
    _log_wandb(
        wandb_run,
        {f"test/{key}": value for key, value in test_metrics.items()},
        step=global_step,
    )
    if wandb_run is not None:
        for key, value in test_metrics.items():
            wandb_run.summary[f"test/{key}"] = value
    _finish_wandb(wandb_run)
    if resume_exact:
        save_json(
            output_dir / "training_completion.json",
            _training_completion_receipt(
                output_dir,
                resume_contract=resume_contract,
                completed_epoch=completed_epoch,
                resume_commit=last_resume_commit,
                stop_reason=stop_reason,
            ),
        )
    return {
        "best_accuracy": best_acc,
        "test": test_metrics,
        "output_dir": str(output_dir),
        "params": params,
        "runtime_contract": runtime_contract,
        "development_control": control_plan.contract,
        "resume_exact": resume_exact,
        "elapsed_time_sec": float(time.perf_counter() - run_started),
    }


def _resolve_augmentation_channel_sets(cfg: dict) -> None:
    augment = cfg.setdefault("augment", {})
    names = augment.get("channel_sets") or []
    if not names:
        return
    with Path("configs/channel_sets.yaml").open("r", encoding="utf-8") as handle:
        registry = yaml.safe_load(handle) or {}
    canonical = CanonicalChannelMap.from_yaml()
    resolved = []
    for name in names:
        if name not in registry:
            raise KeyError(f"Unknown training channel set {name!r}: {sorted(registry)}")
        resolved.append(canonical.get_ids(list(registry[name])))
    augment["channel_subset_ids"] = resolved


def _resolve_split_seed(cfg: dict) -> int:
    """Keep participant assignment fixed across optimization/model seeds."""
    return int(cfg.get("data", {}).get("split_seed", 42))


def _validate_development_control_access(cfg: dict, access: ResearchAccess) -> tuple[str, str]:
    control = normalize_development_control(
        cfg.get("protocol", {}).get("development_control", DEVELOPMENT_CONTROL_NONE)
    )
    training_control = normalize_development_control(
        cfg.get("protocol", {}).get("training_external_metadata_control", DEVELOPMENT_CONTROL_NONE)
    )
    if training_control not in {DEVELOPMENT_CONTROL_NONE, "within_class_shuffle"}:
        raise GovernanceError(
            "training_external_metadata_control currently permits only within_class_shuffle."
        )
    if control != DEVELOPMENT_CONTROL_NONE and training_control != DEVELOPMENT_CONTROL_NONE:
        raise GovernanceError(
            "Use either a model-wide development_control or a train-only external control, "
            "not both."
        )
    effective_control = (
        training_control if training_control != DEVELOPMENT_CONTROL_NONE else control
    )
    if effective_control == DEVELOPMENT_CONTROL_NONE:
        return control, training_control
    if (
        not access.governed
        or access.execution_phase != "development"
        or access.cohort_role != "development"
    ):
        raise GovernanceError(
            f"Development control {effective_control!r} is restricted to the governed S1-S3 "
            "development cohort."
        )
    if effective_control == "counterfactual_wet_dry":
        raise GovernanceError(
            "counterfactual_wet_dry is an evaluation-only intervention on a "
            "correct-metadata A2 checkpoint; training on the swap could learn its inverse."
        )
    protocol = cfg.get("protocol", {})
    condition = cfg.get("model", {}).get("condition_encoder", {})
    if protocol.get("metadata_contract_version") != "0.4-dev":
        raise GovernanceError("Development controls require Protocol 0.4-dev.")
    if cfg.get("data", {}).get("split") not in {
        "cross_subject_train_val",
        "development_subject_fold",
    }:
        raise GovernanceError("Development controls require the S1-S3 train/validation split.")
    if not condition.get("enabled", True) or condition.get("external_metadata_mode") != "observed":
        raise GovernanceError(
            "Development controls require an enabled condition encoder with observed external "
            "metadata before the declared control treatment is applied."
        )
    return control, training_control


def _validate_primary_split_contract(
    cfg: dict,
    manifest: pd.DataFrame,
    split,
    *,
    access: ResearchAccess | None = None,
    cohort: CohortBinding | None = None,
) -> None:
    protocol = cfg.get("protocol", {})
    if protocol.get("primary_ablation_role") not in {
        "A0_eeg_only",
        "A2_structured_condition_prompt",
    }:
        return
    access = access or resolve_research_access(cfg)
    if cohort is None and access.governed:
        cohort = bind_cohort(manifest, access)
    if access.execution_phase == "development":
        expected_splits = {"cross_subject_train_val", "development_subject_fold"}
    elif access.execution_phase == "confirmatory_training":
        expected_splits = {"confirmatory_lockbox"}
    else:
        expected_splits = {"cross_subject_fold"}
    observed_split = cfg.get("data", {}).get("split")
    if (
        protocol.get("metadata_contract_version") != "0.4-dev"
        or observed_split not in expected_splits
    ):
        raise ValueError(
            f"Primary A0/A2 phase {access.execution_phase!r} requires Protocol 0.4 "
            f"one of {sorted(expected_splits)}; got {observed_split!r}."
        )
    split_indices = [
        np.asarray(split.train, dtype=int),
        np.asarray(split.val, dtype=int),
        np.asarray(split.test, dtype=int),
    ]
    combined = np.concatenate(split_indices)
    eligible = set(range(len(manifest)))
    if len(combined) != len(eligible) or set(combined.tolist()) != eligible:
        raise ValueError("Primary split must assign every eligible cohort row exactly once.")
    if access.execution_phase == "development" and len(split.test):
        raise ValueError("Development split must not create an outer-test partition.")
    if access.execution_phase == "confirmatory_training" and len(split.val):
        raise ValueError("Confirmatory lockbox training must not create a validation partition.")
    group_values = manifest["dataset_id"].astype(str) + "::" + manifest["subject_id"].astype(str)
    group_sets = [set(group_values.iloc[indices]) for indices in split_indices]
    if any(
        group_sets[left] & group_sets[right] for left in range(3) for right in range(left + 1, 3)
    ):
        raise ValueError("Primary outer split leaks a participant across train/val/test.")


def _save_split_manifest(manifest: pd.DataFrame, split, path: Path) -> None:
    selected = np.concatenate(
        [
            np.asarray(split.train, dtype=int),
            np.asarray(split.val, dtype=int),
            np.asarray(split.test, dtype=int),
        ]
    )
    assignment = np.full(len(manifest), "unassigned", dtype=object)
    assignment[split.train] = "train"
    assignment[split.val] = "val"
    assignment[split.test] = "test"
    columns = [
        column
        for column in [
            "sample_id",
            "dataset_id",
            "subject_id",
            "session_id",
            "electrode_type",
            "headband_order",
            "condition_period",
            "label",
        ]
        if column in manifest
    ]
    table = manifest.iloc[selected][columns].copy()
    table["split"] = assignment[selected]
    table.to_csv(path, index=False)


def _validate_fresh_governed_output(output_dir: Path) -> None:
    if output_dir.exists() and any(output_dir.iterdir()):
        raise GovernanceError(
            f"Governed output directory is not empty: {output_dir}. Choose a fresh output "
            "directory; existing artifacts are never overwritten automatically."
        )


def _apply_exact_determinism() -> None:
    expected_workspace = ":4096:8"
    observed = os.environ.get("CUBLAS_WORKSPACE_CONFIG")
    if observed not in {None, expected_workspace}:
        raise RuntimeError("Exact resume requires CUBLAS_WORKSPACE_CONFIG=:4096:8 before CUDA use.")
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = expected_workspace
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def _exact_determinism_contract() -> dict[str, object]:
    return {
        "torch_deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "cudnn_benchmark": bool(torch.backends.cudnn.benchmark),
        "cudnn_deterministic": bool(torch.backends.cudnn.deterministic),
        "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
    }


def _validate_exact_resume_options(cfg: dict) -> None:
    data = cfg.get("data", {})
    if bool(data.get("persistent_workers", False)):
        raise ValueError(
            "Exact resume v1 forbids data.persistent_workers=true because worker RNG "
            "state cannot be restored at an epoch boundary."
        )
    wandb = cfg.get("tracking", {}).get("wandb", {})
    if bool(wandb.get("enabled", False)):
        raise ValueError("Exact resume v1 requires W&B disabled to avoid duplicate side effects.")


def _resume_config_sha256(cfg: dict) -> str:
    normalized = copy.deepcopy(cfg)
    normalized.pop("runtime_contract", None)
    return resolved_config_sha256(normalized)


def _validate_resume_preflight(
    state: dict,
    *,
    resolved_config_digest: str,
    current_source: dict[str, object],
) -> None:
    contract = state.get("contract") or {}
    if contract.get("resolved_config_sha256") != resolved_config_digest:
        raise GovernanceError(
            "Exact-resume config differs from the committed run; start a fresh output."
        )
    if contract.get("source_contract") != current_source:
        raise GovernanceError(
            "Exact-resume source tree differs from the committed run; restore the exact source."
        )


def _validate_resume_runtime(state: dict, contract: dict) -> None:
    observed = state.get("contract")
    if observed != contract:
        differing = sorted(
            key
            for key in set(observed or {}) | set(contract)
            if (observed or {}).get(key) != contract.get(key)
        )
        raise GovernanceError(
            f"Exact-resume runtime/data contract changed before state restoration: {differing}."
        )
    progress = state.get("progress") or {}
    completed = int(progress.get("completed_epoch", -1))
    if completed < 0 or completed > int(contract["max_epochs"]):
        raise GovernanceError("Exact-resume progress has an invalid completed epoch.")


def _exact_resume_payload(
    *,
    contract: dict,
    model,
    optimizer,
    scaler,
    train_generator: torch.Generator,
    completed_epoch: int,
    best_metric: float,
    stale: int,
    global_step: int,
    metrics_rows: list[dict],
    best_epoch: int | None,
    best_model_state,
    stop_reason: str | None,
) -> dict:
    return {
        "contract": contract,
        "progress": {
            "phase": "training",
            "completed_epoch": int(completed_epoch),
            "next_epoch": int(completed_epoch + 1),
            "best_metric": float(best_metric),
            "best_epoch": best_epoch,
            "stale": int(stale),
            "global_step": int(global_step),
            "metrics_rows": copy.deepcopy(metrics_rows),
            "stop_reason": stop_reason,
        },
        "training_state": {
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scaler_state": scaler.state_dict(),
        },
        "rng": capture_rng_state(train_generator),
        "best_model_state": best_model_state,
    }


def _write_training_metrics(output_dir: Path, selection_policy: str, rows: list[dict]) -> None:
    name = (
        "metrics_train.csv"
        if selection_policy in {"fixed_epoch_blind", "fixed_epoch_sealed"}
        else "metrics_val.csv"
    )
    pd.DataFrame(rows).to_csv(output_dir / name, index=False)


def _training_completion_receipt(
    output_dir: Path,
    *,
    resume_contract: dict,
    completed_epoch: int,
    resume_commit: dict | None,
    stop_reason: str | None,
) -> dict[str, object]:
    checkpoint_paths = {name: output_dir / name for name in ("best.pt", "final.pt", "last.pt")}
    artifact_paths = {
        name: output_dir / name
        for name in (
            "metrics_train.csv",
            "metrics_val.csv",
            "split.csv",
            "development_control.json",
            "development_control_donors.csv",
        )
    }
    return {
        "schema": "cfeg.training-completion.v1",
        "status": "completed",
        "completed_epoch": int(completed_epoch),
        "stop_reason": stop_reason,
        "resolved_config_sha256": resume_contract["resolved_config_sha256"],
        "runtime_contract_sha256": resume_contract["runtime_contract_sha256"],
        "resume_generation": (
            int(resume_commit["generation"]) if resume_commit is not None else None
        ),
        "resume_state_sha256": (
            str(resume_commit["state_sha256"]) if resume_commit is not None else None
        ),
        "checkpoint_sha256": {
            name: _sha256_file(path) for name, path in checkpoint_paths.items() if path.is_file()
        },
        "artifact_sha256": {
            name: _sha256_file(path) for name, path in artifact_paths.items() if path.is_file()
        },
    }


def _record_runtime_attempt(
    output_dir: Path, attempt_metrics: dict[str, object]
) -> tuple[dict[str, object], Path]:
    attempts_dir = output_dir / "runtime_attempts"
    attempts_dir.mkdir(parents=True, exist_ok=True)
    existing = sorted(attempts_dir.glob("attempt_*.json"))
    index = len(existing)
    attempt_path = attempts_dir / f"attempt_{index:03d}.json"
    if attempt_path.exists():  # pragma: no cover - defensive against sparse manual files
        raise RuntimeError(f"Runtime attempt path already exists: {attempt_path}.")
    save_json(attempt_path, attempt_metrics)
    paths = sorted(attempts_dir.glob("attempt_*.json"))
    attempts = [json.loads(path.read_text(encoding="utf-8")) for path in paths]

    def maximum(field: str):
        values = [attempt.get(field) for attempt in attempts if attempt.get(field) is not None]
        return max(values) if values else None

    aggregate = dict(attempt_metrics)
    aggregate.update(
        {
            "attempt_count": len(attempts),
            "elapsed_time_sec": float(
                sum(float(attempt.get("elapsed_time_sec", 0.0)) for attempt in attempts)
            ),
            "peak_memory_allocated_bytes": maximum("peak_memory_allocated_bytes"),
            "peak_memory_reserved_bytes": maximum("peak_memory_reserved_bytes"),
            "cuda_oom": any(bool(attempt.get("cuda_oom")) for attempt in attempts),
            "runtime_attempts": [
                {
                    "path": path.relative_to(output_dir).as_posix(),
                    "sha256": _sha256_file(path),
                    "status": attempt.get("status"),
                    "elapsed_time_sec": attempt.get("elapsed_time_sec"),
                }
                for path, attempt in zip(paths, attempts)
            ],
        }
    )
    runtime_path = output_dir / "runtime_metrics.json"
    save_json(runtime_path, aggregate)
    return aggregate, runtime_path


def _relative_drop(reference: float, observed: float) -> float:
    if abs(reference) < 1e-12:
        return 0.0
    return float((reference - observed) / reference)


def _resolve_training_selection(cfg: dict, max_epochs: int) -> tuple[str, set[int]]:
    policy = str(cfg.get("train", {}).get("checkpoint_selection", "validation_best"))
    if policy not in {
        "validation_best",
        "fixed_final",
        "fixed_epoch_blind",
        "fixed_epoch_sealed",
    }:
        raise ValueError(
            "train.checkpoint_selection must be validation_best, fixed_final, "
            "fixed_epoch_blind, or fixed_epoch_sealed."
        )
    raw_epochs = cfg.get("train", {}).get("validation_epochs")
    if policy == "validation_best":
        if raw_epochs not in (None, []):
            raise ValueError(
                "train.validation_epochs is only valid with checkpoint_selection=fixed_final."
            )
        return policy, set(range(1, max_epochs + 1))
    if policy in {"fixed_epoch_blind", "fixed_epoch_sealed"}:
        if raw_epochs not in (None, []):
            raise ValueError(
                f"{policy} forbids train.validation_epochs because held-out "
                "development outcomes remain unrevealed until the complete grid exists."
            )
        if max_epochs < 1:
            raise ValueError(f"{policy} training requires train.epochs >= 1.")
        return policy, set()
    if max_epochs < 1:
        raise ValueError("fixed_final training requires train.epochs >= 1.")
    epochs = {int(value) for value in (raw_epochs or [max_epochs])}
    epochs.add(max_epochs)
    invalid = sorted(epoch for epoch in epochs if epoch < 1 or epoch > max_epochs)
    if invalid:
        raise ValueError(
            f"train.validation_epochs must lie within [1, {max_epochs}], got {invalid}."
        )
    return policy, epochs


def _train_epoch(
    model,
    loader,
    optimizer,
    scaler,
    cfg,
    device,
    amp_enabled: bool,
    *,
    wandb_run=None,
    epoch: int = 0,
    global_step: int = 0,
) -> tuple[float, int]:
    model.train()
    total = 0.0
    count = 0
    log_interval = int(cfg.get("train", {}).get("log_interval", 25) or 0)
    n_batches = len(loader)
    for batch_idx, batch in enumerate(loader, start=1):
        batch = _to_device(batch, device)
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast(device_type=device.type, enabled=amp_enabled):
            loss = _step_loss(model, batch, cfg, draw_index=epoch)
        scaler.scale(loss).backward()
        grad_clip = cfg["train"].get("grad_clip_norm")
        if grad_clip:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(grad_clip))
        scaler.step(optimizer)
        scaler.update()
        loss_value = float(loss.detach().cpu())
        total += loss_value * batch["x"].shape[0]
        count += batch["x"].shape[0]
        global_step += 1
        if log_interval and (
            batch_idx == 1 or batch_idx % log_interval == 0 or batch_idx == n_batches
        ):
            running_loss = total / max(count, 1)
            print(
                f"epoch {epoch} batch {batch_idx}/{n_batches} "
                f"loss={loss_value:.4f} running_loss={running_loss:.4f}",
                flush=True,
            )
            _log_wandb(
                wandb_run,
                {
                    "epoch": epoch,
                    "train/batch_loss": loss_value,
                    "train/running_loss": running_loss,
                    "train/batch": batch_idx,
                    "train/epoch_progress": batch_idx / max(n_batches, 1),
                },
                step=global_step,
            )
    return total / max(count, 1), global_step


def _step_loss(model, batch, cfg, *, draw_index: int = 0) -> torch.Tensor:
    loss_cfg = cfg.get("loss", {})
    aug = cfg.get("augment", {})
    y = batch["y"]
    query_corruption = aug.get("query_reliability_corruption", {})
    if bool(query_corruption.get("enabled", False)):
        if aug.get("make_two_views", False):
            raise ValueError(
                "query_reliability_corruption cannot be combined with legacy two-view "
                "augmentation because that path invalidates the Q contract."
            )
        corrupted_x, corrupted_cond, _affected = apply_query_reliability_training_mixture(
            batch["x"],
            batch["cond"],
            sample_ids=list(batch["sample_id"]),
            cell_ids=list(query_corruption.get("cell_ids") or []),
            seed=int(query_corruption.get("seed", cfg.get("seed", 42))),
            draw_index=int(draw_index),
            clean_probability=float(query_corruption.get("clean_probability", 0.25)),
        )
        out = model(corrupted_x, corrupted_cond, use_latent=True)
        return F.cross_entropy(out.logits, y)
    if not aug.get("make_two_views", True):
        out = model(batch["x"], batch["cond"], use_latent=True)
        loss = F.cross_entropy(out.logits, y)
        ce_zero_weight = float(loss_cfg.get("ce_zero_weight", 0.5))
        if out.logits_zero is not None:
            loss = loss + ce_zero_weight * F.cross_entropy(out.logits_zero, y)
        if out.mu is not None:
            loss = loss + float(loss_cfg.get("beta_kl", 0.001)) * kl_normal(out.mu, out.logvar).to(
                loss.device
            )
        return loss

    (x1, cond1), (x2, cond2) = make_two_views(
        batch["x"],
        batch["cond"],
        channel_dropout_prob=float(aug.get("channel_dropout_prob", 0.2)),
        min_channels=int(aug.get("min_channels", 2)),
        channel_subsets=aug.get("channel_subset_ids"),
        channel_subset_prob=float(aug.get("channel_subset_prob", 0.0)),
        noise_std_range=tuple(aug.get("noise_std_range", [0.01, 0.05])),
        time_shift_samples=int(aug.get("time_shift_samples", 8)),
    )
    out1 = model(x1, cond1, use_latent=True)
    out2 = model(x2, cond2, use_latent=True)
    loss = F.cross_entropy(out1.logits, y) + F.cross_entropy(out2.logits, y)
    ce_zero_weight = float(loss_cfg.get("ce_zero_weight", 0.5))
    if out1.logits_zero is not None:
        loss = loss + ce_zero_weight * (
            F.cross_entropy(out1.logits_zero, y) + F.cross_entropy(out2.logits_zero, y)
        )
    loss = loss + float(loss_cfg.get("lambda_cons", 0.1)) * representation_consistency_loss(
        out1.h, out2.h
    )
    loss = loss + float(loss_cfg.get("lambda_logit_cons", 0.05)) * symmetric_kl_logits(
        out1.logits, out2.logits
    )
    if out1.mu is not None:
        loss = loss + float(loss_cfg.get("beta_kl", 0.001)) * (
            kl_normal(out1.mu, out1.logvar).to(loss.device)
            + kl_normal(out2.mu, out2.logvar).to(loss.device)
        )
    return loss


@torch.no_grad()
def evaluate_loader(
    model,
    loader,
    device,
    *,
    trial_time_sec: float | None = None,
    amp_enabled: bool = False,
) -> dict[str, float]:
    model.eval()
    labels = []
    logits = []
    reliability_aux: list[dict[str, torch.Tensor]] = []
    for batch in loader:
        batch = _to_device(batch, device)
        with torch.amp.autocast(device_type=device.type, enabled=amp_enabled):
            out = model(batch["x"], batch["cond"], use_latent=False)
        labels.append(batch["y"].detach().cpu())
        logits.append(out.logits.detach().float().cpu())
        if "spatial_operator_frobenius_norm" in out.aux:
            reliability_aux.append(out.aux)
    if not labels:
        return {
            "accuracy": 0.0,
            "balanced_accuracy": 0.0,
            "macro_f1": 0.0,
            "nll": 0.0,
            "ece": 0.0,
            "n_samples": 0,
        }
    metrics = classification_metrics(
        torch.cat(labels).numpy(),
        torch.cat(logits).numpy(),
        trial_time_sec=trial_time_sec,
    )
    metrics.update(_summarize_reliability_aux(reliability_aux))
    return metrics


def _summarize_reliability_aux(
    batches: list[dict[str, torch.Tensor]],
) -> dict[str, float]:
    keys = (
        "metadata_residual_alpha",
        "spatial_operator_frobenius_norm",
        "query_spatial_operator_frobenius_norm",
        "metadata_spatial_residual_frobenius_norm",
        "spatial_self_gain_min",
        "spatial_self_gain_max",
        "spatial_offdiagonal_row_l1_max",
        "channel_query_available_count",
        "channel_metadata_available_count",
    )
    values: dict[str, list[torch.Tensor]] = {key: [] for key in keys}
    for batch in batches:
        for key in keys:
            value = batch.get(key)
            if torch.is_tensor(value):
                values[key].append(value.detach().float().flatten().cpu())
    summary: dict[str, float] = {}
    for key, parts in values.items():
        if not parts:
            continue
        combined = torch.cat(parts)
        summary[f"diagnostic_{key}_mean"] = float(combined.mean())
        summary[f"diagnostic_{key}_min"] = float(combined.min())
        summary[f"diagnostic_{key}_max"] = float(combined.max())
    return summary


def _to_device(batch, device):
    out = dict(batch)
    out["x"] = batch["x"].to(device)
    out["y"] = batch["y"].to(device)
    out["cond"] = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch["cond"].items()}
    return out


def _processed_asset_provenance(roots: list[Path]) -> dict:
    datasets = []
    for root in roots:
        path = root / "asset_info.json"
        info = {}
        if path.exists():
            with path.open("r", encoding="utf-8") as handle:
                info = json.load(handle)
        analysis_only_fingerprints = _analysis_only_artifact_fingerprints(root, info)
        datasets.append(
            {
                "processed_dir": str(root),
                "dataset_id": info.get("dataset_id"),
                "dataset_revision": info.get("dataset_revision"),
                "source": info.get("source"),
                "source_version": info.get("source_version"),
                "query_qc_extractor_version": info.get("query_qc_extractor_version"),
                "external_continuous_schema": info.get("external_continuous_schema"),
                "impedance_numeric_signature": info.get("impedance_numeric_signature"),
                "processed_subject_count": info.get("processed_subject_count"),
                "file_fingerprints": {
                    name: _sha256_file(root / name)
                    for name in (
                        "signals.h5",
                        "asset_info.json",
                        "manifest.jsonl",
                        "manifest.parquet",
                        "preprocess_config.yaml",
                        "class_map.json",
                        "wearable_v3_audit_receipt.json",
                    )
                    if (root / name).exists()
                },
                "analysis_only_artifact_fingerprints": analysis_only_fingerprints,
                "signals_h5_size_bytes": (
                    (root / "signals.h5").stat().st_size if (root / "signals.h5").exists() else None
                ),
            }
        )
    return {"processed_dirs": [str(root) for root in roots], "datasets": datasets}


def _analysis_only_artifact_fingerprints(root: Path, info: dict) -> dict[str, str]:
    artifacts = info.get("analysis_only_artifacts") or {}
    if not isinstance(artifacts, dict):
        raise TypeError("asset_info.analysis_only_artifacts must be a mapping.")
    fingerprints: dict[str, str] = {}
    for name, contract in sorted(artifacts.items()):
        if Path(name).name != name or not isinstance(contract, dict):
            raise ValueError("Analysis-only artifact paths/contracts must be simple file entries.")
        if contract.get("model_input") is not False:
            raise ValueError(f"Analysis-only artifact {name!r} must declare model_input=false.")
        artifact_path = root / name
        if not artifact_path.is_file():
            raise ValueError(f"Analysis-only artifact is missing: {artifact_path}.")
        observed = _sha256_file(artifact_path)
        if contract.get("sha256") != observed:
            raise ValueError(f"Analysis-only artifact digest mismatch: {artifact_path}.")
        fingerprints[name] = observed
    return fingerprints


def _runtime_contract(
    model,
    manifest,
    split,
    vocab: dict,
    asset_provenance: dict,
    *,
    access: ResearchAccess,
    cohort: CohortBinding,
    control_plan: DevelopmentControlPlan,
    loader_settings: dict,
) -> dict:
    schema = [
        {
            "name": name,
            "shape": list(parameter.shape),
            "dtype": str(parameter.dtype),
            "requires_grad": bool(parameter.requires_grad),
        }
        for name, parameter in model.named_parameters()
    ]
    environment = _environment_contract()
    return {
        "parameter_schema_sha256": _sha256_json(schema),
        "initial_trainable_state_sha256": _trainable_state_sha256(model),
        "split_assignment_sha256": _split_assignment_hash(manifest, split),
        "vocabulary_sha256": _sha256_json(vocab),
        "asset_provenance_sha256": _sha256_json(asset_provenance),
        "execution_phase": access.execution_phase,
        "analysis_plan_status": access.plan_status,
        "analysis_plan_sha256": access.plan_sha256,
        "cohort_role": access.cohort_role,
        "cohort_roles_sha256": access.cohort_roles_sha256,
        "cohort_sha256": cohort.cohort_sha256,
        "cohort_subject_count": cohort.expected_n_subjects,
        "cohort_sample_count": cohort.expected_n_samples,
        "outer_test_access_during_training": access.allow_outer_test_during_training,
        "development_control": control_plan.contract,
        "development_control_sha256": _sha256_json(control_plan.contract),
        "loader_settings": loader_settings,
        "loader_settings_sha256": _sha256_json(loader_settings),
        "environment": environment,
        "environment_sha256": _sha256_json(environment),
        **_source_revision_contract(),
    }


def _trainable_state_sha256(model) -> str:
    digest = hashlib.sha256()
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        digest.update(name.encode("utf-8"))
        tensor = parameter.detach().cpu().contiguous().reshape(-1)
        digest.update(tensor.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def _environment_contract() -> dict[str, object]:
    cuda_available = torch.cuda.is_available()
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": str(torch.__version__),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "cuda_available": cuda_available,
        "torch_cuda": str(torch.version.cuda) if torch.version.cuda is not None else None,
        "cudnn": torch.backends.cudnn.version(),
        "device_count": torch.cuda.device_count() if cuda_available else 0,
        "device_name": torch.cuda.get_device_name(0) if cuda_available else "cpu",
    }


def _source_revision_contract() -> dict[str, object]:
    return current_source_revision_contract()


def _split_assignment_hash(manifest: pd.DataFrame, split) -> str:
    assignments: list[dict[str, str]] = []
    for split_name, indices in (
        ("train", split.train),
        ("val", split.val),
        ("test", split.test),
    ):
        assignments.extend(
            {"sample_id": str(manifest.iloc[int(index)]["sample_id"]), "split": split_name}
            for index in indices
        )
    assignments.sort(key=lambda row: row["sample_id"])
    return _sha256_json(assignments)


def _sha256_json(value) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _init_wandb(
    cfg: dict,
    output_dir: Path,
    params: dict,
    n_train: int,
    n_val: int,
    n_train_batches: int,
    n_val_batches: int,
):
    wandb_cfg = cfg.get("tracking", {}).get("wandb", {})
    if not wandb_cfg.get("enabled", False):
        return None
    try:
        import wandb
    except Exception as exc:
        raise RuntimeError(
            "W&B logging is enabled but wandb is not installed.\n"
            "Install it on the GPU pod with:\n"
            "  python -m pip install wandb\n"
            "or reinstall project requirements:\n"
            "  python -m pip install -r requirements.txt"
        ) from exc
    wandb_dir = Path(wandb_cfg.get("dir") or os.environ.get("WANDB_DIR", str(output_dir)))
    wandb_dir.mkdir(parents=True, exist_ok=True)
    run = wandb.init(
        project=wandb_cfg.get("project", "calibration-free-eeg"),
        entity=wandb_cfg.get("entity"),
        name=cfg.get("run_name"),
        dir=str(wandb_dir),
        mode=wandb_cfg.get("mode", "online"),
        tags=wandb_cfg.get("tags"),
        config=cfg,
    )
    run.summary["params/total"] = params["total"]
    run.summary["params/trainable"] = params["trainable"]
    run.summary["data/n_train"] = int(n_train)
    run.summary["data/n_val"] = int(n_val)
    run.summary["data/n_train_batches"] = int(n_train_batches)
    run.summary["data/n_val_batches"] = int(n_val_batches)
    return run


def _log_wandb(run, metrics: dict, step: int) -> None:
    if run is not None:
        run.log(metrics, step=step)


def _log_checkpoint_artifact(run, ckpt_path: Path, cfg: dict, epoch: int) -> None:
    if run is None or not cfg.get("tracking", {}).get("wandb", {}).get("log_model", False):
        return
    import wandb

    artifact = wandb.Artifact(f"{cfg.get('run_name', 'cfeg')}-best", type="checkpoint")
    artifact.add_file(str(ckpt_path))
    run.log_artifact(artifact, aliases=["best", f"epoch-{epoch}"])


def _watch_wandb(model) -> None:
    import wandb

    wandb.watch(model, log="gradients", log_freq=100)


def _finish_wandb(run) -> None:
    if run is not None:
        run.finish()
