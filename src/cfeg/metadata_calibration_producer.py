from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import safetensors
import torch
from safetensors.torch import load_file

from cfeg.analysis.metadata_calibration_efficiency import (
    MetadataCalibrationBundleBindings,
    metadata_calibration_bundle_spec_for_phase,
)
from cfeg.baselines.metadata_calibration_adapter import (
    DESCRIPTORS,
    BaselineSignalBatch,
    GovernedCalibrationBaselineAdapter,
)
from cfeg.data.metadata_calibration_baseline_sealed import (
    BaselineSealedRows,
    BaselineSealedTarget,
    baseline_support_indices,
    baseline_support_labels,
    exact_source_pool_identities,
    load_baseline_query_only,
    load_baseline_source_fit,
    load_baseline_target,
    official_wearable_channels,
)
from cfeg.data.metadata_calibration_interventions import (
    build_metadata_intervention_mapping,
    resolve_metadata_intervention_batch,
    resolved_context_usage_sha256,
)
from cfeg.data.metadata_calibration_sealed import (
    SealedTargetGroup,
    load_sealed_source_fit,
    load_sealed_target_group,
    support_indices_for_budget,
    support_labels_for_budget,
    tensor_batch,
)
from cfeg.identity import canonical_identity_sha256
from cfeg.metadata_calibration_atomic import validate_and_summarize_support_rows
from cfeg.metadata_calibration_baseline_contract import baseline_execution_binding
from cfeg.metadata_calibration_checkpoint import (
    verify_full_composite_checkpoint,
    write_full_composite_checkpoint,
)
from cfeg.metadata_calibration_execution import (
    ExperimentPhase,
    cell_execution_contract_sha256,
    execution_cells_for_phase,
    resolve_cell_execution_spec,
)
from cfeg.metadata_calibration_training import (
    FittedCandidate,
    refit_held_candidate,
    select_and_refit_candidate,
)
from cfeg.utils.config import load_config

_REPOSITORY = Path(__file__).resolve().parents[2]


def run_candidate_job(
    *,
    group_root: str | Path,
    job_root: str | Path,
    job_spec: Mapping[str, Any],
    phase: ExperimentPhase,
    bindings: MetadataCalibrationBundleBindings,
    source_commit: str,
    source_tree_sha256: str,
    source_tag: str,
    input_checkpoint_group_tree_sha256: str,
) -> dict[str, Any]:
    """Train/refit one seed and emit outcome-free target probabilities."""

    _validate_job_spec(job_spec, phase=phase, producer_kind="candidate")
    runtime = _validated_cuda_runtime()
    device = "cuda:0"
    destination = _new_job_root(job_root)
    source = load_sealed_source_fit(group_root)
    recipe = load_config(
        _REPOSITORY / "configs/train/metadata_calibration_efficiency_v1.yaml",
        strict_env=False,
    )
    if {
        "stage1_amp": recipe["runtime"].get("stage1_amp"),
        "stage2_amp": recipe["runtime"].get("stage2_amp"),
        "eval_amp": recipe["runtime"].get("eval_amp"),
    } != {"stage1_amp": True, "stage2_amp": False, "eval_amp": False}:
        raise RuntimeError("Candidate mixed-precision stages differ from the frozen recipe.")
    seed = int(job_spec["seed"])
    checkpoint_group = str(job_spec["checkpoint_group"])
    _validate_source_group_data(source.frame, job_spec=job_spec)
    if phase == "source_development":
        fold = int(checkpoint_group.removeprefix("fold"))
        fold_spec = recipe["data"]["folds"][fold]
        fitted = select_and_refit_candidate(
            source,
            recipe=recipe,
            seed=seed,
            outer_fold=fold,
            inner_fit_subjects=fold_spec["inner_fit"],
            inner_validation_subjects=fold_spec["inner_validation"],
            device=device,
        )
        outer_fold: int | None = fold
    else:
        held_selected_epochs = job_spec.get("held_selected_epochs")
        if not isinstance(held_selected_epochs, Mapping) or set(
            held_selected_epochs
        ) != {"stage1_epochs", "stage2_epochs"}:
            raise ValueError("Held candidate job lacks source-gate selected epochs.")
        stage1_epochs = int(held_selected_epochs["stage1_epochs"])
        stage2_epochs = int(held_selected_epochs["stage2_epochs"])
        fitted = refit_held_candidate(
            source,
            recipe=recipe,
            seed=seed,
            stage1_epochs=stage1_epochs,
            stage2_epochs=stage2_epochs,
            device=device,
        )
        outer_fold = None
    checkpoint_path = destination / "composite.safetensors"
    checkpoint_receipt_path = destination / "composite.receipt.json"
    checkpoint_receipt = write_full_composite_checkpoint(
        fitted.model,
        checkpoint_path=checkpoint_path,
        receipt_path=checkpoint_receipt_path,
        phase=phase,
        seed=seed,
        outer_fold=outer_fold,
        stage1_freeze=fitted.stage1_freeze,
        stage1_state_sha256=fitted.stage1_state_sha256,
        stage1_recipe_sha256=_json_sha256(recipe["stage_1_common_Q"]),
        stage2_recipe_sha256=_json_sha256(recipe["stage_2_bounded_M"]),
        plan_sha256=bindings.plan_sha256,
        decision_receipt_sha256=bindings.decision_receipt_sha256,
        asset_receipt_sha256=bindings.asset_receipt_sha256,
        asset_fingerprint_bundle_sha256=bindings.asset_fingerprint_bundle_sha256,
        class_map_sha256=bindings.class_map_sha256,
        source_commit=source_commit,
        source_tree_sha256=source_tree_sha256,
        source_tag=source_tag,
    )
    verified_checkpoint_receipt = verify_full_composite_checkpoint(
        checkpoint_path,
        checkpoint_receipt_path,
    )
    if verified_checkpoint_receipt != checkpoint_receipt:
        raise RuntimeError("Composite checkpoint verification changed its receipt payload.")
    fitted.model.load_state_dict(load_file(checkpoint_path, device=str(device)), strict=True)
    # Target EEG, support labels, and acquisition context are not opened until
    # training and the immutable full checkpoint have completed.
    target = load_sealed_target_group(group_root)
    _validate_target_group_data(target.query.frame, job_spec=job_spec)
    if set(source.frame["subject_id"].astype(str)) & set(
        target.query.frame["subject_id"].astype(str)
    ):
        raise ValueError("Producer source and target participant cohorts overlap.")
    predictions = _evaluate_candidate(
        fitted,
        target,
        phase=phase,
        checkpoint_group=checkpoint_group,
        seed=seed,
        checkpoint_sha256=str(checkpoint_receipt["checkpoint_sha256"]),
        bindings=bindings,
        device=device,
    )
    output_path = destination / "probabilities.parquet"
    predictions.to_parquet(output_path, index=False)
    training_path = destination / "training_selection.json"
    _write_json(training_path, fitted.selection.as_dict())
    receipt = _producer_receipt(
        job_spec=job_spec,
        phase=phase,
        producer_kind="candidate",
        input_checkpoint_group_tree_sha256=input_checkpoint_group_tree_sha256,
        output_files=(output_path, checkpoint_path, checkpoint_receipt_path, training_path),
        rows=len(predictions),
        bindings=bindings,
        extra={
            "checkpoint_sha256": checkpoint_receipt["checkpoint_sha256"],
            "selected_stage1_epochs": fitted.selection.stage1_epochs,
            "selected_stage2_epochs": fitted.selection.stage2_epochs,
            "runtime": runtime,
        },
    )
    _write_json(destination / "producer_receipt.json", receipt)
    return receipt


def run_baseline_job(
    *,
    group_root: str | Path,
    job_root: str | Path,
    job_spec: Mapping[str, Any],
    phase: ExperimentPhase,
    bindings: MetadataCalibrationBundleBindings,
    input_checkpoint_group_tree_sha256: str,
) -> dict[str, Any]:
    """Run one deterministic baseline without a query-outcome capability."""

    _validate_job_spec(job_spec, phase=phase, producer_kind="baseline")
    destination = _new_job_root(job_root)
    adapter_id = str(job_spec["adapter_id"])
    descriptor = DESCRIPTORS[adapter_id]  # type: ignore[index]
    source = (
        load_baseline_source_fit(group_root)
        if descriptor.requires_source_labeled_eeg
        else None
    )
    target = (
        load_baseline_query_only(group_root)
        if adapter_id == "strict_FBCCA"
        else load_baseline_target(group_root)
    )
    if source is not None:
        _validate_source_group_data(source.frame, job_spec=job_spec)
    elif job_spec.get("source_pool_subject_ids_sha256") != canonical_identity_sha256([]):
        raise ValueError("A target-only baseline job declares a nonempty source pool.")
    _validate_target_group_data(target.query.frame, job_spec=job_spec)
    scores = _evaluate_baseline(
        adapter_id=adapter_id,
        source=source,
        target=target,
        phase=phase,
        checkpoint_group=str(job_spec["checkpoint_group"]),
        bindings=bindings,
    )
    output_path = destination / "scores.parquet"
    scores.to_parquet(output_path, index=False)
    receipt = _producer_receipt(
        job_spec=job_spec,
        phase=phase,
        producer_kind="baseline",
        input_checkpoint_group_tree_sha256=input_checkpoint_group_tree_sha256,
        output_files=(output_path,),
        rows=len(scores),
        bindings=bindings,
        extra={"adapter_id": adapter_id, "deterministic_seed_column_present": False},
    )
    _write_json(destination / "producer_receipt.json", receipt)
    return receipt


@torch.no_grad()
def _evaluate_candidate(
    fitted: FittedCandidate,
    target: SealedTargetGroup,
    *,
    phase: ExperimentPhase,
    checkpoint_group: str,
    seed: int,
    checkpoint_sha256: str,
    bindings: MetadataCalibrationBundleBindings,
    device: str,
) -> pd.DataFrame:
    model = fitted.model.eval()
    spec = metadata_calibration_bundle_spec_for_phase(phase=phase)
    support_summary, _ = validate_and_summarize_support_rows(
        target.support_rows,
        expected_query_unlabeled=target.expected_query,
        spec=spec,
    )
    support_lookup = {
        (str(row.subject_id), str(row.electrode_type), int(row.budget)): str(
            row.support_identity_sha256
        )
        for row in support_summary.itertuples(index=False)
    }
    query_identity = {
        str(row.query_token): str(row.query_identity_sha256)
        for row in target.expected_query.itertuples(index=False)
    }
    mappings = {
        intervention: build_metadata_intervention_mapping(
            target.block_context,
            context=intervention,  # type: ignore[arg-type]
            shuffle_seed=20260904,
        )
        for intervention in {
            resolve_cell_execution_spec(role=role, context=context, phase=phase).intervention
            for role, context in execution_cells_for_phase(phase=phase)
        }
    }
    records: list[dict[str, Any]] = []
    target_cells = target.query.episode_keys()
    grid_keys = [
        (role, context, budget)
        for role, context in execution_cells_for_phase(phase=phase)
        for budget in spec.budgets
    ]
    pending_by_grid: dict[
        tuple[str, str, int], list[tuple[dict[str, Any], torch.Tensor]]
    ] = {key: [] for key in grid_keys}
    resolved_by_grid: dict[tuple[str, str, int], list[Any]] = {
        key: [] for key in grid_keys
    }
    expected_tokens_by_grid: dict[tuple[str, str, int], list[str]] = {
        key: [] for key in grid_keys
    }
    for subject, interface in target_cells:
        query_indices = target.query.indices_for_episode(subject, interface)
        query_batch = tensor_batch(target.query, query_indices, device=device)
        query_precomputed = model.prepare_evaluation_features(
            query_batch.x,
            query_batch.cond,
            row_tokens=query_batch.tokens,
            base_input_sha256s=query_batch.base_input_sha256s,
            batch_kind="query",
        )
        empty_x = torch.empty((0, 64, 400), dtype=torch.float32, device=device)
        empty_cond: Mapping[str, Any] = {}
        empty_labels = torch.empty(0, dtype=torch.long, device=device)
        support_cache: dict[int, tuple[Any, Any, Any]] = {
            0: (
                empty_labels,
                None,
                model.prepare_evaluation_features(
                    empty_x,
                    empty_cond,
                    row_tokens=(),
                    base_input_sha256s=(),
                    batch_kind="support",
                ),
            )
        }
        for budget in spec.budgets:
            if budget == 0:
                continue
            support_indices = support_indices_for_budget(
                target,
                subject_id=subject,
                electrode_type=interface,
                budget=budget,
            )
            support_batch = tensor_batch(target.support, support_indices, device=device)
            support_labels = torch.as_tensor(
                support_labels_for_budget(
                    target,
                    subject_id=subject,
                    electrode_type=interface,
                    budget=budget,
                ),
                dtype=torch.long,
                device=device,
            )
            support_cache[budget] = (
                support_labels,
                support_batch,
                model.prepare_evaluation_features(
                    support_batch.x,
                    support_batch.cond,
                    row_tokens=support_batch.tokens,
                    base_input_sha256s=support_batch.base_input_sha256s,
                    batch_kind="support",
                ),
            )

        for role, context in execution_cells_for_phase(phase=phase):
            cell = resolve_cell_execution_spec(role=role, context=context, phase=phase)
            mapping = mappings[cell.intervention]
            query_resolved = resolve_metadata_intervention_batch(
                subject_ids=query_batch.subject_ids,
                electrode_types=query_batch.electrode_types,
                run_ids=query_batch.run_ids,
                row_tokens=query_batch.tokens,
                base_input_sha256s=query_batch.base_input_sha256s,
                mapping=mapping,
                sealed_block_context=target.block_context,
                cell_spec=cell,
                phase=phase,
                batch_kind="query",
            )
            for budget in spec.budgets:
                (
                    support_labels,
                    support_batch,
                    support_precomputed,
                ) = support_cache[budget]
                support_resolved = None
                if support_batch is not None:
                    support_resolved = resolve_metadata_intervention_batch(
                        subject_ids=support_batch.subject_ids,
                        electrode_types=support_batch.electrode_types,
                        run_ids=support_batch.run_ids,
                        row_tokens=support_batch.tokens,
                        base_input_sha256s=support_batch.base_input_sha256s,
                        mapping=mapping,
                        sealed_block_context=target.block_context,
                        cell_spec=cell,
                        phase=phase,
                        batch_kind="support",
                    )
                grid_key = (role, context, budget)
                resolved_by_grid[grid_key].append(query_resolved)
                expected_tokens_by_grid[grid_key].extend(query_batch.tokens)
                if support_resolved is not None:
                    resolved_by_grid[grid_key].append(support_resolved)
                    expected_tokens_by_grid[grid_key].extend(support_batch.tokens)
                output = model.forward_cell_from_precomputed(
                    query_precomputed=query_precomputed,
                    query_resolved=query_resolved,
                    support_precomputed=support_precomputed,
                    support_labels=support_labels,
                    support_resolved=support_resolved,
                    cell_spec=cell,
                    phase=phase,
                )
                pending_by_grid[grid_key].append(
                    (
                        {
                            "subject_id": subject,
                            "electrode_type": interface,
                            "support_identity_sha256": support_lookup[
                                (subject, interface, budget)
                            ],
                            "query_tokens": query_batch.tokens,
                        },
                        output.probabilities.detach(),
                    )
                )

    dispatch_hash = cell_execution_contract_sha256(phase=phase)
    frozen_bindings = bindings.as_dict()
    for role, context, budget in grid_keys:
        grid_key = (role, context, budget)
        cell = resolve_cell_execution_spec(role=role, context=context, phase=phase)
        mapping = mappings[cell.intervention]
        usage_hash = resolved_context_usage_sha256(
            resolved_by_grid[grid_key],
            expected_row_tokens=expected_tokens_by_grid[grid_key],
        )
        pending = pending_by_grid[grid_key]
        if not pending:
            raise RuntimeError("Candidate evaluation produced an empty governed grid.")
        grid_probabilities = torch.cat(
            [probability for _, probability in pending], dim=0
        ).cpu().numpy()
        probability_offset = 0
        for pending_row, probability_chunk in pending:
            row_count = len(pending_row["query_tokens"])
            if probability_chunk.shape != (row_count, spec.n_classes):
                raise RuntimeError("Candidate probability chunk has a wrong exact shape.")
            probability = grid_probabilities[
                probability_offset : probability_offset + row_count
            ]
            probability_offset += row_count
            for query_token, values in zip(pending_row["query_tokens"], probability):
                records.append(
                    {
                        "candidate_id": spec.candidate_id,
                        "phase": phase,
                        "checkpoint_group": checkpoint_group,
                        "checkpoint_sha256": checkpoint_sha256,
                        "role": role,
                        "context": context,
                        "seed": seed,
                        "budget": budget,
                        "query_token": query_token,
                        "subject_id": pending_row["subject_id"],
                        "electrode_type": pending_row["electrode_type"],
                        "support_identity_sha256": pending_row[
                            "support_identity_sha256"
                        ],
                        "query_identity_sha256": query_identity[query_token],
                        "intervention_mapping_sha256": mapping.sha256,
                        "resolved_context_usage_sha256": usage_hash,
                        "cell_execution_contract_sha256": dispatch_hash,
                        **frozen_bindings,
                        **{
                            f"prob_{label:03d}": float(values[label])
                            for label in range(spec.n_classes)
                        },
                    }
                )
        if probability_offset != grid_probabilities.shape[0]:
            raise RuntimeError("Candidate probability rows were not consumed exactly once.")
    return pd.DataFrame(records)


def _evaluate_baseline(
    *,
    adapter_id: str,
    source: BaselineSealedRows | None,
    target: BaselineSealedTarget,
    phase: ExperimentPhase,
    checkpoint_group: str,
    bindings: MetadataCalibrationBundleBindings,
) -> pd.DataFrame:
    descriptor = DESCRIPTORS[adapter_id]  # type: ignore[index]
    execution_binding = baseline_execution_binding(adapter_id)
    adapter = GovernedCalibrationBaselineAdapter(
        adapter_id,  # type: ignore[arg-type]
        n_classes=12,
        filterbank=execution_binding.filterbank,
        filterbank_config_path=execution_binding.filterbank_path,
        filterbank_config_sha256=execution_binding.filterbank_file_sha256,
    )
    wearable = (
        load_config(_REPOSITORY / "configs/data/wearable.yaml", strict_env=False)
        if descriptor.uses_frequency_codebook
        else None
    )
    frequencies = (
        np.asarray(wearable["class_frequencies"], dtype=np.float64)
        if wearable is not None
        else None
    )
    phases = (
        np.asarray(wearable["class_phases"], dtype=np.float64)
        if wearable is not None
        else None
    )
    spec = metadata_calibration_bundle_spec_for_phase(phase=phase)
    if target.support is None:
        if descriptor.applicable_budgets != (0,):
            raise PermissionError("A support-dependent baseline lacks support capability.")
        empty_support = _json_sha256([])
        support_lookup = {
            (subject, interface, 0): empty_support
            for subject, interface in target.query.episode_keys()
        }
    else:
        support_summary, _ = validate_and_summarize_support_rows(
            target.support_rows,
            expected_query_unlabeled=target.expected_query,
            spec=spec,
        )
        support_lookup = {
            (str(row.subject_id), str(row.electrode_type), int(row.budget)): str(
                row.support_identity_sha256
            )
            for row in support_summary.itertuples(index=False)
        }
    query_identity = {
        str(row.query_token): str(row.query_identity_sha256)
        for row in target.expected_query.itertuples(index=False)
    }
    source_by_interface: dict[str, tuple[np.ndarray, np.ndarray, str]] = {}
    if descriptor.requires_source_labeled_eeg:
        if source is None:
            raise ValueError("LST baseline requires its exact labeled source pool.")
        assert source.labels is not None
        source_identities = exact_source_pool_identities(source)
        for interface in ("dry", "wet"):
            indices = np.flatnonzero(
                source.frame["electrode_type"].astype(str).eq(interface).to_numpy()
            )
            source_by_interface[interface] = (
                official_wearable_channels(source.x[indices]),
                np.asarray(source.labels[indices], dtype=np.int64),
                source_identities[interface],
            )
    empty_source = canonical_identity_sha256([])
    records: list[dict[str, Any]] = []
    for subject, interface in target.query.episode_keys():
        query_indices = target.query.indices_for_episode(subject, interface)
        query_x = official_wearable_channels(target.query.x[query_indices])
        query_tokens = target.query.frame.iloc[query_indices]["query_token"].astype(str)
        for budget in descriptor.applicable_budgets:
            if budget == 0:
                support_x = support_y = None
            else:
                if target.support is None:
                    raise PermissionError("Baseline support input is not mounted.")
                indices = baseline_support_indices(
                    target,
                    subject_id=subject,
                    electrode_type=interface,
                    budget=budget,
                )
                support_x = official_wearable_channels(target.support.x[indices])
                support_y = baseline_support_labels(
                    target,
                    subject_id=subject,
                    electrode_type=interface,
                    budget=budget,
                )
            if descriptor.requires_source_labeled_eeg:
                source_x, source_y, source_identity = source_by_interface[interface]
            else:
                source_x = source_y = None
                source_identity = empty_source
            output = adapter.predict(
                BaselineSignalBatch(
                    query_x=query_x,
                    target_support_x=support_x,
                    target_support_y=support_y,
                    source_x=source_x,
                    source_y=source_y,
                    frequencies_hz=frequencies,
                    phases_rad=phases,
                    sfreq=200.0,
                ),
                budget=int(budget),
            )
            for query_token, values in zip(query_tokens, output.scores):
                records.append(
                    {
                        "candidate_id": spec.candidate_id,
                        "phase": phase,
                        "checkpoint_group": checkpoint_group,
                        "adapter_id": adapter_id,
                        "budget": int(budget),
                        "query_token": query_token,
                        "subject_id": subject,
                        "electrode_type": interface,
                        "support_identity_sha256": support_lookup[
                            (subject, interface, int(budget))
                        ],
                        "query_identity_sha256": query_identity[query_token],
                        "source_pool_identity_sha256": source_identity,
                        "implementation_sha256": (
                            execution_binding.implementation_sha256
                        ),
                        "config_sha256": execution_binding.config_sha256,
                        "adapter_contract_sha256": (
                            execution_binding.adapter_contract_sha256
                        ),
                        "score_schema": output.score_schema,
                        "score_interpretation": output.score_interpretation,
                        **bindings.as_dict(),
                        **{
                            f"score_{label:03d}": float(values[label])
                            for label in range(spec.n_classes)
                        },
                    }
                )
    return pd.DataFrame(records)


def _producer_receipt(
    *,
    job_spec: Mapping[str, Any],
    phase: ExperimentPhase,
    producer_kind: str,
    input_checkpoint_group_tree_sha256: str,
    output_files: Sequence[Path],
    rows: int,
    bindings: MetadataCalibrationBundleBindings,
    extra: Mapping[str, Any],
) -> dict[str, Any]:
    receipt: dict[str, Any] = {
        "schema": "cfeg.metadata-calibration-producer-receipt.v1",
        "candidate_id": "metadata-calibration-efficiency-v1",
        "phase": phase,
        "producer_kind": producer_kind,
        "job_id": job_spec["job_id"],
        "job_contract_sha256": job_spec["job_contract_sha256"],
        "checkpoint_group": job_spec["checkpoint_group"],
        "input_checkpoint_group_tree_sha256": input_checkpoint_group_tree_sha256,
        "bindings": bindings.as_dict(),
        "output_files": [
            {
                "name": path.name,
                "size_bytes": path.stat().st_size,
                "file_sha256": _sha256_file(path),
            }
            for path in output_files
        ],
        "rows": int(rows),
        "query_outcomes_loaded": False,
        "raw_sample_ids_loaded": False,
        **dict(extra),
    }
    receipt["producer_receipt_sha256"] = _json_sha256(receipt)
    return receipt


def _validate_job_spec(
    job: Mapping[str, Any], *, phase: ExperimentPhase, producer_kind: str
) -> None:
    if job.get("producer_kind") != producer_kind:
        raise ValueError("Producer kind differs from the exact job contract.")
    payload = dict(job)
    claimed = payload.pop("job_contract_sha256", None)
    if claimed != _json_sha256(payload):
        raise ValueError("Producer job contract hash is invalid.")
    if job.get("query_outcomes_loaded") is not False:
        raise PermissionError("A producer job must be outcome-free.")
    expected_groups = {"fold0", "fold1", "fold2"} if phase == "source_development" else {"held"}
    if str(job.get("checkpoint_group")) not in expected_groups:
        raise ValueError("Producer checkpoint group differs from its phase.")
    group = str(job["checkpoint_group"])
    expected_fold = int(group.removeprefix("fold")) if group.startswith("fold") else None
    if job.get("outer_fold") != expected_fold:
        raise ValueError("Producer outer fold differs from its checkpoint group.")
    if producer_kind == "candidate":
        if type(job.get("seed")) is not int or int(job["seed"]) not in {42, 43, 44}:
            raise ValueError("Candidate job seed is not one frozen source-training seed.")
        held_epochs = job.get("held_selected_epochs")
        if phase == "source_development" and held_epochs is not None:
            raise ValueError("Source candidate cannot carry held refit epochs.")
        if phase == "held_participant_evaluation" and not isinstance(
            held_epochs, Mapping
        ):
            raise ValueError("Held candidate must bind source-gate selected epochs.")


def _validated_cuda_runtime() -> dict[str, Any]:
    if os.environ.get("CUBLAS_WORKSPACE_CONFIG") != ":4096:8":
        raise RuntimeError("Candidate worker requires CUBLAS_WORKSPACE_CONFIG=:4096:8.")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)
    torch.set_float32_matmul_precision("highest")
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("Candidate execution requires exactly one visible CUDA device.")
    name = torch.cuda.get_device_name(0)
    if name != "NVIDIA GeForce RTX 4090":
        raise RuntimeError("Candidate execution is frozen to one NVIDIA GeForce RTX 4090.")
    return {
        "device": "cuda:0",
        "visible_device_count": 1,
        "device_name": name,
        "torch_version": torch.__version__,
        "safetensors_version": safetensors.__version__,
        "torch_cuda_version": torch.version.cuda,
        "cudnn_version": torch.backends.cudnn.version(),
        "cublas_workspace_config": os.environ["CUBLAS_WORKSPACE_CONFIG"],
        "stage1_amp_training": True,
        "stage2_amp_training": False,
        "amp_evaluation": False,
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
        "cudnn_deterministic": torch.backends.cudnn.deterministic,
        "cuda_matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
        "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
        "float32_matmul_precision": torch.get_float32_matmul_precision(),
    }


def _validate_source_group_data(
    frame: pd.DataFrame,
    *,
    job_spec: Mapping[str, Any],
) -> None:
    groups = set(frame["checkpoint_group"].astype(str))
    subjects = set(frame["subject_id"].astype(str))
    if groups != {str(job_spec["checkpoint_group"])} or canonical_identity_sha256(
        subjects
    ) != job_spec.get("source_pool_subject_ids_sha256"):
        raise ValueError("Producer source data differs from the exact job contract.")


def _validate_target_group_data(
    frame: pd.DataFrame,
    *,
    job_spec: Mapping[str, Any],
) -> None:
    groups = set(frame["checkpoint_group"].astype(str))
    subjects = set(frame["subject_id"].astype(str))
    if groups != {str(job_spec["checkpoint_group"])} or canonical_identity_sha256(
        subjects
    ) != job_spec.get("target_subject_ids_sha256"):
        raise ValueError("Producer target data differs from the exact job contract.")


def _new_job_root(path: str | Path) -> Path:
    root = Path(path).absolute()
    if root.exists() or root.is_symlink():
        raise FileExistsError("Producer job root must be completely new.")
    root.mkdir(parents=True, mode=0o700)
    return root


def _tree_sha256(root: Path) -> str:
    if not root.is_dir() or root.is_symlink():
        raise ValueError("Producer input group must be one real directory.")
    records = [
        {
            "path": path.relative_to(root).as_posix(),
            "file_sha256": _sha256_file(path),
        }
        for path in sorted(root.rglob("*"))
        if path.is_file()
    ]
    return _json_sha256({"files": records})


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    )
    path.write_text(payload + "\n", encoding="utf-8")


def _json_sha256(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
