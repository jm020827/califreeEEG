from __future__ import annotations

import hashlib
import json
import math
import os
import random
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from cfeg.data.metadata_calibration_features import (
    build_wearable_metadata_features,
    build_wearable_q_features,
)
from cfeg.data.metadata_calibration_interventions import (
    build_metadata_intervention_mapping,
)
from cfeg.data.metadata_calibration_sealed import (
    SealedTensorRows,
    tensor_batch,
)
from cfeg.metadata_calibration_checkpoint import (
    Stage1FreezeBinding,
    capture_stage1_freeze,
)
from cfeg.models.backbones.spectral_transformer import SpectralEEGTransformerBackbone
from cfeg.models.metadata_calibration_composite import MetadataCalibrationComposite
from cfeg.models.metadata_calibration_prior import BoundedMetadataCalibrationPrior

_BUDGETS = (0, 1, 3, 5)
_EAUC_WEIGHTS = {0: 1.0 / 6.0, 1: 1.0 / 2.0, 3: 1.0 / 3.0, 5: 0.0}


@dataclass(frozen=True)
class EpochSelection:
    stage1_epochs: int
    stage2_epochs: int
    stage1_history: tuple[dict[str, float | int], ...]
    stage2_history: tuple[dict[str, float | int], ...]

    def as_dict(self) -> dict[str, Any]:
        return {
            "stage1_epochs": self.stage1_epochs,
            "stage2_epochs": self.stage2_epochs,
            "stage1_history": [dict(row) for row in self.stage1_history],
            "stage2_history": [dict(row) for row in self.stage2_history],
        }


@dataclass(frozen=True)
class FittedCandidate:
    model: MetadataCalibrationComposite
    selection: EpochSelection
    stage1_state_sha256: str
    stage1_freeze: Stage1FreezeBinding


@dataclass(frozen=True)
class EpisodeFeatures:
    subject_id: str
    electrode_type: str
    z: torch.Tensor
    q: torch.Tensor
    m: torch.Tensor
    m_missing: torch.Tensor
    shuffled_m: torch.Tensor
    shuffled_m_missing: torch.Tensor
    labels: torch.Tensor
    blocks: torch.Tensor


def build_metadata_calibration_composite(
    recipe: Mapping[str, Any],
) -> MetadataCalibrationComposite:
    model = recipe["model"]
    backbone_spec = model["backbone"]
    prior_spec = model["prior"]
    backbone = SpectralEEGTransformerBackbone(
        c_max=int(model["c_max"]),
        t_len=int(model["t_len"]),
        target_sfreq=float(model["target_sfreq"]),
        d_model=int(model["embedding_dim"]),
        depth=int(backbone_spec["depth"]),
        n_heads=int(backbone_spec["n_heads"]),
        min_frequency_hz=float(backbone_spec["min_frequency_hz"]),
        max_frequency_hz=float(backbone_spec["max_frequency_hz"]),
        dropout=float(backbone_spec["dropout"]),
        channel_vocab_size=int(model["c_max"]) + 1,
    )
    prior = BoundedMetadataCalibrationPrior(
        embedding_dim=int(model["embedding_dim"]),
        n_classes=int(model["n_classes"]),
        q_feature_dim=int(prior_spec["q_feature_dim"]),
        metadata_feature_dim=int(prior_spec["metadata_feature_dim"]),
        hidden_dim=int(prior_spec["hidden_dim"]),
        q_log_ratio_bound=math.log(float(prior_spec["q_precision_ratio"][1])),
        metadata_log_ratio_bound=math.log(
            float(prior_spec["metadata_precision_ratio"][1])
        ),
        precision_floor=float(prior_spec["absolute_precision"][0]),
        precision_ceiling=float(prior_spec["absolute_precision"][1]),
        source_prior_strength=float(prior_spec["source_prior_strength"]),
    )
    return MetadataCalibrationComposite(backbone=backbone, prior=prior)


def select_and_refit_candidate(
    rows: SealedTensorRows,
    *,
    recipe: Mapping[str, Any],
    seed: int,
    outer_fold: int,
    inner_fit_subjects: Sequence[str],
    inner_validation_subjects: Sequence[str],
    device: str | torch.device = "cuda",
) -> FittedCandidate:
    """Select epochs on exact inner subjects, then freshly refit on their union."""

    if type(outer_fold) is not int or outer_fold not in {0, 1, 2}:
        raise ValueError("Source selection requires outer_fold 0, 1, or 2.")
    fit = tuple(str(value) for value in inner_fit_subjects)
    validation = tuple(str(value) for value in inner_validation_subjects)
    if len(fit) != 20 or len(validation) != 6 or set(fit) & set(validation):
        raise ValueError("Source epoch selection requires exact disjoint inner 20/6 subjects.")
    observed = set(rows.frame["subject_id"].astype(str))
    if observed != set(fit) | set(validation):
        raise ValueError("Sealed source-fit participants differ from the exact inner split.")
    _configure_determinism(seed)
    selected_model = build_metadata_calibration_composite(recipe).to(device)
    stage1_state, stage1_epochs, stage1_history = _select_stage1(
        selected_model,
        rows,
        fit_subjects=fit,
        validation_subjects=validation,
        recipe=recipe,
        seed=seed,
        outer_fold=outer_fold,
        device=device,
    )
    selected_model.load_state_dict(stage1_state)
    stage2_state, stage2_epochs, stage2_history = _select_stage2(
        selected_model,
        rows,
        fit_subjects=fit,
        validation_subjects=validation,
        recipe=recipe,
        seed=seed,
        outer_fold=outer_fold,
        device=device,
    )
    # The selected model is not carried into the outer checkpoint. Recreate it
    # from the same seed and use only the two selected epoch counts.
    del stage2_state
    _configure_determinism(seed)
    refit = build_metadata_calibration_composite(recipe).to(device)
    all_subjects = tuple(sorted(set(fit) | set(validation)))
    _fit_stage1_fixed(
        refit,
        rows,
        subjects=all_subjects,
        epochs=stage1_epochs,
        recipe=recipe,
        seed=seed,
        outer_fold=outer_fold,
        device=device,
    )
    stage1_digest = tensor_state_sha256(refit.state_dict())
    stage1_freeze = capture_stage1_freeze(refit)
    _fit_stage2_fixed(
        refit,
        rows,
        subjects=all_subjects,
        epochs=stage2_epochs,
        recipe=recipe,
        seed=seed,
        outer_fold=outer_fold,
        device=device,
    )
    return FittedCandidate(
        model=refit,
        selection=EpochSelection(
            stage1_epochs=stage1_epochs,
            stage2_epochs=stage2_epochs,
            stage1_history=tuple(stage1_history),
            stage2_history=tuple(stage2_history),
        ),
        stage1_state_sha256=stage1_digest,
        stage1_freeze=stage1_freeze,
    )


def refit_held_candidate(
    rows: SealedTensorRows,
    *,
    recipe: Mapping[str, Any],
    seed: int,
    stage1_epochs: int,
    stage2_epochs: int,
    device: str | torch.device = "cuda",
) -> FittedCandidate:
    subjects = tuple(sorted(set(rows.frame["subject_id"].astype(str))))
    if len(subjects) != 39:
        raise ValueError("Held refit requires all exact 39 source-development participants.")
    _validate_selected_epochs(stage1_epochs, stage2_epochs, recipe)
    _configure_determinism(seed)
    model = build_metadata_calibration_composite(recipe).to(device)
    _fit_stage1_fixed(
        model,
        rows,
        subjects=subjects,
        epochs=stage1_epochs,
        recipe=recipe,
        seed=seed,
        outer_fold=None,
        device=device,
    )
    stage1_digest = tensor_state_sha256(model.state_dict())
    stage1_freeze = capture_stage1_freeze(model)
    _fit_stage2_fixed(
        model,
        rows,
        subjects=subjects,
        epochs=stage2_epochs,
        recipe=recipe,
        seed=seed,
        outer_fold=None,
        device=device,
    )
    return FittedCandidate(
        model=model,
        selection=EpochSelection(
            stage1_epochs=int(stage1_epochs),
            stage2_epochs=int(stage2_epochs),
            stage1_history=(),
            stage2_history=(),
        ),
        stage1_state_sha256=stage1_digest,
        stage1_freeze=stage1_freeze,
    )


def _select_stage1(
    model: MetadataCalibrationComposite,
    rows: SealedTensorRows,
    *,
    fit_subjects: Sequence[str],
    validation_subjects: Sequence[str],
    recipe: Mapping[str, Any],
    seed: int,
    outer_fold: int,
    device: str | torch.device,
) -> tuple[dict[str, torch.Tensor], int, list[dict[str, float | int]]]:
    spec = recipe["stage_1_common_Q"]
    _prepare_stage1_trainables(model)
    optimizer = torch.optim.AdamW(
        _stage1_parameters(model),
        lr=float(spec["learning_rate"]),
        weight_decay=float(spec["weight_decay"]),
    )
    scaler = torch.cuda.amp.GradScaler(
        enabled=bool(recipe["runtime"]["stage1_amp"])
        and torch.device(device).type == "cuda"
    )
    best_state: dict[str, torch.Tensor] | None = None
    best_epoch = 0
    best_key: tuple[float, float] | None = None
    patience = 0
    history: list[dict[str, float | int]] = []
    for epoch in range(1, int(spec["maximum_epochs"]) + 1):
        loss = _train_stage1_epoch(
            model,
            rows,
            subjects=fit_subjects,
            optimizer=optimizer,
            scaler=scaler,
            recipe=recipe,
            seed=seed,
            outer_fold=outer_fold,
            epoch=epoch,
            device=device,
        )
        metrics = _validate_stage1(
            model,
            rows,
            subjects=validation_subjects,
            device=device,
        )
        row = {
            "epoch": epoch,
            "train_loss": loss,
            "validation_eauc": metrics["eauc"],
            "validation_nll": metrics["nll"],
        }
        history.append(row)
        key = (float(metrics["eauc"]), -float(metrics["nll"]))
        if epoch >= int(spec["minimum_epochs"]) and _strictly_better(key, best_key):
            best_key = key
            best_epoch = epoch
            best_state = _cpu_state_dict(model)
            patience = 0
        else:
            patience += 1
        if epoch >= int(spec["minimum_epochs"]) and patience >= int(
            spec["early_stop_patience"]
        ):
            break
    if best_state is None or best_epoch <= 0:
        raise RuntimeError("Stage-1 epoch selection did not produce a checkpoint.")
    return best_state, best_epoch, history


def _select_stage2(
    model: MetadataCalibrationComposite,
    rows: SealedTensorRows,
    *,
    fit_subjects: Sequence[str],
    validation_subjects: Sequence[str],
    recipe: Mapping[str, Any],
    seed: int,
    outer_fold: int,
    device: str | torch.device,
) -> tuple[dict[str, torch.Tensor], int, list[dict[str, float | int]]]:
    model.freeze_common_path_for_stage2()
    spec = recipe["stage_2_bounded_M"]
    optimizer = torch.optim.AdamW(
        model.prior.metadata_parameters(),
        lr=float(spec["learning_rate"]),
        weight_decay=float(spec["weight_decay"]),
    )
    fit_cache = _cache_episode_features(
        model, rows, subjects=fit_subjects, device=device
    )
    validation_cache = _cache_episode_features(
        model, rows, subjects=validation_subjects, device=device
    )
    best_state: dict[str, torch.Tensor] | None = None
    best_epoch = 0
    best_key: tuple[float, float] | None = None
    patience = 0
    history: list[dict[str, float | int]] = []
    for epoch in range(1, int(spec["maximum_epochs"]) + 1):
        loss = _train_stage2_epoch(
            model,
            fit_cache,
            optimizer=optimizer,
            recipe=recipe,
            seed=seed,
            outer_fold=outer_fold,
            epoch=epoch,
        )
        metrics = _validate_stage2(model, validation_cache)
        row = {
            "epoch": epoch,
            "train_loss": loss,
            "validation_increment_eauc": metrics["increment_eauc"],
            "validation_correct_minus_shuffle_eauc": metrics[
                "correct_minus_shuffle_eauc"
            ],
        }
        history.append(row)
        key = (
            float(metrics["increment_eauc"]),
            float(metrics["correct_minus_shuffle_eauc"]),
        )
        if epoch >= int(spec["minimum_epochs"]) and _strictly_better(key, best_key):
            best_key = key
            best_epoch = epoch
            best_state = _cpu_state_dict(model)
            patience = 0
        else:
            patience += 1
        if epoch >= int(spec["minimum_epochs"]) and patience >= int(
            spec["early_stop_patience"]
        ):
            break
    if best_state is None or best_epoch <= 0:
        raise RuntimeError("Stage-2 epoch selection did not produce a checkpoint.")
    model.load_state_dict(best_state)
    return best_state, best_epoch, history


def _fit_stage1_fixed(
    model: MetadataCalibrationComposite,
    rows: SealedTensorRows,
    *,
    subjects: Sequence[str],
    epochs: int,
    recipe: Mapping[str, Any],
    seed: int,
    outer_fold: int | None,
    device: str | torch.device,
) -> None:
    spec = recipe["stage_1_common_Q"]
    _prepare_stage1_trainables(model)
    optimizer = torch.optim.AdamW(
        _stage1_parameters(model),
        lr=float(spec["learning_rate"]),
        weight_decay=float(spec["weight_decay"]),
    )
    scaler = torch.cuda.amp.GradScaler(
        enabled=bool(recipe["runtime"]["stage1_amp"])
        and torch.device(device).type == "cuda"
    )
    for epoch in range(1, int(epochs) + 1):
        _train_stage1_epoch(
            model,
            rows,
            subjects=subjects,
            optimizer=optimizer,
            scaler=scaler,
            recipe=recipe,
            seed=seed,
            outer_fold=outer_fold,
            epoch=epoch,
            device=device,
        )


def _fit_stage2_fixed(
    model: MetadataCalibrationComposite,
    rows: SealedTensorRows,
    *,
    subjects: Sequence[str],
    epochs: int,
    recipe: Mapping[str, Any],
    seed: int,
    outer_fold: int | None,
    device: str | torch.device,
) -> None:
    model.freeze_common_path_for_stage2()
    spec = recipe["stage_2_bounded_M"]
    optimizer = torch.optim.AdamW(
        model.prior.metadata_parameters(),
        lr=float(spec["learning_rate"]),
        weight_decay=float(spec["weight_decay"]),
    )
    cache = _cache_episode_features(model, rows, subjects=subjects, device=device)
    for epoch in range(1, int(epochs) + 1):
        _train_stage2_epoch(
            model,
            cache,
            optimizer=optimizer,
            recipe=recipe,
            seed=seed,
            outer_fold=outer_fold,
            epoch=epoch,
        )


def _train_stage1_epoch(
    model: MetadataCalibrationComposite,
    rows: SealedTensorRows,
    *,
    subjects: Sequence[str],
    optimizer: torch.optim.Optimizer,
    scaler: torch.cuda.amp.GradScaler,
    recipe: Mapping[str, Any],
    seed: int,
    outer_fold: int | None,
    epoch: int,
    device: str | torch.device,
) -> float:
    model.train()
    keys = _episode_keys(rows, subjects)
    batches = _episode_batches(
        keys,
        batch_size=int(recipe["episode"]["participant_condition_episode_batch_size"]),
        seed=_derived_seed("stage1", seed, outer_fold, epoch),
    )
    losses: list[float] = []
    amp = bool(recipe["runtime"]["stage1_amp"]) and torch.device(device).type == "cuda"
    for batch_keys in batches:
        indices_by_episode = [_ordered_episode_indices(rows, *key) for key in batch_keys]
        flat_indices = np.concatenate(indices_by_episode)
        batch = tensor_batch(rows, flat_indices, device=device)
        if batch.labels is None:
            raise ValueError("Source training requires sealed labels.")
        optimizer.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(enabled=amp):
            z = model.encode(batch.x, batch.cond)
            q = build_wearable_q_features(batch.x, batch.cond)
            episode_losses: list[torch.Tensor] = []
            offset = 0
            for indices in indices_by_episode:
                size = len(indices)
                episode_losses.append(
                    _common_episode_loss(
                        model,
                        z[offset : offset + size],
                        q[offset : offset + size],
                        batch.labels[offset : offset + size],
                    )
                )
                offset += size
            loss = torch.stack(episode_losses).mean()
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(
            model.parameters(), float(recipe["stage_1_common_Q"]["gradient_clip_norm"])
        )
        scaler.step(optimizer)
        scaler.update()
        losses.append(float(loss.detach().cpu()))
    return float(np.mean(losses))


def _common_episode_loss(
    model: MetadataCalibrationComposite,
    z: torch.Tensor,
    q: torch.Tensor,
    labels: torch.Tensor,
) -> torch.Tensor:
    support_pool = torch.arange(0, 60, device=z.device)
    query = torch.arange(60, 120, device=z.device)
    losses: list[torch.Tensor] = []
    for budget in _BUDGETS:
        support = support_pool[: budget * 12]
        output = model.prior(
            z[query],
            q[query],
            support_embeddings=z[support],
            support_labels=labels[support],
            support_q_features=q[support],
            query_metadata_mode="off",
            support_metadata_mode="off",
        )
        losses.append(F.cross_entropy(output.logits, labels[query]))
    return torch.stack(losses).mean()


def _train_stage2_epoch(
    model: MetadataCalibrationComposite,
    episodes: Sequence[EpisodeFeatures],
    *,
    optimizer: torch.optim.Optimizer,
    recipe: Mapping[str, Any],
    seed: int,
    outer_fold: int | None,
    epoch: int,
) -> float:
    model.train()
    rng = random.Random(_derived_seed("stage2", seed, outer_fold, epoch))
    order = list(range(len(episodes)))
    rng.shuffle(order)
    batch_size = int(recipe["episode"]["participant_condition_episode_batch_size"])
    dropout_probability = float(
        recipe["stage_2_bounded_M"]["whole_metadata_all_missing_dropout_probability"]
    )
    losses: list[float] = []
    for start in range(0, len(order), batch_size):
        optimizer.zero_grad(set_to_none=True)
        episode_losses: list[torch.Tensor] = []
        for episode_index in order[start : start + batch_size]:
            episode = episodes[episode_index]
            dropped = rng.random() < dropout_probability
            m = torch.zeros_like(episode.m) if dropped else episode.m
            missing = torch.ones_like(episode.m_missing) if dropped else episode.m_missing
            support_pool = torch.arange(0, 60, device=episode.z.device)
            query = torch.arange(60, 120, device=episode.z.device)
            budget_losses: list[torch.Tensor] = []
            for budget in _BUDGETS:
                support = support_pool[: budget * 12]
                output = model.prior(
                    episode.z[query],
                    episode.q[query],
                    support_embeddings=episode.z[support],
                    support_labels=episode.labels[support],
                    support_q_features=episode.q[support],
                    query_metadata_values=m[query],
                    query_metadata_missing=missing[query],
                    support_metadata_values=m[support],
                    support_metadata_missing=missing[support],
                    query_metadata_mode="observed",
                    support_metadata_mode="observed",
                )
                budget_losses.append(F.cross_entropy(output.logits, episode.labels[query]))
            _, residual = model.prior.precision(
                episode.q,
                metadata_values=m,
                metadata_missing=missing,
                metadata_mode="observed",
            )
            episode_losses.append(
                torch.stack(budget_losses).mean() + 0.01 * residual.square().mean()
            )
        loss = torch.stack(episode_losses).mean()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            model.prior.metadata_parameters(),
            float(recipe["stage_2_bounded_M"]["gradient_clip_norm"]),
        )
        optimizer.step()
        losses.append(float(loss.detach().cpu()))
    return float(np.mean(losses))


@torch.no_grad()
def _validate_stage1(
    model: MetadataCalibrationComposite,
    rows: SealedTensorRows,
    *,
    subjects: Sequence[str],
    device: str | torch.device,
) -> dict[str, float]:
    model.eval()
    records: list[dict[str, float | int | str]] = []
    for subject, interface in _episode_keys(rows, subjects):
        indices = _ordered_episode_indices(rows, subject, interface)
        batch = tensor_batch(rows, indices, device=device)
        assert batch.labels is not None
        z = model.encode(batch.x, batch.cond)
        q = build_wearable_q_features(batch.x, batch.cond)
        support_pool = torch.arange(0, 60, device=z.device)
        query = torch.arange(60, 120, device=z.device)
        for budget in _BUDGETS:
            support = support_pool[: budget * 12]
            output = model.prior(
                z[query],
                q[query],
                support_embeddings=z[support],
                support_labels=batch.labels[support],
                support_q_features=q[support],
                metadata_mode="off",
            )
            records.append(
                {
                    "subject_id": subject,
                    "electrode_type": interface,
                    "budget": budget,
                    "ba": _balanced_accuracy(output.logits, batch.labels[query]),
                    "nll": float(F.cross_entropy(output.logits, batch.labels[query]).cpu()),
                }
            )
    frame = pd.DataFrame(records)
    return {
        "eauc": _participant_equal_eauc(frame, value="ba"),
        "nll": _participant_equal_budget_mean(frame, value="nll"),
    }


@torch.no_grad()
def _validate_stage2(
    model: MetadataCalibrationComposite,
    episodes: Sequence[EpisodeFeatures],
) -> dict[str, float]:
    model.eval()
    records: list[dict[str, float | int | str]] = []
    for episode in episodes:
        support_pool = torch.arange(0, 60, device=episode.z.device)
        query = torch.arange(60, 120, device=episode.z.device)
        for budget in _BUDGETS:
            support = support_pool[: budget * 12]
            common = model.prior(
                episode.z[query],
                episode.q[query],
                support_embeddings=episode.z[support],
                support_labels=episode.labels[support],
                support_q_features=episode.q[support],
                metadata_mode="off",
            )
            correct = model.prior(
                episode.z[query],
                episode.q[query],
                support_embeddings=episode.z[support],
                support_labels=episode.labels[support],
                support_q_features=episode.q[support],
                query_metadata_values=episode.m[query],
                query_metadata_missing=episode.m_missing[query],
                support_metadata_values=episode.m[support],
                support_metadata_missing=episode.m_missing[support],
                query_metadata_mode="observed",
                support_metadata_mode="observed",
            )
            shuffled = model.prior(
                episode.z[query],
                episode.q[query],
                support_embeddings=episode.z[support],
                support_labels=episode.labels[support],
                support_q_features=episode.q[support],
                query_metadata_values=episode.shuffled_m[query],
                query_metadata_missing=episode.shuffled_m_missing[query],
                support_metadata_values=episode.shuffled_m[support],
                support_metadata_missing=episode.shuffled_m_missing[support],
                query_metadata_mode="observed",
                support_metadata_mode="observed",
            )
            labels = episode.labels[query]
            records.append(
                {
                    "subject_id": episode.subject_id,
                    "electrode_type": episode.electrode_type,
                    "budget": budget,
                    "common": _balanced_accuracy(common.logits, labels),
                    "correct": _balanced_accuracy(correct.logits, labels),
                    "shuffle": _balanced_accuracy(shuffled.logits, labels),
                }
            )
    frame = pd.DataFrame(records)
    return {
        "increment_eauc": _participant_equal_eauc(frame, value="correct")
        - _participant_equal_eauc(frame, value="common"),
        "correct_minus_shuffle_eauc": _participant_equal_eauc(frame, value="correct")
        - _participant_equal_eauc(frame, value="shuffle"),
    }


@torch.no_grad()
def _cache_episode_features(
    model: MetadataCalibrationComposite,
    rows: SealedTensorRows,
    *,
    subjects: Sequence[str],
    device: str | torch.device,
) -> tuple[EpisodeFeatures, ...]:
    model.eval()
    output: list[EpisodeFeatures] = []
    for subject, interface in _episode_keys(rows, subjects):
        indices = _ordered_episode_indices(rows, subject, interface)
        batch = tensor_batch(rows, indices, device=device)
        assert batch.labels is not None
        z = model.encode(batch.x, batch.cond).detach()
        q = build_wearable_q_features(batch.x, batch.cond).detach()
        m, m_missing = build_wearable_metadata_features(
            batch.cond, electrode_types=batch.electrode_types
        )
        blocks = torch.as_tensor(
            [int(value.removeprefix("block")) for value in batch.run_ids],
            dtype=torch.int64,
            device=z.device,
        )
        donor_blocks = _block_shuffle_donors(
            subject=subject,
            interface=interface,
            blocks=tuple(int(value) for value in blocks.cpu()),
        )
        block_to_first = {
            int(block): index
            for index, block in enumerate(blocks.cpu().tolist())
            if int(block) not in {
                int(value) for value in blocks.cpu().tolist()[:index]
            }
        }
        donor_indices = torch.as_tensor(
            [block_to_first[value] for value in donor_blocks],
            dtype=torch.int64,
            device=z.device,
        )
        output.append(
            EpisodeFeatures(
                subject_id=subject,
                electrode_type=interface,
                z=z,
                q=q,
                m=m.detach(),
                m_missing=m_missing.detach(),
                shuffled_m=m.index_select(0, donor_indices).detach(),
                shuffled_m_missing=m_missing.index_select(0, donor_indices).detach(),
                labels=batch.labels.detach(),
                blocks=blocks,
            )
        )
    return tuple(output)


def _block_shuffle_donors(
    *, subject: str, interface: str, blocks: Sequence[int]
) -> tuple[int, ...]:
    context = pd.DataFrame(
        [
            {
                "dataset_id": "wearable",
                "subject_id": subject,
                "electrode_type": observed_interface,
                "run_id": f"block{block:02d}",
                "impedance_channel_ids": [56, 55, 57, 54, 58, 62, 61, 63],
                "impedance_kohm_by_channel": [float(block)] * 8,
            }
            for observed_interface in ("dry", "wet")
            for block in range(1, 11)
        ]
    )
    mapping = build_metadata_intervention_mapping(
        context, context="block_shuffle", shuffle_seed=20260904
    ).frame.set_index(["subject_id", "electrode_type", "run_id"])
    return tuple(
        int(
            str(mapping.loc[(subject, interface, f"block{block:02d}"), "donor_run_id"])
            .removeprefix("block")
        )
        for block in blocks
    )


def _episode_keys(
    rows: SealedTensorRows, subjects: Iterable[str]
) -> tuple[tuple[str, str], ...]:
    allowed = {str(value) for value in subjects}
    observed = set(rows.frame["subject_id"].astype(str))
    if not allowed or not allowed.issubset(observed):
        raise ValueError("Requested source episode subjects are absent from the sealed view.")
    return tuple(
        (subject, interface)
        for subject in sorted(allowed)
        for interface in ("dry", "wet")
    )


def _ordered_episode_indices(
    rows: SealedTensorRows, subject_id: str, electrode_type: str
) -> np.ndarray:
    indices = rows.indices_for_episode(subject_id, electrode_type)
    frame = rows.frame.iloc[indices]
    if rows.labels is None:
        raise ValueError("Source episode ordering requires labels.")
    labels = rows.labels[indices]
    blocks = frame["run_id"].astype(str).map(lambda value: int(value.removeprefix("block")))
    order = np.lexsort((labels, blocks.to_numpy(dtype=np.int64)))
    ordered = indices[order]
    ordered_labels = rows.labels[ordered].reshape(10, 12)
    if not np.array_equal(ordered_labels, np.tile(np.arange(12), (10, 1))):
        raise ValueError("Each source episode must contain ten class-complete blocks.")
    return ordered


def _episode_batches(
    keys: Sequence[tuple[str, str]], *, batch_size: int, seed: int
) -> tuple[tuple[tuple[str, str], ...], ...]:
    if batch_size <= 0:
        raise ValueError("Episode batch size must be positive.")
    ordered = list(keys)
    random.Random(seed).shuffle(ordered)
    return tuple(
        tuple(ordered[start : start + batch_size])
        for start in range(0, len(ordered), batch_size)
    )


def _participant_equal_eauc(frame: pd.DataFrame, *, value: str) -> float:
    condition = frame.groupby(
        ["subject_id", "electrode_type", "budget"], as_index=False, sort=False
    )[value].mean()
    participant = condition.groupby(
        ["subject_id", "budget"], as_index=False, sort=False
    )[value].mean()
    participant["weighted"] = participant[value] * participant["budget"].map(
        _EAUC_WEIGHTS
    )
    return float(participant.groupby("subject_id")["weighted"].sum().mean())


def _participant_equal_budget_mean(frame: pd.DataFrame, *, value: str) -> float:
    """Give every participant, interface, and frozen budget equal NLL weight."""

    condition = frame.groupby(
        ["subject_id", "electrode_type", "budget"], as_index=False, sort=False
    )[value].mean()
    participant_budget = condition.groupby(
        ["subject_id", "budget"], as_index=False, sort=False
    )[value].mean()
    participant = participant_budget.groupby("subject_id", sort=False)[value].mean()
    return float(participant.mean())


def _balanced_accuracy(logits: torch.Tensor, labels: torch.Tensor) -> float:
    prediction = logits.argmax(dim=-1)
    values = []
    for label in range(12):
        mask = labels.eq(label)
        if int(mask.sum()) != 5:
            raise ValueError("Validation query must contain five trials per class.")
        values.append(float(prediction[mask].eq(labels[mask]).float().mean().cpu()))
    return float(np.mean(values))


def _strictly_better(
    candidate: tuple[float, ...], incumbent: tuple[float, ...] | None
) -> bool:
    if incumbent is None:
        return True
    for left, right in zip(candidate, incumbent):
        if left > right + 1e-12:
            return True
        if left < right - 1e-12:
            return False
    return False


def _cpu_state_dict(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}


def tensor_state_sha256(state: Mapping[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(state.items()):
        value = tensor.detach().contiguous().cpu()
        header = json.dumps(
            {"name": name, "dtype": str(value.dtype), "shape": list(value.shape)},
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        raw = value.reshape(-1).view(torch.uint8).numpy().tobytes() if value.numel() else b""
        digest.update(len(header).to_bytes(8, "big"))
        digest.update(header)
        digest.update(len(raw).to_bytes(8, "big"))
        digest.update(raw)
    return digest.hexdigest()


def _derived_seed(stage: str, seed: int, outer_fold: int | None, epoch: int) -> int:
    if stage not in {"stage1", "stage2"}:
        raise ValueError("Epoch shuffle stage must be stage1 or stage2.")
    if outer_fold is not None and (type(outer_fold) is not int or outer_fold not in {0, 1, 2}):
        raise ValueError("Epoch shuffle outer fold must be 0, 1, 2, or held=None.")
    outer_domain = "held" if outer_fold is None else f"fold{outer_fold}"
    payload = (
        f"metadata-calibration-efficiency-v1:{stage}:{seed}:{outer_domain}:{epoch}"
    ).encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


def _configure_determinism(seed: int) -> None:
    if torch.cuda.is_available() and os.environ.get("CUBLAS_WORKSPACE_CONFIG") != (
        ":4096:8"
    ):
        raise RuntimeError(
            "Deterministic CUDA execution requires CUBLAS_WORKSPACE_CONFIG=:4096:8 "
            "before worker startup."
        )
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def _validate_selected_epochs(
    stage1_epochs: int, stage2_epochs: int, recipe: Mapping[str, Any]
) -> None:
    if not int(recipe["stage_1_common_Q"]["minimum_epochs"]) <= int(
        stage1_epochs
    ) <= int(recipe["stage_1_common_Q"]["maximum_epochs"]):
        raise ValueError("Held stage-1 epoch count is outside the frozen range.")
    if not int(recipe["stage_2_bounded_M"]["minimum_epochs"]) <= int(
        stage2_epochs
    ) <= int(recipe["stage_2_bounded_M"]["maximum_epochs"]):
        raise ValueError("Held stage-2 epoch count is outside the frozen range.")


def _prepare_stage1_trainables(model: MetadataCalibrationComposite) -> None:
    for parameter in model.backbone.parameters():
        parameter.requires_grad_(True)
    for parameter in model.prior.common_parameters():
        parameter.requires_grad_(True)
    for parameter in model.prior.metadata_parameters():
        parameter.requires_grad_(False)


def _stage1_parameters(model: MetadataCalibrationComposite) -> list[torch.nn.Parameter]:
    parameters = list(model.backbone.parameters()) + list(model.prior.common_parameters())
    if not parameters or any(not parameter.requires_grad for parameter in parameters):
        raise RuntimeError("Stage-1 trainable parameter construction failed.")
    return parameters
