from __future__ import annotations

import hashlib
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
import torch

from cfeg.constants import METADATA_CONTRACT_V04_DEV
from cfeg.data.metadata_calibration_features import WEARABLE_CALIBRATION_CHANNEL_IDS
from cfeg.data.metadata_calibration_interventions import (
    build_metadata_intervention_mapping,
    metadata_calibration_base_input_sha256,
    resolve_metadata_intervention_batch,
)
from cfeg.metadata_calibration_execution import (
    execution_cells_for_phase,
    resolve_cell_execution_spec,
)
from cfeg.models.backbones.spectral_transformer import SpectralEEGTransformerBackbone
from cfeg.models.metadata_calibration_composite import MetadataCalibrationComposite
from cfeg.models.metadata_calibration_prior import BoundedMetadataCalibrationPrior


def _cond(batch: int) -> dict[str, object]:
    channel_ids = torch.zeros(batch, 64, dtype=torch.long)
    channel_mask = torch.zeros(batch, 64, dtype=torch.bool)
    query_qc = torch.zeros(batch, 64)
    query_missing = torch.ones(batch, 64, dtype=torch.bool)
    impedance = torch.zeros(batch, 64)
    impedance_missing = torch.ones(batch, 64, dtype=torch.bool)
    for index, channel_id in enumerate(WEARABLE_CALIBRATION_CHANNEL_IDS):
        slot = channel_id - 1
        channel_ids[:, slot] = channel_id
        channel_mask[:, slot] = True
        query_qc[:, slot] = 0.1 + 0.01 * index
        query_missing[:, slot] = False
        impedance[:, slot] = 0.2 + 0.1 * index
        impedance_missing[:, slot] = False
    return {
        "channel_ids": channel_ids,
        "channel_mask": channel_mask,
        "channel_query_qc": query_qc,
        "channel_query_qc_missing": query_missing,
        "channel_impedance": impedance,
        "channel_impedance_missing": impedance_missing,
        "sfreq_processed_float": torch.full((batch,), 200.0),
        "metadata_contract_version": METADATA_CONTRACT_V04_DEV,
        "forbidden_future_metadata": torch.full((batch,), float("nan")),
    }


def _composite() -> MetadataCalibrationComposite:
    torch.manual_seed(21)
    backbone = SpectralEEGTransformerBackbone(
        c_max=64,
        t_len=400,
        target_sfreq=200.0,
        d_model=16,
        depth=1,
        n_heads=4,
        dropout=0.0,
    )
    prior = BoundedMetadataCalibrationPrior(
        embedding_dim=16,
        n_classes=3,
        q_feature_dim=56,
        metadata_feature_dim=18,
        hidden_dim=8,
    )
    return MetadataCalibrationComposite(backbone=backbone, prior=prior)


def _block_context() -> pd.DataFrame:
    rows = []
    for interface_index, interface in enumerate(("dry", "wet")):
        for block in range(1, 11):
            rows.append(
                {
                    "dataset_id": "wearable",
                    "subject_id": "sub004",
                    "electrode_type": interface,
                    "run_id": f"block{block:02d}",
                    "impedance_channel_ids": list(WEARABLE_CALIBRATION_CHANNEL_IDS),
                    "impedance_kohm_by_channel": (
                        np.arange(8, dtype=np.float32)
                        + block
                        + 100 * interface_index
                    ),
                }
            )
    return pd.DataFrame(rows)


def _resolved_row(
    *,
    x: torch.Tensor,
    cond: dict[str, object],
    prefix: str,
    run_id: str,
    batch_kind: str,
    context: pd.DataFrame,
):
    channel_mask = cond["channel_mask"]
    assert torch.is_tensor(channel_mask)
    mapping = build_metadata_intervention_mapping(
        context,
        context="opposite_interface",
    )
    cell_spec = resolve_cell_execution_spec(
        role="A_QM",
        context="opposite_interface",
        phase="held_participant_evaluation",
    )
    token = f"{prefix}_{hashlib.sha256(f'{prefix}:{run_id}'.encode()).hexdigest()}"
    return resolve_metadata_intervention_batch(
        subject_ids=("sub004",),
        electrode_types=("dry",),
        run_ids=(run_id,),
        row_tokens=(token,),
        base_input_sha256s=(
            metadata_calibration_base_input_sha256(x[0], channel_mask[0]),
        ),
        mapping=mapping,
        sealed_block_context=context,
        cell_spec=cell_spec,
        phase="held_participant_evaluation",
        batch_kind=batch_kind,
    )


def _resolved_batch(
    *,
    x: torch.Tensor,
    cond: dict[str, object],
    prefix: str,
    run_ids: tuple[str, ...],
    batch_kind: str,
    context: pd.DataFrame,
    role: str,
    cell_context: str,
    phase: str,
):
    channel_mask = cond["channel_mask"]
    assert torch.is_tensor(channel_mask)
    cell_spec = resolve_cell_execution_spec(
        role=role,
        context=cell_context,
        phase=phase,
    )
    mapping = build_metadata_intervention_mapping(
        context,
        context=cell_spec.intervention,
    )
    tokens = tuple(
        f"{prefix}_{hashlib.sha256(f'{prefix}:{run_id}:{index}'.encode()).hexdigest()}"
        for index, run_id in enumerate(run_ids)
    )
    return resolve_metadata_intervention_batch(
        subject_ids=("sub004",) * len(run_ids),
        electrode_types=("dry",) * len(run_ids),
        run_ids=run_ids,
        row_tokens=tokens,
        base_input_sha256s=tuple(
            metadata_calibration_base_input_sha256(x[index], channel_mask[index])
            for index in range(len(run_ids))
        ),
        mapping=mapping,
        sealed_block_context=context,
        cell_spec=cell_spec,
        phase=phase,
        batch_kind=batch_kind,
    )


def _inputs():
    torch.manual_seed(22)
    query_x = torch.randn(4, 64, 400)
    support_x = torch.randn(6, 64, 400)
    labels = torch.tensor([0, 1, 2, 0, 1, 2])
    return query_x, _cond(4), support_x, _cond(6), labels


def test_aq_and_all_missing_aqm_are_bitwise_equal_from_raw_inputs() -> None:
    model = _composite().eval()
    query_x, query_cond, support_x, support_cond, labels = _inputs()
    aq = model(
        query_x=query_x,
        query_cond=query_cond,
        support_x=support_x,
        support_cond=support_cond,
        support_labels=labels,
        q_mode="observed",
        query_metadata_mode="off",
        support_metadata_mode="off",
    )
    missing = model(
        query_x=query_x,
        query_cond=query_cond,
        support_x=support_x,
        support_cond=support_cond,
        support_labels=labels,
        q_mode="observed",
        query_metadata_mode="all_missing",
        support_metadata_mode="all_missing",
    )

    assert torch.equal(aq.logits, missing.logits)
    assert torch.equal(aq.probabilities, missing.probabilities)
    assert torch.equal(aq.query_precision, missing.query_precision)
    assert torch.equal(aq.support_precision, missing.support_precision)


def test_off_paths_do_not_read_q_or_m_raw_fields() -> None:
    model = _composite().eval()
    query_x, query_cond, support_x, support_cond, labels = _inputs()
    query_structural = {
        "channel_ids": query_cond["channel_ids"],
        "channel_mask": query_cond["channel_mask"],
    }
    support_structural = {
        "channel_ids": support_cond["channel_ids"],
        "channel_mask": support_cond["channel_mask"],
    }
    a0 = model(
        query_x=query_x,
        query_cond=query_structural,
        support_x=support_x,
        support_cond=support_structural,
        support_labels=labels,
        q_mode="off",
        query_metadata_mode="off",
        support_metadata_mode="off",
    )
    assert torch.isfinite(a0.probabilities).all()

    query_m = {
        key: value
        for key, value in query_cond.items()
        if key not in {"channel_query_qc", "channel_query_qc_missing", "sfreq_processed_float"}
    }
    support_m = {
        key: value
        for key, value in support_cond.items()
        if key not in {"channel_query_qc", "channel_query_qc_missing", "sfreq_processed_float"}
    }
    am = model(
        query_x=query_x,
        query_cond=query_m,
        support_x=support_x,
        support_cond=support_m,
        support_labels=labels,
        q_mode="off",
        query_metadata_mode="observed",
        support_metadata_mode="observed",
        query_electrode_types=["dry"] * 4,
        support_electrode_types=["dry"] * 6,
    )
    assert am.probabilities.shape == a0.probabilities.shape


def test_observed_m_changes_only_prior_path_and_stage2_trainables_are_exact() -> None:
    model = _composite().eval()
    query_x, query_cond, support_x, support_cond, labels = _inputs()
    aq = model(
        query_x=query_x,
        query_cond=query_cond,
        support_x=support_x,
        support_cond=support_cond,
        support_labels=labels,
        q_mode="observed",
    )
    with torch.no_grad():
        model.prior.metadata_precision_encoder[-1].weight.fill_(0.3)
        model.prior.metadata_precision_encoder[-1].bias.fill_(0.1)
    aqm = model(
        query_x=query_x,
        query_cond=query_cond,
        support_x=support_x,
        support_cond=support_cond,
        support_labels=labels,
        q_mode="observed",
        query_metadata_mode="observed",
        support_metadata_mode="observed",
        query_electrode_types=["dry"] * 4,
        support_electrode_types=["dry"] * 6,
    )
    assert not torch.equal(aq.probabilities, aqm.probabilities)

    model.freeze_common_path_for_stage2()
    model.train()
    trainable = model.stage2_trainable_parameter_names()
    assert trainable
    assert all(name.startswith("prior.metadata_precision_encoder.") for name in trainable)
    assert model.backbone.training is False
    assert set(model.state_dict()).issuperset(
        {"backbone.cls", "prior.source_anchor", "prior.metadata_precision_encoder.2.weight"}
    )


def test_k0_empty_support_never_requires_support_metadata_or_q() -> None:
    model = _composite().eval()
    query_x, query_cond, _support_x, _support_cond, _labels = _inputs()
    output = model(
        query_x=query_x,
        query_cond=query_cond,
        support_x=torch.empty(0, 64, 400),
        support_cond={},
        support_labels=torch.empty(0, dtype=torch.long),
        q_mode="observed",
        query_metadata_mode="observed",
        support_metadata_mode="observed",
        query_electrode_types=["wet"] * 4,
        support_electrode_types=[],
    )
    assert output.posterior_mean.shape == (3, 16)
    assert output.support_precision.shape == (0, 16)


def test_interface_and_impedance_diagnostics_use_declared_feature_masks() -> None:
    model = _composite().eval()
    query_x, query_cond, support_x, support_cond, labels = _inputs()
    with torch.no_grad():
        model.prior.metadata_precision_encoder[-1].weight.fill_(0.2)
    common = {
        "query_x": query_x,
        "query_cond": query_cond,
        "support_x": support_x,
        "support_cond": support_cond,
        "support_labels": labels,
        "q_mode": "observed",
        "query_metadata_mode": "observed",
        "support_metadata_mode": "observed",
        "query_electrode_types": ["dry"] * 4,
        "support_electrode_types": ["dry"] * 6,
    }
    full = model(**common)
    interface = model(
        **common,
        query_metadata_ablation="interface_only",
        support_metadata_ablation="interface_only",
    )
    impedance = model(
        **common,
        query_metadata_ablation="impedance_only",
        support_metadata_ablation="impedance_only",
    )
    assert not torch.equal(full.query_precision, interface.query_precision)
    assert not torch.equal(full.query_precision, impedance.query_precision)


def test_forward_cell_consumes_resolved_opposite_interface_and_frozen_modes() -> None:
    model = _composite().eval()
    torch.manual_seed(23)
    query_x = torch.randn(1, 64, 400)
    support_x = torch.randn(1, 64, 400)
    query_cond = _cond(1)
    support_cond = _cond(1)
    context = _block_context()
    query_resolved = _resolved_row(
        x=query_x,
        cond=query_cond,
        prefix="q",
        run_id="block06",
        batch_kind="query",
        context=context,
    )
    support_resolved = _resolved_row(
        x=support_x,
        cond=support_cond,
        prefix="s",
        run_id="block01",
        batch_kind="support",
        context=context,
    )
    cell_spec = resolve_cell_execution_spec(
        role="A_QM",
        context="opposite_interface",
        phase="held_participant_evaluation",
    )

    with patch.object(model, "forward", wraps=model.forward) as low_level:
        output = model.forward_cell(
            query_x=query_x,
            query_cond=query_cond,
            query_resolved=query_resolved,
            support_x=support_x,
            support_cond=support_cond,
            support_labels=torch.tensor([0]),
            support_resolved=support_resolved,
            cell_spec=cell_spec,
            phase="held_participant_evaluation",
        )

    called = low_level.call_args.kwargs
    assert called["query_electrode_types"] == ("wet",)
    assert called["support_electrode_types"] == ("wet",)
    assert called["query_metadata_mode"] == "observed"
    assert called["support_metadata_mode"] == "observed"
    assert called["query_metadata_ablation"] == "full"
    assert output.probabilities.shape == (1, 3)

    cached = model.forward_cell_from_precomputed(
        query_precomputed=model.prepare_evaluation_features(
            query_x,
            query_cond,
            row_tokens=query_resolved.row_tokens,
            base_input_sha256s=query_resolved.base_input_sha256s,
            batch_kind="query",
        ),
        query_resolved=query_resolved,
        support_precomputed=model.prepare_evaluation_features(
            support_x,
            support_cond,
            row_tokens=support_resolved.row_tokens,
            base_input_sha256s=support_resolved.base_input_sha256s,
            batch_kind="support",
        ),
        support_labels=torch.tensor([0]),
        support_resolved=support_resolved,
        cell_spec=cell_spec,
        phase="held_participant_evaluation",
    )
    assert torch.equal(cached.logits, output.logits)
    assert torch.equal(cached.probabilities, output.probabilities)

    empty_x = torch.empty(0, 64, 400)
    empty_prepared = model.prepare_evaluation_features(
        empty_x,
        {},
        row_tokens=(),
        base_input_sha256s=(),
        batch_kind="support",
    )
    reference_k0 = model.forward_cell(
        query_x=query_x,
        query_cond=query_cond,
        query_resolved=query_resolved,
        support_x=empty_x,
        support_cond={},
        support_labels=torch.empty(0, dtype=torch.long),
        support_resolved=None,
        cell_spec=cell_spec,
        phase="held_participant_evaluation",
    )
    cached_k0 = model.forward_cell_from_precomputed(
        query_precomputed=model.prepare_evaluation_features(
            query_x,
            query_cond,
            row_tokens=query_resolved.row_tokens,
            base_input_sha256s=query_resolved.base_input_sha256s,
            batch_kind="query",
        ),
        query_resolved=query_resolved,
        support_precomputed=empty_prepared,
        support_labels=torch.empty(0, dtype=torch.long),
        support_resolved=None,
        cell_spec=cell_spec,
        phase="held_participant_evaluation",
    )
    assert torch.equal(cached_k0.probabilities, reference_k0.probabilities)


def test_prepared_evaluation_features_reject_cross_model_and_mutated_input() -> None:
    model = _composite().eval()
    other = _composite().eval()
    query_x = torch.zeros(1, 64, 400)
    query_cond = _cond(1)
    context = _block_context()
    query_resolved = _resolved_row(
        x=query_x,
        cond=query_cond,
        prefix="q",
        run_id="block06",
        batch_kind="query",
        context=context,
    )
    cell_spec = resolve_cell_execution_spec(
        role="A_QM",
        context="opposite_interface",
        phase="held_participant_evaluation",
    )
    query_prepared = model.prepare_evaluation_features(
        query_x,
        query_cond,
        row_tokens=query_resolved.row_tokens,
        base_input_sha256s=query_resolved.base_input_sha256s,
        batch_kind="query",
    )
    assert set(query_prepared.condition) == {
        "channel_ids",
        "channel_mask",
        "channel_query_qc",
        "channel_query_qc_missing",
        "sfreq_processed_float",
        "metadata_contract_version",
    }
    empty_x = torch.empty(0, 64, 400)
    empty_prepared = model.prepare_evaluation_features(
        empty_x,
        {},
        row_tokens=(),
        base_input_sha256s=(),
        batch_kind="support",
    )
    call = {
        "query_precomputed": query_prepared,
        "query_resolved": query_resolved,
        "support_precomputed": empty_prepared,
        "support_labels": torch.empty(0, dtype=torch.long),
        "support_resolved": None,
        "cell_spec": cell_spec,
        "phase": "held_participant_evaluation",
    }
    with pytest.raises(ValueError, match="Precomputed EEG features"):
        other.forward_cell_from_precomputed(**call)

    impedance = query_cond["channel_impedance"]
    assert torch.is_tensor(impedance)
    impedance.add_(1.0)
    assert model.forward_cell_from_precomputed(**call).probabilities.shape == (1, 3)

    query_x.add_(1.0)
    with pytest.raises(ValueError, match="Precomputed EEG features"):
        model.forward_cell_from_precomputed(**call)


@pytest.mark.parametrize(
    "phase",
    ["source_development", "held_participant_evaluation"],
)
@torch.no_grad()
def test_every_governed_cell_and_budget_is_bitwise_equal_when_precomputed(
    phase: str,
) -> None:
    model = _composite().eval()
    context = _block_context()
    torch.manual_seed(24)
    query_x = torch.randn(2, 64, 400)
    query_cond = _cond(2)
    for role, cell_context in execution_cells_for_phase(phase=phase):
        query_resolved = _resolved_batch(
            x=query_x,
            cond=query_cond,
            prefix="q",
            run_ids=("block06", "block07"),
            batch_kind="query",
            context=context,
            role=role,
            cell_context=cell_context,
            phase=phase,
        )
        cell_spec = resolve_cell_execution_spec(
            role=role,
            context=cell_context,
            phase=phase,
        )
        query_precomputed = model.prepare_evaluation_features(
            query_x,
            query_cond,
            row_tokens=query_resolved.row_tokens,
            base_input_sha256s=query_resolved.base_input_sha256s,
            batch_kind="query",
        )
        for budget in (0, 1, 3, 5):
            support_x = torch.randn(budget, 64, 400)
            support_cond = _cond(budget) if budget else {}
            support_labels = torch.arange(budget, dtype=torch.long).remainder(3)
            support_resolved = None
            if budget:
                support_resolved = _resolved_batch(
                    x=support_x,
                    cond=support_cond,
                    prefix="s",
                    run_ids=tuple(f"block{index:02d}" for index in range(1, budget + 1)),
                    batch_kind="support",
                    context=context,
                    role=role,
                    cell_context=cell_context,
                    phase=phase,
                )
            reference = model.forward_cell(
                query_x=query_x,
                query_cond=query_cond,
                query_resolved=query_resolved,
                support_x=support_x,
                support_cond=support_cond,
                support_labels=support_labels,
                support_resolved=support_resolved,
                cell_spec=cell_spec,
                phase=phase,
            )
            cached = model.forward_cell_from_precomputed(
                query_precomputed=query_precomputed,
                query_resolved=query_resolved,
                support_precomputed=model.prepare_evaluation_features(
                    support_x,
                    support_cond,
                    row_tokens=(
                        () if support_resolved is None else support_resolved.row_tokens
                    ),
                    base_input_sha256s=(
                        ()
                        if support_resolved is None
                        else support_resolved.base_input_sha256s
                    ),
                    batch_kind="support",
                ),
                support_labels=support_labels,
                support_resolved=support_resolved,
                cell_spec=cell_spec,
                phase=phase,
            )
            assert torch.equal(cached.logits, reference.logits)
            assert torch.equal(cached.probabilities, reference.probabilities)


def test_forward_cell_rejects_missing_support_resolution_and_phase_drift() -> None:
    model = _composite().eval()
    query_x = torch.zeros(1, 64, 400)
    support_x = torch.zeros(1, 64, 400)
    query_cond = _cond(1)
    support_cond = _cond(1)
    context = _block_context()
    query_resolved = _resolved_row(
        x=query_x,
        cond=query_cond,
        prefix="q",
        run_id="block06",
        batch_kind="query",
        context=context,
    )
    cell_spec = resolve_cell_execution_spec(
        role="A_QM",
        context="opposite_interface",
        phase="held_participant_evaluation",
    )
    common = {
        "query_x": query_x,
        "query_cond": query_cond,
        "query_resolved": query_resolved,
        "support_x": support_x,
        "support_cond": support_cond,
        "support_labels": torch.tensor([0]),
        "support_resolved": None,
        "cell_spec": cell_spec,
    }

    with pytest.raises(ValueError, match="requires governed resolved metadata"):
        model.forward_cell(**common, phase="held_participant_evaluation")
    with pytest.raises(ValueError, match="phase or batch kind"):
        model.forward_cell(**common, phase="source_development")
