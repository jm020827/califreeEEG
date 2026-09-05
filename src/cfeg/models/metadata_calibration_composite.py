from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Literal

import torch
from torch import nn

from cfeg.data.metadata_calibration_features import (
    WEARABLE_METADATA_FEATURE_NAMES,
    WEARABLE_Q_FEATURE_NAMES,
    MetadataFeatureAblation,
    apply_metadata_feature_ablation,
    build_wearable_metadata_features,
    build_wearable_q_features,
)
from cfeg.data.metadata_calibration_interventions import (
    ResolvedMetadataInterventionBatch,
    apply_prevalidated_resolved_metadata_to_condition,
    apply_resolved_metadata_to_condition,
    metadata_calibration_base_input_sha256,
)
from cfeg.metadata_calibration_execution import (
    CellExecutionSpec,
    ExperimentPhase,
    resolve_cell_execution_spec,
)
from cfeg.models.backbones.spectral_transformer import SpectralEEGTransformerBackbone
from cfeg.models.metadata_calibration_prior import (
    BoundedMetadataCalibrationPrior,
    MetadataCalibrationOutput,
    QualityMode,
)

MetadataInputMode = Literal["off", "observed", "all_missing"]


@dataclass(frozen=True)
class PreparedCalibrationEEG:
    """M-independent spectral and Q features cached for governed evaluation.

    ``condition`` contains only structural and EEG-derived-Q inputs. Raw
    acquisition-context M is deliberately excluded from this capability.
    """

    x: torch.Tensor
    condition: Mapping[str, Any]
    embeddings: torch.Tensor
    observed_q: torch.Tensor
    row_tokens: tuple[str, ...]
    base_input_sha256s: tuple[str, ...]
    batch_kind: Literal["query", "support"]
    owner_nonce: object
    x_signature: tuple[Any, ...]
    condition_tensor_signatures: tuple[tuple[str, tuple[Any, ...]], ...]
    model_state_signatures: tuple[tuple[str, tuple[Any, ...]], ...]
    embedding_signature: tuple[Any, ...]
    observed_q_signature: tuple[Any, ...]


class MetadataCalibrationComposite(nn.Module):
    """One checkpoint containing the common spectral path and bounded M prior.

    The backbone receives a newly constructed dictionary containing channel IDs
    and masks only.  Q and M are built through separate access paths, and an
    ``off`` path does not inspect the corresponding raw inputs.
    """

    def __init__(
        self,
        *,
        backbone: SpectralEEGTransformerBackbone,
        prior: BoundedMetadataCalibrationPrior,
    ) -> None:
        super().__init__()
        if not isinstance(backbone, SpectralEEGTransformerBackbone):
            raise TypeError("Metadata calibration is frozen to SpectralEEGTransformerBackbone.")
        if prior.embedding_dim != backbone.d_model:
            raise ValueError("Backbone output width and prior embedding width must match.")
        if prior.q_feature_dim != len(WEARABLE_Q_FEATURE_NAMES):
            raise ValueError("Prior Q width differs from the frozen wearable feature schema.")
        if prior.metadata_feature_dim != len(WEARABLE_METADATA_FEATURE_NAMES):
            raise ValueError("Prior M width differs from the frozen wearable feature schema.")
        self.backbone = backbone
        self.prior = prior
        self._stage2_common_frozen = False
        self._evaluation_cache_nonce = object()

    def freeze_common_path_for_stage2(self) -> None:
        """Freeze backbone, anchors and Q path; leave only the M residual trainable."""

        for parameter in self.backbone.parameters():
            parameter.requires_grad_(False)
        self.prior.freeze_common_path_for_stage2()
        self._stage2_common_frozen = True
        self.backbone.eval()

    def stage2_trainable_parameter_names(self) -> tuple[str, ...]:
        return tuple(name for name, parameter in self.named_parameters() if parameter.requires_grad)

    def train(self, mode: bool = True):
        super().train(mode)
        if self._stage2_common_frozen:
            self.backbone.eval()
        return self

    def encode(
        self,
        x: torch.Tensor,
        cond: Mapping[str, Any],
    ) -> torch.Tensor:
        """Encode EEG while making future metadata consumption by the backbone impossible."""

        if x.ndim != 3:
            raise ValueError("EEG input must have shape [batch,channels,time].")
        if x.shape[0] == 0:
            return torch.empty(
                (0, self.backbone.d_model),
                dtype=x.dtype,
                device=x.device,
            )
        structural_cond = {
            "channel_ids": _aligned_tensor(cond, "channel_ids", x),
            "channel_mask": _aligned_tensor(cond, "channel_mask", x),
        }
        return self.backbone(
            x,
            cond=structural_cond,
            prompt_tokens=None,
            return_tokens=False,
        ).h

    @torch.no_grad()
    def prepare_evaluation_features(
        self,
        x: torch.Tensor,
        cond: Mapping[str, Any],
        *,
        row_tokens: Sequence[str],
        base_input_sha256s: Sequence[str],
        batch_kind: Literal["query", "support"],
    ) -> PreparedCalibrationEEG:
        """Compute the paths that are invariant to governed M interventions."""

        if self.training:
            raise RuntimeError("Prepared evaluation features require model.eval().")
        tokens = tuple(map(str, row_tokens))
        expected_hashes = tuple(map(str, base_input_sha256s))
        if x.shape[0] == 0:
            observed_hashes: tuple[str, ...] = ()
        else:
            channel_mask = _aligned_tensor(cond, "channel_mask", x)
            x_array = x.detach().contiguous().cpu().numpy()
            channel_mask_array = channel_mask.detach().contiguous().cpu().numpy()
            observed_hashes = tuple(
                metadata_calibration_base_input_sha256(
                    x_array[index], channel_mask_array[index]
                )
                for index in range(x.shape[0])
            )
        if (
            batch_kind not in {"query", "support"}
            or len(tokens) != x.shape[0]
            or len(set(tokens)) != len(tokens)
            or observed_hashes != expected_hashes
        ):
            raise ValueError("Prepared EEG row identities do not bind the exact input batch.")
        condition = MappingProxyType(
            {
                name: cond[name]
                for name in (
                    "channel_ids",
                    "channel_mask",
                    "channel_query_qc",
                    "channel_query_qc_missing",
                    "sfreq_processed_float",
                    "metadata_contract_version",
                )
                if name in cond
            }
        )
        embeddings = self.encode(x, condition)
        observed_q = self._q_features(x, condition, mode="observed")
        model_state_signatures = _model_state_signatures(self)
        prepared = PreparedCalibrationEEG(
            x=x,
            condition=condition,
            embeddings=embeddings,
            observed_q=observed_q,
            row_tokens=tokens,
            base_input_sha256s=expected_hashes,
            batch_kind=batch_kind,
            owner_nonce=self._evaluation_cache_nonce,
            x_signature=_tensor_signature(x),
            condition_tensor_signatures=_mapping_tensor_signatures(condition),
            model_state_signatures=model_state_signatures,
            embedding_signature=_tensor_signature(embeddings),
            observed_q_signature=_tensor_signature(observed_q),
        )
        self._validate_prepared_features(
            prepared,
            require_observed_q=True,
            current_model_state_signatures=model_state_signatures,
            check_finite=True,
        )
        return prepared

    def forward(
        self,
        *,
        query_x: torch.Tensor,
        query_cond: Mapping[str, Any],
        support_x: torch.Tensor,
        support_cond: Mapping[str, Any],
        support_labels: torch.Tensor,
        q_mode: QualityMode = "observed",
        query_q_mode: QualityMode | None = None,
        support_q_mode: QualityMode | None = None,
        query_metadata_mode: MetadataInputMode = "off",
        support_metadata_mode: MetadataInputMode = "off",
        query_electrode_types: Sequence[str | None] | None = None,
        support_electrode_types: Sequence[str | None] | None = None,
        query_metadata_ablation: MetadataFeatureAblation = "full",
        support_metadata_ablation: MetadataFeatureAblation = "full",
    ) -> MetadataCalibrationOutput:
        """Low-level feature-path primitive; governed execution uses ``forward_cell``."""

        query_quality_mode = q_mode if query_q_mode is None else query_q_mode
        support_quality_mode = q_mode if support_q_mode is None else support_q_mode
        query_embeddings = self.encode(query_x, query_cond)
        support_embeddings = self.encode(support_x, support_cond)

        query_q = self._q_features(query_x, query_cond, mode=query_quality_mode)
        support_q = self._q_features(support_x, support_cond, mode=support_quality_mode)
        query_m, query_missing, query_prior_mode = self._metadata_features(
            query_cond,
            n_rows=query_x.shape[0],
            mode=query_metadata_mode,
            electrode_types=query_electrode_types,
            ablation=query_metadata_ablation,
            reference=query_embeddings,
        )
        support_m, support_missing, support_prior_mode = self._metadata_features(
            support_cond,
            n_rows=support_x.shape[0],
            mode=support_metadata_mode,
            electrode_types=support_electrode_types,
            ablation=support_metadata_ablation,
            reference=support_embeddings,
        )
        return self.prior(
            query_embeddings,
            query_q,
            support_embeddings=support_embeddings,
            support_labels=support_labels,
            support_q_features=support_q,
            query_metadata_values=query_m,
            query_metadata_missing=query_missing,
            support_metadata_values=support_m,
            support_metadata_missing=support_missing,
            query_q_mode=query_quality_mode,
            support_q_mode=support_quality_mode,
            query_metadata_mode=query_prior_mode,
            support_metadata_mode=support_prior_mode,
        )

    def forward_cell(
        self,
        *,
        query_x: torch.Tensor,
        query_cond: Mapping[str, Any],
        query_resolved: ResolvedMetadataInterventionBatch,
        support_x: torch.Tensor,
        support_cond: Mapping[str, Any],
        support_labels: torch.Tensor,
        support_resolved: ResolvedMetadataInterventionBatch | None,
        cell_spec: CellExecutionSpec,
        phase: ExperimentPhase,
    ) -> MetadataCalibrationOutput:
        """Run one canonical cell without caller-controlled M values or modes.

        The raw ``forward`` method remains useful for isolated component tests and
        source training. Confirmatory producers must enter here: this method
        verifies the phase/cell binding, applies the resolved query and support
        context immediately before feature construction, and derives every Q/M
        mode and feature ablation from the frozen dispatch table.
        """

        if not isinstance(cell_spec, CellExecutionSpec):
            raise TypeError("cell_spec must be one frozen CellExecutionSpec.")
        canonical_spec = resolve_cell_execution_spec(
            role=cell_spec.role,
            context=cell_spec.context,
            phase=phase,
        )
        if cell_spec != canonical_spec:
            raise ValueError("Cell specification differs from the frozen execution dispatch.")
        self._validate_resolved_cell_input(
            query_resolved,
            cell_spec=canonical_spec,
            phase=phase,
            batch_kind="query",
        )
        query_inputs = apply_resolved_metadata_to_condition(
            query_x,
            query_cond,
            query_resolved,
        )

        if support_x.ndim != 3 or support_labels.ndim != 1 or (
            support_labels.shape[0] != support_x.shape[0]
        ):
            raise ValueError("Support EEG and labels must have one aligned batch dimension.")
        if support_x.shape[0] == 0:
            if support_resolved is not None:
                raise ValueError("An empty k=0 support set must not carry resolved metadata.")
            support_condition: Mapping[str, Any] = support_cond
            support_electrode_types: Sequence[str | None] = ()
        else:
            if support_resolved is None:
                raise ValueError("A nonempty support set requires governed resolved metadata.")
            self._validate_resolved_cell_input(
                support_resolved,
                cell_spec=canonical_spec,
                phase=phase,
                batch_kind="support",
            )
            if support_resolved.mapping_sha256 != query_resolved.mapping_sha256:
                raise ValueError("Query and support must use the same intervention mapping.")
            support_inputs = apply_resolved_metadata_to_condition(
                support_x,
                support_cond,
                support_resolved,
            )
            support_condition = support_inputs.condition
            support_electrode_types = support_inputs.electrode_types

        return self.forward(
            query_x=query_x,
            query_cond=query_inputs.condition,
            support_x=support_x,
            support_cond=support_condition,
            support_labels=support_labels,
            query_q_mode=canonical_spec.query_q_mode,
            support_q_mode=canonical_spec.support_q_mode,
            query_metadata_mode=canonical_spec.query_metadata_mode,
            support_metadata_mode=canonical_spec.support_metadata_mode,
            query_electrode_types=query_inputs.electrode_types,
            support_electrode_types=support_electrode_types,
            query_metadata_ablation=canonical_spec.query_metadata_ablation,
            support_metadata_ablation=canonical_spec.support_metadata_ablation,
        )

    @torch.no_grad()
    def forward_cell_from_precomputed(
        self,
        *,
        query_precomputed: PreparedCalibrationEEG,
        query_resolved: ResolvedMetadataInterventionBatch,
        support_precomputed: PreparedCalibrationEEG,
        support_labels: torch.Tensor,
        support_resolved: ResolvedMetadataInterventionBatch | None,
        cell_spec: CellExecutionSpec,
        phase: ExperimentPhase,
    ) -> MetadataCalibrationOutput:
        """Evaluate a frozen cell while reusing only M-independent EEG features."""

        if self.training:
            raise RuntimeError("Prepared cell evaluation requires model.eval().")
        if not isinstance(cell_spec, CellExecutionSpec):
            raise TypeError("cell_spec must be one frozen CellExecutionSpec.")
        canonical_spec = resolve_cell_execution_spec(
            role=cell_spec.role,
            context=cell_spec.context,
            phase=phase,
        )
        if cell_spec != canonical_spec:
            raise ValueError("Cell specification differs from the frozen execution dispatch.")
        model_state_signatures = _model_state_signatures(self)
        self._validate_prepared_features(
            query_precomputed,
            require_observed_q=canonical_spec.query_q_mode == "observed",
            current_model_state_signatures=model_state_signatures,
        )
        self._validate_prepared_features(
            support_precomputed,
            require_observed_q=canonical_spec.support_q_mode == "observed",
            current_model_state_signatures=model_state_signatures,
        )
        query_x = query_precomputed.x
        query_cond = query_precomputed.condition
        support_x = support_precomputed.x
        support_cond = support_precomputed.condition
        self._validate_resolved_cell_input(
            query_resolved,
            cell_spec=canonical_spec,
            phase=phase,
            batch_kind="query",
        )
        if (
            query_precomputed.batch_kind != "query"
            or query_precomputed.row_tokens != query_resolved.row_tokens
            or query_precomputed.base_input_sha256s
            != query_resolved.base_input_sha256s
        ):
            raise ValueError("Prepared query identity differs from its resolved metadata.")
        query_inputs = apply_prevalidated_resolved_metadata_to_condition(
            query_x,
            query_cond,
            query_resolved,
            prevalidated_base_input_sha256s=query_precomputed.base_input_sha256s,
        )
        if support_x.ndim != 3 or support_labels.ndim != 1 or (
            support_labels.shape[0] != support_x.shape[0]
        ):
            raise ValueError("Support EEG and labels must have one aligned batch dimension.")
        if support_x.shape[0] == 0:
            if support_resolved is not None:
                raise ValueError("An empty k=0 support set must not carry resolved metadata.")
            if support_precomputed.batch_kind != "support" or (
                support_precomputed.row_tokens
                or support_precomputed.base_input_sha256s
            ):
                raise ValueError("Prepared k=0 support identity must be exactly empty.")
            support_condition: Mapping[str, Any] = support_cond
            support_electrode_types: Sequence[str | None] = ()
        else:
            if support_resolved is None:
                raise ValueError("A nonempty support set requires governed resolved metadata.")
            self._validate_resolved_cell_input(
                support_resolved,
                cell_spec=canonical_spec,
                phase=phase,
                batch_kind="support",
            )
            if support_resolved.mapping_sha256 != query_resolved.mapping_sha256:
                raise ValueError("Query and support must use the same intervention mapping.")
            if (
                support_precomputed.batch_kind != "support"
                or support_precomputed.row_tokens != support_resolved.row_tokens
                or support_precomputed.base_input_sha256s
                != support_resolved.base_input_sha256s
            ):
                raise ValueError("Prepared support identity differs from resolved metadata.")
            support_inputs = apply_prevalidated_resolved_metadata_to_condition(
                support_x,
                support_cond,
                support_resolved,
                prevalidated_base_input_sha256s=(
                    support_precomputed.base_input_sha256s
                ),
            )
            support_condition = support_inputs.condition
            support_electrode_types = support_inputs.electrode_types

        query_q = self._select_precomputed_q(
            query_precomputed,
            mode=canonical_spec.query_q_mode,
        )
        support_q = self._select_precomputed_q(
            support_precomputed,
            mode=canonical_spec.support_q_mode,
        )
        query_m, query_missing, query_prior_mode = self._metadata_features(
            query_inputs.condition,
            n_rows=query_x.shape[0],
            mode=canonical_spec.query_metadata_mode,
            electrode_types=query_inputs.electrode_types,
            ablation=canonical_spec.query_metadata_ablation,
            reference=query_precomputed.embeddings,
        )
        support_m, support_missing, support_prior_mode = self._metadata_features(
            support_condition,
            n_rows=support_x.shape[0],
            mode=canonical_spec.support_metadata_mode,
            electrode_types=support_electrode_types,
            ablation=canonical_spec.support_metadata_ablation,
            reference=support_precomputed.embeddings,
        )
        return self.prior(
            query_precomputed.embeddings,
            query_q,
            support_embeddings=support_precomputed.embeddings,
            support_labels=support_labels,
            support_q_features=support_q,
            query_metadata_values=query_m,
            query_metadata_missing=query_missing,
            support_metadata_values=support_m,
            support_metadata_missing=support_missing,
            query_q_mode=canonical_spec.query_q_mode,
            support_q_mode=canonical_spec.support_q_mode,
            query_metadata_mode=query_prior_mode,
            support_metadata_mode=support_prior_mode,
        )

    def _validate_prepared_features(
        self,
        prepared: PreparedCalibrationEEG,
        *,
        require_observed_q: bool,
        current_model_state_signatures: tuple[tuple[str, tuple[Any, ...]], ...],
        check_finite: bool = False,
    ) -> None:
        if not isinstance(prepared, PreparedCalibrationEEG):
            raise TypeError("Expected one model-prepared EEG feature batch.")
        x = prepared.x
        if (
            prepared.owner_nonce is not self._evaluation_cache_nonce
            or prepared.x_signature != _tensor_signature(x)
            or prepared.condition_tensor_signatures
            != _mapping_tensor_signatures(prepared.condition)
            or prepared.model_state_signatures != current_model_state_signatures
            or prepared.embedding_signature
            != _tensor_signature(prepared.embeddings)
            or prepared.observed_q_signature
            != _tensor_signature(prepared.observed_q)
            or self.training
            or x.ndim != 3
            or prepared.embeddings.shape != (x.shape[0], self.backbone.d_model)
            or prepared.observed_q.shape
            != (x.shape[0], len(WEARABLE_Q_FEATURE_NAMES))
            or prepared.embeddings.device != x.device
            or prepared.observed_q.device != x.device
            or len(prepared.row_tokens) != x.shape[0]
            or len(prepared.base_input_sha256s) != x.shape[0]
            or prepared.batch_kind not in {"query", "support"}
            or (
                check_finite
                and (
                    not torch.isfinite(prepared.embeddings).all()
                    or (
                        require_observed_q
                        and not torch.isfinite(prepared.observed_q).all()
                    )
                )
            )
        ):
            raise ValueError("Precomputed EEG features do not align with the frozen model input.")

    @staticmethod
    def _select_precomputed_q(
        prepared: PreparedCalibrationEEG,
        *,
        mode: QualityMode,
    ) -> torch.Tensor:
        if mode == "observed":
            return prepared.observed_q
        if mode == "off":
            return torch.zeros(
                (prepared.embeddings.shape[0], len(WEARABLE_Q_FEATURE_NAMES)),
                dtype=torch.float32,
                device=prepared.embeddings.device,
            )
        raise ValueError("Q mode must be 'off' or 'observed'.")

    @staticmethod
    def _validate_resolved_cell_input(
        resolved: ResolvedMetadataInterventionBatch,
        *,
        cell_spec: CellExecutionSpec,
        phase: ExperimentPhase,
        batch_kind: Literal["query", "support"],
    ) -> None:
        if not isinstance(resolved, ResolvedMetadataInterventionBatch):
            raise TypeError("Governed model inputs require a resolved metadata batch.")
        if resolved.phase != phase or resolved.batch_kind != batch_kind:
            raise ValueError("Resolved metadata phase or batch kind differs from the call.")
        if resolved.cell_execution_spec != tuple(sorted(cell_spec.as_dict().items())):
            raise ValueError("Resolved metadata was not built for this canonical cell.")

    @staticmethod
    def _q_features(
        x: torch.Tensor,
        cond: Mapping[str, Any],
        *,
        mode: QualityMode,
    ) -> torch.Tensor:
        if mode == "off":
            return torch.zeros(
                (x.shape[0], len(WEARABLE_Q_FEATURE_NAMES)),
                dtype=torch.float32,
                device=x.device,
            )
        if mode != "observed":
            raise ValueError("Q mode must be 'off' or 'observed'.")
        if x.shape[0] == 0:
            return torch.empty(
                (0, len(WEARABLE_Q_FEATURE_NAMES)),
                dtype=torch.float32,
                device=x.device,
            )
        return build_wearable_q_features(x, cond)

    @staticmethod
    def _metadata_features(
        cond: Mapping[str, Any],
        *,
        n_rows: int,
        mode: MetadataInputMode,
        electrode_types: Sequence[str | None] | None,
        ablation: MetadataFeatureAblation,
        reference: torch.Tensor,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None, Literal["off", "observed"]]:
        if mode == "off":
            return None, None, "off"
        if mode == "all_missing":
            values = torch.zeros(
                (n_rows, len(WEARABLE_METADATA_FEATURE_NAMES)),
                dtype=torch.float32,
                device=reference.device,
            )
            return values, torch.ones_like(values, dtype=torch.bool), "observed"
        if mode != "observed":
            raise ValueError("Metadata mode must be 'off', 'observed', or 'all_missing'.")
        if electrode_types is None:
            raise ValueError("Observed metadata requires governed electrode-interface strings.")
        if n_rows == 0:
            if len(electrode_types) != 0:
                raise ValueError("Empty support requires an empty electrode-interface sequence.")
            values = torch.empty(
                (0, len(WEARABLE_METADATA_FEATURE_NAMES)),
                dtype=torch.float32,
                device=reference.device,
            )
            return values, torch.empty_like(values, dtype=torch.bool), "observed"
        values, missing = build_wearable_metadata_features(
            cond,
            electrode_types=electrode_types,
        )
        values, missing = apply_metadata_feature_ablation(
            values,
            missing,
            ablation=ablation,
        )
        return values, missing, "observed"


def _aligned_tensor(
    cond: Mapping[str, Any],
    name: str,
    x: torch.Tensor,
) -> torch.Tensor:
    value = cond.get(name)
    if not torch.is_tensor(value) or value.shape != x.shape[:2] or value.device != x.device:
        raise ValueError(f"cond[{name!r}] must align with the EEG batch and device.")
    return value


def _tensor_signature(value: torch.Tensor) -> tuple[Any, ...]:
    return (
        id(value),
        value.data_ptr(),
        value._version,
        tuple(value.shape),
        str(value.dtype),
        str(value.device),
    )


def _mapping_tensor_signatures(
    values: Mapping[str, Any],
) -> tuple[tuple[str, tuple[Any, ...]], ...]:
    return tuple(
        (str(key), _tensor_signature(value))
        for key, value in sorted(values.items())
        if torch.is_tensor(value)
    )


def _model_state_signatures(
    model: nn.Module,
) -> tuple[tuple[str, tuple[Any, ...]], ...]:
    parameters = (
        (f"parameter:{name}", _tensor_signature(parameter))
        for name, parameter in model.named_parameters()
    )
    buffers = (
        (f"buffer:{name}", _tensor_signature(buffer))
        for name, buffer in model.named_buffers()
    )
    return (*parameters, *buffers)
