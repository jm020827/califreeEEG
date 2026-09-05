from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

import torch
import torch.nn.functional as F
from torch import nn

MetadataMode = Literal["off", "observed"]
QualityMode = Literal["off", "observed"]


@dataclass(frozen=True)
class MetadataCalibrationOutput:
    """Observable state of the diagonal Gaussian calibration head."""

    logits: torch.Tensor
    probabilities: torch.Tensor
    posterior_mean: torch.Tensor
    posterior_precision: torch.Tensor
    query_precision: torch.Tensor
    support_precision: torch.Tensor
    query_metadata_log_residual: torch.Tensor
    support_metadata_log_residual: torch.Tensor


class BoundedMetadataCalibrationPrior(nn.Module):
    """Few-shot class-anchor calibration with an exactly removable M residual.

    The module consumes frozen EEG embeddings.  Q controls diagonal observation
    precision.  M can only multiply that precision by a bounded, positive
    factor; it cannot transform a waveform, mix channels, or move a source
    anchor directly.  Labeled target support enters only through conjugate
    diagonal-Gaussian sufficient statistics.

    Stage 1 learns the source anchors, base precision, and Q path.  Calling
    :meth:`freeze_common_path_for_stage2` then leaves only the M residual
    trainable.  Runtime ``metadata_mode="off"`` returns the Q precision before
    inspecting M, which makes A_Q invariant to arbitrary metadata values.
    """

    def __init__(
        self,
        *,
        embedding_dim: int,
        n_classes: int,
        q_feature_dim: int,
        metadata_feature_dim: int,
        hidden_dim: int = 32,
        q_log_ratio_bound: float = math.log(4.0),
        metadata_log_ratio_bound: float = math.log(1.25),
        precision_floor: float = 0.05,
        precision_ceiling: float = 20.0,
        source_prior_strength: float = 1.0,
    ) -> None:
        super().__init__()
        if min(embedding_dim, n_classes, q_feature_dim, metadata_feature_dim, hidden_dim) <= 0:
            raise ValueError("All calibration-prior dimensions must be positive.")
        if q_log_ratio_bound <= 0.0 or metadata_log_ratio_bound <= 0.0:
            raise ValueError("Q and metadata log-ratio bounds must be positive.")
        if not 0.0 < precision_floor < precision_ceiling:
            raise ValueError("Precision bounds must satisfy 0 < floor < ceiling.")
        if source_prior_strength <= 0.0:
            raise ValueError("source_prior_strength must be positive.")

        self.embedding_dim = int(embedding_dim)
        self.n_classes = int(n_classes)
        self.q_feature_dim = int(q_feature_dim)
        self.metadata_feature_dim = int(metadata_feature_dim)
        self.q_log_ratio_bound = float(q_log_ratio_bound)
        self.metadata_log_ratio_bound = float(metadata_log_ratio_bound)
        self.precision_floor = float(precision_floor)
        self.precision_ceiling = float(precision_ceiling)
        self.source_prior_strength = float(source_prior_strength)

        self.source_anchor = nn.Parameter(torch.empty(n_classes, embedding_dim))
        self.class_bias = nn.Parameter(torch.zeros(n_classes))
        self.base_log_precision = nn.Parameter(torch.zeros(embedding_dim))
        self.q_precision_encoder = nn.Sequential(
            nn.Linear(q_feature_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, embedding_dim),
        )
        # Values and their availability flags are both explicit.  A missing
        # value is zero-filled only after its flag is retained.
        self.metadata_precision_encoder = nn.Sequential(
            nn.Linear(2 * metadata_feature_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, embedding_dim),
        )
        nn.init.normal_(self.source_anchor, mean=0.0, std=embedding_dim**-0.5)
        nn.init.zeros_(self.q_precision_encoder[-1].weight)
        nn.init.zeros_(self.q_precision_encoder[-1].bias)
        nn.init.zeros_(self.metadata_precision_encoder[-1].weight)
        nn.init.zeros_(self.metadata_precision_encoder[-1].bias)

    def common_parameters(self):
        yield self.source_anchor
        yield self.class_bias
        yield self.base_log_precision
        yield from self.q_precision_encoder.parameters()

    def metadata_parameters(self):
        yield from self.metadata_precision_encoder.parameters()

    def freeze_common_path_for_stage2(self) -> None:
        """Freeze every A_Q parameter and expose only the bounded M branch."""

        for parameter in self.common_parameters():
            parameter.requires_grad_(False)
        for parameter in self.metadata_parameters():
            parameter.requires_grad_(True)

    def precision(
        self,
        q_features: torch.Tensor,
        *,
        q_mode: QualityMode = "observed",
        metadata_values: torch.Tensor | None = None,
        metadata_missing: torch.Tensor | None = None,
        metadata_mode: MetadataMode = "off",
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return positive diagonal precision and the bounded M log residual."""

        if q_features.ndim != 2 or q_features.shape[1] != self.q_feature_dim:
            raise ValueError(
                f"q_features must have shape [sample,q_feature_dim], got {tuple(q_features.shape)}."
            )
        if q_mode not in {"off", "observed"}:
            raise ValueError("q_mode must be 'off' or 'observed'.")
        if metadata_mode not in {"off", "observed"}:
            raise ValueError("metadata_mode must be 'off' or 'observed'.")

        base_log = self.base_log_precision.clamp(
            min=math.log(self.precision_floor),
            max=math.log(self.precision_ceiling),
        )
        if q_mode == "off":
            q_precision = torch.exp(base_log).expand(q_features.shape[0], -1)
        else:
            if not torch.isfinite(q_features).all():
                raise ValueError("Observed q_features must be finite.")
            q_log_residual = self.q_log_ratio_bound * torch.tanh(
                self.q_precision_encoder(q_features)
            )
            q_precision = torch.exp(base_log + q_log_residual).clamp(
                min=self.precision_floor,
                max=self.precision_ceiling,
            )
        zero_residual = torch.zeros_like(q_precision)
        # This early return is the exact-off contract.  In particular, poison
        # values in an M tensor cannot affect or even be validated by A_Q.
        if metadata_mode == "off":
            return q_precision, zero_residual

        if metadata_values is None or metadata_missing is None:
            raise ValueError("Observed metadata mode requires values and missing flags.")
        expected = (q_features.shape[0], self.metadata_feature_dim)
        if metadata_values.shape != expected or metadata_missing.shape != expected:
            raise ValueError(f"metadata values/missing flags must both have shape {expected}.")
        metadata_missing = metadata_missing.bool()
        available = ~metadata_missing
        if not torch.isfinite(metadata_values.masked_select(available)).all():
            raise ValueError("Available metadata values must be finite.")
        safe_values = torch.where(available, metadata_values, torch.zeros_like(metadata_values))
        metadata_input = torch.cat([safe_values, available.to(safe_values.dtype)], dim=-1)
        raw_residual = self.metadata_log_ratio_bound * torch.tanh(
            self.metadata_precision_encoder(metadata_input)
        )
        any_available = available.any(dim=-1, keepdim=True)
        metadata_log_residual = torch.where(
            any_available,
            raw_residual,
            torch.zeros_like(raw_residual),
        )
        adjusted = (q_precision * torch.exp(metadata_log_residual)).clamp(
            min=self.precision_floor,
            max=self.precision_ceiling,
        )
        # Keep all-missing rows bitwise on the original Q tensor rather than
        # relying on exp(0) and a second clamp being numerically identical.
        precision = torch.where(any_available, adjusted, q_precision)
        return precision, metadata_log_residual

    def forward(
        self,
        query_embeddings: torch.Tensor,
        query_q_features: torch.Tensor,
        *,
        support_embeddings: torch.Tensor,
        support_labels: torch.Tensor,
        support_q_features: torch.Tensor,
        query_metadata_values: torch.Tensor | None = None,
        query_metadata_missing: torch.Tensor | None = None,
        support_metadata_values: torch.Tensor | None = None,
        support_metadata_missing: torch.Tensor | None = None,
        q_mode: QualityMode = "observed",
        metadata_mode: MetadataMode = "off",
        query_q_mode: QualityMode | None = None,
        support_q_mode: QualityMode | None = None,
        query_metadata_mode: MetadataMode | None = None,
        support_metadata_mode: MetadataMode | None = None,
    ) -> MetadataCalibrationOutput:
        """Score queries after a closed-form target-support update."""

        self._validate_embeddings(query_embeddings, name="query_embeddings")
        self._validate_embeddings(support_embeddings, name="support_embeddings")
        if support_labels.ndim != 1 or support_labels.shape[0] != support_embeddings.shape[0]:
            raise ValueError("support_labels must contain one label per support embedding.")
        if support_q_features.shape[0] != support_embeddings.shape[0]:
            raise ValueError("support_q_features must align with support embeddings.")
        if support_labels.numel() and (
            support_labels.min().item() < 0 or support_labels.max().item() >= self.n_classes
        ):
            raise ValueError("support_labels fall outside the configured class range.")

        query_mode = metadata_mode if query_metadata_mode is None else query_metadata_mode
        support_mode = metadata_mode if support_metadata_mode is None else support_metadata_mode
        query_quality_mode = q_mode if query_q_mode is None else query_q_mode
        support_quality_mode = q_mode if support_q_mode is None else support_q_mode
        query_precision, query_m_residual = self.precision(
            query_q_features,
            q_mode=query_quality_mode,
            metadata_values=query_metadata_values,
            metadata_missing=query_metadata_missing,
            metadata_mode=query_mode,
        )
        support_precision, support_m_residual = self.precision(
            support_q_features,
            q_mode=support_quality_mode,
            metadata_values=support_metadata_values,
            metadata_missing=support_metadata_missing,
            metadata_mode=support_mode,
        )

        query_z = F.normalize(query_embeddings, dim=-1, eps=1e-8)
        support_z = F.normalize(support_embeddings, dim=-1, eps=1e-8)
        source_anchor = F.normalize(self.source_anchor, dim=-1, eps=1e-8)
        base_precision = torch.exp(
            self.base_log_precision.clamp(
                min=math.log(self.precision_floor),
                max=math.log(self.precision_ceiling),
            )
        )
        prior_precision = self.source_prior_strength * base_precision
        prior_precision_by_class = prior_precision.unsqueeze(0).expand(
            self.n_classes, -1
        )
        prior_natural = prior_precision_by_class * source_anchor

        if support_embeddings.shape[0] > 0:
            support_precision_sum = torch.zeros_like(prior_precision_by_class).index_add(
                0,
                support_labels.long(),
                support_precision,
            )
            support_natural_sum = torch.zeros_like(prior_natural).index_add(
                0,
                support_labels.long(),
                support_precision * support_z,
            )
            posterior_precision = prior_precision_by_class + support_precision_sum
            posterior_natural = prior_natural + support_natural_sum
        else:
            posterior_precision = prior_precision_by_class
            posterior_natural = prior_natural
        posterior_mean = posterior_natural / posterior_precision

        predictive_variance = query_precision.reciprocal().unsqueeze(1) + (
            posterior_precision.reciprocal().unsqueeze(0)
        )
        squared_error = (query_z.unsqueeze(1) - posterior_mean.unsqueeze(0)).square()
        logits = -0.5 * (squared_error / predictive_variance + torch.log(predictive_variance)).sum(
            dim=-1
        )
        logits = logits + self.class_bias
        probabilities = torch.softmax(logits, dim=-1)
        return MetadataCalibrationOutput(
            logits=logits,
            probabilities=probabilities,
            posterior_mean=posterior_mean,
            posterior_precision=posterior_precision,
            query_precision=query_precision,
            support_precision=support_precision,
            query_metadata_log_residual=query_m_residual,
            support_metadata_log_residual=support_m_residual,
        )

    def _validate_embeddings(self, values: torch.Tensor, *, name: str) -> None:
        if values.ndim != 2 or values.shape[1] != self.embedding_dim:
            raise ValueError(
                f"{name} must have shape [sample,{self.embedding_dim}], got {tuple(values.shape)}."
            )
        if not torch.isfinite(values).all():
            raise ValueError(f"{name} must be finite.")
