from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from cfeg.constants import PROTOCOL_V04_EXTERNAL_CATEGORICAL_FIELDS

PROMPT_ADAPTER_V1 = "prompt_adapter_v1"
PHYSICAL_HYBRID_V1 = "physical_hybrid_v1"


@dataclass
class PhysicalConditionState:
    query_vec: torch.Tensor
    query_available: torch.Tensor
    external_vec: torch.Tensor
    external_available: torch.Tensor
    channel_gain: torch.Tensor
    channel_available: torch.Tensor
    metadata_only_vec: torch.Tensor


class ZeroInitResidualFiLM(nn.Module):
    """Bounded FiLM residual that is exactly identity at initialization or when absent."""

    def __init__(
        self,
        d_model: int,
        *,
        max_scale_delta: float = 0.25,
        max_shift: float = 0.25,
    ):
        super().__init__()
        if max_scale_delta < 0.0 or max_shift < 0.0:
            raise ValueError("FiLM bounds must be non-negative.")
        self.max_scale_delta = float(max_scale_delta)
        self.max_shift = float(max_shift)
        self.norm = nn.LayerNorm(d_model, elementwise_affine=False)
        self.to_affine = nn.Linear(d_model, d_model * 2)
        nn.init.zeros_(self.to_affine.weight)
        nn.init.zeros_(self.to_affine.bias)

    def forward(
        self,
        h: torch.Tensor,
        context: torch.Tensor,
        available: torch.Tensor,
    ) -> torch.Tensor:
        if h.shape != context.shape:
            raise ValueError(
                f"FiLM h/context shapes must match, got {tuple(h.shape)} and "
                f"{tuple(context.shape)}."
            )
        if available.shape != (h.shape[0], 1):
            raise ValueError(
                "FiLM availability must have shape [batch,1], got "
                f"{tuple(available.shape)}."
            )
        raw_scale, raw_shift = self.to_affine(context).chunk(2, dim=-1)
        active = available.to(dtype=h.dtype)
        scale = self.max_scale_delta * torch.tanh(raw_scale) * active
        shift = self.max_shift * torch.tanh(raw_shift) * active
        return h + scale * self.norm(h) + shift


class FactorizedPhysicalConditioner(nn.Module):
    """Factorized query, global-acquisition, and channel-quality conditioning.

    External-null and missing values are masked again inside the model. This is
    deliberate defense in depth: stale values in a collated batch cannot make
    the A0 arm non-neutral.
    """

    def __init__(
        self,
        *,
        d_model: int,
        vocab_sizes: dict[str, int],
        fields: list[str],
        external_metadata_mode: str,
        hidden_dim: int = 64,
        n_external_cont_features: int = 2,
        n_query_qc_features: int = 1,
        max_channel_gain_delta: float = 0.25,
        query_enabled: bool = True,
        external_global_enabled: bool = True,
        channel_quality_enabled: bool = True,
    ):
        super().__init__()
        if external_metadata_mode not in {"observed", "null"}:
            raise ValueError(
                "external_metadata_mode must be 'observed' or 'null', got "
                f"{external_metadata_mode!r}."
            )
        forbidden = sorted(set(fields) - set(PROTOCOL_V04_EXTERNAL_CATEGORICAL_FIELDS))
        if forbidden:
            raise ValueError(
                "physical_hybrid_v1 contains forbidden categorical fields: "
                f"{forbidden}."
            )
        if hidden_dim <= 0:
            raise ValueError("physical_hybrid_v1 hidden_dim must be positive.")
        if not 0.0 <= max_channel_gain_delta < 1.0:
            raise ValueError("max_channel_gain_delta must be in [0,1).")

        self.d_model = int(d_model)
        self.fields = list(fields)
        self.external_metadata_mode = external_metadata_mode
        self.max_channel_gain_delta = float(max_channel_gain_delta)
        self.query_enabled = bool(query_enabled)
        self.external_global_enabled = bool(external_global_enabled)
        self.channel_quality_enabled = bool(channel_quality_enabled)
        self.n_external_cont_features = int(n_external_cont_features)
        self.n_query_qc_features = int(n_query_qc_features)

        self.cat_embeddings = nn.ModuleDict(
            {
                name: nn.Embedding(
                    max(int(vocab_sizes.get(name, 1)), 1),
                    hidden_dim,
                    padding_idx=0,
                )
                for name in self.fields
            }
        )
        self.query_encoder = nn.Sequential(
            nn.Linear(self.n_query_qc_features * 2, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, d_model),
            nn.LayerNorm(d_model),
        )
        self.external_encoder = nn.Sequential(
            nn.Linear(hidden_dim + self.n_external_cont_features * 2, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, d_model),
            nn.LayerNorm(d_model),
        )
        self.channel_encoder = nn.Sequential(
            nn.Linear(2, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
        )
        self.channel_context = nn.Linear(hidden_dim, d_model)
        self.channel_score = nn.Linear(hidden_dim, 1)
        nn.init.zeros_(self.channel_score.weight)
        nn.init.zeros_(self.channel_score.bias)

    def forward(self, cond: dict[str, torch.Tensor]) -> PhysicalConditionState:
        external_is_null = self.external_metadata_mode == "null"
        query_values = cond["query_qc"]
        query_missing = cond["query_qc_missing"].bool()
        external_values = cond["external_continuous"]
        external_missing = cond["external_continuous_missing"].bool()
        channel_values = cond["channel_impedance"]
        channel_missing = cond["channel_impedance_missing"].bool()
        channel_mask = cond["channel_mask"].bool()

        self._validate_shapes(
            query_values,
            query_missing,
            external_values,
            external_missing,
            channel_values,
            channel_missing,
            channel_mask,
        )

        query_observed = ~query_missing
        query_input = torch.cat(
            [query_values.masked_fill(query_missing, 0.0), query_observed.float()], dim=-1
        )
        query_available = query_observed.any(dim=-1, keepdim=True) & self.query_enabled
        query_vec = self.query_encoder(query_input) * query_available.to(query_values.dtype)

        if external_is_null:
            external_values = torch.zeros_like(external_values)
            external_missing = torch.ones_like(external_missing)
            channel_values = torch.zeros_like(channel_values)
            channel_missing = torch.ones_like(channel_missing)

        batch = external_values.shape[0]
        cat_vec = external_values.new_zeros((batch, self.channel_context.in_features))
        categorical_available = torch.zeros((batch, 1), dtype=torch.bool, device=cat_vec.device)
        for name, embedding in self.cat_embeddings.items():
            ids = torch.zeros_like(cond[name]) if external_is_null else cond[name]
            valid = (ids >= 0) & (ids < embedding.num_embeddings)
            ids = torch.where(valid, ids, torch.zeros_like(ids))
            cat_vec = cat_vec + embedding(ids)
            categorical_available |= ids.ne(0).unsqueeze(-1)

        external_observed = ~external_missing
        external_input = torch.cat(
            [
                cat_vec,
                external_values.masked_fill(external_missing, 0.0),
                external_observed.float(),
            ],
            dim=-1,
        )
        external_available = (
            categorical_available | external_observed.any(dim=-1, keepdim=True)
        ) & self.external_global_enabled
        external_vec = self.external_encoder(external_input) * external_available.to(
            external_values.dtype
        )

        quality_available = (~channel_missing) & self.channel_quality_enabled
        channel_available = quality_available & channel_mask
        channel_input = torch.stack(
            [
                channel_values.masked_fill(~quality_available, 0.0),
                quality_available.float(),
            ],
            dim=-1,
        )
        channel_hidden = self.channel_encoder(channel_input)
        raw_channel_score = self.channel_score(channel_hidden).squeeze(-1)
        channel_delta = (
            self.max_channel_gain_delta
            * torch.tanh(raw_channel_score)
            * channel_available.to(raw_channel_score.dtype)
        )
        channel_gain = 1.0 + channel_delta

        # The shortcut control may remove EEG structure/channel_mask, but it must
        # still see the complete external quality vector available to A2.
        channel_weight = quality_available.to(channel_hidden.dtype).unsqueeze(-1)
        pooled_channel = (channel_hidden * channel_weight).sum(dim=1) / channel_weight.sum(
            dim=1
        ).clamp_min(1.0)
        pooled_channel = self.channel_context(pooled_channel)
        any_channel = quality_available.any(dim=-1, keepdim=True)
        pooled_channel = pooled_channel * any_channel.to(pooled_channel.dtype)
        metadata_only_vec = external_vec + pooled_channel

        return PhysicalConditionState(
            query_vec=query_vec,
            query_available=query_available,
            external_vec=external_vec,
            external_available=external_available,
            channel_gain=channel_gain,
            channel_available=channel_available,
            metadata_only_vec=metadata_only_vec,
        )

    def _validate_shapes(
        self,
        query_values: torch.Tensor,
        query_missing: torch.Tensor,
        external_values: torch.Tensor,
        external_missing: torch.Tensor,
        channel_values: torch.Tensor,
        channel_missing: torch.Tensor,
        channel_mask: torch.Tensor,
    ) -> None:
        if query_values.ndim != 2 or query_values.shape[1] != self.n_query_qc_features:
            raise ValueError(
                "query_qc must have shape [batch,"
                f"{self.n_query_qc_features}], got {tuple(query_values.shape)}."
            )
        if query_missing.shape != query_values.shape:
            raise ValueError("query_qc and query_qc_missing shapes must match.")
        if (
            external_values.ndim != 2
            or external_values.shape[1] != self.n_external_cont_features
        ):
            raise ValueError(
                "external_continuous must have shape [batch,"
                f"{self.n_external_cont_features}], got {tuple(external_values.shape)}."
            )
        if external_missing.shape != external_values.shape:
            raise ValueError(
                "external_continuous and external_continuous_missing shapes must match."
            )
        if channel_values.ndim != 2:
            raise ValueError("channel_impedance must have shape [batch,channels].")
        if channel_missing.shape != channel_values.shape or channel_mask.shape != channel_values.shape:
            raise ValueError(
                "channel_impedance, channel_impedance_missing, and channel_mask shapes "
                "must match."
            )
        batch = query_values.shape[0]
        if external_values.shape[0] != batch or channel_values.shape[0] != batch:
            raise ValueError("Physical conditioning tensors must share one batch dimension.")
