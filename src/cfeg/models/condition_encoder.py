from __future__ import annotations

import torch
from torch import nn

from cfeg.constants import (
    CONDITION_CATEGORICAL_FIELDS,
    METADATA_CONTRACT_LEGACY,
    METADATA_CONTRACT_V04_DEV,
    PROTOCOL_V04_EXTERNAL_CATEGORICAL_FIELDS,
)


class ConditionEncoder(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_prompt_tokens: int,
        vocab_sizes: dict[str, int],
        n_cont_features: int,
        channel_vocab_size: int,
        dropout: float = 0.1,
        fields: list[str] | None = None,
        include_continuous: bool = True,
        include_channels: bool = True,
        force_missing: bool = False,
        metadata_contract_version: str = METADATA_CONTRACT_LEGACY,
        external_metadata_mode: str = "observed",
        n_external_cont_features: int = 2,
        n_query_qc_features: int = 1,
    ):
        super().__init__()
        self.d_model = d_model
        self.n_prompt_tokens = n_prompt_tokens
        self.include_continuous = include_continuous
        self.include_channels = include_channels
        self.force_missing = force_missing
        self.metadata_contract_version = metadata_contract_version
        self.external_metadata_mode = external_metadata_mode
        if metadata_contract_version not in {
            METADATA_CONTRACT_LEGACY,
            METADATA_CONTRACT_V04_DEV,
        }:
            raise ValueError(f"Unknown metadata_contract_version={metadata_contract_version!r}.")
        if external_metadata_mode not in {"observed", "null"}:
            raise ValueError(
                f"external_metadata_mode must be 'observed' or 'null', got "
                f"{external_metadata_mode!r}."
            )
        if metadata_contract_version == METADATA_CONTRACT_V04_DEV and include_channels:
            raise ValueError(
                "Protocol 0.4-dev reserves channel structure for the common backbone path; "
                "set condition_encoder.include_channels=false."
            )
        selected_fields = CONDITION_CATEGORICAL_FIELDS if fields is None else fields
        if metadata_contract_version == METADATA_CONTRACT_V04_DEV:
            forbidden_fields = sorted(
                set(selected_fields) - set(PROTOCOL_V04_EXTERNAL_CATEGORICAL_FIELDS)
            )
            if forbidden_fields:
                raise ValueError(
                    "Protocol 0.4-dev condition encoder contains forbidden/confounded "
                    f"categorical fields: {forbidden_fields}. Allowed fields are "
                    f"{PROTOCOL_V04_EXTERNAL_CATEGORICAL_FIELDS}."
                )
        self.cat_names = [
            field for field in selected_fields if field in CONDITION_CATEGORICAL_FIELDS
        ]
        self.cat_embeddings = nn.ModuleDict(
            {name: nn.Embedding(vocab_sizes.get(name, 1), d_model) for name in self.cat_names}
        )
        cont_features = (
            n_external_cont_features
            if metadata_contract_version == METADATA_CONTRACT_V04_DEV
            else n_cont_features
        )
        self.cont_mlp = nn.Sequential(
            nn.Linear(cont_features * 2, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model),
        )
        if metadata_contract_version == METADATA_CONTRACT_V04_DEV:
            self.query_qc_mlp = nn.Sequential(
                nn.Linear(n_query_qc_features * 2, d_model),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(d_model, d_model),
            )
            self.channel_embed = None
        else:
            self.query_qc_mlp = None
            self.channel_embed = nn.Embedding(channel_vocab_size, d_model, padding_idx=0)
        self.modality_embedding = nn.Parameter(torch.zeros(1, 3, d_model))
        n_heads = _compatible_heads(d_model, 4)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 2,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.context_encoder = nn.TransformerEncoder(encoder_layer, num_layers=1)
        self.fuse = nn.Sequential(
            nn.LayerNorm(d_model * 3),
            nn.Linear(d_model * 3, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model),
        )
        self.to_prompt = (
            nn.Linear(d_model, n_prompt_tokens * d_model) if n_prompt_tokens > 0 else None
        )

    def forward(self, cond: dict[str, torch.Tensor]) -> tuple[torch.Tensor | None, torch.Tensor]:
        batch = cond["continuous"].shape[0]
        device = cond["continuous"].device
        cat_vec = torch.zeros((batch, self.d_model), device=device)
        external_is_null = self.force_missing or self.external_metadata_mode == "null"
        for name in self.cat_names:
            ids = torch.zeros_like(cond[name]) if external_is_null else cond[name]
            ids = ids.clamp(min=0, max=self.cat_embeddings[name].num_embeddings - 1)
            cat_vec = cat_vec + self.cat_embeddings[name](ids)
        if self.include_continuous:
            if self.metadata_contract_version == METADATA_CONTRACT_V04_DEV:
                continuous = cond["external_continuous"]
                continuous_missing = cond["external_continuous_missing"]
            else:
                continuous = cond["continuous"]
                continuous_missing = cond["continuous_missing"]
            if external_is_null:
                continuous = torch.zeros_like(continuous)
                continuous_missing = torch.ones_like(continuous_missing)
            cont_in = torch.cat([continuous, continuous_missing.float()], dim=-1)
            cont_vec = self.cont_mlp(cont_in)
        else:
            cont_vec = torch.zeros_like(cat_vec)
        if self.metadata_contract_version == METADATA_CONTRACT_V04_DEV:
            query_qc_in = torch.cat([cond["query_qc"], cond["query_qc_missing"].float()], dim=-1)
            ch_vec = self.query_qc_mlp(query_qc_in)
        elif self.include_channels:
            condition_channel_ids = cond.get("condition_channel_ids", cond["channel_ids"])
            condition_channel_mask = cond.get("condition_channel_mask", cond["channel_mask"])
            if self.force_missing:
                condition_channel_ids = torch.zeros_like(condition_channel_ids)
                condition_channel_mask = torch.zeros_like(condition_channel_mask)
            ch_emb = self.channel_embed(
                condition_channel_ids.clamp(min=0, max=self.channel_embed.num_embeddings - 1)
            )
            mask = condition_channel_mask.float().unsqueeze(-1)
            ch_vec = (ch_emb * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)
        else:
            ch_vec = torch.zeros_like(cat_vec)
        context_tokens = torch.stack([cat_vec, cont_vec, ch_vec], dim=1)
        context_tokens = self.context_encoder(context_tokens + self.modality_embedding)
        cond_vec = self.fuse(context_tokens.reshape(batch, self.d_model * 3))
        if self.to_prompt is None:
            return None, cond_vec
        prompt = self.to_prompt(cond_vec).view(batch, self.n_prompt_tokens, self.d_model)
        return prompt, cond_vec


def _compatible_heads(d_model: int, requested: int) -> int:
    for n_heads in range(min(requested, d_model), 0, -1):
        if d_model % n_heads == 0:
            return n_heads
    return 1
