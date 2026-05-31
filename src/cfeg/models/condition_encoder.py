from __future__ import annotations

import torch
from torch import nn

from cfeg.constants import CONDITION_CATEGORICAL_FIELDS


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
    ):
        super().__init__()
        self.d_model = d_model
        self.n_prompt_tokens = n_prompt_tokens
        self.cat_names = [f for f in (fields or CONDITION_CATEGORICAL_FIELDS) if f in CONDITION_CATEGORICAL_FIELDS]
        self.cat_embeddings = nn.ModuleDict(
            {name: nn.Embedding(vocab_sizes.get(name, 1), d_model) for name in self.cat_names}
        )
        self.cont_mlp = nn.Sequential(
            nn.Linear(n_cont_features * 2, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model),
        )
        self.channel_embed = nn.Embedding(channel_vocab_size, d_model, padding_idx=0)
        self.fuse = nn.Sequential(
            nn.LayerNorm(d_model * 3),
            nn.Linear(d_model * 3, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model),
        )
        self.to_prompt = nn.Linear(d_model, n_prompt_tokens * d_model) if n_prompt_tokens > 0 else None

    def forward(self, cond: dict[str, torch.Tensor]) -> tuple[torch.Tensor | None, torch.Tensor]:
        batch = cond["continuous"].shape[0]
        device = cond["continuous"].device
        cat_vec = torch.zeros((batch, self.d_model), device=device)
        for name in self.cat_names:
            ids = cond[name].clamp(min=0, max=self.cat_embeddings[name].num_embeddings - 1)
            cat_vec = cat_vec + self.cat_embeddings[name](ids)
        cont_in = torch.cat([cond["continuous"], cond["continuous_missing"].float()], dim=-1)
        cont_vec = self.cont_mlp(cont_in)
        ch_emb = self.channel_embed(
            cond["channel_ids"].clamp(min=0, max=self.channel_embed.num_embeddings - 1)
        )
        mask = cond["channel_mask"].float().unsqueeze(-1)
        ch_vec = (ch_emb * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)
        cond_vec = self.fuse(torch.cat([cat_vec, cont_vec, ch_vec], dim=-1))
        if self.to_prompt is None:
            return None, cond_vec
        prompt = self.to_prompt(cond_vec).view(batch, self.n_prompt_tokens, self.d_model)
        return prompt, cond_vec

