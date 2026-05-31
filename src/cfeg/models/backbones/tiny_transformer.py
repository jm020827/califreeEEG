from __future__ import annotations

import torch
from torch import nn

from cfeg.models.backbones.base import BackboneOutput, EEGBackbone


class TinyEEGTransformerBackbone(EEGBackbone):
    supports_prompt_tokens = True

    def __init__(
        self,
        c_max: int = 64,
        t_len: int = 400,
        d_model: int = 128,
        patch_size: int = 20,
        depth: int = 4,
        n_heads: int = 4,
        dropout: float = 0.1,
        channel_vocab_size: int = 65,
    ):
        super().__init__()
        if t_len % patch_size != 0:
            raise ValueError(f"t_len={t_len} must be divisible by patch_size={patch_size}")
        self.c_max = c_max
        self.t_len = t_len
        self.d_model = d_model
        self.patch_size = patch_size
        self.n_patches = t_len // patch_size
        self.patch_embed = nn.Linear(patch_size, d_model)
        self.channel_embedding = nn.Embedding(channel_vocab_size, d_model, padding_idx=0)
        self.time_embedding = nn.Parameter(torch.zeros(1, self.n_patches, d_model))
        self.cls = nn.Parameter(torch.zeros(1, 1, d_model))
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=depth)
        self.norm = nn.LayerNorm(d_model)
        nn.init.trunc_normal_(self.cls, std=0.02)
        nn.init.trunc_normal_(self.time_embedding, std=0.02)

    def forward(
        self,
        x: torch.Tensor,
        cond: dict[str, torch.Tensor],
        prompt_tokens: torch.Tensor | None = None,
        return_tokens: bool = False,
    ) -> BackboneOutput:
        bsz, channels, time = x.shape
        if channels != self.c_max:
            raise ValueError(f"Expected {self.c_max} channels, got {channels}")
        if time != self.t_len:
            raise ValueError(f"Expected {self.t_len} time samples, got {time}")
        patches = x.unfold(dimension=-1, size=self.patch_size, step=self.patch_size)
        tok = self.patch_embed(patches)
        channel_ids = cond["channel_ids"].clamp(min=0, max=self.channel_embedding.num_embeddings - 1)
        tok = tok + self.channel_embedding(channel_ids).unsqueeze(2)
        tok = tok + self.time_embedding[:, : self.n_patches].unsqueeze(1)
        tok = tok.reshape(bsz, channels * self.n_patches, self.d_model)
        cls = self.cls.expand(bsz, -1, -1)
        prompt_len = 0
        if prompt_tokens is not None:
            prompt_len = prompt_tokens.shape[1]
            seq = torch.cat([cls, prompt_tokens, tok], dim=1)
        else:
            seq = torch.cat([cls, tok], dim=1)
        pad = ~cond["channel_mask"].bool()
        tok_pad = pad.unsqueeze(-1).expand(-1, -1, self.n_patches).reshape(bsz, -1)
        prefix_pad = torch.zeros((bsz, 1 + prompt_len), dtype=torch.bool, device=x.device)
        key_padding_mask = torch.cat([prefix_pad, tok_pad], dim=1)
        out = self.encoder(seq, src_key_padding_mask=key_padding_mask)
        out = self.norm(out)
        h = out[:, 0]
        return BackboneOutput(h=h, tokens=out if return_tokens else None, aux={})

