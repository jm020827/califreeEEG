from __future__ import annotations

import torch
from torch import nn

from cfeg.models.backbones.base import BackboneOutput, EEGBackbone


class SpectralEEGTransformerBackbone(EEGBackbone):
    """Compact channel transformer over complex FFT features.

    It uses only the current query waveform. No target participant batch,
    covariance, template, history, or label is consumed, so it remains strict
    inductive k=0 while adding the frequency-domain bias missing from the
    original long-token time transformer.
    """

    supports_prompt_tokens = True
    supports_channel_gain = True

    def __init__(
        self,
        *,
        c_max: int,
        t_len: int,
        target_sfreq: float,
        d_model: int,
        depth: int,
        n_heads: int,
        min_frequency_hz: float = 6.0,
        max_frequency_hz: float = 60.0,
        dropout: float = 0.1,
        channel_vocab_size: int = 65,
    ):
        super().__init__()
        if t_len < 4 or target_sfreq <= 0:
            raise ValueError("t_len and target_sfreq must be positive for spectral features.")
        frequencies = torch.fft.rfftfreq(t_len, d=1.0 / float(target_sfreq))
        keep = (frequencies >= float(min_frequency_hz)) & (
            frequencies <= float(max_frequency_hz)
        )
        indices = torch.nonzero(keep, as_tuple=False).flatten()
        if not len(indices):
            raise ValueError("The configured spectral frequency range contains no FFT bins.")
        self.c_max = int(c_max)
        self.t_len = int(t_len)
        self.d_model = int(d_model)
        self.register_buffer("frequency_bin_indices", indices, persistent=True)
        self.register_buffer("frequency_hz", frequencies[indices], persistent=True)
        n_features = len(indices) * 3
        self.spectral_projection = nn.Sequential(
            nn.LayerNorm(n_features),
            nn.Linear(n_features, d_model),
            nn.GELU(),
        )
        self.channel_embedding = nn.Embedding(channel_vocab_size, d_model, padding_idx=0)
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

    def forward(
        self,
        x: torch.Tensor,
        cond: dict[str, torch.Tensor],
        prompt_tokens: torch.Tensor | None = None,
        return_tokens: bool = False,
        channel_gain: torch.Tensor | None = None,
    ) -> BackboneOutput:
        batch, channels, time = x.shape
        if channels != self.c_max or time != self.t_len:
            raise ValueError(
                f"Expected [B,{self.c_max},{self.t_len}], got {tuple(x.shape)}."
            )
        spectrum = torch.fft.rfft(x.float(), dim=-1, norm="ortho")
        selected = spectrum.index_select(-1, self.frequency_bin_indices)
        # Complex leakage patterns carry sub-bin frequency and phase information.
        features = torch.cat(
            [selected.real, selected.imag, torch.log1p(selected.abs())], dim=-1
        ).to(dtype=x.dtype)
        tokens = self.spectral_projection(features)
        if channel_gain is not None:
            if channel_gain.shape != (batch, channels):
                raise ValueError(
                    "channel_gain must have shape [batch,channels], got "
                    f"{tuple(channel_gain.shape)}."
                )
            if not torch.isfinite(channel_gain).all() or torch.any(channel_gain <= 0):
                raise ValueError("channel_gain must contain finite positive values.")
            # Modulate signal evidence only. Canonical channel identity remains a
            # shared structural input and is never down-weighted as A2 treatment.
            tokens = tokens * channel_gain.to(tokens.dtype).unsqueeze(-1)
        channel_ids = cond["channel_ids"].clamp(
            min=0, max=self.channel_embedding.num_embeddings - 1
        )
        tokens = tokens + self.channel_embedding(channel_ids)
        cls = self.cls.expand(batch, -1, -1)
        prompt_len = 0
        if prompt_tokens is not None:
            prompt_len = int(prompt_tokens.shape[1])
            sequence = torch.cat([cls, prompt_tokens, tokens], dim=1)
        else:
            sequence = torch.cat([cls, tokens], dim=1)
        channel_padding = ~cond["channel_mask"].bool()
        prefix_padding = torch.zeros(
            (batch, 1 + prompt_len), dtype=torch.bool, device=x.device
        )
        padding = torch.cat([prefix_padding, channel_padding], dim=1)
        encoded = self.norm(self.encoder(sequence, src_key_padding_mask=padding))
        return BackboneOutput(
            h=encoded[:, 0],
            tokens=encoded if return_tokens else None,
            aux={
                "spectral_min_hz": self.frequency_hz[0],
                "spectral_max_hz": self.frequency_hz[-1],
            },
        )
