from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from cfeg.constants import CATEGORICAL_VOCABS
from cfeg.models.adapters import BottleneckAdapter, ConditionedAdapter
from cfeg.models.backbones.reve import REVEBackbone
from cfeg.models.backbones.tiny_transformer import TinyEEGTransformerBackbone
from cfeg.models.condition_encoder import ConditionEncoder
from cfeg.models.heads import ClassificationHead
from cfeg.models.latent_nuisance import LatentNuisanceEncoder


@dataclass
class ModelOutput:
    logits: torch.Tensor
    logits_zero: torch.Tensor | None
    h: torch.Tensor
    prompt_tokens: torch.Tensor | None
    cond_vec: torch.Tensor | None
    z: torch.Tensor | None
    mu: torch.Tensor | None
    logvar: torch.Tensor | None
    aux: dict[str, torch.Tensor]


class ConditionedEEGDecoder(nn.Module):
    def __init__(self, cfg: dict, vocab_sizes: dict[str, int] | None = None):
        super().__init__()
        self.cfg = cfg
        model_cfg = cfg.get("model", cfg)
        self.model_cfg = model_cfg
        d_model = int(model_cfg.get("d_model", 128))
        backbone_cfg = model_cfg.get("backbone", {"name": "tiny_transformer"})
        name = backbone_cfg.get("name", "tiny_transformer")
        if name == "tiny_transformer":
            self.backbone = TinyEEGTransformerBackbone(
                c_max=int(model_cfg.get("c_max", 64)),
                t_len=int(model_cfg.get("t_len", 400)),
                d_model=d_model,
                patch_size=int(model_cfg.get("patch_size", 20)),
                depth=int(model_cfg.get("depth", 4)),
                n_heads=int(model_cfg.get("n_heads", 4)),
            )
        elif name == "reve":
            reve_cfg = dict(backbone_cfg)
            reve_cfg.setdefault("d_model", d_model)
            self.backbone = REVEBackbone(reve_cfg)
            d_model = self.backbone.d_model
        else:
            raise ValueError(f"Unknown backbone: {name}")

        ce_cfg = model_cfg.get("condition_encoder", {})
        self.condition_enabled = bool(ce_cfg.get("enabled", True))
        vocab_sizes = vocab_sizes or {name: len(vals) for name, vals in CATEGORICAL_VOCABS.items()}
        if self.condition_enabled:
            self.condition_encoder = ConditionEncoder(
                d_model=d_model,
                n_prompt_tokens=int(ce_cfg.get("n_prompt_tokens", 4)),
                vocab_sizes=vocab_sizes,
                n_cont_features=5,
                channel_vocab_size=int(model_cfg.get("c_max", 64)) + 1,
                fields=ce_cfg.get("fields"),
            )
        else:
            self.condition_encoder = None

        adapter_cfg = model_cfg.get("adapter", {})
        if adapter_cfg.get("enabled", True):
            if adapter_cfg.get("type") == "conditioned_feature_adapter":
                self.adapter = ConditionedAdapter(
                    d_model=d_model,
                    bottleneck_dim=int(adapter_cfg.get("bottleneck_dim", 32)),
                )
            else:
                self.adapter = BottleneckAdapter(
                    d_model=d_model,
                    bottleneck_dim=int(adapter_cfg.get("bottleneck_dim", 32)),
                )
        else:
            self.adapter = None

        latent_cfg = model_cfg.get("latent", {})
        self.latent_enabled = bool(latent_cfg.get("enabled", True))
        self.z_dim = int(latent_cfg.get("z_dim", 16)) if self.latent_enabled else 0
        self.z_dropout = float(latent_cfg.get("z_dropout", 0.0))
        if self.latent_enabled:
            self.latent = LatentNuisanceEncoder(d_model, d_model, self.z_dim)
        else:
            self.latent = None
        self.head = ClassificationHead(d_model, int(model_cfg.get("n_classes", 4)), self.z_dim)

    def forward(
        self,
        x: torch.Tensor,
        cond: dict[str, torch.Tensor],
        use_latent: bool | None = None,
        return_repr: bool = True,
    ) -> ModelOutput:
        prompt, cond_vec = (None, None)
        if self.condition_encoder is not None:
            prompt, cond_vec = self.condition_encoder(cond)
        if prompt is not None and not self.backbone.supports_prompt_tokens:
            prompt = None
        backbone_out = self.backbone(x, cond=cond, prompt_tokens=prompt, return_tokens=return_repr)
        h = backbone_out.h
        if self.adapter is not None:
            if isinstance(self.adapter, ConditionedAdapter):
                h = self.adapter(h, cond_vec)
            else:
                h = self.adapter(h)
        z = mu = logvar = None
        logits_zero = None
        use_latent = self.latent_enabled if use_latent is None else use_latent
        if self.latent_enabled:
            zero_z = torch.zeros((h.shape[0], self.z_dim), device=h.device, dtype=h.dtype)
            if self.training and use_latent and cond_vec is not None:
                mu, logvar = self.latent(h, cond_vec)
                z = self.latent.sample(mu, logvar)
                if self.z_dropout > 0:
                    keep = torch.rand((h.shape[0], 1), device=h.device) > self.z_dropout
                    z = torch.where(keep, z, zero_z)
            else:
                z = zero_z
            logits = self.head(h, z)
            logits_zero = self.head(h, zero_z)
        else:
            logits = self.head(h, None)
        return ModelOutput(
            logits=logits,
            logits_zero=logits_zero,
            h=h,
            prompt_tokens=prompt,
            cond_vec=cond_vec,
            z=z,
            mu=mu,
            logvar=logvar,
            aux=backbone_out.aux,
        )

