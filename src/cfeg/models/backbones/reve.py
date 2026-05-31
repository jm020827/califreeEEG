from __future__ import annotations

import torch

from cfeg.assets.errors import MissingAssetError
from cfeg.assets.hf import hf_cache_hint
from cfeg.models.backbones.base import BackboneOutput, EEGBackbone


class REVEBackbone(EEGBackbone):
    supports_prompt_tokens = False

    def __init__(self, cfg: dict):
        super().__init__()
        try:
            from transformers import AutoModel
        except Exception as exc:
            raise MissingAssetError(
                "transformers is required for REVE. Install requirements or use "
                "model.backbone.name=tiny_transformer."
            ) from exc
        self.cfg = cfg
        self.required_sample_rate_hz = float(cfg.get("required_sample_rate_hz", 200.0))
        self.pos_bank = self._load_model(AutoModel, cfg["hf_positions"], cfg)
        self.reve = self._load_model(AutoModel, cfg["hf_model"], cfg)
        self.d_model = int(cfg.get("d_model", getattr(getattr(self.reve, "config", None), "hidden_size", 128)))
        if cfg.get("freeze", True):
            for p in self.reve.parameters():
                p.requires_grad = False

    @staticmethod
    def _load_model(auto_model, repo_id: str, cfg: dict):
        try:
            return auto_model.from_pretrained(
                repo_id,
                cache_dir=cfg.get("cache_dir"),
                trust_remote_code=cfg.get("trust_remote_code", True),
                local_files_only=cfg.get("local_files_only", True),
            )
        except Exception as exc:
            raise MissingAssetError(hf_cache_hint(repo_id, cfg.get("cache_dir"))) from exc

    def forward(
        self,
        x: torch.Tensor,
        cond: dict[str, torch.Tensor],
        prompt_tokens: torch.Tensor | None = None,
        return_tokens: bool = False,
    ) -> BackboneOutput:
        sfreq = cond.get("sfreq_processed_float")
        if sfreq is not None and not torch.allclose(
            sfreq.float(), torch.full_like(sfreq.float(), self.required_sample_rate_hz), atol=1e-3
        ):
            raise ValueError("REVEBackbone requires processed sample rate of 200 Hz.")
        if prompt_tokens is not None:
            # REVE remote code does not expose a stable prompt-token prepend contract.
            prompt_tokens = None
        try:
            out = self.reve(x)
        except TypeError:
            out = self.reve(inputs=x)
        h = extract_reve_representation(out)
        return BackboneOutput(h=h, tokens=None, aux={})


def extract_reve_representation(out):
    if hasattr(out, "pooler_output") and out.pooler_output is not None:
        return out.pooler_output
    if hasattr(out, "last_hidden_state"):
        return out.last_hidden_state.mean(dim=1)
    if isinstance(out, torch.Tensor):
        return out
    if isinstance(out, dict):
        for key in ["pooler_output", "last_hidden_state", "embeddings", "h"]:
            if key in out:
                value = out[key]
                return value.mean(dim=1) if value.ndim == 3 else value
    raise RuntimeError(f"Cannot extract representation from REVE output type: {type(out)}")

