from __future__ import annotations

import torch
from torch import nn

from cfeg.assets.errors import MissingAssetError
from cfeg.assets.hf import hf_cache_hint, resolve_hf_hub_cache
from cfeg.data.preprocess import CanonicalChannelMap
from cfeg.models.backbones.base import BackboneOutput, EEGBackbone


class REVEBackbone(EEGBackbone):
    supports_prompt_tokens = True

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
        self.cache_dir = str(resolve_hf_hub_cache(cfg.get("cache_dir")))
        self.required_sample_rate_hz = float(cfg.get("required_sample_rate_hz", 200.0))
        self.pos_bank = self._load_model(AutoModel, cfg["hf_positions"], cfg)
        self.reve = self._load_model(AutoModel, cfg["hf_model"], cfg)
        self.d_model = _infer_reve_dim(self.reve, cfg)
        self.canonical_map = CanonicalChannelMap.from_yaml()
        self.output_proj: nn.Module | None = None
        n_heads = _compatible_heads(self.d_model, int(cfg.get("prompt_fusion_heads", 8)))
        self.prompt_attention = nn.MultiheadAttention(
            self.d_model, n_heads, dropout=float(cfg.get("prompt_fusion_dropout", 0.1)), batch_first=True
        )
        self.prompt_gate = nn.Sequential(nn.Linear(self.d_model, self.d_model), nn.Sigmoid())
        self.prompt_norm = nn.LayerNorm(self.d_model)
        self._position_cache: dict[tuple[str, ...], tuple[list[int], torch.Tensor]] = {}
        self.freeze = bool(cfg.get("freeze", True))
        for parameter in self.pos_bank.parameters():
            parameter.requires_grad = False
        if self.freeze:
            for parameter in self.reve.parameters():
                parameter.requires_grad = False
            self.reve.eval()
        self.pos_bank.eval()

    @staticmethod
    def _load_model(auto_model, repo_id: str, cfg: dict):
        cache_dir = str(resolve_hf_hub_cache(cfg.get("cache_dir")))
        try:
            return auto_model.from_pretrained(
                repo_id,
                cache_dir=cache_dir,
                trust_remote_code=cfg.get("trust_remote_code", True),
                local_files_only=cfg.get("local_files_only", True),
            )
        except Exception as exc:
            raise MissingAssetError(hf_cache_hint(repo_id, cache_dir)) from exc

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
        x_reve, positions = self._select_channels_and_positions(x, cond)
        if self.freeze:
            with torch.no_grad():
                out = self._forward_reve(x_reve, positions)
        else:
            out = self._forward_reve(x_reve, positions)
        tokens = extract_reve_tokens(out)
        if tokens.shape[-1] != self.d_model:
            if self.output_proj is None:
                self.output_proj = nn.Linear(tokens.shape[-1], self.d_model).to(
                    device=tokens.device, dtype=tokens.dtype
                )
            tokens = self.output_proj(tokens)
        h = tokens.mean(dim=1)
        aux: dict[str, torch.Tensor] = {}
        if prompt_tokens is not None:
            attended, weights = self.prompt_attention(
                self.prompt_norm(prompt_tokens), self.prompt_norm(tokens), self.prompt_norm(tokens)
            )
            prompt_summary = attended.mean(dim=1)
            h = self.prompt_norm(h + self.prompt_gate(prompt_tokens.mean(dim=1)) * prompt_summary)
            aux["prompt_attention_mean"] = weights.mean(dim=(1, 2))
        returned_tokens = torch.cat([prompt_tokens, tokens], dim=1) if (
            return_tokens and prompt_tokens is not None
        ) else (tokens if return_tokens else None)
        return BackboneOutput(h=h, tokens=returned_tokens, aux=aux)

    def train(self, mode: bool = True):
        super().train(mode)
        if getattr(self, "freeze", False):
            self.reve.eval()
            self.pos_bank.eval()
        return self

    def _forward_reve(self, x_reve: torch.Tensor, positions: torch.Tensor):
        try:
            return self.reve(x_reve, positions)
        except TypeError:
            return self.reve(x_reve, pos=positions)

    def _select_channels_and_positions(
        self, x: torch.Tensor, cond: dict[str, torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        mask = cond["channel_mask"].bool()
        active_slots = torch.nonzero(mask.any(dim=0), as_tuple=False).flatten()
        if active_slots.numel() == 0:
            raise ValueError("REVEBackbone received a batch with no active EEG channels.")

        channel_ids = cond["channel_ids"].detach().cpu()
        names: list[str] = []
        keep_slots: list[int] = []
        for slot_tensor in active_slots.detach().cpu():
            slot = int(slot_tensor)
            ids_for_slot = channel_ids[:, slot]
            nonzero = ids_for_slot[ids_for_slot > 0]
            if nonzero.numel() == 0:
                continue
            channel_id = int(nonzero[0].item())
            name = self.canonical_map.id_to_name.get(channel_id)
            if not name:
                continue
            names.append(name)
            keep_slots.append(slot)

        if not keep_slots:
            raise ValueError(
                "REVEBackbone could not map active channel_ids to electrode names. "
                "Check canonical_channel_ids in the processed manifest."
            )

        cache_key = tuple(names)
        if cache_key in self._position_cache:
            keep_indices, base_positions = self._position_cache[cache_key]
        else:
            keep_indices, base_positions, missing_names = self._resolve_positions(names)
            self._position_cache[cache_key] = (keep_indices, base_positions.detach().cpu())
            if missing_names:
                print(
                    "REVEBackbone dropped channel(s) without REVE positions: "
                    f"{', '.join(missing_names)}"
                )

        filtered_slots = [keep_slots[i] for i in keep_indices]
        if not filtered_slots:
            raise ValueError(
                "REVEBackbone could not resolve positions for any active channels. "
                "Check configs/canonical_channels.yaml aliases."
            )

        selected_mask = mask[:, filtered_slots].unsqueeze(-1).to(x.dtype)
        x_reve = x[:, filtered_slots, :] * selected_mask
        positions = base_positions.to(device=x.device, dtype=x.dtype)
        if positions.ndim == 2:
            positions = positions.unsqueeze(0).expand(x.size(0), -1, -1)
        return x_reve, positions

    def _resolve_positions(self, names: list[str]) -> tuple[list[int], torch.Tensor, list[str]]:
        try:
            positions = self._standardize_positions(self.pos_bank(names))
        except Exception as exc:
            raise ValueError(
                f"REVE position bank could not resolve electrode names: {names}. "
                "Check configs/canonical_channels.yaml aliases."
            ) from exc
        if positions.shape[0] == len(names):
            return list(range(len(names))), positions, []

        keep_indices: list[int] = []
        resolved: list[torch.Tensor] = []
        missing_names: list[str] = []
        for i, name in enumerate(names):
            try:
                pos = self._standardize_positions(self.pos_bank([name]))
            except Exception:
                missing_names.append(name)
                continue
            if pos.shape[0] != 1:
                missing_names.append(name)
                continue
            keep_indices.append(i)
            resolved.append(pos)
        if not resolved:
            return [], positions[:0], missing_names
        return keep_indices, torch.cat(resolved, dim=0), missing_names

    @staticmethod
    def _standardize_positions(positions) -> torch.Tensor:
        if not isinstance(positions, torch.Tensor):
            positions = torch.as_tensor(positions)
        if positions.ndim == 3:
            if positions.shape[0] != 1:
                raise ValueError(f"Expected REVE positions batch size 1, got shape {tuple(positions.shape)}")
            positions = positions.squeeze(0)
        if positions.ndim != 2:
            raise ValueError(f"Expected REVE positions with shape [channels, dim], got {tuple(positions.shape)}")
        return positions



def extract_reve_tokens(out) -> torch.Tensor:
    """Standardize REVE output to a trainable prompt-fusion sequence [B, N, D]."""
    value = None
    if hasattr(out, "last_hidden_state") and out.last_hidden_state is not None:
        value = out.last_hidden_state
    elif hasattr(out, "pooler_output") and out.pooler_output is not None:
        value = out.pooler_output
    elif isinstance(out, torch.Tensor):
        value = out
    elif isinstance(out, dict):
        for key in ["last_hidden_state", "embeddings", "h", "pooler_output"]:
            if key in out and out[key] is not None:
                value = out[key]
                break
    if value is None:
        raise RuntimeError(f"Cannot extract representation from REVE output type: {type(out)}")
    if value.ndim == 2:
        return value.unsqueeze(1)
    if value.ndim >= 3:
        return value.reshape(value.shape[0], -1, value.shape[-1])
    raise RuntimeError(f"Cannot standardize REVE tensor with shape {tuple(value.shape)}")


def _compatible_heads(d_model: int, requested: int) -> int:
    for n_heads in range(min(requested, d_model), 0, -1):
        if d_model % n_heads == 0:
            return n_heads
    return 1

def extract_reve_representation(out):
    """Backward-compatible pooled view of the standardized REVE tokens."""
    return extract_reve_tokens(out).mean(dim=1)


def _pool_reve_tensor(value: torch.Tensor) -> torch.Tensor:
    if value.ndim == 2:
        return value
    if value.ndim == 3:
        return value.mean(dim=1)
    if value.ndim > 3:
        # REVE may return structured embeddings such as [B, C, T, D].
        # Keep batch and feature dimensions, pool all structure in between.
        dims = tuple(range(1, value.ndim - 1))
        return value.mean(dim=dims)
    raise RuntimeError(f"Cannot pool REVE tensor with shape {tuple(value.shape)}")


def _infer_reve_dim(model, cfg: dict) -> int:
    if cfg.get("d_model") is not None and cfg.get("force_d_model", False):
        return int(cfg["d_model"])
    config = getattr(model, "config", None)
    for name in [
        "hidden_size",
        "d_model",
        "embed_dim",
        "embedding_dim",
        "encoder_embed_dim",
        "dim",
        "width",
    ]:
        value = getattr(config, name, None)
        if value is not None:
            return int(value)
    return int(cfg.get("d_model", 128))
