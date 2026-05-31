from __future__ import annotations

import os

import torch
from torch import nn

from cfeg.assets.errors import MissingAssetError
from cfeg.assets.hf import hf_cache_hint
from cfeg.data.preprocess import CanonicalChannelMap
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
        self.cache_dir = cfg.get("cache_dir") or os.environ.get("HF_HOME")
        self.required_sample_rate_hz = float(cfg.get("required_sample_rate_hz", 200.0))
        self.pos_bank = self._load_model(AutoModel, cfg["hf_positions"], cfg)
        self.reve = self._load_model(AutoModel, cfg["hf_model"], cfg)
        self.d_model = _infer_reve_dim(self.reve, cfg)
        self.canonical_map = CanonicalChannelMap.from_yaml()
        self.output_proj: nn.Module | None = None
        self._position_cache: dict[tuple[str, ...], tuple[list[int], torch.Tensor]] = {}
        if cfg.get("freeze", True):
            for p in self.reve.parameters():
                p.requires_grad = False

    @staticmethod
    def _load_model(auto_model, repo_id: str, cfg: dict):
        cache_dir = cfg.get("cache_dir") or os.environ.get("HF_HOME")
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
        if prompt_tokens is not None:
            # REVE remote code does not expose a stable prompt-token prepend contract.
            prompt_tokens = None
        x_reve, positions = self._select_channels_and_positions(x, cond)
        try:
            out = self.reve(x_reve, positions)
        except TypeError:
            out = self.reve(x_reve, pos=positions)
        h = extract_reve_representation(out)
        if h.shape[-1] != self.d_model:
            if self.output_proj is None:
                self.output_proj = nn.Linear(h.shape[-1], self.d_model).to(device=h.device, dtype=h.dtype)
            h = self.output_proj(h)
        return BackboneOutput(h=h, tokens=None, aux={})

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

        x_reve = x[:, filtered_slots, :]
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


def extract_reve_representation(out):
    if hasattr(out, "pooler_output") and out.pooler_output is not None:
        return _pool_reve_tensor(out.pooler_output)
    if hasattr(out, "last_hidden_state"):
        return _pool_reve_tensor(out.last_hidden_state)
    if isinstance(out, torch.Tensor):
        return _pool_reve_tensor(out)
    if isinstance(out, dict):
        for key in ["pooler_output", "last_hidden_state", "embeddings", "h"]:
            if key in out:
                value = out[key]
                return _pool_reve_tensor(value)
    raise RuntimeError(f"Cannot extract representation from REVE output type: {type(out)}")


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
