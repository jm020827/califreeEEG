from __future__ import annotations

import torch
from torch import nn
from torch.nn.utils.rnn import pad_sequence

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
            self.d_model,
            n_heads,
            dropout=float(cfg.get("prompt_fusion_dropout", 0.1)),
            batch_first=True,
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
        tokens, token_valid = self._encode_mask_groups(x, cond)
        valid_float = token_valid.unsqueeze(-1).to(tokens.dtype)
        h = (tokens * valid_float).sum(dim=1) / valid_float.sum(dim=1).clamp_min(1.0)
        aux: dict[str, torch.Tensor] = {
            "reve_token_count": token_valid.sum(dim=1),
        }
        if prompt_tokens is not None:
            attended, weights = self.prompt_attention(
                self.prompt_norm(prompt_tokens),
                self.prompt_norm(tokens),
                self.prompt_norm(tokens),
                key_padding_mask=~token_valid,
            )
            prompt_summary = attended.mean(dim=1)
            h = self.prompt_norm(h + self.prompt_gate(prompt_tokens.mean(dim=1)) * prompt_summary)
            safe_weights = weights.clamp_min(torch.finfo(weights.dtype).tiny)
            aux["prompt_attention_entropy"] = (
                -(weights * safe_weights.log()).sum(dim=-1).mean(dim=-1)
            )
            aux["prompt_attention_max"] = weights.max(dim=-1).values.mean(dim=-1)
        returned_tokens = None
        if return_tokens:
            if prompt_tokens is not None:
                returned_tokens = torch.cat([prompt_tokens, tokens], dim=1)
                prompt_valid = torch.ones(
                    (token_valid.shape[0], prompt_tokens.shape[1]),
                    dtype=torch.bool,
                    device=token_valid.device,
                )
                aux["reve_returned_token_valid_mask"] = torch.cat(
                    [prompt_valid, token_valid], dim=1
                )
            else:
                returned_tokens = tokens
                aux["reve_returned_token_valid_mask"] = token_valid
        return BackboneOutput(h=h, tokens=returned_tokens, aux=aux)

    def _encode_mask_groups(
        self, x: torch.Tensor, cond: dict[str, torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Run REVE on exact per-sample structures, batching only equal masks/IDs."""
        batch_size = x.shape[0]
        mask_cpu = cond["channel_mask"].bool().detach().cpu()
        ids_cpu = cond["channel_ids"].detach().cpu()
        groups: dict[tuple[tuple[int, int], ...], list[int]] = {}
        for batch_index in range(batch_size):
            signature = tuple(
                (slot, int(ids_cpu[batch_index, slot].item()))
                for slot in torch.nonzero(mask_cpu[batch_index], as_tuple=False).flatten().tolist()
            )
            groups.setdefault(signature, []).append(batch_index)

        token_rows: list[torch.Tensor | None] = [None] * batch_size
        for indices in groups.values():
            index = torch.as_tensor(indices, dtype=torch.long, device=x.device)
            group_cond = {
                key: (
                    value.index_select(0, index)
                    if torch.is_tensor(value) and value.ndim > 0 and value.shape[0] == batch_size
                    else value
                )
                for key, value in cond.items()
            }
            x_reve, positions = self._select_channels_and_positions(
                x.index_select(0, index), group_cond
            )
            if self.freeze:
                with torch.no_grad():
                    out = self._forward_reve(x_reve, positions)
            else:
                out = self._forward_reve(x_reve, positions)
            group_tokens = self._project_tokens(extract_reve_tokens(out))
            for local_index, batch_index in enumerate(indices):
                token_rows[batch_index] = group_tokens[local_index]

        if any(row is None for row in token_rows):
            raise RuntimeError("REVE failed to produce tokens for every batch sample.")
        rows = [row for row in token_rows if row is not None]
        lengths = torch.as_tensor([row.shape[0] for row in rows], dtype=torch.long, device=x.device)
        padded = pad_sequence(rows, batch_first=True)
        positions = torch.arange(padded.shape[1], device=x.device).unsqueeze(0)
        valid = positions < lengths.unsqueeze(1)
        return padded, valid

    def _project_tokens(self, tokens: torch.Tensor) -> torch.Tensor:
        if tokens.shape[-1] == self.d_model:
            return tokens
        if self.output_proj is None:
            self.output_proj = nn.Linear(tokens.shape[-1], self.d_model).to(
                device=tokens.device, dtype=tokens.dtype
            )
        return self.output_proj(tokens)

    def train(self, mode: bool = True):
        super().train(mode)
        # The position bank is always frozen and must remain deterministic,
        # including the optional full-REVE-finetune variant.
        self.pos_bank.eval()
        if getattr(self, "freeze", False):
            self.reve.eval()
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
        unknown_active = mask.detach().cpu() & channel_ids.eq(0)
        if unknown_active.any():
            raise ValueError(
                "REVEBackbone cannot silently drop active channels with unknown canonical "
                "ID 0. Add the electrode to configs/canonical_channels.yaml or exclude "
                "the dataset from REVE experiments."
            )
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
                raise ValueError(
                    f"REVEBackbone has no electrode name for active canonical ID {channel_id}."
                )
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
        # REVE may add positional noise in-place while fine-tuning. Clone keeps
        # the cached base immutable and avoids writes through stride-0 expand views.
        positions = base_positions.to(device=x.device, dtype=x.dtype).clone()
        if positions.ndim == 2:
            positions = positions.unsqueeze(0).expand(x.size(0), -1, -1).clone()
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
            # The optional third-party position bank does not expose one stable
            # exception type for unknown or malformed electrode names.
            except Exception:  # noqa: BLE001
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
                raise ValueError(
                    f"Expected REVE positions batch size 1, got shape {tuple(positions.shape)}"
                )
            positions = positions.squeeze(0)
        if positions.ndim != 2:
            raise ValueError(
                f"Expected REVE positions with shape [channels, dim], got {tuple(positions.shape)}"
            )
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
