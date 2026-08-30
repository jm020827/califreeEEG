from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from cfeg.constants import (
    CATEGORICAL_VOCABS,
    EXTERNAL_CONTINUOUS_SCHEMA_V1,
    METADATA_CONTRACT_LEGACY,
    METADATA_CONTRACT_V04_DEV,
    PROTOCOL_V04_EXTERNAL_CATEGORICAL_FIELDS,
    QUERY_QC_EXTRACTOR_V1,
)
from cfeg.data.metadata_controls import (
    DEVELOPMENT_CONTROL_NONE,
    normalize_development_control,
)
from cfeg.models.adapters import BottleneckAdapter, ConditionedAdapter
from cfeg.models.backbones.reve import REVEBackbone
from cfeg.models.backbones.spectral_transformer import SpectralEEGTransformerBackbone
from cfeg.models.backbones.tiny_transformer import TinyEEGTransformerBackbone
from cfeg.models.condition_encoder import ConditionEncoder
from cfeg.models.heads import ClassificationHead
from cfeg.models.latent_nuisance import LatentNuisanceEncoder
from cfeg.models.physical_conditioning import (
    PHYSICAL_HYBRID_V1,
    PROMPT_ADAPTER_V1,
    FactorizedPhysicalConditioner,
    PhysicalConditionState,
    ZeroInitResidualFiLM,
)


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
        elif name == "spectral_transformer":
            self.backbone = SpectralEEGTransformerBackbone(
                c_max=int(model_cfg.get("c_max", 64)),
                t_len=int(model_cfg.get("t_len", 400)),
                target_sfreq=float(model_cfg.get("target_sfreq", 200.0)),
                d_model=d_model,
                depth=int(model_cfg.get("depth", 4)),
                n_heads=int(model_cfg.get("n_heads", 4)),
                min_frequency_hz=float(backbone_cfg.get("min_frequency_hz", 6.0)),
                max_frequency_hz=float(backbone_cfg.get("max_frequency_hz", 60.0)),
                dropout=float(backbone_cfg.get("dropout", 0.1)),
                channel_vocab_size=int(model_cfg.get("c_max", 64)) + 1,
            )
        elif name == "reve":
            reve_cfg = dict(backbone_cfg)
            reve_cfg.setdefault("hf_model", "brain-bzh/reve-base")
            reve_cfg.setdefault("hf_positions", "brain-bzh/reve-positions")
            reve_cfg.setdefault("cache_dir", None)
            reve_cfg.setdefault("trust_remote_code", True)
            reve_cfg.setdefault("local_files_only", True)
            reve_cfg.setdefault("freeze", True)
            reve_cfg.setdefault("d_model", d_model)
            self.backbone = REVEBackbone(reve_cfg)
            d_model = self.backbone.d_model
        else:
            raise ValueError(f"Unknown backbone: {name}")

        ce_cfg = model_cfg.get("condition_encoder", {})
        protocol_cfg = cfg.get("protocol", {})
        self.metadata_contract_version = str(
            protocol_cfg.get("metadata_contract_version", METADATA_CONTRACT_LEGACY)
        )
        self.external_metadata_mode = str(ce_cfg.get("external_metadata_mode", "observed"))
        self.development_control = normalize_development_control(
            protocol_cfg.get("development_control", DEVELOPMENT_CONTROL_NONE)
        )
        self.query_qc_extractor_version = str(
            protocol_cfg.get("query_qc_extractor_version", QUERY_QC_EXTRACTOR_V1)
        )
        self.external_continuous_schema = str(
            protocol_cfg.get("external_continuous_schema", EXTERNAL_CONTINUOUS_SCHEMA_V1)
        )
        self.condition_enabled = bool(ce_cfg.get("enabled", True))
        conditioning_cfg = model_cfg.get("conditioning", {})
        self.conditioning_architecture = str(
            conditioning_cfg.get("architecture", PROMPT_ADAPTER_V1)
        )
        if self.conditioning_architecture not in {PROMPT_ADAPTER_V1, PHYSICAL_HYBRID_V1}:
            raise ValueError(
                f"Unknown model.conditioning.architecture={self.conditioning_architecture!r}."
            )
        if self.conditioning_architecture == PHYSICAL_HYBRID_V1 and not self.condition_enabled:
            raise ValueError("physical_hybrid_v1 requires condition_encoder.enabled=true.")
        vocab_sizes = vocab_sizes or {name: len(vals) for name, vals in CATEGORICAL_VOCABS.items()}
        self.physical_conditioner: FactorizedPhysicalConditioner | None = None
        self.query_film: ZeroInitResidualFiLM | None = None
        self.external_film: ZeroInitResidualFiLM | None = None
        if self.condition_enabled and self.conditioning_architecture == PHYSICAL_HYBRID_V1:
            if self.metadata_contract_version != METADATA_CONTRACT_V04_DEV:
                raise ValueError("physical_hybrid_v1 requires metadata_contract_version='0.4-dev'.")
            if int(ce_cfg.get("n_prompt_tokens", 0)) != 0:
                raise ValueError("physical_hybrid_v1 requires condition_encoder.n_prompt_tokens=0.")
            if bool(ce_cfg.get("include_channels", False)):
                raise ValueError("physical_hybrid_v1 keeps channel structure in the backbone only.")
            if not self.backbone.supports_channel_gain:
                raise ValueError(
                    f"{type(self.backbone).__name__} does not support physical channel gain."
                )
            query_cfg = conditioning_cfg.get("common_query_film", {})
            external_cfg = conditioning_cfg.get("external_global_film", {})
            channel_cfg = conditioning_cfg.get("channel_quality", {})
            if channel_cfg.get("source", "channel_impedance") != "channel_impedance":
                raise ValueError("physical_hybrid_v1 channel_quality.source must be channel_impedance.")
            if channel_cfg.get("placement", "pre_channel_embedding") != "pre_channel_embedding":
                raise ValueError(
                    "physical_hybrid_v1 channel_quality.placement must be pre_channel_embedding."
                )
            self.physical_conditioner = FactorizedPhysicalConditioner(
                d_model=d_model,
                vocab_sizes=vocab_sizes,
                fields=list(ce_cfg.get("fields") or ["electrode_type"]),
                external_metadata_mode=self.external_metadata_mode,
                hidden_dim=int(conditioning_cfg.get("hidden_dim", 64)),
                max_channel_gain_delta=float(channel_cfg.get("max_gain_delta", 0.25)),
                query_enabled=bool(query_cfg.get("enabled", True)),
                external_global_enabled=bool(external_cfg.get("enabled", True)),
                channel_quality_enabled=bool(channel_cfg.get("enabled", True)),
            )
            self.query_film = ZeroInitResidualFiLM(
                d_model,
                max_scale_delta=float(query_cfg.get("max_scale_delta", 0.25)),
                max_shift=float(query_cfg.get("max_shift", 0.25)),
            )
            self.external_film = ZeroInitResidualFiLM(
                d_model,
                max_scale_delta=float(external_cfg.get("max_scale_delta", 0.25)),
                max_shift=float(external_cfg.get("max_shift", 0.25)),
            )
            self.condition_encoder = None
        elif self.condition_enabled:
            self.condition_encoder = ConditionEncoder(
                d_model=d_model,
                n_prompt_tokens=int(ce_cfg.get("n_prompt_tokens", 4)),
                vocab_sizes=vocab_sizes,
                n_cont_features=5,
                channel_vocab_size=int(model_cfg.get("c_max", 64)) + 1,
                fields=ce_cfg.get("fields"),
                include_continuous=bool(ce_cfg.get("include_continuous", True)),
                include_channels=bool(ce_cfg.get("include_channels", True)),
                force_missing=bool(ce_cfg.get("force_missing", False)),
                metadata_contract_version=self.metadata_contract_version,
                external_metadata_mode=self.external_metadata_mode,
            )
        else:
            self.condition_encoder = None

        adapter_cfg = model_cfg.get("adapter", {})
        if self.conditioning_architecture == PHYSICAL_HYBRID_V1 and adapter_cfg.get(
            "enabled", True
        ):
            raise ValueError("physical_hybrid_v1 replaces the legacy adapter; set adapter.enabled=false.")
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

    def load_state_dict(self, state_dict, strict: bool = True, assign: bool = False):
        self.materialize_checkpoint_modules(state_dict)
        try:
            return super().load_state_dict(state_dict, strict=strict, assign=assign)
        except TypeError:
            return super().load_state_dict(state_dict, strict=strict)

    def materialize_checkpoint_modules(self, state_dict) -> None:
        projection_weight = state_dict.get("backbone.output_proj.weight")
        if projection_weight is not None and getattr(self.backbone, "output_proj", None) is None:
            out_features, in_features = projection_weight.shape
            self.backbone.output_proj = nn.Linear(in_features, out_features)

    def forward(
        self,
        x: torch.Tensor,
        cond: dict[str, torch.Tensor],
        use_latent: bool | None = None,
        return_repr: bool = True,
    ) -> ModelOutput:
        observed_contract = str(cond.get("metadata_contract_version", METADATA_CONTRACT_LEGACY))
        if observed_contract != self.metadata_contract_version:
            raise ValueError(
                "Batch/model metadata contract mismatch: "
                f"batch={observed_contract!r}, model={self.metadata_contract_version!r}."
            )
        observed_mode = str(cond.get("external_metadata_mode", "observed"))
        if observed_mode != self.external_metadata_mode:
            raise ValueError(
                "Batch/model external metadata treatment mismatch: "
                f"batch={observed_mode!r}, model={self.external_metadata_mode!r}."
            )
        observed_control = normalize_development_control(
            cond.get("development_control", DEVELOPMENT_CONTROL_NONE)
        )
        if observed_control != self.development_control:
            raise ValueError(
                "Batch/model development-control mismatch: "
                f"batch={observed_control!r}, model={self.development_control!r}."
            )
        if self.metadata_contract_version == METADATA_CONTRACT_V04_DEV:
            for key, expected in (
                ("query_qc_extractor_version", self.query_qc_extractor_version),
                ("external_continuous_schema", self.external_continuous_schema),
            ):
                observed = str(cond.get(key, "missing"))
                if observed != expected:
                    raise ValueError(
                        f"Batch/model protocol schema mismatch for {key}: "
                        f"batch={observed!r}, model={expected!r}."
                    )
        prompt, cond_vec = (None, None)
        physical_state: PhysicalConditionState | None = None
        condition_view = _development_control_condition_view(cond, self.development_control)
        if self.physical_conditioner is not None:
            physical_state = self.physical_conditioner(condition_view)
            cond_vec = physical_state.external_vec
        elif self.condition_encoder is not None:
            prompt, cond_vec = self.condition_encoder(
                condition_view
            )
        condition_only = self.development_control in {"metadata_only", "missingness_only"}
        if condition_only and physical_state is not None:
            cond_vec = physical_state.metadata_only_vec
        if condition_only and cond_vec is None:
            raise RuntimeError(f"{self.development_control} requires an enabled condition encoder.")
        if prompt is not None and not self.backbone.supports_prompt_tokens and not condition_only:
            raise RuntimeError(
                f"{type(self.backbone).__name__} does not support condition prompt tokens."
            )
        if condition_only:
            # Shortcut classifiers must be exactly invariant to waveform and
            # structural/query inputs. Keep the backbone parameters in the
            # checkpoint schema, but never execute or consume the backbone.
            h = cond_vec
            backbone_aux = {}
        else:
            # ModelOutput exposes the pooled representation, not backbone tokens.
            # Avoid allocating variable-length padded token tensors that are discarded.
            backbone_kwargs = {}
            if physical_state is not None:
                backbone_kwargs["channel_gain"] = physical_state.channel_gain
            backbone_out = self.backbone(
                x,
                cond=cond,
                prompt_tokens=prompt,
                return_tokens=False,
                **backbone_kwargs,
            )
            h = backbone_out.h
            backbone_aux = backbone_out.aux
            if physical_state is not None:
                if self.query_film is None or self.external_film is None:
                    raise RuntimeError("physical_hybrid_v1 FiLM modules were not initialized.")
                h = self.query_film(
                    h, physical_state.query_vec, physical_state.query_available
                )
                h = self.external_film(
                    h, physical_state.external_vec, physical_state.external_available
                )
                backbone_aux = {
                    **backbone_aux,
                    "query_qc_available": physical_state.query_available.squeeze(-1),
                    "external_metadata_available": physical_state.external_available.squeeze(-1),
                    "channel_impedance_available_count": physical_state.channel_available.sum(
                        dim=-1
                    ),
                    "channel_gain_min": physical_state.channel_gain.min(dim=-1).values,
                    "channel_gain_max": physical_state.channel_gain.max(dim=-1).values,
                }
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
            aux=backbone_aux,
        )


def _development_control_condition_view(
    cond: dict[str, torch.Tensor], development_control: str
) -> dict[str, torch.Tensor]:
    if development_control not in {"metadata_only", "missingness_only"}:
        return cond
    out = {key: value.clone() if torch.is_tensor(value) else value for key, value in cond.items()}
    out["query_qc"].zero_()
    out["query_qc_missing"].fill_(True)
    if development_control == "missingness_only":
        out["external_continuous"].zero_()
        out["channel_impedance"].zero_()
        for field in PROTOCOL_V04_EXTERNAL_CATEGORICAL_FIELDS:
            out[field] = out[field].ne(0).long()
    return out
