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
from cfeg.models.query_reliability_conditioning import (
    QUERY_RELIABILITY_FEATURE_SCHEMA_V3,
    QUERY_RELIABILITY_SPATIAL_V1,
    QueryReliabilityOperator,
    QueryReliabilityState,
)
from cfeg.models.reliability_conditioning import (
    RELIABILITY_SPATIAL_V1,
    ReliabilityConditionState,
    ResidualizedReliabilityOperator,
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
        if self.conditioning_architecture not in {
            PROMPT_ADAPTER_V1,
            PHYSICAL_HYBRID_V1,
            RELIABILITY_SPATIAL_V1,
            QUERY_RELIABILITY_SPATIAL_V1,
        }:
            raise ValueError(
                f"Unknown model.conditioning.architecture={self.conditioning_architecture!r}."
            )
        if (
            self.conditioning_architecture
            in {
                PHYSICAL_HYBRID_V1,
                RELIABILITY_SPATIAL_V1,
                QUERY_RELIABILITY_SPATIAL_V1,
            }
            and not self.condition_enabled
        ):
            raise ValueError(
                f"{self.conditioning_architecture} requires condition_encoder.enabled=true."
            )
        vocab_sizes = vocab_sizes or {name: len(vals) for name, vals in CATEGORICAL_VOCABS.items()}
        self.physical_conditioner: FactorizedPhysicalConditioner | None = None
        self.reliability_conditioner: ResidualizedReliabilityOperator | None = None
        self.query_reliability_conditioner: QueryReliabilityOperator | None = None
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
        elif (
            self.condition_enabled
            and self.conditioning_architecture == QUERY_RELIABILITY_SPATIAL_V1
        ):
            if self.metadata_contract_version != METADATA_CONTRACT_V04_DEV:
                raise ValueError(
                    "query_reliability_spatial_v1 requires metadata_contract_version="
                    "'0.4-dev' for query-QC transport."
                )
            if protocol_cfg.get("reliability_query_feature_schema") != (
                QUERY_RELIABILITY_FEATURE_SCHEMA_V3
            ):
                raise ValueError(
                    "query_reliability_spatial_v1 requires its exact band-matched "
                    "Q feature schema."
                )
            if self.external_metadata_mode != "null":
                raise ValueError(
                    "query_reliability_spatial_v1 forbids external metadata; "
                    "set condition_encoder.external_metadata_mode='null'."
                )
            if int(ce_cfg.get("n_prompt_tokens", 0)) != 0:
                raise ValueError(
                    "query_reliability_spatial_v1 requires condition_encoder.n_prompt_tokens=0."
                )
            if list(ce_cfg.get("fields") or []):
                raise ValueError(
                    "query_reliability_spatial_v1 requires condition_encoder.fields=[]."
                )
            if bool(ce_cfg.get("include_continuous", False)):
                raise ValueError(
                    "query_reliability_spatial_v1 forbids continuous metadata features."
                )
            if bool(ce_cfg.get("include_channels", False)):
                raise ValueError(
                    "query_reliability_spatial_v1 keeps channel identity in the backbone only."
                )
            if bool(ce_cfg.get("force_missing", False)):
                raise ValueError(
                    "query_reliability_spatial_v1 uses an explicit Q access axis; "
                    "condition_encoder.force_missing is forbidden."
                )
            if name != "spectral_transformer":
                raise ValueError(
                    "query_reliability_spatial_v1 is frozen to the spectral_transformer "
                    "backbone."
                )
            reliability_cfg = conditioning_cfg.get("spatial_reliability", {})
            if set(reliability_cfg).intersection(
                {
                    "metadata_operator_norm",
                    "metadata_alpha_limit",
                    "metadata_residual_enabled",
                }
            ):
                raise ValueError(
                    "query_reliability_spatial_v1 must not declare a metadata residual branch."
                )
            if (
                reliability_cfg.get("placement", "pre_backbone_waveform")
                != "pre_backbone_waveform"
            ):
                raise ValueError(
                    "query_reliability_spatial_v1 spatial_reliability.placement must be "
                    "pre_backbone_waveform."
                )
            if reliability_cfg.get("operator_family", "diagonal_low_rank") != (
                "diagonal_low_rank"
            ):
                raise ValueError(
                    "query_reliability_spatial_v1 supports only "
                    "operator_family=diagonal_low_rank."
                )
            self.query_reliability_conditioner = QueryReliabilityOperator(
                hidden_dim=int(conditioning_cfg.get("hidden_dim", 64)),
                operator_rank=int(reliability_cfg.get("operator_rank", 4)),
                max_operator_norm=float(
                    reliability_cfg.get("max_operator_norm", 0.20)
                ),
                query_qc_mode=str(reliability_cfg.get("query_qc_mode", "observed")),
                target_sfreq=float(model_cfg.get("target_sfreq", 200.0)),
                min_frequency_hz=float(backbone_cfg.get("min_frequency_hz", 6.0)),
                max_frequency_hz=float(backbone_cfg.get("max_frequency_hz", 60.0)),
            )
            self.condition_encoder = None
        elif self.condition_enabled and self.conditioning_architecture == RELIABILITY_SPATIAL_V1:
            if self.metadata_contract_version != METADATA_CONTRACT_V04_DEV:
                raise ValueError(
                    "reliability_spatial_v1 requires metadata_contract_version='0.4-dev'."
                )
            if int(ce_cfg.get("n_prompt_tokens", 0)) != 0:
                raise ValueError(
                    "reliability_spatial_v1 requires condition_encoder.n_prompt_tokens=0."
                )
            if bool(ce_cfg.get("include_channels", False)):
                raise ValueError(
                    "reliability_spatial_v1 keeps channel structure in the backbone only."
                )
            if bool(ce_cfg.get("force_missing", False)):
                raise ValueError(
                    "reliability_spatial_v1 does not permit condition_encoder.force_missing=true; "
                    "use the declared Q/M access axes instead."
                )
            if not bool(ce_cfg.get("include_continuous", True)):
                raise ValueError(
                    "reliability_spatial_v1 requires condition_encoder.include_continuous=true."
                )
            if name != "spectral_transformer":
                raise ValueError(
                    "reliability_spatial_v1 is currently frozen to the spectral_transformer "
                    "backbone."
                )
            reliability_cfg = conditioning_cfg.get("spatial_reliability", {})
            if (
                reliability_cfg.get("placement", "pre_backbone_waveform")
                != "pre_backbone_waveform"
            ):
                raise ValueError(
                    "reliability_spatial_v1 spatial_reliability.placement must be "
                    "pre_backbone_waveform."
                )
            if reliability_cfg.get("operator_family", "diagonal_low_rank") != (
                "diagonal_low_rank"
            ):
                raise ValueError(
                    "reliability_spatial_v1 currently supports only "
                    "operator_family=diagonal_low_rank."
                )
            self.reliability_conditioner = ResidualizedReliabilityOperator(
                d_model=d_model,
                vocab_sizes=vocab_sizes,
                fields=list(ce_cfg.get("fields") or ["electrode_type"]),
                external_metadata_mode=self.external_metadata_mode,
                hidden_dim=int(conditioning_cfg.get("hidden_dim", 64)),
                operator_rank=int(reliability_cfg.get("operator_rank", 4)),
                max_operator_norm=float(reliability_cfg.get("max_operator_norm", 0.25)),
                query_operator_norm=float(
                    reliability_cfg.get("query_operator_norm", 0.20)
                ),
                metadata_operator_norm=float(
                    reliability_cfg.get("metadata_operator_norm", 0.05)
                ),
                metadata_alpha_limit=float(
                    reliability_cfg.get("metadata_alpha_limit", 1.0)
                ),
                query_qc_mode=str(reliability_cfg.get("query_qc_mode", "observed")),
                metadata_residual_enabled=bool(
                    reliability_cfg.get("metadata_residual_enabled", True)
                ),
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
        if self.conditioning_architecture in {
            PHYSICAL_HYBRID_V1,
            RELIABILITY_SPATIAL_V1,
            QUERY_RELIABILITY_SPATIAL_V1,
        } and adapter_cfg.get("enabled", True):
            raise ValueError(
                f"{self.conditioning_architecture} replaces the legacy adapter; "
                "set adapter.enabled=false."
            )
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
        if self.conditioning_architecture in {
            RELIABILITY_SPATIAL_V1,
            QUERY_RELIABILITY_SPATIAL_V1,
        } and self.latent_enabled:
            raise ValueError(
                f"{self.conditioning_architecture} is a spatial-operator-only treatment; "
                "set latent.enabled=false."
            )
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
        query_reliability_identity_intervention: bool = False,
        query_reliability_wrong_query_x: torch.Tensor | None = None,
        query_reliability_wrong_query_cond: dict[str, torch.Tensor] | None = None,
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
        reliability_state: ReliabilityConditionState | None = None
        query_reliability_state: QueryReliabilityState | None = None
        wrong_query_intervention = (
            query_reliability_wrong_query_x is not None
            or query_reliability_wrong_query_cond is not None
        )
        if (query_reliability_wrong_query_x is None) != (
            query_reliability_wrong_query_cond is None
        ):
            raise ValueError("The wrong-query intervention requires both donor x and cond.")
        if query_reliability_identity_intervention and wrong_query_intervention:
            raise ValueError("Identity and wrong-query interventions are mutually exclusive.")
        if query_reliability_identity_intervention or wrong_query_intervention:
            if (
                self.query_reliability_conditioner is None
                or self.query_reliability_conditioner.query_qc_mode != "observed"
            ):
                raise ValueError(
                    "Query interventions are defined only for observed-Q1 checkpoints."
                )
            if self.training:
                raise ValueError(
                    "Q1 query interventions are evaluation-only and cannot affect training."
                )
        condition_view = _development_control_condition_view(cond, self.development_control)
        if self.physical_conditioner is not None:
            physical_state = self.physical_conditioner(condition_view)
            cond_vec = physical_state.external_vec
        elif self.reliability_conditioner is not None:
            reliability_state = self.reliability_conditioner(x, condition_view)
        elif self.query_reliability_conditioner is not None:
            query_x = x
            query_condition_view = condition_view
            if wrong_query_intervention:
                assert query_reliability_wrong_query_x is not None
                assert query_reliability_wrong_query_cond is not None
                query_x, query_condition_view = _validate_wrong_query_intervention(
                    target_x=x,
                    target_cond=condition_view,
                    donor_x=query_reliability_wrong_query_x,
                    donor_cond=query_reliability_wrong_query_cond,
                )
            query_reliability_state = self.query_reliability_conditioner(
                query_x, query_condition_view
            )
        elif self.condition_encoder is not None:
            prompt, cond_vec = self.condition_encoder(
                condition_view
            )
        condition_only = self.development_control in {"metadata_only", "missingness_only"}
        if condition_only and physical_state is not None:
            cond_vec = physical_state.metadata_only_vec
        if condition_only and reliability_state is not None:
            raise RuntimeError(
                "reliability_spatial_v1 has no neural metadata-only decoder route; "
                "use its frozen deterministic Stage-1 leakage probe."
            )
        if condition_only and query_reliability_state is not None:
            raise RuntimeError(
                "query_reliability_spatial_v1 does not permit metadata-only controls."
            )
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
            backbone_x = x
            if physical_state is not None:
                backbone_kwargs["channel_gain"] = physical_state.channel_gain
            if reliability_state is not None:
                backbone_x = reliability_state.apply(x)
            if query_reliability_state is not None:
                if query_reliability_identity_intervention:
                    backbone_x = torch.where(
                        query_reliability_state.channel_mask.unsqueeze(-1),
                        x,
                        torch.zeros((), dtype=x.dtype, device=x.device),
                    )
                else:
                    backbone_x = query_reliability_state.apply(x)
            backbone_out = self.backbone(
                backbone_x,
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
            if reliability_state is not None:
                backbone_aux = {
                    **backbone_aux,
                    "channel_spatial_self_gain": reliability_state.diagonal_gain,
                    "spatial_offdiagonal_row_l1": reliability_state.offdiagonal_row_l1,
                    "channel_query_noise_logit": reliability_state.query_noise_logit,
                    "channel_metadata_noise_logit": reliability_state.metadata_noise_logit,
                    "channel_combined_noise_logit": reliability_state.combined_noise_logit,
                    "channel_query_available_count": (
                        reliability_state.query_available.sum(dim=-1)
                    ),
                    "channel_metadata_available_count": (
                        reliability_state.metadata_available.sum(dim=-1)
                    ),
                    "metadata_residual_alpha": reliability_state.metadata_alpha.expand(
                        x.shape[0]
                    ),
                    "spatial_operator_frobenius_norm": torch.linalg.vector_norm(
                        reliability_state.operator_delta.float().flatten(start_dim=1),
                        dim=-1,
                    ),
                    "query_spatial_operator_frobenius_norm": torch.linalg.vector_norm(
                        reliability_state.query_operator_delta.float().flatten(start_dim=1),
                        dim=-1,
                    ),
                    "metadata_spatial_residual_frobenius_norm": torch.linalg.vector_norm(
                        reliability_state.metadata_operator_delta.float().flatten(start_dim=1),
                        dim=-1,
                    ),
                    "spatial_self_gain_min": reliability_state.diagonal_gain.min(
                        dim=-1
                    ).values,
                    "spatial_self_gain_max": reliability_state.diagonal_gain.max(
                        dim=-1
                    ).values,
                    "spatial_offdiagonal_row_l1_max": (
                        reliability_state.offdiagonal_row_l1.max(dim=-1).values
                    ),
                }
            if query_reliability_state is not None:
                active_channels = query_reliability_state.channel_mask
                backbone_aux = {
                    **backbone_aux,
                    "channel_spatial_self_gain": query_reliability_state.diagonal_gain,
                    "spatial_offdiagonal_row_l1": (
                        query_reliability_state.offdiagonal_row_l1
                    ),
                    "channel_query_noise_logit": (
                        query_reliability_state.query_noise_logit
                    ),
                    "channel_query_available_count": (
                        query_reliability_state.query_available.sum(dim=-1)
                    ),
                    "spatial_operator_frobenius_norm": torch.linalg.vector_norm(
                        query_reliability_state.operator_delta.float().flatten(
                            start_dim=1
                        ),
                        dim=-1,
                    ),
                    "query_spatial_operator_frobenius_norm": torch.linalg.vector_norm(
                        query_reliability_state.operator_delta.float().flatten(
                            start_dim=1
                        ),
                        dim=-1,
                    ),
                    "spatial_self_gain_min": (
                        _masked_channel_extreme(
                            query_reliability_state.diagonal_gain,
                            active_channels,
                            reduction="min",
                        )
                    ),
                    "spatial_self_gain_max": (
                        _masked_channel_extreme(
                            query_reliability_state.diagonal_gain,
                            active_channels,
                            reduction="max",
                        )
                    ),
                    "spatial_offdiagonal_row_l1_max": (
                        _masked_channel_extreme(
                            query_reliability_state.offdiagonal_row_l1,
                            active_channels,
                            reduction="max",
                        )
                    ),
                    "query_reliability_identity_intervention": torch.full(
                        (x.shape[0],),
                        bool(query_reliability_identity_intervention),
                        dtype=torch.bool,
                        device=x.device,
                    ),
                    "query_reliability_wrong_query_intervention": torch.full(
                        (x.shape[0],),
                        bool(wrong_query_intervention),
                        dtype=torch.bool,
                        device=x.device,
                    ),
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


def _masked_channel_extreme(
    values: torch.Tensor,
    channel_mask: torch.Tensor,
    *,
    reduction: str,
) -> torch.Tensor:
    if values.shape != channel_mask.shape or values.ndim != 2:
        raise ValueError("Masked channel summary requires matching [batch,channel] tensors.")
    if not channel_mask.any(dim=-1).all():
        raise ValueError("Masked channel summary requires at least one active channel.")
    if reduction == "min":
        fill = torch.inf
        return values.masked_fill(~channel_mask, fill).min(dim=-1).values
    if reduction == "max":
        fill = -torch.inf
        return values.masked_fill(~channel_mask, fill).max(dim=-1).values
    raise ValueError(f"Unknown masked channel reduction: {reduction!r}.")


def _validate_wrong_query_intervention(
    *,
    target_x: torch.Tensor,
    target_cond: dict[str, torch.Tensor],
    donor_x: torch.Tensor,
    donor_cond: dict[str, torch.Tensor],
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Validate an explicit same-checkpoint donor-Q mechanism intervention."""

    if donor_x.shape != target_x.shape:
        raise ValueError("Wrong-query donor x must exactly match the target batch shape.")
    if donor_x.device != target_x.device or donor_x.dtype != target_x.dtype:
        raise ValueError("Wrong-query donor x must share target device and dtype.")
    required = (
        "channel_ids",
        "channel_mask",
        "channel_query_qc",
        "channel_query_qc_missing",
        "sfreq_processed_float",
    )
    for field in required:
        target = target_cond.get(field)
        donor = donor_cond.get(field)
        if not torch.is_tensor(target) or not torch.is_tensor(donor):
            raise ValueError(f"Wrong-query intervention requires tensor field {field}.")
        if target.shape != donor.shape or target.device != donor.device:
            raise ValueError(f"Wrong-query donor field {field} differs in shape or device.")
    for field in ("channel_ids", "channel_mask", "sfreq_processed_float"):
        if not torch.equal(target_cond[field], donor_cond[field]):
            raise ValueError(
                "Wrong-query donors must retain the target channel layout, mask, and "
                f"sample rate; field {field} differs."
            )
    return donor_x, donor_cond


def _development_control_condition_view(
    cond: dict[str, torch.Tensor], development_control: str
) -> dict[str, torch.Tensor]:
    if development_control not in {"metadata_only", "missingness_only"}:
        return cond
    out = {key: value.clone() if torch.is_tensor(value) else value for key, value in cond.items()}
    out["query_qc"].zero_()
    out["query_qc_missing"].fill_(True)
    out["channel_query_qc"].zero_()
    out["channel_query_qc_missing"].fill_(True)
    if development_control == "missingness_only":
        out["external_continuous"].zero_()
        out["channel_impedance"].zero_()
        for field in PROTOCOL_V04_EXTERNAL_CATEGORICAL_FIELDS:
            out[field] = out[field].ne(0).long()
    return out
