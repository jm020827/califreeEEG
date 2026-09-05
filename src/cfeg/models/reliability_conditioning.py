from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from cfeg.constants import PROTOCOL_V04_EXTERNAL_CATEGORICAL_FIELDS

RELIABILITY_SPATIAL_V1 = "reliability_spatial_v1"


@dataclass
class ReliabilityConditionState:
    """Observable state of the query-first spatial reliability operator.

    Scores represent learned channel noise/usefulness, not calibrated signal quality.
    The full operator is retained here for application and focused diagnostics,
    but the training output exposes only compact summaries to avoid accidentally
    serializing a large matrix for every trial.
    """

    operator_delta: torch.Tensor
    query_operator_delta: torch.Tensor
    metadata_operator_delta: torch.Tensor
    diagonal_gain: torch.Tensor
    offdiagonal_row_l1: torch.Tensor
    query_noise_logit: torch.Tensor
    metadata_noise_logit: torch.Tensor
    combined_noise_logit: torch.Tensor
    query_available: torch.Tensor
    metadata_available: torch.Tensor
    metadata_alpha: torch.Tensor

    def apply(self, x: torch.Tensor) -> torch.Tensor:
        expected = (x.shape[0], x.shape[1], x.shape[1]) if x.ndim == 3 else None
        if expected is None or self.operator_delta.shape != expected:
            raise ValueError(
                "Spatial operator/input mismatch: "
                f"operator={tuple(self.operator_delta.shape)}, x={tuple(x.shape)}."
            )
        return x + torch.einsum("bij,bjt->bit", self.operator_delta, x)


class ResidualizedReliabilityOperator(nn.Module):
    """Single-query spatial operator with a zero-initialized metadata residual.

    The primary branch derives per-channel and pairwise reliability evidence
    only from the current EEG window and pre-z-score channel scale QC. External
    metadata forms an additive branch and can contribute only through a bounded
    scalar alpha initialized to zero. No label, stimulus frequency, target
    history, or target-batch statistic is consumed.

    Query and metadata features are deliberately not fused before their spatial
    deltas are formed. This preserves an auditable 2x2 factorial interpretation:
    the metadata-only arm cannot consume query-derived quality features.
    """

    _N_QUERY_FEATURES = 7

    def __init__(
        self,
        *,
        d_model: int,
        vocab_sizes: dict[str, int],
        fields: list[str],
        external_metadata_mode: str,
        hidden_dim: int = 64,
        operator_rank: int = 4,
        n_external_cont_features: int = 2,
        max_operator_norm: float = 0.25,
        query_operator_norm: float = 0.20,
        metadata_operator_norm: float = 0.05,
        metadata_alpha_limit: float = 1.0,
        query_qc_mode: str = "observed",
        metadata_residual_enabled: bool = True,
    ):
        super().__init__()
        if external_metadata_mode not in {"observed", "null"}:
            raise ValueError(
                "external_metadata_mode must be 'observed' or 'null', got "
                f"{external_metadata_mode!r}."
            )
        forbidden = sorted(set(fields) - set(PROTOCOL_V04_EXTERNAL_CATEGORICAL_FIELDS))
        if forbidden:
            raise ValueError(
                "reliability_spatial_v1 contains forbidden categorical fields: "
                f"{forbidden}."
            )
        if hidden_dim <= 0 or operator_rank <= 0:
            raise ValueError("hidden_dim and operator_rank must be positive.")
        if not 0.0 < max_operator_norm < 1.0:
            raise ValueError("max_operator_norm must be in (0,1).")
        if not 0.0 < query_operator_norm <= max_operator_norm:
            raise ValueError("query_operator_norm must be in (0,max_operator_norm].")
        if not 0.0 < metadata_operator_norm <= max_operator_norm:
            raise ValueError("metadata_operator_norm must be in (0,max_operator_norm].")
        if metadata_alpha_limit <= 0.0:
            raise ValueError("metadata_alpha_limit must be positive.")
        if (
            query_operator_norm + metadata_alpha_limit * metadata_operator_norm
            > max_operator_norm + 1e-12
        ):
            raise ValueError(
                "query_operator_norm + metadata_alpha_limit * metadata_operator_norm "
                "must not exceed max_operator_norm."
            )
        if query_qc_mode not in {"observed", "null"}:
            raise ValueError(
                f"query_qc_mode must be 'observed' or 'null', got {query_qc_mode!r}."
            )

        self.d_model = int(d_model)
        self.hidden_dim = int(hidden_dim)
        self.operator_rank = int(operator_rank)
        self.fields = list(fields)
        self.external_metadata_mode = str(external_metadata_mode)
        self.n_external_cont_features = int(n_external_cont_features)
        self.max_operator_norm = float(max_operator_norm)
        self.query_operator_norm = float(query_operator_norm)
        self.metadata_operator_norm = float(metadata_operator_norm)
        self.metadata_alpha_limit = float(metadata_alpha_limit)
        self.query_qc_mode = str(query_qc_mode)
        self.metadata_residual_enabled = bool(metadata_residual_enabled)

        self.cat_embeddings = nn.ModuleDict(
            {
                name: nn.Embedding(
                    max(int(vocab_sizes.get(name, 1)), 1),
                    hidden_dim,
                    padding_idx=0,
                )
                for name in self.fields
            }
        )
        self.query_encoder = nn.Sequential(
            nn.Linear(self._N_QUERY_FEATURES, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
        )
        self.query_diagonal_score = nn.Linear(hidden_dim, 1)
        self.query_left_factor = nn.Linear(hidden_dim, operator_rank)
        self.query_right_factor = nn.Linear(hidden_dim, operator_rank)
        nn.init.zeros_(self.query_diagonal_score.weight)
        nn.init.zeros_(self.query_diagonal_score.bias)
        nn.init.zeros_(self.query_left_factor.weight)
        nn.init.zeros_(self.query_left_factor.bias)

        # Per-channel metadata: absolute normalized impedance, within-block
        # relative impedance, mismatch magnitude, and availability. Global
        # impedance summaries contribute value/availability pairs. Categorical
        # context is embedded once and broadcast to every active channel.
        metadata_features = 4 + 2 * self.n_external_cont_features + hidden_dim
        self.metadata_encoder = nn.Sequential(
            nn.Linear(metadata_features, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
        )
        self.metadata_diagonal_score = nn.Linear(hidden_dim, 1)
        self.metadata_left_factor = nn.Linear(hidden_dim, operator_rank)
        self.metadata_right_factor = nn.Linear(hidden_dim, operator_rank)
        self.metadata_alpha_raw = nn.Parameter(torch.zeros(()))

    def forward(
        self,
        x: torch.Tensor,
        cond: dict[str, torch.Tensor],
    ) -> ReliabilityConditionState:
        if x.ndim != 3:
            raise ValueError(f"EEG input must have shape [batch,channels,time], got {tuple(x.shape)}.")
        batch, channels, _time = x.shape
        channel_mask = cond["channel_mask"].bool()
        channel_query_qc = cond["channel_query_qc"]
        channel_query_missing = cond["channel_query_qc_missing"].bool()
        channel_impedance = cond["channel_impedance"]
        channel_impedance_missing = cond["channel_impedance_missing"].bool()
        external_values = cond["external_continuous"]
        external_missing = cond["external_continuous_missing"].bool()
        self._validate_shapes(
            batch=batch,
            channels=channels,
            channel_mask=channel_mask,
            channel_query_qc=channel_query_qc,
            channel_query_missing=channel_query_missing,
            channel_impedance=channel_impedance,
            channel_impedance_missing=channel_impedance_missing,
            external_values=external_values,
            external_missing=external_missing,
        )

        query_features = self._query_features(
            x,
            channel_mask=channel_mask,
            channel_query_qc=channel_query_qc,
            channel_query_missing=channel_query_missing,
        )
        query_hidden = self.query_encoder(query_features)
        query_available = channel_mask & (self.query_qc_mode == "observed")
        query_weight = query_available.to(query_hidden.dtype)
        query_noise_logit = self.query_diagonal_score(query_hidden).squeeze(-1) * query_weight
        query_left = torch.tanh(self.query_left_factor(query_hidden)) * query_weight.unsqueeze(-1)
        query_right = torch.tanh(self.query_right_factor(query_hidden)) * query_weight.unsqueeze(-1)
        query_operator = _symmetric_low_rank(query_left, query_right)
        query_pair_mask = query_available.unsqueeze(1) & query_available.unsqueeze(2)
        query_diagonal = -_masked_center(query_noise_logit, query_available)
        query_operator = (
            query_operator + torch.diag_embed(query_diagonal)
        ).masked_fill(~query_pair_mask, 0.0)
        query_operator_delta = _project_frobenius_norm(
            query_operator,
            maximum=self.query_operator_norm,
        )

        external_is_null = self.external_metadata_mode == "null"
        if external_is_null:
            channel_impedance = torch.zeros_like(channel_impedance)
            channel_impedance_missing = torch.ones_like(channel_impedance_missing)
            external_values = torch.zeros_like(external_values)
            external_missing = torch.ones_like(external_missing)

        categorical_context = x.new_zeros((batch, self.hidden_dim))
        categorical_available = torch.zeros((batch, 1), dtype=torch.bool, device=x.device)
        for name, embedding in self.cat_embeddings.items():
            ids = torch.zeros_like(cond[name]) if external_is_null else cond[name]
            valid = (ids >= 0) & (ids < embedding.num_embeddings)
            ids = torch.where(valid, ids, torch.zeros_like(ids))
            categorical_context = categorical_context + embedding(ids)
            categorical_available |= ids.ne(0).unsqueeze(-1)

        raw_impedance_available = ~channel_impedance_missing
        metadata_only_control = str(cond.get("development_control", "none")) in {
            "metadata_only",
            "missingness_only",
        }
        impedance_available = (
            raw_impedance_available
            if metadata_only_control
            else raw_impedance_available & channel_mask
        )
        impedance_value = channel_impedance.masked_fill(~impedance_available, 0.0)
        impedance_center = _masked_center(impedance_value, impedance_available)
        external_available = ~external_missing
        global_pairs = torch.cat(
            [
                external_values.masked_fill(external_missing, 0.0),
                external_available.to(external_values.dtype),
            ],
            dim=-1,
        ).unsqueeze(1).expand(-1, channels, -1)
        categorical_channels = categorical_context.unsqueeze(1).expand(-1, channels, -1)
        metadata_features = torch.cat(
            [
                impedance_value.unsqueeze(-1),
                impedance_center.unsqueeze(-1),
                impedance_center.abs().unsqueeze(-1),
                impedance_available.to(x.dtype).unsqueeze(-1),
                global_pairs,
                categorical_channels,
            ],
            dim=-1,
        )
        metadata_hidden = self.metadata_encoder(metadata_features)
        any_global = categorical_available | external_available.any(dim=-1, keepdim=True)
        metadata_available = (
            impedance_available | any_global.expand(-1, channels)
        ) & channel_mask & self.metadata_residual_enabled
        metadata_weight = metadata_available.to(metadata_hidden.dtype)
        metadata_noise_logit = (
            self.metadata_diagonal_score(metadata_hidden).squeeze(-1) * metadata_weight
        )
        metadata_left = torch.tanh(self.metadata_left_factor(metadata_hidden))
        metadata_right = torch.tanh(self.metadata_right_factor(metadata_hidden))
        metadata_left = metadata_left * metadata_weight.unsqueeze(-1)
        metadata_right = metadata_right * metadata_weight.unsqueeze(-1)
        metadata_operator = _symmetric_low_rank(metadata_left, metadata_right)
        metadata_diagonal = -_masked_center(metadata_noise_logit, metadata_available)
        metadata_pair_mask = metadata_available.unsqueeze(1) & metadata_available.unsqueeze(2)
        metadata_operator = (
            metadata_operator + torch.diag_embed(metadata_diagonal)
        ).masked_fill(~metadata_pair_mask, 0.0)
        metadata_operator = _project_frobenius_norm(
            metadata_operator,
            maximum=self.metadata_operator_norm,
        )

        metadata_alpha = self.metadata_alpha_limit * torch.tanh(self.metadata_alpha_raw)
        metadata_operator_delta = metadata_alpha * metadata_operator
        combined_noise_logit = query_noise_logit + metadata_alpha * metadata_noise_logit
        pair_mask = channel_mask.unsqueeze(1) & channel_mask.unsqueeze(2)
        operator_delta = (query_operator_delta + metadata_operator_delta).masked_fill(
            ~pair_mask, 0.0
        )
        diagonal_gain = 1.0 + torch.diagonal(operator_delta, dim1=1, dim2=2)
        eye = torch.eye(channels, dtype=torch.bool, device=x.device).unsqueeze(0)
        offdiagonal_delta = operator_delta.masked_fill(eye, 0.0)
        offdiagonal_row_l1 = offdiagonal_delta.abs().sum(dim=-1)

        return ReliabilityConditionState(
            operator_delta=operator_delta,
            query_operator_delta=query_operator_delta,
            metadata_operator_delta=metadata_operator_delta,
            diagonal_gain=diagonal_gain,
            offdiagonal_row_l1=offdiagonal_row_l1,
            query_noise_logit=query_noise_logit,
            metadata_noise_logit=metadata_noise_logit,
            combined_noise_logit=combined_noise_logit,
            query_available=query_available,
            metadata_available=metadata_available,
            metadata_alpha=metadata_alpha,
        )

    def _query_features(
        self,
        x: torch.Tensor,
        *,
        channel_mask: torch.Tensor,
        channel_query_qc: torch.Tensor,
        channel_query_missing: torch.Tensor,
    ) -> torch.Tensor:
        return extract_query_reliability_features(
            x,
            channel_mask=channel_mask,
            channel_query_qc=channel_query_qc,
            channel_query_missing=channel_query_missing,
        )

    def _validate_shapes(
        self,
        *,
        batch: int,
        channels: int,
        channel_mask: torch.Tensor,
        channel_query_qc: torch.Tensor,
        channel_query_missing: torch.Tensor,
        channel_impedance: torch.Tensor,
        channel_impedance_missing: torch.Tensor,
        external_values: torch.Tensor,
        external_missing: torch.Tensor,
    ) -> None:
        expected_channel_shape = (batch, channels)
        for name, value in (
            ("channel_mask", channel_mask),
            ("channel_query_qc", channel_query_qc),
            ("channel_query_qc_missing", channel_query_missing),
            ("channel_impedance", channel_impedance),
            ("channel_impedance_missing", channel_impedance_missing),
        ):
            if value.shape != expected_channel_shape:
                raise ValueError(
                    f"{name} must have shape {expected_channel_shape}, got {tuple(value.shape)}."
                )
        expected_external_shape = (batch, self.n_external_cont_features)
        if external_values.shape != expected_external_shape:
            raise ValueError(
                "external_continuous must have shape "
                f"{expected_external_shape}, got {tuple(external_values.shape)}."
            )
        if external_missing.shape != external_values.shape:
            raise ValueError(
                "external_continuous and external_continuous_missing shapes must match."
            )


def extract_query_reliability_features(
    x: torch.Tensor,
    *,
    channel_mask: torch.Tensor,
    channel_query_qc: torch.Tensor,
    channel_query_missing: torch.Tensor,
) -> torch.Tensor:
    """Compute the exact label-free Q feature allowlist used by the spatial operator."""

    if x.ndim != 3:
        raise ValueError(f"EEG input must have shape [batch,channels,time], got {tuple(x.shape)}.")
    expected = x.shape[:2]
    if any(value.shape != expected for value in (channel_mask, channel_query_qc, channel_query_missing)):
        raise ValueError(f"Q feature masks/values must all have shape {tuple(expected)}.")
    working = x.float()
    channel_mask = channel_mask.bool()
    active = channel_mask.unsqueeze(-1)
    centered = working - working.mean(dim=-1, keepdim=True)
    centered = centered.masked_fill(~active, 0.0)
    variance = centered.square().mean(dim=-1)
    log_variance = torch.log(variance + 1e-6).clamp(-12.0, 12.0)

    differences = torch.diff(centered, dim=-1)
    difference_energy = differences.square().mean(dim=-1)
    log_difference_energy = torch.log(difference_energy + 1e-6).clamp(-12.0, 12.0)

    normalized = centered / torch.sqrt(variance + 1e-6).unsqueeze(-1)
    correlation = torch.matmul(normalized, normalized.transpose(1, 2)) / float(x.shape[-1])
    correlation = correlation.clamp(-1.0, 1.0)
    pair_mask = channel_mask.unsqueeze(1) & channel_mask.unsqueeze(2)
    eye = torch.eye(x.shape[1], dtype=torch.bool, device=x.device).unsqueeze(0)
    offdiagonal_mask = pair_mask & ~eye
    absolute_correlation = correlation.abs()
    mean_abs_correlation = _masked_mean(
        absolute_correlation,
        offdiagonal_mask,
        dim=-1,
    )
    max_abs_correlation = absolute_correlation.masked_fill(
        ~offdiagonal_mask, -1.0
    ).max(dim=-1).values.clamp_min(0.0)

    spectrum = torch.fft.rfft(centered, dim=-1, norm="ortho")
    power = spectrum.abs().square()[..., 1:]
    spectral_concentration = power.amax(dim=-1) / power.sum(dim=-1).clamp_min(1e-8)

    qc_available = (~channel_query_missing.bool()) & channel_mask
    qc_value = channel_query_qc.float().masked_fill(~qc_available, 0.0)
    features = torch.stack(
        [
            qc_value,
            qc_available.to(working.dtype),
            log_variance,
            log_difference_energy,
            mean_abs_correlation,
            max_abs_correlation,
            spectral_concentration,
        ],
        dim=-1,
    )
    features = features.masked_fill(~channel_mask.unsqueeze(-1), 0.0)
    return features.to(dtype=x.dtype)


def _symmetric_low_rank(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    scale = 2.0 * left.shape[-1] ** 0.5
    product = torch.matmul(left, right.transpose(1, 2))
    return (product + product.transpose(1, 2)) / scale


def _project_frobenius_norm(values: torch.Tensor, *, maximum: float) -> torch.Tensor:
    if maximum == 0.0:
        return torch.zeros_like(values)
    working = values.float()
    norm = torch.linalg.vector_norm(working.flatten(start_dim=1), dim=-1)
    scale = torch.minimum(
        torch.ones_like(norm),
        torch.full_like(norm, maximum) / norm.clamp_min(1e-12),
    )
    return (working * scale[:, None, None]).to(dtype=values.dtype)


def _masked_center(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    mean = _masked_mean(values, mask, dim=-1).unsqueeze(-1)
    return torch.where(mask, values - mean, torch.zeros_like(values))


def _masked_mean(values: torch.Tensor, mask: torch.Tensor, *, dim: int) -> torch.Tensor:
    weights = mask.to(values.dtype)
    while weights.ndim < values.ndim:
        weights = weights.unsqueeze(-1)
    numerator = (values * weights).sum(dim=dim)
    denominator = weights.sum(dim=dim).clamp_min(1.0)
    return numerator / denominator
