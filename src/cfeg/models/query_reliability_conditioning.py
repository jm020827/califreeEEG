from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

QUERY_RELIABILITY_SPATIAL_V1 = "query_reliability_spatial_v1"
QUERY_RELIABILITY_FEATURE_SCHEMA_V3 = (
    "query_window_reliability_features_v3_waveform_6_60_qc_scale_6_90"
)


@dataclass
class QueryReliabilityState:
    """Observable state of the metadata-free, single-query spatial operator.

    ``query_noise_logit`` is a learned task-usefulness score, not a calibrated
    signal-quality measurement.  The operator is constant over the time axis of
    one query, so it can mix channels but cannot create a new temporal frequency.
    """

    operator_delta: torch.Tensor
    diagonal_gain: torch.Tensor
    offdiagonal_row_l1: torch.Tensor
    query_noise_logit: torch.Tensor
    query_available: torch.Tensor
    channel_mask: torch.Tensor

    def apply(self, x: torch.Tensor) -> torch.Tensor:
        expected = (x.shape[0], x.shape[1], x.shape[1]) if x.ndim == 3 else None
        if expected is None or self.operator_delta.shape != expected:
            raise ValueError(
                "Spatial operator/input mismatch: "
                f"operator={tuple(self.operator_delta.shape)}, x={tuple(x.shape)}."
            )
        if self.channel_mask.shape != x.shape[:2]:
            raise ValueError(
                "Spatial operator channel mask/input mismatch: "
                f"mask={tuple(self.channel_mask.shape)}, x={tuple(x.shape)}."
            )
        safe_x = torch.where(
            self.channel_mask.unsqueeze(-1), x, torch.zeros((), dtype=x.dtype, device=x.device)
        )
        working_x = safe_x.to(dtype=self.operator_delta.dtype)
        return working_x + torch.einsum("bij,bjt->bit", self.operator_delta, working_x)


class QueryReliabilityOperator(nn.Module):
    """Bounded query-local spatial operator with no external-metadata branch.

    Five waveform-derived features use 6--60 Hz; the other two inputs are the
    transported pre-zscore 6--90 Hz scale QC and its availability flag. No predecessor
    metadata encoder, head, or alpha is instantiated. ``null`` and ``observed``
    have an identical parameter graph and differ only in whether the Q-derived
    delta is applied.
    """

    def __init__(
        self,
        *,
        hidden_dim: int = 64,
        operator_rank: int = 4,
        max_operator_norm: float = 0.20,
        query_qc_mode: str = "observed",
        target_sfreq: float = 200.0,
        min_frequency_hz: float = 6.0,
        max_frequency_hz: float = 60.0,
    ):
        super().__init__()
        if hidden_dim <= 0 or operator_rank <= 0:
            raise ValueError("hidden_dim and operator_rank must be positive.")
        if not 0.0 < max_operator_norm < 1.0:
            raise ValueError("max_operator_norm must be in (0,1).")
        if query_qc_mode not in {"observed", "null"}:
            raise ValueError(f"query_qc_mode must be 'observed' or 'null', got {query_qc_mode!r}.")
        if target_sfreq <= 0.0:
            raise ValueError("target_sfreq must be positive.")
        if not 0.0 <= min_frequency_hz < max_frequency_hz <= target_sfreq / 2.0:
            raise ValueError("Q frequency support must satisfy 0 <= min < max <= Nyquist.")

        self.hidden_dim = int(hidden_dim)
        self.operator_rank = int(operator_rank)
        self.max_operator_norm = float(max_operator_norm)
        self.query_qc_mode = str(query_qc_mode)
        self.target_sfreq = float(target_sfreq)
        self.min_frequency_hz = float(min_frequency_hz)
        self.max_frequency_hz = float(max_frequency_hz)

        self.query_encoder = nn.Sequential(
            nn.Linear(7, hidden_dim),
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

    def forward(
        self,
        x: torch.Tensor,
        cond: dict[str, torch.Tensor],
    ) -> QueryReliabilityState:
        if x.ndim != 3:
            raise ValueError(
                f"EEG input must have shape [batch,channels,time], got {tuple(x.shape)}."
            )
        batch, channels, _time = x.shape
        channel_mask = cond["channel_mask"].bool()
        channel_query_qc = cond["channel_query_qc"]
        channel_query_missing = cond["channel_query_qc_missing"].bool()
        expected = (batch, channels)
        for name, value in (
            ("channel_mask", channel_mask),
            ("channel_query_qc", channel_query_qc),
            ("channel_query_qc_missing", channel_query_missing),
        ):
            if value.shape != expected:
                raise ValueError(f"{name} must have shape {expected}, got {tuple(value.shape)}.")

        if self.query_qc_mode == "null":
            operator_delta = x.new_zeros((batch, channels, channels), dtype=torch.float32)
            return QueryReliabilityState(
                operator_delta=operator_delta,
                diagonal_gain=x.new_ones((batch, channels), dtype=torch.float32),
                offdiagonal_row_l1=x.new_zeros((batch, channels), dtype=torch.float32),
                query_noise_logit=x.new_zeros((batch, channels), dtype=torch.float32),
                query_available=torch.zeros_like(channel_mask),
                channel_mask=channel_mask,
            )

        active_values = x.masked_select(channel_mask.unsqueeze(-1))
        if not torch.isfinite(active_values).all():
            raise ValueError("Observed Q mode requires finite active EEG samples.")
        qc_available = channel_mask & ~channel_query_missing
        if not torch.isfinite(channel_query_qc.masked_select(qc_available)).all():
            raise ValueError("Observed Q mode requires finite available channel QC values.")

        sfreq = cond.get("sfreq_processed_float")
        if not torch.is_tensor(sfreq) or sfreq.shape != (batch,):
            raise ValueError("sfreq_processed_float must have shape [batch].")
        expected_sfreq = torch.full_like(sfreq.float(), self.target_sfreq)
        if not torch.allclose(sfreq.float(), expected_sfreq, rtol=0.0, atol=1e-6):
            raise ValueError("Query-Q frequency support and processed sample rate disagree.")

        query_features = extract_query_reliability_features_v3(
            x,
            channel_mask=channel_mask,
            channel_query_qc=channel_query_qc,
            channel_query_missing=channel_query_missing,
            target_sfreq=self.target_sfreq,
            min_frequency_hz=self.min_frequency_hz,
            max_frequency_hz=self.max_frequency_hz,
        )
        query_hidden = self.query_encoder(query_features)
        query_available = channel_mask
        query_weight = query_available.to(query_hidden.dtype)
        query_noise_logit = self.query_diagonal_score(query_hidden).squeeze(-1) * query_weight
        query_left = torch.tanh(self.query_left_factor(query_hidden)) * query_weight.unsqueeze(-1)
        query_right = torch.tanh(self.query_right_factor(query_hidden)) * query_weight.unsqueeze(-1)
        query_operator = _symmetric_low_rank(query_left, query_right)
        query_pair_mask = query_available.unsqueeze(1) & query_available.unsqueeze(2)
        query_diagonal = -_masked_center(query_noise_logit, query_available)
        query_operator = (query_operator + torch.diag_embed(query_diagonal)).masked_fill(
            ~query_pair_mask, 0.0
        )
        operator_delta = _project_frobenius_norm(
            query_operator,
            maximum=self.max_operator_norm,
        )

        diagonal_gain = 1.0 + torch.diagonal(operator_delta, dim1=1, dim2=2)
        eye = torch.eye(channels, dtype=torch.bool, device=x.device).unsqueeze(0)
        offdiagonal_row_l1 = operator_delta.masked_fill(eye, 0.0).abs().sum(dim=-1)
        return QueryReliabilityState(
            operator_delta=operator_delta,
            diagonal_gain=diagonal_gain,
            offdiagonal_row_l1=offdiagonal_row_l1,
            query_noise_logit=query_noise_logit,
            query_available=query_available,
            channel_mask=channel_mask,
        )


def extract_query_reliability_features_v3(
    x: torch.Tensor,
    *,
    channel_mask: torch.Tensor,
    channel_query_qc: torch.Tensor,
    channel_query_missing: torch.Tensor,
    target_sfreq: float,
    min_frequency_hz: float,
    max_frequency_hz: float,
) -> torch.Tensor:
    """Compute five 6--60 Hz features plus 6--90 Hz scale QC and availability."""

    if x.ndim != 3:
        raise ValueError(f"EEG input must have shape [batch,channels,time], got {tuple(x.shape)}.")
    expected = x.shape[:2]
    if any(
        value.shape != expected for value in (channel_mask, channel_query_qc, channel_query_missing)
    ):
        raise ValueError(f"Q feature masks/values must all have shape {tuple(expected)}.")
    source = x.float()
    frequencies = torch.fft.rfftfreq(x.shape[-1], d=1.0 / float(target_sfreq), device=x.device)
    frequency_mask = (frequencies >= float(min_frequency_hz)) & (
        frequencies <= float(max_frequency_hz)
    )
    if not bool(frequency_mask.any()):
        raise ValueError("Q feature frequency interval contains no FFT bins.")
    source_spectrum = torch.fft.rfft(source, dim=-1, norm="ortho")
    source_band_power = source_spectrum[..., frequency_mask].abs().square().sum(dim=-1)
    # The gate must itself be band-only.  Comparing with all-frequency power
    # would let a large 60--90-Hz component change a nominally 6--60-Hz Q
    # feature even when the in-band waveform is identical.
    minimum_band_power = torch.finfo(source.dtype).eps * float(x.shape[-1])
    has_band_signal = source_band_power > minimum_band_power
    filtered_spectrum = torch.zeros_like(source_spectrum)
    filtered_spectrum[..., frequency_mask] = source_spectrum[..., frequency_mask]
    filtered_spectrum = filtered_spectrum * has_band_signal.unsqueeze(-1)
    working = torch.fft.irfft(filtered_spectrum, n=x.shape[-1], dim=-1, norm="ortho")
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
    mean_abs_correlation = _masked_mean(absolute_correlation, offdiagonal_mask, dim=-1)
    max_abs_correlation = (
        absolute_correlation.masked_fill(~offdiagonal_mask, -1.0).max(dim=-1).values.clamp_min(0.0)
    )

    spectrum = torch.fft.rfft(centered, dim=-1, norm="ortho")
    power = spectrum[..., frequency_mask].abs().square()
    band_power = power.sum(dim=-1)
    raw_concentration = power.amax(dim=-1) / band_power.clamp_min(1e-8)
    spectral_concentration = torch.where(
        band_power > minimum_band_power,
        raw_concentration,
        torch.zeros_like(raw_concentration),
    )

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
    working = values.float()
    norm = torch.linalg.vector_norm(working.flatten(start_dim=1), dim=-1)
    limit = torch.full_like(norm, maximum)
    # A negligible float32 guard keeps a boundary-valued matrix below the hard
    # limit when AMP produced the unprojected factors in fp16 or bfloat16.
    safe_limit = limit - 8.0 * torch.finfo(working.dtype).eps * limit
    scale = torch.minimum(
        torch.ones_like(norm),
        safe_limit / norm.clamp_min(1e-12),
    )
    return working * scale[:, None, None]


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
