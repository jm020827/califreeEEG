from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal

import torch

from cfeg.constants import METADATA_CONTRACT_V04_DEV
from cfeg.models.query_reliability_conditioning import (
    extract_query_reliability_features_v3,
)

WEARABLE_CALIBRATION_CHANNEL_NAMES = (
    "POz",
    "PO3",
    "PO4",
    "PO5",
    "PO6",
    "Oz",
    "O1",
    "O2",
)
WEARABLE_CALIBRATION_CHANNEL_IDS = (56, 55, 57, 54, 58, 62, 61, 63)
QUERY_CHANNEL_FEATURE_NAMES = (
    "pre_zscore_scale",
    "pre_zscore_scale_available",
    "log_band_variance",
    "log_band_difference_energy",
    "mean_absolute_channel_correlation",
    "max_absolute_channel_correlation",
    "spectral_concentration",
)
WEARABLE_Q_FEATURE_NAMES = tuple(
    f"{channel}_{feature}"
    for channel in WEARABLE_CALIBRATION_CHANNEL_NAMES
    for feature in QUERY_CHANNEL_FEATURE_NAMES
)
WEARABLE_METADATA_FEATURE_NAMES = (
    "interface_dry",
    "interface_wet",
    *(f"{channel}_impedance_log_normalized" for channel in WEARABLE_CALIBRATION_CHANNEL_NAMES),
    *(f"{channel}_impedance_available" for channel in WEARABLE_CALIBRATION_CHANNEL_NAMES),
)
WEARABLE_Q_FEATURE_SCHEMA = "wearable_q_8_channels_x_7_v1"
WEARABLE_METADATA_FEATURE_SCHEMA = "wearable_interface2_impedance8_availability8_v1"
MetadataFeatureAblation = Literal["full", "interface_only", "impedance_only"]


@dataclass(frozen=True)
class MetadataCalibrationFeatures:
    """Leakage-minimal tensors accepted by the metadata calibration prior."""

    q_features: torch.Tensor
    metadata_values: torch.Tensor
    metadata_missing: torch.Tensor


def build_wearable_metadata_calibration_features(
    x: torch.Tensor,
    cond: Mapping[str, Any],
    *,
    electrode_types: Sequence[str | None],
    target_sfreq: float = 200.0,
) -> MetadataCalibrationFeatures:
    """Build the frozen 56-d Q and 18-d pre-query M representations.

    The function intentionally has no label, class, participant, session, file,
    or sample-ID argument.  ``electrode_types`` must be the observed acquisition
    interface (or an explicitly constructed intervention) aligned to each EEG
    row; no categorical IDs from the generic condition encoder are inspected.
    """

    q_features = build_wearable_q_features(x, cond, target_sfreq=target_sfreq)
    metadata_values, metadata_missing = build_wearable_metadata_features(
        cond,
        electrode_types=electrode_types,
    )
    if metadata_values.shape[0] != q_features.shape[0]:
        raise ValueError("Q and metadata rows must describe the same batch.")
    return MetadataCalibrationFeatures(
        q_features=q_features,
        metadata_values=metadata_values,
        metadata_missing=metadata_missing,
    )


def build_wearable_q_features(
    x: torch.Tensor,
    cond: Mapping[str, Any],
    *,
    target_sfreq: float = 200.0,
) -> torch.Tensor:
    """Build Q without reading any external acquisition-context field."""

    if x.ndim != 3 or not x.is_floating_point():
        raise ValueError("x must be a floating EEG tensor with shape [batch,channels,time].")
    batch, channels, time = x.shape
    if time != 400:
        raise ValueError("Wearable calibration Q requires exactly 400 samples (2 s at 200 Hz).")
    _validate_contract_version(cond)
    channel_ids = _require_tensor(cond, "channel_ids", shape=(batch, channels), device=x.device)
    channel_mask = _require_tensor(
        cond, "channel_mask", shape=(batch, channels), device=x.device
    ).bool()
    query_qc = _require_tensor(
        cond, "channel_query_qc", shape=(batch, channels), device=x.device
    )
    query_qc_missing = _require_tensor(
        cond, "channel_query_qc_missing", shape=(batch, channels), device=x.device
    ).bool()
    sfreq = _require_tensor(cond, "sfreq_processed_float", shape=(batch,), device=x.device)

    expected_sfreq = torch.full_like(sfreq.float(), float(target_sfreq))
    if not torch.allclose(sfreq.float(), expected_sfreq, rtol=0.0, atol=1e-6):
        raise ValueError("Processed sample rate differs from the frozen Q feature contract.")
    if not torch.isfinite(x.masked_select(channel_mask.unsqueeze(-1))).all():
        raise ValueError("Active EEG samples must be finite.")
    query_qc_available = channel_mask & ~query_qc_missing
    if not torch.isfinite(query_qc.masked_select(query_qc_available)).all():
        raise ValueError("Available transported query-QC values must be finite.")

    positions = _official_channel_positions(channel_ids, channel_mask)
    all_q = extract_query_reliability_features_v3(
        x.float(),
        channel_mask=channel_mask,
        channel_query_qc=query_qc,
        channel_query_missing=query_qc_missing,
        target_sfreq=float(target_sfreq),
        min_frequency_hz=6.0,
        max_frequency_hz=60.0,
    )
    selected_q = torch.gather(
        all_q,
        dim=1,
        index=positions.unsqueeze(-1).expand(-1, -1, len(QUERY_CHANNEL_FEATURE_NAMES)),
    )
    q_features = selected_q.reshape(batch, len(WEARABLE_Q_FEATURE_NAMES))
    if not torch.isfinite(q_features).all():
        raise ValueError("Constructed Q features must be finite.")
    return q_features.float()


def build_wearable_metadata_features(
    cond: Mapping[str, Any],
    *,
    electrode_types: Sequence[str | None],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build M without reading EEG, Q, labels, or generic categorical IDs."""

    _validate_contract_version(cond)
    channel_ids = cond.get("channel_ids")
    if not torch.is_tensor(channel_ids) or channel_ids.ndim != 2:
        raise ValueError("cond['channel_ids'] must be a rank-two tensor.")
    batch, channels = channel_ids.shape
    if len(electrode_types) != batch:
        raise ValueError("electrode_types must contain exactly one value per metadata row.")
    device = channel_ids.device
    channel_mask = _require_tensor(
        cond, "channel_mask", shape=(batch, channels), device=device
    ).bool()
    impedance = _require_tensor(
        cond, "channel_impedance", shape=(batch, channels), device=device
    )
    impedance_missing = _require_tensor(
        cond, "channel_impedance_missing", shape=(batch, channels), device=device
    ).bool()
    positions = _official_channel_positions(channel_ids, channel_mask)

    selected_impedance = torch.gather(impedance, dim=1, index=positions)
    selected_impedance_missing = torch.gather(impedance_missing, dim=1, index=positions)
    selected_impedance_available = ~selected_impedance_missing
    available_impedance = selected_impedance.masked_select(selected_impedance_available)
    if not torch.isfinite(available_impedance).all():
        raise ValueError("Available normalized impedance values must be finite.")
    if available_impedance.numel() and (
        bool((available_impedance < 0.0).any()) or bool((available_impedance > 2.0).any())
    ):
        raise ValueError("Normalized impedance values must lie in the frozen [0,2] range.")
    safe_impedance = torch.where(
        selected_impedance_available,
        selected_impedance.float(),
        torch.zeros((), dtype=torch.float32, device=device),
    )

    interface_values = torch.zeros((batch, 2), dtype=torch.float32, device=device)
    interface_missing = torch.zeros((batch, 2), dtype=torch.bool, device=device)
    for row, raw_value in enumerate(electrode_types):
        value = "unknown" if raw_value is None else str(raw_value).strip().lower()
        if value == "dry":
            interface_values[row, 0] = 1.0
        elif value == "wet":
            interface_values[row, 1] = 1.0
        elif value == "unknown":
            interface_missing[row] = True
        else:
            raise ValueError(f"Unsupported wearable electrode interface {raw_value!r}.")

    impedance_availability = selected_impedance_available.to(dtype=torch.float32)
    # Availability is itself an observed pre-query context feature.  An explicit
    # all-missing intervention must null this feature too via
    # ``with_all_metadata_missing`` below.
    availability_missing = torch.zeros_like(selected_impedance_missing)
    metadata_values = torch.cat(
        [interface_values, safe_impedance, impedance_availability], dim=-1
    )
    metadata_missing = torch.cat(
        [interface_missing, selected_impedance_missing, availability_missing], dim=-1
    )
    row_all_missing = cond.get("metadata_all_missing_rows")
    if row_all_missing is not None:
        if (
            not torch.is_tensor(row_all_missing)
            or row_all_missing.shape != (batch,)
            or row_all_missing.device != device
            or row_all_missing.dtype != torch.bool
        ):
            raise ValueError(
                "cond['metadata_all_missing_rows'] must be one aligned boolean tensor."
            )
        metadata_values = torch.where(
            row_all_missing.unsqueeze(1),
            torch.zeros((), dtype=torch.float32, device=device),
            metadata_values,
        )
        metadata_missing = metadata_missing | row_all_missing.unsqueeze(1)
    if metadata_values.shape[1] != len(WEARABLE_METADATA_FEATURE_NAMES):
        raise RuntimeError("Metadata feature schema width drifted from its declared names.")
    return metadata_values, metadata_missing


def with_all_metadata_missing(
    features: MetadataCalibrationFeatures,
) -> MetadataCalibrationFeatures:
    """Create the exact-null mechanism intervention without changing Q."""

    return MetadataCalibrationFeatures(
        q_features=features.q_features,
        metadata_values=torch.zeros_like(features.metadata_values),
        metadata_missing=torch.ones_like(features.metadata_missing, dtype=torch.bool),
    )


def apply_metadata_feature_ablation(
    values: torch.Tensor,
    missing: torch.Tensor,
    *,
    ablation: MetadataFeatureAblation,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply the frozen interface/impedance diagnostic masks."""

    expected_width = len(WEARABLE_METADATA_FEATURE_NAMES)
    if (
        values.ndim != 2
        or missing.shape != values.shape
        or values.shape[1] != expected_width
    ):
        raise ValueError("Metadata ablation inputs must use the frozen 18-feature schema.")
    if ablation == "full":
        return values, missing.bool()
    output_values = values.clone()
    output_missing = missing.bool().clone()
    if ablation == "interface_only":
        output_values[:, 2:] = 0.0
        output_missing[:, 2:] = True
    elif ablation == "impedance_only":
        output_values[:, :2] = 0.0
        output_missing[:, :2] = True
    else:
        raise ValueError(f"Unknown metadata feature ablation {ablation!r}.")
    return output_values, output_missing


def _official_channel_positions(
    channel_ids: torch.Tensor,
    channel_mask: torch.Tensor,
) -> torch.Tensor:
    if channel_ids.shape != channel_mask.shape or channel_ids.ndim != 2:
        raise ValueError("channel_ids and channel_mask must have the same rank-two shape.")
    if not channel_ids.gt(0).eq(channel_mask).all():
        raise ValueError("Positive canonical channel IDs must exactly match the active mask.")
    official = torch.tensor(
        WEARABLE_CALIBRATION_CHANNEL_IDS,
        dtype=channel_ids.dtype,
        device=channel_ids.device,
    )
    matches = channel_ids.unsqueeze(-1).eq(official.view(1, 1, -1))
    counts = matches.sum(dim=1)
    if not counts.eq(1).all():
        raise ValueError("Each official wearable canonical channel ID must occur exactly once.")
    positions = matches.to(dtype=torch.int64).argmax(dim=1)
    if not torch.gather(channel_mask, dim=1, index=positions).all():
        raise ValueError("Every official wearable channel must be active.")
    if not channel_mask.sum(dim=1).eq(len(WEARABLE_CALIBRATION_CHANNEL_IDS)).all():
        raise ValueError("Wearable calibration input must have exactly the official eight channels.")
    return positions


def _validate_contract_version(cond: Mapping[str, Any]) -> None:
    if cond.get("metadata_contract_version") != METADATA_CONTRACT_V04_DEV:
        raise ValueError("Wearable calibration features require metadata contract '0.4-dev'.")


def _require_tensor(
    cond: Mapping[str, Any],
    name: str,
    *,
    shape: tuple[int, ...],
    device: torch.device,
) -> torch.Tensor:
    value = cond.get(name)
    if not torch.is_tensor(value) or tuple(value.shape) != shape:
        raise ValueError(f"cond[{name!r}] must be a tensor with shape {shape}.")
    if value.device != device:
        raise ValueError(f"cond[{name!r}] must be on the same device as x.")
    return value
