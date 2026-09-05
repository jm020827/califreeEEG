from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class QueryCorruptionCell:
    cell_id: str
    family: str
    selected_fraction: float
    level: float


# All cells are label-, class-frequency-, phase-, subject-history-, and
# batch-statistic-free.  They are intentionally small and fixed: adding a cell
# after seeing outcomes is a new candidate, not an override.
QUERY_RELIABILITY_CORRUPTION_CELLS_V1: dict[str, QueryCorruptionCell] = {
    "zero_0125": QueryCorruptionCell("zero_0125", "channel_zeroing", 0.125, 0.0),
    "zero_0250": QueryCorruptionCell("zero_0250", "channel_zeroing", 0.250, 0.0),
    "white_snr_0": QueryCorruptionCell("white_snr_0", "white_noise", 0.250, 0.0),
    "white_snr_m5": QueryCorruptionCell("white_snr_m5", "white_noise", 0.250, -5.0),
    "line50_snr_0": QueryCorruptionCell("line50_snr_0", "line_noise_50hz", 0.250, 0.0),
    "line50_snr_m5": QueryCorruptionCell("line50_snr_m5", "line_noise_50hz", 0.250, -5.0),
}

QUERY_RELIABILITY_CORRUPTION_MIN_HZ = 6.0
QUERY_RELIABILITY_CORRUPTION_MAX_HZ = 60.0


def apply_query_reliability_corruption(
    x: torch.Tensor,
    cond: dict,
    *,
    sample_ids: list[str],
    cell_id: str,
    seed: int,
    draw_index: int = 0,
) -> tuple[torch.Tensor, dict, torch.Tensor]:
    """Apply one deterministic, query-QC-aware channel corruption cell.

    Randomness is keyed independently for each sample.  Results therefore do
    not depend on batch order or batch partners.  The pre-z-score channel-scale
    QC is transported through the synthetic corruption, and affected channels
    are standardized again to mirror the stored preprocessing contract.
    """

    if cell_id not in QUERY_RELIABILITY_CORRUPTION_CELLS_V1:
        raise ValueError(
            f"Unknown query-reliability corruption cell {cell_id!r}; expected one of "
            f"{sorted(QUERY_RELIABILITY_CORRUPTION_CELLS_V1)}."
        )
    if x.ndim != 3:
        raise ValueError(f"x must have shape [batch,channels,time], got {tuple(x.shape)}.")
    if len(sample_ids) != x.shape[0]:
        raise ValueError("sample_ids must contain exactly one stable ID per EEG query.")
    channel_mask = cond.get("channel_mask")
    if not torch.is_tensor(channel_mask) or channel_mask.shape != x.shape[:2]:
        raise ValueError("cond.channel_mask must match the first two EEG dimensions.")
    if x.shape[-1] < 2:
        raise ValueError("Query corruption requires at least two temporal samples.")

    cell = QUERY_RELIABILITY_CORRUPTION_CELLS_V1[cell_id]
    out_x = x.clone()
    out_cond = _clone_cond(cond)
    affected = torch.zeros(x.shape[:2], dtype=torch.bool, device=x.device)

    for batch_index, sample_id in enumerate(sample_ids):
        active = torch.nonzero(channel_mask[batch_index].bool(), as_tuple=False).flatten()
        if not len(active):
            continue
        generator = torch.Generator(device="cpu")
        # A family-level key makes severity pairs use the same channel order,
        # white-noise realization, and line phase.  Zeroing fractions are
        # nested prefixes of the same channel permutation.
        generator.manual_seed(
            _sample_seed(
                seed=seed,
                draw_index=draw_index,
                sample_id=str(sample_id),
                cell_id=f"corruption-family:{cell.family}",
            )
        )
        n_selected = max(1, math.ceil(len(active) * cell.selected_fraction))
        selected_local = torch.randperm(len(active), generator=generator)[:n_selected]
        selected = active.detach().cpu()[selected_local].to(device=x.device)
        affected[batch_index, selected] = True

        working = out_x[batch_index, selected].float()
        sfreq = float(out_cond["sfreq_processed_float"][batch_index].item())
        if cell.family == "channel_zeroing":
            corrupted = torch.zeros_like(working)
        elif cell.family == "white_noise":
            noise = torch.randn(
                working.shape,
                generator=generator,
                dtype=torch.float32,
                device="cpu",
            ).to(device=x.device)
            noise = _band_limit(
                noise,
                sfreq=sfreq,
                min_frequency_hz=QUERY_RELIABILITY_CORRUPTION_MIN_HZ,
                max_frequency_hz=QUERY_RELIABILITY_CORRUPTION_MAX_HZ,
            )
            band_signal = _band_limit(
                working,
                sfreq=sfreq,
                min_frequency_hz=QUERY_RELIABILITY_CORRUPTION_MIN_HZ,
                max_frequency_hz=QUERY_RELIABILITY_CORRUPTION_MAX_HZ,
            )
            corrupted = working + _scale_noise_to_snr(band_signal, noise, cell.level)
        elif cell.family == "line_noise_50hz":
            if sfreq <= 100.0:
                raise ValueError("50-Hz line corruption requires sfreq > 100 Hz.")
            phase = (
                2.0
                * math.pi
                * torch.rand((len(selected), 1), generator=generator, dtype=torch.float32).to(
                    device=x.device
                )
            )
            time = (
                torch.arange(x.shape[-1], device=x.device, dtype=torch.float32).unsqueeze(0) / sfreq
            )
            line = torch.sin(2.0 * math.pi * 50.0 * time + phase)
            band_signal = _band_limit(
                working,
                sfreq=sfreq,
                min_frequency_hz=QUERY_RELIABILITY_CORRUPTION_MIN_HZ,
                max_frequency_hz=QUERY_RELIABILITY_CORRUPTION_MAX_HZ,
            )
            corrupted = working + _scale_noise_to_snr(band_signal, line, cell.level)
        else:  # pragma: no cover - constant registry is validated above
            raise RuntimeError(f"Unhandled corruption family {cell.family!r}.")

        original_std = working.std(dim=-1, unbiased=False).clamp_min(1e-6)
        relative_std = corrupted.std(dim=-1, unbiased=False) / original_std
        _update_channel_scale_qc(
            out_cond,
            batch_index=batch_index,
            selected=selected,
            relative_std=relative_std,
        )
        centered = corrupted - corrupted.mean(dim=-1, keepdim=True)
        scale = centered.std(dim=-1, unbiased=False, keepdim=True)
        standardized = torch.where(
            scale > 1e-6,
            centered / scale.clamp_min(1e-6),
            torch.zeros_like(centered),
        )
        out_x[batch_index, selected] = standardized.to(dtype=x.dtype)

    _update_query_scale_summary(out_cond)
    return out_x, out_cond, affected


def apply_query_reliability_training_mixture(
    x: torch.Tensor,
    cond: dict,
    *,
    sample_ids: list[str],
    cell_ids: list[str],
    seed: int,
    draw_index: int,
    clean_probability: float = 0.25,
) -> tuple[torch.Tensor, dict, torch.Tensor]:
    """Assign one preregistered corruption cell to each training query."""

    if not cell_ids:
        raise ValueError("The query-reliability training mixture has no corruption cells.")
    unknown = sorted(set(cell_ids) - set(QUERY_RELIABILITY_CORRUPTION_CELLS_V1))
    if unknown:
        raise ValueError(f"Unknown query-reliability training cells: {unknown}.")
    if len(set(cell_ids)) != len(cell_ids):
        raise ValueError("Query-reliability training cells must be unique.")
    if not 0.0 <= clean_probability < 1.0:
        raise ValueError("clean_probability must be within [0,1).")

    out_x = x.clone()
    out_cond = _clone_cond(cond)
    affected = torch.zeros(x.shape[:2], dtype=torch.bool, device=x.device)
    for batch_index, sample_id in enumerate(sample_ids):
        selector = _sample_seed(
            seed=seed,
            draw_index=draw_index,
            sample_id=str(sample_id),
            cell_id="training-mixture-v1",
        )
        if selector / float(2**63 - 1) < clean_probability:
            continue
        cell_id = cell_ids[selector % len(cell_ids)]
        single_cond = _slice_cond(out_cond, batch_index)
        single_x, single_cond, single_affected = apply_query_reliability_corruption(
            out_x[batch_index : batch_index + 1],
            single_cond,
            sample_ids=[str(sample_id)],
            cell_id=cell_id,
            seed=seed,
            draw_index=draw_index,
        )
        out_x[batch_index : batch_index + 1] = single_x
        affected[batch_index : batch_index + 1] = single_affected
        _assign_cond_slice(out_cond, single_cond, batch_index)
    return out_x, out_cond, affected


def _scale_noise_to_snr(signal: torch.Tensor, noise: torch.Tensor, snr_db: float) -> torch.Tensor:
    signal_rms = signal.square().mean(dim=-1, keepdim=True).sqrt().clamp_min(1e-6)
    noise_rms = noise.square().mean(dim=-1, keepdim=True).sqrt().clamp_min(1e-6)
    desired_noise_rms = signal_rms * (10.0 ** (-float(snr_db) / 20.0))
    return noise * (desired_noise_rms / noise_rms)


def _band_limit(
    signal: torch.Tensor,
    *,
    sfreq: float,
    min_frequency_hz: float,
    max_frequency_hz: float,
) -> torch.Tensor:
    if signal.ndim != 2:
        raise ValueError("Band limiting expects [channels,time].")
    if not 0.0 <= min_frequency_hz < max_frequency_hz <= sfreq / 2.0:
        raise ValueError("Invalid corruption band relative to the sample rate.")
    frequencies = torch.fft.rfftfreq(signal.shape[-1], d=1.0 / float(sfreq), device=signal.device)
    keep = (frequencies >= float(min_frequency_hz)) & (frequencies <= float(max_frequency_hz))
    spectrum = torch.fft.rfft(signal.float(), dim=-1, norm="ortho")
    filtered = torch.zeros_like(spectrum)
    filtered[..., keep] = spectrum[..., keep]
    return torch.fft.irfft(filtered, n=signal.shape[-1], dim=-1, norm="ortho")


def _update_channel_scale_qc(
    cond: dict,
    *,
    batch_index: int,
    selected: torch.Tensor,
    relative_std: torch.Tensor,
) -> None:
    if "channel_query_qc" not in cond:
        return
    values = cond["channel_query_qc"][batch_index, selected].float()
    raw_std = torch.expm1(values.clamp(0.0, 2.0) * math.log1p(100.0))
    transported = raw_std * relative_std.to(raw_std.device)
    normalized = torch.log1p(transported.clamp_min(0.0)) / math.log1p(100.0)
    cond["channel_query_qc"][batch_index, selected] = normalized.clamp(0.0, 2.0).to(
        dtype=cond["channel_query_qc"].dtype
    )


def _update_query_scale_summary(cond: dict) -> None:
    if "query_qc" not in cond or "channel_query_qc" not in cond:
        return
    available = cond["channel_mask"].bool() & ~cond["channel_query_qc_missing"].bool()
    for batch_index in range(available.shape[0]):
        values = cond["channel_query_qc"][batch_index, available[batch_index]]
        if not len(values):
            cond["query_qc"][batch_index].zero_()
            cond["query_qc_missing"][batch_index].fill_(True)
            continue
        raw = torch.expm1(values.float().clamp(0.0, 2.0) * math.log1p(100.0))
        ordered = raw.sort().values
        midpoint = len(ordered) // 2
        median_raw = (
            ordered[midpoint]
            if len(ordered) % 2
            else (ordered[midpoint - 1] + ordered[midpoint]) / 2.0
        )
        normalized = torch.log1p(median_raw.clamp_min(0.0)) / math.log1p(100.0)
        cond["query_qc"][batch_index].fill_(
            normalized.clamp(0.0, 2.0).to(dtype=cond["query_qc"].dtype)
        )
        cond["query_qc_missing"][batch_index].fill_(False)


def _sample_seed(*, seed: int, draw_index: int, sample_id: str, cell_id: str) -> int:
    payload = f"{seed}\0{draw_index}\0{sample_id}\0{cell_id}".encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") % (2**63 - 1)


def _clone_cond(cond: dict) -> dict:
    return {key: value.clone() if torch.is_tensor(value) else value for key, value in cond.items()}


def _slice_cond(cond: dict, index: int) -> dict:
    batch = int(cond["channel_mask"].shape[0])
    return {
        key: value[index : index + 1].clone()
        if torch.is_tensor(value) and value.ndim > 0 and value.shape[0] == batch
        else value
        for key, value in cond.items()
    }


def _assign_cond_slice(target: dict, source: dict, index: int) -> None:
    batch = int(target["channel_mask"].shape[0])
    for key, value in source.items():
        if (
            key in target
            and torch.is_tensor(target[key])
            and torch.is_tensor(value)
            and target[key].ndim > 0
            and target[key].shape[0] == batch
            and value.shape[0] == 1
        ):
            target[key][index : index + 1] = value
