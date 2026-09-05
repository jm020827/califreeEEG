from __future__ import annotations

import copy
import math

import torch

from cfeg.data.query_reliability_corruption import (
    QUERY_RELIABILITY_CORRUPTION_CELLS_V1,
    _band_limit,
    _scale_noise_to_snr,
    apply_query_reliability_corruption,
    apply_query_reliability_training_mixture,
)


def _condition(batch: int = 2, channels: int = 8) -> dict:
    return {
        "channel_mask": torch.ones(batch, channels, dtype=torch.bool),
        "sfreq_processed_float": torch.full((batch,), 200.0),
        "channel_query_qc": torch.full((batch, channels), 0.5),
        "channel_query_qc_missing": torch.zeros(batch, channels, dtype=torch.bool),
        "query_qc": torch.full((batch, 1), 0.5),
        "query_qc_missing": torch.zeros(batch, 1, dtype=torch.bool),
        "external_continuous": torch.arange(batch * 2, dtype=torch.float32).reshape(batch, 2),
        "channel_impedance": torch.arange(batch * channels, dtype=torch.float32).reshape(
            batch, channels
        ),
    }


def test_each_corruption_cell_is_finite_label_free_and_qc_aware() -> None:
    torch.manual_seed(3)
    x = torch.randn(2, 8, 400)
    cond = _condition()
    for cell_id in QUERY_RELIABILITY_CORRUPTION_CELLS_V1:
        out_x, out_cond, affected = apply_query_reliability_corruption(
            x,
            cond,
            sample_ids=["a", "b"],
            cell_id=cell_id,
            seed=19,
        )
        assert torch.isfinite(out_x).all()
        assert affected.sum(dim=1).eq(1 if cell_id == "zero_0125" else 2).all()
        assert torch.equal(out_cond["external_continuous"], cond["external_continuous"])
        assert torch.equal(out_cond["channel_impedance"], cond["channel_impedance"])
        assert torch.equal(out_cond["channel_mask"], cond["channel_mask"])
        assert not torch.equal(out_cond["channel_query_qc"], cond["channel_query_qc"])
        unaffected = ~affected
        torch.testing.assert_close(out_x[unaffected], x[unaffected], rtol=0.0, atol=0.0)
        if not cell_id.startswith("zero"):
            affected_std = out_x.std(dim=-1, unbiased=False)[affected]
            torch.testing.assert_close(
                affected_std, torch.ones_like(affected_std), rtol=1e-5, atol=1e-5
            )


def test_corruption_is_independent_of_batch_order_and_partner() -> None:
    torch.manual_seed(5)
    x = torch.randn(2, 8, 400)
    cond = _condition()
    forward = apply_query_reliability_corruption(
        x,
        cond,
        sample_ids=["a", "b"],
        cell_id="white_snr_m5",
        seed=23,
        draw_index=4,
    )
    order = torch.tensor([1, 0])
    permuted_cond = {
        key: value[order].clone()
        if torch.is_tensor(value) and value.ndim and value.shape[0] == 2
        else copy.deepcopy(value)
        for key, value in cond.items()
    }
    reverse = apply_query_reliability_corruption(
        x[order],
        permuted_cond,
        sample_ids=["b", "a"],
        cell_id="white_snr_m5",
        seed=23,
        draw_index=4,
    )
    inverse = torch.tensor([1, 0])
    torch.testing.assert_close(forward[0], reverse[0][inverse], rtol=0.0, atol=0.0)
    torch.testing.assert_close(
        forward[1]["channel_query_qc"],
        reverse[1]["channel_query_qc"][inverse],
        rtol=0.0,
        atol=0.0,
    )
    assert torch.equal(forward[2], reverse[2][inverse])


def test_training_mixture_is_reproducible_and_rejects_recipe_drift() -> None:
    x = torch.randn(2, 8, 400)
    cond = _condition()
    cells = list(QUERY_RELIABILITY_CORRUPTION_CELLS_V1)
    first = apply_query_reliability_training_mixture(
        x, cond, sample_ids=["a", "b"], cell_ids=cells, seed=29, draw_index=7
    )
    second = apply_query_reliability_training_mixture(
        x, cond, sample_ids=["a", "b"], cell_ids=cells, seed=29, draw_index=7
    )
    torch.testing.assert_close(first[0], second[0], rtol=0.0, atol=0.0)
    assert torch.equal(first[2], second[2])

    try:
        apply_query_reliability_training_mixture(
            x,
            cond,
            sample_ids=["a", "b"],
            cell_ids=["not_registered"],
            seed=29,
            draw_index=7,
        )
    except ValueError as exc:
        assert "Unknown" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("Unknown corruption cell was accepted.")


def test_noise_scaling_hits_requested_pre_restandardization_snr() -> None:
    generator = torch.Generator().manual_seed(41)
    signal = torch.randn(3, 400, generator=generator)
    noise = torch.randn(3, 400, generator=generator)
    for requested in (0.0, -5.0):
        scaled = _scale_noise_to_snr(signal, noise, requested)
        signal_rms = signal.square().mean(dim=-1).sqrt()
        noise_rms = scaled.square().mean(dim=-1).sqrt()
        observed = 20.0 * torch.log10(signal_rms / noise_rms)
        torch.testing.assert_close(
            observed,
            torch.full_like(observed, requested),
            rtol=1e-5,
            atol=1e-5,
        )


def test_white_noise_recipe_uses_the_declared_six_to_sixty_hz_snr_band() -> None:
    time = torch.arange(400, dtype=torch.float32) / 200.0
    signal = (
        torch.sin(2.0 * torch.pi * 10.0 * time) + 4.0 * torch.sin(2.0 * torch.pi * 80.0 * time)
    ).unsqueeze(0)
    noise = torch.randn(1, 400, generator=torch.Generator().manual_seed(43))
    band_signal = _band_limit(signal, sfreq=200.0, min_frequency_hz=6.0, max_frequency_hz=60.0)
    band_noise = _band_limit(noise, sfreq=200.0, min_frequency_hz=6.0, max_frequency_hz=60.0)
    scaled = _scale_noise_to_snr(band_signal, band_noise, -5.0)
    observed = 20.0 * torch.log10(
        band_signal.square().mean(dim=-1).sqrt() / scaled.square().mean(dim=-1).sqrt()
    )
    torch.testing.assert_close(observed, torch.full_like(observed, -5.0), atol=1e-5, rtol=1e-5)
    spectrum = torch.fft.rfft(scaled, dim=-1).abs()
    frequencies = torch.fft.rfftfreq(400, d=1.0 / 200.0)
    outside = (frequencies < 6.0) | (frequencies > 60.0)
    assert spectrum[..., outside].amax() <= spectrum.amax() * 1e-6


def test_local_generators_preserve_global_rng_and_clean_mixture_is_fixed() -> None:
    x = torch.ones(120, 8, 400)
    cond = _condition(batch=120)
    sample_ids = [f"sample-{index:03d}" for index in range(120)]
    before = torch.random.get_rng_state().clone()
    result = apply_query_reliability_training_mixture(
        x,
        cond,
        sample_ids=sample_ids,
        cell_ids=list(QUERY_RELIABILITY_CORRUPTION_CELLS_V1),
        seed=20260904,
        draw_index=3,
        clean_probability=0.25,
    )
    after = torch.random.get_rng_state()
    assert torch.equal(before, after)
    clean_count = int(result[2].sum(dim=1).eq(0).sum())
    assert 18 <= clean_count <= 42


def test_query_summary_uses_average_middle_median_for_even_channels() -> None:
    x = torch.randn(1, 4, 400)
    cond = _condition(batch=1, channels=4)
    cond["channel_query_qc"][0] = torch.tensor([0.1, 0.3, 0.6, 0.9])
    _out_x, out_cond, _affected = apply_query_reliability_corruption(
        x,
        cond,
        sample_ids=["median"],
        cell_id="white_snr_0",
        seed=43,
    )
    normalized = out_cond["channel_query_qc"][0].double()
    raw = torch.expm1(normalized * math.log1p(100.0)).sort().values
    expected_raw = (raw[1] + raw[2]) / 2.0
    expected = torch.log1p(expected_raw) / math.log1p(100.0)
    torch.testing.assert_close(out_cond["query_qc"][0, 0].double(), expected, rtol=1e-6, atol=1e-7)


def test_severity_pairs_share_channels_and_zeroing_fractions_are_nested() -> None:
    x = torch.randn(2, 16, 400)
    cond = _condition(batch=2, channels=16)
    kwargs = {
        "x": x,
        "cond": cond,
        "sample_ids": ["paired-a", "paired-b"],
        "seed": 20260904,
        "draw_index": 2,
    }
    white_0 = apply_query_reliability_corruption(cell_id="white_snr_0", **kwargs)
    white_m5 = apply_query_reliability_corruption(cell_id="white_snr_m5", **kwargs)
    assert torch.equal(white_0[2], white_m5[2])

    line_0 = apply_query_reliability_corruption(cell_id="line50_snr_0", **kwargs)
    line_m5 = apply_query_reliability_corruption(cell_id="line50_snr_m5", **kwargs)
    assert torch.equal(line_0[2], line_m5[2])

    zero_small = apply_query_reliability_corruption(cell_id="zero_0125", **kwargs)
    zero_large = apply_query_reliability_corruption(cell_id="zero_0250", **kwargs)
    assert torch.all(~zero_small[2] | zero_large[2])
