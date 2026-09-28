"""Generated-only signal/learning integration; total 365 optimizer updates.

Five arms x three bands x (pretrain2*3 + five fine1) =165, profile=200.
This is not human-entry qualification: no role reader or whole-run auditor yet.
"""

from __future__ import annotations

import io
import json
import time
from collections import Counter
from pathlib import Path

import numpy as np
import pytest
import torch
from scipy import signal

from cfeg.analysis.dan_signal import (
    EnsembleTRCA,
    fbcca_scores,
    filter_prefix,
    fit_ensemble_trca,
    trca_covariances,
)
from cfeg.analysis.dan_teacher import teacher_block_weights
from cfeg.analysis.dan_training import align_sources, state_digest

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(autouse=True)
def bounded_threads():
    old = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(old)


def config():
    return json.loads((ROOT / "configs/analysis/dan_teacher_v1.json").read_text())


def waves(repeats, channels=8, samples=535, seed=43):
    rng = np.random.default_rng(seed)
    frequencies = np.array(config()["frequencies"])
    time_grid = np.arange(samples) / 250
    fundamental = 2 * np.pi * frequencies[:, None, None] * time_grid
    phase = np.linspace(0, 0.25, channels)[None, :, None]
    template = sum(np.cos(h * fundamental + phase) / h for h in (1, 2, 3))
    return template[None] + rng.normal(0, 0.04, (repeats, 12, channels, samples))


def numpy_forward(state, x):
    weights = {key: value.numpy().astype(np.float64) for key, value in state.items()}
    hidden = np.asarray(x, dtype=float).transpose(0, 2, 1) @ weights["spatial.weight"].T
    flat = hidden.reshape(len(x), -1)
    flat = (flat - weights["normalization.running_mean"]) / np.sqrt(
        weights["normalization.running_var"] + 1e-5)
    flat = flat * weights["normalization.weight"] + weights["normalization.bias"]
    hidden = np.tanh(flat.reshape(hidden.shape) @ weights["hidden.weight"].T
                     + weights["hidden.bias"])
    return (hidden @ weights["output.weight"].T + weights["output.bias"]).transpose(0, 2, 1)


def test_filter_prefix_matches_direct_first_band_and_trial_locality():
    cfg = config()
    raw = waves(2)
    filtered = filter_prefix(raw, cfg)
    assert filtered.shape == (3, 2, 12, 8, 375)
    b, a = signal.iirnotch(50, 35, fs=250)
    order, critical = signal.cheb1ord([8, 90], [6, 100], 3, 40, fs=250)
    sos = signal.cheby1(order, 3, critical, btype="bandpass", fs=250, output="sos")
    direct = signal.sosfiltfilt(sos, signal.filtfilt(b, a, raw[0, 0], axis=-1), axis=-1)
    direct = direct[:, 160:535].copy()
    direct -= direct.mean(-1, keepdims=True)
    direct /= direct.std(-1, ddof=1, keepdims=True)
    np.testing.assert_allclose(filtered[0, 0, 0], direct, rtol=0, atol=1e-12)
    np.testing.assert_array_equal(filter_prefix(raw[:1, :1], cfg), filtered[:, :1, :1])
    np.testing.assert_allclose(filtered.mean(-1), 0, atol=1e-14)
    np.testing.assert_allclose(filtered.std(-1, ddof=1), 1, atol=1e-14)
    with pytest.raises(ValueError, match="exactly"):
        filter_prefix(np.zeros((8, 710)), cfg)
    with pytest.raises(ValueError, match="Zero-power"):
        filter_prefix(np.zeros((8, 535)), cfg)
    with pytest.raises(ValueError):
        filter_prefix(np.full((8, 535), np.nan), cfg)


def test_trca_covariance_matches_explicit_pair_sums():
    x = np.random.default_rng(5).normal(size=(4, 3, 80))
    x -= x.mean(-1, keepdims=True)
    actual_s, actual_q = trca_covariances(x)
    direct_s = sum(x[i] @ x[j].T for i in range(4) for j in range(4) if i != j)
    direct_q = sum(row @ row.T for row in x)
    np.testing.assert_allclose(actual_s, direct_s, rtol=0, atol=1e-12)
    np.testing.assert_allclose(actual_q, direct_q, rtol=0, atol=1e-12)


def test_decoders_identify_generated_stimuli_and_saved_scores_match_scalar_oracle():
    cfg = config()
    calibration = filter_prefix(waves(3), cfg)
    query = filter_prefix(waves(1, seed=11), cfg)[:, 0]
    model = fit_ensemble_trca(calibration)
    before = model.templates.copy()
    scores = model.scores(query)
    np.testing.assert_array_equal(scores.argmax(1), np.arange(12))
    cca = fbcca_scores(query, np.array(cfg["frequencies"]))
    np.testing.assert_array_equal(cca.argmax(1), np.arange(12))
    direct = np.zeros_like(scores)
    for band in range(3):
        for row in range(12):
            for label in range(12):
                projected = model.filters[band].T @ query[band, row]
                template = model.filters[band].T @ model.templates[band, label]
                rho = np.corrcoef(projected.ravel(), template.ravel())[0, 1]
                direct[row, label] += ((band + 1) ** -1.25 + 0.25) * rho * abs(rho)
    np.testing.assert_allclose(scores, direct, rtol=0, atol=1e-12)
    np.testing.assert_array_equal(model.templates, before)
    buffer = io.BytesIO()
    np.savez(buffer, filters=model.filters, templates=model.templates)
    buffer.seek(0)
    with np.load(buffer, allow_pickle=False) as saved:
        restored = EnsembleTRCA(saved["filters"], saved["templates"])
        np.testing.assert_array_equal(restored.scores(query), scores)
    np.testing.assert_allclose(model.scores(query[:, :1]), scores[:1], atol=1e-14)
    reversed_classes = fit_ensemble_trca(calibration[:, :, ::-1])
    np.testing.assert_allclose(reversed_classes.scores(query), scores[:, ::-1], atol=1e-12)


def test_fbcca_matches_generalized_eigenvalue_reference():
    cfg = config()
    query = filter_prefix(waves(1, seed=18), cfg)[:, 0, :1]
    actual = fbcca_scores(query, np.array(cfg["frequencies"][:2]))
    expected = np.zeros((1, 2))
    t = np.arange(375) / 250
    for band in range(3):
        x = query[band, 0]
        x = x - x.mean(-1, keepdims=True)
        xx = x @ x.T
        xx += np.eye(8) * 1e-8 * np.trace(xx) / 8
        for label, frequency in enumerate(cfg["frequencies"][:2]):
            angle = 2 * np.pi * frequency * np.arange(1, 4)[:, None] * t
            y = np.concatenate((np.sin(angle), np.cos(angle)))
            y -= y.mean(-1, keepdims=True)
            yy = y @ y.T
            yy += np.eye(6) * 1e-8 * np.trace(yy) / 6
            xy = x @ y.T
            operator = np.linalg.solve(xx, xy @ np.linalg.solve(yy, xy.T))
            rho_squared = np.linalg.eigvals(operator).real.max()
            expected[0, label] += ((band + 1) ** -1.25 + 0.25) * rho_squared
    np.testing.assert_allclose(actual, expected, rtol=1e-8, atol=1e-8)


def test_five_arm_three_band_generated_learning_and_frozen_decoder():
    cfg = config()
    train = dict(cfg["training"], pretrain_epochs=2, fine_epochs=1)
    rng = np.random.default_rng(14)
    source = rng.normal(size=(3, 5, 6, 12, 2, 96))
    support = rng.normal(size=(3, 3, 12, 2, 96))
    q, q2, metadata = rng.normal(size=(3, 3, 2))
    flags = np.ones_like(q, dtype=bool)
    counts = Counter()
    fitted, fit_records = {}, {}
    for arm in cfg["arms"]:
        aux = {"Q2": q2, "QM": metadata, "SHAM": metadata[::-1]}.get(arm)
        weights = teacher_block_weights(q, arm, auxiliary=aux, observed=flags)
        augmented, fit_records[arm] = [], []
        for band in range(3):
            fit = align_sources(source[band], support[band], weights, train,
                                seed=20260923 + band, device="cpu",
                                consume=lambda key, n: counts.update({key: n}), check=lambda: None)
            assert fit.optimizer_updates == 11 and fit.source_validation_outputs == 7
            assert fit.transformed.shape == (30, 12, 2, 96)
            assert len(fit.states) == len(fit.records) == 6
            assert all(r["initial_sha256"] == fit.records[0]["selected_sha256"]
                       for r in fit.records[1:])
            for record, state in zip(fit.records, fit.states):
                assert record["best_epoch_zero_based"] == int(np.argmin(record["validation_mse"]))
                assert record["selected_sha256"] == state_digest(state)
            for person in range(5):
                direct = numpy_forward(fit.states[person + 1],
                                       source[band, person].reshape(72, 2, 96))
                np.testing.assert_allclose(
                    fit.transformed[person * 6:(person + 1) * 6].reshape(72, 2, 96), direct,
                    rtol=2e-5, atol=2e-6)
            augmented.append(np.concatenate((support[band], fit.transformed)))
            fit_records[arm].append(fit.records)
        fitted[arm] = fit_ensemble_trca(np.stack(augmented))
    assert counts == {"optimizer_fits": 90, "optimizer_updates": 165,
                      "source_validation_outputs": 105}
    for arm in cfg["arms"]:
        for band in range(3):
            for left, right in zip(fit_records["U"][band], fit_records[arm][band]):
                assert left["batch_order_sha256"] == right["batch_order_sha256"]
            assert (fit_records["U"][band][0]["initial_sha256"]
                    == fit_records[arm][band][0]["initial_sha256"])
    # Query is created only after every generated decoder has been fitted.
    query = rng.normal(size=(3, 7, 2, 96))
    for model in fitted.values():
        score = model.scores(query)
        assert score.shape == (7, 12) and np.isfinite(score).all()
    assert not np.allclose(fitted["Q"].templates, fitted["QM"].templates)


def test_full_shape_cuda_two_stage_partial_resource_profile():
    assert torch.cuda.is_available(), "Do not silently skip GPU qualification."
    cfg = config()
    train = dict(cfg["training"], pretrain_epochs=50, fine_epochs=10)
    rng = np.random.default_rng(20260923)
    source = rng.normal(size=(5, 6, 12, 8, 375))
    support = rng.normal(size=(3, 12, 8, 375))
    weights = np.full((3, 8), 1 / 3)
    counts = Counter()
    torch.cuda.reset_peak_memory_stats()
    start = time.monotonic()
    fit = align_sources(source, support, weights, train, seed=20260923, device="cuda",
                        consume=lambda key, n: counts.update({key: n}), check=lambda: None)
    elapsed = time.monotonic() - start
    save_start = time.monotonic()
    packet = io.BytesIO()
    torch.save({"states": fit.states, "records": fit.records}, packet)
    save_seconds = time.monotonic() - save_start
    packet_bytes = packet.tell()
    assert counts == {"optimizer_fits": 6, "optimizer_updates": 200,
                      "source_validation_outputs": 100}
    # 12.5 = max(full/generated update ratio11.25, validation count ratio12.5).
    # Twofold headroom is explicit. Decoder/cache/audit costs still excluded.
    projection = 2 * (elapsed * 12.5 + save_seconds) * 7020
    print("DAN_TWO_STAGE_PROFILE=" + json.dumps({
        "gpu": torch.cuda.get_device_name(), "precision": "float32",
        "shape": [5, 6, 12, 8, 375], "elapsed_seconds": elapsed,
        "checkpoint_memory_serialization_seconds": save_seconds,
        "checkpoint_packet_bytes": packet_bytes,
        "twofold_training_path_projection_seconds": projection,
        "projected_checkpoint_bytes": packet_bytes * 7020,
        "peak_torch_allocated_bytes": torch.cuda.max_memory_allocated(),
        "optimizer_updates": 200, "validation_outputs": 100,
        "whole_run_qualified": False,
        "excluded": ["full validation traces", "filesystem IO", "filter bank", "eTRCA",
                     "cohort scheduling and cache", "saved audit"],
    }, sort_keys=True))
