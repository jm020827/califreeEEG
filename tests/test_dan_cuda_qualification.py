"""Final generated call:72 CUDA updates, persisted CPU re-audit, no human IO."""

import json
from collections import Counter
from pathlib import Path

import numpy as np
import torch

from cfeg.analysis.dan_audit import _decoder_lineage, _trca_scores, audit_saved, numpy_forward
from cfeg.analysis.dan_runtime import save_torch
from cfeg.analysis.dan_signal import filter_prefix, fit_ensemble_trca
from cfeg.analysis.dan_teacher import teacher_block_weights
from cfeg.analysis.dan_training import align_sources

ROOT = Path(__file__).resolve().parents[1]
GENERATED = Path("/home/whwovy/eeg-data/dan-generated-v1-pbg3mb/test_persisted_fit_freeze_scor0/full-flow")


def test_saved_generated_complete_flow_cold_reaudit():
    result = audit_saved(GENERATED)
    assert result["status"] == "PASS_SAVED_MODELS_DECODERS_AND_ENDPOINTS"
    assert result["alignment_cells_checked"] == 45
    assert result["trial_decisions_checked"] == 912
    print("DAN_COLD_REAUDIT=" + json.dumps(result, sort_keys=True))


def test_cuda_full_shape_three_budgets_three_bands_saved_lineage(tmp_path):
    torch.set_num_threads(2)
    assert torch.cuda.is_available()
    cfg = json.loads((ROOT / "configs/analysis/dan_teacher_v1.json").read_text())
    training = dict(cfg["training"], pretrain_epochs=1, fine_epochs=1)
    rng = np.random.default_rng(721)
    time = np.arange(535) / 250
    base = np.cos(2 * np.pi * np.asarray(cfg["frequencies"])[:, None, None] * time)
    source_prefix = rng.normal(0, .7, (5, 6, 12, 8, 535)) + base
    target_prefix = rng.normal(0, .7, (5, 12, 8, 535)) + base
    query_prefix = rng.normal(0, .7, (4, 12, 8, 535)) + base
    source, support = filter_prefix(source_prefix, cfg), filter_prefix(target_prefix, cfg)
    counts = Counter()
    decoders = []
    max_forward_error = 0.
    for k in (2, 3, 5):
        actual, independent = [], []
        for band in range(3):
            q, m = rng.normal(size=(2, k, 8))
            weights = teacher_block_weights(q, "QM", auxiliary=m, observed=np.ones_like(m, dtype=bool))
            fit = align_sources(source[band], support[band, :k], weights, training,
                                seed=20260923 + band, device="cuda",
                                consume=lambda key, n: counts.update({key: n}), check=lambda: None)
            path = tmp_path / f"k{k}-b{band}.pt"
            save_torch(path, {"states": fit.states, "records": fit.records,
                              "teacher": torch.from_numpy(fit.teacher)})
            restored = torch.load(path, map_location="cpu", weights_only=True)
            waves = []
            for stage, (state, record) in enumerate(zip(restored["states"], restored["records"])):
                valid = source[band, 4] if stage == 0 else source[band, stage - 1, 4:]
                computed = numpy_forward(state, valid.reshape(-1, 8, 375))
                teacher = np.tile(restored["teacher"].numpy(), (len(valid), 1, 1))
                mse = np.mean((computed - teacher) ** 2)
                np.testing.assert_allclose(mse, min(record["validation_mse"]), rtol=5e-5, atol=5e-6)
                if stage:
                    direct = numpy_forward(state, source[band, stage - 1].reshape(72, 8, 375))
                    expected = fit.transformed[(stage - 1) * 6:stage * 6].reshape(72, 8, 375)
                    max_forward_error = max(max_forward_error, float(abs(direct - expected).max()))
                    np.testing.assert_allclose(direct, expected, rtol=2e-5, atol=2e-6)
                    waves.append(direct.reshape(6, 12, 8, 375))
            actual.append(np.concatenate((support[band, :k], fit.transformed)))
            independent.append(np.concatenate((support[band, :k], *waves)))
        model = fit_ensemble_trca(np.stack(actual))
        _decoder_lineage(np.stack(independent), model.filters, model.templates)
        decoders.append(model)
    query = filter_prefix(query_prefix, cfg).reshape(3, 48, 8, 375)
    for model in decoders:
        direct = _trca_scores(model.filters, model.templates, query)
        score = model.scores(query)
        np.testing.assert_allclose(direct, score, atol=1e-5, rtol=1e-5)
        np.testing.assert_array_equal(direct.argmax(1), score.argmax(1))
    assert counts == {"optimizer_fits": 54, "optimizer_updates": 72, "source_validation_outputs": 54}
    print("DAN_CUDA_SAVED_QUALIFICATION=" + json.dumps({
        "counts": dict(counts), "selected_states_checked": 54,
        "shape": [5, 6, 12, 8, 375], "budgets": [2, 3, 5], "bands": 3,
        "max_numpy_forward_error": max_forward_error, "query_decisions_checked": 144,
        "argmax_disagreements": 0, "device": torch.cuda.get_device_name(),
        "raw_reads": 0, "held60": 0}, sort_keys=True))
