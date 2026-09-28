"""One bounded cold/steady cost separation: 8 warm + 200 measured updates."""

import io
import json
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch

from cfeg.analysis.dan_training import align_sources


def test_cold_start_and_steady_full_shape_cost():
    torch.set_num_threads(2)
    assert torch.cuda.is_available()
    config = json.loads((Path(__file__).resolve().parents[1]
                         / "configs/analysis/dan_teacher_v1.json").read_text())
    rng = np.random.default_rng(20260923)
    source = rng.normal(size=(5, 6, 12, 8, 375))
    support = rng.normal(size=(3, 12, 8, 375))
    weights = np.full((3, 8), 1 / 3)
    counts = Counter()
    kwargs = {"seed": 20260923, "device": "cuda",
              "consume": lambda key, n: counts.update({key: n}), "check": lambda: None}
    start = time.monotonic()
    warm = align_sources(source, support, weights,
                         dict(config["training"], pretrain_epochs=1, fine_epochs=1), **kwargs)
    torch.cuda.synchronize()
    warm_seconds = time.monotonic() - start
    assert warm.optimizer_updates == 8
    del warm
    torch.cuda.reset_peak_memory_stats()
    start = time.monotonic()
    fit = align_sources(source, support, weights,
                        dict(config["training"], pretrain_epochs=50, fine_epochs=10), **kwargs)
    torch.cuda.synchronize()
    seconds = time.monotonic() - start
    save_start = time.monotonic()
    packet = io.BytesIO()
    torch.save({"states": fit.states, "records": fit.records}, packet)
    save_seconds = time.monotonic() - save_start
    # Fixed conservative factor, no fastest-of-N choice; cold start counted once.
    projected = warm_seconds + 2 * (seconds * 12.5 + save_seconds) * 7020
    assert counts == {"optimizer_fits": 12, "optimizer_updates": 208,
                      "source_validation_outputs": 106}
    print("DAN_STEADY_PROFILE=" + json.dumps({
        "gpu": torch.cuda.get_device_name(), "precision": "float32",
        "warm_seconds": warm_seconds, "steady_200_updates_seconds": seconds,
        "checkpoint_memory_save_seconds": save_seconds,
        "projection_seconds": projected,
        "training_budget_seconds": config["limits"]["gpu_training_seconds"],
        "training_path_within_budget": projected <= config["limits"]["gpu_training_seconds"],
        "optimizer_updates": 208,
        "peak_torch_allocated_bytes": torch.cuda.max_memory_allocated(),
        "whole_run_qualified": False,
        "excluded": ["disk IO", "full traces", "eTRCA", "cache", "cohort orchestration", "audit"],
    }, sort_keys=True))
