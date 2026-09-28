"""Persisted generated qualification: exactly360 optimizer updates, no human IO.

One target/one interface/one seed, all3 budgets/5arms/3bands; pretrain1/fine1.
The separate reader test mocks every raw load/hash and metadata read.
"""

from __future__ import annotations

import copy
import json
import shutil
import time

import numpy as np
import pandas as pd
import pytest
import torch

from cfeg.analysis import dan_data
from cfeg.analysis.dan_audit import _decoder_lineage, audit_saved, numpy_forward
from cfeg.analysis.dan_data import extract_human, save_cache
from cfeg.analysis.dan_roles import FreezeGate
from cfeg.analysis.dan_runner import ROOT, execute, validate_qualification
from cfeg.analysis.dan_runtime import Meter, save_torch, sha_file, write_json
from cfeg.analysis.dan_signal import fbcca_scores, filter_prefix, fit_ensemble_trca
from cfeg.analysis.dan_teacher import support_quality
from cfeg.models.dan_alignment import DanAlignment


@pytest.fixture(autouse=True)
def threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(previous)


def config():
    return json.loads((ROOT / "configs/analysis/dan_teacher_v1.json").read_text())


def generated():
    cfg = config()
    cfg.update(source_subject_ids=list(range(12)), channels=["C0", "C1"], n_samples=96)
    cfg["preprocessing"]["available_prefix_end_exclusive"] = 256
    cfg["training"].update(pretrain_epochs=1, fine_epochs=1, seeds=[20260923])
    rng = np.random.default_rng(849)
    prefix = rng.normal(size=(12, 2, 10, 12, 2, 256))
    t = np.arange(256) / 250
    prefix += np.cos(2 * np.pi * np.array(cfg["frequencies"])[None, None, None, :, None, None]
                     * t[None, None, None, None, None, :])
    bands = np.stack([filter_prefix(x[:, :6], cfg).transpose(1, 0, 2, 3, 4, 5) for x in prefix])
    quality = [[[support_quality(bands[i, e, b, :5], frequencies=cfg["frequencies"], sfreq=250)
                 for b in range(3)] for e in range(2)] for i in range(12)]
    q = np.array([[[p[0] for p in e] for e in person] for person in quality])
    q2 = np.array([[[p[1] for p in e] for e in person] for person in quality])
    return cfg, {"ids": np.arange(12), "orders": np.zeros(12, dtype=int), "bands": bands,
                 "q": q, "q2": q2, "impedance": rng.uniform(0, 100, (12, 2, 5, 2)),
                 "query_prefix": prefix[:, :, 6:10].copy()}


def test_persisted_fit_freeze_score_audit_and_corruption_detection(tmp_path):
    cfg, arrays = generated()
    output = tmp_path / "full-flow"
    output.mkdir()
    write_json(output / "config.json", cfg)
    save_cache(output, arrays)
    meter = Meter(output, cfg["limits"], generated=True)
    before = time.monotonic()
    report = execute(output, cfg, meter, device="cpu", generated_targets=(0,))
    execution_seconds = time.monotonic() - before
    assert meter.counts["adaptation_cells"] == 45
    assert meter.counts["optimizer_fits"] == 270
    assert meter.counts["optimizer_updates"] == 360
    assert meter.counts["source_validation_outputs"] == 270
    assert meter.counts["outer_reveal_batches"] == meter.counts["human_cache_loads"] == 1
    assert len(report["rows"]) == 19
    print("DAN_FULL_FLOW_ARTIFACTS=" + str(output), flush=True)
    audit = audit_saved(output)
    assert audit["alignment_cells_checked"] == 45
    assert audit["selected_states_checked"] == 270
    assert audit["trial_decisions_checked"] == 19 * 48
    assert audit["raw_reads"] == audit["manifest_reads"] == audit["optimizer_updates"] == 0
    write_json(output / "generated_audit.json", audit)
    print("DAN_FULL_FLOW_RESULT=" + json.dumps({"execution_seconds": execution_seconds,
                                               "audit": audit, "output": str(output)}), flush=True)
    corrupt = tmp_path / "teacher-corrupt"
    shutil.copytree(output, corrupt)
    frozen = json.loads((corrupt / "freeze.json").read_text())
    name = frozen["models"][0]["file"]
    packet = torch.load(corrupt / name, weights_only=True)
    packet["teacher"] = packet["teacher"] + 1
    with (corrupt / name).open("wb") as stream:
        torch.save(packet, stream)
    frozen["artifacts"][name] = sha_file(corrupt / name)
    write_json(corrupt / "freeze.json", frozen, replace=True)
    changed_report = json.loads((corrupt / "result.json").read_text())
    changed_report["freeze_sha256"] = sha_file(corrupt / "freeze.json")
    write_json(corrupt / "result.json", changed_report, replace=True)
    with pytest.raises(AssertionError):
        audit_saved(corrupt)
    bad_cost = tmp_path / "cost-corrupt"
    shutil.copytree(output, bad_cost)
    changed_report = json.loads((bad_cost / "result.json").read_text())
    changed_report["fixed_cost_contrast"]["nominal_trial_reduction"] = .99
    write_json(bad_cost / "result.json", changed_report, replace=True)
    with pytest.raises(AssertionError):
        audit_saved(bad_cost)


def test_mocked_reader_exact_allowlist_projection_and_sealed_query(tmp_path, monkeypatch):
    from cfeg.data import prepare_mat, prepare_wearable
    cfg = config()
    records = []
    for subject in cfg["source_subject_ids"]:
        for interface in ("dry", "wet"):
            for block in range(5):
                for _ in range(12):
                    records.append({"subject_id": f"sub{subject:03d}", "electrode_type": interface,
                                    "run_id": f"block{block + 1:02d}", "headband_order": "dry",
                                    "condition_period": "first" if interface == "dry" else "second",
                                    "impedance_kohm_by_channel": np.arange(64, dtype=float)})
    frame = pd.DataFrame.from_records(records, columns=dan_data.MANIFEST_COLUMNS)
    loads, hashes, projections = [], [], []

    def read(path, *, columns, filters):
        projections.append((path, columns, filters))
        return frame

    def digest(path):
        hashes.append(path)
        return dan_data.MANIFEST_SHA if path == dan_data.MANIFEST else "a" * 64

    raw = np.ones(cfg["raw_shape"], dtype=float)
    monkeypatch.setattr(dan_data.pd, "read_parquet", read)
    monkeypatch.setattr(dan_data, "sha_file", digest)
    monkeypatch.setattr(prepare_mat, "_load_arrays", lambda path: loads.append(path) or raw)
    monkeypatch.setattr(prepare_wearable, "_wearable_data", lambda value: value)
    monkeypatch.setattr(dan_data, "filter_prefix", lambda prefix, cfg: np.broadcast_to(
        np.arange(375), (3, 2, 6, 12, 8, 375)))
    monkeypatch.setattr(dan_data, "support_quality", lambda support, **kwargs: (np.zeros((5, 8)), np.ones((5, 8))))
    output = tmp_path / "mock-reader"
    output.mkdir()
    meter = Meter(output, cfg["limits"], generated=True)
    arrays, provenance = extract_human(cfg, meter)
    assert [p.name for p in loads] == [f"S{s:03d}.mat" for s in dan_data.SOURCE_IDS]
    assert len(hashes) == 40 and len(projections) == 1
    assert meter.counts["raw_loads"] == meter.counts["raw_hashes"] == 39
    assert meter.counts["manifest_returned_rows"] == 4680
    assert projections[0][1] == list(dan_data.MANIFEST_COLUMNS)
    assert projections[0][2] == [("subject_id", "in", [f"sub{s:03d}" for s in dan_data.SOURCE_IDS]),
                                 ("run_id", "in", [f"block{b:02d}" for b in range(1, 6)])]
    assert arrays["bands"].shape == (39, 2, 3, 6, 12, 8, 375)
    assert arrays["query_prefix"].shape == (39, 2, 4, 12, 8, 535)
    assert provenance["query_preprocessed_before_freeze"] is False
    vault = dan_data.QueryVault(tuple(arrays["ids"]), arrays["query_prefix"])
    gate = FreezeGate({("all",)})
    with pytest.raises(RuntimeError):
        vault.reveal(gate)
    gate.register(("all",), "a" * 64)
    gate.seal()
    assert vault.reveal(gate).shape[0] == 39
    with pytest.raises(RuntimeError):
        vault.reveal(gate)
    bad_cfg = copy.deepcopy(cfg)
    bad_cfg["source_subject_ids"][0] = 1
    with pytest.raises(ValueError, match="allowlist"):
        extract_human(bad_cfg, meter)
    assert len(loads) == 39


def test_no_human_entry_without_full_qualification_and_budget_limits(tmp_path):
    receipt = tmp_path / "bad_qualification.json"
    write_json(receipt, {"status": "CORE_ONLY", "human_entry_qualified": True})
    with pytest.raises(ValueError, match="qualification"):
        validate_qualification(ROOT, receipt)
    cfg = config()
    meter = Meter(tmp_path, cfg["limits"], generated=True)
    with pytest.raises(RuntimeError, match="budget"):
        meter.consume("optimizer_updates", cfg["limits"]["optimizer_updates"] + 1)
    assert meter.counts == {}
    meter.start -= cfg["limits"]["human_wall_seconds"] + 1
    with pytest.raises(RuntimeError, match="ceiling"):
        meter.check(force=True)


def test_full_shape_nontraining_resource_bound(tmp_path):
    cfg = config()
    rng = np.random.default_rng(332)
    prefix = rng.normal(size=(2, 6, 12, 8, 535))
    started = time.monotonic()
    processed = filter_prefix(prefix, cfg)
    preprocessing_seconds = time.monotonic() - started
    calibration = rng.normal(size=(3, 35, 12, 8, 375))  # worst k5+30source blocks
    started = time.monotonic()
    decoder = fit_ensemble_trca(calibration)
    fit_seconds = time.monotonic() - started
    query = processed[:, 0, :4].reshape(3, 48, 8, 375)
    started = time.monotonic()
    fbcca_scores(query, np.array(cfg["frequencies"]))
    cca_seconds = time.monotonic() - started
    started = time.monotonic()
    decoder.scores(query)
    score_seconds = time.monotonic() - started
    started = time.monotonic()
    _decoder_lineage(calibration.copy(), decoder.filters, decoder.templates)
    lineage_seconds = time.monotonic() - started
    model = DanAlignment(8, 375)
    state = model.state_dict()
    source = rng.normal(size=(72, 8, 375))
    started = time.monotonic()
    numpy_forward(state, source)
    forward_seconds = time.monotonic() - started
    path = tmp_path / "full_shape_checkpoint.pt"
    started = time.monotonic()
    save_torch(path, {"states": [{key: value.clone() for key, value in state.items()}
                                 for _ in range(6)], "teacher": torch.zeros(12, 8, 375),
                      "validation_mse": [.2] * 1250})
    sha_file(path)
    torch.load(path, weights_only=True)
    checkpoint_io_seconds = time.monotonic() - started
    # Explicit 1h margin for orchestration/raw decoding/disk and full trace differences.
    training = 41509.53100283258
    nontraining = 2 * (preprocessing_seconds * 39 * 2 + fit_seconds * 2574
                       + cca_seconds * 78 + score_seconds * 2574 + checkpoint_io_seconds * 7020)
    audit_projection = 2 * (lineage_seconds * 2574 + forward_seconds * 11 * 7020
                             + cca_seconds * 78 + score_seconds * 2574
                             + checkpoint_io_seconds * 7020) + 300
    total_projection = training + nontraining + audit_projection + 3600
    # 6 full distinct model states + teacher + trace per cell; decoder and cache included.
    state_bytes = sum(value.numel() * value.element_size() for value in state.values())
    output_projection = (state_bytes * 6 + 12 * 8 * 375 * 4 + 1250 * 16 + 16384) * 7020
    output_projection += 2574 * 3 * 12 * 8 * 375 * 8 + (1 << 30)
    record = {"preprocessing_seconds": preprocessing_seconds, "decoder_fit_seconds": fit_seconds,
              "cca_seconds": cca_seconds, "score_seconds": score_seconds,
              "decoder_lineage_seconds": lineage_seconds, "numpy_forward_seconds": forward_seconds,
              "checkpoint_io_seconds": checkpoint_io_seconds,
              "training_projection_seconds": training, "nontraining_projection_seconds": nontraining,
              "audit_projection_seconds": audit_projection, "total_projection_seconds": total_projection,
              "output_projection_bytes": output_projection,
              "within_limits": total_projection <= 108000 and audit_projection <= 3600
              and output_projection <= (12 << 30),
              "qualifier": "Generated full-shape microprofile with 2x scaling and 1h margin; runtime guards remain mandatory.",
              "optimizer_updates": 0}
    print("DAN_WHOLE_RESOURCE=" + json.dumps(record, sort_keys=True), flush=True)
    assert record["within_limits"]
