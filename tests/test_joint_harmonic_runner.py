"""Generated end-to-end qualification. All human readers are mocked or forbidden."""

from __future__ import annotations

import copy
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from cfeg.analysis.joint_harmonic import ContextScaler
from cfeg.analysis.joint_harmonic_audit import audit_saved
from cfeg.analysis.joint_harmonic_data import (
    MANIFEST_COLUMNS,
    SOURCE_IDS,
    allowed_raw_path,
    extract_participant,
    parse_metadata,
    prefix_contexts,
    read_human_cache,
    select_role,
)
from cfeg.analysis.joint_harmonic_runner import (
    Fitted,
    Meter,
    evaluate_model,
    execute,
    file_sha,
    human_entry,
    model_from_config,
    save_npz,
    write_json,
)

ROOT = Path(__file__).resolve().parents[1]


def base_config():
    return json.loads((ROOT / "configs/analysis/joint_harmonic_program_v1.json").read_text())


def generated_cache():
    cfg = base_config()
    cfg.update(source_subject_ids=list(range(12)), channels=["C0", "C1"],
               canonical_channel_ids=[1, 2], raw_shape=[2, 220, 2, 10, 12],
               n_samples=200, start_sample=10)
    cfg["training"]["steps"] = 3  # exactly 48*3=144 optimizer updates in the full flow
    cfg["evaluation"]["bootstrap_draws"] = 20
    rng = np.random.default_rng(867)
    parts = []
    t = np.arange(220) / cfg["sfreq"]
    wave = np.cos(2 * np.pi * t[:, None] * np.asarray(cfg["frequencies"])[None])
    for _ in cfg["source_subject_ids"]:
        x = rng.normal(0, 0.8, cfg["raw_shape"])
        x += wave[None, :, None, None, :] * rng.uniform(0.5, 2, (2, 1, 2, 1, 1))
        parts.append(extract_participant(x, cfg))
    cache = {key: np.stack([p[key] for p in parts]) for key in parts[0]}
    cache.update(ids=np.arange(12), orders=np.arange(12) % 2,
                 impedance=rng.uniform(0, 100, (12, 2, 5, 2)))
    return cfg, cache


def metadata_frame(cfg):
    records = []
    for i, subject in enumerate(cfg["source_subject_ids"]):
        order = ("dry", "wet")[i % 2]
        for e in ("dry", "wet"):
            for b in range(5):
                z = np.arange(64, dtype=float) + subject + b
                for _ in cfg["frequencies"]:
                    records.append({"subject_id": f"sub{subject:03d}", "electrode_type": e,
                                    "run_id": f"block{b + 1:02d}",
                                    "impedance_kohm_by_channel": z.copy(), "headband_order": order,
                                    "condition_period": "first" if e == order else "second"})
    return pd.DataFrame.from_records(records, columns=MANIFEST_COLUMNS)


def test_generated_full_flow_and_independent_saved_audit(tmp_path):
    cfg, cache = generated_cache()
    output = tmp_path / "generated"
    output.mkdir()
    write_json(output / "config.json", cfg)
    save_npz(output / "features.npz", **cache)
    write_json(output / "cache_receipt.json", {"sha256": file_sha(output / "features.npz")})
    meter = Meter(output, cfg["limits"], generated=True)
    result = execute(cache, cfg, meter, device="cpu")
    assert meter.counts["human_fits"] == 48
    assert meter.counts["human_updates"] == 144
    assert meter.counts["human_episodes"] == 576
    assert meter.counts["inner_evaluations"] == 24
    assert meter.counts["outer_reveal_batches"] == 1
    assert result["generated"] and len(result["arms"]) == 4
    assert "UNKNOWN" in " ".join(result["limits_of_inference"])
    audit = audit_saved(output, generated=True)
    assert audit["status"] == "PASS_SAVED_FORWARD_ROLES_AND_METRICS"
    assert audit["raw_reads"] == audit["fits"] == 0
    assert audit["model_count"] == 24
    with pytest.raises(FileExistsError):
        execute(cache, cfg, meter, device="cpu")
    assert meter.counts["human_updates"] == 144


def test_support_context_ignores_query_q_and_other_participants():
    _, cache = generated_cache()
    selected = [0, 2, 4, 6]
    before = select_role(cache, selected)
    cache["q"][:, :, 5:] = np.nan
    cache["q2"][:, :, 5:] = np.nan
    cache["spectra"][1::2] = np.nan
    after = select_role(cache, selected)
    for arm in ("Q", "Q2", "QM", "SHAM"):
        a, _ = prefix_contexts(before, arm, sham_seed=7)
        b, _ = prefix_contexts(after, arm, sham_seed=7)
        np.testing.assert_array_equal(a, b)
    assert np.isfinite(after.spectra).all()
    assert not np.shares_memory(after.q, cache["q"])
    assert not after.q.flags.writeable
    cache["impedance"][:] = -1  # Q must not read or validate M, even in role construction.
    q_role = select_role(cache, selected)
    prefix_contexts(q_role, "Q", sham_seed=7)
    with pytest.raises(ValueError):
        prefix_contexts(q_role, "QM", sham_seed=7)


def test_prefix_one_does_not_use_later_support_blocks():
    _, cache = generated_cache()
    a = select_role(cache, [0, 2, 4, 6])
    for key in ("q", "q2", "impedance"):
        cache[key][:, :, 1:5] += 10
    b = select_role(cache, [0, 2, 4, 6])
    for arm in ("Q", "Q2", "QM", "SHAM"):
        x, _ = prefix_contexts(a, arm, sham_seed=7)
        y, _ = prefix_contexts(b, arm, sham_seed=7)
        np.testing.assert_array_equal(x[:, :, 0], y[:, :, 0])
        assert not np.array_equal(x[:, :, 2], y[:, :, 2])


def test_evaluator_refuses_model_fit_participants():
    cfg, cache = generated_cache()
    role = select_role(cache, [0, 1, 2, 3])
    normalizer = ContextScaler(np.zeros((2, 4)), np.ones((2, 4)), role.ids)
    fitted = Fitted(model_from_config(cfg), normalizer, {"arm": "Q", "fold": 0})
    with pytest.raises(ValueError, match="overlaps"):
        evaluate_model(fitted, role, cfg, "cpu")


def test_metadata_projection_duplicate_and_information_boundary():
    cfg = base_config()
    cfg["source_subject_ids"] = [4, 6]
    frame = metadata_frame(cfg)
    order, impedance = parse_metadata(frame, cfg)
    assert order.tolist() == [0, 1] and impedance.shape == (2, 2, 5, 8)
    for bad in (frame.iloc[:-1], frame.assign(label=0), frame.assign(subject_id="sub999")):
        with pytest.raises(ValueError):
            parse_metadata(bad, cfg)
    damaged = copy.deepcopy(frame)
    replacement = damaged.iloc[0].impedance_kohm_by_channel.copy()
    replacement[0] += 1
    damaged.at[0, "impedance_kohm_by_channel"] = replacement
    with pytest.raises(ValueError, match="disagree"):
        parse_metadata(damaged, cfg)


def test_actual_reader_with_mocked_io_uses_exact_source39_and_first_five_blocks(tmp_path, monkeypatch):
    cfg = base_config()
    frame = metadata_frame(cfg)
    seen = {"raw": [], "parquet": []}

    def parquet(path, *, columns, filters):
        seen["parquet"].append(str(path))
        assert columns == list(MANIFEST_COLUMNS)
        assert filters == [("subject_id", "in", [f"sub{s:03d}" for s in SOURCE_IDS]),
                           ("run_id", "in", [f"block{b:02d}" for b in range(1, 6)])]
        return frame

    def load(path):
        seen["raw"].append(path.name)
        return {"data": "generated placeholder"}

    from cfeg.analysis import joint_harmonic_data as data
    from cfeg.data import prepare_mat, prepare_wearable

    monkeypatch.setattr(pd, "read_parquet", parquet)
    monkeypatch.setattr(prepare_mat, "_load_arrays", load)
    monkeypatch.setattr(prepare_wearable, "_wearable_data", lambda arrays: arrays["data"])
    monkeypatch.setattr(data, "extract_participant", lambda raw, config: {"spectra": np.ones((1,))})
    meter = Meter(tmp_path, cfg["limits"], generated=True)
    cache, provenance = read_human_cache(cfg, meter, lambda path: cfg["manifest_sha256"])
    assert seen["raw"] == [f"S{s:03d}.mat" for s in SOURCE_IDS]
    assert len(seen["parquet"]) == 1
    assert cache["ids"].tolist() == list(SOURCE_IDS)
    assert provenance["returned_metadata_rows"] == 4680
    assert meter.counts["raw_loads"] == meter.counts["raw_hashes"] == 39
    assert not provenance["query_scores_computed"]


def test_reject_held_raw_access_and_unqualified_human_entry(tmp_path):
    cfg = base_config()
    with pytest.raises(ValueError, match="outside"):
        allowed_raw_path(1, cfg)
    cfg["source_subject_ids"] = [1]
    with pytest.raises(ValueError, match="boundary"):
        allowed_raw_path(1, cfg)
    # A private fixture, never the mutable real repository state or a live authority.
    for relative in ("docs/joint_harmonic_program_v1_plan.md",
                     "configs/analysis/joint_harmonic_program_v1.json"):
        destination = tmp_path / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / relative, destination)
    (tmp_path / "docs/reports").mkdir()
    write_json(tmp_path / "docs/reports/joint_harmonic_program_v1_state.json", {"status": "DRAFT"})
    with pytest.raises(RuntimeError, match="qualification"):
        human_entry(tmp_path)


def test_meter_enforces_fixed_counts_and_fresh_output(tmp_path):
    meter = Meter(tmp_path, {"human_fits": 1}, generated=True)
    meter.consume("human_fits")
    with pytest.raises(RuntimeError, match="Budget"):
        meter.consume("human_fits")
    write_json(tmp_path / "immutable.json", {"failure": "preserved"})
    with pytest.raises(FileExistsError):
        write_json(tmp_path / "immutable.json", {"failure": "erased"})
