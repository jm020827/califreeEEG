"""Generated integration contract checks; no human input or real run fixture."""

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

from cfeg.analysis import task_trca_temporal_evaluation as ev
from cfeg.analysis import task_trca_temporal_learning as learn

ROOT = Path(__file__).resolve().parents[1]


def module(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / (name + ".py"))
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def test_frozen_design_keeps_old_metadata_gates_and_tightens_full_controls():
    old = json.loads((ROOT / "configs/analysis/task_aligned_trca_shape_v1_design.json").read_text())
    new = json.loads((ROOT / "configs/analysis/task_trca_temporal_v1_design.json").read_text())
    assert new["authority"]["human_execution_authorized"] is False
    assert new["authority"]["human_artifact_reads_this_stage"] is False
    assert new["authority"]["held60_access_authorized"] is False
    assert new["authority"]["autostart"] is False
    assert new["cohort_plan"] == old["cohort_plan"]
    assert new["decision"]["metadata_increment"] == old["decision"]["metadata_increment"]
    assert new["arms"] == ["A0", *ev.ARMS]
    assert new["study_id"] == ev.SCHEMA == learn.SCHEMA
    assert new["score_schema"] == ev.SCORE_SCHEMA == learn.SCORE_SCHEMA
    assert new["learner"]["lambda_candidates"] == list(learn.LAMBDAS)
    assert new["learner"]["steps_per_head"] == learn.STEPS == 200
    gates = new["decision"]["calibration_candidate_additional"]
    assert gates["qm3_minus_full_native3_ci_low_greater_than_pp"] == -1
    assert gates["qm3_minus_full_centered3_ci_low_greater_than_pp"] == -1
    assert "anchor_abs_c_cosine_min" not in new["operator"]
    for key in ("eta", "top_relative_gap_min", "r_max_expression"):
        assert new["operator"][key] == old["operator"][key]


def test_generated_probe_has_one_fixed_seed_and_shared_metadata_prefix():
    probe = module("check_task_trca_temporal_engineering")
    left, right = probe.fixture(), probe.fixture()
    assert set(left) == {*probe.SOURCE_IDS, *probe.EVAL_IDS}
    assert not set(probe.SOURCE_IDS) & set(probe.EVAL_IDS)
    for pid in left:
        for a, b in zip(left[pid], right[pid]):
            np.testing.assert_array_equal(a, b)
    c3 = probe.make_case(probe.SOURCE_IDS[0], left[probe.SOURCE_IDS[0]], 17, 3)
    c5 = probe.make_case(probe.SOURCE_IDS[0], left[probe.SOURCE_IDS[0]], 23, 5)
    assert c3.q.shape == c5.q.shape == (5, 8, 15)
    np.testing.assert_array_equal(c3.packet, c5.packet[:3])


def test_cold_receipt_pin_precedes_any_npz_access(tmp_path, monkeypatch):
    cold = module("audit_task_trca_temporal_engineering")
    (tmp_path / "receipt.json").write_text("not a pinned generated receipt")

    def forbidden(*args, **kwargs):
        pytest.fail("must not load any numeric archive with bad receipt hash")

    monkeypatch.setattr(cold.np, "load", forbidden)
    with pytest.raises(ValueError, match="receipt hash"):
        cold.audit(tmp_path, "0" * 64)
