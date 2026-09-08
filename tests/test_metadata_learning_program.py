"""Prospective budget/science and generated-origin checks; no human data reads."""

import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]


def document(name):
    return json.loads((ROOT / "configs/analysis" / name).read_text())


def generator():
    spec = importlib.util.spec_from_file_location(
        "generated_temporal_inputs", ROOT / "scripts/check_task_trca_temporal_source39_generated.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_program_two_candidates_with_frozen_science_and_no_unbounded_authority():
    plan = document("metadata_learning_program_v1.json")
    assert [row["slot"] for row in plan["candidates"]] == ["C1", "C2"]
    for row in plan["candidates"]:
        assert (
            hashlib.sha256((ROOT / row["design_path"]).read_bytes()).hexdigest()
            == row["design_sha256"]
        )
    budget = plan["budget"]
    assert budget["scientific_candidates_max"] == 2
    assert budget["human_data_attempts_total_max"] == 3
    assert budget["human_optimizer_updates_total_max_including_partial_and_recovery"] == 72000
    assert budget["distinct_final_query_model_reveals_max"] == 2
    assert plan["authority"]["source39_development_authorized_after_candidate_preflight"] is True
    for key in (
        "held60_authorized",
        "retired_s1_s3_authorized",
        "external_outreach_authorized",
        "paid_resources_authorized",
        "old_terminal_candidates_reopened",
    ):
        assert plan["authority"][key] is False
    assert (
        document("task_trca_temporal_v1_design.json")["authority"]["human_execution_authorized"]
        is False
    )


def test_c2_is_s_only_and_not_renamed_template_or_denominator_candidate():
    design = document("task_trca_pair_s_v1_design.json")
    assert design["slot"] == "C2"
    assert design["status"] == "SCIENTIFIC_SPEC_FROZEN_BEFORE_C1_HUMAN_FITTING"
    assert design["operator"]["b"].startswith("C EXACTLY")
    assert "unchangeduniform" in design["operator"]["templates"]
    assert len(design["q_features"]["ordered_features_zero_based"]) == 15
    assert design["q_features"]["q2_indices"] == [2, 4]
    assert len(design["m_features"]["ordered_features"]) == 2
    assert design["controls"]["arms"][2] == "UNIFORM_C2"
    assert design["fitting"]["max_updates"] == 24000
    assert design["decision"]["held60_autostart"] is False


def test_generated_fixture_deterministic_and_query_not_tiled():
    module = generator()
    left, packets, a0 = module.generated_values()
    right, repeated_packets, repeated_a0 = module.generated_values()
    assert packets == repeated_packets
    assert len(packets) == 780
    for pid in module.IDS:
        np.testing.assert_array_equal(left[pid], right[pid])
        np.testing.assert_array_equal(a0[pid], repeated_a0[pid])
        assert left[pid].shape == (2, 10, 12, 5, 8, 17)
        assert not np.array_equal(left[pid][:, 6], left[pid][:, 7])


def test_generated_builder_refuses_overwrite_before_creating_inputs(tmp_path):
    module = generator()
    with pytest.raises(ValueError, match="new unaliased"):
        module.build_inputs(tmp_path)
    assert list(tmp_path.iterdir()) == []
