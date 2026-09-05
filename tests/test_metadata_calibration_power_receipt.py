from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "scripts/analyze_metadata_calibration_power.py"
PLAN = REPO / "configs/analysis/metadata_calibration_efficiency_v1.yaml"


def _module():
    spec = importlib.util.spec_from_file_location("metadata_calibration_power_script", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_power_receipt_is_outcome_free_deterministic_and_self_hashed() -> None:
    module = _module()
    first = module.build_receipt(plan_path=PLAN, draws_override=100)
    second = module.build_receipt(plan_path=PLAN, draws_override=100)
    assert first == second
    assert first["status"] == "quick_diagnostic_not_freeze_receipt"
    assert first["outcome_data_used"] is False
    assert first["data_dependencies"] == []
    assert "held_core_hierarchy_monte_carlo" in first
    assert "held_fixed_sequence_monte_carlo" not in first
    assert first["decision_rules"]["confirmatory_savings"] == (
        "A_QM_k1_noninferior_to_A_Q_k3_only"
    )
    scenarios = {
        row["scenario"] for row in first["held_core_hierarchy_monte_carlo"]
    }
    assert {
        "H2a_boundary_other_components_alternative",
        "H2b_boundary_other_components_alternative",
    }.issubset(scenarios)
    assert [row["n"] for row in first["held_core_sample_size_sensitivity"]] == [
        60,
        75,
        80,
        83,
        90,
    ]
    assert all(
        "hard_gate_promotion_probability" in row
        for row in first["development_gate_monte_carlo"]
    )
    development_scenarios = {
        row["scenario"] for row in first["development_gate_monte_carlo"]
    }
    assert development_scenarios == {
        "planning",
        "metadata_null_viable_baseline",
        "joint_bad_null",
    }
    assert first["held_mechanism_holm_monte_carlo"]
    assert all(
        row["conditional_on_confirmatory_core_open"] is True
        for row in first["held_mechanism_holm_monte_carlo"]
    )
    digest = first.pop("receipt_payload_sha256")
    payload = json.dumps(
        first, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()
    assert digest == hashlib.sha256(payload).hexdigest()


def test_power_cli_exposes_no_eeg_manifest_or_prediction_argument() -> None:
    source = SCRIPT.read_text(encoding="utf-8")
    for forbidden_flag in ("--data", "--eeg", "--manifest", "--prediction", "--checkpoint"):
        assert f'parser.add_argument("{forbidden_flag}"' not in source
