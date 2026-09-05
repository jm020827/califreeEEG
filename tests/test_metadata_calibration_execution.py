from __future__ import annotations

import pytest

from cfeg.metadata_calibration_execution import (
    cell_execution_contract_sha256,
    execution_cells_for_phase,
    resolve_cell_execution_spec,
)


def test_held_role_context_dispatch_is_exact() -> None:
    aq = resolve_cell_execution_spec(
        role="A_Q", context="exact_off", phase="held_participant_evaluation"
    )
    assert aq.query_q_mode == aq.support_q_mode == "observed"
    assert aq.query_metadata_mode == aq.support_metadata_mode == "off"
    assert aq.intervention == "exact_off"

    missing = resolve_cell_execution_spec(
        role="A_QM", context="all_missing", phase="held_participant_evaluation"
    )
    assert missing.query_metadata_mode == missing.support_metadata_mode == "all_missing"
    assert missing.intervention == "all_missing"

    am = resolve_cell_execution_spec(
        role="A_M", context="correct", phase="held_participant_evaluation"
    )
    assert am.query_q_mode == am.support_q_mode == "off"
    assert am.query_metadata_mode == am.support_metadata_mode == "observed"


def test_development_diagnostics_cannot_enter_held_dispatch() -> None:
    support_only = resolve_cell_execution_spec(
        role="A_QM", context="support_metadata_only", phase="source_development"
    )
    assert support_only.query_metadata_mode == "all_missing"
    assert support_only.support_metadata_mode == "observed"

    interface = resolve_cell_execution_spec(
        role="A_QM", context="interface_only", phase="source_development"
    )
    assert interface.query_metadata_ablation == "interface_only"
    assert interface.support_metadata_ablation == "interface_only"

    with pytest.raises(ValueError, match="not declared"):
        resolve_cell_execution_spec(
            role="A_QM",
            context="support_metadata_only",
            phase="held_participant_evaluation",
        )


def test_dispatch_contract_hash_is_phase_specific_and_stable() -> None:
    held = cell_execution_contract_sha256(phase="held_participant_evaluation")
    development = cell_execution_contract_sha256(phase="source_development")
    assert len(held) == len(development) == 64
    assert held != development
    assert held == cell_execution_contract_sha256(phase="held_participant_evaluation")
    assert len(execution_cells_for_phase(phase="held_participant_evaluation")) == 8
    assert len(execution_cells_for_phase(phase="source_development")) == 12


def test_unknown_role_or_phase_is_rejected() -> None:
    with pytest.raises(ValueError, match="not declared"):
        resolve_cell_execution_spec(
            role="A_QM", context="invented", phase="source_development"
        )
    with pytest.raises(ValueError, match="Unknown"):
        resolve_cell_execution_spec(role="A_Q", context="exact_off", phase="invalid")  # type: ignore[arg-type]
