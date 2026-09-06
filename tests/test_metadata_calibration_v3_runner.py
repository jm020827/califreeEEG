from __future__ import annotations

import io
import os
from pathlib import Path

import pytest

from cfeg.metadata_calibration_v3_runner import (
    RunnerError,
    RunStage,
    WorkflowInspection,
    _build_parser,
    emit_selection_bytes,
    resume_once,
    run_cli,
    status_report,
)

_REPOSITORY = Path(__file__).resolve().parents[1]
_RUNNER = _REPOSITORY / "src/cfeg/metadata_calibration_v3_runner.py"
_WRAPPER = _REPOSITORY / "scripts/run_metadata_calibration_v3"
_SELECTED_BYTES = (
    b'{"payload_sha256":"aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",'
    b'"schema":"fake-selected.v1"}\n'
)

_NEXT = {
    RunStage.BUNDLE_PENDING: RunStage.CONTEXT_REFERENCE_PENDING,
    RunStage.CONTEXT_REFERENCE_PENDING: RunStage.DEVELOPMENT_PENDING,
    RunStage.DEVELOPMENT_PENDING: RunStage.SELECTION_HANDOFF,
    RunStage.TEST_EVIDENCE_PENDING: RunStage.CANARY_AUTHORIZATION_PENDING,
    RunStage.CANARY_AUTHORIZATION_PENDING: RunStage.CANARY_CLAIM_PENDING,
    RunStage.CANARY_CLAIM_PENDING: RunStage.CANARY_BEACON_PENDING,
    RunStage.CANARY_BEACON_PENDING: RunStage.CANARY_RESULT_PENDING,
    RunStage.CANARY_RESULT_PENDING: RunStage.FRESH_CANARY_AUDIT_PENDING,
    RunStage.FRESH_CANARY_AUDIT_PENDING: RunStage.CANARY_PASSED,
}


class FakeFacade:
    def __init__(
        self,
        stage: RunStage,
        *,
        development_no_go: bool = False,
        selection_bytes: bytes = _SELECTED_BYTES,
    ) -> None:
        self.stage = stage
        self.development_no_go = development_no_go
        self.exact_selection_bytes = selection_bytes
        self.events: list[str] = []
        self.inspect_count = 0
        self.advance_count = 0
        self.selection_count = 0
        self.execute_count = 0
        self.downstream_count = 0

    def _inspection(self) -> WorkflowInspection:
        if self.stage is RunStage.DEVELOPMENT_NO_GO:
            return WorkflowInspection(
                self.stage,
                selection_status="DEVELOPMENT_NO_GO",
            )
        if self.stage in {
            RunStage.SELECTION_HANDOFF,
            RunStage.TEST_EVIDENCE_PENDING,
            RunStage.CANARY_AUTHORIZATION_PENDING,
            RunStage.CANARY_CLAIM_PENDING,
            RunStage.CANARY_BEACON_PENDING,
            RunStage.CANARY_RESULT_PENDING,
            RunStage.FRESH_CANARY_AUDIT_PENDING,
            RunStage.CANARY_PASSED,
        }:
            return WorkflowInspection(
                self.stage,
                selection_status="SELECTED_METHOD_PROPOSED",
                selected_grid_cell_id="p3-nu_4-lambda_0p20",
            )
        return WorkflowInspection(self.stage)

    def inspect(self) -> WorkflowInspection:
        self.inspect_count += 1
        self.events.append(f"reopen:{self.stage.value}")
        if self.stage is RunStage.SELECTION_HANDOFF:
            self.events.append("recover:canonical_result_and_audited_proof")
        return self._inspection()

    def advance(self, expected_stage: RunStage) -> WorkflowInspection:
        assert expected_stage is self.stage
        self.advance_count += 1
        if self.stage is RunStage.BUNDLE_PENDING:
            self.events.append("publish:development_bundle")
        elif self.stage is RunStage.CONTEXT_REFERENCE_PENDING:
            self.events.append("publish:context_reference")
        elif self.stage is RunStage.DEVELOPMENT_PENDING:
            self.execute_count += 1
            self.events.extend(
                ("execute:authoritative_development", "publish:development_result")
            )
            if self.development_no_go:
                self.stage = RunStage.DEVELOPMENT_NO_GO
                return self.inspect()
        elif self.stage is RunStage.TEST_EVIDENCE_PENDING:
            self.downstream_count += 1
            self.events.extend(
                (
                    "run:focused_v3",
                    "run:repository_full",
                    "publish:test_evidence",
                )
            )
        elif self.stage is RunStage.CANARY_AUTHORIZATION_PENDING:
            self.downstream_count += 1
            self.events.append("publish:canary_authorization")
        elif self.stage is RunStage.CANARY_CLAIM_PENDING:
            self.downstream_count += 1
            self.events.append("publish:canary_claim")
        elif self.stage is RunStage.CANARY_BEACON_PENDING:
            self.downstream_count += 1
            self.events.append("publish:canary_beacon")
        elif self.stage is RunStage.CANARY_RESULT_PENDING:
            self.downstream_count += 1
            self.events.append("publish:canary_result")
        elif self.stage is RunStage.FRESH_CANARY_AUDIT_PENDING:
            self.downstream_count += 1
            self.events.extend(
                (
                    "call:run_fresh_canary_audit_in_exec_subprocess",
                    "call:bridge_canary_audit",
                )
            )
        else:  # pragma: no cover - protects the fake contract itself
            raise AssertionError(f"unexpected fake advance: {self.stage}")
        self.stage = _NEXT[expected_stage]
        return self.inspect()

    def selection_bytes(self) -> bytes:
        self.selection_count += 1
        self.events.append("build:selection_from_audited_recovery")
        return self.exact_selection_bytes


def test_cli_surface_is_only_the_three_fixed_commands() -> None:
    parser = _build_parser()
    for command in ("status", "resume", "emit-selection"):
        assert parser.parse_args([command]).command == command
    for forbidden in (
        ["run"],
        ["resume", "--seed", "1"],
        ["resume", "--output", "/tmp/result"],
        ["resume", "--force"],
        ["resume", "--skip-tests"],
        ["resume", "--endpoint", "https://example.invalid"],
        ["resume", "--scientific"],
        ["resume", "--human-eeg"],
        ["resume", "--network"],
    ):
        with pytest.raises(SystemExit):
            parser.parse_args(forbidden)


def test_wrapper_starts_exact_site_free_python_before_importing_cfeg() -> None:
    wrapper = _WRAPPER.read_text(encoding="utf-8")
    assert wrapper.startswith("#!/bin/sh\nset -eu\n")
    assert (
        "exec /home/whwovy/califreeEEG/.venv/bin/python -I -B -S -c" in wrapper
    )
    assert "assert tuple(sys.path) == standard_library" in wrapper
    assert "sys.path[:] = [repository_src, *standard_library, site_packages]" in wrapper
    assert wrapper.index("assert flags.no_site == 1") < wrapper.index("import cfeg")
    assert wrapper.index("sys.path[:]") < wrapper.index("import cfeg")
    assert os.stat(_WRAPPER).st_mode & 0o111


@pytest.mark.parametrize("stage", tuple(_NEXT))
def test_resume_advances_at_most_one_durable_phase(stage: RunStage) -> None:
    facade = FakeFacade(stage)
    report = resume_once(facade)
    assert facade.advance_count == 1
    assert report["stage_before"] == stage.value
    assert report["stage"] == _NEXT[stage].value
    assert report["durable_phases_advanced"] == 1


def test_existing_artifact_is_reopened_then_next_phase_runs_without_republication() -> None:
    facade = FakeFacade(RunStage.CONTEXT_REFERENCE_PENDING)
    facade.events.append("reopen:existing_development_bundle_after_publish_return_crash")
    report = resume_once(facade)
    assert report["stage"] == RunStage.DEVELOPMENT_PENDING.value
    assert facade.events[0] == (
        "reopen:existing_development_bundle_after_publish_return_crash"
    )
    assert "publish:development_bundle" not in facade.events
    assert facade.events.count("publish:context_reference") == 1


def test_recovered_a_result_emits_proof_only_selection_bytes_without_execution() -> None:
    facade = FakeFacade(RunStage.SELECTION_HANDOFF)
    emitted = emit_selection_bytes(facade)
    assert emitted == _SELECTED_BYTES
    assert facade.selection_count == 1
    assert facade.execute_count == 0
    assert facade.events.count("recover:canonical_result_and_audited_proof") == 1
    assert facade.events[-1] == "build:selection_from_audited_recovery"


def test_development_no_go_is_terminal_and_never_calls_downstream() -> None:
    facade = FakeFacade(RunStage.DEVELOPMENT_NO_GO)
    report = resume_once(facade)
    assert report["terminal"] is True
    assert report["next_command"] is None
    assert report["durable_phases_advanced"] == 0
    assert facade.advance_count == 0
    assert facade.downstream_count == 0
    assert not any(event.startswith("publish:") for event in facade.events)


def test_resume_rejects_a_facade_that_skips_a_durable_boundary() -> None:
    class SkippingFacade(FakeFacade):
        def advance(self, expected_stage: RunStage) -> WorkflowInspection:
            assert expected_stage is RunStage.BUNDLE_PENDING
            self.advance_count += 1
            self.stage = RunStage.DEVELOPMENT_PENDING
            return self.inspect()

    with pytest.raises(RunnerError, match="invalid durable boundary"):
        resume_once(SkippingFacade(RunStage.BUNDLE_PENDING))


def test_test_evidence_runs_focused_then_full_before_single_publication() -> None:
    facade = FakeFacade(RunStage.TEST_EVIDENCE_PENDING)
    resume_once(facade)
    assert facade.events[1:4] == [
        "run:focused_v3",
        "run:repository_full",
        "publish:test_evidence",
    ]


def test_canary_sequence_is_fixed_across_one_phase_resumes() -> None:
    facade = FakeFacade(RunStage.CANARY_AUTHORIZATION_PENDING)
    for expected in (
        "publish:canary_authorization",
        "publish:canary_claim",
        "publish:canary_beacon",
        "publish:canary_result",
        "call:run_fresh_canary_audit_in_exec_subprocess",
    ):
        before = len(facade.events)
        resume_once(facade)
        new_events = facade.events[before:]
        assert expected in new_events
    assert facade.stage is RunStage.CANARY_PASSED
    assert facade.events.index("call:run_fresh_canary_audit_in_exec_subprocess") < (
        facade.events.index("call:bridge_canary_audit")
    )


def test_runner_uses_only_the_high_level_fresh_exec_api() -> None:
    source = _RUNNER.read_text(encoding="utf-8")
    assert "run_fresh_canary_audit_in_exec_subprocess" in source
    assert "_fresh_canary_audit_exec_child_main" not in source
    assert "observe_child_fresh_canary_audit" not in source
    assert "_publish_fresh_canary_audit_from_observed_exec" not in source


def test_runner_has_no_future_scientific_human_network_or_destructive_api() -> None:
    source = _RUNNER.read_text(encoding="utf-8")
    for forbidden in (
        "derive_scientific_seed(",
        "scientific_rng(",
        "publish_scientific_global_claim(",
        "assert_human_EEG_outcome_authorized(",
        ".unlink(",
        ".rmdir(",
        "os.remove(",
        "os.replace(",
    ):
        assert forbidden not in source


def test_emit_selection_cli_writes_raw_exact_bytes_and_no_text_envelope() -> None:
    facade = FakeFacade(RunStage.SELECTION_HANDOFF)
    text = io.StringIO()
    binary = io.BytesIO()
    assert (
        run_cli(
            ["emit-selection"],
            facade=facade,
            text_stdout=text,
            binary_stdout=binary,
        )
        == 0
    )
    assert text.getvalue() == ""
    assert binary.getvalue() == _SELECTED_BYTES


def test_status_is_read_only_and_reports_the_next_operator_command() -> None:
    facade = FakeFacade(RunStage.SELECTION_HANDOFF)
    report = status_report(facade)
    assert report == {
        "command": "status",
        "stage": "SELECTION_HANDOFF",
        "terminal": False,
        "selection_status": "SELECTED_METHOD_PROPOSED",
        "selected_grid_cell_id": "p3-nu_4-lambda_0p20",
        "next_command": "emit-selection",
        "durable_phases_advanced": 0,
    }
    assert facade.advance_count == 0
    assert facade.selection_count == 0
