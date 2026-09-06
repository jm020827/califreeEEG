"""Restartable, fail-closed operator for the frozen V3 synthetic workflow.

The command surface deliberately contains no scientific, human-EEG, network,
seed, path, overwrite, or test-command controls.  Durable state is inferred
only from the fixed artifact paths owned by the governance module.  Every
existing artifact is reopened through its authoritative validator before the
next write-once phase may run.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from enum import Enum
from pathlib import Path
from types import ModuleType
from typing import Any, Protocol, TextIO

V3_SOURCE_REPOSITORY = Path("/home/whwovy/califreeEEG")
V3_PYTHON_EXECUTABLE = V3_SOURCE_REPOSITORY / ".venv/bin/python"
V3_SITE_PACKAGES = V3_SOURCE_REPOSITORY / ".venv/lib/python3.10/site-packages"
V3_REPOSITORY_SRC = V3_SOURCE_REPOSITORY / "src"

_EXPECTED_STANDARD_LIBRARY_PATH = (
    "/usr/lib/python310.zip",
    "/usr/lib/python3.10",
    "/usr/lib/python3.10/lib-dynload",
)
EXPECTED_RUNNER_SYS_PATH = (
    os.fspath(V3_REPOSITORY_SRC),
    *_EXPECTED_STANDARD_LIBRARY_PATH,
    os.fspath(V3_SITE_PACKAGES),
)


class RunnerError(RuntimeError):
    """The fixed runner state or transition is invalid."""


class RunStage(str, Enum):
    BUNDLE_PENDING = "BUNDLE_PENDING"
    CONTEXT_REFERENCE_PENDING = "CONTEXT_REFERENCE_PENDING"
    DEVELOPMENT_PENDING = "DEVELOPMENT_PENDING"
    DEVELOPMENT_NO_GO = "DEVELOPMENT_NO_GO"
    SELECTION_HANDOFF = "SELECTION_HANDOFF"
    TEST_EVIDENCE_PENDING = "TEST_EVIDENCE_PENDING"
    CANARY_AUTHORIZATION_PENDING = "CANARY_AUTHORIZATION_PENDING"
    CANARY_CLAIM_PENDING = "CANARY_CLAIM_PENDING"
    CANARY_BEACON_PENDING = "CANARY_BEACON_PENDING"
    CANARY_RESULT_PENDING = "CANARY_RESULT_PENDING"
    FRESH_CANARY_AUDIT_PENDING = "FRESH_CANARY_AUDIT_PENDING"
    CANARY_PASSED = "CANARY_PASSED"


_TERMINAL_STAGES = frozenset({RunStage.DEVELOPMENT_NO_GO, RunStage.CANARY_PASSED})
_ADVANCEABLE_STAGES = frozenset(
    {
        RunStage.BUNDLE_PENDING,
        RunStage.CONTEXT_REFERENCE_PENDING,
        RunStage.DEVELOPMENT_PENDING,
        RunStage.TEST_EVIDENCE_PENDING,
        RunStage.CANARY_AUTHORIZATION_PENDING,
        RunStage.CANARY_CLAIM_PENDING,
        RunStage.CANARY_BEACON_PENDING,
        RunStage.CANARY_RESULT_PENDING,
        RunStage.FRESH_CANARY_AUDIT_PENDING,
    }
)
_RESUME_TRANSITIONS = {
    RunStage.BUNDLE_PENDING: frozenset({RunStage.CONTEXT_REFERENCE_PENDING}),
    RunStage.CONTEXT_REFERENCE_PENDING: frozenset({RunStage.DEVELOPMENT_PENDING}),
    RunStage.DEVELOPMENT_PENDING: frozenset(
        {RunStage.DEVELOPMENT_NO_GO, RunStage.SELECTION_HANDOFF}
    ),
    RunStage.TEST_EVIDENCE_PENDING: frozenset(
        {RunStage.CANARY_AUTHORIZATION_PENDING}
    ),
    RunStage.CANARY_AUTHORIZATION_PENDING: frozenset({RunStage.CANARY_CLAIM_PENDING}),
    RunStage.CANARY_CLAIM_PENDING: frozenset({RunStage.CANARY_BEACON_PENDING}),
    RunStage.CANARY_BEACON_PENDING: frozenset({RunStage.CANARY_RESULT_PENDING}),
    RunStage.CANARY_RESULT_PENDING: frozenset({RunStage.FRESH_CANARY_AUDIT_PENDING}),
    RunStage.FRESH_CANARY_AUDIT_PENDING: frozenset({RunStage.CANARY_PASSED}),
}


@dataclass(frozen=True, slots=True)
class WorkflowInspection:
    """Small outcome-free view of the exact validated workflow prefix."""

    stage: RunStage
    selection_status: str | None = None
    selected_grid_cell_id: str | None = None

    def __post_init__(self) -> None:
        if type(self.stage) is not RunStage:
            raise TypeError("stage must be an exact RunStage")
        if self.selection_status not in {
            None,
            "SELECTED_METHOD_PROPOSED",
            "DEVELOPMENT_NO_GO",
        }:
            raise RunnerError("unknown development selection status")
        if self.selection_status == "SELECTED_METHOD_PROPOSED":
            if not isinstance(self.selected_grid_cell_id, str) or not self.selected_grid_cell_id:
                raise RunnerError("selected development state requires one grid-cell ID")
        elif self.selected_grid_cell_id is not None:
            raise RunnerError("non-selected development state cannot carry a grid-cell ID")

    @property
    def terminal(self) -> bool:
        return self.stage in _TERMINAL_STAGES

    @property
    def next_command(self) -> str | None:
        if self.stage is RunStage.SELECTION_HANDOFF:
            return "emit-selection"
        if self.terminal:
            return None
        return "resume"


class WorkflowFacade(Protocol):
    """Narrow seam used by the state machine and by side-effect-free unit fakes."""

    def inspect(self) -> WorkflowInspection:
        """Validate and reopen the complete existing durable prefix."""

    def advance(self, expected_stage: RunStage) -> WorkflowInspection:
        """Create at most the one durable artifact implied by ``expected_stage``."""

    def selection_bytes(self) -> bytes:
        """Return exact selected-method artifact bytes without writing them."""


@dataclass(slots=True)
class _ValidatedGraph:
    presence: Mapping[str, bool]
    bundle: Any | None = None
    snapshot: Any | None = None
    reference_rng: Any | None = None
    development_rng: Any | None = None
    reference_value: Any | None = None
    reference_proof: Any | None = None
    context_reference: Any | None = None
    development_result: Any | None = None
    audited_development_result: Any | None = None
    recovery: Any | None = None
    selected_method_freeze: Any | None = None
    test_evidence: Any | None = None
    canary_authorization: Any | None = None
    canary_claim: Any | None = None
    canary_seed: Any | None = None
    canary_beacon: Any | None = None
    published_canary_result: Any | None = None
    fresh_canary_audit: Any | None = None


def _assert_production_bootstrap() -> None:
    """Reject every production entry that did not start from the tracked wrapper."""

    flags = sys.flags
    if (
        flags.isolated != 1
        or flags.ignore_environment != 1
        or flags.dont_write_bytecode != 1
        or flags.no_user_site != 1
        or flags.no_site != 1
    ):
        raise RunnerError("V3 production runner requires -I -B -S from interpreter startup")
    if sys.executable != os.fspath(V3_PYTHON_EXECUTABLE):
        raise RunnerError("V3 production runner executable is not the canonical .venv Python")
    if tuple(sys.path) != EXPECTED_RUNNER_SYS_PATH:
        raise RunnerError("V3 production runner sys.path is not the exact site-free bootstrap")
    expected_module = V3_REPOSITORY_SRC / "cfeg/metadata_calibration_v3_runner.py"
    if Path(__file__).resolve() != expected_module:
        raise RunnerError("V3 production runner module is not loaded from canonical source")


def _rfc3339_now_after(previous: str | None = None) -> str:
    now = datetime.now(timezone.utc).replace(microsecond=0)
    if previous is not None:
        if not isinstance(previous, str) or not previous.endswith("Z"):
            raise RunnerError("prior artifact timestamp is not canonical UTC")
        try:
            prior = datetime.fromisoformat(previous[:-1] + "+00:00")
        except ValueError as exc:
            raise RunnerError("prior artifact timestamp is malformed") from exc
        if prior.tzinfo is None:
            raise RunnerError("prior artifact timestamp lacks a timezone")
        if now <= prior:
            now = prior.astimezone(timezone.utc) + timedelta(seconds=1)
    return now.strftime("%Y-%m-%dT%H:%M:%S.000Z")


def _artifact_payload(capability: Any, name: str) -> Mapping[str, Any]:
    payload = getattr(capability, "_validated_payload", None)
    if not isinstance(payload, Mapping):
        raise RunnerError(f"{name} capability lacks validated payload bytes")
    return payload


class ProductionFacade:
    """Lazy adapter around the V3 core and governance public APIs."""

    _ARTIFACT_ORDER = (
        "bundle",
        "context_reference",
        "development_result",
        "selected_method_freeze",
        "test_evidence",
        "canary_authorization",
        "canary_claim",
        "canary_beacon",
        "canary_result",
        "fresh_canary_audit",
    )

    def __init__(self) -> None:
        _assert_production_bootstrap()
        # These imports are intentionally delayed until after the site-free
        # interpreter, executable, module path, and complete sys.path are checked.
        from cfeg import metadata_calibration_v3_governance as governance
        from cfeg.analysis import metadata_calibration_v3_synthetic as synthetic
        from cfeg.models import metadata_calibration_v3 as model

        self._governance: ModuleType = governance
        self._model: ModuleType = model
        self._synthetic: ModuleType = synthetic
        self._inspection: WorkflowInspection | None = None
        self._graph: _ValidatedGraph | None = None

    def _artifact_paths(self) -> Mapping[str, Path]:
        governance = self._governance
        return {
            "bundle": Path(governance.DEVELOPMENT_BUNDLE_CANONICAL_PATH),
            "context_reference": Path(governance.CONTEXT_REFERENCE_CANONICAL_PATH),
            "development_result": Path(governance.DEVELOPMENT_RESULT_CANONICAL_PATH),
            "selected_method_freeze": (
                Path(governance.V3_SOURCE_REPOSITORY)
                / governance.SELECTED_METHOD_FREEZE_REPOSITORY_PATH
            ),
            "test_evidence": Path(governance.TEST_EVIDENCE_CANONICAL_PATH),
            "canary_authorization": Path(governance.CANARY_CANONICAL_ROOT)
            / "authorization.json",
            "canary_claim": Path(governance.CANARY_CANONICAL_ROOT) / "claim.json",
            "canary_beacon": Path(governance.CANARY_CANONICAL_ROOT) / "beacon.json",
            "canary_result": Path(governance.CANARY_RESULT_CANONICAL_PATH),
            "fresh_canary_audit": Path(governance.CANARY_FRESH_AUDIT_CANONICAL_PATH),
        }

    def _presence(self) -> Mapping[str, bool]:
        return {
            name: os.path.lexists(os.fspath(path))
            for name, path in self._artifact_paths().items()
        }

    def _require_no_gap_before(self, presence: Mapping[str, bool], name: str) -> None:
        index = self._ARTIFACT_ORDER.index(name)
        later = self._ARTIFACT_ORDER[index + 1 :]
        unexpected = tuple(item for item in later if presence[item])
        if unexpected:
            joined = ",".join(unexpected)
            raise RunnerError(f"durable artifact prefix has a gap before {name}: {joined}")

    def _cache(
        self,
        stage: RunStage,
        graph: _ValidatedGraph,
        *,
        selection_status: str | None = None,
        selected_grid_cell_id: str | None = None,
    ) -> WorkflowInspection:
        inspection = WorkflowInspection(
            stage=stage,
            selection_status=selection_status,
            selected_grid_cell_id=selected_grid_cell_id,
        )
        self._graph = graph
        self._inspection = inspection
        return inspection

    def _reopen_a_context(self, graph: _ValidatedGraph) -> None:
        model = self._model
        synthetic = self._synthetic
        governance = self._governance
        reference_rng, development_rng = model.issue_bundle_bound_rng_authorities(
            graph.bundle,
            expected_commit=graph.bundle.clean_commit,
            expected_tree=graph.bundle.clean_tree,
        )
        payload = synthetic.replay_context_reference_payload(rng_authority=reference_rng)
        reference = model.validate_context_reference_payload(payload)
        proof = model.validate_context_reference_for_publication(
            payload,
            rng_authority=reference_rng,
        )
        artifact = governance.validate_context_reference(
            development_bundle=graph.bundle,
            rng_authority=reference_rng,
            core_capability=proof,
        )
        graph.reference_rng = reference_rng
        graph.development_rng = development_rng
        graph.reference_value = reference
        graph.reference_proof = proof
        graph.context_reference = artifact

    def _reopen_b_selection_graph(self, graph: _ValidatedGraph) -> None:
        governance = self._governance
        snapshot = governance.capture_clean_source_snapshot(governance.V3_SOURCE_REPOSITORY)
        recovery = governance.validate_bundle_selected_delta_recovery(
            development_bundle=graph.bundle,
            snapshot=snapshot,
        )
        context = governance.reopen_context_reference(
            development_bundle=graph.bundle,
            recovery_capability=recovery,
        )
        result = governance.reopen_development_result(
            development_bundle=graph.bundle,
            context_reference=context,
            recovery_capability=recovery,
        )
        selected = governance.reopen_selected_method_freeze(
            development_bundle=graph.bundle,
            development_result=result,
            recovery_capability=recovery,
        )
        result_payload = _artifact_payload(result, "development result")
        if result_payload.get("selection_status") != "SELECTED_METHOD_PROPOSED":
            raise RunnerError("selected B is impossible after DEVELOPMENT_NO_GO")
        graph.snapshot = snapshot
        graph.recovery = recovery
        graph.context_reference = context
        graph.development_result = result
        graph.selected_method_freeze = selected

    def _reopen_canary_seed(self, graph: _ValidatedGraph) -> Any:
        if graph.snapshot is None:
            raise RunnerError("canary seed requires selected-source snapshot")
        if graph.canary_seed is None:
            graph.canary_seed = self._governance.reopen_historical_canary_seed(
                snapshot=graph.snapshot
            )
        return graph.canary_seed

    def _bridge(self, graph: _ValidatedGraph) -> None:
        governance = self._governance
        required = (
            graph.bundle,
            graph.development_result,
            graph.selected_method_freeze,
            graph.test_evidence,
            graph.canary_authorization,
            graph.canary_claim,
            graph.canary_beacon,
            graph.published_canary_result,
            graph.fresh_canary_audit,
        )
        if any(item is None for item in required):
            raise RunnerError("fresh-audit bridge graph is incomplete")
        seed = self._reopen_canary_seed(graph)
        scientific = governance.ScientificLifecycle()
        scientific = scientific.advance(
            governance.ScientificState.BUNDLE_FROZEN,
            authority=graph.bundle,
        )
        scientific = scientific.advance(
            governance.ScientificState.DEVELOPMENT_COMPLETE,
            authority=graph.development_result,
        )
        scientific = scientific.advance(
            governance.ScientificState.SELECTED_METHOD_FROZEN,
            authority=graph.selected_method_freeze,
        )
        scientific = scientific.advance(
            governance.ScientificState.VERIFIED,
            authority=graph.test_evidence,
        )
        canary = governance.CanaryLifecycle()
        canary = canary.advance(
            governance.CanaryState.CANARY_AUTHORIZED,
            authority=graph.canary_authorization,
        )
        canary = canary.advance(
            governance.CanaryState.CANARY_CLAIMED,
            authority=graph.canary_claim,
        )
        canary = canary.advance(
            governance.CanaryState.CANARY_BEACON_BOUND,
            authority=graph.canary_beacon,
        )
        canary = canary.advance(
            governance.CanaryState.CANARY_PAYLOAD_VALIDATED,
            authority=seed,
        )
        canary = canary.advance(
            governance.CanaryState.CANARY_RESULT_PUBLISHED,
            authority=graph.published_canary_result,
        )
        canary = canary.advance(
            governance.CanaryState.CANARY_FRESH_AUDITED,
            authority=graph.fresh_canary_audit,
        )
        bridged = governance.bridge_canary_audit(
            scientific,
            canary,
            graph.fresh_canary_audit,
        )
        if bridged.state is not governance.ScientificState.CANARY_PASSED:
            raise RunnerError("fresh canary audit did not reach CANARY_PASSED")

    def inspect(self) -> WorkflowInspection:
        governance = self._governance
        presence = self._presence()
        graph = _ValidatedGraph(presence=presence)

        if not presence["bundle"]:
            self._require_no_gap_before(presence, "bundle")
            return self._cache(RunStage.BUNDLE_PENDING, graph)
        graph.bundle = governance.reopen_development_bundle()

        if not presence["context_reference"]:
            self._require_no_gap_before(presence, "context_reference")
            return self._cache(RunStage.CONTEXT_REFERENCE_PENDING, graph)
        if not presence["development_result"]:
            self._require_no_gap_before(presence, "development_result")
            self._reopen_a_context(graph)
            return self._cache(RunStage.DEVELOPMENT_PENDING, graph)

        if not presence["selected_method_freeze"]:
            self._require_no_gap_before(presence, "selected_method_freeze")
            self._reopen_a_context(graph)
            result, audited = governance.recover_canonical_development_result_at_bundle_source(
                development_bundle=graph.bundle,
                context_reference=graph.context_reference,
            )
            graph.development_result = result
            graph.audited_development_result = audited
            selection_status = getattr(audited, "selection_status", None)
            selected_grid_cell_id = getattr(audited, "selected_grid_cell_id", None)
            if selection_status == "DEVELOPMENT_NO_GO":
                scientific = governance.ScientificLifecycle()
                scientific = scientific.advance(
                    governance.ScientificState.BUNDLE_FROZEN,
                    authority=graph.bundle,
                )
                scientific = scientific.advance(
                    governance.ScientificState.DEVELOPMENT_COMPLETE,
                    authority=result,
                )
                scientific = scientific.advance(
                    governance.ScientificState.DEVELOPMENT_NO_GO,
                    authority=result,
                )
                if scientific.state is not governance.ScientificState.DEVELOPMENT_NO_GO:
                    raise RunnerError("development no-go lifecycle validation failed")
                return self._cache(
                    RunStage.DEVELOPMENT_NO_GO,
                    graph,
                    selection_status=selection_status,
                )
            if selection_status != "SELECTED_METHOD_PROPOSED":
                raise RunnerError("canonical development result has unknown selection status")
            return self._cache(
                RunStage.SELECTION_HANDOFF,
                graph,
                selection_status=selection_status,
                selected_grid_cell_id=selected_grid_cell_id,
            )

        self._reopen_b_selection_graph(graph)
        selected_id = _artifact_payload(
            graph.development_result, "development result"
        )["selected_grid_cell_id"]

        if not presence["test_evidence"]:
            self._require_no_gap_before(presence, "test_evidence")
            return self._cache(
                RunStage.TEST_EVIDENCE_PENDING,
                graph,
                selection_status="SELECTED_METHOD_PROPOSED",
                selected_grid_cell_id=selected_id,
            )
        graph.test_evidence = governance.reopen_test_evidence(
            selected_method_freeze=graph.selected_method_freeze,
            snapshot=graph.snapshot,
        )

        if not presence["canary_authorization"]:
            self._require_no_gap_before(presence, "canary_authorization")
            return self._cache(
                RunStage.CANARY_AUTHORIZATION_PENDING,
                graph,
                selection_status="SELECTED_METHOD_PROPOSED",
                selected_grid_cell_id=selected_id,
            )
        graph.canary_authorization = governance.validate_canary_authorization(
            selected_method_freeze=graph.selected_method_freeze,
            test_evidence=graph.test_evidence,
        )

        if not presence["canary_claim"]:
            self._require_no_gap_before(presence, "canary_claim")
            return self._cache(
                RunStage.CANARY_CLAIM_PENDING,
                graph,
                selection_status="SELECTED_METHOD_PROPOSED",
                selected_grid_cell_id=selected_id,
            )
        graph.canary_claim = governance.validate_canary_claim(
            authorization=graph.canary_authorization
        )

        if not presence["canary_beacon"]:
            self._require_no_gap_before(presence, "canary_beacon")
            return self._cache(
                RunStage.CANARY_BEACON_PENDING,
                graph,
                selection_status="SELECTED_METHOD_PROPOSED",
                selected_grid_cell_id=selected_id,
            )
        seed = self._reopen_canary_seed(graph)
        graph.canary_beacon = governance.validate_canary_beacon(
            claim=graph.canary_claim,
            seed=seed,
        )

        if not presence["canary_result"]:
            self._require_no_gap_before(presence, "canary_result")
            return self._cache(
                RunStage.CANARY_RESULT_PENDING,
                graph,
                selection_status="SELECTED_METHOD_PROPOSED",
                selected_grid_cell_id=selected_id,
            )
        graph.published_canary_result = governance.reopen_published_canary_result(
            beacon=graph.canary_beacon,
            seed=seed,
        )

        if not presence["fresh_canary_audit"]:
            return self._cache(
                RunStage.FRESH_CANARY_AUDIT_PENDING,
                graph,
                selection_status="SELECTED_METHOD_PROPOSED",
                selected_grid_cell_id=selected_id,
            )
        # The sole high-level fresh-exec API first reopens an existing exact
        # artifact; it cannot republish because the path is already consumed.
        graph.fresh_canary_audit = (
            governance.run_fresh_canary_audit_in_exec_subprocess()
        )
        self._bridge(graph)
        return self._cache(
            RunStage.CANARY_PASSED,
            graph,
            selection_status="SELECTED_METHOD_PROPOSED",
            selected_grid_cell_id=selected_id,
        )

    def _require_cached_stage(self, expected_stage: RunStage) -> _ValidatedGraph:
        if expected_stage not in _ADVANCEABLE_STAGES:
            raise RunnerError(f"stage cannot create a durable artifact: {expected_stage.value}")
        if self._inspection is None or self._graph is None:
            raise RunnerError("advance requires a preceding exact inspection")
        if self._inspection.stage is not expected_stage:
            raise RunnerError(
                f"workflow changed before advance: expected {expected_stage.value}, "
                f"observed {self._inspection.stage.value}"
            )
        return self._graph

    def _publish_bundle(self, graph: _ValidatedGraph) -> None:
        del graph
        governance = self._governance
        snapshot = governance.capture_clean_source_snapshot(governance.V3_SOURCE_REPOSITORY)
        focused = governance.run_observed_test("focused_v3", snapshot=snapshot)
        frozen = {
            path: (Path(governance.V3_SOURCE_REPOSITORY) / path).read_bytes()
            for path in governance.FROZEN_FILE_SHA256
        }
        governance.publish_development_bundle(
            snapshot=snapshot,
            frozen_file_bytes=frozen,
            created_at_UTC=_rfc3339_now_after(),
            focused_test_run=focused,
        )
        governance.validate_development_bundle(
            snapshot=snapshot,
            frozen_file_bytes=frozen,
            focused_test_run=focused,
        )

    def _publish_context_reference(self, graph: _ValidatedGraph) -> None:
        if graph.bundle is None:
            raise RunnerError("context-reference phase lacks development bundle")
        model = self._model
        synthetic = self._synthetic
        governance = self._governance
        reference_rng, _ = model.issue_bundle_bound_rng_authorities(
            graph.bundle,
            expected_commit=graph.bundle.clean_commit,
            expected_tree=graph.bundle.clean_tree,
        )
        payload = synthetic.replay_context_reference_payload(rng_authority=reference_rng)
        proof = model.validate_context_reference_for_publication(
            payload,
            rng_authority=reference_rng,
        )
        governance.publish_context_reference(
            payload,
            development_bundle=graph.bundle,
            rng_authority=reference_rng,
            core_capability=proof,
        )
        governance.validate_context_reference(
            development_bundle=graph.bundle,
            rng_authority=reference_rng,
            core_capability=proof,
        )

    def _publish_development_result(self, graph: _ValidatedGraph) -> None:
        if any(
            item is None
            for item in (
                graph.bundle,
                graph.development_rng,
                graph.reference_value,
                graph.reference_proof,
                graph.context_reference,
            )
        ):
            raise RunnerError("development phase lacks an exact validated A graph")
        synthetic = self._synthetic
        governance = self._governance
        contract = synthetic.validate_synthetic_contract()
        synthetic.validate_development_contract(contract)
        payload, proof = synthetic.execute_complete_development(
            contract,
            graph.reference_value,
            rng_authority=graph.development_rng,
            validated_reference=graph.reference_proof,
            context_reference_file_sha256=graph.context_reference.file_sha256,
        )
        governance.publish_development_result(
            payload,
            development_bundle=graph.bundle,
            context_reference=graph.context_reference,
            development_rng_authority=graph.development_rng,
            validated_context_reference=graph.reference_proof,
            core_capability=proof,
        )
        governance.validate_development_result(
            development_bundle=graph.bundle,
            context_reference=graph.context_reference,
            development_rng_authority=graph.development_rng,
            validated_context_reference=graph.reference_proof,
            core_capability=proof,
        )

    def _publish_test_evidence(self, graph: _ValidatedGraph) -> None:
        if graph.snapshot is None or graph.selected_method_freeze is None:
            raise RunnerError("test-evidence phase lacks exact selected B")
        governance = self._governance
        focused = governance.run_observed_test("focused_v3", snapshot=graph.snapshot)
        full = governance.run_observed_test("repository_full", snapshot=graph.snapshot)
        runs = (focused, full)
        payload = governance.seal_test_evidence(
            selected_method_freeze=graph.selected_method_freeze,
            observed_runs=runs,
            snapshot=graph.snapshot,
            recorded_at_UTC=_rfc3339_now_after(),
        )
        governance.publish_test_evidence(
            payload,
            selected_method_freeze=graph.selected_method_freeze,
            observed_runs=runs,
            snapshot=graph.snapshot,
        )
        governance.validate_test_evidence(
            selected_method_freeze=graph.selected_method_freeze,
            observed_runs=runs,
            snapshot=graph.snapshot,
        )

    def _publish_canary_authorization(self, graph: _ValidatedGraph) -> None:
        if graph.selected_method_freeze is None or graph.test_evidence is None:
            raise RunnerError("canary authorization phase lacks selected test evidence")
        governance = self._governance
        tests = _artifact_payload(graph.test_evidence, "test evidence")
        payload = governance.build_canary_authorization(
            selected_method_freeze=graph.selected_method_freeze,
            test_evidence=graph.test_evidence,
            authorized_at_UTC=_rfc3339_now_after(str(tests["recorded_at_UTC"])),
        )
        governance.publish_canary_authorization(
            payload,
            selected_method_freeze=graph.selected_method_freeze,
            test_evidence=graph.test_evidence,
        )
        governance.validate_canary_authorization(
            selected_method_freeze=graph.selected_method_freeze,
            test_evidence=graph.test_evidence,
        )

    def _publish_canary_claim(self, graph: _ValidatedGraph) -> None:
        if graph.canary_authorization is None:
            raise RunnerError("canary claim phase lacks authorization")
        governance = self._governance
        authorization = _artifact_payload(
            graph.canary_authorization, "canary authorization"
        )
        payload = governance.build_canary_claim(
            graph.canary_authorization,
            claimed_at_UTC=_rfc3339_now_after(str(authorization["authorized_at_UTC"])),
        )
        governance.publish_canary_claim(
            payload,
            authorization=graph.canary_authorization,
        )
        governance.validate_canary_claim(authorization=graph.canary_authorization)

    def _publish_canary_beacon(self, graph: _ValidatedGraph) -> None:
        if graph.canary_claim is None:
            raise RunnerError("canary beacon phase lacks claim")
        governance = self._governance
        seed = self._reopen_canary_seed(graph)
        payload = governance.build_canary_beacon(graph.canary_claim, seed)
        governance.publish_canary_beacon(
            payload,
            claim=graph.canary_claim,
            seed=seed,
        )
        governance.validate_canary_beacon(claim=graph.canary_claim, seed=seed)

    def _publish_canary_result(self, graph: _ValidatedGraph) -> None:
        if graph.canary_beacon is None:
            raise RunnerError("canary result phase lacks beacon")
        governance = self._governance
        seed = self._reopen_canary_seed(graph)
        probe = governance.canary_probe_digest(
            seed,
            prevalidation_trace=tuple(governance.CanaryState)[:4],
        )
        governance.publish_canary_result(
            beacon=graph.canary_beacon,
            seed=seed,
            probe_digest_sha256=probe,
        )

    def _publish_fresh_canary_audit(self, graph: _ValidatedGraph) -> None:
        if graph.published_canary_result is None:
            raise RunnerError("fresh audit phase lacks published canary result")
        # Do not call, serialize, or reproduce any private child protocol here.
        graph.fresh_canary_audit = (
            self._governance.run_fresh_canary_audit_in_exec_subprocess()
        )
        self._bridge(graph)

    def advance(self, expected_stage: RunStage) -> WorkflowInspection:
        graph = self._require_cached_stage(expected_stage)
        dispatch = {
            RunStage.BUNDLE_PENDING: self._publish_bundle,
            RunStage.CONTEXT_REFERENCE_PENDING: self._publish_context_reference,
            RunStage.DEVELOPMENT_PENDING: self._publish_development_result,
            RunStage.TEST_EVIDENCE_PENDING: self._publish_test_evidence,
            RunStage.CANARY_AUTHORIZATION_PENDING: self._publish_canary_authorization,
            RunStage.CANARY_CLAIM_PENDING: self._publish_canary_claim,
            RunStage.CANARY_BEACON_PENDING: self._publish_canary_beacon,
            RunStage.CANARY_RESULT_PENDING: self._publish_canary_result,
            RunStage.FRESH_CANARY_AUDIT_PENDING: self._publish_fresh_canary_audit,
        }
        dispatch[expected_stage](graph)
        # Reopen and validate the just-published artifact from its canonical
        # location.  If publication returned only partially, this fails closed.
        return self.inspect()

    def selection_bytes(self) -> bytes:
        inspection = self.inspect()
        if inspection.stage is not RunStage.SELECTION_HANDOFF or self._graph is None:
            raise RunnerError("selection bytes are available only at SELECTION_HANDOFF")
        graph = self._graph
        if graph.development_result is None or graph.audited_development_result is None:
            raise RunnerError("selection handoff lacks exact A recovery proof")
        selected = self._governance.build_selected_method_freeze_from_canonical_development_result(
            development_result=graph.development_result,
            audited_development_result=graph.audited_development_result,
        )
        return self._governance.artifact_bytes(selected)


def status_report(facade: WorkflowFacade) -> Mapping[str, Any]:
    inspection = facade.inspect()
    return {
        "command": "status",
        "stage": inspection.stage.value,
        "terminal": inspection.terminal,
        "selection_status": inspection.selection_status,
        "selected_grid_cell_id": inspection.selected_grid_cell_id,
        "next_command": inspection.next_command,
        "durable_phases_advanced": 0,
    }


def resume_once(facade: WorkflowFacade) -> Mapping[str, Any]:
    before = facade.inspect()
    if before.stage is RunStage.SELECTION_HANDOFF or before.terminal:
        after = before
        advanced = 0
    else:
        if before.stage not in _ADVANCEABLE_STAGES:
            raise RunnerError(f"unhandled workflow stage: {before.stage.value}")
        after = facade.advance(before.stage)
        if after.stage not in _RESUME_TRANSITIONS[before.stage]:
            raise RunnerError(
                f"resume crossed an invalid durable boundary: "
                f"{before.stage.value}->{after.stage.value}"
            )
        advanced = 1
    return {
        "command": "resume",
        "stage_before": before.stage.value,
        "stage": after.stage.value,
        "terminal": after.terminal,
        "selection_status": after.selection_status,
        "selected_grid_cell_id": after.selected_grid_cell_id,
        "next_command": after.next_command,
        "durable_phases_advanced": advanced,
    }


def emit_selection_bytes(facade: WorkflowFacade) -> bytes:
    inspection = facade.inspect()
    if inspection.stage is not RunStage.SELECTION_HANDOFF:
        raise RunnerError("emit-selection requires the exact A selection handoff")
    data = facade.selection_bytes()
    if not isinstance(data, bytes) or not data or not data.endswith(b"\n"):
        raise RunnerError("selection handoff did not return canonical artifact bytes")
    return data


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="run_metadata_calibration_v3",
        description="Resume exactly one frozen V3 synthetic/governance phase.",
        allow_abbrev=False,
    )
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("status", allow_abbrev=False)
    commands.add_parser("resume", allow_abbrev=False)
    commands.add_parser("emit-selection", allow_abbrev=False)
    return parser


def _report_bytes(value: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(
            dict(value),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("utf-8")
        + b"\n"
    )


def run_cli(
    argv: Sequence[str],
    *,
    facade: WorkflowFacade,
    text_stdout: TextIO,
    binary_stdout: Any,
) -> int:
    args = _build_parser().parse_args(tuple(argv))
    if args.command == "status":
        text_stdout.write(_report_bytes(status_report(facade)).decode("ascii"))
    elif args.command == "resume":
        text_stdout.write(_report_bytes(resume_once(facade)).decode("ascii"))
    elif args.command == "emit-selection":
        binary_stdout.write(emit_selection_bytes(facade))
        binary_stdout.flush()
    else:  # pragma: no cover - argparse owns the closed command vocabulary
        raise RunnerError("unknown runner command")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    _assert_production_bootstrap()
    arguments = tuple(sys.argv[1:] if argv is None else argv)
    facade = ProductionFacade()
    try:
        return run_cli(
            arguments,
            facade=facade,
            text_stdout=sys.stdout,
            binary_stdout=sys.stdout.buffer,
        )
    except (RunnerError, OSError, ValueError, TypeError, RuntimeError) as exc:
        error = {
            "command": arguments[0] if arguments else None,
            "error": type(exc).__name__,
            "message": str(exc),
            "status": "FAIL_CLOSED",
        }
        sys.stderr.write(_report_bytes(error).decode("ascii"))
        return 2


__all__ = [
    "EXPECTED_RUNNER_SYS_PATH",
    "ProductionFacade",
    "RunStage",
    "RunnerError",
    "WorkflowFacade",
    "WorkflowInspection",
    "emit_selection_bytes",
    "main",
    "resume_once",
    "run_cli",
    "status_report",
]
