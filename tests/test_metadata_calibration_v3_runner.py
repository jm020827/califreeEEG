from __future__ import annotations

import ast
import hashlib
import importlib.util
import io
import json
import os
import py_compile
import shlex
import shutil
import stat
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from types import MappingProxyType, SimpleNamespace

import pytest

from cfeg.metadata_calibration_v3_runner import (
    _CANDIDATE_ID,
    _DEVELOPMENT_BUNDLE_FIELDS,
    _DEVELOPMENT_BUNDLE_SCHEMA,
    _EXPECTED_GOVERNED_ENVIRONMENT,
    _PYTHON_SITE_INVENTORY_SCHEMA,
    ProductionFacade,
    RunnerError,
    RunStage,
    WorkflowInspection,
    _build_parser,
    _canonical_json_bytes,
    _capture_python_record,
    _capture_python_site_inventory,
    _capture_source_facts,
    _detach_retired_v1_context_payload,
    _PreflightConfig,
    _rfc3339_now_after,
    _validate_existing_bundle,
    emit_selection_bytes,
    resume_once,
    run_cli,
    status_report,
)

_REPOSITORY = Path(__file__).resolve().parents[1]
_RUNNER = _REPOSITORY / "src/cfeg/metadata_calibration_v3_runner.py"
_GOVERNANCE = _REPOSITORY / "src/cfeg/metadata_calibration_v3_governance.py"
_WRAPPER = _REPOSITORY / "scripts/run_metadata_calibration_v3"
_SELECTED_BYTES = (
    b'{"payload_sha256":"aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",'
    b'"schema":"fake-selected.v1"}\n'
)
_PYTHON = Path("/home/whwovy/califreeEEG/.venv/bin/python")
_GIT = Path("/usr/bin/git")
_STANDARD_LIBRARY = (
    "/usr/lib/python310.zip",
    "/usr/lib/python3.10",
    "/usr/lib/python3.10/lib-dynload",
)
_RUNNER_REPOSITORY_PATH = "src/cfeg/metadata_calibration_v3_runner.py"
_GOVERNANCE_ENVIRONMENT = {
    "BLIS_NUM_THREADS": "1",
    "LANG": "C.UTF-8",
    "LC_ALL": "C.UTF-8",
    "MKL_DYNAMIC": "FALSE",
    "MKL_NUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
    "OMP_DYNAMIC": "FALSE",
    "OMP_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "PATH": "/usr/bin:/bin",
    "PYTHONNOUSERSITE": "1",
    "PYTHONDONTWRITEBYTECODE": "1",
    "PYTEST_DISABLE_PLUGIN_AUTOLOAD": "1",
    "VECLIB_MAXIMUM_THREADS": "1",
}

_PREFLIGHT_HARNESS = r"""
import importlib
import pathlib
import runpy
import sys

runner_source, repository, site, bundle, tracked_runner, module_name = sys.argv[1:]
namespace = runpy.run_path(runner_source, run_name="v3_preflight_harness")
config = namespace["_PreflightConfig"](
    repository=pathlib.Path(repository),
    python_executable=pathlib.Path(sys.executable),
    site_packages=pathlib.Path(site),
    repository_src=pathlib.Path(repository) / "src",
    runner_path=pathlib.Path(tracked_runner),
    runner_repository_path="src/cfeg/metadata_calibration_v3_runner.py",
    bundle_path=pathlib.Path(bundle),
    standard_library_path=(
        "/usr/lib/python310.zip",
        "/usr/lib/python3.10",
        "/usr/lib/python3.10/lib-dynload",
    ),
    protected_source_roots=("src",),
    root_import_controls=frozenset(),
    selected_method_relative_path=(
        "configs/governance/metadata_calibration_v3_selected_method_freeze.json"
    ),
    require_direct_invocation=False,
)
try:
    facts = namespace["_run_preimport_preflight"](
        config,
        executing_file=pathlib.Path(runner_source),
    )
except Exception as exc:
    print(f"REJECTED:{type(exc).__name__}:{exc}")
    raise SystemExit(0)
sys.path[:] = [
    *facts.config.standard_library_path,
    str(facts.config.repository_src),
    str(facts.config.site_packages),
]
importlib.import_module(module_name)
print("PREFLIGHT_PASSED")
"""


def _git(repository: Path, *arguments: str) -> str:
    completed = subprocess.run(
        [_GIT, "-C", repository, *arguments],
        check=True,
        capture_output=True,
        text=True,
        env={
            "GIT_CONFIG_GLOBAL": "/dev/null",
            "GIT_CONFIG_NOSYSTEM": "1",
            "GIT_TERMINAL_PROMPT": "0",
            "LANG": "C.UTF-8",
            "LC_ALL": "C.UTF-8",
            "PATH": "/usr/bin:/bin",
        },
    )
    return completed.stdout.strip()


def _make_test_repository(tmp_path: Path, *, module_source: str) -> tuple[Path, Path, Path]:
    repository = tmp_path / "repository"
    site = tmp_path / "site-packages"
    bundle = tmp_path / "artifacts" / "development-bundle.json"
    runner = repository / _RUNNER_REPOSITORY_PATH
    runner.parent.mkdir(parents=True)
    site.mkdir()
    shutil.copyfile(_RUNNER, runner)
    (repository / "src/victim.py").write_text(module_source, encoding="utf-8")
    _git(repository.parent, "init", repository.name)
    _git(repository, "config", "user.name", "V3 Runner Test")
    _git(repository, "config", "user.email", "runner-test@example.invalid")
    _git(repository, "add", ".")
    _git(repository, "commit", "-m", "fixture A")
    return repository, site, bundle


def _test_preflight_config(repository: Path, site: Path, bundle: Path) -> _PreflightConfig:
    return _PreflightConfig(
        repository=repository,
        python_executable=Path(sys.executable),
        site_packages=site,
        repository_src=repository / "src",
        runner_path=repository / _RUNNER_REPOSITORY_PATH,
        runner_repository_path=_RUNNER_REPOSITORY_PATH,
        bundle_path=bundle,
        standard_library_path=_STANDARD_LIBRARY,
        protected_source_roots=("src",),
        root_import_controls=frozenset(),
        selected_method_relative_path=(
            "configs/governance/metadata_calibration_v3_selected_method_freeze.json"
        ),
        require_direct_invocation=False,
    )


def _seal(value: dict[str, object]) -> dict[str, object]:
    sealed = dict(value)
    sealed["payload_sha256"] = hashlib.sha256(_canonical_json_bytes(value)).hexdigest()
    return sealed


def _frozen_context_transport_fixture():
    from cfeg import metadata_calibration_v3_governance as governance

    plain = _seal(
        {
            "schema": "cfeg.test.context-reference.v1",
            "nested": {"values": [1, 2, {"label": "covariate-only"}]},
        }
    )
    exact_bytes = _canonical_json_bytes(plain) + b"\n"
    frozen = governance.parse_artifact_bytes(
        exact_bytes,
        governance.ArtifactSpec(
            schema="cfeg.test.context-reference.v1",
            exact_fields=frozenset(plain),
        ),
    )
    return plain, frozen, exact_bytes


def test_frozen_context_transport_detaches_deeply_without_byte_drift() -> None:
    plain, frozen, exact_bytes = _frozen_context_transport_fixture()
    retired_v1 = object()
    governance = SimpleNamespace(
        retired_v1_context_reference_payload=lambda value: (
            frozen if value is retired_v1 else (_ for _ in ()).throw(AssertionError())
        ),
        retired_v1_context_reference_bytes=lambda value: (
            exact_bytes if value is retired_v1 else (_ for _ in ()).throw(AssertionError())
        ),
    )

    assert type(frozen) is MappingProxyType
    assert type(frozen["nested"]) is MappingProxyType
    assert type(frozen["nested"]["values"]) is tuple
    detached = _detach_retired_v1_context_payload(governance, retired_v1)
    assert detached == plain
    assert type(detached) is dict
    assert type(detached["nested"]) is dict
    assert type(detached["nested"]["values"]) is list
    assert type(detached["nested"]["values"][2]) is dict
    assert _canonical_json_bytes(detached) + b"\n" == exact_bytes


def test_context_resume_transports_real_frozen_payload_before_publication() -> None:
    plain, frozen, exact_bytes = _frozen_context_transport_fixture()
    retired_v1 = object()
    bundle = SimpleNamespace(clean_commit="a" * 40, clean_tree="b" * 40)
    reference_rng = object()
    proof = object()
    events: list[str] = []

    def require_plain(value):
        assert value == plain
        assert type(value) is dict
        assert type(value["nested"]) is dict
        assert type(value["nested"]["values"]) is list

    def issue_reference(*_args, **_kwargs):
        events.append("reference-only-authority")
        return reference_rng

    def validate_for_publication(value, *, rng_authority):
        require_plain(value)
        assert rng_authority is reference_rng
        events.append("core-proof")
        return proof

    def publish(value, **bindings):
        require_plain(value)
        assert bindings == {
            "development_bundle": bundle,
            "rng_authority": reference_rng,
            "core_capability": proof,
        }
        events.append("context-publication")

    governance = SimpleNamespace(
        retired_v1_context_reference_payload=lambda value: (
            frozen if value is retired_v1 else (_ for _ in ()).throw(AssertionError())
        ),
        retired_v1_context_reference_bytes=lambda value: (
            exact_bytes if value is retired_v1 else (_ for _ in ()).throw(AssertionError())
        ),
        publish_context_reference=publish,
        validate_context_reference=lambda **_kwargs: events.append("context-reopen"),
    )
    facade = object.__new__(ProductionFacade)
    facade._governance = governance
    facade._model = SimpleNamespace(
        issue_bundle_bound_context_reference_rng_authority=issue_reference,
        issue_bundle_bound_development_rng_authority=lambda *_args, **_kwargs: (
            _ for _ in ()
        ).throw(AssertionError("context phase minted development authority")),
        validate_context_reference_for_publication=validate_for_publication,
    )
    facade._synthetic = SimpleNamespace(
        replay_context_reference_payload=lambda *, rng_authority: (
            dict(plain)
            if rng_authority is reference_rng
            else (_ for _ in ()).throw(AssertionError())
        ),
        execute_complete_development=lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("context phase executed development DGP")
        ),
    )

    facade._publish_context_reference(SimpleNamespace(bundle=bundle, retired_v1=retired_v1))
    assert events == [
        "reference-only-authority",
        "core-proof",
        "context-publication",
        "context-reopen",
    ]


def test_context_transport_failure_stops_before_publication_and_development() -> None:
    _, frozen, _ = _frozen_context_transport_fixture()
    retired_v1 = object()
    reached: list[str] = []

    def forbidden(name: str):
        return lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError(f"context transport failure reached {name}")
        )

    facade = object.__new__(ProductionFacade)
    facade._governance = SimpleNamespace(
        retired_v1_context_reference_payload=lambda _value: frozen,
        retired_v1_context_reference_bytes=lambda _value: b"{}\n",
        publish_context_reference=forbidden("context publication"),
        validate_context_reference=forbidden("context validation"),
    )
    facade._model = SimpleNamespace(
        issue_bundle_bound_context_reference_rng_authority=lambda *_args, **_kwargs: (
            reached.append("reference-only-authority") or object()
        ),
        issue_bundle_bound_development_rng_authority=forbidden("development RNG authority"),
        validate_context_reference_for_publication=forbidden("core proof"),
    )
    facade._synthetic = SimpleNamespace(
        replay_context_reference_payload=forbidden("reference replay"),
        execute_complete_development=forbidden("development DGP"),
    )
    graph = SimpleNamespace(
        bundle=SimpleNamespace(clean_commit="a" * 40, clean_tree="b" * 40),
        retired_v1=retired_v1,
    )

    with pytest.raises(RunnerError, match="differs from its immutable bytes"):
        facade._publish_context_reference(graph)
    assert reached == ["reference-only-authority"]


def _write_test_bundle(repository: Path, site: Path, bundle: Path) -> None:
    config = _test_preflight_config(repository, site, bundle)
    source = _capture_source_facts(config, executing_file=_RUNNER)
    python = _capture_python_record(config)
    site_inventory = _capture_python_site_inventory(site)
    git_version = subprocess.run(
        [_GIT, "--version"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    runtime: dict[str, object] = {
        "git": {
            "executable_file_sha256": hashlib.sha256(_GIT.read_bytes()).hexdigest(),
            "executable_path": os.fspath(_GIT),
            "version": git_version,
        },
        "python": dict(python),
        "python_site_inventory": dict(site_inventory),
    }
    payload = _seal(
        {
            "artifact_schema_inventory_sha256": "a" * 64,
            "artifact_schemas": {},
            "candidate_id": _CANDIDATE_ID,
            "canonical_path": os.fspath(bundle),
            "clean_git_commit": source.commit,
            "clean_git_tree": source.tree,
            "created_at_UTC": "2026-09-06T12:00:00Z",
            "development_attempt_id": "development-v5",
            "development_start_receipt_path": (
                "/home/whwovy/v3-artifacts/metadata-calibration-efficiency-v3/"
                "development-v5/development-start.json"
            ),
            "development_start_receipt_policy": (
                "O_EXCL_before_development_authority_and_first_RNG_draw_creator_process_only"
            ),
            "development_start_receipt_schema": (
                "cfeg.metadata-calibration-efficiency-v3.development-start.v1"
            ),
            "focused_test_argv": ["test"],
            "focused_test_collected_tests": 1,
            "focused_test_environment_sha256": "b" * 64,
            "focused_test_error_tests": 0,
            "focused_test_executable_file_sha256": "c" * 64,
            "focused_test_failed_tests": 0,
            "focused_test_junit_report_base64": "eA==",
            "focused_test_junit_report_file_sha256": "d" * 64,
            "focused_test_junit_report_path": "/tmp/test.xml",
            "focused_test_passed_tests": 1,
            "focused_test_process_id": 1,
            "focused_test_scope": "test-only",
            "focused_test_skipped_tests": 0,
            "focused_test_stderr_sha256": "e" * 64,
            "focused_test_stdout_sha256": "f" * 64,
            "focused_test_working_directory": os.fspath(repository),
            "focused_test_xfailed_tests": 0,
            "focused_test_xpassed_tests": 0,
            "focused_tests_passed": True,
            "human_EEG_outcome_authorized": False,
            "master_plan_file_sha256": "1" * 64,
            "master_plan_path": "configs/analysis/metadata_calibration_efficiency_v3.yaml",
            "numerical_runtime_fingerprint_sha256": hashlib.sha256(
                _canonical_json_bytes(runtime)
            ).hexdigest(),
            "numerical_runtime_inventory": runtime,
            "original_preoutcome_amendment_file_sha256": (
                "3238761a0a9a257032d6582286953a4d20981e5e646c9f1322c5c0357bd0c202"
            ),
            "original_preoutcome_amendment_path": (
                "configs/governance/metadata_calibration_v3_preoutcome_amendment.json"
            ),
            "preoutcome_amendment_file_sha256": (
                "129c81f1a806e17d68ced5055d094faf22a9e0f017e834175394c5b40e79ee4e"
            ),
            "preoutcome_amendment_path": (
                "configs/governance/metadata_calibration_v3_4_recovery_amendment.json"
            ),
            "prior_recovery_amendment_file_sha256": (
                "d00fbb2722444fd92235eae97c403c6fc0b9bdf0157d571c2c7717c8afa09c11"
            ),
            "prior_recovery_amendment_path": (
                "configs/governance/metadata_calibration_v3_3_recovery_amendment.json"
            ),
            "earlier_recovery_amendment_file_sha256": (
                "1db2d118376975c1817e329f4203dbaf9929d4f2bcb30bfede990af26c2c1ceb"
            ),
            "earlier_recovery_amendment_path": (
                "configs/governance/metadata_calibration_v3_2_recovery_amendment.json"
            ),
            "first_recovery_amendment_file_sha256": (
                "deddca9286f07c6293925409d01b3df1313e89f2ce89688c4a25c14a8ec01238"
            ),
            "first_recovery_amendment_path": (
                "configs/governance/metadata_calibration_v3_1_recovery_amendment.json"
            ),
            "protocol_revision": "V3.4",
            "owner_authority_public_key_file_sha256": "2" * 64,
            "owner_authority_public_key_path": "configs/governance/owner.pub",
            "owner_authority_ssh_fingerprint": "SHA256:test",
            "retired_v1_artifact_inventory_schema": (
                "cfeg.metadata-calibration-efficiency-v3.retired-development-inventory.v1"
            ),
            "retired_v1_artifact_inventory_sha256": (
                "ece2fc0822fc33d952b926efacb076586d90a74e1bb5132b110789359837055b"
            ),
            "retired_v1_continuation_authorized": False,
            "retired_v1_context_reference_file_sha256": (
                "7b78b3092f5824b6e247d97f4b9cd9f9ec22dbfa74a182e2f32c34fb1c8cf3c8"
            ),
            "retired_v1_context_reference_payload_sha256": (
                "c475e9d0ce4f8e436b49c50585eaee37965bba7c052eefdee862bc50f70a9253"
            ),
            "retired_v1_context_reference_path": (
                "/home/whwovy/v3-artifacts/metadata-calibration-efficiency-v3/"
                "development-v1/context-reference.json"
            ),
            "retired_v1_context_reference_root_seed": 20_260_910,
            "retired_v1_context_reference_schema": (
                "cfeg.metadata-calibration-efficiency-v3.context-reference.v1"
            ),
            "retired_v1_development_DGP_executed": False,
            "retired_v1_development_bundle_file_sha256": (
                "9f6027918110cfd877a1b170f3472d897d4b06f6fca7f0991b9f15f6768c4a1a"
            ),
            "retired_v1_development_bundle_payload_sha256": (
                "959ce56a97e3ece034b52c44a993d393ef168bbb1bdbad9a72dee86cdf3430b3"
            ),
            "retired_v1_development_bundle_path": (
                "/home/whwovy/v3-artifacts/metadata-calibration-efficiency-v3/"
                "development-v1/development-bundle.json"
            ),
            "retired_v1_development_bundle_schema": (
                "cfeg.metadata-calibration-efficiency-v3.development-bundle.v1"
            ),
            "retired_v1_development_result_present": False,
            "retired_v1_development_root_seed": 20_260_909,
            "retired_v1_development_seedsequence_created": False,
            "retired_v1_experimental_outcome_observed": False,
            "retired_v1_governed_network_accessed": False,
            "retired_v1_incident_error": "AuthorityError",
            "retired_v1_incident_message": (
                "this governed process role cannot mutate canonical state"
            ),
            "retired_v1_incident_output_sha256": (
                "8cdcfe3961979b2d691d4a241878abbf68aa743d8dd08ccbf01ffe1ec03a796a"
            ),
            "retired_v1_source_commit": "569394da6894a2efed37b9dde162cbe4fe60534d",
            "retired_v1_source_tree": "fe472b6ae07d2a95d87d8ae5f3ec266a5347e8dd",
            "retired_v2_artifact_inventory_schema": (
                "cfeg.metadata-calibration-efficiency-v3.retired-development-inventory.v2"
            ),
            "retired_v2_artifact_inventory_sha256": (
                "7ae6a99b7ec2033b2ce0cdfbfedc05851b77d9e5dfeb21b485ebe4215f18c865"
            ),
            "retired_v2_context_reference_present": False,
            "retired_v2_context_reference_replay_count": 2,
            "retired_v2_context_reference_root_seed": 20_260_910,
            "retired_v2_continuation_authorized": False,
            "retired_v2_development_DGP_executed": False,
            "retired_v2_development_bundle_file_sha256": (
                "a616945040ea45a78c2eed58bf251f1f379f9a82a320ed4f39c26f5efb073cac"
            ),
            "retired_v2_development_bundle_path": (
                "/home/whwovy/v3-artifacts/metadata-calibration-efficiency-v3/"
                "development-v2/development-bundle.json"
            ),
            "retired_v2_development_bundle_payload_sha256": (
                "ce9628ecc5a2b000aa1ce370bba540e6d0fa2d46d3ae068cbcc5b43f089ba312"
            ),
            "retired_v2_development_bundle_schema": (
                "cfeg.metadata-calibration-efficiency-v3.development-bundle.v2"
            ),
            "retired_v2_development_result_present": False,
            "retired_v2_development_rng_authority_issued": False,
            "retired_v2_development_root_seed": 20_260_909,
            "retired_v2_development_seedsequence_created": False,
            "retired_v2_experimental_outcome_observed": False,
            "retired_v2_governed_network_accessed": False,
            "retired_v2_incident_error": "TypeError",
            "retired_v2_incident_message": ("Object of type mappingproxy is not JSON serializable"),
            "retired_v2_incident_output_sha256": (
                "5e21a934ee692d13e3717502442c412d58121547eb2d05ecaf3c1bd827abd04c"
            ),
            "retired_v2_source_bundle_sha256": (
                "a74d3a86f207520fd97c746809378146d75614fe287b1eaab96e50b15da3dbc0"
            ),
            "retired_v2_source_commit": "82e26d60fd17464380697126bb64ad73b36708a2",
            "retired_v2_source_tree": "f7f59bda48610aeaa5bd1b6ffd288a7b56a4bb85",
            "retired_v3_source_commit": "adab7acda840c5e6ed9a994d82c8098094284c6b",
            "retired_v3_source_tree": "b06b43a130f599e0b85b4ee91856954f6061bea7",
            "retired_v3_source_bundle_sha256": (
                "5e7ff5ad96c010d73870857caf6f3544e227e49099fbb0c1ad839649e30cd609"
            ),
            "retired_v3_numerical_runtime_fingerprint_sha256": (
                "d68d72b909b5eed289de51464c3b5a6340854ac189fecdc394fc27edaad9d633"
            ),
            "retired_v3_artifact_inventory_schema": (
                "cfeg.metadata-calibration-efficiency-v3.retired-development-inventory.v3"
            ),
            "retired_v3_artifact_inventory_sha256": (
                "59502b324417c2be6c03aa5483532646fb8630af92de1218e56d862ce72cd6e7"
            ),
            "retired_v3_development_bundle_path": (
                "/home/whwovy/v3-artifacts/metadata-calibration-efficiency-v3/"
                "development-v3/development-bundle.json"
            ),
            "retired_v3_development_bundle_schema": (
                "cfeg.metadata-calibration-efficiency-v3.development-bundle.v3"
            ),
            "retired_v3_development_bundle_payload_sha256": (
                "b47c55453117c30af0507b10ef7e894b3721738d4670b7ea8afa40f861f9fa1f"
            ),
            "retired_v3_development_bundle_file_sha256": (
                "dee4068d2fdce5ec1031de17bf9e004f0a4629f88bfcc0f13c39211d4383d741"
            ),
            "retired_v3_context_reference_path": (
                "/home/whwovy/v3-artifacts/metadata-calibration-efficiency-v3/"
                "development-v3/context-reference.json"
            ),
            "retired_v3_context_reference_schema": (
                "cfeg.metadata-calibration-efficiency-v3.context-reference.v1"
            ),
            "retired_v3_context_reference_payload_sha256": (
                "c475e9d0ce4f8e436b49c50585eaee37965bba7c052eefdee862bc50f70a9253"
            ),
            "retired_v3_context_reference_file_sha256": (
                "7b78b3092f5824b6e247d97f4b9cd9f9ec22dbfa74a182e2f32c34fb1c8cf3c8"
            ),
            "retired_v3_development_start_path": (
                "/home/whwovy/v3-artifacts/metadata-calibration-efficiency-v3/"
                "development-v3/development-start.json"
            ),
            "retired_v3_development_start_schema": (
                "cfeg.metadata-calibration-efficiency-v3.development-start.v1"
            ),
            "retired_v3_development_start_payload_sha256": (
                "5c26e0e95175fa374f80b9c7e5cd11af66c28d4a53f693dec31a0ff8f4ed6486"
            ),
            "retired_v3_development_start_file_sha256": (
                "2183cc2506cbf4b7d6b62bf442c2a03a89fbbdaace5d66e5e73afb5869fd87ff"
            ),
            "retired_v3_development_started_at_UTC": "2026-09-06T21:06:11Z",
            "retired_v3_development_start_creator_process_id": 550_174,
            "retired_v3_incident_error": "TypeError",
            "retired_v3_incident_message": "Object of type mappingproxy is not JSON serializable",
            "retired_v3_incident_output_sha256": (
                "5e21a934ee692d13e3717502442c412d58121547eb2d05ecaf3c1bd827abd04c"
            ),
            "retired_v3_incident_boundary": (
                "after_execute_complete_development_returned_during_governance_frozen_parse_to_"
                "core_publication_validation_before_result_O_EXCL"
            ),
            "retired_v3_context_reference_root_seed": 20_260_910,
            "retired_v3_development_root_seed": 20_260_909,
            "retired_v3_development_seed_consumed": True,
            "retired_v3_control_flow_inference_classification": (
                "strong_control_flow_inference_not_observed_outcome_or_durable_attestation"
            ),
            "retired_v3_development_rng_authority_issued_in_memory": True,
            "retired_v3_development_seedsequence_and_DGP_executed": True,
            "retired_v3_complete_grid_gate_reports_computed_in_memory": True,
            "retired_v3_development_selection_computed_in_memory": True,
            "retired_v3_development_result_present": False,
            "retired_v3_selected_method_present": False,
            "retired_v3_test_evidence_present": False,
            "retired_v3_canary_present": False,
            "retired_v3_experimental_outcome_observed": False,
            "retired_v3_governed_network_accessed": False,
            "retired_v3_human_EEG_outcome_accessed": False,
            "retired_v3_continuation_authorized": False,
            "retired_v4_source_commit": ("198158663efd1c4481b23dd3d012b9d854ada8d1"),
            "retired_v4_source_tree": ("ac7882494955a7abb5046a4dd950bd98d6fdd6b1"),
            "retired_v4_source_bundle_sha256": (
                "c57c81a2ffae4511e4de6159d11ea5ef482e13b8fdebd916eef98bdf7235250c"
            ),
            "retired_v4_numerical_runtime_fingerprint_sha256": (
                "d68d72b909b5eed289de51464c3b5a6340854ac189fecdc394fc27edaad9d633"
            ),
            "retired_v4_artifact_inventory_schema": (
                "cfeg.metadata-calibration-efficiency-v3.retired-development-inventory.v4"
            ),
            "retired_v4_artifact_inventory_sha256": (
                "511de6688fa315830ddf98ed5a78eefde4e0e4ba6f15977c347281403e908dfa"
            ),
            "retired_v4_development_bundle_path": (
                "/home/whwovy/v3-artifacts/metadata-calibration-efficiency-v3/"
                "development-v4/development-bundle.json"
            ),
            "retired_v4_development_bundle_schema": (
                "cfeg.metadata-calibration-efficiency-v3.development-bundle.v4"
            ),
            "retired_v4_development_bundle_payload_sha256": (
                "0fbb581d320a256c1722f6004810dbfb6f501f5eecf6e95c9264be9657f09e4c"
            ),
            "retired_v4_development_bundle_file_sha256": (
                "d257a2f4ea83bc355399ddafea5f7deca8e95bb9e4de7d4a5e3ccb99a1602dd9"
            ),
            "retired_v4_context_reference_path": (
                "/home/whwovy/v3-artifacts/metadata-calibration-efficiency-v3/"
                "development-v4/context-reference.json"
            ),
            "retired_v4_context_reference_schema": (
                "cfeg.metadata-calibration-efficiency-v3.context-reference.v1"
            ),
            "retired_v4_context_reference_payload_sha256": (
                "c475e9d0ce4f8e436b49c50585eaee37965bba7c052eefdee862bc50f70a9253"
            ),
            "retired_v4_context_reference_file_sha256": (
                "7b78b3092f5824b6e247d97f4b9cd9f9ec22dbfa74a182e2f32c34fb1c8cf3c8"
            ),
            "retired_v4_development_start_path": (
                "/home/whwovy/v3-artifacts/metadata-calibration-efficiency-v3/"
                "development-v4/development-start.json"
            ),
            "retired_v4_development_start_schema": (
                "cfeg.metadata-calibration-efficiency-v3.development-start.v1"
            ),
            "retired_v4_development_start_payload_sha256": (
                "b33666e840f9a6669713b79b10f75dbf1100d939b2c96e25bce22d7fe60ec8b2"
            ),
            "retired_v4_development_start_file_sha256": (
                "1bfafb3b265c3cd7c459a8c7d09be4275743588addb2680e0685a1501389c068"
            ),
            "retired_v4_development_started_at_UTC": "2026-09-06T23:35:38Z",
            "retired_v4_development_start_creator_process_id": 1_234_958,
            "retired_v4_incident_error": "TypeError",
            "retired_v4_incident_message": (
                "primary_values_by_endpoint must be an exact endpoint-ordered dictionary."
            ),
            "retired_v4_incident_output_sha256": (
                "5c6a3fa68152b88b3f32745f04afcc52f3c636dd709eb56a937470b45e7ddbcc"
            ),
            "retired_v4_incident_boundary": (
                "after_execute_complete_development_returned_during_governance_canonical_parse_"
                "to_core_publication_validation_before_result_O_EXCL"
            ),
            "retired_v4_context_reference_root_seed": 20_260_910,
            "retired_v4_development_root_seed": 3_156_745_110,
            "retired_v4_development_seed_consumed": True,
            "retired_v4_control_flow_inference_classification": (
                "strong_control_flow_inference_not_observed_outcome_or_durable_attestation"
            ),
            "retired_v4_development_rng_authority_issued_in_memory": True,
            "retired_v4_development_seedsequence_and_DGP_executed": True,
            "retired_v4_complete_grid_gate_reports_computed_in_memory": True,
            "retired_v4_development_selection_computed_in_memory": True,
            "retired_v4_development_result_present": False,
            "retired_v4_selected_method_present": False,
            "retired_v4_test_evidence_present": False,
            "retired_v4_canary_present": False,
            "retired_v4_experimental_outcome_observed": False,
            "retired_v4_governed_network_accessed": False,
            "retired_v4_human_EEG_outcome_accessed": False,
            "retired_v4_continuation_authorized": False,
            "original_retired_scientific_contract_projection_sha256": (
                "0702d01be1e055d3203a3c1b78777db6456b8d527e5525b6d468fb52522f8a79"
            ),
            "retired_scientific_contract_projection_sha256": (
                "5885f39908d8a33b130e0dc923abe560cc4587ebe7c58ed242647ce028f94712"
            ),
            "replacement_seed_preimage_schema": (
                "cfeg.metadata-calibration-efficiency-v3.replacement-development-seed-preimage.v2"
            ),
            "replacement_seed_preimage_sha256": (
                "11f503bdca78e8a7d0e853f9a7b484510dd749018c7735bce61ff7965b70b1cb"
            ),
            "replacement_seed_digest_split": "eight_big_endian_uint32_words",
            "replacement_seed_selection_rule": (
                "first_nonzero_word_outside_forbidden_seed_set_without_reroll_or_counter"
            ),
            "replacement_seed_selected_word_index": 0,
            "replacement_development_root_seed": 301_269_949,
            "replacement_seed_outcome_used": False,
            "replacement_development_evidence_role": (
                "transparent_synthetic_development_and_model_selection_not_human_confirmatory_"
                "evidence"
            ),
            "development_result_endpoint_order_policy": (
                "endpoint_order_array_is_authoritative_JSON_object_keys_are_an_exact_unordered_"
                "set_and_all_validation_iteration_uses_endpoint_order"
            ),
            "schema": _DEVELOPMENT_BUNDLE_SCHEMA,
            "scientific_contract_projection_sha256": (
                "670b61c767ff2a4b02c3a582aa8ac7e46b5632ada2806ffc109e2e79326d05ca"
            ),
            "scientific_lockbox_authorized": False,
            "source_bundle_sha256": source.source_bundle_sha256,
            "synthetic_plan_file_sha256": "3" * 64,
            "synthetic_plan_path": "configs/analysis/metadata_calibration_v3_synthetic.yaml",
            "tracked_source_file_inventory": [item.record() for item in source.tracked_files],
            "v2_deny_overlay_file_sha256": "4" * 64,
            "v2_deny_overlay_path": "configs/governance/v2-deny.json",
            "v2_terminal_audit_file_sha256": "5" * 64,
            "v2_terminal_audit_path": "configs/governance/v2-audit.json",
        }
    )
    bundle.parent.mkdir(parents=True)
    bundle.write_bytes(_canonical_json_bytes(payload) + b"\n")
    bundle.chmod(0o400)


def test_runner_preimport_bundle_inventory_matches_governance_and_rejects_retired_versions(
    tmp_path: Path,
) -> None:
    from cfeg import metadata_calibration_v3_governance as governance

    assert _DEVELOPMENT_BUNDLE_FIELDS == governance._DEVELOPMENT_BUNDLE_FIELDS
    repository, site, bundle = _make_test_repository(tmp_path, module_source="VALUE = 1\n")
    _write_test_bundle(repository, site, bundle)
    config = _test_preflight_config(repository, site, bundle)
    source = _capture_source_facts(config, executing_file=_RUNNER)
    python = _capture_python_record(config)
    site_inventory = _capture_python_site_inventory(site)
    payload = json.loads(bundle.read_text(encoding="utf-8"))

    missing = dict(payload)
    missing.pop("protocol_revision")
    missing.pop("payload_sha256")
    with pytest.raises(RunnerError, match="field inventory differs"):
        _validate_existing_bundle(
            config,
            _canonical_json_bytes(_seal(missing)) + b"\n",
            source=source,
            python=python,
            site_inventory=site_inventory,
        )

    for retired_version in ("v1", "v2", "v3", "v4"):
        retired_schema = dict(payload)
        retired_schema["schema"] = (
            f"cfeg.metadata-calibration-efficiency-v3.development-bundle.{retired_version}"
        )
        retired_schema.pop("payload_sha256")
        with pytest.raises(RunnerError, match="recovery identity differs: schema"):
            _validate_existing_bundle(
                config,
                _canonical_json_bytes(_seal(retired_schema)) + b"\n",
                source=source,
                python=python,
                site_inventory=site_inventory,
            )

    for retired_attempt in (
        "development-v1",
        "development-v2",
        "development-v3",
        "development-v4",
    ):
        retired_path = dict(payload)
        retired_path["canonical_path"] = (
            "/home/whwovy/v3-artifacts/metadata-calibration-efficiency-v3/"
            f"{retired_attempt}/development-bundle.json"
        )
        retired_path.pop("payload_sha256")
        with pytest.raises(RunnerError, match="recovery identity differs: canonical_path"):
            _validate_existing_bundle(
                config,
                _canonical_json_bytes(_seal(retired_path)) + b"\n",
                source=source,
                python=python,
                site_inventory=site_inventory,
            )

    mutations = {
        "prior_recovery_amendment_file_sha256": "0" * 64,
        "earlier_recovery_amendment_file_sha256": "0" * 64,
        "retired_v3_artifact_inventory_sha256": "0" * 64,
        "retired_v3_development_start_file_sha256": "0" * 64,
        "retired_v3_development_seed_consumed": False,
        "retired_v4_artifact_inventory_sha256": "0" * 64,
        "retired_v4_development_start_file_sha256": "0" * 64,
        "retired_v4_development_seed_consumed": False,
        "replacement_seed_preimage_sha256": "0" * 64,
        "replacement_development_root_seed": 301_269_950,
        "retired_scientific_contract_projection_sha256": "0" * 64,
        "scientific_contract_projection_sha256": "0" * 64,
    }
    for field, replacement in mutations.items():
        changed = dict(payload)
        changed[field] = replacement
        changed.pop("payload_sha256")
        with pytest.raises(RunnerError, match=rf"recovery identity differs: {field}"):
            _validate_existing_bundle(
                config,
                _canonical_json_bytes(_seal(changed)) + b"\n",
                source=source,
                python=python,
                site_inventory=site_inventory,
            )


def test_post_context_status_reopens_reference_without_development_or_mutation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[str] = []
    bundle = SimpleNamespace(clean_commit="a" * 40, clean_tree="b" * 40)
    retired_v1 = object()
    retired_v2 = object()
    reference_rng = object()
    reference_value = object()
    reference_proof = object()
    context_artifact = object()
    payload, frozen_payload, exact_bytes = _frozen_context_transport_fixture()

    governance = SimpleNamespace(
        observe_retired_development_attempts=lambda: (
            retired_v1,
            retired_v2,
            object(),
            object(),
        ),
        reopen_development_bundle=lambda: bundle,
        retired_v1_context_reference_payload=lambda value: (
            frozen_payload if value is retired_v1 else (_ for _ in ()).throw(AssertionError())
        ),
        retired_v1_context_reference_bytes=lambda value: (
            exact_bytes if value is retired_v1 else (_ for _ in ()).throw(AssertionError())
        ),
        validate_context_reference=lambda **_kwargs: context_artifact,
    )

    def reference_issuer(value, *, expected_commit, expected_tree):
        assert value is bundle
        assert (expected_commit, expected_tree) == (bundle.clean_commit, bundle.clean_tree)
        events.append("reference-only-authority")
        return reference_rng

    def validate_reference(value):
        assert value == payload
        assert type(value) is dict
        assert type(value["nested"]) is dict
        assert type(value["nested"]["values"]) is list
        return reference_value

    def validate_reference_for_publication(value, *, rng_authority):
        assert value == payload
        assert type(value["nested"]["values"][2]) is dict
        assert rng_authority is reference_rng
        return reference_proof

    model = SimpleNamespace(
        issue_bundle_bound_context_reference_rng_authority=reference_issuer,
        issue_bundle_bound_development_rng_authority=lambda *_args, **_kwargs: (
            _ for _ in ()
        ).throw(AssertionError("status minted development RNG authority")),
        validate_context_reference_payload=validate_reference,
        validate_context_reference_for_publication=validate_reference_for_publication,
    )
    synthetic = SimpleNamespace(
        replay_context_reference_payload=lambda *, rng_authority: (
            dict(payload)
            if rng_authority is reference_rng
            else (_ for _ in ()).throw(AssertionError())
        ),
        execute_complete_development=lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("status executed development DGP")
        ),
    )
    facade = object.__new__(ProductionFacade)
    facade._governance = governance
    facade._model = model
    facade._synthetic = synthetic
    facade._inspection = None
    facade._graph = None
    presence = {name: False for name in ProductionFacade._ARTIFACT_ORDER}
    presence["bundle"] = True
    presence["context_reference"] = True
    monkeypatch.setattr(ProductionFacade, "_presence", lambda _self: presence)
    monkeypatch.setattr(
        ProductionFacade,
        "_require_bundle_matches_preflight",
        lambda _self, _bundle: None,
    )
    for name in (
        "publish_development_bundle",
        "publish_context_reference",
        "publish_development_result",
    ):
        setattr(
            governance,
            name,
            lambda *_args, _name=name, **_kwargs: (_ for _ in ()).throw(
                AssertionError(f"status mutated canonical state through {_name}")
            ),
        )

    observed = status_report(facade)
    assert observed["stage"] == RunStage.DEVELOPMENT_PENDING.value
    assert observed["durable_phases_advanced"] == 0
    assert events == ["reference-only-authority"]
    assert facade._graph.context_reference is context_artifact


def test_development_pending_resume_issues_development_authority_just_in_time(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import cfeg.metadata_calibration_v3_runner as runner_module

    events: list[str] = []
    bundle = SimpleNamespace(clean_commit="a" * 40, clean_tree="b" * 40)
    context_artifact = SimpleNamespace(file_sha256="c" * 64)
    reference_value = object()
    reference_proof = object()
    development_rng = object()
    contract = object()
    payload = {"schema": "development-result"}
    result_proof = object()
    start = object()

    def issue_development(value, context, start_value, *, expected_commit, expected_tree):
        assert value is bundle
        assert context is context_artifact
        assert start_value is start
        assert (expected_commit, expected_tree) == (bundle.clean_commit, bundle.clean_tree)
        events.append("jit-development-authority")
        return development_rng

    model = SimpleNamespace(
        issue_bundle_bound_development_rng_authority=issue_development,
    )

    def execute(
        value,
        reference,
        *,
        rng_authority,
        validated_reference,
        context_reference_file_sha256,
    ):
        assert value is contract
        assert reference is reference_value
        assert rng_authority is development_rng
        assert validated_reference is reference_proof
        assert context_reference_file_sha256 == context_artifact.file_sha256
        events.extend(("first-development-rng-draw", "development-DGP-and-core-validation"))
        return payload, result_proof

    def validate_contract():
        events.append("synthetic-contract-load")
        return contract

    def validate_development(value):
        assert value is contract
        events.append("development-contract-validation")

    synthetic = SimpleNamespace(
        validate_synthetic_contract=validate_contract,
        validate_development_contract=validate_development,
        execute_complete_development=execute,
    )

    def publish(value, **bindings):
        assert value is payload
        assert bindings["development_bundle"] is bundle
        assert bindings["context_reference"] is context_artifact
        assert bindings["development_start"] is start
        assert bindings["development_rng_authority"] is development_rng
        assert bindings["validated_context_reference"] is reference_proof
        assert bindings["core_capability"] is result_proof
        events.append("result-publication")

    def publish_start(**bindings):
        assert bindings == {
            "development_bundle": bundle,
            "context_reference": context_artifact,
            "started_at_UTC": "2026-09-07T00:00:00Z",
        }
        events.append("development-start-O_EXCL-readback")
        return start

    governance = SimpleNamespace(
        publish_development_start=publish_start,
        publish_development_result=publish,
        validate_development_result=lambda **_kwargs: events.append("result-validation"),
    )
    facade = object.__new__(ProductionFacade)
    facade._model = model
    facade._synthetic = synthetic
    facade._governance = governance
    facade._preflight = object()
    monkeypatch.setattr(runner_module, "_require_preflight_unchanged", lambda _value: None)
    monkeypatch.setattr(
        runner_module,
        "_rfc3339_now_after",
        lambda _previous=None: "2026-09-07T00:00:00Z",
    )
    graph = SimpleNamespace(
        bundle=bundle,
        reference_value=reference_value,
        reference_proof=reference_proof,
        context_reference=context_artifact,
    )

    facade._publish_development_result(graph)
    assert events == [
        "synthetic-contract-load",
        "development-contract-validation",
        "development-start-O_EXCL-readback",
        "jit-development-authority",
        "first-development-rng-draw",
        "development-DGP-and-core-validation",
        "result-publication",
        "result-validation",
    ]


@pytest.mark.parametrize("failure_point", ("load", "validation"))
def test_development_contract_failure_precedes_all_rng_and_dgp_calls(
    failure_point: str,
) -> None:
    events: list[str] = []
    contract = object()

    def load_contract():
        events.append("load")
        if failure_point == "load":
            raise ValueError("contract rejected")
        return contract

    def validate_contract(value):
        assert value is contract
        events.append("validation")
        raise ValueError("contract rejected")

    def forbidden(name: str):
        return lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError(f"contract failure reached {name}")
        )

    facade = object.__new__(ProductionFacade)
    facade._synthetic = SimpleNamespace(
        validate_synthetic_contract=load_contract,
        validate_development_contract=validate_contract,
        execute_complete_development=forbidden("development seed or DGP"),
    )
    facade._model = SimpleNamespace(
        issue_bundle_bound_development_rng_authority=forbidden("development RNG authority")
    )
    facade._governance = SimpleNamespace(
        publish_development_start=forbidden("development start receipt"),
        publish_development_result=forbidden("result publication"),
        validate_development_result=forbidden("result validation"),
    )
    graph = SimpleNamespace(
        bundle=SimpleNamespace(clean_commit="a" * 40, clean_tree="b" * 40),
        reference_value=object(),
        reference_proof=object(),
        context_reference=SimpleNamespace(file_sha256="c" * 64),
    )

    with pytest.raises(ValueError, match="contract rejected"):
        facade._publish_development_result(graph)
    assert events == (["load"] if failure_point == "load" else ["load", "validation"])


@pytest.mark.parametrize("failure_point", ("before_authority", "before_first_draw"))
def test_failure_after_start_receipt_consumes_attempt_before_result(
    failure_point: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import cfeg.metadata_calibration_v3_runner as runner_module

    events: list[str] = []
    bundle = SimpleNamespace(clean_commit="a" * 40, clean_tree="b" * 40)
    context = SimpleNamespace(file_sha256="c" * 64)
    start = object()
    development_rng = object()
    contract = object()

    def publish_start(**_kwargs):
        events.append("development-start-O_EXCL-readback")
        return start

    def issue_development(*args, **kwargs):
        assert args == (bundle, context, start)
        assert kwargs == {
            "expected_commit": bundle.clean_commit,
            "expected_tree": bundle.clean_tree,
        }
        events.append("development-authority-boundary")
        if failure_point == "before_authority":
            raise RuntimeError("injected after receipt before authority")
        return development_rng

    def execute(*_args, **_kwargs):
        events.append("executor-entered-before-first-draw")
        raise RuntimeError("injected before first development draw")

    def forbidden_result(*_args, **_kwargs):
        raise AssertionError("consumed attempt reached result publication")

    facade = object.__new__(ProductionFacade)
    facade._preflight = object()
    facade._synthetic = SimpleNamespace(
        validate_synthetic_contract=lambda: contract,
        validate_development_contract=lambda value: value is contract,
        execute_complete_development=execute,
    )
    facade._model = SimpleNamespace(
        issue_bundle_bound_development_rng_authority=issue_development,
    )
    facade._governance = SimpleNamespace(
        publish_development_start=publish_start,
        publish_development_result=forbidden_result,
        validate_development_result=forbidden_result,
    )
    graph = SimpleNamespace(
        bundle=bundle,
        reference_value=object(),
        reference_proof=object(),
        context_reference=context,
        development_start=None,
    )
    monkeypatch.setattr(runner_module, "_require_preflight_unchanged", lambda _value: None)
    monkeypatch.setattr(
        runner_module,
        "_rfc3339_now_after",
        lambda _previous=None: "2026-09-07T00:00:00Z",
    )

    with pytest.raises(RuntimeError, match="injected"):
        facade._publish_development_result(graph)
    assert graph.development_start is start
    assert events[:2] == [
        "development-start-O_EXCL-readback",
        "development-authority-boundary",
    ]
    if failure_point == "before_authority":
        assert events == events[:2]
    else:
        assert events == [*events[:2], "executor-entered-before-first-draw"]


def test_fresh_receipt_without_result_is_terminal_and_never_resumes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bundle = SimpleNamespace(clean_commit="a" * 40, clean_tree="b" * 40)
    context = object()
    start = object()
    events: list[str] = []

    def forbidden(name: str):
        return lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError(f"receipt-only restart reached {name}")
        )

    governance = SimpleNamespace(
        observe_retired_development_attempts=lambda: (
            object(),
            object(),
            object(),
            object(),
        ),
        reopen_development_bundle=lambda: bundle,
        reopen_development_start_receipt=lambda **_kwargs: (
            events.append("read-only-start-reopen") or start
        ),
        recover_canonical_development_result_at_bundle_source=forbidden("development recovery"),
        publish_development_start=forbidden("start republish"),
        publish_development_result=forbidden("result publication"),
    )
    facade = object.__new__(ProductionFacade)
    facade._governance = governance
    facade._inspection = None
    facade._graph = None
    presence = {name: False for name in ProductionFacade._ARTIFACT_ORDER}
    presence.update(
        {
            "bundle": True,
            "context_reference": True,
            "development_start": True,
        }
    )
    monkeypatch.setattr(ProductionFacade, "_presence", lambda _self: presence)
    monkeypatch.setattr(
        ProductionFacade,
        "_require_bundle_matches_preflight",
        lambda _self, _bundle: None,
    )

    def reopen_context(_self, graph):
        events.append("read-only-context-reopen")
        graph.context_reference = context

    monkeypatch.setattr(ProductionFacade, "_reopen_a_context", reopen_context)

    report = resume_once(facade)
    assert report == {
        "command": "resume",
        "stage_before": "DEVELOPMENT_ATTEMPT_CONSUMED",
        "stage": "DEVELOPMENT_ATTEMPT_CONSUMED",
        "terminal": True,
        "selection_status": None,
        "selected_grid_cell_id": None,
        "next_command": None,
        "durable_phases_advanced": 0,
    }
    assert events == ["read-only-context-reopen", "read-only-start-reopen"]


def test_result_without_start_receipt_is_a_durable_prefix_gap(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bundle = SimpleNamespace(clean_commit="a" * 40, clean_tree="b" * 40)
    governance = SimpleNamespace(
        observe_retired_development_attempts=lambda: (
            object(),
            object(),
            object(),
            object(),
        ),
        reopen_development_bundle=lambda: bundle,
    )
    facade = object.__new__(ProductionFacade)
    facade._governance = governance
    facade._inspection = None
    facade._graph = None
    presence = {name: False for name in ProductionFacade._ARTIFACT_ORDER}
    presence.update(
        {
            "bundle": True,
            "context_reference": True,
            "development_result": True,
        }
    )
    monkeypatch.setattr(ProductionFacade, "_presence", lambda _self: presence)
    monkeypatch.setattr(
        ProductionFacade,
        "_require_bundle_matches_preflight",
        lambda _self, _bundle: None,
    )

    with pytest.raises(RunnerError, match="gap before development_start"):
        facade.inspect()


def test_post_result_status_uses_read_only_recovery_without_rng_or_mutation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[str] = []
    bundle = SimpleNamespace(clean_commit="a" * 40, clean_tree="b" * 40)
    retired_v1 = object()
    retired_v2 = object()
    context = object()
    start = object()
    result = object()
    audited = SimpleNamespace(
        selection_status="SELECTED_METHOD_PROPOSED",
        selected_grid_cell_id="nu-01_lambda-010",
    )

    def recover(*, development_bundle, context_reference):
        assert development_bundle is bundle
        assert context_reference is context
        events.append("read-only-development-recovery")
        return result, audited

    def forbidden(name: str):
        return lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError(f"status reached forbidden {name}")
        )

    governance = SimpleNamespace(
        observe_retired_development_attempts=lambda: (
            retired_v1,
            retired_v2,
            object(),
            object(),
        ),
        reopen_development_bundle=lambda: bundle,
        reopen_development_start_receipt=lambda **_kwargs: (
            events.append("read-only-start-reopen") or start
        ),
        recover_canonical_development_result_at_bundle_source=recover,
        publish_development_bundle=forbidden("bundle publication"),
        publish_context_reference=forbidden("context publication"),
        publish_development_result=forbidden("result publication"),
    )
    facade = object.__new__(ProductionFacade)
    facade._governance = governance
    facade._model = SimpleNamespace(
        issue_bundle_bound_development_rng_authority=forbidden("development RNG authority")
    )
    facade._synthetic = SimpleNamespace(execute_complete_development=forbidden("development DGP"))
    facade._inspection = None
    facade._graph = None
    presence = {name: False for name in ProductionFacade._ARTIFACT_ORDER}
    presence.update(
        {
            "bundle": True,
            "context_reference": True,
            "development_start": True,
            "development_result": True,
        }
    )
    monkeypatch.setattr(ProductionFacade, "_presence", lambda _self: presence)
    monkeypatch.setattr(
        ProductionFacade,
        "_require_bundle_matches_preflight",
        lambda _self, _bundle: None,
    )

    def reopen_context(_self, graph):
        events.append("reference-only-replay")
        graph.context_reference = context

    monkeypatch.setattr(ProductionFacade, "_reopen_a_context", reopen_context)

    observed = facade.inspect()
    assert observed.stage is RunStage.SELECTION_HANDOFF
    assert observed.selection_status == "SELECTED_METHOD_PROPOSED"
    assert observed.selected_grid_cell_id == "nu-01_lambda-010"
    assert events == [
        "reference-only-replay",
        "read-only-start-reopen",
        "read-only-development-recovery",
    ]


def test_bad_recovery_binding_is_rejected_before_bundle_publication(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import cfeg.metadata_calibration_v3_runner as runner_module

    repository, site, bundle_path = _make_test_repository(
        tmp_path,
        module_source="VALUE = 1\n",
    )
    _write_test_bundle(repository, site, bundle_path)
    built = json.loads(bundle_path.read_text(encoding="utf-8"))
    built["scientific_contract_projection_sha256"] = "0" * 64
    built.pop("payload_sha256")
    built = _seal(built)
    config = _test_preflight_config(repository, site, bundle_path)
    preflight = SimpleNamespace(
        bundle_payload=None,
        config=config,
        source=_capture_source_facts(config, executing_file=_RUNNER),
        python=_capture_python_record(config),
        python_site_inventory=_capture_python_site_inventory(site),
    )
    reached_publication = False

    def forbidden_publication(**_kwargs):
        nonlocal reached_publication
        reached_publication = True
        raise AssertionError("malformed recovery bundle reached publication")

    governance = SimpleNamespace(
        V3_SOURCE_REPOSITORY=repository,
        FROZEN_FILE_SHA256={},
        capture_clean_source_snapshot=lambda _root: object(),
        run_observed_test=lambda *_args, **_kwargs: object(),
        build_development_bundle=lambda **_kwargs: built,
        publish_development_bundle=forbidden_publication,
    )
    facade = object.__new__(ProductionFacade)
    facade._preflight = preflight
    facade._governance = governance
    monkeypatch.setattr(ProductionFacade, "_require_active_process", lambda _self: None)
    monkeypatch.setattr(
        ProductionFacade,
        "_require_snapshot_matches_preflight",
        lambda _self, _snapshot: None,
    )
    monkeypatch.setattr(
        runner_module, "_require_preflight_unchanged", lambda *_args, **_kwargs: None
    )

    with pytest.raises(
        RunnerError,
        match="recovery identity differs: scientific_contract_projection_sha256",
    ):
        facade._publish_bundle(SimpleNamespace())
    assert reached_publication is False


def test_bad_bundle_self_hash_is_rejected_before_publication(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import cfeg.metadata_calibration_v3_runner as runner_module

    repository, site, bundle_path = _make_test_repository(
        tmp_path,
        module_source="VALUE = 1\n",
    )
    _write_test_bundle(repository, site, bundle_path)
    built = json.loads(bundle_path.read_text(encoding="utf-8"))
    built["payload_sha256"] = "0" * 64
    config = _test_preflight_config(repository, site, bundle_path)
    preflight = SimpleNamespace(
        bundle_payload=None,
        config=config,
        source=_capture_source_facts(config, executing_file=_RUNNER),
        python=_capture_python_record(config),
        python_site_inventory=_capture_python_site_inventory(site),
    )
    reached_publication = False

    def forbidden_publication(**_kwargs):
        nonlocal reached_publication
        reached_publication = True
        raise AssertionError("invalid self-hash reached publication")

    governance = SimpleNamespace(
        V3_SOURCE_REPOSITORY=repository,
        FROZEN_FILE_SHA256={},
        capture_clean_source_snapshot=lambda _root: object(),
        run_observed_test=lambda *_args, **_kwargs: object(),
        build_development_bundle=lambda **_kwargs: built,
        publish_development_bundle=forbidden_publication,
    )
    facade = object.__new__(ProductionFacade)
    facade._preflight = preflight
    facade._governance = governance
    monkeypatch.setattr(ProductionFacade, "_require_active_process", lambda _self: None)
    monkeypatch.setattr(
        ProductionFacade,
        "_require_snapshot_matches_preflight",
        lambda _self, _snapshot: None,
    )
    monkeypatch.setattr(
        runner_module, "_require_preflight_unchanged", lambda *_args, **_kwargs: None
    )

    with pytest.raises(RunnerError, match="built development bundle payload self-hash differs"):
        facade._publish_bundle(SimpleNamespace())
    assert reached_publication is False


def _run_hostile_preflight(
    repository: Path,
    site: Path,
    bundle: Path,
    *,
    module_name: str = "victim",
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            _PYTHON,
            "-I",
            "-B",
            "-S",
            "-c",
            _PREFLIGHT_HARNESS,
            os.fspath(_RUNNER),
            os.fspath(repository),
            os.fspath(site),
            os.fspath(bundle),
            os.fspath(repository / _RUNNER_REPOSITORY_PATH),
            module_name,
        ],
        check=False,
        capture_output=True,
        text=True,
        env=dict(_GOVERNANCE_ENVIRONMENT),
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
            self.events.extend(("execute:authoritative_development", "publish:development_result"))
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
    tokens = shlex.split(wrapper.replace("\\\n", " "), comments=True, posix=True)
    command = tokens[tokens.index("exec") + 1 :]
    assert command[:2] == ["/usr/bin/env", "-i"]
    python_index = command.index(os.fspath(_PYTHON))
    assignments = command[2:python_index]
    assert len(assignments) == len(_GOVERNANCE_ENVIRONMENT)
    assert dict(item.split("=", 1) for item in assignments) == _GOVERNANCE_ENVIRONMENT
    assert command[python_index:] == [
        os.fspath(_PYTHON),
        "-I",
        "-B",
        "-S",
        "/home/whwovy/califreeEEG/src/cfeg/metadata_calibration_v3_runner.py",
        "$@",
    ]
    assert _EXPECTED_GOVERNED_ENVIRONMENT == _GOVERNANCE_ENVIRONMENT
    assert " -c " not in wrapper
    assert "import " not in wrapper
    assert os.stat(_WRAPPER).st_mode & 0o111


def test_runner_and_wrapper_environment_match_governance_source_exactly() -> None:
    governance_source = _GOVERNANCE.read_text(encoding="utf-8")
    tree = ast.parse(governance_source)
    assignment = next(
        statement
        for statement in tree.body
        if isinstance(statement, ast.Assign)
        and any(
            isinstance(target, ast.Name) and target.id == "_OBSERVED_TEST_ENVIRONMENT"
            for target in statement.targets
        )
    )
    assert isinstance(assignment.value, ast.Call)
    assert len(assignment.value.args) == 1
    governance_environment = ast.literal_eval(assignment.value.args[0])
    has_final_governance = any(
        isinstance(statement, (ast.FunctionDef, ast.AsyncFunctionDef))
        and statement.name == "establish_governed_process"
        for statement in tree.body
    )
    if not has_final_governance:
        # This runner worktree intentionally retains the pre-overlay governance
        # parent.  The integrated tree takes the exact branch below; here we
        # also bind the sole known legacy difference instead of silently
        # weakening the comparison.
        assert governance_environment == {
            **_GOVERNANCE_ENVIRONMENT,
            "PYTHONHASHSEED": "0",
        }
        return
    assert governance_environment == _GOVERNANCE_ENVIRONMENT
    assert _EXPECTED_GOVERNED_ENVIRONMENT == governance_environment


def test_tracked_wrapper_clears_a_polluted_parent_before_python_start(
    tmp_path: Path,
) -> None:
    probe = tmp_path / "bootstrap_probe.py"
    copied_wrapper = tmp_path / "run_metadata_calibration_v3"
    expected_orig_argv = (
        os.fspath(_PYTHON),
        "-I",
        "-B",
        "-S",
        os.fspath(probe),
        "status",
    )
    probe.write_text(
        "import os\n"
        "import pathlib\n"
        "import runpy\n"
        "import sys\n"
        "import types\n"
        f"expected_environment = {_GOVERNANCE_ENVIRONMENT!r}\n"
        f"expected_orig_argv = {expected_orig_argv!r}\n"
        f"expected_stdlib = {_STANDARD_LIBRARY!r}\n"
        "assert dict(os.environ) == expected_environment\n"
        "assert tuple(sys.path) == expected_stdlib\n"
        "assert tuple(sys.orig_argv) == expected_orig_argv\n"
        f"assert tuple(sys.argv) == ({os.fspath(probe)!r}, 'status')\n"
        f"assert sys.executable == {os.fspath(_PYTHON)!r}\n"
        "assert sys.flags.isolated == 1\n"
        "assert sys.flags.ignore_environment == 1\n"
        "assert sys.flags.dont_write_bytecode == 1\n"
        "assert sys.flags.no_user_site == 1\n"
        "assert sys.flags.no_site == 1\n"
        "assert sys.flags.hash_randomization == 1\n"
        f"runner = pathlib.Path({os.fspath(_RUNNER)!r})\n"
        "namespace = runpy.run_path(str(runner), run_name='wrapper_bootstrap_probe')\n"
        "config = types.SimpleNamespace(\n"
        "    python_executable=pathlib.Path(sys.executable),\n"
        "    standard_library_path=expected_stdlib,\n"
        "    require_direct_invocation=False,\n"
        ")\n"
        "assert namespace['_require_process_bootstrap'](\n"
        "    config, executing_file=runner\n"
        ") is None\n"
        "print('EXACT_BOOTSTRAP_OK')\n",
        encoding="utf-8",
    )
    wrapper = _WRAPPER.read_text(encoding="utf-8")
    canonical_runner = "/home/whwovy/califreeEEG/src/cfeg/metadata_calibration_v3_runner.py"
    assert wrapper.count(canonical_runner) == 1
    copied_wrapper.write_text(
        wrapper.replace(canonical_runner, os.fspath(probe)),
        encoding="utf-8",
    )
    copied_wrapper.chmod(0o700)
    polluted = dict(os.environ)
    polluted.update(
        {
            "HOME": "/polluted-home",
            "INJECTED_EXTRA": "must-not-survive",
            "OMP_NUM_THREADS": "999",
            "PATH": "/polluted-path",
            "PYTHONINSPECT": "1",
            "PYTHONPATH": "/polluted-python-path",
        }
    )

    completed = subprocess.run(
        [copied_wrapper, "status"],
        check=False,
        capture_output=True,
        text=True,
        cwd=tmp_path,
        env=polluted,
    )

    assert completed.returncode == 0, completed.stderr
    assert completed.stdout == "EXACT_BOOTSTRAP_OK\n"


@pytest.mark.parametrize(
    "mutation",
    (
        lambda value: {key: item for key, item in value.items() if key != "VECLIB_MAXIMUM_THREADS"},
        lambda value: {**value, "UNEXPECTED_EXTRA": "1"},
    ),
)
def test_preflight_environment_requires_the_exact_key_set(
    tmp_path: Path,
    mutation: object,
) -> None:
    repository, site, bundle = _make_test_repository(
        tmp_path,
        module_source="VALUE = 1\n",
    )
    environment = mutation(dict(_GOVERNANCE_ENVIRONMENT))
    completed = subprocess.run(
        [
            _PYTHON,
            "-I",
            "-B",
            "-S",
            "-c",
            _PREFLIGHT_HARNESS,
            os.fspath(_RUNNER),
            os.fspath(repository),
            os.fspath(site),
            os.fspath(bundle),
            os.fspath(repository / _RUNNER_REPOSITORY_PATH),
            "victim",
        ],
        check=False,
        capture_output=True,
        text=True,
        env=environment,
    )
    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.startswith("REJECTED:RunnerError:")
    assert "exact governed environment" in completed.stdout


def test_rfc3339_timestamp_without_previous_is_strict_seconds_utc() -> None:
    before = datetime.now(timezone.utc).replace(microsecond=0)
    value = _rfc3339_now_after()
    after = datetime.now(timezone.utc).replace(microsecond=0)
    parsed = datetime.strptime(value, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)
    assert before <= parsed <= after
    assert "." not in value


def test_rfc3339_timestamp_strictly_advances_a_future_previous_value() -> None:
    assert _rfc3339_now_after("2999-12-31T23:59:58Z") == "2999-12-31T23:59:59Z"


def test_rfc3339_timestamp_fails_closed_when_no_later_second_exists() -> None:
    with pytest.raises(RunnerError, match="cannot be advanced"):
        _rfc3339_now_after("9999-12-31T23:59:59Z")


@pytest.mark.parametrize(
    "previous",
    (
        "2026-09-07T12:00:00.000Z",
        "2026-09-07T12:00:00+00:00",
        "2026-09-07 12:00:00Z",
        "2026-09-07T12:00:00z",
        "2026-02-30T12:00:00Z",
        "not-a-timestamp",
        "",
        False,
    ),
)
def test_rfc3339_timestamp_rejects_malformed_or_noncanonical_previous(
    previous: object,
) -> None:
    with pytest.raises(RunnerError, match="timestamp"):
        _rfc3339_now_after(previous)


def test_generated_timestamp_is_accepted_by_governance_parser() -> None:
    from cfeg import metadata_calibration_v3_governance as governance

    value = _rfc3339_now_after("2999-12-31T23:59:58Z")
    parsed = governance._parse_rfc3339_seconds(value, "runner generated timestamp")
    assert parsed == datetime(2999, 12, 31, 23, 59, 59, tzinfo=timezone.utc)


def test_runner_top_level_is_stdlib_only_and_activates_stdlib_first() -> None:
    source = _RUNNER.read_text(encoding="utf-8")
    tree = ast.parse(source)
    top_level_imports = []
    for statement in tree.body:
        if isinstance(statement, ast.Import):
            top_level_imports.extend(alias.name.split(".")[0] for alias in statement.names)
        elif isinstance(statement, ast.ImportFrom) and statement.module:
            top_level_imports.append(statement.module.split(".")[0])
    assert set(top_level_imports) <= sys.stdlib_module_names
    assert "*_EXPECTED_STANDARD_LIBRARY_PATH,\n    os.fspath(V3_REPOSITORY_SRC)" in source
    main_source = source[source.index("def main(") : source.index('if __name__ == "__main__"')]
    assert main_source.index("_run_preimport_preflight(") < main_source.index(
        "_activate_exact_import_path(preflight)"
    )
    assert source.index("governance.establish_governed_process(") < source.index(
        "from cfeg.analysis import metadata_calibration_v3_synthetic"
    )
    assert source.index("governance.governed_runner_command(") < source.index(
        "governance.establish_governed_process("
    )


def test_isolated_preflight_accepts_a_clean_temp_repo_and_runtime(tmp_path: Path) -> None:
    repository, site, bundle = _make_test_repository(
        tmp_path,
        module_source="VALUE = 1\n",
    )
    completed = _run_hostile_preflight(repository, site, bundle)
    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.strip() == "PREFLIGHT_PASSED"


def test_ignored_valid_timestamp_pyc_is_rejected_before_side_effect(
    tmp_path: Path,
) -> None:
    marker = tmp_path / "pyc-marker"
    malicious = f"from pathlib import Path\nPath({str(marker)!r}).write_text('hit')\n"
    benign = "VALUE = 1\n"
    assert len(benign) < len(malicious)
    benign = benign + "#" * (len(malicious) - len(benign))
    repository, site, bundle = _make_test_repository(tmp_path, module_source=benign)
    (repository / ".gitignore").write_text("__pycache__/\n", encoding="utf-8")
    _git(repository, "add", ".gitignore")
    _git(repository, "commit", "-m", "ignore bytecode")
    source = repository / "src/victim.py"
    source.write_text(malicious, encoding="utf-8")
    fixed_seconds = 1_700_000_000
    os.utime(source, (fixed_seconds, fixed_seconds))
    bytecode = Path(importlib.util.cache_from_source(os.fspath(source)))
    py_compile.compile(
        os.fspath(source),
        cfile=os.fspath(bytecode),
        doraise=True,
        invalidation_mode=py_compile.PycInvalidationMode.TIMESTAMP,
    )
    source.write_text(benign, encoding="utf-8")
    os.utime(source, (fixed_seconds, fixed_seconds))
    header = bytecode.read_bytes()[:16]
    assert int.from_bytes(header[4:8], "little") == 0
    assert int.from_bytes(header[8:12], "little") == fixed_seconds
    assert int.from_bytes(header[12:16], "little") == len(benign.encode("utf-8"))
    assert _git(repository, "status", "--porcelain", "--untracked-files=all") == ""

    completed = _run_hostile_preflight(repository, site, bundle)

    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.startswith("REJECTED:RunnerError:")
    assert "protected source tree" in completed.stdout
    assert not marker.exists()


def test_ignored_hidden_source_shadow_is_rejected_before_side_effect(
    tmp_path: Path,
) -> None:
    marker = tmp_path / "source-marker"
    repository, site, bundle = _make_test_repository(
        tmp_path,
        module_source="VALUE = 1\n",
    )
    (repository / ".gitignore").write_text("/src/shadow.py\n", encoding="utf-8")
    _git(repository, "add", ".gitignore")
    _git(repository, "commit", "-m", "ignore shadow")
    (repository / "src/shadow.py").write_text(
        f"from pathlib import Path\nPath({str(marker)!r}).write_text('hit')\n",
        encoding="utf-8",
    )
    assert _git(repository, "status", "--porcelain", "--untracked-files=all") == ""

    completed = _run_hostile_preflight(
        repository,
        site,
        bundle,
        module_name="shadow",
    )

    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.startswith("REJECTED:RunnerError:")
    assert "uncommitted file in protected source tree" in completed.stdout
    assert not marker.exists()


def test_bundle_at_wrong_head_is_rejected_before_new_head_side_effect(
    tmp_path: Path,
) -> None:
    marker = tmp_path / "head-marker"
    repository, site, bundle = _make_test_repository(
        tmp_path,
        module_source="VALUE = 1\n",
    )
    _write_test_bundle(repository, site, bundle)
    (repository / "src/victim.py").write_text(
        f"from pathlib import Path\nPath({str(marker)!r}).write_text('hit')\n",
        encoding="utf-8",
    )
    _git(repository, "add", "src/victim.py")
    _git(repository, "commit", "-m", "hostile head B")

    completed = _run_hostile_preflight(repository, site, bundle)

    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.startswith("REJECTED:RunnerError:")
    assert "neither bundle A nor exact selected-only B" in completed.stdout
    assert not marker.exists()


def test_bundle_a_accepts_only_the_exact_selected_method_commit_b(
    tmp_path: Path,
) -> None:
    repository, site, bundle = _make_test_repository(
        tmp_path,
        module_source="VALUE = 1\n",
    )
    _write_test_bundle(repository, site, bundle)
    selected = repository / "configs/governance/metadata_calibration_v3_selected_method_freeze.json"
    selected.parent.mkdir(parents=True)
    selected.write_bytes(b"{}\n")
    _git(repository, "add", os.fspath(selected.relative_to(repository)))
    _git(repository, "commit", "-m", "selected-only B")

    completed = _run_hostile_preflight(repository, site, bundle)

    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.strip() == "PREFLIGHT_PASSED"


def test_bundle_with_changed_site_runtime_is_rejected_before_side_effect(
    tmp_path: Path,
) -> None:
    marker = tmp_path / "runtime-marker"
    repository, site, bundle = _make_test_repository(
        tmp_path,
        module_source="VALUE = 1\n",
    )
    _write_test_bundle(repository, site, bundle)
    (site / "runtime_shadow.py").write_text(
        f"from pathlib import Path\nPath({str(marker)!r}).write_text('hit')\n",
        encoding="utf-8",
    )

    completed = _run_hostile_preflight(
        repository,
        site,
        bundle,
        module_name="runtime_shadow",
    )

    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.startswith("REJECTED:RunnerError:")
    assert "Python site inventory differs before import" in completed.stdout
    assert not marker.exists()


def test_site_inventory_schema_includes_all_regular_files_and_directories(
    tmp_path: Path,
) -> None:
    site = tmp_path / "site"
    (site / "package/__pycache__").mkdir(parents=True)
    (site / "package/module.py").write_bytes(b"VALUE = 1\n")
    (site / "package/__pycache__/module.pyc").write_bytes(b"pyc-bytes")
    inventory = _capture_python_site_inventory(site)
    entries = []
    for path in (site, site / "package", site / "package/__pycache__"):
        relative = "." if path == site else path.relative_to(site).as_posix()
        entries.append(
            {
                "kind": "directory",
                "mode": f"0{stat.S_IMODE(path.stat().st_mode):03o}",
                "path": relative,
            }
        )
    for path in (site / "package/__pycache__/module.pyc", site / "package/module.py"):
        relative = path.relative_to(site).as_posix()
        metadata = path.stat()
        entries.append(
            {
                "file_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "kind": "regular_file",
                "link_count": metadata.st_nlink,
                "mode": f"0{stat.S_IMODE(metadata.st_mode):03o}",
                "path": relative,
                "size_bytes": metadata.st_size,
            }
        )
    entries.sort(key=lambda item: item["path"])
    expected_digest = hashlib.sha256(
        _canonical_json_bytes(
            {
                "entries": entries,
                "root_path": os.fspath(site),
                "schema": _PYTHON_SITE_INVENTORY_SCHEMA,
            }
        )
    ).hexdigest()
    assert inventory == {
        "directory_count": 3,
        "inventory_sha256": expected_digest,
        "regular_file_count": 2,
        "root_path": os.fspath(site),
        "schema": _PYTHON_SITE_INVENTORY_SCHEMA,
        "total_regular_file_bytes": len(b"VALUE = 1\npyc-bytes"),
    }


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
    assert facade.events[0] == ("reopen:existing_development_bundle_after_publish_return_crash")
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
