from __future__ import annotations

import ast
import hashlib
import importlib.util
import io
import os
import py_compile
import shutil
import stat
import subprocess
import sys
from pathlib import Path

import pytest

from cfeg.metadata_calibration_v3_runner import (
    _CANDIDATE_ID,
    _DEVELOPMENT_BUNDLE_SCHEMA,
    _PYTHON_SITE_INVENTORY_SCHEMA,
    RunnerError,
    RunStage,
    WorkflowInspection,
    _build_parser,
    _canonical_json_bytes,
    _capture_python_record,
    _capture_python_site_inventory,
    _capture_source_facts,
    _PreflightConfig,
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
_PYTHON = Path("/home/whwovy/califreeEEG/.venv/bin/python")
_GIT = Path("/usr/bin/git")
_STANDARD_LIBRARY = (
    "/usr/lib/python310.zip",
    "/usr/lib/python3.10",
    "/usr/lib/python3.10/lib-dynload",
)
_RUNNER_REPOSITORY_PATH = "src/cfeg/metadata_calibration_v3_runner.py"

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
            "candidate_id": _CANDIDATE_ID,
            "canonical_path": os.fspath(bundle),
            "clean_git_commit": source.commit,
            "clean_git_tree": source.tree,
            "human_EEG_outcome_authorized": False,
            "numerical_runtime_fingerprint_sha256": hashlib.sha256(
                _canonical_json_bytes(runtime)
            ).hexdigest(),
            "numerical_runtime_inventory": runtime,
            "schema": _DEVELOPMENT_BUNDLE_SCHEMA,
            "scientific_lockbox_authorized": False,
            "source_bundle_sha256": source.source_bundle_sha256,
            "tracked_source_file_inventory": [item.record() for item in source.tracked_files],
        }
    )
    bundle.parent.mkdir(parents=True)
    bundle.write_bytes(_canonical_json_bytes(payload) + b"\n")
    bundle.chmod(0o400)


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
        env={
            "LANG": "C.UTF-8",
            "LC_ALL": "C.UTF-8",
            "PATH": "/usr/bin:/bin",
        },
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
    assert (
        "exec /home/whwovy/califreeEEG/.venv/bin/python -I -B -S \\\n"
        "    /home/whwovy/califreeEEG/src/cfeg/metadata_calibration_v3_runner.py" in wrapper
    )
    assert " -c " not in wrapper
    assert "import " not in wrapper
    assert os.stat(_WRAPPER).st_mode & 0o111


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
