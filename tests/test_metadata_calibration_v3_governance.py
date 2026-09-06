from __future__ import annotations

import base64
import json
import os
import subprocess
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

import cfeg.metadata_calibration_v3_governance as gov

_REPOSITORY = Path(__file__).resolve().parents[1]
_FIXTURE = _REPOSITORY / "tests/fixtures/nist_beacon_v2_20260905T180000Z.json"


def _frozen_file_bytes() -> dict[str, bytes]:
    return {path: (_REPOSITORY / path).read_bytes() for path in gov.FROZEN_FILE_SHA256}


def _commit_tiny_repository(root: Path) -> gov.CleanSourceSnapshot:
    module = root / "src/cfeg/metadata_calibration_v3_governance.py"
    module.parent.mkdir(parents=True)
    module.write_text("# immutable test source\n", encoding="utf-8")
    subprocess.run(["git", "init", "-q", os.fspath(root)], check=True)
    subprocess.run(["git", "-C", os.fspath(root), "add", "."], check=True)
    subprocess.run(
        [
            "git",
            "-C",
            os.fspath(root),
            "-c",
            "user.name=V3 Test",
            "-c",
            "user.email=v3-test@example.invalid",
            "commit",
            "-q",
            "-m",
            "fixture",
        ],
        check=True,
    )
    return gov.capture_clean_source_snapshot(root)


def _context_reference() -> dict[str, object]:
    def table(name: str, count: int) -> dict[str, object]:
        return {
            "interface_lookup_key": name,
            "centers": [float(index) / 10.0 for index in range(8)],
            "scales": [0.1] * 8,
            "observed_counts": [count] * 8,
        }

    return dict(
        gov.seal_context_reference(
            {
                "domain": "synthetic_development_and_future_synthetic",
                "canonical_channels": [f"ch{index:02d}" for index in range(8)],
                "interface_tables": [
                    table("neutral", 240),
                    table("wet", 241),
                    table("dry", 242),
                ],
                "pooled_table": table("__pooled__", 723),
                "minimum_per_interface_channel_count": 128,
                "pooled_per_channel_fallback_minimum_count": 256,
                "scale_floor": 0.05,
            }
        )
    )


def test_canonical_payload_and_file_hashes_are_distinct_and_strict() -> None:
    sealed = gov.seal_payload({"schema": "example.v1", "z": 1, "a": "x"})
    assert gov.validate_payload_hash(sealed) == sealed["payload_sha256"]
    assert gov.artifact_bytes(sealed).endswith(b"\n")
    assert gov.file_sha256(gov.artifact_bytes(sealed)) != sealed["payload_sha256"]
    assert gov.canonical_json_bytes(sealed).startswith(b'{"a":"x","payload_sha256"')

    tampered = dict(sealed)
    tampered["payload_sha256"] = "0" * 64
    with pytest.raises(gov.ValidationError, match="mismatch"):
        gov.validate_payload_hash(tampered)
    with pytest.raises(gov.ValidationError, match="keys"):
        gov.canonical_json_bytes({1: "coercion forbidden"})  # type: ignore[dict-item]
    with pytest.raises(gov.ValidationError, match="CR or LF"):
        gov._ascii_lines(("safe", "injected\nfield"))


def test_runtime_inventory_hashes_all_distribution_files_and_rejects_drift(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    inventory, fingerprint = gov._current_numerical_runtime()
    distributions = {item["distribution"]: item for item in inventory["distributions"]}
    assert {"numpy", "scipy", "PyYAML", "cryptography", "pytest"} <= set(
        distributions
    )
    assert len(distributions["scipy"]["installed_file_inventory"]) > 1_000
    assert inventory["python"]["implementation_cache_tag"]
    assert inventory["numpy_runtime_configuration_sha256"]
    altered = json.loads(gov.canonical_json_bytes(inventory))
    altered["distributions"][2]["installed_file_inventory"][0]["file_sha256"] = "0" * 64
    assert gov.numerical_runtime_fingerprint_sha256(altered) != fingerprint
    with pytest.raises(gov.AuthorityError, match="stored numerical runtime inventory"):
        gov._require_current_numerical_runtime_binding(altered, fingerprint)
    altered_fingerprint = gov.numerical_runtime_fingerprint_sha256(altered)
    monkeypatch.setattr(
        gov,
        "_current_numerical_runtime",
        lambda: (altered, altered_fingerprint),
    )
    with pytest.raises(gov.AuthorityError, match="current numerical runtime"):
        gov._require_current_numerical_runtime_binding(inventory, fingerprint)


def test_focused_a_suite_covers_all_six_v3_implementation_boundaries() -> None:
    assert gov._TEST_COMMANDS["focused_v3"][-6:] == (
        "tests/test_metadata_calibration_v3.py",
        "tests/test_metadata_calibration_v3_synthetic.py",
        "tests/test_metadata_calibration_v3_governance.py",
        "tests/test_metadata_calibration_v3_isolation.py",
        "tests/test_metadata_calibration_v3_contract.py",
        "tests/test_metadata_calibration_v3_runner.py",
    )
    environment = gov.frozen_numerical_executor_environment()
    assert "HOME" not in environment
    assert environment["PYTEST_DISABLE_PLUGIN_AUTOLOAD"] == "1"
    assert environment["OPENBLAS_NUM_THREADS"] == "1"
    with pytest.raises(TypeError):
        environment["OPENBLAS_NUM_THREADS"] = "2"  # type: ignore[index]


def test_development_result_size_ceiling_covers_exact_verbose_row_shape() -> None:
    digest = "a" * 64
    representative_row = {
        "schema": "cfeg.metadata-calibration-efficiency-v3.participant-metric.v1",
        "row_type": "participant_metric",
        "grid_cell_index": 9,
        "grid_cell_id": "p3-nu_16-lambda_0p30",
        "operator_instance_sha256": digest,
        "family": "B4_interface_calibrated_impedance_shift",
        "participant_index": 47,
        "condition": "equal_condition_composite",
        "source_range_stress": True,
        "role": "A_QM",
        "control": "within_prefix_packet_shuffle",
        "gate": "reliability_gated",
        "budget": 5,
        "balanced_accuracy": 0.9999999999999999,
        "correct_log_probability": -123.45678901234567,
        "support_enabled": True,
        "exact_A0_query_block_fraction": 0.9999999999999999,
        "helpful_metadata_use_count": 5,
        "helpful_metadata_use_denominator": 5,
        "g_M_by_query_block": [1.2345678901234567] * 5,
        "affinity_min": -123.45678901234567,
        "affinity_max": 123.45678901234567,
        "affinity_standard_deviation": 123.45678901234567,
        "pairing_sha256s": [digest] * 5,
        "support_reliability_capability_sha256s": [digest] * 5,
        "pairing_packet_binding_changed_fraction": 1.0,
        "pairing_mean_derangement_abs_g_M_change_by_query_block": [
            123.45678901234567
        ]
        * 5,
        "pairing_scientifically_changed_count": 5,
        "pairing_potential_unit_count": 5,
        "context_packet_covered_count": 5,
    }
    exact_row_bytes = len(gov.canonical_json_bytes(representative_row)) + 1
    conservative_full_size = exact_row_bytes * 88_992 + 16 * 1024 * 1024
    assert conservative_full_size > 64 * 1024 * 1024
    assert conservative_full_size < gov.DEVELOPMENT_RESULT_MAXIMUM_BYTES


def test_observed_runner_and_bundle_bind_real_process_and_clean_source(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repository = tmp_path / "tiny-repository"
    repository.mkdir()
    (repository / "test_sample.py").write_text(
        "def test_sample():\n    assert 1 + 1 == 2\n",
        encoding="utf-8",
    )
    snapshot = _commit_tiny_repository(repository)
    command = (
        os.fspath(gov.V3_PYTHON_EXECUTABLE),
        "-I",
        "-B",
        "-c",
        gov._PYTEST_ISOLATED_BOOTSTRAP,
        "-q",
        "-o",
        "xfail_strict=true",
        "-p",
        "no:cacheprovider",
        "test_sample.py",
    )
    monkeypatch.setattr(gov, "_TEST_COMMANDS", {"focused_v3": command})
    observed = gov._run_observed_test_under_test_root(
        "focused_v3",
        repository=repository,
        snapshot=snapshot,
    )
    assert observed.process_id != os.getpid()
    assert observed.passed_tests == observed.collected_tests == 1
    assert len(observed.junit_report_file_sha256) == 64
    assert observed.junit_report_path.startswith("/tmp/cfeg-v3-junit-")
    with pytest.raises(gov.AuthorityError, match="binding mismatch"):
        gov.require_observed_test_run_capability(
            observed,
            expected_name="focused_v3",
            snapshot=snapshot,
        )

    value = dict(
        gov._build_development_bundle(
            snapshot=snapshot,
            frozen_file_bytes=_frozen_file_bytes(),
            created_at_UTC="2026-09-06T12:00:00Z",
            focused_test_run=observed,
            expected_test_scope="test-only",
        )
    )
    publication_root = tmp_path / "published-bundle"
    gov._publish_development_bundle_under_test_root(
        publication_root,
        snapshot=snapshot,
        frozen_file_bytes=_frozen_file_bytes(),
        created_at_UTC="2026-09-06T12:00:00Z",
        focused_test_run=observed,
    )
    receipt = gov._validate_development_bundle_under_test_root(
        publication_root,
        snapshot=snapshot,
        frozen_file_bytes=_frozen_file_bytes(),
        focused_test_run=observed,
    )
    assert receipt.file_sha256 == gov.file_sha256(gov.artifact_bytes(value))
    assert value["tracked_source_file_inventory"] == [
        item.as_record() for item in snapshot.tracked_files
    ]
    assert value["focused_test_process_id"] == observed.process_id
    assert value["artifact_schemas"] == dict(gov.ARTIFACT_SCHEMAS)
    with pytest.raises(gov.AuthorityError, match="DevelopmentBundleCapability"):
        gov.require_development_bundle_capability(
            receipt,
            expected_commit=snapshot.identity.commit,
            expected_tree=snapshot.identity.tree,
        )


def test_git_snapshot_ignores_hostile_process_environment(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repository = tmp_path / "source"
    repository.mkdir()
    expected = _commit_tiny_repository(repository)
    fake_bin = tmp_path / "fake-bin"
    fake_bin.mkdir()
    fake_git = fake_bin / "git"
    marker = tmp_path / "fake-git-ran"
    fake_git.write_text(
        f"#!/bin/sh\nprintf hostile > {marker}\nexit 97\n",
        encoding="utf-8",
    )
    fake_git.chmod(0o755)
    hostile_git_dir = tmp_path / "hostile.git"
    subprocess.run(["git", "init", "--bare", "-q", hostile_git_dir], check=True)
    monkeypatch.setenv("PATH", os.fspath(fake_bin))
    monkeypatch.setenv("GIT_DIR", os.fspath(hostile_git_dir))
    monkeypatch.setenv("GIT_WORK_TREE", os.fspath(tmp_path))
    monkeypatch.setenv("GIT_OBJECT_DIRECTORY", os.fspath(hostile_git_dir / "objects"))
    monkeypatch.setenv("GIT_ALTERNATE_OBJECT_DIRECTORIES", os.fspath(hostile_git_dir / "objects"))
    observed = gov.capture_clean_source_snapshot(repository)
    assert observed.identity == expected.identity
    assert observed.tracked_files == expected.tracked_files
    assert not marker.exists()


def test_loaded_governance_source_cannot_be_stale_even_when_git_is_clean(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repository = tmp_path / "stale-import"
    repository.mkdir()
    snapshot = _commit_tiny_repository(repository)
    source_path = repository / "src/cfeg/metadata_calibration_v3_governance.py"
    expected_hash = next(
        item.file_sha256
        for item in snapshot.tracked_files
        if item.path == "src/cfeg/metadata_calibration_v3_governance.py"
    )
    monkeypatch.setattr(gov, "V3_SOURCE_REPOSITORY", repository)
    monkeypatch.setattr(gov, "_GOVERNANCE_IMPORT_SOURCE_PATH", source_path)
    monkeypatch.setattr(gov, "_GOVERNANCE_IMPORT_SOURCE_FILE_SHA256", "0" * 64)

    assert gov.file_sha256(source_path.read_bytes()) == expected_hash
    with pytest.raises(gov.AuthorityError, match="loaded governance module differs"):
        gov._require_loaded_governance_source_at_snapshot(snapshot)


def test_prepublication_source_drift_never_reaches_bundle_o_excl(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repository = tmp_path / "source-drift"
    repository.mkdir()
    snapshot = _commit_tiny_repository(repository)
    (repository / "new-file").write_text("drift\n", encoding="utf-8")
    subprocess.run(["git", "-C", os.fspath(repository), "add", "."], check=True)
    subprocess.run(
        [
            "git",
            "-C",
            os.fspath(repository),
            "-c",
            "user.name=V3 Test",
            "-c",
            "user.email=v3-test@example.invalid",
            "commit",
            "-q",
            "-m",
            "drift",
        ],
        check=True,
    )
    monkeypatch.setattr(gov, "V3_SOURCE_REPOSITORY", repository)
    monkeypatch.setattr(
        gov,
        "build_development_bundle",
        lambda **_kwargs: gov.seal_payload({"schema": gov.DEVELOPMENT_BUNDLE_SCHEMA}),
    )
    reached_writer = False

    def forbidden_writer(*_args: object, **_kwargs: object) -> None:
        nonlocal reached_writer
        reached_writer = True

    monkeypatch.setattr(gov, "_publish_write_once", forbidden_writer)
    with pytest.raises(gov.AuthorityError, match="immediately before publication"):
        gov.publish_development_bundle(
            snapshot=snapshot,
            frozen_file_bytes={},
            created_at_UTC="2026-09-06T12:00:00Z",
            focused_test_run=object(),  # type: ignore[arg-type]
        )
    assert not reached_writer


def test_test_and_result_publishers_repeat_guards_before_o_excl(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    value = gov.seal_payload({"schema": "test-only.guard-probe.v1"})
    calls: list[str] = []

    monkeypatch.setattr(
        gov,
        "_validate_test_evidence_bytes",
        lambda *_args, **_kwargs: calls.append("test-validated"),
    )
    monkeypatch.setattr(
        gov,
        "_require_prepublication_current_source",
        lambda _snapshot: (_ for _ in ()).throw(gov.AuthorityError("B drift")),
    )
    monkeypatch.setattr(
        gov,
        "_publish_write_once",
        lambda *_args, **_kwargs: calls.append("writer"),
    )
    with pytest.raises(gov.AuthorityError, match="B drift"):
        gov.publish_test_evidence(
            value,
            selected_method_freeze=object(),  # type: ignore[arg-type]
            observed_runs=(),
            snapshot=object(),  # type: ignore[arg-type]
        )
    assert calls == ["test-validated"]

    calls.clear()
    monkeypatch.setattr(
        gov,
        "_validate_development_result_bytes",
        lambda *_args, **_kwargs: calls.append("result-validated"),
    )
    monkeypatch.setattr(
        gov,
        "require_development_rng_bundle_capability",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            gov.AuthorityError("A drift")
        ),
    )
    with pytest.raises(gov.AuthorityError, match="A drift"):
        gov.publish_development_result(
            value,
            development_bundle=SimpleNamespace(  # type: ignore[arg-type]
                clean_commit="1" * 40,
                clean_tree="2" * 40,
            ),
            context_reference=object(),  # type: ignore[arg-type]
            development_rng_authority=object(),
            validated_context_reference=object(),
            core_capability=object(),
        )
    assert calls == ["result-validated"]

    calls.clear()
    monkeypatch.setattr(
        gov,
        "_validate_canary_claim_bytes",
        lambda *_args, **_kwargs: calls.append("canary-validated"),
    )
    monkeypatch.setattr(
        gov,
        "require_canary_artifact_capability",
        lambda *_args, **_kwargs: SimpleNamespace(
            source_commit="1" * 40,
            source_tree="2" * 40,
        ),
    )
    monkeypatch.setattr(
        gov,
        "_require_current_source_identity",
        lambda **_kwargs: (_ for _ in ()).throw(gov.AuthorityError("canary B drift")),
    )
    with pytest.raises(gov.AuthorityError, match="canary B drift"):
        gov.publish_canary_claim(
            value,
            authorization=object(),  # type: ignore[arg-type]
        )
    assert calls == ["canary-validated"]


def test_observed_runner_rejects_source_mutation_even_after_zero_exit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repository = tmp_path / "mutating-repository"
    repository.mkdir()
    (repository / "test_mutation.py").write_text(
        (
            "from pathlib import Path\n\n"
            "def test_mutates_source():\n"
            "    Path('src/cfeg/metadata_calibration_v3_governance.py').write_text('changed')\n"
        ),
        encoding="utf-8",
    )
    snapshot = _commit_tiny_repository(repository)
    command = (
        os.fspath(gov.V3_PYTHON_EXECUTABLE),
        "-I",
        "-B",
        "-c",
        gov._PYTEST_ISOLATED_BOOTSTRAP,
        "-q",
        "-o",
        "xfail_strict=true",
        "-p",
        "no:cacheprovider",
        "test_mutation.py",
    )
    monkeypatch.setattr(gov, "_TEST_COMMANDS", {"focused_v3": command})
    with pytest.raises(gov.ValidationError, match="already-clean worktree"):
        gov._run_observed_test_under_test_root(
            "focused_v3",
            repository=repository,
            snapshot=snapshot,
        )


def test_observed_runner_rejects_skips_and_malformed_junit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repository = tmp_path / "receipt-repository"
    repository.mkdir()
    (repository / "test_skip.py").write_text(
        "import pytest\n\n@pytest.mark.skip(reason='not allowed')\ndef test_skip():\n    pass\n",
        encoding="utf-8",
    )
    snapshot = _commit_tiny_repository(repository)
    monkeypatch.setattr(
        gov,
        "_TEST_COMMANDS",
        {
            "focused_v3": (
                os.fspath(gov.V3_PYTHON_EXECUTABLE),
                "-I",
                "-B",
                "-c",
                gov._PYTEST_ISOLATED_BOOTSTRAP,
                "-q",
                "-o",
                "xfail_strict=true",
                "-p",
                "no:cacheprovider",
                "test_skip.py",
            )
        },
    )
    with pytest.raises(gov.AuthorityError, match="did not pass completely"):
        gov._run_observed_test_under_test_root(
            "focused_v3",
            repository=repository,
            snapshot=snapshot,
        )
    with pytest.raises(gov.AuthorityError, match="malformed"):
        gov._parse_pytest_junit_report(b"999 passed in 0.01s")


def test_observed_runner_never_chmods_a_symlinked_junit_target(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repository = tmp_path / "junit-symlink-repository"
    repository.mkdir()
    snapshot = _commit_tiny_repository(repository)
    victim = tmp_path / "victim"
    victim.write_bytes(b"do not touch")
    victim.chmod(0o600)
    script = (
        "import os,sys; "
        "path=sys.argv[-1].split('=',1)[1]; "
        f"os.symlink({os.fspath(victim)!r},path)"
    )
    monkeypatch.setattr(
        gov,
        "_TEST_COMMANDS",
        {"focused_v3": (os.fspath(gov.V3_PYTHON_EXECUTABLE), "-c", script)},
    )
    with pytest.raises(OSError):
        gov._run_observed_test_under_test_root(
            "focused_v3",
            repository=repository,
            snapshot=snapshot,
        )
    assert victim.read_bytes() == b"do not touch"
    assert victim.stat().st_mode & 0o777 == 0o600


def test_selected_source_delta_allows_only_new_frozen_blob() -> None:
    before = (
        gov.TrackedSourceFile(path="src/cfeg/core.py", kind="regular_file", file_sha256="1" * 64),
    )
    selected = gov.TrackedSourceFile(
        path=gov.SELECTED_METHOD_FREEZE_REPOSITORY_PATH.as_posix(),
        kind="regular_file",
        file_sha256="2" * 64,
    )
    gov._require_exact_selected_inventory_delta(before, (*before, selected))

    changed = gov.TrackedSourceFile(
        path="src/cfeg/core.py",
        kind="regular_file",
        file_sha256="3" * 64,
    )
    with pytest.raises(gov.AuthorityError, match="only by one"):
        gov._require_exact_selected_inventory_delta(before, (changed, selected))
    extra = gov.TrackedSourceFile(
        path="src/cfeg/unreviewed.py",
        kind="regular_file",
        file_sha256="4" * 64,
    )
    with pytest.raises(gov.AuthorityError, match="only by one"):
        gov._require_exact_selected_inventory_delta(before, (*before, selected, extra))


def test_selected_reopen_uses_only_durable_b_audit_not_ephemeral_a_proposal(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    core = ModuleType("cfeg.analysis.metadata_calibration_v3_synthetic")
    monkeypatch.setitem(
        sys.modules,
        "cfeg.analysis.metadata_calibration_v3_synthetic",
        core,
    )
    result_data = b"durable development result bytes"
    result_payload_sha256 = "1" * 64
    result_file_sha256 = gov.file_sha256(result_data)
    result = {
        "payload_sha256": result_payload_sha256,
        "complete_grid_gate_report": [],
    }
    selected = {"payload_sha256": "2" * 64}
    selected_data = b"committed selected freeze bytes"
    recovery = SimpleNamespace(
        bundle_commit="7" * 40,
        bundle_tree="8" * 40,
        selected_commit="3" * 40,
        selected_tree="4" * 40,
    )
    bundle = SimpleNamespace(
        schema=gov.DEVELOPMENT_BUNDLE_SCHEMA,
        payload_sha256="5" * 64,
        file_sha256="6" * 64,
        clean_commit="7" * 40,
        clean_tree="8" * 40,
    )
    result_capability = SimpleNamespace(
        schema=gov.ARTIFACT_SCHEMAS["development_result"],
        payload_sha256=result_payload_sha256,
        file_sha256=result_file_sha256,
    )
    expected = object()
    observed: list[str] = []

    monkeypatch.setattr(
        gov,
        "_load_durable_development_selection_graph",
        lambda _capability: (
            recovery,
            selected,
            selected_data,
            result,
            result_data,
        ),
    )
    monkeypatch.setattr(
        gov,
        "require_development_bundle_capability",
        lambda *_args, **_kwargs: bundle,
    )
    monkeypatch.setattr(
        gov,
        "require_validated_artifact_capability",
        lambda *_args, **_kwargs: result_capability,
    )
    monkeypatch.setattr(
        gov,
        "_require_core_selected_method_proposal",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("ephemeral A proposal must not be used during B recovery")
        ),
    )

    def audit(
        value: object,
        *,
        development_result: object,
        development_result_file_sha256: str,
        recovery_capability: object,
    ) -> object:
        assert value is selected
        assert development_result is result
        assert development_result_file_sha256 == result_file_sha256
        assert recovery_capability is recovery
        observed.append("durable-core-audit")
        return selected

    monkeypatch.setattr(
        core,
        "validate_selected_method_freeze_for_audit",
        audit,
        raising=False,
    )
    monkeypatch.setattr(
        gov,
        "_validate_schema_artifact",
        lambda *_args, **_kwargs: expected,
    )

    assert (
        gov.reopen_selected_method_freeze(
            development_bundle=bundle,  # type: ignore[arg-type]
            development_result=result_capability,  # type: ignore[arg-type]
            recovery_capability=recovery,  # type: ignore[arg-type]
        )
        is expected
    )
    assert observed == ["durable-core-audit"]


def test_context_authority_requires_exact_core_nominal_capability() -> None:
    value = _context_reference()
    with pytest.raises((gov.AuthorityError, TypeError, ValueError)):
        gov.publish_context_reference(
            value,
            core_capability={"payload_sha256": value["payload_sha256"]},
        )


def test_context_cross_seam_tmp_publication_stays_test_only(tmp_path: Path) -> None:
    core = pytest.importorskip("cfeg.models.metadata_calibration_v3")
    value = _context_reference()
    with pytest.raises(TypeError):
        core.validate_context_reference_for_publication(value)
    with pytest.raises(TypeError):
        gov._publish_context_reference_under_test_root(
            tmp_path / "context-publication",
            value,
            core_capability=object(),
        )


def test_fabricated_development_summary_cannot_mint_authority() -> None:
    content = {
        "generator_revision": "v1_single_insertion_context_trust",
        "master_plan_file_sha256": gov.FROZEN_FILE_SHA256[gov.MASTER_PLAN_PATH],
        "synthetic_plan_file_sha256": gov.FROZEN_FILE_SHA256[gov.SYNTHETIC_PLAN_PATH],
        "development_bundle_schema": gov.DEVELOPMENT_BUNDLE_SCHEMA,
        "development_bundle_payload_sha256": "1" * 64,
        "development_bundle_file_sha256": "2" * 64,
        "context_reference_schema": gov.ARTIFACT_SCHEMAS["context_reference"],
        "context_reference_payload_sha256": "3" * 64,
        "context_reference_file_sha256": "4" * 64,
        "development_rng_primitive_schema": (
            "cfeg.metadata-calibration-efficiency-v3.development-rng-binding.v1"
        ),
        "development_root_seed": 20_260_909,
        "participant_count": 48,
        "B4_source_range_stress_participant_indices": [0, 1, 2, 3, 4],
        "grid_cells": [
            dict(zip(gov._GRID_CELL_FIELDS, cell, strict=True)) for cell in gov._GRID_CELLS
        ],
        "participant_metric_rows": 88_992,
        "invariant_rows": 90,
        "complete_grid_gate_report": {"all_named_components_pass": True},
        "selection_status": "SELECTED_METHOD_PROPOSED",
        "selected_grid_cell_id": "p3-nu_4-lambda_0p20",
    }
    with pytest.raises(gov.ValidationError, match="89,082 exact rows"):
        gov.seal_development_result(content)


def test_a_result_reopen_uses_canonical_audit_without_rng_reexecution(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    core = ModuleType("cfeg.analysis.metadata_calibration_v3_synthetic")
    monkeypatch.setitem(
        sys.modules,
        "cfeg.analysis.metadata_calibration_v3_synthetic",
        core,
    )
    result = dict(
        gov.seal_payload({"schema": gov.ARTIFACT_SCHEMAS["development_result"]})
    )
    context_payload = dict(
        gov.seal_payload({"schema": gov.ARTIFACT_SCHEMAS["context_reference"]})
    )
    context = SimpleNamespace(_validated_payload=context_payload)
    bundle = SimpleNamespace(schema=gov.DEVELOPMENT_BUNDLE_SCHEMA)
    audited_proof = SimpleNamespace(selection_status="DEVELOPMENT_NO_GO")
    recovery = SimpleNamespace(
        _validated_payload=result,
        _context_reference=context,
        _bundle=bundle,
        development_bundle_payload_sha256="3" * 64,
        development_bundle_file_sha256="4" * 64,
        context_reference_schema=gov.ARTIFACT_SCHEMAS["context_reference"],
        context_reference_payload_sha256=context_payload["payload_sha256"],
        context_reference_file_sha256="5" * 64,
    )
    expected = object()
    observed: list[str] = []
    monkeypatch.setattr(
        gov,
        "observe_canonical_development_result_for_a_recovery",
        lambda **_kwargs: recovery,
    )
    monkeypatch.setattr(
        gov,
        "require_canonical_development_result_recovery_capability",
        lambda value: value,
    )

    def audit(
        value: object,
        *,
        context_reference: object,
        recovery_capability: object,
    ) -> object:
        assert value is result
        assert context_reference is context._validated_payload
        assert recovery_capability is recovery
        observed.append("A-canonical-audit")
        return audited_proof

    monkeypatch.setattr(
        core,
        "validate_development_result_for_a_recovery",
        audit,
        raising=False,
    )
    monkeypatch.setattr(
        core,
        "execute_complete_development",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("A recovery must not re-execute synthetic development")
        ),
        raising=False,
    )
    monkeypatch.setattr(
        gov,
        "_require_core_audited_development_result",
        lambda value, **_kwargs: value,
    )
    monkeypatch.setattr(
        gov,
        "_validate_schema_artifact",
        lambda *_args, **_kwargs: expected,
    )

    recovered_artifact, recovered_audit = (
        gov.recover_canonical_development_result_at_bundle_source(
            development_bundle=bundle,  # type: ignore[arg-type]
            context_reference=context,  # type: ignore[arg-type]
        )
    )
    assert recovered_artifact is expected
    assert recovered_audit is audited_proof
    assert observed == ["A-canonical-audit"]


def test_a_selection_recovery_requires_the_sealed_core_audit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    core = ModuleType("cfeg.analysis.metadata_calibration_v3_synthetic")
    monkeypatch.setitem(
        sys.modules,
        "cfeg.analysis.metadata_calibration_v3_synthetic",
        core,
    )
    result_payload = dict(
        gov.seal_payload(
            {
                "schema": gov.ARTIFACT_SCHEMAS["development_result"],
                "complete_grid_gate_report": [],
                "selected_grid_cell_id": "p3-nu_4-lambda_0p20",
            }
        )
    )
    result = SimpleNamespace(
        schema=gov.ARTIFACT_SCHEMAS["development_result"],
        payload_sha256=result_payload["payload_sha256"],
        file_sha256="1" * 64,
        _validated_payload=result_payload,
    )
    audited = SimpleNamespace(
        payload_schema=result.schema,
        payload_sha256=result.payload_sha256,
        development_result_file_sha256=result.file_sha256,
        development_bundle_payload_sha256="2" * 64,
        development_bundle_file_sha256="3" * 64,
        selection_status="SELECTED_METHOD_PROPOSED",
        selected_grid_cell_id="p3-nu_4-lambda_0p20",
        clean_commit="4" * 40,
        clean_tree="5" * 40,
    )
    gate_sha256 = gov.file_sha256(
        gov.canonical_json_bytes(
            {
                "schema": (
                    "cfeg.metadata-calibration-efficiency-v3."
                    "complete-grid-gate-report.v1"
                ),
                "reports": [],
            }
        )
    )
    selected = dict(
        gov.seal_payload(
            {
                "schema": gov.ARTIFACT_SCHEMAS["selected_method_freeze"],
                "selected_grid_cell_id": audited.selected_grid_cell_id,
                "master_plan_file_sha256": gov.FROZEN_FILE_SHA256[gov.MASTER_PLAN_PATH],
                "synthetic_plan_file_sha256": gov.FROZEN_FILE_SHA256[
                    gov.SYNTHETIC_PLAN_PATH
                ],
                "development_bundle_schema": gov.DEVELOPMENT_BUNDLE_SCHEMA,
                "development_bundle_payload_sha256": (
                    audited.development_bundle_payload_sha256
                ),
                "development_bundle_file_sha256": audited.development_bundle_file_sha256,
                "development_result_schema": result.schema,
                "development_result_payload_sha256": result.payload_sha256,
                "development_result_file_sha256": result.file_sha256,
                "complete_grid_gate_report_sha256": gate_sha256,
                "clean_commit": audited.clean_commit,
                "clean_tree": audited.clean_tree,
            }
        )
    )
    monkeypatch.setattr(
        gov,
        "require_validated_artifact_capability",
        lambda *_args, **_kwargs: result,
    )

    def require_audit(value: object, **_kwargs: object) -> object:
        assert value is audited
        return value

    def build(value: object, *, audited_development_result: object) -> object:
        assert value is result_payload
        assert audited_development_result is audited
        return selected

    monkeypatch.setattr(core, "require_audited_development_result", require_audit, raising=False)
    monkeypatch.setattr(
        core,
        "build_selected_method_freeze_payload_for_a_recovery",
        build,
        raising=False,
    )
    monkeypatch.setattr(gov, "parse_artifact_bytes", lambda *_args, **_kwargs: selected)
    monkeypatch.setattr(
        gov,
        "_validate_selected_method_freeze_content",
        lambda _value: None,
    )

    with pytest.raises(TypeError):
        gov.build_selected_method_freeze_from_canonical_development_result(
            development_result=result,  # type: ignore[arg-type]
        )
    recovered = gov.build_selected_method_freeze_from_canonical_development_result(
        development_result=result,  # type: ignore[arg-type]
        audited_development_result=audited,
    )
    assert recovered is selected


def test_lifecycle_cannot_be_constructed_at_arbitrary_state_or_use_mappings() -> None:
    assert not hasattr(gov.ScientificLifecycle, "_issued")
    assert not hasattr(gov.CanaryLifecycle, "_issued")
    with pytest.raises(TypeError):
        gov.ScientificLifecycle(state=gov.ScientificState.VERIFIED)  # type: ignore[call-arg]
    with pytest.raises(TypeError):
        gov.CanaryLifecycle(state=gov.CanaryState.CANARY_RESULT_PUBLISHED)  # type: ignore[call-arg]
    with pytest.raises(gov.TransitionError, match="cross-domain"):
        gov.ScientificLifecycle().advance(gov.CanaryState.CANARY_AUTHORIZED)
    with pytest.raises(gov.AuthorityError, match="DevelopmentBundleCapability"):
        gov.ScientificLifecycle().advance(
            gov.ScientificState.BUNDLE_FROZEN,
            authority={"schema": gov.DEVELOPMENT_BUNDLE_SCHEMA},
        )
    with pytest.raises(gov.AuthorityError, match="CanaryArtifactCapability"):
        gov.CanaryLifecycle().advance(
            gov.CanaryState.CANARY_AUTHORIZED,
            authority={"schema": gov.ARTIFACT_SCHEMAS["canary_authorization"]},
        )
    forged = object.__new__(gov.ScientificLifecycle)
    with pytest.raises(gov.AuthorityError, match="issued lifecycle"):
        forged.advance(gov.ScientificState.BUNDLE_FROZEN)

    mutated_scientific = gov.ScientificLifecycle()
    object.__setattr__(
        mutated_scientific,
        "state",
        gov.ScientificState.BUNDLE_FROZEN,
    )
    object.__setattr__(
        mutated_scientific,
        "trace",
        (
            gov.ScientificState.DECLARED,
            gov.ScientificState.BUNDLE_FROZEN,
        ),
    )
    with pytest.raises(gov.AuthorityError, match="sealed lifecycle chain"):
        mutated_scientific.advance(
            gov.ScientificState.DEVELOPMENT_COMPLETE,
            authority=object(),
        )

    mutated_canary = gov.CanaryLifecycle()
    object.__setattr__(
        mutated_canary,
        "state",
        gov.CanaryState.CANARY_AUTHORIZED,
    )
    object.__setattr__(
        mutated_canary,
        "trace",
        (
            gov.CanaryState.CANARY_DECLARED,
            gov.CanaryState.CANARY_AUTHORIZED,
        ),
    )
    with pytest.raises(gov.AuthorityError, match="sealed lifecycle chain"):
        mutated_canary.advance(
            gov.CanaryState.CANARY_CLAIMED,
            authority=object(),
        )


def test_tokenless_internal_issuers_cannot_mint_production_authority() -> None:
    with pytest.raises(TypeError, match="canonical A-stage disk observation"):
        gov.CanonicalDevelopmentResultRecoveryCapability()
    with pytest.raises(TypeError):
        gov._issue_canonical_development_result_recovery_capability(  # type: ignore[call-arg]
            bundle=object(),
            context_reference=object(),
            result={},
            result_file_bytes=b"{}\n",
            snapshot=object(),
        )
    with pytest.raises(gov.AuthorityError, match="canonical observation"):
        gov._issue_canonical_development_result_recovery_capability(
            bundle=object(),  # type: ignore[arg-type]
            context_reference=object(),  # type: ignore[arg-type]
            result={},
            result_file_bytes=b"{}\n",
            snapshot=object(),  # type: ignore[arg-type]
            _observer=object(),
        )
    with pytest.raises(TypeError):
        gov._issue_development_bundle_capability(  # type: ignore[call-arg]
            {},
            file_bytes=b"{}\n",
            snapshot=object(),
        )
    with pytest.raises(gov.AuthorityError, match="canonical observation"):
        gov._issue_development_bundle_capability(
            {},
            file_bytes=b"{}\n",
            snapshot=object(),  # type: ignore[arg-type]
            _observer=object(),
        )


def test_fresh_audit_requires_pinned_isolated_exec_and_parent_observation() -> None:
    assert not hasattr(gov, "publish_fresh_canary_audit")
    assert not hasattr(gov, "_publish_fresh_canary_audit_from_exec_child")
    with pytest.raises(TypeError, match="pinned parent launcher"):
        gov.ObservedFreshCanaryExecCapability()
    with pytest.raises(gov.AuthorityError, match="fresh audit child"):
        gov._assert_fresh_canary_exec_child_context()
    with pytest.raises(gov.AuthorityError, match="pinned parent observer"):
        gov._issue_observed_fresh_exec_capability(
            process=object(),  # type: ignore[arg-type]
            receipt_bytes=b"{}\n",
            snapshot=object(),  # type: ignore[arg-type]
            _observer=object(),
        )

    read_fd, write_fd = os.pipe()
    child = os.fork()
    if child == 0:  # pragma: no cover - assertion is observed by the parent pipe
        os.close(read_fd)
        try:
            gov._assert_fresh_canary_exec_child_context()
        except gov.AuthorityError:
            os.write(write_fd, b"rejected")
            os._exit(0)
        os.write(write_fd, b"accepted")
        os._exit(1)
    os.close(write_fd)
    observed = os.read(read_fd, 32)
    os.close(read_fd)
    _, status = os.waitpid(child, 0)
    assert observed == b"rejected"
    assert os.waitstatus_to_exitcode(status) == 0
    with pytest.raises(TypeError):
        gov._validate_schema_artifact(  # type: ignore[call-arg]
            {},
            file_bytes=b"{}\n",
            schema="fake.v1",
            exact_fields=frozenset(),
            expected_bindings={},
            canonical_path=gov.CONTEXT_REFERENCE_CANONICAL_PATH,
            scope="canonical-publication",
        )
    with pytest.raises(gov.AuthorityError, match="canonical observation"):
        gov._validate_schema_artifact(
            {},
            file_bytes=b"{}\n",
            schema="fake.v1",
            exact_fields=frozenset(),
            expected_bindings={},
            canonical_path=gov.CONTEXT_REFERENCE_CANONICAL_PATH,
            scope="canonical-publication",
            _observer=object(),
        )
    with pytest.raises(TypeError):
        gov._issue_canary_artifact_from_canonical_path(  # type: ignore[call-arg]
            gov.CANARY_RESULT_CANONICAL_PATH,
            value={},
            file_bytes=b"{}\n",
            schema=gov.ARTIFACT_SCHEMAS["canary_result"],
            exact_fields=frozenset(),
        )
    with pytest.raises(gov.AuthorityError, match="canonical observation"):
        gov._issue_canary_artifact_from_canonical_path(
            gov.CANARY_RESULT_CANONICAL_PATH,
            value={},
            file_bytes=b"{}\n",
            schema=gov.ARTIFACT_SCHEMAS["canary_result"],
            exact_fields=frozenset(),
            source_commit="1" * 40,
            source_tree="2" * 40,
            _observer=object(),
        )


def test_canary_seed_and_probe_exact_oracles_never_enter_scientific_executor() -> None:
    seed = gov.validate_canary_fixture(
        _FIXTURE.read_bytes(),
        clock=gov.HistoricalCanaryClock(),
    )
    assert seed.seed_digest_sha256 == gov.CANARY_SEED_DIGEST
    assert seed.full_unsigned_256_bit_root_seed == gov.CANARY_ROOT_SEED
    assert gov.canary_probe_digest(
        seed,
        prevalidation_trace=tuple(gov.CanaryState)[:4],
    ) == gov.CANARY_PROBE_DIGEST
    with pytest.raises(gov.AuthorityError, match="ScientificSeedCapability"):
        gov.scientific_seed_sequence(seed)  # type: ignore[arg-type]
    with pytest.raises(gov.AuthorityError, match="ScientificBeaconCapability"):
        gov.derive_scientific_seed(beacon=seed)  # type: ignore[arg-type]


def test_seed_statement_is_target_bound_noncircular_and_rejects_endpoint_drift() -> None:
    statement = gov.build_seed_statement(
        scientific_candidate_id=f"{gov.CANDIDATE_ID}-p3-nu_4-lambda_0p20",
        selected_method_freeze_payload_sha256="a" * 64,
        exact_target_timestamp_UTC="2030-01-01T00:10:00Z",
        exact_HTTPS_endpoint=(
            "https://beacon.nist.gov/beacon/2.0/pulse/time/1893456600000"
        ),
    )
    assert statement.endswith(b"\n")
    assert b"attempt" not in statement
    assert base64.b64decode(base64.b64encode(statement), validate=True) == statement
    with pytest.raises(gov.ValidationError, match="target-bound NIST"):
        gov.build_seed_statement(
            scientific_candidate_id=f"{gov.CANDIDATE_ID}-p3-nu_4-lambda_0p20",
            selected_method_freeze_payload_sha256="a" * 64,
            exact_target_timestamp_UTC="2030-01-01T00:10:00Z",
            exact_HTTPS_endpoint="https://example.com/beacon",
        )


def test_future_authority_and_human_outcomes_fail_closed() -> None:
    with pytest.raises(gov.AuthorityError, match="future-beacon clock"):
        gov.LiveFutureBeaconClock("2030-01-01T00:10:00Z")
    with pytest.raises(gov.AuthorityError, match="future beacon/endpoint"):
        gov.validate_scientific_beacon({})
    with pytest.raises(gov.AuthorityError, match="scientific execution"):
        gov.publish_scientific_global_claim({}, {})  # type: ignore[arg-type]
    with pytest.raises(gov.AuthorityError, match="human EEG"):
        gov.assert_human_EEG_outcome_authorized()


def test_pinned_owner_public_key_is_exact_rsa4096_without_private_material() -> None:
    public_bytes = (_REPOSITORY / gov.OWNER_AUTHORITY_PUBLIC_KEY_PATH).read_bytes()
    key = gov._validate_owner_public_key_bytes(public_bytes)
    assert key.key_size == 4096
    assert gov.file_sha256(public_bytes) == gov.FROZEN_FILE_SHA256[
        gov.OWNER_AUTHORITY_PUBLIC_KEY_PATH
    ]
    with pytest.raises(gov.ValidationError, match="file hash mismatch"):
        gov._validate_owner_public_key_bytes(public_bytes + b"\n")


def test_scientific_seed_digest_uses_full_256_bits_without_running_science() -> None:
    digest = gov._scientific_seed_digest(
        attempt_manifest_payload_sha256="b" * 64,
        exact_target_timestamp_UTC="2030-01-01T00:10:00Z",
        chain_index=7,
        pulse_index=123_456,
        output_value="A" * 128,
    )
    expected = gov.file_sha256(
        (
            f"{gov.SCIENTIFIC_SEED_DOMAIN}\n{'b' * 64}\n"
            f"2030-01-01T00:10:00Z\n7\n123456\n{'A' * 128}\n"
        ).encode("ascii")
    )
    assert digest == expected
    assert int(digest, 16).bit_length() > 128
