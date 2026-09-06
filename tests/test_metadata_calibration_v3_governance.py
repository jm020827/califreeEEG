from __future__ import annotations

import base64
import json
import os
import subprocess
import sys
import tempfile
from collections.abc import Iterator
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

import cfeg.metadata_calibration_v3_governance as gov

_REPOSITORY = Path(__file__).resolve().parents[1]
_FIXTURE = _REPOSITORY / "tests/fixtures/nist_beacon_v2_20260905T180000Z.json"
_TEST_ONLY_PYTEST_BOOTSTRAP = (
    gov._GOVERNED_PYTHON_BOOTSTRAP_PREAMBLE
    + f"_site_prefix={gov._PYTHON_SITE_INVENTORY_ARG_PREFIX!r}\n"
    + "_site_args=[i for i,v in enumerate(sys.argv) if v.startswith(_site_prefix)]\n"
    + "if len(_site_args)!=1:\n raise RuntimeError('missing test site digest')\n"
    + "del sys.argv[_site_args[0]]\n"
    + "import pytest\n"
    + "raise SystemExit(pytest.main(sys.argv[1:]))\n"
)


def _test_only_pytest_command(*paths: str) -> tuple[str, ...]:
    return (
        *gov.governed_python_subprocess_prefix(),
        "-c",
        _TEST_ONLY_PYTEST_BOOTSTRAP,
        "-q",
        "-c",
        "pyproject.toml",
        "--rootdir=.",
        "-o",
        "addopts=",
        "-o",
        "pythonpath=",
        "-o",
        "xfail_strict=true",
        "--import-mode=importlib",
        "-p",
        "no:cacheprovider",
        *paths,
    )


@pytest.fixture
def private_test_path() -> Iterator[Path]:
    with tempfile.TemporaryDirectory(prefix="cfeg-v3-test-", dir="/tmp") as raw:
        path = Path(raw)
        path.chmod(0o700)
        yield path


def _frozen_file_bytes() -> dict[str, bytes]:
    return {path: (_REPOSITORY / path).read_bytes() for path in gov.FROZEN_FILE_SHA256}


def _commit_tiny_repository(
    root: Path,
    *,
    include_frozen_files: bool = False,
) -> gov.CleanSourceSnapshot:
    module = root / "src/cfeg/metadata_calibration_v3_governance.py"
    module.parent.mkdir(parents=True)
    module.write_text("# immutable test source\n", encoding="utf-8")
    (root / "pyproject.toml").write_text(
        '[tool.pytest.ini_options]\naddopts = "--basetemp=.pytest-temp"\n',
        encoding="utf-8",
    )
    if include_frozen_files:
        for relative_path, data in _frozen_file_bytes().items():
            path = root / relative_path
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(data)
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
    assert {"numpy", "scipy", "PyYAML", "cryptography", "pytest"} <= set(distributions)
    assert {
        "exceptiongroup",
        "iniconfig",
        "packaging",
        "pluggy",
        "Pygments",
        "tomli",
        "typing_extensions",
        "cffi",
        "pycparser",
    } <= set(distributions)
    assert len(distributions["scipy"]["installed_file_inventory"]) > 1_000
    assert all(
        set(record) == {"path", "size_bytes", "mode", "file_sha256"}
        for record in distributions["pluggy"]["installed_file_inventory"]
    )
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


def test_pluggy_dependency_drift_changes_the_runtime_fingerprint() -> None:
    inventory, fingerprint = gov._current_numerical_runtime()
    altered = json.loads(gov.canonical_json_bytes(inventory))
    pluggy = next(item for item in altered["distributions"] if item["distribution"] == "pluggy")
    pluggy["installed_file_inventory"][0]["file_sha256"] = "0" * 64
    pluggy["installed_file_inventory_sha256"] = gov.file_sha256(
        gov.canonical_json_bytes({"files": pluggy["installed_file_inventory"]})
    )
    assert gov.numerical_runtime_fingerprint_sha256(altered) != fingerprint
    with pytest.raises(gov.AuthorityError, match="stored numerical runtime inventory"):
        gov._require_current_numerical_runtime_binding(altered, fingerprint)


def test_complete_site_inventory_binds_pyc_pth_paths_modes_and_hardlinks(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    site = tmp_path / "site-packages"
    cache = site / "demo/__pycache__"
    cache.mkdir(parents=True)
    source = site / "demo/module.py"
    bytecode = cache / "module.cpython-310.pyc"
    path_control = site / "editable.pth"
    source.write_bytes(b"value = 1\n")
    bytecode.write_bytes(b"bound bytecode\n")
    path_control.write_bytes(b"/exact/source\n")
    hardlink = site / "demo/module-hardlink.py"
    os.link(source, hardlink)
    monkeypatch.setattr(gov, "V3_SITE_PACKAGES", site)

    baseline = gov._capture_complete_site_packages_inventory()
    assert baseline["schema"] == (
        "cfeg.metadata-calibration-efficiency-v3.python-site-inventory.v1"
    )
    assert baseline["regular_file_count"] == 4
    assert baseline["directory_count"] == 3

    bytecode.write_bytes(b"changed bytecode\n")
    changed_bytes = gov._capture_complete_site_packages_inventory()
    assert changed_bytes["inventory_sha256"] != baseline["inventory_sha256"]
    bytecode.write_bytes(b"bound bytecode\n")
    source.chmod(0o600)
    changed_mode = gov._capture_complete_site_packages_inventory()
    assert changed_mode["inventory_sha256"] != baseline["inventory_sha256"]
    source.chmod(0o644)
    path_control.unlink()
    missing = gov._capture_complete_site_packages_inventory()
    assert missing["inventory_sha256"] != baseline["inventory_sha256"]
    path_control.write_bytes(b"/exact/source\n")
    (site / "unexpected.py").write_bytes(b"hostile = True\n")
    extra = gov._capture_complete_site_packages_inventory()
    assert extra["inventory_sha256"] != baseline["inventory_sha256"]


@pytest.mark.parametrize("kind", ("symlink", "fifo"))
def test_complete_site_inventory_rejects_links_and_special_files(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    kind: str,
) -> None:
    site = tmp_path / "site-packages"
    site.mkdir()
    target = site / "target.py"
    target.write_bytes(b"safe = True\n")
    hostile = site / "hostile"
    if kind == "symlink":
        hostile.symlink_to(target)
    else:
        os.mkfifo(hostile)
    monkeypatch.setattr(gov, "V3_SITE_PACKAGES", site)
    with pytest.raises(gov.AuthorityError, match="unsafe type"):
        gov._capture_complete_site_packages_inventory()


def test_complete_site_inventory_rejects_directory_swap(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    site = tmp_path / "site-packages"
    package = site / "demo"
    package.mkdir(parents=True)
    (package / "module.py").write_bytes(b"safe = True\n")
    monkeypatch.setattr(gov, "V3_SITE_PACKAGES", site)
    original_open = gov.os.open
    swapped = False
    monkeypatch.setattr(gov, "_require_dirfd_primitives", lambda: None)

    def swap_before_open(
        path: object,
        flags: int,
        *args: object,
        **kwargs: object,
    ) -> int:
        nonlocal swapped
        if path == "demo" and flags & os.O_DIRECTORY and not swapped:
            swapped = True
            package.rename(site / "demo-old")
            package.mkdir()
        return original_open(path, flags, *args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(gov.os, "open", swap_before_open)
    with pytest.raises(gov.AuthorityError, match="changed during traversal"):
        gov._capture_complete_site_packages_inventory()


def test_runtime_mismatch_rejects_before_external_import(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    site = tmp_path / "site-packages"
    site.mkdir()
    dependency = site / "dependency.py"
    dependency.write_bytes(b"side_effect = False\n")
    monkeypatch.setattr(gov, "V3_SITE_PACKAGES", site)
    baseline = gov._capture_complete_site_packages_inventory()
    expected_inventory = {"python_site_inventory": dict(baseline)}
    expected_fingerprint = gov.numerical_runtime_fingerprint_sha256(expected_inventory)
    dependency.write_bytes(b"side_effect = True\n")
    imported: list[str] = []
    monkeypatch.setattr(
        gov.importlib,
        "import_module",
        lambda name: imported.append(name),
    )
    with pytest.raises(gov.AuthorityError, match="before external runtime code"):
        gov._require_current_numerical_runtime_binding(
            expected_inventory,
            expected_fingerprint,
        )
    assert imported == []


def test_focused_a_suite_covers_all_six_v3_implementation_boundaries() -> None:
    expected_prefix = (
        os.fspath(gov.V3_PYTHON_EXECUTABLE),
        "-I",
        "-B",
        "-S",
    )
    assert gov.governed_python_subprocess_prefix() == expected_prefix
    assert gov._FRESH_CANARY_EXEC_COMMAND[:4] == expected_prefix
    assert gov._TEST_COMMANDS["focused_v3"][:4] == expected_prefix
    assert gov._TEST_COMMANDS["repository_full"][:4] == expected_prefix
    assert gov._TEST_COMMANDS["focused_v3"][-6:] == (
        "tests/test_metadata_calibration_v3.py",
        "tests/test_metadata_calibration_v3_synthetic.py",
        "tests/test_metadata_calibration_v3_governance.py",
        "tests/test_metadata_calibration_v3_isolation.py",
        "tests/test_metadata_calibration_v3_contract.py",
        "tests/test_metadata_calibration_v3_runner.py",
    )
    for command in gov._TEST_COMMANDS.values():
        assert "--rootdir=." in command
        assert "--import-mode=importlib" in command
        assert ("-o", "addopts=") == command[
            command.index("addopts=") - 1 : command.index("addopts=") + 1
        ]
        assert ("-o", "pythonpath=") == command[
            command.index("pythonpath=") - 1 : command.index("pythonpath=") + 1
        ]
        assert ("-p", "no:cacheprovider") == command[
            command.index("no:cacheprovider") - 1 : command.index("no:cacheprovider") + 1
        ]
    environment = gov.frozen_numerical_executor_environment()
    assert "HOME" not in environment
    assert environment["PYTEST_DISABLE_PLUGIN_AUTOLOAD"] == "1"
    assert environment["OPENBLAS_NUM_THREADS"] == "1"
    with pytest.raises(TypeError):
        environment["OPENBLAS_NUM_THREADS"] = "2"  # type: ignore[index]


def test_no_site_bootstrap_adds_only_frozen_source_and_site_paths() -> None:
    script = (
        gov._GOVERNED_PYTHON_BOOTSTRAP_PREAMBLE
        + "import pytest\n"
        + "print(int(sys.flags.no_site))\n"
        + "print(int(sys.flags.hash_randomization))\n"
        + "print('|'.join(sys.path))\n"
        + "print(pathlib.Path(pytest.__file__).resolve())\n"
    )
    completed = subprocess.run(
        (*gov.governed_python_subprocess_prefix(), "-c", script),
        check=True,
        capture_output=True,
        env=dict(gov.frozen_numerical_executor_environment()),
    )
    lines = completed.stdout.decode("utf-8").splitlines()
    assert lines == [
        "1",
        "1",
        "|".join(
            (
                *gov._ISOLATED_STDLIB_SYS_PATH,
                os.fspath(gov.V3_SOURCE_REPOSITORY / "src"),
                os.fspath(gov.V3_SITE_PACKAGES),
            )
        ),
        os.fspath(gov.V3_SITE_PACKAGES / "pytest/__init__.py"),
    ]


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
        "pairing_mean_derangement_abs_g_M_change_by_query_block": [123.45678901234567] * 5,
        "pairing_scientifically_changed_count": 5,
        "pairing_potential_unit_count": 5,
        "context_packet_covered_count": 5,
    }
    exact_row_bytes = len(gov.canonical_json_bytes(representative_row)) + 1
    conservative_full_size = exact_row_bytes * 88_992 + 16 * 1024 * 1024
    assert conservative_full_size > 64 * 1024 * 1024
    assert conservative_full_size < gov.DEVELOPMENT_RESULT_MAXIMUM_BYTES


def test_observed_runner_and_bundle_bind_real_process_and_clean_source(
    private_test_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repository = private_test_path / "tiny-repository"
    repository.mkdir()
    (repository / "test_sample.py").write_text(
        "def test_sample():\n    assert 1 + 1 == 2\n",
        encoding="utf-8",
    )
    snapshot = _commit_tiny_repository(repository, include_frozen_files=True)
    command = _test_only_pytest_command("test_sample.py")
    monkeypatch.setattr(gov, "_TEST_COMMANDS", {"focused_v3": command})
    git_exclude = repository / ".git/info/exclude"
    git_exclude.write_text(
        git_exclude.read_text(encoding="utf-8") + "\n.pytest-temp/\n.pytest_cache/\n",
        encoding="utf-8",
    )
    observed = gov._run_observed_test_under_test_root(
        "focused_v3",
        repository=repository,
        snapshot=snapshot,
    )
    assert observed.process_id != os.getpid()
    assert observed.passed_tests == observed.collected_tests == 1
    assert len(observed.junit_report_file_sha256) == 64
    assert observed.junit_report_path.startswith("/tmp/cfeg-v3-junit-")
    site_argument, basetemp_argument, junit_argument = observed.argv[-3:]
    assert site_argument.startswith(gov._PYTHON_SITE_INVENTORY_ARG_PREFIX)
    assert len(site_argument.removeprefix(gov._PYTHON_SITE_INVENTORY_ARG_PREFIX)) == 64
    assert basetemp_argument.startswith("--basetemp=/tmp/cfeg-v3-junit-")
    assert basetemp_argument.endswith("/pytest-temp")
    assert junit_argument == f"--junitxml={observed.junit_report_path}"
    assert ("-o", "addopts=") == observed.argv[
        observed.argv.index("addopts=") - 1 : observed.argv.index("addopts=") + 1
    ]
    assert "--import-mode=importlib" in observed.argv
    assert ("-p", "no:cacheprovider") == observed.argv[
        observed.argv.index("no:cacheprovider") - 1 : observed.argv.index("no:cacheprovider") + 1
    ]
    assert not (repository / ".pytest-temp").exists()
    assert not (repository / ".pytest_cache").exists()
    with pytest.raises(gov.AuthorityError, match="frozen command template"):
        gov._require_frozen_test_argv(
            "focused_v3",
            (*observed.argv[:-2], "--basetemp=/tmp/hostile", observed.argv[-1]),
            observed.junit_report_path,
            expected_python_site_inventory_sha256=site_argument.removeprefix(
                gov._PYTHON_SITE_INVENTORY_ARG_PREFIX
            ),
        )
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
    publication_root = private_test_path / "published-bundle"
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
    with pytest.raises(gov.AuthorityError, match="exact canonical path"):
        gov._validate_development_bundle_at_path(
            publication_root / "development-bundle.json",
            snapshot=snapshot,
            frozen_file_bytes=_frozen_file_bytes(),
            focused_test_run=observed,
            scope="canonical",
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


def test_committed_frozen_file_drift_cannot_consume_bundle_path(
    private_test_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repository = private_test_path / "frozen-drift-repository"
    repository.mkdir()
    (repository / "test_sample.py").write_text(
        "def test_sample():\n    assert True\n",
        encoding="utf-8",
    )
    _commit_tiny_repository(repository, include_frozen_files=True)
    drifted = repository / gov.MASTER_PLAN_PATH
    drifted.write_bytes(drifted.read_bytes() + b"\n# committed hostile drift\n")
    subprocess.run(["git", "-C", os.fspath(repository), "add", gov.MASTER_PLAN_PATH], check=True)
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
            "drift frozen plan",
        ],
        check=True,
    )
    snapshot = gov.capture_clean_source_snapshot(repository)
    command = _test_only_pytest_command("test_sample.py")
    monkeypatch.setattr(gov, "_TEST_COMMANDS", {"focused_v3": command})
    observed = gov._run_observed_test_under_test_root(
        "focused_v3",
        repository=repository,
        snapshot=snapshot,
    )
    publication_root = private_test_path / "must-remain-absent"
    with pytest.raises(gov.AuthorityError, match="frozen file is not the exact regular blob"):
        gov._publish_development_bundle_under_test_root(
            publication_root,
            snapshot=snapshot,
            frozen_file_bytes=_frozen_file_bytes(),
            created_at_UTC="2026-09-06T12:00:00Z",
            focused_test_run=observed,
        )
    assert not publication_root.exists()


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


@pytest.mark.parametrize(
    ("set_flag", "clear_flag"),
    (
        ("--assume-unchanged", "--no-assume-unchanged"),
        ("--skip-worktree", "--no-skip-worktree"),
    ),
)
def test_git_index_hints_cannot_hide_tracked_disk_drift(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    set_flag: str,
    clear_flag: str,
) -> None:
    repository = tmp_path / "hidden-drift"
    repository.mkdir()
    snapshot = _commit_tiny_repository(repository)
    relative_path = "src/cfeg/metadata_calibration_v3_governance.py"
    source_path = repository / relative_path
    original = source_path.read_bytes()
    subprocess.run(
        ["git", "-C", os.fspath(repository), "update-index", set_flag, relative_path],
        check=True,
    )
    source_path.write_bytes(b"# drift hidden from porcelain status\n")
    try:
        status = subprocess.run(
            [
                "git",
                "-C",
                os.fspath(repository),
                "status",
                "--porcelain=v1",
                "--untracked-files=all",
                "-z",
            ],
            check=True,
            capture_output=True,
        )
        assert status.stdout == b""
        with pytest.raises(gov.AuthorityError, match="special Git index state"):
            gov.capture_clean_source_snapshot(repository)
        monkeypatch.setattr(gov, "V3_SOURCE_REPOSITORY", repository)
        with pytest.raises(gov.AuthorityError, match="special Git index state"):
            gov._require_prepublication_current_source(snapshot)
    finally:
        subprocess.run(
            [
                "git",
                "-C",
                os.fspath(repository),
                "update-index",
                clear_flag,
                relative_path,
            ],
            check=True,
        )
        source_path.write_bytes(original)
    assert gov.capture_clean_source_snapshot(repository).identity == snapshot.identity


def test_sparse_checkout_state_cannot_enter_source_authority(tmp_path: Path) -> None:
    repository = tmp_path / "sparse-source"
    repository.mkdir()
    snapshot = _commit_tiny_repository(repository)
    subprocess.run(
        [
            "git",
            "-C",
            os.fspath(repository),
            "config",
            "--local",
            "core.sparseCheckout",
            "true",
        ],
        check=True,
    )
    try:
        with pytest.raises(gov.AuthorityError, match="sparse checkout"):
            gov.capture_clean_source_snapshot(repository)
    finally:
        subprocess.run(
            [
                "git",
                "-C",
                os.fspath(repository),
                "config",
                "--local",
                "--unset",
                "core.sparseCheckout",
            ],
            check=True,
        )
    assert gov.capture_clean_source_snapshot(repository).identity == snapshot.identity


def test_git_exclude_cannot_hide_untracked_tests_conftest(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repository = tmp_path / "excluded-conftest"
    repository.mkdir()
    tracked_test = repository / "tests/test_clean.py"
    tracked_test.parent.mkdir(parents=True)
    tracked_test.write_text("def test_clean():\n    assert True\n", encoding="utf-8")
    snapshot = _commit_tiny_repository(repository)
    hidden = repository / "tests/conftest.py"
    (repository / ".git/info/exclude").write_text(
        "tests/conftest.py\n",
        encoding="utf-8",
    )
    hidden.write_text("pytest_plugins = ['hostile_plugin']\n", encoding="utf-8")
    status = subprocess.run(
        [
            "git",
            "-C",
            os.fspath(repository),
            "status",
            "--porcelain=v1",
            "--untracked-files=all",
            "-z",
        ],
        check=True,
        capture_output=True,
    )
    assert status.stdout == b""
    with pytest.raises(gov.AuthorityError, match="execution tree contains untracked"):
        gov.capture_clean_source_snapshot(repository)

    spawned = False
    real_popen = subprocess.Popen

    def guarded_spawn(*args: object, **kwargs: object) -> subprocess.Popen[bytes]:
        nonlocal spawned
        argv = args[0]
        if isinstance(argv, (tuple, list)) and argv[0] == os.fspath(gov.V3_PYTHON_EXECUTABLE):
            spawned = True
            raise AssertionError("hidden conftest reached Python subprocess launch")
        return real_popen(*args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(subprocess, "Popen", guarded_spawn)
    with pytest.raises(gov.AuthorityError, match="execution tree contains untracked"):
        gov._run_observed_test_under_test_root(
            "focused_v3",
            repository=repository,
            snapshot=snapshot,
        )
    assert not spawned


@pytest.mark.parametrize(
    "hidden_relative_path",
    (
        "src/cfeg/__pycache__/poison.cpython-310.pyc",
        "src/calibration_free_eeg.egg-info/PKG-INFO",
    ),
)
def test_ignored_generated_import_tree_is_rejected_before_execution(
    tmp_path: Path,
    hidden_relative_path: str,
) -> None:
    repository = tmp_path / "ignored-import-tree"
    repository.mkdir()
    _commit_tiny_repository(repository)
    (repository / ".git/info/exclude").write_text(
        f"/{hidden_relative_path}\n",
        encoding="utf-8",
    )
    hidden = repository / hidden_relative_path
    hidden.parent.mkdir(parents=True)
    hidden.write_bytes(b"untracked import metadata")
    status = subprocess.run(
        [
            "git",
            "-C",
            os.fspath(repository),
            "status",
            "--porcelain=v1",
            "--untracked-files=all",
            "-z",
        ],
        check=True,
        capture_output=True,
    )
    assert status.stdout == b""
    with pytest.raises(gov.AuthorityError, match="execution tree contains untracked"):
        gov.capture_clean_source_snapshot(repository)


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
    monkeypatch.setattr(
        gov,
        "_require_active_governed_process",
        lambda **_kwargs: object(),
    )
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
    monkeypatch.setattr(
        gov,
        "_require_active_governed_process",
        lambda **_kwargs: object(),
    )
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
        lambda *_args, **_kwargs: (_ for _ in ()).throw(gov.AuthorityError("A drift")),
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
    command = _test_only_pytest_command("test_mutation.py")
    monkeypatch.setattr(gov, "_TEST_COMMANDS", {"focused_v3": command})
    with pytest.raises(gov.ValidationError, match="already-clean worktree"):
        gov._run_observed_test_under_test_root(
            "focused_v3",
            repository=repository,
            snapshot=snapshot,
        )


def test_observed_runner_rejects_python_command_without_no_site(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repository = tmp_path / "missing-no-site"
    repository.mkdir()
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
                "raise SystemExit(0)",
            )
        },
    )
    with pytest.raises(gov.AuthorityError, match="exact python -I -B -S -c"):
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
        {"focused_v3": _test_only_pytest_command("test_skip.py")},
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
        gov._GOVERNED_PYTHON_BOOTSTRAP_PREAMBLE
        + "import os\n"
        + "path=sys.argv[-1].split('=',1)[1]\n"
        + f"os.symlink({os.fspath(victim)!r},path)\n"
    )
    monkeypatch.setattr(
        gov,
        "_TEST_COMMANDS",
        {
            "focused_v3": (
                *gov.governed_python_subprocess_prefix(),
                "-c",
                script,
            )
        },
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
    monkeypatch.setattr(
        gov,
        "_require_active_governed_process",
        lambda **_kwargs: object(),
    )
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
    monkeypatch.setattr(
        gov,
        "_require_active_governed_process",
        lambda **_kwargs: object(),
    )
    core = ModuleType("cfeg.analysis.metadata_calibration_v3_synthetic")
    monkeypatch.setitem(
        sys.modules,
        "cfeg.analysis.metadata_calibration_v3_synthetic",
        core,
    )
    result = dict(gov.seal_payload({"schema": gov.ARTIFACT_SCHEMAS["development_result"]}))
    context_payload = dict(gov.seal_payload({"schema": gov.ARTIFACT_SCHEMAS["context_reference"]}))
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

    recovered_artifact, recovered_audit = gov.recover_canonical_development_result_at_bundle_source(
        development_bundle=bundle,  # type: ignore[arg-type]
        context_reference=context,  # type: ignore[arg-type]
    )
    assert recovered_artifact is expected
    assert recovered_audit is audited_proof
    assert observed == ["A-canonical-audit"]


def test_a_selection_recovery_requires_the_sealed_core_audit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        gov,
        "_require_active_governed_process",
        lambda **_kwargs: object(),
    )
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
                "schema": ("cfeg.metadata-calibration-efficiency-v3.complete-grid-gate-report.v1"),
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
                "synthetic_plan_file_sha256": gov.FROZEN_FILE_SHA256[gov.SYNTHETIC_PLAN_PATH],
                "development_bundle_schema": gov.DEVELOPMENT_BUNDLE_SCHEMA,
                "development_bundle_payload_sha256": (audited.development_bundle_payload_sha256),
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
    with pytest.raises(
        gov.AuthorityError,
        match="exact -I -B -S bootstrap|fresh audit child command line differs",
    ):
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
    assert (
        gov.canary_probe_digest(
            seed,
            prevalidation_trace=tuple(gov.CanaryState)[:4],
        )
        == gov.CANARY_PROBE_DIGEST
    )
    with pytest.raises(gov.AuthorityError, match="ScientificSeedCapability"):
        gov.scientific_seed_sequence(seed)  # type: ignore[arg-type]
    with pytest.raises(gov.AuthorityError, match="ScientificBeaconCapability"):
        gov.derive_scientific_seed(beacon=seed)  # type: ignore[arg-type]


def test_seed_statement_is_target_bound_noncircular_and_rejects_endpoint_drift() -> None:
    statement = gov.build_seed_statement(
        scientific_candidate_id=f"{gov.CANDIDATE_ID}-p3-nu_4-lambda_0p20",
        selected_method_freeze_payload_sha256="a" * 64,
        exact_target_timestamp_UTC="2030-01-01T00:10:00Z",
        exact_HTTPS_endpoint=("https://beacon.nist.gov/beacon/2.0/pulse/time/1893456600000"),
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
    exponent, modulus = gov._validate_owner_public_key_bytes(public_bytes)
    assert exponent == 65_537
    assert modulus.bit_length() == 4096
    assert (
        gov.file_sha256(public_bytes) == gov.FROZEN_FILE_SHA256[gov.OWNER_AUTHORITY_PUBLIC_KEY_PATH]
    )
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
