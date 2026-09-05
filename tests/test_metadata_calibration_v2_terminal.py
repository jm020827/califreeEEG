from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

from cfeg import metadata_calibration_v2_terminal as terminal
from cfeg.analysis import metadata_calibration_v2 as analysis
from cfeg.analysis import metadata_calibration_v2_synthetic as synthetic
from cfeg.data import metadata_calibration_v2_requests as requests
from cfeg.metadata_calibration_v2_terminal import (
    V2_TERMINAL_DENY_SHA256,
    deny_v2_terminal_operational_action,
    validate_v2_terminal_deny_overlay,
)

REPOSITORY = Path(__file__).resolve().parents[1]


def test_terminal_deny_overlay_is_exact_and_source_pinned() -> None:
    overlay = validate_v2_terminal_deny_overlay()
    assert len(V2_TERMINAL_DENY_SHA256) == 64
    assert overlay["status"] == "consumed_inconclusive_infrastructure_error"
    assert overlay["scientific_result_created"] is False
    assert overlay["automatic_retry"] == "forbidden"
    assert "wearable_held60_v2_access_or_outcome" in overlay["blocked_operational_actions"]


@pytest.mark.parametrize(
    "action",
    (
        "beta_dong_independent_gate",
        "beta_dong_outcome_reduction_or_selection",
        "beta_dong_query_label_join",
        "beta_dong_score_request_or_prediction",
        "choi2019_v2_replication_outcome",
        "synthetic_lockbox_reexecution",
        "wearable_held60_v2_access_or_outcome",
    ),
)
def test_terminal_overlay_denies_every_declared_operational_action(action: str) -> None:
    with pytest.raises(PermissionError, match="terminally denied"):
        deny_v2_terminal_operational_action(action)


def test_unknown_terminal_capability_fails_closed() -> None:
    with pytest.raises(PermissionError, match="Unknown"):
        deny_v2_terminal_operational_action("invented-bypass")


@pytest.mark.parametrize(
    "call",
    (
        lambda: analysis.score_v2_external_request({}, contract=None),
        lambda: analysis.reduce_v2_participant_balanced_accuracy(
            None, contract=None, cohort="development"
        ),
        lambda: analysis.derive_v2_participant_eauc_deltas(
            None, contract=None, cohort="development"
        ),
        lambda: analysis.select_v2_development_candidate(None, contract=None),
        lambda: analysis.evaluate_v2_independent_aq_gate(
            None, development_selection_receipt={}, contract=None
        ),
        lambda: requests.build_external_score_request(
            score_cache=None,
            candidate=None,
            budget=0,
            contract=None,
            gate_enabled=False,
        ),
        lambda: requests.evaluate_query_labels_after_predictions(
            [], asset=None, participant=None, token_secret=b""
        ),
        lambda: synthetic.run_lockbox_to_path(
            None, authorization_path="poison", output_path="poison"
        ),
    ),
)
def test_library_effectful_entrypoints_deny_before_parsing_poison_inputs(call) -> None:
    with pytest.raises(PermissionError, match="terminally denied"):
        call()


def test_missing_or_tampered_terminal_overlay_fails_closed(tmp_path: Path, monkeypatch) -> None:
    missing = tmp_path / "missing.json"
    monkeypatch.setattr(terminal, "V2_TERMINAL_DENY_PATH", missing)
    with pytest.raises(PermissionError, match="unavailable"):
        terminal.validate_v2_terminal_deny_overlay()

    tampered = tmp_path / "tampered.json"
    tampered.write_text("{}\n", encoding="utf-8")
    monkeypatch.setattr(terminal, "V2_TERMINAL_DENY_PATH", tampered)
    with pytest.raises(PermissionError, match="byte hash drifted"):
        terminal.validate_v2_terminal_deny_overlay()

    symlink = tmp_path / "overlay-link.json"
    symlink.symlink_to(tampered)
    monkeypatch.setattr(terminal, "V2_TERMINAL_DENY_PATH", symlink)
    with pytest.raises(PermissionError, match="nonsymlink"):
        terminal.validate_v2_terminal_deny_overlay()


def test_analysis_cli_denies_scoring_before_opening_request_or_output(tmp_path: Path) -> None:
    completed = subprocess.run(
        [
            sys.executable,
            str(REPOSITORY / "scripts/run_metadata_calibration_v2.py"),
            "score",
            "--request",
            str(tmp_path / "does-not-exist.json"),
            "--output",
            str(tmp_path / "must-not-exist.json"),
        ],
        cwd=REPOSITORY,
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode != 0
    assert "terminally denied" in completed.stderr
    assert not (tmp_path / "must-not-exist.json").exists()


def test_external_cli_denies_request_before_opening_assets_or_secret(tmp_path: Path) -> None:
    completed = subprocess.run(
        [
            sys.executable,
            str(REPOSITORY / "scripts/run_metadata_calibration_v2_external.py"),
            "request",
            "--dataset",
            "beta_v1",
            "--phase",
            "development",
            "--processed-dir",
            str(tmp_path / "does-not-exist"),
            "--subject-id",
            "sub001",
            "--candidate-key",
            "irrelevant",
            "--budget",
            "0",
            "--token-secret-file",
            str(tmp_path / "does-not-exist.secret"),
            "--cache-dir",
            str(tmp_path / "cache"),
            "--output",
            str(tmp_path / "must-not-exist.json"),
        ],
        cwd=REPOSITORY,
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode != 0
    assert "terminally denied" in completed.stderr
    assert not (tmp_path / "must-not-exist.json").exists()


def test_synthetic_cli_denies_lockbox_before_plan_or_artifact_access(tmp_path: Path) -> None:
    completed = subprocess.run(
        [
            sys.executable,
            str(REPOSITORY / "scripts/run_metadata_calibration_v2_synthetic.py"),
            "--plan",
            str(tmp_path / "does-not-exist.yaml"),
            "lockbox",
        ],
        cwd=REPOSITORY,
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode != 0
    assert "terminally denied" in completed.stderr
