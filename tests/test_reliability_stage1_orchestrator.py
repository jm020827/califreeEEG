from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))

import run_ablation
from run_ablation import (
    _noncanonical_reliability_attempts,
    _publish_reliability_stage1_root,
    _release_reliability_stage1_execution,
    _reserve_reliability_stage1_execution,
    _rewrite_reliability_success_paths,
    _write_failed_reliability_attempt,
)


def test_reliability_stage1_root_is_relocated_and_atomically_published(
    tmp_path, monkeypatch
) -> None:
    staging = tmp_path / ".stage1.staging-test"
    canonical = tmp_path / "stage1"
    staging.mkdir()
    for role in ("A0", "A_M", "A_Q", "A_QM"):
        role_root = staging / role
        role_root.mkdir()
        (role_root / "best.pt").write_bytes(b"checkpoint")
        (role_root / "runtime_metrics.json").write_text("{}", encoding="utf-8")
    analysis = staging / "analysis"
    analysis.mkdir()
    (analysis / "stage1_receipt.json").write_text("{}", encoding="utf-8")
    (staging / "summary.csv").write_text("variant,status\nA0,completed\n", encoding="utf-8")
    rows = [
        {
            "output_dir": str(staging / "A0"),
            "runtime_metrics_path": str(staging / "A0/runtime_metrics.json"),
            "stage1_analysis_receipt": str(analysis / "stage1_receipt.json"),
        }
    ]

    _rewrite_reliability_success_paths(
        rows,
        staging_root=staging,
        canonical_root=canonical,
    )
    assert rows[0]["output_dir"] == str(canonical / "A0")
    assert rows[0]["runtime_metrics_path"] == str(
        canonical / "A0/runtime_metrics.json"
    )
    assert rows[0]["stage1_analysis_receipt"] == str(
        canonical / "analysis/stage1_receipt.json"
    )

    validated = []
    monkeypatch.setattr(
        run_ablation,
        "validate_synthetic_reliability_stage1_publication_inputs",
        lambda root: validated.append(Path(root)),
    )
    monkeypatch.setattr(
        run_ablation,
        "_validate_reliability_stage1_summary",
        lambda *_args, **_kwargs: None,
    )
    _publish_reliability_stage1_root(staging, canonical_root=canonical)

    assert validated == [staging]
    assert not staging.exists()
    assert (canonical / "summary.csv").is_file()
    assert (canonical / "analysis/stage1_receipt.json").is_file()


def test_failed_reliability_attempt_stays_noncanonical_with_receipt(tmp_path) -> None:
    staging = tmp_path / ".stage1.staging-failed"
    canonical = tmp_path / "stage1"
    staging.mkdir()
    (staging / "summary.csv").write_text(
        "variant,status,error\nA0,failed,injected\n", encoding="utf-8"
    )

    receipt_path = _write_failed_reliability_attempt(
        staging,
        canonical_root=canonical,
        rows=[
            {
                "variant": "A0",
                "status": "failed",
                "error_type": "RuntimeError",
                "error": "injected",
            }
        ],
        attempt_contract={
            "candidate_id": "reliability-spatial-v1",
            "candidate_plan_sha256": "a" * 64,
            "stage0_receipt_sha256": "b" * 64,
        },
    )

    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    assert receipt["status"] == "failed_noncanonical_attempt"
    assert receipt["canonical_published"] is False
    assert receipt["retry_authorized"] is False
    assert receipt["roles_completed"] == []
    assert receipt["claims"]["synthetic_integration_claim_allowed"] is False
    assert not canonical.exists()
    assert _noncanonical_reliability_attempts(canonical) == [staging]


def test_reliability_stage1_execution_reservation_is_exclusive(tmp_path) -> None:
    canonical = tmp_path / "stage1"
    first = _reserve_reliability_stage1_execution(
        canonical,
        contract={"candidate_id": "reliability-spatial-v1"},
    )

    with pytest.raises(run_ablation.GovernanceError, match="already reserved"):
        _reserve_reliability_stage1_execution(
            canonical,
            contract={"candidate_id": "reliability-spatial-v1"},
        )

    _release_reliability_stage1_execution(first)
    assert not first.exists()

    canonical.mkdir()
    with pytest.raises(run_ablation.GovernanceError, match="canonical output appeared"):
        _reserve_reliability_stage1_execution(
            canonical,
            contract={"candidate_id": "reliability-spatial-v1"},
        )
    assert not run_ablation._reliability_stage1_reservation_path(canonical).exists()


def test_reliability_dry_run_cannot_create_canonical_root(
    tmp_path, monkeypatch
) -> None:
    canonical = tmp_path / "stage1"
    monkeypatch.setattr(run_ablation, "CANONICAL_RELIABILITY_STAGE1_ROOT", canonical)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_ablation.py",
            "--config",
            str(run_ablation.CANONICAL_RELIABILITY_SUITE),
            "--output-root",
            str(canonical),
            "--dry-run",
        ],
    )

    with pytest.raises(run_ablation.GovernanceError, match="dry-run cannot target"):
        run_ablation.main()

    assert not canonical.exists()


@pytest.mark.parametrize("unsafe_kind", ["descendant", "reservation_namespace"])
def test_reliability_dry_run_is_disjoint_from_stage1_namespaces(
    tmp_path, monkeypatch, unsafe_kind
) -> None:
    canonical = tmp_path / "stage1"
    monkeypatch.setattr(run_ablation, "CANONICAL_RELIABILITY_STAGE1_ROOT", canonical)
    unsafe = (
        canonical / "dryrun"
        if unsafe_kind == "descendant"
        else canonical.parent / ".stage1.execution-reservation"
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_ablation.py",
            "--config",
            str(run_ablation.CANONICAL_RELIABILITY_SUITE),
            "--output-root",
            str(unsafe),
            "--dry-run",
        ],
    )

    with pytest.raises(run_ablation.GovernanceError, match="dry-run"):
        run_ablation.main()

    assert not unsafe.exists()
    assert not canonical.exists()


def _mock_reliability_main(
    monkeypatch,
    *,
    canonical: Path,
    finalizer_failure: bool,
) -> list[Path]:
    monkeypatch.setattr(run_ablation, "CANONICAL_RELIABILITY_STAGE1_ROOT", canonical)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_ablation.py",
            "--config",
            str(run_ablation.CANONICAL_RELIABILITY_SUITE),
            "--output-root",
            str(canonical),
        ],
    )
    monkeypatch.setattr(
        run_ablation,
        "validate_primary_pair_and_hash",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        run_ablation,
        "validate_reliability_family",
        lambda *_args, **_kwargs: "f" * 64,
    )
    monkeypatch.setattr(
        run_ablation,
        "validate_reliability_stage0_gate",
        lambda *_args, **_kwargs: "0" * 64,
    )
    observed_roles: list[str] = []

    def fake_training(
        cfg,
        *,
        dry_run,
        resume_exact,
        _artifact_output_dir,
        _reliability_orchestrator_authorization,
    ):
        assert dry_run is False
        assert resume_exact is False
        assert (
            _reliability_orchestrator_authorization
            is run_ablation._RELIABILITY_ORCHESTRATOR_AUTHORIZATION
        )
        role = str(cfg["run_name"])
        observed_roles.append(role)
        role_root = Path(_artifact_output_dir)
        role_root.mkdir(parents=True)
        runtime = role_root / "runtime_metrics.json"
        runtime.write_text("{}", encoding="utf-8")
        return {
            "output_dir": str(role_root),
            "best_validation_accuracy": 0.25,
            "params": {"total": 10, "trainable": 10},
            "runtime_contract": {},
            "runtime_metrics": {
                "schema": "cfeg.runtime-metrics.v1",
                "resolved_device": "cuda:0",
                "cuda_oom": False,
                "elapsed_time_sec": 1.0,
            },
            "runtime_metrics_path": str(runtime),
            "runtime_metrics_sha256": hashlib.sha256(runtime.read_bytes()).hexdigest(),
            "elapsed_time_sec": 1.0,
            "total_elapsed_time_sec": 1.0,
        }

    monkeypatch.setattr(run_ablation, "run_training", fake_training)

    def fake_runtime_family(rows, _family):
        assert [row["variant"] for row in rows] == ["A0", "A_M", "A_Q", "A_QM"]
        assert all(row["status"] == "completed" for row in rows)
        return "verified_equal"

    monkeypatch.setattr(
        run_ablation,
        "validate_reliability_runtime_family",
        fake_runtime_family,
    )

    def fake_finalizer(root, **_kwargs):
        if finalizer_failure:
            raise RuntimeError("injected finalizer failure")
        analysis = Path(root) / "analysis"
        analysis.mkdir()
        receipt = analysis / "stage1_receipt.json"
        receipt.write_text("{}", encoding="utf-8")
        return receipt

    monkeypatch.setattr(
        run_ablation,
        "finalize_synthetic_reliability_stage1",
        fake_finalizer,
    )
    validated: list[Path] = []
    monkeypatch.setattr(
        run_ablation,
        "validate_synthetic_reliability_stage1_publication_inputs",
        lambda root: validated.append(Path(root)),
    )
    return validated


def test_reliability_main_publishes_complete_suite_atomically(
    tmp_path, monkeypatch
) -> None:
    canonical = tmp_path / "stage1"
    validated = _mock_reliability_main(
        monkeypatch,
        canonical=canonical,
        finalizer_failure=False,
    )

    run_ablation.main()

    assert len(validated) == 1
    assert not validated[0].exists()
    assert not run_ablation._reliability_stage1_reservation_path(canonical).exists()
    assert {path.name for path in canonical.iterdir()} == {
        "A0",
        "A_M",
        "A_Q",
        "A_QM",
        "analysis",
        "summary.csv",
    }
    summary = canonical.joinpath("summary.csv").read_text(encoding="utf-8")
    assert str(canonical / "A_QM") in summary
    assert ".stage1.staging-" not in summary


def test_reliability_main_finalizer_failure_stays_noncanonical(
    tmp_path, monkeypatch
) -> None:
    canonical = tmp_path / "stage1"
    validated = _mock_reliability_main(
        monkeypatch,
        canonical=canonical,
        finalizer_failure=True,
    )

    with pytest.raises(SystemExit) as exc_info:
        run_ablation.main()

    assert exc_info.value.code == 1
    assert validated == []
    assert not canonical.exists()
    attempts = _noncanonical_reliability_attempts(canonical)
    assert len(attempts) == 1
    assert run_ablation._reliability_stage1_reservation_path(canonical).is_dir()
    receipt = json.loads(
        (attempts[0] / "stage1_attempt_receipt.json").read_text(encoding="utf-8")
    )
    assert receipt["canonical_published"] is False
    assert receipt["retry_requires_new_owner_decision"] is True


def test_reliability_main_publication_guard_failure_gets_attempt_receipt(
    tmp_path, monkeypatch
) -> None:
    canonical = tmp_path / "stage1"
    _mock_reliability_main(
        monkeypatch,
        canonical=canonical,
        finalizer_failure=False,
    )

    def reject_publication(_root):
        raise ValueError("injected final input drift")

    monkeypatch.setattr(
        run_ablation,
        "validate_synthetic_reliability_stage1_publication_inputs",
        reject_publication,
    )
    with pytest.raises(SystemExit) as exc_info:
        run_ablation.main()

    assert exc_info.value.code == 1
    assert not canonical.exists()
    attempts = _noncanonical_reliability_attempts(canonical)
    assert len(attempts) == 1
    assert run_ablation._reliability_stage1_reservation_path(canonical).is_dir()
    receipt = json.loads(
        (attempts[0] / "stage1_attempt_receipt.json").read_text(encoding="utf-8")
    )
    assert receipt["failures"] == [
        {
            "variant": "__stage1_publication_guard__",
            "error_type": "ValueError",
            "error": "injected final input drift",
        }
    ]


def test_reliability_main_path_rewrite_failure_gets_attempt_receipt(
    tmp_path, monkeypatch
) -> None:
    canonical = tmp_path / "stage1"
    _mock_reliability_main(
        monkeypatch,
        canonical=canonical,
        finalizer_failure=False,
    )
    monkeypatch.setattr(
        run_ablation,
        "_rewrite_reliability_success_paths",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            ValueError("injected escaped success path")
        ),
    )

    with pytest.raises(SystemExit) as exc_info:
        run_ablation.main()

    assert exc_info.value.code == 1
    assert not canonical.exists()
    attempts = _noncanonical_reliability_attempts(canonical)
    assert len(attempts) == 1
    receipt = json.loads(
        (attempts[0] / "stage1_attempt_receipt.json").read_text(encoding="utf-8")
    )
    assert receipt["failures"] == [
        {
            "variant": "__stage1_summary_guard__",
            "error_type": "ValueError",
            "error": "injected escaped success path",
        }
    ]
