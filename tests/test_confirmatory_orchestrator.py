from __future__ import annotations

import json
from pathlib import Path

import pytest


def _orchestrator(monkeypatch):
    repo = Path(__file__).resolve().parents[1]
    monkeypatch.syspath_prepend(str(repo / "scripts"))
    import run_confirmatory_primary

    return run_confirmatory_primary


def _job(final_path: Path, index: int = 0) -> dict:
    return {
        "job_id": f"job-{index}",
        "prediction_csv": str(final_path),
        "checkpoint_path": str(final_path.parent.parent / "runs" / f"job-{index}" / "final.pt"),
        "output_dir": str(final_path.parent.parent / "runs" / f"job-{index}"),
        "resolved_config_sha256": str(index) * 64,
        "role": "A0_eeg_only",
        "seed": 42 + index,
        "fold_index": 0,
    }


def test_existing_final_grid_is_verified_before_success(tmp_path, monkeypatch) -> None:
    module = _orchestrator(monkeypatch)
    final_dir = tmp_path / "predictions"
    final_dir.mkdir()
    jobs = [_job(final_dir / f"job-{index}.csv", index) for index in range(2)]
    for job in jobs:
        path = Path(job["prediction_csv"])
        path.write_text("sample_id\na\n", encoding="utf-8")
        path.with_name(f"{path.stem}_provenance.json").write_text("{}", encoding="utf-8")
    observed = []
    monkeypatch.setattr(
        module,
        "_validate_prediction_bundle",
        lambda path, **kwargs: observed.append((path, kwargs["job"]["job_id"])),
    )

    module._predict_lockbox_grid_atomically(
        manifest_path=tmp_path / "execution_manifest.json",
        manifest={
            "jobs": jobs,
            "execution_contract_sha256": "e" * 64,
        },
        processed_dir=str(tmp_path / "processed"),
    )

    assert [job_id for _, job_id in observed] == ["job-0", "job-1"]


def test_staging_resume_rejects_unregistered_sidecar_path(tmp_path, monkeypatch) -> None:
    module = _orchestrator(monkeypatch)
    final_path = tmp_path / "predictions" / "job.csv"
    manifest = {"jobs": [_job(final_path)], "execution_contract_sha256": "e" * 64}
    staging = tmp_path / f".predictions.{'e' * 64}.staging"
    staging.mkdir()
    (staging / "job.csv").write_text("sample_id\na\n", encoding="utf-8")
    (staging / "job_provenance.json").write_text(
        json.dumps({"prediction_csv": str(tmp_path / "third-location.csv")}),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="unexpected CSV path"):
        module._predict_lockbox_grid_atomically(
            manifest_path=tmp_path / "execution_manifest.json",
            manifest=manifest,
            processed_dir=str(tmp_path / "processed"),
        )
    assert not final_path.parent.exists()


def test_valid_staging_resume_is_reverified_before_and_after_publish(
    tmp_path,
    monkeypatch,
) -> None:
    module = _orchestrator(monkeypatch)
    final_path = tmp_path / "predictions" / "job.csv"
    manifest = {"jobs": [_job(final_path)], "execution_contract_sha256": "e" * 64}
    staging = tmp_path / f".predictions.{'e' * 64}.staging"
    staging.mkdir()
    staged_path = staging / "job.csv"
    staged_path.write_text("sample_id\na\n", encoding="utf-8")
    (staging / "job_provenance.json").write_text(
        json.dumps({"prediction_csv": str(staged_path.resolve())}),
        encoding="utf-8",
    )
    observed = []
    monkeypatch.setattr(
        module,
        "_validate_prediction_bundle",
        lambda path, **kwargs: observed.append(
            (Path(path).resolve(), Path(kwargs["expected_recorded_path"]).resolve())
        ),
    )

    module._predict_lockbox_grid_atomically(
        manifest_path=tmp_path / "execution_manifest.json",
        manifest=manifest,
        processed_dir=str(tmp_path / "processed"),
    )

    assert final_path.is_file()
    assert not staging.exists()
    sidecar = json.loads(
        final_path.with_name("job_provenance.json").read_text(encoding="utf-8")
    )
    assert sidecar["prediction_csv"] == str(final_path.resolve())
    assert observed == [
        (staged_path.resolve(), staged_path.resolve()),
        (staged_path.resolve(), final_path.resolve()),
        (final_path.resolve(), final_path.resolve()),
    ]


def test_pair_wide_fairness_is_checked_before_lockbox_reveal(tmp_path, monkeypatch) -> None:
    module = _orchestrator(monkeypatch)
    fairness = "f" * 64
    runtime = {
        "parameter_schema_sha256": "1" * 64,
        "initial_trainable_state_sha256": "2" * 64,
        "split_assignment_sha256": "3" * 64,
        "vocabulary_sha256": "4" * 64,
        "asset_provenance_sha256": "5" * 64,
        "execution_phase": "confirmatory_training",
        "analysis_plan_sha256": "6" * 64,
        "cohort_roles_sha256": "7" * 64,
        "cohort_sha256": "8" * 64,
        "outer_test_access_during_training": False,
        "source_commit_sha": "a" * 40,
        "source_dirty": False,
        "source_tree_sha256": "9" * 64,
        "environment_sha256": "b" * 64,
        "development_control": {"name": "none"},
        "development_control_sha256": "c" * 64,
        "loader_settings_sha256": "d" * 64,
    }
    records = []
    jobs = []
    for role, mode, marker in (
        ("A0_eeg_only", "null", b"a0"),
        ("A2_structured_condition_prompt", "observed", b"a2"),
    ):
        root = tmp_path / role
        root.mkdir()
        (root / "split.csv").write_text("sample_id,split\na,test\n", encoding="utf-8")
        (root / "final.pt").write_bytes(marker)
        job = {
            "job_id": role,
            "output_dir": str(root),
            "role": role,
            "seed": 42,
            "fold_index": 0,
        }
        checkpoint = {
            "config": {
                "model": {"condition_encoder": {"external_metadata_mode": mode}},
                "protocol": {"primary_fairness_hash": fairness},
                "runtime_contract": dict(runtime),
            }
        }
        jobs.append(job)
        records.append({"job": job, "checkpoint": checkpoint})
    manifest = {"jobs": jobs, "primary_fairness_hash": fairness}

    module._validate_pair_wide_training_fairness(records, manifest)
    records[1]["checkpoint"]["config"]["runtime_contract"][
        "initial_trainable_state_sha256"
    ] = "e" * 64
    with pytest.raises(ValueError, match="fairness differs before reveal"):
        module._validate_pair_wide_training_fairness(records, manifest)
