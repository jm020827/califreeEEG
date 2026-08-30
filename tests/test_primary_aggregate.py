from __future__ import annotations

import hashlib
from pathlib import Path

import pandas as pd
import pytest

from cfeg.analysis.primary_aggregate import aggregate_primary_confirmatory
from cfeg.analysis.provenance import (
    PRIMARY_ANALYSIS_MANIFEST_COLUMNS,
    PRIMARY_ANALYSIS_MANIFEST_SCHEMA,
    VerifiedPredictionBundle,
    primary_analysis_manifest_sha256,
)


def _sha(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _sample_hash(sample_ids) -> str:
    digest = hashlib.sha256()
    for sample_id in sorted(str(value) for value in sample_ids):
        encoded = sample_id.encode("utf-8")
        digest.update(len(encoded).to_bytes(8, "big"))
        digest.update(encoded)
    return digest.hexdigest()


def _manifest() -> pd.DataFrame:
    rows = []
    for subject in range(1, 5):
        for condition in ("dry", "wet"):
            for label in (0, 1):
                rows.append(
                    {
                        "sample_id": f"sub{subject:03d}_{condition}_{label}",
                        "dataset_id": "wearable",
                        "subject_id": f"sub{subject:03d}",
                        "electrode_type": condition,
                        "label": label,
                    }
                )
    manifest = pd.DataFrame(rows)
    for column in PRIMARY_ANALYSIS_MANIFEST_COLUMNS:
        if column not in manifest:
            manifest[column] = 0
    return manifest


def _frame(manifest: pd.DataFrame, *, seed: int, fold: int, role: str) -> pd.DataFrame:
    subjects = {0: {"sub001", "sub002"}, 1: {"sub003", "sub004"}}[fold]
    selected = manifest.loc[manifest["subject_id"].isin(subjects)].copy()
    mode = "null" if role == "A0_eeg_only" else "observed"
    prediction = 0 if role == "A0_eeg_only" else selected["label"].astype(int)
    frame = pd.DataFrame(
        {
            "sample_id": selected["sample_id"].astype(str),
            "label": selected["label"].astype(int),
            "prediction": prediction,
        }
    )
    constants = {
        "prediction_schema_version": "cfeg.predictions.v3",
        "scenario": "clean",
        "selection_split": "test",
        "sample_identity_sha256": _sample_hash(frame["sample_id"]),
        "checkpoint_sha256": _sha(f"checkpoint-{role}-{seed}-{fold}"),
        "execution_job_id": f"seed{seed}-fold{fold}-{role}",
        "execution_contract_sha256": _sha("execution-contract"),
        "execution_manifest_content_sha256": _sha("execution-manifest"),
        "lockbox_reveal_receipt_sha256": _sha("lockbox-reveal"),
        "planned_config_sha256": _sha(f"planned-{role}-{seed}-{fold}"),
        "training_completion_sha256": _sha(f"completion-{role}-{seed}-{fold}"),
        "primary_ablation_role": role,
        "primary_fairness_hash": _sha("fairness"),
        "metadata_contract_version": "0.4-dev",
        "conditioning_architecture": "physical_hybrid_v1",
        "external_metadata_mode": mode,
        "optimization_seed": seed,
        "split_seed": 42,
        "n_folds": 2,
        "fold_index": fold,
        "parameter_schema_sha256": _sha("schema"),
        "initial_trainable_state_sha256": _sha(f"initial-{seed}"),
        "split_assignment_sha256": _sha(f"assignment-{fold}"),
        "vocabulary_sha256": _sha("vocabulary"),
        "asset_provenance_sha256": _sha("asset"),
        "execution_phase": "confirmatory_training",
        "analysis_plan_sha256": _sha("analysis-plan"),
        "cohort_roles_sha256": _sha("cohort-roles"),
        "cohort_sha256": _sha("cohort"),
        "outer_test_access_during_training": False,
        "source_commit_sha": "a" * 40,
        "source_dirty": False,
        "source_tree_sha256": _sha("source-tree"),
        "environment_sha256": _sha("environment"),
        "inference_environment_sha256": _sha("inference-environment"),
        "checkpoint_role": "confirmatory_fixed_epoch",
        "checkpoint_selection_split": "none",
        "checkpoint_selection_metric": "fixed_epoch",
        "analysis_manifest_schema": PRIMARY_ANALYSIS_MANIFEST_SCHEMA,
        "analysis_manifest_sha256": primary_analysis_manifest_sha256(manifest, frame["sample_id"]),
        "split_manifest_sha256": _sha(f"split-file-{fold}"),
    }
    for name, value in constants.items():
        frame[name] = value
    return frame


def _bundle(frame: pd.DataFrame, name: str) -> VerifiedPredictionBundle:
    return VerifiedPredictionBundle(
        frame=frame,
        artifact={"prediction_csv_sha256": _sha(f"csv-{name}")},
        path=Path(f"{name}.csv"),
        provenance_path=Path(f"{name}_provenance.json"),
        provenance_sha256=_sha(f"provenance-{name}"),
    )


def _pairs(manifest: pd.DataFrame):
    return [
        (
            _bundle(
                _frame(manifest, seed=seed, fold=fold, role="A0_eeg_only"),
                f"a0-{seed}-{fold}",
            ),
            _bundle(
                _frame(
                    manifest,
                    seed=seed,
                    fold=fold,
                    role="A2_structured_condition_prompt",
                ),
                f"a2-{seed}-{fold}",
            ),
        )
        for seed in (1, 2, 3)
        for fold in (0, 1)
    ]


def _execution_manifest() -> dict:
    jobs = []
    for seed in (1, 2, 3):
        for fold in (0, 1):
            for role in ("A0_eeg_only", "A2_structured_condition_prompt"):
                jobs.append(
                    {
                        "job_id": f"seed{seed}-fold{fold}-{role}",
                        "seed": seed,
                        "fold_index": fold,
                        "role": role,
                        "resolved_config_sha256": _sha(f"planned-{role}-{seed}-{fold}"),
                    }
                )
    return {
        "schema": "cfeg.primary-execution-manifest.v1",
        "plan_status": "frozen",
        "execution_allowed": True,
        "execution_contract_sha256": _sha("execution-contract"),
        "manifest_content_sha256": _sha("execution-manifest"),
        "lockbox_reveal_receipt_sha256": _sha("lockbox-reveal"),
        "jobs": jobs,
    }


def test_primary_aggregate_averages_seeds_before_subject_inference() -> None:
    manifest = _manifest()
    runs, subject_seed, subjects, condition_subjects, summary = aggregate_primary_confirmatory(
        _pairs(manifest),
        manifest,
        expected_seeds=[1, 2, 3],
        expected_n_folds=2,
        expected_n_samples=16,
        expected_n_subjects=4,
        expected_n_labels=2,
        primary_excluded_subject_ids=[],
        primary_included_subject_ids=sorted(manifest["subject_id"].unique()),
        execution_manifest=_execution_manifest(),
        success_threshold=0.75,
        sesoi_balanced_accuracy=0.01,
        inference_alpha=0.05,
        inference_alternative="greater",
        n_resamples=100,
    )

    assert len(runs) == 6
    assert len(subject_seed) == 12
    assert len(subjects) == 4
    assert len(condition_subjects) == 8
    assert subjects["n_seeds"].eq(3).all()
    assert subjects["baseline_balanced_accuracy"].eq(0.5).all()
    assert subjects["candidate_balanced_accuracy"].eq(1.0).all()
    assert summary["n_subjects"] == 4
    assert summary["learned_condition_cells"] == 8
    assert summary["inference_resamples"] == 100
    assert summary["null_margin_balanced_accuracy"] == 0.01
    # Four independent subjects cannot attain one-sided p <= .05 under the
    # exact sign-flip test (minimum p is 1/16), regardless of the point estimate.
    assert summary["primary_minimum_effect_claim_supported"] is False


def test_primary_aggregate_rejects_missing_seed_fold_pair() -> None:
    manifest = _manifest()

    with pytest.raises(ValueError, match="seed×fold grid is incomplete"):
        aggregate_primary_confirmatory(
            _pairs(manifest)[:-1],
            manifest,
            expected_seeds=[1, 2, 3],
            expected_n_folds=2,
            expected_n_samples=16,
            expected_n_subjects=4,
            expected_n_labels=2,
            primary_excluded_subject_ids=[],
            primary_included_subject_ids=sorted(manifest["subject_id"].unique()),
            execution_manifest=_execution_manifest(),
            n_resamples=10,
        )
