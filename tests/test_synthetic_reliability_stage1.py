from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

import cfeg.analysis.synthetic_reliability_stage1 as stage1
from cfeg.analysis.synthetic_reliability_stage1 import (
    _build_intervention_overrides,
    _metadata_safety,
    _run_metadata_only_probe,
    _subject_contrasts,
    _validate_best_metric_artifact,
    _validate_intervention_predictions,
)
from cfeg.data.collate import build_vocabularies, collate_eeg, overwrite_external_metadata
from cfeg.data.datasets import EEGProcessedDataset
from cfeg.data.schema import load_manifest
from cfeg.data.ssvep_synthetic import generate_quality_synthetic_processed
from cfeg.governance import current_source_revision_contract
from cfeg.reliability_contract import RELIABILITY_ROLES
from cfeg.utils.config import load_config

REPO = Path(__file__).resolve().parents[1]


def _asset(tmp_path):
    root = tmp_path / "synthetic_quality_v1"
    generate_quality_synthetic_processed(
        root,
        n_subjects=4,
        n_blocks_per_interface=2,
        n_repetitions_per_class_per_block=2,
        n_classes=4,
        duration_sec=0.5,
        seed=42,
    )
    return root


def test_stage1_checkpoint_selection_must_match_validation_history(tmp_path) -> None:
    metrics = tmp_path / "metrics_val.csv"
    metrics.write_text(
        "epoch,val_accuracy\n1,0.25\n2,0.75\n3,0.75\n",
        encoding="utf-8",
    )

    _validate_best_metric_artifact(
        metrics,
        checkpoint_epoch=2,
        checkpoint_metric=0.75,
    )
    with pytest.raises(ValueError, match="does not match"):
        _validate_best_metric_artifact(
            metrics,
            checkpoint_epoch=1,
            checkpoint_metric=0.25,
        )


def test_stage1_block_interventions_are_complete_label_free_and_coherent(tmp_path) -> None:
    manifest = load_manifest(_asset(tmp_path))
    selected = manifest.loc[manifest["subject_id"].eq("sub003")].reset_index(drop=True)

    overrides, mapping = _build_intervention_overrides(selected)

    assert set(overrides) == {"missing", "stale_block", "wrong_interface"}
    assert all(len(values) == len(selected) for values in overrides.values())
    assert len(mapping) == 3 * len(selected)
    assert not mapping["label_used_for_pairing"].any()
    assert not mapping.loc[
        mapping["scenario"].isin(["stale_block", "wrong_interface"]),
        "mapping_fixed_point",
    ].any()
    stale = mapping.loc[mapping["scenario"].eq("stale_block")]
    assert stale["subject_id"].eq(stale["donor_subject_id"]).all()
    assert stale["target_electrode_type"].eq(stale["donor_electrode_type"]).all()
    assert stale["target_run_id"].ne(stale["donor_run_id"]).all()
    wrong = mapping.loc[mapping["scenario"].eq("wrong_interface")]
    assert wrong["subject_id"].eq(wrong["donor_subject_id"]).all()
    assert wrong["target_run_id"].eq(wrong["donor_run_id"]).all()
    assert wrong["target_electrode_type"].ne(wrong["donor_electrode_type"]).all()
    assert mapping.loc[
        mapping["scenario"].isin(["stale_block", "wrong_interface"]),
        "bundle_changed",
    ].all()

    relabeled = selected.copy()
    relabeled["label"] = np.roll(relabeled["label"].to_numpy(), 1)
    _, relabeled_mapping = _build_intervention_overrides(relabeled)
    pd.testing.assert_frame_equal(mapping, relabeled_mapping)


def test_stage1_missing_override_changes_only_external_m_bundle(tmp_path) -> None:
    root = _asset(tmp_path)
    manifest = load_manifest(root)
    selected = manifest.loc[manifest["subject_id"].eq("sub003")].reset_index(drop=True)
    overrides, _mapping = _build_intervention_overrides(selected)
    dataset = EEGProcessedDataset(
        [root],
        expected_revisions={"synthetic_quality": "synthetic_quality_v1"},
        expected_protocol={
            "metadata_contract_version": "0.4-dev",
            "query_qc_extractor_version": (
                "filtered_cropped_pre_zscore_channel_std_median_v1"
            ),
            "external_continuous_schema": "impedance_mean_max_v1",
        },
    )
    sample_indices = [
        index
        for index, entry in enumerate(dataset.entries)
        if entry[2]["subject_id"] == "sub003"
    ][:2]
    samples = [dataset[index] for index in sample_indices]
    vocab = build_vocabularies(entry[2] for entry in dataset.entries)
    batch = collate_eeg(
        samples,
        vocabularies=vocab,
        metadata_contract_version="0.4-dev",
        external_metadata_mode="observed",
    )
    original = copy.deepcopy(batch["cond"])
    changed = overwrite_external_metadata(
        batch["cond"],
        batch["sample_id"],
        overrides["missing"],
        vocab,
    )

    for key in (
        "channel_ids",
        "channel_mask",
        "continuous",
        "continuous_missing",
        "sfreq_processed_float",
        "query_qc",
        "query_qc_missing",
        "channel_query_qc",
        "channel_query_qc_missing",
    ):
        torch.testing.assert_close(changed[key], original[key], rtol=0.0, atol=0.0)
    for key in (
        "metadata_contract_version",
        "external_metadata_mode",
        "query_qc_extractor_version",
        "external_continuous_schema",
        "development_control",
    ):
        assert changed[key] == original[key]
    assert changed["external_continuous"].eq(0).all()
    assert changed["external_continuous_missing"].all()
    assert changed["channel_impedance"].eq(0).all()
    assert changed["channel_impedance_missing"].all()
    assert changed["reference"].eq(0).all()
    assert changed["electrode_type"].eq(0).all()
    assert changed["cap_type"].eq(0).all()


def test_stage1_metadata_only_probe_is_block_invariant_and_at_chance(tmp_path) -> None:
    manifest = load_manifest(_asset(tmp_path))
    train_indices = np.flatnonzero(
        manifest["subject_id"].isin(["sub000", "sub001", "sub002"]).to_numpy()
    )
    val_indices = np.flatnonzero(manifest["subject_id"].eq("sub003").to_numpy())

    result = _run_metadata_only_probe(
        manifest,
        train_indices=train_indices,
        val_indices=val_indices,
        n_classes=4,
        active_channel_ids=[48, 55, 57, 56, 53, 61, 62, 63],
        ridge_alphas=(1e-6, 0.0001, 0.01, 1.0, 10.0, 100.0),
        invariance_tolerance=1e-12,
        chance_tolerance=1e-12,
    )

    assert result["model"]["predictor_allowlist_only"] is True
    assert result["model"]["block_logit_invariance_verified"] is True
    assert result["subject_metrics"]["balanced_accuracy"].tolist() == [0.25]
    assert result["predictions"].groupby("acquisition_block_id")[
        "predicted_label"
    ].nunique().eq(1).all()


def test_stage1_subject_contrasts_and_intervention_guard() -> None:
    metrics = pd.DataFrame(
        [
            {"role": role, "subject_id": subject, "balanced_accuracy": value}
            for subject, values in {
                "sub000": [0.2, 0.3, 0.4, 0.6],
                "sub001": [0.3, 0.4, 0.5, 0.55],
            }.items()
            for role, value in zip(("A0", "A_M", "A_Q", "A_QM"), values)
        ]
    )
    contrasts = _subject_contrasts(metrics)
    np.testing.assert_allclose(contrasts["delta_metadata_given_q"], [0.2, 0.05])
    np.testing.assert_allclose(contrasts["delta_metadata_without_q"], [0.1, 0.1])
    np.testing.assert_allclose(contrasts["interaction"], [0.1, -0.05])

    intervention_metrics = pd.DataFrame(
        [
            {
                "scenario": scenario,
                "subject_id": subject,
                "balanced_accuracy": value,
            }
            for subject, values in {
                "sub000": [0.6, 0.45, 0.4, 0.35],
                "sub001": [0.55, 0.5, 0.45, 0.4],
            }.items()
            for scenario, value in zip(
                ("correct", "missing", "stale_block", "wrong_interface"), values
            )
        ]
    )
    safety = _metadata_safety(
        intervention_metrics,
        metrics.loc[metrics["role"].eq("A_Q"), ["subject_id", "balanced_accuracy"]],
    )
    np.testing.assert_allclose(safety["missing_minus_A_Q"], [0.05, 0.0])
    np.testing.assert_allclose(safety["wrong_interface_minus_A_Q"], [-0.05, -0.1])

    prediction_rows = []
    for scenario in ("correct", "missing", "stale_block", "wrong_interface"):
        prediction_rows.append(
            {
                "scenario": scenario,
                "sample_id": "sample",
                "label": 0,
                "checkpoint_sha256": "a" * 64,
                "channel_metadata_available_count": 0.0 if scenario == "missing" else 8.0,
                "metadata_spatial_residual_frobenius_norm": (
                    0.0 if scenario == "missing" else 0.01
                ),
                "query_spatial_operator_frobenius_norm": 0.1,
                "channel_query_available_count": 8.0,
                "metadata_residual_alpha": 0.2,
                "channel_query_noise_logit_sha256": "b" * 64,
                "query_noise_logit_0": 0.3,
            }
        )
    mapping = pd.DataFrame(
        [
            {
                "scenario": scenario,
                "target_sample_id": "sample",
                "mapping_fixed_point": False,
                "bundle_changed": True,
                "label_used_for_pairing": False,
                "target_electrode_type": "wet",
                "subject_id": "sub000",
                "donor_subject_id": "sub000",
                "donor_electrode_type": "wet" if scenario == "stale_block" else "dry",
                "target_run_id": "block00",
                "donor_run_id": "block01" if scenario == "stale_block" else "block00",
            }
            for scenario in ("missing", "stale_block", "wrong_interface")
        ]
    )
    _validate_intervention_predictions(
        pd.DataFrame(prediction_rows), mapping, tolerance=1e-6
    )
    corrupted = pd.DataFrame(prediction_rows)
    corrupted.loc[corrupted["scenario"].eq("wrong_interface"), "query_noise_logit_0"] = 0.4
    try:
        _validate_intervention_predictions(corrupted, mapping, tolerance=1e-6)
    except ValueError as exc:
        assert "independent Q branch" in str(exc)
    else:  # pragma: no cover - the guard must reject the injected corruption
        raise AssertionError("Q-branch corruption was not rejected")


def test_stage1_analysis_orchestration_writes_verified_atomic_receipt(
    tmp_path, monkeypatch
) -> None:
    root = tmp_path / "stage1"
    root.mkdir()
    plan_path = REPO / "configs/analysis/reliability_spatial_v1.yaml"
    plan = load_config(plan_path, strict_env=False)
    plan_sha256 = hashlib.sha256(plan_path.read_bytes()).hexdigest()
    stage0_sha256 = "0" * 64
    source = current_source_revision_contract()
    manifest = pd.DataFrame(
        [
            {
                "sample_id": f"{subject}_sample{label}",
                "subject_id": subject,
                "electrode_type": "wet" if label % 2 == 0 else "dry",
                "acquisition_block_id": f"{subject}_block{label % 2}",
                "label": label,
            }
            for subject in ("sub000", "sub001")
            for label in range(4)
        ]
    )
    shared_runtime = {
        "parameter_schema_sha256": "1" * 64,
        "initial_trainable_state_sha256": "2" * 64,
        "split_assignment_sha256": "3" * 64,
        "vocabulary_sha256": "4" * 64,
        "asset_provenance_sha256": "5" * 64,
        "environment_sha256": "6" * 64,
        "execution_phase": "synthetic_engineering",
        "analysis_plan_status": None,
        "analysis_plan_sha256": None,
        "cohort_role": "simulation_only",
        "cohort_roles_sha256": None,
        "cohort_sha256": "7" * 64,
        "cohort_subject_count": 8,
        "cohort_sample_count": 256,
        "outer_test_access_during_training": False,
        "loader_settings_sha256": "8" * 64,
        "development_control_sha256": "9" * 64,
        **source,
    }
    runtime_rows = []
    contexts = {}
    for role in RELIABILITY_ROLES:
        role_root = root / role
        role_root.mkdir()
        (role_root / "best.pt").write_bytes(f"checkpoint-{role}".encode())
        (role_root / "split.csv").write_text(
            "sample_id,split\nplaceholder,val\n", encoding="utf-8"
        )
        (role_root / "metrics_val.csv").write_text(
            "epoch,val_accuracy\n1,1.0\n", encoding="utf-8"
        )
        attempts_root = role_root / "runtime_attempts"
        attempts_root.mkdir()
        attempt_path = attempts_root / "attempt_000.json"
        attempt_payload = {
            "schema": "cfeg.runtime-metrics.v1",
            "status": "completed",
            "run_mode": "training_and_validation",
            "requested_device": "cuda",
            "execution_device_type": "cuda",
            "resolved_device": "cuda:0",
            "cuda_oom": False,
            "elapsed_time_sec": 1.0,
        }
        attempt_path.write_text(json.dumps(attempt_payload), encoding="utf-8")
        runtime_path = role_root / "runtime_metrics.json"
        runtime_path.write_text(
            json.dumps(
                {
                    "schema": "cfeg.runtime-metrics.v1",
                    "status": "completed",
                    "run_mode": "training_and_validation",
                    "requested_device": "cuda",
                    "execution_device_type": "cuda",
                    "resolved_device": "cuda:0",
                    "cuda_oom": False,
                    "attempt_count": 1,
                    "runtime_attempts": [
                        {
                            "path": "runtime_attempts/attempt_000.json",
                            "sha256": hashlib.sha256(
                                attempt_path.read_bytes()
                            ).hexdigest(),
                            "status": "completed",
                            "elapsed_time_sec": 1.0,
                        }
                    ],
                }
            ),
            encoding="utf-8",
        )
        protocol = {
            "candidate_id": "reliability-spatial-v1",
            "candidate_plan_sha256": plan_sha256,
            "predecessor_retirement_receipt_sha256": "a" * 64,
            "reliability_query_feature_schema": "query_window_reliability_features_v1",
            "reliability_mechanism_family_sha256": "b" * 64,
            "reliability_family_role": role,
            "reliability_stage0_receipt_sha256": stage0_sha256,
        }
        row = {
            "variant": role,
            "status": "completed",
            "reliability_family_runtime_status": "verified_equal",
            "reliability_stage0_receipt_sha256": stage0_sha256,
            "best_validation_accuracy": 1.0,
            "runtime_metrics_path": str(runtime_path),
            "runtime_metrics_sha256": hashlib.sha256(runtime_path.read_bytes()).hexdigest(),
            "runtime_metrics_schema": "cfeg.runtime-metrics.v1",
            "resolved_device": "cuda:0",
            **shared_runtime,
            **protocol,
        }
        runtime_rows.append(row)
        contexts[role] = {
            "checkpoint": {
                "checkpoint_role": "source_validation_best",
                "selection_split": "val",
                "selection_metric": "accuracy",
                "best_metric": 1.0,
                "epoch": 1,
            },
            "train_config": {
                "protocol": protocol,
                "runtime_contract": shared_runtime,
                "model": {"n_classes": 4},
            },
            "manifest": manifest,
        }

    def fake_context(_eval_cfg, checkpoint_path, access_route):
        assert access_route == "reliability_stage1_analysis"
        return contexts[Path(checkpoint_path).parent.name]

    def fake_predictions(
        _context,
        _indices,
        *,
        role,
        scenario,
        checkpoint_sha256,
        best_epoch,
        external_overrides=None,
    ):
        del external_overrides
        rows = []
        for item in manifest.itertuples(index=False):
            logits = [-4.0] * 4
            logits[item.label] = 4.0
            rows.append(
                {
                    "role": role,
                    "scenario": scenario,
                    "sample_id": item.sample_id,
                    "subject_id": item.subject_id,
                    "electrode_type": item.electrode_type,
                    "acquisition_block_id": item.acquisition_block_id,
                    "label": item.label,
                    "predicted_label": item.label,
                    "confidence": 0.99,
                    "best_epoch": best_epoch,
                    "checkpoint_sha256": checkpoint_sha256,
                    **{f"logit_{index}": value for index, value in enumerate(logits)},
                    "query_spatial_operator_frobenius_norm": 0.1,
                    "channel_query_available_count": 8.0,
                    "metadata_residual_alpha": 0.2,
                    "channel_metadata_available_count": (
                        0.0 if scenario == "missing" else 8.0
                    ),
                    "metadata_spatial_residual_frobenius_norm": (
                        0.0 if scenario == "missing" else 0.01
                    ),
                    "query_noise_logit_0": 0.3,
                }
            )
        return pd.DataFrame(rows)

    mapping_rows = []
    for item in manifest.itertuples(index=False):
        for scenario in ("missing", "stale_block", "wrong_interface"):
            stale = scenario == "stale_block"
            mapping_rows.append(
                {
                    "scenario": scenario,
                    "target_sample_id": item.sample_id,
                    "subject_id": item.subject_id,
                    "donor_subject_id": item.subject_id,
                    "target_electrode_type": item.electrode_type,
                    "donor_electrode_type": (
                        item.electrode_type
                        if stale
                        else ("dry" if item.electrode_type == "wet" else "wet")
                    ),
                    "target_run_id": "block00",
                    "donor_run_id": "block01" if stale else "block00",
                    "mapping_fixed_point": False,
                    "bundle_changed": True,
                    "label_used_for_pairing": False,
                }
            )
    mapping = pd.DataFrame(mapping_rows)
    overrides = {
        scenario: {sample_id: {} for sample_id in manifest["sample_id"]}
        for scenario in ("missing", "stale_block", "wrong_interface")
    }
    probe_predictions = manifest[
        [
            "sample_id",
            "subject_id",
            "electrode_type",
            "acquisition_block_id",
            "label",
        ]
    ].copy()
    probe_predictions["predicted_label"] = 0
    for index in range(4):
        probe_predictions[f"logit_{index}"] = 0.0

    monkeypatch.setattr(stage1, "load_evaluation_context", fake_context)
    monkeypatch.setattr(
        stage1,
        "validate_reliability_training_preflight",
        lambda _cfg, dry_run: stage0_sha256,
    )
    monkeypatch.setattr(
        stage1,
        "validate_reliability_stage0_gate",
        lambda _cfg, require_receipt: stage0_sha256,
    )
    monkeypatch.setattr(
        stage1,
        "_checkpoint_split_indices",
        lambda _context, split: np.arange(len(manifest)),
    )
    monkeypatch.setattr(stage1, "_collect_prediction_frame", fake_predictions)
    monkeypatch.setattr(
        stage1,
        "_build_intervention_overrides",
        lambda _manifest: (overrides, mapping),
    )
    monkeypatch.setattr(
        stage1,
        "_run_metadata_only_probe",
        lambda *_args, **_kwargs: {
            "predictions": probe_predictions,
            "subject_metrics": pd.DataFrame(
                {
                    "subject_id": ["sub000", "sub001"],
                    "balanced_accuracy": [0.25, 0.25],
                }
            ),
            "model": {
                "schema": "cfeg.synthetic-reliability-metadata-only-ridge.v1",
                "predictor_allowlist_only": True,
                "block_logit_invariance_verified": True,
            },
        },
    )

    receipt_path = stage1.finalize_synthetic_reliability_stage1(
        root,
        runtime_rows=runtime_rows,
        candidate_plan=plan,
        candidate_plan_path=plan_path,
        stage0_receipt_sha256=stage0_sha256,
    )

    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    assert receipt["status"] == "completed_integrity_verified_descriptive_only"
    assert receipt["checkpoint_fingerprints"]["A_Q"]["path"] == "A_Q/best.pt"
    assert receipt["runtime_artifacts"]["A_Q"]["path"] == "A_Q/runtime_metrics.json"
    assert receipt["runtime_attempt_artifacts"]["A_Q"]["path"] == (
        "A_Q/runtime_attempts/attempt_000.json"
    )
    assert receipt["training_artifacts"]["A_Q"]["split"]["path"] == (
        "A_Q/split.csv"
    )
    assert receipt["training_artifacts"]["A_Q"]["validation_metrics"]["path"] == (
        "A_Q/metrics_val.csv"
    )
    assert receipt["integrity"]["a_q_reference_comparison_present"] is True
    assert {path.name for path in receipt_path.parent.iterdir()} == set(stage1.OUTPUT_FILES)

    stage1.validate_synthetic_reliability_stage1_publication_inputs(root)
    split_path = root / "A_Q/split.csv"
    split_bytes = split_path.read_bytes()
    split_path.write_bytes(split_bytes + b"\n")
    with pytest.raises(ValueError, match="bound artifact changed"):
        stage1.validate_synthetic_reliability_stage1_publication_inputs(root)
    split_path.write_bytes(split_bytes)
    attempt_path = root / "A_Q/runtime_attempts/attempt_000.json"
    attempt_path.write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="bound artifact changed"):
        stage1.validate_synthetic_reliability_stage1_publication_inputs(root)
