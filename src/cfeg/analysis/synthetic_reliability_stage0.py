from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import pandas as pd
import torch
import yaml
from scipy.stats import rankdata

from cfeg.data.datasets import EEGProcessedDataset
from cfeg.data.schema import load_manifest, validate_manifest
from cfeg.data.ssvep_synthetic import DEFAULT_FREQS
from cfeg.data.synthetic_quality import (
    SYNTHETIC_QUALITY_ASSET_FILES,
    load_synthetic_quality_truth,
)
from cfeg.governance import (
    current_source_revision_contract,
    implementation_contract_sha256,
)
from cfeg.models.reliability_conditioning import extract_query_reliability_features

STAGE0_PLAN_SCHEMA = "cfeg.synthetic-reliability-stage0-plan.v1"
STAGE0_RECEIPT_SCHEMA = "cfeg.synthetic-reliability-stage0-receipt.v1"

Q_FEATURES = (
    "q_pre_z_channel_std",
    "q_pre_z_channel_std_available",
    "q_log_post_z_variance",
    "q_log_first_difference_energy",
    "q_mean_abs_cross_channel_correlation",
    "q_max_abs_cross_channel_correlation",
    "q_spectral_concentration",
)
M_CHANNEL_FEATURES = (
    "m_channel_impedance",
    "m_channel_impedance_centered",
    "m_channel_impedance_abs_centered",
    "m_channel_impedance_available",
)
M_GLOBAL_FEATURES = (
    "m_impedance_mean",
    "m_impedance_max",
    "m_impedance_mean_available",
    "m_impedance_max_available",
    "m_electrode_wet",
    "m_electrode_dry",
)
M_FEATURES = (*M_CHANNEL_FEATURES, *M_GLOBAL_FEATURES)
PREDICTOR_FORBIDDEN = {
    "label",
    "stimulus_frequency_hz",
    "stimulus_phase_rad",
    "subject_id",
    "session_id",
    "run_id",
    "trial_id",
    "sample_id",
    "acquisition_block_id",
    "canonical_channel_id",
}
OUTPUT_FILES = (
    "predictions.csv",
    "subject_metrics.csv",
    "contrasts.csv",
    "fold_models.json",
    "receipt.json",
)
OUTPUT_ARTIFACT_FILES = OUTPUT_FILES[:-1]


def run_stage0(
    processed_dir: str | Path,
    plan: dict[str, Any],
    *,
    plan_path: str | Path | None = None,
) -> dict[str, Any]:
    """Run a simulation-only, participant-disjoint reliability identifiability assay."""

    _validate_plan(plan)
    root = Path(processed_dir)
    bound_source = current_source_revision_contract()
    bound_implementation_sha256 = implementation_contract_sha256()
    bound_assets = _asset_fingerprints(root)
    bound_plan_file_sha256 = (
        _sha256_file(Path(plan_path)) if plan_path is not None else None
    )
    frame, integrity = _build_feature_table(root, plan)
    subjects = sorted(frame["subject_id"].unique())
    if len(subjects) < 3:
        raise ValueError("Stage-0 nested LOSO requires at least three subjects.")

    target = str(plan["primary_target"])
    alphas = tuple(float(value) for value in plan["evaluation"]["ridge_alphas"])
    prediction_parts: list[pd.DataFrame] = []
    model_contracts: dict[str, Any] = {}
    split_rows: list[dict[str, Any]] = []
    for held_subject in subjects:
        train = frame.loc[frame["subject_id"] != held_subject].reset_index(drop=True)
        test = frame.loc[frame["subject_id"] == held_subject].reset_index(drop=True)
        if set(train["subject_id"]) & set(test["subject_id"]):
            raise RuntimeError("Outer Stage-0 subject split overlaps.")
        inner_subjects = sorted(train["subject_id"].unique())
        split_rows.append(
            {
                "outer_subject": held_subject,
                "outer_train_subjects": inner_subjects,
                "outer_test_subjects": [held_subject],
                "outer_train_row_identity_sha256": _row_identity_sha256(train),
                "outer_test_row_identity_sha256": _row_identity_sha256(test),
                "inner_folds": [
                    {
                        "inner_test_subject": inner_subject,
                        "inner_train_subjects": [
                            subject
                            for subject in inner_subjects
                            if subject != inner_subject
                        ],
                    }
                    for inner_subject in inner_subjects
                ],
            }
        )
        output = test[
            [
                "sample_id",
                "subject_id",
                "acquisition_block_id",
                "electrode_type",
                "canonical_channel_id",
                target,
            ]
        ].copy()
        fold_models: dict[str, Any] = {}
        fitted: dict[str, dict[str, Any]] = {}
        for arm, features in {
            "Q": Q_FEATURES,
            "M": M_FEATURES,
            "QM": (*Q_FEATURES, *M_FEATURES),
        }.items():
            alpha, inner_scores = _select_alpha_nested_loso(
                train,
                features=features,
                target=target,
                alphas=alphas,
            )
            model = _fit_ridge(
                train.loc[:, list(features)].to_numpy(float),
                train[target].to_numpy(float),
                train["subject_id"].to_numpy(str),
                alpha=alpha,
                features=features,
            )
            output[f"prediction_{arm}"] = _predict_ridge(
                model, test.loc[:, list(features)].to_numpy(float)
            )
            fitted[arm] = model
            fold_models[arm] = {
                **_serializable_model(model),
                "inner_subject_macro_mse_by_alpha": inner_scores,
            }

        output["prediction_null"] = float(train[target].mean())
        shuffled_m, block_changed = _block_bundle_shuffle(test)
        permuted_m, channel_changed = _channel_metadata_permutation(test)
        output["prediction_QM_block_shuffle"] = _predict_ridge(
            fitted["QM"],
            np.concatenate(
                [test.loc[:, list(Q_FEATURES)].to_numpy(float), shuffled_m], axis=1
            ),
        )
        output["prediction_QM_channel_permutation"] = _predict_ridge(
            fitted["QM"],
            np.concatenate(
                [test.loc[:, list(Q_FEATURES)].to_numpy(float), permuted_m], axis=1
            ),
        )
        fold_models["control_changed_fraction"] = {
            "block_bundle_shuffle": block_changed,
            "channel_impedance_permutation": channel_changed,
        }
        model_contracts[held_subject] = fold_models
        prediction_parts.append(output)

    predictions = pd.concat(prediction_parts, ignore_index=True)
    subject_metrics = _subject_metrics(predictions, target=target)
    leakage_audit = _label_leakage_audit(
        frame,
        target=target,
        contract=plan["leakage_audit"],
    )
    integrity.update(leakage_audit)
    integrity["all_checks_passed"] = bool(
        integrity["label_only_subject_macro_r2_passed"]
        and integrity["class_target_mean_range_passed"]
    )
    contrasts = _aggregate_contrasts(
        subject_metrics,
        plan["gate"],
        integrity=integrity,
    )
    split_sha256 = _sha256_json(split_rows)
    feature_contract = {
        "schema": str(plan["feature_schema"]),
        "Q": list(Q_FEATURES),
        "M": list(M_FEATURES),
        "QM": [*Q_FEATURES, *M_FEATURES],
        "forbidden_predictors": sorted(PREDICTOR_FORBIDDEN),
    }
    receipt = {
        "schema": STAGE0_RECEIPT_SCHEMA,
        "candidate_id": plan["candidate_id"],
        "status": contrasts.iloc[0]["status"],
        "scope": "synthetic_engineering_only",
        "primary_target": target,
        "row_unit": plan["row_unit"],
        "n_subjects": len(subjects),
        "n_rows": len(frame),
        "subject_ids": subjects,
        "plan_sha256": _sha256_json(plan),
        "plan_file_sha256": bound_plan_file_sha256,
        "feature_contract": feature_contract,
        "feature_contract_sha256": _sha256_json(feature_contract),
        "split_assignment_sha256": split_sha256,
        "asset_fingerprints": bound_assets,
        "generator_contract": plan["generator"],
        "manifest_semantic_sha256": integrity["manifest_semantic_sha256"],
        "integrity": integrity,
        "gate": contrasts.iloc[0].to_dict(),
        "simulation_only": True,
        "population_inference_allowed": False,
        "confirmatory_execution_authorized": False,
        "wearable_s1_s3_reuse_allowed": False,
        "decoder_checkpoint_reuse_allowed": False,
        "interpretation": (
            "Generator and data-path identifiability check only; this is not human EEG "
            "evidence and does not estimate population benefit."
        ),
        "implementation_contract_sha256": bound_implementation_sha256,
        **bound_source,
    }
    publish_guard = {
        "processed_dir": str(root.resolve()),
        "asset_fingerprints": bound_assets,
        "plan_path": str(Path(plan_path).resolve()) if plan_path is not None else None,
        "plan_file_sha256": bound_plan_file_sha256,
        "implementation_contract_sha256": bound_implementation_sha256,
        **bound_source,
    }
    _validate_stage0_publish_guard(publish_guard)
    return {
        "predictions": predictions,
        "subject_metrics": subject_metrics,
        "contrasts": contrasts,
        "fold_models": model_contracts,
        "receipt": receipt,
        "publish_guard": publish_guard,
    }


def write_stage0_outputs(
    result: dict[str, Any],
    output_dir: str | Path,
    *,
    _execution_reservation: Path | None = None,
) -> Path:
    """Publish the terminal Stage-0 outcome exactly once into a fresh root."""

    output = Path(output_dir)
    _validate_stage0_publish_guard(result.get("publish_guard"))
    _validate_result_finite(result)
    _reject_symlink_components(output)
    if output.exists() or output.is_symlink():
        raise FileExistsError(
            f"Stage-0 output root already exists; the one-shot outcome cannot be "
            f"overwritten or superseded: {output}."
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    reservation = (
        _reserve_stage0_execution(output)
        if _execution_reservation is None
        else _execution_reservation
    )
    expected_reservation = _stage0_reservation_path(output)
    if (
        reservation != expected_reservation
        or not reservation.is_dir()
        or reservation.is_symlink()
        or list(reservation.iterdir()) != []
    ):
        raise ValueError("Stage-0 execution reservation is missing or unsafe.")
    staging = Path(
        tempfile.mkdtemp(prefix=f".{output.name}.staging-", dir=output.parent)
    )
    artifact_paths = {
        "predictions.csv": staging / "predictions.csv",
        "subject_metrics.csv": staging / "subject_metrics.csv",
        "contrasts.csv": staging / "contrasts.csv",
        "fold_models.json": staging / "fold_models.json",
    }
    try:
        _write_csv_fsynced(result["predictions"], artifact_paths["predictions.csv"])
        _write_csv_fsynced(result["subject_metrics"], artifact_paths["subject_metrics.csv"])
        _write_csv_fsynced(result["contrasts"], artifact_paths["contrasts.csv"])
        _write_json_fsynced(artifact_paths["fold_models.json"], result["fold_models"])
        receipt = dict(result["receipt"])
        receipt["artifacts"] = {
            name: {"sha256": _sha256_file(path), "size_bytes": path.stat().st_size}
            for name, path in artifact_paths.items()
        }
        receipt_path = staging / "receipt.json"
        _write_json_fsynced(receipt_path, receipt)
        validate_stage0_output_tree(result, staging)
        _validate_stage0_publish_guard(result.get("publish_guard"))
        _fsync_directory(staging)

        if output.exists() or output.is_symlink():
            raise FileExistsError(
                "Stage-0 output root appeared before atomic publication."
            )
        os.replace(staging, output)
        _fsync_directory(output.parent)
        _release_stage0_execution(reservation)
        return output / "receipt.json"
    finally:
        if staging.exists():
            shutil.rmtree(staging)


def _stage0_reservation_path(output: Path) -> Path:
    return output.parent / f".{output.name}.execution-reservation"


def _reserve_stage0_execution(output: Path) -> Path:
    if output.exists() or output.is_symlink():
        raise FileExistsError(
            f"Stage-0 output root already exists; refusing another assay: {output}."
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    reservation = _stage0_reservation_path(output)
    try:
        reservation.mkdir(mode=0o700)
    except FileExistsError as exc:
        raise FileExistsError(
            "Stage-0 is already reserved by another or prior execution; owner review "
            "is required before any retry."
        ) from exc
    if output.exists() or output.is_symlink():
        reservation.rmdir()
        _fsync_directory(reservation.parent)
        raise FileExistsError(
            "Stage-0 output appeared before reservation acquisition; refusing another assay."
        )
    _fsync_directory(reservation.parent)
    return reservation


def _release_stage0_execution(reservation: Path) -> None:
    if (
        not reservation.is_dir()
        or reservation.is_symlink()
        or list(reservation.iterdir()) != []
    ):
        raise ValueError("Stage-0 execution reservation is unsafe to release.")
    reservation.rmdir()
    _fsync_directory(reservation.parent)


def _validate_stage0_publish_guard(guard: Any) -> None:
    if not isinstance(guard, dict):
        raise TypeError("Stage-0 result lacks its publication guard.")
    root = Path(str(guard.get("processed_dir", "")))
    if (
        not root.is_dir()
        or root.is_symlink()
        or _asset_fingerprints(root) != guard.get("asset_fingerprints")
    ):
        raise ValueError("Stage-0 input asset changed before publication.")
    plan_value = guard.get("plan_path")
    if plan_value is not None:
        plan_path = Path(str(plan_value))
        if (
            not plan_path.is_file()
            or plan_path.is_symlink()
            or _sha256_file(plan_path) != guard.get("plan_file_sha256")
        ):
            raise ValueError("Stage-0 plan changed before publication.")
    if implementation_contract_sha256() != guard.get(
        "implementation_contract_sha256"
    ):
        raise ValueError("Stage-0 implementation changed before publication.")
    current_source = current_source_revision_contract()
    if any(
        current_source[field] != guard.get(field)
        for field in ("source_commit_sha", "source_dirty", "source_tree_sha256")
    ):
        raise ValueError("Stage-0 source changed before publication.")


def validate_stage0_output_tree(
    result: dict[str, Any], output_dir: str | Path
) -> dict[str, Any]:
    """Recompute and compare every substantive Stage-0 output artifact."""

    output = Path(output_dir)
    observed_entries = {path.name for path in output.iterdir()}
    if observed_entries != set(OUTPUT_FILES) or any(
        path.is_symlink() or not path.is_file() for path in output.iterdir()
    ):
        raise ValueError("Stage-0 output tree has a missing or unexpected file.")
    loaded_frames = {
        "predictions": pd.read_csv(output / "predictions.csv"),
        "subject_metrics": pd.read_csv(output / "subject_metrics.csv"),
        "contrasts": pd.read_csv(output / "contrasts.csv"),
    }
    for name, observed in loaded_frames.items():
        expected = result[name].reset_index(drop=True)
        try:
            pd.testing.assert_frame_equal(
                observed.reset_index(drop=True),
                expected,
                check_dtype=False,
                check_exact=False,
                rtol=1e-12,
                atol=1e-12,
            )
        except AssertionError as exc:
            raise ValueError(f"Stage-0 {name} does not match recomputation.") from exc
    loaded_models = json.loads((output / "fold_models.json").read_text(encoding="utf-8"))
    if loaded_models != _json_roundtrip(result["fold_models"]):
        raise ValueError("Stage-0 fold models do not match recomputation.")
    receipt = json.loads((output / "receipt.json").read_text(encoding="utf-8"))
    expected_receipt = result["receipt"]
    if set(receipt) != {*expected_receipt, "artifacts"}:
        raise ValueError("Stage-0 receipt has a missing or unexpected top-level field.")
    artifacts = receipt.get("artifacts")
    if not isinstance(artifacts, dict) or set(artifacts) != set(OUTPUT_ARTIFACT_FILES):
        raise ValueError("Stage-0 receipt artifact inventory is incomplete or unexpected.")
    for name, contract in artifacts.items():
        path = output / name
        if (
            not isinstance(contract, dict)
            or path.is_symlink()
            or contract.get("sha256") != _sha256_file(path)
            or contract.get("size_bytes") != path.stat().st_size
        ):
            raise ValueError("Stage-0 receipt artifact fingerprint is stale.")
    for field in expected_receipt:
        if receipt.get(field) != _json_roundtrip(expected_receipt.get(field)):
            raise ValueError(f"Stage-0 receipt field {field!r} differs from recomputation.")
    return receipt


def _build_feature_table(
    root: Path, plan: dict[str, Any]
) -> tuple[pd.DataFrame, dict[str, Any]]:
    _reject_symlink_components(root)
    generator = plan["generator"]
    required_files = tuple(generator["required_asset_files"])
    observed_entries = {path.name for path in root.iterdir()}
    if observed_entries != set(required_files):
        raise ValueError(
            "Stage-0 synthetic asset inventory differs from the frozen plan: "
            f"expected={sorted(required_files)}, observed={sorted(observed_entries)}."
        )
    if any(not (root / name).is_file() or (root / name).is_symlink() for name in required_files):
        raise ValueError("Stage-0 synthetic assets must be regular non-symlink files.")
    info = json.loads((root / "asset_info.json").read_text(encoding="utf-8"))
    if info.get("dataset_revision") != plan["dataset_revision"]:
        raise ValueError("Stage-0 processed asset revision differs from its plan.")
    if info.get("generator_schema") != plan["generator_schema"]:
        raise ValueError("Stage-0 generator schema differs from its plan.")
    expected_info = {
        "dataset_id": "synthetic_quality",
        "dataset_revision": plan["dataset_revision"],
        "generator_schema": plan["generator_schema"],
        "seed": int(generator["seed"]),
        "rng_schema": generator["rng_schema"],
        "processed_subject_count": int(generator["n_subjects"]),
        "n_blocks_per_interface": int(generator["n_blocks_per_interface"]),
        "n_repetitions_per_class_per_block": int(
            generator["n_repetitions_per_class_per_block"]
        ),
        "n_classes": int(generator["n_classes"]),
        "target_sfreq": float(generator["target_sfreq"]),
        "duration_sec": float(generator["duration_sec"]),
        "c_max": int(generator["c_max"]),
        "n_samples": int(generator["expected_n_samples"]),
        "n_quality_rows": int(generator["expected_n_quality_rows"]),
        "n_active_channels": len(generator["active_channel_names"]),
        "active_channel_names": list(generator["active_channel_names"]),
        "active_canonical_channel_ids": list(
            generator["active_canonical_channel_ids"]
        ),
        "conditions": ["wet", "dry"],
        "balanced_complete_blocks": True,
    }
    mismatched_info = {
        key: {"expected": expected, "observed": info.get(key)}
        for key, expected in expected_info.items()
        if info.get(key) != expected
    }
    if mismatched_info:
        raise ValueError(f"Stage-0 generator recipe mismatch: {mismatched_info}.")

    json_manifest = pd.read_json(root / "manifest.jsonl", lines=True)
    parquet_manifest = pd.read_parquet(root / "manifest.parquet")
    _validate_manifest_semantic_equality(json_manifest, parquet_manifest)

    # Reuse the same revision/protocol/count/class-map checks as Stage-1 before
    # constructing any analysis-only truth table.
    EEGProcessedDataset(
        [root],
        expected_revisions={"synthetic_quality": plan["dataset_revision"]},
        expected_protocol={
            "metadata_contract_version": "0.4-dev",
            "query_qc_extractor_version": (
                "filtered_cropped_pre_zscore_channel_std_median_v1"
            ),
            "external_continuous_schema": "impedance_mean_max_v1",
        },
        expected_dataset_counts={
            "synthetic_quality": {
                "n_samples": int(generator["expected_n_samples"]),
                "n_subjects": int(generator["n_subjects"]),
                "n_targets": int(generator["n_classes"]),
            }
        },
    )
    manifest = load_manifest(root).sort_values("h5_index").reset_index(drop=True)
    validate_manifest(manifest)
    truth = load_synthetic_quality_truth(root)
    _validate_manifest_design(manifest, generator)
    with h5py.File(root / "signals.h5", "r") as handle:
        x = handle["x"][:].astype(np.float32)
        masks = handle["channel_mask"][:].astype(bool)
        labels = handle["y"][:].astype(np.int64)
    _validate_observed_generator_structure(
        root,
        manifest,
        x=x,
        masks=masks,
        labels=labels,
        generator=generator,
    )
    if len(x) != len(manifest) or not np.array_equal(labels, manifest["label"].to_numpy(int)):
        raise ValueError("Stage-0 signals/manifest row or label alignment failed.")
    canonical = _stack_vectors(manifest["canonical_channel_ids"], x.shape[1], dtype=int)
    query_std = _stack_vectors(manifest["query_signal_std_by_channel"], x.shape[1])
    impedance = _stack_vectors(manifest["impedance_kohm_by_channel"], x.shape[1])
    expected_ids = np.zeros(x.shape[1], dtype=int)
    planned_ids = np.asarray(generator["active_canonical_channel_ids"], dtype=int)
    expected_ids[planned_ids - 1] = planned_ids
    if not np.array_equal(canonical, np.broadcast_to(expected_ids, canonical.shape)):
        raise ValueError("Stage-0 active canonical channel IDs differ from the frozen plan.")
    if not np.array_equal(np.isfinite(query_std), masks):
        raise ValueError("Stage-0 query-QC availability differs from the signal mask.")
    if not np.array_equal(np.isfinite(impedance), masks):
        raise ValueError("Stage-0 impedance availability differs from the signal mask.")
    normalized_query = _normalized_log_feature(query_std)
    q_tensor = extract_query_reliability_features(
        torch.from_numpy(x),
        channel_mask=torch.from_numpy(masks),
        channel_query_qc=torch.from_numpy(normalized_query),
        channel_query_missing=torch.from_numpy(~np.isfinite(query_std)),
    ).cpu().numpy()

    truth_index = truth.set_index(["sample_id", "canonical_channel_id"])
    observed_keys: set[tuple[str, int]] = set()
    rows: list[dict[str, Any]] = []
    for sample_index, manifest_row in manifest.iterrows():
        active_slots = np.flatnonzero(masks[sample_index])
        imp_norm = _normalized_log_feature(impedance[sample_index])
        imp_center = imp_norm[active_slots] - imp_norm[active_slots].mean()
        global_mean = _normalized_log_scalar(float(manifest_row["impedance_mean_kohm"]))
        global_max = _normalized_log_scalar(float(manifest_row["impedance_max_kohm"]))
        electrode = str(manifest_row["electrode_type"])
        for active_position, slot in enumerate(active_slots):
            channel_id = int(canonical[sample_index, slot])
            if channel_id <= 0:
                raise ValueError("Stage-0 active signal channel lacks a canonical ID.")
            key = (str(manifest_row["sample_id"]), channel_id)
            if key in observed_keys or key not in truth_index.index:
                raise ValueError("Stage-0 truth/signal sample-channel mapping is not one-to-one.")
            observed_keys.add(key)
            target_row = truth_index.loc[key, :]
            row = {
                "sample_id": key[0],
                "subject_id": str(manifest_row["subject_id"]),
                "acquisition_block_id": str(manifest_row["acquisition_block_id"]),
                "electrode_type": electrode,
                "canonical_channel_id": channel_id,
                "label": int(manifest_row["label"]),
                **dict(zip(Q_FEATURES, q_tensor[sample_index, slot].astype(float))),
                "m_channel_impedance": float(imp_norm[slot]),
                "m_channel_impedance_centered": float(imp_center[active_position]),
                "m_channel_impedance_abs_centered": float(abs(imp_center[active_position])),
                "m_channel_impedance_available": 1.0,
                "m_impedance_mean": global_mean,
                "m_impedance_max": global_max,
                "m_impedance_mean_available": 1.0,
                "m_impedance_max_available": 1.0,
                "m_electrode_wet": float(electrode == "wet"),
                "m_electrode_dry": float(electrode == "dry"),
            }
            row.update({name: float(target_row[name]) for name in plan["secondary_targets"]})
            row[str(plan["primary_target"])] = float(target_row[plan["primary_target"]])
            rows.append(row)
    expected_keys = set(truth_index.index)
    if observed_keys != expected_keys:
        raise ValueError("Stage-0 truth includes rows outside the active signal channels.")
    frame = pd.DataFrame(rows)
    predictors = {*Q_FEATURES, *M_FEATURES}
    if predictors & PREDICTOR_FORBIDDEN or not np.isfinite(frame[list(predictors)]).all().all():
        raise ValueError("Stage-0 predictor allowlist is invalid or non-finite.")
    integrity = {
        "all_checks_passed": False,
        "balanced_complete_blocks": True,
        "truth_model_input": False,
        "predictor_allowlist_only": True,
        "metadata_precedes_query": True,
        "active_channel_truth_coverage": 1.0,
        "natural_missingness_assay": "not_identifiable_complete_metadata",
        "asset_inventory_exact": True,
        "manifest_jsonl_parquet_semantically_equal": True,
        "manifest_semantic_sha256": _sha256_json(
            [
                {
                    name: _semantic_manifest_value(value)
                    for name, value in row.items()
                }
                for row in json_manifest.to_dict(orient="records")
            ]
        ),
        "generator_contract_matches": True,
    }
    return frame, integrity


def _validate_manifest_design(
    manifest: pd.DataFrame, generator: dict[str, Any]
) -> None:
    required = {
        "sample_id",
        "subject_id",
        "label",
        "electrode_type",
        "acquisition_block_id",
        "metadata_measurement_time_sec",
        "query_time_sec",
        "impedance_kohm_by_channel",
    }
    missing = sorted(required - set(manifest.columns))
    if missing:
        raise ValueError(f"Stage-0 manifest is missing columns: {missing}.")
    if not (
        manifest["metadata_measurement_time_sec"].astype(float)
        < manifest["query_time_sec"].astype(float)
    ).all():
        raise ValueError("Stage-0 metadata must be measured before every query.")
    counts = manifest.groupby(["acquisition_block_id", "label"]).size().unstack(fill_value=0)
    if counts.empty or not counts.eq(counts.iloc[0, 0]).all().all():
        raise ValueError("Stage-0 requires identical complete label counts in every block.")
    for _block, block in manifest.groupby("acquisition_block_id"):
        if block["subject_id"].nunique() != 1 or block["electrode_type"].nunique() != 1:
            raise ValueError("Stage-0 acquisition blocks may not cross subject/interface boundaries.")
        vectors = {
            tuple(
                None if not np.isfinite(item) else round(float(item), 8)
                for item in np.asarray(value, dtype=float)
            )
            for value in block["impedance_kohm_by_channel"]
        }
        if len(vectors) != 1:
            raise ValueError("Stage-0 impedance metadata must be block-constant.")


def _validate_observed_generator_structure(
    root: Path,
    manifest: pd.DataFrame,
    *,
    x: np.ndarray,
    masks: np.ndarray,
    labels: np.ndarray,
    generator: dict[str, Any],
) -> None:
    """Validate the generated files themselves, not only asset-info claims."""

    n_subjects = int(generator["n_subjects"])
    n_blocks = int(generator["n_blocks_per_interface"])
    n_repetitions = int(generator["n_repetitions_per_class_per_block"])
    n_classes = int(generator["n_classes"])
    n_rows = int(generator["expected_n_samples"])
    c_max = int(generator["c_max"])
    target_sfreq = float(generator["target_sfreq"])
    duration_sec = float(generator["duration_sec"])
    t_len = round(target_sfreq * duration_sec)
    if n_rows != n_subjects * 2 * n_blocks * n_repetitions * n_classes:
        raise ValueError("Stage-0 planned sample count does not equal its factorial grid.")
    if x.shape != (n_rows, c_max, t_len):
        raise ValueError(
            "Stage-0 signal tensor shape differs from the frozen generator recipe: "
            f"expected={(n_rows, c_max, t_len)}, observed={x.shape}."
        )
    if masks.shape != (n_rows, c_max) or labels.shape != (n_rows,):
        raise ValueError("Stage-0 mask/label tensor shapes differ from the frozen recipe.")
    if not np.isfinite(x).all():
        raise ValueError("Stage-0 signal tensor contains non-finite values.")
    planned_ids = np.asarray(generator["active_canonical_channel_ids"], dtype=int)
    expected_mask = np.zeros(c_max, dtype=bool)
    expected_mask[planned_ids - 1] = True
    if not np.array_equal(masks, np.broadcast_to(expected_mask, masks.shape)):
        raise ValueError("Stage-0 active signal mask differs from the frozen channel set.")

    if len(manifest) != n_rows:
        raise ValueError("Stage-0 manifest row count differs from the frozen factorial grid.")
    expected_subjects = {f"sub{index:03d}" for index in range(n_subjects)}
    if set(manifest["subject_id"].astype(str)) != expected_subjects:
        raise ValueError("Stage-0 manifest subject IDs differ from the frozen generator grid.")
    if set(manifest["electrode_type"].astype(str)) != {"wet", "dry"}:
        raise ValueError("Stage-0 manifest must contain the exact wet/dry interfaces.")
    expected_labels = set(range(n_classes))
    if set(manifest["label"].astype(int)) != expected_labels:
        raise ValueError("Stage-0 manifest labels differ from the frozen class grid.")

    expected_samples: set[str] = set()
    expected_group_keys: set[tuple[str, str, str, int]] = set()
    for subject in sorted(expected_subjects):
        for electrode in ("wet", "dry"):
            for block in range(n_blocks):
                run_id = f"block{block:02d}"
                block_id = f"synthetic_q_{subject}_{electrode}_{run_id}"
                expected_group_keys.update(
                    (subject, electrode, run_id, label) for label in range(n_classes)
                )
                block_rows = manifest.loc[
                    (manifest["subject_id"].astype(str) == subject)
                    & (manifest["electrode_type"].astype(str) == electrode)
                    & (manifest["run_id"].astype(str) == run_id)
                ]
                if len(block_rows) != n_repetitions * n_classes:
                    raise ValueError("Stage-0 manifest is missing a frozen acquisition block.")
                if not block_rows["session_id"].astype(str).eq(electrode).all():
                    raise ValueError("Stage-0 session/interface mapping differs from the recipe.")
                if not block_rows["acquisition_block_id"].astype(str).eq(block_id).all():
                    raise ValueError("Stage-0 acquisition-block IDs differ from the recipe.")
                if not block_rows["metadata_measurement_id"].astype(str).eq(block_id).all():
                    raise ValueError("Stage-0 metadata-measurement IDs differ from the recipe.")
                if not block_rows["reattach_flag"].astype(bool).eq(block > 0).all():
                    raise ValueError("Stage-0 reattachment flags differ from the recipe.")
                for repetition in range(n_repetitions):
                    for label in range(n_classes):
                        expected_samples.add(
                            f"{block_id}_rep{repetition:02d}_cls{label:02d}"
                        )
    observed_counts = manifest.groupby(
        ["subject_id", "electrode_type", "run_id", "label"]
    ).size()
    observed_group_keys = {
        (str(subject), str(electrode), str(run_id), int(label))
        for subject, electrode, run_id, label in observed_counts.index
    }
    if observed_group_keys != expected_group_keys or not observed_counts.eq(
        n_repetitions
    ).all():
        raise ValueError("Stage-0 manifest does not match the exact factorial class grid.")
    if set(manifest["sample_id"].astype(str)) != expected_samples:
        raise ValueError("Stage-0 sample IDs differ from the frozen factorial grid.")

    exact_manifest_scalars = {
        "dataset_id": "synthetic_quality",
        "sfreq_original": target_sfreq,
        "sfreq_processed": target_sfreq,
        "window_start_sec": 0.0,
        "window_duration_sec": duration_sec,
        "n_channels_original": len(generator["active_channel_names"]),
        "n_channels_used": len(generator["active_channel_names"]),
        "environment_note_code": "synthetic_quality_v1",
    }
    for field, expected in exact_manifest_scalars.items():
        observed = manifest[field]
        if isinstance(expected, float):
            matches = np.isclose(
                observed.astype(float).to_numpy(), expected, rtol=0.0, atol=1e-12
            ).all()
        else:
            matches = observed.eq(expected).all()
        if not matches:
            raise ValueError(f"Stage-0 manifest field {field!r} differs from the recipe.")

    preprocess = yaml.safe_load((root / "preprocess_config.yaml").read_text(encoding="utf-8"))
    expected_preprocess = {
        "target_sfreq": target_sfreq,
        "window_start_sec": 0.0,
        "window_duration_sec": duration_sec,
        "bandpass_low_hz": None,
        "bandpass_high_hz": None,
        "notch_hz": None,
        "normalize": "per_trial_channel_zscore",
        "c_max": c_max,
    }
    if preprocess != expected_preprocess:
        raise ValueError("Stage-0 preprocess_config.yaml differs from the frozen recipe.")
    class_map = json.loads((root / "class_map.json").read_text(encoding="utf-8"))
    expected_class_map = {
        str(label): {
            "label": label,
            "stimulus_frequency_hz": float(DEFAULT_FREQS[label]),
        }
        for label in range(n_classes)
    }
    if class_map != expected_class_map:
        raise ValueError("Stage-0 class_map.json differs from the frozen generator classes.")


def _select_alpha_nested_loso(
    frame: pd.DataFrame,
    *,
    features: tuple[str, ...],
    target: str,
    alphas: tuple[float, ...],
) -> tuple[float, dict[str, float]]:
    subjects = sorted(frame["subject_id"].unique())
    scores: dict[str, float] = {}
    for alpha in alphas:
        inner_mse: list[float] = []
        for held_subject in subjects:
            train = frame.loc[frame["subject_id"] != held_subject]
            test = frame.loc[frame["subject_id"] == held_subject]
            if train.empty or test.empty:
                raise ValueError("Nested LOSO produced an empty split.")
            model = _fit_ridge(
                train.loc[:, list(features)].to_numpy(float),
                train[target].to_numpy(float),
                train["subject_id"].to_numpy(str),
                alpha=alpha,
                features=features,
            )
            prediction = _predict_ridge(model, test.loc[:, list(features)].to_numpy(float))
            inner_mse.append(float(np.mean(np.square(test[target].to_numpy(float) - prediction))))
        scores[f"{alpha:.12g}"] = float(np.mean(inner_mse))
    selected = min(alphas, key=lambda value: (scores[f"{value:.12g}"], value))
    return float(selected), scores


def _fit_ridge(
    x: np.ndarray,
    y: np.ndarray,
    subjects: np.ndarray,
    *,
    alpha: float,
    features: tuple[str, ...],
) -> dict[str, Any]:
    if alpha <= 0 or len(x) != len(y) or len(y) != len(subjects):
        raise ValueError("Invalid Stage-0 ridge inputs.")
    unique, counts = np.unique(subjects, return_counts=True)
    count_by_subject = dict(zip(unique, counts))
    weights = np.asarray([len(y) / (len(unique) * count_by_subject[s]) for s in subjects])
    weight_sum = weights.sum()
    mean = np.sum(x * weights[:, None], axis=0) / weight_sum
    variance = np.sum(np.square(x - mean) * weights[:, None], axis=0) / weight_sum
    scale = np.sqrt(variance)
    scale[scale < 1e-8] = 1.0
    standardized = (x - mean) / scale
    y_mean = float(np.sum(y * weights) / weight_sum)
    sqrt_weight = np.sqrt(weights)
    design = standardized * sqrt_weight[:, None]
    response = (y - y_mean) * sqrt_weight
    gram = design.T @ design + alpha * np.eye(design.shape[1])
    coefficient = np.linalg.solve(gram, design.T @ response)
    return {
        "features": list(features),
        "alpha": float(alpha),
        "feature_mean": mean,
        "feature_scale": scale,
        "target_mean": y_mean,
        "coefficient": coefficient,
    }


def _predict_ridge(model: dict[str, Any], x: np.ndarray) -> np.ndarray:
    return model["target_mean"] + (
        (x - model["feature_mean"]) / model["feature_scale"]
    ) @ model["coefficient"]


def _block_bundle_shuffle(frame: pd.DataFrame) -> tuple[np.ndarray, float]:
    shuffled = frame.loc[:, M_FEATURES].copy()
    changed = np.zeros(len(frame), dtype=bool)
    for (_subject, _electrode), group in frame.groupby(["subject_id", "electrode_type"]):
        blocks = sorted(group["acquisition_block_id"].unique())
        if len(blocks) < 2:
            raise ValueError("Block-bundle shuffle requires at least two blocks per interface.")
        donor_by_block = dict(zip(blocks, blocks[1:] + blocks[:1]))
        source = group.drop_duplicates(["acquisition_block_id", "canonical_channel_id"])
        source = source.set_index(["acquisition_block_id", "canonical_channel_id"])
        for row_index in group.index:
            row = frame.loc[row_index]
            donor_key = (
                donor_by_block[str(row["acquisition_block_id"])],
                int(row["canonical_channel_id"]),
            )
            donor = source.loc[donor_key, list(M_FEATURES)].to_numpy(float)
            shuffled.loc[row_index, list(M_FEATURES)] = donor
            changed[row_index] = not np.allclose(
                donor, frame.loc[row_index, list(M_FEATURES)].to_numpy(float)
            )
    return shuffled.to_numpy(float), float(changed.mean())


def _channel_metadata_permutation(frame: pd.DataFrame) -> tuple[np.ndarray, float]:
    permuted = frame.loc[:, M_FEATURES].copy()
    changed = np.zeros(len(frame), dtype=bool)
    for _sample, group in frame.groupby("sample_id"):
        ordered = group.sort_values("canonical_channel_id")
        indices = ordered.index.to_numpy()
        source = ordered.loc[:, M_CHANNEL_FEATURES].to_numpy(float)
        rotated = np.roll(source, shift=1, axis=0)
        permuted.loc[indices, list(M_CHANNEL_FEATURES)] = rotated
        changed[indices] = np.any(~np.isclose(source, rotated), axis=1)
    return permuted.to_numpy(float), float(changed.mean())


def _subject_metrics(predictions: pd.DataFrame, *, target: str) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for subject, frame in predictions.groupby("subject_id"):
        y = frame[target].to_numpy(float)
        row: dict[str, Any] = {"subject_id": subject, "n_rows": len(frame)}
        errors: dict[str, float] = {}
        ranks: dict[str, float] = {}
        for arm in (
            "null",
            "Q",
            "M",
            "QM",
            "QM_block_shuffle",
            "QM_channel_permutation",
        ):
            prediction = frame[f"prediction_{arm}"].to_numpy(float)
            error = float(np.sum(np.square(y - prediction)))
            errors[arm] = error
            denominator = float(np.sum(np.square(y - y.mean())))
            row[f"{arm}_r2"] = 1.0 - error / max(denominator, 1e-12)
            row[f"{arm}_rmse"] = float(np.sqrt(error / len(y)))
            row[f"{arm}_mae"] = float(np.mean(np.abs(y - prediction)))
            sample_ranks = [
                _spearman(
                    sample[target].to_numpy(float),
                    sample[f"prediction_{arm}"].to_numpy(float),
                )
                for _sample_id, sample in frame.groupby("sample_id")
            ]
            ranks[arm] = float(np.mean(sample_ranks))
            row[f"{arm}_channel_rank_rho"] = ranks[arm]
        row["metadata_partial_r2_beyond_Q"] = (errors["Q"] - errors["QM"]) / max(
            errors["Q"], 1e-12
        )
        row["QM_minus_Q_channel_rank_rho"] = ranks["QM"] - ranks["Q"]
        row["block_shuffle_relative_sse_gain"] = (
            errors["QM_block_shuffle"] - errors["QM"]
        ) / max(errors["QM_block_shuffle"], 1e-12)
        row["channel_permutation_relative_sse_gain"] = (
            errors["QM_channel_permutation"] - errors["QM"]
        ) / max(errors["QM_channel_permutation"], 1e-12)
        rows.append(row)
    return pd.DataFrame(rows).sort_values("subject_id").reset_index(drop=True)


def _aggregate_contrasts(
    metrics: pd.DataFrame,
    gate: dict[str, Any],
    *,
    integrity: dict[str, Any],
) -> pd.DataFrame:
    observed = {
        "q_mean_r2": float(metrics["Q_r2"].mean()),
        "metadata_partial_r2_mean": float(metrics["metadata_partial_r2_beyond_Q"].mean()),
        "metadata_partial_r2_positive_subjects": int(
            metrics["metadata_partial_r2_beyond_Q"].gt(0).sum()
        ),
        "channel_rank_rho_gain_mean": float(metrics["QM_minus_Q_channel_rank_rho"].mean()),
        "channel_rank_rho_gain_positive_subjects": int(
            metrics["QM_minus_Q_channel_rank_rho"].gt(0).sum()
        ),
        "block_shuffle_relative_sse_gain_mean": float(
            metrics["block_shuffle_relative_sse_gain"].mean()
        ),
        "block_shuffle_positive_subjects": int(
            metrics["block_shuffle_relative_sse_gain"].gt(0).sum()
        ),
        "channel_permutation_relative_sse_gain_mean": float(
            metrics["channel_permutation_relative_sse_gain"].mean()
        ),
        "channel_permutation_positive_subjects": int(
            metrics["channel_permutation_relative_sse_gain"].gt(0).sum()
        ),
    }
    checks = {
        "q_identifiable": observed["q_mean_r2"] > float(gate["q_mean_r2_strict_min"]),
        "metadata_increment_size": observed["metadata_partial_r2_mean"]
        >= float(gate["metadata_partial_r2_mean_min"]),
        "metadata_increment_consistency": observed["metadata_partial_r2_positive_subjects"]
        >= int(gate["metadata_partial_r2_positive_subjects_min"]),
        "channel_rank_increment_size": observed["channel_rank_rho_gain_mean"]
        >= float(gate["channel_rank_rho_gain_mean_min"]),
        "channel_rank_increment_consistency": observed[
            "channel_rank_rho_gain_positive_subjects"
        ]
        >= int(gate["channel_rank_rho_gain_positive_subjects_min"]),
        "block_pairing_reliance": observed["block_shuffle_relative_sse_gain_mean"]
        >= float(gate["block_shuffle_relative_sse_gain_mean_min"]),
        "block_pairing_consistency": observed["block_shuffle_positive_subjects"]
        >= int(gate["block_shuffle_positive_subjects_min"]),
        "channel_pairing_reliance": observed["channel_permutation_relative_sse_gain_mean"]
        >= float(gate["channel_permutation_relative_sse_gain_mean_min"]),
        "channel_pairing_consistency": observed["channel_permutation_positive_subjects"]
        >= int(gate["channel_permutation_positive_subjects_min"]),
        "label_only_null": bool(integrity["label_only_subject_macro_r2_passed"]),
        "class_target_balance": bool(integrity["class_target_mean_range_passed"]),
    }
    return pd.DataFrame(
        [
            {
                "schema": "cfeg.synthetic-reliability-stage0-contrasts.v1",
                "status": "passed" if all(checks.values()) else "failed",
                **observed,
                "label_only_subject_macro_r2": integrity[
                    "label_only_subject_macro_r2"
                ],
                "class_target_mean_range": integrity["class_target_mean_range"],
                **{f"check_{name}": passed for name, passed in checks.items()},
            }
        ]
    )


def _label_leakage_audit(
    frame: pd.DataFrame,
    *,
    target: str,
    contract: dict[str, Any],
) -> dict[str, Any]:
    subject_r2: list[float] = []
    for held_subject in sorted(frame["subject_id"].unique()):
        train = frame.loc[frame["subject_id"] != held_subject]
        test = frame.loc[frame["subject_id"] == held_subject]
        global_mean = float(train[target].mean())
        means = train.groupby("label")[target].mean().to_dict()
        prediction = test["label"].map(means).fillna(global_mean).to_numpy(float)
        values = test[target].to_numpy(float)
        error = float(np.sum(np.square(values - prediction)))
        denominator = float(np.sum(np.square(values - values.mean())))
        subject_r2.append(1.0 - error / max(denominator, 1e-12))
    label_only_r2 = float(np.mean(subject_r2))
    class_means = frame.groupby("label")[target].mean()
    class_range = float(class_means.max() - class_means.min())
    r2_max = float(contract["label_only_subject_macro_r2_max"])
    range_max = float(contract["class_target_mean_range_max"])
    return {
        "label_only_subject_macro_r2": label_only_r2,
        "label_only_subject_macro_r2_max": r2_max,
        "label_only_subject_macro_r2_passed": label_only_r2 <= r2_max,
        "class_target_mean_range": class_range,
        "class_target_mean_range_max": range_max,
        "class_target_mean_range_passed": class_range <= range_max,
    }


def _spearman(left: np.ndarray, right: np.ndarray) -> float:
    left_rank = rankdata(left)
    right_rank = rankdata(right)
    if np.std(left_rank) < 1e-12 or np.std(right_rank) < 1e-12:
        return 0.0
    return float(np.corrcoef(left_rank, right_rank)[0, 1])


def _validate_plan(plan: dict[str, Any]) -> None:
    if plan.get("schema") != STAGE0_PLAN_SCHEMA:
        raise ValueError("Unknown synthetic reliability Stage-0 plan schema.")
    required_false = (
        "population_inference_allowed",
        "confirmatory_execution_authorized",
        "wearable_s1_s3_reuse_allowed",
        "decoder_checkpoint_reuse_allowed",
    )
    if any(plan.get(field) is not False for field in required_false):
        raise ValueError("Stage-0 plan must prohibit population, confirmatory, wearable, and reuse.")
    if plan.get("scope") != "synthetic_engineering_only":
        raise ValueError("Stage-0 is restricted to synthetic engineering.")
    if plan.get("asset_publication") != {
        "freshness": "output_must_not_exist",
        "concurrency_reservation": "fixed_name_atomic_mkdir",
        "transaction": "hidden_staging_fsync_then_atomic_root_rename",
        "reservation_release": "after_success_or_handled_prepublication_failure",
    }:
        raise ValueError("Stage-0 asset publication contract is missing or stale.")
    if plan.get("outcome_publication") != {
        "output_root": "outputs/engineering/reliability-spatial-v1/stage0",
        "freshness": "output_must_not_exist",
        "concurrency_reservation": "fixed_name_atomic_mkdir_before_assay",
        "transaction": "hidden_staging_fsync_then_atomic_root_rename",
        "failed_gate_is_terminal": True,
        "overwrite_or_supersede_allowed": False,
        "reservation_release": "only_after_terminal_outcome_publication",
    }:
        raise ValueError("Stage-0 one-shot outcome publication contract is missing or stale.")
    if (
        plan.get("candidate_id") != "reliability-spatial-v1"
        or plan.get("dataset_revision") != "synthetic_quality_v1"
        or plan.get("generator_schema") != "cfeg.synthetic-quality-generator.v2"
        or plan.get("row_unit") != "sample_x_active_canonical_channel"
    ):
        raise ValueError("Stage-0 candidate/dataset/generator/row-unit contract is invalid.")
    if plan.get("primary_target") != "realized_signal_fraction":
        raise ValueError("Stage-0 primary target must include realized dropout corruption.")
    if plan.get("secondary_targets") != [
        "latent_badness",
        "oracle_signal_fraction",
        "realized_clean_signal_mse",
    ]:
        raise ValueError("Stage-0 secondary target allowlist is invalid.")
    if plan.get("feature_schema") != "query_window_reliability_features_v1":
        raise ValueError("Unknown Stage-0 Q feature schema.")
    evaluation = plan.get("evaluation") or {}
    if (
        evaluation.get("outer_split") != "leave_one_subject_out"
        or evaluation.get("inner_selection") != "leave_one_training_subject_out"
        or evaluation.get("subject_macro_aggregation") is not True
        or plan.get("controls")
        != {
            "block_bundle_shuffle": "within_subject_and_electrode_type",
            "channel_impedance_permutation": "within_sample_active_channel_rotation",
        }
    ):
        raise ValueError("Stage-0 split/control contract is invalid.")
    alphas = evaluation.get("ridge_alphas") or []
    if not alphas or any(float(alpha) <= 0 for alpha in alphas):
        raise ValueError("Stage-0 ridge alpha grid must be positive and non-empty.")
    if len(set(map(float, alphas))) != len(alphas):
        raise ValueError("Stage-0 ridge alpha grid values must be unique.")
    generator = plan.get("generator") or {}
    required_generator_fields = {
        "seed",
        "rng_schema",
        "n_subjects",
        "n_blocks_per_interface",
        "n_repetitions_per_class_per_block",
        "n_classes",
        "target_sfreq",
        "duration_sec",
        "c_max",
        "active_channel_names",
        "active_canonical_channel_ids",
        "expected_n_samples",
        "expected_n_quality_rows",
        "required_asset_files",
    }
    if set(generator) != required_generator_fields:
        raise ValueError("Stage-0 generator contract fields are incomplete or unexpected.")
    integer_positive = (
        "n_subjects",
        "n_blocks_per_interface",
        "n_repetitions_per_class_per_block",
        "n_classes",
        "c_max",
        "expected_n_samples",
        "expected_n_quality_rows",
    )
    if any(int(generator[field]) <= 0 for field in integer_positive):
        raise ValueError("Stage-0 generator count values must be positive.")
    if int(generator["n_subjects"]) < 3:
        raise ValueError("Stage-0 nested LOSO requires at least three planned subjects.")
    if float(generator["target_sfreq"]) <= 0 or float(generator["duration_sec"]) <= 0:
        raise ValueError("Stage-0 sampling frequency and duration must be positive.")
    channel_names = list(generator["active_channel_names"])
    channel_ids = [int(value) for value in generator["active_canonical_channel_ids"]]
    if (
        not channel_names
        or len(channel_names) != len(channel_ids)
        or len(set(channel_names)) != len(channel_names)
        or len(set(channel_ids)) != len(channel_ids)
        or any(value <= 0 or value > int(generator["c_max"]) for value in channel_ids)
    ):
        raise ValueError("Stage-0 active channel contract is invalid.")
    expected_samples = (
        int(generator["n_subjects"])
        * 2
        * int(generator["n_blocks_per_interface"])
        * int(generator["n_repetitions_per_class_per_block"])
        * int(generator["n_classes"])
    )
    if (
        int(generator["expected_n_samples"]) != expected_samples
        or int(generator["expected_n_quality_rows"])
        != expected_samples * len(channel_ids)
        or tuple(generator["required_asset_files"])
        != SYNTHETIC_QUALITY_ASSET_FILES
        or generator.get("rng_schema") != "numpy_seedsequence_hierarchical_v1"
    ):
        raise ValueError("Stage-0 generator dimensions or asset inventory are inconsistent.")
    leakage = plan.get("leakage_audit") or {}
    if set(leakage) != {
        "label_only_subject_macro_r2_max",
        "class_target_mean_range_max",
    } or any(not np.isfinite(float(value)) or float(value) < 0 for value in leakage.values()):
        raise ValueError("Stage-0 leakage-audit thresholds are invalid.")
    expected_gate_fields = {
        "q_mean_r2_strict_min",
        "metadata_partial_r2_mean_min",
        "metadata_partial_r2_positive_subjects_min",
        "channel_rank_rho_gain_mean_min",
        "channel_rank_rho_gain_positive_subjects_min",
        "block_shuffle_relative_sse_gain_mean_min",
        "block_shuffle_positive_subjects_min",
        "channel_permutation_relative_sse_gain_mean_min",
        "channel_permutation_positive_subjects_min",
    }
    gate = plan.get("gate") or {}
    if set(gate) != expected_gate_fields:
        raise ValueError("Stage-0 gate fields are incomplete or unexpected.")
    for field in expected_gate_fields:
        value = float(gate[field])
        if not np.isfinite(value):
            raise ValueError("Stage-0 gate values must be finite.")
        if field.endswith("positive_subjects_min") and not (
            0 <= int(value) <= int(generator["n_subjects"]) and value == int(value)
        ):
            raise ValueError("Stage-0 consistency gates must be valid subject counts.")


def _stack_vectors(series: pd.Series, length: int, dtype=float) -> np.ndarray:
    values = [np.asarray(value, dtype=dtype).reshape(-1) for value in series]
    if any(len(value) != length for value in values):
        raise ValueError(f"Stage-0 aligned vectors must all have length {length}.")
    return np.stack(values)


def _normalized_log_feature(values: np.ndarray) -> np.ndarray:
    finite = np.isfinite(values)
    out = np.zeros_like(values, dtype=np.float32)
    out[finite] = np.clip(
        np.log1p(np.maximum(values[finite], 0.0)) / np.log1p(100.0), 0.0, 2.0
    )
    return out


def _normalized_log_scalar(value: float) -> float:
    return float(np.clip(np.log1p(max(value, 0.0)) / np.log1p(100.0), 0.0, 2.0))


def _serializable_model(model: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value.tolist() if isinstance(value, np.ndarray) else value
        for key, value in model.items()
    }


def _asset_fingerprints(root: Path) -> dict[str, str]:
    return {
        name: _sha256_file(root / name)
        for name in SYNTHETIC_QUALITY_ASSET_FILES
    }


def _row_identity_sha256(frame: pd.DataFrame) -> str:
    rows = sorted(
        (str(row.sample_id), int(row.canonical_channel_id))
        for row in frame.itertuples(index=False)
    )
    return _sha256_json(rows)


def _validate_manifest_semantic_equality(
    left: pd.DataFrame, right: pd.DataFrame
) -> None:
    if list(left.columns) != list(right.columns) or len(left) != len(right):
        raise ValueError("Stage-0 JSONL and Parquet manifest schemas differ.")
    for left_row, right_row in zip(
        left.to_dict(orient="records"), right.to_dict(orient="records")
    ):
        if any(
            not _manifest_values_equal(left_row[name], right_row[name])
            for name in left.columns
        ):
            raise ValueError("Stage-0 JSONL and Parquet manifests differ semantically.")


def _manifest_values_equal(left: Any, right: Any) -> bool:
    if isinstance(left, (list, tuple, np.ndarray)) or isinstance(
        right, (list, tuple, np.ndarray)
    ):
        left_values = list(left) if isinstance(left, (list, tuple, np.ndarray)) else []
        right_values = list(right) if isinstance(right, (list, tuple, np.ndarray)) else []
        return len(left_values) == len(right_values) and all(
            _manifest_values_equal(a, b) for a, b in zip(left_values, right_values)
        )
    if isinstance(left, np.generic):
        left = left.item()
    if isinstance(right, np.generic):
        right = right.item()
    left_missing = left is None or (isinstance(left, float) and np.isnan(left))
    right_missing = right is None or (isinstance(right, float) and np.isnan(right))
    if left_missing or right_missing:
        return left_missing and right_missing
    if (
        isinstance(left, (int, float))
        and not isinstance(left, bool)
        and isinstance(right, (int, float))
        and not isinstance(right, bool)
    ):
        return bool(np.isclose(float(left), float(right), rtol=1e-9, atol=1e-8))
    return bool(left == right)


def _semantic_manifest_value(value: Any) -> Any:
    if isinstance(value, (list, tuple, np.ndarray)):
        return [_semantic_manifest_value(item) for item in value]
    if isinstance(value, np.generic):
        value = value.item()
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None
    if isinstance(value, float):
        return round(value, 9)
    return value


def _validate_result_finite(result: dict[str, Any]) -> None:
    for name in ("predictions", "subject_metrics", "contrasts"):
        frame = result.get(name)
        if not isinstance(frame, pd.DataFrame):
            raise TypeError(f"Stage-0 {name} must be a DataFrame.")
        numeric = frame.select_dtypes(include=[np.number])
        if not np.isfinite(numeric.to_numpy(dtype=float)).all():
            raise ValueError(f"Stage-0 {name} contains non-finite values.")
    # Strict JSON serialization recursively rejects non-finite model/receipt values.
    _json_roundtrip(result.get("fold_models"))
    _json_roundtrip(result.get("receipt"))


def _json_roundtrip(value: Any) -> Any:
    return json.loads(
        json.dumps(value, sort_keys=True, allow_nan=False, default=_json_scalar)
    )


def _json_scalar(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"Value is not JSON serializable: {type(value).__name__}.")


def _write_csv_fsynced(frame: pd.DataFrame, path: Path) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        frame.to_csv(handle, index=False)
        handle.flush()
        os.fsync(handle.fileno())


def _write_json_fsynced(path: Path, value: Any) -> None:
    payload = json.dumps(
        value,
        indent=2,
        sort_keys=True,
        allow_nan=False,
        default=_json_scalar,
    )
    with path.open("w", encoding="utf-8") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _reject_symlink_components(path: Path) -> None:
    candidate = path.absolute()
    for component in (candidate, *candidate.parents):
        if component.exists() and component.is_symlink():
            raise ValueError(f"Stage-0 paths cannot traverse symlinks: {component}.")


def _sha256_json(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
