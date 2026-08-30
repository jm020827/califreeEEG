from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
import pandas as pd

from cfeg.analysis.ood_coverage import validate_primary_contract_pair
from cfeg.analysis.primary_inference import primary_subject_inference
from cfeg.analysis.provenance import (
    PRIMARY_ANALYSIS_MANIFEST_COLUMNS,
    VerifiedPredictionBundle,
    primary_analysis_manifest_sha256,
)
from cfeg.metrics import balanced_accuracy


def aggregate_primary_confirmatory(
    prediction_pairs: Sequence[tuple[VerifiedPredictionBundle, VerifiedPredictionBundle]],
    manifest: pd.DataFrame,
    *,
    expected_seeds: Sequence[int],
    expected_n_folds: int = 5,
    expected_n_samples: int,
    expected_n_subjects: int,
    expected_n_labels: int = 12,
    expected_dataset_id: str = "wearable",
    expected_split_seed: int = 42,
    condition_column: str = "electrode_type",
    expected_conditions: Sequence[str] = ("dry", "wet"),
    inference_seed: int = 42,
    n_resamples: int = 10_000,
    inference_alpha: float = 0.05,
    inference_alternative: str = "greater",
    success_threshold: float | None = None,
    sesoi_balanced_accuracy: float | None = None,
    primary_excluded_subject_ids: Sequence[str],
    primary_included_subject_ids: Sequence[str] | None = None,
    execution_manifest: Mapping[str, object],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, object]]:
    """Aggregate the frozen seed×split A0/A2 grid at the participant level."""
    seeds = tuple(sorted(int(seed) for seed in expected_seeds))
    if len(seeds) not in {3, 4, 5} or len(set(seeds)) != len(seeds):
        raise ValueError("expected_seeds must freeze exactly 3–5 distinct seed values.")
    if expected_n_folds < 1:
        raise ValueError("expected_n_folds must be at least one.")
    if expected_n_subjects < 2 or expected_n_labels < 2:
        raise ValueError("Expected subject and label counts must both be at least two.")
    execution_jobs, execution_contract_sha256 = _validate_execution_grid(
        execution_manifest,
        seeds=seeds,
        n_folds=expected_n_folds,
    )
    execution_manifest_content_sha256 = _required_sha256(
        execution_manifest.get("manifest_content_sha256"),
        "execution manifest content",
    )
    lockbox_reveal_receipt_sha256 = _required_sha256(
        execution_manifest.get("lockbox_reveal_receipt_sha256"),
        "lockbox reveal receipt",
    )
    required_manifest = {"sample_id", "dataset_id", "subject_id", "label", condition_column}
    missing_manifest = sorted(required_manifest - set(manifest.columns))
    if missing_manifest:
        raise ValueError(f"Manifest missing confirmatory columns: {missing_manifest}")
    if condition_column not in PRIMARY_ANALYSIS_MANIFEST_COLUMNS:
        raise ValueError("Condition column is not bound by the analysis manifest contract.")
    metadata = manifest.copy()
    metadata["sample_id"] = metadata["sample_id"].astype(str)
    if metadata["sample_id"].duplicated().any():
        raise ValueError("Confirmatory manifest sample IDs must be unique.")
    excluded = {str(value) for value in primary_excluded_subject_ids}
    included = (
        {str(value) for value in primary_included_subject_ids}
        if primary_included_subject_ids is not None
        else None
    )
    observed_subjects = set(metadata["subject_id"].astype(str))
    missing_exclusions = sorted(excluded - observed_subjects)
    if missing_exclusions:
        raise ValueError(f"Primary exclusion IDs are absent from the asset: {missing_exclusions}.")
    if included is not None:
        missing_inclusions = sorted(included - observed_subjects)
        if missing_inclusions:
            raise ValueError(
                f"Primary inclusion IDs are absent from the asset: {missing_inclusions}."
            )
        if included & excluded:
            raise ValueError("Primary inclusion and exclusion subject IDs overlap.")
        metadata = metadata.loc[
            metadata["subject_id"].astype(str).isin(included)
        ].copy()
    else:
        metadata = metadata.loc[~metadata["subject_id"].astype(str).isin(excluded)].copy()
    if len(metadata) != expected_n_samples:
        raise ValueError(
            f"Expected {expected_n_samples} manifest samples, observed {len(metadata)}."
        )
    if set(metadata["label"].astype(int)) != set(range(expected_n_labels)):
        raise ValueError("Manifest labels do not match the frozen contiguous label set.")
    if set(metadata["dataset_id"].astype(str)) != {str(expected_dataset_id)}:
        raise ValueError("Manifest dataset identity does not match the frozen primary dataset.")
    if metadata["subject_id"].astype(str).nunique() != expected_n_subjects:
        raise ValueError("Manifest subject count does not match the frozen primary cohort.")
    condition_values = {str(value) for value in expected_conditions}
    if set(metadata[condition_column].astype(str)) != condition_values:
        raise ValueError("Manifest acquisition conditions do not match the frozen set.")

    pair_records: list[dict[str, object]] = []
    subject_seed_rows: list[dict[str, object]] = []
    condition_seed_rows: list[dict[str, object]] = []
    samples_by_seed_fold: dict[tuple[int, int], set[str]] = {}
    contracts_by_seed_fold: dict[tuple[int, int], dict[str, object]] = {}
    for baseline_bundle, candidate_bundle in prediction_pairs:
        if not isinstance(baseline_bundle, VerifiedPredictionBundle) or not isinstance(
            candidate_bundle, VerifiedPredictionBundle
        ):
            raise TypeError("Confirmatory aggregation requires verified prediction bundles.")
        baseline_frame = baseline_bundle.frame
        candidate_frame = candidate_bundle.frame
        contract = validate_primary_contract_pair(baseline_frame, candidate_frame)
        seed = int(contract["optimization_seed"])
        fold = int(contract["fold_index"])
        key = (seed, fold)
        if key in contracts_by_seed_fold:
            raise ValueError(f"Duplicate confirmatory prediction pair for seed/fold {key}.")
        if contract["scenario"] != "clean":
            raise ValueError("Primary confirmatory aggregation accepts scenario=clean only.")
        if int(contract["n_folds"]) != expected_n_folds or not 0 <= fold < expected_n_folds:
            raise ValueError(f"Prediction pair has invalid outer-fold contract: {key}.")
        for role, prefix in (
            ("A0_eeg_only", "baseline"),
            ("A2_structured_condition_prompt", "candidate"),
        ):
            planned_job = execution_jobs[(seed, fold, role)]
            observed = {
                "execution_job_id": contract[f"{prefix}_execution_job_id"],
                "planned_config_sha256": contract[f"{prefix}_planned_config_sha256"],
            }
            expected = {
                "execution_job_id": planned_job["job_id"],
                "planned_config_sha256": planned_job["resolved_config_sha256"],
            }
            if observed != expected:
                raise ValueError(
                    f"Prediction bundle is outside the execution manifest for {key}/{role}: "
                    f"expected={expected}, observed={observed}."
                )
        if contract["execution_contract_sha256"] != execution_contract_sha256:
            raise ValueError(f"Prediction execution contract differs from the manifest for {key}.")
        if (
            contract["execution_manifest_content_sha256"]
            != execution_manifest_content_sha256
            or contract["lockbox_reveal_receipt_sha256"]
            != lockbox_reveal_receipt_sha256
        ):
            raise ValueError(
                f"Prediction reveal provenance differs from the manifest for {key}."
            )
        baseline = _prediction_rows(baseline_frame, "baseline")
        candidate = _prediction_rows(candidate_frame, "candidate")
        paired = baseline.rename(
            columns={"label": "baseline_label", "prediction": "baseline_prediction"}
        ).merge(
            candidate.rename(
                columns={"label": "candidate_label", "prediction": "candidate_prediction"}
            ),
            on="sample_id",
            validate="one_to_one",
        )
        if len(paired) != len(baseline) or len(paired) != len(candidate):
            raise ValueError(f"A0/A2 sample sets differ for seed/fold {key}.")
        if not np.array_equal(paired["baseline_label"], paired["candidate_label"]):
            raise ValueError(f"A0/A2 labels differ for seed/fold {key}.")
        for column in (
            "baseline_label",
            "candidate_label",
            "baseline_prediction",
            "candidate_prediction",
        ):
            if not paired[column].between(0, expected_n_labels - 1).all():
                raise ValueError(f"{column} is outside the frozen class range for {key}.")
        sample_ids = paired["sample_id"].astype(str).tolist()
        observed_manifest_hash = primary_analysis_manifest_sha256(metadata, sample_ids)
        if contract["analysis_manifest_sha256"] != observed_manifest_hash:
            raise ValueError(f"Manifest contract mismatch for seed/fold {key}.")
        columns = ["sample_id", "dataset_id", "subject_id", "label", condition_column]
        paired = paired.merge(metadata[columns], on="sample_id", validate="one_to_one")
        if len(paired) != len(baseline):
            raise ValueError(f"Manifest is missing prediction samples for seed/fold {key}.")
        if not np.array_equal(paired["baseline_label"].astype(int), paired["label"].astype(int)):
            raise ValueError(f"Prediction labels disagree with the manifest for {key}.")

        samples_by_seed_fold[key] = set(sample_ids)
        contracts_by_seed_fold[key] = contract
        pair_records.append(
            {
                "optimization_seed": seed,
                "fold_index": fold,
                "n_samples": len(paired),
                "sample_identity_sha256": contract["sample_identity_sha256"],
                "split_assignment_sha256": contract["split_assignment_sha256"],
                "split_manifest_sha256": contract["split_manifest_sha256"],
                "primary_fairness_hash": contract["primary_fairness_hash"],
                "baseline_prediction_csv": str(baseline_bundle.path),
                "candidate_prediction_csv": str(candidate_bundle.path),
                "baseline_prediction_csv_sha256": baseline_bundle.artifact.get(
                    "prediction_csv_sha256"
                ),
                "candidate_prediction_csv_sha256": candidate_bundle.artifact.get(
                    "prediction_csv_sha256"
                ),
                "baseline_provenance_json": (
                    str(baseline_bundle.provenance_path)
                    if baseline_bundle.provenance_path is not None
                    else None
                ),
                "candidate_provenance_json": (
                    str(candidate_bundle.provenance_path)
                    if candidate_bundle.provenance_path is not None
                    else None
                ),
                "baseline_provenance_sha256": baseline_bundle.provenance_sha256,
                "candidate_provenance_sha256": candidate_bundle.provenance_sha256,
                "baseline_checkpoint_sha256": contract["baseline_checkpoint_sha256"],
                "candidate_checkpoint_sha256": contract["candidate_checkpoint_sha256"],
            }
        )
        subject_seed_rows.extend(
            _group_pair_metrics(
                paired,
                ["dataset_id", "subject_id"],
                seed=seed,
                fold=fold,
                expected_n_labels=expected_n_labels,
            )
        )
        condition_seed_rows.extend(
            _group_pair_metrics(
                paired,
                ["dataset_id", "subject_id", condition_column],
                seed=seed,
                fold=fold,
                expected_n_labels=expected_n_labels,
            )
        )

    expected_keys = {(seed, fold) for seed in seeds for fold in range(expected_n_folds)}
    observed_keys = set(contracts_by_seed_fold)
    if observed_keys != expected_keys:
        raise ValueError(
            "Confirmatory seed×fold grid is incomplete or unexpected: "
            f"missing={sorted(expected_keys - observed_keys)}, "
            f"extra={sorted(observed_keys - expected_keys)}."
        )
    _validate_cross_pair_contracts(contracts_by_seed_fold, seeds, expected_n_folds)
    if {int(contract["split_seed"]) for contract in contracts_by_seed_fold.values()} != {
        int(expected_split_seed)
    }:
        raise ValueError("Prediction split seed does not match the frozen analysis plan.")
    _validate_outer_test_coverage(samples_by_seed_fold, metadata, seeds, expected_n_folds)

    seed_subjects = pd.DataFrame(subject_seed_rows).sort_values(
        ["dataset_id", "subject_id", "optimization_seed"]
    )
    if len(seed_subjects) != expected_n_subjects * len(seeds):
        raise ValueError(
            "Subject×seed table is incomplete: expected "
            f"{expected_n_subjects * len(seeds)}, observed {len(seed_subjects)}."
        )
    counts = seed_subjects.groupby(["dataset_id", "subject_id"])["optimization_seed"].agg(
        lambda values: tuple(sorted(int(value) for value in values))
    )
    bad_seed_sets = counts[~counts.map(lambda value: value == seeds)]
    if len(bad_seed_sets):
        raise ValueError("At least one subject is missing a frozen optimization seed.")
    fold_counts = seed_subjects.groupby(["dataset_id", "subject_id"])["fold_index"].nunique()
    if not fold_counts.eq(1).all():
        raise ValueError("A subject appears in more than one outer test fold across seeds.")

    subjects = _average_subject_seeds(seed_subjects, ["dataset_id", "subject_id"])
    if len(subjects) != expected_n_subjects:
        raise ValueError(
            f"Expected {expected_n_subjects} independent subjects, observed {len(subjects)}."
        )
    condition_seed = pd.DataFrame(condition_seed_rows).sort_values(
        ["dataset_id", "subject_id", condition_column, "optimization_seed"]
    )
    observed_conditions = condition_seed.groupby(["dataset_id", "subject_id"])[
        condition_column
    ].agg(lambda values: tuple(sorted({str(value) for value in values})))
    expected_condition_tuple = tuple(sorted(condition_values))
    if not observed_conditions.map(lambda value: value == expected_condition_tuple).all():
        raise ValueError("At least one subject is missing a frozen acquisition condition.")
    condition_seed_counts = condition_seed.groupby(["dataset_id", "subject_id", condition_column])[
        "optimization_seed"
    ].agg(lambda values: tuple(sorted(int(value) for value in values)))
    if not condition_seed_counts.map(lambda value: value == seeds).all():
        raise ValueError("At least one subject-condition is missing a frozen seed.")
    condition_subjects = _average_subject_seeds(
        condition_seed, ["dataset_id", "subject_id", condition_column]
    )
    if success_threshold is not None:
        if not 0.0 <= float(success_threshold) <= 1.0:
            raise ValueError("success_threshold must be within [0, 1].")
        condition_subjects["baseline_success"] = condition_subjects[
            "baseline_balanced_accuracy"
        ].ge(float(success_threshold))
        condition_subjects["candidate_success"] = condition_subjects[
            "candidate_balanced_accuracy"
        ].ge(float(success_threshold))
        condition_subjects["transition"] = [
            _transition(baseline, candidate)
            for baseline, candidate in zip(
                condition_subjects["baseline_success"],
                condition_subjects["candidate_success"],
            )
        ]
    null_margin = float(sesoi_balanced_accuracy or 0.0)
    inference = primary_subject_inference(
        subjects,
        alpha=inference_alpha,
        alternative=inference_alternative,
        null_margin_ba=null_margin,
        seed=inference_seed,
        n_resamples=n_resamples,
    )
    condition_descriptive = {
        str(condition): {
            "n_subjects": len(group),
            "mean_subject_balanced_accuracy_delta": float(
                group["balanced_accuracy_delta"].mean()
            ),
            "median_subject_balanced_accuracy_delta": float(
                group["balanced_accuracy_delta"].median()
            ),
            "inferential_claim_allowed": False,
        }
        for condition, group in condition_subjects.groupby(condition_column, sort=True)
    }
    global_contract = contracts_by_seed_fold[(seeds[0], 0)]
    summary: dict[str, object] = {
        "status": "primary_grid_validated",
        "expected_seeds": list(seeds),
        "n_seeds": len(seeds),
        "n_folds": expected_n_folds,
        "n_subjects": len(subjects),
        "n_labels": expected_n_labels,
        "n_prediction_pairs": len(pair_records),
        "n_unique_test_samples": len(metadata),
        "mean_seed_averaged_baseline_balanced_accuracy": float(
            subjects["baseline_balanced_accuracy"].mean()
        ),
        "mean_seed_averaged_candidate_balanced_accuracy": float(
            subjects["candidate_balanced_accuracy"].mean()
        ),
        "baseline_worst_subject_balanced_accuracy": float(
            subjects["baseline_balanced_accuracy"].min()
        ),
        "candidate_worst_subject_balanced_accuracy": float(
            subjects["candidate_balanced_accuracy"].min()
        ),
        "condition_descriptive": condition_descriptive,
        "multiplicity_scope": "single_overall_primary_contrast_conditions_descriptive",
        "success_threshold": success_threshold,
        "sesoi_balanced_accuracy": sesoi_balanced_accuracy,
        "primary_minimum_effect_claim_supported": (
            inference["minimum_effect_claim_supported"]
            if sesoi_balanced_accuracy is not None
            else None
        ),
        "learned_condition_cells": (
            int(condition_subjects["transition"].eq("learned").sum())
            if "transition" in condition_subjects
            else None
        ),
        "forgotten_condition_cells": (
            int(condition_subjects["transition"].eq("forgotten").sum())
            if "transition" in condition_subjects
            else None
        ),
        "learned_minus_forgotten_condition_cells": (
            int(
                condition_subjects["transition"].eq("learned").sum()
                - condition_subjects["transition"].eq("forgotten").sum()
            )
            if "transition" in condition_subjects
            else None
        ),
        "primary_excluded_subject_ids": sorted(excluded),
        "primary_included_subject_ids": sorted(included) if included is not None else None,
        "execution_contract_sha256": execution_contract_sha256,
        "execution_manifest_content_sha256": execution_manifest_content_sha256,
        "lockbox_reveal_receipt_sha256": lockbox_reveal_receipt_sha256,
        "execution_manifest_job_count": len(execution_jobs),
        "execution_phase": global_contract["execution_phase"],
        "analysis_plan_sha256": global_contract["analysis_plan_sha256"],
        "cohort_roles_sha256": global_contract["cohort_roles_sha256"],
        "cohort_sha256": global_contract["cohort_sha256"],
        "parameter_schema_sha256": global_contract["parameter_schema_sha256"],
        "asset_provenance_sha256": global_contract["asset_provenance_sha256"],
        "source_commit_sha": global_contract["source_commit_sha"],
        "source_dirty": global_contract["source_dirty"],
        "source_tree_sha256": global_contract["source_tree_sha256"],
        **inference,
    }
    runs = pd.DataFrame(pair_records).sort_values(["optimization_seed", "fold_index"])
    return runs, seed_subjects, subjects, condition_subjects, summary


def _validate_execution_grid(
    manifest: Mapping[str, object],
    *,
    seeds: tuple[int, ...],
    n_folds: int,
) -> tuple[dict[tuple[int, int, str], Mapping[str, object]], str]:
    if manifest.get("schema") != "cfeg.primary-execution-manifest.v1":
        raise ValueError("Confirmatory aggregation requires a primary execution manifest.")
    if manifest.get("execution_allowed") is not True or manifest.get("plan_status") != "frozen":
        raise ValueError("Execution manifest is not authorized by a frozen plan.")
    contract = str(manifest.get("execution_contract_sha256", ""))
    if len(contract) != 64 or any(char not in "0123456789abcdef" for char in contract):
        raise ValueError("Execution manifest has no valid execution-contract digest.")
    jobs = list(manifest.get("jobs") or [])
    mapped = {
        (int(job["seed"]), int(job["fold_index"]), str(job["role"])): job
        for job in jobs
    }
    expected = {
        (seed, fold, role)
        for seed in seeds
        for fold in range(n_folds)
        for role in ("A0_eeg_only", "A2_structured_condition_prompt")
    }
    if len(mapped) != len(jobs) or set(mapped) != expected:
        raise ValueError("Execution manifest does not match the exact analysis grid.")
    return mapped, contract


def _required_sha256(value: object, name: str) -> str:
    digest = str(value or "")
    if len(digest) != 64 or any(
        character not in "0123456789abcdef" for character in digest
    ):
        raise ValueError(f"Confirmatory {name} has no valid SHA-256 digest.")
    return digest


def _prediction_rows(frame: pd.DataFrame, name: str) -> pd.DataFrame:
    required = ["sample_id", "label", "prediction"]
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise ValueError(f"{name} predictions missing columns: {missing}")
    result = frame[required].copy()
    result["sample_id"] = result["sample_id"].astype(str)
    if result["sample_id"].duplicated().any():
        raise ValueError(f"{name} predictions contain duplicate sample IDs.")
    for column in ("label", "prediction"):
        numeric = pd.to_numeric(result[column], errors="coerce")
        if numeric.isna().any() or not np.equal(numeric, np.floor(numeric)).all():
            raise ValueError(f"{name} {column} values must be finite integers.")
        result[column] = numeric.astype(int)
    return result


def _group_pair_metrics(
    paired: pd.DataFrame,
    group_columns: list[str],
    *,
    seed: int,
    fold: int,
    expected_n_labels: int,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    grouper = group_columns[0] if len(group_columns) == 1 else group_columns
    for keys, group in paired.groupby(grouper, dropna=False, sort=True):
        key_values = (keys,) if len(group_columns) == 1 else keys
        n_labels = int(group["baseline_label"].nunique())
        if n_labels != expected_n_labels:
            raise ValueError(
                f"Confirmatory group {dict(zip(group_columns, key_values))} has "
                f"{n_labels} labels, expected {expected_n_labels}."
            )
        baseline_score = balanced_accuracy(
            group["baseline_label"].to_numpy(),
            group["baseline_prediction"].to_numpy(),
            n_classes=expected_n_labels,
        )
        candidate_score = balanced_accuracy(
            group["candidate_label"].to_numpy(),
            group["candidate_prediction"].to_numpy(),
            n_classes=expected_n_labels,
        )
        rows.append(
            {
                **dict(zip(group_columns, key_values)),
                "optimization_seed": seed,
                "fold_index": fold,
                "n_samples": len(group),
                "n_labels": n_labels,
                "baseline_balanced_accuracy": baseline_score,
                "candidate_balanced_accuracy": candidate_score,
                "balanced_accuracy_delta": candidate_score - baseline_score,
            }
        )
    return rows


def _validate_cross_pair_contracts(
    contracts: dict[tuple[int, int], dict[str, object]],
    seeds: tuple[int, ...],
    n_folds: int,
) -> None:
    global_equal = (
        "prediction_schema_version",
        "primary_fairness_hash",
        "scenario",
        "selection_split",
        "metadata_contract_version",
        "split_seed",
        "n_folds",
        "parameter_schema_sha256",
        "vocabulary_sha256",
        "asset_provenance_sha256",
        "execution_phase",
        "execution_manifest_content_sha256",
        "lockbox_reveal_receipt_sha256",
        "analysis_plan_sha256",
        "cohort_roles_sha256",
        "cohort_sha256",
        "outer_test_access_during_training",
        "source_commit_sha",
        "source_dirty",
        "source_tree_sha256",
        "environment_sha256",
        "inference_environment_sha256",
        "checkpoint_role",
        "checkpoint_selection_split",
        "checkpoint_selection_metric",
        "analysis_manifest_schema",
    )
    reference = contracts[(seeds[0], 0)]
    mismatched = sorted(
        {
            field
            for contract in contracts.values()
            for field in global_equal
            if contract[field] != reference[field]
        }
    )
    if mismatched:
        raise ValueError(f"Confirmatory runs differ in global contracts: {mismatched}.")
    for fold in range(n_folds):
        fold_contracts = [contracts[(seed, fold)] for seed in seeds]
        for field in (
            "sample_identity_sha256",
            "analysis_manifest_sha256",
            "split_assignment_sha256",
            "split_manifest_sha256",
        ):
            if len({contract[field] for contract in fold_contracts}) != 1:
                raise ValueError(f"Fold {fold} differs across seeds in {field}.")
    for seed in seeds:
        seed_contracts = [contracts[(seed, fold)] for fold in range(n_folds)]
        if len({contract["initial_trainable_state_sha256"] for contract in seed_contracts}) != 1:
            raise ValueError(f"Seed {seed} has inconsistent initial model states across folds.")
    initial_states = {contracts[(seed, 0)]["initial_trainable_state_sha256"] for seed in seeds}
    if len(initial_states) != len(seeds):
        raise ValueError("Distinct optimization seeds produced identical initial model states.")


def _validate_outer_test_coverage(
    samples: dict[tuple[int, int], set[str]],
    manifest: pd.DataFrame,
    seeds: tuple[int, ...],
    n_folds: int,
) -> None:
    expected = set(manifest["sample_id"].astype(str))
    reference_fold_samples: dict[int, set[str]] = {}
    for seed in seeds:
        observed: set[str] = set()
        for fold in range(n_folds):
            current = samples[(seed, fold)]
            overlap = observed & current
            if overlap:
                raise ValueError(
                    f"Outer test folds overlap for seed {seed}: {sorted(overlap)[:5]}."
                )
            observed.update(current)
            if fold not in reference_fold_samples:
                reference_fold_samples[fold] = current
            elif current != reference_fold_samples[fold]:
                raise ValueError(f"Outer fold {fold} sample set changes across seeds.")
        if observed != expected:
            raise ValueError(f"Outer folds for seed {seed} do not cover the full manifest exactly.")


def _average_subject_seeds(table: pd.DataFrame, group_columns: list[str]) -> pd.DataFrame:
    grouped = table.groupby(group_columns, dropna=False, sort=True)
    averaged = grouped.agg(
        n_seeds=("optimization_seed", "nunique"),
        fold_index=("fold_index", "first"),
        n_samples_per_seed=("n_samples", "first"),
        baseline_balanced_accuracy=("baseline_balanced_accuracy", "mean"),
        candidate_balanced_accuracy=("candidate_balanced_accuracy", "mean"),
        seed_delta_std=("balanced_accuracy_delta", "std"),
        seed_delta_min=("balanced_accuracy_delta", "min"),
        seed_delta_max=("balanced_accuracy_delta", "max"),
    ).reset_index()
    averaged["balanced_accuracy_delta"] = (
        averaged["candidate_balanced_accuracy"] - averaged["baseline_balanced_accuracy"]
    )
    return averaged


def _transition(baseline_success: bool, candidate_success: bool) -> str:
    if baseline_success and candidate_success:
        return "retained"
    if not baseline_success and candidate_success:
        return "learned"
    if baseline_success and not candidate_success:
        return "forgotten"
    return "unresolved"
