from __future__ import annotations

import hashlib
import itertools
from collections.abc import Mapping, Sequence

import numpy as np
import pandas as pd

from cfeg.analysis.provenance import (
    PRIMARY_ANALYSIS_MANIFEST_COLUMNS,
    PRIMARY_ANALYSIS_MANIFEST_SCHEMA,
    VerifiedPredictionBundle,
    primary_analysis_manifest_sha256,
)
from cfeg.metrics import balanced_accuracy

REQUIRED_PREDICTION_COLUMNS = ("sample_id", "label", "prediction")
PRIMARY_PROVENANCE_COLUMNS = (
    "prediction_schema_version",
    "scenario",
    "selection_split",
    "sample_identity_sha256",
    "checkpoint_sha256",
    "execution_job_id",
    "execution_contract_sha256",
    "execution_manifest_content_sha256",
    "lockbox_reveal_receipt_sha256",
    "planned_config_sha256",
    "training_completion_sha256",
    "primary_ablation_role",
    "primary_fairness_hash",
    "metadata_contract_version",
    "conditioning_architecture",
    "external_metadata_mode",
    "optimization_seed",
    "split_seed",
    "n_folds",
    "fold_index",
    "parameter_schema_sha256",
    "initial_trainable_state_sha256",
    "split_assignment_sha256",
    "vocabulary_sha256",
    "asset_provenance_sha256",
    "execution_phase",
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
    "analysis_manifest_sha256",
    "split_manifest_sha256",
)
PRIMARY_EQUAL_COLUMNS = (
    "prediction_schema_version",
    "scenario",
    "selection_split",
    "sample_identity_sha256",
    "execution_contract_sha256",
    "execution_manifest_content_sha256",
    "lockbox_reveal_receipt_sha256",
    "primary_fairness_hash",
    "metadata_contract_version",
    "conditioning_architecture",
    "optimization_seed",
    "split_seed",
    "n_folds",
    "fold_index",
    "parameter_schema_sha256",
    "initial_trainable_state_sha256",
    "split_assignment_sha256",
    "vocabulary_sha256",
    "asset_provenance_sha256",
    "execution_phase",
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
    "analysis_manifest_sha256",
    "split_manifest_sha256",
)


def compare_ood_coverage(
    baseline_predictions: pd.DataFrame | VerifiedPredictionBundle,
    candidate_predictions: pd.DataFrame | VerifiedPredictionBundle,
    manifest: pd.DataFrame,
    *,
    cell_columns: Sequence[str],
    success_threshold: float,
    expected_n_labels: int | None = None,
    tags: Mapping[str, object] | None = None,
    require_primary_contract: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object]]:
    """Compare two models on exactly paired samples.

    A cell is ``learned`` when only the candidate reaches ``success_threshold``
    and ``forgotten`` when only the baseline reaches it. Balanced accuracy is
    computed independently inside every cell; labels absent from a cell are not
    included in that cell's macro recall.
    """
    if not 0.0 <= success_threshold <= 1.0:
        raise ValueError("success_threshold must be within [0, 1].")
    if not cell_columns:
        raise ValueError("cell_columns must contain at least one column.")
    if expected_n_labels is not None and expected_n_labels < 2:
        raise ValueError("expected_n_labels must be at least 2 when provided.")
    if require_primary_contract and not (
        isinstance(baseline_predictions, VerifiedPredictionBundle)
        and isinstance(candidate_predictions, VerifiedPredictionBundle)
    ):
        raise ValueError("Primary comparison requires VerifiedPredictionBundle inputs.")
    baseline_frame = (
        baseline_predictions.frame
        if isinstance(baseline_predictions, VerifiedPredictionBundle)
        else baseline_predictions
    )
    candidate_frame = (
        candidate_predictions.frame
        if isinstance(candidate_predictions, VerifiedPredictionBundle)
        else candidate_predictions
    )

    baseline = _validate_predictions(baseline_frame, "baseline")
    candidate = _validate_predictions(candidate_frame, "candidate")
    primary_contract = (
        validate_primary_contract_pair(baseline_frame, candidate_frame)
        if require_primary_contract
        else None
    )
    metadata = _validate_manifest(manifest)
    tag_values = dict(tags or {})

    reserved_tags = (
        set(metadata.columns)
        | set(baseline_frame.columns)
        | set(candidate_frame.columns)
        | {
            "baseline_balanced_accuracy",
            "candidate_balanced_accuracy",
            "balanced_accuracy_delta",
            "transition",
        }
    )
    overlap = set(tag_values) & reserved_tags
    if overlap:
        raise ValueError(
            "Tag names may not overwrite prediction, manifest, or metric columns: "
            f"{sorted(overlap)}"
        )
    missing_cells = [column for column in cell_columns if column not in metadata.columns]
    if missing_cells:
        raise ValueError(f"Manifest missing cell columns: {missing_cells}")
    if "subject_id" not in metadata.columns:
        raise ValueError("Manifest missing subject_id required for paired inference.")

    baseline_ids = set(baseline["sample_id"])
    candidate_ids = set(candidate["sample_id"])
    if baseline_ids != candidate_ids:
        only_baseline = sorted(baseline_ids - candidate_ids)[:5]
        only_candidate = sorted(candidate_ids - baseline_ids)[:5]
        raise ValueError(
            "Prediction sample sets differ; comparisons must be paired exactly. "
            f"Only baseline={only_baseline}, only candidate={only_candidate}"
        )
    if require_primary_contract:
        unsupported_cells = sorted(set(cell_columns) - set(PRIMARY_ANALYSIS_MANIFEST_COLUMNS))
        if unsupported_cells:
            raise ValueError(
                "Primary cell columns are not bound by the analysis manifest contract: "
                f"{unsupported_cells}."
            )
        observed_manifest_hash = primary_analysis_manifest_sha256(metadata, baseline_ids)
        if primary_contract["analysis_manifest_sha256"] != observed_manifest_hash:
            raise ValueError(
                "Supplied analysis manifest does not match the prediction artifact contract."
            )

    paired = baseline.rename(
        columns={"label": "baseline_label", "prediction": "baseline_prediction"}
    ).merge(
        candidate.rename(
            columns={"label": "candidate_label", "prediction": "candidate_prediction"}
        ),
        on="sample_id",
        validate="one_to_one",
    )
    if not np.array_equal(
        paired["baseline_label"].to_numpy(), paired["candidate_label"].to_numpy()
    ):
        bad = paired.loc[
            paired["baseline_label"] != paired["candidate_label"], "sample_id"
        ].tolist()[:5]
        raise ValueError(f"Prediction labels disagree for samples: {bad}")
    subject_columns = ["subject_id"]
    if "dataset_id" in metadata.columns:
        subject_columns.insert(0, "dataset_id")
    metadata_columns = list(dict.fromkeys(["sample_id", *cell_columns, *subject_columns]))
    paired = paired.merge(metadata[metadata_columns], on="sample_id", validate="one_to_one")
    if len(paired) != len(baseline):
        missing = sorted(baseline_ids - set(paired["sample_id"]))[:5]
        raise ValueError(f"Manifest has no metadata for prediction samples: {missing}")
    if "label" in metadata.columns:
        manifest_labels = metadata.set_index("sample_id")["label"]
        expected_labels = paired["sample_id"].map(manifest_labels).astype(int).to_numpy()
        if not np.array_equal(paired["baseline_label"].astype(int).to_numpy(), expected_labels):
            raise ValueError("Prediction labels do not match the supplied manifest labels.")
    for name, value in tag_values.items():
        paired[name] = value

    group_columns = [*cell_columns, *tag_values]
    cells = _group_metrics(
        paired, group_columns, success_threshold, expected_n_labels=expected_n_labels
    )
    subjects = _group_metrics(
        paired, subject_columns, success_threshold, expected_n_labels=expected_n_labels
    )
    valid_cells = cells.loc[cells["transition"] != "insufficient"]
    valid_subjects = subjects.loc[subjects["transition"] != "insufficient"]
    if valid_cells.empty or valid_subjects.empty:
        raise ValueError(
            "No complete OOD cells/subjects remain after the expected label-count gate."
        )
    inference = paired_subject_inference(valid_subjects)
    summary: dict[str, object] = {
        "success_threshold": float(success_threshold),
        "n_paired_samples": len(paired),
        "n_cells": len(cells),
        "n_valid_cells": len(valid_cells),
        "insufficient_cells": int((cells["transition"] == "insufficient").sum()),
        "expected_n_labels": expected_n_labels,
        "primary_contract": primary_contract,
        "baseline_coverage": float(valid_cells["baseline_success"].mean()),
        "candidate_coverage": float(valid_cells["candidate_success"].mean()),
        "coverage_delta": float(
            valid_cells["candidate_success"].mean() - valid_cells["baseline_success"].mean()
        ),
        "learned_cells": int((valid_cells["transition"] == "learned").sum()),
        "forgotten_cells": int((valid_cells["transition"] == "forgotten").sum()),
        "learned_minus_forgotten": int(
            (valid_cells["transition"] == "learned").sum()
            - (valid_cells["transition"] == "forgotten").sum()
        ),
        "baseline_worst_cell_balanced_accuracy": float(
            valid_cells["baseline_balanced_accuracy"].min()
        ),
        "candidate_worst_cell_balanced_accuracy": float(
            valid_cells["candidate_balanced_accuracy"].min()
        ),
        **inference,
    }
    return cells, subjects, summary


def paired_subject_inference(
    subjects: pd.DataFrame,
    *,
    seed: int = 42,
    n_resamples: int = 10_000,
) -> dict[str, object]:
    """Return paired subject effect, bootstrap CI, and sign-flip p-value."""
    required = {"subject_id", "balanced_accuracy_delta"}
    missing = required - set(subjects.columns)
    if missing:
        raise ValueError(f"Subject table missing columns: {sorted(missing)}")
    differences = subjects["balanced_accuracy_delta"].to_numpy(dtype=float)
    if not len(differences):
        raise ValueError("Subject table is empty.")
    if n_resamples < 1:
        raise ValueError("n_resamples must be positive.")
    rng = np.random.default_rng(seed)
    draws = rng.choice(differences, size=(n_resamples, len(differences)), replace=True)
    ci_low, ci_high = np.quantile(draws.mean(axis=1), [0.025, 0.975])
    observed = abs(float(differences.mean()))
    if len(differences) <= 16:
        signs = np.asarray(list(itertools.product((-1.0, 1.0), repeat=len(differences))))
    else:
        signs = rng.choice((-1.0, 1.0), size=(n_resamples, len(differences)))
    null_effects = np.abs((signs * differences).mean(axis=1))
    p_value = float((np.count_nonzero(null_effects >= observed) + 1) / (len(null_effects) + 1))
    return {
        "n_subjects": len(differences),
        "mean_subject_balanced_accuracy_delta": float(differences.mean()),
        "median_subject_balanced_accuracy_delta": float(np.median(differences)),
        "subject_delta_ci95_low": float(ci_low),
        "subject_delta_ci95_high": float(ci_high),
        "paired_sign_flip_p_value": p_value,
        "inference_seed": int(seed),
        "inference_resamples": int(n_resamples),
    }


def _validate_predictions(frame: pd.DataFrame, name: str) -> pd.DataFrame:
    missing = [column for column in REQUIRED_PREDICTION_COLUMNS if column not in frame.columns]
    if missing:
        raise ValueError(f"{name} predictions missing columns: {missing}")
    result = frame.loc[:, REQUIRED_PREDICTION_COLUMNS].copy()
    result["sample_id"] = result["sample_id"].astype(str)
    if result["sample_id"].duplicated().any():
        duplicates = result.loc[result["sample_id"].duplicated(), "sample_id"].tolist()[:5]
        raise ValueError(f"{name} predictions contain duplicate sample IDs: {duplicates}")
    return result


def validate_primary_contract_pair(
    baseline: pd.DataFrame, candidate: pd.DataFrame
) -> dict[str, object]:
    baseline_contract = _constant_contract(baseline, "baseline")
    candidate_contract = _constant_contract(candidate, "candidate")
    if (
        baseline_contract["primary_ablation_role"] != "A0_eeg_only"
        or baseline_contract["external_metadata_mode"] != "null"
    ):
        raise ValueError("Primary baseline must be A0_eeg_only with external metadata null.")
    if (
        candidate_contract["primary_ablation_role"] != "A2_structured_condition_prompt"
        or candidate_contract["external_metadata_mode"] != "observed"
    ):
        raise ValueError(
            "Primary candidate must be A2_structured_condition_prompt with external metadata observed."
        )
    if baseline_contract["selection_split"] != "test":
        raise ValueError("Primary comparison requires checkpoint-held-out selection_split=test.")
    if baseline_contract["metadata_contract_version"] != "0.4-dev":
        raise ValueError("Primary comparison requires metadata_contract_version=0.4-dev.")
    if baseline_contract["conditioning_architecture"] != "physical_hybrid_v1":
        raise ValueError("Primary comparison requires physical_hybrid_v1 conditioning.")
    if baseline_contract["execution_phase"] != "confirmatory_training":
        raise ValueError("Primary comparison requires confirmatory_training checkpoints.")
    if bool(baseline_contract["outer_test_access_during_training"]):
        raise ValueError("Primary training must not open outer-test performance.")
    if baseline_contract["prediction_schema_version"] != "cfeg.predictions.v3":
        raise ValueError("Primary comparison requires cfeg.predictions.v3 artifacts.")
    if (
        baseline_contract["checkpoint_role"] != "confirmatory_fixed_epoch"
        or baseline_contract["checkpoint_selection_split"] != "none"
    ):
        raise ValueError(
            "Primary comparison requires the pre-registered fixed-epoch lockbox checkpoint."
        )
    if baseline_contract["checkpoint_selection_metric"] != "fixed_epoch":
        raise ValueError("Primary checkpoint selection metric must be fixed_epoch.")
    if baseline_contract["analysis_manifest_schema"] != PRIMARY_ANALYSIS_MANIFEST_SCHEMA:
        raise ValueError("Primary analysis manifest schema is unsupported.")
    if bool(baseline_contract["source_dirty"]):
        raise ValueError("Primary comparison requires artifacts trained from a clean Git tree.")

    mismatched = [
        column
        for column in PRIMARY_EQUAL_COLUMNS
        if baseline_contract[column] != candidate_contract[column]
    ]
    if mismatched:
        raise ValueError(
            f"Primary prediction contracts differ outside treatment access: {mismatched}."
        )
    if baseline_contract["checkpoint_sha256"] == candidate_contract["checkpoint_sha256"]:
        raise ValueError("Primary A0/A2 predictions unexpectedly use the same checkpoint.")

    for name, frame, contract in (
        ("baseline", baseline, baseline_contract),
        ("candidate", candidate, candidate_contract),
    ):
        expected_identity = _sample_identity_sha256(frame["sample_id"].astype(str))
        if contract["sample_identity_sha256"] != expected_identity:
            raise ValueError(f"{name} sample_identity_sha256 does not match its sample IDs.")
        for column in (
            "checkpoint_sha256",
            "execution_contract_sha256",
            "execution_manifest_content_sha256",
            "lockbox_reveal_receipt_sha256",
            "planned_config_sha256",
            "training_completion_sha256",
            "primary_fairness_hash",
            "parameter_schema_sha256",
            "initial_trainable_state_sha256",
            "split_assignment_sha256",
            "vocabulary_sha256",
            "asset_provenance_sha256",
            "analysis_plan_sha256",
            "cohort_roles_sha256",
            "cohort_sha256",
            "source_tree_sha256",
            "environment_sha256",
            "inference_environment_sha256",
            "analysis_manifest_sha256",
            "split_manifest_sha256",
        ):
            if not _is_sha256(contract[column]):
                raise ValueError(f"{name} {column} is not a valid SHA-256 digest.")
        commit = str(contract["source_commit_sha"])
        if len(commit) not in {40, 64} or any(
            char not in "0123456789abcdef" for char in commit.lower()
        ):
            raise ValueError(f"{name} source_commit_sha is not a valid Git object ID.")
    return {
        **{column: baseline_contract[column] for column in PRIMARY_EQUAL_COLUMNS},
        "baseline_checkpoint_sha256": baseline_contract["checkpoint_sha256"],
        "candidate_checkpoint_sha256": candidate_contract["checkpoint_sha256"],
        "baseline_execution_job_id": baseline_contract["execution_job_id"],
        "candidate_execution_job_id": candidate_contract["execution_job_id"],
        "baseline_planned_config_sha256": baseline_contract["planned_config_sha256"],
        "candidate_planned_config_sha256": candidate_contract["planned_config_sha256"],
        "baseline_training_completion_sha256": baseline_contract[
            "training_completion_sha256"
        ],
        "candidate_training_completion_sha256": candidate_contract[
            "training_completion_sha256"
        ],
        "baseline_role": baseline_contract["primary_ablation_role"],
        "candidate_role": candidate_contract["primary_ablation_role"],
    }


def _constant_contract(frame: pd.DataFrame, name: str) -> dict[str, object]:
    missing = [column for column in PRIMARY_PROVENANCE_COLUMNS if column not in frame.columns]
    if missing:
        raise ValueError(f"{name} predictions missing primary provenance columns: {missing}")
    contract: dict[str, object] = {}
    for column in PRIMARY_PROVENANCE_COLUMNS:
        values = frame[column]
        if values.isna().any() or values.nunique(dropna=False) != 1:
            raise ValueError(
                f"{name} prediction provenance {column} must be one non-missing constant."
            )
        value = values.iloc[0]
        value = value.item() if isinstance(value, np.generic) else value
        if column == "source_dirty":
            normalized = str(value).strip().lower()
            if normalized not in {"true", "false"}:
                raise ValueError(f"{name} source_dirty must be a boolean constant.")
            value = normalized == "true"
        contract[column] = value
    return contract


def _sample_identity_sha256(sample_ids: pd.Series) -> str:
    digest = hashlib.sha256()
    for sample_id in sorted(sample_ids.astype(str)):
        encoded = sample_id.encode("utf-8")
        digest.update(len(encoded).to_bytes(8, byteorder="big", signed=False))
        digest.update(encoded)
    return digest.hexdigest()


def _is_sha256(value: object) -> bool:
    text = str(value)
    return len(text) == 64 and all(char in "0123456789abcdef" for char in text.lower())


def _validate_manifest(manifest: pd.DataFrame) -> pd.DataFrame:
    if "sample_id" not in manifest.columns:
        raise ValueError("Manifest missing sample_id.")
    result = manifest.copy()
    result["sample_id"] = result["sample_id"].astype(str)
    if result["sample_id"].duplicated().any():
        raise ValueError("Manifest sample_id values must be unique.")
    return result


def _group_metrics(
    paired: pd.DataFrame,
    group_columns: Sequence[str],
    threshold: float,
    *,
    expected_n_labels: int | None = None,
) -> pd.DataFrame:
    if "subject_id" not in paired.columns:
        raise ValueError("subject_id is required for subject-level inference.")
    rows: list[dict[str, object]] = []
    grouper = group_columns[0] if len(group_columns) == 1 else list(group_columns)
    for keys, group in paired.groupby(grouper, dropna=False, sort=True):
        key_values = (keys,) if len(group_columns) == 1 else keys
        base_score = balanced_accuracy(
            group["baseline_label"].to_numpy(), group["baseline_prediction"].to_numpy()
        )
        candidate_score = balanced_accuracy(
            group["candidate_label"].to_numpy(), group["candidate_prediction"].to_numpy()
        )
        n_labels = int(group["baseline_label"].nunique())
        complete = expected_n_labels is None or n_labels == expected_n_labels
        base_success = bool(base_score >= threshold) if complete else False
        candidate_success = bool(candidate_score >= threshold) if complete else False
        rows.append(
            {
                **dict(zip(group_columns, key_values)),
                "n_samples": len(group),
                "n_labels": n_labels,
                "baseline_balanced_accuracy": base_score,
                "candidate_balanced_accuracy": candidate_score,
                "balanced_accuracy_delta": candidate_score - base_score,
                "baseline_success": base_success,
                "candidate_success": candidate_success,
                "transition": (
                    _transition(base_success, candidate_success) if complete else "insufficient"
                ),
            }
        )
    return pd.DataFrame(rows)


def _transition(baseline_success: bool, candidate_success: bool) -> str:
    if baseline_success and candidate_success:
        return "retained"
    if not baseline_success and candidate_success:
        return "learned"
    if baseline_success and not candidate_success:
        return "forgotten"
    return "unresolved"
