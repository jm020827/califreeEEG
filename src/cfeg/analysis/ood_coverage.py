from __future__ import annotations

import itertools
from collections.abc import Mapping, Sequence

import numpy as np
import pandas as pd

from cfeg.metrics import balanced_accuracy

REQUIRED_PREDICTION_COLUMNS = ("sample_id", "label", "prediction")


def compare_ood_coverage(
    baseline_predictions: pd.DataFrame,
    candidate_predictions: pd.DataFrame,
    manifest: pd.DataFrame,
    *,
    cell_columns: Sequence[str],
    success_threshold: float,
    expected_n_labels: int | None = None,
    tags: Mapping[str, object] | None = None,
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

    baseline = _validate_predictions(baseline_predictions, "baseline")
    candidate = _validate_predictions(candidate_predictions, "candidate")
    metadata = _validate_manifest(manifest)
    tag_values = dict(tags or {})

    reserved_tags = (
        set(metadata.columns)
        | set(baseline_predictions.columns)
        | set(candidate_predictions.columns)
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
    metadata_columns = list(
        dict.fromkeys(["sample_id", *cell_columns, *subject_columns])
    )
    paired = paired.merge(metadata[metadata_columns], on="sample_id", validate="one_to_one")
    if len(paired) != len(baseline):
        missing = sorted(baseline_ids - set(paired["sample_id"]))[:5]
        raise ValueError(f"Manifest has no metadata for prediction samples: {missing}")
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
        "baseline_coverage": float(valid_cells["baseline_success"].mean()),
        "candidate_coverage": float(valid_cells["candidate_success"].mean()),
        "coverage_delta": float(
            valid_cells["candidate_success"].mean()
            - valid_cells["baseline_success"].mean()
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
                    _transition(base_success, candidate_success)
                    if complete
                    else "insufficient"
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
