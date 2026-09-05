from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from cfeg.analysis.metadata_calibration_efficiency import (
    MetadataCalibrationBundleBindings,
    MetadataCalibrationBundleSpec,
)
from cfeg.metadata_calibration_atomic import validate_baseline_private_staging


@dataclass(frozen=True)
class BaselineAnalysisBundle:
    sample_predictions: pd.DataFrame
    class_accuracy: pd.DataFrame
    condition_balanced_accuracy: pd.DataFrame
    participant_balanced_accuracy: pd.DataFrame
    summary: pd.DataFrame
    metadata: dict[str, object]


def reduce_metadata_calibration_baseline_scores(
    scores: pd.DataFrame,
    *,
    expected_query_with_label: pd.DataFrame,
    expected_support: pd.DataFrame,
    expected_source_pools: dict[tuple[str, str], str],
    bindings: MetadataCalibrationBundleBindings,
    spec: MetadataCalibrationBundleSpec,
) -> BaselineAnalysisBundle:
    expected_unlabeled = expected_query_with_label.drop(columns="label")
    validate_baseline_private_staging(
        scores,
        expected_query_unlabeled=expected_unlabeled,
        expected_support=expected_support,
        expected_source_pools=expected_source_pools,
        bindings=bindings,
        spec=spec,
    )
    labels = expected_query_with_label.loc[:, ["query_token", "label"]].copy()
    if labels["query_token"].astype(str).duplicated().any():
        raise ValueError("Baseline query-label join requires unique opaque tokens.")
    numeric_labels = pd.to_numeric(labels["label"], errors="raise").to_numpy()
    if not np.equal(numeric_labels, np.floor(numeric_labels)).all() or (
        (numeric_labels < 0) | (numeric_labels >= spec.n_classes)
    ).any():
        raise ValueError("Baseline query labels are invalid.")
    score_columns = [f"score_{label:03d}" for label in range(spec.n_classes)]
    sample = scores.merge(labels, on="query_token", how="left", validate="many_to_one")
    if sample["label"].isna().any():
        raise ValueError("Baseline score has no finalizer-side query label.")
    sample["label"] = sample["label"].astype(int)
    sample["prediction"] = np.argmax(
        sample.loc[:, score_columns].to_numpy(dtype=np.float64),
        axis=1,
    )
    sample["correct"] = sample["prediction"].eq(sample["label"]).astype(float)
    sample_columns = [
        "adapter_id",
        "budget",
        "query_token",
        "subject_id",
        "electrode_type",
        "label",
        "prediction",
        "correct",
    ]
    sample = sample.loc[:, sample_columns].sort_values(
        ["adapter_id", "budget", "subject_id", "electrode_type", "query_token"],
        kind="mergesort",
    )
    class_accuracy = (
        sample.groupby(
            ["adapter_id", "budget", "subject_id", "electrode_type", "label"],
            as_index=False,
            sort=False,
        )
        .agg(n_trials=("correct", "size"), class_accuracy=("correct", "mean"))
        .sort_values(
            ["adapter_id", "budget", "subject_id", "electrode_type", "label"],
            kind="mergesort",
        )
    )
    if not class_accuracy["n_trials"].eq(5).all():
        raise ValueError("Every baseline class accuracy must contain five query trials.")
    class_counts = class_accuracy.groupby(
        ["adapter_id", "budget", "subject_id", "electrode_type"], sort=False
    )["label"].nunique()
    if not class_counts.eq(spec.n_classes).all():
        raise ValueError("Every baseline cell must contain the full class vocabulary.")
    condition = (
        class_accuracy.groupby(
            ["adapter_id", "budget", "subject_id", "electrode_type"],
            as_index=False,
            sort=False,
        )["class_accuracy"]
        .mean()
        .rename(columns={"class_accuracy": "balanced_accuracy"})
    )
    condition_counts = condition.groupby(
        ["adapter_id", "budget", "subject_id"], sort=False
    )["electrode_type"].nunique()
    if not condition_counts.eq(2).all():
        raise ValueError("Every baseline participant must contain dry and wet conditions.")
    participant = (
        condition.groupby(
            ["adapter_id", "budget", "subject_id"],
            as_index=False,
            sort=False,
        )["balanced_accuracy"]
        .mean()
        .rename(columns={"balanced_accuracy": "participant_balanced_accuracy"})
        .sort_values(["adapter_id", "budget", "subject_id"], kind="mergesort")
        .reset_index(drop=True)
    )
    summary = summarize_baseline_participant_accuracy(participant)
    return BaselineAnalysisBundle(
        sample_predictions=sample.reset_index(drop=True),
        class_accuracy=class_accuracy.reset_index(drop=True),
        condition_balanced_accuracy=condition.reset_index(drop=True),
        participant_balanced_accuracy=participant,
        summary=summary,
        metadata={
            "schema": "cfeg.metadata-calibration-baseline-analysis-bundle.v1",
            "candidate_id": spec.candidate_id,
            "phase": spec.phase,
            "bindings": bindings.as_dict(),
            "score_interpretation": "uncalibrated_raw_class_score_argmax_only",
            "seed_reduction": "none_deterministic_once",
            "tie_break": "lowest_zero_based_class_id",
            "primary_inference_role": "descriptive_resource_matched_context_only",
        },
    )


def summarize_baseline_participant_accuracy(participant: pd.DataFrame) -> pd.DataFrame:
    """Apply the one canonical descriptive reduction used by producer and verifier."""

    required = {
        "adapter_id",
        "budget",
        "subject_id",
        "participant_balanced_accuracy",
    }
    if set(participant) != required or participant.empty:
        raise ValueError("Baseline participant aggregate has a wrong exact schema.")
    ordered = participant.sort_values(
        ["adapter_id", "budget", "subject_id"], kind="mergesort"
    ).reset_index(drop=True)
    return (
        ordered.groupby(["adapter_id", "budget"], as_index=False, sort=False)[
            "participant_balanced_accuracy"
        ]
        .agg(
            n_participants="size",
            mean="mean",
            sd="std",
            median="median",
            q25=lambda values: values.quantile(0.25),
            q75=lambda values: values.quantile(0.75),
            minimum="min",
            maximum="max",
        )
        .reset_index(drop=True)
    )
