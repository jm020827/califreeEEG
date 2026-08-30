from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from pathlib import Path

import pandas as pd

from cfeg.metrics import balanced_accuracy

DEVELOPMENT_ROLES = ("A0_eeg_only", "A2_structured_condition_prompt")


def aggregate_complete_development_loso(
    jobs: Sequence[dict],
    manifest: pd.DataFrame,
    *,
    expected_subjects: Sequence[str],
    expected_n_samples: int,
) -> tuple[pd.DataFrame, dict[str, object]]:
    """Reveal descriptive S1-S3 outcomes only after the six-artifact grid is complete."""

    expected_keys = {(42, fold, role) for fold in range(3) for role in DEVELOPMENT_ROLES}
    observed_keys = {
        (int(job["seed"]), int(job["fold_index"]), str(job["role"])) for job in jobs
    }
    if len(jobs) != 6 or observed_keys != expected_keys:
        raise ValueError("Development LOSO grid must contain the exact six A0/A2 fold jobs.")
    metadata = manifest.loc[
        manifest["subject_id"].astype(str).isin({str(value) for value in expected_subjects})
    ].copy()
    metadata["sample_id"] = metadata["sample_id"].astype(str)
    if len(metadata) != expected_n_samples or metadata["sample_id"].duplicated().any():
        raise ValueError("Development manifest does not match the frozen S1-S3 cohort.")

    frames: dict[tuple[int, str], pd.DataFrame] = {}
    coverage: dict[str, set[str]] = {role: set() for role in DEVELOPMENT_ROLES}
    held_subjects: dict[str, list[str]] = {role: [] for role in DEVELOPMENT_ROLES}
    for job in jobs:
        path = Path(job["prediction_csv"])
        provenance_path = path.with_name(f"{path.stem}_provenance.json")
        if not path.is_file() or not provenance_path.is_file():
            raise ValueError("Development prediction grid is incomplete; outcomes remain sealed.")
        provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
        if provenance.get("prediction_csv_sha256") != _sha256_file(path):
            raise ValueError(f"Development prediction artifact changed after creation: {path}.")
        frame = pd.read_csv(path)
        required = {
            "sample_id",
            "label",
            "prediction",
            "selection_split",
            "primary_ablation_role",
            "optimization_seed",
            "fold_index",
            "checkpoint_role",
            "checkpoint_selection_metric",
        }
        missing = required - set(frame.columns)
        if missing:
            raise ValueError(f"Development prediction is missing columns: {sorted(missing)}.")
        role = str(job["role"])
        fold = int(job["fold_index"])
        if (
            set(frame["selection_split"].astype(str)) != {"val"}
            or set(frame["primary_ablation_role"].astype(str)) != {role}
            or set(frame["optimization_seed"].astype(int)) != {42}
            or set(frame["fold_index"].astype(int)) != {fold}
            or set(frame["checkpoint_role"].astype(str)) != {"development_fixed_epoch"}
            or set(frame["checkpoint_selection_metric"].astype(str)) != {"fixed_epoch"}
        ):
            raise ValueError(f"Development prediction contract mismatch for fold={fold}, role={role}.")
        frame["sample_id"] = frame["sample_id"].astype(str)
        if frame["sample_id"].duplicated().any() or coverage[role] & set(frame["sample_id"]):
            raise ValueError(f"Development held-out samples overlap for role={role}.")
        joined = frame.merge(
            metadata[["sample_id", "subject_id"]], on="sample_id", validate="one_to_one"
        )
        if len(joined) != len(frame) or joined["subject_id"].nunique() != 1:
            raise ValueError("Each development fold must contain exactly one held-out subject.")
        coverage[role].update(frame["sample_id"])
        held_subjects[role].append(str(joined["subject_id"].iloc[0]))
        frames[(fold, role)] = joined

    expected_ids = set(metadata["sample_id"])
    for role in DEVELOPMENT_ROLES:
        if coverage[role] != expected_ids or sorted(held_subjects[role]) != sorted(
            str(value) for value in expected_subjects
        ):
            raise ValueError(f"Development LOSO coverage is incomplete for role={role}.")

    rows = []
    for fold in range(3):
        baseline = frames[(fold, DEVELOPMENT_ROLES[0])]
        candidate = frames[(fold, DEVELOPMENT_ROLES[1])]
        paired = baseline[["sample_id", "subject_id", "label", "prediction"]].merge(
            candidate[["sample_id", "label", "prediction"]],
            on="sample_id",
            suffixes=("_a0", "_a2"),
            validate="one_to_one",
        )
        if len(paired) != len(baseline) or not paired["label_a0"].equals(
            paired["label_a2"]
        ):
            raise ValueError(f"A0/A2 development pair differs in fold={fold}.")
        a0 = balanced_accuracy(paired["label_a0"], paired["prediction_a0"], 12)
        a2 = balanced_accuracy(paired["label_a2"], paired["prediction_a2"], 12)
        rows.append(
            {
                "subject_id": str(paired["subject_id"].iloc[0]),
                "fold_index": fold,
                "n_samples": len(paired),
                "a0_balanced_accuracy": a0,
                "a2_balanced_accuracy": a2,
                "balanced_accuracy_delta": a2 - a0,
            }
        )
    subjects = pd.DataFrame(rows).sort_values("subject_id").reset_index(drop=True)
    summary = {
        "schema": "cfeg.development-loso-summary.v1",
        "status": "technical-valid",
        "scientific_interpretation": "descriptive_falsification_only_inconclusive",
        "confirmatory_claim_allowed": False,
        "population_inference_allowed": False,
        "n_subjects": 3,
        "n_samples_per_role": expected_n_samples,
        "mean_a0_balanced_accuracy": float(subjects["a0_balanced_accuracy"].mean()),
        "mean_a2_balanced_accuracy": float(subjects["a2_balanced_accuracy"].mean()),
        "mean_balanced_accuracy_delta": float(subjects["balanced_accuracy_delta"].mean()),
        "minimum_one_sided_exact_p_value": 0.125,
        "outcome_taxonomy": [
            "technical-valid",
            "directional-warning",
            "benefit-not-demonstrated",
            "population-inconclusive",
            "invalid_assay",
        ],
        "warning": (
            "N=3 cannot establish population benefit, equivalence, shortcut absence, "
            "safety, or architecture superiority."
        ),
    }
    return subjects, summary


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
