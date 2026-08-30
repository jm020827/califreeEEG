from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd
import pytest

from cfeg.analysis.development_loso import aggregate_complete_development_loso


def _sha256(path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _fixture(tmp_path):
    manifest_rows = []
    jobs = []
    for fold, subject in enumerate(("sub001", "sub002", "sub003")):
        subject_rows = []
        for label in range(12):
            row = {
                "sample_id": f"{subject}-{label}",
                "subject_id": subject,
                "label": label,
            }
            manifest_rows.append(row)
            subject_rows.append(row)
        for role in ("A0_eeg_only", "A2_structured_condition_prompt"):
            path = tmp_path / f"fold{fold}-{role}.csv"
            prediction = [0] * 12 if role == "A0_eeg_only" else list(range(12))
            frame = pd.DataFrame(
                {
                    "sample_id": [row["sample_id"] for row in subject_rows],
                    "label": list(range(12)),
                    "prediction": prediction,
                    "selection_split": "val",
                    "primary_ablation_role": role,
                    "optimization_seed": 42,
                    "fold_index": fold,
                    "checkpoint_role": "development_fixed_epoch",
                    "checkpoint_selection_metric": "fixed_epoch",
                }
            )
            frame.to_csv(path, index=False)
            path.with_name(f"{path.stem}_provenance.json").write_text(
                json.dumps({"prediction_csv_sha256": _sha256(path)}), encoding="utf-8"
            )
            jobs.append(
                {
                    "seed": 42,
                    "fold_index": fold,
                    "role": role,
                    "prediction_csv": str(path),
                }
            )
    return jobs, pd.DataFrame(manifest_rows)


def test_complete_development_loso_is_descriptive_and_nonconfirmatory(tmp_path) -> None:
    jobs, manifest = _fixture(tmp_path)

    subjects, summary = aggregate_complete_development_loso(
        jobs,
        manifest,
        expected_subjects=["sub001", "sub002", "sub003"],
        expected_n_samples=36,
    )

    assert len(subjects) == 3
    assert subjects["a0_balanced_accuracy"].eq(1 / 12).all()
    assert subjects["a2_balanced_accuracy"].eq(1.0).all()
    assert summary["confirmatory_claim_allowed"] is False
    assert summary["population_inference_allowed"] is False
    assert summary["minimum_one_sided_exact_p_value"] == 0.125


def test_incomplete_development_loso_does_not_reveal_metrics(tmp_path) -> None:
    jobs, manifest = _fixture(tmp_path)

    with pytest.raises(ValueError, match="exact six"):
        aggregate_complete_development_loso(
            jobs[:-1],
            manifest,
            expected_subjects=["sub001", "sub002", "sub003"],
            expected_n_samples=36,
        )


def test_development_loso_rejects_prediction_tampering(tmp_path) -> None:
    jobs, manifest = _fixture(tmp_path)
    prediction = Path(jobs[0]["prediction_csv"])
    prediction.write_text(prediction.read_text(encoding="utf-8") + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="changed after creation"):
        aggregate_complete_development_loso(
            jobs,
            manifest,
            expected_subjects=["sub001", "sub002", "sub003"],
            expected_n_samples=36,
        )
