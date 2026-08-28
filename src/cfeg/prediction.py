from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from cfeg.eval_loop import (
    _checkpoint_split_indices,
    _loader,
    _sample_id_identity_sha256,
    collect_predictions,
    load_evaluation_context,
)

PREDICTION_SCHEMA_VERSION = "cfeg.predictions.v1"
OOD_REQUIRED_COLUMNS = ("sample_id", "label", "prediction")


def run_prediction(
    ckpt_path: str | Path,
    processed_dir: str | Path,
    output_path: str | Path,
    *,
    split: str = "test",
    scenario: str = "clean",
) -> dict[str, Any]:
    """Write an auditable prediction artifact for paired held-out analysis.

    The safe default selects the checkpoint-adjacent ``split.csv`` test rows.
    Whole-manifest inference remains available only through explicit
    ``split="all"`` selection.
    """

    if split not in {"test", "all"}:
        raise ValueError(f"split must be 'test' or 'all', got {split!r}.")
    if not str(scenario).strip():
        raise ValueError("scenario must be non-empty.")

    checkpoint_path = Path(ckpt_path).expanduser().resolve()
    processed_path = Path(processed_dir).expanduser().resolve()
    output = Path(output_path).expanduser()
    eval_cfg = {"data": {"processed_dirs": [str(processed_path)]}}
    context = load_evaluation_context(eval_cfg, checkpoint_path)
    indices = _prediction_indices(context, split=split)
    y_true, logits, sample_ids = collect_predictions(
        context["model"], _loader(context, indices), context["device"]
    )
    frame, artifact = _prediction_artifact(
        context=context,
        checkpoint_path=checkpoint_path,
        processed_path=processed_path,
        split=split,
        scenario=str(scenario),
        sample_ids=sample_ids,
        y_true=y_true,
        logits=logits,
    )

    output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output, index=False)
    artifact["prediction_csv"] = str(output.resolve())
    artifact["prediction_csv_sha256"] = _file_sha256(output)
    provenance_path = output.with_name(f"{output.stem}_provenance.json")
    provenance_path.write_text(
        json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return {
        "output_csv": str(output),
        "provenance_json": str(provenance_path),
        **artifact,
    }


def _prediction_indices(context: dict[str, Any], *, split: str) -> np.ndarray:
    if split == "test":
        return _checkpoint_split_indices(context, split="test")
    if split == "all":
        return np.arange(len(context["dataset"]), dtype=int)
    raise ValueError(f"split must be 'test' or 'all', got {split!r}.")


def _prediction_artifact(
    *,
    context: dict[str, Any],
    checkpoint_path: Path,
    processed_path: Path,
    split: str,
    scenario: str,
    sample_ids: list[str],
    y_true: np.ndarray,
    logits: np.ndarray,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    labels = np.asarray(y_true)
    scores = np.asarray(logits)
    if scores.ndim != 2:
        raise ValueError(f"logits must have shape [N,K], got {scores.shape}.")
    if len(sample_ids) != len(labels) or len(labels) != len(scores):
        raise ValueError(
            "Prediction arrays have inconsistent lengths: "
            f"sample_ids={len(sample_ids)}, labels={len(labels)}, logits={len(scores)}."
        )
    normalized_ids = [str(sample_id) for sample_id in sample_ids]
    if len(set(normalized_ids)) != len(normalized_ids):
        raise ValueError("Prediction sample_id values must be unique within one artifact.")

    probabilities = torch.softmax(torch.from_numpy(scores), dim=1).numpy()
    predicted = scores.argmax(axis=1)
    class_map = context["checkpoint"].get("class_map") or {}
    frequencies = [
        (class_map.get(str(int(label))) or {}).get("stimulus_frequency_hz")
        for label in predicted
    ]
    sample_hash = _sample_id_identity_sha256(normalized_ids)
    checkpoint_hash = _file_sha256(checkpoint_path)
    frame = pd.DataFrame(
        {
            "sample_id": normalized_ids,
            "label": labels.astype(int),
            "prediction": predicted.astype(int),
            "predicted_frequency_hz": frequencies,
            "confidence": probabilities.max(axis=1),
            "scenario": scenario,
            "selection_split": split,
            "sample_identity_sha256": sample_hash,
            "checkpoint_sha256": checkpoint_hash,
            "prediction_schema_version": PREDICTION_SCHEMA_VERSION,
        }
    )
    split_path = checkpoint_path.parent / "split.csv" if split == "test" else None
    artifact: dict[str, Any] = {
        "prediction_schema_version": PREDICTION_SCHEMA_VERSION,
        "created_by": "scripts/predict.py",
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_sha256": checkpoint_hash,
        "processed_dir": str(processed_path),
        "scenario": scenario,
        "selection_split": split,
        "split_manifest_path": str(split_path) if split_path is not None else None,
        "split_manifest_sha256": (
            _file_sha256(split_path) if split_path is not None else None
        ),
        "n_samples": len(frame),
        "sample_identity_sha256": sample_hash,
        "required_ood_columns": list(OOD_REQUIRED_COLUMNS),
    }
    return frame, artifact


def _file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
