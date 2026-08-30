from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from cfeg.baselines.fbcca import (
    predict_cca,
    predict_fbcca,
    resolve_filterbank_parameters,
)
from cfeg.data.datasets import EEGProcessedDataset
from cfeg.data.preprocess import CanonicalChannelMap
from cfeg.data.schema import load_manifest
from cfeg.governance import GovernanceError, bind_cohort, resolve_research_access
from cfeg.metrics import accuracy, balanced_accuracy, itr_bits_per_min, macro_f1


def evaluate_frequency_baseline(
    processed_dir: str | Path,
    *,
    method: str,
    channel_set: str,
    selection_csv: str | Path | None = None,
    selection_split: str = "test",
    strict_channel_set: bool = True,
    channel_sets_path: str | Path = "configs/channel_sets.yaml",
    max_subjects: int | None = None,
    n_harmonics: int = 3,
    regularization: float = 1e-8,
    filterbank: Any = None,
    trial_time_sec: float = 2.0,
    research_config: dict | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object]]:
    """Evaluate training-free CCA/FBCCA and return sample, subject, and summary results."""
    if method not in {"cca", "fbcca"}:
        raise ValueError(f"method must be 'cca' or 'fbcca', got {method!r}.")
    resolved_filterbank = filterbank
    if method == "fbcca" and (filterbank is None or isinstance(filterbank, Mapping)):
        resolved_filterbank = dict(filterbank or {})
        resolved_filterbank.setdefault("n_harmonics", n_harmonics)
        resolved_filterbank.setdefault("regularization", regularization)
    governed_access = None
    governed_cohort = None
    if _processed_dir_is_wearable_v3(processed_dir):
        if research_config is None:
            raise GovernanceError(
                "wearable_v3 baseline access requires --research-config and a governed "
                "development cohort."
            )
        governed_access = resolve_research_access(research_config)
        if governed_access.execution_phase != "development":
            raise GovernanceError(
                "Confirmatory CCA/FBCCA evaluation is not registered in the frozen action plan."
            )
        governed_cohort = bind_cohort(load_manifest(processed_dir), governed_access)
    dataset = EEGProcessedDataset(
        [processed_dir],
        allowed_subject_ids=(
            set(governed_cohort.selected_subject_ids)
            if governed_cohort is not None
            else None
        ),
        persistent_hdf5_handles=bool(
            (research_config or {}).get("data", {}).get(
                "persistent_hdf5_handles", False
            )
        ),
    )
    if _is_wearable_v3(dataset):
        if governed_access is None:
            raise GovernanceError(
                "wearable_v3 asset identity was not available for the required preflight gate."
            )
    elif governed_access is not None:
        raise GovernanceError("Governed baseline path does not resolve to wearable_v3.")
    labels = sorted(int(label) for label in dataset.class_map)
    if labels != list(range(len(labels))):
        raise ValueError(f"class_map labels must be contiguous from zero, got {labels}.")
    frequencies = np.asarray(
        [float(dataset.class_map[str(label)]["stimulus_frequency_hz"]) for label in labels]
    )
    channel_ids = _resolve_channel_ids(channel_set, channel_sets_path)
    manifest_sample_ids = [str(entry[2]["sample_id"]) for entry in dataset.entries]
    if len(manifest_sample_ids) != len(set(manifest_sample_ids)):
        raise ValueError("Processed manifest contains duplicate sample_id values.")
    selection_ids: set[str] | None = None
    if selection_csv is not None:
        selection_ids = _load_selection_ids(selection_csv, split=selection_split)
        missing_ids = sorted(selection_ids - set(manifest_sample_ids))
        if missing_ids:
            raise ValueError(
                f"Selection CSV contains {len(missing_ids)} sample IDs absent from the "
                f"processed manifest: {missing_ids[:5]}"
            )
    eligible = [
        index
        for index, sample_id in enumerate(manifest_sample_ids)
        if selection_ids is None or sample_id in selection_ids
    ]
    if selection_ids is not None:
        selected_outside_role = selection_ids - set(manifest_sample_ids)
        if selected_outside_role:
            raise GovernanceError(
                "Baseline selection contains samples outside the authorized cohort role."
            )
    if not eligible:
        raise ValueError("Baseline evaluation selection contains no samples.")
    selected_subjects = sorted(
        {str(dataset.entries[index][2].get("subject_id", "unknown")) for index in eligible}
    )
    if max_subjects is not None:
        if max_subjects < 1:
            raise ValueError("max_subjects must be positive.")
        selected_subjects = selected_subjects[:max_subjects]
    selected = [
        index
        for index in eligible
        if str(dataset.entries[index][2].get("subject_id", "unknown")) in selected_subjects
    ]
    if not selected:
        raise ValueError("Baseline evaluation selection contains no samples.")

    rows: list[dict[str, object]] = []
    effective_filterbanks: list[dict[str, object]] = []
    effective_channel_ids: set[tuple[int, ...]] = set()
    for index in selected:
        sample = dataset[index]
        eeg = _select_eeg_channels(
            sample.x,
            sample.channel_mask,
            sample.canonical_channel_ids,
            channel_ids,
            strict=bool(strict_channel_set and channel_ids is not None),
        )
        selected_ids = tuple(
            sorted(
                int(value)
                for value in np.asarray(sample.canonical_channel_ids)[
                    np.asarray(sample.channel_mask, dtype=bool)
                    & (
                        np.ones_like(sample.channel_mask, dtype=bool)
                        if channel_ids is None
                        else np.isin(sample.canonical_channel_ids, list(channel_ids))
                    )
                ]
            )
        )
        effective_channel_ids.add(selected_ids)
        if method == "cca":
            prediction, scores = predict_cca(
                eeg,
                frequencies,
                sample.sfreq,
                n_harmonics=n_harmonics,
                regularization=regularization,
            )
        else:
            parameters = resolve_filterbank_parameters(resolved_filterbank, sample.sfreq)
            if parameters not in effective_filterbanks:
                effective_filterbanks.append(parameters)
            prediction, scores = predict_fbcca(
                eeg,
                frequencies,
                sample.sfreq,
                filterbank=resolved_filterbank,
            )
        ordered = np.sort(scores)
        margin = float(ordered[-1] - ordered[-2]) if len(ordered) > 1 else float(ordered[-1])
        row: dict[str, object] = {
            "sample_id": sample.sample_id,
            "dataset_id": sample.dataset_id,
            "subject_id": sample.subject_id,
            "electrode_type": sample.electrode_type or "unknown",
            "label": int(sample.y),
            "prediction": int(prediction),
            "predicted_frequency_hz": float(frequencies[prediction]),
            "true_frequency_hz": float(frequencies[sample.y]),
            "max_score": float(scores[prediction]),
            "score_margin": margin,
            "n_channels": int(eeg.shape[0]),
        }
        row.update({f"score_label_{label:02d}": float(scores[label]) for label in labels})
        rows.append(row)

    predictions = pd.DataFrame(rows)
    subject_rows = []
    for (dataset_id, subject_id), group in predictions.groupby(
        ["dataset_id", "subject_id"], sort=True
    ):
        subject_rows.append(
            {
                "dataset_id": dataset_id,
                "subject_id": subject_id,
                **_prediction_metrics(group, n_classes=len(labels), trial_time_sec=trial_time_sec),
            }
        )
    subjects = pd.DataFrame(subject_rows)
    aggregate = _prediction_metrics(
        predictions, n_classes=len(labels), trial_time_sec=trial_time_sec
    )
    condition_metrics = {
        str(condition): _prediction_metrics(
            group, n_classes=len(labels), trial_time_sec=trial_time_sec
        )
        for condition, group in predictions.groupby("electrode_type", sort=True)
    }
    subject_condition_metrics = []
    for (dataset_id, subject_id, condition), group in predictions.groupby(
        ["dataset_id", "subject_id", "electrode_type"], sort=True
    ):
        subject_condition_metrics.append(
            {
                "dataset_id": str(dataset_id),
                "subject_id": str(subject_id),
                "electrode_type": str(condition),
                **_prediction_metrics(group, n_classes=len(labels), trial_time_sec=trial_time_sec),
            }
        )
    canonical = CanonicalChannelMap.from_yaml()
    effective_channel_names = [
        [canonical.id_to_name.get(channel_id, str(channel_id)) for channel_id in ids]
        for ids in sorted(effective_channel_ids)
    ]
    summary: dict[str, object] = {
        "method": method,
        "processed_dir": str(Path(processed_dir).expanduser().resolve()),
        "selection_source": (
            str(Path(selection_csv).expanduser().resolve())
            if selection_csv is not None
            else "whole_manifest"
        ),
        "selection_split": selection_split if selection_csv is not None else None,
        "selection_csv_sha256": (
            _file_sha256(selection_csv) if selection_csv is not None else None
        ),
        "channel_set": channel_set,
        "strict_channel_set": bool(strict_channel_set),
        "effective_channel_ids": [list(ids) for ids in sorted(effective_channel_ids)],
        "effective_channel_names": effective_channel_names,
        "n_harmonics": int(n_harmonics),
        "regularization": float(regularization),
        "filterbank": effective_filterbanks if method == "fbcca" else None,
        "fbcca_implementation": (
            "independent_weighted_squared_correlation_implementation"
            if method == "fbcca"
            else None
        ),
        "condition_metrics": condition_metrics,
        "subject_condition_metrics": subject_condition_metrics,
        "trial_time_sec": float(trial_time_sec),
        "itr_interpretation": "theoretical_for_stated_trial_time",
        "n_subjects": len(subjects),
        "mean_subject_balanced_accuracy": float(subjects["balanced_accuracy"].mean()),
        "median_subject_balanced_accuracy": float(subjects["balanced_accuracy"].median()),
        "worst_subject_balanced_accuracy": float(subjects["balanced_accuracy"].min()),
        "sample_identity_sha256": _sample_identity_sha256(predictions["sample_id"]),
        "label_frequency_identity_sha256": _label_frequency_identity_sha256(predictions),
        "processed_metadata_sha256": _processed_metadata_sha256(processed_dir),
        "research_access": (
            governed_access.contract() if governed_access is not None else None
        ),
        "cohort": governed_cohort.contract() if governed_cohort is not None else None,
        **aggregate,
    }
    return predictions, subjects, summary


def _is_wearable_v3(dataset: EEGProcessedDataset) -> bool:
    return any(
        info.get("dataset_id") == "wearable"
        and info.get("dataset_revision") == "wearable_v3"
        for info in dataset.asset_infos
    )


def _processed_dir_is_wearable_v3(processed_dir: str | Path) -> bool:
    path = Path(processed_dir).expanduser()
    info_path = path / "asset_info.json"
    if info_path.is_file():
        try:
            info = json.loads(info_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            info = {}
        if info.get("dataset_id") == "wearable" and info.get("dataset_revision") == "wearable_v3":
            return True
    return path.name == "wearable_v3"


def _load_selection_ids(path: str | Path, *, split: str = "test") -> set[str]:
    table = pd.read_csv(path, dtype=str)
    if "sample_id" not in table.columns:
        raise ValueError(f"Selection CSV {path} is missing sample_id.")
    if table["sample_id"].duplicated().any():
        duplicates = sorted(
            table.loc[table["sample_id"].duplicated(), "sample_id"].astype(str).unique()
        )
        raise ValueError(
            f"Selection CSV {path} contains duplicate sample_id values: {duplicates[:5]}"
        )
    if "split" in table.columns:
        table = table.loc[table["split"].astype(str).str.lower().eq(split.lower())]
    selected = set(table["sample_id"].astype(str))
    if not selected:
        qualifier = f" assigned to split {split!r}" if "split" in table.columns else ""
        raise ValueError(f"Selection CSV {path} contains no sample IDs{qualifier}.")
    return selected


def _resolve_channel_ids(channel_set: str, channel_sets_path: str | Path) -> set[int] | None:
    if channel_set == "all":
        return None
    with Path(channel_sets_path).open(encoding="utf-8") as handle:
        registry = yaml.safe_load(handle) or {}
    if channel_set not in registry:
        raise KeyError(f"Unknown channel set {channel_set!r}: {sorted(registry)}")
    canonical = CanonicalChannelMap.from_yaml()
    ids = set(canonical.get_ids(list(registry[channel_set])))
    ids.discard(0)
    if not ids:
        raise ValueError(f"Channel set {channel_set!r} contains no canonical channels.")
    return ids


def _select_eeg_channels(
    x: np.ndarray,
    channel_mask: np.ndarray,
    canonical_channel_ids: np.ndarray,
    keep_ids: set[int] | None,
    *,
    strict: bool = False,
) -> np.ndarray:
    eeg = np.asarray(x, dtype=np.float64)
    mask = np.asarray(channel_mask, dtype=bool)
    ids = np.asarray(canonical_channel_ids, dtype=int)
    if eeg.ndim != 2 or mask.shape != (eeg.shape[0],) or ids.shape != (eeg.shape[0],):
        raise ValueError("EEG, channel mask, and canonical IDs have inconsistent shapes.")
    if keep_ids is not None and strict:
        available = {int(value) for value in ids[mask] if int(value) > 0}
        missing = sorted(keep_ids - available)
        if missing:
            raise ValueError(
                "Requested strict channel set is incomplete for this sample; "
                f"missing canonical channel IDs {missing}."
            )
    selected = mask if keep_ids is None else mask & np.isin(ids, list(keep_ids))
    if not selected.any():
        raise ValueError("Requested channel set has no channels in this sample.")
    return eeg[selected]


def _prediction_metrics(
    predictions: pd.DataFrame, *, n_classes: int, trial_time_sec: float
) -> dict[str, float | int]:
    y_true = predictions["label"].to_numpy(dtype=int)
    y_pred = predictions["prediction"].to_numpy(dtype=int)
    acc = accuracy(y_true, y_pred)
    return {
        "accuracy": acc,
        "balanced_accuracy": balanced_accuracy(y_true, y_pred, n_classes=n_classes),
        "macro_f1": macro_f1(y_true, y_pred, n_classes=n_classes),
        "itr_bits_per_min": itr_bits_per_min(n_classes, acc, trial_time_sec),
        "n_samples": len(predictions),
    }


def _sample_identity_sha256(sample_ids: pd.Series) -> str:
    digest = hashlib.sha256()
    for sample_id in sorted(sample_ids.astype(str)):
        encoded = sample_id.encode()
        digest.update(len(encoded).to_bytes(8, "big"))
        digest.update(encoded)
    return digest.hexdigest()


def _label_frequency_identity_sha256(predictions: pd.DataFrame) -> str:
    digest = hashlib.sha256()
    columns = ["sample_id", "label", "true_frequency_hz"]
    for row in (
        predictions.loc[:, columns].sort_values("sample_id").itertuples(index=False, name=None)
    ):
        encoded = "\x1f".join(str(value) for value in row).encode()
        digest.update(len(encoded).to_bytes(8, "big"))
        digest.update(encoded)
    return digest.hexdigest()


def _processed_metadata_sha256(processed_dir: str | Path) -> str:
    root = Path(processed_dir)
    candidates = [
        root / "signals.h5",
        root / "manifest.jsonl",
        root / "manifest.parquet",
        root / "class_map.json",
        root / "preprocess_config.yaml",
        root / "asset_info.json",
    ]
    digest = hashlib.sha256()
    for path in candidates:
        if not path.exists():
            continue
        name = path.name.encode()
        digest.update(len(name).to_bytes(8, "big"))
        digest.update(name)
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    return digest.hexdigest()


def _file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
