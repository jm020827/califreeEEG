from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Protocol

import numpy as np

from cfeg.baselines.calibration import (
    predict_filterbank_ensemble_trca,
    predict_lst_filterbank_ensemble_trca,
    predict_same_filterbank_ensemble_trca,
    predict_supervised_template_correlation,
)
from cfeg.baselines.fbcca import apply_filterbank, predict_fbcca
from cfeg.utils.config import load_config

BaselineAdapterId = Literal[
    "strict_FBCCA",
    "target_template_correlation",
    "target_filterbank_eTRCA",
    "same3_filterbank_eTRCA",
    "chiang2021_LST_filterbank_eTRCA",
]


@dataclass(frozen=True)
class BaselineDescriptor:
    adapter_id: BaselineAdapterId
    version: str
    applicable_budgets: tuple[int, ...]
    requires_target_support_labels: bool
    minimum_target_trials_per_class: int
    requires_source_labeled_eeg: bool
    uses_frequency_codebook: bool
    external_context_access: bool = False
    query_batch_transduction: bool = False
    deterministic: bool = True


@dataclass(frozen=True)
class BaselineSignalBatch:
    query_x: np.ndarray
    target_support_x: np.ndarray | None
    target_support_y: np.ndarray | None
    source_x: np.ndarray | None
    source_y: np.ndarray | None
    frequencies_hz: np.ndarray | None
    sfreq: float
    phases_rad: np.ndarray | None = None


@dataclass(frozen=True)
class AtomicBaselineScoreOutput:
    scores: np.ndarray
    score_schema: str
    score_interpretation: str = "uncalibrated_raw_class_score_argmax_only"


class AtomicBaselineAdapter(Protocol):
    descriptor: BaselineDescriptor

    def predict(self, batch: BaselineSignalBatch, *, budget: int) -> AtomicBaselineScoreOutput: ...


DESCRIPTORS: dict[BaselineAdapterId, BaselineDescriptor] = {
    "strict_FBCCA": BaselineDescriptor(
        adapter_id="strict_FBCCA",
        version="v1",
        applicable_budgets=(0,),
        requires_target_support_labels=False,
        minimum_target_trials_per_class=0,
        requires_source_labeled_eeg=False,
        uses_frequency_codebook=True,
    ),
    "target_template_correlation": BaselineDescriptor(
        adapter_id="target_template_correlation",
        version="v1",
        applicable_budgets=(1, 3, 5),
        requires_target_support_labels=True,
        minimum_target_trials_per_class=1,
        requires_source_labeled_eeg=False,
        uses_frequency_codebook=False,
    ),
    "target_filterbank_eTRCA": BaselineDescriptor(
        adapter_id="target_filterbank_eTRCA",
        version="v1",
        applicable_budgets=(3, 5),
        requires_target_support_labels=True,
        minimum_target_trials_per_class=2,
        requires_source_labeled_eeg=False,
        uses_frequency_codebook=False,
    ),
    "same3_filterbank_eTRCA": BaselineDescriptor(
        adapter_id="same3_filterbank_eTRCA",
        version="cleanroom_common_protocol_v1",
        applicable_budgets=(1,),
        requires_target_support_labels=True,
        minimum_target_trials_per_class=1,
        requires_source_labeled_eeg=False,
        uses_frequency_codebook=True,
    ),
    "chiang2021_LST_filterbank_eTRCA": BaselineDescriptor(
        adapter_id="chiang2021_LST_filterbank_eTRCA",
        version="cleanroom_common_protocol_v1",
        applicable_budgets=(1, 3, 5),
        requires_target_support_labels=True,
        minimum_target_trials_per_class=1,
        requires_source_labeled_eeg=True,
        uses_frequency_codebook=False,
    ),
}


class GovernedCalibrationBaselineAdapter:
    """Label-free-query adapter for the frozen calibration baseline family."""

    def __init__(
        self,
        adapter_id: BaselineAdapterId,
        *,
        n_classes: int,
        filterbank: dict[str, object] | None = None,
        filterbank_config_path: str | Path | None = None,
        filterbank_config_sha256: str | None = None,
        lst_ridge_ratio: float = 1e-6,
        trca_ridge_ratio: float = 1e-6,
        same_seed: int = 20260904,
    ) -> None:
        if adapter_id not in DESCRIPTORS:
            raise ValueError(f"Unknown calibration baseline adapter {adapter_id!r}.")
        if n_classes <= 1:
            raise ValueError("n_classes must exceed one.")
        descriptor = DESCRIPTORS[adapter_id]
        requires_filterbank = adapter_id in {
            "strict_FBCCA",
            "target_filterbank_eTRCA",
            "same3_filterbank_eTRCA",
            "chiang2021_LST_filterbank_eTRCA",
        }
        if requires_filterbank:
            if (
                filterbank is None
                or filterbank_config_path is None
                or not _is_sha256(filterbank_config_sha256)
            ):
                raise ValueError(
                    "Filter-bank adapters require an explicit config path and bound file SHA-256."
                )
            config_path = Path(filterbank_config_path).resolve()
            if not config_path.is_file() or _sha256_file(config_path) != filterbank_config_sha256:
                raise ValueError("Filter-bank config file does not match its frozen SHA-256.")
            file_payload = load_config(config_path, strict_env=False)
            if _canonical_json_sha256(
                _normalize_config_scalars(file_payload)
            ) != _canonical_json_sha256(_normalize_config_scalars(filterbank)):
                raise ValueError("Filter-bank runtime payload differs from the bound config file.")
        elif any(
            value is not None
            for value in (filterbank, filterbank_config_path, filterbank_config_sha256)
        ):
            raise ValueError("Template correlation must not receive a filter-bank config.")
        self.descriptor = descriptor
        self.n_classes = int(n_classes)
        self.filterbank = filterbank
        self.filterbank_config_sha256 = filterbank_config_sha256
        self.filterbank_config_path = (
            str(Path(filterbank_config_path).resolve())
            if filterbank_config_path is not None
            else None
        )
        self.filterbank_payload_sha256 = (
            _canonical_json_sha256(filterbank) if filterbank is not None else None
        )
        self.lst_ridge_ratio = float(lst_ridge_ratio)
        self.trca_ridge_ratio = float(trca_ridge_ratio)
        self.same_seed = int(same_seed)
        self._lst_source_filter_cache: dict[
            str, tuple[np.ndarray, dict[str, object]]
        ] = {}

    def predict(
        self, batch: BaselineSignalBatch, *, budget: int
    ) -> AtomicBaselineScoreOutput:
        if int(budget) not in self.descriptor.applicable_budgets:
            raise ValueError(
                f"{self.descriptor.adapter_id} is not applicable at budget k={budget}."
            )
        query = _query_array(batch.query_x)
        if not np.isfinite(batch.sfreq) or batch.sfreq <= 0.0:
            raise ValueError("Baseline sampling rate must be positive and finite.")
        adapter_id = self.descriptor.adapter_id
        if adapter_id == "strict_FBCCA":
            if any(
                value is not None
                for value in (
                    batch.target_support_x,
                    batch.target_support_y,
                    batch.source_x,
                    batch.source_y,
                )
            ):
                raise ValueError("Strict FBCCA k=0 must not receive labeled EEG.")
            frequencies = np.asarray(batch.frequencies_hz, dtype=float)
            if frequencies.shape != (self.n_classes,):
                raise ValueError("Strict FBCCA requires one frequency per class.")
            scores = np.stack(
                [
                    predict_fbcca(
                        trial,
                        frequencies,
                        batch.sfreq,
                        filterbank=self.filterbank,
                    )[1]
                    for trial in query
                ]
            )
            schema = "strict_fbcca_bound_filterbank_raw_scores_v1"
        else:
            support, support_labels = _labeled_trials(
                batch.target_support_x,
                batch.target_support_y,
                n_classes=self.n_classes,
                minimum_per_class=self.descriptor.minimum_target_trials_per_class,
                name="target support",
            )
            if adapter_id == "target_template_correlation":
                output = predict_supervised_template_correlation(
                    query, support, support_labels, n_classes=self.n_classes
                )
            elif adapter_id == "target_filterbank_eTRCA":
                output = predict_filterbank_ensemble_trca(
                    query,
                    support,
                    support_labels,
                    n_classes=self.n_classes,
                    sfreq=batch.sfreq,
                    filterbank=self.filterbank,
                    ridge_ratio=self.trca_ridge_ratio,
                )
            elif adapter_id == "same3_filterbank_eTRCA":
                if batch.frequencies_hz is None or batch.phases_rad is None:
                    raise ValueError("SAME requires the frozen frequency/phase codebook.")
                if batch.source_x is not None or batch.source_y is not None:
                    raise ValueError("SAME target-only comparator must not receive source EEG.")
                output = predict_same_filterbank_ensemble_trca(
                    query,
                    support,
                    support_labels,
                    frequencies_hz=batch.frequencies_hz,
                    phases_rad=batch.phases_rad,
                    n_classes=self.n_classes,
                    sfreq=batch.sfreq,
                    filterbank=self.filterbank,
                    seed=self.same_seed,
                    trca_ridge_ratio=self.trca_ridge_ratio,
                )
            else:
                source, source_labels = _labeled_trials(
                    batch.source_x,
                    batch.source_y,
                    n_classes=self.n_classes,
                    minimum_per_class=1,
                    name="source pool",
                )
                source_cache_key = _source_filter_cache_key(
                    source,
                    source_labels,
                    sfreq=batch.sfreq,
                    filterbank_payload_sha256=self.filterbank_payload_sha256,
                )
                filtered_source = self._lst_source_filter_cache.get(source_cache_key)
                if filtered_source is None:
                    filtered_source = apply_filterbank(
                        source,
                        sfreq=batch.sfreq,
                        filterbank=self.filterbank,
                    )
                    self._lst_source_filter_cache[source_cache_key] = filtered_source
                output = predict_lst_filterbank_ensemble_trca(
                    query,
                    support,
                    support_labels,
                    source,
                    source_labels,
                    n_classes=self.n_classes,
                    sfreq=batch.sfreq,
                    filterbank=self.filterbank,
                    lst_ridge_ratio=self.lst_ridge_ratio,
                    trca_ridge_ratio=self.trca_ridge_ratio,
                    filtered_source_cache=filtered_source,
                )
            scores = output.scores
            schema = output.score_schema
        if scores.shape != (len(query), self.n_classes) or not np.isfinite(scores).all():
            raise ValueError("Baseline adapter must return one finite full class-score vector.")
        return AtomicBaselineScoreOutput(scores=scores, score_schema=schema)


def _query_array(value: np.ndarray) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.ndim != 3 or not len(array) or not np.isfinite(array).all():
        raise ValueError("Query EEG must be a nonempty finite [trial,channel,time] array.")
    return array


def _labeled_trials(
    x: np.ndarray | None,
    y: np.ndarray | None,
    *,
    n_classes: int,
    minimum_per_class: int,
    name: str,
) -> tuple[np.ndarray, np.ndarray]:
    if x is None or y is None:
        raise ValueError(f"{name} EEG and labels are required.")
    values = np.asarray(x, dtype=np.float64)
    labels_raw = np.asarray(y)
    if values.ndim != 3 or not len(values) or not np.isfinite(values).all():
        raise ValueError(f"{name} must be a finite [trial,channel,time] array.")
    if labels_raw.shape != (len(values),):
        raise ValueError(f"{name} labels must align with trials.")
    labels = labels_raw.astype(np.int64)
    if not np.equal(labels_raw.astype(float), labels).all():
        raise ValueError(f"{name} labels must be exact integers.")
    if labels.min() < 0 or labels.max() >= n_classes:
        raise ValueError(f"{name} labels fall outside the class vocabulary.")
    counts = np.bincount(labels, minlength=n_classes)
    if not np.all(counts == counts[0]) or counts[0] < minimum_per_class:
        raise ValueError(f"{name} must be class-balanced with sufficient trials.")
    return values, labels


def _canonical_json_sha256(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _source_filter_cache_key(
    source: np.ndarray,
    labels: np.ndarray,
    *,
    sfreq: float,
    filterbank_payload_sha256: str | None,
) -> str:
    digest = hashlib.sha256()
    for value in (np.ascontiguousarray(source), np.ascontiguousarray(labels)):
        digest.update(str(value.dtype).encode())
        digest.update(json.dumps(list(value.shape), separators=(",", ":")).encode())
        digest.update(value.view(np.uint8).tobytes())
    digest.update(repr(float(sfreq)).encode())
    digest.update(str(filterbank_payload_sha256).encode())
    return digest.hexdigest()


def _normalize_config_scalars(value: object) -> object:
    if isinstance(value, dict):
        return {str(key): _normalize_config_scalars(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_normalize_config_scalars(item) for item in value]
    if isinstance(value, str):
        try:
            number = float(value)
        except ValueError:
            return value
        if np.isfinite(number):
            return int(number) if number.is_integer() else number
    return value


def _is_sha256(value: object) -> bool:
    if not isinstance(value, str) or len(value) != 64:
        return False
    try:
        int(value, 16)
    except ValueError:
        return False
    return value == value.lower()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
