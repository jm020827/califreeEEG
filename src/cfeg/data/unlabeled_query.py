from __future__ import annotations

import copy
import hashlib
import hmac
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import pandas as pd
import torch

from cfeg.constants import METADATA_CONTRACT_V04_DEV
from cfeg.data.io_hdf5 import HDF5SampleReader
from cfeg.data.metadata_calibration_features import WEARABLE_CALIBRATION_CHANNEL_IDS
from cfeg.data.metadata_calibration_interventions import (
    metadata_calibration_base_input_sha256,
)

_QUERY_TOKEN = re.compile(r"q_[0-9a-f]{64}")
_QUERY_COLUMNS = {
    "query_token",
    "signal_index",
    "subject_id",
    "electrode_type",
    "checkpoint_group",
    "run_id",
    "canonical_channel_ids",
    "sfreq_processed",
    "query_signal_std_by_channel",
    "query_signal_std_missing_by_channel",
    "impedance_kohm_by_channel",
    "impedance_missing_by_channel",
    "base_input_sha256",
}
_FORBIDDEN_COLUMNS = {
    "h5_index",
    "sample_id",
    "label",
    "target_label",
    "prediction",
    "trial_id",
    "stimulus_frequency_hz",
    "stimulus_phase_rad",
    "source_file",
}


@dataclass(frozen=True)
class UnlabeledQuerySample:
    x: np.ndarray
    channel_mask: np.ndarray
    query_token: str
    subject_id: str
    electrode_type: str
    checkpoint_group: str
    run_id: str
    canonical_channel_ids: np.ndarray
    sfreq: float
    query_signal_std_by_channel: np.ndarray
    query_signal_std_missing_by_channel: np.ndarray
    impedance_kohm_by_channel: np.ndarray
    impedance_missing_by_channel: np.ndarray
    base_input_sha256: str


@dataclass(frozen=True)
class UnlabeledQueryBatch:
    """Model-ready query tensors with no label or raw-identity field."""

    x: torch.Tensor
    cond: dict[str, Any]
    query_tokens: tuple[str, ...]
    subject_ids: tuple[str, ...]
    electrode_types: tuple[str, ...]
    checkpoint_groups: tuple[str, ...]
    run_ids: tuple[str, ...]
    base_input_sha256s: tuple[str, ...]


@dataclass(frozen=True)
class OpaqueTokenDomain:
    phase: str
    checkpoint_group: str
    purpose: str
    raw_asset_fingerprint_sha256: str
    seal_decision_sha256: str


def opaque_metadata_row_token(
    sample_id: str,
    *,
    secret_key: bytes,
    domain: OpaqueTokenDomain,
) -> str:
    """Tokenize one raw identity with phase, group, and purpose separation."""

    if not isinstance(secret_key, bytes) or len(secret_key) < 32:
        raise ValueError("Opaque query-token keys must contain at least 32 bytes.")
    if not sample_id:
        raise ValueError("Cannot tokenize an empty sample identity.")
    prefixes = {"evaluation_query": "q", "evaluation_support": "s", "source_fit": "f"}
    if domain.purpose not in prefixes:
        raise ValueError("Opaque token purpose is not part of the sealed input contract.")
    if domain.phase not in {"source_development", "held_participant_evaluation"}:
        raise ValueError("Opaque token phase is invalid.")
    if not domain.checkpoint_group:
        raise ValueError("Opaque token checkpoint group must be nonempty.")
    for value in (domain.raw_asset_fingerprint_sha256, domain.seal_decision_sha256):
        if not re.fullmatch(r"[0-9a-f]{64}", value):
            raise ValueError("Opaque token domain hashes must be lowercase SHA-256 values.")
    domain_payload = json.dumps(
        {
            "schema": "cfeg.metadata-calibration-token-domain.v1",
            "candidate_id": "metadata-calibration-efficiency-v1",
            "dataset_revision": "wearable_v3",
            "raw_asset_fingerprint_sha256": domain.raw_asset_fingerprint_sha256,
            "seal_decision_sha256": domain.seal_decision_sha256,
            "phase": domain.phase,
            "checkpoint_group": domain.checkpoint_group,
            "purpose": domain.purpose,
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    domain_key = hmac.new(secret_key, b"KEY\0" + domain_payload, hashlib.sha256).digest()
    encoded_identity = sample_id.encode("utf-8")
    row_payload = (
        b"ROW\0"
        + len(encoded_identity).to_bytes(8, byteorder="big", signed=False)
        + encoded_identity
    )
    digest = hmac.new(domain_key, row_payload, hashlib.sha256).hexdigest()
    return f"{prefixes[domain.purpose]}_{digest}"


def opaque_query_token(
    sample_id: str,
    *,
    secret_key: bytes,
    domain: OpaqueTokenDomain,
) -> str:
    """Create a query token only inside an explicit evaluation-query domain."""

    if domain.purpose != "evaluation_query":
        raise ValueError("opaque_query_token requires purpose='evaluation_query'.")
    return opaque_metadata_row_token(sample_id, secret_key=secret_key, domain=domain)


def token_secret_commitment_sha256(secret_key: bytes) -> str:
    if not isinstance(secret_key, bytes) or len(secret_key) < 32:
        raise ValueError("Opaque token secrets must contain at least 32 bytes.")
    return hashlib.sha256(
        b"cfeg.metadata-calibration-secret-commitment.v1\0" + secret_key
    ).hexdigest()


class UnlabeledQueryReader:
    """Read a sealed query view that cannot expose query labels through its API."""

    def __init__(
        self,
        signals_h5: str | Path,
        query_view: pd.DataFrame,
        *,
        persistent_hdf5_handle: bool = False,
    ) -> None:
        leaked = _FORBIDDEN_COLUMNS.intersection(query_view.columns)
        if leaked:
            raise ValueError(f"Sealed query view contains forbidden columns: {sorted(leaked)}.")
        missing = _QUERY_COLUMNS - set(query_view.columns)
        extra = set(query_view.columns) - _QUERY_COLUMNS
        if missing or extra:
            raise ValueError(
                "Sealed query view must use its exact allowlist; "
                f"missing={sorted(missing)}, extra={sorted(extra)}."
            )
        frame = pd.DataFrame.from_records(
            copy.deepcopy(
                query_view.loc[:, sorted(_QUERY_COLUMNS)].to_dict(orient="records")
            ),
            columns=sorted(_QUERY_COLUMNS),
        )
        frame["query_token"] = frame["query_token"].astype(str)
        if frame["query_token"].duplicated().any() or not frame["query_token"].map(
            lambda value: bool(_QUERY_TOKEN.fullmatch(value))
        ).all():
            raise ValueError("Query tokens must be unique opaque q_<HMAC-SHA256> values.")
        indices = pd.to_numeric(frame["signal_index"], errors="raise").to_numpy(dtype=float)
        if (
            not np.isfinite(indices).all()
            or not np.equal(indices, np.floor(indices)).all()
            or (indices < 0).any()
            or len(set(indices.astype(int).tolist())) != len(indices)
        ):
            raise ValueError("Sealed query signal indices must be unique nonnegative integers.")
        frame["signal_index"] = indices.astype(int)
        frame["base_input_sha256"] = frame["base_input_sha256"].astype(str)
        if not frame["base_input_sha256"].str.fullmatch(r"[0-9a-f]{64}").all():
            raise ValueError("Every sealed query row must bind one base input SHA-256.")
        if not np.array_equal(frame["signal_index"].to_numpy(), np.arange(len(frame))) or frame[
            "query_token"
        ].tolist() != sorted(frame["query_token"].tolist()):
            raise ValueError(
                "Sealed query rows, local signal indices and signals must use opaque-token order, "
                "not source row order."
            )
        if set(frame["electrode_type"].astype(str)) != {"dry", "wet"}:
            raise ValueError("Sealed query view must contain exact dry and wet interfaces.")
        self._signals_h5 = Path(signals_h5).resolve()
        if not self._signals_h5.is_file():
            raise FileNotFoundError(self._signals_h5)
        with h5py.File(self._signals_h5, "r") as handle:
            if set(handle.keys()) != {"x", "channel_mask"}:
                raise ValueError(
                    "Sealed query signal artifact must contain only x and channel_mask."
                )
            for name in ("x", "channel_mask"):
                if not isinstance(handle.get(name, getlink=True), h5py.HardLink):
                    raise TypeError("Sealed query datasets must use local hard links.")
                dataset = handle[name]
                if not isinstance(dataset, h5py.Dataset):
                    raise TypeError("Sealed query signal entries must be HDF5 datasets.")
                if dataset.is_virtual or dataset.external:
                    raise ValueError("Sealed query datasets cannot be virtual or external.")
            x_dataset = handle["x"]
            mask_dataset = handle["channel_mask"]
            if x_dataset.dtype != np.dtype(np.float32) or x_dataset.ndim != 3:
                raise ValueError("Sealed query x must use exact float32 rank-three storage.")
            if x_dataset.shape[1:] != (64, 400):
                raise ValueError("Sealed query x must have exact shape [N,64,400].")
            if mask_dataset.dtype != np.dtype(bool) or mask_dataset.ndim != 2:
                raise ValueError("Sealed query channel_mask must use exact boolean rank-two storage.")
            n_signals = len(x_dataset)
            if mask_dataset.shape != (n_signals, 64):
                raise ValueError(
                    "Sealed query channel_mask must align with x as exact shape [N,64]."
                )
        if sorted(frame["signal_index"].tolist()) != list(range(n_signals)):
            raise ValueError(
                "Sealed query signal indices must cover the local no-y artifact exactly."
            )
        self._frame = frame.reset_index(drop=True)
        self._reader = HDF5SampleReader(persistent=persistent_hdf5_handle)

    def __len__(self) -> int:
        return len(self._frame)

    def __getitem__(self, index: int) -> UnlabeledQuerySample:
        row = self._frame.iloc[int(index)]
        x, mask = self._reader.read_unlabeled(self._signals_h5, int(row["signal_index"]))
        observed_base_sha256 = metadata_calibration_base_input_sha256(x, mask)
        if observed_base_sha256 != str(row["base_input_sha256"]):
            raise ValueError("Sealed query signal bytes differ from their row binding.")
        signal_std = _numeric_vector(
            row["query_signal_std_by_channel"],
            dtype=np.float32,
            name="query_signal_std_by_channel",
        )
        signal_std_missing = _boolean_vector(
            row["query_signal_std_missing_by_channel"],
            name="query_signal_std_missing_by_channel",
        )
        impedance = _numeric_vector(
            row["impedance_kohm_by_channel"],
            dtype=np.float32,
            name="impedance_kohm_by_channel",
        )
        impedance_missing = _boolean_vector(
            row["impedance_missing_by_channel"],
            name="impedance_missing_by_channel",
        )
        canonical_channel_ids = _numeric_vector(
            row["canonical_channel_ids"],
            dtype=np.int64,
            name="canonical_channel_ids",
        )
        expected_channel_ids = np.zeros(64, dtype=np.int64)
        for channel_id in WEARABLE_CALIBRATION_CHANNEL_IDS:
            expected_channel_ids[channel_id - 1] = channel_id
        if not np.array_equal(canonical_channel_ids, expected_channel_ids) or not np.array_equal(
            mask,
            expected_channel_ids > 0,
        ):
            raise ValueError(
                "Sealed query channel IDs and mask must match the official wearable layout."
            )
        if str(row["run_id"]) not in {f"block{index:02d}" for index in range(6, 11)}:
            raise ValueError("Sealed evaluation queries must come from exact block06-block10.")
        sfreq = float(row["sfreq_processed"])
        if not np.isfinite(sfreq) or sfreq != 200.0:
            raise ValueError("Sealed query sampling rate must be exactly 200 Hz.")
        if not (
            signal_std.shape
            == signal_std_missing.shape
            == impedance.shape
            == impedance_missing.shape
            == (64,)
        ):
            raise ValueError("Query QC, impedance and missing-mask widths must match.")
        if np.any(signal_std[signal_std_missing] != 0.0) or np.any(
            impedance[impedance_missing] != 0.0
        ):
            raise ValueError("Missing query values must be safely zero-filled.")
        return UnlabeledQuerySample(
            x=x,
            channel_mask=mask,
            query_token=str(row["query_token"]),
            subject_id=str(row["subject_id"]),
            electrode_type=str(row["electrode_type"]),
            checkpoint_group=str(row["checkpoint_group"]),
            run_id=str(row["run_id"]),
            canonical_channel_ids=canonical_channel_ids,
            sfreq=sfreq,
            query_signal_std_by_channel=signal_std,
            query_signal_std_missing_by_channel=signal_std_missing,
            impedance_kohm_by_channel=impedance,
            impedance_missing_by_channel=impedance_missing,
            base_input_sha256=observed_base_sha256,
        )

    def close(self) -> None:
        self._reader.close()


def collate_metadata_calibration_queries(
    samples: list[UnlabeledQuerySample],
) -> UnlabeledQueryBatch:
    """Collate query samples without introducing ``y`` or ``sample_id``."""

    if not samples:
        raise ValueError("Cannot collate an empty metadata-calibration query batch.")
    x_array = np.stack([sample.x for sample in samples]).astype(np.float32, copy=False)
    channel_mask = np.stack([sample.channel_mask for sample in samples]).astype(bool, copy=False)
    channel_ids = np.stack([sample.canonical_channel_ids for sample in samples]).astype(
        np.int64, copy=False
    )
    if x_array.ndim != 3 or channel_mask.shape != x_array.shape[:2]:
        raise ValueError("Query EEG and channel masks do not align.")
    if channel_ids.shape != channel_mask.shape:
        raise ValueError("Query canonical channel IDs do not align with EEG channels.")
    if not np.isfinite(x_array[channel_mask]).all():
        raise ValueError("Active query EEG samples must be finite.")

    query_qc, query_missing = _normalized_channel_feature(
        samples,
        value_name="query_signal_std_by_channel",
        missing_name="query_signal_std_missing_by_channel",
        channel_mask=channel_mask,
    )
    impedance, impedance_missing = _normalized_channel_feature(
        samples,
        value_name="impedance_kohm_by_channel",
        missing_name="impedance_missing_by_channel",
        channel_mask=channel_mask,
    )
    sfreq = np.asarray([sample.sfreq for sample in samples], dtype=np.float32)
    if not np.isfinite(sfreq).all() or (sfreq <= 0.0).any():
        raise ValueError("Query sampling rates must be positive and finite.")
    cond: dict[str, Any] = {
        "channel_ids": torch.from_numpy(channel_ids),
        "channel_mask": torch.from_numpy(channel_mask),
        "sfreq_processed_float": torch.from_numpy(sfreq),
        "channel_query_qc": torch.from_numpy(query_qc),
        "channel_query_qc_missing": torch.from_numpy(query_missing),
        "channel_impedance": torch.from_numpy(impedance),
        "channel_impedance_missing": torch.from_numpy(impedance_missing),
        "metadata_contract_version": METADATA_CONTRACT_V04_DEV,
    }
    return UnlabeledQueryBatch(
        x=torch.from_numpy(x_array),
        cond=cond,
        query_tokens=tuple(sample.query_token for sample in samples),
        subject_ids=tuple(sample.subject_id for sample in samples),
        electrode_types=tuple(sample.electrode_type for sample in samples),
        checkpoint_groups=tuple(sample.checkpoint_group for sample in samples),
        run_ids=tuple(sample.run_id for sample in samples),
        base_input_sha256s=tuple(sample.base_input_sha256 for sample in samples),
    )


def _numeric_vector(value: object, *, dtype: np.dtype, name: str) -> np.ndarray:
    numeric = np.asarray(value, dtype=np.float64)
    if numeric.ndim != 1 or not np.isfinite(numeric).all():
        raise ValueError(f"{name} must be one finite vector.")
    if np.issubdtype(np.dtype(dtype), np.integer) and not np.equal(
        numeric, np.floor(numeric)
    ).all():
        raise ValueError(f"{name} must contain exact integer values before casting.")
    return numeric.astype(dtype)


def _boolean_vector(value: object, *, name: str) -> np.ndarray:
    array = np.asarray(value)
    if array.ndim != 1 or array.dtype.kind != "b":
        raise ValueError(f"{name} must be one boolean vector.")
    return array.astype(bool, copy=False)


def _normalized_channel_feature(
    samples: list[UnlabeledQuerySample],
    *,
    value_name: str,
    missing_name: str,
    channel_mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    values = np.stack([getattr(sample, value_name) for sample in samples]).astype(
        np.float64, copy=False
    )
    missing = np.stack([getattr(sample, missing_name) for sample in samples]).astype(
        bool, copy=False
    )
    if values.shape != channel_mask.shape or missing.shape != channel_mask.shape:
        raise ValueError(f"{value_name} and its missing mask must align with EEG channels.")
    missing = missing | ~channel_mask
    available = ~missing
    if not np.isfinite(values[available]).all() or (values[available] < 0.0).any():
        raise ValueError(f"Available {value_name} values must be finite and nonnegative.")
    normalized = np.zeros_like(values, dtype=np.float32)
    normalized[available] = np.clip(
        np.log1p(values[available]) / np.log1p(100.0), 0.0, 2.0
    ).astype(np.float32)
    return normalized, missing
