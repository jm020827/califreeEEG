from __future__ import annotations

import json
import os
import re
import stat
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import h5py
import numpy as np
import pandas as pd
import torch

from cfeg.constants import METADATA_CONTRACT_V04_DEV
from cfeg.data.metadata_calibration_features import WEARABLE_CALIBRATION_CHANNEL_IDS
from cfeg.data.metadata_calibration_interventions import (
    metadata_calibration_base_input_sha256,
)
from cfeg.identity import canonical_identity_sha256

_TOKEN = re.compile(r"[qsf]_[0-9a-f]{64}")
_BASE_COLUMNS = {
    "base_input_sha256",
    "canonical_channel_ids",
    "checkpoint_group",
    "electrode_type",
    "impedance_kohm_by_channel",
    "impedance_missing_by_channel",
    "query_signal_std_by_channel",
    "query_signal_std_missing_by_channel",
    "run_id",
    "sfreq_processed",
    "signal_index",
    "subject_id",
}


@dataclass(frozen=True)
class SealedTensorRows:
    """In-memory view of a sealed no-raw-identity signal artifact."""

    x: np.ndarray
    channel_mask: np.ndarray
    frame: pd.DataFrame
    token_column: str
    labels: np.ndarray | None
    trials_per_episode: int

    def episode_keys(self) -> tuple[tuple[str, str], ...]:
        pairs = self.frame.loc[:, ["subject_id", "electrode_type"]].drop_duplicates()
        return tuple(
            (str(row.subject_id), str(row.electrode_type))
            for row in pairs.sort_values(
                ["subject_id", "electrode_type"], kind="mergesort"
            ).itertuples(index=False)
        )

    def indices_for_episode(self, subject_id: str, electrode_type: str) -> np.ndarray:
        mask = self.frame["subject_id"].astype(str).eq(str(subject_id)) & self.frame[
            "electrode_type"
        ].astype(str).eq(str(electrode_type))
        indices = np.flatnonzero(mask.to_numpy())
        if len(indices) != self.trials_per_episode:
            raise ValueError(
                "A sealed participant-interface episode has an unexpected trial count."
            )
        return indices


@dataclass(frozen=True)
class SealedTargetGroup:
    support: SealedTensorRows
    support_rows: pd.DataFrame
    query: SealedTensorRows
    expected_query: pd.DataFrame
    block_context: pd.DataFrame


@dataclass(frozen=True)
class TensorRowBatch:
    x: torch.Tensor
    cond: dict[str, Any]
    tokens: tuple[str, ...]
    subject_ids: tuple[str, ...]
    electrode_types: tuple[str, ...]
    run_ids: tuple[str, ...]
    base_input_sha256s: tuple[str, ...]
    labels: torch.Tensor | None


def load_sealed_source_fit(group_root: str | Path) -> SealedTensorRows:
    root = _exact_group_root(group_root)
    rows = _load_tensor_rows(
        root / "source_fit/labeled_view.jsonl",
        root / "source_fit/signals.h5",
        token_column="source_token",
        token_prefix="f_",
        include_label=True,
        trials_per_episode=120,
    )
    _validate_episode_blocks(rows, expected_blocks=range(1, 11))
    return rows


def load_sealed_target_group(group_root: str | Path) -> SealedTargetGroup:
    root = _exact_group_root(group_root)
    support = _load_tensor_rows(
        root / "target_support/model_view.jsonl",
        root / "target_support/signals.h5",
        token_column="support_token",
        token_prefix="s_",
        include_label=False,
        trials_per_episode=60,
    )
    query = _load_tensor_rows(
        root / "target_query/model_view.jsonl",
        root / "target_query/signals.h5",
        token_column="query_token",
        token_prefix="q_",
        include_label=False,
        trials_per_episode=60,
    )
    support_rows = _read_exact_jsonl(
        root / "target_support/support_labels.jsonl",
        {
            "base_input_sha256",
            "budget",
            "checkpoint_group",
            "electrode_type",
            "run_id",
            "subject_id",
            "support_label",
            "support_token",
        },
    )
    expected_query = _read_exact_jsonl(
        root / "target_query/expected_unlabeled.jsonl",
        {
            "checkpoint_group",
            "electrode_type",
            "query_identity_sha256",
            "query_token",
            "subject_id",
        },
    )
    block_context = _read_exact_jsonl(
        root / "block_context.jsonl",
        {
            "dataset_id",
            "electrode_type",
            "impedance_channel_ids",
            "impedance_kohm_by_channel",
            "run_id",
            "subject_id",
        },
    )
    _validate_target_group(support, support_rows, query, expected_query, block_context)
    return SealedTargetGroup(
        support=support,
        support_rows=support_rows,
        query=query,
        expected_query=expected_query,
        block_context=block_context,
    )


def tensor_batch(
    rows: SealedTensorRows,
    indices: np.ndarray | list[int],
    *,
    device: torch.device | str,
) -> TensorRowBatch:
    selected = np.asarray(indices, dtype=np.int64)
    if selected.ndim != 1 or len(selected) == 0:
        raise ValueError("A sealed tensor batch requires nonempty one-dimensional indices.")
    if selected.min() < 0 or selected.max() >= len(rows.frame):
        raise IndexError("A sealed tensor batch index is out of range.")
    frame = rows.frame.iloc[selected]
    x = torch.from_numpy(np.asarray(rows.x[selected], dtype=np.float32)).to(device)
    mask_array = np.asarray(rows.channel_mask[selected], dtype=bool)
    channel_ids_array = np.stack(frame["canonical_channel_ids"].map(np.asarray)).astype(
        np.int64
    )
    query_values, query_missing = _normalize_channel_feature(
        frame["query_signal_std_by_channel"],
        frame["query_signal_std_missing_by_channel"],
        mask_array,
        name="query_signal_std_by_channel",
    )
    impedance_values, impedance_missing = _normalize_channel_feature(
        frame["impedance_kohm_by_channel"],
        frame["impedance_missing_by_channel"],
        mask_array,
        name="impedance_kohm_by_channel",
    )
    target = torch.device(device)
    cond: dict[str, Any] = {
        "channel_ids": torch.from_numpy(channel_ids_array).to(target),
        "channel_mask": torch.from_numpy(mask_array).to(target),
        "sfreq_processed_float": torch.as_tensor(
            frame["sfreq_processed"].to_numpy(dtype=np.float32), device=target
        ),
        "channel_query_qc": torch.from_numpy(query_values).to(target),
        "channel_query_qc_missing": torch.from_numpy(query_missing).to(target),
        "channel_impedance": torch.from_numpy(impedance_values).to(target),
        "channel_impedance_missing": torch.from_numpy(impedance_missing).to(target),
        "metadata_contract_version": METADATA_CONTRACT_V04_DEV,
    }
    labels = (
        None
        if rows.labels is None
        else torch.from_numpy(np.asarray(rows.labels[selected], dtype=np.int64)).to(target)
    )
    return TensorRowBatch(
        x=x,
        cond=cond,
        tokens=tuple(frame[rows.token_column].astype(str)),
        subject_ids=tuple(frame["subject_id"].astype(str)),
        electrode_types=tuple(frame["electrode_type"].astype(str)),
        run_ids=tuple(frame["run_id"].astype(str)),
        base_input_sha256s=tuple(frame["base_input_sha256"].astype(str)),
        labels=labels,
    )


def extract_official_wearable_channels(x: np.ndarray) -> np.ndarray:
    values = np.asarray(x)
    if values.ndim != 3 or values.shape[1:] != (64, 400):
        raise ValueError("Wearable baseline EEG must have shape [trial,64,400].")
    positions = np.asarray(WEARABLE_CALIBRATION_CHANNEL_IDS, dtype=np.int64) - 1
    selected = np.asarray(values[:, positions, :], dtype=np.float64)
    if selected.shape[1:] != (8, 400) or not np.isfinite(selected).all():
        raise ValueError("Official wearable channel extraction produced invalid EEG.")
    return selected


def support_indices_for_budget(
    target: SealedTargetGroup,
    *,
    subject_id: str,
    electrode_type: str,
    budget: int,
) -> np.ndarray:
    if budget == 0:
        return np.empty(0, dtype=np.int64)
    if budget not in {1, 3, 5}:
        raise ValueError("Calibration budget must be one of 0,1,3,5.")
    labels = target.support_rows
    selected = labels.loc[
        labels["subject_id"].astype(str).eq(str(subject_id))
        & labels["electrode_type"].astype(str).eq(str(electrode_type))
        & pd.to_numeric(labels["budget"], errors="raise").eq(budget)
    ].copy()
    if len(selected) != budget * 12:
        raise ValueError("Target support budget does not contain exact complete blocks.")
    token_to_index = {
        str(token): index
        for index, token in enumerate(target.support.frame["support_token"].astype(str))
    }
    try:
        indices = np.asarray(
            [token_to_index[str(value)] for value in selected["support_token"]],
            dtype=np.int64,
        )
    except KeyError as error:
        raise ValueError("A support label token has no sealed signal row.") from error
    # Preserve run and class order for auditable complete-block calibration.
    order = np.lexsort(
        (
            pd.to_numeric(selected["support_label"], errors="raise").to_numpy(),
            selected["run_id"].astype(str).to_numpy(),
        )
    )
    return indices[order]


def support_labels_for_budget(
    target: SealedTargetGroup,
    *,
    subject_id: str,
    electrode_type: str,
    budget: int,
) -> np.ndarray:
    if budget == 0:
        return np.empty(0, dtype=np.int64)
    labels = target.support_rows.loc[
        target.support_rows["subject_id"].astype(str).eq(str(subject_id))
        & target.support_rows["electrode_type"].astype(str).eq(str(electrode_type))
        & pd.to_numeric(target.support_rows["budget"], errors="raise").eq(budget)
    ].copy()
    order = np.lexsort(
        (
            pd.to_numeric(labels["support_label"], errors="raise").to_numpy(),
            labels["run_id"].astype(str).to_numpy(),
        )
    )
    result = pd.to_numeric(labels["support_label"], errors="raise").to_numpy(
        dtype=np.int64
    )[order]
    if result.shape != (budget * 12,):
        raise ValueError("Target support labels do not match the requested budget.")
    return result


def _load_tensor_rows(
    view_path: Path,
    signals_path: Path,
    *,
    token_column: str,
    token_prefix: Literal["q_", "s_", "f_"],
    include_label: bool,
    trials_per_episode: int,
) -> SealedTensorRows:
    fields = set(_BASE_COLUMNS) | {token_column}
    if include_label:
        fields.add("label")
    frame = _read_exact_jsonl(view_path, fields)
    if len(frame) == 0 or frame[token_column].astype(str).duplicated().any():
        raise ValueError("Sealed model view must have unique nonempty row tokens.")
    if not frame[token_column].astype(str).map(
        lambda value: bool(_TOKEN.fullmatch(value)) and value.startswith(token_prefix)
    ).all():
        raise ValueError("Sealed model view has an invalid purpose-bound token.")
    signal_indices = pd.to_numeric(frame["signal_index"], errors="raise").to_numpy()
    if not np.equal(signal_indices, np.floor(signal_indices)).all() or not np.array_equal(
        signal_indices.astype(np.int64), np.arange(len(frame))
    ):
        raise ValueError("Sealed signal indices must be contiguous and row aligned.")
    x, channel_mask = load_exact_sealed_signal_h5(
        signals_path,
        expected_rows=len(frame),
    )
    if x.shape != (len(frame), 64, 400) or channel_mask.shape != (len(frame), 64):
        raise ValueError("Sealed signal artifact shape differs from its model view.")
    if not np.isfinite(x[channel_mask]).all() or not np.all(x[~channel_mask] == 0.0):
        raise ValueError("Sealed EEG has invalid active values or nonzero inactive slots.")
    expected_ids = np.zeros(64, dtype=np.int64)
    for channel_id in WEARABLE_CALIBRATION_CHANNEL_IDS:
        expected_ids[channel_id - 1] = channel_id
    for index, row in frame.iterrows():
        channel_ids = np.asarray(row["canonical_channel_ids"], dtype=np.int64)
        if not np.array_equal(channel_ids, expected_ids) or not np.array_equal(
            channel_mask[index], expected_ids > 0
        ):
            raise ValueError("Sealed channel layout differs from the official wearable layout.")
        if metadata_calibration_base_input_sha256(x[index], channel_mask[index]) != str(
            row["base_input_sha256"]
        ):
            raise ValueError("Sealed signal bytes differ from their base-input binding.")
        if float(row["sfreq_processed"]) != 200.0:
            raise ValueError("Sealed wearable sampling rate must be exactly 200 Hz.")
    labels = None
    if include_label:
        numeric = pd.to_numeric(frame["label"], errors="raise").to_numpy()
        if not np.equal(numeric, np.floor(numeric)).all():
            raise ValueError("Source labels must be exact integers.")
        labels = numeric.astype(np.int64)
        if labels.min() < 0 or labels.max() >= 12:
            raise ValueError("Source labels fall outside the frozen 12-class vocabulary.")
    if len(set(frame["checkpoint_group"].astype(str))) != 1:
        raise ValueError("A sealed worker view must bind one checkpoint group.")
    by_subject = frame.groupby("subject_id", sort=False)["electrode_type"].agg(
        lambda values: set(map(str, values))
    )
    if by_subject.empty or not by_subject.map(lambda values: values == {"dry", "wet"}).all():
        raise ValueError("Every sealed participant must contain exact dry and wet interfaces.")
    return SealedTensorRows(
        x=x,
        channel_mask=channel_mask,
        frame=frame.reset_index(drop=True),
        token_column=token_column,
        labels=labels,
        trials_per_episode=int(trials_per_episode),
    )


def _read_exact_jsonl(path: Path, fields: set[str]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"Invalid JSONL at {path}:{line_number}.") from error
            if not isinstance(row, dict) or set(row) != fields:
                raise ValueError(f"Sealed JSONL schema drifted at {path}:{line_number}.")
            rows.append(row)
    return pd.DataFrame(rows, columns=sorted(fields))


def _normalize_channel_feature(
    values_series: pd.Series,
    missing_series: pd.Series,
    channel_mask: np.ndarray,
    *,
    name: str,
) -> tuple[np.ndarray, np.ndarray]:
    values = np.stack(values_series.map(lambda value: np.asarray(value, dtype=np.float64)))
    missing = np.stack(missing_series.map(lambda value: np.asarray(value, dtype=bool)))
    if values.shape != channel_mask.shape or missing.shape != channel_mask.shape:
        raise ValueError(f"{name} and its missingness must align with the channel mask.")
    missing = missing | ~channel_mask
    available = ~missing
    if not np.isfinite(values[available]).all() or (values[available] < 0.0).any():
        raise ValueError(f"Available {name} values must be finite and nonnegative.")
    normalized = np.zeros_like(values, dtype=np.float32)
    normalized[available] = np.clip(
        np.log1p(values[available]) / np.log1p(100.0), 0.0, 2.0
    ).astype(np.float32)
    return normalized, missing


def _validate_target_group(
    support: SealedTensorRows,
    support_rows: pd.DataFrame,
    query: SealedTensorRows,
    expected_query: pd.DataFrame,
    block_context: pd.DataFrame,
) -> None:
    groups = (
        set(support.frame["checkpoint_group"].astype(str))
        | set(query.frame["checkpoint_group"].astype(str))
        | set(support_rows["checkpoint_group"].astype(str))
        | set(expected_query["checkpoint_group"].astype(str))
    )
    if len(groups) != 1:
        raise ValueError("Target artifacts must bind one exact checkpoint group.")
    support_cells = set(support.episode_keys())
    query_cells = set(query.episode_keys())
    if support_cells != query_cells:
        raise ValueError("Target support and query participant-interface cells differ.")
    if set(support.frame["support_token"].astype(str)) & set(
        query.frame["query_token"].astype(str)
    ):
        raise ValueError("Target support and query tokens overlap.")
    query_join = query.frame.loc[
        :, ["query_token", "subject_id", "electrode_type", "checkpoint_group"]
    ].merge(
        expected_query,
        on=["query_token", "subject_id", "electrode_type", "checkpoint_group"],
        how="outer",
        validate="one_to_one",
        indicator=True,
    )
    if not query_join["_merge"].eq("both").all():
        raise ValueError("Expected query rows differ from the sealed model view.")
    if set(support_rows["support_token"].astype(str)) != set(
        support.frame["support_token"].astype(str)
    ):
        raise ValueError("Support label rows differ from the sealed support model view.")
    support_identity_columns = (
        "subject_id",
        "electrode_type",
        "checkpoint_group",
        "run_id",
        "base_input_sha256",
    )
    support_binding = support_rows.merge(
        support.frame.loc[:, ["support_token", *support_identity_columns]],
        on="support_token",
        how="left",
        suffixes=("", "_view"),
        validate="many_to_one",
        indicator=True,
    )
    if not support_binding["_merge"].eq("both").all() or any(
        not support_binding[column]
        .astype(str)
        .eq(support_binding[f"{column}_view"].astype(str))
        .all()
        for column in support_identity_columns
    ):
        raise ValueError("Support labels are not bound to their exact sealed signal rows.")
    token_consistency = support_rows.groupby("support_token", sort=False).agg(
        labels=("support_label", "nunique"),
        subjects=("subject_id", "nunique"),
        interfaces=("electrode_type", "nunique"),
        groups=("checkpoint_group", "nunique"),
        runs=("run_id", "nunique"),
        inputs=("base_input_sha256", "nunique"),
    )
    if not token_consistency.eq(1).all().all():
        raise ValueError("A nested support token changes identity or label across budgets.")
    _validate_nested_support_labels(support_rows)
    budgets = pd.to_numeric(support_rows["budget"], errors="raise")
    labels = pd.to_numeric(support_rows["support_label"], errors="raise")
    if set(budgets) != {1, 3, 5} or not np.equal(labels, np.floor(labels)).all() or (
        (labels < 0) | (labels >= 12)
    ).any():
        raise ValueError("Support labels do not contain exact nested budgets 1,3,5.")
    support_key = support_rows.loc[
        :, ["budget", "support_token", "subject_id", "electrode_type"]
    ]
    if support_key.duplicated().any():
        raise ValueError("Support label rows contain duplicate budget-token identities.")
    _validate_episode_blocks(support, expected_blocks=range(1, 6))
    _validate_episode_blocks(query, expected_blocks=range(6, 11))
    cells = query.frame.loc[:, ["subject_id", "electrode_type"]].drop_duplicates()
    for row in cells.itertuples(index=False):
        support.indices_for_episode(str(row.subject_id), str(row.electrode_type))
        query.indices_for_episode(str(row.subject_id), str(row.electrode_type))
        for budget in (1, 3, 5):
            indices = support_indices_for_budget(
                SealedTargetGroup(
                    support=support,
                    support_rows=support_rows,
                    query=query,
                    expected_query=expected_query,
                    block_context=block_context,
                ),
                subject_id=str(row.subject_id),
                electrode_type=str(row.electrode_type),
                budget=budget,
            )
            expected_runs = {f"block{index:02d}" for index in range(1, budget + 1)}
            if set(support.frame.iloc[indices]["run_id"].astype(str)) != expected_runs:
                raise ValueError("Support budget is not the frozen nested block prefix.")
        episode = query.frame.loc[
            query.frame["subject_id"].astype(str).eq(str(row.subject_id))
            & query.frame["electrode_type"].astype(str).eq(str(row.electrode_type))
        ]
        expected_identity = canonical_identity_sha256(
            episode["query_token"].astype(str)
        )
        identities = expected_query.loc[
            expected_query["subject_id"].astype(str).eq(str(row.subject_id))
            & expected_query["electrode_type"].astype(str).eq(str(row.electrode_type)),
            "query_identity_sha256",
        ].astype(str)
        if set(identities) != {expected_identity}:
            raise ValueError("Expected-query identity does not bind the exact episode tokens.")
    expected_context = len(cells) * 10
    if len(block_context) != expected_context:
        raise ValueError("Sealed block context does not cover every target block exactly once.")
    context_keys = block_context.loc[:, ["subject_id", "electrode_type", "run_id"]]
    if context_keys.duplicated().any() or set(block_context["dataset_id"].astype(str)) != {
        "wearable"
    }:
        raise ValueError("Sealed block context has duplicate keys or a wrong dataset.")
    official = list(WEARABLE_CALIBRATION_CHANNEL_IDS)
    for row in cells.itertuples(index=False):
        selected = block_context.loc[
            block_context["subject_id"].astype(str).eq(str(row.subject_id))
            & block_context["electrode_type"].astype(str).eq(str(row.electrode_type))
        ]
        if set(selected["run_id"].astype(str)) != {
            f"block{index:02d}" for index in range(1, 11)
        }:
            raise ValueError("Block context must contain exact blocks 01 through 10.")
        for context in selected.to_dict(orient="records"):
            if list(context["impedance_channel_ids"]) != official:
                raise ValueError("Block context channel IDs differ from the official eight.")
            values = np.asarray(context["impedance_kohm_by_channel"], dtype=np.float64)
            if values.shape != (8,) or not np.isfinite(values).all() or (values < 0).any():
                raise ValueError("Block context impedance must be eight finite nonnegative values.")


def _validate_episode_blocks(
    rows: SealedTensorRows,
    *,
    expected_blocks: range,
) -> None:
    block_names = {f"block{index:02d}" for index in expected_blocks}
    for subject_id, electrode_type in rows.episode_keys():
        indices = rows.indices_for_episode(subject_id, electrode_type)
        episode = rows.frame.iloc[indices]
        if set(episode["run_id"].astype(str)) != block_names:
            raise ValueError("A sealed episode contains the wrong frozen blocks.")
        counts = episode.groupby("run_id", sort=False).size()
        if not counts.eq(12).all():
            raise ValueError("Each sealed episode block must contain exactly twelve trials.")
        if rows.labels is not None:
            labeled = episode.assign(label=rows.labels[indices])
            for _run_id, block in labeled.groupby("run_id", sort=False):
                if sorted(block["label"].astype(int).tolist()) != list(range(12)):
                    raise ValueError("Each source block must contain every class exactly once.")
                impedance = [
                    (
                        np.asarray(value, dtype=np.float64),
                        np.asarray(missing, dtype=bool),
                    )
                    for value, missing in zip(
                        block["impedance_kohm_by_channel"],
                        block["impedance_missing_by_channel"],
                    )
                ]
                first_value, first_missing = impedance[0]
                if any(
                    not np.array_equal(first_value, value)
                    or not np.array_equal(first_missing, missing)
                    for value, missing in impedance[1:]
                ):
                    raise ValueError("Source impedance must remain constant within each block.")


def _validate_nested_support_labels(support_rows: pd.DataFrame) -> None:
    for (_subject, _interface), episode in support_rows.groupby(
        ["subject_id", "electrode_type"],
        sort=False,
    ):
        token_sets: dict[int, set[str]] = {}
        for budget in (1, 3, 5):
            selected = episode.loc[
                pd.to_numeric(episode["budget"], errors="raise").eq(budget)
            ]
            expected_runs = {
                f"block{index:02d}" for index in range(1, budget + 1)
            }
            if len(selected) != budget * 12 or set(
                selected["run_id"].astype(str)
            ) != expected_runs:
                raise ValueError("A support budget is not an exact class-complete block prefix.")
            for _run_id, block in selected.groupby("run_id", sort=False):
                labels = sorted(
                    pd.to_numeric(block["support_label"], errors="raise")
                    .astype(int)
                    .tolist()
                )
                if labels != list(range(12)):
                    raise ValueError("Each support block must contain every class exactly once.")
            token_sets[budget] = set(selected["support_token"].astype(str))
        if not token_sets[1] < token_sets[3] or not token_sets[3] < token_sets[5]:
            raise ValueError("Support tokens must be strictly nested from k1 to k3 to k5.")


def load_exact_sealed_signal_h5(
    path: Path,
    *,
    expected_rows: int,
) -> tuple[np.ndarray, np.ndarray]:
    observed = path.lstat()
    if stat.S_ISLNK(observed.st_mode) or not stat.S_ISREG(observed.st_mode):
        raise ValueError("Sealed signal artifact must be one regular non-symlink file.")
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    try:
        opened = os.fstat(descriptor)
        if not stat.S_ISREG(opened.st_mode) or opened.st_nlink != 1 or (
            opened.st_dev,
            opened.st_ino,
        ) != (observed.st_dev, observed.st_ino):
            raise ValueError("Sealed signal artifact changed or is hard-linked.")
        with os.fdopen(descriptor, "rb", closefd=False) as file_object, h5py.File(
            file_object,
            "r",
            driver="fileobj",
        ) as handle:
            if set(handle) != {"channel_mask", "x"}:
                raise ValueError(
                    "Sealed signal artifact must contain exactly x and channel_mask."
                )
            for name in ("x", "channel_mask"):
                if not isinstance(handle.get(name, getlink=True), h5py.HardLink):
                    raise TypeError("Sealed signal datasets must use local hard links.")
                dataset = handle[name]
                if not isinstance(dataset, h5py.Dataset):
                    raise TypeError("Sealed signal entries must be HDF5 datasets.")
                if dataset.is_virtual or dataset.external:
                    raise ValueError("Sealed signal datasets cannot be virtual or external.")
            x_dataset = handle["x"]
            mask_dataset = handle["channel_mask"]
            if x_dataset.dtype != np.dtype(np.float32) or x_dataset.shape != (
                expected_rows,
                64,
                400,
            ):
                raise ValueError("Sealed x must use exact float32 [N,64,400] storage.")
            if mask_dataset.dtype != np.dtype(bool) or mask_dataset.shape != (
                expected_rows,
                64,
            ):
                raise ValueError("Sealed channel_mask must use exact bool [N,64] storage.")
            x = np.asarray(x_dataset[:])
            channel_mask = np.asarray(mask_dataset[:])
    finally:
        os.close(descriptor)
    return x, channel_mask


def _exact_group_root(path: str | Path) -> Path:
    root = Path(path).absolute()
    if not root.is_dir() or root.is_symlink():
        raise ValueError("Sealed checkpoint-group root must be one real directory.")
    # A worker receives exactly this group root; resolving or discovering its
    # siblings is deliberately absent from this module.
    return root
