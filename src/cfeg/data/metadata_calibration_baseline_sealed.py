from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd

from cfeg.data.metadata_calibration_features import WEARABLE_CALIBRATION_CHANNEL_IDS
from cfeg.data.metadata_calibration_interventions import (
    metadata_calibration_base_input_sha256,
)
from cfeg.data.metadata_calibration_sealed import load_exact_sealed_signal_h5
from cfeg.identity import canonical_identity_sha256

_TOKEN = re.compile(r"[qsf]_[0-9a-f]{64}")
_BASELINE_COLUMNS = {
    "base_input_sha256",
    "checkpoint_group",
    "electrode_type",
    "run_id",
    "signal_index",
    "subject_id",
}


@dataclass(frozen=True)
class BaselineSealedRows:
    x: np.ndarray
    frame: pd.DataFrame
    token_column: str
    labels: np.ndarray | None
    trials_per_episode: int

    def episode_keys(self) -> tuple[tuple[str, str], ...]:
        cells = self.frame.loc[:, ["subject_id", "electrode_type"]].drop_duplicates()
        return tuple(
            (str(row.subject_id), str(row.electrode_type))
            for row in cells.sort_values(
                ["subject_id", "electrode_type"], kind="mergesort"
            ).itertuples(index=False)
        )

    def indices_for_episode(self, subject_id: str, electrode_type: str) -> np.ndarray:
        selected = self.frame["subject_id"].astype(str).eq(str(subject_id)) & self.frame[
            "electrode_type"
        ].astype(str).eq(str(electrode_type))
        indices = np.flatnonzero(selected.to_numpy())
        if len(indices) != self.trials_per_episode:
            raise ValueError("Baseline participant-interface episode has a wrong row count.")
        return indices


@dataclass(frozen=True)
class BaselineSealedTarget:
    support: BaselineSealedRows | None
    support_rows: pd.DataFrame
    query: BaselineSealedRows
    expected_query: pd.DataFrame


def load_baseline_source_fit(group_root: str | Path) -> BaselineSealedRows:
    root = _group_root(group_root)
    rows = _load_rows(
        root / "source_fit/baseline_view.jsonl",
        root / "source_fit/signals.h5",
        token_column="source_token",
        prefix="f_",
        include_label=True,
        trials_per_episode=120,
    )
    _validate_blocks(rows, range(1, 11))
    return rows


def load_baseline_target(group_root: str | Path) -> BaselineSealedTarget:
    root = _group_root(group_root)
    support = _load_rows(
        root / "target_support/baseline_view.jsonl",
        root / "target_support/signals.h5",
        token_column="support_token",
        prefix="s_",
        include_label=False,
        trials_per_episode=60,
    )
    query = _load_rows(
        root / "target_query/baseline_view.jsonl",
        root / "target_query/signals.h5",
        token_column="query_token",
        prefix="q_",
        include_label=False,
        trials_per_episode=60,
    )
    support_rows = _read_jsonl(
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
    expected_query = _read_jsonl(
        root / "target_query/expected_unlabeled.jsonl",
        {
            "checkpoint_group",
            "electrode_type",
            "query_identity_sha256",
            "query_token",
            "subject_id",
        },
    )
    _validate_target(support, support_rows, query, expected_query)
    return BaselineSealedTarget(
        support=support,
        support_rows=support_rows,
        query=query,
        expected_query=expected_query,
    )


def load_baseline_query_only(group_root: str | Path) -> BaselineSealedTarget:
    """Load the strict k=0 FBCCA view without any calibration-label capability."""

    root = _group_root(group_root)
    query = _load_rows(
        root / "target_query/baseline_view.jsonl",
        root / "target_query/signals.h5",
        token_column="query_token",
        prefix="q_",
        include_label=False,
        trials_per_episode=60,
    )
    expected_query = _read_jsonl(
        root / "target_query/expected_unlabeled.jsonl",
        {
            "checkpoint_group",
            "electrode_type",
            "query_identity_sha256",
            "query_token",
            "subject_id",
        },
    )
    _validate_query_only(query, expected_query)
    return BaselineSealedTarget(
        support=None,
        support_rows=pd.DataFrame(),
        query=query,
        expected_query=expected_query,
    )


def official_wearable_channels(x: np.ndarray) -> np.ndarray:
    values = np.asarray(x)
    if values.ndim != 3 or values.shape[1:] != (64, 400):
        raise ValueError("Baseline EEG must have exact shape [N,64,400].")
    positions = np.asarray(WEARABLE_CALIBRATION_CHANNEL_IDS, dtype=np.int64) - 1
    selected = np.asarray(values[:, positions, :], dtype=np.float64)
    if not np.isfinite(selected).all():
        raise ValueError("Baseline EEG contains non-finite official-channel values.")
    return selected


def baseline_support_indices(
    target: BaselineSealedTarget,
    *,
    subject_id: str,
    electrode_type: str,
    budget: int,
) -> np.ndarray:
    if target.support is None:
        raise PermissionError("This baseline job has no target-support capability.")
    if budget == 0:
        return np.empty(0, dtype=np.int64)
    selected = target.support_rows.loc[
        target.support_rows["subject_id"].astype(str).eq(str(subject_id))
        & target.support_rows["electrode_type"].astype(str).eq(str(electrode_type))
        & pd.to_numeric(target.support_rows["budget"], errors="raise").eq(budget)
    ].copy()
    if len(selected) != budget * 12:
        raise ValueError("Baseline support budget has a wrong row count.")
    lookup = {
        str(token): index
        for index, token in enumerate(target.support.frame["support_token"].astype(str))
    }
    indices = np.asarray([lookup[str(token)] for token in selected["support_token"]])
    order = np.lexsort(
        (
            pd.to_numeric(selected["support_label"], errors="raise").to_numpy(),
            selected["run_id"].astype(str).to_numpy(),
        )
    )
    return indices[order]


def baseline_support_labels(
    target: BaselineSealedTarget,
    *,
    subject_id: str,
    electrode_type: str,
    budget: int,
) -> np.ndarray:
    if budget == 0:
        return np.empty(0, dtype=np.int64)
    selected = target.support_rows.loc[
        target.support_rows["subject_id"].astype(str).eq(str(subject_id))
        & target.support_rows["electrode_type"].astype(str).eq(str(electrode_type))
        & pd.to_numeric(target.support_rows["budget"], errors="raise").eq(budget)
    ].copy()
    order = np.lexsort(
        (
            pd.to_numeric(selected["support_label"], errors="raise").to_numpy(),
            selected["run_id"].astype(str).to_numpy(),
        )
    )
    labels = pd.to_numeric(selected["support_label"], errors="raise").to_numpy(
        dtype=np.int64
    )[order]
    if labels.shape != (budget * 12,):
        raise ValueError("Baseline support labels have a wrong row count.")
    return labels


def exact_source_pool_identities(
    rows: BaselineSealedRows,
) -> dict[str, str]:
    if rows.labels is None:
        raise ValueError("Source-pool identity requires labeled source rows.")
    output: dict[str, str] = {}
    for interface in ("dry", "wet"):
        selected = rows.frame.loc[
            rows.frame["electrode_type"].astype(str).eq(interface)
        ].copy()
        selected["label"] = rows.labels[selected.index.to_numpy(dtype=np.int64)]
        records = selected.loc[
            :,
            [
                "source_token",
                "base_input_sha256",
                "subject_id",
                "electrode_type",
                "run_id",
                "label",
            ],
        ].sort_values("source_token", kind="mergesort").to_dict(orient="records")
        output[interface] = _json_sha256(records)
    if output["dry"] == output["wet"]:
        raise ValueError("Dry and wet exact source-pool identities unexpectedly coincide.")
    return output


def _load_rows(
    view_path: Path,
    signals_path: Path,
    *,
    token_column: str,
    prefix: Literal["q_", "s_", "f_"],
    include_label: bool,
    trials_per_episode: int,
) -> BaselineSealedRows:
    fields = set(_BASELINE_COLUMNS) | {token_column}
    if include_label:
        fields.add("label")
    frame = _read_jsonl(view_path, fields)
    tokens = frame[token_column].astype(str)
    if frame.empty or tokens.duplicated().any() or not tokens.map(
        lambda value: bool(_TOKEN.fullmatch(value)) and value.startswith(prefix)
    ).all():
        raise ValueError("Baseline view has invalid or duplicate opaque tokens.")
    indices = pd.to_numeric(frame["signal_index"], errors="raise").to_numpy()
    if not np.equal(indices, np.floor(indices)).all() or not np.array_equal(
        indices.astype(np.int64), np.arange(len(frame))
    ):
        raise ValueError("Baseline signal indices are not contiguous and row aligned.")
    x, mask = load_exact_sealed_signal_h5(signals_path, expected_rows=len(frame))
    expected_mask = np.zeros(64, dtype=bool)
    expected_mask[
        np.asarray(WEARABLE_CALIBRATION_CHANNEL_IDS, dtype=np.int64) - 1
    ] = True
    if not np.array_equal(mask, np.broadcast_to(expected_mask, mask.shape)):
        raise ValueError("Baseline sealed EEG differs from the official wearable layout.")
    if not np.isfinite(x[mask]).all() or not np.all(x[~mask] == 0.0):
        raise ValueError("Baseline sealed EEG has invalid active or inactive samples.")
    for index, row in frame.iterrows():
        if metadata_calibration_base_input_sha256(x[index], mask[index]) != str(
            row["base_input_sha256"]
        ):
            raise ValueError("Baseline view does not bind its exact signal bytes.")
    if len(set(frame["checkpoint_group"].astype(str))) != 1:
        raise ValueError("Baseline worker view must bind one checkpoint group.")
    by_subject = frame.groupby("subject_id", sort=False)["electrode_type"].agg(
        lambda values: set(map(str, values))
    )
    if not by_subject.map(lambda values: values == {"dry", "wet"}).all():
        raise ValueError("Every baseline participant must contain dry and wet interfaces.")
    labels = None
    if include_label:
        numeric = pd.to_numeric(frame["label"], errors="raise").to_numpy()
        if not np.equal(numeric, np.floor(numeric)).all() or (numeric < 0).any() or (
            numeric >= 12
        ).any():
            raise ValueError("Baseline source labels are outside the 12-class vocabulary.")
        labels = numeric.astype(np.int64)
    return BaselineSealedRows(
        x=x,
        frame=frame.reset_index(drop=True),
        token_column=token_column,
        labels=labels,
        trials_per_episode=trials_per_episode,
    )


def _validate_blocks(rows: BaselineSealedRows, expected: range) -> None:
    block_names = {f"block{index:02d}" for index in expected}
    for subject, interface in rows.episode_keys():
        indices = rows.indices_for_episode(subject, interface)
        episode = rows.frame.iloc[indices]
        if set(episode["run_id"].astype(str)) != block_names:
            raise ValueError("Baseline episode contains wrong frozen blocks.")
        for _run_id, block in episode.groupby("run_id", sort=False):
            if len(block) != 12:
                raise ValueError("Baseline episode block must contain twelve trials.")
            if rows.labels is not None and sorted(rows.labels[block.index].tolist()) != list(
                range(12)
            ):
                raise ValueError("Baseline source block must contain every class exactly once.")


def _validate_target(
    support: BaselineSealedRows,
    support_rows: pd.DataFrame,
    query: BaselineSealedRows,
    expected_query: pd.DataFrame,
) -> None:
    _validate_blocks(support, range(1, 6))
    _validate_query_only(query, expected_query)
    if set(support.episode_keys()) != set(query.episode_keys()):
        raise ValueError("Baseline support and query participant-interface cells differ.")
    shared = (
        "subject_id",
        "electrode_type",
        "checkpoint_group",
        "run_id",
        "base_input_sha256",
    )
    joined = support_rows.merge(
        support.frame.loc[:, ["support_token", *shared]],
        on="support_token",
        how="left",
        suffixes=("", "_view"),
        validate="many_to_one",
        indicator=True,
    )
    if not joined["_merge"].eq("both").all() or any(
        not joined[field].astype(str).eq(joined[f"{field}_view"].astype(str)).all()
        for field in shared
    ):
        raise ValueError("Baseline support labels are not bound to exact signal rows.")
    consistency = support_rows.groupby("support_token", sort=False).agg(
        labels=("support_label", "nunique"),
        subjects=("subject_id", "nunique"),
        interfaces=("electrode_type", "nunique"),
        groups=("checkpoint_group", "nunique"),
        runs=("run_id", "nunique"),
        inputs=("base_input_sha256", "nunique"),
    )
    if not consistency.eq(1).all().all():
        raise ValueError("A baseline support token changes identity or label across budgets.")
    for subject, interface in query.episode_keys():
        token_sets: dict[int, set[str]] = {}
        for budget in (1, 3, 5):
            indices = baseline_support_indices(
                BaselineSealedTarget(support, support_rows, query, expected_query),
                subject_id=subject,
                electrode_type=interface,
                budget=budget,
            )
            labels = baseline_support_labels(
                BaselineSealedTarget(support, support_rows, query, expected_query),
                subject_id=subject,
                electrode_type=interface,
                budget=budget,
            )
            frame = support.frame.iloc[indices].assign(label=labels)
            expected_runs = {f"block{index:02d}" for index in range(1, budget + 1)}
            if set(frame["run_id"].astype(str)) != expected_runs:
                raise ValueError("Baseline support budget is not the frozen block prefix.")
            for _run_id, block in frame.groupby("run_id", sort=False):
                if sorted(block["label"].astype(int).tolist()) != list(range(12)):
                    raise ValueError("Baseline support block must contain every class once.")
            token_sets[budget] = set(frame["support_token"].astype(str))
        if not token_sets[1] < token_sets[3] or not token_sets[3] < token_sets[5]:
            raise ValueError("Baseline support tokens must be strictly nested k1<k3<k5.")


def _validate_query_only(
    query: BaselineSealedRows, expected_query: pd.DataFrame
) -> None:
    _validate_blocks(query, range(6, 11))
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
        raise ValueError("Baseline expected query differs from the no-context view.")
    for subject, interface in query.episode_keys():
        episode = query.frame.loc[
            query.frame["subject_id"].astype(str).eq(subject)
            & query.frame["electrode_type"].astype(str).eq(interface)
        ]
        expected_identity = canonical_identity_sha256(
            episode["query_token"].astype(str)
        )
        observed = expected_query.loc[
            expected_query["subject_id"].astype(str).eq(subject)
            & expected_query["electrode_type"].astype(str).eq(interface),
            "query_identity_sha256",
        ].astype(str)
        if set(observed) != {expected_identity}:
            raise ValueError("Baseline expected-query identity is invalid.")


def _read_jsonl(path: Path, fields: set[str]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"Invalid JSONL at {path}:{line_number}.") from error
            if not isinstance(row, dict) or set(row) != fields:
                raise ValueError(f"Baseline JSONL schema drifted at {path}:{line_number}.")
            rows.append(row)
    return pd.DataFrame(rows, columns=sorted(fields))


def _group_root(path: str | Path) -> Path:
    root = Path(path).absolute()
    if not root.is_dir() or root.is_symlink():
        raise ValueError("Baseline checkpoint-group root must be one real directory.")
    return root


def _json_sha256(value: object) -> str:
    import hashlib

    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()
