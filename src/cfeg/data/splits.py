from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class SplitIndices:
    train: np.ndarray
    val: np.ndarray
    test: np.ndarray


def make_cross_subject_split(
    manifest: pd.DataFrame,
    seed: int,
    val_ratio: float,
    test_ratio: float,
) -> SplitIndices:
    groups = _subject_groups(manifest)
    train_groups, val_groups, test_groups = _split_groups(
        groups, seed=seed, val_ratio=val_ratio, test_ratio=test_ratio
    )
    split = SplitIndices(
        train=_indices_for_groups(groups, train_groups),
        val=_indices_for_groups(groups, val_groups),
        test=_indices_for_groups(groups, test_groups),
    )
    _ensure_nonempty(split.train, split.val, split.test, context="cross-subject")
    return split


def make_within_dataset_leave_subjects_out(
    manifest: pd.DataFrame, dataset_id: str, seed: int
) -> SplitIndices:
    indices = np.flatnonzero(manifest["dataset_id"].astype(str).eq(dataset_id).to_numpy())
    local = make_cross_subject_split(
        manifest.iloc[indices].reset_index(drop=True), seed=seed, val_ratio=0.2, test_ratio=0.2
    )
    return SplitIndices(train=indices[local.train], val=indices[local.val], test=indices[local.test])


def make_cross_dataset_split(
    manifest: pd.DataFrame,
    train_datasets: list[str],
    test_datasets: list[str],
    *,
    seed: int = 42,
    val_ratio: float = 0.2,
) -> SplitIndices:
    ds = manifest["dataset_id"].astype(str)
    source_indices = np.flatnonzero(ds.isin(train_datasets).to_numpy())
    test = np.flatnonzero(ds.isin(test_datasets).to_numpy())
    train, val = _source_train_val(manifest, source_indices, seed=seed, val_ratio=val_ratio)
    _ensure_nonempty(train, val, test, context="cross-dataset")
    return SplitIndices(train=train, val=val, test=test)


def make_cross_condition_split(
    manifest: pd.DataFrame,
    train_filter: dict[str, str | list[str]],
    test_filter: dict[str, str | list[str]],
    *,
    seed: int = 42,
    val_ratio: float = 0.2,
) -> SplitIndices:
    source_indices = np.flatnonzero(_filter_mask(manifest, train_filter))
    test = np.flatnonzero(_filter_mask(manifest, test_filter))
    train, val = _source_train_val(manifest, source_indices, seed=seed, val_ratio=val_ratio)
    _ensure_nonempty(train, val, test, context="cross-condition")
    return SplitIndices(train=train, val=val, test=test)


def make_openbci_external_split(manifest: pd.DataFrame) -> SplitIndices:
    idx = np.arange(len(manifest))
    return SplitIndices(train=np.array([], dtype=int), val=idx, test=idx)


def _subject_groups(manifest: pd.DataFrame) -> np.ndarray:
    return (
        manifest["dataset_id"].astype(str)
        + "::"
        + manifest["subject_id"].astype(str)
    ).to_numpy()


def _source_train_val(
    manifest: pd.DataFrame, source_indices: np.ndarray, *, seed: int, val_ratio: float
) -> tuple[np.ndarray, np.ndarray]:
    if not len(source_indices):
        return source_indices, source_indices
    groups = _subject_groups(manifest.iloc[source_indices].reset_index(drop=True))
    unique = np.array(sorted(set(groups)))
    rng = np.random.default_rng(seed)
    rng.shuffle(unique)
    n_val = max(1, int(round(len(unique) * val_ratio))) if len(unique) > 1 else 0
    val_groups = set(unique[:n_val])
    local_val = np.flatnonzero(np.isin(groups, list(val_groups)))
    local_train = np.flatnonzero(~np.isin(groups, list(val_groups)))
    return source_indices[local_train], source_indices[local_val]


def _split_groups(
    groups: np.ndarray, *, seed: int, val_ratio: float, test_ratio: float
) -> tuple[set[str], set[str], set[str]]:
    unique = np.array(sorted(set(groups)))
    rng = np.random.default_rng(seed)
    rng.shuffle(unique)
    n_test = max(1, int(round(len(unique) * test_ratio)))
    n_val = max(1, int(round(len(unique) * val_ratio))) if len(unique) > 2 else 0
    test = set(unique[:n_test])
    val = set(unique[n_test : n_test + n_val])
    train = set(unique[n_test + n_val :])
    if not train:
        train = {unique[-1]}
        test.discard(unique[-1])
        val.discard(unique[-1])
    return train, val, test


def _indices_for_groups(groups: np.ndarray, selected: set[str]) -> np.ndarray:
    return np.flatnonzero(np.isin(groups, list(selected)))


def _filter_mask(
    manifest: pd.DataFrame, filters: dict[str, str | list[str]]
) -> np.ndarray:
    mask = np.ones(len(manifest), dtype=bool)
    for column, expected in filters.items():
        if column not in manifest:
            raise KeyError(f"Unknown manifest filter column: {column}")
        values = expected if isinstance(expected, list) else [expected]
        mask &= manifest[column].astype(str).isin([str(value) for value in values]).to_numpy()
    return mask


def _ensure_nonempty(train: np.ndarray, val: np.ndarray, test: np.ndarray, *, context: str) -> None:
    if not len(train) or not len(val) or not len(test):
        raise ValueError(
            f"{context} split is empty: train={len(train)}, val={len(val)}, test={len(test)}. "
            "At least two source subjects and a separate test group are required; "
            "check processed_dirs and dataset/metadata filters."
        )
