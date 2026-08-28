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


def make_cross_subject_fold_split(
    manifest: pd.DataFrame,
    *,
    seed: int,
    n_folds: int,
    fold_index: int,
    val_ratio: float,
) -> SplitIndices:
    """Create one outer participant fold with source-only participant validation."""
    groups = _subject_groups(manifest)
    unique = np.asarray(sorted(set(groups)))
    if n_folds < 3:
        raise ValueError("cross-subject-fold requires at least three folds.")
    if n_folds > len(unique):
        raise ValueError(
            f"cross-subject-fold has {len(unique)} subjects but n_folds={n_folds}."
        )
    if not 0 <= fold_index < n_folds:
        raise ValueError(
            f"fold_index must be within [0, {n_folds - 1}], got {fold_index}."
        )
    rng = np.random.default_rng(seed)
    rng.shuffle(unique)
    outer_folds = np.array_split(unique, n_folds)
    test_groups = set(outer_folds[fold_index].tolist())
    source_groups = np.asarray(
        [group for group in unique if group not in test_groups]
    )
    rng.shuffle(source_groups)
    n_val = max(1, round(len(source_groups) * val_ratio))
    val_groups = set(source_groups[:n_val].tolist())
    train_groups = set(source_groups[n_val:].tolist())
    split = SplitIndices(
        train=_indices_for_groups(groups, train_groups),
        val=_indices_for_groups(groups, val_groups),
        test=_indices_for_groups(groups, test_groups),
    )
    _ensure_nonempty(
        split.train,
        split.val,
        split.test,
        context=f"cross-subject-fold-{fold_index}",
    )
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


def make_joint_subject_condition_split(
    manifest: pd.DataFrame,
    train_filter: dict[str, str | list[str]],
    test_filter: dict[str, str | list[str]],
    *,
    seed: int = 42,
    val_ratio: float = 0.2,
    test_ratio: float = 0.2,
) -> SplitIndices:
    """Hold out both target subjects and their acquisition condition.

    Only subjects represented in both filtered conditions are eligible. Their
    identities are partitioned first; source-condition rows from the train and
    validation identities form the corresponding splits, while target-condition
    rows from disjoint identities form the final test split.
    """

    source_mask = _filter_mask(manifest, train_filter)
    target_mask = _filter_mask(manifest, test_filter)
    groups = _subject_groups(manifest)
    eligible_groups = sorted(set(groups[source_mask]) & set(groups[target_mask]))
    if len(eligible_groups) < 3:
        raise ValueError(
            "joint-subject-condition requires at least three subjects represented "
            "in both source and target conditions."
        )
    train_groups, val_groups, test_groups = _split_groups(
        np.asarray(eligible_groups),
        seed=seed,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
    )
    train = np.flatnonzero(source_mask & np.isin(groups, list(train_groups)))
    val = np.flatnonzero(source_mask & np.isin(groups, list(val_groups)))
    test = np.flatnonzero(target_mask & np.isin(groups, list(test_groups)))
    _ensure_nonempty(train, val, test, context="joint-subject-condition")
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
    n_val = max(1, round(len(unique) * val_ratio)) if len(unique) > 1 else 0
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
    n_test = max(1, round(len(unique) * test_ratio))
    n_val = max(1, round(len(unique) * val_ratio)) if len(unique) > 2 else 0
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
