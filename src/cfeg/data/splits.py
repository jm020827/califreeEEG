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
    subjects = np.array(sorted(manifest["subject_id"].astype(str).unique()))
    rng = np.random.default_rng(seed)
    rng.shuffle(subjects)
    n_test = max(1, int(round(len(subjects) * test_ratio)))
    n_val = max(1, int(round(len(subjects) * val_ratio))) if len(subjects) > 2 else 0
    test_sub = set(subjects[:n_test])
    val_sub = set(subjects[n_test : n_test + n_val])
    train_sub = set(subjects[n_test + n_val :])
    if not train_sub:
        train_sub = set(subjects[n_test + n_val - 1 : n_test + n_val])
        val_sub = set(subjects[n_test:n_test + max(0, n_val - 1)])
    subject_col = manifest["subject_id"].astype(str)
    return SplitIndices(
        train=np.flatnonzero(subject_col.isin(train_sub).to_numpy()),
        val=np.flatnonzero(subject_col.isin(val_sub).to_numpy()),
        test=np.flatnonzero(subject_col.isin(test_sub).to_numpy()),
    )


def make_within_dataset_leave_subjects_out(
    manifest: pd.DataFrame, dataset_id: str, seed: int
) -> SplitIndices:
    subset = manifest[manifest["dataset_id"] == dataset_id].reset_index(drop=True)
    return make_cross_subject_split(subset, seed=seed, val_ratio=0.2, test_ratio=0.2)


def make_cross_dataset_split(
    manifest: pd.DataFrame, train_datasets: list[str], test_datasets: list[str]
) -> SplitIndices:
    ds = manifest["dataset_id"].astype(str)
    train = np.flatnonzero(ds.isin(train_datasets).to_numpy())
    test = np.flatnonzero(ds.isin(test_datasets).to_numpy())
    return SplitIndices(train=train, val=test, test=test)


def make_openbci_external_split(manifest: pd.DataFrame) -> SplitIndices:
    idx = np.arange(len(manifest))
    return SplitIndices(train=np.array([], dtype=int), val=idx, test=idx)

