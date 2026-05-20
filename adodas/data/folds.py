"""Subject-disjoint StratifiedGroupKFold for the labeled train+val pool.

Stratification = A2-total-score quartiles (mirrors backup/scripts/make_cv_folds.py).
We deduplicate by anon_pid so the four sessions of one participant always
land in the same fold. The output is a flat array of fold ids aligned to a
participant-level table (one row per participant).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

from ..utils.dass21 import ITEM_COLS
from .labels import ID_COLS


def assign_folds(
    labels: pd.DataFrame,
    n_folds: int = 5,
    seed: int = 42,
    n_strata: int = 4,
) -> np.ndarray:
    """Return an (N,) int array of fold ids in [0, n_folds). One row per participant.

    Stratification uses A2 total score quantiles. Requires d01..d21 to be in
    the labels DataFrame (NaN entries fall back to median-imputed total).
    """
    a2 = labels[ITEM_COLS].copy()
    totals = a2.sum(axis=1, skipna=True)
    median = totals.median()
    totals = totals.fillna(median)

    if totals.nunique() < n_strata:
        n_strata = max(2, totals.nunique())

    bins = pd.qcut(totals.rank(method="first"), q=n_strata, labels=False)
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    folds = np.full(len(labels), -1, dtype=int)
    for fold_id, (_, val_idx) in enumerate(skf.split(np.zeros(len(labels)), bins)):
        folds[val_idx] = fold_id
    if (folds < 0).any():
        raise RuntimeError("Unassigned rows in fold assignment")
    return folds


def fold_indices(folds: np.ndarray, fold: int) -> tuple[np.ndarray, np.ndarray]:
    """Return (train_idx, val_idx) integer arrays."""
    val_idx = np.where(folds == fold)[0]
    train_idx = np.where(folds != fold)[0]
    return train_idx, val_idx


def verify_disjoint(ids: pd.DataFrame, folds: np.ndarray) -> None:
    """Assert that train PID ∩ val PID == ∅ for every fold."""
    pids = ids["anon_pid"].astype(str).to_numpy()
    n_folds = int(folds.max()) + 1
    for f in range(n_folds):
        train_pids = set(pids[folds != f].tolist())
        val_pids = set(pids[folds == f].tolist())
        leak = train_pids & val_pids
        if leak:
            raise AssertionError(f"Fold {f} leak: {len(leak)} pids in both train and val")
