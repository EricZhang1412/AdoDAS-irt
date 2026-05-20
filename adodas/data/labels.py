"""Subject-level label extraction from manifest CSVs."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from ..utils.dass21 import A1_COLS, ITEM_COLS


ID_COLS = ["anon_school", "anon_class", "anon_pid"]


def load_manifest(path: str | Path) -> pd.DataFrame:
    """Read a manifest CSV. Expects columns: anon_school, anon_class, anon_pid, session, ..."""
    df = pd.read_csv(path)
    required = {"anon_school", "anon_class", "anon_pid", "session"}
    missing = required - set(df.columns)
    if missing:
        raise KeyError(f"Manifest {path} missing columns: {missing}")
    return df


def subject_labels(manifest: pd.DataFrame) -> pd.DataFrame:
    """Collapse a session-level manifest into one row per participant.

    Picks the first non-NaN A1 / A2 label seen per (school, class, pid). Labels
    are the same across the 4 sessions of one participant, so this is safe.
    Missing labels stay as NaN and are filtered downstream.
    """
    cols = ID_COLS + [c for c in A1_COLS + ITEM_COLS if c in manifest.columns]
    sub = manifest[cols].copy()
    grouped = sub.groupby(ID_COLS, as_index=False).first()
    return grouped


def split_features_and_labels(
    table: pd.DataFrame,
) -> tuple[pd.DataFrame, np.ndarray | None, np.ndarray | None, pd.DataFrame]:
    """Return (X, y_a1, y_a2, ids). y_* are None for the test split (no labels)."""
    ids = table[ID_COLS].reset_index(drop=True)

    have_a1 = all(c in table.columns for c in A1_COLS)
    have_a2 = all(c in table.columns for c in ITEM_COLS)

    y_a1 = table[A1_COLS].to_numpy(np.float32) if have_a1 else None
    y_a2 = table[ITEM_COLS].to_numpy(np.float32) if have_a2 else None

    drop_cols = ID_COLS + (A1_COLS if have_a1 else []) + (ITEM_COLS if have_a2 else [])
    X = table.drop(columns=[c for c in drop_cols if c in table.columns])
    X = X.reset_index(drop=True)
    return X, y_a1, y_a2, ids


def valid_label_mask(y: np.ndarray | None) -> np.ndarray:
    """True for rows where all targets are non-NaN and non-negative.

    A1: values are 0/1; -1 / NaN signal missing. A2: same convention.
    Test rows have NaN labels and must be excluded from supervised training.
    """
    if y is None:
        return np.array([], dtype=bool)
    valid = np.isfinite(y).all(axis=1)
    return valid & (y >= 0).all(axis=1)
