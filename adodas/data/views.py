"""Slice a wide table into modality views (audio_only / video_only / fused)."""
from __future__ import annotations

from typing import Iterable

import pandas as pd

from .labels import ID_COLS


def select_view(
    df: pd.DataFrame,
    view: str,
    audio_features: Iterable[str],
    video_features: Iterable[str],
) -> pd.DataFrame:
    """Return the subset of df whose feature columns belong to `view`.

    Diff/mean columns inherit the modality of their constituents (since their
    column name starts with `audio__` or `video__` already).
    """
    feat_cols = [c for c in df.columns if c not in ID_COLS]
    audio_cols, video_cols = _classify(feat_cols)

    if view == "fused":
        keep = audio_cols + video_cols
    elif view == "audio_only":
        keep = audio_cols
    elif view == "video_only":
        keep = video_cols
    else:
        raise ValueError(f"Unknown view: {view}")

    head = [c for c in ID_COLS if c in df.columns]
    return df[head + keep].copy()


def _classify(cols: list[str]) -> tuple[list[str], list[str]]:
    a, v = [], []
    for c in cols:
        if c.startswith("audio__"):
            a.append(c)
        elif c.startswith("video__"):
            v.append(c)
        else:
            # Bias toward audio for unknown columns; never lose data silently.
            a.append(c)
    return a, v
