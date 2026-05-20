"""Session-difference and emotional-contrast features.

For every (modality, feature_set, pooled_name) that exists in *all* sessions
of a pair, we emit a diff column. This is conservative — pairs where one
session is missing get NaN and will be imputed downstream.

Naming convention:
    audio__egemaps__diff__B03-B02__F0semitoneFrom27.5Hz_sma3nz_amean
    audio__egemaps__mean3__B01_B02_B03__F0semitoneFrom27.5Hz_sma3nz_amean
"""
from __future__ import annotations

import re
from typing import Iterable

import pandas as pd

from ..utils.dass21 import SESSIONS
from .labels import ID_COLS


# Pattern matches columns built by pooled_table:
#   audio__egemaps__A01__<feature>
#   audio__ssl_embed__chinese-hubert-large__B03__<feature>
# We split at '__' and look for a session token (one of SESSIONS).
def _parse_col(col: str) -> tuple[str, str, str] | None:
    """Return (key_prefix, session, suffix) or None if not a session-tagged col."""
    parts = col.split("__")
    for i, p in enumerate(parts):
        if p in SESSIONS and i > 0 and i < len(parts) - 1:
            key_prefix = "__".join(parts[:i])
            session = parts[i]
            suffix = "__".join(parts[i + 1:])
            if suffix.endswith("present") or suffix == "present":
                return None
            return key_prefix, session, suffix
    return None


def build_session_diffs(
    df: pd.DataFrame,
    pairs: Iterable[tuple[str, str]],
    add_open_ended_mean: bool = True,
) -> pd.DataFrame:
    """Add diff and mean3 columns. Returns the full table (originals + new cols)."""
    feat_cols = [c for c in df.columns if c not in ID_COLS]

    by_key_session: dict[tuple[str, str], dict[str, str]] = {}
    for c in feat_cols:
        parsed = _parse_col(c)
        if parsed is None:
            continue
        key, sess, suffix = parsed
        by_key_session.setdefault((key, suffix), {})[sess] = c

    new_cols: dict[str, pd.Series] = {}

    for (key, suffix), session_map in by_key_session.items():
        for a, b in pairs:
            if a in session_map and b in session_map:
                col_a = session_map[a]
                col_b = session_map[b]
                new_name = f"{key}__diff__{a}-{b}__{suffix}"
                new_cols[new_name] = df[col_a] - df[col_b]

        if add_open_ended_mean:
            present = [s for s in ("B01", "B02", "B03") if s in session_map]
            if len(present) >= 2:
                stack = pd.concat([df[session_map[s]] for s in present], axis=1)
                tag = "_".join(present)
                new_name = f"{key}__mean__{tag}__{suffix}"
                new_cols[new_name] = stack.mean(axis=1, skipna=True)

    if not new_cols:
        return df

    extra = pd.DataFrame(new_cols, index=df.index)
    return pd.concat([df, extra], axis=1)
