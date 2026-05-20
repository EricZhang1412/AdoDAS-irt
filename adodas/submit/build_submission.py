"""Build the participant-level submission CSV in the official schema.

Schema (from backup/infer.py:266-349):
  A1: [anon_school, anon_class, anon_pid, p_D, p_A, p_S]   floats in [0,1]
  A2: [anon_school, anon_class, anon_pid, d01, ..., d21]   ints in {0,1,2,3}

The session column is omitted at participant level. Both files are written
to <output_dir>/submission/.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from ..utils.dass21 import A1_COLS, ITEM_COLS
from ..utils.io import ensure_dir


def write_a1_submission(
    ids: pd.DataFrame,
    p_a1: np.ndarray,
    output_dir: str | Path,
    filename: str = "submission_a1.csv",
) -> Path:
    """ids: (N, 3) DataFrame with anon_school/class/pid. p_a1: (N, 3)."""
    df = ids.copy()
    p_a1 = np.clip(p_a1, 0.0, 1.0)
    for c, col in enumerate(["p_D", "p_A", "p_S"]):
        df[col] = p_a1[:, c].astype(np.float32)
    out_dir = ensure_dir(Path(output_dir) / "submission")
    path = out_dir / filename
    df.to_csv(path, index=False, float_format="%.6f")
    return path


def write_a2_submission(
    ids: pd.DataFrame,
    a2_int: np.ndarray,
    output_dir: str | Path,
    filename: str = "submission_a2.csv",
) -> Path:
    df = ids.copy()
    arr = np.clip(np.rint(a2_int), 0, 3).astype(np.int32)
    for j, col in enumerate(ITEM_COLS):
        df[col] = arr[:, j]
    out_dir = ensure_dir(Path(output_dir) / "submission")
    path = out_dir / filename
    df.to_csv(path, index=False)
    return path
