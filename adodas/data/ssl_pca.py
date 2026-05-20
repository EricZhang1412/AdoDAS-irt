"""PCA reduction for the very-high-dim SSL pooled blocks.

SSL pooled features are typically 5 * D (mean / std / p10 / p50 / p90) per
session. For chinese-hubert-large (D=1024) that's 5120 columns × 4 sessions
= ~20k columns per subject from SSL alone. We fit PCA per
(modality, feature_set, model_tag, session) block on labeled data only, then
transform train/val/test consistently.

Outputs: a dict of fitted PCA objects + the column index they consume, so
test-time can reproduce the transform exactly.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer

from .labels import ID_COLS

# Match SSL blocks (audio ssl_embed, vision_ssl_embed) per session.
# Columns look like:
#   audio__ssl_embed__chinese-hubert-large__A01__embed_0023_mean
#   video__vision_ssl_embed__dinov2-large__B02__embed_0500_p90
_SSL_BLOCK_RE = re.compile(
    r"^(?P<modality>audio|video)__(?P<feat>ssl_embed|vision_ssl_embed)"
    r"__(?P<tag>[^_]+(?:-[^_]+)*)__(?P<session>A01|B01|B02|B03)__"
)


@dataclass
class PCABlock:
    block_key: tuple[str, str, str, str]  # (modality, feat, tag, session)
    columns: list[str]
    pca: PCA
    imputer: SimpleImputer


def discover_ssl_blocks(df: pd.DataFrame) -> dict[tuple[str, str, str, str], list[str]]:
    """Group SSL columns by (modality, feat, tag, session)."""
    blocks: dict[tuple[str, str, str, str], list[str]] = {}
    for c in df.columns:
        if c in ID_COLS:
            continue
        m = _SSL_BLOCK_RE.match(c)
        if m is None:
            continue
        key = (m["modality"], m["feat"], m["tag"], m["session"])
        blocks.setdefault(key, []).append(c)
    for k in blocks:
        blocks[k].sort()
    return blocks


def fit_ssl_pca(
    df_train: pd.DataFrame,
    n_components_audio: int = 96,
    n_components_video: int = 96,
    random_state: int = 42,
    whiten: bool = False,
) -> list[PCABlock]:
    """Fit one PCA per SSL block on labeled data."""
    blocks = discover_ssl_blocks(df_train)
    fitted: list[PCABlock] = []
    for key, cols in blocks.items():
        modality = key[0]
        n_comp = n_components_audio if modality == "audio" else n_components_video
        n_comp = min(n_comp, len(cols), len(df_train) - 1)
        if n_comp <= 0:
            continue
        X = df_train[cols].to_numpy(np.float32)
        imp = SimpleImputer(strategy="median").fit(X)
        X = imp.transform(X)
        pca = PCA(n_components=n_comp, random_state=random_state, whiten=whiten).fit(X)
        fitted.append(PCABlock(block_key=key, columns=cols, pca=pca, imputer=imp))
    return fitted


def transform_ssl_pca(df: pd.DataFrame, blocks: list[PCABlock]) -> pd.DataFrame:
    """Replace SSL block columns with PCA-reduced ones. Returns a new DataFrame."""
    drop_cols: list[str] = []
    new_cols: dict[str, pd.Series] = {}
    for blk in blocks:
        mod, feat, tag, sess = blk.block_key
        X = df.reindex(columns=blk.columns).to_numpy(np.float32)
        X = blk.imputer.transform(X)
        Z = blk.pca.transform(X)
        prefix = f"{mod}__{feat}__{tag}__{sess}__pca"
        for k in range(Z.shape[1]):
            new_cols[f"{prefix}_{k:03d}"] = pd.Series(Z[:, k], index=df.index, dtype=np.float32)
        drop_cols.extend(blk.columns)

    if not new_cols:
        return df
    base = df.drop(columns=[c for c in drop_cols if c in df.columns])
    return pd.concat([base, pd.DataFrame(new_cols, index=df.index)], axis=1)


def block_summary(blocks: Iterable[PCABlock]) -> list[dict[str, object]]:
    """Compact json-able summary of each fitted block, for run logs."""
    out = []
    for b in blocks:
        evr = float(b.pca.explained_variance_ratio_.sum())
        out.append({
            "block_key": list(b.block_key),
            "n_input_cols": len(b.columns),
            "n_components": int(b.pca.n_components_),
            "explained_variance_ratio_sum": evr,
        })
    return out
