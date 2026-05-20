"""Build a participant × (session × feature_set) wide table from feature_root.

For each (subject, session, modality, feature_set) we read pooled stats (mean,
std, percentiles, plus per-feature_set scalars from pooled.json). The result
is one row per participant with columns named like:

    audio__egemaps__A01__F0semitoneFrom27.5Hz_sma3nz_amean
    video__qc_stats__B03__blur_std
    audio__ssl_embed__chinese-hubert-large__B02__embed_0007_p50
    audio__ssl_embed__present__A01            (1.0 if present, else 0.0)

The output of this module feeds into ssl_pca (compresses SSL blocks) and views
(slices columns into audio/video/fused views).
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from ..utils.dass21 import SESSIONS
from .feature_io import load_pooled
from .labels import ID_COLS

log = logging.getLogger(__name__)


# Modality / feature-set classification.
AUDIO_FEATURES_DEFAULT = ("mel_mfcc", "vad", "egemaps", "ssl_embed")
VIDEO_FEATURES_DEFAULT = (
    "headpose_geom", "face_behavior", "qc_stats", "vad_agg",
    "body_pose", "global_motion", "vision_ssl_embed",
)
SSL_FEATURES = {"ssl_embed", "vision_ssl_embed"}


@dataclass
class TableConfig:
    feature_root: str
    sessions: tuple[str, ...] = SESSIONS
    audio_features: tuple[str, ...] = AUDIO_FEATURES_DEFAULT
    video_features: tuple[str, ...] = VIDEO_FEATURES_DEFAULT
    audio_ssl_model_tag: str = "chinese-hubert-large"
    video_ssl_model_tag: str = "dinov2-large"
    drop_high_na: float = 0.5      # column-wise NaN threshold
    drop_zero_variance: bool = True

    def modality_of(self, feat: str) -> str:
        if feat in self.audio_features:
            return "audio"
        if feat in self.video_features:
            return "video"
        raise ValueError(f"Feature {feat} not declared as audio nor video")

    def model_tag_of(self, feat: str) -> str | None:
        if feat == "ssl_embed":
            return self.audio_ssl_model_tag
        if feat == "vision_ssl_embed":
            return self.video_ssl_model_tag
        return None

    all_features: tuple[str, ...] = field(init=False, default=())

    def __post_init__(self) -> None:
        self.all_features = tuple(list(self.audio_features) + list(self.video_features))


def build_table(
    manifest: pd.DataFrame,
    cfg: TableConfig,
    split: str,
    progress: bool = True,
) -> pd.DataFrame:
    """One row per participant; columns = pooled stats from every modality/session.

    Missing files become NaN. A separate `_present` column (1.0 / 0.0) is added
    per (feature_set, session) so downstream models can encode missingness.
    """
    root = Path(cfg.feature_root)
    pids = manifest[ID_COLS].drop_duplicates().reset_index(drop=True)

    iterator = pids.itertuples(index=False, name=None)
    if progress:
        iterator = tqdm(iterator, total=len(pids), desc=f"build_table[{split}]")

    rows: list[dict[str, float]] = []
    column_template: dict[str, list[str]] | None = None
    column_template_lock = False

    for school, cls, pid in iterator:
        row: dict[str, float] = {
            "anon_school": str(school),
            "anon_class": str(cls),
            "anon_pid": str(pid),
        }
        for session in cfg.sessions:
            for feat in cfg.all_features:
                modality = cfg.modality_of(feat)
                tag = cfg.model_tag_of(feat)
                vals, names = load_pooled(
                    root, split, str(school), str(cls), str(pid),
                    modality, feat, session, model_tag=tag,
                )
                present_key = _present_key(modality, feat, session, tag)
                if vals is None or names is None:
                    row[present_key] = 0.0
                    continue
                row[present_key] = 1.0
                prefix = _col_prefix(modality, feat, session, tag)
                # Initialize template once from the first non-empty row.
                if not column_template_lock:
                    column_template = column_template or {}
                    column_template[prefix] = list(names)
                for n, v in zip(names, vals):
                    row[f"{prefix}__{n}"] = float(v)
        rows.append(row)
        # Lock template after the first participant so subsequent participants
        # don't keep mutating it. Missing columns will be NaN by reindex.
        if not column_template_lock and rows:
            column_template_lock = True

    df = pd.DataFrame(rows)
    df = df.reindex(columns=_canonical_columns(df, column_template))
    df = _post_filter(df, cfg)
    return df


def _present_key(modality: str, feat: str, session: str, tag: str | None) -> str:
    pieces = [modality, feat]
    if tag is not None:
        pieces.append(tag)
    pieces.append(session)
    pieces.append("present")
    return "__".join(pieces)


def _col_prefix(modality: str, feat: str, session: str, tag: str | None) -> str:
    pieces = [modality, feat]
    if tag is not None:
        pieces.append(tag)
    pieces.append(session)
    return "__".join(pieces)


def _canonical_columns(df: pd.DataFrame, template: dict[str, list[str]] | None) -> list[str]:
    # Identity columns first, then everything else in sorted (deterministic) order.
    head = [c for c in ID_COLS if c in df.columns]
    rest = sorted(c for c in df.columns if c not in head)
    return head + rest


def _post_filter(df: pd.DataFrame, cfg: TableConfig) -> pd.DataFrame:
    """Drop columns that are mostly NaN or zero-variance.

    The drop decisions live in feature-building, not in training. We persist
    the kept-column list so test-time can use the same schema.
    """
    feat_cols = [c for c in df.columns if c not in ID_COLS]
    if not feat_cols:
        return df

    na_rate = df[feat_cols].isna().mean(axis=0)
    keep_mask = na_rate < cfg.drop_high_na
    drop_na = list(na_rate[~keep_mask].index)
    if drop_na:
        log.info(f"drop {len(drop_na)} cols with >{cfg.drop_high_na:.0%} NaN")

    if cfg.drop_zero_variance:
        sub = df[feat_cols].copy()
        variances = sub.var(axis=0, skipna=True)
        zero_var = list(variances[(variances == 0) | variances.isna()].index)
        keep_mask &= ~variances.index.isin(zero_var)
        if zero_var:
            log.info(f"drop {len(zero_var)} zero-variance cols")

    kept = list(keep_mask[keep_mask].index)
    return df[[c for c in ID_COLS if c in df.columns] + kept]


def merge_with_labels(table: pd.DataFrame, label_table: pd.DataFrame) -> pd.DataFrame:
    """Inner-join wide table with subject_labels output."""
    return table.merge(label_table, on=ID_COLS, how="left")
