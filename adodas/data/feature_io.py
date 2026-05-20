"""Filesystem layout for AdoDAS pre-extracted features.

Directly mirrors backup/common/data/feature_io.py so that the same feature_root
trees from the challenge organizers can be read without any conversion. We
extend the original with a pooled.json/.parquet loader for any modality (not
just egemaps), since the hybrid stack only ever needs pooled stats.

Layout:
  <feature_root>/<split>/<anon_school>/<anon_class>/<anon_pid>/
    audio/{mel_mfcc,vad,egemaps,ssl_embed/<model_tag>}/<session>/
        sequence.npz | pooled.json | pooled.parquet
    video/{...,vision_ssl_embed/<model_tag>}/<session>/
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import NamedTuple

import numpy as np


class SequenceData(NamedTuple):
    features: np.ndarray
    timestamps_ms: np.ndarray
    valid_mask: np.ndarray


_MEL_MFCC_KEYS = ("mel_features", "mfcc_features")
_GENERIC_KEY = "features"


def feature_dir(
    root: Path,
    split: str,
    anon_school: str,
    anon_class: str,
    anon_pid: str,
    modality: str,
    feature_set: str,
    session: str,
    model_tag: str | None = None,
) -> Path:
    parts: list[str] = [
        str(root), split, anon_school, anon_class, anon_pid,
        modality, feature_set,
    ]
    if model_tag is not None:
        parts.append(model_tag)
    parts.append(session)
    return Path(*parts)


def load_sequence(
    root: Path,
    split: str,
    anon_school: str,
    anon_class: str,
    anon_pid: str,
    modality: str,
    feature_set: str,
    session: str,
    model_tag: str | None = None,
) -> SequenceData:
    seq_path = feature_dir(
        root, split, anon_school, anon_class, anon_pid,
        modality, feature_set, session, model_tag,
    ) / "sequence.npz"
    if not seq_path.exists():
        raise FileNotFoundError(f"Missing sequence file: {seq_path}")

    data = np.load(str(seq_path), allow_pickle=True)

    if feature_set == "mel_mfcc":
        arrays = []
        for k in _MEL_MFCC_KEYS:
            if k not in data:
                raise KeyError(f"Expected key '{k}' in {seq_path}; found {list(data.keys())}")
            arrays.append(data[k].astype(np.float32))
        features = np.concatenate(arrays, axis=-1)
    elif _GENERIC_KEY in data:
        features = data[_GENERIC_KEY].astype(np.float32)
    else:
        raise KeyError(f"No known feature key in {seq_path}. Keys: {list(data.keys())}")

    if features.ndim == 1:
        features = features[:, np.newaxis]

    timestamps_ms = data["timestamps_ms"].astype(np.float64)
    valid_mask = (
        data["valid_mask"].astype(bool) if "valid_mask" in data
        else np.ones(len(timestamps_ms), dtype=bool)
    )

    return SequenceData(features=features, timestamps_ms=timestamps_ms, valid_mask=valid_mask)


def load_pooled(
    root: Path,
    split: str,
    anon_school: str,
    anon_class: str,
    anon_pid: str,
    modality: str,
    feature_set: str,
    session: str,
    model_tag: str | None = None,
) -> tuple[np.ndarray, list[str]] | tuple[None, None]:
    """Load pooled stats for one (subject, session, modality, feature_set).

    Looks for pooled.parquet first, then pooled.json. Returns (values, names)
    or (None, None) if neither file exists. Names are the column ordering used
    for downstream concatenation; the caller is responsible for stitching them
    into the participant wide-table column names.
    """
    base = feature_dir(
        root, split, anon_school, anon_class, anon_pid,
        modality, feature_set, session, model_tag,
    )

    parquet_path = base / "pooled.parquet"
    if parquet_path.exists():
        df = _read_parquet(parquet_path)
        if df is not None and len(df) > 0:
            row = df.iloc[0]
            return row.values.astype(np.float32), list(row.index.astype(str))

    json_path = base / "pooled.json"
    if json_path.exists():
        try:
            with open(json_path) as f:
                meta = json.load(f)
        except Exception:
            meta = None
        if meta is not None:
            vals, names = _flatten_pooled_json(meta)
            if vals is not None:
                return vals, names

    return None, None


def _read_parquet(path: Path):
    """Try pyarrow, then fastparquet; return None if both fail."""
    import pandas as pd
    for engine in ("pyarrow", "fastparquet"):
        try:
            return pd.read_parquet(path, engine=engine)
        except Exception:
            continue
    return None


def _flatten_pooled_json(meta: dict) -> tuple[np.ndarray | None, list[str] | None]:
    """Flatten a pooled.json structure into (values, names).

    Supported shapes:
      {"features": {name: value, ...}}        — egemaps style
      {name: value | [vals]}                  — generic flat dict
    """
    if "features" in meta and isinstance(meta["features"], dict):
        d = meta["features"]
    elif isinstance(meta, dict):
        d = meta
    else:
        return None, None

    names: list[str] = []
    vals: list[float] = []
    for k, v in d.items():
        if isinstance(v, (int, float)):
            names.append(str(k))
            vals.append(float(v))
        elif isinstance(v, list):
            for i, item in enumerate(v):
                if isinstance(item, (int, float)):
                    names.append(f"{k}_{i:03d}")
                    vals.append(float(item))
    if not vals:
        return None, None
    return np.asarray(vals, dtype=np.float32), names
