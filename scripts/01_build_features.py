#!/usr/bin/env python3
"""Stage 01: build participant-level wide tables for train / val / test.

Outputs (under output_dir):
  features/train_raw.parquet                — pre-PCA wide table
  features/val_raw.parquet
  features/test_raw.parquet
  features/pca_blocks.npz                   — fitted PCA blocks (column lists + components)
  features/{view}_train.parquet             — view × split after PCA + session diffs
  features/{view}_val.parquet
  features/{view}_test.parquet
  features/labels.parquet                   — subject_labels(train ∪ val)

Per the project design, PCA is fit on labeled (train ∪ val) only and applied
to all splits. Session-diff features are appended *after* PCA.
"""
from __future__ import annotations

import argparse
import logging
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from adodas.data.feature_io import load_pooled  # noqa: E402  (path fix above)
from adodas.data.labels import ID_COLS, load_manifest, subject_labels  # noqa: E402
from adodas.data.pooled_table import TableConfig, build_table  # noqa: E402
from adodas.data.session_diffs import build_session_diffs  # noqa: E402
from adodas.data.ssl_pca import fit_ssl_pca, transform_ssl_pca, block_summary  # noqa: E402
from adodas.data.views import select_view  # noqa: E402
from adodas.utils.io import ensure_dir, load_yaml, save_json, save_yaml  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("01_build_features")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--paths-config", default="configs/paths.yaml")
    p.add_argument("--features-config", default="configs/features.yaml")
    p.add_argument("--feature-root", default=None, help="override feature_root in paths.yaml")
    p.add_argument("--manifest-dir", default=None, help="override manifest_dir in paths.yaml")
    p.add_argument("--output-dir", default=None, help="override output_dir in paths.yaml")
    p.add_argument("--splits", nargs="+", default=["train", "val", "test"])
    return p.parse_args()


def main() -> None:
    args = parse_args()
    paths = load_yaml(args.paths_config)
    feats = load_yaml(args.features_config)
    if args.feature_root:
        paths["feature_root"] = args.feature_root
    if args.manifest_dir:
        paths["manifest_dir"] = args.manifest_dir
    if args.output_dir:
        paths["output_dir"] = args.output_dir

    manifest_dir = Path(paths["manifest_dir"])
    output_dir = ensure_dir(Path(paths["output_dir"]) / "features")

    tcfg = TableConfig(
        feature_root=paths["feature_root"],
        sessions=tuple(paths["sessions"]),
        audio_features=tuple(paths["audio_features"]),
        video_features=tuple(paths["video_features"]),
        audio_ssl_model_tag=paths["audio_ssl_model_tag"],
        video_ssl_model_tag=paths["video_ssl_model_tag"],
        drop_high_na=feats.get("drop_high_na", 0.5),
        drop_zero_variance=feats.get("drop_zero_variance", True),
    )

    raw_tables: dict[str, pd.DataFrame] = {}
    for split in args.splits:
        manifest_path = manifest_dir / paths.get(f"manifest_{split}", f"{split}.csv")
        if not manifest_path.exists():
            log.warning(f"manifest not found for split={split}: {manifest_path}; skipping")
            continue
        manifest = load_manifest(manifest_path)
        log.info(f"split={split} manifest rows={len(manifest)}")
        df = build_table(manifest, tcfg, split=split, progress=True)
        raw_tables[split] = df
        df.to_parquet(output_dir / f"{split}_raw.parquet", index=False)
        log.info(f"  raw table → {output_dir / f'{split}_raw.parquet'}  shape={df.shape}")

    # Align column schema across splits (test may have a slightly different set).
    common_cols = set()
    for split, df in raw_tables.items():
        if not common_cols:
            common_cols = set(df.columns)
        else:
            common_cols &= set(df.columns)
    common_cols = sorted(common_cols)
    for split in raw_tables:
        head = [c for c in ID_COLS if c in common_cols]
        rest = [c for c in common_cols if c not in head]
        raw_tables[split] = raw_tables[split][head + rest]
        raw_tables[split].to_parquet(output_dir / f"{split}_raw.parquet", index=False)

    # PCA: either fit fresh on (train + val), or load a previously-saved fit
    # (so we can rebuild only the test split after the labeled run is done).
    labeled_splits = [s for s in ("train", "val") if s in raw_tables]
    pca_path = output_dir / "pca_blocks.pkl"
    if not labeled_splits:
        if not pca_path.exists():
            log.error(
                "no labeled splits available AND no saved pca_blocks.pkl — "
                "first run features on train+val (default --splits), then "
                "rerun for test alone"
            )
            sys.exit(1)
        with open(pca_path, "rb") as f:
            pca_blocks = pickle.load(f)
        log.info(f"loaded {len(pca_blocks)} saved PCA blocks from {pca_path}")
    else:
        df_labeled = pd.concat([raw_tables[s] for s in labeled_splits], axis=0, ignore_index=True)
        log.info(f"PCA fit pool rows={len(df_labeled)}")
        pca_cfg = feats.get("pca", {})
        pca_blocks = fit_ssl_pca(
            df_labeled,
            n_components_audio=int(pca_cfg.get("audio_ssl_dims", 96)),
            n_components_video=int(pca_cfg.get("video_ssl_dims", 96)),
            random_state=int(pca_cfg.get("random_state", 42)),
            whiten=bool(pca_cfg.get("whiten", False)),
        )
        with open(pca_path, "wb") as f:
            pickle.dump(pca_blocks, f)
        save_json(block_summary(pca_blocks), output_dir / "pca_summary.json")
        log.info(f"PCA blocks fitted: {len(pca_blocks)}")

    # Transform every split with the same PCA + add session diffs.
    diff_cfg = feats.get("session_diffs", {})
    pairs = [tuple(p) for p in diff_cfg.get("pairs", [])]
    add_mean3 = bool(diff_cfg.get("open_ended_mean", True))

    views_cfg = feats.get("views", {"fused": "all"})

    for split, df in raw_tables.items():
        df_pca = transform_ssl_pca(df, pca_blocks)
        df_pca = build_session_diffs(df_pca, pairs=pairs, add_open_ended_mean=add_mean3)

        for view_name in views_cfg.keys():
            view_df = select_view(
                df_pca, view_name,
                audio_features=paths["audio_features"],
                video_features=paths["video_features"],
            )
            # If the labeled run already wrote {view}_train.parquet, align this
            # split's columns to that schema. Without this, test-only rebuilds
            # can introduce/lose session-diff columns (depending on which
            # sessions exist), and stage 07's tree refit would error on shape.
            train_view_path = output_dir / f"{view_name}_train.parquet"
            if split != "train" and train_view_path.exists():
                ref_cols = list(pd.read_parquet(train_view_path).columns)
                view_df = view_df.reindex(columns=ref_cols)
            out = output_dir / f"{view_name}_{split}.parquet"
            view_df.to_parquet(out, index=False)
            log.info(f"split={split} view={view_name} → {out.name}  shape={view_df.shape}")

    # Build subject-level labels (train ∪ val). Skipped on a test-only rerun
    # since labels.parquet would already exist from the earlier labeled run.
    if labeled_splits:
        label_frames: list[pd.DataFrame] = []
        for split in labeled_splits:
            manifest_path = manifest_dir / paths.get(f"manifest_{split}", f"{split}.csv")
            if manifest_path.exists():
                mani = load_manifest(manifest_path)
                label_frames.append(subject_labels(mani))
        labels = pd.concat(label_frames, axis=0, ignore_index=True).drop_duplicates(subset=ID_COLS)
        labels.to_parquet(output_dir / "labels.parquet", index=False)
        log.info(f"subject_labels → {output_dir / 'labels.parquet'}  shape={labels.shape}")
    else:
        log.info("test-only rerun: keeping existing labels.parquet untouched")

    save_yaml({"splits": list(raw_tables.keys()), "views": list(views_cfg.keys())}, output_dir / "manifest.yaml")
    log.info("01_build_features done")


if __name__ == "__main__":
    main()
