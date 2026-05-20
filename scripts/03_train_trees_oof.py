#!/usr/bin/env python3
"""Stage 03: train tree-expert OOF for A1 and A2.

For every (family, view) combo: per-item regressors (A2, 21) + per-target
binary classifiers (A1, 3). All share the same 5-fold split from 02.

Outputs:
  oof/trees_a2.npy          (N, 21, E_a2)  continuous OOF in [0,3]
  oof/trees_a1.npy          (N, 3,  E_a1)  probability OOF
  oof/trees_sources.json    column → (family, view) mapping for stacking
  oof/trees_metrics.json    per-(family,view) per-item OOF QWK / F1
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from adodas.data.labels import ID_COLS, split_features_and_labels  # noqa: E402
from adodas.models.tree_experts import train_oof  # noqa: E402
from adodas.utils.io import ensure_dir, load_yaml  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("03_train_trees")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--paths-config", default="configs/paths.yaml")
    p.add_argument("--trees-config", default="configs/trees.yaml")
    p.add_argument("--families", nargs="+", default=None)
    p.add_argument("--views", nargs="+", default=None)
    p.add_argument("--items", nargs="+", default=None,
                   help="optional: restrict to a subset of items for smoke runs (e.g. d03)")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    paths = load_yaml(args.paths_config)
    trees_cfg = load_yaml(args.trees_config)

    output_dir = Path(paths["output_dir"])
    feature_dir = output_dir / "features"
    folds_dir = output_dir / "folds"
    oof_dir = ensure_dir(output_dir / "oof")

    folds = np.load(folds_dir / "folds.npy")
    n_folds = int(folds.max()) + 1
    labels_df = pd.read_parquet(folds_dir / "labels_aligned.parquet")
    _, y_a1, y_a2, ids = split_features_and_labels(labels_df)

    families = args.families or trees_cfg["families"]
    views = args.views or trees_cfg["views"]

    a2_columns: list[np.ndarray] = []
    a1_columns: list[np.ndarray] = []
    source_names: list[dict[str, str]] = []
    metrics_log: dict[str, dict] = {}

    for view in views:
        feat_path = feature_dir / f"{view}_train.parquet"
        val_path = feature_dir / f"{view}_val.parquet"
        if not feat_path.exists():
            log.warning(f"missing {feat_path}; skip view={view}")
            continue
        df_train = pd.read_parquet(feat_path)
        if val_path.exists():
            df_val = pd.read_parquet(val_path)
            df = pd.concat([df_train, df_val], axis=0, ignore_index=True)
        else:
            df = df_train
        # Align to canonical participant order via merge against ids.
        df = ids.merge(df, on=ID_COLS, how="left")
        X = df.drop(columns=[c for c in ID_COLS if c in df.columns]).reset_index(drop=True)

        for family in families:
            params = trees_cfg[family]
            log.info(f"train: family={family} view={view} X.shape={X.shape}")
            res = train_oof(
                family, view, X, y_a2, y_a1, folds,
                params["regressor"], params["classifier"],
                n_folds=n_folds,
            )
            a2_columns.append(res.a2_oof)
            a1_columns.append(res.a1_oof)
            source_names.append({"family": family, "view": view})
            metrics_log[f"{family}__{view}"] = {
                "a2_per_item": [float(x) for x in res.a2_per_item],
                "a2_mean_qwk": float(np.mean(res.a2_per_item)),
                "a1_per_target": [float(x) for x in res.a1_per_target],
                "a1_mean_f1": float(np.mean(res.a1_per_target)),
            }
            log.info(
                f"  done {family} × {view}: "
                f"a2_mean_qwk={metrics_log[f'{family}__{view}']['a2_mean_qwk']:.4f} "
                f"a1_mean_f1={metrics_log[f'{family}__{view}']['a1_mean_f1']:.4f}"
            )

    a2_arr = np.stack(a2_columns, axis=-1).astype(np.float32)  # (N, 21, E)
    a1_arr = np.stack(a1_columns, axis=-1).astype(np.float32)  # (N, 3, E)

    np.save(oof_dir / "trees_a2.npy", a2_arr)
    np.save(oof_dir / "trees_a1.npy", a1_arr)
    with open(oof_dir / "trees_sources.json", "w") as f:
        json.dump(source_names, f, indent=2)
    with open(oof_dir / "trees_metrics.json", "w") as f:
        json.dump(metrics_log, f, indent=2)
    log.info(f"trees_a2.npy shape={a2_arr.shape}, trees_a1.npy shape={a1_arr.shape}")


if __name__ == "__main__":
    main()
