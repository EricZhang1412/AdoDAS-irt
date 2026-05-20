#!/usr/bin/env python3
"""Stage 02: generate subject-disjoint 5-fold CV split.

Outputs:
  folds/folds.npy           — (N_participants,) int array of fold ids
  folds/labels_aligned.parquet — labels reordered to match folds.npy index
  folds/summary.json        — per-fold size + stratification distribution

The fold assignment is participant-level. Downstream stages must align their
participant tables (one row per participant) to this ordering before consuming.
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

from adodas.data.folds import assign_folds, verify_disjoint  # noqa: E402
from adodas.data.labels import ID_COLS  # noqa: E402
from adodas.utils.io import ensure_dir, load_yaml  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("02_make_folds")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--paths-config", default="configs/paths.yaml")
    p.add_argument("--output-dir", default=None)
    p.add_argument("--n-folds", type=int, default=None)
    p.add_argument("--seed", type=int, default=None)
    args = p.parse_args()

    paths = load_yaml(args.paths_config)
    output_dir = Path(args.output_dir or paths["output_dir"])
    feature_dir = output_dir / "features"
    folds_dir = ensure_dir(output_dir / "folds")

    labels_path = feature_dir / "labels.parquet"
    if not labels_path.exists():
        log.error(f"missing labels.parquet — run 01_build_features.py first ({labels_path})")
        sys.exit(1)
    labels = pd.read_parquet(labels_path)

    # Align labels with the canonical participant order from features/fused_train.parquet
    # so downstream OOF indexing is consistent.
    fused_train = pd.read_parquet(feature_dir / "fused_train.parquet")
    if (feature_dir / "fused_val.parquet").exists():
        fused_val = pd.read_parquet(feature_dir / "fused_val.parquet")
        canonical_ids = pd.concat(
            [fused_train[ID_COLS], fused_val[ID_COLS]],
            axis=0,
            ignore_index=True,
        ).drop_duplicates(subset=ID_COLS).reset_index(drop=True)
    else:
        canonical_ids = fused_train[ID_COLS].drop_duplicates().reset_index(drop=True)

    aligned = canonical_ids.merge(labels, on=ID_COLS, how="left")
    log.info(f"aligned label rows: {len(aligned)} (labels missing: {aligned['d01'].isna().sum() if 'd01' in aligned.columns else 'n/a'})")

    n_folds = args.n_folds or int(paths.get("n_folds", 5))
    seed = args.seed if args.seed is not None else int(paths.get("fold_seed", 42))

    folds = assign_folds(aligned, n_folds=n_folds, seed=seed)
    verify_disjoint(aligned, folds)

    np.save(folds_dir / "folds.npy", folds)
    aligned.to_parquet(folds_dir / "labels_aligned.parquet", index=False)

    summary = {
        "n_folds": int(n_folds),
        "seed": int(seed),
        "n_participants": int(len(aligned)),
        "fold_sizes": [int((folds == f).sum()) for f in range(n_folds)],
    }
    with open(folds_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    log.info(f"folds.npy → {folds_dir}  sizes={summary['fold_sizes']}")


if __name__ == "__main__":
    main()
