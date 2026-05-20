#!/usr/bin/env python3
"""Stage 04: train the IRT/factor joint head OOF.

For each of the 5 folds:
  - fit on the union of other folds
  - predict CORAL z-logits and A1 probabilities on the held-out fold
  - average TTA noise replicas at inference

Outputs:
  oof/irt_a2_z.npy          (N, 21, 3)   CORAL cumulative logits
  oof/irt_a1.npy            (N, 3)       A1 probabilities (sigmoid)
  oof/irt_theta.npy         (N, 3)       Latent θ for diagnostics
  oof/irt_metrics.json      per-fold val QWK / F1 + mean
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from adodas.data.labels import ID_COLS, split_features_and_labels  # noqa: E402
from adodas.models.heads import expected_score_from_cumlogits, argmax_from_cumlogits  # noqa: E402
from adodas.models.irt_joint import IRTConfig  # noqa: E402
from adodas.models.irt_trainer import TrainConfig, train_one_fold  # noqa: E402
from adodas.utils.io import ensure_dir, load_yaml  # noqa: E402
from adodas.utils.metrics import binary_f1, mean_qwk  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("04_train_irt")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--paths-config", default="configs/paths.yaml")
    p.add_argument("--irt-config", default="configs/irt.yaml")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    paths = load_yaml(args.paths_config)
    irt_cfg = load_yaml(args.irt_config)

    output_dir = Path(paths["output_dir"])
    feature_dir = output_dir / "features"
    folds_dir = output_dir / "folds"
    oof_dir = ensure_dir(output_dir / "oof")

    folds = np.load(folds_dir / "folds.npy")
    n_folds = int(folds.max()) + 1
    labels_df = pd.read_parquet(folds_dir / "labels_aligned.parquet")
    _, y_a1, y_a2, ids = split_features_and_labels(labels_df)

    view = irt_cfg.get("view", "fused")
    df_train = pd.read_parquet(feature_dir / f"{view}_train.parquet")
    if (feature_dir / f"{view}_val.parquet").exists():
        df_val = pd.read_parquet(feature_dir / f"{view}_val.parquet")
        df = pd.concat([df_train, df_val], axis=0, ignore_index=True)
    else:
        df = df_train
    df = ids.merge(df, on=ID_COLS, how="left")
    X = df.drop(columns=[c for c in ID_COLS if c in df.columns]).fillna(0.0).to_numpy(np.float32)

    n = X.shape[0]
    z_oof = np.zeros((n, 21, 3), dtype=np.float32)
    a1_oof = np.zeros((n, 3), dtype=np.float32)
    theta_oof = np.zeros((n, 3), dtype=np.float32)
    per_fold: list[dict] = []

    anchors = irt_cfg.get("anchor_items", {"D": 2, "A": 1, "S": 0})
    model_cfg = IRTConfig(
        d_in=X.shape[1],
        d_hidden=int(irt_cfg.get("d_hidden", 256)),
        encoder_layers=int(irt_cfg.get("encoder_layers", 2)),
        encoder_dropout=float(irt_cfg.get("encoder_dropout", 0.3)),
        feature_noise_std=float(irt_cfg.get("feature_noise_std", 0.02)),
        d_theta=int(irt_cfg.get("d_theta", 3)),
        bifactor=bool(irt_cfg.get("bifactor", False)),
        label_smoothing=float(irt_cfg.get("label_smoothing", 0.05)),
        consistency_temperature=float(irt_cfg.get("consistency_temperature", 1.0)),
        lambda_a1=float(irt_cfg.get("lambda_a1", 1.0)),
        lambda_consist=float(irt_cfg.get("lambda_consist", 0.3)),
        lambda_qwk=float(irt_cfg.get("lambda_qwk", 0.2)),
        lambda_anchor=float(irt_cfg.get("lambda_anchor", 1.0)),
        anchor_items=anchors,
    )
    train_cfg = TrainConfig(
        epochs=int(irt_cfg.get("epochs", 80)),
        batch_size=int(irt_cfg.get("batch_size", 32)),
        lr=float(irt_cfg.get("lr", 1e-3)),
        weight_decay=float(irt_cfg.get("weight_decay", 0.01)),
        warmup_epochs=int(irt_cfg.get("warmup_epochs", 3)),
        patience=int(irt_cfg.get("patience", 12)),
        grad_clip=float(irt_cfg.get("grad_clip", 1.0)),
        amp=bool(irt_cfg.get("amp", False)),
        seed=int(irt_cfg.get("seed", 42)),
        device="cuda" if torch.cuda.is_available() else "cpu",
        tta_replicas=int(irt_cfg.get("tta_replicas", 8)),
        tta_noise_std=float(irt_cfg.get("tta_noise_std", 0.01)),
    )

    seed_offset = int(irt_cfg.get("fold_seed_offset", 1000))
    valid = np.isfinite(y_a2).all(axis=1) & np.isfinite(y_a1).all(axis=1) & (y_a2 >= 0).all(axis=1) & (y_a1 >= 0).all(axis=1)

    for fold in range(n_folds):
        train_cfg.seed = train_cfg.seed + seed_offset * 0 + fold * 7919
        val_mask = (folds == fold) & valid
        train_mask = (folds != fold) & valid
        if val_mask.sum() == 0 or train_mask.sum() == 0:
            continue
        log.info(f"fold={fold} train_n={int(train_mask.sum())} val_n={int(val_mask.sum())}")
        model, preds = train_one_fold(
            model_cfg, train_cfg,
            X[train_mask], y_a2[train_mask], y_a1[train_mask],
            X[val_mask],   y_a2[val_mask],   y_a1[val_mask],
        )
        z_oof[val_mask] = preds["z"].astype(np.float32)
        a1_oof[val_mask] = preds["p_a1"].astype(np.float32)
        theta_oof[val_mask] = preds["theta"].astype(np.float32)

        val_int = argmax_from_cumlogits(preds["z"])
        qwk = mean_qwk(val_int, y_a2[val_mask].astype(int))
        f1 = binary_f1(preds["p_a1"], y_a1[val_mask].astype(int))
        per_fold.append({"fold": fold, "val_qwk": qwk, "val_f1": f1, "n_val": int(val_mask.sum())})
        log.info(f"  fold={fold} val_qwk={qwk:.4f} val_f1={f1:.4f}")

    # Per-fold continuous A2 score (E[y]) for downstream stacking.
    a2_continuous_oof = expected_score_from_cumlogits(z_oof)

    np.save(oof_dir / "irt_a2_z.npy", z_oof)
    np.save(oof_dir / "irt_a2_score.npy", a2_continuous_oof.astype(np.float32))
    np.save(oof_dir / "irt_a1.npy", a1_oof)
    np.save(oof_dir / "irt_theta.npy", theta_oof)

    summary = {
        "per_fold": per_fold,
        "mean_qwk": float(np.mean([f["val_qwk"] for f in per_fold])) if per_fold else 0.0,
        "mean_f1": float(np.mean([f["val_f1"] for f in per_fold])) if per_fold else 0.0,
    }
    with open(oof_dir / "irt_metrics.json", "w") as f:
        json.dump(summary, f, indent=2)
    log.info(f"IRT OOF done: mean QWK={summary['mean_qwk']:.4f}, mean F1={summary['mean_f1']:.4f}")


if __name__ == "__main__":
    main()
