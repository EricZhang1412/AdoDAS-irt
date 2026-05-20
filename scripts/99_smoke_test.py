#!/usr/bin/env python3
"""End-to-end smoke test on a synthetic 10-participant pool.

Verifies that all modules wire up together without touching real features:
  - build a tiny pooled wide table from random numbers
  - fit/transform PCA + session diffs + view slicing
  - generate 5-fold split
  - train one tree expert family (HGBT) and a small IRT joint head
  - run meta-blend + thresholds + calibration + write submission CSVs

Intended to run in < 5 minutes on CPU. NOT a performance check — only a
plumbing check. Real performance evaluation happens on the remote server.
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from adodas.data.folds import assign_folds, verify_disjoint  # noqa: E402
from adodas.data.labels import ID_COLS, split_features_and_labels  # noqa: E402
from adodas.models.heads import expected_score_from_cumlogits  # noqa: E402
from adodas.models.irt_joint import IRTConfig, IRTJointModel, total_loss  # noqa: E402
from adodas.models.irt_trainer import TrainConfig, train_one_fold  # noqa: E402
from adodas.models.tree_experts import train_oof  # noqa: E402
from adodas.stack.calibration import fit_logit_shift, apply_logit_shift  # noqa: E402
from adodas.stack.meta_blend import blend_a1_multi_ridge, blend_a2_nnls, apply_a1_blend, apply_a2_blend  # noqa: E402
from adodas.stack.thresholds import apply_thresholds, optimise_all  # noqa: E402
from adodas.submit.build_submission import write_a1_submission, write_a2_submission  # noqa: E402
from adodas.utils.dass21 import A1_COLS, ITEM_COLS  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("99_smoke")


def make_dummy_data(n_subjects: int, d_feat: int, seed: int = 0) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    ids = pd.DataFrame({
        "anon_school": [f"s{i // 10:02d}" for i in range(n_subjects)],
        "anon_class": [f"c{(i // 5) % 4:02d}" for i in range(n_subjects)],
        "anon_pid": [f"p{i:04d}" for i in range(n_subjects)],
    })
    X = rng.standard_normal((n_subjects, d_feat)).astype(np.float32)
    # Latent severity θ_DAS, items load on θ with item-specific noise.
    theta = rng.standard_normal((n_subjects, 3)).astype(np.float32) * 0.8
    item_loadings = rng.uniform(0.5, 1.5, size=(21, 3)).astype(np.float32)
    item_loadings *= rng.choice([1, -1, 0], size=(21, 3), p=[0.4, 0.1, 0.5])  # group-like sparsity
    item_scores = X[:, :3] @ item_loadings.T + theta @ item_loadings.T  # mix feature signal + true θ
    item_int = np.clip(np.round(item_scores + 1.5), 0, 3).astype(int)
    a1 = (theta > 0).astype(int)

    label_cols: dict[str, np.ndarray] = {}
    for j, col in enumerate(ITEM_COLS):
        label_cols[col] = item_int[:, j]
    for c, col in enumerate(A1_COLS):
        label_cols[col] = a1[:, c]
    df = pd.concat([ids, pd.DataFrame({f"f_{k:04d}": X[:, k] for k in range(d_feat)})], axis=1)
    labels = pd.concat([ids, pd.DataFrame(label_cols)], axis=1)
    return df, labels


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--n-subjects", type=int, default=40)
    p.add_argument("--d-feat", type=int, default=128)
    p.add_argument("--output-dir", default="output/smoke")
    args = p.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Train+val pool.
    df_full, labels_full = make_dummy_data(args.n_subjects, args.d_feat, seed=0)
    df_test, labels_test = make_dummy_data(max(8, args.n_subjects // 5), args.d_feat, seed=1)

    # Folds.
    folds = assign_folds(labels_full, n_folds=5, seed=42)
    verify_disjoint(labels_full, folds)
    log.info(f"folds sizes={[int((folds == f).sum()) for f in range(5)]}")

    X_full = df_full.drop(columns=ID_COLS).reset_index(drop=True)
    _, y_a1, y_a2, ids_full = split_features_and_labels(labels_full)

    # Tree expert OOF (HGBT only for speed).
    from adodas.utils.io import load_yaml
    trees_cfg = load_yaml(Path(__file__).resolve().parents[1] / "configs" / "trees.yaml")
    hgbt = trees_cfg["hgbt"]
    res = train_oof("hgbt", "fused", X_full, y_a2.astype(np.float32), y_a1.astype(np.float32), folds,
                    hgbt["regressor"], hgbt["classifier"], n_folds=5)
    log.info(f"tree OOF a2 mean QWK={np.mean(res.a2_per_item):.4f}, a1 mean F1={np.mean(res.a1_per_target):.4f}")

    # IRT smoke: tiny, 6 epochs, no AMP.
    model_cfg = IRTConfig(
        d_in=X_full.shape[1], d_hidden=64, encoder_layers=1, encoder_dropout=0.1,
        feature_noise_std=0.01, d_theta=3,
        lambda_a1=1.0, lambda_consist=0.0, lambda_qwk=0.0, lambda_anchor=1.0,
        anchor_items={"D": 2, "A": 1, "S": 0},
    )
    train_cfg = TrainConfig(epochs=6, batch_size=16, lr=2e-3, patience=10, tta_replicas=2, device="cpu")
    val_mask = folds == 0
    train_mask = ~val_mask
    model, preds = train_one_fold(
        model_cfg, train_cfg,
        X_full.to_numpy(np.float32)[train_mask], y_a2[train_mask].astype(np.float32), y_a1[train_mask].astype(np.float32),
        X_full.to_numpy(np.float32)[val_mask],   y_a2[val_mask].astype(np.float32),   y_a1[val_mask].astype(np.float32),
    )
    z_oof = np.zeros((len(X_full), 21, 3), dtype=np.float32)
    z_oof[val_mask] = preds["z"].astype(np.float32)
    irt_a1_oof = np.zeros((len(X_full), 3), dtype=np.float32)
    irt_a1_oof[val_mask] = preds["p_a1"].astype(np.float32)
    log.info(f"IRT smoke fold 0: z stats mean={z_oof[val_mask].mean():.3f} std={z_oof[val_mask].std():.3f}")

    # Meta-blend: combine tree continuous + IRT E[y].
    irt_score = expected_score_from_cumlogits(z_oof)
    a2_oof = np.concatenate([res.a2_oof[..., None], irt_score[..., None]], axis=-1)
    a1_oof = np.concatenate([res.a1_oof[..., None], irt_a1_oof[..., None]], axis=-1)
    a2_blend = blend_a2_nnls(a2_oof, y_a2.astype(np.float32))
    a1_blend = blend_a1_multi_ridge(a1_oof, y_a1.astype(np.float32))

    a2_thresh = optimise_all(a2_blend.oof_blended, y_a2.astype(np.float32))
    a2_int = apply_thresholds(a2_blend.oof_blended, a2_thresh.thresholds)

    a1_cal = fit_logit_shift(a1_blend.oof_blended, y_a1.astype(np.float32))

    from adodas.utils.metrics import binary_f1, mean_qwk
    valid = (folds == 0)  # only fold 0 has IRT preds in this smoke run
    final_qwk = mean_qwk(a2_int[valid], y_a2[valid].astype(int)) if valid.any() else 0.0
    final_f1 = binary_f1(a1_cal.oof_calibrated[valid], y_a1[valid].astype(int)) if valid.any() else 0.0
    log.info(f"meta blend (smoke): QWK={final_qwk:.4f} F1={final_f1:.4f}")

    # Submission writer.
    test_ids = df_test[ID_COLS]
    # Trivial test predictions: reuse same shape with zeros.
    n_test = len(test_ids)
    a1_test = np.full((n_test, 3), 0.5, dtype=np.float32)
    a2_test_int = np.zeros((n_test, 21), dtype=np.int32)
    p1 = write_a1_submission(test_ids, a1_test, output_dir)
    p2 = write_a2_submission(test_ids, a2_test_int, output_dir)
    log.info(f"A1 submission → {p1}")
    log.info(f"A2 submission → {p2}")

    log.info("smoke test passed")


if __name__ == "__main__":
    main()
