#!/usr/bin/env python3
"""Stage 07: refit on combined train+val, predict on test, apply meta + post-proc.

Writes the two submission CSVs into output/submission/.
"""
from __future__ import annotations

import argparse
import logging
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from adodas.data.labels import ID_COLS, split_features_and_labels  # noqa: E402
from adodas.models.heads import expected_score_from_cumlogits  # noqa: E402
from adodas.models.irt_joint import IRTConfig  # noqa: E402
from adodas.models.irt_trainer import TrainConfig  # noqa: E402
from adodas.stack.calibration import CalibrationResult, apply_isotonic, apply_logit_shift  # noqa: E402
from adodas.stack.meta_blend import BlendResult, apply_a1_blend, apply_a2_blend  # noqa: E402
from adodas.stack.thresholds import apply_thresholds  # noqa: E402
from adodas.submit.build_submission import write_a1_submission, write_a2_submission  # noqa: E402
from adodas.submit.refit_all_labeled import refit_irt, refit_trees  # noqa: E402
from adodas.utils.io import load_yaml  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("07_refit_predict")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--paths-config", default="configs/paths.yaml")
    p.add_argument("--trees-config", default="configs/trees.yaml")
    p.add_argument("--irt-config", default="configs/irt.yaml")
    p.add_argument("--stack-config", default="configs/stack.yaml")
    p.add_argument("--irt-runs", type=int, default=3)
    args = p.parse_args()

    paths = load_yaml(args.paths_config)
    trees_cfg = load_yaml(args.trees_config)
    irt_cfg = load_yaml(args.irt_config)
    stack_cfg = load_yaml(args.stack_config)

    output_dir = Path(paths["output_dir"])
    feature_dir = output_dir / "features"
    meta_dir = output_dir / "meta"
    folds_dir = output_dir / "folds"

    labels_df = pd.read_parquet(folds_dir / "labels_aligned.parquet")
    _, y_a1, y_a2, ids_labeled = split_features_and_labels(labels_df)

    # Load meta payloads.
    a2_blend_npz = np.load(meta_dir / "blend_a2.npz", allow_pickle=False)
    a2_blend = BlendResult(
        weights=a2_blend_npz["weights"],
        bias=a2_blend_npz["bias"] if a2_blend_npz["bias"].size else None,
        oof_blended=a2_blend_npz["oof_blended"],
    )
    a1_blend_npz = np.load(meta_dir / "blend_a1.npz", allow_pickle=False)
    a1_blend = BlendResult(
        weights=a1_blend_npz["weights"],
        bias=a1_blend_npz["bias"] if a1_blend_npz["bias"].size else None,
        oof_blended=a1_blend_npz["oof_blended"],
    )
    thr_npz = np.load(meta_dir / "thresholds.npz")
    thresholds = thr_npz["thresholds"]
    with open(meta_dir / "calibration.pkl", "rb") as f:
        cal_payload = pickle.load(f)
    cal_result = CalibrationResult(method=cal_payload["method"], payload=cal_payload["payload"], oof_calibrated=np.empty((0, 3), dtype=np.float32))

    # Build aligned views for train+val and test.
    views = list(trees_cfg["views"])
    views_train: dict[str, pd.DataFrame] = {}
    views_test: dict[str, pd.DataFrame] = {}
    for view in views:
        df_train = pd.read_parquet(feature_dir / f"{view}_train.parquet")
        if (feature_dir / f"{view}_val.parquet").exists():
            df_val = pd.read_parquet(feature_dir / f"{view}_val.parquet")
            df_train_all = pd.concat([df_train, df_val], axis=0, ignore_index=True)
        else:
            df_train_all = df_train
        df_train_all = ids_labeled.merge(df_train_all, on=ID_COLS, how="left")
        views_train[view] = df_train_all.drop(columns=[c for c in ID_COLS if c in df_train_all.columns]).reset_index(drop=True).fillna(0.0)

        test_path = feature_dir / f"{view}_test.parquet"
        if not test_path.exists():
            raise FileNotFoundError(f"missing {test_path}; rerun 01_build_features.py with the test split")
        df_test = pd.read_parquet(test_path)
        views_test[view] = df_test.reset_index(drop=True)

    test_ids = views_test[views[0]][[c for c in ID_COLS if c in views_test[views[0]].columns]].reset_index(drop=True)
    for view in views:
        views_test[view] = views_test[view].drop(columns=[c for c in ID_COLS if c in views_test[view].columns]).fillna(0.0)

    # Refit tree experts on full labeled pool.
    tree_pred = refit_trees(
        families=trees_cfg["families"],
        views=views_train,
        X_test_views=views_test,
        y_a2=y_a2,
        y_a1=y_a1,
        family_params=trees_cfg,
    )
    log.info(f"tree refit a2={tree_pred.a2.shape}, a1={tree_pred.a1.shape}")

    # Refit IRT on the fused view (matches OOF setup).
    fused_X_train = views_train["fused"].to_numpy(np.float32)
    fused_X_test = views_test["fused"].to_numpy(np.float32)
    anchors = irt_cfg.get("anchor_items", {"D": 2, "A": 1, "S": 0})
    model_cfg = IRTConfig(
        d_in=fused_X_train.shape[1],
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
    irt_pred = refit_irt(fused_X_train, y_a2, y_a1, fused_X_test, model_cfg, train_cfg, n_runs=int(args.irt_runs))
    log.info(f"irt refit a2={irt_pred.a2.shape}, a1={irt_pred.a1.shape}")

    # Combine test predictions in the same source order as 06_meta_blend.py used.
    sources = stack_cfg.get("sources", ["trees", "irt"])
    a2_parts: list[np.ndarray] = []
    a1_parts: list[np.ndarray] = []
    for src in sources:
        if src == "trees":
            a2_parts.append(tree_pred.a2)
            a1_parts.append(tree_pred.a1)
        elif src == "irt":
            a2_parts.append(irt_pred.a2)
            a1_parts.append(irt_pred.a1)

    a2_test_oof = np.concatenate(a2_parts, axis=-1).astype(np.float32)
    a1_test_oof = np.concatenate(a1_parts, axis=-1).astype(np.float32)

    # Apply meta-blend learned on OOF.
    a2_test_score = apply_a2_blend(a2_test_oof, a2_blend)
    a1_test_prob = apply_a1_blend(a1_test_oof, a1_blend)

    # Post-processing.
    a2_test_int = apply_thresholds(a2_test_score, thresholds)
    if cal_result.method == "isotonic":
        a1_test_cal = apply_isotonic(a1_test_prob, cal_result)
    else:
        a1_test_cal = apply_logit_shift(a1_test_prob, cal_result)

    a1_path = write_a1_submission(test_ids, a1_test_cal, output_dir)
    a2_path = write_a2_submission(test_ids, a2_test_int, output_dir)
    log.info(f"A1 submission → {a1_path}")
    log.info(f"A2 submission → {a2_path}")


if __name__ == "__main__":
    main()
