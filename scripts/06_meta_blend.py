#!/usr/bin/env python3
"""Stage 06: meta-blend tree + IRT OOFs; optimise A2 thresholds; calibrate A1.

Outputs:
  meta/blend_a2.npz    weights/bias arrays + blended OOF continuous score
  meta/blend_a1.npz    weights/bias + calibrated OOF probs
  meta/thresholds.npz  per-item ordinal thresholds + per-item achieved QWK
  meta/calibration.pkl fitted isotonic or logit_shift
  meta/summary.json    final OOF QWK / F1 (the headline numbers)
"""
from __future__ import annotations

import argparse
import json
import logging
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from adodas.data.labels import split_features_and_labels  # noqa: E402
from adodas.models.heads import expected_score_from_cumlogits  # noqa: E402
from adodas.stack.calibration import fit_isotonic, fit_logit_shift  # noqa: E402
from adodas.stack.meta_blend import blend_a1_multi_ridge, blend_a2_nnls  # noqa: E402
from adodas.stack.thresholds import apply_thresholds, optimise_all  # noqa: E402
from adodas.utils.io import ensure_dir, load_yaml  # noqa: E402
from adodas.utils.metrics import binary_f1, mean_qwk  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("06_meta_blend")


def stack_oofs(output_dir: Path, sources: list[str]) -> tuple[np.ndarray, np.ndarray]:
    oof_dir = output_dir / "oof"
    a2_parts: list[np.ndarray] = []
    a1_parts: list[np.ndarray] = []

    for src in sources:
        if src == "trees":
            a2_parts.append(np.load(oof_dir / "trees_a2.npy"))
            a1_parts.append(np.load(oof_dir / "trees_a1.npy"))
        elif src == "irt":
            # Use IRT continuous E[y] for A2 blending alongside tree continuous outputs.
            a2_score = np.load(oof_dir / "irt_a2_score.npy")
            a2_parts.append(a2_score[..., None])
            a1 = np.load(oof_dir / "irt_a1.npy")
            a1_parts.append(a1[..., None])
        else:
            raise ValueError(f"unknown source {src}")

    a2 = np.concatenate(a2_parts, axis=-1).astype(np.float32)
    a1 = np.concatenate(a1_parts, axis=-1).astype(np.float32)
    return a2, a1


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--paths-config", default="configs/paths.yaml")
    p.add_argument("--stack-config", default="configs/stack.yaml")
    args = p.parse_args()

    paths = load_yaml(args.paths_config)
    stack_cfg = load_yaml(args.stack_config)

    output_dir = Path(paths["output_dir"])
    folds_dir = output_dir / "folds"
    meta_dir = ensure_dir(output_dir / "meta")

    labels_df = pd.read_parquet(folds_dir / "labels_aligned.parquet")
    _, y_a1, y_a2, _ = split_features_and_labels(labels_df)

    sources = stack_cfg.get("sources", ["trees", "irt"])
    a2_oof, a1_oof = stack_oofs(output_dir, sources)
    log.info(f"stacked OOF: a2 shape={a2_oof.shape}, a1 shape={a1_oof.shape}")

    a2_meta_cfg = stack_cfg.get("a2_meta", {})
    a2_blend = blend_a2_nnls(
        a2_oof, y_a2,
        sum_to_one=bool(a2_meta_cfg.get("sum_to_one", True)),
        add_constant=True,
    )
    log.info(f"A2 blend weights mean={a2_blend.weights.mean():.4f}, weight magnitudes example=\n{a2_blend.weights[:3]}")

    a1_meta_cfg = stack_cfg.get("a1_meta", {})
    a1_blend = blend_a1_multi_ridge(
        a1_oof, y_a1,
        alpha=float(a1_meta_cfg.get("ridge_alpha", 0.5)),
    )

    # A2 per-item threshold optimization on blended continuous OOF.
    a2_thresh = optimise_all(
        a2_blend.oof_blended, y_a2,
        init=tuple(stack_cfg.get("a2_thresholds", {}).get("init", [0.5, 1.5, 2.5])),
        bounds_pad=float(stack_cfg.get("a2_thresholds", {}).get("bounds_pad", 0.4)),
    )
    a2_int = apply_thresholds(a2_blend.oof_blended, a2_thresh.thresholds)

    # A1 calibration (pick one).
    a1_cal_cfg = stack_cfg.get("a1_calibration", {"method": "logit_shift"})
    if a1_cal_cfg["method"] == "isotonic":
        a1_cal = fit_isotonic(a1_blend.oof_blended, y_a1)
    else:
        a1_cal = fit_logit_shift(a1_blend.oof_blended, y_a1)

    valid_a2 = np.isfinite(y_a2).all(axis=1) & (y_a2 >= 0).all(axis=1)
    valid_a1 = np.isfinite(y_a1).all(axis=1) & (y_a1 >= 0).all(axis=1)
    final_qwk = mean_qwk(a2_int[valid_a2], y_a2[valid_a2].astype(int))
    final_f1 = binary_f1(a1_cal.oof_calibrated[valid_a1], y_a1[valid_a1].astype(int))
    log.info(f"meta OOF QWK={final_qwk:.4f}  F1={final_f1:.4f}")

    np.savez(meta_dir / "blend_a2.npz",
             weights=a2_blend.weights, bias=a2_blend.bias if a2_blend.bias is not None else np.zeros(0),
             oof_blended=a2_blend.oof_blended)
    np.savez(meta_dir / "blend_a1.npz",
             weights=a1_blend.weights, bias=a1_blend.bias if a1_blend.bias is not None else np.zeros(0),
             oof_blended=a1_blend.oof_blended,
             oof_calibrated=a1_cal.oof_calibrated)
    np.savez(meta_dir / "thresholds.npz",
             thresholds=a2_thresh.thresholds, per_item_qwk=a2_thresh.qwk)
    with open(meta_dir / "calibration.pkl", "wb") as f:
        pickle.dump({"method": a1_cal.method, "payload": a1_cal.payload}, f)
    with open(meta_dir / "summary.json", "w") as f:
        json.dump({
            "sources": sources,
            "qwk": final_qwk,
            "f1": final_f1,
            "per_item_qwk": [float(x) for x in a2_thresh.qwk],
            "a1_calibration": a1_cal.method,
        }, f, indent=2)
    log.info(f"meta blend done → {meta_dir}")


if __name__ == "__main__":
    main()
