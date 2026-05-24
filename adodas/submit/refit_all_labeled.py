"""Stage-10-style final fit on combined train+val data.

After OOF-driven hyperparameter selection, we refit every base learner on the
union of train+val labels, predict on test, then apply the meta-blend weights
and post-processing (thresholds for A2 / calibration for A1) learned from OOF.
This squeezes ~3-5% more training data per fold-worth without compromising the
fold-disjoint validation that informed model choice.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import torch

from ..models.irt_joint import IRTConfig, IRTJointModel
from ..models.irt_trainer import TrainConfig, train_one_fold
from ..models.tree_experts import refit_and_predict

log = logging.getLogger(__name__)


@dataclass
class RefitPrediction:
    a2: np.ndarray   # (N_test, 21, E)  one column per base learner
    a1: np.ndarray   # (N_test, 3, E)
    source_names: list[str]


def refit_trees(
    families: Iterable[str],
    views: dict[str, pd.DataFrame],
    X_test_views: dict[str, pd.DataFrame],
    y_a2: np.ndarray,
    y_a1: np.ndarray,
    family_params: dict[str, dict],
) -> RefitPrediction:
    """For every (family, view), refit and predict on test."""
    a2_list: list[np.ndarray] = []
    a1_list: list[np.ndarray] = []
    names: list[str] = []
    import time
    for family in families:
        params = family_params[family]
        for view_name, X_tr in views.items():
            X_te = X_test_views[view_name]
            log.info(
                f"refit START: {family} × {view_name}  "
                f"X_tr={X_tr.shape} X_te={X_te.shape} (24 models: 21 A2 + 3 A1)"
            )
            t0 = time.time()
            a2_pred, a1_pred = refit_and_predict(
                family, X_tr, y_a2, y_a1, X_te,
                params["regressor"], params["classifier"],
            )
            a2_list.append(a2_pred)
            a1_list.append(a1_pred)
            names.append(f"{family}__{view_name}")
            log.info(f"refit done : {family} × {view_name}  ({time.time() - t0:.1f}s)")
    a2 = np.stack(a2_list, axis=-1).astype(np.float32)
    a1 = np.stack(a1_list, axis=-1).astype(np.float32)
    return RefitPrediction(a2, a1, names)


def refit_irt(
    X_train_all: np.ndarray,
    y_a2: np.ndarray,
    y_a1: np.ndarray,
    X_test: np.ndarray,
    model_cfg: IRTConfig,
    train_cfg: TrainConfig,
    n_runs: int = 3,
) -> RefitPrediction:
    """Refit IRT joint head on full labeled pool with `n_runs` seeded restarts.

    Each restart contributes one (a2, a1) prediction column; averaging across
    restarts dampens the variance from random init.
    """
    a2_runs: list[np.ndarray] = []
    a1_runs: list[np.ndarray] = []
    for run in range(n_runs):
        run_cfg = TrainConfig(**{**train_cfg.__dict__, "seed": train_cfg.seed + run})
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Relaxed mask: keep a subject as long as it has at least one valid A2
        # item AND at least one valid A1 target. Per-position NaN masking inside
        # the loss (irt_joint.coral_bce / a1_bce) handles individual missing
        # cells. The old strict ALL-valid mask was wiping out 90%+ of subjects.
        any_a2 = (np.isfinite(y_a2) & (y_a2 >= 0)).any(axis=1)
        any_a1 = (np.isfinite(y_a1) & (y_a1 >= 0)).any(axis=1)
        mask = any_a2 & any_a1
        if run == 0:
            log.info(
                f"IRT mask: {int(mask.sum())} / {len(mask)} subjects survive "
                f"(any-valid-item, was previously all-valid)"
            )
        X = X_train_all[mask]
        ya2 = y_a2[mask]
        ya1 = y_a1[mask]

        # Use last 10% as a tiny val for early stopping.
        n = len(X)
        n_val = max(8, n // 10)
        idx = np.random.RandomState(run_cfg.seed).permutation(n)
        val_idx = idx[:n_val]
        tr_idx = idx[n_val:]

        from ..models.irt_trainer import train_one_fold as _trf
        model, _ = _trf(
            model_cfg, run_cfg,
            X[tr_idx], ya2[tr_idx], ya1[tr_idx],
            X[val_idx], ya2[val_idx], ya1[val_idx],
        )
        preds = _predict_test(model, X_test, run_cfg, device)
        a2_runs.append(preds["a2_continuous"])
        a1_runs.append(preds["a1"])
        log.info(f"IRT refit run {run + 1}/{n_runs} done")

    a2_mean = np.mean(np.stack(a2_runs, axis=-1), axis=-1)
    a1_mean = np.mean(np.stack(a1_runs, axis=-1), axis=-1)
    # Treat the mean as one column.
    return RefitPrediction(a2_mean[..., None], a1_mean[..., None], ["irt"])


def _predict_test(model: IRTJointModel, X: np.ndarray, train_cfg: TrainConfig, device: torch.device) -> dict[str, np.ndarray]:
    """Predict A2 continuous score + A1 sigmoid prob on test."""
    from ..models.irt_trainer import predict_with_tta
    out = predict_with_tta(model, X, train_cfg, device)
    from ..models.heads import expected_score_from_cumlogits
    a2 = expected_score_from_cumlogits(out["z"])
    return {"a2_continuous": a2, "a1": out["p_a1"]}
