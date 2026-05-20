"""Meta-blend per A2 item and per A1 target on OOF predictions.

Two estimators:
- per-item NNLS for A2 (non-negative weights, optional sum-to-≤1)
- per-target multi-output ridge for A1 (D/A/S share a 3×E coupling matrix)

Inputs are stacked OOF matrices of shape (N, K, E) where E = number of base
learners. Missing OOF entries (rows where a model couldn't train on that fold)
are imputed with the per-item mean before blending — those rows contribute
zero gradient to the NNLS anyway.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import nnls
from sklearn.linear_model import Ridge


@dataclass
class BlendResult:
    weights: np.ndarray              # (K, E)  for A2, or (3, E) for A1
    bias: np.ndarray | None          # (K,) optional intercept
    oof_blended: np.ndarray          # (N, K)


def blend_a2_nnls(
    oof: np.ndarray,
    y: np.ndarray,
    sum_to_one: bool = True,
    add_constant: bool = True,
) -> BlendResult:
    """Per-item non-negative least squares on continuous OOF.

    oof: (N, 21, E) — E base learners per item.
    y:   (N, 21)    — continuous targets in [0, 3].
    Returns weights (21, E), optional bias (21,), and blended OOF (N, 21).
    """
    N, I, E = oof.shape
    weights = np.zeros((I, E), dtype=np.float32)
    bias = np.zeros(I, dtype=np.float32) if add_constant else None
    blended = np.full((N, I), np.nan, dtype=np.float32)

    for j in range(I):
        X = oof[:, j, :]                      # (N, E)
        yj = y[:, j]
        mask = np.isfinite(yj) & (yj >= 0)
        if mask.sum() < 5:
            continue
        Xfit = _impute_columns(X[mask])
        yfit = yj[mask]
        if add_constant:
            Xfit = np.concatenate([Xfit, np.ones((Xfit.shape[0], 1), dtype=np.float32)], axis=1)
        w, _ = nnls(Xfit.astype(np.float64), yfit.astype(np.float64))
        w = w.astype(np.float32)
        if sum_to_one and w[:E].sum() > 1.0:
            w_part = w[:E]
            w_sum = w_part.sum()
            w[:E] = w_part / max(w_sum, 1e-8)
        weights[j] = w[:E]
        if add_constant:
            bias[j] = w[E]
        Xall = _impute_columns(X)
        blended[:, j] = Xall @ weights[j] + (bias[j] if add_constant else 0.0)
        blended[:, j] = np.clip(blended[:, j], 0.0, 3.0)

    return BlendResult(weights, bias, blended)


def blend_a1_multi_ridge(
    oof: np.ndarray,
    y: np.ndarray,
    alpha: float = 0.5,
) -> BlendResult:
    """Multi-output ridge on (N, 3, E) → (N, 3).

    Each of D/A/S can use information from the other two targets through the
    shared E columns — equivalently, we fit a single ridge on (N, 3*E) → (N, 3).
    """
    N, C, E = oof.shape
    assert C == 3
    flat = _impute_columns(oof.reshape(N, C * E))
    mask = np.isfinite(y).all(axis=1) & (y >= 0).all(axis=1)
    if mask.sum() < 5:
        return BlendResult(np.zeros((C, C * E)), None, np.full_like(y, 0.5))

    model = Ridge(alpha=alpha, fit_intercept=True).fit(flat[mask], y[mask])
    coef = model.coef_                        # (3, 3*E)
    intercept = model.intercept_              # (3,)
    blended = model.predict(flat).clip(0.0, 1.0).astype(np.float32)
    weights = coef.astype(np.float32)
    return BlendResult(weights, intercept.astype(np.float32), blended)


def apply_a2_blend(oof: np.ndarray, res: BlendResult) -> np.ndarray:
    """Apply learned A2 weights to a new OOF/test tensor of shape (N, 21, E)."""
    N, I, _ = oof.shape
    out = np.full((N, I), np.nan, dtype=np.float32)
    for j in range(I):
        X = _impute_columns(oof[:, j, :])
        bias_j = res.bias[j] if res.bias is not None else 0.0
        out[:, j] = np.clip(X @ res.weights[j] + bias_j, 0.0, 3.0)
    return out


def apply_a1_blend(oof: np.ndarray, res: BlendResult) -> np.ndarray:
    N, C, E = oof.shape
    flat = _impute_columns(oof.reshape(N, C * E))
    if res.bias is None:
        return (flat @ res.weights.T).clip(0.0, 1.0).astype(np.float32)
    return (flat @ res.weights.T + res.bias).clip(0.0, 1.0).astype(np.float32)


def _impute_columns(X: np.ndarray) -> np.ndarray:
    """Column-mean impute NaNs. Used because some base learners fail on
    very-low-sample items and emit NaN OOF on that fold."""
    X = X.astype(np.float32, copy=True)
    if not np.isnan(X).any():
        return X
    col_mean = np.nanmean(X, axis=0)
    col_mean = np.where(np.isfinite(col_mean), col_mean, 0.0)
    idx = np.where(np.isnan(X))
    X[idx] = col_mean[idx[1]]
    return X
