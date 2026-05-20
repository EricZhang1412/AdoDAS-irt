"""Tree-model experts: per-item A2 regressors + per-target A1 classifiers.

Each (family, view) combo produces OOF columns under the shared 5-fold split.
A2 is regressed in continuous [0, 3] space; threshold optimization later turns
it into integers. A1 is binary classification — we keep probabilities, calibration
happens in stack/calibration.py.

The driver loop is generic enough to plug any sklearn-compatible estimator,
plus lightgbm and xgboost.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Callable

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier, ExtraTreesRegressor, HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from ..utils.dass21 import A1_COLS, ITEM_COLS
from .heads import sigmoid as _sigmoid

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Estimator factories. Each returns an unfitted sklearn-style model.

def _lgb_regressor(params: dict[str, Any]):
    import lightgbm as lgb
    return lgb.LGBMRegressor(**params)


def _lgb_classifier(params: dict[str, Any]):
    import lightgbm as lgb
    return lgb.LGBMClassifier(**params)


def _xgb_regressor(params: dict[str, Any]):
    import xgboost as xgb
    return xgb.XGBRegressor(**params)


def _xgb_classifier(params: dict[str, Any]):
    import xgboost as xgb
    return xgb.XGBClassifier(**params)


def _hgb_regressor(params: dict[str, Any]):
    return HistGradientBoostingRegressor(**params)


def _hgb_classifier(params: dict[str, Any]):
    return HistGradientBoostingClassifier(**params)


def _et_regressor(params: dict[str, Any]):
    return ExtraTreesRegressor(**params)


def _et_classifier(params: dict[str, Any]):
    return ExtraTreesClassifier(**params)


_FACTORIES: dict[str, tuple[Callable, Callable]] = {
    "lightgbm": (_lgb_regressor, _lgb_classifier),
    "xgboost": (_xgb_regressor, _xgb_classifier),
    "hgbt": (_hgb_regressor, _hgb_classifier),
    "extratrees": (_et_regressor, _et_classifier),
}


def make_regressor(family: str, params: dict[str, Any]):
    if family not in _FACTORIES:
        raise ValueError(f"Unknown tree family {family}")
    return _FACTORIES[family][0](params)


def make_classifier(family: str, params: dict[str, Any]):
    if family not in _FACTORIES:
        raise ValueError(f"Unknown tree family {family}")
    return _FACTORIES[family][1](params)


# ---------------------------------------------------------------------------
# Pre-processing.

def make_preprocessor(family: str) -> Pipeline:
    """Pipe: median impute → (standardize only for ExtraTrees-style models)."""
    steps: list[tuple[str, Any]] = [("imp", SimpleImputer(strategy="median"))]
    if family in ("extratrees",):
        steps.append(("scaler", StandardScaler(with_mean=True, with_std=True)))
    return Pipeline(steps)


# ---------------------------------------------------------------------------
# OOF training driver.

@dataclass
class OOFResult:
    family: str
    view: str
    a2_oof: np.ndarray             # (N, 21) continuous OOF predictions
    a1_oof: np.ndarray             # (N, 3) calibrated probability OOF
    a2_per_item: list[float]       # OOF QWK per item after argmin-rounding
    a1_per_target: list[float]     # OOF F1@0.5 per target


def train_oof(
    family: str,
    view: str,
    X: pd.DataFrame,
    y_a2: np.ndarray,
    y_a1: np.ndarray,
    folds: np.ndarray,
    reg_params: dict[str, Any],
    cls_params: dict[str, Any],
    n_folds: int = 5,
) -> OOFResult:
    """Train per-item regressors + per-target binary classifiers across folds.

    All 21 items / 3 targets share the same fold assignment. Returns continuous
    A2 predictions (no rounding) and calibrated A1 probabilities.
    """
    n = len(X)
    a2_oof = np.full((n, 21), np.nan, dtype=np.float32)
    a1_oof = np.full((n, 3), np.nan, dtype=np.float32)

    X_np = X.to_numpy(np.float32)

    for fold in range(n_folds):
        val_mask = folds == fold
        train_mask = ~val_mask
        X_tr, X_va = X_np[train_mask], X_np[val_mask]

        pre = make_preprocessor(family)
        pre.fit(X_tr)
        X_tr_p = pre.transform(X_tr)
        X_va_p = pre.transform(X_va)

        for j, col in enumerate(ITEM_COLS):
            y = y_a2[:, j]
            mask_tr = train_mask & np.isfinite(y) & (y >= 0)
            if mask_tr.sum() < 10:
                continue
            reg = make_regressor(family, reg_params)
            reg.fit(pre.transform(X_np[mask_tr]), y[mask_tr])
            a2_oof[val_mask, j] = np.clip(reg.predict(X_va_p), 0.0, 3.0)

        for t, col in enumerate(A1_COLS):
            y = y_a1[:, t]
            mask_tr = train_mask & np.isfinite(y) & (y >= 0)
            if mask_tr.sum() < 10:
                continue
            cls = make_classifier(family, cls_params)
            cls.fit(pre.transform(X_np[mask_tr]), y[mask_tr].astype(int))
            proba = _classifier_proba(cls, X_va_p)
            a1_oof[val_mask, t] = proba

    a2_per_item = _per_item_qwk_from_continuous(a2_oof, y_a2)
    a1_per_target = _per_target_f1(a1_oof, y_a1)
    return OOFResult(family, view, a2_oof, a1_oof, a2_per_item, a1_per_target)


def _classifier_proba(cls, X: np.ndarray) -> np.ndarray:
    """Predict positive-class probability robustly across sklearn / lgb / xgb."""
    if hasattr(cls, "predict_proba"):
        p = cls.predict_proba(X)
        return p[:, 1].astype(np.float32) if p.ndim == 2 and p.shape[1] >= 2 else p.astype(np.float32)
    if hasattr(cls, "decision_function"):
        return _sigmoid(cls.decision_function(X)).astype(np.float32)
    return cls.predict(X).astype(np.float32)


def _per_item_qwk_from_continuous(a2_oof: np.ndarray, y_a2: np.ndarray) -> list[float]:
    """Quick OOF QWK with naive 0.5/1.5/2.5 thresholds. Final QWK uses per-item
    threshold optimization (stack/thresholds.py)."""
    from ..utils.metrics import quadratic_weighted_kappa
    scores = []
    for j in range(a2_oof.shape[1]):
        pred = a2_oof[:, j]
        y = y_a2[:, j]
        valid = np.isfinite(pred) & np.isfinite(y) & (y >= 0)
        if valid.sum() == 0:
            scores.append(0.0)
            continue
        rounded = np.clip(np.round(pred[valid]), 0, 3).astype(int)
        scores.append(quadratic_weighted_kappa(y[valid].astype(int), rounded))
    return scores


def _per_target_f1(a1_oof: np.ndarray, y_a1: np.ndarray, threshold: float = 0.5) -> list[float]:
    from sklearn.metrics import f1_score
    scores = []
    for t in range(a1_oof.shape[1]):
        p = a1_oof[:, t]
        y = y_a1[:, t]
        valid = np.isfinite(p) & np.isfinite(y) & (y >= 0)
        if valid.sum() == 0:
            scores.append(0.0)
            continue
        pred = (p[valid] >= threshold).astype(int)
        scores.append(float(f1_score(y[valid].astype(int), pred, zero_division=0.0)))
    return scores


# ---------------------------------------------------------------------------
# Refit-on-all: produce test-time predictions after final fit on combined data.

def refit_and_predict(
    family: str,
    X_train_all: pd.DataFrame,
    y_a2: np.ndarray,
    y_a1: np.ndarray,
    X_test: pd.DataFrame,
    reg_params: dict[str, Any],
    cls_params: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray]:
    """Fit one regressor per A2 item and one classifier per A1 target on the
    full labeled pool, then predict on X_test. Used by stage-10-style refit."""
    pre = make_preprocessor(family)
    pre.fit(X_train_all.to_numpy(np.float32))
    Xtr_np = pre.transform(X_train_all.to_numpy(np.float32))
    Xte_np = pre.transform(X_test.to_numpy(np.float32))

    a2_pred = np.full((len(X_test), 21), np.nan, dtype=np.float32)
    for j in range(21):
        y = y_a2[:, j]
        mask = np.isfinite(y) & (y >= 0)
        if mask.sum() < 10:
            continue
        reg = make_regressor(family, reg_params)
        reg.fit(pre.transform(X_train_all.to_numpy(np.float32)[mask]), y[mask])
        a2_pred[:, j] = np.clip(reg.predict(Xte_np), 0.0, 3.0)

    a1_pred = np.full((len(X_test), 3), np.nan, dtype=np.float32)
    for t in range(3):
        y = y_a1[:, t]
        mask = np.isfinite(y) & (y >= 0)
        if mask.sum() < 10:
            continue
        cls = make_classifier(family, cls_params)
        cls.fit(pre.transform(X_train_all.to_numpy(np.float32)[mask]), y[mask].astype(int))
        a1_pred[:, t] = _classifier_proba(cls, Xte_np)

    return a2_pred, a1_pred
