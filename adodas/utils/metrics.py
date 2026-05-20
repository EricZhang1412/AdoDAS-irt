"""Metrics — F1 (A1) and QWK (A2). Ported from backup/common/utils/metrics.py."""
from __future__ import annotations

import numpy as np
from sklearn.metrics import f1_score, mean_absolute_error, roc_auc_score


def binary_f1(probs: np.ndarray, labels: np.ndarray, threshold: float = 0.5) -> float:
    """Mean F1 across (N, C) binary targets at a fixed threshold."""
    preds = (probs >= threshold).astype(int)
    scores = []
    for c in range(probs.shape[1]):
        scores.append(f1_score(labels[:, c], preds[:, c], zero_division=0.0))
    return float(np.mean(scores))


def per_class_f1(probs: np.ndarray, labels: np.ndarray, threshold: float = 0.5) -> list[float]:
    preds = (probs >= threshold).astype(int)
    return [
        float(f1_score(labels[:, c], preds[:, c], zero_division=0.0))
        for c in range(probs.shape[1])
    ]


def macro_auroc(probs: np.ndarray, labels: np.ndarray) -> float:
    scores = []
    for c in range(probs.shape[1]):
        unique = np.unique(labels[:, c])
        if len(unique) < 2:
            scores.append(0.0)
        else:
            scores.append(float(roc_auc_score(labels[:, c], probs[:, c])))
    return float(np.mean(scores))


def quadratic_weighted_kappa(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int = 4) -> float:
    """QWK for one ordinal item. y_true / y_pred are 1-D int arrays in [0, num_classes)."""
    N = num_classes
    w = np.zeros((N, N), dtype=np.float64)
    for i in range(N):
        for j in range(N):
            w[i, j] = (i - j) ** 2 / ((N - 1) ** 2)

    hist_true = np.bincount(y_true, minlength=N).astype(np.float64)
    hist_pred = np.bincount(y_pred, minlength=N).astype(np.float64)
    n = len(y_true)

    O = np.zeros((N, N), dtype=np.float64)
    for t, p in zip(y_true, y_pred):
        O[int(t), int(p)] += 1

    E = np.outer(hist_true, hist_pred) / max(n, 1)
    num = np.sum(w * O)
    den = np.sum(w * E)
    if den == 0:
        return 1.0
    return float(1.0 - num / den)


def mean_qwk(preds: np.ndarray, labels: np.ndarray) -> float:
    """Mean QWK across (N, I) ordinal items."""
    return float(np.mean(per_item_qwk(preds, labels)))


def per_item_qwk(preds: np.ndarray, labels: np.ndarray) -> list[float]:
    return [quadratic_weighted_kappa(labels[:, c], preds[:, c]) for c in range(preds.shape[1])]


def mean_mae(preds: np.ndarray, labels: np.ndarray) -> float:
    scores = [float(mean_absolute_error(labels[:, c], preds[:, c])) for c in range(preds.shape[1])]
    return float(np.mean(scores))
