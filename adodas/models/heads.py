"""Shared head utilities used by both the IRT joint model and the meta stage.

This is the New-World version of backup/common/models/heads.py — only the
pieces we actually need for the hybrid stack. Names are kept compatible with
the legacy code so we can import the same DASS-21 grouping verbatim.

Numerically: sigmoid and CORAL monotonic decoding are both in numpy here so
they work outside of torch (calibration / threshold opt / submission writer).
The torch version lives in irt_joint.py.
"""
from __future__ import annotations

import numpy as np

from ..utils.dass21 import (
    DASS21_GROUP_CUTOFFS,
    DASS21_GROUP_ITEMS,
    DASS21_GROUP_ORDER,
)

__all__ = [
    "sigmoid",
    "logit",
    "expected_score_from_cumlogits",
    "monotonic_class_probs",
    "argmax_from_cumlogits",
    "expected_score_from_class_probs",
    "soft_dass21_from_expected",
    "DASS21_GROUP_ITEMS",
    "DASS21_GROUP_CUTOFFS",
    "DASS21_GROUP_ORDER",
]


def sigmoid(x: np.ndarray | float) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    out = np.empty_like(x)
    pos = x >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-x[pos]))
    ex = np.exp(x[~pos])
    out[~pos] = ex / (1.0 + ex)
    return out.astype(np.float32)


def logit(p: np.ndarray | float, eps: float = 1e-6) -> np.ndarray:
    p = np.clip(np.asarray(p, dtype=np.float64), eps, 1.0 - eps)
    return np.log(p / (1.0 - p)).astype(np.float32)


def expected_score_from_cumlogits(z: np.ndarray) -> np.ndarray:
    """E[Y] = sum_k sigmoid(z_k). Works for any shape ending in (n_thresholds,)."""
    return sigmoid(z).sum(axis=-1)


def monotonic_class_probs(z: np.ndarray) -> np.ndarray:
    """CORAL cumulative-logit → simplex over {0, 1, 2, 3}. Last axis is thresholds.

    Returns array of shape z.shape[:-1] + (z.shape[-1] + 1,).
    """
    s = sigmoid(z)
    p1 = s[..., 0]
    p2 = np.minimum(s[..., 1], p1)
    p3 = np.minimum(s[..., 2], p2)
    P0 = 1.0 - p1
    P1 = p1 - p2
    P2 = p2 - p3
    P3 = p3
    probs = np.stack([P0, P1, P2, P3], axis=-1).clip(0.0)
    probs = probs / probs.sum(axis=-1, keepdims=True).clip(1e-8)
    return probs


def argmax_from_cumlogits(z: np.ndarray) -> np.ndarray:
    return monotonic_class_probs(z).argmax(axis=-1).astype(np.int32)


def expected_score_from_class_probs(probs: np.ndarray) -> np.ndarray:
    K = probs.shape[-1]
    grid = np.arange(K, dtype=np.float32)
    return (probs * grid).sum(axis=-1)


def soft_dass21_from_expected(
    e_y: np.ndarray,
    temperature: float = 1.0,
) -> np.ndarray:
    """Take per-item E[Y] (shape (..., 21)) → soft (..., 3) [D, A, S] indicators.

    Implements the official DASS-21 rule: subscale_sum = 2 * Σ items_in_group,
    then sigmoid around the cutoff. This is what dass21_consistency_loss does
    in backup/common/models/heads.py — we keep the numpy version here for
    post-hoc analyses and the torch version in irt_joint.py.
    """
    inv_tau = 1.0 / max(float(temperature), 1e-6)
    out = []
    for g in DASS21_GROUP_ORDER:
        idx = np.asarray(DASS21_GROUP_ITEMS[g], dtype=np.int64)
        s_g = e_y[..., idx].sum(axis=-1)
        out.append(sigmoid((2.0 * s_g - DASS21_GROUP_CUTOFFS[g]) * inv_tau))
    return np.stack(out, axis=-1).astype(np.float32)
