"""Per-item ordinal threshold optimization for A2.

Given continuous OOF scores in [0, 3] and integer labels in {0, 1, 2, 3},
find three monotone thresholds (t1 < t2 < t3) that maximise QWK on this item.

We use a coarse grid + golden-section refinement on a 1-D loss surface, fixing
the other two thresholds at their previous estimate (coordinate-wise descent).
This is much more stable than Nelder-Mead in 3-D for samples in the low hundreds.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..utils.metrics import quadratic_weighted_kappa


@dataclass
class ThresholdResult:
    thresholds: np.ndarray   # (21, 3) per-item t1/t2/t3
    qwk: np.ndarray          # (21,)   per-item achieved QWK


def threshold_to_int(score: np.ndarray, t: np.ndarray) -> np.ndarray:
    """score: (N,), t: (3,) sorted ascending. Output ints in {0,1,2,3}."""
    out = np.zeros_like(score, dtype=np.int32)
    out[score >= t[0]] = 1
    out[score >= t[1]] = 2
    out[score >= t[2]] = 3
    return out


def _eval_qwk(score: np.ndarray, t: np.ndarray, y: np.ndarray) -> float:
    pred = threshold_to_int(score, t)
    return quadratic_weighted_kappa(y, pred)


def optimise_per_item(
    score: np.ndarray,
    y: np.ndarray,
    init: tuple[float, float, float] = (0.5, 1.5, 2.5),
    bounds_pad: float = 0.4,
    coarse_grid: int = 9,
    n_passes: int = 3,
) -> tuple[np.ndarray, float]:
    """Optimise (t1, t2, t3) for one item by coordinate-wise grid + golden-section.

    The output is sorted ascending so threshold_to_int is well defined.
    """
    valid = np.isfinite(score) & np.isfinite(y) & (y >= 0)
    if valid.sum() < 5 or len(np.unique(y[valid])) < 2:
        return np.asarray(init, dtype=np.float32), 0.0

    s = score[valid].astype(np.float32)
    y_int = y[valid].astype(np.int32)
    t = np.asarray(init, dtype=np.float32)

    bounds = np.stack([t - bounds_pad, t + bounds_pad], axis=1)
    bounds[:, 0] = np.maximum(bounds[:, 0], 0.0)
    bounds[:, 1] = np.minimum(bounds[:, 1], 3.0)

    for _ in range(n_passes):
        for k in range(3):
            grid = np.linspace(bounds[k, 0], bounds[k, 1], coarse_grid)
            best_t, best_q = t[k], _eval_qwk(s, _ordered(t), y_int)
            for cand in grid:
                t_try = t.copy()
                t_try[k] = cand
                q = _eval_qwk(s, _ordered(t_try), y_int)
                if q > best_q:
                    best_q = q
                    best_t = cand
            t[k] = best_t

            # Golden-section refine inside (best_t - step, best_t + step)
            step = (bounds[k, 1] - bounds[k, 0]) / (coarse_grid - 1)
            lo, hi = max(bounds[k, 0], best_t - step), min(bounds[k, 1], best_t + step)
            t[k] = _golden(s, t, y_int, k, lo, hi)

    return _ordered(t), _eval_qwk(s, _ordered(t), y_int)


def _golden(s: np.ndarray, t: np.ndarray, y_int: np.ndarray, k: int, lo: float, hi: float, tol: float = 1e-3) -> float:
    """Golden-section search on coordinate k; maximises QWK."""
    phi = (np.sqrt(5.0) - 1.0) / 2.0
    a, b = lo, hi
    c = b - phi * (b - a)
    d = a + phi * (b - a)
    def f(x: float) -> float:
        t_try = t.copy()
        t_try[k] = x
        return _eval_qwk(s, _ordered(t_try), y_int)
    fc, fd = f(c), f(d)
    while abs(b - a) > tol:
        if fc > fd:
            b, d, fd = d, c, fc
            c = b - phi * (b - a)
            fc = f(c)
        else:
            a, c, fc = c, d, fd
            d = a + phi * (b - a)
            fd = f(d)
    return 0.5 * (a + b)


def _ordered(t: np.ndarray) -> np.ndarray:
    return np.sort(t)


def optimise_all(
    scores: np.ndarray,
    y: np.ndarray,
    init: tuple[float, float, float] = (0.5, 1.5, 2.5),
    bounds_pad: float = 0.4,
) -> ThresholdResult:
    """Optimise per-item thresholds. scores / y are (N, 21)."""
    n_items = scores.shape[1]
    thr = np.zeros((n_items, 3), dtype=np.float32)
    qwks = np.zeros(n_items, dtype=np.float32)
    for j in range(n_items):
        t, q = optimise_per_item(scores[:, j], y[:, j], init=init, bounds_pad=bounds_pad)
        thr[j] = t
        qwks[j] = q
    return ThresholdResult(thr, qwks)


def apply_thresholds(scores: np.ndarray, thresholds: np.ndarray) -> np.ndarray:
    """Vectorised application of (21, 3) thresholds to (N, 21) scores."""
    N, I = scores.shape
    out = np.zeros_like(scores, dtype=np.int32)
    for j in range(I):
        t = np.sort(thresholds[j])
        out[scores[:, j] >= t[0], j] = 1
        out[scores[:, j] >= t[1], j] = 2
        out[scores[:, j] >= t[2], j] = 3
    return out
