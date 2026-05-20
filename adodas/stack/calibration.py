"""A1 calibration. Pick exactly ONE of isotonic or logit_shift.

isotonic: per-target IsotonicRegression on OOF probs → submission emits the
          calibrated probability. Official scorer thresholds at 0.5, so we
          want OOF F1@0.5 to match the optimal threshold's F1.

logit_shift: find OOF threshold t* that maximises F1; emit
          p' = sigmoid(logit(p) - logit(t*))  so submission@0.5 ≈ original@t*.
          Cheaper, doesn't risk over-calibrating on small validation sets.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.isotonic import IsotonicRegression

from ..models.heads import logit as np_logit, sigmoid as np_sigmoid
from ..utils.metrics import binary_f1


@dataclass
class CalibrationResult:
    method: str
    # isotonic: list of fitted IsotonicRegression per target
    # logit_shift: array of shifts (one per target)
    payload: list[IsotonicRegression] | np.ndarray
    oof_calibrated: np.ndarray


def fit_isotonic(
    oof: np.ndarray,
    y: np.ndarray,
    clip: tuple[float, float] = (0.01, 0.99),
) -> CalibrationResult:
    """Fit a per-target isotonic regression. y is binary (N, C)."""
    N, C = oof.shape
    fitted: list[IsotonicRegression] = []
    cal = np.empty_like(oof)
    for c in range(C):
        p = oof[:, c]
        yc = y[:, c]
        mask = np.isfinite(p) & np.isfinite(yc) & (yc >= 0)
        ir = IsotonicRegression(out_of_bounds="clip", y_min=clip[0], y_max=clip[1])
        if mask.sum() < 5:
            cal[:, c] = np.clip(p, clip[0], clip[1])
            fitted.append(ir.fit([0.0, 1.0], [0.0, 1.0]))  # identity fallback
        else:
            ir.fit(p[mask], yc[mask])
            cal[:, c] = ir.transform(np.clip(p, 0.0, 1.0))
            fitted.append(ir)
    return CalibrationResult("isotonic", fitted, cal)


def apply_isotonic(probs: np.ndarray, result: CalibrationResult) -> np.ndarray:
    assert result.method == "isotonic"
    out = np.empty_like(probs)
    for c, ir in enumerate(result.payload):  # type: ignore[arg-type]
        out[:, c] = ir.transform(np.clip(probs[:, c], 0.0, 1.0))
    return out


def fit_logit_shift(oof: np.ndarray, y: np.ndarray, n_grid: int = 101) -> CalibrationResult:
    """Find per-target threshold t* (max F1 on OOF), then emit shifts so the
    submission's 0.5 threshold becomes t* in original probability space."""
    N, C = oof.shape
    shifts = np.zeros(C, dtype=np.float32)
    cal = np.empty_like(oof)
    grid = np.linspace(0.05, 0.95, n_grid)
    for c in range(C):
        p = oof[:, c]
        yc = y[:, c]
        mask = np.isfinite(p) & np.isfinite(yc) & (yc >= 0)
        if mask.sum() < 5:
            cal[:, c] = p
            continue
        best_t, best_f1 = 0.5, -1.0
        for t in grid:
            f1 = _binary_f1_single((p[mask] >= t).astype(int), yc[mask].astype(int))
            if f1 > best_f1:
                best_f1 = f1
                best_t = t
        shifts[c] = np_logit(np.asarray([best_t]))[0]
        cal[:, c] = np_sigmoid(np_logit(np.clip(p, 0.01, 0.99)) - shifts[c])
    return CalibrationResult("logit_shift", shifts, cal)


def apply_logit_shift(probs: np.ndarray, result: CalibrationResult) -> np.ndarray:
    assert result.method == "logit_shift"
    shifts = result.payload  # (C,)
    out = np_sigmoid(np_logit(np.clip(probs, 0.01, 0.99)) - shifts)  # broadcast over rows
    return out


def _binary_f1_single(pred: np.ndarray, y: np.ndarray) -> float:
    from sklearn.metrics import f1_score
    return float(f1_score(y, pred, zero_division=0.0))


def report_calibration(oof_before: np.ndarray, oof_after: np.ndarray, y: np.ndarray) -> dict[str, float]:
    """Two F1@0.5 numbers: before and after. They should differ — the after
    should match the F1@best_t on the raw (uncalibrated) OOF."""
    return {
        "f1@0.5_before": binary_f1(oof_before, y),
        "f1@0.5_after": binary_f1(oof_after, y),
    }
