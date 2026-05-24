"""Lightweight training loop for the IRT joint head.

Intentionally not the 1845-line `backup/common/runner.py`. Inputs are dense
participant-level numpy matrices; the model is small enough that batch-size
== whole train set works fine. Even so we batch (B=32) to get more SGD steps.
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Iterable

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from ..utils.metrics import binary_f1, mean_qwk
from .heads import argmax_from_cumlogits, sigmoid as np_sigmoid
from .irt_joint import IRTConfig, IRTJointModel, total_loss

log = logging.getLogger(__name__)


@dataclass
class TrainConfig:
    epochs: int = 80
    batch_size: int = 32
    lr: float = 1e-3
    weight_decay: float = 0.01
    warmup_epochs: int = 3
    patience: int = 12
    grad_clip: float = 1.0
    amp: bool = False
    seed: int = 42
    device: str = "cuda"
    tta_replicas: int = 8
    tta_noise_std: float = 0.01


def _make_loader(X: np.ndarray, y_a2: np.ndarray, y_a1: np.ndarray, batch_size: int, shuffle: bool) -> DataLoader:
    ds = TensorDataset(
        torch.from_numpy(np.ascontiguousarray(X, dtype=np.float32)),
        torch.from_numpy(np.ascontiguousarray(y_a2, dtype=np.float32)),
        torch.from_numpy(np.ascontiguousarray(y_a1, dtype=np.float32)),
    )
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, drop_last=False)


def _cosine_with_warmup(epoch: int, total: int, warmup: int) -> float:
    if epoch < warmup:
        return (epoch + 1) / max(1, warmup)
    progress = (epoch - warmup) / max(1, total - warmup)
    return 0.5 * (1.0 + math.cos(math.pi * progress))


def train_one_fold(
    model_cfg: IRTConfig,
    train_cfg: TrainConfig,
    X_tr: np.ndarray,
    y_a2_tr: np.ndarray,
    y_a1_tr: np.ndarray,
    X_va: np.ndarray,
    y_a2_va: np.ndarray,
    y_a1_va: np.ndarray,
) -> tuple[IRTJointModel, dict[str, np.ndarray]]:
    """Train one fold; return the best model (by val QWK) and val predictions."""
    torch.manual_seed(train_cfg.seed)
    np.random.seed(train_cfg.seed)
    device = torch.device(train_cfg.device if torch.cuda.is_available() and train_cfg.device == "cuda" else "cpu")

    model = IRTJointModel(model_cfg).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=train_cfg.lr, weight_decay=train_cfg.weight_decay)

    train_loader = _make_loader(X_tr, y_a2_tr, y_a1_tr, train_cfg.batch_size, shuffle=True)
    val_loader = _make_loader(X_va, y_a2_va, y_a1_va, train_cfg.batch_size * 4, shuffle=False)

    # Early-stop on val_loss (more stable than val_qwk on tiny val splits where
    # random init can lock in a noisy "best" at epoch 0). Track val_qwk for the
    # log; checkpoint when EITHER val_loss improves OR val_qwk improves beyond
    # a healthy margin (so we still capture the model that's actually best at
    # the metric we care about).
    best_val_loss = math.inf
    best_val_qwk = -math.inf
    best_state: dict | None = None
    patience_left = train_cfg.patience
    # Don't trust epoch-0 metrics — warmup means real gradients haven't kicked in.
    grace_epochs = max(train_cfg.warmup_epochs, 3)

    for epoch in range(train_cfg.epochs):
        lr_scale = _cosine_with_warmup(epoch, train_cfg.epochs, train_cfg.warmup_epochs)
        for pg in opt.param_groups:
            pg["lr"] = train_cfg.lr * lr_scale

        model.train()
        for xb, y2b, y1b in train_loader:
            xb = xb.to(device, non_blocking=True)
            y2b = y2b.to(device, non_blocking=True)
            y1b = y1b.to(device, non_blocking=True)
            out = model(xb)
            loss, _ = total_loss(out, y2b, y1b, model.discrim(), model_cfg)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), train_cfg.grad_clip)
            opt.step()

        # Eval (with val_loss for early-stop stability).
        val_qwk, val_f1, val_loss = _evaluate(model, val_loader, device, model_cfg, return_loss=True)
        loss_improved = val_loss < best_val_loss - 1e-4
        qwk_improved = val_qwk > best_val_qwk + 5e-3
        improved = (epoch >= grace_epochs) and (loss_improved or qwk_improved)
        if improved or best_state is None:
            best_val_loss = min(best_val_loss, val_loss)
            best_val_qwk = max(best_val_qwk, val_qwk)
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            patience_left = train_cfg.patience
        elif epoch >= grace_epochs:
            patience_left -= 1
        log.info(
            f"epoch {epoch:03d} lr={train_cfg.lr * lr_scale:.5f} val_loss={val_loss:.4f} "
            f"val_qwk={val_qwk:.4f} val_f1={val_f1:.4f} best_loss={best_val_loss:.4f} "
            f"best_qwk={best_val_qwk:.4f} patience={patience_left}"
        )
        if patience_left <= 0:
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    val_preds = predict_with_tta(model, X_va, train_cfg, device)
    return model, val_preds


def _evaluate(
    model: IRTJointModel,
    loader: DataLoader,
    device: torch.device,
    model_cfg=None,
    return_loss: bool = False,
):
    """Evaluate val metrics. Returns (qwk, f1, val_loss_or_preds_dict).

    Per-item QWK now masks NaN/-1 labels per position (matches training loss),
    so subjects with partial labels still contribute on items they answered.
    """
    from .irt_joint import total_loss as _total_loss  # local to avoid cycle
    model.eval()
    all_z: list[np.ndarray] = []
    all_a1: list[np.ndarray] = []
    all_y2: list[np.ndarray] = []
    all_y1: list[np.ndarray] = []
    loss_sum = 0.0
    n_batches = 0
    with torch.no_grad():
        for xb, y2b, y1b in loader:
            xb = xb.to(device)
            y2_dev = y2b.to(device)
            y1_dev = y1b.to(device)
            out = model(xb)
            if model_cfg is not None:
                loss, _ = _total_loss(out, y2_dev, y1_dev, model.discrim(), model_cfg)
                loss_sum += float(loss.item())
                n_batches += 1
            all_z.append(out["z"].cpu().numpy())
            all_a1.append(torch.sigmoid(out["a1_logits"]).cpu().numpy())
            all_y2.append(y2b.numpy())
            all_y1.append(y1b.numpy())
    z = np.concatenate(all_z, axis=0)
    p_a1 = np.concatenate(all_a1, axis=0)
    y_a2 = np.concatenate(all_y2, axis=0)
    y_a1 = np.concatenate(all_y1, axis=0)

    preds_int = argmax_from_cumlogits(z)
    qwk = _masked_mean_qwk(preds_int, y_a2)
    f1 = _masked_mean_f1(p_a1, y_a1)
    val_loss = (loss_sum / max(n_batches, 1)) if n_batches else float("nan")

    if return_loss:
        return qwk, f1, val_loss
    return qwk, f1, {"z": z, "p_a1": p_a1, "y_a2": y_a2, "y_a1": y_a1}


def _masked_mean_qwk(preds: np.ndarray, y: np.ndarray) -> float:
    """Per-item QWK, masking NaN / -1 entries; mean over items with ≥ 2 unique labels."""
    from ..utils.metrics import quadratic_weighted_kappa
    if preds.size == 0:
        return 0.0
    scores = []
    for j in range(preds.shape[1]):
        col_pred = preds[:, j]
        col_y = y[:, j]
        m = np.isfinite(col_y) & (col_y >= 0)
        if m.sum() < 2 or len(np.unique(col_y[m])) < 2:
            continue
        scores.append(quadratic_weighted_kappa(col_y[m].astype(int), col_pred[m].astype(int)))
    return float(np.mean(scores)) if scores else 0.0


def _masked_mean_f1(p: np.ndarray, y: np.ndarray, threshold: float = 0.5) -> float:
    from sklearn.metrics import f1_score
    if p.size == 0:
        return 0.0
    scores = []
    for c in range(p.shape[1]):
        m = np.isfinite(y[:, c]) & (y[:, c] >= 0)
        if m.sum() < 2:
            continue
        pred = (p[m, c] >= threshold).astype(int)
        scores.append(float(f1_score(y[m, c].astype(int), pred, zero_division=0.0)))
    return float(np.mean(scores)) if scores else 0.0


def predict_with_tta(
    model: IRTJointModel,
    X: np.ndarray,
    train_cfg: TrainConfig,
    device: torch.device,
) -> dict[str, np.ndarray]:
    """Average predictions over TTA replicas of feature noise."""
    model.eval()
    x_t = torch.from_numpy(np.ascontiguousarray(X, dtype=np.float32)).to(device)
    z_acc = None
    a1_acc = None
    replicas = max(1, train_cfg.tta_replicas)
    with torch.no_grad():
        for r in range(replicas):
            x_in = x_t
            if r > 0 and train_cfg.tta_noise_std > 0:
                x_in = x_t + torch.randn_like(x_t) * train_cfg.tta_noise_std
            out = model(x_in)
            if z_acc is None:
                z_acc = out["z"].clone()
                a1_acc = torch.sigmoid(out["a1_logits"]).clone()
            else:
                z_acc = z_acc + out["z"]
                a1_acc = a1_acc + torch.sigmoid(out["a1_logits"])
    z_acc = z_acc / replicas
    a1_acc = a1_acc / replicas
    return {
        "z": z_acc.cpu().numpy(),
        "p_a1": a1_acc.cpu().numpy(),
        "theta": model(x_t)["theta"].detach().cpu().numpy(),
    }
