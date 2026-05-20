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

    best_val_qwk = -math.inf
    best_state: dict | None = None
    patience_left = train_cfg.patience

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

        # Eval
        val_qwk, val_f1, _ = _evaluate(model, val_loader, device)
        improved = val_qwk > best_val_qwk + 1e-5
        if improved:
            best_val_qwk = val_qwk
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            patience_left = train_cfg.patience
        else:
            patience_left -= 1
        log.info(
            f"epoch {epoch:03d} lr={train_cfg.lr * lr_scale:.5f} val_qwk={val_qwk:.4f} val_f1={val_f1:.4f} "
            f"best_qwk={best_val_qwk:.4f} patience_left={patience_left}"
        )
        if patience_left <= 0:
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    val_preds = predict_with_tta(model, X_va, train_cfg, device)
    return model, val_preds


def _evaluate(model: IRTJointModel, loader: DataLoader, device: torch.device) -> tuple[float, float, dict[str, np.ndarray]]:
    model.eval()
    all_z: list[np.ndarray] = []
    all_a1: list[np.ndarray] = []
    all_y2: list[np.ndarray] = []
    all_y1: list[np.ndarray] = []
    with torch.no_grad():
        for xb, y2b, y1b in loader:
            out = model(xb.to(device))
            all_z.append(out["z"].cpu().numpy())
            all_a1.append(torch.sigmoid(out["a1_logits"]).cpu().numpy())
            all_y2.append(y2b.numpy())
            all_y1.append(y1b.numpy())
    z = np.concatenate(all_z, axis=0)
    p_a1 = np.concatenate(all_a1, axis=0)
    y_a2 = np.concatenate(all_y2, axis=0)
    y_a1 = np.concatenate(all_y1, axis=0)

    preds_int = argmax_from_cumlogits(z)
    valid_a2 = np.isfinite(y_a2).all(axis=1) & (y_a2 >= 0).all(axis=1)
    valid_a1 = np.isfinite(y_a1).all(axis=1) & (y_a1 >= 0).all(axis=1)
    qwk = mean_qwk(preds_int[valid_a2], y_a2[valid_a2].astype(int)) if valid_a2.any() else 0.0
    f1 = binary_f1(p_a1[valid_a1], y_a1[valid_a1].astype(int)) if valid_a1.any() else 0.0

    return qwk, f1, {"z": z, "p_a1": p_a1, "y_a2": y_a2, "y_a1": y_a1}


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
