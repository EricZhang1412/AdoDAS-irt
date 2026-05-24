"""IRT/factor-structured joint head for A1 + A2.

Architecture (all 21 items share 3 latent dimensions θ_DAS):

    h = Encoder(x)                ∈ R^d_hidden
    θ = LayerNorm(W_θ h)          ∈ R^3                 # D, A, S
    a_j = 1 + 0.5 softplus(α_j)   ∈ R                    # discrimination ≥ 1
    s_j = a_j θ_{g(j)} - b_j      ∈ R
    c_{j,k} = c_{j,k-1} + softplus(δ_{j,k})              # monotone thresholds
    z_{j,k} = s_j - c_{j,k}                              # CORAL cumulative logits
    p_a1 = α_t θ_t + β_t  →  sigmoid                     # A1 binary

Losses (see config irt.yaml):
    L_a2     = BCE(z, ordinal_targets)
    L_a1     = BCE_with_logits(p_a1, y_a1)
    L_consist= BCE_with_logits(p_a1, soft_dass21(z).detach())
    L_qwk    = soft_QWK regularizer
    L_anchor = max(0, -a_j_anchor) for each subscale anchor item
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..utils.dass21 import (
    DASS21_GROUP_CUTOFFS,
    DASS21_GROUP_ITEMS,
    DASS21_GROUP_ORDER,
    item_to_group_vector,
)


@dataclass
class IRTConfig:
    d_in: int
    d_hidden: int = 256
    encoder_layers: int = 2
    encoder_dropout: float = 0.3
    feature_noise_std: float = 0.02
    d_theta: int = 3
    bifactor: bool = False
    label_smoothing: float = 0.05
    consistency_temperature: float = 1.0
    lambda_a1: float = 1.0
    lambda_consist: float = 0.3
    lambda_qwk: float = 0.2
    lambda_anchor: float = 1.0
    anchor_items: dict[str, int] | None = None  # {'D': 2, 'A': 1, 'S': 0}
    n_items: int = 21
    n_thresholds: int = 3


class Encoder(nn.Module):
    """Plain MLP with LayerNorm; optional Gaussian noise during training."""

    def __init__(self, d_in: int, d_hidden: int, n_layers: int, dropout: float, noise_std: float):
        super().__init__()
        self.noise_std = noise_std
        layers: list[nn.Module] = []
        prev = d_in
        for _ in range(n_layers):
            layers.extend([
                nn.LayerNorm(prev),
                nn.Linear(prev, d_hidden),
                nn.GELU(),
                nn.Dropout(dropout),
            ])
            prev = d_hidden
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.training and self.noise_std > 0:
            x = x + torch.randn_like(x) * self.noise_std
        return self.net(x)


class IRTHead(nn.Module):
    """Factor-structured CORAL ordinal head shared across 21 DASS-21 items."""

    def __init__(self, d_in: int, cfg: IRTConfig):
        super().__init__()
        self.cfg = cfg
        self.theta_proj = nn.Linear(d_in, cfg.d_theta)
        self.theta_norm = nn.LayerNorm(cfg.d_theta)

        self.alpha_raw = nn.Parameter(torch.zeros(cfg.n_items))         # discrim
        self.b = nn.Parameter(torch.zeros(cfg.n_items))                  # difficulty
        # First threshold is anchored to 0; remaining k-1 are positive gaps.
        self.delta_raw = nn.Parameter(torch.zeros(cfg.n_items, cfg.n_thresholds - 1))

        # A1 head: per-subscale linear over θ.
        self.a1_alpha = nn.Parameter(torch.ones(3))
        self.a1_beta = nn.Parameter(torch.zeros(3))

        # Map item index → subscale θ axis index.
        item_to_group = item_to_group_vector()  # list[int] length 21
        self.register_buffer(
            "item_group",
            torch.tensor(item_to_group, dtype=torch.long),
            persistent=False,
        )

    def thresholds(self) -> torch.Tensor:
        gaps = F.softplus(self.delta_raw)                       # (21, K-1)
        first = torch.zeros(self.cfg.n_items, 1, device=gaps.device)
        return torch.cat([first, first + torch.cumsum(gaps, dim=-1)], dim=-1)  # (21, K)

    def discrim(self) -> torch.Tensor:
        return 1.0 + 0.5 * F.softplus(self.alpha_raw)            # (21,)

    def forward(self, h: torch.Tensor) -> dict[str, torch.Tensor]:
        theta = self.theta_norm(self.theta_proj(h))              # (B, 3)
        a = self.discrim()                                       # (21,)
        item_theta = theta[:, self.item_group]                   # (B, 21)
        score = a.unsqueeze(0) * item_theta - self.b.unsqueeze(0)  # (B, 21)
        c = self.thresholds()                                    # (21, K)
        z = score.unsqueeze(-1) - c.unsqueeze(0)                 # (B, 21, K)

        a1_logits = self.a1_alpha.unsqueeze(0) * theta + self.a1_beta.unsqueeze(0)  # (B, 3)

        return {"theta": theta, "score": score, "thresholds": c, "z": z, "a1_logits": a1_logits}


class IRTJointModel(nn.Module):
    def __init__(self, cfg: IRTConfig):
        super().__init__()
        self.cfg = cfg
        self.encoder = Encoder(cfg.d_in, cfg.d_hidden, cfg.encoder_layers, cfg.encoder_dropout, cfg.feature_noise_std)
        self.head = IRTHead(cfg.d_hidden, cfg)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        h = self.encoder(x)
        return self.head(h)

    def discrim(self) -> torch.Tensor:
        return self.head.discrim()


# ---------------------------------------------------------------------------
# Losses.


def _valid_mask(y: torch.Tensor) -> torch.Tensor:
    """True where label is finite AND non-negative. Used for per-position masking
    so partial-label participants still contribute on items they DID answer."""
    return torch.isfinite(y) & (y >= 0)


def build_ordinal_targets(labels: torch.Tensor, n_thresholds: int = 3) -> torch.Tensor:
    """labels: (B, n_items) in {0..K}; returns (B, n_items, K) binary."""
    thresh = torch.arange(1, n_thresholds + 1, device=labels.device).float()
    return (labels.unsqueeze(-1).float() >= thresh.view(1, 1, -1)).float()


def coral_bce(z: torch.Tensor, y_a2: torch.Tensor, label_smoothing: float = 0.0) -> torch.Tensor:
    """CORAL BCE with per-position masking — NaN/-1 labels skip loss for that
    (subject, item) pair instead of dropping the whole subject row."""
    valid = _valid_mask(y_a2)                                # (B, 21)
    y_safe = torch.where(valid, y_a2, torch.zeros_like(y_a2))
    targets = build_ordinal_targets(y_safe, n_thresholds=z.size(-1))
    if label_smoothing > 0.0:
        targets = targets * (1.0 - label_smoothing) + 0.5 * label_smoothing
    loss_pp = F.binary_cross_entropy_with_logits(z, targets, reduction="none")  # (B, 21, K)
    mask3 = valid.unsqueeze(-1).expand_as(loss_pp).float()
    return (loss_pp * mask3).sum() / mask3.sum().clamp_min(1.0)


def a1_bce(logits: torch.Tensor, y_a1: torch.Tensor) -> torch.Tensor:
    """A1 BCE with per-position masking."""
    valid = _valid_mask(y_a1)                                # (B, 3)
    y_safe = torch.where(valid, y_a1, torch.zeros_like(y_a1))
    loss_pp = F.binary_cross_entropy_with_logits(logits, y_safe.float(), reduction="none")
    mask = valid.float()
    return (loss_pp * mask).sum() / mask.sum().clamp_min(1.0)


def monotonic_class_probs(z: torch.Tensor) -> torch.Tensor:
    """Differentiable simplex over 4 classes from cumulative logits."""
    s = torch.sigmoid(z)
    p1 = s[..., 0]
    p2 = torch.min(s[..., 1], p1)
    p3 = torch.min(s[..., 2], p2)
    P0 = 1.0 - p1
    P1 = p1 - p2
    P2 = p2 - p3
    P3 = p3
    probs = torch.stack([P0, P1, P2, P3], dim=-1).clamp_min(0.0)
    return probs / probs.sum(dim=-1, keepdim=True).clamp_min(1e-8)


def soft_qwk_loss(class_probs: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Differentiable soft QWK as 1 - kappa; bounded by ordinal cost matrix.

    Per-item masking: rows with NaN/-1 label for an item are removed from
    that item's contingency table. If an item has no valid samples in the
    batch it contributes 0 to the loss.
    """
    b, n_items, k = class_probs.shape
    valid = _valid_mask(labels)                              # (B, n_items)
    safe_labels = torch.where(valid, labels, torch.zeros_like(labels)).long().clamp(0, k - 1)
    true_oh = F.one_hot(safe_labels, k).to(class_probs.dtype)  # (B, n_items, K)
    # Zero out invalid rows so they don't contribute to histograms.
    mask = valid.unsqueeze(-1).float()                       # (B, n_items, 1)
    true_oh = true_oh * mask
    probs_masked = class_probs * mask

    idx = torch.arange(k, device=class_probs.device, dtype=class_probs.dtype)
    w = (idx.unsqueeze(0) - idx.unsqueeze(1)).pow(2) / ((k - 1) ** 2)

    n_valid_per_item = mask.squeeze(-1).sum(dim=0).clamp_min(1.0)  # (n_items,)
    o = torch.einsum("bit,bij->itj", true_oh, probs_masked) / n_valid_per_item.view(-1, 1, 1)
    hist_true = true_oh.sum(dim=0) / n_valid_per_item.view(-1, 1)
    hist_pred = probs_masked.sum(dim=0) / n_valid_per_item.view(-1, 1)
    e_mat = hist_true.unsqueeze(-1) * hist_pred.unsqueeze(-2)
    num = (w.unsqueeze(0) * o).sum(dim=(1, 2))
    den = (w.unsqueeze(0) * e_mat).sum(dim=(1, 2)).clamp_min(1e-7)
    kappa = 1.0 - num / den
    return (1.0 - kappa).mean()


def soft_dass21_indicators(z: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
    """(B, 21, K) cumulative logits → (B, 3) soft DASS-21 binary indicators.

    Mirror of backup/common/models/heads.py:19-39.
    """
    e_y = torch.sigmoid(z).sum(dim=-1)
    inv_tau = 1.0 / max(float(temperature), 1e-6)
    out = []
    for g in DASS21_GROUP_ORDER:
        idx = torch.as_tensor(DASS21_GROUP_ITEMS[g], device=e_y.device, dtype=torch.long)
        s_g = e_y.index_select(-1, idx).sum(dim=-1)
        out.append(torch.sigmoid((2.0 * s_g - DASS21_GROUP_CUTOFFS[g]) * inv_tau))
    return torch.stack(out, dim=-1)


def consistency_loss(a1_logits: torch.Tensor, z: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
    """Detach soft DASS-21 targets so A2 head isn't pulled by weak A1 BCE."""
    with torch.no_grad():
        soft_targets = soft_dass21_indicators(z, temperature=temperature)
    return F.binary_cross_entropy_with_logits(a1_logits, soft_targets)


def anchor_loss(discrim: torch.Tensor, anchor_items: dict[str, int]) -> torch.Tensor:
    """Penalize negative discrimination on anchor items to break sign indeterminacy.

    Anchor items must load *positively* on their subscale; if their a_j drifts
    negative, θ has flipped sign — penalize until it doesn't.
    """
    if not anchor_items:
        return discrim.new_zeros(())
    pieces = []
    for g, j in anchor_items.items():
        pieces.append(F.relu(-discrim[j]))
    return torch.stack(pieces).sum()


def total_loss(
    out: dict[str, torch.Tensor],
    y_a2: torch.Tensor,
    y_a1: torch.Tensor,
    discrim: torch.Tensor,
    cfg: IRTConfig,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Combined loss. Returns (loss_tensor, scalar_dict_for_logging)."""
    l_a2 = coral_bce(out["z"], y_a2, label_smoothing=cfg.label_smoothing)
    l_a1 = a1_bce(out["a1_logits"], y_a1)

    probs = monotonic_class_probs(out["z"])
    l_qwk = soft_qwk_loss(probs, y_a2)

    l_consist = consistency_loss(out["a1_logits"], out["z"], temperature=cfg.consistency_temperature)
    l_anchor = anchor_loss(discrim, cfg.anchor_items or {})

    total = (
        l_a2
        + cfg.lambda_a1 * l_a1
        + cfg.lambda_consist * l_consist
        + cfg.lambda_qwk * l_qwk
        + cfg.lambda_anchor * l_anchor
    )

    scalars = {
        "loss/total": float(total.detach().item()),
        "loss/a2": float(l_a2.detach().item()),
        "loss/a1": float(l_a1.detach().item()),
        "loss/qwk": float(l_qwk.detach().item()),
        "loss/consist": float(l_consist.detach().item()),
        "loss/anchor": float(l_anchor.detach().item()),
    }
    return total, scalars
