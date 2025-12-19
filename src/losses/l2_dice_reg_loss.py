"""Composite loss: L2 to soft labels + Dice regularizer to hard labels.

Intended use-case:
- Train a model to regress a soft label map (e.g., SAUNA) with L2.
- Add a Dice-based regularizer against the hard ground-truth mask to
  encourage foreground structure.
"""

from __future__ import annotations

from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.registry.losses import register_loss


DicePredMode = Literal["soft", "hard", "hard_ste"]


@register_loss(
    name="l2_dice_reg",
    description="L2 regression to soft labels + lambda * Dice regularizer to hard GT",
    supports_multiclass=False,
    supports_soft_labels=True,
)
class L2DiceRegLoss(nn.Module):
    """L2 regression loss with Dice regularizer.

    Args:
        soft_label: If True, labels are interpreted as soft targets in [0, 1].
        dice_lambda: Weight for the Dice regularizer term.
        threshold: Threshold tau used when dice_pred_mode is 'hard' or 'hard_ste'.
        dice_pred_mode:
            - 'soft': Dice is computed on probabilities (fully differentiable).
            - 'hard': Dice is computed on hard-thresholded predictions (no gradient through threshold).
            - 'hard_ste': Forward uses hard threshold, backward uses soft probs (straight-through).
        eps: Small constant to avoid division-by-zero.
    """

    def __init__(
        self,
        soft_label: bool = True,
        dice_lambda: float = 0.1,
        threshold: float = 0.5,
        dice_pred_mode: DicePredMode = "hard_ste",
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        self.soft_label = soft_label
        self.dice_lambda = float(dice_lambda)
        self.threshold = float(threshold)
        self.dice_pred_mode: DicePredMode = dice_pred_mode
        self.eps = float(eps)

    def forward(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        *,
        hard_labels: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        # Convert logits -> foreground probabilities (B, H, W)
        if logits.dim() == 4:
            probs = F.softmax(logits, dim=1)[:, 1, :, :]
        else:
            probs = logits

        # Soft target (SAUNA, smoothing, etc.)
        soft_target = labels.float()
        if soft_target.max() > 1.0:
            soft_target = soft_target / 255.0

        # L2 term
        l2 = (probs - soft_target).pow(2).mean()

        if self.dice_lambda <= 0:
            return l2, {"l2": l2}

        # Hard GT term for Dice regularization
        if hard_labels is None:
            hard_gt = (soft_target > 0.5).float()
        else:
            hard_gt = hard_labels.float()
            if hard_gt.max() > 1.0:
                hard_gt = hard_gt / 255.0
            hard_gt = (hard_gt > 0.5).float()

        pred_for_dice: torch.Tensor
        if self.dice_pred_mode == "soft":
            pred_for_dice = probs
        else:
            pred_hard = (probs > self.threshold).float()
            if self.dice_pred_mode == "hard_ste":
                pred_for_dice = pred_hard.detach() - probs.detach() + probs
            else:  # "hard"
                pred_for_dice = pred_hard

        # Dice loss on foreground (binary): 1 - Dice
        pred_sum = pred_for_dice.sum(dim=(1, 2))
        gt_sum = hard_gt.sum(dim=(1, 2))
        intersection = (pred_for_dice * hard_gt).sum(dim=(1, 2))
        dice = (2.0 * intersection + self.eps) / (pred_sum + gt_sum + self.eps)
        dice_loss = 1.0 - dice.mean()

        total = l2 + self.dice_lambda * dice_loss
        return total, {"l2": l2, "dice_reg": dice_loss}
