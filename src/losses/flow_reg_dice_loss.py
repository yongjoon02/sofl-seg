"""Flow matching regression loss with Dice regularizer (binary, 1-channel).

목적:
- v vs. ut 에 대해 L1/L2 회귀를 기본 손실로 사용하고,
- 현재 geometry 예측(xt)을 하드 스레시홀드한 마스크와 GT 하드 마스크 사이의 Dice를
  정규항으로 더해 foreground 구조를 보강한다.
"""

from __future__ import annotations

from typing import Literal

import torch
import torch.nn as nn

from src.registry.losses import register_loss

DicePredMode = Literal["soft", "hard", "hard_ste"]


@register_loss(
    name="flow_reg_dice",
    description="Flow matching: base L1/L2 + Dice regularizer on binary masks",
    supports_multiclass=False,
    supports_soft_labels=True,
)
class FlowRegDiceLoss(nn.Module):
    """
    Args:
        base: 'l1' or 'l2' for flow regression term (v vs ut).
        dice_lambda: Weight λ for the Dice regularizer.
        threshold: Threshold τ for binarizing geometry (xt, geometry) when computing Dice.
        dice_pred_mode:
            - 'soft': Dice on probabilities (xt ∈ [0,1])
            - 'hard': Dice on hard-thresholded masks (no grad through threshold)
            - 'hard_ste': Forward uses hard mask, backward uses soft (straight-through)
        eps: Small constant to avoid div-by-zero.
    """

    def __init__(
        self,
        base: str = "l1",
        dice_lambda: float = 0.1,
        threshold: float = 0.5,
        dice_pred_mode: DicePredMode = "hard_ste",
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        if base not in ("l1", "l2"):
            raise ValueError(f"base must be 'l1' or 'l2', got {base}")
        self.base = base
        self.dice_lambda = float(dice_lambda)
        self.threshold = float(threshold)
        self.dice_pred_mode: DicePredMode = dice_pred_mode
        self.eps = float(eps)

    def forward(
        self,
        v: torch.Tensor,
        ut: torch.Tensor,
        xt: torch.Tensor,
        geometry: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        # Base flow regression term
        if self.base == "l1":
            base_loss = (v - ut).abs().mean()
            base_name = "l1"
        else:
            base_loss = (v - ut).pow(2).mean()
            base_name = "l2"

        if self.dice_lambda <= 0:
            return base_loss, {base_name: base_loss}

        # Prepare probabilities (xt and geometry are 1-channel in [0,1])
        pred_probs = xt
        if pred_probs.dim() == 4 and pred_probs.shape[1] == 1:
            pred_probs = pred_probs.squeeze(1)
        pred_probs = pred_probs.clamp(0.0, 1.0)

        target_probs = geometry
        if target_probs.dim() == 4 and target_probs.shape[1] == 1:
            target_probs = target_probs.squeeze(1)
        if target_probs.max() > 1.0:
            target_probs = target_probs / 255.0
        target_hard = (target_probs > self.threshold).float()

        # Choose prediction used for Dice
        if self.dice_pred_mode == "soft":
            pred_for_dice = pred_probs
        else:
            pred_hard = (pred_probs > self.threshold).float()
            if self.dice_pred_mode == "hard_ste":
                pred_for_dice = pred_hard.detach() - pred_probs.detach() + pred_probs
            else:
                pred_for_dice = pred_hard

        # Binary Dice loss (1 - Dice)
        intersection = (pred_for_dice * target_hard).sum(dim=(1, 2))
        pred_sum = pred_for_dice.sum(dim=(1, 2))
        target_sum = target_hard.sum(dim=(1, 2))
        dice = (2 * intersection + self.eps) / (pred_sum + target_sum + self.eps)
        dice_loss = 1.0 - dice.mean()

        total = base_loss + self.dice_lambda * dice_loss
        return total, {base_name: base_loss, "dice_reg": dice_loss}
