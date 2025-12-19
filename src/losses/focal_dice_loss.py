"""Focal Dice Loss for segmentation."""
import torch
import torch.nn as nn
import torch.nn.functional as F

from .dice_loss import DiceLoss

from src.registry.losses import register_loss


@register_loss(
    name='focal_dice',
    description='Combined Focal + Dice loss',
    supports_multiclass=True,
    supports_soft_labels=True,
)
class FocalDiceLoss(nn.Module):
    """Combined Focal + Dice loss for segmentation."""
    def __init__(self, wf: float = 0.1, wd: float = 0.9, gamma: float = 2.0):
        super().__init__()
        self.dice = DiceLoss()
        self.wf = wf  # Focal weight
        self.wd = wd  # Dice weight
        self.gamma = gamma  # Focal gamma

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred: (B, C, H, W) predictions (probabilities)
            target: (B, H, W) targets
        """
        # Focal loss
        if pred.dim() == 4:
            pred_prob = F.softmax(pred, dim=1)[:, 1]
        else:
            pred_prob = pred
        
        eps = 1e-7
        pred_prob = torch.clamp(pred_prob, eps, 1 - eps)
        
        focal = -target * ((1 - pred_prob) ** self.gamma) * torch.log(pred_prob)
        focal = focal.mean()
        
        # Dice loss
        dice = self.dice(pred, target)
        
        return self.wf * focal + self.wd * dice
