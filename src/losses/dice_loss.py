"""Dice Loss for segmentation."""
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.registry.losses import register_loss


@register_loss(
    name='dice',
    description='Dice loss for segmentation',
    supports_multiclass=True,
    supports_soft_labels=True,
)
class DiceLoss(nn.Module):
    """Dice loss for segmentation."""
    def __init__(self, smooth: float = 1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred: (B, C, H, W) or (B, H, W) predictions (probabilities)
            target: (B, H, W) targets
        """
        if pred.dim() == 4:
            pred = F.softmax(pred, dim=1)[:, 1]  # Take foreground probability
        
        pred_flat = pred.view(pred.size(0), -1)
        target_flat = target.view(target.size(0), -1)
        
        intersection = (pred_flat * target_flat).sum(dim=1)
        union = pred_flat.sum(dim=1) + target_flat.sum(dim=1)
        
        dice = (2.0 * intersection + self.smooth) / (union + self.smooth)
        return 1.0 - dice.mean()
