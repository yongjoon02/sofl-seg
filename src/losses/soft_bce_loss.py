"""Soft Binary Cross Entropy Loss for segmentation."""
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.registry.losses import register_loss


@register_loss(
    name='soft_bce',
    description='Binary Cross Entropy loss for soft labels',
    supports_multiclass=False,
    supports_soft_labels=True,
)
class SoftBCELoss(nn.Module):
    """Binary Cross Entropy loss for soft labels.
    
    Converts 2-class logits to probability via softmax, then applies BCE.
    
    Args:
        soft_label: Whether to use soft label mode (default: False)
    """
    def __init__(self, soft_label: bool = False):
        super().__init__()
        self.soft_label = soft_label
        self.hard_ce = nn.CrossEntropyLoss()
    
    def forward(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits: (B, C, H, W) model output logits (C=2 for binary segmentation)
            labels: (B, H, W) soft labels in [0, 1]
        """
        if not self.soft_label:
            return self.hard_ce(logits, (labels > 0.5).long())
        
        # Get probability for foreground class (class 1)
        probs = F.softmax(logits, dim=1)[:, 1]  # (B, H, W)
        
        # BCE loss with soft labels
        eps = 1e-7
        probs = torch.clamp(probs, eps, 1 - eps)
        loss = -(labels * torch.log(probs) + (1 - labels) * torch.log(1 - probs))
        return loss.mean()
