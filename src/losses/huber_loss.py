"""Huber (Smooth L1) Loss for segmentation."""
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.registry.losses import register_loss


@register_loss(
    name='huber',
    description='Huber (Smooth L1) loss for segmentation',
    supports_multiclass=True,
    supports_soft_labels=True,
)
class HuberLoss(nn.Module):
    """Huber (Smooth L1) loss.
    
    Supports hard/soft labels. For logits, foreground prob을 사용합니다.
    
    Args:
        delta: Huber transition point (beta in torch smooth_l1_loss), default 1.0
        soft_label: If True, labels는 연속값으로 그대로 사용
    """
    def __init__(self, delta: float = 1.0, soft_label: bool = False):
        super().__init__()
        self.delta = delta
        self.soft_label = soft_label
    
    def forward(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        # Convert logits to probabilities (binary: class 1)
        if logits.dim() == 4:
            probs = F.softmax(logits, dim=1)[:, 1, :, :]
        else:
            probs = logits
        
        # Prepare targets
        if self.soft_label:
            targets = labels.float()
            if targets.max() > 1.0:
                targets = targets / 255.0
        else:
            targets = (labels > 0.5).float()
        
        return F.smooth_l1_loss(probs, targets, beta=self.delta, reduction='mean')
