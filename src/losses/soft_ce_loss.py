"""Soft Cross Entropy Loss for segmentation."""
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.registry.losses import register_loss


@register_loss(
    name='soft_ce',
    description='Cross entropy loss that supports soft labels',
    supports_multiclass=True,
    supports_soft_labels=True,
)
class SoftCrossEntropyLoss(nn.Module):
    """Cross entropy loss that supports soft labels.
    
    For soft labels, converts them to 2-class distribution and computes
    cross entropy with log_softmax.
    
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
            logits: (B, C, H, W) model output logits
            labels: (B, H, W) soft labels in [0, 1] or hard labels
        """
        if not self.soft_label:
            # Hard label mode
            return self.hard_ce(logits, (labels > 0.5).long())
        
        # Soft label mode: convert soft label to 2-class distribution
        soft_target = torch.stack([1 - labels, labels], dim=1)  # (B, 2, H, W)
        
        # Compute soft cross entropy
        log_probs = F.log_softmax(logits, dim=1)  # (B, C, H, W)
        loss = -torch.sum(soft_target * log_probs, dim=1)  # (B, H, W)
        return loss.mean()
