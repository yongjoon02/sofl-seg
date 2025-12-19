"""L2 Loss (Mean Squared Error) for segmentation."""
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.registry.losses import register_loss


@register_loss(
    name='l2',
    description='L2 loss (Mean Squared Error) for segmentation',
    supports_multiclass=True,
    supports_soft_labels=True,
)
class L2Loss(nn.Module):
    """L2 Loss (Mean Squared Error) for segmentation.
    
    Supports both hard and soft labels.
    
    Args:
        soft_label: Whether to use soft label mode (default: False)
    """
    def __init__(self, soft_label: bool = False):
        super().__init__()
        self.soft_label = soft_label
    
    def forward(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits: (B, C, H, W) model output logits
            labels: (B, H, W) soft labels in [0, 1] or hard labels
        """
        # Convert logits to probabilities
        if logits.dim() == 4:
            # Binary segmentation: take foreground class (index 1)
            probs = F.softmax(logits, dim=1)[:, 1, :, :]  # (B, H, W)
        else:
            probs = logits
        
        # Convert labels to [0, 1] range if needed
        if not self.soft_label:
            # Hard label mode: convert to binary
            labels_float = (labels > 0.5).float()
        else:
            # Soft label mode: use as is (should be in [0, 1])
            labels_float = labels.float()
            if labels_float.max() > 1.0:
                labels_float = labels_float / 255.0
        
        # L2 loss: (pred - target)^2
        loss = (probs - labels_float) ** 2
        return loss.mean()
