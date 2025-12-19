"""
Topology-aware loss using topolosses library.

Notes:
- Uses BettiMatchingLoss from topolosses for stable topological loss computation
- Much more stable than gudhi-based implementation
- Requires `topolosses` (pip install topolosses)
"""
import torch
import torch.nn as nn
import torch.distributed as dist
import warnings

try:
    from topolosses.losses import BettiMatchingLoss
except ImportError:
    BettiMatchingLoss = None
    warnings.warn("topolosses is not installed. TopoLoss will not function. Install via `pip install topolosses`.")

from src.registry.losses import register_loss


# Old gudhi-based implementation removed - using topolosses library instead


@register_loss(
    name='topo',
    description='Topology-aware loss using Betti matching',
    supports_multiclass=False,
    supports_soft_labels=False,  # TopoLoss typically works with binary masks
)
class TopoLoss(nn.Module):
    """
    Topological loss using BettiMatchingLoss from topolosses library.
    
    Much more stable than gudhi-based implementation.
    Works with both single GPU and DDP training.
    
    Args:
        lambda_weight: Weight for topological loss (default: 0.1)
        maxdim: Maximum Betti number dimension to consider (0=components, 1=loops)
        
    Notes:
    - Use small lambda to start (e.g., 0.01-0.1)
    - Training step only (exclude from validation for speed)
    - Requires `topolosses` library: pip install topolosses
    """

    def __init__(
        self,
        lambda_weight: float = 0.1,
        maxdim: int = 1,
        topo_size: int = 128,  # Downsample to this size for speed
        pers_thresh: float = 0.0,  # Kept for compatibility but not used
        pers_thresh_perfect: float = 0.99,  # Kept for compatibility but not used
    ):
        super().__init__()
        self.lambda_weight = lambda_weight
        self.maxdim = maxdim
        self.topo_size = topo_size  # Downsample to this size for faster computation
        
        if BettiMatchingLoss is None:
            raise ImportError(
                "topolosses is required for TopoLoss. "
                "Install via: pip install topolosses"
            )
        
        # Initialize BettiMatchingLoss
        # topology_weights: (betti0_weight, betti1_weight)
        # maxdim=0: only betti0, maxdim=1: betti0 + betti1
        if maxdim == 0:
            topology_weights = (1.0, 0.0)  # Only components
        else:
            topology_weights = (1.0, 1.0)  # Components + loops
        
        self.betti_loss = BettiMatchingLoss(
            topology_weights=topology_weights,
            alpha=1.0,              # topo term weight (여기선 그냥 1로 두고 바깥에서 lambda_weight로 스케일)
            softmax=False,          # 우리가 이미 확률로 만든 후 넣을 거면 False
            use_base_loss=False,  # Dice 같은 베이스 로스는 안 쓰고 topo만
            base_loss=None,
            num_processes=1,  # Single GPU
        )

    def forward(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Compute topological loss using BettiMatchingLoss.
        
        Args:
            logits: (B, C, H, W) raw logits (before softmax/sigmoid)
            labels: (B, H, W) binary/soft labels in [0, 1]
        
        Returns:
            topo_loss: scalar tensor
        """
        # Early return if logits contain NaN/Inf
        if torch.isnan(logits).any() or torch.isinf(logits).any():
            warnings.warn("Logits contain NaN/Inf. Skipping TopoLoss.")
            return torch.tensor(0.0, device=logits.device, dtype=logits.dtype)
        
        # Convert logits to probabilities
        # For binary segmentation, take foreground class (index 1)
        pred_probs = torch.softmax(logits, dim=1)[:, 1, :, :]  # (B, H, W)
        # Topology loss requires binary mask (threshold soft labels)
        # This is safe because topology is computed on binary structures
        labels_float = (labels > 0.5).float()
        
        # Clamp to [0, 1] to ensure valid probability range
        pred_probs = torch.clamp(pred_probs, min=0.0, max=1.0)
        labels_float = torch.clamp(labels_float, min=0.0, max=1.0)
        
        # Check for NaN/Inf after processing
        if torch.isnan(pred_probs).any() or torch.isinf(pred_probs).any():
            warnings.warn("Predicted probabilities contain NaN/Inf. Skipping TopoLoss.")
            return torch.tensor(0.0, device=logits.device, dtype=logits.dtype)
        
        # Downsample for faster computation (if larger than topo_size)
        B, H, W = pred_probs.shape
        if H > self.topo_size or W > self.topo_size:
            # Use adaptive average pooling to downsample
            pred_probs = pred_probs.unsqueeze(1)  # (B, 1, H, W)
            labels_float = labels_float.unsqueeze(1)  # (B, 1, H, W)
            
            # Downsample to topo_size x topo_size
            pred_probs = torch.nn.functional.adaptive_avg_pool2d(
                pred_probs, (self.topo_size, self.topo_size)
            ).squeeze(1)  # (B, topo_size, topo_size)
            labels_float = torch.nn.functional.adaptive_avg_pool2d(
                labels_float, (self.topo_size, self.topo_size)
            ).squeeze(1)  # (B, topo_size, topo_size)
            
            # Re-clamp after downsampling
            pred_probs = torch.clamp(pred_probs, min=0.0, max=1.0)
            labels_float = torch.clamp(labels_float, min=0.0, max=1.0)
        
        # Use all samples in batch for more stable gradients
        # If batch is too large, use a subset (e.g., first 4 samples) for speed
        max_samples = 4  # Process up to 4 samples per batch
        if B > max_samples:
            # Use first max_samples for TopoLoss to balance speed and stability
            pred_probs = pred_probs[:max_samples]  # (max_samples, H, W)
            labels_float = labels_float[:max_samples]  # (max_samples, H, W)
        
        # BettiMatchingLoss expects (B, H, W) probability maps
        # pred: predicted probability map in [0, 1]
        # target: ground truth binary/soft mask in [0, 1]
        try:
            topo_loss = self.betti_loss(pred_probs, labels_float)
            
            # Check for NaN or Inf
            if torch.isnan(topo_loss) or torch.isinf(topo_loss):
                warnings.warn(f"TopoLoss returned NaN/Inf: {topo_loss}. Returning zero loss.")
                return torch.tensor(0.0, device=logits.device, dtype=logits.dtype, requires_grad=True)
            
            # Clamp to reasonable range to avoid instability
            # Use smaller max value to prevent gradient explosion
            topo_loss = torch.clamp(topo_loss, min=0.0, max=5.0)
            
            # Apply lambda weight
            topo_loss_weighted = self.lambda_weight * topo_loss
            
            return topo_loss_weighted
        except Exception as e:
            # Fallback to zero if computation fails
            warnings.warn(f"TopoLoss computation failed: {e}. Returning zero loss.")
            return torch.tensor(0.0, device=logits.device, dtype=logits.dtype, requires_grad=True)
