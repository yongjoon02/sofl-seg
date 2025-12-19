"""Loss functions for segmentation tasks.

All loss functions are organized in separate files for better maintainability:
- soft_ce_loss.py: Soft Cross Entropy Loss
- soft_bce_loss.py: Soft Binary Cross Entropy Loss
- dice_loss.py: Dice Loss
- focal_dice_loss.py: Focal Dice Loss
- l1_loss.py: L1 Loss (Mean Absolute Error)
- l2_loss.py: L2 Loss (Mean Squared Error)
- topo_loss.py: Topology-aware Loss
 - flow_matching_loss.py: Composite loss for flow matching models

Usage:
    from src.losses import SoftCrossEntropyLoss, SoftBCELoss, DiceLoss, FocalDiceLoss, L1Loss, L2Loss, TopoLoss
"""

# Import all loss functions from separate files
from .soft_ce_loss import SoftCrossEntropyLoss
from .soft_bce_loss import SoftBCELoss
from .dice_loss import DiceLoss
from .focal_dice_loss import FocalDiceLoss
from .l1_loss import L1Loss
from .l2_loss import L2Loss
from .l2_dice_reg_loss import L2DiceRegLoss
from .flow_reg_dice_loss import FlowRegDiceLoss
from .topo_loss import TopoLoss
from .flow_matching_loss import FlowMatchingLoss
from .huber_loss import HuberLoss


__all__ = [
    'SoftCrossEntropyLoss',
    'SoftBCELoss',
    'DiceLoss',
    'FocalDiceLoss',
    'L1Loss',
    'L2Loss',
    'L2DiceRegLoss',
    'FlowRegDiceLoss',
    'TopoLoss',
    'FlowMatchingLoss',
    'HuberLoss',
]


# Losses are automatically registered via @register_loss decorators above
# No need for manual registration function
