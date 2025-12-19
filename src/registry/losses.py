"""Loss function registry.

Usage:
    # 데코레이터로 등록
    @register_loss(name='dice_loss', description='Dice loss for segmentation')
    class DiceLoss(nn.Module):
        pass
    
    # 조회
    loss_cls = LOSS_REGISTRY.get('dice_loss')
"""

from dataclasses import dataclass
from typing import Optional, List, Type

from .base import Registry, LOSS_REGISTRY


@dataclass
class LossInfo:
    """Loss function metadata."""
    name: str
    class_ref: Type
    description: str
    supports_multiclass: bool = True
    supports_soft_labels: bool = False


def register_loss(
    name: str,
    description: str = '',
    supports_multiclass: bool = True,
    supports_soft_labels: bool = False,
):
    """Decorator to register a loss function.
    
    Example:
        @register_loss(
            name='dice_loss',
            description='Soft Dice loss for segmentation',
            supports_soft_labels=True,
        )
        class DiceLoss(nn.Module):
            pass
    """
    def decorator(cls: Type) -> Type:
        metadata = {
            'description': description,
            'supports_multiclass': supports_multiclass,
            'supports_soft_labels': supports_soft_labels,
        }
        LOSS_REGISTRY.register(name=name, obj=cls, metadata=metadata)
        return cls
    return decorator


def get_loss_info(loss_name: str) -> LossInfo:
    """Get loss function metadata."""
    cls = LOSS_REGISTRY.get(loss_name)
    metadata = LOSS_REGISTRY.get_metadata(loss_name)
    
    return LossInfo(
        name=loss_name,
        class_ref=cls,
        **metadata
    )


def list_losses() -> List[str]:
    """List all available loss functions."""
    return LOSS_REGISTRY.list()

