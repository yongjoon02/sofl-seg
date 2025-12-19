"""Architecture registry for neural network backbones.

Usage:
    # 데코레이터로 등록 (권장)
    @register_arch(
        name='my_unet',
        in_channels=1,
        out_channels=1,
        description='My custom UNet',
    )
    class MyUNet(nn.Module):
        pass
    
    # 조회
    arch_cls = ARCHS_REGISTRY.get('my_unet')
"""

from dataclasses import dataclass
from typing import Optional, List, Type

from .base import Registry, ARCHS_REGISTRY


@dataclass
class ArchInfo:
    """Architecture metadata."""
    name: str
    class_ref: Type
    in_channels: int
    out_channels: int
    description: str
    paper_url: Optional[str] = None


def register_arch(
    name: str,
    in_channels: int = 1,
    out_channels: int = 1,
    description: str = '',
    paper_url: Optional[str] = None,
):
    """Decorator to register an architecture with metadata.
    
    Example:
        @register_arch(
            name='dhariwal_unet',
            in_channels=2,
            out_channels=1,
            description='Dhariwal UNet for diffusion',
        )
        class DhariwalUNet(nn.Module):
            pass
    """
    def decorator(cls: Type) -> Type:
        metadata = {
            'in_channels': in_channels,
            'out_channels': out_channels,
            'description': description,
            'paper_url': paper_url,
        }
        ARCHS_REGISTRY.register(name=name, obj=cls, metadata=metadata)
        return cls
    return decorator


def get_arch_info(arch_name: str) -> ArchInfo:
    """Get architecture metadata."""
    cls = ARCHS_REGISTRY.get(arch_name)
    metadata = ARCHS_REGISTRY.get_metadata(arch_name)
    
    return ArchInfo(
        name=arch_name,
        class_ref=cls,
        **metadata
    )


def list_archs() -> List[str]:
    """List all available architectures."""
    return ARCHS_REGISTRY.list()

