"""Model registry with metadata support.

Usage:
    # 데코레이터로 등록 (권장)
    @register_model(
        name='my_model',
        task='supervised',
        params=1_000_000,
        speed='fast',
        description='My custom model',
    )
    class MyModel(nn.Module):
        pass
    
    # 조회
    model_cls = MODEL_REGISTRY.get('my_model')
    model_info = get_model_info('my_model')
"""

from dataclasses import dataclass
from typing import Callable, Literal, Optional, List

from .base import Registry, MODEL_REGISTRY


@dataclass
class ModelInfo:
    """Model metadata for automatic configuration and documentation."""
    name: str
    class_ref: Callable  # Model class or factory function
    task: Literal['supervised', 'diffusion', 'flow']
    params: int  # Number of parameters
    speed: str   # 'fast', 'medium', 'slow'
    description: str
    paper_url: Optional[str] = None
    default_lr: float = 2e-4
    default_epochs: int = 300


def register_model(
    name: str,
    task: Literal['supervised', 'diffusion', 'flow'],
    params: int,
    speed: str,
    description: str,
    paper_url: Optional[str] = None,
    default_lr: float = 2e-4,
    default_epochs: int = 300,
):
    """Decorator to register a model with metadata.
    
    Example:
        @register_model(
        name='csnet',
        task='supervised',
        params=8_400_196,
        speed='fast',
            description='Channel & Spatial Attention Network',
        default_lr=2e-3,
        )
        class CSNet(nn.Module):
            pass
    """
    def decorator(cls: Callable) -> Callable:
        metadata = {
            'task': task,
            'params': params,
            'speed': speed,
            'description': description,
            'paper_url': paper_url,
            'default_lr': default_lr,
            'default_epochs': default_epochs,
        }
        MODEL_REGISTRY.register(name=name, obj=cls, metadata=metadata)
        return cls
    return decorator


def get_model_info(model_name: str) -> ModelInfo:
    """Get model metadata as ModelInfo dataclass.
    
    Args:
        model_name: Name of the model
    
    Returns:
        ModelInfo with all metadata
    
    Raises:
        KeyError: If model not found
    """
    cls = MODEL_REGISTRY.get(model_name)
    metadata = MODEL_REGISTRY.get_metadata(model_name)
    
    return ModelInfo(
        name=model_name,
        class_ref=cls,
        **metadata
    )


def list_models(task: Optional[str] = None) -> List[str]:
    """List available models, optionally filtered by task.
    
    Args:
        task: Filter by task ('supervised', 'diffusion', 'flow')
    
    Returns:
        List of model names
    """
    if task is None:
        return MODEL_REGISTRY.list()
    return MODEL_REGISTRY.list(task=task)


# =============================================================================
# Register built-in models (lazy import to avoid circular imports)
# =============================================================================

def _register_builtin_models():
    """Register built-in models. Called on first access."""
    from src.archs.components import CSNet, DSCNet
    from src.archs.components.unet import DhariwalConcatUNet, DhariwalConcatUNetMultiHead
    from src.archs.components.medsegdiff_flow import MedSegDiffFlow
    from src.archs.components.medsegdiff_flow_multitask import MedSegDiffFlowMultiTask
    from src.archs.components.segdiff_flow import SegDiffFlow
    from src.archs.components.binomial_diffusion import create_berdiff
    from src.archs.components.cold_diffusion import create_colddiff
    from src.archs.components.gaussian_diffusion import create_medsegdiff, create_segdiff
    
    # CSNet
    if 'csnet' not in MODEL_REGISTRY:
        MODEL_REGISTRY.register(
            name='csnet',
            obj=CSNet,
            metadata={
                'task': 'supervised',
                'params': 8_400_196,
                'speed': 'fast',
                'description': 'Channel & Spatial Attention Network (27.19 it/s)',
                'paper_url': None,
                'default_lr': 2e-3,
                'default_epochs': 300,
            }
        )
    
    # DSCNet
    if 'dscnet' not in MODEL_REGISTRY:
        MODEL_REGISTRY.register(
            name='dscnet',
            obj=DSCNet,
            metadata={
                'task': 'supervised',
                'params': 5_843_106,
                'speed': 'medium',
                'description': 'Dynamic Snake Convolution Network (3.80 it/s)',
                'paper_url': None,
                'default_lr': 2e-3,
                'default_epochs': 300,
            }
        )
    
    # MedSegDiff
    if 'medsegdiff' not in MODEL_REGISTRY:
        MODEL_REGISTRY.register(
            name='medsegdiff',
            obj=create_medsegdiff,
            metadata={
                'task': 'diffusion',
                'params': 16_224_737,
                'speed': 'slow',
                'description': 'MedSegDiff with Gaussian DDPM (stable)',
                'paper_url': None,
                'default_lr': 2e-4,
                'default_epochs': 500,
            }
        )

    # SegDiff
    if 'segdiff' not in MODEL_REGISTRY:
        MODEL_REGISTRY.register(
            name='segdiff',
            obj=create_segdiff,
            metadata={
                'task': 'diffusion',
                'params': 16_000_000,
                'speed': 'slow',
                'description': 'SegDiff with RRDB conditioning (Gaussian DDPM)',
                'paper_url': None,
                'default_lr': 2e-4,
                'default_epochs': 500,
            }
        )
    
    # BerDiff
    if 'berdiff' not in MODEL_REGISTRY:
        MODEL_REGISTRY.register(
            name='berdiff',
            obj=create_berdiff,
            metadata={
                'task': 'diffusion',
                'params': 9_327_521,
                'speed': 'slow',
                'description': 'Bernoulli Diffusion (binary-optimized)',
                'paper_url': None,
                'default_lr': 2e-4,
                'default_epochs': 500,
            }
        )

    # ColdDiff
    if 'colddiff' not in MODEL_REGISTRY:
        MODEL_REGISTRY.register(
            name='colddiff',
            obj=create_colddiff,
            metadata={
                'task': 'diffusion',
                'params': 16_224_737,
                'speed': 'slow',
                'description': 'Cold Diffusion (deterministic image->seg restoration)',
                'paper_url': None,
                'default_lr': 2e-4,
                'default_epochs': 500,
            }
        )
    
    # Dhariwal Concat UNet (Flow)
    if 'dhariwal_concat_unet' not in MODEL_REGISTRY:
        MODEL_REGISTRY.register(
            name='dhariwal_concat_unet',
            obj=DhariwalConcatUNet,
            metadata={
                'task': 'flow',
                'params': 20_000_000,
                'speed': 'slow',
                'description': 'Dhariwal UNet with concat conditioning (Flow Matching)',
                'paper_url': None,
                'default_lr': 2e-4,
                'default_epochs': 500,
            }
        )

    # Dhariwal Concat UNet Multi-head (Flow)
    if 'dhariwal_concat_unet_multihead' not in MODEL_REGISTRY:
        MODEL_REGISTRY.register(
            name='dhariwal_concat_unet_multihead',
            obj=DhariwalConcatUNetMultiHead,
            metadata={
                'task': 'flow',
                'params': 20_000_000,
                'speed': 'slow',
                'description': 'Dhariwal UNet concat conditioning with flow+geometry heads',
                'paper_url': None,
                'default_lr': 2e-4,
                'default_epochs': 500,
            }
        )

    # MedSegDiff Flow backbone (Flow)
    if 'medsegdiff_flow' not in MODEL_REGISTRY:
        MODEL_REGISTRY.register(
            name='medsegdiff_flow',
            obj=MedSegDiffFlow,
            metadata={
                'task': 'flow',
                'params': 25_000_000,
                'speed': 'slow',
                'description': 'MedSegDiff UNet backbone for flow matching (flow head only)',
                'paper_url': None,
                'default_lr': 2e-4,
                'default_epochs': 500,
            }
        )

    # MedSegDiff Flow multi-task (Flow)
    if 'medsegdiff_flow_multitask' not in MODEL_REGISTRY:
        MODEL_REGISTRY.register(
            name='medsegdiff_flow_multitask',
            obj=MedSegDiffFlowMultiTask,
            metadata={
                'task': 'flow',
                'params': 30_000_000,
                'speed': 'slow',
                'description': 'MedSegDiff flow with dual heads (hard+soft) multi-task',
                'paper_url': None,
                'default_lr': 2e-4,
                'default_epochs': 500,
            }
        )

    # SegDiff Flow backbone (Flow)
    if 'segdiff_flow' not in MODEL_REGISTRY:
        MODEL_REGISTRY.register(
            name='segdiff_flow',
            obj=SegDiffFlow,
            metadata={
                'task': 'flow',
                'params': 25_000_000,
                'speed': 'slow',
                'description': 'SegDiff UNet backbone for flow matching (flow head only)',
                'paper_url': None,
                'default_lr': 2e-4,
                'default_epochs': 500,
            }
        )


# Auto-register on import
_register_builtin_models()
