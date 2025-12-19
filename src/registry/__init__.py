"""Registry system for models, datasets, architectures, losses, and metrics.

Usage:
    from src.registry import (
        # Registry instances
        MODEL_REGISTRY,
        DATASET_REGISTRY,
        ARCHS_REGISTRY,
        LOSS_REGISTRY,
        METRIC_REGISTRY,
        
        # Helper functions
        get_model_info,
        get_dataset_info,
        list_models,
        list_datasets,
        
        # Decorators for registration
        register_model,
        register_dataset,
        register_arch,
        register_loss,
        register_metric,
    )
    
    # Example: Register a new model
    @register_model(
        name='my_model',
        task='supervised',
        params=1_000_000,
        speed='fast',
        description='My custom model',
    )
    class MyModel(nn.Module):
        pass
    
    # Example: Get model info
    info = get_model_info('csnet')
    print(info.params)  # 8400196
"""

# Base Registry class and global instances
from .base import (
    Registry,
    RegistryEntry,
    DATASET_REGISTRY,
    ARCHS_REGISTRY,
    MODEL_REGISTRY,
    LOSS_REGISTRY,
    METRIC_REGISTRY,
)

# Dataset registry
from .datasets import (
    DatasetInfo,
    register_dataset,
    get_dataset_info,
    list_datasets,
)

# Model registry
from .models import (
    ModelInfo,
    register_model,
    get_model_info,
    list_models,
)

# Architecture registry
from .archs import (
    ArchInfo,
    register_arch,
    get_arch_info,
    list_archs,
)

# Loss registry
from .losses import (
    LossInfo,
    register_loss,
    get_loss_info,
    list_losses,
)

# Metric registry
from .metrics import (
    MetricInfo,
    register_metric,
    get_metric_info,
    list_metrics,
)


__all__ = [
    # Base
    'Registry',
    'RegistryEntry',
    
    # Global registry instances
    'MODEL_REGISTRY',
    'DATASET_REGISTRY',
    'ARCHS_REGISTRY',
    'LOSS_REGISTRY',
    'METRIC_REGISTRY',
    
    # Info dataclasses
    'ModelInfo',
    'DatasetInfo',
    'ArchInfo',
    'LossInfo',
    'MetricInfo',
    
    # Registration decorators
    'register_model',
    'register_dataset',
    'register_arch',
    'register_loss',
    'register_metric',
    
    # Getter functions
    'get_model_info',
    'get_dataset_info',
    'get_arch_info',
    'get_loss_info',
    'get_metric_info',
    
    # List functions
    'list_models',
    'list_datasets',
    'list_archs',
    'list_losses',
    'list_metrics',
]
