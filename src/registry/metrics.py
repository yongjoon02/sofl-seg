"""Metric registry for evaluation metrics.

Usage:
    # 데코레이터로 등록
    @register_metric(name='dice', description='Dice coefficient')
    class Dice(Metric):
        pass
    
    # 조회
    metric_cls = METRIC_REGISTRY.get('dice')
"""

from dataclasses import dataclass
from typing import Optional, List, Type

from .base import Registry, METRIC_REGISTRY


@dataclass
class MetricInfo:
    """Metric metadata."""
    name: str
    class_ref: Type
    description: str
    higher_is_better: bool = True
    range: tuple = (0.0, 1.0)


def register_metric(
    name: str,
    description: str = '',
    higher_is_better: bool = True,
    range: tuple = (0.0, 1.0),
):
    """Decorator to register a metric.
    
    Example:
        @register_metric(
            name='dice',
            description='Dice coefficient for segmentation',
            higher_is_better=True,
            range=(0.0, 1.0),
        )
        class Dice(Metric):
            pass
    """
    def decorator(cls: Type) -> Type:
        metadata = {
            'description': description,
            'higher_is_better': higher_is_better,
            'range': range,
        }
        METRIC_REGISTRY.register(name=name, obj=cls, metadata=metadata)
        return cls
    return decorator


def get_metric_info(metric_name: str) -> MetricInfo:
    """Get metric metadata."""
    cls = METRIC_REGISTRY.get(metric_name)
    metadata = METRIC_REGISTRY.get_metadata(metric_name)
    
    return MetricInfo(
        name=metric_name,
        class_ref=cls,
        **metadata
    )


def list_metrics(higher_is_better: Optional[bool] = None) -> List[str]:
    """List all available metrics.
    
    Args:
        higher_is_better: Filter by optimization direction
    """
    if higher_is_better is None:
        return METRIC_REGISTRY.list()
    return METRIC_REGISTRY.list(higher_is_better=higher_is_better)

