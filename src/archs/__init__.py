"""Model/architecture package.

Keep this module import-light to avoid circular imports during registry
initialization (e.g., when importing loss modules that register themselves).
"""

from __future__ import annotations

__all__ = [
    'SupervisedModel',
    'DiffusionModel'
]


def __getattr__(name: str):
    if name == 'SupervisedModel':
        from .supervised_model import SupervisedModel
        return SupervisedModel
    if name == 'DiffusionModel':
        from .diffusion_model import DiffusionModel
        return DiffusionModel
    raise AttributeError(name)
