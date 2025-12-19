"""OCT segmentation model components.

This module provides the core components for vessel segmentation models:
- CSNet: Channel and Spatial attention Network
- DSCNet: Dynamic Snake Convolution Network

Common components used by models:
- diffusion_unet: UNet architectures for diffusion models
- gaussian_diffusion: Gaussian diffusion process (MedSegDiff)
- binomial_diffusion: Bernoulli diffusion process (BerDiff)
- unet: UNet architectures for flow matching models (EDM-style)
"""

from .csnet import CSNet
from .dscnet import DSCNet
from . import unet  # Register flow matching architectures

__all__ = [
    'CSNet',
    'DSCNet',
    'unet',
]
