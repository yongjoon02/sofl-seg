"""
Soft Label Generator for Medical Image Segmentation.

This module provides utilities to generate soft labels from binary masks
for use in diffusion and flow matching models. Soft labels incorporate
boundary uncertainty and vessel thickness information.
"""
from typing import Literal, Optional

import numpy as np
import torch

SoftLabelType = Literal['none', 'boundary', 'thickness', 'sauna']


class SoftLabelGenerator:
    """
    Generate soft labels from binary masks during training.
    
    This is designed for diffusion/flow matching models where the denoising
    target should be soft (probabilistic) rather than hard (binary).
    
    Methods supported:
    - 'none': Keep binary labels (baseline)
    - 'boundary': Boundary uncertainty based on distance transform
    - 'thickness': Vessel thickness uncertainty  
    - 'sauna': Combined boundary + thickness (SAUNA method)
    
    Args:
        method: Soft label generation method
        cache: Whether to cache generated soft labels (default: True)
        fg_max: Maximum distance for foreground normalization (default: 11)
        thickness_max: Maximum thickness for normalization (default: 9)
        kernel_ratio: Kernel size ratio for thickness computation (default: 1.0)
    
    Example:
        >>> generator = SoftLabelGenerator(method='sauna', cache=True)
        >>> soft_labels = generator(binary_labels)  # [B, 1, H, W] -> [B, 1, H, W]
    """

    def __init__(
        self,
        method: SoftLabelType = 'none',
        cache: bool = True,
        fg_max: int = 11,
        thickness_max: int = 9,
        kernel_ratio: float = 1.0,
    ):
        self.method = method
        self.cache = cache
        self.fg_max = fg_max
        self.thickness_max = thickness_max
        self.kernel_ratio = kernel_ratio

        # Cache storage: {sample_id: soft_label_tensor}
        self._cache = {} if cache else None

        # Lazy import to avoid circular dependency
        self._uncertainty_functions = None

    def _get_uncertainty_functions(self):
        """Lazy import of uncertainty extraction functions."""
        if self._uncertainty_functions is None:
            from src.data.generate_uncertainty import (
                ensure_binary_gt,
                extract_boundary_uncertainty_map,
                extract_combined_uncertainty_map,
                extract_thickness_uncertainty_map,
            )
            self._uncertainty_functions = {
                'boundary': extract_boundary_uncertainty_map,
                'thickness': extract_thickness_uncertainty_map,
                'combined': extract_combined_uncertainty_map,
                'ensure_binary': ensure_binary_gt,
            }
        return self._uncertainty_functions

    def __call__(
        self,
        binary_labels: torch.Tensor,
        sample_ids: Optional[list] = None
    ) -> torch.Tensor:
        """
        Generate soft labels from binary labels.
        
        Args:
            binary_labels: Binary label tensor [B, 1, H, W], values in {0, 1}
            sample_ids: Optional list of sample IDs for caching (length B)
        
        Returns:
            soft_labels: Soft label tensor [B, 1, H, W], values in [0, 1]
        """
        if self.method == 'none':
            # Return binary labels as-is (convert to float)
            return binary_labels.float()

        batch_size = binary_labels.shape[0]
        device = binary_labels.device
        soft_labels = []

        for i in range(batch_size):
            sample_id = sample_ids[i] if sample_ids else None

            # Check cache
            if self.cache and sample_id and sample_id in self._cache:
                soft_labels.append(self._cache[sample_id])
                continue

            # Generate soft label
            soft_label = self._generate_single(binary_labels[i])

            # Cache if enabled
            if self.cache and sample_id:
                self._cache[sample_id] = soft_label.cpu()

            soft_labels.append(soft_label)

        # Stack and move to device
        result = torch.stack(soft_labels, dim=0).to(device)
        return result

    def _generate_single(self, binary_label: torch.Tensor) -> torch.Tensor:
        """
        Generate soft label for a single sample.
        
        Args:
            binary_label: [1, H, W] binary tensor
            
        Returns:
            soft_label: [1, H, W] soft tensor in [0, 1]
        """
        funcs = self._get_uncertainty_functions()

        # Convert to numpy
        label_np = binary_label.squeeze(0).cpu().numpy()
        gt = funcs['ensure_binary'](label_np)

        if self.method == 'boundary':
            # Boundary uncertainty only
            gt_b, _ = funcs['boundary'](gt)
            # Convert from [-1, 1] to [0, 1]
            soft_np = (gt_b + 1.0) / 2.0

        elif self.method == 'thickness':
            # Thickness uncertainty only
            gt_t, _ = funcs['thickness'](
                gt,
                target_c_label="h",
                kernel_ratio=self.kernel_ratio
            )
            # gt_t is already in [0, 1] range
            soft_np = gt_t

        elif self.method == 'sauna':
            # Combined boundary + thickness (SAUNA)
            gt_b, _ = funcs['boundary'](gt)
            gt_t, _ = funcs['thickness'](
                gt,
                target_c_label="h",
                kernel_ratio=self.kernel_ratio
            )
            gt_c = funcs['combined'](gt_b, gt_t, target_c_label="h")
            # Convert from [-1, 1] to [0, 1]
            soft_np = (gt_c + 1.0) / 2.0

        else:
            raise ValueError(f"Unknown soft label method: {self.method}")

        # Ensure valid range and convert to tensor
        soft_np = np.clip(soft_np, 0.0, 1.0)
        soft_tensor = torch.from_numpy(soft_np).float().unsqueeze(0)

        return soft_tensor

    def clear_cache(self):
        """Clear the cache to free memory."""
        if self._cache is not None:
            self._cache.clear()

    def get_cache_size(self) -> int:
        """Get the number of cached samples."""
        return len(self._cache) if self._cache is not None else 0
