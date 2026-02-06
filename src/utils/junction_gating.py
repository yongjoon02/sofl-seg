"""
Junction-aware FiLM gating utilities.

This module provides functionality to generate spatial gate masks that identify
vessel junction regions for selective FiLM application.
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Optional, Dict, Any


def compute_junction_gate(
    mask: torch.Tensor,
    config: Dict[str, Any],
    target_sizes: Optional[list] = None
) -> Dict[int, torch.Tensor]:
    """
    Compute junction gate masks from a binary segmentation mask.
    
    Args:
        mask: Binary mask tensor (B, 1, H, W) or (B, H, W), values in [0, 1]
        config: Junction gating configuration dict with keys:
            - threshold: float, binarization threshold (default 0.5)
            - degree_threshold: int, min degree to consider junction (default 3)
            - radius_px: int, dilation radius around junctions (default 8)
            - gate_value_in_junction: float, gate value in junction region (default 0.0)
            - interp: str, interpolation mode for resizing (default "nearest")
        target_sizes: Optional list of (H, W) tuples for each stage resolution
            If provided, returns gates at multiple resolutions
    
    Returns:
        Dict mapping stage_idx to gate tensors (B, 1, H, W)
        Gate values: 1.0 = normal FiLM, gate_value_in_junction = suppressed FiLM
    """
    if not config.get('enabled', False):
        # Return all-ones gates (no gating)
        if target_sizes is None:
            return {0: torch.ones_like(mask[:, :1] if mask.dim() == 4 else mask.unsqueeze(1))}
        else:
            gates = {}
            for idx, (h, w) in enumerate(target_sizes):
                gates[idx] = torch.ones(
                    mask.shape[0], 1, h, w, 
                    device=mask.device, dtype=mask.dtype
                )
            return gates
    
    # Ensure mask is (B, 1, H, W)
    if mask.dim() == 3:
        mask = mask.unsqueeze(1)
    
    # Extract config parameters
    threshold = config.get('threshold', 0.5)
    degree_thresh = config.get('degree_threshold', 3)
    radius = config.get('radius_px', 8)
    gate_val = config.get('gate_value_in_junction', 0.0)
    interp = config.get('interp', 'nearest')
    
    batch_size = mask.shape[0]
    device = mask.device
    dtype = mask.dtype
    
    # Binarize mask
    binary_mask = (mask > threshold).float()
    
    # Compute junction regions for each sample in batch
    junction_regions = []
    for b in range(batch_size):
        mask_b = binary_mask[b, 0]  # (H, W)
        try:
            junction_region = _detect_junctions_heuristic(
                mask_b, 
                degree_thresh=degree_thresh, 
                radius=radius
            )
        except Exception as e:
            # Fallback: no junctions detected
            junction_region = torch.zeros_like(mask_b)
        junction_regions.append(junction_region)
    
    # Stack into batch
    junction_batch = torch.stack(junction_regions, dim=0).unsqueeze(1)  # (B, 1, H, W)
    
    # Create gate: 1.0 everywhere, gate_val in junction regions
    base_gate = torch.ones_like(junction_batch)
    base_gate = base_gate * (1.0 - junction_batch) + gate_val * junction_batch
    
    # Generate gates at multiple resolutions if requested
    if target_sizes is None:
        return {0: base_gate}
    
    gates = {}
    for idx, (h, w) in enumerate(target_sizes):
        if h == base_gate.shape[2] and w == base_gate.shape[3]:
            gates[idx] = base_gate
        else:
            # Resize gate to target resolution
            gates[idx] = F.interpolate(
                base_gate, 
                size=(h, w), 
                mode=interp,
                align_corners=None if interp == 'nearest' else False
            )
    
    return gates


def _detect_junctions_heuristic(
    binary_mask: torch.Tensor,
    degree_thresh: int = 3,
    radius: int = 8
) -> torch.Tensor:
    """
    Detect junction regions using skeleton + node degree heuristic.
    
    Args:
        binary_mask: Binary mask (H, W) on GPU/CPU
        degree_thresh: Minimum degree to consider as junction
        radius: Dilation radius around detected junctions
    
    Returns:
        Junction region mask (H, W), 1.0 at junctions, 0.0 elsewhere
    """
    # Try to use skimage for skeletonization (optional dependency)
    try:
        from skimage.morphology import skeletonize
        
        # Move to CPU numpy for skimage
        mask_np = binary_mask.detach().cpu().numpy().astype(bool)
        
        if not mask_np.any():
            # Empty mask, no junctions
            return torch.zeros_like(binary_mask)
        
        # Skeletonize
        skeleton = skeletonize(mask_np)
        
        # Compute node degrees (count 8-neighbors)
        from scipy import ndimage
        
        # Create 8-connectivity kernel
        kernel = np.array([[1, 1, 1],
                          [1, 0, 1],
                          [1, 1, 1]], dtype=np.float32)
        
        # Count neighbors: convolve with kernel
        degree_map = ndimage.convolve(
            skeleton.astype(np.float32),
            kernel,
            mode='constant'
        )
        
        # Multiply by skeleton to get degree only at skeleton pixels
        degree_map = degree_map * skeleton
        
        # Junction seeds: degree >= threshold
        junction_seeds = (degree_map >= degree_thresh)
        
        if not junction_seeds.any():
            # No junctions found
            return torch.zeros_like(binary_mask)
        
        # Dilate junction seeds by radius
        dilation_struct = ndimage.generate_binary_structure(2, 1)
        junction_region = ndimage.binary_dilation(
            junction_seeds, 
            structure=dilation_struct, 
            iterations=radius
        )
        
        # Convert back to torch tensor
        junction_tensor = torch.from_numpy(junction_region.astype(np.float32))
        return junction_tensor.to(binary_mask.device)
        
    except ImportError:
        # Fallback: simple torch-based approximation
        return _detect_junctions_torch_fallback(binary_mask, radius)


def _detect_junctions_torch_fallback(
    binary_mask: torch.Tensor,
    radius: int = 8
) -> torch.Tensor:
    """
    Fallback junction detection using pure PyTorch (less accurate).
    
    Uses morphological operations to approximate junction detection.
    """
    if not binary_mask.any():
        return torch.zeros_like(binary_mask)
    
    # Simple heuristic: find pixels with many neighbors
    # Use a 3x3 kernel to count neighbors
    kernel = torch.ones(1, 1, 3, 3, device=binary_mask.device, dtype=binary_mask.dtype)
    kernel[0, 0, 1, 1] = 0  # Don't count center pixel
    
    mask_4d = binary_mask.unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)
    neighbor_count = F.conv2d(mask_4d, kernel, padding=1)
    
    # Junction heuristic: skeleton pixels with >= 3 neighbors
    # Approximate skeleton as thin regions
    eroded = F.max_pool2d(mask_4d, 3, stride=1, padding=1)
    thinned = mask_4d * (mask_4d == eroded).float()
    
    # Junctions: thinned pixels with many neighbors
    junction_seeds = (thinned.squeeze() > 0.5) & (neighbor_count.squeeze() >= 3)
    
    if not junction_seeds.any():
        return torch.zeros_like(binary_mask)
    
    # Dilate junction seeds
    junction_4d = junction_seeds.float().unsqueeze(0).unsqueeze(0)
    dilation_kernel = torch.ones(1, 1, 2*radius+1, 2*radius+1, device=binary_mask.device)
    dilated = F.conv2d(junction_4d, dilation_kernel, padding=radius)
    junction_region = (dilated.squeeze() > 0.5).float()
    
    return junction_region


def visualize_junction_gate(
    image: torch.Tensor,
    mask: torch.Tensor,
    gate: torch.Tensor,
    save_path: Optional[str] = None
) -> np.ndarray:
    """
    Create a visualization overlay of junction gate on image.
    
    Args:
        image: Input image (B, C, H, W) or (H, W)
        mask: Binary mask (B, 1, H, W) or (H, W)
        gate: Gate tensor (B, 1, H, W) or (H, W)
        save_path: Optional path to save visualization
    
    Returns:
        RGB visualization as numpy array
    """
    # Convert to numpy and take first sample if batched
    if image.dim() == 4:
        image = image[0]
    if image.dim() == 3 and image.shape[0] in [1, 3]:
        image = image.permute(1, 2, 0)
    
    if mask.dim() == 4:
        mask = mask[0, 0]
    elif mask.dim() == 3:
        mask = mask[0]
    
    if gate.dim() == 4:
        gate = gate[0, 0]
    elif gate.dim() == 3:
        gate = gate[0]
    
    # To numpy
    image_np = image.detach().cpu().numpy()
    mask_np = mask.detach().cpu().numpy()
    gate_np = gate.detach().cpu().numpy()
    
    # Normalize image to [0, 1]
    if image_np.max() > 1.0:
        image_np = image_np / 255.0
    
    # Create RGB visualization
    if len(image_np.shape) == 2:
        vis = np.stack([image_np] * 3, axis=-1)
    else:
        vis = image_np.copy()
    
    # Overlay mask in green
    mask_overlay = np.zeros_like(vis)
    mask_overlay[:, :, 1] = mask_np
    vis = 0.7 * vis + 0.3 * mask_overlay
    
    # Overlay junction regions (gate < 1.0) in red
    junction_mask = (gate_np < 0.99)
    vis[junction_mask, 0] = 0.8  # Red for suppressed regions
    vis[junction_mask, 1] = 0.0
    vis[junction_mask, 2] = 0.0
    
    # Clip to [0, 1]
    vis = np.clip(vis, 0, 1)
    
    if save_path is not None:
        from PIL import Image
        img_pil = Image.fromarray((vis * 255).astype(np.uint8))
        img_pil.save(save_path)
    
    return vis
