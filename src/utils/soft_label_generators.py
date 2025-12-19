"""
Soft Label Generators for Medical Image Segmentation.

This module provides three different soft label generation methods:
1. Label Smoothing: Simple uniform smoothing of binary labels
2. Gaussian Boundary: Distance-based soft boundaries using Gaussian kernel
3. SAUNA Transform: Structure-Aware UNcertainty Adaptation (boundary + thickness)

Usage:
    python -m src.utils.soft_label_generators --method sauna --input-dir data/xca_full/train/label --output-dir data/xca_full/train/label_sauna
"""

import os
from typing import Literal, Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from scipy.ndimage import distance_transform_edt


# =============================================================================
# Helper Functions (from generate_uncertainty.py)
# =============================================================================

def ensure_grayscale(image: np.ndarray) -> np.ndarray:
    """Ensure input is grayscale."""
    if len(image.shape) == 3:
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return image


def ensure_binary_gt(gt: np.ndarray) -> np.ndarray:
    """Ensure ground truth is binary (0 or 1) and uint8."""
    gt = ensure_grayscale(gt)
    gt = (gt > 0).astype(np.uint8)
    return gt


def compute_distance_transform(image: np.ndarray) -> np.ndarray:
    """Compute distance transform of binary image."""
    image = ensure_grayscale(image)
    return cv2.distanceTransform(image, cv2.DIST_L2, cv2.DIST_MASK_5)


def do_max_pooling(img_dist: np.ndarray, kernel_size: int = None, kernel_ratio: float = 1.0) -> np.ndarray:
    """Apply max pooling for thickness computation."""
    thickness_max = img_dist.max()
    kernel_size = thickness_max if kernel_size is None else kernel_size
    kernel_size = int(np.ceil(kernel_size))
    kernel_size = int(kernel_size * kernel_ratio)
    kernel_size = kernel_size + 1 if kernel_size % 2 == 0 else kernel_size
    kernel_size = max(kernel_size, 3)  # Minimum kernel size of 3
    
    img_dist_tensor = torch.tensor(img_dist).unsqueeze(0).unsqueeze(0).float()
    maxpool = torch.nn.MaxPool2d(kernel_size=kernel_size, stride=1, padding=kernel_size // 2)
    img_maxpool = maxpool(img_dist_tensor)[0, 0, :, :].numpy()
    return img_maxpool


# =============================================================================
# 1. Label Smoothing
# =============================================================================

def generate_label_smoothing(
    binary_mask: np.ndarray,
    smoothing_factor: float = 0.1,
) -> np.ndarray:
    """
    Generate soft labels using label smoothing.
    
    Simple approach: instead of hard 0/1, use (smoothing_factor) and (1 - smoothing_factor).
    This is the most basic form of soft labeling.
    
    Args:
        binary_mask: Binary segmentation mask (H, W), values in {0, 1} or {0, 255}
        smoothing_factor: Amount of smoothing (default: 0.1)
            - Background: 0 -> smoothing_factor
            - Foreground: 1 -> 1 - smoothing_factor
    
    Returns:
        soft_label: Soft label map (H, W) with values in [smoothing_factor, 1-smoothing_factor]
    
    Example:
        >>> mask = np.array([[0, 0, 1], [0, 1, 1], [1, 1, 1]])
        >>> soft = generate_label_smoothing(mask, smoothing_factor=0.1)
        >>> # Background pixels: 0.1, Foreground pixels: 0.9
    """
    # Normalize to binary 0/1
    gt = ensure_binary_gt(binary_mask)
    
    # Apply label smoothing
    soft_label = gt.astype(np.float32) * (1 - 2 * smoothing_factor) + smoothing_factor
    
    return soft_label.astype(np.float32)


# =============================================================================
# 2. Gaussian Boundary Soft Label
# =============================================================================

def generate_gaussian_boundary(
    binary_mask: np.ndarray,
    sigma: float = 3.0,
    boundary_width: int = 10,
    min_value: float = 0.0,
    max_value: float = 1.0,
) -> np.ndarray:
    """
    Generate soft labels with Gaussian-smoothed boundaries.
    
    Uses distance transform to create smooth transitions at vessel boundaries.
    The boundary region has uncertainty (soft values), while the core vessel
    and background have hard values.
    
    Args:
        binary_mask: Binary segmentation mask (H, W), values in {0, 1} or {0, 255}
        sigma: Gaussian sigma for boundary smoothing (default: 3.0)
        boundary_width: Width of the boundary region in pixels (default: 10)
        min_value: Minimum soft label value for background (default: 0.0)
        max_value: Maximum soft label value for foreground (default: 1.0)
    
    Returns:
        soft_label: Soft label map (H, W) with smooth boundaries
    
    Algorithm:
        1. Compute distance transform from boundary (both inside and outside)
        2. Apply sigmoid function based on signed distance
        3. Normalize to [min_value, max_value] range
    """
    gt = ensure_binary_gt(binary_mask)
    
    # Compute signed distance transform
    dist_inside = compute_distance_transform(gt)
    dist_outside = compute_distance_transform(1 - gt)
    signed_distance = dist_inside - dist_outside
    
    # Apply sigmoid function for smooth transition
    # sigmoid(x/sigma) gives smooth transition centered at boundary
    soft_label = 1.0 / (1.0 + np.exp(-signed_distance / sigma))
    
    # Normalize to [min_value, max_value]
    soft_label = soft_label * (max_value - min_value) + min_value
    
    return soft_label.astype(np.float32)


# =============================================================================
# 3. SAUNA Transform (Structure-Aware UNcertainty Adaptation)
# =============================================================================

def extract_boundary_uncertainty_map(
    gt: np.ndarray,
) -> Tuple[np.ndarray, float]:
    """
    Extract boundary uncertainty map using distance transform.
    
    Maps distance to boundary to uncertainty values:
    - At boundary (dist=0): value is 0
    - Far inside foreground: value approaches +1
    - Far outside (background): value approaches -1
    
    Args:
        gt: Binary mask (H, W), values in {0, 255} or {0, 1}
    
    Returns:
        boundary_map: Boundary uncertainty in [-1, 1] range
        fg_max: Normalization factor used (dynamic, from image)
    """
    gt = ensure_binary_gt(gt)
    # Convert to 0-255 for cv2.distanceTransform compatibility
    gt_255 = (gt * 255).astype(np.uint8)
    
    # Foreground distance transform
    img_fg_dist = compute_distance_transform(gt_255)
    
    # Dynamic fg_max from actual image (official code behavior)
    fg_max = img_fg_dist.max()
    if fg_max == 0.0:
        return np.full_like(img_fg_dist, -1.0, dtype=np.float32), fg_max
    
    # Background distance transform (negative)
    gt_bg = 255 - gt_255
    img_bg_dist = compute_distance_transform(gt_bg)
    img_bg_dist = -img_bg_dist
    img_bg_dist[img_bg_dist <= -fg_max] = -fg_max
    
    # Combined distance map normalized to [-1, 1]
    img_dist = (img_fg_dist + img_bg_dist) / (fg_max + 1e-6)
    
    return img_dist.astype(np.float32), fg_max


def extract_thickness_uncertainty_map(
    gt: np.ndarray,
    kernel_ratio: float = 1.0,
    target_c_label: str = "h",
) -> Tuple[np.ndarray, float]:
    """
    Extract thickness-based uncertainty map.
    
    Args:
        gt: Binary mask (H, W), values in {0, 255} or {0, 1}
        kernel_ratio: Scaling factor for max pooling kernel
        target_c_label: Type of thickness map ('h' for combined fg/bg)
    
    Returns:
        thickness_map: Thickness values normalized to [0, 1]
        thickness_max: Normalization factor used (dynamic, from image)
    """
    gt = ensure_binary_gt(gt)
    # Convert to 0-255 for cv2.distanceTransform compatibility
    gt_255 = (gt * 255).astype(np.uint8)
    
    # Foreground distance transform
    img_dist = compute_distance_transform(gt_255)
    
    # Dynamic thickness_max from actual image (official code behavior)
    thickness_max = img_dist.max()
    if thickness_max == 0.0:
        return np.zeros_like(img_dist, dtype=np.float32), thickness_max
    
    fg_max = thickness_max
    img_maxpool = do_max_pooling(img_dist, kernel_ratio=kernel_ratio)
    
    if target_c_label == "h":
        # Foreground thickness
        img_thick_pos = (gt_255 > 0) * img_maxpool / (thickness_max + 1e-6)
        
        # Background thickness
        img_bg_dist = compute_distance_transform(255 - gt_255)
        img_bg_dist[img_bg_dist >= fg_max] = fg_max
        img_bg_maxpool = do_max_pooling(img_bg_dist, kernel_ratio=kernel_ratio)
        img_thick_neg = (gt_255 <= 0) * img_bg_maxpool / (thickness_max + 1e-6)
        img_thick_neg = np.clip(img_thick_neg, 0.0, 1.0)
        
        img_swt = np.where(gt_255 > 0, img_thick_pos, img_thick_neg)
    else:
        # Foreground only
        img_thick_pos = (gt_255 > 0) * img_maxpool / (thickness_max + 1e-6)
        img_swt = img_thick_pos
    
    return img_swt.astype(np.float32), thickness_max


def extract_combined_uncertainty_map(
    gt_b: np.ndarray,
    gt_t: np.ndarray,
    target_c_label: str = "h",
) -> np.ndarray:
    """
    Combine boundary and thickness uncertainty maps (SAUNA).
    
    Args:
        gt_b: Boundary uncertainty map [-1, 1]
        gt_t: Thickness uncertainty map [0, 1]
        target_c_label: Combination method ('h' recommended)
    
    Returns:
        gt_c: Combined uncertainty map [-1, 1]
    """
    fg = gt_b > 0
    gt_c = gt_b.copy()
    gt_t_abs = np.abs(gt_t)
    
    if target_c_label == "c":
        gt_c[fg] = gt_b[fg] * gt_t[fg]
    elif target_c_label == "h":
        # Foreground: boundary + (1 - thickness)
        fg_mask = (gt_t_abs > 0) & (gt_b > 0)
        gt_c[fg_mask] = gt_b[fg_mask] + (1.0 - gt_t[fg_mask])
        
        # Background: boundary - (1 - thickness)
        bg_mask = (gt_t_abs > 0) & (gt_b < 0)
        gt_c[bg_mask] = gt_b[bg_mask] - (1.0 - gt_t[bg_mask])
    elif target_c_label == "hh":
        center = 1.0
        fg_mask = (gt_t_abs > 0) & (gt_b > 0)
        gt_c[fg_mask] = gt_b[fg_mask] + (center - gt_t[fg_mask])
        
        bg_mask = (gt_t_abs > 0) & (gt_b < 0)
        bg_zero_thickness = (gt_t_abs == 0) & (gt_c < 0)
        gt_c[bg_mask] = gt_b[bg_mask] - (center - gt_t[bg_mask])
        gt_c[bg_zero_thickness] = -1.0
    
    gt_c = np.clip(gt_c, -1.0, 1.0)
    return gt_c.astype(np.float32)


def generate_sauna_transform(
    binary_mask: np.ndarray,
    kernel_ratio: float = 1.0,
    target_c_label: str = "h",
) -> np.ndarray:
    """
    Generate SAUNA (Structure-Aware UNcertainty Adaptation) soft labels.
    
    Combines boundary uncertainty and thickness uncertainty:
    - Boundary uncertainty: distance from vessel boundary
    - Thickness uncertainty: vessel width information
    
    This creates soft labels where values close to 1 indicate confident foreground,
    values close to 0 indicate confident background, and values near 0.5 indicate
    uncertainty (typically at boundaries).
    
    Args:
        binary_mask: Binary segmentation mask (H, W), values in {0, 1} or {0, 255}
        kernel_ratio: Scaling for thickness max pooling (default: 1.0)
        target_c_label: Combination method ('h' recommended)
    
    Returns:
        soft_label: SAUNA soft label map (H, W) with values in [0, 1]
    
    Reference:
        SAUNA: Structure-Aware Uncertainty for Accurate Vessel Segmentation
    """
    gt = ensure_binary_gt(binary_mask)
    
    # Extract boundary uncertainty [-1, 1] (dynamic normalization)
    gt_b, _ = extract_boundary_uncertainty_map(gt)
    
    # Extract thickness uncertainty [0, 1] (dynamic normalization)
    gt_t, _ = extract_thickness_uncertainty_map(
        gt,
        kernel_ratio=kernel_ratio,
        target_c_label=target_c_label,
    )
    
    # Combine boundary and thickness (SAUNA)
    gt_c = extract_combined_uncertainty_map(gt_b, gt_t, target_c_label=target_c_label)
    
    # Convert from [-1, 1] to [0, 1]
    soft_label = (gt_c + 1.0) / 2.0
    
    return soft_label.astype(np.float32)


# =============================================================================
# Unified Interface
# =============================================================================

SoftLabelMethod = Literal['label_smoothing', 'gaussian_boundary', 'sauna']


def generate_soft_label(
    binary_mask: np.ndarray,
    method: SoftLabelMethod = 'sauna',
    **kwargs,
) -> np.ndarray:
    """
    Unified interface for generating soft labels.
    
    Args:
        binary_mask: Binary segmentation mask (H, W)
        method: Soft label method ('label_smoothing', 'gaussian_boundary', 'sauna')
        **kwargs: Method-specific parameters
    
    Returns:
        soft_label: Soft label map (H, W) with values in [0, 1]
    
    Example:
        >>> mask = load_binary_mask('label.png')
        >>> soft_ls = generate_soft_label(mask, method='label_smoothing', smoothing_factor=0.1)
        >>> soft_gb = generate_soft_label(mask, method='gaussian_boundary', sigma=3.0)
        >>> soft_sauna = generate_soft_label(mask, method='sauna', fg_max=11)
    """
    if method == 'label_smoothing':
        return generate_label_smoothing(binary_mask, **kwargs)
    elif method == 'gaussian_boundary':
        return generate_gaussian_boundary(binary_mask, **kwargs)
    elif method == 'sauna':
        return generate_sauna_transform(binary_mask, **kwargs)
    else:
        raise ValueError(f"Unknown soft label method: {method}")


# =============================================================================
# Batch Processing for Dataset
# =============================================================================

def process_dataset(
    input_dir: str,
    output_dir: str,
    method: SoftLabelMethod = 'sauna',
    save_as_uint8: bool = True,
    **kwargs,
) -> None:
    """
    Process all label images in a directory and generate soft labels.
    
    Args:
        input_dir: Directory containing binary label images
        output_dir: Directory to save soft label images
        method: Soft label method
        save_as_uint8: If True, save as 8-bit image (0-255), else float32 npy
        **kwargs: Method-specific parameters
    """
    from PIL import Image
    from tqdm import tqdm
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Get image files
    extensions = ('.png', '.bmp', '.jpg', '.jpeg', '.tif', '.tiff')
    image_files = [f for f in os.listdir(input_dir) if f.lower().endswith(extensions)]
    
    print(f"Processing {len(image_files)} images with method: {method}")
    print(f"Parameters: {kwargs}")
    
    for filename in tqdm(image_files, desc=f"Generating {method} soft labels"):
        # Load binary mask
        input_path = os.path.join(input_dir, filename)
        binary_mask = np.array(Image.open(input_path))
        
        # Generate soft label
        soft_label = generate_soft_label(binary_mask, method=method, **kwargs)
        
        # Save
        output_path = os.path.join(output_dir, filename)
        if save_as_uint8:
            soft_uint8 = (soft_label * 255).astype(np.uint8)
            Image.fromarray(soft_uint8).save(output_path)
        else:
            np.save(output_path.replace(os.path.splitext(filename)[1], '.npy'), soft_label)
    
    print(f"Saved soft labels to {output_dir}")


# =============================================================================
# CLI
# =============================================================================

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Generate soft labels for segmentation')
    parser.add_argument('--input-dir', required=True, help='Input directory with binary labels')
    parser.add_argument('--output-dir', required=True, help='Output directory for soft labels')
    parser.add_argument('--method', choices=['label_smoothing', 'gaussian_boundary', 'sauna'],
                        default='sauna', help='Soft label method')
    
    # Label smoothing params
    parser.add_argument('--smoothing-factor', type=float, default=0.1,
                        help='Smoothing factor for label_smoothing method')
    
    # Gaussian boundary params
    parser.add_argument('--sigma', type=float, default=3.0,
                        help='Gaussian sigma for gaussian_boundary method')
    parser.add_argument('--boundary-width', type=int, default=10,
                        help='Boundary width for gaussian_boundary method')
    
    # SAUNA params
    parser.add_argument('--kernel-ratio', type=float, default=1.0,
                        help='Kernel ratio for SAUNA thickness')
    
    args = parser.parse_args()
    
    # Build kwargs based on method
    kwargs = {}
    if args.method == 'label_smoothing':
        kwargs['smoothing_factor'] = args.smoothing_factor
    elif args.method == 'gaussian_boundary':
        kwargs['sigma'] = args.sigma
        kwargs['boundary_width'] = args.boundary_width
    elif args.method == 'sauna':
        kwargs['kernel_ratio'] = args.kernel_ratio
    
    process_dataset(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        method=args.method,
        **kwargs
    )
