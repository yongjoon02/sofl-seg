"""
SAUNA (Soft label with Uncertainty) geometry transformation.

This module provides functions to convert binary masks to SAUNA geometry maps
and vice versa. SAUNA combines boundary uncertainty (signed distance transform)
and thickness uncertainty (stroke width transform) to create soft labels.

Based on the SAUNA paper implementation.
"""
import numpy as np
import torch
import cv2


# ---- SAUNA helper functions (based on original implementation) ----

def compute_distance_transform(image):
    """Compute distance transform using OpenCV (same as SAUNA)."""
    image = cv2.distanceTransform(image, cv2.DIST_L2, cv2.DIST_MASK_5)
    return image


def transform_distance_map(dist, t):
    """Transform distance map (identity function in SAUNA)."""
    return dist


def do_max_pooling(img_dist, kernel_size=None, kernel_ratio=1):
    """Stroke width transform using max pooling (same as SAUNA)."""
    thickness_max = img_dist.max()
    kernel_size = thickness_max if kernel_size is None else kernel_size
    kernel_size = int(np.ceil(kernel_size))
    kernel_size = int(kernel_size * kernel_ratio)
    kernel_size = kernel_size + 1 if kernel_size % 2 == 0 else kernel_size
    
    img_dist_tensor = torch.tensor(img_dist).unsqueeze(0).unsqueeze(0)
    maxpool = torch.nn.MaxPool2d(
        kernel_size=kernel_size,
        stride=1, padding=kernel_size//2)
    img_maxpool = maxpool(img_dist_tensor)[0, 0, :, :].numpy()
    return img_maxpool


def extract_boundary_uncertainty_map(gt, transform_function=None):
    """
    Extract boundary uncertainty map (signed distance transform).
    Based on SAUNA's extract_boundary_uncertainty_map.
    """
    gt = gt.astype(np.uint8)
    
    # Determine the distance transform using OpenCV (same as SAUNA)
    img_fg_dist = compute_distance_transform(gt)
    img_fg_dist = transform_distance_map(img_fg_dist, transform_function)
    
    fg_max = img_fg_dist.max()
    if fg_max == 0.0:
        return np.full_like(img_fg_dist, -1.0), fg_max
    
    gt_bg = 255 - gt
    img_bg_dist = compute_distance_transform(gt_bg)
    img_bg_dist = transform_distance_map(img_bg_dist, transform_function)
    img_bg_dist = -img_bg_dist
    img_bg_dist[img_bg_dist <= -fg_max] = -fg_max
    img_dist = (img_fg_dist + img_bg_dist) / (fg_max + 1e-6)
    return img_dist.astype(np.float32), fg_max


def extract_thickness_uncertainty_map(gt, kernel_size=None, tr=None, target_c_label="b", kernel_ratio=1):
    """
    Extract thickness uncertainty map (stroke width transform).
    Based on SAUNA's extract_thickness_uncertainty_map.
    """
    gt = gt.astype(np.uint8)
    
    # Determine the distance transform
    img_dist = compute_distance_transform(gt)
    img_dist = transform_distance_map(img_dist, tr)
    thickness_max = img_dist.max()
    fg_max = thickness_max

    if thickness_max == 0.0:
        return np.zeros_like(img_dist, dtype=np.float32), thickness_max

    img_maxpool = do_max_pooling(
        img_dist,
        kernel_size=kernel_size,
        kernel_ratio=kernel_ratio,
    )
    
    if target_c_label in ["hh"]:
        img_thick_pos = (gt > 0) * img_maxpool / (thickness_max + 1e-6)
        img_maxpool = do_max_pooling(
            img_dist,
            kernel_size=kernel_size,
            kernel_ratio=kernel_ratio,
        )
        img_thick_neg = (gt <= 0) * img_maxpool / (thickness_max + 1e-6)
        img_swt = np.where(gt > 0, img_thick_pos, img_thick_neg)
    elif target_c_label in ["h"]:
        img_thick_pos = (gt > 0) * img_maxpool / (thickness_max + 1e-6)
        img_bg_dist = compute_distance_transform(255 - gt)
        img_bg_dist = transform_distance_map(img_bg_dist, tr)
        img_bg_dist[img_bg_dist >= fg_max] = fg_max
        img_bg_maxpool = do_max_pooling(
            img_bg_dist,
            kernel_size=kernel_size,
            kernel_ratio=kernel_ratio,
        )
        img_thick_neg = (gt <= 0) * img_bg_maxpool / (thickness_max + 1e-6)
        img_thick_neg = np.clip(img_thick_neg, a_min=0.0, a_max=1.0)
        img_swt = np.where(gt > 0, img_thick_pos, img_thick_neg)
    else:
        img_thick_pos = (gt > 0) * img_maxpool / (thickness_max + 1e-6)
        img_swt = img_thick_pos
    
    return img_swt.astype(np.float32), thickness_max


def extract_combined_uncertainty_map(gt_b, gt_t, target_c_label):
    """
    Combine boundary and thickness uncertainty maps.
    Based on SAUNA's extract_combined_uncertainty_map.
    """
    fg = gt_b > 0
    gt_c = gt_b.copy()
    gt_t_abs = np.abs(gt_t)
    
    if target_c_label == "c":
        gt_c[fg] = gt_b[fg] * gt_t[fg]
    elif target_c_label == "h":
        fg = (gt_t_abs > 0) & (gt_b > 0)
        gt_c[fg] = gt_b[fg] + (1.0 - gt_t[fg])
        bg = (gt_t_abs > 0) & (gt_b < 0)
        gt_c[bg] = gt_b[bg] - (1.0 - gt_t[bg])
    elif target_c_label == "hh":
        center = 1.0
        fg = (gt_t_abs > 0) & (gt_b > 0)
        gt_c[fg] = gt_b[fg] + (center - gt_t[fg])
        bg = (gt_t_abs > 0) & (gt_b < 0)
        bg_zero_thickness = (gt_t_abs == 0) & (gt_c < 0)
        gt_c[bg] = gt_b[bg] - (center - gt_t[bg])
        gt_c[bg_zero_thickness] = -1.0
    
    gt_c = np.clip(gt_c, a_min=-1.0, a_max=1.0)
    return gt_c.astype(np.float32)


# ---- public torch API ----

@torch.no_grad()
def to_geometry(
    mask: torch.Tensor,
    use_thickness: bool = True,
    target_c_label: str = "h",
    kernel_size: int = None,
    kernel_ratio: float = 1.0
) -> torch.Tensor:
    """
    Convert binary mask to SAUNA geometry map.
    Based on SAUNA's extract_boundary_uncertainty_map and extract_thickness_uncertainty_map.
    
    Args:
        mask: (B,1,H,W), {0,1} binary mask
        use_thickness: If True, combine boundary and thickness. If False, use boundary only.
        target_c_label: Combination mode ("h", "hh", "c")
        kernel_size: Kernel size for thickness max pooling (None = auto)
        kernel_ratio: Kernel ratio for thickness max pooling
    
    Returns:
        (B,1,H,W) in [-1,1], SAUNA geometry map
    """
    assert mask.dim() == 4 and mask.size(1) == 1
    
    mask_np = mask.detach().cpu().numpy()
    B = mask_np.shape[0]
    outs = []
    
    for b in range(B):
        m = (mask_np[b, 0] > 0.5).astype(np.uint8) * 255  # Convert to {0, 255}
        
        # Extract boundary uncertainty map
        gt_b, _ = extract_boundary_uncertainty_map(m, transform_function=None)
        
        if use_thickness:
            # Extract thickness uncertainty map
            gt_t, _ = extract_thickness_uncertainty_map(
                m,
                kernel_size=kernel_size,
                tr=None,
                target_c_label=target_c_label,
                kernel_ratio=kernel_ratio,
            )
            # Combine boundary and thickness
            gt_c = extract_combined_uncertainty_map(gt_b, gt_t, target_c_label)
        else:
            # Use boundary only
            gt_c = gt_b
        
        outs.append(gt_c)
    
    arr = np.stack(outs)[:, None]  # (B,1,H,W)
    return torch.from_numpy(arr).to(mask.device, dtype=torch.float32)


@torch.no_grad()
def from_geometry(
    geo: torch.Tensor,
    thresh: float = 0.0,
    dtype: torch.dtype | None = None
) -> torch.Tensor:
    """
    Convert SAUNA geometry map back to binary mask.
    
    Args:
        geo: (B,1,H,W) in [-1,1], from to_geometry(...)
        thresh: Threshold for binarization
        dtype: Output dtype
    
    Returns:
        (B,1,H,W) binary mask, such that mask ≈ original
    """
    assert geo.dim() == 4 and geo.size(1) == 1
    if dtype is None:
        dtype = torch.float32
    # geo == sauna, mask is sauna > 0 ↔ geo > 0
    return (geo > thresh).to(dtype=dtype)
