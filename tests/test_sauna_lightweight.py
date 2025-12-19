"""
Lightweight unit tests for SAUNA transformation (numpy only, no torch).

Tests:
1. SAUNA transform produces continuous values
2. Output range validation
3. Normalization to [0, 1]
4. Threshold consistency
"""
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
import cv2


def compute_distance_transform(image):
    """Compute distance transform using OpenCV."""
    image = cv2.distanceTransform(image, cv2.DIST_L2, cv2.DIST_MASK_5)
    return image


def extract_boundary_uncertainty_map(gt):
    """Extract boundary uncertainty map (signed distance transform)."""
    gt = gt.astype(np.uint8)
    
    # Foreground distance
    img_fg_dist = compute_distance_transform(gt)
    fg_max = img_fg_dist.max()
    
    if fg_max == 0.0:
        return np.full_like(img_fg_dist, -1.0), fg_max
    
    # Background distance
    gt_bg = 255 - gt
    img_bg_dist = compute_distance_transform(gt_bg)
    img_bg_dist = -img_bg_dist
    img_bg_dist[img_bg_dist <= -fg_max] = -fg_max
    
    # Combine
    img_dist = (img_fg_dist + img_bg_dist) / (fg_max + 1e-6)
    return img_dist.astype(np.float32), fg_max


def test_sauna_boundary_transform():
    """Test 1: SAUNA boundary transform produces continuous values"""
    print("\n" + "="*60)
    print("Test 1: SAUNA Boundary Transform")
    print("="*60)
    
    # Create simple binary mask (10x10 square in center)
    mask = np.zeros((32, 32), dtype=np.uint8)
    mask[11:21, 11:21] = 255
    
    print(f"Input mask shape: {mask.shape}")
    print(f"Input mask unique values: {np.unique(mask).tolist()}")
    
    # Apply boundary transform
    boundary_map, fg_max = extract_boundary_uncertainty_map(mask)
    
    print(f"\nBoundary map shape: {boundary_map.shape}")
    print(f"Boundary map range: [{boundary_map.min():.4f}, {boundary_map.max():.4f}]")
    print(f"Boundary map unique values: {len(np.unique(boundary_map))} unique values")
    print(f"Foreground max distance: {fg_max:.2f}")
    
    # Validate
    assert boundary_map.min() >= -1.0 - 1e-5, f"Boundary map min {boundary_map.min()} < -1.0"
    assert boundary_map.max() <= 1.0 + 1e-5, f"Boundary map max {boundary_map.max()} > 1.0"
    assert len(np.unique(boundary_map)) > 2, "Boundary map should be continuous (not binary)"
    
    print("✅ Test 1 PASSED: Boundary transform produces continuous values in [-1, 1]")
    return True


def test_sauna_normalization():
    """Test 2: SAUNA normalization to [0, 1]"""
    print("\n" + "="*60)
    print("Test 2: SAUNA Normalization to [0, 1]")
    print("="*60)
    
    # Create binary mask
    mask = np.zeros((32, 32), dtype=np.uint8)
    mask[11:21, 11:21] = 255
    
    # Apply boundary transform (returns [-1, 1])
    boundary_map, _ = extract_boundary_uncertainty_map(mask)
    
    # Normalize to [0, 1]
    normalized = (boundary_map + 1.0) / 2.0
    
    print(f"Raw boundary range: [{boundary_map.min():.4f}, {boundary_map.max():.4f}]")
    print(f"Normalized range: [{normalized.min():.4f}, {normalized.max():.4f}]")
    
    # Validate
    assert normalized.min() >= -1e-5, f"Normalized min {normalized.min()} < 0.0"
    assert normalized.max() <= 1.0 + 1e-5, f"Normalized max {normalized.max()} > 1.0"
    
    print("✅ Test 2 PASSED: Normalization produces values in [0, 1]")
    return True


def test_threshold_consistency():
    """Test 3: Threshold consistency (soft → hard label)"""
    print("\n" + "="*60)
    print("Test 3: Threshold Consistency")
    print("="*60)
    
    # Create binary mask
    mask = np.zeros((32, 32), dtype=np.uint8)
    mask[11:21, 11:21] = 255
    
    # SAUNA transform + normalize
    boundary_map, _ = extract_boundary_uncertainty_map(mask)
    geometry = (boundary_map + 1.0) / 2.0
    
    # Apply threshold
    preds = (geometry > 0.5).astype(np.int32)
    gt = (mask > 127).astype(np.int32)
    
    print(f"Geometry range: [{geometry.min():.4f}, {geometry.max():.4f}]")
    print(f"Predictions unique values: {np.unique(preds).tolist()}")
    print(f"GT unique values: {np.unique(gt).tolist()}")
    
    # Count foreground pixels
    pred_fg = (preds == 1).sum()
    gt_fg = (gt == 1).sum()
    
    print(f"\nForeground pixels - Pred: {pred_fg}, GT: {gt_fg}")
    
    # Validate: pred_fg should be similar to gt_fg
    fg_ratio = pred_fg / (gt_fg + 1e-6)
    print(f"Foreground preservation ratio: {fg_ratio:.2f}")
    
    assert fg_ratio > 0.7, f"Threshold should preserve most foreground (ratio: {fg_ratio:.2f})"
    
    print("✅ Test 3 PASSED: Threshold produces reasonable binary predictions")
    return True


def test_hard_vs_soft_label():
    """Test 4: Hard label vs Soft label comparison"""
    print("\n" + "="*60)
    print("Test 4: Hard Label vs Soft Label")
    print("="*60)
    
    # Create binary mask
    mask = np.zeros((32, 32), dtype=np.uint8)
    mask[11:21, 11:21] = 255
    
    # Hard label (identity)
    hard_label = (mask / 255.0).astype(np.float32)
    
    # Soft label (SAUNA)
    boundary_map, _ = extract_boundary_uncertainty_map(mask)
    soft_label = (boundary_map + 1.0) / 2.0
    
    print(f"Hard label unique values: {len(np.unique(hard_label))}")
    print(f"Soft label unique values: {len(np.unique(soft_label))}")
    
    print(f"\nHard label range: [{hard_label.min():.4f}, {hard_label.max():.4f}]")
    print(f"Soft label range: [{soft_label.min():.4f}, {soft_label.max():.4f}]")
    
    # Validate
    assert len(np.unique(hard_label)) == 2, "Hard label should be binary"
    assert len(np.unique(soft_label)) > 2, "Soft label should be continuous"
    
    print("✅ Test 4 PASSED: Hard and soft labels are correctly distinguished")
    return True


def test_loss_calculation_simulation():
    """Test 5: Loss calculation simulation"""
    print("\n" + "="*60)
    print("Test 5: Loss Calculation Simulation")
    print("="*60)
    
    # Create binary mask
    mask = np.zeros((32, 32), dtype=np.uint8)
    mask[11:21, 11:21] = 255
    
    # Hard label case
    hard_label = (mask / 255.0).astype(np.float32)
    noise_hard = np.random.randn(*hard_label.shape).astype(np.float32)
    velocity_target_hard = hard_label - noise_hard
    velocity_pred_hard = np.random.randn(*hard_label.shape).astype(np.float32)
    loss_hard = np.abs(velocity_pred_hard - velocity_target_hard).mean()
    
    print(f"Hard label loss: {loss_hard:.4f}")
    print(f"Hard label unique values: {len(np.unique(hard_label))}")
    
    # Soft label case
    boundary_map, _ = extract_boundary_uncertainty_map(mask)
    soft_label = (boundary_map + 1.0) / 2.0
    noise_soft = np.random.randn(*soft_label.shape).astype(np.float32)
    velocity_target_soft = soft_label - noise_soft
    velocity_pred_soft = np.random.randn(*soft_label.shape).astype(np.float32)
    loss_soft = np.abs(velocity_pred_soft - velocity_target_soft).mean()
    
    print(f"\nSoft label loss: {loss_soft:.4f}")
    print(f"Soft label unique values: {len(np.unique(soft_label))}")
    
    # Validate
    assert not np.isnan(loss_hard), "Hard label loss should not be NaN"
    assert not np.isnan(loss_soft), "Soft label loss should not be NaN"
    assert loss_hard > 0, "Hard label loss should be positive"
    assert loss_soft > 0, "Soft label loss should be positive"
    
    print("✅ Test 5 PASSED: Loss computation works for both hard and soft labels")
    return True


def test_edge_cases():
    """Test 6: Edge cases (empty mask, full mask)"""
    print("\n" + "="*60)
    print("Test 6: Edge Cases")
    print("="*60)
    
    # Empty mask
    empty_mask = np.zeros((32, 32), dtype=np.uint8)
    boundary_empty, fg_max_empty = extract_boundary_uncertainty_map(empty_mask)
    
    print(f"Empty mask - fg_max: {fg_max_empty:.2f}")
    print(f"Empty mask boundary range: [{boundary_empty.min():.4f}, {boundary_empty.max():.4f}]")
    
    # Full mask
    full_mask = np.ones((32, 32), dtype=np.uint8) * 255
    boundary_full, fg_max_full = extract_boundary_uncertainty_map(full_mask)
    
    print(f"\nFull mask - fg_max: {fg_max_full:.2f}")
    print(f"Full mask boundary range: [{boundary_full.min():.4f}, {boundary_full.max():.4f}]")
    
    # Validate: should not crash and produce valid outputs
    assert not np.isnan(boundary_empty).any(), "Empty mask should not produce NaN"
    assert not np.isnan(boundary_full).any(), "Full mask should not produce NaN"
    
    print("✅ Test 6 PASSED: Edge cases handled correctly")
    return True


def run_all_tests():
    """Run all tests"""
    print("\n" + "="*60)
    print("SAUNA Transform Lightweight Unit Tests (numpy only)")
    print("="*60)
    
    tests = [
        test_sauna_boundary_transform,
        test_sauna_normalization,
        test_threshold_consistency,
        test_hard_vs_soft_label,
        test_loss_calculation_simulation,
        test_edge_cases,
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            if test():
                passed += 1
        except Exception as e:
            failed += 1
            print(f"❌ {test.__name__} FAILED: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n" + "="*60)
    print(f"Test Results: {passed} passed, {failed} failed")
    print("="*60)
    
    if failed == 0:
        print("✅ All tests passed!")
        return True
    else:
        print(f"❌ {failed} test(s) failed")
        return False


if __name__ == '__main__':
    success = run_all_tests()
    sys.exit(0 if success else 1)

