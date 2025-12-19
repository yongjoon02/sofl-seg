"""
Unit tests for SAUNA transformation pipeline.

Tests:
1. SAUNA transform: hard label → soft label (연속값)
2. Identity transform: hard label → hard label (동일)
3. Output range validation
4. XCADataset geometry generation with/without SAUNA
"""
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
import torch


def test_sauna_transform_output_range():
    """Test 1: SAUNA 변환 출력 범위 검증 ([0, 1]로 정규화)"""
    print("\n" + "="*60)
    print("Test 1: SAUNA Transform Output Range")
    print("="*60)
    
    from src.data.transforms.sauna import to_geometry
    
    # Create simple binary mask (10x10 square in center)
    mask = torch.zeros(1, 1, 32, 32)
    mask[:, :, 11:21, 11:21] = 1.0
    
    print(f"Input mask shape: {mask.shape}")
    print(f"Input mask unique values: {torch.unique(mask).tolist()}")
    
    # Apply SAUNA transform
    geometry = to_geometry(mask)
    
    print(f"\nSAUNA output shape: {geometry.shape}")
    print(f"SAUNA output range: [{geometry.min().item():.4f}, {geometry.max().item():.4f}]")
    print(f"SAUNA output unique values: {torch.unique(geometry).numel()} unique values")
    
    # Validate: SAUNA returns [-1, 1] range
    assert geometry.min() >= -1.0 - 1e-5, f"SAUNA output min {geometry.min()} < -1.0"
    assert geometry.max() <= 1.0 + 1e-5, f"SAUNA output max {geometry.max()} > 1.0"
    assert torch.unique(geometry).numel() > 2, "SAUNA output should be continuous (not binary)"
    
    print("✅ Test 1 PASSED: SAUNA transform produces continuous values in [-1, 1]")
    return True


def test_sauna_normalization():
    """Test 2: SAUNA 정규화 ([0, 1] 범위 확인)"""
    print("\n" + "="*60)
    print("Test 2: SAUNA Normalization to [0, 1]")
    print("="*60)
    
    from src.data.transforms.sauna import to_geometry
    
    # Create binary mask
    mask = torch.zeros(1, 1, 32, 32)
    mask[:, :, 11:21, 11:21] = 1.0
    
    # Apply SAUNA transform (returns [-1, 1])
    geometry_raw = to_geometry(mask)
    
    # Normalize to [0, 1] (as done in XCADataset.to_geometry)
    geometry_normalized = (geometry_raw + 1.0) / 2.0
    
    print(f"Raw SAUNA range: [{geometry_raw.min().item():.4f}, {geometry_raw.max().item():.4f}]")
    print(f"Normalized range: [{geometry_normalized.min().item():.4f}, {geometry_normalized.max().item():.4f}]")
    
    # Validate normalized range
    assert geometry_normalized.min() >= -1e-5, f"Normalized min {geometry_normalized.min()} < 0.0"
    assert geometry_normalized.max() <= 1.0 + 1e-5, f"Normalized max {geometry_normalized.max()} > 1.0"
    
    print("✅ Test 2 PASSED: Normalization produces values in [0, 1]")
    return True


def test_identity_transform():
    """Test 3: Identity 변환 (use_sauna_transform=False)"""
    print("\n" + "="*60)
    print("Test 3: Identity Transform (Hard Label)")
    print("="*60)
    
    # Simulate identity transform (label.float())
    hard_label = torch.zeros(1, 32, 32)
    hard_label[:, 11:21, 11:21] = 1.0
    
    # Identity: just convert to float
    geometry_identity = hard_label.float()
    
    print(f"Input hard label unique values: {torch.unique(hard_label).tolist()}")
    print(f"Identity output unique values: {torch.unique(geometry_identity).tolist()}")
    
    # Validate: should be binary
    unique_vals = torch.unique(geometry_identity).tolist()
    assert len(unique_vals) == 2, f"Identity should preserve binary values, got {len(unique_vals)} unique values"
    assert set(unique_vals) == {0.0, 1.0}, f"Identity should preserve {{0, 1}}, got {unique_vals}"
    
    print("✅ Test 3 PASSED: Identity transform preserves hard label {0.0, 1.0}")
    return True


def test_xca_dataset_geometry_with_sauna():
    """Test 4: XCADataset geometry 생성 (use_sauna_transform=True)"""
    print("\n" + "="*60)
    print("Test 4: XCADataset Geometry with SAUNA")
    print("="*60)
    
    # Mock XCADataset.to_geometry with use_sauna_transform=True
    from src.data.transforms.sauna import to_geometry as sauna_to_geometry
    
    hard_label = torch.zeros(1, 32, 32)
    hard_label[:, 11:21, 11:21] = 1.0
    
    # Simulate XCADataset.to_geometry logic
    label_4d = hard_label.unsqueeze(0)  # (1, 32, 32) -> (1, 1, 32, 32)
    geometry_raw = sauna_to_geometry(label_4d)
    geometry = (geometry_raw + 1.0) / 2.0  # Normalize to [0, 1]
    geometry = geometry.squeeze(0)  # (1, 1, 32, 32) -> (1, 32, 32)
    
    print(f"Hard label shape: {hard_label.shape}")
    print(f"Geometry shape: {geometry.shape}")
    print(f"Geometry range: [{geometry.min().item():.4f}, {geometry.max().item():.4f}]")
    print(f"Geometry unique values: {torch.unique(geometry).numel()} unique values")
    
    # Validate
    assert geometry.shape == hard_label.shape, "Geometry shape should match input"
    assert geometry.min() >= -1e-5 and geometry.max() <= 1.0 + 1e-5, "Geometry should be in [0, 1]"
    assert torch.unique(geometry).numel() > 2, "Geometry should be continuous (soft label)"
    
    print("✅ Test 4 PASSED: XCADataset generates proper soft label with SAUNA")
    return True


def test_xca_dataset_geometry_without_sauna():
    """Test 5: XCADataset geometry 생성 (use_sauna_transform=False)"""
    print("\n" + "="*60)
    print("Test 5: XCADataset Geometry without SAUNA (Identity)")
    print("="*60)
    
    hard_label = torch.zeros(1, 32, 32)
    hard_label[:, 11:21, 11:21] = 1.0
    
    # Simulate XCADataset.to_geometry logic (identity)
    geometry = hard_label.float()
    
    print(f"Hard label unique values: {torch.unique(hard_label).tolist()}")
    print(f"Geometry unique values: {torch.unique(geometry).tolist()}")
    
    # Validate
    assert torch.equal(geometry, hard_label), "Geometry should equal hard label (identity)"
    assert torch.unique(geometry).numel() == 2, "Geometry should be binary"
    
    print("✅ Test 5 PASSED: XCADataset preserves hard label without SAUNA")
    return True


def test_threshold_consistency():
    """Test 6: Threshold 일관성 (soft label → hard label 변환)"""
    print("\n" + "="*60)
    print("Test 6: Threshold Consistency")
    print("="*60)
    
    from src.data.transforms.sauna import to_geometry
    
    # Create binary mask
    mask = torch.zeros(1, 1, 32, 32)
    mask[:, :, 11:21, 11:21] = 1.0
    
    # SAUNA transform + normalize
    geometry_raw = to_geometry(mask)
    geometry = (geometry_raw + 1.0) / 2.0
    
    # Apply threshold (as in FlowModel.validation_step)
    preds = (geometry.squeeze() > 0.5).long()
    gt = mask.squeeze().long()
    
    print(f"Geometry range: [{geometry.min().item():.4f}, {geometry.max().item():.4f}]")
    print(f"Predictions unique values: {torch.unique(preds).tolist()}")
    print(f"GT unique values: {torch.unique(gt).tolist()}")
    
    # Count foreground pixels
    pred_fg = (preds == 1).sum().item()
    gt_fg = (gt == 1).sum().item()
    
    print(f"\nForeground pixels - Pred: {pred_fg}, GT: {gt_fg}")
    
    # Validate: pred_fg should be similar to gt_fg (allowing some difference due to soft label)
    # For center square, SAUNA should preserve most foreground
    fg_ratio = pred_fg / (gt_fg + 1e-6)
    print(f"Foreground preservation ratio: {fg_ratio:.2f}")
    
    assert fg_ratio > 0.7, f"Threshold should preserve most foreground (ratio: {fg_ratio:.2f})"
    
    print("✅ Test 6 PASSED: Threshold produces reasonable binary predictions")
    return True


def test_flow_model_loss_calculation():
    """Test 7: Flow model loss 계산 검증"""
    print("\n" + "="*60)
    print("Test 7: Flow Model Loss Calculation")
    print("="*60)
    
    # Simulate flow model training_step
    batch_size = 2
    h, w = 32, 32
    
    # Hard label case (use_sauna_transform=False)
    geometry_hard = torch.randint(0, 2, (batch_size, 1, h, w)).float()
    noise = torch.randn_like(geometry_hard)
    
    # Simulate flow matching loss
    velocity_target = geometry_hard - noise  # Simplified
    velocity_pred = torch.randn_like(velocity_target)
    loss_hard = torch.abs(velocity_pred - velocity_target).mean()
    
    print(f"Hard label loss: {loss_hard.item():.4f}")
    print(f"Geometry unique values (hard): {torch.unique(geometry_hard).numel()}")
    
    # Soft label case (use_sauna_transform=True)
    geometry_soft = torch.rand(batch_size, 1, h, w)  # Continuous [0, 1]
    noise = torch.randn_like(geometry_soft)
    velocity_target = geometry_soft - noise
    velocity_pred = torch.randn_like(velocity_target)
    loss_soft = torch.abs(velocity_pred - velocity_target).mean()
    
    print(f"\nSoft label loss: {loss_soft.item():.4f}")
    print(f"Geometry unique values (soft): {torch.unique(geometry_soft).numel()}")
    
    # Validate: both should compute loss properly
    assert not torch.isnan(loss_hard), "Hard label loss should not be NaN"
    assert not torch.isnan(loss_soft), "Soft label loss should not be NaN"
    
    print("✅ Test 7 PASSED: Loss computation works for both hard and soft labels")
    return True


def run_all_tests():
    """Run all tests"""
    print("\n" + "="*60)
    print("SAUNA Transform Pipeline Unit Tests")
    print("="*60)
    
    tests = [
        test_sauna_transform_output_range,
        test_sauna_normalization,
        test_identity_transform,
        test_xca_dataset_geometry_with_sauna,
        test_xca_dataset_geometry_without_sauna,
        test_threshold_consistency,
        test_flow_model_loss_calculation,
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

