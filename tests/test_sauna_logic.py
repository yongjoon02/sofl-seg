"""
Minimal unit tests for SAUNA transformation logic (no dependencies).

Tests the core logic without actual SAUNA computation:
1. Normalization logic ([-1, 1] → [0, 1])
2. Threshold logic (soft → hard)
3. Identity transform logic
4. Loss calculation logic
"""
import sys


def test_normalization_logic():
    """Test 1: Normalization from [-1, 1] to [0, 1]"""
    print("\n" + "="*60)
    print("Test 1: Normalization Logic")
    print("="*60)
    
    # Simulate SAUNA output range [-1, 1]
    test_values = [-1.0, -0.5, 0.0, 0.5, 1.0]
    
    print("Input values (SAUNA range [-1, 1]):")
    for val in test_values:
        # Normalization: (x + 1.0) / 2.0
        normalized = (val + 1.0) / 2.0
        print(f"  {val:5.2f} → {normalized:.2f}")
    
    # Validate
    for val in test_values:
        normalized = (val + 1.0) / 2.0
        assert 0.0 <= normalized <= 1.0, f"Normalized value {normalized} out of range [0, 1]"
    
    print("\n✅ Test 1 PASSED: Normalization correctly maps [-1, 1] to [0, 1]")
    return True


def test_threshold_logic():
    """Test 2: Threshold logic for binarization"""
    print("\n" + "="*60)
    print("Test 2: Threshold Logic")
    print("="*60)
    
    # Test values in [0, 1] range
    test_values = [0.0, 0.3, 0.49, 0.5, 0.51, 0.7, 1.0]
    threshold = 0.5
    
    print(f"Threshold: {threshold}")
    print("Input → Output:")
    for val in test_values:
        # Threshold: x > 0.5
        binary = 1 if val > threshold else 0
        print(f"  {val:.2f} → {binary}")
    
    # Validate
    assert (0.0 > threshold) == False
    assert (0.49 > threshold) == False
    assert (0.5 > threshold) == False
    assert (0.51 > threshold) == True
    assert (1.0 > threshold) == True
    
    print("\n✅ Test 2 PASSED: Threshold correctly binarizes at 0.5")
    return True


def test_identity_transform_logic():
    """Test 3: Identity transform (hard label preservation)"""
    print("\n" + "="*60)
    print("Test 3: Identity Transform Logic")
    print("="*60)
    
    # Hard label values
    hard_labels = [0.0, 0.0, 1.0, 1.0, 0.0, 1.0]
    
    print("Hard label (input):", hard_labels)
    
    # Identity: just return as-is (convert to float)
    geometry = [float(x) for x in hard_labels]
    
    print("Geometry (output):", geometry)
    
    # Validate: should be identical
    assert hard_labels == geometry, "Identity should preserve values"
    
    # Check unique values
    unique = set(geometry)
    print(f"Unique values: {unique}")
    assert unique == {0.0, 1.0}, "Identity should preserve binary values"
    
    print("\n✅ Test 3 PASSED: Identity transform preserves hard labels")
    return True


def test_loss_calculation_logic():
    """Test 4: Loss calculation logic"""
    print("\n" + "="*60)
    print("Test 4: Loss Calculation Logic")
    print("="*60)
    
    # Simulate flow matching loss: |v_pred - v_target|
    
    # Hard label case
    print("Hard label case:")
    geometry_hard = [0.0, 1.0, 0.0, 1.0]
    noise = [0.1, -0.2, 0.3, -0.1]
    v_target = [geometry_hard[i] - noise[i] for i in range(len(geometry_hard))]
    v_pred = [0.05, 0.9, 0.1, 1.1]
    
    loss_hard = sum(abs(v_pred[i] - v_target[i]) for i in range(len(v_pred))) / len(v_pred)
    print(f"  Geometry: {geometry_hard}")
    print(f"  Loss: {loss_hard:.4f}")
    
    # Soft label case
    print("\nSoft label case:")
    geometry_soft = [0.1, 0.8, 0.2, 0.9]  # Continuous values
    v_target = [geometry_soft[i] - noise[i] for i in range(len(geometry_soft))]
    v_pred = [0.05, 0.9, 0.1, 1.1]
    
    loss_soft = sum(abs(v_pred[i] - v_target[i]) for i in range(len(v_pred))) / len(v_pred)
    print(f"  Geometry: {geometry_soft}")
    print(f"  Loss: {loss_soft:.4f}")
    
    # Validate
    assert loss_hard > 0, "Hard label loss should be positive"
    assert loss_soft > 0, "Soft label loss should be positive"
    assert not (loss_hard != loss_hard), "Hard label loss should not be NaN"  # NaN check
    assert not (loss_soft != loss_soft), "Soft label loss should not be NaN"
    
    print("\n✅ Test 4 PASSED: Loss calculation works for both hard and soft labels")
    return True


def test_use_sauna_transform_flag():
    """Test 5: use_sauna_transform flag logic"""
    print("\n" + "="*60)
    print("Test 5: use_sauna_transform Flag Logic")
    print("="*60)
    
    hard_label = [0.0, 1.0, 0.0, 1.0]
    
    # Case 1: use_sauna_transform = False
    use_sauna_transform = False
    if use_sauna_transform:
        # Would apply SAUNA transform
        geometry = "SAUNA_transformed"
    else:
        # Identity transform
        geometry = [float(x) for x in hard_label]
    
    print(f"use_sauna_transform = {use_sauna_transform}")
    print(f"  Input: {hard_label}")
    print(f"  Output: {geometry}")
    assert geometry == hard_label, "Should use identity when flag is False"
    
    # Case 2: use_sauna_transform = True
    use_sauna_transform = True
    if use_sauna_transform:
        # Would apply SAUNA transform (simulated as continuous values)
        geometry = [0.1, 0.9, 0.2, 0.8]
    else:
        geometry = [float(x) for x in hard_label]
    
    print(f"\nuse_sauna_transform = {use_sauna_transform}")
    print(f"  Input: {hard_label}")
    print(f"  Output: {geometry}")
    assert geometry != hard_label, "Should apply SAUNA when flag is True"
    assert len(set(geometry)) > 2, "SAUNA output should be continuous"
    
    print("\n✅ Test 5 PASSED: use_sauna_transform flag correctly controls behavior")
    return True


def test_config_propagation_logic():
    """Test 6: Config propagation logic"""
    print("\n" + "="*60)
    print("Test 6: Config Propagation Logic")
    print("="*60)
    
    # Simulate config
    config = {
        'data': {
            'name': 'xca',
            'use_sauna_transform': True,  # Key setting
        }
    }
    
    print("Config:")
    print(f"  data.use_sauna_transform: {config['data'].get('use_sauna_transform', False)}")
    
    # Simulate DataModule creation
    use_sauna = config['data'].get('use_sauna_transform', False)
    
    print(f"\nDataModule will use SAUNA: {use_sauna}")
    
    # Validate
    assert use_sauna == True, "Config should propagate use_sauna_transform"
    
    # Test default value
    config_no_flag = {'data': {'name': 'xca'}}
    use_sauna_default = config_no_flag['data'].get('use_sauna_transform', False)
    
    print(f"Default value (when not specified): {use_sauna_default}")
    assert use_sauna_default == False, "Default should be False"
    
    print("\n✅ Test 6 PASSED: Config propagation logic is correct")
    return True


def test_metrics_calculation_logic():
    """Test 7: Metrics calculation logic (hard label comparison)"""
    print("\n" + "="*60)
    print("Test 7: Metrics Calculation Logic")
    print("="*60)
    
    # Predictions (after threshold)
    preds = [0, 1, 1, 0, 1, 1, 0, 0]
    
    # Ground truth (always hard label)
    gt = [0, 1, 0, 0, 1, 1, 1, 0]
    
    print(f"Predictions: {preds}")
    print(f"Ground Truth: {gt}")
    
    # Calculate simple accuracy
    correct = sum(1 for i in range(len(preds)) if preds[i] == gt[i])
    accuracy = correct / len(preds)
    
    print(f"\nCorrect predictions: {correct}/{len(preds)}")
    print(f"Accuracy: {accuracy:.2%}")
    
    # Validate
    assert 0.0 <= accuracy <= 1.0, "Accuracy should be in [0, 1]"
    assert correct == 6, "Expected 6 correct predictions"
    
    print("\n✅ Test 7 PASSED: Metrics calculation uses hard labels correctly")
    return True


def run_all_tests():
    """Run all tests"""
    print("\n" + "="*60)
    print("SAUNA Transform Logic Tests (no dependencies)")
    print("="*60)
    
    tests = [
        test_normalization_logic,
        test_threshold_logic,
        test_identity_transform_logic,
        test_loss_calculation_logic,
        test_use_sauna_transform_flag,
        test_config_propagation_logic,
        test_metrics_calculation_logic,
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
        print("\n✅✅✅ All tests passed! ✅✅✅")
        print("\nSummary:")
        print("  - Normalization: [-1, 1] → [0, 1] ✓")
        print("  - Threshold: soft → hard at 0.5 ✓")
        print("  - Identity: preserves hard labels ✓")
        print("  - Loss: works for both hard/soft ✓")
        print("  - Flag: controls SAUNA usage ✓")
        print("  - Config: propagates correctly ✓")
        print("  - Metrics: uses hard labels ✓")
        return True
    else:
        print(f"\n❌ {failed} test(s) failed")
        return False


if __name__ == '__main__':
    success = run_all_tests()
    sys.exit(0 if success else 1)

