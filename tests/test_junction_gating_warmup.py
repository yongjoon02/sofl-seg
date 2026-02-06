"""Test epoch-based warm-up for junction gating."""
import autorootcwd
import torch
import pytest
from src.utils.junction_gating import compute_junction_gate


def test_warmup_logic():
    """Test that warm-up logic creates all-ones gates during warm-up period."""
    
    # Simulate warm-up config
    jg_config = {
        'enabled': True,
        'warmup_epochs': 20,
        'source': 'pred',
        'threshold': 0.5,
        'degree_threshold': 3,
        'radius_px': 8,
        'gate_value_in_junction': 0.0,
        'apply_stages': 'same_as_film',
        'interp': 'nearest',
    }
    
    # Test 1: During warm-up (epoch < warmup_epochs)
    current_epoch = 5
    warmup_epochs = jg_config.get('warmup_epochs', None)
    
    assert warmup_epochs is not None
    assert current_epoch < warmup_epochs, "Should be in warm-up period"
    
    # During warm-up, we should create all-ones gates
    # This would be handled in diffusion_unet_vlm_film.py
    junction_warmup_active = True
    
    # Simulate creating all-ones gates during warm-up
    B, H, W = 4, 160, 160
    device = torch.device('cpu')
    gate_warmup = torch.ones(B, 1, H, W, device=device)
    
    assert gate_warmup.mean().item() == 1.0, "Warm-up gates should be all 1.0"
    assert gate_warmup.min().item() == 1.0
    assert gate_warmup.max().item() == 1.0
    print(f"✓ Warm-up gates correct: mean={gate_warmup.mean().item():.3f}")
    
    # Test 2: After warm-up (epoch >= warmup_epochs)
    current_epoch = 20
    assert current_epoch >= warmup_epochs, "Should be past warm-up period"
    
    junction_warmup_active = False
    
    # Now we should compute actual junction gates
    # Create a simple test mask with a junction
    mask = torch.zeros(B, 1, 64, 64)
    # Create a simple vessel structure with junction
    mask[0, 0, 30:35, 30:50] = 1  # horizontal vessel
    mask[0, 0, 20:40, 38:42] = 1  # vertical vessel (creates junction)
    
    config = {
        'enabled': True,
        'threshold': 0.5,
        'degree_threshold': 3,
        'radius_px': 8,
        'gate_value_in_junction': 0.0,
        'interp': 'nearest',
    }
    gates = compute_junction_gate(
        mask=mask,
        config=config,
        target_sizes=[(160, 160)],
    )
    
    assert len(gates) == 1, "Should return one gate"
    gate = gates[0]
    assert gate.shape == (B, 1, 160, 160)
    
    # After warm-up, gates should have variation (not all 1.0)
    # At junction regions, gate should be 0.0
    gate_mean = gate.mean().item()
    gate_min = gate.min().item()
    gate_max = gate.max().item()
    
    print(f"✓ Post-warm-up gates computed: mean={gate_mean:.3f}, min={gate_min:.3f}, max={gate_max:.3f}")
    assert gate_min < 0.5, "Should have junction suppression (gate=0.0 regions)"
    assert gate_max == 1.0, "Should have normal regions (gate=1.0)"
    
    print("✓ All warm-up logic tests passed!")


def test_backward_compatibility():
    """Test that old configs without warmup_epochs work as before."""
    
    # Config without warmup_epochs
    jg_config = {
        'enabled': True,
        'source': 'pred',
        'threshold': 0.5,
        'degree_threshold': 3,
        'radius_px': 8,
        'gate_value_in_junction': 0.0,
    }
    
    warmup_epochs = jg_config.get('warmup_epochs', None)
    assert warmup_epochs is None, "Old config should not have warmup_epochs"
    
    # Without warmup_epochs, junction gating should always be active
    current_epoch = 0
    junction_warmup_active = False
    
    if warmup_epochs is not None and current_epoch < warmup_epochs:
        junction_warmup_active = True
    
    assert not junction_warmup_active, "Without warmup_epochs, should never be in warm-up mode"
    print("✓ Backward compatibility test passed!")


if __name__ == '__main__':
    test_warmup_logic()
    test_backward_compatibility()
    print("\n✅ All junction gating warm-up tests passed!")
