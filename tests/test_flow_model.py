"""Flow Model Unit Test Script"""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '5'

import sys
sys.path.insert(0, '.')

import torch
import torch.nn as nn

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")

# ============================================================
# 1. Registry Check
# ============================================================
print("="*60)
print("1. Checking Registered Architectures")
print("="*60)

try:
    from src.registry import ARCHS_REGISTRY
    print(f"Total registered: {len(list(ARCHS_REGISTRY.keys()))}")
    for name in ARCHS_REGISTRY.keys():
        print(f"  - {name}")
    print("✅ Registry loaded successfully")
except Exception as e:
    print(f"❌ Registry error: {e}")

print()

# ============================================================
# 2. Flow Model Import Test
# ============================================================
print("="*60)
print("2. Importing FlowModel")
print("="*60)

try:
    from src.archs.flow_model import FlowModel
    print("✅ FlowModel imported successfully")
except Exception as e:
    print(f"❌ FlowModel import error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print()

# ============================================================
# 3. Flow Matcher Check
# ============================================================
print("="*60)
print("3. Checking Flow Matcher")
print("="*60)

try:
    from src.archs.components.flow import SchrodingerBridgeConditionalFlowMatcher
    flow_matcher = SchrodingerBridgeConditionalFlowMatcher(sigma=0.25)
    
    # Test flow sampling
    batch_size = 2
    x0 = torch.randn(batch_size, 1, 32, 32)  # noise
    x1 = torch.randn(batch_size, 1, 32, 32)  # target
    
    t, xt, ut = flow_matcher.sample_location_and_conditional_flow(x0, x1)
    print(f"  t shape: {t.shape}")
    print(f"  xt shape: {xt.shape}")
    print(f"  ut shape: {ut.shape}")
    print("✅ Flow matcher works correctly")
except Exception as e:
    print(f"❌ Flow matcher error: {e}")
    import traceback
    traceback.print_exc()

print()

# ============================================================
# 4. Check if dhariwal_concat_unet exists
# ============================================================
print("="*60)
print("4. Checking dhariwal_concat_unet Architecture")
print("="*60)

arch_name = 'dhariwal_concat_unet'
if arch_name in ARCHS_REGISTRY:
    print(f"✅ {arch_name} is registered")
    arch_class = ARCHS_REGISTRY.get(arch_name)
    print(f"  Class: {arch_class}")
else:
    print(f"❌ {arch_name} is NOT registered in ARCHS_REGISTRY")
    print("  Available architectures:")
    for name in ARCHS_REGISTRY.keys():
        print(f"    - {name}")

print()

# ============================================================
# 5. Try creating FlowModel (may fail if arch not registered)
# ============================================================
print("="*60)
print("5. Creating FlowModel Instance")
print("="*60)

try:
    model = FlowModel(
        arch_name='dhariwal_concat_unet',
        image_size=64,  # small for testing
        patch_plan=[(64, 1)],
        dim=32,
        timesteps=5,
        sigma=0.25,
        learning_rate=1e-4,
        weight_decay=1e-5,
        num_classes=2,
        num_ensemble=1,
        model_channels=16,  # small for testing
        channel_mult=[1, 2],
        num_blocks=1,
        attn_resolutions=[8],
    )
    print(f"✅ FlowModel created successfully")
    print(f"  Model type: {type(model)}")
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Total parameters: {total_params:,}")
    
    # Move model to GPU
    model = model.to(device)
    print(f"  Model moved to: {device}")
    
except Exception as e:
    print(f"❌ FlowModel creation error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print()

# ============================================================
# 6. Training Step Test (Forward Pass)
# ============================================================
print("="*60)
print("6. Testing Training Step (Forward Pass)")
print("="*60)

try:
    # Create dummy batch on GPU
    batch = {
        'image': torch.randn(2, 1, 64, 64, device=device),  # condition
        'geometry': torch.randn(2, 1, 64, 64, device=device),  # target
        'label': torch.randint(0, 2, (2, 1, 64, 64), device=device).float(),
    }
    
    # Training step
    model.train()
    loss = model.training_step(batch, batch_idx=0)
    
    print(f"  Loss: {loss.item():.4f}")
    print(f"  Loss shape: {loss.shape}")
    print("✅ Training step works correctly")
    
except Exception as e:
    print(f"❌ Training step error: {e}")
    import traceback
    traceback.print_exc()

print()

# ============================================================
# 7. Sampling Test (Inference)
# ============================================================
print("="*60)
print("7. Testing Sampling (Inference)")
print("="*60)

try:
    model.eval()
    with torch.no_grad():
        noise = torch.randn(2, 1, 64, 64, device=device)
        images = torch.randn(2, 1, 64, 64, device=device)
        
        output = model.sample(noise, images, return_intermediate=False)
        
        print(f"  Output shape: {output.shape}")
        print(f"  Output range: [{output.min().item():.4f}, {output.max().item():.4f}]")
        print("✅ Sampling works correctly")
        
except Exception as e:
    print(f"❌ Sampling error: {e}")
    import traceback
    traceback.print_exc()

print()

# ============================================================
# 8. Data Module Check
# ============================================================
print("="*60)
print("8. Checking XCA DataModule")
print("="*60)

try:
    from src.data.xca import XCADataModule
    
    # Check if directories exist
    import os
    train_exists = os.path.exists('data/xca_full/train')
    val_exists = os.path.exists('data/xca_full/val')
    test_exists = os.path.exists('data/xca_full/test')
    
    print(f"  Train dir exists: {train_exists}")
    print(f"  Val dir exists: {val_exists}")
    print(f"  Test dir exists: {test_exists}")
    print("✅ XCADataModule imported successfully")
    
except Exception as e:
    print(f"❌ DataModule error: {e}")
    import traceback
    traceback.print_exc()

print()
print("="*60)
print("UNIT TEST COMPLETE")
print("="*60)
