#!/usr/bin/env python
"""Test config loading for OCTA500_6M and ROSSA datasets"""
import autorootcwd
from pathlib import Path
from src.utils.config import load_config
from src.registry import DATASET_REGISTRY

# Import datasets to register them
from src.data.octa500 import OCTA500_6M_DataModule
from src.data.rossa import ROSSA_DataModule

# Test OCTA500_6M
print("=" * 60)
print("Testing OCTA500_6M Config")
print("=" * 60)
octa_config = load_config("configs/flow/octa500_6m/flow_sauna_medsegdiff.yaml")
print(f"Dataset name: {octa_config['data']['name']}")
print(f"Model arch: {octa_config['model']['arch_name']}")
print(f"Mode: {octa_config['model']['mode']}")
print(f"Image size: {octa_config['model']['image_size']}")
print(f"VLM-FiLM enabled: {octa_config['model']['use_vlm_film']}")
print(f"VLM-FiLM stages: {octa_config['model']['vlm_film_decoder_stages']}")
print(f"Batch strategy: {octa_config['model']['vlm_film_config']['batch_strategy']}")
print(f"Junction gating enabled: {octa_config['model']['junction_gating_config']['enabled']}")
print(f"Attention resolutions: {octa_config['model']['attn_resolutions']}")

# Test dataset registration
try:
    dm_class = DATASET_REGISTRY.get('octa500_6m')
    print(f"\n✅ Dataset 'octa500_6m' registered: {dm_class}")
except Exception as e:
    print(f"\n❌ Dataset 'octa500_6m' NOT registered: {e}")

# Test ROSSA
print("\n" + "=" * 60)
print("Testing ROSSA Config")
print("=" * 60)
rossa_config = load_config("configs/flow/rossa/flow_sauna_medsegdiff.yaml")
print(f"Dataset name: {rossa_config['data']['name']}")
print(f"Model arch: {rossa_config['model']['arch_name']}")
print(f"Mode: {rossa_config['model']['mode']}")
print(f"Image size: {rossa_config['model']['image_size']}")
print(f"VLM-FiLM enabled: {rossa_config['model']['use_vlm_film']}")
print(f"VLM-FiLM stages: {rossa_config['model']['vlm_film_decoder_stages']}")
print(f"Batch strategy: {rossa_config['model']['vlm_film_config']['batch_strategy']}")
print(f"Junction gating enabled: {rossa_config['model']['junction_gating_config']['enabled']}")
print(f"Attention resolutions: {rossa_config['model']['attn_resolutions']}")

# Test dataset registration
try:
    dm_class = DATASET_REGISTRY.get('rossa')
    print(f"\n✅ Dataset 'rossa' registered: {dm_class}")
except Exception as e:
    print(f"\n❌ Dataset 'rossa' NOT registered: {e}")

print("\n" + "=" * 60)
print("✅ Config validation completed!")
print("=" * 60)
