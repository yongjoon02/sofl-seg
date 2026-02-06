#!/bin/bash
# Train Flow Matching with SAUNA + VLM-FiLM (Stage 4/3 post-concat)

set -e  # Exit on error

# NCCL settings for multi-GPU
export NCCL_P2P_DISABLE=1

# Config file
CONFIG="configs/flow/xca/flow_sauna_medsegdiff.yaml"

echo "========================================"
echo "Training Flow Matching with VLM-FiLM"
echo "Config: $CONFIG"
echo "========================================"

# Run training
python scripts/train.py --config "$CONFIG"

echo "========================================"
echo "Training completed!"
echo "========================================"
