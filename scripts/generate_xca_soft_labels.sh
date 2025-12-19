#!/bin/bash
# Generate all 3 types of soft labels for XCA dataset
# 1. Label Smoothing
# 2. Gaussian Boundary  
# 3. SAUNA Transform

set -e

BASE_DIR="data/xca_full"
SPLITS=("train" "val" "test")

echo "========================================="
echo "Generating Soft Labels for XCA Dataset"
echo "========================================="

for SPLIT in "${SPLITS[@]}"; do
    INPUT_DIR="${BASE_DIR}/${SPLIT}/label"
    
    if [ ! -d "$INPUT_DIR" ]; then
        echo "⚠️  Skipping ${SPLIT}: ${INPUT_DIR} not found"
        continue
    fi
    
    echo ""
    echo "Processing ${SPLIT} split..."
    
    # 1. Label Smoothing (factor=0.1)
    OUTPUT_DIR="${BASE_DIR}/${SPLIT}/label_smooth"
    echo "  → Label Smoothing: ${OUTPUT_DIR}"
    uv run python -m src.utils.soft_label_generators \
        --input-dir "${INPUT_DIR}" \
        --output-dir "${OUTPUT_DIR}" \
        --method label_smoothing \
        --smoothing-factor 0.1
    
    # 2. Gaussian Boundary (sigma=3.0)
    OUTPUT_DIR="${BASE_DIR}/${SPLIT}/label_gaussian"
    echo "  → Gaussian Boundary: ${OUTPUT_DIR}"
    uv run python -m src.utils.soft_label_generators \
        --input-dir "${INPUT_DIR}" \
        --output-dir "${OUTPUT_DIR}" \
        --method gaussian_boundary \
        --sigma 3.0 \
        --boundary-width 10
    
    # 3. SAUNA Transform (dynamic normalization like official code)
    OUTPUT_DIR="${BASE_DIR}/${SPLIT}/label_sauna"
    echo "  → SAUNA Transform: ${OUTPUT_DIR}"
    uv run python -m src.utils.soft_label_generators \
        --input-dir "${INPUT_DIR}" \
        --output-dir "${OUTPUT_DIR}" \
        --method sauna \
        --kernel-ratio 1.0

done

echo ""
echo "========================================="
echo "✅ All soft labels generated!"
echo "========================================="
echo ""
echo "Output directories:"
echo "  - label_smooth:   Label Smoothing (factor=0.1)"
echo "  - label_gaussian: Gaussian Boundary (sigma=3.0)"
echo "  - label_sauna:    SAUNA Transform"
