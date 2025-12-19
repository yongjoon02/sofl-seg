#!/bin/bash
# XCA Soft Label Evaluation Script
# Evaluate CSNet trained with 3 different soft label types
# Results saved separately: xca_sauna, xca_gaussian, xca_smoothing

set -e  # Exit on error

cd /home/yongjun/soft-seg
source .venv/bin/activate

echo "============================================================"
echo "XCA Soft Label Evaluation"
echo "============================================================"
echo ""

# Find the latest checkpoint matching a pattern
find_latest_checkpoint() {
    local base_dir=$1
    local pattern=$2
    
    if [ -d "$base_dir" ]; then
        # Find directories matching the pattern, sorted by time (newest first)
        local latest=$(ls -dt "$base_dir"/${pattern}*/ 2>/dev/null | head -1)
        if [ -n "$latest" ] && [ -f "$latest/checkpoints/best.ckpt" ]; then
            echo "$latest/checkpoints/best.ckpt"
        else
            echo ""
        fi
    else
        echo ""
    fi
}

# Get checkpoint paths (experiments/csnet/xca/csnet_xca_<tag>_<date>/)
CKPT_SAUNA=$(find_latest_checkpoint "experiments/csnet/xca" "csnet_xca_sauna")
CKPT_GAUSSIAN=$(find_latest_checkpoint "experiments/csnet/xca" "csnet_xca_gaussian")
CKPT_SMOOTH=$(find_latest_checkpoint "experiments/csnet/xca" "csnet_xca_smooth")

echo "Found checkpoints:"
echo "  SAUNA:    ${CKPT_SAUNA:-'NOT FOUND'}"
echo "  Gaussian: ${CKPT_GAUSSIAN:-'NOT FOUND'}"
echo "  Smooth:   ${CKPT_SMOOTH:-'NOT FOUND'}"
echo ""

# Evaluate SAUNA
if [ -n "$CKPT_SAUNA" ]; then
    echo "[1/3] Evaluating SAUNA model..."
    python scripts/evaluate.py \
        --data xca \
        --models csnet \
        --checkpoint "$CKPT_SAUNA" \
        --output results/xca_sauna \
        --gpu 0
    echo "✅ SAUNA evaluation saved to results/xca_sauna/"
else
    echo "❌ SAUNA checkpoint not found, skipping..."
fi
echo ""

# Evaluate Gaussian
if [ -n "$CKPT_GAUSSIAN" ]; then
    echo "[2/3] Evaluating Gaussian model..."
    python scripts/evaluate.py \
        --data xca \
        --models csnet \
        --checkpoint "$CKPT_GAUSSIAN" \
        --output results/xca_gaussian \
        --gpu 0
    echo "✅ Gaussian evaluation saved to results/xca_gaussian/"
else
    echo "❌ Gaussian checkpoint not found, skipping..."
fi
echo ""

# Evaluate Smoothing
if [ -n "$CKPT_SMOOTH" ]; then
    echo "[3/3] Evaluating Label Smoothing model..."
    python scripts/evaluate.py \
        --data xca \
        --models csnet \
        --checkpoint "$CKPT_SMOOTH" \
        --output results/xca_smoothing \
        --gpu 0
    echo "✅ Smoothing evaluation saved to results/xca_smoothing/"
else
    echo "❌ Smoothing checkpoint not found, skipping..."
fi

echo ""
echo "============================================================"
echo "Evaluation Summary"
echo "============================================================"

# Print results if available
for result_dir in results/xca_sauna results/xca_gaussian results/xca_smoothing; do
    if [ -f "$result_dir/evaluation_xca.csv" ]; then
        echo ""
        echo "=== $(basename $result_dir) ==="
        cat "$result_dir/evaluation_xca.csv"
    fi
done

echo ""
echo "============================================================"
echo "✅ All evaluations completed!"
echo "============================================================"
