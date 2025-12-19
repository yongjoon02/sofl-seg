#!/bin/bash
# XCA Soft Label Training Script
# Train any network with 3 different soft label types: SAUNA, Gaussian, Smoothing
#
# Usage:
#   bash scripts/train_xca_soft_labels.sh --model csnet --gpus "0"       # 순차 학습
#   bash scripts/train_xca_soft_labels.sh --model csnet --gpus "0,1,2"   # 병렬 학습
#   bash scripts/train_xca_soft_labels.sh --model dscnet --gpus "3,4"    # 2개 병렬 + 1개 순차

set -e  # Exit on error

cd /home/yongjun/soft-seg
source .venv/bin/activate

# 기본값 (GPU는 필수)
MODEL=""
GPUS=""

# 인자 파싱
while [[ $# -gt 0 ]]; do
    case $1 in
        --model|-m)
            MODEL="$2"
            shift 2
            ;;
        --gpus|-g)
            GPUS="$2"
            shift 2
            ;;
        --help|-h)
            echo "Usage: bash scripts/train_xca_soft_labels.sh --model <model> --gpus <gpus>"
            echo ""
            echo "Required Options:"
            echo "  --model, -m    Model name (csnet, dscnet, etc.)"
            echo "  --gpus, -g     GPU indices (1개: 순차, 3개: 병렬)"
            echo ""
            echo "Examples:"
            echo "  # 순차 학습 (GPU 1개)"
            echo "  bash scripts/train_xca_soft_labels.sh --model csnet --gpus '0'"
            echo ""
            echo "  # 병렬 학습 (GPU 3개)"
            echo "  bash scripts/train_xca_soft_labels.sh --model csnet --gpus '0,1,2'"
            echo ""
            echo "  # 2개 병렬 + 1개 순차 (GPU 2개)"
            echo "  bash scripts/train_xca_soft_labels.sh --model dscnet --gpus '3,4'"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# 필수 인자 확인
if [[ -z "$MODEL" ]]; then
    echo "❌ Error: --model is required"
    echo "Usage: bash scripts/train_xca_soft_labels.sh --model <model> --gpus <gpus>"
    exit 1
fi

if [[ -z "$GPUS" ]]; then
    echo "❌ Error: --gpus is required"
    echo "Usage: bash scripts/train_xca_soft_labels.sh --model <model> --gpus <gpus>"
    exit 1
fi

# GPU 배열로 변환
IFS=',' read -ra GPU_ARRAY <<< "$GPUS"
NUM_GPUS=${#GPU_ARRAY[@]}

# Config 파일 경로
CONFIG_DIR="configs/supervised/xca"
CONFIG_SAUNA="${CONFIG_DIR}/${MODEL}_label_sauna.yaml"
CONFIG_GAUSSIAN="${CONFIG_DIR}/${MODEL}_label_gaussian.yaml"
CONFIG_SMOOTH="${CONFIG_DIR}/${MODEL}_label_smooth.yaml"

# Config 파일 존재 확인
check_config() {
    if [[ ! -f "$1" ]]; then
        echo "❌ Config file not found: $1"
        echo ""
        echo "Available configs in ${CONFIG_DIR}:"
        ls -1 ${CONFIG_DIR}/*.yaml 2>/dev/null || echo "  (none)"
        echo ""
        echo "💡 Tip: Create config files for ${MODEL}:"
        echo "   ${MODEL}_label_sauna.yaml"
        echo "   ${MODEL}_label_gaussian.yaml"
        echo "   ${MODEL}_label_smooth.yaml"
        exit 1
    fi
}

check_config "$CONFIG_SAUNA"
check_config "$CONFIG_GAUSSIAN"
check_config "$CONFIG_SMOOTH"

echo "============================================================"
echo "XCA Soft Label Training"
echo "============================================================"
echo "Model: ${MODEL}"
echo "GPUs:  ${GPUS} (${NUM_GPUS}개)"
if [[ $NUM_GPUS -eq 1 ]]; then
    echo "Mode:  순차 학습"
elif [[ $NUM_GPUS -ge 3 ]]; then
    echo "Mode:  병렬 학습 (3개 동시)"
else
    echo "Mode:  부분 병렬 (${NUM_GPUS}개 동시 + 순차)"
fi
echo ""
echo "Configs:"
echo "  SAUNA:    ${CONFIG_SAUNA}"
echo "  Gaussian: ${CONFIG_GAUSSIAN}"
echo "  Smooth:   ${CONFIG_SMOOTH}"
echo "============================================================"
echo ""

# GPU 개수에 따라 실행 방식 결정
if [[ $NUM_GPUS -ge 3 ]]; then
    # 3개 이상: 모두 병렬
    GPU_SAUNA=${GPU_ARRAY[0]}
    GPU_GAUSSIAN=${GPU_ARRAY[1]}
    GPU_SMOOTH=${GPU_ARRAY[2]}
    
    echo "[1/3] Training ${MODEL} with SAUNA on GPU $GPU_SAUNA..."
    CUDA_VISIBLE_DEVICES=$GPU_SAUNA uv run python scripts/train.py --config "$CONFIG_SAUNA" &
PID_SAUNA=$!

    echo "[2/3] Training ${MODEL} with Gaussian on GPU $GPU_GAUSSIAN..."
    CUDA_VISIBLE_DEVICES=$GPU_GAUSSIAN uv run python scripts/train.py --config "$CONFIG_GAUSSIAN" &
PID_GAUSSIAN=$!

    echo "[3/3] Training ${MODEL} with Smooth on GPU $GPU_SMOOTH..."
    CUDA_VISIBLE_DEVICES=$GPU_SMOOTH uv run python scripts/train.py --config "$CONFIG_SMOOTH" &
PID_SMOOTH=$!

echo ""
    echo "All jobs started in parallel:"
echo "  SAUNA:    PID=$PID_SAUNA (GPU $GPU_SAUNA)"
echo "  Gaussian: PID=$PID_GAUSSIAN (GPU $GPU_GAUSSIAN)"
echo "  Smooth:   PID=$PID_SMOOTH (GPU $GPU_SMOOTH)"
echo ""
    
    wait $PID_SAUNA && echo "✅ SAUNA completed"
    wait $PID_GAUSSIAN && echo "✅ Gaussian completed"
    wait $PID_SMOOTH && echo "✅ Smooth completed"

elif [[ $NUM_GPUS -eq 2 ]]; then
    # 2개: 2개 병렬 후 1개 순차
    GPU_1=${GPU_ARRAY[0]}
    GPU_2=${GPU_ARRAY[1]}
    
    echo "[1/3] Training ${MODEL} with SAUNA on GPU $GPU_1..."
    CUDA_VISIBLE_DEVICES=$GPU_1 uv run python scripts/train.py --config "$CONFIG_SAUNA" &
    PID_SAUNA=$!
    
    echo "[2/3] Training ${MODEL} with Gaussian on GPU $GPU_2..."
    CUDA_VISIBLE_DEVICES=$GPU_2 uv run python scripts/train.py --config "$CONFIG_GAUSSIAN" &
    PID_GAUSSIAN=$!
    
    echo ""
    echo "2 jobs started in parallel..."
    wait $PID_SAUNA && echo "✅ SAUNA completed"
    wait $PID_GAUSSIAN && echo "✅ Gaussian completed"
    
    echo ""
    echo "[3/3] Training ${MODEL} with Smooth on GPU $GPU_1..."
    CUDA_VISIBLE_DEVICES=$GPU_1 uv run python scripts/train.py --config "$CONFIG_SMOOTH"
    echo "✅ Smooth completed"

else
    # 1개: 모두 순차
    GPU=${GPU_ARRAY[0]}
    
    echo "[1/3] Training ${MODEL} with SAUNA on GPU $GPU..."
    CUDA_VISIBLE_DEVICES=$GPU uv run python scripts/train.py --config "$CONFIG_SAUNA"
    echo "✅ SAUNA completed"
    echo ""
    
    echo "[2/3] Training ${MODEL} with Gaussian on GPU $GPU..."
    CUDA_VISIBLE_DEVICES=$GPU uv run python scripts/train.py --config "$CONFIG_GAUSSIAN"
    echo "✅ Gaussian completed"
    echo ""
    
    echo "[3/3] Training ${MODEL} with Smooth on GPU $GPU..."
    CUDA_VISIBLE_DEVICES=$GPU uv run python scripts/train.py --config "$CONFIG_SMOOTH"
    echo "✅ Smooth completed"
fi

echo ""
echo "============================================================"
echo "✅ All ${MODEL} soft label training completed!"
echo "============================================================"
