#!/bin/bash
# 범용 학습 스크립트
# Usage:
#   # 데이터셋 전체 학습 (4개 모델 병렬)
#   bash scripts/train.sh --data xca --gpus "0,1,2,3"
#
#   # 특정 모델만 학습
#   bash scripts/train.sh --data xca --models "csnet,dscnet" --gpus "0,1"
#
#   # 단일 config 학습
#   bash scripts/train.sh --config configs/supervised/xca/csnet.yaml --gpu 0
#
#   # DDP 학습 (여러 GPU로 1개 모델)
#   bash scripts/train.sh --config configs/flow/xca/flow.yaml --ddp --gpus "0,1,2,3"
#
#   # 체크포인트에서 이어서 학습
#   bash scripts/train.sh --config configs/flow/xca/flow.yaml --ddp --gpus "0,1,2,3" \
#       --resume experiments/.../checkpoints/last.ckpt

set -e

cd /home/yongjun/sofl-seg
source .venv/bin/activate
export PYTHONUNBUFFERED=1  # 즉시 로그 flush

# 기본값
CONFIG=""
DATA=""
MODELS=""
GPU=""
GPUS=""
RESUME=""
DDP=false
DDP_STRATEGY="ddp"
DDP_PRECISION="32-true"  # Default to FP32 for stability (can override with --precision)
LOG_SUFFIX=""  # 로그 파일명에 덧붙일 접미사 (예: _01, _02)

# 모든 모델 목록
ALL_MODELS=("csnet" "dscnet" "nnunet" "medsegdiff" "berdiff" "segdiff")

# 인자 파싱
while [[ $# -gt 0 ]]; do
    case $1 in
        --config|-c)
            CONFIG="$2"
            shift 2
            ;;
        --data|-d)
            DATA="$2"
            shift 2
            ;;
        --models|-m)
            MODELS="$2"
            shift 2
            ;;
        --gpu|-g)
            GPU="$2"
            shift 2
            ;;
        --gpus)
            GPUS="$2"
            shift 2
            ;;
        --resume|-r)
            RESUME="$2"
            shift 2
            ;;
        --ddp)
            DDP=true
            shift 1
            ;;
        --strategy)
            DDP_STRATEGY="$2"
            shift 2
            ;;
        --precision)
            DDP_PRECISION="$2"
            shift 2
            ;;
        --log-suffix)
            LOG_SUFFIX="$2"
            shift 2
            ;;
        --help|-h)
            echo "Usage:"
            echo "  # 데이터셋 전체 학습 (4개 모델 병렬, 각각 단일 GPU)"
            echo "  bash scripts/train.sh --data <dataset> --gpus <g1,g2,g3,g4>"
            echo ""
            echo "  # 특정 모델만 학습"
            echo "  bash scripts/train.sh --data <dataset> --models <m1,m2> --gpus <g1,g2>"
            echo ""
            echo "  # 단일 config 학습 (단일 GPU)"
            echo "  bash scripts/train.sh --config <config.yaml> --gpu <N>"
            echo ""
            echo "  # DDP 학습 (여러 GPU로 1개 모델)"
            echo "  bash scripts/train.sh --config <config.yaml> --ddp --gpus <g1,g2,g3>"
            echo ""
            echo "  # 체크포인트에서 이어서 학습"
            echo "  bash scripts/train.sh --config <config.yaml> --gpu <N> --resume <ckpt_path>"
            echo ""
            echo "Options:"
            echo "  --data, -d       Dataset (xca, octa500_3m, octa500_6m, rossa)"
            echo "  --models, -m     Models (csnet,dscnet,nnunet,medsegdiff,berdiff,segdiff)"
            echo "  --gpus           GPU indices (comma-separated)"
            echo "  --config, -c     Single config file path"
            echo "  --gpu, -g        Single GPU index"
            echo "  --resume, -r     Resume from checkpoint path"
            echo "  --ddp            Enable DDP mode"
            echo "  --strategy       DDP strategy (default: ddp)"
            echo "  --precision      DDP precision (default: 16-mixed)"
            echo "  --log-suffix     Append to log filename (e.g., _01, _02)"
            echo ""
            echo "Examples:"
            echo "  bash scripts/train.sh -d xca --gpus '0,1,2,3'"
            echo "  bash scripts/train.sh -c configs/flow/xca/flow.yaml --ddp --gpus '0,1,2,3'"
            echo "  bash scripts/train.sh -c configs/supervised/xca/csnet.yaml -g 0"
            echo "  bash scripts/train.sh -c configs/flow/xca/flow.yaml --ddp --gpus '0,1,2,3' \\"
            echo "      --resume experiments/.../checkpoints/last.ckpt"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# 로그 디렉토리
LOG_DIR="logs"
mkdir -p "${LOG_DIR}"

# Resume 옵션 처리
RESUME_ARG=""
if [[ -n "$RESUME" ]]; then
    if [[ ! -f "$RESUME" ]]; then
        echo "❌ Checkpoint file not found: $RESUME"
        exit 1
    fi
    RESUME_ARG="--resume ${RESUME}"
    echo "📂 Resuming from: ${RESUME}"
fi

# ============================================================
# Mode 1: DDP 학습 (여러 GPU로 1개 모델)
# ============================================================
if ${DDP}; then
    if [[ -z "$CONFIG" ]]; then
        echo "❌ Error: --config is required with --ddp"
        exit 1
    fi
    
    if [[ -z "$GPUS" ]]; then
        echo "❌ Error: --gpus is required with --ddp"
        exit 1
    fi
    
    if [[ ! -f "$CONFIG" ]]; then
        echo "❌ Config file not found: $CONFIG"
        exit 1
    fi
    
    export CUDA_VISIBLE_DEVICES=${GPUS}
    NUM_GPUS=$(echo "${GPUS}" | tr ',' '\n' | wc -l)
    
    CONFIG_NAME=$(basename "${CONFIG%.*}")
    TEMP_LOG_FILE="${LOG_DIR}/${CONFIG_NAME}_ddp_train${LOG_SUFFIX}.log"
    
    echo "============================================================"
    echo "DDP Training"
    echo "============================================================"
    echo "Config:    ${CONFIG}"
    echo "GPUs:      ${GPUS} (${NUM_GPUS} devices)"
    echo "Strategy:  ${DDP_STRATEGY}"
    echo "Precision: ${DDP_PRECISION}"
    echo "Log:       ${TEMP_LOG_FILE} (will be renamed with experiment_id)"
    if [[ -n "$RESUME" ]]; then
        echo "Resume:    ${RESUME}"
    fi
    echo "============================================================"
    echo ""
    
    # DDP 환경변수
    export NCCL_P2P_DISABLE=1
    
    TRAIN_CMD="python scripts/train.py --config ${CONFIG} ${RESUME_ARG}"
    
    echo "🚀 Starting DDP training..."
    echo "   Monitor: tail -f ${TEMP_LOG_FILE}"
    echo ""
    
    nohup bash -c "source .venv/bin/activate && \
        CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} \
        PYTHONUNBUFFERED=1 \
        DDP_DEVICES=-1 \
        DDP_STRATEGY=${DDP_STRATEGY} \
        DDP_PRECISION=${DDP_PRECISION} \
        uv run ${TRAIN_CMD}" > "${TEMP_LOG_FILE}" 2>&1 &
    
    PID=$!
    
    # Wait for experiment_id to appear in log, then rename file
    (
        sleep 5  # Wait for training to start
        for i in {1..90}; do
            if [[ -f "${TEMP_LOG_FILE}" ]]; then
                EXP_ID=$(grep -m 1 "Experiment ID:" "${TEMP_LOG_FILE}" 2>/dev/null | sed 's/.*Experiment ID: //' | tr -d '[:space:]')
                if [[ -n "$EXP_ID" ]]; then
                    FINAL_LOG_FILE="${LOG_DIR}/${CONFIG_NAME}_${EXP_ID}_ddp_train.log"
                    mv "${TEMP_LOG_FILE}" "${FINAL_LOG_FILE}" 2>/dev/null && echo "📝 Log renamed to: ${FINAL_LOG_FILE}" || true
                    break
                fi
            fi
            sleep 1
        done
    ) &
    echo "   PID: ${PID}"
    echo ""
    echo "============================================================"
    echo "✅ Training started in background!"
    echo "============================================================"
    echo ""
    echo "Monitor:  tail -f ${TEMP_LOG_FILE}"
    echo "         (Log will be renamed with experiment_id shortly)"
    echo "Stop:     kill ${PID}"
    exit 0
fi

# ============================================================
# Mode 2: 단일 config 학습 (단일 GPU)
# ============================================================
if [[ -n "$CONFIG" ]]; then
    if [[ -z "$GPU" ]]; then
        echo "❌ Error: --gpu is required with --config (or use --ddp --gpus for multi-GPU)"
        exit 1
    fi
    
    if [[ ! -f "$CONFIG" ]]; then
        echo "❌ Config file not found: $CONFIG"
        exit 1
    fi
    
    CONFIG_NAME=$(basename "${CONFIG%.*}")
    TEMP_LOG_FILE="${LOG_DIR}/${CONFIG_NAME}_train${LOG_SUFFIX}.log"
    
    echo "============================================================"
    echo "Single GPU Training"
    echo "============================================================"
    echo "Config: ${CONFIG}"
    echo "GPU:    ${GPU}"
    echo "Log:    ${TEMP_LOG_FILE} (will be renamed with experiment_id)"
    if [[ -n "$RESUME" ]]; then
        echo "Resume: ${RESUME}"
    fi
    echo "============================================================"
    echo ""
    
    TRAIN_CMD="python scripts/train.py --config ${CONFIG} ${RESUME_ARG}"
    
    echo "🚀 Starting training..."
    echo "   Monitor: tail -f ${TEMP_LOG_FILE}"
    echo ""
    
    nohup bash -c "CUDA_VISIBLE_DEVICES=${GPU} PYTHONUNBUFFERED=1 uv run ${TRAIN_CMD}" > "${TEMP_LOG_FILE}" 2>&1 &
    
    PID=$!
    
    # Wait for experiment_id to appear in log, then rename file
    (
        sleep 5  # Wait for training to start
        for i in {1..90}; do
            if [[ -f "${TEMP_LOG_FILE}" ]]; then
                EXP_ID=$(grep -m 1 "Experiment ID:" "${TEMP_LOG_FILE}" 2>/dev/null | sed 's/.*Experiment ID: //' | tr -d '[:space:]')
                if [[ -n "$EXP_ID" ]]; then
                    FINAL_LOG_FILE="${LOG_DIR}/${CONFIG_NAME}_${EXP_ID}_train.log"
                    mv "${TEMP_LOG_FILE}" "${FINAL_LOG_FILE}" 2>/dev/null && echo "📝 Log renamed to: ${FINAL_LOG_FILE}" || true
                    break
                fi
            fi
            sleep 1
        done
    ) &
    
    echo "   PID: ${PID}"
    echo ""
    echo "============================================================"
    echo "✅ Training started in background!"
    echo "============================================================"
    echo ""
    echo "Monitor:  tail -f ${TEMP_LOG_FILE}"
    echo "         (Log will be renamed with experiment_id shortly)"
    echo "Stop:     kill ${PID}"
    exit 0
fi

# ============================================================
# Mode 3: 데이터셋 기반 다중 모델 학습 (각각 단일 GPU)
# ============================================================
if [[ -z "$DATA" ]]; then
    echo "❌ Error: --data, --config, or --ddp is required"
    echo ""
    echo "Usage:"
    echo "  bash scripts/train.sh --data <dataset> --gpus <g1,g2,g3,g4>"
    echo "  bash scripts/train.sh --config <config.yaml> --gpu <N>"
    echo "  bash scripts/train.sh --config <config.yaml> --ddp --gpus <g1,g2,g3>"
    exit 1
fi

if [[ -z "$GPUS" ]]; then
    echo "❌ Error: --gpus is required with --data"
    echo "Usage: bash scripts/train.sh --data ${DATA} --gpus '0,1,2,3'"
    exit 1
fi

# GPU 배열로 변환
IFS=',' read -ra GPU_ARRAY <<< "$GPUS"
NUM_GPUS=${#GPU_ARRAY[@]}

# 모델 목록 결정
if [[ -n "$MODELS" ]]; then
    IFS=',' read -ra MODEL_ARRAY <<< "$MODELS"
else
    MODEL_ARRAY=("${ALL_MODELS[@]}")
fi
NUM_MODELS=${#MODEL_ARRAY[@]}

# GPU 개수 확인
if [[ $NUM_GPUS -lt $NUM_MODELS ]]; then
    echo "⚠️  Warning: ${NUM_MODELS} models but only ${NUM_GPUS} GPUs"
    echo "   Models will be queued (sequential when GPU busy)"
fi

echo "============================================================"
echo "Multi-Model Training"
echo "============================================================"
echo "Dataset: ${DATA}"
echo "Models:  ${MODEL_ARRAY[*]}"
echo "GPUs:    ${GPUS}"
echo "============================================================"
echo ""

# Config 파일 찾기 함수
get_config() {
    local model=$1
    local data=$2
    
    # Supervised models
    if [[ "$model" == "csnet" || "$model" == "dscnet" || "$model" == "nnunet" ]]; then
        echo "configs/supervised/${data}/${model}.yaml"
    # Diffusion models
    elif [[ "$model" == "medsegdiff" || "$model" == "berdiff" || "$model" == "segdiff" ]]; then
        echo "configs/diffusion/${data}/${model}.yaml"
    # Flow models
    elif [[ "$model" == "flow" || "$model" == "dhariwal_concat_unet" ]]; then
        echo "configs/flow/${data}/flow.yaml"
    else
        echo ""
    fi
}

# 학습 시작
PIDS=()
for i in "${!MODEL_ARRAY[@]}"; do
    MODEL="${MODEL_ARRAY[$i]}"
    GPU_IDX=$((i % NUM_GPUS))
    GPU="${GPU_ARRAY[$GPU_IDX]}"
    
    CONFIG=$(get_config "$MODEL" "$DATA")
    
    if [[ -z "$CONFIG" || ! -f "$CONFIG" ]]; then
        echo "⚠️  Config not found for ${MODEL} on ${DATA}, skipping..."
        continue
    fi
    
    TEMP_LOG_FILE="${LOG_DIR}/${DATA}_${MODEL}_train.log"
    
    echo "🚀 [${MODEL}] Starting on GPU ${GPU}..."
    echo "   Config: ${CONFIG}"
    echo "   Log: ${TEMP_LOG_FILE} (will be renamed with experiment_id)"
    
    nohup bash -c "CUDA_VISIBLE_DEVICES=${GPU} uv run python scripts/train.py --config ${CONFIG}" > "${TEMP_LOG_FILE}" 2>&1 &
    PIDS+=($!)
    
    # Wait for experiment_id to appear in log, then rename file
    (
        sleep 5  # Wait for training to start
        for i in {1..30}; do
            if [[ -f "${TEMP_LOG_FILE}" ]]; then
                EXP_ID=$(grep -m 1 "Experiment ID:" "${TEMP_LOG_FILE}" 2>/dev/null | sed 's/.*Experiment ID: //' | tr -d '[:space:]')
                if [[ -n "$EXP_ID" ]]; then
                    FINAL_LOG_FILE="${LOG_DIR}/${DATA}_${MODEL}_${EXP_ID}_train.log"
                    mv "${TEMP_LOG_FILE}" "${FINAL_LOG_FILE}" 2>/dev/null && echo "📝 [${MODEL}] Log renamed to: ${FINAL_LOG_FILE}" || true
                    break
                fi
            fi
            sleep 1
        done
    ) &
    
    sleep 2
done

echo ""
echo "============================================================"
echo "✅ All training jobs started in background!"
echo "============================================================"
echo ""
echo "PIDs: ${PIDS[*]}"
echo ""
echo "Monitor:"
echo "  tail -f ${LOG_DIR}/${DATA}_*.log"
echo "  watch -n 1 nvidia-smi"
echo ""
echo "Stop all:"
echo "  kill ${PIDS[*]}"
