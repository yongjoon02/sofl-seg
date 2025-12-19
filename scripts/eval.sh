#!/bin/bash
# 범용 평가 스크립트
# Usage:
#   bash scripts/eval.sh --data <dataset> --gpu <N>
#   bash scripts/eval.sh -d xca -g 0
#   bash scripts/eval.sh -d octa500_3m -g 1 --models "csnet,dscnet"

set -e

cd /home/yongjun/soft-seg
source .venv/bin/activate

# 기본값
DATA=""
GPU=0
MODELS=""
SAVE_PREDICTIONS=false

# 인자 파싱
while [[ $# -gt 0 ]]; do
    case $1 in
        --data|-d)
            DATA="$2"
            shift 2
            ;;
        --gpu|-g)
            GPU="$2"
            shift 2
            ;;
        --models|-m)
            MODELS="$2"
            shift 2
            ;;
        --save-predictions|-s)
            SAVE_PREDICTIONS=true
            shift 1
            ;;
        --help|-h)
            echo "Usage: bash scripts/eval.sh --data <dataset> --gpu <N>"
            echo ""
            echo "Required:"
            echo "  --data, -d       Dataset name (xca, octa500_3m, octa500_6m, rossa)"
            echo ""
            echo "Optional:"
            echo "  --gpu, -g        GPU index (default: 0)"
            echo "  --models, -m     Models to evaluate (comma-separated, default: all)"
            echo "  --save-predictions, -s  Save prediction images"
            echo ""
            echo "Examples:"
            echo "  bash scripts/eval.sh -d xca -g 0"
            echo "  bash scripts/eval.sh -d octa500_3m -g 1 -m 'csnet,dscnet'"
            echo "  bash scripts/eval.sh -d rossa -g 0 --save-predictions"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# 필수 인자 확인
if [[ -z "$DATA" ]]; then
    echo "❌ Error: --data is required"
    echo "Available: xca, octa500_3m, octa500_6m, rossa"
    exit 1
fi

# 로그 디렉토리
LOG_DIR="logs"
LOG_FILE="${LOG_DIR}/eval_${DATA}.log"
mkdir -p "${LOG_DIR}"

echo "============================================================"
echo "Evaluation"
echo "============================================================"
echo "Dataset: ${DATA}"
echo "GPU:     ${GPU}"
echo "Models:  ${MODELS:-all}"
echo "Log:     ${LOG_FILE}"
echo "============================================================"
echo ""

# 평가 실행
CMD="uv run python scripts/evaluate.py --data ${DATA} --output results/${DATA} --gpu ${GPU}"

if [[ -n "$MODELS" ]]; then
    CMD="${CMD} --models ${MODELS}"
fi

if ${SAVE_PREDICTIONS}; then
    CMD="${CMD} --save-predictions"
fi

echo "🚀 Starting evaluation..."
eval ${CMD} 2>&1 | tee "${LOG_FILE}"

echo ""
echo "============================================================"
echo "✅ Evaluation completed!"
echo "Results: results/${DATA}/"
echo "============================================================"

