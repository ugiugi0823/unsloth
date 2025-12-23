#!/bin/bash

# ============================================================================
# QA PEFT 학습 실행 스크립트 (Unsloth 멀티 GPU, DeepSpeed 없음)
# - Unsloth: 빠른 학습 속도와 메모리 효율성 (QLoRA 4-bit)
# ============================================================================

# GPU 설정
# GPU_IDS="0,1"
GPU_IDS="1"

# 💾 PyTorch CUDA 메모리 할당 최적화
export PYTORCH_ALLOC_CONF=expandable_segments:True
export CUDA_VISIBLE_DEVICES=${GPU_IDS}

# 환경 설정
source /home/rex/workspace/nl2sql/tr/real/bin/activate

# 로그 디렉토리
mkdir -p ./logs

# Config 파일 (DeepSpeed 없는 버전)
CONFIG_FILE="config/unsloth_full1.json"

# GPU 개수 계산
NUM_GPUS=$(echo "${GPU_IDS}" | tr ',' '\n' | wc -l)

LOG_FILE="./logs/full_GPU_${GPU_IDS}_$(date +%Y%m%d_%H%M%S).log"

echo "========================================================"
echo "🚀 Unsloth 모드로 학습 시작 (DeepSpeed 없음)..."
echo "   - Framework   : Unsloth (QLoRA 4-bit)"
echo "   - GPU         : ${GPU_IDS} (총 ${NUM_GPUS}개)"
echo "   - Config      : ${CONFIG_FILE}"
echo "========================================================"

nohup python src/train.py \
    --config ${CONFIG_FILE} \
    >> "${LOG_FILE}" 2>&1 &

echo "[STARTED] PID=$! | Log=${LOG_FILE}"
echo ""
echo "📋 로그 확인: tail -f ${LOG_FILE}"

