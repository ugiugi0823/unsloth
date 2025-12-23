#!/bin/bash

# ============================================================================
# QA PEFT 학습 실행 스크립트 (Unsloth + DeepSpeed ZeRO)
# - Unsloth: 빠른 학습 속도와 메모리 효율성 (QLoRA 4-bit)
# - DeepSpeed ZeRO: 대규모 모델 분산 학습
# ============================================================================

# ┌─────────────────────────────────────────────────────────────────────────┐
# │ 🔧 설정 영역 (여기서 직접 수정)                                         │
# └─────────────────────────────────────────────────────────────────────────┘

# GPU 설정 (사용할 GPU 번호를 쉼표로 구분)
GPU_IDS="0,1"


# ZeRO 스테이지 설정: "2" 또는 "3"
ZERO_STAGE="3"


# ==========================================
# 🐞 디버깅 옵션 (여기만 추가하세요)
# ==========================================

# 1. NCCL 로그 레벨 상향 (통신 과정 확인)
export NCCL_DEBUG=INFO

# 2. PyTorch 디버그 모드 (데드락 위치 추적)
export TORCH_DISTRIBUTED_DEBUG=DETAIL



# ┌─────────────────────────────────────────────────────────────────────────┐
# │ 🛡️ NCCL 안정화 설정 (에러 방지용 추가)                                  │
# └─────────────────────────────────────────────────────────────────────────┘

# 1. 타임아웃 연장 (기본 10분 -> 1시간)
# 초기 컴파일이나 데이터 로드 시간이 길어질 때 멈추는 것을 방지합니다.
export NCCL_TIMEOUT=3600
export TORCH_NCCL_BLOCKING_WAIT=1

# 2. P2P/IB 비활성화 (통신 충돌 방지)
# 하드웨어 호환성 문제로 인한 Deadlock을 방지하기 위해 공유 메모리를 사용합니다.
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1

# 3. 디버그 로그 (필요시 주석 해제하여 로그 확인)
# export NCCL_DEBUG=INFO


# ┌─────────────────────────────────────────────────────────────────────────┐
# │ 💾 PyTorch CUDA 메모리 할당 최적화 (메모리 파편화 방지)                  │
# └─────────────────────────────────────────────────────────────────────────┘

# 메모리 파편화 문제를 해결하기 위한 환경 변수
# expandable_segments:True는 CUDA 메모리 할당 시 확장 가능한 세그먼트를 사용하여
# 메모리 파편화를 줄이고 OOM(Out of Memory) 오류를 방지합니다.
# PYTORCH_CUDA_ALLOC_CONF는 deprecated되었으므로 PYTORCH_ALLOC_CONF를 사용합니다.
export PYTORCH_ALLOC_CONF=expandable_segments:True


# ┌─────────────────────────────────────────────────────────────────────────┐
# │ 🚀 실행 영역 (수정 불필요)                                              │
# └─────────────────────────────────────────────────────────────────────────┘

# 환경 설정 (Unsloth가 설치된 가상환경 활성화)
# Unsloth 설치: pip install unsloth
source /home/rex/workspace/nl2sql/tr/real/bin/activate

# 로그 디렉토리 보장
mkdir -p ./logs

# GPU 개수 계산 (쉼표 개수 + 1)
NUM_GPUS=$(echo "${GPU_IDS}" | tr ',' '\n' | wc -l)

# ZeRO 스테이지에 따른 config 파일 선택
CONFIG_FILE="config/gemma_zero${ZERO_STAGE}.json"
DS_CONFIG_FILE="ds_config_zero${ZERO_STAGE}.json"

# ============================================================================
# 🚀 Unsloth + DeepSpeed 모드 (멀티 GPU, QLoRA 4-bit, BF16)
# ============================================================================
LOG_FILE="./logs/deepspeed_$(date +%Y%m%d_%H%M%S).log"

echo "========================================================"
echo "🚀 Unsloth + DeepSpeed 모드로 학습 시작..."
echo "   - Framework   : Unsloth (QLoRA 4-bit, Fast Training)"
echo "   - GPU IDs     : ${GPU_IDS} (총 ${NUM_GPUS}개)"
echo "   - ZeRO Stage  : ${ZERO_STAGE}"
echo "   - Config      : ${CONFIG_FILE}"
echo "   - NCCL Timeout: ${NCCL_TIMEOUT}초 (설정됨)"
echo "   - P2P Disable : YES (설정됨)"
echo "   - Memory Opt  : expandable_segments:True (설정됨)"
echo "========================================================"

# --include 옵션으로 특정 GPU 지정
nohup deepspeed --include localhost:${GPU_IDS} src/train_deepspeed.py \
    --config ${CONFIG_FILE} \
    --deepspeed ${DS_CONFIG_FILE} \
    >> "${LOG_FILE}" 2>&1 &

echo "[STARTED] PID=$! | Log=${LOG_FILE}"
echo ""
echo "📋 로그 확인: tail -f ${LOG_FILE}"