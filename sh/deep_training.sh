#!/bin/bash

# ============================================================================
# QA PEFT 학습 실행 스크립트 (DeepSpeed)
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
# │ 🚀 실행 영역 (수정 불필요)                                              │
# └─────────────────────────────────────────────────────────────────────────┘

# 환경 설정 (가상환경 경로가 맞는지 확인하세요)
source /home/rex/workspace/nl2sql/tr/unsloth/bin/activate

# 로그 디렉토리 보장
mkdir -p ./logs

# GPU 개수 계산 (쉼표 개수 + 1)
NUM_GPUS=$(echo "${GPU_IDS}" | tr ',' '\n' | wc -l)

# ZeRO 스테이지에 따른 config 파일 선택
CONFIG_FILE="config/gemma_zero${ZERO_STAGE}.json"
DS_CONFIG_FILE="ds_config_zero${ZERO_STAGE}.json"

# ============================================================================
# 🚀 DeepSpeed 모드 (멀티 GPU, BF16)
# ============================================================================
LOG_FILE="./logs/deepspeed_$(date +%Y%m%d_%H%M%S).log"

echo "========================================================"
echo "🚀 DeepSpeed 모드로 학습 시작..."
echo "   - GPU IDs     : ${GPU_IDS} (총 ${NUM_GPUS}개)"
echo "   - ZeRO Stage  : ${ZERO_STAGE}"
echo "   - Config      : ${CONFIG_FILE}"
echo "   - NCCL Timeout: ${NCCL_TIMEOUT}초 (설정됨)"
echo "   - P2P Disable : YES (설정됨)"
echo "========================================================"

# --include 옵션으로 특정 GPU 지정
nohup deepspeed --include localhost:${GPU_IDS} src/train_deepspeed.py \
    --config ${CONFIG_FILE} \
    --deepspeed ${DS_CONFIG_FILE} \
    >> "${LOG_FILE}" 2>&1 &

echo "[STARTED] PID=$! | Log=${LOG_FILE}"
echo ""
echo "📋 로그 확인: tail -f ${LOG_FILE}"