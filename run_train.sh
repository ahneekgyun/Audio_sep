#!/bin/bash

# 작업 디렉토리 설정
WORKSPACE="workspace/AudioSep"


# yaml 파일 경로 설정
# 원하는 config로 경로 변경 필수
CONFIG_YAML="config/audiosep_LAAT_test.yaml"

# 체크포인트 경로 설정
CHECKPOINT_PATH=""

# 로그 파일 경로
LOG_DIR="/home/work/AHN/dcase2024_task9_baseline/nohup_output"
LOG_FILE="$LOG_DIR/LAAT_test_$(date +'%Y%m%d_%H%M').log"

# 로그 디렉토리가 없으면 생성
mkdir -p "$LOG_DIR"

# 1. 작업 디렉토리로 이동
cd AHN/dcase2024_task9_baseline

# 2. Python 학습 스크립트 실행
nohup python train.py \
    --workspace "$WORKSPACE" \
    --config_yaml "$CONFIG_YAML" \
    --resume_checkpoint_path "$CHECKPOINT_PATH" > "$LOG_FILE" 2>&1 &
