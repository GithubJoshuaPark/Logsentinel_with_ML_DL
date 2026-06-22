#!/bin/bash
# LogSentinel 모의 테스트 클라이언트 구동 스크립트

# 1. 일반 가상환경(.venv)이 존재하면 우선 활성화
if [ -d ".venv" ]; then
    source .venv/bin/activate
else
    # 2. conda 환경이 사용 가능한지 확인
    CONDA_BASE=$(conda info --base 2>/dev/null || echo "/usr/local/anaconda3")
    if [ -f "$CONDA_BASE/etc/profile.d/conda.sh" ]; then
        source "$CONDA_BASE/etc/profile.d/conda.sh"
        conda activate logsentinel
    fi
fi

# 3. 테스트 클라이언트 스크립트 실행
echo "[LogSentinel] 모의 로그 시퀀스 추론 요청 시작..."
python3 ./src/test_client.py
