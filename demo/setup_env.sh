#!/bin/bash
# LogSentinel 데모 구동 환경 자동 빌드 스크립트

echo "============================================="
echo "  LogSentinel Demo Environment Setup"
echo "============================================="

# 1. 가상환경 생성
if [ ! -d ".venv" ]; then
    echo "[1/3] 가상환경(.venv) 생성 중..."
    python3 -m venv .venv
    if [ $? -ne 0 ]; then
        echo "[오류] 가상환경 생성에 실패했습니다. Python3 및 venv 모듈 설치 여부를 확인하세요."
        exit 1
    fi
else
    echo "[정보] 이미 가상환경(.venv)이 존재합니다. 생략합니다."
fi

# 2. 가상환경 활성화 및 의존 패키지 설치
echo "[2/3] 가상환경 활성화 및 의존 라이브러리 설치 중..."
source .venv/bin/activate

pip install --upgrade pip
pip install -r requirements.txt

if [ $? -ne 0 ]; then
    echo "[오류] 라이브러리 설치 중 에러가 발생했습니다."
    exit 1
fi

# 3. 구동 스크립트 실행 권한 부여
echo "[3/3] 쉘 스크립트 실행 권한 부여 중..."
chmod +x run_server.sh run_client.sh

echo "============================================="
echo "  설정 완료! 아래 스크립트로 데모를 시작하세요."
echo "  - 서버 구동: ./run_server.sh"
echo "  - 클라이언트 시연: ./run_client.sh"
echo "============================================="
