#!/bin/bash
# LogSentinel FastAPI 데모 서버 구동 스크립트

# 1. 일반 가상환경(.venv)이 존재하면 우선 활성화
if [ -d ".venv" ]; then
    echo "[정보] 로컬 가상환경(.venv)을 활성화합니다."
    source .venv/bin/activate
else
    # 2. conda 환경이 사용 가능한지 확인
    CONDA_BASE=$(conda info --base 2>/dev/null || echo "/usr/local/anaconda3")
    if [ -f "$CONDA_BASE/etc/profile.d/conda.sh" ]; then
        echo "[정보] Conda 환경(logsentinel)을 활성화합니다."
        source "$CONDA_BASE/etc/profile.d/conda.sh"
        conda activate logsentinel
    else
        echo "[경고] 독립 가상환경이 발견되지 않아 로컬 글로벌 Python 환경에서 실행을 시도합니다."
    fi
fi

# 3. npx localtunnel 백그라운드 기동 및 주소 추출
# localtunnel 실행 출력을 임시 파일로 라우팅
LT_LOG=$(mktemp)
npx localtunnel --port 8000 > "$LT_LOG" 2>&1 &
LT_PID=$!

# 스크립트가 종료될 때(Ctrl+C 등) 백그라운드 localtunnel도 함께 클린업 종료
cleanup() {
    echo -e "\n[LogSentinel] Cleaning up localtunnel..."
    kill $LT_PID 2>/dev/null
    rm -f "$LT_LOG"
    exit 0
}
trap cleanup INT TERM EXIT

# 4. localtunnel 주소가 활성화될 때까지 최대 5초 대기
echo -n "[LogSentinel] Connecting to localtunnel service..."
LT_URL=""
for i in {1..10}; do
    sleep 0.5
    if grep -q "your url is:" "$LT_LOG"; then
        LT_URL=$(grep "your url is:" "$LT_LOG" | awk '{print $4}')
        echo " [OK]"
        break
    fi
    echo -n "."
done
if [ -z "$LT_URL" ]; then
    echo " [FAILED]"
fi

# 5. Spring Boot 스타일의 배너 및 기동 정보 출력
# ANSI Color Codes 적용 (Blue-Green 컨셉)
BLUE='\033[1;34m'
GREEN='\033[1;32m'
CYAN='\033[1;36m'
NC='\033[0m' # No Color

echo -e "${BLUE}"
if [ -f "banner.txt" ]; then
    # banner.txt 내용을 읽어서 변수들을 치환한 뒤 출력
    sed -e "s/\${application.title}/LogSentinel/g" \
        -e "s/\${application.version}/1.0.0/g" \
        -e "s/Spring Boot \${spring-boot.version}/FastAPI \& PyTorch/g" \
        banner.txt
else
    echo "  _                  _____            _   _            _"
    echo " | |                / ____|          | | (_)          | |"
    echo " | |     ___   __ _| (___   ___ _ __ | |_ _ _ __   ___| |"
    echo " | |    / _ \ / _\` |\___ \ / _ \ '_ \| __| | '_ \ / _ \ |"
    echo " | |___| (_) | (_| |____) |  __/ | | | |_| | | | |  __/ |"
    echo " |______\___/ \__, |_____/ \___|_| |_|\__|_|_| |_|\___|_|"
    echo "               __/ |                                     "
    echo "              |___/                                      "
fi
echo -e "${NC}"

echo -e "========================================================================="
echo -e "  * ${GREEN}Local Swagger UI${NC} : http://127.0.0.1:8000/docs"
if [ -n "$LT_URL" ]; then
    echo -e "  * ${CYAN}Public Demo URL${NC}  : ${LT_URL}/docs"
else
    echo -e "  * ${CYAN}Public Demo URL${NC}  : localtunnel 접속 지연 (npx localtunnel 확인 필요)"
fi
echo -e "========================================================================="
echo ""

# 6. uvicorn 서버 실행
python3 -m uvicorn src.main:app --port 8000 --reload

