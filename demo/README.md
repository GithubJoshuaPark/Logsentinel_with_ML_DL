# [기록] 발표 당일용 독립 데모 폴더(demo) 구축 완료 보고

**작성일**: 2026년 06월 22일 22시 26분  
**작성자**: Antigravity (Gemini)  
**수신자**: Joshua님  
**대상 디렉토리**: [demo/](file:///Users/soromiso/Desktop/Dev/soromiso/LogSentinel/demo)

---

## 1. 작업 개요
- **요청 사항**: 발표용 PC(기본 파이썬 설치 환경)에 바로 폴더째 복사해 전달하여 즉석에서 데모를 구동할 수 있도록 독립적이고 휴대 가능한 `demo/` 폴더 구성 요청.
- **수행 내용**:
  - `demo/` 폴더를 새로 만들고 필수 소스 코드(`src/`), 로컬 가중치 파일들(`models/`), 그리고 발표용 PC의 환경에 맞춰 파이썬 가상환경을 구축하고 필요한 패키지들을 자동 설치해주는 스크립트 모음을 완비하였습니다.

---

## 2. demo/ 폴더 구조 및 파일 명세
```text
demo/
├── requirements.txt      # 데모 구동에 필수적인 패키지 명세 (Pandas, PyTorch, FastAPI 등)
├── setup_env.sh          # 발표 PC에서 최초 1회 실행하는 가상환경 생성 및 패키지 설치 스크립트 [NEW]
├── run_server.sh         # FastAPI 추론 API 서버 구동 스크립트 [NEW]
├── run_client.sh         # 모의 로그 시퀀스 6개 추론 요청 시연 스크립트 [NEW]
├── src/                  # 백엔드 핵심 알고리즘 소스 코드 폴더 [COPY]
│   ├── classifier.py     # LSTM 딥러닝 다중 분류 모델 구조 정의
│   ├── detector.py       # Isolation Forest 비지도 이상 탐지 래퍼
│   ├── parser.py         # Regex 로그 정제 및 TF-IDF 벡터화 모듈
│   ├── main.py           # FastAPI 실시간 추론 컨트롤러
│   └── test_client.py    # HTTP POST 예측 요청 테스트 클라이언트
└── models/               # 로컬 fallback용 오프라인 학습 완료 가중치 폴더 [COPY]
    ├── vectorizer.pkl    # TF-IDF 피처 추출 학습 객체
    ├── iso_forest_optimized.pkl # Isolation Forest 비지도 이상 탐지 가중치
    └── lstm_model.pth    # LSTM 딥러닝 다중 분류 가중치
```

---

## 3. 발표용 PC에서의 데모 가동 프로세스 (전달용 가이드)

### [Step 1] 의존성 자동 빌드 (최초 1회)
발표용 PC로 `demo/` 폴더를 이동한 뒤, 터미널에서 해당 폴더 경로로 진입해 다음 명령을 실행합니다:
```bash
./setup_env.sh
```
*   **역할**: 파이썬 가상환경 `.venv`를 생성하고, `requirements.txt`에 지정된 PyTorch, FastAPI 등의 필수 패키지를 로컬 환경 간섭 없이 가상환경 내에 깨끗하게 자동 설치합니다.

### [Step 2] API 서버 가동
새로운 터미널 탭에서 다음 스크립트를 실행하여 FastAPI 백엔드 서버를 켭니다:
```bash
./run_server.sh
```
*   **역할**: 가상환경을 자동으로 소싱한 뒤 Uvicorn 엔진을 이용해 실시간 로그 추론 API를 8000포트로 대기 상태로 진입시킵니다.
*   **라이브 확인**: 브라우저에서 `http://127.0.0.1:8000/docs`에 접속하여 정상 기동을 증명할 수 있습니다.

### [Step 3] 모의 시연 클라이언트 실행
서버가 켜진 상태에서 또 다른 터미널 탭을 열어 시연 스크립트를 작동시킵니다:
```bash
./run_client.sh
```
*   **역할**: 테스트 로그 시퀀스 6개를 API 서버로 밀어 넣고, `정상/이상 여부`, `장애 등급(ALERT/FATAL/WARNING)`, `예측 신뢰도(Confidence)` 결과가 콘솔에 정형화되어 실시간 출력되는 모습을 시연합니다.
