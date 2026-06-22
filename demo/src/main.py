import os
from collections import deque
from typing import List

import numpy as np
import torch
import torch.nn.functional as F
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.classifier import LSTMClassifier
from src.detector import AnomalyDetector
from src.parser import LogParser

app = FastAPI(title="LogSentinel", version="1.0.0")

# 경로 설정
SSD_PATH = "/Volumes/T7/LogSentinel_Data/models"
LOCAL_PATH = "./models"

# 경로 탐색 헬퍼 (T7 SSD 우선, 없으면 로컬 fallback)
def get_model_path(filename):
    """지정한 모델 파일의 사용 가능한 물리적 경로를 탐색하여 반환합니다.
    
    외장 T7 SSD 경로를 우선적으로 확인하고, 파일이 존재하지 않는 경우
    로컬 './models/' 디렉토리에서 Fallback 경로를 찾아 반환합니다.

    Args:
        filename (str): 검색할 모델 파일명 (예: 'vectorizer.pkl').

    Returns:
        str: 실제 파일이 존재하거나 기본 참조될 모델 파일의 절대 혹은 상대 경로.
    """
    ssd_file = os.path.join(SSD_PATH, filename)
    local_file = os.path.join(LOCAL_PATH, filename)
    if os.path.exists(ssd_file):
        return ssd_file
    return local_file

# 글로벌 모델 및 객체 초기화
parser = None
detector = None
classifier = None
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 실시간 시퀀스 캐시 버퍼 (최근 5개 로그의 벡터 유지)
SEQ_LEN = 5
INPUT_DIM = 500
HIDDEN_DIM = 64
NUM_CLASSES = 3  # ALERT, FATAL, WARNING
log_vector_buffer = deque(maxlen=SEQ_LEN)

class LogRequest(BaseModel):
    logs: List[str] = Field(
        ...,
        description="BGL 표준 포맷의 시스템 로그 메시지 리스트입니다. 정상(INFO) 및 장애(FATAL/ERROR) 로그를 기입할 수 있습니다.",
        json_schema_extra={
            "example": [
                "- 1117975659 2005.06.05 R26-M0-N6-C:J12-U01 2005-06-05-05.47.39.638358 R26-M0-N6-C:J12-U01 RAS KERNEL INFO generating core.1573",
                "APPSEV 1126798120 2005.09.15 R15-M0-NC-I:J18-U01 2005-09-15-08.28.40.548048 R15-M0-NC-I:J18-U01 RAS APP FATAL ciod: Error reading message prefix after LOAD_MESSAGE on CioStream socket to 172.16.96.116:37502: Link has been severed"
            ]
        }
    )

class LogPrediction(BaseModel):
    log: str
    is_anomaly: bool
    classification: str  # NORMAL, ALERT, FATAL, WARNING
    confidence: float

class LogResponse(BaseModel):
    results: List[LogPrediction]

@app.on_event("startup")
def load_models():
    """FastAPI 애플리케이션 시작 시점에 머신러닝/딥러닝 모델들을 메모리에 적재 및 인스턴스화합니다.
    
    1. LogParser(TF-IDF Vectorizer) 로드
    2. Anomaly Detector(Isolation Forest) 최적 격리 모델 로드
    3. LSTM Classifier(장애 다중 분류 PyTorch 신경망) 모델 가중치 로드
    4. 실시간 맥락 유지를 위한 시퀀스 버퍼(deque)를 0 벡터로 패딩 초기화
    """
    global parser, detector, classifier
    
    # 1. Parser & Vectorizer 로드
    parser = LogParser(max_features=INPUT_DIM)
    vec_path = get_model_path("vectorizer.pkl")
    if os.path.exists(vec_path):
        parser.load_vectorizer(vec_path)
        print(f"[LogSentinel] Vectorizer loaded from {vec_path}")
    else:
        print("[LogSentinel] Warning: Vectorizer file not found. Run training first.")

    # 2. Anomaly Detector 로드
    detector = AnomalyDetector()
    det_path = get_model_path("iso_forest_optimized.pkl")
    if os.path.exists(det_path):
        detector.load_model(det_path)
        print(f"[LogSentinel] Anomaly Detector loaded from {det_path}")
    else:
        print("[LogSentinel] Warning: Anomaly Detector file not found. Run training first.")

    # 3. LSTM Classifier 로드
    classifier = LSTMClassifier(input_dim=INPUT_DIM, hidden_dim=HIDDEN_DIM, num_classes=NUM_CLASSES)
    cls_path = get_model_path("lstm_model.pth")
    if os.path.exists(cls_path):
        classifier.load_state_dict(torch.load(cls_path, map_location=device))
        classifier.to(device)
        classifier.eval()
        print(f"[LogSentinel] LSTM Classifier loaded from {cls_path}")
    else:
        print("[LogSentinel] Warning: LSTM Classifier file not found. Run training first.")

    # 초기 버퍼를 0 벡터로 패딩
    for _ in range(SEQ_LEN):
        log_vector_buffer.append(np.zeros(INPUT_DIM))

@app.post(
    "/predict", 
    response_model=LogResponse,
    summary="실시간 로그 장애 예측 파이프라인 API",
    description="""
    수신된 실시간 BGL 원시 로그 스트림을 차례로 정제, 이상 탐지(ML), 다중 장애 등급 분류(DL) 순으로 관통 추론합니다.
    
    **💡 데모 테스트 팁 (BGL 로그 템플릿 복사처):**
    인풋에 대입할 만한 로그 형식을 찾고 계신다면, 아래 LogPAI BGL 벤치마크 공개 데이터 파일에서 임의의 한 줄을 드래그해 복사한 후 `logs` 배열에 붙여넣어 실행(Execute)해 보세요.
    - **BGL 2,000 라인 샘플 파일**: [LogHub BGL_2k.log 파일 링크](https://github.com/logpai/loghub/blob/master/BGL/BGL_2k.log)
    """
)
async def predict_logs(request: LogRequest):
    """수신된 실시간 로그 리스트를 받아 정제, 이상 탐지, 장애 레벨 다중 분류 파이프라인을 관통 추론합니다.
    
    1. 전처리: 로그 텍스트에서 IP, 시간, 노드 등의 노이즈를 일반화 정제 후 TF-IDF 벡터로 변환합니다.
    2. 이상 탐지 (ML): Isolation Forest 스코어가 임계치(0.210) 미만일 경우 이상치(Anomaly)로 판정합니다.
    3. 에러 분류 (DL): 이상치로 판정된 경우, 버퍼의 시퀀스 맥락(5개 로그)을 PyTorch LSTM에 공급하여
       장애 등급(ALERT, FATAL, WARNING) 및 추론 신뢰도(Confidence)를 예측합니다.
    4. 결과를 LogResponse JSON 규격으로 반환합니다.

    Args:
        request (LogRequest): 검증 대상 원본 로그 메시지 리스트를 포함하는 Pydantic 모델 객체.

    Returns:
        LogResponse: 이상 탐지 여부, 예측 장애 유형, 신뢰도 매핑 결과 리스트를 담은 Pydantic 모델 객체.
    """
    global parser, detector, classifier
    if parser is None or detector is None or classifier is None:
        raise HTTPException(status_code=503, detail="Models are not initialized. Check server logs.")

    results = []
    class_labels = {0: "ALERT", 1: "FATAL", 2: "WARNING"}

    for log in request.logs:
        # 1단계: 정제 및 TF-IDF 벡터화 (BGL 원본 포맷 대응)
        parts = log.strip().split()
        if len(parts) >= 10:
            msg = " ".join(parts[9:])
        else:
            msg = log
        cleaned = parser.clean_log(msg)
        vector = parser.transform([cleaned])[0] # shape: (500,)
        
        # 2단계: Isolation Forest를 통한 이상 탐지
        # scikit-learn decision_function 점수가 최적 임계값(0.210)보다 작으면 이상치로 판단
        score = detector.get_anomaly_scores([vector])[0]
        THRESHOLD = 0.210
        is_anomaly = bool(score < THRESHOLD)
        print(f"[추론 디버그] msg: '{msg}' | clean: '{cleaned}' | score: {score:.6f} | threshold: {THRESHOLD:.3f} | is_anomaly: {is_anomaly}")

        # 정상 로그이든 이상 로그이든 상관없이 시퀀스 맥락 유지를 위해 버퍼에 인입
        log_vector_buffer.append(vector)

        if not is_anomaly:
            # 정상 로그
            results.append(LogPrediction(
                log=log,
                is_anomaly=False,
                classification="NORMAL",
                confidence=1.0
            ))
        else:
            # 3단계: 이상 로그인 경우 LSTM을 통한 장애 등급 다중 분류
            # 최근 5개의 로그 벡터를 가져와 시퀀스 Tensor 구성 -> shape: (1, 5, 500)
            seq_data = np.array(list(log_vector_buffer))
            seq_tensor = torch.tensor(seq_data, dtype=torch.float32).unsqueeze(0).to(device)

            with torch.no_grad():
                outputs = classifier(seq_tensor) # shape: (1, 3)
                probabilities = F.softmax(outputs, dim=1)
                confidence, pred_idx = torch.max(probabilities, dim=1)
                
            pred_class_idx = pred_idx.item()
            pred_class_label = class_labels.get(pred_class_idx, "UNKNOWN")
            pred_conf = confidence.item()

            results.append(LogPrediction(
                log=log,
                is_anomaly=True,
                classification=pred_class_label,
                confidence=float(pred_conf)
            ))

    return LogResponse(results=results)

@app.get("/health")
async def health_check():
    """FastAPI 서버 가동 여부 및 핵심 3단계 추론 모델들의 메모리 정상 로드 상태를 체크합니다.

    Returns:
        dict: 서버 상태('status': 'healthy') 및 모델 적재 준비성 여부('models_ready': bool)가 매핑된 딕셔너리.
    """
    # 모델 로드 상태 포함하여 헬스체크 응답
    models_ready = parser is not None and detector is not None and classifier is not None
    return {
        "status": "healthy",
        "models_ready": models_ready
    }
