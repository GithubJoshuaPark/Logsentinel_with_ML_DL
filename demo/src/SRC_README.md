# LogSentinel 소스 코드 디렉토리(./src/) 구조 및 역할 정의

- **작성 일시**: 2026년 06월 19일 21시 59분 00초
- **작성자**: Antigravity AI
- **수신자**: Joshua (프로젝트 담당자)

---

## 1. 개요

프로젝트 루트 기준 `./src/` 디렉토리에 구성되어 있는 각 소스 코드 파일들의 논리적 의미와 담당 역할을 정의한 설명서입니다.

---

## 2. 소스 파일별 역할 명세

| 파일명                        | 파일 설명                        | 핵심 역할                                                 |
| :---------------------------- | :------------------------------- | :-------------------------------------------------------- |
| **`__init__.py`**             | 패키지 초기화 파일               | `./src` 디렉토리를 파이썬 모듈 패키지로 인식하게 함       |
| **`parser.py`**               | 로그 메시지 정제 및 벡터화       | 정규식 노이즈 제거 및 TF-IDF 기반 500차원 벡터 변환       |
| **`detector.py`**             | Isolation Forest 모델 인터페이스 | 비지도 학습 이상치 탐지 모델의 로드, 훈련, 스코어 계산    |
| **`classifier.py`**           | PyTorch LSTM 모델 정의           | 최근 5개 로그 시퀀스 입력을 통한 장애 등급 다중 분류      |
| **`main.py`**                 | FastAPI 웹 서빙 엔트리포인트     | API 시동 시 모델 적재 및 `/predict` 실시간 관통 추론 제어 |
| **`sample_generator.py`**     | 데이터 수집 및 전처리            | 원본 데이터 다운로드 및 학습용 균형 데이터셋 가공         |
| **`train_detector.py`**       | 초기 ML 분석/훈련 스크립트       | Isolation Forest의 contamination 하이퍼파라미터 스캔      |
| **`train_detector_fixed.py`** | 고도화 ML 분석/훈련 스크립트     | 정상 50k 기반 학습 및 최적 임계치(`0.210`) 도출           |
| **`test_client.py`**          | API 모의 테스트 클라이언트       | FastAPI 연계 관통 추론 실시간 성능 동작 검증 시뮬레이터   |

### 상세 파일별 핵심 역할 설명

1. **[**init**.py](file:///Users/joshuapark/Desktop/Dev/soromiso/LogSentinel/src/**init**.py)**
   - `./src` 디렉토리를 하나의 파이썬 패키지로 인식하게 하여 외부 및 타 모듈에서 모듈 참조를 손쉽게 할 수 있게 합니다.

2. **[parser.py](file:///Users/joshuapark/Desktop/Dev/soromiso/LogSentinel/src/parser.py)**
   - 로그의 정밀 정제를 총괄합니다. 정규표현식 매핑을 통해 로그 내부의 IP, HEX 주소, DateTime, Node명 등 가변적 노이즈를 치환 및 정제한 뒤, `TfidfVectorizer`를 이용해 500차원의 고성능 수치 벡터로 조밀하게 변환합니다.

3. **[detector.py](file:///Users/joshuapark/Desktop/Dev/soromiso/LogSentinel/src/detector.py)**
   - 비지도 학습인 `Isolation Forest` 모델 객체를 인스턴스화하고, 훈련(`fit`) 및 직렬화 저장/로드를 대행하는 인터페이스입니다. 각 로그 벡터의 이상치 고립 점수(`decision_function`)를 계산하여 전달하는 핵심 연산을 수행합니다.

4. **[classifier.py](file:///Users/joshuapark/Desktop/Dev/soromiso/LogSentinel/src/classifier.py)**
   - 이상 징후로 판단된 로그들에 대해 정밀 분류를 수행하는 PyTorch 기반 LSTM 다중 클래스 분류망입니다. 실시간으로 수입되는 최근 5개 로그 시퀀스를 시계열 데이터로 가공 및 전달받아 예측 분류(ALERT, FATAL, WARNING)를 수행합니다.

5. **[main.py](file:///Users/joshuapark/Desktop/Dev/soromiso/LogSentinel/src/main.py)**
   - 실시간 서빙의 중심 컨트롤러입니다. FastAPI 서버로서 동작하며, 시동 시 3대 핵심 추론 모듈을 메모리에 적재(`load_models`)하고, `/predict` 요청이 올 때마다 `텍스트 정제 ➔ 이상 탐지 ➔ LSTM 장애 예측` 순서로 관통 연계 추론을 제어합니다.

6. **[sample_generator.py](file:///Users/joshuapark/Desktop/Dev/soromiso/LogSentinel/src/sample_generator.py)**
   - 훈련에 쓰이는 대용량 BGL 데이터의 자동 다운로드(curl 우회 호출 기법 적용)를 진행하고, 극단적인 불균형 데이터 구조(정상 99% vs 장애 1% 미만)를 훈련 효율성에 적합하게 가중 샘플링하여 균형 데이터셋(`BGL_balanced.log`)을 가공하는 기초 유틸리티입니다.

7. **[train_detector.py](file:///Users/joshuapark/Desktop/Dev/soromiso/LogSentinel/src/train_detector.py)**
   - Isolation Forest 튜닝 분석에 쓰이는 스크립트로, contamination 변수를 변경해가며 혼동 행렬 지표를 모니터링하기 위해 구축된 1차 분석 파일입니다.

8. **[train_detector_fixed.py](file:///Users/joshuapark/Desktop/Dev/soromiso/LogSentinel/src/train_detector_fixed.py)**
   - 고도화 반영을 통해 완성된 Anomaly Detector의 정석적인 학습 스크립트입니다. 원본 `BGL.log`로부터 정상 로그 50,000행만 발췌하여 훈련을 진행하고, 혼합 평가 데이터 42.8만 행으로 의사결정 임계값을 정밀 탐색하여 `models/iso_forest_optimized.pkl`을 독립 생성합니다.

9. **[test_client.py](file:///Users/joshuapark/Desktop/Dev/soromiso/LogSentinel/src/test_client.py)**
   - 개발 및 튜닝 완료 후 FastAPI `/predict` 엔드포인트의 작동성 유효 검증을 위해, 실제 정상 및 장애 로그 패키지를 순차로 밀어넣고 예측 신뢰도와 분류 카테고리를 출력을 검증하는 테스트 시뮬레이터입니다.
