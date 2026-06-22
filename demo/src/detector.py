import joblib
import numpy as np
from sklearn.ensemble import IsolationForest


class AnomalyDetector:
    """Isolation Forest 모델을 활용한 비지도 학습 로그 이상 탐지(Anomaly Detection) 클래스입니다.
    
    벡터화된 로그 데이터를 학습하여 정상 패턴 분포를 구성하고,
    정상 분포 영역에서 멀리 벗어난 로그를 이상치(Anomaly)로 식별합니다.
    """
    
    def __init__(self, contamination: float = 0.001):
        """AnomalyDetector 클래스 생성자.
        
        Args:
            contamination (float): 데이터셋 내 예상 이상치 비율(오염도). 
                0.0에서 0.5 사이의 실수값으로 지정하며, 학습 및 이상치 임계값 설정의 기준이 됩니다.
                기본값은 0.001 (0.1%) 입니다.
        
        Attributes:
            contamination (float): 이상치 설정 비율.
            model (IsolationForest): Scikit-learn의 Isolation Forest 이상 탐지 모델 객체.
        """
        self.contamination = contamination
        self.model = IsolationForest(
            contamination=self.contamination, 
            n_estimators=100, 
            random_state=42,
            n_jobs=-1
        )

    def train(self, X_train):
        """벡터화된 로그 데이터를 학습 데이터로 삼아 Isolation Forest 모델을 훈련(Fit)합니다.
        
        Args:
            X_train (array-like): (샘플 수, 피처 수) 크기의 훈련용 수치 행렬 (예: TF-IDF 매트릭스).
        """
        self.model.fit(X_train)

    def predict(self, X) -> np.ndarray:
        """입력된 데이터의 이상 여부를 판별하여 결과를 반환합니다.
        
        Isolation Forest 예측 모델의 출력에 따라 정상 샘플은 1,
        비정상(이상치) 샘플은 -1로 라벨링하여 반환합니다.
        
        Args:
            X (array-like): (테스트 샘플 수, 피처 수) 크기의 검증 대상 수치 행렬.
            
        Returns:
            np.ndarray: 예측 결과 배열 (정상: 1, 이상치: -1).
            
        Raises:
            ValueError: 모델이 아직 훈련(`train`)되거나 파일에서 로드되지 않았을 경우 발생합니다.
        """
        if self.model is None:
            raise ValueError("Model has not been trained or loaded yet.")
        return self.model.predict(X)

    def get_anomaly_scores(self, X) -> np.ndarray:
        """각 로그 데이터의 이상치 원본 점수(Decision Function Score)를 반환합니다.
        
        점수가 낮을수록(음수 방향) 정상 데이터 분포에서 크게 벗어난
        이상치(Anomaly)일 확률이 높음을 뜻합니다.
        
        Args:
            X (array-like): (샘플 수, 피처 수) 크기의 타겟 수치 행렬.
            
        Returns:
            np.ndarray: 각 샘플에 대응하는 이상치 점수 배열.
            
        Raises:
            ValueError: 모델이 아직 훈련되거나 로드되지 않았을 경우 발생합니다.
        """
        if self.model is None:
            raise ValueError("Model has not been trained or loaded yet.")
        return self.model.decision_function(X)

    def save_model(self, path: str):
        """훈련된 Isolation Forest 모델 객체를 디스크에 바이너리 파일로 저장(직렬화)합니다.
        
        Args:
            path (str): 저장할 파일 경로 (예: 'models/detector.pkl').
        """
        joblib.dump(self.model, path)

    def load_model(self, path: str):
        """디스크에 저장되어 있던 Isolation Forest 모델 파일을 불러와 현재 인스턴스에 복원(역직렬화)합니다.
        
        Args:
            path (str): 읽어올 직렬화 파일 경로.
        """
        self.model = joblib.load(path)
