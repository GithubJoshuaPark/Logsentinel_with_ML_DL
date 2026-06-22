import re
import joblib
import numpy as np
from typing import Iterable
from sklearn.feature_extraction.text import TfidfVectorizer

class LogParser:
    def __init__(self, max_features: int = 500):
        """LogParser 클래스 생성자.
        
        Args:
            max_features (int): 추출할 핵심 단어의 가중치 높은 상위 개수.
                피처 행렬의 차원이 너무 커져 발생하는 오버핏(Overfitting)을 방지하고 연산 속도를 조절합니다.
        
        Attributes:
            max_features (int): 피처 어휘 제한 크기.
            vectorizer (TfidfVectorizer): Scikit-learn의 TF-IDF 변환기 객체.
            patterns (list): 가변적인 텍스트 노이즈를 식별하기 위해 순서대로 정의된 (정규식, 치환태그) 매핑 목록.
                - [IP]: IPv4 규격의 IP 주소 매칭.
                - [HEX]: 0x로 시작하는 16진수 주소값 매칭.
                - [NODE]: BGL 슈퍼컴퓨터 로그 고유의 물리 노드 및 컴포넌트 정보 매칭.
                - [DATETIME] & [DATE]: 날짜 및 시각 포맷 매칭.
                - [NUM]: 일반 정수형 숫자 정보 매칭.
        """
        self.max_features = max_features
        self.vectorizer = TfidfVectorizer(max_features=max_features)
        
        # 정규식 패턴 정의
        self.patterns = [
            (r'\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b', ' [IP] '),
            (r'0x[0-9a-fA-F]+', ' [HEX] '),
            (r'R\d{2}-M\d-N[0-9a-zA-Z]+-C:J\d{2}-U\d{2}|R\d{2}-M\d-N\w+', ' [NODE] '),
            (r'\d{4}-\d{2}-\d{2}-\d{2}\.\d{2}\.\d{2}\.\d+', ' [DATETIME] '),
            (r'\d{4}\.\d{2}\.\d{2}', ' [DATE] '),
            (r'\b\d+\b', ' [NUM] '),
        ]

    def clean_log(self, log_line: str) -> str:
        """원본 로그 메시지에서 가변적인 노이즈 정보를 감추고 텍스트를 정제합니다.
        
        양 끝 공백을 제거한 후, 정의된 self.patterns 정규식을 순회 적용하여
        IP, 16진수 주소, 특정 노드 정보, 숫자, 시간 정보를 대응하는 특수 태그([IP], [NUM] 등)로 치환합니다.
        이후 연속된 다중 공백을 단일 공백으로 치환하고 소문자화하여 단어의 매칭 성능을 극대화합니다.
        
        Args:
            log_line (str): 정제할 원본 로그 텍스트 문자열.
            
        Returns:
            str: 노이즈 정보가 치환 및 표준화된 정제 로그 문자열.
        """
        cleaned = log_line.strip()
        for pattern, replacement in self.patterns:
            cleaned = re.sub(pattern, replacement, cleaned)
        
        # 연속된 공백 제거 및 소문자화 (TF-IDF 매칭 극대화)
        cleaned = re.sub(r'\s+', ' ', cleaned).strip().lower()
        return cleaned

    def fit_transform(self, cleaned_logs: Iterable[str]) -> np.ndarray:
        """훈련용 정제 로그를 입력받아 TF-IDF 어휘 사전을 구축하고 2차원 수치 행렬로 변환합니다.
        
        입력 로그 전체에 대해 중요도가 높은 핵심 어휘 토큰 분포를 추출/피팅하고(Fit),
        이를 기초로 로그들을 500차원의 수치형 벡터(Transform)로 변환합니다.
        Scikit-learn의 기본 희소 행렬(Sparse Matrix) 대신 연산에 편리한 NumPy Dense Matrix 형태로 반환합니다.
        
        Args:
            cleaned_logs (Iterable[str]): 전처리 및 정제가 완료된 로그 메시지 리스트.
            
        Returns:
            np.ndarray: (전체 로그 수, max_features) 크기를 지닌 실수형 2차원 행렬.
        """
        return self.vectorizer.fit_transform(cleaned_logs).toarray()

    def transform(self, cleaned_logs: Iterable[str]) -> np.ndarray:
        """기존에 구축 완료된 TF-IDF 가중치를 적용하여 신규 정제 로그를 수치 벡터로 변환합니다.
        
        어휘 사전 및 가중치 스키마 설정을 고정한 채 변환을 진행하므로, 실시간 이상 탐지나
        평가 검증 단계에서 훈련 기준과 완전히 일치하는 일관성 있는 수치화 처리를 보장합니다.
        
        Args:
            cleaned_logs (Iterable[str]): 전처리가 완료된 신규 정제 로그 메시지 리스트.
            
        Returns:
            np.ndarray: (신규 로그 수, max_features) 크기를 지닌 실수형 2차원 행렬.
        """
        return self.vectorizer.transform(cleaned_logs).toarray()

    def save_vectorizer(self, path: str):
        """학습(Fit) 완료된 TfidfVectorizer 가중치 모델 객체를 지정된 경로에 직렬화 저장합니다.
        
        Args:
            path (str): 저장할 파일의 절대 혹은 상대 경로 (예: 'models/vectorizer.pkl').
        """
        joblib.dump(self.vectorizer, path)

    def load_vectorizer(self, path: str):
        """디스크에 저장되어 있던 벡터라이저 가중치 모델을 읽어와 현재 인스턴스에 복원(역직렬화)합니다.
        
        이를 통해 훈련 단계와 평가/추론 단계 간 피처 매핑 및 가중치의 완벽한 일관성을 보장합니다.
        
        Args:
            path (str): 읽어올 파일의 절대 혹은 상대 경로.
        """
        self.vectorizer = joblib.load(path)
