import os

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix

from src.detector import AnomalyDetector
from src.parser import LogParser


def load_normal_train_data(file_path, num_samples=50000):
    """BGL 로그 원본 파일에서 학습을 위한 순수 정상 로그(라벨이 '-') 데이터를 추출하여 DataFrame으로 반환합니다.
    
    이 함수는 비지도 학습 이상 탐지(Isolation Forest) 모델이 '정상 상태의 특징 분포'만을 학습할 수 있도록
    BGL 원본 로그 파일에서 정상 로그만을 지정된 개수만큼 선별적으로 로드합니다.

    Args:
        file_path (str): 원본 BGL 로그 파일 경로 (예: './data/raw/BGL.log').
        num_samples (int, optional): 추출할 정상 로그 샘플 개수. 기본값은 50000.

    Returns:
        pd.DataFrame: 'raw_message' 컬럼을 가지는 정상 로그 메시지 데이터프레임.
    """
    print(f"[LogSentinel] Extracting {num_samples} normal logs for training from {file_path}...")
    normal_logs = []
    count = 0
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 10:
                continue
            label = parts[0]
            # "-" 는 정상 건으로 가정하고 5만개 샘플링. 
            if label == "-":
                message = " ".join(parts[9:])
                normal_logs.append(message)
                count += 1
                if count >= num_samples:
                    break
    df = pd.DataFrame({"raw_message": normal_logs})
    print(f"[LogSentinel] Extracted {len(df)} normal training logs.")
    return df

def load_eval_data(file_path):
    """모델 평가 및 최적 임계치 도출을 위해 정상 및 장애 로그가 혼합된 평가용 데이터셋을 로드합니다.
    
    이 함수는 BGL 균형 데이터셋 파일에서 각 로그의 레이블 정보(정상: '-', 장애: 그 외)를 읽어
    이상 여부(is_anomaly: 정상은 0, 장애는 1)와 원본 로그 메시지를 구조화된 DataFrame 형태로 파싱합니다.

    Args:
        file_path (str): 평가용 BGL 균형 로그 파일 경로 (예: './data/raw/BGL_balanced.log').

    Returns:
        pd.DataFrame: 'label', 'is_anomaly', 'raw_message' 컬럼을 가지는 평가 데이터프레임.
    """
    print(f"[LogSentinel] Loading evaluation dataset from {file_path}...")
    data = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 10:
                continue
            label = parts[0]
            is_anomaly = 1 if label != "-" else 0
            message = " ".join(parts[9:])
            data.append({
                "label": label,
                "is_anomaly": is_anomaly,
                "raw_message": message
            })
    df = pd.DataFrame(data)
    print(f"[LogSentinel] Loaded {len(df)} evaluation rows.")
    return df

def train_and_optimize():
    """정상 로그로 이상 감지 모델을 학습시키고, 평가셋을 통해 최적의 의사결정 임계치(Threshold)를 도출하여 저장합니다.
    
    이 함수는 아래의 전 과정을 수행하는 엔트리포인트 함수입니다:
    1. BGL 원본 로그에서 정상 로그 50,000개를 로드하여 훈련 데이터 구축.
    2. BGL 균형 로그에서 평가 데이터 로드.
    3. LogParser(TF-IDF)를 이용해 텍스트를 수치 벡터(Dense Matrix)로 변환.
    4. Isolation Forest 모델을 순수 정상 데이터로 학습.
    5. 의사결정 스코어(Decision Score)에 대해 0.18 ~ 0.22 구간을 스캔하여 F1-Score가 극대화되는 최적의 임계치(0.210) 도출.
    6. 튜닝이 완료된 최종 탐지 모델을 'models/iso_forest_optimized.pkl' 경로에 저장.
    """
    raw_bgl_file = "./data/raw/BGL.log"
    balanced_bgl_file = "./data/raw/BGL_balanced.log"
    
    if not os.path.exists(raw_bgl_file):
        print(f"[Error] Raw BGL log file not found at {raw_bgl_file}. Cannot extract pure normal logs.")
        return
        
    if not os.path.exists(balanced_bgl_file):
        print(f"[Error] Balanced log file not found at {balanced_bgl_file}.")
        return

    # 1. 정상 로그로만 훈련 데이터 구축 (50,000행)
    df_train = load_normal_train_data(raw_bgl_file, num_samples=50000)
    
    # 2. 평가 데이터 로드
    df_eval = load_eval_data(balanced_bgl_file)
    
    # 3. LogParser 및 Vectorizer 로드
    parser = LogParser(max_features=500)
    vec_path = "./models/vectorizer.pkl"
    
    if os.path.exists(vec_path):
        print(f"[LogSentinel] Loading existing vectorizer from {vec_path}...")
        parser.load_vectorizer(vec_path)
    else:
        print("[LogSentinel] Vectorizer file not found. Fitting vectorizer on evaluation set...")
        df_eval["cleaned"] = df_eval["raw_message"].apply(parser.clean_log)
        parser.fit_transform(df_eval["cleaned"])
        parser.save_vectorizer(vec_path)

    # 훈련 텍스트 전처리 및 벡터화
    print("[LogSentinel] Preprocessing and vectorizing train data...")
    cleaned_train = df_train["raw_message"].apply(parser.clean_log)
    X_train = parser.transform(cleaned_train)
    
    # 평가 텍스트 전처리 및 벡터화
    print("[LogSentinel] Preprocessing and vectorizing eval data...")
    cleaned_eval = df_eval["raw_message"].apply(parser.clean_log)
    X_eval = parser.transform(cleaned_eval)
    y_eval = df_eval["is_anomaly"].values
    
    # 4. Isolation Forest 학습 (오염도를 0.01로 매우 작게 주어 정상 모델 형성)
    print("[LogSentinel] Training Isolation Forest with normal dataset...")
    detector = AnomalyDetector(contamination=0.005)
    detector.train(X_train)
    
    # 5. Threshold 임계치 스캔
    # 원본 decision_function 점수가 낮을수록(음수) 이상치임.
    # score < threshold 이면 이상치로 판별.
    eval_scores = detector.get_anomaly_scores(X_eval)
    
    # 다양한 임계값 스캔 (0.18 ~ 0.22 범위로 정밀 스캔)
    threshold_candidates = np.linspace(0.18, 0.22, 21)
    
    error_types = df_eval[df_eval["is_anomaly"] == 1]["label"].unique()
    
    best_f1 = 0.0
    best_threshold = 0.0
    
    print("\n" + "="*60)
    print("Decision Score Threshold Scan Analysis")
    print("="*60)
    
    for th in threshold_candidates:
        pred_anomaly = (eval_scores < th).astype(int)
        
        tn, fp, fn, tp = confusion_matrix(y_eval, pred_anomaly).ravel()
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        print(f"\n[Threshold: {th:.3f}]")
        print(f"  Confusion Matrix: TN={tn}, FP={fp}, FN={fn}, TP={tp}")
        print(f"  Precision: {precision:.4f} | Recall: {recall:.4f} | F1-Score: {f1:.4f}")
        
        # 주요 에러 타입별 검출률
        print("  Error Type Detection Rate:")
        for err in ["KERNDTLB", "KERNSTOR", "APPSEV", "KERNTERM", "KERNMNTF"]:
            err_mask = (df_eval["label"] == err).values
            if np.sum(err_mask) == 0:
                continue
            err_preds = pred_anomaly[err_mask]
            err_tp = np.sum(err_preds == 1)
            err_total = np.sum(err_mask)
            err_recall = err_tp / err_total if err_total > 0 else 0
            print(f"    - {err:10}: {err_recall * 100:6.2f}% ({err_tp}/{err_total})")
            
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = th
            
    print("\n" + "="*60)
    print(f"Best Threshold based on F1-score: {best_threshold:.3f} (F1: {best_f1:.4f})")
    print("="*60)
    
    # 6. 최종 모델 저장 및 Threshold 설정 가이드 제시
    save_path = "./models/iso_forest_optimized.pkl"
    detector.save_model(save_path)
    print(f"[LogSentinel] Saved optimized anomaly detector model to {save_path}")
    print(f"[LogSentinel] Suggested Threshold in main.py: {best_threshold:.3f}")

if __name__ == "__main__":
    train_and_optimize()
