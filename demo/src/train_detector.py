import os

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix

from src.detector import AnomalyDetector
from src.parser import LogParser


def load_data(file_path):
    print(f"[LogSentinel] Loading log dataset from {file_path}...")
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
    print(f"[LogSentinel] Total loaded rows: {len(df)}")
    print(f"[LogSentinel] Class distribution:\n{df['is_anomaly'].value_counts()}")
    print(f"[LogSentinel] Label distribution:\n{df['label'].value_counts()}")
    return df

def train_and_evaluate():
    log_file = "./data/raw/BGL_balanced.log"
    if not os.path.exists(log_file):
        # fallback to 2k log
        log_file = "./data/raw/BGL_2k.log"
        print(f"[Warning] Balanced log not found. Falling back to {log_file}")
        
    df = load_data(log_file)
    
    # 1. LogParser 초기화 및 vectorizer 로드
    parser = LogParser(max_features=500)
    vec_path = "./models/vectorizer.pkl"
    
    # 텍스트 전처리
    print("[LogSentinel] Cleaning messages...")
    df["cleaned_message"] = df["raw_message"].apply(parser.clean_log)
    
    if os.path.exists(vec_path):
        print(f"[LogSentinel] Loading existing vectorizer from {vec_path}...")
        parser.load_vectorizer(vec_path)
        X = parser.transform(df["cleaned_message"])
    else:
        print("[LogSentinel] Vectorizer file not found. Fitting a new one...")
        X = parser.fit_transform(df["cleaned_message"])
        parser.save_vectorizer(vec_path)
        
    y = df["is_anomaly"].values
    
    # 2. Contamination 비율 조사
    # 이전에 미탐이 되었던 KERNSTOR, KERNDTLB 등 구체적 에러 타입별 탐지율을 파악하기 위한 준비
    error_types = df[df["is_anomaly"] == 1]["label"].unique()
    print(f"[LogSentinel] Distinct anomaly error types in dataset: {error_types}")
    
    contamination_ratios = [0.001, 0.01, 0.05, 0.1, 0.15, 0.20, 0.25, 0.30]
    best_f1 = 0
    best_c = 0.1
    best_detector = None
    
    print("\n" + "="*60)
    print("Contamination Scan Analysis")
    print("="*60)
    
    for c in contamination_ratios:
        detector = AnomalyDetector(contamination=c)
        detector.train(X)
        preds = detector.predict(X) # 1: 정상, -1: 이상
        pred_anomaly = (preds == -1).astype(int)
        
        tn, fp, fn, tp = confusion_matrix(y, pred_anomaly).ravel()
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        print(f"\n[Contamination: {c}]")
        print(f"  Confusion Matrix: TN={tn}, FP={fp}, FN={fn}, TP={tp}")
        print(f"  Precision: {precision:.4f} | Recall: {recall:.4f} | F1-Score: {f1:.4f}")
        
        # 주요 에러 타입별 검출률(Recall) 측정
        print("  Error Type Detection Rate (Recall by Type):")
        for err in error_types:
            err_mask = (df["label"] == err).values
            err_preds = pred_anomaly[err_mask]
            err_tp = np.sum(err_preds == 1)
            err_total = np.sum(err_mask)
            err_recall = err_tp / err_total if err_total > 0 else 0
            print(f"    - {err:10}: {err_recall * 100:6.2f}% ({err_tp}/{err_total})")
            
        if f1 > best_f1:
            best_f1 = f1
            best_c = c
            best_detector = detector
            
    print("\n" + "="*60)
    print(f"Best Contamination based on F1-score: {best_c} (F1: {best_f1:.4f})")
    print("="*60)
    
    # 최적 모델 저장
    if best_detector is not None:
        save_path = "./models/iso_forest.pkl"
        best_detector.save_model(save_path)
        print(f"[LogSentinel] Saved best anomaly detector model to {save_path}")

if __name__ == "__main__":
    train_and_evaluate()
