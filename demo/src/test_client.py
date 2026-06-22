import urllib.request
import json

def test_api():
    """FastAPI 로그 분석 서버의 실시간 추론 API(/predict)에 가상의 로그 시퀀스를 보내 동작을 검증합니다.
    
    이 함수는 정상 로그와 다양한 유형의 장애 로그(APPSEV, KERNSTOR, APPREAD 등)가 혼합된
    테스트 로그 시퀀스를 JSON 포맷으로 패킹하여 API 서버에 HTTP POST 요청으로 전달합니다.
    서버로부터 수신된 결과에서 각 로그별 이상 여부(is_anomaly), 장애 유형(classification), 
    그리고 분류 신뢰도(confidence)를 가독성 있게 콘솔에 출력합니다.
    """
    url = "http://127.0.0.1:8000/predict"
    
    # 테스트할 로그 시퀀스들
    test_logs = [
        # 1. 정상 로그 (INFO)
        "- 1117975659 2005.06.05 R26-M0-N6-C:J12-U01 2005-06-05-05.47.39.638358 R26-M0-N6-C:J12-U01 RAS KERNEL INFO generating core.1573",
        "- 1133448654 2005.12.01 R43-M1-N3-C:J12-U01 2005-12-01-06.50.54.563527 R43-M1-N3-C:J12-U01 RAS KERNEL INFO 10752 total interrupts.",
        
        # 2. 장애 로그 (FATAL) - APPSEV
        "APPSEV 1126798120 2005.09.15 R15-M0-NC-I:J18-U01 2005-09-15-08.28.40.548048 R15-M0-NC-I:J18-U01 RAS APP FATAL ciod: Error reading message prefix after LOAD_MESSAGE on CioStream socket to 172.16.96.116:37502: Link has been severed",
        
        # 3. 정상 로그 (INFO)
        "- 1121311419 2005.07.13 R24-M0-N0-C:J09-U11 2005-07-13-20.23.39.919832 R24-M0-N0-C:J09-U11 RAS KERNEL INFO generating core.15614",
        
        # 4. 장애 로그 (FATAL) - KERNSTOR
        "KERNSTOR 1118709808 2005.06.13 R15-M0-N4-C:J04-U11 2005-06-13-17.43.28.900807 R15-M0-N4-C:J04-U11 RAS KERNEL FATAL data storage interrupt",
        
        # 5. 장애 로그 (FATAL) - APPREAD
        "APPREAD 1118959387 2005.06.16 R35-M1-N4-I:J18-U11 2005-06-16-15.03.07.394876 R35-M1-N4-I:J18-U11 RAS APP FATAL ciod: failed to read message prefix on control stream (CioStream socket to 172.16.96.116:52329"
    ]
    
    payload = {"logs": test_logs}
    data = json.dumps(payload).encode("utf-8")
    
    req = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST"
    )
    
    print("[LogSentinel] Sending logs to API Server...")
    try:
        with urllib.request.urlopen(req) as res:
            response_body = res.read().decode("utf-8")
            results = json.loads(response_body)["results"]
            
            print("\n=== Real-time Inference Results ===")
            for idx, result in enumerate(results, 1):
                log_snippet = result["log"][:60] + "..." if len(result["log"]) > 60 else result["log"]
                print(f"[{idx}] Log: {log_snippet}")
                print(f"    - Is Anomaly: {result['is_anomaly']}")
                print(f"    - Classification: {result['classification']}")
                print(f"    - Confidence: {result['confidence']:.4f}\n")
    except Exception as e:
        print(f"[Error] Failed to communicate with API server: {e}")

if __name__ == "__main__":
    test_api()
