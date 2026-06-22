#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import random
import subprocess
import tarfile

def download_and_extract_bgl(local_dir: str):
    """curl 명령어를 사용하여 Zenodo에서 BGL.tar.gz를 다운로드하고 압축을 푼 뒤, 임시 tar.gz 파일을 삭제합니다."""
    tar_path = os.path.join(local_dir, "BGL.tar.gz")
    
    # 디렉토리 생성
    os.makedirs(local_dir, exist_ok=True)
    
    # Zenodo의 BGL.tar.gz API direct download URL
    url = "https://zenodo.org/api/records/3227177/files/BGL.tar.gz/content"
    print(f"[다운로드 시작] {url} ➔ {tar_path}")
    
    try:
        # curl -L -# -o <tar_path> <url> 실행 (진행 상태 바 노출)
        cmd = ["curl", "-L", "-#", "-o", tar_path, url]
        subprocess.run(cmd, check=True)
        print("[다운로드 완료]")
        
        print(f"[압축 해제 시작] {tar_path} ➔ {local_dir}")
        with tarfile.open(tar_path, "r:gz") as tar_ref:
            tar_ref.extractall(path=local_dir)
        print("[압축 해제 완료]")
        
    except Exception as e:
        print(f"\n[다운로드/해제 에러] {str(e)}", file=sys.stderr)
        if os.path.exists(tar_path):
            os.remove(tar_path)
        raise e
    finally:
        # 임시 tar.gz 파일 삭제
        if os.path.exists(tar_path):
            print(f"[임시 파일 삭제] {tar_path}")
            os.remove(tar_path)

def sample_bgl(source_path: str, target_path: str, normal_sample_size: int = 80000):
    """BGL 전체 로그에서 모든 장애/에러 로그를 가져오고, 정상 로그는 지정된 개수만큼 
    랜덤 샘플링하여 결합된 균형 데이터셋 로그 파일을 빌드합니다.
    """
    if not os.path.exists(source_path):
        raise FileNotFoundError(f"원본 BGL 로그 파일을 찾을 수 없습니다: {source_path}")
        
    print(f"[샘플링 시작] 원본: {source_path}")
    
    anomalies = []
    normal_pool = []
    
    # 대용량 파일을 메모리 효율적으로 한 줄씩 순차 스트리밍
    try:
        with open(source_path, "r", encoding="utf-8", errors="ignore") as f:
            for line_idx, line in enumerate(f):
                parts = line.strip().split()
                if not parts:
                    continue
                
                # BGL 로그 파일 형식상 첫 번째 공백 문자 이전의 컬럼이 라벨(Alert Indicator)입니다.
                # '-' 이면 정상, 그 외의 특정 영문 태그는 이상치를 가리킵니다.
                label = parts[0]
                if label != "-":
                    anomalies.append(line)
                else:
                    normal_pool.append(line)
                    
                if (line_idx + 1) % 1000000 == 0:
                    print(f".. {line_idx + 1} 행 읽는 중 ..")
                    
    except Exception as e:
        print(f"[샘플링 에러] 스트리밍 읽기 중 오류 발생: {str(e)}", file=sys.stderr)
        raise e
        
    print(f"[파싱 완료] 정상 로그 풀: {len(normal_pool)}개, 에러 로그 풀: {len(anomalies)}개")
    
    # 정상 로그 무작위 추출
    sample_size = min(normal_sample_size, len(normal_pool))
    print(f"[샘플링 추출] 정상 로그 풀에서 {sample_size}개 랜덤 샘플링 중...")
    sampled_normals = random.sample(normal_pool, sample_size)
    
    # 데이터 병합 및 학습 바이어스 방지를 위한 셔플
    final_dataset = anomalies + sampled_normals
    print("[샘플링 셔플] 균형 데이터셋 셔플 진행 중...")
    random.shuffle(final_dataset)
    
    # 최종 파일 출력
    os.makedirs(os.path.dirname(target_path), exist_ok=True)
    print(f"[샘플링 쓰기] 파일 생성 중 ➔ {target_path}")
    try:
        with open(target_path, "w", encoding="utf-8") as out:
            out.writelines(final_dataset)
        print(f"[샘플링 완료] 총 {len(final_dataset)}개 행 저장 완료 (정상: {len(sampled_normals)} / 장애: {len(anomalies)})")
    except Exception as e:
        print(f"[샘플링 에러] 파일 쓰기 중 오류 발생: {str(e)}", file=sys.stderr)
        raise e

if __name__ == "__main__":
    # T7 SSD 장착 여부에 따른 파일 경로 자동 분기 (방어 코딩)
    SSD_PATH = "/Volumes/T7/LogSentinel_Data/raw"
    LOCAL_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "raw"))
    
    if os.path.exists("/Volumes/T7"):
        source_dir = SSD_PATH
        target_dir = SSD_PATH
    else:
        source_dir = LOCAL_PATH
        target_dir = LOCAL_PATH
        
    SOURCE = os.path.join(source_dir, "BGL.log")
    TARGET = os.path.join(target_dir, "BGL_balanced.log")
    
    # 1. 최종 타겟 BGL_balanced.log 가 이미 존재하는지 확인
    if os.path.exists(TARGET):
        print(f"[안내] 이미 균형 데이터셋 파일이 존재합니다: {TARGET}")
        print("재생성을 원하시면 해당 파일을 삭제하고 다시 실행해 주세요.")
        sys.exit(0)
        
    # 2. 원본 BGL.log 가 존재하지 않고, 현재 로컬 모드로 동작할 경우 다운로드 수행
    if not os.path.exists(SOURCE) and source_dir == LOCAL_PATH:
        print(f"[안내] 원본 파일 {SOURCE} 이 존재하지 않아 인터넷에서 다운로드를 시작합니다.")
        try:
            download_and_extract_bgl(LOCAL_PATH)
        except Exception as e:
            print("원본 데이터 준비 실패로 작업을 중단합니다.", file=sys.stderr)
            sys.exit(1)
            
    # 3. 샘플링 작업 수행
    try:
        sample_bgl(SOURCE, TARGET, normal_sample_size=80000)
    except Exception as e:
        sys.exit(1)
