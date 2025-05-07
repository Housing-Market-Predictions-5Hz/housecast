#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
부동산 가격 예측 프로젝트 - Random Forest 모델
유틸리티 모듈

이 모듈은 여러 모듈에서 공통적으로 사용되는 유틸리티 함수를 제공합니다.
"""

import os
import time
import pandas as pd
import numpy as np
import datetime

def create_directory(path):
    """
    디렉토리가 존재하지 않으면 생성하는 함수
    
    Args:
        path (str): 생성할 디렉토리 경로
    """
    os.makedirs(path, exist_ok=True)
    print(f"디렉토리 생성 완료: {path}")

def save_dataframe(df, path, index=False):
    """
    데이터프레임을 CSV 파일로 저장하는 함수
    
    Args:
        df (pd.DataFrame): 저장할 데이터프레임
        path (str): 저장할 파일 경로
        index (bool): 인덱스 저장 여부
    """
    directory = os.path.dirname(path)
    create_directory(directory)
    df.to_csv(path, index=index)
    print(f"데이터프레임이 저장되었습니다: {path}")

def get_timestamp():
    """
    현재 시간을 포맷팅된 문자열로 반환하는 함수
    
    Returns:
        str: 'YYYYMMDD_HHMMSS' 형식의 시간 문자열
    """
    return datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

def save_log(message, log_path='output/log.txt'):
    """
    로그 메시지를 파일에 저장하는 함수
    
    Args:
        message (str): 로그 메시지
        log_path (str): 로그 파일 경로
    """
    directory = os.path.dirname(log_path)
    create_directory(directory)
    
    timestamp = get_timestamp()
    with open(log_path, 'a', encoding='utf-8') as f:
        f.write(f"[{timestamp}] {message}\n")

def print_summary_statistics(df, title='데이터 요약 통계'):
    """
    데이터프레임의 요약 통계를 출력하는 함수
    
    Args:
        df (pd.DataFrame): 요약 통계를 출력할 데이터프레임
        title (str): 출력할 제목
    """
    print(f"\n{title}")
    print("-" * 80)
    print(f"데이터 크기: {df.shape[0]} 행 x {df.shape[1]} 열")
    print(f"결측치 수: {df.isna().sum().sum()}")
    
    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
    if len(numeric_cols) > 0:
        print("\n수치형 변수 요약:")
        print(df[numeric_cols].describe().T)
    
    categorical_cols = df.select_dtypes(include=['object']).columns
    if len(categorical_cols) > 0:
        print("\n범주형 변수 요약:")
        for col in categorical_cols:
            print(f"\n{col}:")
            print(df[col].value_counts().head(5))
    
    print("-" * 80)

def calculate_execution_time(start_time, end_time=None):
    """
    실행 시간을 계산하는 함수
    
    Args:
        start_time (float): 시작 시간
        end_time (float, optional): 종료 시간. None이면 현재 시간 사용
        
    Returns:
        str: 형식화된 실행 시간 문자열
    """
    if end_time is None:
        end_time = time.time()
    
    execution_time = end_time - start_time
    
    if execution_time < 60:
        return f"{execution_time:.2f}초"
    elif execution_time < 3600:
        minutes = int(execution_time // 60)
        seconds = execution_time % 60
        return f"{minutes}분 {seconds:.2f}초"
    else:
        hours = int(execution_time // 3600)
        minutes = int((execution_time % 3600) // 60)
        seconds = execution_time % 60
        return f"{hours}시간 {minutes}분 {seconds:.2f}초"

if __name__ == "__main__":
    # 유틸리티 함수 테스트
    
    # 디렉토리 생성 테스트
    create_directory('taem/random-forest/test')
    
    # 타임스탬프 테스트
    print(f"현재 타임스탬프: {get_timestamp()}")
    
    # 로그 저장 테스트
    save_log("유틸리티 모듈 테스트")
    
    # 실행 시간 계산 테스트
    start = time.time()
    time.sleep(2)  # 2초 대기
    print(f"실행 시간: {calculate_execution_time(start)}")
    
    # 더미 데이터프레임으로 요약 통계 출력 테스트
    df = pd.DataFrame({
        'A': np.random.randn(100),
        'B': np.random.randn(100),
        'C': np.random.choice(['X', 'Y', 'Z'], 100)
    })
    print_summary_statistics(df, "테스트 데이터프레임 요약") 