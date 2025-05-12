#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
부동산 가격 예측 프로젝트 - KNN 모델
데이터 로더 모듈

이 모듈은 원본 데이터를 로드하는 기능을 제공합니다.
"""

import pandas as pd
import numpy as np
import os

def load_data():
    """
    원본 데이터를 로드하는 함수
    
    Returns:
        tuple: (train_data, test_data, bus_data, subway_data)
    """
    # 파일 경로 설정 - 프로젝트 루트 기준 절대 경로 사용
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.join(os.path.dirname(os.path.dirname(current_dir)))
    
    raw_dir = os.path.join(project_root, 'raw')
    train_path = os.path.join(raw_dir, 'train.csv')
    test_path = os.path.join(raw_dir, 'test.csv')
    bus_path = os.path.join(raw_dir, 'bus_feature.csv')
    subway_path = os.path.join(raw_dir, 'subway_feature.csv')
    
    # 파일 로드
    try:
        train_data = pd.read_csv(train_path)
        test_data = pd.read_csv(test_path)
        bus_data = pd.read_csv(bus_path)
        subway_data = pd.read_csv(subway_path)
        
        print("모든 데이터 파일이 성공적으로 로드되었습니다.")
        
        # 데이터 간단한 정보 출력
        print(f"훈련 데이터 크기: {train_data.shape}, 결측치: {train_data.isna().sum().sum()}")
        print(f"테스트 데이터 크기: {test_data.shape}, 결측치: {test_data.isna().sum().sum()}")
        print(f"버스 데이터 크기: {bus_data.shape}, 결측치: {bus_data.isna().sum().sum()}")
        print(f"지하철 데이터 크기: {subway_data.shape}, 결측치: {subway_data.isna().sum().sum()}")
        
        return train_data, test_data, bus_data, subway_data
    
    except FileNotFoundError as e:
        print(f"파일을 찾을 수 없습니다: {e}")
        raise
    except Exception as e:
        print(f"데이터 로드 중 오류 발생: {e}")
        raise

if __name__ == "__main__":
    # 테스트 실행
    train_data, test_data, bus_data, subway_data = load_data()
    
    # 데이터 미리보기
    print("\n훈련 데이터 미리보기:")
    print(train_data.head())
    
    print("\n테스트 데이터 미리보기:")
    print(test_data.head())
    
    print("\n버스 데이터 미리보기:")
    print(bus_data.head())
    
    print("\n지하철 데이터 미리보기:")
    print(subway_data.head()) 