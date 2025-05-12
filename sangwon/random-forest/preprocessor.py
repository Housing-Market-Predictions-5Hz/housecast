#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
부동산 가격 예측 프로젝트 - Random Forest 모델
전처리 모듈

이 모듈은 데이터 전처리, 특성 엔지니어링 및 변환을 담당합니다.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer

def preprocess_data(train_data, test_data, bus_data, subway_data):
    """
    데이터 전처리 및 특성 엔지니어링을 수행하는 함수
    
    Args:
        train_data (pd.DataFrame): 훈련 데이터셋
        test_data (pd.DataFrame): 테스트 데이터셋
        bus_data (pd.DataFrame): 버스 관련 데이터
        subway_data (pd.DataFrame): 지하철 관련 데이터
        
    Returns:
        tuple: (X, y, X_test)
    """
    print("데이터 전처리 및 특성 엔지니어링 시작...")
    
    # 데이터 복사본 생성
    train_df = train_data.copy()
    test_df = test_data.copy()
    
    # 타겟 변수 분리
    y = train_df['target'].values
    X = train_df.drop(columns=['target'])
    
    # 테스트 데이터
    X_test = test_df.copy()
    
    # 결측치 처리
    print("1. 결측치 처리 중...")
    imputer = SimpleImputer(strategy='most_frequent')
    X = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)
    X_test = pd.DataFrame(imputer.transform(X_test), columns=X_test.columns)
    
    # 범주형 변수 인코딩
    print("2. 범주형 변수 인코딩 중...")
    categorical_columns = X.select_dtypes(include=['object']).columns
    for col in categorical_columns:
        le = LabelEncoder()
        
        # 훈련 데이터와 테스트 데이터를 결합하여 LabelEncoder 학습
        combined_data = pd.concat([X[col], X_test[col]], axis=0).astype(str)
        le.fit(combined_data)
        
        # 변환
        X[col] = le.transform(X[col].astype(str))
        # 테스트 데이터에 없는 값은 -1로 처리
        X_test[col] = X_test[col].astype(str).apply(lambda x: le.transform([x])[0] if x in le.classes_ else -1)
    
    # 스케일링
    print("3. 스케일링 중...")
    scaler = StandardScaler()
    X = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)
    X_test = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns)
    
    return X, y, X_test

if __name__ == "__main__":
    # 테스트 실행을 위한 데이터 로드
    from data_loader import load_data
    train_data, test_data, bus_data, subway_data = load_data()
    
    # 전처리 테스트
    X, y, X_test = preprocess_data(train_data, test_data, bus_data, subway_data)
    
    # 결과 확인
    print("\n전처리 결과 확인:")
    print(f"X 형태: {X.shape}")
    print(f"y 형태: {y.shape}")
    print(f"X_test 형태: {X_test.shape}")