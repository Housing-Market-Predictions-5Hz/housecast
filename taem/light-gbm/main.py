#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
부동산 가격 예측 프로젝트 - LightGBM 모델
메인 스크립트

이 스크립트는 부동산 가격 예측 프로젝트의 메인 실행 파일입니다.
데이터 로드, 전처리, 모델 훈련, 예측 및 결과 저장의 전체 과정을 실행합니다.
"""

import os
import pandas as pd
import numpy as np
import time
import joblib
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_squared_error
from pathlib import Path

# 자체 모듈 임포트
from preprocessor.preprocessor_enhanced import load_data, preprocess_data
from model_trainer import train_model
from evaluator import evaluate_model, create_evaluation_plots

def create_output_dir():
    """결과 저장을 위한 디렉토리 생성"""
    os.makedirs('output', exist_ok=True)
    os.makedirs('models', exist_ok=True)
    os.makedirs('data', exist_ok=True)

def clean_column_names(df):
    """특수문자 제거 + 중복 컬럼 처리 함수"""
    df.columns = df.columns.str.replace('[^A-Za-z0-9_]+', '_', regex=True)
    seen = {}
    new_cols = []
    for col in df.columns:
        if col not in seen:
            seen[col] = 1
            new_cols.append(col)
        else:
            seen[col] += 1
            new_cols.append(f"{col}_{seen[col]-1}")
    df.columns = new_cols
    return df

def ensure_numeric(df):
    """object 타입 컬럼 자동 변환 함수"""
    bad_cols = df.select_dtypes(include=["object"]).columns.tolist()
    if bad_cols:
        print(f"⚠️ object 타입 컬럼 자동 변환: {bad_cols}")
        for col in bad_cols:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)
    return df

def main():
    print("부동산 가격 예측 프로젝트 시작 - LightGBM 모델")
    start_time = time.time()

    # 결과 저장 디렉토리 생성
    create_output_dir()

    # 데이터 파일 경로 설정
    raw_dir = Path("../../raw")
    train_path = raw_dir / "train.csv"
    test_path = raw_dir / "test.csv"
    bus_path = raw_dir / "bus_feature.csv"
    subway_path = raw_dir / "subway_feature.csv"
    submission_path = raw_dir / "sample_submission.csv"

    # 데이터 로드
    print("1. 데이터 로드 중...")
    train_data, test_data, bus_data, subway_data, sample_submission = load_data(
        train_path, test_path, bus_path, subway_path, submission_path
    )
    print(f"   - 훈련 데이터 크기: {train_data.shape}")
    print(f"   - 테스트 데이터 크기: {test_data.shape}")
    print(f"   - 버스 데이터 크기: {bus_data.shape}")
    print(f"   - 지하철 데이터 크기: {subway_data.shape}")

    # 데이터 전처리
    print("2. 데이터 전처리 중...")
    train_processed, test_processed, _, _ = preprocess_data(train_data, test_data, bus_data, subway_data)
    
    # 컬럼명 정리 및 데이터 타입 확인
    train_processed = clean_column_names(train_processed)
    test_processed = clean_column_names(test_processed)
    train_processed = ensure_numeric(train_processed)
    test_processed = ensure_numeric(test_processed)
    
    print(f"   - 전처리 후 훈련 데이터 크기: {train_processed.shape}")
    print(f"   - 전처리 후 테스트 데이터 크기: {test_processed.shape}")
    
    # 타겟 변수 분리
    y_train = train_processed["target"].values
    X_train = train_processed.drop(columns=["target"])
    X_test = test_processed
    
    # 중요 정보 저장
    X_train.to_csv('output/X_train_processed.csv', index=False)
    X_test.to_csv('output/X_test_processed.csv', index=False)
    pd.DataFrame(y_train, columns=['target']).to_csv('output/y_train.csv', index=False)
    
    # 모델 훈련
    print("3. 모델 훈련 중...")
    model, feature_importance = train_model(X_train, y_train)
    
    # 모델 저장
    print("4. 모델 저장 중...")
    joblib.dump(model, 'models/lightgbm_model.pkl')
    
    # 중요 특성 저장
    feature_importance.to_csv('output/feature_importance.csv', index=False)
    
    # 모델 평가
    print("5. 모델 평가 중...")
    evaluation_results = evaluate_model(model, X_train, y_train)
    print(f"   - RMSE (훈련 데이터): {evaluation_results['train_rmse']:.4f}")
    print(f"   - R² (훈련 데이터): {evaluation_results['train_r2']:.4f}")
    
    # 평가 시각화
    print("6. 평가 시각화 생성 중...")
    y_pred_train = model.predict(X_train)
    create_evaluation_plots(y_train, y_pred_train)
    
    # 테스트 데이터에 대한 예측
    print("7. 테스트 데이터 예측 중...")
    predictions = model.predict(X_test)
    
    # 예측값을 정수형으로 변환
    predictions = np.round(predictions).astype(int)
    
    # 제출 파일 생성
    print("8. 제출 파일 생성 중...")
    sample_submission['target'] = predictions
    
    # 정수 타입으로 저장
    sample_submission['target'] = sample_submission['target'].astype(int)
    
    # 결과 저장
    sample_submission.to_csv('output/output.csv', index=False)
    
    # 디버깅용 파일도 저장
    sample_submission.to_csv('output/lgbm_submission.csv', index=False)
    
    # 실행 시간 계산
    end_time = time.time()
    execution_time = end_time - start_time
    print(f"프로젝트 실행 완료! 총 실행 시간: {execution_time:.2f}초")
    print(f"결과는 'output' 디렉토리에 저장되었습니다.")
    print(f"제출용 파일: output/output.csv (정수형 값으로 저장됨)")

if __name__ == "__main__":
    main() 