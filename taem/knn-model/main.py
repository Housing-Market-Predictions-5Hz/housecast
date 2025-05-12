#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
부동산 가격 예측 프로젝트 - KNN 모델
메인 스크립트

이 스크립트는 부동산 가격 예측 프로젝트의 메인 실행 파일입니다.
데이터 로드, 전처리, 모델 훈련, 예측 및 결과 저장의 전체 과정을 실행합니다.
"""

import os
import pandas as pd
import numpy as np
import time
import joblib
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error

# 자체 모듈 임포트
from data_loader import load_data
from preprocessor import preprocess_data
from model_trainer import train_model, analyze_k_values
from evaluator import evaluate_model

def create_output_dir():
    """결과 저장을 위한 디렉토리 생성"""
    os.makedirs('output', exist_ok=True)
    os.makedirs('models', exist_ok=True)

def main():
    print("부동산 가격 예측 프로젝트 시작 - KNN 모델")
    start_time = time.time()

    # 결과 저장 디렉토리 생성
    create_output_dir()

    # 데이터 로드
    print("1. 데이터 로드 중...")
    train_data, test_data, bus_data, subway_data = load_data()
    print(f"   - 훈련 데이터 크기: {train_data.shape}")
    print(f"   - 테스트 데이터 크기: {test_data.shape}")
    print(f"   - 버스 데이터 크기: {bus_data.shape}")
    print(f"   - 지하철 데이터 크기: {subway_data.shape}")

    # 데이터 전처리
    print("2. 데이터 전처리 중...")
    X_train, y_train, X_test, processed_data = preprocess_data(train_data, test_data, bus_data, subway_data)
    print(f"   - 전처리 후 훈련 데이터 크기: {X_train.shape}")
    print(f"   - 전처리 후 테스트 데이터 크기: {X_test.shape}")
    
    # 중요 정보 저장
    processed_data['X_train'].to_csv('output/X_train_processed.csv', index=False)
    processed_data['X_test'].to_csv('output/X_test_processed.csv', index=False)
    pd.DataFrame(y_train, columns=['target']).to_csv('output/y_train.csv', index=False)
    
    # 최적 k 값 탐색
    print("3. 최적의 k 값 분석 중...")
    k_analysis = analyze_k_values(X_train, y_train, max_k=20)
    best_k = k_analysis['best_k']
    print(f"   - 최적 k 값: {best_k}")
    
    # 최적 k 값으로 모델 훈련
    print("4. 최적 k 값으로 모델 훈련 중...")
    model = KNeighborsRegressor(
        n_neighbors=best_k,
        weights='distance',
        p=2,
        leaf_size=30,
        n_jobs=-1
    )
    model.fit(X_train, y_train)
    
    # 모델 저장
    print("5. 모델 저장 중...")
    joblib.dump(model, 'models/knn_model.pkl')
    
    # 모델 평가
    print("6. 모델 평가 중...")
    evaluation_results = evaluate_model(model, X_train, y_train)
    print(f"   - RMSE (훈련 데이터): {evaluation_results['train_rmse']:.4f}")
    print(f"   - R² (훈련 데이터): {evaluation_results['train_r2']:.4f}")
    
    # 테스트 데이터에 대한 예측
    print("7. 테스트 데이터 예측 중...")
    predictions = model.predict(X_test)
    
    # 예측값을 정수형으로 변환
    predictions = predictions.astype(int)
    
    # 제출 파일 생성
    print("8. 제출 파일 생성 중...")
    sample_submission_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'raw', 'sample_submission.csv')
    sample_submission = pd.read_csv(sample_submission_path)
    sample_submission['target'] = predictions
    
    # float가 아닌 integer 타입으로 저장
    sample_submission['target'] = sample_submission['target'].astype(int)
    
    # 변경된 파일명(output.csv)으로 저장
    sample_submission.to_csv('output/output.csv', index=False)
    
    # 디버깅용 원본 파일도 저장 (선택사항)
    sample_submission.to_csv('output/knn_submission.csv', index=False)
    
    # 추가 정보 저장
    model_info = {
        'best_k': best_k,
        'rmse': evaluation_results['train_rmse'],
        'r2': evaluation_results['train_r2'],
        'features_count': X_train.shape[1]
    }
    pd.DataFrame([model_info]).to_csv('output/model_info.csv', index=False)
    
    # 실행 시간 계산
    end_time = time.time()
    execution_time = end_time - start_time
    print(f"프로젝트 실행 완료! 총 실행 시간: {execution_time:.2f}초")
    print(f"결과는 'output' 디렉토리에 저장되었습니다.")
    print(f"제출용 파일: output/output.csv (정수형 값으로 저장됨)")

if __name__ == "__main__":
    main() 