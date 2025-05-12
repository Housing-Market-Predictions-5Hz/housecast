#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
부동산 가격 예측 프로젝트 - 선형 회귀 모델
메인 스크립트

이 스크립트는 부동산 가격 예측 프로젝트의 메인 실행 파일입니다.
데이터 로드, 전처리, 모델 훈련, 예측 및 결과 저장의 전체 과정을 실행합니다.
"""

import os
import pandas as pd
import numpy as np
import time
import joblib
import argparse
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.metrics import mean_squared_error

# 자체 모듈 임포트
from data_loader import load_data
from preprocessor import preprocess_data
from model_trainer import train_model, compare_linear_models
from evaluator import evaluate_model, analyze_coefficients

def create_output_dir():
    """결과 저장을 위한 디렉토리 생성"""
    os.makedirs('output', exist_ok=True)
    os.makedirs('models', exist_ok=True)

def main(model_type=None, tune_hyperparams=True):
    """
    메인 함수
    
    Args:
        model_type (str, optional): 사용할 모델 타입. 지정하지 않으면 가장 좋은 모델을 자동 선택
                                    ('linear', 'ridge', 'lasso', 'elastic_net', 'polynomial')
        tune_hyperparams (bool): 하이퍼파라미터 튜닝 여부
    """
    print("부동산 가격 예측 프로젝트 시작 - 선형 회귀 모델")
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
    
    # 모델 훈련 및 선택
    if model_type is None:
        # 다양한 선형 모델 훈련 및 비교
        print("3. 선형 모델 비교 중...")
        model_results, best_model_type = compare_linear_models(X_train, y_train, tune_hyperparams=False)
        print(f"   - 최적 모델 타입: {best_model_type}")
    else:
        # 사용자가 지정한 모델 사용
        best_model_type = model_type
        print(f"3. 사용자 지정 모델({best_model_type}) 사용...")
    
    # 최적 모델로 최종 훈련
    print(f"4. 최적 모델({best_model_type})로 최종 훈련 중...")
    best_model, feature_importance = train_model(X_train, y_train, tune_hyperparams=tune_hyperparams, model_type=best_model_type)
    
    # 모델 저장
    print("5. 모델 저장 중...")
    if best_model_type == 'polynomial':
        # 다항 회귀 모델은 딕셔너리 형태
        joblib.dump(best_model, f'models/polynomial_model.pkl')
    else:
        # 일반 선형 모델
        joblib.dump(best_model, f'models/{best_model_type}_model.pkl')
    
    # 특성 중요도 저장
    feature_importance.to_csv('output/feature_importance.csv', index=False)
    
    # 계수 분석 (다항 회귀 제외)
    if best_model_type != 'polynomial':
        print("6. 모델 계수 분석 중...")
        feature_names = processed_data['X_train'].columns.tolist()
        coef_analysis = analyze_coefficients(best_model, feature_names)
    
    # 모델 평가
    print("7. 모델 평가 중...")
    evaluation_results = evaluate_model(best_model, X_train, y_train, model_type=best_model_type)
    print(f"   - RMSE (훈련 데이터): {evaluation_results['train_rmse']:.4f}")
    print(f"   - R² (훈련 데이터): {evaluation_results['train_r2']:.4f}")
    
    # 테스트 데이터에 대한 예측
    print("8. 테스트 데이터 예측 중...")
    if best_model_type == 'polynomial':
        # 다항 회귀 모델 예측
        num_features = best_model['num_features']
        X_test_subset = X_test[:, :num_features]
        X_test_poly = best_model['poly'].transform(X_test_subset)
        predictions = best_model['lr'].predict(X_test_poly)
    else:
        # 일반 선형 모델 예측
        predictions = best_model.predict(X_test)
    
    # 예측값을 정수형으로 변환
    predictions = predictions.astype(int)
    
    # 제출 파일 생성
    print("9. 제출 파일 생성 중...")
    sample_submission_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'raw', 'sample_submission.csv')
    sample_submission = pd.read_csv(sample_submission_path)
    sample_submission['target'] = predictions
    
    # float가 아닌 integer 타입으로 저장
    sample_submission['target'] = sample_submission['target'].astype(int)
    
    # 변경된 파일명(output.csv)으로 저장
    sample_submission.to_csv('output/output.csv', index=False)
    
    # 디버깅용 원본 파일도 저장 (선택사항)
    sample_submission.to_csv(f'output/{best_model_type}_submission.csv', index=False)
    
    # 추가 정보 저장
    model_info = {
        'model_type': best_model_type,
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
    print(f"사용된 모델: {best_model_type}")

if __name__ == "__main__":
    # 명령줄 인자 파싱
    parser = argparse.ArgumentParser(description='부동산 가격 예측 프로젝트 - 선형 회귀 모델')
    parser.add_argument('--model', type=str, choices=['linear', 'ridge', 'lasso', 'elastic_net', 'polynomial'],
                        help='사용할 모델 타입 (기본값: 자동 선택)')
    parser.add_argument('--tune', action='store_true', help='하이퍼파라미터 튜닝 수행 여부')
    
    args = parser.parse_args()
    
    # 기본값으로 Ridge 모델 사용 (일반적으로 성능이 좋음)
    default_model = 'ridge'
    
    if args.model:
        # 사용자가 모델 지정한 경우
        main(model_type=args.model, tune_hyperparams=args.tune)
    else:
        # 모델 지정하지 않은 경우 (비교 후 최적 모델 선택 또는 기본 모델 사용)
        main(model_type=default_model, tune_hyperparams=args.tune) 