#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
부동산 가격 예측 프로젝트 - K-Nearest Neighbors 모델
모델 훈련 모듈

이 모듈은 KNN 모델을 훈련하고 파라미터 튜닝을 수행합니다.
"""

import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

def train_knn_model(X_train, y_train, tune_hyperparams=False):
    """
    KNN 모델을 훈련하는 함수
    
    Args:
        X_train (np.ndarray): 훈련 데이터의 특성 행렬
        y_train (np.ndarray): 훈련 데이터의 타겟 값
        tune_hyperparams (bool): 하이퍼파라미터 튜닝 여부
        
    Returns:
        tuple: (trained_model, best_params, training_metrics)
    """
    print("K-Nearest Neighbors 모델 훈련 시작...")
    start_time = time.time()
    
    if tune_hyperparams:
        print("하이퍼파라미터 튜닝 시작...")
        
        # GridSearchCV를 이용한 하이퍼파라미터 튜닝
        param_grid = {
            'n_neighbors': [3, 5, 7, 9, 11, 13, 15, 20],
            'weights': ['uniform', 'distance'],
            'p': [1, 2],  # p=1: Manhattan, p=2: Euclidean
            'leaf_size': [10, 20, 30, 40, 50]
        }
        
        knn = KNeighborsRegressor()
        grid_search = GridSearchCV(
            estimator=knn,
            param_grid=param_grid,
            cv=3,
            scoring='neg_root_mean_squared_error',
            n_jobs=2,
            verbose=1
        )
        
        grid_search.fit(X_train, y_train)
        model = grid_search.best_estimator_
        best_params = grid_search.best_params_
        
        print(f"최적 하이퍼파라미터: {best_params}")
        print(f"최적 CV 점수 (RMSE): {-grid_search.best_score_:.4f}")
    else:
        # 기본 하이퍼파라미터로 모델 훈련
        best_params = {
            'n_neighbors': 5,
            'weights': 'distance',
            'p': 2,
            'leaf_size': 30
        }
        model = KNeighborsRegressor(
            n_neighbors=best_params['n_neighbors'],
            weights=best_params['weights'],
            p=best_params['p'],
            leaf_size=best_params['leaf_size']
        )
        model.fit(X_train, y_train)
    
    # 모델 성능 평가
    train_pred = model.predict(X_train)
    rmse = np.sqrt(mean_squared_error(y_train, train_pred))
    mae = mean_absolute_error(y_train, train_pred)
    r2 = r2_score(y_train, train_pred)
    
    training_metrics = {
        'rmse': rmse,
        'mae': mae,
        'r2': r2
    }
    
    end_time = time.time()
    train_time = end_time - start_time
    print(f"모델 훈련 완료! 훈련 시간: {train_time:.2f}초")
    print(f"훈련 데이터 성능 - RMSE: {rmse:.4f}, MAE: {mae:.4f}, R²: {r2:.4f}")
    
    return model, best_params, training_metrics

def analyze_k_values(X_train, y_train, X_test=None, y_test=None, max_k=50):
    """
    다양한 k 값에 대한 성능 분석
    
    Args:
        X_train (np.ndarray): 훈련 데이터의 특성 행렬
        y_train (np.ndarray): 훈련 데이터의 타겟 값
        X_test (np.ndarray): 테스트 데이터의 특성 행렬 (선택적)
        y_test (np.ndarray): 테스트 데이터의 타겟 값 (선택적)
        max_k (int): 실험할 최대 k 값
        
    Returns:
        dict: k 값에 따른 성능 지표
    """
    k_values = list(range(1, max_k + 1))
    train_rmse = []
    train_r2 = []
    test_rmse = []
    test_r2 = []
    
    print(f"K 값 분석 시작 (1부터 {max_k}까지)...")
    
    for k in k_values:
        model = KNeighborsRegressor(n_neighbors=k, weights='distance')
        model.fit(X_train, y_train)
        
        # 훈련 데이터 성능
        train_pred = model.predict(X_train)
        train_rmse.append(np.sqrt(mean_squared_error(y_train, train_pred)))
        train_r2.append(r2_score(y_train, train_pred))
        
        # 테스트 데이터가 제공된 경우 테스트 성능
        if X_test is not None and y_test is not None:
            test_pred = model.predict(X_test)
            test_rmse.append(np.sqrt(mean_squared_error(y_test, test_pred)))
            test_r2.append(r2_score(y_test, test_pred))
    
    # 결과 시각화
    plt.figure(figsize=(14, 7))
    
    plt.subplot(1, 2, 1)
    plt.plot(k_values, train_rmse, 'bo-', label='훈련 RMSE')
    if test_rmse:
        plt.plot(k_values, test_rmse, 'ro-', label='테스트 RMSE')
    plt.xlabel('k 값')
    plt.ylabel('RMSE')
    plt.legend()
    plt.title('k 값에 따른 RMSE 변화')
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    plt.plot(k_values, train_r2, 'bo-', label='훈련 R²')
    if test_r2:
        plt.plot(k_values, test_r2, 'ro-', label='테스트 R²')
    plt.xlabel('k 값')
    plt.ylabel('R²')
    plt.legend()
    plt.title('k 값에 따른 R² 변화')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('output/knn_k_analysis.png')
    print("K 값 분석 결과가 'output/knn_k_analysis.png'에 저장되었습니다.")
    
    # 최적의 k 값 찾기
    if test_rmse:
        best_k_idx = np.argmin(test_rmse)
        best_k = k_values[best_k_idx]
        print(f"테스트 RMSE 기준 최적 k: {best_k} (RMSE: {test_rmse[best_k_idx]:.4f})")
    else:
        best_k_idx = np.argmin(train_rmse)
        best_k = k_values[best_k_idx]
        print(f"훈련 RMSE 기준 최적 k: {best_k} (RMSE: {train_rmse[best_k_idx]:.4f})")
    
    return {
        'k_values': k_values,
        'train_rmse': train_rmse,
        'train_r2': train_r2,
        'test_rmse': test_rmse if test_rmse else None,
        'test_r2': test_r2 if test_r2 else None,
        'best_k': best_k
    }

def compare_with_random_forest(X_train, y_train, knn_model, rf_model, fold=3):
    """
    KNN과 Random Forest 모델의 교차 검증 성능을 비교
    
    Args:
        X_train (np.ndarray): 훈련 데이터의 특성 행렬
        y_train (np.ndarray): 훈련 데이터의 타겟 값
        knn_model: 훈련된 KNN 모델
        rf_model: 훈련된 Random Forest 모델
        fold (int): 교차 검증 폴드 수
        
    Returns:
        dict: 비교 결과
    """
    print(f"{fold}겹 교차 검증을 통한 모델 비교 시작...")
    
    # 교차 검증 수행
    knn_rmse_scores = -cross_val_score(knn_model, X_train, y_train, 
                                      cv=fold, scoring='neg_root_mean_squared_error')
    knn_r2_scores = cross_val_score(knn_model, X_train, y_train, 
                                   cv=fold, scoring='r2')
    
    rf_rmse_scores = -cross_val_score(rf_model, X_train, y_train, 
                                     cv=fold, scoring='neg_root_mean_squared_error')
    rf_r2_scores = cross_val_score(rf_model, X_train, y_train, 
                                  cv=fold, scoring='r2')
    
    # 결과 출력
    print("\n교차 검증 결과:")
    print(f"KNN - 평균 RMSE: {knn_rmse_scores.mean():.4f} ± {knn_rmse_scores.std():.4f}")
    print(f"KNN - 평균 R²: {knn_r2_scores.mean():.4f} ± {knn_r2_scores.std():.4f}")
    print(f"Random Forest - 평균 RMSE: {rf_rmse_scores.mean():.4f} ± {rf_rmse_scores.std():.4f}")
    print(f"Random Forest - 평균 R²: {rf_r2_scores.mean():.4f} ± {rf_r2_scores.std():.4f}")
    
    # 결과 시각화
    plt.figure(figsize=(12, 6))
    
    # RMSE 비교
    plt.subplot(1, 2, 1)
    labels = ['KNN', 'Random Forest']
    rmse_means = [knn_rmse_scores.mean(), rf_rmse_scores.mean()]
    rmse_stds = [knn_rmse_scores.std(), rf_rmse_scores.std()]
    
    bars = plt.bar(labels, rmse_means, yerr=rmse_stds, capsize=10, color=['skyblue', 'lightgreen'])
    plt.ylabel('RMSE (낮을수록 좋음)')
    plt.title('RMSE 비교')
    plt.grid(axis='y')
    
    # 값 표시
    for bar, val in zip(bars, rmse_means):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1, f'{val:.4f}', 
                ha='center', va='bottom')
    
    # R² 비교
    plt.subplot(1, 2, 2)
    r2_means = [knn_r2_scores.mean(), rf_r2_scores.mean()]
    r2_stds = [knn_r2_scores.std(), rf_r2_scores.std()]
    
    bars = plt.bar(labels, r2_means, yerr=r2_stds, capsize=10, color=['skyblue', 'lightgreen'])
    plt.ylabel('R² (높을수록 좋음)')
    plt.title('R² 비교')
    plt.grid(axis='y')
    
    # 값 표시
    for bar, val in zip(bars, r2_means):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, f'{val:.4f}', 
                ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig('output/knn_vs_rf_comparison.png')
    print("모델 비교 결과가 'output/knn_vs_rf_comparison.png'에 저장되었습니다.")
    
    return {
        'knn_rmse': {
            'mean': knn_rmse_scores.mean(),
            'std': knn_rmse_scores.std()
        },
        'knn_r2': {
            'mean': knn_r2_scores.mean(),
            'std': knn_r2_scores.std()
        },
        'rf_rmse': {
            'mean': rf_rmse_scores.mean(),
            'std': rf_rmse_scores.std()
        },
        'rf_r2': {
            'mean': rf_r2_scores.mean(),
            'std': rf_r2_scores.std()
        }
    }

if __name__ == "__main__":
    # 테스트 실행을 위한 데이터 로드
    from data_loader import load_data
    from preprocessor import preprocess_data
    from model_trainer import train_model
    import os
    
    # 결과 저장 디렉토리 생성
    os.makedirs('output', exist_ok=True)
    
    # 데이터 로드 및 전처리
    train_data, test_data, bus_data, subway_data = load_data()
    X_train, y_train, X_test, processed_data = preprocess_data(train_data, test_data, bus_data, subway_data)
    
    # KNN 모델 훈련 (기본 하이퍼파라미터)
    knn_model, knn_params, knn_metrics = train_knn_model(X_train, y_train, tune_hyperparams=False)
    
    # 다양한 k 값 분석
    k_analysis = analyze_k_values(X_train, y_train, max_k=30)
    
    # 최적 k를 적용한 모델 재훈련
    best_k = k_analysis['best_k']
    print(f"\n최적 k={best_k}로 KNN 모델 재훈련...")
    optimized_knn_model = KNeighborsRegressor(
        n_neighbors=best_k,
        weights='distance',
        p=2
    )
    optimized_knn_model.fit(X_train, y_train)
    
    # Random Forest 모델 훈련 (비교용)
    rf_model, _ = train_model(X_train, y_train, tune_hyperparams=False)
    
    # 모델 비교
    comparison_results = compare_with_random_forest(X_train, y_train, 
                                                  optimized_knn_model, rf_model)
    
    # 최종 결과 요약
    print("\nKNN 모델 총평:")
    if comparison_results['knn_rmse']['mean'] < comparison_results['rf_rmse']['mean']:
        print("KNN 모델이 Random Forest보다 더 나은 성능을 보여줍니다.")
    else:
        print("Random Forest 모델이 KNN보다 더 나은 성능을 보여줍니다.")
        
    print("\n부동산 가격 예측에 관한 KNN 모델 분석 결론:")
    print("1. KNN은 가까운 이웃의 값을 기반으로 예측하므로 지역적 트렌드를 잘 포착할 수 있습니다.")
    print("2. 하지만 고차원 데이터에서는 차원의 저주 문제로 성능이 저하될 수 있습니다.")
    print("3. Random Forest는 여러 특성 간의 복잡한 관계를 학습할 수 있어 부동산 데이터에 적합할 수 있습니다.")
    print("4. 최종적으로 두 모델의 성능을 비교하여 더 나은 모델을 선택하는 것이 중요합니다.") 