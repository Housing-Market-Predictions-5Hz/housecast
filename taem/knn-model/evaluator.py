#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
부동산 가격 예측 프로젝트 - KNN 모델
모델 평가 모듈

이 모듈은 훈련된 모델을 평가하고 성능 지표를 계산합니다.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import cross_val_score

def evaluate_model(model, X_train, y_train, X_val=None, y_val=None):
    """
    모델 성능을 평가하는 함수
    
    Args:
        model: 훈련된 모델 객체
        X_train (np.ndarray): 훈련 데이터 특성
        y_train (np.ndarray): 훈련 데이터 타겟
        X_val (np.ndarray, optional): 검증 데이터 특성
        y_val (np.ndarray, optional): 검증 데이터 타겟
        
    Returns:
        dict: 평가 지표 결과
    """
    results = {}
    
    # 훈련 데이터에 대한 성능 평가
    train_pred = model.predict(X_train)
    train_rmse = np.sqrt(mean_squared_error(y_train, train_pred))
    train_mae = mean_absolute_error(y_train, train_pred)
    train_r2 = r2_score(y_train, train_pred)
    
    results['train_rmse'] = train_rmse
    results['train_mae'] = train_mae
    results['train_r2'] = train_r2
    
    print("훈련 데이터 성능 평가:")
    print(f"RMSE: {train_rmse:.4f}")
    print(f"MAE: {train_mae:.4f}")
    print(f"R²: {train_r2:.4f}")
    
    # 검증 데이터가 있으면 검증 데이터에 대한 성능 평가
    if X_val is not None and y_val is not None:
        val_pred = model.predict(X_val)
        val_rmse = np.sqrt(mean_squared_error(y_val, val_pred))
        val_mae = mean_absolute_error(y_val, val_pred)
        val_r2 = r2_score(y_val, val_pred)
        
        results['val_rmse'] = val_rmse
        results['val_mae'] = val_mae
        results['val_r2'] = val_r2
        
        print("\n검증 데이터 성능 평가:")
        print(f"RMSE: {val_rmse:.4f}")
        print(f"MAE: {val_mae:.4f}")
        print(f"R²: {val_r2:.4f}")
    
    # 교차 검증 성능 평가
    print("\n교차 검증 성능 평가:")
    cv_rmse = -cross_val_score(model, X_train, y_train, cv=5, scoring='neg_root_mean_squared_error')
    cv_mae = -cross_val_score(model, X_train, y_train, cv=5, scoring='neg_mean_absolute_error')
    cv_r2 = cross_val_score(model, X_train, y_train, cv=5, scoring='r2')
    
    results['cv_rmse_mean'] = cv_rmse.mean()
    results['cv_rmse_std'] = cv_rmse.std()
    results['cv_mae_mean'] = cv_mae.mean()
    results['cv_mae_std'] = cv_mae.std()
    results['cv_r2_mean'] = cv_r2.mean()
    results['cv_r2_std'] = cv_r2.std()
    
    print(f"CV RMSE: {cv_rmse.mean():.4f} ± {cv_rmse.std():.4f}")
    print(f"CV MAE: {cv_mae.mean():.4f} ± {cv_mae.std():.4f}")
    print(f"CV R²: {cv_r2.mean():.4f} ± {cv_r2.std():.4f}")
    
    return results

def create_evaluation_plots(y_true, y_pred, save_path='output'):
    """
    모델 평가를 위한 시각화 그래프를 생성하는 함수
    
    Args:
        y_true (np.ndarray): 실제 타겟 값
        y_pred (np.ndarray): 예측 타겟 값
        save_path (str): 그래프 저장 경로
    """
    try:
        # 예측 vs 실제 산점도
        plt.figure(figsize=(10, 6))
        plt.scatter(y_true, y_pred, alpha=0.5)
        plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--')
        plt.xlabel('실제 가격')
        plt.ylabel('예측 가격')
        plt.title('실제 가격 vs 예측 가격')
        plt.tight_layout()
        plt.savefig(f'{save_path}/actual_vs_predicted.png')
        
        # 잔차 히스토그램
        residuals = y_true - y_pred
        plt.figure(figsize=(10, 6))
        sns.histplot(residuals, kde=True)
        plt.xlabel('잔차')
        plt.ylabel('빈도')
        plt.title('잔차 분포')
        plt.axvline(x=0, color='r', linestyle='--')
        plt.tight_layout()
        plt.savefig(f'{save_path}/residuals_histogram.png')
        
        # 잔차 vs 예측 값 산점도
        plt.figure(figsize=(10, 6))
        plt.scatter(y_pred, residuals, alpha=0.5)
        plt.axhline(y=0, color='r', linestyle='--')
        plt.xlabel('예측 가격')
        plt.ylabel('잔차')
        plt.title('예측 가격 vs 잔차')
        plt.tight_layout()
        plt.savefig(f'{save_path}/predicted_vs_residuals.png')
        
        print(f"평가 그래프가 '{save_path}' 디렉토리에 저장되었습니다.")
    
    except Exception as e:
        print(f"그래프 생성 중 오류 발생: {e}")

def compare_k_performance(X_train, y_train, k_values=None):
    """
    다양한 k 값에 따른 모델 성능을 비교하는 함수
    
    Args:
        X_train (np.ndarray): 훈련 데이터 특성
        y_train (np.ndarray): 훈련 데이터 타겟
        k_values (list, optional): 테스트할 k 값 목록
        
    Returns:
        tuple: (best_k, performance_data)
    """
    from sklearn.neighbors import KNeighborsRegressor
    from sklearn.model_selection import KFold
    
    if k_values is None:
        k_values = list(range(1, 21))  # 기본값: 1부터 20까지의 k 값
    
    cv_rmse = []
    cv_r2 = []
    
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    
    for k in k_values:
        print(f"k = {k} 평가 중...")
        model = KNeighborsRegressor(n_neighbors=k, weights='distance')
        
        # 교차 검증 성능
        rmse_scores = []
        r2_scores = []
        
        for train_idx, val_idx in kf.split(X_train):
            X_train_cv, X_val_cv = X_train[train_idx], X_train[val_idx]
            y_train_cv, y_val_cv = y_train[train_idx], y_train[val_idx]
            
            model.fit(X_train_cv, y_train_cv)
            val_pred = model.predict(X_val_cv)
            
            rmse = np.sqrt(mean_squared_error(y_val_cv, val_pred))
            r2 = r2_score(y_val_cv, val_pred)
            
            rmse_scores.append(rmse)
            r2_scores.append(r2)
        
        cv_rmse.append(np.mean(rmse_scores))
        cv_r2.append(np.mean(r2_scores))
    
    # 결과 시각화
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(k_values, cv_rmse, 'o-')
    plt.xlabel('k 값')
    plt.ylabel('RMSE')
    plt.title('k 값에 따른 RMSE (교차 검증)')
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    plt.plot(k_values, cv_r2, 'o-')
    plt.xlabel('k 값')
    plt.ylabel('R²')
    plt.title('k 값에 따른 R² (교차 검증)')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('output/k_performance_comparison.png')
    
    # 최적 k 값 찾기
    best_k_idx = np.argmin(cv_rmse)
    best_k = k_values[best_k_idx]
    
    print(f"\n최적 k 값: {best_k}")
    print(f"최적 k 값의 RMSE: {cv_rmse[best_k_idx]:.4f}")
    print(f"최적 k 값의 R²: {cv_r2[best_k_idx]:.4f}")
    
    performance_data = pd.DataFrame({
        'k': k_values,
        'rmse': cv_rmse,
        'r2': cv_r2
    })
    
    return best_k, performance_data

if __name__ == "__main__":
    # 테스트 실행
    # 필요한 모듈 로드
    from data_loader import load_data
    from preprocessor import preprocess_data
    from model_trainer import train_model
    from sklearn.model_selection import train_test_split
    import os
    
    # 결과 저장 디렉토리 생성
    os.makedirs('output', exist_ok=True)
    
    # 데이터 로드 및 전처리
    train_data, test_data, bus_data, subway_data = load_data()
    X_train, y_train, X_test, processed_data = preprocess_data(train_data, test_data, bus_data, subway_data)
    
    # 훈련/검증 데이터 분할
    X_train_split, X_val, y_train_split, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)
    
    # k 값 성능 비교
    best_k, k_perf = compare_k_performance(X_train_split, y_train_split)
    
    # 최적 k로 모델 훈련
    from sklearn.neighbors import KNeighborsRegressor
    model = KNeighborsRegressor(n_neighbors=best_k, weights='distance')
    model.fit(X_train, y_train)
    
    # 모델 평가
    results = evaluate_model(model, X_train, y_train)
    
    # 평가 그래프 생성
    y_pred = model.predict(X_train)
    create_evaluation_plots(y_train, y_pred) 