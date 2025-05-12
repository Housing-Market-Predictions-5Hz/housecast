#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
부동산 가격 예측 프로젝트 - KNN 모델
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

def train_model(X_train, y_train, tune_hyperparams=False):
    """
    KNN 모델을 훈련하는 함수
    
    Args:
        X_train (np.ndarray): 훈련 데이터의 특성 행렬
        y_train (np.ndarray): 훈련 데이터의 타겟 값
        tune_hyperparams (bool): 하이퍼파라미터 튜닝 여부
        
    Returns:
        tuple: (trained_model, feature_importance)
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
        model = KNeighborsRegressor(
            n_neighbors=5,
            weights='distance',
            p=2,
            leaf_size=30,
            n_jobs=-1
        )
        model.fit(X_train, y_train)
    
    end_time = time.time()
    train_time = end_time - start_time
    print(f"모델 훈련 완료! 훈련 시간: {train_time:.2f}초")
    
    # KNN은 특성 중요도 측정 방법이 내장되어 있지 않음
    # 특성 중요도 추정을 위한 순열 중요도(Permutation Importance) 사용
    print("특성 중요도 계산 중...")
    
    # 특성 이름이 필요하면 DataFrame으로 변환
    if isinstance(X_train, np.ndarray):
        try:
            feature_names = [f'feature_{i}' for i in range(X_train.shape[1])]
            X_train_df = pd.DataFrame(X_train, columns=feature_names)
        except:
            X_train_df = pd.DataFrame(X_train)
    else:
        X_train_df = X_train
    
    # 순열 중요도 계산
    from sklearn.inspection import permutation_importance
    perm_importance = permutation_importance(model, X_train, y_train, n_repeats=10, random_state=42)
    
    # 특성 중요도 저장
    feature_importance = pd.DataFrame({
        'feature': X_train_df.columns,
        'importance': perm_importance.importances_mean
    })
    feature_importance = feature_importance.sort_values('importance', ascending=False).reset_index(drop=True)
    
    # 상위 10개 특성 출력
    print("\n상위 10개 중요 특성:")
    for i, row in feature_importance.head(10).iterrows():
        print(f"{i+1}. {row['feature']}: {row['importance']:.4f}")
    
    return model, feature_importance

def analyze_k_values(X_train, y_train, X_val=None, y_val=None, max_k=30):
    """
    다양한 k 값에 대한 성능 분석
    
    Args:
        X_train (np.ndarray): 훈련 데이터의 특성 행렬
        y_train (np.ndarray): 훈련 데이터의 타겟 값
        X_val (np.ndarray): 검증 데이터의 특성 행렬 (선택적)
        y_val (np.ndarray): 검증 데이터의 타겟 값 (선택적)
        max_k (int): 실험할 최대 k 값
        
    Returns:
        dict: k 값에 따른 성능 지표
    """
    k_values = list(range(1, max_k + 1))
    train_rmse = []
    train_r2 = []
    val_rmse = []
    val_r2 = []
    
    print(f"K 값 분석 시작 (1부터 {max_k}까지)...")
    
    for k in k_values:
        model = KNeighborsRegressor(n_neighbors=k, weights='distance')
        model.fit(X_train, y_train)
        
        # 훈련 데이터 성능
        train_pred = model.predict(X_train)
        train_rmse.append(np.sqrt(mean_squared_error(y_train, train_pred)))
        train_r2.append(r2_score(y_train, train_pred))
        
        # 검증 데이터가 제공된 경우 검증 성능
        if X_val is not None and y_val is not None:
            val_pred = model.predict(X_val)
            val_rmse.append(np.sqrt(mean_squared_error(y_val, val_pred)))
            val_r2.append(r2_score(y_val, val_pred))
    
    # 결과 시각화
    plt.figure(figsize=(14, 7))
    
    plt.subplot(1, 2, 1)
    plt.plot(k_values, train_rmse, 'bo-', label='훈련 RMSE')
    if val_rmse:
        plt.plot(k_values, val_rmse, 'ro-', label='검증 RMSE')
    plt.xlabel('k 값')
    plt.ylabel('RMSE')
    plt.legend()
    plt.title('k 값에 따른 RMSE 변화')
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    plt.plot(k_values, train_r2, 'bo-', label='훈련 R²')
    if val_r2:
        plt.plot(k_values, val_r2, 'ro-', label='검증 R²')
    plt.xlabel('k 값')
    plt.ylabel('R²')
    plt.legend()
    plt.title('k 값에 따른 R² 변화')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('output/knn_k_analysis.png')
    print("K 값 분석 결과가 'output/knn_k_analysis.png'에 저장되었습니다.")
    
    # 최적의 k 값 찾기
    if val_rmse:
        best_k_idx = np.argmin(val_rmse)
        best_k = k_values[best_k_idx]
        print(f"검증 RMSE 기준 최적 k: {best_k} (RMSE: {val_rmse[best_k_idx]:.4f})")
    else:
        # 교차 검증 사용
        print("교차 검증을 통한 최적 k 탐색...")
        cv_scores = []
        for k in k_values:
            model = KNeighborsRegressor(n_neighbors=k, weights='distance')
            scores = -cross_val_score(model, X_train, y_train, cv=3, scoring='neg_root_mean_squared_error')
            cv_scores.append(scores.mean())
        
        best_k_idx = np.argmin(cv_scores)
        best_k = k_values[best_k_idx]
        print(f"교차 검증 RMSE 기준 최적 k: {best_k} (RMSE: {cv_scores[best_k_idx]:.4f})")
    
    return {
        'k_values': k_values,
        'train_rmse': train_rmse,
        'train_r2': train_r2,
        'val_rmse': val_rmse if val_rmse else None,
        'val_r2': val_r2 if val_r2 else None,
        'best_k': best_k
    }

def save_feature_importance_plot(feature_importance, output_path='output/feature_importance.png'):
    """
    특성 중요도를 시각화하여 저장하는 함수
    
    Args:
        feature_importance (pd.DataFrame): 특성 중요도 데이터프레임
        output_path (str): 저장할 파일 경로
    """
    try:
        plt.figure(figsize=(12, 8))
        plt.barh(feature_importance['feature'].head(20), feature_importance['importance'].head(20))
        plt.xlabel('중요도')
        plt.ylabel('특성')
        plt.title('KNN 모델 상위 20개 특성 중요도')
        plt.tight_layout()
        plt.savefig(output_path)
        print(f"특성 중요도 시각화가 {output_path}에 저장되었습니다.")
    except Exception as e:
        print(f"시각화 저장 중 오류 발생: {e}")

if __name__ == "__main__":
    # 테스트 실행을 위한 데이터 로드
    from data_loader import load_data
    from preprocessor import preprocess_data
    import os
    
    # 결과 저장 디렉토리 생성
    os.makedirs('output', exist_ok=True)
    
    # 데이터 로드 및 전처리
    train_data, test_data, bus_data, subway_data = load_data()
    X_train, y_train, X_test, processed_data = preprocess_data(train_data, test_data, bus_data, subway_data)
    
    # 다양한 k 값 분석
    k_analysis = analyze_k_values(X_train, y_train, max_k=30)
    
    # 최적 k를 적용한 모델 훈련
    best_k = k_analysis['best_k']
    print(f"\n최적 k={best_k}로 KNN 모델 재훈련...")
    
    # 기본 하이퍼파라미터로 모델 훈련
    model = KNeighborsRegressor(
        n_neighbors=best_k,
        weights='distance',
        p=2,
        leaf_size=30,
        n_jobs=-1
    )
    model.fit(X_train, y_train)
    
    # 순열 중요도 계산
    from sklearn.inspection import permutation_importance
    perm_importance = permutation_importance(model, X_train, y_train, n_repeats=10, random_state=42)
    
    # 특성 중요도 저장
    feature_names = processed_data['X_train'].columns
    feature_importance = pd.DataFrame({
        'feature': feature_names,
        'importance': perm_importance.importances_mean
    })
    feature_importance = feature_importance.sort_values('importance', ascending=False).reset_index(drop=True)
    
    # 특성 중요도 확인
    print("\n특성 중요도:")
    print(feature_importance.head(10))
    
    # 특성 중요도 시각화 저장
    save_feature_importance_plot(feature_importance) 