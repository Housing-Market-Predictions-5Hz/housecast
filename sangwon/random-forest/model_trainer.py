#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
부동산 가격 예측 프로젝트 - Random Forest 모델
모델 훈련 모듈

이 모듈은 랜덤 포레스트 모델을 훈련하고 특성 중요도를 분석합니다.
"""

import pandas as pd
import numpy as np
import time
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, KFold
from sklearn.metrics import mean_squared_error

def train_model(X_train, y_train, tune_hyperparams=False):
    """
    Random Forest 모델을 훈련하는 함수
    
    Args:
        X_train (np.ndarray): 훈련 데이터의 특성 행렬
        y_train (np.ndarray): 훈련 데이터의 타겟 값
        tune_hyperparams (bool): 하이퍼파라미터 튜닝 여부
        
    Returns:
        tuple: (trained_model, feature_importance)
    """
    print("Random Forest 모델 훈련 시작...")
    start_time = time.time()
    
    # 특성 이름이 필요하면 DataFrame으로 변환
    if isinstance(X_train, np.ndarray):
        try:
            feature_names = [f'feature_{i}' for i in range(X_train.shape[1])]
            X_train_df = pd.DataFrame(X_train, columns=feature_names)
        except:
            X_train_df = pd.DataFrame(X_train)
    else:
        X_train_df = X_train
    
    if tune_hyperparams:
        print("하이퍼파라미터 튜닝 시작...")
        
        # RandomizedSearchCV를 이용한 하이퍼파라미터 튜닝
        param_grid = {
            'n_estimators': [100, 200, 300, 500],
            'max_depth': [None, 10, 20, 30, 40, 50],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'max_features': ['sqrt', 'log2', None]
        }
        
        rf = RandomForestRegressor(random_state=42)
        random_search = RandomizedSearchCV(
            estimator=rf,
            param_distributions=param_grid,
            n_iter=20,
            cv=3,
            verbose=1,
            random_state=42,
            n_jobs=-1
        )
        
        random_search.fit(X_train, y_train)
        model = random_search.best_estimator_
        
        print(f"최적 하이퍼파라미터: {random_search.best_params_}")
        print(f"최적 CV 점수: {random_search.best_score_:.4f}")
    else:
        # 기본 하이퍼파라미터로 모델 훈련
        # model = RandomForestRegressor(
        #     n_estimators=300,
        #     max_depth=None,
        #     min_samples_split=2,
        #     min_samples_leaf=1,
        #     max_features='sqrt',
        #     n_jobs=-1,
        #     random_state=42
        # )
        model = RandomForestRegressor(n_estimators=5, criterion='squared_error', random_state=1, n_jobs=-1)
        model.fit(X_train, y_train)
    
    end_time = time.time()
    train_time = end_time - start_time
    print(f"모델 훈련 완료! 훈련 시간: {train_time:.2f}초")
    
    # 특성 중요도 분석
    feature_importance = pd.DataFrame({
        'feature': X_train_df.columns,
        'importance': model.feature_importances_
    })
    feature_importance = feature_importance.sort_values('importance', ascending=False).reset_index(drop=True)
    
    # 상위 10개 특성 출력
    print("\n상위 10개 중요 특성:")
    for i, row in feature_importance.head(10).iterrows():
        print(f"{i+1}. {row['feature']}: {row['importance']:.4f}")
    
    return model, feature_importance

def train_model_with_kfold(X, y, n_splits=5):
    """
    K-Fold 교차 검증을 사용하여 Random Forest 모델을 훈련하는 함수
    
    Args:
        X (pd.DataFrame or np.ndarray): 전체 데이터의 특성 행렬
        y (pd.Series or np.ndarray): 전체 데이터의 타겟 값
        n_splits (int): K-Fold의 분할 수
        
    Returns:
        list: 각 Fold의 성능 지표 (RMSE)
    """
    print("K-Fold 교차 검증 시작...")
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    fold_results = []
    
    for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
        print(f"\nFold {fold + 1}/{n_splits}")
        
        # 데이터 분할
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        
        # 모델 초기화
        model = RandomForestRegressor(random_state=42)
        
        # 모델 훈련
        model.fit(X_train, y_train)
        
        # 검증 데이터 예측
        y_val_pred = model.predict(X_val)
        
        # RMSE 계산 (squared=False 대신 수동 계산)
        mse = mean_squared_error(y_val, y_val_pred)
        rmse = np.sqrt(mse)
        fold_results.append(rmse)
        
        print(f"Fold {fold + 1} RMSE: {rmse:.4f}")
    
    print("\nK-Fold 교차 검증 완료!")
    print(f"평균 RMSE: {np.mean(fold_results):.4f}")
    return fold_results

def save_feature_importance_plot(feature_importance, output_path='output/feature_importance.png'):
    """
    특성 중요도를 시각화하여 저장하는 함수
    
    Args:
        feature_importance (pd.DataFrame): 특성 중요도 데이터프레임
        output_path (str): 저장할 파일 경로
    """
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
        
        plt.figure(figsize=(12, 8))
        sns.barplot(x='importance', y='feature', data=feature_importance.head(20))
        plt.title('Random Forest 상위 20개 특성 중요도')
        plt.tight_layout()
        plt.savefig(output_path)
        print(f"특성 중요도 시각화가 {output_path}에 저장되었습니다.")
    except Exception as e:
        print(f"시각화 저장 중 오류 발생: {e}")

if __name__ == "__main__":
    # 테스트 실행을 위한 데이터 로드
    from data_loader import load_data
    from preprocessor import preprocess_data
    
    # 데이터 로드 및 전처리
    train_data, test_data, bus_data, subway_data = load_data()
    X_train, y_train, X_test, processed_data = preprocess_data(train_data, test_data, bus_data, subway_data)
    
    # 모델 훈련 테스트
    model, feature_importance = train_model(X_train, y_train, tune_hyperparams=False)
    
    # 특성 중요도 확인
    print("\n특성 중요도:")
    print(feature_importance.head(10))
    
    # 특성 중요도 시각화 저장
    save_feature_importance_plot(feature_importance)