#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
부동산 가격 예측 프로젝트 - 선형 회귀 모델
모델 훈련 모듈

이 모듈은 다양한 선형 회귀 모델을 훈련하고 비교합니다.
"""

import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

def train_model(X_train, y_train, tune_hyperparams=False, model_type='ridge'):
    """
    선형 회귀 모델을 훈련하는 함수
    
    Args:
        X_train (np.ndarray): 훈련 데이터의 특성 행렬
        y_train (np.ndarray): 훈련 데이터의 타겟 값
        tune_hyperparams (bool): 하이퍼파라미터 튜닝 여부
        model_type (str): 모델 타입 ('linear', 'ridge', 'lasso', 'elastic_net', 'polynomial')
        
    Returns:
        tuple: (trained_model, feature_importance)
    """
    print(f"선형 회귀 모델({model_type}) 훈련 시작...")
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
        
    # 모델 선택 및 훈련
    if model_type == 'linear':
        model = LinearRegression()
        model.fit(X_train, y_train)
        
    elif model_type == 'ridge':
        if tune_hyperparams:
            print("Ridge 하이퍼파라미터 튜닝 중...")
            param_grid = {'alpha': [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]}
            grid_search = GridSearchCV(Ridge(), param_grid, cv=3, scoring='neg_root_mean_squared_error', n_jobs=2)
            grid_search.fit(X_train, y_train)
            model = grid_search.best_estimator_
            print(f"최적 alpha: {grid_search.best_params_['alpha']}")
        else:
            model = Ridge(alpha=1.0)
            model.fit(X_train, y_train)
            
    elif model_type == 'lasso':
        if tune_hyperparams:
            print("Lasso 하이퍼파라미터 튜닝 중...")
            param_grid = {'alpha': [0.0001, 0.001, 0.01, 0.1, 1.0, 10.0]}
            grid_search = GridSearchCV(Lasso(max_iter=10000), param_grid, cv=3, scoring='neg_root_mean_squared_error', n_jobs=2)
            grid_search.fit(X_train, y_train)
            model = grid_search.best_estimator_
            print(f"최적 alpha: {grid_search.best_params_['alpha']}")
        else:
            model = Lasso(alpha=0.1, max_iter=10000)
            model.fit(X_train, y_train)
            
    elif model_type == 'elastic_net':
        if tune_hyperparams:
            print("ElasticNet 하이퍼파라미터 튜닝 중...")
            param_grid = {
                'alpha': [0.001, 0.01, 0.1, 1.0, 10.0],
                'l1_ratio': [0.1, 0.3, 0.5, 0.7, 0.9]
            }
            grid_search = GridSearchCV(ElasticNet(max_iter=10000), param_grid, cv=3, scoring='neg_root_mean_squared_error', n_jobs=2)
            grid_search.fit(X_train, y_train)
            model = grid_search.best_estimator_
            print(f"최적 alpha: {grid_search.best_params_['alpha']}, l1_ratio: {grid_search.best_params_['l1_ratio']}")
        else:
            model = ElasticNet(alpha=0.1, l1_ratio=0.5, max_iter=10000)
            model.fit(X_train, y_train)
            
    elif model_type == 'polynomial':
        print("다항 회귀 모델 훈련 중...")
        poly_degree = 2  # 2차 다항식
        
        # 특성 수가 많을 수 있으므로 일부 특성만 다항 변환
        top_features = min(10, X_train.shape[1])  # 최대 10개 특성 사용
        X_train_subset = X_train[:, :top_features]
        
        poly = PolynomialFeatures(degree=poly_degree, include_bias=False)
        X_train_poly = poly.fit_transform(X_train_subset)
        
        # 다항 변환된 특성으로 선형 회귀 모델 훈련
        lr_poly = LinearRegression()
        lr_poly.fit(X_train_poly, y_train)
        
        # 복합 모델 반환
        model = {
            'poly': poly,
            'lr': lr_poly,
            'num_features': top_features
        }
    else:
        raise ValueError(f"지원되지 않는 모델 타입: {model_type}")
    
    end_time = time.time()
    train_time = end_time - start_time
    print(f"모델 훈련 완료! 훈련 시간: {train_time:.2f}초")
    
    # 특성 중요도 계산 (계수의 절대값 기준)
    if model_type == 'polynomial':
        # 다항 회귀의 경우 변환된 특성들에 대한 계수 사용
        poly_feature_names = poly.get_feature_names_out(X_train_df.columns[:top_features])
        feature_importance = pd.DataFrame({
            'feature': poly_feature_names,
            'importance': np.abs(model['lr'].coef_)
        })
    else:
        # 일반 선형 모델의 경우 계수 사용
        feature_importance = pd.DataFrame({
            'feature': X_train_df.columns,
            'importance': np.abs(model.coef_)
        })
    
    feature_importance = feature_importance.sort_values('importance', ascending=False).reset_index(drop=True)
    
    # 상위 10개 특성 출력
    print("\n상위 10개 중요 특성:")
    for i, row in feature_importance.head(10).iterrows():
        print(f"{i+1}. {row['feature']}: {row['importance']:.4f}")
    
    return model, feature_importance

def compare_linear_models(X_train, y_train, tune_hyperparams=False):
    """
    다양한 선형 회귀 모델을 훈련하고 비교하는 함수
    
    Args:
        X_train (np.ndarray): 훈련 데이터의 특성 행렬
        y_train (np.ndarray): 훈련 데이터의 타겟 값
        tune_hyperparams (bool): 하이퍼파라미터 튜닝 여부
        
    Returns:
        dict: 모델별 훈련 결과
    """
    print("모델 비교 시작...")
    results = {}
    
    # 1. 기본 선형 회귀
    linear_model, linear_importance = train_model(X_train, y_train, tune_hyperparams, model_type='linear')
    linear_pred = linear_model.predict(X_train)
    linear_rmse = np.sqrt(mean_squared_error(y_train, linear_pred))
    linear_r2 = r2_score(y_train, linear_pred)
    
    results['linear'] = {
        'model': linear_model,
        'importance': linear_importance,
        'rmse': linear_rmse,
        'r2': linear_r2
    }
    
    # 2. Ridge 회귀
    ridge_model, ridge_importance = train_model(X_train, y_train, tune_hyperparams, model_type='ridge')
    ridge_pred = ridge_model.predict(X_train)
    ridge_rmse = np.sqrt(mean_squared_error(y_train, ridge_pred))
    ridge_r2 = r2_score(y_train, ridge_pred)
    
    results['ridge'] = {
        'model': ridge_model,
        'importance': ridge_importance,
        'rmse': ridge_rmse,
        'r2': ridge_r2
    }
    
    # 3. Lasso 회귀
    lasso_model, lasso_importance = train_model(X_train, y_train, tune_hyperparams, model_type='lasso')
    lasso_pred = lasso_model.predict(X_train)
    lasso_rmse = np.sqrt(mean_squared_error(y_train, lasso_pred))
    lasso_r2 = r2_score(y_train, lasso_pred)
    
    results['lasso'] = {
        'model': lasso_model,
        'importance': lasso_importance,
        'rmse': lasso_rmse,
        'r2': lasso_r2
    }
    
    # 4. ElasticNet 회귀
    elastic_model, elastic_importance = train_model(X_train, y_train, tune_hyperparams, model_type='elastic_net')
    elastic_pred = elastic_model.predict(X_train)
    elastic_rmse = np.sqrt(mean_squared_error(y_train, elastic_pred))
    elastic_r2 = r2_score(y_train, elastic_pred)
    
    results['elastic_net'] = {
        'model': elastic_model,
        'importance': elastic_importance,
        'rmse': elastic_rmse,
        'r2': elastic_r2
    }
    
    # 5. 다항 회귀
    poly_model, poly_importance = train_model(X_train, y_train, tune_hyperparams, model_type='polynomial')
    
    # 다항 회귀 예측
    num_features = poly_model['num_features']
    X_train_subset = X_train[:, :num_features]
    X_train_poly = poly_model['poly'].transform(X_train_subset)
    poly_pred = poly_model['lr'].predict(X_train_poly)
    
    poly_rmse = np.sqrt(mean_squared_error(y_train, poly_pred))
    poly_r2 = r2_score(y_train, poly_pred)
    
    results['polynomial'] = {
        'model': poly_model,
        'importance': poly_importance,
        'rmse': poly_rmse,
        'r2': poly_r2
    }
    
    # 결과 비교
    print("\n모델 성능 비교:")
    for model_name, model_result in results.items():
        print(f"{model_name} - RMSE: {model_result['rmse']:.4f}, R²: {model_result['r2']:.4f}")
    
    # 시각화
    model_names = list(results.keys())
    rmse_values = [results[model]['rmse'] for model in model_names]
    r2_values = [results[model]['r2'] for model in model_names]
    
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.bar(model_names, rmse_values)
    plt.ylabel('RMSE (낮을수록 좋음)')
    plt.title('모델별 RMSE')
    
    plt.subplot(1, 2, 2)
    plt.bar(model_names, r2_values)
    plt.ylabel('R² (높을수록 좋음)')
    plt.title('모델별 R²')
    
    plt.tight_layout()
    plt.savefig('output/linear_models_comparison.png')
    print("모델 비교 결과가 'output/linear_models_comparison.png'에 저장되었습니다.")
    
    # 최적 모델 선택
    best_model = min(results.items(), key=lambda x: x[1]['rmse'])[0]
    print(f"\n최적 모델: {best_model} (RMSE: {results[best_model]['rmse']:.4f})")
    
    return results, best_model

def save_feature_importance_plot(feature_importance, output_path='output/feature_importance.png'):
    """
    특성 중요도를 시각화하여 저장하는 함수
    
    Args:
        feature_importance (pd.DataFrame): 특성 중요도 데이터프레임
        output_path (str): 저장할 파일 경로
    """
    try:
        plt.figure(figsize=(12, 8))
        top_features = feature_importance.head(20)
        plt.barh(top_features['feature'], top_features['importance'])
        plt.xlabel('중요도 (계수 절대값)')
        plt.ylabel('특성')
        plt.title('선형 회귀 상위 20개 특성 중요도')
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
    
    # 모델 비교
    results, best_model_type = compare_linear_models(X_train, y_train, tune_hyperparams=False)
    
    # 최적 모델로 최종 훈련
    print(f"\n최적 모델({best_model_type})로 최종 훈련...")
    best_model, feature_importance = train_model(X_train, y_train, tune_hyperparams=True, model_type=best_model_type)
    
    # 특성 중요도 확인
    print("\n특성 중요도:")
    print(feature_importance.head(10))
    
    # 특성 중요도 시각화 저장
    save_feature_importance_plot(feature_importance) 