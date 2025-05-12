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

def train_linear_models(X_train, y_train, tune_hyperparams=False):
    """
    다양한 선형 회귀 모델을 훈련하고 비교하는 함수
    
    Args:
        X_train (np.ndarray): 훈련 데이터의 특성 행렬
        y_train (np.ndarray): 훈련 데이터의 타겟 값
        tune_hyperparams (bool): 하이퍼파라미터 튜닝 여부
        
    Returns:
        dict: 모델별 훈련 결과
    """
    print("선형 회귀 모델 훈련 시작...")
    start_time = time.time()
    
    results = {}
    
    # 1. 기본 선형 회귀
    print("\n1. 기본 선형 회귀 모델 훈련")
    lr = LinearRegression()
    lr.fit(X_train, y_train)
    
    lr_pred = lr.predict(X_train)
    lr_rmse = np.sqrt(mean_squared_error(y_train, lr_pred))
    lr_r2 = r2_score(y_train, lr_pred)
    lr_mae = mean_absolute_error(y_train, lr_pred)
    
    print(f"선형 회귀 - RMSE: {lr_rmse:.4f}, MAE: {lr_mae:.4f}, R²: {lr_r2:.4f}")
    results['linear_regression'] = {
        'model': lr,
        'rmse': lr_rmse,
        'r2': lr_r2,
        'mae': lr_mae
    }
    
    # 2. Ridge 회귀
    if tune_hyperparams:
        print("\n2. Ridge 회귀 - 하이퍼파라미터 튜닝")
        ridge_params = {'alpha': [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]}
        ridge_grid = GridSearchCV(Ridge(), ridge_params, cv=3, scoring='neg_root_mean_squared_error')
        ridge_grid.fit(X_train, y_train)
        
        ridge = ridge_grid.best_estimator_
        best_alpha = ridge_grid.best_params_['alpha']
        print(f"Ridge 최적 alpha: {best_alpha}")
    else:
        print("\n2. Ridge 회귀 - 기본 하이퍼파라미터")
        ridge = Ridge(alpha=1.0)
        ridge.fit(X_train, y_train)
    
    ridge_pred = ridge.predict(X_train)
    ridge_rmse = np.sqrt(mean_squared_error(y_train, ridge_pred))
    ridge_r2 = r2_score(y_train, ridge_pred)
    ridge_mae = mean_absolute_error(y_train, ridge_pred)
    
    print(f"Ridge 회귀 - RMSE: {ridge_rmse:.4f}, MAE: {ridge_mae:.4f}, R²: {ridge_r2:.4f}")
    results['ridge'] = {
        'model': ridge,
        'rmse': ridge_rmse,
        'r2': ridge_r2,
        'mae': ridge_mae
    }
    
    # 3. Lasso 회귀
    if tune_hyperparams:
        print("\n3. Lasso 회귀 - 하이퍼파라미터 튜닝")
        lasso_params = {'alpha': [0.0001, 0.001, 0.01, 0.1, 1.0, 10.0]}
        lasso_grid = GridSearchCV(Lasso(max_iter=10000), lasso_params, cv=3, scoring='neg_root_mean_squared_error')
        lasso_grid.fit(X_train, y_train)
        
        lasso = lasso_grid.best_estimator_
        best_alpha = lasso_grid.best_params_['alpha']
        print(f"Lasso 최적 alpha: {best_alpha}")
    else:
        print("\n3. Lasso 회귀 - 기본 하이퍼파라미터")
        lasso = Lasso(alpha=0.1, max_iter=10000)
        lasso.fit(X_train, y_train)
    
    lasso_pred = lasso.predict(X_train)
    lasso_rmse = np.sqrt(mean_squared_error(y_train, lasso_pred))
    lasso_r2 = r2_score(y_train, lasso_pred)
    lasso_mae = mean_absolute_error(y_train, lasso_pred)
    
    print(f"Lasso 회귀 - RMSE: {lasso_rmse:.4f}, MAE: {lasso_mae:.4f}, R²: {lasso_r2:.4f}")
    results['lasso'] = {
        'model': lasso,
        'rmse': lasso_rmse,
        'r2': lasso_r2,
        'mae': lasso_mae
    }
    
    # 4. ElasticNet 회귀
    if tune_hyperparams:
        print("\n4. ElasticNet 회귀 - 하이퍼파라미터 튜닝")
        elastic_params = {
            'alpha': [0.001, 0.01, 0.1, 1.0, 10.0],
            'l1_ratio': [0.1, 0.3, 0.5, 0.7, 0.9]
        }
        elastic_grid = GridSearchCV(ElasticNet(max_iter=10000), elastic_params, cv=3, scoring='neg_root_mean_squared_error')
        elastic_grid.fit(X_train, y_train)
        
        elastic = elastic_grid.best_estimator_
        best_alpha = elastic_grid.best_params_['alpha']
        best_l1_ratio = elastic_grid.best_params_['l1_ratio']
        print(f"ElasticNet 최적 alpha: {best_alpha}, l1_ratio: {best_l1_ratio}")
    else:
        print("\n4. ElasticNet 회귀 - 기본 하이퍼파라미터")
        elastic = ElasticNet(alpha=0.1, l1_ratio=0.5, max_iter=10000)
        elastic.fit(X_train, y_train)
    
    elastic_pred = elastic.predict(X_train)
    elastic_rmse = np.sqrt(mean_squared_error(y_train, elastic_pred))
    elastic_r2 = r2_score(y_train, elastic_pred)
    elastic_mae = mean_absolute_error(y_train, elastic_pred)
    
    print(f"ElasticNet 회귀 - RMSE: {elastic_rmse:.4f}, MAE: {elastic_mae:.4f}, R²: {elastic_r2:.4f}")
    results['elastic_net'] = {
        'model': elastic,
        'rmse': elastic_rmse,
        'r2': elastic_r2,
        'mae': elastic_mae
    }
    
    # 5. 다항 회귀
    print("\n5. 다항 회귀 모델 훈련")
    poly_degree = 2  # 2차 다항식
    
    # 특성 수가 많을 수 있으므로 일부 특성만 다항 변환
    # 데이터 차원이 너무 커지는 것을 방지하기 위함
    top_features = min(10, X_train.shape[1])  # 최대 10개 특성 사용
    X_train_subset = X_train[:, :top_features]
    
    poly = PolynomialFeatures(degree=poly_degree, include_bias=False)
    X_train_poly = poly.fit_transform(X_train_subset)
    
    # 다항 변환된 특성으로 선형 회귀 모델 훈련
    lr_poly = LinearRegression()
    lr_poly.fit(X_train_poly, y_train)
    
    poly_pred = lr_poly.predict(X_train_poly)
    poly_rmse = np.sqrt(mean_squared_error(y_train, poly_pred))
    poly_r2 = r2_score(y_train, poly_pred)
    poly_mae = mean_absolute_error(y_train, poly_pred)
    
    print(f"다항 회귀 (차수={poly_degree}) - RMSE: {poly_rmse:.4f}, MAE: {poly_mae:.4f}, R²: {poly_r2:.4f}")
    results['polynomial'] = {
        'model': {
            'poly': poly,
            'lr': lr_poly
        },
        'degree': poly_degree,
        'rmse': poly_rmse,
        'r2': poly_r2,
        'mae': poly_mae,
        'num_features': top_features
    }
    
    end_time = time.time()
    train_time = end_time - start_time
    print(f"\n모델 훈련 완료! 훈련 시간: {train_time:.2f}초")
    
    # 결과 요약 및 시각화
    plot_model_comparison(results)
    
    return results

def plot_model_comparison(results):
    """
    모델 성능 비교 시각화
    
    Args:
        results (dict): 모델별 훈련 결과
    """
    models = list(results.keys())
    rmse_values = [results[model]['rmse'] for model in models]
    r2_values = [results[model]['r2'] for model in models]
    mae_values = [results[model]['mae'] for model in models]
    
    # 모델 이름 변환
    model_names = {
        'linear_regression': '선형 회귀',
        'ridge': 'Ridge',
        'lasso': 'Lasso',
        'elastic_net': 'ElasticNet',
        'polynomial': '다항 회귀'
    }
    
    plot_names = [model_names.get(model, model) for model in models]
    
    # RMSE 비교
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    plt.bar(plot_names, rmse_values, color='skyblue')
    plt.title('RMSE 비교 (낮을수록 좋음)')
    plt.ylabel('RMSE')
    plt.xticks(rotation=45)
    
    # MAE 비교
    plt.subplot(1, 3, 2)
    plt.bar(plot_names, mae_values, color='lightgreen')
    plt.title('MAE 비교 (낮을수록 좋음)')
    plt.ylabel('MAE')
    plt.xticks(rotation=45)
    
    # R² 비교
    plt.subplot(1, 3, 3)
    plt.bar(plot_names, r2_values, color='salmon')
    plt.title('R² 비교 (높을수록 좋음)')
    plt.ylabel('R²')
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    plt.savefig('output/linear_models_comparison.png')
    print("모델 비교 결과가 'output/linear_models_comparison.png'에 저장되었습니다.")

def analyze_coefficients(results, feature_names=None):
    """
    선형 모델의 계수 분석
    
    Args:
        results (dict): 모델별 훈련 결과
        feature_names (list): 특성 이름 목록
    """
    if feature_names is None:
        # 특성 이름이 없으면 인덱스로 대체
        feature_names = [f'feature_{i}' for i in range(len(results['linear_regression']['model'].coef_))]
    
    # 1. 일반 선형 회귀 계수
    lr_coef = results['linear_regression']['model'].coef_
    
    # 2. Ridge 계수
    ridge_coef = results['ridge']['model'].coef_
    
    # 3. Lasso 계수 (특성 선택 역할도 함)
    lasso_coef = results['lasso']['model'].coef_
    
    # 계수 데이터프레임 생성
    coef_df = pd.DataFrame({
        'feature': feature_names,
        'linear': lr_coef,
        'ridge': ridge_coef,
        'lasso': lasso_coef
    })
    
    # 절대값 기준 상위 10개 특성 찾기
    coef_abs = np.abs(lr_coef)
    top_indices = np.argsort(coef_abs)[-10:]
    top_features = [feature_names[i] for i in top_indices]
    top_coef_df = coef_df[coef_df['feature'].isin(top_features)]
    
    # 결과 저장
    coef_df.to_csv('output/linear_coefficients.csv', index=False)
    print("계수 분석 결과가 'output/linear_coefficients.csv'에 저장되었습니다.")
    
    # 상위 특성 시각화
    plt.figure(figsize=(12, 8))
    
    # 3개 모델의 계수를 나란히 표시
    for i, model in enumerate(['linear', 'ridge', 'lasso']):
        plt.subplot(3, 1, i+1)
        sorted_df = top_coef_df.sort_values(model)
        plt.barh(sorted_df['feature'], sorted_df[model])
        plt.title(f'{model.capitalize()} 회귀 상위 특성 계수')
    
    plt.tight_layout()
    plt.savefig('output/top_coefficients.png')
    print("상위 특성 계수 시각화가 'output/top_coefficients.png'에 저장되었습니다.")
    
    # Lasso를 통한 특성 선택 결과
    nonzero_lasso = coef_df[np.abs(coef_df['lasso']) > 1e-5]
    print(f"\nLasso가 선택한 특성 수: {len(nonzero_lasso)}/{len(feature_names)}")
    
    return coef_df

def compare_with_other_models(X_train, y_train, linear_results, other_models, fold=3):
    """
    선형 모델과 다른 모델(Random Forest, KNN 등)의 성능 비교
    
    Args:
        X_train (np.ndarray): 훈련 데이터의 특성 행렬
        y_train (np.ndarray): 훈련 데이터의 타겟 값
        linear_results (dict): 선형 모델 훈련 결과
        other_models (dict): 다른 모델들의 dictionary
        fold (int): 교차 검증 폴드 수
        
    Returns:
        dict: 비교 결과
    """
    print(f"\n{fold}겹 교차 검증을 통한 모델 비교 시작...")
    
    all_models = {}
    
    # 선형 모델 중 가장 좋은 모델 선택
    best_linear_model = None
    best_linear_name = None
    best_linear_rmse = float('inf')
    
    for model_name, model_info in linear_results.items():
        if model_info['rmse'] < best_linear_rmse:
            best_linear_rmse = model_info['rmse']
            best_linear_model = model_info['model']
            best_linear_name = model_name
    
    if best_linear_name == 'polynomial':
        # 다항 회귀의 경우 파이프라인 생성
        poly = best_linear_model['poly']
        lr = best_linear_model['lr']
        num_features = linear_results['polynomial']['num_features']
        
        def predict_poly(X):
            return lr.predict(poly.transform(X[:, :num_features]))
        
        # 교차 검증을 위한 래퍼 모델 생성
        from sklearn.base import BaseEstimator
        class PolyWrapper(BaseEstimator):
            def __init__(self, poly, lr, num_features):
                self.poly = poly
                self.lr = lr
                self.num_features = num_features
            
            def fit(self, X, y):
                X_poly = self.poly.fit_transform(X[:, :self.num_features])
                self.lr.fit(X_poly, y)
                return self
            
            def predict(self, X):
                X_poly = self.poly.transform(X[:, :self.num_features])
                return self.lr.predict(X_poly)
        
        best_linear_model = PolyWrapper(poly, lr, num_features)
    
    # 모델 이름 변환
    model_names = {
        'linear_regression': '선형 회귀',
        'ridge': 'Ridge',
        'lasso': 'Lasso',
        'elastic_net': 'ElasticNet',
        'polynomial': '다항 회귀',
        'random_forest': 'Random Forest',
        'knn': 'KNN'
    }
    
    # 최종 비교 모델 목록
    all_models[model_names.get(best_linear_name, best_linear_name)] = best_linear_model
    
    for model_name, model in other_models.items():
        all_models[model_names.get(model_name, model_name)] = model
    
    # 교차 검증 및 결과 저장
    compare_results = {}
    
    for model_name, model in all_models.items():
        rmse_scores = -cross_val_score(model, X_train, y_train,
                                      cv=fold, scoring='neg_root_mean_squared_error')
        r2_scores = cross_val_score(model, X_train, y_train,
                                   cv=fold, scoring='r2')
        
        compare_results[model_name] = {
            'rmse': {
                'mean': rmse_scores.mean(),
                'std': rmse_scores.std()
            },
            'r2': {
                'mean': r2_scores.mean(),
                'std': r2_scores.std()
            }
        }
        
        print(f"{model_name} - 평균 RMSE: {rmse_scores.mean():.4f} ± {rmse_scores.std():.4f}")
        print(f"{model_name} - 평균 R²: {r2_scores.mean():.4f} ± {r2_scores.std():.4f}")
    
    # 결과 시각화
    plt.figure(figsize=(14, 6))
    
    # RMSE 비교
    plt.subplot(1, 2, 1)
    models = list(compare_results.keys())
    rmse_means = [compare_results[model]['rmse']['mean'] for model in models]
    rmse_stds = [compare_results[model]['rmse']['std'] for model in models]
    
    bars = plt.bar(models, rmse_means, yerr=rmse_stds, capsize=10)
    plt.ylabel('RMSE (낮을수록 좋음)')
    plt.title('모델 간 RMSE 비교')
    plt.xticks(rotation=45)
    plt.grid(axis='y')
    
    # R² 비교
    plt.subplot(1, 2, 2)
    r2_means = [compare_results[model]['r2']['mean'] for model in models]
    r2_stds = [compare_results[model]['r2']['std'] for model in models]
    
    bars = plt.bar(models, r2_means, yerr=r2_stds, capsize=10)
    plt.ylabel('R² (높을수록 좋음)')
    plt.title('모델 간 R² 비교')
    plt.xticks(rotation=45)
    plt.grid(axis='y')
    
    plt.tight_layout()
    plt.savefig('output/model_comparison.png')
    print("모델 비교 결과가 'output/model_comparison.png'에 저장되었습니다.")
    
    return compare_results

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
    
    # 선형 모델 훈련
    linear_results = train_linear_models(X_train, y_train, tune_hyperparams=False)
    
    # 특성 이름 (데이터프레임에서 가져옴)
    feature_names = processed_data['X_train'].columns.tolist()
    
    # 계수 분석
    coef_analysis = analyze_coefficients(linear_results, feature_names)
    
    # 다른 모델과 비교 (다른 모델이 있는 경우)
    try:
        from model_trainer import train_model
        from knn_model import train_knn_model
        
        print("\n다른 모델과 성능 비교를 위한 모델 훈련...")
        
        # Random Forest 모델 훈련
        rf_model, _ = train_model(X_train, y_train, tune_hyperparams=False)
        
        # KNN 모델 훈련
        knn_model, _, _ = train_knn_model(X_train, y_train, tune_hyperparams=False)
        
        # 모델 비교
        other_models = {
            'random_forest': rf_model,
            'knn': knn_model
        }
        
        comparison = compare_with_other_models(X_train, y_train, linear_results, other_models)
        
    except ImportError:
        print("\n다른 모델 파일을 찾을 수 없어 비교를 건너뜁니다.")
    
    print("\n부동산 가격 예측에 관한 선형 모델 분석 결론:")
    print("1. 선형 모델은 단순하고 해석이 용이하며 계산이 빠릅니다.")
    print("2. Ridge, Lasso, ElasticNet 같은 정규화 모델은 과적합을 방지하고 특성 선택에 도움을 줍니다.")
    print("3. 다항 회귀는 비선형 관계를 모델링할 수 있지만, 고차원 데이터에서는 성능이 제한될 수 있습니다.")
    print("4. 부동산 데이터는 복잡한 비선형 관계를 포함하므로 Random Forest 같은 트리 기반 모델이 더 효과적일 수 있습니다.")
    print("5. 하지만 특성의 영향력 해석이 중요하다면 선형 모델이 더 유용할 수 있습니다.") 