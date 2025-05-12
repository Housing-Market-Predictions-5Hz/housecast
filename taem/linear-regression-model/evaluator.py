#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
부동산 가격 예측 프로젝트 - 선형 회귀 모델
모델 평가 모듈

이 모듈은 훈련된 모델을 평가하고 성능 지표를 계산합니다.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import cross_val_score

def evaluate_model(model, X_train, y_train, X_val=None, y_val=None, model_type='linear'):
    """
    모델 성능을 평가하는 함수
    
    Args:
        model: 훈련된 모델 객체
        X_train (np.ndarray): 훈련 데이터 특성
        y_train (np.ndarray): 훈련 데이터 타겟
        X_val (np.ndarray, optional): 검증 데이터 특성
        y_val (np.ndarray, optional): 검증 데이터 타겟
        model_type (str): 모델 타입 ('linear', 'ridge', 'lasso', 'elastic_net', 'polynomial')
        
    Returns:
        dict: 평가 지표 결과
    """
    results = {}
    
    # 다항 회귀 모델 특별 처리
    if model_type == 'polynomial':
        # 훈련 데이터 예측
        num_features = model['num_features']
        X_train_subset = X_train[:, :num_features]
        X_train_poly = model['poly'].transform(X_train_subset)
        train_pred = model['lr'].predict(X_train_poly)
        
        # 검증 데이터 예측
        if X_val is not None and y_val is not None:
            X_val_subset = X_val[:, :num_features]
            X_val_poly = model['poly'].transform(X_val_subset)
            val_pred = model['lr'].predict(X_val_poly)
    else:
        # 일반 선형 모델 예측
        train_pred = model.predict(X_train)
        val_pred = model.predict(X_val) if X_val is not None and y_val is not None else None
    
    # 훈련 데이터에 대한 성능 평가
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
    
    # 다항 회귀 모델은 교차 검증 생략 (복잡함)
    if model_type != 'polynomial':
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
        
        # QQ 플롯 (잔차 정규성 확인)
        from scipy import stats
        plt.figure(figsize=(10, 6))
        stats.probplot(residuals, dist="norm", plot=plt)
        plt.title('잔차 QQ 플롯')
        plt.tight_layout()
        plt.savefig(f'{save_path}/residuals_qq_plot.png')
        
        print(f"평가 그래프가 '{save_path}' 디렉토리에 저장되었습니다.")
    
    except Exception as e:
        print(f"그래프 생성 중 오류 발생: {e}")

def analyze_coefficients(model, feature_names, output_path='output'):
    """
    선형 모델의 계수를 분석하는 함수
    
    Args:
        model: 훈련된 모델 객체
        feature_names (list): 특성 이름 목록
        output_path (str): 결과 저장 경로
    """
    try:
        # 계수 추출
        if hasattr(model, 'coef_'):
            coef = model.coef_
        else:
            print("모델에 계수가 없습니다.")
            return
        
        # 계수 정보 저장
        coef_df = pd.DataFrame({
            'feature': feature_names,
            'coefficient': coef,
            'abs_coefficient': np.abs(coef)
        })
        
        # 절대값 기준 내림차순 정렬
        coef_df = coef_df.sort_values('abs_coefficient', ascending=False)
        
        # 계수 정보 저장
        coef_df.to_csv(f'{output_path}/coefficients.csv', index=False)
        
        # 상위 20개 계수 시각화
        plt.figure(figsize=(12, 8))
        top_coef = coef_df.head(20)
        
        # 계수 값 기준 정렬
        top_coef = top_coef.sort_values('coefficient')
        
        plt.barh(top_coef['feature'], top_coef['coefficient'])
        plt.xlabel('계수 값')
        plt.ylabel('특성')
        plt.title('선형 모델 상위 20개 계수')
        plt.axvline(x=0, color='r', linestyle='--')
        plt.tight_layout()
        plt.savefig(f'{output_path}/top_coefficients.png')
        
        print(f"계수 분석 결과가 '{output_path}' 디렉토리에 저장되었습니다.")
        
        return coef_df
    
    except Exception as e:
        print(f"계수 분석 중 오류 발생: {e}")

def compare_models(models_dict, X_train, y_train, save_path='output'):
    """
    여러 모델의 성능을 비교하는 함수
    
    Args:
        models_dict (dict): 모델 이름과 객체가 담긴 딕셔너리
        X_train (np.ndarray): 훈련 데이터 특성
        y_train (np.ndarray): 훈련 데이터 타겟
        save_path (str): 결과 저장 경로
    """
    results = {}
    
    for model_name, model_info in models_dict.items():
        model = model_info['model']
        model_type = model_info['type']
        
        print(f"\n{model_name} 모델 평가 중...")
        
        # 모델 평가
        if model_type == 'polynomial':
            # 다항 회귀 모델 특별 처리
            num_features = model['num_features']
            X_train_subset = X_train[:, :num_features]
            X_train_poly = model['poly'].transform(X_train_subset)
            y_pred = model['lr'].predict(X_train_poly)
        else:
            # 일반 선형 모델
            y_pred = model.predict(X_train)
        
        # 성능 지표 계산
        rmse = np.sqrt(mean_squared_error(y_train, y_pred))
        mae = mean_absolute_error(y_train, y_pred)
        r2 = r2_score(y_train, y_pred)
        
        results[model_name] = {
            'rmse': rmse,
            'mae': mae,
            'r2': r2
        }
        
        print(f"RMSE: {rmse:.4f}")
        print(f"MAE: {mae:.4f}")
        print(f"R²: {r2:.4f}")
    
    # 결과 시각화
    model_names = list(results.keys())
    rmse_values = [results[name]['rmse'] for name in model_names]
    mae_values = [results[name]['mae'] for name in model_names]
    r2_values = [results[name]['r2'] for name in model_names]
    
    # RMSE 시각화
    plt.figure(figsize=(10, 6))
    bars = plt.bar(model_names, rmse_values, color='skyblue')
    plt.xlabel('모델')
    plt.ylabel('RMSE (낮을수록 좋음)')
    plt.title('모델별 RMSE 비교')
    plt.xticks(rotation=45)
    
    # 값 표시
    for bar, val in zip(bars, rmse_values):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1, f'{val:.4f}', 
                ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig(f'{save_path}/models_rmse_comparison.png')
    
    # R² 시각화
    plt.figure(figsize=(10, 6))
    bars = plt.bar(model_names, r2_values, color='lightgreen')
    plt.xlabel('모델')
    plt.ylabel('R² (높을수록 좋음)')
    plt.title('모델별 R² 비교')
    plt.xticks(rotation=45)
    
    # 값 표시
    for bar, val in zip(bars, r2_values):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, f'{val:.4f}', 
                ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig(f'{save_path}/models_r2_comparison.png')
    
    # 결과 저장
    results_df = pd.DataFrame({
        'model': model_names,
        'rmse': rmse_values,
        'mae': mae_values,
        'r2': r2_values
    })
    
    results_df.to_csv(f'{save_path}/models_comparison.csv', index=False)
    print(f"모델 비교 결과가 '{save_path}' 디렉토리에 저장되었습니다.")
    
    # 최적 모델 선택
    best_model = min(results.items(), key=lambda x: x[1]['rmse'])[0]
    print(f"\n최적 모델: {best_model} (RMSE: {results[best_model]['rmse']:.4f})")
    
    return results_df, best_model

if __name__ == "__main__":
    # 테스트 실행
    # 필요한 모듈 로드
    from data_loader import load_data
    from preprocessor import preprocess_data
    from model_trainer import train_model, compare_linear_models
    from sklearn.model_selection import train_test_split
    import os
    
    # 결과 저장 디렉토리 생성
    os.makedirs('output', exist_ok=True)
    
    # 데이터 로드 및 전처리
    train_data, test_data, bus_data, subway_data = load_data()
    X_train, y_train, X_test, processed_data = preprocess_data(train_data, test_data, bus_data, subway_data)
    
    # 훈련/검증 데이터 분할
    X_train_split, X_val, y_train_split, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)
    
    # 여러 모델 훈련
    print("여러 모델 훈련 및 비교 중...")
    model_results, best_model_type = compare_linear_models(X_train_split, y_train_split, tune_hyperparams=False)
    
    # 최적 모델 선택
    best_model, feature_importance = train_model(X_train, y_train, tune_hyperparams=False, model_type=best_model_type)
    
    # 모델 평가
    print(f"\n최적 모델({best_model_type}) 평가:")
    results = evaluate_model(best_model, X_train, y_train, model_type=best_model_type)
    
    # 평가 그래프 생성
    if best_model_type == 'polynomial':
        # 다항 회귀 모델 예측
        num_features = best_model['num_features']
        X_train_subset = X_train[:, :num_features]
        X_train_poly = best_model['poly'].transform(X_train_subset)
        y_pred = best_model['lr'].predict(X_train_poly)
    else:
        # 일반 선형 모델 예측
        y_pred = best_model.predict(X_train)
    
    create_evaluation_plots(y_train, y_pred)
    
    # 계수 분석 (다항 회귀 제외)
    if best_model_type != 'polynomial':
        feature_names = processed_data['X_train'].columns.tolist()
        analyze_coefficients(best_model, feature_names) 