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
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

def train_model(X_train, y_train, tune_hyperparams=False, simple_experiment=False):
    """
    Random Forest 모델을 훈련하는 함수
    
    Args:
        X_train (np.ndarray): 훈련 데이터의 특성 행렬
        y_train (np.ndarray): 훈련 데이터의 타겟 값
        tune_hyperparams (bool): 하이퍼파라미터 튜닝 여부
        simple_experiment (bool): 간단한 트리 수 실험 여부
        
    Returns:
        tuple: (trained_model, feature_importance) 또는 실험 결과
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
    
    if simple_experiment:
        print("다양한 트리 수에 대한 간단한 실험 시작...")
        from sklearn.model_selection import cross_val_score
        import matplotlib.pyplot as plt
        
        # 실험할 트리 수 범위
        n_estimators_list = [5, 10, 15, 20, 50, 100]
        cv_scores = []
        
        for n_est in n_estimators_list:
            print(f"n_estimators={n_est} 테스트 중...")
            model = RandomForestRegressor(
                n_estimators=n_est,
                criterion='squared_error',
                min_samples_split=10,
                min_samples_leaf=2,
                max_features='sqrt',
                random_state=1,
                n_jobs=-1
            )
            scores = cross_val_score(model, X_train, y_train, cv=3, scoring='r2')
            mean_score = scores.mean()
            cv_scores.append(mean_score)
            print(f"  평균 CV 점수: {mean_score:.4f}")
        
        # 트리 수에 따른 성능 시각화
        plt.figure(figsize=(10, 6))
        plt.plot(n_estimators_list, cv_scores, marker='o', linestyle='-')
        plt.xlabel('트리 수 (n_estimators)')
        plt.ylabel('교차 검증 R2 점수')
        plt.title('트리 수에 따른 랜덤 포레스트 성능')
        plt.grid(True)
        plt.savefig('output/n_estimators_experiment.png')
        print(f"실험 결과 그래프가 output/n_estimators_experiment.png에 저장되었습니다.")
        
        # 최적의 트리 수 선택
        best_idx = cv_scores.index(max(cv_scores))
        best_n_estimators = n_estimators_list[best_idx]
        print(f"\n최적의 트리 수: {best_n_estimators} (CV 점수: {max(cv_scores):.4f})")
        
        # 최적의 트리 수로 최종 모델 훈련
        model = RandomForestRegressor(
            n_estimators=best_n_estimators,
            criterion='squared_error',
            min_samples_split=10,
            min_samples_leaf=2,
            max_features='sqrt',
            random_state=1,
            n_jobs=-1
        )
        model.fit(X_train, y_train)
        
        return model, pd.DataFrame({
            'n_estimators': n_estimators_list,
            'cv_score': cv_scores
        })
    
    elif tune_hyperparams:
        print("하이퍼파라미터 튜닝 시작...")
        
        # RandomizedSearchCV를 이용한 하이퍼파라미터 튜닝 - 메모리 사용 최적화
        param_grid = {
            'n_estimators': [100, 200, 300],
            'max_depth': [None, 10, 20, 30],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'max_features': ['sqrt', 'log2']
        }
        
        rf = RandomForestRegressor(random_state=42)
        random_search = RandomizedSearchCV(
            estimator=rf,
            param_distributions=param_grid,
            n_iter=10,
            cv=2,
            verbose=1,
            random_state=42,
            n_jobs=2
        )
        
        random_search.fit(X_train, y_train)
        model = random_search.best_estimator_
        
        print(f"최적 하이퍼파라미터: {random_search.best_params_}")
        print(f"최적 CV 점수: {random_search.best_score_:.4f}")
    else:
        # 기본 하이퍼파라미터로 모델 훈련
        model = RandomForestRegressor(
            n_estimators=5,
            criterion='squared_error',
            max_depth=None,
            min_samples_split=2,
            min_samples_leaf=1,
            max_features='sqrt',
            n_jobs=-1,
            random_state=1
        )
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
    
    # 모델 훈련 테스트 - 간단한 모델 사용
    model, feature_importance = train_model(X_train, y_train, tune_hyperparams=False)
    
    # 특성 중요도 확인
    print("\n특성 중요도:")
    print(feature_importance.head(10))
    
    # 특성 중요도 시각화 저장
    save_feature_importance_plot(feature_importance) 