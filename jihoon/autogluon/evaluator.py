#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
부동산 가격 예측 프로젝트 - LightGBM 모델
모델 평가 모듈

이 모듈은 훈련된 모델을 평가하고 성능 지표를 계산하고, 시각화를 제공합니다.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.font_manager as fm
import matplotlib as mpl
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# ✅ 한글 폰트 설정 + 유니코드 minus sign 대응
try:
    font_path = "/usr/share/fonts/truetype/nanum/NanumGothic.ttf"
    fontprop = fm.FontProperties(fname=font_path)
    plt.rcParams['font.family'] = [fontprop.get_name(), 'DejaVu Sans']
except Exception as e:
    print(f"⚠️ 한글 폰트 설정 실패: {e}")

mpl.rcParams['axes.unicode_minus'] = False  # ✅ minus sign 경고 제거

def evaluate_model(model, X_train, y_train, X_val=None, y_val=None):
    results = {}

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

    return results

def create_evaluation_plots(y_true, y_pred, save_path='output', suffix=''):
    try:
        # 실제 vs 예측
        plt.figure(figsize=(10, 6))
        plt.scatter(y_true, y_pred, alpha=0.5)
        plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--')
        plt.xlabel('실제 가격')
        plt.ylabel('예측 가격')
        plt.title('실제 가격 vs 예측 가격')
        plt.tight_layout()
        plt.savefig(f'{save_path}/actual_vs_predicted{suffix}.png')

        # 잔차 분포
        residuals = y_true - y_pred
        plt.figure(figsize=(10, 6))
        sns.histplot(residuals, kde=True)
        plt.xlabel('잔차')
        plt.ylabel('빈도')
        plt.title('잔차 분포')
        plt.axvline(x=0, color='r', linestyle='--')
        plt.tight_layout()
        plt.savefig(f'{save_path}/residuals_histogram{suffix}.png')

        # 예측 vs 잔차
        plt.figure(figsize=(10, 6))
        plt.scatter(y_pred, residuals, alpha=0.5)
        plt.axhline(y=0, color='r', linestyle='--')
        plt.xlabel('예측 가격')
        plt.ylabel('잔차')
        plt.title('예측 가격 vs 잔차')
        plt.tight_layout()
        plt.savefig(f'{save_path}/predicted_vs_residuals{suffix}.png')

        print(f"✅ 평가 그래프가 '{save_path}' 디렉토리에 저장되었습니다.")
    except Exception as e:
        print(f"⚠️ 그래프 생성 중 오류 발생: {e}")

def print_metrics(rmse, mae, r2):
    print("훈련 데이터 성능 평가:")
    print(f"RMSE: {rmse:.4f}")
    print(f"MAE : {mae:.4f}")
    print(f"R²  : {r2:.4f}")