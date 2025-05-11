#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import os
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

import time
import pandas as pd
import numpy as np

from preprocessor.preprocessor_full import load_data, preprocess_data
from model_trainer import train_model, evaluate_model, save_model
from evaluator import print_metrics, create_evaluation_plots

# ✅ 특수문자 제거 + 중복 컬럼 처리 함수
def clean_column_names(df):
    df.columns = df.columns.str.replace('[^A-Za-z0-9_]+', '_', regex=True)
    seen = {}
    new_cols = []
    for col in df.columns:
        if col not in seen:
            seen[col] = 1
            new_cols.append(col)
        else:
            seen[col] += 1
            new_cols.append(f"{col}_{seen[col]-1}")
    df.columns = new_cols
    return df

# ✅ object 타입 컬럼 자동 변환 함수
def ensure_numeric(df):
    bad_cols = df.select_dtypes(include=["object"]).columns.tolist()
    if bad_cols:
        print(f"⚠️ object 타입 컬럼 자동 변환: {bad_cols}")
        for col in bad_cols:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)
    return df

def main():
    print("🏠 부동산 가격 예측 프로젝트 시작 - LightGBM 모델")
    start_time = time.time()

    # ✅ 인코딩 방식 선택: "label", "frequency", "target"
    encoding_method = "label"

    # ✅ 저장 디렉토리 구성
    output_dir = f"output/{encoding_method}"
    os.makedirs(output_dir, exist_ok=True)

    # 1. 데이터 로드
    print("1. 데이터 로드 중...")
    train_path = "data/train.csv"
    test_path = "data/test.csv"
    bus_path = "data/bus_feature.csv"
    subway_path = "data/subway_feature.csv"
    submission_path = "data/sample_submission.csv"

    train_df, test_df, bus_df, subway_df, sample_submission = load_data(
        train_path, test_path, bus_path, subway_path, submission_path
    )

    # 2. 데이터 전처리
    print(f"2. 데이터 전처리 중... (encoding: {encoding_method})")
    train_processed, test_processed, _, _ = preprocess_data(
        train_df, test_df, bus_df, subway_df, encoding=encoding_method
    )

    y_train = train_processed["target"]
    X_train = train_processed.drop(columns=["target"])
    X_test = test_processed

    # ✅ 특수문자 및 object 컬럼 처리
    X_train = clean_column_names(X_train)
    X_test = clean_column_names(X_test)
    X_train = ensure_numeric(X_train)
    X_test = ensure_numeric(X_test)

    print(f"   - 훈련 데이터 크기: {X_train.shape}")
    print(f"   - 테스트 데이터 크기: {X_test.shape}")

    # 3. 모델 훈련
    print("3. 모델 훈련 중...")
    model, feature_importance = train_model(X_train, y_train)

    # 4. 모델 평가
    print("4. 모델 평가 중...")
    rmse, mae, r2 = evaluate_model(model, X_train, y_train)
    print_metrics(rmse, mae, r2)

    # 5. 시각화
    print("5. 평가 시각화 생성 중...")
    y_pred_train = model.predict(X_train)
    create_evaluation_plots(
        y_train,
        y_pred_train,
        save_path=output_dir,
        suffix=f"_{encoding_method}"  # ✅ 인코딩 방식 구분용 suffix
    )

    # 6. 모델 저장
    print("6. 모델 저장 중...")
    save_model(model, output_dir=output_dir, filename=f"model_lgbm_{encoding_method}.pkl")

    # 7. 테스트 데이터 예측
    print("7. 테스트 데이터 예측 중...")
    predictions = model.predict(X_test)
    sample_submission["target"] = np.round(predictions).astype(int)

    # 8. 제출 파일 저장
    print("8. 제출 파일 생성 중...")
    output_path = f"{output_dir}/output_{encoding_method}.csv"
    sample_submission.to_csv(output_path, index=False)
    print(f"   - 제출 파일 저장 완료: {output_path}")

    end_time = time.time()
    print(f"\n✅ 프로젝트 실행 완료! 총 소요 시간: {end_time - start_time:.2f}초")

if __name__ == "__main__":
    main()