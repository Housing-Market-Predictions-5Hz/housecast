import sys
import os
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

import time
import pandas as pd
import numpy as np
import joblib

from preprocessor.preprocessor_enhanced import load_data, preprocess_enhanced
from evaluator import print_metrics, create_evaluation_plots
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from lightgbm import LGBMRegressor

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

def ensure_numeric(df):
    bad_cols = df.select_dtypes(include=["object"]).columns.tolist()
    if bad_cols:
        print(f"⚠️ object 타입 컬럼 자동 변환: {bad_cols}")
        for col in bad_cols:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)
    return df

def main():
    print("\U0001f3e0 부동산 가격 예측 프로젝트 시작 - LightGBM + K-Fold")
    start_time = time.time()

    encoding_method = "optuna"
    output_dir = f"output/{encoding_method}"
    os.makedirs(output_dir, exist_ok=True)

    # 1. 데이터 로딩
    train_path = "data/train.csv"
    test_path = "data/test.csv"
    bus_path = "data/bus_feature.csv"
    subway_path = "data/subway_feature.csv"
    submission_path = "data/sample_submission.csv"

    train_df, test_df, bus_df, subway_df, sample_submission = load_data(
        train_path, test_path, bus_path, subway_path, submission_path
    )

    print("2. 데이터 전처리 중...")
    train_processed, test_processed, _, _ = preprocess_enhanced(
        train_df, test_df, bus_df, subway_df, radius_km=1.0
    )

    y = train_processed["target"].values
    X = train_processed.drop(columns=["target"])
    X_test = test_processed

    X = clean_column_names(X)
    X_test = clean_column_names(X_test)
    X = ensure_numeric(X)
    X_test = ensure_numeric(X_test)

    print(f"   - 훈련 데이터 크기: {X.shape}")
    print(f"   - 테스트 데이터 크기: {X_test.shape}")

    # 3. K-Fold 교차검정
    print("3. K-Fold 교차검정 시작")
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    scores = []

    for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
        print(f"\n--- Fold {fold + 1} ---")
        X_train_fold, X_val_fold = X.iloc[train_idx], X.iloc[val_idx]
        y_train_fold, y_val_fold = y[train_idx], y[val_idx]

        model = LGBMRegressor(
            learning_rate=0.11256033454760318,
            num_leaves=293,
            max_depth=13,
            min_child_samples=10,
            subsample=0.9307725175324212,
            colsample_bytree=0.8501308663155136,
            n_estimators=1000,
            objective='regression',
            random_state=42
        )
        model.fit(X_train_fold, y_train_fold)
        y_pred_val = model.predict(X_val_fold)

        rmse = np.sqrt(mean_squared_error(y_val_fold, y_pred_val))
        mae = mean_absolute_error(y_val_fold, y_pred_val)
        r2 = r2_score(y_val_fold, y_pred_val)
        print_metrics(rmse, mae, r2)
        scores.append((rmse, mae, r2))

    avg_rmse = np.mean([s[0] for s in scores])
    avg_mae = np.mean([s[1] for s in scores])
    avg_r2 = np.mean([s[2] for s in scores])
    print("\n✅ 평균 성능 (K-Fold):")
    print_metrics(avg_rmse, avg_mae, avg_r2)

    # 4. 전체 학습 데이터로 최종 모델 학습
    print("\n4. 전체 데이터로 최종 모델 학습 중...")
    final_model = LGBMRegressor(
        learning_rate=0.11256033454760318,
        num_leaves=293,
        max_depth=13,
        min_child_samples=10,
        subsample=0.9307725175324212,
        colsample_bytree=0.8501308663155136,
        n_estimators=1000,
        objective='regression',
        random_state=42
    )
    final_model.fit(X, y)
    y_pred_train = final_model.predict(X)
    print("최종 모델 학습 완료!")

    print("5. 전체 학습 데이터 평가 및 시각화")
    rmse = np.sqrt(mean_squared_error(y, y_pred_train))
    mae = mean_absolute_error(y, y_pred_train)
    r2 = r2_score(y, y_pred_train)
    print_metrics(rmse, mae, r2)

    create_evaluation_plots(
        y, y_pred_train,
        save_path=output_dir,
        suffix=f"_{encoding_method}"
    )

    print("6. 모델 저장 및 테스트 예측")
    joblib.dump(final_model, f"{output_dir}/model_lgbm_{encoding_method}.pkl")
    predictions = final_model.predict(X_test)
    sample_submission["target"] = np.round(predictions).astype(int)
    sample_submission.to_csv(f"{output_dir}/output_{encoding_method}.csv", index=False)

    end_time = time.time()
    print(f"\n✅ 전체 파이프라인 완료! 총 소요 시간: {end_time - start_time:.2f}초")

if __name__ == "__main__":
    main()
