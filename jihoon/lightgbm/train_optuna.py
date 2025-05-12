import optuna
import pandas as pd
import numpy as np
import lightgbm as lgb
from lightgbm import early_stopping, log_evaluation
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import joblib
import os

from preprocessor.preprocessor_enhanced import load_data, preprocess_data

# ✅ 컬럼명 정제 함수 (중복 방지 포함)
def clean_column_names(df):
    df.columns = df.columns.str.replace(r'[^A-Za-z0-9_]+', '_', regex=True)
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

# ✅ object 타입 컬럼 numeric으로 자동 변환
def ensure_numeric(df):
    bad_cols = df.select_dtypes(include=["object"]).columns.tolist()
    if bad_cols:
        print(f"⚠️ object 타입 컬럼 자동 변환: {bad_cols}")
        for col in bad_cols:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)
    return df

# 경로 설정
data_dir = "data"
output_dir = "output/optuna"
os.makedirs(output_dir, exist_ok=True)

train_path = f"{data_dir}/train.csv"
test_path = f"{data_dir}/test.csv"
bus_path = f"{data_dir}/bus_feature.csv"
subway_path = f"{data_dir}/subway_feature.csv"
submission_path = f"{data_dir}/sample_submission.csv"

# 데이터 로드 및 전처리
def prepare_data():
    train_df, test_df, bus_df, subway_df, submission = load_data(
        train_path, test_path, bus_path, subway_path, submission_path
    )
    train_processed, test_processed, _, _ = preprocess_data(train_df, test_df, bus_df, subway_df)

    train_processed = clean_column_names(train_processed)
    test_processed = clean_column_names(test_processed)

    X = train_processed.drop(columns=["target"])
    y = train_processed["target"]

    X = ensure_numeric(X)
    test_processed = ensure_numeric(test_processed)

    return X, y, test_processed, submission

X, y, X_test, submission = prepare_data()

# Optuna 목적 함수 정의
def objective(trial):
    X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=42)

    param = {
        'objective': 'regression',
        'metric': 'rmse',
        'verbosity': -1,
        'boosting_type': 'gbdt',
        'n_estimators': 1000,
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
        'num_leaves': trial.suggest_int('num_leaves', 20, 300),
        'max_depth': trial.suggest_int('max_depth', 3, 15),
        'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0)
    }

    model = lgb.LGBMRegressor(**param)
    model.fit(
        X_train,
        y_train,
        eval_set=[(X_valid, y_valid)],
        eval_metric="rmse",
        callbacks=[early_stopping(50), log_evaluation(0)]
    )

    preds = model.predict(X_valid)
    rmse = mean_squared_error(y_valid, preds, squared=False)
    return rmse

# 튜닝 실행
study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=200)

# 최적 파라미터 및 모델 저장
best_params = study.best_trial.params
print("\n✅ Best RMSE:", study.best_value)
print("✅ Best Parameters:", best_params)

model = lgb.LGBMRegressor(**best_params, n_estimators=1000)
model.fit(X, y)

joblib.dump(model, f"{output_dir}/model_lgbm_optuna.pkl")

# 테스트 예측 및 저장
preds = model.predict(X_test)
submission["target"] = np.round(preds).astype(int)
submission.to_csv(f"{output_dir}/output_optuna.csv", index=False)
print(f"\n✅ 제출 파일 저장 완료: {output_dir}/output_optuna.csv")
