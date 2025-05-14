import optuna
import pandas as pd
import numpy as np
import lightgbm as lgb
from lightgbm import early_stopping, log_evaluation
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
import joblib
import os

from preprocessor.preprocessor_enhanced import load_data, preprocess_enhanced

def clean_column_names(df):
    original_cols = df.columns.tolist()
    new_cols = df.columns.str.replace(r'[^A-Za-z0-9_]+', '_', regex=True)
    seen = {}
    final_cols = []
    col_map = {}

    for orig, clean in zip(original_cols, new_cols):
        if clean not in seen:
            seen[clean] = 1
            final_name = clean
        else:
            seen[clean] += 1
            final_name = f"{clean}_{seen[clean] - 1}"
        final_cols.append(final_name)
        col_map[final_name] = orig  # ✅ 원래 이름 저장

    df.columns = final_cols

    # ✅ 매핑 CSV 저장
    pd.Series(col_map).to_csv("output/optuna/column_name_map.csv")
    print("📁 column_name_map.csv 저장 완료")

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
    train_processed, test_processed, _, _ = preprocess_enhanced(
        train_df, test_df, bus_df, subway_df, radius_km=1.0
    )

    train_processed = clean_column_names(train_processed)
    test_processed = clean_column_names(test_processed)

    X = train_processed.drop(columns=["target"])
    y = np.log1p(train_processed["target"])  # ✅ 로그 변환

    X = ensure_numeric(X)
    test_processed = ensure_numeric(test_processed)

    return X, y, test_processed, submission

X, y, X_test, submission = prepare_data()

# ✅ Optuna 목적 함수 정의 (KFold 기반)
def objective(trial):
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    rmse_scores = []

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

    for train_idx, valid_idx in kf.split(X):
        X_train, X_valid = X.iloc[train_idx], X.iloc[valid_idx]
        y_train, y_valid = y.iloc[train_idx], y.iloc[valid_idx]

        model = lgb.LGBMRegressor(**param)
        model.fit(
            X_train,
            y_train,
            eval_set=[(X_valid, y_valid)],
            eval_metric="rmse",
            callbacks=[early_stopping(50), log_evaluation(0)]
        )

        preds = model.predict(X_valid)
        preds = np.expm1(preds)  # ✅ 예측 복원
        y_valid_exp = np.expm1(y_valid)
        rmse = np.sqrt(mean_squared_error(y_valid_exp, preds))
        rmse_scores.append(rmse)

    return np.mean(rmse_scores)

# ✅ Optuna 튜닝 실행
study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=20)  # ✅ trial 횟수 20으로 축소

# ✅ 최적 파라미터 및 모델 저장
best_params = study.best_trial.params
print("\n✅ Best RMSE:", study.best_value)
print("✅ Best Parameters:", best_params)

model = lgb.LGBMRegressor(**best_params, n_estimators=1000)
model.fit(X, y)

# ✅ STEP 1: Feature Importance 저장
feature_importance = pd.DataFrame({
    "feature": X.columns,
    "importance": model.feature_importances_
}).sort_values(by="importance", ascending=False)

feature_importance.to_csv(f"{output_dir}/feature_importance.csv", index=False)
print("\n📊 feature_importance.csv 저장 완료")

# ✅ STEP 2: 상위 N개 변수 기반 재학습
TOP_N = 20
selected_features = feature_importance["feature"].head(TOP_N).tolist()
X_selected = X[selected_features]
X_test_selected = X_test[selected_features]

model = lgb.LGBMRegressor(**best_params, n_estimators=1000)
model.fit(X_selected, y)

joblib.dump(model, f"{output_dir}/model_lgbm_optuna_top{TOP_N}.pkl")

preds = model.predict(X_test_selected)
preds = np.expm1(preds)
submission["target"] = np.round(preds).astype(int)
submission.to_csv(f"{output_dir}/output_optuna_top{TOP_N}.csv", index=False)
print(f"\n✅ Top {TOP_N} 변수 기반 제출 파일 저장 완료: output_optuna_top{TOP_N}.csv")