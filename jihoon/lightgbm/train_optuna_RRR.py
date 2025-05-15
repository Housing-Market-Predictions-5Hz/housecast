# ✅ Optuna + LightGBM 실험 스크립트 (Top-N feature 기반 성능 검증 포함)

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

# 설정
DATA_DIR = "data"
OUTPUT_DIR = "output/optuna"
os.makedirs(OUTPUT_DIR, exist_ok=True)
COLUMN_MAP_PATH = os.path.join(OUTPUT_DIR, "column_name_map.csv")

TOP_N = 25
N_TRIALS = 10

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
        col_map[final_name] = orig

    df.columns = final_cols
    pd.Series(col_map).to_csv(COLUMN_MAP_PATH)
    print("column_name_map.csv 저장 완료")
    return df

def ensure_numeric(df):
    bad_cols = df.select_dtypes(include=["object"]).columns.tolist()
    if bad_cols:
        print(f"object 타입 컬럼 자동 변환: {bad_cols}")
        for col in bad_cols:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)
    return df

def prepare_data():
    train_df, test_df, bus_df, subway_df, submission = load_data(
        f"{DATA_DIR}/train.csv", f"{DATA_DIR}/test.csv",
        f"{DATA_DIR}/bus_feature.csv", f"{DATA_DIR}/subway_feature.csv",
        f"{DATA_DIR}/sample_submission.csv"
    )
    train_processed, test_processed, _, _ = preprocess_enhanced(
        train_df, test_df, bus_df, subway_df, radius_km=0.4
    )

    train_processed = clean_column_names(train_processed)
    test_processed = clean_column_names(test_processed)

    X = train_processed.drop(columns=["target"])
    y = np.log1p(train_processed["target"])

    X = ensure_numeric(X)
    test_processed = ensure_numeric(test_processed)

    return X, y, test_processed, submission

X, y, X_test, submission = prepare_data()

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
        model = lgb.LGBMRegressor(**param)
        model.fit(
            X.iloc[train_idx], y.iloc[train_idx],
            eval_set=[(X.iloc[valid_idx], y.iloc[valid_idx])],
            eval_metric="rmse",
            callbacks=[early_stopping(50), log_evaluation(10)]
        )
        preds = np.expm1(model.predict(X.iloc[valid_idx]))
        y_valid_exp = np.expm1(y.iloc[valid_idx])
        rmse = np.sqrt(mean_squared_error(y_valid_exp, preds))
        rmse_scores.append(rmse)

    return np.mean(rmse_scores)

def save_best_trial_only(study, trial):
    if study.best_trial.number == trial.number:
        print(f"Best Trial 결과 갱신! Trial #{trial.number}")
        study.trials_dataframe().to_csv(f"{OUTPUT_DIR}/optuna_trials.csv", index=False)

study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=42))
study.optimize(objective, n_trials=N_TRIALS, callbacks=[save_best_trial_only])

best_params = study.best_trial.params
print("\nBest RMSE:", study.best_value)
print("Best Parameters:", best_params)

# 전체 데이터로 학습하여 feature importance 추출
model = lgb.LGBMRegressor(**best_params, n_estimators=1000)
model.fit(X, y)

feature_importance = pd.DataFrame({
    "feature": X.columns,
    "importance": model.feature_importances_
}).sort_values(by="importance", ascending=False)

total_importance = feature_importance["importance"].sum()
feature_importance["importance_pct"] = (
    feature_importance["importance"] / total_importance * 100
).round(2)

feature_importance.to_csv(f"{OUTPUT_DIR}/feature_importance.csv", index=False)
print("feature_importance.csv 저장 완료")

# Top-N feature 선택 및 매핑 출력
selected_features = feature_importance["feature"].head(TOP_N).tolist()
col_map = pd.read_csv(COLUMN_MAP_PATH, index_col=0).to_dict().get("0", {})

print(f"\n[DEBUG] Top {TOP_N} selected features with importance (%):")
for i, row in feature_importance.head(TOP_N).iterrows():
    clean = row["feature"]
    orig = col_map.get(clean, "(매핑없음)")
    imp = row["importance"]
    pct = row["importance_pct"]
    print(f"{i+1:>2}. {clean:25} → {orig:25} | 중요도: {imp:5} ({pct:.2f}%)")

X_selected = X[selected_features]
X_test_selected = X_test[selected_features]

# TopN 기반 성능 재평가
kf = KFold(n_splits=5, shuffle=True, random_state=42)
rmse_scores = []
for train_idx, valid_idx in kf.split(X_selected):
    model = lgb.LGBMRegressor(**best_params, n_estimators=1000)
    model.fit(X_selected.iloc[train_idx], y.iloc[train_idx])
    preds = np.expm1(model.predict(X_selected.iloc[valid_idx]))
    y_valid_exp = np.expm1(y.iloc[valid_idx])
    rmse = np.sqrt(mean_squared_error(y_valid_exp, preds))
    rmse_scores.append(rmse)

print(f"\nTop {TOP_N} 모델 평균 RMSE: {np.mean(rmse_scores):.4f}")

# 최종 학습 및 제출 생성
model = lgb.LGBMRegressor(**best_params, n_estimators=1000)
model.fit(X_selected, y)
joblib.dump(model, f"{OUTPUT_DIR}/model_lgbm_optuna_top{TOP_N}.pkl")

preds = np.expm1(model.predict(X_test_selected))
if len(preds) != len(submission):
    raise ValueError(f"예측 개수 불일치: preds={len(preds)}, submission={len(submission)}")

submission = submission.copy()
submission["target"] = np.round(preds).astype(int)
submission.to_csv(f"{OUTPUT_DIR}/output_optuna_top{TOP_N}.csv", index=False)
print(f"최종 제출 파일 저장 완료: output_optuna_top{TOP_N}.csv")