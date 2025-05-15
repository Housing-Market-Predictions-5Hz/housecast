# 개선된 Optuna + LightGBM 실험 스크립트 (Top-N feature 기반 성능 검증 포함)

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

# 설정 경로 및 파라미터 정의
DATA_DIR = "data"
OUTPUT_DIR = "output/optuna"
os.makedirs(OUTPUT_DIR, exist_ok=True)
COLUMN_MAP_PATH = os.path.join(OUTPUT_DIR, "column_name_map.csv")

TOP_N = 25  # 사용할 상위 feature 개수
N_TRIALS = 50  # Optuna 탐색 횟수

# 컬럼명을 정제하고 원래 이름과 매핑 저장
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

# object 타입 컬럼들을 숫자형으로 변환
def ensure_numeric(df):
    bad_cols = df.select_dtypes(include=["object"]).columns.tolist()
    if bad_cols:
        print(f"object 타입 컬럼 자동 변환: {bad_cols}")
        for col in bad_cols:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)
    return df

# 데이터 로딩 및 전처리 수행

def prepare_data():
    train_df, test_df, bus_df, subway_df, submission = load_data(
        f"{DATA_DIR}/train.csv", f"{DATA_DIR}/test.csv",
        f"{DATA_DIR}/bus_feature.csv", f"{DATA_DIR}/subway_feature.csv",
        f"{DATA_DIR}/sample_submission.csv"
    )
    train_processed, test_processed, _, _ = preprocess_enhanced(
        train_df, test_df, bus_df, subway_df, radius_km=1.0
    )

    train_processed = clean_column_names(train_processed)
    test_processed = clean_column_names(test_processed)

    X = train_processed.drop(columns=["target"])
    y = np.log1p(train_processed["target"])

    X = ensure_numeric(X)
    test_processed = ensure_numeric(test_processed)

    return X, y, test_processed, submission

# 전처리된 데이터 준비
X, y, X_test, submission = prepare_data()

# Optuna 목적 함수 정의 (LightGBM RMSE 기준)
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

# Optuna trial 결과 저장 함수
def save_best_trial_only(study, trial):
    if study.best_trial.number == trial.number:
        print(f"Best Trial 갱신됨! Trial #{trial.number}")
        study.trials_dataframe().to_csv(f"{OUTPUT_DIR}/optuna_trials.csv", index=False)

# Optuna를 통한 하이퍼파라미터 탐색
study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=42))
study.optimize(objective, n_trials=N_TRIALS, callbacks=[save_best_trial_only])

# 최적 파라미터 및 성능 출력
best_params = study.best_trial.params
print("Best RMSE:", study.best_value)
print("Best Parameters:", best_params)

# 전체 데이터로 최종 모델 학습 및 변수 중요도 계산
model = lgb.LGBMRegressor(**best_params, n_estimators=1000)
model.fit(X, y)
feature_importance = pd.DataFrame({
    "feature": X.columns,
    "importance": model.feature_importances_
}).sort_values(by="importance", ascending=False)

# 중요도 백분율 계산
feature_importance["importance_pct"] = (
    feature_importance["importance"] / feature_importance["importance"].sum() * 100
).round(2)

feature_importance.to_csv(f"{OUTPUT_DIR}/feature_importance.csv", index=False)
print("feature_importance.csv 저장 완료")

# Top-N 변수 선택 및 원본 매핑 출력
selected_features = feature_importance["feature"].head(TOP_N).tolist()
col_map = pd.read_csv(COLUMN_MAP_PATH, index_col=0).to_dict().get("0", {})

print(f"\n[DEBUG] Top {TOP_N} selected features with importance (%):")
for i, row in feature_importance.head(TOP_N).iterrows():
    clean = row["feature"]
    orig = col_map.get(clean, "(매핑없음)")
    imp = row["importance"]
    pct = row["importance_pct"]
    print(f"{i+1:>2}. {clean:25} → {orig:25} | 중요도: {imp:5} ({pct:.2f}%)")

# 선택된 변수 기반으로 재학습 및 KFold 평가
X_selected = X[selected_features]
X_test_selected = X_test[selected_features]

kf = KFold(n_splits=5, shuffle=True, random_state=42)
rmse_scores = []
for train_idx, valid_idx in kf.split(X_selected):
    model = lgb.LGBMRegressor(**best_params, n_estimators=1000)
    model.fit(X_selected.iloc[train_idx], y.iloc[train_idx])
    preds = np.expm1(model.predict(X_selected.iloc[valid_idx]))
    y_valid_exp = np.expm1(y.iloc[valid_idx])
    rmse = np.sqrt(mean_squared_error(y_valid_exp, preds))
    rmse_scores.append(rmse)

print(f"\n재학습된 Top {TOP_N} 모델 평균 RMSE: {np.mean(rmse_scores):.4f}")

# 최종 모델 저장
model = lgb.LGBMRegressor(**best_params, n_estimators=1000)
model.fit(X_selected, y)
joblib.dump(model, f"{OUTPUT_DIR}/model_lgbm_optuna_top{TOP_N}.pkl")

# 예측 결과 생성 및 제출 파일 저장
preds = np.expm1(model.predict(X_test_selected))
if len(preds) != len(submission):
    raise ValueError(f"예측 개수 불일치: preds={len(preds)}, submission={len(submission)}")

submission = submission.copy()
submission["target"] = np.round(preds).astype(int)
submission.to_csv(f"{OUTPUT_DIR}/output_optuna_top{TOP_N}.csv", index=False)
print(f"Top {TOP_N} 변수 기반 제출 파일 저장 완료: output_optuna_top{TOP_N}.csv")
