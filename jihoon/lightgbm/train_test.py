# ✅ proximity feature 실험 스크립트 (Optuna 제거)
# Top40 기반 + proximity feature 100~1000m 순차적 추가 테스트

import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.metrics import mean_squared_error
import joblib
import os
import matplotlib.pyplot as plt
import seaborn as sns

from preprocessor.preprocessor_enhanced import load_data, preprocess_enhanced

# 경로 설정
DATA_DIR = "data"
OUTPUT_DIR = "output/proximity_test"
os.makedirs(OUTPUT_DIR, exist_ok=True)

train_path = f"{DATA_DIR}/train.csv"
test_path = f"{DATA_DIR}/test.csv"
bus_path = f"{DATA_DIR}/bus_feature.csv"
subway_path = f"{DATA_DIR}/subway_feature.csv"
submission_path = f"{DATA_DIR}/sample_submission.csv"

# 베스트 파라미터 수동 입력
best_params = {
    'learning_rate': 0.11861663446573512,
    'num_leaves': 287,
    'max_depth': 12,
    'min_child_samples': 62,
    'subsample': 0.5780093202212182,
    'colsample_bytree': 0.5779972601681014
}

# feature_importance.csv 상위 40개 불러오기
importance_df = pd.read_csv("output/optuna/feature_importance.csv")
base_top_features = importance_df["feature"].head(40).tolist()

# 전처리
train_df, test_df, bus_df, subway_df, submission = load_data(train_path, test_path, bus_path, subway_path, submission_path)
train_processed, test_processed, _, _ = preprocess_enhanced(train_df, test_df, bus_df, subway_df, radius_km=1.0)

X = train_processed.drop(columns=["target"])
y = np.log1p(train_processed["target"])
X_test = test_processed.copy()

# object 제거
X = X.apply(pd.to_numeric, errors="coerce").fillna(0)
X_test = X_test.apply(pd.to_numeric, errors="coerce").fillna(0)

# proximity 실험 대상 리스트
PROXIMITY_FEATURES = [
    "대장_근접여부_100m",
    "대장_근접여부_200m",
    "대장_근접여부_300m",
    "대장_근접여부_500m",
    "대장_근접여부_1000m"
]

# 결과 기록용 리스트
results = []

for prox_col in PROXIMITY_FEATURES:
    print(f"\n🚩 실험: {prox_col}")
    features = base_top_features + [prox_col]

    # 존재하지 않는 컬럼 제거 (예: TE로 인해 이름이 바뀌거나 drop된 경우)
    missing_features = [f for f in features if f not in X.columns]
    if missing_features:
        print(f"⚠️ 누락된 feature 제거: {missing_features}")
    features = [f for f in features if f in X.columns]

    X_selected = X[features]
    X_test_selected = X_test[features]

    model = lgb.LGBMRegressor(**best_params, n_estimators=1000)
    model.fit(X_selected, y)

    preds_train = model.predict(X_selected)
    rmse = np.sqrt(mean_squared_error(np.expm1(y), np.expm1(preds_train)))
    print(f"✅ RMSE: {rmse:.4f}")

    preds_test = np.expm1(model.predict(X_test_selected))
    submission_copy = submission.copy()
    submission_copy["target"] = np.round(preds_test).astype(int)
    out_path = f"{OUTPUT_DIR}/output_top40_{prox_col}.csv"
    submission_copy.to_csv(out_path, index=False)
    print(f"📁 저장 완료: {out_path}")

    results.append({"proximity": prox_col, "rmse": rmse})

# 결과 정리
results_df = pd.DataFrame(results)
results_df.to_csv(f"{OUTPUT_DIR}/proximity_test_summary.csv", index=False)
print("\n📊 모든 실험 완료 → proximity_test_summary.csv 저장됨")