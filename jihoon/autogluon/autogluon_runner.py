import os
os.environ["RAY_USE_MULTIPROCESSING_CPU_COUNT"] = "1"

import pandas as pd
import numpy as np
from autogluon.tabular import TabularPredictor

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

# ✅ object 타입 컬럼 자동 변환 함수 (AutoGluon-friendly)
def ensure_numeric(df):
    bad_cols = df.select_dtypes(include=["object"]).columns.tolist()
    if bad_cols:
        print(f"⚠️ object 타입 컬럼 자동 변환: {bad_cols}")
        for col in bad_cols:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)
    return df

# 경로 설정
data_dir = "data"
output_dir = "output/autogluon"
os.makedirs(output_dir, exist_ok=True)

train_path = f"{data_dir}/train.csv"
test_path = f"{data_dir}/test.csv"
bus_path = f"{data_dir}/bus_feature.csv"
subway_path = f"{data_dir}/subway_feature.csv"
submission_path = f"{data_dir}/sample_submission.csv"

print("🏠 AutoGluon 기반 모델 학습 및 예측 시작")

# 1. 데이터 로딩 및 전처리
train_df, test_df, bus_df, subway_df, submission = load_data(
    train_path, test_path, bus_path, subway_path, submission_path
)
train_processed, test_processed, _, _ = preprocess_data(train_df, test_df, bus_df, subway_df)

train_processed = clean_column_names(train_processed)
test_processed = clean_column_names(test_processed)

print("✅ 데이터 전처리 완료")

# 2. 학습
predictor = TabularPredictor(label="target", path=output_dir, eval_metric="rmse")
predictor.fit(train_data=train_processed, presets="best_quality", time_limit=7200)

# 3. 예측 및 저장
preds = predictor.predict(test_processed)
submission["target"] = preds.values.round().astype(int)
submission.to_csv(f"{output_dir}/output_autogluon.csv", index=False)
print(f"✅ 예측 완료 및 저장: {output_dir}/output_autogluon.csv")